"""``si.crosstalk`` (:func:`ai_eda.validation.si.crosstalk_result`), its neighbour search and the tool-backed ``domain.high_speed.si``.

Every board is a synthetic one (the fixture library of ``tests/test_routing.py``
and the boards of ``tests/test_si_checks.py``): ``LONG`` (R1.1 an output ->
R2.1) and ``SHORT`` run side by side, ``GND`` is a ground net. What is proved:

* a complete parallel run of equal width over a plane gets the closed-form
  numbers of :mod:`ai_eda.tools.calc.xtalk` and still a NOT_VERIFIED row:
  the victim ends in R_drive and C_load, not Z0 - the check never PASSes or
  FAILs anything;
* every neighbour no opened formula covers is named, never estimated (an
  inner layer, no reference plane, unequal widths, a gap the coupled-line
  formula refuses, a missing aggressor edge, a plane a keep-out clears,
  broadside, non-parallel, touching copper, another net's zone);
* no neighbour within 10 h is NOT_APPLICABLE with the count beyond the window;
* the other ``si.*`` results do not change, the result keeps its own tool and
  version on a typed-via board, the output is deterministic, the IR is not
  mutated, and ``domain.high_speed.si`` copies the rows under its own tool.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import pytest

from ai_eda.design.stackup import board_layers, escalation_stackup
from ai_eda.ir import CircuitIR, Keepout, Provenance, ProvenanceKind, Zone, user_requirement
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc.tline import NO_REFERENCE_PLANE, edge_coupled_microstrip, propagation_delay, solve_microstrip_width
from ai_eda.tools.calc.xtalk import fext_rel, kb_eq8, l_sat_mm, next_rel
from ai_eda.tools.routing.coupling import NeighbourIndex, coupled_pieces, parallel_runs
from ai_eda.tools.si.measure import NOT_DRC, measure_nets
from ai_eda.validation import ValidationContext
from ai_eda.validation.si import XT_NO_ASYMMETRIC, XT_NO_BROADSIDE, XT_NON_PARALLEL, crosstalk_result, si_results
from tests.test_si_checks import cls, default_classes, driver, long_board, si_of, track
from tests.test_routing import fixture_library

#: the 50-ohm width on the 4-layer generic F.Cu (h 0.2 mm, t 35 um, er 4.5), and an edge gap of 1.5 h
W = round(solve_microstrip_width(50.0, 0.2, 35.0, 4.5), 4)
GAP = 0.3
#: a complete row whose decks cannot be built (si.crosstalk 0.2): here no KiCad library, so no pad-to-pad path
TERMINATION = "the matched-termination closed forms do not bound a victim driven through 25 ohm into 5e-12 F; no coupled-line deck: "
NO_PATH = TERMINATION + "victim LONG has no extracted pad-to-pad path (no KiCad library: the pad positions are not known)"


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def pair_board(tmp_path: Path, lib, layers: int | None = 4, gap: float = GAP, *, width: float = W, aggressor_width: float | None = None,
               layer: str = "F.Cu") -> CircuitIR:
    """LONG along y = 5 from x 5 to 95; SHORT parallel to it from x 20 to 40 at edge gap ``gap`` (a 20 mm run)."""
    ir = long_board(tmp_path, lib, layers)
    ir.si = si_of(*default_classes())
    wa = width if aggressor_width is None else aggressor_width
    y = 5.0 + width / 2.0 + gap + wa / 2.0
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0), width=width, layer=layer), track("SHORT", (20.0, y), (40.0, y), width=wa, layer=layer)]
    return ir


def xt(ir: CircuitIR):
    return crosstalk_result(ir, ir.si)


def rows(result) -> dict[str, dict]:
    return {r["net"]: r for r in result.details["rows"]}


def test_a_complete_parallel_run_gets_the_closed_forms_and_still_no_verdict(tmp_path: Path, lib):
    r = xt(pair_board(tmp_path, lib))
    assert r.check_id == "si.crosstalk" and r.tool == "si.crosstalk" and r.tool_version == "0.2" and r.status is S.NOT_VERIFIED
    assert r.message.endswith(f"({NOT_DRC})") and r.details["kind"] == "ir_geometry+calculators" and r.is_tool_backed
    by = rows(r)
    long, short, gnd = by["LONG"], by["SHORT"], by["GND"]
    assert long["status"] == S.NOT_VERIFIED.value and long["reason"] == NO_PATH and long["judged_by"] is None and long["not_computed"] == []
    assert gnd["status"] == S.NOT_APPLICABLE.value and gnd["reason"] == "a ground net carries no driven edge: neither victim nor aggressor"
    (a,) = long["aggressors"]
    c = edge_coupled_microstrip(W, a["gap_mm"], 0.2, 35.0, 4.5)
    kb = kb_eq8(c.z_even, c.z_odd, math.sqrt(c.z_even * c.z_odd))
    ls = l_sat_mm(1e-9, propagation_delay(c.e_eff_even))
    assert a["net"] == "SHORT" and a["layer"] == "F.Cu" and a["length_mm"] == pytest.approx(20.0) and a["width_mm"] == W
    assert a["gap_mm"] == pytest.approx(GAP, abs=1e-9) and a["s_over_h"] == pytest.approx(1.5, abs=1e-6)
    assert a["kb"] == pytest.approx(0.04534, rel=1e-3) and a["kb"] == pytest.approx(kb, rel=1e-5) and a["l_sat_mm"] == pytest.approx(ls, abs=1e-6)
    assert a["next_rel"] == pytest.approx(next_rel(kb, 20.0, ls), rel=1e-5) == pytest.approx(0.0115, rel=2e-3)
    assert a["fext_rel"] == pytest.approx(fext_rel(20.0, c.e_eff_even, c.e_eff_odd, 1e-9), rel=1e-5) == pytest.approx(-0.00427, rel=2e-3)
    assert a["t_r_s"] == 1e-9 and a["t_r_source"].startswith("si.net_classes[DEFAULT].t_rise_s")
    assert a["ids"] == ["pcb.stackup.dielectrics[0].thickness_mm", "pcb.stackup.copper[F.Cu].thickness_um", "pcb.stackup.dielectrics[0].er"]
    assert long["matched_estimate_rel"] == pytest.approx(abs(a["next_rel"]) + abs(a["fext_rel"]), rel=1e-5)
    assert short["status"] == S.NOT_VERIFIED.value and [x["net"] for x in short["aggressors"]] == ["LONG"]
    assert short["aggressors"][0]["length_mm"] == pytest.approx(20.0)
    assert "matched-termination estimate, not a verdict" in r.message and "never PASS or FAIL" in r.message


def test_no_neighbour_within_ten_h_is_not_applicable_and_the_window_edge_counts(tmp_path: Path, lib):
    far = xt(pair_board(tmp_path, lib, gap=2.5))  # 12.5 h: beyond the window
    long = rows(far)["LONG"]
    assert far.status is S.NOT_APPLICABLE and long["status"] == S.NOT_APPLICABLE.value and long["beyond_window"] == 1
    assert long["reason"] == "no other net's copper within s <= 10 h of its tracks (1 other net(s) on its layers lie beyond the window)"
    edge = rows(xt(pair_board(tmp_path, lib, gap=2.0)))["LONG"]  # exactly 10 h: inside, and the formula's range includes it
    assert edge["status"] == S.NOT_VERIFIED.value and edge["reason"] == NO_PATH and edge["aggressors"][0]["s_over_h"] == pytest.approx(10.0)
    lone = pair_board(tmp_path, lib)
    lone.pcb.tracks = lone.pcb.tracks[:1]
    r = xt(lone)
    assert r.status is S.NOT_APPLICABLE and rows(r)["SHORT"]["reason"] == "no routed track" and rows(r)["LONG"]["beyond_window"] == 0


def test_a_neighbour_on_an_inner_layer_is_named_with_the_line_geometry_text(tmp_path: Path, lib):
    ir = pair_board(tmp_path, lib, layer="In2.Cu")
    ir.pcb.stackup = escalation_stackup(6, ground_net="GND")
    ir.pcb.layers = board_layers(ir.pcb.stackup)
    r = xt(ir)
    long = rows(r)["LONG"]
    assert long["status"] == S.NOT_VERIFIED.value and long["aggressors"] == []
    assert "In2.Cu is an inner signal layer: stripline geometry (the plane spacing b) is not derived from a stackup by this version" in long["reason"]
    assert r.details["layers"]["In2.Cu"]["h_mm"] == pytest.approx(0.2) and "a choice" in r.details["layers"]["In2.Cu"]["how"]
    assert ["In2.Cu", "In3.Cu"] in [list(p) for p in r.details["broadside_pairs"]]
    # an inner pair with no plane between: a track straight under LONG on In3.Cu is a broadside neighbour
    ir.pcb.tracks[1] = track("SHORT", (20.0, 5.0), (40.0, 5.0), width=W, layer="In3.Cu")
    long = rows(xt(ir))["LONG"]
    assert long["status"] == S.NOT_VERIFIED.value and XT_NO_BROADSIDE in long["reason"] and "In2.Cu over In3.Cu" in long["reason"]


def test_a_two_layer_board_names_the_missing_reference_plane_and_broadside(tmp_path: Path, lib):
    ir = pair_board(tmp_path, lib, layers=2)
    r = xt(ir)
    long = rows(r)["LONG"]
    assert long["status"] == S.NOT_VERIFIED.value and NO_REFERENCE_PLANE in long["reason"] and long["aggressors"] == []
    assert r.details["layers"]["F.Cu"]["h_mm"] == pytest.approx(1.53) and "a choice" in r.details["layers"]["F.Cu"]["how"]
    ir.pcb.tracks[1] = track("SHORT", (20.0, 5.0), (40.0, 5.0), width=W, layer="B.Cu")  # under LONG, the core between
    (item,) = rows(xt(ir))["LONG"]["not_computed"]
    assert item["kind"] == "broadside" and XT_NO_BROADSIDE in item["reason"] and item["nets"] == ["SHORT"] and item["length_mm"] == pytest.approx(20.0)


def test_each_piece_no_formula_covers_is_named(tmp_path: Path, lib):
    wide = rows(xt(pair_board(tmp_path, lib, aggressor_width=0.4)))["LONG"]
    assert wide["status"] == S.NOT_VERIFIED.value and f"{XT_NO_ASYMMETRIC} (victim {W:g} mm, aggressor 0.4 mm" in wide["reason"]
    tight = rows(xt(pair_board(tmp_path, lib, gap=0.06)))["LONG"]  # s <= 2t = 70 um: Jansen's thickness correction is not stated there
    assert "calc.tline.edge_coupled_microstrip refuses" in tight["reason"] and "is not larger than 2 t" in tight["reason"]
    # an aggressor whose class states no edge: Kb is computed, NEXT / FEXT are not, and the victim names the missing value
    ir = pair_board(tmp_path, lib)
    ir.si = si_of(*default_classes(), cls("SLOW", nets=["SHORT"]))
    by = rows(xt(ir))
    long = by["LONG"]
    assert "an aggressor's edge is missing: si.net_classes[SLOW].t_rise_s" in long["reason"] and long["matched_estimate_rel"] is None
    assert [it["nets"] for it in long["not_computed"]] == [["SHORT"]]
    assert long["aggressors"][0]["kb"] == pytest.approx(0.04534, rel=1e-3) and long["aggressors"][0]["next_rel"] is None
    # SHORT as the victim: its aggressor LONG has an edge, but SHORT's own ends are not stated
    assert by["SHORT"]["reason"].startswith("the matched-termination closed forms assume Z0 at both victim ends, and its termination is not stated")
    assert "si.net_classes[SLOW].r_drive_ohm" in by["SHORT"]["reason"] and by["SHORT"]["matched_estimate_rel"] is not None
    # a keep-out that clears the GND plane under the run: no microstrip there
    ir = pair_board(tmp_path, lib)
    ir.pcb.keepouts = [Keepout(id="KO1", layers=["In1.Cu"], rect=user_requirement([25.0, 0.0, 10.0, 10.0], "mm"), forbids=["zones"], reason="test clearance",
                               provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test"))]
    ko = rows(xt(ir))["LONG"]
    assert "impedance is undefined without a reference plane under keep-out KO1" in ko["reason"] and ko["aggressors"] == []


def test_non_parallel_touching_and_zone_neighbours_are_named(tmp_path: Path, lib):
    ir = pair_board(tmp_path, lib)
    ir.pcb.tracks[1] = track("SHORT", (50.0, 5.8), (50.0, 12.0), width=W)  # ends 0.45 mm from LONG's edge
    ir.pcb.tracks.append(track("SHORT", (96.0, 5.0), (100.0, 5.0), width=W))  # in line with LONG, 0.65 mm past its end
    (nonpar, inline) = rows(xt(ir))["LONG"]["not_computed"]
    assert nonpar["kind"] == "non_parallel" and nonpar["reason"] == XT_NON_PARALLEL and nonpar["min_gap_mm"] == pytest.approx(0.8 - W, abs=1e-6)
    assert inline["reason"] == XT_NON_PARALLEL + " (parallel, no projected overlap)" and inline["min_gap_mm"] == pytest.approx(1.0 - W, abs=1e-6)
    ir.pcb.tracks = [ir.pcb.tracks[0], track("SHORT", (50.0, 2.0), (50.0, 8.0), width=W)]  # crosses LONG
    (touch,) = rows(xt(ir))["LONG"]["not_computed"]
    assert touch["kind"] == "contact" and "touches or crosses the victim" in touch["reason"]
    ir.pcb.tracks = ir.pcb.tracks[:1]
    ir.pcb.zones = [Zone(net="SHORT", layer="F.Cu", polygon=[(30.0, 6.0), (40.0, 6.0), (40.0, 8.0), (30.0, 8.0)])]
    ir.pcb.zones.append(Zone(net="GND", layer="F.Cu", polygon=[(60.0, 6.0), (70.0, 6.0), (70.0, 8.0), (60.0, 8.0)]))  # a ground pour: not an aggressor
    long = rows(xt(ir))["LONG"]
    (zone,) = long["not_computed"]
    assert zone["kind"] == "zone" and zone["nets"] == ["SHORT"] and zone["min_gap_mm"] == pytest.approx(1.0 - W / 2.0, abs=1e-6)


def test_no_stackup_and_never_a_verdict(tmp_path: Path, lib):
    none = xt(pair_board(tmp_path, lib, layers=None))
    bare = rows(none)["LONG"]
    assert bare["status"] == S.NOT_VERIFIED.value and bare["reason"].startswith("no stackup") and "no stackup: no window" in none.message
    statuses = set()
    for ir in (pair_board(tmp_path / "a", lib), pair_board(tmp_path / "b", lib, layers=2), pair_board(tmp_path / "c", lib, gap=2.5),
               pair_board(tmp_path / "d", lib, aggressor_width=0.4), pair_board(tmp_path / "e", lib, layers=None)):
        r = xt(ir)
        statuses |= {r.status.value} | {row["status"] for row in r.details["rows"]}
    assert statuses <= {S.NOT_VERIFIED.value, S.NOT_APPLICABLE.value}


def test_si_results_place_it_after_the_pairs_keep_the_others_and_its_own_version(tmp_path: Path, lib):
    from tests.test_via_kinds import _via

    ir = pair_board(tmp_path, lib)
    before = copy.deepcopy(ir)
    h = ir.content_hash()
    got = si_results(ir, lib)
    ids = [r.check_id for r in got]
    assert ids.count("si.crosstalk") == 1 and ids.index("si.crosstalk") == len(ids) - 1  # no pair, no timing path here: last
    a = [r.model_dump(exclude={"timestamp"}) for r in si_results(ir, lib)]
    b = [r.model_dump(exclude={"timestamp"}) for r in si_results(before, lib)]
    assert a == b and ir.content_hash() == h  # deterministic, the IR only read
    others = [r for r in got if r.check_id != "si.crosstalk"]
    assert {(r.tool, r.tool_version) for r in others} == {("si", "0.1")}
    ir.pcb.vias = [_via(20.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="GND", d=0.3, drill=0.1)]  # a typed via: si 0.2, the crosstalk rows keep theirs
    typed = {r.check_id: r for r in si_results(ir, lib)}
    assert (typed["si.crosstalk"].tool, typed["si.crosstalk"].tool_version) == ("si.crosstalk", "0.2") and typed["si.critical_length"].tool_version == "0.2"
    # with a declared pair and a timing path: after si.diff, before si.timing
    from tests.test_si_checks import _timing_ir

    t = _timing_ir(tmp_path / "t", lib)
    ids = [r.check_id for r in si_results(t)]
    assert ids.index("si.crosstalk") == ids.index("si.timing.SPI") - 1


def test_coupled_pieces_keep_their_keys_for_si_diff():
    a = track("P", (0.0, 0.0), (10.0, 0.0), width=0.3)
    b = track("N", (0.0, 0.5), (10.0, 0.5), width=0.3)
    (piece,) = coupled_pieces([a], [b])
    assert set(piece) == {"layer", "length_mm", "width_mm", "gap_mm"} and piece["gap_mm"] == pytest.approx(0.2)


def test_parallel_runs_records_the_overlap_on_both_tracks():
    v = [track("V", (0.0, 0.0), (10.0, 0.0), width=0.3)]
    others = [track("A", (4.0, 0.8), (14.0, 0.8), width=0.3), track("V", (0.0, 0.6), (10.0, 0.6), width=0.3),
              track("B", (2.0, -0.7), (3.0, -0.7), width=0.5, layer="B.Cu")]
    recs = parallel_runs(v, others, {"F.Cu": 0.2}, g_max=10.0, adjacent={"F.Cu": {"B.Cu": 1.0}})
    par, broad = recs
    assert par["kind"] == "parallel" and par["parallel"] and par["net"] == "A" and par["aggressor_index"] == 0  # the victim's own net is skipped
    assert par["length_mm"] == pytest.approx(6.0) and par["gap_mm"] == pytest.approx(0.5) and par["s_over_h"] == pytest.approx(2.5)
    assert par["victim_span"] == [pytest.approx((4.0, 0.0)), pytest.approx((10.0, 0.0))]
    assert par["aggressor_span"] == [pytest.approx((4.0, 0.8)), pytest.approx((10.0, 0.8))]
    assert broad["kind"] == "broadside" and broad["layer"] == "B.Cu" and broad["victim_layer"] == "F.Cu" and broad["lateral_gap_mm"] == pytest.approx(0.3)
    index = NeighbourIndex(others, {"F.Cu": 0.2}, g_max=10.0, adjacent={"F.Cu": {"B.Cu": 1.0}})
    assert index.runs(v) == recs == parallel_runs(v, others, {"F.Cu": 0.2}, g_max=10.0, adjacent={"F.Cu": {"B.Cu": 1.0}})
    assert parallel_runs(v, others, {"F.Cu": 0.04}, g_max=10.0) == []  # a 0.4 mm window: A's 0.5 mm gap lies outside


def test_the_high_speed_domain_check_copies_the_crosstalk_rows(tmp_path: Path, lib):
    from ai_eda.validation.domain import HS_TOOL, HS_VERSION, SignalIntegrityValidator

    v = SignalIntegrityValidator()
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    ir = pair_board(tmp_path, lib)
    (r,) = v.validate(ir, ctx)
    ref = crosstalk_result(ir, ir.si, measure_nets(ir, library=lib))  # the domain check measures with the context's library
    assert r.is_tool_backed and (r.tool, r.tool_version) == (HS_TOOL, HS_VERSION) == ("domain.high_speed", "0.2")
    assert r.status is ref.status is S.NOT_VERIFIED and r.details["rows"] == ref.details["rows"] and r.details["check"] == "si.crosstalk"
    assert r.message.startswith("si.crosstalk: ") and r.details["check_version"] == "0.2"
    ir.si = None
    (none,) = v.validate(ir, ctx)
    assert none.status is S.NOT_VERIFIED and none.is_tool_backed and "no ir.si" in none.message and "placed board" not in none.message
    ir.pcb.placements = []
    (both,) = v.validate(ir, ctx)
    assert "no ir.si" in both.message and "no placed board" in both.message


def test_the_si_report_lists_the_crosstalk_rows(tmp_path: Path, lib):
    from ai_eda.design.base import NO_RECORD
    from ai_eda.report.si_report import crosstalk_lines

    ir = pair_board(tmp_path, lib)
    assert any(line.startswith(f"`si.crosstalk` {NO_RECORD}") for line in crosstalk_lines(ir))  # nothing recorded yet: said so
    ir.validation.add(xt(ir))
    text = "\n".join(crosstalk_lines(ir))
    assert "### 누화 (`si.crosstalk` 기록)" in text and "`si.crosstalk`: **NOT_VERIFIED**" in text and "`LONG`" in text and "`SHORT`" in text
    assert "계산한 평행 구간" in text and "4.534" in text  # Kb in % from the recorded piece


def test_extraction_scales_to_a_board_of_thousands_of_tracks(tmp_path: Path, lib):
    """120 nets x 25 segments (3000 tracks) at a 0.7 mm pitch on F.Cu: the per-layer grid keeps the search local (a loose bound against a
    quadratic search; the 4259-track KR 447 transceiver board took 0.47 s on the development PC, 2026-11-13)."""
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(cls("DEFAULT", default=True, **driver()))
    from ai_eda.ir import Net, NetKind

    ir.nets = [Net(name=f"N{i}", kind=NetKind.SIGNAL, pins=[], provenance=ir.nets[0].provenance) for i in range(120)]
    ir.pcb.tracks = [track(f"N{i}", (4.0 * k, 0.7 * i), (4.0 * k + 4.0, 0.7 * i), width=W) for i in range(120) for k in range(25)]
    mid = rows(xt(ir))["N60"]
    assert {a["net"] for a in mid["aggressors"]} == {"N57", "N58", "N59", "N61", "N62", "N63"} and mid["reason"].startswith(TERMINATION + "victim N60 joins 0 pads")
    assert len(mid["aggressors"]) == 6 and mid["beyond_window"] == 113  # each net one 100 mm straight run: its 25 touching segments merged
    # 0.5 mm gaps along each net: 3000 separate runs, and each neighbour's next segment is a staggered parallel neighbour without overlap
    ir.pcb.tracks = [track(f"N{i}", (4.0 * k, 0.7 * i), (4.0 * k + 3.5, 0.7 * i), width=W) for i in range(120) for k in range(25)]
    t0 = time.perf_counter()
    mid = rows(xt(ir))["N60"]
    assert time.perf_counter() - t0 < 30.0
    assert len(mid["aggressors"]) == 6 * 25 and mid["reason"] == XT_NON_PARALLEL + " (parallel, no projected overlap)"
