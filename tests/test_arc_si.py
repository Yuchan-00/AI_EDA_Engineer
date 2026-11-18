"""Signal integrity on copper that holds arc tracks: lengths, the pad-to-pad path, keep-outs, neck-downs, straight runs, pair
coupling, the crosstalk neighbour search, ``si.crosstalk`` and the ``spice.xtalk`` plan.

A track with ``mid`` is a circular arc (``ai_eda.ir.arc``). Every board is synthetic (the fixture library of
``tests/test_routing.py``, the boards of ``tests/test_si_checks.py``), so nothing here needs KiCad or ngspice. What is proved:

* an arc counts at its arc length - never at the chord between its ends - in a net's routed length and delay, in the impedance,
  length and delay rows, in the critical-length rule and its promotion, in ``si.rf_length`` and in the line ``spice.si`` reads;
* the pad-to-pad path takes an arc as one edge with its exact length, splits it only where a track end or a via of the net lies
  on it (by the fraction along the arc, the parts adding up to the arc), carries the arc piece's ``mid`` in its edges, and does
  not depend on how an arc would be cut into chords: an arc that ends inside a pad box or a via's disc behaves like a segment ending
  there, and a piece without a mid point prints as it did before pieces had one;
* a keep-out that clears the plane is judged on the arc's chords with the bound on the reporting side, a neck-down is sampled on
  the arc itself, and an arc is never a segment of an escape stub;
* an arc is never part of a straight run, of a coupled piece or of a parallel run: a pair's arc is uncoupled copper, and another
  net's copper within the window of an arc is one named ``curved`` record per pair of tracks with the smallest gap, which
  ``si.crosstalk`` names and never feeds to a closed form;
* ``spice.xtalk`` places a span along a path with an arc at the exact distances and refuses a piece on an arc by name;
* a board with an arc is stamped ``si`` 0.3, ``si.crosstalk`` 0.3 and deck version 0.2 (a typed via too: still 0.3), and a board
  without one gives the records, rows and versions it gave before.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_eda.ir import (
    CircuitIR,
    DiffPair,
    Keepout,
    PinElectricalType,
    PinRef,
    Provenance,
    ProvenanceKind,
    Track,
    Via,
    Zone,
    track_length,
    track_points,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.arc import ARC_SAGITTA_MM, arc_point_at, arc_problem, arc_split, point_arc_distance, point_arc_fraction
from ai_eda.tools.calc.tline import C0, edge_coupled_microstrip, microstrip, solve_microstrip_width
from ai_eda.tools.routing.coupling import NeighbourIndex, _chain_dist, _seg_dist, coupled_pieces, parallel_runs, uncoupled_lengths
from ai_eda.tools.si.measure import SI_ARC_VERSION, SI_TYPED_VIA_VERSION, SI_VERSION, measure_nets, plane_keepout, si_version
from ai_eda.tools.si.paths import PadBox, PathEdge, longest_path, net_pads
from ai_eda.tools.si.promote import critical_rows, promote
from ai_eda.validation import ValidationContext
from ai_eda.validation.si import XT_ARC_VERSION, XT_CURVED, XT_VERSION, _escape_segment, _neckdown, _straight_runs, crosstalk_result, crosstalk_scan, si_results, xt_version
from tests.test_routing import board_ir, fixture_library
from tests.test_si_checks import NECK_PROV, cls, default_classes, long_board, si_of, stacked, track, u

S2 = math.sqrt(0.5)
#: the 50-ohm width on the 4-layer generic F.Cu (h 0.2 mm, t 35 um, er 4.5) and an edge gap of 1.5 h (as tests/test_si_crosstalk.py)
W = round(solve_microstrip_width(50.0, 0.2, 35.0, 4.5), 4)
GAP = 0.3
PITCH = W + GAP
H = 0.2
#: delay per metre of the hand-made path graphs below (any number: only lengths and edges are judged there)
T_PD = {("F.Cu", 0.4): (6.0e-9, False), ("B.Cu", 0.4): (6.0e-9, False)}
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def arc(net: str, start, mid, end, width: float = 0.4, layer: str = "F.Cu") -> Track:
    return Track(net=net, layer=layer, start=start, mid=mid, end=end, width_mm=width)


def quarter(net: str, cx: float, cy: float, r: float, width: float = 0.4, layer: str = "F.Cu") -> Track:
    """The quarter arc of the circle around ``(cx, cy)`` from its top point ``(cx, cy - r)`` to its right point ``(cx + r, cy)``
    (Y down: the corner of a track that runs right and then down). Its three points lie on the circle to float precision, so its
    length is ``r pi / 2`` to about 1e-15 (a stored arc's points are rounded to 1 nm, which moves the length by less than that)."""
    return arc(net, (cx, cy - r), (cx + r * S2, cy - r * S2), (cx + r, cy), width, layer)


def bend_board(tmp: Path, lib, layers: int | None = 4, *, r: float = 3.0, width: float = 0.4) -> CircuitIR:
    """``LONG`` from R1 (5, 5) right to (25, 5), round a quarter arc of radius ``r`` and down to R2 (25 + r, 30): 45 - r mm of
    straight track and the arc; ``SHORT`` 10 mm, far from it; the generic stack. R1.1 is an output and R2.1 an input."""
    x2 = 25.0 + r
    parts = [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", x2, 30.0), ("R3", "PAD1", 5.0, 36.0), ("R4", "PAD1", 15.0, 36.0)]
    ir = board_ir(tmp, lib, parts, {"LONG": [("R1", "1"), ("R2", "1")], "SHORT": [("R3", "1"), ("R4", "1")]}, (45.0, 40.0))
    ir.component("R1").pins[0].electrical_type = PinElectricalType.OUTPUT
    ir.component("R2").pins[0].electrical_type = PinElectricalType.INPUT
    stacked(ir, layers)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (25.0, 5.0), width=width), quarter("LONG", 25.0, 5.0 + r, r, width=width),
                     track("LONG", (x2, 5.0 + r), (x2, 30.0), width=width), track("SHORT", (5.0, 36.0), (15.0, 36.0), width=width)]
    return ir


def results(ir: CircuitIR, lib=None) -> dict:
    return {r.check_id: r for r in si_results(ir, lib)}


def rows(result) -> dict[str, dict]:
    return {r["net"]: r for r in result.details["rows"]}


# --------------------------------------------------------------------------- lengths and delays


def test_a_nets_length_and_delay_count_an_arc_at_its_arc_length_not_at_its_chord(tmp_path: Path, lib):
    ir = bend_board(tmp_path, lib)
    true, chords = 42.0 + 3.0 * math.pi / 2.0, 42.0 + 3.0 * math.sqrt(2.0)
    m = measure_nets(ir)["LONG"]
    assert m.track_length_mm == pytest.approx(true, abs=1e-9) and m.length_mm == m.track_length_mm and true - chords > 0.46
    (seg,) = m.segments
    assert (seg.layer, seg.width_mm) == ("F.Cu", 0.4) and seg.length_mm == pytest.approx(true, abs=1e-9)
    t_pd = math.sqrt(microstrip(0.4, 0.2, 35.0, 4.5).e_eff) / C0
    assert m.delay_s == pytest.approx(true / 1000.0 * t_pd, rel=1e-12) and m.summary()["track_length_mm"] == round(true, 6)
    assert sum(track_length(t) for t in ir.pcb.tracks[:3]) == pytest.approx(true, abs=1e-9)
    # the same corner as a straight chamfer between the same two points is the chord: shorter by 0.4712 mm
    ir.pcb.tracks[1] = track("LONG", (25.0, 5.0), (28.0, 8.0))
    assert measure_nets(ir)["LONG"].track_length_mm == pytest.approx(chords, abs=1e-9)


def test_the_arcs_length_counts_under_its_own_layer_and_width_in_the_impedance_length_and_delay_rows(tmp_path: Path, lib):
    ir = bend_board(tmp_path, lib, width=W)
    ir.pcb.tracks[1] = quarter("LONG", 25.0, 8.0, 3.0, width=0.5)  # the corner wider than the two legs: a segment of its own
    ir.si = si_of(*default_classes(), cls("FAST", nets=["LONG"], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1), max_length_mm=u(46.5, "mm"),
                                          max_delay_s=u(280e-12, "s")))
    bend = 1.5 * math.pi
    m = measure_nets(ir)["LONG"]
    assert [(s.layer, s.width_mm, pytest.approx(s.length_mm, abs=1e-9)) for s in m.segments] == [("F.Cu", W, 42.0), ("F.Cu", 0.5, bend)]
    res = results(ir)
    imp = {(r["layer"], r["width_mm"]): r for r in res["si.impedance.FAST"].details["segments"]}
    assert imp[("F.Cu", W)]["length_mm"] == 42.0 and imp[("F.Cu", W)]["status"] == S.PASS.value
    assert imp[("F.Cu", 0.5)]["length_mm"] == round(bend, 6) and imp[("F.Cu", 0.5)]["status"] == S.FAIL.value  # judged at its own width
    # 42 + 4.712 mm of track: over the 46.5 mm budget the chord sum (46.243 mm) would meet
    (length,) = res["si.length.FAST"].details["nets"]
    assert res["si.length.FAST"].status is S.FAIL and length["track_length_mm"] == round(42.0 + bend, 6) and 42.0 + 3.0 * math.sqrt(2.0) < 46.5
    t_legs, t_bend = (math.sqrt(microstrip(w, 0.2, 35.0, 4.5).e_eff) / C0 for w in (W, 0.5))
    (delay,) = res["si.delay.FAST"].details["nets"]
    assert delay["delay_ps"] == pytest.approx((42.0 * t_legs + bend * t_bend) / 1000.0 * 1e12, abs=1e-3)


# --------------------------------------------------------------------------- the pad-to-pad path


def test_the_longest_path_takes_an_arc_as_one_edge_of_its_exact_length(tmp_path: Path, lib):
    ir = bend_board(tmp_path, lib)
    bend = ir.pcb.tracks[1]
    m = measure_nets(ir, library=lib)["LONG"]
    p = m.path
    assert m.path_problem is None and p.ends == ("R1.1", "R2.1") and p.vias == 0 and p.main == ("F.Cu", 0.4)
    assert p.length_mm == pytest.approx(42.0 + 1.5 * math.pi, abs=1e-9) and p.track_length_mm == p.length_mm
    assert [e.length_mm for e in p.edges] == [20.0, pytest.approx(1.5 * math.pi, abs=1e-9), 22.0]  # three edges: no node per chord
    one = p.edges[1]
    assert (one.layer, one.width_mm, one.start, one.end) == ("F.Cu", 0.4, (25.0, 5.0), (28.0, 8.0))
    assert one.mid == (round(bend.mid[0], 6), round(bend.mid[1], 6)) and p.edges[0].mid is None and p.edges[2].mid is None
    t_pd = math.sqrt(microstrip(0.4, 0.2, 35.0, 4.5).e_eff) / C0
    assert one.delay_s == pytest.approx(1.5 * math.pi / 1000.0 * t_pd, rel=1e-12) and p.delay_s == pytest.approx(p.length_mm / 1000.0 * t_pd, rel=1e-12)
    assert m.line.how == "path" and m.line.length_mm == p.length_mm and len(track_points(bend)) > 20  # the chain has 30 chords: none is a node
    # a straight piece is the edge it was: built without a mid point, equal to one built with None
    assert p.edges[0] == PathEdge("F.Cu", 0.4, 20.0, (5.0, 5.0), (25.0, 5.0), p.edges[0].delay_s)


def _tee_board(tmp: Path, lib, fraction: float, far: tuple[float, float]) -> tuple[CircuitIR, tuple[float, float]]:
    """The bend board with a third pad R7 at ``far``, joined by a track that ends on the arc at ``fraction`` of its length."""
    ir = bend_board(tmp, lib)
    bend = ir.pcb.tracks[1]
    x, y = arc_point_at(bend.start, bend.mid, bend.end, fraction)
    tee = (round(x, 6), round(y, 6))
    ir.components.append(ir.component("R2").model_copy(deep=True, update={"ref": "R7"}))
    ir.pcb.placements.append(ir.pcb.placement("R2").model_copy(update={"component_ref": "R7", "x_mm": far[0], "y_mm": far[1]}))
    ir.net("LONG").pins.append(PinRef(component_ref="R7", pin_number="1"))
    ir.pcb.tracks.append(track("LONG", tee, far))
    return ir, tee


def test_a_track_that_ends_on_an_arc_splits_it_there_by_the_fraction_along_the_arc(tmp_path: Path, lib):
    total = 1.5 * math.pi
    ir, tee = _tee_board(tmp_path / "near", lib, 0.3, (40.0, 2.0))
    bend = ir.pcb.tracks[1]
    d, f = point_arc_fraction(tee, bend.start, bend.mid, bend.end)
    chain = track_points(bend)
    # the junction lies on the arc (within the rounding of its point), between two vertices of the chain and on none of them
    assert d <= 1e-6 and f == pytest.approx(0.3, abs=1e-6) and min(math.dist(tee, v) for v in chain) > 0.02
    p = measure_nets(ir, library=lib)["LONG"].path
    assert p.ends == ("R1.1", "R2.1") and len(p.edges) == 4  # the branch is shorter: the line still runs through the whole arc
    first, second = p.edges[1], p.edges[2]
    assert (first.start, first.end, second.start, second.end) == ((25.0, 5.0), tee, tee, (28.0, 8.0))
    assert first.length_mm == pytest.approx(f * total, abs=1e-12) and second.length_mm == pytest.approx((1.0 - f) * total, abs=1e-12)
    assert first.length_mm + second.length_mm == pytest.approx(total, abs=1e-12) and p.length_mm == pytest.approx(42.0 + total, abs=1e-9)
    for e in (first, second):  # each part is an arc of its own: its mid point lies on the arc, between its ends
        assert arc_problem(e.start, e.mid, e.end) is None and point_arc_distance(e.mid, bend.start, bend.mid, bend.end) <= 1e-6
    assert point_arc_fraction(first.mid, bend.start, bend.mid, bend.end)[1] == pytest.approx(f / 2.0, abs=1e-6)
    assert point_arc_fraction(second.mid, bend.start, bend.mid, bend.end)[1] == pytest.approx((1.0 + f) / 2.0, abs=1e-6)
    (_, m1, _), (_, m2, _) = arc_split(bend.start, bend.mid, bend.end, f)  # the mid points of the two parts the arc kit cuts there
    assert (first.mid, second.mid) == (m1, m2)
    # a long branch: the line is R2 - the arc's second part - the branch, and the part walked backwards keeps its mid point
    ir, tee = _tee_board(tmp_path / "far", lib, 0.3, (60.0, 2.0))
    branch = math.dist(tee, (60.0, 2.0))
    p = measure_nets(ir, library=lib)["LONG"].path
    assert p.ends == ("R2.1", "R7.1") and p.length_mm == pytest.approx(22.0 + (1.0 - f) * total + branch, abs=1e-9)
    assert [e.length_mm for e in p.edges] == [22.0, pytest.approx((1.0 - f) * total, abs=1e-12), pytest.approx(branch, abs=1e-12)]
    assert (p.edges[1].start, p.edges[1].end, p.edges[1].mid) == ((28.0, 8.0), tee, second.mid)


def _pad(label: str, x: float, y: float, half: float = 0.8) -> PadBox:
    return PadBox(label, x, y, half, half, frozenset({"F.Cu", "B.Cu"}))


def _path(tracks, pads, vias=()):
    return longest_path("N", list(tracks), list(vias), {p.label: [p] for p in pads}, None, T_PD, 7.0e-9)


#: a hand-made net: A (0, 0) - 10 mm - a quarter arc of radius 2 - 8 mm - B (12, 10)
LEAD_A, BEND, LEAD_B = track("N", (0.0, 0.0), (10.0, 0.0)), quarter("N", 10.0, 2.0, 2.0), track("N", (12.0, 2.0), (12.0, 10.0))
PAD_A, PAD_B = _pad("A.1", 0.0, 0.0), _pad("B.1", 12.0, 10.0)


def _on_bend(fraction: float) -> tuple[float, float]:
    x, y = arc_point_at(BEND.start, BEND.mid, BEND.end, fraction)
    return (round(x, 6), round(y, 6))


def test_two_junctions_and_a_via_on_an_arc_cut_it_into_arc_pieces_that_add_up():
    t1, t2 = _on_bend(0.25), _on_bend(0.75)
    stubs = [track("N", t1, (t1[0] + 2.0, t1[1] - 2.0)), track("N", t2, (t2[0] + 2.0, t2[1] - 2.0))]
    p, why = _path([LEAD_A, BEND, LEAD_B, *stubs], [PAD_A, PAD_B, _pad("C.1", t1[0] + 2.0, t1[1] - 2.0, 0.3), _pad("D.1", t2[0] + 2.0, t2[1] - 2.0, 0.3)])
    assert why is None and p.ends == ("A.1", "B.1") and p.length_mm == pytest.approx(18.0 + math.pi, abs=1e-9)
    pieces = p.edges[1:4]
    assert [(e.start, e.end) for e in pieces] == [((10.0, 0.0), t1), (t1, t2), (t2, (12.0, 2.0))]
    assert [e.length_mm for e in pieces] == [pytest.approx(math.pi / 4.0, abs=1e-6), pytest.approx(math.pi / 2.0, abs=1e-6), pytest.approx(math.pi / 4.0, abs=1e-6)]
    assert sum(e.length_mm for e in pieces) == pytest.approx(math.pi, abs=1e-12)
    assert all(arc_problem(e.start, e.mid, e.end) is None and point_arc_distance(e.mid, BEND.start, BEND.mid, BEND.end) <= 1e-6 for e in pieces)
    # a via whose centre lies on the arc: the arc is cut at it, and the line goes down its barrel to the far pad on the other layer
    vx, vy = _on_bend(0.5)
    via = Via(net="N", x_mm=vx, y_mm=vy, drill_mm=0.3, diameter_mm=0.6)
    p, why = _path([LEAD_A, BEND, LEAD_B, track("N", (vx, vy), (40.0, vy), layer="B.Cu")], [PAD_A, PAD_B, _pad("C.1", 40.0, vy)], [via])
    assert why is None and p.ends == ("A.1", "C.1") and p.vias == 1 and [e.layer for e in p.edges] == ["F.Cu", "F.Cu", None, "B.Cu"]
    assert p.edges[1].length_mm == pytest.approx(math.pi / 2.0, abs=1e-6) and (p.edges[1].start, p.edges[1].end) == ((10.0, 0.0), (vx, vy))
    assert p.edges[1].mid is not None and p.edges[2].mid is None and p.length_mm == pytest.approx(10.0 + math.pi / 2.0 + 40.0 - vx, abs=1e-6)


def test_the_path_graph_reads_the_arc_never_its_chord():
    # a track from the middle of the chord joins nothing: no copper is there
    p, why = _path([LEAD_A, BEND, LEAD_B, track("N", (11.0, 1.0), (11.0, 6.0))], [PAD_A, PAD_B, _pad("C.1", 11.0, 6.0)])
    assert p is None and why == "the copper does not join pad(s) C.1 to the others"
    # a straight track between the arc's two ends beside it: two ways round, a loop
    p, why = _path([LEAD_A, BEND, LEAD_B, track("N", (10.0, 0.0), (12.0, 2.0))], [PAD_A, PAD_B])
    assert p is None and why == "the copper forms a loop (4 pieces on 4 nodes): the pad-to-pad path is not unique"
    # a branch that starts one grid step beyond the arc's end is that end: joined, the arc not split (at x = 40 the step is
    # 9.99999997e-7 mm in floats, inside the 1e-6 mm tolerance)
    turn = arc("N", (37.0, 13.0), (40.0 - 3.0 * S2, 13.0 - 3.0 * S2), (40.0, 10.0))
    p, why = _path([turn, track("N", (40.000001, 10.0), (50.0, 10.0))], [_pad("A.1", 37.0, 13.0), _pad("C.1", 50.0, 10.0)])
    assert why is None and [e.length_mm for e in p.edges] == [pytest.approx(1.5 * math.pi, abs=1e-9), pytest.approx(9.999999, abs=1e-9)]
    assert p.edges[0].mid is not None and (p.edges[0].start, p.edges[0].end) == ((37.0, 13.0), (40.0, 10.0))


def test_an_arc_that_ends_inside_a_pad_box_behaves_like_a_segment_ending_there():
    box = _pad("B.1", 12.3, 2.3)  # holds the arc's end (12, 2) and the last half millimetre of the arc, not its start
    inside = [v for v in track_points(BEND)[1:-1] if box.contains(*v)]
    assert box.contains(12.0, 2.0) and not box.contains(10.0, 0.0) and len(inside) >= 3  # chain vertices a pad would swallow as nodes
    curved, why = _path([LEAD_A, BEND], [PAD_A, box])
    stub, why_stub = _path([LEAD_A, track("N", (10.0, 0.0), (12.0, 2.0))], [PAD_A, box])
    assert why is None and why_stub is None and curved.ends == stub.ends == ("A.1", "B.1")
    assert curved.edges[0] == stub.edges[0] and len(curved.edges) == len(stub.edges) == 2
    assert (curved.edges[1].start, curved.edges[1].end) == (stub.edges[1].start, stub.edges[1].end) == ((10.0, 0.0), (12.0, 2.0))
    # the whole track counts in both, whatever part of it lies inside the pad: the arc at pi, the stub at its 2.828 mm
    assert curved.edges[1].length_mm == pytest.approx(math.pi, abs=1e-12) and stub.edges[1].length_mm == pytest.approx(2.0 * math.sqrt(2.0), abs=1e-12)
    assert curved.edges[1].mid is not None and stub.edges[1].mid is None
    assert curved.length_mm - stub.length_mm == pytest.approx(math.pi - 2.0 * math.sqrt(2.0), abs=1e-12)


def test_junctions_at_an_arcs_start_on_its_bulge_and_just_inside_its_end():
    # a branch that starts one grid step beyond the arc's start (the same arc, stored from its other end): that start, joined, not split
    turn = arc("N", (40.0, 10.0), (40.0 - 3.0 * S2, 13.0 - 3.0 * S2), (37.0, 13.0))
    p, why = _path([turn, track("N", (40.000001, 10.0), (50.0, 10.0))], [_pad("A.1", 37.0, 13.0), _pad("C.1", 50.0, 10.0)])
    assert why is None and [e.length_mm for e in p.edges] == [pytest.approx(1.5 * math.pi, abs=1e-9), pytest.approx(9.999999, abs=1e-9)]
    assert (p.edges[0].start, p.edges[0].end) == ((37.0, 13.0), (40.0, 10.0))
    # a branch from the top of a half circle: the junction lies 5 mm outside the box of the arc's two ends (the line y = 0)
    half = arc("N", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0))
    p, why = _path([half, track("N", (5.0, -5.0), (5.0, -20.0))], [_pad("A.1", 0.0, 0.0), _pad("B.1", 10.0, 0.0), _pad("C.1", 5.0, -20.0)])
    assert why is None and p.ends == ("A.1", "C.1") and p.length_mm == pytest.approx(2.5 * math.pi + 15.0, abs=1e-9)
    assert [(e.start, e.end) for e in p.edges] == [((0.0, 0.0), (5.0, -5.0)), ((5.0, -5.0), (5.0, -20.0))] and p.edges[0].mid is not None
    # a branch 10 um before the arc's end, and one 3 nm before it: both beyond the 1 nm tolerance, so both split the arc, and the
    # line leaves through the branch without the arc's last piece
    for before in (0.01, 3.0e-6):
        tee = _on_bend(1.0 - before / math.pi)
        far = (tee[0] + 30.0, tee[1])
        p, why = _path([LEAD_A, BEND, track("N", tee, far)], [PAD_A, _pad("C.1", *far)])
        assert why is None and p.ends == ("A.1", "C.1") and p.edges[1].end == tee != (12.0, 2.0)
        assert p.edges[1].length_mm == pytest.approx(math.pi - before, abs=1e-6) and p.length_mm == pytest.approx(40.0 + math.pi - before, abs=1e-6)


def test_an_arc_that_ends_inside_a_via_disc_joins_the_via_and_two_arcs_between_two_pads_are_a_loop():
    # the arc ends 0.2 mm from the centre of a 0.6 mm via: inside its disc, so it joins the via as a straight track ending there does
    via = Via(net="N", x_mm=12.2, y_mm=2.0, drill_mm=0.3, diameter_mm=0.6)
    down = track("N", (12.2, 2.0), (30.0, 2.0), layer="B.Cu")
    curved, why = _path([LEAD_A, BEND, down], [PAD_A, _pad("C.1", 30.0, 2.0)], [via])
    stub, why_stub = _path([LEAD_A, track("N", (10.0, 0.0), (12.0, 2.0)), down], [PAD_A, _pad("C.1", 30.0, 2.0)], [via])
    assert why is None and why_stub is None and [e.layer for e in curved.edges] == [e.layer for e in stub.edges] == ["F.Cu", "F.Cu", None, "B.Cu"]
    assert curved.length_mm == pytest.approx(10.0 + math.pi + 17.8, abs=1e-9) and stub.length_mm == pytest.approx(10.0 + 2.0 * math.sqrt(2.0) + 17.8, abs=1e-9)
    # a via whose centre lies on the arc at nine tenths and whose disc holds the arc's end: the last tenth is inside the via, as the
    # end of a straight track inside a via's disc is
    vx, vy = _on_bend(0.9)
    wide = Via(net="N", x_mm=vx, y_mm=vy, drill_mm=0.3, diameter_mm=0.8)
    p, why = _path([LEAD_A, BEND, LEAD_B], [PAD_A, PAD_B], [wide])
    assert math.dist((vx, vy), (12.0, 2.0)) < 0.4 and why is None and p.vias == 0
    assert [e.length_mm for e in p.edges] == [10.0, pytest.approx(0.9 * math.pi, abs=1e-6), 8.0] and p.edges[1].end == (vx, vy)
    # two half circles between the same two pads: two ways round
    top, bottom = arc("N", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0)), arc("N", (0.0, 0.0), (5.0, 5.0), (10.0, 0.0))
    p, why = _path([top, bottom], [_pad("A.1", 0.0, 0.0), _pad("B.1", 10.0, 0.0)])
    assert p is None and why == "the copper forms a loop (2 pieces on 2 nodes): the pad-to-pad path is not unique"


def test_a_path_piece_without_a_mid_point_prints_as_it_did_before_pieces_had_one():
    straight = PathEdge("F.Cu", 0.4, 20.0, (5.0, 5.0), (25.0, 5.0), 1.2e-10)
    barrel = PathEdge(None, None, 1.6, (1.0, 2.0), (1.0, 2.0), None)
    assert repr(straight) == "PathEdge(layer='F.Cu', width_mm=0.4, length_mm=20.0, start=(5.0, 5.0), end=(25.0, 5.0), delay_s=1.2e-10)"
    assert repr(barrel) == "PathEdge(layer=None, width_mm=None, length_mm=1.6, start=(1.0, 2.0), end=(1.0, 2.0), delay_s=None)"
    curved = PathEdge("F.Cu", 0.4, 3.0, (0.0, 0.0), (2.0, 2.0), None, (1.5, 0.5))
    assert repr(curved) == "PathEdge(layer='F.Cu', width_mm=0.4, length_mm=3.0, start=(0.0, 0.0), end=(2.0, 2.0), delay_s=None, mid=(1.5, 0.5))"
    # a whole path, as extracted: without an arc no piece names a mid point, with one only the arc's piece does
    plain, _ = _path([LEAD_A, track("N", (10.0, 0.0), (12.0, 10.0))], [PAD_A, PAD_B])
    bent, _ = _path([LEAD_A, BEND, LEAD_B], [PAD_A, PAD_B])
    assert repr(plain).count("PathEdge(") == 2 and "mid" not in repr(plain)
    assert repr(bent).count("PathEdge(") == 3 and repr(bent).count("mid=(") == 1


# --------------------------------------------------------------------------- the version stamps


def test_a_board_with_an_arc_is_stamped_with_the_arc_versions_and_every_other_board_as_before(tmp_path: Path, lib):
    from tests.test_via_kinds import _via

    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    assert (SI_VERSION, SI_TYPED_VIA_VERSION, SI_ARC_VERSION, XT_VERSION, XT_ARC_VERSION) == ("0.1", "0.2", "0.3", "0.2", "0.3")
    assert si_version(ir) == "0.1" and xt_version(ir) == "0.2"
    assert {(r.tool, r.tool_version) for r in si_results(ir, lib)} == {("si", "0.1"), ("si.crosstalk", "0.2")}
    assert promote(ir)[1][0].provenance.tool_version == "0.1"
    typed = _via(20.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="GND", d=0.3, drill=0.1)
    ir.pcb.vias = [typed]
    assert si_version(ir) == "0.2" and xt_version(ir) == "0.2"
    # SHORT's 10 mm with a half-circle bump in the middle: one arc on the board
    bump = [track("SHORT", (5.0, 15.0), (9.0, 15.0)), arc("SHORT", (9.0, 15.0), (10.0, 14.0), (11.0, 15.0)), track("SHORT", (11.0, 15.0), (15.0, 15.0))]
    ir.pcb.tracks = [ir.pcb.tracks[0], *bump]
    assert si_version(ir) == "0.3" and xt_version(ir) == "0.3"  # a typed via and an arc: the arc version
    assert {(r.tool, r.tool_version) for r in si_results(ir, lib)} == {("si", "0.3"), ("si.crosstalk", "0.3")}
    ir.pcb.vias = []
    assert si_version(ir) == "0.3" and xt_version(ir) == "0.3"
    assert {(r.tool, r.tool_version) for r in si_results(ir, lib)} == {("si", "0.3"), ("si.crosstalk", "0.3")}
    assert promote(ir)[1][0].provenance.tool_version == "0.3" and measure_nets(ir)["SHORT"].track_length_mm == pytest.approx(8.0 + math.pi, abs=1e-9)


def test_the_rules_that_read_a_nets_line_get_the_arc_true_length(tmp_path: Path, lib):
    """55 mm of straight track and a half circle of radius 10: 86.42 mm of copper where the chords say 75 mm; l_crit is 82.64 mm."""
    from ai_eda.tools.spice.si_check import spice_si_results
    from ai_eda.validation.domain import SignalIntegrityValidator
    from tests.test_rf_checks import rf_board, rf_class, rf_si

    ir = long_board(tmp_path / "p", lib, 4, length=75.0)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (30.0, 5.0)), arc("LONG", (30.0, 5.0), (40.0, 15.0), (50.0, 5.0)), track("LONG", (50.0, 5.0), (80.0, 5.0))]
    true = 55.0 + 10.0 * math.pi
    row = {r.net: r for r in critical_rows(ir, library=lib)}["LONG"]
    assert row.status == "long" and row.length_mm == pytest.approx(true, abs=1e-9) and 75.0 < row.l_crit_mm < true
    assert row.measure == "the longest pad-to-pad path R1.1-R2.1"
    (promotion,) = promote(ir, library=lib)[1]
    assert promotion.net == "LONG" and promotion.length_mm == round(true, 6) and promotion.provenance.tool_version == "0.3"
    crit = results(ir, lib)["si.critical_length"]
    assert crit.details["long_over_plane"] == ["LONG"] and crit.tool_version == "0.3"
    (deck,) = spice_si_results(ir, {"kicad_library": lib}, tmp_path / "w")  # no engine: the line the deck would take is in the details
    assert deck.status is S.NOT_VERIFIED and deck.details["line_mm"] == round(true, 6) and deck.details["length_mm"] == round(true, 6)
    (hs,) = SignalIntegrityValidator().validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert hs.details["check"] == "si.crosstalk" and hs.details["check_version"] == "0.3"
    # the same ends with a straight track where the half circle is: 75 mm, short, nothing promoted, the versions of before
    ir.pcb.tracks[1] = track("LONG", (30.0, 5.0), (50.0, 5.0))
    assert {r.net: r for r in critical_rows(ir, library=lib)}["LONG"].status == "short" and promote(ir, library=lib)[1] == []
    assert results(ir, lib)["si.critical_length"].tool_version == "0.1"
    # si.rf_length at 900 MHz on two layers (l_crit 15.70 mm by the no-plane bound): 8 mm and a half circle of radius 3 are
    # 17.42 mm - possibly long - where the chords say 14 mm
    rf = rf_board(tmp_path / "rf", lib, 2, 14.0)
    rf.si = rf_si(rf_class())
    rf.pcb.tracks = [track("ANT", (5.0, 5.0), (9.0, 5.0)), arc("ANT", (9.0, 5.0), (12.0, 8.0), (15.0, 5.0)), track("ANT", (15.0, 5.0), (19.0, 5.0)),
                     track("GND", (5.0, 15.0), (15.0, 15.0))]
    r = results(rf, lib)["si.rf_length"]
    (ant,) = r.details["nets"]
    assert r.status is S.NOT_VERIFIED and "possibly long at 900 MHz" in r.message and ant["length_mm"] == round(8.0 + 3.0 * math.pi, 6) > ant["l_crit_mm"] > 14.0
    rf.pcb.tracks[1] = track("ANT", (9.0, 5.0), (15.0, 5.0))
    assert results(rf, lib)["si.rf_length"].status is S.PASS


# --------------------------------------------------------------------------- keep-outs, neck-downs, escape stubs


def _keepout(x: float, y: float, w: float, h: float) -> Keepout:
    return Keepout(id="KO1", layers=["In1.Cu"], rect=user_requirement([x, y, w, h], "mm"), forbids=["zones"], reason="test clearance", provenance=USER)


def test_a_cleared_plane_is_judged_on_the_arc_with_the_bound_on_the_reporting_side(tmp_path: Path, lib):
    ir = bend_board(tmp_path, lib, width=W)
    ir.si = si_of(*default_classes(), cls("FAST", nets=["LONG"], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    leg_in, bend, leg_out = ir.pcb.tracks[:3]
    chord = track("LONG", (25.0, 5.0), (28.0, 8.0), width=W)
    # under the arc's bulge, clear of the chord between its ends: the arc has no plane there, a chamfer would have one
    ir.pcb.keepouts = [_keepout(27.0, 5.2, 0.6, 0.6)]
    assert plane_keepout(ir, chord) is None and plane_keepout(ir, leg_in) is None and plane_keepout(ir, leg_out) is None
    assert plane_keepout(ir, bend) == ("impedance is undefined without a reference plane under keep-out KO1: it forbids zones on In1.Cu (the GND plane) "
                                       "under this copper (test clearance)")
    (seg,) = measure_nets(ir)["LONG"].segments
    assert seg.keepout_mm == pytest.approx(1.5 * math.pi, abs=1e-9) and seg.keepout_id == "KO1"
    imp = results(ir)["si.impedance.FAST"]
    under, over = sorted(imp.details["segments"], key=lambda r: r["length_mm"])
    assert imp.status is S.NOT_VERIFIED and under["status"] == S.NOT_VERIFIED.value and under["length_mm"] == round(1.5 * math.pi, 6)
    assert "under keep-out KO1" in under["reason"] and over["status"] == S.PASS.value and over["length_mm"] == 42.0
    # inside the corner, on the chord and clear of the arc: the arc has its plane, a chamfer would not
    ir.pcb.keepouts = [_keepout(26.4, 6.4, 0.2, 0.2)]
    assert plane_keepout(ir, bend) is None and plane_keepout(ir, chord) is not None
    # an area whose edge is 0.5 um clear of the copper at the point where the arc ends and the leg begins: within the arc's
    # 1 um chain bound it counts as over it (the reporting side), the straight leg is judged exactly; 1.5 um clear, neither
    half = W / 2.0
    ir.pcb.keepouts = [_keepout(28.0 + half + 0.0005, 7.5, 1.0, 0.5)]
    assert ARC_SAGITTA_MM == 0.001 and plane_keepout(ir, bend) is not None and plane_keepout(ir, leg_out) is None
    ir.pcb.keepouts = [_keepout(28.0 + half + 0.0015, 7.5, 1.0, 0.5)]
    assert plane_keepout(ir, bend) is None and plane_keepout(ir, leg_out) is None
    ir.pcb.keepouts = [_keepout(28.0 + half - 0.0005, 7.5, 1.0, 0.5)]
    assert plane_keepout(ir, bend) is not None and plane_keepout(ir, leg_out) is not None


def test_an_arc_over_a_cleared_plane_is_always_named_and_one_judged_as_a_microstrip_is_surely_clear(tmp_path: Path, lib):
    ir = bend_board(tmp_path, lib, width=W)
    half = arc("LONG", (10.0, 20.0), (15.0, 15.0), (20.0, 20.0), width=W)  # a half circle: its top, (15, 15), lies between two points of its chain
    top = min(y for _, y in track_points(half))  # the chord under the arc's top (Y down): there the arc lies most of the bound outside its chords
    assert 0.0005 < top - 15.0 <= ARC_SAGITTA_MM

    def named(edge: float) -> bool:
        """Whether the arc is named over a cleared area above its top that reaches down to ``edge``."""
        ir.pcb.keepouts = [_keepout(14.0, edge - 1.0, 2.0, 1.0)]
        return plane_keepout(ir, half) is not None

    copper, chords = 15.0 - W / 2.0, top - W / 2.0  # the top edge of the arc's copper, and of its chords' copper
    # the arc's copper reaches 0.1 um into the area: named, though its chords stay most of a micron clear of it
    assert named(copper + 0.0001) and chords - (copper + 0.0001) > 0.0003
    # its chords come within the bound of the area (half a micron clear): named too - the reporting side
    assert named(chords - 0.0005)
    # its chords are 1.5 um clear, so the arc is at least 0.5 um clear: it has its plane
    assert not named(chords - 0.0015) and copper - (chords - 0.0015) >= 0.0005
    # between the two the chords decide: here the arc's copper is 0.5 um clear of the area, and it is not named
    assert not named(copper - 0.0005)


def test_a_neck_down_is_sampled_on_the_arc_and_an_arc_is_never_an_escape_segment(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)  # R2's 1.6 mm pad at (95, 5); the rule's reach is 1 mm + one 0.25 mm grid step from its box
    pads = net_pads(ir, lib).pads["LONG"]
    flat = track("LONG", (93.0, 5.0), (97.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV})
    bulge = arc("LONG", (93.0, 5.0), (95.0, 2.0), (97.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV})
    near = arc("LONG", (93.2, 5.0), (93.6, 4.6), (94.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV})
    assert _neckdown(flat, pads) == (True, "neck-down recorded by the router within 1 mm + one 0.25 mm grid step of the net's pads")
    assert _neckdown(near, pads) == _neckdown(flat, pads)
    # the same two ends, the copper 3 mm out from the pad: its chord stays within the reach, the arc leaves it
    assert _neckdown(bulge, pads) == (False, "its rule names a neck-down, but the track leaves 1 mm + one 0.25 mm grid step of the net's pads")
    stub = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.6",
                      derived_from=["net:N", "escape:U1.3:straight:width=0.25:points=1.0,1.0;2.0,1.0;2.0,3.0"])
    assert _escape_segment(Track(net="N", layer="F.Cu", start=(1.0, 1.0), end=(2.0, 1.0), width_mm=0.25, provenance=stub)).startswith("escape stub of pad U1.3")
    assert _escape_segment(Track(net="N", layer="F.Cu", start=(1.0, 1.0), mid=(1.5, 0.8), end=(2.0, 1.0), width_mm=0.25, provenance=stub)) is None


def test_a_neck_downs_samples_follow_the_arcs_length_not_the_distance_between_its_ends(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    pads = net_pads(ir, lib).pads["LONG"]
    # a loop of radius 2 that starts 1.2 mm from R2's pad, runs more than 3 mm out of the rule's reach and ends 60 um from where it
    # started: 12.5 mm of copper between two ends that both lie within the reach
    loop = arc("LONG", (93.0, 5.0), (93.0, 1.0), (93.059991, 4.9991), width=0.25).model_copy(update={"provenance": NECK_PROV})
    assert track_length(loop) == pytest.approx(2.0 * (2.0 * math.pi - 0.03), abs=1e-5) and math.dist(loop.start, loop.end) < 0.06
    assert _neckdown(loop, pads) == (False, "its rule names a neck-down, but the track leaves 1 mm + one 0.25 mm grid step of the net's pads")


# --------------------------------------------------------------------------- straight runs and pair coupling


def test_a_straight_run_is_cut_at_an_arc():
    # a bump on a straight line: the chord of its arc lies on the line and touches both legs
    tracks = [track("N", (0.0, 0.0), (5.0, 0.0)), arc("N", (5.0, 0.0), (6.0, 1.0), (7.0, 0.0)), track("N", (7.0, 0.0), (12.0, 0.0)),
              track("N", (12.0, 0.0), (15.0, 0.0))]
    runs = _straight_runs(tracks, [0, 1, 2, 3])
    assert [(r.start, r.end, r.tracks, r.mid) for r in runs] == [((0.0, 0.0), (5.0, 0.0), (0,), None), ((5.0, 0.0), (7.0, 0.0), (1,), (6.0, 1.0)),
                                                                 ((7.0, 0.0), (15.0, 0.0), (2, 3), None)]
    tracks[1] = track("N", (5.0, 0.0), (7.0, 0.0))  # a straight track there instead: one run from end to end, as before
    (run,) = _straight_runs(tracks, [0, 1, 2, 3])
    assert (run.start, run.end, run.tracks, run.mid) == ((0.0, 0.0), (15.0, 0.0), (0, 1, 2, 3), None)
    # two arcs that meet (an S-curve): their chords are collinear and touch, and they stay two runs
    s_curve = [arc("N", (0.0, 0.0), (1.0 - S2, S2), (1.0, 1.0)), arc("N", (1.0, 1.0), (1.0 + S2, 2.0 - S2), (2.0, 2.0))]
    assert [(r.tracks, r.mid) for r in _straight_runs(s_curve, [0, 1])] == [((0,), s_curve[0].mid), ((1,), s_curve[1].mid)]


def _pair_copper(p: str = "P", n: str = "N", w: float = 0.3, s: float = 0.2, x0: float = 0.0) -> tuple[list[Track], list[Track]]:
    """A pair that runs 10 mm right, round two concentric quarter arcs (radius 2 and 2 + pitch) and 10 mm down."""
    pitch = w + s
    return ([track(p, (x0, 0.0), (x0 + 10.0, 0.0), width=w), quarter(p, x0 + 10.0, 2.0, 2.0, width=w), track(p, (x0 + 12.0, 2.0), (x0 + 12.0, 12.0), width=w)],
            [track(n, (x0, -pitch), (x0 + 10.0, -pitch), width=w), quarter(n, x0 + 10.0, 2.0, 2.0 + pitch, width=w),
             track(n, (x0 + 12.0 + pitch, 2.0), (x0 + 12.0 + pitch, 12.0), width=w)])


def test_an_arc_of_a_pair_is_uncoupled_copper_and_never_a_coupled_piece():
    tp, tn = _pair_copper()
    pieces = coupled_pieces(tp, tn)
    assert [(x["length_mm"], pytest.approx(x["gap_mm"], abs=1e-9)) for x in pieces] == [(10.0, 0.2), (10.0, 0.2)]  # the two straight stretches
    coupled, unc_p, unc_n, again = uncoupled_lengths(tp, tn)
    assert again == pieces and coupled == 20.0
    assert unc_p == pytest.approx(math.pi, abs=1e-9) and unc_n == pytest.approx(2.5 * math.pi / 2.0, abs=1e-9)  # each net's whole arc
    # read as their chords the two arcs would be a "coupled" piece at a gap that is not on the board (0.054 mm for 0.2 mm)
    chords = coupled_pieces([track("P", t.start, t.end, width=0.3) for t in tp], [track("N", t.start, t.end, width=0.3) for t in tn])
    assert len(chords) == 3 and min(x["gap_mm"] for x in chords) == pytest.approx(0.5 * S2 - 0.3, abs=1e-9)
    assert coupled_pieces([tp[1]], tn) == [] and coupled_pieces(tp, [tn[1]]) == []
    assert uncoupled_lengths([tp[1]], [tn[1]]) == (0.0, pytest.approx(math.pi, abs=1e-12), pytest.approx(1.25 * math.pi, abs=1e-12), [])


def test_an_arc_beside_a_straight_track_of_the_other_net_is_no_coupled_piece_from_either_side():
    # a shallow bump of N beside a straight stretch of P: its chord (2, 0.5)-(8, 0.5) runs parallel to P at the coupling gap, the arc does not
    p = track("P", (0.0, 0.0), (10.0, 0.0), width=0.3)
    bump = arc("N", (2.0, 0.5), (5.0, 0.9), (8.0, 0.5), width=0.3)
    chord = track("N", (2.0, 0.5), (8.0, 0.5), width=0.3)
    piece = {"layer": "F.Cu", "length_mm": 6.0, "width_mm": 0.3, "gap_mm": pytest.approx(0.2, abs=1e-12)}
    assert coupled_pieces([p], [chord]) == [piece] and coupled_pieces([chord], [p]) == [piece]  # what the chord would be read as
    assert coupled_pieces([p], [bump]) == [] and coupled_pieces([bump], [p]) == []  # the arc as N's copper, and as P's
    bent = track_length(bump)
    assert bent > 6.07 and uncoupled_lengths([p], [bump]) == (0.0, 10.0, bent, []) and uncoupled_lengths([bump], [p]) == (0.0, bent, 10.0, [])


def test_si_diff_counts_a_pairs_arcs_as_uncoupled_and_says_so(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    w, s = 0.35, 0.2
    z = edge_coupled_microstrip(w, s, 0.2, 35.0, 4.5).z_diff
    ir.si = si_of(*default_classes(), cls("DIFF", nets=["LONG", "SHORT"], pairs=[DiffPair(p="LONG", n="SHORT")], target_zdiff_ohm=u(round(z), "ohm"), zdiff_tol_rel=u(0.1),
                                          pair_uncoupled_max_mm=u(3.5, "mm"), pair_max_skew_s=u(10e-12, "s")))
    tp, tn = _pair_copper("LONG", "SHORT", w, s, x0=10.0)
    ir.pcb.tracks = [*tp, *tn]
    d = results(ir)["si.diff.LONG/SHORT"]
    arc_p, arc_n = math.pi, (2.0 + w + s) * math.pi / 2.0
    judged = [r for r in d.details["rows"] if "z_diff_ohm" in r]
    assert d.tool_version == "0.3" and d.details["coupled_mm"] == 20.0 and [r["status"] for r in judged] == ["PASS", "PASS"]
    by_net = {r["net"]: r for r in d.details["rows"] if "net" in r}
    assert by_net["LONG"]["uncoupled_mm"] == by_net["LONG"]["curved_mm"] == round(arc_p, 6) and by_net["LONG"]["status"] == "PASS"
    assert by_net["LONG"]["reason"] == "LONG uncoupled 3.142 mm <= 3.5 mm (3.142 mm of it in arc track(s): curved copper is never counted as coupled)"
    assert by_net["SHORT"]["uncoupled_mm"] == by_net["SHORT"]["curved_mm"] == round(arc_n, 6) and by_net["SHORT"]["status"] == "FAIL"
    assert d.status is S.FAIL and "SHORT uncoupled 4.006 mm > 3.5 mm (4.006 mm of it in arc track(s)" in d.message
    # the two arcs alone: nothing straight runs side by side, and what the curved copper does is not judged
    ir.pcb.tracks = [tp[1], tn[1]]
    alone = results(ir)["si.diff.LONG/SHORT"].details["rows"][0]
    assert alone["status"] == "NOT_VERIFIED" and alone["reason"].startswith("no straight coupled section, and the pair holds arc track(s) (LONG 3.142 mm, SHORT 4.006 mm)")
    # without an arc the rows are the ones of before: no curved_mm, and a pair that never runs side by side fails
    ir.pcb.tracks = [tp[0], tn[2]]
    plain = results(ir)["si.diff.LONG/SHORT"]
    assert plain.tool_version == "0.1" and all("curved_mm" not in r for r in plain.details["rows"])
    assert plain.details["rows"][0] == {"status": "FAIL", "reason": "no coupled section: the two nets never run side by side"}


# --------------------------------------------------------------------------- the crosstalk neighbour search


def _nested() -> tuple[list[Track], list[Track]]:
    """A victim that runs 10 mm right, round a quarter arc of radius 2 and 10 mm down, and an aggressor nested outside it at one
    pitch (edge gap 0.3 mm = 1.5 h): straight beside straight, arc beside arc."""
    return ([track("V", (0.0, 0.0), (10.0, 0.0), width=W), quarter("V", 10.0, 2.0, 2.0, width=W), track("V", (12.0, 2.0), (12.0, 12.0), width=W)],
            [track("A", (0.0, -PITCH), (10.0, -PITCH), width=W), quarter("A", 10.0, 2.0, 2.0 + PITCH, width=W),
             track("A", (12.0 + PITCH, 2.0), (12.0 + PITCH, 12.0), width=W)])


def test_copper_within_the_window_of_an_arc_is_one_curved_record_per_pair_of_tracks_with_the_smallest_gap():
    victim, others = _nested()
    recs = parallel_runs(victim, others, {"F.Cu": H}, g_max=10.0)
    assert [(r["kind"], r["victim_index"], r["aggressor_index"]) for r in recs] == [
        ("parallel", 0, 0), ("non_parallel", 0, 1), ("non_parallel", 1, 0), ("non_parallel", 1, 1), ("non_parallel", 1, 2), ("non_parallel", 2, 1),
        ("parallel", 2, 2)]
    curved = [r for r in recs if r["kind"] == "non_parallel"]
    for r in curved:  # every pair that involves an arc: curved, never parallel, the gap never over-stated
        arcs = int(r["victim_index"] == 1) + int(r["aggressor_index"] == 1)
        assert set(r) == {"kind", "parallel", "gap_mm", "s_over_h", "curved", "net", "layer", "h_mm", "victim_index", "aggressor_index"}
        assert r["curved"] is True and r["parallel"] is False and (r["net"], r["layer"], r["h_mm"]) == ("A", "F.Cu", H) and r["s_over_h"] == r["gap_mm"] / H
        # the true edge gap is 0.3 mm in each pair (at the shared corner points, or between the concentric arcs): the record's
        # is the chord gap lowered by 1 um per arc - at most the true gap, and within twice the bound per arc of it
        assert GAP - 2.0 * arcs * ARC_SAGITTA_MM - 1e-9 <= r["gap_mm"] <= GAP + 1e-9
    assert [r["gap_mm"] for r in curved if (r["victim_index"], r["aggressor_index"]) == (1, 1)][0] <= GAP - 2.0 * ARC_SAGITTA_MM + 1e-9
    # the straight stretches give the records they give without the arcs (a track without length in each arc's place keeps the indices)
    plain = parallel_runs([victim[0], track("V", (10.0, 0.0), (10.0, 0.0), width=W), victim[2]],
                          [others[0], track("A", (10.0, -PITCH), (10.0, -PITCH), width=W), others[2]], {"F.Cu": H}, g_max=10.0)
    assert [r["kind"] for r in plain] == ["parallel", "parallel"] and [r for r in recs if r["kind"] == "parallel"] == plain
    assert all("curved" not in r for r in plain) and plain[0]["length_mm"] == plain[1]["length_mm"] == 10.0
    # beyond the window nothing: the same copper with a 0.25 mm window (the 0.3 mm gap lies outside, also for the arcs)
    assert parallel_runs(victim, others, {"F.Cu": 0.025}, g_max=10.0) == []
    # an arc of the victim's own net is skipped like any of its tracks; a neighbour without length is none
    assert parallel_runs([victim[1]], [victim[0], track("A", (11.0, 0.0), (11.0, 0.0), width=W)], {"F.Cu": H}, g_max=10.0) == []
    # one index for many victims gives the same records, however often it is asked
    index = NeighbourIndex([*victim, *others], {"F.Cu": H}, g_max=10.0)
    again = parallel_runs(victim, [*victim, *others], {"F.Cu": H}, g_max=10.0)
    assert index.runs(victim) == again == index.runs(victim) and [{**r, "aggressor_index": r["aggressor_index"] - 3} for r in again] == recs


def test_the_distance_of_two_chains_is_the_smallest_over_every_pair_of_their_chords():
    def brute(pa, pb) -> float:
        return min(_seg_dist(p0, p1, q0, q1) for p0, p1 in zip(pa, pa[1:]) for q0, q1 in zip(pb, pb[1:]))

    inner, outer = track_points(quarter("A", 10.0, 2.0, 2.0)), track_points(quarter("B", 10.0, 2.0, 2.0 + PITCH))
    half = track_points(arc("C", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0)))
    line, through, far = [(3.0, -5.6), (7.0, -5.6)], [(5.0, -7.0), (5.0, -3.0)], [(40.0, 40.0), (41.0, 40.0)]
    crossing = track_points(arc("D", (10.5, 1.5), (11.6, 0.6), (12.5, -0.5)))  # from inside the inner quarter arc's circle out through it
    for pa, pb in ((inner, outer), (outer, inner), (half, line), (line, half), (half, far), (outer, half), (inner, line), (line, far)):
        assert 0.0 < _chain_dist(pa, pb) == pytest.approx(brute(pa, pb), abs=1e-12)
    assert PITCH - ARC_SAGITTA_MM <= _chain_dist(inner, outer) <= PITCH  # concentric: one pitch apart, the outer chords at most 1 um nearer
    # chords that cross, and two chains that share an end point (the half circle ends where the inner quarter arc starts): no distance
    for pa, pb in ((half, through), (through, half), (inner, crossing), (crossing, inner), (inner, half)):
        assert _chain_dist(pa, pb) == brute(pa, pb) == 0.0


def test_an_arc_is_found_by_its_bulge_and_touches_only_when_it_surely_does():
    # a half circle from (0, 0) over (5, -5) to (10, 0): the box of its two ends is the line y = 0, its copper reaches y = -5
    half = arc("A", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0), width=0.2)
    above = track("V", (3.0, -5.6), (7.0, -5.6), width=0.2)  # 0.6 mm above the top of the bulge: an edge gap of 0.4 mm = 2 h
    (rec,) = parallel_runs([above], [half], {"F.Cu": H}, g_max=10.0)
    assert rec["kind"] == "non_parallel" and rec["curved"] is True and 0.4 - 2.0 * ARC_SAGITTA_MM <= rec["gap_mm"] <= 0.4
    (back,) = parallel_runs([half], [above], {"F.Cu": H}, g_max=10.0)  # and the other way round: the arc as the victim
    assert back["kind"] == "non_parallel" and back["curved"] is True and back["gap_mm"] == pytest.approx(rec["gap_mm"], abs=1e-12)
    # no broadside record from an arc: only an exactly parallel overlap on the other layer is one
    below = track("V", (3.0, -5.0), (7.0, -5.0), width=0.2, layer="B.Cu")
    assert parallel_runs([below], [half], {}, g_max=10.0, adjacent={"B.Cu": {"F.Cu": 1.0}}) == []
    assert parallel_runs([half], [below], {}, g_max=10.0, adjacent={"F.Cu": {"B.Cu": 1.0}}) == []
    # a track through the arc: contact; one whose edge is 0.5 um from the arc's (inside the 1 um bound): not surely touching
    (cross,) = parallel_runs([track("V", (5.0, -7.0), (5.0, -3.0), width=0.2)], [half], {"F.Cu": H}, g_max=10.0)
    assert cross["kind"] == "contact" and cross["curved"] is True and cross["gap_mm"] <= -0.2
    (close,) = parallel_runs([track("V", (3.0, -5.2005), (7.0, -5.2005), width=0.2)], [half], {"F.Cu": H}, g_max=10.0)
    assert close["kind"] == "non_parallel" and close["curved"] is True and -ARC_SAGITTA_MM <= close["gap_mm"] <= 0.0005
    # a zone beside an arc victim: measured from the arc, not from the line between its ends (5.5 mm from it, beyond the window)
    zone = Zone(net="Z", layer="F.Cu", polygon=[(4.0, -6.5), (6.0, -6.5), (6.0, -5.5), (4.0, -5.5)])
    (z,) = parallel_runs([half], [], {"F.Cu": H}, g_max=10.0, zones=[zone])
    assert z["kind"] == "zone" and z["curved"] is True and 0.4 - ARC_SAGITTA_MM - 1e-9 <= z["gap_mm"] <= 0.4


def test_a_curved_neighbour_touches_only_when_the_chords_overlap_by_more_than_the_bound():
    half = arc("A", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0), width=0.2)
    top = min(y for _, y in track_points(half))  # the highest point of the arc's chords (Y down): at most 1 um below the arc's top, y = -5
    assert -5.0 <= top <= -5.0 + ARC_SAGITTA_MM

    def beside(clear: float) -> dict:
        """The record of a 0.2 mm track above the half circle whose edge is ``clear`` from the edge of the arc's chords (negative: they overlap)."""
        y = top - 0.2 - clear
        (rec,) = parallel_runs([track("V", (3.0, y), (7.0, y), width=0.2)], [half], {"F.Cu": H}, g_max=10.0)
        assert rec["curved"] is True and rec["gap_mm"] == pytest.approx(clear - ARC_SAGITTA_MM, abs=1e-9)
        return rec

    # the chords clear the track by half a micron: within the bound the arc may reach it - named with a gap below zero, not a contact
    assert beside(0.0005)["kind"] == "non_parallel"
    # the chords overlap it by half a micron, still within the bound: by the chain's two-sided bound alone a touch is not sure
    assert beside(-0.0005)["kind"] == "non_parallel"
    # the chords overlap it by more than the bound: the copper touches wherever the arc runs within the bound of its chords
    assert beside(-0.0015)["kind"] == "contact"


def test_a_curved_neighbour_is_found_at_the_windows_edge_with_both_widths():
    # a 0.6 mm quarter arc whose right end is 2.25 mm, centre to centre, from a 0.2 mm track: an edge gap of 1.85 mm, inside the 2 mm
    # window (10 h) - though the centrelines are farther apart than the window is wide, and in cells of the search grid that do not touch
    wide = quarter("V", 0.4, 5.0, 1.5, width=0.6)
    thin = track("A", (4.15, 4.5), (4.15, 5.5), width=0.2)
    (rec,) = parallel_runs([wide], [thin], {"F.Cu": H}, g_max=10.0)
    (back,) = parallel_runs([thin], [wide], {"F.Cu": H}, g_max=10.0)
    assert rec["kind"] == back["kind"] == "non_parallel" and rec["curved"] is True and back["curved"] is True
    assert rec["gap_mm"] == pytest.approx(2.25 - (0.6 + 0.2) / 2.0 - ARC_SAGITTA_MM, abs=1e-9) and back["gap_mm"] == pytest.approx(rec["gap_mm"], abs=1e-12)
    # the window is judged on the gap lowered by the bound (the reporting side): chords half a micron beyond its edge are inside it,
    # chords a micron and a half beyond it are not - seen from either track
    half = arc("A", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0), width=0.2)
    top = min(y for _, y in track_points(half))
    for beyond, found in ((0.0005, 1), (0.0015, 0)):
        y = top - 0.2 - 10.0 * H - beyond
        above = track("V", (3.0, y), (7.0, y), width=0.2)
        recs = parallel_runs([above], [half], {"F.Cu": H}, g_max=10.0)
        assert len(recs) == len(parallel_runs([half], [above], {"F.Cu": H}, g_max=10.0)) == found
        assert all(r["gap_mm"] == pytest.approx(10.0 * H + beyond - ARC_SAGITTA_MM, abs=1e-9) for r in recs)


def test_an_arc_on_an_unsearched_layer_and_zones_that_are_no_neighbours_give_no_record():
    half = arc("A", (0.0, 0.0), (5.0, -5.0), (10.0, 0.0), width=0.2)
    above = track("V", (3.0, -5.6), (7.0, -5.6), width=0.2)
    assert len(parallel_runs([half], [above], {"F.Cu": H}, g_max=10.0)) == len(parallel_runs([above], [half], {"F.Cu": H}, g_max=10.0)) == 1
    # a layer the window does not name is not searched - also when its copper is indexed as the broadside side of a searched layer:
    # no record for an arc victim on it, none for an arc beside a victim on it
    for adjacent in (None, {"B.Cu": {"F.Cu": 1.0}}):
        assert parallel_runs([half], [above], {"B.Cu": H}, g_max=10.0, adjacent=adjacent) == []
        assert parallel_runs([above], [half], {"B.Cu": H}, g_max=10.0, adjacent=adjacent) == []
    # a zone of the arc's own net and a zone on another layer are no neighbours of an arc victim (as of a straight one)
    area = [(4.0, -6.5), (6.0, -6.5), (6.0, -5.5), (4.0, -5.5)]
    assert parallel_runs([half], [], {"F.Cu": H}, g_max=10.0, zones=[Zone(net="A", layer="F.Cu", polygon=area), Zone(net="Z", layer="B.Cu", polygon=area)]) == []


#: the neighbour records of a small board without an arc, as the search gave them before it read ``Track.mid``: a parallel run, a
#: non-parallel and a touching neighbour, an in-line one, a zone, a broadside overlap, and the same from the second victim track
PLAIN_RECORDS = [
    {"kind": "parallel", "parallel": True, "length_mm": 6.0, "victim_width_mm": 0.3, "aggressor_width_mm": 0.3, "gap_mm": 0.5, "s_over_h": 2.5,
     "victim_span": [(4.0, 0.0), (10.0, 0.0)], "aggressor_span": [(4.0, 0.8), (10.0, 0.8)], "net": "A", "layer": "F.Cu", "h_mm": 0.2, "victim_index": 0,
     "aggressor_index": 0},
    {"kind": "non_parallel", "parallel": False, "gap_mm": 1.1642135623730951, "s_over_h": 5.8210678118654755, "net": "C", "layer": "F.Cu", "h_mm": 0.2,
     "victim_index": 0, "aggressor_index": 3},
    {"kind": "contact", "parallel": False, "gap_mm": -0.25, "s_over_h": -1.25, "net": "D", "layer": "F.Cu", "h_mm": 0.2, "victim_index": 0, "aggressor_index": 4},
    {"kind": "non_parallel", "parallel": True, "gap_mm": 0.7, "s_over_h": 3.4999999999999996, "net": "E", "layer": "F.Cu", "h_mm": 0.2, "victim_index": 0,
     "aggressor_index": 5},
    {"kind": "zone", "parallel": False, "net": "Z", "layer": "F.Cu", "gap_mm": 0.85, "s_over_h": 4.25, "h_mm": 0.2, "victim_index": 0, "zone_index": 0},
    {"kind": "broadside", "parallel": True, "length_mm": 1.0, "lateral_gap_mm": 0.29999999999999993, "net": "B", "layer": "B.Cu", "victim_layer": "F.Cu",
     "h_mm": 1.0, "victim_index": 0, "aggressor_index": 2},
    {"kind": "contact", "parallel": False, "gap_mm": -0.3, "s_over_h": -1.4999999999999998, "net": "A", "layer": "F.Cu", "h_mm": 0.2, "victim_index": 1,
     "aggressor_index": 0},
    {"kind": "non_parallel", "parallel": False, "gap_mm": 1.1642135623730951, "s_over_h": 5.8210678118654755, "net": "C", "layer": "F.Cu", "h_mm": 0.2,
     "victim_index": 1, "aggressor_index": 3},
    {"kind": "non_parallel", "parallel": False, "gap_mm": 0.7, "s_over_h": 3.4999999999999996, "net": "E", "layer": "F.Cu", "h_mm": 0.2, "victim_index": 1,
     "aggressor_index": 5},
]


def test_a_board_without_an_arc_gives_the_records_and_lengths_it_gave_before():
    victim = [track("V", (0.0, 0.0), (10.0, 0.0), width=0.3), track("V", (10.0, 0.0), (10.0, 4.0), width=0.3)]
    others = [track("A", (4.0, 0.8), (14.0, 0.8), width=0.3), track("V", (0.0, 0.6), (10.0, 0.6), width=0.3),
              track("B", (2.0, -0.7), (3.0, -0.7), width=0.5, layer="B.Cu"), track("C", (11.0, -1.0), (12.0, -2.0), width=0.2),
              track("D", (5.0, -0.2), (5.0, 0.2), width=0.2), track("E", (11.0, 0.0), (13.0, 0.0), width=0.3)]
    zones = [Zone(net="Z", layer="F.Cu", polygon=[(1.0, -2.0), (3.0, -2.0), (3.0, -1.0), (1.0, -1.0)])]
    assert parallel_runs(victim, others, {"F.Cu": 0.2}, g_max=10.0, adjacent={"F.Cu": {"B.Cu": 1.0}}, zones=zones) == PLAIN_RECORDS
    tp = [track("P", (0.0, 0.0), (10.0, 0.0), width=0.3), track("P", (10.0, 0.0), (11.0, 1.0), width=0.3)]
    tn = [track("N", (0.0, 0.5), (10.5, 0.5), width=0.3), track("N", (10.5, 0.5), (11.5, 1.5), width=0.3)]
    piece = {"layer": "F.Cu", "length_mm": 10.0, "width_mm": 0.3, "gap_mm": 0.2}
    assert coupled_pieces(tp, tn) == [piece] and uncoupled_lengths(tp, tn) == (10.0, 1.4142135623730958, 1.9142135623730958, [piece])


# --------------------------------------------------------------------------- si.crosstalk


def _crosstalk_board(tmp: Path, lib, short_copper: list[Track]) -> CircuitIR:
    """``LONG`` runs 20 mm right, round a quarter arc of radius 2 and 13 mm down to R2; ``SHORT``'s copper is the one given."""
    parts = [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", 27.0, 20.0), ("R3", "PAD1", 5.0, 1.0), ("R4", "PAD1", 34.0, 20.0)]
    ir = board_ir(tmp, lib, parts, {"LONG": [("R1", "1"), ("R2", "1")], "SHORT": [("R3", "1"), ("R4", "1")]}, (40.0, 25.0))
    ir.component("R1").pins[0].electrical_type = PinElectricalType.OUTPUT
    ir.component("R2").pins[0].electrical_type = PinElectricalType.INPUT
    stacked(ir, 4)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (25.0, 5.0), width=W), quarter("LONG", 25.0, 7.0, 2.0, width=W), track("LONG", (27.0, 7.0), (27.0, 20.0), width=W),
                     *short_copper]
    return ir


def test_si_crosstalk_names_a_curved_neighbour_and_computes_none(tmp_path: Path, lib):
    nested = [track("SHORT", (10.0, 5.0 - PITCH), (25.0, 5.0 - PITCH), width=W), quarter("SHORT", 25.0, 7.0, 2.0 + PITCH, width=W),
              track("SHORT", (27.0 + PITCH, 7.0), (27.0 + PITCH, 15.0), width=W)]
    ir = _crosstalk_board(tmp_path, lib, nested)
    r = crosstalk_result(ir, ir.si)
    long = rows(r)["LONG"]
    assert (r.tool, r.tool_version, r.status) == ("si.crosstalk", "0.3", S.NOT_VERIFIED) and long["status"] == S.NOT_VERIFIED.value
    assert long["reason"] == XT_CURVED and "curved neighbour not computed (an arc track" in XT_CURVED and long["judged_by"] is None
    (item,) = long["not_computed"]  # five pairs of tracks involve an arc: one piece each, the smallest gap named
    assert (item["kind"], item["layer"], item["reason"], item["nets"], item["pieces"]) == ("non_parallel", "F.Cu", XT_CURVED, ["SHORT"], 5)
    assert GAP - 4.0 * ARC_SAGITTA_MM <= item["min_gap_mm"] <= GAP - 2.0 * ARC_SAGITTA_MM
    # the straight stretches are still computed, the arcs in none of them: 15 mm and 8 mm at the 0.3 mm gap
    assert [(a["net"], a["length_mm"], a["gap_mm"]) for a in long["aggressors"]] == [("SHORT", 15.0, pytest.approx(GAP, abs=1e-9)), ("SHORT", 8.0, pytest.approx(GAP, abs=1e-9))]
    assert rows(r)["SHORT"]["reason"] == XT_CURVED and "2 with neighbours not computed (named per row)" in r.message


def test_an_arc_beyond_the_window_leaves_the_row_what_it_was(tmp_path: Path, lib):
    beside = [track("SHORT", (10.0, 5.0 - PITCH), (20.0, 5.0 - PITCH), width=W)]  # ends 5 mm before LONG's corner: 25 h
    with_arc = _crosstalk_board(tmp_path / "a", lib, beside)
    chamfer = _crosstalk_board(tmp_path / "b", lib, beside)
    chamfer.pcb.tracks[1] = track("LONG", (25.0, 5.0), (27.0, 7.0), width=W)  # the corner as a straight track: a board without an arc
    for library in (None, lib):  # without the pad boxes (no path) and with them (LONG's path runs through its arc)
        a = crosstalk_result(with_arc, with_arc.si, measure_nets(with_arc, library=library))
        b = crosstalk_result(chamfer, chamfer.si, measure_nets(chamfer, library=library))
        assert (a.tool_version, b.tool_version) == ("0.3", "0.2") and a.status is b.status and a.message == b.message
        assert a.details == b.details and rows(a)["LONG"]["not_computed"] == []
        assert [x["length_mm"] for x in rows(a)["LONG"]["aggressors"]] == [10.0]


# --------------------------------------------------------------------------- spice.xtalk


def _xtalk_board(tmp: Path, lib) -> CircuitIR:
    """``VIC``: R1 (10, 10) - 20 mm right - a quarter arc of radius 3 - 27 mm down - R2 (33, 40). ``AGG`` comes up on B.Cu, runs
    beside VIC's second leg on F.Cu from y 20 to y 35 at an edge gap of 1.5 h and goes down again through two vias."""
    xa = 33.0 + PITCH
    parts = [("R1", "PAD1", 10.0, 10.0), ("R2", "PAD1", 33.0, 40.0), ("R3", "PAD1", 45.0, 20.0), ("R4", "PAD1", 45.0, 35.0)]
    ir = board_ir(tmp, lib, parts, {"VIC": [("R1", "1"), ("R2", "1")], "AGG": [("R3", "1"), ("R4", "1")]}, (55.0, 50.0))
    stacked(ir, 4)
    for ref, kind in (("R1", PinElectricalType.OUTPUT), ("R3", PinElectricalType.OUTPUT), ("R2", PinElectricalType.INPUT), ("R4", PinElectricalType.INPUT)):
        ir.component(ref).pins[0].electrical_type = kind
    ir.si = si_of(*default_classes(xtalk_budget_rel=user_requirement(0.05)))
    ir.pcb.tracks = [track("VIC", (10.0, 10.0), (30.0, 10.0), width=W), quarter("VIC", 30.0, 13.0, 3.0, width=W), track("VIC", (33.0, 13.0), (33.0, 40.0), width=W),
                     track("AGG", (45.0, 20.0), (xa, 20.0), width=W, layer="B.Cu"), track("AGG", (xa, 20.0), (xa, 35.0), width=W),
                     track("AGG", (xa, 35.0), (45.0, 35.0), width=W, layer="B.Cu")]
    ir.pcb.vias = [Via(net="AGG", x_mm=xa, y_mm=20.0, diameter_mm=0.6, drill_mm=0.3), Via(net="AGG", x_mm=xa, y_mm=35.0, diameter_mm=0.6, drill_mm=0.3)]
    return ir


def test_a_span_is_placed_along_a_path_with_an_arc_at_the_exact_distances(tmp_path: Path, lib):
    from ai_eda.tools.spice.si_check import deck_stem
    from ai_eda.tools.spice.xtalk_check import XT_DECK_ARC_VERSION, XT_DECK_VERSION, _locate, deck_version, spice_xtalk_results, xtalk_deck
    from tests.test_spice_xtalk import Stub

    ir = _xtalk_board(tmp_path, lib)
    bend = 1.5 * math.pi
    measures = measure_nets(ir, library=lib)
    path = measures["VIC"].path
    assert path.length_mm == pytest.approx(47.0 + bend, abs=1e-9) and [e.mid is not None for e in path.edges] == [False, True, False]
    scan = crosstalk_scan(ir, ir.si, measures)
    vic = rows(scan.result)["VIC"]
    assert scan.result.tool_version == "0.3" and vic["status"] == S.NOT_APPLICABLE.value and vic["judged_by"] == "spice.xtalk.VIC" and vic["not_computed"] == []
    plan = scan.plans["VIC"]
    (span,) = plan.spans
    # the span y 20..35 of the second leg lies 20 mm, the arc and 7 mm from R1 - by the chord it would be 0.47 mm nearer
    assert span.v_pos == pytest.approx((27.0 + bend, 42.0 + bend), abs=1e-9) and span.length_mm == pytest.approx(15.0, abs=1e-9)
    d = xtalk_deck("xt_" + deck_stem("VIC"), victim=plan.victim, aggressor=plan.aggressors["AGG"], span=span, v_drive=0, a_drive=0, t_r=1e-9,
                   r_v=25.0, c_v=5e-12, r_a=25.0, c_a=5e-12)
    t_pd = math.sqrt(microstrip(W, 0.2, 35.0, 4.5).e_eff) / C0
    assert d.info["lines"]["TV0"]["length_mm"] == round(27.0 + bend, 6) and d.info["lines"]["TV0"]["td_s"] == pytest.approx((27.0 + bend) / 1000.0 * t_pd, rel=1e-12)
    # a point of the arc is located on the arc piece, at its distance along the arc; a point of the chord lies on no copper
    on_arc = (30.0 + 3.0 * S2, 13.0 - 3.0 * S2)
    assert _locate(path, "F.Cu", on_arc) == pytest.approx(20.0 + bend / 2.0, abs=1e-9) and _locate(path, "F.Cu", (31.5, 11.5)) is None
    assert _locate(path, "F.Cu", (30.0, 10.0)) == pytest.approx(20.0, abs=1e-12) and _locate(path, "F.Cu", (33.0, 13.0)) == pytest.approx(20.0 + bend, abs=1e-9)
    # the results of a board with an arc carry the arc deck version; the same board with a chamfer in its place the one of before
    assert (XT_DECK_VERSION, XT_DECK_ARC_VERSION, deck_version(ir)) == ("0.1", "0.2", "0.2")
    stub = Stub({("VIC", "AGG"): 0.02, ("AGG", "VIC"): 0.02})
    res = {r.check_id: r for r in spice_xtalk_results(ir, {"kicad_library": lib, "spice": stub}, tmp_path / "w")}
    assert res["spice.xtalk.VIC"].status is S.PASS and {r.details["deck_version"] for r in res.values()} == {"0.2"}
    ir.pcb.tracks[1] = track("VIC", (30.0, 10.0), (33.0, 13.0), width=W)
    res = {r.check_id: r for r in spice_xtalk_results(ir, {"kicad_library": lib, "spice": stub}, tmp_path / "w2")}
    assert deck_version(ir) == "0.1" and res["spice.xtalk.VIC"].status is S.PASS and {r.details["deck_version"] for r in res.values()} == {"0.1"}


def test_a_coupled_piece_on_an_arc_is_refused_by_name(tmp_path: Path, lib):
    from ai_eda.tools.spice.xtalk_check import CURVED_SPAN, XtPiece, plan_victim

    ir = _xtalk_board(tmp_path, lib)
    measures = measure_nets(ir, library=lib)
    (real,) = crosstalk_scan(ir, ir.si, measures).pieces["VIC"]

    def piece(a, b, length: float) -> XtPiece:
        return XtPiece(aggressor="AGG", layer="F.Cu", width_mm=W, gap_mm=GAP, length_mm=length, victim_span=(a, b), aggressor_span=real.aggressor_span,
                       z_even=real.z_even, z_odd=real.z_odd, e_eff_even=real.e_eff_even, e_eff_odd=real.e_eff_odd)

    refusal = f"a coupled piece beside AGG on F.Cu runs over an arc track of VIC (its path R1.1-R2.1): {CURVED_SPAN}"
    assert CURVED_SPAN == "the coupled-line deck models a straight span only, curved copper is not computed"
    assert plan_victim(ir, "VIC", [piece((30.0, 10.0), (33.0, 13.0), 3.0 * math.sqrt(2.0))], measures) == refusal  # the arc from end to end
    assert plan_victim(ir, "VIC", [piece((25.0, 10.0), (33.0, 20.0), 12.0)], measures) == refusal  # across the corner
    # a piece on the chord is on no copper of the path
    assert plan_victim(ir, "VIC", [piece((31.5, 11.5), (33.0, 13.0), 1.5 * math.sqrt(2.0))], measures) == (
        "a coupled piece beside AGG on F.Cu lies on copper of VIC off its pad-to-pad path R1.1-R2.1")
    assert not isinstance(plan_victim(ir, "VIC", [real], measures), str)  # the real piece, on the straight leg, is planned


def test_a_piece_that_reaches_into_an_arc_of_either_path_is_refused_and_one_that_ends_on_it_is_planned(tmp_path: Path, lib):
    from ai_eda.tools.spice.xtalk_check import CURVED_SPAN, XtPiece, plan_victim

    ir = _xtalk_board(tmp_path, lib)
    xa = 33.0 + PITCH
    # AGG now stays on F.Cu after its leg beside VIC: 13 mm down from (xa, 20), round a quarter arc of radius 2 and on to R4 at (45, 35)
    ir.pcb.tracks[4:] = [track("AGG", (xa, 20.0), (xa, 33.0), width=W), arc("AGG", (xa, 33.0), (xa + 2.0 - 2.0 * S2, 33.0 + 2.0 * S2), (xa + 2.0, 35.0), width=W),
                         track("AGG", (xa + 2.0, 35.0), (45.0, 35.0), width=W)]
    ir.pcb.vias = ir.pcb.vias[:1]
    measures = measure_nets(ir, library=lib)
    agg = measures["AGG"].path
    assert agg.ends == ("R3.1", "R4.1") and [e.layer for e in agg.edges] == ["B.Cu", None, "F.Cu", "F.Cu", "F.Cu"]
    assert [e.mid is not None for e in agg.edges] == [False, False, False, True, False] and agg.edges[3].length_mm == pytest.approx(math.pi, abs=1e-9)
    (real,) = crosstalk_scan(ir, ir.si, measures).pieces["VIC"]  # VIC's leg beside AGG's, y 20..33

    def piece(v_span, a_span, length: float) -> XtPiece:
        return XtPiece(aggressor="AGG", layer="F.Cu", width_mm=W, gap_mm=GAP, length_mm=length, victim_span=v_span, aggressor_span=a_span,
                       z_even=real.z_even, z_odd=real.z_odd, e_eff_even=real.e_eff_even, e_eff_odd=real.e_eff_odd)

    assert real.length_mm == pytest.approx(13.0, abs=1e-9) and not isinstance(plan_victim(ir, "VIC", [real], measures), str)
    # the aggressor's side of a piece runs round AGG's arc: refused, naming that net and its path
    across = piece(real.victim_span, ((xa, 30.0), (xa + 5.0, 35.0)), 13.0)
    assert plan_victim(ir, "VIC", [across], measures) == f"a coupled piece beside AGG on F.Cu runs over an arc track of AGG (its path R3.1-R4.1): {CURVED_SPAN}"
    # the victim's side starts half a millimetre before the end of VIC's arc: refused, that much of it is curved copper
    bend = ir.pcb.tracks[1]
    on_arc = arc_point_at(bend.start, bend.mid, bend.end, 1.0 - 0.5 / (1.5 * math.pi))
    into = piece((on_arc, (33.0, 20.0)), ((xa, 20.0), (xa, 27.5)), 7.5)
    assert plan_victim(ir, "VIC", [into], measures) == f"a coupled piece beside AGG on F.Cu runs over an arc track of VIC (its path R1.1-R2.1): {CURVED_SPAN}"
    # a piece that starts on the very end of VIC's arc lies on its straight leg, and its other side on AGG's last stretch, past
    # AGG's arc: planned, each side at its distance along the copper - the arcs at their arc lengths
    onto = piece(((33.0, 13.0), (33.0, 20.0)), ((xa + 3.0, 35.0), (xa + 10.0, 35.0)), 7.0)
    plan = plan_victim(ir, "VIC", [onto], measures)
    assert not isinstance(plan, str)
    (span,) = plan.spans
    before_arc = agg.edges[0].length_mm + agg.edges[1].length_mm + 13.0  # the B.Cu stretch, the via's barrel and the leg
    assert agg.edges[0].length_mm == pytest.approx(45.0 - xa, abs=1e-9) and span.v_pos == pytest.approx((20.0 + 1.5 * math.pi, 27.0 + 1.5 * math.pi), abs=1e-9)
    assert span.a_pos == pytest.approx((before_arc + math.pi + 1.0, before_arc + math.pi + 8.0), abs=1e-9)
