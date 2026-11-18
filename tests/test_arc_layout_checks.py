"""Arc tracks (``Track.mid``) in the IR geometry checks: ``pcb.routing.connectivity`` / ``pcb.routing.clearance`` and ``pcb.keepout``.

A track with ``mid`` is a circular arc, and the checks measure it as one - never as the straight chord between its ends. Boards are
hand-built on the synthetic fixture library of ``tests/test_routing.py``, most of them around one corner arc: a quarter of the circle of
radius 2 mm around (7, 5), from (7, 3) round the corner (9, 3) to (9, 5). Its bulge over the chord (7, 3)-(9, 5) is
2 (1 - cos 45 deg) = 0.586 mm, more than a track width, so each case is also run with the arc replaced by that chord (:func:`_as_chord`)
to show the chord reading gets it wrong:

* connectivity: a net routed through an arc is one copper set; a track that ends on the arc between two points of its chain, a via and a
  pad centred on it are joined (measured against the arc exactly), and so are a via the arc ends on, a pad only the bulge reaches and a
  track end from outside the bend; two tracks within the arc tolerance of touching leave the net NOT_VERIFIED naming them, but only
  where the net's row depends on them;
* clearance: foreign copper under the bulge is a violation the chord would miss, foreign copper inside the bend is clear where the chord
  would fail; a pair whose distance lies within the arc tolerance of the limit is a NOT_VERIFIED row, never PASS or FAIL; a via, a
  custom pad's bounding disc and its known copper are measured exactly (known copper inside the limit is a violation, as for a
  straight track), and the prefilter uses the arc's own box;
* the outline and track keep-outs: an arc that bulges out of the outline or into a keep-out is reported; copper boxes that are apart
  decide exactly, where the chords could not;
* an arc whose three points are no arc is a FAIL row like every malformed track, never a crash;
* versions: 0.3 on a board with an arc (also beside a typed via), every other board exactly as before - one small straight-track board is
  compared with the results the checks gave before they knew arcs, and the frozen routed boards of ``tests/data/layout_baselines`` with
  digests of theirs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from ai_eda.ir import (
    CircuitIR,
    Keepout,
    ManufacturingConstraints,
    PinRef,
    Provenance,
    ProvenanceKind,
    Track,
    ValidationStatus as S,
    Via,
    assumption,
    user_requirement,
)
from ai_eda.ir.arc import ARC_SAGITTA_MM, arc_points
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import (
    ARC_MODEL,
    ARC_TOOL_VERSION,
    CLEARANCE_CHECK,
    CONNECTIVITY_CHECK,
    KEEPOUT_CHECK,
    KEEPOUT_TOOL_VERSION,
    SILK_TOOL_ID,
    SILK_TOOL_VERSION,
    TOOL_ID,
    TOOL_VERSION,
    TYPED_VIA_TOOL_VERSION,
    RoutingValidator,
    _Board,
    _connected,
    _copper_distance,
    _seg_point_distance,
    clearance_rows,
)
from tests.layout_baseline_data import load_geometry
from tests.test_layout_validation import _two_pads
from tests.test_routing import board_ir, fixture_library
from tests.test_routing_custom_pads import _library as custom_pad_library

NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
W = 0.4
LIMIT = 0.25
#: the corner arc: its centre and radius (module docstring)
CENTRE, RADIUS = (7.0, 5.0), 2.0


def _at(deg: float, r: float = RADIUS, centre: tuple[float, float] = CENTRE) -> tuple[float, float]:
    """The point at ``deg`` degrees (an atan2 angle, Y down) and distance ``r`` from ``centre``, at the file resolution (1 nm)."""
    a = math.radians(deg)
    return (round(centre[0] + r * math.cos(a), 6), round(centre[1] + r * math.sin(a), 6))


START, MID, END = _at(-90.0), _at(-45.0), _at(0.0)


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _track(net: str, a: tuple[float, float], b: tuple[float, float], layer: str = "F.Cu", w: float = W) -> Track:
    return Track(net=net, layer=layer, start=a, end=b, width_mm=w, provenance=NET_P)


def _arc(net: str, a: tuple[float, float], mid: tuple[float, float], b: tuple[float, float], layer: str = "F.Cu", w: float = W) -> Track:
    return Track(net=net, layer=layer, start=a, mid=mid, end=b, width_mm=w, provenance=NET_P)


def _limit(ir: CircuitIR, mm: float = LIMIT) -> None:
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(mm, note="any provenance counts: a limit is a limit"))


def _corner(tmp_path: Path, lib: KicadLibrary, parts: list | None = None, nets: dict | None = None) -> CircuitIR:
    """R1.1 at (3, 3) and R2.1 at (9, 9) on net N: a track to (7, 3), the corner arc, a track from (9, 5). 12 x 12 mm, limit 0.25 mm."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 9.0), *(parts or [])], {"N": [("R1", "1"), ("R2", "1")], **(nets or {})}, (12.0, 12.0))
    _limit(ir)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), START), _arc("N", START, MID, END), _track("N", END, (9.0, 9.0))]
    return ir


def _as_chord(ir: CircuitIR) -> CircuitIR:
    """A copy of ``ir`` with every arc replaced by the straight track between its ends: what a reader that ignored ``mid`` would see."""
    x = ir.model_copy(deep=True)
    x.pcb.tracks = [t if t.mid is None else _track(t.net, t.start, t.end, t.layer, t.width_mm) for t in x.pcb.tracks]
    return x


def _checks(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, version: str = ARC_TOOL_VERSION) -> dict:
    results = default_registry.get(TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert [r.check_id for r in results] == [CONNECTIVITY_CHECK, CLEARANCE_CHECK]
    for r in results:
        assert r.tool == TOOL_ID and r.tool_version == version and r.details["kind"] == "ir_geometry" and "not DRC" in r.message
    return {r.check_id: r for r in results}


def _keepout(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    [res] = default_registry.get(KEEPOUT_CHECK).validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    return res


def _outside(gap: float, w: float = W) -> Track:
    """A foreign track (net M) outside the bulge, across the radius at -45 degrees, its copper ``gap`` from the corner arc's copper."""
    m = _at(-45.0, RADIUS + W / 2.0 + w / 2.0 + gap)
    ux, uy = 0.3 * math.cos(math.radians(45.0)), 0.3 * math.sin(math.radians(45.0))
    return _track("M", (round(m[0] - ux, 6), round(m[1] - uy, 6)), (round(m[0] + ux, 6), round(m[1] + uy, 6)), w=w)


def _inside(gap: float, w: float = W) -> Track:
    """A foreign track (net M) inside the bend, along the radius at -45 degrees, its end ``gap`` from the corner arc's copper."""
    rho = RADIUS - W / 2.0 - w / 2.0 - gap
    return _track("M", _at(-45.0, rho), _at(-45.0, rho - 1.0), w=w)


def _foreign(tmp_path: Path, lib: KicadLibrary, track: Track | None = None) -> CircuitIR:
    """The corner board with net M (its pad R3.1 far away at (2, 10)) and one foreign track of M."""
    ir = _corner(tmp_path, lib, [("R3", "PAD1", 2.0, 10.0)], {"M": [("R3", "1")]})
    if track is not None:
        ir.pcb.tracks.append(track)
    return ir


# --------------------------------------------------------------------------- connectivity


def test_a_net_routed_through_an_arc_is_one_copper_set(tmp_path: Path, lib: KicadLibrary):
    assert (START, MID, END) == ((7.0, 3.0), (8.414214, 3.585786), (9.0, 5.0))
    ir = _corner(tmp_path, lib)
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.message == "1 net(s) connected through IR copper (3 track(s), 0 via(s)) (IR geometry check, not DRC)"
    assert c.details["nets"] == [{"net": "N", "pads": ["R1.1", "R2.1"], "tracks": 3, "vias": 0, "status": "PASS", "message": "2 pad(s) in one copper set"}]
    assert c.details["tracks"] == 3 and c.details["arcs"] == 1 and c.details["arc_model"] == ARC_MODEL  # one item per track; the model is named
    assert f"{ARC_SAGITTA_MM:g} mm" in ARC_MODEL and "a stated choice" in ARC_MODEL
    d = checks[CLEARANCE_CHECK]
    assert d.status is S.PASS and d.details["violations"] == [] and d.details["arcs"] == 1 and "arc_tolerance" not in d.details
    assert d.details["not_compared"] == ["pad-to-pad (footprint / placement geometry; DRC judges it)"]  # an arc leaves nothing uncompared
    # drawn from its other end the arc is the same copper
    ir.pcb.tracks[1] = _arc("N", END, MID, START)
    assert {k: v.status for k, v in _checks(ir, tmp_path, lib).items()} == {CONNECTIVITY_CHECK: S.PASS, CLEARANCE_CHECK: S.PASS}
    # the arc is what joins the two straight tracks: without it the net is open (and the board has no arc: stamped as before)
    ir.pcb.tracks.pop(1)
    c = _checks(ir, tmp_path, lib, TOOL_VERSION)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"] and "arcs" not in c.details and "arc_model" not in c.details


def test_two_arcs_that_share_an_end_and_an_arc_between_two_pads_are_joined(tmp_path: Path, lib: KicadLibrary):
    # a U of two quarter arcs (a half circle round (7, 5)) between two straight tracks: the arcs share the point (9, 5) exactly
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 3.0, 7.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 12.0))
    ir.pcb.tracks = [_track("N", (3.0, 3.0), START), _arc("N", START, MID, END), _arc("N", END, _at(45.0), _at(90.0)), _track("N", _at(90.0), (3.0, 7.0))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.details["arcs"] == 2 and c.details["nets"][0]["tracks"] == 4
    # one arc from pad centre to pad centre: its ends lie in the pads
    ir.pcb.tracks = [_arc("N", (3.0, 3.0), (5.0, 5.0), (3.0, 7.0))]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    # an arc that stops short of R2's copper (its end 1.2 mm from the pad centre, the pad's radius 0.8 + half a width 0.2) joins only R1
    ir.pcb.tracks = [_arc("N", (3.0, 3.0), (4.4, 4.4), (3.0, 5.8))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"]


def test_a_track_a_via_and_a_pad_on_the_arc_between_its_chain_points_are_joined(tmp_path: Path, lib: KicadLibrary):
    """A T junction on the arc, a via centred on it and a small pad centred on it, each away from the arc's ends and from the points of
    its chain: all are 0 mm from the arc and more than their reach from the chord (7, 3)-(9, 5), so only the arc reading joins them."""
    tee, via_at, pad_at = _at(-56.7), _at(-30.0), _at(-70.0)
    chain = arc_points(START, MID, END)
    assert len(chain) - 1 == 25 and all(min(math.dist(p, q) for q in chain) > 0.01 for p in (tee, via_at, pad_at))  # between chain points
    assert _seg_point_distance(START, END, tee) > W  # track - track: the two half widths
    assert _seg_point_distance(START, END, via_at) > W / 2.0 + 0.25  # track - via: half a width and the via's radius
    assert _seg_point_distance(START, END, pad_at) > W / 2.0 + 0.1  # track - pad: half a width and the pad's inscribed radius
    r3, r4 = _at(-56.7, 4.5), (10.5, 2.0)
    ir = _corner(tmp_path, lib, [("R3", "PAD1", *r3), ("R4", "PAD1", *r4), ("U1", "TINY", *pad_at)])
    ir.nets[0].pins += [PinRef(component_ref=ref, pin_number="1") for ref in ("R3", "R4", "U1")]
    ir.pcb.tracks += [_track("N", tee, r3), _track("N", via_at, r4, layer="B.Cu")]  # the T on F.Cu; the via's net goes on on B.Cu
    ir.pcb.vias = [Via(net="N", x_mm=via_at[0], y_mm=via_at[1], drill_mm=0.3, diameter_mm=0.5, provenance=NET_P)]
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.details["nets"][0]["message"] == "5 pad(s) in one copper set", c.message
    assert checks[CLEARANCE_CHECK].status is S.PASS
    # read as its chord the arc would leave all three apart
    c = _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R3.1", "R4.1", "U1.1"]
    # the via against the arc, asked either way round: the same exact distance (the via's centre lies on the arc, at the file resolution)
    board = _Board(ir, lib)
    arc, [via] = board.tracks[1], board.vias
    assert arc.mid == MID and _connected(arc, via) is True and _connected(via, arc) is True
    assert _copper_distance(via, arc) == _copper_distance(arc, via) == pytest.approx(-(W / 2.0 + 0.25), abs=2e-6)


def test_an_arc_that_ends_on_a_via_and_a_pad_only_its_bulge_reaches_are_joined(tmp_path: Path, lib: KicadLibrary):
    # the arc ends on a via's centre and the net goes on on B.Cu from there: the via is what joins the two layers
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 11.0, 9.0)], {"N": [("R1", "1"), ("R2", "1")]}, (14.0, 12.0))
    ir.pcb.tracks = [_track("N", (3.0, 3.0), START), _arc("N", START, MID, END), _track("N", END, (11.0, 9.0), layer="B.Cu")]
    ir.pcb.vias = [Via(net="N", x_mm=END[0], y_mm=END[1], drill_mm=0.3, diameter_mm=0.6, provenance=NET_P)]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    ir.pcb.vias = []
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"]
    # R2 (inscribed radius 0.8, so copper within 1.0 mm of its centre joins it) lies outside the bend at the arc's apex, away from both
    # ends: 0.9 mm from the arc it is joined, 1.05 mm from it it is not - and it is 1.49 mm from the chord (7, 3)-(9, 5)
    for off, joined in ((0.9, True), (1.05, False)):
        centre = _at(-45.0, RADIUS + off)
        assert _seg_point_distance(START, END, centre) > 1.4
        ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", *centre)], {"N": [("R1", "1"), ("R2", "1")]}, (14.0, 12.0))
        ir.pcb.tracks = [_track("N", (3.0, 3.0), START), _arc("N", START, MID, END)]
        c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
        assert c.status is (S.PASS if joined else S.FAIL), (off, c.message)
        assert _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CONNECTIVITY_CHECK].status is S.FAIL


def _touching(tmp_path: Path, lib: KicadLibrary, overlap: float, parts: list | None = None, *, inside: bool = False) -> CircuitIR:
    """The corner board with R3.1, whose only copper is a straight track of net N (``track[3:N]``) that comes beside the arc at -45
    degrees, its copper overlapping the arc's by ``overlap`` mm (negative: a gap). Outside the bend it passes on a tangent line, its
    ends - like the arc's - far from the other track; ``inside`` it runs along the radius from R3 at the arc's centre, and its end is
    what comes near."""
    if inside:
        a, b = _at(-45.0, RADIUS - W + overlap), CENTRE
    else:
        m, ux, uy = _at(-45.0, RADIUS + W - overlap), math.cos(math.radians(45.0)), math.sin(math.radians(45.0))
        a, b = (round(m[0] - 1.5 * ux, 6), round(m[1] - 1.5 * uy, 6)), (round(m[0] + 3.0 * ux, 6), round(m[1] + 3.0 * uy, 6))
    ir = _corner(tmp_path, lib, [("R3", "PAD1", *b), *(parts or [])])
    ir.nets[0].pins.append(PinRef(component_ref="R3", pin_number="1"))
    ir.pcb.tracks.append(_track("N", a, b))
    return ir


def test_tracks_within_the_arc_tolerance_of_touching_leave_the_net_not_verified(tmp_path: Path, lib: KicadLibrary):
    """Two tracks of a net overlap when their centrelines are closer than the half widths (0.4 mm here). An arc is measured on its chords,
    within 1 um of it: copper that overlaps by 0.4 um may or may not, so the net that depends on it is neither connected nor open."""
    ir = _touching(tmp_path, lib, 0.0004)
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    [row] = c.details["nets"]
    assert c.status is S.NOT_VERIFIED and row["status"] == "NOT_VERIFIED" and row["unconnected"] == ["R3.1"], c.message
    assert row["arc_tolerance"] == [{"a": "track[1:N]", "b": "track[3:N]"}]
    assert row["message"] == ("R3.1 not joined to R1.1 through copper measured to overlap; track[1:N] / track[3:N] lie within the arc tolerance of "
                              "touching (an arc is measured on its chords, which lie within 0.001 mm of it): whether they join the net is not measured here")
    assert c.message == ("1 net(s) joined or left open only by tracks within the arc tolerance of touching, which this check does not measure: N: "
                         + row["message"] + " (IR geometry, not DRC)")
    assert checks[CLEARANCE_CHECK].status is S.PASS  # one net: nothing foreign to compare
    # a gap of 0.4 um is never read as a joint (the chords run inside the arc, so from outside the bend they can show a gap this small)
    assert _checks(_touching(tmp_path, lib, -0.0004), tmp_path, lib)[CONNECTIVITY_CHECK].status is not S.PASS
    # beyond the tolerance the same pair is measured: an overlap of 2.5 um joins, a gap of 2.5 um is an open
    assert _checks(_touching(tmp_path, lib, 0.0025), tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    c = _checks(_touching(tmp_path, lib, -0.0025), tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R3.1 not connected to R1.1" and "arc_tolerance" not in c.details["nets"][0]
    # inside the bend the chords are the nearer copper: a gap of 0.4 um is undecided there, a gap of 2.5 um an open - and an overlap of
    # 0.4 um is a joint, because the track's end is a point and a point is measured against the arc exactly
    c = _checks(_touching(tmp_path, lib, -0.0004, inside=True), tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.NOT_VERIFIED and c.details["nets"][0]["arc_tolerance"] == [{"a": "track[1:N]", "b": "track[3:N]"}]
    assert _checks(_touching(tmp_path, lib, -0.0025, inside=True), tmp_path, lib)[CONNECTIVITY_CHECK].status is S.FAIL
    assert _checks(_touching(tmp_path, lib, 0.0004, inside=True), tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS


def test_an_undecided_pair_is_named_only_where_the_nets_row_depends_on_it(tmp_path: Path, lib: KicadLibrary):
    # R3 is also joined to R2 by a straight track: the net is one copper set whatever the undecided pair does
    ir = _touching(tmp_path, lib, 0.0004)
    r3 = ir.pcb.tracks[3].end
    ir.pcb.tracks.append(_track("N", r3, (9.0, 9.0)))
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and "arc_tolerance" not in c.details["nets"][0]
    # ... and a pad without any copper is a measured open whatever the pair does: the row both readings give is the row
    ir = _touching(tmp_path, lib, 0.0004, [("R4", "PAD1", 2.0, 10.0)])
    ir.nets[0].pins.append(PinRef(component_ref="R4", pin_number="1"))
    ir.pcb.tracks.append(_track("N", ir.pcb.tracks[3].end, (9.0, 9.0)))
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R4.1 not connected to R1.1" and "arc_tolerance" not in c.details["nets"][0]
    # without the sure track the pair decides R3 (R4 is open either way): the row would differ, so it is not judged, and the result
    # is NOT_VERIFIED (no other net fails)
    ir.pcb.tracks.pop()
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    row = c.details["nets"][0]
    assert c.status is S.NOT_VERIFIED and row["unconnected"] == ["R3.1", "R4.1"] and row["arc_tolerance"] == [{"a": "track[1:N]", "b": "track[3:N]"}]
    # beside a net that is open for sure the result FAILs and still says which net was not judged
    ir = _touching(tmp_path, lib, 0.0004, [("R5", "PAD1", 2.0, 8.0), ("R6", "PAD1", 2.0, 10.5)])
    ir.nets.append(ir.nets[0].model_copy(update={"name": "K", "pins": [PinRef(component_ref="R5", pin_number="1"), PinRef(component_ref="R6", pin_number="1")]}))
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and {r["net"]: r["status"] for r in c.details["nets"]} == {"N": "NOT_VERIFIED", "K": "FAIL"}
    assert c.message == ("1 net(s) not connected through IR copper: K: R6.1 not connected to R5.1; "
                         "1 net(s) not judged where tracks lie within the arc tolerance of touching (N) (IR geometry, not DRC)")
    # a stub of the net that reaches no pad lies within the tolerance of the arc: joined to it or not, the same pads are joined, so the
    # row is the measured one - R4, which has no copper, is open
    m, u = _at(-45.0, RADIUS + W - 0.0004), math.cos(math.radians(45.0))
    ir = _corner(tmp_path, lib, [("R4", "PAD1", 2.0, 10.0)])
    ir.nets[0].pins.append(PinRef(component_ref="R4", pin_number="1"))
    ir.pcb.tracks.append(_track("N", (round(m[0] - 1.5 * u, 6), round(m[1] - 1.5 * u, 6)), (round(m[0] + 3.0 * u, 6), round(m[1] + 3.0 * u, 6))))
    board = _Board(ir, lib)
    assert _connected(board.tracks[1], board.tracks[3]) is None  # the pair itself is undecided
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R4.1 not connected to R1.1" and "arc_tolerance" not in c.details["nets"][0]


def test_a_track_end_is_measured_against_the_arc_exactly_from_outside_the_bend_too(tmp_path: Path, lib: KicadLibrary):
    """R3's track runs along a radius toward the arc from outside the bend and stops short of its centreline (a T junction). Outside the
    bend the chords lie farther away than the arc, so on them an overlap of 0.4 um could not be told from a gap; the track's end is a
    point, and a point is measured against the arc exactly: it joins. A gap of 0.4 um is an open - the chords are then more than the
    tolerance away."""
    def end_on(overlap: float) -> CircuitIR:
        r3 = _at(-45.0, RADIUS + 2.5)
        ir = _corner(tmp_path, lib, [("R3", "PAD1", *r3)])
        ir.nets[0].pins.append(PinRef(component_ref="R3", pin_number="1"))
        ir.pcb.tracks.append(_track("N", _at(-45.0, RADIUS + W - overlap), r3))
        return ir

    ir = end_on(0.0004)
    board = _Board(ir, lib)
    arc, stub = board.tracks[1], board.tracks[3]
    assert 0.0 < _copper_distance(arc, stub) < ARC_SAGITTA_MM  # on the chords the copper even reads as apart, by less than the tolerance
    assert _connected(arc, stub) is True and _connected(stub, arc) is True
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.details["nets"][0]["message"] == "3 pad(s) in one copper set" and "arc_tolerance" not in c.details["nets"][0]
    c = _checks(end_on(-0.0004), tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R3.1 not connected to R1.1" and "arc_tolerance" not in c.details["nets"][0]


# --------------------------------------------------------------------------- clearance


def test_foreign_copper_under_the_bulge_is_a_violation_the_chord_would_miss(tmp_path: Path, lib: KicadLibrary):
    """A foreign track 0.1 mm from the arc's copper on the outside of the bend: 0.686 mm from the copper of the chord (7, 3)-(9, 5)."""
    ir = _foreign(tmp_path, lib, _outside(0.1))
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS
    c = checks[CLEARANCE_CHECK]
    [row] = c.details["violations"]
    assert c.status is S.FAIL and (row["a"], row["b"], row["layer"], row["limit_mm"], row["status"]) == ("track[1:N]", "track[3:M]", "F.Cu", LIMIT, "FAIL")
    assert 0.1 - 1e-6 <= row["distance_mm"] <= 0.1 + ARC_SAGITTA_MM and row["arc_tolerance_mm"] == ARC_SAGITTA_MM  # measured on the chords
    assert row["message"] == (f"track[1:N] vs track[3:M] on F.Cu: {row['distance_mm']:g} mm < 0.25 mm "
                              "(measured on the chords of an arc, within 0.001 mm of the true distance)")
    assert c.message == f"1 copper pair(s) of different nets closer than 0.25 mm: {row['message']} (IR geometry, not DRC)"
    assert "arc_tolerance" not in c.details and c.details["arcs"] == 1
    assert _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK].status is S.PASS  # the chord reading: a short gap passed
    # a foreign track across the arc is a short whatever the limit
    ir = _foreign(tmp_path, lib, _track("M", _at(-45.0, 1.9), _at(-45.0, 3.0)))
    _limit(ir, 0.0)
    [row] = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].details["violations"]
    assert row["distance_mm"] == 0.0 and row["message"] == ("track[1:N] vs track[3:M] on F.Cu: copper overlaps (a short, whatever the 0 mm limit) "
                                                            "(measured on the chords of an arc, within 0.001 mm of the true distance)")


def test_foreign_copper_inside_the_bend_is_clear_where_the_chord_would_fail(tmp_path: Path, lib: KicadLibrary):
    """A foreign track that ends 0.4 mm from the arc's copper on the inside of the bend: its copper overlaps the chord's."""
    ir = _foreign(tmp_path, lib, _inside(0.4))
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and c.details["violations"] == [] and "arc_tolerance" not in c.details, c.message
    assert c.message.startswith("8 copper pair(s) of different nets keep >= 0.25 mm on a shared layer, 4 track(s) / 0 via(s) inside the outline")
    c = _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"], r["distance_mm"]) for r in c.details["violations"]] == [("track[1:N]", "track[3:M]", 0.0)]


def test_a_pair_within_the_arc_tolerance_of_the_limit_is_not_verified(tmp_path: Path, lib: KicadLibrary):
    """Foreign copper whose true distance from the arc's copper is 0.4 um under / over the 0.25 mm limit: the chords the arc is measured
    on lie within 1 um of it, so the measured distance cannot say on which side of the limit the pair is. Neither PASS nor FAIL."""
    for track in (_outside(LIMIT - 0.0004), _inside(LIMIT + 0.0004)):
        c = _checks(_foreign(tmp_path, lib, track), tmp_path, lib)[CLEARANCE_CHECK]
        [row] = c.details["arc_tolerance"]
        assert c.status is S.NOT_VERIFIED and c.details["violations"] == [], c.message
        assert (row["a"], row["b"], row["layer"], row["status"]) == ("track[1:N]", "track[3:M]", "F.Cu", "NOT_VERIFIED")
        assert row["within_arc_tolerance"] is True and row["arc_tolerance_mm"] == ARC_SAGITTA_MM and abs(row["distance_mm"] - LIMIT) < 2 * ARC_SAGITTA_MM
        assert row["message"] == (f"track[1:N] vs track[3:M] on F.Cu: {row['distance_mm']:g} mm measured on the chords of an arc, within 0.001 mm of the "
                                  "true distance: only that arc tolerance decides whether the pair keeps the 0.25 mm limit")
        assert c.message == ("1 copper pair(s) with an arc that only the arc tolerance puts on either side of 0.25 mm (not measured): "
                             + row["message"] + " (IR geometry, not DRC)")
    # 2.5 um from the limit the tolerance cannot change the verdict: measured, on either side of the bend
    for near, far in ((_outside(LIMIT - 0.0025), _outside(LIMIT + 0.0025)), (_inside(LIMIT - 0.0025), _inside(LIMIT + 0.0025))):
        c = _checks(_foreign(tmp_path, lib, near), tmp_path, lib)[CLEARANCE_CHECK]
        assert c.status is S.FAIL and len(c.details["violations"]) == 1 and "arc_tolerance" not in c.details
        c = _checks(_foreign(tmp_path, lib, far), tmp_path, lib)[CLEARANCE_CHECK]
        assert c.status is S.PASS and c.details["violations"] == [] and "arc_tolerance" not in c.details
    # at a 0 mm limit the question is whether the copper touches (a short): 0.4 um apart inside the bend, where the chords are the
    # nearer copper, is not measured; 2.5 um apart is clear, and so is 0.4 um outside the bend (there the chords can only be farther)
    ir = _foreign(tmp_path, lib, _inside(0.0004))
    _limit(ir, 0.0)
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.NOT_VERIFIED and c.details["arc_tolerance"][0]["distance_mm"] <= 0.0004 + 1e-6
    assert "only that arc tolerance decides whether the pair touches (a short, whatever the 0 mm limit)" in c.message
    ir.pcb.tracks[3] = _inside(0.0025)
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS
    ir.pcb.tracks[3] = _outside(0.0004)
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is not S.FAIL


def test_a_pad_box_is_measured_on_the_chords_and_a_via_exactly(tmp_path: Path, lib: KicadLibrary):
    def pad_board(gap: float) -> CircuitIR:
        # a foreign 1 x 1 mm rect pad (its box is its copper) whose corner nearest the arc is ``gap`` from the arc's copper
        q = _at(-45.0, RADIUS + W / 2.0 + gap)
        return _corner(tmp_path, lib, [("S1", "SMD1", q[0] + 0.5, q[1] - 0.5)], {"M": [("S1", "1")]})

    c = _checks(pad_board(LIMIT - 0.0004), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.NOT_VERIFIED and [(r["a"], r["b"], r["arc_tolerance_mm"]) for r in c.details["arc_tolerance"]] == [("track[1:N]", "S1.1", ARC_SAGITTA_MM)]
    c = _checks(pad_board(LIMIT - 0.0025), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"]) for r in c.details["violations"]] == [("track[1:N]", "S1.1")]
    assert _checks(pad_board(LIMIT + 0.0025), tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS

    def via_board(gap: float) -> CircuitIR:
        ir = _foreign(tmp_path, lib)
        x, y = _at(-30.0, RADIUS + W / 2.0 + 0.3 + gap)
        ir.pcb.vias = [Via(net="M", x_mm=x, y_mm=y, drill_mm=0.3, diameter_mm=0.6, provenance=NET_P)]
        return ir

    # a via is a point with a radius, and a point is measured against the arc exactly: 0.4 um decide, and no row speaks of a tolerance
    c = _checks(via_board(LIMIT - 0.0004), tmp_path, lib)[CLEARANCE_CHECK]
    [row] = c.details["violations"]
    assert c.status is S.FAIL and (row["a"], row["b"]) == ("track[1:N]", "via[0:M]") and row["distance_mm"] == pytest.approx(LIMIT - 0.0004, abs=2e-6)
    assert "arc_tolerance_mm" not in row and row["message"] == f"track[1:N] vs via[0:M] on F.Cu: {row['distance_mm']:g} mm < 0.25 mm"
    c = _checks(via_board(LIMIT + 0.0004), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and "arc_tolerance" not in c.details


def test_two_arcs_are_measured_with_the_tolerance_of_both(tmp_path: Path, lib: KicadLibrary):
    """Net M's arc runs round the same centre outside net N's corner arc (a bus corner): the true copper gap is the difference of the radii
    less the width - also between M's arc and N's two straight tracks, whose ends its ends pass. Each arc's chords lie within 1 um of
    it: 1 um for an arc against a straight track, 2 um for the two arcs."""
    def bus(gap: float) -> CircuitIR:
        r = RADIUS + W + gap
        ir = _foreign(tmp_path, lib)
        ir.pcb.tracks.append(_arc("M", _at(-90.0, r), _at(-45.0, r), _at(0.0, r)))
        return ir

    pairs = [("track[0:N]", "track[3:M]", ARC_SAGITTA_MM), ("track[1:N]", "track[3:M]", 2 * ARC_SAGITTA_MM), ("track[2:N]", "track[3:M]", ARC_SAGITTA_MM)]
    c = _checks(bus(LIMIT - 0.01), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"], r["arc_tolerance_mm"]) for r in c.details["violations"]] == pairs and c.details["arcs"] == 2
    for row in c.details["violations"]:  # each measured distance within its own tolerance of the true 0.24 mm
        assert abs(row["distance_mm"] - (LIMIT - 0.01)) <= row["arc_tolerance_mm"] + 1e-6
        assert row["message"].endswith(f"(measured on the chords of an arc, within {row['arc_tolerance_mm']:g} mm of the true distance)")
    assert _checks(bus(LIMIT + 0.01), tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS
    c = _checks(bus(LIMIT + 0.0005), tmp_path, lib)[CLEARANCE_CHECK]  # half a micrometre over the limit: inside both tolerances
    assert c.status is S.NOT_VERIFIED and c.details["violations"] == [] and [(r["a"], r["b"], r["arc_tolerance_mm"]) for r in c.details["arc_tolerance"]] == pairs
    assert c.message.startswith("3 copper pair(s) with an arc that only the arc tolerance puts on either side of 0.25 mm (not measured): track[0:N] vs track[3:M]")
    # the rows function itself: (pairs compared, rows), as for straight copper
    compared, rows = clearance_rows(_Board(bus(LIMIT + 0.0005), lib), LIMIT)
    assert compared == 8 and [r["status"] for r in rows] == ["NOT_VERIFIED"] * 3


def test_an_arc_is_measured_exactly_against_a_custom_pads_disc(tmp_path: Path):
    """MK1 (``Test:RING``) at (8, 6): a ring whose copper lies in the disc of r 1.75 around (8, 6) (its box 6.25 .. 9.75), the anchor on
    it at (8, 4.625). A foreign arc round the same centre keeps exactly its radius less 0.2 (half its width) less 1.75 from that disc -
    measured so, although its chord runs through the ring."""
    lib = custom_pad_library(tmp_path / "kicad")

    def ring(radius: float) -> CircuitIR:
        ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0), ("R3", "PAD1", 13.0, 10.0)],
                      {"M": [("MK1", "1"), ("R1", "1")], "Q": [("R3", "1")]}, (16.0, 12.0))
        _limit(ir)
        c = (8.0, 6.0)
        ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 4.625)), _track("M", (3.0, 4.625), (8.0, 4.625)),
                         _arc("Q", _at(0.0, radius, c), _at(45.0, radius, c), _at(90.0, radius, c))]
        return ir

    c = _checks(ring(2.25), tmp_path, lib)[CLEARANCE_CHECK]  # 0.3 mm from the disc: clear, however deep in the ring's box
    assert c.status is S.PASS and "custom_pad_bounds" not in c.details and "arc_tolerance" not in c.details, c.message
    c = _checks(_as_chord(ring(2.25)), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK]  # its chord crosses the ring's disc
    assert c.status is S.NOT_VERIFIED and [(r["a"], r["b"]) for r in c.details["custom_pad_bounds"]] == [("track[2:Q]", "MK1.1")]
    c = _checks(ring(2.05), tmp_path, lib)[CLEARANCE_CHECK]  # 0.1 mm from the disc that only bounds the ring's copper: not measured
    [row] = c.details["custom_pad_bounds"]
    assert c.status is S.NOT_VERIFIED and c.details["violations"] == [] and (row["a"], row["b"], row["bound"]) == ("track[2:Q]", "MK1.1", True)
    assert row["distance_mm"] == pytest.approx(0.1, abs=1e-6) and "arc_tolerance_mm" not in row and "from the disc that bounds" in row["message"]
    # connectivity: an arc of the ring's own net that ends inside the ring's disc, away from the anchor, may or may not touch the ring
    ir = ring(2.25)
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 6.0)), _arc("M", (3.0, 6.0), (4.6, 7.6), (6.2, 6.0))]  # ends 1.8 mm from the ring's centre
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    row = next(r for r in c.details["nets"] if r["net"] == "M")
    assert c.status is S.NOT_VERIFIED and row["bounded_pads"] == ["MK1.1"] and "arc_tolerance" not in row, c.message
    ir.pcb.tracks[1] = _arc("M", (3.0, 6.0), (4.5, 7.5), (6.0, 6.0))  # ends 2.0 mm from the centre: 0.05 mm clear of the disc, a measured open
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and "bounded_pads" not in next(r for r in c.details["nets"] if r["net"] == "M")


def test_known_copper_of_a_custom_pad_inside_the_limit_of_an_arc_is_a_violation(tmp_path: Path):
    """MK1's anchor (``Test:RING`` at (8, 6): a 0.75 mm circle round (8, 4.625), copper for sure; the ring's disc tops out at the same
    point (8, 4.25)) under a foreign arc that bulges down to it. The arc's distance from that known copper is measured exactly, so
    half a micrometre inside the limit is a violation - the very rows a straight track at that distance gets -, although the anchor's
    box, measured on the arc's chords (the apex lies between two points of the chain), reads within the arc tolerance of the limit."""
    lib = custom_pad_library(tmp_path / "kicad")

    def under(gap: float, arc: bool = True, limit: float = LIMIT) -> CircuitIR:
        ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 10.0), ("R3", "PAD1", 13.0, 10.0)],
                      {"M": [("MK1", "1"), ("R1", "1")], "Q": [("R3", "1")]}, (16.0, 12.0))
        _limit(ir, limit)
        y = 4.25 - gap - W / 2.0  # the foreign centreline's lowest point, straight above the anchor
        c = (8.0, y - RADIUS)
        ir.pcb.tracks = [_arc("Q", _at(45.0, RADIUS, c), _at(90.0, RADIUS, c), _at(135.0, RADIUS, c)) if arc else _track("Q", (6.5, y), (9.5, y))]
        return ir

    for gap in (LIMIT - 0.0005, LIMIT - 0.0015):
        [foreign] = under(gap).pcb.tracks
        assert (len(arc_points(foreign.start, foreign.mid, foreign.end)) - 1) % 2 == 1  # an odd number of chords: the apex is no chain point
        c = _checks(under(gap), tmp_path, lib)[CLEARANCE_CHECK]
        assert c.status is S.FAIL and "arc_tolerance" not in c.details, c.message
        assert c.details["violations"] == [{"a": "track[0:Q]", "b": "MK1.1", "layer": "F.Cu", "distance_mm": round(gap, 6), "limit_mm": LIMIT, "status": "FAIL",
                                            "message": f"track[0:Q] vs MK1.1 on F.Cu: {gap:g} mm < 0.25 mm"}]
        [ring] = c.details["custom_pad_bounds"]  # the ring's disc only bounds its copper: its own row, at the exact distance too
        assert ring["distance_mm"] == round(gap, 6) and ring["message"].startswith(f"track[0:Q] vs MK1.1 on F.Cu: {gap:g} mm from the disc that bounds")
        straight = _checks(under(gap, arc=False), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK]
        assert (straight.details["violations"], straight.details["custom_pad_bounds"]) == (c.details["violations"], c.details["custom_pad_bounds"])
    # half a micrometre beyond the limit both parts are clear by their discs, exactly
    c = _checks(under(LIMIT + 0.0005), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and "custom_pad_bounds" not in c.details and "arc_tolerance" not in c.details, c.message
    # at a 0 mm limit: copper half a micrometre into the anchor's is a short, as for the straight track
    c = _checks(under(-0.0005, limit=0.0), tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["distance_mm"], r["message"]) for r in c.details["violations"]] == [
        (0.0, "track[0:Q] vs MK1.1 on F.Cu: copper overlaps (a short, whatever the 0 mm limit)")]
    assert c.details["violations"] == _checks(under(-0.0005, arc=False, limit=0.0), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK].details["violations"]


def test_an_arc_within_its_tolerance_of_a_custom_pads_bound_may_touch_the_pad(tmp_path: Path):
    """The ring's disc (r 1.75 round (8, 6)) touches its bounding box at (6.25, 6). An arc of the ring's net bulges toward that point, its
    copper 0.4 um into the disc (measured exactly) - while its chords, which run inside the bulge, stay 0.6 um short of the box. The
    contact is not excluded: NOT_VERIFIED naming the pad, never the FAIL of a measured open. 0.4 um short of the disc it is an open."""
    lib = custom_pad_library(tmp_path / "kicad")

    def toward(overlap: float) -> CircuitIR:
        ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
        c = (6.25 + overlap - W / 2.0 - RADIUS, 6.0)  # the arc's apex (its copper's edge at 6.25 + overlap) faces the ring
        start = _at(-45.0, RADIUS, c)
        ir.pcb.tracks = [_track("M", (3.0, 3.0), start), _arc("M", start, _at(0.0, RADIUS, c), _at(45.0, RADIUS, c))]
        return ir

    ir = toward(0.0004)
    board = _Board(ir, lib)
    ring = next(p for p in board.pads if p.label == "MK1.1" and not p.joins)
    assert 0.0 < _copper_distance(board.tracks[1], ring) < ARC_SAGITTA_MM  # on the chords: apart from the ring's box, by less than the tolerance
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    row = next(r for r in c.details["nets"] if r["net"] == "M")
    assert c.status is S.NOT_VERIFIED and row["status"] == "NOT_VERIFIED" and row["bounded_pads"] == ["MK1.1"] and "arc_tolerance" not in row, c.message
    c = _checks(toward(-0.0004), tmp_path, lib)[CONNECTIVITY_CHECK]
    row = next(r for r in c.details["nets"] if r["net"] == "M")
    assert c.status is S.FAIL and row["status"] == "FAIL" and "bounded_pads" not in row, c.message


def test_an_undecided_pair_is_named_beside_an_unknown_pad_and_beside_a_custom_pads_bound(tmp_path: Path):
    lib = custom_pad_library(tmp_path / "kicad")
    # a trapezoid's copper is unknown, so the result is NOT_VERIFIED for that already: the pair the tolerance decides is named too, and
    # is not counted among the pairs that keep the limit
    ir = _corner(tmp_path, lib, [("R3", "PAD1", 2.0, 10.0), ("U9", "TRAP", 5.0, 10.0)], {"M": [("R3", "1")], "X": [("U9", "1")]})
    ir.pcb.tracks.append(_outside(LIMIT - 0.0004))
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    [row] = c.details["arc_tolerance"]
    assert c.status is S.NOT_VERIFIED and c.details["pairs_compared"] == 8 and c.details["violations"] == []
    assert c.message == ("pad geometry unknown: pad U9.1 of footprint Test:TRAP has shape 'trapezoid', whose copper is not bounded by its (size) box "
                         "(a trapezoid's rect_delta is not read); 7 copper pair(s) of known items keep >= 0.25 mm; 1 copper pair(s) with an arc that only "
                         "the arc tolerance puts on either side of 0.25 mm (not measured): " + row["message"] + " (IR geometry, not DRC)")
    # a straight track 0.05 mm from the disc that only bounds a ring's copper, and the same undecided pair: both reasons are given
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 9.0), ("R3", "PAD1", 2.0, 10.0), ("MK1", "RING", 13.0, 7.0)],
                  {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")], "X": [("MK1", "1")]}, (18.0, 12.0))
    _limit(ir)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), START), _arc("N", START, MID, END), _track("N", END, (9.0, 9.0)), _outside(LIMIT - 0.0004),
                     _track("M", (13.0, 9.0), (16.0, 9.0))]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    [bound], [row] = c.details["custom_pad_bounds"], c.details["arc_tolerance"]
    assert c.status is S.NOT_VERIFIED and (bound["a"], bound["b"], bound["distance_mm"]) == ("track[4:M]", "MK1.1", 0.05) and "arc_tolerance_mm" not in bound
    assert c.message == ("1 copper pair(s) closer than 0.25 mm only to the box / disc that bounds a custom pad's copper, where the copper is not known: "
                         + bound["message"] + "; 1 copper pair(s) with an arc that only the arc tolerance puts on either side of 0.25 mm (not measured): "
                         + row["message"] + " (IR geometry, not DRC)")


# --------------------------------------------------------------------------- the outline and keep-outs


def _span(tmp_path: Path, lib: KicadLibrary, apex_y: float) -> CircuitIR:
    """R1.1 at (3, 3) and R2.1 at (9, 3) joined by one arc over (6, ``apex_y``) on a 12 x 6 mm board: its chord is the straight track
    between the pads, 2.8 mm inside the top edge."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    ir.pcb.tracks = [_arc("N", (3.0, 3.0), (6.0, apex_y), (9.0, 3.0))]
    return ir


def test_an_arc_whose_bulge_leaves_the_outline_is_reported(tmp_path: Path, lib: KicadLibrary):
    ir = _span(tmp_path, lib, 0.1)  # the copper's edge 0.1 mm beyond the top edge
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS
    c = checks[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["limit_mm"] is None  # FAIL whatever the clearance limit, as for straight copper
    assert c.details["outline"] == [{"item": "track[0:N]", "status": "FAIL", "message": "track[0:N] copper leaves the 12 x 6 mm outline at (0, 0)"}]
    c = _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK]
    assert c.details["outline"] == [] and c.status is S.NOT_VERIFIED  # the chord stays inside (and no limit is recorded)
    # copper that only touches the edge has not left the outline; 0.1 mm inside it neither
    for apex in (0.2, 0.3):
        ir = _span(tmp_path, lib, apex)
        _limit(ir)
        c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
        assert c.status is S.PASS and c.details["outline"] == [], (apex, c.message)


def test_the_pair_prefilter_uses_the_arcs_own_box(tmp_path: Path, lib: KicadLibrary):
    """A foreign via above the apex of the arc, 0.1 mm from its copper: 1.6 mm from the box of the arc's two ends, which a prefilter on
    that box would never compare."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 10.5, 5.0)], {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (12.0, 6.0))
    _limit(ir)
    ir.pcb.tracks = [_arc("N", (3.0, 3.0), (6.0, 1.5), (9.0, 3.0))]
    ir.pcb.vias = [Via(net="M", x_mm=6.0, y_mm=0.9, drill_mm=0.3, diameter_mm=0.6, provenance=NET_P)]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"]) for r in c.details["violations"]] == [("track[0:N]", "via[0:M]")]
    assert c.details["violations"][0]["distance_mm"] == pytest.approx(0.1, abs=1e-6)
    assert _checks(_as_chord(ir), tmp_path, lib, TOOL_VERSION)[CLEARANCE_CHECK].status is S.PASS


def _fenced(tmp_path: Path, lib: KicadLibrary, height: float, **more) -> CircuitIR:
    """The arc over (6, 1.5) - its copper's edge at y = 1.3 - and a keep-out for tracks on F.Cu from y = 0.5 down to 0.5 + ``height``."""
    ir = _span(tmp_path, lib, 1.5)
    ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([5.0, 0.5, 2.0, height], "mm"), forbids=["tracks"], reason="test area",
                               provenance=NET_P, **more)]
    return ir


def test_an_arc_bulging_into_a_track_keep_out_is_reported(tmp_path: Path, lib: KicadLibrary):
    ir = _fenced(tmp_path, lib, 1.2)  # the keep-out reaches y = 1.7: the arc's copper is 0.4 mm inside it
    res = _keepout(ir, tmp_path, lib)
    assert res.status is S.FAIL and res.tool_version == ARC_TOOL_VERSION and res.details["arc_model"] == ARC_MODEL and res.details["arcs"] == 1
    assert res.details["rows"] == [{"item": "track[0:N]", "status": "FAIL", "message": "keep-out K forbids tracks on F.Cu; track[0:N] (N) reaches into it", "keepout": "K"}]
    chord = _keepout(_as_chord(ir), tmp_path, lib)  # the chord at y = 3 is 1.1 mm away
    assert chord.status is S.PASS and chord.tool_version == KEEPOUT_TOOL_VERSION == "0.1" and "arc_model" not in chord.details
    # the keep-out's exception holds for an arc as for a straight track
    res = _keepout(_fenced(tmp_path, lib, 1.2, allowed_nets=["N"]), tmp_path, lib)
    assert res.status is S.PASS and res.details["keepouts"][0]["exempt"] == ["track[0:N]"] and res.details["rows"] == []
    # 10 um into the keep-out and 10 um clear of it are measured
    assert _keepout(_fenced(tmp_path, lib, 0.81), tmp_path, lib).status is S.FAIL
    res = _keepout(_fenced(tmp_path, lib, 0.79), tmp_path, lib)
    assert res.status is S.PASS and res.details["keepouts"][0]["compared"] == 1 and res.details["keepouts"][0]["exempt"] == []
    # 0.4 um into it: the chords lie within 1 um of the arc, so whether it reaches in is not measured - not judged, and no exception
    # is claimed for it either
    res = _keepout(_fenced(tmp_path, lib, 0.8004), tmp_path, lib)
    assert res.status is S.NOT_VERIFIED and res.details["rows"] == [{
        "item": "track[0:N]", "status": "NOT_VERIFIED", "keepout": "K",
        "message": ("keep-out K forbids tracks on F.Cu; whether track[0:N] (N) reaches into it is decided only by the arc tolerance "
                    "(an arc is measured on its chords, which lie within 0.001 mm of it): not measured here"),
    }]
    assert res.message.startswith("not every item could be judged: keep-out K forbids tracks on F.Cu; whether track[0:N] (N) reaches into it")
    res = _keepout(_fenced(tmp_path, lib, 0.8004, allowed_nets=["N"]), tmp_path, lib)
    assert res.status is S.PASS and res.details["rows"] == [] and res.details["keepouts"][0]["exempt"] == []
    # a keep-out the arc's box does not meet (here beyond the pads) is clear without a chord being measured
    ir = _span(tmp_path, lib, 1.5)
    ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([4.0, 4.0, 4.0, 1.5], "mm"), forbids=["tracks"], reason="test area", provenance=NET_P)]
    res = _keepout(ir, tmp_path, lib)
    assert res.status is S.PASS and res.details["keepouts"][0]["compared"] == 1


def test_copper_boxes_that_are_apart_decide_exactly(tmp_path: Path, lib: KicadLibrary):
    """An arc's box is its own, exact one, so copper whose box is clear of the arc's copper box is clear of the arc for sure - also by
    0.4 um, where the chords alone could not say. The arc over (6, 1.7) has an even number of chords: its apex (the copper's edge at
    y = 1.5) is a point of its chain, so there the chords come as near as the arc does."""
    assert (len(arc_points((3.0, 3.0), (6.0, 1.7), (9.0, 3.0))) - 1) % 2 == 0

    def beside(gap: float) -> CircuitIR:
        # R3's straight track of net N passes above the apex, its copper ``gap`` from the arc's (negative: into it)
        y = 1.5 - gap - W / 2.0
        ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 10.5, y)], {"N": [("R1", "1"), ("R2", "1"), ("R3", "1")]}, (12.0, 6.0))
        ir.pcb.tracks = [_arc("N", (3.0, 3.0), (6.0, 1.7), (9.0, 3.0)), _track("N", (4.5, y), (10.5, y))]
        return ir

    c = _checks(beside(0.0004), tmp_path, lib)[CONNECTIVITY_CHECK]  # the two copper boxes are 0.4 um apart: a measured open
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R3.1 not connected to R1.1" and "arc_tolerance" not in c.details["nets"][0]
    c = _checks(beside(-0.0004), tmp_path, lib)[CONNECTIVITY_CHECK]  # 0.4 um into each other: the chords cannot say
    assert c.status is S.NOT_VERIFIED and c.details["nets"][0]["arc_tolerance"] == [{"a": "track[0:N]", "b": "track[1:N]"}]
    assert _checks(beside(-0.0025), tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS

    def fenced(height: float) -> CircuitIR:
        ir = _span(tmp_path, lib, 1.7)
        ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([5.0, 0.5, 2.0, height], "mm"), forbids=["tracks"], reason="test area", provenance=NET_P)]
        return ir

    res = _keepout(fenced(1.0 - 0.0004), tmp_path, lib)  # the keep-out ends 0.4 um above the arc's copper box: clear, exactly
    assert res.status is S.PASS and res.details["rows"] == [] and res.details["keepouts"][0]["compared"] == 1
    res = _keepout(fenced(1.0 + 0.0004), tmp_path, lib)  # 0.4 um into it: only the tolerance decides
    assert res.status is S.NOT_VERIFIED and [r["status"] for r in res.details["rows"]] == ["NOT_VERIFIED"]


# --------------------------------------------------------------------------- malformed arcs


def test_three_points_that_are_no_arc_are_a_fail_row_never_a_crash(tmp_path: Path, lib: KicadLibrary):
    """``Track`` refuses such points, so none comes out of a file; an IR changed in the process can still hold one. Like every track that
    cannot be copper it joins nothing, is compared with nothing and FAILs both checks by name."""
    nan = float("nan")
    cases = [
        ((3.0, 3.0), (6.0, 3.0), (9.0, 3.0), "track[1:N] is no arc (start (3.0, 3.0), mid (6.0, 3.0), end (9.0, 3.0)): its mid point lies on the line "
         "through its start and end (within 1e-06 mm): the three points define no circle; its copper cannot be measured (the Track model refuses these points too)"),
        ((3.0, 3.0), (3.0, 3.0), (9.0, 3.0), "track[1:N] is no arc (start (3.0, 3.0), mid (3.0, 3.0), end (9.0, 3.0)): its mid point coincides with its start; "
         "its copper cannot be measured (the Track model refuses these points too)"),
        ((3.0, 3.0), (6.0, 1.5), (3.0, 3.0), "track[1:N] is no arc (start (3.0, 3.0), mid (6.0, 1.5), end (3.0, 3.0)): its start and end coincide "
         "(a full circle has no start / mid / end form); its copper cannot be measured (the Track model refuses these points too)"),
        ((3.0, 3.0), (nan, 1.5), (9.0, 3.0), "track[1:N] has a non-finite coordinate or width (start (3.0, 3.0), mid (nan, 1.5), end (9.0, 3.0), width 0.4): "
         "its copper cannot be measured"),
        ((0.0, 0.0), (1e200, 1e200), (2e200, 0.0), "track[1:N] is no arc (start (0.0, 0.0), mid (1e+200, 1e+200), end (2e+200, 0.0)): its circle cannot be "
         "computed (its coordinates are too large to multiply); its copper cannot be measured (the Track model refuses these points too)"),
        ((0.0, 0.0), (10000.0, 2e-06), (20000.0, 0.0), "track[1:N] is no arc (start (0.0, 0.0), mid (10000.0, 2e-06), end (20000.0, 0.0)): its radius "
         "(2.5e+13 mm) is above 1e+06 mm, beyond which its points cannot be computed to the file resolution; its copper cannot be measured (the Track model "
         "refuses these points too)"),
    ]
    for start, mid, end, message in cases:
        ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
        _limit(ir)
        bad = _arc("N", (3.0, 3.0), (6.0, 1.5), (9.0, 3.0))
        bad.start, bad.mid, bad.end = start, mid, end  # past the model's validator
        ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0)), bad]
        ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([5.0, 0.5, 2.0, 1.2], "mm"), forbids=["tracks"], reason="test area", provenance=NET_P)]
        checks = _checks(ir, tmp_path, lib)
        c, d = checks[CONNECTIVITY_CHECK], checks[CLEARANCE_CHECK]
        assert c.status is S.FAIL and c.details["nets"][0]["status"] == "PASS" and c.details["nets"][0]["tracks"] == 1  # the good track joins the pads
        assert c.details["items"] == [{"item": "track[1:N]", "status": "FAIL", "message": message}], c.details["items"]
        assert "1 track(s) / via(s) that cannot be copper: " + message in c.message
        assert d.status is S.FAIL and d.details["malformed"] == c.details["items"] and d.message.startswith("1 track(s) / via(s) that cannot be copper: track[1:N] ")
        assert c.details["tracks"] == 2 and c.details["arcs"] == 1  # counted as the IR has them
        assert _keepout(ir, tmp_path, lib).status is S.PASS  # the keep-out check compares the copper that can exist: the straight track
    # an arc of non-positive width is refused like a straight track of one
    ir = _span(tmp_path, lib, 1.5)
    ir.pcb.tracks[0].width_mm = 0.0
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and [r["message"] for r in c.details["items"]] == ["track[0:N] has non-positive width 0 mm: no copper (the compiler refuses it too)"]


# --------------------------------------------------------------------------- versions, and boards without an arc


def test_results_are_stamped_0_3_only_on_a_board_with_an_arc(tmp_path: Path, lib: KicadLibrary):
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})

    def versions(ir: CircuitIR) -> dict[str, str]:
        return {r.check_id: r.tool_version for tool in (TOOL_ID, KEEPOUT_CHECK) for r in default_registry.get(tool).validate(ir, ctx)}

    ir = _corner(tmp_path, lib)
    ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([1.0, 9.0, 2.0, 2.0], "mm"), forbids=["tracks"], reason="test area", provenance=NET_P)]
    micro = Via(net="N", x_mm=5.0, y_mm=3.0, drill_mm=0.1, diameter_mm=0.3, layers=("F.Cu", "B.Cu"), kind="micro", provenance=NET_P)
    checks = (CONNECTIVITY_CHECK, CLEARANCE_CHECK, KEEPOUT_CHECK)
    assert ARC_TOOL_VERSION == "0.3" and (TOOL_VERSION, KEEPOUT_TOOL_VERSION, TYPED_VIA_TOOL_VERSION) == ("0.1", "0.1", "0.2")
    assert versions(ir) == dict.fromkeys(checks, "0.3")  # one arc
    straight = _as_chord(ir)
    assert versions(straight) == dict.fromkeys(checks, "0.1")  # no arc: today's versions
    plain = _two_pads(tmp_path, lib)  # the two-pad board of tests/test_layout_validation.py with its one straight track
    plain.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    assert versions(plain) == dict.fromkeys(checks, "0.1")
    straight.pcb.vias = [micro]
    assert versions(straight) == dict.fromkeys(checks, "0.2")  # a typed via, no arc: the typed-via version, as before
    ir.pcb.vias = [micro]
    assert versions(ir) == dict.fromkeys(checks, "0.3")  # a typed via and an arc
    # a result that judged nothing is stamped like the others
    assert {r.tool_version for r in default_registry.get(TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path))} == {"0.3"}
    # the silkscreen checks read no track: their version does not move
    silk = default_registry.get(SILK_TOOL_ID).validate(ir, ctx)
    assert len(silk) == 3 and {r.tool_version for r in silk} == {SILK_TOOL_VERSION} == {"0.1"}
    assert all("tracks (silk over mask-covered copper is normal)" in r.details["not_compared"] for r in silk if "not_compared" in r.details)


#: what the three checks said about :func:`_chamfered` before they knew arcs (status, message, version and every detail)
BEFORE_ARCS = {
    CONNECTIVITY_CHECK: {
        "status": "PASS", "tool": "pcb.routing", "tool_version": "0.1",
        "message": "1 net(s) connected through IR copper (4 track(s), 1 via(s)) (IR geometry check, not DRC)",
        "details": {
            "kind": "ir_geometry",
            "nets": [{"net": "N", "pads": ["R1.1", "R2.1"], "tracks": 3, "vias": 1, "status": "PASS", "message": "2 pad(s) in one copper set"},
                     {"net": "M", "pads": ["R3.1"], "tracks": 1, "vias": 0, "status": "NOT_APPLICABLE", "message": "1 pad(s): nothing to connect"}],
            "items": [], "tracks": 4, "vias": 1, "zones": 0, "unknown": [],
        },
    },
    CLEARANCE_CHECK: {
        "status": "FAIL", "tool": "pcb.routing", "tool_version": "0.1",
        "message": ("2 copper pair(s) of different nets closer than 0.25 mm: track[1:N] vs track[3:M] on F.Cu: 0.165685 mm < 0.25 mm; "
                    "track[1:N] vs R3.1 on F.Cu: 0.082843 mm < 0.25 mm (IR geometry, not DRC)"),
        "details": {
            "kind": "ir_geometry", "limit_mm": 0.25, "hole_to_edge_limit_mm": None, "outline": [], "malformed": [],
            "not_compared": ["pad-to-pad (footprint / placement geometry; DRC judges it)"], "tracks": 4, "vias": 1, "pairs_compared": 10,
            "violations": [
                {"a": "track[1:N]", "b": "track[3:M]", "layer": "F.Cu", "distance_mm": 0.165685, "limit_mm": 0.25, "status": "FAIL",
                 "message": "track[1:N] vs track[3:M] on F.Cu: 0.165685 mm < 0.25 mm"},
                {"a": "track[1:N]", "b": "R3.1", "layer": "F.Cu", "distance_mm": 0.082843, "limit_mm": 0.25, "status": "FAIL",
                 "message": "track[1:N] vs R3.1 on F.Cu: 0.082843 mm < 0.25 mm"},
            ],
        },
    },
    KEEPOUT_CHECK: {
        "status": "FAIL", "tool": "pcb.keepout", "tool_version": "0.1",
        "message": ("2 violation(s): keep-out K forbids tracks on F.Cu; track[1:N] (N) reaches into it; keep-out K forbids tracks on F.Cu; "
                    "track[3:M] (M) reaches into it (IR geometry, not DRC; KiCad's DRC reads the compiled rule areas, which know no exception)"),
        "details": {
            "kind": "ir_geometry",
            "keepouts": [{"id": "K", "layers": ["F.Cu"], "forbids": ["tracks", "vias"], "allowed_refs": [], "allowed_nets": [], "reason": "test area",
                          "compared": 5, "exempt": []}],
            "regions": [],
            "rows": [{"item": "track[1:N]", "status": "FAIL", "message": "keep-out K forbids tracks on F.Cu; track[1:N] (N) reaches into it", "keepout": "K"},
                     {"item": "track[3:M]", "status": "FAIL", "message": "keep-out K forbids tracks on F.Cu; track[3:M] (M) reaches into it", "keepout": "K"}],
            "unknown": [], "repair": "human",
        },
    },
}


def _chamfered(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """The corner board with a 45-degree chamfer (7, 3)-(9, 5) in the arc's place, a via of N, the foreign pad R3.1 at the corner (9, 3)
    with a short track of its net M toward the chamfer, and a keep-out for tracks and vias beside it: straight copper only."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 9.0), ("R3", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (12.0, 12.0))
    _limit(ir)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (7.0, 3.0)), _track("N", (7.0, 3.0), (9.0, 5.0)), _track("N", (9.0, 5.0), (9.0, 9.0)), _track("M", (9.0, 3.0), (8.4, 3.6))]
    ir.pcb.vias = [Via(net="N", x_mm=5.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)]
    ir.pcb.keepouts = [Keepout(id="K", layers=["F.Cu"], rect=user_requirement([7.5, 3.2, 1.0, 1.0], "mm"), forbids=["tracks", "vias"], reason="test area", provenance=NET_P)]
    return ir


def test_a_board_without_an_arc_is_judged_exactly_as_before(tmp_path: Path, lib: KicadLibrary):
    ir = _chamfered(tmp_path, lib)
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    got = {r.check_id: r.model_dump(mode="json", include={"status", "message", "tool", "tool_version", "details"})
           for tool in (TOOL_ID, KEEPOUT_CHECK) for r in default_registry.get(tool).validate(ir, ctx)}
    assert got == BEFORE_ARCS
    for check, before in BEFORE_ARCS.items():  # the same keys in the same order: a saved result reads byte for byte as before
        assert list(got[check]["details"]) == list(before["details"])
        assert [list(row) for row in got[check]["details"].get("violations", [])] == [list(row) for row in before["details"].get("violations", [])]
    # the same board with the chamfer rounded into the corner arc: other copper, other results, the arc version
    ir.pcb.tracks[1] = _arc("N", START, MID, END)
    after = {r.check_id: r for tool in (TOOL_ID, KEEPOUT_CHECK) for r in default_registry.get(tool).validate(ir, ctx)}
    assert {r.tool_version for r in after.values()} == {ARC_TOOL_VERSION}
    assert [(r["a"], r["b"], r["distance_mm"]) for r in after[CLEARANCE_CHECK].details["violations"]] == [("track[1:N]", "track[3:M]", 0.0), ("track[1:N]", "R3.1", 0.0)]


#: sha256 of the two ``pcb.routing`` results (check id, status, message, version and every detail) on the frozen routed boards of
#: ``tests/data/layout_baselines`` - straight copper with 45-degree chamfers, no arc - at the given clearance limit, as the checks gave
#: them before they knew arcs: the divider (17 tracks, all clear), the ATmega128 board (976 tracks, 43 violation rows at 0.3 mm) and the
#: STM32 display board (1215 tracks, 334 vias, 615 violation rows at 0.3 mm)
FROZEN_BOARD_DIGESTS = {
    ("atmega128_devboard", 0.3): "ddc697116eecb28d68a09e4e6a713cdb149b9a8c29b4770c3d124e17f1309274",
    ("divider", 0.2): "b581c5a71b8447ebc362d3498fc0e5788ae4c02d68d90b442156bcb8a649fe00",
    ("stm32_display", 0.3): "3d6823860bc84221adae864870aaa5e63d633d9260a6884ecbcf934be189f539",
}


@pytest.mark.parametrize(("name", "limit"), sorted(FROZEN_BOARD_DIGESTS))
def test_the_frozen_routed_boards_keep_every_row_and_message(name: str, limit: float):
    ir, footprints = load_geometry(name)
    assert ir.pcb.tracks and all(t.mid is None for t in ir.pcb.tracks)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(limit, note="a limit for the comparison"),
                                                    min_hole_to_edge_mm=assumption(0.4, note="a limit for the comparison"))
    board, validator = _Board(ir, footprints), RoutingValidator()
    results = [r.model_dump(mode="json") for r in (validator._connectivity(ir, board), validator._clearance(ir.pcb, board))]
    canonical = json.dumps([[r["check_id"], r["status"], r["message"], r["tool_version"], r["details"]] for r in results], ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == FROZEN_BOARD_DIGESTS[(name, limit)]
