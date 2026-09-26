"""The ``pcb.routing`` validator (``ai_eda.validation.layout``): IR geometry, never DRC.

Boards are hand-built on the synthetic fixture library of ``tests/test_routing.py``
(``Test:PAD1`` one 1.6 mm round THT pad, ``Test:SMD1`` one 1.0 mm square
SMD pad on ``F.Cu``): a track that joins two pads, a track that stops
short, a track across a foreign pad, two tracks that overlap, copper on a
net or layer the IR does not have, copper outside the outline, a through
via over an inner-layer track, a short at a 0 mm limit, copper that cannot
exist (NaN, non-positive width), pads whose copper the size box does not
bound (``Test:TRAP`` / ``Test:CUST``), pads sharing one number
(``Test:DUP1``) and a net joined only by a zone. One test hands the
router's own output to the validator: an independent
geometry check of the router's promise (every net connected, the recorded
clearance kept), with the caveat every message carries - IR geometry, not
DRC.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.ir import (
    BoardOutline,
    BoardSide,
    CircuitIR,
    Layer,
    ManufacturingConstraints,
    Net,
    PCBDesign,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Track,
    ValidationStatus as S,
    Via,
    Zone,
    assumption,
    authoritative,
)
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import RoutingParams, route_board
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import (
    CLEARANCE_CHECK,
    CONNECTIVITY_CHECK,
    NOT_COMPARED,
    TOOL_ID,
    TOOL_VERSION,
    ZONES_NOT_COMPARED,
    RoutingValidator,
    _seg_box_distance,
    _seg_point_distance,
    _seg_seg_distance,
)
from tests.conftest import DS
from tests.test_routing import board_ir, fixture_library

NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
W = 0.4


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _track(net: str, a: tuple[float, float], b: tuple[float, float], layer: str = "F.Cu", w: float = W) -> Track:
    return Track(net=net, layer=layer, start=a, end=b, width_mm=w, provenance=NET_P)


def _checks(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary | None):
    tools = {"kicad_library": lib} if lib is not None else {}
    results = default_registry.get(TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path, tools=tools))
    assert [r.check_id for r in results] == [CONNECTIVITY_CHECK, CLEARANCE_CHECK]
    for r in results:
        assert r.tool == TOOL_ID == "pcb.routing" and r.tool_version == TOOL_VERSION == "0.1" and r.details["kind"] == "ir_geometry"
        assert "not DRC" in r.message and "DRC" not in r.check_id  # never readable as a KiCad verdict
    return {r.check_id: r for r in results}


def _two_pads(tmp_path: Path, lib: KicadLibrary, **extra) -> CircuitIR:
    """R1.1 at (3, 3) and R2.1 at (9, 3) on net N, a foreign R3.1 at (6, 3) on net M when ``obstacle``, 12 x 6 mm."""
    parts = [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)]
    nets = {"N": [("R1", "1"), ("R2", "1")]}
    if extra.get("obstacle"):
        parts.append(("R3", "PAD1", 6.0, 3.0))
        nets["M"] = [("R3", "1")]
    return board_ir(tmp_path, lib, parts, nets, (12.0, 6.0))


def _limit(ir: CircuitIR, mm: float = 0.25, **more) -> None:
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(mm, note="any provenance counts: a limit is a limit"), **more)


# --------------------------------------------------------------------------- registration and the empty cases


def test_registered_for_every_domain_and_not_applicable_without_a_board(tmp_path: Path, lib: KicadLibrary):
    v = default_registry.get(TOOL_ID)
    assert isinstance(v, RoutingValidator) and v.domains == frozenset() and v.consumes == frozenset()
    ir = CircuitIR(project=ProjectMeta(id="x", name="x", workdir=str(tmp_path)))
    assert v in default_registry.select(ir)
    for check in _checks(ir, tmp_path, lib).values():
        assert check.status is S.NOT_APPLICABLE and "no ir.pcb" in check.message
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10.0, height_mm=10.0))
    for check in _checks(ir, tmp_path, lib).values():
        assert check.status is S.NOT_APPLICABLE and "no placements" in check.message
    ir = _two_pads(tmp_path, lib)
    ir.nets = []
    for check in _checks(ir, tmp_path, lib).values():
        assert check.status is S.NOT_APPLICABLE and "no nets" in check.message


def test_without_a_library_or_a_placed_footprint_nothing_is_claimed(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    for check in _checks(ir, tmp_path, None).values():
        assert check.status is S.NOT_VERIFIED and check.message.startswith("no KiCad library: pad geometry unknown")
    # a component without a placement / footprint, or with a footprint the library does not have: its pads are unknown, so is the board
    x = _two_pads(tmp_path, lib)
    x.pcb.placements = x.pcb.placements[:1]
    for check in _checks(x, tmp_path, lib).values():
        assert check.status is S.NOT_VERIFIED and check.message.startswith("pad geometry unknown: R2 has no placement") and check.details["unknown"] == ["R2 has no placement"]
    x = _two_pads(tmp_path, lib)
    x.component("R2").footprint = None
    assert _checks(x, tmp_path, lib)[CONNECTIVITY_CHECK].details["unknown"] == ["R2 has no footprint"]
    x = _two_pads(tmp_path, lib)
    x.component("R2").footprint.name = "Missing"
    assert _checks(x, tmp_path, lib)[CONNECTIVITY_CHECK].details["unknown"] == ["footprint Test:Missing of R2 was not found in a KiCad library"]


# --------------------------------------------------------------------------- connectivity


def test_a_track_joining_the_pad_centres_connects_the_net(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.message == "1 net(s) connected through IR copper (1 track(s), 0 via(s)) (IR geometry check, not DRC)"
    assert c.details["nets"] == [{"net": "N", "pads": ["R1.1", "R2.1"], "tracks": 1, "vias": 0, "status": "PASS", "message": "2 pad(s) in one copper set"}]
    assert c.details["items"] == []
    # a chain of tracks that share endpoints, ending inside the pads (not at the centres), connects too
    ir.pcb.tracks = [_track("N", (3.5, 3.0), (6.0, 3.0)), _track("N", (6.0, 3.0), (6.0, 4.0)), _track("N", (6.0, 4.0), (8.6, 4.0)), _track("N", (8.6, 4.0), (8.6, 3.2))]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    # a track that only touches the pad edge (0.8 + 0.2 = exactly 1.0 mm from the centre) is not counted: touching is not overlap
    ir.pcb.tracks = [_track("N", (4.0, 3.0), (8.0, 3.0))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"]


def test_a_gap_fails_naming_the_pads_left_out(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("C2", "PAD1", 9.0, 7.0), ("J1", "PAD1", 3.0, 7.0)],
        {"N": [("R1", "1"), ("R2", "1"), ("C2", "1"), ("J1", "1")]}, (12.0, 10.0),
    )
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (7.0, 3.0)), _track("N", (9.0, 7.0), (3.0, 7.0))]  # R1 alone; R2 alone; C2 - J1 joined
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL
    assert c.message == "1 net(s) not connected through IR copper: N: R1.1, R2.1 not connected to C2.1 (IR geometry, not DRC)"
    assert c.details["nets"][0]["unconnected"] == ["R1.1", "R2.1"]  # natural ref order (C2 < J1 < R1): the first pad is the reference, J1.1 hangs on it


def test_a_placed_unrouted_board_fails_connectivity_listing_the_nets(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 3.0, 7.0), ("R4", "PAD1", 9.0, 7.0), ("R5", "PAD1", 6.0, 5.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")], "S": [("R5", "1")], "E": []}, (12.0, 10.0),
    )
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.message.startswith("2 net(s) not connected through IR copper: A: R2.1 not connected to R1.1; B: R4.1 not connected to R3.1")
    assert {r["net"]: r["status"] for r in c.details["nets"]} == {"A": "FAIL", "B": "FAIL", "S": "NOT_APPLICABLE", "E": "NOT_APPLICABLE"}
    assert checks[CLEARANCE_CHECK].status is S.NOT_APPLICABLE and "no IR copper" in checks[CLEARANCE_CHECK].message
    # nets with fewer than two pads only: nothing to connect, nothing claimed
    ir.nets = [n for n in ir.nets if n.name in ("S", "E")]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.NOT_APPLICABLE and "no net with two or more pads" in c.message


def test_copper_on_an_unknown_net_or_layer_fails(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0)), _track("GHOST", (1.0, 1.0), (2.0, 1.0)), _track("N", (3.0, 3.0), (3.0, 4.0), layer="In1.Cu")]
    ir.pcb.vias = [Via(net="N", x_mm=3.0, y_mm=4.0, drill_mm=0.4, diameter_mm=0.8, layers=("F.Cu", "In1.Cu"), provenance=NET_P)]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["status"] == "PASS"  # the net itself is joined; the stray copper is the failure
    assert [r["message"] for r in c.details["items"]] == [
        "track[1:GHOST] names net 'GHOST', which is not in the IR",
        "track[2:N] is on layer 'In1.Cu', which ir.pcb.layers does not list",
        "via[0:N] is on layer 'In1.Cu', which ir.pcb.layers does not list",
    ]
    assert "3 track(s) / via(s) on a net or layer the IR does not have" in c.message
    # a net pin the placed footprint has no pad for is a FAIL row of that net, not a guess
    ir = _two_pads(tmp_path, lib)
    ir.nets[0].pins.append(PinRef(component_ref="R1", pin_number="7"))
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R1.7: no such pad in the placed footprint"


def test_vias_join_layers_and_pads_on_the_layer_they_are_on(tmp_path: Path, lib: KicadLibrary):
    # R1 SMD on top at (3, 4), R2 SMD on the bottom at (9, 4): a top track to a via, a bottom track from the via into R2
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM})
    via = Via(net="N", x_mm=6.0, y_mm=4.0, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)
    ir.pcb.tracks = [_track("N", (3.0, 4.0), (6.0, 4.0)), _track("N", (6.0, 4.0), (9.0, 4.0), layer="B.Cu")]
    ir.pcb.vias = [via]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    # the same bottom track on F.Cu ends over the bottom pad without touching it: R2.1 is on B.Cu only
    ir.pcb.tracks[1] = _track("N", (6.0, 4.0), (9.0, 4.0))
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"]
    # without the via the two tracks share a point on different layers: not connected
    ir.pcb.tracks[1] = _track("N", (6.0, 4.0), (9.0, 4.0), layer="B.Cu")
    ir.pcb.vias = []
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.FAIL
    # a via inside the bottom pad itself, reached by a top track, is a legal end (via-in-pad)
    ir.pcb.tracks = [_track("N", (3.0, 4.0), (9.0, 4.0))]
    ir.pcb.vias = [Via(net="N", x_mm=9.0, y_mm=4.0, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS


# --------------------------------------------------------------------------- clearance


def test_clearance_is_judged_only_against_a_recorded_limit(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib, obstacle=True)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]  # straight through R3.1 (net M)
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS  # connected, yes; clean, nobody said
    c = checks[CLEARANCE_CHECK]
    assert c.status is S.NOT_VERIFIED and c.message.startswith("no clearance limit in ir.pcb.manufacturing; the router's own clearance is a parameter, not a rule; DRC with the fab rules decides")
    assert c.details["limit_mm"] is None and c.details["violations"] == [] and c.details["not_compared"] == ["pad-to-pad (footprint / placement geometry; DRC judges it)"]
    _limit(ir, 0.25)
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["limit_mm"] == 0.25
    assert c.details["violations"] == [{"a": "track[0:N]", "b": "R3.1", "layer": "F.Cu", "distance_mm": 0.0, "limit_mm": 0.25, "status": "FAIL", "message": "track[0:N] vs R3.1 on F.Cu: 0 mm < 0.25 mm"}]
    assert c.message.startswith("1 copper pair(s) of different nets closer than 0.25 mm: track[0:N] vs R3.1 on F.Cu: 0 mm < 0.25 mm")
    # the provenance of the limit does not matter here (the capability check judges grounding)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=authoritative(0.25, DS))
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.FAIL
    # around the pad at exactly the limit: the track centre 0.8 (pad) + 0.25 (limit) + 0.2 (half width) = 1.25 mm from the pad centre passes;
    # 0.01 mm closer fails with the measured distance
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (3.0, 4.25)), _track("N", (3.0, 4.25), (9.0, 4.25)), _track("N", (9.0, 4.25), (9.0, 3.0))]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and c.details["pairs_compared"] == 3 and c.details["violations"] == []
    assert c.message.startswith("3 copper pair(s) of different nets keep >= 0.25 mm on a shared layer, 3 track(s) / 0 via(s) inside the outline")
    ir.pcb.tracks[1] = _track("N", (3.0, 4.24), (9.0, 4.24))
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"], r["distance_mm"]) for r in c.details["violations"]] == [("track[1:N]", "R3.1", 0.24)]
    # a net-less pad (no pin names it) is foreign to every net
    ir.nets = [ir.nets[0]]
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["violations"][0]["b"] == "R3.1"


def test_a_short_between_nets_is_a_zero_distance_violation(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 6.0, 1.5), ("R4", "PAD1", 6.0, 7.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 9.0),
    )
    _limit(ir)
    ir.pcb.tracks = [_track("A", (3.0, 3.0), (9.0, 3.0)), _track("B", (6.0, 1.5), (6.0, 7.0))]  # B crosses A
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL
    assert [(r["a"], r["b"], r["distance_mm"]) for r in c.details["violations"]] == [("track[0:A]", "track[1:B]", 0.0)]
    # the same crossing on different layers is no violation; a via of B under A's track is
    ir.pcb.tracks[1] = _track("B", (6.0, 1.5), (6.0, 7.0), layer="B.Cu")
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS, c.message
    ir.pcb.vias = [Via(net="B", x_mm=6.0, y_mm=3.5, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(r["a"], r["b"], r["distance_mm"]) for r in c.details["violations"]] == [("track[0:A]", "via[0:B]", 0.0)]
    # same-net copper is never compared with itself, and touching tracks of one net are fine
    ir.pcb.vias = []
    ir.pcb.tracks = [_track("A", (3.0, 3.0), (6.0, 3.0)), _track("A", (6.0, 3.0), (9.0, 3.0)), _track("B", (6.0, 1.5), (6.0, 7.0), layer="B.Cu")]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and c.details["pairs_compared"] == 6  # each A track vs R3.1 / R4.1 on F.Cu, the B.Cu track vs R1.1 / R2.1 (THT pads are on both layers)


def test_copper_outside_the_outline_and_via_holes_near_the_edge_fail(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0)), _track("N", (9.0, 3.0), (11.9, 3.0))]  # 11.9 + 0.2 > 12
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["limit_mm"] is None  # FAIL even without a clearance limit
    assert [r["message"] for r in c.details["outline"]] == ["track[1:N] copper leaves the 12 x 6 mm outline at (0, 0)"]
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    ir.pcb.vias = [Via(net="N", x_mm=0.5, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)]
    _limit(ir, 0.25, min_hole_to_edge_mm=assumption(0.4, note="x"))
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [r["message"] for r in c.details["outline"]] == ["via[0:N] hole edge 0.3 mm from the outline < min_hole_to_edge_mm 0.4"]
    assert c.details["hole_to_edge_limit_mm"] == 0.4
    ir.pcb.vias[0].x_mm = 0.6  # hole edge at 0.4: allowed
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS
    # a board without an outline: pairs are judged, the edge is not, and the result says so
    ir.pcb.outline = None
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.NOT_VERIFIED and "no outline: edge distances not judged" in c.message


def test_a_through_via_is_copper_on_every_layer_of_the_board(tmp_path: Path, lib: KicadLibrary):
    """The compiler writes a via without a type (a through via in KiCad), so on a four-layer IR a track of another net on In1.Cu under it is a short."""
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 6.0, 1.0), ("R4", "PAD1", 6.0, 9.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0),
    )
    ir.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="signal"), Layer(name="In2.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")]
    _limit(ir)
    ir.pcb.tracks = [_track("A", (3.0, 3.0), (6.0, 3.0)), _track("A", (6.0, 3.0), (9.0, 3.0), layer="B.Cu"), _track("B", (6.0, 1.0), (6.0, 9.0), layer="In1.Cu")]
    ir.pcb.vias = [Via(net="A", x_mm=6.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, layers=("F.Cu", "B.Cu"), provenance=NET_P)]
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS
    c = checks[CLEARANCE_CHECK]
    assert c.status is S.FAIL, c.message
    assert [(r["a"], r["b"], r["layer"], r["distance_mm"]) for r in c.details["violations"]] == [("track[2:B]", "via[0:A]", "In1.Cu", 0.0)]
    # the In1.Cu track also joins B's THT pads (on every copper layer) and the via joins A's B.Cu track: connectivity as before
    ir.pcb.tracks[2] = _track("B", (6.0, 1.0), (6.0, 9.0), layer="In2.Cu")
    assert [r["layer"] for r in _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].details["violations"]] == ["In2.Cu"]
    # moved off the via: no violation on any layer
    ir.pcb.tracks[2] = _track("B", (6.0, 1.0), (4.0, 5.0), layer="In1.Cu")
    ir.pcb.tracks.append(_track("B", (4.0, 5.0), (6.0, 9.0), layer="In1.Cu"))
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS


def test_a_dead_short_fails_whatever_the_limit(tmp_path: Path, lib: KicadLibrary):
    """``min_clearance_mm`` 0 (which a fab file's '0 mm' grounds): overlapping copper of two nets is still a short, never ``0 >= 0``."""
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 6.0, 1.5), ("R4", "PAD1", 6.0, 7.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 9.0),
    )
    _limit(ir, 0.0)
    ir.pcb.tracks = [_track("A", (3.0, 3.0), (9.0, 3.0)), _track("B", (6.0, 1.5), (6.0, 7.0))]  # B crosses A
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["limit_mm"] == 0.0
    assert [(r["a"], r["b"], r["distance_mm"], r["message"]) for r in c.details["violations"]] == [
        ("track[0:A]", "track[1:B]", 0.0, "track[0:A] vs track[1:B] on F.Cu: copper overlaps (a short, whatever the 0 mm limit)"),
    ]
    # touching copper (edge to edge, distance exactly 0) is a short too; 0.01 mm apart is not, at a 0 mm limit
    ir.pcb.tracks[1] = _track("B", (6.0, 1.5), (6.0, 2.6))  # copper 2.4..2.8 vs A's 2.8..3.2
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.FAIL
    ir.pcb.tracks[1] = _track("B", (6.0, 1.5), (6.0, 2.59))
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS


def test_copper_that_cannot_exist_is_a_fail_row_never_a_pass(tmp_path: Path, lib: KicadLibrary):
    """A non-finite coordinate or a non-positive width (which the compiler refuses) joins nothing and is compared with nothing: both checks FAIL naming the item."""
    ir = _two_pads(tmp_path, lib, obstacle=True)
    _limit(ir)
    # a negative width used to turn the half-width subtraction into an addition: a track straight through R3.1 (net M) passed both checks
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0), w=-1.0)]
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["R2.1"]  # the bad track joins nothing
    assert [r["message"] for r in c.details["items"]] == ["track[0:N] has non-positive width -1 mm: no copper (the compiler refuses it too)"]
    assert "1 track(s) / via(s) that cannot be copper: track[0:N] has non-positive width -1 mm" in c.message
    d = checks[CLEARANCE_CHECK]
    assert d.status is S.FAIL and d.message.startswith("1 track(s) / via(s) that cannot be copper: track[0:N] has non-positive width -1 mm") and d.details["malformed"] == c.details["items"]
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0), w=0.0)]
    assert {k: v.status for k, v in _checks(ir, tmp_path, lib).items()} == {CONNECTIVITY_CHECK: S.FAIL, CLEARANCE_CHECK: S.FAIL}
    # a NaN coordinate compares false with everything: silently skipped before, a FAIL row now (no file can carry one; an in-process IR can)
    nan = float("nan")
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0)), _track("N", (nan, 5.0), (9.0, 5.0))]
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.FAIL and checks[CONNECTIVITY_CHECK].details["nets"][0]["status"] == "PASS"  # the good track joins the pads
    assert [r["message"] for r in checks[CONNECTIVITY_CHECK].details["items"]] == [
        "track[1:N] has a non-finite coordinate or width (start (nan, 5.0), end (9.0, 5.0), width 0.4): its copper cannot be measured",
    ]
    assert checks[CLEARANCE_CHECK].status is S.FAIL and "track[1:N] has a non-finite coordinate" in checks[CLEARANCE_CHECK].message
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    ir.pcb.vias = [Via(net="N", x_mm=nan, y_mm=5.0, drill_mm=0.4, diameter_mm=0.8, provenance=NET_P)]
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.FAIL and checks[CLEARANCE_CHECK].status is S.FAIL
    assert checks[CONNECTIVITY_CHECK].details["items"][0]["message"].startswith("via[0:N] has a non-finite coordinate or size")
    ir.pcb.vias = [Via(net="N", x_mm=6.0, y_mm=5.0, drill_mm=0.8, diameter_mm=0.8, provenance=NET_P)]  # no annular ring: the compiler refuses it
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.FAIL and checks[CONNECTIVITY_CHECK].details["items"][0]["message"] == "via[0:N] needs diameter 0.8 mm > drill 0.8 mm > 0: no copper (the compiler refuses it too)"


def test_pads_and_vias_with_copper_the_size_box_does_not_bound_are_not_verified(tmp_path: Path, lib: KicadLibrary):
    """A trapezoid (rect_delta) or custom (primitives) pad: its copper is unknown to the library reader, so nothing is claimed about the board."""
    for name, shape in (("TRAP", "trapezoid"), ("CUST", "custom")):
        ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 10.0, 4.0), ("U1", name, 6.0, 4.0)], {"A": [("R1", "1"), ("R2", "1")], "B": [("U1", "1")]}, (12.0, 8.0))
        _limit(ir)
        ir.pcb.tracks = [_track("A", (2.0, 4.0), (2.0, 6.25)), _track("A", (2.0, 6.25), (10.0, 6.25)), _track("A", (10.0, 6.25), (10.0, 4.0))]  # 0.75 mm from the size box, inside the real copper
        for check in _checks(ir, tmp_path, lib).values():
            assert check.status is S.NOT_VERIFIED, check
            assert check.message.startswith(f"pad geometry unknown: pad U1.1 of footprint Test:{name} has shape '{shape}', whose copper is not bounded by its (size) box")
            assert check.details["unknown"] == [f"pad U1.1 of footprint Test:{name} has shape '{shape}', whose copper is not bounded by its (size) box (custom primitives / trapezoid rect_delta are not read)"]


def test_pads_sharing_a_number_are_one_logical_pad(tmp_path: Path, lib: KicadLibrary):
    """``Test:DUP1`` has two pads "1" (KiCad: internally connected); copper reaching one of them reaches U1.1, and the label appears once."""
    ir = board_ir(tmp_path, lib, [("U1", "DUP1", 6.0, 3.0), ("R1", "PAD1", 1.5, 3.0)], {"N": [("U1", "1"), ("R1", "1")]}, (12.0, 6.0))
    ir.pcb.tracks = [_track("N", (1.5, 3.0), (4.5, 3.0))]  # ends in the first pad "1" only
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.PASS, c.message
    assert c.details["nets"] == [{"net": "N", "pads": ["R1.1", "U1.1"], "tracks": 1, "vias": 0, "status": "PASS", "message": "2 pad(s) in one copper set"}]
    ir.pcb.tracks = [_track("N", (1.5, 3.0), (3.0, 3.0))]  # reaches neither
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["unconnected"] == ["U1.1"] and c.message.startswith("1 net(s) not connected through IR copper: N: U1.1 not connected to R1.1")


def test_a_net_joined_only_by_a_copper_pour_is_not_verified_never_fail(tmp_path: Path, lib: KicadLibrary):
    """A zone is IR copper the compiler emits and KiCad fills; whether the fill reaches a pad is not decided by the polygon, so the net is NOT_VERIFIED."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 3.0, 7.0), ("R4", "PAD1", 9.0, 7.0)],
                  {"GND": [("R1", "1"), ("R2", "1")], "K": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0))
    ir.pcb.zones = [Zone(net="GND", layer="F.Cu", polygon=[(0.5, 0.5), (11.5, 0.5), (11.5, 5.5), (0.5, 5.5)], provenance=NET_P)]
    ir.pcb.tracks = [_track("K", (3.0, 7.0), (9.0, 7.0))]
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.NOT_VERIFIED, c.message
    assert c.message == ("1 net(s) joined only by a copper pour, which this check cannot judge: GND: R2.1 not joined to R1.1 by tracks / vias; "
                         "whether the copper pour zone[0:GND] reaches them is decided by KiCad's fill and DRC, not by the polygon (IR geometry, not DRC)")
    assert {r["net"]: r["status"] for r in c.details["nets"]} == {"GND": "NOT_VERIFIED", "K": "PASS"} and c.details["nets"][0]["zones"] == ["zone[0:GND]"] and c.details["zones"] == 1
    d = checks[CLEARANCE_CHECK]
    assert d.details["not_compared"] == [*NOT_COMPARED, ZONES_NOT_COMPARED] and d.status is S.NOT_VERIFIED  # no limit recorded
    # a FAIL elsewhere still outranks it; tracks that do join the pads make the pour irrelevant (PASS)
    ir.pcb.tracks = []
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and {r["net"]: r["status"] for r in c.details["nets"]} == {"GND": "NOT_VERIFIED", "K": "FAIL"}
    ir.pcb.tracks = [_track("K", (3.0, 7.0), (9.0, 7.0)), _track("GND", (3.0, 3.0), (9.0, 3.0))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and c.details["nets"][0]["status"] == "PASS"
    # without a zone the same unrouted net is a FAIL, as before
    ir.pcb.zones = []
    ir.pcb.tracks = [_track("K", (3.0, 7.0), (9.0, 7.0))]
    c = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and c.details["nets"][0]["message"] == "R2.1 not connected to R1.1" and c.details["zones"] == 0


# --------------------------------------------------------------------------- geometry helpers


def test_distance_helpers_are_exact():
    assert _seg_point_distance((0.0, 0.0), (4.0, 0.0), (2.0, 3.0)) == 3.0
    assert _seg_point_distance((0.0, 0.0), (4.0, 0.0), (7.0, 4.0)) == 5.0  # beyond the end: to the endpoint
    assert _seg_point_distance((1.0, 1.0), (1.0, 1.0), (4.0, 5.0)) == 5.0  # degenerate segment
    assert _seg_seg_distance((0.0, 0.0), (4.0, 0.0), (2.0, -1.0), (2.0, 1.0)) == 0.0  # a proper crossing
    assert _seg_seg_distance((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 1.0)) == 0.0  # a T: endpoint on the segment
    assert _seg_seg_distance((0.0, 0.0), (4.0, 0.0), (5.0, 0.0), (8.0, 0.0)) == 1.0  # collinear, apart
    assert _seg_seg_distance((0.0, 0.0), (4.0, 0.0), (0.0, 2.0), (4.0, 2.0)) == 2.0  # parallel
    assert _seg_seg_distance((0.0, 0.0), (4.0, 0.0), (1.0, 0.0), (3.0, 0.0)) == 0.0  # collinear overlap
    box = (2.0, 2.0, 4.0, 4.0)
    assert _seg_box_distance((0.0, 3.0), (1.0, 3.0), box) == 1.0
    assert _seg_box_distance((0.0, 0.0), (1.0, 1.0), box) == pytest.approx(2**0.5)
    assert _seg_box_distance((0.0, 3.0), (6.0, 3.0), box) == 0.0  # through
    assert _seg_box_distance((2.5, 2.5), (3.5, 3.5), box) == 0.0  # inside
    assert _seg_box_distance((0.0, 0.0), (3.0, 3.0), box) == 0.0  # ends inside


# --------------------------------------------------------------------------- the router's promise, checked independently


def test_the_maze_routers_output_keeps_its_promise_on_a_crowded_board(tmp_path: Path, lib: KicadLibrary):
    """Every net the router reports as routed is connected, and its copper keeps the router's clearance from every foreign item."""
    ir = board_ir(
        tmp_path, lib,
        [("R1", "SMD2", 2.5, 4.0), ("R2", "SMD2", 10.5, 4.0), ("W1", "WALL2", 6.5, 8.0), ("R3", "PAD2", 2.5, 12.0), ("R4", "PAD2", 10.5, 12.0), ("R5", "PAD1", 6.5, 2.0)],
        {"N": [("R1", "2"), ("R2", "1")], "K": [("R3", "2"), ("R4", "1")], "X": [("R1", "1"), ("R3", "1")], "Y": [("R2", "2"), ("R4", "2"), ("R5", "1")]}, (14.0, 16.0),
    )
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.vias, r.stats
    ir.pcb.tracks = list(r.tracks)
    ir.pcb.vias = list(r.vias)
    _limit(ir, RoutingParams().clearance_mm)
    checks = _checks(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.PASS and {row["net"]: row["status"] for row in c.details["nets"]} == {"N": "PASS", "K": "PASS", "X": "PASS", "Y": "PASS"}
    d = checks[CLEARANCE_CHECK]
    assert d.status is S.PASS, d.message
    assert d.details["pairs_compared"] > 0 and d.details["violations"] == [] and d.details["outline"] == []
    # the IR_BUILD path: the registry runs it with the other validators, stamped like them
    results = default_registry.run(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert {res.check_id: res.status for res in results if res.check_id.startswith("pcb.routing.")} == {CONNECTIVITY_CHECK: S.PASS, CLEARANCE_CHECK: S.PASS}
    # a hand edit that moves one track onto a foreign pad is caught
    ir.pcb.tracks[0] = _track("N", (2.5, 4.0), (6.5, 2.0))  # into R5.1 (net Y)
    checks = _checks(ir, tmp_path, lib)
    assert checks[CLEARANCE_CHECK].status is S.FAIL and any(v["b"] == "R5.1" for v in checks[CLEARANCE_CHECK].details["violations"])


def test_layers_of_the_ir_decide_where_a_through_hole_pad_is(tmp_path: Path, lib: KicadLibrary):
    """A THT pad is on every copper layer the IR lists; a track of another net over it on B.Cu violates the clearance just as on F.Cu."""
    ir = _two_pads(tmp_path, lib, obstacle=True)
    _limit(ir)
    ir.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")]
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0), layer="B.Cu")]
    c = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and c.details["violations"][0]["layer"] == "B.Cu"
    # an SMD pad is on the copper layer it lists only: a B.Cu track under a front SMD pad of another net is no violation
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("S1", "SMD1", 6.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")], "M": [("S1", "1")]}, (12.0, 6.0))
    _limit(ir)
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0), layer="B.Cu")]
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0))]
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.FAIL
    # ... unless the part sits on the bottom side, where the mirrored pad is on B.Cu
    ir.pcb.placements[2] = Placement(component_ref="S1", x_mm=6.0, y_mm=3.0, side=BoardSide.BOTTOM, provenance=NET_P)
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.PASS
    ir.pcb.tracks = [_track("N", (3.0, 3.0), (9.0, 3.0), layer="B.Cu")]
    assert _checks(ir, tmp_path, lib)[CLEARANCE_CHECK].status is S.FAIL
