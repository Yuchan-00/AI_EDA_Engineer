"""The deterministic maze router (``ai_eda.tools.routing.maze``).

Every test runs on a synthetic KiCad library written into ``tmp_path``: the
``Test:VR1`` symbol / ``Test:FP`` footprint of ``tests/test_parts_existence.py``
plus the through-hole and SMD fixture footprints written by
:func:`fixture_library` below (``Test:PAD1`` one 1.6 mm round THT pad,
``Test:SMD1`` one 1.0 mm square SMD pad on ``F.Cu``, ``Test:WALL`` a net-less
0.5 x 20 mm SMD pad on ``F.Cu``, ``Test:CAGE`` a THT pad fenced by four
unnumbered THT bars, ``Test:TINY`` a 0.2 mm SMD pad, ``Test:PAD2`` / ``Test:SMD2`` / ``Test:WALL2``
two-pad versions of the first three, ``Test:SMD054`` a 0.54 x 0.64 mm and
``Test:SMD0510`` a 0.5 x 1.0 mm SMD pad, ``Test:TRAP`` a trapezoid,
``Test:CUST`` a custom-shape pad, ``Test:CUSTX`` a custom pad with a primitive the
library reader does not read (custom pads themselves: ``tests/test_routing_custom_pads.py``),
``Test:DUP1`` two pads numbered "1", ``Test:VBAR`` /
``Test:HBAR`` net-less 0.5 x 4 mm F.Cu bars and ``Test:BPLANE`` a net-less 60 x 60 mm
B.Cu pad - a copper plane that leaves one routable layer and no via site). Boards are a few
millimetres, so the grids are small. Nothing here claims DRC: the router's
clearances are its parameters, and the tests check the IR geometry it
promised (the ``pcb.routing.*`` validators do the same in the pipeline).

Router 0.2 negotiates congestion (rip-up and reroute). What 0.1 - nets one
after another, each against the copper of the nets before it, no rip-up -
would do is reproduced by :func:`first_come`: on a board without a via site
it is exactly 0.1's pass (same search, same costs, same tie-breaking), so a
board it cannot finish in either net order is a board 0.1 could not route.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.errors import CompileError
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
    Via,
    assumption,
    authoritative,
)
from ai_eda.ir.pcb import UNRECORDED_ORIGIN
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import ROUTER_ID, ROUTER_VERSION, Routing, RoutingParams, effective_params, route_board
from ai_eda.tools.routing.maze import BLOCKED, LAYERS, _Board, _Negotiation, _route_net
from tests.conftest import DS
from tests.test_parts_existence import make_part, synthetic_library

NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
P = RoutingParams()
#: the negotiation's numbers, recorded after the geometry and the search costs in every ``params:`` entry
NEGOTIATION_ENTRY = "base_cost=1.0,history_cost=1.0,present_cost=0.5,present_growth=2.0,max_iterations=40,window=10.0"
DEFAULT_ENTRY = f"params:grid=0.25,width=0.4,clearance=0.25,via=0.8/0.4,edge=0.3,via_cost=12.0,bend_cost=0.6,{NEGOTIATION_ENTRY}"
#: the router's own promise: track centre to foreign pad edge >= clearance + width / 2
PAD_KEEPOUT = P.clearance_mm + P.track_width_mm / 2

_FOOTPRINTS = {
    "PAD1": '(pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
    "SMD1": '(pad "1" smd rect (at 0 0) (size 1.0 1.0) (layers "F.Cu" "F.Mask" "F.Paste"))',
    "WALL": '(pad "1" smd rect (at 0 0) (size 0.5 20.0) (layers "F.Cu" "F.Mask" "F.Paste"))',  # taller than any test board: cuts F.Cu
    "CAGE": (
        '(pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n'
        '  (pad "" thru_hole rect (at -1.4 0) (size 0.4 3.2) (drill 0.3) (layers "*.Cu" "*.Mask"))\n'
        '  (pad "" thru_hole rect (at 1.4 0) (size 0.4 3.2) (drill 0.3) (layers "*.Cu" "*.Mask"))\n'
        '  (pad "" thru_hole rect (at 0 -1.4) (size 3.2 0.4) (drill 0.3) (layers "*.Cu" "*.Mask"))\n'
        '  (pad "" thru_hole rect (at 0 1.4) (size 3.2 0.4) (drill 0.3) (layers "*.Cu" "*.Mask"))'
    ),
    "TINY": '(pad "1" smd rect (at 0 0) (size 0.2 0.2) (layers "F.Cu" "F.Mask" "F.Paste"))',
    # two-pad parts (pins 1 and 2, as the Test:VR1 symbol has) for the test that compiles the board
    "PAD2": '(pad "1" thru_hole circle (at -1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n  (pad "2" thru_hole circle (at 1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
    "SMD2": '(pad "1" smd rect (at -1.5 0) (size 1.0 1.0) (layers "F.Cu" "F.Mask" "F.Paste"))\n  (pad "2" smd rect (at 1.5 0) (size 1.0 1.0) (layers "F.Cu" "F.Mask" "F.Paste"))',
    "WALL2": '(pad "1" smd rect (at -1 0) (size 0.5 20.0) (layers "F.Cu" "F.Mask" "F.Paste"))\n  (pad "2" smd rect (at 1 0) (size 0.5 20.0) (layers "F.Cu" "F.Mask" "F.Paste"))',
    # small SMD pads (0402-like) whose terminal cell can fall inside a neighbour's or the edge's keep-out
    "SMD054": '(pad "1" smd rect (at 0 0) (size 0.54 0.64) (layers "F.Cu" "F.Mask" "F.Paste"))',
    "SMD0510": '(pad "1" smd rect (at 0 0) (size 0.5 1.0) (layers "F.Cu" "F.Mask" "F.Paste"))',
    # copper beyond the (size) box: a trapezoid (rect_delta) and a custom pad (primitives), as the official libraries have
    "TRAP": '(pad "1" smd trapezoid (at 0 0) (size 2 2) (rect_delta 1.8 0) (layers "F.Cu" "F.Mask" "F.Paste"))',
    "CUST": (
        '(pad "1" smd custom (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask" "F.Paste")\n'
        '    (primitives (gr_poly (pts (xy -2 -2) (xy 2 -2) (xy 2 2) (xy -2 2)) (width 0) (fill yes))))'
    ),
    # a custom pad with a primitive head the library reader does not know: its copper is unknown
    "CUSTX": (
        '(pad "1" smd custom (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask" "F.Paste")\n'
        '    (primitives (gr_blob (pts (xy -2 -2) (xy 2 -2) (xy 2 2)) (width 0))))'
    ),
    # two pads that share the number "1" (split thermal / mounting pads): one logical pad in KiCad
    "DUP1": '(pad "1" thru_hole circle (at -1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n  (pad "1" thru_hole circle (at 1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
    # net-less F.Cu bars and a B.Cu plane (larger than any test board): single-layer boards without a via site
    "VBAR": '(pad "" smd rect (at 0 0) (size 0.5 4.0) (layers "F.Cu" "F.Mask"))',
    "HBAR": '(pad "" smd rect (at 0 0) (size 4.0 0.5) (layers "F.Cu" "F.Mask"))',
    "BPLANE": '(pad "" smd rect (at 0 0) (size 60 60) (layers "B.Cu"))',
}


def fixture_library(root: Path) -> KicadLibrary:
    """The synthetic library plus the fixture footprints of the module docstring."""
    lib = synthetic_library(root)
    pretty = root / "footprints" / "Test.pretty"
    for name, pads in _FOOTPRINTS.items():
        (pretty / f"{name}.kicad_mod").write_text(
            f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr through_hole)\n'
            f'  (fp_rect (start -1 -1) (end 1 1) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
            f"  {pads})\n",
            encoding="utf-8",
        )
    return lib


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


Part = tuple[str, str, float, float]  # ref, footprint, x, y


def board_ir(
    tmp_path: Path,
    lib: KicadLibrary,
    parts: list[Part],
    nets: dict[str, list[tuple[str, str]]],
    size: tuple[float, float],
    *,
    sides: dict[str, BoardSide] | None = None,
) -> CircuitIR:
    """Verified Test parts at the given board positions (top side unless ``sides`` says), the nets and a rectangular outline at (0, 0)."""
    ir = CircuitIR(project=ProjectMeta(id="route", name="route", workdir=str(tmp_path)))
    placements = []
    for ref, fp, x, y in parts:
        c = make_part(ref, footprint=fp)
        c.symbol = lib.resolve_symbol(c.symbol)
        c.footprint = lib.resolve_footprint(c.footprint)
        assert c.footprint.verified, fp
        ir.components.append(c)
        placements.append(Placement(component_ref=ref, x_mm=x, y_mm=y, side=(sides or {}).get(ref, BoardSide.TOP), provenance=NET_P))
    for name, pins in nets.items():
        ir.nets.append(Net(name=name, pins=[PinRef(component_ref=r, pin_number=n) for r, n in pins], provenance=NET_P))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=size[0], height_mm=size[1]), placements=placements)
    return ir


def _connected(tracks: list[Track], points: list[tuple[float, float]]) -> bool:
    """Whether ``points`` all lie in one chain of tracks that share endpoints exactly (grid points and pad centres)."""
    parent: dict[tuple[float, float], tuple[float, float]] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tracks:
        parent[find(t.start)] = find(t.end)
    return len({find(pt) for pt in points}) == 1 and all(pt in parent for pt in points)


def _segment_box_distance(a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    """Distance from the segment a-b to the axis-aligned box (x1, y1, x2, y2), sampled every 0.01 mm (test helper)."""
    x1, y1, x2, y2 = box
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(1, int(length / 0.01))
    best = math.inf
    for s in range(steps + 1):
        x = a[0] + (b[0] - a[0]) * s / steps
        y = a[1] + (b[1] - a[1]) * s / steps
        best = min(best, math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)))
    return best


def _geometry(r: Routing) -> list:
    return [design_data(t) for t in r.tracks] + [design_data(v) for v in r.vias]


# --------------------------------------------------------------------------- the router


def test_two_pads_one_net_one_straight_track_ending_at_the_pad_centres(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    before = ir.content_hash()
    r = route_board(ir, lib)
    assert ir.content_hash() == before and ir.pcb.tracks == [] and ir.pcb.vias == []  # pure: the caller stores the result
    assert r.unrouted == {} and r.vias == []
    assert [(t.net, t.layer, t.start, t.end, t.width_mm) for t in r.tracks] == [("N", "F.Cu", (3.0, 3.0), (9.0, 3.0), 0.4)]
    assert r.stats["routed_nets"] == 1 and r.stats["net_length_mm"] == {"N": 6.0} and r.stats["total_length_mm"] == 6.0
    assert r.stats["track_count"] == 1 and r.stats["via_count"] == 0 and r.stats["skipped_nets"] == [] and r.stats["raised"] == {}
    assert r.stats["grid"] == (49, 25, 49 * 25) and r.params == RoutingParams()
    # an off-grid pad centre: the path ends at the nearest grid cell and one stub joins it to the exact centre
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.1, 3.0), ("R2", "PAD1", 9.0, 3.05)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and all(t.layer == "F.Cu" for t in r.tracks)
    assert _connected(r.tracks, [(3.1, 3.0), (9.0, 3.05)])
    assert (3.0, 3.0) in {t.start for t in r.tracks} | {t.end for t in r.tracks}  # the terminal cell
    assert all(t.start != t.end for t in r.tracks)  # never a zero-length track
    assert sorted((t.start, t.end) for t in r.tracks if math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) < 0.2) == [((3.0, 3.0), (3.1, 3.0)), ((9.0, 3.0), (9.0, 3.05))]


def test_every_track_and_via_is_traced_to_the_router_the_net_the_placements_and_the_parameters(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM})
    r = route_board(ir, lib)
    assert r.unrouted == {} and len(r.vias) == 1, r.stats  # top SMD to bottom SMD: exactly one layer change
    assert ROUTER_ID == "routing.maze" and ROUTER_VERSION == "0.2"
    assert r.stats["iterations"] == 1 and r.stats["legal"] is True and r.stats["dropped"] == [] and r.stats["recovered"] == []
    for item in [*r.tracks, *r.vias]:
        prov = item.provenance
        assert prov.kind is ProvenanceKind.DERIVED and not prov.needs_verification and prov.note != UNRECORDED_ORIGIN
        assert prov.tool == ROUTER_ID and prov.tool_version == ROUTER_VERSION
        assert prov.derived_from == ["net:N", "placement:R1", "placement:R2", DEFAULT_ENTRY]
        assert prov.inputs == {}  # the calculator role map stays empty
        assert "DRC" in prov.note and "IR geometry" in prov.note
        assert "(negotiated-congestion route, 1 iteration(s))" in prov.note  # the iteration count is named
    via_at = (r.vias[0].x_mm, r.vias[0].y_mm)
    assert r.vias[0].layers == ("F.Cu", "B.Cu") and any(t.layer == "B.Cu" for t in r.tracks)  # the via sits beside the bottom pad, a B.Cu track enters it
    assert r.vias[0].drill_mm == 0.4 and r.vias[0].diameter_mm == 0.8
    assert all(t.layer == "F.Cu" for t in r.tracks if (3.0, 4.0) in (t.start, t.end))
    assert any(t.layer == "B.Cu" and (9.0, 4.0) in (t.start, t.end) for t in r.tracks)
    assert any(via_at in (t.start, t.end) for t in r.tracks)  # the via joins the copper


def test_a_foreign_pad_in_the_way_is_kept_at_clearance(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 4.0), ("R2", "PAD1", 9.0, 4.0), ("R3", "PAD1", 6.0, 4.0)],
        {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (12.0, 8.0),
    )
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.vias == [] and r.stats["skipped_nets"] == ["M"]  # around, not through, and not a via
    assert _connected(r.tracks, [(3.0, 4.0), (9.0, 4.0)]) and all(t.net == "N" and t.layer == "F.Cu" for t in r.tracks)
    box = (6.0 - 0.8, 4.0 - 0.8, 6.0 + 0.8, 4.0 + 0.8)
    for t in r.tracks:
        assert _segment_box_distance(t.start, t.end, box) >= PAD_KEEPOUT - 1e-9, t
    assert r.stats["net_length_mm"]["N"] > 6.0  # the detour is real copper


def test_an_smd_wall_on_the_front_forces_exactly_two_vias(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0), ("W1", "WALL", 6.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and len(r.vias) == 2 and r.stats["via_count"] == 2, r.stats
    assert _connected(r.tracks, [(3.0, 4.0), (9.0, 4.0)])
    back = [t for t in r.tracks if t.layer == "B.Cu"]
    front = [t for t in r.tracks if t.layer == "F.Cu"]
    assert any(min(t.start[0], t.end[0]) < 6.0 < max(t.start[0], t.end[0]) for t in back)  # the crossing is on B.Cu
    assert all(max(t.start[0], t.end[0]) < 6.0 - 0.25 - PAD_KEEPOUT + 1e-9 or min(t.start[0], t.end[0]) > 6.0 + 0.25 + PAD_KEEPOUT - 1e-9 for t in front)
    for v in r.vias:
        assert v.net == "N" and v.layers == ("F.Cu", "B.Cu")
        assert min(v.x_mm, 12.0 - v.x_mm, v.y_mm, 8.0 - v.y_mm) >= P.edge_clearance_mm + P.via_diameter_mm / 2 - 1e-9
        assert _segment_box_distance((v.x_mm, v.y_mm), (v.x_mm, v.y_mm), (5.75, -6.0, 6.25, 14.0)) >= P.via_diameter_mm / 2 + P.clearance_mm - 1e-9
    # every via joins front copper (a track end or a pad centre) to back copper
    for v in r.vias:
        at = (v.x_mm, v.y_mm)
        assert any(at in (t.start, t.end) for t in front) or at in ((3.0, 4.0), (9.0, 4.0))
        assert any(at in (t.start, t.end) for t in back)


def test_tracks_stay_inside_the_outline_minus_the_edge_clearance(tmp_path: Path, lib: KicadLibrary):
    # a pad right at the corner: the path leaves it inward, never along the edge keep-out
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 1.0, 1.0), ("R2", "PAD1", 7.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (8.0, 6.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and _connected(r.tracks, [(1.0, 1.0), (7.0, 5.0)])
    limit = P.edge_clearance_mm + P.track_width_mm / 2
    for t in r.tracks:
        for x, y in (t.start, t.end):
            if (x, y) in ((1.0, 1.0), (7.0, 5.0)):
                continue  # the pad centres themselves (the stubs end there)
            assert min(x, 8.0 - x, y, 6.0 - y) >= limit - 1e-9, t
    # the board's own edge keep-out: cells closer than edge_clearance + width/2 are BLOCKED on both layers
    board = _Board(ir, lib, P)
    for k in range(board.n):
        x, y = board.pos(k)
        if min(x, 8.0 - x, y, 6.0 - y) < limit - 1e-9:
            assert board.owner[0][k] == BLOCKED and board.owner[1][k] == BLOCKED


def test_same_ir_same_copper(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0), ("W1", "WALL", 6.0, 4.0), ("R3", "PAD1", 3.0, 7.0), ("R4", "PAD1", 9.0, 7.0)],
        {"N": [("R1", "1"), ("R2", "1")], "K": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0),
    )
    a = route_board(ir, lib)
    b = route_board(copy.deepcopy(ir), fixture_library(tmp_path / "kicad2"))
    assert a.unrouted == {} and _geometry(a) == _geometry(b) and a.stats == b.stats and a.params == b.params
    assert [t.net for t in a.tracks] == sorted(t.net for t in a.tracks)  # net order (pad count, name), then path order
    assert design_data(PCBDesign(tracks=a.tracks, vias=a.vias)) == design_data(PCBDesign(tracks=b.tracks, vias=b.vias))


def test_parameters_are_raised_to_the_fab_minimums_and_recorded(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM})
    ir.pcb.manufacturing = ManufacturingConstraints(
        fab="JLCPCB",
        min_track_width_mm=assumption(0.5, note="fab page not read"),  # any provenance counts: a limit is a limit
        min_clearance_mm=authoritative(0.2, DS),  # below the parameter: nothing to raise
        min_via_drill_mm=assumption(0.5, note="idem"),
        min_via_diameter_mm=assumption(1.0, note="idem"),
        min_hole_to_edge_mm=assumption(0.6, note="idem"),
    )
    eff, raised = effective_params(ir)
    assert raised == {"track_width_mm": (0.4, 0.5), "via_drill_mm": (0.4, 0.5), "via_diameter_mm": (0.8, 1.0), "edge_clearance_mm": (0.3, 0.6 - 0.25)}
    assert eff == RoutingParams(track_width_mm=0.5, via_drill_mm=0.5, via_diameter_mm=1.0, edge_clearance_mm=0.35)
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.params == eff and len(r.vias) == 1
    assert all(t.width_mm == 0.5 for t in r.tracks) and r.vias[0].drill_mm == 0.5 and r.vias[0].diameter_mm == 1.0
    assert r.stats["raised"] == {"track_width_mm": [0.4, 0.5], "via_drill_mm": [0.4, 0.5], "via_diameter_mm": [0.8, 1.0], "edge_clearance_mm": [0.3, 0.35]}
    assert r.tracks[0].provenance.derived_from[-1] == f"params:grid=0.25,width=0.5,clearance=0.25,via=1.0/0.5,edge=0.35,via_cost=12.0,bend_cost=0.6,{NEGOTIATION_ENTRY}"
    # a caller's own parameters are raised the same way, and effective_params never lowers one
    eff, raised = effective_params(ir, RoutingParams(track_width_mm=0.6, clearance_mm=0.1))
    assert eff.track_width_mm == 0.6 and eff.clearance_mm == 0.2 and raised == {"clearance_mm": (0.1, 0.2), "via_drill_mm": (0.4, 0.5), "via_diameter_mm": (0.8, 1.0), "edge_clearance_mm": (0.3, 0.35)}
    # a drill limit at or above the via diameter is refused, not guessed around
    ir.pcb.manufacturing = ManufacturingConstraints(min_via_drill_mm=assumption(0.8, note="x"))
    with pytest.raises(CompileError, match="via drill to 0.8 mm, which is not below the via diameter 0.8 mm"):
        effective_params(ir)
    # nonsense parameters are refused, the negotiation's included
    for bad in (
        RoutingParams(grid_mm=0.0), RoutingParams(track_width_mm=-1.0), RoutingParams(clearance_mm=-0.1), RoutingParams(via_diameter_mm=0.4, via_drill_mm=0.4),
        RoutingParams(bend_cost=-1.0), RoutingParams(base_cost=0.0), RoutingParams(history_cost=-1.0), RoutingParams(present_cost=math.nan),
        RoutingParams(present_growth=0.9), RoutingParams(max_iterations=0), RoutingParams(max_iterations=2.5), RoutingParams(max_iterations=True),
        RoutingParams(window_mm=0.0), RoutingParams(via_cost=math.inf),
    ):
        with pytest.raises(CompileError, match="routing parameter|via diameter"):
            route_board(ir, lib, bad)


def test_refusals_instead_of_guesses(tmp_path: Path, lib: KicadLibrary):
    def ir():
        return board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))

    x = ir()
    x.pcb.placements = [p for p in x.pcb.placements if p.component_ref != "R2"]
    with pytest.raises(CompileError, match="component 'R2' has no placement"):
        route_board(x, lib)
    x = ir()
    x.component("R2").footprint = None
    with pytest.raises(CompileError, match="component 'R2' has no footprint"):
        route_board(x, lib)
    x = ir()
    x.component("R2").footprint.name = "Missing"
    with pytest.raises(CompileError, match="Test:Missing of 'R2' was not found in a KiCad library"):
        route_board(x, lib)
    x = ir()
    x.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="power"), Layer(name="In2.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")]
    with pytest.raises(CompileError, match=r"knows only \['F.Cu', 'B.Cu'\], ir.pcb.layers also has \['In1.Cu', 'In2.Cu'\]"):
        route_board(x, lib)
    x = ir()
    x.pcb.layers = [Layer(name="F.Cu", kind="signal")]
    with pytest.raises(CompileError, match=r"lacks \['B.Cu'\]"):
        route_board(x, lib)
    # a pad too small for the grid: the nearest grid point (3.0, 3.0) is 0.12 mm from the 0.2 mm pad's centre
    x = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("T1", "TINY", 9.12, 3.0)], {"N": [("R1", "1"), ("T1", "1")]}, (12.0, 6.0))
    with pytest.raises(CompileError, match=r"pad T1.1 \(0.2 x 0.2 mm\) is too small for the 0.25 mm routing grid: the nearest grid point is 0.1200 mm"):
        route_board(x, lib)
    assert route_board(board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("T1", "TINY", 9.0, 3.0)], {"N": [("R1", "1"), ("T1", "1")]}, (12.0, 6.0)), lib).unrouted == {}
    # a net naming an unplaced / unknown component, or a pin the footprint does not have
    x = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1"), ("R3", "1")]}, (12.0, 6.0))
    with pytest.raises(CompileError, match="net 'N' references unknown component 'R3'"):
        route_board(x, lib)
    x = ir()
    x.components.append(make_part("R3"))  # in the IR but never placed
    with pytest.raises(CompileError, match="component 'R3' has no placement"):
        route_board(x, lib)
    x = ir()
    x.nets[0].pins.append(PinRef(component_ref="R1", pin_number="7"))
    with pytest.raises(CompileError, match="net 'N' references R1.7 but footprint Test:PAD1 has no pad '7'"):
        route_board(x, lib)
    x = ir()
    x.pcb.placements.append(Placement(component_ref="R9", x_mm=1.0, y_mm=1.0))
    with pytest.raises(CompileError, match=r"placement\(s\) of unknown component\(s\) \['R9'\]"):
        route_board(x, lib)
    # no board, no outline
    x = ir()
    x.pcb = None
    with pytest.raises(CompileError, match="ir.pcb is None"):
        route_board(x, lib)
    x = ir()
    x.pcb.outline = None
    with pytest.raises(CompileError, match="ir.pcb.outline is None"):
        route_board(x, lib)
    x = ir()
    x.pcb.placements[1] = Placement(component_ref="R2", x_mm=20.0, y_mm=3.0)  # outside the 12 x 6 outline
    with pytest.raises(CompileError, match=r"pad R2.1 at \(20.0, 3.0\) lies outside the board outline"):
        route_board(x, lib)


def test_a_pad_inside_the_outline_but_past_the_last_grid_cell_is_refused_with_the_true_reason(tmp_path: Path, lib: KicadLibrary):
    """9.4 mm wide, 0.25 mm grid: the last cell is at 9.25; a pad centred at 9.39 is inside the outline, and the message must not say otherwise."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 3.0), ("R2", "PAD1", 9.39, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (9.4, 6.0))
    with pytest.raises(CompileError, match=r"pad R2.1 at \(9.39, 3.0\) is inside the 9.4 x 6 mm outline but nearer than half a 0.25 mm grid step to its far edge") as e:
        route_board(ir, lib)
    assert "outside the board outline" not in str(e.value)
    ir.pcb.placements[1] = Placement(component_ref="R2", x_mm=9.41, y_mm=3.0)  # 0.01 mm further: now really outside
    with pytest.raises(CompileError, match=r"pad R2.1 at \(9.41, 3.0\) lies outside the board outline"):
        route_board(ir, lib)


def test_a_terminal_cell_inside_a_foreign_keep_out_leaves_the_net_unrouted_never_copper_through_it(tmp_path: Path, lib: KicadLibrary):
    """R1 (0.54 x 0.64 mm, net A) and R3 (1.0 mm, net B) sit exactly 0.25 mm apart - legal at the router's clearance - but R1's terminal cell
    (5.0, 3.0) is 0.395 mm from R3's box, inside the ``clearance + width/2 + grid/2`` keep-out: BLOCKED on the owner map. The router must not
    enter it (a track cap there would be 0.195 mm from R3), so A is unrouted with that reason and B still routes with its clearance kept."""
    ir = board_ir(
        tmp_path, lib, [("R1", "SMD054", 4.875, 3.0), ("R2", "PAD1", 1.0, 3.0), ("R3", "SMD1", 5.895, 3.0), ("R4", "PAD1", 9.0, 3.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (10.0, 6.0),
    )
    board = _Board(ir, lib, P)
    t = board.terminals["A"][0]
    assert t.label == "R1.1" and board.pos(t.cell) == (5.0, 3.0) and board.owner[0][t.cell] == BLOCKED and board.usable_layers(t, board.net_index["A"]) == ()
    r = route_board(ir, lib)
    assert r.unrouted == {
        "A": "R1.1 terminal cell (5, 3) is inside a keep-out on every copper layer of the pad "
        "(a foreign pad, a pad without a net or the board edge is within clearance 0.25 + width/2 of it)"
    }
    assert r.tracks and all(t.net == "B" for t in r.tracks) and r.stats["routed_nets"] == 1 and r.stats["unrouted_nets"] == 1
    r3_box = (5.895 - 0.5, 3.0 - 0.5, 5.895 + 0.5, 3.0 + 0.5)
    r1_box = (4.875 - 0.27, 3.0 - 0.32, 4.875 + 0.27, 3.0 + 0.32)
    for t in r.tracks:  # B's copper keeps the router's clearance from A's pad (its own pad it may touch)
        assert _segment_box_distance(t.start, t.end, r1_box) >= PAD_KEEPOUT - 1e-9, t
    assert any(_segment_box_distance(t.start, t.end, r3_box) == 0.0 for t in r.tracks)
    # the same at the outline: a 0.5 x 1.0 mm pad whose edge is 0.1 mm inside the board has its terminal cell (0.25, 3.0) in the edge
    # keep-out (edge_clearance + width/2 = 0.5): BLOCKED on both layers, so no track cap 0.05 mm from the outline is ever emitted
    ir = board_ir(tmp_path, lib, [("R1", "SMD0510", 0.35, 3.0), ("R2", "PAD1", 6.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (10.0, 6.0))
    board = _Board(ir, lib, P)
    t = board.terminals["N"][0]
    assert board.pos(t.cell) == (0.25, 3.0) and board.owner[0][t.cell] == BLOCKED and board.owner[1][t.cell] == BLOCKED
    r = route_board(ir, lib)
    assert r.tracks == [] and r.vias == [] and list(r.unrouted) == ["N"] and r.unrouted["N"].startswith("R1.1 terminal cell (0.25, 3) is inside a keep-out")
    # a THT pad whose centre lies in the edge keep-out likewise (its copper would even leave the board): unrouted, not routed past the edge
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 4.0, 4.0), ("R2", "PAD1", 13.85, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (14.0, 8.0))
    r = route_board(ir, lib)
    assert r.tracks == [] and r.unrouted["N"].startswith("R2.1 terminal cell (13.75, 4) is inside a keep-out")
    # a target cell is never entered against the owner map: the terminal test above also guards the seed side (R1.1 is the first terminal of A)


def test_a_via_is_drilled_beside_a_pad_never_through_it(tmp_path: Path, lib: KicadLibrary):
    """Top SMD to bottom SMD, one net: the via must sit outside both pad boxes (its copper disc may touch a box edge, never overlap it)."""
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 2.0, 3.0), ("R2", "SMD1", 8.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (10.0, 6.0), sides={"R2": BoardSide.BOTTOM})
    r = route_board(ir, lib)
    assert r.unrouted == {} and len(r.vias) == 1 and _connected(r.tracks, [(2.0, 3.0), (8.0, 3.0)])
    v = r.vias[0]
    for box in ((1.5, 2.5, 2.5, 3.5), (7.5, 2.5, 8.5, 3.5)):
        assert _segment_box_distance((v.x_mm, v.y_mm), (v.x_mm, v.y_mm), box) >= P.via_diameter_mm / 2 - 1e-9, (v.x_mm, v.y_mm)
    assert any(t.layer == "B.Cu" and (8.0, 3.0) in (t.start, t.end) for t in r.tracks)  # a bottom track enters the bottom pad
    # the owner map says the same for every pad, the net's own included
    board = _Board(ir, lib, P)
    for k in range(board.n):
        x, y = board.pos(k)
        inside = any(_segment_box_distance((x, y), (x, y), box) < P.via_diameter_mm / 2 - 1e-9 for box in ((1.5, 2.5, 2.5, 3.5), (7.5, 2.5, 8.5, 3.5)))
        assert board.via_pad_ok[k] == (not inside)
        if inside:
            assert not board.via_allowed(k, board.net_index["N"])


def test_pads_whose_copper_is_not_bounded_by_the_size_box_are_refused(tmp_path: Path, lib: KicadLibrary):
    """A trapezoid (rect_delta) has copper the library reader does not keep, and so has a custom pad with a primitive the reader does not
    read: refused, never modelled as the size box (a custom pad whose primitives are read is routed: tests/test_routing_custom_pads.py)."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 10.0, 4.0), ("U1", "TRAP", 6.0, 4.0)], {"A": [("R1", "1"), ("R2", "1")], "B": [("U1", "1")]}, (12.0, 8.0))
    with pytest.raises(CompileError, match=r"cannot route: pad U1.1 of footprint Test:TRAP has shape 'trapezoid'; its copper is not bounded by its \(size\) box"):
        route_board(ir, lib)
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 10.0, 4.0), ("U1", "CUSTX", 6.0, 4.0)], {"A": [("R1", "1"), ("R2", "1")], "B": [("U1", "1")]}, (12.0, 8.0))
    with pytest.raises(CompileError, match=r"cannot route: pad U1.1 of footprint Test:CUSTX: custom pad '1' has primitive\(s\) gr_blob the library reader does not read"):
        route_board(ir, lib)


def test_an_unreachable_terminal_leaves_the_net_unrouted_without_copper_and_the_others_routed(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("C1", "CAGE", 9.0, 3.0), ("R2", "PAD1", 3.0, 7.0), ("R3", "PAD1", 9.0, 7.0), ("R4", "PAD1", 6.0, 9.0)],
        {"N": [("R1", "1"), ("C1", "1")], "K": [("R2", "1"), ("R3", "1")], "S": [("R4", "1")]}, (12.0, 10.0),
    )
    r = route_board(ir, lib)  # no exception
    assert r.unrouted == {"N": "R1.1 unreachable from the routed part of the net"}  # C1.1 seeds the tree (natural ref order), R1.1 stays unreachable
    assert all(t.net == "K" for t in r.tracks) and r.vias == [] and _connected(r.tracks, [(3.0, 7.0), (9.0, 7.0)])
    assert r.stats["routed_nets"] == 1 and r.stats["unrouted_nets"] == 1 and r.stats["skipped_nets"] == ["S"]
    assert list(r.stats["net_length_mm"]) == ["K"]
    # the caged pad's cell is fenced: 0.75 and 1.0 mm out in every direction the cells are BLOCKED on both layers (the bars have no net)
    board = _Board(ir, lib, P)
    cell = board.terminals["N"][0].cell
    assert board.terminals["N"][0].label == "C1.1"
    for layer in range(len(LAYERS)):
        for d in (1, -1, board.nx, -board.nx):
            assert board.owner[layer][cell + 3 * d] == BLOCKED and board.owner[layer][cell + 4 * d] == BLOCKED
    # a net with pads on both sides of a bare board still fails honestly when the other side is fenced too
    assert route_board(ir, lib, RoutingParams(via_cost=0.0)).unrouted == r.unrouted


def test_nothing_to_route(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"A": [("R1", "1")], "B": []}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.tracks == [] and r.vias == [] and r.unrouted == {} and r.stats["skipped_nets"] == ["B", "A"] and r.stats["routed_nets"] == 0  # net order: (pad count, name)
    assert r.stats["total_length_mm"] == 0.0 and r.stats["net_length_mm"] == {}
    # existing copper is neither reused nor an obstacle: the result is the same and the IR is untouched
    ir.pcb.tracks = [Track(net="A", layer="F.Cu", start=(1.0, 1.0), end=(2.0, 1.0), width_mm=0.25)]
    before = ir.content_hash()
    assert route_board(ir, lib).tracks == [] and ir.content_hash() == before


def test_the_compiled_board_carries_one_segment_per_track_and_one_via_per_via(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(
        tmp_path, lib, [("R1", "SMD2", 2.5, 4.0), ("R2", "SMD2", 10.5, 4.0), ("W1", "WALL2", 6.5, 8.0), ("R3", "PAD2", 2.5, 12.0), ("R4", "PAD2", 10.5, 12.0)],
        {"N": [("R1", "2"), ("R2", "1")], "K": [("R3", "2"), ("R4", "1")]}, (14.0, 16.0),
    )
    r = route_board(ir, lib)
    assert r.unrouted == {} and len(r.vias) >= 2 and len(r.tracks) >= 4, r.stats  # N crosses the two net-less walls on B.Cu
    ir.pcb.tracks = list(r.tracks)
    ir.pcb.vias = list(r.vias)
    assert all(prov.note != UNRECORDED_ORIGIN and not prov.needs_verification for _, prov in ir.pcb.layout_items() if not _.startswith("placement"))
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    board = sexpr.parse_file(Path(art.path))
    segments = sexpr.find_all(board, "segment")
    vias = sexpr.find_all(board, "via")
    assert len(segments) == len(r.tracks) and len(vias) == len(r.vias)
    starts = {(float(sexpr.find(s, "start")[1]), float(sexpr.find(s, "start")[2]), str(sexpr.find(s, "layer")[1]), str(sexpr.find(s, "net")[1])) for s in segments}
    assert starts == {(t.start[0], t.start[1], t.layer, t.net) for t in r.tracks}
    assert {(float(sexpr.find(v, "at")[1]), float(sexpr.find(v, "at")[2])) for v in vias} == {(v.x_mm, v.y_mm) for v in r.vias}
    assert all(float(sexpr.find(s, "width")[1]) == 0.4 for s in segments)
    assert all(isinstance(v, Via) for v in ir.pcb.vias)


def _astable(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """The grid-placed synthetic astable: the placements the pipeline's grid placer produces for the ``astable`` template on the
    synthetic template library (THT R / C / Q, an SMD 1x03 header)."""
    from ai_eda.ir import LibraryRef

    ir = CircuitIR(project=ProjectMeta(id="osc", name="osc", workdir=str(tmp_path)))
    parts = {
        "C1": ("Capacitor_THT", "C_Disc_D5.0mm_W2.5mm_P5.00mm", 5.34, 3.2), "C2": ("Capacitor_THT", "C_Disc_D5.0mm_W2.5mm_P5.00mm", 16.83, 3.2),
        "J1": ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical", 27.78, 3.2), "Q1": ("Package_TO_SOT_THT", "TO-92_Inline", 41.08, 3.2),
        "Q2": ("Package_TO_SOT_THT", "TO-92_Inline", 6.61, 6.6), "R1": ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal", 16.83, 6.6),
        "R2": ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal", 28.32, 6.6), "R3": ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal", 39.81, 6.6),
        "R4": ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal", 5.34, 10.0),
    }
    placements = []
    for ref, (library, name, x, y) in parts.items():
        c = make_part(ref)
        c.symbol = None
        c.footprint = lib.resolve_footprint(LibraryRef(library=library, name=name))
        assert c.footprint.verified, name
        ir.components.append(c)
        placements.append(Placement(component_ref=ref, x_mm=x, y_mm=y, provenance=NET_P))
    ir.nets = [Net(name=n, pins=[PinRef(component_ref=r, pin_number=k) for r, k in pins], provenance=NET_P) for n, pins in ASTABLE_NETS.items()]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=48.96, height_mm=13.2), placements=placements)
    return ir


ASTABLE_NETS = {
    "VCC": [("J1", "1"), ("R1", "1"), ("R2", "1"), ("R3", "1"), ("R4", "1")], "Q1_C": [("R1", "2"), ("Q1", "3"), ("C1", "1")],
    "Q2_B": [("C1", "2"), ("R4", "2"), ("Q2", "2")], "OUT": [("R2", "2"), ("Q2", "3"), ("C2", "1"), ("J1", "2")],
    "Q1_B": [("C2", "2"), ("R3", "2"), ("Q1", "2")], "GND": [("J1", "3"), ("Q1", "1"), ("Q2", "1")],
}


def test_the_synthetic_astable_is_negotiated_to_a_legal_board(tmp_path: Path):
    """0.1 walled the 5-pad VCC net in on this board and needed a second pass with VCC first; 0.2 negotiates: iteration 1 routes every
    net against the others' copper as a cost, the conflicts are ripped up and rerouted until no net's copper lies in another's halo.
    The result is legal (the independent validator agrees at the router's clearance), whole and a pure function of the inputs."""
    from tests.test_circuit_templates import template_library
    from ai_eda.validation import ValidationContext, default_registry

    lib = template_library(tmp_path / "kicad")
    ir = _astable(tmp_path, lib)
    r = route_board(ir, lib)
    s = r.stats
    assert r.unrouted == {} and s["routed_nets"] == 6 and s["legal"] is True and s["dropped"] == [] and s["recovered"] == [], s
    assert s["net_order"] == ["GND", "Q1_B", "Q1_C", "Q2_B", "OUT", "VCC"]  # (pad count, name): no reordering any more
    assert s["iterations"] == len(s["history"]) > 1 and s["history"][-1]["conflicting"] == [] and s["history"][0]["rerouted"] == 6
    assert all(row["conflicting"] for row in s["history"][:-1])  # the loop ran exactly until the conflicts were gone
    assert [row["rerouted"] for row in s["history"][1:]] == [len(row["conflicting"]) for row in s["history"][:-1]]  # only conflicting nets rerouted
    assert set(s["net_length_mm"]) == set(ASTABLE_NETS) and "passes" not in s
    assert all(f"{s['iterations']} iteration(s)" in t.provenance.note for t in r.tracks)
    assert _geometry(r) == _geometry(route_board(copy.deepcopy(ir), template_library(tmp_path / "kicad2")))  # still a pure function of the inputs
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(P.clearance_mm, note="the router's own clearance as the limit"))
    checks = {c.check_id: c for c in default_registry.get("pcb.routing").validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))}
    assert checks["pcb.routing.connectivity"].status.value == "PASS" and checks["pcb.routing.clearance"].status.value == "PASS", checks
    # a board that is legal after one iteration reports it
    x = board_ir(tmp_path, fixture_library(tmp_path / "kicad3"), [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    s = route_board(x, fixture_library(tmp_path / "kicad3")).stats
    assert s["iterations"] == 1 and s["legal"] and s["net_order"] == ["N"] and s["history"] == [{"iteration": 1, "rerouted": 1, "overused_cells": 0, "conflicting": []}]


# --------------------------------------------------------------------------- routing rules from the pad pitch


def _placed_mcu(tmp_path: Path, *, shift: tuple[float, float] = (0.0, 0.0)):
    """The synthetic 64-pad MCU board of ``tests/test_core_ring.py`` placed by core_ring (every placement moved by ``shift``)."""
    from tests.test_core_ring import mcu_ir, mcu_library
    from ai_eda.tools.placement.core_ring import core_ring_placement

    mlib = mcu_library(tmp_path / "mcu_kicad")
    ir = mcu_ir(tmp_path, mlib)
    ring = core_ring_placement(ir, mlib)
    placements = [p.model_copy(update={"x_mm": p.x_mm + shift[0], "y_mm": p.y_mm + shift[1]}) for p in ring.placements]
    ir.pcb = PCBDesign(outline=ring.outline, placements=placements)
    return ir, mlib


def test_fine_rules_are_chosen_from_the_finest_pad_pitch(tmp_path: Path):
    from ai_eda.tools.routing import FINE_PITCH_MM, FINE_RULES, finest_pad_pitch

    ir, mlib = _placed_mcu(tmp_path)
    assert finest_pad_pitch(ir, mlib) == (0.8, "Test_MCU:QFP64")  # the QFP's 0.8 mm, not the 1.65 mm of the 0603 parts
    p = RoutingParams.for_board(ir, mlib)
    assert FINE_PITCH_MM == 1.0 and FINE_RULES == {"grid_mm": 0.2, "track_width_mm": 0.25, "clearance_mm": 0.2, "via_diameter_mm": 0.6, "via_drill_mm": 0.3, "edge_clearance_mm": 0.3}
    assert p == RoutingParams(**FINE_RULES, rules="fine", pad_pitch_mm=0.8, pitch_footprint="Test_MCU:QFP64")
    assert (p.via_cost, p.bend_cost) == (RoutingParams().via_cost, RoutingParams().bend_cost)
    assert p.derived_from_entry() == (
        f"params:grid=0.2,width=0.25,clearance=0.2,via=0.6/0.3,edge=0.3,via_cost=12.0,bend_cost=0.6,{NEGOTIATION_ENTRY},rules=fine,pad_pitch=0.8,pitch_footprint=Test_MCU:QFP64"
    )
    # fab minimums still raise the fine rules, and the reason stays recorded
    ir.pcb.manufacturing = ManufacturingConstraints(min_track_width_mm=assumption(0.3, note="fab page not read"), min_clearance_mm=assumption(0.15, note="idem"))
    eff, raised = effective_params(ir, p)
    assert raised == {"track_width_mm": (0.25, 0.3)} and eff.track_width_mm == 0.3 and eff.clearance_mm == 0.2 and eff.rules == "fine" and eff.pad_pitch_mm == 0.8
    # nonsense is refused
    with pytest.raises(CompileError, match="pad_pitch_mm must be a finite number > 0"):
        RoutingParams(pad_pitch_mm=0.0).check()


def test_default_rules_on_boards_without_a_fine_pitch(tmp_path: Path, lib: KicadLibrary):
    from ai_eda.ir import LibraryRef
    from ai_eda.tools.routing import finest_pad_pitch
    from tests.fixtures_kicad import divider_with_connector_ir
    from tests.test_circuit_templates import template_library
    from tests.test_core_ring import mcu_library

    # the astable's THT parts (TO-92 inline at 2.54 mm on the synthetic template library) and the divider's 0603 / header parts
    tlib = template_library(tmp_path / "tpl")
    ir = CircuitIR(project=ProjectMeta(id="osc", name="osc", workdir=str(tmp_path)))
    for ref, (library, name) in {"Q1": ("Package_TO_SOT_THT", "TO-92_Inline"), "R1": ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal"),
                                 "C1": ("Capacitor_THT", "C_Disc_D5.0mm_W2.5mm_P5.00mm"), "J1": ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical")}.items():
        c = make_part(ref)
        c.footprint = tlib.resolve_footprint(LibraryRef(library=library, name=name))
        ir.components.append(c)
    assert finest_pad_pitch(ir, tlib)[0] >= 1.0
    assert RoutingParams.for_board(ir, tlib) == RoutingParams() and RoutingParams.for_board(ir, tlib).rules is None
    div = divider_with_connector_ir(tmp_path, tlib)
    assert RoutingParams.for_board(div, tlib) == RoutingParams()
    assert RoutingParams().derived_from_entry() == DEFAULT_ENTRY  # no rule set recorded below 1.0 mm pitch
    # what counts as a pitch: pads that can carry different nets, with copper, not on top of each other
    mlib = mcu_library(tmp_path / "mcu")
    x = board_ir(tmp_path, lib, [("R1", "DUP1", 3.0, 3.0), ("R2", "CAGE", 9.0, 3.0)], {}, (14.0, 8.0))
    assert finest_pad_pitch(x, lib) == (1.4, "Test:CAGE")  # DUP1's two pads "1" are one logical pad; CAGE's unnumbered bars are copper 1.4 mm from its pad
    x = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0)], {}, (14.0, 8.0))
    assert finest_pad_pitch(x, lib) is None and RoutingParams.for_board(x, lib) == RoutingParams()  # one pad per footprint: no pitch at all
    sw = CircuitIR(project=ProjectMeta(id="sw", name="sw", workdir=str(tmp_path)))
    for ref, name in (("SW1", "SW4"), ("H1", "HOLE")):
        c = make_part(ref)
        c.footprint = mlib.resolve_footprint(LibraryRef(library="Test_MCU", name=name))
        sw.components.append(c)
    assert finest_pad_pitch(sw, mlib) == (4.5, "Test_MCU:SW4")  # 6.5 mm between the two "1" pads does not count, 4.5 mm between "1" and "2" does; the NPTH has no copper
    # a component whose footprint is not on disk is not measured (route_board refuses that board anyway)
    sw.components[0].footprint = LibraryRef(library="Test_MCU", name="Missing")
    assert finest_pad_pitch(sw, mlib) is None


def test_the_qfp_terminal_rule_holds_at_the_fine_rules_and_fails_at_the_defaults(tmp_path: Path):
    """0.45 mm pads at 0.8 mm pitch: at the fine rules every U1 terminal is a usable cell inside its pad's inscribed circle - also with the core
    off the routing grid by 0.1 mm (the worst case) -, while the default rules' keep-out (0.25 + 0.2 + 0.125 = 0.575 mm, exactly the gap from a
    pad centre to its neighbour's edge) fences most of them: whichever way a terminal cell is off the pad centre, one neighbour is nearer."""
    from tests.test_core_ring import QFP_PITCH, qfp_pad

    fine = None
    for shift in ((0.0, 0.0), (0.1, 0.1), (0.1, 0.0)):
        ir, mlib = _placed_mcu(tmp_path / f"s{shift[0]}_{shift[1]}", shift=shift)
        fine = RoutingParams.for_board(ir, mlib)
        board = _Board(ir, mlib, fine)
        terms = [t for net in board.terminals.values() for t in net if t.pad.ref == "U1"]
        assert len(terms) == 28  # the U1 pins the fixture wires: VCC x2, GND x3, AVCC, AREF, RESET, XTAL x2, UART x2, ports A / B (8 + 8)
        for t in terms:
            x, y = board.pos(t.cell)
            assert math.hypot(x - t.pad.cx, y - t.pad.cy) <= fine.grid_mm * math.sqrt(2) / 2 + 1e-9 < t.pad.inscribed_r == 0.225
            assert board.usable_layers(t, board.net_index[next(n for n, ts in board.terminals.items() if t in ts)]) == (0,), (t.label, shift)
        default = _Board(ir, mlib, RoutingParams())
        fenced = [t for net, ts in default.terminals.items() for t in ts if t.pad.ref == "U1" and not default.usable_layers(t, default.net_index[net])]
        assert len(fenced) >= 14, len(fenced)  # 21 of 28 at the core on the grid
    # the pad geometry the numbers above rely on
    assert qfp_pad(1)[2:] == (1.5, 0.45) and QFP_PITCH - 0.45 == pytest.approx(0.35)
    assert fine.clearance_mm + fine.track_width_mm / 2 + fine.grid_mm / 2 < QFP_PITCH - 0.45 / 2 - fine.grid_mm / 2  # 0.425 < 0.475: the neighbour never fences a terminal


def test_qfp_pads_escape_along_their_axis_at_the_fine_rules(tmp_path: Path):
    """U1's 8 port-A pins to the J1 header, alone on the board: every net routes, and the segment that crosses a U1 pad's edge - whichever
    end of it lies on the pad - leaves along the pad's long axis, or across it only from a side with no neighbouring pad (a corner pad's free
    end: pins 1, 16, 17, 32, 33, 48, 49, 64). Segments entirely inside the pad (the stub from the terminal cell to the pad centre) are not
    escapes and are skipped."""
    ir, mlib = _placed_mcu(tmp_path)
    keep = {"U1", "J1"}
    ir.components = [c for c in ir.components if c.ref in keep]
    ir.nets = [n for n in ir.nets if n.name.startswith("PA")]
    ir.pcb.placements = [p for p in ir.pcb.placements if p.component_ref in keep]
    p = RoutingParams.for_board(ir, mlib)
    r = route_board(ir, mlib, p)
    assert r.unrouted == {} and r.stats["routed_nets"] == 8 and r.params == p, r.stats
    assert all(t.width_mm == 0.25 and t.provenance.derived_from[-1] == p.derived_from_entry() for t in r.tracks)
    u1 = ir.pcb.placement("U1")
    fp = mlib.load_footprint(ir.component("U1").footprint)
    across_seen: list[str] = []
    for net in ir.nets:
        pin = next(q.pin_number for q in net.pins if q.component_ref == "U1")
        pad = fp.pad(pin)
        cx, cy = u1.x_mm + pad.x, u1.y_mm + pad.y
        horizontal = pad.size_w > pad.size_h

        def on(pt: tuple[float, float]) -> bool:
            return abs(pt[0] - cx) <= pad.size_w / 2 + 1e-9 and abs(pt[1] - cy) <= pad.size_h / 2 + 1e-9

        touching = [t for t in r.tracks if t.net == net.name and (on(t.start) or on(t.end))]
        crossing = [t for t in touching if not (on(t.start) and on(t.end))]
        assert crossing, (net.name, touching)
        # the row's neighbours: the U1 pads one pitch away along the row (the row runs across a horizontal pad's long axis)
        neighbours = [q for q in fp.pads if q.number != pin and math.isclose(math.hypot(q.x - pad.x, q.y - pad.y), 0.8, abs_tol=1e-6)]
        assert 1 <= len(neighbours) <= 2
        for t in crossing:
            inside, outside = (t.start, t.end) if on(t.start) else (t.end, t.start)
            if (inside[1] == outside[1]) if horizontal else (inside[0] == outside[0]):
                continue  # along the long axis
            across = (outside[1] - inside[1]) if horizontal else (outside[0] - inside[0])
            toward = [q.number for q in neighbours if ((q.y - pad.y) if horizontal else (q.x - pad.x)) * across > 0]
            assert toward == [], (net.name, pin, t.start, t.end, toward)  # never from between two pads
            assert pin in ("1", "16", "17", "32", "33", "48", "49", "64"), (net.name, pin)
            across_seen.append(pin)
    assert across_seen == ["48"]  # PA3's corner pad, entered from the free end of the east row at this placement
    assert _geometry(r) == _geometry(route_board(copy.deepcopy(ir), mcu_library_copy(tmp_path), p))


def mcu_library_copy(tmp_path: Path) -> KicadLibrary:
    from tests.test_core_ring import mcu_library

    return mcu_library(tmp_path / "mcu_kicad_again")


# --------------------------------------------------------------------------- negotiated congestion (0.2)


def first_come(ir: CircuitIR, lib: KicadLibrary, order: list[str]) -> dict[str, bool]:
    """0.1's pass: each net in ``order`` routed against the copper of the nets before it as an obstacle, no rip-up; ``{net: routed}``.

    The strict search keeps out of every other net's halo, which on a board
    without a via site is exactly 0.1's owner-map rule (same A*, same costs,
    same tie-breaking; a 2-pad net starts at its first pad in natural order
    either way).
    """
    p, _ = effective_params(ir)
    board = _Board(ir, lib, p)
    neg = _Negotiation(board, p)
    out: dict[str, bool] = {}
    for name in order:
        net = next(n for n in ir.nets if n.name == name)
        got = _route_net(neg, net, board.net_index[name], board.terminals[name], strict=True)
        out[name] = not isinstance(got, str)
        if out[name]:
            neg.add(got)
    return out


def _routing_checks(ir: CircuitIR, r: Routing, lib: KicadLibrary, tmp_path: Path) -> dict:
    """The ``pcb.routing`` validator on ``ir`` carrying ``r``'s copper, the router's own clearance as the limit (IR geometry, not DRC)."""
    from ai_eda.validation import ValidationContext, default_registry

    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(r.tracks), list(r.vias)
    x.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(r.params.clearance_mm, note="the router's own clearance as the limit"))
    return {c.check_id: c for c in default_registry.get("pcb.routing").validate(x, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))}


#: two nets that cross on one layer (the B.Cu plane leaves no second layer and no via site): A from R1 (bottom left) to R2 (top right),
#: B from R3 (bottom right) to R4 (top left); the net-less bars W0 / W1 close the left edge except for one gap
SWAP_PARTS: list[Part] = [
    ("Z1", "BPLANE", 6.0, 4.5), ("R1", "SMD1", 4.0, 7.0), ("R2", "SMD1", 10.5, 1.5), ("R3", "SMD1", 8.5, 7.5), ("R4", "SMD1", 4.5, 1.5),
    ("W0", "HBAR", 1.0, 3.0), ("W1", "VBAR", 1.0, 6.5),
]
SWAP_NETS = {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}


def test_two_crossing_nets_that_first_come_routing_cannot_finish_in_either_order_are_negotiated(tmp_path: Path, lib: KicadLibrary):
    """Whichever net goes first takes its shortest L and, with a pad and a bar, walls the other net's pads apart: 0.1 routed A then B,
    then B first (its second pass), and left one net unrouted both times. 0.2 rips the conflict up until A and B trade their shortest
    routes for a pair that fits - B around R1 through the gap in the bars - and the result is legal, single-layer and deterministic
    (two runs compile to byte-identical boards)."""
    ir = board_ir(tmp_path, lib, SWAP_PARTS, SWAP_NETS, (12.0, 9.0))
    assert first_come(ir, lib, ["A", "B"]) == {"A": True, "B": False}
    assert first_come(ir, lib, ["B", "A"]) == {"B": True, "A": False}
    r = route_board(ir, lib)
    s = r.stats
    assert r.unrouted == {} and s["routed_nets"] == 2 and s["legal"] and s["dropped"] == [] and s["recovered"] == [], s
    assert s["iterations"] > 1 and s["history"][0]["conflicting"] == ["A", "B"] and s["history"][-1]["conflicting"] == []
    assert r.vias == [] and {t.layer for t in r.tracks} == {"F.Cu"}  # the plane leaves one layer
    assert _connected([t for t in r.tracks if t.net == "A"], [(4.0, 7.0), (10.5, 1.5)]) and _connected([t for t in r.tracks if t.net == "B"], [(8.5, 7.5), (4.5, 1.5)])
    assert s["net_length_mm"]["A"] + s["net_length_mm"]["B"] > 12.0 + 10.0  # longer than the two shortest routes together: the price of fitting
    checks = _routing_checks(ir, r, lib, tmp_path)
    assert checks["pcb.routing.connectivity"].status.value == "PASS" and checks["pcb.routing.clearance"].status.value == "PASS", checks
    # deterministic: two more runs on copies with other library instances serialise to the same bytes - the design view of the copper
    # (provenance included; its wall-clock stamp is not design content) and the stats
    def as_bytes(x: Routing) -> bytes:
        return json.dumps(design_data(PCBDesign(tracks=x.tracks, vias=x.vias)), sort_keys=True).encode() + repr(x.stats).encode()

    runs = [as_bytes(route_board(copy.deepcopy(ir), fixture_library(tmp_path / f"kicad_run{k}"))) for k in (1, 2)]
    assert runs[0] == runs[1] == as_bytes(r) and b"routing.maze" in runs[0]


def test_the_iteration_cap_keeps_only_whole_legal_nets_and_names_the_rest(tmp_path: Path, lib: KicadLibrary):
    """At the cap, conflicting nets are ripped up (most partners, then fewer pads, then name) until the rest is legal; a ripped-up net
    gets one more route against the legal copper as an obstacle and keeps it when that exists. What is emitted is always whole and legal;
    a net without a legal route has no copper and a reason - the caller applies all or nothing (or, on request, the whole nets)."""
    ir = board_ir(tmp_path, lib, SWAP_PARTS, SWAP_NETS, (12.0, 9.0))
    capped = RoutingParams(max_iterations=1)
    r = route_board(ir, lib, capped)
    s = r.stats
    assert s["iterations"] == 1 and s["legal"] is False and s["dropped"] == ["A"] and s["recovered"] == []  # a tie on partners and pads: the name
    assert r.unrouted == {"A": "no legal route after 1 negotiation iteration(s): its copper still broke the clearance of B, and a route against the legal copper as an obstacle was not found"}
    assert r.tracks and {t.net for t in r.tracks} == {"B"} and s["routed_nets"] == 1 and s["unrouted_nets"] == 1 and list(s["net_length_mm"]) == ["B"]
    assert all("kept after the conflicting nets were ripped up" in t.provenance.note and "1 iteration(s)" in t.provenance.note for t in r.tracks)
    assert all(t.provenance.derived_from[-1] == capped.derived_from_entry() and "max_iterations=1" in t.provenance.derived_from[-1] for t in r.tracks)
    checks = _routing_checks(ir, r, lib, tmp_path)
    connectivity = checks["pcb.routing.connectivity"]
    assert connectivity.status.value == "FAIL" and {row["net"]: row["status"] for row in connectivity.details["nets"]} == {"A": "FAIL", "B": "PASS"}
    assert checks["pcb.routing.clearance"].status.value == "PASS"  # what is emitted keeps every clearance
    # a crossing that has room around a pad: the ripped-up net is recovered against the legal copper
    ir = board_ir(tmp_path, lib, [("Z1", "BPLANE", 6.0, 5.0), ("R1", "SMD1", 2.0, 5.0), ("R2", "SMD1", 10.0, 5.0), ("R3", "SMD1", 6.0, 3.0), ("R4", "SMD1", 6.0, 7.0)],
                  {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0))
    r = route_board(ir, lib, capped)
    s = r.stats
    assert s["legal"] is False and s["dropped"] == ["A"] and s["recovered"] == ["A"] and r.unrouted == {} and s["routed_nets"] == 2, s
    notes = {t.net: t.provenance.note for t in r.tracks}
    assert "routed against the legal copper as an obstacle after 1 negotiation iteration(s) left it in conflict" in notes["A"]
    assert "kept after the conflicting nets were ripped up" in notes["B"]
    checks = _routing_checks(ir, r, lib, tmp_path)
    assert checks["pcb.routing.connectivity"].status.value == "PASS" and checks["pcb.routing.clearance"].status.value == "PASS", checks
    full = route_board(ir, lib)  # without the cap the negotiation itself makes it legal
    assert full.stats["legal"] and full.unrouted == {} and full.stats["recovered"] == []


def test_a_recovered_net_never_starts_at_a_terminal_inside_a_kept_nets_halo(tmp_path: Path, lib: KicadLibrary):
    """A 0.9 mm track on a 0.25 mm grid (parameters ``check()`` accepts, as ``PCBAgent(routing=...)`` or a large fab minimum give):
    after the cap drops A, B's kept track at x = 6.0 has a halo that covers A's own terminal cell R1.1 at (5.0, 5.0) - 1.0 mm off,
    where two 0.9 mm tracks need 1.1 mm. The strict search checks the halo only on the cells it steps into, and a seed is never stepped
    into, so the recovery used to start there and emit A 0.1 mm from B. A terminal cell inside another net's halo is no seed and no
    target against legal copper: A stays unrouted with its drop reason, and what is emitted keeps the router's clearance."""
    parts = [("R1", "TINY", 5.0, 5.0), ("R2", "SMD1", 3.5, 3.25), ("R3", "SMD1", 6.0, 3.25), ("R4", "TINY", 5.5, 7.0), ("Z1", "BPLANE", 5.0, 5.0)]
    ir = board_ir(tmp_path, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (10.0, 10.0))
    p = RoutingParams(grid_mm=0.25, track_width_mm=0.9, clearance_mm=0.2, max_iterations=1, via_diameter_mm=1.0, via_drill_mm=0.4)
    r = route_board(ir, lib, p)
    s = r.stats
    assert s["dropped"] == ["A"] and s["recovered"] == [] and s["legal"] is False, s
    assert r.unrouted == {"A": "no legal route after 1 negotiation iteration(s): its copper still broke the clearance of B, and a route against the legal copper as an obstacle was not found"}
    assert {t.net for t in r.tracks} == {"B"}
    checks = _routing_checks(ir, r, lib, tmp_path)
    assert checks["pcb.routing.clearance"].status.value == "PASS", checks["pcb.routing.clearance"].message
    assert {row["net"]: row["status"] for row in checks["pcb.routing.connectivity"].details["nets"]} == {"A": "FAIL", "B": "PASS"}
    # the strict search itself: against B's copper, A's terminal is fenced with that reason instead of seeding a path
    board = _Board(ir, lib, p)
    neg = _Negotiation(board, p)
    b = _route_net(neg, next(x for x in ir.nets if x.name == "B"), board.net_index["B"], board.terminals["B"])
    assert not isinstance(b, str)
    neg.add(b)
    got = _route_net(neg, next(x for x in ir.nets if x.name == "A"), board.net_index["A"], board.terminals["A"], strict=True)
    assert got == "R1.1 terminal cell (5, 5) is inside another net's clearance halo on every copper layer of the pad", got


def test_a_multi_terminal_net_grows_a_steiner_tree_from_the_terminal_nearest_the_centroid(tmp_path: Path, lib: KicadLibrary):
    """Five pads in a plus: the tree starts at the centre pad R3 (nearest the centroid, although R1 comes first in ref order) and adds
    the nearest unconnected pad each time - four straight arms, 16 mm, one copper set."""
    parts = [("R1", "PAD1", 2.0, 6.0), ("R2", "PAD1", 6.0, 2.0), ("R3", "PAD1", 6.0, 6.0), ("R4", "PAD1", 6.0, 10.0), ("R5", "PAD1", 10.0, 6.0), ("R6", "PAD1", 2.0, 2.0), ("R7", "PAD1", 10.0, 10.0)]
    ir = board_ir(tmp_path, lib, parts, {"S": [(r, "1") for r in ("R1", "R2", "R3", "R4", "R5")], "T": [("R6", "1"), ("R7", "1")]}, (12.0, 12.0))
    r = route_board(ir, lib)
    s = r.stats
    assert r.unrouted == {} and s["legal"] and s["net_order"] == ["T", "S"], s
    star = [t for t in r.tracks if t.net == "S"]
    assert (6.0, 6.0) == star[0].start  # the first segment leaves the centre pad
    assert sorted((t.start, t.end) for t in star) == [((6.0, 6.0), (2.0, 6.0)), ((6.0, 6.0), (6.0, 2.0)), ((6.0, 6.0), (6.0, 10.0)), ((6.0, 6.0), (10.0, 6.0))]
    assert s["net_length_mm"]["S"] == 16.0 and r.vias == []
    checks = _routing_checks(ir, r, lib, tmp_path)
    assert checks["pcb.routing.connectivity"].status.value == "PASS" and checks["pcb.routing.clearance"].status.value == "PASS", checks
    assert {row["net"]: row["message"] for row in checks["pcb.routing.connectivity"].details["nets"]}["S"] == "5 pad(s) in one copper set"


def test_the_synthetic_64_pin_mcu_board_routes_completely(tmp_path: Path):
    """The whole core-ring-placed 64-pad MCU fixture (28 nets, the QFP at 0.8 mm pitch, fine rules), which 0.1 never finished: every
    net routed, legal after the negotiation, and the independent validator agrees at the router's 0.2 mm clearance."""
    ir, mlib = _placed_mcu(tmp_path)
    p = RoutingParams.for_board(ir, mlib)
    r = route_board(ir, mlib, p)
    s = r.stats
    assert r.unrouted == {} and s["routed_nets"] == len(ir.nets) == 28 and s["legal"] and s["dropped"] == [], (s["history"], r.unrouted)
    assert s["iterations"] <= p.max_iterations and s["via_count"] == len(r.vias) and s["track_count"] == len(r.tracks)
    checks = _routing_checks(ir, r, mlib, tmp_path)
    assert checks["pcb.routing.connectivity"].status.value == "PASS", checks["pcb.routing.connectivity"].message
    assert checks["pcb.routing.clearance"].status.value == "PASS", checks["pcb.routing.clearance"].message
    assert all(t.width_mm == 0.25 for t in r.tracks) and all((v.diameter_mm, v.drill_mm) == (0.6, 0.3) for v in r.vias)


def test_pads_that_repeat_a_number_are_each_a_terminal_with_their_own_stub(tmp_path: Path, lib: KicadLibrary):
    """``Test:DUP1`` has two pads numbered "1" (a switch's paired pins): both carry the net, both are reached and both get the stub to
    their exact (off-grid) centre - the router tracks terminals by position, never by the label they share."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("S1", "DUP1", 6.1, 4.1)], {"N": [("R1", "1"), ("S1", "1")]}, (10.0, 8.0))
    board = _Board(ir, lib, P)
    assert [t.label for t in board.terminals["N"]] == ["R1.1", "S1.1", "S1.1"]
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.stats["legal"], r.stats
    ends = {pt for t in r.tracks for pt in (t.start, t.end)}
    assert {(4.6, 4.1), (7.6, 4.1)} <= ends  # both exact pad centres: one stub each
    assert _connected(r.tracks, [(2.0, 4.0), (4.6, 4.1), (7.6, 4.1)])
    checks = _routing_checks(ir, r, lib, tmp_path)
    assert checks["pcb.routing.connectivity"].status.value == "PASS" and checks["pcb.routing.clearance"].status.value == "PASS"
