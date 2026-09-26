"""The deterministic maze router (``ai_eda.tools.routing.maze``).

Every test runs on a synthetic KiCad library written into ``tmp_path``: the
``Test:VR1`` symbol / ``Test:FP`` footprint of ``tests/test_parts_existence.py``
plus the through-hole and SMD fixture footprints written by
:func:`fixture_library` below (``Test:PAD1`` one 1.6 mm round THT pad,
``Test:SMD1`` one 1.0 mm square SMD pad on ``F.Cu``, ``Test:WALL`` a net-less
0.5 x 20 mm SMD pad on ``F.Cu``, ``Test:CAGE`` a THT pad fenced by four
unnumbered THT bars, ``Test:TINY`` a 0.2 mm SMD pad, ``Test:PAD2`` / ``Test:SMD2`` / ``Test:WALL2``
two-pad versions of the first three, ``Test:SMD054`` a 0.54 x 0.64 mm and
``Test:SMD0510`` a 0.5 x 1.0 mm SMD pad, ``Test:TRAP`` a trapezoid and
``Test:CUST`` a custom-shape pad, ``Test:DUP1`` two pads numbered "1"). Boards are a few
millimetres, so the grids are small. Nothing here claims DRC: the router's
clearances are its parameters, and the tests check the IR geometry it
promised (the ``pcb.routing.*`` validators do the same in the pipeline).
"""

from __future__ import annotations

import copy
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
from ai_eda.tools.routing.maze import BLOCKED, LAYERS, _Board
from tests.conftest import DS
from tests.test_parts_existence import make_part, synthetic_library

NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
P = RoutingParams()
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
    # two pads that share the number "1" (split thermal / mounting pads): one logical pad in KiCad
    "DUP1": '(pad "1" thru_hole circle (at -1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n  (pad "1" thru_hole circle (at 1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
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
    assert ROUTER_ID == "routing.maze" and ROUTER_VERSION == "0.1"
    for item in [*r.tracks, *r.vias]:
        prov = item.provenance
        assert prov.kind is ProvenanceKind.DERIVED and not prov.needs_verification and prov.note != UNRECORDED_ORIGIN
        assert prov.tool == ROUTER_ID and prov.tool_version == ROUTER_VERSION
        assert prov.derived_from == ["net:N", "placement:R1", "placement:R2", "params:grid=0.25,width=0.4,clearance=0.25,via=0.8/0.4,edge=0.3,via_cost=12.0,bend_cost=0.6"]
        assert prov.inputs == {}  # the calculator role map stays empty
        assert "DRC" in prov.note and "IR geometry" in prov.note
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
    assert r.tracks[0].provenance.derived_from[-1] == "params:grid=0.25,width=0.5,clearance=0.25,via=1.0/0.5,edge=0.35,via_cost=12.0,bend_cost=0.6"
    # a caller's own parameters are raised the same way, and effective_params never lowers one
    eff, raised = effective_params(ir, RoutingParams(track_width_mm=0.6, clearance_mm=0.1))
    assert eff.track_width_mm == 0.6 and eff.clearance_mm == 0.2 and raised == {"clearance_mm": (0.1, 0.2), "via_drill_mm": (0.4, 0.5), "via_diameter_mm": (0.8, 1.0), "edge_clearance_mm": (0.3, 0.35)}
    # a drill limit at or above the via diameter is refused, not guessed around
    ir.pcb.manufacturing = ManufacturingConstraints(min_via_drill_mm=assumption(0.8, note="x"))
    with pytest.raises(CompileError, match="via drill to 0.8 mm, which is not below the via diameter 0.8 mm"):
        effective_params(ir)
    # nonsense parameters are refused
    for bad in (RoutingParams(grid_mm=0.0), RoutingParams(track_width_mm=-1.0), RoutingParams(clearance_mm=-0.1), RoutingParams(via_diameter_mm=0.4, via_drill_mm=0.4), RoutingParams(bend_cost=-1.0)):
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
    """A trapezoid (rect_delta) or custom (primitives) pad has copper the library reader does not keep: refused, never modelled as its size box."""
    for name, shape in (("TRAP", "trapezoid"), ("CUST", "custom")):
        ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 10.0, 4.0), ("U1", name, 6.0, 4.0)], {"A": [("R1", "1"), ("R2", "1")], "B": [("U1", "1")]}, (12.0, 8.0))
        with pytest.raises(CompileError, match=rf"cannot route: pad U1.1 of footprint Test:{name} has shape '{shape}'; its copper is not bounded by its \(size\) box"):
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


def test_a_net_the_first_order_starves_is_routed_first_in_a_second_pass(tmp_path: Path):
    """The grid-placed synthetic astable: five nets routed in (pad count, name) order wall the 5-pad VCC net in, a fresh pass with VCC first routes all six.

    Same placements the pipeline's grid placer produces for the ``astable``
    template on the synthetic template library (THT R / C / Q, an SMD 1x03
    header); the result is a pure function of the inputs either way.
    """
    from ai_eda.ir import LibraryRef
    from tests.test_circuit_templates import template_library

    lib = template_library(tmp_path / "kicad")
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
    nets = {
        "VCC": [("J1", "1"), ("R1", "1"), ("R2", "1"), ("R3", "1"), ("R4", "1")], "Q1_C": [("R1", "2"), ("Q1", "3"), ("C1", "1")],
        "Q2_B": [("C1", "2"), ("R4", "2"), ("Q2", "2")], "OUT": [("R2", "2"), ("Q2", "3"), ("C2", "1"), ("J1", "2")],
        "Q1_B": [("C2", "2"), ("R3", "2"), ("Q1", "2")], "GND": [("J1", "3"), ("Q1", "1"), ("Q2", "1")],
    }
    ir.nets = [Net(name=n, pins=[PinRef(component_ref=r, pin_number=k) for r, k in pins], provenance=NET_P) for n, pins in nets.items()]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=48.96, height_mm=13.2), placements=placements)
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.stats["routed_nets"] == 6 and r.stats["passes"] == 2, r.stats
    assert r.stats["net_order"] == ["VCC", "GND", "Q1_B", "Q1_C", "Q2_B", "OUT"]  # the starved net first, the others in the first pass's order
    assert [t.net for t in r.tracks][0] == "VCC" and set(r.stats["net_length_mm"]) == set(nets)
    assert _geometry(r) == _geometry(route_board(copy.deepcopy(ir), template_library(tmp_path / "kicad2")))  # still a pure function of the inputs
    # a board that routes in one pass reports it
    x = board_ir(tmp_path, fixture_library(tmp_path / "kicad3"), [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    s = route_board(x, fixture_library(tmp_path / "kicad3")).stats
    assert s["passes"] == 1 and s["net_order"] == ["N"]
