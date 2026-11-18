"""The router's layer tables: the seam for N signal layers (multilayer plan, phase P3).

``ai_eda.tools.routing.maze._Board`` carries its routing layers as tables
- ``layer_names`` (the routing layers' copper names in routing-index order),
``R`` (their count), ``route_layers`` (their indices), ``T`` (via planes per
via clearance), ``barrels`` (extra search states), ``search_n`` and
``layer_index(name)`` - and the shared code reads them where it held a
two-layer literal. On every board today they are the two outer layers
(``layer_names is LAYERS``, F.Cu = 0, B.Cu = 1), one via plane and no
barrel state, so each rewrite computes the value the literal had. These
tests pin those tables and the array sizes the negotiation allocates from
them; the copper itself is pinned byte for byte by the golden digests of
``tests/test_routing_rules.py`` (``GOLDEN_0_2`` / ``GOLDEN_0_6``) and the
boards of ``tests/test_routing_fine_pitch.py``.

The plan never edits ``_search``, ``_search_rules``, ``_route_pair`` or
``_Negotiation.cover``: the search holds most of the run time and its push
order settles the heap tie-breaks. Their source is pinned here by its
SHA-256 at c270797.

Phase P4 (routing.maze 0.9, stack mode): ``route_board(..., stack=...)``
on the 4-layer rung (signal layers F.Cu / B.Cu, GND on In1.Cu, +3V3 on
In2.Cu): the typed micro plane vias, the stitch rule, the typed emission,
the 0.9 stamp and stats, every refusal, determinism, the compiled board's
pcb.routing checks, and the unchanged copper of a board routed without a
stack. All on the synthetic fixture library.

Phase P5 (routing.maze 0.9, N signal layers): the 6- / 8-layer escalation
stacks' inner signal layers, the barrel-state search (``_search_n``) with
through, blind and buried signal vias, the span-aware negotiation
(``_NegotiationN``) with VPR's schedule and the per-iteration net order, a
rule net kept on F.Cu / B.Cu, strict recovery, determinism, the array
sizes, the compiled typed vias' pcb.routing checks, and the cost oracle:
on every 2-layer fixture of ``tests/test_routing_rules.py`` the barrel
search with the through-only stack finds the same cheapest cost as
``_search``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
from array import array
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.design.stackup import ESCALATION_STACKS, GENERIC_STACKS, board_layers, escalation_stackup, generic_stackup, plane_zones
from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, ManufacturingConstraints, NetKind, Via, assumption
from ai_eda.ir.pcb import MICRO_VIA_ADJACENT
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import ROUTER_STACK_VERSION, maze
from ai_eda.tools.routing.maze import (
    _FREE,
    _STATES_PER_CELL,
    LAYERS,
    MICROVIA_SPAN_SOURCE,
    STACK_SEARCH,
    STACKED_VIA_SOURCE,
    NetRule,
    RoutingParams,
    StackSearchParams,
    _Board,
    _Negotiation,
    _Plan,
    _provenance,
    effective_params,
    route_board,
)
from ai_eda.tools.routing.stack import THROUGH, RouterStack, ViaType
from ai_eda.validation.layout import CLEARANCE_CHECK, CONNECTIVITY_CHECK
from tests.test_rf_floorplan import PLANE, _plane_board, _run, keepout, rf_library
from tests.test_routing import _routing_checks, board_ir, fixture_library
from tests.test_routing_baseline_digests import BASELINE
from tests.test_routing_rules import FIXTURES, canonical, digest
from tests.test_si_checks import stacked

P = RoutingParams()

#: SHA-256 of ``inspect.getsource`` of the functions the multilayer plan never edits, at c270797
FROZEN_SOURCE: dict[str, str] = {
    "_search": "c73e35357a093b24b5a72abde8bd0f6300d0438ebe7e5f07c4c0d2328e329f47",
    "_search_rules": "59920ab2b1c0b198193e6e988bec7c9d2464864fbeebd0031554d6b53f60356b",
    "_route_pair": "bb94cc59015d6048a8ee66300e6d242b9d60205a11bb71ff459b804d11e7f0e0",
    "_Negotiation.cover": "e031ca643315693816fa90d4c67549494d0de01b8397eb6e9a4fc98d2ad46361",
}


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _two_layer_board(tmp_path: Path, lib: KicadLibrary) -> _Board:
    """A through-hole pad and an F.Cu SMD pad on one net, and a second net, on a two-layer board."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0), ("R3", "PAD1", 3.0, 7.0), ("R4", "PAD1", 9.0, 7.0)],
                  {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0))
    return _Board(ir, lib, P)


def _assert_outer_tables(board: _Board) -> None:
    assert board.layer_names is LAYERS
    assert board.R == 2 and board.route_layers == (0, 1)
    assert board.T == 1 and board.barrels == 0 and board.search_n is False
    assert [board.layer_index(name) for name in LAYERS] == [0, 1]


def test_a_two_layer_boards_tables_are_the_outer_layers(tmp_path: Path, lib: KicadLibrary):
    board = _two_layer_board(tmp_path, lib)
    _assert_outer_tables(board)
    with pytest.raises(ValueError):
        board.layer_index("In1.Cu")  # an inner layer is no routing layer
    # a through-hole pad is on every routing layer, an SMD pad where it says
    layers = {(g.ref, g.number): g.layers for g in board.pads}
    assert layers[("R1", "1")] == (0, 1) and layers[("R2", "1")] == (0,)


def test_a_four_layer_boards_inner_layers_stay_unrouted(tmp_path: Path, lib: KicadLibrary):
    ir = _plane_board(tmp_path, lib)
    board = _Board(ir, lib, P, inner_layers=True, plane={"GND": P.track_width_mm}, plane_areas={"GND": [PLANE, PLANE]})
    _assert_outer_tables(board)
    assert board.inner_layers == ["In1.Cu", "In2.Cu"]
    # the plane vias claim both routing layers, as the literal (0, 1) did
    vias = [g for g in board.pads if g.kind == "plane-via"]
    assert len(vias) == 3 and all(g.layers == (0, 1) for g in vias)


def test_the_maps_and_the_negotiation_keep_their_two_layer_sizes(tmp_path: Path, lib: KicadLibrary):
    board = _two_layer_board(tmp_path, lib)
    n = board.n
    assert len(board.owner) == 2 and all(len(m) == n for m in board.owner)
    # the flat static map is still the F.Cu map followed by the B.Cu map
    assert board.stat == [_FREE if v is None else v for v in board.owner[0]] + [_FREE if v is None else v for v in board.owner[1]]
    keys = [(P.track_width_mm, P.clearance_mm), (0.3, 0.35)]
    neg = _Negotiation(board, P, keys, [P.clearance_mm, 0.35])
    assert neg.T == 1
    for key in keys:
        assert len(neg.t_halo[key]) == len(neg.t_ncost[key]) == 2 * n
    for c in neg.via_keys:
        assert len(neg.v_halo[c]) == len(neg.v_cost[c]) == n
    assert len(neg.hist) == 2 * n and len(neg.vhist) == n
    assert len(neg.g_best) == len(neg.prev) == 2 * n * _STATES_PER_CELL


def test_the_provenance_note_names_f_cu_b_cu_unless_a_plan_names_the_layers(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.tracks and all(t.provenance.note.startswith("grid maze route on F.Cu/B.Cu (") for t in r.tracks)
    plan = _Plan("0.4", (), "routing.maze 0.4: words")
    assert plan.layers is None
    assert _provenance(ir.nets[0], P, "negotiated", 1, True, plan=plan).note.startswith("grid maze route on F.Cu/B.Cu (")
    named = _Plan("0.4", (), "routing.maze 0.4: words", layers="F.Cu/In2.Cu/B.Cu")
    assert _provenance(ir.nets[0], P, "negotiated", 1, True, plan=named).note.startswith("grid maze route on F.Cu/In2.Cu/B.Cu (")


@pytest.mark.parametrize("name", sorted(FROZEN_SOURCE))
def test_the_functions_the_plan_never_edits_are_c270797s(name: str):
    owner, _, attr = name.rpartition(".")
    func = getattr(getattr(maze, owner), attr) if owner else getattr(maze, name)
    assert hashlib.sha256(inspect.getsource(func).encode()).hexdigest() == FROZEN_SOURCE[name]


# --------------------------------------------------------------------------- P4: stack mode on the 4-layer rung (routing.maze 0.9)

G4 = GENERIC_STACKS[4]
#: the plane zones of the 20 x 12 mm board below (the outline inset by 0.5 mm)
AREA = [(0.5, 0.5), (19.5, 0.5), (19.5, 11.5), (0.5, 11.5)]
PLANES = {"GND": [("In1.Cu", AREA)], "+3V3": [("In2.Cu", AREA)]}
MICRO_F = ("F.Cu", "In1.Cu")


def _router_stack(ir: CircuitIR) -> RouterStack:
    """The 4-layer escalation rung's router stack (P2): the board's stackup, the generic stack's micro vias and directions."""
    return RouterStack.from_stackup(ir.pcb.stackup, G4.vias, G4.title, directions=G4.directions)


def _gnd_3v3_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """``stacked(ir, 4)`` with GND on In1.Cu and +3V3 on In2.Cu: two decoupling capacitors (SMD pad 1 GND, pad 2 +3V3, on F.Cu) and
    ``S`` between two through-hole pads straight across x = 7, where C1's GND via lands one grid step east of its pad (two-pad parts
    throughout, as the Test:VR1 symbol has two pins, so the board compiles)."""
    ir = board_ir(tmp_path, lib, [("C1", "SMD2", 7.5, 6.0), ("C2", "SMD2", 13.5, 9.0), ("R1", "PAD2", 8.5, 1.5), ("R2", "PAD2", 8.5, 10.5)],
                  {"GND": [("C1", "1"), ("C2", "1")], "+3V3": [("C1", "2"), ("C2", "2")], "S": [("R1", "1"), ("R2", "1")]}, (20.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    ir.nets[1].kind = NetKind.POWER
    stacked(ir, 4)
    ir.pcb.stackup = generic_stackup(4, "fixture", confirmed=True, ground_net="GND", power_net="+3V3")
    ir.pcb.layers = board_layers(ir.pcb.stackup)
    return ir


def _gnd_only_board(tmp_path: Path, lib: KicadLibrary, gnd: list[tuple[str, float, float]]) -> CircuitIR:
    """``stacked(ir, 4)`` (GND on both planes) with GND SMD pads only - no through-hole GND pad - and a signal net ``S``."""
    parts = [(ref, "SMD1", x, y) for ref, x, y in gnd] + [("R5", "PAD1", 4.0, 14.0), ("R6", "PAD1", 16.0, 14.0)]
    ir = board_ir(tmp_path, lib, parts, {"GND": [(ref, "1") for ref, _x, _y in gnd], "S": [("R5", "1"), ("R6", "1")]}, (20.0, 18.0))
    ir.nets[0].kind = NetKind.GROUND
    return stacked(ir, 4)


def _seg_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def test_a_stack_boards_tables_name_its_via_types(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_3v3_board(tmp_path, lib)
    st = _router_stack(ir)
    board = _Board(ir, lib, P, plane={"GND": P.track_width_mm, "+3V3": P.track_width_mm}, plane_areas={n: [AREA] for n in PLANES},
                   plane_layer_areas={n: [(layer, AREA) for layer, _a in e] for n, e in PLANES.items()}, stack=st)
    assert board.layer_names == LAYERS and board.R == 2 and board.stack_pos == (0, 3)
    assert board.inner_layers == ["In1.Cu", "In2.Cu"]  # the stack's inner plane layers: stats["inner_layers"] keeps its meaning
    assert [(vt.index, vt.kind, vt.top, vt.bottom, vt.lo, vt.hi, vt.layers, vt.planes) for vt in board.vtypes] == [
        (0, "through", "F.Cu", "B.Cu", 0, 3, (0, 1), ("In1.Cu", "In2.Cu")),
        (1, "micro", "F.Cu", "In1.Cu", 0, 1, (0,), ("In1.Cu",)),
        (2, "micro", "In2.Cu", "B.Cu", 2, 3, (1,), ("In2.Cu",)),
    ]
    assert [vt.index for vt in board.signal_types] == [0] and [vt.index for vt in board.plane_types] == [1, 2]
    assert board.T == 1 and len(board.via_static_n) == 1 and board.via_static_n[0] is board.via_static  # the search reads that list
    # the GND micro vias claim F.Cu only; the +3V3 pads' vias are through vias on both routing layers
    vias = {(g.ref, g.number, g.layers) for g in board.pads if g.kind == "plane-via"}
    assert vias == {("C1", "1:plane-via", (0,)), ("C2", "1:plane-via", (0,)), ("C1", "2:plane-via", (0, 1)), ("C2", "2:plane-via", (0, 1))}
    assert board.via_layers_of(Via(net="GND", x_mm=7.0, y_mm=6.0, drill_mm=0.4, diameter_mm=0.8, layers=MICRO_F, kind="micro")) == (0,)
    assert board.via_layers_of(Via(net="GND", x_mm=7.0, y_mm=6.0, drill_mm=0.4, diameter_mm=0.8)) == (0, 1)
    assert board.plane_link_types == {"GND": [1, 1], "+3V3": [None, None]} and board.plane_stitch == {}


def test_gnd_pads_get_micro_vias_and_3v3_pads_through_vias(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_3v3_board(tmp_path, lib)
    r = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    assert r.unrouted == {} and r.version == ROUTER_STACK_VERSION == "0.9"
    gnd = [v for v in r.vias if v.net == "GND"]
    assert len(gnd) == 2 and all(v.kind == "micro" and v.layers == MICRO_F for v in gnd)
    power = [v for v in r.vias if v.net == "+3V3"]
    assert len(power) == 2 and all(v.kind is None and v.layers == LAYERS for v in power)  # no micro via reaches In2.Cu from F.Cu
    assert all(v.drill_mm == P.via_drill_mm and v.diameter_mm == P.via_diameter_mm for v in r.vias)  # one via size for every type
    rows = r.stats["plane_nets"]
    assert rows["GND"]["via_kinds"] == ["micro:F.Cu-In1.Cu"] * 2 and rows["+3V3"]["via_kinds"] == ["through:F.Cu-B.Cu"] * 2
    assert r.stats["inner_layers"] == ["In1.Cu", "In2.Cu"]
    assert r.stats["stack"] == {
        "label": "4-layer 1.6 mm FR-4",
        "signal_layers": ["F.Cu", "B.Cu"],
        "plane_layers": {"In1.Cu": "GND", "In2.Cu": "+3V3"},
        "via_types": [{"kind": "through", "top": "F.Cu", "bottom": "B.Cu", "count": 2}, {"kind": "micro", "top": "F.Cu", "bottom": "In1.Cu", "count": 2},
                      {"kind": "micro", "top": "In2.Cu", "bottom": "B.Cu", "count": 0}],
        "stitch": {},
    }


def test_a_b_cu_track_passes_under_a_gnd_micro_via(tmp_path: Path, lib: KicadLibrary):
    """S runs straight down x = 7 on B.Cu under C1's micro via at (7, 6); routed with a through via there, it must keep its clearance."""
    reach = P.via_diameter_mm / 2.0 + P.clearance_mm + P.track_width_mm / 2.0  # a track centre nearer than this to a via breaks the clearance
    ir = _gnd_3v3_board(tmp_path, lib)
    r = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    [via] = [v for v in r.vias if v.net == "GND" and (v.x_mm, v.y_mm) == (7.0, 6.0)]
    assert via.kind == "micro"
    under = [t for t in r.tracks if t.net == "S" and _seg_distance((7.0, 6.0), t.start, t.end) < reach]
    assert under and all(t.layer == "B.Cu" for t in under)
    through = route_board(_gnd_3v3_board(tmp_path, lib), lib, plane_nets=PLANES, inner_layers=True)
    assert through.unrouted == {} and [(v.x_mm, v.y_mm, v.kind) for v in through.vias if v.net == "GND"] == [(7.0, 6.0, None), (13.0, 9.0, None)]
    s = [t for t in through.tracks if t.net == "S"]
    assert s and all(_seg_distance((7.0, 6.0), t.start, t.end) >= reach - 1e-9 for t in s)  # S detours around the through via


def test_the_stitch_rule_gives_the_first_pad_the_through_via(tmp_path: Path, lib: KicadLibrary):
    """GND on In1.Cu and In2.Cu and no through-hole GND pad: C1 takes the through via that joins both planes, C2 / C3 micro vias."""
    ir = _gnd_only_board(tmp_path, lib, [("C1", 4.0, 4.0), ("C2", 10.0, 4.0), ("C3", 16.0, 4.0)])
    st = _router_stack(ir)
    assert st.planes == (("In1.Cu", "GND"), ("In2.Cu", "GND")) and len(st.signal) == 2
    r = route_board(ir, lib, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]}, stack=st)
    assert r.unrouted == {}
    assert [(v.x_mm, v.kind, v.layers) for v in r.vias if v.net == "GND"] == [(5.0, None, LAYERS), (11.0, "micro", MICRO_F), (17.0, "micro", MICRO_F)]
    assert r.stats["stack"]["stitch"] == {"GND": "C1.1"}
    assert r.stats["plane_nets"]["GND"]["via_kinds"] == ["through:F.Cu-B.Cu", "micro:F.Cu-In1.Cu", "micro:F.Cu-In1.Cu"]
    # with a through-hole GND pad its barrel stitches the planes: every SMD pad takes its micro via (the P0 plane board)
    lib2 = rf_library(tmp_path / "rf")
    plane = _plane_board(tmp_path, lib2)
    r = route_board(plane, lib2, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]}, stack=_router_stack(plane))
    assert r.stats["stack"]["stitch"] == {} and [v.kind for v in r.vias if v.net == "GND"] == ["micro"] * 3
    # the rule counts the stack's plane layers of the net: with a zone on In1.Cu only, In2.Cu (filled from the stack) still gets its via
    ir = _gnd_only_board(tmp_path / "one", lib, [("C1", 4.0, 4.0), ("C2", 10.0, 4.0), ("C3", 16.0, 4.0)])
    r = route_board(ir, lib, plane_nets={"GND": [("In1.Cu", PLANE)]}, stack=_router_stack(ir))
    assert r.stats["stack"]["stitch"] == {"GND": "C1.1"} and [v.kind for v in r.vias if v.net == "GND"] == [None, "micro", "micro"]


def test_without_a_through_via_site_the_net_is_placed_as_0_4_places_it(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """A B.Cu track keep-out over both GND pads: their micro walks find sites, no through walk does - so the micro vias are undone and
    the net gets exactly the through-only copper the same board gets without a stack (0.6's escape vias here)."""
    gnd = [("C1", 4.0, 4.0), ("C2", 10.0, 4.0)]
    ko = [keepout("KB", [0.5, 0.5, 15.0, 9.0], forbids=["tracks"], layers=["B.Cu"])]
    planes = {"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]}
    walks: list[tuple[str, tuple[int, ...] | None, bool]] = []
    walk = _Board._plane_walk

    def spy(self, t, idx, layer, width, reach, placed, areas, via_layers=None):
        site = walk(self, t, idx, layer, width, reach, placed, areas, via_layers)
        walks.append((t.label, via_layers, site is not None))
        return site

    monkeypatch.setattr(_Board, "_plane_walk", spy)
    ir = _gnd_only_board(tmp_path, lib, gnd)
    r = route_board(ir, lib, plane_nets=planes, keepouts=ko, stack=_router_stack(ir))
    assert r.stats["stack"]["stitch"] == {"GND": "through-only"}
    # the first try: each pad's through walk fails, its micro walk finds a site; then the through-only placement (both fail again)
    assert walks == [("C1.1", None, False), ("C1.1", (0,), True), ("C2.1", None, False), ("C2.1", (0,), True), ("C1.1", None, False),
                     ("C2.1", None, False)]
    monkeypatch.undo()
    old = route_board(_gnd_only_board(tmp_path, lib, gnd), lib, plane_nets=planes, keepouts=ko, inner_layers=True)

    def copper(res):
        return ([(t.layer, t.start, t.end, t.width_mm) for t in res.tracks if t.net == "GND"],
                [(v.x_mm, v.y_mm, v.layers, v.kind) for v in res.vias if v.net == "GND"])

    assert copper(r) == copper(old) and len(copper(r)[1]) == 2 and all(kind is None for *_, kind in copper(r)[1])
    assert r.unrouted == old.unrouted == {}


def test_every_item_is_stamped_0_9_with_the_stack_entry_last(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_3v3_board(tmp_path, lib)
    r = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    entry = ("stack:4-layer 1.6 mm FR-4;signal=F.Cu+B.Cu;planes=In1.Cu=GND,In2.Cu=+3V3;"
             "vias=through:F.Cu-B.Cu,micro:F.Cu-In1.Cu,micro:In2.Cu-B.Cu;size=0.8/0.4;cost=12.0;model=span+barrel+xy_all_vias+no_stacked;"
             "kicad_minimums=track:0.2,clearance:0.2,edge:0.5,via:0.5/0.3,hole_to_hole:0.25")
    words = ("routing.maze 0.9: plane net(s) +3V3, GND join their pads by vias to the plane, never by tracks; routed on the signal layers F.Cu, "
             "B.Cu of the 4-layer 1.6 mm FR-4 stack; via types through F.Cu-B.Cu, micro F.Cu-In1.Cu, micro In2.Cu-B.Cu (a generic stack's "
             "stated choice, not a fab's: one via size and one via cost for every type, every via kept clear of every other via in x-y, no "
             "stacked vias); rule nets and coupled pairs stay on F.Cu/B.Cu with through vias; every track and escape stub at least 0.2 mm "
             "wide, 0.2 mm from other nets' copper and copper 0.5 mm from the board edge, every via at least 0.5 mm across with a hole of "
             "at least 0.3 mm - the minimums kicad-cli DRC checks this board against (the project file's rules, else KiCad's own)")
    items = [*r.tracks, *r.vias]
    assert items and all(i.provenance.tool_version == "0.9" and i.provenance.derived_from[-1] == entry for i in items)
    assert all(i.provenance.note.startswith("grid maze route on F.Cu/B.Cu (") and f"; {words}; validity is decided" in i.provenance.note for i in items)


def _refused(ir: CircuitIR, lib: KicadLibrary, stack: RouterStack, **kw) -> str:
    with pytest.raises(CompileError) as e:
        route_board(ir, lib, stack=stack, **kw)
    return str(e.value)


def test_every_refusal_names_its_reason(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_3v3_board(tmp_path, lib)
    st = _router_stack(ir)
    assert _refused(ir, lib, st, inner_layers=True) == "cannot route: pass either inner_layers=True or a stack, not both"
    two = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    assert _refused(two, lib, st) == "cannot route: the stack's copper ['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu'] is not ir.pcb.layers ['F.Cu', 'B.Cu']"
    assert _refused(ir, lib, dataclasses.replace(st, signal=("F.Cu",))) == "cannot route: the stack's signal layers must include F.Cu and B.Cu (got ['F.Cu'])"
    both = dataclasses.replace(st, signal=("F.Cu", "In1.Cu", "B.Cu"))
    assert _refused(ir, lib, both) == "cannot route: In1.Cu is both a signal layer and a plane layer of the stack"
    stray = dataclasses.replace(st, planes=(("In1.Cu", "GND"), ("In3.Cu", "+3V3")))
    assert _refused(ir, lib, stray) == ("cannot route: the stack's signal / plane layer(s) ['In3.Cu'] are not among its copper "
                                        "['F.Cu', 'In1.Cu', 'In2.Cu', 'B.Cu']")
    assert _refused(ir, lib, dataclasses.replace(st, vias=st.vias[1:])) == "cannot route: the stack's first via type must be the through via F.Cu-B.Cu"
    bad = dataclasses.replace(st, vias=(THROUGH, ViaType("micro", "F.Cu", "In2.Cu")))
    assert _refused(ir, lib, bad) == f"cannot route: via type micro F.Cu-In2.Cu: {MICRO_VIA_ADJACENT}"
    assert _refused(ir, lib, dataclasses.replace(st, vias=(THROUGH, THROUGH))) == (
        "cannot route: via type through F.Cu-B.Cu: 'through' is not a via kind (blind, buried, micro)")
    assert _refused(ir, lib, dataclasses.replace(st, vias=(*st.vias, st.vias[1]))) == "cannot route: via type micro F.Cu-In1.Cu is listed twice"
    assert _refused(ir, lib, st, plane_nets={"GND": [("In2.Cu", AREA)]}) == (
        "cannot route: plane net 'GND' has a zone on In2.Cu, which is not a plane layer of that net in the stack")
    # P5 lifted P4's last refusal (an inner signal layer, or a second via type joining two signal layers): see the P5 tests below
    # the board refuses a stack it is given directly, too
    with pytest.raises(CompileError, match="pass either inner_layers=True or a stack, not both"):
        _Board(ir, lib, P, inner_layers=True, stack=st)


def test_two_runs_give_the_same_bytes(tmp_path: Path):
    def run(tmp: Path):
        lib = fixture_library(tmp / "kicad")
        ir = _gnd_3v3_board(tmp, lib)
        return route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert canonical(a) == canonical(b)


def test_the_compiled_board_carries_micro_vias_and_passes_connectivity_and_clearance(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_3v3_board(tmp_path, lib)
    r = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    ir.pcb.zones = plane_zones(ir.pcb.stackup, ir.pcb.outline, 0.5)
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    tree = sexpr.parse_file(Path(art.path))
    tokens = [(v[1] if isinstance(v[1], str) else None, tuple(str(x) for x in sexpr.args(sexpr.find(v, "layers")))) for v in sexpr.find_all(tree, "via")]
    assert sorted(tokens, key=repr) == sorted([("micro", MICRO_F)] * 2 + [(None, LAYERS)] * 2, key=repr)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(P.clearance_mm, note="the router's clearance as the limit (a limit is a limit)"))
    got = _run("pcb.routing", ir, tmp_path, lib)
    # S on B.Cu passes under a micro via F.Cu-In1.Cu: compared on the via's span only, it keeps every clearance
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message
    conn = got[CONNECTIVITY_CHECK]
    rows = {row["net"]: row for row in conn.details["nets"]}
    assert rows["S"]["status"] == "PASS" and conn.status.value == "NOT_VERIFIED"
    assert all(rows[n]["status"] == "NOT_VERIFIED" and rows[n]["plane"] is True for n in ("GND", "+3V3"))  # joined through the fill, as 0.4's
    # the same via drawn through would reach B.Cu, where S passes beneath it: a short the check names
    ir.pcb.vias = [v.model_copy(update={"kind": None, "layers": LAYERS}) if (v.x_mm, v.y_mm) == (7.0, 6.0) else v for v in ir.pcb.vias]
    assert _run("pcb.routing", ir, tmp_path, lib)[CLEARANCE_CHECK].status.value == "FAIL"


def test_the_same_board_without_a_stack_keeps_its_c270797_copper(tmp_path: Path):
    """The P0 plane board: with the stack its SMD GND pads take micro vias (0.9); with ``stack=None`` it is c270797's, byte for byte."""
    planes = {"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]}
    lib = rf_library(tmp_path / "a" / "kicad")
    ir = _plane_board(tmp_path / "a", lib)
    assert digest(route_board(ir, lib, inner_layers=True, plane_nets=planes, stack=None)) == BASELINE["plane_gnd_4layer"]
    lib = rf_library(tmp_path / "b" / "kicad")
    ir = _plane_board(tmp_path / "b", lib)
    r = route_board(ir, lib, plane_nets=planes, stack=_router_stack(ir))
    assert r.version == "0.9" and [v.kind for v in r.vias if v.net == "GND"] == ["micro"] * 3


def test_the_via_rules_name_their_sources():
    """The rules the stack mode applies name their documents (research_synthesis.json), and what was not opened says so."""
    assert "IPC-T-50M" in MICROVIA_SPAN_SOURCE and "NCAB 'Microvias'" in MICROVIA_SPAN_SOURCE and "not opened" in MICROVIA_SPAN_SOURCE
    assert all(s in STACKED_VIA_SOURCE for s in ("NCAB 'Microvias'", "Wurth", "Eurocircuits", "IPC-WP-023", "not opened"))
    doc = maze.__doc__
    assert "MICROVIA_SPAN_SOURCE" in doc and "STACKED_VIA_SOURCE" in doc and "routing.maze 0.9" in doc


# --------------------------------------------------------------------------- P5: N signal layers (routing.maze 0.9)

#: the keep-out that walls R2's F.Cu and B.Cu side on the 20 x 10 mm boards below, so R2 is reached on In2.Cu / In3.Cu only
KO_RIGHT = keepout("KR", [12.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["F.Cu", "B.Cu"])
#: the GND plane zone of :func:`_grid_board` (the 20 x 12 mm outline inset by 0.5 mm)
GRID_AREA = [(0.5, 0.5), (19.5, 0.5), (19.5, 11.5), (0.5, 11.5)]


def _n_stack(ir: CircuitIR, n: int = 6, *, directions: bool = True, ground: str = "GND", power: str | None = None) -> RouterStack:
    """The ``n``-layer escalation stack (P2) on ``ir``: its stackup and layers, and the router stack with its vias (and directions)."""
    ir.pcb.stackup = escalation_stackup(n, ground_net=ground, power_net=power)
    ir.pcb.layers = board_layers(ir.pcb.stackup)
    g = ESCALATION_STACKS[n]
    return RouterStack.from_stackup(ir.pcb.stackup, g.vias, g.title, directions=g.directions if directions else ())


def _through_vs_blind_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """N from the F.Cu SMD pad R1 (4, 5) to the through-hole pad R2 (16, 5), whose F.Cu / B.Cu side :data:`KO_RIGHT` walls off."""
    return board_ir(tmp_path, lib, [("R1", "SMD1", 4.0, 5.0), ("R2", "PAD1", 16.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (20.0, 10.0))


def _via_conflict_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """A and B each from a top 0.2 mm pad to a bottom one, 1 mm apart: in the free first pass their through vias land closer than
    ``via_diameter + clearance``."""
    parts = [("R1", "TINY", 4.0, 4.0), ("R2", "TINY", 8.0, 4.0), ("R3", "TINY", 4.0, 5.0), ("R4", "TINY", 8.0, 5.0)]
    return board_ir(tmp_path, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 9.0),
                    sides={"R2": BoardSide.BOTTOM, "R4": BoardSide.BOTTOM})


def _grid_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """Four horizontal nets H0-H3 crossing four vertical nets V0-V3, every pad an F.Cu SMD pad, plus a GND pair of through-hole pads."""
    parts: list[tuple[str, str, float, float]] = [("G1", "PAD1", 1.5, 11.0), ("G2", "PAD1", 18.5, 11.0)]
    nets: dict[str, list[tuple[str, str]]] = {"GND": [("G1", "1"), ("G2", "1")]}
    for q in range(4):
        parts += [(f"H{q}a", "SMD1", 2.0, 3.0 + 2.0 * q), (f"H{q}b", "SMD1", 18.0, 3.0 + 2.0 * q)]
        parts += [(f"V{q}a", "SMD1", 6.0 + 2.0 * q, 1.0), (f"V{q}b", "SMD1", 6.0 + 2.0 * q, 11.0)]
        nets[f"H{q}"] = [(f"H{q}a", "1"), (f"H{q}b", "1")]
        nets[f"V{q}"] = [(f"V{q}a", "1"), (f"V{q}b", "1")]
    ir = board_ir(tmp_path, lib, parts, nets, (20.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    return ir


def _no_shared_via_cell(r) -> bool:
    cells = [(v.x_mm, v.y_mm) for v in r.vias]
    return len(cells) == len(set(cells))


def test_an_n_layer_boards_tables_add_the_inner_signal_layers_in_stack_order(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0)], {"A": [("R1", "1"), ("R2", "1")]}, (12.0, 10.0))
    board = _Board(ir, lib, P, stack=_n_stack(ir))
    assert board.layer_names == ("F.Cu", "B.Cu", "In2.Cu", "In3.Cu") and board.R == 4 and board.route_layers == (0, 1, 2, 3)
    assert board.stack_pos == (0, 5, 2, 3) and board.inner_layers == ["In1.Cu", "In4.Cu"]
    # the signal types: the through via first, then span length, then the upper end; every type's layers in stack order
    assert [(vt.label, vt.layers) for vt in board.signal_types] == [
        ("through:F.Cu-B.Cu", (0, 2, 3, 1)), ("buried:In2.Cu-In3.Cu", (2, 3)), ("blind:F.Cu-In3.Cu", (0, 2, 3))]
    assert [vt.label for vt in board.plane_types] == ["micro:F.Cu-In1.Cu", "micro:In4.Cu-B.Cu"]
    assert board.types_of == ((0, 2), (0,), (0, 1, 2), (0, 1, 2))
    assert board.layer_dirs == ("horizontal", "vertical", "vertical", "horizontal")
    assert board.T == 3 and board.search_n is True and board.barrels == 3 * board.n and len(board.via_static_n) == 3
    # the maps cover every routing layer; a through-hole pad is on all four, an SMD pad where it says
    assert len(board.owner) == 4 and len(board.stat) == 4 * board.n
    layers = {(g.ref, g.number): g.layers for g in board.pads}
    assert layers[("R1", "1")] == (0, 1, 2, 3) and layers[("R2", "1")] == (0,)
    # the arrays: R * n track cells, T * n via planes, R * n * 5 + T * n search states, as array-module arrays
    neg = maze._NegotiationN(board, P)
    n = board.n
    assert len(neg.t_halo[neg.key0]) == len(neg.t_ncost[neg.key0]) == len(neg.hist) == 4 * n
    assert len(neg.v_halo[neg.vkey0]) == len(neg.v_cost[neg.vkey0]) == len(neg.vhist) == 3 * n
    assert len(neg.g_best) == len(neg.prev) == 4 * n * _STATES_PER_CELL + 3 * n
    assert (neg.g_best.typecode, neg.prev.typecode, neg.t_halo[neg.key0].typecode, neg.v_cost[neg.vkey0].typecode) == ("d", "i", "i", "d")
    # each type's via plane starts at via_cost times its kind's factor (through 1, blind / buried 1.5); the free pass first
    assert [neg.v_cost[neg.vkey0][t * n] for t in range(3)] == [12.0, 18.0, 18.0] and neg.pres == 0.0


def test_the_8_layer_board_and_the_flat_two_type_stack_are_routed_now(tmp_path: Path, lib: KicadLibrary):
    """P4 refused these with "need the N-layer search"; P5 routes them (an inner signal layer; a second type joining two signal layers)."""
    ir = _gnd_3v3_board(tmp_path, lib)
    st = _n_stack(ir, 8, power="+3V3")
    r = route_board(ir, lib, plane_nets={"GND": [("In1.Cu", AREA), ("In6.Cu", AREA)], "+3V3": [("In4.Cu", AREA)]}, stack=st)
    assert r.unrouted == {} and r.version == "0.9"
    assert r.stats["stack"]["search"]["routing_layers"] == ["F.Cu", "B.Cu", "In2.Cu", "In3.Cu", "In5.Cu"]
    two = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0),
                   sides={"R2": BoardSide.BOTTOM})
    flat = RouterStack("2-layer", LAYERS, LAYERS, (), (THROUGH, ViaType("micro", "F.Cu", "B.Cu")))
    r = route_board(two, lib, stack=flat)
    assert r.unrouted == {} and [v.kind for v in r.vias] == [None]  # the through via is the cheaper of the two (factor 1 < 2)
    assert r.stats["stack"]["search"]["signal_types"] == ["through:F.Cu-B.Cu", "micro:F.Cu-B.Cu"]


def test_a_net_walled_off_f_cu_and_b_cu_routes_in2_to_in3_by_a_buried_via(tmp_path: Path, lib: KicadLibrary):
    """F.Cu / B.Cu are walled by a track keep-out, In2.Cu by one on the right half, In3.Cu on the left: one buried via joins them."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 4.0, 5.0), ("R2", "PAD1", 16.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (20.0, 10.0))
    ko = [keepout("KFB", [0.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["F.Cu", "B.Cu"]),
          keepout("K2", [12.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["In2.Cu"]),
          keepout("K3", [0.0, 0.0, 8.0, 10.0], forbids=["tracks"], layers=["In3.Cu"])]
    r = route_board(ir, lib, keepouts=ko, stack=_n_stack(ir))
    assert r.unrouted == {} and r.version == "0.9"
    [via] = r.vias
    assert via.kind == "buried" and via.layers == ("In2.Cu", "In3.Cu") and 8.0 < via.x_mm < 12.0
    assert {t.layer for t in r.tracks} == {"In2.Cu", "In3.Cu"}
    assert r.stats["stack"]["via_types"][-1] == {"kind": "buried", "top": "In2.Cu", "bottom": "In3.Cu", "count": 1}


def test_the_through_via_is_taken_until_its_b_cu_is_congested_then_the_blind_via(tmp_path: Path, lib: KicadLibrary):
    """Through 1 < blind 1.5 (NXP AN10778 sec. 4-5): uncongested, N leaves R1 by a through via; with another net's copper over
    B.Cu wherever F.Cu is open, the through via's plane is congested there and the blind F.Cu-In3.Cu via (no B.Cu) is cheaper."""
    ir = _through_vs_blind_board(tmp_path, lib)
    st = _n_stack(ir)
    r = route_board(ir, lib, keepouts=[KO_RIGHT], stack=st)
    assert r.unrouted == {} and [(v.kind, v.layers) for v in r.vias] == [(None, LAYERS)]
    assert {t.layer for t in r.tracks} == {"F.Cu", "In3.Cu"}  # In3.Cu runs horizontally (its preferred direction), In2.Cu would not
    board = _Board(ir, lib, P, keepouts=[KO_RIGHT], stack=st)
    n, idx = board.n, board.net_index["N"]
    t1, t2 = board.terminals["N"]
    seeds = [t1.cell]  # R1 on F.Cu
    targets = {r * n + t2.cell for r in board.usable_layers(t2, idx)}
    assert {c // n for c in targets} == {2, 3}
    full = (0, 0, board.nx - 1, board.ny - 1)
    blind = [vt.label for vt in board.signal_types].index("blind:F.Cu-In3.Cu")
    neg = maze._NegotiationN(board, P)
    assert maze._search_n(neg, idx, seeds, targets, divmod(t2.cell, board.nx), full, False)[1] == [0]
    # another net's B.Cu copper over x < 12 mm (where F.Cu is open), at the present factor 1
    band = maze._NetRoute(path_cells={n + k for k in range(n) if board.xy(k)[0] < 12.0})
    band.keyed, band.via_key = [(neg.key0, band.path_cells)], neg.vkey0
    neg.cover(band)
    neg.add(band)
    neg.pres = 1.0
    neg.recompute()
    path, types = maze._search_n(neg, idx, seeds, targets, divmod(t2.cell, board.nx), full, False)
    assert types == [blind] and all(c // n != 1 for c in path)


def test_a_branch_leaves_a_through_vias_barrel_on_an_inner_layer_without_a_second_via(tmp_path: Path, lib: KicadLibrary):
    """R1 (F.Cu) and R2 (B.Cu, bottom side) are joined by a through via; R3 is reachable on In2.Cu / In3.Cu only: its branch starts
    on the via's barrel."""
    parts = [("R1", "SMD1", 4.0, 5.0), ("R2", "SMD1", 4.0, 8.0), ("R3", "PAD1", 16.0, 5.0)]
    ir = board_ir(tmp_path, lib, parts, {"N": [("R1", "1"), ("R2", "1"), ("R3", "1")]}, (20.0, 10.0), sides={"R2": BoardSide.BOTTOM})
    r = route_board(ir, lib, keepouts=[KO_RIGHT], stack=_n_stack(ir))
    assert r.unrouted == {}
    [via] = r.vias
    assert via.kind is None
    inner = [t for t in r.tracks if t.layer in ("In2.Cu", "In3.Cu")]
    assert inner and (via.x_mm, via.y_mm) in {pt for t in inner for pt in (t.start, t.end)}
    assert {t.layer for t in r.tracks} == {"F.Cu", "B.Cu", inner[0].layer}


def test_a_net_never_drills_a_second_via_where_its_tree_has_one(tmp_path: Path, lib: KicadLibrary):
    """``no_via``: a cell holding a via of the net's tree is no via site of a later branch."""
    ir = _through_vs_blind_board(tmp_path, lib)
    board = _Board(ir, lib, P, keepouts=[KO_RIGHT], stack=_n_stack(ir))
    n, idx = board.n, board.net_index["N"]
    t1, t2 = board.terminals["N"]
    targets = {r * n + t2.cell for r in board.usable_layers(t2, idx)}
    full = (0, 0, board.nx - 1, board.ny - 1)
    neg = maze._NegotiationN(board, P)
    path, _types = maze._search_n(neg, idx, [t1.cell], targets, divmod(t2.cell, board.nx), full, False)
    vias = {a % n for a, b in zip(path, path[1:]) if a // n != b // n}
    again = maze._search_n(neg, idx, [t1.cell], targets, divmod(t2.cell, board.nx), full, False, no_via=vias)
    assert again is not None
    moved = {a % n for a, b in zip(again[0], again[0][1:]) if a // n != b // n}
    assert moved and not moved & vias


def test_a_signal_via_keeps_its_distance_from_a_plane_via_whatever_the_spans(tmp_path: Path, lib: KicadLibrary):
    """A buried In2.Cu-In3.Cu via shares no layer with a GND micro via F.Cu-In1.Cu, yet keeps ``via_diameter + clearance`` from it
    (module docstring: no stacked vias, every via clear of every other in x-y)."""
    ir = board_ir(tmp_path, lib, [("C1", "SMD1", 10.0, 5.0), ("C2", "SMD1", 4.0, 5.0)], {"GND": [("C1", "1"), ("C2", "1")]}, (20.0, 10.0))
    ir.nets[0].kind = NetKind.GROUND
    area = [(0.5, 0.5), (19.5, 0.5), (19.5, 9.5), (0.5, 9.5)]
    board = _Board(ir, lib, P, plane={"GND": P.track_width_mm}, plane_areas={"GND": [area]}, plane_layer_areas={"GND": [("In1.Cu", area)]},
                   stack=_n_stack(ir, power="VCC"))  # GND is In1.Cu's plane only: no stitching through via
    vias = [g for g in board.pads if g.kind == "plane-via"]
    assert len(vias) == 2 and all(g.layers == (0,) for g in vias)  # micro F.Cu-In1.Cu: claimed on F.Cu only
    buried = [vt.label for vt in board.signal_types].index("buried:In2.Cu-In3.Cu")
    reach = P.via_diameter_mm + P.clearance_mm
    near, beyond = [], []
    for k in range(board.n):
        x, y = board.xy(k)
        d = min(math.hypot(x - g.cx, y - g.cy) for g in vias)
        if d < reach - 1e-9:
            near.append(board.via_static_at(k, buried))
        elif d < reach + 0.5 and min(abs(x - 10.0), abs(x - 4.0)) > 1.0:
            beyond.append(board.via_static_at(k, buried))
    assert near and set(near) == {maze.BLOCKED} and _FREE in beyond


def test_vpr_schedule_history_rows_and_a_per_iteration_net_order(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Iteration 1 is a free pass (present factor 0): A's and B's through vias collide (two over-used via cells of the through type);
    iteration 2 runs at 0.5, rerouting A and B in sha256 order, and is legal."""
    calls: list[str] = []
    route_net = maze._route_net

    def spy(neg, net, *a, **k):
        calls.append(net.name)
        return route_net(neg, net, *a, **k)

    monkeypatch.setattr(maze, "_route_net", spy)
    ir = _via_conflict_board(tmp_path, lib)
    r = route_board(ir, lib, stack=_n_stack(ir))
    assert r.unrouted == {} and r.stats["legal"] is True and r.stats["iterations"] == 2
    first, second = r.stats["history"]
    assert first["present_factor"] == 0.0 and first["conflicting"] == ["A", "B"]
    assert first["overused_vias"] == {"through:F.Cu-B.Cu": 2, "buried:In2.Cu-In3.Cu": 0, "blind:F.Cu-In3.Cu": 0}
    assert second["present_factor"] == 0.5 and second["overused_cells"] == 0 and first["states_touched"] > 0 and second["states_touched"] > 0
    assert calls[:2] == ["A", "B"] and calls[2:] == sorted(["A", "B"], key=lambda name: maze._shuffle_key(2, name))
    assert maze._shuffle_key(2, "A") == hashlib.sha256(b"2:A").hexdigest()
    assert _no_shared_via_cell(r)


def test_the_negotiation_advances_on_vprs_schedule(tmp_path: Path, lib: KicadLibrary):
    ir = _through_vs_blind_board(tmp_path, lib)
    board = _Board(ir, lib, P, stack=_n_stack(ir))
    neg = maze._NegotiationN(board, P)
    assert neg.pres == 0.0
    neg.advance(1, {5: 2}, {7: 1})
    assert neg.pres == 0.5 and neg.hist[5] == 0.0 and neg.vhist[7] == 0.0  # no history after the free pass
    neg.advance(2, {5: 2}, {7: 1})
    assert math.isclose(neg.pres, 0.65) and neg.hist[5] == 2.0 and neg.vhist[7] == 1.0  # acc += overuse * acc_fac
    assert neg.ncost[5] == P.base_cost * 3.0 and neg.vcost[7] == (P.via_cost + P.base_cost) * 2.0 - P.base_cost
    for it in range(3, 60):
        neg.advance(it, {}, {})
    assert neg.pres == 1000.0  # capped at max_pres_fac
    assert STACK_SEARCH == StackSearchParams() and STACK_SEARCH.derived_from_entry() == (
        "search=barrel;astar_fac=1.2;wrong_way=2.5;via_factors=through:1.0,blind:1.5,buried:1.5,micro:2.0;negotiation=vpr:"
        "first_iter_pres_fac=0.0,initial_pres_fac=0.5,pres_fac_mult=1.3,max_pres_fac=1000.0,acc_fac=1.0;order=sha256(iteration:net)")
    for bad in (StackSearchParams(astar_fac=0.9), StackSearchParams(wrong_way=0.5), StackSearchParams(via_factors=(("through", 1.0),)),
                StackSearchParams(via_factors=(("through", 1.0), ("blind",), ("buried", 1.0), ("micro", 1.0))),
                StackSearchParams(via_factors=(("through", 1.0), ("blind", 0.5), ("buried", 1.0), ("micro", 1.0))), StackSearchParams(acc_fac=-1.0)):
        with pytest.raises(CompileError, match="stack search parameter"):
            bad.check()


def test_strict_recovery_routes_a_dropped_net_on_another_layer(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """No preferred directions and one iteration: the free pass puts A and B both on F.Cu, crossing; the end game (P8) reroutes A
    against B's copper as an obstacle - on another layer, legal by construction - so nothing is dropped and the board ends legal.
    With the end game switched off, the drop phase does the same: A is dropped and recovered there."""
    parts = [("R1", "PAD1", 2.0, 5.0), ("R2", "PAD1", 18.0, 5.0), ("R3", "PAD1", 10.0, 1.5), ("R4", "PAD1", 10.0, 8.5)]
    ir = board_ir(tmp_path, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (20.0, 10.0))
    r = route_board(ir, lib, RoutingParams(max_iterations=1), stack=_n_stack(ir, directions=False))
    assert r.stats["legal"] is True and r.stats["dropped"] == [] and r.unrouted == {}
    assert r.stats["stack"]["search"]["end_game"]["moved"] == [{"net": "A", "how": "rerouted", "ripped_up": []}]
    assert not {t.layer for t in r.tracks if t.net == "A"} & {t.layer for t in r.tracks if t.net == "B"}
    assert all(t.provenance.note.startswith("grid maze route on F.Cu/In2.Cu/In3.Cu/B.Cu (routed again against the legal copper as an obstacle "
                                            "in the end game") for t in r.tracks if t.net == "A")
    got = _routing_checks(ir, r, lib, tmp_path)
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message
    monkeypatch.setattr(maze, "_end_game", lambda neg, order, routes, route_one: [])
    old = route_board(ir, lib, RoutingParams(max_iterations=1), stack=_n_stack(ir, directions=False))
    assert old.stats["legal"] is False and old.stats["dropped"] == ["A"] and old.stats["recovered"] == ["A"] and old.unrouted == {}
    assert all(t.provenance.note.startswith("grid maze route on F.Cu/In2.Cu/In3.Cu/B.Cu (routed against the legal copper")
               for t in old.tracks if t.net == "A")
    assert [(t.layer, t.start, t.end) for t in old.tracks if t.net == "A"] == [(t.layer, t.start, t.end) for t in r.tracks if t.net == "A"]


def test_a_rule_net_on_an_8_layer_board_stays_on_f_cu_b_cu_with_through_vias(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0), ("R3", "PAD1", 3.0, 8.0), ("R4", "PAD1", 9.0, 8.0)],
                  {"N": [("R1", "1"), ("R2", "1")], "K": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0), sides={"R2": BoardSide.BOTTOM})
    st = _n_stack(ir, 8)
    rules = {"N": NetRule(width_mm=0.3), "K": NetRule(width_mm=0.3)}
    r = route_board(ir, lib, rules=rules, stack=st)
    assert r.unrouted == {} and r.version == "0.9"
    assert {t.layer for t in r.tracks} <= set(LAYERS) and [v.kind for v in r.vias if v.net == "N"] == [None]
    # with F.Cu and B.Cu walled off between their pads the rule nets have no route (they never take an inner layer); K without its
    # rule takes one
    ko = [keepout("KW", [5.0, 0.0, 7.0, 10.0], forbids=["tracks"], layers=["F.Cu", "B.Cu"])]
    r = route_board(ir, lib, rules=rules, keepouts=ko, stack=st)
    assert set(r.unrouted) == {"N", "K"}
    plain = route_board(ir, lib, rules={"N": NetRule(width_mm=0.3)}, keepouts=ko, stack=st)
    assert "K" not in plain.unrouted and {t.layer for t in plain.tracks if t.net == "K"} - set(LAYERS)


def _outside(kid: str, layer: str, x0: float, y0: float, x1: float, y1: float) -> list:
    """Track keep-outs on ``layer`` over the 14 x 10 mm board except the window ``x0..x1`` x ``y0..y1``."""
    return [keepout(f"{kid}a", [0.0, 0.0, 14.0, y0], forbids=["tracks"], layers=[layer]),
            keepout(f"{kid}b", [0.0, y1, 14.0, 10.0 - y1], forbids=["tracks"], layers=[layer]),
            keepout(f"{kid}c", [0.0, y0, x0, y1 - y0], forbids=["tracks"], layers=[layer]),
            keepout(f"{kid}d", [x1, y0, 14.0 - x1, y1 - y0], forbids=["tracks"], layers=[layer])]


def _holes_apart(r, gap: float) -> bool:
    """Whether every two vias of one net keep ``gap`` between their centres."""
    return all(math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) >= gap - 1e-9 for k, a in enumerate(r.vias) for b in r.vias[k + 1:] if a.net == b.net)


def test_a_nets_own_vias_keep_their_drills_apart(tmp_path: Path, lib: KicadLibrary):
    """Review finding (the reviewer's probe): R1 is reachable on In2.Cu only, R2 on In3.Cu only, R3 is an F.Cu SMD pad; no through via fits
    (B.Cu kept out) and the inner windows leave a few via sites. The first branch drills a buried In2.Cu-In3.Cu via at (7.25, 5), the second
    needed a blind F.Cu-In3.Cu via and took the next grid cell (7.5, 5) - 0.25 mm between two 0.4 mm drills. A net's own vias now keep the
    via drill plus the hole gap between centres (hole to hole), whatever their spans: the clearance, or KiCad's 0.25 mm hole-to-hole minimum
    where it is larger (the DRC phase)."""
    gap = P.via_drill_mm + max(P.clearance_mm, 0.25)
    parts = [("R1", "PAD1", 6.0, 5.0), ("R2", "PAD1", 10.0, 5.0), ("R3", "SMD1", 7.5, 9.0)]

    def route(in2_right: float):
        ir = board_ir(tmp_path, lib, parts, {"N": [("R1", "1"), ("R2", "1"), ("R3", "1")]}, (14.0, 10.0))
        kos = [keepout("KB", [0.0, 0.0, 14.0, 10.0], forbids=["tracks"], layers=["B.Cu"]),
               keepout("KF1", [5.9, 4.9, 0.2, 0.2], forbids=["tracks"], layers=["F.Cu"]),
               keepout("K31", [5.9, 4.9, 0.2, 0.2], forbids=["tracks"], layers=["In3.Cu"]),
               keepout("KF2", [9.9, 4.9, 0.2, 0.2], forbids=["tracks"], layers=["F.Cu"]),
               *_outside("K2", "In2.Cu", 3.0, 3.5, in2_right, 6.5), *_outside("K3", "In3.Cu", 6.0, 3.5, 13.0, 6.5)]
        return route_board(ir, lib, keepouts=kos, stack=_n_stack(ir))

    # an In2.Cu window to x = 9.1 leaves a blind-via site clear of the buried via's drill: both are drilled, apart (0.25 mm before)
    r = route(9.1)
    assert r.unrouted == {} and sorted(v.kind for v in r.vias) == ["blind", "buried"] and _holes_apart(r, gap), [(v.kind, v.x_mm, v.y_mm) for v in r.vias]
    entry = r.vias[0].provenance.derived_from[-1]
    assert f";net_hole_gap={P.via_drill_mm}+0.25;" in entry and entry.endswith(STACK_SEARCH.derived_from_entry())
    assert (f"a net's own vias {gap:g} mm apart centre to centre (via drill + the hole-to-hole gap: the larger of the router's clearance and "
            "the 0.25 mm kicad-cli DRC checks") in r.vias[0].provenance.note
    # the probe's own window (to x = 8.6) has six via sites, all within 0.56 mm of each other: no two drills fit, and none overlap
    assert _holes_apart(route(8.6), gap)


def test_a_branch_that_drills_twice_within_the_gap_names_the_cells_to_search_again_without(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0)], {"A": [("R1", "1"), ("R2", "1")]}, (12.0, 10.0))
    board = _Board(ir, lib, P, stack=_n_stack(ir))
    n, k0 = board.n, board.nx * 20 + 20
    disc: set[int] = set()
    board.halo_cells(board.net_hole_halo, k0, 0, disc)
    assert len(disc) == 21  # cells closer than 0.4 + 0.25 mm on the 0.25 mm grid: the centre, 4 at 0.25, 4 at 0.354, 4 at 0.5, 8 at 0.559
    # F.Cu -> In2.Cu at k0, two cells east on In2.Cu, In2.Cu -> In3.Cu there (0.5 mm): the first via's disc without its own cell
    path = [k0, 2 * n + k0, 2 * n + k0 + 1, 2 * n + k0 + 2, 3 * n + k0 + 2]
    assert maze._own_hole_clash(board, path) == disc - {k0}
    # three cells east (0.75 mm) is clear; a second via on the first one's cell is that cell alone
    assert maze._own_hole_clash(board, [k0, 2 * n + k0, 2 * n + k0 + 1, 2 * n + k0 + 2, 2 * n + k0 + 3, 3 * n + k0 + 3]) is None
    assert maze._own_hole_clash(board, [k0, 2 * n + k0, 2 * n + k0 + 1, 2 * n + k0, 3 * n + k0]) == {k0}


def test_two_n_layer_runs_give_the_same_bytes_and_no_two_vias_share_a_cell(tmp_path: Path):
    def run(tmp: Path):
        lib = fixture_library(tmp / "kicad")
        ir = _grid_board(tmp, lib)
        return route_board(ir, lib, plane_nets={"GND": [("In1.Cu", GRID_AREA)]}, stack=_n_stack(ir))

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert canonical(a) == canonical(b)
    assert a.unrouted == {} and a.stats["legal"] is True and _no_shared_via_cell(a)
    assert {t.layer for t in a.tracks} >= {"F.Cu", "In2.Cu"}  # the vertical nets cross on In2.Cu, the vertical layer
    entry = a.tracks[0].provenance.derived_from[-1]
    assert entry.startswith("stack:6-layer 1.6 mm FR-4;signal=F.Cu+In2.Cu+In3.Cu+B.Cu;") and entry.endswith(STACK_SEARCH.derived_from_entry())
    note = a.tracks[0].provenance.note
    assert note.startswith("grid maze route on F.Cu/In2.Cu/In3.Cu/B.Cu (") and "VPR's negotiation schedule (MIT, Copyright 2012 VTR Developers" in note


def test_compiled_typed_signal_vias_pass_layout_validation(tmp_path: Path, lib: KicadLibrary):
    """The buried-via board with a GND net (two-pad through-hole parts, as the Test:VR1 symbol has two pins, so the board compiles; GND
    on both planes, reached by its pads' barrels): the compiled board carries the buried via's token and span and the inner-layer
    tracks, and pcb.routing's clearance check passes with the router's clearance as the limit."""
    parts = [("R1", "PAD2", 4.0, 5.0), ("R2", "PAD2", 16.0, 5.0)]
    ir = board_ir(tmp_path, lib, parts, {"GND": [("R1", "2"), ("R2", "1")], "N": [("R1", "1"), ("R2", "2")]}, (20.0, 10.0))
    ir.nets[0].kind = NetKind.GROUND
    st = _n_stack(ir)
    area = [(0.5, 0.5), (19.5, 0.5), (19.5, 9.5), (0.5, 9.5)]
    ko = [keepout("KFB", [0.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["F.Cu", "B.Cu"]),
          keepout("K2", [12.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["In2.Cu"]),
          keepout("K3", [0.0, 0.0, 8.0, 10.0], forbids=["tracks"], layers=["In3.Cu"])]
    r = route_board(ir, lib, keepouts=ko, plane_nets={"GND": [("In1.Cu", area), ("In4.Cu", area)]}, stack=st)
    assert r.unrouted == {} and [(v.net, v.kind, v.layers) for v in r.vias] == [("N", "buried", ("In2.Cu", "In3.Cu"))]
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    ir.pcb.zones = plane_zones(ir.pcb.stackup, ir.pcb.outline, 0.5)
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    tree = sexpr.parse_file(Path(art.path))
    tokens = [(v[1] if isinstance(v[1], str) else None, tuple(str(x) for x in sexpr.args(sexpr.find(v, "layers")))) for v in sexpr.find_all(tree, "via")]
    assert tokens == [("buried", ("In2.Cu", "In3.Cu"))]
    assert {t.layer for t in r.tracks if t.net == "N"} == {"In2.Cu", "In3.Cu"}
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(P.clearance_mm, note="the router's clearance as the limit (a limit is a limit)"))
    got = _run("pcb.routing", ir, tmp_path, lib)
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message
    rows = {row["net"]: row for row in got[CONNECTIVITY_CHECK].details["nets"]}
    assert rows["N"]["status"] == "PASS"


#: two facing rows of five 0.7 x 0.25 mm pads at a 0.5 mm pitch (the inner sides of a fine-pitch package 3.2 mm across): the west row
#: (pads 1-5) leaves west, its inner edge faces the east row (pads 6-10, net-less here)
DUAL5 = "\n  ".join(
    [f'(pad "{i + 1}" smd rect (at -1.6 {y}) (size 0.7 0.25) (layers "F.Cu" "F.Mask" "F.Paste"))' for i, y in enumerate((-1.0, -0.5, 0, 0.5, 1.0))]
    + [f'(pad "{i + 6}" smd rect (at 1.6 {y}) (size 0.7 0.25) (layers "F.Cu" "F.Mask" "F.Paste"))' for i, y in enumerate((1.0, 0.5, 0, -0.5, -1.0))])
#: the outline of the 20 x 18 mm fine-pitch boards inset by 0.5 mm (a plane zone)
FINE_AREA = [(0.5, 0.5), (19.5, 0.5), (19.5, 17.5), (0.5, 17.5)]


def _dual_board(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary]:
    """``DUAL5`` at (12, 8): pads 1, 2, 4, 5 on their own nets to through-hole pads at x = 3, pad 3 on GND with a through-hole pad."""
    from tests.test_routing_fine_pitch import fine_library

    lib = fine_library(tmp_path / "kicad")
    (tmp_path / "kicad" / "footprints" / "Test.pretty" / "DUAL5.kicad_mod").write_text(
        f'(footprint "DUAL5" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n  {DUAL5})\n', encoding="utf-8")
    parts = [("U1", "DUAL5", 12.0, 8.0)] + [(f"R{i}", "PAD1", 3.0, 2.0 + 3.0 * i) for i in (1, 2, 4, 5)] + [("R3", "PAD1", 16.0, 15.0)]
    nets = {f"S{i}": [("U1", str(i)), (f"R{i}", "1")] for i in (1, 2, 4, 5)}
    nets["GND"] = [("U1", "3"), ("R3", "1")]
    ir = board_ir(tmp_path, lib, parts, nets, (20.0, 18.0))
    ir.net("GND").kind = NetKind.GROUND
    return ir, lib


def test_a_plane_pad_of_a_via_fanout_footprint_looks_under_its_package_first_on_an_n_layer_board(tmp_path: Path):
    """routing.maze 0.9, N signal layers (module docstring: plane pads under the package): a waiting plane pad of a footprint the via
    fan-out escapes in alternation (0.8) looks for its via out of its inner edge first - on the STM32F407's LQFP-100 the GND pads'
    vias in front of the rows closed their neighbours' lanes; every other board keeps the outer edge alone."""
    from tests.test_routing_fine_pitch import FINE

    ir, lib = _dual_board(tmp_path)
    st = _n_stack(ir, 6)
    board = _Board(ir, lib, FINE, plane={"GND": FINE.track_width_mm}, plane_areas={"GND": [FINE_AREA]},
                   plane_layer_areas={"GND": [("In1.Cu", FINE_AREA)]}, stack=st)
    assert board.search_n and not board.rules08  # 0.6 escapes the row: the via fan-out never runs, the rule is not in force
    t = next(t for t in board.terminals["GND"] if t.label == "U1.3")
    idx, pad = board.net_index["GND"], t.pad
    [row] = [row for row in board.escapes if row["pad"] == "U1.3"]
    assert row["kind"] == "via" and row["cell"][0] < pad.cx  # in front of the row, as before
    assert board._plane_inward(pad) is False
    board.rules08, board.dogbone_refs = True, ["U1"]
    r1 = next(t.pad for t in board.terminals["S1"] if t.pad.ref == "R1")
    assert board._plane_inward(pad) is True and board._plane_inward(r1) is False  # R1: not a fanned-out footprint
    kw = {"via": True, "w_cap": FINE.track_width_mm, "areas": [FINE_AREA], "placed": []}
    inner = board._find_escape(t, idx, "fenced", inward=True, **kw)
    outer = board._find_escape(t, idx, "fenced", **kw)
    assert isinstance(inner, maze._Escape) and isinstance(outer, maze._Escape)
    assert pad.cx + pad.hw < board.xy(inner.cell)[0] < 12.0 < 12.0 + 1.6 - 0.35  # under the package, between the two rows
    assert board.xy(outer.cell)[0] < pad.cx - pad.hw
    # the 4-layer rung (signal layers F.Cu / B.Cu, no N-layer search) and a board without a stack never look inward
    ir4, lib4 = _dual_board(tmp_path / "four")
    stacked(ir4, 4)  # GND on In1.Cu and In2.Cu
    four = _Board(ir4, lib4, FINE, plane={"GND": FINE.track_width_mm}, plane_areas={"GND": [FINE_AREA]},
                  plane_layer_areas={"GND": [("In1.Cu", FINE_AREA)]}, stack=_router_stack(ir4))
    plain = _Board(*_dual_board(tmp_path / "plain"), FINE)
    for b in (four, plain):
        b.rules08, b.dogbone_refs = True, ["U1"]
        assert b.search_n is False and b._plane_inward(next(g for g in b.pads if g.ref == "U1" and g.number == "3")) is False


def test_an_n_layer_boards_via_sites_are_measured_exactly_and_a_two_layer_boards_keep_0_2s(tmp_path: Path, lib: KicadLibrary):
    """routing.maze 0.9, N signal layers (module docstring: exact via sites): R1 (net A, a 1 mm SMD pad at x 3.5 .. 4.5) and R3 (net B,
    x 5.55 .. 6.55). A via (0.8 mm, clearance 0.25 mm) needs its centre 0.65 mm from a foreign pad; 0.2's check keeps A's owner cells
    (0.575 mm round the pad) out of its 0.85 mm disc, so it gives A alone every site up to ~1.4 mm from the pad. On the 6-layer stack
    a site 0.5 mm from both pads is blocked, 0.5 mm from A's alone A's, 0.75 mm from A's (or 0.7 mm from B's) free for every net; on
    the same board without a stack 0.2's values stay."""
    parts = [("R1", "SMD1", 4.0, 5.0), ("R2", "SMD1", 4.0, 8.5), ("R3", "SMD1", 6.05, 5.0), ("R4", "SMD1", 9.0, 8.5)]
    nets = {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}
    old = _Board(board_ir(tmp_path / "old", lib, parts, nets, (12.0, 10.0)), lib, P)
    ir = board_ir(tmp_path, lib, parts, nets, (12.0, 10.0))
    board = _Board(ir, lib, P, stack=_n_stack(ir))
    a, b = board.net_index["A"], board.net_index["B"]

    def at(bd: _Board, x: float, t: int = 0) -> int:
        i, j = bd._nearest_cell(x, 5.0)
        return bd.via_static_at(j * bd.nx + i, t)

    assert at(board, 5.0) == maze.BLOCKED  # 0.5 mm from A's pad and 0.55 mm from B's: within 0.65 mm of both
    assert at(board, 3.0) == a and at(old, 3.0) == a  # 0.5 mm west of A's pad: A's own (the via disc clears the box by 0.1 mm)
    assert at(board, 2.75) == _FREE and at(old, 2.75) == a  # 0.75 mm from A's pad: legal for any net, 0.2 kept it A's
    assert all(at(board, 2.75, t) == _FREE for t in range(board.T))  # every signal type: the pad is on F.Cu, every type spans it
    assert at(board, 7.25) == _FREE and at(old, 7.25) == b  # 0.7 mm east of B's pad
    entry = route_board(ir, lib, stack=_n_stack(ir)).tracks[0].provenance.derived_from[-1]
    assert f";net_hole_gap={P.via_drill_mm}+0.25;via_sites=exact;" in entry


def _qfp20_board(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary]:
    """A synthetic QFP-20 (five 1.2 x 0.3 mm pads a side at a 0.5 mm pitch, pin 1 at the west side's top, counter-clockwise) at
    (12, 12) on a 24 x 24 mm board, every pin wired to its own through-hole pad 8 mm out, pin 3 (west side, middle) on GND."""
    from tests.test_routing_fine_pitch import fine_library

    lib = fine_library(tmp_path / "kicad")
    pads, nets, parts = [], {}, [("U1", "QFP20", 12.0, 12.0)]
    for i in range(1, 21):
        side, q = divmod(i - 1, 5)
        t = -1.0 + 0.5 * q
        x, y, w, h = ((-2.2, t, 1.2, 0.3), (t, 2.2, 0.3, 1.2), (2.2, -t, 1.2, 0.3), (-t, -2.2, 0.3, 1.2))[side]
        pads.append(f'(pad "{i}" smd rect (at {x:g} {y:g}) (size {w} {h}) (layers "F.Cu" "F.Mask" "F.Paste"))')
        rx, ry = ((4.0, 12.0 + 3 * t), (12.0 + 3 * t, 20.0), (20.0, 12.0 - 3 * t), (12.0 - 3 * t, 4.0))[side]
        parts.append((f"R{i}", "PAD1", rx, ry))
        nets["GND" if i == 3 else f"P{i}"] = [("U1", str(i)), (f"R{i}", "1")]
    (tmp_path / "kicad" / "footprints" / "Test.pretty" / "QFP20.kicad_mod").write_text(
        '(footprint "QFP20" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n  ' + "\n  ".join(pads) + ")\n", encoding="utf-8")
    ir = board_ir(tmp_path, lib, parts, nets, (24.0, 24.0))
    ir.net("GND").kind = NetKind.GROUND
    return ir, lib


def test_route_board_puts_a_via_fanout_footprints_plane_via_under_the_package_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End to end on the synthetic QFP-20 (0.6's trials cut to the pad-order one, as the LQFP-48 test of tests/test_routing_fine_pitch.py
    does, so 0.8 runs): on the 6-layer stack U1.3's GND via lies under the package, ``stats["stack"]["search"]["planes_under_package"]``
    and the stack entry name it; with the rule switched off it lies in front of the row."""
    from tests.test_routing_fine_pitch import FINE

    monkeypatch.setattr(maze, "_ESCAPE_TRIALS", (("natural", "near", "plane"),))
    area = [(0.5, 0.5), (23.5, 0.5), (23.5, 23.5), (0.5, 23.5)]
    ir, lib = _qfp20_board(tmp_path)
    r = route_board(ir, lib, FINE, plane_nets={"GND": [("In1.Cu", area)]}, stack=_n_stack(ir, 6))
    assert r.unrouted == {} and r.stats["legal"] is True and r.stats["via_fanout"]["footprints"] == ["U1"]
    assert r.stats["stack"]["search"]["planes_under_package"] == ["U1.3"] and "end_game" not in r.stats["stack"]["search"]
    [via] = [v for v in r.vias if v.net == "GND"]
    assert 12.0 - 2.2 + 0.6 < via.x_mm < 12.0  # beyond the pad's inner end (x = 10.4), inside the ring
    entry = r.tracks[0].provenance.derived_from[-1]
    assert entry.endswith(STACK_SEARCH.derived_from_entry() + ";planes_under_package=U1.3")
    assert "the waiting plane pads of U1 looked for their via under the package first (U1.3 took one there)" in r.tracks[0].provenance.note
    got = _routing_checks(ir, r, lib, tmp_path)
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message
    monkeypatch.setattr(maze._Board, "_plane_inward", lambda self, geom: False)
    ir2, lib2 = _qfp20_board(tmp_path / "front")
    front = route_board(ir2, lib2, FINE, plane_nets={"GND": [("In1.Cu", area)]}, stack=_n_stack(ir2, 6))
    [via] = [v for v in front.vias if v.net == "GND"]
    assert via.x_mm < 12.0 - 2.2 - 0.6 and front.stats["stack"]["search"]["planes_under_package"] == []
    assert front.tracks[0].provenance.derived_from[-1].endswith(";planes_under_package=none")


def _corridor_board(tmp_path: Path, lib: KicadLibrary) -> tuple[CircuitIR, list]:
    """A (R1 -> R2) and B (R3 -> R4) across a wall at x = 9.5 .. 10.5 on every signal layer of the 6-layer stack, open on F.Cu in a
    corridor at y = 4.4 .. 5.6 (one track wide) and, for B alone, in a gap at y = 7.6 .. 8.8 - B's longer way."""
    parts = [("R1", "PAD1", 2.0, 5.0), ("R2", "PAD1", 18.0, 5.0), ("R3", "PAD1", 2.0, 3.0), ("R4", "PAD1", 18.0, 3.0)]
    ir = board_ir(tmp_path, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (20.0, 10.0))
    kos = [keepout("KF1", [9.5, 0.0, 1.0, 4.4], forbids=["tracks"], layers=["F.Cu"]),
           keepout("KF2", [9.5, 5.6, 1.0, 2.0], forbids=["tracks"], layers=["F.Cu"]),
           keepout("KFB", [9.5, 7.6, 1.0, 1.2], forbids=["tracks"], layers=["F.Cu"], nets=["B"]),
           keepout("KF3", [9.5, 8.8, 1.0, 1.2], forbids=["tracks"], layers=["F.Cu"]),
           keepout("KI", [9.5, 0.0, 1.0, 10.0], forbids=["tracks"], layers=["In2.Cu", "In3.Cu", "B.Cu"])]
    return ir, kos


def test_the_end_game_rips_up_the_net_in_the_way_and_routes_it_again_around(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """routing.maze 0.9, N signal layers (module docstring: the end game): one free iteration puts A and B in the one-track corridor;
    against B's copper A has no route (B's own gap is closed to A), so the end game routes A at the negotiated costs, rips up B (the
    one net A's route crosses) and routes B again strictly - through its own gap. The board ends legal with both nets; without the
    end game (the drop phase alone, as before) A was dropped and never recovered."""
    ir, kos = _corridor_board(tmp_path, lib)
    st = _n_stack(ir, directions=False)
    r = route_board(ir, lib, RoutingParams(max_iterations=1), keepouts=kos, stack=st)
    assert r.stats["history"][-1]["conflicting"] == ["A", "B"]  # the free pass
    assert r.unrouted == {} and r.stats["legal"] is True and r.stats["dropped"] == [] and r.stats["recovered"] == []
    moved = [{"net": "A", "how": "made room", "ripped_up": ["B"]}]
    assert r.stats["stack"]["search"]["end_game"] == {"rip_up_limit": maze.END_GAME_RIP_UP, "moved": moved}
    def crossing(net: str) -> set[float]:  # the y of the net's F.Cu segments that cross x = 10 (the wall's middle)
        return {t.start[1] for t in r.tracks if t.net == net and t.layer == "F.Cu" and min(t.start[0], t.end[0]) <= 10.0 <= max(t.start[0], t.end[0])}

    assert len(crossing("B")) == 1 and 7.6 < crossing("B").pop() < 8.8  # B through its own gap now
    assert len(crossing("A")) == 1 and 4.4 < crossing("A").pop() < 5.6  # A through the corridor
    a = next(t for t in r.tracks if t.net == "A")
    b = next(t for t in r.tracks if t.net == "B")
    assert "the nets it crossed were ripped up and routed again around it" in a.provenance.note
    assert "routed again against the legal copper as an obstacle in the end game" in b.provenance.note
    assert a.provenance.derived_from[-1].endswith(f";end_game=rip_up<={maze.END_GAME_RIP_UP}:A")
    got = _routing_checks(ir, r, lib, tmp_path)
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message
    # the drop phase alone (the end game switched off): A is dropped and its strict recovery finds no route
    monkeypatch.setattr(maze, "_end_game", lambda neg, order, routes, route_one: [])
    old = route_board(ir, lib, RoutingParams(max_iterations=1), keepouts=kos, stack=_n_stack(ir, directions=False))
    assert old.stats["dropped"] == ["A"] and old.stats["recovered"] == [] and list(old.unrouted) == ["A"]


def _path_cost(neg: _Negotiation, path: list[int]) -> float:
    """What ``path`` costs at ``neg``'s costs, as :func:`maze._search` counts it (a step its node cost plus a bend, a via its via plane's
    cost plus the landing cell's node cost less ``base``; no bend after a via)."""
    n, nx = neg.board.n, neg.board.nx
    base, bend = neg.p.base_cost, neg.p.bend_cost
    code = {1: 0, nx: 1, -1: 2, -nx: 3}
    cost, d = 0.0, None
    for a, b in zip(path, path[1:]):
        if a // n != b // n:
            cost += neg.vcost[a % n] + neg.ncost[b] - base
            d = None
            continue
        nd = code[b - a]
        cost += neg.ncost[b] + (bend if d is not None and nd != d else 0.0)
        d = nd
    return cost


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_barrel_search_finds_searchs_cheapest_cost_on_every_2_layer_fixture(tmp_path: Path, name: str):
    """Cost oracle: a :class:`_NegotiationN` with barrels ``T * n`` and the through-only stack (no preferred direction), A* weight 1, on
    the same uneven history costs as 0.2's negotiation, finds a path as cheap as :func:`maze._search` between the first two terminals
    of every net (within 1e-9)."""
    ir, lib, params = FIXTURES[name](tmp_path)
    p, _raised = effective_params(ir, params)
    old_board = _Board(ir, lib, p)
    board = _Board(ir, lib, p, stack=RouterStack("through-only", LAYERS, LAYERS, (), (THROUGH,)))
    assert board.search_n is False  # the 4-layer rung's case: P4 routes it with 0.2's search
    board.search_n, board.barrels = True, board.T * board.n
    # the oracle compares the two searches, not the via models: the barrel search reads 0.2's via sites here (an N-layer board's own
    # are measured exactly, P8 - module docstring: exact via sites)
    board.via_static_n[0][:] = [old_board.via_static_at(k) for k in range(board.n)]
    old, new = _Negotiation(old_board, p), maze._NegotiationN(board, p)
    n = board.n
    hist = [(c * 7919 % 13) / 4.0 for c in range(2 * n)]
    vhist = [(k * 104729 % 7) / 2.0 for k in range(n)]
    old.hist, old.vhist = list(hist), list(vhist)
    new.hist, new.vhist = array("d", hist), array("d", vhist)
    old.recompute()
    new.recompute()
    assert list(new.ncost) == old.ncost and list(new.vcost) == old.vcost
    full = (0, 0, board.nx - 1, board.ny - 1)
    compared = 0
    for net in ir.nets:
        terms = board.terminals[net.name]
        if len(terms) < 2 or any(t.refusal is not None for t in terms[:2]):
            continue
        idx = board.net_index[net.name]
        use = [board.usable_layers(t, idx) for t in terms[:2]]
        if not use[0] or not use[1]:
            continue
        assert use == [old_board.usable_layers(t, idx) for t in old_board.terminals[net.name][:2]]
        seeds = [layer * n + terms[0].cell for layer in use[0]]
        targets = {layer * n + terms[1].cell for layer in use[1]}
        tij = divmod(terms[1].cell, board.nx)
        a = maze._search(old, idx, seeds, targets, tij, full, False)
        b = maze._search_n(new, idx, seeds, targets, tij, full, False, astar_fac=1.0)
        assert (a is None) == (b is None), (name, net.name)
        if a is not None:
            assert abs(_path_cost(old, a) - _path_cost(old, b[0])) <= 1e-9, (name, net.name, _path_cost(old, a), _path_cost(old, b[0]))
            assert b[1] == [0] * sum(1 for x, y in zip(b[0], b[0][1:]) if x // n != y // n)
            compared += 1
    assert compared, name
