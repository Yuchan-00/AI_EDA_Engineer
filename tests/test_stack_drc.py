"""Stack boards routed against the rules KiCad's DRC checks (routing.maze 0.9, stack mode): KiCad's minimums and the plane web.

The STM32F407 example routed 91 / 91 nets on 6 layers but kicad-cli 10.0.6 DRC found 22 errors and 2 unconnected items (2026-11-13):
tracks and vias 0.475 / 0.3 mm from the edge against KiCad's 0.5 mm, escape stubs 0.15 mm wide against KiCad's 0.2 mm (the compiled
project file states no rule, so KiCad's own minimums apply), and two GND vias between fan-out vias under the LQFP-100 that the In1.Cu
fill no longer reached (``via_dangling``, ``unconnected_items``). These tests pin, on synthetic libraries:

* :func:`ai_eda.compilers.project.checked_rules` - the minimums kicad-cli checks beside the compiled project file;
* :func:`ai_eda.tools.routing.maze.kicad_floor` and a stack board's raised parameters, record and copper; a board without a stack keeps
  its parameters (its copper is pinned by the golden digests);
* :func:`ai_eda.tools.routing.plane_web.plane_web` - the fill model on the measured geometry and on rings, bridges, spans and pads;
  where kicad-cli exists, a canary compiles that geometry and compares DRC's ``via_dangling`` with the check;
* the router's plane web pass: nothing changes where nothing is cut off; a cut-off plane via's cutters are routed again away from it,
  closed via sites included, and the record says so; a reroute that fails is undone; static copper is left and named.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler, ProjectFileCompiler
from ai_eda.compilers.pcb import ZONE_CLEARANCE_MM, ZONE_MIN_THICKNESS_MM
from ai_eda.compilers.project import (
    KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM,
    KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM,
    KICAD_DEFAULT_MIN_THROUGH_HOLE_MM,
    KICAD_DEFAULT_MIN_TRACK_WIDTH_MM,
    KICAD_DEFAULT_MIN_VIA_DIAMETER_MM,
    KICAD_DEFAULT_NETCLASS_CLEARANCE_MM,
    checked_rules,
)
from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, ManufacturingConstraints, NetKind, Provenance, ProvenanceKind, Track, Traced, Via, Zone, assumption
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import maze
from ai_eda.tools.routing.maze import RoutingParams, kicad_floor, route_board
from ai_eda.tools.routing.plane_web import WEB_STRICTER, PlaneItem, plane_web
from ai_eda.validation.layout import CLEARANCE_CHECK
from tests.test_routing import _routing_checks, board_ir, fixture_library
from tests.test_routing_layers import _gnd_3v3_board, _n_stack, _qfp20_board, _router_stack, PLANES

kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not kicad.available(), reason="kicad-cli not installed")
P = RoutingParams()
FILL = {"clearance": ZONE_CLEARANCE_MM, "min_thickness": ZONE_MIN_THICKNESS_MM}
HAND = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand for the test")


def test_corner_reservations_are_cutouts_of_one_zone_per_layer(tmp_path, lib):
    from ai_eda.agents.pcb import _clip_planes
    from ai_eda.ir import Keepout
    from ai_eda.tools.kicad import sexpr
    from ai_eda.validation.layout import KEEPOUT_CHECK
    from tests.test_rf_floorplan import _run
    ir = board_ir(tmp_path, lib, [("R1", "PAD2", 10, 10)], {"GND": [("R1", "1")]}, (20, 20))
    from tests.test_si_checks import stacked
    stacked(ir, 4)
    outer = [(0.5, 0.5), (19.5, 0.5), (19.5, 19.5), (0.5, 19.5)]
    zones = [Zone(net="GND", layer=layer, polygon=outer, provenance=HAND) for layer in ("In1.Cu", "In2.Cu")]
    ir.pcb.keepouts = [Keepout(id=f"H{i}", layers=["*.Cu"], rect=Traced(value=[x, y, 3, 3], unit="mm", provenance=HAND),
                      forbids=["zones"], reason="mounting reservation", provenance=HAND)
                      for i, (x, y) in enumerate(((1, 1), (16, 1), (1, 16), (16, 16)))]
    ir.pcb.zones = _clip_planes(zones, ir.pcb.keepouts, [])
    assert len(ir.pcb.zones) == 2
    assert all(z.polygon == outer and len(z.holes) == 4 for z in ir.pcb.zones)
    node = PCBCompiler().build(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    copper = [z for z in sexpr.find_all(node, "zone") if sexpr.find(z, "keepout") is None]
    assert len(copper) == 2 and all(len(sexpr.find_all(z, "polygon")) == 5 for z in copper)
    assert _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK].status.value == "PASS"


def test_zone_without_keepouts_keeps_its_bytes():
    from ai_eda.agents.pcb import _clip_planes
    from ai_eda.compilers import ids
    from ai_eda.tools.kicad import sexpr
    from ai_eda.tools.kicad.sexpr import S, Q
    z = Zone(net="GND", layer="In1.Cu", polygon=[(1, 1), (9, 1), (9, 9), (1, 9)], provenance=HAND)
    assert "holes" not in z.model_dump()
    old = S("zone", S("net", Q("GND")), S("layer", Q("In1.Cu")), S("uuid", Q(ids.net_item_uuid("p", "zone", 0))),
            S("hatch", "edge", 0.5), S("connect_pads", "yes", S("clearance", 0.5)), S("min_thickness", 0.25),
            S("fill", "yes", S("thermal_gap", 0.5), S("thermal_bridge_width", 0.5)),
            S("polygon", S("pts", *[S("xy", float(x), float(y)) for x, y in z.polygon])))
    assert sexpr.dumps(PCBCompiler._zone("p", 0, _clip_planes([z], [], [])[0], {"GND"}, ["In1.Cu"])) == sexpr.dumps(old)


def test_plane_web_excludes_a_cutout_without_splitting_the_fill():
    outer = [(0, 0), (12, 0), (12, 10), (0, 10)]
    hole = [(4, 2), (8, 2), (8, 8), (4, 8)]
    def item(x, y):
        return PlaneItem("via", str((x, y)), "GND", x, y, 0.3, (x-.3, y-.3, x+.3, y+.3), frozenset({"In1.Cu"}))
    items = [item(2, 5), item(10, 5), item(6, 5)]
    planes = {"GND": [("In1.Cu", outer)]}
    assert plane_web(planes, items, **FILL).cut == {"GND": []}
    assert plane_web(planes, items, holes={"GND": [[hole]]}, **FILL).cut == {"GND": [2]}


def test_plane_contours_merge_overlapping_reservations_and_edge_notches():
    from ai_eda.tools.keepout import rect_contours, polygon_area
    parts = rect_contours((0, 0, 20, 20), [(2, 2, 6, 6), (4, 4, 8, 8), (0, 10, 3, 15)])
    [(outer, holes)] = parts
    assert len(holes) == 1
    assert abs(polygon_area(outer)) - sum(abs(polygon_area(h)) for h in holes) == 357
    assert rect_contours((0, 0, 20, 20), [(0, 0, 20, 20)]) == []


def test_plane_via_sites_respect_zone_cutouts_without_a_via_ban(tmp_path, lib):
    from ai_eda.tools.keepout import point_area_distance
    ir = _gnd_3v3_board(tmp_path, lib)
    before = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    via = next(v for v in before.vias if v.net == "GND")
    x, y = via.x_mm, via.y_mm
    hole = [(x-.4, y-.4), (x+.4, y-.4), (x+.4, y+.4), (x-.4, y+.4)]
    layer, outer = PLANES["GND"][0]
    ir.pcb.zones = [Zone(net="GND", layer=layer, polygon=outer, holes=[hole], provenance=HAND)]
    after = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    assert not after.unrouted
    assert all(point_area_distance((v.x_mm, v.y_mm), hole) >= v.diameter_mm / 2 - 1e-6
               for v in after.vias if v.net == "GND")


@needs_kicad
def test_kicad_reads_zone_cutouts_as_holes(tmp_path, lib):
    ir = board_ir(tmp_path, lib, [("R1", "PAD2", 10, 10)], {"GND": [("R1", "1"), ("R1", "2")]}, (20, 20))
    _n_stack(ir, 6)
    outer = [(0.5, .5), (19.5, .5), (19.5, 19.5), (.5, 19.5)]
    hole = [(2, 2), (6, 2), (6, 6), (2, 6)]
    ir.pcb.zones = [Zone(net="GND", layer="In1.Cu", polygon=outer, holes=[hole], provenance=HAND)]
    ir.pcb.vias = [Via(net="GND", x_mm=4, y_mm=4, drill_mm=.3, diameter_mm=.6, provenance=HAND)]
    artifact = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    report = tmp_path / "out" / "cutouts-drc.json"
    kicad.run_drc(Path(artifact.path), report)
    measured = json.loads(report.read_text(encoding="utf-8"))
    assert not any(v["type"] == "zones_intersect" for v in measured["violations"])
    assert _dangling_at(measured, [(4, 4)]) == [True]


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


# --------------------------------------------------------------------------- the minimums kicad-cli checks


def _fab(**limits: Traced) -> ManufacturingConstraints:
    return ManufacturingConstraints(**limits)


def _authoritative(value: float) -> Traced:
    return Traced(value=value, unit="mm", provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="the fab's page, for the test"))


def test_kicad_checks_its_own_minimums_where_the_project_file_states_none(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0)], {}, (10.0, 6.0))
    assert (KICAD_DEFAULT_MIN_TRACK_WIDTH_MM, KICAD_DEFAULT_NETCLASS_CLEARANCE_MM, KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM) == (0.2, 0.2, 0.5)
    assert (KICAD_DEFAULT_MIN_VIA_DIAMETER_MM, KICAD_DEFAULT_MIN_THROUGH_HOLE_MM, KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM) == (0.5, 0.3, 0.25)
    vias = {"min_via_diameter": 0.5, "min_through_hole_diameter": 0.3, "min_hole_to_hole": 0.25}
    assert checked_rules(ir) == {"min_track_width": 0.2, "min_clearance": 0.2, "min_copper_edge_clearance": 0.5, **vias}
    # an authoritative fab limit becomes the project file's rule, and that is what kicad-cli checks; a stricter clearance goes into
    # the Default netclass too, a looser one leaves the netclass's 0.2 mm in force
    ir.pcb.manufacturing = _fab(min_track_width_mm=_authoritative(0.15), min_clearance_mm=_authoritative(0.3))
    assert checked_rules(ir) == {"min_track_width": 0.15, "min_clearance": 0.3, "min_copper_edge_clearance": 0.5, **vias}
    # a written via rule replaces KiCad's own (measured: a 0.45 / 0.2 mm via passed beside min_via_diameter 0.4 and
    # min_through_hole_diameter 0.2); nothing writes a hole-to-hole rule
    ir.pcb.manufacturing = _fab(min_via_diameter_mm=_authoritative(0.4), min_via_drill_mm=_authoritative(0.2))
    assert checked_rules(ir) == {"min_track_width": 0.2, "min_clearance": 0.2, "min_copper_edge_clearance": 0.5, "min_via_diameter": 0.4,
                                 "min_through_hole_diameter": 0.2, "min_hole_to_hole": 0.25}
    ir.pcb.manufacturing = _fab(min_clearance_mm=_authoritative(0.15))
    assert checked_rules(ir)["min_clearance"] == 0.2
    # an assumed limit is never written, so KiCad's own minimum stays the checked one (the router still honours it: effective_params)
    ir.pcb.manufacturing = _fab(min_track_width_mm=assumption(0.1, note="a guess"))
    assert checked_rules(ir)["min_track_width"] == 0.2
    assert ProjectFileCompiler().build(ir)["board"]["design_settings"]["rules"] == {}


def _wall_board(tmp_path: Path, lib: KicadLibrary, bottom: float, layers: int | None):
    """N between two through-hole pads, an F.Cu and a B.Cu wall between them whose lower end is ``bottom`` mm from the top edge: N must
    pass between the walls and the edge. ``layers=4`` puts the board on the 4-layer stack."""
    from tests.test_si_checks import stacked

    parts = [("R1", "PAD1", 1.0, 3.0), ("R2", "PAD1", 7.0, 3.0), ("W1", "WALL", 4.0, bottom + 10.0), ("W2", "WALL", 4.0, bottom + 10.0)]
    ir = board_ir(tmp_path, lib, parts, {"N": [("R1", "1"), ("R2", "1")]}, (8.0, 6.0), sides={"W2": BoardSide.BOTTOM})
    if layers is None:
        return route_board(ir, lib)
    stacked(ir, layers)
    return route_board(ir, lib, plane_nets={}, stack=_router_stack(ir))


def _edge_gap(r) -> float:
    """The least distance of the routed copper from the 8 x 6 mm outline."""
    gaps = [min(x, 8.0 - x, y, 6.0 - y) - t.width_mm / 2.0 for t in r.tracks for x, y in (t.start, t.end)]
    return min(gaps + [min(v.x_mm, 8.0 - v.x_mm, v.y_mm, 6.0 - v.y_mm) - v.diameter_mm / 2.0 for v in r.vias])


def test_a_stack_board_is_routed_at_kicads_minimums_and_a_board_without_one_keeps_its_parameters(tmp_path: Path, lib: KicadLibrary):
    """The gap between the walls and the edge leaves N one track at y = 0.5 mm: at the router's 0.3 mm edge clearance the 2-layer board
    takes it - copper 0.3 mm from the edge, which kicad-cli's 0.5 mm flags (as on the STM32's 6-layer board, 0.475 mm); the same board
    on the 4-layer stack keeps KiCad's 0.5 mm and finds no way through. With a wider gap both route and the stack board's copper keeps
    0.5 mm. The record says why."""
    plain = _wall_board(tmp_path / "a", lib, 1.2, None)
    assert plain.unrouted == {} and plain.params.edge_clearance_mm == 0.3 and "kicad_minimums" not in plain.stats
    assert math.isclose(_edge_gap(plain), 0.3)
    narrow = _wall_board(tmp_path / "b", lib, 1.2, 4)
    assert list(narrow.unrouted) == ["N"] and narrow.params.edge_clearance_mm == 0.5
    r = _wall_board(tmp_path / "c", lib, 1.45, 4)
    assert r.unrouted == {} and _edge_gap(r) >= 0.5 - 1e-9 and r.params.track_width_mm == P.track_width_mm
    assert r.stats["kicad_minimums"] == {"checked": {"min_track_width": 0.2, "min_clearance": 0.2, "min_copper_edge_clearance": 0.5,
                                                     "min_via_diameter": 0.5, "min_through_hole_diameter": 0.3, "min_hole_to_hole": 0.25},
                                         "raised": {"edge_clearance_mm": [0.3, 0.5]}}
    assert r.stats["raised"] == {}  # the fab raises stay apart (the agent calls them ir.pcb.manufacturing minimums)
    entry = r.tracks[0].provenance.derived_from
    assert entry[-1].endswith(";kicad_minimums=track:0.2,clearance:0.2,edge:0.5,via:0.5/0.3,hole_to_hole:0.25") and ",edge=0.5," in _params_entry(r)
    assert ("copper 0.5 mm from the board edge, every via at least 0.5 mm across with a hole of at least 0.3 mm - the minimums kicad-cli DRC "
            "checks this board against") in r.tracks[0].provenance.note
    # the floor never lowers a value: a stricter one stays, a looser one is raised
    ir = board_ir(tmp_path / "d", lib, [("R1", "PAD1", 1.0, 3.0)], {}, (8.0, 6.0))
    p, checked, raised = kicad_floor(ir, RoutingParams(edge_clearance_mm=0.8, track_width_mm=0.15))
    assert p.edge_clearance_mm == 0.8 and p.track_width_mm == 0.2 and checked["min_clearance"] == 0.2
    assert raised == {"track_width_mm": (0.15, 0.2), "escape_min_width_mm": (0.1, 0.2)}
    # a via below KiCad's 0.5 / 0.3 mm (kicad-cli flags a through or blind one: via_diameter, drill_out_of_range) is raised to it; a
    # drill the raise would bring to the diameter is refused by name
    p, _checked, raised = kicad_floor(ir, RoutingParams(via_diameter_mm=0.45, via_drill_mm=0.2))
    assert (p.via_diameter_mm, p.via_drill_mm) == (0.5, 0.3) and raised["via_diameter_mm"] == (0.45, 0.5) and raised["via_drill_mm"] == (0.2, 0.3)
    # (a written min_via_diameter of 0.25 mm keeps a 0.3 mm via, KiCad's own 0.3 mm hole minimum still applies)
    ir.pcb.manufacturing = _fab(min_via_diameter_mm=_authoritative(0.25))
    with pytest.raises(CompileError, match=r"KiCad's minimum via hole 0\.3 mm .* leaves the via drill 0\.3 mm at or above the via diameter 0\.3 mm"):
        kicad_floor(ir, RoutingParams(via_diameter_mm=0.3, via_drill_mm=0.2))


def _params_entry(r) -> str:
    return next(e for e in r.tracks[0].provenance.derived_from if e.startswith("params:"))


def test_a_stack_boards_escape_stubs_are_at_least_kicads_minimum_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The synthetic QFP-20 on the 6-layer stack (its escapes at the fine rules): the narrowest stub is raised from 0.1 mm to KiCad's
    0.2 mm, recorded with the escape knobs (a board with escapes), and no stub is narrower."""
    from tests.test_routing_fine_pitch import FINE

    monkeypatch.setattr(maze, "_ESCAPE_TRIALS", (("natural", "near", "plane"),))
    area = [(0.5, 0.5), (23.5, 0.5), (23.5, 23.5), (0.5, 23.5)]
    ir, lib = _qfp20_board(tmp_path)
    r = route_board(ir, lib, FINE, plane_nets={"GND": [("In1.Cu", area)]}, stack=_n_stack(ir, 6))
    assert r.unrouted == {} and r.stats["escapes"]
    assert r.stats["kicad_minimums"]["raised"] == {"escape_min_width_mm": [0.1, 0.2], "edge_clearance_mm": [0.3, 0.5]}
    assert r.params.escape_min_width_mm == 0.2 and ",escape_min_width=0.2," in _params_entry(r)
    stubs = [t for t in r.tracks if maze.escape_entry(t) is not None]
    assert stubs and min(t.width_mm for t in stubs) >= 0.2 - 1e-9


def test_the_fabs_raise_of_the_narrowest_escape_stub_stays_the_fabs_on_a_stack_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Review finding: on a stack board KiCad's floor raised the narrowest stub before the fab's ``min_track_width_mm`` did, so a fab
    raise was filed under ``kicad_minimums`` (authoritative limit) or recorded from the floored 0.2 mm (assumed limit). The fab's raise
    is recorded against the caller's 0.1 mm in ``stats["raised"]``; KiCad's floor records only what it adds beyond the fab's."""
    from tests.test_routing_fine_pitch import FINE

    monkeypatch.setattr(maze, "_ESCAPE_TRIALS", (("natural", "near", "plane"),))
    area = [(0.5, 0.5), (23.5, 0.5), (23.5, 23.5), (0.5, 23.5)]
    for k, (traced, floor) in enumerate(((_authoritative(0.25), {}), (assumption(0.25, note="a guess"), {}),
                                         (assumption(0.15, note="a guess"), {"escape_min_width_mm": [0.15, 0.2]}))):
        ir, lib = _qfp20_board(tmp_path / str(k))
        ir.pcb.manufacturing = _fab(min_track_width_mm=traced)
        r = route_board(ir, lib, FINE, plane_nets={"GND": [("In1.Cu", area)]}, stack=_n_stack(ir, 6))
        assert r.unrouted == {} and r.stats["escapes"]
        assert r.stats["raised"] == {"escape_min_width_mm": [0.1, traced.value]}
        assert r.stats["kicad_minimums"]["raised"] == {**floor, "edge_clearance_mm": [0.3, 0.5]}
        assert r.params.escape_min_width_mm == max(traced.value, 0.2)


def test_the_agents_rules_note_names_the_raises_to_kicads_minimums():
    """Review finding: an unrouted stack rung's note printed the floored values (edge 0.5 mm) as the fine rules' own."""
    from ai_eda.agents.pcb import PCBAgent
    from tests.test_routing_fine_pitch import FINE

    p = replace(FINE, edge_clearance_mm=0.5, rules="fine", pad_pitch_mm=0.5, pitch_footprint="Test:QFP20")
    note = PCBAgent._rules_note(p, {"clearance_mm": (0.2, 0.25)}, {"edge_clearance_mm": [0.3, 0.5]})
    assert note.endswith("edge 0.5 mm (raised to ir.pcb.manufacturing minimums: clearance_mm 0.2 -> 0.25) (raised to the minimums kicad-cli DRC "
                         "checks: edge_clearance_mm 0.3 -> 0.5)")
    assert PCBAgent._rules_note(p, None, {}) == PCBAgent._rules_note(p)  # a board without a stack: as before


def test_a_stack_board_routes_its_vias_at_kicads_minimums_and_keeps_its_own_drills_apart(tmp_path: Path, lib: KicadLibrary):
    """Review finding: a caller's 0.45 / 0.2 mm via routed on a stack board although kicad-cli flags it (via_diameter,
    drill_out_of_range), and one net's holes were kept only the clearance (0.2 mm) apart against KiCad's 0.25 mm hole-to-hole."""
    ir, fine, planes = _near_via_board(tmp_path, lib)
    r = route_board(ir, lib, replace(fine, via_diameter_mm=0.45, via_drill_mm=0.2), plane_nets=planes, stack=_n_stack(ir))
    assert r.unrouted == {} and r.vias and all((v.diameter_mm, v.drill_mm) == (0.5, 0.3) for v in r.vias)
    raised = r.stats["kicad_minimums"]["raised"]
    assert raised["via_diameter_mm"] == [0.45, 0.5] and raised["via_drill_mm"] == [0.2, 0.3]
    entry = r.vias[0].provenance.derived_from[-1]
    assert ";size=0.5/0.3;" in entry and ";net_hole_gap=0.3+0.25;" in entry
    assert "a net's own vias 0.55 mm apart centre to centre" in r.vias[0].provenance.note


# --------------------------------------------------------------------------- the plane web check (pure geometry)


def _via_item(net: str | None, x: float, y: float, layers: tuple[str, ...] = ("F.Cu", "In1.Cu", "B.Cu"), r: float = 0.3, label: str = "") -> PlaneItem:
    return PlaneItem("via", label or f"{net}@{x}/{y}", net, x, y, r, (x - r, y - r, x + r, y + r), frozenset(layers))


def _pad_item(net: str | None, x: float, y: float, half: float = 0.8, label: str = "") -> PlaneItem:
    return PlaneItem("pad", label or f"{net} pad {x}/{y}", net, x, y, half, (x - half, y - half, x + half, y + half), frozenset(("F.Cu", "In1.Cu", "B.Cu")))


ZONE = [(0.5, 0.5), (13.5, 0.5), (13.5, 11.5), (0.5, 11.5)]
#: the STM32F407 example's In1.Cu around U1.10's GND via (56.8, 61.2), moved by (-50, -55): PC14's fan-out via, U1.11's +3V3 via and
#: PC14's negotiated via (kicad-cli 10.0.6: the GND via touched no fill - via_dangling)
MEASURED = {"V": (6.8, 6.2), "F1": (6.6, 5.4), "F2": (6.6, 7.0), "F3": (7.4, 5.4)}


def _anchors() -> list[PlaneItem]:
    return [_pad_item("GND", 2.0, 2.0, label="G1"), _pad_item("GND", 12.0, 10.0, label="G2")]


def test_the_check_cuts_off_the_via_kicad_left_dangling_and_joins_it_without_the_negotiated_via():
    v = _via_item("GND", *MEASURED["V"], label="V")
    foreign = [_via_item("PC14", *MEASURED["F1"]), _via_item("+3V3", *MEASURED["F2"])]
    web = plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *foreign, _via_item("PC14", *MEASURED["F3"])], **FILL)
    assert web.cut == {"GND": [2]} and web.items == {"GND": [0, 1, 2]} and web.parts["GND"] == 2
    # the fill's best point within the via's reach misses it by 0.9 um (the measurement, the module docstring): the check must not
    # be generous; without PC14's second via the web between the two flanking vias is 0.075 mm wider than the pruning needs
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *foreign], **FILL).cut == {"GND": []}


def _arc(radius: float, start: float, end: float, steps: int) -> list[PlaneItem]:
    """Foreign through vias on a circle round (7, 6) from angle ``start`` to ``end`` (degrees), ``steps`` steps apart."""
    angles = [math.radians(start + k * (end - start) / steps) for k in range(steps + 1)]
    return [_via_item("S", 7.0 + radius * math.cos(a), 6.0 + radius * math.sin(a)) for a in angles]


def test_a_ring_of_knockouts_isolates_the_plane_inside_it_and_an_own_via_across_it_bridges():
    """Twelve vias 1 mm round a GND via (0.52 mm apart: their knockouts overlap) leave it an island; half of them leave it open. On a
    4 mm ring whose widest gap is 1.84 mm - no web passes, the knockouts and the pruning need 1.87 - a GND via in that gap whose
    copper reaches the fill on both sides joins the island to the plane outside, as U1.74's GND via did on the STM32F407 board."""
    inside = _via_item("GND", 7.0, 6.0, label="in")
    web = plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), inside, *_arc(1.0, 0.0, 330.0, 11)], **FILL)
    assert [web.cut["GND"], web.parts["GND"]] == [[2], 2]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), inside, *_arc(1.0, 150.0, 330.0, 6)], **FILL).cut == {"GND": []}
    ring = _arc(4.0, 13.3, 346.7, 23)  # 24 vias, at most 1.05 mm apart but across the gap at angle 0: 2 x 4 sin(13.3 deg) = 1.84 mm
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), inside, *ring], **FILL).cut == {"GND": [2]}
    bridge = _via_item("GND", 11.0, 6.0, label="bridge")
    web = plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), inside, bridge, *ring], **FILL)
    assert web.cut == {"GND": []} and web.parts["GND"] == 1


def test_only_copper_on_the_plane_layer_knocks_it_out_and_pads_count_as_copper():
    v = _via_item("GND", 7.0, 6.0, label="V")
    flank = [_via_item("S", 6.2, 6.0, ("In2.Cu", "In3.Cu")), _via_item("S", 7.8, 6.0, ("In2.Cu", "In3.Cu")),
             _via_item("S", 7.0, 5.2, ("In2.Cu", "In3.Cu")), _via_item("S", 7.0, 6.8, ("In2.Cu", "In3.Cu"))]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *flank], **FILL).cut == {"GND": []}  # buried vias: not on In1.Cu
    through = [_via_item("S", it.x, it.y) for it in flank]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *through], **FILL).cut == {"GND": [2]}
    # a net-less through-hole pad (an NPTH hole) is a knockout as well; a pad of the plane's own net is joined like a via
    holes = [_pad_item(None, 7.0, 6.0 + dy, half=0.5) for dy in (-1.4, 1.4)] + [_pad_item(None, 7.0 + dx, 6.0, half=0.5) for dx in (-1.4, 1.4)]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *holes], **FILL).cut == {"GND": [2]}
    gnd_pad = _pad_item("GND", 7.0, 6.0, half=0.6)
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), gnd_pad], **FILL).cut == {"GND": []}
    # a track of another net on the plane layer is a knockout too (its copper grown by the clearance); on another layer it is not;
    # the plane net's own track there adds no fill the check counts (left out: the safe side)
    square = [((6.0, 5.0), (8.0, 5.0)), ((8.0, 5.0), (8.0, 7.0)), ((8.0, 7.0), (6.0, 7.0)), ((6.0, 7.0), (6.0, 5.0))]

    def tracks(net: str, layer: str) -> list[PlaneItem]:
        return [PlaneItem("track", f"{net} {a}-{b}", net, a[0], a[1], 0.125, (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])),
                          frozenset((layer,)), end=b) for a, b in square]

    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *tracks("S", "In1.Cu")], **FILL).cut == {"GND": [2]}
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *tracks("S", "In2.Cu")], **FILL).cut == {"GND": []}
    web = plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), v, *tracks("GND", "In1.Cu")], **FILL)
    assert web.cut == {"GND": []} and web.items == {"GND": [0, 1, 2]}
    # a via outside every zone of its net touches no fill
    assert plane_web({"GND": [("In1.Cu", [(0.5, 0.5), (5.0, 0.5), (5.0, 5.0), (0.5, 5.0)])]}, [_anchors()[0], _anchors()[0], v], **FILL).cut == {"GND": [2]}


#: review finding (KiCad deletes a shrunk fill's islands below the minimum thickness): four foreign vias of A round (7, 6) leave a pocket
#: smaller than that between them; GND via D in the gap above touches the pocket and the plane, GND via B touches only the pocket
POCKET = {"A": [(7.0 + dx, 6.0 + dy) for dx in (-0.85, 0.85) for dy in (-0.475, 0.475)], "D": (7.0, 5.525), "B": (7.0, 6.375)}


def test_a_pocket_smaller_than_the_minimum_thickness_is_no_fill_whatever_bridges_it():
    """Every via pair at least 0.8 mm apart (the router's via_diameter + clearance). The pocket is no fill, so B is cut off although D's
    copper reaches both the pocket and the plane - as kicad-cli 10.0.6 left it dangling (the canary below); before, the check joined B
    through the pocket and D."""
    foreign = [_via_item("A", x, y) for x, y in POCKET["A"]]
    d, b = _via_item("GND", *POCKET["D"], label="D"), _via_item("GND", *POCKET["B"], label="B")
    web = plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), d, b, *foreign], **FILL)
    assert web.cut == {"GND": [3]} and web.parts["GND"] == 2
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), b, *foreign], **FILL).cut == {"GND": [2]}
    # the fill outside the pocket is untouched: D, the anchors and a via in a wide web stay joined
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, [*_anchors(), d, *foreign], **FILL).cut == {"GND": []}


#: review finding: a plane split round an interior keep-out into touching rectangles (``rect_difference``, as ``agents.pcb._clip_planes``
#: passes it on), three GND through-hole parts joined only by In1.Cu - in the top band, the bottom band and the left strip
SEAM_PARTS = [[(0.5, 0.5), (19.5, 0.5), (19.5, 4.0), (0.5, 4.0)], [(0.5, 4.0), (8.0, 4.0), (8.0, 8.0), (0.5, 8.0)],
              [(12.0, 4.0), (19.5, 4.0), (19.5, 8.0), (12.0, 8.0)], [(0.5, 8.0), (19.5, 8.0), (19.5, 11.5), (0.5, 11.5)]]
SEAM_PADS = [(4.0, 2.0), (16.0, 10.0), (4.0, 6.0)]


def _seam_items() -> list[PlaneItem]:
    return [_pad_item("GND", x + dx, y, label=f"R{i + 1}.{k + 1}") for i, (x, y) in enumerate(SEAM_PADS) for k, dx in enumerate((-1.5, 1.5))]


def test_touching_zones_of_one_net_are_one_plane_and_each_zone_is_pruned_on_its_own():
    """Before, every shared edge was a 0.27 mm cut and R2 / R3 were cut off (3 parts); kicad-cli 10.0.6 joined the touching zones (the
    canary below). The zones join where both fills reach their shared edge; each is still pruned on its own, so a seam with a knockout
    across it, or two zones touching only at a corner, joins nothing there."""
    from ai_eda.tools.keepout import rect_difference

    assert rect_difference((0.5, 0.5, 19.5, 11.5), [(8.0, 4.0, 12.0, 8.0)]) == SEAM_PARTS
    web = plane_web({"GND": [("In1.Cu", poly) for poly in SEAM_PARTS]}, _seam_items(), **FILL)
    assert web.cut == {"GND": []} and web.parts["GND"] == 1
    # the left strip (R3) touches the bands at y = 4 and y = 8; foreign through-hole pads on both seams cut both joins
    walls = [_pad_item("S", x, y, half=0.4) for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) for y in (4.0, 8.0)]
    web = plane_web({"GND": [("In1.Cu", poly) for poly in SEAM_PARTS]}, [*_seam_items(), *walls], **FILL)
    assert web.cut == {"GND": [4, 5]} and web.parts["GND"] == 2
    # two wall pads off the y = 4 seam reopen it
    web = plane_web({"GND": [("In1.Cu", poly) for poly in SEAM_PARTS]},
                    [*_seam_items(), *(w for w in walls if (w.x, w.y) not in ((4.0, 4.0), (5.0, 4.0)))], **FILL)
    assert web.cut == {"GND": []}
    # a vertical seam: two halves of the zone, a GND pad in each, joined
    halves = [[(0.5, 0.5), (7.0, 0.5), (7.0, 11.5), (0.5, 11.5)], [(7.0, 0.5), (13.5, 0.5), (13.5, 11.5), (7.0, 11.5)]]
    assert plane_web({"GND": [("In1.Cu", poly) for poly in halves]}, _anchors(), **FILL).cut == {"GND": []}
    # zones touching at a corner only share no seam (the smaller one, G2's, is cut off); each zone is pruned at its own outline - a
    # zone 0.2 mm wide between two others has no fill at all and joins nothing, however wide the three are together
    corner = [[(0.5, 0.5), (7.0, 0.5), (7.0, 6.0), (0.5, 6.0)], [(7.0, 6.0), (13.0, 6.0), (13.0, 11.5), (7.0, 11.5)]]
    assert plane_web({"GND": [("In1.Cu", poly) for poly in corner]}, _anchors(), **FILL).cut == {"GND": [1]}
    strip = [[(0.5, 0.5), (7.0, 0.5), (7.0, 11.5), (0.5, 11.5)], [(7.0, 0.5), (7.2, 0.5), (7.2, 11.5), (7.0, 11.5)],
             [(7.2, 0.5), (13.5, 0.5), (13.5, 11.5), (7.2, 11.5)]]
    assert plane_web({"GND": [("In1.Cu", poly) for poly in strip]}, _anchors(), **FILL).cut == {"GND": [1]}
    assert plane_web({"GND": [("In1.Cu", poly) for poly in strip[::2]] + [("In1.Cu", [(7.0, 0.5), (7.2, 0.5), (7.2, 11.5), (7.0, 11.5)])]},
                     _anchors(), **FILL).cut == {"GND": [1]}


def test_each_zone_is_filled_at_its_own_clearance():
    """Review finding: the compiler writes a zone's own clearance_mm; KiCad filled the canary geometry at 0.8 mm and left the GND via
    dangling (the canary below), while the check at 0.5 mm joined it."""
    v = _via_item("GND", *MEASURED["V"], label="V")
    items = [*_anchors(), v, _via_item("PC14", *MEASURED["F1"]), _via_item("+3V3", *MEASURED["F2"])]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, items, **FILL).cut == {"GND": []}
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, items, **FILL, clearances={"GND": [0.8]}).cut == {"GND": [2]}
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, items, **FILL, clearances={"+3V3": [0.8]}).cut == {"GND": []}  # another net's zones
    with pytest.raises(ValueError, match="2 clearance"):
        plane_web({"GND": [("In1.Cu", ZONE)]}, items, **FILL, clearances={"GND": [0.8, 0.8]})


@needs_kicad
def test_kicad_leaves_dangling_exactly_the_via_the_check_cuts_off(tmp_path: Path, lib: KicadLibrary):
    """Canary (kicad-cli): the measured geometry compiled - a GND via fed by an F.Cu track from C1's GND pad, between two foreign vias
    0.825 mm away, with and without a third 1.0 mm away, GND on In1.Cu (a through-hole GND pad in the fill): kicad-cli's
    ``--refill-zones`` DRC reports the via ``via_dangling`` exactly when :func:`plane_web` cuts it off - also when the zone has its own
    0.8 mm clearance (review finding: the check at 0.5 mm joined the via KiCad left dangling)."""
    def board(with_third: bool, tmp: Path, zone_clearance: float | None = None) -> tuple[bool, bool]:
        parts = [("C1", "SMD2", 3.3, 6.2), ("R1", "PAD2", 8.0, 9.5)]
        ir = board_ir(tmp, lib, parts, {"GND": [("C1", "2"), ("R1", "1"), ("R1", "2")], "A": [("C1", "1")]}, (12.0, 11.0))
        ir.nets[0].kind = NetKind.GROUND
        _n_stack(ir, 6)
        vx, vy = MEASURED["V"]
        foreign = [MEASURED["F1"], MEASURED["F2"], *([MEASURED["F3"]] if with_third else [])]
        ir.pcb.vias = [Via(net="GND", x_mm=vx, y_mm=vy, drill_mm=0.3, diameter_mm=0.6, provenance=HAND),
                       *(Via(net="A", x_mm=x, y_mm=y, drill_mm=0.3, diameter_mm=0.6, provenance=HAND) for x, y in foreign)]
        ir.pcb.tracks = [Track(net="GND", layer="F.Cu", start=(4.8, vy), end=(vx, vy), width_mm=0.25, provenance=HAND)]
        zone = [(0.5, 0.5), (11.5, 0.5), (11.5, 10.5), (0.5, 10.5)]
        ir.pcb.zones = [Zone(net="GND", layer="In1.Cu", polygon=zone, provenance=HAND, clearance_mm=zone_clearance)]
        out = tmp / "out"
        art = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": lib}))
        kicad.run_drc(Path(art.path), out / "drc.json")
        report = json.loads((out / "drc.json").read_text(encoding="utf-8"))
        dangling = any(v["type"] == "via_dangling" and any(abs(i["pos"]["x"] - vx) < 1e-6 and abs(i["pos"]["y"] - vy) < 1e-6 for i in v["items"])
                       for v in report["violations"])
        items = [_via_item("GND", vx, vy, tuple(ir.pcb.layers[k].name for k in range(6))), *(_via_item("A", x, y, ("F.Cu", "In1.Cu")) for x, y in foreign),
                 _pad_item("GND", 6.5, 9.5, half=0.8), _pad_item("GND", 9.5, 9.5, half=0.8)]
        clearances = maze._web_clearances(ir, {"GND": [("In1.Cu", zone)]}, 0.2)
        cut = plane_web({"GND": [("In1.Cu", zone)]}, items, **FILL, clearances=clearances).cut["GND"] == [0]
        return dangling, cut

    assert board(True, tmp_path / "three") == (True, True)
    assert board(False, tmp_path / "two") == (False, False)
    assert board(False, tmp_path / "own", zone_clearance=0.8) == (True, True)


def _dangling_at(report: dict, points: list[tuple[float, float]]) -> list[bool]:
    return [any(v["type"] == "via_dangling" and any(abs(i["pos"]["x"] - x) < 1e-6 and abs(i["pos"]["y"] - y) < 1e-6 for i in v["items"])
                for v in report["violations"]) for x, y in points]


@needs_kicad
def test_kicad_deletes_the_pocket_and_joins_touching_zones_as_the_check_does(tmp_path: Path, lib: KicadLibrary):
    """Canary (kicad-cli) for the two fill steps the review found missing. :data:`POCKET` compiled - D and B each on its own 0.2 mm GND
    SMD pad (their F.Cu side), GND on In1.Cu: kicad-cli 10.0.6 leaves B dangling and joins D, as the check cuts B off and joins D. And
    :data:`SEAM_PARTS` as four GND zones on In1.Cu with the three through-hole parts: no unconnected item, as the check joins them
    (kicad-cli also reports the touching zones as ``zones_intersect``, the keep-out clip's own matter)."""
    (tmp_path / "kicad" / "footprints" / "Test.pretty" / "TINY2.kicad_mod").write_text(  # the lib fixture's footprints
        '(footprint "TINY2" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n'
        '  (pad "1" smd rect (at 0 0) (size 0.2 0.2) (layers "F.Cu" "F.Mask"))\n'
        '  (pad "2" smd rect (at 0 -3.2) (size 0.2 0.2) (layers "F.Cu" "F.Mask")))\n', encoding="utf-8")
    parts = [("G1", "PAD2", 3.0, 9.5), ("U1", "TINY2", *POCKET["D"]), ("U2", "TINY2", *POCKET["B"])]
    ir = board_ir(tmp_path / "pocket", lib, parts, {"GND": [("G1", "1"), ("G1", "2"), ("U1", "1"), ("U2", "1")], "A": [("U1", "2")]}, (14.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    _n_stack(ir, 6)
    gnd = [POCKET["D"], POCKET["B"]]
    ir.pcb.vias = [*(Via(net="GND", x_mm=x, y_mm=y, drill_mm=0.3, diameter_mm=0.6, provenance=HAND) for x, y in gnd),
                   *(Via(net="A", x_mm=x, y_mm=y, drill_mm=0.3, diameter_mm=0.6, provenance=HAND) for x, y in POCKET["A"])]
    ir.pcb.zones = [Zone(net="GND", layer="In1.Cu", polygon=ZONE, provenance=HAND)]
    out = tmp_path / "pocket" / "out"
    art = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": lib}))
    kicad.run_drc(Path(art.path), out / "drc.json")
    assert _dangling_at(json.loads((out / "drc.json").read_text(encoding="utf-8")), gnd) == [False, True]
    items = [_pad_item("GND", 1.5, 9.5), _pad_item("GND", 4.5, 9.5), *(_via_item("GND", x, y) for x, y in gnd),
             *(_via_item("A", x, y) for x, y in POCKET["A"])]
    assert plane_web({"GND": [("In1.Cu", ZONE)]}, items, **FILL).cut == {"GND": [3]}
    # the seam board
    seam = board_ir(tmp_path / "seam", lib, [(f"R{i + 1}", "PAD2", x, y) for i, (x, y) in enumerate(SEAM_PADS)],
                    {"GND": [(f"R{i}", n) for i in (1, 2, 3) for n in ("1", "2")]}, (20.0, 12.0))
    seam.nets[0].kind = NetKind.GROUND
    _n_stack(seam, 6)
    seam.pcb.zones = [Zone(net="GND", layer="In1.Cu", polygon=poly, provenance=HAND) for poly in SEAM_PARTS]
    out = tmp_path / "seam" / "out"
    art = PCBCompiler().compile(seam, CompileContext(workdir=out, tools={"kicad_library": lib}))
    kicad.run_drc(Path(art.path), out / "drc.json")
    assert json.loads((out / "drc.json").read_text(encoding="utf-8"))["unconnected_items"] == []
    assert plane_web({"GND": [("In1.Cu", poly) for poly in SEAM_PARTS]}, _seam_items(), **FILL).cut == {"GND": []}


# --------------------------------------------------------------------------- the router's plane web pass


def test_a_board_without_a_cut_off_plane_via_keeps_its_copper_and_provenance(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    ir = _gnd_3v3_board(tmp_path, lib)
    r = route_board(ir, lib, plane_nets=PLANES, stack=_router_stack(ir))
    assert r.stats["plane_web"] == {"clearance_mm": 0.5, "min_thickness_mm": 0.25, "stricter": WEB_STRICTER, "keep_mm": 1.16,
                                    "checked": {"+3V3": 2, "GND": 2}, "rerouted": [], "cut": [], "why": {}}
    assert "no link of fill parts closer than the minimum thickness" in WEB_STRICTER  # a cut may be a false one (module docstring)
    assert not any("plane_web" in i.provenance.derived_from[-1] for i in [*r.tracks, *r.vias])
    monkeypatch.setattr(maze, "_plane_web_pass", lambda *a: {"rerouted": [], "cut": []})
    off = route_board(_gnd_3v3_board(tmp_path / "off", lib), lib, plane_nets=PLANES, stack=_router_stack(ir))
    assert [(t.model_dump(exclude={"provenance"}), t.provenance.derived_from, t.provenance.note) for t in r.tracks] == \
           [(t.model_dump(exclude={"provenance"}), t.provenance.derived_from, t.provenance.note) for t in off.tracks]
    assert [v.model_dump(exclude={"provenance"}) for v in r.vias] == [v.model_dump(exclude={"provenance"}) for v in off.vias]


def _near_via_board(tmp_path: Path, lib: KicadLibrary):
    """At the fine rules on the 6-layer stack: C1's GND pad gets its micro via at (5.8, 4.0) and S (a top pad over a bottom pad at
    (5.2, 5.0)) its through via at (5.6, 5.0), 1.02 mm away - nearer than the plane web's keep (0.3 + 1.06 mm), a GND through-hole
    pad G1 in the fill."""
    from tests.test_routing_fine_pitch import FINE

    parts = [("C1", "SMD1", 5.0, 4.0), ("G1", "PAD1", 2.0, 7.0), ("R1", "TINY", 5.2, 5.0), ("R2", "TINY", 5.2, 5.0)]
    ir = board_ir(tmp_path, lib, parts, {"GND": [("C1", "1"), ("G1", "1")], "S": [("R1", "1"), ("R2", "1")]}, (12.0, 9.0),
                  sides={"R2": BoardSide.BOTTOM})
    ir.nets[0].kind = NetKind.GROUND
    return ir, FINE, {"GND": [("In1.Cu", [(0.5, 0.5), (11.5, 0.5), (11.5, 8.5), (0.5, 8.5)])]}


def _report_cut(monkeypatch: pytest.MonkeyPatch, label: str, *later: str | None) -> list[int]:
    """The plane web check, but its first call also reports ``label`` cut off, and its next calls the labels of ``later`` in turn (``None``
    for none; the geometry of a real cut is the pure tests' business; these test what the pass does with one). Returns the list the calls
    are counted in."""
    real = maze.plane_web
    calls: list[int] = []

    def check(planes, items, **kw):
        web = real(planes, items, **kw)
        extra = ([label, *later] + [None])[min(len(calls), len(later) + 1)]
        if extra is not None:
            i = next(i for i, it in enumerate(items) if it.label == extra)
            web.cut = {**web.cut, "GND": sorted({*web.cut.get("GND", []), i})}
        calls.append(len(items))
        return web

    monkeypatch.setattr(maze, "plane_web", check)
    return calls


def test_a_cut_off_plane_vias_cutters_are_routed_again_away_from_it(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    ir, fine, planes = _near_via_board(tmp_path, lib)
    before = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    [gnd] = [(v.x_mm, v.y_mm) for v in before.vias if v.net == "GND"]
    [s_via] = [(v.x_mm, v.y_mm) for v in before.vias if v.net == "S"]
    assert gnd == (5.8, 4.0) and s_via == (5.6, 5.0) and before.stats["plane_web"]["cut"] == []
    calls = _report_cut(monkeypatch, "C1.1:via")
    r = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    assert len(calls) == 2  # the pass checked again after the move, and found nothing more
    assert r.unrouted == {} and r.stats["legal"] is True
    web = r.stats["plane_web"]
    assert web["keep_mm"] == 1.06 and web["rerouted"] == [{"item": "C1.1:via", "net": "GND", "at": [5.8, 4.0], "nets": ["S"]}]
    assert web["cut"] == [] and web["why"] == {}
    s_vias = [v for v in r.vias if v.net == "S"]
    assert s_vias and all(math.hypot(v.x_mm - gnd[0], v.y_mm - gnd[1]) - 0.3 >= 1.06 for v in s_vias)  # outside the keep of its copper
    assert [(v.x_mm, v.y_mm) for v in r.vias if v.net == "GND"] == [gnd]  # the plane via itself never moves
    s = next(t for t in r.tracks if t.net == "S")
    assert "a via of its earlier route cut a plane via of another net off from its plane" in s.provenance.note
    entry = s.provenance.derived_from[-1]
    assert entry.endswith(";plane_web=clearance:0.5,min_thickness:0.25,keep:1.06:C1.1:via>S")
    assert "the nets whose vias cut them off (S) were routed again against the legal copper" in s.provenance.note
    got = _routing_checks(ir, r, lib, tmp_path)
    assert got[CLEARANCE_CHECK].status.value == "PASS", got[CLEARANCE_CHECK].message


def test_a_reroute_that_finds_no_way_is_undone_and_static_copper_is_named(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    from tests.test_rf_floorplan import keepout

    ir, fine, planes = _near_via_board(tmp_path, lib)
    # S's vias kept out everywhere but a 0.6 mm window round its via site (GND's allowed, so C1 keeps its plane via): S cannot move
    kos = [keepout("KV1", [0.0, 0.0, 12.0, 4.7], forbids=["vias"], nets=["GND"]), keepout("KV2", [0.0, 5.3, 12.0, 3.7], forbids=["vias"], nets=["GND"]),
           keepout("KV3", [0.0, 4.7, 5.3, 0.6], forbids=["vias"], nets=["GND"]), keepout("KV4", [5.9, 4.7, 6.1, 0.6], forbids=["vias"], nets=["GND"])]
    before = route_board(ir, lib, fine, plane_nets=planes, keepouts=kos, stack=_n_stack(ir))
    _report_cut(monkeypatch, "C1.1:via")
    r = route_board(ir, lib, fine, plane_nets=planes, keepouts=kos, stack=_n_stack(ir))
    assert r.stats["plane_web"]["rerouted"] == [] and r.stats["plane_web"]["cut"] == ["C1.1:via"]
    assert r.stats["plane_web"]["why"] == {"C1.1:via": "no strict route for S"}
    assert [(t.layer, t.start, t.end) for t in r.tracks if t.net == "S"] == [(t.layer, t.start, t.end) for t in before.tracks if t.net == "S"]
    assert r.tracks[0].provenance.derived_from[-1].endswith(";plane_web=clearance:0.5,min_thickness:0.25,keep:1.06:none;cut=C1.1:via")
    assert "no net's via could be moved; still cut off: C1.1:via" in r.tracks[0].provenance.note
    # a through-hole pad far from every negotiated via: its cutters can only be static copper
    monkeypatch.undo()
    _report_cut(monkeypatch, "G1.1")
    r = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    assert r.stats["plane_web"]["cut"] == ["G1.1"] and r.stats["plane_web"]["why"] == {"G1.1": "no negotiated via near it"}


def test_a_move_that_leaves_the_via_cut_off_or_cuts_another_off_is_undone_and_named(tmp_path: Path, lib: KicadLibrary,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    """Review finding: a reroute that did not join the item (static copper or vias farther than keep cut it too) was kept, recorded as
    a repair, and its item named "not tried". The pass checks again after the move and keeps it only when the item is joined and no
    item joined before is cut off; otherwise the move is undone and ``why`` says what it did."""
    ir, fine, planes = _near_via_board(tmp_path, lib)
    before = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    s_copper = [(t.layer, t.start, t.end) for t in before.tracks if t.net == "S"]
    calls = _report_cut(monkeypatch, "C1.1:via", "C1.1:via")  # still cut off after the move
    r = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    assert len(calls) == 2 and r.unrouted == {}
    web = r.stats["plane_web"]
    assert web["rerouted"] == [] and web["cut"] == ["C1.1:via"]
    assert web["why"] == {"C1.1:via": "still cut off after routing S again (static copper or vias farther than keep cut it too)"}
    assert [(t.layer, t.start, t.end) for t in r.tracks if t.net == "S"] == s_copper
    assert r.tracks[0].provenance.derived_from[-1].endswith(";plane_web=clearance:0.5,min_thickness:0.25,keep:1.06:none;cut=C1.1:via")
    monkeypatch.undo()
    calls = _report_cut(monkeypatch, "C1.1:via", "G1.1")  # joined after the move, but G1's pad cut off by it
    r = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    assert len(calls) == 2 and r.stats["plane_web"]["rerouted"] == [] and r.stats["plane_web"]["cut"] == ["C1.1:via"]
    assert r.stats["plane_web"]["why"] == {"C1.1:via": "routing S again cut G1.1 off"}
    assert [(t.layer, t.start, t.end) for t in r.tracks if t.net == "S"] == s_copper


def test_the_pass_fills_each_plane_zone_at_its_own_clearance(tmp_path: Path, lib: KicadLibrary):
    """Review finding: the pass modelled every zone at 0.5 mm although the compiler writes a zone's own clearance_mm (and KiCad's fill
    keeps at least the checked clearance). The IR zone behind a plane (net, layer, polygon) gives it; keep grows with the largest."""
    ir, fine, planes = _near_via_board(tmp_path, lib)
    [(layer, poly)] = planes["GND"]
    assert maze._web_clearances(ir, planes, 0.2) == {"GND": [0.5]}  # no IR zone behind it: the compiler's default
    ir.pcb.zones = [Zone(net="GND", layer=layer, polygon=poly, provenance=HAND, clearance_mm=0.8),
                    Zone(net="GND", layer="In2.Cu", polygon=poly, provenance=HAND, clearance_mm=1.5)]  # another layer: not this plane's
    assert maze._web_clearances(ir, planes, 0.2) == {"GND": [0.8]} and maze._web_clearances(ir, planes, 0.9) == {"GND": [0.9]}
    ir.pcb.zones = ir.pcb.zones[:1]
    r = route_board(ir, lib, fine, plane_nets=planes, stack=_n_stack(ir))
    web = r.stats["plane_web"]
    assert web["clearance_mm"] == 0.8 and web["keep_mm"] == 1.36  # 0.3 + 0.8 + 0.25 + 0.01


def test_a_plane_split_round_a_keep_out_is_not_cut_at_its_seams(tmp_path: Path, lib: KicadLibrary):
    """Review finding: the router's pass on a plane given as the touching rectangles of a keep-out clip called the through-hole pads of
    the strips cut off at every seam ("no negotiated via near it", the stack entry ending ``;cut=...``); KiCad joins the zones."""
    parts = [("R1", "PAD2", 4.0, 2.0), ("R2", "PAD2", 16.0, 10.0), ("R3", "PAD2", 4.0, 6.0), ("C1", "SMD2", 15.0, 2.5)]
    ir = board_ir(tmp_path, lib, parts, {"GND": [(r, n) for r in ("R1", "R2", "R3") for n in ("1", "2")] + [("C1", "1")]}, (20.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    r = route_board(ir, lib, plane_nets={"GND": [("In1.Cu", poly) for poly in SEAM_PARTS]}, stack=_n_stack(ir, 6))
    assert r.stats["plane_web"]["cut"] == [] and r.stats["plane_web"]["rerouted"] == []
    assert not any("plane_web" in i.provenance.derived_from[-1] for i in [*r.tracks, *r.vias])


# U1.28's supply via and its neighbours, translated by (-40, -35) from the routed board.
LANE = [("GND", 5.0, 2.6), ("GND", 5.4, 5.0), ("PA4", 6.2, 5.0), ("PA4", 6.2, 5.8),
        ("VDDA", 4.8, 4.2), ("VDDA", 4.0, 4.0), ("A", 5.0, 3.4), ("A", 4.6, 5.0)]


def test_a_negotiated_via_cuts_a_static_fanout_lane_and_a_nearby_site_restores_the_fill():
    items = [_via_item(net, x, y) for net, x, y in LANE]
    planes = {"GND": [("In1.Cu", ZONE)]}
    assert plane_web(planes, items, **FILL).cut == {"GND": [1]}
    assert plane_web(planes, items[:2] + items[3:], **FILL).cut == {"GND": []}
    # Removing nearby VDDA does not help; the static fan-out via of PA4 stays in both counterfactuals.
    assert plane_web(planes, items[:4] + items[5:], **FILL).cut == {"GND": [1]}
    items[2] = _via_item("PA4", 6.4, 5.2)
    assert plane_web(planes, items, **FILL).cut == {"GND": []}


@pytest.mark.parametrize("neighbour_routes", [False, True])
def test_plane_web_retries_without_moving_an_unnecessary_neighbour(neighbour_routes):
    from types import SimpleNamespace

    p = RoutingParams(grid_mm=.2, via_diameter_mm=.6, via_drill_mm=.3, clearance_mm=.2)
    nx, ny = 71, 61
    n = nx * ny
    def cell(x, y):
        return round(y / p.grid_mm) * nx + round(x / p.grid_mm)
    def xy(k):
        j, i = divmod(k, nx)
        return (i * p.grid_mm, j * p.grid_mm)
    def path(x, y):
        k = cell(x, y)
        return maze._NetRoute(steps=[("path", [k, n + k], (0,))])

    links = {}
    for i in (0, 1, 3, 5, 6, 7):
        net, x, y = LANE[i]
        links.setdefault(net, []).append((SimpleNamespace(label=f"P{i}"), 0, cell(x, y)))
    links["A"].append((SimpleNamespace(label="anchor"), 0, cell(2, 9)))
    board = SimpleNamespace(p=p, nx=nx, ny=ny, n=n, ox=0, oy=0, xy=xy, pos=xy, pads=[],
                            stack=SimpleNamespace(copper=["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]),
                            plane_links=links, plane_link_types={}, plane_problems={}, plane_holes={},
                            net_index={name: i for i, name in enumerate(("GND", "PA4", "VDDA", "A"))},
                            signal_types=[SimpleNamespace(kind="through", planes=("In1.Cu", "In2.Cu"))], via_static_n=[[None] * n])
    routes = {"PA4": path(6.2, 5), "VDDA": path(4.8, 4.2)}
    original = dict(routes)
    active = {id(r) for r in routes.values()}
    neg = SimpleNamespace(remove=lambda r: active.remove(id(r)), add=lambda r: active.add(id(r)))
    attempts = []
    def route_one(net, strict):
        assert strict
        attempts.append(net.name)
        if net.name == "VDDA" and not neighbour_routes:
            return "no strict route in the fan-out lane"
        # Moving VDDA cuts P6 off its other plane; PA4 alone reconnects P1 with P6 still joined.
        point = (5.8, 3.6) if net.name == "VDDA" else (4.0, 6.0)
        assert board.via_static_n[0][cell(*point)] != maze.BLOCKED
        return path(*point)

    web = maze._plane_web_pass(board, neg, [SimpleNamespace(name=name) for name in routes], routes, route_one,
                               {"GND": [("In1.Cu", ZONE)], "A": [("In2.Cu", ZONE)]})
    assert active == {id(r) for r in routes.values()}
    assert web["cut"] == [] and len(web["rerouted"]) == 1
    move = web["rerouted"][0]
    assert move["item"] == "P1:via" and move["nets"] == ["PA4"]
    assert attempts.count("PA4") >= 2 and "VDDA" in attempts
    assert routes["VDDA"] is original["VDDA"] and active == {id(r) for r in routes.values()}
    assert board.via_static_n[0][cell(6.2, 5)] == maze.BLOCKED
    assert board.via_static_n[0][cell(4.0, 6.0)] is None
    assert board.via_static_n[0][cell(4.8, 4.2)] == maze.BLOCKED  # the neighbour stays even though its site is closed to new vias


@needs_kicad
@pytest.mark.parametrize("moved", [False, True])
def test_kicad_confirms_the_fanout_lane_cut_and_its_nearby_replacement(tmp_path, lib, moved):
    pretty = lib.roots[0] / "footprints" / "Test.pretty"
    (pretty / "LANE2.kicad_mod").write_text(
        '(footprint "LANE2" (version 20260206) (generator "pcbnew") (layer "F.Cu")\n'
        '  (pad "1" smd rect (at 0 0) (size .2 .2) (layers "F.Cu" "F.Mask"))\n'
        '  (pad "2" smd rect (at 0 3) (size .2 .2) (layers "F.Cu" "F.Mask")))\n', encoding="utf-8")
    ir = board_ir(tmp_path, lib, [("C1", "LANE2", 5.5, 7.175), ("G1", "PAD2", 3.5, 2)],
                  {"GND": [("C1", "1"), ("G1", "1"), ("G1", "2")], "A": [("C1", "2")]}, (14, 12))
    _n_stack(ir, 6)
    points = list(LANE)
    if moved:
        points[2] = ("PA4", 6.4, 5.2)
    # The extra nets need only a name for this copper-only fill canary.
    from ai_eda.ir import Net
    ir.nets.extend(Net(name=name, provenance=HAND) for name in ("PA4", "VDDA"))
    ir.pcb.vias = [Via(net=net, x_mm=x, y_mm=y, diameter_mm=.6, drill_mm=.3, provenance=HAND) for net, x, y in points]
    pts = [(5.5, 7.175), (5.5, 6.375), (5.4, 5)]
    ir.pcb.tracks = [Track(net="GND", layer="F.Cu", start=a, end=b, width_mm=.25, provenance=HAND)
                     for a, b in zip(pts, pts[1:])]
    ir.pcb.zones = [Zone(net="GND", layer="In4.Cu", polygon=ZONE, provenance=HAND)]
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    report = tmp_path / "out" / "drc.json"
    kicad.run_drc(Path(art.path), report)
    measured = json.loads(report.read_text(encoding="utf-8"))
    assert _dangling_at(measured, [(5.4, 5)]) == [not moved]
    assert any("[GND]" in item["description"] for row in measured["unconnected_items"] for item in row["items"]) is (not moved)
