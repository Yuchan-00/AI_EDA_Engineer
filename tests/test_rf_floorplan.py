"""RF layout (wave-1 part P6): the floorplan placer, keep-outs in the router / SI / compiler / ``pcb.keepout``, and plane-net pad vias.

Everything runs on the synthetic fixture library of ``tests/test_routing.py``
plus the footprints written here (``Test:CAN10`` a 10 x 10 mm shield can
whose pads, all numbered "1", fence a 9 x 9 mm inside and whose courtyard is
a ring, as the Laird BMI-S cans are; ``Test:SMDB`` a 2-pad 0603-like part;
``Test:BIG`` a 14 x 13 mm part - the RK09K pot's size). No KiCad, no
ngspice. What is proved:

* ``placement.rf_floorplan`` packs every block in its region in chain order,
  the can first and its parts inside the fence less the ring, the rest
  around the regions, honours keep-outs that ban footprints, and refuses
  (never overlaps) a region too small for a part or a can, a region outside
  the outline or on another, and a shielded block without a region; it
  places a board the grid placer cannot;
* the router (routing.maze 0.4, only when keep-outs or plane nets are given)
  keeps every track out of a track keep-out (an allowed net may pass), no
  via in a via keep-out, joins a plane net's SMD pads by a via each (none
  for a through-hole pad or for an exposed pad on its footprint's own
  same-numbered thermal vias; a TO-252 tab's via beyond its own copper -
  ``Test:TAB`` / ``Test:EPTV`` here, the real TO-252-2 / QFN-12 ThermalVias
  footprints when the KiCad libraries are installed), never by tracks, and
  leaves a board without either exactly as before;
* ``tools/si/measure.py`` and ``si.impedance`` treat copper over a plane
  cleared by a zone keep-out as having no reference plane (the critic's
  false-PASS path is closed), its delay bounded by the largest er of the
  stack (not the prepreg's: the field reaches the core), and ``spice.si``,
  ``si.critical_length`` and ``si.rf_length`` name the keep-out instead of
  "use pcb_layers=4"; ``pcb.routing.connectivity`` calls a
  plane-joined net NOT_VERIFIED and an SMD pad that no pour can reach FAIL;
* the compiler writes KiCad rule areas deterministically, the exemptions
  cut out of the polygon; ``pcb.keepout`` judges keep-outs, regions and can
  fences on the IR geometry;
* the PCB agent picks the floorplan for an IR with RF blocks, clips the
  planes by the keep-outs and hands the router its keep-outs and plane nets.

The IR types of part P1 (``ai_eda.ir.rf``, ``PCBDesign.keepouts``,
``Keepout``) are used when they exist; before that merge the tests build
stand-ins with the documented attribute names (the code under test reads
them by attribute) and attach them with ``model_copy(update=...)``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.design.board import PLANE_CLEARANCE_KEY
from ai_eda.design.stackup import STACKUP_TOOL, plane_zones
from ai_eda.errors import CompileError
from ai_eda.ir import (
    CircuitIR,
    ManufacturingConstraints,
    NetKind,
    Provenance,
    ProvenanceKind,
    Track,
    Zone,
    authoritative,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.tools import keepout as kg
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.grid import grid_placement
from ai_eda.tools.placement.rf_floorplan import PLACER_ID, RING_MM, fence_box, rf_floorplan_placement
from ai_eda.tools.routing.maze import PLANE_VIA_REACH_MM, ROUTER_KEEPOUT_VERSION, ROUTER_VERSION, route_board
from ai_eda.tools.calc.tline import propagation_delay
from ai_eda.tools.si.measure import KEEPOUT_NO_PLANE, measure_nets
from ai_eda.tools.spice.si_check import spice_si_results
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import CONNECTIVITY_CHECK, KEEPOUT_CHECK
from ai_eda.validation.si import si_results
from tests.conftest import DS
from tests.test_routing import board_ir, fixture_library
from tests.test_si_checks import default_classes, long_board, si_of, stacked
from tests.test_si_checks import track as si_track

#: part P1's IR types, when merged (the tests fall back to stand-ins with the documented names otherwise)
HAS_KEEPOUT_TYPE = importlib.util.find_spec("ai_eda.ir.rf") is not None
_REAL = KicadLibrary()
#: the real KiCad 10 footprint libraries (KICAD10_SYMBOL_DIR / an installed KiCad): the plane-via twins on the transceiver's footprints
needs_real_libs = pytest.mark.skipif(
    _REAL.footprint_file("Package_TO_SOT_SMD", "TO-252-2") is None or _REAL.footprint_file("Package_DFN_QFN", "QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm_ThermalVias") is None,
    reason="KiCad libraries not installed")
MM = "mm"


# --------------------------------------------------------------------------- fixtures


_EXTRA_FOOTPRINTS = {
    # a 10 x 10 mm can: fence pads (all "1") whose inner edges are 4.5 mm from the centre; a ring courtyard like BMI-S-102
    "CAN10": (
        '(attr smd)\n'
        '  (fp_rect (start -5.25 -5.25) (end 5.25 5.25) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        '  (fp_rect (start -4.25 -4.25) (end 4.25 4.25) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        + "\n".join(
            f'  (pad "1" smd rect (at {x} {y}) (size {w} {h}) (layers "F.Cu" "F.Mask" "F.Paste"))'
            for x, y, w, h in (
                *((-4.75, v, 0.5, 2.0) for v in (-3, 0, 3)), *((4.75, v, 0.5, 2.0) for v in (-3, 0, 3)),
                *((v, -4.75, 2.0, 0.5) for v in (-3, 0, 3)), *((v, 4.75, 2.0, 0.5) for v in (-3, 0, 3)),
            )
        )
    ),
    "SMDB": (
        '(attr smd)\n'
        '  (fp_rect (start -1.5 -0.8) (end 1.5 0.8) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        '  (pad "1" smd rect (at -0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
        '  (pad "2" smd rect (at 0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask" "F.Paste"))'
    ),
    "BIG": (
        '(attr through_hole)\n'
        '  (fp_rect (start -7 -6.5) (end 7 6.5) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        '  (pad "1" thru_hole circle (at -5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n'
        '  (pad "2" thru_hole circle (at 5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))'
    ),
    # a TO-252-2 (DPAK) like the LP38693DT's: a 6.4 x 5.8 mm tab (pad 2) - wider than the plane-via reach from its centre
    "TAB": (
        '(attr smd)\n'
        '  (fp_rect (start -6 -3.5) (end 4.9 3.5) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        '  (pad "1" smd rect (at -4.5 -2.28) (size 1.1 1.2) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
        '  (pad "2" smd rect (at 1.26 0) (size 6.4 5.8) (layers "F.Cu" "F.Mask"))'
    ),
    # a QFN "..._ThermalVias" exposed pad: EP 2 on F.Cu and B.Cu, four same-numbered through-hole vias on it, and a ring of
    # unconnected leads (pad 1) on every axis, so no stub can leave the EP along the grid axes
    "EPTV": (
        '(attr smd)\n'
        '  (fp_rect (start -2.2 -2.2) (end 2.2 2.2) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        '  (pad "2" smd rect (at 0 0) (size 1.6 1.6) (layers "F.Cu" "F.Mask"))\n'
        '  (pad "2" smd rect (at 0 0) (size 1.6 1.6) (layers "B.Cu" "B.Mask"))\n'
        + "\n".join(f'  (pad "2" thru_hole circle (at {x} {y}) (size 0.5 0.5) (drill 0.2) (layers "*.Cu"))' for x, y in ((-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)))
        + "\n"
        + "\n".join(f'  (pad "1" smd rect (at {x} {y}) (size 0.5 0.5) (layers "F.Cu" "F.Mask" "F.Paste"))' for x, y in ((-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)))
    ),
}


def rf_library(root: Path) -> KicadLibrary:
    """The routing fixture library plus the footprints of the module docstring."""
    lib = fixture_library(root)
    pretty = root / "footprints" / "Test.pretty"
    for name, body in _EXTRA_FOOTPRINTS.items():
        (pretty / f"{name}.kicad_mod").write_text(f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu")\n  {body})\n', encoding="utf-8")
    return lib


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return rf_library(tmp_path / "kicad")


def keepout(kid: str, rect: list[float] | None = None, *, polygon: list[list[float]] | None = None, layers: list[str] | None = None,
            forbids: list[str], refs: list[str] | None = None, nets: list[str] | None = None, reason: str = "test keep-out"):
    """A keep-out: P1's :class:`Keepout` when it exists, else a stand-in with the documented attribute names."""
    layers = layers or ["*.Cu"]
    if HAS_KEEPOUT_TYPE:
        from ai_eda.ir.pcb import Keepout

        return Keepout(id=kid, layers=layers, rect=None if rect is None else user_requirement(rect, MM),
                       polygon=None if polygon is None else user_requirement(polygon, MM), forbids=forbids,
                       allowed_refs=refs or [], allowed_nets=nets or [], reason=reason,
                       provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test"))
    return SimpleNamespace(id=kid, layers=layers, rect=None if rect is None else SimpleNamespace(value=rect, unit=MM),
                           polygon=None if polygon is None else SimpleNamespace(value=polygon, unit=MM), forbids=forbids,
                           allowed_refs=refs or [], allowed_nets=nets or [], reason=reason)


def block(bid: str, refs: list[str], *, chain: list[str] | None = None, shield: str | None = None, region: tuple[float, float, float, float] | None = None):
    """An RF block: P1's :class:`RFBlock` when it exists, else a stand-in (``region`` is ``(x, y, w, h)`` mm)."""
    if HAS_KEEPOUT_TYPE:
        from ai_eda.ir.rf import RFBlock, RFRegion

        reg = None if region is None else RFRegion(**{k: user_requirement(float(v), MM) for k, v in zip("xywh", region)})
        return RFBlock(id=bid, refs=refs, chain=chain or [], shield_ref=shield, region=reg)
    reg = None if region is None else SimpleNamespace(**{k: SimpleNamespace(value=float(v), unit=MM) for k, v in zip("xywh", region)})
    return SimpleNamespace(id=bid, title="", refs=refs, chain=chain or [], shield_ref=shield, region=reg, ports=[])


def with_rf(ir: CircuitIR, blocks: list) -> CircuitIR:
    if HAS_KEEPOUT_TYPE:
        from ai_eda.ir.rf import RFDesign

        return ir.model_copy(update={"rf": RFDesign(blocks=blocks)})
    return ir.model_copy(update={"rf": SimpleNamespace(blocks=blocks)})


def with_keepouts(ir: CircuitIR, kos: list) -> CircuitIR:
    ir.pcb = ir.pcb.model_copy(update={"keepouts": kos})
    return ir


def unplaced(tmp_path: Path, lib: KicadLibrary, parts: list[tuple[str, str]], nets: dict | None = None, size: tuple[float, float] | None = (40.0, 30.0)) -> CircuitIR:
    """Parts without placements (the placer's input) on an outline of ``size`` (``None``: no outline)."""
    ir = board_ir(tmp_path, lib, [(ref, fp, 0.0, 0.0) for ref, fp in parts], nets or {}, size or (1.0, 1.0))
    ir.pcb.placements = []
    if size is None:
        ir.pcb.outline = None
    return ir


def _box(e) -> tuple[float, float, float, float]:
    return e.x1, e.y1, e.x2, e.y2


def _inside(e, box, tol: float = 1e-6) -> bool:
    return e.x1 >= box[0] - tol and e.y1 >= box[1] - tol and e.x2 <= box[2] + tol and e.y2 <= box[3] + tol


def _run(validator_id: str, ir: CircuitIR, tmp_path: Path, lib: KicadLibrary | None) -> dict:
    """``check id -> result`` of one registered validator (``pcb.routing`` gives two results, ``pcb.keepout`` one)."""
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib} if lib is not None else {})
    return {r.check_id: r for r in default_registry.get(validator_id).validate(ir, ctx)}


# --------------------------------------------------------------------------- keep-out geometry


def test_rect_difference_gives_simple_polygons_and_splits_a_hole_into_rectangles():
    band = kg.rect_difference((0.0, 0.0, 10.0, 10.0), [(0.0, 0.0, 10.0, 2.0)])
    assert band == [[(0.0, 2.0), (10.0, 2.0), (10.0, 10.0), (0.0, 10.0)]]
    notch = kg.rect_difference((0.0, 0.0, 10.0, 10.0), [(-1.0, 4.0, 3.0, 6.0)])
    assert len(notch) == 1 and len(notch[0]) == 8 and abs(kg.polygon_area(notch[0])) == pytest.approx(100.0 - 6.0)
    hole = kg.rect_difference((0.0, 0.0, 10.0, 10.0), [(4.0, 4.0, 6.0, 6.0)])  # one outline cannot hold a hole: rectangles
    assert len(hole) == 4 and sum(abs(kg.polygon_area(p)) for p in hole) == pytest.approx(96.0)
    split = kg.rect_difference((0.0, 0.0, 10.0, 10.0), [(4.0, -1.0, 6.0, 11.0)])
    assert [kg.area_bbox(p) for p in split] == [(0.0, 0.0, 4.0, 10.0), (6.0, 0.0, 10.0, 10.0)]
    assert kg.rect_difference((0.0, 0.0, 10.0, 10.0), [(-1.0, -1.0, 11.0, 11.0)]) == []


def test_area_tests_count_touching_as_no_overlap_and_refuse_to_guess_non_convex_pairs():
    sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert kg.point_in_polygon((10.0, 5.0), sq) and not kg.point_in_polygon((10.1, 5.0), sq)
    assert kg.box_area_overlap((10.0, 0.0, 12.0, 5.0), sq) == 0.0 and kg.box_area_overlap((5.0, 5.0, 15.0, 15.0), sq) == pytest.approx(25.0)
    assert kg.segment_area_distance((-5.0, -1.0), (15.0, -1.0), sq) == pytest.approx(1.0)
    assert kg.segment_area_distance((-5.0, 5.0), (15.0, 5.0), sq) == 0.0  # crossing
    u = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (7.0, 10.0), (7.0, 3.0), (3.0, 3.0), (3.0, 10.0), (0.0, 10.0)]
    assert not kg.is_convex(u) and kg.polygon_overlap_area(u, u) is None
    assert kg.polygon_overlap_area(u, [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]) == 0.0  # the convex one clips: the slot is empty
    k = keepout("K", [1.0, 2.0, 3.0, 4.0], forbids=["tracks"], layers=["F.Cu"])
    assert kg.area_points(k) == [(1.0, 2.0), (4.0, 2.0), (4.0, 6.0), (1.0, 6.0)] and kg.area_is_rect(k)
    assert kg.covers_layer(k, "F.Cu") and not kg.covers_layer(k, "B.Cu") and kg.covered_layers(keepout("A", [0, 0, 1, 1], forbids=["zones"]), ["F.Cu", "In1.Cu", "B.Cu"]) == ["F.Cu", "In1.Cu", "B.Cu"]


# --------------------------------------------------------------------------- the router (routing.maze 0.4)


def _two_pads(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """``N`` between two THT pads 14 mm apart on a 20 x 12 mm board."""
    return board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 5.0), ("R2", "PAD1", 17.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (20.0, 12.0))


def test_a_board_without_keepouts_or_plane_nets_routes_exactly_as_before(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    plain = route_board(ir, lib)
    for kw in ({"keepouts": None, "plane_nets": None}, {"keepouts": [], "plane_nets": {}}):
        again = route_board(ir, lib, **kw)
        assert [design_data(t) for t in again.tracks] == [design_data(t) for t in plain.tracks] and again.stats == plain.stats
        assert again.version == ROUTER_VERSION and "keepouts" not in again.stats


def test_a_track_keepout_is_an_obstacle_an_allowed_net_may_cross_and_a_closed_one_leaves_the_net_unrouted(tmp_path: Path, lib: KicadLibrary):
    ir = _two_pads(tmp_path, lib)
    wall = keepout("K1", [8.0, 0.0, 4.0, 8.0], forbids=["tracks"])  # the straight line y = 5 crosses it; the board is free below y = 8
    r = route_board(ir, lib, keepouts=[wall])
    pts = kg.area_points(wall)
    assert r.unrouted == {} and r.version == ROUTER_KEEPOUT_VERSION and r.stats["keepouts"][0]["id"] == "K1"
    assert all(kg.segment_area_distance(t.start, t.end, pts) >= t.width_mm / 2.0 - 1e-9 for t in r.tracks)  # no copper in the area
    assert all(t.provenance.tool_version == "0.4" and "keepouts:K1" in t.provenance.derived_from for t in r.tracks)
    straight = route_board(ir, lib, keepouts=[keepout("K1", [8.0, 0.0, 4.0, 8.0], forbids=["tracks"], nets=["N"])])
    assert straight.stats["total_length_mm"] < r.stats["total_length_mm"] and straight.stats["total_length_mm"] == pytest.approx(14.0)
    closed = route_board(ir, lib, keepouts=[keepout("K1", [8.0, 0.0, 4.0, 12.0], forbids=["tracks"])])
    assert closed.tracks == [] and "keep-out(s) K1 are obstacles" in closed.unrouted["N"]
    only_front = route_board(ir, lib, keepouts=[keepout("K1", [8.0, 0.0, 4.0, 12.0], forbids=["tracks"], layers=["F.Cu"])])
    assert only_front.unrouted == {} and {t.layer for t in only_front.tracks if kg.segment_area_distance(t.start, t.end, pts) < t.width_mm / 2} == {"B.Cu"}


def test_a_via_keepout_forbids_vias_except_for_its_allowed_net(tmp_path: Path, lib: KicadLibrary):
    # an SMD wall on F.Cu between two SMD pads forces the net onto B.Cu through two vias (tests/test_routing.py)
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 2.0, 5.0), ("W1", "WALL", 6.0, 5.0), ("R2", "SMD1", 10.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 10.0))
    free = route_board(ir, lib)
    assert free.unrouted == {} and len(free.vias) == 2
    banned = route_board(ir, lib, keepouts=[keepout("NOVIA", [0.0, 0.0, 12.0, 10.0], forbids=["vias"])])
    assert banned.vias == [] and banned.tracks == [] and "N" in banned.unrouted
    allowed = route_board(ir, lib, keepouts=[keepout("NOVIA", [0.0, 0.0, 12.0, 10.0], forbids=["vias"], nets=["N"])])
    assert allowed.unrouted == {} and len(allowed.vias) == 2 and allowed.version == ROUTER_KEEPOUT_VERSION


def _plane_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """A 4-layer board (GND on both inner planes): three SMD GND pads, a through-hole GND pad and a signal net ``S``."""
    ir = board_ir(tmp_path, lib, [("C1", "SMD1", 4.0, 4.0), ("C2", "SMD1", 10.0, 4.0), ("C3", "SMD1", 16.0, 4.0), ("R4", "PAD1", 10.0, 10.0),
                                  ("R5", "PAD1", 4.0, 14.0), ("R6", "PAD1", 16.0, 14.0)],
                  {"GND": [("C1", "1"), ("C2", "1"), ("C3", "1"), ("R4", "1")], "S": [("R5", "1"), ("R6", "1")]}, (20.0, 18.0))
    ir.nets[0].kind = NetKind.GROUND
    return stacked(ir, 4)


PLANE = [(0.5, 0.5), (19.5, 0.5), (19.5, 17.5), (0.5, 17.5)]


def test_a_plane_nets_smd_pads_get_a_via_each_and_no_track_joins_two_pads(tmp_path: Path, lib: KicadLibrary):
    ir = _plane_board(tmp_path, lib)
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]})
    assert r.unrouted == {} and r.version == ROUTER_KEEPOUT_VERSION
    row = r.stats["plane_nets"]["GND"]
    assert row["vias"] == 3 and row["pads"] == ["C1.1", "C2.1", "C3.1"] and row["through_hole"] == ["R4.1"] and row["planes"] == ["In1.Cu", "In2.Cu"]
    gnd = [t for t in r.tracks if t.net == "GND"]
    vias = [v for v in r.vias if v.net == "GND"]
    assert len(vias) == 3 and all(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) <= PLANE_VIA_REACH_MM + 1e-9 for t in gnd)
    pads = {(4.0, 4.0), (10.0, 4.0), (16.0, 4.0)}
    for v in vias:  # each via beside exactly one pad, joined to it by its own stub, whole inside the plane
        near = [p for p in pads if math.hypot(v.x_mm - p[0], v.y_mm - p[1]) <= PLANE_VIA_REACH_MM + 1e-9]
        assert len(near) == 1 and kg.point_in_polygon((v.x_mm, v.y_mm), PLANE)
    assert all(t.provenance.tool_version == "0.4" and "plane_nets:GND=In1.Cu+In2.Cu;reach=3.0" in t.provenance.derived_from for t in r.tracks)
    assert r.stats["routed_nets"] == 2 and [t.net for t in r.tracks if t.net == "S"]
    # the other net keeps the router's clearance from the plane vias and stubs (independent IR geometry check)
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=authoritative(0.25, DS, unit="mm"))
    ir.pcb.zones = plane_zones(ir.pcb.stackup, ir.pcb.outline, 0.5)
    got = _run("pcb.routing", ir, tmp_path, lib)
    assert got["pcb.routing.clearance"].status is S.PASS, got["pcb.routing.clearance"].message
    conn = got[CONNECTIVITY_CHECK]
    rows = {row["net"]: row for row in conn.details["nets"]}
    assert conn.status is S.NOT_VERIFIED and rows["S"]["status"] == "PASS" and rows["GND"]["status"] == "NOT_VERIFIED" and rows["GND"]["plane"] is True
    assert "connected only through a plane fill the IR does not measure" in rows["GND"]["message"]


def test_a_plane_pad_without_a_via_site_inside_its_plane_leaves_the_plane_net_unrouted(tmp_path: Path, lib: KicadLibrary):
    ir = _plane_board(tmp_path, lib)
    corner = [(12.0, 12.0), (19.5, 12.0), (19.5, 17.5), (12.0, 17.5)]  # a plane zone nowhere near C1 / C2 / C3
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", corner)]})
    assert "GND" in r.unrouted and "no legal via site" in r.unrouted["GND"] and not [t for t in r.tracks if t.net == "GND"]
    assert r.stats["plane_nets"]["GND"]["reason"].startswith("pad C1.1") and [t for t in r.tracks if t.net == "S"]
    with pytest.raises(CompileError, match="no plane zone given"):
        route_board(ir, lib, inner_layers=True, plane_nets={"GND": []})


def _big_pad_plane_board(tmp_path: Path, lib: KicadLibrary, fp: str) -> tuple[CircuitIR, list[tuple[float, float]]]:
    """A 4-layer GND/GND board: ``U1`` (``fp``, pad 2 on GND) and a small SMD GND pad ``C1``; the plane covers the board."""
    ir = board_ir(tmp_path, lib, [("U1", fp, 12.0, 12.0), ("C1", "SMD1", 26.0, 12.0)], {"GND": [("U1", "2"), ("C1", "1")]}, (32.0, 24.0))
    ir.nets[0].kind = NetKind.GROUND
    return stacked(ir, 4), [(0.5, 0.5), (31.5, 0.5), (31.5, 23.5), (0.5, 23.5)]


def test_a_large_plane_pad_gets_its_via_beyond_its_own_copper(tmp_path: Path, lib: KicadLibrary):
    """A TO-252 tab (6.4 x 5.8 mm, the LP38693DT's GND): the nearest legal via site is 3.3 / 3.6 mm from its centre, beyond the 3 mm reach
    counted from the terminal cell - the reach now counts from where a via disc clears the pad's own copper (it was 'no legal via site')."""
    ir, plane = _big_pad_plane_board(tmp_path, lib, "TAB")
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", plane), ("In2.Cu", plane)]})
    assert r.unrouted == {}, r.unrouted
    row = r.stats["plane_nets"]["GND"]
    assert row["pads"] == ["C1.1", "U1.2"] and row["vias"] == 2 and "joined_by_footprint_vias" not in row
    tab = (12.0 + 1.26, 12.0, 3.2, 2.9)  # centre and half extents of the tab on the board
    v = next(v for v in r.vias if math.hypot(v.x_mm - tab[0], v.y_mm - tab[1]) < 8.0)
    clear = max(abs(v.x_mm - tab[0]) - tab[2], abs(v.y_mm - tab[1]) - tab[3])
    assert 0.4 - 1e-9 <= clear <= PLANE_VIA_REACH_MM + 0.4 + 1e-9  # outside the tab (never via-in-pad), within the reach beyond its edge


def test_an_exposed_pad_on_the_footprints_own_thermal_vias_needs_no_via_of_its_own(tmp_path: Path, lib: KicadLibrary):
    """A QFN ThermalVias EP whose every axis leaves through a lead: the footprint's same-numbered through-hole pads join both SMD copies
    (F.Cu, B.Cu) to the plane, so the net routes (it was 'pad U1.2: no legal via site') and says how each pad reaches the plane."""
    ir, plane = _big_pad_plane_board(tmp_path, lib, "EPTV")
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", plane), ("In2.Cu", plane)]})
    assert r.unrouted == {}, r.unrouted
    row = r.stats["plane_nets"]["GND"]
    assert row["pads"] == ["C1.1"] and row["vias"] == 1 and row["joined_by_footprint_vias"] == ["U1.2"] and row["through_hole"] == ["U1.2"] * 4
    story = next(t for t in r.tracks if t.net == "GND").provenance.note
    assert "1 SMD pad(s) by their footprint's own same-numbered through-hole pads" in story
    # the connectivity check sees the same: every pad reaches the plane (a via or a barrel), connected only through the fill
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    ir.pcb.zones = plane_zones(ir.pcb.stackup, ir.pcb.outline, 0.5)
    rows = {row["net"]: row for row in _run("pcb.routing", ir, tmp_path, lib)[CONNECTIVITY_CHECK].details["nets"]}
    assert rows["GND"]["status"] == "NOT_VERIFIED" and rows["GND"]["plane"] is True


@needs_real_libs
def test_real_to252_tab_and_qfn_thermal_via_ep_route_on_a_gnd_plane_board(tmp_path: Path):
    """The transceiver's parts on the real KiCad 10.0.6 footprints: U101 LP38693DT-5.0 (``Package_TO_SOT_SMD:TO-252-2``, GND tab pad 2) and
    U901 MMZ09332BT1 (``Package_DFN_QFN:QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm_ThermalVias``, EP 13) on a 4-layer board with GND on In1 / In2."""
    from ai_eda.ir import BoardOutline, BoardSide, Component, LibraryRef, Net, PCBDesign, PinRef, Placement, ProjectMeta

    real = KicadLibrary()
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    for fp_lib, fp, pad in (("Package_TO_SOT_SMD", "TO-252-2", "2"), ("Package_DFN_QFN", "QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm_ThermalVias", "13")):
        ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
        ir.components += [Component(ref="U1", value="x", footprint=real.resolve_footprint(LibraryRef(library=fp_lib, name=fp)), provenance=prov),
                          Component(ref="C1", value="x", footprint=real.resolve_footprint(LibraryRef(library="Capacitor_SMD", name="C_0603_1608Metric")), provenance=prov)]
        assert all(c.footprint.verified for c in ir.components), fp
        ir.nets.append(Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="U1", pin_number=pad), PinRef(component_ref="C1", pin_number="2")], provenance=prov))
        ir.pcb = PCBDesign(outline=BoardOutline(width_mm=40.0, height_mm=30.0), placements=[
            Placement(component_ref="U1", x_mm=15.0, y_mm=15.0, side=BoardSide.TOP, provenance=prov),
            Placement(component_ref="C1", x_mm=32.0, y_mm=15.0, side=BoardSide.TOP, provenance=prov)])
        stacked(ir, 4)
        plane = [(0.5, 0.5), (39.5, 0.5), (39.5, 29.5), (0.5, 29.5)]
        from ai_eda.tools.routing.maze import RoutingParams

        r = route_board(ir, real, RoutingParams.for_board(ir, real), inner_layers=True, plane_nets={"GND": [("In1.Cu", plane), ("In2.Cu", plane)]})
        assert r.unrouted == {}, (fp, r.unrouted)
        row = r.stats["plane_nets"]["GND"]
        assert f"U1.{pad}" in row["pads"] + row.get("joined_by_footprint_vias", []), (fp, row)


def test_connectivity_fails_an_smd_pad_no_plane_can_reach(tmp_path: Path, lib: KicadLibrary):
    ir = _plane_board(tmp_path, lib)
    ir.pcb.zones = plane_zones(ir.pcb.stackup, ir.pcb.outline, 0.5)
    conn = _run("pcb.routing", ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    rows = {row["net"]: row for row in conn.details["nets"]}
    # no copper at all: the SMD pads C1..C3 are on F.Cu only and no via joins them to the In1 / In2 plane - a real disconnection
    assert conn.status is S.FAIL and rows["GND"]["status"] == "FAIL" and rows["GND"]["unconnected"] == ["C1.1", "C2.1", "C3.1"]
    assert "only a via or a through-hole barrel reaches an inner plane" in rows["GND"]["message"]


# --------------------------------------------------------------------------- SI: a keep-out that clears the plane


def _feed_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """``FEED`` (an RF net in class Z50) routed straight on F.Cu over the GND plane of a 4-layer board, 20 mm long, at the class's width."""
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", 25.0, 5.0)], {"FEED": [("R1", "1"), ("R2", "1")]}, (30.0, 10.0))
    ir.nets[0].kind = NetKind.RF
    stacked(ir, 4)
    si = si_of(*default_classes())
    z50 = si.net_class("Z50")
    si.net_classes = [c for c in si.net_classes if c.name != "Z50"] + [z50.model_copy(update={"nets": ["FEED"]})]
    ir.si = si
    from ai_eda.tools.si.rules import controlled_width

    width = controlled_width(ir, 50.0)[0]
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.pcb.tracks = [Track(net="FEED", layer="F.Cu", start=(5.0, 5.0), end=(15.0, 5.0), width_mm=width, provenance=prov),
                     Track(net="FEED", layer="F.Cu", start=(15.0, 5.0), end=(25.0, 5.0), width_mm=width, provenance=prov)]
    return ir


def test_copper_over_a_plane_cleared_by_a_zone_keepout_has_no_reference_plane(tmp_path: Path, lib: KicadLibrary):
    ir = _feed_board(tmp_path, lib)
    before = {r.check_id: r for r in si_results(ir, library=lib)}
    assert before["si.impedance.Z50"].status is S.PASS  # over the plane everywhere
    base = measure_nets(ir, library=lib)["FEED"]
    with_keepouts(ir, [keepout("ANT", [18.0, 0.0, 12.0, 10.0], forbids=["zones"], layers=["In1.Cu", "In2.Cu"], reason="antenna band")])
    m = measure_nets(ir, library=lib)["FEED"]
    seg = m.segments[0]
    assert seg.keepout_mm == pytest.approx(10.0) and seg.keepout_reason.startswith(f"{KEEPOUT_NO_PLANE} ANT: it forbids zones on In1.Cu (the GND plane)")
    assert m.bound and not base.bound and m.delay_s > base.delay_s and m.line.bound and m.line.delay_s > base.line.delay_s
    after = {r.check_id: r for r in si_results(ir, library=lib)}
    imp = after["si.impedance.Z50"]
    rows = imp.details["segments"]
    assert imp.status is S.NOT_VERIFIED and [r["status"] for r in rows] == ["PASS", "NOT_VERIFIED"]
    assert rows[1]["reason"].startswith(f"{KEEPOUT_NO_PLANE} ANT") and "antenna band" in rows[1]["reason"] and "add a plane" not in rows[1]["reason"]
    rf = default_registry.get("domain.rf.impedance").validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))[0]
    assert rf.status is S.NOT_VERIFIED and "keep-out ANT" in rf.message  # the critic's false PASS over a cleared plane is closed
    # a keep-out that allows the plane's net, or one on a layer that is no reference plane of the track, changes nothing
    for k in (keepout("ANT", [18.0, 0.0, 12.0, 10.0], forbids=["zones"], nets=["GND"]), keepout("ANT", [18.0, 0.0, 12.0, 10.0], forbids=["zones"], layers=["B.Cu"])):
        with_keepouts(ir, [k])
        assert measure_nets(ir, library=lib)["FEED"].summary() == base.summary()
        assert {r.check_id: r for r in si_results(ir, library=lib)}["si.impedance.Z50"].status is S.PASS


def _long_under_keepout(tmp_path: Path, lib: KicadLibrary, ers: tuple[float, float, float] | None = None, length: float = 90.0) -> CircuitIR:
    """``LONG`` (a driver -> receiver line, ``length`` mm on F.Cu) on the 4-layer stack (dielectric er ``ers`` if given), over keep-out ANT
    that forbids zones on In1 / In2 under part of it."""
    ir = long_board(tmp_path, lib, 4, length=length)
    if ers is not None:
        st = ir.pcb.stackup
        ir.pcb.stackup = st.model_copy(update={"dielectrics": [d.model_copy(update={"er": d.er.model_copy(update={"value": er})})
                                                               for d, er in zip(st.dielectrics, ers)]})
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [si_track("LONG", (5.0, 5.0), (5.0 + length, 5.0)), si_track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    return with_keepouts(ir, [keepout("ANT", [40.0, 0.0, 10.0, 10.0], forbids=["zones"], layers=["In1.Cu", "In2.Cu"], reason="antenna band")])


def test_a_keepout_cleared_plane_bounds_the_delay_by_the_largest_er_of_the_stack(tmp_path: Path, lib: KicadLibrary):
    """With the In1 plane cleared the field of an F.Cu line reaches the core: on a 3.0 / 10.2 / 3.0 stack sqrt(3.0)/c0 is no upper bound.
    The bound is sqrt(er_max)/c0 (as the via barrel's), named; an 84 mm line under the keep-out is no longer 'short' (l_crit 86.5 mm by the
    prepreg's er made it a false PASS; by er_max it is 46.9 mm, not provably short)."""
    ir = _long_under_keepout(tmp_path, lib, ers=(3.0, 10.2, 3.0), length=84.0)
    with_keepouts(ir, [keepout("K", [0.0, 0.0, 94.0, 10.0], forbids=["zones"], layers=["In1.Cu"], reason="cleared under the line")])
    seg = measure_nets(ir, library=lib)["LONG"].segments[0]
    assert seg.keepout_mm == pytest.approx(84.0) and seg.keepout_t_pd == pytest.approx(propagation_delay(10.2), rel=1e-12)
    assert seg.keepout_id == "K" and "pcb.stackup.dielectrics[1].er = 10.2" in seg.keepout_bound
    crit = {r.check_id: r for r in si_results(ir, library=lib)}["si.critical_length"]
    row = next(r for r in crit.details["nets"] if r["net"] == "LONG")
    assert row["status"] == "NOT_VERIFIED" and row["l_crit_mm"] == pytest.approx(0.5e-9 / propagation_delay(10.2) * 1e3, rel=1e-6), row
    assert row["l_crit_mm"] == pytest.approx(46.93, abs=0.01) and "keep-out K" in row["reason"]
    # a board whose stack is uniform gives the same bound as before (the layer's own dielectric: every er is 4.5)
    uniform = _long_under_keepout(tmp_path, lib)
    assert measure_nets(uniform, library=lib)["LONG"].segments[0].keepout_t_pd == pytest.approx(propagation_delay(4.5), rel=1e-12)


def test_every_no_plane_verdict_names_the_keepout_never_pcb_layers_4(tmp_path: Path, lib: KicadLibrary):
    """On a 4-layer board whose plane a keep-out clears, spice.si, si.critical_length and si.rf_length name the keep-out; the remedy
    'use pcb_layers=4 or add a plane' (wrong: the board has its planes) is given only for a layer the stack gives no plane."""
    ir = _long_under_keepout(tmp_path, lib)
    sp = {r.check_id: r for r in spice_si_results(ir, {"kicad_library": lib}, tmp_path)}["spice.si.LONG"]
    assert sp.status is S.NOT_VERIFIED and "under keep-out ANT" in sp.message and "pcb_layers=4" not in sp.message, sp.message
    assert sp.details["keepouts"] == ["ANT"]
    crit = {r.check_id: r for r in si_results(ir, library=lib)}["si.critical_length"]
    row = next(r for r in crit.details["nets"] if r["net"] == "LONG")
    assert crit.status is S.NOT_VERIFIED and "under keep-out ANT" in row["reason"] and "pcb_layers=4" not in row["reason"] and row["keepouts"] == ["ANT"]
    assert "LONG (keep-out ANT)" in crit.message and "pcb_layers=4" not in crit.message and crit.details["long_under_keepout"] == ["LONG"], crit.message
    # si.rf_length: an RF feed at 2.4 GHz whose plane the keep-out clears
    feed = _feed_board(tmp_path, lib)
    z50 = feed.si.net_class("Z50")
    feed.si = feed.si.model_copy(update={"net_classes": [c for c in feed.si.net_classes if c.name != "Z50"] + [z50.model_copy(update={"rf_frequency_hz": user_requirement(2.4e9, "Hz")})],
                                         "rf_length_fraction": user_requirement(0.1)})
    with_keepouts(feed, [keepout("ANT", [18.0, 0.0, 12.0, 10.0], forbids=["zones"], layers=["In1.Cu", "In2.Cu"], reason="antenna band")])
    rfl = {r.check_id: r for r in si_results(feed, library=lib)}["si.rf_length"]
    rrow = next(r for r in rfl.details["nets"] if r["net"] == "FEED")
    assert "under keep-out ANT" in rrow["reason"] and "pcb_layers=4" not in rrow["reason"] and rfl.status is S.NOT_VERIFIED, rrow
    # the plane-less layer still gets its remedy (a 2-layer board: no keep-out, no plane)
    two = long_board(tmp_path, lib, 2)
    two.si = si_of(*default_classes())
    two.pcb.tracks = [si_track("LONG", (5.0, 5.0), (95.0, 5.0)), si_track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    assert "use pcb_layers=4 or add a plane" in {r.check_id: r for r in spice_si_results(two, {"kicad_library": lib}, tmp_path)}["spice.si.LONG"].message


# --------------------------------------------------------------------------- the compiler: KiCad rule areas


def _compiled_rule_areas(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> tuple[list, str]:
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    text = Path(art.path).read_text(encoding="utf-8")
    board = sexpr.parse(text)
    return [z for z in sexpr.find_all(board, "zone") if sexpr.find(z, "keepout") is not None], text


def _area_of(zone) -> list[tuple[float, float]]:
    return [(float(p[1]), float(p[2])) for p in sexpr.find_all(sexpr.find(sexpr.find(zone, "polygon"), "pts"), "xy")]


def test_keepouts_compile_to_rule_areas_with_the_exemptions_cut_out_deterministically(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD2", 3.0, 3.0), ("R2", "SMD2", 12.0, 3.0), ("R3", "PAD2", 3.0, 10.0)],
                  {"N": [("R1", "2"), ("R2", "1")], "K": [("R3", "1"), ("R3", "2")]}, (16.0, 14.0))
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.pcb.tracks = [Track(net="N", layer="F.Cu", start=(4.5, 3.0), end=(10.5, 3.0), width_mm=0.4, provenance=prov)]
    with_keepouts(ir, [
        keepout("BAND", [0.0, 7.0, 16.0, 7.0], forbids=["tracks", "vias", "zones"], layers=["F.Cu"]),
        keepout("FEED", [5.0, 1.0, 4.0, 4.0], forbids=["tracks", "footprints"], nets=["N"]),
    ])
    zones, text = _compiled_rule_areas(ir, tmp_path / "a", lib)
    names = [str(sexpr.find(z, "name")[1]) for z in zones]
    # FEED's tracks ban has N's track channel cut across it: two pieces; its footprints ban (no exception there) is one area
    assert names == ["keepout_BAND", "keepout_FEED_1", "keepout_FEED_1_2", "keepout_FEED_2"] and all(sexpr.find(z, "net") is None for z in zones)
    band = zones[0]
    assert [str(x) for x in sexpr.find(band, "layers")[1:]] == ["F.Cu"]
    assert {str(item[0]): str(item[1]) for item in sexpr.find(band, "keepout")[1:]} == {
        "tracks": "not_allowed", "vias": "not_allowed", "pads": "allowed", "copperpour": "not_allowed", "footprints": "allowed"}
    assert kg.area_bbox(_area_of(band)) == (0.0, 7.0, 16.0, 14.0)
    # FEED: the tracks ban has N's track cut out (its box + the margin), the footprints ban is written whole (nothing is exempt there)
    by = {str(sexpr.find(z, "name")[1]): z for z in zones}
    tracks_zone, fp_zone = by["keepout_FEED_1"], by["keepout_FEED_2"]
    assert {str(i[0]): str(i[1]) for i in sexpr.find(tracks_zone, "keepout")[1:]}["tracks"] == "not_allowed"
    assert {str(i[0]): str(i[1]) for i in sexpr.find(fp_zone, "keepout")[1:]}["footprints"] == "not_allowed"
    assert [str(x) for x in sexpr.find(fp_zone, "layers")[1:]] == ["F.Cu", "B.Cu"]
    cut = kg.area_points(keepout("x", [5.0, 1.0, 4.0, 4.0], forbids=["tracks"]))
    track_box = (4.5 - 0.3, 3.0 - 0.3, 10.5 + 0.3, 3.0 + 0.3)
    pieces = [z for z in zones if str(sexpr.find(z, "name")[1]).startswith("keepout_FEED_1")]
    assert pieces and all(kg.box_area_overlap(track_box, _area_of(z)) == 0.0 for z in pieces)
    assert sum(abs(kg.polygon_area(_area_of(z))) for z in pieces) == pytest.approx(abs(kg.polygon_area(cut)) - 4.0 * 0.6)
    _, again = _compiled_rule_areas(ir, tmp_path / "b", lib)
    assert again == text  # deterministic: the same IR gives the same bytes


def test_a_non_convex_polygon_keepout_with_exceptions_is_refused_by_the_compiler(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD2", 3.0, 3.0)], {}, (16.0, 14.0))
    u = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [7.0, 10.0], [7.0, 3.0], [3.0, 3.0], [3.0, 10.0], [0.0, 10.0]]
    with_keepouts(ir, [keepout("U", polygon=u, forbids=["footprints"], refs=["R1"])])
    with pytest.raises(CompileError, match="non-convex polygon with allowed refs"):
        PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    with_keepouts(ir, [keepout("U", polygon=u, forbids=["footprints"])])  # without exceptions the polygon is written as it is
    zones, _ = _compiled_rule_areas(ir, tmp_path / "ok", lib)
    assert len(zones) == 1 and _area_of(zones[0]) == [tuple(p) for p in u]


# --------------------------------------------------------------------------- pcb.keepout


def test_pcb_keepout_judges_every_forbidden_item_and_names_the_exceptions(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "SMD2", 3.0, 3.0), ("R2", "SMD2", 12.0, 3.0), ("R3", "PAD2", 3.0, 10.0)],
                  {"N": [("R1", "2"), ("R2", "1")], "K": [("R3", "1"), ("R3", "2")]}, (16.0, 14.0))
    assert KEEPOUT_CHECK not in [v.id for v in default_registry.select(ir)]  # no keep-out, no RF block: no row at all
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.pcb.tracks = [Track(net="N", layer="F.Cu", start=(4.5, 3.0), end=(10.5, 3.0), width_mm=0.4, provenance=prov)]
    with_keepouts(ir, [keepout("FEED", [5.0, 1.0, 4.0, 4.0], forbids=["tracks", "footprints"], nets=["N"])])
    assert KEEPOUT_CHECK in [v.id for v in default_registry.select(ir)]
    ok = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert ok.status is S.PASS and ok.details["keepouts"][0]["exempt"] == ["track[0:N]"] and "not DRC" in ok.message
    with_keepouts(ir, [keepout("FEED", [5.0, 1.0, 4.0, 4.0], forbids=["tracks", "footprints"])])
    bad = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert bad.status is S.FAIL and [r["item"] for r in bad.details["rows"]] == ["track[0:N]"]
    with_keepouts(ir, [keepout("PARTS", [0.0, 8.0, 7.0, 5.0], forbids=["footprints", "pads"])])  # R3 (PAD2 at (3, 10)) sits in it
    parts = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert parts.status is S.FAIL and {r["item"] for r in parts.details["rows"]} == {"footprint R3", "pad R3.1", "pad R3.2"}
    with_keepouts(ir, [keepout("PARTS", [0.0, 8.0, 7.0, 5.0], forbids=["footprints", "pads"], refs=["R3"])])
    assert _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK].status is S.PASS
    ir.pcb.zones = [Zone(net="K", layer="B.Cu", polygon=[(0.0, 0.0), (16.0, 0.0), (16.0, 14.0), (0.0, 14.0)], provenance=prov)]
    with_keepouts(ir, [keepout("POUR", [10.0, 8.0, 4.0, 4.0], forbids=["zones"], layers=["B.Cu"])])
    pour = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert pour.status is S.FAIL and "covers 16.000 mm^2" in pour.message
    assert _run(KEEPOUT_CHECK, ir, tmp_path, None)[KEEPOUT_CHECK].status is S.FAIL  # the zone needs no library: its FAIL stands
    with_keepouts(ir, [keepout("PARTS", [0.0, 8.0, 7.0, 5.0], forbids=["footprints", "pads"], refs=["R3"])])
    blind = _run(KEEPOUT_CHECK, ir, tmp_path, None)[KEEPOUT_CHECK]
    assert blind.status is S.NOT_VERIFIED and "no KiCad library" in blind.message  # pads and extents unknown: never a PASS


# --------------------------------------------------------------------------- placement.rf_floorplan


def _rf_ir(tmp_path: Path, lib: KicadLibrary, size: tuple[float, float] = (60.0, 50.0)) -> CircuitIR:
    parts = [("SH1", "CAN10"), *[(f"C{i}", "SMDB") for i in range(1, 7)], ("J1", "PAD2"), ("R1", "SMDB"), ("R2", "SMDB"), ("RV1", "BIG")]
    ir = unplaced(tmp_path, lib, parts, size=size)
    return with_rf(ir, [
        block("fe", ["SH1", "C1", "C2", "C3", "C4", "C5", "C6"], chain=["C3", "C1", "C2"], shield="SH1", region=(0.0, 0.0, 20.0, 20.0)),
        block("io", ["J1", "R1", "R2"], chain=["J1", "R2", "R1"], region=(20.0, 0.0, 40.0, 20.0)),
    ])


def test_the_floorplan_packs_each_block_in_its_region_in_chain_order_the_can_first_and_its_parts_inside_the_fence(tmp_path: Path, lib: KicadLibrary):
    ir = with_keepouts(_rf_ir(tmp_path, lib), [keepout("band", [0.0, 40.0, 60.0, 10.0], forbids=["footprints", "tracks"])])
    got = rf_floorplan_placement(ir, lib, outline=ir.pcb.outline)
    at = {p.component_ref: p for p in got.placements}
    ext = got.extents
    assert got.outline == ir.pcb.outline and got.block_of["C3"] == "fe" and got.block_of["J1"] == "io" and got.block_of["RV1"] == ""
    assert all(p.provenance.tool == PLACER_ID and p.provenance.kind == ProvenanceKind.DERIVED for p in got.placements)
    assert all(_inside(ext[r], (0.0, 0.0, 20.0, 20.0)) for r in ("SH1", "C1", "C2", "C3", "C4", "C5", "C6"))
    assert all(_inside(ext[r], (20.0, 0.0, 60.0, 20.0)) for r in ("J1", "R1", "R2"))
    # the can is centred in its region (less the margin / spacing) and its parts sit inside its fence less the ring
    fence = fence_box(lib.load_footprint(ir.component("SH1").footprint))
    sh = at["SH1"]
    inner = (sh.x_mm + fence.x1 + RING_MM, sh.y_mm + fence.y1 + RING_MM, sh.x_mm + fence.x2 - RING_MM, sh.y_mm + fence.y2 - RING_MM)
    assert all(_inside(ext[f"C{i}"], inner) for i in range(1, 7)) and "inside_can:SH1,ring_mm:1.0" in at["C1"].provenance.derived_from
    # chain order: C3, C1, C2 lead, row by row; J1 -> R2 -> R1 left to right
    order = sorted(("C1", "C2", "C3", "C4", "C5", "C6"), key=lambda r: (ext[r].y1, ext[r].x1))
    assert order[:3] == ["C3", "C1", "C2"]
    assert ext["J1"].x2 < ext["R2"].x1 < ext["R1"].x1
    # the part outside the regions packs around them and keeps out of the band that bans footprints
    assert ext["RV1"].y1 >= 20.0 and ext["RV1"].y2 <= 40.0
    assert got.keepouts == ["band"] and "keep-outs band honoured" in got.description("user", 1.0, 2.0, RING_MM)
    again = rf_floorplan_placement(_rf_ir(tmp_path / "again", lib), lib, outline=ir.pcb.outline, keepouts=list(ir.pcb.keepouts))
    assert [design_data(p) for p in again.placements] == [design_data(p) for p in got.placements]


def test_the_floorplan_places_a_board_the_grid_placer_cannot(tmp_path: Path, lib: KicadLibrary):
    parts = [("RV1", "BIG"), *[(f"C{i}", "SMDB") for i in range(1, 41)]]
    ir = unplaced(tmp_path, lib, parts, size=(50.0, 60.0))
    with pytest.raises(CompileError, match="do not fit inside"):  # one 15 x 14 mm cell per part, 4 columns
        grid_placement(ir, lib, outline=ir.pcb.outline)
    ir = with_rf(ir, [block("audio", [r for r, _ in parts], chain=["RV1"], region=(0.0, 0.0, 50.0, 60.0))])
    got = rf_floorplan_placement(ir, lib, outline=ir.pcb.outline)
    assert len(got.placements) == 41 and all(_inside(e, (0.0, 0.0, 50.0, 60.0)) for e in got.extents.values())


def test_the_floorplan_refuses_instead_of_overlapping(tmp_path: Path, lib: KicadLibrary):
    small = with_rf(unplaced(tmp_path, lib, [("RV1", "BIG")], size=(60.0, 50.0)), [block("pot", ["RV1"], region=(0.0, 0.0, 12.0, 30.0))])
    with pytest.raises(CompileError, match="larger than the box it belongs in"):
        rf_floorplan_placement(small, lib, outline=small.pcb.outline)
    tight = with_rf(unplaced(tmp_path, lib, [("SH1", "CAN10")], size=(60.0, 50.0)), [block("fe", ["SH1"], shield="SH1", region=(0.0, 0.0, 12.0, 12.0))])
    with pytest.raises(CompileError, match="does not fit its region"):
        rf_floorplan_placement(tight, lib, outline=tight.pcb.outline)
    full = with_rf(unplaced(tmp_path, lib, [("SH1", "CAN10"), *[(f"C{i}", "SMDB") for i in range(1, 20)]], size=(60.0, 50.0)),
                   [block("fe", ["SH1", *[f"C{i}" for i in range(1, 20)]], shield="SH1", region=(0.0, 0.0, 20.0, 20.0))])
    with pytest.raises(CompileError, match="inside the fence of SH1"):
        rf_floorplan_placement(full, lib, outline=full.pcb.outline)
    ir = _rf_ir(tmp_path, lib)
    outside = with_rf(ir, [block("fe", ["SH1"], shield="SH1", region=(50.0, 0.0, 20.0, 20.0))])
    with pytest.raises(CompileError, match="leaves the 60.0 x 50.0 mm outline"):
        rf_floorplan_placement(outside, lib, outline=outside.pcb.outline)
    overlap = with_rf(ir, [block("a", ["J1"], region=(0.0, 0.0, 20.0, 20.0)), block("b", ["R1"], region=(10.0, 10.0, 20.0, 20.0))])
    with pytest.raises(CompileError, match="overlap"):
        rf_floorplan_placement(overlap, lib, outline=overlap.pcb.outline)
    unknown = with_rf(ir, [block("a", ["X9"], region=(0.0, 0.0, 20.0, 20.0))])
    with pytest.raises(CompileError, match="not a component"):
        rf_floorplan_placement(unknown, lib, outline=unknown.pcb.outline)
    # a shielded block without a region: its can would be shelf-packed beside its own parts (pcb.keepout then FAILed them) - refused
    no_region = with_rf(unplaced(tmp_path, lib, [("SH1", "CAN10"), ("C1", "SMDB"), ("C2", "SMDB"), ("J1", "PAD2")], size=(60.0, 50.0)),
                        [block("fe", ["SH1", "C1", "C2"], shield="SH1"), block("io", ["J1"], region=(40.0, 0.0, 20.0, 20.0))])
    with pytest.raises(CompileError, match="block fe: shield can SH1 but no region - a shielded block needs a region"):
        rf_floorplan_placement(no_region, lib, outline=no_region.pcb.outline)


def test_without_an_outline_the_regions_size_the_board(tmp_path: Path, lib: KicadLibrary):
    ir = with_rf(unplaced(tmp_path, lib, [("J1", "PAD2"), ("R1", "SMDB")], size=None), [block("io", ["J1", "R1"], chain=["J1", "R1"], region=(0.0, 0.0, 30.0, 12.0))])
    got = rf_floorplan_placement(ir, lib)
    assert (got.outline.width_mm, got.outline.height_mm, got.outline.origin_x_mm, got.outline.origin_y_mm) == (30.0, 12.0, 0.0, 0.0)


def test_pcb_keepout_checks_regions_and_can_fences(tmp_path: Path, lib: KicadLibrary):
    ir = _rf_ir(tmp_path, lib)
    got = rf_floorplan_placement(ir, lib, outline=ir.pcb.outline)
    ir.pcb.placements = list(got.placements)
    ok = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert ok.status is S.PASS, ok.message
    fence_row = next(r for r in ok.details["regions"] if r.get("can") == "SH1")
    assert fence_row["ring_mm"] >= RING_MM - 1e-6
    moved = [p.model_copy(update={"x_mm": p.x_mm + 30.0}) if p.component_ref == "C1" else p for p in ir.pcb.placements]
    ir.pcb.placements = moved
    bad = _run(KEEPOUT_CHECK, ir, tmp_path, lib)[KEEPOUT_CHECK]
    assert bad.status is S.FAIL and {r["item"] for r in bad.details["rows"]} == {"block fe: C1", "can SH1: C1"}


# --------------------------------------------------------------------------- the PCB agent


def _rf_board_for_agent(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """Two blocks on a 4-layer board (GND planes) with a top band that forbids parts, tracks and zones except the feed."""
    parts = [("J1", "PAD2"), ("C1", "SMDB"), ("C2", "SMDB"), ("R1", "SMDB"), ("R2", "SMDB")]
    nets = {"GND": [("C1", "1"), ("C2", "1"), ("R1", "1"), ("J1", "1")], "A": [("C1", "2"), ("R1", "2")], "B": [("C2", "2"), ("R2", "2")],
            "FEED": [("R2", "1"), ("J1", "2")]}
    ir = unplaced(tmp_path, lib, parts, nets, size=(30.0, 30.0))
    ir.nets[0].kind = NetKind.GROUND
    stacked(ir, 4)
    ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    ir = with_rf(ir, [block("rf", ["J1", "R2", "C2"], chain=["J1", "R2", "C2"], region=(0.0, 8.0, 30.0, 10.0)),
                      block("dc", ["C1", "R1"], chain=["C1", "R1"], region=(0.0, 18.0, 30.0, 12.0))])
    return with_keepouts(ir, [keepout("ANT", [0.0, 0.0, 30.0, 6.0], forbids=["footprints", "tracks", "zones"], nets=["FEED"], reason="antenna band")])


def test_the_agent_places_rf_blocks_by_the_floorplan_and_routes_with_keepouts_and_plane_vias(tmp_path: Path, lib: KicadLibrary):
    ir = _rf_board_for_agent(tmp_path, lib)
    before = ir.content_hash()
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert ir.content_hash() == before and [p.target for p in res.proposals] == ["pcb"]
    notes = " | ".join(res.notes)
    board = res.proposals[0].payload
    assert all(p.provenance.tool == PLACER_ID for p in board.placements) and "placement.rf_floorplan 0.1" in notes
    assert board.tracks and all(t.provenance.tool_version == ROUTER_KEEPOUT_VERSION for t in board.tracks), notes
    assert "keep-outs ANT are obstacles" in notes and "plane net(s) GND" in notes
    band = kg.area_points(keepout("x", [0.0, 0.0, 30.0, 6.0], forbids=["tracks"]))
    assert all(kg.segment_area_distance(t.start, t.end, band) >= t.width_mm / 2.0 - 1e-9 for t in board.tracks if t.net != "FEED")
    gnd = [t for t in board.tracks if t.net == "GND"]
    assert gnd and all(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) <= PLANE_VIA_REACH_MM + 1e-9 for t in gnd)
    planes = [z for z in board.zones if z.provenance.tool == STACKUP_TOOL]
    assert {(z.net, z.layer) for z in planes} == {("GND", "In1.Cu"), ("GND", "In2.Cu")}
    assert all(kg.area_bbox([tuple(p) for p in z.polygon])[1] >= 6.0 for z in planes)  # clipped below the band
    assert all("pcb.keepouts[ANT]" in z.provenance.derived_from for z in planes) and "plane zone GND on In1.Cu clipped by keep-out(s) ANT" in notes
    again = PCBAgent().run(_rf_board_for_agent(tmp_path / "again", lib), AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert design_data(again.proposals[0].payload) == design_data(res.proposals[0].payload)


def test_the_agent_leaves_a_board_without_rf_blocks_or_keepouts_to_the_existing_placers(tmp_path: Path, lib: KicadLibrary):
    ir = unplaced(tmp_path, lib, [("R1", "SMDB"), ("R2", "SMDB")], {"N": [("R1", "1"), ("R2", "1")]}, size=None)
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    board = res.proposals[0].payload
    assert {p.provenance.tool for p in board.placements} == {"placement.grid"} and {t.provenance.tool_version for t in board.tracks} == {ROUTER_VERSION}
