"""``placement.fixed``: the design's fixed parts and pack regions, a template proposing the board layout, and a template's silk texts.

Offline tests run on the synthetic fixture library of ``tests/test_routing.py``
(``Test:PAD2`` a two-pad through-hole part, extent 4.6 x 2.0 mm; ``Test:SMD2``
a two-pad SMD part, extent 4.0 x 2.0 mm - both courtyards 2 x 2 mm with the
pads reaching beyond them) and on the ``Test_Silk`` library of
``tests/test_silkscreen.py`` (a chip resistor with its own silk and
``Reference``, a 1x3 header). What is proved:

* the placer copies the fixed parts verbatim (position, rotation, side,
  provenance), packs each region's parts inside its rectangle in the region's
  own list order (rows, left to right) around them, packs the other parts in
  the free board area, keeps clear of keep-outs (a bottom-side fixed part
  judged by the keep-outs of its own side, as the ``pcb.keepout`` check
  judges it) and of the silk texts already on the board, and refuses -
  naming the parts - every case its module docstring lists;
* the PCB agent uses it for a board with fixed parts or pack regions (and
  routes the result), never for any other board, and combines it with each
  placement answer as its module docstring says;
* a template plan's ``outline`` / ``fixed`` / ``pack_regions`` /
  ``silk_texts`` reach ``ir.pcb`` through ``add_board`` with a table line each
  (appended to an existing board without replacing anything), and a plan
  without them proposes the board it proposed before;
* a template's silk texts survive the silkscreen stage: they stay where they
  are, the references, pin labels and title are placed around them, the
  ``pcb.silk.*`` checks pass and the compiled board carries them on the silk.

One test needs ``kicad-cli`` and the installed KiCad 10 libraries (it skips
elsewhere): three ``SW_PUSH_6mm`` keys fixed in a row with their note names,
the rest packed in two regions, routed, silk-placed, compiled and checked by
real DRC.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai_eda.agents.pcb as pcb_agent
from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.keys import PLACEMENT_KEY, PLACEMENT_POLICY_KEY, ROUTING_KEY, SILKSCREEN_KEY
from ai_eda.agents.pcb import FIXED_RATIONALE
from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.design.base import DesignChange, Plan, structural_provenance
from ai_eda.design.board import add_board, has_layout, layout_lines, template_silk_text
from ai_eda.design.library_parts import library_component
from ai_eda.design.templates import TEMPLATES
from ai_eda.errors import CompileError
from ai_eda.ir import (
    BoardOutline,
    BoardSide,
    CircuitIR,
    Keepout,
    Net,
    PackRegion,
    PCBDesign,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SilkKind,
    SilkText,
    assumption,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.ir.rf import RFBlock, RFDesign
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.geometry import footprint_bbox
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.fixed import FIXED, FREE_AREA, PLACER_ID, PLACER_VERSION, fixed_placement
from ai_eda.tools.placement.grid import MARGIN_MM, SPACING_MM, _disjoint, _inside, footprint_extent
from ai_eda.tools.silkscreen.geometry import ir_text_box
from ai_eda.tools.silkscreen.place import PLACER_KEEP_VERSION, place_silkscreen
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import CONNECTIVITY_CHECK, KEEPOUT_CHECK, SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK
from ai_eda.workflow import Orchestrator
from tests.test_parts_existence import make_part
from tests.test_routing import fixture_library
from tests.test_silkscreen import _checks as silk_checks
from tests.test_silkscreen import _part as silk_part
from tests.test_silkscreen import silk_library

ANSWERS = {"application": "test", "jurisdiction": "EU"}
#: who fixed the parts in these fixtures: a template's structural decision
LAYOUT = structural_provenance("keys_fixture", "fixture layout")
NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
MM = "mm"
#: the fixture extents (tests/test_routing.py footprints: courtyard -1..1, pads at +-1.5): PAD2 4.6 x 2.0 mm, SMD2 4.0 x 2.0 mm
PAD2_W, SMD2_W, PART_H = 4.6, 4.0, 2.0


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _region(rid: str, rect: list[float], refs: list[str], side: BoardSide = BoardSide.TOP) -> PackRegion:
    return PackRegion(id=rid, rect=assumption(rect, "fixture region", MM), refs=refs, side=side, provenance=LAYOUT)


def keys_ir(tmp_path: Path, lib: KicadLibrary, *, size: tuple[float, float] = (40.0, 30.0), nets: bool = False) -> CircuitIR:
    """S1..S3 (PAD2) fixed in a row along the lower edge, R1..R5 (SMD2) in region ``chain`` in the order R5 R3 R1 R2 R4, Q1..Q4 (SMD2) in
    region ``osc`` (three per row), C1 (SMD2) in neither."""
    ir = CircuitIR(project=ProjectMeta(id="keys", name="keys", workdir=str(tmp_path)))
    for ref, fp in [*((f"S{i}", "PAD2") for i in (1, 2, 3)), *((f"R{i}", "SMD2") for i in range(1, 6)), *((f"Q{i}", "SMD2") for i in range(1, 5)),
                    ("C1", "SMD2")]:
        c = make_part(ref, footprint=fp)
        c.symbol = lib.resolve_symbol(c.symbol)
        c.footprint = lib.resolve_footprint(c.footprint)
        assert c.footprint.verified, fp
        ir.components.append(c)
    if nets:  # each key's pad 2 to its chain resistor, the resistors' pads 2 to C1, the keys' pads 1 to a common rail
        for i in (1, 2, 3):
            ir.nets.append(Net(name=f"T{i}", pins=[PinRef(component_ref=f"S{i}", pin_number="2"), PinRef(component_ref=f"R{i}", pin_number="1")], provenance=NET_P))
        ir.nets.append(Net(name="OUT", pins=[*(PinRef(component_ref=f"R{i}", pin_number="2") for i in (1, 2, 3)), PinRef(component_ref="C1", pin_number="1")],
                           provenance=NET_P))
        ir.nets.append(Net(name="VCC", pins=[*(PinRef(component_ref=f"S{i}", pin_number="1") for i in (1, 2, 3)), PinRef(component_ref="C1", pin_number="2")],
                           provenance=NET_P))
    fixed = [Placement(component_ref=f"S{i}", x_mm=8.0 + 10.0 * (i - 1), y_mm=25.0, provenance=LAYOUT) for i in (1, 2, 3)]
    regions = [_region("chain", [2.0, 12.0, 36.0, 8.0], ["R5", "R3", "R1", "R2", "R4"]), _region("osc", [2.0, 2.0, 14.0, 8.0], ["Q3", "Q1", "Q2", "Q4"])]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=size[0], height_mm=size[1]), fixed=fixed, pack_regions=regions)
    return ir


def _row(p: Placement) -> int:
    return int(next(e for e in p.provenance.derived_from if e.startswith("row:"))[4:])


def _ctx(tmp_path: Path, lib: KicadLibrary, **answers: str) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={**ANSWERS, **answers})


def _agent(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, **answers: str):
    before = ir.content_hash()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **answers))
    assert ir.content_hash() == before and res.questions == [] and not res.blocked_on_user  # proposes, never mutates, never asks
    return res


# --------------------------------------------------------------------------- the placer


def test_fixed_parts_are_kept_and_each_region_is_packed_in_its_list_order(tmp_path: Path, lib: KicadLibrary):
    ir = keys_ir(tmp_path, lib)
    got = fixed_placement(ir, lib)
    by = {p.component_ref: p for p in got.placements}
    # every component once, in natural ref order
    assert [p.component_ref for p in got.placements] == ["C1", "Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3"]
    # the fixed parts exactly as the design gave them (position, rotation, side, provenance)
    for given in ir.pcb.fixed:
        assert design_data(by[given.component_ref]) == design_data(given) and by[given.component_ref].provenance == LAYOUT
        assert got.region_of[given.component_ref] == FIXED
    # region chain: one row from the region's top-left corner (the margin is inside it already), in the list order R5 R3 R1 R2 R4
    chain = ["R5", "R3", "R1", "R2", "R4"]
    assert [(by[r].x_mm, by[r].y_mm) for r in chain] == [(2.0 + SMD2_W / 2 + i * (SMD2_W + SPACING_MM), 12.0 + PART_H / 2) for i in range(5)]
    # region osc (14 mm wide): three per row in list order, the fourth starts the next row below the tallest part + the spacing
    osc = ["Q3", "Q1", "Q2", "Q4"]
    assert [(_row(by[r]), by[r].x_mm, by[r].y_mm) for r in osc] == [
        (0, 4.0, 3.0), (0, 4.0 + SMD2_W + SPACING_MM, 3.0), (0, 4.0 + 2 * (SMD2_W + SPACING_MM), 3.0), (1, 4.0, 3.0 + PART_H + SPACING_MM)]
    assert all(got.region_of[r] == "chain" for r in chain) and all(got.region_of[r] == "osc" for r in osc) and got.region_of["C1"] == FREE_AREA
    # every extent is the library footprint's at its placement, inside its region / the outline, pairwise apart
    for p in got.placements:
        assert got.extents[p.component_ref] == footprint_bbox(p, lib.load_footprint(ir.component(p.component_ref).footprint))
        assert _inside(got.extents[p.component_ref], ir.pcb.outline) and p.rotation_deg == 0.0 and p.side == BoardSide.TOP
    for region in ir.pcb.pack_regions:
        x1, y1, x2, y2 = region.box()
        for ref in region.refs:
            e = got.extents[ref]
            assert x1 <= e.x1 and e.x2 <= x2 and y1 <= e.y1 and e.y2 <= y2, (ref, e)
    refs = list(got.extents)
    assert all(_disjoint(got.extents[a], got.extents[b]) for i, a in enumerate(refs) for b in refs[i + 1:])
    # C1, in no region: the free board area, every region an obstacle, margin from the edge
    c1 = got.extents["C1"]
    assert all(not (r.box()[0] < c1.x2 and c1.x1 < r.box()[2] and r.box()[1] < c1.y2 and c1.y1 < r.box()[3]) for r in ir.pcb.pack_regions)
    assert MARGIN_MM <= c1.x1 and MARGIN_MM <= c1.y1
    # provenance of a packed part: this tool, the region, the row, the fixed parts it was packed around, the parameters
    prov = by["R5"].provenance
    assert prov.kind == ProvenanceKind.DERIVED and (prov.tool, prov.tool_version) == (PLACER_ID, PLACER_VERSION) == ("placement.fixed", "0.1")
    assert prov.derived_from == ["footprint:Test:SMD2", "region:chain:2,12,36,8", "row:0", "rotation:0", "around_fixed:S1,S2,S3",
                                 f"spacing_mm:{SPACING_MM}", f"margin_mm:{MARGIN_MM}"] and prov.inputs == {}
    assert "region:(free board area)" in by["C1"].provenance.derived_from
    assert got.description("user", SPACING_MM, MARGIN_MM) == (
        "placement.fixed 0.1: 13 component(s) on a 40.0 x 30.0 mm user outline at (0.0, 0.0), 3 fixed part(s) where the design put them, "
        "2 pack region(s) (chain 5, osc 4) packed in their list order, 1 part(s) in the free board area in natural ref order; shelf packing by "
        "each part's own extent around the fixed parts, spacing 1.0 mm, margin 2.0 mm")
    # deterministic: the same IR and another library instance give the same placements
    again = fixed_placement(keys_ir(tmp_path, fixture_library(tmp_path / "kicad2")), fixture_library(tmp_path / "kicad3"))
    assert [design_data(p) for p in again.placements] == [design_data(p) for p in got.placements]


def test_a_rotated_fixed_part_and_a_part_that_fits_only_turned(tmp_path: Path, lib: KicadLibrary):
    """A fixed part keeps its rotation (its extent measured turned); a region too narrow for a part's width packs it turned 90 degrees."""
    ir = keys_ir(tmp_path, lib)
    turned = ir.pcb.fixed[0].model_copy(update={"rotation_deg": 90.0})
    ir.pcb = ir.pcb.model_copy(update={"fixed": [turned, *ir.pcb.fixed[1:]], "pack_regions": [_region("slot", [34.0, 2.0, 3.0, 8.0], ["C1"]), *ir.pcb.pack_regions]})
    got = fixed_placement(ir, lib)
    by = {p.component_ref: p for p in got.placements}
    assert by["S1"].rotation_deg == 90.0 and (got.extents["S1"].width, got.extents["S1"].height) == pytest.approx((PART_H, PAD2_W))
    assert by["C1"].rotation_deg == 90.0 and got.region_of["C1"] == "slot" and got.extents["C1"].width == pytest.approx(PART_H)


def test_a_region_is_packed_around_a_fixed_part_from_the_margin_and_the_free_parts_in_natural_ref_order(tmp_path: Path, lib: KicadLibrary):
    """C1 fixed inside region ``chain``, which reaches the board edge: the region's parts start at the margin, the one after the first
    skips past C1 by its extent + the spacing; Q1..Q4 (in no region) fill the free board area in natural ref order. Two regions over
    the same rectangle are packed in their list order, the second around the first."""
    ir = keys_ir(tmp_path, lib)
    c1 = Placement(component_ref="C1", x_mm=12.0, y_mm=12.0, provenance=LAYOUT)  # extent (10, 11)-(14, 13)
    ir.pcb = ir.pcb.model_copy(update={"fixed": [*ir.pcb.fixed, c1], "pack_regions": [_region("chain", [0.0, 10.0, 40.0, 10.0], ["R5", "R3", "R1", "R2", "R4"])]})
    got = fixed_placement(ir, lib)
    by = {p.component_ref: p for p in got.placements}
    assert got.extents["R5"].x1 == MARGIN_MM  # the region begins at the board edge: its parts keep the margin
    assert got.extents["R3"].x1 == got.extents["C1"].x2 + SPACING_MM  # past the fixed part by its extent + the spacing
    assert [(by[r].x_mm, by[r].y_mm) for r in ["R5", "R3", "R1", "R2", "R4"]] == [(4.0, 11.0), (17.0, 11.0), (22.0, 11.0), (27.0, 11.0), (32.0, 11.0)]
    assert [(by[q].x_mm, by[q].y_mm, got.region_of[q]) for q in ("Q1", "Q2", "Q3", "Q4")] == [
        (4.0 + i * (SMD2_W + SPACING_MM), 3.0, FREE_AREA) for i in range(4)]
    assert design_data(by["C1"]) == design_data(c1) and got.region_of["C1"] == FIXED
    # two regions over one rectangle: the first's parts first, in its order; the second's after them, in its order
    ir = keys_ir(tmp_path, lib)
    same = [2.0, 2.0, 20.0, 4.0]
    ir.pcb = ir.pcb.model_copy(update={"pack_regions": [_region("first", same, ["Q3", "Q1"]), _region("second", same, ["Q2", "Q4"]), ir.pcb.pack_regions[0]]})
    by = {p.component_ref: p for p in fixed_placement(ir, lib).placements}
    assert [by[q].x_mm for q in ("Q3", "Q1", "Q2", "Q4")] == [4.0 + i * (SMD2_W + SPACING_MM) for i in range(4)]


def test_keepouts_and_the_silk_texts_on_the_board_are_kept_clear(tmp_path: Path, lib: KicadLibrary):
    ir = keys_ir(tmp_path, lib)
    plain = {p.component_ref: p for p in fixed_placement(ir, lib).placements}
    # a keep-out banning footprints where R5 went, R3 allowed in it; a template's text where C1 went
    ko = Keepout(id="hole", layers=["*.Cu"], rect=assumption([2.0, 12.0, 4.5, 2.5], "fixture", MM), forbids=["footprints"], allowed_refs=["R3"],
                 reason="fixture", provenance=LAYOUT)
    text = SilkText(text="NOTE", x_mm=plain["C1"].x_mm, y_mm=plain["C1"].y_mm, kind=SilkKind.USER, provenance=LAYOUT)
    ir.pcb = ir.pcb.model_copy(update={"keepouts": [ko], "silkscreen": [text]})
    got = fixed_placement(ir, lib)
    by = {p.component_ref: p for p in got.placements}
    assert by["R5"].x_mm > 6.5  # moved right past the keep-out, still first in the row
    assert [r for r in ["R5", "R3", "R1", "R2", "R4"]] == sorted(["R5", "R3", "R1", "R2", "R4"], key=lambda r: (_row(by[r]), by[r].x_mm))
    tb = ir_text_box(text).bbox()
    e = got.extents["C1"]
    assert e.x1 >= tb[2] + SPACING_MM - 1e-6 or e.x2 <= tb[0] - SPACING_MM + 1e-6 or e.y1 >= tb[3] + SPACING_MM - 1e-6 or e.y2 <= tb[1] - SPACING_MM + 1e-6
    assert got.keepouts == ["hole"] and got.texts == 1
    assert got.description("user", SPACING_MM, MARGIN_MM).endswith("; 1 silk text(s) on the board kept clear; keep-outs hole honoured")
    # a text on a fab layer (a reference drawn on the part's assembly layer) is no silk text: nothing is kept clear of it
    fab = SilkText(text="C1", x_mm=plain["C1"].x_mm, y_mm=plain["C1"].y_mm, kind=SilkKind.REFERENCE, component_ref="C1", layer="F.Fab", provenance=LAYOUT)
    ir.pcb = ir.pcb.model_copy(update={"keepouts": [], "silkscreen": [fab]})
    got = fixed_placement(ir, lib)
    assert [design_data(p) for p in got.placements] == [design_data(p) for p in plain.values()] and got.texts == 0


def _refused(ir: CircuitIR, lib: KicadLibrary, match: str) -> None:
    with pytest.raises(CompileError, match=match):
        fixed_placement(ir, lib)


def test_a_bottom_side_fixed_part_is_judged_by_the_keepouts_of_its_own_side(tmp_path: Path, lib: KicadLibrary):
    """C1 (SMD2: pads on F.Cu in the library) fixed on the bottom side under a keep-out: its footprint is on B.Cu and so are its pads.
    The placer bans it exactly where the ``pcb.keepout`` check fails it - never for a ban on the front copper only."""
    def board(layers: list[str], forbids: list[str], allowed: list[str] | None = None) -> CircuitIR:
        ir = keys_ir(tmp_path, lib)
        under = Placement(component_ref="C1", x_mm=30.0, y_mm=5.0, side=BoardSide.BOTTOM, provenance=LAYOUT)
        ko = Keepout(id="under", layers=layers, rect=assumption([28.0, 4.0, 4.0, 2.0], "fixture", MM), forbids=forbids, allowed_refs=allowed or [],
                     reason="fixture")
        ir.pcb = ir.pcb.model_copy(update={"fixed": [*ir.pcb.fixed, under], "keepouts": [ko]})
        return ir

    def keepout_check(ir: CircuitIR, placements: list[Placement]) -> S:
        placed = ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"placements": placements})})
        [result] = default_registry.get(KEEPOUT_CHECK).validate(placed, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
        return result.status

    for forbids in (["footprints"], ["pads"]):
        front = board(["F.Cu"], forbids)  # the front copper only: C1 on the back is not in it
        got = fixed_placement(front, lib)
        assert design_data(next(p for p in got.placements if p.component_ref == "C1")) == design_data(front.pcb.fixed[-1])
        assert keepout_check(front, got.placements) is S.PASS
        back = board(["B.Cu"], forbids)  # the back copper: C1's footprint and pads are there
        what = "footprints on B.Cu, the bottom side" if forbids == ["footprints"] else "pads on B.Cu"
        _refused(back, lib, rf"placed extent of 'C1' .* lies in a keep-out that bans it \(under: {what}\)")
        assert keepout_check(back, got.placements) is S.FAIL  # the check fails what the placer refuses
        allowed = board(["B.Cu"], forbids, ["C1"])  # a keep-out that allows C1 by name does not ban it
        assert [design_data(p) for p in fixed_placement(allowed, lib).placements] == [design_data(p) for p in got.placements]


def test_each_refusal_names_what_cannot_be_placed(tmp_path: Path, lib: KicadLibrary):
    def board(**update) -> CircuitIR:
        ir = keys_ir(tmp_path, lib)
        ir.pcb = ir.pcb.model_copy(update=update)  # model_copy skips the IR model's own checks: the placer does not rely on them
        return ir

    fixed = keys_ir(tmp_path, lib).pcb.fixed
    _refused(board(outline=None), lib, r"no board outline \(ir.pcb.outline\): placement.fixed needs one")
    _refused(board(fixed=[], pack_regions=[]), lib, "placement.fixed has nothing to keep")
    _refused(board(fixed=[*fixed[:2], fixed[2].model_copy(update={"x_mm": 39.0})]), lib, r"fixed part\(s\) \['S3'\] lie outside the 40.0 x 30.0 mm outline")
    _refused(board(fixed=[fixed[0], fixed[1].model_copy(update={"x_mm": 11.0}), fixed[2]]), lib, r"fixed parts overlap: S1 .* and S2 ")
    _refused(board(fixed=[*fixed, Placement(component_ref="X9", x_mm=20.0, y_mm=20.0)]), lib, r"fixed part\(s\) \['X9'\] are not components of the IR")
    _refused(board(pack_regions=[_region("chain", [2.0, 12.0, 36.0, 8.0], ["R5", "X9"])]), lib, r"pack region chain names \['X9'\], which are not components")
    _refused(board(pack_regions=[_region("chain", [30.0, 12.0, 36.0, 8.0], ["R5"])]), lib, r"pack region chain \(30, 12\)-\(66, 20\) leaves the 40.0 x 30.0 mm outline")
    _refused(board(pack_regions=[_region("chain", [2.0, 12.0, 12.0, 3.0], ["R5", "R3", "R1"])]), lib, r"pack region chain \(12 x 3 mm at \(2, 12\)\): R1 .*does not fit")
    _refused(board(pack_regions=[_region("chain", [2.0, 12.0, 36.0, 8.0], ["R5"], BoardSide.BOTTOM)]), lib, "packs the top side only")
    _refused(board(fixed=[*fixed, fixed[0]]), lib, "S1 is fixed twice")
    _refused(board(pack_regions=[_region("chain", [2.0, 12.0, 36.0, 8.0], ["R5", "S1"])]), lib, "S1 is both fixed and in pack region chain")
    _refused(board(pack_regions=[_region("a", [2.0, 12.0, 36.0, 8.0], ["R5"]), _region("b", [2.0, 2.0, 14.0, 8.0], ["R5"])]), lib,
             "R5 is both in pack region a and in pack region b")
    # a component in neither list that finds no room in the free board area (a region holds the whole board) ends up nowhere: refused,
    # never dropped or overlapped
    _refused(board(pack_regions=[_region("chain", [0.0, 0.0, 40.0, 30.0], ["R5", "R3", "R1", "R2", "R4", "Q1", "Q2", "Q3", "Q4"])]), lib,
             r"the free board area \(the parts neither fixed nor in a pack region\): C1 .*(does not fit|fits no row)")
    # a fixed part in a keep-out that bans it: the guard refuses (the design contradicts itself)
    ko = Keepout(id="band", layers=["*.Cu"], rect=assumption([0.0, 23.0, 12.0, 4.0], "fixture", MM), forbids=["footprints"], reason="fixture")
    _refused(board(keepouts=[ko]), lib, r"placed extent of 'S1' .* lies in a keep-out that bans it")
    # an IR with RF blocks too: neither placer reads the other's layout
    rf = keys_ir(tmp_path, lib)
    rf.rf = RFDesign(blocks=[RFBlock(id="b", refs=["C1"])])
    _refused(rf, lib, r"the IR has RF blocks \(ir.rf.blocks\) and fixed parts / pack regions")
    # a component without a footprint: refused like every placer refuses it
    ir = keys_ir(tmp_path, lib)
    ir.component("C1").footprint = None
    _refused(ir, lib, "'C1' has no footprint")


# --------------------------------------------------------------------------- the PCB agent


def test_the_agent_places_a_board_with_fixed_parts_by_this_placer_and_routes_it(tmp_path: Path, lib: KicadLibrary):
    ir = keys_ir(tmp_path, lib, nets=True)
    res = _agent(ir, tmp_path, lib, **{SILKSCREEN_KEY: "skip"})
    [proposal] = res.proposals
    board = proposal.payload
    want = fixed_placement(ir, lib)
    assert [design_data(p) for p in board.placements] == [design_data(p) for p in want.placements]
    assert board.fixed == ir.pcb.fixed and board.pack_regions == ir.pcb.pack_regions and board.outline == ir.pcb.outline  # the design's layout stays
    assert res.notes[0] == want.description("user", SPACING_MM, MARGIN_MM) and proposal.description.startswith(res.notes[0])
    assert proposal.rationale.startswith(FIXED_RATIONALE)
    # routed: every net connected through IR copper
    assert board.tracks and any(n.startswith("routing.maze") for n in res.notes), res.notes
    Orchestrator.apply_proposals(ir, res.proposals)
    checks = {r.check_id: r for r in default_registry.get("pcb.routing").validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))}
    assert checks[CONNECTIVITY_CHECK].status is S.PASS, checks[CONNECTIVITY_CHECK].message


def test_the_placement_answers_keep_their_meaning_on_a_board_with_fixed_parts(tmp_path: Path, lib: KicadLibrary):
    ir = keys_ir(tmp_path, lib)
    skip = {ROUTING_KEY: "skip", SILKSCREEN_KEY: "skip"}
    res = _agent(ir, tmp_path, lib, **{PLACEMENT_KEY: "skip"})
    assert res.proposals == [] and res.notes == ["placement skipped by answer"]
    # functional: its placer refuses fixed parts and pack regions (its meaning before this placer existed)
    res = _agent(ir, tmp_path, lib, **skip, **{PLACEMENT_POLICY_KEY: "functional"})
    assert res.proposals == [] and res.notes[-1].startswith("not placed: functional: fixed parts / pack_regions require placement.fixed")
    # legacy: the grid / core-ring / RF placers read neither - refused with that reason, never placed without the design's positions
    res = _agent(ir, tmp_path, lib, **skip, **{PLACEMENT_POLICY_KEY: " Legacy "})
    assert res.proposals == [] and res.notes == [
        "not placed: pcb.placement_policy='legacy' places by the grid / core-ring / RF placers, which do not read the 3 fixed part(s) and 2 pack "
        "region(s) in ir.pcb; placement.fixed places them - leave pcb.placement_policy unanswered"]
    # unanswered: placement.fixed, routing skipped as asked
    res = _agent(ir, tmp_path, lib, **skip)
    assert res.notes[0].startswith("placement.fixed 0.1: 13 component(s)") and res.proposals[0].payload.tracks == []
    # a refusal of the placer is the agent's "not placed:" note, nothing proposed
    bad = keys_ir(tmp_path, lib)
    bad.pcb = bad.pcb.model_copy(update={"outline": None})
    res = _agent(bad, tmp_path, lib, **skip)
    assert res.proposals == [] and res.notes[-1].startswith("not placed: no board outline (ir.pcb.outline): placement.fixed needs one")
    # RF blocks too: placement.fixed refuses (the RF floorplan reads no fixed part), never another placer in its place
    rf = keys_ir(tmp_path, lib)
    rf.rf = RFDesign(blocks=[RFBlock(id="b", refs=["C1"])])
    res = _agent(rf, tmp_path, lib, **skip)
    assert res.proposals == [] and res.notes[-1].startswith("not placed: the IR has RF blocks (ir.rf.blocks) and fixed parts / pack regions")


def test_every_other_board_never_reaches_this_placer(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    def fail(*a, **kw):
        raise AssertionError("a board without fixed parts or pack regions reached placement.fixed")

    monkeypatch.setattr(pcb_agent, "fixed_placement", fail)
    ir = keys_ir(tmp_path, lib)
    ir.pcb = PCBDesign()
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip", SILKSCREEN_KEY: "skip"})
    assert {p.provenance.tool for p in res.proposals[0].payload.placements} == {"placement.grid"}


# --------------------------------------------------------------------------- a template proposes the layout


def _divider():
    return next(t for t in TEMPLATES if t.id == "divider")


def _layout_plan(**layout) -> Plan:
    plan = Plan(template="divider", **layout)
    plan.changes.append(DesignChange(description="x", target="topology", operation="set", payload=None))
    return plan


def _layout(tid: str = "divider") -> dict:
    prov = structural_provenance(tid, "key row")
    return {
        "outline": BoardOutline(width_mm=40.0, height_mm=30.0),
        "fixed": [Placement(component_ref="R1", x_mm=10.0, y_mm=25.0, provenance=prov), Placement(component_ref="R2", x_mm=20.0, y_mm=25.0, provenance=prov)],
        "pack_regions": [PackRegion(id="top", rect=assumption([2.0, 2.0, 36.0, 10.0], "fixture", MM), refs=["J1"], provenance=prov)],
        "silk_texts": [SilkText(text="C4", x_mm=10.0, y_mm=28.0, kind=SilkKind.USER, component_ref="R1", provenance=prov),
                       SilkText(text="KEYS", x_mm=3.0, y_mm=20.0, justify="left", kind=SilkKind.TITLE, provenance=prov)],
    }


def _apply(ir: CircuitIR, plan: Plan) -> None:
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale)
                                      for c in plan.changes])


def test_a_template_plan_proposes_its_layout_with_the_board_and_lists_it_in_the_table(tmp_path: Path):
    divider = _divider()
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    layout = _layout()
    without, with_layout = _layout_plan(), _layout_plan(**layout)
    assert not has_layout(without) and has_layout(with_layout) and layout_lines(without) == []
    assert add_board(divider, ir, without, confirmed=True) is None and add_board(divider, ir, with_layout, confirmed=True) is None
    board = lambda p: next(c for c in p.changes if c.target == "pcb")  # noqa: E731
    # without a layout: the board as before - the stack, its layers and the (empty) keep-outs only, no layout line in the table
    assert board(without).payload.model_fields_set == {"layers", "stackup", "keepouts"} and board(without).rationale == "template divider: generic 2-layer stack"
    assert [c.target for c in without.changes] == [c.target for c in with_layout.changes]
    assert design_data(board(with_layout).payload.model_copy(update={"outline": None, "fixed": [], "pack_regions": [], "silkscreen": []})) == design_data(board(without).payload)
    # with one: into the proposed board, and one table line each
    pcb = board(with_layout).payload
    assert (pcb.outline, pcb.fixed, pcb.pack_regions, pcb.silkscreen) == (layout["outline"], layout["fixed"], layout["pack_regions"], layout["silk_texts"])
    assert board(with_layout).rationale == "template divider: generic 2-layer stack; board layout: outline, 2 fixed part(s), 1 pack region(s), 2 silk text(s)"
    lines = [
        "board outline: 40 x 30 mm at (0, 0) mm",
        "fixed part R1: at (10, 25) mm, rotation 0 deg, top side",
        "fixed part R2: at (20, 25) mm, rotation 0 deg, top side",
        "pack region top: rect x 2 y 2 w 36 h 10 mm, top side, packed in this order: J1",
        "silk text 'C4' (user): at (10, 28) mm on F.SilkS, 1 mm, beside R1",
        "silk text 'KEYS' (title): at (3, 20) mm on F.SilkS, 1 mm",
    ]
    assert layout_lines(with_layout) == lines and all(line in with_layout.board for line in lines) and not any(line in without.board for line in lines)
    table = with_layout.table()
    assert all(f"  {line}" in table for line in lines) and "board outline" not in without.table()
    # an outline alone is a layout too: the board gets it, the table its line
    outline_only = _layout_plan(outline=layout["outline"])
    assert has_layout(outline_only) and add_board(divider, ir, outline_only, confirmed=True) is None
    assert board(outline_only).payload.outline == layout["outline"] and lines[0] in outline_only.board
    # applied through the orchestrator like every plan
    _apply(ir, with_layout)
    assert ir.pcb.fixed == layout["fixed"] and ir.pcb.silkscreen == layout["silk_texts"] and all(template_silk_text(t) for t in ir.pcb.silkscreen)


def test_a_template_layout_is_appended_to_an_existing_board_and_never_replaces_anything(tmp_path: Path):
    divider = _divider()
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    ir.pcb = PCBDesign()
    layout = _layout()
    plan = _layout_plan(**layout)
    assert add_board(divider, ir, plan, confirmed=True) is None
    assert [(c.target, c.operation) for c in plan.changes if c.target.startswith("pcb.") and c.target not in ("pcb.layers", "pcb.stackup")] == [
        ("pcb.outline", "set"), ("pcb.fixed", "append"), ("pcb.fixed", "append"), ("pcb.pack_regions", "append"), ("pcb.silkscreen", "append"),
        ("pcb.silkscreen", "append")]
    _apply(ir, plan)
    assert ir.pcb.outline == layout["outline"] and ir.pcb.fixed == layout["fixed"] and ir.pcb.pack_regions == layout["pack_regions"]
    assert ir.pcb.silkscreen == layout["silk_texts"] and ir.pcb.stackup is not None
    # the same layout again: the outline is never replaced
    assert "ir.pcb already has an outline (40 x 30 mm); the template never replaces it" == add_board(divider, ir, _layout_plan(**_layout()), confirmed=True)
    no_outline = {k: v for k, v in _layout().items() if k != "outline"}
    assert add_board(divider, ir, _layout_plan(**no_outline), confirmed=True) == (
        "part(s) R1 (fixed), R2 (fixed), J1 (in pack region top) already in ir.pcb; the template never replaces a fixed part or a pack region")
    only = lambda **kw: _layout_plan(**kw)  # noqa: E731
    region = PackRegion(id="top", rect=assumption([2.0, 2.0, 5.0, 5.0], "fixture", MM), refs=["C1"], provenance=layout["fixed"][0].provenance)
    assert add_board(divider, ir, only(pack_regions=[region]), confirmed=True) == "pack region(s) ['top'] already exist in ir.pcb.pack_regions; the template never replaces one"
    assert add_board(divider, ir, only(silk_texts=[layout["silk_texts"][0]]), confirmed=True) == (
        "silk text(s) ['C4'] already exist in ir.pcb.silkscreen; the template never doubles a text")
    placed = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    placed.pcb = PCBDesign(placements=[Placement(component_ref="R9", x_mm=1.0, y_mm=1.0)])
    assert add_board(divider, placed, only(fixed=layout["fixed"]), confirmed=True) == (
        "ir.pcb already has 1 placement(s): the PCB agent never re-places a board, so the template's fixed parts and pack regions would never be read")


def test_a_template_layout_the_board_could_not_keep_is_refused(tmp_path: Path):
    divider = _divider()
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    layout = _layout()
    # a text without the template's structural provenance (the PCB agent would not keep it), another template's, a reference text
    hand = SilkText(text="C4", x_mm=1.0, y_mm=1.0, kind=SilkKind.USER, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn"))
    other = hand.model_copy(update={"provenance": structural_provenance("astable", "x")})
    ref = hand.model_copy(update={"kind": SilkKind.REFERENCE, "component_ref": "R1", "text": "R1", "provenance": structural_provenance("divider", "x")})
    for text in (hand, other, ref):
        why = add_board(divider, ir, _layout_plan(silk_texts=[text]), confirmed=True)
        assert why is not None and why.startswith(f"silk text {text.text!r} ({text.kind}") and "its structural provenance (design.template.divider)" in why
    # a layout the IR model refuses
    twice = _layout_plan(fixed=[layout["fixed"][0], layout["fixed"][0]])
    assert add_board(divider, ir, twice, confirmed=True) == "the template's board layout is refused: Value error, a part is fixed twice: ['R1']"
    assert twice.board == [] and twice.choices == []  # refused before the plan was touched


# --------------------------------------------------------------------------- a template's silk texts and the silkscreen stage


def _silk_board(tmp_path: Path, lib: KicadLibrary, *, title: bool = False) -> CircuitIR:
    """R1..R3 (Test_Silk:R) fixed in a row with a note name under each (a template's user texts), J1 (a 1x3 header) in a region."""
    ir = CircuitIR(project=ProjectMeta(id="silk", name="silk", workdir=str(tmp_path)))
    for ref in ("R1", "R2", "R3", "J1"):
        ir.components.append(silk_part(lib, ref))
    pins = {"VIN": [("J1", "1"), ("R1", "1")], "VMID": [("J1", "2"), ("R2", "1")], "GND": [("J1", "3"), ("R3", "1")],
            "OUT": [("R1", "2"), ("R2", "2"), ("R3", "2")]}
    for name, members in pins.items():
        ir.nets.append(Net(name=name, pins=[PinRef(component_ref=r, pin_number=n) for r, n in members], provenance=NET_P))
    fixed = [Placement(component_ref=f"R{i}", x_mm=6.0 + 8.0 * (i - 1), y_mm=14.0, provenance=LAYOUT) for i in (1, 2, 3)]
    texts = [SilkText(text=n, x_mm=p.x_mm, y_mm=17.0, kind=SilkKind.USER, component_ref=p.component_ref, provenance=LAYOUT)
             for n, p in zip(("C4", "D4", "E4"), fixed)]
    if title:
        texts.append(SilkText(text="PIANO", x_mm=1.0, y_mm=1.5, size_mm=1.5, thickness_mm=0.225, justify="left", kind=SilkKind.TITLE, provenance=LAYOUT))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0, height_mm=20.0), fixed=fixed,
                       pack_regions=[_region("power", [18.0, 1.0, 11.0, 10.0], ["J1"])], silkscreen=texts)
    return ir


def test_template_silk_texts_survive_the_silkscreen_stage_and_the_rest_is_placed_around_them(tmp_path: Path):
    lib = silk_library(tmp_path / "kicad")
    ir = _silk_board(tmp_path, lib)
    kept = list(ir.pcb.silkscreen)
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip"})
    [proposal] = res.proposals
    silk = proposal.payload.silkscreen
    # the template's texts first, exactly as they were; then a reference for every part, the pin labels and the project title
    assert [design_data(t) for t in silk[:3]] == [design_data(t) for t in kept]
    placed = silk[3:]
    assert sorted(t.component_ref for t in placed if t.kind == SilkKind.REFERENCE and t.layer == "F.SilkS") == ["J1", "R1", "R2", "R3"]
    assert {t.text for t in placed if t.kind == SilkKind.PIN_LABEL} == {"VIN", "VMID", "GND"}
    assert [t.text for t in placed if t.kind == SilkKind.TITLE] == ["silk"]
    assert all((t.provenance.tool, t.provenance.tool_version) == ("silkscreen.place", PLACER_KEEP_VERSION) and "kept_texts:3" in t.provenance.derived_from
               for t in placed)
    assert res.notes[-1].startswith("silkscreen.place 0.2: 3 template text(s) kept where the template put them, 4 reference(s) on the silkscreen, ")
    # nothing overlaps: the pcb.silk checks judge every text, the template's included
    Orchestrator.apply_proposals(ir, res.proposals)
    checks = silk_checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].status is S.PASS and checks[SILK_OVERLAP_CHECK].status is S.PASS, [c.message for c in checks.values()]
    assert checks[SILK_CLEARANCE_CHECK].details["designed_texts"] == len(silk)
    # the circuit report says the template's texts were kept (a line only boards with such texts get)
    from ai_eda.report.stages import _silkscreen_section

    section = _silkscreen_section(ir)
    assert ("- 템플릿이 보드와 함께 제안한 문자 3개 ('C4', 'D4', 'E4'): 템플릿이 둔 자리 그대로입니다. 실크 배치기(`silkscreen.place` 0.2)는 이 문자들을 "
            "옮기지 않고 금지 영역으로 삼아 나머지 문자를 그 둘레에 놓았습니다.") in section
    plain = _silk_board(tmp_path, lib)
    plain.pcb = plain.pcb.model_copy(update={"silkscreen": [t.model_copy(update={"provenance": NET_P}) for t in plain.pcb.silkscreen],
                                             "placements": ir.pcb.placements})
    assert not any("템플릿이 보드와 함께" in line for line in _silkscreen_section(plain))
    # the compiled board carries the note names on the silk layer, in IR order with the other board texts
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    tree = sexpr.parse(Path(art.path).read_text(encoding="utf-8"))
    texts = [(str(n[1]), sexpr.get(n, "layer")) for n in sexpr.find_all(tree, "gr_text")]
    assert texts[:3] == [("C4", "F.SilkS"), ("D4", "F.SilkS"), ("E4", "F.SilkS")]
    assert [t for t, _ in texts] == [t.text for t in ir.pcb.silkscreen if t.kind != SilkKind.REFERENCE]


def test_a_template_text_is_a_keepout_for_the_texts_the_placer_adds(tmp_path: Path):
    """A note name where R1's reference would go and a text where the title would go: both stay, and the reference and the title
    move to their next free positions - never onto the template's texts (the pcb.silk checks pass)."""
    lib = silk_library(tmp_path / "kicad")
    bare = _silk_board(tmp_path, lib)
    bare.pcb = bare.pcb.model_copy(update={"silkscreen": []})
    free = _agent(bare, tmp_path, lib, **{ROUTING_KEY: "skip"}).proposals[0].payload.silkscreen  # where they go on an empty silk
    ref = next(t for t in free if t.kind == SilkKind.REFERENCE and t.component_ref == "R1")
    title = next(t for t in free if t.kind == SilkKind.TITLE)
    texts = [SilkText(text="C4", x_mm=ref.x_mm, y_mm=ref.y_mm, size_mm=ref.size_mm, thickness_mm=ref.thickness_mm, kind=SilkKind.USER,
                      component_ref="R1", provenance=LAYOUT),
             SilkText(text="TUNE", x_mm=title.x_mm, y_mm=title.y_mm, size_mm=title.size_mm, thickness_mm=title.thickness_mm, justify=title.justify,
                      kind=SilkKind.USER, provenance=LAYOUT)]
    ir = _silk_board(tmp_path, lib)
    ir.pcb = ir.pcb.model_copy(update={"silkscreen": texts})
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip"})
    silk = res.proposals[0].payload.silkscreen
    assert silk[:2] == texts
    moved_ref = next(t for t in silk if t.kind == SilkKind.REFERENCE and t.component_ref == "R1")
    moved_title = next(t for t in silk if t.kind == SilkKind.TITLE)
    assert (moved_ref.x_mm, moved_ref.y_mm) != (ref.x_mm, ref.y_mm) and (moved_title.x_mm, moved_title.y_mm) != (title.x_mm, title.y_mm)
    Orchestrator.apply_proposals(ir, res.proposals)
    checks = silk_checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].status is S.PASS and checks[SILK_OVERLAP_CHECK].status is S.PASS, [c.message for c in checks.values()]


def test_a_template_title_is_the_board_title_and_other_texts_still_stop_the_placer(tmp_path: Path):
    lib = silk_library(tmp_path / "kicad")
    ir = _silk_board(tmp_path, lib, title=True)
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip"})
    silk = res.proposals[0].payload.silkscreen
    assert [t.text for t in silk if t.kind == SilkKind.TITLE] == ["PIANO"]  # no second title
    assert any(n.startswith("silkscreen.place 0.2: 4 template text(s) kept") and n.endswith("title 'PIANO' kept where the template put it") for n in res.notes)
    assert "silkscreen title: the template's title 'PIANO' is kept where the template put it; no second title placed" in res.notes
    # a text nobody stamped as a template's (drawn by hand) beside them: the silkscreen is designed already and is never replaced
    hand = _silk_board(tmp_path, lib)
    hand.pcb.silkscreen.append(SilkText(text="MY BOARD", x_mm=15.0, y_mm=19.0, kind=SilkKind.USER, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand")))
    res = _agent(hand, tmp_path, lib, **{ROUTING_KEY: "skip"})
    assert res.notes[-1] == "silkscreen not placed: ir.pcb already has 4 silkscreen text(s); the agent never replaces silkscreen"
    assert res.proposals[0].payload.silkscreen == hand.pcb.silkscreen


def test_the_silkscreen_placer_refuses_a_kept_text_it_cannot_keep(tmp_path: Path):
    lib = silk_library(tmp_path / "kicad")
    ir = _silk_board(tmp_path, lib)
    ir.pcb = ir.pcb.model_copy(update={"placements": fixed_placement(ir, lib).placements})
    base = place_silkscreen(ir, lib)  # without keep: exactly the 0.1 placer (the board's texts are not read)
    assert all(t.provenance.tool_version == "0.1" for t in base.texts) and base.kept == []
    assert [design_data(t) for t in place_silkscreen(ir, lib, keep=[]).texts] == [design_data(t) for t in base.texts]
    note = ir.pcb.silkscreen[0]
    with pytest.raises(CompileError, match="kept silk text 'R1' is a reference text; only title / user texts are kept"):
        place_silkscreen(ir, lib, keep=[note.model_copy(update={"kind": SilkKind.REFERENCE, "text": "R1"})])
    with pytest.raises(CompileError, match="kept silk text refused: user text 'C4' is on the fab layer 'F.Fab'"):
        place_silkscreen(ir, lib, keep=[note.model_copy(update={"layer": "F.Fab"})])
    with pytest.raises(CompileError, match="names component 'X9', which is not in the IR"):
        place_silkscreen(ir, lib, keep=[note.model_copy(update={"component_ref": "X9"})])


# --------------------------------------------------------------------------- the real KiCad 10 libraries and kicad-cli


_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.footprint_file(lib, name) is not None for lib, name in (
    ("Button_Switch_THT", "SW_PUSH_6mm"), ("Resistor_SMD", "R_0603_1608Metric"), ("Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical"))) \
    and all(_REAL.symbol_file(lib) is not None for lib in ("Switch", "Device", "Connector_Generic"))
_KICAD = KicadCli()
needs_kicad = pytest.mark.skipif(not (_KICAD.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")


@needs_kicad
def test_a_key_row_is_placed_routed_silk_placed_and_passes_real_drc(tmp_path: Path):
    """Three SW_PUSH_6mm keys fixed in a row along the lower edge (pitch: the footprint's extent read from the library + 3 mm), a note
    name under each, the chain resistors and the header packed in their regions; the PCB agent places, routes and silk-places it, the
    ``pcb.silk.*`` / ``pcb.routing.connectivity`` checks pass, and kicad-cli 10.0.6 DRC on the compiled board reports no violation and
    nothing unconnected (measured on the Windows PC)."""
    ir = CircuitIR(project=ProjectMeta(id="keys", name="keys", workdir=str(tmp_path)))
    for i in (1, 2, 3):
        ir.components.append(library_component(_REAL, f"SW{i}", "SW_Push", "key", ("Switch", "SW_Push"), ("Button_Switch_THT", "SW_PUSH_6mm"), LAYOUT, []))
        ir.components.append(library_component(_REAL, f"R{i}", "1k", "chain", ("Device", "R"), ("Resistor_SMD", "R_0603_1608Metric"), LAYOUT, []))
    ir.components.append(library_component(_REAL, "J1", "Conn", "power", ("Connector_Generic", "Conn_01x02"),
                                           ("Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical"), LAYOUT, []))
    nets = {"VCC": [("J1", "1"), ("SW1", "1"), ("SW2", "1"), ("SW3", "1")], "T1": [("SW1", "2"), ("R1", "1")], "T2": [("SW2", "2"), ("R2", "1")],
            "T3": [("SW3", "2"), ("R3", "1")], "OUT": [("R1", "2"), ("R2", "2"), ("R3", "2"), ("J1", "2")]}
    for name, members in nets.items():
        ir.nets.append(Net(name=name, pins=[PinRef(component_ref=r, pin_number=n) for r, n in members], provenance=NET_P))
    key = footprint_extent(_REAL.load_footprint(ir.component("SW1").footprint))
    pitch, gap, height = key.width + 3.0, 3.0, 40.0
    width = 2 * gap + 3 * key.width + 2 * 3.0
    fixed = [Placement(component_ref=f"SW{i}", x_mm=gap - key.x1 + (i - 1) * pitch, y_mm=height - 6.0 - key.y2, provenance=LAYOUT) for i in (1, 2, 3)]
    notes = [SilkText(text=n, x_mm=p.x_mm + (key.x1 + key.x2) / 2, y_mm=height - 3.5, size_mm=1.5, thickness_mm=0.2, kind=SilkKind.USER,
                      component_ref=p.component_ref, provenance=LAYOUT) for n, p in zip(("C4", "D4", "E4"), fixed)]
    regions = [_region("chain", [3.0, 16.0, width - 6.0, 6.0], ["R1", "R2", "R3"]), _region("power", [3.0, 3.0, 12.0, 10.0], ["J1"])]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=width, height_mm=height), fixed=fixed, pack_regions=regions, silkscreen=notes)
    res = _agent(ir, tmp_path, _REAL)
    assert res.notes[0].startswith("placement.fixed 0.1: 7 component(s)") and any(n.startswith("routing.maze") and "5 net(s) routed" in n for n in res.notes), res.notes
    Orchestrator.apply_proposals(ir, res.proposals)
    by = {p.component_ref: p for p in ir.pcb.placements}
    assert [design_data(by[p.component_ref]) for p in fixed] == [design_data(p) for p in fixed]  # the keys where the design put them
    assert by["R1"].x_mm < by["R2"].x_mm < by["R3"].x_mm and {by[r].y_mm for r in ("R1", "R2", "R3")} == {by["R1"].y_mm}  # the chain in order
    assert [t.text for t in ir.pcb.silkscreen[:3]] == ["C4", "D4", "E4"] and ir.pcb.silkscreen[:3] == notes
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": _REAL})
    silk = {r.check_id: r for r in default_registry.get("pcb.silk").validate(ir, ctx)}
    assert silk[SILK_CLEARANCE_CHECK].status is S.PASS and silk[SILK_OVERLAP_CHECK].status is S.PASS, [r.message for r in silk.values()]
    routing = {r.check_id: r for r in default_registry.get("pcb.routing").validate(ir, ctx)}
    assert routing[CONNECTIVITY_CHECK].status is S.PASS, routing[CONNECTIVITY_CHECK].message
    out = tmp_path / "out"
    art = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": _REAL}))
    tree = sexpr.parse(Path(art.path).read_text(encoding="utf-8"))
    assert [str(n[1]) for n in sexpr.find_all(tree, "gr_text") if sexpr.get(n, "layer") == "F.SilkS"][:3] == ["C4", "D4", "E4"]
    _KICAD.run_drc(Path(art.path), out / "drc.json")
    report = json.loads((out / "drc.json").read_text(encoding="utf-8"))
    assert report["violations"] == [] and report["unconnected_items"] == [], (
        [(v["type"], v["description"]) for v in report["violations"]], len(report["unconnected_items"]))
