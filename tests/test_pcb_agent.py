"""PCBAgent + the deterministic grid placer + the PLACEMENT stage.

Offline tests run on the synthetic KiCad library of ``tests/test_parts_existence.py``
(``Test:VR1`` symbol, ``Test:FP`` footprint: courtyard 2.0 x 1.2 mm, pads
extending to 2.4 mm - extent 2.4 x 1.2 mm). The one test that needs
``kicad-cli`` and the installed KiCad libraries skips elsewhere and runs
against the real binary on the Windows PC; it asserts the honest verdict (a
placed-but-unrouted board FAILs DRC) and never a DRC-clean board.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.pcb import PLACEMENT_KEY
from ai_eda.agents.requirement import CONTROL_KEYS
from ai_eda.cli import run_exit_code
from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler
from ai_eda.errors import CompileError
from ai_eda.ir import (
    ArtifactKind,
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
    assumption,
    authoritative,
)
from ai_eda.ir.provenance import design_data
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.geometry import footprint_bbox
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary, LibraryFormatError, Pad
from ai_eda.tools.placement import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, footprint_extent, grid_pitch, grid_placement
from ai_eda.tools.placement.grid import _disjoint
from ai_eda.workflow import Orchestrator, Stage
from ai_eda.workflow.stages import STAGE_ORDER
from tests.conftest import DS
from tests.fixtures_kicad import SCOPE_ANSWERS, divider_with_connector_ir
from tests.test_parts_existence import make_part, synthetic_library

ANSWERS = {"application": "test", "jurisdiction": "EU"}
NET_P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
#: the synthetic Test:FP extent at the origin: courtyard (-1, -0.6)..(1, 0.6) union pads at +-0.8 of 0.8 x 0.9 -> x +-1.2, y +-0.6
FP_W, FP_H = 2.4, 1.2

_lib = KicadLibrary()
HAS_LIBS = _lib.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and _lib.symbol_file("Device") is not None
_kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (_kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")


# --------------------------------------------------------------------------- fixtures


def parts_ir(tmp_path: Path, lib: KicadLibrary, refs: tuple[str, ...] = ("R2", "R10", "R1", "R3", "R4"), *, nets: bool = True) -> CircuitIR:
    """``refs`` as verified Test:VR1 / Test:FP parts (given order kept), chained pairwise into nets when ``nets``."""
    ir = CircuitIR(project=ProjectMeta(id="parts", name="parts", workdir=str(tmp_path)))
    for ref in refs:
        c = make_part(ref)
        c.symbol = lib.resolve_symbol(c.symbol)
        c.footprint = lib.resolve_footprint(c.footprint)
        ir.components.append(c)
    if nets and refs:
        chain = sorted(refs)
        for i, (a, b) in enumerate(zip(chain, chain[1:])):
            ir.nets.append(Net(name=f"N{i}", pins=[PinRef(component_ref=a, pin_number="2"), PinRef(component_ref=b, pin_number="1")], provenance=NET_P))
        ir.nets.append(Net(name="IN", pins=[PinRef(component_ref=chain[0], pin_number="1")], provenance=NET_P))
        ir.nets.append(Net(name="OUT", pins=[PinRef(component_ref=chain[-1], pin_number="2")], provenance=NET_P))
    return ir


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return synthetic_library(tmp_path / "kicad")


def _ctx(tmp_path: Path, lib: KicadLibrary | None, **answers: str) -> AgentContext:
    tools = {"kicad_library": lib} if lib is not None else {}
    return AgentContext(workdir=tmp_path, tools=tools, answers={**ANSWERS, **answers})


def _run_agent(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary | None, **answers: str):
    """Run the agent alone; the IR must come back untouched (content hash), whatever the outcome."""
    before = ir.content_hash()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **answers))
    assert ir.content_hash() == before  # the agent proposes, never mutates
    assert res.questions == [] and not res.blocked_on_user  # no question, ever
    return res


# --------------------------------------------------------------------------- the tool


def test_grid_placement_is_deterministic_and_ordered(tmp_path: Path, lib: KicadLibrary):
    ir = parts_ir(tmp_path, lib)
    g = grid_placement(ir, lib)
    assert [p.component_ref for p in g.placements] == ["R1", "R2", "R3", "R4", "R10"]  # natural order, not IR order
    assert g.pitch == (FP_W + SPACING_MM, FP_H + SPACING_MM) == (3.4, 2.2)
    # the tool rounds to KiCad resolution (6 decimals); the expectation is the same arithmetic rounded the same way
    assert (g.outline.width_mm, g.outline.height_mm) == (round(2 * MARGIN_MM + 3 * 3.4 + FP_W, 6), round(2 * MARGIN_MM + 1 * 2.2 + FP_H, 6)) == (16.6, 7.4)
    assert (g.outline.origin_x_mm, g.outline.origin_y_mm) == (0.0, 0.0)
    # row-major on 4 columns: R10 starts the second row under R1
    by_ref = {p.component_ref: p for p in g.placements}
    assert (by_ref["R1"].x_mm, by_ref["R1"].y_mm) == (MARGIN_MM + FP_W / 2, MARGIN_MM + FP_H / 2) == (3.2, 2.6)
    assert (by_ref["R2"].x_mm, by_ref["R2"].y_mm) == (3.2 + 3.4, 2.6)
    assert (by_ref["R10"].x_mm, by_ref["R10"].y_mm) == (3.2, round(2.6 + 2.2, 6))
    assert all(p.rotation_deg == 0.0 and p.side == "top" for p in g.placements)
    # the same result from a deep copy of the IR and from another library instance (no cache / identity effects)
    other = synthetic_library(tmp_path / "kicad2")
    again = grid_placement(copy.deepcopy(ir), other)
    payload = PCBDesign(outline=g.outline, placements=g.placements)
    assert design_data(PCBDesign(outline=again.outline, placements=again.placements)) == design_data(payload)
    assert again.extents == g.extents and again.pitch == g.pitch


@pytest.mark.parametrize("columns", [1, 4, 5])
def test_placed_extents_do_not_overlap_and_lie_inside_the_outline(tmp_path: Path, lib: KicadLibrary, columns: int):
    ir = parts_ir(tmp_path, lib)
    g = grid_placement(ir, lib, columns=columns)
    fp = lib.load_footprint(ir.component("R1").footprint)
    boxes = {p.component_ref: footprint_bbox(p, fp) for p in g.placements}
    assert boxes == g.extents  # the tool reports the extents it measured after placing
    refs = list(boxes)
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            x, y = boxes[a], boxes[b]
            assert x.x2 < y.x1 or y.x2 < x.x1 or x.y2 < y.y1 or y.y2 < x.y1, (a, b)  # strictly apart: touching counts as overlap
    o = g.outline
    for ref, box in boxes.items():
        assert o.origin_x_mm + MARGIN_MM <= box.x1 and box.x2 <= o.origin_x_mm + o.width_mm - MARGIN_MM, ref
        assert o.origin_y_mm + MARGIN_MM <= box.y1 and box.y2 <= o.origin_y_mm + o.height_mm - MARGIN_MM, ref
    rows = -(-len(refs) // columns)
    assert (o.width_mm, o.height_mm) == (round(2 * MARGIN_MM + (min(columns, len(refs)) - 1) * 3.4 + FP_W, 6), round(2 * MARGIN_MM + (rows - 1) * 2.2 + FP_H, 6))


def test_tool_refuses_instead_of_guessing(tmp_path: Path, lib: KicadLibrary):
    with pytest.raises(CompileError, match="no components"):
        grid_placement(parts_ir(tmp_path, lib, refs=()), lib)
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.component("R2").footprint = None
    with pytest.raises(CompileError, match="'R2' has no footprint"):
        grid_placement(ir, lib)
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.component("R2").footprint.name = "Missing"
    with pytest.raises(CompileError, match="Test:Missing of 'R2' was not found"):
        grid_placement(ir, lib)
    with pytest.raises(CompileError, match="columns"):
        grid_placement(parts_ir(tmp_path, lib, refs=("R1",)), lib, columns=0)
    # an extent nobody can measure: neither courtyard nor pads
    bare = FootprintDef(lib_id="Test:Bare", name="Bare", node=["footprint", "Bare"], pads=[], attr="smd", courtyard=None)
    with pytest.raises(CompileError, match="neither a courtyard nor pads"):
        footprint_extent(bare)
    with pytest.raises(CompileError, match="no extents"):
        grid_pitch({}, SPACING_MM)
    # a user outline that is too small is refused by the inside guard, never resized
    ir = parts_ir(tmp_path, lib)
    with pytest.raises(CompileError, match=r"do not fit inside the 10.0 x 5.0 mm outline.*not resized"):
        grid_placement(ir, lib, outline=BoardOutline(width_mm=10.0, height_mm=5.0))
    # a user outline elsewhere on the sheet anchors the grid at its origin + margin and is kept verbatim
    g = grid_placement(ir, lib, outline=BoardOutline(width_mm=20.0, height_mm=10.0, origin_x_mm=50.0, origin_y_mm=40.0))
    assert g.outline == BoardOutline(width_mm=20.0, height_mm=10.0, origin_x_mm=50.0, origin_y_mm=40.0)
    assert (g.placements[0].x_mm, g.placements[0].y_mm) == (50.0 + 3.2, 40.0 + 2.6)


def test_touching_extents_are_refused_by_the_guard(tmp_path: Path, lib: KicadLibrary):
    """With ``spacing=0`` the pitch equals the extent, so neighbouring extents share an edge: the re-measured guard refuses the grid."""
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    with pytest.raises(CompileError, match=r"placed extents of 'R1' and 'R2' touch or overlap .*; refusing the grid"):
        grid_placement(ir, lib, spacing=0.0)
    with pytest.raises(CompileError, match="touch or overlap"):  # one column: the shared edge is horizontal
        grid_placement(ir, lib, spacing=0.0, columns=1)
    assert len(grid_placement(ir, lib, spacing=0.01).placements) == 2  # any gap at all is apart
    # the rule itself: a shared edge is not disjoint, a 0.01 mm gap is, and the test is symmetric
    a = BBox(0.0, 0.0, 2.0, 1.0)
    for b in (BBox(2.0, 0.0, 4.0, 1.0), BBox(0.0, 1.0, 2.0, 2.0), BBox(-2.0, 0.0, 0.0, 1.0), BBox(0.0, -1.0, 2.0, 0.0), BBox(1.0, 0.5, 3.0, 1.5), a):
        assert not _disjoint(a, b) and not _disjoint(b, a), b
    for b in (BBox(2.01, 0.0, 4.0, 1.0), BBox(0.0, 1.01, 2.0, 2.0), BBox(-2.0, 0.0, -0.01, 1.0), BBox(0.0, -1.0, 2.0, -0.01)):
        assert _disjoint(a, b) and _disjoint(b, a), b


def _write_footprint(root: Path, name: str, at: str, size: str) -> KicadLibrary:
    """``Test:<name>`` with the synthetic courtyard and one pad whose ``(at ...)`` / ``(size ...)`` are given verbatim."""
    pretty = root / "footprints" / "Test.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    (pretty / f"{name}.kicad_mod").write_text(
        f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n'
        f'  (fp_rect (start -1 -0.6) (end 1 0.6) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        f'  (pad "1" smd rect (at {at}) (size {size}) (layers "F.Cu" "F.Mask" "F.Paste")))\n',
        encoding="utf-8",
    )
    return KicadLibrary(roots=[root])


@pytest.mark.parametrize("at, size, what", [("nan 0", "0.8 0.9", "pad '1' (at x)"), ("0.8 0", "inf 0.9", "pad '1' (size w)"), ("0.8 -inf", "0.8 0.9", "pad '1' (at y)")])
def test_non_finite_footprint_geometry_is_refused_not_dropped(tmp_path: Path, lib: KicadLibrary, at: str, size: str, what: str):
    """``float("nan")`` vanishes from ``min`` / ``max`` and ``inf`` is everywhere: such a pad is refused at the loader, the extent and the agent, never placed without it."""
    fresh = _write_footprint(tmp_path / "kicad_bad", "FP", at, size)  # its own root: ``lib`` (the good Test:FP) resolves the part first
    ir = parts_ir(tmp_path, lib, refs=("R1",))
    ref = ir.component("R1").footprint
    assert fresh.footprint_file("Test", "FP") is not None  # the file exists; it is the geometry that is refused (resolve loads it, so it refuses too)
    pattern = rf"FP\.kicad_mod: footprint 'FP': {what.replace('(', chr(92) + '(').replace(')', chr(92) + ')')} is not a finite number"
    with pytest.raises(LibraryFormatError, match=pattern):
        fresh.load_footprint(ref)
    with pytest.raises(LibraryFormatError, match=pattern):
        fresh.resolve_footprint(ref)
    with pytest.raises(CompileError, match="not a finite number"):
        grid_placement(ir, fresh)
    res = _run_agent(ir, tmp_path, fresh)
    assert res.proposals == [] and len(res.notes) == 1 and res.notes[0].startswith("not placed: ") and "is not a finite number" in res.notes[0] and "FP.kicad_mod" in res.notes[0]


def test_extent_of_a_hand_built_footprint_with_non_finite_geometry_is_refused():
    """Belt and braces behind the loader: ``footprint_bbox`` / ``footprint_extent`` never return a box that dropped a NaN pad or reaches infinity."""

    def pad(**over: float) -> Pad:
        fields = {"number": "1", "pad_type": "smd", "shape": "rect", "x": -0.8, "y": 0.0, "rotation": 0.0, "size_w": 0.8, "size_h": 0.9, "drill": None, "layers": ["F.Cu"]}
        return Pad(**{**fields, **over})

    courtyard = BBox(-1.0, -0.6, 1.0, 0.6)
    finite = FootprintDef(lib_id="Test:Finite", name="Finite", node=["footprint", "Finite"], pads=[pad()], attr="smd", courtyard=courtyard)
    assert footprint_extent(finite) == BBox(-1.2, -0.6, 1.0, 0.6)
    placed = Placement(component_ref="R1", x_mm=5.0, y_mm=5.0, rotation_deg=0.0, side=BoardSide.TOP)
    # a NaN pad would silently shrink the extent to the courtyard and an infinite one stretches it to infinity: both are refused
    for pads, cy in (
        ([pad(), pad(number="2", x=float("nan"))], courtyard),
        ([pad(), pad(number="2", x=0.8, size_w=float("inf"))], courtyard),
        ([pad()], BBox(-1.0, float("nan"), 1.0, 0.6)),
        ([pad()], BBox(-1.0, -0.6, float("inf"), 0.6)),
    ):
        fp = FootprintDef(lib_id="Test:Bad", name="Bad", node=["footprint", "Bad"], pads=pads, attr="smd", courtyard=cy)
        with pytest.raises(CompileError, match="footprint Test:Bad .*non-finite"):
            footprint_extent(fp)
        with pytest.raises(CompileError, match="footprint Test:Bad .*non-finite"):
            footprint_bbox(placed, fp)


def test_every_placement_is_traced_to_the_tool_and_its_inputs(tmp_path: Path, lib: KicadLibrary):
    ir = parts_ir(tmp_path, lib)
    for p in grid_placement(ir, lib).placements:
        prov = p.provenance
        assert prov.kind is ProvenanceKind.DERIVED and not prov.needs_verification
        assert prov.tool == PLACER_ID == "placement.grid" and prov.tool_version == PLACER_VERSION
        assert prov.derived_from == ["footprint:Test:FP", f"spacing_mm:{SPACING_MM}", f"margin_mm:{MARGIN_MM}", f"columns:{COLUMNS}"]
        assert prov.inputs == {}  # the calculator role map stays empty: a placement is not a calculator output
        assert prov.note and "DRC" in prov.note
    # the reviewer accepts derived placements on a fresh board
    res = _run_agent(ir, tmp_path, lib)
    Orchestrator.apply_proposals(ir, res.proposals)
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    ir.artifacts[ArtifactKind.PCB] = art
    areas = {r.check_id: r for r in IndependentReviewer(tools={"kicad_library": lib}).review(ir, tmp_path).results}
    assert areas[ReviewArea.IR_VS_PCB].status is S.PASS, areas[ReviewArea.IR_VS_PCB].message
    # ... and not a placement whose origin was replaced by an assumption
    ir.pcb.placements[0].provenance = Provenance(kind=ProvenanceKind.ASSUMPTION, note="moved by hand")
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    areas = {r.check_id: r for r in IndependentReviewer(tools={"kicad_library": lib}).review(ir, tmp_path).results}
    assert areas[ReviewArea.IR_VS_PCB].status is S.NOT_VERIFIED and "placement[R1]" in areas[ReviewArea.IR_VS_PCB].details["unverified_layout"]


# --------------------------------------------------------------------------- the agent


def test_agent_proposes_only_when_nothing_is_placed(tmp_path: Path, lib: KicadLibrary):
    # ir.pcb None -> one whole-pcb proposal that validates through apply_proposals
    ir = parts_ir(tmp_path, lib)
    res = _run_agent(ir, tmp_path, lib)
    assert len(res.proposals) == 1 and res.proposals[0].target == "pcb" and res.proposals[0].operation == "set"
    assert isinstance(res.proposals[0].payload, PCBDesign) and "5 component(s)" in res.proposals[0].description
    assert any("unconnected_items" in n for n in res.notes)  # the note says what DRC will say, never "clean"
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.pcb is not None and [p.component_ref for p in ir.pcb.placements] == ["R1", "R2", "R3", "R4", "R10"]
    assert ir.pcb.outline == BoardOutline(width_mm=16.6, height_mm=7.4)
    # ir.pcb with placements -> nothing proposed, the reason noted
    res = _run_agent(ir, tmp_path, lib)
    assert res.proposals == [] and res.notes == ["not placed: ir.pcb already has 5 placement(s); the agent never replaces a layout"]
    # ir.pcb with layers / manufacturing constraints but no placements -> the proposal keeps them and fills outline + placements
    ir = parts_ir(tmp_path, lib)
    layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="power"), Layer(name="In2.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")]
    ir.pcb = PCBDesign(layers=layers, manufacturing=ManufacturingConstraints(fab="JLCPCB", min_track_width_mm=assumption(0.127, note="fab page not read")))
    res = _run_agent(ir, tmp_path, lib)
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.pcb.layers == layers and ir.pcb.manufacturing.fab == "JLCPCB" and ir.pcb.manufacturing.min_track_width_mm.value == 0.127
    assert len(ir.pcb.placements) == 5 and ir.pcb.outline == BoardOutline(width_mm=16.6, height_mm=7.4)
    # a user outline (origin not at 0, 0) that is large enough is kept verbatim and the grid starts at origin + margin
    ir = parts_ir(tmp_path, lib)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=25.0, height_mm=12.0, origin_x_mm=100.0, origin_y_mm=60.0))
    res = _run_agent(ir, tmp_path, lib)
    assert len(res.proposals) == 1 and "user outline at (100.0, 60.0)" in res.proposals[0].description
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.pcb.outline == BoardOutline(width_mm=25.0, height_mm=12.0, origin_x_mm=100.0, origin_y_mm=60.0)
    assert (ir.pcb.placement("R1").x_mm, ir.pcb.placement("R1").y_mm) == (103.2, 62.6)
    # too small -> nothing proposed, the outline untouched, the reason noted
    ir = parts_ir(tmp_path, lib)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10.0, height_mm=5.0))
    res = _run_agent(ir, tmp_path, lib)
    assert res.proposals == [] and len(res.notes) == 1 and res.notes[0].startswith("not placed: component(s) [") and "not resized" in res.notes[0]
    assert ir.pcb.outline == BoardOutline(width_mm=10.0, height_mm=5.0) and ir.pcb.placements == []


def test_agent_refuses_to_guess(tmp_path: Path, lib: KicadLibrary):
    # a component without a footprint
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.component("R2").footprint = None
    res = _run_agent(ir, tmp_path, lib)
    assert res.proposals == [] and res.notes == ["not placed: component 'R2' has no footprint; the placer never picks one"]
    # a footprint that is not in the library
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.component("R2").footprint.name = "Missing"
    res = _run_agent(ir, tmp_path, lib)
    assert res.proposals == [] and len(res.notes) == 1 and "Test:Missing of 'R2' was not found" in res.notes[0] and res.notes[0].startswith("not placed: ")
    # no components
    res = _run_agent(parts_ir(tmp_path, lib, refs=()), tmp_path, lib)
    assert res.proposals == [] and res.notes == ["not placed: no components to place"]
    # no library in the tool context
    res = _run_agent(parts_ir(tmp_path, lib), tmp_path, None)
    assert res.proposals == [] and res.notes == ["not placed: no KiCad library in ctx.tools['kicad_library']; footprint extents cannot be read"]
    # a corrupt .kicad_mod: the library's format error is a note, not a crash (the file exists, so resolve says verified=False or load raises)
    root = tmp_path / "kicad"
    (root / "footprints" / "Test.pretty" / "FP.kicad_mod").write_text("(footprint \"FP\" (pad \"1\" smd", encoding="utf-8")
    fresh = KicadLibrary(roots=[root])
    ir = parts_ir(tmp_path, lib, refs=("R1",))
    with pytest.raises(LibraryFormatError):
        fresh.load_footprint(ir.component("R1").footprint)
    res = _run_agent(ir, tmp_path, fresh)
    assert res.proposals == [] and len(res.notes) == 1 and res.notes[0].startswith("not placed: ") and "FP.kicad_mod" in res.notes[0]


def test_agent_never_places_under_existing_copper(tmp_path: Path, lib: KicadLibrary):
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0, height_mm=20.0), tracks=[Track(net="N0", layer="F.Cu", start=(1.0, 1.0), end=(5.0, 1.0), width_mm=0.25)])
    res = _run_agent(ir, tmp_path, lib)
    assert res.proposals == [] and res.notes == ["not placed: ir.pcb has copper without placements (1 track(s), 0 via(s), 0 zone(s)); placing under existing copper would be a guess"]


def test_placement_answer_is_a_control_key_and_skip_proposes_nothing(tmp_path: Path, lib: KicadLibrary):
    assert PLACEMENT_KEY == "pcb.placement" and PLACEMENT_KEY in CONTROL_KEYS
    # skip: nothing proposed, the note says so
    ir = parts_ir(tmp_path, lib)
    res = _run_agent(ir, tmp_path, lib, **{PLACEMENT_KEY: " Skip "})
    assert res.proposals == [] and res.notes == ["placement skipped by answer"]
    # any other value: noted as not understood, placement proceeds
    res = _run_agent(ir, tmp_path, lib, **{PLACEMENT_KEY: "yes"})
    assert len(res.proposals) == 1 and res.notes[0] == "pcb.placement='yes' not understood (the only answer is 'skip'); placing as usual"
    # through the pipeline: the answer steers the stage and never becomes a requirement or a question
    for answer, placed in (("skip", False), ("yes", True)):
        ir = parts_ir(tmp_path / answer, lib)
        state = Orchestrator(_ctx(tmp_path / answer, lib, **{PLACEMENT_KEY: answer})).run(ir, stop_after=Stage.IR_BUILD)
        assert ir.requirements.get(PLACEMENT_KEY) is None and all(r.key != PLACEMENT_KEY for r in ir.requirements.requirements)
        assert all(q.key != PLACEMENT_KEY for q in state.optional_questions + state.open_questions)
        out = state.outcome(Stage.PLACEMENT)
        assert out.status is S.NOT_VERIFIED
        assert (ir.pcb is not None and len(ir.pcb.placements) == 5) is placed
        assert ("placement skipped by answer" in out.message) is not placed


# --------------------------------------------------------------------------- compiled


def test_proposed_board_compiles_with_the_existing_compiler(tmp_path: Path, lib: KicadLibrary):
    ir = parts_ir(tmp_path, lib)
    Orchestrator.apply_proposals(ir, _run_agent(ir, tmp_path, lib).proposals)
    a = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "a", tools={"kicad_library": lib}))
    b = PCBCompiler().compile(copy.deepcopy(ir), CompileContext(workdir=tmp_path / "b", tools={"kicad_library": synthetic_library(tmp_path / "kicad2")}))
    assert Path(a.path).read_bytes() == Path(b.path).read_bytes() and a.content_hash == b.content_hash
    board = sexpr.parse_file(Path(a.path))
    at = {}
    for fp in sexpr.find_all(board, "footprint"):
        ref = next(str(p[2]) for p in sexpr.find_all(fp, "property") if str(p[1]) == "Reference")
        at[ref] = tuple(float(v) for v in sexpr.find(fp, "at")[1:3])
    assert at == {p.component_ref: (p.x_mm, p.y_mm) for p in ir.pcb.placements}
    rect = next(n for n in sexpr.find_all(board, "gr_rect"))
    assert [float(v) for v in sexpr.find(rect, "start")[1:]] == [0.0, 0.0] and [float(v) for v in sexpr.find(rect, "end")[1:]] == [16.6, 7.4]
    assert SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "a", tools={"kicad_library": lib})).path.endswith("parts.kicad_sch")


# --------------------------------------------------------------------------- the stage


def test_placement_stage_runs_before_ir_build_and_keeps_validator_hashes_fresh(tmp_path: Path, lib: KicadLibrary):
    assert STAGE_ORDER.index(Stage.PLACEMENT) == STAGE_ORDER.index(Stage.COMPONENT_SELECTION) + 1
    # FAB_CAPABILITY sits between PLACEMENT (which may create ir.pcb) and IR_BUILD, so the grounded fab limits land in the placed board
    assert STAGE_ORDER.index(Stage.PLACEMENT) == STAGE_ORDER.index(Stage.FAB_CAPABILITY) - 1 == STAGE_ORDER.index(Stage.IR_BUILD) - 2
    ir = parts_ir(tmp_path, lib)
    state = Orchestrator(_ctx(tmp_path, lib)).run(ir)
    assert not state.blocked and [o.stage for o in state.outcomes] == list(Stage)
    out = state.outcome(Stage.PLACEMENT)
    assert out.status is S.NOT_VERIFIED and out.message.startswith("1 proposal(s) applied, nothing verified; placement.grid 0.1: 5 component(s)")
    assert out.questions == []
    assert state.outcome(Stage.PCB).status is S.PASS and ArtifactKind.PCB in ir.artifacts
    assert state.outcome(Stage.DRC).status is S.NOT_VERIFIED  # kicad-cli decides, and it is not here
    # every validator result is about the placed design: nothing is stale at RELEASE
    current = ir.content_hash()
    stamped = [r for r in ir.validation.results if r.check_id.startswith("ir.")]
    assert stamped and all(r.ir_hash == current for r in stamped)
    assert "another IR version" not in state.outcome(Stage.RELEASE).message
    assert state.outcome(Stage.RELEASE).status is not S.PASS


def test_inconsistent_ir_is_still_placed_and_blocked_at_ir_build(tmp_path: Path, lib: KicadLibrary):
    ir = parts_ir(tmp_path, lib, refs=("R1", "R2"))
    ir.nets.append(Net(name="X", pins=[PinRef(component_ref="R9", pin_number="1")], provenance=NET_P))  # ir.connectivity FAIL
    ir.parameters["guess"] = assumption(1.0, note="unconfirmed", unit="V")  # ir.assumptions USER_INPUT_REQUIRED
    state = Orchestrator(_ctx(tmp_path, lib)).run(ir)
    assert state.outcome(Stage.PLACEMENT).status is S.NOT_VERIFIED and "1 proposal(s) applied" in state.outcome(Stage.PLACEMENT).message
    assert ir.pcb is not None and [p.component_ref for p in ir.pcb.placements] == ["R1", "R2"]  # the placer walks components only
    assert state.outcome(Stage.IR_BUILD).status is S.FAIL and state.blocked and state.current is Stage.IR_BUILD
    assert [q.key for q in state.open_questions] == ["ir.assumptions"]


def test_bom_fail_survives_a_gerber_tool_unavailable(tmp_path: Path, lib: KicadLibrary):
    """A missing kicad-cli makes the gerber export NOT_VERIFIED; it must not discard the BOM refusal recorded before it."""
    ir = parts_ir(tmp_path, lib)
    ir.component("R1").mpn = authoritative("=CMD()", DS)  # the BOM compiler refuses this identity cell
    ctx = _ctx(tmp_path, lib)
    orch = Orchestrator(ctx)
    state = orch.run(ir, stop_after=Stage.PCB)
    assert state.outcome(Stage.PCB).status is S.PASS and "kicad_cli" not in ctx.tools
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert out.status is S.FAIL, out.message
    assert "bom compile refused" in out.message and "gerber export skipped" in out.message and "kicad_cli" in out.message
    assert ir.validation.latest("compile.bom").status is S.FAIL and ArtifactKind.GERBER not in ir.artifacts


# --------------------------------------------------------------------------- real kicad-cli (Windows PC)


@needs_kicad
def test_placed_but_unrouted_divider_fails_real_drc_with_unconnected_items(tmp_path: Path):
    """The honest verdict on a KiCad machine: a components-only IR gets a placed board, DRC FAILs (unconnected), exit code 1.

    The expected count is derived from the IR (one ratsnest line per pad
    beyond the first in each net: VIN 1 + VOUT 2 + GND 1 = 4 for the
    fixture); the run on the Windows PC confirms it. Nothing here claims a
    DRC-clean board.
    """
    ir = divider_with_connector_ir(tmp_path, _lib)
    ir.pcb = None  # components only: no outline, no placements, no copper
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": _lib, "kicad_cli": _kicad}, answers=SCOPE_ANSWERS)
    state = Orchestrator(ctx).run(ir)
    assert not state.blocked
    placement = state.outcome(Stage.PLACEMENT)
    assert placement.status is S.NOT_VERIFIED and "1 proposal(s) applied" in placement.message
    assert [p.component_ref for p in ir.pcb.placements] == ["J1", "R1", "R2"]
    assert state.outcome(Stage.PCB).status is S.PASS, state.outcome(Stage.PCB).message
    drc = state.outcome(Stage.DRC)
    assert drc.status is S.FAIL, drc.message
    res = ir.validation.latest("kicad.drc")
    assert res.is_tool_backed and res.tool and res.tool_version and res.artifact_hash == ir.artifacts[ArtifactKind.PCB].content_hash
    unconnected = res.details["unconnected_items"]
    types = sorted({v.get("type") for v in res.details["errors"] + res.details["warnings"] + unconnected})
    print("DRC violation types on the placed, unrouted divider:", types, "unconnected:", len(unconnected))
    expected = sum(len(n.pins) - 1 for n in ir.nets)
    assert unconnected, f"an unrouted board with nets must report unconnected items (types seen: {types})"
    assert len(unconnected) == expected, f"expected {expected} unconnected items derived from the IR nets, DRC reported {len(unconnected)} (types: {types})"
    assert state.outcome(Stage.RELEASE).status is S.FAIL and run_exit_code(state) == 1
