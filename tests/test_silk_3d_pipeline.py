"""Silkscreen + 3D through the pipeline and the reports: the parts compose, the preview GLB is a registered derived artifact, KiCad's
own 3D exports run after the gerbers where kicad-cli exists, and the board figure / circuit report show the silk and the 3D preview.

Everything runs offline on the synthetic libraries of ``tests/test_silkscreen.py`` (silk-bearing footprints),
``tests/test_model3d.py`` (footprints with ``(model ...)`` references + a hand-written STEP file) and
``tests/test_parts_existence.py`` (the pipeline fixture of ``tests/test_pcb_agent.py``). KiCad's 3D exports are exercised through a
fake ``KicadCli`` whose ``_run`` writes files with the right signatures - that proves the wiring, never what the real kicad-cli
10.0.6 accepts (``tests/test_kicad_3d_export.py`` is that canary; it skips without kicad-cli). One test reads the installed KiCad
footprints and the 3D model library ``AI_EDA_TEST_3DMODEL_DIR`` names; it skips unless both are there.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import CompileContext, GlbExporter, Preview3DCompiler, RenderExporter, StepExporter
from ai_eda.compilers.model3d import EXPORT_3D_DIR, MODEL_DIR_TOOL, PREVIEW_SUFFIX, RENDER_VIEWS, preview_path
from ai_eda.errors import CompileError, NothingToCompileError, ToolExecutionError
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    BoardOutline,
    BoardSide,
    CircuitIR,
    Component,
    LibraryRef,
    PCBDesign,
    Placement,
    ProjectMeta,
    SilkKind,
    SilkText,
    ValidationStatus as S,
    hash_file_set,
)
from ai_eda.report import build_stage_document, circuit_report, render_report_file
from ai_eda.report.figures import SILK_COLOUR, board_figure, model3d_figure
from ai_eda.report.stages import SLOT_ISO3D, final_report
from ai_eda.tools.kicad.cli import GLB_EXPORT_FLAGS, PNG_SIGNATURE, STEP_EXPORT_FLAGS, KicadCli
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.model3d import BODY_CAPTION, build_scene, iso_svg, write_glb
from ai_eda.tools.silkscreen import place_silkscreen
from ai_eda.tools.silkscreen.geometry import ir_text_box
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK, SILK_TOOL_ID
from ai_eda.workflow import Orchestrator, Stage
from ai_eda.workflow.orchestrator import KICAD_3D_EXPORTS, NO_KICAD_3D
from tests.test_model3d import USER, _box_step, _ir, _library, _place
from tests.test_parts_existence import synthetic_library
from tests.test_pcb_agent import parts_ir
from tests.test_silkscreen import ANSWERS, basic_ir, silk_ir, silk_library

SVG_NS = "{http://www.w3.org/2000/svg}"
#: every check id a 3D output could have been recorded under: none may exist (a picture is not evidence)
_3D_CHECK_WORDS = ("model_3d", "kicad_step", "kicad_glb", "kicad_render", "preview")


@pytest.fixture(autouse=True)
def _no_3d_library_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests name their 3D library explicitly; a KICAD10_3DMODEL_DIR of the machine running them must not change a byte."""
    monkeypatch.delenv("KICAD10_3DMODEL_DIR", raising=False)
    monkeypatch.delenv("KICAD_3DMODEL_DIR", raising=False)


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return silk_library(tmp_path / "kicad")


def _silked(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    ir = basic_ir(tmp_path, lib)
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    return ir


def _no_3d_results(ir: CircuitIR) -> None:
    assert not [r.check_id for r in ir.validation.results if any(w in r.check_id for w in _3D_CHECK_WORDS)]


# --------------------------------------------------------------------------- (1) the two parts compose


def test_the_3d_scene_draws_the_library_silk_strokes_and_only_counts_the_placed_silk_texts(tmp_path: Path, lib: KicadLibrary):
    """The placer's texts (references, pin labels, title) change no solid of the scene: its silk strokes are the footprints' own,
    the IR texts are counted in the notes (fab-layer references are not silk and not counted), and the IR is not touched."""
    bare = basic_ir(tmp_path, lib)
    plain = build_scene(bare, lib, model_dir=None)
    ir = _silked(tmp_path, lib)
    ir.pcb.silkscreen.append(SilkText(text="R1", x_mm=10.0, y_mm=10.0, layer="F.Fab", kind=SilkKind.REFERENCE, component_ref="R1", provenance=USER))
    before = ir.content_hash()
    scene = build_scene(ir, lib, model_dir=None)
    assert ir.content_hash() == before
    assert scene.solids == plain.solids and [s for s in scene.solids if s.kind == "silk"]
    on_silk = sum(1 for t in ir.pcb.silkscreen if t.layer.endswith(".SilkS"))
    assert on_silk == len(ir.pcb.silkscreen) - 1 and any(f"IR 실크 문자 {on_silk}개" in n for n in scene.notes)
    # both writers take the composed scene
    assert write_glb(scene)[:4] == b"glTF" and ET.fromstring(iso_svg(scene)).tag == f"{SVG_NS}svg"


# --------------------------------------------------------------------------- (2) the preview GLB: a registered derived artifact


def _box_world(tmp_path: Path) -> tuple[KicadLibrary, Path, CircuitIR]:
    lib = _library(tmp_path / "kicad")
    models = tmp_path / "3d"
    _box_step(models)
    ir = _ir([("U1", "BOX", _place("U1", x=8.0, y=8.0)), ("J1", "THT", _place("J1", x=20.0, y=10.0))])
    return lib, models, ir


def test_preview_compiler_writes_a_deterministic_glb_registered_with_the_ir_hash(tmp_path: Path):
    lib, models, ir = _box_world(tmp_path)
    work = tmp_path / "work"
    compiler = Preview3DCompiler()
    before = ir.model_dump_json()
    art, scene = compiler.compile_scene(ir, CompileContext(workdir=work, tools={"kicad_library": lib, MODEL_DIR_TOOL: models}))
    assert ir.model_dump_json() == before and ir.artifacts == {}  # the compiler registers nothing itself: the stage does
    path = Path(art.path)
    assert path == preview_path(ir, work) == work / f"m3d{PREVIEW_SUFFIX}" and path.is_file()
    data = path.read_bytes()
    assert art.kind is ArtifactKind.MODEL_3D and art.generated_from_ir_hash == ir.content_hash() and art.matches_disk()
    assert art.content_hash == "sha256:" + hashlib.sha256(data).hexdigest() and (art.generator, art.generator_version) == ("compiler.preview_glb", "0.1")
    magic, version, length = struct.unpack("<III", data[:12])
    assert data[:4] == b"glTF" and version == 2 and length == len(data) and not art.notes
    assert [b.ref for b in scene.bodies_with_step] == ["U1"] and [b.ref for b in scene.bodies_without_step] == ["J1"]
    assert str(tmp_path).encode() not in data  # no path inside
    # byte-identical from a fresh library instance; compile() is compile_scene()'s artifact
    again = compiler.compile(ir, CompileContext(workdir=tmp_path / "again", tools={"kicad_library": KicadLibrary(roots=lib.roots), MODEL_DIR_TOOL: models}))
    assert Path(again.path).read_bytes() == data and again.content_hash == art.content_hash
    # the 3D library: None = none (every body flat), else discovered like KiCad's own (the environment variable here)
    flat = compiler.compile(ir, CompileContext(workdir=tmp_path / "flat", tools={"kicad_library": lib, MODEL_DIR_TOOL: None}))
    assert flat.content_hash != art.content_hash
    discovered = compiler.compile(ir, CompileContext(workdir=tmp_path / "none", tools={"kicad_library": lib}))
    assert discovered.content_hash == flat.content_hash  # nothing to discover: no env variable, no <root>/3dmodels
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KICAD10_3DMODEL_DIR", str(models))
        assert compiler.compile(ir, CompileContext(workdir=tmp_path / "env", tools={"kicad_library": lib})).content_hash == art.content_hash


def test_preview_compiler_refuses_instead_of_guessing(tmp_path: Path):
    lib, models, ir = _box_world(tmp_path)
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib, MODEL_DIR_TOOL: models})
    compiler = Preview3DCompiler()
    for mutate, error, words in (
        (lambda i: setattr(i, "pcb", None), NothingToCompileError, "ir.pcb is None"),
        (lambda i: setattr(i, "components", []), NothingToCompileError, "no components"),
        (lambda i: setattr(i.project, "id", "a b"), CompileError, "not usable as a file stem"),
        (lambda i: setattr(i.pcb, "outline", None), CompileError, "outline"),
        (lambda i: setattr(i.components[0], "footprint", LibraryRef(library="Nope", name="Missing")), CompileError, "not found in a KiCad library"),
        (lambda i: i.pcb.placements.pop(), CompileError, "no placement"),
    ):
        bad = ir.model_copy(deep=True)
        mutate(bad)
        with pytest.raises(error, match=words):
            compiler.compile(bad, ctx)
    with pytest.raises(CompileError, match="not a directory path"):
        compiler.compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib, MODEL_DIR_TOOL: 3}))
    assert not list(tmp_path.glob(f"*{PREVIEW_SUFFIX}"))


def _pipeline(tmp_path: Path, **extra_tools) -> tuple[CircuitIR, AgentContext, Orchestrator]:
    slib = synthetic_library(tmp_path / "kicad")
    ir = parts_ir(tmp_path, slib)
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": slib, **extra_tools}, answers=dict(ANSWERS))
    return ir, ctx, Orchestrator(ctx)


def test_pcb_stage_compiles_the_preview_after_the_board_and_claims_no_status(tmp_path: Path):
    ir, ctx, orch = _pipeline(tmp_path)
    state = orch.run(ir, stop_after=Stage.PCB)
    pcb = state.outcome(Stage.PCB)
    assert pcb.status is S.PASS and ir.pcb.silkscreen  # placed, routed and silk-screened by the PLACEMENT stage
    art = ir.artifacts[ArtifactKind.MODEL_3D]
    assert Path(art.path) == tmp_path / "parts.preview.glb" and art.generated_from_ir_hash == ir.content_hash() and art.matches_disk()
    assert pcb.message.startswith(f"{ir.artifacts[ArtifactKind.PCB].path}; 3D preview parts.preview.glb: ")
    assert "0 part box(es) with a STEP height, 5 flat outline(s) without (a picture of the IR, not a check)" in pcb.message
    _no_3d_results(ir)
    assert ir.validation.latest("compile.kicad_pcb").status is S.PASS
    # a preview the scene refuses is a note: the stage keeps the board's status and the old preview is unregistered
    class Refuses(Preview3DCompiler):
        def scene(self, ir, ctx):
            raise CompileError("no picture today")

    ctx.tools["compilers"][ArtifactKind.MODEL_3D] = Refuses()
    out = orch.stages[Stage.PCB](ir, ctx)
    assert out.status is S.PASS and out.message.endswith("; 3D preview not compiled: no picture today") and ArtifactKind.MODEL_3D not in ir.artifacts
    _no_3d_results(ir)
    # a board that does not compile leaves no preview either
    ctx.tools["compilers"][ArtifactKind.MODEL_3D] = Preview3DCompiler()
    assert orch.stages[Stage.PCB](ir, ctx).status is S.PASS and ArtifactKind.MODEL_3D in ir.artifacts
    ir.component("R1").footprint = LibraryRef(library="Nope", name="Missing")
    out = orch.stages[Stage.PCB](ir, ctx)
    assert out.status is S.FAIL and out.message.endswith("; 3D preview not compiled: the board did not compile") and ArtifactKind.MODEL_3D not in ir.artifacts


def test_the_html_report_and_the_final_report_list_the_preview_artifact(tmp_path: Path):
    ir, ctx, orch = _pipeline(tmp_path)
    orch.run(ir, stop_after=Stage.PCB)
    ir_path = tmp_path / "ir.json"
    ir.save(ir_path)
    html = render_report_file(ir_path).read_text(encoding="utf-8")
    assert "<td>model_3d</td>" in html and "parts.preview.glb" in html
    md = final_report(ir, None)
    row = next(line for line in md.splitlines() if line.startswith("| `model_3d` |"))
    assert "| parts.preview.glb |" in row and "현재 IR 에서 생성" in row and str(tmp_path) not in md


# --------------------------------------------------------------------------- (3) KiCad's own 3D exports in MANUFACTURING_OUTPUTS


class FakeKicad(KicadCli):
    """A kicad-cli double: ``_run`` writes a file with the signature each export checks (or fails the named subcommands). Wiring only."""

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        super().__init__(binary="kicad-cli-fake")
        self.fail = set(fail)
        self.calls: list[list[str]] = []

    def _run(self, args: list[str], timeout: int = 600, ok_codes: tuple[int, ...] = (0, 5)) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if args == ["version"]:
            return subprocess.CompletedProcess(args, 0, "10.0.6-fake\n", "")
        what = args[2] if args[:2] == ["pcb", "export"] else args[1]
        if what in self.fail:
            raise ToolExecutionError(f"kicad-cli {' '.join(args)} exited 1: unknown flag")
        out = Path(args[args.index("--output") + 1])
        out.write_bytes({"step": b"ISO-10303-21;\nHEADER;", "glb": b"glTF\x02\x00\x00\x00", "render": PNG_SIGNATURE + b"fake"}[what])
        return subprocess.CompletedProcess(args, 0, "", "")


class _NoGerbers:
    """Stands in for the gerber / drill exporters so the stage reaches the 3D exports without a real kicad-cli."""

    id = "stub"
    version = "0"

    def compile(self, ir, ctx):
        raise NothingToCompileError("stubbed out in this test")


def test_without_kicad_cli_the_outcome_says_3d_export_skipped_never_fail(tmp_path: Path):
    ir, ctx, orch = _pipeline(tmp_path)
    orch.run(ir, stop_after=Stage.PCB)
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert out.status is S.NOT_VERIFIED and "gerber export skipped" in out.message and out.message.endswith(f"; {NO_KICAD_3D}")
    assert NO_KICAD_3D == "3D export skipped: kicad-cli not found"
    missing = KicadCli()
    missing.binary = None  # a KicadCli that found no binary
    ctx.tools["kicad_cli"] = missing
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert out.status is S.NOT_VERIFIED and out.message.endswith(f"; {NO_KICAD_3D}")
    assert not any(kind in ir.artifacts for kind in KICAD_3D_EXPORTS)
    _no_3d_results(ir)
    # no board at all: said for the 3D exports too
    ir.artifacts.pop(ArtifactKind.PCB)
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert "gerber/drill skipped (no PCB); 3D export skipped (no PCB)" in out.message


def test_with_kicad_cli_the_3d_exports_follow_the_gerbers_and_are_registered_like_them(tmp_path: Path):
    ir, ctx, orch = _pipeline(tmp_path)
    orch.run(ir, stop_after=Stage.PCB)
    fake = FakeKicad()
    ctx.tools["kicad_cli"] = fake
    ctx.tools["compilers"][ArtifactKind.GERBER] = _NoGerbers()
    ctx.tools["compilers"][ArtifactKind.DRILL] = _NoGerbers()
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert out.status is S.NOT_VERIFIED  # the stubbed gerbers' verdict; the 3D exports add none
    assert out.message.index("gerber: ") < out.message.index("kicad_step exported parts.step (3D, not a check)")
    assert "kicad_glb exported parts.glb (3D, not a check)" in out.message and "kicad_render exported parts-top.png, parts-bottom.png (3D, not a check)" in out.message
    pcb_path = ir.artifacts[ArtifactKind.PCB].path
    out_dir = tmp_path / EXPORT_3D_DIR
    expected = {ArtifactKind.KICAD_STEP: ["parts.step"], ArtifactKind.KICAD_GLB: ["parts.glb"], ArtifactKind.KICAD_RENDER: [f"parts-{v}.png" for v in RENDER_VIEWS]}
    for kind, names in expected.items():
        art = ir.artifacts[kind]
        files = [out_dir / n for n in names]
        assert [Path(f) for f in art.files] == files and Path(art.path) == files[0] and art.content_hash == hash_file_set(files)
        assert art.generated_from_ir_hash == ir.content_hash() and art.generator_version == "0.1/kicad-cli 10.0.6-fake"
    step_call = next(c for c in fake.calls if c[:3] == ["pcb", "export", "step"])
    assert step_call == ["pcb", "export", "step", *STEP_EXPORT_FLAGS, "--output", str(out_dir / "parts.step"), pcb_path]
    assert next(c for c in fake.calls if c[:3] == ["pcb", "export", "glb"])[3:3 + len(GLB_EXPORT_FLAGS)] == list(GLB_EXPORT_FLAGS)
    assert [c[c.index("--side") + 1] for c in fake.calls if c[:2] == ["pcb", "render"]] == list(RENDER_VIEWS)
    _no_3d_results(ir)
    # kicad-cli refuses the render (its flags are not measured on 10.0.6): a note, the stale render unregistered, nothing else changes
    ctx.tools["kicad_cli"] = FakeKicad(fail=("render",))
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert out.status is S.NOT_VERIFIED and "kicad_render export failed (3D, not a check): kicad-cli pcb render" in out.message
    assert ArtifactKind.KICAD_RENDER not in ir.artifacts and ArtifactKind.KICAD_STEP in ir.artifacts and ArtifactKind.KICAD_GLB in ir.artifacts
    # a board that is stale with respect to the IR is never exported
    ir.pcb.placements[0].x_mm += 1.0
    out = orch.stages[Stage.MANUFACTURING_OUTPUTS](ir, ctx)
    assert "kicad_step export failed (3D, not a check): PCB artifact was generated from IR" in out.message
    assert not any(kind in ir.artifacts for kind in KICAD_3D_EXPORTS)
    _no_3d_results(ir)


def test_kicad_3d_exporters_refuse_without_a_kicad_cli(tmp_path: Path):
    ir, ctx, orch = _pipeline(tmp_path)
    orch.run(ir, stop_after=Stage.PCB)
    for exporter in (StepExporter(), GlbExporter(), RenderExporter()):
        with pytest.raises(Exception, match=r"3D export needs ctx.tools\['kicad_cli'\]"):
            exporter.compile(ir, CompileContext(workdir=tmp_path, tools={}))


# --------------------------------------------------------------------------- (4) the board figure draws the silkscreen


def _svg(fig) -> ET.Element:
    return ET.fromstring(fig.svg)


def _group(root: ET.Element, cls: str) -> ET.Element | None:
    return next((g for g in root.iter(f"{SVG_NS}g") if g.get("class") == cls), None)


def _class(root: ET.Element, cls: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.get("class") == cls]


def test_board_figure_draws_the_designed_silkscreen_at_its_ir_positions(tmp_path: Path, lib: KicadLibrary):
    ir = _silked(tmp_path, lib)
    fig = board_figure(ir, lib)
    root = _svg(fig)
    silk = _group(root, "silk")
    assert silk is not None and [g.get("class") for g in silk] == ["layer-B_SilkS", "layer-F_SilkS"]
    front = _group(root, "layer-F_SilkS")
    texts = [e for e in front if e.get("class") == "silk-text"]
    assert len(texts) == len(ir.pcb.silkscreen) and {t.get("data-kind") for t in texts} == {"reference", "pin_label", "title"}
    assert all(t.get("fill") == SILK_COLOUR for t in texts) and _group(root, "layer-B_SilkS").findall("*") == []
    # a reference sits where the IR puts it (box centre = anchor for a centred, unrotated text), at its size
    scale = (672 - 2 * 40) / ir.pcb.outline.width_mm
    r1 = next(t for t in ir.pcb.silkscreen if t.component_ref == "R1" and t.kind == SilkKind.REFERENCE)
    drawn = next(t for t in texts if t.get("data-ref") == "R1" and t.get("data-kind") == "reference")
    assert drawn.text == "R1" and drawn.get("transform") == f"translate({40 + r1.x_mm * scale:.2f} {40 + r1.y_mm * scale:.2f})"
    assert float(drawn.get("font-size")) == pytest.approx(r1.size_mm * scale * 1.4, abs=0.01) and drawn.get("data-size-mm") == "1"
    # a justified pin label is centred in its estimated box
    label = next(t for t in ir.pcb.silkscreen if t.kind == SilkKind.PIN_LABEL)
    box = ir_text_box(label)
    cx, cy = (sum(p[0] for p in box.points) / 4, sum(p[1] for p in box.points) / 4)
    assert any(t.get("transform", "").startswith(f"translate({40 + cx * scale:.2f} {40 + cy * scale:.2f})") for t in texts if t.text == label.text)
    # the footprints' own silk lines at their stroke width; the old ref / value labels are gone
    lines = [e for e in front if e.get("class") == "silk-line"]
    assert lines and {e.get("data-ref") for e in lines} >= {"R1", "M1", "J1"} and all(e.get("stroke") == SILK_COLOUR for e in lines)
    assert f"{0.12 * scale:.2f}" in {e.get("stroke-width") for e in lines}
    assert _group(root, "labels").findall("*") == [] and not _class(root, "ref") and not _class(root, "value")  # the group a viewer toggles stays
    assert "실크 문자 7개 (IR 위치)" in fig.caption and "검정 = 실크스크린" in fig.caption and "스트로크 글꼴이 아님" in fig.caption
    assert fig.svg == board_figure(ir, KicadLibrary(roots=lib.roots)).svg
    # the placement figure (no copper) carries the same silk
    assert len(_class(_svg(board_figure(ir, lib, copper=False)), "silk-text")) == len(texts)


def test_board_figure_mirrors_the_back_rotates_and_skips_what_is_not_silk(tmp_path: Path, lib: KicadLibrary):
    ir = silk_ir(tmp_path, lib, [("R1", 10.0, 10.0), ("R2", 25.0, 10.0, 0.0, BoardSide.BOTTOM)], {}, (40.0, 20.0))
    ir.pcb.silkscreen = [
        SilkText(text="R2", x_mm=25.0, y_mm=7.5, layer="B.SilkS", kind=SilkKind.REFERENCE, component_ref="R2", provenance=USER),
        SilkText(text="UP", x_mm=5.0, y_mm=15.0, rotation_deg=90.0, provenance=USER),
        SilkText(text="nope", x_mm=5.0, y_mm=5.0, layer="Edge.Cuts", provenance=USER),  # the compiler refuses it: not drawn
    ]
    root = _svg(fig := board_figure(ir, lib))
    back = _group(root, "layer-B_SilkS")
    r2 = next(e for e in back if e.get("class") == "silk-text")
    assert r2.text == "R2" and r2.get("transform").endswith(" scale(-1 1)") and [e.get("data-ref") for e in back if e.get("class") == "silk-line"] == ["R2", "R2"]
    front = _group(root, "layer-F_SilkS")
    kinds = {e.text: e.get("data-kind") for e in front if e.get("class") == "silk-text"}
    # R1 has no designed reference: its library position is drawn; the rotated user text turns counter-clockwise on screen
    assert kinds == {"R1": "library_reference", "UP": "user"}
    up = next(e for e in front if e.text == "UP")
    assert " rotate(-90.00)" in up.get("transform") and "nope" not in fig.svg
    assert "실크 문자 2개 (IR 위치) + 라이브러리 실크 문자 1개" in fig.caption and "그리지 않은 실크 문자 1개 (컴파일러 거부)" in fig.caption
    # a reference moved to the fab layer is not silk: counted, not drawn
    ir.pcb.silkscreen = [SilkText(text="R1", x_mm=10.0, y_mm=10.0, layer="F.Fab", kind=SilkKind.REFERENCE, component_ref="R1", provenance=USER)]
    fig = board_figure(ir, lib)
    assert "F.Fab 참조 1개 (실크 아님)" in fig.caption and "R1" not in {e.text for e in _class(_svg(fig), "silk-text")}


def test_board_figure_without_designed_silk_draws_the_library_strokes_and_keeps_its_labels(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    fig = board_figure(ir, lib)
    root = _svg(fig)
    assert _class(root, "silk-line") and not _class(root, "silk-text")
    assert _group(root, "labels") is not None and {e.text for e in _class(root, "ref")} == {"R1", "R2", "M1", "J1"}
    assert "검정 선 = 라이브러리 실크" in fig.caption


# --------------------------------------------------------------------------- (4) the circuit report explains the silk and embeds the 3D preview


def _checked(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> None:
    results = default_registry.get(SILK_TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    for r in results:
        r.ir_hash = ir.content_hash()
        ir.validation.add(r)


def test_circuit_report_explains_the_silk_rules_quotes_the_checks_and_embeds_the_3d_preview(tmp_path: Path, lib: KicadLibrary):
    ir = _silked(tmp_path, lib)
    _checked(ir, tmp_path, lib)
    doc = build_stage_document(Stage.PCB, ir, lib, None)
    md = doc.markdown
    silk = md.split("## 실크스크린", 1)[1].split("## 3D 미리보기", 1)[0]
    assert "- 실크 도구: `silkscreen.place` v0.1" in silk
    assert "`silk_to_pad` = 0.15, `silk_to_edge` = 0.3, `gap` = 0.1, `reference_sizes` = 1/0.8, `pin_label` = 0.8, `title` = 1.5" in silk
    assert "- 실크 문자 7개: 참조 4개 (실크 4개, 조립도 층 0개), 커넥터 핀 라벨 2개, 제목 'silk', 사용자 문자 0개" in silk
    assert "### 실크 배치 규칙 (파라미터 값을 넣은 것)" in silk and "+ silk_to_pad = 0.15 mm" in silk and "silk_to_edge = 0.3 mm" in silk
    assert "트랙은 금지 영역이 아닙니다" in silk and "폭 = 글자 수 × 크기 × 0.9 + 굵기, 높이 = 크기 × 1.2 + 굵기" in silk and "kicad-cli DRC 의 몫" in silk
    assert silk.count("| 참조 | `") == 4 and "| 제목 | `silk` |" in silk and "모든 실크 문자의 출처: 계산기 출력 (silkscreen.place v0.1)." in silk
    assert "| 정렬 |\n" in silk  # one shared origin: no per-row column
    for check in (SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK):
        r = ir.validation.latest(check)
        assert f"- `{check}`: **{r.status}** (`pcb.silk` v0.1) — " in silk
    assert "**PASS**" in silk and "**NOT_VERIFIED**" in silk  # clearance / overlap pass, size has no limit (copied, never computed)
    three_d = md.split("## 3D 미리보기", 1)[1].split("## 단계 기록", 1)[0]
    assert three_d.startswith("\n\n![fig](fig:iso3d)\n\n*silk: 3D 미리보기 (등각) — silk: 보드 40 × 20 × 1.6 mm; 부품 상자 0개 (STEP 높이), 평면 외곽선 4개; ")
    assert BODY_CAPTION in three_d and "- 내장 미리보기 파일: 등록되지 않음" in three_d and "KiCad 3D 모델(실제 부품 모양): 등록된 파일 없음" in three_d
    assert three_d.count("| 없음 (평면 외곽선) | courtyard | - | 풋프린트에 (보이는) 3D 모델 참조가 없음 |") == 4
    fig = doc.figures.figures[SLOT_ISO3D]
    assert fig.svg in doc.html and ET.fromstring(fig.svg).tag == f"{SVG_NS}svg" and 'class="caption"' not in fig.svg  # the caption is the figure's
    assert fig.svg == model3d_figure(ir, lib).svg
    assert str(tmp_path) not in md and str(tmp_path) not in doc.html and not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", md)
    assert build_stage_document(Stage.PCB, ir, lib, None).html == doc.html  # deterministic
    # registered 3D outputs are named by file, with their freshness
    preview = Preview3DCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib, MODEL_DIR_TOOL: None}))
    ir.artifacts[ArtifactKind.MODEL_3D] = preview
    (tmp_path / "3d").mkdir()
    step = tmp_path / "3d" / "silk.step"
    step.write_bytes(b"ISO-10303-21;")
    ir.artifacts[ArtifactKind.KICAD_STEP] = ArtifactRef(kind=ArtifactKind.KICAD_STEP, path=str(step), content_hash=hash_file_set([step]),
                                                         generated_from_ir_hash="sha256:" + "1" * 64, files=[str(step)])
    three_d = circuit_report(ir, lib, None).split("## 3D 미리보기", 1)[1]
    assert "- 내장 미리보기 파일: `silk.preview.glb` (산출물 `model_3d`, 현재 IR 에서 생성)" in three_d
    assert "- `kicad_step`: `silk.step` (오래됨 (IR sha256:111111111)" in three_d and "- `kicad_glb`: 등록되지 않음" in three_d
    # texts of different origins keep their origin per row
    ir.pcb.silkscreen.append(SilkText(text="REV A", x_mm=20.0, y_mm=18.0, provenance=USER))
    silk = circuit_report(ir, lib, None).split("## 실크스크린", 1)[1].split("## 3D 미리보기", 1)[0]
    assert "모든 실크 문자의 출처" not in silk and "| 정렬 | 출처 |" in silk and "| 사용자 | `REV A` |" in silk and "사용자 문자 1개" in silk


def test_circuit_report_without_designed_silk_says_where_the_references_are(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    _checked(ir, tmp_path, lib)
    silk = circuit_report(ir, lib, None).split("## 실크스크린", 1)[1].split("## 3D 미리보기", 1)[0]
    assert "IR 에 설계된 실크 문자가 없습니다" in silk and "라이브러리의 기본 위치" in silk and "### 실크 배치 규칙" not in silk
    assert f"- `{SILK_CLEARANCE_CHECK}`: **" in silk


# --------------------------------------------------------------------------- the real KiCad libraries (skip without them)


def test_real_library_preview_uses_the_step_height_of_the_to92_model(tmp_path: Path):
    """With the installed KiCad footprints and the 3D model library ``AI_EDA_TEST_3DMODEL_DIR`` names (only that variable: the test
    never reads a real STEP file unasked, not even the installed ``<share>/3dmodels``): a TO-92 part's box is 7.3 mm high."""
    real = KicadLibrary()
    ref = real.resolve_footprint(LibraryRef(library="Package_TO_SOT_THT", name="TO-92_Inline"))
    named = os.environ.get("AI_EDA_TEST_3DMODEL_DIR")  # this module's autouse fixture hides KICAD10_3DMODEL_DIR: the gate is explicit
    models = Path(named) if named else None
    if not ref.verified or models is None or not (models / "Package_TO_SOT_THT.3dshapes" / "TO-92_Inline.step").is_file():
        pytest.skip("needs the KiCad footprint library and AI_EDA_TEST_3DMODEL_DIR naming the KiCad 3D model library (TO-92_Inline.step)")
    ir = CircuitIR(project=ProjectMeta(id="to92", name="to92"))
    ir.components = [Component(ref="Q1", value="2N3904", footprint=ref, provenance=USER)]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10.0, height_mm=10.0), placements=[Placement(component_ref="Q1", x_mm=4.0, y_mm=5.0, provenance=USER)])
    art, scene = Preview3DCompiler().compile_scene(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": real, MODEL_DIR_TOOL: models}))
    assert [(b.ref, b.height_mm) for b in scene.bodies] == [("Q1", pytest.approx(7.3, abs=1e-6))] and art.matches_disk()
