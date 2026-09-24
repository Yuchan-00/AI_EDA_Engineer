"""BOM free-text cell neutralisation: ``Value`` / ``Description`` cells that spreadsheet software would execute are
written with a leading apostrophe and the alteration is reported; every other cell keeps the refusal.

Runs without KiCad: the reviewer comparison is exercised against a fake fresh board (``read_board_footprints``
monkeypatched); the real-library path of ``tests/test_findings_regressions.py`` covers the same comparison on the
Windows PC.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import BOMCompiler, CompileContext, CPLCompiler
from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, PCBDesign, Placement, ProjectMeta, SourceRef, ValidationStatus as S, authoritative
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.review import reviewer as reviewer_module
from ai_eda.tools.kicad.board import BoardFootprint
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.csv_cells import TEXT_PREFIX, bom_cell_text, free_text_cell, unsafe_cell
from ai_eda.workflow import Orchestrator, Stage
from tests.conftest import AUTH, DS, make_component

HEADER = "Reference,Value,Description,Manufacturer,MPN,Package,Footprint,Supplier,SupplierPN,DatasheetHash,Qty\n"
ANSWERS = {"application": "test", "jurisdiction": "EU"}


def _ir(tmp_path: Path, *components) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    ir.components = list(components)
    return ir


def _compile(ir: CircuitIR, workdir: Path) -> ArtifactRef:
    return BOMCompiler().compile(ir, CompileContext(workdir=workdir, tools={}))


def _rows(art: ArtifactRef) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(Path(art.path).read_text(encoding="utf-8"))))


# --------------------------------------------------------------------------- the encoding


def test_encoding_is_injective_and_bom_cell_text_is_its_inverse():
    for text in ("-5V", "+3V3", "=1+1", "@cmd", "'-5V", "''x", "'", "10k", "", " -1", "  '-5V", "a'b"):
        cell, note = free_text_cell(text, "R1.Value")
        assert bom_cell_text(cell) == text, text
        assert unsafe_cell(cell.strip()) is None, cell
        assert (note is None) == (cell == text)
    # distinct texts never share a cell (the apostrophe is prefixed to a text that already starts with it)
    assert free_text_cell("-5V", "x")[0] != free_text_cell("'-5V", "x")[0]
    assert free_text_cell("'-5V", "x")[0] == TEXT_PREFIX + "'-5V"
    assert bom_cell_text("''x") == "'x" and bom_cell_text("x") == "x"


def test_value_and_description_formula_cells_are_neutralised_with_exact_bytes(tmp_path: Path):
    r3 = make_component("R3", "10k")
    r3.description = "@cmd"
    art = _compile(_ir(tmp_path, make_component("R1", "=1+1"), make_component("R2", "-5V"), r3), tmp_path)
    assert Path(art.path).read_bytes() == (
        HEADER
        + "R1,'=1+1,resistor,NOT_VERIFIED,MPN-=1+1,0603,Resistor_SMD:R_0603_1608Metric,NOT_VERIFIED,NOT_VERIFIED,NOT_VERIFIED,1\n"
        + "R2,'-5V,resistor,NOT_VERIFIED,MPN--5V,0603,Resistor_SMD:R_0603_1608Metric,NOT_VERIFIED,NOT_VERIFIED,NOT_VERIFIED,1\n"
        + "R3,10k,'@cmd,NOT_VERIFIED,MPN-10k,0603,Resistor_SMD:R_0603_1608Metric,NOT_VERIFIED,NOT_VERIFIED,NOT_VERIFIED,1\n"
    ).encode("utf-8")
    assert art.generator_version == BOMCompiler.version == "0.2" and art.generator == "compiler.bom"
    assert len(art.notes) == 3 and [n.split(":", 1)[0] for n in art.notes] == ["R1.Value", "R2.Value", "R3.Description"]
    assert all("apostrophe" in n and "formula character" in n for n in art.notes)
    assert "spreadsheet software reads it as text" not in " ".join(art.notes)  # the note says only what is checked
    rows = _rows(art)
    assert rows[0]["Value"] == "'=1+1" and rows[1]["Value"] == "'-5V" and rows[2]["Description"] == "'@cmd"
    assert [bom_cell_text(r["Value"]) for r in rows] == ["=1+1", "-5V", "10k"]


def test_every_written_free_text_cell_reads_back_as_plain_text(tmp_path: Path):
    texts = ["-12V", "+5V", "=SUM(A1)", "@x", "'quoted", "-2+3|cmd", " +1", "plain", "", "a'b", "10k"]
    parts = []
    for i, t in enumerate(texts):
        c = make_component(f"R{i + 1}", t)
        c.description = texts[-1 - i]
        parts.append(c)
    art = _compile(_ir(tmp_path, *parts), tmp_path)
    rows = _rows(art)
    assert len(rows) == len(texts)
    for row in rows:
        for col in ("Value", "Description"):
            assert unsafe_cell(row[col].strip()) is None, (row["Reference"], col, row[col])
    # and the decoder gives the design's words back, row by row
    by_ref = {r["Reference"]: r for r in rows}
    for c in parts:
        assert bom_cell_text(by_ref[c.ref]["Value"]) == c.value and bom_cell_text(by_ref[c.ref]["Description"]) == c.description
    assert len(art.notes) == sum(1 for c in parts for t in (c.value, c.description) if unsafe_cell(t.strip()) or t[:1] == TEXT_PREFIX)


def test_plain_cells_carry_no_note_and_bytes_are_unchanged(tmp_path: Path):
    art = _compile(_ir(tmp_path, make_component("R1", "10k")), tmp_path)
    assert art.notes == []
    row = Path(art.path).read_text(encoding="utf-8").splitlines()[1]
    assert row == "R1,10k,resistor,NOT_VERIFIED,MPN-10k,0603,Resistor_SMD:R_0603_1608Metric,NOT_VERIFIED,NOT_VERIFIED,NOT_VERIFIED,1"


def test_neutralisation_is_deterministic_and_hash_neutral(tmp_path: Path):
    ir = _ir(tmp_path, make_component("R1", "-5V"))
    before = ir.content_hash()
    a = _compile(ir, tmp_path / "a")
    b = _compile(ir, tmp_path / "b")
    saved = ir.save(tmp_path / "ir.json")
    c = _compile(CircuitIR.load(saved), tmp_path / "c")
    assert Path(a.path).read_bytes() == Path(b.path).read_bytes() == Path(c.path).read_bytes()
    assert a.content_hash == b.content_hash == c.content_hash and a.notes == b.notes == c.notes and a.notes
    assert ir.content_hash() == before  # notes live on the artifact, outside the design view
    ir.artifacts[ArtifactKind.BOM] = a
    assert ir.content_hash() == before
    # the notes survive the JSON round trip of the reference and of the whole IR
    assert ArtifactRef.model_validate_json(a.model_dump_json()).notes == a.notes
    ir.save(tmp_path / "ir2.json")
    assert CircuitIR.load(tmp_path / "ir2.json").artifacts[ArtifactKind.BOM].notes == a.notes


def test_cpl_numeric_cells_are_never_neutralised(tmp_path: Path):
    ir = _ir(tmp_path, make_component("R1", "10k"))
    ir.pcb = PCBDesign(placements=[Placement(component_ref="R1", x_mm=-1.0, y_mm=0.0, rotation_deg=-90, provenance=AUTH)])
    art = CPLCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))
    assert Path(art.path).read_text(encoding="utf-8").splitlines()[1] == "R1,-1.0000mm,0.0000mm,-90,Top"
    assert art.notes == []
    ir.pcb = PCBDesign(placements=[Placement(component_ref="-R1", x_mm=0.0, y_mm=0.0, provenance=AUTH)])
    with pytest.raises(CompileError, match="Designator.*refusing to write a cell"):
        CPLCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))


# --------------------------------------------------------------------------- refusals still win


def test_identity_refusal_outranks_free_text_neutralisation(tmp_path: Path):
    part = make_component("R1", "=1+1")
    part.mpn = authoritative("=1+1", DS)
    with pytest.raises(CompileError, match="MPN.*refusing"):
        _compile(_ir(tmp_path, part), tmp_path)
    assert not (tmp_path / "bom.csv").exists()


def test_datasheet_hash_cell_is_refused_not_neutralised(tmp_path: Path):
    part = make_component("R1", "10k")
    part.mpn = authoritative("RC0603", SourceRef(title="ds", content_hash="=1+1", document_path=str(tmp_path / "ds.pdf")))
    with pytest.raises(CompileError, match="R1.DatasheetHash.*refusing"):
        _compile(_ir(tmp_path, part), tmp_path)
    assert not (tmp_path / "bom.csv").exists()


@pytest.mark.parametrize(
    "value, description, cell",
    [
        ("10k\x00", "resistor", "R1.Value"),
        ("10k", "a\rb", "R1.Description"),
        ("10k", "line1\nline2", "R1.Description"),  # a newline would open a new CSV row that may start with '='
        ("\t=1+1", "resistor", "R1.Value"),  # the tab is a control character even though the stripped check would neutralise it
        ("10k\x7f", "resistor", "R1.Value"),
    ],
)
def test_control_character_in_free_text_is_refused(tmp_path: Path, value: str, description: str, cell: str):
    part = make_component("R1", value)
    part.description = description
    with pytest.raises(CompileError, match=f"{cell}.*control character"):
        _compile(_ir(tmp_path, part), tmp_path)
    assert not (tmp_path / "bom.csv").exists()
    with pytest.raises(ValueError, match="control character"):
        free_text_cell(value + description, cell)


# --------------------------------------------------------------------------- the reviewer decodes the cell


def _fake_board(ir: CircuitIR, workdir: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    """A fresh, unchanged PCB artifact whose footprints carry ``values`` (no KiCad needed)."""
    path = workdir / "fake.kicad_pcb"
    path.write_text("(kicad_pcb)", encoding="utf-8")
    art = ArtifactRef(kind=ArtifactKind.PCB, path=str(path), generated_from_ir_hash=ir.content_hash(), generator="test")
    art.content_hash = art.disk_hash()
    ir.artifacts[ArtifactKind.PCB] = art
    fps = [BoardFootprint(ref=ref, value=v, lib_id="Resistor_SMD:R_0603_1608Metric", x=0.0, y=0.0, rotation=0.0, layer="F.Cu") for ref, v in values.items()]
    monkeypatch.setattr(reviewer_module, "read_board_footprints", lambda p: fps)


def _review(ir: CircuitIR, workdir: Path):
    return next(r for r in IndependentReviewer().review(ir, workdir).results if r.check_id == ReviewArea.PCB_VS_BOM)


def test_reviewer_compares_value_through_the_bom_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir = _ir(tmp_path, make_component("R1", "-5V"), make_component("R2", "10k"))
    ir.artifacts[ArtifactKind.BOM] = _compile(ir, tmp_path)
    assert _rows(ir.artifacts[ArtifactKind.BOM])[0]["Value"] == "'-5V"
    _fake_board(ir, tmp_path, monkeypatch, {"R1": "-5V", "R2": "10k"})
    r = _review(ir, tmp_path)
    assert r.status is S.PASS and "value" in r.message, r.message
    assert [e.path for e in r.evidence] == [ir.artifacts[ArtifactKind.BOM].path, ir.artifacts[ArtifactKind.PCB].path] and all(e.content_hash for e in r.evidence)
    # a real mismatch: the design's words in the text, the cell named because it differs from them
    _fake_board(ir, tmp_path, monkeypatch, {"R1": "-6V", "R2": "10k"})
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details["repair"] == "human"
    assert r.details["mismatches"] == ["R1: value BOM '-5V' vs board '-6V' (BOM cell \"'-5V\")"]
    # a board whose value is literally the encoded cell is not the design's value
    _fake_board(ir, tmp_path, monkeypatch, {"R1": "'-5V", "R2": "10k"})
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details["mismatches"] == ["R1: value BOM '-5V' vs board \"'-5V\" (BOM cell \"'-5V\")"]
    # a plain mismatch keeps the unencoded wording (the needs_libs regression asserts this substring)
    _fake_board(ir, tmp_path, monkeypatch, {"R1": "-5V", "R2": "47k"})
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details["mismatches"] == ["R2: value BOM '10k' vs board '47k'"]
    # a control character in the board's value is a mismatch, never an exception
    _fake_board(ir, tmp_path, monkeypatch, {"R1": "a\x00", "R2": "10k"})
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details["mismatches"] == ["R1: value BOM '-5V' vs board 'a\\x00' (BOM cell \"'-5V\")"]


# --------------------------------------------------------------------------- the pipeline reports it


def test_orchestrator_reports_neutralised_cells(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[0].value = "-5V"
    # an empty library root: this run is about a design without a board on every machine (with KiCad libraries PLACEMENT would place R1/R2)
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=ANSWERS, tools={"kicad_library": KicadLibrary(roots=[tmp_path / "nolib"])})).run(divider_ir)
    assert state.outcome(Stage.PLACEMENT).status is S.NOT_VERIFIED and "not placed" in state.outcome(Stage.PLACEMENT).message and divider_ir.pcb is None
    out = state.outcome(Stage.MANUFACTURING_OUTPUTS)
    assert out.status is S.NOT_VERIFIED and "no PCB" in out.message, out.message  # no board here: the stage stays unverified
    assert "bom: 1 cell(s) neutralised (R1.Value)" in out.message
    art = divider_ir.artifacts[ArtifactKind.BOM]
    assert art.notes and art.notes[0].startswith("R1.Value:")
    res = divider_ir.validation.latest("compile.bom")
    assert res.status is S.PASS and res.tool == "compiler.bom" and res.tool_version == BOMCompiler.version
    assert res.artifact_hash == art.content_hash and res.ir_hash == divider_ir.content_hash()
    assert res.details["neutralised"] == art.notes and len(art.notes) == 1
    assert res.message.startswith(f"compiled {art.path}") and "1 cell(s) neutralised (R1.Value)" in res.message
    assert "apostrophe" not in res.message  # the full note is in the details, the message stays one line
    # the CPL result carries no such details
    cpl = divider_ir.validation.latest("compile.cpl")
    assert cpl.status is S.PASS and "neutralised" not in cpl.details
    assert state.outcome(Stage.RELEASE) is not None
