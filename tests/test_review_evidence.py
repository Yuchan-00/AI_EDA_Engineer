"""Reviewer / repair logic that reads tool evidence - exercised without kicad-cli.

The vertical-slice test proves these paths with the real tool; these tests
pin down the decision rules (what counts as stale, what counts as parity
evidence, how multi-file artifacts are hashed) on any machine.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_eda.errors import ToolExecutionError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, ValidationResult, ValidationStatus, hash_file_set
from ai_eda.repair.strategies import RegenerateArtifact, RerunTool, select_strategy
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.manufacturing.outputs import check_gerber_set, check_output_artifact

S = ValidationStatus


def _check(ir: CircuitIR, area: ReviewArea, tmp_path: Path) -> ValidationResult:
    return next(r for r in IndependentReviewer().review(ir, tmp_path).results if r.check_id == area)


def _single(path: Path, kind: ArtifactKind, ir: CircuitIR, text: str = "x") -> ArtifactRef:
    path.write_text(text, encoding="utf-8")  # CRLF on Windows: hash what is really on disk
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(kind=kind, path=str(path), content_hash=digest, generated_from_ir_hash=ir.content_hash())


def _gerber_file(path: Path, function: str) -> Path:
    path.write_text(f"%TF.GenerationSoftware,KiCad,Pcbnew,10.0.6*%\n%TF.FileFunction,{function}*%\n%TF.FilePolarity,Positive*%\nM02*\n", encoding="utf-8")
    return path


def _gerber_set(tmp_path: Path, ir: CircuitIR, protel: bool = False) -> ArtifactRef:
    out = tmp_path / "gerber"
    out.mkdir(exist_ok=True)
    layers = {"F_Cu": ("Copper,L1,Top", "gtl"), "B_Cu": ("Copper,L2,Bot", "gbl"), "F_Mask": ("Soldermask,Top", "gts"), "B_Mask": ("Soldermask,Bot", "gbs"), "Edge_Cuts": ("Profile,NP", "gm1")}
    files = [_gerber_file(out / f"b-{name}.{ext if protel else 'gbr'}", fn) for name, (fn, ext) in layers.items()]
    job = out / "b-job.gbrjob"
    job.write_text('{"FilesAttributes": [' + ", ".join(f'{{"Path": "{f.name}", "FileFunction": "x"}}' for f in files) + "]}", encoding="utf-8")
    files.append(job)
    return ArtifactRef(kind=ArtifactKind.GERBER, path=str(job), files=[str(f) for f in files], content_hash=hash_file_set(files), generated_from_ir_hash=ir.content_hash())


# --------------------------------------------------------------------------- multi-file artifacts


def test_file_set_hash_is_order_independent_and_location_independent(tmp_path: Path):
    a, b = tmp_path / "a.gbr", tmp_path / "b.gbr"
    a.write_text("A"), b.write_text("B")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "a.gbr").write_text("A"), (other / "b.gbr").write_text("B")
    assert hash_file_set([a, b]) == hash_file_set([b, a]) == hash_file_set([other / "b.gbr", other / "a.gbr"])
    b.write_text("changed")
    assert hash_file_set([a, b]) != hash_file_set([other / "a.gbr", other / "b.gbr"])


def test_multi_file_artifact_matches_disk_covers_every_member(tmp_path: Path, divider_ir: CircuitIR):
    art = _gerber_set(tmp_path, divider_ir)
    assert art.matches_disk()
    Path(art.files[2]).write_text("tampered", encoding="utf-8")
    assert not art.matches_disk()
    Path(art.files[2]).unlink()
    assert not art.matches_disk() and art.disk_hash() is None


# --------------------------------------------------------------------------- output checks


def test_gerber_check_classifies_by_header_not_extension(tmp_path: Path, divider_ir: CircuitIR):
    art = _gerber_set(tmp_path, divider_ir, protel=True)
    files = [Path(f) for f in art.files]
    assert check_gerber_set(files).status is S.PASS
    without_profile = [f for f in files if not f.name.endswith(".gm1")]
    res = check_gerber_set(without_profile)
    assert res.status is S.FAIL and "manifest lists files not in the set" in res.message and "Profile" in res.message
    (tmp_path / "gerber" / "b-job.gbrjob").write_text("not json", encoding="utf-8")
    assert "not a valid gerber job manifest" in check_gerber_set(files).message
    (tmp_path / "gerber" / "b-F_Cu.gtl").write_text("G04 no X2 header*\nM02*\n", encoding="utf-8")
    res = check_gerber_set(files[:-1])
    assert "missing %TF.FileFunction header" in res.message and "Copper,L1" in res.message


def test_output_check_is_stamped_with_the_artifact_hash(tmp_path: Path, divider_ir: CircuitIR):
    art = _gerber_set(tmp_path, divider_ir)
    res = check_output_artifact(art)
    assert res.check_id == "mfg.gerber" and res.status is S.PASS and res.artifact_hash == art.content_hash
    drl = _single(tmp_path / "b.drl", ArtifactKind.DRILL, divider_ir, "M48\n%\nM30\n")
    res = check_output_artifact(drl)
    assert res.check_id == "mfg.drill" and res.status is S.PASS and res.artifact_hash == drl.content_hash
    with pytest.raises(ValueError):
        check_output_artifact(_single(tmp_path / "bom.csv", ArtifactKind.BOM, divider_ir))


# --------------------------------------------------------------------------- review.manufacturing_outputs


def test_manufacturing_outputs_review_rules(tmp_path: Path, divider_ir: CircuitIR):
    ir = divider_ir
    area = ReviewArea.MANUFACTURING_OUTPUTS
    assert _check(ir, area, tmp_path).status is S.NOT_VERIFIED  # nothing exported

    gerber = _gerber_set(tmp_path, ir)
    drill = _single(tmp_path / "b.drl", ArtifactKind.DRILL, ir, "M48\n%\nM30\n")
    ir.artifacts[ArtifactKind.GERBER] = gerber
    r = _check(ir, area, tmp_path)
    assert r.status is S.NOT_VERIFIED and "drill" in r.message  # one of the two is missing
    ir.artifacts[ArtifactKind.DRILL] = drill
    r = _check(ir, area, tmp_path)
    assert r.status is S.NOT_VERIFIED and "mfg.gerber" in r.message and "not run" in r.message

    ir.validation.add(check_output_artifact(gerber))
    ir.validation.add(check_output_artifact(drill))
    assert _check(ir, area, tmp_path).status is S.PASS

    # the drill file was re-exported (new content, artifact re-registered) but mfg.drill still ran on the old one
    ir.artifacts[ArtifactKind.DRILL] = _single(tmp_path / "b.drl", ArtifactKind.DRILL, ir, "M48\n%\nT1C0.8\nM30\n")
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and r.details == {"tool_check": "mfg.drill", "tool_checks": ["mfg.drill"], "repair": "rerun_tool"}
    assert isinstance(select_strategy(r), RerunTool)
    action = RerunTool().apply(ir, r, tmp_path, {})
    assert action.succeeded and ir.validation.latest("mfg.drill").artifact_hash == ir.artifacts[ArtifactKind.DRILL].content_hash
    assert _check(ir, area, tmp_path).status is S.PASS

    # a failing check is FAIL without a deterministic repair
    ir.artifacts[ArtifactKind.DRILL] = _single(tmp_path / "b.drl", ArtifactKind.DRILL, ir, "not excellon")
    ir.validation.add(check_output_artifact(ir.artifacts[ArtifactKind.DRILL]))
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and "no M48 header" in r.message and "repair" not in r.details

    # the design changes: both outputs are stale -> one regenerate action for both kinds
    ir.components[0].value = "22k"
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and r.details["repair"] == "regenerate"
    assert r.details["artifacts"] == [ArtifactKind.GERBER, ArtifactKind.DRILL] and r.details["artifact"] == ArtifactKind.GERBER
    assert RegenerateArtifact().describe(r) == "regenerate gerber, drill from IR"
    action = RegenerateArtifact().apply(ir, r, tmp_path, {"compilers": {}})
    assert not action.succeeded and "no compiler registered for gerber" in action.error and "drill" in action.error


# --------------------------------------------------------------------------- review.schematic_vs_pcb


def test_schematic_vs_pcb_reads_real_parity_evidence(tmp_path: Path, divider_ir: CircuitIR):
    ir = divider_ir
    area = ReviewArea.SCHEMATIC_VS_PCB
    sch = _single(tmp_path / "p.kicad_sch", ArtifactKind.SCHEMATIC, ir, "(kicad_sch)")
    pcb = _single(tmp_path / "p.kicad_pcb", ArtifactKind.PCB, ir, "(kicad_pcb)")
    ir.artifacts[ArtifactKind.SCHEMATIC] = sch
    assert _check(ir, area, tmp_path).status is S.NOT_VERIFIED
    ir.artifacts[ArtifactKind.PCB] = pcb
    assert "not been run" in _check(ir, area, tmp_path).message

    def drc(**details) -> ValidationResult:
        base = {"schematic_parity_checked": True, "schematic_hash": sch.content_hash, "schematic_parity": []}
        return ValidationResult(check_id="kicad.drc", status=S.PASS, tool="kicad-cli", artifact_hash=pcb.content_hash, details={**base, **details})

    ir.validation.add(drc(artifact_hash=None))
    ir.validation.results[-1].artifact_hash = "sha256:other-board"
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and r.details == {"tool_check": "kicad.drc", "repair": "rerun_tool"} and "different board" in r.message

    ir.validation.add(drc(schematic_parity_checked=False, schematic_parity_reason="no schematic artifact"))
    r = _check(ir, area, tmp_path)
    assert r.status is S.NOT_VERIFIED and "no schematic artifact" in r.message

    ir.validation.add(drc(schematic_hash="sha256:older-schematic"))
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and r.details["repair"] == "rerun_tool" and "different schematic" in r.message

    # a parity finding is FAIL even though KiCad rates it a warning, and it is not auto-repairable
    ir.validation.add(drc(schematic_parity=[{"type": "footprint_symbol_mismatch", "severity": "warning", "items": []}]))
    r = _check(ir, area, tmp_path)
    assert r.status is S.FAIL and r.details["repair"] == "human" and "footprint_symbol_mismatch" in r.message

    ir.validation.add(drc())
    r = _check(ir, area, tmp_path)
    assert r.status is S.PASS and "0 issues" in r.message


# --------------------------------------------------------------------------- repair strategies


def test_rerun_tool_reports_missing_inputs_instead_of_raising(tmp_path: Path, divider_ir: CircuitIR):
    f = ValidationResult(check_id=ReviewArea.ERC, status=S.FAIL, details={"repair": "rerun_tool", "tool_check": "kicad.erc"})
    action = RerunTool().apply(divider_ir, f, tmp_path, {})
    assert not action.succeeded and "kicad-cli not available" in action.error
    f = ValidationResult(check_id="x", status=S.FAIL, details={"repair": "rerun_tool", "tool_checks": ["mfg.gerber", "nope"]})
    action = RerunTool().apply(divider_ir, f, tmp_path, {})
    assert not action.succeeded and "no gerber artifact" in action.error and "unknown tool check" in action.error
    assert RerunTool().describe(f) == "re-run mfg.gerber, nope"


def test_regenerate_reports_compile_refusals_instead_of_raising(tmp_path: Path, divider_ir: CircuitIR):
    from ai_eda.compilers import GerberExporter, PCBCompiler

    f = ValidationResult(check_id=ReviewArea.IR_VS_PCB, status=S.FAIL, details={"repair": "regenerate", "artifact": ArtifactKind.PCB})
    action = RegenerateArtifact().apply(divider_ir, f, tmp_path, {"compilers": {ArtifactKind.PCB: PCBCompiler()}})
    assert not action.succeeded and "ir.pcb is None" in action.error
    assert action.ir_hash_before == action.ir_hash_after == divider_ir.content_hash()
    f = ValidationResult(check_id=ReviewArea.MANUFACTURING_OUTPUTS, status=S.FAIL, details={"repair": "regenerate", "artifact": ArtifactKind.GERBER})
    action = RegenerateArtifact().apply(divider_ir, f, tmp_path, {"compilers": {ArtifactKind.GERBER: GerberExporter()}})
    assert not action.succeeded and "kicad_cli" in action.error


def test_gerber_exporter_refuses_stale_or_missing_board(tmp_path: Path, divider_ir: CircuitIR):
    from ai_eda.compilers import CompileContext, GerberExporter
    from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
    from ai_eda.tools.kicad import KicadCli

    with pytest.raises(ToolUnavailableError, match="kicad_cli"):
        GerberExporter().compile(divider_ir, CompileContext(workdir=tmp_path))
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_cli": KicadCli()})
    if not ctx.tools["kicad_cli"].available():
        with pytest.raises(ToolUnavailableError):
            GerberExporter().compile(divider_ir, ctx)
        return
    with pytest.raises(NothingToCompileError, match="no PCB artifact"):
        GerberExporter().compile(divider_ir, ctx)
    divider_ir.artifacts[ArtifactKind.PCB] = _single(tmp_path / "p.kicad_pcb", ArtifactKind.PCB, divider_ir, "(kicad_pcb)")
    divider_ir.artifacts[ArtifactKind.PCB].generated_from_ir_hash = "sha256:older"
    with pytest.raises(CompileError, match="regenerate the board"):
        GerberExporter().compile(divider_ir, ctx)
    divider_ir.artifacts[ArtifactKind.PCB].generated_from_ir_hash = divider_ir.content_hash()
    Path(divider_ir.artifacts[ArtifactKind.PCB].path).write_text("(kicad_pcb edited by hand)", encoding="utf-8")
    with pytest.raises(CompileError, match="does not match its recorded hash"):
        GerberExporter().compile(divider_ir, ctx)
