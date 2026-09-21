from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import CompileContext, Compiler
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, ProjectMeta, ValidationStatus
from ai_eda.workflow import Orchestrator, Stage


def test_empty_project_blocks_on_baseline_questions(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    state = Orchestrator(AgentContext(workdir=tmp_path)).run(ir)
    assert state.blocked
    assert state.current == Stage.REQUIREMENT_ANALYSIS
    assert {q.key for q in state.open_questions} == {"application", "jurisdiction"}


def test_answers_become_requirements_and_jurisdiction(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ctx = AgentContext(workdir=tmp_path, answers={"application": "bench supply", "jurisdiction": "EU, KR"})
    state = Orchestrator(ctx).run(ir, stop_after=Stage.REGULATORY_RESEARCH)
    assert not state.blocked
    assert ir.requirements.get("application").value.value == "bench supply"
    assert [j.code for j in ir.regulatory.jurisdictions] == ["EU", "KR"]
    reg = next(o for o in state.outcomes if o.stage == Stage.REGULATORY_RESEARCH)
    assert reg.status == ValidationStatus.NOT_VERIFIED  # research pipeline not implemented, never PASS


def test_full_run_never_releases_unverified_design(divider_ir: CircuitIR, tmp_path: Path):
    ctx = AgentContext(workdir=tmp_path, answers={"application": "test", "jurisdiction": "EU"})
    state = Orchestrator(ctx).run(divider_ir)
    assert not state.blocked
    release = state.outcomes[-1]
    assert release.stage == Stage.RELEASE
    assert release.status != ValidationStatus.PASS
    # divider_ir has no ir.pcb: the PCB stage has nothing to lay out -> NOT_VERIFIED (not FAIL, not a crash)
    pcb = state.outcome(Stage.PCB)
    assert pcb.status == ValidationStatus.NOT_VERIFIED and "ir.pcb is None" in pcb.message
    assert ArtifactKind.PCB not in divider_ir.artifacts
    assert state.outcome(Stage.DRC).status == ValidationStatus.NOT_VERIFIED
    assert state.outcome(Stage.MANUFACTURING_OUTPUTS).status == ValidationStatus.NOT_VERIFIED
    assert "no PCB" in state.outcome(Stage.MANUFACTURING_OUTPUTS).message
    assert ArtifactKind.BOM in divider_ir.artifacts and ArtifactKind.GERBER not in divider_ir.artifacts


def test_empty_ir_compiles_nothing_and_is_not_verified(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="empty", name="empty", workdir=str(tmp_path)))
    ctx = AgentContext(workdir=tmp_path, answers={"application": "x", "jurisdiction": "EU"})
    state = Orchestrator(ctx).run(ir)
    for stage in (Stage.SCHEMATIC, Stage.ERC, Stage.PCB, Stage.DRC, Stage.MANUFACTURING_OUTPUTS):
        assert state.outcome(stage).status == ValidationStatus.NOT_VERIFIED, stage
    assert "no components" in state.outcome(Stage.SCHEMATIC).message
    assert not (tmp_path / "empty.kicad_sch").exists()  # an empty schematic that ERC would vacuously pass is never written
    assert state.outcomes[-1].status != ValidationStatus.PASS


def test_inconsistent_ir_fails_the_compile_stage_with_the_reason(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[0].symbol.verified = False  # claims a symbol that was never resolved against a library
    ctx = AgentContext(workdir=tmp_path, answers={"application": "test", "jurisdiction": "EU"})
    state = Orchestrator(ctx).run(divider_ir, stop_after=Stage.ERC)
    sch = state.outcome(Stage.SCHEMATIC)
    assert sch.status == ValidationStatus.FAIL
    assert "R1" in sch.message and "not verified" in sch.message
    assert ArtifactKind.SCHEMATIC not in divider_ir.artifacts
    assert state.outcome(Stage.ERC).status == ValidationStatus.NOT_VERIFIED


def test_stale_artifact_is_dropped_when_compile_is_refused(divider_ir: CircuitIR, tmp_path: Path):
    old = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path=str(tmp_path / "old.kicad_sch"), content_hash="sha256:old", generated_from_ir_hash="sha256:older")
    divider_ir.artifacts[ArtifactKind.SCHEMATIC] = old
    divider_ir.components[1].pins = divider_ir.components[1].pins[:1]  # R2 now claims one pin; Device:R has two
    ctx = AgentContext(workdir=tmp_path, answers={"application": "test", "jurisdiction": "EU"})
    state = Orchestrator(ctx).run(divider_ir, stop_after=Stage.SCHEMATIC)
    assert state.outcome(Stage.SCHEMATIC).status == ValidationStatus.FAIL
    assert ArtifactKind.SCHEMATIC not in divider_ir.artifacts  # the stale file is not left as evidence


def test_unexpected_compiler_exceptions_propagate(divider_ir: CircuitIR, tmp_path: Path):
    class Broken(Compiler):
        id = "compiler.broken"
        kind = ArtifactKind.SCHEMATIC

        def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
            raise RuntimeError("defect, not a verdict")

    ctx = AgentContext(workdir=tmp_path, answers={"application": "test", "jurisdiction": "EU"})
    orch = Orchestrator(ctx)
    ctx.tools["compilers"][ArtifactKind.SCHEMATIC] = Broken()
    with pytest.raises(RuntimeError, match="defect"):
        orch.run(divider_ir)


def test_compile_error_classes_map_to_statuses(divider_ir: CircuitIR, tmp_path: Path):
    class Refuses(Compiler):
        id = "compiler.refuses"
        kind = ArtifactKind.PCB
        exc: type[Exception] = CompileError

        def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
            raise self.exc("why")

    ctx = AgentContext(workdir=tmp_path)
    orch = Orchestrator(ctx)
    for exc, expected in ((NothingToCompileError, ValidationStatus.NOT_VERIFIED), (NotImplementedError, ValidationStatus.NOT_VERIFIED), (CompileError, ValidationStatus.FAIL)):
        Refuses.exc = exc
        ctx.tools["compilers"][ArtifactKind.PCB] = Refuses()
        status, message = orch._compile(divider_ir, ctx, ArtifactKind.PCB)
        assert (status, "why" in message) == (expected, True), exc
