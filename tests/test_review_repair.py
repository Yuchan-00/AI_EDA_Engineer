from pathlib import Path

from ai_eda.compilers import BOMCompiler, CPLCompiler, CompileContext
from ai_eda.ir import ArtifactKind, CircuitIR, ValidationResult, ValidationStatus, llm_generated
from ai_eda.repair import NON_REPAIRABLE, RepairLoop, RepairStrategy, RepairAction, select_strategy
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.errors import NotRepairableError
from tests.conftest import make_component

S = ValidationStatus


def _tools() -> dict:
    return {"compilers": {ArtifactKind.BOM: BOMCompiler(), ArtifactKind.CPL: CPLCompiler()}}


def test_review_covers_all_14_areas(divider_ir: CircuitIR, tmp_path: Path):
    report = IndependentReviewer().review(divider_ir, tmp_path)
    assert [r.check_id for r in report.results] == [a.value for a in ReviewArea]
    assert all(r.ir_hash == divider_ir.content_hash() for r in report.results)


def test_bom_marks_unverified_identity(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[1].mpn = llm_generated("GUESS", model="m")
    art = BOMCompiler().compile(divider_ir, CompileContext(workdir=tmp_path))
    text = Path(art.path).read_text()
    assert "MPN-10k" in text and "NOT_VERIFIED" in text
    assert art.generated_from_ir_hash == divider_ir.content_hash()


def test_stale_bom_is_detected_and_regenerated(divider_ir: CircuitIR, tmp_path: Path):
    tools = _tools()
    divider_ir.artifacts[ArtifactKind.BOM] = tools["compilers"][ArtifactKind.BOM].compile(divider_ir, CompileContext(workdir=tmp_path))
    # design changes after the BOM was produced
    divider_ir.components.append(make_component("R3", "1k"))

    report = IndependentReviewer().review(divider_ir, tmp_path)
    bom = next(r for r in report.results if r.check_id == ReviewArea.PCB_VS_BOM)
    assert bom.status == S.FAIL and bom.details["repair"] == "regenerate"

    outcome = RepairLoop(tools=tools).run(divider_ir, tmp_path)
    assert outcome.iterations == 1
    assert [a.succeeded for a in outcome.actions] == [True]
    assert outcome.actions[0].ir_hash_before == outcome.actions[0].ir_hash_after  # artifact-only repair
    final = next(r for r in outcome.final_review.results if r.check_id == ReviewArea.PCB_VS_BOM)
    # the regenerated BOM matches the IR again, but "PCB vs BOM" cannot be PASS without a board to compare with
    assert final.status == S.NOT_VERIFIED and "no PCB artifact" in final.message and "3 rows" in final.message
    assert final.evidence[0].path == divider_ir.artifacts[ArtifactKind.BOM].path and final.evidence[0].content_hash
    assert "R3" in Path(divider_ir.artifacts[ArtifactKind.BOM].path).read_text()


def test_non_repairable_findings_are_reported_not_fixed(divider_ir: CircuitIR, tmp_path: Path):
    # requirement with no component serving it -> FAIL with repair=human
    from ai_eda.ir import Requirement, RequirementKind

    divider_ir.requirements.requirements.append(Requirement(id="req.efficiency", key="efficiency", text="eff > 90%", kind=RequirementKind.EXPLICIT))
    outcome = RepairLoop(tools=_tools()).run(divider_ir, tmp_path)
    assert outcome.actions == []
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.REQUIREMENTS_VS_IR]
    assert "human" in outcome.unresolved[0].message
    assert "no repairable" in outcome.stopped_reason


def test_select_strategy_refuses_non_repairable():
    for cat in NON_REPAIRABLE:
        f = ValidationResult(check_id="x", status=S.FAIL, details={"repair": cat})
        try:
            select_strategy(f)
        except NotRepairableError:
            continue
        raise AssertionError(f"{cat} should not be repairable")


def test_oscillation_detection(divider_ir: CircuitIR, tmp_path: Path):
    class NoOp(RepairStrategy):
        id = "noop"

        def can_repair(self, finding):
            return finding.details.get("repair") == "regenerate"

        def apply(self, ir, finding, workdir, tools):
            return RepairAction(strategy=self.id, finding_check_id=finding.check_id, description="pretend", ir_hash_before=ir.content_hash(), succeeded=True)

    divider_ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(divider_ir, CompileContext(workdir=tmp_path))
    divider_ir.components.append(make_component("R3", "1k"))  # BOM now stale
    outcome = RepairLoop(tools={}, strategies=[NoOp()], max_iterations=10).run(divider_ir, tmp_path)
    assert "oscillation" in outcome.stopped_reason
    assert outcome.iterations == 1


def test_max_iterations_cap(divider_ir: CircuitIR, tmp_path: Path):
    outcome = RepairLoop(tools={}, max_iterations=0).run(divider_ir, tmp_path)
    assert outcome.iterations == 0
