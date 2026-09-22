from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_eda.errors import CompileError, NotRepairableError, ToolExecutionError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, CircuitIR, ValidationResult
from ai_eda.tools.kicad.cli import KicadCli, fresh_artifact, run_drc_for, run_erc_for
from ai_eda.tools.manufacturing.outputs import OUTPUT_CHECKS, check_output_artifact
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID, run_spice_for

#: Finding categories that must never be auto-fixed (spec section 18).
NON_REPAIRABLE = frozenset(
    {
        "human",  # generic marker set by the reviewer
        "regulatory_conflict",
        "unverified_component_data",
        "fab_capability_shortfall",
        "circuit_structure_change",
    }
)


class RepairAction(BaseModel):
    strategy: str
    finding_check_id: str
    description: str
    #: hash of the IR before/after - should be identical for artifact-only repairs
    ir_hash_before: str
    ir_hash_after: str | None = None
    succeeded: bool = False
    error: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RepairStrategy(ABC):
    id: str = "repair.base"

    @abstractmethod
    def can_repair(self, finding: ValidationResult) -> bool: ...

    @abstractmethod
    def apply(self, ir: CircuitIR, finding: ValidationResult, workdir: Path, tools: dict[str, Any]) -> RepairAction: ...

    def describe(self, finding: ValidationResult) -> str:
        """The deterministic action ``apply`` would take for ``finding``.

        Two findings with the same description ask for the same action, so the
        loop runs it once per iteration. The default is unique per finding.
        """
        return f"{self.id} for {finding.check_id}"


def _listed(details: dict, singular: str, plural: str) -> list[str]:
    """``details[plural]`` if present, else ``[details[singular]]`` - the reviewer may name several at once."""
    if details.get(plural):
        return [str(x) for x in details[plural]]
    return [str(details[singular])]


class RegenerateArtifact(RepairStrategy):
    """Regenerate derived artifacts (BOM, CPL, schematic, PCB, gerber set, ...) from the IR.

    A compiler that refuses (CompileError) or a missing tool is a failed
    action, reported as such; the IR is never touched.
    """

    id = "repair.regenerate_artifact"

    def can_repair(self, finding: ValidationResult) -> bool:
        return finding.details.get("repair") == "regenerate" and ("artifact" in finding.details or "artifacts" in finding.details)

    @staticmethod
    def _kinds(finding: ValidationResult) -> list[ArtifactKind]:
        return [ArtifactKind(k) for k in _listed(finding.details, "artifact", "artifacts")]

    def describe(self, finding: ValidationResult) -> str:
        return "regenerate " + ", ".join(str(k) for k in self._kinds(finding)) + " from IR"

    def apply(self, ir: CircuitIR, finding: ValidationResult, workdir: Path, tools: dict[str, Any]) -> RepairAction:
        kinds = self._kinds(finding)
        action = RepairAction(strategy=self.id, finding_check_id=finding.check_id, description=self.describe(finding), ir_hash_before=ir.content_hash())
        compilers = tools.get("compilers", {})
        from ai_eda.compilers.base import CompileContext

        errors: list[str] = []
        for kind in kinds:
            compiler = compilers.get(kind)
            if compiler is None:
                errors.append(f"no compiler registered for {kind}")
                continue
            try:
                ir.artifacts[kind] = compiler.compile(ir, CompileContext(workdir=workdir, tools=tools))
            except (NotImplementedError, CompileError, ToolUnavailableError, ToolExecutionError) as e:
                errors.append(f"{kind}: {e}")
        action.succeeded = not errors
        action.error = "; ".join(errors) or None
        action.ir_hash_after = ir.content_hash()
        return action


class RerunTool(RepairStrategy):
    """Re-run an external check (ERC/DRC, SPICE, gerber/drill format check) whose report is stale.

    The check only runs on an artifact that is fresh with respect to the IR
    and unchanged on disk (:func:`~ai_eda.tools.kicad.cli.fresh_artifact`);
    otherwise the action fails and the loop has to regenerate first - a
    check of stale or hand-edited files must never become evidence about the
    current design. ``spice`` re-runs every analysis and expectation through
    :func:`~ai_eda.tools.spice.stage.run_spice_for`, exactly as the SPICE
    stage did.
    """

    id = "repair.rerun_tool"

    def can_repair(self, finding: ValidationResult) -> bool:
        return finding.details.get("repair") == "rerun_tool" and ("tool_check" in finding.details or "tool_checks" in finding.details)

    def describe(self, finding: ValidationResult) -> str:
        return "re-run " + ", ".join(_listed(finding.details, "tool_check", "tool_checks"))

    def apply(self, ir: CircuitIR, finding: ValidationResult, workdir: Path, tools: dict[str, Any]) -> RepairAction:
        checks = _listed(finding.details, "tool_check", "tool_checks")
        action = RepairAction(strategy=self.id, finding_check_id=finding.check_id, description=self.describe(finding), ir_hash_before=ir.content_hash())
        errors: list[str] = []
        for check in checks:
            try:
                ir.validation.extend(self._rerun(ir, check, workdir, tools))
            except (KeyError, ToolUnavailableError, ToolExecutionError, ValueError) as e:
                errors.append(f"{check}: {e!r}")
        action.succeeded = not errors
        action.error = "; ".join(errors) or None
        action.ir_hash_after = ir.content_hash()
        return action

    @staticmethod
    def _rerun(ir: CircuitIR, check: str, workdir: Path, tools: dict[str, Any]) -> list[ValidationResult]:
        output_kinds = {check_id: kind for kind, check_id in OUTPUT_CHECKS.items()}
        if check not in output_kinds and check not in ("kicad.erc", "kicad.drc", SPICE_CHECK_ID):
            raise ValueError(f"unknown tool check {check}")
        if check in output_kinds:
            art = fresh_artifact(ir, output_kinds[check])
            res = check_output_artifact(art, ir)
            res.ir_hash = ir.content_hash()
            return [res]
        if check == SPICE_CHECK_ID:
            # ToolUnavailableError without an engine, ToolExecutionError when the netlist is stale
            return run_spice_for(ir, tools, workdir)
        kicad = tools.get("kicad_cli")
        if not isinstance(kicad, KicadCli) or not kicad.available():
            raise ToolUnavailableError("kicad-cli not available")
        if check == "kicad.erc":
            if ArtifactKind.SCHEMATIC not in ir.artifacts:
                raise ToolExecutionError("no schematic artifact to check")
            return [run_erc_for(ir, kicad, workdir)]
        if ArtifactKind.PCB not in ir.artifacts:
            raise ToolExecutionError("no PCB artifact to check")
        return [run_drc_for(ir, kicad, workdir)]


DEFAULT_STRATEGIES: list[RepairStrategy] = [RegenerateArtifact(), RerunTool()]


def select_strategy(finding: ValidationResult, strategies: list[RepairStrategy] | None = None) -> RepairStrategy:
    if finding.details.get("repair") in NON_REPAIRABLE:
        raise NotRepairableError(f"{finding.check_id}: requires human decision ({finding.message})")
    for s in strategies or DEFAULT_STRATEGIES:
        if s.can_repair(finding):
            return s
    raise NotRepairableError(f"{finding.check_id}: no deterministic strategy ({finding.message})")
