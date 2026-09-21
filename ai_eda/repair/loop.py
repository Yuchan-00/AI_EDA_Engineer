"""Closed-loop review -> repair -> re-review with iteration cap and oscillation detection.

Invariant (spec 18, CLAUDE.md #5): a repair never changes the design. The
loop hashes the IR around every action; a strategy whose action leaves a
different IR hash is recorded as a failed action ("mutated the IR"), the
loop stops with that reason, and the finding stays unresolved. Every attempt
is kept in :attr:`RepairOutcome.actions` (strategy, description, hashes,
error, timestamp) and :class:`~ai_eda.agents.repair.RepairAgent` persists
the whole outcome as the ``repair.loop`` validation result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_eda.errors import NotRepairableError
from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.repair.strategies import DEFAULT_STRATEGIES, RepairAction, RepairStrategy, select_strategy
from ai_eda.review import IndependentReviewer, ReviewReport

MUTATION_STOP = "repair strategy mutated the IR"


class RepairOutcome(BaseModel):
    iterations: int = 0
    actions: list[RepairAction] = Field(default_factory=list)
    unresolved: list[ValidationResult] = Field(default_factory=list)
    stopped_reason: str = ""
    final_review: ReviewReport

    def summary(self) -> str:
        ok = sum(1 for a in self.actions if a.succeeded)
        return f"repair: {self.iterations} iteration(s), {ok}/{len(self.actions)} action(s) succeeded, {len(self.unresolved)} unresolved; stopped: {self.stopped_reason}"

    @property
    def mutated_ir(self) -> bool:
        return any(a.ir_hash_after is not None and a.ir_hash_after != a.ir_hash_before for a in self.actions)

    def as_validation_result(self, ir_hash: str, tool_version: str) -> ValidationResult:
        """The outcome as a tool-backed ``repair.loop`` result so ``ir.json`` keeps every attempt.

        FAIL when findings stay unresolved or a strategy mutated the IR,
        NOT_APPLICABLE when there was nothing to repair, PASS when every
        action succeeded and the final review has no failures.
        """
        if self.unresolved or self.mutated_ir:
            status = ValidationStatus.FAIL
        elif not self.actions:
            status = ValidationStatus.NOT_APPLICABLE
        else:
            status = ValidationStatus.PASS
        return ValidationResult(
            check_id="repair.loop",
            status=status,
            message=self.summary(),
            tool="repair.loop",
            tool_version=tool_version,
            ir_hash=ir_hash,
            details={
                "iterations": self.iterations,
                "stopped_reason": self.stopped_reason,
                "actions": [a.model_dump(mode="json") for a in self.actions],
                "unresolved": [{"check_id": u.check_id, "message": u.message, "details": u.details} for u in self.unresolved],
                "final_review": self.final_review.summary(),
            },
        )


class RepairLoop:
    version = "0.2"

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        strategies: list[RepairStrategy] | None = None,
        max_iterations: int = 5,
        reviewer: IndependentReviewer | None = None,
    ) -> None:
        self.tools = tools or {}
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.max_iterations = max_iterations
        self.reviewer = reviewer or IndependentReviewer(tools=self.tools)

    def run(self, ir: CircuitIR, workdir: Path) -> RepairOutcome:
        actions: list[RepairAction] = []
        unresolved: dict[str, ValidationResult] = {}
        #: fingerprint of failing check ids + artifact hashes per iteration, for oscillation detection
        seen_states: list[str] = []
        report = self.reviewer.review(ir, workdir)
        iterations = 0
        stopped = "no failures"

        mutated = False
        while report.failures:
            if iterations >= self.max_iterations:
                stopped = f"max iterations ({self.max_iterations}) reached"
                break
            fingerprint = self._fingerprint(ir, report)
            if fingerprint in seen_states:
                stopped = "oscillation detected: design state repeated"
                break
            seen_states.append(fingerprint)
            iterations += 1

            progressed = False
            #: (strategy, description) of actions already applied in this iteration - two findings that
            #: ask for the same deterministic action (e.g. re-run kicad.drc) do not run it twice
            applied: set[tuple[str, str]] = set()
            for finding in report.failures:
                if finding.check_id in unresolved:
                    continue
                try:
                    strategy = select_strategy(finding, self.strategies)
                except NotRepairableError as e:
                    unresolved[finding.check_id] = finding.model_copy(update={"message": str(e)})
                    continue
                key = (strategy.id, strategy.describe(finding))
                if key in applied:
                    continue  # already done this iteration; the re-review decides whether it helped
                before = ir.content_hash()
                action = strategy.apply(ir, finding, workdir, self.tools)
                after = ir.content_hash()
                # the loop measures the IR itself; a strategy's own bookkeeping is not trusted
                action.ir_hash_before = before
                action.ir_hash_after = after
                if after != before:
                    action.succeeded = False
                    action.error = (
                        f"{strategy.id} changed the design (IR {before[:16]} -> {after[:16]}); "
                        "repairs may only regenerate artifacts or re-run tools" + (f"; {action.error}" if action.error else "")
                    )
                    actions.append(action)
                    unresolved[finding.check_id] = finding.model_copy(update={"message": f"{finding.message} ({action.error})"})
                    mutated = True
                    break
                actions.append(action)
                if action.succeeded:
                    progressed = True
                    applied.add(key)
                else:
                    unresolved[finding.check_id] = finding.model_copy(update={"message": f"{finding.message} (repair failed: {action.error})"})
            if mutated:
                stopped = MUTATION_STOP
                report = self.reviewer.review(ir, workdir)
                break
            if not progressed:
                stopped = "no repairable failures remain"
                report = self.reviewer.review(ir, workdir)
                break
            report = self.reviewer.review(ir, workdir)
        else:
            stopped = "all failures resolved" if iterations else stopped

        # Report still-failing checks; prefer the annotated copy that says *why* repair was refused.
        still_failing = [unresolved.get(f.check_id, f) for f in report.failures]
        if mutated:
            # a mutating strategy may have made its own finding disappear; the design change itself stays unresolved
            failing_ids = {f.check_id for f in still_failing}
            still_failing += [u for u in unresolved.values() if u.check_id not in failing_ids and "changed the design" in u.message]
        return RepairOutcome(
            iterations=iterations,
            actions=actions,
            unresolved=still_failing,
            stopped_reason=stopped,
            final_review=report,
        )

    @staticmethod
    def _fingerprint(ir: CircuitIR, report: ReviewReport) -> str:
        fails = sorted(r.check_id for r in report.failures)
        arts = sorted(f"{k}:{v.content_hash}" for k, v in ir.artifacts.items())
        return ir.content_hash() + "|" + ",".join(fails) + "|" + ",".join(arts)
