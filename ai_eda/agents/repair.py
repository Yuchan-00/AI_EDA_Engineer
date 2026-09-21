"""Repair Agent: runs the deterministic repair loop; it does not invent fixes.

Every attempt the loop made is persisted: besides the final review results,
the agent returns a ``repair.loop`` :class:`~ai_eda.ir.ValidationResult`
whose details hold the full action log (strategy, description, IR hash
before / after, error, timestamp), the unresolved findings and the stop
reason, so ``ir.json`` and the GUI can show what was tried and why it
stopped (spec 17-19).
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR
from ai_eda.llm.router import TaskKind
from ai_eda.repair import RepairLoop


class RepairAgent(Agent):
    name = "repair"
    task = TaskKind.REPAIR_PLANNING

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        loop = RepairLoop(tools=ctx.tools)
        outcome = loop.run(ir, ctx.workdir)
        log = outcome.as_validation_result(ir.content_hash(), loop.version)
        return self._result(validation=[*outcome.final_review.results, log], notes=[outcome.summary()])
