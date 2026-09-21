"""Review Agent: thin wrapper that runs the independent reviewer and reports its findings."""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR
from ai_eda.llm.router import TaskKind
from ai_eda.review import IndependentReviewer


class ReviewAgent(Agent):
    name = "review"
    task = TaskKind.REVIEW

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        report = IndependentReviewer(tools=ctx.tools).review(ir, ctx.workdir)
        return self._result(validation=report.results, notes=[report.summary()])
