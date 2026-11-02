"""Review Agent: thin wrapper that runs the independent reviewer and reports its findings.

The stage message is the reviewer's status counts, plus the message of
``review.requirements_vs_ir`` when it FAILs or waits for the user: that is
the one review verdict the user answers (it names the untraced requirements
and the ``--answer leave_out=<key>`` that leaves them out, or the open
questions), so the run that builds the design shows it and never only a count.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR, ValidationStatus
from ai_eda.llm.router import TaskKind
from ai_eda.review import IndependentReviewer, ReviewArea

#: the review verdicts whose message the stage note repeats (the user acts on them with an answer)
ANSWERED_BY_USER = (ValidationStatus.FAIL, ValidationStatus.USER_INPUT_REQUIRED)


class ReviewAgent(Agent):
    name = "review"
    task = TaskKind.REVIEW

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        report = IndependentReviewer(tools=ctx.tools).review(ir, ctx.workdir)
        notes = [report.summary()]
        notes += [f"{r.check_id} {r.status}: {r.message}" for r in report.results
                  if r.check_id == ReviewArea.REQUIREMENTS_VS_IR and r.status in ANSWERED_BY_USER]
        return self._result(validation=report.results, notes=notes)
