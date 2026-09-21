"""Regulatory Agent.

Never guesses jurisdiction. With no jurisdiction it returns
USER_INPUT_REQUIRED. With one, it is expected to research *official* sources
and attach :class:`RegulatoryProvenance` with URL, date, section and hash.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR, MissingInformation, ValidationResult, ValidationStatus
from ai_eda.llm.router import TaskKind


class RegulatoryAgent(Agent):
    name = "regulatory"
    task = TaskKind.REGULATORY_RESEARCH

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        if not ir.regulatory.jurisdiction_known:
            return self._result(
                questions=[MissingInformation(key="jurisdiction", question="Which jurisdictions apply? (e.g. EU, US, KR)", rationale="regulatory scope cannot be assumed")],
                validation=[ValidationResult(check_id="regulatory.scope", status=ValidationStatus.USER_INPUT_REQUIRED, message="jurisdiction not provided")],
            )
        return self._result(
            validation=[
                ValidationResult(
                    check_id="regulatory.research",
                    status=ValidationStatus.NOT_VERIFIED,
                    message="official-source research pipeline not implemented",
                    details={"jurisdictions": [j.code for j in ir.regulatory.jurisdictions]},
                )
            ]
        )
