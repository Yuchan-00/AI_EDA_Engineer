"""Requirement Agent.

Turns ``ir.requirements.raw_input`` into structured requirements and a list
of questions. The scaffold version has no LLM; it applies a deterministic
checklist so the question mechanism works end to end today.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.ir import CircuitIR, Jurisdiction, MissingInformation, Requirement, RequirementKind, user_requirement
from ai_eda.llm.router import TaskKind

#: Baseline information every design needs before we may proceed.
BASELINE_QUESTIONS: list[MissingInformation] = [
    MissingInformation(key="application", question="What is the intended application / use of this circuit?", rationale="drives implicit requirements and regulatory scope"),
    MissingInformation(key="jurisdiction", question="Which markets / jurisdictions will the product be used or sold in?", rationale="regulatory research must not guess jurisdiction"),
    MissingInformation(key="operating_temperature", question="What is the operating temperature range?", required=False),
    MissingInformation(key="protection", question="Which protections are required (reverse polarity, OVP, OCP, ESD, ...)?", required=False),
]


class RequirementAgent(Agent):
    name = "requirement"
    task = TaskKind.REQUIREMENT_ANALYSIS

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        questions = [
            q for q in BASELINE_QUESTIONS
            if q.key not in ctx.answers and ir.requirements.get(q.key) is None
        ]
        proposals: list[IRProposal] = []
        # Answers the user gave become explicit requirements with user_requirement provenance.
        for key, answer in ctx.answers.items():
            if ir.requirements.get(key) is not None:
                continue
            req = Requirement(
                id=f"req.{key}",
                key=key,
                text=f"{key}: {answer}",
                kind=RequirementKind.EXPLICIT,
                value=user_requirement(answer),
                category="regulatory" if key == "jurisdiction" else "application" if key == "application" else "electrical",
            )
            proposals.append(IRProposal(description=f"record user answer for {key}", target="requirements.requirements", operation="append", payload=req))
            if key == "jurisdiction":
                for code in [c.strip() for c in answer.replace(";", ",").split(",") if c.strip()]:
                    if not any(j.code == code for j in ir.regulatory.jurisdictions):
                        proposals.append(
                            IRProposal(
                                description=f"add jurisdiction {code}",
                                target="regulatory.jurisdictions",
                                operation="append",
                                payload=Jurisdiction(code=code, name=code, provided_by_user=True),
                            )
                        )
        notes = []
        if ctx.llm is None:
            notes.append("no LLM configured: free-text parsing skipped, baseline checklist only")
        # TODO: with an LLM, extract explicit/implicit requirements and conflicts from raw_input
        #       into IRProposals with llm_generated provenance.
        return self._result(questions=questions, proposals=proposals, notes=notes)
