"""PCB Agent: proposes placement / routing intent into ``ir.pcb``; DRC is done by KiCad, not here."""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR
from ai_eda.llm.router import TaskKind


class PCBAgent(Agent):
    name = "pcb"
    task = TaskKind.CIRCUIT_DESIGN

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        return self._result(notes=["placement / routing proposal not implemented"])
