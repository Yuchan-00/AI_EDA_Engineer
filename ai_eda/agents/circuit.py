"""Circuit Design Agent: proposes topology, blocks, nets and design parameters.

Numeric values it needs (resistor values, dissipation, ...) must come from
:mod:`ai_eda.tools.calc` so they carry ``derived`` provenance; the agent is
not allowed to emit a number it computed "in its head".
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR
from ai_eda.llm.router import TaskKind


class CircuitDesignAgent(Agent):
    name = "circuit_design"
    task = TaskKind.CIRCUIT_DESIGN

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        return self._result(notes=["circuit design proposal not implemented"])
