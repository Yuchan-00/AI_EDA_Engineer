"""Simulation Agent: decides which analyses to run and interprets real SPICE output."""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.llm.router import TaskKind
from ai_eda.tools.spice import SpiceRunner


class SimulationAgent(Agent):
    name = "simulation"
    task = TaskKind.RESULT_INTERPRETATION

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        runner: SpiceRunner | None = ctx.tools.get("spice")
        if runner is None or not runner.available():
            return self._result(
                validation=[ValidationResult(check_id="spice", status=ValidationStatus.NOT_VERIFIED, message="no SPICE engine available")],
            )
        return self._result(notes=["netlist compilation + analysis selection not implemented"])
