"""Component Agent.

Selection order (from the spec): electrical -> safety -> regulatory ->
environment -> reliability -> manufacturability -> sourcing -> cost.

The agent may *propose* a part, but the proposal enters the IR with
``llm_generated`` provenance. It only becomes authoritative after a
datasheet / official part record is attached (``Component.datasheet`` with
a hash) and the library lookup verifies symbol and footprint.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR
from ai_eda.llm.router import TaskKind


class ComponentAgent(Agent):
    name = "component"
    task = TaskKind.COMPONENT_PROPOSAL

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        notes = ["component proposal requires an LLM and a datasheet retrieval pipeline; not implemented"]
        return self._result(notes=notes)
