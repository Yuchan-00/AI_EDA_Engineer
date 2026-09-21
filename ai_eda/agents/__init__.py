"""Specialist agents.

An agent *proposes*; it never decides truth. Every agent returns an
:class:`AgentResult` containing proposals (IR mutations tagged with
``llm_generated`` / ``assumption`` provenance), questions for the user, and
any validation results it obtained by calling deterministic tools.
"""

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.requirement import RequirementAgent
from ai_eda.agents.component import ComponentAgent
from ai_eda.agents.circuit import CircuitDesignAgent
from ai_eda.agents.simulation import SimulationAgent
from ai_eda.agents.pcb import PCBAgent
from ai_eda.agents.regulatory import RegulatoryAgent
from ai_eda.agents.manufacturing import ManufacturingAgent
from ai_eda.agents.review import ReviewAgent
from ai_eda.agents.repair import RepairAgent

__all__ = [
    "Agent", "AgentContext", "AgentResult", "IRProposal",
    "RequirementAgent", "ComponentAgent", "CircuitDesignAgent", "SimulationAgent",
    "PCBAgent", "RegulatoryAgent", "ManufacturingAgent", "ReviewAgent", "RepairAgent",
]
