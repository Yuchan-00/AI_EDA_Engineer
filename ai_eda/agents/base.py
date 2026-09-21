from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_eda.ir import CircuitIR, MissingInformation, ValidationResult
from ai_eda.llm.client import LLMClient
from ai_eda.llm.router import ModelRouter, TaskKind
from ai_eda.llm.usage import UsageTracker


class AgentContext(BaseModel):
    workdir: Path
    llm: LLMClient | None = None
    router: ModelRouter | None = None
    usage: UsageTracker = Field(default_factory=UsageTracker)
    #: deterministic tool handles: kicad_cli, kicad_library, spice, ...
    tools: dict[str, Any] = Field(default_factory=dict)
    #: answers the user has given to earlier questions, keyed by MissingInformation.key
    answers: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class IRProposal(BaseModel):
    """A proposed change to the IR, described declaratively so it can be shown to the user
    and applied (or rejected) by the orchestrator - the agent does not mutate the IR itself."""

    description: str
    #: dotted path into the IR, e.g. "components", "parameters.v_out", "topology"
    target: str
    operation: str  # "set" | "append" | "remove"
    payload: Any = None
    rationale: str = ""


class AgentResult(BaseModel):
    agent: str
    proposals: list[IRProposal] = Field(default_factory=list)
    questions: list[MissingInformation] = Field(default_factory=list)
    validation: list[ValidationResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def blocked_on_user(self) -> bool:
        return any(q.required for q in self.questions)


class Agent(ABC):
    name: str = "agent"
    task: TaskKind = TaskKind.CHAT

    @abstractmethod
    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult: ...

    def _result(self, **kw) -> AgentResult:
        return AgentResult(agent=self.name, **kw)
