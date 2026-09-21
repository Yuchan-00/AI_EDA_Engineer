"""Task-based model routing with fallback."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TaskKind(StrEnum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    COMPONENT_PROPOSAL = "component_proposal"
    CIRCUIT_DESIGN = "circuit_design"
    REGULATORY_RESEARCH = "regulatory_research"
    RESULT_INTERPRETATION = "result_interpretation"
    REVIEW = "review"
    REPAIR_PLANNING = "repair_planning"
    CHAT = "chat"


class ModelConfig(BaseModel):
    model: str  # provider model id, e.g. "anthropic/claude-sonnet-5"
    temperature: float = 0.0
    max_tokens: int | None = None
    supports_tools: bool = True
    supports_structured: bool = True


class ModelRouter(BaseModel):
    default: ModelConfig
    by_task: dict[TaskKind, ModelConfig] = Field(default_factory=dict)
    fallbacks: list[ModelConfig] = Field(default_factory=list)

    def for_task(self, task: TaskKind) -> ModelConfig:
        return self.by_task.get(task, self.default)

    def candidates(self, task: TaskKind) -> list[ModelConfig]:
        primary = self.for_task(task)
        return [primary, *[f for f in self.fallbacks if f.model != primary.model]]
