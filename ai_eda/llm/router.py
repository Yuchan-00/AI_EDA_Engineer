"""Task-based model routing with fallback.

:func:`default_router` is the project's default: ``anthropic/claude-sonnet-5``
for every task (requirement analysis included) with
``anthropic/claude-haiku-4.5`` as the fallback, both at temperature 0. The
router only *orders* candidates; whether a candidate is tried at all is the
service's decision (budget, retryable error class).

Every request carries a completion cap: :attr:`ModelConfig.max_tokens` when
the config sets one, else :data:`DEFAULT_MAX_TOKENS` for the task
(:data:`DEFAULT_MAX_TOKENS_ANY` for tasks not listed). A request without a
cap would let a single call spend an unbounded amount against an approved
budget; the service additionally lowers the cap to the remaining token budget.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#: OpenRouter ids used by :func:`default_router`
DEFAULT_PRIMARY_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_FALLBACK_MODEL = "anthropic/claude-haiku-4.5"


class TaskKind(StrEnum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    COMPONENT_PROPOSAL = "component_proposal"
    CIRCUIT_DESIGN = "circuit_design"
    REGULATORY_RESEARCH = "regulatory_research"
    RESULT_INTERPRETATION = "result_interpretation"
    REVIEW = "review"
    REPAIR_PLANNING = "repair_planning"
    CHAT = "chat"


#: completion-token cap sent with every request of a task unless the caller or the config says otherwise
DEFAULT_MAX_TOKENS: dict[TaskKind, int] = {
    TaskKind.REQUIREMENT_ANALYSIS: 4096,
    TaskKind.COMPONENT_PROPOSAL: 4096,
    TaskKind.CIRCUIT_DESIGN: 8192,
    TaskKind.REGULATORY_RESEARCH: 4096,
    TaskKind.RESULT_INTERPRETATION: 2048,
    TaskKind.REVIEW: 4096,
    TaskKind.REPAIR_PLANNING: 2048,
    TaskKind.CHAT: 2048,
}
#: cap for a task without an entry above
DEFAULT_MAX_TOKENS_ANY = 2048


def max_tokens_for(task: TaskKind, cfg: "ModelConfig | None" = None, requested: int | None = None) -> int:
    """The completion cap a request must carry: ``requested``, else ``cfg.max_tokens``, else the task default (never ``None``)."""
    if requested is not None:
        return max(1, int(requested))
    if cfg is not None and cfg.max_tokens is not None:
        return max(1, int(cfg.max_tokens))
    return DEFAULT_MAX_TOKENS.get(task, DEFAULT_MAX_TOKENS_ANY)


class ModelConfig(BaseModel):
    model: str  # provider model id, e.g. "anthropic/claude-sonnet-5"
    temperature: float = 0.0
    #: completion cap for this model; ``None`` means "the task default" (:func:`max_tokens_for`), never "unlimited"
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


def default_router(model: str | None = None, *, fallback: str | None = DEFAULT_FALLBACK_MODEL) -> ModelRouter:
    """The default routing: ``model`` (default Sonnet 5) for every task, Haiku 4.5 as the fallback, temperature 0.

    ``fallback=None`` disables the fallback; a fallback equal to the primary is dropped by :meth:`ModelRouter.candidates`.
    """
    primary = ModelConfig(model=model or DEFAULT_PRIMARY_MODEL, temperature=0.0)
    fallbacks = [ModelConfig(model=fallback, temperature=0.0)] if fallback else []
    return ModelRouter(default=primary, by_task={TaskKind.REQUIREMENT_ANALYSIS: primary}, fallbacks=fallbacks)
