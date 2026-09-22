"""LLM abstraction layer.

The LLM is a proposal/orchestration engine. Its outputs enter the IR only as
``llm_generated`` provenance and must be verified by a tool, a source or the
user before anything downstream relies on them.

Modules: ``client`` (provider-independent contract + typed ``LLMError``),
``openrouter`` (HTTP client), ``fake`` (in-process scripted client for tests
and ``--llm fake:<json>``), ``router`` (task -> model candidates),
``usage`` (accounting; unknown cost stays unknown), ``service`` (budget +
approval + retry / fallback + schema validation), ``prompts`` (model-facing
text) and ``extraction`` (strict schemas + deterministic grounding of a
requirement extraction).
"""

from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from ai_eda.llm.extraction import (
    CONFIRM_KEY,
    GroundedExtraction,
    RequirementExtraction,
    build_extraction_messages,
    confirmation_question,
    ground_extraction,
    is_confirmation,
    json_schema,
    request_hash,
    upgrade_confirmed,
)
from ai_eda.llm.fake import ScriptedLLMClient
from ai_eda.llm.openrouter import OpenRouterClient
from ai_eda.llm.router import ModelConfig, ModelRouter, TaskKind, default_router
from ai_eda.llm.service import BudgetExceededError, LLMAttempt, LLMBudget, LLMService, StructuredOutputError
from ai_eda.llm.usage import UsageTracker

__all__ = [
    "LLMClient", "LLMError", "LLMMessage", "LLMResponse", "ToolCall", "ToolSpec", "Usage",
    "ModelConfig", "ModelRouter", "TaskKind", "default_router",
    "OpenRouterClient", "ScriptedLLMClient", "UsageTracker",
    "LLMService", "LLMBudget", "LLMAttempt", "BudgetExceededError", "StructuredOutputError",
    "CONFIRM_KEY", "GroundedExtraction", "RequirementExtraction", "build_extraction_messages",
    "confirmation_question", "ground_extraction", "is_confirmation", "json_schema", "request_hash", "upgrade_confirmed",
]
