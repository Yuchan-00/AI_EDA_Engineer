"""LLM abstraction layer.

The LLM is a proposal/orchestration engine. Its outputs enter the IR only as
``llm_generated`` provenance and must be verified by a tool, a source or the
user before anything downstream relies on them.

Modules: ``client`` (provider-independent contract + typed ``LLMError``),
``openrouter`` (HTTP client, per-call billing), ``claude_cli`` (the Claude
Code CLI on the user's subscription login; no per-call charge),
``codex_cli`` (the Codex CLI on the user's ChatGPT login; a subscription
too), ``openai_compat`` (any OpenAI-compatible chat-completions server:
the OpenAI API, llama.cpp, Ollama, vLLM, LM Studio; free on this machine,
per call elsewhere), ``providers``
(``ProviderClient`` over several providers, ``describe_providers`` for the
GUI), ``fake`` (in-process scripted client for tests and ``--llm
fake:<json>``), ``router`` (``provider:model`` specs, task -> model
candidates, explicit fallback only), ``usage`` (accounting; unknown cost stays
unknown, a subscription call is a known zero charge), ``service`` (budget +
approval + retry / fallback + schema validation + the cross-check runs),
``crosscheck`` (deterministic field-by-field comparison of grounded
proposals from several models), ``prompts`` (model-facing
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
from ai_eda.llm.providers import ProviderClient, ProviderInfo, describe_providers
from ai_eda.llm.router import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    KNOWN_PROVIDERS,
    ModelConfig,
    ModelRouter,
    TaskKind,
    default_router,
    parse_model_spec,
    same_model_fallback,
)
from ai_eda.llm.service import BudgetExceededError, LLMAttempt, LLMBudget, LLMService, StructuredOutputError
from ai_eda.llm.usage import UsageTracker


def __getattr__(name: str):
    # The HTTP client (and with it httpx) and the CLI client are imported only when asked for, so the agents, the
    # document archive (add_file / load / verify) and the offline CLI paths import without the network extras installed.
    if name == "OpenRouterClient":
        from ai_eda.llm.openrouter import OpenRouterClient

        return OpenRouterClient
    if name == "ClaudeCodeClient":
        from ai_eda.llm.claude_cli import ClaudeCodeClient

        return ClaudeCodeClient
    if name == "CodexCliClient":
        from ai_eda.llm.codex_cli import CodexCliClient

        return CodexCliClient
    if name == "OpenAICompatClient":
        from ai_eda.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LLMClient", "LLMError", "LLMMessage", "LLMResponse", "ToolCall", "ToolSpec", "Usage",
    "ModelConfig", "ModelRouter", "TaskKind", "default_router", "parse_model_spec", "same_model_fallback",
    "DEFAULT_CLAUDE_MODEL", "DEFAULT_FALLBACK_MODEL", "DEFAULT_PRIMARY_MODEL", "KNOWN_PROVIDERS",
    "OpenRouterClient", "ClaudeCodeClient", "CodexCliClient", "OpenAICompatClient",
    "ScriptedLLMClient", "ProviderClient", "ProviderInfo", "describe_providers", "UsageTracker",
    "LLMService", "LLMBudget", "LLMAttempt", "BudgetExceededError", "StructuredOutputError",
    "CONFIRM_KEY", "GroundedExtraction", "RequirementExtraction", "build_extraction_messages",
    "confirmation_question", "ground_extraction", "is_confirmation", "json_schema", "request_hash", "upgrade_confirmed",
]
