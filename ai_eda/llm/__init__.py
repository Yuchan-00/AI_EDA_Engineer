"""LLM abstraction layer.

The LLM is a proposal/orchestration engine. Its outputs enter the IR only as
``llm_generated`` provenance and must be verified by a tool or a source
before anything downstream relies on them.
"""

from ai_eda.llm.client import LLMClient, LLMMessage, LLMResponse, ToolSpec, Usage
from ai_eda.llm.router import ModelConfig, ModelRouter, TaskKind
from ai_eda.llm.openrouter import OpenRouterClient
from ai_eda.llm.usage import UsageTracker

__all__ = [
    "LLMClient", "LLMMessage", "LLMResponse", "ToolSpec", "Usage",
    "ModelConfig", "ModelRouter", "TaskKind", "OpenRouterClient", "UsageTracker",
]
