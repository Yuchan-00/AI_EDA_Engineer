"""OpenRouter client (scaffold).

Requires the optional ``httpx`` dependency. The API key is read from the
``OPENROUTER_API_KEY`` environment variable and never stored in the IR or in
logs. Calls are paid, so they pass through the approval gate the first time
a session uses them.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMClient, LLMMessage, LLMResponse, ToolSpec

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str | None = None, base_url: str = OPENROUTER_URL) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url

    def _ensure(self) -> None:
        if not self.api_key:
            raise ToolUnavailableError("OPENROUTER_API_KEY is not set")
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ToolUnavailableError("httpx not installed; pip install 'ai-eda-engineer[llm]'") from e

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self._ensure()
        raise NotImplementedError("OpenRouter request/response mapping not implemented yet")

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self._ensure()
        raise NotImplementedError("OpenRouter streaming not implemented yet")
