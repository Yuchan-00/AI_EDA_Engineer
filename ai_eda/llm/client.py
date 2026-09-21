from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


class LLMResponse(BaseModel):
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...
