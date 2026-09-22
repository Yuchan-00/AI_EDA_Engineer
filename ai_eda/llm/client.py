"""Provider-independent LLM client contract.

Invariant enforced here: the model is a *proposal engine*. Nothing in an
:class:`LLMResponse` is authoritative - the caller parses ``content`` /
``structured`` / ``tool_calls`` into schemas and checks every claim
deterministically. The response therefore carries the facts a caller needs
to *account* for the call (``model_used``, ``usage`` incl. cost - ``None``
when the provider did not report one, never ``0``) and to *reject* it
(``finish_reason``, ``raw_error``), but never a "trusted" flag.

Errors are typed (:class:`LLMError`) so a service layer can decide retry /
fallback from ``status`` / ``code`` / ``retry_after`` without parsing text,
and their message never contains the API key (the provider client redacts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from pydantic import BaseModel, Field

from ai_eda.errors import AiEdaError

#: HTTP statuses (and matching ``error.code`` values) that a service may retry or fall back on.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class ToolCall(BaseModel):
    id: str
    name: str
    #: JSON-decoded ``function.arguments``; ``{}`` when the model emitted invalid JSON
    #: (then ``raw_arguments`` / ``parse_error`` say so - a malformed call is a failed call, not a crash)
    arguments: dict[str, Any]
    #: the provider's ``function.arguments`` string, kept verbatim when it could not be decoded into a dict
    raw_arguments: str | None = None
    parse_error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.parse_error is None


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    #: ``None`` only for an assistant message that carries ``tool_calls`` (the provider contract's ``content: null``)
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    #: the tool calls an earlier assistant turn made; must be echoed back before the ``role="tool"`` results
    tool_calls: list[ToolCall] | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: provider-reported cost (OpenRouter ``usage.cost`` credits, taken as USD); ``None`` when not reported
    cost_usd: float | None = None
    #: where ``cost_usd`` came from: "provider" (usage accounting), "estimate" (pricing x tokens), "script"
    cost_source: str | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    is_byok: bool | None = None


class LLMResponse(BaseModel):
    #: the model the caller asked for
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: ``content`` decoded as a JSON object when a response schema was requested and the content parsed
    structured: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None
    #: the model the provider actually served (``response.model``; billing follows this, not ``model``)
    model_used: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    #: provider response id (OpenRouter ``gen-...``), usable with the generation endpoint
    id: str | None = None
    generation_id: str | None = None
    provider_name: str | None = None
    #: non-fatal defects in an otherwise delivered response (malformed tool arguments, unparseable
    #: structured content, ...). The response is delivered, but a caller must treat it as suspect.
    raw_error: str | None = None
    #: model reasoning text when the provider returned it (never parsed, never trusted)
    reasoning: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMError(AiEdaError):
    """A request to the model provider failed.

    ``kind`` is one of ``"http"`` (non-2xx status), ``"response"`` (2xx body that
    carried an error object or was unparseable), ``"stream"`` (error event after
    the stream was committed), ``"transport"`` (connection / timeout - no
    status), ``"script"`` (an in-process scripted client ran out of answers).
    ``status`` is the HTTP status, ``code`` the provider's ``error.code``
    (int when it was numeric), ``retry_after`` seconds from the ``Retry-After``
    header when the provider sent one. The message is redacted: it never
    contains the API key, even when the provider echoed it.

    ``usage`` is the usage object the provider reported *with* the error (a
    2xx body or an SSE frame that carried both ``error`` and ``usage``): the
    generation may have been billed, and when the provider says how much, the
    accounting must keep it. ``sent`` says whether the request reached the
    provider: ``False`` for a connection failure / connect timeout (nothing
    can have been billed), ``True`` once a response or stream started, ``None``
    when unknown (a read timeout after the request went out is ``True``).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "http",
        status: int | None = None,
        code: int | str | None = None,
        retry_after: float | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        usage: Usage | None = None,
        sent: bool | None = None,
    ) -> None:
        self.kind = kind
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.metadata = metadata or {}
        self.model = model
        self.usage = usage
        self.sent = sent
        self.message = message
        bits = [f"{kind}"]
        if status is not None:
            bits.append(f"status={status}")
        if code is not None and code != status:
            bits.append(f"code={code}")
        if retry_after is not None:
            bits.append(f"retry_after={retry_after:g}s")
        if model:
            bits.append(f"model={model}")
        super().__init__(f"LLM request failed ({', '.join(bits)}): {message}")

    def __reduce__(self):  # type: ignore[override]
        return (
            _rebuild_llm_error,
            (self.message, self.kind, self.status, self.code, self.retry_after, self.metadata, self.model, self.usage, self.sent),
        )

    @property
    def maybe_billed(self) -> bool:
        """Whether the provider may have generated (and billed) something for this failed request.

        A 2xx body or stream that carried an error committed the request; a
        transport failure after the request was sent (read timeout) may have
        too. Only a request that never reached the provider (``sent is
        False``) or an HTTP error status cannot have been billed.
        """
        if self.kind in ("response", "stream"):
            return True
        if self.kind == "transport":
            return self.sent is not False
        return False

    @property
    def retryable(self) -> bool:
        """Transient by the provider's own classification (timeouts, rate limits, upstream outages)."""
        if self.kind == "transport":
            return True
        if self.status in RETRYABLE_STATUSES:
            return True
        return isinstance(self.code, int) and self.code in RETRYABLE_STATUSES

    @property
    def rate_limited(self) -> bool:
        return self.status == 429 or self.code == 429

    @property
    def insufficient_credits(self) -> bool:
        return self.status == 402 or self.code == 402


def _rebuild_llm_error(
    message: str, kind: str, status: int | None, code: int | str | None, retry_after: float | None,
    metadata: dict[str, Any], model: str | None, usage: Usage | None = None, sent: bool | None = None,
) -> LLMError:
    return LLMError(message, kind=kind, status=status, code=code, retry_after=retry_after, metadata=metadata, model=model, usage=usage, sent=sent)


class LLMClient(ABC):
    #: whether a call to this client can cost money. A service with a ``0`` USD budget refuses every
    #: call to a paid client (the first call cannot be priced in advance) and allows a free one.
    paid: bool = True

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
