"""In-process scripted :class:`~ai_eda.llm.client.LLMClient` - no HTTP, no key.

Used by agent tests and by the CLI's ``--llm fake:<json file>`` for offline
demos. It returns a fixed sequence of answers and records every call, so a
test can assert *what the agent asked* and *what it did with the answer*
without a network. It enforces nothing about design truth - like any other
client its output is a proposal that the caller must ground and verify.

Script items (constructor ``responses`` or the JSON file's ``"responses"`` list):

* ``"text"`` - a plain content string;
* ``{"content": "..."}`` - the same, with optional ``"model"``, ``"finish_reason"``,
  ``"usage": {"prompt_tokens", "completion_tokens", "cost_usd"}``;
* ``{"structured": {...}}`` - a structured answer (``content`` is its JSON text);
* ``{"tool_calls": [{"name": "...", "arguments": {...} | "<raw string>", "id"?: "..."}]}``;
* ``{"error": {"message": "...", "status"?: int, "code"?: int, "retry_after"?: float, "kind"?: str, "sent"?: bool}}``
  - raised once as :class:`~ai_eda.llm.client.LLMError` (consumed like any other item); ``sent`` says whether
  the request reached the provider (a ``transport`` error without it counts as possibly billed);
* a ready :class:`~ai_eda.llm.client.LLMResponse` or an ``Exception`` instance (raised once).

When the script runs out the client raises ``LLMError(kind="script")`` unless
``repeat_last=True`` (the last answer is repeated) or ``default`` is given.
Cost is ``None`` unless the item states one (unknown, never 0).

The client mirrors the real client's contract where a script could otherwise
exercise a path the real client never delivers: an item with
``finish_reason: "error"`` is raised as ``LLMError(kind="response", status=200)``
(the real client refuses such a choice), ``complete`` / ``stream`` with no
messages raise ``ValueError`` before anything is recorded, and a stream that
the consumer closes early still leaves ``last_stream_response`` with the
pieces delivered so far and an *unknown* usage (the usage frame was never
reached), exactly like the HTTP client. It is ``paid = False``: a ``0`` USD
budget allows it (see :class:`~ai_eda.llm.service.LLMService`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage


@dataclass
class RecordedCall:
    model: str
    messages: list[LLMMessage]
    tools: list[ToolSpec] | None
    response_schema: dict[str, Any] | None
    temperature: float
    max_tokens: int | None
    stream: bool = False

    @property
    def system_text(self) -> str:
        return "\n".join(m.content or "" for m in self.messages if m.role == "system")

    @property
    def user_text(self) -> str:
        return "\n".join(m.content or "" for m in self.messages if m.role == "user")

    @property
    def all_text(self) -> str:
        return "\n".join(m.content or "" for m in self.messages)


@dataclass
class ScriptedResponse:
    content: str | None = None
    structured: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str = "stop"
    raw_error: str | None = None
    error: BaseException | None = None  # raised instead of returning

    @classmethod
    def from_item(cls, item: Any) -> ScriptedResponse:
        if isinstance(item, ScriptedResponse):
            return item
        if isinstance(item, BaseException):
            return cls(error=item)
        if isinstance(item, str):
            return cls(content=item)
        if isinstance(item, LLMResponse):
            return cls(
                content=item.content, structured=item.structured, tool_calls=list(item.tool_calls), usage=item.usage,
                model=item.model_used or item.model, finish_reason=item.finish_reason or "stop", raw_error=item.raw_error,
            )
        if not isinstance(item, dict):
            raise TypeError(f"unsupported script item: {type(item).__name__}")
        if "error" in item:
            e = item["error"] if isinstance(item["error"], dict) else {"message": str(item["error"])}
            return cls(error=LLMError(
                str(e.get("message", "scripted error")), kind=str(e.get("kind", "http")),
                status=e.get("status"), code=e.get("code"), retry_after=e.get("retry_after"),
                metadata=e.get("metadata") if isinstance(e.get("metadata"), dict) else None,
                sent=e.get("sent") if isinstance(e.get("sent"), bool) else None,
            ))
        usage = None
        if isinstance(item.get("usage"), dict):
            u = dict(item["usage"])
            if "cost" in u and "cost_usd" not in u:
                u["cost_usd"] = u.pop("cost")
            if u.get("cost_usd") is not None:
                u.setdefault("cost_source", "script")
            usage = Usage(**{k: v for k, v in u.items() if k in Usage.model_fields})
            if not usage.total_tokens:
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        elif isinstance(item.get("usage"), Usage):
            usage = item["usage"]
        calls: list[ToolCall] = []
        for i, tc in enumerate(item.get("tool_calls") or []):
            if isinstance(tc, ToolCall):
                calls.append(tc)
                continue
            args = tc.get("arguments", {})
            call_id = str(tc.get("id") or f"call_{i}")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except ValueError as e:
                    calls.append(ToolCall(id=call_id, name=tc["name"], arguments={}, raw_arguments=args, parse_error=f"arguments are not valid JSON: {e}"))
                    continue
                if not isinstance(parsed, dict):
                    calls.append(ToolCall(id=call_id, name=tc["name"], arguments={}, raw_arguments=args, parse_error="arguments must be a JSON object"))
                    continue
                args = parsed
            calls.append(ToolCall(id=call_id, name=tc["name"], arguments=dict(args)))
        structured = item.get("structured") if isinstance(item.get("structured"), dict) else None
        content = item.get("content")
        if content is None and structured is not None:
            content = json.dumps(structured, ensure_ascii=False)
        raw_error = item.get("raw_error")
        if raw_error is None and any(c.parse_error for c in calls):
            raw_error = "; ".join(f"tool call {c.name!r} ({c.id}): {c.parse_error}" for c in calls if c.parse_error)
        return cls(
            content=content if isinstance(content, str) else None, structured=structured, tool_calls=calls, usage=usage,
            model=item.get("model"), finish_reason=str(item.get("finish_reason") or ("tool_calls" if calls else "stop")),
            raw_error=raw_error,
        )


class ScriptedLLMClient(LLMClient):
    """See the module docstring for the script format."""

    paid = False

    def __init__(
        self,
        responses: Sequence[Any] = (),
        *,
        repeat_last: bool = False,
        default: Any = None,
        default_usage: Usage | None = None,
    ) -> None:
        self._script: list[ScriptedResponse] = [ScriptedResponse.from_item(r) for r in responses]
        self._pos = 0
        self.repeat_last = repeat_last
        self.default = ScriptedResponse.from_item(default) if default is not None else None
        self.default_usage = default_usage
        self.calls: list[RecordedCall] = []
        self.last_stream_usage: Usage | None = None
        self.last_stream_response: LLMResponse | None = None

    # ------------------------------------------------------------ loading

    @classmethod
    def from_spec(cls, spec: Any) -> ScriptedLLMClient:
        """``spec`` is the JSON document: a list of items or ``{"responses": [...], "repeat_last"?: bool}``."""
        if isinstance(spec, list):
            return cls(spec)
        if not isinstance(spec, dict):
            raise TypeError("fake LLM spec must be a list or an object with a 'responses' list")
        items = spec.get("responses", [])
        if not isinstance(items, list):
            raise TypeError("'responses' must be a list")
        return cls(items, repeat_last=bool(spec.get("repeat_last", False)), default=spec.get("default"))

    @classmethod
    def from_file(cls, path: str | Path) -> ScriptedLLMClient:
        return cls.from_spec(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------ script

    def add(self, item: Any) -> None:
        self._script.append(ScriptedResponse.from_item(item))

    @property
    def remaining(self) -> int:
        return max(0, len(self._script) - self._pos)

    def _take(self, model: str) -> ScriptedResponse:
        if self._pos < len(self._script):
            item = self._script[self._pos]
            self._pos += 1
        elif self.repeat_last and self._script:
            item = self._script[-1]
        elif self.default is not None:
            item = self.default
        else:
            raise LLMError(f"scripted client has no answer left (call #{len(self.calls)})", kind="script", model=model)
        if item.error is not None:
            raise item.error
        if item.finish_reason == "error":
            # the real client raises on a choice with finish_reason "error": a partial completion is not a completion
            raise LLMError("provider reported finish_reason 'error'", kind="response", status=200, model=model, usage=item.usage, sent=True)
        return item

    def _response(self, model: str, item: ScriptedResponse, want_structured: bool) -> LLMResponse:
        structured = item.structured
        raw_error = item.raw_error
        if want_structured and structured is None and item.content:
            try:
                obj = json.loads(item.content)
            except ValueError as e:
                raw_error = (raw_error + "; " if raw_error else "") + f"structured content is not valid JSON: {e}"
            else:
                if isinstance(obj, dict):
                    structured = obj
                else:
                    raw_error = (raw_error + "; " if raw_error else "") + "structured content is not a JSON object"
        return LLMResponse(
            model=model,
            content=item.content,
            tool_calls=list(item.tool_calls),
            structured=structured,
            usage=item.usage or self.default_usage or Usage(),
            model_used=item.model or model,
            finish_reason=item.finish_reason,
            native_finish_reason=item.finish_reason,
            id=f"fake-{len(self.calls)}",
            provider_name="scripted",
            raw_error=raw_error,
        )

    # ------------------------------------------------------------ LLMClient

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        self.calls.append(RecordedCall(model, list(messages), tools, response_schema, temperature, max_tokens))
        item = self._take(model)
        return self._response(model, item, want_structured=response_schema is not None)

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        if not messages:
            raise ValueError("messages must not be empty")
        self.calls.append(RecordedCall(model, list(messages), None, None, temperature, max_tokens, stream=True))
        self.last_stream_usage = None
        self.last_stream_response = None
        item = self._take(model)
        resp = self._response(model, item, want_structured=False)
        text = resp.content or ""
        # word-sized pieces, like a real stream; whitespace stays attached so the join is exact
        pieces: list[str] = []
        start = 0
        for i, ch in enumerate(text):
            if ch.isspace() and i > start:
                pieces.append(text[start:i])
                start = i
        pieces.append(text[start:])
        delivered: list[str] = []
        completed = False
        try:
            for piece in pieces:
                if piece:
                    delivered.append(piece)
                    yield piece
            completed = True
        finally:
            if completed:
                self.last_stream_usage = resp.usage
                self.last_stream_response = resp
            else:
                # closed early: what was delivered is known, what it cost is not (the usage frame comes last)
                self.last_stream_usage = None
                self.last_stream_response = resp.model_copy(update={"content": "".join(delivered) or None, "usage": Usage(), "finish_reason": None})
