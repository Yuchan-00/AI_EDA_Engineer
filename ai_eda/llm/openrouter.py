"""OpenRouter chat-completions client.

Speaks the OpenAI-compatible contract of ``POST {base_url}/chat/completions``
(``Authorization: Bearer``, ``tools`` / ``tool_calls`` with JSON-string
arguments, ``response_format: json_schema``, ``usage.cost``, SSE streaming
with ``: OPENROUTER PROCESSING`` keepalives, a final usage chunk and
``data: [DONE]``).

Invariants enforced here:

* The API key comes from the constructor or ``OPENROUTER_API_KEY`` and is
  never stored anywhere but the private attribute: not in ``repr``, not in
  logs, not in exceptions - every string that leaves this module passes
  through :meth:`OpenRouterClient._redact`: error messages and metadata
  (redacted *before* any truncation, so a key straddling the cut cannot
  leak its prefix), ``raw`` payloads, ``content``, ``reasoning``, tool-call
  arguments (decoded and verbatim) and every streamed delta. Even a provider
  that echoes the key back cannot leak it through this client.
* Every failure is a typed :class:`~ai_eda.llm.client.LLMError` (status,
  provider code, ``Retry-After``) so the service layer decides retry /
  fallback. A ``200`` whose body carries an ``error`` object, a choice with
  ``finish_reason == "error"`` and an unparseable body are errors too: a
  partial completion is not a completion. The ``usage`` such a body or SSE
  frame reports travels on the error (``LLMError.usage``) so the accounting
  keeps a cost the provider did report; transport failures say whether the
  request was sent (``LLMError.sent``).
* The base URL is the constructor argument, else ``OPENROUTER_BASE_URL``,
  else the public endpoint - so every code path, ``doctor --online``
  included, can be pointed at the local fake server.
* A model's output is data. Tool-call arguments that are not valid JSON are
  kept verbatim on the :class:`~ai_eda.llm.client.ToolCall` (``parse_error``)
  and noted in ``LLMResponse.raw_error``; they never raise and never become
  ``{}`` silently.
* Cost is ``usage.cost`` when the provider reports it and ``None`` otherwise,
  never ``0`` - :meth:`ai_eda.llm.usage.UsageTracker.total_cost_usd` then
  reports "unknown" instead of a wrong number.
* No message content is logged at INFO level; INFO carries only model,
  status, token counts and timings.
* All timeouts are explicit (``timeout`` / ``connect_timeout``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import ssl
import time
from urllib.parse import urlsplit
from email.utils import parsedate_to_datetime
from typing import Any, Iterator

import httpx

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#: kept for older imports; the chat endpoint under the default base URL
OPENROUTER_URL = OPENROUTER_BASE_URL + "/chat/completions"
ENV_KEY = "OPENROUTER_API_KEY"
#: optional override of the base URL (tests point it at ``tests/fake_openrouter.py``)
ENV_BASE_URL = "OPENROUTER_BASE_URL"
REDACTED = "[REDACTED]"
#: longest error-body excerpt kept in an :class:`LLMError` message (applied after redaction)
ERROR_TEXT_LIMIT = 2000


def default_base_url() -> str:
    """``OPENROUTER_BASE_URL`` when set (non-blank), else the public endpoint."""
    return os.environ.get(ENV_BASE_URL, "").strip() or OPENROUTER_BASE_URL


#: hosts a plain-http base URL may point at (the local fake server of the tests); everything else must be https
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


def check_base_url(url: str) -> str:
    """``url`` without its trailing slash, or :class:`~ai_eda.errors.ToolUnavailableError` when the key would travel in clear.

    Every request carries ``Authorization: Bearer <key>``, so the endpoint
    must be ``https://`` - except a loopback host (``http://127.0.0.1:<port>``),
    which is how the tests point the client at ``tests/fake_openrouter.py``.
    The offending URL is named in the error; the key never is.
    """
    base = (url or "").strip().rstrip("/")
    parts = urlsplit(base)
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and host:
        return base
    if parts.scheme == "http" and host in LOOPBACK_HOSTS:
        return base
    raise ToolUnavailableError(f"{ENV_BASE_URL} / base_url must be https:// (or http:// on a loopback host for a local fake): got {base!r}")

log = logging.getLogger("ai_eda.llm.openrouter")


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """One verified TLS context per process: httpx builds it from the CA bundle on every Client() otherwise (~0.3 s)."""
    return httpx.create_ssl_context()


#: a held key prefix longer than this is redacted when the stream ends without completing it (a truncated echo)
PARTIAL_KEY_REDACT_LEN = 12


class _StreamRedactor:
    """Redacts the key from streamed text even when it arrives split across deltas.

    ``push`` returns the text safe to yield now; a trailing fragment that
    could be the start of the key is held back until more text arrives.
    ``flush`` returns what is still held when the stream ends (redacted when
    it is a long enough piece of the key to matter).
    """

    def __init__(self, client: "OpenRouterClient") -> None:
        self._client = client
        self._pending = ""

    def push(self, text: str) -> str:
        buf = self._client._redact(self._pending + text)
        hold = self._client._held_prefix_len(buf)
        out, self._pending = (buf[:-hold], buf[-hold:]) if hold else (buf, "")
        return out

    def flush(self) -> str:
        out, self._pending = self._pending, ""
        return REDACTED if len(out) > PARTIAL_KEY_REDACT_LEN else out

    @property
    def pending(self) -> str:
        return self._pending


class OpenRouterClient(LLMClient):
    """HTTP client for OpenRouter. See the module docstring for the invariants."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        app_url: str | None = None,
        app_title: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        reasoning: dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        schema_name: str = "ai_eda_response",
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(ENV_KEY)
        if not key or not key.strip():
            raise ToolUnavailableError(f"{ENV_KEY} is not set (pass api_key= or export {ENV_KEY})")
        self._api_key = key.strip()
        self.base_url = check_base_url(base_url if base_url is not None else default_base_url())
        self.app_url = app_url
        self.app_title = app_title
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        #: request-level ``reasoning`` object sent with every call (e.g. ``{"effort": "none"}`` for cheap extraction)
        self.reasoning = reasoning
        #: request-level ``provider`` preferences (e.g. ``{"require_parameters": True}``)
        self.provider = provider
        self.schema_name = schema_name
        #: usage reported by the final chunk of the most recent :meth:`stream` (``None`` when the stream sent none)
        self.last_stream_usage: Usage | None = None
        #: full accounting view of the most recent :meth:`stream` (content joined, model_used, finish_reason, usage)
        self.last_stream_response: LLMResponse | None = None
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            headers=self._headers(),
            verify=_ssl_context(),
        )

    # ------------------------------------------------------------------ basics

    def __repr__(self) -> str:
        return f"OpenRouterClient(base_url={self.base_url!r}, api_key=<set>)"

    __str__ = __repr__

    @staticmethod
    def key_present() -> bool:
        return bool(os.environ.get(ENV_KEY, "").strip())

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.app_url:
            h["HTTP-Referer"] = self.app_url
        if self.app_title:
            h["X-OpenRouter-Title"] = self.app_title
            h["X-Title"] = self.app_title  # older spelling, still accepted
        return h

    def _redact(self, value: Any) -> Any:
        """Replace the key wherever it appears in a string / nested structure."""
        if isinstance(value, str):
            return value.replace(self._api_key, REDACTED) if self._api_key in value else value
        if isinstance(value, dict):
            return {self._redact(k): self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        return value

    def _held_prefix_len(self, text: str) -> int:
        """Length of the longest proper suffix of ``text`` that is a prefix of the key.

        A streamed key can be split across SSE deltas; such a suffix is
        withheld by :class:`_StreamRedactor` until the next delta shows
        whether it completes the key.
        """
        key = self._api_key
        for n in range(min(len(text), len(key) - 1), 0, -1):
            if text.endswith(key[:n]):
                return n
        return 0

    # ---------------------------------------------------------- request build

    @staticmethod
    def _serialize_message(m: LLMMessage) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name:
            out["name"] = m.name
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # the contract wants a JSON string; a call we could not decode is echoed verbatim
                        "arguments": tc.raw_arguments if tc.parse_error else json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
            if out["content"] == "":
                out["content"] = None
        elif out["content"] is None:
            out["content"] = ""
        return out

    def _build_body(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None,
        response_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty")
        body: dict[str, Any] = {
            "model": model,
            "messages": [self._serialize_message(m) for m in messages],
            "temperature": temperature,
            "stream": stream,
            # documented as deprecated/no-op, but harmless and makes the intent explicit on older gateways
            "usage": {"include": True},
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.schema_name, "strict": True, "schema": response_schema},
            }
        if self.reasoning is not None:
            body["reasoning"] = self.reasoning
        if self.provider is not None:
            body["provider"] = self.provider
        return body

    # --------------------------------------------------------- error mapping

    @staticmethod
    def _retry_after(headers: httpx.Headers) -> float | None:
        raw = headers.get("Retry-After")
        if raw is None:
            return None
        raw = raw.strip()
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            return None
        return max(0.0, when.timestamp() - time.time())

    @staticmethod
    def _error_fields(obj: Any) -> tuple[int | str | None, str, dict[str, Any]]:
        """``(code, message, metadata)`` from a provider ``error`` object of any of the documented shapes."""
        if not isinstance(obj, dict):
            return None, str(obj), {}
        code = obj.get("code")
        if isinstance(code, str) and code.strip().lstrip("-").isdigit():
            code = int(code)
        elif not isinstance(code, (int, str)) and code is not None:
            code = str(code)
        message = obj.get("message")
        if not isinstance(message, str):
            message = json.dumps(obj, ensure_ascii=False)
        meta = obj.get("metadata")
        return code, message, meta if isinstance(meta, dict) else {}

    def _http_error(self, model: str | None, status: int, headers: httpx.Headers, body: bytes) -> LLMError:
        text = body.decode("utf-8", errors="replace")
        code: int | str | None = None
        # the whole body is redacted before the message is truncated below: truncating first would let a
        # key that straddles the cut keep its prefix (the shortened string no longer holds the full key)
        message = self._redact(text) if text else f"HTTP {status} with empty body"
        meta: dict[str, Any] = {}
        try:
            data = json.loads(text) if text else None
        except ValueError:
            data = None
        if isinstance(data, dict) and "error" in data:
            code, message, meta = self._error_fields(data["error"])
            code = self._redact(code) if isinstance(code, str) else code
        elif isinstance(data, dict) and isinstance(data.get("message"), str):
            message = data["message"]
        return LLMError(
            self._redact(message)[:ERROR_TEXT_LIMIT],
            kind="http",
            status=status,
            code=code,
            retry_after=self._retry_after(headers),
            metadata=self._redact(meta),
            model=model,
            sent=True,
        )

    def _transport_error(self, model: str | None, exc: httpx.HTTPError) -> LLMError:
        """A connection / timeout failure, with ``sent`` saying whether the request reached the provider."""
        sent: bool | None
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
            sent = False  # never got a connection: nothing can have been generated or billed
        elif isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError)):
            sent = True  # the request (or part of it) went out; the provider may be generating
        else:
            sent = None
        if isinstance(exc, httpx.TimeoutException):
            message = f"timeout after {self.timeout:g}s: {exc.__class__.__name__}"
        else:
            message = f"{exc.__class__.__name__}: {exc}"
        return LLMError(self._redact(message), kind="transport", model=model, sent=sent)

    def _body_error(self, model: str | None, data: dict[str, Any], *, kind: str = "response", usage: Usage | None = None) -> LLMError | None:
        """An error carried inside a 2xx body (top-level ``error`` or a choice with an error) - or ``None``.

        ``usage`` is what the same body / frame reported (parsed by the caller
        *before* this check) and rides on the error so a billed failure keeps
        its provider-reported cost.
        """
        if isinstance(data.get("error"), dict) or isinstance(data.get("error"), str):
            code, message, meta = self._error_fields(data["error"])
            code = self._redact(code) if isinstance(code, str) else code
            return LLMError(self._redact(message), kind=kind, status=200, code=code, metadata=self._redact(meta), model=model, usage=usage, sent=True)
        choices = data.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                if ch.get("error") is not None:
                    code, message, meta = self._error_fields(ch["error"])
                    code = self._redact(code) if isinstance(code, str) else code
                    return LLMError(self._redact(message), kind=kind, status=200, code=code, metadata=self._redact(meta), model=model, usage=usage, sent=True)
                if ch.get("finish_reason") == "error":
                    return LLMError("provider reported finish_reason 'error'", kind=kind, status=200, model=model, usage=usage, sent=True)
        return None

    # --------------------------------------------------------- response parse

    @staticmethod
    def _parse_usage(u: Any) -> Usage:
        if not isinstance(u, dict):
            return Usage()

        def _int(v: Any) -> int:
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

        def _opt_int(v: Any) -> int | None:
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        cost = u.get("cost")
        cost_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        pd = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
        cd = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
        prompt = _int(u.get("prompt_tokens"))
        completion = _int(u.get("completion_tokens"))
        total = _int(u.get("total_tokens")) or (prompt + completion)
        byok = u.get("is_byok")
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=cost_usd,
            cost_source="provider" if cost_usd is not None else None,
            cached_tokens=_opt_int(pd.get("cached_tokens")),
            cache_write_tokens=_opt_int(pd.get("cache_write_tokens")),
            reasoning_tokens=_opt_int(cd.get("reasoning_tokens")),
            is_byok=byok if isinstance(byok, bool) else None,
        )

    @staticmethod
    def _content_text(content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # content parts
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "".join(parts)
        return str(content)

    def _parse_tool_call(self, tc: Any, index: int) -> ToolCall | None:
        """One ``tool_calls[]`` entry as a :class:`ToolCall`; name, arguments and the verbatim string are redacted."""
        if not isinstance(tc, dict):
            return None
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") or tc.get("name")
        if not isinstance(name, str) or not name:
            return None
        name = self._redact(name)
        call_id = tc.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        call_id = self._redact(call_id)
        raw_args = fn.get("arguments", tc.get("arguments"))
        if raw_args is None or raw_args == "":
            return ToolCall(id=call_id, name=name, arguments={})
        if isinstance(raw_args, dict):  # some providers already decode it
            return ToolCall(id=call_id, name=name, arguments=self._redact(raw_args))
        if not isinstance(raw_args, str):
            raw_args = json.dumps(raw_args, ensure_ascii=False)
        raw_args = self._redact(raw_args)
        try:
            parsed = json.loads(raw_args)
        except ValueError as e:
            return ToolCall(id=call_id, name=name, arguments={}, raw_arguments=raw_args, parse_error=self._redact(f"arguments are not valid JSON: {e}"))
        if not isinstance(parsed, dict):
            return ToolCall(
                id=call_id, name=name, arguments={}, raw_arguments=raw_args,
                parse_error=f"arguments must be a JSON object, got {type(parsed).__name__}",
            )
        return ToolCall(id=call_id, name=name, arguments=self._redact(parsed))

    def _parse_completion(
        self,
        model: str,
        data: dict[str, Any],
        headers: httpx.Headers,
        want_structured: bool,
    ) -> LLMResponse:
        # usage first: a body that carries an error object may still report what was billed
        usage = self._parse_usage(data.get("usage")) if isinstance(data.get("usage"), dict) else None
        err = self._body_error(model, data, usage=usage)
        if err is not None:
            raise err
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("response has no choices", kind="response", status=200, model=model, usage=usage, sent=True)
        choice = choices[0]
        msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = self._redact(self._content_text(msg.get("content")))
        problems: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_calls = msg.get("tool_calls")
        if isinstance(raw_calls, list):
            for i, tc in enumerate(raw_calls):
                parsed = self._parse_tool_call(tc, i)
                if parsed is None:
                    problems.append(f"tool_calls[{i}] is not a function call object")
                    continue
                tool_calls.append(parsed)
                if parsed.parse_error:
                    problems.append(f"tool call {parsed.name!r} ({parsed.id}): {parsed.parse_error}")
        structured: dict[str, Any] | None = None
        if want_structured and content:
            try:
                obj = json.loads(content)
            except ValueError as e:
                problems.append(f"structured content is not valid JSON: {e}")
            else:
                if isinstance(obj, dict):
                    structured = obj
                else:
                    problems.append(f"structured content is a JSON {type(obj).__name__}, not an object")
        reasoning = msg.get("reasoning")
        resp = LLMResponse(
            model=model,
            content=content,
            tool_calls=tool_calls,
            structured=structured,
            usage=usage or Usage(),
            raw=self._redact(data),
            # every string field is redacted, not only ``raw`` / ``content``: a provider (or a proxy behind
            # OPENROUTER_BASE_URL) that echoes the key in ``model`` / ``id`` / a header would otherwise
            # put it into the INFO log, the usage records and the IR's validation details
            model_used=self._redact(data.get("model")) if isinstance(data.get("model"), str) else None,
            finish_reason=self._redact(choice.get("finish_reason")) if isinstance(choice.get("finish_reason"), str) else None,
            native_finish_reason=self._redact(choice.get("native_finish_reason")) if isinstance(choice.get("native_finish_reason"), str) else None,
            id=self._redact(data.get("id")) if isinstance(data.get("id"), str) else None,
            generation_id=self._redact(headers.get("X-Generation-Id")),
            provider_name=self._redact(headers.get("X-Provider-Name")),
            raw_error=self._redact("; ".join(problems)) if problems else None,
            reasoning=self._redact(reasoning) if isinstance(reasoning, str) else None,
        )
        return resp

    # ------------------------------------------------------------------- API

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        body = self._build_body(model, messages, tools, response_schema, temperature, max_tokens, stream=False)
        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        try:
            r = self._http.post(url, json=body)
        except httpx.HTTPError as e:
            raise self._transport_error(model, e) from None
        elapsed = time.monotonic() - t0
        if r.status_code != 200:
            err = self._http_error(model, r.status_code, r.headers, r.content)
            log.info("openrouter complete model=%s status=%s code=%s elapsed=%.2fs", model, r.status_code, err.code, elapsed)
            raise err
        try:
            data = r.json()
        except ValueError as e:
            raise LLMError(self._redact(f"response is not valid JSON: {e}"), kind="response", status=200, model=model, sent=True) from None
        if not isinstance(data, dict):
            raise LLMError("response JSON is not an object", kind="response", status=200, model=model, sent=True)
        resp = self._parse_completion(model, data, r.headers, want_structured=response_schema is not None)
        log.info(
            "openrouter complete model=%s used=%s finish=%s prompt_tokens=%d completion_tokens=%d cost=%s elapsed=%.2fs",
            model, resp.model_used, resp.finish_reason, resp.usage.prompt_tokens, resp.usage.completion_tokens,
            "unknown" if resp.usage.cost_usd is None else f"{resp.usage.cost_usd:.6f}", elapsed,
        )
        return resp

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        body = self._build_body(model, messages, None, None, temperature, max_tokens, stream=True)
        url = f"{self.base_url}/chat/completions"
        self.last_stream_usage = None
        self.last_stream_response = None
        pieces: list[str] = []
        usage: Usage | None = None
        model_used: str | None = None
        finish_reason: str | None = None
        native_finish: str | None = None
        gen_id: str | None = None
        header_gen_id: str | None = None
        provider_name: str | None = None
        tool_frags: dict[int, dict[str, Any]] = {}
        redactor = _StreamRedactor(self)
        t0 = time.monotonic()

        def _finish() -> None:
            self.last_stream_usage = usage
            tail = redactor.flush()  # a fragment held back at the end (stream cut, consumer left) still belongs to the content
            if tail:
                pieces.append(tail)
            self.last_stream_response = LLMResponse(
                model=model,
                content=self._redact("".join(pieces)) if pieces else None,
                tool_calls=[
                    tc for tc in (
                        self._parse_tool_call(
                            {"id": f.get("id"), "function": {"name": f.get("name"), "arguments": f.get("arguments", "")}}, i
                        )
                        for i, f in sorted(tool_frags.items())
                    ) if tc is not None
                ],
                usage=usage or Usage(),
                model_used=model_used,
                finish_reason=finish_reason,
                native_finish_reason=native_finish,
                id=gen_id,
                generation_id=header_gen_id,
                provider_name=provider_name,
            )
            log.info(
                "openrouter stream model=%s used=%s finish=%s prompt_tokens=%s completion_tokens=%s cost=%s elapsed=%.2fs",
                model, model_used, finish_reason,
                usage.prompt_tokens if usage else "?", usage.completion_tokens if usage else "?",
                "unknown" if usage is None or usage.cost_usd is None else f"{usage.cost_usd:.6f}",
                time.monotonic() - t0,
            )

        try:
            with self._http.stream("POST", url, json=body) as r:
                header_gen_id = self._redact(r.headers.get("X-Generation-Id"))
                provider_name = self._redact(r.headers.get("X-Provider-Name"))
                if r.status_code != 200:
                    raise self._http_error(model, r.status_code, r.headers, r.read())
                ctype = r.headers.get("Content-Type", "")
                if "text/event-stream" not in ctype:
                    # the provider answered without committing to a stream (e.g. a JSON body): treat as one completion
                    raw = r.read()
                    try:
                        data = json.loads(raw.decode("utf-8", errors="replace"))
                    except ValueError as e:
                        raise LLMError(self._redact(f"non-SSE response is not valid JSON: {e}"), kind="response", status=200, model=model, sent=True) from None
                    if not isinstance(data, dict):
                        raise LLMError("response JSON is not an object", kind="response", status=200, model=model, sent=True)
                    resp = self._parse_completion(model, data, r.headers, want_structured=False)
                    usage, model_used, finish_reason = resp.usage, resp.model_used, resp.finish_reason
                    native_finish, gen_id = resp.native_finish_reason, resp.id
                    if resp.content:
                        pieces.append(resp.content)
                        yield resp.content
                    return
                data_lines: list[str] = []
                done = False

                def _dispatch(payload: str) -> str | None:
                    """Handle one SSE event payload; return content to yield (or None). Raises on error events."""
                    nonlocal usage, model_used, finish_reason, native_finish, gen_id, done
                    if payload.strip() == "[DONE]":
                        done = True
                        return None
                    try:
                        chunk = json.loads(payload)
                    except ValueError as e:
                        raise LLMError(self._redact(f"malformed SSE chunk: {e}"), kind="stream", status=200, model=model, usage=usage, sent=True) from None
                    if not isinstance(chunk, dict):
                        raise LLMError("SSE chunk is not a JSON object", kind="stream", status=200, model=model, usage=usage, sent=True)
                    # usage before the error check: an error frame may carry what was billed up to the failure
                    if isinstance(chunk.get("usage"), dict):
                        usage = self._parse_usage(chunk["usage"])
                    err = self._body_error(model, chunk, kind="stream", usage=usage)
                    if err is not None:
                        raise err
                    if isinstance(chunk.get("model"), str):
                        model_used = self._redact(chunk["model"])
                    if isinstance(chunk.get("id"), str):
                        gen_id = self._redact(chunk["id"])
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        return None
                    ch = choices[0]
                    fr = ch.get("finish_reason")
                    if isinstance(fr, str) and fr:
                        finish_reason = self._redact(fr)
                    nfr = ch.get("native_finish_reason")
                    if isinstance(nfr, str) and nfr:
                        native_finish = self._redact(nfr)
                    delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                    calls = delta.get("tool_calls")
                    if isinstance(calls, list):
                        for j, frag in enumerate(calls):
                            if not isinstance(frag, dict):
                                continue
                            idx = frag.get("index", j)
                            idx = idx if isinstance(idx, int) else j
                            slot = tool_frags.setdefault(idx, {"arguments": ""})
                            if isinstance(frag.get("id"), str):
                                slot["id"] = frag["id"]
                            fn = frag.get("function") if isinstance(frag.get("function"), dict) else {}
                            if isinstance(fn.get("name"), str):
                                slot["name"] = fn["name"]
                            if isinstance(fn.get("arguments"), str):
                                slot["arguments"] += fn["arguments"]
                    text = self._content_text(delta.get("content"))
                    return text if text else None

                for line in r.iter_lines():
                    if done:
                        break
                    if line == "":
                        if data_lines:
                            payload = "\n".join(data_lines)
                            data_lines = []
                            out = _dispatch(payload)
                            if out:
                                out = redactor.push(out)  # the key may straddle deltas: a possible prefix is held back
                            if out:
                                pieces.append(out)
                                yield out
                        continue
                    if line.startswith(":"):
                        continue  # keepalive comment (": OPENROUTER PROCESSING")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].removeprefix(" "))
                        continue
                    # "event:", "id:", "retry:" and unknown fields are ignored
                if data_lines and not done:  # stream ended without a trailing blank line
                    out = _dispatch("\n".join(data_lines))
                    if out:
                        out = redactor.push(out)
                    if out:
                        pieces.append(out)
                        yield out
                tail = redactor.flush()
                if tail:
                    pieces.append(tail)
                    yield tail
        except httpx.HTTPError as e:
            err = self._transport_error(model, e)
            err.usage = usage
            raise err from None
        finally:
            _finish()

    # ------------------------------------------------------- account helpers

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            r = self._http.get(url, params=params)
        except httpx.HTTPError as e:
            raise self._transport_error(None, e) from None
        if r.status_code != 200:
            raise self._http_error(None, r.status_code, r.headers, r.content)
        try:
            data = r.json()
        except ValueError as e:
            raise LLMError(self._redact(f"response is not valid JSON: {e}"), kind="response", status=200, sent=True) from None
        if not isinstance(data, dict):
            raise LLMError("response JSON is not an object", kind="response", status=200, sent=True)
        err = self._body_error(None, data)
        if err is not None:
            raise err
        return self._redact(data)

    def key_info(self, endpoint: str = "key") -> dict[str, Any]:
        """``GET /key`` (or ``/auth/key``, the documented twin) -> the ``data`` object (label, limit, limit_remaining, usage, ...)."""
        if endpoint not in ("key", "auth/key"):
            raise ValueError("endpoint must be 'key' or 'auth/key'")
        data = self._get_json(endpoint)
        inner = data.get("data")
        return inner if isinstance(inner, dict) else data

    def generation(self, generation_id: str) -> dict[str, Any]:
        """``GET /generation?id=`` -> the ``data`` object (total_cost, native token counts, ...).

        The record may lag the completion by a few seconds; a 404 surfaces as ``LLMError(status=404)``.
        """
        data = self._get_json("generation", params={"id": generation_id})
        inner = data.get("data")
        return inner if isinstance(inner, dict) else data

    def list_models(self) -> list[dict[str, Any]]:
        """``GET /models`` -> the ``data`` list (public; pricing values are decimal strings in USD per unit)."""
        data = self._get_json("models")
        models = data.get("data")
        return models if isinstance(models, list) else []
