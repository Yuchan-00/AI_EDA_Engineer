"""A local, in-process fake of OpenRouter's HTTP API for tests.

It speaks the documented contract of ``POST /api/v1/chat/completions`` so that
:class:`ai_eda.llm.openrouter.OpenRouterClient` can be tested end to end
without a key and without the network:

* ``Authorization: Bearer <key>`` is checked on every endpoint except the public
  ``/models``. A missing header answers exactly like the live service
  (``401 {"error":{"message":"No cookie auth credentials found","code":401}}``),
  a wrong key with ``401 {"error":{"message":"User not found.","code":401}}``.
* Non-streaming answers are ``chat.completion`` JSON with ``usage`` (tokens +
  ``cost`` in credits) and the ``X-Generation-Id`` / ``X-Provider-Name`` headers.
* ``stream: true`` answers are ``text/event-stream``: ``: OPENROUTER PROCESSING``
  keepalive comment lines, ``data:`` chunks with the first one carrying
  ``delta.role``, the terminal content chunk carrying ``finish_reason``, a final
  usage chunk (``choices: [{delta: {content: "", role: "assistant"}, finish_reason}]``
  plus ``usage``) and the literal ``data: [DONE]``. Tool calls stream as
  OpenAI-style fragments accumulated by ``index``.
* Errors are ``{"error": {"code": <int>, "message": str, "metadata"?: {...}}}``
  with the HTTP status equal to ``code``; 429 carries ``Retry-After`` and the
  ``X-RateLimit-*`` headers; a mid-stream failure is an SSE event with a
  top-level ``error`` and ``finish_reason: "error"`` at HTTP 200.
* ``GET /api/v1/key`` (and ``/auth/key``), ``GET /api/v1/generation?id=`` and
  ``GET /api/v1/models`` return the documented shapes; generation records are
  created for every completion the fake served.

Answers are scripted per call with :meth:`FakeOpenRouter.script` and the
``add_*`` helpers (status, body, delay, malformed bytes, served model); an
empty queue answers with an echo of the last user message. Every request is
recorded as a :class:`RecordedRequest` for assertions. The server binds
``127.0.0.1`` on an ephemeral port; use it as a context manager or call
``start()`` / ``stop()``.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

DEFAULT_API_KEY = "sk-or-v1-fake-test-key-0123456789abcdef"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
PROVIDER_NAME = "FakeProvider"
KEEPALIVE_LINE = ": OPENROUTER PROCESSING"

#: model-level pricing as the live ``/models`` endpoint reported it on 2026-09-22 (USD per token, decimal strings)
PRICING: dict[str, dict[str, str]] = {
    "anthropic/claude-sonnet-5": {
        "prompt": "0.000002", "completion": "0.00001", "web_search": "0.01",
        "input_cache_read": "0.0000002", "input_cache_write": "0.0000025", "input_cache_write_1h": "0.000004",
    },
    "anthropic/claude-haiku-4.5": {
        "prompt": "0.000001", "completion": "0.000005", "web_search": "0.01",
        "input_cache_read": "0.0000001", "input_cache_write": "0.00000125", "input_cache_write_1h": "0.000002",
    },
}

NO_AUTH_BODY = {"error": {"message": "No cookie auth credentials found", "code": 401}}
BAD_KEY_BODY = {"error": {"message": "User not found.", "code": 401}}
NOT_FOUND_BODY = {"error": {"message": "Not Found", "code": 404}}
EXPOSE_HEADERS = "X-Generation-Id,X-Provider-Name,request-id,cf-ray"


@dataclass
class RecordedRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]  # lower-cased names
    body: bytes
    json: Any = None  # parsed body when it was valid JSON

    @property
    def authorization(self) -> str | None:
        return self.headers.get("authorization")

    @property
    def is_chat(self) -> bool:
        return self.method == "POST" and self.path.endswith("/chat/completions")

    @property
    def stream(self) -> bool:
        return bool(isinstance(self.json, dict) and self.json.get("stream"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass
class Scripted:
    """One scripted answer to the next ``/chat/completions`` request.

    With ``status == 200`` and neither ``body`` nor ``raw`` set, a completion is
    built from ``content`` / ``tool_calls`` / ``usage`` and rendered as JSON or
    SSE according to the request's ``stream`` flag.
    """

    status: int = 200
    body: Any = None  # explicit JSON body (dict) - sent verbatim
    raw: bytes | None = None  # explicit raw bytes - sent verbatim (malformed responses)
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0  # seconds before the status line
    chunk_delay: float = 0.0  # seconds between SSE chunks
    # --- completion parameters
    content: str | None = None  # None -> echo of the last user message
    tool_calls: list[dict[str, Any]] | None = None  # [{"name", "arguments": str | dict, "id"?}]
    model: str | None = None  # served model; None -> the requested one (``model`` or ``models[0]``)
    finish_reason: str | None = None  # None -> "tool_calls" when tool calls are present, else "stop"
    usage: dict[str, Any] | None = None  # overrides for the usage object (merged over the computed one)
    cost: float | None = None  # explicit cost; None -> computed from the fake pricing
    omit_cost: bool = False
    omit_usage: bool = False
    reasoning: str | None = None
    stream_pieces: int = 3
    keepalives: int = 1
    usage_frame_style: str = "openrouter"  # "openrouter": choices repeated; "openai": choices []
    mid_stream_error: dict[str, Any] | None = None  # emitted after the first content chunk
    mid_stream_error_usage: bool = False  # the error frame also carries the ``usage`` object (tokens + cost so far)
    committed_error: dict[str, Any] | None = None  # non-stream 200 body with a top-level error


class FakeOpenRouter:
    def __init__(self, api_key: str = DEFAULT_API_KEY, host: str = "127.0.0.1") -> None:
        self.api_key = api_key
        self.host = host
        self.requests: list[RecordedRequest] = []
        self.generations: dict[str, dict[str, Any]] = {}
        self.credits_used: float = 0.0
        self.key_limit: float | None = 10.0
        self._queue: deque[Scripted] = deque()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> FakeOpenRouter:
        if self._server is not None:
            return self
        fake = self

        class Handler(_Handler):
            pass

        Handler.fake = fake  # type: ignore[attr-defined]
        self._server = ThreadingHTTPServer((self.host, 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> FakeOpenRouter:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/api/v1"

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self._queue.clear()
            self.generations.clear()
            self.credits_used = 0.0

    # ------------------------------------------------------------ scripting

    def script(self, item: Scripted | None = None, /, **kw: Any) -> Scripted:
        s = item if item is not None else Scripted(**kw)
        with self._lock:
            self._queue.append(s)
        return s

    def add_completion(self, content: str | None = None, **kw: Any) -> Scripted:
        return self.script(content=content, **kw)

    def add_tool_call(self, name: str, arguments: str | dict[str, Any], *, call_id: str | None = None,
                      content: str | None = None, **kw: Any) -> Scripted:
        call: dict[str, Any] = {"name": name, "arguments": arguments}
        if call_id:
            call["id"] = call_id
        return self.script(content=content if content is not None else "", tool_calls=[call], **kw)

    def add_error(self, status: int, message: str, *, code: int | str | None = None, retry_after: float | None = None,
                  metadata: dict[str, Any] | None = None, headers: dict[str, str] | None = None, **kw: Any) -> Scripted:
        err: dict[str, Any] = {"code": status if code is None else code, "message": message}
        if metadata is not None:
            err["metadata"] = metadata
        h = dict(headers or {})
        if retry_after is not None:
            h["Retry-After"] = str(int(retry_after)) if float(retry_after).is_integer() else str(retry_after)
        return self.script(status=status, body={"error": err}, headers=h, **kw)

    def add_rate_limited(self, retry_after: float = 5, message: str = "Rate limit exceeded", **kw: Any) -> Scripted:
        headers = {"X-RateLimit-Limit": "40", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(time.time()) + int(retry_after))}
        return self.add_error(429, message, retry_after=retry_after, headers=headers, **kw)

    def add_server_error(self, message: str = "Internal Server Error", status: int = 500, **kw: Any) -> Scripted:
        return self.add_error(status, message, **kw)

    def add_provider_down(self, message: str = "Provider returned error", **kw: Any) -> Scripted:
        return self.add_error(502, message, metadata={"error_type": "upstream", "provider_name": PROVIDER_NAME}, **kw)

    def add_insufficient_credits(self, message: str = "Insufficient credits", limit_source: str = "openrouter_credits", **kw: Any) -> Scripted:
        return self.add_error(402, message, metadata={"limit_source": limit_source, "remedy_hint": "add credits"}, **kw)

    def add_moderation_flag(self, flagged: str = "...", **kw: Any) -> Scripted:
        return self.add_error(403, "Input flagged", metadata={
            "reasons": ["policy"], "flagged_input": flagged[:100], "provider_name": PROVIDER_NAME, "model_slug": DEFAULT_MODEL,
        }, **kw)

    def add_invalid_json(self, raw: bytes = b"<html>502 Bad Gateway</html>", status: int = 200,
                         content_type: str = "application/json", **kw: Any) -> Scripted:
        return self.script(status=status, raw=raw, content_type=content_type, **kw)

    def add_committed_error(self, message: str = "upstream failed after commit", code: int = 502,
                            partial: str = "partial", **kw: Any) -> Scripted:
        return self.script(content=partial, committed_error={"code": code, "message": message}, **kw)

    def add_mid_stream_error(self, message: str = "provider disconnected", code: int | str = 502,
                             content: str = "partial answer", *, with_usage: bool = False, **kw: Any) -> Scripted:
        """A stream that fails after its first content chunk; ``with_usage=True`` puts ``usage`` on the error frame."""
        return self.script(content=content, mid_stream_error={"code": code, "message": message}, mid_stream_error_usage=with_usage, **kw)

    def add_fallback_served(self, served_model: str, content: str | None = None, **kw: Any) -> Scripted:
        """The request's primary model was unavailable; ``served_model`` answered (echoed in ``response.model``)."""
        return self.script(content=content, model=served_model, **kw)

    # ------------------------------------------------------------ recording

    @property
    def chat_requests(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.is_chat]

    @property
    def last_request(self) -> RecordedRequest:
        return self.requests[-1]

    # ------------------------------------------------------------ internals

    def _next(self) -> Scripted:
        with self._lock:
            return self._queue.popleft() if self._queue else Scripted()

    def _record(self, req: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(req)

    def _new_generation_id(self) -> str:
        return "gen-" + secrets.token_urlsafe(18)

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _pricing(self, model: str) -> dict[str, str]:
        return PRICING.get(model, PRICING[DEFAULT_MODEL])

    def _usage(self, req: dict[str, Any], model: str, content: str, s: Scripted) -> dict[str, Any] | None:
        if s.omit_usage:
            return None
        prompt = self._approx_tokens(json.dumps(req.get("messages", []), ensure_ascii=False))
        completion = self._approx_tokens(content) if content else 1
        u: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        if s.usage:
            u.update(s.usage)
            if "total_tokens" not in s.usage:
                u["total_tokens"] = int(u["prompt_tokens"]) + int(u["completion_tokens"])
        if not s.omit_cost and "cost" not in u:
            if s.cost is not None:
                u["cost"] = s.cost
            else:
                p = self._pricing(model)
                cost = Decimal(u["prompt_tokens"]) * Decimal(p["prompt"]) + Decimal(u["completion_tokens"]) * Decimal(p["completion"])
                u["cost"] = float(cost)
        u.setdefault("is_byok", False)
        u.setdefault("cost_details", {"upstream_inference_cost": None})
        u.setdefault("prompt_tokens_details", {"cached_tokens": 0, "cache_write_tokens": 0})
        u.setdefault("completion_tokens_details", {"reasoning_tokens": 0})
        return u

    @staticmethod
    def _last_user_text(req: dict[str, Any]) -> str:
        for m in reversed(req.get("messages", []) or []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    return "".join(p.get("text", "") for p in c if isinstance(p, dict))
                return c if isinstance(c, str) else ""
        return ""

    @staticmethod
    def _served_model(req: dict[str, Any], s: Scripted) -> str:
        if s.model:
            return s.model
        if isinstance(req.get("model"), str):
            return req["model"]
        models = req.get("models")
        if isinstance(models, list) and models and isinstance(models[0], str):
            return models[0]
        return DEFAULT_MODEL

    @staticmethod
    def _tool_calls_json(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for i, c in enumerate(calls):
            args = c.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append({"id": c.get("id") or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": args}})
        return out

    def _record_generation(self, gen_id: str, model: str, usage: dict[str, Any] | None, streamed: bool, finish: str) -> None:
        cost = (usage or {}).get("cost")
        rec = {
            "id": gen_id, "upstream_id": "chatcmpl-" + secrets.token_hex(6), "model": model, "provider_name": PROVIDER_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(), "generation_time": 120, "latency": 150, "moderation_latency": None,
            "tokens_prompt": (usage or {}).get("prompt_tokens"), "tokens_completion": (usage or {}).get("completion_tokens"),
            "native_tokens_prompt": (usage or {}).get("prompt_tokens"), "native_tokens_completion": (usage or {}).get("completion_tokens"),
            "native_tokens_reasoning": 0, "native_tokens_cached": 0,
            "total_cost": cost if cost is not None else 0.0, "usage": cost if cost is not None else 0.0,
            "upstream_inference_cost": None, "cache_discount": None, "streamed": streamed, "cancelled": False,
            "finish_reason": finish, "native_finish_reason": finish, "origin": "fake", "is_byok": False, "app_id": None,
        }
        with self._lock:
            self.generations[gen_id] = rec
            if cost is not None:
                self.credits_used += float(cost)

    def _key_info(self) -> dict[str, Any]:
        with self._lock:
            used = self.credits_used
        return {"data": {
            "label": "fake-key", "limit": self.key_limit, "limit_reset": None,
            "limit_remaining": None if self.key_limit is None else self.key_limit - used, "include_byok_in_limit": False,
            "usage": used, "usage_daily": used, "usage_weekly": used, "usage_monthly": used,
            "byok_usage": 0, "byok_usage_daily": 0, "byok_usage_weekly": 0, "byok_usage_monthly": 0,
            "is_free_tier": False, "free_model_daily_requests": {"used": 0, "limit": 50, "remaining": 50},
            # older fields, still emitted so clients prove they treat them as optional
            "rate_limit": {"requests": 40, "interval": "10s"}, "is_provisioning_key": False,
        }}

    def _models(self) -> dict[str, Any]:
        data = []
        for mid, pricing in PRICING.items():
            data.append({
                "id": mid, "canonical_slug": mid + "-fake", "hugging_face_id": None, "name": mid.split("/")[1], "created": 1750000000,
                "description": "fake", "context_length": 200000,
                "architecture": {"modality": "text->text", "input_modalities": ["text"], "output_modalities": ["text"], "tokenizer": "Claude", "instruct_type": None},
                "pricing": dict(pricing), "top_provider": {"context_length": 200000, "max_completion_tokens": 64000, "is_moderated": True},
                "per_request_limits": None,
                "supported_parameters": ["max_tokens", "tools", "tool_choice", "response_format", "structured_outputs", "temperature"],
                "default_parameters": {}, "supported_voices": None, "knowledge_cutoff": None, "expiration_date": None,
                "links": {"details": f"/api/v1/models/{mid}/endpoints"},
            })
        return {"data": data, "total_count": len(data), "links": {"next": None}}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake: FakeOpenRouter  # set per server by FakeOpenRouter.start()

    # ----------------------------------------------------------- plumbing

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature fixed by the base class
        pass

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    def _record(self) -> RecordedRequest:
        parts = urlsplit(self.path)
        body = self._read_body() if self.command in ("POST", "PUT", "PATCH") else b""
        parsed: Any = None
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                parsed = None
        req = RecordedRequest(
            method=self.command, path=parts.path,
            query={k: v[-1] for k, v in parse_qs(parts.query).items()},
            headers={k.lower(): v for k, v in self.headers.items()}, body=body, json=parsed,
        )
        self.fake._record(req)
        return req

    def _send_json(self, status: int, obj: Any, headers: dict[str, str] | None = None, content_type: str = "application/json") -> None:
        self._send_raw(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), headers, content_type)

    def _send_raw(self, status: int, payload: bytes, headers: dict[str, str] | None = None, content_type: str = "application/json") -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Expose-Headers", EXPOSE_HEADERS)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True

    def _authorized(self, req: RecordedRequest) -> bool:
        auth = req.authorization
        if not auth:
            self._send_json(401, NO_AUTH_BODY)
            return False
        if auth != f"Bearer {self.fake.api_key}":
            self._send_json(401, BAD_KEY_BODY)
            return False
        return True

    # ----------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        req = self._record()
        path = req.path
        if path == "/api/v1/models":
            self._send_json(200, self.fake._models(), {"Cache-Control": "public, max-age=120"})
            return
        if not self._authorized(req):
            return
        if path in ("/api/v1/key", "/api/v1/auth/key"):
            self._send_json(200, self.fake._key_info())
            return
        if path == "/api/v1/generation":
            gen = self.fake.generations.get(req.query.get("id", ""))
            if gen is None:
                self._send_json(404, NOT_FOUND_BODY)
            else:
                self._send_json(200, {"data": gen})
            return
        self._send_json(404, NOT_FOUND_BODY)

    def do_POST(self) -> None:  # noqa: N802
        req = self._record()
        if req.path != "/api/v1/chat/completions":
            self._send_json(404, NOT_FOUND_BODY)
            return
        if not self._authorized(req):
            return
        if not isinstance(req.json, dict):
            self._send_json(400, {"error": {"code": 400, "message": "Invalid JSON body"}})
            return
        s = self.fake._next()
        if s.delay:
            time.sleep(s.delay)
        if s.raw is not None:
            self._send_raw(s.status, s.raw, s.headers, s.content_type or "application/json")
            return
        if s.body is not None or s.status != 200:
            self._send_json(s.status, s.body if s.body is not None else {"error": {"code": s.status, "message": "scripted error"}},
                            s.headers, s.content_type or "application/json")
            return
        if req.stream:
            self._send_stream(req.json, s)
        else:
            self._send_completion(req.json, s)

    # ----------------------------------------------------------- completions

    def _send_completion(self, req: dict[str, Any], s: Scripted) -> None:
        fake = self.fake
        gen_id = fake._new_generation_id()
        model = fake._served_model(req, s)
        if s.tool_calls:
            content = s.content
        else:
            content = s.content if s.content is not None else "echo: " + fake._last_user_text(req)
        finish = s.finish_reason or ("tool_calls" if s.tool_calls else "stop")
        usage = fake._usage(req, model, content or "", s)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if s.tool_calls:
            message["content"] = content if content else None
            message["tool_calls"] = fake._tool_calls_json(s.tool_calls)
        if s.reasoning is not None:
            message["reasoning"] = s.reasoning
        body: dict[str, Any] = {
            "id": gen_id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "native_finish_reason": finish, "logprobs": None}],
        }
        if s.committed_error is not None:
            body["error"] = dict(s.committed_error)
            body["choices"][0]["finish_reason"] = "error"
            body["choices"][0]["native_finish_reason"] = "error"
            finish = "error"
        if usage is not None:
            body["usage"] = usage
        fake._record_generation(gen_id, model, usage, streamed=False, finish=finish)
        headers = {"X-Generation-Id": gen_id, "X-Provider-Name": PROVIDER_NAME, **s.headers}
        self._send_json(200, body, headers, s.content_type or "application/json")

    @staticmethod
    def _split(text: str, n: int) -> list[str]:
        if not text:
            return [""]
        n = max(1, min(n, len(text)))
        size = -(-len(text) // n)
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _send_stream(self, req: dict[str, Any], s: Scripted) -> None:
        fake = self.fake
        gen_id = fake._new_generation_id()
        model = fake._served_model(req, s)
        content = s.content
        if content is None and not s.tool_calls:
            content = "echo: " + fake._last_user_text(req)
        finish = s.finish_reason or ("tool_calls" if s.tool_calls else "stop")
        usage = fake._usage(req, model, content or "", s)
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish_reason: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
            c: dict[str, Any] = {
                "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if finish_reason:
                c["choices"][0]["native_finish_reason"] = finish_reason
            if extra:
                c.update(extra)
            return c

        frames: list[str] = [KEEPALIVE_LINE] * max(0, s.keepalives)
        events: list[dict[str, Any]] = []
        pieces = self._split(content or "", s.stream_pieces)
        for i, piece in enumerate(pieces):
            delta: dict[str, Any] = {"content": piece}
            if i == 0:
                delta["role"] = "assistant"
            last = i == len(pieces) - 1 and not s.tool_calls
            events.append(chunk(delta, finish if last else None))
            if s.mid_stream_error is not None and i == 0:
                err_frame: dict[str, Any] = {
                    "id": gen_id, "error": dict(s.mid_stream_error),
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
                }
                if s.mid_stream_error_usage and usage is not None:
                    err_frame["usage"] = usage
                events.append(err_frame)
                break
        if s.tool_calls and s.mid_stream_error is None:
            calls = fake._tool_calls_json(s.tool_calls)
            for idx, call in enumerate(calls):
                args = call["function"]["arguments"]
                half = max(1, len(args) // 2)
                first = {"index": idx, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": args[:half]}}
                rest = {"index": idx, "function": {"arguments": args[half:]}}
                events.append(chunk({"tool_calls": [first]}))
                events.append(chunk({"tool_calls": [rest]}, finish if idx == len(calls) - 1 else None))
        if s.mid_stream_error is None and usage is not None:
            if s.usage_frame_style == "openai":
                events.append({"id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage})
            else:
                events.append(chunk({"content": "", "role": "assistant"}, finish, {"usage": usage}))
        fake._record_generation(gen_id, model, usage, streamed=True, finish="error" if s.mid_stream_error else finish)

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Expose-Headers", EXPOSE_HEADERS)
            self.send_header("X-Generation-Id", gen_id)
            self.send_header("X-Provider-Name", PROVIDER_NAME)
            for k, v in s.headers.items():
                self.send_header(k, v)
            self.end_headers()
            for line in frames:
                self.wfile.write((line + "\n\n").encode("utf-8"))
            self.wfile.flush()
            for ev in events:
                if s.chunk_delay:
                    time.sleep(s.chunk_delay)
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        self.close_connection = True
