"""Any OpenAI-compatible chat-completions server (``openai_compat:<model>``): one HTTP route for the OpenAI API and local servers.

The OpenAI API, llama.cpp's ``llama-server``, Ollama, vLLM and LM Studio all
serve ``POST {base_url}/chat/completions``; this client speaks only that
documented contract: ``model``, ``messages``, ``temperature``,
``max_tokens``, ``tools`` (function tools, JSON-string ``arguments``),
``response_format: {"type": "json_schema", "json_schema": {"name", "strict",
"schema"}}``, ``stream`` with ``stream_options: {"include_usage": true}``
(SSE ``data:`` chunks ending in ``data: [DONE]``) and the ``usage`` object
(``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
``prompt_tokens_details.cached_tokens``,
``completion_tokens_details.reasoning_tokens``). It is the OpenRouter
client (:class:`~ai_eda.llm.openrouter.OpenRouterClient`: the same request
building, error typing, key redaction and response parsing) minus
everything OpenRouter adds: no ``usage.include`` / ``reasoning`` /
``provider`` body fields, no attribution headers, and ``usage.cost`` is not
read (no OpenAI-compatible server documents it).

Invariants enforced here:

* **The base URL is explicit** (``openai_base_url`` / ``--llm-openai-base-url``,
  else ``AI_EDA_OPENAI_BASE_URL``; there is no default endpoint) and must be
  ``https://`` - or ``http://`` on this machine (``localhost``, ``127.0.0.0/8``,
  ``::1``), where llama.cpp / Ollama / LM Studio listen.
* **The key comes from an environment variable whose *name* the user gives**
  (``openai_key_env`` / ``--llm-openai-key-env``, default ``OPENAI_API_KEY``);
  it is sent as ``Authorization: Bearer`` only when set, and it is optional
  only on this machine. Like the OpenRouter client, every string that leaves
  this module is redacted, so the key is never in a log, an error or a
  response (with no key there is nothing to redact).
* **Billing follows the base URL** (:func:`~ai_eda.llm.providers.compat_billing`):
  a remote endpoint is ``per_call`` (``paid``; the run needs a budget, and
  since the server reports no cost every call's cost is *unknown* - a USD
  budget alone then stops after the first call, so give a token budget); a
  server on this machine is ``local`` (``paid=False``; no budget required, the
  service records the ``LOCAL_MODEL_USE`` approval saying it is free, and a
  call is a known zero charge). A local server that forwards to a paid
  upstream (a proxy) is not seen here: its cost is the proxy's business.

Known limits (not measured: no OpenAI-compatible server was called while
this was written; the tests use a local fake): OpenAI's reasoning models
reject ``max_tokens`` (they take ``max_completion_tokens``) and a
``temperature`` other than their default, so this route serves the
non-reasoning models of the OpenAI API; whether a given local server honours
``response_format: json_schema`` strictly depends on the server (the service
validates every reply against the schema either way).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMMessage, ToolSpec, Usage
from ai_eda.llm.openrouter import OpenRouterClient, _ssl_context
from ai_eda.llm.providers import LOCAL_BILLING, OPENAI_COMPAT_ENV_BASE_URL, compat_billing, is_loopback_host

#: the key's environment variable when the user names none
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
#: ``Usage.cost_source`` of a call on a local server (a known zero charge)
LOCAL_COST_SOURCE = LOCAL_BILLING


def check_compat_base_url(url: str) -> str:
    """``url`` without its trailing slash, or :class:`~ai_eda.errors.ToolUnavailableError`: ``https://``, or ``http://`` on this machine."""
    base = (url or "").strip().rstrip("/")
    parts = urlsplit(base)
    host = parts.hostname or ""
    if parts.scheme == "https" and host:
        return base
    if parts.scheme == "http" and is_loopback_host(host):
        return base
    raise ToolUnavailableError(
        f"--llm-openai-base-url / {OPENAI_COMPAT_ENV_BASE_URL} must be https:// (or http:// on this machine: localhost, 127.0.0.1, ::1): got {base!r}"
    )


class OpenAICompatClient(OpenRouterClient):
    """HTTP client for an OpenAI-compatible server. See the module docstring for the invariants.

    The options are named ``openai_*`` so that the OpenRouter client's
    ``base_url`` / ``api_key`` (routed to every member by constructor
    signature) never reach this one, and the reverse.
    """

    def __init__(
        self,
        openai_base_url: str | None = None,
        *,
        openai_key_env: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        schema_name: str = "ai_eda_response",
    ) -> None:
        given = openai_base_url if openai_base_url is not None else os.environ.get(OPENAI_COMPAT_ENV_BASE_URL, "")
        if not (given or "").strip():
            raise ToolUnavailableError(
                f"openai_compat needs the server's base URL: pass --llm-openai-base-url URL (e.g. http://127.0.0.1:8080/v1 for "
                f"llama-server, http://localhost:11434/v1 for Ollama, https://api.openai.com/v1) or set {OPENAI_COMPAT_ENV_BASE_URL}"
            )
        self.base_url = check_compat_base_url(given)
        #: ``"local"`` on this machine (free, no budget), ``"per_call"`` otherwise
        self.billing = compat_billing(self.base_url)
        self.paid = self.billing != LOCAL_BILLING
        self.key_env = (openai_key_env or DEFAULT_KEY_ENV).strip()
        key = os.environ.get(self.key_env, "").strip() if self.key_env else ""
        if not key and self.paid:
            raise ToolUnavailableError(f"{self.key_env or 'the key variable'} is not set: the remote endpoint {self.base_url} needs its API key "
                                       "(the key is optional only on this machine); --llm-openai-key-env names another variable")
        self._api_key = key
        self.app_url = None
        self.app_title = None
        self.reasoning = None
        self.provider = None
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.schema_name = schema_name
        self.last_stream_usage: Usage | None = None
        self.last_stream_response = None
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            headers=self._headers(),
            verify=_ssl_context(),
        )

    def __repr__(self) -> str:
        return f"OpenAICompatClient(base_url={self.base_url!r}, key_env={self.key_env!r}, api_key=<{'set' if self._api_key else 'none'}>, billing={self.billing!r})"

    __str__ = __repr__

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _redact(self, value: Any) -> Any:
        return super()._redact(value) if self._api_key else value

    def _held_prefix_len(self, text: str) -> int:
        return super()._held_prefix_len(text) if self._api_key else 0

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
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
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
        return body

    def _parse_usage(self, u: Any) -> Usage:  # type: ignore[override]
        """The documented ``usage`` fields; cost ``0.0`` (known, ``local``) on this machine, ``None`` (unknown) elsewhere - never read."""
        base = OpenRouterClient._parse_usage({k: v for k, v in u.items() if k not in ("cost", "is_byok")} if isinstance(u, dict) else u)
        if not self.paid:
            return base.model_copy(update={"cost_usd": 0.0, "cost_source": LOCAL_COST_SOURCE, "billing": LOCAL_BILLING})
        return base.model_copy(update={"billing": "per_call"})


__all__ = ["DEFAULT_KEY_ENV", "LOCAL_COST_SOURCE", "OpenAICompatClient", "check_compat_base_url"]
