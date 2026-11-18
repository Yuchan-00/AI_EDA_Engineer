"""The OpenAI-compatible client against the local fake HTTP server (tests/fake_openrouter.py on 127.0.0.1): no network, no key.

What is pinned: only the documented chat-completions fields are sent (no
OpenRouter extras), the key is optional on this machine and required
elsewhere, ``usage.cost`` is never read, a loopback server is free (known
zero charge, LOCAL_MODEL_USE approval, no budget) and a remote one is billed
per call (unknown cost, PAID_API_CALL budget).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

import ai_eda.llm.openai_compat as openai_compat
import tests.fake_openrouter as fake_openrouter
from ai_eda.errors import ApprovalRequiredError, ToolUnavailableError
from ai_eda.llm.client import LLMError, LLMMessage
from ai_eda.llm.openai_compat import OpenAICompatClient, check_compat_base_url
from ai_eda.llm.providers import compat_billing, is_loopback_host
from ai_eda.llm.router import TaskKind, default_router
from ai_eda.llm.service import LLMBudget, LLMService
from ai_eda.security import ApprovalGate
from ai_eda.security.approval import ExternalAction
from tests.fake_openrouter import DEFAULT_API_KEY, FakeOpenRouter

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"], "additionalProperties": False}


class Word(BaseModel):
    word: str


@pytest.fixture
def server():
    with FakeOpenRouter() as s:
        yield s


@pytest.fixture
def keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local server that asks for no key (llama-server / Ollama default): the fake accepts any request."""
    monkeypatch.setattr(fake_openrouter._Handler, "_authorized", lambda self, req: True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _user(text: str = "U") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=text)]


def test_loopback_hosts_and_billing():
    for host in ("localhost", "LOCALHOST", "127.0.0.1", "127.3.2.1", "::1", "[::1]"):
        assert is_loopback_host(host), host
    for host in ("192.168.0.10", "api.openai.com", "", None, "localhost.example.com"):
        assert not is_loopback_host(host), host
    assert compat_billing("http://127.0.0.1:8080/v1") == compat_billing("http://[::1]:11434/v1") == "local"
    assert compat_billing("https://api.openai.com/v1") == compat_billing(None) == compat_billing("not a url") == "per_call"


def test_base_url_is_explicit_and_https_or_this_machine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AI_EDA_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ToolUnavailableError, match="--llm-openai-base-url"):
        OpenAICompatClient()
    with pytest.raises(ToolUnavailableError, match="must be https://"):
        check_compat_base_url("http://192.168.1.5:8080/v1")
    assert check_compat_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert check_compat_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    monkeypatch.setenv("AI_EDA_OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    assert OpenAICompatClient().base_url == "http://127.0.0.1:1/v1"


def test_the_key_is_required_remotely_and_optional_here(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ToolUnavailableError, match="OPENAI_API_KEY is not set"):
        OpenAICompatClient("https://api.openai.com/v1")
    monkeypatch.setenv("MY_KEY", "sk-test-secret-0123456789")
    remote = OpenAICompatClient("https://api.openai.com/v1", openai_key_env="MY_KEY")
    assert remote.billing == "per_call" and remote.paid is True and "sk-test" not in repr(remote) and "<set>" in repr(remote)
    local = OpenAICompatClient("http://127.0.0.1:1/v1")
    assert local.billing == "local" and local.paid is False and "<none>" in repr(local)
    assert "Authorization" not in local._http.headers
    remote.close()
    local.close()


def test_local_call_sends_only_documented_fields_and_is_a_known_zero_charge(server: FakeOpenRouter, keyless: None):
    server.add_completion('{"word":"OK"}', cost=0.5)
    client = OpenAICompatClient(server.base_url)
    resp = client.complete("llama3.1:8b", _user(), response_schema=SCHEMA, max_tokens=64)
    req = server.chat_requests[-1]
    assert req.authorization is None
    assert set(req.json) == {"model", "messages", "temperature", "stream", "max_tokens", "response_format"}
    assert req.json["model"] == "llama3.1:8b" and req.json["stream"] is False and req.json["max_tokens"] == 64
    assert req.json["response_format"] == {"type": "json_schema", "json_schema": {"name": "ai_eda_response", "strict": True, "schema": SCHEMA}}
    assert resp.structured == {"word": "OK"}
    # usage.cost is not part of the OpenAI contract: never read; a local call is a known zero charge
    assert resp.usage.cost_usd == 0.0 and resp.usage.cost_source == "local" and resp.usage.billing == "local" and resp.usage.prompt_tokens > 0
    client.close()


def test_stream_asks_for_the_usage_chunk(server: FakeOpenRouter, keyless: None):
    server.add_completion("hello world", usage_frame_style="openai")
    client = OpenAICompatClient(server.base_url)
    assert "".join(client.stream("m", _user())) == "hello world"
    req = server.chat_requests[-1]
    assert req.json["stream_options"] == {"include_usage": True} and "usage" not in req.json and "reasoning" not in req.json
    assert client.last_stream_usage is not None and client.last_stream_usage.billing == "local"
    client.close()


def test_remote_route_sends_the_key_reads_no_cost_and_redacts(server: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_compat, "compat_billing", lambda url: "per_call")  # the loopback fake stands in for a remote endpoint
    monkeypatch.setenv("MY_KEY", DEFAULT_API_KEY)
    client = OpenAICompatClient(server.base_url, openai_key_env="MY_KEY")
    server.add_completion("hi", cost=0.5)
    resp = client.complete("gpt-x", _user())
    assert server.chat_requests[-1].authorization == f"Bearer {DEFAULT_API_KEY}"
    assert resp.usage.cost_usd is None and resp.usage.billing == "per_call"  # unknown, never 0 and never the unsanctioned 0.5
    server.add_error(500, f"upstream echoed {DEFAULT_API_KEY}")
    with pytest.raises(LLMError) as ei:
        client.complete("gpt-x", _user())
    assert DEFAULT_API_KEY not in str(ei.value) and "[REDACTED]" in str(ei.value) and ei.value.status == 500
    client.close()


def test_openai_compat_has_no_default_model():
    with pytest.raises(ValueError, match="no default model"):
        default_router(default_provider="openai_compat")
    assert default_router("openai_compat:llama3.1:8b", default_provider="openai_compat").default.model == "llama3.1:8b"


def test_service_on_a_local_server_needs_no_budget_and_says_it_is_free(server: FakeOpenRouter, keyless: None):
    gate = ApprovalGate()
    svc = LLMService.from_env(LLMBudget(), router=default_router("openai_compat:llama3", default_provider="openai_compat"),
                              providers=("openai_compat",), gate=gate, openai_base_url=server.base_url)
    assert not svc.budget_required and svc.local_providers == ["openai_compat"] and svc.per_call_providers == []
    grant = gate.audit[0]
    assert grant["event"] == "grant" and grant["action"] == ExternalAction.LOCAL_MODEL_USE and grant["by"] == "cli --llm openai_compat"
    assert server.base_url in grant["detail"] and "free" in grant["detail"] and "no budget required" in grant["detail"]
    assert gate.audit[1]["event"] == "consume"
    server.add_completion('{"word":"OK"}', cost=0.5)
    server.add_completion('{"word":"again"}', omit_usage=True)
    word, _ = svc.structured(TaskKind.REQUIREMENT_ANALYSIS, _user(), Word)
    again, _ = svc.structured(TaskKind.REQUIREMENT_ANALYSIS, _user(), Word)
    assert (word.word, again.word) == ("OK", "again")
    first, second = svc.usage.records
    assert first.usage.cost_usd == 0.0 and first.usage.billing == "local" and first.via == "openai_compat"
    # the server reported no usage: still a known zero charge on a local route, never an unknown cost
    assert second.usage.cost_usd == 0.0 and second.usage.billing == "local" and second.tokens == 0
    assert svc.spent() == (0.0, 0, first.tokens)
    assert "(not required: no per-call provider)" in svc.summary()
    svc.client.close()


def test_service_on_a_remote_endpoint_needs_the_budget(server: FakeOpenRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_compat, "compat_billing", lambda url: "per_call")
    monkeypatch.setenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    router = default_router("openai_compat:gpt-x", default_provider="openai_compat")
    with pytest.raises(ApprovalRequiredError):
        LLMService.from_env(LLMBudget(), router=router, providers=("openai_compat",), gate=ApprovalGate(), openai_base_url=server.base_url)
    svc = LLMService.from_env(LLMBudget(max_tokens=10_000), router=router, providers=("openai_compat",), gate=ApprovalGate(), openai_base_url=server.base_url)
    assert svc.per_call_providers == ["openai_compat"] and svc.local_providers == []
    server.add_completion('{"word":"OK"}', cost=0.5)
    svc.structured(TaskKind.REQUIREMENT_ANALYSIS, _user(), Word)
    [record] = svc.usage.records
    assert record.charge and record.usage.cost_usd is None  # the endpoint reports no cost: unknown, bounded by the token budget
    svc.client.close()
