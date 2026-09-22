"""OpenRouterClient against the local fake server (tests/fake_openrouter.py). No key, no network.

Everything that spends real credits lives in tests/test_openrouter_live.py (skipped without a key).
"""

from __future__ import annotations

import json
import logging
import pickle
from typing import Iterator

import pytest

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMError, LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from ai_eda.llm.fake import ScriptedLLMClient
from ai_eda.llm.openrouter import OPENROUTER_BASE_URL, REDACTED, OpenRouterClient
from ai_eda.llm.usage import UsageTracker
from ai_eda.llm.router import TaskKind
from tests.fake_openrouter import DEFAULT_MODEL, PROVIDER_NAME, FakeOpenRouter

MODEL = DEFAULT_MODEL


@pytest.fixture(scope="module")
def server() -> Iterator[FakeOpenRouter]:
    with FakeOpenRouter() as s:
        yield s


@pytest.fixture
def fake(server: FakeOpenRouter) -> FakeOpenRouter:
    server.reset()
    return server


@pytest.fixture
def client(fake: FakeOpenRouter) -> Iterator[OpenRouterClient]:
    c = OpenRouterClient(fake.api_key, fake.base_url, app_url="https://example.test/ai-eda", app_title="AI EDA ENGINEER", timeout=5.0)
    yield c
    c.close()


def _msgs(text: str = "hello", system: str | None = None) -> list[LLMMessage]:
    out = []
    if system:
        out.append(LLMMessage(role="system", content=system))
    out.append(LLMMessage(role="user", content=text))
    return out


# --------------------------------------------------------------- construction


def test_key_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ToolUnavailableError):
        OpenRouterClient()
    with pytest.raises(ToolUnavailableError):
        OpenRouterClient("   ")
    assert not OpenRouterClient.key_present()


def test_key_from_env_and_never_in_repr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret-from-env")
    assert OpenRouterClient.key_present()
    c = OpenRouterClient()
    try:
        assert c.base_url == OPENROUTER_BASE_URL
        assert "secret" not in repr(c) and "secret" not in str(c)
        assert "api_key=<set>" in repr(c)
        assert not hasattr(c, "api_key")
    finally:
        c.close()


# ------------------------------------------------------------- request shape


def test_request_shape(fake: FakeOpenRouter, client: OpenRouterClient):
    schema = {"type": "object", "properties": {"v": {"type": "number"}}, "required": ["v"], "additionalProperties": False}
    tools = [ToolSpec(name="lookup", description="look up a part", parameters={"type": "object", "properties": {"mpn": {"type": "string"}}})]
    fake.add_completion('{"v": 6}')
    resp = client.complete(MODEL, _msgs("what is v?", system="You are a tool."), tools=tools, response_schema=schema, temperature=0.2, max_tokens=64)
    req = fake.last_request
    assert req.headers["authorization"] == f"Bearer {fake.api_key}"
    assert req.headers["content-type"].startswith("application/json")
    assert req.headers["http-referer"] == "https://example.test/ai-eda"
    assert req.headers["x-openrouter-title"] == "AI EDA ENGINEER" and req.headers["x-title"] == "AI EDA ENGINEER"
    body = req.json
    assert body["model"] == MODEL and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "You are a tool."}, {"role": "user", "content": "what is v?"}]
    assert body["usage"] == {"include": True}
    assert body["temperature"] == 0.2 and body["max_tokens"] == 64
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "ai_eda_response", "strict": True, "schema": schema}}
    assert body["tools"] == [{"type": "function", "function": {"name": "lookup", "description": "look up a part", "parameters": tools[0].parameters}}]
    assert "reasoning" not in body and "provider" not in body
    assert resp.structured == {"v": 6}


def test_optional_request_fields(fake: FakeOpenRouter):
    c = OpenRouterClient(fake.api_key, fake.base_url, timeout=5.0, reasoning={"effort": "none"}, provider={"require_parameters": True})
    try:
        c.complete(MODEL, _msgs())
        body = fake.last_request.json
        assert body["reasoning"] == {"effort": "none"} and body["provider"] == {"require_parameters": True}
        assert "max_tokens" not in body and "tools" not in body and "response_format" not in body
        assert "http-referer" not in fake.last_request.headers
    finally:
        c.close()


def test_tool_call_round_trip_serialization(fake: FakeOpenRouter, client: OpenRouterClient):
    call = ToolCall(id="call_1", name="lookup", arguments={"mpn": "RC0603"})
    bad = ToolCall(id="call_2", name="lookup", arguments={}, raw_arguments='{"mpn": ', parse_error="not json")
    msgs = [
        LLMMessage(role="user", content="find it"),
        LLMMessage(role="assistant", content=None, tool_calls=[call, bad]),
        LLMMessage(role="tool", content='{"found": true}', tool_call_id="call_1", name="lookup"),
    ]
    client.complete(MODEL, msgs)
    sent = fake.last_request.json["messages"]
    assert sent[1]["content"] is None
    assert sent[1]["tool_calls"][0] == {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"mpn": "RC0603"}'}}
    assert sent[1]["tool_calls"][1]["function"]["arguments"] == '{"mpn": '  # echoed verbatim, not re-encoded
    assert sent[2] == {"role": "tool", "content": '{"found": true}', "name": "lookup", "tool_call_id": "call_1"}


def test_empty_messages_rejected(client: OpenRouterClient):
    with pytest.raises(ValueError):
        client.complete(MODEL, [])


# ------------------------------------------------------------ happy paths


def test_complete_happy_path(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("A resistive divider.", usage={"prompt_tokens": 1000, "completion_tokens": 200}, model="anthropic/claude-sonnet-5")
    resp = client.complete(MODEL, _msgs())
    assert isinstance(resp, LLMResponse)
    assert resp.content == "A resistive divider."
    assert resp.model == MODEL and resp.model_used == "anthropic/claude-sonnet-5"
    assert resp.finish_reason == "stop" and resp.native_finish_reason == "stop" and not resp.truncated
    assert resp.usage.prompt_tokens == 1000 and resp.usage.completion_tokens == 200 and resp.usage.total_tokens == 1200
    assert resp.usage.cost_usd == pytest.approx(0.004) and resp.usage.cost_source == "provider"
    assert resp.usage.cached_tokens == 0 and resp.usage.reasoning_tokens == 0 and resp.usage.is_byok is False
    assert resp.id and resp.id.startswith("gen-") and resp.generation_id == resp.id
    assert resp.provider_name == PROVIDER_NAME
    assert resp.raw_error is None and resp.tool_calls == [] and resp.structured is None
    assert resp.raw["id"] == resp.id


def test_usage_absent_means_unknown_cost(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("x", omit_cost=True)
    resp = client.complete(MODEL, _msgs())
    assert resp.usage.cost_usd is None and resp.usage.cost_source is None and resp.usage.prompt_tokens > 0
    fake.add_completion("y", omit_usage=True)
    resp = client.complete(MODEL, _msgs())
    assert resp.usage == Usage()
    tracker = UsageTracker()
    tracker.record(TaskKind.CHAT, MODEL, resp.usage)
    assert tracker.total_cost_usd() is None  # unknown stays unknown, never 0


def test_structured_content(fake: FakeOpenRouter, client: OpenRouterClient):
    schema = {"type": "object"}
    fake.add_completion('{"v_out": 6, "unit": "V"}')
    resp = client.complete(MODEL, _msgs(), response_schema=schema)
    assert resp.structured == {"v_out": 6, "unit": "V"} and resp.raw_error is None
    fake.add_completion('{"v_out": 6,')  # truncated JSON
    resp = client.complete(MODEL, _msgs(), response_schema=schema)
    assert resp.structured is None and "not valid JSON" in resp.raw_error
    fake.add_completion("[1, 2]")
    resp = client.complete(MODEL, _msgs(), response_schema=schema)
    assert resp.structured is None and "not an object" in resp.raw_error
    fake.add_completion('{"v": 1}')  # no schema requested -> content stays text
    resp = client.complete(MODEL, _msgs())
    assert resp.structured is None and resp.content == '{"v": 1}'


def test_tool_calls_parsed_and_malformed_kept_raw(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_tool_call("lookup", {"mpn": "RC0603FR-0710KL", "qty": 2}, call_id="call_ok")
    resp = client.complete(MODEL, _msgs())
    assert resp.finish_reason == "tool_calls" and resp.content is None
    assert resp.tool_calls == [ToolCall(id="call_ok", name="lookup", arguments={"mpn": "RC0603FR-0710KL", "qty": 2})]
    assert resp.tool_calls[0].is_valid and resp.raw_error is None

    fake.add_tool_call("lookup", '{"mpn": "RC06', call_id="call_bad")
    resp = client.complete(MODEL, _msgs())  # no exception
    tc = resp.tool_calls[0]
    assert tc.arguments == {} and tc.raw_arguments == '{"mpn": "RC06' and not tc.is_valid
    assert "not valid JSON" in tc.parse_error
    assert "call_bad" in resp.raw_error and "not valid JSON" in resp.raw_error

    fake.add_tool_call("lookup", "[1, 2]", call_id="call_list")
    resp = client.complete(MODEL, _msgs())
    assert resp.tool_calls[0].arguments == {} and "JSON object" in resp.tool_calls[0].parse_error


def test_reasoning_field_is_kept_but_not_content(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("answer", reasoning="thinking...")
    resp = client.complete(MODEL, _msgs())
    assert resp.content == "answer" and resp.reasoning == "thinking..."


def test_unicode_round_trip(fake: FakeOpenRouter, client: OpenRouterClient):
    text = "12 V 입력을 6 V로 나누는 저항 분압기 🙂 ±1 % — µF"
    resp = client.complete(MODEL, _msgs(text))
    assert fake.last_request.json["messages"][0]["content"] == text
    assert fake.last_request.body.decode("utf-8")  # the wire is UTF-8
    assert resp.content == "echo: " + text
    got = "".join(client.stream(MODEL, _msgs(text)))
    assert got == "echo: " + text


# ----------------------------------------------------------------- streaming


def test_stream_happy_path(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("Hello, streaming world", stream_pieces=4, keepalives=3, usage={"prompt_tokens": 10, "completion_tokens": 5}, cost=0.00025)
    pieces = list(client.stream(MODEL, _msgs("hi"), temperature=0.5, max_tokens=32))
    assert len(pieces) == 4 and "".join(pieces) == "Hello, streaming world"
    assert fake.last_request.json["stream"] is True and fake.last_request.json["max_tokens"] == 32
    u = client.last_stream_usage
    assert u is not None and u.prompt_tokens == 10 and u.completion_tokens == 5 and u.cost_usd == pytest.approx(0.00025)
    r = client.last_stream_response
    assert r is not None and r.content == "Hello, streaming world" and r.finish_reason == "stop"
    assert r.model_used == MODEL and r.id and r.id.startswith("gen-") and r.generation_id == r.id
    assert r.provider_name == PROVIDER_NAME and r.usage == u


def test_stream_openai_style_usage_frame(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("abc", stream_pieces=1, usage_frame_style="openai")
    assert "".join(client.stream(MODEL, _msgs())) == "abc"
    assert client.last_stream_usage is not None and client.last_stream_usage.cost_usd is not None


def test_stream_without_usage_frame(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("abc", stream_pieces=2, omit_usage=True)
    assert "".join(client.stream(MODEL, _msgs())) == "abc"
    assert client.last_stream_usage is None
    assert client.last_stream_response is not None and client.last_stream_response.usage.cost_usd is None


def test_stream_tool_call_fragments_accumulated(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_tool_call("lookup", {"mpn": "ABC-123"}, call_id="call_1")
    assert list(client.stream(MODEL, _msgs())) == []
    r = client.last_stream_response
    assert r is not None and r.finish_reason == "tool_calls"
    assert r.tool_calls == [ToolCall(id="call_1", name="lookup", arguments={"mpn": "ABC-123"})]


def test_stream_mid_stream_error(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_mid_stream_error("provider disconnected", code=502, content="partial answer", stream_pieces=3)
    got: list[str] = []
    with pytest.raises(LLMError) as ei:
        for piece in client.stream(MODEL, _msgs()):
            got.append(piece)
    assert got and "".join(got) != "partial answer"  # partial content arrived, then the error
    e = ei.value
    assert e.kind == "stream" and e.status == 200 and e.code == 502 and e.retryable
    assert "provider disconnected" in str(e)
    assert client.last_stream_usage is None


def test_stream_mid_stream_error_with_string_code(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_mid_stream_error("x", code="503")
    with pytest.raises(LLMError) as ei:
        list(client.stream(MODEL, _msgs()))
    assert ei.value.code == 503 and ei.value.retryable


def test_stream_http_error_before_commit(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_rate_limited(retry_after=3)
    with pytest.raises(LLMError) as ei:
        list(client.stream(MODEL, _msgs()))
    assert ei.value.status == 429 and ei.value.retry_after == 3 and ei.value.rate_limited


def test_stream_non_sse_200_body_is_one_completion(fake: FakeOpenRouter, client: OpenRouterClient):
    body = {"id": "gen-x", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "whole"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.000006}}
    fake.script(body=body)
    assert list(client.stream(MODEL, _msgs())) == ["whole"]
    assert client.last_stream_usage is not None and client.last_stream_usage.cost_usd == pytest.approx(0.000006)


def test_stream_early_close_is_clean(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_completion("one two three four", stream_pieces=4)
    it = client.stream(MODEL, _msgs())
    first = next(it)
    it.close()
    assert first
    assert client.last_stream_response is not None and client.last_stream_response.content == first
    # the client is still usable afterwards
    fake.add_completion("again")
    assert client.complete(MODEL, _msgs()).content == "again"


# -------------------------------------------------------------------- errors


def test_429_surfaces_retry_after(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_rate_limited(retry_after=7, message="Rate limit exceeded")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    e = ei.value
    assert e.kind == "http" and e.status == 429 and e.code == 429 and e.retry_after == 7.0
    assert e.rate_limited and e.retryable and not e.insufficient_credits
    assert e.message == "Rate limit exceeded" and "status=429" in str(e) and "retry_after=7s" in str(e)


def test_retry_after_http_date(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_error(503, "No provider", headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status == 503 and ei.value.retry_after == 0.0 and ei.value.retryable


def test_500_and_402_are_typed(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_server_error()
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status == 500 and ei.value.code == 500 and ei.value.retryable and ei.value.retry_after is None
    fake.add_insufficient_credits()
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    e = ei.value
    assert e.status == 402 and e.code == 402 and e.insufficient_credits and not e.retryable
    assert e.metadata["limit_source"] == "openrouter_credits"
    fake.add_moderation_flag("nope")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status == 403 and ei.value.metadata["flagged_input"] == "nope" and not ei.value.retryable


def test_401_bodies(fake: FakeOpenRouter):
    c = OpenRouterClient("sk-or-v1-wrong", fake.base_url, timeout=5.0)
    try:
        with pytest.raises(LLMError) as ei:
            c.complete(MODEL, _msgs())
        assert ei.value.status == 401 and ei.value.message == "User not found." and not ei.value.retryable
    finally:
        c.close()


def test_200_with_error_body_is_an_error(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_committed_error("upstream failed after commit", code=502, partial="half an answer")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.kind == "response" and ei.value.status == 200 and ei.value.code == 502 and ei.value.retryable
    # a choice-level error without a top-level one
    fake.script(body={"id": "gen-y", "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                                                  "finish_reason": "error", "error": {"code": 500, "message": "boom"}}]})
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == 500 and "boom" in str(ei.value)
    # finish_reason error alone
    fake.script(body={"id": "gen-z", "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "error"}]})
    with pytest.raises(LLMError):
        client.complete(MODEL, _msgs())


def test_malformed_bodies(fake: FakeOpenRouter, client: OpenRouterClient):
    fake.add_invalid_json(b"<html>oops</html>")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.kind == "response" and ei.value.status == 200 and "not valid JSON" in str(ei.value)
    fake.add_invalid_json(b"<html>Bad Gateway</html>", status=502, content_type="text/html")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status == 502 and ei.value.code is None and "Bad Gateway" in str(ei.value)
    fake.script(body={"id": "gen-q", "model": MODEL, "choices": []})
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert "no choices" in str(ei.value)
    fake.script(body=[1, 2, 3])
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert "not an object" in str(ei.value)


def test_timeout_is_transport_error(fake: FakeOpenRouter):
    c = OpenRouterClient(fake.api_key, fake.base_url, timeout=0.2, connect_timeout=1.0)
    try:
        fake.add_completion("late", delay=1.0)
        with pytest.raises(LLMError) as ei:
            c.complete(MODEL, _msgs())
        assert ei.value.kind == "transport" and ei.value.status is None and ei.value.retryable
        assert "timeout" in str(ei.value)
    finally:
        c.close()


def test_connection_refused_is_transport_error():
    c = OpenRouterClient("sk-or-v1-x", "http://127.0.0.1:9/api/v1", timeout=0.5, connect_timeout=0.3)
    try:
        with pytest.raises(LLMError) as ei:
            c.complete(MODEL, _msgs())
        assert ei.value.kind == "transport" and ei.value.retryable
        with pytest.raises(LLMError) as ei:
            list(c.stream(MODEL, _msgs()))
        assert ei.value.kind == "transport"
    finally:
        c.close()


# ---------------------------------------------------------------- key hygiene


def test_key_never_appears_in_errors_even_when_echoed(fake: FakeOpenRouter, client: OpenRouterClient, caplog: pytest.LogCaptureFixture):
    key = fake.api_key
    fake.add_error(401, f"invalid key {key} rejected", metadata={"echo": key, "nested": [key, {"k": key}]}, headers={"X-Debug": key})
    caplog.set_level(logging.DEBUG, logger="ai_eda.llm.openrouter")
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    e = ei.value
    assert key not in str(e) and key not in repr(e) and key not in e.message
    assert key not in json.dumps(e.metadata)
    assert e.metadata["echo"] == REDACTED and e.metadata["nested"] == [REDACTED, {"k": REDACTED}]
    assert all(key not in r.getMessage() for r in caplog.records)
    # a successful body that echoes the key is scrubbed from ``raw`` too
    fake.add_completion(f"your key is {key}")
    resp = client.complete(MODEL, _msgs())
    assert key not in json.dumps(resp.raw)
    # a mid-stream error that echoes the key
    fake.add_mid_stream_error(f"bad {key}")
    with pytest.raises(LLMError) as ei:
        list(client.stream(MODEL, _msgs()))
    assert key not in str(ei.value)


def test_no_content_logged_at_info(fake: FakeOpenRouter, client: OpenRouterClient, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="ai_eda.llm.openrouter")
    fake.add_completion("SECRET-ANSWER-9f3")
    client.complete(MODEL, _msgs("SECRET-PROMPT-7a1"))
    list(client.stream(MODEL, _msgs("SECRET-PROMPT-7a1")))
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    assert "openrouter complete" in text and "openrouter stream" in text
    assert "SECRET-ANSWER" not in text and "SECRET-PROMPT" not in text and fake.api_key not in text


# ---------------------------------------------------------------- account


def test_key_info_and_generation(fake: FakeOpenRouter, client: OpenRouterClient):
    info = client.key_info()
    assert info["label"] == "fake-key" and info["limit_remaining"] == fake.key_limit
    resp = client.complete(MODEL, _msgs())
    gen = client.generation(resp.generation_id)
    assert gen["id"] == resp.id and gen["total_cost"] == pytest.approx(resp.usage.cost_usd)
    with pytest.raises(LLMError) as ei:
        client.generation("gen-does-not-exist")
    assert ei.value.status == 404 and ei.value.message == "Not Found"
    models = client.list_models()
    assert any(m["id"] == "anthropic/claude-sonnet-5" for m in models)


# ---------------------------------------------------------- scripted client


def test_scripted_client_basics():
    schema = {"type": "object"}
    c = ScriptedLLMClient([
        "plain text",
        {"structured": {"v": 6}, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.001}, "model": "fake/model"},
        {"tool_calls": [{"name": "lookup", "arguments": {"mpn": "X"}}, {"name": "bad", "arguments": "{oops", "id": "c9"}]},
        {"error": {"message": "rate limited", "status": 429, "retry_after": 2}},
        {"content": "after error"},
    ])
    r1 = c.complete("m", _msgs("q1"))
    assert r1.content == "plain text" and r1.model_used == "m" and r1.usage.cost_usd is None
    r2 = c.complete("m", _msgs("q2"), response_schema=schema)
    assert r2.structured == {"v": 6} and r2.content == '{"v": 6}' and r2.model_used == "fake/model"
    assert r2.usage.total_tokens == 12 and r2.usage.cost_usd == 0.001 and r2.usage.cost_source == "script"
    r3 = c.complete("m", _msgs("q3"))
    assert r3.finish_reason == "tool_calls"
    assert r3.tool_calls[0] == ToolCall(id="call_0", name="lookup", arguments={"mpn": "X"})
    assert r3.tool_calls[1].id == "c9" and not r3.tool_calls[1].is_valid and r3.tool_calls[1].raw_arguments == "{oops"
    assert "c9" in r3.raw_error
    with pytest.raises(LLMError) as ei:
        c.complete("m", _msgs("q4"))
    assert ei.value.status == 429 and ei.value.retry_after == 2 and ei.value.retryable
    assert c.complete("m", _msgs("q5")).content == "after error"  # the error was consumed once
    with pytest.raises(LLMError) as ei:
        c.complete("m", _msgs("q6"))
    assert ei.value.kind == "script" and not ei.value.retryable
    assert [call.user_text for call in c.calls] == ["q1", "q2", "q3", "q4", "q5", "q6"]
    assert c.calls[1].response_schema == schema and c.remaining == 0


def test_scripted_client_structured_from_text_and_exceptions():
    c = ScriptedLLMClient(['{"a": 1}', "not json", RuntimeError("boom"), LLMResponse(model="x", content="ready", model_used="y")], repeat_last=True)
    assert c.complete("m", _msgs(), response_schema={}).structured == {"a": 1}
    r = c.complete("m", _msgs(), response_schema={})
    assert r.structured is None and "not valid JSON" in r.raw_error
    with pytest.raises(RuntimeError):
        c.complete("m", _msgs())
    assert c.complete("m", _msgs()).model_used == "y"
    assert c.complete("m", _msgs()).content == "ready"  # repeat_last


def test_scripted_client_stream_and_default():
    c = ScriptedLLMClient(["Hello  streaming\nworld 🙂"], default={"content": "default answer", "usage": {"cost": 0.5}})
    pieces = list(c.stream("m", _msgs()))
    assert len(pieces) > 1 and "".join(pieces) == "Hello  streaming\nworld 🙂"
    assert c.last_stream_usage is not None and c.calls[-1].stream
    assert "".join(c.stream("m", _msgs())) == "default answer"
    assert c.last_stream_usage.cost_usd == 0.5 and c.last_stream_usage.cost_source == "script"


def test_scripted_client_from_file(tmp_path):
    spec = {"responses": ["one", {"structured": {"k": "v"}}, {"error": {"message": "x", "status": 500}}], "repeat_last": False}
    p = tmp_path / "script.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    c = ScriptedLLMClient.from_file(p)
    assert c.complete("m", _msgs()).content == "one"
    assert c.complete("m", _msgs(), response_schema={}).structured == {"k": "v"}
    with pytest.raises(LLMError) as ei:
        c.complete("m", _msgs())
    assert ei.value.status == 500
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert ScriptedLLMClient.from_file(lst).complete("m", _msgs()).content == "a"
    with pytest.raises(TypeError):
        ScriptedLLMClient.from_spec({"responses": "nope"})


def test_llm_error_classification():
    assert LLMError("x", status=503).retryable and LLMError("x", status=408).retryable
    assert not LLMError("x", status=400).retryable and not LLMError("x", status=401).retryable
    assert LLMError("x", kind="response", status=200, code=502).retryable
    assert not LLMError("x", kind="response", status=200, code="weird").retryable
    assert LLMError("x", kind="transport").retryable and LLMError("x", kind="transport").status is None
    original = LLMError("m", status=429, retry_after=1.5, metadata={"a": 1}, model="x")
    copy = pickle.loads(pickle.dumps(original))
    assert (copy.message, copy.status, copy.retry_after, copy.metadata, copy.model) == ("m", 429, 1.5, {"a": 1}, "x")
    assert str(copy) == str(original) == "LLM request failed (http, status=429, retry_after=1.5s, model=x): m"
