"""Self-test of tests/fake_openrouter.py against the documented OpenRouter contract, using raw httpx."""

from __future__ import annotations

import json
import time
from typing import Iterator

import httpx
import pytest

from tests.fake_openrouter import (
    BAD_KEY_BODY,
    DEFAULT_MODEL,
    KEEPALIVE_LINE,
    NO_AUTH_BODY,
    NOT_FOUND_BODY,
    PRICING,
    FakeOpenRouter,
)


@pytest.fixture(scope="module")
def server() -> Iterator[FakeOpenRouter]:
    with FakeOpenRouter() as s:
        yield s


@pytest.fixture
def fake(server: FakeOpenRouter) -> FakeOpenRouter:
    server.reset()
    return server


@pytest.fixture(scope="module")
def http(server: FakeOpenRouter) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=server.base_url, headers={"Authorization": f"Bearer {server.api_key}"}, timeout=5.0, verify=False) as c:
        yield c


@pytest.fixture(scope="module")
def anon(server: FakeOpenRouter) -> Iterator[httpx.Client]:
    """No Authorization header."""
    with httpx.Client(base_url=server.base_url, timeout=5.0, verify=False) as c:
        yield c


def _chat(text: str = "hi", **extra):
    body = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return body


def _sse_frames(text: str) -> list[str]:
    return [f for f in text.split("\n\n") if f]


# ----------------------------------------------------------------------------- auth


def test_missing_authorization_matches_live_body(fake: FakeOpenRouter, anon: httpx.Client):
    r = anon.post("/chat/completions", json=_chat())
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == NO_AUTH_BODY
    # also for a streaming request: the live service answers JSON, not SSE, before committing
    r = anon.post("/chat/completions", json=_chat(stream=True))
    assert r.status_code == 401 and r.json() == NO_AUTH_BODY
    # and on the account endpoints
    assert anon.get("/key").json() == NO_AUTH_BODY and anon.get("/generation", params={"id": "x"}).json() == NO_AUTH_BODY


def test_wrong_key(fake: FakeOpenRouter, anon: httpx.Client):
    r = anon.post("/chat/completions", json=_chat(), headers={"Authorization": "Bearer sk-or-v1-invalid"})
    assert r.status_code == 401
    assert r.json() == BAD_KEY_BODY


def test_models_is_public(fake: FakeOpenRouter, anon: httpx.Client):
    r = anon.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert data["total_count"] == len(PRICING) and data["links"] == {"next": None}
    sonnet = next(m for m in data["data"] if m["id"] == "anthropic/claude-sonnet-5")
    assert sonnet["pricing"]["prompt"] == "0.000002" and isinstance(sonnet["pricing"]["completion"], str)


def test_unknown_path_404(http: httpx.Client):
    assert http.get("/nope").json() == NOT_FOUND_BODY
    assert http.post("/nope", json={}).json() == NOT_FOUND_BODY


# ------------------------------------------------------------------- non-stream


def test_default_echo_completion_shape(fake: FakeOpenRouter, http: httpx.Client):
    r = http.post("/chat/completions", json=_chat("안녕 🙂 world"))
    assert r.status_code == 200
    assert r.headers["x-provider-name"] and r.headers["x-generation-id"].startswith("gen-")
    assert "X-Generation-Id" in r.headers["access-control-expose-headers"]
    d = r.json()
    assert d["object"] == "chat.completion" and d["model"] == DEFAULT_MODEL and d["id"] == r.headers["x-generation-id"]
    ch = d["choices"][0]
    assert ch["index"] == 0 and ch["finish_reason"] == "stop" and ch["message"]["role"] == "assistant"
    assert ch["message"]["content"] == "echo: 안녕 🙂 world"
    u = d["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert isinstance(u["cost"], float) and u["cost"] > 0
    assert u["prompt_tokens_details"]["cached_tokens"] == 0 and u["is_byok"] is False
    # request was recorded with headers and parsed body
    req = fake.last_request
    assert req.is_chat and not req.stream and req.json["messages"][0]["content"] == "안녕 🙂 world"
    assert req.headers["authorization"] == f"Bearer {fake.api_key}"


def test_scripted_content_usage_and_cost(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_completion("fixed", usage={"prompt_tokens": 1000, "completion_tokens": 200}, model="anthropic/claude-sonnet-5")
    d = http.post("/chat/completions", json=_chat()).json()
    assert d["choices"][0]["message"]["content"] == "fixed"
    assert d["model"] == "anthropic/claude-sonnet-5"
    assert d["usage"]["prompt_tokens"] == 1000 and d["usage"]["completion_tokens"] == 200
    assert d["usage"]["cost"] == pytest.approx(0.004)  # 1000 * 2e-6 + 200 * 1e-5
    fake.add_completion("no cost", omit_cost=True)
    d = http.post("/chat/completions", json=_chat()).json()
    assert "cost" not in d["usage"]
    fake.add_completion("no usage", omit_usage=True)
    d = http.post("/chat/completions", json=_chat()).json()
    assert "usage" not in d


def test_queue_is_consumed_in_order_then_echo(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_completion("one")
    fake.add_completion("two")
    assert http.post("/chat/completions", json=_chat("a")).json()["choices"][0]["message"]["content"] == "one"
    assert http.post("/chat/completions", json=_chat("b")).json()["choices"][0]["message"]["content"] == "two"
    assert http.post("/chat/completions", json=_chat("c")).json()["choices"][0]["message"]["content"] == "echo: c"
    assert len(fake.chat_requests) == 3


def test_tool_call_arguments_are_a_json_string(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_tool_call("lookup", {"mpn": "RC0603FR-0710KL"}, call_id="call_abc")
    d = http.post("/chat/completions", json=_chat(tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}])).json()
    ch = d["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None
    tc = ch["message"]["tool_calls"][0]
    assert tc == {"id": "call_abc", "type": "function", "function": {"name": "lookup", "arguments": '{"mpn": "RC0603FR-0710KL"}'}}
    fake.add_tool_call("lookup", '{"mpn": ')  # malformed, passed through verbatim
    d = http.post("/chat/completions", json=_chat()).json()
    assert d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"mpn": '


def test_fallback_served_model_is_echoed(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_fallback_served("anthropic/claude-haiku-4.5", "from fallback")
    body = {"models": ["anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5"], "messages": [{"role": "user", "content": "x"}]}
    d = http.post("/chat/completions", json=body).json()
    assert d["model"] == "anthropic/claude-haiku-4.5"
    # without a script, models[0] is the primary
    d = http.post("/chat/completions", json=body).json()
    assert d["model"] == "anthropic/claude-sonnet-5"


# ------------------------------------------------------------------------ errors


def test_error_helpers(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_rate_limited(retry_after=7)
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 429 and r.headers["retry-after"] == "7" and r.headers["x-ratelimit-remaining"] == "0"
    assert r.json()["error"]["code"] == 429
    fake.add_server_error()
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 500 and r.json()["error"] == {"code": 500, "message": "Internal Server Error"}
    fake.add_insufficient_credits()
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 402 and r.json()["error"]["metadata"]["limit_source"] == "openrouter_credits"
    fake.add_moderation_flag("bad input")
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 403 and r.json()["error"]["metadata"]["flagged_input"] == "bad input"
    fake.add_provider_down()
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 502 and r.json()["error"]["metadata"]["provider_name"]
    fake.add_error(408, "Request timed out", retry_after=1.5)
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 408 and r.headers["retry-after"] == "1.5"


def test_invalid_json_and_committed_error(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_invalid_json(b"not json at all")
    r = http.post("/chat/completions", json=_chat())
    assert r.status_code == 200 and r.text == "not json at all"
    with pytest.raises(ValueError):
        r.json()
    fake.add_committed_error("upstream died", code=502, partial="half")
    d = http.post("/chat/completions", json=_chat()).json()
    assert d["error"] == {"code": 502, "message": "upstream died"}
    assert d["choices"][0]["finish_reason"] == "error" and d["choices"][0]["message"]["content"] == "half"


def test_delay_is_honoured(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_completion("slow", delay=0.3)
    t0 = time.monotonic()
    assert http.post("/chat/completions", json=_chat()).status_code == 200
    assert time.monotonic() - t0 >= 0.25


def test_invalid_request_body_400(fake: FakeOpenRouter, http: httpx.Client):
    r = http.post("/chat/completions", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == 400


# ---------------------------------------------------------------------- streaming


def test_sse_framing(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_completion("Hello, streaming world", stream_pieces=3, keepalives=2)
    r = http.post("/chat/completions", json=_chat(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-generation-id"].startswith("gen-")
    frames = _sse_frames(r.text)
    assert frames[0] == KEEPALIVE_LINE and frames[1] == KEEPALIVE_LINE
    assert frames[-1] == "data: [DONE]"
    data = [json.loads(f[len("data: "):]) for f in frames[2:-1]]
    assert all(d["object"] == "chat.completion.chunk" and d["model"] == DEFAULT_MODEL for d in data)
    assert data[0]["choices"][0]["delta"]["role"] == "assistant"
    content_chunks = data[:-1]
    assert "".join(c["choices"][0]["delta"]["content"] for c in content_chunks) == "Hello, streaming world"
    assert [c["choices"][0]["finish_reason"] for c in content_chunks] == [None, None, "stop"]
    usage_frame = data[-1]
    assert usage_frame["choices"] == [{"index": 0, "delta": {"content": "", "role": "assistant"}, "finish_reason": "stop", "native_finish_reason": "stop"}]
    assert usage_frame["usage"]["cost"] > 0 and usage_frame["usage"]["total_tokens"] > 0
    assert fake.last_request.stream


def test_sse_openai_style_usage_frame(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_completion("x", stream_pieces=1, usage_frame_style="openai")
    frames = _sse_frames(http.post("/chat/completions", json=_chat(stream=True)).text)
    usage_frame = json.loads(frames[-2][len("data: "):])
    assert usage_frame["choices"] == [] and "usage" in usage_frame


def test_sse_tool_call_fragments(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_tool_call("lookup", {"mpn": "ABC-123"}, call_id="call_1")
    frames = _sse_frames(http.post("/chat/completions", json=_chat(stream=True)).text)
    data = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: {")]
    frags = [c["choices"][0]["delta"]["tool_calls"][0] for c in data if "tool_calls" in c["choices"][0]["delta"]]
    assert frags[0]["index"] == 0 and frags[0]["id"] == "call_1" and frags[0]["function"]["name"] == "lookup"
    assert "id" not in frags[1] and "name" not in frags[1]["function"]
    assert "".join(f["function"]["arguments"] for f in frags) == '{"mpn": "ABC-123"}'
    finishes = [c["choices"][0]["finish_reason"] for c in data]
    assert finishes[-2] == "tool_calls" and finishes[-1] == "tool_calls"  # terminal + usage frame


def test_sse_mid_stream_error(fake: FakeOpenRouter, http: httpx.Client):
    fake.add_mid_stream_error("provider disconnected", code=502, content="partial answer")
    frames = _sse_frames(http.post("/chat/completions", json=_chat(stream=True)).text)
    data = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: {")]
    assert data[0]["choices"][0]["delta"]["content"]  # some content arrived first
    err = data[1]
    assert err["error"] == {"code": 502, "message": "provider disconnected"}
    assert err["choices"][0]["finish_reason"] == "error"
    assert frames[-1] == "data: [DONE]"
    assert not any("usage" in d for d in data)


def test_sse_unicode_survives_chunking(fake: FakeOpenRouter, http: httpx.Client):
    text = "저항 분압기 🙂 입력 12 V → 출력 6 V"
    fake.add_completion(text, stream_pieces=7)
    frames = _sse_frames(http.post("/chat/completions", json=_chat(stream=True)).text)
    data = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: {")]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in data) == text


# ------------------------------------------------------------- key / generation


def test_key_info_and_generation_record(fake: FakeOpenRouter, http: httpx.Client):
    before = http.get("/key").json()["data"]
    assert before["usage"] == 0 and before["limit_remaining"] == fake.key_limit
    assert before["free_model_daily_requests"]["limit"] == 50
    assert http.get("/auth/key").status_code == 200
    r = http.post("/chat/completions", json=_chat())
    gen_id = r.headers["x-generation-id"]
    cost = r.json()["usage"]["cost"]
    rec = http.get("/generation", params={"id": gen_id}).json()["data"]
    assert rec["id"] == gen_id and rec["total_cost"] == pytest.approx(cost) and rec["streamed"] is False
    assert rec["tokens_prompt"] == r.json()["usage"]["prompt_tokens"] and rec["finish_reason"] == "stop"
    assert http.get("/generation", params={"id": "gen-unknown"}).json() == NOT_FOUND_BODY
    after = http.get("/key").json()["data"]
    assert after["usage"] == pytest.approx(cost) and after["limit_remaining"] == pytest.approx(fake.key_limit - cost)
