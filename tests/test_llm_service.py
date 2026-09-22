"""LLMService (budget, approval, retry / fallback, schema validation) against the local fake OpenRouter.

No key, no network: every HTTP behaviour is scripted on ``tests/fake_openrouter.py``;
the in-process ``ScriptedLLMClient`` covers the transport-error class.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ConfigDict

from ai_eda.errors import ApprovalRequiredError, ToolUnavailableError
from ai_eda.llm.client import LLMError, LLMMessage, Usage
from ai_eda.llm.fake import ScriptedLLMClient
from ai_eda.llm.openrouter import OpenRouterClient
from ai_eda.llm.router import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL, ModelConfig, ModelRouter, TaskKind, default_router
from ai_eda.llm.service import BudgetExceededError, LLMBudget, LLMService, StructuredOutputError
from ai_eda.llm.usage import UsageTracker
from ai_eda.security import ApprovalGate, ExternalAction
from tests.fake_openrouter import FakeOpenRouter

TASK = TaskKind.REQUIREMENT_ANALYSIS
MSGS = [LLMMessage(role="system", content="answer tersely"), LLMMessage(role="user", content="say hi")]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


@pytest.fixture
def fake():
    with FakeOpenRouter() as f:
        yield f


def _client(fake: FakeOpenRouter) -> OpenRouterClient:
    return OpenRouterClient(api_key=fake.api_key, base_url=fake.base_url, timeout=10.0, connect_timeout=2.0)


def _service(fake: FakeOpenRouter, budget: LLMBudget | None = None, *, router: ModelRouter | None = None, **kw) -> LLMService:
    sleeps: list[float] = []
    svc = LLMService(
        _client(fake), router or default_router(), UsageTracker(), budget or LLMBudget(max_usd=1.0),
        gate=kw.pop("gate", ApprovalGate()), sleep=sleeps.append, **kw,
    )
    svc.sleeps = sleeps  # type: ignore[attr-defined]
    return svc


# ------------------------------------------------------------------- approval / budget


def test_construction_records_the_budget_as_a_consumed_approval(fake: FakeOpenRouter):
    gate = ApprovalGate()
    svc = _service(fake, LLMBudget(max_usd=0.5, max_tokens=1000), gate=gate, approved_by="cli --llm-budget-usd")
    events = [(e["event"], e["action"]) for e in gate.audit]
    assert events == [("grant", ExternalAction.PAID_API_CALL), ("consume", ExternalAction.PAID_API_CALL)]
    assert gate.audit[0]["detail"] == "openrouter budget max_usd=0.5 max_tokens=1000" == svc.approval_detail
    assert gate.audit[0]["by"] == "cli --llm-budget-usd"
    assert fake.requests == []  # construction sends nothing


def test_no_budget_is_refused_by_the_gate_before_any_http_call(fake: FakeOpenRouter):
    gate = ApprovalGate()
    with pytest.raises(ApprovalRequiredError, match="paid_api_call"):
        LLMService(_client(fake), default_router(), UsageTracker(), LLMBudget(), gate=gate)
    assert gate.audit == [{"event": "denied", "action": ExternalAction.PAID_API_CALL, "detail": "openrouter budget none"}]
    assert fake.requests == []


def test_cost_budget_exceeded_after_the_first_call_makes_no_second_request(fake: FakeOpenRouter):
    svc = _service(fake, LLMBudget(max_usd=1e-9))
    resp = svc.complete(TASK, MSGS)
    assert resp.usage.cost_usd is not None and resp.usage.cost_usd > 1e-9
    with pytest.raises(BudgetExceededError, match="budget 1e-09 USD"):
        svc.complete(TASK, MSGS)
    assert len(fake.chat_requests) == 1
    assert svc.attempts[-1].outcome == "budget" and not svc.attempts[-1].ok


def test_zero_budget_allows_calls_that_cost_nothing_and_refuses_after_any_cost():
    free = ScriptedLLMClient([{"content": "a", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0}}], repeat_last=True)
    svc = LLMService(free, default_router(), UsageTracker(), LLMBudget(max_usd=0.0), gate=ApprovalGate())
    svc.complete(TASK, MSGS)
    svc.complete(TASK, MSGS)  # spent 0.0 is not > 0.0
    assert len(free.calls) == 2
    paid = ScriptedLLMClient([{"content": "a", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.001}}], repeat_last=True)
    svc = LLMService(paid, default_router(), UsageTracker(), LLMBudget(max_usd=0.0), gate=ApprovalGate())
    svc.complete(TASK, MSGS)  # the first call cannot be priced before it is made
    with pytest.raises(BudgetExceededError):
        svc.complete(TASK, MSGS)
    assert len(paid.calls) == 1


def test_unknown_cost_without_a_token_budget_refuses_the_next_call(fake: FakeOpenRouter):
    fake.add_completion("no cost here", omit_cost=True)
    svc = _service(fake, LLMBudget(max_usd=1.0))
    resp = svc.complete(TASK, MSGS)
    assert resp.usage.cost_usd is None and svc.usage.total_cost_usd() is None
    with pytest.raises(BudgetExceededError, match="unknown and no token budget"):
        svc.complete(TASK, MSGS)
    assert len(fake.chat_requests) == 1


def test_token_budget_covers_calls_whose_cost_is_absent(fake: FakeOpenRouter):
    for _ in range(3):
        fake.add_completion("ok", omit_cost=True, usage={"prompt_tokens": 40, "completion_tokens": 10})
    svc = _service(fake, LLMBudget(max_usd=1.0, max_tokens=120))
    svc.complete(TASK, MSGS)  # 50 tokens
    svc.complete(TASK, MSGS)  # 100 tokens: not > 120
    svc.complete(TASK, MSGS)  # allowed at 100, brings it to 150
    with pytest.raises(BudgetExceededError, match="spent 150 tokens, budget 120 tokens"):
        svc.complete(TASK, MSGS)
    assert len(fake.chat_requests) == 3
    assert "3 served call(s), 150 tokens" in svc.summary() and "+3 call(s) with unknown cost" in svc.summary()


def test_budget_check_applies_to_retries_and_fallbacks_too(fake: FakeOpenRouter):
    fake.add_completion("first")
    fake.add_server_error()
    svc = _service(fake, LLMBudget(max_usd=1e-9))
    svc.complete(TASK, MSGS)
    with pytest.raises(BudgetExceededError):
        svc.complete(TASK, MSGS)  # refused before the scripted 500 is even fetched
    assert len(fake.chat_requests) == 1


# ------------------------------------------------------------------- retry / fallback


def test_429_with_retry_after_is_retried_once_on_the_same_model(fake: FakeOpenRouter):
    fake.add_rate_limited(retry_after=5)
    fake.add_completion("after the wait")
    svc = _service(fake, backoff_cap=2.5)
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "after the wait"
    assert [r.json["model"] for r in fake.chat_requests] == [DEFAULT_PRIMARY_MODEL, DEFAULT_PRIMARY_MODEL]
    assert svc.sleeps == [2.5]  # Retry-After 5 s honoured but capped
    assert [a.outcome for a in svc.last_attempts] == ["http", "served"]
    assert svc.last_attempts[0].status == 429
    assert len(svc.usage.records) == 1  # a refused request served nothing and is not "spend"


def test_persistent_429_falls_back_to_the_next_candidate(fake: FakeOpenRouter):
    fake.add_rate_limited(retry_after=1)
    fake.add_rate_limited(retry_after=1)
    fake.add_completion("haiku answers")
    svc = _service(fake)
    resp = svc.complete(TASK, MSGS)
    assert resp.model_used == DEFAULT_FALLBACK_MODEL
    assert [r.json["model"] for r in fake.chat_requests] == [DEFAULT_PRIMARY_MODEL, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
    assert len(svc.sleeps) == 1


def test_500_falls_back_to_the_next_model_and_reports_model_used(fake: FakeOpenRouter):
    fake.add_server_error()
    fake.add_completion("from the fallback")
    svc = _service(fake)
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "from the fallback"
    assert resp.model == DEFAULT_FALLBACK_MODEL and resp.model_used == DEFAULT_FALLBACK_MODEL
    assert [r.json["model"] for r in fake.chat_requests] == [DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
    assert svc.sleeps == []
    [rec] = svc.usage.records
    assert rec.model == DEFAULT_FALLBACK_MODEL and rec.task == TASK and rec.usage.cost_usd is not None


def test_provider_fallback_served_model_is_what_gets_recorded(fake: FakeOpenRouter):
    fake.add_fallback_served(DEFAULT_FALLBACK_MODEL, content="served elsewhere")
    svc = _service(fake)
    resp = svc.complete(TASK, MSGS)
    assert resp.model == DEFAULT_PRIMARY_MODEL and resp.model_used == DEFAULT_FALLBACK_MODEL
    assert svc.usage.records[0].model == DEFAULT_FALLBACK_MODEL  # billing follows the model that answered


def test_all_candidates_failing_raises_the_last_error(fake: FakeOpenRouter):
    fake.add_provider_down()
    fake.add_server_error(status=503)
    svc = _service(fake)
    with pytest.raises(LLMError) as ei:
        svc.complete(TASK, MSGS)
    assert ei.value.status == 503
    assert len(fake.chat_requests) == 2


@pytest.mark.parametrize("status", [401, 402, 400, 403])
def test_terminal_statuses_are_never_retried_and_never_fall_back(fake: FakeOpenRouter, status: int):
    if status == 401:
        fake.add_error(401, "User not found.")
    elif status == 402:
        fake.add_insufficient_credits()
    elif status == 403:
        fake.add_moderation_flag()
    else:
        fake.add_error(400, "bad request")
    fake.add_completion("must not be reached")
    svc = _service(fake)
    with pytest.raises(LLMError) as ei:
        svc.complete(TASK, MSGS)
    assert ei.value.status == status
    assert len(fake.chat_requests) == 1 and svc.sleeps == []
    assert svc.usage.records == []


def test_transport_failure_falls_back():
    # "connection refused" never reached the provider (sent=False): nothing was billed, nothing is recorded
    client = ScriptedLLMClient([LLMError("connection refused", kind="transport", sent=False), {"content": "second model"}])
    svc = LLMService(client, default_router(), UsageTracker(), LLMBudget(max_usd=1.0), gate=ApprovalGate())
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "second model" and resp.model == DEFAULT_FALLBACK_MODEL
    assert [c.model for c in client.calls] == [DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
    assert [r.outcome for r in svc.usage.records] == ["served"]


def test_a_200_with_an_error_body_is_recorded_with_the_cost_it_reported(fake: FakeOpenRouter):
    fake.add_committed_error()  # the fake's error body carries usage.cost, as the live service does
    fake.add_completion("recovered")
    svc = _service(fake, LLMBudget(max_usd=1.0, max_tokens=10_000))
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "recovered"
    failed, served = svc.usage.records
    assert failed.outcome == "failed" and failed.usage.cost_usd is not None and failed.usage.cost_usd > 0
    assert served.outcome == "served" and served.usage.cost_usd == resp.usage.cost_usd
    assert svc.usage.total_cost_usd() == pytest.approx(failed.usage.cost_usd + served.usage.cost_usd)
    # an error body without usage: may have been billed, cost unknown - never 0
    fake.add_committed_error(omit_usage=True)
    fake.add_completion("recovered again")
    svc.complete(TASK, MSGS)
    assert [r.usage.cost_usd for r in svc.usage.records][2] is None
    assert svc.usage.total_cost_usd() is None


# ------------------------------------------------------------------- structured


def test_structured_invalid_json_then_valid_on_the_feedback_retry(fake: FakeOpenRouter):
    fake.add_completion("Sure! here is prose, not JSON")
    fake.add_completion('{"value": 7}')
    svc = _service(fake)
    inst, resp = svc.structured(TASK, MSGS, Answer)
    assert inst == Answer(value=7) and resp.model_used == DEFAULT_PRIMARY_MODEL
    first, second = fake.chat_requests
    assert first.json["model"] == second.json["model"] == DEFAULT_PRIMARY_MODEL
    assert first.json["response_format"]["type"] == "json_schema" and first.json["response_format"]["json_schema"]["strict"] is True
    assert first.json["response_format"]["json_schema"]["schema"]["properties"] == {"value": {"title": "Value", "type": "integer"}}
    roles = [m["role"] for m in second.json["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert second.json["messages"][2]["content"] == "Sure! here is prose, not JSON"
    assert "rejected" in second.json["messages"][3]["content"] and "not a JSON object" in second.json["messages"][3]["content"]
    assert [a.outcome for a in svc.last_attempts] == ["served", "validation", "served"]
    assert len(svc.usage.records) == 2  # both served replies are spend


def test_structured_schema_violation_is_fed_back_with_the_pydantic_error(fake: FakeOpenRouter):
    fake.add_completion('{"value": "seven", "extra": 1}')
    fake.add_completion('{"value": 7}')
    svc = _service(fake)
    inst, _ = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 7
    feedback = fake.chat_requests[1].json["messages"][3]["content"]
    assert "Extra inputs are not permitted" in feedback and "value" in feedback


def test_structured_truncated_reply_counts_as_a_validation_failure(fake: FakeOpenRouter):
    fake.add_completion('{"value": 1', finish_reason="length")
    fake.add_completion('{"value": 1}')
    svc = _service(fake)
    inst, _ = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 1
    assert "truncated" in fake.chat_requests[1].json["messages"][3]["content"]


def test_structured_fenced_json_from_a_prompted_model_is_accepted(fake: FakeOpenRouter):
    fake.add_completion('```json\n{"value": 3}\n```')
    router = ModelRouter(default=ModelConfig(model=DEFAULT_PRIMARY_MODEL, supports_structured=False))
    svc = _service(fake, router=router)
    inst, _ = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 3
    body = fake.chat_requests[0].json
    assert "response_format" not in body  # the model does not support it: the schema goes into the prompt instead
    assert body["messages"][-1]["role"] == "system" and '"value"' in body["messages"][-1]["content"]
    assert "Respond with a single JSON object" in body["messages"][-1]["content"]


def test_structured_falls_back_after_two_rejections_and_raises_when_all_fail(fake: FakeOpenRouter):
    for _ in range(4):
        fake.add_completion("nope")
    svc = _service(fake)
    with pytest.raises(StructuredOutputError, match="no model produced a reply matching the schema") as ei:
        svc.structured(TASK, MSGS, Answer)
    assert [r.json["model"] for r in fake.chat_requests] == [DEFAULT_PRIMARY_MODEL] * 2 + [DEFAULT_FALLBACK_MODEL] * 2
    assert len(ei.value.attempts) == 8  # 4 served + 4 validation notes
    fake.reset()
    fake.add_completion("nope")
    fake.add_completion("nope")
    fake.add_completion('{"value": 9}')
    svc = _service(fake)
    inst, resp = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 9 and resp.model_used == DEFAULT_FALLBACK_MODEL


def test_structured_transport_failure_then_fallback_validates(fake: FakeOpenRouter):
    fake.add_server_error(status=502)
    fake.add_completion('{"value": 5}')
    svc = _service(fake)
    inst, resp = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 5 and resp.model_used == DEFAULT_FALLBACK_MODEL


def test_structured_custom_json_schema_is_what_gets_sent(fake: FakeOpenRouter):
    fake.add_completion('{"value": 2}')
    svc = _service(fake)
    strict = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}
    svc.structured(TASK, MSGS, Answer, json_schema=strict)
    assert fake.chat_requests[0].json["response_format"]["json_schema"]["schema"] == strict


# ------------------------------------------------------------------- stream / logs / factories


def test_stream_yields_pieces_and_records_usage(fake: FakeOpenRouter):
    fake.add_completion("one two three")
    svc = _service(fake)
    out = "".join(svc.stream(TASK, MSGS))
    assert out == "one two three"
    [rec] = svc.usage.records
    assert rec.usage.completion_tokens > 0 and rec.usage.cost_usd is not None
    assert svc.last_attempts[-1].outcome == "served"


def test_stream_without_usage_frame_records_unknown_cost(fake: FakeOpenRouter):
    fake.add_completion("x y", omit_usage=True)
    svc = _service(fake, LLMBudget(max_usd=1.0, max_tokens=100))
    "".join(svc.stream(TASK, MSGS))
    [rec] = svc.usage.records
    assert rec.usage.cost_usd is None and rec.usage.total_tokens == 0


def test_key_never_appears_in_logs_and_no_content_at_info(fake: FakeOpenRouter, caplog):
    secret_content = "SECRET-CONTENT-7c1d"
    fake.add_completion("reply: " + secret_content)
    caplog.set_level(logging.DEBUG, logger="ai_eda")
    svc = _service(fake)
    svc.complete(TASK, [LLMMessage(role="user", content=secret_content)])
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert fake.api_key not in text
    assert secret_content not in text
    assert any(r.levelno == logging.INFO and "prompt_tokens=" in r.getMessage() for r in caplog.records)


def test_from_script_and_from_env(monkeypatch, tmp_path):
    svc = LLMService.from_script([{"content": "scripted"}], LLMBudget(max_tokens=100), gate=ApprovalGate())
    assert svc.complete(TASK, MSGS).content == "scripted"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ToolUnavailableError):
        LLMService.from_env(LLMBudget(max_usd=0.1), gate=ApprovalGate())
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-only-never-real")
    svc = LLMService.from_env(LLMBudget(max_usd=0.1), gate=ApprovalGate())
    try:
        assert isinstance(svc.client, OpenRouterClient)
        assert svc.client.reasoning == {"effort": "none"} and svc.client.app_title == "AI EDA ENGINEER"
        assert [c.model for c in svc.router.candidates(TASK)] == [DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
        assert "sk-or-v1-test-only-never-real" not in repr(svc.client)
    finally:
        svc.client.close()


def test_default_router_orders_primary_then_fallback():
    r = default_router()
    assert [c.model for c in r.candidates(TaskKind.REQUIREMENT_ANALYSIS)] == [DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
    assert all(c.temperature == 0.0 for c in r.candidates(TaskKind.CHAT))
    custom = default_router("anthropic/claude-haiku-4.5")
    assert [c.model for c in custom.candidates(TaskKind.REQUIREMENT_ANALYSIS)] == ["anthropic/claude-haiku-4.5"]  # fallback == primary is dropped
    assert [c.model for c in default_router("x/y", fallback=None).candidates(TaskKind.CHAT)] == ["x/y"]


def test_usage_records_carry_task_and_the_served_model(fake: FakeOpenRouter):
    fake.add_completion("a")
    fake.add_completion("b", usage={"prompt_tokens": 3, "completion_tokens": 4})
    svc = _service(fake)
    svc.complete(TaskKind.CHAT, MSGS)
    svc.complete(TaskKind.REVIEW, MSGS)
    assert [(r.task, r.model) for r in svc.usage.records] == [(TaskKind.CHAT, DEFAULT_PRIMARY_MODEL), (TaskKind.REVIEW, DEFAULT_PRIMARY_MODEL)]
    assert svc.usage.records[1].usage == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7, cost_usd=svc.usage.records[1].usage.cost_usd, cost_source="provider", cached_tokens=0, cache_write_tokens=0, reasoning_tokens=0, is_byok=False)
    assert svc.usage.total_cost_usd() is not None
