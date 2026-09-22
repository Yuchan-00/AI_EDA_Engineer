"""Live OpenRouter tests - the only tests that need a key and spend credits.

Every test here is skipped when ``OPENROUTER_API_KEY`` is absent (the reason
says so), so the suite is green offline. With a key they run against the
real service on ``anthropic/claude-haiku-4.5`` with small ``max_tokens`` and a
tight :class:`~ai_eda.llm.service.LLMBudget`; nothing here writes the key
anywhere.
"""

from __future__ import annotations

import os

import pytest

from ai_eda.llm.client import LLMMessage
from ai_eda.llm.extraction import RequirementExtraction, build_extraction_messages, ground_extraction, json_schema
from ai_eda.llm.openrouter import ENV_KEY, OpenRouterClient
from ai_eda.llm.router import TaskKind, default_router
from ai_eda.llm.service import LLMBudget, LLMService
from ai_eda.llm.usage import UsageTracker
from ai_eda.security import ApprovalGate

HAIKU = "anthropic/claude-haiku-4.5"
RAW = "12V 입력을 5V 2A로 변환하는 회로, 효율 90% 이상, EU에서 판매"

needs_key = pytest.mark.skipif(not os.environ.get(ENV_KEY), reason=f"{ENV_KEY} not set: live OpenRouter tests are skipped (they spend credits)")


@needs_key
def test_live_openrouter_contract():
    c = OpenRouterClient(timeout=60.0, reasoning={"effort": "none"})
    try:
        resp = c.complete(HAIKU, [LLMMessage(role="user", content="Reply with the single word: ok")], max_tokens=16)
        assert resp.model_used and resp.model_used.startswith(HAIKU)
        assert resp.usage.cost_usd is not None and resp.usage.cost_usd > 0
        assert resp.generation_id or resp.id
        pieces = list(c.stream(HAIKU, [LLMMessage(role="user", content="Reply with the single word: ok")], max_tokens=16))
        assert pieces and c.last_stream_usage is not None and c.last_stream_usage.cost_usd is not None
    finally:
        c.close()


@needs_key
def test_live_structured_extraction_grounds_the_korean_request():
    usage = UsageTracker()
    svc = LLMService.from_env(LLMBudget(max_usd=0.02), router=default_router(HAIKU, fallback=None), usage=usage, gate=ApprovalGate(), timeout=90.0)
    try:
        messages = build_extraction_messages(RAW, include_schema=False)
        extraction, resp = svc.structured(TaskKind.REQUIREMENT_ANALYSIS, messages, RequirementExtraction, max_tokens=1500, json_schema=json_schema())
    finally:
        svc.client.close()
    assert resp.model_used and resp.model_used.startswith(HAIKU)
    assert usage.records and all(r.usage.cost_usd is not None for r in usage.records)
    assert usage.total_cost_usd() is not None and 0 < usage.total_cost_usd() < 0.02
    g = ground_extraction(RAW, extraction, resp.model_used)
    by_key = {r.key: r for r in g.grounded_explicit}
    volts = [r for r in by_key.values() if r.value is not None and r.value.unit == "V"]
    assert {12.0, 5.0} <= {float(r.value.value) for r in volts}, (g.demoted, g.dropped, [r.key for r in g.requirements])
