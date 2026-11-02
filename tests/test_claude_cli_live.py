"""One tiny call through the REAL Claude Code CLI on the user's own login - opt-in only.

Skipped unless ``AI_EDA_CLAUDE_LIVE=1`` *and* a ``claude`` binary is found
(``AI_EDA_CLAUDE_CLI`` or PATH). It uses the user's subscription quota (a few
hundred tokens), so it is never on by default and CI never sets the variable;
the measurement calls recorded in ``ai_eda/llm/claude_cli.py`` are the
facts the client is built on, this test only re-checks that the contract
still holds on the installed version. Nothing here is a test of what the
model says: the reply is a proposal like every other model output.
"""

from __future__ import annotations

import os

import pytest

from ai_eda.llm.claude_cli import BILLING, COST_SOURCE, ClaudeCodeClient, find_claude_cli
from ai_eda.llm.client import LLMMessage

LIVE_ENV = "AI_EDA_CLAUDE_LIVE"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get(LIVE_ENV) != "1" or find_claude_cli() is None,
        reason=f"live Claude Code CLI call: set {LIVE_ENV}=1 with a logged-in `claude` on PATH (uses the subscription quota)",
    ),
    # the only test that may discover the real CLI (tests/conftest.py hides everything but the fake otherwise)
    pytest.mark.real_claude_cli,
]


def test_one_tiny_call_on_the_subscription_route():
    client = ClaudeCodeClient(timeout=120.0)
    try:
        state = client.login_state()
        assert state.error is None and state.logged_in is not False, state
        resp = client.complete(
            "claude-sonnet-5",
            [LLMMessage(role="system", content="You answer tersely."), LLMMessage(role="user", content="Reply with the single word OK.")],
        )
    finally:
        client.close()
    assert resp.content is not None and "OK" in resp.content
    assert resp.finish_reason == "stop" and resp.native_finish_reason == "success" and resp.id
    assert resp.model_used and resp.usage.total_tokens > 0
    # a known zero charge with the CLI's API-equivalent estimate beside it - never a budgeted number
    assert resp.usage.cost_usd == 0.0 and resp.usage.cost_source == COST_SOURCE and resp.usage.billing == BILLING
    assert resp.usage.estimated_cost_usd is not None and resp.usage.estimated_cost_usd >= 0.0
    assert resp.raw is not None and resp.raw.get("type") == "result" and resp.raw.get("is_error") is False
