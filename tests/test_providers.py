"""The provider layer: ``provider:model`` specs, ``ProviderClient`` routing, and the service's budget / approval rules per billing kind.

No key, no network, no CLI: the members are ``ScriptedLLMClient``s. The
``claude`` double is a scripted client flagged ``billing="subscription"``
(``paid=False``) with a ``cli`` path, the ``openrouter`` double one flagged
``paid=True``; the real clients are never constructed here.
"""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from ai_eda.errors import ApprovalRequiredError, ToolUnavailableError
from ai_eda.llm.client import LLMError, LLMMessage, ToolSpec, Usage
from ai_eda.llm.fake import ScriptedLLMClient
from ai_eda.llm.providers import (
    OPENROUTER_ENV_KEY,
    ProviderClient,
    ProviderInfo,
    build_client,
    check_providers,
    describe_providers,
    members_of,
    provider_names_of,
    subscription_detail,
)
from ai_eda.llm.router import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    KNOWN_PROVIDERS,
    ModelConfig,
    ModelRouter,
    TaskKind,
    default_model,
    default_router,
    parse_model_spec,
    same_model_fallback,
    same_model_on,
    split_model_spec,
    translate_model,
)
from ai_eda.llm.service import SUBSCRIPTION_APPROVED_BY, BudgetExceededError, LLMBudget, LLMService, check_router_providers
from ai_eda.llm.usage import UsageTracker, is_charge
from ai_eda.security import ApprovalGate, ExternalAction

TASK = TaskKind.REQUIREMENT_ANALYSIS
MSGS = [LLMMessage(role="system", content="answer tersely"), LLMMessage(role="user", content="say hi")]
ANTHROPIC_MESSAGE = "direct Messages-API access is not implemented; use claude: (Claude Code CLI) or openrouter:"
FAKE_CLI = "/fake/bin/claude"

#: what the CLI double reports for one call: a known zero charge plus the CLI's API-equivalent estimate
SUB_USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.0, "cost_source": "subscription", "billing": "subscription", "estimated_cost_usd": 0.0123}
#: what the OpenRouter double reports: a provider charge
PAID_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.002, "cost_source": "provider", "billing": "per_call"}


def _sub(items: list[Any] | None = None, **kw: Any) -> ScriptedLLMClient:
    """The claude double: subscription-billed, free, answers ``items`` (default: one served reply with SUB_USAGE)."""
    c = ScriptedLLMClient(items if items is not None else [{"content": "from claude", "usage": SUB_USAGE}], **kw)
    c.billing = "subscription"
    c.cli = FAKE_CLI
    return c


def _paid(items: list[Any] | None = None, **kw: Any) -> ScriptedLLMClient:
    """The openrouter double: per-call billed and paid."""
    c = ScriptedLLMClient(items if items is not None else [{"content": "from openrouter", "usage": PAID_USAGE}], **kw)
    c.paid = True
    return c


def _mixed(paid: ScriptedLLMClient | None = None, sub: ScriptedLLMClient | None = None, default: str = "openrouter") -> ProviderClient:
    return ProviderClient({"openrouter": paid or _paid(), "claude": sub or _sub()}, default=default)


def _service(client, router: ModelRouter | None = None, budget: LLMBudget | None = None, **kw: Any) -> LLMService:
    kw.setdefault("gate", ApprovalGate())
    return LLMService(client, router or default_router(), UsageTracker(), budget if budget is not None else LLMBudget(max_usd=1.0), **kw)


# --------------------------------------------------------------------------- specs


def test_parse_model_spec_prefixed_and_capabilities():
    cfg = parse_model_spec("openrouter:anthropic/claude-sonnet-5")
    assert (cfg.provider, cfg.model, cfg.spec) == ("openrouter", "anthropic/claude-sonnet-5", "openrouter:anthropic/claude-sonnet-5")
    assert cfg.supports_tools and cfg.supports_structured and cfg.temperature == 0.0
    cli = parse_model_spec("claude:claude-sonnet-5")
    assert (cli.provider, cli.model) == ("claude", "claude-sonnet-5")
    assert cli.supports_tools is False and cli.supports_structured is True  # the CLI route: no caller tools, --json-schema
    assert parse_model_spec("claude:sonnet").model == "sonnet"  # CLI aliases pass through
    assert parse_model_spec("CLAUDE:opus").provider == "claude"  # the prefix is case-insensitive
    assert parse_model_spec("script:anything").provider == "script"


def test_parse_model_spec_unprefixed_means_the_default_provider():
    bare = parse_model_spec(DEFAULT_PRIMARY_MODEL)
    assert bare.provider is None and bare.spec == DEFAULT_PRIMARY_MODEL and bare.supports_tools
    on_claude = parse_model_spec("sonnet", default_provider="claude")
    assert on_claude.spec == "claude:sonnet" and on_claude.supports_tools is False
    on_or = parse_model_spec(DEFAULT_PRIMARY_MODEL, default_provider="openrouter")
    assert on_or.spec == f"openrouter:{DEFAULT_PRIMARY_MODEL}" and on_or.supports_tools
    assert parse_model_spec("claude:sonnet", default_provider="openrouter").provider == "claude"  # an explicit prefix wins
    with pytest.raises(ValueError, match="unknown provider 'foo'"):
        parse_model_spec("x", default_provider="foo")
    with pytest.raises(ValueError, match="direct Messages-API"):
        parse_model_spec("x", default_provider="anthropic")


def test_parse_model_spec_keeps_a_colon_inside_a_native_id():
    assert split_model_spec("foo:bar") == (None, "foo:bar")  # an unknown prefix is part of the id, not a provider
    assert split_model_spec("openrouter:some/model:free") == ("openrouter", "some/model:free")  # split on the first colon only
    assert parse_model_spec("foo:bar", default_provider="openrouter").spec == "openrouter:foo:bar"
    assert split_model_spec("  claude:sonnet ") == ("claude", "sonnet")


def test_anthropic_prefix_is_refused_with_the_two_routes_named():
    with pytest.raises(ValueError, match="direct Messages-API access is not implemented; use claude: \\(Claude Code CLI\\) or openrouter:") as ei:
        parse_model_spec("anthropic:claude-sonnet-5")
    assert ANTHROPIC_MESSAGE in str(ei.value)
    for bad, msg in (("", "empty"), ("   ", "empty"), ("claude:", "no model"), ("openrouter: ", "no model")):
        with pytest.raises(ValueError, match=msg):
            parse_model_spec(bad)


def test_model_config_spec_property():
    assert ModelConfig(model="x").spec == "x"
    assert ModelConfig(model="x", provider="claude").spec == "claude:x"
    assert ModelConfig(model="a/b:c", provider="openrouter").spec == "openrouter:a/b:c"
    assert parse_model_spec(ModelConfig(model="a/b:c", provider="openrouter").spec).model == "a/b:c"  # round trip


def test_known_providers_and_defaults():
    assert KNOWN_PROVIDERS == ("openrouter", "claude", "script")
    assert DEFAULT_CLAUDE_MODEL == "claude-sonnet-5"
    assert default_model(None) == default_model("openrouter") == default_model("script") == DEFAULT_PRIMARY_MODEL
    assert default_model("claude") == DEFAULT_CLAUDE_MODEL
    assert check_providers(("openrouter", "claude")) == ("openrouter", "claude")
    with pytest.raises(ValueError, match="configured twice"):
        check_providers(("claude", "claude"))
    with pytest.raises(ValueError, match="no provider"):
        check_providers(())


# --------------------------------------------------------------------------- router


def test_candidates_dedup_by_spec_and_keep_order():
    r = default_router("openrouter:m", default_provider="openrouter", fallback=["m", "claude:m2", "claude:m2", "openrouter:m"])
    assert [c.spec for c in r.candidates(TASK)] == ["openrouter:m", "claude:m2"]
    # a hand-built router with the same native id under two providers keeps both: they are different routes
    hand = ModelRouter(default=ModelConfig(model="m", provider="openrouter"), fallbacks=[ModelConfig(model="m", provider="claude"), ModelConfig(model="m", provider="openrouter")])
    assert [c.spec for c in hand.candidates(TASK)] == ["openrouter:m", "claude:m"]
    assert hand.providers() == ["openrouter", "claude"]


def test_no_default_fallback_for_any_provider():
    assert default_router().fallbacks == [] and [c.spec for c in default_router().candidates(TASK)] == [DEFAULT_PRIMARY_MODEL]
    assert default_router(default_provider="openrouter").fallbacks == []
    on_claude = default_router(default_provider="claude")
    assert on_claude.fallbacks == [] and on_claude.default.spec == f"claude:{DEFAULT_CLAUDE_MODEL}" and on_claude.default.supports_tools is False
    assert default_router("claude:opus").default.spec == "claude:opus" and default_router("claude:opus").fallbacks == []
    explicit = default_router(fallback=DEFAULT_FALLBACK_MODEL)
    assert [c.spec for c in explicit.candidates(TASK)] == [DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL]
    assert [c.spec for c in default_router("claude:sonnet", fallback=("openrouter:a/b", "claude:opus")).candidates(TASK)] == ["claude:sonnet", "openrouter:a/b", "claude:opus"]


def test_same_fallback_is_the_primary_model_on_the_other_provider():
    primary = parse_model_spec("openrouter:anthropic/claude-sonnet-5")
    same = same_model_fallback(primary, ("openrouter", "claude"))
    assert same.spec == "claude:claude-sonnet-5" and same.supports_tools is False
    back = same_model_fallback(parse_model_spec("claude:claude-sonnet-5"), ("claude", "openrouter"))
    assert back.spec == "openrouter:anthropic/claude-sonnet-5" and back.supports_tools
    # an unprefixed primary belongs to the first configured provider
    assert same_model_fallback(parse_model_spec("anthropic/claude-sonnet-5"), ("openrouter", "claude")).spec == "claude:claude-sonnet-5"
    assert same_model_fallback(parse_model_spec("claude-sonnet-5"), ("claude", "openrouter")).spec == "openrouter:anthropic/claude-sonnet-5"
    # temperature and cap travel with the model
    warm = parse_model_spec("openrouter:anthropic/claude-sonnet-5", temperature=0.3, max_tokens=99)
    assert same_model_on(warm, "claude").model_dump() == {"model": "claude-sonnet-5", "provider": "claude", "temperature": 0.3, "max_tokens": 99, "supports_tools": False, "supports_structured": True}
    router = default_router(primary.spec, default_provider="openrouter", fallback=same.spec)
    assert [c.spec for c in router.candidates(TASK)] == ["openrouter:anthropic/claude-sonnet-5", "claude:claude-sonnet-5"]


def test_same_fallback_refusals_and_verbatim_script_ids():
    with pytest.raises(ValueError, match="second configured provider"):
        same_model_fallback(parse_model_spec("claude:claude-sonnet-5"), ("claude",))
    with pytest.raises(ValueError, match="needs a configured provider"):
        same_model_fallback(parse_model_spec("x"), ())
    with pytest.raises(ValueError, match="alias, not a full model name"):
        same_model_fallback(parse_model_spec("claude:sonnet"), ("claude", "openrouter"))
    with pytest.raises(ValueError, match="not an Anthropic model on OpenRouter"):
        same_model_fallback(parse_model_spec("openrouter:openai/gpt-x"), ("openrouter", "claude"))
    assert translate_model("whatever:x", "script", "claude") == "whatever:x"
    assert translate_model("claude-sonnet-5", "claude", "script") == "claude-sonnet-5"
    assert translate_model("a/b", "openrouter", "openrouter") == "a/b"
    assert same_model_fallback(parse_model_spec("script:m"), ("script", "claude")).spec == "claude:m"


@pytest.mark.parametrize(
    "model, source, target",
    [
        (DEFAULT_FALLBACK_MODEL, "openrouter", "claude"),  # anthropic/claude-haiku-4.5: the CLI spells it claude-haiku-4-5
        ("anthropic/claude-haiku-4-5", "openrouter", "claude"),
        ("claude-haiku-4-5", "claude", "openrouter"),  # OpenRouter spells it anthropic/claude-haiku-4.5
        ("claude-haiku-4-5-20251001", "claude", "openrouter"),  # a dated CLI id
        ("claude-haiku-4.5", "claude", "openrouter"),
    ],
)
def test_same_fallback_refuses_an_id_whose_spelling_differs_between_the_routes(model: str, source: str, target: str):
    with pytest.raises(ValueError, match="only a claude-<family>-<major> id is known to be spelled the same.*name the fallback explicitly"):
        translate_model(model, source, target)
    with pytest.raises(ValueError, match="name the fallback explicitly"):
        same_model_fallback(parse_model_spec(f"{source}:{model}"), (source, target))
    # the measured spelling still translates both ways
    assert translate_model("anthropic/claude-sonnet-5", "openrouter", "claude") == "claude-sonnet-5"
    assert translate_model("claude-sonnet-5", "claude", "openrouter") == "anthropic/claude-sonnet-5"


def test_task_model_override_in_default_router():
    r = default_router("openrouter:a/b", default_provider="openrouter", by_task={"review": "claude:sonnet", TaskKind.CHAT: "c/d"})
    assert r.for_task(TaskKind.REVIEW).spec == "claude:sonnet" and r.for_task(TaskKind.CHAT).spec == "openrouter:c/d"
    assert r.for_task(TASK).spec == "openrouter:a/b" and r.for_task(TaskKind.REPAIR_PLANNING).spec == "openrouter:a/b"
    with pytest.raises(ValueError, match="unknown task 'nope': use one of requirement_analysis, component_proposal"):
        default_router(by_task={"nope": "x"})


# --------------------------------------------------------------------------- ProviderClient


def test_provider_client_routes_with_the_native_id_and_stamps_via_and_the_served_spec():
    paid, sub = _paid([{"content": "or", "usage": PAID_USAGE}]), _sub([{"content": "cl", "usage": SUB_USAGE, "model": "claude-sonnet-5-20260101"}])
    pc = _mixed(paid, sub)
    assert pc.providers == ["openrouter", "claude"] and pc.default == "openrouter" and pc.member("claude") is sub
    r1 = pc.complete("claude:sonnet", MSGS)
    assert sub.calls[0].model == "sonnet" and paid.calls == []  # the member sees the native id
    assert r1.via == "claude" and r1.model == "claude:sonnet" and r1.model_used == "claude:claude-sonnet-5-20260101"  # the served id, prefixed
    assert r1.usage.cost_usd == 0.0 and r1.usage.cost_source == "subscription" and r1.usage.estimated_cost_usd == pytest.approx(0.0123)
    r2 = pc.complete("anthropic/x", MSGS, response_schema={"type": "object"}, temperature=0.5, max_tokens=33)
    assert paid.calls[0].model == "anthropic/x" and paid.calls[0].response_schema == {"type": "object"} and paid.calls[0].temperature == 0.5 and paid.calls[0].max_tokens == 33
    assert r2.via == "openrouter" and r2.model == "openrouter:anthropic/x" and r2.model_used == "openrouter:anthropic/x"
    assert "openrouter" in repr(pc) and "claude" in repr(pc)


def test_provider_client_paid_and_billing_follow_the_members():
    assert (_mixed().paid, _mixed().billing) == (True, "mixed")
    subs = ProviderClient({"claude": _sub()}, default="claude")
    assert (subs.paid, subs.billing) == (False, "subscription")
    both_paid = ProviderClient({"openrouter": _paid(), "script": ScriptedLLMClient()}, default="script")
    assert (both_paid.paid, both_paid.billing) == (True, "per_call")
    free_per_call = ProviderClient({"script": ScriptedLLMClient()}, default="script")
    assert (free_per_call.paid, free_per_call.billing) == (False, "per_call")


def test_provider_client_stream_delegates_and_last_stream_fields_are_stamped():
    sub = _sub([{"content": "one two three", "usage": SUB_USAGE}])
    pc = _mixed(sub=sub)
    assert "".join(pc.stream("claude:sonnet", MSGS, temperature=0.0, max_tokens=50)) == "one two three"
    assert sub.calls[-1].stream and sub.calls[-1].model == "sonnet" and sub.calls[-1].max_tokens == 50
    assert pc.last_stream_usage == sub.last_stream_usage and pc.last_stream_usage is not None and pc.last_stream_usage.cost_usd == 0.0
    final = pc.last_stream_response
    assert final is not None and final.via == "claude" and final.model_used == "claude:sonnet" and final.content == "one two three"
    fresh = _mixed()
    assert fresh.last_stream_usage is None and fresh.last_stream_response is None


def test_provider_client_unknown_or_unconfigured_prefix_raises_before_anything_is_sent():
    paid, sub = _paid(), _sub()
    pc = _mixed(paid, sub)
    with pytest.raises(ValueError, match="provider 'script' is not configured \\(configured: openrouter, claude\\)"):
        pc.complete("script:x", MSGS)
    with pytest.raises(ValueError, match=ANTHROPIC_MESSAGE.replace("(", "\\(").replace(")", "\\)")):
        pc.complete("anthropic:claude-sonnet-5", MSGS)
    with pytest.raises(ValueError, match="not configured"):
        pc.stream("script:x", MSGS)  # resolved at the call, not at the first piece
    with pytest.raises(ValueError, match="unknown provider 'foo'"):
        pc.member("foo")
    assert paid.calls == [] and sub.calls == []
    with pytest.raises(ValueError, match="default provider 'claude' is not a member"):
        ProviderClient({"openrouter": paid}, default="claude")
    with pytest.raises(ValueError, match="at least one member"):
        ProviderClient({}, default="openrouter")


def test_provider_client_errors_carry_via_and_the_prefixed_model():
    sub = _sub([{"error": {"message": "boom", "status": 500}}, {"error": {"message": "committed", "kind": "response", "status": 200, "code": 502, "sent": True}}])
    pc = _mixed(sub=sub)
    with pytest.raises(LLMError) as ei:
        pc.complete("claude:sonnet", MSGS)
    assert ei.value.via == "claude" and ei.value.model == "claude:sonnet" and ei.value.status == 500
    with pytest.raises(LLMError) as ei2:
        pc.complete("claude:sonnet", MSGS)
    assert ei2.value.via == "claude" and ei2.value.maybe_billed
    with pytest.raises(LLMError) as ei3:
        list(pc.stream("claude:sonnet", MSGS))  # the script is exhausted: a scripted-client error, stamped the same way
    assert ei3.value.kind == "script" and ei3.value.via == "claude" and ei3.value.model == "claude:sonnet"
    err = pickle.loads(pickle.dumps(LLMError("x", kind="http", status=503, via="openrouter")))
    assert err.via == "openrouter" and err.status == 503


def test_provider_client_close_closes_every_member():
    class Closable(ScriptedLLMClient):
        closed = False

        def close(self) -> None:
            self.closed = True

    a, b = Closable(), Closable()
    pc = ProviderClient({"openrouter": a, "claude": b}, default="openrouter")
    pc.close()
    assert a.closed and b.closed
    assert provider_names_of(pc) == ["openrouter", "claude"] and provider_names_of(a) == ["script"]  # a subclass of the scripted client
    assert members_of(pc) == [("openrouter", a), ("claude", b)] and members_of(a) == [("script", a)] and members_of(a, ["claude"]) == [("claude", a)]


# --------------------------------------------------------------------------- build_client / from_env


class FakeOpenRouterClient(ScriptedLLMClient):
    """Stands in for the HTTP client: records its constructor options."""

    paid = True
    instances: list["FakeOpenRouterClient"] = []

    def __init__(self, api_key: str | None = None, *, reasoning: dict | None = None, app_title: str | None = None, timeout: float = 120.0) -> None:
        super().__init__([{"content": "from openrouter", "usage": PAID_USAGE}], repeat_last=True)
        self.options = {"api_key": api_key, "reasoning": reasoning, "app_title": app_title, "timeout": timeout}
        self.closed = False
        FakeOpenRouterClient.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeClaudeCodeClient(ScriptedLLMClient):
    """Stands in for the CLI client: subscription billing, a ``cli`` path, ``version`` / ``login_state`` for ``doctor``."""

    paid = False
    billing = "subscription"
    instances: list["FakeClaudeCodeClient"] = []

    def __init__(self, cli: str | None = None, *, max_budget_usd: float | None = None, timeout: float = 600.0, missing: bool = False) -> None:
        if missing:
            raise ToolUnavailableError("claude not found (install Claude Code and run `claude login`, or set AI_EDA_CLAUDE_CLI)")
        super().__init__([{"content": "from claude", "usage": SUB_USAGE}], repeat_last=True)
        self.cli = cli or FAKE_CLI
        self.options = {"cli": cli, "max_budget_usd": max_budget_usd, "timeout": timeout}
        FakeClaudeCodeClient.instances.append(self)

    def version(self) -> str:
        return "2.1.283"

    def login_state(self) -> dict[str, Any]:
        return {"loggedIn": True, "authMethod": "oauth_token"}


FACTORIES = {"openrouter": FakeOpenRouterClient, "claude": FakeClaudeCodeClient}


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeOpenRouterClient.instances.clear()
    FakeClaudeCodeClient.instances.clear()
    yield
    FakeOpenRouterClient.instances.clear()
    FakeClaudeCodeClient.instances.clear()


def test_build_client_bare_for_one_provider_and_routing_for_several():
    bare = build_client(("claude",), factories=FACTORIES, cli="/x/claude")
    assert isinstance(bare, FakeClaudeCodeClient) and bare.options["cli"] == "/x/claude"
    both = build_client(("claude", "openrouter"), factories=FACTORIES, timeout=7.0, reasoning={"effort": "none"}, cli="/y/claude")
    assert isinstance(both, ProviderClient) and both.providers == ["claude", "openrouter"] and both.default == "claude"
    # an option is routed by each constructor's signature: shared names reach both, the others only where accepted
    assert both.member("openrouter").options == {"api_key": None, "reasoning": {"effort": "none"}, "app_title": None, "timeout": 7.0}
    assert both.member("claude").options == {"cli": "/y/claude", "max_budget_usd": None, "timeout": 7.0}
    with pytest.raises(TypeError, match="client option\\(s\\) bogus are not accepted by any configured provider \\(claude, openrouter\\)"):
        build_client(("claude", "openrouter"), factories=FACTORIES, bogus=1)
    assert FakeOpenRouterClient.instances == [both.member("openrouter")]  # refused before any client is built
    # soft options apply only where accepted
    soft = build_client(("openrouter", "claude"), factories=FACTORIES, soft_kw={"max_budget_usd": 0.5, "nothing": 1})
    assert soft.member("claude").options["max_budget_usd"] == 0.5
    # a member that fails to build closes the ones built before it
    with pytest.raises(ToolUnavailableError, match="claude not found"):
        build_client(("openrouter", "claude"), factories=FACTORIES, missing=True)
    assert FakeOpenRouterClient.instances[-1].closed
    with pytest.raises(ValueError, match="unknown provider 'foo'"):
        build_client(("foo",), factories=FACTORIES)


def test_from_env_single_subscription_provider_needs_no_budget_and_records_subscription_use():
    gate = ApprovalGate()
    svc = LLMService.from_env(LLMBudget(), providers=("claude",), factories=FACTORIES, gate=gate, approved_by="cli --llm-budget-usd/--llm-budget-tokens")
    assert isinstance(svc.client, FakeClaudeCodeClient) and svc.providers == ["claude"] and svc.budget_required is False
    detail = f"claude subscription via {FAKE_CLI} (no per-call cost; usage shown)"
    assert [(e["event"], e["action"], e["detail"]) for e in gate.audit] == [("grant", ExternalAction.SUBSCRIPTION_USE, detail), ("consume", ExternalAction.SUBSCRIPTION_USE, detail)]
    assert gate.audit[0]["by"] == SUBSCRIPTION_APPROVED_BY == "cli --llm claude"
    assert svc.subscription_details == [detail] and svc.approval_detail == "claude budget none"
    assert svc.router.default.spec == f"claude:{DEFAULT_CLAUDE_MODEL}" and svc.router.fallbacks == []
    assert svc.client.options["max_budget_usd"] is None  # no USD budget given: no --max-budget-usd passthrough
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "from claude" and svc.client.calls[0].model == DEFAULT_CLAUDE_MODEL  # the bare client gets the native id
    [rec] = svc.usage.records
    assert rec.model == DEFAULT_CLAUDE_MODEL and rec.via == "claude" and rec.usage.cost_usd == 0.0 and not rec.charge
    assert svc.attempts[-1].via == "claude" and svc.attempts[-1].model_used == DEFAULT_CLAUDE_MODEL


def test_from_env_mixed_providers_require_a_budget_and_record_both_approvals():
    gate = ApprovalGate()
    with pytest.raises(ApprovalRequiredError, match="paid_api_call"):
        LLMService.from_env(LLMBudget(), providers=("openrouter", "claude"), factories=FACTORIES, gate=gate)
    assert gate.audit == [{"event": "denied", "action": ExternalAction.PAID_API_CALL, "detail": "openrouter+claude budget none"}]
    gate = ApprovalGate()
    svc = LLMService.from_env(LLMBudget(max_usd=0.25), providers=("openrouter", "claude"), factories=FACTORIES, gate=gate, approved_by="cli --llm-budget-usd")
    assert isinstance(svc.client, ProviderClient) and svc.providers == ["openrouter", "claude"] and svc.budget_required
    assert svc.approval_detail == "openrouter+claude budget max_usd=0.25"
    events = [(e["event"], e["action"]) for e in gate.audit]
    assert events == [("grant", ExternalAction.PAID_API_CALL), ("consume", ExternalAction.PAID_API_CALL), ("grant", ExternalAction.SUBSCRIPTION_USE), ("consume", ExternalAction.SUBSCRIPTION_USE)]
    assert gate.audit[0]["by"] == "cli --llm-budget-usd" and gate.audit[2]["by"] == SUBSCRIPTION_APPROVED_BY
    or_client, cl_client = svc.client.member("openrouter"), svc.client.member("claude")
    assert or_client.options["reasoning"] == {"effort": "none"} and or_client.options["app_title"] == "AI EDA ENGINEER"
    assert cl_client.options["max_budget_usd"] == 0.25  # the USD budget is the CLI's second safety net
    assert svc.router.default.spec == f"openrouter:{DEFAULT_PRIMARY_MODEL}"


def test_from_env_refuses_a_router_naming_an_unconfigured_provider_before_building_anything():
    with pytest.raises(ValueError, match="model spec\\(s\\) claude:opus name a provider that is not configured \\(--llm openrouter\\)"):
        LLMService.from_env(LLMBudget(max_usd=1.0), router=default_router("claude:opus"), providers=("openrouter",), factories=FACTORIES, gate=ApprovalGate())
    with pytest.raises(ValueError, match="openrouter:a/b"):
        LLMService.from_env(LLMBudget(), router=default_router("claude:opus", fallback="openrouter:a/b"), providers=("claude",), factories=FACTORIES, gate=ApprovalGate())
    assert FakeOpenRouterClient.instances == [] and FakeClaudeCodeClient.instances == []
    with pytest.raises(ValueError, match="configured twice"):
        LLMService.from_env(LLMBudget(), providers=("claude", "claude"), factories=FACTORIES, gate=ApprovalGate())
    check_router_providers(default_router("x"), ("openrouter",))  # an unprefixed spec belongs to the default provider: fine
    check_router_providers(default_router("openrouter:x", by_task={"chat": "claude:y"}), ("openrouter", "claude"))


# --------------------------------------------------------------------------- LLMService with scripted members


def test_subscription_only_service_records_subscription_use_and_needs_no_budget():
    gate = ApprovalGate()
    sub = _sub(repeat_last=True)
    svc = _service(sub, default_router(default_provider="claude"), LLMBudget(), gate=gate, providers=("claude",))
    detail = f"claude subscription via {FAKE_CLI} (no per-call cost; usage shown)"
    assert [(e["event"], e["action"], e["detail"], e.get("by")) for e in gate.audit] == [
        ("grant", ExternalAction.SUBSCRIPTION_USE, detail, "cli --llm claude"), ("consume", ExternalAction.SUBSCRIPTION_USE, detail, None),
    ]
    assert svc.budget_required is False and svc.per_call_providers == [] and svc.subscription_providers == ["claude"]
    svc.check_budget()  # no budget, nothing to refuse
    for _ in range(3):
        assert svc.complete(TASK, MSGS).content == "from claude"
    assert len(sub.calls) == 3 and all(r.via == "claude" and r.usage.cost_usd == 0.0 for r in svc.usage.records)
    assert svc.summary() == (
        "3 served call(s), 450 tokens, cost 0.000000 USD; budget none (not required: subscription only); "
        "3 call(s) on the subscription: 450 tokens, estimated API-equivalent cost 0.036900 USD (not charged)"
    )
    assert svc.usage.total_cost_usd() == 0.0 and svc.usage.total_estimated_cost_usd() == pytest.approx(0.0369)
    # a bare scripted client without the claude label is labelled by its class
    plain = _service(_sub(), default_router(), LLMBudget(), gate=ApprovalGate())
    assert plain.providers == ["script"] and plain.subscription_details == [f"script subscription via {FAKE_CLI} (no per-call cost; usage shown)"]
    assert subscription_detail("claude", ScriptedLLMClient()) == "claude subscription via claude (no per-call cost; usage shown)"


def test_subscription_only_service_still_honours_a_token_budget():
    sub = _sub(repeat_last=True)
    svc = _service(sub, default_router(default_provider="claude"), LLMBudget(max_tokens=200), gate=ApprovalGate(), providers=("claude",))
    assert svc.budget_required is False and svc.approval_detail == "claude budget max_tokens=200"
    svc.complete(TASK, MSGS)  # 150 tokens
    assert sub.calls[0].max_tokens == 200  # the cap is what is left
    svc.complete(TASK, MSGS)  # allowed at 150 < 200; brings it to 300
    with pytest.raises(BudgetExceededError, match="spent 300 tokens, budget 200 tokens"):
        svc.complete(TASK, MSGS)
    assert len(sub.calls) == 2 and svc.attempts[-1].outcome == "budget" and svc.attempts[-1].via == "claude"
    assert "budget max_tokens=200;" in svc.summary() and "(not required" not in svc.summary()


def test_openrouter_in_the_mix_requires_a_budget():
    gate = ApprovalGate()
    with pytest.raises(ApprovalRequiredError, match="paid_api_call"):
        _service(_mixed(), default_router(default_provider="openrouter"), LLMBudget(), gate=gate)
    assert gate.audit == [{"event": "denied", "action": ExternalAction.PAID_API_CALL, "detail": "openrouter+claude budget none"}]  # no subscription grant either
    gate = ApprovalGate()
    svc = _service(_mixed(default="claude"), default_router(default_provider="claude"), LLMBudget(max_tokens=10), gate=gate)
    assert svc.budget_required and svc.per_call_providers == ["openrouter"] and svc.subscription_providers == ["claude"]
    assert svc.approval_detail == "openrouter+claude budget max_tokens=10"
    assert [(e["event"], e["action"]) for e in gate.audit] == [("grant", ExternalAction.PAID_API_CALL), ("consume", ExternalAction.PAID_API_CALL), ("grant", ExternalAction.SUBSCRIPTION_USE), ("consume", ExternalAction.SUBSCRIPTION_USE)]
    # a service whose budget was revoked by hand refuses with the per-call provider named
    svc.budget = LLMBudget()
    with pytest.raises(BudgetExceededError, match="no LLM budget granted for the per-call provider\\(s\\) openrouter"):
        svc.complete(TASK, MSGS)


def test_usd_budget_counts_charges_only_while_tokens_count_everything():
    paid, sub = _paid(repeat_last=True), _sub(repeat_last=True)
    pc = ProviderClient({"openrouter": paid, "claude": sub}, default="openrouter")
    router = default_router("openrouter:a/b", default_provider="openrouter", by_task={"chat": "claude:sonnet"})
    svc = _service(pc, router, LLMBudget(max_usd=0.002, max_tokens=500))
    assert svc.complete(TASK, MSGS).via == "openrouter"  # charge 0.002 (15 tokens)
    with pytest.raises(BudgetExceededError, match="spent 0.002000 USD, budget 0.002 USD: nothing left"):
        svc.complete(TASK, MSGS)  # the USD budget is reached for a per-call route
    for _ in range(3):
        assert svc.complete(TaskKind.CHAT, MSGS).via == "claude"  # a subscription route is not a charge: still allowed (150 tokens each)
    assert svc.spent() == (pytest.approx(0.002), 0, 465)  # charges: the one paid call; tokens: everything
    assert svc.is_per_call(router.for_task(TASK)) and not svc.is_per_call(router.for_task(TaskKind.CHAT))
    svc.complete(TaskKind.CHAT, MSGS)  # 465 < 500 allows one more (brings it to 615)
    with pytest.raises(BudgetExceededError, match="spent 615 tokens, budget 500 tokens"):
        svc.complete(TaskKind.CHAT, MSGS)  # the token budget binds the subscription route too
    assert len(sub.calls) == 4 and len(paid.calls) == 1
    assert svc.usage.total_cost_usd() == pytest.approx(0.002) and all(is_charge(r.usage) == (r.via == "openrouter") for r in svc.usage.records)
    # the openrouter side still sees the USD refusal; the note names its route
    with pytest.raises(BudgetExceededError, match="USD"):
        svc.complete(TASK, MSGS)
    assert svc.attempts[-1].outcome == "budget" and svc.attempts[-1].via == "openrouter" and svc.attempts[-1].model == "openrouter:a/b"


def test_a_subscription_failure_without_usage_is_not_an_unknown_cost_charge():
    """A failed call on the subscription route (a CLI timeout: sent, no usage) is a known zero charge: a USD-only budget
    still covers the per-call side, and the summary does not call it possibly billed."""
    timeout = {"error": {"message": "timeout after 600s", "kind": "transport", "sent": True}}
    unparsed = {"error": {"message": "stdout is not a JSON envelope", "kind": "response", "status": None}}
    sub = _sub([timeout, unparsed, {"content": "late", "usage": {"prompt_tokens": 3, "completion_tokens": 1}}])
    paid = _paid(repeat_last=True)
    router = default_router("claude:sonnet", default_provider="claude", fallback="openrouter:a/b", by_task={"review": "openrouter:a/b"})
    svc = _service(ProviderClient({"claude": sub, "openrouter": paid}, default="claude"), router, LLMBudget(max_usd=1.0))
    assert svc.complete(TASK, MSGS).via == "openrouter"  # the timeout falls back to the explicitly configured candidate
    with pytest.raises(LLMError):
        svc.complete(TASK, MSGS)  # a response error without status: no fallback, no charge
    assert svc.complete(TaskKind.REVIEW, MSGS).via == "openrouter"  # later per-call requests are still covered
    resp = svc.complete(TASK, MSGS)  # served without billing fields: the subscription route still makes it a known zero
    assert resp.via == "claude"
    subs = [r for r in svc.usage.records if r.via == "claude"]
    assert [(r.outcome, r.charge, r.usage.cost_usd, r.usage.billing) for r in subs] == [("failed", False, 0.0, "subscription")] * 2 + [("served", False, 0.0, "subscription")]
    assert svc.spent() == (pytest.approx(0.004), 0, 34)
    assert "2 failed call(s) on the subscription" in svc.summary() and "possibly billed" not in svc.summary()
    # a per-call failure is still possibly billed, with its cost unknown
    paid2 = _paid([{"error": {"message": "cut", "kind": "transport", "sent": True}}])
    svc2 = _service(paid2, default_router("a/b"), LLMBudget(max_usd=1.0, max_tokens=100), providers=("openrouter",))
    with pytest.raises(LLMError):
        svc2.complete(TASK, MSGS)
    assert svc2.usage.records[0].charge and svc2.usage.records[0].usage.cost_usd is None
    assert "1 failed call(s) possibly billed" in svc2.summary() and "on the subscription" not in svc2.summary()


def test_summary_wording_with_one_and_with_several_providers():
    paid, sub = _paid(repeat_last=True), _sub(repeat_last=True)
    pc = ProviderClient({"openrouter": paid, "claude": sub}, default="openrouter")
    router = default_router("openrouter:a/b", default_provider="openrouter", by_task={"chat": "claude:sonnet"})
    svc = _service(pc, router, LLMBudget(max_usd=1.0))
    svc.complete(TASK, MSGS)
    assert svc.summary() == "1 served call(s), 15 tokens, cost 0.002000 USD; budget max_usd=1"  # one provider recorded: no breakdown
    svc.complete(TaskKind.CHAT, MSGS)
    svc.complete(TaskKind.CHAT, MSGS)
    assert svc.summary() == (
        "3 served call(s), 315 tokens, cost 0.002000 USD; budget max_usd=1; "
        "2 call(s) on the subscription: 300 tokens, estimated API-equivalent cost 0.024600 USD (not charged); "
        "by provider: openrouter 1 call(s) 15 tokens 0.002000 USD, claude 2 call(s) 300 tokens subscription (estimated 0.024600 USD, not charged)"
    )
    # a subscription record without an estimate: the sentence says unknown instead of inventing a number
    sub.add({"content": "x", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "cost_source": "subscription", "billing": "subscription"}})
    sub.repeat_last = False
    svc.complete(TaskKind.CHAT, MSGS)
    assert "estimated API-equivalent cost unknown (not charged)" in svc.summary() and "claude 3 call(s) 302 tokens subscription (estimated unknown, not charged)" in svc.summary()


def test_approval_detail_labels():
    assert _service(_paid(), providers=("openrouter",)).approval_detail == "openrouter budget max_usd=1"
    assert _service(_sub(), default_router(default_provider="claude"), LLMBudget(), providers=("claude",)).approval_detail == "claude budget none"
    assert _service(_mixed(), default_router(default_provider="openrouter")).approval_detail == "openrouter+claude budget max_usd=1"
    assert _service(ProviderClient({"claude": _sub(), "openrouter": _paid()}, default="claude"), default_router(default_provider="claude"), LLMBudget(max_usd=2.0, max_tokens=5)).approval_detail == "claude+openrouter budget max_usd=2 max_tokens=5"
    scripted = LLMService.from_script([{"content": "s"}], LLMBudget(max_tokens=100), gate=ApprovalGate())
    assert scripted.approval_detail == "script budget max_tokens=100" and scripted.providers == ["script"] and scripted.budget_required


def test_task_model_override_routes_each_task_to_its_provider_and_records_the_served_spec():
    paid, sub = _paid([{"content": "or", "usage": PAID_USAGE, "model": "a/b-2026"}]), _sub([{"content": "cl", "usage": SUB_USAGE, "model": "claude-sonnet-5-2026"}])
    pc = ProviderClient({"openrouter": paid, "claude": sub}, default="openrouter")
    router = default_router("a/b", default_provider="openrouter", by_task={"review": "claude:sonnet"})
    svc = _service(pc, router, LLMBudget(max_usd=1.0))
    r_or = svc.complete(TASK, MSGS)
    r_cl = svc.complete(TaskKind.REVIEW, MSGS)
    assert paid.calls[0].model == "a/b" and sub.calls[0].model == "sonnet"
    assert (r_or.via, r_or.model_used) == ("openrouter", "openrouter:a/b-2026") and (r_cl.via, r_cl.model_used) == ("claude", "claude:claude-sonnet-5-2026")
    assert [(r.task, r.model, r.via) for r in svc.usage.records] == [(TASK, "openrouter:a/b-2026", "openrouter"), (TaskKind.REVIEW, "claude:claude-sonnet-5-2026", "claude")]
    assert [(a.model, a.model_used, a.via) for a in svc.attempts] == [("openrouter:a/b", "openrouter:a/b-2026", "openrouter"), ("claude:sonnet", "claude:claude-sonnet-5-2026", "claude")]


def test_retryable_failure_falls_back_to_the_same_model_on_the_other_provider_only_when_configured():
    def make(with_fallback: bool):
        paid = _paid([{"error": {"message": "upstream", "kind": "response", "status": 200, "code": 502, "sent": True}}, {"content": "never", "usage": PAID_USAGE}])
        sub = _sub([{"content": "from claude", "usage": SUB_USAGE}])
        pc = ProviderClient({"openrouter": paid, "claude": sub}, default="openrouter")
        primary = parse_model_spec("openrouter:anthropic/claude-sonnet-5")
        fallback = same_model_fallback(primary, ("openrouter", "claude")).spec if with_fallback else None
        router = default_router(primary.spec, default_provider="openrouter", fallback=fallback)
        return _service(pc, router, LLMBudget(max_usd=1.0, max_tokens=10_000)), paid, sub

    svc, paid, sub = make(with_fallback=False)
    with pytest.raises(LLMError) as ei:
        svc.complete(TASK, MSGS)
    assert ei.value.via == "openrouter" and ei.value.code == 502 and sub.calls == [] and len(paid.calls) == 1  # no default fallback: the error surfaces
    assert [r.outcome for r in svc.usage.records] == ["failed"] and svc.usage.records[0].via == "openrouter" and svc.usage.records[0].model == "openrouter:anthropic/claude-sonnet-5"
    svc, paid, sub = make(with_fallback=True)
    resp = svc.complete(TASK, MSGS)
    assert resp.content == "from claude" and resp.via == "claude" and resp.model_used == "claude:claude-sonnet-5"
    assert sub.calls[0].model == "claude-sonnet-5" and len(paid.calls) == 1
    assert [(r.outcome, r.via, r.model) for r in svc.usage.records] == [("failed", "openrouter", "openrouter:anthropic/claude-sonnet-5"), ("served", "claude", "claude:claude-sonnet-5")]
    assert [(a.outcome, a.via) for a in svc.last_attempts] == [("response", "openrouter"), ("served", "claude")]
    assert "1 failed call(s) possibly billed" in svc.summary() and "by provider: openrouter 1 call(s) 0 tokens 0.000000 USD (+1 unknown), claude 1 call(s) 150 tokens" in svc.summary()


def test_structured_through_the_claude_route_uses_the_schema_and_the_served_spec():
    from pydantic import BaseModel, ConfigDict

    class Answer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    sub = _sub([{"structured": {"value": 4}, "usage": SUB_USAGE, "model": "claude-sonnet-5-2026"}])
    svc = _service(ProviderClient({"claude": sub}, default="claude"), default_router(default_provider="claude"), LLMBudget(), providers=("claude",))
    inst, resp = svc.structured(TASK, MSGS, Answer)
    assert inst.value == 4 and resp.model_used == "claude:claude-sonnet-5-2026" and resp.via == "claude"
    assert sub.calls[0].response_schema is not None and sub.calls[0].response_schema["properties"] == {"value": {"title": "Value", "type": "integer"}}  # supports_structured: the schema is requested
    assert sub.calls[0].tools is None and sub.calls[0].model == DEFAULT_CLAUDE_MODEL


def test_tools_skip_a_route_that_carries_none():
    tool = ToolSpec(name="lookup", description="x", parameters={"type": "object"})
    sub = _sub()
    svc = _service(ProviderClient({"claude": sub}, default="claude"), default_router(default_provider="claude"), LLMBudget(), providers=("claude",))
    with pytest.raises(ValueError, match="no candidate model carries caller tools \\(claude:claude-sonnet-5\\); nothing was sent"):
        svc.complete(TASK, MSGS, tools=[tool])
    assert sub.calls == [] and svc.last_attempts[-1].outcome == "unsupported" and svc.last_attempts[-1].via == "claude"
    paid = _paid()
    pc = ProviderClient({"claude": sub, "openrouter": paid}, default="claude")
    svc = _service(pc, default_router(default_provider="claude", fallback="openrouter:a/b"), LLMBudget(max_usd=1.0))
    resp = svc.complete(TASK, MSGS, tools=[tool])
    assert resp.via == "openrouter" and sub.calls == [] and paid.calls[0].tools == [tool] and paid.calls[0].model == "a/b"
    assert [a.outcome for a in svc.last_attempts] == ["unsupported", "served"]
    assert svc.complete(TASK, MSGS).via == "claude"  # without tools the CLI route is the primary again


def test_stream_through_the_service_records_the_served_spec_and_via():
    sub = _sub([{"content": "a b c", "usage": SUB_USAGE}])
    svc = _service(ProviderClient({"claude": sub}, default="claude"), default_router("claude:sonnet"), LLMBudget(max_tokens=1000), providers=("claude",))
    assert "".join(svc.stream(TaskKind.CHAT, MSGS)) == "a b c"
    [rec] = svc.usage.records
    assert rec.model == "claude:sonnet" and rec.via == "claude" and rec.usage.cost_usd == 0.0 and rec.tokens == 150
    assert svc.attempts[-1].model == "claude:sonnet" and svc.attempts[-1].via == "claude"


def test_bare_client_with_a_prefixed_config_receives_the_native_id():
    sub = _sub()
    svc = _service(sub, default_router("claude:sonnet"), LLMBudget(), providers=("claude",))
    resp = svc.complete(TASK, MSGS)
    assert sub.calls[0].model == "sonnet" and resp.model_used == "sonnet" and svc.usage.records[0].model == "sonnet" and svc.usage.records[0].via == "claude"


# --------------------------------------------------------------------------- usage semantics


def test_subscription_usage_is_a_known_zero_charge_with_an_estimate():
    assert is_charge(Usage()) and is_charge(Usage(cost_usd=0.0, cost_source="script")) and is_charge(Usage(cost_usd=None))
    assert not is_charge(Usage(cost_usd=0.0, billing="subscription")) and not is_charge(Usage(cost_usd=0.0, cost_source="subscription"))
    t = UsageTracker()
    t.record(TASK, "claude:sonnet", Usage(**SUB_USAGE), via="claude")
    t.record(TASK, "openrouter:a/b", Usage(**PAID_USAGE), via="openrouter")
    assert [r.charge for r in t.records] == [False, True] and t.total_cost_usd() == pytest.approx(0.002)  # 0.0 is known, so the total is known
    assert t.total_estimated_cost_usd() == pytest.approx(0.0123) and [r.model for r in t.subscription_records()] == ["claude:sonnet"]
    assert UsageTracker().total_estimated_cost_usd() is None
    assert t.records[0].tokens == 150 and t.records[1].tokens == 15


# --------------------------------------------------------------------------- GUI hook


def test_describe_providers_openrouter_by_key_and_claude_by_cli_login():
    rows = describe_providers(environ={}, claude_factory=lambda: FakeClaudeCodeClient(cli="/x/claude"))
    assert [r.name for r in rows] == ["openrouter", "claude"] and all(isinstance(r, ProviderInfo) for r in rows)
    orow, crow = rows
    assert (orow.available, orow.reason, orow.billing) == (False, "OPENROUTER_API_KEY not set", "per_call")
    assert (crow.available, crow.reason, crow.billing, crow.logged_in) == (True, "/x/claude (2.1.283, logged in: yes, auth: oauth_token)", "subscription", True)
    assert describe_providers(environ={OPENROUTER_ENV_KEY: "sk-or-v1-x"}, claude_factory=lambda: FakeClaudeCodeClient())[0].model_dump() == {"name": "openrouter", "available": True, "reason": "OPENROUTER_API_KEY set", "billing": "per_call", "logged_in": None}
    assert describe_providers(environ={OPENROUTER_ENV_KEY: "   "}, claude_factory=lambda: FakeClaudeCodeClient())[0].available is False
    from ai_eda.llm.openrouter import ENV_KEY

    assert OPENROUTER_ENV_KEY == ENV_KEY


def test_describe_providers_claude_states():
    def missing():
        raise ToolUnavailableError("claude not found (install Claude Code and run `claude login`, or set AI_EDA_CLAUDE_CLI)")

    row = describe_providers(environ={}, claude_factory=missing)[1]
    assert row.available is False and row.reason.startswith("claude not found") and row.logged_in is None

    class LoggedOut(FakeClaudeCodeClient):
        def login_state(self):
            return {"loggedIn": False, "authMethod": None}

    row = describe_providers(environ={}, claude_factory=LoggedOut)[1]
    assert row.available is False and row.reason == f"{FAKE_CLI} (2.1.283, logged in: no, auth: unknown)" and row.logged_in is False

    class Unknown(FakeClaudeCodeClient):
        def version(self):
            return None

        def login_state(self):
            return {}

    row = describe_providers(environ={}, claude_factory=Unknown)[1]
    assert row.available is True and row.reason == f"{FAKE_CLI} (version unknown, logged in: unknown, auth: unknown)" and row.logged_in is None

    class Attrs(FakeClaudeCodeClient):
        def login_state(self):
            class State:
                logged_in = True
                auth_method = "api_key"

            return State()

    assert describe_providers(environ={}, claude_factory=Attrs)[1].reason.endswith("logged in: yes, auth: api_key)")

    class Broken(FakeClaudeCodeClient):
        def login_state(self):
            raise RuntimeError("auth status exited 1")

    row = describe_providers(environ={}, claude_factory=Broken)[1]
    assert row.available is False and row.reason.endswith("logged in: unknown, auth: unknown); auth status exited 1")

    class Reported(FakeClaudeCodeClient):
        def login_state(self):
            class State:
                logged_in = None
                auth_method = None
                error = "claude auth status could not run"

            return State()

    row = describe_providers(environ={}, claude_factory=Reported)[1]
    assert row.available is False and row.reason.endswith("logged in: unknown, auth: unknown); claude auth status could not run")

    def import_error():
        raise ImportError("no module named ai_eda.llm.claude_cli")

    row = describe_providers(environ={}, claude_factory=import_error)[1]
    assert row.available is False and "Claude Code CLI client unavailable: no module named" in row.reason


def test_package_exports():
    import ai_eda.llm as llm

    assert llm.ProviderClient is ProviderClient and llm.describe_providers is describe_providers and llm.parse_model_spec is parse_model_spec
    assert llm.DEFAULT_CLAUDE_MODEL == DEFAULT_CLAUDE_MODEL and "ClaudeCodeClient" in llm.__all__ and "OpenRouterClient" in llm.__all__
    with pytest.raises(AttributeError):
        llm.NoSuchThing  # noqa: B018
