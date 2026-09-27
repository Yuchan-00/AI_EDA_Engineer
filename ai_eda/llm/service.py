"""LLM service: budget, approval, retry / fallback and schema validation around an :class:`LLMClient`.

Invariants enforced here:

* **Spending needs explicit approval; a subscription needs the flag.** An
  :class:`LLMBudget` is what the user granted (the CLI's ``--llm-budget-usd``
  / ``--llm-budget-tokens`` flags are that grant). Whenever a *per-call*
  provider is configured (``client.billing`` is ``"per_call"`` or
  ``"mixed"``: OpenRouter, the scripted client, or a mix that includes one),
  construction records the budget in the :class:`~ai_eda.security.ApprovalGate`
  as a single-use ``PAID_API_CALL`` approval and consumes it at once, so the
  audit log shows who allowed which budget; a budget with neither limit
  grants nothing and construction is refused by the gate
  (``ApprovalRequiredError``) before any request is built. When ONLY
  subscription providers are configured (``client.billing ==
  "subscription"``: the Claude Code CLI on the user's own login) no budget
  is required - the ``--llm claude`` flag is the approval: construction
  records ``SUBSCRIPTION_USE`` with the detail ``claude subscription via
  <cli path> (no per-call cost; usage shown)`` and consumes it exactly like
  the budget grant; an optional token budget is still honoured. A mix
  records both approvals (the budget for the per-call side, the
  subscription use for the other).
* **Every request is checked against the budget first, and every request
  is capped.** Spend is what the :class:`~ai_eda.llm.usage.UsageTracker`
  holds. The USD budget counts *charges* only - records that are not
  subscription records (:func:`~ai_eda.llm.usage.is_charge`): provider-
  reported cost where the provider reported one, unknown otherwise - and
  binds only a request to a per-call route (a subscription call cannot
  add a charge); the token budget counts every record, subscription calls
  included, and binds every request. The budget is a *pre-condition*: a cost budget refuses as soon as the known
  charges have *reached* it (``known >= max_usd``), a ``0`` USD budget
  refuses every call to a paid client outright (the first call cannot be
  priced in advance, so "0" means "no paid call") and allows only a client
  that declares itself free (``LLMClient.paid is False``: the scripted
  client, a subscription client) as long as it reports no charge; when any
  charge record has an unknown cost and there is no token budget the
  service refuses rather than assume ``0``. With both budgets set,
  unknown-cost records count as ``0`` USD and only the token budget bounds
  them - :meth:`LLMService.summary` and a WARNING log line say so
  explicitly whenever that happens. Every request carries ``max_tokens``:
  the caller's value, else the model config's, else the task default
  (:func:`~ai_eda.llm.router.max_tokens_for`), lowered to the remaining
  token budget when one is set - except that the Claude Code CLI route
  cannot honour it (the CLI has no such setting; the client records
  ``raw["max_tokens_ignored"]``), so there a token budget is a
  pre-condition only. A refusal is :class:`BudgetExceededError`, raised
  *instead of* calling. The service never retries into a different budget:
  a retry or fallback attempt is checked exactly like a first attempt.

  Known gap: the cost of the *next* request is not estimated (that needs
  per-model pricing from ``/models`` and a tokenizer estimate), so a budget
  of ``X`` bounds the spend at ``X`` plus one request capped by
  ``max_tokens`` - not at ``X`` exactly; on the CLI route that one request
  is not capped at all (its only in-call cap is the USD-denominated
  ``--max-budget-usd``, passed only with a USD budget).
* **Retry / fallback follow the provider's error class, not text.**
  ``429`` is retried once on the same model after a backoff that honours
  ``Retry-After`` capped at ``backoff_cap`` seconds; ``408`` / ``5xx`` /
  transport failures (and a ``429`` that persists) fall back to the next
  :meth:`~ai_eda.llm.router.ModelRouter.candidates` entry - which exists
  only when the user configured one (decision 4: no default fallback);
  ``400`` / ``401`` / ``402`` / ``403`` / ``404`` / ``405`` / ``413`` /
  ``422`` (:data:`TERMINAL_STATUSES`) are never retried and never fall back
  - a bad key, an empty account or an oversized request is not something
  another model fixes. A candidate whose route carries no caller tools
  (``supports_tools=False``, the CLI) is skipped for a request with tools
  and noted; when no candidate can carry them the request is a
  ``ValueError`` before anything is sent.
* **A schema failure gets one feedback turn.** :meth:`structured` validates
  the reply with pydantic; an unparseable, truncated (``finish_reason ==
  "length"``) or invalid reply is fed back once (the error text, never a
  hint about what to answer) to the *same* model; a second failure moves on
  to the next candidate. What survives validation is still a proposal - the
  caller grounds it.
* **Accounting is complete and names the route.** Every served reply is
  recorded in the tracker under its task and the *served spec*
  (``response.model_used``: ``provider:<id the provider reported>`` through
  a :class:`~ai_eda.llm.providers.ProviderClient`, the native id from a
  bare client; billing follows the model that answered, which under
  fallback is not the one asked for) with ``via`` = the provider. So is
  every failure the provider may have billed - a 2xx body or SSE frame that
  carried an error (with the usage it reported, when it reported one), a
  transport failure after the request was sent - and every stream the
  consumer stopped reading before the end (``outcome="abandoned"``, usage
  from the client's ``last_stream_response`` when it has one). Every attempt,
  served or failed, is in :attr:`LLMService.attempts` /
  :attr:`LLMService.last_attempts`. A reply without a provider cost is
  recorded with ``cost_usd=None`` (unknown), never ``0``; a subscription
  reply's ``0.0`` is a known zero charge and its estimate is shown by
  :meth:`summary`, never budgeted. A record on a subscription route is a
  subscription record even when the route reported no usage (a CLI timeout,
  an unparseable reply): a known zero charge with unknown tokens and no
  estimate - never an unknown-cost *charge*, which would stop the USD
  budget of the per-call side.
* **Logs carry no content and never the key.** INFO lines hold task, model,
  provider, status, token counts, cost and timings only.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ai_eda.errors import AiEdaError
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolSpec, Usage
from ai_eda.llm.prompts import json_only_system_message, rejection_feedback_message
from ai_eda.llm.providers import ClientFactory, ProviderClient, build_client, check_providers, members_of, provider_names_of, subscription_detail
from ai_eda.llm.router import ModelConfig, ModelRouter, TaskKind, default_router, max_tokens_for
from ai_eda.llm.usage import SUBSCRIPTION, UsageTracker
from ai_eda.security.approval import ApprovalGate, ExternalAction, default_gate, require_approval

log = logging.getLogger("ai_eda.llm.service")

M = TypeVar("M", bound=BaseModel)

#: HTTP statuses that are never retried and never fall back (the request or the account is wrong, not the model)
TERMINAL_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 405, 413, 422})
#: who approves the subscription use: the ``--llm claude`` flag itself (no budget flag exists for it)
SUBSCRIPTION_APPROVED_BY = "cli --llm claude"


class BudgetExceededError(AiEdaError):
    """The granted :class:`LLMBudget` does not cover another request; nothing was sent."""


class StructuredOutputError(AiEdaError):
    """No candidate model produced a reply that validates against the requested schema."""

    def __init__(self, message: str, attempts: list["LLMAttempt"]) -> None:
        self.attempts = attempts
        super().__init__(message)


class LLMBudget(BaseModel):
    """What the user allowed the service to spend. ``None`` means "no limit of that kind", not "unlimited"."""

    max_usd: float | None = None
    max_tokens: int | None = None

    @property
    def granted(self) -> bool:
        return self.max_usd is not None or self.max_tokens is not None

    def describe(self) -> str:
        parts = []
        if self.max_usd is not None:
            parts.append(f"max_usd={self.max_usd:g}")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        return " ".join(parts) if parts else "none"


class LLMAttempt(BaseModel):
    """One request the service made (or was refused before making) - the audit trail of a call."""

    task: TaskKind
    #: the spec the request was addressed to (the native id with a bare client)
    model: str
    #: the served spec (``provider:<id reported>``) or the native id from a bare client
    model_used: str | None = None
    #: the provider that served or failed the request, when known
    via: str | None = None
    ok: bool
    #: "served" | "abandoned" | "http" | "response" | "stream" | "transport" | "script" | "validation" | "budget" | "unsupported"
    outcome: str
    status: int | None = None
    error: str | None = None
    usage: Usage | None = None
    elapsed_s: float = 0.0
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd if self.usage is not None else None


class LLMService:
    """See the module docstring for the invariants."""

    def __init__(
        self,
        client: LLMClient,
        router: ModelRouter,
        usage: UsageTracker,
        budget: LLMBudget,
        *,
        gate: ApprovalGate | None = None,
        approved_by: str = "user",
        providers: Sequence[str] | None = None,
        subscription_approved_by: str = SUBSCRIPTION_APPROVED_BY,
        backoff_cap: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.router = router
        self.usage = usage
        self.budget = budget
        self.gate = gate or default_gate()
        self.backoff_cap = float(backoff_cap)
        self._sleep = sleep
        self.attempts: list[LLMAttempt] = []
        self.last_attempts: list[LLMAttempt] = []
        #: the configured providers in order ("openrouter", "claude", "script"; a mix lists both)
        self.providers: list[str] = list(providers) if providers else provider_names_of(client)
        #: a ProviderClient takes specs; a bare client is one provider and takes native ids
        self._routes_specs = isinstance(client, ProviderClient)
        members = members_of(client, self.providers)
        #: providers billed per call (need the budget) and on a subscription (need the flag)
        self.per_call_providers = [name for name, m in members if getattr(m, "billing", "per_call") != SUBSCRIPTION]
        self.subscription_providers = [name for name, m in members if getattr(m, "billing", "per_call") == SUBSCRIPTION]
        self.label = "+".join(self.providers)
        self.approval_detail = f"{self.label} budget {budget.describe()}"
        if self.per_call_providers:
            # The budget *is* the user's approval of paid calls; record it and consume it so the audit shows both.
            if budget.granted:
                self.gate.grant(ExternalAction.PAID_API_CALL, self.approval_detail, approved_by=approved_by)
            require_approval(ExternalAction.PAID_API_CALL, self.approval_detail, self.gate)
        #: the SUBSCRIPTION_USE details recorded (one per subscription member; empty without one)
        self.subscription_details: list[str] = []
        for name, member in members:
            if getattr(member, "billing", "per_call") == SUBSCRIPTION:
                detail = subscription_detail(name, member)
                self.gate.grant(ExternalAction.SUBSCRIPTION_USE, detail, approved_by=subscription_approved_by)
                require_approval(ExternalAction.SUBSCRIPTION_USE, detail, self.gate)
                self.subscription_details.append(detail)
        log.info("llm service ready providers=%s budget=%s client=%s", self.label, budget.describe(), type(client).__name__)

    @property
    def budget_required(self) -> bool:
        """Whether a budget must be granted: yes as soon as a per-call provider is configured."""
        return bool(self.per_call_providers)

    # ------------------------------------------------------------ factories

    @classmethod
    def from_env(
        cls,
        budget: LLMBudget,
        router: ModelRouter | None = None,
        usage: UsageTracker | None = None,
        *,
        gate: ApprovalGate | None = None,
        approved_by: str = "user",
        providers: Sequence[str] = ("openrouter",),
        factories: Mapping[str, ClientFactory] | None = None,
        **client_kw: Any,
    ) -> "LLMService":
        """A service over the real clients of ``providers`` (the first is the default provider for unprefixed specs).

        One provider gives a bare client (OpenRouter: the key comes from
        ``OPENROUTER_API_KEY``, ``ToolUnavailableError`` when absent; claude:
        :class:`~ai_eda.llm.claude_cli.ClaudeCodeClient`, ``ToolUnavailableError``
        when no CLI is found), several a :class:`~ai_eda.llm.providers.ProviderClient`.
        ``client_kw`` reaches each client by its constructor signature (see
        :func:`~ai_eda.llm.providers.build_client`); ``factories`` overrides a
        provider's class (tests pass a fake for ``claude``). A router candidate
        naming a provider that is not configured is a ``ValueError`` before any
        client is built.

        OpenRouter reasoning is off (``{"effort": "none"}``): Sonnet 5 reasons by
        default and bills it as output tokens, which extraction does not need;
        the app title is sent for attribution only. The CLI client gets
        ``max_budget_usd`` from the USD budget when it accepts that option (a
        second safety net; the service's own check remains the first).
        """
        names = check_providers(providers)
        router = router or default_router(default_provider=names[0])
        check_router_providers(router, names)
        if "openrouter" in names:
            client_kw.setdefault("reasoning", {"effort": "none"})
            client_kw.setdefault("app_title", "AI EDA ENGINEER")
        soft_kw: dict[str, Any] = {}
        if "claude" in names and budget.max_usd is not None:
            soft_kw["max_budget_usd"] = budget.max_usd
        client = build_client(names, factories=factories, soft_kw=soft_kw, **client_kw)
        return cls(client, router, usage or UsageTracker(), budget, gate=gate, approved_by=approved_by, providers=names)

    @classmethod
    def from_script(
        cls,
        script: str | Path | Sequence[Any] | LLMClient,
        budget: LLMBudget,
        router: ModelRouter | None = None,
        usage: UsageTracker | None = None,
        *,
        gate: ApprovalGate | None = None,
        approved_by: str = "user",
    ) -> "LLMService":
        """A service over a :class:`~ai_eda.llm.fake.ScriptedLLMClient` (a JSON file path, a list of script items, or a client)."""
        from ai_eda.llm.fake import ScriptedLLMClient

        if isinstance(script, LLMClient):
            client = script
        elif isinstance(script, (str, Path)):
            client = ScriptedLLMClient.from_file(script)
        else:
            client = ScriptedLLMClient.from_spec(list(script))
        return cls(client, router or default_router(), usage or UsageTracker(), budget, gate=gate, approved_by=approved_by)

    # ------------------------------------------------------------ budget

    def spent(self) -> tuple[float, int, int]:
        """``(known_charges_usd, charge_records_with_unknown_cost, tokens)``: charges over the tracker's charge records, tokens over every record."""
        known = 0.0
        unknown = 0
        tokens = 0
        for r in self.usage.records:
            tokens += r.tokens
            if not r.charge:
                continue
            if r.usage.cost_usd is None:
                unknown += 1
            else:
                known += r.usage.cost_usd
        return known, unknown, tokens

    def subscription_spent(self) -> tuple[int, int, float | None]:
        """``(calls, tokens, estimated_api_equivalent_usd)`` over the subscription records (the estimate is ``None`` when any lacks one)."""
        subs = self.usage.subscription_records()
        return len(subs), sum(r.tokens for r in subs), self.usage.total_estimated_cost_usd()

    def provider_of(self, cfg: ModelConfig) -> str | None:
        """The provider that would serve ``cfg``: its own, else the routing client's default, else the bare client's."""
        return self._via(None, cfg)

    def is_per_call(self, cfg: ModelConfig) -> bool:
        """Whether a request to ``cfg`` is a charge (its provider bills per call); a subscription route is not."""
        return self.provider_of(cfg) not in self.subscription_providers

    def check_budget(self, cfg: ModelConfig | None = None) -> None:
        """Raise :class:`BudgetExceededError` when another request is not covered (see the module docstring).

        The USD limits bind a request to a per-call route (``cfg`` omitted:
        any route); a subscription route cannot add a charge, so only the
        token budget binds it.
        """
        b = self.budget
        if not b.granted:
            if not self.budget_required:
                return  # subscription only: the flag was the approval; nothing is charged per call
            raise BudgetExceededError(
                f"no LLM budget granted for the per-call provider(s) {'+'.join(self.per_call_providers)} "
                "(pass --llm-budget-usd and/or --llm-budget-tokens)"
            )
        known, unknown, tokens = self.spent()
        if b.max_usd is not None and (cfg is None or self.is_per_call(cfg)):
            if b.max_usd <= 0 and getattr(self.client, "paid", True):
                raise BudgetExceededError(
                    f"budget {b.max_usd:g} USD: a paid client is never called for free (the first call cannot be priced "
                    "in advance); grant --llm-budget-usd > 0"
                )
            if unknown and b.max_tokens is None:
                raise BudgetExceededError(
                    f"cost of {unknown} earlier call(s) is unknown and no token budget is set; "
                    f"known spend {known:.6f} USD of {b.max_usd:g} USD"
                )
            if b.max_usd > 0 and known >= b.max_usd:
                raise BudgetExceededError(f"spent {known:.6f} USD, budget {b.max_usd:g} USD: nothing left for another request")
            if b.max_usd <= 0 and known > 0:
                raise BudgetExceededError(f"spent {known:.6f} USD, budget {b.max_usd:g} USD")
        if b.max_tokens is not None and tokens >= b.max_tokens:
            raise BudgetExceededError(f"spent {tokens} tokens, budget {b.max_tokens} tokens")

    def request_cap(self, task: TaskKind, cfg: ModelConfig, requested: int | None) -> int:
        """The ``max_tokens`` a request carries: the task/config cap lowered to the remaining token budget."""
        cap = max_tokens_for(task, cfg, requested)
        if self.budget.max_tokens is not None:
            _, _, tokens = self.spent()
            cap = max(1, min(cap, self.budget.max_tokens - tokens))
        return cap

    def summary(self) -> str:
        known, unknown, tokens = self.spent()
        by_outcome: dict[str, int] = {}
        for r in self.usage.records:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        failed_charges = sum(1 for r in self.usage.records if r.outcome == "failed" and r.charge)
        failed_subscription = by_outcome.get("failed", 0) - failed_charges
        parts = [f"{by_outcome.get('served', 0)} served call(s)"]
        if failed_charges:
            parts.append(f"{failed_charges} failed call(s) possibly billed")
        if failed_subscription:
            parts.append(f"{failed_subscription} failed call(s) on the subscription")
        if by_outcome.get("abandoned"):
            parts.append(f"{by_outcome['abandoned']} abandoned stream(s)")
        cost = f"{known:.6f} USD"
        if unknown:
            cost += f" (+{unknown} call(s) with unknown cost"
            if self.budget.max_usd is not None and self.budget.max_tokens is not None:
                cost += ", counted as 0 USD against the USD budget - bounded by the token budget only"
            cost += ")"
        budget = self.budget.describe()
        if not self.budget.granted and not self.budget_required:
            budget += " (not required: subscription only)"
        text = f"{', '.join(parts)}, {tokens} tokens, cost {cost}; budget {budget}"
        sub_calls, sub_tokens, estimate = self.subscription_spent()
        if sub_calls:
            est = f"{estimate:.6f} USD" if estimate is not None else "unknown"
            text += f"; {sub_calls} call(s) on the subscription: {sub_tokens} tokens, estimated API-equivalent cost {est} (not charged)"
        vias = []
        for r in self.usage.records:
            if r.via not in vias:
                vias.append(r.via)
        if len(vias) > 1:
            text += "; by provider: " + ", ".join(self._provider_line(v) for v in vias)
        return text

    def _provider_line(self, via: str | None) -> str:
        recs = [r for r in self.usage.records if r.via == via]
        tokens = sum(r.tokens for r in recs)
        charges = [r for r in recs if r.charge]
        if charges:
            known = sum(r.usage.cost_usd for r in charges if r.usage.cost_usd is not None)
            unknown = sum(1 for r in charges if r.usage.cost_usd is None)
            cost = f"{known:.6f} USD" + (f" (+{unknown} unknown)" if unknown else "")
        else:
            estimates = [r.usage.estimated_cost_usd for r in recs]
            est = f"{sum(e for e in estimates if e is not None):.6f} USD" if estimates and all(e is not None for e in estimates) else "unknown"
            cost = f"subscription (estimated {est}, not charged)"
        return f"{via or 'unknown'} {len(recs)} call(s) {tokens} tokens {cost}"

    # ------------------------------------------------------------ plumbing

    def _model_arg(self, cfg: ModelConfig) -> str:
        """What the client is asked for: the spec through a ProviderClient, the native id through a bare client."""
        return cfg.spec if self._routes_specs else cfg.model

    def _via(self, reported: str | None, cfg: ModelConfig) -> str | None:
        """The provider of a record: what the client reported, else the config's, else the client's default (a bare client is one provider)."""
        if reported:
            return reported
        if cfg.provider:
            return cfg.provider
        if self._routes_specs:
            return getattr(self.client, "default", None)
        return self.providers[0] if len(self.providers) == 1 else None

    def _record(self, task: TaskKind, cfg: ModelConfig, resp: LLMResponse, elapsed: float, *, outcome: str = "served") -> LLMAttempt:
        asked = self._model_arg(cfg)
        model_used = resp.model_used or asked
        via = self._via(resp.via, cfg)
        usage = self._unreported_usage(via, resp.usage)
        self.usage.record(task, model_used, usage, outcome=outcome, via=via)
        a = LLMAttempt(task=task, model=asked, model_used=model_used, via=via, ok=True, outcome=outcome, status=200, usage=usage, elapsed_s=elapsed)
        self.attempts.append(a)
        self.last_attempts.append(a)
        log.info(
            "llm %s model=%s used=%s via=%s outcome=%s finish=%s prompt_tokens=%d completion_tokens=%d cost=%s estimated=%s elapsed=%.2fs",
            task, asked, model_used, via, outcome, resp.finish_reason, usage.prompt_tokens, usage.completion_tokens,
            "unknown" if usage.cost_usd is None else f"{usage.cost_usd:.6f}",
            "n/a" if usage.estimated_cost_usd is None else f"{usage.estimated_cost_usd:.6f}", elapsed,
        )
        self._warn_unknown_cost(task, model_used, usage)
        return a

    def _on_subscription(self, via: str | None) -> bool:
        """Whether ``via`` names a configured subscription provider (its calls are never charges)."""
        return via is not None and via in self.subscription_providers

    def _unreported_usage(self, via: str | None, usage: Usage | None = None) -> Usage:
        """The usage to record for a call on ``via``, when the route reported none or reported it without billing.

        A per-call route: ``usage`` as reported, else ``Usage()`` - cost unknown,
        never 0. A subscription route: a known zero charge (``cost_usd=0.0``,
        ``billing`` / ``cost_source`` ``"subscription"``) with whatever tokens and
        estimate were reported - tokens 0 and no estimate when nothing was.
        """
        if not self._on_subscription(via):
            return usage if usage is not None else Usage()
        if usage is None:
            return Usage(cost_usd=0.0, cost_source=SUBSCRIPTION, billing=SUBSCRIPTION)
        if usage.billing is None and usage.cost_source is None:
            return usage.model_copy(update={"cost_usd": 0.0 if usage.cost_usd is None else usage.cost_usd, "cost_source": SUBSCRIPTION, "billing": SUBSCRIPTION})
        return usage

    def _record_failure(self, task: TaskKind, cfg: ModelConfig, err: LLMError, elapsed: float) -> LLMAttempt:
        asked = self._model_arg(cfg)
        via = self._via(err.via, cfg)
        usage: Usage | None = None
        if err.maybe_billed:
            # the provider committed to the request (2xx error body / stream error / transport failure after the
            # request went out): something may have been generated and billed. Keep the cost it reported with the
            # error; otherwise the cost is unknown, never 0 - except on a subscription route, which bills no call.
            usage = self._unreported_usage(via, err.usage)
            self.usage.record(task, err.model or asked, usage, outcome="failed", via=via)
            self._warn_unknown_cost(task, err.model or asked, usage)
        a = LLMAttempt(task=task, model=asked, model_used=err.model, via=via, ok=False, outcome=err.kind, status=err.status, error=str(err), usage=usage, elapsed_s=elapsed)
        self.attempts.append(a)
        self.last_attempts.append(a)
        log.info(
            "llm %s model=%s via=%s failed kind=%s status=%s code=%s retry_after=%s billed=%s cost=%s elapsed=%.2fs",
            task, asked, via, err.kind, err.status, err.code, err.retry_after,
            "maybe" if err.maybe_billed else "no",
            "n/a" if usage is None else ("unknown" if usage.cost_usd is None else f"{usage.cost_usd:.6f}"), elapsed,
        )
        return a

    def _warn_unknown_cost(self, task: TaskKind, model: str, usage: Usage) -> None:
        """Say out loud when a USD budget stops binding because a charge record has no cost and only the token budget bounds it."""
        if usage.cost_usd is None and self.budget.max_usd is not None and self.budget.max_tokens is not None:
            log.warning(
                "llm %s model=%s recorded with unknown cost: counted as 0 USD against the %g USD budget; only the %d token budget bounds it",
                task, model, self.budget.max_usd, self.budget.max_tokens,
            )

    def _note(self, task: TaskKind, cfg: ModelConfig, outcome: str, error: str) -> LLMAttempt:
        a = LLMAttempt(task=task, model=self._model_arg(cfg), via=self._via(None, cfg), ok=False, outcome=outcome, error=error)
        self.attempts.append(a)
        self.last_attempts.append(a)
        return a

    def _backoff(self, err: LLMError) -> float:
        wait = err.retry_after if err.retry_after is not None else 1.0
        return max(0.0, min(float(wait), self.backoff_cap))

    def _call(
        self,
        task: TaskKind,
        cfg: ModelConfig,
        messages: list[LLMMessage],
        *,
        tools: list[ToolSpec] | None,
        response_schema: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        """One request to ``cfg`` with the budget check, the completion cap, the single 429 backoff and accounting.

        Raises :class:`LLMError` (caller decides fallback), :class:`BudgetExceededError`.
        """
        rate_limited_once = False
        asked = self._model_arg(cfg)
        while True:
            try:
                self.check_budget(cfg)
            except BudgetExceededError as e:
                self._note(task, cfg, "budget", str(e))
                raise
            cap = self.request_cap(task, cfg, max_tokens)
            t0 = time.monotonic()
            try:
                resp = self.client.complete(
                    asked, messages, tools=tools, response_schema=response_schema,
                    temperature=cfg.temperature, max_tokens=cap,
                )
            except LLMError as e:
                self._record_failure(task, cfg, e, time.monotonic() - t0)
                if e.rate_limited and e.status not in TERMINAL_STATUSES and not rate_limited_once:
                    rate_limited_once = True
                    wait = self._backoff(e)
                    log.info("llm %s model=%s rate limited; retrying once after %.2fs", task, asked, wait)
                    self._sleep(wait)
                    continue
                raise
            self._record(task, cfg, resp, time.monotonic() - t0)
            return resp

    @staticmethod
    def _fallback_ok(err: LLMError) -> bool:
        if err.status in TERMINAL_STATUSES or (isinstance(err.code, int) and err.code in TERMINAL_STATUSES):
            return False
        return err.retryable

    # ------------------------------------------------------------ public API

    def complete(
        self,
        task: TaskKind,
        messages: list[LLMMessage],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """A completion with fallback across the router's candidates (see the module docstring)."""
        self.last_attempts = []
        last_error: LLMError | None = None
        skipped: list[str] = []
        for cfg in self.router.candidates(task):
            if tools and not cfg.supports_tools:
                self._note(task, cfg, "unsupported", f"{cfg.spec}: this route carries no caller tools")
                skipped.append(cfg.spec)
                continue
            try:
                return self._call(task, cfg, messages, tools=tools, response_schema=response_schema, max_tokens=max_tokens)
            except LLMError as e:
                last_error = e
                if not self._fallback_ok(e):
                    raise
                log.info("llm %s model=%s failed (%s); trying next candidate", task, cfg.spec, e.kind)
        if last_error is None:
            raise ValueError(f"no candidate model carries caller tools ({', '.join(skipped)}); nothing was sent")
        raise last_error

    def structured(
        self,
        task: TaskKind,
        messages: list[LLMMessage],
        schema: type[M],
        max_tokens: int | None = None,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[M, LLMResponse]:
        """A reply validated against ``schema`` (one feedback retry per model, then the next candidate).

        The JSON schema (``json_schema`` when given - e.g. a strict-mode
        variant - else ``schema.model_json_schema()``) is sent as
        ``response_format`` (strict) when the candidate supports it, otherwise
        embedded in an extra system message. Returns the validated instance
        and the response it came from (``response.model_used`` is the model
        that answered).
        """
        self.last_attempts = []
        if json_schema is None:
            json_schema = schema.model_json_schema()
        last_error: LLMError | None = None
        validation_errors: list[str] = []
        for cfg in self.router.candidates(task):
            if cfg.supports_structured:
                msgs, response_schema = list(messages), json_schema
            else:
                msgs, response_schema = [*messages, json_only_system_message(json_schema)], None
            try:
                for turn in range(2):
                    resp = self._call(task, cfg, msgs, tools=None, response_schema=response_schema, max_tokens=max_tokens)
                    instance, problem = self._validate(schema, resp)
                    if instance is not None:
                        return instance, resp
                    assert problem is not None
                    validation_errors.append(f"{cfg.spec}: {problem.splitlines()[0][:200]}")
                    self._note(task, cfg, "validation", problem)
                    log.info("llm %s model=%s reply rejected by schema (turn %d)", task, cfg.spec, turn + 1)
                    if turn == 0:
                        msgs = [*msgs, LLMMessage(role="assistant", content=resp.content or ""), rejection_feedback_message(problem)]
            except LLMError as e:
                last_error = e
                if not self._fallback_ok(e):
                    raise
                log.info("llm %s model=%s failed (%s); trying next candidate", task, cfg.spec, e.kind)
        if last_error is not None and not validation_errors:
            raise last_error
        raise StructuredOutputError(
            "no model produced a reply matching the schema: " + "; ".join(validation_errors)
            + (f"; last transport error: {last_error}" if last_error is not None else ""),
            list(self.last_attempts),
        )

    @staticmethod
    def _validate(schema: type[M], resp: LLMResponse) -> tuple[M | None, str | None]:
        if resp.truncated:
            return None, f"reply was truncated (finish_reason={resp.finish_reason!r}); the JSON object is incomplete"
        data = resp.structured
        if data is None:
            data = _json_object(resp.content)
            if data is None:
                return None, f"reply is not a JSON object ({resp.raw_error or 'no JSON object found in the content'})"
        try:
            return schema.model_validate(data), None
        except ValidationError as e:
            return None, str(e)

    def stream(self, task: TaskKind, messages: list[LLMMessage], *, max_tokens: int | None = None) -> Iterator[str]:
        """Streamed content with fallback only *before* the first piece was delivered; usage recorded at the end.

        The provider's usage frame (``client.last_stream_usage``) is recorded
        after the iterator is exhausted; a stream that ended without one is
        recorded with unknown cost. A stream the consumer stops reading
        (``close()``, ``break``, an exception of its own) is recorded too, as
        ``outcome="abandoned"`` with whatever the client's
        ``last_stream_response`` holds (the real client fills it in its
        ``finally``): the provider generated - and billed - regardless.
        """
        self.last_attempts = []
        last_error: LLMError | None = None
        for cfg in self.router.candidates(task):
            try:
                self.check_budget(cfg)
            except BudgetExceededError as e:
                self._note(task, cfg, "budget", str(e))
                raise
            cap = self.request_cap(task, cfg, max_tokens)
            t0 = time.monotonic()
            delivered = False
            completed = False
            failed = False
            inner = self.client.stream(self._model_arg(cfg), messages, temperature=cfg.temperature, max_tokens=cap)
            try:
                for piece in inner:
                    delivered = True
                    yield piece
                completed = True
            except LLMError as e:
                failed = True
                self._record_failure(task, cfg, e, time.monotonic() - t0)
                last_error = e
                if delivered or not self._fallback_ok(e):
                    raise
                continue
            finally:
                if not completed and not failed:
                    # the consumer left (GeneratorExit) or something other than the provider failed: close the
                    # client's stream first so its accounting view exists, then record what we know
                    close = getattr(inner, "close", None)
                    if callable(close):
                        close()
                    self._record_stream_end(task, cfg, t0, outcome="abandoned")
            self._record_stream_end(task, cfg, t0, outcome="served")
            return
        assert last_error is not None
        raise last_error

    def _record_stream_end(self, task: TaskKind, cfg: ModelConfig, t0: float, *, outcome: str) -> None:
        usage = getattr(self.client, "last_stream_usage", None)
        final = getattr(self.client, "last_stream_response", None)
        resp = final if isinstance(final, LLMResponse) else LLMResponse(model=self._model_arg(cfg))
        # no usage frame: tokens unknown (0) and cost unknown (None) - on a subscription route a known zero charge
        usage = self._unreported_usage(self._via(resp.via, cfg), usage if isinstance(usage, Usage) else None)
        resp = resp.model_copy(update={"usage": usage})
        self._record(task, cfg, resp, time.monotonic() - t0, outcome=outcome)


def check_router_providers(router: ModelRouter, providers: Sequence[str]) -> None:
    """``ValueError`` when a router config names a provider outside ``providers`` (before any client exists)."""
    bad = sorted({c.spec for c in router.configs() if c.provider is not None and c.provider not in providers})
    if bad:
        raise ValueError(f"model spec(s) {', '.join(bad)} name a provider that is not configured (--llm {','.join(providers)})")


def _json_object(content: str | None) -> dict[str, Any] | None:
    """The JSON object in ``content``: the whole text, or the outermost ``{...}`` span (models wrap JSON in fences)."""
    if not content:
        return None
    text = content.strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


__all__ = [
    "BudgetExceededError",
    "LLMAttempt",
    "LLMBudget",
    "LLMService",
    "SUBSCRIPTION_APPROVED_BY",
    "StructuredOutputError",
    "TERMINAL_STATUSES",
    "check_router_providers",
]
