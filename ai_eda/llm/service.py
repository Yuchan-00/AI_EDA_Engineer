"""LLM service: budget, approval, retry / fallback and schema validation around an :class:`LLMClient`.

Invariants enforced here:

* **Spending needs explicit approval.** An :class:`LLMBudget` is what the
  user granted (the CLI's ``--llm-budget-usd`` / ``--llm-budget-tokens`` flags
  are that grant). Construction records it in the
  :class:`~ai_eda.security.ApprovalGate` as a single-use
  ``PAID_API_CALL`` approval and consumes it at once, so the audit log shows
  who allowed which budget; a budget with neither limit grants nothing and
  construction is refused by the gate (``ApprovalRequiredError``) before any
  request is built.
* **Every request is checked against the budget first, and every request
  is capped.** Spend is what the :class:`~ai_eda.llm.usage.UsageTracker`
  holds: provider-reported cost where the provider reported one, tokens
  otherwise. The budget is a *pre-condition*: a cost budget refuses as soon
  as the known spend has *reached* it (``known >= max_usd``), a ``0`` USD
  budget refuses every call to a paid client outright (the first call cannot
  be priced in advance, so "0" means "no paid call") and allows only a client
  that declares itself free (``LLMClient.paid is False``, the scripted
  client) as long as it reports no cost; when any recorded call has an
  unknown cost and there is no token budget the service refuses rather than
  assume ``0``. With both budgets set, unknown-cost records count as ``0``
  USD and only the token budget bounds them - :meth:`LLMService.summary`
  and a WARNING log line say so explicitly whenever that happens. Every
  request carries ``max_tokens``: the caller's value, else the model
  config's, else the task default (:func:`~ai_eda.llm.router.max_tokens_for`),
  lowered to the remaining token budget when one is set. A refusal is
  :class:`BudgetExceededError`, raised *instead of* calling. The service
  never retries into a different budget: a retry or fallback attempt is
  checked exactly like a first attempt.

  Known gap: the cost of the *next* request is not estimated (that needs
  per-model pricing from ``/models`` and a tokenizer estimate), so a budget
  of ``X`` bounds the spend at ``X`` plus one request capped by
  ``max_tokens`` - not at ``X`` exactly.
* **Retry / fallback follow the provider's error class, not text.**
  ``429`` is retried once on the same model after a backoff that honours
  ``Retry-After`` capped at ``backoff_cap`` seconds; ``408`` / ``5xx`` /
  transport failures (and a ``429`` that persists) fall back to the next
  :meth:`~ai_eda.llm.router.ModelRouter.candidates` entry; ``400`` / ``401``
  / ``402`` / ``403`` / ``404`` / ``405`` / ``413`` / ``422``
  (:data:`TERMINAL_STATUSES`) are never retried and never fall back - a bad
  key, an empty account or an oversized request is not something another
  model fixes.
* **A schema failure gets one feedback turn.** :meth:`structured` validates
  the reply with pydantic; an unparseable, truncated (``finish_reason ==
  "length"``) or invalid reply is fed back once (the error text, never a
  hint about what to answer) to the *same* model; a second failure moves on
  to the next candidate. What survives validation is still a proposal - the
  caller grounds it.
* **Accounting is complete.** Every served reply is recorded in the tracker
  under its task and ``response.model_used`` (billing follows the model that
  answered, which under fallback is not the one asked for). So is every
  failure the provider may have billed - a 2xx body or SSE frame that
  carried an error (with the usage it reported, when it reported one), a
  transport failure after the request was sent - and every stream the
  consumer stopped reading before the end (``outcome="abandoned"``, usage
  from the client's ``last_stream_response`` when it has one). Every attempt,
  served or failed, is in :attr:`LLMService.attempts` /
  :attr:`LLMService.last_attempts`. A reply without a provider cost is
  recorded with ``cost_usd=None`` (unknown), never ``0``.
* **Logs carry no content and never the key.** INFO lines hold task, model,
  status, token counts, cost and timings only.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ai_eda.errors import AiEdaError
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolSpec, Usage
from ai_eda.llm.prompts import json_only_system_message, rejection_feedback_message
from ai_eda.llm.router import ModelConfig, ModelRouter, TaskKind, default_router, max_tokens_for
from ai_eda.llm.usage import UsageTracker
from ai_eda.security.approval import ApprovalGate, ExternalAction, default_gate, require_approval

log = logging.getLogger("ai_eda.llm.service")

M = TypeVar("M", bound=BaseModel)

#: HTTP statuses that are never retried and never fall back (the request or the account is wrong, not the model)
TERMINAL_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 405, 413, 422})


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
    model: str
    model_used: str | None = None
    ok: bool
    #: "served" | "abandoned" | "http" | "response" | "stream" | "transport" | "script" | "validation" | "budget"
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
        self.approval_detail = f"openrouter budget {budget.describe()}"
        # The budget *is* the user's approval of paid calls; record it and consume it so the audit shows both.
        if budget.granted:
            self.gate.grant(ExternalAction.PAID_API_CALL, self.approval_detail, approved_by=approved_by)
        require_approval(ExternalAction.PAID_API_CALL, self.approval_detail, self.gate)
        log.info("llm service ready budget=%s client=%s", budget.describe(), type(client).__name__)

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
        **client_kw: Any,
    ) -> "LLMService":
        """An OpenRouter-backed service; the key comes from ``OPENROUTER_API_KEY`` (``ToolUnavailableError`` when absent).

        Reasoning is off (``{"effort": "none"}``): Sonnet 5 reasons by default and bills it as output
        tokens, which extraction does not need. The app title is sent for attribution only.
        """
        from ai_eda.llm.openrouter import OpenRouterClient

        client_kw.setdefault("reasoning", {"effort": "none"})
        client_kw.setdefault("app_title", "AI EDA ENGINEER")
        client = OpenRouterClient(**client_kw)
        return cls(client, router or default_router(), usage or UsageTracker(), budget, gate=gate, approved_by=approved_by)

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
        """``(known_cost_usd, unknown_cost_records, tokens)`` over everything the tracker holds."""
        known = 0.0
        unknown = 0
        for r in self.usage.records:
            if r.usage.cost_usd is None:
                unknown += 1
            else:
                known += r.usage.cost_usd
        tokens = sum(r.usage.total_tokens or (r.usage.prompt_tokens + r.usage.completion_tokens) for r in self.usage.records)
        return known, unknown, tokens

    def check_budget(self) -> None:
        """Raise :class:`BudgetExceededError` when another request is not covered (see the module docstring)."""
        b = self.budget
        if not b.granted:
            raise BudgetExceededError("no LLM budget granted (pass --llm-budget-usd and/or --llm-budget-tokens)")
        known, unknown, tokens = self.spent()
        if b.max_usd is not None:
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
        parts = [f"{by_outcome.get('served', 0)} served call(s)"]
        if by_outcome.get("failed"):
            parts.append(f"{by_outcome['failed']} failed call(s) possibly billed")
        if by_outcome.get("abandoned"):
            parts.append(f"{by_outcome['abandoned']} abandoned stream(s)")
        cost = f"{known:.6f} USD"
        if unknown:
            cost += f" (+{unknown} call(s) with unknown cost"
            if self.budget.max_usd is not None and self.budget.max_tokens is not None:
                cost += ", counted as 0 USD against the USD budget - bounded by the token budget only"
            cost += ")"
        return f"{', '.join(parts)}, {tokens} tokens, cost {cost}; budget {self.budget.describe()}"

    # ------------------------------------------------------------ plumbing

    def _record(self, task: TaskKind, cfg: ModelConfig, resp: LLMResponse, elapsed: float, *, outcome: str = "served") -> LLMAttempt:
        model_used = resp.model_used or cfg.model
        self.usage.record(task, model_used, resp.usage, outcome=outcome)
        a = LLMAttempt(task=task, model=cfg.model, model_used=model_used, ok=True, outcome=outcome, status=200, usage=resp.usage, elapsed_s=elapsed)
        self.attempts.append(a)
        self.last_attempts.append(a)
        log.info(
            "llm %s model=%s used=%s outcome=%s finish=%s prompt_tokens=%d completion_tokens=%d cost=%s elapsed=%.2fs",
            task, cfg.model, model_used, outcome, resp.finish_reason, resp.usage.prompt_tokens, resp.usage.completion_tokens,
            "unknown" if resp.usage.cost_usd is None else f"{resp.usage.cost_usd:.6f}", elapsed,
        )
        self._warn_unknown_cost(task, model_used, resp.usage)
        return a

    def _record_failure(self, task: TaskKind, cfg: ModelConfig, err: LLMError, elapsed: float) -> LLMAttempt:
        usage: Usage | None = None
        if err.maybe_billed:
            # the provider committed to the request (2xx error body / stream error / transport failure after the
            # request went out): something may have been generated and billed. Keep the cost it reported with the
            # error; otherwise the cost is unknown, never 0.
            usage = err.usage if err.usage is not None else Usage()
            self.usage.record(task, err.model or cfg.model, usage, outcome="failed")
            self._warn_unknown_cost(task, err.model or cfg.model, usage)
        a = LLMAttempt(task=task, model=cfg.model, model_used=err.model, ok=False, outcome=err.kind, status=err.status, error=str(err), usage=usage, elapsed_s=elapsed)
        self.attempts.append(a)
        self.last_attempts.append(a)
        log.info(
            "llm %s model=%s failed kind=%s status=%s code=%s retry_after=%s billed=%s cost=%s elapsed=%.2fs",
            task, cfg.model, err.kind, err.status, err.code, err.retry_after,
            "maybe" if err.maybe_billed else "no",
            "n/a" if usage is None else ("unknown" if usage.cost_usd is None else f"{usage.cost_usd:.6f}"), elapsed,
        )
        return a

    def _warn_unknown_cost(self, task: TaskKind, model: str, usage: Usage) -> None:
        """Say out loud when a USD budget stops binding because a record has no cost and only the token budget bounds it."""
        if usage.cost_usd is None and self.budget.max_usd is not None and self.budget.max_tokens is not None:
            log.warning(
                "llm %s model=%s recorded with unknown cost: counted as 0 USD against the %g USD budget; only the %d token budget bounds it",
                task, model, self.budget.max_usd, self.budget.max_tokens,
            )

    def _note(self, task: TaskKind, cfg: ModelConfig, outcome: str, error: str) -> LLMAttempt:
        a = LLMAttempt(task=task, model=cfg.model, ok=False, outcome=outcome, error=error)
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
        """One request to ``cfg.model`` with the budget check, the completion cap, the single 429 backoff and accounting.

        Raises :class:`LLMError` (caller decides fallback), :class:`BudgetExceededError`.
        """
        rate_limited_once = False
        while True:
            try:
                self.check_budget()
            except BudgetExceededError as e:
                self._note(task, cfg, "budget", str(e))
                raise
            cap = self.request_cap(task, cfg, max_tokens)
            t0 = time.monotonic()
            try:
                resp = self.client.complete(
                    cfg.model, messages, tools=tools, response_schema=response_schema,
                    temperature=cfg.temperature, max_tokens=cap,
                )
            except LLMError as e:
                self._record_failure(task, cfg, e, time.monotonic() - t0)
                if e.rate_limited and e.status not in TERMINAL_STATUSES and not rate_limited_once:
                    rate_limited_once = True
                    wait = self._backoff(e)
                    log.info("llm %s model=%s rate limited; retrying once after %.2fs", task, cfg.model, wait)
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
        for cfg in self.router.candidates(task):
            try:
                return self._call(task, cfg, messages, tools=tools, response_schema=response_schema, max_tokens=max_tokens)
            except LLMError as e:
                last_error = e
                if not self._fallback_ok(e):
                    raise
                log.info("llm %s model=%s failed (%s); trying next candidate", task, cfg.model, e.kind)
        assert last_error is not None
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
                    validation_errors.append(f"{cfg.model}: {problem.splitlines()[0][:200]}")
                    self._note(task, cfg, "validation", problem)
                    log.info("llm %s model=%s reply rejected by schema (turn %d)", task, cfg.model, turn + 1)
                    if turn == 0:
                        msgs = [*msgs, LLMMessage(role="assistant", content=resp.content or ""), rejection_feedback_message(problem)]
            except LLMError as e:
                last_error = e
                if not self._fallback_ok(e):
                    raise
                log.info("llm %s model=%s failed (%s); trying next candidate", task, cfg.model, e.kind)
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
                self.check_budget()
            except BudgetExceededError as e:
                self._note(task, cfg, "budget", str(e))
                raise
            cap = self.request_cap(task, cfg, max_tokens)
            t0 = time.monotonic()
            delivered = False
            completed = False
            failed = False
            inner = self.client.stream(cfg.model, messages, temperature=cfg.temperature, max_tokens=cap)
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
        resp = final if isinstance(final, LLMResponse) else LLMResponse(model=cfg.model)
        if not isinstance(usage, Usage):
            usage = Usage()  # no usage frame: tokens unknown (0) and cost unknown (None)
        resp = resp.model_copy(update={"usage": usage})
        self._record(task, cfg, resp, time.monotonic() - t0, outcome=outcome)


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
    "StructuredOutputError",
    "TERMINAL_STATUSES",
]
