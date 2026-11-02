"""Usage accounting for model calls.

Invariants:

* Everything the provider may have billed is a record - served replies,
  replies that failed after the provider committed to them (a 2xx error
  body, a mid-stream error, a transport failure after the request went out)
  and streams the consumer abandoned. A record whose cost the provider did
  not report keeps ``cost_usd=None``; :meth:`UsageTracker.total_cost_usd`
  then answers ``None`` (unknown), never a number that leaves such calls out.
* A record names the route that served it: ``model`` is the *served* spec
  (``openrouter:<response.model>``, ``claude:<the id the CLI reported>``;
  a bare single-provider client records the native id) and ``via`` the
  provider name, so a per-provider breakdown never mixes a subscription
  call with a charged one.
* ``cost_usd == 0.0`` with ``cost_source == "subscription"`` (``billing ==
  "subscription"``) is a *known* zero charge: a subscription is not billed
  per call. It is not an unknown (that is ``None``) and it is not a charge
  the USD budget counts (:func:`is_charge`). Such a record carries the
  CLI's API-equivalent estimate in ``estimated_cost_usd``: it is shown by
  the service's summary and never budgeted.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ai_eda.llm.client import Usage
from ai_eda.llm.router import TaskKind

#: the ``Usage.billing`` / ``Usage.cost_source`` value of a call on the user's subscription (a known zero charge)
SUBSCRIPTION = "subscription"


def is_charge(usage: Usage) -> bool:
    """Whether ``usage.cost_usd`` is a per-call charge (a USD budget counts it): a subscription record never is."""
    return usage.billing != SUBSCRIPTION and usage.cost_source != SUBSCRIPTION


class UsageRecord(BaseModel):
    task: TaskKind
    #: the served spec (see the module docstring)
    model: str
    usage: Usage
    #: "served" (a delivered reply), "failed" (billed or possibly billed failure), "abandoned" (stream closed early)
    outcome: str = "served"
    #: the provider that served the call ("openrouter" / "claude" / "script"); ``None`` when the client did not say
    via: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def charge(self) -> bool:
        return is_charge(self.usage)

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens or (self.usage.prompt_tokens + self.usage.completion_tokens)


class UsageTracker(BaseModel):
    records: list[UsageRecord] = Field(default_factory=list)

    def record(self, task: TaskKind, model: str, usage: Usage, outcome: str = "served", via: str | None = None) -> None:
        self.records.append(UsageRecord(task=task, model=model, usage=usage, outcome=outcome, via=via))

    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.records)

    def total_cost_usd(self) -> float | None:
        costs = [r.usage.cost_usd for r in self.records]
        if any(c is None for c in costs):
            return None  # unknown cost is reported as unknown, not zero
        return float(sum(costs))

    def subscription_records(self) -> list[UsageRecord]:
        """The records that are not charges (calls on the user's subscription)."""
        return [r for r in self.records if not r.charge]

    def total_estimated_cost_usd(self) -> float | None:
        """The summed API-equivalent estimate of the subscription records; ``None`` when any of them carries none (or there are none)."""
        subs = self.subscription_records()
        estimates = [r.usage.estimated_cost_usd for r in subs]
        if not estimates or any(e is None for e in estimates):
            return None
        return float(sum(e for e in estimates if e is not None))


__all__ = ["SUBSCRIPTION", "UsageRecord", "UsageTracker", "is_charge"]
