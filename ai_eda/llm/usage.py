"""Usage accounting for model calls.

Invariant: everything the provider may have billed is a record - served
replies, replies that failed after the provider committed to them (a 2xx
error body, a mid-stream error, a transport failure after the request went
out) and streams the consumer abandoned. A record whose cost the provider
did not report keeps ``cost_usd=None``; :meth:`UsageTracker.total_cost_usd`
then answers ``None`` (unknown), never a number that leaves such calls out.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ai_eda.llm.client import Usage
from ai_eda.llm.router import TaskKind


class UsageRecord(BaseModel):
    task: TaskKind
    model: str
    usage: Usage
    #: "served" (a delivered reply), "failed" (billed or possibly billed failure), "abandoned" (stream closed early)
    outcome: str = "served"
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageTracker(BaseModel):
    records: list[UsageRecord] = Field(default_factory=list)

    def record(self, task: TaskKind, model: str, usage: Usage, outcome: str = "served") -> None:
        self.records.append(UsageRecord(task=task, model=model, usage=usage, outcome=outcome))

    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.records)

    def total_cost_usd(self) -> float | None:
        costs = [r.usage.cost_usd for r in self.records]
        if any(c is None for c in costs):
            return None  # unknown cost is reported as unknown, not zero
        return float(sum(costs))
