from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ai_eda.llm.client import Usage
from ai_eda.llm.router import TaskKind


class UsageRecord(BaseModel):
    task: TaskKind
    model: str
    usage: Usage
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageTracker(BaseModel):
    records: list[UsageRecord] = Field(default_factory=list)

    def record(self, task: TaskKind, model: str, usage: Usage) -> None:
        self.records.append(UsageRecord(task=task, model=model, usage=usage))

    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.records)

    def total_cost_usd(self) -> float | None:
        costs = [r.usage.cost_usd for r in self.records]
        if any(c is None for c in costs):
            return None  # unknown cost is reported as unknown, not zero
        return float(sum(costs))
