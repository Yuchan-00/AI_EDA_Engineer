"""Validation status model.

Statuses are deliberately *not* boolean. "We could not check" is a distinct,
first-class outcome and must never be collapsed into PASS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, Field


class ValidationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_VERIFIED = "NOT_VERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    USER_INPUT_REQUIRED = "USER_INPUT_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


#: Severity order used when aggregating. Higher index = worse.
_SEVERITY = [
    ValidationStatus.NOT_APPLICABLE,
    ValidationStatus.PASS,
    ValidationStatus.NOT_VERIFIED,
    ValidationStatus.USER_INPUT_REQUIRED,
    ValidationStatus.UNRESOLVED,
    ValidationStatus.FAIL,
]


def worst_status(statuses: Iterable[ValidationStatus]) -> ValidationStatus:
    """Aggregate statuses conservatively.

    An empty input is NOT_VERIFIED, not PASS: absence of evidence is not evidence.
    """
    worst: ValidationStatus | None = None
    for s in statuses:
        if worst is None or _SEVERITY.index(s) > _SEVERITY.index(worst):
            worst = s
    return worst if worst is not None else ValidationStatus.NOT_VERIFIED


class Evidence(BaseModel):
    """Something a human can open to confirm a result: a report file, a tool log, a document."""

    description: str
    path: str | None = None
    url: str | None = None
    content_hash: str | None = None


class ValidationResult(BaseModel):
    check_id: str  # e.g. "kicad.erc", "review.pcb_vs_bom", "reg.EU.LVD"
    status: ValidationStatus
    message: str = ""
    #: the deterministic tool that produced this result. Results without a tool are opinions.
    tool: str | None = None
    tool_version: str | None = None
    #: hash of the artifact the tool actually ran on, so staleness is detectable
    artifact_hash: str | None = None
    #: hash of the IR at the time of the check
    ir_hash: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_tool_backed(self) -> bool:
        return self.tool is not None


class ValidationState(BaseModel):
    results: list[ValidationResult] = Field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def extend(self, results: Iterable[ValidationResult]) -> None:
        self.results.extend(results)

    def latest(self, check_id: str) -> ValidationResult | None:
        for r in reversed(self.results):
            if r.check_id == check_id:
                return r
        return None

    def latest_by_check(self) -> dict[str, ValidationResult]:
        out: dict[str, ValidationResult] = {}
        for r in self.results:
            out[r.check_id] = r
        return out

    def overall(self) -> ValidationStatus:
        return worst_status(r.status for r in self.latest_by_check().values())

    def failing(self) -> list[ValidationResult]:
        return [r for r in self.latest_by_check().values() if r.status == ValidationStatus.FAIL]
