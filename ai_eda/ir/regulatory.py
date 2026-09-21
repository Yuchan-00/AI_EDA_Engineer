"""Regulatory model.

Two rules from the spec are encoded here:

1. Jurisdiction is never guessed. ``RegulatoryState.jurisdictions`` empty
   means the regulatory stage returns USER_INPUT_REQUIRED.
2. Having read an official document is not the same as being compliant.
   ``verification_status`` describes what *we* checked; it never asserts
   legal certification.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.validation import ValidationStatus


class Jurisdiction(BaseModel):
    code: str  # "EU", "US", "KR", "JP", ...
    name: str
    #: how we know this applies: always user supplied
    provided_by_user: bool = True


class RegulatoryProvenance(BaseModel):
    jurisdiction: str
    authority: str  # e.g. "European Commission", "FCC", "KC/RRA"
    source_title: str
    source_url: str | None = None
    retrieved_at: datetime | None = None
    section: str | None = None
    applicability_rationale: str = ""
    verification_status: ValidationStatus = ValidationStatus.NOT_VERIFIED
    source_document: str | None = None  # archived local path
    content_hash: str | None = None


class RegulatoryRequirement(BaseModel):
    id: str  # "reg.EU.LVD.2014-35-EU"
    jurisdiction: str
    title: str
    summary: str = ""
    #: what the design must satisfy, in engineering terms, if we could extract it
    engineering_implication: str = ""
    status: ValidationStatus = ValidationStatus.NOT_VERIFIED
    provenance: RegulatoryProvenance


class RegulatoryState(BaseModel):
    jurisdictions: list[Jurisdiction] = Field(default_factory=list)
    intended_use: str | None = None
    requirements: list[RegulatoryRequirement] = Field(default_factory=list)

    @property
    def jurisdiction_known(self) -> bool:
        return bool(self.jurisdictions)
