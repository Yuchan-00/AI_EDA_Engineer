"""Requirement model.

The Requirement Agent turns free text into this structure. Nothing here is
"the design" - it is what the user asked for, what we inferred, what is
missing and what conflicts. Design decisions live elsewhere in the IR.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Traced


class RequirementKind(StrEnum):
    EXPLICIT = "explicit"  # stated by the user
    IMPLICIT = "implicit"  # follows from the application / domain
    ASSUMPTION = "assumption"  # we had to assume it; must be confirmed


class RequirementStatus(StrEnum):
    GIVEN = "given"
    MISSING = "missing"
    CONFLICTING = "conflicting"
    ASSUMED = "assumed"


class Requirement(BaseModel):
    id: str  # e.g. "req.output_voltage"
    key: str  # machine key, e.g. "output_voltage"
    text: str  # human readable statement
    kind: RequirementKind
    status: RequirementStatus = RequirementStatus.GIVEN
    value: Traced | None = None
    #: category used to select validators: "electrical", "thermal", "regulatory", "mechanical", ...
    category: str = "electrical"


class MissingInformation(BaseModel):
    """A question the system must ask before it may proceed."""

    key: str
    question: str
    required: bool = True
    #: the choices we can offer; empty means free-form
    options: list[str] = Field(default_factory=list)
    rationale: str = ""


class RequirementConflict(BaseModel):
    requirement_ids: list[str]
    description: str


class RequirementSet(BaseModel):
    raw_input: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    missing: list[MissingInformation] = Field(default_factory=list)
    conflicts: list[RequirementConflict] = Field(default_factory=list)

    def get(self, key: str) -> Requirement | None:
        for r in self.requirements:
            if r.key == key:
                return r
        return None

    @property
    def blocking_questions(self) -> list[MissingInformation]:
        return [m for m in self.missing if m.required]

    @property
    def can_proceed(self) -> bool:
        return not self.blocking_questions and not self.conflicts
