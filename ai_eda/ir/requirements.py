"""Requirement model.

The Requirement Agent turns free text into this structure. Nothing here is
"the design" - it is what the user asked for, what we inferred, what is
missing and what conflicts. Design decisions live elsewhere in the IR.

Trust model of a requirement's value (``Requirement.value.provenance.kind``):

* ``user_requirement`` - the user said so: typed as an answer, or an LLM
  extraction the user *confirmed* (the note then keeps the model and the
  verbatim quote it was grounded on).
* ``llm_generated`` - extracted by a model and grounded deterministically
  (verbatim quote + re-parsed number) but **not yet confirmed**; also every
  *implicit* requirement a model inferred. Never authoritative.
* ``assumption`` - a value the model assumed, or an "explicit" claim whose
  quote/number could not be grounded in the user's words (demoted). Surfaced
  by ``ir.assumptions`` as ``USER_INPUT_REQUIRED``.

``RequirementSet.corrections`` are the user's later corrections to the
request (design content: the user's own words, so they are hashed).
``RequirementSet.extraction_cache`` is **not** design content: it memoises
what a model returned for a given request text (keyed by
``sha256`` of :meth:`RequirementSet.request_text`) so an unchanged request
never costs a second call. It is part of the IR *file* but excluded from
:meth:`ai_eda.ir.CircuitIR.content_hash` - the design is what entered
``requirements``, not what the model said.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Traced, drop_in_design_view


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
    """A question the system must ask before it may proceed.

    ``source`` says who wrote the question text: ``"system"`` for the
    deterministic checklist and the pipeline's own questions, ``"llm"`` for a
    question a model authored during extraction. A model-authored question
    is the one channel through which request content reaches the human as
    text addressed to them, so it is labelled wherever it is shown.
    """

    key: str
    question: str
    required: bool = True
    #: the choices we can offer; empty means free-form
    options: list[str] = Field(default_factory=list)
    rationale: str = ""
    #: "system" | "llm"
    source: str = "system"


class RequirementConflict(BaseModel):
    requirement_ids: list[str]
    description: str


#: label under which a user correction is appended to the request text an extraction is grounded on
CORRECTION_LABEL = "Correction from user:"


class RequirementSet(BaseModel):
    raw_input: str = ""
    #: later corrections the user gave to the request (their own words; appended by :meth:`request_text`)
    corrections: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    missing: list[MissingInformation] = Field(default_factory=list)
    conflicts: list[RequirementConflict] = Field(default_factory=list)
    #: memo of model extractions keyed by ``sha256:`` of :meth:`request_text`; not design content
    #: (excluded from the design hash), see the module docstring
    extraction_cache: dict[str, Any] = Field(default_factory=dict)

    _design = drop_in_design_view("extraction_cache")


    def request_text(self) -> str:
        """The request as an extraction sees it: ``raw_input`` plus every correction on its own labelled line."""
        if not self.corrections:
            return self.raw_input
        return "\n".join([self.raw_input, "", *(f"{CORRECTION_LABEL} {c}" for c in self.corrections)])

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
