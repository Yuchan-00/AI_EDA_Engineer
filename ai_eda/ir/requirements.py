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
``RequirementSet.left_out`` records the requirement keys the user left out of
the design with ``--answer leave_out=<key>`` (:class:`LeftOutRequirement`):
the requirement is moved there, so no template, closed world or reviewer sees
it, and the record is design content (left out of the design view only while
empty, so IRs saved before the field existed keep their hashes).
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

from ai_eda.ir.provenance import Traced, drop_empty_in_design_view, drop_in_design_view


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

    ``answer_key`` names the control answer that answers the question with
    the question's ``key`` as its value: ``"leave_out"`` on a closed-world
    refusal (``--answer leave_out=<key>``; a value typed under the refused key
    is the dead end the refusal is about). ``None`` - every other question -
    is answered by a value under ``key``. Added after IRs were saved with
    questions in ``requirements.missing``, so it is left out of the design
    view while ``None`` and every older IR keeps its hash.
    """

    key: str
    question: str
    required: bool = True
    #: the choices we can offer; empty means free-form
    options: list[str] = Field(default_factory=list)
    rationale: str = ""
    #: "system" | "llm"
    source: str = "system"
    #: the control key answered with ``key`` as its value (``leave_out`` for a closed-world refusal); ``None``: answer under ``key``
    answer_key: str | None = None

    _design = drop_empty_in_design_view("answer_key")


class RequirementConflict(BaseModel):
    requirement_ids: list[str]
    description: str


class LeftOutRequirement(BaseModel):
    """A requirement key the user left out of the design (``--answer leave_out=<key>``): the user's decision, recorded.

    ``requirement`` is the requirement as it stood when it was left out. It is
    *moved* out of :attr:`RequirementSet.requirements` into
    :attr:`RequirementSet.left_out`, so no template reads it, no closed world
    refuses on it and the reviewer does not demand it - and it is never
    silently dropped: the record is design content (hashed), and the
    confirmation table, the stage reports and ``ai-eda report`` list it.
    ``requirement`` is ``None`` when the answer closed an open question of the
    requirement stage under ``key`` that no requirement answered yet
    (``question`` is that question as it was asked); beside a requirement,
    ``question`` is the question under the key only while it was still open
    (no requirement of the key answered it). ``note`` says what the
    decision was, in words (no clock: the record is the same whenever it is
    read).
    """

    key: str
    requirement: Requirement | None = None
    question: MissingInformation | None = None
    note: str = ""


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
    #: bookkeeping of what the user has been shown, by answer key: ``sha256:`` of the text of a system question
    #: (a template's ``confirm_design`` table) as it was last asked. A confirmation counts only for the text the
    #: user saw, so the agent that asks records it here and compares before it honours the answer. Not design
    #: content (a table names library paths; the same design shown on two machines must hash the same): excluded
    #: from the design hash like ``extraction_cache``.
    presented: dict[str, str] = Field(default_factory=dict)
    #: the project's model pin: the primary ``provider:model`` spec of the first run that made at least one served
    #: model call (``ai-eda run`` sets it; ``--llm-allow-model-change`` re-pins after a served call). A later run
    #: with another primary spec is refused before any call, so one project's model proposals come from one model
    #: unless the user says otherwise. Bookkeeping of the runs, not design content: excluded from the design hash
    #: like ``extraction_cache`` / ``presented`` (the same design run through another provider hashes the same).
    llm_model_spec: str | None = None
    #: requirement keys the user left out of the design (``--answer leave_out=<key>``, :class:`LeftOutRequirement`): the
    #: user's design decision, so design content - but added after IRs were saved, so it is left out of the design view
    #: while empty and every older IR keeps its hash
    left_out: list[LeftOutRequirement] = Field(default_factory=list)

    _design = drop_in_design_view("extraction_cache", "presented", "llm_model_spec", while_empty=("left_out",))

    @property
    def left_out_keys(self) -> set[str]:
        """The requirement keys the user left out of the design."""
        return {x.key for x in self.left_out}


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
        """The required open questions; one under a key the user left out of the design is closed (:attr:`left_out`)."""
        left = self.left_out_keys
        return [m for m in self.missing if m.required and m.key not in left]

    @property
    def can_proceed(self) -> bool:
        return not self.blocking_questions and not self.conflicts
