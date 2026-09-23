"""Regulatory model.

Three rules from the spec are encoded here:

1. Jurisdiction is never guessed. ``RegulatoryState.jurisdictions`` empty
   means the regulatory stage returns USER_INPUT_REQUIRED.
2. Having read an official document is not the same as being compliant.
   ``RegulatoryProvenance.verification_status`` describes what *we* checked
   (``PASS`` means only "the official text was archived and every claimed
   quote was found in it"); it never asserts legal certification, and no
   field anywhere in this module carries a compliance verdict.
3. Applicability is a deterministic decision from the user's answers and the
   IR's requirements against a declarative rule (:mod:`ai_eda.regulatory.applicability`);
   :class:`RegulatoryRequirement.applicability` records that decision and
   :class:`RegulatoryRequirement.status` mirrors it as a validation status:
   ``NOT_APPLICABLE`` when the rule excludes the design, ``NOT_VERIFIED``
   when the regulation applies (its compliance is not verified by this
   system), ``USER_INPUT_REQUIRED`` while an input the rule needs is
   missing, ``FAIL`` when the curated entry could not be trusted (a claimed
   quote is not in the official text).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import drop_in_design_view
from ai_eda.ir.validation import ValidationStatus


class Jurisdiction(BaseModel):
    code: str  # "EU", "US", "KR", "JP", ...
    name: str
    #: how we know this applies: the user typed it, or confirmed an extraction that quoted it. ``False`` only
    #: for a grounded LLM extraction the user has not confirmed yet - such an entry is not "known"
    provided_by_user: bool = True


class Applicability(StrEnum):
    """Whether a regulation applies to the design, as decided by its declarative rule (never by a model)."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    #: an input the rule needs (a scope answer, a requirement value) is missing
    UNDECIDED = "undecided"


class RegulatoryProvenance(BaseModel):
    jurisdiction: str
    authority: str  # e.g. "European Commission", "FCC", "KC/RRA"
    source_title: str
    source_url: str | None = None
    retrieved_at: datetime | None = None
    section: str | None = None
    applicability_rationale: str = ""
    #: what *we* verified about the source: PASS = archived and every claimed quote found; never a compliance verdict
    verification_status: ValidationStatus = ValidationStatus.NOT_VERIFIED
    source_document: str | None = None  # archived local path
    content_hash: str | None = None

    # the verdict and the local path are state about the design (what a run found where), not the design
    _design = drop_in_design_view("verification_status", "source_document")


class GroundedQuote(BaseModel):
    """One phrase the curated list claims the official text contains, and whether the archived text does."""

    section: str
    quote: str
    found: bool = False
    #: 1-based page of the archived document the quote was found on
    page: int | None = None
    #: the document's text around the hit, the match in brackets
    context: str | None = None
    #: final URL and hash of the archived document the quote was looked for in
    source_url: str | None = None
    content_hash: str | None = None

    #: why it was not found or not looked for (offline, blocked, wrong document, ...)
    reason: str | None = None

    # the claim (section, quote, which document) is design provenance; whether, where and why-not it was found is a
    # verdict of one run (an offline run says "offline: ...", the next online run finds it) - not the design
    _design = drop_in_design_view("found", "page", "context", "reason")


class RegulatoryRequirement(BaseModel):
    id: str  # "reg.EU.LVD.2014-35-EU"
    jurisdiction: str
    title: str
    summary: str = ""
    #: what the design must satisfy, in engineering terms, if we could extract it
    engineering_implication: str = ""
    status: ValidationStatus = ValidationStatus.NOT_VERIFIED
    provenance: RegulatoryProvenance
    #: id of the curated candidate (or model proposal) this entry was built from
    candidate_id: str | None = None
    #: "curated" (shipped candidate list) or "llm_proposed" (a model's proposal the user accepted)
    basis: str = "curated"
    applicability: Applicability = Applicability.UNDECIDED
    #: the inputs the rule read, ``key -> "value (where it came from)"``
    applicability_inputs: dict[str, str] = Field(default_factory=dict)
    #: input keys the rule still needs (``USER_INPUT_REQUIRED`` names them)
    missing_inputs: list[str] = Field(default_factory=list)
    grounded_quotes: list[GroundedQuote] = Field(default_factory=list)
    #: what happened to the official source: ok | archived | offline | blocked | missing | refused | error |
    #: wrong_document | no_text | unfetchable | unresolved | quote_missing | tampered (the copy the IR recorded was altered on disk)
    source_status: str | None = None

    # ``status`` / ``source_status`` are what a run found (they flip between an offline and an online run), not the design
    _design = drop_in_design_view("status", "source_status")


class ProposedRegulation(BaseModel):
    """A regulation a model proposed for the jurisdiction / application - never authoritative by itself.

    It enters the IR only as a record of what was proposed, screened and
    shown; it becomes a :class:`RegulatoryRequirement` only after the user
    accepted it *and* its official document (from an allow-listed host) was
    archived and found to contain ``title_quote``.
    """

    id: str
    jurisdiction: str
    title: str
    authority: str
    official_url: str
    summary: str = ""
    #: a short phrase the model claims appears verbatim in the official document (grounded before acceptance)
    title_quote: str = ""
    model: str
    #: hash of the inputs the proposal was made for (jurisdictions + application), so an unchanged input costs no second call
    request_hash: str
    #: whether the proposal was shown to the user in an earlier run (only then may an acceptance count)
    presented: bool = False
    #: "accepted" | "rejected" | None; set only from the user's answer
    decision: str | None = None
    #: why the proposal can never be used (host not allow-listed, directive phrase, ...) - shown, never fetched
    refused: str | None = None
    #: set once the accepted proposal's document was archived and the title quote grounded
    grounded: bool = False


class RegulatoryState(BaseModel):
    jurisdictions: list[Jurisdiction] = Field(default_factory=list)
    intended_use: str | None = None
    requirements: list[RegulatoryRequirement] = Field(default_factory=list)
    #: the scope answers the regulatory stage used (``mains_powered``, ``radio``, ...): the user's own words, kept so a
    #: later offline run decides the same way without re-asking
    scope_answers: dict[str, str] = Field(default_factory=dict)
    #: model proposals with their screening / presentation / decision state (see :class:`ProposedRegulation`)
    proposed_candidates: list[ProposedRegulation] = Field(default_factory=list)

    @property
    def jurisdiction_known(self) -> bool:
        """At least one jurisdiction the user supplied or confirmed; an unconfirmed extraction does not count."""
        return any(j.provided_by_user for j in self.jurisdictions)

    @property
    def known_jurisdictions(self) -> list[str]:
        """Codes of the jurisdictions that count as known, in the order given."""
        return [j.code for j in self.jurisdictions if j.provided_by_user]
