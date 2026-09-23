"""The curated candidate list: which regulations *might* apply, where their official text lives, and how to decide.

Invariants this module enforces:

* **The list is a proposal, not a fact.** ``candidates.json`` ships with the
  package and its top-level ``provenance`` says so: every entry is
  *unverified until fetched and grounded*. The loader exposes that note, the
  curation date and the file's sha256 so every result built from the list
  can say which list it came from.
* **Only allow-listed hosts.** Every URL an entry names (``official_url``,
  ``extra_documents``, per-quote ``url``) must have its host in the entry's
  own ``allowed_domains`` - the loader rejects the file otherwise. The
  research step trusts exactly those hosts on the network policy
  (:meth:`ai_eda.tools.sources.NetworkPolicy.trust_host`), so nothing the
  list does not name is ever fetched. A ``{PLACEHOLDER}`` may appear in the
  path or query of a URL, never in its host.
* **Rules are declarative.** :class:`ApplicabilityRule` is data
  (``always`` / ``never`` / ``voltage_range`` / ``answer`` / ``all_of`` /
  ``any_of`` / ``not``); :mod:`ai_eda.regulatory.applicability` evaluates it
  deterministically. A rule may cite the grounding quote that states it
  (``evidence`` = the quote's ``section`` label) so the decision can be
  reported with the official sentence it rests on. A ``voltage_range`` rule
  names the requirement keys it reads and the scope answer
  (``scope_answer``) the user gives for the product's highest voltage
  rating; every scope answer a rule reads must be a question the list asks.
  What a rule does *not* evaluate (an exclusion list the entry quotes but
  does not model) is stated in the entry's ``not_evaluated`` and travels
  with every decision.
* **Grounding quotes are verbatim.** Each ``grounding_quotes[].quote`` is a
  phrase that must be found in the fetched official text with the
  requirement stage's normalisation (exact characters, whitespace free,
  token boundaries). A quote that is *not* found means the list is wrong for
  that entry, and the research step reports ``FAIL`` for it.
* **Unfetchable entries stay in the list.** An entry whose official text
  could not be fetched by machine (``fetchable: false`` + reason) is kept so
  the stage reports it as ``NOT_VERIFIED`` with the reason, instead of
  silently omitting a regulation. Such an entry may carry no
  ``official_url`` and no grounding quotes - nothing was fetched that could
  have verified them, and a URL is never guessed. A fetchable entry needs
  both.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_eda.tools.sources.policy import host_key, host_of, normalise_url

CANDIDATES_SCHEMA = 1
#: the file shipped with the package
DEFAULT_CANDIDATES_PATH = Path(__file__).with_name("candidates.json")
#: the phrase the file's provenance note must carry: a curated list proposes, it never asserts
UNVERIFIED_NOTE = "unverified until fetched and grounded"

RuleKind = Literal["always", "never", "voltage_range", "answer", "all_of", "any_of", "not"]
DocumentForm = Literal["html", "xml", "pdf", "json", "text"]

_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_ID_RE = re.compile(r"^reg\.[A-Z]{2,3}\.[A-Za-z0-9_.\-]+$")
_JURISDICTION_RE = re.compile(r"^[A-Z]{2,3}$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicabilityRule(_Strict):
    """A declarative rule; see :mod:`ai_eda.regulatory.applicability` for the evaluation semantics."""

    kind: RuleKind
    #: ``voltage_range``: the requirement key that carries the design's rated voltage (``input_voltage``)
    requirement: str | None = None
    #: ``voltage_range``: further requirement keys whose voltage rates the equipment too (``output_voltage``); read when present, never required
    requirements: list[str] = Field(default_factory=list)
    #: ``voltage_range``: the scope-question key under which the user states the highest voltage rating anywhere in the product;
    #: without that answer the rule can not say NOT_APPLICABLE (an input rail alone proves nothing about the equipment's rating)
    scope_answer: str | None = None
    #: ``voltage_range``: inclusive ``[low, high]`` band in volts for alternating / direct current (``None`` = never in scope)
    ac: list[float] | None = None
    dc: list[float] | None = None
    #: ``answer``: the scope-question key and the value(s) that make the rule hold (``yes`` / ``no`` are canonicalised)
    key: str | None = None
    equals: str | list[str] | None = None
    #: ``all_of`` / ``any_of``
    rules: list[ApplicabilityRule] = Field(default_factory=list)
    #: ``not``
    rule: ApplicabilityRule | None = None
    #: ``section`` label of the grounding quote that states this rule (cited in the rationale, must exist on the candidate)
    evidence: str | None = None
    #: free text shown with the decision (unverified, curated)
    text: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> ApplicabilityRule:
        k = self.kind
        if k == "voltage_range":
            if not self.requirement or not _KEY_RE.match(self.requirement):
                raise ValueError("voltage_range needs a snake_case 'requirement' key")
            for extra in self.requirements:
                if not _KEY_RE.match(extra):
                    raise ValueError(f"voltage_range 'requirements' entries must be snake_case keys, got {extra!r}")
            if self.scope_answer is not None and not _KEY_RE.match(self.scope_answer):
                raise ValueError(f"voltage_range 'scope_answer' must be a snake_case key, got {self.scope_answer!r}")
            if self.ac is None and self.dc is None:
                raise ValueError("voltage_range needs an 'ac' and/or 'dc' band")
            for name, band in (("ac", self.ac), ("dc", self.dc)):
                if band is not None and (len(band) != 2 or band[0] > band[1] or band[0] < 0):
                    raise ValueError(f"{name} band must be [low, high] with 0 <= low <= high, got {band}")
        elif k == "answer":
            if not self.key or not _KEY_RE.match(self.key):
                raise ValueError("answer rule needs a snake_case 'key'")
            if self.equals is None or (isinstance(self.equals, list) and not self.equals):
                raise ValueError("answer rule needs 'equals'")
        elif k in ("all_of", "any_of"):
            if not self.rules:
                raise ValueError(f"{k} needs at least one rule")
        elif k == "not":
            if self.rule is None:
                raise ValueError("not needs a 'rule'")
        for name in ("requirement", "ac", "dc", "key", "equals", "scope_answer"):
            if k not in ("voltage_range", "answer") and getattr(self, name) is not None:
                raise ValueError(f"{name!r} is not a field of a {k} rule")
        if k != "voltage_range" and self.requirements:
            raise ValueError(f"'requirements' is not a field of a {k} rule")
        if k != "not" and self.rule is not None:
            raise ValueError("'rule' is only for a not rule")
        if k not in ("all_of", "any_of") and self.rules:
            raise ValueError("'rules' is only for all_of / any_of")
        return self

    def answer_keys(self) -> list[str]:
        """Every scope-question key the rule (transitively) reads, in order of first appearance."""
        out: list[str] = []
        if self.kind == "answer" and self.key:
            out.append(self.key)
        elif self.kind == "voltage_range":
            out.append("mains_powered")  # AC/DC falls back to this answer when the requirement does not say
            if self.scope_answer:
                out.append(self.scope_answer)
        for r in self.rules:
            out.extend(k for k in r.answer_keys() if k not in out)
        if self.rule is not None:
            out.extend(k for k in self.rule.answer_keys() if k not in out)
        return out

    def requirement_keys(self) -> list[str]:
        out: list[str] = []
        if self.kind == "voltage_range" and self.requirement:
            out.append(self.requirement)
            out.extend(k for k in self.requirements if k not in out)
        for r in self.rules:
            out.extend(k for k in r.requirement_keys() if k not in out)
        if self.rule is not None:
            out.extend(k for k in self.rule.requirement_keys() if k not in out)
        return out

    def evidence_labels(self) -> list[str]:
        out: list[str] = []
        if self.evidence:
            out.append(self.evidence)
        for r in self.rules:
            out.extend(e for e in r.evidence_labels() if e not in out)
        if self.rule is not None:
            out.extend(e for e in self.rule.evidence_labels() if e not in out)
        return out


class GroundingQuote(_Strict):
    section: str
    quote: str
    #: the document the quote lives in; ``None`` = the candidate's ``official_url``
    url: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _non_empty(self) -> GroundingQuote:
        if not self.quote.strip() or not self.section.strip():
            raise ValueError("a grounding quote needs a section label and a non-empty quote")
        return self


class DateDiscovery(_Strict):
    """How to fill a ``{DATE}`` placeholder from an official JSON index (the eCFR ``titles.json``)."""

    url: str
    #: key of the list in the JSON object, and the fields an item must equal to be the one
    list_key: str
    match: dict[str, Any]
    #: the field of that item whose value replaces ``{DATE}``; it must match ``pattern``
    field: str
    pattern: str = r"^\d{4}-\d{2}-\d{2}$"


class CandidateDocument(_Strict):
    url: str
    #: "official" (the entry's main text), "consolidated", "annual_edition", ...
    role: str = "official"
    form: DocumentForm = "html"
    #: any one of these must appear (title or text) in the fetched document, else it is not the expected document
    expected_markers: list[str] = Field(default_factory=list)
    date_discovery: DateDiscovery | None = None
    #: point-in-time identifiers (CELEX dated id, MST + 시행일자, eCFR date) - metadata, unverified
    version: dict[str, str] = Field(default_factory=dict)


class ScopeQuestion(_Strict):
    key: str
    question: str
    options: list[str] = Field(default_factory=list)
    #: empty = every jurisdiction
    jurisdictions: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _key(self) -> ScopeQuestion:
        if not _KEY_RE.match(self.key):
            raise ValueError(f"scope question key {self.key!r} must be snake_case")
        return self


class Placeholder(_Strict):
    """A ``{NAME}`` in a URL path/query filled from an environment variable or a default (never from a model)."""

    env: str | None = None
    default: str
    note: str = ""


class RegulatoryCandidate(_Strict):
    id: str
    jurisdiction: str
    title: str
    authority: str
    #: the official text; ``None`` only for an unfetchable entry (a URL is never guessed)
    official_url: str | None = None
    document_form: DocumentForm = "html"
    expected_markers: list[str] = Field(default_factory=list)
    extra_documents: list[CandidateDocument] = Field(default_factory=list)
    allowed_domains: list[str]
    #: curated prose: what the regulation is about (unverified)
    summary: str = ""
    #: curated prose: what the design must evidence if it applies (unverified)
    engineering_implication: str = ""
    applicability_rule: ApplicabilityRule
    #: curated prose: what the rule does *not* evaluate (exclusions, transitional provisions); carried into every applicability
    #: rationale and the ``regulatory.applicability`` message so a decision never reads as more than it is
    not_evaluated: str | None = None
    #: required for a fetchable entry; an unfetchable one has nothing verified to claim
    grounding_quotes: list[GroundingQuote] = Field(default_factory=list)
    fetchable: bool = True
    unfetchable_reason: str | None = None
    #: point-in-time identifiers of the official document (metadata, unverified)
    version: dict[str, str] = Field(default_factory=dict)
    #: pointers that are *not* fetched (canonical citation URL, explanatory pages) - metadata for humans
    related: list[dict[str, str | None]] = Field(default_factory=list)
    curation_note: str | None = None
    #: optional date discovery for a ``{DATE}`` in ``official_url``
    date_discovery: DateDiscovery | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RegulatoryCandidate:
        if not _ID_RE.match(self.id):
            raise ValueError(f"candidate id {self.id!r} must look like reg.<JUR>.<name>")
        if not _JURISDICTION_RE.match(self.jurisdiction):
            raise ValueError(f"jurisdiction {self.jurisdiction!r} must be an upper-case code")
        if self.id.split(".")[1] != self.jurisdiction:
            raise ValueError(f"candidate {self.id} carries jurisdiction {self.jurisdiction}")
        if not self.allowed_domains:
            raise ValueError(f"{self.id}: allowed_domains is empty")
        if not self.fetchable and not self.unfetchable_reason:
            raise ValueError(f"{self.id}: an unfetchable entry must say why")
        if self.fetchable and not self.official_url:
            raise ValueError(f"{self.id}: a fetchable entry needs an official_url")
        if self.fetchable and not self.grounding_quotes:
            raise ValueError(f"{self.id}: at least one grounding quote is required for a fetchable entry")
        allowed = {host_key(d) for d in self.allowed_domains}
        for url in [*([self.official_url] if self.official_url else []), *(d.url for d in self.extra_documents), *(q.url for q in self.grounding_quotes if q.url),
                    *(d.date_discovery.url for d in self.extra_documents if d.date_discovery)]:
            check_url_host(url, allowed, self.id)
        if self.date_discovery is not None:
            check_url_host(self.date_discovery.url, allowed, self.id)
        doc_urls = {d.url for d in self.documents()}
        for q in self.grounding_quotes:
            if q.url is not None and q.url not in doc_urls:
                raise ValueError(f"{self.id}: quote {q.section!r} names a URL that is not one of the entry's documents: {q.url}")
        sections = {q.section for q in self.grounding_quotes}
        for label in self.applicability_rule.evidence_labels():
            if label not in sections:
                raise ValueError(f"{self.id}: rule evidence {label!r} is not a grounding quote section")
        return self

    def documents(self) -> list[CandidateDocument]:
        """The official document first (built from the entry's own fields; absent when the entry has no URL), then the extra ones."""
        if not self.official_url:
            return list(self.extra_documents)
        official = CandidateDocument(url=self.official_url, role="official", form=self.document_form, expected_markers=list(self.expected_markers),
                                     date_discovery=self.date_discovery, version=dict(self.version))
        return [official, *self.extra_documents]

    def quotes_for(self, url: str) -> list[GroundingQuote]:
        """The grounding quotes that live in the document at ``url`` (``None`` URLs mean the official one)."""
        return [q for q in self.grounding_quotes if (q.url or self.official_url) == url]

    def quote(self, section: str) -> GroundingQuote | None:
        for q in self.grounding_quotes:
            if q.section == section:
                return q
        return None


def check_url_host(url: str, allowed: set[str], owner: str) -> str:
    """The host key of ``url`` when it is https-able and in ``allowed``; ``ValueError`` otherwise (placeholders may sit in path/query only)."""
    if _PLACEHOLDER_RE.search(url.split("://", 1)[-1].split("/", 1)[0]):
        raise ValueError(f"{owner}: a placeholder may not appear in the host of {url}")
    probe = _PLACEHOLDER_RE.sub("x", url)
    try:
        norm, _ = normalise_url(probe)
    except ValueError as e:
        raise ValueError(f"{owner}: unusable URL {url!r}: {e}") from e
    host = host_key(host_of(norm))
    if host not in allowed:
        raise ValueError(f"{owner}: host {host!r} of {url} is not in allowed_domains {sorted(allowed)}")
    return host


class CandidateList(_Strict):
    schema_version: int
    #: must say the entries are unverified until fetched and grounded
    provenance: str
    curated_at: str
    curated_by: str = ""
    placeholders: dict[str, Placeholder] = Field(default_factory=dict)
    scope_questions: list[ScopeQuestion]
    candidates: list[RegulatoryCandidate]
    #: sources the curation looked at and did not include, with the reason (metadata)
    not_included: list[dict[str, str | None]] = Field(default_factory=list)
    #: filled by :func:`load_candidates`: where the list came from and its hash
    source_path: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CandidateList:
        if self.schema_version != CANDIDATES_SCHEMA:
            raise ValueError(f"candidates schema {self.schema_version} is not {CANDIDATES_SCHEMA}")
        if UNVERIFIED_NOTE not in self.provenance:
            raise ValueError(f"provenance must state that entries are {UNVERIFIED_NOTE!r}")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.curated_at):
            raise ValueError("curated_at must be an ISO date (YYYY-MM-DD)")
        ids = [c.id for c in self.candidates]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate candidate ids: {dupes}")
        qkeys = [q.key for q in self.scope_questions]
        qdupes = sorted({k for k in qkeys if qkeys.count(k) > 1})
        if qdupes:
            raise ValueError(f"duplicate scope question keys: {qdupes}")
        known_keys = set(qkeys)
        for c in self.candidates:
            for key in c.applicability_rule.answer_keys():
                if key not in known_keys:
                    raise ValueError(f"{c.id}: rule reads answer {key!r} but no scope question asks it")
            for url in [*([c.official_url] if c.official_url else []), *(d.url for d in c.extra_documents)]:
                for name in _PLACEHOLDER_RE.findall(url):
                    if name != "DATE" and name not in self.placeholders:
                        raise ValueError(f"{c.id}: placeholder {{{name}}} in {url} is not declared")
                    if name == "DATE" and not (c.date_discovery or any(d.date_discovery for d in c.extra_documents if d.url == url)):
                        raise ValueError(f"{c.id}: {{DATE}} in {url} has no date_discovery")
        return self

    # ------------------------------------------------------------ lookups

    def jurisdictions(self) -> list[str]:
        out: list[str] = []
        for c in self.candidates:
            if c.jurisdiction not in out:
                out.append(c.jurisdiction)
        return out

    def for_jurisdiction(self, code: str) -> list[RegulatoryCandidate]:
        return [c for c in self.candidates if c.jurisdiction == code.upper()]

    def get(self, candidate_id: str) -> RegulatoryCandidate | None:
        for c in self.candidates:
            if c.id == candidate_id:
                return c
        return None

    def questions_for(self, codes: list[str]) -> list[ScopeQuestion]:
        """The scope questions that apply to any of ``codes`` (global ones first, in file order)."""
        wanted = {c.upper() for c in codes}
        return [q for q in self.scope_questions if not q.jurisdictions or wanted & set(q.jurisdictions)]

    def allowed_hosts(self, codes: list[str] | None = None) -> dict[str, str]:
        """Host key -> the candidate ids that allow it (for the network policy's trust reasons)."""
        out: dict[str, list[str]] = {}
        for c in self.candidates:
            if codes is not None and c.jurisdiction not in {x.upper() for x in codes}:
                continue
            for d in c.allowed_domains:
                out.setdefault(host_key(d), []).append(c.id)
        return {h: ", ".join(ids) for h, ids in out.items()}

    def placeholder_value(self, name: str, env: dict[str, str] | None = None) -> tuple[str, str]:
        """``(value, where it came from)`` for a declared placeholder: the environment variable when set, else the default."""
        p = self.placeholders[name]
        environ = os.environ if env is None else env
        if p.env and environ.get(p.env):
            return environ[p.env], f"environment variable {p.env}"
        return p.default, "default from candidates.json" + (f" ({p.note})" if p.note else "")

    def describe(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "sha256": self.sha256,
            "provenance": self.provenance,
            "curated_at": self.curated_at,
            "curated_by": self.curated_by,
            "jurisdictions": self.jurisdictions(),
            "candidates": len(self.candidates),
        }


def load_candidates(path: Path | str | None = None) -> CandidateList:
    """Load and validate a candidate list (the packaged one by default); ``ValueError`` names every problem."""
    p = Path(path) if path is not None else DEFAULT_CANDIDATES_PATH
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise ValueError(f"cannot read candidates file {p}: {e}") from e
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"{p} is not a UTF-8 JSON document: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{p}: the top level must be an object")
    data = dict(data)
    data["source_path"] = str(p)
    data["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        return CandidateList.model_validate(data)
    except Exception as e:  # pydantic ValidationError carries every problem; re-raised as the module's contract
        raise ValueError(f"{p} is not a valid candidates file: {e}") from e


def find_placeholders(url: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(url)


__all__ = [
    "CANDIDATES_SCHEMA",
    "DEFAULT_CANDIDATES_PATH",
    "UNVERIFIED_NOTE",
    "ApplicabilityRule",
    "CandidateDocument",
    "CandidateList",
    "DateDiscovery",
    "GroundingQuote",
    "Placeholder",
    "RegulatoryCandidate",
    "ScopeQuestion",
    "check_url_host",
    "find_placeholders",
    "load_candidates",
]
