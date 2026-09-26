"""Regulatory research: official texts through the archive, quotes grounded, applicability decided - never compliance.

Invariants this module enforces:

* **Nothing is fetched that the list does not name, and nothing without the
  user's online session.** The only hosts trusted for this stage are the
  candidates' ``allowed_domains`` (registered on the archive's policy with
  the candidate ids as the reason); every fetch goes through
  :meth:`~ai_eda.tools.sources.DocumentArchive.fetch`, which refuses without
  approval. Offline (no archive, or one without an online session) nothing
  is fetched: a hash-verified copy from an earlier run is used when the
  archive has one, otherwise the source is ``NOT_VERIFIED`` with
  ``not fetched (offline)``. A copy the IR recorded that no longer hashes
  to its name is reported as ``tampered`` and never used.
* **A fetched page is the document only when it says so.** Each candidate
  document lists ``expected_markers``; a response that carries none of them
  (a bot wall the archive let through, an API error XML answered with HTTP
  200) is ``wrong_document`` and ``NOT_VERIFIED`` - never grounds a quote
  and never FAILs the list.
* **A quote the official text does not contain is the list's fault.** When
  the expected document was archived and a claimed ``grounding_quote`` is
  not found in it (:meth:`~ai_eda.tools.sources.ArchivedDocument.find_quote`,
  the requirement stage's normalisation), the candidate is ``FAIL``: the
  curated entry is wrong and a human must fix it. ``PASS`` for a source means
  exactly "archived, expected document, every quote found".
* **Applicability is decided by :mod:`ai_eda.regulatory.applicability`**
  from the answers and requirements given, and reported with its inputs,
  its missing keys and the grounding quotes it rests on. The
  ``regulatory.applicability`` result is ``PASS`` only when every candidate
  was decided *and* every quote its decision cites was found in the archived
  text; otherwise ``NOT_VERIFIED`` naming what is missing.
* **``regulatory.compliance`` is always ``NOT_VERIFIED``.** Reading a
  regulation is not compliance; the message says an engineer / notified
  body must assess it. ``regulatory.research`` summarises the three and can
  therefore never be ``PASS``.
* **Every requirement carries the ten provenance fields.** Jurisdiction,
  authority, source title, final URL, retrieval time, section (label + page
  of the first grounded quote), applicability rationale (rule + inputs),
  verification status, archived path and content hash - filled from the
  archived document when there is one, honestly ``None`` when there is not.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_eda.ir.provenance import SourceRef
from ai_eda.ir.regulatory import Applicability, GroundedQuote, RegulatoryProvenance, RegulatoryRequirement, RegulatoryState
from ai_eda.ir.validation import Evidence, ValidationResult, ValidationStatus, worst_status
from ai_eda.regulatory.applicability import APPLICABILITY_VERSION, Evaluation, evaluate
from ai_eda.regulatory.candidates import CandidateDocument, CandidateList, RegulatoryCandidate, find_placeholders
from ai_eda.tools.sources.archive import ArchivedDocument, DocumentArchive, FetchOutcome

RESEARCH_TOOL = "regulatory.research"
RESEARCH_VERSION = "1"
SOURCES_CHECK = "regulatory.sources"
APPLICABILITY_CHECK = "regulatory.applicability"
COMPLIANCE_CHECK = "regulatory.compliance"
RESEARCH_CHECK = "regulatory.research"
COMPLIANCE_MESSAGE = (
    "compliance is not assessed by this system: an engineer and, where the regulation requires it, a notified body / "
    "certification body must assess the design against each applicable regulation; this stage reports source provenance "
    "and applicability only"
)
OFFLINE_REASON = "not fetched (offline)"

_EXPECT = {"pdf": "pdf", "html": "html", "xml": "any", "json": "any", "text": "any"}


class DocumentResult(BaseModel):
    """What happened to one candidate document in this run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    role: str
    #: ok | archived | offline | blocked | missing | refused | error | wrong_document | no_text | unfetchable | unresolved
    status: str
    reason: str | None = None
    final_url: str | None = None
    sha256: str | None = None
    path: str | None = None
    retrieved_at: str | None = None
    title: str | None = None
    page_count: int | None = None
    #: the expected marker that identified the document
    marker: str | None = None
    fetch: dict[str, Any] | None = None
    document: ArchivedDocument | None = Field(default=None, exclude=True)

    @property
    def usable(self) -> bool:
        return self.document is not None and self.status in ("ok", "archived")


class CandidateResult(BaseModel):
    candidate_id: str
    jurisdiction: str
    title: str
    evaluation: Evaluation
    documents: list[DocumentResult]
    quotes: list[GroundedQuote]
    #: the candidate's source status (see :class:`DocumentResult.status`, plus ``quote_missing``)
    source_status: str
    verification: ValidationStatus
    reason: str
    #: whether every quote the applicability decision cites was found
    evidence_grounded: bool
    requirement: RegulatoryRequirement


class ResearchOutcome(BaseModel):
    state: RegulatoryState
    results: list[ValidationResult]
    candidates: list[CandidateResult]
    #: candidate id -> the input keys it still needs
    undecided: dict[str, list[str]]
    notes: list[str]
    online: bool

    def result(self, check_id: str) -> ValidationResult:
        for r in self.results:
            if r.check_id == check_id:
                return r
        raise KeyError(check_id)


# --------------------------------------------------------------------------- documents


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def marker_found(doc: ArchivedDocument, markers: list[str]) -> tuple[bool, str | None]:
    """``(found, marker)``: no markers = accepted; else the first marker present in the title or the text (whitespace-tolerant)."""
    if not markers:
        return True, None
    title = _norm_ws(doc.title or "")
    for m in markers:
        if m and _norm_ws(m) in title:
            return True, m
        if doc.find_quote(m):
            return True, m
    return False, None


def _fill_placeholders(url: str, candidates: CandidateList, env: dict[str, str] | None, notes: list[str]) -> tuple[str, list[str]]:
    """Every declared ``{NAME}`` filled; ``{DATE}`` is left for :func:`_discover_date`. Returns the URL and the placeholders that remain."""
    out = url
    for name in find_placeholders(url):
        if name == "DATE":
            continue
        value, origin = candidates.placeholder_value(name, env)
        out = out.replace("{" + name + "}", value)
        notes.append(f"{{{name}}} = {value!r} ({origin})")
    return out, find_placeholders(out)


def _obtain(archive: DocumentArchive | None, url: str, purpose: str, expect: str, online: bool) -> tuple[ArchivedDocument | None, str, str | None, FetchOutcome | None]:
    """``(document, status, reason, fetch outcome)`` for ``url``: a fresh fetch when online, else / on failure an archived copy."""
    if archive is None:
        return None, "offline", OFFLINE_REASON, None
    outcome: FetchOutcome | None = None
    if online:
        outcome = archive.fetch(url, purpose=purpose, expect=expect)  # type: ignore[arg-type]
        if outcome.ok and outcome.document is not None:
            return outcome.document, "ok", outcome.reason, outcome
    prior = archive.lookup(url)
    if prior is not None:
        if outcome is None:
            return prior, "archived", f"archived copy retrieved {prior.meta.get('retrieved_at')} used; {OFFLINE_REASON}", None
        return prior, "archived", f"fetch {outcome.status} ({outcome.reason}); archived copy retrieved {prior.meta.get('retrieved_at')} used instead", outcome
    if outcome is None:
        return None, "offline", OFFLINE_REASON, None
    return None, outcome.status, outcome.reason or outcome.status, outcome


def _discover_date(doc: CandidateDocument, candidates: CandidateList, archive: DocumentArchive | None, online: bool, owner: str) -> tuple[str | None, DocumentResult]:
    """Fill ``{DATE}`` from the document's ``date_discovery`` index; the index itself is archived as evidence."""
    disc = doc.date_discovery
    assert disc is not None
    notes: list[str] = []
    index_url, _ = _fill_placeholders(disc.url, candidates, None, notes)
    index_doc, status, reason, outcome = _obtain(archive, index_url, f"{owner}: date discovery index", "any", online)
    res = DocumentResult(url=index_url, role="date_discovery", status=status, reason=reason, fetch=outcome.log_record() if outcome else None)
    if index_doc is None:
        return None, res
    res.final_url, res.sha256, res.path = index_doc.final_url or index_doc.url, index_doc.sha256, str(index_doc.path)
    res.retrieved_at, res.title, res.page_count = index_doc.meta.get("retrieved_at"), index_doc.title, index_doc.page_count
    res.document = index_doc
    try:
        data = json.loads(index_doc.pages[0] if index_doc.pages else "")
        items = data[disc.list_key]
        item = next(i for i in items if all(i.get(k) == v for k, v in disc.match.items()))
        value = str(item[disc.field])
    except (ValueError, KeyError, TypeError, StopIteration, IndexError) as e:
        res.status, res.reason = "unresolved", f"date discovery: {index_url} does not carry {disc.list_key}[{disc.match}].{disc.field}: {type(e).__name__}: {e}"
        return None, res
    if not re.match(disc.pattern, value):
        res.status, res.reason = "unresolved", f"date discovery: {disc.field} = {value!r} does not match {disc.pattern}"
        return None, res
    res.reason = f"{{DATE}} = {value} from {disc.list_key}[{disc.match}].{disc.field}"
    return value, res


def obtain_document(doc: CandidateDocument, candidate: RegulatoryCandidate, candidates: CandidateList, archive: DocumentArchive | None,
                    online: bool, env: dict[str, str] | None = None) -> list[DocumentResult]:
    """Resolve, fetch (or look up) and identify one candidate document; the discovery index, when used, comes first."""
    out: list[DocumentResult] = []
    notes: list[str] = []
    url, remaining = _fill_placeholders(doc.url, candidates, env, notes)
    if "DATE" in remaining:
        if doc.date_discovery is None:
            out.append(DocumentResult(url=url, role=doc.role, status="unresolved", reason="{DATE} placeholder without date discovery"))
            return out
        date, disc_res = _discover_date(doc, candidates, archive, online, candidate.id)
        out.append(disc_res)
        if date is None:
            out.append(DocumentResult(url=url, role=doc.role, status="unresolved", reason=f"{{DATE}} could not be discovered: {disc_res.reason}"))
            return out
        url = url.replace("{DATE}", date)
        notes.append(f"{{DATE}} = {date}")
        remaining = find_placeholders(url)
    if remaining:
        out.append(DocumentResult(url=url, role=doc.role, status="unresolved", reason=f"unfilled placeholders {remaining}"))
        return out
    document, status, reason, outcome = _obtain(archive, url, f"{candidate.id}: {doc.role} text", _EXPECT.get(doc.form, "any"), online)
    res = DocumentResult(url=url, role=doc.role, status=status, reason="; ".join([*notes, reason] if reason else notes) or None,
                         fetch=outcome.log_record() if outcome else None)
    if outcome is not None:
        res.final_url = outcome.final_url
    if document is None:
        out.append(res)
        return out
    res.final_url = document.final_url or document.url or url
    res.sha256, res.path, res.retrieved_at = document.sha256, str(document.path), document.meta.get("retrieved_at")
    res.title, res.page_count = document.title, document.page_count
    if not document.text_available:
        res.status = "no_text"
        res.reason = f"archived but no text could be extracted ({document.extraction_error or 'empty document'})"
        res.document = document
        out.append(res)
        return out
    found, marker = marker_found(document, doc.expected_markers)
    if not found:
        res.status = "wrong_document"
        res.reason = (f"the fetched document carries none of the expected markers {doc.expected_markers} (title {document.title!r}, "
                      f"{document.page_count} page(s)): not the expected official text")
        res.document = document
        out.append(res)
        return out
    res.marker = marker
    res.document = document
    out.append(res)
    return out


# --------------------------------------------------------------------------- per candidate


def _ground_quotes(candidate: RegulatoryCandidate, docs: dict[str, DocumentResult]) -> list[GroundedQuote]:
    out: list[GroundedQuote] = []
    for q in candidate.grounding_quotes:
        key = q.url or candidate.official_url
        dres = docs.get(key)
        if dres is None or not dres.usable:
            reason = dres.reason if dres is not None else "document not obtained"
            # no archived document was looked in: the URL a 404 / interstitial came from is this run's story, not a document
            out.append(GroundedQuote(section=q.section, quote=q.quote, found=False, reason=f"{dres.status if dres else 'unresolved'}: {reason}"))
            continue
        assert dres.document is not None
        hits = dres.document.find_quote(q.quote)
        if not hits:
            out.append(GroundedQuote(section=q.section, quote=q.quote, found=False, source_url=dres.final_url, content_hash=dres.sha256,
                                     reason="quote not found in the archived official text (the candidate list is wrong for this entry)"))
            continue
        h = hits[0]
        out.append(GroundedQuote(section=q.section, quote=q.quote, found=True, page=h.page, context=h.context, source_url=dres.final_url, content_hash=dres.sha256))
    return out


def _source_status(candidate: RegulatoryCandidate, docs: list[DocumentResult], quotes: list[GroundedQuote]) -> tuple[str, ValidationStatus, str]:
    if not candidate.fetchable:
        return "unfetchable", ValidationStatus.NOT_VERIFIED, f"official text not fetchable by machine: {candidate.unfetchable_reason}"
    main = [d for d in docs if d.role != "date_discovery"]
    bad = [d for d in main if not d.usable]
    if bad:
        first = bad[0]
        if all(d.status == "offline" for d in bad):
            return "offline", ValidationStatus.NOT_VERIFIED, OFFLINE_REASON
        return first.status, ValidationStatus.NOT_VERIFIED, f"{first.role} document {first.status}: {first.reason}"
    missing = [q for q in quotes if not q.found]
    if missing:
        return "quote_missing", ValidationStatus.FAIL, (f"{len(missing)} of {len(quotes)} claimed quote(s) not found in the archived official text "
                                                       f"({', '.join(q.section for q in missing)}): the candidate list is wrong for {candidate.id}")
    fresh = all(d.status == "ok" for d in main)
    return ("ok" if fresh else "archived"), ValidationStatus.PASS, (
        f"official text archived ({'fetched in this run' if fresh else 'hash-verified copy from an earlier run'}) and all {len(quotes)} quote(s) found")


def research_candidate(candidate: RegulatoryCandidate, candidates: CandidateList, archive: DocumentArchive | None, online: bool,
                       answers: dict[str, str], requirements: Any, env: dict[str, str] | None = None) -> CandidateResult:
    """One candidate: applicability from the inputs, documents through the archive, quotes grounded, requirement built."""
    ev = evaluate(candidate.applicability_rule, answers, requirements)
    doc_results: list[DocumentResult] = []
    by_url: dict[str, DocumentResult] = {}
    if candidate.fetchable:
        for cdoc in candidate.documents():
            got = obtain_document(cdoc, candidate, candidates, archive, online, env)
            doc_results.extend(got)
            by_url[cdoc.url] = got[-1]
    else:
        for cdoc in candidate.documents():
            r = DocumentResult(url=cdoc.url, role=cdoc.role, status="unfetchable", reason=candidate.unfetchable_reason)
            doc_results.append(r)
            by_url[cdoc.url] = r
    quotes = _ground_quotes(candidate, by_url)
    source_status, verification, reason = _source_status(candidate, doc_results, quotes)
    found_sections = {q.section for q in quotes if q.found}
    evidence_grounded = all(label in found_sections for label in ev.evidence)
    official = by_url.get(candidate.official_url)
    # the section the decision cites is the claim; where (page) and whether it was found are this run's verdict (GroundedQuote)
    section = ", ".join(ev.evidence) if ev.evidence else (candidate.grounding_quotes[0].section if candidate.grounding_quotes else None)
    rationale = f"rule: {ev.rationale}"
    if ev.inputs_used:
        rationale += "; inputs: " + ", ".join(f"{k} = {v}" for k, v in ev.inputs_used.items())
    if ev.missing:
        rationale += "; missing: " + ", ".join(f"{m.key} ({m.reason})" for m in ev.missing)
    if ev.evidence:
        rationale += "; evidence: " + ", ".join(ev.evidence)  # grounded or not is GroundedQuote.found / regulatory.sources, not the design
    if candidate.not_evaluated:
        rationale += f"; not evaluated: {candidate.not_evaluated}"
    retrieved = None
    if official is not None and official.document is not None and official.usable:
        retrieved = official.document.retrieved_at
    provenance = RegulatoryProvenance(
        jurisdiction=candidate.jurisdiction,
        authority=candidate.authority,
        source_title=candidate.title + (f" [{official.title}]" if official is not None and official.title else ""),
        source_url=(official.final_url if official is not None and official.usable else (official.url if official is not None else candidate.official_url)),
        retrieved_at=retrieved,
        section=section,
        applicability_rationale=rationale,
        verification_status=verification,
        source_document=official.path if official is not None and official.usable else None,
        content_hash=official.sha256 if official is not None and official.usable else None,
    )
    status = ValidationStatus.FAIL if verification is ValidationStatus.FAIL else ev.status
    requirement = RegulatoryRequirement(
        id=candidate.id, jurisdiction=candidate.jurisdiction, title=candidate.title, summary=candidate.summary,
        engineering_implication=candidate.engineering_implication, status=status, provenance=provenance, candidate_id=candidate.id,
        basis="curated", applicability=ev.applicability, applicability_inputs=dict(ev.inputs_used), missing_inputs=ev.missing_keys,
        grounded_quotes=quotes, source_status=source_status,
    )
    return CandidateResult(candidate_id=candidate.id, jurisdiction=candidate.jurisdiction, title=candidate.title, evaluation=ev, documents=doc_results,
                           quotes=quotes, source_status=source_status, verification=verification, reason=reason, evidence_grounded=evidence_grounded,
                           requirement=requirement)


# --------------------------------------------------------------------------- the stage


def _doc_summary(d: DocumentResult) -> dict[str, Any]:
    return d.model_dump(mode="json", exclude={"fetch"}) | {"fetch_status": (d.fetch or {}).get("status"), "http_status": (d.fetch or {}).get("http_status")}


def research(
    state: RegulatoryState,
    candidates: CandidateList,
    archive: DocumentArchive | None,
    answers: dict[str, str] | None,
    requirements: Any,
    *,
    jurisdictions: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> ResearchOutcome:
    """Research every curated candidate of the known jurisdictions; see the module docstring for what each result means.

    ``archive`` may be ``None`` (offline, nothing archived) or an archive
    with or without an online session. ``answers`` are the scope answers
    (``mains_powered``, ``radio``, ...); ``requirements`` the IR's
    requirement set. The returned state keeps the given state's
    jurisdictions, intended use, scope answers and proposals, and replaces
    the curated requirements of the researched jurisdictions.
    """
    answers = dict(answers or {})
    codes = [c.upper() for c in (jurisdictions if jurisdictions is not None else state.known_jurisdictions)]
    online = archive is not None and archive.online
    notes: list[str] = []
    if online:
        assert archive is not None
        for host, ids in candidates.allowed_hosts(codes).items():
            archive.policy.trust_host(host, f"official domain listed for {ids} in candidates.json ({candidates.sha256})")
    results: list[CandidateResult] = []
    for code in codes:
        cands = candidates.for_jurisdiction(code)
        if not cands:
            notes.append(f"no curated candidates for jurisdiction {code}: nothing researched, nothing decided")
            continue
        for c in cands:
            results.append(research_candidate(c, candidates, archive, online, answers, requirements, env))

    # a copy the IR recorded that no longer hashes to its name is named as tampered; it was never used (lookup() skips it)
    tampered: dict[str, str] = {}
    if archive is not None:
        prior = {r.candidate_id or r.id: r.provenance.content_hash for r in state.requirements if r.provenance.content_hash}
        for r in results:
            h = prior.get(r.candidate_id)
            if not h or archive.verify(SourceRef(title=r.title, content_hash=h)) != "tampered":
                continue
            tampered[r.candidate_id] = h
            if not any(d.usable and d.role != "date_discovery" for d in r.documents):
                r.source_status = r.requirement.source_status = "tampered"
                r.reason += f"; the archived copy {h} recorded in the IR no longer hashes to its name (altered on disk) and was not used"
            notes.append(f"{r.candidate_id}: the archived copy {h} recorded in the IR is tampered (altered on disk); not used")

    # the state: curated requirements of the researched jurisdictions are replaced, everything else is kept
    researched_ids = {r.candidate_id for r in results}
    kept = [r for r in state.requirements if not (r.basis == "curated" and (r.candidate_id in researched_ids or r.jurisdiction in codes))]
    new_state = state.model_copy(deep=True)
    new_state.requirements = kept + [r.requirement for r in results]

    undecided = {r.candidate_id: r.evaluation.missing_keys for r in results if r.evaluation.applicability is Applicability.UNDECIDED}
    missing_keys: list[str] = []
    for keys in undecided.values():
        missing_keys.extend(k for k in keys if k not in missing_keys)
    list_info = candidates.describe()
    common = {"tool": RESEARCH_TOOL, "tool_version": RESEARCH_VERSION}

    # regulatory.sources
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for r in results:
        for d in r.documents:
            if d.usable and d.sha256 and d.sha256 not in seen:
                seen.add(d.sha256)
                evidence.append(Evidence(description=f"{r.candidate_id} {d.role} text ({d.status})", path=d.path, url=d.final_url or d.url, content_hash=d.sha256))
    if not results:
        src_status, src_msg = ValidationStatus.NOT_VERIFIED, f"no curated candidates for {codes or 'the known jurisdictions'}"
    else:
        src_status = worst_status(r.verification for r in results)
        counts = {s: sum(1 for r in results if r.verification is s) for s in (ValidationStatus.PASS, ValidationStatus.NOT_VERIFIED, ValidationStatus.FAIL)}
        src_msg = (f"{len(results)} candidate(s) for {', '.join(codes)}: {counts[ValidationStatus.PASS]} archived and grounded, "
                   f"{counts[ValidationStatus.NOT_VERIFIED]} not verified, {counts[ValidationStatus.FAIL]} with quotes missing from the official text")
        reused = [r.candidate_id for r in results if r.source_status == "archived"]
        if reused:
            src_msg += f"; {len(reused)} used a hash-verified copy from an earlier run ({', '.join(reused)})"
        if not online:
            src_msg += f"; {OFFLINE_REASON}" + (" - pass --online to fetch the official texts" if archive is None or not archive.online else "")
        problems = [f"{r.candidate_id}: {r.reason}" for r in results if r.verification is not ValidationStatus.PASS]
        if problems:
            src_msg += "; " + "; ".join(problems)
    sources = ValidationResult(
        check_id=SOURCES_CHECK, status=src_status, message=src_msg, evidence=evidence, **common,
        details={
            "online": online, "jurisdictions": codes, "candidate_list": list_info, "archive_root": str(archive.root) if archive is not None else None,
            "candidates": [
                {"id": r.candidate_id, "jurisdiction": r.jurisdiction, "source_status": r.source_status, "verification": r.verification, "reason": r.reason,
                 "documents": [_doc_summary(d) for d in r.documents],
                 "quotes": [q.model_dump(mode="json") for q in r.quotes]}
                for r in results
            ],
            "tampered": tampered,
            "notes": list(notes),
        },
    )

    # regulatory.applicability
    ungrounded = [r.candidate_id for r in results if r.evaluation.applicability is not Applicability.UNDECIDED and not r.evidence_grounded]
    if not results:
        app_status, app_msg = ValidationStatus.NOT_VERIFIED, f"nothing to decide: no curated candidates for {codes or 'the known jurisdictions'}"
    elif undecided:
        app_status = ValidationStatus.NOT_VERIFIED
        app_msg = (f"{len(undecided)} of {len(results)} candidate(s) undecided - answer {', '.join(missing_keys)} (--answer key=value): "
                   + "; ".join(f"{cid} needs {', '.join(keys)}" for cid, keys in undecided.items()))
    elif ungrounded:
        app_status = ValidationStatus.NOT_VERIFIED
        app_msg = (f"all {len(results)} candidate(s) decided on the curated rules, but the official sentence(s) the decision cites were not grounded for "
                   f"{', '.join(ungrounded)} ({OFFLINE_REASON if not online else 'see regulatory.sources'})")
    else:
        app_status = ValidationStatus.PASS
        app_msg = (f"all {len(results)} candidate(s) decided from the answers and requirements on the curated inclusion / exclusion rules, "
                   "each decision grounded in the archived official text")
    decided = [f"{r.candidate_id}: {r.evaluation.applicability.value}" for r in results if r.evaluation.applicability is not Applicability.UNDECIDED]
    if decided:
        app_msg += "; " + ", ".join(decided)
    not_evaluated: dict[str, str] = {}
    for r in results:
        cand = candidates.get(r.candidate_id)
        if cand is not None and cand.not_evaluated:
            not_evaluated[r.candidate_id] = cand.not_evaluated
    if not_evaluated:
        app_msg += "; not evaluated by the rules: " + "; ".join(f"{cid}: {note}" for cid, note in not_evaluated.items())
    applicability = ValidationResult(
        check_id=APPLICABILITY_CHECK, status=app_status, message=app_msg, evidence=list(evidence), tool=RESEARCH_TOOL,
        tool_version=f"{RESEARCH_VERSION}/applicability-{APPLICABILITY_VERSION}",
        details={
            "jurisdictions": codes, "candidate_list": list_info, "answers": {k: v for k, v in answers.items()},
            "undecided": undecided, "missing_keys": missing_keys, "not_evaluated": not_evaluated,
            "candidates": [
                {"id": r.candidate_id, "applicability": r.evaluation.applicability, "status": r.requirement.status, "rationale": r.evaluation.rationale,
                 "inputs": r.evaluation.inputs_used, "missing": [m.model_dump() for m in r.evaluation.missing], "evidence": r.evaluation.evidence,
                 "evidence_grounded": r.evidence_grounded}
                for r in results
            ],
        },
    )

    # regulatory.compliance - always NOT_VERIFIED
    compliance = ValidationResult(
        check_id=COMPLIANCE_CHECK, status=ValidationStatus.NOT_VERIFIED, message=COMPLIANCE_MESSAGE, **common,
        details={"applicable": [r.candidate_id for r in results if r.evaluation.applicability is Applicability.APPLICABLE],
                 "not_applicable": [r.candidate_id for r in results if r.evaluation.applicability is Applicability.NOT_APPLICABLE],
                 "undecided": list(undecided)},
    )

    # regulatory.research - the stage's summary, never PASS (compliance never is)
    overall = worst_status([sources.status, applicability.status, compliance.status])
    summary = ValidationResult(
        check_id=RESEARCH_CHECK, status=overall, **common,
        message=f"sources {sources.status}; applicability {applicability.status}; compliance {compliance.status} (never assessed here)",
        details={"sources": sources.status, "applicability": applicability.status, "compliance": compliance.status, "candidates": [r.candidate_id for r in results]},
    )
    return ResearchOutcome(state=new_state, results=[sources, applicability, compliance, summary], candidates=results, undecided=undecided, notes=notes, online=online)


def as_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = [
    "APPLICABILITY_CHECK",
    "COMPLIANCE_CHECK",
    "COMPLIANCE_MESSAGE",
    "OFFLINE_REASON",
    "RESEARCH_CHECK",
    "RESEARCH_TOOL",
    "RESEARCH_VERSION",
    "SOURCES_CHECK",
    "CandidateResult",
    "DocumentResult",
    "ResearchOutcome",
    "marker_found",
    "obtain_document",
    "research",
    "research_candidate",
]
