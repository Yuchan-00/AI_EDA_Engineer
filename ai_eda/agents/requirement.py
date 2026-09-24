"""Requirement Agent.

Turns ``ir.requirements.raw_input`` into structured requirements and a list
of questions.

Without an LLM (``ctx.llm is None``) it applies a deterministic checklist:
the :data:`BASELINE_QUESTIONS` not yet answered are asked, and every answer
the user gave becomes an explicit requirement with ``user_requirement``
provenance (``jurisdiction`` answers also become
:class:`~ai_eda.ir.Jurisdiction` entries). That path is unchanged by the LLM
work below.

With an LLM it is a proposal pipeline with a deterministic gate at every
step - the model never decides what enters the IR:

1. The request text is ``raw_input`` plus the user's corrections
   (:meth:`~ai_eda.ir.RequirementSet.request_text`). An answer to
   ``confirm_requirements`` that describes what is wrong *is* a correction
   (:func:`~ai_eda.llm.extraction.is_correction`): it is appended and the
   extraction runs again on the longer text. A bare ``no`` or a reply too
   short to say anything is not: the agent asks again without paying for a
   re-extraction.
2. The extraction is cached in ``ir.requirements.extraction_cache`` under
   ``sha256`` of that text; an unchanged request costs no call. An entry
   also records the :func:`~ai_eda.llm.extraction.extraction_fingerprint`
   (extraction version, prompt hash, schema hash); an entry made under other
   rules, or one whose stored reply no longer validates, is a *miss* (noted),
   never a replay and never a crash. The model sees the request text only -
   the user's ``--answer`` values are applied deterministically (they become
   ``user_requirement`` entries and silence the matching questions) and are
   not part of the prompt, so the cache is a pure function of the request.
3. :meth:`~ai_eda.llm.service.LLMService.structured` returns a
   :class:`~ai_eda.llm.extraction.RequirementExtraction` (strict schema).
4. :func:`~ai_eda.llm.extraction.ground_extraction` checks every claim:
   explicit items need a verbatim quote at a token boundary and a number
   the deterministic quantity parser re-reads from the request at that
   place, or they are demoted to assumptions; directive phrases are dropped;
   keys are canonicalised.
5. Proposals replace the extraction-derived requirements (a typed user
   answer is kept and wins on a key clash), set the open questions and the
   conflicts, and add grounded jurisdictions as *unconfirmed*
   (``provided_by_user=False``, which does not count as known). One required
   question, ``confirm_requirements``, lists everything as a table with each
   quote in its request context; the cache entry is marked ``presented``.
6. On ``confirm_requirements`` = yes / y / ok / 네 / 확인 ... the grounded
   explicit items become ``user_requirement`` (the note keeps the model and
   the quote), the jurisdictions become user-provided, the question is
   cleared and the cache entry is marked confirmed. A confirmation counts
   only for an extraction that an *earlier* run presented (cache hit with
   ``presented``): one given in the run that (re)generates the extraction is
   ignored with a note - the user must confirm what they saw. It is refused
   while conflicts exist. Implicit items stay ``llm_generated`` and
   assumptions stay ``assumption`` - the user confirmed what they *said*.
   They are decided one by one with ``--answer accept_implicit=k1,k2`` /
   ``--answer reject_implicit=k3`` (:func:`~ai_eda.llm.extraction.decide_inferred`):
   accepted items become ``user_requirement`` (note keeps model + rationale),
   rejected ones are removed; decisions are stored in the cache entry so a
   rebuild from the same extraction applies them again. Until decided, the
   ``ir.llm_requirements`` validator blocks IR_BUILD on them and the reviewer
   neither enforces nor counts them.
7. The stage reports a ``requirements.extraction`` result (``tool`` = the
   model asked, ``tool_version`` = the model that answered; counts, demoted,
   dropped, cost over every billed attempt): ``USER_INPUT_REQUIRED`` until
   confirmed, ``PASS`` after, and ``NOT_VERIFIED`` with the reason when the
   model could not be called (budget, transport, schema) - the agent then
   falls back to the checklist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.component import CONFIRM_FACTS_KEY, CONFIRM_PARTS_KEY, EXTRACT_FACTS_KEY, FACTS_FILE_KEY
from ai_eda.agents.pcb import PLACEMENT_KEY
from ai_eda.agents.regulatory import ACCEPT_REGS_KEY, PROPOSE_REGS_KEY, REJECT_REGS_KEY
from ai_eda.ir import (
    CircuitIR,
    Jurisdiction,
    MissingInformation,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    RequirementSet,
    RequirementStatus,
    ValidationResult,
    ValidationStatus,
    user_requirement,
)
from ai_eda.llm.client import LLMError
from ai_eda.llm.extraction import (
    ACCEPT_KEY,
    CONFIRM_KEY,
    MIN_CORRECTION_CHARS,
    REJECT_KEY,
    GroundedExtraction,
    RequirementExtraction,
    application_requirement,
    build_extraction_messages,
    cache_entry_staleness,
    confirmation_question,
    decide_inferred,
    extraction_fingerprint,
    from_extraction,
    ground_extraction,
    is_confirmation,
    is_correction,
    is_grounded_explicit,
    is_inferred,
    json_schema,
    request_hash,
    upgrade_confirmed,
)
from ai_eda.llm.router import TaskKind
from ai_eda.llm.service import BudgetExceededError, LLMService, StructuredOutputError
from ai_eda.regulatory.candidates import CandidateList, load_candidates

#: Baseline information every design needs before we may proceed.
BASELINE_QUESTIONS: list[MissingInformation] = [
    MissingInformation(key="application", question="What is the intended application / use of this circuit?", rationale="drives implicit requirements and regulatory scope"),
    MissingInformation(key="jurisdiction", question="Which markets / jurisdictions will the product be used or sold in?", rationale="regulatory research must not guess jurisdiction"),
    MissingInformation(key="operating_temperature", question="What is the operating temperature range?", required=False),
    MissingInformation(key="protection", question="Which protections are required (reverse polarity, OVP, OCP, ESD, ...)?", required=False),
]

EXTRACTION_CHECK = "requirements.extraction"

#: answer keys that steer an agent (this one, the component, regulatory and PCB agents) and are never requirements themselves
CONTROL_KEYS: frozenset[str] = frozenset({
    CONFIRM_KEY, ACCEPT_KEY, REJECT_KEY, CONFIRM_PARTS_KEY, CONFIRM_FACTS_KEY, FACTS_FILE_KEY, EXTRACT_FACTS_KEY, ACCEPT_REGS_KEY, REJECT_REGS_KEY, PROPOSE_REGS_KEY,
    PLACEMENT_KEY,
})


def _split_codes(answer: str) -> list[str]:
    return [c.strip() for c in answer.replace(";", ",").split(",") if c.strip()]


def regulatory_scope_keys(ctx: AgentContext) -> frozenset[str]:
    """The regulatory stage's scope-question keys (``mains_powered``, ``radio``, ...) from the candidate list the run uses.

    An answer to one of them is the user's regulatory scope statement, not an
    electrical requirement a component could serve; it is recorded with
    category ``regulatory`` so the reviewer does not demand a component for it.
    """
    given = ctx.tools.get("regulatory_candidates")
    try:
        if isinstance(given, CandidateList):
            return frozenset(q.key for q in given.scope_questions)
        if isinstance(given, (str, Path)):
            return frozenset(q.key for q in load_candidates(given).scope_questions)
        return _packaged_scope_keys()
    except ValueError:
        return _packaged_scope_keys()


@lru_cache(maxsize=1)
def _packaged_scope_keys() -> frozenset[str]:
    return frozenset(q.key for q in load_candidates().scope_questions)


def _answer_requirement(key: str, answer: str, scope_keys: frozenset[str] = frozenset()) -> Requirement:
    if key == "jurisdiction" or key in scope_keys:
        category = "regulatory"
    elif key == "application":
        category = "application"
    else:
        category = "electrical"
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {answer}", kind=RequirementKind.EXPLICIT, value=user_requirement(answer), category=category)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _answerable(ir: CircuitIR, key: str) -> bool:
    """Whether a typed answer for ``key`` enters the IR: no requirement yet, or one the extraction produced (a typed answer wins)."""
    existing = ir.requirements.get(key)
    return existing is None or from_extraction(existing)


def _superseded_ids(ir: CircuitIR, answer_reqs: list[Requirement], notes: list[str]) -> set[str]:
    """Ids of the extraction-derived requirements a typed answer replaces (noted)."""
    keys = {a.key for a in answer_reqs}
    out = {r.id for r in ir.requirements.requirements if r.key in keys}
    for r in ir.requirements.requirements:
        if r.id in out:
            notes.append(f"{r.key}: the user's answer takes precedence over the extraction ({r.id} replaced)")
    return out


def _answer_proposals(ir: CircuitIR, answer_reqs: list[Requirement], notes: list[str]) -> list[IRProposal]:
    """Proposals that record typed answers: appended when new, or the list rewritten when one replaces an extraction item."""
    if not answer_reqs:
        return []
    superseded = _superseded_ids(ir, answer_reqs, notes)
    if superseded:
        merged = [r for r in ir.requirements.requirements if r.id not in superseded] + answer_reqs
        return [IRProposal(description="record user answers (replacing extraction items of the same key)", target="requirements.requirements", operation="set", payload=merged)]
    return [IRProposal(description=f"record user answer for {req.key}", target="requirements.requirements", operation="append", payload=req) for req in answer_reqs]


def _unique_ids(items: list[Requirement], notes: list[str]) -> list[Requirement]:
    """``items`` with a duplicate id dropped (the first kept, noted): ids are how conflicts, reviewers and confirmations refer to them."""
    seen: set[str] = set()
    out: list[Requirement] = []
    for r in items:
        if r.id in seen:
            notes.append(f"{r.id}: a second requirement with this id was dropped (the first kept)")
            continue
        seen.add(r.id)
        out.append(r)
    return out


class RequirementAgent(Agent):
    name = "requirement"
    task = TaskKind.REQUIREMENT_ANALYSIS

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        answers = {k: v for k, v in ctx.answers.items() if k not in CONTROL_KEYS}  # control answers are never requirements
        confirm_answer = ctx.answers.get(CONFIRM_KEY)
        accept = set(_split_codes(ctx.answers.get(ACCEPT_KEY, "")))
        reject = set(_split_codes(ctx.answers.get(REJECT_KEY, "")))
        # Answers the user gave become explicit requirements with user_requirement provenance (regulatory scope answers as such).
        # A typed answer wins over what a model extracted, assumed or had confirmed for the same key (the item is replaced);
        # an answer the user typed earlier is kept.
        scope_keys = regulatory_scope_keys(ctx)
        answer_reqs = [_answer_requirement(k, v, scope_keys) for k, v in answers.items() if _answerable(ir, k)]
        answer_jurisdictions = [
            Jurisdiction(code=code, name=code, provided_by_user=True)
            for k, v in answers.items() if k == "jurisdiction" and _answerable(ir, k)  # the same rule as the requirement row
            for code in _split_codes(v)
            if not any(j.code == code for j in ir.regulatory.jurisdictions)
        ]
        if ctx.llm is not None and ir.requirements.raw_input.strip():
            return self._with_llm(ir, ctx.llm, answers, confirm_answer, accept, reject, answer_reqs, answer_jurisdictions)
        proposals: list[IRProposal] = []
        notes: list[str] = []
        proposals.extend(_answer_proposals(ir, answer_reqs, notes))
        for j in answer_jurisdictions:
            proposals.append(IRProposal(description=f"add jurisdiction {j.code}", target="regulatory.jurisdictions", operation="append", payload=j))
        # a jurisdiction recorded on ir.regulatory (an earlier answer, or a confirmed extraction) is answered: do not ask again
        questions = [
            q for q in BASELINE_QUESTIONS
            if q.key not in answers and ir.requirements.get(q.key) is None and not (q.key == "jurisdiction" and ir.regulatory.jurisdictions)
        ]
        if ctx.llm is None:
            notes.append("no LLM configured: free-text parsing skipped, baseline checklist only")
        else:
            notes.append("empty request: nothing to extract, baseline checklist only")
        return self._result(questions=questions, proposals=proposals, notes=notes)

    # ------------------------------------------------------------------ LLM path

    def _with_llm(
        self,
        ir: CircuitIR,
        llm: LLMService,
        answers: dict[str, str],
        confirm_answer: str | None,
        accept: set[str],
        reject: set[str],
        answer_reqs: list[Requirement],
        answer_jurisdictions: list[Jurisdiction],
    ) -> AgentResult:
        proposals: list[IRProposal] = []
        notes: list[str] = []

        # 1. corrections: an answer to the confirmation question that says what is wrong extends the request
        corrections = list(ir.requirements.corrections)
        correction_added = False
        wants_confirm = confirm_answer is not None and is_confirmation(confirm_answer)
        if confirm_answer is not None and not wants_confirm:
            if is_correction(confirm_answer):
                text = confirm_answer.strip()
                if text not in corrections:
                    corrections.append(text)
                    correction_added = True
                    proposals.append(IRProposal(description="append user correction to the request", target="requirements.corrections", operation="set", payload=corrections))
                    notes.append("correction appended; extraction re-run on the corrected request")
                else:
                    notes.append(f"correction {text!r} is already part of the request: reply yes to confirm the table, or describe a new change")
            else:
                notes.append(
                    f"answer {confirm_answer.strip()!r} to {CONFIRM_KEY} not understood: reply yes to confirm, or describe what is wrong "
                    f"(at least {MIN_CORRECTION_CHARS} characters); asking again without a new extraction"
                )
        request_text = RequirementSet(raw_input=ir.requirements.raw_input, corrections=corrections).request_text()
        key = request_hash(request_text)

        # 2. cache: an unchanged request text under unchanged rules never costs a second call
        cache = dict(ir.requirements.extraction_cache)
        entry: dict[str, Any] | None = cache.get(key)
        cache_changed = False
        called = False
        if entry is not None:
            stale = cache_entry_staleness(entry)
            if stale is not None:
                notes.append(stale)
                entry = None
        if entry is None:
            try:
                entry = self._extract(llm, request_text)
            except (LLMError, BudgetExceededError, StructuredOutputError) as e:
                return self._unavailable(ir, answers, answer_reqs, answer_jurisdictions, proposals, notes, key, e)
            called = True
            cache_changed = True
        else:
            entry = dict(entry)

        # 3./4. deterministic grounding (re-run on a cache hit: it is pure and cheap)
        extraction = RequirementExtraction.model_validate(entry["extraction"])
        grounded = ground_extraction(request_text, extraction, entry["model"])
        notes.extend(grounded.notes)
        confirmed_before = bool(entry.get("confirmed"))
        presented_before = bool(entry.get("presented"))
        confirm_now = False
        if wants_confirm and not confirmed_before and not correction_added:
            if called or not presented_before:
                notes.append(
                    "confirmation ignored: the extraction was (re)generated in this run and has not been shown to you yet; "
                    "review the table and confirm again"
                )
            elif grounded.conflicts:
                notes.append(f"confirmation refused: {len(grounded.conflicts)} conflict(s) must be resolved with a correction first")
            else:
                confirm_now = True
        confirmed = confirmed_before or confirm_now

        # per-item decisions on inferred items: this run's decisions are applied now and remembered in the entry
        # (only those that named an undecided inferred item - a key that matches nothing is reported, never
        # remembered); remembered ones are re-applied whenever the IR is rebuilt from the cached extraction
        stored = {k: v for k, v in (entry.get("decisions") or {}).items() if v in ("accept", "reject")}
        new_decisions = {**{k: "accept" for k in accept}, **{k: "reject" for k in reject}}
        all_accept = {k for k, v in stored.items() if v == "accept"} | accept
        all_reject = {k for k, v in stored.items() if v == "reject"} | reject
        inferred_keys: set[str] = set()

        # 5./6. proposals
        existing = list(ir.requirements.requirements)
        if confirmed_before:
            superseded = _superseded_ids(ir, answer_reqs, notes)
            merged = [r for r in existing if r.id not in superseded] + answer_reqs
            decided = False
            if new_decisions:
                inferred_keys = {r.key for r in merged if is_inferred(r)}
                merged, dnotes = decide_inferred(merged, set(accept), set(reject))
                notes.extend(dnotes)
                decided = True
            if answer_reqs or decided:
                proposals.append(IRProposal(description="record user answers / decisions", target="requirements.requirements", operation="set", payload=merged))
            jurisdictions = list(ir.regulatory.jurisdictions) + answer_jurisdictions
            if answer_jurisdictions:
                proposals.append(IRProposal(description="add user jurisdictions", target="regulatory.jurisdictions", operation="set", payload=jurisdictions))
            conflicts = list(ir.requirements.conflicts)
        else:
            kept = [r for r in existing if not from_extraction(r)]
            removed_ids = {r.id for r in existing if from_extraction(r)}
            taken = {r.key for r in kept} | {r.key for r in answer_reqs}
            items = list(grounded.requirements)
            app = application_requirement(grounded)
            if app is not None and not any(r.key == "application" for r in items):
                items.append(app)  # an explicit item keyed ``application`` already carries the user's words: one req.application
            new_items: list[Requirement] = []
            for r in items:
                if r.key in taken:
                    notes.append(f"{r.key}: the user's answer takes precedence over the extraction")
                    continue
                new_items.append(r)
            merged = _unique_ids(kept + answer_reqs + new_items, notes)
            if all_accept or all_reject:
                inferred_keys = {r.key for r in merged if is_inferred(r)}
                merged, dnotes = decide_inferred(merged, all_accept, all_reject)
                # stored decisions are re-applied silently; only this run's decisions are reported
                notes.extend(n for n in dnotes if any(n.startswith(f"{k}:") for k in new_decisions))
            if confirm_now:
                merged = upgrade_confirmed(merged)
            proposals.append(IRProposal(
                description=f"{'confirm' if confirm_now else 'propose'} {len(new_items)} extracted requirement(s) ({grounded.model})",
                target="requirements.requirements", operation="set", payload=merged,
                rationale="grounded LLM extraction; llm_generated until the user confirms",
            ))
            conflicts = [c for c in ir.requirements.conflicts if not (set(c.requirement_ids) & removed_ids)] + grounded.conflicts
            proposals.append(IRProposal(description="set requirement conflicts", target="requirements.conflicts", operation="set", payload=conflicts))
            user_j = [j for j in ir.regulatory.jurisdictions if j.provided_by_user] + answer_jurisdictions
            known_codes = {j.code for j in user_j}
            extracted_j = [Jurisdiction(code=c, name=c, provided_by_user=confirm_now) for c in grounded.jurisdictions if c not in known_codes]
            jurisdictions = user_j + extracted_j
            if [j.model_dump() for j in jurisdictions] != [j.model_dump() for j in ir.regulatory.jurisdictions]:
                proposals.append(IRProposal(
                    description=f"{'confirm' if confirm_now else 'propose'} jurisdictions {[j.code for j in extracted_j]} from the request",
                    target="regulatory.jurisdictions", operation="set", payload=jurisdictions,
                    rationale="grounded on a verbatim quote; unconfirmed jurisdictions do not count as known",
                ))
            if confirm_now:
                entry["confirmed"] = True
                entry["confirmed_at"] = _now()
                cache_changed = True
        applied = {k: v for k, v in new_decisions.items() if k in inferred_keys}
        if applied:
            entry["decisions"] = {**stored, **applied}
            cache_changed = True

        # questions: baseline + the model's, minus what is answered; plus the confirmation until confirmed
        answered = {
            r.key for r in merged
            if r.value is not None and (is_grounded_explicit(r) or r.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT)
        }
        questions: list[MissingInformation] = []
        for q in BASELINE_QUESTIONS:
            if q.key in answers or q.key in answered or (q.key == "jurisdiction" and jurisdictions):
                continue
            questions.append(q)
        for q in grounded.questions:
            if q.key in answers or q.key in answered or any(q.key == x.key for x in questions):
                continue
            questions.append(q)
        if not confirmed:
            questions.append(confirmation_question(grounded))
            if not presented_before:
                # the table is in front of the user from this run on: only a later run may confirm it
                entry["presented"] = True
                entry["presented_at"] = _now()
                cache_changed = True
        proposals.append(IRProposal(description="record open questions", target="requirements.missing", operation="set", payload=questions))
        if cache_changed:
            cache[key] = entry
            proposals.append(IRProposal(description="cache the extraction for this request text", target="requirements.extraction_cache", operation="set", payload=cache))

        # 7. the stage's own result: what was extracted, at what cost, and whether the user has confirmed it
        result = self._extraction_result(entry, grounded, key, cached=not called, confirmed=confirmed)
        notes.insert(0, result.message)
        return self._result(questions=questions, proposals=proposals, validation=[result], notes=notes)

    @staticmethod
    def _extract(llm: LLMService, request_text: str) -> dict[str, Any]:
        """One structured extraction call; the returned dict is the cache entry (model output kept verbatim).

        The prompt carries the request text only (no user answers), so the
        entry is a function of the text and the fingerprint. The cost sums
        every attempt the service recorded a usage row for - served replies
        and failures the provider may have billed - and is unknown as soon
        as one of them is.
        """
        messages = build_extraction_messages(request_text, None, include_schema=False)
        extraction, resp = llm.structured(TaskKind.REQUIREMENT_ANALYSIS, messages, RequirementExtraction, json_schema=json_schema())
        billed = [a for a in llm.last_attempts if a.usage is not None]
        costs = [a.cost_usd for a in billed]
        cost = None if not costs or any(c is None for c in costs) else float(sum(c for c in costs if c is not None))
        return {
            **extraction_fingerprint(),
            "request_hash": request_hash(request_text),
            "requested_model": resp.model,
            "model": resp.model_used or resp.model,
            "extraction": extraction.model_dump(mode="json"),
            "usage": resp.usage.model_dump(mode="json"),
            "cost_usd": cost,
            "billed_attempts": len(billed),
            "attempts": [a.model_dump(mode="json") for a in llm.last_attempts],
            "finish_reason": resp.finish_reason,
            "response_id": resp.id,
            "provider": resp.provider_name,
            "created_at": _now(),
            "presented": False,
            "confirmed": False,
            "decisions": {},
        }

    @staticmethod
    def _extraction_result(entry: dict[str, Any], grounded: GroundedExtraction, key: str, *, cached: bool, confirmed: bool) -> ValidationResult:
        reqs = grounded.requirements
        counts = {
            "explicit_grounded": sum(1 for r in reqs if is_grounded_explicit(r)),
            "implicit": sum(1 for r in reqs if r.kind == RequirementKind.IMPLICIT),
            "assumptions": sum(1 for r in reqs if r.kind == RequirementKind.ASSUMPTION),
            "conflicting": sum(1 for r in reqs if r.status == RequirementStatus.CONFLICTING),
            "questions": len(grounded.questions),
            "conflicts": len(grounded.conflicts),
            "jurisdictions": len(grounded.jurisdictions),
            "application": grounded.application is not None,
            "demoted": len(grounded.demoted),
            "dropped": len(grounded.dropped),
        }
        cost = entry.get("cost_usd")
        usage = entry.get("usage") or {}
        message = (
            f"{counts['explicit_grounded']} explicit grounded, {counts['implicit']} implicit, {counts['assumptions']} assumption(s), "
            f"{counts['questions']} question(s), {counts['conflicts']} conflict(s); {counts['demoted']} demoted, {counts['dropped']} dropped; "
            f"model {entry.get('model')}{' (cached)' if cached else ''}; cost {'unknown' if cost is None else f'{cost:.6f} USD'}; "
            f"{'confirmed by user' if confirmed else 'awaiting user confirmation'}"
        )
        return ValidationResult(
            check_id=EXTRACTION_CHECK,
            status=ValidationStatus.PASS if confirmed else ValidationStatus.USER_INPUT_REQUIRED,
            message=message,
            tool=str(entry.get("requested_model") or entry.get("model")),
            tool_version=str(entry.get("model")),
            details={
                "request_hash": key,
                "model": entry.get("model"),
                "requested_model": entry.get("requested_model"),
                "extraction_version": entry.get("extraction_version"),
                "prompt_hash": entry.get("prompt_hash"),
                "schema_hash": entry.get("schema_hash"),
                "cached": cached,
                "presented": bool(entry.get("presented")),
                "confirmed": confirmed,
                "decisions": dict(entry.get("decisions") or {}),
                "counts": counts,
                "demoted": [{"key": k, "reason": why} for k, why in grounded.demoted],
                "dropped": [{"key": k, "reason": why} for k, why in grounded.dropped],
                "notes": list(grounded.notes),
                "cost_usd": cost,
                "cost_known": cost is not None,
                "billed_attempts": entry.get("billed_attempts"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "attempts": entry.get("attempts", []),
            },
        )

    def _unavailable(
        self,
        ir: CircuitIR,
        answers: dict[str, str],
        answer_reqs: list[Requirement],
        answer_jurisdictions: list[Jurisdiction],
        proposals: list[IRProposal],
        notes: list[str],
        key: str,
        error: Exception,
    ) -> AgentResult:
        """The model could not be called: report it honestly and fall back to the deterministic checklist."""
        proposals.extend(_answer_proposals(ir, answer_reqs, notes))
        for j in answer_jurisdictions:
            proposals.append(IRProposal(description=f"add jurisdiction {j.code}", target="regulatory.jurisdictions", operation="append", payload=j))
        questions = [
            q for q in BASELINE_QUESTIONS
            if q.key not in answers and ir.requirements.get(q.key) is None and not (q.key == "jurisdiction" and ir.regulatory.jurisdictions)
        ]
        kind = type(error).__name__
        result = ValidationResult(
            check_id=EXTRACTION_CHECK,
            status=ValidationStatus.NOT_VERIFIED,
            message=f"extraction not run ({kind}): {error}",
            details={"request_hash": key, "error_type": kind, "error": str(error)},
        )
        notes.append(f"LLM extraction unavailable ({kind}); baseline checklist only")
        return self._result(questions=questions, proposals=proposals, validation=[result], notes=notes)
