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

A model question is still only a question: one the model marked required
under a key nothing in the pipeline reads (no template key or alias, no
regulatory scope key, not ``application`` / ``jurisdiction``) is asked as
*optional*, noted - its answer could serve nothing the pipeline checks, so it
must not stop the pipeline (the model never decides what blocks a design).
A required model question under a key a template reads (``tx_power``) stays
required. A typed answer to any question is recorded as before - under a key
no template reads it is a design requirement (category ``electrical``: no
category is guessed from a key's spelling, and nothing the user typed is
silently left unenforced), which the closed world refuses with the exact
answer that leaves it out.

``--answer leave_out=<key>,<key>`` (on every path: checklist, extraction,
model unavailable) leaves those requirements out of the design
(:mod:`ai_eda.agents.leave_out`): they move into ``ir.requirements.left_out``
as the user's recorded decision, an open question under the key is closed
(asked no more, also when no requirement answered it), a later extraction
item under the key is not proposed, a conflict naming a left-out requirement
is dropped, and the confirmation table lists them. The leave-out wins over a
typed answer to the same key in the same run; a typed answer in a later run
brings the key back.

Only the extraction path rewrites ``requirements.missing``. A run without it
(checklist, empty request, model unavailable) brings the recorded questions
up to date instead (:func:`_refreshed_missing`): closed by a leave-out,
answered in this run, or required and already answered - dropped; a model's
required question under a key nothing reads - recorded as optional. So the
missing-information gate, ``RequirementSet.blocking_questions`` (the
reviewer) and ``ai-eda report`` read one list; an IR with nothing stale is
not rewritten (its hash stays).
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.keys import CONTROL_KEYS, LEAVE_OUT_KEY
from ai_eda.agents.leave_out import NEVER_LEFT_OUT, answers_key, apply_leave_out, drop_typed_left_out, same_left_out, split_keys, withdraw_brought_back
from ai_eda.design.base import AMBIGUOUS_KEYS, DESIGN_CATEGORIES, leave_out_answer, template_reads_key
from ai_eda.ir import (
    CircuitIR,
    Jurisdiction,
    LeftOutRequirement,
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

#: the answer keys that steer an agent (this one, the component, regulatory, circuit-design and PCB agents) are never
#: requirements: ``CONTROL_KEYS`` is defined once in :mod:`ai_eda.agents.keys` and re-exported here


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


def _kept_notes(ir: CircuitIR, answers: dict[str, str]) -> list[str]:
    """One note per typed answer that is dropped because its key already holds a typed value (both values named).

    The earlier typed answer is kept (it is the user's own value, not an
    extraction); the user learns that the new one did nothing and how to
    change the requirement instead of looping on the same question.
    """
    out: list[str] = []
    for key, given in answers.items():
        if _answerable(ir, key):
            continue
        existing = ir.requirements.get(key)
        assert existing is not None
        held = existing.value.value if existing.value is not None else None
        out.append(
            f"{key}: answer {given!r} not applied - {existing.id} already holds your earlier answer {held!r}, which is kept; "
            f"to change it edit that requirement in the IR (or leave it out with {leave_out_answer([key])} and answer {key} again in a later run)"
        )
    return out


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


def _protected_keys(scope_keys: frozenset[str]) -> frozenset[str]:
    """Keys a leave-out never takes: the product's application and jurisdiction, the regulatory scope answers and the control keys."""
    return NEVER_LEFT_OUT | scope_keys | CONTROL_KEYS


def _known_question_key(key: str, scope_keys: frozenset[str]) -> bool:
    """Whether an answer under ``key`` is something the pipeline reads: a template key (canonical or alias), a scope key, ``application`` or ``jurisdiction``."""
    return template_reads_key(key) or key in scope_keys or key in NEVER_LEFT_OUT


def blocks_pipeline(q: MissingInformation, ir: CircuitIR, scope_keys: frozenset[str]) -> bool:
    """Whether a recorded question stops the pipeline: required, not under a key the user left out, and - for a model question -
    under a key the pipeline reads (the rule :func:`_model_question` applies when it asks; this one also holds for an IR whose
    questions were recorded before that rule existed)."""
    if not q.required or q.key in ir.requirements.left_out_keys:
        return False
    return not (q.source == "llm" and not _known_question_key(q.key, scope_keys))


def _model_question(q: MissingInformation, scope_keys: frozenset[str]) -> MissingInformation:
    """A model's question as the pipeline asks it: a *required* one under a key nothing reads is asked as optional.

    A model question is still only a question. Under a key no template, scope
    rule or baseline item reads, its answer can serve nothing the pipeline
    checks - recorded, it is a design requirement the closed world refuses -
    so it must not stop the pipeline: the model never decides what blocks a
    design on its own. It stays asked (optional), and ``leave_out=<key>``
    closes it. A required question under a key the pipeline reads (a
    template input such as ``tx_power``) stays required. The caller notes the
    demotion (:func:`_demotion_note`) only where the question is actually
    asked or recorded - never for one already answered or left out.
    """
    if not q.required or _known_question_key(q.key, scope_keys):
        return q
    why = (f"asked as required by the model, but no template reads '{q.key}', so it does not block: an answer becomes a design requirement a template "
           f"must serve (the closed world refuses it otherwise); {leave_out_answer([q.key])} closes the question")
    return q.model_copy(update={"required": False, "rationale": f"{q.rationale} [{why}]" if q.rationale else why})


def _demotion_note(key: str, how: str = "asked") -> str:
    """The note for a model's required question :func:`_model_question` asks (or records) as optional."""
    return f"{key}: the model asked this as required, but no template reads {key}: {how} as optional (answer it, or close it with {leave_out_answer([key])})"


def _refreshed_missing(
    ir: CircuitIR, answer_reqs: list[Requirement], left_keys: set[str], scope_keys: frozenset[str], notes: list[str],
) -> list[MissingInformation]:
    """``requirements.missing`` brought up to date on a run without an extraction (checklist, empty request, model unavailable).

    Only the extraction path rewrites the recorded questions, so without it a
    question an earlier run recorded would stay open for ever and the
    missing-information gate, the reviewer (``blocking_questions``) and
    ``ai-eda report`` would disagree. Dropped: a question under a key the
    user left out of the design (closed), one under a key this run's typed
    answers record, and a *required* one a recorded requirement already
    answers (:func:`~ai_eda.agents.leave_out.answers_key`; it would block
    every later run). A model's required question under a key nothing reads
    is recorded as optional (:func:`_model_question`, noted). Control keys
    (``confirm_requirements`` ...) are never answered here: their questions
    stay. An IR with nothing stale gets the same list back (the caller then
    proposes nothing, so its hash is unchanged).
    """
    typed = {r.key for r in answer_reqs}
    answered = {r.key for r in ir.requirements.requirements if answers_key(r)}
    out: list[MissingInformation] = []
    for q in ir.requirements.missing:
        if q.key in left_keys or q.key in typed or (q.required and q.key in answered and q.key not in CONTROL_KEYS):
            continue
        if q.source == "llm":
            asked = _model_question(q, scope_keys)
            if asked.required != q.required:
                notes.append(_demotion_note(q.key, how="recorded"))
            q = asked
        out.append(q)
    return out


def _missing_proposal(ir: CircuitIR, missing: list[MissingInformation]) -> list[IRProposal]:
    """The proposal that records ``missing`` - none when it equals what is recorded (an IR with nothing stale keeps its bytes and hash)."""
    if [q.model_dump(mode="json") for q in missing] == [q.model_dump(mode="json") for q in ir.requirements.missing]:
        return []
    return [IRProposal(description="record open questions (closed, answered and model questions brought up to date)", target="requirements.missing",
                       operation="set", payload=missing)]


def _left_out_table_lines(left: list[LeftOutRequirement]) -> list[str]:
    """The left-out keys as the extraction's confirmation table lists them (the requirement's id and text, or the closed question)."""
    return [f"{x.key}: {x.requirement.id} ({x.requirement.text})" if x.requirement is not None else f"{x.key}: (no requirement; the open question is closed)" for x in left]


def _unread_keys(grounded: GroundedExtraction, closed: set[str]) -> list[str]:
    """Keys of extracted design-category items that no template reads (and that are not left out yet), for the confirmation table.

    Once such an item is the user's requirement, every template's closed
    world refuses it until it is left out; the table says so before the user
    confirms. A bare ambiguous key (``frequency``) is left to its own question.
    """
    return list(dict.fromkeys(
        r.key for r in grounded.requirements
        if r.category in DESIGN_CATEGORIES and not template_reads_key(r.key) and r.key not in AMBIGUOUS_KEYS and r.key not in closed
    ))


def _left_out_proposals(ir: CircuitIR, requirements: list[Requirement], left_out: list[LeftOutRequirement], *, force_requirements: bool) -> list[IRProposal]:
    """The proposals recording a leave-out: the requirement list (when changed or ``force_requirements``) and ``requirements.left_out`` (when changed)."""
    out: list[IRProposal] = []
    same_reqs = [r.model_dump(mode="json") for r in requirements] == [r.model_dump(mode="json") for r in ir.requirements.requirements]
    if force_requirements or not same_reqs:
        out.append(IRProposal(description="record user answers / decisions", target="requirements.requirements", operation="set", payload=requirements))
    if not same_left_out(left_out, ir.requirements.left_out):
        out.append(IRProposal(
            description=f"record the requirements left out of the design ({', '.join(x.key for x in left_out) or 'none'})",
            target="requirements.left_out", operation="set", payload=left_out,
            rationale="the user's decision (--answer leave_out / a typed answer bringing a key back); design content, listed wherever requirements are",
        ))
    return out


class RequirementAgent(Agent):
    name = "requirement"
    task = TaskKind.REQUIREMENT_ANALYSIS

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        answers = {k: v for k, v in ctx.answers.items() if k not in CONTROL_KEYS}  # control answers are never requirements
        confirm_answer = ctx.answers.get(CONFIRM_KEY)
        accept = set(_split_codes(ctx.answers.get(ACCEPT_KEY, "")))
        reject = set(_split_codes(ctx.answers.get(REJECT_KEY, "")))
        scope_keys = regulatory_scope_keys(ctx)
        # leave_out=<key>,<key>: the requirements under those keys leave the design (moved into requirements.left_out, the
        # user's recorded decision); a typed answer to one of them in the same run is not recorded - the leave-out wins
        leave = split_keys(ctx.answers.get(LEAVE_OUT_KEY))
        notes: list[str] = []
        answers = drop_typed_left_out(answers, leave, _protected_keys(scope_keys), notes)
        # Answers the user gave become explicit requirements with user_requirement provenance (regulatory scope answers as such).
        # A typed answer wins over what a model extracted, assumed or had confirmed for the same key (the item is replaced);
        # an answer the user typed earlier is kept, and a note names both values so the dropped one is not a silent no-op.
        answer_reqs = [_answer_requirement(k, v, scope_keys) for k, v in answers.items() if _answerable(ir, k)]
        notes.extend(_kept_notes(ir, answers))
        answer_jurisdictions = [
            Jurisdiction(code=code, name=code, provided_by_user=True)
            for k, v in answers.items() if k == "jurisdiction" and _answerable(ir, k)  # the same rule as the requirement row
            for code in _split_codes(v)
            if not any(j.code == code for j in ir.regulatory.jurisdictions)
        ]
        if ctx.llm is not None and ir.requirements.raw_input.strip():
            return self._with_llm(ir, ctx.llm, answers, confirm_answer, accept, reject, answer_reqs, answer_jurisdictions, notes, leave, scope_keys)
        proposals: list[IRProposal] = []
        # a jurisdiction recorded on ir.regulatory (an earlier answer, or a confirmed extraction) is answered: do not ask again
        questions = [
            q for q in BASELINE_QUESTIONS
            if q.key not in answers and ir.requirements.get(q.key) is None and not (q.key == "jurisdiction" and ir.regulatory.jurisdictions)
        ]
        # a key-only leave-out may close a baseline question or one an earlier extraction run recorded
        recorded, left_keys = self._checklist_requirements(ir, answer_reqs, leave, [*questions, *ir.requirements.missing], scope_keys, notes)
        proposals.extend(recorded)
        questions = [q for q in questions if q.key not in left_keys]  # a key left out of the design is not asked again
        # the questions an earlier extraction run recorded, brought up to date (only the extraction path rewrites them otherwise)
        proposals.extend(_missing_proposal(ir, _refreshed_missing(ir, answer_reqs, left_keys, scope_keys, notes)))
        for j in answer_jurisdictions:
            proposals.append(IRProposal(description=f"add jurisdiction {j.code}", target="regulatory.jurisdictions", operation="append", payload=j))
        if ctx.llm is None:
            notes.append("no LLM configured: free-text parsing skipped, baseline checklist only")
        else:
            notes.append("empty request: nothing to extract, baseline checklist only")
        return self._result(questions=questions, proposals=proposals, notes=notes)

    @staticmethod
    def _checklist_requirements(
        ir: CircuitIR, answer_reqs: list[Requirement], leave: list[str], questions: list[MissingInformation], scope_keys: frozenset[str], notes: list[str],
    ) -> tuple[list[IRProposal], set[str]]:
        """The proposals that record typed answers without an extraction, with this run's leave-outs and bring-backs applied, and the left-out keys after them.

        Without any leave-out activity (no ``leave_out`` answer, no typed
        answer to a left-out key) the proposals are exactly the ones recorded
        before leave-outs existed (append per answer, or the list set when an
        answer replaces an extraction item).
        """
        typed = {r.key for r in answer_reqs}
        if not leave and not (typed & ir.requirements.left_out_keys):
            return _answer_proposals(ir, answer_reqs, notes), ir.requirements.left_out_keys
        superseded = _superseded_ids(ir, answer_reqs, notes)
        merged = [r for r in ir.requirements.requirements if r.id not in superseded] + answer_reqs
        kept = withdraw_brought_back(ir, typed, notes)
        merged, left = apply_leave_out(ir, merged, kept, leave, questions, _protected_keys(scope_keys), notes)
        return _left_out_proposals(ir, merged, left, force_requirements=False), {x.key for x in left}

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
        notes: list[str],
        leave: list[str] | None = None,
        scope_keys: frozenset[str] = frozenset(),
    ) -> AgentResult:
        proposals: list[IRProposal] = []
        leave = list(leave or [])

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
                return self._unavailable(ir, answers, answer_reqs, answer_jurisdictions, proposals, notes, key, e, leave, scope_keys)
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
        # the model's questions as the pipeline asks them: a required one under a key nothing reads does not block
        model_questions = [_model_question(q, scope_keys) for q in grounded.questions]  # noted below, only where one is asked
        # leave-outs: the keys a typed answer of this run brings back, then this run's leave_out answer; a key with no
        # requirement is left out only under an open question of this stage (baseline, the model's, or recorded)
        protected = _protected_keys(scope_keys)
        kept_left = withdraw_brought_back(ir, {r.key for r in answer_reqs}, notes)
        stage_questions = [*BASELINE_QUESTIONS, *model_questions, *ir.requirements.missing]

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
            merged, left = apply_leave_out(ir, merged, kept_left, leave, stage_questions, protected, notes)
            proposals.extend(_left_out_proposals(ir, merged, left, force_requirements=bool(answer_reqs or decided)))
            jurisdictions = list(ir.regulatory.jurisdictions) + answer_jurisdictions
            if answer_jurisdictions:
                proposals.append(IRProposal(description="add user jurisdictions", target="regulatory.jurisdictions", operation="set", payload=jurisdictions))
            left_ids = {x.requirement.id for x in left if x.requirement is not None}
            conflicts = [c for c in ir.requirements.conflicts if not (set(c.requirement_ids) & left_ids)]
            if len(conflicts) != len(ir.requirements.conflicts):
                notes.append(f"{len(ir.requirements.conflicts) - len(conflicts)} conflict(s) dropped: a requirement they name was left out of the design")
                proposals.append(IRProposal(description="set requirement conflicts", target="requirements.conflicts", operation="set", payload=conflicts))
        else:
            kept = [r for r in existing if not from_extraction(r)]
            removed_ids = {r.id for r in existing if from_extraction(r)}
            taken = {r.key for r in kept} | {r.key for r in answer_reqs}
            left_keys = {x.key for x in kept_left}
            items = list(grounded.requirements)
            app = application_requirement(grounded)
            if app is not None and not any(r.key == "application" for r in items):
                items.append(app)  # an explicit item keyed ``application`` already carries the user's words: one req.application
            new_items: list[Requirement] = []
            for r in items:
                if r.key in taken:
                    notes.append(f"{r.key}: the user's answer takes precedence over the extraction")
                    continue
                if r.key in left_keys:  # the user's leave-out outlives any re-extraction: a model never undoes it
                    notes.append(f"{r.key}: left out of the design by your decision; the extraction's {r.kind} item is not proposed")
                    continue
                new_items.append(r)
            merged = _unique_ids(kept + answer_reqs + new_items, notes)
            if all_accept or all_reject:
                inferred_keys = {r.key for r in merged if is_inferred(r)}
                merged, dnotes = decide_inferred(merged, all_accept, all_reject)
                # stored decisions are re-applied silently; only this run's decisions are reported
                notes.extend(n for n in dnotes if any(n.startswith(f"{k}:") for k in new_decisions))
            merged, left = apply_leave_out(ir, merged, kept_left, leave, stage_questions, protected, notes)
            # a conflict naming a requirement the user left out no longer concerns the design
            left_ids = {x.requirement.id for x in left if x.requirement is not None}
            live = [c for c in grounded.conflicts if not (set(c.requirement_ids) & left_ids)]
            if len(live) != len(grounded.conflicts):
                notes.append(f"{len(grounded.conflicts) - len(live)} extracted conflict(s) dropped: a requirement they name was left out of the design")
            if wants_confirm and not correction_added:
                if called or not presented_before:
                    notes.append(
                        "confirmation ignored: the extraction was (re)generated in this run and has not been shown to you yet; "
                        "review the table and confirm again"
                    )
                elif live:
                    notes.append(f"confirmation refused: {len(live)} conflict(s) must be resolved with a correction first")
                else:
                    confirm_now = True
            if confirm_now:
                merged = upgrade_confirmed(merged)
            proposals.append(IRProposal(
                description=f"{'confirm' if confirm_now else 'propose'} {len(new_items)} extracted requirement(s) ({grounded.model})",
                target="requirements.requirements", operation="set", payload=merged,
                rationale="grounded LLM extraction; llm_generated until the user confirms",
            ))
            proposals.extend(p for p in _left_out_proposals(ir, merged, left, force_requirements=False) if p.target == "requirements.left_out")
            conflicts = [c for c in ir.requirements.conflicts if not (set(c.requirement_ids) & (removed_ids | left_ids))] + live
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
        confirmed = confirmed_before or confirm_now
        applied = {k: v for k, v in new_decisions.items() if k in inferred_keys}
        if applied:
            entry["decisions"] = {**stored, **applied}
            cache_changed = True

        # questions: baseline + the model's, minus what is answered or left out of the design; plus the confirmation until confirmed
        answered = {
            r.key for r in merged
            if r.value is not None and (is_grounded_explicit(r) or r.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT)
        }
        closed = {x.key for x in left}
        questions: list[MissingInformation] = []
        for q in BASELINE_QUESTIONS:
            if q.key in answers or q.key in answered or q.key in closed or (q.key == "jurisdiction" and jurisdictions):
                continue
            questions.append(q)
        for original, q in zip(grounded.questions, model_questions):
            if q.key in answers or q.key in answered or q.key in closed or any(q.key == x.key for x in questions):
                continue
            questions.append(q)
            if original.required and not q.required:
                notes.append(_demotion_note(q.key))
        if not confirmed:
            questions.append(confirmation_question(grounded, left_out=_left_out_table_lines(left), unread=_unread_keys(grounded, closed)))
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
        leave: list[str] | None = None,
        scope_keys: frozenset[str] = frozenset(),
    ) -> AgentResult:
        """The model could not be called: report it honestly and fall back to the deterministic checklist."""
        questions = [
            q for q in BASELINE_QUESTIONS
            if q.key not in answers and ir.requirements.get(q.key) is None and not (q.key == "jurisdiction" and ir.regulatory.jurisdictions)
        ]
        recorded, left_keys = self._checklist_requirements(ir, answer_reqs, list(leave or []), [*questions, *ir.requirements.missing], scope_keys, notes)
        proposals.extend(recorded)
        questions = [q for q in questions if q.key not in left_keys]
        proposals.extend(_missing_proposal(ir, _refreshed_missing(ir, answer_reqs, left_keys, scope_keys, notes)))
        for j in answer_jurisdictions:
            proposals.append(IRProposal(description=f"add jurisdiction {j.code}", target="regulatory.jurisdictions", operation="append", payload=j))
        kind = type(error).__name__
        result = ValidationResult(
            check_id=EXTRACTION_CHECK,
            status=ValidationStatus.NOT_VERIFIED,
            message=f"extraction not run ({kind}): {error}",
            details={"request_hash": key, "error_type": kind, "error": str(error)},
        )
        notes.append(f"LLM extraction unavailable ({kind}); baseline checklist only")
        return self._result(questions=questions, proposals=proposals, validation=[result], notes=notes)
