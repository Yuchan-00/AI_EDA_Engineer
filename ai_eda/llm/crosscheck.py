"""Cross-check ("complement") of two or more models' grounded proposals - deterministic, field by field.

With ``--llm-crosscheck SPEC`` every structured task is also sent, with the
same prompt and schema, to the cross-check model(s)
(:meth:`~ai_eda.llm.service.LLMService.cross_structured`). Each reply is a
proposal and goes through the task's own grounding first; what survives is
compared here.

Invariants enforced here:

* **No model judges another model's answer.** The comparison is code: field
  by field, with the normalisation the grounding itself uses - numbers agree
  within the grounding's relative tolerance (``REL_TOL``, the rule
  :func:`~ai_eda.llm.extraction._same_value` applies to duplicates), text
  compares with whitespace removed and case folded (``_squash``), kinds,
  statuses and provenance kinds must match (``_same_kind``), and the
  category the model assigned must match. Nothing is widened for the
  cross-check: two values the grounding would keep apart differ here too.
* **Agreement changes nothing but the record.** An agreed item enters the IR
  exactly as without the cross-check; its provenance note names the
  cross-check model(s) that agreed (:func:`agreement_note`).
* **Disagreement chooses nothing.** A field the models proposed differently
  - a value, a kind, a category, an item only one of them proposed, or a
  conflict between requirements that not every model declared (compared as
  the set of keys it names) - is withheld from the proposals
  (:func:`withhold`) and becomes a question (:func:`disagreement_question`)
  that shows every model's grounded proposal and which model made it; the
  user answers it with their own value. The models' own questions are prose
  and are not compared but joined: a question any model asked is asked
  (``extra_questions``, its rationale naming the model).
* A cross-check model that failed is reported by the caller (its error is in
  the :class:`~ai_eda.llm.service.CrossCheckReply`); it is never compared and
  never counted as agreement.

The requirement extraction (:func:`compare_extractions`) is the task this
module compares; see the requirement agent for how its result is applied.
The other structured tasks (part candidates, datasheet facts, regulation
proposals) are not cross-checked yet; each says so in its stage notes when
cross-check models are configured and it calls its model
(:func:`not_crosschecked_note`), so the gap is never silent. Their replies are
still grounded and confirmed by the user row by row, as before.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_eda.ir import MissingInformation, Requirement
from ai_eda.llm.extraction import GroundedExtraction, _describe, _same_kind, _same_value, _squash

#: what a model proposed for a field it did not propose at all
NOT_PROPOSED = "(not proposed)"
#: the keys under which the extraction's application statement and jurisdictions are compared (and asked)
APPLICATION_KEY = "application"
JURISDICTION_KEY = "jurisdiction"
#: the marker an agreed item's provenance note carries
AGREEMENT_MARKER = "cross-checked: agreed by "


class FieldProposal(BaseModel):
    #: the model that proposed it (the served spec)
    model: str
    #: the grounded proposal as shown to the user (:data:`NOT_PROPOSED` when the model proposed nothing under the key)
    value: str


class Disagreement(BaseModel):
    """One field the models proposed differently: the question key and every model's grounded proposal (the primary's first)."""

    key: str
    proposals: list[FieldProposal] = Field(default_factory=list)


class ExtractionCrossCheck(BaseModel):
    """The comparison of a primary grounded extraction with the grounded extractions of the cross-check models."""

    primary_model: str
    #: the cross-check models compared (only those whose reply was served and validated)
    models: list[str] = Field(default_factory=list)
    #: every key compared (requirement keys, then ``application`` / ``jurisdiction`` when any model stated one), in order
    compared: list[str] = Field(default_factory=list)
    agreed: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    #: the grounded questions a cross-check model asked under a key the primary did not ask (each asked too: a question is never
    #: dropped because one model did not think of it), their rationale naming the model
    extra_questions: list[MissingInformation] = Field(default_factory=list)

    @property
    def differing_keys(self) -> list[str]:
        return [d.key for d in self.disagreements]


def _items_agree(a: list[Requirement], b: list[Requirement]) -> bool:
    """Two models' items under one key agree: as many items, pairwise the same kind/status/provenance kind, value and category."""
    return len(a) == len(b) and all(_same_kind(x, y) and _same_value(x, y) and x.category == y.category for x, y in zip(a, b))


def _show_items(items: list[Requirement]) -> str:
    return " | ".join(f"{_describe(r)} ({r.kind}/{r.status}, {r.category})" for r in items) if items else NOT_PROPOSED


def _by_key(g: GroundedExtraction) -> dict[str, list[Requirement]]:
    out: dict[str, list[Requirement]] = {}
    for r in g.requirements:
        out.setdefault(r.key, []).append(r)
    return out


def _application(g: GroundedExtraction) -> str | None:
    return _squash(g.application) if g.application else None


def _conflict_sets(g: GroundedExtraction) -> set[frozenset[str]]:
    """The model-declared conflicts of ``g`` as sets of requirement keys (their description is prose and is not compared)."""
    key_of = {r.id: r.key for r in g.requirements}
    return {frozenset(key_of.get(i, i) for i in c.requirement_ids) for c in g.conflicts}


def _show_conflicts(sets: set[frozenset[str]], key: str) -> str:
    mine = sorted(" <-> ".join(sorted(s)) for s in sets if key in s)
    return f"conflict {'; '.join(mine)}" if mine else "no conflict"


def compare_extractions(primary: GroundedExtraction, others: list[GroundedExtraction]) -> ExtractionCrossCheck:
    """Compare ``primary`` with each of ``others`` (all grounded on the same request text), key by key (see the module docstring).

    Requirement items are grouped by key in the order grounding produced
    them; the application statement is compared under ``application``
    (squashed text) and the jurisdictions under ``jurisdiction`` (the set of
    codes). A conflict the grounding kept is compared as the set of keys it
    names: one that not every model declared puts each of its keys in
    dispute. A key differs when any cross-check model differs from the
    primary on it; the proposals then list every model's value. The
    questions are not compared but joined: a question a cross-check model
    asked under a key the primary did not ask is returned in
    ``extra_questions`` (asking more decides nothing for the user).
    """
    groups = [_by_key(primary), *(_by_key(o) for o in others)]
    extractions = [primary, *others]
    keys: list[str] = []
    for g in groups:
        for k in g:
            if k not in keys:
                keys.append(k)
    result = ExtractionCrossCheck(primary_model=primary.model, models=[o.model for o in others])
    shown: dict[str, list[str]] = {}
    conflicts = [_conflict_sets(e) for e in extractions]
    disputed = {k for s in set().union(*conflicts) - set.intersection(*conflicts) for k in s}
    for key in keys:
        result.compared.append(key)
        if key not in disputed and all(_items_agree(groups[0].get(key, []), g.get(key, [])) for g in groups[1:]):
            continue
        shown[key] = [_show_items(g.get(key, [])) + (f"; {_show_conflicts(c, key)}" if key in disputed else "") for g, c in zip(groups, conflicts)]
    asked = {q.key for q in primary.questions}
    for o in others:
        for q in o.questions:
            if q.key in asked:
                continue
            asked.add(q.key)
            why = f"asked by {o.model}, not by {primary.model} (cross-check)"
            result.extra_questions.append(q.model_copy(update={"rationale": f"{q.rationale} [{why}]" if q.rationale else why}))
    if any(e.application for e in extractions):
        if APPLICATION_KEY not in result.compared:
            result.compared.append(APPLICATION_KEY)
        if any(_application(e) != _application(primary) for e in others):
            summaries = [e.application or "(not stated)" for e in extractions]
            prior = shown.get(APPLICATION_KEY)
            shown[APPLICATION_KEY] = [f"{p}; summary: {s}" for p, s in zip(prior, summaries)] if prior else summaries
    if any(e.jurisdictions for e in extractions):
        if JURISDICTION_KEY not in result.compared:
            result.compared.append(JURISDICTION_KEY)
        if any(sorted(set(e.jurisdictions)) != sorted(set(primary.jurisdictions)) for e in others):
            codes = [", ".join(sorted(set(e.jurisdictions))) or "(none)" for e in extractions]
            prior = shown.get(JURISDICTION_KEY)
            shown[JURISDICTION_KEY] = [f"{p}; codes: {c}" for p, c in zip(prior, codes)] if prior else codes
    for key in result.compared:
        if key in shown:
            result.disagreements.append(Disagreement(
                key=key, proposals=[FieldProposal(model=e.model, value=v) for e, v in zip(extractions, shown[key])],
            ))
        else:
            result.agreed.append(key)
    return result


def withhold(grounded: GroundedExtraction, keys: set[str]) -> GroundedExtraction:
    """``grounded`` without the fields under ``keys``: their requirement items, a conflict naming one of them, their request
    contexts, and - under ``application`` / ``jurisdiction`` - the application statement / the jurisdictions. Nothing else changes."""
    if not keys:
        return grounded
    dropped_ids = {r.id for r in grounded.requirements if r.key in keys}
    return grounded.model_copy(update={
        "requirements": [r for r in grounded.requirements if r.key not in keys],
        "conflicts": [c for c in grounded.conflicts if not (set(c.requirement_ids) & dropped_ids)],
        "contexts": {k: v for k, v in grounded.contexts.items() if k not in dropped_ids and not (APPLICATION_KEY in keys and k == f"req.{APPLICATION_KEY}")},
        "application": None if APPLICATION_KEY in keys else grounded.application,
        "application_quote": None if APPLICATION_KEY in keys else grounded.application_quote,
        "jurisdictions": [] if JURISDICTION_KEY in keys else list(grounded.jurisdictions),
    })


def agreement_note(models: list[str]) -> str:
    """The suffix an agreed item's provenance note carries: which cross-check model(s) proposed the same."""
    return f"{AGREEMENT_MARKER}{', '.join(models)}"


def mark_agreed(requirements: list[Requirement], agreed: set[str], models: list[str]) -> list[Requirement]:
    """``requirements`` with ``; cross-checked: agreed by <models>`` appended to the provenance note of every item under an agreed key."""
    if not agreed or not models:
        return requirements
    suffix = agreement_note(models)
    out: list[Requirement] = []
    for r in requirements:
        if r.key not in agreed or r.value is None or suffix in (r.value.provenance.note or ""):
            out.append(r)
            continue
        note = r.value.provenance.note
        prov = r.value.provenance.model_copy(update={"note": f"{note}; {suffix}" if note else suffix})
        out.append(r.model_copy(update={"value": r.value.model_copy(update={"provenance": prov})}))
    return out


def not_crosschecked_note(what: str, model: str) -> str:
    """The note of a task the cross-check does not cover yet, when cross-check models are configured: never a silent gap."""
    return f"cross-check (--llm-crosscheck) covers the requirement extraction only: these {what} from {model} were not cross-checked"


def disagreement_question(d: Disagreement, *, leave_out: str | None = None) -> MissingInformation:
    """The required question for one disagreement: every model's grounded proposal, which model made it, and how to answer.

    ``leave_out`` is the answer that leaves the key out of the design, when
    the key may be left out (the caller knows which keys may not).
    """
    lines = [f"The models' grounded proposals for {d.key!r} differ, so neither entered the design (a model never decides between models):"]
    lines += [f"  - {p.model}: {p.value}" for p in d.proposals]
    how = f"Answer with your own value: --answer {d.key}=<value>"
    if leave_out:
        how += f", or leave it out of the design: --answer {leave_out}"
    lines += ["", how + "."]
    return MissingInformation(
        key=d.key,
        question="\n".join(lines),
        required=True,
        rationale="cross-check (--llm-crosscheck): the models disagree on this field; only the user decides",
    )


__all__ = [
    "AGREEMENT_MARKER",
    "APPLICATION_KEY",
    "JURISDICTION_KEY",
    "NOT_PROPOSED",
    "Disagreement",
    "ExtractionCrossCheck",
    "FieldProposal",
    "agreement_note",
    "compare_extractions",
    "disagreement_question",
    "mark_agreed",
    "not_crosschecked_note",
    "withhold",
]
