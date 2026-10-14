"""Leaving requirements out of the design: ``--answer leave_out=<key>,<key>`` (the requirement agent applies it).

Invariant: every closed-world refusal is leavable by an answer, and the
answer is the user's recorded decision - never a silent drop, never a
template input, never a requirement.

* ``leave_out`` is a control key (:data:`~ai_eda.agents.keys.CONTROL_KEYS`),
  so it never becomes a requirement or a regulatory scope answer, and a
  requirement decision (:data:`~ai_eda.agents.keys.REQUIREMENT_DECISION_KEYS`):
  a ``confirm_design=yes`` given in the same run is ignored and the table -
  which lists the left-out requirements, so its hash changes - is asked
  again, exactly as beside ``accept_implicit``.
* Each named key's requirements are **moved** from
  ``ir.requirements.requirements`` into ``ir.requirements.left_out``
  (:class:`~ai_eda.ir.LeftOutRequirement`, with the question under the key
  when it was still open - no requirement of the key answering it,
  :func:`answers_key`). So no template reads them (:func:`~ai_eda.design.inputs.read_inputs`
  and every categorical reader see only the requirement list), no closed
  world refuses on them and ``review.requirements_vs_ir`` does not demand
  them; the record is design content (hashed once non-empty) and the
  confirmation tables, the stage reports and ``ai-eda report`` list it.
* Only a *design* requirement (a :data:`~ai_eda.design.base.DESIGN_CATEGORIES`
  category) can be left out: a regulatory scope answer, ``application``,
  ``jurisdiction`` and the control keys are refused with a note (they never
  refuse a template). A requirement the design already references (a
  component or net serving it, a parameter copied from it, an expectation
  verifying it - found by its id anywhere in the design view outside the
  requirements) is refused too: leaving it out would untrace the design.
* A key with no requirement but an open question of the requirement stage
  (a baseline or a model question) is recorded without a requirement: the
  question is closed, and a later extraction item under the key is not
  proposed while the record exists (a model can never undo the user's
  decision). Any other key is reported and not recorded.
* Idempotent: naming a key that is already left out changes nothing (noted).
* A typed answer to the key in the **same** run is not recorded - the
  leave-out wins (noted). A typed answer in a **later** run brings the key
  back: the answer is recorded as usual and the record is removed (noted),
  so a key left out by mistake - even one a template needs - is never a
  dead end either.
"""

from __future__ import annotations

from typing import Any

from ai_eda.design.base import DESIGN_CATEGORIES, LEAVE_OUT_KEY, design_references, design_view_outside_requirements, leave_out_answer
from ai_eda.ir import CircuitIR, LeftOutRequirement, MissingInformation, ProvenanceKind, Requirement
from ai_eda.llm.extraction import is_grounded_explicit

#: requirement keys a leave-out never takes: they describe the product or its regulatory scope, never a circuit a template refuses
NEVER_LEFT_OUT: frozenset[str] = frozenset({"application", "jurisdiction"})


def split_keys(answer: str | None) -> list[str]:
    """The keys of a ``leave_out`` answer (comma / semicolon separated, blanks dropped, order kept, repeats dropped)."""
    if not answer:
        return []
    return list(dict.fromkeys(k.strip() for k in answer.replace(";", ",").split(",") if k.strip()))


def left_out_note(key: str) -> str:
    return f"left out of the design by the user ({leave_out_answer([key])})"


def answers_key(r: Requirement) -> bool:
    """Whether ``r`` answers the question under its key: a value that is the user's, or a grounded explicit extraction item.

    The requirement agent's own rule for a question it no longer asks. A
    leave-out attaches the open question under a key to the record only when
    no requirement answers it - a question answered long before is not the
    one the leave-out closed.
    """
    return r.value is not None and (is_grounded_explicit(r) or r.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT)


def withdraw_brought_back(ir: CircuitIR, typed_keys: set[str], notes: list[str]) -> list[LeftOutRequirement]:
    """``ir.requirements.left_out`` without the keys a typed answer of this run brings back (noted once per key)."""
    kept: list[LeftOutRequirement] = []
    noted: set[str] = set()
    for x in ir.requirements.left_out:
        if x.key in typed_keys:
            if x.key not in noted:
                noted.add(x.key)
                was = f"{x.requirement.id} ({x.requirement.text})" if x.requirement is not None else "no requirement, a closed question"
                notes.append(f"{x.key}: your typed answer brings {x.key} back into the design; the earlier leave-out ({was}) is withdrawn")
            continue
        kept.append(x)
    return kept


def drop_typed_left_out(answers: dict[str, str], named: list[str], protected: frozenset[str], notes: list[str]) -> dict[str, str]:
    """``answers`` without the typed answers to keys this run also leaves out: the leave-out wins (noted)."""
    out = dict(answers)
    for key in named:
        if key in out and key not in protected:
            notes.append(
                f"{key}: answer {out.pop(key)!r} not recorded - you also left {key} out in this run ({LEAVE_OUT_KEY} wins); "
                f"answer {key} alone in a later run to bring it back"
            )
    return out


def apply_leave_out(
    ir: CircuitIR,
    requirements: list[Requirement],
    left_out: list[LeftOutRequirement],
    named: list[str],
    questions: list[MissingInformation],
    protected: frozenset[str],
    notes: list[str],
) -> tuple[list[Requirement], list[LeftOutRequirement]]:
    """``(requirements, left_out)`` after this run's leave-outs (``named``) on the list the run would record.

    ``left_out`` is the current record (after :func:`withdraw_brought_back`);
    ``questions`` are the requirement stage's open questions (a key with no
    requirement is left out only when one of them is under it); ``protected``
    are the keys never left out (:data:`NEVER_LEFT_OUT`, the regulatory scope
    keys, the control keys). Every decision - applied, repeated, refused or
    matching nothing - is noted.
    """
    reqs = list(requirements)
    left = list(left_out)
    open_q = {q.key: q for q in questions}
    view: dict[str, Any] | None = None  # the design view, built once and only when a requirement is about to move
    for key in named:
        if key in protected:
            notes.append(f"{key}: not left out - it is not a design requirement key (the product's application, its jurisdiction, a regulatory scope answer "
                         f"or a control key never refuses a template)")
            continue
        matching = [r for r in reqs if r.key == key]
        if not matching:
            if any(x.key == key for x in left):
                notes.append(f"{key}: already left out of the design; nothing changed")
            elif key in open_q:
                left.append(LeftOutRequirement(key=key, question=open_q[key], note=left_out_note(key)))
                notes.append(f"{key}: no requirement under {key}; the open question under it is closed and recorded as left out of the design")
            else:
                notes.append(f"{key}: no requirement and no open question under {key}; nothing left out")
            continue
        for r in matching:
            if r.category not in DESIGN_CATEGORIES:
                notes.append(f"{key}: {r.id} is a {r.category} requirement, not a design requirement - it never refuses a template; not left out")
                continue
            if view is None:
                view = design_view_outside_requirements(ir)
            refs = design_references(ir, r.id, data=view)
            if refs:
                notes.append(f"{key}: {r.id} is referenced by the design ({', '.join(refs)}); leaving it out would untrace the design; not left out")
                continue
            reqs.remove(r)
            # the question under the key is recorded only while it was still open: none of the key's requirements answers it
            question = open_q.get(key) if not any(answers_key(m) for m in matching) else None
            left.append(LeftOutRequirement(key=key, requirement=r, question=question, note=left_out_note(key)))
            notes.append(f"{key}: {r.id} ({r.text}) left out of the design by your decision (recorded in ir.requirements.left_out; "
                         f"a later typed answer to {key} brings it back)")
    return reqs, left


def same_left_out(a: list[LeftOutRequirement], b: list[LeftOutRequirement]) -> bool:
    return [x.model_dump(mode="json") for x in a] == [x.model_dump(mode="json") for x in b]


__all__ = [
    "NEVER_LEFT_OUT",
    "answers_key",
    "apply_leave_out",
    "design_references",
    "drop_typed_left_out",
    "left_out_note",
    "same_left_out",
    "split_keys",
    "withdraw_brought_back",
]
