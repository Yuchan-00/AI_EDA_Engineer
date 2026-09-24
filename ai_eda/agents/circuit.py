"""Circuit Design Agent: starts an empty design from a verified template the user confirms.

Invariants this agent keeps:

* No model, no numbers "from its head". The circuit comes from one of the
  deterministic templates in :mod:`ai_eda.design` selected by *confirmed*
  requirement values (``user_requirement`` / ``authoritative``; a model's
  extraction or assumption is never read). Every number is a registered
  calculator's output (``derived``, re-verified by the CALCULATION stage) or
  a value copied from a requirement; every part comes from the KiCad
  library on disk (``ctx.tools["kicad_library"]``).
* Present, then confirm (ARCHITECTURE invariant 10). The first run applies
  nothing: it asks the one required question :data:`CONFIRM_DESIGN_KEY`
  whose text is a table of the template (id, version), the inputs it read
  (requirement ids and values), the free choices it makes (R2 = 10 kohm, a
  tolerance, the ac grid, the LED's ideal-drop model) and the parts it
  instantiates, together with the template's advisory questions (the
  divider's load question, so a ``0 A`` answer can arrive before the
  build). ``--answer confirm_design=yes`` on a later run applies the plan
  with those choices as the user's values (``user_requirement``, note
  ``design choice confirmed by user; template <id> v<version>: ...``);
  ``no`` applies nothing and says so; anything else is not understood and
  the table is asked again. The key is a control key (``CONTROL_KEYS``),
  never a requirement.
* A confirmation counts only for the table the user saw, checked two ways.
  (a) By the answers of this run: any answer that can change what
  :func:`~ai_eda.design.read_inputs` sees - a non-control answer (a template
  input or any other requirement) or a decision on the requirement
  extraction (``confirm_requirements`` / ``accept_implicit`` /
  ``reject_implicit``, which turn a model's value into the user's within
  the same run) - means the table this run would show was not the one on
  screen: the confirmation is ignored with a note and the table is asked.
  (b) By content: every run that asks the table records ``sha256`` of its
  text under the key in ``ir.requirements.presented`` (bookkeeping outside
  the design hash; a proposal like any other), and ``yes`` applies the plan
  only when the recorded hash equals the hash of the table this run would
  show - a requirement edited in the IR, a changed choice, another library
  or a moved project between the two runs re-asks instead.
* It proposes only into an empty design (no components, nets, topology or
  simulation, and no ``ir.parameters`` key it would overwrite) and never
  emits a ``ValidationResult`` on a run that proposes the design: a
  selection is not evidence, the stage stays NOT_VERIFIED and truth comes
  from ``calc.recompute``, SPICE, ERC and the reviewer.
* On a run that finds a design it proposes no circuit and, for a
  template-made design, reports ``design.inputs_vs_requirements``
  (:func:`~ai_eda.design.checks.check_inputs_vs_requirements`): the copied
  inputs re-read from their requirements - PASS, FAIL ``human`` on a
  mismatch, NOT_VERIFIED when a requirement is gone. The result is left
  unstamped like every agent result (a later stage of the same run may
  still change the IR; RELEASE accepts an unstamped result only from the
  run that produced it). On a divider design it applies the template's
  own rule to a load stated after the build
  (:func:`~ai_eda.design.late_load_changes`): an ``output_current`` that
  reads exactly 0 A is served by VOUT through a ``nets`` proposal, a
  non-zero one is named as unservable. It also names confirmed design
  requirements that nothing serves, so the reviewer's ``requirements not
  traced`` FAIL is explained.
* Refusals are notes and non-required questions under the *real*
  requirement key, never a system-authored key: a request no template
  serves (a divider asked for 2 A, an ``efficiency`` requirement), two
  templates triggered at once, an unreadable or ambiguous value, a missing
  library entry, LED pins not named ``A`` / ``K``. A refusal caused by a
  requirement that already exists does not suggest an ``--answer`` for its
  key (a typed value is kept by the requirement agent): it says the
  requirement must change or another design be chosen. Only a template
  input that no requirement states at all is a required question (the
  LED's missing ``led_forward_current``).
"""

from __future__ import annotations

import hashlib

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.keys import CONTROL_KEYS, REQUIREMENT_DECISION_KEYS
from ai_eda.design import (
    CONFIRM_DESIGN_KEY,
    DESIGN_CATEGORIES,
    IGNORED_KEYS,
    Plan,
    check_inputs_vs_requirements,
    design_from_requirements,
    late_load_changes,
    read_inputs,
    template_keys_text,
)
from ai_eda.ir import CircuitIR, MissingInformation
from ai_eda.llm.extraction import is_confirmation, is_rejection
from ai_eda.llm.router import TaskKind
from ai_eda.tools.kicad.library import KicadLibrary


def table_hash(text: str) -> str:
    """``sha256:`` of a question text as shown: what ``ir.requirements.presented`` records and a confirmation is compared with."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class CircuitDesignAgent(Agent):
    name = "circuit_design"
    task = TaskKind.CIRCUIT_DESIGN

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        answer = ctx.answers.get(CONFIRM_DESIGN_KEY)
        if ir.components or ir.nets or ir.topology is not None or ir.simulation is not None:
            return self._design_present(ir, answer)
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return self._result(notes=["no KiCad library in ctx.tools['kicad_library']: a template cannot resolve its symbols and footprints, nothing proposed"])
        inputs, unusable = read_inputs(ir)
        notes = [f"{k} not usable: {why}" for k, why in unusable.items()]
        confirmed = answer is not None and is_confirmation(answer)
        plan = design_from_requirements(ir, library, inputs, unusable, confirmed=confirmed)
        if plan is None:
            notes.append(f"no template matches the confirmed requirements (templates: {template_keys_text()}); nothing proposed")
            if answer is not None:
                notes.append(f"{CONFIRM_DESIGN_KEY} ignored: no template to confirm")
            return self._result(notes=notes)
        notes.extend(plan.notes)
        if not plan.buildable:
            if answer is not None:
                notes.append(f"{CONFIRM_DESIGN_KEY} ignored: template {plan.template} proposes nothing")
            return self._result(questions=plan.questions, notes=notes)
        clash = sorted(k for k in ir.parameters if any(c.target == f"parameters.{k}" for c in plan.changes))
        if clash:
            notes.append(f"template {plan.template} not proposed: ir.parameters {clash} already exist and the template would overwrite them")
            return self._result(notes=notes)
        table = MissingInformation(
            key=CONFIRM_DESIGN_KEY, question=plan.table(), rationale=f"template {plan.template} v{plan.version}: design choices need the user's confirmation before they enter the IR",
        )
        shown = table_hash(table.question)
        recorded = ir.requirements.presented.get(CONFIRM_DESIGN_KEY)
        if answer is None:
            notes.append(f"template {plan.template} v{plan.version} can be built: {len(plan.changes)} change(s) wait for {CONFIRM_DESIGN_KEY}=yes")
            return self._ask(ir, plan, table, shown, recorded, notes)
        if confirmed:
            given_now = sorted(k for k in ctx.answers if k not in CONTROL_KEYS or k in REQUIREMENT_DECISION_KEYS)
            if given_now:
                notes.append(
                    f"{CONFIRM_DESIGN_KEY} ignored: answer(s) {given_now} were given in this run and can change the inputs a template reads, "
                    f"so the table below has not been shown to you yet; review it and confirm again"
                )
                return self._ask(ir, plan, table, shown, recorded, notes)
            if recorded != shown:
                before = "no table was recorded as shown" if recorded is None else f"the table shown to you before was {recorded[:19]}..."
                notes.append(
                    f"{CONFIRM_DESIGN_KEY} ignored: the table below ({shown[:19]}...) is not the one you confirmed - {before}; "
                    f"an input, a choice, a part or the library changed since; review it and confirm again"
                )
                return self._ask(ir, plan, table, shown, recorded, notes)
            proposals = [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale) for c in plan.changes]
            choices = ", ".join(c.text() for c in plan.choices)
            notes.append(f"template {plan.template} v{plan.version} confirmed by the user: {len(proposals)} proposal(s); design choices recorded as the user's values: {choices}")
            return self._result(proposals=proposals, questions=plan.questions, notes=notes)
        if is_rejection(answer):
            notes.append(f"{CONFIRM_DESIGN_KEY}={answer.strip()!r}: template {plan.template} not applied, the design stays empty (run again without the answer to see the table)")
            return self._result(notes=notes)
        notes.append(f"answer {answer.strip()!r} to {CONFIRM_DESIGN_KEY} not understood: reply yes to apply the template or no to leave the design empty; asking again")
        return self._ask(ir, plan, table, shown, recorded, notes)

    def _ask(self, ir: CircuitIR, plan: Plan, table: MissingInformation, shown: str, recorded: str | None, notes: list[str]) -> AgentResult:
        """The table (with the template's advisory questions) and, when it differs from the last one shown, the record of it."""
        proposals: list[IRProposal] = []
        if recorded != shown:
            proposals.append(IRProposal(
                description=f"record the {CONFIRM_DESIGN_KEY} table shown to the user ({shown[:19]}...)", target="requirements.presented", operation="set",
                payload={**ir.requirements.presented, CONFIRM_DESIGN_KEY: shown},
                rationale="a confirmation counts only for the table the user saw; bookkeeping outside the design hash",
            ))
        return self._result(proposals=proposals, questions=[table, *plan.questions], notes=notes)

    def _design_present(self, ir: CircuitIR, answer: str | None) -> AgentResult:
        what = [f"{len(ir.components)} component(s)", f"{len(ir.nets)} net(s)"]
        if ir.topology is not None:
            what.append("a topology")
        if ir.simulation is not None:
            what.append("a simulation setup")
        notes = [f"design content already present ({', '.join(what)}); templates only start an empty design, nothing proposed"]
        if answer is not None:
            notes.append(f"{CONFIRM_DESIGN_KEY} ignored: design content already present")
        validation = []
        result = check_inputs_vs_requirements(ir)
        if result is not None:
            # unstamped on purpose: a later stage of this run (PLACEMENT, FAB_CAPABILITY) may still change the IR, and
            # RELEASE accepts an unstamped agent result only from the run that produced it (see Orchestrator._release)
            validation.append(result)
            notes.append(f"{result.check_id} {result.status}: {result.message}")
        late = late_load_changes(ir)
        notes.extend(late.notes)
        proposals = [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale) for c in late.changes]
        unserved = [rid for rid in _unserved(ir) if rid not in late.served]
        if unserved:
            notes.append(f"confirmed design requirement(s) no component or net serves: {', '.join(unserved)} (the reviewer reports them as not traced)")
        return self._result(proposals=proposals, validation=validation, notes=notes)


def _unserved(ir: CircuitIR) -> list[str]:
    served = {rid for c in ir.components for rid in c.serves_requirements} | {rid for n in ir.nets for rid in n.serves_requirements}
    return [
        r.id for r in ir.requirements.requirements
        if r.category in DESIGN_CATEGORIES and r.key not in IGNORED_KEYS and r.id not in served
        and not (r.value is not None and r.value.provenance.needs_verification)
    ]


__all__ = ["CONFIRM_DESIGN_KEY", "CircuitDesignAgent", "Plan", "table_hash"]
