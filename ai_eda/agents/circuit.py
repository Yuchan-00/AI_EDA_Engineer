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
  instantiates. ``--answer confirm_design=yes`` on a later run applies the
  plan with those choices as the user's values (``user_requirement``, note
  ``design choice confirmed by user; template <id> v<version>: ...``);
  ``no`` applies nothing and says so; anything else is not understood and
  the table is asked again. A confirmation given in the run that first
  supplies one of the template's inputs is ignored (the table has not been
  shown yet) and the table is asked. The key is a control key
  (``CONTROL_KEYS``), never a requirement.
* It proposes only into an empty design (no components, nets, topology or
  simulation, and no ``ir.parameters`` key it would overwrite) and never
  emits a ``ValidationResult`` on a run that proposes: a selection is not
  evidence, the stage stays NOT_VERIFIED and truth comes from
  ``calc.recompute``, SPICE, ERC and the reviewer.
* On a run that finds a design it proposes nothing and, for a template-made
  design, reports ``design.inputs_vs_requirements``
  (:func:`~ai_eda.design.checks.check_inputs_vs_requirements`, stamped with
  the IR hash): the copied inputs re-read from their requirements - PASS,
  FAIL ``human`` on a mismatch, NOT_VERIFIED when a requirement is gone. It
  also names confirmed design requirements that nothing serves, so the
  reviewer's ``requirements not traced`` FAIL is explained.
* Refusals are notes and non-required questions under the *real*
  requirement key, never a system-authored key: a request no template
  serves (a divider asked for 2 A, an ``efficiency`` requirement), two
  templates triggered at once, an unreadable or ambiguous value, a missing
  library entry, LED pins not named ``A`` / ``K``. Only a template input
  that no requirement states at all is a required question (the LED's
  missing ``led_forward_current``).
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.design import (
    CONFIRM_DESIGN_KEY,
    DESIGN_CATEGORIES,
    IGNORED_KEYS,
    KEY_ALIASES,
    Plan,
    check_inputs_vs_requirements,
    design_from_requirements,
    read_inputs,
    template_keys_text,
)
from ai_eda.ir import CircuitIR, MissingInformation
from ai_eda.llm.extraction import is_confirmation, is_rejection
from ai_eda.llm.router import TaskKind
from ai_eda.tools.kicad.library import KicadLibrary


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
        if answer is None:
            notes.append(f"template {plan.template} v{plan.version} can be built: {len(plan.changes)} change(s) wait for {CONFIRM_DESIGN_KEY}=yes")
            return self._result(questions=[table], notes=notes)
        if confirmed:
            given_now = sorted(k for k in plan.inputs if any(a in ctx.answers for a in _aliases(k)))
            if given_now:
                notes.append(
                    f"{CONFIRM_DESIGN_KEY} ignored: input(s) {given_now} were given in this run, so the table below has not been shown to you yet; "
                    f"review it and confirm again"
                )
                return self._result(questions=[table], notes=notes)
            proposals = [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale) for c in plan.changes]
            choices = ", ".join(c.text() for c in plan.choices)
            notes.append(f"template {plan.template} v{plan.version} confirmed by the user: {len(proposals)} proposal(s); design choices recorded as the user's values: {choices}")
            return self._result(proposals=proposals, questions=plan.questions, notes=notes)
        if is_rejection(answer):
            notes.append(f"{CONFIRM_DESIGN_KEY}={answer.strip()!r}: template {plan.template} not applied, the design stays empty (run again without the answer to see the table)")
            return self._result(notes=notes)
        notes.append(f"answer {answer.strip()!r} to {CONFIRM_DESIGN_KEY} not understood: reply yes to apply the template or no to leave the design empty; asking again")
        return self._result(questions=[table], notes=notes)

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
            result.ir_hash = ir.content_hash()  # about the IR as it stands: this run proposes nothing
            validation.append(result)
            notes.append(f"{result.check_id} {result.status}: {result.message}")
        unserved = _unserved(ir)
        if unserved:
            notes.append(f"confirmed design requirement(s) no component or net serves: {', '.join(unserved)} (the reviewer reports them as not traced)")
        return self._result(validation=validation, notes=notes)


def _aliases(key: str) -> tuple[str, ...]:
    return KEY_ALIASES.get(key, (key,))


def _unserved(ir: CircuitIR) -> list[str]:
    served = {rid for c in ir.components for rid in c.serves_requirements} | {rid for n in ir.nets for rid in n.serves_requirements}
    return [
        r.id for r in ir.requirements.requirements
        if r.category in DESIGN_CATEGORIES and r.key not in IGNORED_KEYS and r.id not in served
        and not (r.value is not None and r.value.provenance.needs_verification)
    ]


__all__ = ["CONFIRM_DESIGN_KEY", "CircuitDesignAgent", "Plan"]
