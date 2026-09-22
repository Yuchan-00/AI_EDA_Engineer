"""Domain-independent structural validators on the IR itself.

These are cheap, deterministic and always run. They catch problems *before*
KiCad ever sees the design (dangling pin refs, nets on non-existent pins,
components with model-guessed identities, ...).
"""

from __future__ import annotations

from ai_eda.ir import CircuitIR, ProvenanceKind, ValidationResult, ValidationStatus
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry


class ConnectivityValidator(Validator):
    id = "ir.connectivity"
    description = "Every net pin points at a real component pin; no duplicate refs"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        problems: list[str] = []
        refs = [c.ref for c in ir.components]
        dupes = {r for r in refs if refs.count(r) > 1}
        if dupes:
            problems.append(f"duplicate reference designators: {sorted(dupes)}")
        for net in ir.nets:
            for pr in net.pins:
                comp = ir.component(pr.component_ref)
                if comp is None:
                    problems.append(f"net {net.name}: unknown component {pr.component_ref}")
                elif comp.pin(pr.pin_number) is None:
                    problems.append(f"net {net.name}: {pr.component_ref} has no pin {pr.pin_number}")
        if problems:
            return [ValidationResult(check_id=self.id, status=ValidationStatus.FAIL, message="; ".join(problems), tool=self.id)]
        if not ir.nets:
            return [self.not_verified("no nets defined")]
        return [ValidationResult(check_id=self.id, status=ValidationStatus.PASS, tool=self.id)]


class ComponentProvenanceValidator(Validator):
    id = "ir.component_provenance"
    description = "Component identity (MPN/package/pins) must be authoritative, never model output or absent"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        unverified: list[str] = []
        for c in ir.components:
            fields = {"mpn": c.mpn, "package": c.package, "manufacturer": c.manufacturer}
            for name, traced in fields.items():
                if traced is not None and traced.provenance.kind == ProvenanceKind.LLM_GENERATED:
                    unverified.append(f"{c.ref}.{name}")
            if not c.has_authoritative_identity:
                # no MPN, or one that is not backed by official part data: nobody can source this part
                unverified.append(f"{c.ref}.mpn[{c.mpn.provenance.kind if c.mpn is not None else 'missing'}]")
            for p in c.pins:
                if p.provenance.kind == ProvenanceKind.LLM_GENERATED:
                    unverified.append(f"{c.ref}.pin[{p.number}]")
        if unverified:
            return [
                ValidationResult(
                    check_id=self.id,
                    status=ValidationStatus.NOT_VERIFIED,
                    message="component identity must be verified against a datasheet / official part data",
                    tool=self.id,
                    details={"unverified": sorted(set(unverified))},
                )
            ]
        if not ir.components:
            return [self.not_applicable("no components")]
        return [ValidationResult(check_id=self.id, status=ValidationStatus.PASS, tool=self.id)]


def simulation_assumptions(ir: CircuitIR) -> list[str]:
    """Every ``assumption``-provenance value or decision that would reach ngspice, as ``path: rationale``.

    Walks each component's SPICE binding (the binding itself, ``value``,
    ``params``, ``model_card``), the stimuli (themselves, ``value``,
    ``params``), the analyses (themselves, ``params``), the expectations
    (themselves, ``nominal``, ``tol_abs``, ``tol_rel``, ``at``) and
    ``temperature_c`` - the same items the netlist compiler lists under
    ``report["assumptions"]``.
    """
    out: list[str] = []

    def note(path: str, prov) -> None:
        if prov.kind == ProvenanceKind.ASSUMPTION:
            out.append(f"{path}: {prov.note or '(no rationale)'}")

    for c in ir.components:
        b = c.spice
        if b is None:
            continue
        note(f"components[{c.ref}].spice", b.provenance)
        if b.value is not None:
            note(f"components[{c.ref}].spice.value", b.value.provenance)
        if b.model_card is not None:
            note(f"components[{c.ref}].spice.model_card", b.model_card.provenance)
        for k, t in b.params.items():
            note(f"components[{c.ref}].spice.params[{k}]", t.provenance)
    setup = ir.simulation
    if setup is None:
        return out
    for s in setup.stimuli:
        note(f"simulation.stimuli[{s.id}]", s.provenance)
        if s.value is not None:
            note(f"simulation.stimuli[{s.id}].value", s.value.provenance)
        for k, t in s.params.items():
            note(f"simulation.stimuli[{s.id}].params[{k}]", t.provenance)
    for a in setup.analyses:
        note(f"simulation.analyses[{a.id}]", a.provenance)
        for k, t in a.params.items():
            note(f"simulation.analyses[{a.id}].params[{k}]", t.provenance)
    for e in setup.expectations:
        note(f"simulation.expectations[{e.id}]", e.provenance)
        for label, t in (("nominal", e.nominal), ("tol_abs", e.tol_abs), ("tol_rel", e.tol_rel), ("at", e.at)):
            if t is not None:
                note(f"simulation.expectations[{e.id}].{label}", t.provenance)
    if setup.temperature_c is not None:
        note("simulation.temperature_c", setup.temperature_c.provenance)
    return out


class AssumptionsSurfacedValidator(Validator):
    id = "ir.assumptions"
    description = "Every assumption in requirements / parameters / the simulation setup is listed for user confirmation"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        assumptions: list[str] = []
        for key, t in ir.parameters.items():
            if t.provenance.kind == ProvenanceKind.ASSUMPTION:
                assumptions.append(f"{key}: {t.provenance.note or '(no rationale)'}")
        for r in ir.requirements.requirements:
            if r.value is not None and r.value.provenance.kind == ProvenanceKind.ASSUMPTION:
                assumptions.append(f"{r.key}: {r.value.provenance.note or '(no rationale)'}")
        assumptions += simulation_assumptions(ir)
        if assumptions:
            return [
                ValidationResult(
                    check_id=self.id,
                    status=ValidationStatus.USER_INPUT_REQUIRED,
                    message=f"{len(assumptions)} assumption(s) need confirmation",
                    tool=self.id,
                    details={"assumptions": assumptions},
                )
            ]
        return [ValidationResult(check_id=self.id, status=ValidationStatus.PASS, tool=self.id)]


class LLMRequirementsValidator(Validator):
    id = "ir.llm_requirements"
    description = "Every requirement a model inferred or extracted and the user has not decided on is listed; nothing downstream may rely on it"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY, REJECT_KEY

        pending: list[dict[str, str]] = []
        for r in ir.requirements.requirements:
            if r.value is not None and r.value.provenance.kind == ProvenanceKind.LLM_GENERATED:
                pending.append({"key": r.key, "id": r.id, "kind": str(r.kind), "note": r.value.provenance.note or "(no rationale)"})
        if pending:
            keys = ", ".join(p["key"] for p in pending)
            return [
                ValidationResult(
                    check_id=self.id,
                    status=ValidationStatus.USER_INPUT_REQUIRED,
                    message=(
                        f"{len(pending)} model-inferred requirement(s) not yet accepted ({keys}): decide with "
                        f"--answer {ACCEPT_KEY}=<keys> / --answer {REJECT_KEY}=<keys>, or correct the extraction via {CONFIRM_KEY}"
                    ),
                    tool=self.id,
                    details={"pending": pending, "accept_key": ACCEPT_KEY, "reject_key": REJECT_KEY},
                )
            ]
        return [ValidationResult(check_id=self.id, status=ValidationStatus.PASS, tool=self.id)]


for _v in (ConnectivityValidator(), ComponentProvenanceValidator(), AssumptionsSurfacedValidator(), LLMRequirementsValidator()):
    default_registry.register(_v)
