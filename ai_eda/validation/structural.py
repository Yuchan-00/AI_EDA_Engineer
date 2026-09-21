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


class AssumptionsSurfacedValidator(Validator):
    id = "ir.assumptions"
    description = "Every assumption in requirements/parameters is listed for user confirmation"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        assumptions: list[str] = []
        for key, t in ir.parameters.items():
            if t.provenance.kind == ProvenanceKind.ASSUMPTION:
                assumptions.append(f"{key}: {t.provenance.note or '(no rationale)'}")
        for r in ir.requirements.requirements:
            if r.value is not None and r.value.provenance.kind == ProvenanceKind.ASSUMPTION:
                assumptions.append(f"{r.key}: {r.value.provenance.note or '(no rationale)'}")
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


for _v in (ConnectivityValidator(), ComponentProvenanceValidator(), AssumptionsSurfacedValidator()):
    default_registry.register(_v)
