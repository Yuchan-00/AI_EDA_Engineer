"""The template's own deterministic check on a design it built: are the copied inputs still the requirements?

Invariant: a parameter :func:`~ai_eda.design.inputs.read_value` copied from a
requirement (``derived_from`` = that requirement id, note ``parsed from
req.<key>``) is the design's reading of the user's words. On every later run
the circuit agent re-reads the requirement with the same parser and
reports ``design.inputs_vs_requirements``: PASS when every copied parameter
still equals the re-read value, FAIL (``repair: human``) when one differs -
the requirement moved after the design was built and nobody may pick which
of the two is right - and NOT_VERIFIED when a source requirement is gone or
no longer readable. A design without copied inputs (not template-made)
gets no result: there is nothing to compare.
"""

from __future__ import annotations

import math

from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.tools.calc.quantity import QUANTITY_VERSION

from ai_eda.design.base import TEMPLATE_VERSION, TOOL_ID
from ai_eda.design.inputs import is_template_input, read_value

INPUTS_CHECK = "design.inputs_vs_requirements"


def check_inputs_vs_requirements(ir: CircuitIR) -> ValidationResult | None:
    """The ``design.inputs_vs_requirements`` result, or ``None`` when no parameter was copied from a requirement."""
    copied = [(key, t) for key, t in ir.parameters.items() if is_template_input(t)]
    if not copied:
        return None
    by_id = {r.id: r for r in ir.requirements.requirements}
    per: dict[str, dict] = {}
    mismatches: list[str] = []
    unverified: list[str] = []
    for key, t in copied:
        rid = t.provenance.derived_from[0]
        entry: dict = {"requirement": rid, "stored": t.value, "unit": t.unit}
        per[key] = entry
        req = by_id.get(rid)
        if req is None:
            entry["status"] = ValidationStatus.NOT_VERIFIED.value
            entry["reason"] = "requirement no longer in the IR"
            unverified.append(f"{key}: {rid} is no longer in the IR")
            continue
        now, why = read_value(req, t.unit or "")
        if now is None:
            entry["status"] = ValidationStatus.NOT_VERIFIED.value
            entry["reason"] = why
            unverified.append(f"{key}: {why}")
            continue
        entry["reread"] = now.value
        if math.isclose(float(now.value), float(t.value), rel_tol=1e-12, abs_tol=0.0):
            entry["status"] = ValidationStatus.PASS.value
        else:
            entry["status"] = ValidationStatus.FAIL.value
            mismatches.append(f"{key}: the design was built from {t.value:.12g} {t.unit}, {rid} now reads {now.value:.12g} {t.unit}")
    details: dict = {"parameters": per, "mismatches": mismatches, "unverified": unverified, "quantity_version": QUANTITY_VERSION}
    stamp = dict(check_id=INPUTS_CHECK, tool=TOOL_ID, tool_version=TEMPLATE_VERSION, details=details)
    if mismatches:
        details["repair"] = "human"
        return ValidationResult(status=ValidationStatus.FAIL, message=f"{len(mismatches)} template input(s) no longer equal their requirement: " + "; ".join(mismatches), **stamp)
    if unverified:
        return ValidationResult(status=ValidationStatus.NOT_VERIFIED, message=f"{len(unverified)} template input(s) could not be re-read: " + "; ".join(unverified), **stamp)
    return ValidationResult(status=ValidationStatus.PASS, message=f"{len(copied)} template input(s) still equal the requirements they were parsed from", **stamp)


__all__ = ["INPUTS_CHECK", "check_inputs_vs_requirements"]
