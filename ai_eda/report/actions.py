"""Deterministic action groups over stored results; no verdicts or IR changes.

Order is fixed by group, then check id. Last occurrence wins, just as in
ValidationState.latest_by_check; timestamps are never used. Every latest
row that still asks for something (not PASS, not NOT_APPLICABLE) is retained
with its original status; a check that does not apply needs no action.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel

from ai_eda.ir.validation import ValidationResult, ValidationStatus


class ActionCheck(BaseModel):
    check_id: str
    status: str
    message: str


class ActionGroup(BaseModel):
    id: str
    title: str
    role: str
    next_action: str
    check_ids: list[str]
    checks: list[ActionCheck]


# Separate document causes because supplying an MPN, a pointer and an online
# fetch are different next steps. A result with several needs stays whole.
_GROUPS = {
    "missing_part_identity": ("Missing part identity", "user", "Supply the listed parts' MPNs and datasheets."),
    "missing_part_documents": ("Missing part documents", "user", "Supply datasheets for the listed parts, including any missing MPNs."),
    "offline_part_documents": ("Part documents not fetched", "user", "Run with --online or supply the datasheets, including any missing MPNs."),
    "missing_fab_inputs": ("Missing fab inputs", "user", "Supply the fab's capability file with --fab-capability FILE; review all listed limits."),
    "unsupported_backends": ("Known unsupported backends", "software", "Backend implementation is needed; no user input can enable this check yet."),
    "scope_questions": ("Explicit scope questions", "user", "Answer the listed scope questions with --answer key=value."),
    "unmapped": ("Unmapped", "undetermined", "Review each recorded status and reason to determine the next step."),
}


def _group_id(result: ValidationResult) -> str:
    check, message, details = result.check_id, result.message, result.details
    if check.startswith("component.existence."):
        subchecks = details.get("checks")
        problems = []
        if isinstance(subchecks, list):
            problems = [c.get("message", "") for c in subchecks
                        if isinstance(c, dict) and c.get("status") not in ("PASS", "NOT_APPLICABLE")]
        # The result message uses '; <subcheck>: <reason>' when details are absent.
        reasons = [m for m in problems if isinstance(m, str) and m] or [message] + [p.split(": ", 1)[-1] for p in message.split("; ")[1:]]
        if any(m.startswith("no datasheet pointer") for m in reasons):
            return "missing_part_documents"
        if any(m.startswith("not fetched (offline") for m in reasons):
            return "offline_part_documents"
        if any(m.startswith("no MPN in the IR") for m in reasons):
            return "missing_part_identity"
    if check in ("ir.component_provenance", "review.component_provenance"):
        if ": no MPN:" in message:
            return "missing_part_identity"
    if check == "domain.power.thermal" and message.startswith("no component carries theta_ja / t_j_max"):
        return "missing_part_documents"
    if check == "domain.power.pi" and message.startswith("power integrity backend not implemented"):
        return "unsupported_backends"
    if check == "regulatory.applicability" and (details.get("missing_keys") or "candidate(s) undecided - answer " in message):
        return "scope_questions"
    if check == "requirements.crosscheck" and details.get("open"):
        return "scope_questions"
    if check in ("mfg.capability", "mfg.capability_source", "review.manufacturing_capabilities", "pcb.routing.clearance", "pcb.silk.size"):
        if message.startswith(("no fab capability file", "no clearance limit in ir.pcb.manufacturing")):
            return "missing_fab_inputs"
        compared = details.get("compared")
        if check == "review.manufacturing_capabilities":
            live = details.get("live")
            compared = live.get("compared") if isinstance(live, dict) else None
        if isinstance(compared, list) and any(isinstance(c, dict) and str(c.get("message", "")).startswith("no ")
                                              and " in ir.pcb.manufacturing" in str(c.get("message", "")) for c in compared):
            return "missing_fab_inputs"
        if check == "pcb.silk.size" and message.startswith("no min_silk_") and "--fab-capability" in message:
            return "missing_fab_inputs"
    return "unmapped"


def group_actions(results: Iterable[ValidationResult]) -> list[ActionGroup]:
    """Copy each latest result that is neither PASS nor NOT_APPLICABLE into exactly one group, without changing it.

    Match only known check families and emitted reasons; unknown or missing
    metadata goes to unmapped. Grouping is a suggestion, never a verdict.
    """
    latest = {r.check_id: r for r in results}
    grouped: dict[str, list[ActionCheck]] = {}
    for check, result in sorted(latest.items()):
        if result.status in (ValidationStatus.PASS, ValidationStatus.NOT_APPLICABLE):
            continue
        grouped.setdefault(_group_id(result), []).append(ActionCheck(check_id=check, status=str(result.status), message=result.message))
    return [ActionGroup(id=key, title=title, role=role, next_action=action, check_ids=[c.check_id for c in grouped[key]], checks=grouped[key])
            for key, (title, role, action) in _GROUPS.items() if key in grouped]
