from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.errors import ApprovalRequiredError


class ExternalAction(StrEnum):
    GIT_PUSH = "git_push"
    EXTERNAL_UPLOAD = "external_upload"
    SECRET_ACCESS = "secret_access"
    SYSTEM_INSTALL = "system_install"
    SYSTEM_DELETE = "system_delete"
    PCB_ORDER = "pcb_order"
    COMPONENT_PURCHASE = "component_purchase"
    PAID_API_CALL = "paid_api_call"
    MANUFACTURING_ORDER = "manufacturing_order"
    REGULATORY_SUBMISSION = "regulatory_submission"
    #: opening a network connection to fetch a document (datasheet, official text); granted once per online session
    NETWORK_FETCH = "network_fetch"


class Approval(BaseModel):
    action: ExternalAction
    detail: str
    approved_by: str
    granted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    consumed: bool = False


class ApprovalGate:
    """Holds single-use approvals. Approval for one action never generalises."""

    def __init__(self) -> None:
        self._approvals: list[Approval] = []
        self.audit: list[dict] = []

    def grant(self, action: ExternalAction, detail: str, approved_by: str) -> Approval:
        a = Approval(action=action, detail=detail, approved_by=approved_by)
        self._approvals.append(a)
        self.audit.append({"event": "grant", "action": action, "detail": detail, "by": approved_by})
        return a

    def check(self, action: ExternalAction, detail: str) -> None:
        for a in self._approvals:
            if not a.consumed and a.action == action and a.detail == detail:
                a.consumed = True
                self.audit.append({"event": "consume", "action": action, "detail": detail})
                return
        self.audit.append({"event": "denied", "action": action, "detail": detail})
        raise ApprovalRequiredError(action, detail)


_default_gate = ApprovalGate()


def require_approval(action: ExternalAction, detail: str, gate: ApprovalGate | None = None) -> None:
    (gate or _default_gate).check(action, detail)


def default_gate() -> ApprovalGate:
    return _default_gate
