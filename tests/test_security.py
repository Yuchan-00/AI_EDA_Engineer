import pytest

from ai_eda.errors import ApprovalRequiredError
from ai_eda.security import ApprovalGate, ExternalAction, require_approval


def test_denied_without_approval():
    gate = ApprovalGate()
    with pytest.raises(ApprovalRequiredError):
        require_approval(ExternalAction.PCB_ORDER, "order #1", gate)
    assert gate.audit[-1]["event"] == "denied"


def test_approval_is_single_use_and_specific():
    gate = ApprovalGate()
    gate.grant(ExternalAction.GIT_PUSH, "origin/main", approved_by="user")
    require_approval(ExternalAction.GIT_PUSH, "origin/main", gate)  # consumes
    with pytest.raises(ApprovalRequiredError):
        require_approval(ExternalAction.GIT_PUSH, "origin/main", gate)
    gate.grant(ExternalAction.GIT_PUSH, "origin/main", approved_by="user")
    with pytest.raises(ApprovalRequiredError):  # different detail, same action
        require_approval(ExternalAction.GIT_PUSH, "origin/release", gate)
