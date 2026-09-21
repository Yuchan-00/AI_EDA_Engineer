"""Approval gate for outward-facing actions.

Anything that leaves the project (git push, uploads, orders, purchases,
regulatory submissions, secret handling, system-wide installs) must pass
through :func:`require_approval` with an explicit, per-action approval.
"""

from ai_eda.security.approval import ApprovalGate, ExternalAction, require_approval

__all__ = ["ApprovalGate", "ExternalAction", "require_approval"]
