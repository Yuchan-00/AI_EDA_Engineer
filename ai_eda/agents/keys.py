"""The answer keys that steer an agent (control keys) - defined once, so no agent can miss one.

Invariant: a control key is never a requirement and never a regulatory
scope answer. ``--answer <control key>=...`` tells an agent what to do with
something it showed the user (confirm a table, accept a proposal, skip a
stage); it says nothing about the product. Every agent that filters
``ctx.answers`` filters with :data:`CONTROL_KEYS` from this module - a
second literal set drifted once (``pcb.placement`` and ``confirm_design``
reached the regulatory rules and were recorded as scope answers).

The keys that steer the confirmation flow of the requirement extraction live
with it (:mod:`ai_eda.llm.extraction`) and the design confirmation key with
the templates (:mod:`ai_eda.design.base`, which must not import the agents);
the component, regulatory and PCB agents' keys are defined here and
re-exported by those modules.
"""

from __future__ import annotations

from ai_eda.design.base import CONFIRM_DESIGN_KEY
from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY, REJECT_KEY

#: component agent: confirm the presented part candidates / datasheet facts, name a facts file, request the billed extraction
CONFIRM_PARTS_KEY = "confirm_parts"
CONFIRM_FACTS_KEY = "confirm_facts"
FACTS_FILE_KEY = "datasheet_facts_file"
EXTRACT_FACTS_KEY = "extract_datasheet_facts"
#: regulatory agent: decide on shown model proposals (comma-separated ids), request the billed proposal call
ACCEPT_REGS_KEY = "accept_regulations"
REJECT_REGS_KEY = "reject_regulations"
PROPOSE_REGS_KEY = "propose_regulations"
#: PCB agent: ``--answer pcb.placement=skip`` proposes no placement
PLACEMENT_KEY = "pcb.placement"

#: the answer keys that decide on something the requirement extraction showed: they can turn a model's value into the
#: user's within one run, so a design confirmation given beside them refers to a table that was never shown
REQUIREMENT_DECISION_KEYS: frozenset[str] = frozenset({CONFIRM_KEY, ACCEPT_KEY, REJECT_KEY})

#: every answer key that steers an agent; never a requirement, never a scope answer
CONTROL_KEYS: frozenset[str] = frozenset({
    CONFIRM_KEY, ACCEPT_KEY, REJECT_KEY,
    CONFIRM_PARTS_KEY, CONFIRM_FACTS_KEY, FACTS_FILE_KEY, EXTRACT_FACTS_KEY,
    ACCEPT_REGS_KEY, REJECT_REGS_KEY, PROPOSE_REGS_KEY,
    PLACEMENT_KEY, CONFIRM_DESIGN_KEY,
})

__all__ = [
    "ACCEPT_KEY",
    "ACCEPT_REGS_KEY",
    "CONFIRM_DESIGN_KEY",
    "CONFIRM_FACTS_KEY",
    "CONFIRM_KEY",
    "CONFIRM_PARTS_KEY",
    "CONTROL_KEYS",
    "EXTRACT_FACTS_KEY",
    "FACTS_FILE_KEY",
    "PLACEMENT_KEY",
    "PROPOSE_REGS_KEY",
    "REJECT_KEY",
    "REJECT_REGS_KEY",
    "REQUIREMENT_DECISION_KEYS",
]
