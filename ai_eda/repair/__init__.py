"""Automatic repair.

Repairs are deterministic and limited to *regenerating derived artifacts from
the IR* or *re-running a tool*. Anything that would change the design itself
(nets, components, values, regulatory scope) is NOT repairable here and is
reported for a human decision.
"""

from ai_eda.repair.strategies import (
    NON_REPAIRABLE,
    RegenerateArtifact,
    RepairAction,
    RepairStrategy,
    RerunTool,
    select_strategy,
)
from ai_eda.repair.loop import RepairLoop, RepairOutcome

__all__ = [
    "NON_REPAIRABLE",
    "RegenerateArtifact",
    "RepairAction",
    "RepairStrategy",
    "RerunTool",
    "select_strategy",
    "RepairLoop",
    "RepairOutcome",
]
