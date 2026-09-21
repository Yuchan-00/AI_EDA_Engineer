"""Design constraints (electrical limits, thermal, mechanical, manufacturing, regulatory)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Provenance, Traced


class ConstraintKind(StrEnum):
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    MANUFACTURING = "manufacturing"
    REGULATORY = "regulatory"
    SIGNAL_INTEGRITY = "signal_integrity"
    POWER_INTEGRITY = "power_integrity"
    RF = "rf"


class Constraint(BaseModel):
    id: str
    kind: ConstraintKind
    description: str
    #: what the constraint applies to: a component ref, a net name, a block id, or "*"
    target: str = "*"
    #: e.g. {"max": Traced(85, "degC")} or {"min_track_width": Traced(0.127, "mm")}
    parameters: dict[str, Traced] = Field(default_factory=dict)
    provenance: Provenance
