"""Circuit topology / architecture model.

``CircuitDomain`` drives validator selection: a power design gets
thermal/power analysis, an RF design gets impedance checks, etc.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from ai_eda.ir.nets import PinRef
from ai_eda.ir.provenance import Provenance


class CircuitDomain(StrEnum):
    ANALOG = "analog"
    DIGITAL = "digital"
    MIXED_SIGNAL = "mixed_signal"
    POWER = "power"
    RF = "rf"
    HIGH_SPEED = "high_speed"
    SENSOR_INTERFACE = "sensor_interface"
    EMBEDDED = "embedded"
    OTHER = "other"


class Block(BaseModel):
    """A functional block; ``function`` names its role, ``component_refs`` its members."""

    id: str
    function: str
    domain: CircuitDomain = CircuitDomain.OTHER
    component_refs: list[str] = Field(default_factory=list)
    input_nets: list[str] = Field(default_factory=list)
    output_nets: list[str] = Field(default_factory=list)
    provenance: Provenance
    inferred: bool = Field(default=False, exclude_if=lambda value: not value)


class PinAssociation(BaseModel):
    """Declared support terminal -> IC pin intent, beyond shared-net connectivity."""

    terminal: PinRef
    ic_pin: PinRef
    role: str = Field(min_length=1)
    provenance: Provenance


class Topology(BaseModel):
    name: str  # e.g. "linear regulator", "synchronous buck"
    domains: list[CircuitDomain] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    # Empty intent preserves both the serialized legacy topology and its design hash.
    pin_associations: list[PinAssociation] = Field(default_factory=list, exclude_if=lambda value: not value)
    schematic_layout: Literal["legacy", "textbook"] | None = Field(default=None, exclude_if=lambda value: value is None)
    rationale: str = ""
    provenance: Provenance
