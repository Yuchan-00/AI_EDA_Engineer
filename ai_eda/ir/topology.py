"""Circuit topology / architecture model.

``CircuitDomain`` drives validator selection: a power design gets
thermal/power analysis, an RF design gets impedance checks, etc.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

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
    """A functional block (e.g. "buck converter", "input protection", "MCU")."""

    id: str
    function: str
    domain: CircuitDomain = CircuitDomain.OTHER
    component_refs: list[str] = Field(default_factory=list)
    input_nets: list[str] = Field(default_factory=list)
    output_nets: list[str] = Field(default_factory=list)
    provenance: Provenance


class Topology(BaseModel):
    name: str  # e.g. "linear regulator", "synchronous buck"
    domains: list[CircuitDomain] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    rationale: str = ""
    provenance: Provenance
