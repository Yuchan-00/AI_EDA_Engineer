"""Net / connectivity model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Provenance


class NetKind(StrEnum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    ANALOG = "analog"
    HIGH_SPEED = "high_speed"
    RF = "rf"


class PinRef(BaseModel):
    component_ref: str
    pin_number: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.component_ref}.{self.pin_number}"


class Net(BaseModel):
    name: str  # "VIN", "GND", "N$3", ...
    kind: NetKind = NetKind.SIGNAL
    pins: list[PinRef] = Field(default_factory=list)
    #: net class for PCB rules (width, clearance) - resolved against PCB constraints
    net_class: str = "Default"
    provenance: Provenance
    #: requirement ids this net exists for (e.g. an interface or a power rail requirement), so the
    #: chain pad -> net -> requirement does not stop at the net
    serves_requirements: list[str] = Field(default_factory=list)
