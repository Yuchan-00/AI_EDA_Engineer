from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.ir import (
    CircuitDomain,
    CircuitIR,
    Component,
    LibraryRef,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SourceRef,
    Topology,
    authoritative,
    derived,
    user_requirement,
)

DS = SourceRef(title="Generic resistor datasheet", authority="Vendor", content_hash="sha256:abc")
AUTH = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)


def _pin(n: str) -> Pin:
    return Pin(number=n, name=f"~{n}", electrical_type=PinElectricalType.PASSIVE, provenance=AUTH)


def make_component(ref: str, value: str, verified_lib: bool = True) -> Component:
    return Component(
        ref=ref,
        value=value,
        description="resistor",
        mpn=authoritative(f"MPN-{value}", DS),
        package=authoritative("0603", DS),
        pins=[_pin("1"), _pin("2")],
        symbol=LibraryRef(library="Device", name="R", verified=verified_lib),
        footprint=LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric", verified=verified_lib),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test", note="fixture"),
        serves_requirements=["req.v_out"],
    )


@pytest.fixture
def divider_ir(tmp_path: Path) -> CircuitIR:
    """A two-resistor divider with authoritative parts, derived parameters, no artifacts."""
    ir = CircuitIR(project=ProjectMeta(id="t", name="divider", workdir=str(tmp_path)))
    ir.topology = Topology(name="resistive divider", domains=[CircuitDomain.ANALOG], provenance=AUTH)
    ir.components = [make_component("R1", "10k"), make_component("R2", "10k")]
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
    ir.nets = [
        Net(name="VIN", kind=NetKind.POWER, pins=[PinRef(component_ref="R1", pin_number="1")], provenance=net_p),
        Net(name="VOUT", pins=[PinRef(component_ref="R1", pin_number="2"), PinRef(component_ref="R2", pin_number="1")], provenance=net_p),
        Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="R2", pin_number="2")], provenance=net_p),
    ]
    ir.parameters["v_in"] = user_requirement(12.0, "V")
    ir.parameters["r1"] = authoritative(10_000.0, DS, "ohm")
    ir.parameters["r2"] = authoritative(10_000.0, DS, "ohm")
    ir.parameters["v_out"] = derived(6.0, tool="calc.divider.v_out", inputs={"v_in": "v_in", "r1": "r1", "r2": "r2"}, unit="V")
    return ir
