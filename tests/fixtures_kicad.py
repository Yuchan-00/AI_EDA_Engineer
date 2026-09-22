"""Shared IR fixtures for the KiCad compiler tests (schematic *and* PCB).

The reference vertical slice is a resistive divider with a 3-pin header so
that every net has at least two pins (a single-pin net makes KiCad's ERC emit
``isolated_pin_label`` warnings):

* ``R1``, ``R2``  ``Device:R`` / ``Resistor_SMD:R_0603_1608Metric``
* ``J1``          ``Connector_Generic:Conn_01x03`` /
                  ``Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical``
* nets ``VIN = {J1.1, R1.1}``, ``VOUT = {J1.2, R1.2, R2.1}``, ``GND = {J1.3, R2.2}``

Every pin carries authoritative provenance and every :class:`LibraryRef` is
resolved through :class:`KicadLibrary` so ``verified`` reflects what is really
on disk (``False`` when the KiCad libraries are not installed - tests that need
them skip). This module is deliberately self-contained: it depends only on
``ai_eda.ir``, ``ai_eda.tools.calc`` and ``ai_eda.tools.kicad.library``.

SPICE: the resistors are bound as ``R`` elements with their authoritative
resistance, the header is excluded (no electrical model), and
:func:`divider_simulation` describes the 12 V DC stimulus on ``VIN``, an
``op`` and a ``dc`` sweep (0..12 V in 1 V steps) with two expectations whose
nominals are calculator outputs stored in ``ir.parameters``: ``v(VOUT)`` at
the operating point = ``v_out`` (6 V, verifying ``req.v_out`` = 6 V) and
``v(VOUT)`` at ``VIN = v_in_mid`` (6 V) on the sweep = ``v_out_mid`` (3 V,
verifying ``req.v_out_half`` = 3 V - the "half the input" clause at a second
input), both within 1 %. Each expectation's nominal is the value of the
requirement it is traced to: the reviewer compares the two. The netlist the
compiler writes for it is ``divider_conn\\nR1 VIN VOUT 10k\\nR2 VOUT 0 10k\\nVVIN VIN 0 DC 12\\n.end\\n``.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.ir import (
    AnalysisSpec,
    BoardOutline,
    BoardSide,
    CircuitDomain,
    CircuitIR,
    Component,
    Expectation,
    LibraryRef,
    Net,
    NetKind,
    PCBDesign,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    Requirement,
    RequirementKind,
    SimulationSetup,
    SourceRef,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    authoritative,
    user_requirement,
)
from ai_eda.tools.calc import voltage_divider_output
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

__all__ = [
    "PROJECT_ID",
    "EXPECTED_NETS",
    "HAND_PLACED",
    "ROUTABLE_PLACEMENTS",
    "SHORTING_PLACEMENTS",
    "RESISTOR_DS",
    "HEADER_DS",
    "USER",
    "divider_with_connector_ir",
    "divider_simulation",
    "ir_net_map",
    "resistor",
    "pin_header",
    "pin_header_1x03",
]

#: file stem of every artifact the compilers write for this fixture
PROJECT_ID = "divider_conn"

#: provenance of the simulation setup: the person who wrote the fixture decided what to simulate
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture simulation setup")

RESISTOR_DS = SourceRef(
    title="Generic thick film chip resistor datasheet",
    authority="Vendor",
    content_hash="sha256:fixture-resistor-datasheet",
)
HEADER_DS = SourceRef(
    title="2.54 mm pin header datasheet",
    authority="Vendor",
    content_hash="sha256:fixture-header-datasheet",
)

#: net name -> {(ref, pin)} - what the exported KiCad netlist must reproduce exactly
EXPECTED_NETS: dict[str, set[tuple[str, str]]] = {
    "VIN": {("J1", "1"), ("R1", "1")},
    "VOUT": {("J1", "2"), ("R1", "2"), ("R2", "1")},
    "GND": {("J1", "3"), ("R2", "2")},
}

#: Provenance of the hand-made fixture layout: a person placed these parts and validated the
#: result with kicad-cli 10.0.6 DRC, so the reviewer's layout-traceability check is satisfied.
HAND_PLACED = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture layout, hand-placed and DRC-validated with kicad-cli 10.0.6")

#: Placements that the *naive* router (straight pad-centre chains on F.Cu,
#: see :mod:`ai_eda.tools.routing.naive`) connects with 0 DRC errors and 0
#: warnings on the 30 x 20 mm board - validated with kicad-cli 10.0.6. The
#: header's pad row runs down the left edge (pad 1 VIN at (5, 6), pad 2 VOUT
#: at (5, 8.54), pad 3 GND at (5, 11.08)); the resistors stand vertically
#: (rot 270: pad 1 on top) at x = 14 so VOUT reaches R1.2 / R2.1 without
#: grazing a neighbouring pad.
ROUTABLE_PLACEMENTS: list[Placement] = [
    Placement(component_ref="J1", x_mm=5.0, y_mm=6.0, rotation_deg=0.0, side=BoardSide.TOP, provenance=HAND_PLACED),
    Placement(component_ref="R1", x_mm=14.0, y_mm=6.0, rotation_deg=270.0, side=BoardSide.TOP, provenance=HAND_PLACED),
    Placement(component_ref="R2", x_mm=14.0, y_mm=11.08, rotation_deg=270.0, side=BoardSide.TOP, provenance=HAND_PLACED),
]

#: The first layout tried (J1 rotated 90 so its pad row points at the
#: resistors). Straight pad-centre chains short J1's pads: real DRC reports
#: shorting_items / clearance / tracks_crossing. Kept as the negative case
#: that shows the naive router is not a router - DRC is the judge.
SHORTING_PLACEMENTS: list[Placement] = [
    Placement(component_ref="J1", x_mm=5.0, y_mm=10.0, rotation_deg=90.0, side=BoardSide.TOP, provenance=HAND_PLACED),
    Placement(component_ref="R1", x_mm=15.0, y_mm=7.0, rotation_deg=0.0, side=BoardSide.TOP, provenance=HAND_PLACED),
    Placement(component_ref="R2", x_mm=15.0, y_mm=13.0, rotation_deg=0.0, side=BoardSide.TOP, provenance=HAND_PLACED),
]


def _auth(source: SourceRef) -> Provenance:
    return Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=source)


def _pin(number: str, name: str, source: SourceRef) -> Pin:
    return Pin(number=number, name=name, electrical_type=PinElectricalType.PASSIVE, provenance=_auth(source))


def resistor(ref: str, value: str, ohms: float, library: KicadLibrary, serves: tuple[str, ...] = ("req.v_out",)) -> Component:
    """A verified 0603 chip resistor, bound in SPICE as an ideal ``R`` with its authoritative resistance.

    Pin names are the library's (``Device:R`` pins are unnamed).
    """
    resistance = authoritative(ohms, RESISTOR_DS, "ohm")
    return Component(
        ref=ref,
        value=value,
        description="Resistor",
        manufacturer=authoritative("Generic", RESISTOR_DS),
        mpn=authoritative(f"RC0603FR-07{value}L", RESISTOR_DS),
        datasheet=RESISTOR_DS,
        package=authoritative("0603", RESISTOR_DS),
        pins=[_pin("1", "", RESISTOR_DS), _pin("2", "", RESISTOR_DS)],
        electrical={"resistance": resistance},
        symbol=library.resolve_symbol(LibraryRef(library="Device", name="R")),
        footprint=library.resolve_footprint(LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric")),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="fixture", note="divider element"),
        serves_requirements=list(serves),
        spice=SpiceBinding(
            device=SpiceDevice.R,
            value=resistance,
            provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=RESISTOR_DS, note="ideal resistor at the datasheet nominal value"),
        ),
    )


def pin_header(ref: str, n_pins: int, library: KicadLibrary, serves: tuple[str, ...] = ("req.interface",)) -> Component:
    """A verified 1xN 2.54 mm vertical pin header (``Connector_Generic:Conn_01x0N``), excluded from SPICE."""
    return Component(
        ref=ref,
        value=f"Conn_01x{n_pins:02d}",
        description=f"Generic connector, single row, 01x{n_pins:02d}",
        manufacturer=authoritative("Generic", HEADER_DS),
        mpn=authoritative(f"PH1-{n_pins:02d}-UA", HEADER_DS),
        datasheet=HEADER_DS,
        package=authoritative(f"PinHeader_1x{n_pins:02d}_P2.54mm_Vertical", HEADER_DS),
        pins=[_pin(str(i), f"Pin_{i}", HEADER_DS) for i in range(1, n_pins + 1)],
        symbol=library.resolve_symbol(LibraryRef(library="Connector_Generic", name=f"Conn_01x{n_pins:02d}")),
        footprint=library.resolve_footprint(
            LibraryRef(library="Connector_PinHeader_2.54mm", name=f"PinHeader_1x{n_pins:02d}_P2.54mm_Vertical")
        ),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="fixture", note="board header"),
        serves_requirements=list(serves),
        spice=SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=USER),
    )


def pin_header_1x03(ref: str, library: KicadLibrary) -> Component:
    """A verified 1x03 2.54 mm vertical pin header (``Connector_Generic:Conn_01x03``)."""
    c = pin_header(ref, 3, library)
    c.provenance = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture", note="VIN / VOUT / GND header")
    return c


def divider_simulation(ir: CircuitIR) -> SimulationSetup:
    """The simulation setup of the divider: stimulus, ``op`` + ``dc`` analyses, expectations against ``ir.parameters``.

    ``ir.parameters`` must already hold ``v_out`` (op nominal), ``v_in_mid``
    (the sweep point) and ``v_out_mid`` (the nominal there): the expectations
    reference those traced values, so the CALCULATION stage recomputes the
    very numbers ngspice is checked against.
    """
    return SimulationSetup(
        stimuli=[
            Stimulus(
                id="VIN", source="voltage", net="VIN", reference_net="GND", kind=StimulusKind.DC,
                value=user_requirement(12.0, "V", note="input supply of the requirement"), provenance=USER, serves_requirements=["req.v_in"],
            )
        ],
        analyses=[
            AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=USER),
            AnalysisSpec(
                id="dc_vin", kind=SpiceAnalysis.DC,
                params={"source": user_requirement("VIN"), "start": user_requirement(0.0, "V"), "stop": user_requirement(12.0, "V"), "step": user_requirement(1.0, "V")},
                provenance=USER,
            ),
        ],
        expectations=[
            Expectation(
                id="v_out", analysis_id="op", vector="v(VOUT)", reduce=Reduce.VALUE, nominal=ir.parameters["v_out"],
                tol_rel=user_requirement(0.01, note="1 % output accuracy"), requirement_id="req.v_out", provenance=USER,
            ),
            Expectation(
                id="v_out_mid", analysis_id="dc_vin", vector="v(VOUT)", reduce=Reduce.AT, at=ir.parameters["v_in_mid"], nominal=ir.parameters["v_out_mid"],
                tol_rel=user_requirement(0.01, note="1 % output accuracy"), requirement_id="req.v_out_half", provenance=USER,
            ),
        ],
    )


def divider_with_connector_ir(tmp_path: Path, library: KicadLibrary | None = None) -> CircuitIR:
    """Build the divider-with-connector IR, with a 30 x 20 mm board and :data:`ROUTABLE_PLACEMENTS`.

    ``tmp_path`` becomes ``project.workdir``; ``library`` (default: the installed
    KiCad libraries) resolves the symbol / footprint references. No tracks are
    included: callers that need copper run ``ai_eda.tools.routing.route_naive``.
    """
    lib = library or KicadLibrary()
    ir = CircuitIR(
        project=ProjectMeta(
            id=PROJECT_ID,
            name="Resistive divider with header",
            description="12 V -> 6 V resistive divider on a 3-pin header",
            workdir=str(tmp_path),
        )
    )
    ir.topology = Topology(
        name="resistive divider",
        domains=[CircuitDomain.ANALOG],
        provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture"),
    )
    ir.requirements.requirements = [
        Requirement(id="req.v_in", key="v_in", text="12 V DC input on the header", kind=RequirementKind.EXPLICIT, value=user_requirement(12.0, "V")),
        Requirement(id="req.v_out", key="v_out", text="6 V output (half the input) within 1 %", kind=RequirementKind.EXPLICIT, value=user_requirement(6.0, "V")),
        Requirement(
            id="req.v_out_half", key="v_out_half", text="the output stays half the input: 3 V at a 6 V input, within 1 %",
            kind=RequirementKind.EXPLICIT, value=user_requirement(3.0, "V"),
        ),
        Requirement(id="req.interface", key="interface", text="VIN / VOUT / GND on a 3-pin 2.54 mm header", kind=RequirementKind.EXPLICIT, category="mechanical"),
    ]
    ir.components = [
        resistor("R1", "10k", 10_000.0, lib),
        resistor("R2", "10k", 10_000.0, lib),
        pin_header_1x03("J1", lib),
    ]
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.nets = [
        Net(
            name="VIN",
            kind=NetKind.POWER,
            pins=[PinRef(component_ref="J1", pin_number="1"), PinRef(component_ref="R1", pin_number="1")],
            provenance=net_p,
            serves_requirements=["req.v_in"],
        ),
        Net(
            name="VOUT",
            pins=[
                PinRef(component_ref="J1", pin_number="2"),
                PinRef(component_ref="R1", pin_number="2"),
                PinRef(component_ref="R2", pin_number="1"),
            ],
            provenance=net_p,
            serves_requirements=["req.v_out", "req.v_out_half"],
        ),
        Net(
            name="GND",
            kind=NetKind.GROUND,
            pins=[PinRef(component_ref="J1", pin_number="3"), PinRef(component_ref="R2", pin_number="2")],
            provenance=net_p,
        ),
    ]
    params = ir.parameters
    params["v_in"] = user_requirement(12.0, "V")
    params["r1"] = ir.component("R1").electrical["resistance"]
    params["r2"] = ir.component("R2").electrical["resistance"]
    params["v_out"] = voltage_divider_output(params["v_in"], params["r1"], params["r2"], ("v_in", "r1", "r2"))  # 6 V by calc.divider.v_out
    params["v_in_mid"] = user_requirement(6.0, "V", note="dc sweep point checked against the calculator")
    params["v_out_mid"] = voltage_divider_output(params["v_in_mid"], params["r1"], params["r2"], ("v_in_mid", "r1", "r2"))  # 3 V
    ir.simulation = divider_simulation(ir)
    # copies: tests mutate placements in place, the module-level fixture list must stay pristine
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0, height_mm=20.0), placements=[p.model_copy() for p in ROUTABLE_PLACEMENTS])
    return ir


def ir_net_map(ir: CircuitIR) -> dict[str, set[tuple[str, str]]]:
    """``{net name: {(ref, pin number)}}`` straight from the IR - compare with a parsed netlist."""
    return {n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets}
