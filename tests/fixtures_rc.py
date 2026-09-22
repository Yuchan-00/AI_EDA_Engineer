"""RC low-pass fixture for the SPICE stage: a transient expectation whose nominal is a calculator output.

* ``R1`` 1 k (``Device:R``), ``C1`` 1 u (``Device:C`` / ``Capacitor_SMD:C_0603_1608Metric``),
  ``J1`` a 2-pin header (excluded from SPICE)
* nets ``IN = {J1.1, R1.1}``, ``OUT = {R1.2, C1.1}``, ``GND = {J1.2, C1.2}``
* stimulus ``VIN``: ``PULSE(0 5 0 1n 1n 1 2)`` on ``IN`` - a 5 V step at t = 0
  (1 ns edges so ngspice sees a true step; the pulse outlasts the analysis)
* analysis ``tran`` step 1 us, stop 5 ms
* expectation ``v_out_tau``: ``v(OUT)`` at ``t = tau`` equals
  ``v_out_at_tau = rc_step_response(v_step, tau, tau)`` = 5 V (1 - e^-1) =
  3.1606 V within 2 %, where ``tau = rc_time_constant(r, c)`` = 1 ms - both
  in ``ir.parameters`` with ``derived`` provenance, so the CALCULATION stage
  recomputes exactly the numbers ngspice is checked against. That is the
  point of the fixture: calculator and SPICE must agree on the same design.
  The requirement it is traced to (``req.step_response``) carries that same
  3.1606 V as its value - the reviewer compares nominal and requirement -
  while the 5 V step itself is the ``v_step`` parameter.

No operating point is requested, so ``domain.analog.bias`` stays NOT_VERIFIED
for this fixture (documented, not hidden).
"""

from __future__ import annotations

import math
from pathlib import Path

from ai_eda.ir import (
    AnalysisSpec,
    CircuitDomain,
    CircuitIR,
    Component,
    Expectation,
    LibraryRef,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
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
from ai_eda.tools.calc import rc_step_response, rc_time_constant
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis
from tests.fixtures_kicad import USER, pin_header, resistor

__all__ = ["PROJECT_ID", "CAPACITOR_DS", "R_OHM", "C_FARAD", "V_STEP", "capacitor", "rc_lowpass_ir"]

PROJECT_ID = "rc_lowpass"
R_OHM = 1_000.0
C_FARAD = 1e-6
V_STEP = 5.0

CAPACITOR_DS = SourceRef(
    title="Generic MLCC chip capacitor datasheet",
    authority="Vendor",
    content_hash="sha256:fixture-capacitor-datasheet",
)


def capacitor(ref: str, value: str, farad: float, library: KicadLibrary) -> Component:
    """A verified 0603 MLCC, bound in SPICE as an ideal ``C`` with its authoritative capacitance."""
    capacitance = authoritative(farad, CAPACITOR_DS, "F")
    auth = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=CAPACITOR_DS)
    return Component(
        ref=ref,
        value=value,
        description="Capacitor",
        manufacturer=authoritative("Generic", CAPACITOR_DS),
        mpn=authoritative(f"CL10B{value}KB8NNNC", CAPACITOR_DS),
        datasheet=CAPACITOR_DS,
        package=authoritative("0603", CAPACITOR_DS),
        pins=[
            Pin(number="1", name="", electrical_type=PinElectricalType.PASSIVE, provenance=auth),
            Pin(number="2", name="", electrical_type=PinElectricalType.PASSIVE, provenance=auth),
        ],
        electrical={"capacitance": capacitance},
        symbol=library.resolve_symbol(LibraryRef(library="Device", name="C")),
        footprint=library.resolve_footprint(LibraryRef(library="Capacitor_SMD", name="C_0603_1608Metric")),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="fixture", note="low-pass element"),
        serves_requirements=["req.step_response"],
        spice=SpiceBinding(
            device=SpiceDevice.C,
            value=capacitance,
            provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=CAPACITOR_DS, note="ideal capacitor at the datasheet nominal value"),
        ),
    )


def rc_lowpass_ir(tmp_path: Path, library: KicadLibrary | None = None) -> CircuitIR:
    lib = library or KicadLibrary()
    ir = CircuitIR(
        project=ProjectMeta(id=PROJECT_ID, name="RC low-pass", description="1 k / 1 u RC low-pass driven by a 5 V step", workdir=str(tmp_path))
    )
    ir.topology = Topology(name="rc low-pass", domains=[CircuitDomain.ANALOG], provenance=USER)
    ir.requirements.requirements = [
        Requirement(
            id="req.step_response", key="step_response", text="after one time constant the output reaches (1 - 1/e) of a 5 V step = 3.1606 V",
            kind=RequirementKind.EXPLICIT, value=user_requirement(V_STEP * (1.0 - math.exp(-1.0)), "V", note="(1 - 1/e) of the 5 V step"),
        ),
        Requirement(id="req.interface", key="interface", text="IN / GND on a 2-pin 2.54 mm header", kind=RequirementKind.EXPLICIT, category="mechanical"),
    ]
    ir.components = [
        resistor("R1", "1k", R_OHM, lib, serves=("req.step_response",)),
        capacitor("C1", "1u", C_FARAD, lib),
        pin_header("J1", 2, lib),
    ]
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.nets = [
        Net(name="IN", kind=NetKind.SIGNAL, pins=[PinRef(component_ref="J1", pin_number="1"), PinRef(component_ref="R1", pin_number="1")], provenance=net_p),
        Net(name="OUT", pins=[PinRef(component_ref="R1", pin_number="2"), PinRef(component_ref="C1", pin_number="1")], provenance=net_p, serves_requirements=["req.step_response"]),
        Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="J1", pin_number="2"), PinRef(component_ref="C1", pin_number="2")], provenance=net_p),
    ]
    p = ir.parameters
    p["r"] = ir.component("R1").electrical["resistance"]
    p["c"] = ir.component("C1").electrical["capacitance"]
    p["v_step"] = user_requirement(V_STEP, "V")
    p["tau"] = rc_time_constant(p["r"], p["c"], ("r", "c"))  # 1 ms, derived by calc.rc.tau
    p["v_out_at_tau"] = rc_step_response(p["v_step"], p["tau"], p["tau"], ("v_step", "tau", "tau"))  # 3.1606 V, derived by calc.rc.step_response
    ir.simulation = SimulationSetup(
        stimuli=[
            Stimulus(
                id="VIN", source="voltage", net="IN", reference_net="GND", kind=StimulusKind.PULSE,
                params={
                    "v1": user_requirement(0.0, "V"), "v2": user_requirement(V_STEP, "V"), "td": user_requirement(0.0, "s"),
                    "tr": user_requirement(1e-9, "s"), "tf": user_requirement(1e-9, "s"), "pw": user_requirement(1.0, "s"), "per": user_requirement(2.0, "s"),
                },
                provenance=USER, serves_requirements=["req.step_response"],
            )
        ],
        analyses=[
            AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(1e-6, "s"), "stop": user_requirement(5e-3, "s")}, provenance=USER),
        ],
        expectations=[
            Expectation(
                id="v_out_tau", analysis_id="tran", vector="v(OUT)", reduce=Reduce.AT, at=p["tau"], nominal=p["v_out_at_tau"],
                tol_rel=user_requirement(0.02, note="2 %: ngspice's trapezoidal integration vs the closed form"), requirement_id="req.step_response", provenance=USER,
            )
        ],
    )
    return ir
