"""Small circuits shared by schematic policy checks and render exports."""

from pathlib import Path

from ai_eda.agents import AgentContext
from ai_eda.ir import CircuitIR, ProjectMeta
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.workflow import Orchestrator, Stage

CASES = {
    "divider": {"input_voltage": "12 V", "output_voltage": "5 V"},
    "led": {"input_voltage": "5 V", "led_forward_voltage": "2 V", "led_forward_current": "10 mA"},
    "rc_lowpass": {"cutoff_frequency": "1 kHz"},
    "lc_lowpass": {"cutoff_frequency": "100 kHz", "system_impedance": "50 ohm"},
    "bjt_lowside_switch": {"input_voltage": "5 V", "output_voltage": "12 V", "output_current": "50 mA", "max_voltage_drop": "0.3 V"},
    "astable": {"input_voltage": "5 V", "oscillation_frequency": "1 kHz"},
    "i2c_pullup": {"input_voltage": "3.3 V", "clock_frequency": "400 kHz", "bus_capacitance": "150 pF", "bus_rise_time": "300 ns"},
}


def circuit(name: str, workdir: Path, library: KicadLibrary) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(workdir)))
    tools = {"kicad_library": library}
    Orchestrator(AgentContext(workdir=workdir, tools=tools, answers={"application": "bench", "jurisdiction": "EU", **CASES[name]})).run(ir, stop_after=Stage.ARCHITECTURE)
    state = Orchestrator(AgentContext(workdir=workdir, tools=tools, answers={"confirm_design": "yes"})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert ir.components and not state.blocked, state
    return ir


def amplifier(workdir: Path, library: KicadLibrary) -> CircuitIR:
    """One library op-amp with an input resistor and a feedback resistor."""
    from ai_eda.ir import Component, LibraryRef, Net, Pin, PinElectricalType, PinRef, Provenance, ProvenanceKind, Topology

    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="amplifier fixture")
    ir = CircuitIR(project=ProjectMeta(id="amplifier", name="amplifier", workdir=str(workdir)),
                   topology=Topology(name="amplifier", provenance=prov))
    for ref, value, lib, name in (("R1", "10k", "Device", "R"), ("R2", "100k", "Device", "R"),
                                  ("U1", "TL081", "Amplifier_Operational", "TL081"),
                                  ("J1", "I/O", "Connector_Generic", "Conn_01x04")):
        libref = library.resolve_symbol(LibraryRef(library=lib, name=name))
        symbol = library.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=(PinElectricalType.NO_CONNECT
                                             if ref == "U1" and p.number in {"1", "5", "8"} else p.ir_type), provenance=prov)
                                             for p in symbol.pins]))
    for name, kind, pins in (("IN", "signal", ("J1.1", "R1.1")),
                             ("SUM", "signal", ("R1.2", "R2.1", "U1.2")),
                             ("OUT", "signal", ("J1.2", "R2.2", "U1.6")),
                             ("VCC", "power", ("J1.3", "U1.7")),
                             ("GND", "ground", ("J1.4", "U1.3", "U1.4"))):
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split('.')[0], pin_number=p.split('.')[1]) for p in pins]))
    return ir
