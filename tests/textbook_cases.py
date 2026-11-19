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


def dual_amplifier(workdir: Path, library: KicadLibrary, project_id: str = "dual_amplifier") -> CircuitIR:
    """One MCP6002-xSN dual op-amp: unit A an inverting stage (R1 10k in, R2 100k feedback), unit B a follower whose '+' is on
    GND (the spare tied as the IR states), unit C its supply pins; a 1x04 header for IN, OUT, VCC and GND. Every part has a
    verified footprint, so the board compiles too.

    The op-amp pins are found in the library symbol by unit and by name ('+', '-', V+, V-) or, for the unnamed outputs, by
    type: library facts, not checked against the manufacturer's datasheet."""
    from ai_eda.ir import Component, LibraryRef, Net, Pin, PinRef, Provenance, ProvenanceKind, Topology

    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="dual amplifier fixture")
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id, workdir=str(workdir)),
                   topology=Topology(name="dual amplifier", provenance=prov))
    symbols = {}
    for ref, value, (lib, name), (fp_lib, fp_name) in (
            ("R1", "10k", ("Device", "R"), ("Resistor_SMD", "R_0603_1608Metric")),
            ("R2", "100k", ("Device", "R"), ("Resistor_SMD", "R_0603_1608Metric")),
            ("U1", "MCP6002", ("Amplifier_Operational", "MCP6002-xSN"), ("Package_SO", "SOIC-8_3.9x4.9mm_P1.27mm")),
            ("J1", "I/O", ("Connector_Generic", "Conn_01x04"), ("Connector_PinHeader_2.54mm", "PinHeader_1x04_P2.54mm_Vertical"))):
        libref = library.resolve_symbol(LibraryRef(library=lib, name=name))
        symbols[ref] = symbol = library.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       footprint=library.resolve_footprint(LibraryRef(library=fp_lib, name=fp_name)),
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))

    def u1(unit: int, name: str = "", etype: str = "") -> str:
        (pin,) = [p for p in symbols["U1"].pins if p.unit == unit and (p.electrical_type == etype if etype else p.name == name)]
        return f"U1.{pin.number}"

    for name, kind, pins in (("IN", "signal", ("J1.1", "R1.1")),
                             ("SUM", "signal", ("R1.2", "R2.1", u1(1, "-"))),
                             ("OUT", "signal", ("J1.2", "R2.2", u1(1, etype="output"))),
                             ("SPARE", "signal", (u1(2, "-"), u1(2, etype="output"))),
                             ("VCC", "power", ("J1.3", u1(3, "V+"))),
                             ("GND", "ground", ("J1.4", u1(1, "+"), u1(2, "+"), u1(3, "V-")))):
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split('.')[0], pin_number=p.split('.')[1]) for p in pins]))
    return ir


#: the op-amp stage variants of :func:`amplifier_variant`
AMPLIFIER_VARIANTS = ("fb_cap", "bias_r", "dual_supply", "noninv")


def amplifier_variant(workdir: Path, library: KicadLibrary, name: str) -> CircuitIR:
    """:func:`amplifier` changed into one op-amp stage variant: ``fb_cap`` a capacitor C1 across the feedback resistor R2,
    ``bias_r`` the '+' input to ground through R3, ``dual_supply`` V- on its own negative rail VEE from a 5-pin connector,
    ``noninv`` a non-inverting stage (IN on '+', R1 from OUT to FB on '-', R2 from FB to ground)."""
    from ai_eda.ir import Component, LibraryRef, Net, Pin, PinRef

    ir = amplifier(workdir, library)
    ir.project.id = ir.project.name = f"amplifier_{name}"
    prov = ir.components[0].provenance

    def add(ref: str, value: str, lib: str, symbol_name: str) -> None:
        libref = library.resolve_symbol(LibraryRef(library=lib, name=symbol_name))
        symbol = library.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))

    def move(pin: str, to: str, kind: str = "signal") -> None:
        ref, number = pin.split(".")
        for net in ir.nets:
            net.pins = [p for p in net.pins if (p.component_ref, p.pin_number) != (ref, number)]
        target = next((n for n in ir.nets if n.name == to), None)
        if target is None:
            target = Net(name=to, kind=kind, provenance=prov, pins=[])
            ir.nets.append(target)
        target.pins.append(PinRef(component_ref=ref, pin_number=number))

    if name == "fb_cap":
        add("C1", "100p", "Device", "C")
        move("C1.1", "SUM")
        move("C1.2", "OUT")
    elif name == "bias_r":
        add("R3", "10k", "Device", "R")
        move("U1.3", "PLUS")
        move("R3.1", "PLUS")
        move("R3.2", "GND")
    elif name == "dual_supply":
        ir.components = [c for c in ir.components if c.ref != "J1"]
        add("J1", "I/O", "Connector_Generic", "Conn_01x05")
        move("U1.4", "VEE", "power")
        move("J1.5", "VEE", "power")
    elif name == "noninv":
        for net in ir.nets:
            net.pins = [p for p in net.pins if p.component_ref not in ("R1", "R2")]
        move("U1.3", "IN")
        move("U1.2", "FB")
        move("R1.1", "OUT")
        move("R1.2", "FB")
        move("R2.1", "FB")
        move("R2.2", "GND")
    else:
        raise ValueError(name)
    ir.nets = [n for n in ir.nets if n.pins]
    return ir


def oscillator_bank(workdir: Path, library: KicadLibrary, project_id: str = "oscillator_bank") -> CircuitIR:
    """Three op-amp relaxation oscillators of one quad op-amp (MCP6004 units A-C, unit D a spare follower on ground) and a
    push-pull output stage (MCP6002 unit A driving a 2N3904 / 2N3906 pair inside its loop, unit B a spare).

    Each oscillator: C1x from '-' to ground, R1x feedback from the output to '-', '+' on the wiper of trimmer RVx whose
    ends run through R2x to the output and through R3x to the shared reference VREF (R1 / R2 / C1 from VCC). The output
    stage: '+' from the header, the followers' emitters on '-' (the loop closes there), R7 between the base and emitter
    nodes, the load R8 and C20 in series to the speaker terminal J2.

    Pins are found in the library symbols by unit and name or type: library facts, not checked against a datasheet."""
    from ai_eda.ir import Block, Component, LibraryRef, Net, Pin, PinRef, Provenance, ProvenanceKind, Topology

    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="oscillator bank fixture")
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id, workdir=str(workdir)),
                   topology=Topology(name="oscillator bank", provenance=prov))
    symbols = {}
    parts = [("U1", "MCP6004", "Amplifier_Operational", "MCP6004"), ("U2", "MCP6002", "Amplifier_Operational", "MCP6002-xSN"),
             ("R1", "10k", "Device", "R"), ("R2", "10k", "Device", "R"), ("C1", "10u", "Device", "C"),
             ("Q1", "2N3904", "Transistor_BJT", "2N3904"), ("Q2", "2N3906", "Transistor_BJT", "2N3906"),
             ("R7", "1k", "Device", "R"), ("R8", "6.8", "Device", "R"), ("C20", "470u", "Device", "C"),
             ("J1", "IO", "Connector_Generic", "Conn_01x06"), ("J2", "Speaker", "Connector", "Screw_Terminal_01x02")]
    for i in (1, 2, 3):
        parts += [(f"C1{i}", "47n", "Device", "C"), (f"R1{i}", "47k", "Device", "R"), (f"RV{i}", "20k", "Device", "R_Potentiometer_Trim"),
                  (f"R2{i}", "220k", "Device", "R"), (f"R3{i}", "100k", "Device", "R")]
    for ref, value, lib, name in parts:
        libref = library.resolve_symbol(LibraryRef(library=lib, name=name))
        symbols[ref] = symbol = library.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))

    def unit(ref: str, u: int, name: str = "", etype: str = "") -> str:
        (pin,) = [p for p in symbols[ref].pins if p.unit == u and (p.electrical_type == etype if etype else p.name == name)]
        return f"{ref}.{pin.number}"

    def named(ref: str, name: str) -> str:
        (pin,) = [p for p in symbols[ref].pins if p.name == name]
        return f"{ref}.{pin.number}"

    nets: dict[str, tuple[str, list[str]]] = {"VCC": ("power", ["J1.1"]), "GND": ("ground", ["J1.2"]), "VREF": ("signal", [])}

    def join(net: str, *pins: str) -> None:
        nets.setdefault(net, ("signal", []))[1].extend(pins)

    power_u1 = max(p.unit for p in symbols["U1"].pins)
    join("VCC", unit("U1", power_u1, "V+"), unit("U2", 3, "V+"), "R1.1", named("Q1", "C"))
    join("GND", unit("U1", power_u1, "V-"), unit("U2", 3, "V-"), "R2.2", "C1.2", named("Q2", "C"), "J2.2", unit("U1", 4, "+"), unit("U2", 2, "+"))
    join("VREF", "R1.2", "R2.1", "C1.1")
    for i in (1, 2, 3):
        join(f"OSC{i}", unit("U1", i, etype="output"), f"R1{i}.1", f"R2{i}.1", f"J1.{i + 2}")
        join(f"INM{i}", unit("U1", i, "-"), f"R1{i}.2", f"C1{i}.1")
        join("GND", f"C1{i}.2")
        join(f"INP{i}", unit("U1", i, "+"), f"RV{i}.2")
        join(f"HF{i}", f"RV{i}.1", f"R2{i}.2")
        join(f"HH{i}", f"RV{i}.3", f"R3{i}.1")
        join("VREF", f"R3{i}.2")
    join("SPARE1", unit("U1", 4, "-"), unit("U1", 4, etype="output"))
    join("SPARE2", unit("U2", 2, "-"), unit("U2", 2, etype="output"))
    join("IN", "J1.6", unit("U2", 1, "+"))
    join("DRV", unit("U2", 1, etype="output"), named("Q1", "B"), named("Q2", "B"), "R7.1")
    join("SPK", unit("U2", 1, "-"), named("Q1", "E"), named("Q2", "E"), "R7.2", "R8.1")
    join("SPKC", "R8.2", "C20.1")
    join("SPKP", "C20.2", "J2.1")
    for name, (kind, pins) in nets.items():
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split(".")[0], pin_number=p.split(".")[1]) for p in pins]))
    voices = ["U1"] + [f"{k}{i}" for i in (1, 2, 3) for k in ("C1", "R1", "RV", "R2", "R3")]
    for bid, refs, ins, outs in (("io", ["J1"], [], ["IN", "OSC1", "OSC2", "OSC3"]), ("reference", ["R1", "R2", "C1"], [], ["VREF"]),
                                 ("voices", voices, ["VREF"], ["OSC1", "OSC2", "OSC3"]),
                                 ("driver", ["U2", "Q1", "Q2", "R7", "R8", "C20", "J2"], ["IN"], [])):
        ir.topology.blocks.append(Block(id=bid, function=bid, component_refs=refs, input_nets=ins, output_nets=outs, provenance=prov))
    return ir


def op_amp_piano(workdir: Path, library: KicadLibrary, project_id: str = "op_amp_piano") -> CircuitIR:
    """An eight-voice op-amp chord piano on a 9 V battery, laid out like the boards it stands for.

    - power: battery J1; reverse protection and soft power switch by two back-to-back P-MOSFETs Q3 / Q4 in the battery
      line (gate network R50 || C23 gate to source, C22 to VBAT, R51 to the slide switch SW9 whose throws go to ground and
      through R52 back to the source); C1 / C20 on VBAT; regulator U4 to VDD with C5, C21 + R9, R10 + LED D1; R1 to the
      oscillator rail VDDO with C2 / C3 / C4.
    - reference: R2 / R3 / C6, VREF = VDDO / 2.
    - voices_low / voices_high: two LM324 quads, four relaxation oscillators each (C1x from '-' to ground, R1x feedback,
      '+' on the wiper of trimmer RVx between R2x to the output and R3x to VREF).
    - keyboard: eight keys SWk in series with R4k from OSCk onto SUM, summed by TL082 unit A (feedback R4, '+' on VREF).
    - mixer_tone: volume RV9 (a potentiometer with a mounting pin on ground; its far end through C7 to ground), then a
      Sallen-Key low pass R5 / R6 / C9 with C8 back from the speaker node.
    - driver: TL082 unit B with BC337 / BC327 followers inside its loop, R7 between the nodes, R8 and C10 to the speaker J2.

    Pins are found in the library symbols by unit and name or type: library facts, not checked against a datasheet."""
    from ai_eda.ir import Block, Component, LibraryRef, Net, Pin, PinRef, Provenance, ProvenanceKind, Topology

    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="op-amp piano fixture")
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id, workdir=str(workdir)),
                   topology=Topology(name="op-amp piano", provenance=prov))
    symbols = {}
    parts = [("J1", "Battery_9V", "Connector_Generic", "Conn_01x02"), ("Q3", "IRF4905", "Transistor_FET", "IRF4905"),
             ("Q4", "IRF4905", "Transistor_FET", "IRF4905"), ("R50", "2M2", "Device", "R"), ("C23", "470n", "Device", "C"),
             ("C22", "47n", "Device", "C"), ("R51", "100k", "Device", "R"), ("SW9", "SW_SPDT", "Switch", "SW_SPDT"),
             ("R52", "10k", "Device", "R"), ("C1", "470u", "Device", "C_Polarized"), ("C20", "1u", "Device", "C"),
             ("U4", "LP2950-50", "Regulator_Linear", "LP2950-5.0_TO92"), ("C5", "100n", "Device", "C"), ("C21", "3u3", "Device", "C"),
             ("R9", "1", "Device", "R"), ("R10", "6k8", "Device", "R"), ("D1", "LED", "Device", "LED"), ("R1", "56", "Device", "R"),
             ("C2", "470u", "Device", "C_Polarized"), ("C3", "100n", "Device", "C"), ("C4", "100n", "Device", "C"),
             ("R2", "10k", "Device", "R"), ("R3", "10k", "Device", "R"), ("C6", "33u", "Device", "C_Polarized"),
             ("U1", "LM324", "Amplifier_Operational", "LM324"), ("U2", "LM324", "Amplifier_Operational", "LM324"),
             ("U3", "TL082", "Amplifier_Operational", "TL082"), ("R4", "806", "Device", "R"),
             ("RV9", "50k A", "Device", "R_Potentiometer_MountingPin"), ("C7", "4u7", "Device", "C"), ("R5", "10k7", "Device", "R"),
             ("R6", "10k7", "Device", "R"), ("C8", "22n", "Device", "C"), ("C9", "10n", "Device", "C"),
             ("Q1", "BC337", "Transistor_BJT", "BC337"), ("Q2", "BC327", "Transistor_BJT", "BC327"), ("R7", "1k", "Device", "R"),
             ("R8", "1.8", "Device", "R"), ("C10", "470u", "Device", "C_Polarized"), ("J2", "Speaker_8R", "Connector", "Screw_Terminal_01x02")]
    feedback = ("59k", "52k3", "46k4", "44k2", "39k2", "34k8", "30k9", "29k4")
    for i in range(1, 9):
        parts += [(f"C1{i}", "47n", "Device", "C"), (f"R1{i}", feedback[i - 1], "Device", "R"), (f"R2{i}", "210k", "Device", "R"),
                  (f"RV{i}", "20k", "Device", "R_Potentiometer_Trim"), (f"R3{i}", "100k", "Device", "R"),
                  (f"SW{i}", "SW_Push", "Switch", "SW_Push"), (f"R4{i}", "15k", "Device", "R")]
    for ref, value, lib, name in parts:
        libref = library.resolve_symbol(LibraryRef(library=lib, name=name))
        symbols[ref] = symbol = library.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))

    def unit(ref: str, u: int, name: str = "", etype: str = "") -> str:
        (pin,) = [p for p in symbols[ref].pins if p.unit == u and (p.electrical_type == etype if etype else p.name == name)]
        return f"{ref}.{pin.number}"

    def named(ref: str, name: str) -> str:
        (pin,) = [p for p in symbols[ref].pins if p.name == name]
        return f"{ref}.{pin.number}"

    nets: dict[str, tuple[str, list[str]]] = {"BATP": ("power", []), "VBAT": ("power", []), "VDD": ("power", []), "VDDO": ("power", []),
                                              "GND": ("ground", [])}

    def join(net: str, *pins: str) -> None:
        nets.setdefault(net, ("signal", []))[1].extend(pins)

    join("BATP", "J1.1", named("Q3", "D"))
    join("GND", "J1.2")
    join("VS", named("Q3", "S"), named("Q4", "S"), "R50.1", "C23.1", "R52.2")
    join("VG", named("Q3", "G"), named("Q4", "G"), "R50.2", "C23.2", "C22.1", "R51.1")
    join("SWP", "R51.2", named("SW9", "B"))
    join("SWOFF", named("SW9", "C"), "R52.1")
    join("GND", named("SW9", "A"))
    join("VBAT", named("Q4", "D"), "C22.2", "C1.1", "C20.1", named("U4", "VI"), named("Q1", "C"))
    join("GND", "C1.2", "C20.2", named("U4", "GND"))
    join("VDD", named("U4", "VO"), "C5.1", "C21.1", "R10.1", "R1.1", unit("U3", 3, "V+"))
    join("LDOC", "C21.2", "R9.1")
    join("LEDA", "R10.2", named("D1", "A"))
    join("GND", "C5.2", "R9.2", named("D1", "K"), unit("U3", 3, "V-"))
    join("VDDO", "R1.2", "C2.1", "C3.1", "C4.1", unit("U1", 5, "V+"), unit("U2", 5, "V+"), "R2.1")
    join("GND", "C2.2", "C3.2", "C4.2", unit("U1", 5, "V-"), unit("U2", 5, "V-"), "R3.2", "C6.2")
    join("VREF", "R2.2", "R3.1", "C6.1", unit("U3", 1, "+"))
    for i in range(1, 9):
        u, n = ("U1", i) if i <= 4 else ("U2", i - 4)
        join(f"OSC{i}", unit(u, n, etype="output"), f"R1{i}.1", f"R2{i}.1", f"SW{i}.1")
        join(f"INM{i}", unit(u, n, "-"), f"R1{i}.2", f"C1{i}.1")
        join("GND", f"C1{i}.2")
        join(f"INP{i}", unit(u, n, "+"), f"RV{i}.2")
        join(f"HF{i}", f"RV{i}.1", f"R2{i}.2")
        join(f"HH{i}", f"RV{i}.3", f"R3{i}.1")
        join("VREF", f"R3{i}.2")
        join(f"KEY{i}", f"SW{i}.2", f"R4{i}.1")
        join("SUM", f"R4{i}.2")
    join("SUM", unit("U3", 1, "-"), "R4.2")
    join("MIX", unit("U3", 1, etype="output"), "R4.1", "RV9.3")
    join("VOLC", "RV9.1", "C7.1")
    join("GND", "C7.2", "RV9.MP", "C9.2", named("Q2", "C"), "J2.2")
    join("WIP", "RV9.2", "R5.1")
    join("SKA", "R5.2", "R6.1", "C8.1")
    join("SKB", "R6.2", "C9.1", unit("U3", 2, "+"))
    join("DRV", unit("U3", 2, etype="output"), named("Q1", "B"), named("Q2", "B"), "R7.1")
    join("SPK", named("Q1", "E"), named("Q2", "E"), "R7.2", "R8.1", "C8.2", unit("U3", 2, "-"))
    join("SPKC", "R8.2", "C10.1")
    join("SPKP", "C10.2", "J2.1")
    for name, (kind, pins) in nets.items():
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split(".")[0], pin_number=p.split(".")[1]) for p in pins]))

    def voices(r: range) -> list[str]:
        return [f"{k}{i}" for i in r for k in ("R1", "C1", "R2", "RV", "R3")]

    for bid, refs, ins, outs in (
            ("power", ["J1", "Q3", "Q4", "R50", "R51", "R52", "C22", "C23", "SW9", "C1", "C20", "U4", "C21", "R9", "R10", "D1", "R1",
                       "C2", "C3", "C4", "C5"], ["BATP"], ["VBAT", "VDD", "VDDO"]),
            ("reference", ["R2", "R3", "C6"], ["VDDO"], ["VREF"]),
            ("voices_low", ["U1", *voices(range(1, 5))], ["VDDO", "VREF"], [f"OSC{i}" for i in range(1, 5)]),
            ("voices_high", ["U2", *voices(range(5, 9))], ["VDDO", "VREF"], [f"OSC{i}" for i in range(5, 9)]),
            ("keyboard", [f"{k}{i}" for i in range(1, 9) for k in ("SW", "R4")], [f"OSC{i}" for i in range(1, 9)], ["SUM"]),
            ("mixer_tone", ["U3", "R4", "RV9", "C7", "R5", "R6", "C8", "C9"], ["SUM", "VREF"], ["DRV"]),
            ("driver", ["Q1", "Q2", "R7", "R8", "C10", "J2"], ["DRV", "VBAT"], ["SPKP"])):
        ir.topology.blocks.append(Block(id=bid, function=bid, component_refs=refs, input_nets=ins, output_nets=outs, provenance=prov))
    return ir
