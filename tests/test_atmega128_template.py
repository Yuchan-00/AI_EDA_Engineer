"""The ``atmega128_devboard`` template on a synthetic KiCad library (``tests/fixtures_atmega.py``).

What is checked here, offline: the confirm_design table (every input,
choice, calculator output, part, net and simulation line), what a
confirmation applies (every one of the 64 U1 pins in exactly one net, found
by library pin *name*; PE0 / PE1 shared by the ISP and UART0 headers; the
SPICE exclusions with their reasons; the stimuli, the C8 initial condition,
``op`` + ``tran ... uic`` and the three expectations), the compiled netlist
text, the refusals (supply 5 V / 20 V, clock 20 MHz / 500 kHz, a renamed MCU
or regulator pin, a missing symbol, a missing input, an unserved
requirement), ``calc.recompute`` PASS, the inputs check PASS, hash
determinism, the input aliases and the theory text / figures / part notes.
Where ngspice is found (``needs_dll``) the SPICE stage runs against the real
engine and the three expectations must PASS, and the whole pipeline runs to
RELEASE on the stacked-pin library (``pcb.routing=skip``: the placement is
the core ring, and what is true about the copper - none - is asserted:
``pcb.routing.connectivity`` FAIL, RELEASE FAIL, ERC / DRC NOT_VERIFIED).
Without ngspice the same run checks the four stage reports (figures, the
core-ring rule) and the GUI previews (the 64-pin symbol with its three
hidden stacked pins, the board with all 64 TQFP pads and no overlapping
labels). The real KiCad 10.0.6 symbol
stacks its hidden pins 52 / 53 / 63 on 21 / 22; the schematic compiler
accepts that stack because the IR puts the pins in one net (one stub and one
label per point) and refuses it when the IR splits it. Nothing here claims
the MCU works: it is not simulated.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ai_eda.agents import CircuitDesignAgent
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.agents.keys import PLACEMENT_KEY, ROUTING_KEY
from ai_eda.compilers import BOMCompiler, CompileContext, SchematicCompiler
from ai_eda.compilers.schematic_layout import pin_position
from ai_eda.compilers.spice import analysis_command
from ai_eda.design import CHOICE_NOTE_PREFIX, INPUTS_CHECK, NO_RECORD, TEMPLATE_VERSION, UNVERIFIED_SUBSTITUTE, check_inputs_vs_requirements, read_inputs
from ai_eda.design.atmega128 import ISP_PINS, PORTS, UART_PINS, Atmega128DevboardTemplate, mcu_pin_names
from ai_eda.design.templates import THEORY_CURVE_NOTE
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Net, NetKind, PinRef, ProvenanceKind, Reduce, Requirement, RequirementKind, ValidationStatus as S, user_requirement
from ai_eda.gui.preview import board_svg, project_schematic_svg
from ai_eda.report.figures import _text_width
from ai_eda.report.stages import build_stage_document
from ai_eda.tools.calc import CALC_VERSION, part_value_agrees, recompute_parameters
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import footprint_bbox, pads_bbox
from ai_eda.tools.placement.core_ring import MARGIN_MM
from ai_eda.tools.routing import RoutingParams
from ai_eda.workflow import Stage
from tests.fixtures_atmega import ATMEGA128_PINS, atmega_library
from tests.test_report_figures import labels_on_foreign_copper
from tests.test_circuit_templates import BASE, _confirm, _ctx, _ir, _netlist, _present, _run, _validate, needs_dll

ATMEGA = {"input_voltage": "9 V", "clock_frequency": "16 MHz"}
TEMPLATE = Atmega128DevboardTemplate()
#: v(RESET) one time constant after the step: 5 V (1 - e^-1)
V_RESET_TAU = 5.0 * (1.0 - math.exp(-1.0))
#: the parts the netlist leaves out (no model, the unsimulated input side, DC-floating nodes, connectors, the button)
EXCLUDED = ["U1", "U2", "J1", "D1", "C1", "D2", "C7", "SW1", "Y1", "C9", "C10", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11"]
REFS = ["U1", "U2", "J1", "D1", "C1", "C2", "C3", "D2", "R1", "C4", "C5", "L1", "C6", "C7", "R2", "C8", "SW1", "R3", "Y1", "C9", "C10",
        "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11"]
PORT_NETS = [f"P{letter}{bit}" for letter, _, bits in PORTS for bit in range(bits)]
NETLIST = (
    "atm\nC2 +5V 0 1e-5\nC3 +5V 0 1e-7\nC4 +5V 0 1e-7\nC5 +5V 0 1e-7\nC6 AVCC 0 1e-7\nC8 RESET 0 1e-7 ic=0\nL1 +5V AVCC 1e-5\n"
    "R1 +5V LED_A 1.5k\nR2 +5V RESET 10k\nR3 +5V PEN 10k\nVV5V +5V 0 DC 5\nVVLED LED_A 0 DC 2\n.end\n"
)


def _lib(tmp_path: Path, **kw):
    return atmega_library(tmp_path / "kicad", **kw)


def _built(tmp_path: Path, stop_after: Stage | None = Stage.ARCHITECTURE, answers: dict[str, str] | None = None, *, spice: bool = False):
    """Present the table on the synthetic library, then confirm it (``stop_after`` for the confirming run)."""
    lib = _lib(tmp_path)
    ir = _ir(tmp_path, "atm")
    _present(ir, tmp_path, lib, answers or ATMEGA)
    state, ctx = _confirm(ir, tmp_path, lib, stop_after, spice=spice)
    return ir, lib, state, ctx


def _pin_to_net(ir: CircuitIR) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for n in ir.nets:
        for p in n.pins:
            out.setdefault((p.component_ref, p.pin_number), []).append(n.name)
    return out


# --------------------------------------------------------------------------- the table


def test_table_lists_every_input_choice_computed_value_part_net_and_simulation_line(tmp_path: Path):
    lib = _lib(tmp_path)
    ir = _ir(tmp_path, "atm")
    question = _present(ir, tmp_path, lib, ATMEGA)
    assert question.startswith(f"Template 'atmega128_devboard' v{TEMPLATE_VERSION} (ATmega128 development board)")
    assert "req.clock_frequency: clock_frequency = 16000000 Hz (stated as '16 MHz')" in question and "req.input_voltage: input_voltage = 9 V (stated as '9 V')" in question
    for line in (
        "v_out_reg = 5 V - the L7805's nominal output voltage", "i_load_budget = 0.05 A - design load budget of the +5V rail", "p_reg_max = 1 W - no-heatsink dissipation budget",
        "c_in = 1e-05 F - C1: input bulk electrolytic capacitor", "c_out = 1e-05 F - C2: output bulk electrolytic capacitor", "c_dec = 1e-07 F - ceramic 100 nF",
        "l_avcc = 1e-05 H - L1: AVCC filter inductor", "c_xtal = 2.2e-11 F - C9 / C10: crystal load capacitors", "c_stray = 4e-12 F - stray capacitance",
        "r_reset = 10000 ohm - R2: ~RESET pull-up", "c_reset = 1e-07 F - C8: ~RESET capacitor", "c8_ic = 0 V - initial voltage across C8 for the transient (uic",
        "r_pen = 10000 ohm - R3: ~PEN pull-up", "v_f_led = 2 V - forward voltage of the power LED D2", "i_led = 0.002 A - power LED current",
        "tol_rel = 0.02 - relative tolerance of the i_led and v_reset_tau expectations", "v_avcc_tol_abs = 0.001 V - absolute tolerance of the v(AVCC) expectation",
        "regulator_model: U2 L7805 has no SPICE model in the KiCad libraries: it is excluded from the netlist and its 5 V output is the ideal stimulus V5V",
        "led_model: ideal constant-V_f LED: D2 is excluded from the netlist and replaced by the stimulus VLED = v_f_led",
        "mcu_model: U1 has no SPICE model: excluded; its pins are wired in the schematic and checked structurally only (nothing simulates or claims that the MCU runs)",
        "jack_pinout: J1 Connector:Barrel_Jack_Switch pins carry no names in the KiCad library, so its polarity is a pin-number assumption: pin 1 = centre pin (+, VIN_RAW)",
        "cp_polarity: C1 / C2 Device:C_Polarized pins carry no names in the KiCad library, so their polarity is a pin-number assumption: pin 1 = +",
        # every calculator output, with its tool and the ids that filled its roles
        "p_reg = 0.2 W [calc.regulator.p_dissipation from v_in, v_out_reg, i_load_budget]", "r_led = 1500 ohm [calc.led.R from v_out_reg, v_f_led, i_led]",
        "i_led_design = 0.002 A [calc.led.I from v_out_reg, v_f_led, r_led]", "tau_reset = 0.001 s [calc.rc.tau from r_reset, c_reset]",
        f"v_reset_tau = {V_RESET_TAU:.12g} V [calc.rc.step_response from v_out_reg, tau_reset, tau_reset]", "tran_step = 1e-05 s [calc.rc.tran_step from tau_reset]",
        "tran_stop = 0.005 s [calc.rc.tran_stop from tau_reset]", "c_load = 1.5e-11 F [calc.crystal.load_capacitance from c_xtal, c_xtal, c_stray]",
        f"f_avcc = {1.0 / (2 * math.pi * math.sqrt(1e-12)):.12g} Hz [calc.lc.cutoff from l_avcc, c_dec]",
        # parts from the library on disk
        "U1 MCU_Microchip_ATmega:ATmega128-16A / Package_QFP:TQFP-64_14x14mm_P0.8mm, value ATmega128-16A (pins " + ", ".join(str(k) for k in range(1, 65)) + " from ",
        "U2 Regulator_Linear:L7805 / Package_TO_SOT_THT:TO-220-3_Vertical, value L7805", "J1 Connector:Barrel_Jack_Switch / Connector_BarrelJack:BarrelJack_Horizontal",
        "D1 Device:D / Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal, value 1N4007", "C1 Device:C_Polarized / Capacitor_THT:CP_Radial_D5.0mm_P2.50mm, value 10u",
        "R1 Device:R / Resistor_SMD:R_0805_2012Metric, value 1.5k", "D2 Device:LED / LED_SMD:LED_0805_2012Metric, value LED", "L1 Device:L / Inductor_SMD:L_0805_2012Metric, value 10u",
        "C8 Device:C / Capacitor_SMD:C_0603_1608Metric, value 100n", "SW1 Switch:SW_Push / Button_Switch_THT:SW_PUSH_6mm", "Y1 Device:Crystal / Crystal:Crystal_HC49-4H_Vertical, value 16MHz",
        "C9 Device:C / Capacitor_SMD:C_0603_1608Metric, value 22p", "J2 Connector_Generic:Conn_02x03_Odd_Even / Connector_PinHeader_2.54mm:PinHeader_2x03_P2.54mm_Vertical",
        "J3 Connector_Generic:Conn_01x04 / Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical", "J9 Connector_Generic:Conn_01x08 / Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical",
        "J10 Connector_Generic:Conn_01x05 / Connector_PinHeader_2.54mm:PinHeader_1x05_P2.54mm_Vertical", "J11 Connector_Generic:Conn_01x02 / Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        # nets, pins found by name (D1 / D2: 1 = K, 2 = A; U2: 1 IN, 2 GND, 3 OUT; U1 by the ATmega128 names)
        "VIN_RAW (power): J1.1, D1.2", "VIN (power): D1.1, C1.1, U2.1",
        "+5V (power): U2.3, C2.1, C3.1, R1.1, U1.21, U1.52, C4.1, C5.1, L1.1, R2.1, R3.1, J2.2, J3.2, J11.1",
        "GND (ground): J1.2, J1.3, C1.2, U2.2, C2.2, C3.2, D2.1, U1.22, U1.53, U1.63, C4.2, C5.2, C6.2, C7.2, C8.2, SW1.2, C9.2, C10.2, J2.6, J3.1, J11.2",
        "LED_A (signal): R1.2, D2.2", "AVCC (power): L1.2, C6.1, U1.64", "AREF (analog): C7.1, U1.62", "RESET (signal): R2.2, C8.1, SW1.1, U1.20, J2.5", "PEN (signal): R3.2, U1.1",
        "XTAL1 (signal): Y1.1, C9.1, U1.24", "XTAL2 (signal): Y1.2, C10.1, U1.23",
        "PA0 (signal): U1.51, J4.1", "PA7 (signal): U1.44, J4.8", "PB1 (signal): U1.11, J5.2, J2.3", "PE0 (signal): U1.2, J8.1, J2.4, J3.3", "PE1 (signal): U1.3, J8.2, J2.1, J3.4",
        "PF0 (signal): U1.61, J9.1", "PG4 (signal): U1.19, J10.5",
        # the simulation
        "stimuli V5V = 5 V on +5V (the ideal L7805 output), VLED = 2 V on LED_A (the ideal LED); C8 ic = 0 V; analyses op and tran 1e-05 0.005 uic (s)",
        "expectation i_led: i(VLED) = 0.002 A +/- 2% at the operating point", f"expectation v_reset_tau: v(RESET) at t = tau_reset = 0.001 s = {V_RESET_TAU:.12g} V +/- 2% on the transient",
        "expectation v_avcc: v(AVCC) = 5 V +/- 0.001 V at the operating point", "excluded from the netlist (21): " + ", ".join(EXCLUDED),
        "structural only, never simulated: U1 itself, the crystal oscillator (Y1, C9, C10)",
    ):
        assert line in question, line
    assert len(re.findall(r"^  P[A-G][0-7] \(signal\): U1\.\d+, J\d+\.\d", question, re.M)) == 53  # one net per port bit, U1 first, then its header
    assert [q.key for q in CircuitDesignAgent().run(ir, _ctx(tmp_path, lib, {})).questions] == [CONFIRM_DESIGN_KEY]  # no advisory question
    assert ir.components == [] and ir.parameters == {} and ir.simulation is None


# --------------------------------------------------------------------------- what a confirmation applies


def test_confirm_applies_the_board_with_every_mcu_pin_in_exactly_one_net(tmp_path: Path):
    ir, lib, state, _ = _built(tmp_path)
    out = state.outcome(Stage.ARCHITECTURE)
    assert out.status is S.NOT_VERIFIED and out.message.startswith(f"129 proposal(s) applied, nothing verified; template atmega128_devboard v{TEMPLATE_VERSION} confirmed by the user: 129 proposal(s)")
    assert [c.ref for c in ir.components] == REFS
    assert [n.name for n in ir.nets] == ["VIN_RAW", "VIN", "+5V", "GND", "LED_A", "AVCC", "AREF", "RESET", "PEN", "XTAL1", "XTAL2", *PORT_NETS]
    # every pin of every part is in exactly one net; U1's 64 pins, found by name, land where the ATmega128 pinout says
    where = _pin_to_net(ir)
    for c in ir.components:
        for p in c.pins:
            assert len(where.get((c.ref, p.number), [])) == 1, (c.ref, p.number, where.get((c.ref, p.number)))
    u1 = ir.component("U1")
    assert len(u1.pins) == 64 and [(p.number, p.name, p.electrical_type.value) for p in u1.pins] == [(n, name, t) for n, name, t, *_ in ATMEGA128_PINS]
    assert u1.pins[0].provenance.kind is ProvenanceKind.AUTHORITATIVE and u1.pins[0].provenance.source.document_path.endswith("MCU_Microchip_ATmega.kicad_sym")
    expected_net = {"VCC": "+5V", "GND": "GND", "AVCC": "AVCC", "AREF": "AREF", "~{RESET}": "RESET", "~{PEN}": "PEN", "XTAL1": "XTAL1", "XTAL2": "XTAL2", **{n: n for n in PORT_NETS}}
    assert set(expected_net) == set(mcu_pin_names())
    for p in u1.pins:
        assert where[("U1", p.number)] == [expected_net[p.name]], (p.number, p.name)
    assert sorted(p.pin_number for p in ir.net("+5V").pins if p.component_ref == "U1") == ["21", "52"]
    assert sorted(p.pin_number for p in ir.net("GND").pins if p.component_ref == "U1") == ["22", "53", "63"]
    # the ISP and UART0 headers share PE0 / PE1 (the ATmega128 programs through PDI / PDO), SCK is PB1
    for header, table in (("J2", ISP_PINS), ("J3", UART_PINS)):
        for pin, net, _ in table:
            assert where[(header, pin)] == [net], (header, pin)
    assert [f"{p.component_ref}.{p.pin_number}" for p in ir.net("PE0").pins] == ["U1.2", "J8.1", "J2.4", "J3.3"]
    assert [f"{p.component_ref}.{p.pin_number}" for p in ir.net("PE1").pins] == ["U1.3", "J8.2", "J2.1", "J3.4"]
    assert ir.net("PE0").provenance.note == "port E bit 0: U1 pin 2 to J8.1; ISP MOSI (PDI) on J2.4; UART0 RXD0 on J3.3"
    # the parameters: inputs copied from the requirements, choices the user's, the rest calculator outputs
    p = ir.parameters
    assert list(p) == ["v_in", "f_clk", "v_out_reg", "i_load_budget", "p_reg_max", "c_in", "c_out", "c_dec", "l_avcc", "c_xtal", "c_stray", "r_reset", "c_reset", "c8_ic",
                       "r_pen", "v_f_led", "i_led", "tol_rel", "v_avcc_tol_abs", "p_reg", "r_led", "i_led_design", "tau_reset", "v_reset_tau", "tran_step", "tran_stop",
                       "c_load", "f_avcc"]
    for key, rid, value in (("v_in", "req.input_voltage", 9.0), ("f_clk", "req.clock_frequency", 16e6)):
        assert p[key].value == value and p[key].provenance.kind is ProvenanceKind.USER_REQUIREMENT and p[key].provenance.derived_from == [rid]
    for key in list(p)[2:19]:
        assert p[key].provenance.kind is ProvenanceKind.USER_REQUIREMENT and p[key].provenance.tool is None, key
        assert p[key].provenance.note.startswith(f"{CHOICE_NOTE_PREFIX}; template atmega128_devboard v{TEMPLATE_VERSION}: {key} = "), key
    tools = {k: (p[k].provenance.tool, p[k].provenance.tool_version) for k in list(p)[19:]}
    assert tools == {
        "p_reg": ("calc.regulator.p_dissipation", CALC_VERSION), "r_led": ("calc.led.R", CALC_VERSION), "i_led_design": ("calc.led.I", CALC_VERSION),
        "tau_reset": ("calc.rc.tau", CALC_VERSION), "v_reset_tau": ("calc.rc.step_response", CALC_VERSION), "tran_step": ("calc.rc.tran_step", CALC_VERSION),
        "tran_stop": ("calc.rc.tran_stop", CALC_VERSION), "c_load": ("calc.crystal.load_capacitance", CALC_VERSION), "f_avcc": ("calc.lc.cutoff", CALC_VERSION),
    }
    assert p["v_reset_tau"].provenance.inputs == {"v_step": "v_out_reg", "t": "tau_reset", "tau": "tau_reset"} and p["v_reset_tau"].value == pytest.approx(V_RESET_TAU)
    assert (p["p_reg"].value, p["r_led"].value, p["c_load"].value) == (pytest.approx(0.2), pytest.approx(1500.0), pytest.approx(15e-12))
    # the requirements are served by the input side and by the clock
    served = {rid: sorted({c.ref for c in ir.components if rid in c.serves_requirements} | {n.name for n in ir.nets if rid in n.serves_requirements})
              for rid in ("req.input_voltage", "req.clock_frequency")}
    assert served == {"req.input_voltage": ["C1", "D1", "J1", "U2", "VIN", "VIN_RAW"], "req.clock_frequency": ["C10", "C9", "U1", "XTAL1", "XTAL2", "Y1"]}
    assert ir.topology.name == "ATmega128 development board" and [b.id for b in ir.topology.blocks] == ["power_input", "power_led", "mcu", "avcc_filter", "reset", "pen", "clock", "headers"]
    assert [c.id for c in ir.constraints] == ["c.atmega.isp_uart_shared", "c.atmega.regulator_thermal", "c.atmega.crystal_load", "c.atmega.polarity_by_pin_number"]
    thermal = ir.constraints[1]
    assert thermal.parameters["p_dissipation"].value == pytest.approx(0.2) and thermal.parameters["p_budget"].value == 1.0 and "no thermal verdict is claimed" in thermal.description
    assert ir.constraints[3].provenance.kind is ProvenanceKind.USER_REQUIREMENT and "jack_pinout" in ir.constraints[3].provenance.note


def test_the_simulation_excludes_what_has_no_model_and_judges_three_analog_facts(tmp_path: Path):
    ir, lib, _, _ = _built(tmp_path)
    excluded = {c.ref: c.spice.exclude_reason for c in ir.components if c.spice.exclude}
    assert list(excluded) == EXCLUDED
    assert excluded["U1"].startswith("U1 has no SPICE model: excluded") and excluded["U2"].startswith("U2 L7805 has no SPICE model in the KiCad libraries")
    assert "unsimulated input side" in excluded["D1"] and "DC-floating node" in excluded["C1"] and excluded["D2"].startswith("ideal constant-V_f LED")
    for ref in ("C7", "C9", "C10"):
        assert "DC-floating node (ngspice: singular matrix at the operating point)" in excluded[ref], ref
    assert excluded["J4"] == "connector, no electrical model" and "no switch model" in excluded["SW1"] and "no authoritative motional model" in excluded["Y1"]
    for ref in ("U1", "U2", "D1", "D2", "J1", "C7"):  # model decisions the user confirmed with the table
        assert ir.component(ref).spice.provenance.kind is ProvenanceKind.USER_REQUIREMENT, ref
    included = {c.ref: (c.spice.device.value, c.spice.value.value) for c in ir.components if not c.spice.exclude}
    assert included == {"C2": ("C", 1e-5), "C3": ("C", 1e-7), "R1": ("R", pytest.approx(1500.0)), "C4": ("C", 1e-7), "C5": ("C", 1e-7), "L1": ("L", 1e-5), "C6": ("C", 1e-7),
                        "R2": ("R", 1e4), "C8": ("C", 1e-7), "R3": ("R", 1e4)}
    c8 = ir.component("C8").spice
    assert c8.params["ic"].value == 0.0 and c8.params["ic"].provenance.kind is ProvenanceKind.USER_REQUIREMENT and "c8_ic = 0.0 V" in c8.params["ic"].provenance.note
    assert ir.component("L1").electrical["inductance"].value == 1e-5 and ir.component("C9").electrical["capacitance"].value == 22e-12
    # part values are KiCad display text (the netlist, NETLIST above, keeps format_spice_number's exact spelling: 1e-7, 1e-5)
    values = {c.ref: c.value for c in ir.components if c.electrical}
    assert values == {"C1": "10u", "C2": "10u", "C3": "100n", "R1": "1.5k", "C4": "100n", "C5": "100n", "L1": "10u", "C6": "100n", "C7": "100n",
                      "R2": "10k", "C8": "100n", "R3": "10k", "C9": "22p", "C10": "22p"}
    for c in ir.components:
        for key, t in c.electrical.items():  # parsed with the quantity parser, within the 5-significant-digit display rounding
            assert part_value_agrees(c.value, t.value, t.unit) is True, (c.ref, key, c.value, t.value)
    sim = ir.simulation
    v5v, vled = sim.stimulus("V5V"), sim.stimulus("VLED")
    assert (v5v.net, v5v.reference_net, v5v.value.value, v5v.serves_requirements) == ("+5V", "GND", 5.0, ["req.input_voltage"]) and "regulator_model" in v5v.provenance.note
    assert (vled.net, vled.value.value) == ("LED_A", 2.0) and "led_model" in vled.provenance.note
    op, tran = sim.analysis("op"), sim.analysis("tran")
    assert analysis_command(op, sim) == "op" and analysis_command(tran, sim) == "tran 1e-5 5m uic"
    assert tran.params["uic"].value is True and "c8_ic" in tran.params["uic"].provenance.note and "start" not in tran.params
    assert tran.params["step"].provenance.tool == "calc.rc.tran_step" and tran.params["stop"].provenance.tool == "calc.rc.tran_stop"
    exps = {e.id: e for e in sim.expectations}
    assert list(exps) == ["i_led", "v_reset_tau", "v_avcc"] and all(e.requirement_id is None for e in exps.values())  # none of them is a requirement's value
    assert (exps["i_led"].analysis_id, exps["i_led"].vector, exps["i_led"].reduce, exps["i_led"].nominal.value, exps["i_led"].tol_rel.value) == ("op", "i(VLED)", Reduce.VALUE, pytest.approx(2e-3), 0.02)
    e = exps["v_reset_tau"]
    assert (e.analysis_id, e.vector, e.reduce, e.at.value, e.nominal.value, e.tol_rel.value) == ("tran", "v(RESET)", Reduce.AT, pytest.approx(1e-3), pytest.approx(V_RESET_TAU), 0.02)
    assert e.at.provenance.tool == "calc.rc.tau" and e.nominal.provenance.tool == "calc.rc.step_response"
    e = exps["v_avcc"]
    assert (e.analysis_id, e.vector, e.reduce, e.nominal.value, e.tol_abs.value, e.tol_rel) == ("op", "v(AVCC)", Reduce.VALUE, 5.0, 1e-3, None)
    assert _netlist(ir, tmp_path, lib) == NETLIST


def test_recompute_inputs_check_validators_and_compiled_views_of_the_confirmed_board(tmp_path: Path):
    ir, lib, _, _ = _built(tmp_path)
    rec = recompute_parameters(ir)
    assert rec.status is S.PASS and rec.message == "16 value(s) recomputed"
    assert {k for k in rec.details["parameters"] if not k.startswith(("components", "simulation"))} == {
        "p_reg", "r_led", "i_led_design", "tau_reset", "v_reset_tau", "tran_step", "tran_stop", "c_load", "f_avcc"}
    assert {k for k in rec.details["parameters"] if k.startswith("simulation")} == {
        "simulation.analyses[tran].params[step]", "simulation.analyses[tran].params[stop]", "simulation.expectations[i_led].nominal",
        "simulation.expectations[v_reset_tau].nominal", "simulation.expectations[v_reset_tau].at"}
    inputs = check_inputs_vs_requirements(ir)
    assert inputs.status is S.PASS and set(inputs.details["parameters"]) == {"v_in", "f_clk"}
    validators = _validate(ir, tmp_path, lib)
    assert validators["ir.assumptions"] is S.PASS and validators["ir.llm_requirements"] is S.PASS and validators["ir.connectivity"] is S.PASS
    # the 64-pin symbol lays out on the schematic (the fixture's distinct power-pin positions) and the BOM carries every part
    sch = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "sch", tools={"kicad_library": lib}))
    text = Path(sch.path).read_text(encoding="utf-8")
    assert '(lib_id "MCU_Microchip_ATmega:ATmega128-16A")' in text and text.count('(global_label "PE0"') == 4
    bom = BOMCompiler().compile(ir, CompileContext(workdir=tmp_path / "bom", tools={"kicad_library": lib}))
    bom_text = Path(bom.path).read_text(encoding="utf-8")
    assert all(ref in bom_text for ref in REFS) and "power output header (+5V / GND)" in bom_text  # J11's description does not start with a formula character
    saved = ir.save(tmp_path / "ir.json")
    assert CircuitIR.load(saved).content_hash() == ir.content_hash()
    state, _ = _run(ir, tmp_path, lib, {})  # a later run: the design is present, the copied inputs are re-read
    assert state.outcome(Stage.ARCHITECTURE).status is S.PASS and ir.validation.latest(INPUTS_CHECK).status is S.PASS and len(ir.components) == 31


def test_stacked_power_pins_of_the_real_symbol_share_one_stub_and_label_and_other_nets_are_refused(tmp_path: Path):
    """The real KiCad 10.0.6 symbol stacks hidden 52 VCC on 21 and 53 / 63 GND on 22: KiCad connects every pin at one point, so the
    compiler accepts the stack because the IR puts the pins in one net, and draws one stub + one label per point (none stacked on
    another). A stack the IR splits over two nets is the ``coincides with`` refusal, never a silent merge."""
    lib = atmega_library(tmp_path / "kicad", stacked_power_pins=True)
    ir = _ir(tmp_path, "atm")
    _present(ir, tmp_path, lib, ATMEGA)
    _confirm(ir, tmp_path, lib)
    assert len(ir.components) == 31
    art = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "sch", tools={"kicad_library": lib}))
    node = sexpr.parse_file(art.path)
    u1 = next(s for s in sexpr.find_all(node, "symbol") if sexpr.get(s, "lib_id") == "MCU_Microchip_ATmega:ATmega128-16A")
    at = sexpr.find(u1, "at")
    symbol = lib.load_symbol(ir.component("U1").symbol)
    where = {p.number: pin_position(float(at[1]), float(at[2]), 0, None, p) for p in symbol.pins}
    assert where["52"] == where["21"] and where["53"] == where["22"] == where["63"]
    starts = [tuple(float(v) for v in sexpr.find_all(sexpr.find(w, "pts"), "xy")[0][1:3]) for w in sexpr.find_all(node, "wire")]
    assert starts.count(where["21"]) == 1 and starts.count(where["22"]) == 1  # one stub per stacked point
    labels = [str(g[1]) for g in sexpr.find_all(node, "global_label")]
    members = {n.name: len(n.pins) for n in ir.nets}
    assert labels.count("+5V") == members["+5V"] - 1 and labels.count("GND") == members["GND"] - 2 and len(labels) == sum(members.values()) - 3
    # the sheet grows with the table the 64-pin symbol needs: every extent inside it, U1's row alone is tall
    paper = sexpr.find(node, "paper")
    assert str(paper[1]) in ("A3", "A2", "A1") and '(paper "A4")' not in Path(art.path).read_text(encoding="utf-8")
    # the same stack split over two nets in the IR: refused as coinciding points
    plus5 = next(n for n in ir.nets if n.name == "+5V")
    plus5.pins = [p for p in plus5.pins if (p.component_ref, p.pin_number) != ("U1", "52")]
    ir.nets.append(Net(name="VCC2", kind=NetKind.POWER, pins=[PinRef(component_ref="U1", pin_number="52")], provenance=plus5.provenance))
    with pytest.raises(CompileError, match=r"pin U1\.52 at .* coincides with pin U1\.21; KiCad would connect them .*'VCC2' and '\+5V'"):
        SchematicCompiler().build(ir, lib)
    # ... and a stacked pin the IR leaves out of every net is the same refusal (a net and no net), never a silent merge
    ir.nets = [n for n in ir.nets if n.name != "VCC2"]
    with pytest.raises(CompileError, match=r"pin U1\.52 at .* coincides with pin U1\.21; KiCad would connect them \(stacked pins of .*\), but the IR puts them in no net and '\+5V'"):
        SchematicCompiler().build(ir, lib)
    # stacked pins in one net but pointing different ways (the library turned pin 52 around) are refused: one stub cannot leave both
    turned = atmega_library(tmp_path / "turned", stacked_power_pins=True, pin_angles={"52": 90})
    ir2 = _ir(tmp_path / "t", "atm")
    _present(ir2, tmp_path / "t", turned, ATMEGA)
    _confirm(ir2, tmp_path / "t", turned)
    with pytest.raises(CompileError, match=r"pin U1\.52 at .* coincides with pin U1\.21; KiCad would connect them \(stacked pins of .* pointing in different directions; unsupported\)"):
        SchematicCompiler().build(ir2, turned)


def test_reference_and_value_fields_sit_where_the_library_puts_them_clear_of_the_pins(tmp_path: Path):
    """Each Reference / Value field is written at the library symbol's own position (turned by the instance's transform) with the
    library's justification: the ATmega128-16A's reference above the body's top-left corner and its value below the body, clear of every
    pin and its number - never at a fixed offset from the origin, which put the value inside the 64-pin body across PC0 / pin 35."""
    lib = atmega_library(tmp_path / "kicad", stacked_power_pins=True)
    ir = _ir(tmp_path, "atm")
    _present(ir, tmp_path, lib, ATMEGA)
    _confirm(ir, tmp_path, lib)
    node = sexpr.parse_file(SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "sch", tools={"kicad_library": lib})).path)
    lib_syms = {str(s[1]): s for s in sexpr.find_all(sexpr.find(node, "lib_symbols"), "symbol")}
    for inst in sexpr.find_all(node, "symbol"):
        at = sexpr.find(inst, "at")
        x, y = float(at[1]), float(at[2])
        assert float(at[3]) == 0.0 and sexpr.find(inst, "mirror") is None
        own = {str(p[1]): p for p in sexpr.find_all(inst, "property")}
        for key in ("Reference", "Value"):
            lib_at = sexpr.find(next(p for p in sexpr.find_all(lib_syms[str(sexpr.get(inst, "lib_id"))], "property") if str(p[1]) == key), "at")
            got = sexpr.find(own[key], "at")
            assert (float(got[1]), float(got[2])) == pytest.approx((x + float(lib_at[1]), y - float(lib_at[2]))), (sexpr.get(inst, "lib_id"), key)
    u1 = next(i for i in sexpr.find_all(node, "symbol") if sexpr.get(i, "lib_id") == "MCU_Microchip_ATmega:ATmega128-16A")
    x, y = (float(v) for v in sexpr.find(u1, "at")[1:3])
    own = {str(p[1]): p for p in sexpr.find_all(u1, "property")}
    assert [str(t) for t in sexpr.find(sexpr.find(own["Value"], "effects"), "justify")[1:]] == ["left", "top"]
    vx, vy = (float(v) for v in sexpr.find(own["Value"], "at")[1:3])
    rx, ry = (float(v) for v in sexpr.find(own["Reference"], "at")[1:3])
    value_box = (vx, vy, vx + 1.27 * len("ATmega128-16A"), vy + 1.27)  # left / top justified, a generous 1.27 mm per character
    ref_box = (rx, ry - 1.27, rx + 1.27 * len("U1"), ry)  # left / bottom justified
    assert value_box[1] >= y + 48.26 and ref_box[3] <= y - 48.26  # below and above the body (half height 48.26 mm)
    symbol = lib.load_symbol(ir.component("U1").symbol)
    for pin in symbol.pins:  # every pin line, widened by a text height on both sides (its number sits beside it)
        (ax, ay), rad = pin_position(x, y, 0, None, pin), math.radians(pin.angle)
        bx, by = ax + pin.length * math.cos(rad), ay - pin.length * math.sin(rad)
        pin_box = (min(ax, bx) - 1.27, min(ay, by) - 1.27, max(ax, bx) + 1.27, max(ay, by) + 1.27)
        for box in (value_box, ref_box):
            assert not (box[0] < pin_box[2] and pin_box[0] < box[2] and box[1] < pin_box[3] and pin_box[1] < box[3]), (pin.number, pin.name, box)


def test_two_independent_builds_hash_the_same(tmp_path: Path):
    hashes, tables = [], []
    for name in ("a", "b"):
        lib = atmega_library(tmp_path / name / "kicad")
        ir = _ir(tmp_path / name, "atm")
        tables.append(_present(ir, tmp_path / name, lib, ATMEGA))
        _confirm(ir, tmp_path / name, lib)
        hashes.append(ir.content_hash())
    assert hashes[0] == hashes[1]
    assert tables[0] != tables[1] and tables[0].replace(str(tmp_path / "a"), "") == tables[1].replace(str(tmp_path / "b"), "")  # only the library paths differ


# --------------------------------------------------------------------------- refusals


def test_refuses_supplies_and_clocks_outside_its_validity_and_libraries_it_cannot_wire_by_name(tmp_path: Path):
    lib = _lib(tmp_path)
    cases = {
        ("input_voltage", "5 V"): ("supply 5 V (req.input_voltage) is outside 7..15 V", "below 7 V the input is less than 5 V + the L7805's ~2 V dropout (the regulator family's typical value, not a grounded datasheet fact)"),
        ("input_voltage", "20 V"): ("supply 20 V (req.input_voltage) is outside 7..15 V", "at the design load budget (here 0.75 W) exceeds half of the 1 W no-heatsink budget of U2's TO-220 package"),
        ("clock_frequency", "20 MHz"): ("clock frequency 20000000 Hz (req.clock_frequency) is outside 1000000..16000000 Hz", "exceeds the ATmega128-16A's maximum (the KiCad symbol's Description reads '16MHz')"),
        ("clock_frequency", "500 kHz"): ("clock frequency 500000 Hz (req.clock_frequency) is outside 1000000..16000000 Hz", "outside the crystal range the ATmega128 datasheet gives for its oscillator (stated by the template, not grounded in this IR)"),
    }
    for (key, text), (head, why) in cases.items():
        work = tmp_path / text.replace(" ", "")
        ir = _ir(work, "atm")
        state, _ = _run(ir, work, lib, {**BASE, **ATMEGA, key: text}, None)
        out = state.outcome(Stage.ARCHITECTURE)
        assert not state.blocked and out.status is S.NOT_VERIFIED and out.questions == [] and ir.components == [] and ir.parameters == {}, text
        assert f"template atmega128_devboard not proposed: {head}" in out.message and why in out.message, (text, out.message)
        assert state.outcomes[-1].stage is Stage.RELEASE and state.outcomes[-1].status is not S.PASS
    # the bounds themselves are inside
    for key, text in (("input_voltage", "7 V"), ("input_voltage", "15 V"), ("clock_frequency", "1 MHz"), ("clock_frequency", "16 MHz")):
        work = tmp_path / f"edge_{key}_{text.replace(' ', '')}"
        state, _ = _run(_ir(work, "atm"), work, lib, {**BASE, **ATMEGA, key: text})
        assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY], text
    # a missing supply is a required question with an example inside the template's range
    ir = _ir(tmp_path / "nov", "atm")
    state, _ = _run(ir, tmp_path / "nov", lib, {**BASE, "clock_frequency": "16 MHz"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and out.status is S.USER_INPUT_REQUIRED and [(q.key, q.required) for q in out.questions] == [("input_voltage", True)]
    assert out.questions[0].question == 'The ATmega128 development board template needs input_voltage in V: answer input_voltage=<value V> (e.g. input_voltage="9 V")'
    # a closed world, like every template
    ir = _ir(tmp_path / "eff", "atm")
    state, _ = _run(ir, tmp_path / "eff", lib, {**BASE, **ATMEGA, "efficiency": "90 %"})
    assert not state.blocked and ir.components == [] and "is not served by the ATmega128 development board template, which serves only clock_frequency, input_voltage" in state.outcome(Stage.ARCHITECTURE).message


@pytest.mark.parametrize(("kw", "expected"), [
    ({"rename": {"PA0": "PA0_AD0"}}, "template atmega128_devboard not proposed: U1 (MCU_Microchip_ATmega:ATmega128-16A) has no pin named ['PA0'] in this library "
                                     "(pins with other names: [('51', 'PA0_AD0')]); the template wires the MCU by pin name and will not guess which pin carries them"),
    ({"rename": {"~{RESET}": "RESET"}}, "has no pin named ['~{RESET}'] in this library (pins with other names: [('20', 'RESET')])"),
    ({"rename": {"PG4": "PG3"}}, "has no pin named ['PG4'] in this library (pins with other names: [])"),  # a duplicated name is a missing one
    # a 65th pin: a name the template does not wire, and a third VCC (every name the template wires, but not as many times as it wires it)
    ({"extra_pins": (("65", "NC", "passive", -15.24, -45.72, 0),)},
     "has pins the template does not wire [('65', 'NC')]; every MCU pin must end in a net, so the template refuses this symbol"),
    ({"extra_pins": (("65", "VCC", "power_in", -15.24, -45.72, 0),)},
     "U1 (MCU_Microchip_ATmega:ATmega128-16A) pin name counts differ from the ATmega128's: VCC: 3 pin(s), expected 2 (C4 / C5 decouple the two VCC pins"),
    ({"regulator_pin_names": ("VI", "GND", "VO")}, "Regulator_Linear:L7805 pins are not named IN / GND / OUT in this library (pins: [('1', 'VI'), ('2', 'GND'), ('3', 'VO')]); the template will not guess the regulator's pinout"),
    ({"omit_symbols": ("Connector:Barrel_Jack_Switch",)}, "template atmega128_devboard not proposed: symbol Connector:Barrel_Jack_Switch not found in the KiCad libraries"),
])
def test_refuses_a_library_whose_pins_or_parts_differ(tmp_path: Path, kw: dict, expected: str):
    lib = atmega_library(tmp_path / "kicad", **kw)
    ir = _ir(tmp_path, "atm")
    state, _ = _run(ir, tmp_path, lib, {**BASE, **ATMEGA})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and out.questions == [] and expected in out.message, out.message


def test_input_aliases_select_the_template(tmp_path: Path):
    """``dc_input`` / ``vin_dc`` mean input_voltage, ``crystal_frequency`` / ``mcu_clock`` / ``f_clk`` / ``fclk`` mean clock_frequency."""
    for v_key, f_key in (("dc_input", "crystal_frequency"), ("vin_dc", "mcu_clock"), ("input_voltage", "f_clk"), ("vin", "fclk")):
        ir = _ir(tmp_path, "atm")
        ir.requirements.requirements = [
            Requirement(id=f"req.{v_key}", key=v_key, text="9 V", kind=RequirementKind.EXPLICIT, value=user_requirement("9 V")),
            Requirement(id=f"req.{f_key}", key=f_key, text="8 MHz", kind=RequirementKind.EXPLICIT, value=user_requirement("8 MHz")),
        ]
        found, unusable = read_inputs(ir)
        assert unusable == {} and found["input_voltage"].traced.value == 9.0 and found["clock_frequency"].traced.value == 8e6, (v_key, f_key)
        assert found["clock_frequency"].traced.unit == "Hz" and found["clock_frequency"].requirement.id == f"req.{f_key}"
        assert TEMPLATE.triggered(found)


# --------------------------------------------------------------------------- theory, figures and part notes


def test_theory_figures_and_part_notes_render_with_the_designs_numbers(tmp_path: Path):
    ir, lib, _, _ = _built(tmp_path)
    sections = TEMPLATE.theory(ir)
    assert [s.title for s in sections] == [
        "보드 구성과 신호 흐름", "전원부: L7805 와 레귤레이터 손실", "리셋 회로: RC 지연", "AVCC LC 필터", "크리스탈과 부하 커패시턴스", "ISP 와 UART0 의 핀 공유, ~PEN",
        "시뮬레이션: 검증하는 것과 하지 않는 것", "유효 범위(템플릿이 거부하는 조건)와 그 이유",
    ]
    text = "\n".join(s.body for s in sections)
    assert NO_RECORD not in text
    for fragment in ("= 0.2 W", "τ = R2·C8 = 10 kΩ·100 nF = 1 ms", "= 3.1606 V   (`calc.rc.step_response`)", "= 159.15 kHz   (`calc.lc.cutoff`)", "= 15 pF",
                     "V_out + P_예산/(2·I_load) = 15 V", "이 파이프라인의 어떤 결과도 MCU 가 동작한다고 주장하지 않습니다", "| 4 | `PE0` | MOSI (PDI) |", "| 3 | `PE0` | RXD0 |"):
        assert fragment in text, fragment
    figures = TEMPLATE.theory_figures(ir)
    assert [f.id for f in figures] == ["theory_p_reg", "theory_reset_rc", "theory_avcc_lc", "theory_crystal_cl"]
    for fig in figures:
        assert fig.svg.startswith("<svg") and fig.svg.rstrip().endswith("</svg>") and fig.caption.endswith(THEORY_CURVE_NOTE), fig.id
    assert "설계점 V_in = 9 V, P = 0.2 W" in figures[0].svg and "τ = 1 ms" in figures[1].svg and "f_clk = 16 MHz" in figures[2].svg
    # a number the IR lacks drops its figure and prints the no-record marker, never a guess
    del ir.parameters["f_avcc"]
    assert [f.id for f in TEMPLATE.theory_figures(ir)] == ["theory_p_reg", "theory_reset_rc", "theory_crystal_cl"]
    assert f"f_0 = 1/(2π·√(L1·C6)) = 1/(2π·√(10 µH·100 nF)) = {NO_RECORD}" in "\n".join(s.body for s in TEMPLATE.theory(ir))
    ir2, _, _, _ = _built(tmp_path / "again")
    notes = TEMPLATE.part_notes(ir2)
    assert set(notes) == set(REFS)
    for ref, note in notes.items():
        assert note.role and note.why and note.criteria, ref
        assert note.substitutes and all(UNVERIFIED_SUBSTITUTE in s for s in note.substitutes), ref
    assert any("ATmega128A-AU" in s for s in notes["U1"].substitutes) and any("AMS1117-5.0" in s for s in notes["U2"].substitutes)
    assert any("1N5819" in s for s in notes["D1"].substitutes) and any("HC-49S" in s and "20 pF" in s for s in notes["Y1"].substitutes)
    assert "jack_pinout" in notes["J1"].why and "cp_polarity" in notes["C1"].why
    # the stage reports render: the theory report with its four figures inline, the parts report with a note per part
    doc = build_stage_document(Stage.ARCHITECTURE, ir2, lib, None)
    assert [fid for fid in doc.figures.slots["theory"]] == ["theory_p_reg", "theory_reset_rc", "theory_avcc_lc", "theory_crystal_cl"]
    assert all(f"![fig](fig:{fid})" in doc.markdown for fid in doc.figures.slots["theory"]) and doc.html.count("<svg") >= 4
    parts = build_stage_document(Stage.COMPONENT_SELECTION, ir2, lib, None)
    assert "ATmega128A-AU" in parts.markdown and parts.markdown.count(UNVERIFIED_SUBSTITUTE) >= len(REFS)


def test_the_expectation_table_keeps_five_cells_and_the_notes_keep_their_own_rules(tmp_path: Path):
    """The pass rules |measured - nominal| <= ... sit in a five-column table: their bars are escaped, so every row has five cells in the
    Markdown, the HTML shows the bars; the fuse note names CKOPT (a 16 MHz crystal) and JTAGEN (PF4..PF7 on J9); C1's substitutes meet
    its own >= 2 x 15 V rule (no 25 V part, no tantalum behind the jack) and the BOM states both electrolytics' rating rule."""
    from ai_eda.report.pdf import _split_row

    ir, lib, _, _ = _built(tmp_path)
    sim = next(sec for sec in TEMPLATE.theory(ir) if sec.title.startswith("시뮬레이션"))
    rows = [line for line in sim.body.splitlines() if line.startswith("| `")]
    assert [r.split("`")[1] for r in rows] == ["i_led", "v_reset_tau", "v_avcc"]
    for row in rows:
        cells = _split_row(row)
        assert len(cells) == 5 and cells[4].startswith("|측정값 − ") and "| ≤ " in cells[4], cells
    assert _split_row(rows[2])[4] == "|측정값 − 5 V| ≤ 1 mV"
    doc = build_stage_document(Stage.ARCHITECTURE, ir, lib, None)
    assert "<td>|측정값 − 2 mA| ≤ 2 % × 2 mA = 40 µA</td>" in doc.html
    crystal = next(sec for sec in TEMPLATE.theory(ir) if sec.title.startswith("크리스탈")).body
    for fragment in ("CKOPT 퓨즈가 프로그램되지 않은 크리스탈 모드는 8 MHz 까지", "(이 설계의 f_clk = 16 MHz 가 그 경우: CKSEL3..1 = 111, CKOPT = 0)",
                     "M103C 도 해제", "PF4..PF7(J9 의 5..8번 핀)은 JTAG 핀(TCK / TMS / TDO / TDI)", "JTD 비트", "이 IR 에 근거를 둔 사실은 아닙니다"):
        assert fragment in crystal, fragment
    notes = TEMPLATE.part_notes(ir)
    assert "JTAGEN" in notes["J9"].why and "PF4..PF7(5..8번 핀)" in notes["J9"].why and all("JTAG" not in notes[r].why for r in ("J4", "J5", "J6", "J7", "J8", "J10"))
    assert "정격 전압 ≥ 2 × 15 V = 30 V" in notes["C1"].criteria
    for sub in notes["C1"].substitutes:
        volts = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?) V\b", sub)]
        assert volts and min(volts) >= 30.0, sub  # never a part below its own criterion
        assert "25 V" not in sub and not sub.startswith(("탄탈", "10 µF / 25"))
    assert any("탄탈은" in s and "권하지 않음" in s for s in notes["C1"].substitutes)
    bom = BOMCompiler().compile(ir, CompileContext(workdir=tmp_path / "bom", tools={"kicad_library": lib}))
    text = Path(bom.path).read_text(encoding="utf-8")
    assert "input bulk electrolytic capacitor on VIN, rated >= 30 V (2 x the 15 V input bound)" in text
    assert "regulator output bulk electrolytic capacitor, rated >= 10 V (2 x the +5V rail)" in text


# --------------------------------------------------------------------------- the real engine


@needs_dll
def test_led_current_reset_delay_and_avcc_level_are_verified_by_ngspice(tmp_path: Path):
    lib = _lib(tmp_path)
    ir = _ir(tmp_path, "atm")
    _present(ir, tmp_path, lib, ATMEGA)
    # the placement stage is skipped (a control answer beside the confirmation): this test is about the SPICE stage only
    state, _ = _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes", PLACEMENT_KEY: "skip"}, Stage.SPICE, spice=True)
    assert state.outcome(Stage.CALCULATION).status is S.PASS and state.outcome(Stage.SPICE).status is S.PASS
    summary = ir.validation.latest("spice")
    assert summary.status is S.PASS and summary.details["analyses"]["tran"]["command"] == "tran 1e-5 5m uic" and summary.details["analyses"]["op"]["succeeded"]
    assert sorted(e["ref"] for e in summary.details["excluded"]) == sorted(EXCLUDED)
    i_led, v_reset, v_avcc = (ir.validation.latest(f"spice.{k}") for k in ("i_led", "v_reset_tau", "v_avcc"))
    assert i_led.status is S.PASS and i_led.details["measured"] == pytest.approx(2e-3, rel=1e-6)
    assert v_reset.status is S.PASS and v_reset.details["measured"] == pytest.approx(V_RESET_TAU, rel=1e-3) and v_reset.details["reduce"] == "at"
    bracket = v_reset.details.get("bracket")
    assert bracket is None or bracket["exact"] or bracket["x0"] <= 1e-3 <= bracket["x1"]  # an interpolation names the two samples around t = tau
    assert v_avcc.status is S.PASS and abs(v_avcc.details["measured"] - 5.0) <= 1e-3
    assert i_led.tool == "ngspice-shared" and i_led.evidence


# --------------------------------------------------------------------------- the whole pipeline, the reports and the GUI previews


def _release(tmp_path: Path, *, spice: bool):
    """Present, confirm (with ``pcb.routing=skip``) and run to RELEASE on the synthetic library whose MCU stacks its hidden power pins
    as the real KiCad 10.0.6 symbol does. Routing is skipped here: the maze router's full attempt on this 64-pin board takes minutes
    and leaves nets unrouted (the deliverable run records it), so the IR gets the placement only either way."""
    lib = atmega_library(tmp_path / "kicad", stacked_power_pins=True)
    ir = _ir(tmp_path, "atm")
    _present(ir, tmp_path, lib, ATMEGA)
    state, ctx = _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes", ROUTING_KEY: "skip"}, None, spice=spice)
    assert not state.blocked and state.outcomes[-1].stage is Stage.RELEASE
    return ir, lib, state


@needs_dll
def test_the_board_runs_to_release_with_the_analog_facts_verified_and_nothing_claimed_about_the_mcu(tmp_path: Path):
    ir, lib, state = _release(tmp_path, spice=True)
    assert state.outcome(Stage.CALCULATION).status is S.PASS and state.outcome(Stage.SPICE).status is S.PASS
    for key, measured in (("i_led", 2e-3), ("v_reset_tau", V_RESET_TAU), ("v_avcc", 5.0)):
        r = ir.validation.latest(f"spice.{key}")
        assert r.status is S.PASS and r.details["measured"] == pytest.approx(measured, rel=1e-3) and r.tool == "ngspice-shared", key
    assert sorted(e["ref"] for e in ir.validation.latest("spice").details["excluded"]) == sorted(EXCLUDED)  # U1 among them: never simulated
    # the placement: the 64-pad MCU in the centre of the generated outline, the other 30 parts on the two rings; no copper (routing skipped)
    placement = state.outcome(Stage.PLACEMENT)
    assert placement.status is S.NOT_VERIFIED and "placement.core_ring 0.2: 31 component(s) on a" in placement.message
    assert "core U1 (64 pads)" in placement.message and "routing skipped by answer" in placement.message
    o, u1 = ir.pcb.outline, ir.pcb.placement("U1")
    assert len(ir.pcb.placements) == 31 and (u1.x_mm, u1.y_mm) == (o.origin_x_mm + o.width_mm / 2, o.origin_y_mm + o.height_mm / 2)
    assert ir.pcb.tracks == [] and ir.pcb.vias == []
    # the router would route this board at the fine rules its 0.8 mm TQFP pitch selects
    params = RoutingParams.for_board(ir, lib)
    assert (params.rules, params.pad_pitch_mm, params.pitch_footprint, params.grid_mm, params.track_width_mm) == ("fine", 0.8, "Package_QFP:TQFP-64_14x14mm_P0.8mm", 0.2, 0.25)
    # what is true about the copper: none, so every multi-pin net is unconnected - FAIL, and RELEASE says so; ERC / DRC need kicad-cli
    conn = ir.validation.latest("pcb.routing.connectivity")
    assert conn.status is S.FAIL and conn.message.startswith("64 net(s) not connected through IR copper")
    assert state.outcome(Stage.RELEASE).status is S.FAIL and "pcb.routing.connectivity" in state.outcome(Stage.RELEASE).message
    assert state.outcome(Stage.ERC).status is S.NOT_VERIFIED and state.outcome(Stage.DRC).status is S.NOT_VERIFIED
    # the real symbol's stacked hidden power pins compile (one stub per stacked point) and the board compiles with all 64 pads
    assert state.outcome(Stage.SCHEMATIC).status is S.PASS and state.outcome(Stage.PCB).status is S.PASS
    assert ir.validation.latest("review.spice_vs_requirements").status is S.PASS and ir.validation.latest("calc.recompute").status is S.PASS


def test_the_circuit_report_compares_track_capacity_with_the_rails_load_budget(tmp_path: Path):
    """The dev board states its steady current as the +5V rail's design load budget (``i_load_budget``, a choice, not a measurement): with
    copper on the board the circuit report sets each track width's IPC-2221 capacity against it. The demo board is placement only, so a
    synthetic +5V track stands in for routed copper here (a view: nothing is saved)."""
    from ai_eda.ir import Track
    from ai_eda.report.stages import ipc2221_current_a

    ir, lib, state = _release(tmp_path, spice=False)
    no_copper = build_stage_document(Stage.PCB, ir, lib, state).markdown
    assert "I_load" not in no_copper  # no track, no width to compare: nothing printed
    ir.pcb.tracks = [Track(net="+5V", layer="F.Cu", start=(10.0, 10.0), end=(20.0, 10.0), width_mm=0.25)]
    text = build_stage_document(Stage.PCB, ir, lib, state).markdown
    ratio = ipc2221_current_a(0.25) / 0.05
    assert f"설계의 최대 정상 전류 I_load (+5V 레일의 설계 부하 예산, 측정값 아님) = 50 mA 이므로 여유는 {ratio:.3g} 배입니다." in text
    assert "이 설계의 템플릿은 정상 전류를 명시하지 않으므로" not in text


def test_stage_reports_and_gui_previews_render_the_64_pin_board(tmp_path: Path):
    """The four stage reports carry the board's figures and the core-ring rule; the GUI draws the compiled schematic's 64-pin symbol
    (the three stacked hidden pins dashed) and the board with every TQFP pad and legible labels. Views only: nothing is saved."""
    ir, lib, state = _release(tmp_path, spice=False)
    before = ir.content_hash()
    theory = build_stage_document(Stage.ARCHITECTURE, ir, lib, state)
    assert theory.figures.slots["theory"] == ["theory_p_reg", "theory_reset_rc", "theory_avcc_lc", "theory_crystal_cl"]
    circuit = build_stage_document(Stage.PCB, ir, lib, state)
    text = circuit.markdown
    assert circuit.figures.slots["placement"] == ["placement"] and "![fig](fig:placement)" in text and circuit.html.count("<svg") >= 1
    assert "| `U1` | `Package_QFP:TQFP-64_14x14mm_P0.8mm` | 44 | 44 | 0 | top | 계산기 출력 (placement.core_ring v0.2) |" in text
    assert "| ref | 링 | 당김 각 (°) | 당김 출처 |" in text and "| `U1` | 코어 | none | `core` |" in text
    assert "`placement.core_ring` 는 패드가 32개 이상인 부품이 있을 때" in text and "(이 보드: `U1`)" in text and "안쪽 링(이 보드 13개)" in text
    assert "몸체 끝이 외곽에서 `margin_mm` 안쪽에 오고 그 앞에는 부품이 없습니다" in text and "그룹의 커넥터(없으면 첫 부품)부터" in text
    assert "- 배치 파라미터 (provenance `derived_from`): `margin_mm` = 2.0, `spacing_mm` = 1.0" in text  # the per-part entries are columns, not this line
    # the DC jack is turned radially: the side its body overhangs (6.5 mm past the pads, BarrelJack_Horizontal's plug end) is the board edge,
    # so no part stands in front of it; the input chain J1 -> D1 -> C1 -> U2 and the LED with its resistor follow each other on that edge
    j1 = ir.pcb.placement("J1")
    jack = lib.load_footprint(ir.component("J1").footprint)
    ext, pads = footprint_bbox(j1, jack), pads_bbox(j1, jack)
    assert (j1.rotation_deg, ext.y1) == (270.0, MARGIN_MM) and pads.y1 - ext.y1 == pytest.approx(6.5)

    def box(ref: str):
        return footprint_bbox(ir.pcb.placement(ref), lib.load_footprint(ir.component(ref).footprint))

    top = sorted((c.ref for c in ir.components if box(c.ref).y1 == MARGIN_MM), key=lambda r: -(box(r).x1 + box(r).x2))
    first = top.index("J1")
    assert top[first:first + 4] == ["J1", "D1", "C1", "U2"] and abs(top.index("D2") - top.index("R1")) == 1, top
    assert "잇지 못한 넷과 그 이유" in text and "- `pcb.routing.connectivity`: **FAIL**" in text
    final = build_stage_document(Stage.RELEASE, ir, lib, state)
    assert final.figures.slots["tolerance"] == ["tolerance"]
    assert ir.content_hash() == before
    # the GUI schematic preview: the compiled file's own lib_symbols, the 64 pins of U1 with 52 / 53 / 63 hidden, every label drawn
    svg = project_schematic_svg(ir, tmp_path)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    u1 = next(g for g in root.iter(f"{ns}g") if g.get("class") == "symbol" and g.get("data-ref") == "U1")
    pins = [e for e in u1.iter(f"{ns}line") if e.get("class") == "pin"]
    assert len(pins) == 64 and sorted(p.get("data-number") for p in pins if p.get("data-hidden") == "yes") == ["52", "53", "63"]
    labels = [g for g in root.iter(f"{ns}g") if g.get("class") == "global-label"]
    assert len(labels) == sum(len(n.pins) for n in ir.nets) - 3 and root.get("viewBox") != "0 0 297 210"  # the sheet grew past A4
    # the board preview: all 64 TQFP pads, the pitch at least 5 px at the preview scale, U1's labels inside its pad ring, no labels on top of each other
    board = ET.fromstring(board_svg(ir, lib))
    pads = [g for g in board.iter(f"{ns}g") if g.get("class") == "pad" and g.get("data-ref") == "U1"]
    assert len(pads) == 64
    scale = float(next(e for e in board.iter(f"{ns}rect") if e.get("class") == "outline").get("width")) / ir.pcb.outline.width_mm
    assert 0.8 * scale >= 5.0
    texts = [t for t in board.iter(f"{ns}text") if t.get("class") in ("ref", "value")]
    boxes = []
    for t in texts:
        px = float(t.get("font-size"))
        w = _text_width(t.text, px)
        x, y, anchor = float(t.get("x")), float(t.get("y")), t.get("text-anchor")
        left = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
        boxes.append((t.text, (left, y - px, left + w, y + 3)))
    clashes = [(a, c) for i, (a, b) in enumerate(boxes) for c, d in boxes[i + 1:] if b[0] < d[2] and d[0] < b[2] and b[1] < d[3] and d[1] < b[3]]
    assert clashes == []
    assert labels_on_foreign_copper(board) == []  # no label covers another part's pads (the TQFP's pads stay legible)
    u1_x = [float(r.get("x") or r.get("cx")) for g in pads for r in g if r.tag in (f"{ns}rect", f"{ns}circle")]
    u1_label = next(t for t in texts if t.get("class") == "ref" and t.text == "U1")
    assert min(u1_x) < float(u1_label.get("x")) < max(u1_x)
