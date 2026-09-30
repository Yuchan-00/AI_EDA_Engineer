"""The ATmega128 development board template (``atmega128_devboard``): a minimal board around an ATmega128 from two confirmed values.

Selected by ``clock_frequency`` (the crystal) and needing ``input_voltage``
(the DC jack), it builds: J1 DC jack -> D1 series rectifier (reverse-polarity
protection) -> C1 -> U2 L7805 -> C2 / C3 -> ``+5V``; the power LED D2 with
R1; U1 ATmega128-16A (TQFP-64) with one 100 nF decoupling capacitor per VCC
pin (C4, C5), AVCC through the L1 / C6 low-pass, C7 on AREF, the RC reset
R2 / C8 with the push button SW1, the ~PEN pull-up R3 and the crystal Y1 with
its load capacitors C9 / C10; every port on a header (J4..J9 1x8 for
PA..PF, J10 1x5 for PG0..PG4), the ISP header J2 (2x3), the UART0 header J3
(1x4) and a 5 V output header J11 (1x2). Every U1 pin ends in a net.

Library facts this template relies on, read through
:class:`~ai_eda.tools.kicad.library.KicadLibrary` from the packed KiCad
10.0.6 libraries (symbols from the ``10.0.6`` tag of kicad-symbols,
footprints from the same tag) and re-checked on every build - a missing
symbol, footprint or pin name refuses the template, nothing is guessed:

* ``MCU_Microchip_ATmega:ATmega128-16A`` (``extends`` ``ATmega64L-8A``): one
  unit, 64 pins, names by number ``1 ~{PEN}``, ``2..9 PE0..PE7``,
  ``10..17 PB0..PB7``, ``18 PG3``, ``19 PG4``, ``20 ~{RESET}``, ``21 VCC``,
  ``22 GND``, ``23 XTAL2``, ``24 XTAL1``, ``25..32 PD0..PD7``, ``33 PG0``,
  ``34 PG1``, ``35..42 PC0..PC7``, ``43 PG2``, ``44..51 PA7..PA0``,
  ``52 VCC``, ``53 GND``, ``54..61 PF7..PF0``, ``62 AREF``, ``63 GND``,
  ``64 AVCC``; types: ~{PEN} / ~{RESET} / XTAL1 input, XTAL2 output, 21 VCC /
  22 GND / 64 AVCC power_in, 52 / 53 / 62 / 63 passive, the ports
  bidirectional. 52 / 53 / 63 are *hidden* pins stacked on 21 / 22 (the same
  symbol position). Footprint property ``Package_QFP:TQFP-64_14x14mm_P0.8mm``
  (64 pads, 1.475 x 0.55 mm at 0.8 mm pitch); Description "16MHz, 128kB
  Flash, 4kB SRAM, 4kB EEPROM, JTAG, TQFP-64" - the 16 MHz upper bound of
  ``clock_frequency`` is that library text.
* ``Regulator_Linear:L7805``: ``1 IN`` (power_in), ``2 GND`` (power_in),
  ``3 OUT`` (power_out); wired by name. Footprint
  ``Package_TO_SOT_THT:TO-220-3_Vertical`` (pads 1..3).
* ``Device:D`` and ``Device:LED``: ``1 K``, ``2 A``; wired by name.
* ``Connector:Barrel_Jack`` has two pins, but the generic
  ``Connector_BarrelJack:BarrelJack_Horizontal`` footprint has three
  electrical pads (1, 2, 3) and the PCB compiler refuses a footprint pad that
  is no IR pin; the template therefore uses ``Connector:Barrel_Jack_Switch``
  (pins 1, 2, 3) with that footprint. Its pins carry **no names** (empty
  strings), so the polarity is a pin-number assumption the user confirms
  (``jack_pinout``): 1 = centre pin (+), 2 = sleeve, 3 = the sleeve's switch
  contact, tied to GND so the third pad is not left floating.
* ``Device:C_Polarized``: pins ``1`` / ``2`` with no names; pin 1 = + is a
  pin-number assumption the user confirms (``cp_polarity``).
* ``Device:Crystal`` (``1`` / ``2``), ``Switch:SW_Push`` (``1`` / ``2``; its
  ``Button_Switch_THT:SW_PUSH_6mm`` footprint has two pads per number),
  ``Device:L`` (``1`` / ``2``), ``Device:R`` / ``Device:C`` (unnamed):
  symmetric two-terminal parts.
* ``Connector_Generic:Conn_02x03_Odd_Even`` / ``Conn_01x08`` / ``Conn_01x05``
  / ``Conn_01x04`` / ``Conn_01x02`` (``Pin_1`` ..) with the
  ``Connector_PinHeader_2.54mm`` footprints; ``Diode_THT:D_DO-41_SOD81_P10.16mm_Horizontal``,
  ``Capacitor_THT:CP_Radial_D5.0mm_P2.50mm``, ``Crystal:Crystal_HC49-4H_Vertical``,
  ``Capacitor_SMD:C_0603_1608Metric``, ``Resistor_SMD:R_0603_1608Metric`` /
  ``R_0805_2012Metric``, ``Inductor_SMD:L_0805_2012Metric`` and
  ``LED_SMD:LED_0805_2012Metric``: all present.

Numbers: every value is a calculator output (``calc.led.R`` / ``calc.led.I``
for the power LED, ``calc.rc.tau`` / ``calc.rc.step_response`` for the reset
delay, ``calc.rc.tran_step`` / ``calc.rc.tran_stop`` for its transient
window, ``calc.regulator.p_dissipation``, ``calc.crystal.load_capacitance``,
``calc.lc.cutoff``), a value copied from a requirement, or a choice the user
confirms in the ``confirm_design`` table.

Simulation: the MCU (U1) and the regulator (U2) have no SPICE model in the
KiCad libraries and are excluded; U2's output is the ideal stimulus ``V5V``
and everything upstream (J1, D1, C1) is not simulated. What is simulated:
the power LED current (``i(VLED)`` at the operating point, the LED an ideal
V_f source like the ``led`` template's), the reset delay (``v(RESET)`` at
t = tau of a ``tran ... uic`` run from C8's ic = 0; the step is tau / 100 so
t = tau is a multiple of it) and the AVCC filter's DC level (``v(AVCC)`` at
the operating point). C7, C9 and C10 are excluded too: each has a terminal
on a node that only excluded parts reach (AREF: U1; XTAL1 / XTAL2: U1 and
Y1), so it would be a DC-floating node and ngspice's operating point fails
(measured on ngspice-42: "singular matrix: check node xtal2", gmin and
source stepping fail). The crystal oscillator, the ~PEN pull-up's function
and the decoupling are structural only; nothing here claims the MCU works.
With ``uic`` the ideal L1 / C6 start at 0 A / 0 V and ring undamped at f_0
through the transient (no resistance in ideal parts); v(AVCC) is therefore
judged at the operating point, never from the transient.

Validity: 7 V <= ``input_voltage`` <= 15 V and 1 MHz <= ``clock_frequency``
<= 16 MHz. Below 7 V the input is less than 5 V + the L7805 family's typical
~2 V dropout (a typical value, not a grounded datasheet fact); above 15 V the
dissipation at the 50 mA design load budget exceeds half of the 1 W
no-heatsink budget the template keeps for the TO-220 package (the template's
conservative rule: 15 V = V_out + P_budget / (2 I_load), a 2x margin because
the load budget is an estimate). 16 MHz is the library Description's
maximum; below 1 MHz is outside the crystal range the ATmega128 datasheet
gives for its oscillator (stated, not grounded in this IR). D1's forward drop
(~0.7..1 V for a 1N4007-class rectifier, not grounded) comes on top of the
dropout, so the headroom near the 7 V edge rests on the regulator's dropout
at light load, which nothing here verifies (the theory report says so).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ai_eda.ir import (
    AnalysisSpec,
    NetClass,
    TimingPath,
    Block,
    CircuitDomain,
    CircuitIR,
    Component,
    Constraint,
    ConstraintKind,
    Expectation,
    Net,
    NetKind,
    PinRef,
    Reduce,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    Traced,
)
from ai_eda.tools.calc.basic import (
    crystal_load_capacitance,
    lc_cutoff,
    led_current,
    led_series_resistor,
    rc_step_response,
    rc_time_constant,
    rc_tran_step,
    rc_tran_stop,
    regulator_dissipation,
)
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.base import (
    NO_RECORD,
    Choice,
    PartNote,
    Plan,
    Template,
    TheorySection,
    choice_provenance,
    number,
    parameter_value,
    quantity,
    structural_provenance,
    unverified,
)
from ai_eda.design.inputs import DesignInput
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name, require_pins, two_terminals
from ai_eda.design.templates import (
    CAPACITOR,
    HEADER_2,
    RESISTOR,
    THEORY_CURVE_NOTE,
    _add,
    _changes,
    _choice,
    _curve_figure,
    _div,
    _header_note,
    _is_choice,
    _judging_line,
    _known,
    _lin_grid,
    _log_grid,
    _missing_inputs,
    _mul,
    _net_line,
    _part_line,
    _refused,
    _resistor_criteria,
    _resistor_substitutes,
    _spelled,
    _sub,
    _value_text,
)

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

MCU = (("MCU_Microchip_ATmega", "ATmega128-16A"), ("Package_QFP", "TQFP-64_14x14mm_P0.8mm"))
REGULATOR = (("Regulator_Linear", "L7805"), ("Package_TO_SOT_THT", "TO-220-3_Vertical"))
DC_JACK = (("Connector", "Barrel_Jack_Switch"), ("Connector_BarrelJack", "BarrelJack_Horizontal"))
RECTIFIER = (("Device", "D"), ("Diode_THT", "D_DO-41_SOD81_P10.16mm_Horizontal"))
CAP_POLARIZED = (("Device", "C_Polarized"), ("Capacitor_THT", "CP_Radial_D5.0mm_P2.50mm"))
RESISTOR_0805 = (("Device", "R"), ("Resistor_SMD", "R_0805_2012Metric"))
INDUCTOR = (("Device", "L"), ("Inductor_SMD", "L_0805_2012Metric"))
LED_0805 = (("Device", "LED"), ("LED_SMD", "LED_0805_2012Metric"))
CRYSTAL = (("Device", "Crystal"), ("Crystal", "Crystal_HC49-4H_Vertical"))
PUSH_BUTTON = (("Switch", "SW_Push"), ("Button_Switch_THT", "SW_PUSH_6mm"))
ISP_HEADER = (("Connector_Generic", "Conn_02x03_Odd_Even"), ("Connector_PinHeader_2.54mm", "PinHeader_2x03_P2.54mm_Vertical"))
HEADER_8 = (("Connector_Generic", "Conn_01x08"), ("Connector_PinHeader_2.54mm", "PinHeader_1x08_P2.54mm_Vertical"))
HEADER_5 = (("Connector_Generic", "Conn_01x05"), ("Connector_PinHeader_2.54mm", "PinHeader_1x05_P2.54mm_Vertical"))
HEADER_4 = (("Connector_Generic", "Conn_01x04"), ("Connector_PinHeader_2.54mm", "PinHeader_1x04_P2.54mm_Vertical"))

#: port letter, header ref, number of bits: every port on its own header, PG (5 bits) on a 1x5
PORTS: tuple[tuple[str, str, int], ...] = (("A", "J4", 8), ("B", "J5", 8), ("C", "J6", 8), ("D", "J7", 8), ("E", "J8", 8), ("F", "J9", 8), ("G", "J10", 5))
#: the U1 pins wired one by one, by library name
SINGLE_PINS: tuple[str, ...] = ("~{PEN}", "~{RESET}", "XTAL1", "XTAL2", "AVCC", "AREF")
#: how many pins the ATmega128 symbol names VCC / GND (each one wired; another count refuses: C4 / C5 decouple the two VCC pins)
POWER_PIN_COUNTS: dict[str, int] = {"VCC": 2, "GND": 3}
#: the AVR 6-pin ISP header J2: pin -> net. The ATmega128 is serially programmed through PDI = PE0 / PDO = PE1 with SCK
#: on PB1 (datasheet doc2467, "Serial Programming Pin Mapping"; stated by the template, not grounded in the IR)
ISP_PINS: tuple[tuple[str, str, str], ...] = (("1", "PE1", "MISO (PDO)"), ("2", "+5V", "VCC"), ("3", "PB1", "SCK"), ("4", "PE0", "MOSI (PDI)"), ("5", "RESET", "~RESET"), ("6", "GND", "GND"))
#: the UART0 header J3: pin -> net (RXD0 = PE0, TXD0 = PE1: the MCU's own receive / transmit pins)
UART_PINS: tuple[tuple[str, str, str], ...] = (("1", "GND", "GND"), ("2", "+5V", "+5V"), ("3", "PE0", "RXD0"), ("4", "PE1", "TXD0"))


def port_pin_names() -> list[str]:
    """``PA0`` .. ``PG4`` in header order: every port bit the template puts on a header."""
    return [f"P{letter}{bit}" for letter, _, bits in PORTS for bit in range(bits)]


def mcu_pin_names() -> dict[str, int]:
    """Every U1 pin name the template wires, with how many pins carry it (the closed set a library symbol must match)."""
    return {**{n: 1 for n in SINGLE_PINS}, **{n: 1 for n in port_pin_names()}, **POWER_PIN_COUNTS}


class Atmega128DevboardTemplate(Template):
    """ATmega128 development board: DC jack + L7805, crystal at ``clock_frequency``, every port on headers, ISP and UART0 (see the module docstring)."""

    id = "atmega128_devboard"
    title = "ATmega128 development board"
    plane_nets = ("GND", "+5V")
    triggers = ("clock_frequency",)
    needs = ("clock_frequency", "input_voltage")
    serves = ("clock_frequency", "input_voltage")

    V_IN_MIN, V_IN_MAX = 7.0, 15.0
    F_MIN, F_MAX = 1e6, 16e6
    #: the L7805 family's typical dropout the lower input bound is asserted against (typical, not a grounded datasheet fact)
    DROPOUT_TYP_V = 2.0
    V_OUT_REG_V = 5.0
    I_LOAD_BUDGET_A = 0.05
    P_REG_MAX_W = 1.0
    C_IN_F = 10e-6
    C_OUT_F = 10e-6
    C_DEC_F = 100e-9
    L_AVCC_H = 10e-6
    C_XTAL_F = 22e-12
    C_STRAY_F = 4e-12
    R_RESET_OHM = 10e3
    C_RESET_F = 100e-9
    C8_IC_V = 0.0
    R_PEN_OHM = 10e3
    V_F_LED_V = 2.0
    I_LED_A = 2e-3
    TOL_REL = 0.02
    V_AVCC_TOL_ABS_V = 1e-3
    #: the rectifier's forward drop a report uses for the headroom display (1N4007 class at tens of mA; not grounded, never in the IR)
    V_D1_DISPLAY_V = 0.8

    REGULATOR_MODEL = (
        "U2 L7805 has no SPICE model in the KiCad libraries: it is excluded from the netlist and its 5 V output is the ideal stimulus V5V "
        "(= v_out_reg between +5V and GND); the input side (J1, D1, C1) is not simulated"
    )
    LED_MODEL = (
        "ideal constant-V_f LED: D2 is excluded from the netlist and replaced by the stimulus VLED = v_f_led between LED_A and GND; "
        "the expectation i(VLED) measures the current through that ideal source, not through a diode model (no authoritative model card)"
    )
    MCU_MODEL = (
        "U1 has no SPICE model: excluded; its pins are wired in the schematic and checked structurally only (nothing simulates or claims "
        "that the MCU runs); so are the parts whose only other connection is U1 or Y1: C7 (AREF), C9 / C10 (XTAL1 / XTAL2) and the crystal Y1"
    )
    JACK_PINOUT = (
        "J1 Connector:Barrel_Jack_Switch pins carry no names in the KiCad library, so its polarity is a pin-number assumption: pin 1 = centre "
        "pin (+, VIN_RAW), pin 2 = sleeve (GND), pin 3 = the sleeve's switch contact, tied to GND so the BarrelJack_Horizontal footprint's "
        "third pad is not left floating; the plug must be centre-positive (D1 blocks a reversed plug) - check the fitted jack's datasheet"
    )
    CP_POLARITY = (
        "C1 / C2 Device:C_Polarized pins carry no names in the KiCad library, so their polarity is a pin-number assumption: pin 1 = + "
        "(the symbol's '+' mark and CP_Radial's square pad 1), pin 2 = - on GND - check the part's marking before assembly"
    )

    def _out_of_range(self, inputs: dict[str, DesignInput]) -> str | None:
        """The refusal sentence for the first *present* input outside the validity range (supply, then clock), or ``None``."""
        v_in, f_in = inputs.get("input_voltage"), inputs.get("clock_frequency")
        if v_in is not None and not self.V_IN_MIN <= v_in.traced.value <= self.V_IN_MAX:
            v, req_v = v_in.traced.value, v_in.requirement.id
            p_at = (v - self.V_OUT_REG_V) * self.I_LOAD_BUDGET_A
            return (
                f"supply {v:.12g} V ({req_v}) is outside {self.V_IN_MIN:.12g}..{self.V_IN_MAX:.12g} V: below {self.V_IN_MIN:.12g} V the input is less than "
                f"{self.V_OUT_REG_V:.12g} V + the L7805's ~{self.DROPOUT_TYP_V:.12g} V dropout (the regulator family's typical value, not a grounded datasheet fact), "
                f"so the +5V rail would not hold; above {self.V_IN_MAX:.12g} V the regulator dissipation (V_in - {self.V_OUT_REG_V:.12g} V) x {self.I_LOAD_BUDGET_A:.12g} A "
                f"at the design load budget" + (f" (here {p_at:.12g} W)" if p_at > 0 else "") + f" exceeds half of the {self.P_REG_MAX_W:.12g} W no-heatsink budget "
                f"of U2's TO-220 package (the template's conservative rule: a 2x margin because the load budget is an estimate)"
            )
        if f_in is not None and not self.F_MIN <= f_in.traced.value <= self.F_MAX:
            f, req_f = f_in.traced.value, f_in.requirement.id
            return (
                f"clock frequency {f:.12g} Hz ({req_f}) is outside {self.F_MIN:.12g}..{self.F_MAX:.12g} Hz: above {self.F_MAX / 1e6:.12g} MHz exceeds the "
                f"ATmega128-16A's maximum (the KiCad symbol's Description reads '16MHz'); below {self.F_MIN / 1e6:.12g} MHz is outside the crystal range the "
                f"ATmega128 datasheet gives for its oscillator (stated by the template, not grounded in this IR)"
            )
        return None

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        # range first: a present input outside the validity range refuses before a missing one is asked for (the answer
        # to that required question could only lead to this refusal)
        why = self._out_of_range(inputs)
        if why is not None:
            plan.inputs = {k: inputs[k] for k in self.needs if k in inputs}
            return _refused(plan, why)
        missing = [k for k in self.needs if k not in inputs]
        if missing:
            return _missing_inputs(plan, "The ATmega128 development board template", missing, unusable, examples={"input_voltage": "9 V", "clock_frequency": "16 MHz"}, ir=ir)
        f_in, v_in = inputs["clock_frequency"], inputs["input_voltage"]
        plan.inputs = {"clock_frequency": f_in, "input_voltage": v_in}
        req_f, req_v = f_in.requirement.id, v_in.requirement.id
        v, f = v_in.traced.value, f_in.traced.value

        def ch(key: str, value: float, unit: str | None, description: str) -> tuple[Choice, Traced]:
            return _choice(t, key, value, unit, description, confirmed)

        numbered = [
            ch("v_out_reg", self.V_OUT_REG_V, "V", "the L7805's nominal output voltage: the +5V rail (and the ideal stimulus V5V of the simulation)"),
            ch("i_load_budget", self.I_LOAD_BUDGET_A, "A", "design load budget of the +5V rail (the MCU at 16 MHz plus what the headers feed): the sizing value for the regulator's dissipation, not a measured current"),
            ch("p_reg_max", self.P_REG_MAX_W, "W", "no-heatsink dissipation budget kept for U2's TO-220 package (a rule of thumb, not a datasheet theta_JA fact); the 15 V input bound keeps the design load at half of it"),
            ch("c_in", self.C_IN_F, "F", "C1: input bulk electrolytic capacitor on VIN (after D1)"),
            ch("c_out", self.C_OUT_F, "F", "C2: output bulk electrolytic capacitor on +5V"),
            ch("c_dec", self.C_DEC_F, "F", "ceramic 100 nF: C3 at the regulator output, C4 / C5 at the two VCC pins, C6 on AVCC after L1, C7 on AREF"),
            ch("l_avcc", self.L_AVCC_H, "H", "L1: AVCC filter inductor between +5V and AVCC (with C6 an LC low-pass for the analog supply)"),
            ch("c_xtal", self.C_XTAL_F, "F", "C9 / C10: crystal load capacitors from XTAL1 / XTAL2 to GND"),
            ch("c_stray", self.C_STRAY_F, "F", "stray capacitance of the XTAL traces and U1's pins added to the series load (an estimate, not measured)"),
            ch("r_reset", self.R_RESET_OHM, "ohm", "R2: ~RESET pull-up to +5V"),
            ch("c_reset", self.C_RESET_F, "F", "C8: ~RESET capacitor to GND (with R2 the power-on reset delay tau = R2 C8)"),
            ch("c8_ic", self.C8_IC_V, "V", (
                "initial voltage across C8 for the transient (uic: the run starts from the elements' initial conditions instead of the operating point, "
                "where C8 would already be charged): the reset delay is simulated from a discharged C8 with the +5V rail present from t = 0"
            )),
            ch("r_pen", self.R_PEN_OHM, "ohm", "R3: ~PEN pull-up to +5V (keeps the programming-enable input inactive)"),
            ch("v_f_led", self.V_F_LED_V, "V", "forward voltage of the power LED D2 (a typical indicator value; R1 is solved from it)"),
            ch("i_led", self.I_LED_A, "A", "power LED current (an indicator at 2 mA; R1 is solved from it)"),
            ch("tol_rel", self.TOL_REL, None, (
                "relative tolerance of the i_led and v_reset_tau expectations (2 %: ideal parts, so the simulation reproduces the formulas; v(RESET) at t = tau is "
                "interpolated between the two neighbouring samples when ngspice's step control does not land on it, and both must be inside)"
            )),
            ch("v_avcc_tol_abs", self.V_AVCC_TOL_ABS_V, "V", "absolute tolerance of the v(AVCC) expectation (1 mV: the ideal inductor has no DC drop)"),
        ]
        models = [Choice("regulator_model", self.REGULATOR_MODEL), Choice("led_model", self.LED_MODEL), Choice("mcu_model", self.MCU_MODEL),
                  Choice("jack_pinout", self.JACK_PINOUT), Choice("cp_polarity", self.CP_POLARITY)]
        plan.choices = [c for c, _ in numbered] + models
        reg_prov = choice_provenance(t, f"regulator_model: {self.REGULATOR_MODEL}", confirmed)
        led_prov = choice_provenance(t, f"led_model: {self.LED_MODEL}", confirmed)
        mcu_prov = choice_provenance(t, f"mcu_model: {self.MCU_MODEL}", confirmed)
        jack_prov = choice_provenance(t, f"jack_pinout: {self.JACK_PINOUT}", confirmed)
        cp_prov = choice_provenance(t, f"cp_polarity: {self.CP_POLARITY}", confirmed)

        params: dict[str, Traced] = {"v_in": v_in.traced, "f_clk": f_in.traced}
        params.update({c.key: traced for c, traced in numbered})
        try:
            params["p_reg"] = regulator_dissipation(params["v_in"], params["v_out_reg"], params["i_load_budget"], ("v_in", "v_out_reg", "i_load_budget"))
            params["r_led"] = led_series_resistor(params["v_out_reg"], params["v_f_led"], params["i_led"], ("v_out_reg", "v_f_led", "i_led"))
            params["i_led_design"] = led_current(params["v_out_reg"], params["v_f_led"], params["r_led"], ("v_out_reg", "v_f_led", "r_led"))
            params["tau_reset"] = rc_time_constant(params["r_reset"], params["c_reset"], ("r_reset", "c_reset"))
            params["v_reset_tau"] = rc_step_response(params["v_out_reg"], params["tau_reset"], params["tau_reset"], ("v_out_reg", "tau_reset", "tau_reset"))
            params["tran_step"] = rc_tran_step(params["tau_reset"], ("tau_reset",))
            params["tran_stop"] = rc_tran_stop(params["tau_reset"], ("tau_reset",))
            params["c_load"] = crystal_load_capacitance(params["c_xtal"], params["c_xtal"], params["c_stray"], ("c_xtal", "c_xtal", "c_stray"))
            params["f_avcc"] = lc_cutoff(params["l_avcc"], params["c_dec"], ("l_avcc", "c_dec"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({req_v} = {v:.12g} V, {req_f} = {f:.12g} Hz)")
        plan.computed = [(k, params[k]) for k in ("p_reg", "r_led", "i_led_design", "tau_reset", "v_reset_tau", "tran_step", "tran_stop", "c_load", "f_avcc")]

        try:
            parts = self._parts(library, params, f, req_v, req_f)
            pins = self._pins(parts)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components = list(parts.values())
        self._bind(parts, params, reg_prov, led_prov, mcu_prov, jack_prov, cp_prov)
        nets = self._nets(pins, req_v, req_f)
        sim = self._simulation(params, reg_prov, led_prov, req_v)
        topology = self._topology(f)
        constraints = self._constraints(params, jack_prov)
        plan.parts = [_part_line(c) for c in components]
        plan.nets = [_net_line(n) for n in nets]
        excluded = [c for c in components if c.spice is not None and c.spice.exclude]
        plan.simulation = [
            f"stimuli V5V = {self.V_OUT_REG_V:.12g} V on +5V (the ideal L7805 output), VLED = {self.V_F_LED_V:.12g} V on LED_A (the ideal LED); C8 ic = {self.C8_IC_V:.12g} V; "
            f"analyses op and tran {params['tran_step'].value:.12g} {params['tran_stop'].value:.12g} uic (s)",
            f"expectation i_led: i(VLED) = {params['i_led_design'].value:.12g} A +/- {self.TOL_REL:.0%} at the operating point (current through the ideal source that replaces D2; no requirement)",
            f"expectation v_reset_tau: v(RESET) at t = tau_reset = {params['tau_reset'].value:.12g} s = {params['v_reset_tau'].value:.12g} V +/- {self.TOL_REL:.0%} "
            f"on the transient (C8 charging through R2 from 0 V; no requirement)",
            f"expectation v_avcc: v(AVCC) = {self.V_OUT_REG_V:.12g} V +/- {self.V_AVCC_TOL_ABS_V:.12g} V at the operating point (DC through the ideal L1; no requirement)",
            f"excluded from the netlist ({len(excluded)}): " + ", ".join(c.ref for c in excluded) + " (reasons in each part's SPICE binding)",
            "structural only, never simulated: U1 itself, the crystal oscillator (Y1, C9, C10), the ~PEN pull-up's function, the decoupling's high-frequency role, "
            "the regulator and its input side; PE0 / PE1 are shared by the ISP header J2 and the UART0 header J3 (see constraint c.atmega.isp_uart_shared)",
        ]
        plan.changes = _changes(t, self.title, topology, components, nets, params, sim, constraints)
        return plan

    # --- build steps ---------------------------------------------------------------------

    def _parts(self, library: KicadLibrary, params: dict[str, Traced], f: float, req_v: str, req_f: str) -> dict[str, Component]:
        """Every part from the library on disk, in reference order; ``TemplateRefusal`` names the first one missing."""
        t = self.id

        def part(ref: str, value: str, description: str, spec: tuple[tuple[str, str], tuple[str, str]], serves: list[str] | None = None) -> Component:
            return library_component(library, ref, value, description, *spec, structural_provenance(t, description), serves or [])

        dec = _value_text(params["c_dec"].value)
        out: dict[str, Component] = {}
        for c in (
            part("U1", MCU[0][1], "ATmega128 microcontroller (TQFP-64)", MCU, [req_f]),
            part("U2", REGULATOR[0][1], "5 V linear regulator (L7805, TO-220)", REGULATOR, [req_v]),
            part("J1", DC_JACK[0][1], "DC barrel jack: the power input", DC_JACK, [req_v]),
            part("D1", "1N4007", "series rectifier: reverse-polarity protection of the input", RECTIFIER, [req_v]),
            part("C1", _value_text(params["c_in"].value), f"input bulk electrolytic capacitor on VIN, rated >= {2 * self.V_IN_MAX:.12g} V (2 x the {self.V_IN_MAX:.12g} V input bound)",
                 CAP_POLARIZED, [req_v]),
            part("C2", _value_text(params["c_out"].value), f"regulator output bulk electrolytic capacitor, rated >= {2 * params['v_out_reg'].value:.12g} V (2 x the +5V rail)",
                 CAP_POLARIZED),
            part("C3", dec, "regulator output ceramic capacitor", CAPACITOR),
            part("D2", "LED", "power indicator LED", LED_0805),
            part("R1", _value_text(params["r_led"].value), "power LED series resistor", RESISTOR_0805),
            part("C4", dec, "decoupling capacitor at U1's first VCC pin", CAPACITOR),
            part("C5", dec, "decoupling capacitor at U1's second VCC pin", CAPACITOR),
            part("L1", _value_text(params["l_avcc"].value), "AVCC filter inductor", INDUCTOR),
            part("C6", dec, "AVCC filter capacitor", CAPACITOR),
            part("C7", dec, "AREF decoupling capacitor", CAPACITOR),
            part("R2", _value_text(params["r_reset"].value), "~RESET pull-up resistor", RESISTOR),
            part("C8", _value_text(params["c_reset"].value), "~RESET delay capacitor", CAPACITOR),
            part("SW1", PUSH_BUTTON[0][1], "reset push button", PUSH_BUTTON),
            part("R3", _value_text(params["r_pen"].value), "~PEN pull-up resistor", RESISTOR),
            part("Y1", f"{f / 1e6:.12g}MHz", "crystal: the MCU clock", CRYSTAL, [req_f]),
            part("C9", _value_text(params["c_xtal"].value), "XTAL1 load capacitor", CAPACITOR, [req_f]),
            part("C10", _value_text(params["c_xtal"].value), "XTAL2 load capacitor", CAPACITOR, [req_f]),
            part("J2", ISP_HEADER[0][1], "ISP header (AVR 6-pin: PDI / PDO / SCK / ~RESET)", ISP_HEADER),
            part("J3", HEADER_4[0][1], "UART0 header (GND / +5V / RXD0 / TXD0)", HEADER_4),
            *[part(ref, (HEADER_8 if bits == 8 else HEADER_5)[0][1], f"port {letter} header (P{letter}0..P{letter}{bits - 1})", HEADER_8 if bits == 8 else HEADER_5) for letter, ref, bits in PORTS],
            part("J11", HEADER_2[0][1], "power output header (+5V / GND)", HEADER_2),
        ):
            out[c.ref] = c
        return out

    def _pins(self, parts: dict[str, Component]) -> dict[str, dict[str, str] | dict[str, list[str]]]:
        """The pin numbers every net needs, found by *name* where the library names them; ``TemplateRefusal`` otherwise."""
        u1 = parts["U1"]
        lib_id = f"{u1.symbol.library}:{u1.symbol.name}"  # type: ignore[union-attr]
        by_name: dict[str, list[str]] = {}
        for p in u1.pins:
            by_name.setdefault(p.name, []).append(p.number)
        wanted = mcu_pin_names()
        missing = [n for n in wanted if n not in by_name]
        unexpected = [(p.number, p.name) for p in u1.pins if p.name not in wanted]
        if missing:
            raise TemplateRefusal(
                f"U1 ({lib_id}) has no pin named {missing} in this library (pins with other names: {unexpected}); "
                f"the template wires the MCU by pin name and will not guess which pin carries them"
            )
        if unexpected:
            raise TemplateRefusal(f"U1 ({lib_id}) has pins the template does not wire {unexpected}; every MCU pin must end in a net, so the template refuses this symbol")
        wrong_count = [(n, len(by_name[n]), k) for n, k in wanted.items() if len(by_name[n]) != k]
        if wrong_count:
            raise TemplateRefusal(
                "U1 (" + lib_id + ") pin name counts differ from the ATmega128's: "
                + ", ".join(f"{n}: {got} pin(s), expected {k}" for n, got, k in wrong_count)
                + " (C4 / C5 decouple the two VCC pins; each name must be one pin otherwise)"
            )
        u2 = parts["U2"]
        u2_pins = {name: pin_by_name(u2, name) for name in ("IN", "GND", "OUT")}
        if any(v is None for v in u2_pins.values()) or len(u2.pins) != 3:
            raise TemplateRefusal(f"{REGULATOR[0][0]}:{REGULATOR[0][1]} pins are not named IN / GND / OUT in this library (pins: {[(p.number, p.name) for p in u2.pins]}); the template will not guess the regulator's pinout")
        diodes: dict[str, dict[str, str]] = {}
        for ref in ("D1", "D2"):
            d = parts[ref]
            a, k = pin_by_name(d, "A"), pin_by_name(d, "K")
            if a is None or k is None or len(d.pins) != 2:
                raise TemplateRefusal(f"{d.symbol.library}:{d.symbol.name} ({ref}) pins are not named A / K in this library (pins: {[(p.number, p.name) for p in d.pins]}); the template will not guess the polarity")  # type: ignore[union-attr]
            diodes[ref] = {"A": a, "K": k}
        require_pins(parts["J1"], ("1", "2", "3"))
        if len(parts["J1"].pins) != 3:
            raise TemplateRefusal(f"J1 ({DC_JACK[0][0]}:{DC_JACK[0][1]}) has {len(parts['J1'].pins)} pins, the template wires three (centre, sleeve, switch)")
        for ref in ("C1", "C2"):
            require_pins(parts[ref], ("1", "2"))
            two_terminals(parts[ref])
        two: dict[str, tuple[str, str]] = {ref: two_terminals(parts[ref]) for ref in ("C3", "R1", "C4", "C5", "L1", "C6", "C7", "R2", "C8", "SW1", "R3", "Y1", "C9", "C10")}
        require_pins(parts["J2"], tuple(p for p, _, _ in ISP_PINS))
        require_pins(parts["J3"], tuple(p for p, _, _ in UART_PINS))
        for _, ref, bits in PORTS:
            require_pins(parts[ref], tuple(str(b + 1) for b in range(bits)))
        require_pins(parts["J11"], ("1", "2"))
        return {"U1": by_name, "U2": u2_pins, **diodes, **{ref: {"a": a, "b": b} for ref, (a, b) in two.items()}}  # type: ignore[dict-item]

    def _bind(self, parts: dict[str, Component], params: dict[str, Traced], reg_prov, led_prov, mcu_prov, jack_prov, cp_prov) -> None:
        """Electrical values (the BOM's) and SPICE bindings: included R / C / L at their design values, everything else excluded with the reason."""
        t = self.id

        def value(ref: str, device: SpiceDevice, key: str, extra: dict[str, Traced] | None = None, note: str = "") -> None:
            c = parts[ref]
            c.electrical[{SpiceDevice.R: "resistance", SpiceDevice.C: "capacitance", SpiceDevice.L: "inductance"}[device]] = params[key]
            c.spice = SpiceBinding(device=device, value=params[key], pin_order=[p.number for p in c.pins], params=dict(extra or {}),
                                   provenance=structural_provenance(t, note or f"ideal {device.value} at the design value"))

        def exclude(ref: str, reason: str, prov) -> None:
            parts[ref].spice = SpiceBinding(exclude=True, exclude_reason=reason, provenance=prov)

        for ref, device, key in (("R1", SpiceDevice.R, "r_led"), ("R2", SpiceDevice.R, "r_reset"), ("R3", SpiceDevice.R, "r_pen"), ("C2", SpiceDevice.C, "c_out"),
                                 ("C3", SpiceDevice.C, "c_dec"), ("C4", SpiceDevice.C, "c_dec"), ("C5", SpiceDevice.C, "c_dec"), ("C6", SpiceDevice.C, "c_dec"),
                                 ("L1", SpiceDevice.L, "l_avcc")):
            value(ref, device, key)
        value("C8", SpiceDevice.C, "c_reset", {"ic": params["c8_ic"]}, "ideal capacitor at the design value; its initial condition starts the reset transient (uic)")
        for ref, key in (("C1", "c_in"), ("C7", "c_dec"), ("C9", "c_xtal"), ("C10", "c_xtal")):
            parts[ref].electrical["capacitance"] = params[key]
        exclude("U1", self.MCU_MODEL, mcu_prov)
        exclude("U2", self.REGULATOR_MODEL, reg_prov)
        exclude("D1", "input rectifier on the unsimulated input side: the ideal regulator output V5V replaces everything upstream of U2 (regulator_model)", reg_prov)
        exclude("C1", "input bulk capacitor on the unsimulated input side: VIN reaches only excluded parts (D1, U2), so C1 alone would be a DC-floating node (regulator_model)", reg_prov)
        exclude("J1", "DC jack on the unsimulated input side; connector, no electrical model", jack_prov)
        exclude("D2", self.LED_MODEL, led_prov)
        exclude("Y1", "crystal: no authoritative motional model, and its oscillator amplifier is inside U1, which is not simulated (mcu_model)", mcu_prov)
        for ref, node in (("C7", "AREF: U1"), ("C9", "XTAL1: U1 and Y1"), ("C10", "XTAL2: U1 and Y1")):
            exclude(ref, f"its node {node} reaches only excluded parts, so it would be a DC-floating node (ngspice: singular matrix at the operating point); structural only (mcu_model)", mcu_prov)
        exclude("SW1", "reset push button: open in normal operation, no switch model; the reset delay is simulated with the button open", structural_provenance(t, "push button excluded from the netlist"))
        for ref in ("J2", "J3", *[r for _, r, _ in PORTS], "J11"):
            exclude(ref, "connector, no electrical model", structural_provenance(t, "header excluded from the netlist"))

    def _nets(self, pins: dict, req_v: str, req_f: str) -> list[Net]:
        """Every net by name, each pin found by name where the library names it; every U1 pin ends in exactly one net."""
        t = self.id
        mcu: dict[str, list[str]] = pins["U1"]
        u2, d1, d2 = pins["U2"], pins["D1"], pins["D2"]

        def a(ref: str) -> tuple[str, str]:
            return ref, pins[ref]["a"]

        def b(ref: str) -> tuple[str, str]:
            return ref, pins[ref]["b"]

        def u(name: str) -> tuple[str, str]:
            return "U1", mcu[name][0]

        extra: dict[str, list[tuple[str, str]]] = {}
        roles: dict[str, list[str]] = {}
        for header, kind, table in (("J2", "ISP", ISP_PINS), ("J3", "UART0", UART_PINS)):
            for pin, net, role in table:
                extra.setdefault(net, []).append((header, pin))
                roles.setdefault(net, []).append(f"{kind} {role} on {header}.{pin}")

        def net(name: str, kind: NetKind, members: list[tuple[str, str]], note: str, serves: list[str] | None = None) -> Net:
            return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in members], provenance=structural_provenance(t, note), serves_requirements=list(serves or []))

        out = [
            net("VIN_RAW", NetKind.POWER, [("J1", "1"), ("D1", d1["A"])], "the DC jack's centre pin (+) to D1's anode (pin-number assumption jack_pinout)", [req_v]),
            net("VIN", NetKind.POWER, [("D1", d1["K"]), ("C1", "1"), ("U2", u2["IN"])], "the protected input: D1's cathode, C1 (+), U2 IN", [req_v]),
            net("+5V", NetKind.POWER, [
                ("U2", u2["OUT"]), ("C2", "1"), a("C3"), a("R1"), *[("U1", n) for n in mcu["VCC"]], a("C4"), a("C5"), a("L1"), a("R2"), a("R3"), *extra["+5V"], ("J11", "1"),
            ], "the regulated 5 V rail: U2 OUT, C2 (+), C3, both VCC pins with C4 / C5, L1, the pull-ups, J2.2, J3.2, J11.1"),
            net("GND", NetKind.GROUND, [
                ("J1", "2"), ("J1", "3"), ("C1", "2"), ("U2", u2["GND"]), ("C2", "2"), b("C3"), ("D2", d2["K"]), *[("U1", n) for n in mcu["GND"]], b("C4"), b("C5"), b("C6"), b("C7"),
                b("C8"), b("SW1"), b("C9"), b("C10"), *extra["GND"], ("J11", "2"),
            ], "ground: the jack's sleeve and switch contact, every capacitor's return, the three U1 GND pins, the headers"),
            net("LED_A", NetKind.SIGNAL, [b("R1"), ("D2", d2["A"])], "R1 to the power LED's anode"),
            net("AVCC", NetKind.POWER, [b("L1"), a("C6"), u("AVCC")], "the filtered analog supply: L1 to AVCC with C6"),
            net("AREF", NetKind.ANALOG, [a("C7"), u("AREF")], "the ADC reference pin with its decoupling capacitor C7"),
            net("RESET", NetKind.SIGNAL, [b("R2"), a("C8"), a("SW1"), u("~{RESET}"), *extra["RESET"]], "~RESET: R2 pull-up, C8 delay, SW1 to GND, ISP J2.5"),
            net("PEN", NetKind.SIGNAL, [b("R3"), u("~{PEN}")], "~PEN held high by R3"),
            net("XTAL1", NetKind.SIGNAL, [a("Y1"), a("C9"), u("XTAL1")], "crystal Y1 to XTAL1 with the load capacitor C9", [req_f]),
            net("XTAL2", NetKind.SIGNAL, [b("Y1"), a("C10"), u("XTAL2")], "crystal Y1 to XTAL2 with the load capacitor C10", [req_f]),
        ]
        for letter, header, bits in PORTS:
            for bit in range(bits):
                name = f"P{letter}{bit}"
                note = f"port {letter} bit {bit}: U1 pin {mcu[name][0]} to {header}.{bit + 1}" + "".join(f"; {r}" for r in roles.get(name, []))
                out.append(net(name, NetKind.SIGNAL, [u(name), (header, str(bit + 1)), *extra.get(name, [])], note))
        return out

    def _simulation(self, params: dict[str, Traced], reg_prov, led_prov, req_v: str) -> SimulationSetup:
        t = self.id
        uic = Traced(value=True, provenance=params["c8_ic"].provenance)  # the same confirmed decision as C8's initial condition: the flag makes the ic act
        return SimulationSetup(
            stimuli=[
                Stimulus(id="V5V", source="voltage", net="+5V", reference_net="GND", kind=StimulusKind.DC, value=params["v_out_reg"], provenance=reg_prov, serves_requirements=[req_v]),
                Stimulus(id="VLED", source="voltage", net="LED_A", reference_net="GND", kind=StimulusKind.DC, value=params["v_f_led"], provenance=led_prov),
            ],
            analyses=[
                AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=structural_provenance(t, "operating point: the LED current and the AVCC level")),
                AnalysisSpec(
                    id="tran", kind=SpiceAnalysis.TRAN, params={"step": params["tran_step"], "stop": params["tran_stop"], "uic": uic},
                    provenance=structural_provenance(t, "transient of 5 tau from a discharged C8 (uic), 100 steps per tau"),
                ),
            ],
            expectations=[
                Expectation(
                    id="i_led", analysis_id="op", vector="i(VLED)", reduce=Reduce.VALUE, nominal=params["i_led_design"], tol_rel=params["tol_rel"],
                    provenance=structural_provenance(t, "the current through the ideal source VLED that replaces D2 (ideal constant-V_f model), not through a diode model"),
                ),
                Expectation(
                    id="v_reset_tau", analysis_id="tran", vector="v(RESET)", reduce=Reduce.AT, at=params["tau_reset"], nominal=params["v_reset_tau"], tol_rel=params["tol_rel"],
                    provenance=structural_provenance(t, "~RESET one time constant after power-up: C8 charged through R2 from 0 V to V(1 - e^-1)"),
                ),
                Expectation(
                    id="v_avcc", analysis_id="op", vector="v(AVCC)", reduce=Reduce.VALUE, nominal=params["v_out_reg"], tol_abs=params["v_avcc_tol_abs"],
                    provenance=structural_provenance(t, "the AVCC filter passes DC: v(AVCC) equals the +5V rail through the ideal L1"),
                ),
            ],
        )

    def _topology(self, f: float) -> Topology:
        t = self.id

        def block(bid: str, function: str, domain: CircuitDomain, refs: list[str], ins: list[str], outs: list[str]) -> Block:
            return Block(id=bid, function=function, domain=domain, component_refs=refs, input_nets=ins, output_nets=outs, provenance=structural_provenance(t, "block"))

        return Topology(
            name="ATmega128 development board", domains=[CircuitDomain.EMBEDDED, CircuitDomain.POWER, CircuitDomain.ANALOG],
            rationale=(
                "DC jack -> reverse-polarity diode -> L7805 5 V linear regulator; ATmega128 with a decoupling capacitor per VCC pin, an LC-filtered AVCC, "
                f"an AREF capacitor, an RC power-on reset with a push button, a ~PEN pull-up and a {f / 1e6:.12g} MHz crystal; every port on a header, "
                "ISP 2x3 and UART0 1x4 (PE0 / PE1 shared). The MCU and the regulator have no SPICE model: the analog parts around them are simulated "
                "(power LED current, reset RC delay, AVCC filter DC level), the rest is structural"
            ),
            provenance=structural_provenance(t, "selected by clock_frequency with input_voltage"),
            blocks=[
                block("power_input", "DC jack, reverse-polarity diode and L7805 5 V linear regulator", CircuitDomain.POWER, ["J1", "D1", "C1", "U2", "C2", "C3"], ["VIN_RAW"], ["+5V"]),
                block("power_led", "power indicator LED with series resistor", CircuitDomain.ANALOG, ["R1", "D2"], ["+5V"], ["LED_A"]),
                block("mcu", "ATmega128 with VCC decoupling and AREF capacitor", CircuitDomain.EMBEDDED, ["U1", "C4", "C5", "C7"],
                      ["+5V", "AVCC", "RESET", "PEN", "XTAL1", "XTAL2"], port_pin_names()),
                block("avcc_filter", "LC low-pass for the analog supply", CircuitDomain.ANALOG, ["L1", "C6"], ["+5V"], ["AVCC"]),
                block("reset", "RC power-on reset with push button", CircuitDomain.ANALOG, ["R2", "C8", "SW1"], ["+5V"], ["RESET"]),
                block("pen", "~PEN pull-up", CircuitDomain.ANALOG, ["R3"], ["+5V"], ["PEN"]),
                block("clock", "crystal with load capacitors (Pierce oscillator inside U1)", CircuitDomain.EMBEDDED, ["Y1", "C9", "C10"], [], ["XTAL1", "XTAL2"]),
                block("headers", "port, ISP, UART0 and power headers", CircuitDomain.EMBEDDED, ["J2", "J3", *[r for _, r, _ in PORTS], "J11"], port_pin_names(), []),
            ],
        )

    def _constraints(self, params: dict[str, Traced], jack_prov) -> list[Constraint]:
        t = self.id
        return [
            Constraint(
                id="c.atmega.isp_uart_shared", kind=ConstraintKind.ELECTRICAL, target="PE0",
                description=(
                    "PE0 (PDI / RXD0) and PE1 (PDO / TXD0) are shared by the ISP header J2 and the UART0 header J3: the ATmega128 is serially programmed "
                    "through PDI / PDO, not through MOSI / MISO on PB2 / PB3 (datasheet doc2467; stated by the template, not grounded in this IR) - "
                    "disconnect whatever drives J3 while programming through J2"
                ),
                provenance=structural_provenance(t, "pin sharing the header layout implies"),
            ),
            Constraint(
                id="c.atmega.regulator_thermal", kind=ConstraintKind.THERMAL, target="U2",
                parameters={"p_dissipation": params["p_reg"], "p_budget": params["p_reg_max"]},
                description=(
                    f"U2 dissipates about (V_in - V_out) x I_load = {quantity(params['p_reg'].value, 'W')} at the design load budget (D1's drop, which would lower it, "
                    f"is not counted: an upper estimate); keep P <= {quantity(params['p_reg_max'].value, 'W')} for the TO-220 without a heatsink - the rule the "
                    "15 V input bound encodes with a 2x margin; theta_JA and the junction temperature are not grounded, so no thermal verdict is claimed"
                ),
                provenance=structural_provenance(t, "template validity condition"),
            ),
            Constraint(
                id="c.atmega.crystal_load", kind=ConstraintKind.ELECTRICAL, target="Y1", parameters={"c_load": params["c_load"]},
                description=(
                    f"the load the oscillator presents, C_L = C9 C10 / (C9 + C10) + C_stray = {quantity(params['c_load'].value, 'F')}, must match the fitted crystal's "
                    "specified load capacitance (its datasheet): not grounded here, compared with nothing"
                ),
                provenance=structural_provenance(t, "part selection condition"),
            ),
            Constraint(
                id="c.atmega.polarity_by_pin_number", kind=ConstraintKind.ELECTRICAL, target="J1",
                description=f"{self.JACK_PINOUT}; {self.CP_POLARITY}",
                provenance=jack_prov,
            ),
        ]

    # --- report hooks (views: numbers recomputed from ir.parameters with the formulas shown, nothing written) ---

    _KEYS: tuple[str, ...] = (
        "v_in", "f_clk", "v_out_reg", "i_load_budget", "p_reg_max", "c_in", "c_out", "c_dec", "l_avcc", "c_xtal", "c_stray", "r_reset", "c_reset", "c8_ic",
        "r_pen", "v_f_led", "i_led", "tol_rel", "v_avcc_tol_abs", "p_reg", "r_led", "i_led_design", "tau_reset", "v_reset_tau", "tran_step", "tran_stop",
        "c_load", "f_avcc",
    )

    def _numbers(self, ir: CircuitIR) -> dict[str, float | None]:
        return {k: parameter_value(ir, k) for k in self._KEYS}

    def _p_at(self, n: dict[str, float | None], v: float) -> float | None:
        """(V − V_out)·I_load at a display voltage ``v`` (the formula of ``calc.regulator.p_dissipation``)."""
        return _mul(_sub(v, n["v_out_reg"]), n["i_load_budget"])

    # --- signal integrity: only what this circuit needs ------------------------------------

    #: the crystal loop's copper budget per XTAL net (MCU pin, crystal, load capacitor): a layout rule, not a datasheet number
    XTAL_MAX_LENGTH_MM = 25.0
    #: the temperature rise the supply rails' IPC-2221 minimum width is sized for
    POWER_TEMP_RISE_C = 10.0
    #: SCK of serial programming at f_clk / 4: faster than the ATmega128 accepts (its serial-programming rule: SCK high and low each
    #: > 2 CPU clock cycles below 12 MHz, >= 3 at 12 MHz and above; f_clk / 4 gives exactly 2), so the timing margins computed at
    #: this rate are pessimistic - a conservative choice, not the device's highest rate (stated by the template, not grounded)
    ISP_SCK_DIVIDER = 4.0
    #: SPI mode 0: MOSI changes on the falling SCK edge and is sampled on the rising one, half a period later
    ISP_CAPTURE_FRACTION = 0.5
    #: the supply / ground nets of the POWER class
    POWER_NETS: tuple[str, ...] = ("VIN_RAW", "VIN", "+5V", "GND", "AVCC")

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """The crystal-loop length, the supply rails' IPC-2221 minimum width, the ISP SPI timing path (module docstring of :mod:`ai_eda.design.board`).

        U1 is the driver of the default class: its grounded datasheet facts
        (``t_rise``, ``r_out``, ``c_in``) replace the conservative choices on
        the nets it drives (an output / bidirectional pin there,
        :mod:`ai_eda.tools.si.driver`) - MISO (PDO, ``PE1``) among them. SCK
        and MOSI (``PB1`` / ``PE0``) are driven by the programmer behind J2
        during serial programming (and by U1 as port pins otherwise), so the
        ``ISP_SPI`` class names no driver: the conservative edge stands. The
        programmer is not a part of this IR, so its clock-to-output time is
        named ``J2.t_co`` (grounded, if ever, on the programmer's own
        document); ``U1.t_su`` / ``U1.t_h`` are U1's. Offline they are absent
        and the timing path is NOT_VERIFIED naming them. ``RESET`` is the RC
        reset (R2 to +5V, C8 to GND: ``tau_reset`` = R2·C8, pulled low by SW1
        or the programmer) - no fast driven edge, so its class states no edge
        and the critical-length rule and ``spice.si`` leave it alone, like the
        crystal loop.
        """
        from ai_eda.design.board import SIDeclarations
        from ai_eda.tools.calc.basic import clock_divided, ipc2221_width_for_current

        out = SIDeclarations(driver="U1")
        xtal = ctx.choice("si.xtal_max_length", self.XTAL_MAX_LENGTH_MM, "mm", (
            "keep the oscillator loop short: at most this much copper per XTAL net (U1 pin, crystal Y1, load capacitor C9 / C10) - "
            "a layout rule of thumb, not a datasheet number"))
        out.classes.append(NetClass(name="XTAL", nets=["XTAL1", "XTAL2"], max_length_mm=xtal, description="the crystal loop: short copper, no driven edge",
                                    provenance=ctx.structural("crystal loop length budget")))
        dt = ctx.choice("power_temp_rise", self.POWER_TEMP_RISE_C, "degC", (
            "temperature rise the supply rails' IPC-2221 minimum width is sized for (at the i_load_budget current)"), param=True)
        if ctx.stackup is not None:
            t_id = "pcb.stackup.copper[F.Cu].thickness_um"
            layer = ctx.stackup.copper_layer("F.Cu")
            assert layer is not None
            w = ctx.computed_param("w_power_min", ipc2221_width_for_current(ctx.params["i_load_budget"], dt, layer.thickness_um, ("i_load_budget", "power_temp_rise", t_id)))
            out.classes.append(NetClass(
                name="POWER", nets=list(self.POWER_NETS), min_width_mm=w, power_current_a=ctx.params["i_load_budget"], power_temp_rise_c=dt,
                description="the supply and ground rails: at least the IPC-2221 width for the load budget", provenance=ctx.structural("supply rails' minimum width"),
            ))
        div = ctx.choice("isp_sck_divider", self.ISP_SCK_DIVIDER, None, (
            "serial programming clock SCK = f_clk / 4 for the timing path: faster than the ATmega128 accepts (its serial-programming rule: "
            "SCK high and low each > 2 CPU clock cycles below 12 MHz, >= 3 at 12 MHz and above; f_clk / 4 gives exactly 2), so the margins "
            "are pessimistic - a conservative choice, not the device's highest rate; stated by the template, not grounded"), param=True)
        capture = ctx.choice("si.isp_capture_fraction", self.ISP_CAPTURE_FRACTION, None, (
            "SPI mode 0: MOSI is launched on the falling SCK edge and sampled on the rising one, half a period later"))
        f_sck = ctx.computed_param("f_sck", clock_divided(ctx.params["f_clk"], div, ("f_clk", "isp_sck_divider")))
        drv = ctx.driver
        out.classes.append(NetClass(
            name="ISP_SPI", nets=["PB1", "PE0"],
            description=("the ISP header's SCK and MOSI (PDI): driven by the programmer at J2 during serial programming (off-board) and by U1 "
                         "as port pins otherwise - no single driver's facts apply, the conservative edge stands; MISO (PDO, PE1) is U1's output "
                         "and stays in the default class"),
            t_rise_s=drv["t_rise_s"], r_drive_ohm=drv["r_drive_ohm"], c_load_f=drv["c_load_f"], ringing_tol_rel=drv["ringing_tol_rel"], driver=None,
            promote_to="Z50", provenance=ctx.structural("ISP interface nets"),
        ))
        out.classes.append(NetClass(
            name="RC_RESET", nets=["RESET"],
            description=("~RESET: the RC reset (R2 to +5V, C8 to GND, tau_reset = R2*C8 from calc.rc.tau), pulled low by SW1 or the programmer "
                         "at J2 - no fast driven edge, so no critical length and no SPICE line check"),
            provenance=ctx.structural("RC reset net"),
        ))
        out.timing_paths.append(TimingPath(
            name="ISP", clock_net="PB1", data_nets=["PE0"], direction="ISP programmer at J2 -> U1 (MOSI / PDI sampled by U1 on the rising SCK edge)",
            f_clk_hz=f_sck, capture_fraction=capture,
            terms_from={"t_co_max_s": "J2.t_co", "t_co_min_s": "J2.t_co_min", "t_su_min_s": "U1.t_su", "t_h_min_s": "U1.t_h"},
            provenance=ctx.structural("ISP serial programming: SCK -> MOSI"),
        ))
        return out

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        n = self._numbers(ir)
        v_in, v_out, i_load, p_reg, p_max = n["v_in"], n["v_out_reg"], n["i_load_budget"], n["p_reg"], n["p_reg_max"]
        tau, v_tau = n["tau_reset"], n["v_reset_tau"]
        f0, f_clk = n["f_avcc"], n["f_clk"]
        headroom = _sub(_sub(v_in, self.V_D1_DISPLAY_V), v_out)
        v_bound = _add(v_out, _div(p_max, _mul(2.0, i_load)))  # V_out + P_budget / (2 I_load): where the design load reaches half the budget
        i_press = _div(v_out, n["r_reset"])
        h_clk = None if not _known(f0, f_clk) or f0 <= 0 or f_clk == f0 else 1.0 / abs(1.0 - (f_clk / f0) ** 2)
        p_rows = "".join(f"| {quantity(v, 'V')} | {quantity(self._p_at(n, v), 'W')} |\n" for v in (self.V_IN_MIN, 9.0, 12.0, self.V_IN_MAX))
        dec = quantity(n["c_dec"], "F")
        # a narrow tree (ASCII / symbols only, no double-width text) so the code block never wraps in the HTML / PDF column
        diagram = "\n".join([
            f"J1 DC jack ({quantity(self.V_IN_MIN, 'V')}..{quantity(self.V_IN_MAX, 'V')})",
            f" └ VIN_RAW ─ D1 1N4007 ─ VIN (C1 {quantity(n['c_in'], 'F')}) ─ U2 L7805",
            f"    └ +5V (C2 {quantity(n['c_out'], 'F')}, C3 {dec})",
            f"       ├ U1 VCC pins 21, 52 (C4, C5 {dec})",
            f"       ├ L1 {quantity(n['l_avcc'], 'H')} ─ AVCC ─ U1 AVCC pin 64 (C6)",
            f"       ├ R1 {quantity(n['r_led'], 'ohm')} ─ LED_A ─ D2 ─ GND",
            f"       ├ R2 {quantity(n['r_reset'], 'ohm')} ─ RESET ─ U1 ~RESET pin 20 (C8, SW1, J2.5)",
            f"       ├ R3 {quantity(n['r_pen'], 'ohm')} ─ PEN ─ U1 ~PEN pin 1",
            "       └ J2.2, J3.2, J11.1",
            f"Y1 {quantity(f_clk, 'Hz')} ─ XTAL1 pin 24 / XTAL2 pin 23 (C9, C10 {quantity(n['c_xtal'], 'F')})",
            "U1 AREF pin 62 (C7)",
            "PA..PF → J4..J9 (1x8), PG0..PG4 → J10 (1x5)",
            "ISP J2 (2x3), UART0 J3 (1x4), 5 V out J11 (1x2)",
        ])
        def cell(rule: str) -> str:
            """A pass rule as a table cell: its absolute-value bars escaped, so they neither split the row nor vanish."""
            return rule.replace("|", "\\|")

        sim_rows = (
            "| 기대값 | 해석 | 벡터 | 공칭값 (계산기) | 판정 |\n|---|---|---|---|---|\n"
            f"| `i_led` | op | i(VLED) | {quantity(n['i_led_design'], 'A')} (`calc.led.I`) | {cell(_judging_line(n['i_led_design'], 'A', tol_rel=n['tol_rel']))} |\n"
            f"| `v_reset_tau` | tran (uic) | v(RESET) @ t = τ = {quantity(tau, 's')} | {quantity(v_tau, 'V')} (`calc.rc.step_response`) | {cell(_judging_line(v_tau, 'V', tol_rel=n['tol_rel']))} |\n"
            f"| `v_avcc` | op | v(AVCC) | {quantity(v_out, 'V')} (선택값 v_out_reg) | {cell(_judging_line(v_out, 'V', tol_abs=n['v_avcc_tol_abs']))} |"
        )
        return [
            TheorySection("보드 구성과 신호 흐름", (
                "```\n" + diagram + "\n```\n\n"
                "DC 잭으로 들어온 전압은 역접속 보호 다이오드 D1 을 지나 선형 레귤레이터 U2(L7805)에서 5 V 레일(+5V)이 됩니다. "
                "+5V 는 MCU U1(ATmega128)의 두 VCC 핀(각각 바이패스 커패시터 C4·C5), LC 필터 L1·C6 을 거친 아날로그 전원 AVCC, 전원 LED, 리셋·PEN 풀업, "
                "그리고 ISP·UART·전원 헤더로 갑니다. 크리스탈 Y1 과 부하 커패시터 C9·C10 이 MCU 내부 발진기의 클록을 정하고, 모든 포트 핀은 헤더로 나옵니다. "
                "모든 부품·핀 이름·풋프린트는 디스크의 KiCad 라이브러리에서 읽었고, MCU 는 핀 *이름* 으로 배선했습니다(64핀 모두 넷에 연결)."
            )),
            TheorySection("전원부: L7805 와 레귤레이터 손실", (
                "선형 레귤레이터는 입력과 출력의 전압 차를 부하 전류만큼 열로 버립니다.\n\n"
                "    P = (V_in − V_out)·I_load   (`calc.regulator.p_dissipation`)\n"
                f"      = ({quantity(v_in, 'V')} − {quantity(v_out, 'V')})·{quantity(i_load, 'A')} = {quantity(p_reg, 'W')}\n\n"
                f"I_load = {quantity(i_load, 'A')} 는 측정값이 아니라 +5V 레일의 설계 예산(선택값)입니다. TO-220 을 방열판 없이 쓸 때 템플릿이 지키는 예산은 "
                f"{quantity(p_max, 'W')}(선택값 `p_reg_max`, 경험칙이며 데이터시트 θ_JA 로 근거를 둔 값이 아님)이고, 입력 상한 {quantity(self.V_IN_MAX, 'V')} 은 "
                f"V_out + P_예산/(2·I_load) = {quantity(v_bound, 'V')} 로 예산의 절반에서 멈추는 규칙입니다(부하 예산이 추정치이므로 2배 여유).\n\n"
                "| 입력 전압 | P = (V_in − 5 V)·I_load |\n|---|---|\n" + p_rows + "\n"
                f"이 식은 D1 의 순방향 강하를 빼지 않은 V_in 을 쓰므로 실제 손실보다 큰 상한 추정입니다. 반대로 D1 강하(1N4007 계열 수십 mA 에서 약 {quantity(self.V_D1_DISPLAY_V, 'V')}, "
                f"이 IR 에 근거 없음)는 레귤레이터 입력을 낮춥니다: 이 설계에서 U2 입력과 출력의 차이는 약 {quantity(headroom, 'V')} 입니다. "
                f"L7805 계열의 전형적 드롭아웃은 약 {quantity(self.DROPOUT_TYP_V, 'V')}(1 A 기준 전형값, 근거 없음)이며, 입력 하한 {quantity(self.V_IN_MIN, 'V')} 근처에서는 "
                "이 차이가 그보다 작아질 수 있어 경부하 드롭아웃에 기대게 됩니다 — 이 파이프라인은 그것을 검증하지 않았습니다(레귤레이터는 시뮬레이션하지 않음). "
                "열 판정(`domain.power.thermal`)은 θ_JA 가 IR 에 없으므로 PASS 를 주장하지 않습니다(제약 조건 `c.atmega.regulator_thermal`)."
            )),
            TheorySection("리셋 회로: RC 지연", (
                "~RESET 은 R2 로 +5V 에 풀업되고 C8 로 GND 에 연결됩니다. 전원이 들어오면 C8 이 R2 를 통해 충전되므로 ~RESET 이 천천히 올라가 MCU 가 전원 안정 뒤에 리셋에서 풀려납니다.\n\n"
                f"    τ = R2·C8 = {quantity(n['r_reset'], 'ohm')}·{quantity(n['c_reset'], 'F')} = {quantity(tau, 's')}   (`calc.rc.tau`)\n"
                "    v(t) = V_cc·(1 − e^(−t/τ))\n"
                f"    v(τ) = V_cc·(1 − e^(−1)) = {quantity(v_out, 'V')}·0.63212 = {quantity(v_tau, 'V')}   (`calc.rc.step_response`)\n\n"
                f"SW1 을 누르면 ~RESET 이 GND 로 당겨져 리셋되고, 그동안 R2 에는 V_cc/R2 = {quantity(i_press, 'A')} 가 흐릅니다. ISP 헤더 J2 의 5번 핀도 같은 RESET 넷이라 "
                "프로그래머가 ~RESET 을 내릴 수 있습니다. MCU 의 리셋 문턱 전압과 내부 POR/BOD 는 이 IR 에 근거가 없어 판정에 쓰지 않습니다."
            )),
            TheorySection("AVCC LC 필터", (
                "AVCC(ADC 와 포트 F 의 전원)는 L1 과 C6 의 2차 저역통과 필터를 거친 +5V 입니다.\n\n"
                f"    f_0 = 1/(2π·√(L1·C6)) = 1/(2π·√({quantity(n['l_avcc'], 'H')}·{quantity(n['c_dec'], 'F')})) = {quantity(f0, 'Hz')}   (`calc.lc.cutoff`)\n"
                "    |H(f)| = 1/|1 − (f/f_0)²|   (이상적인 L·C, 손실 없음)\n\n"
                f"f_0 위에서는 10배마다 −40 dB 로 줄어듭니다: 클록 f_clk = {quantity(f_clk, 'Hz')} 에서 이론 |H| = {number(h_clk, 4)} 입니다. "
                "이상적인 부품이라 f_0 의 공진 봉우리는 이론상 무한대이며, 실제 봉우리는 L1 의 권선 저항과 C6 의 ESR 이 정합니다(모델에 없음). "
                f"DC 에서 이상적 인덕터는 전압 강하가 없으므로 동작점(op) 해석의 v(AVCC) 는 +5V 와 같아야 합니다(기대값 `v_avcc`). "
                "과도 해석은 uic 로 시작해 L1·C6 도 0 A·0 V 에서 출발하므로 저항 없는 이상 회로에서 f_0 로 감쇠 없이 진동합니다 — 그래서 AVCC 는 과도 파형이 아니라 동작점에서만 판정합니다."
            )),
            TheorySection("크리스탈과 부하 커패시턴스", (
                "MCU 내부의 피어스(Pierce) 발진기는 XTAL1–XTAL2 사이의 크리스탈에 두 부하 커패시터의 직렬 합과 배선·핀의 표유 용량을 부하로 보여 줍니다.\n\n"
                "    C_L = C9·C10/(C9 + C10) + C_stray   (`calc.crystal.load_capacitance`)\n"
                f"        = {quantity(n['c_xtal'], 'F')}·{quantity(n['c_xtal'], 'F')}/({quantity(n['c_xtal'], 'F')} + {quantity(n['c_xtal'], 'F')}) + {quantity(n['c_stray'], 'F')} = {quantity(n['c_load'], 'F')}\n\n"
                f"C_stray = {quantity(n['c_stray'], 'F')} 는 추정 선택값입니다. C_L 은 실제로 장착할 크리스탈의 데이터시트 부하 용량과 같아야 정확한 주파수로 발진합니다 — "
                "그 데이터시트는 이 IR 에 없으므로 이 값은 아무것과도 비교되지 않았습니다(제약 조건 `c.atmega.crystal_load`). "
                f"클록 f_clk = {quantity(f_clk, 'Hz')} 의 상한 {quantity(self.F_MAX, 'Hz')} 는 KiCad 심볼 Description 의 '16MHz' 입니다. "
                "ATmega128 은 공장 출하 시 내부 RC 발진기와 ATmega103 호환 모드(M103C 퓨즈 프로그램됨)로 설정되어 있습니다. 외부 크리스탈을 쓰려면 CKSEL 퓨즈를 "
                "크리스탈 모드로 프로그램해야 하고, CKOPT 퓨즈가 프로그램되지 않은 크리스탈 모드는 8 MHz 까지이므로 그보다 높은 크리스탈은 CKOPT 도 프로그램해야 합니다"
                + (f"(이 설계의 f_clk = {quantity(f_clk, 'Hz')} 가 그 경우: CKSEL3..1 = 111, CKOPT = 0)" if f_clk is not None and f_clk > 8e6 else "")
                + ". ATmega128 모드로 쓰려면 M103C 도 해제합니다. 또 공장 출하 시 JTAGEN 퓨즈가 프로그램되어 있어 PF4..PF7(J9 의 5..8번 핀)은 JTAG 핀(TCK / TMS / TDO / TDI)이며, "
                "JTAGEN 을 해제하거나 MCUCSR 의 JTD 비트를 설정하기 전에는 일반 입출력으로 쓸 수 없습니다 — 모두 데이터시트(doc2467) 기재 사항이며 이 IR 에 근거를 둔 사실은 아닙니다."
            )),
            TheorySection("ISP 와 UART0 의 핀 공유, ~PEN", (
                "| J2 (ISP 2x3) | 넷 | 역할 |\n|---|---|---|\n"
                + "".join(f"| {pin} | `{net}` | {role} |\n" for pin, net, role in ISP_PINS)
                + "\n| J3 (UART0 1x4) | 넷 | 역할 |\n|---|---|---|\n"
                + "".join(f"| {pin} | `{net}` | {role} |\n" for pin, net, role in UART_PINS)
                + "\nATmega128 의 직렬 프로그래밍은 PB2/PB3(MOSI/MISO)가 아니라 PDI = PE0, PDO = PE1, SCK = PB1 을 씁니다(데이터시트 doc2467 의 'Serial Programming Pin Mapping'; "
                "이 IR 에 근거를 둔 사실은 아님). PE0/PE1 은 UART0 의 RXD0/TXD0 이기도 하므로 J2.4 와 J3.3 은 같은 넷 `PE0`, J2.1 과 J3.4 는 같은 넷 `PE1` 입니다: "
                "ISP 로 프로그래밍하는 동안에는 J3 에 연결한 UART 어댑터를 빼야 합니다(제약 조건 `c.atmega.isp_uart_shared`). RXD0 은 MCU 의 수신 입력이므로 어댑터의 TX 와 연결합니다. "
                f"~PEN 은 전원 리셋 중 LOW 이면 프로그래밍 모드를 강제하는 핀이라 R3 = {quantity(n['r_pen'], 'ohm')} 로 +5V 에 풀업해 둡니다(데이터시트 기재 사항, 근거 없음)."
            )),
            TheorySection("시뮬레이션: 검증하는 것과 하지 않는 것", (
                "U1(ATmega128)과 U2(L7805)는 KiCad 라이브러리에 SPICE 모델이 없어 넷리스트에서 제외합니다. 레귤레이터 출력은 이상 전압원 V5V = "
                f"{quantity(v_out, 'V')} 로 대체하고, 그 위쪽(J1, D1, C1)은 시뮬레이션하지 않습니다. 전원 LED D2 는 LED 템플릿과 같은 이상 정전압(V_f) 원 VLED 로 대체합니다. "
                "C7(AREF), C9·C10(XTAL1·XTAL2)은 한쪽 노드가 제외된 부품(U1, Y1)에만 닿아 DC 경로가 없는 노드가 되므로(ngspice op 가 'singular matrix' 로 실패) 함께 제외합니다.\n\n"
                + sim_rows + "\n\n"
                f"과도 해석 창(`calc.rc.tran_step` / `calc.rc.tran_stop`): step = τ/100 = {quantity(n['tran_step'], 's')}, stop = 5·τ = {quantity(n['tran_stop'], 's')}, "
                f"`uic` 로 동작점 계산을 건너뛰고 C8 의 초기 전압 {quantity(n['c8_ic'], 'V')} 에서 시작합니다(확인된 선택값). "
                "t = τ 가 step 의 정수배라도 ngspice 의 시간 간격 제어가 그 점에 정확히 떨어지지 않으면 이웃 두 표본 사이를 보간하고, 두 이웃이 모두 허용치 안일 때만 PASS 입니다.\n\n"
                "시뮬레이션하지 않는 것(구조만 확인): MCU 자체의 동작, 크리스탈 발진(Y1, C9, C10), ~PEN 풀업의 기능, 바이패스 커패시터의 고주파 역할, 레귤레이터와 입력부. "
                "이 파이프라인의 어떤 결과도 MCU 가 동작한다고 주장하지 않습니다. ERC/DRC 는 kicad-cli 가 있을 때만 판정됩니다."
            )),
            TheorySection("유효 범위(템플릿이 거부하는 조건)와 그 이유", (
                "| 조건 | 근거 |\n|---|---|\n"
                f"| {quantity(self.V_IN_MIN, 'V')} ≤ V_in ≤ {quantity(self.V_IN_MAX, 'V')} | {quantity(self.V_IN_MIN, 'V')} 미만: 5 V + L7805 계열의 전형 드롭아웃 약 "
                f"{quantity(self.DROPOUT_TYP_V, 'V')}(전형값, 근거 없음)에 못 미침. {quantity(self.V_IN_MAX, 'V')} 초과: 설계 부하 예산에서의 손실이 무방열판 예산 "
                f"{quantity(p_max, 'W')} 의 절반을 넘음(템플릿의 보수적 규칙) |\n"
                f"| {quantity(self.F_MIN, 'Hz')} ≤ f_clk ≤ {quantity(self.F_MAX, 'Hz')} | {quantity(self.F_MAX, 'Hz')} 초과: ATmega128-16A 의 최대 클록(KiCad 심볼 Description 의 '16MHz'). "
                f"{quantity(self.F_MIN, 'Hz')} 미만: ATmega128 데이터시트가 크리스탈 발진기에 주는 범위 밖(기재 사항, 근거 없음) |"
            )),
        ]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """(a) regulator dissipation against V_in with the validity window, the budget and half of it; (b) the reset step response with
        the tau marker; (c) the ideal AVCC LC filter |H(f)| with f_0 and the clock marked; (d) the crystal load C_L against the load capacitors."""
        n = self._numbers(ir)
        out: list[Figure] = []
        v_in, v_out, i_load, p_reg, p_max = n["v_in"], n["v_out_reg"], n["i_load_budget"], n["p_reg"], n["p_reg_max"]
        if _known(v_in, v_out, i_load, p_reg) and i_load > 0:
            hi = max(self.V_IN_MAX * 1.4, v_in)
            if p_max is not None and p_max > 0:
                hi = max(hi, v_out + 1.1 * p_max / i_load)
            xs = _lin_grid(v_out, hi, 200)
            ys = [(x - v_out) * i_load for x in xs]
            bands = [("x", self.V_IN_MIN, self.V_IN_MAX, f"유효 범위 {quantity(self.V_IN_MIN, 'V')} … {quantity(self.V_IN_MAX, 'V')}")]
            if p_max is not None:
                bands += [("y", p_max, p_max, f"무방열판 예산 {quantity(p_max, 'W')}"), ("y", p_max / 2.0, p_max / 2.0, f"템플릿 한계 = 예산의 절반 {quantity(p_max / 2.0, 'W')}")]
            caption = (
                f"'전원부' 절의 식 P = (V_in − V_out)·I_load (`calc.regulator.p_dissipation`) 을 이 설계의 V_out = {quantity(v_out, 'V')}, I_load = {quantity(i_load, 'A')} (설계 부하 예산) 로 "
                f"입력 전압에 대해 그린 것. 음영 = 템플릿 유효 범위, 안내선 = 무방열판 예산과 그 절반, 점 = 설계점 V_in = {quantity(v_in, 'V')}, P = {quantity(p_reg, 'W')}. "
                f"D1 강하를 빼지 않은 상한 추정입니다. {THEORY_CURVE_NOTE}"
            )
            out.append(_curve_figure(
                "theory_p_reg", "레귤레이터 손실 대 입력 전압", caption, [("P(V_in)", xs, ys)], x_label="입력 전압 V_in (V)", y_label="손실 P (W)", bands=bands,
                markers=[(v_in, p_reg, f"설계점 V_in = {quantity(v_in, 'V')}, P = {quantity(p_reg, 'W')}")],
            ))
        tau, v_tau = n["tau_reset"], n["v_reset_tau"]
        if _known(tau, v_tau, v_out) and tau > 0:
            xs = _lin_grid(0.0, 5.0 * tau, 300)
            ys = [v_out * (1.0 - math.exp(-x / tau)) for x in xs]
            caption = (
                f"'리셋 회로' 절의 식 v(t) = V_cc·(1 − e^(−t/τ)) 을 이 설계의 V_cc = {quantity(v_out, 'V')}, τ = R2·C8 = {quantity(tau, 's')} (`calc.rc.tau`) 로 "
                f"과도 해석 창 0 … 5τ 에 그린 것. 점 = t = τ 에서 v = {quantity(v_tau, 'V')} (`calc.rc.step_response`: 기대값 `v_reset_tau` 의 공칭값). {THEORY_CURVE_NOTE}"
            )
            out.append(_curve_figure(
                "theory_reset_rc", "~RESET 충전 곡선 (RC 지연)", caption, [("v_RESET(t)", xs, ys)], x_label="시간 t (s)", y_label="~RESET 전압 (V)",
                bands=[("y", v_tau, v_tau, f"v(τ) = {quantity(v_tau, 'V')}")], markers=[(tau, v_tau, f"τ = {quantity(tau, 's')}")],
            ))
        f0, f_clk = n["f_avcc"], n["f_clk"]
        if _known(f0) and f0 > 0:
            hi = max(100.0 * f0, 2.0 * f_clk) if f_clk is not None else 100.0 * f0
            xs = [x for x in _log_grid(f0 / 100.0, hi, 300) if x != f0]
            ys = [1.0 / abs(1.0 - (x / f0) ** 2) for x in xs]
            markers = []
            if f_clk is not None and f_clk > 0 and f_clk != f0:
                h_clk = 1.0 / abs(1.0 - (f_clk / f0) ** 2)
                markers.append((f_clk, h_clk, f"f_clk = {quantity(f_clk, 'Hz')}, |H| = {number(h_clk, 3)}"))
            caption = (
                f"'AVCC LC 필터' 절의 식 |H(f)| = 1/|1 − (f/f_0)²| 을 이 설계의 L1 = {quantity(n['l_avcc'], 'H')}, C6 = {quantity(n['c_dec'], 'F')} "
                f"(f_0 = {quantity(f0, 'Hz')}, `calc.lc.cutoff`) 로 그린 것(양축 로그). 안내선 = f_0; 점 = 클록 주파수에서의 이론 감쇠. "
                f"손실 없는 이상 부품이라 f_0 의 봉우리는 이론상 무한대이고 그림의 높이는 격자가 정한 것입니다(실제 봉우리는 권선 저항·ESR 이 정함). 동작점 기대값 `v_avcc` 는 DC(f = 0)의 |H| = 1 을 확인합니다. {THEORY_CURVE_NOTE}"
            )
            out.append(_curve_figure(
                "theory_avcc_lc", "AVCC LC 필터 |H(f)| (이상적 L·C)", caption, [("|H(f)|", xs, ys)], x_label="주파수 f (Hz)", y_label="|H(f)|", log_x=True, log_y=True,
                bands=[("x", f0, f0, f"f_0 = {quantity(f0, 'Hz')}")], markers=markers,
            ))
        c_x, c_s, c_l = n["c_xtal"], n["c_stray"], n["c_load"]
        if _known(c_x, c_s, c_l) and c_x > 0:
            xs = _lin_grid(c_x / 4.0, c_x * 2.0, 200)
            ys = [x / 2.0 + c_s for x in xs]
            caption = (
                f"'크리스탈과 부하 커패시턴스' 절의 식 C_L = C9·C10/(C9 + C10) + C_stray 를 C9 = C10 = C 로 두고 이 설계의 C_stray = {quantity(c_s, 'F')} 로 C 에 대해 그린 것. "
                f"점 = 설계점 C = {quantity(c_x, 'F')}, C_L = {quantity(c_l, 'F')} (`calc.crystal.load_capacitance`). 크리스탈의 규정 부하 용량은 이 IR 에 없어 비교선이 없습니다. {THEORY_CURVE_NOTE}"
            )
            out.append(_curve_figure(
                "theory_crystal_cl", "크리스탈 부하 용량 C_L 대 부하 커패시터", caption, [("C_L(C)", xs, ys)], x_label="부하 커패시터 C9 = C10 (F)", y_label="C_L (F)",
                markers=[(c_x, c_l, f"설계점 C = {quantity(c_x, 'F')}, C_L = {quantity(c_l, 'F')}")],
            ))
        return out

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        n = self._numbers(ir)
        v_in, v_out, i_load, p_reg = n["v_in"], n["v_out_reg"], n["i_load_budget"], n["p_reg"]
        v_max = self.V_IN_MAX
        drop_led = _sub(v_out, n["v_f_led"])
        tol_text = f"{number(_mul(100.0, n['tol_rel']), 3)} % 이하 (판정 허용치)" if n["tol_rel"] is not None else f"1 % ({NO_RECORD})"
        f_clk = n["f_clk"]

        def cap(role: str, why: str, value: float | None, v_rating: float | None, dielectric: str, subs: list[str]) -> PartNote:
            return PartNote(
                role=role, why=why,
                criteria=[f"정전용량 {quantity(value, 'F')}", f"정격 전압 ≥ 2 × {quantity(v_rating, 'V')} = {quantity(_mul(2.0, v_rating), 'V')}", dielectric],
                substitutes=subs,
            )

        dec_subs = [unverified(f"같은 값 {quantity(n['c_dec'], 'F')}의 0603 X7R MLCC (어느 제조사든)", "현재 풋프린트 그대로"),
                    unverified(f"같은 값의 0805 X7R MLCC", "풋프린트를 C_0805_2012Metric 으로 바꾸어야 함")]
        jtag = ("PF4..PF7(5..8번 핀)은 공장 출하 퓨즈(JTAGEN 프로그램됨)에서 JTAG 핀(TCK / TMS / TDO / TDI)이라, JTAGEN 을 해제하거나 MCUCSR 의 JTD 비트를 "
                "설정해야 일반 입출력이 됨(데이터시트 doc2467 기재 사항, 근거 없음). ")
        port_notes = {
            ref: _header_note(f"{ref}: 포트 {letter} 헤더", f"1..{bits} = P{letter}0..P{letter}{bits - 1}",
                              (jtag if letter == "F" else "") + "포트 핀이 MCU 에 직접 연결되므로 외부 회로는 ATmega128 의 핀 정격(데이터시트) 안에서만 연결해야 함.")
            for letter, ref, bits in PORTS
        }
        return {
            "U1": PartNote(
                role="U1: ATmega128 마이크로컨트롤러 (TQFP-64, 0.8 mm 피치)",
                why=(
                    f"사용자가 고른 개발 보드의 MCU. KiCad 심볼 `MCU_Microchip_ATmega:ATmega128-16A` 의 핀 이름으로 64핀을 모두 배선(VCC 2개, GND 3개 포함). "
                    f"'-16A' 는 16 MHz 등급이며 클록 f_clk = {quantity(f_clk, 'Hz')} 는 그 안(심볼 Description '16MHz'). SPICE 모델이 없어 시뮬레이션에서 제외(구조만 확인)."
                ),
                criteria=[
                    f"최대 클록 ≥ f_clk = {quantity(f_clk, 'Hz')} 이고 5 V 동작 등급 (ATmega128L 은 저전압 8 MHz 등급이라 불가)",
                    "TQFP-64 14 × 14 mm, 0.8 mm 피치 (풋프린트 Package_QFP:TQFP-64_14x14mm_P0.8mm)",
                    "핀 배치가 KiCad 심볼과 같을 것 (PE0 = PDI, PE1 = PDO 로 ISP 하는 ATmega128 계열)",
                ],
                substitutes=[
                    unverified("ATmega128A-AU", "핀 호환, 같은 TQFP-64 풋프린트; 퓨즈·전기적 특성 차이는 데이터시트에서 확인"),
                    unverified("ATmega64-16AU", "핀 호환이나 플래시 64 kB"),
                ],
            ),
            "U2": PartNote(
                role="U2: 5 V 선형 레귤레이터 L7805 (TO-220)",
                why=(
                    f"입력 {quantity(self.V_IN_MIN, 'V')}–{quantity(v_max, 'V')} 에서 +5V 레일을 만드는 가장 단순한 부품. 설계 부하 예산 {quantity(i_load, 'A')} 에서 손실 "
                    f"{quantity(p_reg, 'W')} (`calc.regulator.p_dissipation`). KiCad 심볼의 핀 이름 IN / GND / OUT 으로 배선. SPICE 모델이 없어 출력은 이상 전압원 V5V 로 대체."
                ),
                criteria=[
                    f"출력 5 V, 출력 전류 ≥ {quantity(i_load, 'A')} (여유 권장)",
                    f"최대 입력 전압 ≥ {quantity(v_max, 'V')} (템플릿 입력 상한)",
                    f"손실 {quantity(p_reg, 'W')} 에서 방열판 없이 허용될 것 (예산 {quantity(n['p_reg_max'], 'W')}, θ_JA 는 데이터시트에서 확인)",
                    "TO-220 핀 순서 IN-GND-OUT (1-2-3)",
                ],
                substitutes=[
                    unverified("LM7805 / MC7805 (TO-220)", "같은 핀 순서 IN-GND-OUT 의 78xx 계열"),
                    unverified("AMS1117-5.0 (SOT-223)", "풋프린트가 다르고 핀 순서도 GND-OUT-IN 이며 출력 커패시터 조건이 다름: 심볼·풋프린트·넷을 다시 확인"),
                ],
            ),
            "J1": PartNote(
                role="J1: DC 배럴 잭 (전원 입력)",
                why=(
                    "KiCad 의 일반 수평형 풋프린트 `Connector_BarrelJack:BarrelJack_Horizontal` 은 패드가 3개(1, 2, 3)라 스위치 접점이 있는 심볼 "
                    "`Connector:Barrel_Jack_Switch` 를 씀(2핀 `Barrel_Jack` 은 패드 3 이 IR 핀이 아니어서 PCB 컴파일러가 거부). "
                    f"핀 이름이 비어 있어 극성은 핀 번호 가정(선택값 `jack_pinout`): {self.JACK_PINOUT}."
                ),
                criteria=["중심 핀 + (센터 플러스) 어댑터용", f"정격 전압 ≥ {quantity(v_max, 'V')}, 정격 전류 ≥ {quantity(i_load, 'A')}", "풋프린트 BarrelJack_Horizontal 과 핀·패드 배치가 같을 것 (데이터시트 확인)"],
                substitutes=[unverified("CUI PJ-002A 계열 2.1 mm 수평형 잭"), unverified("2.5 mm 내경 잭", "어댑터 플러그 규격에 맞출 것")],
            ),
            "D1": PartNote(
                role="D1: 역접속 보호 직렬 정류 다이오드 (1N4007 급)",
                why=f"잭 극성이 반대인 어댑터를 꽂아도 전류가 흐르지 않게 막음. 대가는 순방향 강하(수십 mA 에서 약 {quantity(self.V_D1_DISPLAY_V, 'V')}, 근거 없음)만큼 레귤레이터 입력 여유가 줄어드는 것. KiCad 심볼의 핀 이름 A / K 로 배선. 시뮬레이션에서 제외(입력부).",
                criteria=[f"역전압 정격 ≥ 2 × {quantity(v_max, 'V')} = {quantity(2.0 * v_max, 'V')}", f"순방향 전류 정격 ≥ {quantity(_mul(2.0, i_load), 'A')}", "DO-41 축형 (10.16 mm 피치 풋프린트)"],
                substitutes=[unverified("1N4001 … 1N4007 (DO-41)", "역전압 정격만 다름"), unverified("1N5819 쇼트키 (DO-41)", "순방향 강하가 낮아 여유가 늘지만 역전압 정격 40 V")],
            ),
            "C1": cap("C1: 입력 벌크 전해 커패시터 (VIN)", f"레귤레이터 입력 안정과 잭 배선의 서지 완화. 선택값 {quantity(n['c_in'], 'F')}. 극성은 핀 번호 가정(`cp_polarity`: 핀 1 = +). 입력부라 시뮬레이션에서 제외.",
                      n["c_in"], v_max, "알루미늄 전해, 5 mm 원형 2.5 mm 피치", [
                          unverified(f"{quantity(n['c_in'], 'F')} / 50 V 전해 (5 mm)"),
                          unverified(f"{quantity(n['c_in'], 'F')} / 정격 {quantity(_mul(2.0, v_max), 'V')} 이상 MLCC (X7R)",
                                     "풋프린트를 바꾸어야 함; MLCC 는 DC 바이어스에서 용량이 줄어드므로 데이터시트로 확인. 탄탈은 핫플러그되는 잭 바로 뒤의 돌입 서지에 약해 권하지 않음"),
                      ]),
            "C2": cap("C2: 레귤레이터 출력 벌크 전해 커패시터 (+5V)", f"부하 과도 응답과 레귤레이터 안정. 선택값 {quantity(n['c_out'], 'F')}. 극성은 `cp_polarity` (핀 1 = +).",
                      n["c_out"], v_out, "알루미늄 전해, 5 mm 원형 2.5 mm 피치", [unverified(f"{quantity(n['c_out'], 'F')} / 16 V 이상 전해 (5 mm)"), unverified(f"{quantity(n['c_out'], 'F')} MLCC", "풋프린트를 바꾸어야 함")]),
            "C3": cap("C3: 레귤레이터 출력 세라믹 커패시터", f"고주파 바이패스. 선택값 {quantity(n['c_dec'], 'F')}.", n["c_dec"], v_out, "X7R 이상", dec_subs),
            "C4": cap("C4: U1 첫 번째 VCC 핀 바이패스", f"VCC 핀마다 {quantity(n['c_dec'], 'F')} 하나: MCU 의 스위칭 전류를 핀 가까이에서 공급. 배치 시 해당 VCC 핀에 붙여야 효과가 있음(회로도는 연결만 보장).", n["c_dec"], v_out, "X7R 이상", dec_subs),
            "C5": cap("C5: U1 두 번째 VCC 핀 바이패스", f"C4 와 같음 (다른 VCC 핀).", n["c_dec"], v_out, "X7R 이상", dec_subs),
            "C6": cap("C6: AVCC 필터 커패시터", f"L1 과 함께 f_0 = {quantity(n['f_avcc'], 'Hz')} 의 LC 저역통과.", n["c_dec"], v_out, "X7R 이상", dec_subs),
            "C7": cap("C7: AREF 바이패스", "ADC 기준 전압 핀의 잡음 억제 (외부 기준을 쓰지 않을 때). 시뮬레이션에서 제외(AREF 는 U1 에만 닿음).", n["c_dec"], v_out, "X7R 이상", dec_subs),
            "C8": cap("C8: ~RESET 지연 커패시터", f"R2 와 함께 τ = {quantity(n['tau_reset'], 's')} 의 전원 리셋 지연.", n["c_reset"], v_out, "X7R 이상 (값 오차가 τ 오차가 됨)", dec_subs),
            "C9": cap("C9: XTAL1 부하 커패시터", f"C10 과 함께 C_L = {quantity(n['c_load'], 'F')} (`calc.crystal.load_capacitance`). 시뮬레이션에서 제외(구조만 확인).", n["c_xtal"], v_out,
                      "C0G/NP0 (용량이 온도·전압에 따라 변하지 않아야 발진 주파수가 안정)", [unverified(f"{quantity(n['c_xtal'], 'F')} 0603 C0G MLCC"), unverified("크리스탈 데이터시트의 C_L 에 맞춘 다른 값", "C_L = C/2 + C_stray 로 다시 계산")]),
            "C10": cap("C10: XTAL2 부하 커패시터", "C9 와 같음 (XTAL2 쪽).", n["c_xtal"], v_out, "C0G/NP0", [unverified(f"{quantity(n['c_xtal'], 'F')} 0603 C0G MLCC")]),
            "D2": PartNote(
                role="D2: 전원 표시 LED (시뮬레이션에서는 V_f 의 이상 정전압원 VLED 로 대체)",
                why=f"+5V 가 살아 있음을 보여 줌. 선택값 V_f = {quantity(n['v_f_led'], 'V')}, I = {quantity(n['i_led'], 'A')}. 핀 이름 A / K 로 극성을 잡음. 0805 SMD.",
                criteria=[f"V_f(@ {quantity(n['i_led'], 'A')}) ≈ {quantity(n['v_f_led'], 'V')} (다르면 R1 을 다시 계산)", f"I_f(max) ≥ {quantity(n['i_led'], 'A')} 에 여유", "0805 풋프린트, 극성 표시 확인"],
                substitutes=[unverified("같은 V_f 급의 0805 표시 LED (어느 제조사든)", "V_f 를 데이터시트에서 확인")],
            ),
            "R1": PartNote(
                role="R1: 전원 LED 직렬 저항",
                why=f"R = (V_out − V_f)/I = {quantity(n['r_led'], 'ohm')} (`calc.led.R`, 계산값 그대로). 0805 SMD.",
                criteria=_resistor_criteria(n["r_led"], _mul(drop_led, n["i_led_design"]), tol_text, chosen=_is_choice(ir, "r_led"), spelled=_spelled(ir, "R1")),
                substitutes=_resistor_substitutes(n["r_led"], "Resistor_SMD"),
            ),
            "R2": PartNote(
                role="R2: ~RESET 풀업 저항",
                why=f"선택값 {quantity(n['r_reset'], 'ohm')}: C8 과 함께 τ = {quantity(n['tau_reset'], 's')}. SW1 을 누르면 V_cc/R2 가 흐름.",
                criteria=_resistor_criteria(n["r_reset"], _div(_mul(v_out, v_out), n["r_reset"]), "5 % 이하 (τ 오차)", chosen=_is_choice(ir, "r_reset"), spelled=_spelled(ir, "R2")),
                substitutes=_resistor_substitutes(n["r_reset"], "Resistor_SMD"),
            ),
            "R3": PartNote(
                role="R3: ~PEN 풀업 저항",
                why=f"선택값 {quantity(n['r_pen'], 'ohm')}: ~PEN 을 HIGH 로 유지 (전류는 핀 누설분뿐).",
                criteria=_resistor_criteria(n["r_pen"], _div(_mul(v_out, v_out), n["r_pen"]), "5 % 이하", chosen=_is_choice(ir, "r_pen"), spelled=_spelled(ir, "R3")),
                substitutes=_resistor_substitutes(n["r_pen"], "Resistor_SMD"),
            ),
            "L1": PartNote(
                role="L1: AVCC 필터 인덕터",
                why=f"선택값 {quantity(n['l_avcc'], 'H')}: C6 과 함께 f_0 = {quantity(n['f_avcc'], 'Hz')} (`calc.lc.cutoff`). 0805 SMD.",
                criteria=[f"인덕턴스 {quantity(n['l_avcc'], 'H')}", "정격 전류 ≥ AVCC 소비 전류 (데이터시트; 수 mA 급)", "DC 저항이 작을 것 (AVCC 와 VCC 의 차이는 데이터시트가 정한 범위 안이어야 함)"],
                substitutes=[unverified(f"{quantity(n['l_avcc'], 'H')} 0805 칩 인덕터"), unverified("페라이트 비드 (0805)", "필터 특성이 다름: f_0 식이 맞지 않음")],
            ),
            "SW1": PartNote(
                role="SW1: 리셋 푸시 버튼",
                why="누르면 ~RESET 을 GND 로 당김. 6 mm 택트 스위치(THT). 시뮬레이션에서 제외(열린 상태로 리셋 지연을 시뮬레이션).",
                criteria=["순간 접점(a 접점)", "6 mm 택트 스위치 4핀 풋프린트 (같은 번호의 패드 두 쌍이 내부 연결)"],
                substitutes=[unverified("임의의 6 × 6 mm THT 택트 스위치")],
            ),
            "Y1": PartNote(
                role=f"Y1: {quantity(f_clk, 'Hz')} 크리스탈 (HC-49/4H)",
                why=f"사용자 요구 클록 f_clk = {quantity(f_clk, 'Hz')}. 부하 커패시터 C9 = C10 = {quantity(n['c_xtal'], 'F')} 와 표유 용량으로 C_L = {quantity(n['c_load'], 'F')}.",
                criteria=[f"공칭 주파수 {quantity(f_clk, 'Hz')} (기본파)", f"규정 부하 용량 ≈ {quantity(n['c_load'], 'F')} (다르면 C9 / C10 을 다시 계산; 제약 조건 `c.atmega.crystal_load`)", "HC-49 THT 2핀"],
                substitutes=[unverified(f"{quantity(f_clk, 'Hz')} HC-49S (낮은 높이) 20 pF 부하", "C_L 이 다르므로 C9 / C10 을 다시 계산"), unverified(f"{quantity(f_clk, 'Hz')} SMD 크리스탈", "풋프린트를 바꾸어야 함")],
            ),
            "J2": PartNote(
                role="J2: ISP 헤더 (AVR 6핀 2x3)",
                why="1 = MISO(PDO, PE1), 2 = VCC, 3 = SCK(PB1), 4 = MOSI(PDI, PE0), 5 = ~RESET, 6 = GND: 표준 AVR ISP 배치. ATmega128 은 PE0/PE1 로 프로그래밍함(데이터시트, 근거 없음).",
                criteria=["2 × 3 2.54 mm 헤더, 핀 1 표시가 프로그래머 케이블과 맞을 것", "UART0 헤더 J3 와 PE0/PE1 을 공유 (제약 조건 `c.atmega.isp_uart_shared`)"],
                substitutes=[unverified("2 × 3 2.54 mm 박스 헤더", "풋프린트를 바꾸어야 함")],
            ),
            "J3": _header_note("J3: UART0 헤더", "1 = GND, 2 = +5V, 3 = RXD0 (PE0), 4 = TXD0 (PE1)", "RXD0 은 MCU 입력이므로 어댑터의 TX 에 연결; ISP 와 PE0/PE1 공유."),
            **port_notes,
            "J11": _header_note("J11: +5V / GND 전원 출력 헤더", "1 = +5V, 2 = GND", f"여기서 끌어 쓰는 전류도 설계 부하 예산 {quantity(i_load, 'A')} 안이어야 함."),
        }


__all__ = ["DC_JACK", "ISP_PINS", "MCU", "PORTS", "REGULATOR", "UART_PINS", "Atmega128DevboardTemplate", "mcu_pin_names", "port_pin_names"]
