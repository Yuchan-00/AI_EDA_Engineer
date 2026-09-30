"""The parts of the KR 447 MHz walkie-talkie family: which library symbol / footprint each role uses and how its pins are found.

Invariant (CLAUDE.md #2): a part is instantiated only from the KiCad library
on disk (:func:`~ai_eda.design.library_parts.library_component`), and every
pin a block wires is found **by its library name**. A name the library does
not spell exactly - or spells on a different number of pins than this table
expects - refuses the part (:class:`~ai_eda.design.library_parts.TemplateRefusal`),
naming the part, the name and the pins the library has; nothing is guessed.
Two kinds of pins have no name to look up, and each is handled by a rule
stated in the table, never silently:

* an **unnamed pin** (the op-amp output of ``MCP6001-OT`` pin 1, the
  ``LMV331`` open-collector output pin 4, the ``PHA-1`` RF input / output) is
  identified by its *library electrical type*, which must be unique among the
  symbol's unnamed pins (or, for the ``74LVC1G08``'s two unnamed inputs, the
  whole group, whose members are interchangeable because an AND gate is
  commutative); the :class:`PlacedPart` carries a note saying so;
* a **symmetric two-terminal part** (R, C, L, fuse, crystal, trimmer,
  ferrite, push button) has two interchangeable pins taken in library order,
  and the one polarised unnamed part (``Device:C_Polarized``) is wired by pin
  number as an *assumption* the template must show as a choice
  (``cp_polarity``: pin 1 = +, as in the ``atmega128_devboard`` template).

Every pin of the symbol must be claimed by exactly one row (library pins the
table does not name refuse the part: a changed library is a refusal, not a
dangling pin); ``no_connect`` pins may stay unclaimed. The same rules the two
compilers apply are checked here first, with their reasons, so a table row
that cannot compile refuses at build time instead of at compile time:

* a multi-unit symbol (``symbol.units != [1]``) - both compilers refuse it
  (:func:`ai_eda.compilers.pins.load_verified_symbol`): the dual op-amp
  ``MCP6002-xSN`` (units [1, 2, 3]) is therefore **not** in the table; every
  op-amp is an ``MCP6001-OT`` single (kr447 design §2.1, critic2);
* a repeated pin number, a symbol pin without a pad, or an electrical pad
  that is no symbol pin (the PCB compiler's pin/pad rule): the plain
  ``Device:R_Potentiometer`` on the RK09K footprint (pad ``MP`` has no pin)
  and ``RF_Switch:SKY13380-350LF`` (exposed pad 17 has no pin) are kept in
  :data:`REFUSED_PARTS` with their reasons and refused by :func:`instantiate`;
* **stacked pins** - library pins at one symbol point, which KiCad connects -
  must share one net (the schematic compiler refuses stacked pins in two
  nets). A row names the functions whose pins the library stacks
  (``stacked``); the stacks found on disk must be exactly those, each inside
  one function, so the wiring (:meth:`PlacedPart.at`) puts them in one net:
  the four ``MAX9814`` GND pins (4 / 7 / 11 / 15), the three ``ADEX-10`` GND
  pins (1 / 4 / 5), the two ``KT2520K-T`` GND pins (1 / 3).

A row may also name phrases its library ``Description`` must contain
(``facts``): parts chosen by a rating the library states - ``LP38693DT-5.0``
"500-mA", ``AO3401A`` "-4.0A Id", ``KT2520K-T`` "10-60MHz" - refuse when the
library no longer says it. Those library texts are descriptions, not grounded
datasheet facts: every rating, dropout, R_DS(on), theta_JA and RF parameter a
row mentions is ``[UNVERIFIED: <datasheet>]`` until the datasheet-facts path
grounds it.

Measured on the packed KiCad 10.0.6 libraries (symbols and footprints from the
``10.0.6`` tags) on 2026-09-29: every :data:`PARTS` row instantiates and a board
holding one of each compiles through the real schematic and PCB compilers
(``tests/test_rf_design_lib.py``, skipped without the libraries); the three
:data:`REFUSED_PARTS` rows refuse with the compilers' reasons. Re-measured
on 2026-09-30 after the microphone row became the through-hole
``Sensor_Audio:POM-2244P-C3310-2-R`` (decision 1A; it replaced the SMT
``CUI_CMC-4013-SMT``, whose signal pad 2 lies inside its ring-shaped custom
pad 1, so no track reaches it on F.Cu without a via in the pad): its pads
``1`` / ``2`` are the symbol's pins ``-`` / ``+``, and which terminal is the
case is UNVERIFIED (the row says so).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_eda.ir import Component, LibraryRef, PinElectricalType, Provenance
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError, SymbolDef

from ai_eda.design.library_parts import TemplateRefusal, library_component


def _natural(s: str) -> tuple:
    return tuple((0, int(t)) if t.isdigit() else (1, t) for t in re.findall(r"\d+|\D+", s))


@dataclass(frozen=True)
class PinSpec:
    """How one function of a part is found among the library pins (exactly one of ``name`` / ``unnamed_type`` / ``number``).

    * ``name`` - the exact library pin name; exactly ``count`` pins must carry
      it (the ``SA605D``'s two ``LIMITER_DECOUPL`` pins, the ``MMZ09332BT1``'s
      three ``VCC2/RFOUT`` pins).
    * ``unnamed_type`` - the library electrical type of an *unnamed* pin; the
      unnamed pins of that type must be exactly ``count`` (1: the pin is
      identified by that unique type; more: an interchangeable group, and
      ``interchangeable`` says why).
    * ``number`` - a pin number, allowed only for a pin without a name of its
      own (empty, or equal to its number): a symmetric two-terminal part, or
      a pin-number ``assumption`` (the choice key that makes it the user's).
    """

    function: str
    name: str | None = None
    count: int = 1
    unnamed_type: str | None = None
    number: str | None = None
    assumption: str | None = None
    interchangeable: str | None = None

    def __post_init__(self) -> None:
        modes = [m for m in (self.name, self.unnamed_type, self.number) if m is not None]
        if len(modes) != 1:
            raise ValueError(f"pin spec {self.function!r}: give exactly one of name / unnamed_type / number")
        if self.count < 1 or (self.number is not None and self.count != 1):
            raise ValueError(f"pin spec {self.function!r}: count must be >= 1 (1 for a pin number)")
        if self.unnamed_type is not None and self.count > 1 and not self.interchangeable:
            raise ValueError(f"pin spec {self.function!r}: a group of {self.count} unnamed pins needs the reason they are interchangeable")


@dataclass(frozen=True)
class PartDef:
    """One row of the parts table: a role, its library symbol / footprint, its pins by function, and what is unverified about it."""

    key: str
    role: str
    symbol: tuple[str, str]
    footprint: tuple[str, str]
    pins: tuple[PinSpec, ...]
    #: how the part appears in the design deck and the fixtures (text for the table; the binding is the block's)
    spice: str
    #: functions whose library pins are stacked at one symbol point (they must share one net)
    stacked: tuple[str, ...] = ()
    #: phrases the library ``Description`` must contain (the part was chosen by a rating the library states)
    facts: tuple[str, ...] = ()
    #: unverified facts that matter for this role (ratings, pinout, RF parameters), each ending in ``[UNVERIFIED: ...]``
    unverified: tuple[str, ...] = ()
    #: "primary" (used by the design), "substitute" (named in part notes only), "alternate" (a library alternate that also passes the rules)
    status: str = "primary"
    #: a symmetric two-terminal part: its two pins are interchangeable
    symmetric: bool = False

    @property
    def lib_id(self) -> str:
        return f"{self.symbol[0]}:{self.symbol[1]}"

    @property
    def footprint_id(self) -> str:
        return f"{self.footprint[0]}:{self.footprint[1]}"


def _named(*names: str) -> tuple[PinSpec, ...]:
    return tuple(PinSpec(n, name=n) for n in names)


def _two() -> tuple[PinSpec, ...]:
    """The two pins of a symmetric two-terminal part, functions ``1`` / ``2`` (library order)."""
    return (PinSpec("1", number="1"), PinSpec("2", number="2"))


#: why a symmetric part's two pins are taken in library order
SYMMETRIC_NOTE = "a symmetric two-terminal part: its two pins are interchangeable, so they are taken in library order"
#: the pin-number assumption of the polarised capacitor (unnamed pins): shown as the choice ``cp_polarity``
CP_POLARITY = "cp_polarity"
_AND_COMMUTATIVE = "the two unnamed inputs of an AND gate are interchangeable (A AND B = B AND A)"

_SOT23 = ("Package_TO_SOT_SMD", "SOT-23")
_SOT23_5 = ("Package_TO_SOT_SMD", "SOT-23-5")
_SOD323 = ("Diode_SMD", "D_SOD-323")
_R0402, _R0603 = ("Resistor_SMD", "R_0402_1005Metric"), ("Resistor_SMD", "R_0603_1608Metric")
_C0402, _C0603 = ("Capacitor_SMD", "C_0402_1005Metric"), ("Capacitor_SMD", "C_0603_1608Metric")
_UFL = ("Connector_Coaxial", "U.FL_Hirose_U.FL-R-SMT-1_Vertical")

_ROWS: tuple[PartDef, ...] = (
    # ---------------------------------------------------------------- power, PTT, audio (stage 1)
    PartDef("conn_2_jst", "2-pin JST PH connector: the 2S pack (off-board, protected, charged externally) or the speaker",
            ("Connector_Generic", "Conn_01x02"), ("Connector_JST", "JST_PH_B2B-PH-K_1x02_P2.00mm_Vertical"),
            _named("Pin_1", "Pin_2"), "excluded (connector); the pack voltage is a DC stimulus",
            unverified=("the pack is a protected 2S Li-ion pack with its own cut-off below the confirmed 6.4 V choice [UNVERIFIED: pack datasheet]",)),
    PartDef("fuse", "pack fuse", ("Device", "Fuse"), ("Fuse", "Fuse_1206_3216Metric"), _two(),
            "R at model.fuse.r (a model choice)", symmetric=True,
            unverified=("fuse rating and cold resistance [UNVERIFIED: fuse datasheet]",)),
    PartDef("sw_spdt", "power switch SW101: switches only the gate of the main P-FET (pole B = pin 2; throw A = pin 1 to GND is on, throw C = pin 3 open is off)",
            ("Switch", "SW_SPDT"), ("Button_Switch_SMD", "SW_SPDT_PCM12"), _named("A", "B", "C"),
            "excluded (no switch model); the ON position is a DC 0 V stimulus on the gate net",
            unverified=("PCM12 contact rating about 0.3 A [UNVERIFIED: C&K PCM12 datasheet] - it carries only the gate current (about 84 uA) here",)),
    PartDef("pfet", "P-FET high-side switch: main switch Q105 (gate switched by SW101), RX / TX rail switches Q101 / Q102, PA supply switch Q201",
            ("Transistor_FET", "AO3401A"), _SOT23, _named("G", "S", "D"),
            "X card of model.pmos (a level-1 PMOS wrapped as a 3-terminal subcircuit, bulk tied to source)", facts=("-4.0A Id",),
            unverified=("-4.0 A I_D, R_DS(on) about 60 mohm and the V_GS limit against 8.4 V applied [UNVERIFIED: AOS AO3401A datasheet]",)),
    PartDef("npn_small", "small-signal NPN: PTT inverters Q103 / Q104, PA-switch driver Q202, POWER_DOWN inverter Q203, the emitter follower Q801 between the two PM tanks",
            ("Transistor_BJT", "MMBT3904"), _SOT23, _named("B", "E", "C"), "Q card of model.npn (generic, not a vendor model)",
            facts=("NPN",), unverified=("MMBT3904 ratings and hFE [UNVERIFIED: MMBT3904 datasheet]",)),
    PartDef("ldo_rx5v", "RX_5V regulator U101 (decision 3A): LP38693DT-5.0 on TO-252-2",
            ("Regulator_Linear", "LP38693DT-5.0"), ("Package_TO_SOT_SMD", "TO-252-2"), _named("OUT", "GND", "IN"),
            "excluded (no model); RX_5V is an ideal source gated like its rail", facts=("500-mA",),
            unverified=("dropout about 0.25-0.45 V and theta_JA [UNVERIFIED: TI LP38693 datasheet]",
                        "pin order 1 OUT / 2 GND / 3 IN is the library's; the TO-252-2 alternates BD50FC0FP / LF50_TO252 are the other way round - "
                        "check TI's pin table before the BOM [UNVERIFIED: TI LP38693 datasheet]")),
    PartDef("ldo_tx5v", "TX_5V regulator U102 (PA supply, about 1 W loss in TX)",
            ("Regulator_Linear", "LM1117DT-5.0"), ("Package_TO_SOT_SMD", "TO-252-3_TabPin2"), _named("GND", "VO", "VI"),
            "excluded (no model)", facts=("800mA",),
            unverified=("1.2 V dropout choice and theta_JA [UNVERIFIED: TI LM1117 datasheet]",)),
    PartDef("ldo_3v3", "RX_3V3 / TX_3V3 low-noise regulators U103 / U104",
            ("Regulator_Linear", "LP5907MFX-3.3"), _SOT23_5, (*_named("IN", "GND", "EN", "OUT"), PinSpec("NC", name="NC")),
            "excluded (no model); the 3.3 V rails are ideal sources", facts=("250-mA",),
            unverified=("noise and dropout [UNVERIFIED: TI LP5907 datasheet]",)),
    PartDef("led", "TX indicator LED", ("Device", "LED"), ("LED_SMD", "LED_0603_1608Metric"), _named("K", "A"),
            "ideal forward-drop model (the led template's precedent)"),
    PartDef("sw_push", "PTT push button", ("Switch", "SW_Push"), ("Button_Switch_SMD", "SW_SPST_TL3342"),
            (PinSpec("1", name="1"), PinSpec("2", name="2")), "excluded (no switch model); PTT_N is a PWL stimulus", symmetric=True),
    PartDef("diode_sw", "switching diode (settle-delay discharge, low-pack clamp, limiter clipper)",
            ("Diode", "1N4148WS"), _SOD323, _named("K", "A"), "D card of model.diode (ngspice default diode, not the 1N4148 model)",
            unverified=("1N4148WS forward drop and capacitance [UNVERIFIED: 1N4148WS datasheet]",)),
    PartDef("comparator", "comparator: settle delay U201, low-pack TX inhibit U204 (enforces power.pack_cutoff_v), squelch U403; the open-collector output needs its pull-up",
            ("Comparator", "LMV331"), _SOT23_5, (PinSpec("+", name="+"), PinSpec("V-", name="V-"), PinSpec("-", name="-"),
                                              PinSpec("OUT", unnamed_type="open_collector"), PinSpec("V+", name="V+")),
            "excluded (no model); its input networks are simulated",
            unverified=("absolute maximum supply about 5.5 V: supply it from a 3.3 V rail, never V_SYS [UNVERIFIED: TI LMV331 datasheet]",)),
    PartDef("and_gate", "interlock AND gate U202 (PTT_ACTIVE AND DLY_OK -> PA_ON)",
            ("74xGxx", "74LVC1G08"), _SOT23_5, (PinSpec("IN", unnamed_type="input", count=2, interchangeable=_AND_COMMUTATIVE),
                                             PinSpec("GND", name="GND"), PinSpec("OUT", unnamed_type="output"), PinSpec("VCC", name="VCC")),
            "excluded (no model)",
            unverified=("absolute maximum supply about 5.5 V: supply it from TX_3V3 [UNVERIFIED: 74LVC1G08 datasheet]",)),
    PartDef("counter_4060", "optional transmit time-out timer (only when tx_timeout is stated)",
            ("4xxx", "4060"), ("Package_SO", "SOIC-16_3.9x9.9mm_P1.27mm"),
            _named("Q12", "Q13", "Q14", "Q6", "Q5", "Q7", "Q4", "VSS", "~{Φ0}", "Φ0", "~{Φ1}", "CLR", "Q9", "Q8", "Q10", "VDD"),
            "excluded (no model); the time-out is calc.rf.tot.* from the RC formula",
            unverified=("the 4060 RC oscillator formula constant [UNVERIFIED: 4060 datasheet]",)),
    PartDef("mic", "electret microphone (through-hole capsule: two THT pads 1 / 2 at 1.9 mm, drill 0.65 mm)", ("Device", "Microphone_Condenser"),
            ("Sensor_Audio", "POM-2244P-C3310-2-R"), _named("-", "+"),
            "excluded; MICOUT is a SINE stimulus",
            unverified=("which terminal is the case / negative one: the library symbol names pin 1 '-' and pin 2 '+', the footprint's pads 1 / 2 "
                        "follow those numbers and its silkscreen '+' mark lies on the pad-2 side, but PUI Audio's pinout is not checked "
                        "[UNVERIFIED: PUI Audio POM-2244P-C3310-2-R datasheet]",
                        "operating current and sensitivity at the MAX9814's MICBIAS through the 2.2 kohm bias resistor "
                        "[UNVERIFIED: PUI Audio POM-2244P-C3310-2-R datasheet]")),
    PartDef("mic_amp", "microphone preamplifier with AGC", ("Amplifier_Audio", "MAX9814"),
            ("Package_DFN_QFN", "DFN-14-1EP_3x3mm_P0.4mm_EP1.78x2.35mm"),
            (*_named("CT", "~{SHDN}", "CG", "VDD", "MICOUT", "MICIN", "A/R", "GAIN", "BIAS", "MICBIAS", "TH"), PinSpec("GND", name="GND", count=4)),
            "excluded (no model); MICOUT is a SINE stimulus", stacked=("GND",),
            unverified=("GAIN / A/R / TH strapping [UNVERIFIED: Maxim MAX9814 datasheet]",)),
    PartDef("opamp", ("single op-amp of every audio stage (each MCP6002-xSN dual of the design becomes two of these: U302 / U305, U303 / U306 the 4th-order splatter "
             "sections, U304 / U307 the integrator and PM_DRIVE buffer, U401 / U404 - the compilers refuse the dual as a multi-unit symbol)"),
            ("Amplifier_Operational", "MCP6001-OT"), _SOT23_5,
            (PinSpec("OUT", unnamed_type="output"), PinSpec("V-", name="V-"), PinSpec("+", name="+"), PinSpec("-", name="-"), PinSpec("V+", name="V+")),
            "X card of model.opamp (generic single-pole macro; its supply pins are ignored by the macro)", facts=("1MHz",),
            unverified=("GBW 1 MHz and rail-to-rail output [UNVERIFIED: Microchip MCP6001 datasheet]",)),
    PartDef("trimpot", "deviation trim (lab alignment)", ("Device", "R_Potentiometer_Trim"),
            ("Potentiometer_SMD", "Potentiometer_Bourns_3314J_Vertical"), (PinSpec("END1", name="1"), PinSpec("WIPER", name="2"), PinSpec("END3", name="3")),
            "X card of the potentiometer model (two resistors at the confirmed wiper position)"),
    PartDef("pot", "panel potentiometer (volume, squelch) with its mounting pin",
            ("Device", "R_Potentiometer_MountingPin"), ("Potentiometer_THT", "Potentiometer_Alps_RK09K_Single_Vertical"),
            (PinSpec("END1", name="1"), PinSpec("WIPER", name="2"), PinSpec("END3", name="3"), PinSpec("MOUNT", name="MountPin")),
            "X card of the potentiometer model; the mounting pin is ignored by the model"),
    PartDef("audio_amp", "speaker amplifier", ("Amplifier_Audio", "LM386"), ("Package_SO", "SOIC-8_3.9x4.9mm_P1.27mm"),
            (PinSpec("GAIN", name="GAIN", count=2), *_named("-", "+", "GND", "V+", "BYPASS"), PinSpec("OUT", unnamed_type="output")),
            "excluded (no model)", unverified=("LM386 output power at the V_RX rail [UNVERIFIED: TI LM386 datasheet]",)),
    PartDef("cap_polarized", "electrolytic capacitor (speaker coupling, bulk)", ("Device", "C_Polarized"), ("Capacitor_SMD", "CP_Elec_5x5.4"),
            (PinSpec("+", number="1", assumption=CP_POLARITY), PinSpec("-", number="2", assumption=CP_POLARITY)),
            "C at its value", unverified=("the + terminal is pin 1 by pin number: the library pins carry no names (choice cp_polarity) [UNVERIFIED: footprint polarity mark]",)),
    PartDef("header_3", "3-pin bench interface header", ("Connector_Generic", "Conn_01x03"),
            ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical"), _named("Pin_1", "Pin_2", "Pin_3"), "excluded (connector)"),
    # ---------------------------------------------------------------- passives
    PartDef("res_0402", "resistor (RF, 0402)", ("Device", "R"), _R0402, _two(), "R at its value", symmetric=True),
    PartDef("res_0603", "resistor (0603)", ("Device", "R"), _R0603, _two(), "R at its value", symmetric=True),
    PartDef("cap_0402", "capacitor (RF, 0402)", ("Device", "C"), _C0402, _two(), "C at its value", symmetric=True),
    PartDef("cap_0603", "capacitor (0603)", ("Device", "C"), _C0603, _two(), "C at its value", symmetric=True),
    PartDef("ind_0603", "inductor (0603)", ("Device", "L"), ("Inductor_SMD", "L_0603_1608Metric"), _two(),
            "L at its value; in a fixture with a series R from model.l_q.*", symmetric=True,
            unverified=("inductor Q [UNVERIFIED: Coilcraft / Murata data]",)),
    PartDef("ind_1210", "inductor (450 kHz IF2 resonators, quadrature coil)", ("Device", "L"), ("Inductor_SMD", "L_1210_3225Metric"), _two(),
            "L at its value; in a fixture with a series R from model.l_q.if2", symmetric=True,
            unverified=("inductor Q at 450 kHz [UNVERIFIED: inductor datasheet]",)),
    PartDef("ind_0604hq", "high-Q UHF inductor (BPFs, LPF, T/R switch, matches)", ("Device", "L"), ("Inductor_SMD", "L_Coilcraft_0604HQ_1610Metric"), _two(),
            "L at its value; in a fixture with a series R from model.l_q.uhf", symmetric=True,
            unverified=("0604HQ Q at 447 MHz [UNVERIFIED: Coilcraft 0604HQ data]",)),
    PartDef("ferrite", "ferrite bead (rail decoupling)", ("Device", "FerriteBead_Small"), ("Inductor_SMD", "L_0603_1608Metric"), _two(),
            "excluded or a small R (block choice)", symmetric=True, unverified=("bead impedance [UNVERIFIED: bead datasheet]",)),
    PartDef("ctrim", "trimmer capacitor (tank / ladder / quadrature alignment)", ("Device", "C_Trim"), ("Capacitor_SMD", "C_Trimmer_Murata_TZB4-A"), _two(),
            "C at the confirmed mid position", symmetric=True,
            unverified=("C_min / C_max and which pad is the rotor (the library pins carry no names) [UNVERIFIED: Murata TZB4 datasheet]",)),
    PartDef("crystal", "crystal (21.4 MHz ladder, 20.950 MHz LO2)", ("Device", "Crystal"), ("Crystal", "Crystal_SMD_HC49-SD"),
            (PinSpec("1", name="1"), PinSpec("2", name="2")), "X card of model.xtal21 (motional R-L-C with C0) in the ladder fixture; excluded elsewhere",
            symmetric=True, unverified=("motional Lm / Cm / Rm / C0 of the ordered crystals (measure, G3UUR method) [UNVERIFIED: crystal datasheet]",)),
    PartDef("varactor", "varactors D801 / D802 of the two buffered phase-modulator tanks", ("Device", "D_Capacitance"), _SOD323, _named("K", "A"),
            "D card of model.varactor (CJO / VJ / M choices)", unverified=("varactor MPN and its C(V) [UNVERIFIED: varactor datasheet]",)),
    PartDef("pin_diode", "PIN diodes of the T/R switch: D1001 series (TX), D1002 shunt at the RX end (BAR64-03W class)", ("Device", "D"), _SOD323, _named("K", "A"),
            "fixture states: R at model.pin.r_on (TX) / C at model.pin.c_off (RX)",
            unverified=("R_on at 10 mA, C_off and power handling at +27 dBm [UNVERIFIED: BAR64-03W datasheet]",)),
    PartDef("testpoint", "alignment test point", ("Connector", "TestPoint"), ("TestPoint", "TestPoint_Pad_D1.0mm"), (PinSpec("1", name="1"),),
            "excluded (test point)"),
    # ---------------------------------------------------------------- RF
    PartDef("coax_ufl", "U.FL coaxial connector (bench ports; the conducted variant's antenna port)", ("Connector", "Conn_Coaxial"), _UFL,
            _named("In", "Ext"), "excluded; a fixture port of system_impedance"),
    PartDef("fm_if", "FM IF system U501: mixer, LO2 oscillator, IF amplifier, limiter, quadrature detector, RSSI, mute",
            ("RF_AM_FM", "SA605D"), ("Package_SO", "SO-20_12.8x7.5mm_P1.27mm"),
            (*_named("RF_IN", "RF_BYPASS", "OSC_OUT", "OSC_IN", "MUTE_INPUT", "VCC", "RSSI_OUT", "MUTED_AUD_OUTP", "UNMUTED_AUD_OUTP",
                     "QUADRATURE_IN", "LIMITER_OUT", "LIMITER_IN", "GND", "IF_AMP_OUT", "IF_AMP_IN", "MIXER_OUT"),
             PinSpec("LIMITER_DECOUPL", name="LIMITER_DECOUPL", count=2), PinSpec("IF_AMP_DECOUPL", name="IF_AMP_DECOUPL", count=2)),
            "excluded (no model); its ports are model.sa605.port_r in the fixtures",
            unverified=("VCC range (about 4.5 V minimum), port impedances and NF [UNVERIFIED: NXP SA605 datasheet]",)),
    PartDef("rf_npn", "wideband NPN: LNA Q601, IF1 post-amplifier Q602, the x3 / x2 / x2 double-tuned multiplier stages Q701-Q703 (LO) and Q802-Q804 (TX)", ("Transistor_BJT", "BFR92"),
            ("Package_TO_SOT_SMD", "SOT-323_SC-70"), _named("B", "E", "C"), "bias op on model.npn; excluded from the fixtures (port models model.bfr92.*)",
            facts=("5GHz",), unverified=("S-parameters, NF and class-C behaviour [UNVERIFIED: NXP BFR92AW datasheet]",)),
    PartDef("mixer", "first mixer U601 (passive double-balanced, +7 dBm LO)", ("RF_Mixer", "ADEX-10"),
            ("RF_Mini-Circuits", "Mini-Circuits_CD542_LandPatternPL-052"), (PinSpec("GND", name="GND", count=3), *_named("IF", "RF", "LO")),
            "excluded (no model)", stacked=("GND",), facts=("10 to 1000 MHz",),
            unverified=("conversion loss, 2x2 half-IF suppression and port impedances [UNVERIFIED: Mini-Circuits ADEX-10 datasheet]",)),
    PartDef("tcxo", "reference TCXO (custom frequencies, decision 1B): Y801 TX 37.296875 MHz = f_c / 12, Y701 RX 35.5135417 MHz = (f_c - 21.4 MHz) / 12", ("Oscillator", "KT2520K-T"),
            ("Oscillator", "Oscillator_SMD_Kyocera_2520-6Pin_2.5x2.0mm"),
            (PinSpec("GND", name="GND", count=2), PinSpec("OUT", name="OUT"), PinSpec("VCC", name="VCC"), PinSpec("NC", name="NC", count=2)),
            "excluded; a fixture port of model.tcxo.r_out", stacked=("GND",), facts=("10-60MHz",),
            unverified=("availability of the custom frequencies, stability (2.5 ppm choice) and start-up time [UNVERIFIED: Kyocera KT2520K datasheet]",)),
    PartDef("rf_amp", "RF gain block: LO buffer U701, TX driver U802", ("RF_Amplifier", "PHA-1"), ("Package_TO_SOT_SMD", "SOT-89-3"),
            (PinSpec("RF_IN", unnamed_type="input"), PinSpec("GND", name="GND"), PinSpec("RF_OUT", unnamed_type="output")),
            "excluded (no model)", facts=("50-6000MHz",),
            unverified=("gain, P1dB and supply current [UNVERIFIED: Mini-Circuits PHA-1 datasheet]",)),
    PartDef("pa", "power amplifier U901 (<= 0.5 W)", ("RF_Amplifier", "MMZ09332BT1"),
            ("Package_DFN_QFN", "QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm_ThermalVias"),
            (*_named("VBA1", "VBIAS", "RF_IN", "PDET", "POWER_DOWN", "VCC1", "VBA2", "GND"), PinSpec("VCC2/RFOUT", name="VCC2/RFOUT", count=3),
             PinSpec("NC", name="NC", count=2)),
            "excluded (no model)",
            unverified=("coverage of 447 MHz, 0.5 W at 5 V, gain and POWER_DOWN polarity (pins 7-9 are typed input in the library) [UNVERIFIED: NXP MMZ09332B datasheet]",)),
    PartDef("antenna", "ANT1: the integral straight lambda/4 wire (decision 5A), soldered to the feed pad", ("Device", "Antenna"),
            ("Connector_Wire", "SolderWire-0.5sqmm_1x01_D0.9mm_OD2.1mm"), _named("A"),
            "a fixture port of antenna_impedance when confirmed, else none",
            unverified=("feed impedance, efficiency and the velocity factor 0.95 choice [UNVERIFIED: VNA measurement]",)),
    PartDef("shield_102", "shield can SH601 over the LNA and BPFs", ("Device", "RFShield_OnePiece"),
            ("RF_Shielding", "Laird_Technologies_BMI-S-102_16.50x16.50mm"), _named("Shield"), "excluded"),
    PartDef("shield_103", "shield can SH701 over the LO chain", ("Device", "RFShield_OnePiece"),
            ("RF_Shielding", "Laird_Technologies_BMI-S-103_26.21x26.21mm"), _named("Shield"), "excluded"),
    PartDef("shield_105", "shield can SH801 over the modulator and the multiplier chain", ("Device", "RFShield_OnePiece"),
            ("RF_Shielding", "Laird_Technologies_BMI-S-105_38.10x25.40mm"), _named("Shield"), "excluded"),
    # ---------------------------------------------------------------- substitutes and library alternates (named, not used by default)
    PartDef("trsw_alt", "substitute T/R switch (GaAs SPDT)", ("RF_Switch", "AS179-92LF"), ("Package_TO_SOT_SMD", "SOT-363_SC-70-6"),
            _named("J1", "J2", "J3", "V1", "V2", "GND"), "excluded (no model)", status="substitute",
            unverified=("power handling at +27 dBm [UNVERIFIED: Skyworks AS179-92LF datasheet]",)),
    PartDef("mixer_alt", "substitute first mixer (active, low current)", ("RF_Mixer", "LT5560"),
            ("Package_DFN_QFN", "DFN-8-1EP_3x3mm_P0.5mm_EP1.66x2.38mm"), _named("LO-", "EN", "IN+", "IN-", "OUT-", "OUT+", "VCC", "LO+", "PGND"),
            "excluded (no model)", status="substitute", unverified=("LO drive and conversion gain [UNVERIFIED: ADI LT5560 datasheet]",)),
    PartDef("pa_alt", "substitute power amplifier", ("RF_Amplifier", "CMX901"),
            ("Package_DFN_QFN", "QFN-28-1EP_5x5mm_P0.5mm_EP3.35x3.35mm_ThermalVias"),
            (PinSpec("GND", name="GND", count=16), PinSpec("RFIN", name="RFIN", count=2), PinSpec("RFOUT", name="RFOUT", count=5),
             *_named("VDD1", "VGS1", "VA", "VGS3", "VDD2", "VGS2")),
            "excluded (no model)", stacked=("GND", "RFIN", "RFOUT"), status="substitute",
            unverified=("coverage of 447 MHz and 0.5 W [UNVERIFIED: CML CMX901 datasheet]",)),
    PartDef("lpf_alt", "substitute harmonic low-pass filter (490 MHz)", ("RF_Filter", "LFCN-490"), ("Filter", "Filter_Mini-Circuits_FV1206"),
            (PinSpec("IN", name="IN"), PinSpec("OUT", name="OUT"), PinSpec("GND", name="GND", count=2)),
            "excluded (no model)", stacked=("GND",), status="substitute",
            unverified=("insertion loss at 447 MHz and rejection at 2 f_c [UNVERIFIED: Mini-Circuits LFCN-490 datasheet]",)),
    PartDef("ldo_rx5v_bd50", "library alternate for RX_5V (pin order 1 VCC / 3 VO, the reverse of LP38693DT-5.0)",
            ("Regulator_Linear", "BD50FC0FP"), ("Package_TO_SOT_SMD", "TO-252-2"), _named("VCC", "GND", "VO"), "excluded (no model)",
            status="alternate", unverified=("1 A rating [UNVERIFIED: ROHM BD50FC0FP datasheet]",)),
    PartDef("ldo_rx5v_ifx", "library alternate for RX_5V", ("Regulator_Linear", "IFX27001TFV50"), ("Package_TO_SOT_SMD", "TO-252-3_TabPin2"),
            _named("GND", "Q", "I"), "excluded (no model)", status="alternate", unverified=("1 A rating [UNVERIFIED: Infineon IFX27001 datasheet]",)),
    PartDef("ldo_rx5v_lf50", "library alternate for RX_5V (pin order 1 VI / 3 VO, the reverse of LP38693DT-5.0)",
            ("Regulator_Linear", "LF50_TO252"), ("Package_TO_SOT_SMD", "TO-252-2"), _named("VI", "GND", "VO"), "excluded (no model)",
            status="alternate", unverified=("500 mA rating [UNVERIFIED: ST LF50 datasheet]",)),
)

#: every usable row, by key
PARTS: dict[str, PartDef] = {p.key: p for p in _ROWS}


@dataclass(frozen=True)
class RefusedPart:
    """A symbol / footprint pair the design considered and the compilers refuse, kept so nobody adds it back."""

    symbol: tuple[str, str]
    footprint: tuple[str, str]
    reason: str
    instead: str


#: (library, symbol name) -> why it is refused (kr447 design §2.6 COMPILE ERROR rows and the critic2 multi-unit finding)
REFUSED_PARTS: dict[tuple[str, str], RefusedPart] = {
    ("Amplifier_Operational", "MCP6002-xSN"): RefusedPart(
        ("Amplifier_Operational", "MCP6002-xSN"), ("Package_SO", "SOIC-8_3.9x4.9mm_P1.27mm"),
        "a multi-unit symbol (units [1, 2, 3]): both compilers refuse it (multi-unit symbols are not supported yet)",
        "two Amplifier_Operational:MCP6001-OT singles on SOT-23-5 (part 'opamp')"),
    ("Device", "R_Potentiometer"): RefusedPart(
        ("Device", "R_Potentiometer"), ("Potentiometer_THT", "Potentiometer_Alps_RK09K_Single_Vertical"),
        "footprint pad MP is no symbol pin: the PCB compiler refuses an electrical pad without a pin",
        "Device:R_Potentiometer_MountingPin (part 'pot')"),
    ("RF_Switch", "SKY13380-350LF"): RefusedPart(
        ("RF_Switch", "SKY13380-350LF"), ("Package_DFN_QFN", "QFN-16-1EP_3x3mm_P0.5mm_EP1.7x1.7mm"),
        "the footprint's exposed pad 17 has no symbol pin: the PCB compiler refuses it",
        "the PIN-diode T/R switch (parts 'pin_diode' + passives), or RF_Switch:AS179-92LF (part 'trsw_alt')"),
}


@dataclass
class PlacedPart:
    """A component instantiated from a :class:`PartDef`, with its pins resolved by function.

    ``functions`` maps each function of the row to the library pin numbers
    that carry it (natural order); ``stacks`` are the pin-number groups the
    library puts at one symbol point (each inside one function); ``notes``
    say how any pin without a usable name was identified.
    """

    part: PartDef
    component: Component
    functions: dict[str, tuple[str, ...]]
    stacks: tuple[tuple[str, ...], ...] = ()
    no_connect: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        return self.component.ref

    def pins(self, function: str) -> tuple[str, ...]:
        """Every pin number of ``function``; refuses a function the row does not have."""
        try:
            return self.functions[function]
        except KeyError:
            raise TemplateRefusal(f"{self.ref} ({self.part.lib_id}) has no function {function!r}; it has {sorted(self.functions)}") from None

    def pin(self, function: str) -> str:
        """The one pin number of ``function``; refuses a function carried by several pins (name which one with :meth:`pins`)."""
        numbers = self.pins(function)
        if len(numbers) != 1:
            raise TemplateRefusal(f"{self.ref} ({self.part.lib_id}) function {function!r} is carried by pins {list(numbers)}; pick one with pins()")
        return numbers[0]

    def at(self, function: str) -> list[tuple[str, str]]:
        """``(ref, pin)`` for every pin of ``function``: the net members that keep stacked pins in one net."""
        return [(self.ref, n) for n in self.pins(function)]


def symbol_stacks(symbol: SymbolDef) -> tuple[tuple[str, ...], ...]:
    """Pin-number groups of ``symbol`` that sit at one point (KiCad connects them), natural order, groups of two or more."""
    at: dict[tuple[float, float], list[str]] = {}
    for p in symbol.pins:
        if p.body_style not in (0, 1):
            continue
        group = at.setdefault((round(p.x, 6), round(p.y, 6)), [])
        if p.number not in group:
            group.append(p.number)
    return tuple(sorted((tuple(sorted(g, key=_natural)) for g in at.values() if len(g) > 1), key=lambda g: _natural(g[0])))


def _resolve(part: PartDef, component: Component) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], list[str]]:
    """``(functions, unclaimed no_connect pins, notes)``; refuses a missing or ambiguous name and any unclaimed pin."""
    pins = component.pins
    ref = component.ref
    listing = [(p.number, p.name) for p in pins]
    functions: dict[str, tuple[str, ...]] = {}
    claimed: dict[str, str] = {}
    notes: list[str] = []
    for spec in part.pins:
        if spec.function in functions:
            raise TemplateRefusal(f"parts table row {part.key!r} names function {spec.function!r} twice")
        if spec.name is not None:
            found = [p.number for p in pins if p.name == spec.name]
            if len(found) != spec.count:
                what = "no pin" if not found else f"{len(found)} pin(s) {sorted(found, key=_natural)}"
                raise TemplateRefusal(
                    f"{ref} ({part.lib_id}): the library has {what} named {spec.name!r}, the template expects {spec.count}; "
                    f"library pins (number, name): {listing} - the template will not guess the pinout"
                )
        elif spec.unnamed_type is not None:
            found = [p.number for p in pins if p.name == "" and PinElectricalType(p.electrical_type).value == spec.unnamed_type]
            if len(found) != spec.count:
                raise TemplateRefusal(
                    f"{ref} ({part.lib_id}): {spec.function} is an unnamed pin found by its electrical type {spec.unnamed_type!r}, but the library has "
                    f"{len(found)} unnamed pin(s) of that type (the template expects {spec.count}); library pins (number, name): {listing}"
                )
            how = f"identified as {spec.function} by its library electrical type {spec.unnamed_type!r}"
            if spec.count == 1:
                notes.append(f"{ref} pin {found[0]} has no name in the library; {how} (the only unnamed pin of that type)")
            else:
                notes.append(f"{ref} pins {sorted(found, key=_natural)} have no names in the library; {how}: {spec.interchangeable}")
        else:
            pin = component.pin(spec.number or "")
            if pin is None:
                raise TemplateRefusal(f"{ref} ({part.lib_id}) has no pin {spec.number!r}; library pins (number, name): {listing}")
            if pin.name not in ("", pin.number):
                raise TemplateRefusal(
                    f"{ref} ({part.lib_id}) pin {spec.number} is named {pin.name!r} in the library: a named pin is found by its name, never by number"
                )
            found = [pin.number]
            if spec.assumption:
                notes.append(f"{ref} pin {pin.number} is {spec.function} by pin number (the library pin has no name): an assumption shown as the choice {spec.assumption!r}")
            elif not part.symmetric:
                raise TemplateRefusal(f"parts table row {part.key!r}: a pin by number needs a symmetric part or an assumption")
            elif not any(SYMMETRIC_NOTE in n for n in notes):
                notes.append(f"{ref} ({part.lib_id}): {SYMMETRIC_NOTE}")
        for n in found:
            if n in claimed:
                raise TemplateRefusal(f"{ref} ({part.lib_id}) pin {n} is claimed by functions {claimed[n]!r} and {spec.function!r}")
            claimed[n] = spec.function
        functions[spec.function] = tuple(sorted(found, key=_natural))
    unclaimed = [p for p in pins if p.number not in claimed]
    missing = [(p.number, p.name) for p in unclaimed if p.electrical_type is not PinElectricalType.NO_CONNECT]
    if missing:
        raise TemplateRefusal(
            f"{ref} ({part.lib_id}): library pins {missing} are named by no row of the parts table (row {part.key!r}); "
            f"the library changed or the table is incomplete - no pin is left unwired silently"
        )
    no_connect = tuple(sorted((p.number for p in unclaimed), key=_natural))
    return functions, no_connect, notes


def instantiate(
    library: KicadLibrary,
    key: str,
    ref: str,
    value: str,
    description: str,
    provenance: Provenance,
    serves: list[str] | None = None,
) -> PlacedPart:
    """The part ``key`` of :data:`PARTS` as component ``ref``, verified against the library on disk (see the module docstring for every refusal)."""
    part = PARTS.get(key)
    if part is None:
        raise TemplateRefusal(f"no part {key!r} in the kr447 parts table (known: {sorted(PARTS)})")
    refused = REFUSED_PARTS.get(part.symbol)
    if refused is not None:  # a table edit that re-adds a refused symbol
        raise TemplateRefusal(f"{ref}: {part.lib_id} is refused: {refused.reason}; use {refused.instead}")
    component = library_component(library, ref, value, description, part.symbol, part.footprint, provenance, list(serves or []))
    try:
        symbol = library.load_symbol(component.symbol)  # type: ignore[arg-type]
        footprint = library.load_footprint(component.footprint)  # type: ignore[arg-type]
    except (LibraryLookupError, LibraryFormatError) as e:
        raise TemplateRefusal(f"{ref} ({part.lib_id}): {e}") from e
    check_compilable(ref, symbol, footprint.pads, part.footprint_id)
    description_text = symbol.properties.get("Description", "")
    for fact in part.facts:
        if fact not in description_text:
            raise TemplateRefusal(
                f"{ref} ({part.lib_id}): the library description {description_text!r} no longer says {fact!r}, the text this part was chosen by"
            )
    functions, no_connect, notes = _resolve(part, component)
    stacks = symbol_stacks(symbol)
    of_pin = {n: f for f, numbers in functions.items() for n in numbers}
    stacked_functions: set[str] = set()
    for group in stacks:
        owners = {of_pin.get(n) for n in group}
        if len(owners) != 1 or None in owners:
            raise TemplateRefusal(
                f"{ref} ({part.lib_id}): the library stacks pins {list(group)} at one point (KiCad connects them), but the table gives them "
                f"functions {sorted(str(o) for o in owners)} - stacked pins must be one function in one net"
            )
        stacked_functions |= owners  # type: ignore[arg-type]
    if stacked_functions != set(part.stacked):
        raise TemplateRefusal(
            f"{ref} ({part.lib_id}): the library stacks the pins of {sorted(stacked_functions)}, the parts table records {sorted(part.stacked)}; "
            f"the table must name every stack so each is wired as one net"
        )
    return PlacedPart(part=part, component=component, functions=functions, stacks=stacks, no_connect=no_connect, notes=tuple(notes))


def check_compilable(ref: str, symbol: SymbolDef, pads, footprint_id: str) -> None:
    """Refuse, with the compilers' own reasons, a symbol / footprint pair the schematic or PCB compiler would refuse."""
    if symbol.units != [1]:
        raise TemplateRefusal(f"{ref}: symbol {symbol.lib_id!r} has units {symbol.units}; multi-unit symbols are not supported yet (both compilers refuse it)")
    numbers = [p.number for p in symbol.pins if p.body_style in (0, 1)]
    repeated = sorted({n for n in numbers if numbers.count(n) > 1}, key=_natural)
    if repeated:
        raise TemplateRefusal(f"{ref}: library symbol {symbol.lib_id!r} repeats pin number(s) {repeated} (stacked pins); unsupported - one physical pin per number")
    pins = set(numbers)
    pad_numbers = {p.number for p in pads if p.number}
    electrical = {p.number for p in pads if p.number and p.pad_type != "np_thru_hole"}
    without_pad = sorted(pins - pad_numbers, key=_natural)
    if without_pad:
        raise TemplateRefusal(f"{ref}: symbol pins {without_pad} of {symbol.lib_id!r} have no pad in footprint {footprint_id} (the PCB compiler refuses it)")
    extra = sorted(electrical - pins, key=_natural)
    if extra:
        raise TemplateRefusal(f"{ref}: footprint {footprint_id} has pads {extra} that are not pins of {symbol.lib_id!r} (the PCB compiler refuses it)")


def check_refused(library: KicadLibrary, symbol: tuple[str, str]) -> str:
    """Why a :data:`REFUSED_PARTS` pair is refused, re-derived from the library on disk (the compilers' reason), or the recorded reason."""
    row = REFUSED_PARTS[symbol]
    try:
        sym = library.load_symbol(library.resolve_symbol(LibraryRef(library=row.symbol[0], name=row.symbol[1])))
        fp = library.load_footprint(library.resolve_footprint(LibraryRef(library=row.footprint[0], name=row.footprint[1])))
        check_compilable("probe", sym, fp.pads, f"{row.footprint[0]}:{row.footprint[1]}")
    except TemplateRefusal as e:
        return str(e)
    except (LibraryLookupError, LibraryFormatError) as e:
        return f"not in the library: {e}"
    return ""


def part_line(placed: PlacedPart) -> str:
    """One confirmation-table line: ref, library ids, value and the pin notes."""
    c = placed.component
    notes = f"; {'; '.join(placed.notes)}" if placed.notes else ""
    return f"{c.ref} {placed.part.lib_id} / {placed.part.footprint_id}, value {c.value} ({placed.part.role}){notes}"


__all__ = [
    "CP_POLARITY",
    "PARTS",
    "REFUSED_PARTS",
    "SYMMETRIC_NOTE",
    "PartDef",
    "PinSpec",
    "PlacedPart",
    "RefusedPart",
    "check_compilable",
    "check_refused",
    "instantiate",
    "part_line",
    "symbol_stacks",
]
