"""The power block of the KR 447 MHz family (stage 1 and every later board): pack input, main switch, RX / TX rails, regulators.

Invariant: the block is built only from the parts table (:mod:`ai_eda.design.rf.parts`,
pins by library name), every number is a confirmed choice, an UNVERIFIED
``model.*`` value (:mod:`ai_eda.design.rf.models`), the confirmed
``input_voltage`` requirement or a registered calculator's output over those,
and nothing here claims a regulator works: no regulator has a SPICE model.

The circuit (kr447 design §2.1, decision 3A; local references, re-based by
100 on a board: ``J1`` -> ``J101``):

* J1 (pack connector, 2S Li-ion, protected, charged externally) -> F1 fuse ->
  Q5 AO3401A main P-FET switch (``VBAT_F`` -> ``V_SYS``), its gate pulled up
  to the source by R6 and taken to GND by the pole of SW1 (throw A on, throw C
  left open: off). SW1 carries only the gate current.
* RX / TX rails (``modes``): Q1 (``V_SYS`` -> ``V_RX``) and Q2 (``V_SYS`` ->
  ``V_TX``) are high-side P-FETs. ``PTT_N`` (the PTT switch of the ptt block,
  low while pressed) drives the inverter Q3, whose collector is ``PTT_ACTIVE``
  (pulled up to ``V_SYS`` by R8: low in RX, high in TX) - the gate of Q1
  directly (RX rail on while PTT is released) and, through the inverter Q4
  (R9 / R10), the gate of Q2 (TX rail on while PTT is pressed). A board
  without TX (``modes=("rx",)``) ties Q1's gate to GND: the RX rail is always
  on and nothing is sequenced.
* U1 LP38693DT-5.0 (TO-252-2, decision 3A) ``V_RX`` -> ``RX_5V``; U3
  LP5907-3.3 ``RX_5V`` -> ``RX_3V3``; U2 LM1117DT-5.0 ``V_TX`` -> ``TX_5V``; U4
  LP5907-3.3 ``TX_5V`` -> ``TX_3V3``; the TX LED D1 with R11 on ``V_TX``.
  The LP38693 and LP5907 take ceramic output capacitors; the LM1117 needs an
  output capacitor with ESR in a bounded window, so U2's output C5 is an
  electrolytic (``power.c_tx5v_out``, ``cap_polarized``) [UNVERIFIED: TI
  LM1117 datasheet] and TX_5V's stability is part of the lab item
  ``power_rails``.

Simulation (the design deck): the pack is the stimulus ``VBAT`` =
``input_voltage``, SW1's ON position the 0 V source ``SWON`` on ``MAIN_G``,
the P-FETs the level-1 ``model.pmos`` card and the inverters the generic
``model.npn`` card. No regulator has a model: each regulator's *input* is
simulated as a resistor ``model.ldo_rx.r_in`` / ``model.ldo_tx.r_in`` from IN
to GND (the load it draws, an UNVERIFIED model value), its output pin is
ignored, and the four output rails are ideal DC sources at the regulators'
nominal voltages (``power.rx_5v`` ...; always on - the regulators' own
sequencing is a lab item). The LED has no model and is excluded (its
resistor, then dead, goes with it). Checked: ``main_switch_on`` (op) and, on
a board with both modes, the PTT rail sequencing ``tx_rail_on`` /
``rx_rail_off_tx`` / ``tx_rail_off_rx`` on the ptt block's ``tran_ptt``.

The rail budgets (``ir.rf.rails``) are the design's current estimates, the
regulators' ratings as their library descriptions state them and the
dropouts - every one an UNVERIFIED choice, so ``power.rail_budget.*`` /
``power.headroom.*`` stay NOT_VERIFIED. ``power.pack_cutoff_v`` (6.4 V,
decision 3A) is the minimum input of the pack-fed regulators; the ptt
block's low-pack TX inhibit enforces it.

The module also holds the small helpers every stage-1 block uses (parts with
their SPICE binding, model values and choices declared once per block,
requirement copies, the rail levels, the op analysis); two blocks that
declare the same choice give identical rows, which
:func:`~ai_eda.design.rf.blocks.base.merge_results` merges.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ai_eda.ir import AnalysisSpec, Expectation, NetKind, Reduce, SpiceBinding, SpiceDevice, Stimulus, StimulusKind, Traced
from ai_eda.ir.rf import LabItem, RailBudget, RFPort
from ai_eda.tools.calc.basic import led_series_resistor
from ai_eda.tools.calc.part_value import format_part_value
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.inputs import MODULATION_ALIASES, MODULATION_KEY, RADIO_BUILD_ALIASES, RADIO_BUILD_KEY, read_modulation, read_radio_build
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import models
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.models import ModelCard, ModelValue
from ai_eda.design.rf.parts import CP_POLARITY, PlacedPart

#: the block's interface nets (the composition keeps these names)
RX_NETS: tuple[str, ...] = ("V_SYS", "V_RX", "RX_5V", "RX_3V3")
TX_NETS: tuple[str, ...] = ("V_TX", "TX_5V", "TX_3V3", "PTT_N", "PTT_ACTIVE")
#: the pack voltage parameter (a copy of the confirmed ``input_voltage``)
V_IN_KEY = "power.v_in"
#: the categorical requirement keys a part may serve -> (their aliases, their reader)
CATEGORICAL_READERS = {MODULATION_KEY: (MODULATION_ALIASES, read_modulation), RADIO_BUILD_KEY: (RADIO_BUILD_ALIASES, read_radio_build)}
#: the confirmed pack cut-off (decision 3A), read by ``power.headroom.*``
PACK_CUTOFF_KEY = "power.pack_cutoff_v"
PACK_CUTOFF_V = 6.4
#: 2S Li-ion full charge: the highest pack voltage the stage-1 parts see
PACK_MAX_V = 8.4

#: rail -> (parameter key, nominal V, the regulator it comes from)
RAIL_LEVELS: dict[str, tuple[str, float, str]] = {
    "RX_5V": ("power.rx_5v", 5.0, "U101 LP38693DT-5.0"),
    "TX_5V": ("power.tx_5v", 5.0, "U102 LM1117DT-5.0"),
    "RX_3V3": ("power.rx_3v3", 3.3, "U103 LP5907MFX-3.3"),
    "TX_3V3": ("power.tx_3v3", 3.3, "U104 LP5907MFX-3.3"),
}
#: rail -> (i_min, i_max, i_rating, dropout, path resistance or None): the design's budget (§6 risk 10, decision 3A), all UNVERIFIED
RAIL_BUDGET: dict[str, tuple[float, float, float, float, float | None]] = {
    "RX_5V": (0.086, 0.161, 0.5, 0.45, 0.17),
    "TX_5V": (0.287, 0.441, 0.8, 1.2, 0.17),
    "RX_3V3": (0.001, 0.01, 0.25, 0.25, None),
    "TX_3V3": (0.005, 0.02, 0.25, 0.25, None),
}
_BUDGET_SOURCE = {
    "RX_5V": ("86-161 mA (the LO buffer PHA-1 dominates on the transceiver)", '"500-mA" in the LP38693DT-5.0 library description', "0.25-0.45 V", "TI LP38693 datasheet"),
    "TX_5V": ("287-441 mA (PA 0.5 W at 35-50 %, driver, 3 BFR92, the PM buffer, PIN bias)", '"800mA" in the LM1117DT-5.0 library description', "1.2 V", "TI LM1117 datasheet"),
    "RX_3V3": ("1-10 mA (squelch comparator, pull-ups)", '"250-mA" in the LP5907 library description', "about 0.25 V", "TI LP5907 datasheet"),
    "TX_3V3": ("5-20 mA (MAX9814, op-amps, comparators, AND gate)", '"250-mA" in the LP5907 library description', "about 0.25 V", "TI LP5907 datasheet"),
}

#: the regulators' input drawn as a resistance in the design deck (no regulator has a model)
LDO_RX_LOAD = ModelValue("model.ldo_rx.r_in", 47.0, "ohm",
                         "U101's input drawn as a resistor from IN to GND in the design deck (about 157 mA at 7.4 V; the regulator has no model and its "
                         "output is the ideal source RX_5V)", "the board's measured RX supply current")
LDO_TX_LOAD = ModelValue("model.ldo_tx.r_in", 47.0, "ohm",
                         "U102's input drawn as a resistor from IN to GND in the design deck (about 157 mA at 7.4 V; the bench board draws far less, the "
                         "transceiver up to 441 mA; the regulator has no model and its output is the ideal source TX_5V)", "the board's measured TX supply current")


# --------------------------------------------------------------------------- helpers every stage-1 block uses


def choice_once(b: BlockBuilder, key: str, value: Any, unit: str | None, description: str) -> Traced:
    """The choice ``key`` of this block (declared on first use; the same key again returns it, a different value refuses)."""
    have = b.result.params.get(key)
    if have is not None:
        if have.value != value or have.unit != unit:
            raise TemplateRefusal(f"block {b.result.block_id}: choice {key!r} declared twice with different values")
        return have
    return b.choice(key, value, unit, description)


def model_once(b: BlockBuilder, value: str | ModelValue) -> Traced:
    """The model value of this block (declared on first use)."""
    key = value if isinstance(value, str) else value.key
    have = b.result.params.get(key)
    return have if have is not None else b.model(value)


def computed_once(b: BlockBuilder, key: str, make: Any) -> Traced:
    """The calculator output ``key`` (``make()`` computes it on first use); a calculator refusal refuses the block with its sentence."""
    have = b.result.params.get(key)
    if have is not None:
        return have
    try:
        traced = make()
    except (ValueError, ArithmeticError) as e:
        raise TemplateRefusal(f"block {b.result.block_id}: {key}: {e}") from e
    return b.computed(key, traced)


def input_copy(b: BlockBuilder, canon: str, key: str) -> Traced:
    """The confirmed requirement ``canon`` as the parameter ``key`` (its own provenance: the user's, derived from the requirement id)."""
    have = b.result.params.get(key)
    if have is not None:
        return have
    inp = b.ctx.inputs.get(canon)
    if inp is None:
        raise TemplateRefusal(f"block {b.result.block_id} needs the confirmed requirement {canon}")
    b.result.params[key] = inp.traced
    return inp.traced


def pack_voltage(b: BlockBuilder) -> Traced:
    """``power.v_in``: the stated pack voltage (``input_voltage``)."""
    return input_copy(b, "input_voltage", V_IN_KEY)


def pack_cutoff(b: BlockBuilder) -> Traced:
    """``power.pack_cutoff_v`` (decision 3A): the confirmed minimum pack voltage."""
    return choice_once(b, PACK_CUTOFF_KEY, PACK_CUTOFF_V, "V", (
        "pack cut-off (decision 3A): the 2S pack is used only above it; the ptt block's low-pack TX inhibit enforces it, and it is the minimum "
        "input of the pack-fed regulators. LM1117-5.0 keeps TX_5V in regulation down to about 6.275 V with a 1.2 V dropout and 170 mohm of path "
        "[UNVERIFIED: TI LM1117 datasheet]; the pack's own protection cut-off is lower [UNVERIFIED: pack datasheet]"))


def rail_level(b: BlockBuilder, rail: str) -> Traced:
    """The nominal level of ``RX_5V`` / ``TX_5V`` / ``RX_3V3`` / ``TX_3V3``: the regulator's output and the deck's ideal source."""
    key, volts, reg = RAIL_LEVELS[rail]
    return choice_once(b, key, volts, "V", (
        f"{rail}: the nominal output of {reg}; in the design deck an ideal DC source (the regulator has no SPICE model), always on - "
        f"the regulator's own start-up and sequencing are not simulated [UNVERIFIED: the regulator's datasheet]"))


#: the description of the choice ``cp_polarity`` every block with a ``Device:C_Polarized`` declares (identical rows merge on a board)
CP_POLARITY_TEXT = ("Device:C_Polarized pins carry no names in the KiCad library, so every electrolytic's polarity on this board is a pin-number "
                    "assumption: pin 1 = + (the symbol's '+' mark) on the net with the higher DC level, pin 2 = - - check each part's marking before "
                    "assembly [UNVERIFIED: footprint polarity mark]")


def cp_polarity(b: BlockBuilder) -> Traced:
    """The choice ``cp_polarity`` (the parts table's pin-number assumption of ``cap_polarized``), declared once per block."""
    return choice_once(b, CP_POLARITY, "pin 1 = +", None, CP_POLARITY_TEXT)


def value_text(t: Traced) -> str:
    return format_part_value(float(t.value))


def passive(b: BlockBuilder, key: str, ref: str, value: Traced, description: str, serves: Iterable[str] = ()) -> PlacedPart:
    """A resistor / capacitor / inductor / fuse of the parts table, simulated at ``value`` (R / C / L by the value's unit)."""
    device = {"ohm": SpiceDevice.R, "F": SpiceDevice.C, "H": SpiceDevice.L}.get(value.unit or "")
    if device is None:
        raise TemplateRefusal(f"{ref}: a passive's value must be in ohm, F or H, got {value.unit!r}")
    placed = b.part(key, ref, value_text(value), description, serves)
    pins = [placed.pin(f) for f in placed.functions] if key != "cap_polarized" else [placed.pin("+"), placed.pin("-")]
    b.bind(placed.ref, SpiceBinding(device=device, value=value, pin_order=pins, provenance=b.ctx.provenance(f"{ref} simulated at its value")))
    return b.result.placed[placed.ref]


def resistor(b: BlockBuilder, ref: str, value: Traced, description: str, *, key: str = "res_0603", serves: Iterable[str] = ()) -> PlacedPart:
    return passive(b, key, ref, value, description, serves)


def capacitor(b: BlockBuilder, ref: str, value: Traced, description: str, *, key: str = "cap_0603", serves: Iterable[str] = ()) -> PlacedPart:
    return passive(b, key, ref, value, description, serves)


def serving(ctx: BlockContext, *keys: str) -> list[str]:
    """The requirement ids a part serves: each confirmed numeric input among ``keys`` (``BlockContext.requirement_id``) and, for the
    categorical ``modulation`` / ``radio_build``, every requirement its reader reads (only when it reads one value). Keys the user did not
    state give nothing - a part never claims to serve a requirement that is not there."""
    out: list[str] = []
    for key in keys:
        if key in CATEGORICAL_READERS:
            aliases, reader = CATEGORICAL_READERS[key]
            if reader(ctx.ir)[0] is not None:
                out += [r.id for r in ctx.ir.requirements.requirements if r.key in aliases and r.id not in out]
            continue
        rid = ctx.requirement_id(key)
        if rid is not None and rid not in out:
            out.append(rid)
    return out


def exclude(b: BlockBuilder, ref: str, reason: str) -> None:
    b.bind(ref, SpiceBinding(exclude=True, exclude_reason=reason, provenance=b.ctx.provenance(f"{ref} excluded from the netlist")))


def card_bind(b: BlockBuilder, ref: str, card: ModelCard, *, functions: tuple[str, ...] | None = None, ignored: dict[str, str] | None = None) -> None:
    """Bind ``ref`` to ``card`` (the card's confirmation row is declared once per block)."""
    text = b.card(card)
    placed = b.result.placed[ref]
    b.bind(ref, models.card_binding(placed, card, text, b.ctx.provenance(f"{ref} simulated with {card.key}"), functions=functions, ignored=ignored))


def op_analysis(b: BlockBuilder) -> str:
    """The operating point ``op`` (declared once per block; identical in every block, so a board merges them)."""
    if not any(a.id == "op" for a in b.result.analyses):
        b.result.analyses.append(AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=b.ctx.provenance("operating point: supply, references and thresholds")))
    return "op"


def dc_source(b: BlockBuilder, sid: str, net: str, value: Traced, note: str) -> None:
    if not any(s.id == sid for s in b.result.stimuli):
        b.result.stimuli.append(Stimulus(id=sid, source="voltage", net=net, reference_net="GND", kind=StimulusKind.DC, value=value, provenance=b.ctx.provenance(note)))


def rail_source(b: BlockBuilder, rail: str) -> None:
    """The ideal DC source of ``rail`` (the regulator it comes from has no model)."""
    dc_source(b, f"V{rail.replace('_', '')}", rail, rail_level(b, rail), f"{rail}: ideal source at the regulator's nominal output (the regulator is not simulated)")


def expectation(b: BlockBuilder, eid: str, analysis: str, vector: str, reduce: Reduce, nominal: Traced, note: str, **kw: Any) -> None:
    b.result.expectations.append(Expectation(id=eid, analysis_id=analysis, vector=vector, reduce=reduce, nominal=nominal, provenance=b.ctx.provenance(note), **kw))


def rail_port(name: str, net: str, level: Traced, direction: str) -> RFPort:
    return RFPort(name=name, net=net, kind="rail", voltage_v=level, direction=direction)  # type: ignore[arg-type]


#: the kind of every stage-1 interface net: two blocks that name one must agree (the composition refuses a mismatch)
INTERFACE_KINDS: dict[str, NetKind] = {
    **{n: NetKind.POWER for n in ("V_SYS", "V_RX", "V_TX", "RX_5V", "RX_3V3", "TX_5V", "TX_3V3", "PA_5V")},
    **{n: NetKind.SIGNAL for n in ("PTT_N", "PTT_ACTIVE", "DLY", "DLY_OK", "PA_ON", "PA_PD", "MUTE")},
    **{n: NetKind.ANALOG for n in ("MICOUT", "PM_DRIVE", "DISC_OUT", "RSSI", "SPK+")},
    "GND": NetKind.GROUND,
}


def net(b: BlockBuilder, name: str, members: list[tuple[str, str]], note: str, kind: NetKind | None = None) -> None:
    """A net of the block: an interface net takes its kind from :data:`INTERFACE_KINDS` (``kind`` is for internal nets, SIGNAL by default)."""
    b.net(name, INTERFACE_KINDS.get(name, kind or NetKind.SIGNAL), members, note)


# --------------------------------------------------------------------------- the block


class PowerBlock(Block):
    """Pack input, main switch, PTT-switched RX / TX rails and their regulators (module docstring).

    ``modes``: ``("rx", "tx")`` (every stage-1 board and the transceiver),
    ``("rx",)`` (a receive-only board: Q101 always on, no TX section) or
    ``("tx",)`` (a transmit-only board: no RX section). ``rx_load`` /
    ``tx_load``: the model values the regulators' inputs are simulated with.
    A board with a TX section must also build the ptt block (its ``tran_ptt``
    runs the rail checks).
    """

    id = "power"
    title = "power: pack input, main switch, RX / TX rails, regulators"

    def __init__(self, modes: tuple[str, ...] = ("rx", "tx"), rx_load: ModelValue = LDO_RX_LOAD, tx_load: ModelValue = LDO_TX_LOAD) -> None:
        if not modes or set(modes) - {"rx", "tx"} or len(set(modes)) != len(modes):
            raise ValueError(f"modes must be a non-empty subset of ('rx', 'tx'), got {modes!r}")
        self.modes = tuple(m for m in ("rx", "tx") if m in modes)
        self.rx_load, self.tx_load = rx_load, tx_load
        nets = ["V_SYS"]
        if "rx" in self.modes:
            nets += ["V_RX", "RX_5V", "RX_3V3"]
        if "tx" in self.modes:
            nets += list(TX_NETS)
        self.interface_nets = tuple(nets)

    def build_local(self, ctx: BlockContext) -> BlockResult:
        rx, tx = "rx" in self.modes, "tx" in self.modes
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        v_in = pack_voltage(b)
        pack_cutoff(b)
        r_gate = choice_once(b, "power.r_main_gate", 100e3, "ohm", "R106: Q105's gate pull-up to its source (Q105 off while SW101 is open)")
        c_bulk = choice_once(b, "power.c_bulk", 10e-6, "F", (
            "bulk / input capacitor on V_SYS and V_TX and U101's output capacitor on RX_5V (10 uF ceramic; the LP38693 is stable with ceramic "
            "output capacitors [UNVERIFIED: TI LP38693 datasheet]; U102's output takes power.c_tx5v_out)"))
        c_small = choice_once(b, "power.c_small", 1e-6, "F", (
            "regulator capacitor on V_RX, RX_3V3, TX_3V3 and the LP5907 inputs (1 uF ceramic [UNVERIFIED: their datasheets])"))
        sw_on = choice_once(b, "power.sw_on", 0.0, "V", (
            "SW101 in the ON position: its pole ties Q105's gate (MAIN_G) to GND; the switch has no model, so the design deck drives MAIN_G with this "
            "0 V source (a modelling choice)"))
        v_sys_tol = choice_once(b, "power.v_sys_tol", 0.1, "V", (
            "main_switch_on: V_SYS within this of the pack voltage at the operating point (the drop of F101 and Q105 under the RX load)"))
        fuse_r = model_once(b, "model.fuse.r")
        pmos, npn = models.pmos_card(), models.npn_card()
        j1 = b.part("conn_2_jst", "J1", "2S Li-ion", "2S Li-ion pack connector (pack off-board, protected, charged externally)", serving(ctx, "input_voltage"))
        f1 = b.part("fuse", "F1", "Fuse", "pack fuse")
        sw1 = b.part("sw_spdt", "SW1", "SW_SPDT", "power switch: its pole B switches Q105's gate (A = GND is on, C open is off); it carries only the gate current")
        q5 = b.part("pfet", "Q5", "AO3401A", "main power switch (P-FET high side, VBAT_F -> V_SYS)")
        r6 = resistor(b, "R6", r_gate, "Q105 gate pull-up to its source")
        c1 = capacitor(b, "C1", c_bulk, "V_SYS bulk capacitor")
        b.bind("F1", SpiceBinding(device=SpiceDevice.R, value=fuse_r, pin_order=[f1.pin("1"), f1.pin("2")],
                                  provenance=ctx.provenance("F1 simulated as its cold resistance model.fuse.r")))
        exclude(b, "J1", "connector: the pack is the DC stimulus VBAT")
        exclude(b, "SW1", "switch without a model: the ON position is the 0 V stimulus SWON on MAIN_G")
        b.leave_open("SW1", "C", "throw C (the OFF position) is left open: R106 then holds Q105's gate at its source")
        card_bind(b, "Q5", pmos)
        members: dict[str, list[tuple[str, str]]] = {
            "VBAT": [*j1.at("Pin_1"), *f1.at("1")],
            "VBAT_F": [*f1.at("2"), *q5.at("S"), *r6.at("1")],
            "MAIN_G": [*q5.at("G"), *r6.at("2"), *sw1.at("B")],
            "V_SYS": [*q5.at("D"), *c1.at("1")],
            "GND": [*j1.at("Pin_2"), *sw1.at("A"), *c1.at("2")],
        }
        notes = {
            "VBAT": "the pack's + terminal to the fuse", "VBAT_F": "after the fuse: Q105's source and gate pull-up",
            "MAIN_G": "Q105's gate: R106 to the source, SW101's pole", "V_SYS": "the switched pack rail after the main switch",
            "V_RX": "the switched RX rail: Q101's drain, U101's input", "V_TX": "the switched TX rail: Q102's drain, U102's input, the TX LED",
            "RX_5V": "U101's output: RX audio and U103's input", "RX_3V3": "U103's output: the squelch comparator",
            "TX_5V": "U102's output: the PA supply switch and U104's input", "TX_3V3": "U104's output: TX audio, the PTT logic and comparators",
            "PTT_N": "the PTT switch (ptt block), low while pressed: Q103's base resistor", "PTT_ACTIVE": "Q103's collector: low in RX, high in TX",
            "PTT_INV_B": "Q103's base", "TXSW_B": "Q104's base", "TXSW_G": "Q102's gate: pulled up to V_SYS, pulled low by Q104 while PTT is pressed",
            "LED_A": "R111 to the TX LED's anode", "GND": "ground",
        }
        dc_source(b, "VBAT", "VBAT", v_in, "the 2S pack at the stated input_voltage")
        dc_source(b, "SWON", "MAIN_G", sw_on, "SW101 in the ON position (its pole to GND)")
        op_analysis(b)
        expectation(b, "main_switch_on", "op", "v(V_SYS)", Reduce.VALUE, v_in,
                    "V_SYS follows the pack through F101 and the ON main switch Q105 (model.pmos)", tol_abs=v_sys_tol)
        chain = ["J1", "F1", "Q5"]
        ports: list[RFPort] = []
        rails: list[RailBudget] = []

        def budget(rail: str, ref: str) -> None:
            i_min, i_max, rating, dropout, path = RAIL_BUDGET[rail]
            what, rated, drop, sheet = _BUDGET_SOURCE[rail]
            low = rail.lower()
            rails.append(RailBudget(
                rail=rail, regulator_ref=ref, v_out=rail_level(b, rail),
                i_min=choice_once(b, f"power.{low}.i_min", i_min, "A", f"{rail} load current, lowest: {what} [UNVERIFIED: an estimate, not a measurement]"),
                i_max=choice_once(b, f"power.{low}.i_max", i_max, "A", f"{rail} load current, highest: {what} [UNVERIFIED: an estimate, not a measurement]"),
                i_rating=choice_once(b, f"power.{low}.i_rating", rating, "A", f"{ref}'s current rating, {rated} [UNVERIFIED: {sheet}]"),
                dropout_v=choice_once(b, f"power.{low}.dropout", dropout, "V", f"{ref}'s dropout at the rail's load, {drop} [UNVERIFIED: {sheet}]"),
                path_r_ohm=None if path is None else choice_once(b, "power.path_r", path, "ohm", (
                    "resistance of the pack path before a pack-fed regulator: F101 (about 50 mohm) + Q105 + Q101 / Q102 R_DS(on) (about 60 mohm each) "
                    "[UNVERIFIED: fuse and AOS AO3401A datasheets]")),
            ))

        if rx:
            q1 = b.part("pfet", "Q1", "AO3401A", "RX rail switch (P-FET high side, V_SYS -> V_RX)" + (": on while PTT is released" if tx else ": always on (no TX section)"))
            u1 = b.part("ldo_rx5v", "U1", "LP38693DT-5.0", "RX_5V regulator (decision 3A: TO-252-2, 500 mA library description)")
            u3 = b.part("ldo_3v3", "U3", "LP5907MFX-3.3", "RX_3V3 low-noise regulator")
            c2 = capacitor(b, "C2", c_small, "U101 input capacitor (V_RX)")
            c3 = capacitor(b, "C3", c_bulk, "U101 output capacitor (RX_5V)")
            c6 = capacitor(b, "C6", c_small, "U103 input capacitor (RX_5V)")
            c7 = capacitor(b, "C7", c_small, "U103 output capacitor (RX_3V3)")
            card_bind(b, "Q1", pmos)
            b.bind("U1", SpiceBinding(device=SpiceDevice.R, value=model_once(b, self.rx_load), pin_order=[u1.pin("IN"), u1.pin("GND")],
                                      ignored_pins={u1.pin("OUT"): "the regulator's output: RX_5V is the ideal source VRX5V (LP38693 has no model)"},
                                      provenance=ctx.provenance(f"U1's input drawn as {self.rx_load.key} (the regulator has no model)")))
            exclude(b, "U3", "LP5907 has no model: RX_3V3 is the ideal source VRX3V3")
            members["V_SYS"] += q1.at("S")
            members["V_RX"] = [*q1.at("D"), *u1.at("IN"), *c2.at("1")]
            members["RX_5V"] = [*u1.at("OUT"), *c3.at("1"), *u3.at("IN"), *u3.at("EN"), *c6.at("1")]
            members["RX_3V3"] = [*u3.at("OUT"), *c7.at("1")]
            members["GND"] += [*u1.at("GND"), *u3.at("GND"), *c2.at("2"), *c3.at("2"), *c6.at("2"), *c7.at("2")]
            if not tx:
                members["GND"] += q1.at("G")  # always on: nothing is sequenced on a receive-only board
            rail_source(b, "RX_5V")
            rail_source(b, "RX_3V3")
            budget("RX_5V", "U1")
            budget("RX_3V3", "U3")
            ports += [rail_port("rx_5v", "RX_5V", rail_level(b, "RX_5V"), "out"), rail_port("rx_3v3", "RX_3V3", rail_level(b, "RX_3V3"), "out")]
            chain += ["Q1", "U1", "U3"]
        if tx:
            from ai_eda.design.rf.blocks.ptt import ptt_timing  # the PTT test plan: the ptt block declares the same rows

            r_inv = choice_once(b, "power.r_inv_base", 47e3, "ohm", "R107: Q103's base resistor from PTT_N (Q103 saturates while PTT is released)")
            r_active = choice_once(b, "power.r_ptt_active", 2.2e3, "ohm", "R108: PTT_ACTIVE pull-up to V_SYS (Q103's collector load; PTT_ACTIVE is Q101's gate)")
            r_txb = choice_once(b, "power.r_txsw_base", 220e3, "ohm", "R109: Q104's base resistor from PTT_ACTIVE (light: PTT_ACTIVE must stay near V_SYS to hold Q101 off)")
            r_txg = choice_once(b, "power.r_txsw_gate", 100e3, "ohm", "R110: Q102's gate pull-up to V_SYS (the TX rail is off unless Q104 conducts)")
            v_f = choice_once(b, "power.v_f_led", 2.0, "V", "forward voltage of the TX LED D101 (a typical indicator value; R111 is solved from it)")
            i_led = choice_once(b, "power.i_led", 2e-3, "A", "TX LED current (an indicator at 2 mA; R111 is solved from it)")
            r_led = computed_once(b, "power.r_led", lambda: led_series_resistor(v_in, v_f, i_led, (V_IN_KEY, "power.v_f_led", "power.i_led")))
            q2 = b.part("pfet", "Q2", "AO3401A", "TX rail switch (P-FET high side, V_SYS -> V_TX): on while PTT is pressed")
            q3 = b.part("npn_small", "Q3", "MMBT3904", "PTT inverter: PTT_N -> PTT_ACTIVE (Q101's gate)")
            q4 = b.part("npn_small", "Q4", "MMBT3904", "PTT inverter: PTT_ACTIVE -> Q102's gate")
            r7 = resistor(b, "R7", r_inv, "Q103 base resistor")
            r8 = resistor(b, "R8", r_active, "PTT_ACTIVE pull-up")
            r9 = resistor(b, "R9", r_txb, "Q104 base resistor")
            r10 = resistor(b, "R10", r_txg, "Q102 gate pull-up")
            u2 = b.part("ldo_tx5v", "U2", "LM1117DT-5.0", "TX_5V regulator (PA supply; about 1 W loss in TX on the transceiver)")
            u4 = b.part("ldo_3v3", "U4", "LP5907MFX-3.3", "TX_3V3 low-noise regulator (logic and TX audio supply)")
            c_tx5 = choice_once(b, "power.c_tx5v_out", 22e-6, "F", (
                "C105: U102's output capacitor, an aluminium electrolytic (Device:C_Polarized, CP_Elec_5x5.4, 16 V or more): the LM1117 is stable only "
                "with at least 10 uF on its output whose ESR lies inside a bounded window (about 0.3-22 ohm) - a ceramic's few milliohms are below it and "
                "TX_5V could oscillate [UNVERIFIED: TI LM1117 datasheet; the chosen part's ESR over temperature]. The ceramics elsewhere on TX_5V "
                "(U104's input capacitor, the stages' decoupling) sit in parallel: the rail's stability is the lab item power_rails"))
            c4 = capacitor(b, "C4", c_bulk, "U102 input capacitor (V_TX)")
            cp_polarity(b)
            c5 = capacitor(b, "C5", c_tx5, "U102 output capacitor (TX_5V): electrolytic for the LM1117's ESR window (pin 1 = +, choice cp_polarity)",
                           key="cap_polarized")
            c8 = capacitor(b, "C8", c_small, "U104 input capacitor (TX_5V)")
            c9 = capacitor(b, "C9", c_small, "U104 output capacitor (TX_3V3)")
            d1 = b.part("led", "D1", "LED", "TX indicator LED")
            r11 = resistor(b, "R11", r_led, "TX LED series resistor")
            card_bind(b, "Q2", pmos)
            card_bind(b, "Q3", npn)
            card_bind(b, "Q4", npn)
            b.bind("U2", SpiceBinding(device=SpiceDevice.R, value=model_once(b, self.tx_load), pin_order=[u2.pin("VI"), u2.pin("GND")],
                                      ignored_pins={u2.pin("VO"): "the regulator's output: TX_5V is the ideal source VTX5V (LM1117 has no model)"},
                                      provenance=ctx.provenance(f"U2's input drawn as {self.tx_load.key} (the regulator has no model)")))
            exclude(b, "U4", "LP5907 has no model: TX_3V3 is the ideal source VTX3V3")
            exclude(b, "D1", "LED without a model: the TX indicator is structural only")
            members["V_SYS"] += [*q2.at("S"), *r8.at("1"), *r10.at("1")]
            members["PTT_N"] = r7.at("1")
            members["PTT_ACTIVE"] = [*q3.at("C"), *r8.at("2"), *r9.at("1"), *(q1.at("G") if rx else [])]
            members["PTT_INV_B"] = [*r7.at("2"), *q3.at("B")]
            members["TXSW_B"] = [*r9.at("2"), *q4.at("B")]
            members["TXSW_G"] = [*q4.at("C"), *r10.at("2"), *q2.at("G")]
            members["V_TX"] = [*q2.at("D"), *u2.at("VI"), *c4.at("1"), *r11.at("1")]
            members["TX_5V"] = [*u2.at("VO"), *c5.at("+"), *u4.at("IN"), *u4.at("EN"), *c8.at("1")]
            members["TX_3V3"] = [*u4.at("OUT"), *c9.at("1")]
            members["LED_A"] = [*r11.at("2"), *d1.at("A")]
            members["GND"] += [*q3.at("E"), *q4.at("E"), *u2.at("GND"), *u4.at("GND"), *c4.at("2"), *c5.at("-"), *c8.at("2"), *c9.at("2"), *d1.at("K")]
            rail_source(b, "TX_5V")
            rail_source(b, "TX_3V3")
            budget("TX_5V", "U2")
            budget("TX_3V3", "U4")
            ports += [rail_port("tx_5v", "TX_5V", rail_level(b, "TX_5V"), "out"), rail_port("tx_3v3", "TX_3V3", rail_level(b, "TX_3V3"), "out")]
            chain += ["Q3", "Q4", "Q2", "U2", "U4"]
            t = ptt_timing(b)
            expectation(b, "tx_rail_on", t.analysis, "v(V_TX)", Reduce.AT, t.rail_on_min, "the TX rail is up once PTT is pressed (Q104 -> Q102)",
                        at=t.check_on, bound="at_least")
            expectation(b, "tx_rail_off_rx", t.analysis, "v(V_TX)", Reduce.AT, t.rail_off_max,
                        "the TX rail is down after PTT is released: the TX reference and chain are unpowered in RX (12 x Y_TX = f_c)",
                        at=t.check_off, bound="at_most")
            if rx:
                expectation(b, "rx_rail_off_tx", t.analysis, "v(V_RX)", Reduce.AT, t.rail_off_max,
                            "the RX rail is down while PTT is pressed (PTT_ACTIVE high holds Q101 off)", at=t.check_on, bound="at_most")
        for name, pins in members.items():
            net(b, name, pins, notes[name], NetKind.GROUND if name == "GND" else NetKind.POWER if name in ("VBAT", "VBAT_F") else None)
        b.result.ports = ports
        b.result.rails = rails
        b.result.chain = chain
        b.result.lab_items = [LabItem(id="power_rails", block=self.id,
                                      what="the rails' real currents in RX and TX, the regulators' dropouts and the temperature of Q105, U101 and U102"
                                      + (", and TX_5V's stability (no oscillation of U102 with its electrolytic output capacitor under a load step)" if tx else ""),
                                      instruments=["DMM", "adjustable supply", "thermal camera", *(["oscilloscope"] if tx else [])],
                                      reason="every current, rating and dropout of the rail budget is an UNVERIFIED choice; no regulator is simulated"
                                      + ("; the LM1117's output-capacitor ESR window is a datasheet fact [UNVERIFIED]" if tx else ""))]
        return b.done()


__all__ = [
    "CP_POLARITY_TEXT",
    "INTERFACE_KINDS",
    "LDO_RX_LOAD",
    "LDO_TX_LOAD",
    "PACK_CUTOFF_KEY",
    "PACK_CUTOFF_V",
    "PACK_MAX_V",
    "RAIL_BUDGET",
    "RAIL_LEVELS",
    "RX_NETS",
    "TX_NETS",
    "V_IN_KEY",
    "PowerBlock",
    "capacitor",
    "card_bind",
    "choice_once",
    "computed_once",
    "cp_polarity",
    "dc_source",
    "exclude",
    "expectation",
    "input_copy",
    "model_once",
    "net",
    "op_analysis",
    "pack_cutoff",
    "pack_voltage",
    "passive",
    "rail_level",
    "rail_port",
    "rail_source",
    "resistor",
    "value_text",
]
