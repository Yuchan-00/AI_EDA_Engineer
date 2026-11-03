"""The blocks of the UHF breadboard FM radio (``radio_build = uhf_breadboard``): discrete transistors and op-amps, through-hole for a perfboard.

Invariant: every part comes from the parts table by library name
(:mod:`ai_eda.design.rf.parts`, the ``bb_*`` rows), every number is a choice
of the confirmation table, a ``model.*`` value (UNVERIFIED) or a registered
calculator's output over them and the confirmed requirements, and every
passive network between unmodelled transistors is an RF fixture
(``ir.rf.networks``) the fixture runner judges on ngspice. Nothing here says
that the oscillator oscillates, that a multiplier multiplies, that the PA
delivers its power or that the super-regenerative detector receives: the
transistors of the RF path have no model, the crystal and the varactor are
excluded, and those claims are lab items (``ir.rf.lab_items``). A fixture
PASS is "a network verdict under confirmed model values (not a measured
part)" at schematic level - no lead, wire or ground-return inductance, which
on a perfboard at 432 MHz is the largest unknown.

The radio (local references, re-based by hundreds on the board):

* :class:`BbPowerBlock` (``power``, 1xx): 9 V battery header -> power slide
  switch -> 1N5817 -> ``V9`` -> the TX / RX slide switch -> ``V_TX`` /
  ``V_RX``; two L78L05 give ``V5T`` (the VXO) and ``V5R`` (the receiver).
* :class:`BbTxAudioBlock` (``tx_audio``, 2xx): electret -> TL071
  non-inverting stage with a diode soft limiter across its feedback ->
  TL071 unity-gain Sallen-Key Butterworth low-pass at ``audio_bandwidth`` ->
  deviation trimmer -> ``VAR`` (the varactor's bias node, at a divider's DC
  level).
* :class:`BbTxRfBlock` (``tx_rf``, 3xx): the 2N3904 Colpitts VXO with the
  crystal at f_c / 12 in series with the varactor (direct FM) -> BF199 x3
  -> BFR92A x2 -> BFR92A x2, each collector feeding a single-tuned top-C
  tank (``tx_tank1`` / ``tx_tank2``) or the 3-resonator band-pass
  ``tx_bpf`` at f_c through its feed choke -> the BFR92A class-A PA -> the
  low-pass L-match ``pa_match`` to the system impedance -> the 5th-order
  Chebyshev low-pass ``lpf`` -> ``TX_OUT``.
* :class:`BbTrxBlock` (``trx``, 6xx): the diode T/R switch - 1N4148 in
  series from ``TX_OUT`` to the antenna node, a lumped lambda/4 pi to the
  receiver and a 1N4148 shunting its far end, both forward-biased from
  ``V_TX`` only while transmitting - and the SMA.
* :class:`BbRxBlock` (``rx``, 4xx): BFR92A common-base RF amplifier ->
  self-quenched BFR92A super-regenerative detector (tank tuned by a
  trimmer, regeneration by a trimpot) -> ``AF_RAW`` through the quench
  filter.
* :class:`BbRxAudioBlock` (``rx_audio``, 5xx): TL071 gain stage -> TL071
  Sallen-Key low-pass (the voice band; it also removes the quench) ->
  volume -> TL071 driving a BD139 / BD140 complementary follower inside its
  feedback loop -> speaker.

The design deck simulates the audio chains, the references and the output
stage (op-amps as ``model.opamp``, BD139 / BD140 as the generic
``model.npn`` / ``model.pnp``, diodes as ``model.diode``); every RF part is
excluded from it and judged in its fixture or in the lab.
"""

from __future__ import annotations

from collections.abc import Iterable

from ai_eda.ir import NetKind, Reduce, SpiceBinding, SpiceDevice, Stimulus, StimulusKind, Traced
from ai_eda.ir.rf import LabItem, PlanLine, RFNetwork, RFPort, RFState
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import clock_divided, parallel_resistance, rc_time_constant, voltage_divider_output
from ai_eda.tools.calc.radio import butterworth_db_at, butterworth_q, db_sum, emphasis_corner, highpass1_db_at, lowpass1_db_at, noninv_gain_db
from ai_eda.tools.calc.radio import sallen_key_c1, sallen_key_c2
from ai_eda.tools.calc.rf import (
    chebyshev_g,
    lc_c_for_resonance,
    lmatch_lowpass_c_shunt,
    lmatch_lowpass_l_series,
    lpf_series_l,
    lpf_shunt_c,
)
from ai_eda.tools.calc.part_value import format_part_value

from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import models
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.power import card_bind, choice_once, computed_once, cp_polarity, dc_source, expectation, input_copy, model_once, op_analysis, serving
from ai_eda.design.rf.blocks.tx_audio import ac_point, db_row
from ai_eda.design.rf.blocks.tx_chain import NetBook, PortLoad, ac_sweep, build_top_c, fixture_port, index_choices, row
from ai_eda.design.rf.parts import PlacedPart

#: the multiplication factor of the TX chain and its three stages (36 MHz class crystal -> x3 -> x2 -> x2)
N_MULT = 12
STAGE_MULTIPLIERS: tuple[int, ...] = (3, 2, 2)
#: the carrier range of the board: the amateur 70 cm band as ITU Region 3 / KR allocate it [UNVERIFIED: the licence's band plan]
F_C_RANGE_HZ: tuple[float, float] = (430e6, 440e6)
#: the battery range the board is designed for (V): an L78L05 needs about 1.7 V of headroom over 5 V behind the 1N5817's drop, a 9 V battery or a 12 V supply
V_IN_RANGE: tuple[float, float] = (7.5, 12.0)
#: the interface nets of every block and their kinds (two blocks that name one must agree)
INTERFACE_KINDS: dict[str, NetKind] = {
    **{n: NetKind.POWER for n in ("V9", "V_TX", "V_RX", "V5T", "V5R")},
    **{n: NetKind.ANALOG for n in ("VAR", "AF_RAW")},
    **{n: NetKind.RF for n in ("TX_OUT", "ANT", "RX_IN")},
    GROUND_NET: NetKind.GROUND,
}
#: the design deck's audio analyses (single-point ac at each row's frequency)
AC_REF, AC_BW, AC_2BW, AC_QUENCH = "ac_f_ref", "ac_f_bw", "ac_f_2bw", "ac_f_quench"
#: the quench-rate row both receiver blocks declare (identical rows merge on the board)
QUENCH_TEXT = "the expected quench rate (ngspice-42 transient of this detector with a generic 5 GHz card) [UNVERIFIED: a lab measurement]"
#: why the RF transistors and the crystal are not in the design deck
NO_RF_MODEL = ("an RF-path part of the UHF breadboard radio: no model is simulated in the design deck (its networks are RF fixtures, its behaviour a lab "
               "item)")


# --------------------------------------------------------------------------- helpers


def signal_order(b: BlockBuilder) -> list[str]:
    """Every part of the block in the order the block built it - its signal path, stage by stage (the floorplan packs a region in this order,
    so each stage's bias parts sit beside its transistor: on the KiCad board and on the perfboard alike)."""
    return [c.ref for c in b.result.components]


def _c(b: BlockBuilder, key: str, value, unit: str | None, text: str) -> Traced:
    return choice_once(b, key, value, unit, text)


def _k(b: BlockBuilder, key: str, make) -> Traced:
    return computed_once(b, key, make)


def _ref(b: BlockBuilder, key: str) -> Traced:
    """A plan value an earlier block of the board wrote (``ctx.shared``) or this block did: refuses when neither did."""
    have = b.result.params.get(key)
    if have is None:
        have = b.ctx.shared.get(key)
    if have is None:
        raise TemplateRefusal(f"block {b.result.block_id}: plan value {key!r} is written by another block of the board: compose that block first")
    b.result.params[key] = have
    return have


def _bind_value(b: BlockBuilder, placed: PlacedPart, value: Traced, note: str) -> SpiceBinding:
    device = {"ohm": SpiceDevice.R, "F": SpiceDevice.C, "H": SpiceDevice.L}.get(value.unit or "")
    if device is None:
        raise TemplateRefusal(f"{placed.ref}: a passive's value must be in ohm, F or H, got {value.unit!r}")
    pins = [placed.pin("+"), placed.pin("-")] if "+" in placed.functions else [placed.pin("1"), placed.pin("2")]
    return SpiceBinding(device=device, value=value, pin_order=pins, provenance=b.ctx.provenance(f"{placed.ref}: {note}"))


def _exclude(b: BlockBuilder, ref: str, reason: str) -> None:
    b.bind(ref, SpiceBinding(exclude=True, exclude_reason=reason, provenance=b.ctx.provenance(f"{ref} excluded from the design deck")))


def part(b: BlockBuilder, key: str, ref: str, value: Traced, what: str, *, deck: bool = True, network: str = "", serves: Iterable[str] = ()) -> tuple[PlacedPart, SpiceBinding]:
    """A resistor / capacitor / coil / trimmer / electrolytic of the ``bb_*`` rows at ``value``: bound in the deck, or excluded (a fixture member)."""
    if key == "bb_cap_elec":
        cp_polarity(b)
    placed = b.part(key, ref, format_part_value(float(value.value)), what, serves)
    binding = _bind_value(b, placed, value, f"at its value{' (' + network + ' fixture binding)' if network else ''}")
    if deck:
        b.bind(ref, binding)
    elif network == "lab":
        _exclude(b, ref, NO_RF_MODEL)
    else:
        _exclude(b, ref, f"an RF network part: simulated only in its fixture ({network}) at its calculator value")
    return b.result.placed[ref], binding


def _p(b: BlockBuilder, nb: NetBook, key: str, ref: str, value: Traced, what: str, n1: str, n2: str, *, deck: bool = True, network: str = "",
       serves: Iterable[str] = ()) -> SpiceBinding:
    """A two-terminal part between nets ``n1`` (pin 1 / +) and ``n2`` (pin 2 / -), added to the net book."""
    placed, binding = part(b, key, ref, value, what, deck=deck, network=network, serves=serves)
    a, c = ("+", "-") if "+" in placed.functions else ("1", "2")
    _add(nb, n1, [(ref, placed.pin(a))])
    _add(nb, n2, [(ref, placed.pin(c))])
    return binding


def _add(nb: NetBook, net: str, pins: list[tuple[str, str]], note: str = "") -> None:
    nb.add(net, INTERFACE_KINDS.get(net, NetKind.ANALOG), pins, note)


def _ac_plan(b: BlockBuilder) -> tuple[Traced, Traced, Traced, Traced, Traced]:
    """``(f_ref, bandwidth, f_2bw, ac level, tolerance)``: the audio test plan both audio blocks share (identical rows merge on the board)."""
    f_ref = _c(b, "audio.f_ref", 1000.0, "Hz", "the reference tone: 1 kHz (the gain rows)")
    if "audio_bandwidth" in b.ctx.inputs:
        bw = input_copy(b, "audio_bandwidth", "audio.bandwidth")
    else:
        bw = _c(b, "audio.bandwidth", 3000.0, "Hz", "audio_bandwidth is not stated: the voice band ends at 3 kHz (both Sallen-Key low-passes' corner)")
    mult = _c(b, "audio.bw_multiple", 2.0, None, "the low-passes' stopband row is checked at twice the audio bandwidth")
    f_2bw = _k(b, "audio.f_2bw", lambda: radio.harmonic(bw, mult, ("audio.bandwidth", "audio.bw_multiple")))
    ac = _c(b, "audio.ac_level", 1.0, "V", "the ac magnitude of the audio stimuli (the rows are ratios, so it cancels)")
    tol = _c(b, "audio.tol_db", 0.5, "dB", ("tolerance of the audio ac rows (0.5 dB: the nominal multiplies first-order terms, which this topology "
                                            "only approximates, and the model.opamp macro's finite GBW)"))
    variation = _c(b, "audio.ac_variation", "lin", None, "every audio ac analysis is a single point 'ac lin 1 f f', so every dB row reads an exact sample")
    points = _c(b, "audio.ac_points", 1, None, "one point per ac analysis (at the row's own frequency)")
    for aid, f, what in ((AC_REF, f_ref, "the reference tone"), (AC_BW, bw, "the audio bandwidth"), (AC_2BW, f_2bw, "twice the audio bandwidth")):
        ac_point(b, aid, f, variation, points, what)
    return f_ref, bw, f_2bw, ac, tol


def _opamp(b: BlockBuilder, ref: str, what: str, serves: Iterable[str] = ()) -> PlacedPart:
    """A TL071 on the generic op-amp macro; its offset-null pins left open, its supply pins ignored by the macro."""
    u = b.part("bb_opamp", ref, "TL071", what, serves)
    b.leave_open(ref, "NULL", "the TL071's offset-null pins are not used (the stages are AC-coupled)")
    a0, gbw = model_once(b, "model.opamp.a0"), model_once(b, "model.opamp.gbw")
    card_bind(b, ref, models.opamp_card(a0, gbw), ignored={"V+": models.OPAMP_SUPPLY_IGNORED, "V-": models.OPAMP_SUPPLY_IGNORED})
    return b.result.placed[ref]


def _gain_stage(b: BlockBuilder, nb: NetBook, *, prefix: str, refs: dict[str, str], nets: dict[str, str], vref: str, rail: str, f_ref: Traced,
                what: str, limiter: bool, serves: Iterable[str] = ()) -> tuple[str, Traced]:
    """A TL071 non-inverting AC gain stage on a mid-supply reference; returns ``(key, nominal dB at f_ref)`` of v(out) re v(in).

    ``in`` -> C_in -> ``+`` (R_bias to ``vref``); R_f // C_f from the output to ``-``; R_g + C_g from ``-`` to GND (unity DC gain, so the
    output sits at ``vref``). With ``limiter`` two 1N4148 across R_f clip the output at about one diode drop from ``vref`` (a soft limiter).
    The nominal is the product of the stage's first-order terms (``calc.audio.noninv_gain.db`` + the C_f low-pass + the C_g and C_in
    high-passes, summed by ``calc.rf.db_sum``): within 0.05 dB of the exact network at 1 kHz for these values.
    """
    p = prefix
    r_f = _c(b, f"{p}.r_f", 100e3, "ohm", f"{what}: feedback resistor R_f")
    r_g = _c(b, f"{p}.r_g", 1e3, "ohm", f"{what}: ground-leg resistor R_g (the mid-band gain 1 + R_f / R_g)")
    c_f = _c(b, f"{p}.c_f", 470e-12, "F", f"{what}: capacitor across R_f (a first-order roll-off above the voice band)")
    c_g = _c(b, f"{p}.c_g", 10e-6, "F", f"{what}: ground-leg electrolytic (unity gain at DC, so the output stays on the reference)")
    c_in = _c(b, f"{p}.c_in", 1e-6, "F", f"{what}: input coupling electrolytic")
    r_b = _c(b, f"{p}.r_bias", 100e3, "ohm", f"{what}: bias resistor from the + input to the reference")
    g = _k(b, f"{p}.gain_db", lambda: noninv_gain_db(r_f, r_g, (f"{p}.r_f", f"{p}.r_g")))
    f_f = _k(b, f"{p}.f_cf", lambda: emphasis_corner(_k(b, f"{p}.tau_cf", lambda: rc_time_constant(r_f, c_f, (f"{p}.r_f", f"{p}.c_f"))), (f"{p}.tau_cf",)))
    f_g = _k(b, f"{p}.f_cg", lambda: emphasis_corner(_k(b, f"{p}.tau_cg", lambda: rc_time_constant(r_g, c_g, (f"{p}.r_g", f"{p}.c_g"))), (f"{p}.tau_cg",)))
    f_i = _k(b, f"{p}.f_cin", lambda: emphasis_corner(_k(b, f"{p}.tau_cin", lambda: rc_time_constant(r_b, c_in, (f"{p}.r_bias", f"{p}.c_in"))), (f"{p}.tau_cin",)))
    lp = _k(b, f"{p}.lp_ref", lambda: lowpass1_db_at(f_ref, f_f, ("audio.f_ref", f"{p}.f_cf")))
    hg = _k(b, f"{p}.hpg_ref", lambda: highpass1_db_at(f_ref, f_g, ("audio.f_ref", f"{p}.f_cg")))
    hi = _k(b, f"{p}.hpin_ref", lambda: highpass1_db_at(f_ref, f_i, ("audio.f_ref", f"{p}.f_cin")))
    s1 = _k(b, f"{p}.nom_1", lambda: db_sum(g, lp, (f"{p}.gain_db", f"{p}.lp_ref")))
    s2 = _k(b, f"{p}.nom_2", lambda: db_sum(s1, hg, (f"{p}.nom_1", f"{p}.hpg_ref")))
    nom = _k(b, f"{p}.nom_ref", lambda: db_sum(s2, hi, (f"{p}.nom_2", f"{p}.hpin_ref")))
    if limiter:  # the limiter diodes first: the floorplan packs them ahead of the op-amp
        card = models.diode_card()
        for ref, a, k in ((refs["d1"], nets["out"], nets["minus"]), (refs["d2"], nets["minus"], nets["out"])):
            d = b.part("bb_diode", ref, "1N4148", f"{what}: soft limiter diode across R_f")
            card_bind(b, ref, card)
            _add(nb, a, d.at("A"))
            _add(nb, k, d.at("K"))
    u = _opamp(b, refs["u"], what, serves)
    _p(b, nb, "bb_cap_elec", refs["c_in"], c_in, f"{what}: input coupling", nets["plus"], nets["in"])
    _p(b, nb, "bb_res", refs["r_bias"], r_b, f"{what}: bias to the reference", nets["plus"], vref)
    _p(b, nb, "bb_res", refs["r_f"], r_f, f"{what}: feedback resistor", nets["out"], nets["minus"])
    _p(b, nb, "bb_cap", refs["c_f"], c_f, f"{what}: feedback roll-off capacitor", nets["out"], nets["minus"])
    _p(b, nb, "bb_res", refs["r_g"], r_g, f"{what}: ground-leg resistor", nets["minus"], nets["g"])
    _p(b, nb, "bb_cap_elec", refs["c_g"], c_g, f"{what}: ground-leg electrolytic", nets["g"], GROUND_NET)
    dec = _c(b, "audio.c_decouple", 100e-9, "F", "supply decoupling of each op-amp")
    _p(b, nb, "bb_cap", refs["c_dec"], dec, f"{what}: supply decoupling", rail, GROUND_NET)
    _add(nb, nets["plus"], u.at("+"))
    _add(nb, nets["minus"], u.at("-"))
    _add(nb, nets["out"], u.at("OUT"))
    _add(nb, rail, u.at("V+"))
    _add(nb, GROUND_NET, u.at("V-"))
    return f"{p}.nom_ref", nom


def _sallen_key(b: BlockBuilder, nb: NetBook, *, prefix: str, refs: dict[str, str], nets: dict[str, str], rail: str, bw: Traced, what: str,
                serves: Iterable[str] = ()) -> None:
    """A unity-gain Sallen-Key Butterworth low-pass (R1 = R2 = R, C1 / C2 from ``calc.audio.sallen_key.*`` at Q = ``calc.audio.butterworth.q``(2, 1))."""
    p = prefix
    order = _c(b, "audio.sk_order", 2.0, None, "each Sallen-Key section is a 2nd-order Butterworth low-pass (one pole pair, Q 0.7071)")
    pair = _c(b, "audio.sk_pair", 1.0, None, "the Sallen-Key section realises the Butterworth pole pair k = 1 of order 2")
    r = _c(b, "audio.sk_r", 10e3, "ohm", "the Sallen-Key resistors (R1 = R2 = 10 kohm; C1 / C2 are solved from them)")
    q = _k(b, "audio.sk_q", lambda: butterworth_q(order, pair, ("audio.sk_order", "audio.sk_pair")))
    c1 = _k(b, "audio.sk_c1", lambda: sallen_key_c1(q, bw, r, ("audio.sk_q", "audio.bandwidth", "audio.sk_r")))
    c2 = _k(b, "audio.sk_c2", lambda: sallen_key_c2(q, bw, r, ("audio.sk_q", "audio.bandwidth", "audio.sk_r")))
    u = _opamp(b, refs["u"], what, serves)
    _p(b, nb, "bb_res", refs["r1"], r, f"{what}: R1", nets["in"], nets["n1"])
    _p(b, nb, "bb_res", refs["r2"], r, f"{what}: R2", nets["n1"], nets["n2"])
    _p(b, nb, "bb_cap", refs["c1"], c1, f"{what}: C1 (feedback)", nets["n1"], nets["out"])
    _p(b, nb, "bb_cap", refs["c2"], c2, f"{what}: C2 (to ground)", nets["n2"], GROUND_NET)
    dec = _c(b, "audio.c_decouple", 100e-9, "F", "supply decoupling of each op-amp")
    _p(b, nb, "bb_cap", refs["c_dec"], dec, f"{what}: supply decoupling", rail, GROUND_NET)
    _add(nb, nets["n2"], u.at("+"))
    _add(nb, nets["out"], [*u.at("-"), *u.at("OUT")])
    _add(nb, rail, u.at("V+"))
    _add(nb, GROUND_NET, u.at("V-"))


def _vref(b: BlockBuilder, nb: NetBook, *, refs: tuple[str, str, str], net: str, rail: str, rail_key: str, rail_v: Traced, eid: str) -> None:
    """The mid-supply reference of a single-supply audio chain: two equal resistors and an electrolytic, its level an op row."""
    r = _c(b, "audio.r_vref", 100e3, "ohm", "the mid-supply reference dividers VREF_T / VREF_R (two equal resistors)")
    c = _c(b, "audio.c_vref", 10e-6, "F", "the reference's decoupling electrolytic (VREF is the signal ground of the single-supply stages)")
    tol = _c(b, "audio.vref_tol", 0.01, None, "the references' op rows: within 1 % of the divider (ideal resistors; the op-amps' bias current is not modelled)")
    v = _k(b, f"{net.lower()}.v", lambda: voltage_divider_output(rail_v, r, r, (rail_key, "audio.r_vref", "audio.r_vref")))
    _p(b, nb, "bb_res", refs[0], r, f"{net} divider (top)", rail, net)
    _p(b, nb, "bb_res", refs[1], r, f"{net} divider (bottom)", net, GROUND_NET)
    _p(b, nb, "bb_cap_elec", refs[2], c, f"{net} decoupling", net, GROUND_NET)
    expectation(b, eid, op_analysis(b), f"v({net})", Reduce.VALUE, v, f"{net}: half the rail (the reference divider)", tol_rel=tol)


def _rails(b: BlockBuilder) -> tuple[Traced, Traced]:
    """``(bb.v_rail, bb.v5)``: the switched 9 V rails' level (the confirmed input voltage) and the L78L05 outputs' nominal 5 V."""
    v_rail = input_copy(b, "input_voltage", "bb.v_rail")
    v5 = _c(b, "bb.v5", 5.0, "V", ("V5T / V5R: the L78L05 outputs; in the design deck ideal DC sources (the regulator has no model) "
                                   "[UNVERIFIED: ST L78L05 datasheet]"))
    return v_rail, v5


# --------------------------------------------------------------------------- power


class BbPowerBlock(Block):
    """9 V battery header, power switch, reverse-polarity diode, TX / RX switch and the two 5 V regulators (module docstring)."""

    id = "power"
    title = "power: 9 V battery, power switch, 1N5817, TX / RX switch, two L78L05"
    interface_nets = ("V9", "V_TX", "V_RX", "V5T", "V5R")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        v_rail, v5 = _rails(b)
        c_bulk = _c(b, "bb.c_bulk", 100e-6, "F", "the rails' bulk electrolytics (V9 and V_RX, which feeds the speaker stage)")
        c_dec = _c(b, "bb.c_dec", 100e-9, "F", "rail decoupling capacitors")
        c_reg_in = _c(b, "bb.c_reg_in", 330e-9, "F", "the L78L05 input capacitor [UNVERIFIED: ST L78L05 datasheet]")
        j1 = b.part("bb_header_2", "J1", "9V", "9 V battery snap (Pin_1 = +)", serving(ctx, "input_voltage"))
        _exclude(b, "J1", "connector: the battery is the rails' DC sources")
        sw1 = b.part("bb_sw_slide", "SW1", "POWER", "power switch (B common, A = on, C = off)")
        _exclude(b, "SW1", "switch: no model (the deck is the ON position)")
        b.leave_open("SW1", "C", "the power switch's off throw is left open")
        d1 = b.part("bb_schottky", "D1", "1N5817", "reverse-polarity diode in series with the battery")
        _exclude(b, "D1", "the rails are ideal DC sources in the deck")
        sw2 = b.part("bb_sw_slide", "SW2", "TX/RX", "transmit / receive switch (B common from V9, A = receive -> V_RX, C = transmit -> V_TX)")
        _exclude(b, "SW2", "switch: no model (both rails are DC sources in the deck)")
        u1 = b.part("bb_reg5", "U1", "L78L05", "V5T: the VXO's regulated 5 V (on only while transmitting)", serving(ctx, "input_voltage"))
        u2 = b.part("bb_reg5", "U2", "L78L05", "V5R: the receiver's regulated 5 V (on only while receiving)", serving(ctx, "input_voltage"))
        for ref in ("U1", "U2"):
            _exclude(b, ref, "regulator: no model (its output is an ideal DC source)")
        _add(nb, "VBAT", [*j1.at("Pin_1"), *sw1.at("B")], "the battery's + terminal")
        _add(nb, "VBAT_ON", [*sw1.at("A"), *d1.at("A")], "after the power switch")
        _add(nb, "V9", [*d1.at("K"), *sw2.at("B")], "the switched battery rail")
        _add(nb, "V_RX", [*sw2.at("A"), *u2.at("VI")], "the receive rail (9 V while receiving)")
        _add(nb, "V_TX", [*sw2.at("C"), *u1.at("VI")], "the transmit rail (9 V while transmitting)")
        _add(nb, "V5T", u1.at("VO"), "the VXO's 5 V")
        _add(nb, "V5R", u2.at("VO"), "the receiver's 5 V")
        _add(nb, GROUND_NET, [*j1.at("Pin_2"), *u1.at("GND"), *u2.at("GND")])
        _p(b, nb, "bb_cap_elec", "C1", c_bulk, "V9 bulk", "V9", GROUND_NET)
        _p(b, nb, "bb_cap", "C2", c_dec, "V9 decoupling", "V9", GROUND_NET)
        _p(b, nb, "bb_cap", "C3", c_reg_in, "U1 input capacitor", "V_TX", GROUND_NET)
        _p(b, nb, "bb_cap", "C4", c_dec, "V5T decoupling", "V5T", GROUND_NET)
        _p(b, nb, "bb_cap", "C5", c_reg_in, "U2 input capacitor", "V_RX", GROUND_NET)
        _p(b, nb, "bb_cap", "C6", c_dec, "V5R decoupling", "V5R", GROUND_NET)
        _p(b, nb, "bb_cap_elec", "C7", c_bulk, "V_RX bulk (the speaker stage)", "V_RX", GROUND_NET)
        nb.declare(b)
        for sid, net, value, note in (("VV9", "V9", v_rail, "V9: the battery behind the switch (the 1N5817's drop is not simulated)"),
                                      ("VVTX", "V_TX", v_rail, "V_TX: the transmit rail as an ideal source (both switch positions are simulated at once)"),
                                      ("VVRX", "V_RX", v_rail, "V_RX: the receive rail as an ideal source"),
                                      ("VV5T", "V5T", v5, "V5T: U1's nominal output (the regulator is not simulated)"),
                                      ("VV5R", "V5R", v5, "V5R: U2's nominal output")):
            dc_source(b, sid, net, value, note)
        b.result.chain = signal_order(b)
        b.result.ports = [RFPort(name=n, net=n, kind="rail", voltage_v=v, direction="out") for n, v in (("V_TX", v_rail), ("V_RX", v_rail), ("V5T", v5), ("V5R", v5))]
        b.result.lab_items = [LabItem(id="bb_supply", block=self.id, what="the battery current while receiving and while transmitting, and both L78L05 outputs",
                                      instruments=["multimeter"], reason="the regulators and the RF stages' currents are not simulated")]
        return b.done()


# --------------------------------------------------------------------------- TX audio


class BbTxAudioBlock(Block):
    """Electret -> TL071 gain + soft limiter -> TL071 Sallen-Key low-pass -> deviation trimmer -> ``VAR`` (module docstring)."""

    id = "tx_audio"
    title = "TX audio: microphone amplifier with soft limiter, Sallen-Key low-pass, deviation trimmer, varactor bias"
    interface_nets = ("V_TX", "V5T", "VAR")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        v_rail, v5 = _rails(b)
        f_ref, bw, f_2bw, ac, tol = _ac_plan(b)
        mic_r = _c(b, "tx.mic_bias_r", 4.7e3, "ohm", "R3: electret bias resistor from V5T [UNVERIFIED: PUI Audio POM-2244P-C3310-2-R datasheet]")
        r_dev = _c(b, "tx.r_dev", 47e3, "ohm", "R5: from the deviation path's coupling capacitor to the varactor node VAR (with R6 it halves the audio there)")
        c_wip = _c(b, "tx.c_dev_wiper", 1e-6, "F", "C12: couples the deviation trimmer's wiper into R5 (no DC through R5: VAR sits exactly on VBIAS)")
        r_vb = _c(b, "tx.r_var_bias", 47e3, "ohm", "R6: from the bias divider VBIAS to VAR (the varactor's reverse bias)")
        r_b1 = _c(b, "tx.r_vbias_top", 27e3, "ohm", "R7: VBIAS divider from V5T (top)")
        r_b2 = _c(b, "tx.r_vbias_bottom", 33e3, "ohm", "R8: VBIAS divider to GND (bottom)")
        c_vb = _c(b, "tx.c_vbias", 10e-6, "F", "C4: VBIAS decoupling electrolytic")
        c_dev = _c(b, "tx.c_dev", 4.7e-6, "F", "C3: the low-pass output's coupling into the deviation trimmer")
        r_trim = _c(b, "tx.trim_r", 10e3, "ohm", "RV1: the deviation trimmer's track (a divider from the coupled audio to GND, set in the lab)")
        test = _c(b, "tx.test_level", 0.01, "V", "the deck's microphone signal: a 10 mV peak sine at 1 kHz (an electret at normal speech [UNVERIFIED])")
        tol_vb = _c(b, "tx.vbias_tol", 0.02, None, "the varactor bias row: within 2 % of the divider (ideal resistors; no DC flows into VAR)")
        v_bias = _k(b, "tx.v_bias", lambda: voltage_divider_output(v5, r_b1, r_b2, ("bb.v5", "tx.r_vbias_top", "tx.r_vbias_bottom")))
        _vref(b, nb, refs=("R1", "R2", "C1"), net="VREF_T", rail="V_TX", rail_key="bb.v_rail", rail_v=v_rail, eid="vref_t")
        mk = b.part("mic", "MK1", "Mic", "electret microphone (pin 2 '+' to the bias resistor)")
        _exclude(b, "MK1", "microphone: no model (MIC_P is the stimulus VMIC)")
        _add(nb, "MIC_P", mk.at("+"))
        _add(nb, GROUND_NET, mk.at("-"))
        _p(b, nb, "bb_res", "R3", mic_r, "electret bias resistor", "V5T", "MIC_P")
        key, nom = _gain_stage(b, nb, prefix="tx.amp", refs={"u": "U1", "c_in": "C5", "r_bias": "R4", "r_f": "R9", "c_f": "C6", "r_g": "R10", "c_g": "C7",
                                                              "c_dec": "C8", "d1": "D1", "d2": "D2"},
                               nets={"in": "MIC_P", "plus": "TXP", "minus": "TXN", "out": "TXA", "g": "TXG"}, vref="VREF_T", rail="V_TX", f_ref=f_ref,
                               what="microphone amplifier (TL071, soft limiter)", limiter=True, serves=())
        _sallen_key(b, nb, prefix="tx.sk", refs={"u": "U2", "r1": "R11", "r2": "R12", "c1": "C9", "c2": "C10", "c_dec": "C11"},
                    nets={"in": "TXA", "n1": "TXS1", "n2": "TXS2", "out": "TXF"}, rail="V_TX", bw=bw, what="TX low-pass (Sallen-Key)")
        _p(b, nb, "bb_cap_elec", "C3", c_dev, "coupling into the deviation trimmer", "TXF", "DEV_TOP")
        rv = b.part("bb_trimpot", "RV1", format_part_value(float(r_trim.value)), "deviation trimmer")
        pos = model_once(b, "model.pot.position")
        card_bind(b, "RV1", models.potentiometer_card(r_trim, pos, key="model.pot.dev", name="POTDEV"))
        _add(nb, "DEV_TOP", rv.at("END3"))
        _add(nb, GROUND_NET, rv.at("END1"))
        _add(nb, "DEV_W", rv.at("WIPER"))
        _p(b, nb, "bb_cap_elec", "C12", c_wip, "deviation coupling (+ on the biased side)", "DEV_C", "DEV_W")
        _p(b, nb, "bb_res", "R5", r_dev, "deviation resistor into the varactor node", "DEV_C", "VAR")
        _p(b, nb, "bb_res", "R6", r_vb, "varactor bias", "VBIAS", "VAR")
        _p(b, nb, "bb_res", "R7", r_b1, "VBIAS divider (top)", "V5T", "VBIAS")
        _p(b, nb, "bb_res", "R8", r_b2, "VBIAS divider (bottom)", "VBIAS", GROUND_NET)
        _p(b, nb, "bb_cap_elec", "C4", c_vb, "VBIAS decoupling", "VBIAS", GROUND_NET)
        nb.declare(b)
        zero = Traced(value=0.0, unit="V", provenance=test.provenance)
        b.result.stimuli.append(Stimulus(id="VMIC", source="voltage", net="MIC_P", reference_net="GND", kind=StimulusKind.SINE,
                                         params={"vo": zero, "va": test, "freq": f_ref, "ac": ac},
                                         provenance=ctx.provenance("MIC_P driven by the test tone (the electret is not simulated)")))
        note = "TX microphone amplifier: 1 + R_f / R_g with the C_f low-pass and the C_g / C_in high-passes (product of first-order terms)"
        db_row(b, "tx_gain_1k", AC_REF, "v(TXA)", "v(MIC_P)", f_ref, nom, note, tol_db=tol)
        bw_n = _k(b, "tx.sk_bw", lambda: butterworth_db_at(_ref(b, "audio.sk_order"), bw, bw, ("audio.sk_order", "audio.bandwidth", "audio.bandwidth")))
        bw2_n = _k(b, "tx.sk_2bw", lambda: butterworth_db_at(_ref(b, "audio.sk_order"), f_2bw, bw, ("audio.sk_order", "audio.f_2bw", "audio.bandwidth")))
        db_row(b, "tx_lpf_bw", AC_BW, "v(TXF)", "v(TXA)", bw, bw_n, "TX Sallen-Key low-pass at its corner: -3.01 dB (2nd-order Butterworth)", tol_db=tol)
        db_row(b, "tx_lpf_2bw", AC_2BW, "v(TXF)", "v(TXA)", f_2bw, bw2_n, "TX Sallen-Key low-pass at twice its corner (2nd-order Butterworth)", tol_db=tol)
        expectation(b, "tx_var_bias", op_analysis(b), "v(VAR)", Reduce.VALUE, v_bias, "the varactor's reverse bias: VBIAS through R6 (C12 blocks the wiper's DC)",
                    tol_rel=tol_vb)
        b.result.chain = signal_order(b)
        b.result.lab_items = [LabItem(id="tx_audio_level", block=self.id, what="the microphone's level at TXA and the soft limiter's clip level at loud speech",
                                      instruments=["oscilloscope"], reason="the electret is not simulated and the limiter is checked only below its knee")]
        return b.done()


# --------------------------------------------------------------------------- TX RF


def _stage(b: BlockBuilder, nb: NetBook, *, key: str, ref_q: str, value: str, refs: dict[str, str], nets: dict[str, str], vals: dict[str, Traced],
           what: str, serves: Iterable[str]) -> tuple[PlacedPart, list[tuple[str, str]], SpiceBinding]:
    """A divider-biased stage: the transistor (excluded), R_b1 / R_b2, R_E // C_E, the feed resistor + decoupling and the collector choke.

    Returns the transistor, the collector pins (its C and the choke's far end) and the choke's binding (a fixture binding of the stage's
    network: the deck drops the choke as a dead branch once the transistor is excluded); the feed net is ``nets['feed']`` behind ``refs['r_feed']``
    from ``V_TX``. Every passive is bound in the deck at its value; the transistor is excluded (no model), so exclude_floating drops the dead
    branches it leaves.
    """
    q = b.part(key, ref_q, value, what, serves)
    _exclude(b, ref_q, NO_RF_MODEL)
    _p(b, nb, "bb_res", refs["r_feed"], vals["r_feed"], f"{what}: supply feed resistor (decouples the stage from V_TX)", "V_TX", nets["feed"])
    _p(b, nb, "bb_cap", refs["c_dec"], vals["c_dec"], f"{what}: feed decoupling", nets["feed"], GROUND_NET)
    _p(b, nb, "bb_res", refs["r_b1"], vals["r_b1"], f"{what}: upper base-divider resistor", nets["feed"], nets["base"])
    _p(b, nb, "bb_res", refs["r_b2"], vals["r_b2"], f"{what}: lower base-divider resistor", nets["base"], GROUND_NET)
    _p(b, nb, "bb_res", refs["r_e"], vals["r_e"], f"{what}: emitter resistor", nets["emitter"], GROUND_NET)
    _p(b, nb, "bb_cap", refs["c_e"], vals["c_e"], f"{what}: emitter bypass", nets["emitter"], GROUND_NET)
    choke = b.part("bb_coil", refs["choke"], format_part_value(float(vals["l_choke"].value)), f"{what}: collector feed choke")
    choke_binding = _bind_value(b, choke, vals["l_choke"], "at its value")
    b.bind(refs["choke"], choke_binding)
    _add(nb, nets["feed"], [(refs["choke"], choke.pin("1"))])
    _add(nb, nets["base"], q.at("B"))
    _add(nb, nets["emitter"], q.at("E"))
    return q, [*q.at("C"), (refs["choke"], choke.pin("2"))], choke_binding


class BbTxRfBlock(Block):
    """VXO (crystal f_c / 12 + varactor) -> BF199 x3 -> BFR92A x2 -> BFR92A x2 -> band-pass -> BFR92A PA -> L-match -> low-pass -> ``TX_OUT``."""

    id = "tx_rf"
    title = "TX RF: crystal VXO with varactor FM, x3 / x2 / x2 multipliers, 3-resonator band-pass, class-A PA, L-match, 5th-order low-pass"
    interface_nets = ("V_TX", "V5T", "VAR", "TX_OUT")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        carrier = ctx.inputs.get("carrier_frequency")
        if carrier is None:
            raise TemplateRefusal(f"block {self.id} needs the confirmed carrier_frequency")
        serves_fc = [carrier.requirement.id]
        f_c = input_copy(b, "carrier_frequency", "rf.f_c")
        v_rail, v5 = _rails(b)
        z0 = _c(b, "rf.z0", 50.0, "ohm", "system impedance of the antenna port, the low-pass and the T/R network")
        n = _c(b, "rf.n_mult", float(N_MULT), None, "the TX multiplication factor N = 12 = 3 x 2 x 2: the crystal runs at f_c / 12 (36.000 MHz gives 432.000 MHz)")
        mults = [(f"tx.mult.{k}", _c(b, f"tx.mult.{k}", float(m), None, f"TX multiplier stage {k}: x{m}")) for k, m in enumerate(STAGE_MULTIPLIERS, start=1)]
        f_x = _k(b, "tx.f_xtal", lambda: clock_divided(f_c, n, ("rf.f_c", "rf.n_mult")))
        stages: list[tuple[str, Traced]] = []
        prev = ("tx.f_xtal", f_x)
        for k, (mkey, m) in enumerate(mults, start=1):
            key = f"tx.f{k}"
            prev = (key, _k(b, key, lambda prev=prev, m=m, mkey=mkey: radio.mult_stage(prev[1], m, (prev[0], mkey))))
            stages.append(prev)
        if abs(float(stages[-1][1].value) - float(f_c.value)) > 1e-9 * float(f_c.value):
            raise TemplateRefusal(f"the TX stages {[float(m.value) for _, m in mults]} do not multiply f_c / N back to f_c")
        q_key = "model.bb.coil_q"
        q_u = model_once(b, q_key)
        r_out = ("model.bfr92.r_out", model_once(b, "model.bfr92.r_out"))
        r_in = ("model.bfr92.r_in", model_once(b, "model.bfr92.r_in"))
        r_out199 = ("model.bf199.r_out", model_once(b, "model.bf199.r_out"))
        tol = _c(b, "tx.net_tol", 1.0, "dB", "tolerance of every exact-network fixture row of the TX chain (the netlist realises the designed network)")
        lin = _c(b, "tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures")
        pts = _c(b, "tx.sweep.points", 401, None, "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)")
        idx = index_choices(b, "tx", 3)
        o_m = ("tx.order.m1", _c(b, "tx.order.m1", -1.0, None, "order -1: the lower neighbour f - f_X"))
        o_p = ("tx.order.p1", _c(b, "tx.order.p1", 1.0, None, "order +1: the upper neighbour f + f_X"))
        tx_port = RFPort(name="v_tx", net="V_TX", kind="rail", voltage_v=v_rail, direction="in")
        # ---- VXO
        vxo = {
            "r_b": _c(b, "vxo.r_b", 22e3, "ohm", "R1 / R2: the VXO's base divider from V5T (V_B = 2.5 V)"),
            "c_fb": _c(b, "vxo.c_fb", 68e-12, "F", "C2 / C3: the Colpitts divider (base-emitter and emitter-ground); ngspice-42 with a generic 2N3904 "
                                                    "card shows about -155 ohm at the base at 36 MHz against the crystal's about 50 ohm in this circuit "
                                                    "[UNVERIFIED: the real oscillator's start-up is a lab item]"),
            "r_e": _c(b, "vxo.r_e", 1e3, "ohm", "R3: the VXO's emitter resistor (about 1.7 mA)"),
            "c_out": _c(b, "vxo.c_out", 22e-12, "F", "C5: the VXO's light coupling from its emitter to the tripler's base"),
            "c_trim": _c(b, "vxo.c_trim", 10e-12, "F", "CT1: the frequency trimmer across the varactor (sets the centre frequency; with the varactor the crystal's load)"),
        }
        y1 = b.part("bb_xtal", "Y1", f"{float(f_x.value) / 1e6:.6g}M", f"VXO crystal at f_c / 12 = {float(f_x.value) / 1e6:.7g} MHz (fundamental mode)", serves_fc)
        _exclude(b, "Y1", NO_RF_MODEL)
        d1 = b.part("bb_varactor", "D1", "BB910", "VXO varactor in series with the crystal (direct FM)", serving(ctx, "modulation", "radio_build"))
        _exclude(b, "D1", NO_RF_MODEL)
        q1 = b.part("bb_npn_osc", "Q1", "2N3904", "VXO transistor (Colpitts, crystal from base to the varactor)", serves_fc)
        _exclude(b, "Q1", NO_RF_MODEL)
        _add(nb, "XB", [*y1.at("1"), *q1.at("B")], "the VXO's base: the crystal's hot end")
        _add(nb, "VAR", [*y1.at("2"), *d1.at("K")], "the varactor node: the crystal's cold end, the audio and the bias")
        _add(nb, GROUND_NET, d1.at("A"))
        _add(nb, "V5T", q1.at("C"))
        _add(nb, "XE", q1.at("E"), "the VXO's emitter: its output")
        _p(b, nb, "bb_ctrim", "CT1", vxo["c_trim"], "VXO frequency trimmer (across the varactor)", "VAR", GROUND_NET, deck=False, network="lab")
        _p(b, nb, "bb_res", "R1", vxo["r_b"], "VXO base divider (top)", "V5T", "XB")
        _p(b, nb, "bb_res", "R2", vxo["r_b"], "VXO base divider (bottom)", "XB", GROUND_NET)
        _p(b, nb, "bb_cap", "C2", vxo["c_fb"], "Colpitts base-emitter capacitor", "XB", "XE", deck=False, network="lab")
        _p(b, nb, "bb_cap", "C3", vxo["c_fb"], "Colpitts emitter-ground capacitor", "XE", GROUND_NET, deck=False, network="lab")
        _p(b, nb, "bb_res", "R3", vxo["r_e"], "VXO emitter resistor", "XE", GROUND_NET)
        _p(b, nb, "bb_cap", "C4", _c(b, "vxo.c_dec", 100e-9, "F", "C4: V5T decoupling at the VXO"), "V5T decoupling at the VXO", "V5T", GROUND_NET)
        _p(b, nb, "bb_cap", "C5", vxo["c_out"], "VXO output coupling to the tripler", "XE", "TX_X3_B", deck=False, network="lab")
        # ---- multiplier stages and their networks
        common = {"r_feed": _c(b, "tx.r_feed", 47.0, "ohm", "every RF stage's supply feed resistor from V_TX (with its 1 nF decoupling: an RC between stages)"),
                  "c_dec": _c(b, "tx.c_dec", 1e-9, "F", "every RF stage's feed decoupling capacitor"),
                  "c_e": _c(b, "tx.c_e", 1e-9, "F", "every RF stage's emitter bypass capacitor")}
        stage_defs = (
            ("x3", "bb_npn_x3", "BF199", 47e3, 4.7e3, 220.0, 470e-9, r_out199),
            ("x6", "bb_npn_rf", "BFR92A", 47e3, 4.7e3, 220.0, 220e-9, r_out),
            ("x12", "bb_npn_rf", "BFR92A", 47e3, 4.7e3, 330.0, 100e-9, r_out),
        )
        stage_refs = (
            {"r_b1": "R4", "r_b2": "R5", "r_e": "R6", "c_e": "C6", "r_feed": "R7", "c_dec": "C7", "choke": "L1"},
            {"r_b1": "R8", "r_b2": "R9", "r_e": "R10", "c_e": "C8", "r_feed": "R11", "c_dec": "C9", "choke": "L3"},
            {"r_b1": "R12", "r_b2": "R13", "r_e": "R14", "c_e": "C10", "r_feed": "R15", "c_dec": "C11", "choke": "L5"},
        )
        vals_by: list[dict[str, Traced]] = []
        for k, (name, _key, _v, rb1, rb2, re_, lch, _port) in enumerate(stage_defs):
            f_k = stages[k][1]
            what = f"multiplier stage {k + 1} (x{int(float(mults[k][1].value))} to {float(f_k.value) / 1e6:.6g} MHz)"
            vals_by.append({
                **common,
                "r_b1": _c(b, f"tx.{name}.r_b1", rb1, "ohm", f"{what}: upper base-divider resistor (a slight forward bias: class AB / C by the drive)"),
                "r_b2": _c(b, f"tx.{name}.r_b2", rb2, "ohm", f"{what}: lower base-divider resistor"),
                "r_e": _c(b, f"tx.{name}.r_e", re_, "ohm", f"{what}: emitter resistor"),
                "l_choke": _c(b, f"tx.{name}.l_choke", lch, "H", f"{what}: collector feed choke (part of the stage's network: a fixture member) "
                                                                    "[UNVERIFIED: its self-resonance; a hand-wound air coil]"),
            })
        nets_of = lambda name: {"feed": f"TX_{name.upper()}_VC", "base": f"TX_{name.upper()}_B", "emitter": f"TX_{name.upper()}_E"}  # noqa: E731
        nb_base_extra: list[tuple[str, str]] = []
        tank_l = (150e-9, 68e-9)
        tank_bw = (8e6, 14e6)
        for k, (name, key, value, _rb1, _rb2, _re, _lch, port_r) in enumerate(stage_defs):
            fk_key, f_k = stages[k]
            vals = vals_by[k]
            what = f"multiplier stage {k + 1} ({value}, x{int(float(mults[k][1].value))} to {float(f_k.value) / 1e6:.6g} MHz)"
            refs = stage_refs[k]
            nets = nets_of(name)
            q, collector, choke_binding = _stage(b, nb, key=key, ref_q=f"Q{k + 2}", value=value, refs=refs, nets=nets, vals=vals, what=what, serves=serves_fc)
            coll_net = f"TX_{name.upper()}_C"
            if k < 2:
                nxt = vals_by[k + 1]
                nxt_name = stage_defs[k + 1][0]
                div = _k(b, f"tx.{nxt_name}.r_div", lambda nxt=nxt, nxt_name=nxt_name: parallel_resistance(nxt["r_b1"], nxt["r_b2"], (f"tx.{nxt_name}.r_b1", f"tx.{nxt_name}.r_b2")))
                r_eff = (f"tx.tank{k + 1}.r_load_eff", _k(b, f"tx.tank{k + 1}.r_load_eff", lambda div=div, nxt_name=nxt_name: parallel_resistance(
                    r_in[1], div, (r_in[0], f"tx.{nxt_name}.r_div"))))
                port = PortLoad(l_port=(f"tx.{name}.l_choke", vals["l_choke"]), q_port=(q_key, q_u), r_load_eff=r_eff)
                net_id = f"tx_tank{k + 1}"
                n_t = (f"tx.tank.n", _c(b, "tx.tank.n", 1.0, None, "resonators per multiplier tank (single-tuned: one coil and one trimmer per stage)"))
                l_k = (f"tx.tank{k + 1}.l", _c(b, f"tx.tank{k + 1}.l", tank_l[k], "H", f"{net_id}: resonator coil (hand-wound; R_p = Q_e w0 L must exceed "
                                                                                          "the collector port's resistance: a tap only transforms down)"))
                bw_k = (f"tx.tank{k + 1}.bw", _c(b, f"tx.tank{k + 1}.bw", tank_bw[k], "Hz", f"{net_id}: bandwidth of the single-tuned tank"))
                top = build_top_c(b, net_id, what=f"{net_id} (single-tuned tank at {float(f_k.value) / 1e6:.6g} MHz)", n=n_t, f0=(fk_key, f_k), bw=bw_k, l=l_k,
                                  r_source=port_r, r_load=r_in, idx=idx, cap_refs=[f"C{12 + 3 * k}", f"CT{2 + k}", f"C{13 + 3 * k}"], ind_refs=[f"L{2 + 2 * k}"],
                                  nodes=[f"TX_T{k + 1}"], ind_part="bb_coil", cap_part="bb_cap", shunt_part="bb_ctrim", netbook=nb, port=port)
                f_m = (f"{net_id}.f_m", _k(b, f"{net_id}.f_m", lambda f_k=f_k, fk_key=fk_key: radio.mult_spur(f_k, f_x, o_m[1], (fk_key, "tx.f_xtal", o_m[0]))))
                f_p = (f"{net_id}.f_p", _k(b, f"{net_id}.f_p", lambda f_k=f_k, fk_key=fk_key: radio.mult_spur(f_k, f_x, o_p[1], (fk_key, "tx.f_xtal", o_p[0]))))
                s21 = _k(b, f"{net_id}.s21", lambda top=top, f_k=f_k, fk_key=fk_key: top.s21(q_u, q_key, f_k, fk_key))
                rel_m = _k(b, f"{net_id}.rel_m", lambda top=top, f_m=f_m, f_k=f_k, fk_key=fk_key: top.rel(q_u, q_key, f_m[1], f_m[0], f_k, fk_key))
                rel_p = _k(b, f"{net_id}.rel_p", lambda top=top, f_p=f_p, f_k=f_k, fk_key=fk_key: top.rel(q_u, q_key, f_p[1], f_p[0], f_k, fk_key))
                f_lo = _c(b, f"tx.tank{k + 1}.sweep.fstart", round(float(f_k.value) * 0.5 / 1e6) * 1e6, "Hz", f"start of {net_id}'s sweep (about f / 2)")
                f_hi = _c(b, f"tx.tank{k + 1}.sweep.fstop", round(float(f_k.value) * 1.5 / 1e6) * 1e6, "Hz", f"stop of {net_id}'s sweep (about 1.5 f)")
                out_net = f"TX_{nxt_name.upper()}_B"
                nxt_refs = stage_refs[k + 1]
                b.result.networks.append(RFNetwork(
                    id=net_id, block=self.id, members=[*top.members, refs["choke"], refs["r_feed"], refs["c_dec"], nxt_refs["r_b1"], nxt_refs["r_b2"]],
                    bindings={**top.bindings, refs["choke"]: choke_binding}, loss_q={**{r: q_u for r in top.inductors}, refs["choke"]: q_u}, q_ref_hz=f_k,
                    ports=[fixture_port("coll", coll_net, port_r[1], "in"), fixture_port("next", out_net, r_in[1], "out"), tx_port],
                    sweep=[ac_sweep(b, f"{net_id}_sweep", lin, pts, f_lo, f_hi, f"{net_id}: f / 2 .. 1.5 f")],
                    expectations=[row("s21", "s21_db", "coll", "next", f_k, s21, tol=tol),
                                  row("rel_m", "rel_s21_db", "coll", "next", f_m[1], rel_m, tol=tol, ref_at=f_k),
                                  row("rel_p", "rel_s21_db", "coll", "next", f_p[1], rel_p, tol=tol, ref_at=f_k)],
                ))
            else:
                pa_vals = self._pa_values(b, v_rail)
                port = PortLoad(l_port=(f"tx.{name}.l_choke", vals["l_choke"]), q_port=(q_key, q_u), r_load_eff=pa_vals["r_load_eff"])
                n_b = ("tx.bpf.n", _c(b, "tx.bpf.n", 3.0, None, "resonators of the x12 stage's band-pass at the carrier (3: f_c -/+ f_X about 36-43 dB down "
                                                                 "relative under Q 80; two gave about 24-28 dB)"))
                bw_b = ("tx.bpf.bw", _c(b, "tx.bpf.bw", 12e6, "Hz", "band-pass bandwidth at the carrier (12 MHz: narrow enough for f_c -/+ f_X, wide "
                                                                     "enough for the coil Q)"))
                l_b = ("tx.bpf.l", _c(b, "tx.bpf.l", 15e-9, "H", ("band-pass resonator coil (about 2 turns, 4 mm diameter, hand-wound [UNVERIFIED]); "
                                                                  "R_p = Q_e w0 L must exceed the collector port's resistance")))
                top = build_top_c(b, "tx_bpf", what="tx_bpf (the x12 stage's 3-resonator band-pass at f_c)", n=n_b, f0=("rf.f_c", f_c), bw=bw_b, l=l_b,
                                  r_source=port_r, r_load=pa_vals["r_in"], idx=idx,
                                  cap_refs=["C18", "CT4", "C19", "CT5", "C20", "CT6", "C21"], ind_refs=["L6", "L7", "L8"],
                                  nodes=["TX_B1", "TX_B2", "TX_B3"], ind_part="bb_coil", cap_part="bb_cap", shunt_part="bb_ctrim", netbook=nb, port=port)
                f_m = ("tx_bpf.f_m", _k(b, "tx_bpf.f_m", lambda: radio.mult_spur(f_c, f_x, o_m[1], ("rf.f_c", "tx.f_xtal", o_m[0]))))
                f_p = ("tx_bpf.f_p", _k(b, "tx_bpf.f_p", lambda: radio.mult_spur(f_c, f_x, o_p[1], ("rf.f_c", "tx.f_xtal", o_p[0]))))
                s21 = _k(b, "tx_bpf.s21", lambda top=top: top.s21(q_u, q_key, f_c, "rf.f_c"))
                rel_m = _k(b, "tx_bpf.rel_m", lambda top=top: top.rel(q_u, q_key, f_m[1], f_m[0], f_c, "rf.f_c"))
                rel_p = _k(b, "tx_bpf.rel_p", lambda top=top: top.rel(q_u, q_key, f_p[1], f_p[0], f_c, "rf.f_c"))
                b.result.networks.append(RFNetwork(
                    id="tx_bpf", block=self.id, members=[*top.members, refs["choke"], refs["r_feed"], refs["c_dec"], "R16", "R17"],
                    bindings={**top.bindings, refs["choke"]: choke_binding},
                    loss_q={**{r: q_u for r in top.inductors}, refs["choke"]: q_u}, q_ref_hz=f_c,
                    ports=[fixture_port("coll", coll_net, port_r[1], "in"), fixture_port("pa", "PA_B", r_in[1], "out"), tx_port],
                    sweep=[ac_sweep(b, "tx_bpf_sweep", lin, pts, _c(b, "tx.bpf.sweep.fstart", 300e6, "Hz", "start of the band-pass sweep"),
                                    _c(b, "tx.bpf.sweep.fstop", 600e6, "Hz", "stop of the band-pass sweep"), "tx_bpf: 300-600 MHz")],
                    expectations=[row("s21_fc", "s21_db", "coll", "pa", f_c, s21, tol=tol),
                                  row("rel_m1", "rel_s21_db", "coll", "pa", f_m[1], rel_m, tol=tol, ref_at=f_c),
                                  row("rel_p1", "rel_s21_db", "coll", "pa", f_p[1], rel_p, tol=tol, ref_at=f_c)],
                ))
                for pid, f, note in (("tx_spur_m1", f_m[1], "f_c - f_X: the closest multiplier spur below the carrier"),
                                     ("tx_spur_p1", f_p[1], "f_c + f_X: the closest multiplier spur above the carrier (the weak side of a top-C network)")):
                    b.result.plan_lines.append(PlanLine(id=pid, kind="margin", f_hz=f, ref_hz=f_c, min_margin_hz=bw_b[1],
                                                        note=f"{note}: outside the band-pass's passband; its rejection there is a tx_bpf row, the level at the "
                                                             "antenna the lab item tx_spurious"))
            _add(nb, coll_net, [*collector, *top.in_pins], f"stage {k + 1} collector: the choke and the network's input tap")
            _add(nb, GROUND_NET, top.gnd_pins)
            nb_base_extra = top.out_pins
            if k < 2:
                _add(nb, f"TX_{stage_defs[k + 1][0].upper()}_B", nb_base_extra)
        self._pa(b, nb, f_c, z0, q_u, q_key, lin, pts, tx_port, pa_extra=nb_base_extra, serves=serves_fc)
        nb.declare(b)
        b.result.chain = signal_order(b)
        b.result.ports = [RFPort(name="TX_OUT", net="TX_OUT", kind="port", z0_ohm=z0, frequency_hz=f_c, direction="out"),
                          RFPort(name="V_TX", net="V_TX", kind="rail", voltage_v=v_rail, direction="in"),
                          RFPort(name="V5T", net="V5T", kind="rail", voltage_v=v5, direction="in")]
        b.result.lab_items = [
            LabItem(id="vxo", block=self.id, what="the VXO's start-up, its frequency f_c / 12 set by CT1 and its deviation set by RV1 (a calibrated receiver or "
                                                  "a frequency counter at the crystal, x12 at the carrier)", instruments=["frequency counter", "FM receiver"],
                    reason="the crystal, the varactor and the 2N3904 are not simulated"),
            LabItem(id="tx_tank_alignment", block=self.id, what="peaking each multiplier tank (CT2, CT3) and the band-pass (CT4-CT6) on its harmonic, the "
                                                               "band-pass's 0.2 pF 'gimmick' couplings included", instruments=["RF probe", "spectrum analyser or RTL-SDR"],
                    reason="hand-wound coils and perfboard strays detune the designed networks by more than the trimmers' design point"),
            LabItem(id="tx_power", block=self.id, what="the PA's output into a 50 ohm dummy load (about 20-30 mW designed) and the BFR92A's temperature",
                    instruments=["RF power meter or diode probe", "50 ohm dummy load"], reason="the PA transistor is not simulated"),
            LabItem(id="tx_spurious", block=self.id, what="the multiplier spurs (f_c -/+ f_X, -/+ 2 f_X), the harmonics and the residues at the antenna port, "
                                                         "against the amateur licence's spurious limits, before any antenna is connected",
                    instruments=["spectrum analyser"], reason="spur generation in class-C stages is not modelled; only the networks' rejection is a fixture row"),
        ]
        return b.done()

    @staticmethod
    def _pa_values(b: BlockBuilder, v_rail: Traced) -> dict[str, Traced | tuple[str, Traced]]:
        """The PA's bias choices and the band-pass's effective load (the PA base port model in parallel with its divider)."""
        r_in = ("model.bfr92.r_in", model_once(b, "model.bfr92.r_in"))
        rb1 = _c(b, "pa.r_b1", 3.3e3, "ohm", "R16: the PA's upper base divider (class A, about 16 mA with R18)")
        rb2 = _c(b, "pa.r_b2", 1.2e3, "ohm", "R17: the PA's lower base divider")
        div = _k(b, "pa.r_div", lambda: parallel_resistance(rb1, rb2, ("pa.r_b1", "pa.r_b2")))
        eff = ("pa.r_load_eff", _k(b, "pa.r_load_eff", lambda: parallel_resistance(r_in[1], div, (r_in[0], "pa.r_div"))))
        return {"r_in": r_in, "r_load_eff": eff, "r_b1": rb1, "r_b2": rb2}

    def _pa(self, b: BlockBuilder, nb: NetBook, f_c: Traced, z0: Traced, q_u: Traced, q_key: str, lin: Traced, pts: Traced, tx_port: RFPort, *,
            pa_extra: list[tuple[str, str]], serves: list[str]) -> None:
        """The BFR92A class-A PA (Q5), its low-pass L-match ``pa_match`` and the 5th-order Chebyshev low-pass ``lpf`` into ``TX_OUT``."""
        v = self._pa_values(b, tx_port.voltage_v)  # type: ignore[arg-type]
        re_ = _c(b, "pa.r_e", 100.0, "ohm", "R18: the PA's emitter resistor (I_C about 16 mA: (2.4 V - 0.8 V) / 100 ohm)")
        l_ch = _c(b, "pa.l_choke", 470e-9, "H", "L9: the PA's collector feed choke (hand-wound, a pa_match fixture member) [UNVERIFIED: its self-resonance]")
        v_swing = _c(b, "pa.v_cc_eff", 7.0, "V", "the PA's collector swing headroom: V_TX less the emitter's 1.6 V and the feed resistor's drop")
        v_sat = _c(b, "pa.v_sat", 0.5, "V", "the BFR92A's saturation margin at the swing's bottom [UNVERIFIED: NXP BFR92A datasheet]")
        p_out = _c(b, "pa.p_out", 0.03, "W", "the PA's design output: 30 mW (class A at about 16 mA can give at most about 55 mW)")
        r_l = _k(b, "pa.r_load_line", lambda: radio.pa_load_line_r(v_swing, v_sat, p_out, ("pa.v_cc_eff", "pa.v_sat", "pa.p_out")))
        c_m = _k(b, "pa.match.c_shunt", lambda: lmatch_lowpass_c_shunt(f_c, r_l, z0, ("rf.f_c", "pa.r_load_line", "rf.z0")))
        l_m = _k(b, "pa.match.l_series", lambda: lmatch_lowpass_l_series(f_c, r_l, z0, ("rf.f_c", "pa.r_load_line", "rf.z0")))
        c_dc = _c(b, "pa.c_dc", 100e-12, "F", "C24: the DC block between the match and the low-pass")
        s21_min = _c(b, "pa.match.s21_min", -1.0, "dB", "pa_match s21 at f_c at least -1 dB (with the choke and Q-80 coils)")
        s11_max = _c(b, "pa.match.s11_max", -10.0, "dB", "pa_match s11 at f_c at most -10 dB (the choke shifts the load line slightly)")
        q5 = b.part("bb_npn_rf", "Q5", "BFR92A", "class-A PA (about 30 mW designed)", serves)
        _exclude(b, "Q5", NO_RF_MODEL)
        common = {"r_feed": _ref(b, "tx.r_feed"), "c_dec": _ref(b, "tx.c_dec"), "c_e": _ref(b, "tx.c_e")}
        _p(b, nb, "bb_res", "R19", common["r_feed"], "PA supply feed resistor", "V_TX", "PA_VC")
        _p(b, nb, "bb_cap", "C22", common["c_dec"], "PA feed decoupling", "PA_VC", GROUND_NET)
        _p(b, nb, "bb_res", "R16", v["r_b1"], "PA upper base divider", "PA_VC", "PA_B")  # type: ignore[arg-type]
        _p(b, nb, "bb_res", "R17", v["r_b2"], "PA lower base divider", "PA_B", GROUND_NET)  # type: ignore[arg-type]
        _p(b, nb, "bb_res", "R18", re_, "PA emitter resistor", "PA_E", GROUND_NET)
        _p(b, nb, "bb_cap", "C28", common["c_e"], "PA emitter bypass", "PA_E", GROUND_NET)
        _add(nb, "PA_B", [*q5.at("B"), *pa_extra])
        _add(nb, "PA_E", q5.at("E"))
        b_l9 = _p(b, nb, "bb_coil", "L9", l_ch, "PA collector feed choke", "PA_VC", "PA_C")
        _add(nb, "PA_C", q5.at("C"))
        b_c23 = _p(b, nb, "bb_cap", "C23", c_m, "pa_match: shunt capacitor at the load line", "PA_C", GROUND_NET, deck=False, network="pa_match")
        b_l10 = _p(b, nb, "bb_coil", "L10", l_m, "pa_match: series coil to the system impedance", "PA_C", "PA_M", deck=False, network="pa_match")
        b_c24 = _p(b, nb, "bb_cap", "C24", c_dc, "DC block into the low-pass", "PA_M", "LPF_IN", deck=False, network="pa_match")
        b.result.networks.append(RFNetwork(
            id="pa_match", block=self.id, members=["C23", "L10", "C24", "L9", "R19", "C22"], bindings={"C23": b_c23, "L10": b_l10, "C24": b_c24, "L9": b_l9},
            loss_q={"L10": q_u, "L9": q_u}, q_ref_hz=f_c,
            ports=[fixture_port("pa_out", "PA_C", r_l, "in"), fixture_port("lpf_in", "LPF_IN", z0, "out"), tx_port],
            sweep=[ac_sweep(b, "pa_match_sweep", lin, pts, _ref(b, "tx.bpf.sweep.fstart"), _ref(b, "tx.bpf.sweep.fstop"), "pa_match: 300-600 MHz")],
            expectations=[row("s21_fc", "s21_db", "pa_out", "lpf_in", f_c, s21_min, bound="at_least"),
                          row("s11_fc", "s11_db", "pa_out", "pa_out", f_c, s11_max, bound="at_most")],
        ))
        # the 5th-order 0.1 dB Chebyshev low-pass, shunt C first
        order = _c(b, "lpf.n", 5.0, None, "harmonic low-pass order (5th: shunt C first; equal terminations need an odd order)")
        rip = _c(b, "lpf.ripple", 0.1, "dB", "harmonic low-pass passband ripple (Chebyshev 0.1 dB)")
        f_e = _c(b, "lpf.f_edge", 480e6, "Hz", "harmonic low-pass ripple edge (480 MHz: the 430-440 MHz band in the passband, 2 f_c far in the stopband)")
        l_s21 = _c(b, "lpf.s21_min", -1.5, "dB", "lpf s21 at f_c at least -1.5 dB")
        l_s11 = _c(b, "lpf.s11_max", -15.0, "dB", "lpf s11 at f_c at most -15 dB")
        h2 = _c(b, "lpf.h2_max", -27.0, "dB", "lpf s21 at 2 f_c at most -27 dB (ngspice-42 reads about -29.5 dB with Q-80 coils at 432 MHz)")
        h3 = _c(b, "lpf.h3_max", -46.0, "dB", "lpf s21 at 3 f_c at most -46 dB (ngspice-42 reads about -49.3 dB)")
        k2 = _c(b, "lpf.k2", 2.0, None, "harmonic number 2")
        k3 = _c(b, "lpf.k3", 3.0, None, "harmonic number 3")
        f2 = _k(b, "lpf.f_2", lambda: radio.harmonic(f_c, k2, ("rf.f_c", "lpf.k2")))
        f3 = _k(b, "lpf.f_3", lambda: radio.harmonic(f_c, k3, ("rf.f_c", "lpf.k3")))
        lidx = index_choices(b, "lpf", 5)
        values = []
        for k in range(1, 6):
            g = _k(b, f"lpf.g.{k}", lambda k=k: chebyshev_g(order, lidx[k][1], rip, ("lpf.n", lidx[k][0], "lpf.ripple")))
            if k % 2:
                values.append(_k(b, f"lpf.c.{k}", lambda k=k, g=g: lpf_shunt_c(g, f_e, z0, (f"lpf.g.{k}", "lpf.f_edge", "rf.z0"))))
            else:
                values.append(_k(b, f"lpf.l.{k}", lambda k=k, g=g: lpf_series_l(g, f_e, z0, (f"lpf.g.{k}", "lpf.f_edge", "rf.z0"))))
        refs = ["C25", "L11", "C26", "L12", "C27"]
        bindings: dict[str, SpiceBinding] = {}
        bindings["C25"] = _p(b, nb, "bb_cap", "C25", values[0], "lpf element 1 (shunt C)", "LPF_IN", GROUND_NET, deck=False, network="lpf")
        bindings["L11"] = _p(b, nb, "bb_coil", "L11", values[1], "lpf element 2 (series L)", "LPF_IN", "LPF_N1", deck=False, network="lpf")
        bindings["C26"] = _p(b, nb, "bb_cap", "C26", values[2], "lpf element 3 (shunt C)", "LPF_N1", GROUND_NET, deck=False, network="lpf")
        bindings["L12"] = _p(b, nb, "bb_coil", "L12", values[3], "lpf element 4 (series L)", "LPF_N1", "TX_OUT", deck=False, network="lpf")
        bindings["C27"] = _p(b, nb, "bb_cap", "C27", values[4], "lpf element 5 (shunt C)", "TX_OUT", GROUND_NET, deck=False, network="lpf")
        b.result.networks.append(RFNetwork(
            id="lpf", block=self.id, members=refs, bindings=bindings, loss_q={"L11": q_u, "L12": q_u}, q_ref_hz=f_c,
            ports=[fixture_port("lpf_in", "LPF_IN", z0, "in"), fixture_port("lpf_out", "TX_OUT", z0, "out")],
            sweep=[ac_sweep(b, "lpf_sweep", lin, pts, _c(b, "lpf.sweep.fstart", 100e6, "Hz", "start of the lpf sweep"),
                            _c(b, "lpf.sweep.fstop", 1500e6, "Hz", "stop of the lpf sweep (past 3 f_c)"), "lpf: 100 MHz .. 1.5 GHz")],
            expectations=[row("s21_fc", "s21_db", "lpf_in", "lpf_out", f_c, l_s21, bound="at_least"),
                          row("s11_fc", "s11_db", "lpf_in", "lpf_in", f_c, l_s11, bound="at_most"),
                          row("s21_2fc", "s21_db", "lpf_in", "lpf_out", f2, h2, bound="at_most"),
                          row("s21_3fc", "s21_db", "lpf_in", "lpf_out", f3, h3, bound="at_most")],
        ))


# --------------------------------------------------------------------------- T/R switch


class BbTrxBlock(Block):
    """The 1N4148 T/R switch with a lumped lambda/4 pi and the SMA (module docstring)."""

    id = "trx"
    title = "T/R switch: series 1N4148 to the antenna, lumped lambda/4 to the receiver, shunt 1N4148 at its far end, SMA"
    interface_nets = ("V_TX", "TX_OUT", "ANT", "RX_IN")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        f_c = _ref(b, "rf.f_c")
        z0 = _ref(b, "rf.z0")
        v_rail, _v5 = _rails(b)
        q_u = model_once(b, "model.bb.coil_q")
        r_on = model_once(b, "model.bb.d_on_r")
        c_off = model_once(b, "model.bb.d_off_c")
        lin, pts = _ref(b, "tx.sweep.lin"), _ref(b, "tx.sweep.points")
        l_q = _k(b, "trsw.l", lambda: radio.quarter_wave_lumped_l(f_c, z0, ("rf.f_c", "rf.z0")))
        c_q = _k(b, "trsw.c", lambda: radio.quarter_wave_lumped_c(f_c, z0, ("rf.f_c", "rf.z0")))
        r_bias = _c(b, "trsw.r_bias", 1e3, "ohm", "R1: the diodes' bias from V_TX (about 7 mA through D1, L1 and D2 while transmitting)")
        l_rfc = _c(b, "trsw.l_rfc", 470e-9, "H", "L2: the bias choke onto TX_OUT (hand-wound) [UNVERIFIED: its self-resonance]")
        c_blk = _c(b, "trsw.c_block", 100e-12, "F", "C4 / C5: DC blocks to the antenna and the receiver")
        c_bp = _c(b, "trsw.c_bypass", 1e-9, "F", "C3: the bias feed's RF bypass")
        tx_il = _c(b, "trsw.tx_il_min", -2.0, "dB", "trsw TX state: TX_OUT -> antenna at least -2 dB")
        tx_iso = _c(b, "trsw.tx_iso_max", -15.0, "dB", "trsw TX state: TX_OUT -> receiver at most -15 dB (the shunt diode on, lambda/4 turns it into an open)")
        rx_il = _c(b, "trsw.rx_il_min", -3.0, "dB", "trsw RX state: antenna -> receiver at least -3 dB (both 1N4148 off: their 4 pF load the antenna node)")
        d1 = b.part("bb_diode", "D1", "1N4148", "T/R switch: series diode TX_OUT -> antenna (on while transmitting)")
        d2 = b.part("bb_diode", "D2", "1N4148", "T/R switch: shunt diode at the receiver's end of the lambda/4 (on while transmitting)")
        for ref in ("D1", "D2"):
            _exclude(b, ref, "a T/R diode: simulated only in the trsw fixture's states")
        _add(nb, "TX_OUT", d1.at("A"))
        _add(nb, "ANT_N", d1.at("K"), "the antenna node")
        _add(nb, "RXQ", d2.at("A"), "the receiver's end of the lambda/4")
        _add(nb, GROUND_NET, d2.at("K"))
        bl1 = _p(b, nb, "bb_coil", "L1", l_q, "lumped lambda/4: series coil", "ANT_N", "RXQ", deck=False, network="trsw")
        bc1 = _p(b, nb, "bb_cap", "C1", c_q, "lumped lambda/4: shunt C at the antenna end", "ANT_N", GROUND_NET, deck=False, network="trsw")
        bc2 = _p(b, nb, "bb_cap", "C2", c_q, "lumped lambda/4: shunt C at the receiver end", "RXQ", GROUND_NET, deck=False, network="trsw")
        bc4 = _p(b, nb, "bb_cap", "C4", c_blk, "DC block to the SMA", "ANT_N", "ANT", deck=False, network="trsw")
        bc5 = _p(b, nb, "bb_cap", "C5", c_blk, "DC block to the receiver", "RXQ", "RX_IN", deck=False, network="trsw")
        br1 = _p(b, nb, "bb_res", "R1", r_bias, "T/R bias resistor", "V_TX", "TR_B", deck=False, network="trsw")
        bl2 = _p(b, nb, "bb_coil", "L2", l_rfc, "T/R bias choke onto TX_OUT", "TR_B", "TX_OUT", deck=False, network="trsw")
        bc3 = _p(b, nb, "bb_cap", "C3", c_bp, "T/R bias bypass", "TR_B", GROUND_NET, deck=False, network="trsw")
        j1 = b.part("bb_sma", "J1", "SMA", "antenna / 50 ohm dummy-load connector", serving(ctx, "radio_build", "carrier_frequency"))
        _exclude(b, "J1", "connector: a fixture port of the system impedance")
        _add(nb, "ANT", j1.at("In"))
        _add(nb, GROUND_NET, j1.at("Ext"))
        nb.declare(b)
        states = []
        for sid, device, value, note in (("tx", SpiceDevice.R, r_on, "forward-biased: model.bb.d_on_r"), ("rx", SpiceDevice.C, c_off, "unbiased: model.bb.d_off_c")):
            states.append(RFState(id=sid, bindings={
                p.ref: SpiceBinding(device=device, value=value, pin_order=[p.pin("A"), p.pin("K")], provenance=ctx.provenance(f"{p.ref} in the trsw {sid} state: {note}"))
                for p in (d1, d2)}))
        b.result.networks.append(RFNetwork(
            id="trsw", block=self.id, members=["D1", "L1", "C1", "C2", "C4", "C5", "R1", "L2", "C3", "D2"],
            bindings={"L1": bl1, "C1": bc1, "C2": bc2, "C4": bc4, "C5": bc5, "R1": br1, "L2": bl2, "C3": bc3}, loss_q={"L1": q_u, "L2": q_u}, q_ref_hz=f_c,
            ports=[fixture_port("tx", "TX_OUT", z0, "in"), fixture_port("ant", "ANT", z0, "bidir"), fixture_port("rx", "RX_IN", z0, "out"),
                   RFPort(name="v_tx", net="V_TX", kind="rail", voltage_v=v_rail, direction="in")],
            states=states,
            sweep=[ac_sweep(b, "trsw_sweep", lin, pts, _c(b, "trsw.sweep.fstart", 300e6, "Hz", "start of the trsw sweep"),
                            _c(b, "trsw.sweep.fstop", 600e6, "Hz", "stop of the trsw sweep"), "trsw: 300-600 MHz")],
            expectations=[row("tx_s21", "s21_db", "tx", "ant", f_c, tx_il, bound="at_least", state="tx"),
                          row("tx_iso", "s21_db", "tx", "rx", f_c, tx_iso, bound="at_most", state="tx"),
                          row("rx_s21", "s21_db", "ant", "rx", f_c, rx_il, bound="at_least", state="rx")],
        ))
        b.result.chain = signal_order(b)
        b.result.ports = [RFPort(name="ANT", net="ANT", kind="port", z0_ohm=z0, frequency_hz=f_c, direction="bidir")]
        b.result.lab_items = [LabItem(id="trsw", block=self.id, what="the T/R switch's TX loss and RX isolation with real 1N4148s at the carrier",
                                      instruments=["RF power meter", "50 ohm dummy load"], reason="a 1N4148 is no PIN diode: its RF resistance and capacitance "
                                                                                                "are model choices")]
        return b.done()


# --------------------------------------------------------------------------- RX


class BbRxBlock(Block):
    """BFR92A common-base RF amplifier -> self-quenched BFR92A super-regenerative detector -> ``AF_RAW`` (module docstring)."""

    id = "rx"
    title = "RX: common-base RF amplifier and self-quenched super-regenerative detector"
    interface_nets = ("V5R", "RX_IN", "AF_RAW")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        f_c = _ref(b, "rf.f_c")
        serves_fc = [ctx.inputs["carrier_frequency"].requirement.id] if "carrier_frequency" in ctx.inputs else []
        l_t = _c(b, "rx.sr.l", 22e-9, "H", "L1: the detector's tank coil (hand-wound, about 2-3 turns) [UNVERIFIED]")
        c_t = _k(b, "rx.sr.c_tank", lambda: lc_c_for_resonance(f_c, l_t, ("rf.f_c", "rx.sr.l")))
        vals = {
            "re1": _c(b, "rx.amp.r_e", 1e3, "ohm", "R1: the RF amplifier's emitter resistor (about 0.85 mA: its common-base input about 30 ohm)"),
            "rb1": _c(b, "rx.amp.r_b1", 10e3, "ohm", "R2: the RF amplifier's base divider (top)"),
            "rb2": _c(b, "rx.amp.r_b2", 4.7e3, "ohm", "R3: the RF amplifier's base divider (bottom)"),
            "rc": _c(b, "rx.amp.r_c", 1e3, "ohm", "R4: the RF amplifier's collector load (broadband)"),
            "c_k": _c(b, "rx.amp.c_couple", 1e-12, "F", "C3: light coupling from the RF amplifier into the detector's tank"),
            "rb": _c(b, "rx.sr.r_b", 10e3, "ohm", "R5 / R6: the detector's base divider"),
            "c_fb": _c(b, "rx.sr.c_fb", 2.2e-12, "F", "C5: the detector's collector-emitter feedback (ngspice-42 with a generic 5 GHz card: self-quenching at about 190 kHz)"),
            "l_rfc": _c(b, "rx.sr.l_rfc", 1e-6, "H", "L2: the detector's emitter choke"),
            "r_q": _c(b, "rx.sr.r_q", 1e3, "ohm", "R7: the quench resistor"),
            "c_q": _c(b, "rx.sr.c_q", 4.7e-9, "F", "C6: the quench capacitor (with R7 and the oscillation's own rectification the quench rate)"),
            "r_reg": _c(b, "rx.sr.r_reg", 47.0, "ohm", "R8: in series with the regeneration trimmer from V5R"),
            "rv": _c(b, "rx.sr.rv", 1e3, "ohm", "RV1: the regeneration trimmer (a rheostat feeding the tank)"),
            "r_af": _c(b, "rx.sr.r_af", 10e3, "ohm", "R9: the quench filter's resistor to AF_RAW"),
            "c_af": _c(b, "rx.sr.c_af", 4.7e-9, "F", "C8: the quench filter's capacitor at AF_RAW (3.4 kHz with R9)"),
            "c_bp": _c(b, "rx.c_bypass", 1e-9, "F", "the RF bypass capacitors of the RX bases and the tank's cold end"),
        }
        _c(b, "rx.sr.f_quench", 190e3, "Hz", QUENCH_TEXT)
        q1 = b.part("bb_npn_rf", "Q1", "BFR92A", "RX RF amplifier (common base)", serves_fc)
        q2 = b.part("bb_npn_rf", "Q2", "BFR92A", "super-regenerative detector", [*serves_fc, *serving(ctx, "modulation")])
        for ref in ("Q1", "Q2"):
            _exclude(b, ref, NO_RF_MODEL)
        _add(nb, "RX_IN", q1.at("E"), "the RF amplifier's emitter: the receiver's input")
        _add(nb, "RXB", q1.at("B"))
        _add(nb, "RXC", q1.at("C"))
        _p(b, nb, "bb_res", "R1", vals["re1"], "RF amplifier emitter resistor", "RX_IN", GROUND_NET)
        _p(b, nb, "bb_res", "R2", vals["rb1"], "RF amplifier base divider (top)", "V5R", "RXB")
        _p(b, nb, "bb_res", "R3", vals["rb2"], "RF amplifier base divider (bottom)", "RXB", GROUND_NET)
        _p(b, nb, "bb_cap", "C1", vals["c_bp"], "RF amplifier base bypass", "RXB", GROUND_NET)
        _p(b, nb, "bb_res", "R4", vals["rc"], "RF amplifier collector load", "V5R", "RXC")
        _p(b, nb, "bb_cap", "C3", vals["c_k"], "coupling into the detector's tank", "RXC", "SR_C", deck=False, network="lab")
        _add(nb, "SR_B", q2.at("B"))
        _add(nb, "SR_C", q2.at("C"), "the detector's collector: the tank's hot end")
        _add(nb, "SR_E", q2.at("E"))
        _p(b, nb, "bb_res", "R5", vals["rb"], "detector base divider (top)", "V5R", "SR_B")
        _p(b, nb, "bb_res", "R6", vals["rb"], "detector base divider (bottom)", "SR_B", GROUND_NET)
        _p(b, nb, "bb_cap", "C4", vals["c_bp"], "detector base bypass", "SR_B", GROUND_NET)
        _p(b, nb, "bb_coil", "L1", l_t, "detector tank coil", "SR_VT", "SR_C", deck=False, network="lab")
        _p(b, nb, "bb_ctrim", "CT1", c_t, "detector tuning trimmer (set to the carrier)", "SR_C", "SR_VT", deck=False, network="lab")
        _p(b, nb, "bb_cap", "C5", vals["c_fb"], "detector feedback capacitor", "SR_C", "SR_E", deck=False, network="lab")
        _p(b, nb, "bb_cap", "C7", vals["c_bp"], "tank cold-end bypass", "SR_VT", GROUND_NET, deck=False, network="lab")
        _p(b, nb, "bb_coil", "L2", vals["l_rfc"], "detector emitter choke", "SR_E", "SR_Q", deck=False, network="lab")
        _p(b, nb, "bb_res", "R7", vals["r_q"], "quench resistor", "SR_Q", GROUND_NET, deck=False, network="lab")
        _p(b, nb, "bb_cap", "C6", vals["c_q"], "quench capacitor", "SR_Q", GROUND_NET, deck=False, network="lab")
        _p(b, nb, "bb_res", "R8", vals["r_reg"], "regeneration series resistor", "V5R", "SR_REG", deck=False, network="lab")
        rv = b.part("bb_trimpot", "RV1", format_part_value(float(vals["rv"].value)), "regeneration trimmer (rheostat)")
        _exclude(b, "RV1", NO_RF_MODEL)
        _add(nb, "SR_REG", rv.at("END1"))
        _add(nb, "SR_VT", [*rv.at("WIPER"), *rv.at("END3")], "the tank's cold end: the regulated feed")
        _p(b, nb, "bb_res", "R9", vals["r_af"], "quench filter resistor", "SR_Q", "AF_RAW", deck=False, network="lab")
        _p(b, nb, "bb_cap", "C8", vals["c_af"], "quench filter capacitor", "AF_RAW", GROUND_NET, deck=False, network="lab")
        nb.declare(b)
        b.result.chain = signal_order(b)
        b.result.ports = [RFPort(name="RX_IN", net="RX_IN", kind="port", z0_ohm=_ref(b, "rf.z0"), frequency_hz=f_c, direction="in")]
        b.result.lab_items = [
            LabItem(id="rx_superregen", block=self.id, what=("the detector's quench (oscilloscope at SR_Q), its tuning to the carrier with CT1 and the "
                                                             "regeneration with RV1 (just past the onset of the hiss), its sensitivity and its slope "
                                                             "detection of narrow-band FM (tune slightly off the carrier)"),
                    instruments=["oscilloscope", "signal generator or a licensed FM handheld"], reason="a super-regenerative detector is not simulated here"),
            LabItem(id="rx_radiation", block=self.id, what="the detector's radiation at the antenna port behind the common-base RF amplifier",
                    instruments=["spectrum analyser"], reason="a super-regenerative detector radiates; the RF amplifier's isolation is not modelled"),
        ]
        return b.done()


# --------------------------------------------------------------------------- RX audio


class BbRxAudioBlock(Block):
    """TL071 gain -> TL071 Sallen-Key low-pass -> volume -> TL071 + BD139 / BD140 follower -> speaker (module docstring)."""

    id = "rx_audio"
    title = "RX audio: gain stage, Sallen-Key low-pass, volume, op-amp driven BD139 / BD140 output stage"
    interface_nets = ("V_RX", "AF_RAW")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        v_rail, _v5 = _rails(b)
        f_ref, bw, f_2bw, ac, tol = _ac_plan(b)
        _vref(b, nb, refs=("R1", "R2", "C1"), net="VREF_R", rail="V_RX", rail_key="bb.v_rail", rail_v=v_rail, eid="vref_r")
        _key, nom = _gain_stage(b, nb, prefix="rx.amp", refs={"u": "U1", "c_in": "C2", "r_bias": "R3", "r_f": "R4", "c_f": "C3", "r_g": "R5", "c_g": "C4",
                                                              "c_dec": "C12"},
                                nets={"in": "AF_RAW", "plus": "RXP", "minus": "RXN", "out": "RXA", "g": "RXG"}, vref="VREF_R", rail="V_RX", f_ref=f_ref,
                                what="RX audio amplifier (TL071)", limiter=False)
        _sallen_key(b, nb, prefix="rx.sk", refs={"u": "U2", "r1": "R6", "r2": "R7", "c1": "C5", "c2": "C6", "c_dec": "C13"},
                    nets={"in": "RXA", "n1": "RXS1", "n2": "RXS2", "out": "RXF"}, rail="V_RX", bw=bw, what="RX low-pass (Sallen-Key; also the quench filter)")
        c_vol = _c(b, "rx.c_vol", 4.7e-6, "F", "C7: the volume pot's coupling electrolytic")
        c_pa = _c(b, "rx.c_pa_in", 1e-6, "F", "C8: the output stage's input coupling electrolytic")
        r_pab = _c(b, "rx.r_pa_bias", 100e3, "ohm", "R8: the output stage's input bias to VREF_R")
        r_drv = _c(b, "rx.r_drive", 1e3, "ohm", "R9 / R10: the output transistors' base feed (about 3.6 mA through the bias diodes)")
        r_em = _c(b, "rx.r_emitter", 1.0, "ohm", "R11 / R12: the output transistors' emitter resistors")
        r_f = _c(b, "rx.pa.r_f", 22e3, "ohm", "R13: the output stage's feedback resistor (from the output to U3's - input)")
        r_g = _c(b, "rx.pa.r_g", 4.7e3, "ohm", "R14: the output stage's ground-leg resistor (gain 1 + R13 / R14 = 5.7)")
        c_g = _c(b, "rx.pa.c_g", 10e-6, "F", "C9: the output stage's ground-leg electrolytic (unity DC gain)")
        c_out = _c(b, "rx.c_out", 220e-6, "F", "C10: the speaker coupling electrolytic (with 8 ohm a 90 Hz corner)")
        r_zo = _c(b, "rx.r_zobel", 10.0, "ohm", "R15: the Zobel resistor at the output")
        c_zo = _c(b, "rx.c_zobel", 100e-9, "F", "C11: the Zobel capacitor")
        dc_tol = _c(b, "rx.out_dc_tol", 0.3, "V", "the output's DC level within 0.3 V of VREF_R (the loop holds it; generic transistor cards)")
        pab_n = _c(b, "rx.quench_max", -20.0, "dB", "the chain AF_RAW -> RXF at the quench rate at most -20 dB (the 2nd-order low-pass and the gain stage's roll-off)")
        rv = b.part("pot", "RV1", "10k", "volume")
        pos = model_once(b, "model.pot.position")
        card_bind(b, "RV1", models.potentiometer_card(_c(b, "rx.vol_r", 10e3, "ohm", "RV1: the volume pot's track resistance"), pos, key="model.pot.vol",
                                                      name="POTVOL"), ignored={"MOUNT": models.POT_MOUNT_IGNORED})
        _p(b, nb, "bb_cap_elec", "C7", c_vol, "volume coupling", "RXF", "VOL_TOP")
        _add(nb, "VOL_TOP", rv.at("END3"))
        _add(nb, GROUND_NET, [*rv.at("END1"), *rv.at("MOUNT")])
        _add(nb, "VOL_W", rv.at("WIPER"))
        _p(b, nb, "bb_cap_elec", "C8", c_pa, "output stage input coupling", "PA_IN", "VOL_W")
        _p(b, nb, "bb_res", "R8", r_pab, "output stage input bias", "PA_IN", "VREF_R")
        u3 = _opamp(b, "U3", "output stage driver (TL071, the BD139 / BD140 inside its loop)")
        _add(nb, "PA_IN", u3.at("+"))
        _add(nb, "PA_N", u3.at("-"))
        _add(nb, "DRV", u3.at("OUT"))
        _add(nb, "V_RX", u3.at("V+"))
        _add(nb, GROUND_NET, u3.at("V-"))
        _p(b, nb, "bb_cap", "C14", _ref(b, "audio.c_decouple"), "U3 supply decoupling", "V_RX", GROUND_NET)
        diode = models.diode_card()
        for ref, a, k in (("D1", "BN", "DRV"), ("D2", "DRV", "BP")):
            d = b.part("bb_diode", ref, "1N4148", "output stage bias diode (two drops between the bases)")
            card_bind(b, ref, diode)
            _add(nb, a, d.at("A"))
            _add(nb, k, d.at("K"))
        _p(b, nb, "bb_res", "R9", r_drv, "upper base feed", "V_RX", "BN")
        _p(b, nb, "bb_res", "R10", r_drv, "lower base feed", "BP", GROUND_NET)
        q1 = b.part("bb_npn_out", "Q1", "BD139", "output stage NPN (upper half)")
        q2 = b.part("bb_pnp_out", "Q2", "BD140", "output stage PNP (lower half)")
        card_bind(b, "Q1", models.npn_card())
        card_bind(b, "Q2", models.pnp_card())
        _add(nb, "BN", q1.at("B"))
        _add(nb, "V_RX", q1.at("C"))
        _add(nb, "E1", q1.at("E"))
        _add(nb, "BP", q2.at("B"))
        _add(nb, GROUND_NET, q2.at("C"))
        _add(nb, "E2", q2.at("E"))
        _p(b, nb, "bb_res", "R11", r_em, "upper emitter resistor", "E1", "OUTE")
        _p(b, nb, "bb_res", "R12", r_em, "lower emitter resistor", "E2", "OUTE")
        _p(b, nb, "bb_res", "R13", r_f, "output stage feedback", "OUTE", "PA_N")
        _p(b, nb, "bb_res", "R14", r_g, "output stage ground leg", "PA_N", "PA_G")
        _p(b, nb, "bb_cap_elec", "C9", c_g, "output stage ground-leg electrolytic", "PA_G", GROUND_NET)
        _p(b, nb, "bb_cap_elec", "C10", c_out, "speaker coupling", "OUTE", "SPK_P")
        _p(b, nb, "bb_res", "R15", r_zo, "Zobel resistor", "OUTE", "ZOB")
        _p(b, nb, "bb_cap", "C11", c_zo, "Zobel capacitor", "ZOB", GROUND_NET)
        j1 = b.part("bb_header_2", "J1", "SPK", "8 ohm speaker (Pin_1 = +)")
        _exclude(b, "J1", "connector: the speaker is not simulated")
        _add(nb, "SPK_P", j1.at("Pin_1"))
        _add(nb, GROUND_NET, j1.at("Pin_2"))
        nb.declare(b)
        zero = Traced(value=0.0, unit="V", provenance=ac.provenance)
        level = _c(b, "rx.test_level", 0.005, "V", "the deck's detector audio at AF_RAW: a 5 mV peak sine at 1 kHz (ngspice-42: the detector moves its "
                                                    "quench node by a few mV for a 0.3-3 mV carrier) [UNVERIFIED]")
        b.result.stimuli.append(Stimulus(id="VAF", source="voltage", net="AF_RAW", reference_net="GND", kind=StimulusKind.SINE,
                                         params={"vo": zero, "va": level, "freq": f_ref, "ac": ac},
                                         provenance=ctx.provenance("AF_RAW driven by the test tone (the detector is not simulated)")))
        note = "RX audio amplifier: 1 + R_f / R_g with the C_f low-pass and the C_g / C_in high-passes (product of first-order terms)"
        db_row(b, "rx_gain_1k", AC_REF, "v(RXA)", "v(AF_RAW)", f_ref, nom, note, tol_db=tol)
        bw_n = _k(b, "rx.sk_bw", lambda: butterworth_db_at(_ref(b, "audio.sk_order"), bw, bw, ("audio.sk_order", "audio.bandwidth", "audio.bandwidth")))
        db_row(b, "rx_lpf_bw", AC_BW, "v(RXF)", "v(RXA)", bw, bw_n, "RX Sallen-Key low-pass at its corner: -3.01 dB (2nd-order Butterworth)", tol_db=tol)
        f_q = _c(b, "rx.sr.f_quench", 190e3, "Hz", QUENCH_TEXT)
        variation, points = _ref(b, "audio.ac_variation"), _ref(b, "audio.ac_points")
        ac_point(b, AC_QUENCH, f_q, variation, points, "the quench rate")
        db_row(b, "rx_quench", AC_QUENCH, "v(RXF)", "v(AF_RAW)", f_q, pab_n, "the quench left at RXF (bound)", bound="at_most")
        g_pa = _k(b, "rx.pa.gain_db", lambda: noninv_gain_db(r_f, r_g, ("rx.pa.r_f", "rx.pa.r_g")))
        f_pg = _k(b, "rx.pa.f_cg", lambda: emphasis_corner(_k(b, "rx.pa.tau_cg", lambda: rc_time_constant(r_g, c_g, ("rx.pa.r_g", "rx.pa.c_g"))), ("rx.pa.tau_cg",)))
        hp = _k(b, "rx.pa.hpg_ref", lambda: highpass1_db_at(f_ref, f_pg, ("audio.f_ref", "rx.pa.f_cg")))
        pa_nom = _k(b, "rx.pa.nom_ref", lambda: db_sum(g_pa, hp, ("rx.pa.gain_db", "rx.pa.hpg_ref")))
        db_row(b, "rx_pa_gain_1k", AC_REF, "v(OUTE)", "v(PA_IN)", f_ref, pa_nom, "output stage: 1 + R13 / R14 with the C9 high-pass (the loop around BD139 / BD140)",
               tol_db=tol)
        vref = _ref(b, "vref_r.v")
        expectation(b, "rx_out_dc", op_analysis(b), "v(OUTE)", Reduce.VALUE, vref, "the output's DC level sits on VREF_R (unity DC gain around the follower)",
                    tol_abs=dc_tol)
        b.result.chain = signal_order(b)
        b.result.lab_items = [LabItem(id="rx_speaker", block=self.id, what="the output power into the 8 ohm speaker and the crossover distortion at low volume",
                                      instruments=["oscilloscope"], reason="generic transistor cards; the speaker is not simulated")]
        return b.done()


__all__ = [
    "F_C_RANGE_HZ",
    "INTERFACE_KINDS",
    "N_MULT",
    "V_IN_RANGE",
    "BbPowerBlock",
    "BbRxAudioBlock",
    "BbRxBlock",
    "BbTrxBlock",
    "BbTxAudioBlock",
    "BbTxRfBlock",
]
