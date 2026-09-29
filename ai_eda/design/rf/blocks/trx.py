"""The antenna end of the KR 447 MHz FM transceiver (kr447 design §2.5, part P13): PIN T/R switch -> harmonic low-pass -> antenna match -> antenna or U.FL.

Invariant: every part comes from the kr447 parts table by library name, every
number is a registered calculator's output over the block's confirmed
choices, the ``model.*`` values (UNVERIFIED) and the confirmed requirements,
and every passive network the PA, the LNA and the antenna see here is an RF
fixture judged on ngspice. Nothing here says that the switch survives the
PA's power, that the antenna radiates or that the radio is legal: the PIN
diodes are ``model.pin.*`` values (R_on / C_off of a BAR64-03W class part,
[UNVERIFIED]), the antenna has no model and its impedance is known only when
the user states ``antenna_impedance``, and a fixture PASS is "a network
verdict under confirmed model values (not a measured part)" at schematic
level - no track, via or ground-return inductance.

Two blocks, because the transceiver's two build variants differ exactly in
the part at the antenna feed (kr447 design §2.0 / §2.5):

* :class:`TrxBlock` (``trx``, local references re-based to 10xx): the
  series-shunt PIN T/R switch between the PA output ``TX_RF`` and the
  receiver input ``RX_RF`` - ``D1`` (series, TX side) from ``TRSW_TX`` to
  the common node ``TRSW_COM``, the lumped quarter-wave pi ``C1`` - ``L1`` -
  ``C2`` (``calc.rf.quarter_wave_lumped.c`` / ``.l`` at f_c and ``rf.z0``:
  7.112 pF / 17.78 nH / 7.112 pF) from ``TRSW_COM`` to the RX end
  ``TRSW_RXE``, and ``D2`` (shunt to GND at the RX end). The DC bias comes
  from ``TX_5V`` through the 0 ohm link ``R2`` (an ideal ammeter in the
  design deck), ``R1`` (``calc.led.R``: (V_TX5 - 2 V_F) / I = 340 ohm for the
  choices 1.6 V and 10 mA) and the RF choke ``L2`` into ``TRSW_TX``; the
  current runs through ``D1``, the quarter-wave inductor and ``D2`` to
  ground, so in TX (``TX_5V`` on) both diodes conduct: ``D1`` passes the
  carrier to the antenna and ``D2`` shorts the RX end, which the quarter-wave
  section turns into an open circuit at ``TRSW_COM``. In RX (``TX_5V`` off)
  both diodes are off and the section is a 50 ohm line to ``RX_RF``. DC
  blocks ``C3`` (TX side), ``C4`` (to the low-pass) and ``C5`` (RX side) keep
  the bias inside the switch. After ``C4`` the harmonic low-pass
  (:func:`~ai_eda.design.rf.blocks.pa.add_harmonic_lpf`, the stage-4 filter
  moved behind the switch as the design says: the switch's own harmonics are
  filtered too, and the receiver pays its loss) from ``LPF_IN`` to the
  antenna side. With a confirmed ``antenna_impedance`` that differs from the
  system impedance (``with_match``, the ``transceiver`` build only) the
  low-pass ends on ``ANT_PORT`` and the L-match ``L3`` (series, on the
  lower-resistance side) / ``C6`` (shunt, across the higher resistance)
  (``calc.rf.lmatch.lowpass.*``) transforms ``rf.z0`` to the antenna's
  resistance on ``ANT_FEED``; otherwise the low-pass ends on ``ANT_FEED``
  directly (a straight system-impedance feed; the match is a lab item).
* :class:`AntennaBlock` (``antenna``, references written as the design
  numbers them, prefix 0): ``ANT1`` - the integral straight quarter-wave wire
  (decision 5A, ``Connector_Wire:SolderWire-0.5sqmm_1x01_D0.9mm_OD2.1mm``),
  length ``trx.ant.length`` = ``calc.rf.quarter_wave`` at f_c with the
  velocity-factor choice 0.95 (159.09 mm at 447.5625 MHz) - on the
  ``transceiver`` build, or ``J1001`` - a U.FL in its place (the
  ``transceiver_conducted`` build: bench and KC conducted samples, never
  fitted with an antenna). The antenna has no model: it is an RF port only
  when its impedance is confirmed; the U.FL is a port of the system
  impedance.

Fixtures (``ir.rf.networks``):

* ``trsw`` (block ``trx``): the switch with its bias feed (``R1`` / ``L2``
  to the ``pin_bias`` rail port - an ideal DC source, an ac short) and its
  DC blocks, between the ports ``tx`` (``TX_RF``), ``com`` (``LPF_IN``) and
  ``rx`` (``RX_RF``), all at ``rf.z0``; state ``tx`` binds both diodes as
  ``model.pin.r_on``, state ``rx`` as ``model.pin.c_off``. Rows (the
  design's one-sided bounds): tx s21 tx -> com at least ``trx.tx_il_min``
  -0.5 dB and tx -> rx at most ``trx.tx_iso_max`` -25 dB; rx s21 com -> rx at
  least ``trx.rx_il_min`` -0.5 dB and com -> tx at most ``trx.rx_iso_max``
  -20 dB. Both inductors carry the series loss of ``model.l_q.uhf``.
* ``lpf`` (block ``trx``): the stage-4 rows (s21 at f_c at least -1.5 dB,
  s11 at f_c at most -15 dB, s21 at 2 f_c at most -45 dB, at 3 f_c at most
  -60 dB) between ``LPF_IN`` and the antenna side.
* ``ant_match`` (block ``trx``, only ``with_match``): s21 at f_c at least
  ``trx.match.s21_min`` -0.5 dB and s11 at f_c at most ``trx.match.s11_max``
  -15 dB between ``ANT_PORT`` (``rf.z0``) and ``ANT_FEED`` (the confirmed
  antenna resistance). Without it ``rf.lab.antenna_match`` says the antenna
  impedance is not stated and the match is a design change after a VNA
  measurement.

The design deck (``op_bias``, the RF transistors' analysis): ``pin_bias`` -
the current through ``R2`` (the diodes on the generic ``model.diode`` card,
both inductors at their values) at the nominal ``calc.led.I`` of the bias
choices, tol_rel ``trx.pin.bias_tol`` 20 % (the design's row): a principle
check of the bias network, never the diodes' R_on. ``TX_5V`` is the power
block's always-on ideal source in the deck, so the row is the TX state; that
the switch is unbiased in RX follows from ``tx_rail_off_rx`` (the ptt /
power rows) and the real hold-up against ``PA_5V`` is the lab item
``tr_sequencing``.

Interface nets (never prefixed): ``TX_RF`` (from the PA's match, block
``pa``), ``RX_RF`` (to the front end's first band-pass, block
``rx_frontend``), ``TX_5V`` and ``ANT_FEED`` (to the antenna block). Their
ports feed ``block.interface.*``.
"""

from __future__ import annotations

from ai_eda.ir import Expectation, NetKind, Reduce, SpiceBinding, SpiceDevice, Traced
from ai_eda.ir.rf import LabItem, RFNetwork, RFPort, RFState
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import led_current, led_series_resistor
from ai_eda.tools.calc.rf import lmatch_lowpass_c_shunt, lmatch_lowpass_l_series, quarter_wave

from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.pa import add_harmonic_lpf
from ai_eda.design.rf.blocks.tx_chain import (
    BIAS_ANALYSIS,
    NetBook,
    ac_sweep,
    bias_analysis,
    calc,
    copy_input,
    exclude,
    fixture_port,
    model_q,
    model_value,
    passive,
    plan_value,
    rail_level,
    requirement_ids,
    row,
)
from ai_eda.design.rf.models import card_binding, diode_card

#: block ids (``ir.rf.blocks`` / the networks' ``block``)
TRX_ID = "trx"
ANTENNA_ID = "antenna"
#: the two build variants (``radio_build``) and the part at the antenna feed of each
VARIANT_ANTENNA = "transceiver"
VARIANT_CONDUCTED = "transceiver_conducted"
FEED_REF: dict[str, str] = {VARIANT_ANTENNA: "ANT1", VARIANT_CONDUCTED: "J1001"}
#: the nets the composition keeps
TX_NET, RX_NET, FEED_NET, PORT_NET = "TX_RF", "RX_RF", "ANT_FEED", "ANT_PORT"
TRX_INTERFACE: tuple[str, ...] = (TX_NET, RX_NET, "TX_5V", FEED_NET)
#: the fixture networks of the trx block (``ant_match`` only with the antenna match)
TRX_NETWORKS: tuple[str, ...] = ("trsw", "lpf", "ant_match")
#: the design-deck row of the PIN bias current
PIN_BIAS_ID = "pin_bias"
#: the antenna_impedance range an L-match of loaded Q at most 2 covers from 50 ohm (ohm)
ANT_R_RANGE: tuple[float, float] = (10.0, 250.0)
#: requirement keys the part at the antenna feed serves (what they are about is the antenna, or the conducted sample's antenna port)
ANTENNA_KEYS: tuple[str, ...] = ("erp", "eirp", "antenna_gain", "antenna_impedance", "link_range", "field_strength_limit")


def trx_net_classes(*, with_match: bool) -> dict[str, list[str]]:
    """The signal-integrity class of each trx net: ``RF50_H`` carries the PA's harmonics (before the low-pass), ``RF50`` the system-impedance
    lines after it and on the RX side, ``RF_ANT`` the antenna side of the match (its target is the antenna's resistance), ``TX_LUMPED`` the
    low-pass's inner nodes."""
    out = {"RF50_H": [TX_NET, "TRSW_TX", "TRSW_COM", "LPF_IN"], "RF50": ["TRSW_RXE", RX_NET], "TX_LUMPED": ["LPF_N1", "LPF_N2"]}
    if with_match:
        out["RF50"].append(PORT_NET)
        out["RF_ANT"] = [FEED_NET]
    else:
        out["RF50"].append(FEED_NET)
    return out


def antenna_resistance(ctx: BlockContext) -> Traced | None:
    """The confirmed ``antenna_impedance`` as a real resistance (ohm), refused outside :data:`ANT_R_RANGE`; ``None`` when not stated."""
    inp = ctx.inputs.get("antenna_impedance")
    if inp is None:
        return None
    v = float(inp.traced.value)
    lo, hi = ANT_R_RANGE
    if not lo <= v <= hi:
        raise TemplateRefusal(f"antenna_impedance {v:.6g} ohm ({inp.requirement.id}) is outside {lo:g}..{hi:g} ohm, the range an L-match of loaded Q at most 2 "
                              "covers from the system impedance (a wider transformation is a human design change: a pi-match or another antenna)")
    return inp.traced


class TrxBlock(Block):
    """PIN T/R switch, harmonic low-pass after it and the optional antenna L-match (module docstring)."""

    id = TRX_ID
    interface_nets = TRX_INTERFACE

    def __init__(self, *, with_match: bool = False) -> None:
        self.with_match = with_match
        # design data (ir.rf.blocks[].title, the topology): it names the antenna L-match only on the build that has one
        self.title = "PIN T/R switch, harmonic low-pass and antenna L-match" if with_match else "PIN T/R switch and harmonic low-pass"

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_z = requirement_ids(ctx, "system_impedance")
        serves_p = requirement_ids(ctx, "tx_power")
        serves_ant = requirement_ids(ctx, "antenna_impedance")
        serves_build = requirement_ids(ctx, "radio_build")
        c = lambda key, value, unit, text: b.choice(key, value, unit, text)  # noqa: E731
        try:
            carrier = ctx.inputs.get("carrier_frequency")
            if carrier is None:
                raise TemplateRefusal(f"block {self.id} needs the confirmed carrier_frequency")
            f_c = ("rf.f_c", copy_input(b, "rf.f_c", carrier.traced))
            z0_in = ctx.inputs.get("system_impedance")
            if z0_in is not None:
                z0 = ("rf.z0", copy_input(b, "rf.z0", z0_in.traced))
            else:
                z0 = ("rf.z0", plan_value(b, "rf.z0", lambda: b.choice("rf.z0", 50.0, "ohm",
                                                                       "system impedance of the RF and IF ports (no system_impedance requirement stated)")))
            v5_key, v5 = rail_level(b, "TX_5V")
            q_u = model_q(b, float(f_c[1].value))
            r_on = model_value(b, "model.pin.r_on")
            c_off = model_value(b, "model.pin.c_off")
            diode_text = b.card(diode_card())
            l_q = ("trx.lq.l", calc(b, "trx.lq.l", lambda: radio.quarter_wave_lumped_l(f_c[1], z0[1], (f_c[0], z0[0]))))
            c_q = ("trx.lq.c", calc(b, "trx.lq.c", lambda: radio.quarter_wave_lumped_c(f_c[1], z0[1], (f_c[0], z0[0]))))
            i_pin = ("trx.pin.i_bias", c("trx.pin.i_bias", 0.01, "A", "PIN diode bias current in TX (10 mA: R_on of a BAR64-03W class diode is specified there [UNVERIFIED: BAR64-03W datasheet])"))
            v_f2 = ("trx.pin.v_f_pair", c("trx.pin.v_f_pair", 1.6, "V", (
                "forward drop of the two PIN diodes in series on the bias path (0.8 V each, a choice [UNVERIFIED: BAR64-03W datasheet]); sets R1001 with calc.led.R")))
            r_bias = ("trx.pin.r", calc(b, "trx.pin.r", lambda: led_series_resistor(v5, v_f2[1], i_pin[1], (v5_key, v_f2[0], i_pin[0]))))
            i_nom = calc(b, "trx.pin.i_nom", lambda: led_current(v5, v_f2[1], r_bias[1], (v5_key, v_f2[0], r_bias[0])))
            bias_tol = c("trx.pin.bias_tol", 0.2, None, "relative tolerance of the pin_bias row (the design's +/- 20 %: model.diode is ngspice's default diode, not a PIN model)")
            link_v = c("trx.link_v", 0.0, "V", "the PIN bias link R1002 simulated as a 0 V source (an ideal ammeter): exact for a 0 ohm link; on the board the lab lifts it to measure the bias")
            l_choke = c("trx.l_choke", 470e-9, "H", "PIN bias RF choke into TRSW_TX (1.32 kohm at f_c; its self-resonance must lie above f_c [UNVERIFIED: choke datasheet])")
            c_blk = c("trx.c_block", 100e-12, "F", "the T/R switch's DC blocks on its TX, common and RX sides (3.6 ohm at f_c)")
            tx_il = c("trx.tx_il_min", -0.5, "dB", "trsw in TX: s21 TX -> common at least -0.5 dB")
            tx_iso = c("trx.tx_iso_max", -25.0, "dB", "trsw in TX: s21 TX -> RX at most -25 dB (the RX end shorted by the shunt diode)")
            rx_il = c("trx.rx_il_min", -0.5, "dB", "trsw in RX: s21 common -> RX at least -0.5 dB")
            rx_iso = c("trx.rx_iso_max", -20.0, "dB", "trsw in RX: s21 common -> TX at most -20 dB (the series diode's C_off)")
            lin = plan_value(b, "tx.sweep.lin", lambda: b.choice("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
            pts = plan_value(b, "tx.sweep.points", lambda: b.choice("tx.sweep.points", 401, None,
                                                                   "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
            f_lo = c("trx.sweep.fstart", 300e6, "Hz", "start of the trsw / ant_match sweeps")
            f_hi = c("trx.sweep.fstop", 600e6, "Hz", "stop of the trsw / ant_match sweeps")
            r_ant: tuple[str, Traced] | None = None
            if self.with_match:
                ant = antenna_resistance(ctx)
                if ant is None:
                    raise TemplateRefusal(f"block {self.id}: the antenna match needs the confirmed antenna_impedance")
                r_ant = ("trx.ant.r", copy_input(b, "trx.ant.r", ant))
                if abs(float(r_ant[1].value) - float(z0[1].value)) <= 1e-9 * float(z0[1].value):
                    raise TemplateRefusal(f"block {self.id}: antenna_impedance equals the system impedance - no match to build (compose the block without it)")
                l_m = ("trx.match.l", calc(b, "trx.match.l", lambda: lmatch_lowpass_l_series(f_c[1], z0[1], r_ant[1], (f_c[0], z0[0], r_ant[0]))))  # type: ignore[index]
                c_m = ("trx.match.c", calc(b, "trx.match.c", lambda: lmatch_lowpass_c_shunt(f_c[1], z0[1], r_ant[1], (f_c[0], z0[0], r_ant[0]))))  # type: ignore[index]
                m_s21 = c("trx.match.s21_min", -0.5, "dB", "ant_match s21 at f_c at least -0.5 dB")
                m_s11 = c("trx.match.s11_max", -15.0, "dB", "ant_match s11 at f_c (from the low-pass side) at most -15 dB")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        what = "PIN T/R switch"
        pin = lambda placed, fn: (placed.ref, placed.pin(fn))  # noqa: E731
        d1 = b.part("pin_diode", "D1", "BAR64-03W", f"{what}: series PIN diode on the TX side (BAR64-03W class [UNVERIFIED])", [*serves_build, *serves_p])
        d2 = b.part("pin_diode", "D2", "BAR64-03W", f"{what}: shunt PIN diode at the RX end of the quarter-wave section (BAR64-03W class [UNVERIFIED])", serves_build)
        card = diode_card()
        for placed in (d1, d2):
            b.bind(placed.ref, card_binding(placed, card, diode_text, ctx.provenance(f"{placed.ref}: model.diode in the design deck (the PIN bias path)")))
        c1, c1_bind = passive(b, "cap_0402", "C1", c_q[1], f"{what}: quarter-wave section, shunt capacitor at the common node", deck=False, network="trsw", serves=serves_fc)
        l1, l1_bind = passive(b, "ind_0604hq", "L1", l_q[1], f"{what}: quarter-wave section, series inductor (also the bias path)", serves=serves_fc)
        c2, c2_bind = passive(b, "cap_0402", "C2", c_q[1], f"{what}: quarter-wave section, shunt capacitor at the RX end", deck=False, network="trsw", serves=serves_fc)
        l2, l2_bind = passive(b, "ind_0603", "L2", l_choke, f"{what}: PIN bias RF choke")
        r1, r1_bind = passive(b, "res_0402", "R1", r_bias[1], f"{what}: PIN bias resistor from TX_5V")
        link = b.part("res_0402", "R2", "0", f"{what}: 0 ohm PIN bias link (lift it to measure the bias current)")
        b.bind("R2", SpiceBinding(device=SpiceDevice.V, value=link_v, pin_order=[link.pin("1"), link.pin("2")],
                                  provenance=ctx.provenance("R1002: a 0 ohm link simulated as a 0 V source (an ideal ammeter)")))
        c3, c3_bind = passive(b, "cap_0402", "C3", c_blk, f"{what}: DC block on the TX side", deck=False, network="trsw")
        c4, c4_bind = passive(b, "cap_0402", "C4", c_blk, f"{what}: DC block to the harmonic low-pass", deck=False, network="trsw")
        c5, c5_bind = passive(b, "cap_0402", "C5", c_blk, f"{what}: DC block on the RX side", deck=False, network="trsw", serves=serves_z)
        nb.add(TX_NET, NetKind.RF, [pin(c3, "1")], "the PA's match output into the T/R switch", serves=serves_z)
        nb.add("TRSW_TX", NetKind.RF, [pin(c3, "2"), pin(d1, "A"), pin(l2, "2")], "T/R switch: the series diode's TX side and the bias feed")
        nb.add("TRSW_COM", NetKind.RF, [pin(d1, "K"), pin(c1, "1"), pin(l1, "1"), pin(c4, "1")], "T/R switch: the common node")
        nb.add("TRSW_RXE", NetKind.RF, [pin(l1, "2"), pin(c2, "1"), pin(d2, "A"), pin(c5, "1")], "T/R switch: the RX end of the quarter-wave section")
        nb.add(RX_NET, NetKind.RF, [pin(c5, "2")], "the T/R switch's RX side into the front end", serves=serves_z)
        nb.add("LPF_IN", NetKind.RF, [pin(c4, "2")], "the harmonic low-pass's input after the switch")
        nb.add("PIN_CHOKE", NetKind.ANALOG, [pin(l2, "1"), pin(r1, "2")], "PIN bias: between the resistor and the choke")
        nb.add("PIN_FEED", NetKind.POWER, [pin(r1, "1"), pin(link, "2")], "PIN bias: after the 0 ohm link")
        nb.add("TX_5V", NetKind.POWER, [pin(link, "1")], "TX_5V rail (the PIN bias is on only in TX)")
        nb.add(GROUND_NET, NetKind.GROUND, [pin(c1, "2"), pin(c2, "2"), pin(d2, "K")])
        lpf_out = PORT_NET if self.with_match else FEED_NET
        lpf = add_harmonic_lpf(b, nb, block_id=self.id, in_net="LPF_IN", out_net=lpf_out, cap_refs=("C10", "C11", "C12", "C13"),
                               ind_refs=("L10", "L11", "L12"), nodes=("LPF_N1", "LPF_N2"), f_c=f_c, z0=z0)
        # the chain from the antenna end (the feed sits at the top-left edge): low-pass, the common DC block, the TX side, the quarter-wave to the RX side
        chain = ["C13", "L12", "C12", "L11", "C11", "L10", "C10", "C4", "D1", "C3", "C1", "L1", "C2", "D2", "C5"]
        networks = []
        if self.with_match:
            assert r_ant is not None
            l3, l3_bind = passive(b, "ind_0604hq", "L3", l_m[1], "antenna L-match: series inductor on the lower-resistance side", deck=False, network="ant_match",
                                  serves=serves_ant)
            c6, c6_bind = passive(b, "cap_0402", "C6", c_m[1], "antenna L-match: shunt capacitor across the higher resistance", deck=False, network="ant_match",
                                  serves=serves_ant)
            high = PORT_NET if float(z0[1].value) > float(r_ant[1].value) else FEED_NET
            nb.add(PORT_NET, NetKind.RF, [pin(l3, "1"), *([pin(c6, "1")] if high == PORT_NET else [])], "the low-pass output at the system impedance")
            nb.add(FEED_NET, NetKind.RF, [pin(l3, "2"), *([pin(c6, "1")] if high == FEED_NET else [])], "the antenna feed", serves=serves_ant)
            nb.add(GROUND_NET, NetKind.GROUND, [pin(c6, "2")])
            chain = ["C6", "L3", *chain]
            networks.append(RFNetwork(
                id="ant_match", block=self.id, members=["L3", "C6"], bindings={"L3": l3_bind, "C6": c6_bind}, loss_q={"L3": q_u[1]}, q_ref_hz=f_c[1],
                ports=[fixture_port("ant_port", PORT_NET, z0[1], "in"), fixture_port("ant_feed", FEED_NET, r_ant[1], "out")],
                sweep=[ac_sweep(b, "ant_match_sweep", lin, pts, f_lo, f_hi, "ant_match: 300-600 MHz")],
                expectations=[row("s21_fc", "s21_db", "ant_port", "ant_feed", f_c[1], m_s21, bound="at_least"),
                              row("s11_fc", "s11_db", "ant_port", "ant_port", f_c[1], m_s11, bound="at_most")],
            ))
        nb.declare(b)
        pin_states = []
        for sid, device, value, note in (("tx", SpiceDevice.R, r_on[1], "model.pin.r_on (TX: forward biased)"), ("rx", SpiceDevice.C, c_off[1], "model.pin.c_off (RX: unbiased)")):
            pin_states.append(RFState(id=sid, bindings={
                p.ref: SpiceBinding(device=device, value=value, pin_order=[p.pin("A"), p.pin("K")], provenance=ctx.provenance(f"{p.ref} in the trsw {sid} state: {note}"))
                for p in (d1, d2)}))
        trsw = RFNetwork(
            id="trsw", block=self.id, members=["C3", "D1", "C1", "L1", "C2", "D2", "C4", "C5", "L2", "R1"],
            bindings={"C3": c3_bind, "C1": c1_bind, "L1": l1_bind, "C2": c2_bind, "C4": c4_bind, "C5": c5_bind, "L2": l2_bind, "R1": r1_bind},
            loss_q={"L1": q_u[1], "L2": q_u[1]}, q_ref_hz=f_c[1],
            ports=[fixture_port("tx", TX_NET, z0[1], "in"), fixture_port("com", "LPF_IN", z0[1], "bidir"), fixture_port("rx", RX_NET, z0[1], "out"),
                   RFPort(name="pin_bias", net="PIN_FEED", kind="rail", voltage_v=v5, direction="in")],
            states=pin_states,
            sweep=[ac_sweep(b, "trsw_sweep", lin, pts, f_lo, f_hi, "trsw: 300-600 MHz")],
            expectations=[row("s21_tx_com", "s21_db", "tx", "com", f_c[1], tx_il, bound="at_least", state="tx"),
                          row("s21_tx_rx", "s21_db", "tx", "rx", f_c[1], tx_iso, bound="at_most", state="tx"),
                          row("s21_com_rx", "s21_db", "com", "rx", f_c[1], rx_il, bound="at_least", state="rx"),
                          row("s21_com_tx", "s21_db", "com", "tx", f_c[1], rx_iso, bound="at_most", state="rx")],
        )
        b.result.networks += [trsw, lpf, *networks]
        bias_analysis(b)
        b.result.expectations.append(Expectation(
            id=PIN_BIAS_ID, analysis_id=BIAS_ANALYSIS, vector="i(R2)", reduce=Reduce.VALUE, nominal=i_nom, tol_rel=bias_tol,
            provenance=ctx.provenance("the PIN bias current through the 0 ohm link R1002 at the nominal of calc.led.I (both diodes on model.diode; the TX state)"),
        ))
        b.result.ports = [
            RFPort(name=TX_NET, net=TX_NET, kind="port", z0_ohm=z0[1], frequency_hz=f_c[1], direction="in"),
            RFPort(name=RX_NET, net=RX_NET, kind="port", z0_ohm=z0[1], frequency_hz=f_c[1], direction="out"),
            RFPort(name="TX_5V", net="TX_5V", kind="rail", voltage_v=v5, direction="in"),
            RFPort(name=FEED_NET, net=FEED_NET, kind="port", z0_ohm=(r_ant[1] if r_ant is not None else z0[1]), frequency_hz=f_c[1], direction="bidir"),
        ]
        b.result.chain = chain
        b.result.net_classes = trx_net_classes(with_match=self.with_match)
        b.result.lab_items += [
            LabItem(id="trsw_power", block=self.id, what="T/R switch insertion loss and isolation at +27 dBm, the PIN diodes' temperature and harmonics",
                    instruments=["power meter", "spectrum analyser", "thermal camera"],
                    reason="the trsw rows are small-signal network verdicts under model.pin.* values; power handling is [UNVERIFIED: BAR64-03W datasheet]"),
            LabItem(id="tr_sequencing", block=self.id, what="the PIN bias hold-up against PA_5V at PTT release and key-up (the PA must be off before the switch loses bias)",
                    instruments=["oscilloscope"], reason="the design deck's regulators are ideal sources; the real decay of TX_5V against PA_5V is measured"),
        ]
        if not self.with_match:
            b.result.lab_items.append(LabItem(
                id="antenna_match", block=self.id, what="the antenna feed impedance and its match to the system impedance (VNA), then a new antenna_impedance",
                instruments=["VNA"], reason="antenna_impedance not stated: the feed is a straight system-impedance line, and the match is a human design change after a VNA measurement"))
        b.result.notes.append(f"{self.id}: the harmonic low-pass sits after the T/R switch (kr447 design §1.2): the switch's own harmonics are filtered and the "
                              "receiver pays the low-pass's loss; every trsw / lpf / ant_match PASS is a network verdict under confirmed model values")
        return b.done()


class AntennaBlock(Block):
    """The part at the antenna feed: ``ANT1`` (the quarter-wave wire) or ``J1001`` (the conducted variant's U.FL) - see the module docstring."""

    id = ANTENNA_ID
    interface_nets = (FEED_NET,)

    def __init__(self, variant: str) -> None:
        if variant not in FEED_REF:
            raise ValueError(f"antenna variant must be one of {sorted(FEED_REF)}, got {variant!r}")
        self.variant = variant
        self.title = ("integral straight quarter-wave wire antenna ANT1 (decision 5A)" if variant == VARIANT_ANTENNA
                      else "conducted antenna port J1001 (U.FL in place of the antenna; never fitted with one)")

    @property
    def ref(self) -> str:
        return FEED_REF[self.variant]

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        serves = [*requirement_ids(ctx, "radio_build"), *requirement_ids(ctx, *ANTENNA_KEYS)]
        serves_z = requirement_ids(ctx, "system_impedance")
        try:
            carrier = ctx.inputs.get("carrier_frequency")
            if carrier is None:
                raise TemplateRefusal(f"block {self.id} needs the confirmed carrier_frequency")
            f_c = copy_input(b, "rf.f_c", carrier.traced)
            z0_in = ctx.inputs.get("system_impedance")
            z0 = copy_input(b, "rf.z0", z0_in.traced) if z0_in is not None else plan_value(
                b, "rf.z0", lambda: b.choice("rf.z0", 50.0, "ohm", "system impedance of the RF and IF ports (no system_impedance requirement stated)"))
            if self.variant == VARIANT_ANTENNA:
                vf = b.choice("trx.ant.vf", 0.95, None, ("velocity factor of the straight quarter-wave wire (a choice: the wire's diameter and the plane end shorten it; "
                                                       "the real resonant length is a lab item after a VNA measurement)"))
                length = calc(b, "trx.ant.length", lambda: quarter_wave(f_c, vf, ("rf.f_c", "trx.ant.vf")))
                r_ant = ctx.inputs.get("antenna_impedance")
                if r_ant is not None:
                    antenna_resistance(ctx)
                    r_ant_t = copy_input(b, "trx.ant.r", r_ant.traced)
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        if self.variant == VARIANT_ANTENNA:
            ant = b.part("antenna", "ANT1", f"wire {float(length.value) * 1000.0:.1f}mm", "integral straight quarter-wave wire antenna (decision 5A), soldered to the feed pad at the plane edge", serves)
            exclude(b, "ANT1", "antenna: no model (its feed impedance is a lab item; an RF port only when antenna_impedance is confirmed)")
            b.net(FEED_NET, NetKind.RF, ant.at("A"), "the antenna feed", serves=serves)
            if ctx.inputs.get("antenna_impedance") is not None:
                b.result.ports = [RFPort(name=FEED_NET, net=FEED_NET, kind="port", z0_ohm=r_ant_t, frequency_hz=f_c, direction="bidir")]
            b.result.lab_items += [
                LabItem(id="antenna_length", block=self.id, what="the quarter-wave wire's resonant length against the ground plane (the velocity factor is a choice)",
                        instruments=["VNA"], reason="calc.rf.quarter_wave is the straight-wire formula; the plane, the feed pad and the wire's first millimetres over it are not modelled"),
                LabItem(id="antenna_radiation", block=self.id, what="ERP / EIRP, radiation efficiency and pattern of the integral antenna",
                        instruments=["anechoic chamber", "calibrated antenna"], reason="no antenna model: every radiated number is a lab measurement"),
                LabItem(id="radiated_spurious", block=self.id, what="radiated spurious emissions of the complete radio against kr447.spurious_max",
                        instruments=["spectrum analyser", "anechoic chamber"], reason="the fixtures judge networks, never the radiated spectrum"),
                LabItem(id="sar", block=self.id, what="RF exposure / SAR of the handheld if applicable [UNVERIFIED: 「전자파 인체보호기준」]",
                        instruments=["SAR test system"], reason="lab only; whether it applies is itself unverified"),
            ]
        else:
            j = b.part("coax_ufl", "J1001", "U.FL", "conducted antenna port (U.FL in place of the antenna, for bench and KC conducted samples; never fitted with an antenna)",
                       [*serves, *serves_z])
            exclude(b, "J1001", "U.FL connector: excluded; a fixture port of the system impedance")
            b.net(FEED_NET, NetKind.RF, j.at("In"), "the conducted antenna port", serves=[*serves, *serves_z])
            b.net(GROUND_NET, NetKind.GROUND, j.at("Ext"), "ground")
            b.result.ports = [RFPort(name=FEED_NET, net=FEED_NET, kind="port", z0_ohm=z0, frequency_hz=f_c, direction="bidir")]
            b.result.lab_items.append(LabItem(
                id="conducted_sample", block=self.id, what="conducted power, spurious emissions and frequency error at J1001 into an attenuator or a dummy load",
                instruments=["power meter", "spectrum analyser", "attenuator", "dummy load"],
                reason="the conducted sample is measured, never fitted with an antenna; whether KC accepts a temporary antenna connector is [UNVERIFIED: RRA test method]"))
        b.result.lab_items.append(LabItem(
            id="kc_conformity", block=self.id, what="KC conformity assessment of the radio at a designated test lab before any transmission",
            instruments=["designated test lab"],
            reason="every kr447.* number is an UNVERIFIED placeholder; 전파법 제58조의2 applies to a self-built unit too [UNVERIFIED: 「방송통신기자재등의 적합성평가에 관한 고시」]"))
        b.result.chain = [self.ref]
        return b.done()


__all__ = [
    "ANTENNA_ID",
    "ANTENNA_KEYS",
    "ANT_R_RANGE",
    "FEED_NET",
    "FEED_REF",
    "PIN_BIAS_ID",
    "PORT_NET",
    "RX_NET",
    "TRX_ID",
    "TRX_INTERFACE",
    "TRX_NETWORKS",
    "TX_NET",
    "VARIANT_ANTENNA",
    "VARIANT_CONDUCTED",
    "AntennaBlock",
    "TrxBlock",
    "antenna_resistance",
    "trx_net_classes",
]
