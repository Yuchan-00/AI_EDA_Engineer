"""The power amplifier of the KR 447 MHz FM radio (kr447 design §2.4, part P12): PA -> load-line L-match -> 7th-order Chebyshev LPF -> conducted output.

Invariant: every part comes from the kr447 parts table by library name, every
number is a registered calculator's output over the block's confirmed
choices, the ``model.*`` values (UNVERIFIED) and the confirmed requirements,
and every passive network the PA sees is an RF fixture judged on ngspice.
Nothing here says that the PA delivers its power, is stable or is legal: the
MMZ09332BT1 has no model (its coverage of 447 MHz, 0.5 W at 5 V, its gain and
the POWER_DOWN polarity are [UNVERIFIED: NXP MMZ09332B datasheet]), its bias
networks are choices from an application circuit nobody here has read, and a
fixture PASS is "a network verdict under confirmed model values (not a
measured part)" at schematic level.

The block (:class:`PaBlock`, ``pa``, local references re-based to 9xx):

* ``U1`` MMZ09332BT1: ``RF_IN`` through the DC block ``C7`` from ``PA_IN``
  (the PA-drive pad of the ``tx_driver`` block); ``VCC1`` on ``PA_5V``
  (decoupling ``C4``); ``VBA1`` / ``VBA2`` / ``VBIAS`` fed from ``PA_5V``
  through ``R1`` / ``R2`` / ``R3`` with their decoupling ``C1`` .. ``C3``
  (``pa.r_vba`` / ``pa.r_vbias``: choices [UNVERIFIED: the MMZ09332B
  application circuit; ground them with --online]); ``VCC2/RFOUT`` (three
  stacked-by-function pins) fed through the RF choke ``L1`` from ``PA_5V``
  (bulk ``C5`` + RF bypass ``C6``); ``POWER_DOWN`` on ``PA_PD`` (high =
  powered down, from the library pin name - ``pa.power_down_polarity`` of
  the PTT block [UNVERIFIED]); ``PDET`` to the test point ``TP1``; the NC
  pins left open.
* The load-line L-match ``pa_match``: the PA's optimum load
  R_L = (V_CC - V_sat)^2 / (2 P) (``calc.rf.pa.load_line_r``: 20.25 ohm for
  5 V / ``pa.v_sat`` 0.5 V / 0.5 W) transformed to the system impedance
  ``rf.z0`` by a low-pass L (series ``L2`` at the PA, shunt ``C8`` at the
  50 ohm side, ``calc.rf.lmatch.lowpass.*``), then the DC block ``C9``.
  Its fixture (the choke ``L1`` to the ``PA_5V`` rail included) is judged
  s21 at f_c at least ``pa.match.s21_min`` -0.5 dB and s11 at most
  ``pa.match.s11_max`` -15 dB.
* The harmonic LPF ``lpf`` (:func:`add_harmonic_lpf`, also the trx block's
  on the transceiver, where the LPF sits after the T/R switch): 7th-order
  0.1 dB Chebyshev, ripple edge ``lpf.f_edge`` 480 MHz, shunt C first
  (``calc.rf.lpf.chebyshev.g`` / ``.shunt_c`` / ``.series_l``: 7.833 pF /
  23.588 nH / 13.904 pF / 26.085 nH / ...), judged under ``model.l_q.uhf``:
  s21 at f_c at least -1.5 dB (the Q-40 network gives -1.045 dB), s11 at f_c
  at most -15 dB, s21 at 2 f_c at most -45 dB and at 3 f_c at most -60 dB
  (the lossless prototype gives 52.75 / 80.3 dB). The stopband verdict holds
  only for a layout with a via to the plane at each shunt pad (at 447 MHz,
  1 mm of track to ground is about 1 nH; ``routing.maze`` 0.4 gives each GND
  pad of the RF blocks its plane via).
* ``J1`` U.FL - the **conducted** output into an attenuator or a dummy load
  (``TX_OUT``, the system impedance). This board never gets an antenna.
  ``U1`` and ``J1`` serve the confirmed ``radio_build`` (the transmitter's
  output stage and its output), so the build requirement is traced to parts.

Its nets' signal-integrity classes (:func:`pa_net_classes`, the block's
``net_classes``; the composing template declares the classes): ``RF50`` the
drive, ``RF50_H`` the match output and the LPF input (they carry the
harmonics), ``RF_OUT`` the output, ``TX_LUMPED`` the load-line node and the
LPF's inner nodes.

The design deck: the PA has no model, so ``U1`` is simulated as its supply
draw (the card ``model.pa.supply``): ``model.pa.r_supply`` from ``VCC1`` to
GND while ``POWER_DOWN`` is below ``model.pa.v_pd`` (powered up), only
``model.pa.r_off`` while it is above (powered down), every other pin ignored
with the reason. The release edge that opens the PTT block's PA switch also
raises ``PA_PD``, so the PA's draw is not what empties ``PA_5V`` at release:
the PTT block's active discharge is (``pa_supply_off_first`` judges it). The
match and LPF parts are fixture members only.
"""

from __future__ import annotations

from ai_eda.ir import NetKind, SpiceBinding, Traced
from ai_eda.ir.rf import LabItem, RFNetwork, RFPort
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.rf import chebyshev_g, lmatch_lowpass_c_shunt, lmatch_lowpass_l_series, lpf_series_l, lpf_shunt_c

from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.tx_chain import (
    PA_INPUT,
    NetBook,
    ac_sweep,
    calc,
    copy_input,
    exclude,
    fixture_port,
    index_choices,
    model_q,
    model_value,
    passive,
    plan_value,
    rail_level,
    requirement_ids,
    row,
)
from ai_eda.design.rf.models import ModelValue, card_binding, pa_supply_card
from ai_eda.design.rf.profile import PROFILE_BY_KEY

#: block id and interface nets
PA_ID = "pa"
#: the network ids this block declares (``lpf`` only with the LPF)
PA_NETWORKS: tuple[str, ...] = ("pa_match", "lpf")
#: the PA's supply draw in the design deck (the PA has no model): powered up, powered down, and the POWER_DOWN level between them
PA_SUPPLY = ModelValue("model.pa.r_supply", 25.0, "ohm",
                       "the PA's supply draw from VCC1 while powered up, simulated as a resistor in the design deck (about 200 mA at 5 V: 0.5 W out at "
                       "about 50 % efficiency; the PA has no model)", "NXP MMZ09332B datasheet and the board's measured TX current")
PA_SUPPLY_OFF = ModelValue("model.pa.r_off", 1e6, "ohm",
                           "the PA's supply draw from VCC1 while POWER_DOWN is asserted (about 5 uA at 5 V: it discharges nothing in a millisecond)",
                           "NXP MMZ09332B datasheet (the powered-down supply current)")
PA_PD_THRESHOLD = ModelValue("model.pa.v_pd", 1.2, "V",
                             "the POWER_DOWN pin's logic threshold: at or above it the PA is powered down (PA_PD swings between 0 V and TX_3V3)",
                             "NXP MMZ09332B datasheet (POWER_DOWN input levels)")
#: the tx_power range the PA stage is designed for (W): above 1 W the MMZ09332BT1 class is out of reach [UNVERIFIED]; the profile's limit is a separate check
P_OUT_RANGE_W: tuple[float, float] = (0.01, 1.0)


def pa_net_classes(*, with_lpf: bool = True, out_net: str = "TX_OUT") -> dict[str, list[str]]:
    """The signal-integrity class of each PA net (``BlockResult.net_classes``): ``RF50`` the drive, ``RF50_H`` the lines that carry the harmonics
    before the low-pass, ``RF_OUT`` the conducted output, ``TX_LUMPED`` the load-line node and the low-pass's inner nodes."""
    return {"RF50": ["PA_IN", "PA_RFIN"], "RF50_H": ["PA_MATCH", *(["LPF_IN"] if with_lpf else [])], "RF_OUT": [out_net],
            "TX_LUMPED": ["PA_OUT", *(["LPF_N1", "LPF_N2"] if with_lpf else [])]}


def add_harmonic_lpf(b: BlockBuilder, nb: NetBook, *, block_id: str, in_net: str, out_net: str, cap_refs: tuple[str, str, str, str],
                     ind_refs: tuple[str, str, str], nodes: tuple[str, str], f_c: tuple[str, Traced], z0: tuple[str, Traced]) -> RFNetwork:
    """The 7th-order 0.1 dB Chebyshev low-pass ``lpf`` (shunt C first) between ``in_net`` and ``out_net``: parts, nets and its fixture.

    Every value is a calculator output over the choices ``lpf.n`` /
    ``lpf.ripple`` / ``lpf.f_edge`` and ``z0``; the rows are one-sided bounds
    (``lpf.s21_min``, ``lpf.s11_max``, ``lpf.h2_max``, ``lpf.h3_max``) at f_c,
    2 f_c and 3 f_c (``calc.rf.harmonic``). The keys are the network's own
    (``lpf.*``), so the transceiver's trx block builds the same filter with
    the same rows after the T/R switch.
    """
    c = lambda key, value, unit, text: plan_value(b, key, lambda: b.choice(key, value, unit, text))  # noqa: E731
    try:
        n = ("lpf.n", c("lpf.n", 7.0, None, "harmonic low-pass order (7th: 52.75 dB at 2 f_c lossless; shunt C first, equal terminations need an odd order)"))
        rip = ("lpf.ripple", c("lpf.ripple", 0.1, "dB", "harmonic low-pass passband ripple (Chebyshev 0.1 dB)"))
        f_e = ("lpf.f_edge", c("lpf.f_edge", 480e6, "Hz", "harmonic low-pass ripple edge (480 MHz: the carrier in the passband with margin, 2 f_c far in the stopband)"))
        s21_min = c("lpf.s21_min", -1.5, "dB", "lpf s21 at f_c at least -1.5 dB (the Q-40 network gives -1.045 dB)")
        s11_max = c("lpf.s11_max", -15.0, "dB", "lpf s11 at f_c at most -15 dB")
        h2_max = c("lpf.h2_max", -45.0, "dB", "lpf s21 at 2 f_c at most -45 dB (the lossless prototype gives -52.75 dB)")
        h3_max = c("lpf.h3_max", -60.0, "dB", "lpf s21 at 3 f_c at most -60 dB (the lossless prototype gives -80.3 dB)")
        k2 = ("lpf.k2", c("lpf.k2", 2.0, None, "harmonic number 2 (the lpf's second-harmonic row)"))
        k3 = ("lpf.k3", c("lpf.k3", 3.0, None, "harmonic number 3 (the lpf's third-harmonic row)"))
        f2 = calc(b, "lpf.f_2", lambda: radio.harmonic(f_c[1], k2[1], (f_c[0], k2[0])))
        f3 = calc(b, "lpf.f_3", lambda: radio.harmonic(f_c[1], k3[1], (f_c[0], k3[0])))
        order = int(round(float(n[1].value)))
        if order != 7:
            raise TemplateRefusal(f"the harmonic low-pass is laid out for 7 elements (4 shunt C, 3 series L); lpf.n = {order}")
        idx = index_choices(b, "lpf", order)
        values: list[Traced] = []
        for k in range(1, order + 1):
            g = calc(b, f"lpf.g.{k}", lambda k=k: chebyshev_g(n[1], idx[k][1], rip[1], (n[0], idx[k][0], rip[0])))
            if k % 2:
                values.append(calc(b, f"lpf.c.{k}", lambda k=k, g=g: lpf_shunt_c(g, f_e[1], z0[1], (f"lpf.g.{k}", f_e[0], z0[0]))))
            else:
                values.append(calc(b, f"lpf.l.{k}", lambda k=k, g=g: lpf_series_l(g, f_e[1], z0[1], (f"lpf.g.{k}", f_e[0], z0[0]))))
        q_u = model_q(b, float(f_c[1].value))
        lin = plan_value(b, "tx.sweep.lin", lambda: b.choice("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
        pts = plan_value(b, "tx.sweep.points", lambda: b.choice("tx.sweep.points", 401, None,
                                                               "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
        f_lo = c("lpf.sweep.fstart", 100e6, "Hz", "start of the lpf sweep")
        f_hi = c("lpf.sweep.fstop", 1500e6, "Hz", "stop of the lpf sweep (past 3 f_c)")
    except ValueError as e:
        raise TemplateRefusal(f"block {block_id}: {e}") from e
    what = "harmonic low-pass (7th-order 0.1 dB Chebyshev)"
    refs = [cap_refs[0], ind_refs[0], cap_refs[1], ind_refs[1], cap_refs[2], ind_refs[2], cap_refs[3]]
    bindings: dict[str, SpiceBinding] = {}
    placed = {}
    for k, (ref, value) in enumerate(zip(refs, values), start=1):
        kind = "cap_0402" if k % 2 else "ind_0604hq"
        desc = f"{what}: element {k} ({'shunt C' if k % 2 else 'series L'})"
        placed[ref], bindings[ref] = passive(b, kind, ref, value, desc, deck=False, network="lpf")
    chain_nets = [in_net, nodes[0], nodes[1], out_net]
    for i, lref in enumerate(ind_refs):
        nb.add(chain_nets[i], NetKind.RF if chain_nets[i] in (in_net, out_net) else NetKind.ANALOG, [(lref, placed[lref].pin("1"))])
        nb.add(chain_nets[i + 1], NetKind.RF if chain_nets[i + 1] in (in_net, out_net) else NetKind.ANALOG, [(lref, placed[lref].pin("2"))])
    for cref, net in zip(cap_refs, chain_nets):
        nb.add(net, NetKind.RF if net in (in_net, out_net) else NetKind.ANALOG, [(cref, placed[cref].pin("1"))], f"{what}: node" if net in nodes else "")
        nb.add(GROUND_NET, NetKind.GROUND, [(cref, placed[cref].pin("2"))])
    sweep = ac_sweep(b, "lpf_sweep", lin, pts, f_lo, f_hi, "lpf: 100 MHz .. 1.5 GHz")
    return RFNetwork(
        id="lpf", block=block_id, members=refs, bindings=bindings, loss_q={r: q_u[1] for r in ind_refs}, q_ref_hz=f_c[1],
        ports=[fixture_port("lpf_in", in_net, z0[1], "in"), fixture_port("lpf_out", out_net, z0[1], "out")], sweep=[sweep],
        expectations=[row("s21_fc", "s21_db", "lpf_in", "lpf_out", f_c[1], s21_min, bound="at_least"),
                      row("s11_fc", "s11_db", "lpf_in", "lpf_in", f_c[1], s11_max, bound="at_most"),
                      row("s21_2fc", "s21_db", "lpf_in", "lpf_out", f2, h2_max, bound="at_most"),
                      row("s21_3fc", "s21_db", "lpf_in", "lpf_out", f3, h3_max, bound="at_most")],
    )


class PaBlock(Block):
    """PA, load-line match, harmonic LPF and the conducted U.FL output (module docstring).

    ``with_lpf`` / ``with_output``: the stage-4 exciter builds both; the
    transceiver moves the LPF after its T/R switch (the trx block builds it
    with :func:`add_harmonic_lpf`) and has no connector here, and the match's
    DC block then drives ``out_net`` (the switch's TX side).
    """

    id = PA_ID

    def __init__(self, *, with_lpf: bool = True, with_output: bool = True, out_net: str = "TX_OUT") -> None:
        if with_output and not with_lpf:
            raise ValueError("the conducted output connector sits after the harmonic low-pass: with_output needs with_lpf")
        self.with_lpf = with_lpf
        self.with_output = with_output
        self.out_net = out_net
        self.interface_nets = ("PA_IN", "PA_5V", "PA_PD", out_net)
        # design data (ir.rf.blocks[].title, the topology): it names only what this build of the block holds
        if with_lpf and with_output:
            self.title = "PA, load-line match, harmonic low-pass and the conducted output"
        elif with_lpf:
            self.title = "PA, load-line match and harmonic low-pass"
        else:
            self.title = f"PA and load-line match into {out_net} (the harmonic low-pass and the output are another block's)"

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_p = requirement_ids(ctx, "tx_power")
        serves_z = requirement_ids(ctx, "system_impedance")
        serves_build = requirement_ids(ctx, "radio_build")  # the transmitter's output stage and its conducted output serve the build itself
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
            pa_in = model_value(b, PA_INPUT)
            r_sup = model_value(b, PA_SUPPLY)
            r_off = model_value(b, PA_SUPPLY_OFF)
            v_pd = model_value(b, PA_PD_THRESHOLD)
            sup_card = pa_supply_card(r_sup[1], r_off[1], v_pd[1])
            sup_text = b.card(sup_card)
            p_in = ctx.inputs.get("tx_power")
            if p_in is not None:
                lo, hi = P_OUT_RANGE_W
                if not lo <= float(p_in.traced.value) <= hi:
                    raise TemplateRefusal(f"tx_power {float(p_in.traced.value):.6g} W ({p_in.requirement.id}) is outside {lo:g}..{hi:g} W, the class the "
                                          "MMZ09332BT1 stage is designed for [UNVERIFIED: NXP MMZ09332B datasheet]")
                p_out = ("pa.p_out", copy_input(b, "pa.p_out", p_in.traced))
            else:
                limit = PROFILE_BY_KEY["kr447.max_power"]
                p_out = ("pa.p_out", c("pa.p_out", float(limit.value), "W", (
                    "tx_power is not stated: the PA's load line is designed for the profile's kr447.max_power "
                    f"[UNVERIFIED: {limit.document}; a placeholder, not a grounded limit]")))
            v_sat = ("pa.v_sat", c("pa.v_sat", 0.5, "V", "the PA's output saturation voltage for the load line [UNVERIFIED: NXP MMZ09332B datasheet]"))
            r_l = ("pa.r_l", calc(b, "pa.r_l", lambda: radio.pa_load_line_r(v5, v_sat[1], p_out[1], (v5_key, v_sat[0], p_out[0]))))
            if not float(r_l[1].value) < float(z0[1].value):
                raise TemplateRefusal(f"the load line R_L = {float(r_l[1].value):.6g} ohm is not below the system impedance {float(z0[1].value):.6g} ohm: "
                                      "the low-pass L-match of this block transforms up only")
            l_m = calc(b, "pa.match.l", lambda: lmatch_lowpass_l_series(f_c[1], r_l[1], z0[1], (f_c[0], r_l[0], z0[0])))
            c_m = calc(b, "pa.match.c", lambda: lmatch_lowpass_c_shunt(f_c[1], r_l[1], z0[1], (f_c[0], r_l[0], z0[0])))
            m_s21 = c("pa.match.s21_min", -0.5, "dB", "pa_match s21 at f_c at least -0.5 dB")
            m_s11 = c("pa.match.s11_max", -15.0, "dB", "pa_match s11 at f_c (looking from the PA's load line) at most -15 dB")
            l_ch = c("pa.l_choke", 220e-9, "H", "VCC2 / RF output choke from PA_5V (618 ohm at f_c; its current rating and self-resonance [UNVERIFIED])")
            c_blk = c("pa.c_block", 100e-12, "F", "PA input and output DC blocks (3.6 ohm at f_c)")
            c_rf = c("pa.c_rf", 100e-12, "F", "RF decoupling at the PA's supply pins")
            c_bulk = c("pa.c_bulk", 10e-6, "F", "PA_5V bulk capacitor at the PA")
            r_vba = c("pa.r_vba", 10.0, "ohm", "VBA1 / VBA2 feed resistors from PA_5V [UNVERIFIED: the MMZ09332B application circuit - ground it with --online]")
            r_vbias = c("pa.r_vbias", 1e3, "ohm", "VBIAS feed resistor from PA_5V [UNVERIFIED: the MMZ09332B application circuit - ground it with --online]")
            q_u = model_q(b, float(f_c[1].value))
            lin = plan_value(b, "tx.sweep.lin", lambda: b.choice("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
            pts = plan_value(b, "tx.sweep.points", lambda: b.choice("tx.sweep.points", 401, None,
                                                                   "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
            f_lo = c("pa.sweep.fstart", 300e6, "Hz", "start of the pa_match sweep")
            f_hi = c("pa.sweep.fstop", 600e6, "Hz", "stop of the pa_match sweep")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        u = b.part("pa", "U1", "MMZ09332BT1", "power amplifier (0.5 W class at 5 V) [UNVERIFIED: NXP MMZ09332B datasheet]", [*serves_build, *serves_fc, *serves_p])
        ignored = {f: "PA pin: the PA has no model; only its supply draw (VCC1 -> GND, switched by POWER_DOWN) is simulated"
                   for f in ("VBA1", "VBIAS", "RF_IN", "PDET", "VBA2", "VCC2/RFOUT")}
        b.bind("U1", card_binding(u, sup_card, sup_text, ctx.provenance(
            "U1 simulated as its supply draw model.pa.supply: model.pa.r_supply from VCC1 to GND while POWER_DOWN is low, model.pa.r_off while it is "
            "high (the PA has no model)"), ignored=ignored))
        b.leave_open("U1", "NC", "MMZ09332BT1 pins 4 / 5 are not connected")
        r1, _ = passive(b, "res_0402", "R1", r_vba, "VBA1 feed resistor")
        r2, _ = passive(b, "res_0402", "R2", r_vba, "VBA2 feed resistor")
        r3, _ = passive(b, "res_0402", "R3", r_vbias, "VBIAS feed resistor")
        c1, _ = passive(b, "cap_0402", "C1", c_rf, "VBA1 decoupling")
        c2, _ = passive(b, "cap_0402", "C2", c_rf, "VBA2 decoupling")
        c3, _ = passive(b, "cap_0402", "C3", c_rf, "VBIAS decoupling")
        c4, _ = passive(b, "cap_0402", "C4", c_rf, "VCC1 decoupling")
        c5, _ = passive(b, "cap_0603", "C5", c_bulk, "PA_5V bulk capacitor")
        c6, _ = passive(b, "cap_0402", "C6", c_rf, "RF bypass at the output choke's supply end")
        l1, l1_bind = passive(b, "ind_0603", "L1", l_ch, "VCC2 / RF output choke from PA_5V")
        c7, _ = passive(b, "cap_0402", "C7", c_blk, "PA input DC block")
        l2, l2_bind = passive(b, "ind_0604hq", "L2", l_m, "load-line L-match: series inductor at the PA", deck=False, network="pa_match", serves=serves_p)
        c8, c8_bind = passive(b, "cap_0402", "C8", c_m, "load-line L-match: shunt capacitor at the system impedance", deck=False, network="pa_match", serves=serves_z)
        c9, c9_bind = passive(b, "cap_0402", "C9", c_blk, "output DC block after the match", deck=False, network="pa_match")
        tp = b.part("testpoint", "TP1", "TestPoint", "test point: the PA's power detector PDET")
        exclude(b, "TP1", "test point: no electrical model")
        pin = lambda placed, fn: (placed.ref, placed.pin(fn))  # noqa: E731
        nb.add("PA_5V", NetKind.POWER, [*u.at("VCC1"), pin(r1, "1"), pin(r2, "1"), pin(r3, "1"), pin(c4, "1"), pin(c5, "1"), pin(c6, "1"), pin(l1, "1")],
               "the switched PA supply")
        nb.add("PA_VBA1", NetKind.ANALOG, [*u.at("VBA1"), pin(r1, "2"), pin(c1, "1")], "PA VBA1 bias feed")
        nb.add("PA_VBA2", NetKind.ANALOG, [*u.at("VBA2"), pin(r2, "2"), pin(c2, "1")], "PA VBA2 bias feed")
        nb.add("PA_VBIAS", NetKind.ANALOG, [*u.at("VBIAS"), pin(r3, "2"), pin(c3, "1")], "PA VBIAS feed")
        nb.add("PA_PD", NetKind.SIGNAL, u.at("POWER_DOWN"), "the PA's POWER_DOWN line (high = powered down [UNVERIFIED])")
        nb.add("PA_PDET", NetKind.ANALOG, [*u.at("PDET"), *tp.at("1")], "the PA's power detector output")
        nb.add("PA_IN", NetKind.RF, [pin(c7, "1")], "the PA drive from the PA-drive pad")
        nb.add("PA_RFIN", NetKind.RF, [pin(c7, "2"), *u.at("RF_IN")], "the PA's RF input after its DC block")
        nb.add("PA_OUT", NetKind.ANALOG, [*u.at("VCC2/RFOUT"), pin(l1, "2"), pin(l2, "1")], "the PA output at its load line (a lumped node, not a 50 ohm line)")
        match_out = "PA_MATCH"
        nb.add(match_out, NetKind.RF, [pin(l2, "2"), pin(c8, "1"), pin(c9, "1")], "the load-line match's system-impedance side")
        dc_out = "LPF_IN" if self.with_lpf else self.out_net
        nb.add(dc_out, NetKind.RF, [pin(c9, "2")], "after the output DC block")
        nb.add(GROUND_NET, NetKind.GROUND, [*u.at("GND"), *(pin(x, "2") for x in (c1, c2, c3, c4, c5, c6, c8))])
        b.result.networks.append(RFNetwork(
            id="pa_match", block=self.id, members=["L1", "L2", "C8", "C9"], bindings={"L1": l1_bind, "L2": l2_bind, "C8": c8_bind, "C9": c9_bind},
            loss_q={"L1": q_u[1], "L2": q_u[1]}, q_ref_hz=f_c[1],
            ports=[fixture_port("pa_out", "PA_OUT", r_l[1], "in"), fixture_port("match_out", dc_out, z0[1], "out"),
                   RFPort(name="pa_5v", net="PA_5V", kind="rail", voltage_v=v5, direction="in")],
            sweep=[ac_sweep(b, "pa_match_sweep", lin, pts, f_lo, f_hi, "pa_match: 300-600 MHz")],
            expectations=[row("s21_fc", "s21_db", "pa_out", "match_out", f_c[1], m_s21, bound="at_least"),
                          row("s11_fc", "s11_db", "pa_out", "pa_out", f_c[1], m_s11, bound="at_most")],
        ))
        chain = ["C7", "U1", "L1", "L2", "C8", "C9"]
        if self.with_lpf:
            b.result.networks.append(add_harmonic_lpf(b, nb, block_id=self.id, in_net="LPF_IN", out_net=self.out_net, cap_refs=("C10", "C11", "C12", "C13"),
                                                      ind_refs=("L3", "L4", "L5"), nodes=("LPF_N1", "LPF_N2"), f_c=f_c, z0=z0))
            chain += ["C10", "L3", "C11", "L4", "C12", "L5", "C13"]
        if self.with_output:
            j = b.part("coax_ufl", "J1", "U.FL", "conducted RF output (U.FL into an attenuator or a dummy load; never an antenna)", [*serves_build, *serves_z, *serves_p])
            exclude(b, "J1", "connector: the conducted output is a 50 ohm fixture port")
            nb.add(self.out_net, NetKind.RF, j.at("In"), "the conducted RF output")
            nb.add(GROUND_NET, NetKind.GROUND, j.at("Ext"))
            chain.append("J1")
        nb.declare(b)
        b.result.ports = [
            RFPort(name="PA_IN", net="PA_IN", kind="port", z0_ohm=pa_in[1], frequency_hz=f_c[1], direction="in"),
            RFPort(name="PA_5V", net="PA_5V", kind="rail", voltage_v=v5, direction="in"),
            RFPort(name=self.out_net, net=self.out_net, kind="port", z0_ohm=z0[1], frequency_hz=f_c[1], direction="out"),
        ]
        b.result.chain = chain
        b.result.net_classes = pa_net_classes(with_lpf=self.with_lpf, out_net=self.out_net)
        where = "at the conducted output" if self.with_output else f"at the radio's output after {self.out_net} (the block that holds the harmonic low-pass)"
        b.result.lab_items += [
            LabItem(id="pa_power", block=self.id,
                    what=(f"conducted output power into the dummy load (designed for {float(p_out[1].value):g} W on the load line)" if self.with_output else
                          f"output power {where} (designed for {float(p_out[1].value):g} W on the load line)"),
                    instruments=["power meter", "attenuator", "dummy load"], reason="the PA has no model: its power, gain and efficiency are [UNVERIFIED]"),
            LabItem(id="pa_harmonics", block=self.id, what=f"harmonics {where} (2 f_c, 3 f_c) against kr447.spurious_max",
                    instruments=["spectrum analyser"],
                    reason=("the lpf rows are network verdicts" if self.with_lpf else "the low-pass rows (lpf, another block's) are network verdicts")
                    + "; the PA's harmonic levels and the board's ground returns are not modelled"),
            LabItem(id="pa_stability", block=self.id, what="PA stability into mismatch and the key-up transient",
                    instruments=["spectrum analyser", "VNA", "oscilloscope"], reason="large-signal PA behaviour is not modelled"),
            LabItem(id="pa_thermal", block=self.id, what="PA and TX_5V regulator temperature at continuous transmission",
                    instruments=["thermal camera"], reason="theta_JA and the dissipations are [UNVERIFIED] (domain.power.thermal / component.fit stay NOT_VERIFIED)"),
        ]
        return b.done()


__all__ = [
    "PA_ID",
    "PA_NETWORKS",
    "PA_PD_THRESHOLD",
    "PA_SUPPLY",
    "PA_SUPPLY_OFF",
    "P_OUT_RANGE_W",
    "PaBlock",
    "add_harmonic_lpf",
    "pa_net_classes",
]
