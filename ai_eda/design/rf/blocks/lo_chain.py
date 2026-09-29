"""The first local oscillator of the KR 447 MHz FM receiver (kr447 design §2.3, decision 1B, part P11): TCXO -> x3 -> x2 -> x2 + LO BPF -> PHA-1 -> pad.

Invariant: every part comes from the kr447 parts table by library name, every
number is a registered calculator's output over the blocks' confirmed
choices, the ``model.*`` values (UNVERIFIED) and the confirmed requirements,
and every passive network between the excluded / unmodelled parts is an RF
fixture judged on ngspice. Nothing here says that the multipliers multiply or
that the LO reaches +7 dBm: a class-C harmonic generator is not modelled
(``model.npn`` is a generic card biased at an operating point), the TCXO and
the PHA-1 are excluded, and a fixture PASS is "a network verdict under
confirmed model values (not a measured part)".

The frequency plan (decision 1B): LO1 = f_c - IF1 (``rf.lo1``, low side),
N = ``rf.n_mult`` = 12 = 3 x 2 x 2 (``lo.mult.1`` .. ``.3``, refused unless
their product is N), the reference f_R = LO1 / N (``calc.clock.divided``:
35.5135417 MHz for the 447.5625 MHz channel, inside the KT2520K-T library
range "10-60MHz" - a custom frequency whose availability and stability are
lab / procurement items), the stage outputs f_1 / f_2 / f_3 = 3 / 6 / 12 f_R
(``calc.rf.mult.stage``: 106.54 / 213.08 / 426.16 MHz).

Two blocks, because a shield can holds every part of its block:

* :class:`LoChainBlock` (``lo_chain``, local references 1..49, re-based to
  7xx) - under the can ``SH1``: the TCXO ``Y1`` (``RX_3V3``, decoupling
  ``C1``, coupling ``C2``), three BFR92 stages ``Q1`` .. ``Q3`` (divider
  bias, emitter R // C, 0 ohm link + decoupling, collector choke - the
  front end's :func:`~ai_eda.design.rf.blocks.rx_frontend.bias_stage`); the
  x3 and x6 collectors each drive a double-tuned (2-resonator) top-C tank
  ``lo_tank1`` / ``lo_tank2`` with end external Q ``lo.tank_qe`` = 20
  (bandwidth ``calc.rf.resonator.top_c.bw_for_qe``) into the next stage's
  base, the x12 collector the 4-resonator output band-pass ``lo_bpf`` at
  LO1 straight into ``LO1_RAW`` - the PHA-1's input port
  ``model.pha1.port_r`` (the design's last tank and its 2-pole LO band-pass
  designed as one network: two filters joined tap to tap have no resistive
  node between them, so their two fixtures, each between its own port
  models, did not measure the cascade);
* :class:`LoBufferBlock` (``lo_buffer``, local references 50..99, re-based to
  7xx): the PHA-1 gain block ``U50`` on ``LO1_RAW`` (bias through the choke
  ``L52`` from ``RX_5V`` into RF_OUT, output DC block ``C56``) and the
  matched pi pad ``lo_pad`` ``R50`` / ``R51`` / ``R52``
  (``calc.rf.attenuator.pi.*``, ``lo.pad.a_db``) into ``LO1_MIX``, the
  ADEX-10's LO port.

Every network is designed between *loaded* ports, and every part on its port
nets is a member of its fixture: the source is the collector port model
``model.bfr92.r_out`` (1 kohm) in parallel with the stage's collector choke
(its Q at the stage frequency, returned to AC ground through the 0 ohm link,
the fixture's rail port ``rx_5v`` on ``RX_5V``; the feed decoupling between
link and choke is a member too - across the 0 V link it changes nothing in
the fixture, but it is a part on the network's node), whose reactance the input
tap absorbs (``calc.rf.resonator.top_c.port_r`` / ``.port_x`` /
``.c_tap_reactive``); the load of a tank is the next stage's port model
``model.bfr92.r_in`` (500 ohm) in parallel with that stage's base divider
(``lo.<next>.r_div``, ``lo.tank<k>.r_load_eff``). The inductances are
choices (``lo.tank1.l`` 150 nH, ``.tank2.l`` 68 nH, the band-pass
``lo.bpf.l`` 27 nH) such that the capacitive taps can transform: a tap only
transforms down, so R_p = Q_e w0 L must exceed the loaded collector port
(the design's 50 ohm example values - 30 pF below 300 MHz, 4.7 nH above -
give R_p 948 / 474 / 264 ohm, which ``calc.rf.resonator.top_c`` refuses;
kr447 design §2.7 critic2 (c)). Every row's nominal is the exact ported
network (``.ported_s21_db`` at f_k; ``.ported_rel_s21_db`` at f_k -/+ f_R,
the neighbouring lines of the chain, ``calc.rf.mult.spur``), tol_abs
``lo.net_tol``: at the default choices (Q_u 40) the tanks pass -5.50 /
-5.55 dB (Cohn's 4.34 dB counts the resonators alone) and reject - / + f_R
by 47.79 / 26.31 dB (x3, 106.54 MHz) and 29.05 / 17.65 dB (x2, 213.08 MHz);
``lo_bpf`` (``lo.bpf.n`` 4, ``lo.bpf.bw`` 30 MHz) passes -8.01 dB at LO1
and rejects LO1 -/+ f_R by 27.94 / 19.22 dB and -/+ 2 f_R by 56.15 /
37.72 dB. The symmetric narrowband formulas are only ceilings (the + side
of a top-C network is the weak one). The networks carry no trimmers: the
Murata TZB4-A trimmer's footprint (7.5 x 4.5 mm) would take more than half
of the BMI-S-103 fence for six of them; alignment is by part selection and
is a lab item (``tank_alignment``).

The LO-spur responses (the LO's residual lines at LO1 + k f_R converted by
the mixer, ``calc.rf.superhet.lo_spur_response``): 440.28 MHz (k = +1, - IF1:
the image + f_R, 7.29 MHz below the carrier - the closest), 475.79 (k = +2,
- IF1), 412.05 (k = -1, + IF1) and 483.08 MHz (k = +1, + IF1) are ``response``
rows of the frequency plan: no verdict of their own, they point to the tank /
BPF rows and the lab item ``lo_spur_response``. The birdie rows are
arithmetic margins against the profile's channel raster: the f_R harmonic
nearest the carrier (13 f_R, 14.11 MHz above it) and f_R against IF1.

The design deck is the three stages' bias (``ic_x3``, ``ic_x6``,
``ic_x12``) under ``model.npn``. Plan values shared with other blocks
(``rf.*``, ``lo.*`` of the frequency plan) are taken from
:attr:`BlockContext.shared` when a composing template wrote them; the rails'
levels ``power.rx_5v`` / ``power.rx_3v3`` must be shared by the supply block.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import NetKind, SpiceDevice, Traced
from ai_eda.ir.rf import LabItem, PlanLine, RFNetwork, RFPort
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import clock_divided, parallel_resistance, voltage_divider_ratio
from ai_eda.tools.calc.rf import ppm_offset, voltage_ratio_to_db

from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.rx_frontend import (
    PortLoad,
    ADEX_PORT,
    RASTER_KEY,
    NetBook,
    RadioPlan,
    ac_sweep,
    bias_analysis,
    bias_stage,
    bias_values,
    build_top_c,
    common_bias,
    copy_input,
    exclude,
    fixture_port,
    index_choices,
    model_q,
    model_value,
    part_value,
    plan_value,
    radio_plan,
    rail_level,
    requirement_ids,
    row,
    two_terminal,
)
from ai_eda.design.rf.common import N_MULT, N_MULT_KEY, N_MULT_TEXT
from ai_eda.design.rf.models import ModelValue, npn_card

#: block ids
CHAIN_ID = "lo_chain"
BUFFER_ID = "lo_buffer"
#: the interface nets each block keeps (never prefixed)
CHAIN_INTERFACE: tuple[str, ...] = ("LO1_RAW", "RX_5V", "RX_3V3")
BUFFER_INTERFACE: tuple[str, ...] = ("LO1_RAW", "LO1_MIX", "RX_5V")
#: the fixture networks each block declares
CHAIN_NETWORKS: tuple[str, ...] = ("lo_tank1", "lo_tank2", "lo_bpf")
BUFFER_NETWORKS: tuple[str, ...] = ("lo_pad",)
#: the multiplier of each stage (decision 1B: 3 x 2 x 2 = 12)
STAGE_MULTIPLIERS: tuple[int, ...] = (3, 2, 2)
#: the default tank inductances (H) of the x3 and x6 stages' tanks: R_p = Q_e w0 L above the collector port with its choke (module docstring)
TANK_L: tuple[float, ...] = (150e-9, 68e-9)
#: the default collector chokes of the three stages (H)
STAGE_CHOKES: tuple[float, ...] = (4.7e-6, 1e-6, 470e-9)

PHA1_PORT = ModelValue("model.pha1.port_r", 50.0, "ohm", "input and output resistance of the PHA-1 LO buffer (excluded from every netlist)",
                       "Mini-Circuits PHA-1 datasheet")


@dataclass
class LoPlan:
    """The LO frequency plan (key, traced): N, the stage multipliers, f_R, the stage outputs, the orders used by the rows."""

    radio: RadioPlan
    n_mult: tuple[str, Traced]
    mults: list[tuple[str, Traced]]
    f_ref: tuple[str, Traced]
    stages: list[tuple[str, Traced]]
    orders: dict[str, tuple[str, Traced]]


def lo_plan(b: BlockBuilder) -> LoPlan:
    """``rf.n_mult``, ``lo.mult.<k>``, ``lo.f_ref`` (LO1 / N), ``lo.f<k>`` (the stage outputs) and the orders -/+1, -/+2 - shared first."""
    p = radio_plan(b)
    n = plan_value(b, N_MULT_KEY, lambda: b.choice(N_MULT_KEY, N_MULT, None, N_MULT_TEXT))  # the family's one row (ai_eda.design.rf.common)
    mults = []
    for k, m in enumerate(STAGE_MULTIPLIERS, start=1):
        key = f"lo.mult.{k}"
        mults.append((key, plan_value(b, key, lambda k=k, m=m, key=key: b.choice(key, float(m), None, f"LO multiplier stage {k}: x{m} (decision 1B: 3 x 2 x 2)"))))
    product = 1.0
    for _, t in mults:
        product *= float(t.value)
    if product != float(n.value):
        raise TemplateRefusal(f"the LO stage multipliers {[float(t.value) for _, t in mults]} multiply to {product:g}, not rf.n_mult = {float(n.value):g}")
    f_ref = plan_value(b, "lo.f_ref", lambda: b.computed("lo.f_ref", clock_divided(p.lo1[1], n, (p.lo1[0], "rf.n_mult"))))
    stages: list[tuple[str, Traced]] = []
    prev = ("lo.f_ref", f_ref)
    for k, (mkey, m) in enumerate(mults, start=1):
        key = f"lo.f{k}"
        prev = (key, plan_value(b, key, lambda prev=prev, m=m, mkey=mkey, key=key: b.computed(key, radio.mult_stage(prev[1], m, (prev[0], mkey)))))
        stages.append(prev)
    orders: dict[str, tuple[str, Traced]] = {}
    for name, value, text in (("m1", -1.0, "order -1 (the lower neighbour: f - x)"), ("p1", 1.0, "order +1 (the upper neighbour: f + x)"),
                              ("m2", -2.0, "order -2 (f - 2 x)"), ("p2", 2.0, "order +2 (f + 2 x)")):
        key = f"lo.order.{name}"
        orders[name] = (key, plan_value(b, key, lambda key=key, value=value, text=text: b.choice(key, value, None, text)))
    return LoPlan(radio=p, n_mult=("rf.n_mult", n), mults=mults, f_ref=("lo.f_ref", f_ref), stages=stages, orders=orders)


def _spur(b: BlockBuilder, key: str, f: tuple[str, Traced], x: tuple[str, Traced], k: tuple[str, Traced]) -> tuple[str, Traced]:
    return key, plan_value(b, key, lambda: b.computed(key, radio.mult_spur(f[1], x[1], k[1], (f[0], x[0], k[0]))))


def _net_tol(b: BlockBuilder) -> Traced:
    return plan_value(b, "lo.net_tol", lambda: b.choice("lo.net_tol", 1.0, "dB",
                                                        "tolerance of every exact-network fixture row of the LO chain (the netlist realises the designed network)"))


class LoChainBlock(Block):
    """TCXO -> x3 -> x2 -> x2 with a double-tuned top-C tank after each stage, under one shield can (see the module docstring)."""

    id = CHAIN_ID
    title = "LO1 chain: TCXO, x3 / x2 / x2 BFR92 multipliers with double-tuned tanks (under the shield can)"
    interface_nets = CHAIN_INTERFACE

    def __init__(self, shield: str = "shield_103") -> None:
        self.shield = shield

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_tol = requirement_ids(ctx, "frequency_tolerance")
        try:
            lp = lo_plan(b)
            p = lp.radio
            rail5_key, rail5 = rail_level(b, "RX_5V")
            rail3_key, rail3 = rail_level(b, "RX_3V3")
            raster = ctx.shared.get(RASTER_KEY)
            if raster is None:
                raise TemplateRefusal(f"block {self.id} reads the profile's {RASTER_KEY} from the composing template (BlockContext.shared), which did not write it")
            npn_text = b.card(npn_card())
            r_out = model_value(b, "model.bfr92.r_out")
            r_in = model_value(b, "model.bfr92.r_in")
            pha1 = model_value(b, PHA1_PORT)
            tol = _net_tol(b)
            idx = index_choices(b, "lo", 4)
            qe = ("lo.tank_qe", plan_value(b, "lo.tank_qe", lambda: b.choice(
                "lo.tank_qe", 20.0, None, "end external Q of every double-tuned multiplier tank (decision 1B: 4.21 dB network loss at Q_u 40; Q_e 25 buys about "
                "3 dB more rejection per stage for 1 dB more loss)")))
            n_t = ("lo.tank.n", b.choice("lo.tank.n", 2.0, None, "resonators per multiplier tank (double-tuned)"))
            n_b = ("lo.bpf.n", b.choice("lo.bpf.n", 4.0, None, (
                "resonators of the x12 stage's LO output band-pass: the design's double-tuned last tank and its 2-pole band-pass as one 4-resonator "
                "network (two filters joined by their taps, with no resistive node between them, gave 9 dB less rejection at LO1 + f_R than their two "
                "fixtures claimed)")))
            bw_b = ("lo.bpf.bw", b.choice("lo.bpf.bw", 30e6, "Hz", "LO output band-pass bandwidth (4 resonators, 30 MHz: about the loss of the design's tank + "
                                                                   "band-pass pair, 4 dB more rejection on the weak + side)"))
            l_b = ("lo.bpf.l", b.choice("lo.bpf.l", 27e-9, "H", ("LO output band-pass resonator inductance (0604HQ): R_p = Q_e w0 L must exceed the "
                                                               "collector port's resistance with its choke - a tap only transforms down")))
            stab_in = ctx.inputs.get("frequency_tolerance")
            if stab_in is not None:
                stab = ("rf.frequency_tolerance", copy_input(b, "rf.frequency_tolerance", stab_in.traced))
            else:
                stab = ("rf.tcxo_stability", plan_value(b, "rf.tcxo_stability", lambda: b.choice(
                    "rf.tcxo_stability", 2.5, "ppm", "TCXO stability specified for procurement (no frequency_tolerance stated; the profile's 2.5 ppm placeholder) "
                    "[UNVERIFIED: Kyocera KT2520K datasheet]")))
            lo_err = b.computed("lo.lo1_error", ppm_offset(p.lo1[1], stab[1], (p.lo1[0], stab[0])))
            c_tcxo = b.choice("lo.c_tcxo", 100e-9, "F", "TCXO supply decoupling [UNVERIFIED: Kyocera KT2520K application note]")
            c_in = b.choice("lo.c_in", 100e-12, "F", "coupling capacitor TCXO -> first multiplier base (45 ohm at 35.5 MHz, against its 1.5 kohm divider)")
            common = common_bias(b, "lo")
            bias_analysis(b)
            # ---- sweep
            lin = b.choice("lo.sweep.lin", "lin", None, "linear ac sweep variation")
            pts = b.choice("lo.sweep.points", 401, None, "points of each tank's sweep (rows are read at their own point analyses, the sweep is the figure)")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        # ---- TCXO
        y = b.part("tcxo", "Y1", part_value(lp.f_ref[1].value),
                   f"LO1 reference TCXO: {float(lp.f_ref[1].value) / 1e6:.9g} MHz = LO1 / N (a custom frequency inside the KT2520K-T range; availability and "
                   f"stability {float(stab[1].value):g} ppm [UNVERIFIED: Kyocera KT2520K datasheet])", [*serves_fc, *serves_tol])
        exclude(b, "Y1", "KT2520K-T TCXO has no SPICE model: excluded (its output drives the first multiplier through the coupling capacitor)")
        b.leave_open("Y1", "NC", "KT2520K-T pins 2 / 5 are not connected (library type no_connect)")
        c1 = b.part("cap_0402", "C1", part_value(c_tcxo.value), "TCXO supply decoupling")
        c2 = b.part("cap_0402", "C2", part_value(c_in.value), "coupling capacitor TCXO -> first multiplier")
        b.bind("C1", two_terminal(b, c1, SpiceDevice.C, c_tcxo, "decoupling at its value"))
        b.bind("C2", two_terminal(b, c2, SpiceDevice.C, c_in, "coupling at its value"))
        nb.add("RX_3V3", NetKind.POWER, [*y.at("VCC"), ("C1", c1.pin("1"))], "RX_3V3 rail (TCXO supply)")
        nb.add("LO_TCXO", NetKind.RF, [*y.at("OUT"), ("C2", c2.pin("1"))], "TCXO output")
        nb.add(GROUND_NET, NetKind.GROUND, [*y.at("GND"), ("C1", c1.pin("2"))])
        # ---- stages and their networks
        stage_refs = (
            {"q": "Q1", "r_b1": "R1", "r_b2": "R2", "r_e": "R3", "c_e": "C3", "link": "R4", "c_dec": "C4", "choke": "L1"},
            {"q": "Q2", "r_b1": "R5", "r_b2": "R6", "r_e": "R7", "c_e": "C10", "link": "R8", "c_dec": "C11", "choke": "L4"},
            {"q": "Q3", "r_b1": "R9", "r_b2": "R10", "r_e": "R11", "c_e": "C17", "link": "R12", "c_dec": "C18", "choke": "L7"},
        )
        net_refs = (
            (["C5", "C6", "C7", "C8", "C9"], ["L2", "L3"]),
            (["C12", "C13", "C14", "C15", "C16"], ["L5", "L6"]),
            ([f"C{i}" for i in range(19, 28)], ["L8", "L9", "L10", "L11"]),
        )
        names = ("x3", "x6", "x12")
        base_net = "LO_X3_B"
        base_extra = [("C2", c2.pin("2"))]
        chain = ["Y1", "C2"]
        rows_by_tank: list[tuple[str, list]] = []
        rail_port = RFPort(name="rx_5v", net="RX_5V", kind="rail", voltage_v=rail5, direction="in")
        try:
            # the three stages' bias first: a network's fixture holds the next stage's base divider, so its values are needed before the network
            vals_by = []
            for k in range(3):
                m = int(float(lp.mults[k][1].value))
                f_k = lp.stages[k][1]
                what = f"LO multiplier stage {k + 1} Q{k + 1} (BFR92, x{m} to {float(f_k.value) / 1e6:.6g} MHz)"
                vals_by.append((what, bias_values(b, "lo", names[k], what, r_b1=4700.0, r_b2=2200.0, r_e=330.0, c_e=1e-9, l_choke=STAGE_CHOKES[k],
                                                  choke_note=(f"its reactance at {float(f_k.value) / 1e6:.4g} MHz is of the order of the 1 kohm collector port, "
                                                              "so it is part of the network: a member of the stage's fixture, absorbed by the network's input tap"),
                                                  common=common, rail_key=rail5_key, rail=rail5)))
            f3_key, f3 = lp.stages[-1]
            q_u = model_q(b, float(f3.value))
            for k in range(3):
                name = names[k]
                f_key, f_k = lp.stages[k]
                what, vals = vals_by[k]
                refs = stage_refs[k]
                st = bias_stage(b, vals, q_ref=refs["q"], refs={x: refs[x] for x in ("r_b1", "r_b2", "r_e", "c_e", "link", "c_dec", "choke")},
                                nets={"emitter": f"LO_{name.upper()}_E", "feed": f"LO_{name.upper()}_VC"}, what=what, exp_id=f"ic_{name}", npn_text=npn_text,
                                netbook=nb, serves=serves_fc)
                nb.add(base_net, NetKind.RF, [*base_extra, *st.base_pins], f"stage {k + 1} base: the previous network's output, the base and its divider")
                collector = f"LO_{name.upper()}_C"
                caps, inds = net_refs[k]
                if k < 2:
                    q_k = model_q(b, float(f_k.value))
                    nxt_key, nxt = names[k + 1], vals_by[k + 1][1]
                    div = b.computed(f"lo.{nxt_key}.r_div", parallel_resistance(nxt.r_b1[1], nxt.r_b2[1], (nxt.r_b1[0], nxt.r_b2[0])))
                    r_eff = (f"lo.tank{k + 1}.r_load_eff", b.computed(f"lo.tank{k + 1}.r_load_eff", parallel_resistance(r_in[1], div, (r_in[0], f"lo.{nxt_key}.r_div"))))
                    port = PortLoad(l_port=vals.l_choke, q_port=q_k, r_load_eff=r_eff)
                    l_key = f"lo.tank{k + 1}.l"
                    l_k = (l_key, b.choice(l_key, TANK_L[k], "H", f"lo_tank{k + 1}: resonator inductance (R_p = Q_e w0 L must exceed the collector port's "
                                                               "resistance with its choke: a tap only transforms down)"))
                    bw_key = f"lo.tank{k + 1}.bw"
                    bw_k = (bw_key, b.computed(bw_key, radio.top_c_bw_for_qe(n_t[1], f_k, qe[1], (n_t[0], f_key, qe[0]))))
                    net_id = f"lo_tank{k + 1}"
                    top = build_top_c(b, net_id, what=f"lo_tank{k + 1} (double-tuned tank at {float(f_k.value) / 1e6:.6g} MHz)", n=n_t, f0=(f_key, f_k), bw=bw_k,
                                      l=l_k, r_source=r_out, r_load=r_in, idx=idx, cap_refs=caps, ind_refs=inds,
                                      nodes=[f"LO_T{k + 1}_R1", f"LO_T{k + 1}_R2"], ind_part="ind_0603" if f_k.value < 300e6 else "ind_0604hq", netbook=nb, port=port)
                    f_m = _spur(b, f"lo.tank{k + 1}.f_m", (f_key, f_k), lp.f_ref, lp.orders["m1"])
                    f_p = _spur(b, f"lo.tank{k + 1}.f_p", (f_key, f_k), lp.f_ref, lp.orders["p1"])
                    s21 = b.computed(f"{net_id}.s21", top.s21(q_k[1], q_k[0], f_k, f_key))
                    rel_m = b.computed(f"{net_id}.rel_m", top.rel(q_k[1], q_k[0], f_m[1], f_m[0], f_k, f_key))
                    rel_p = b.computed(f"{net_id}.rel_p", top.rel(q_k[1], q_k[0], f_p[1], f_p[0], f_k, f_key))
                    b.computed(f"{net_id}.loss", radio.bpf_dissipation_loss(n_t[1], f_k, bw_k[1], q_k[1], (n_t[0], f_key, bw_key, q_k[0])))
                    f_lo = b.choice(f"lo.tank{k + 1}.sweep.fstart", round(float(f_k.value) * 0.5 / 1e6) * 1e6, "Hz", f"start of lo_tank{k + 1}'s sweep (about f_{k + 1} / 2)")
                    f_hi = b.choice(f"lo.tank{k + 1}.sweep.fstop", round(float(f_k.value) * 1.5 / 1e6) * 1e6, "Hz", f"stop of lo_tank{k + 1}'s sweep (about 1.5 f_{k + 1})")
                    out_net = f"LO_{names[k + 1].upper()}_B"
                    nxt_refs = stage_refs[k + 1]
                    b.result.networks.append(RFNetwork(
                        id=net_id, block=self.id, members=[*top.members, refs["choke"], refs["link"], refs["c_dec"], nxt_refs["r_b1"], nxt_refs["r_b2"]], bindings=top.bindings,
                        loss_q={**{r: q_k[1] for r in top.inductors}, refs["choke"]: q_k[1]}, q_ref_hz=f_k,
                        ports=[fixture_port("coll", collector, r_out[1], "in"), fixture_port("next", out_net, r_in[1], "out"), rail_port],
                        sweep=[ac_sweep(b, f"{net_id}_sweep", lin, pts, f_lo, f_hi, f"lo_tank{k + 1}: f_{k + 1} / 2 .. 1.5 f_{k + 1}")],
                        expectations=[row("s21", "s21_db", "coll", "next", f_k, s21, tol=tol), row("rel_m", "rel_s21_db", "coll", "next", f_m[1], rel_m, tol=tol, ref_at=f_k),
                                      row("rel_p", "rel_s21_db", "coll", "next", f_p[1], rel_p, tol=tol, ref_at=f_k)],
                    ))
                    rows_by_tank.append((net_id, [f_m, f_p]))
                    chain += [refs["q"], refs["choke"], *caps[:1], inds[0], caps[1], caps[2], inds[1], caps[3], caps[4]]
                    base_net = out_net
                else:
                    port = PortLoad(l_port=vals.l_choke, q_port=q_u, r_load_eff=pha1)
                    top = build_top_c(b, "lo_bpf", what="lo_bpf (the LO x12 stage's 4-resonator output band-pass at LO1)", n=n_b, f0=(f3_key, f3), bw=bw_b,
                                      l=l_b, r_source=r_out, r_load=pha1, idx=idx, cap_refs=caps, ind_refs=inds, nodes=[f"LO_B_R{i}" for i in range(1, 5)],
                                      ind_part="ind_0604hq", netbook=nb, port=port)
                    f_rows = {nm: _spur(b, key, (f3_key, f3), lp.f_ref, lp.orders[o])
                              for nm, key, o in (("m1", "lo.bpf.f_m1", "m1"), ("p1", "lo.bpf.f_p1", "p1"), ("m2", "lo.bpf.f_m2", "m2"), ("p2", "lo.bpf.f_p2", "p2"))}
                    s21 = b.computed("lo_bpf.s21", top.s21(q_u[1], q_u[0], f3, f3_key))
                    rel = {nm: b.computed(f"lo_bpf.rel_{nm}", top.rel(q_u[1], q_u[0], f[1], f[0], f3, f3_key)) for nm, f in f_rows.items()}
                    b.computed("lo_bpf.loss", radio.bpf_dissipation_loss(n_b[1], f3, bw_b[1], q_u[1], (n_b[0], f3_key, bw_b[0], q_u[0])))
                    f_lo = b.choice("lo.bpf.sweep.fstart", 300e6, "Hz", "start of the LO output band-pass sweep")
                    f_hi = b.choice("lo.bpf.sweep.fstop", 600e6, "Hz", "stop of the LO output band-pass sweep")
                    b.result.networks.append(RFNetwork(
                        id="lo_bpf", block=self.id, members=[*top.members, refs["choke"], refs["link"], refs["c_dec"]], bindings=top.bindings,
                        loss_q={**{r: q_u[1] for r in top.inductors}, refs["choke"]: q_u[1]}, q_ref_hz=f3,
                        ports=[fixture_port("coll", collector, r_out[1], "in"), fixture_port("buf_in", "LO1_RAW", pha1[1], "out"), rail_port],
                        sweep=[ac_sweep(b, "lo_bpf_sweep", lin, pts, f_lo, f_hi, "LO output band-pass: 300-600 MHz")],
                        expectations=[row("s21_lo1", "s21_db", "coll", "buf_in", f3, s21, tol=tol),
                                      *[row(f"rel_{nm}", "rel_s21_db", "coll", "buf_in", f_rows[nm][1], rel[nm], tol=tol, ref_at=f3) for nm in ("m1", "p1", "m2", "p2")]],
                    ))
                    chain += [refs["q"], refs["choke"], caps[0], *[x for i in range(4) for x in (inds[i], caps[1 + 2 * i], caps[2 + 2 * i])]]
                nb.add(collector, NetKind.RF, [*st.collector_pins, *top.in_pins], f"stage {k + 1} collector: the choke and the network's input tap")
                nb.add(GROUND_NET, NetKind.GROUND, top.gnd_pins)
                base_extra = top.out_pins
            nb.add("LO1_RAW", NetKind.RF, base_extra, "the LO x12 stage's output band-pass into the PHA-1 buffer's input")
            # ---- plan rows: LO-spur responses and birdie margins
            k_birdie = b.choice("lo.birdie.k", float(round(float(p.f_c[1].value) / float(lp.f_ref[1].value))), None,
                                "the harmonic of f_R nearest the carrier (round(f_c / f_R)); its margin row checks that it stays off the channel")
            birdie = b.computed("lo.birdie.f", radio.harmonic(lp.f_ref[1], k_birdie, (lp.f_ref[0], "lo.birdie.k")))
            responses = {}
            for rid, k_name, s_name in (("lo_spur_p1m", "p1", "m1"), ("lo_spur_p2m", "p2", "m1"), ("lo_spur_m1p", "m1", "p1"), ("lo_spur_p1p", "p1", "p1")):
                k_o, s_o = lp.orders[k_name], lp.orders[s_name]
                responses[rid] = b.computed(f"lo.{rid}", radio.superhet_lo_spur_response(p.lo1[1], lp.f_ref[1], k_o[1], p.if1[1], s_o[1],
                                                                                         (p.lo1[0], lp.f_ref[0], k_o[0], p.if1[0], s_o[0])))
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        sh = b.part(self.shield, "SH1", {"shield_102": "BMI-S-102", "shield_103": "BMI-S-103", "shield_105": "BMI-S-105"}.get(self.shield, "RFShield"),
                    "shield can over the LO1 multiplier chain and its output band-pass")
        exclude(b, "SH1", "shield can: no electrical model (its fence pads are GND)")
        nb.add(GROUND_NET, NetKind.GROUND, sh.at("Shield"))
        nb.declare(b)
        b.result.ports = [
            RFPort(name="LO1_RAW", net="LO1_RAW", kind="port", z0_ohm=pha1[1], frequency_hz=lp.stages[-1][1], direction="out"),
            RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=rail5, direction="in"),
            RFPort(name="RX_3V3", net="RX_3V3", kind="rail", voltage_v=rail3, direction="in"),
        ]
        b.result.chain = chain
        b.result.shield_ref = "SH1"
        tank_p = [f"spice.rf.{n}.rel_p" for n, _ in rows_by_tank]
        tank_m = [f"spice.rf.{n}.rel_m" for n, _ in rows_by_tank]
        lab = "rf.lab.lo_spur_response"
        b.result.plan_lines += [
            PlanLine(id="lo_spur_p1m", kind="response", f_hz=responses["lo_spur_p1m"], ref_hz=p.f_c[1],
                     points_to=[*tank_p, "spice.rf.lo_bpf.rel_p1", "spice.rf.fe_bpf3", lab],
                     note="LO1 + f_R - IF1 = the image + f_R, the closest LO-spur response: the front end rejects it by only a few dB, the LO chain's + side carries it"),
            PlanLine(id="lo_spur_p2m", kind="response", f_hz=responses["lo_spur_p2m"], ref_hz=p.f_c[1], points_to=["spice.rf.lo_bpf.rel_p2", "spice.rf.fe_bpf3", lab],
                     note="LO1 + 2 f_R - IF1"),
            PlanLine(id="lo_spur_m1p", kind="response", f_hz=responses["lo_spur_m1p"], ref_hz=p.f_c[1],
                     points_to=[*tank_m, "spice.rf.lo_bpf.rel_m1", "spice.rf.fe_bpf3", lab], note="LO1 - f_R + IF1"),
            PlanLine(id="lo_spur_p1p", kind="response", f_hz=responses["lo_spur_p1p"], ref_hz=p.f_c[1],
                     points_to=[*tank_p, "spice.rf.lo_bpf.rel_p1", "spice.rf.fe_bpf3", lab], note="LO1 + f_R + IF1"),
            PlanLine(id="birdie_fr_harmonic", kind="margin", f_hz=birdie, ref_hz=p.f_c[1], min_margin_hz=raster,
                     note="the f_R harmonic nearest the carrier must stay at least one channel raster away (a birdie in the channel otherwise)"),
            PlanLine(id="birdie_fr_if1", kind="margin", f_hz=lp.f_ref[1], ref_hz=p.if1[1], min_margin_hz=raster,
                     note="the TCXO frequency must stay at least one channel raster away from IF1"),
        ]
        b.result.lab_items += [
            LabItem(id="lo_spur_response", block=self.id,
                    what="LO-spur responses of the receiver (the LO's residual lines LO1 -/+ k f_R at the ADEX-10 LO port and the responses they make, the closest at the image + f_R)",
                    instruments=["spectrum analyser", "two signal generators", "SINAD meter"],
                    reason="the fixtures judge each tank and the output band-pass (collector to the PHA-1's input, as built) as a network; the multipliers' "
                           "spur generation and PM regrowth are not modelled"),
            LabItem(id="lo_frequency", block=self.id,
                    what=f"LO1 frequency and its drift (the TCXO's {float(stab[1].value):g} ppm is {float(lo_err.value):.4g} Hz at LO1)",
                    instruments=["frequency counter", "temperature chamber"], reason="the TCXO's stability is a datasheet fact of a custom part [UNVERIFIED]"),
            LabItem(id="tank_alignment", block=self.id,
                    what="alignment of the two double-tuned tanks and the 4-resonator LO output band-pass (no trimmers: part selection; couplings of about 0.2 pF)",
                    instruments=["spectrum analyser", "VNA"], reason="parts tolerances and board parasitics detune the designed networks"),
            LabItem(id="procurement_rx_tcxo", block=self.id,
                    what=f"availability of the custom {float(lp.f_ref[1].value) / 1e6:.9g} MHz TCXO (do not order before the channel is confirmed from the official text)",
                    instruments=[], reason="custom frequency; the channel itself is an unverified placeholder of the KR 447 MHz profile"),
        ]
        return b.done()


class LoBufferBlock(Block):
    """``LO1_RAW`` -> PHA-1 -> DC block -> matched pi pad ``lo_pad`` -> ``LO1_MIX`` (see the module docstring; the LO band-pass is the chain's)."""

    id = BUFFER_ID
    title = "LO1 PHA-1 buffer and pad to the mixer's LO port"
    interface_nets = BUFFER_INTERFACE

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        try:
            lp = lo_plan(b)
            p = lp.radio
            rail5_key, rail5 = rail_level(b, "RX_5V")
            pha1 = model_value(b, PHA1_PORT)
            adex = model_value(b, ADEX_PORT)
            f3 = lp.stages[-1][1]
            l_ch = b.choice("lo.buf.l_choke", 470e-9, "H", "PHA-1 bias choke from RX_5V into RF_OUT (about 1.26 kohm at LO1) [UNVERIFIED: PHA-1 application circuit, self-resonance]")
            c_out = b.choice("lo.buf.c_out", 100e-12, "F", "PHA-1 output DC block (3.7 ohm at LO1)")
            c_dec = b.choice("lo.buf.c_dec", 1e-9, "F", "RX_5V decoupling at the PHA-1 choke")
            a_db = b.choice("lo.pad.a_db", 6.0, "dB", "LO pad attenuation (sets the ADEX-10's LO drive: about +7 dBm wanted; re-chosen after the LO level is measured - "
                                                      "a human design change) [UNVERIFIED: PHA-1 output level]")
            r_sh = b.computed("lo.pad.r_shunt", radio.attenuator_pi_r_shunt(adex[1], a_db, (adex[0], "lo.pad.a_db")))
            r_se = b.computed("lo.pad.r_series", radio.attenuator_pi_r_series(adex[1], a_db, (adex[0], "lo.pad.a_db")))
            r_par = b.computed("lo.pad.r_par", parallel_resistance(r_sh, adex[1], ("lo.pad.r_shunt", adex[0])))
            ratio = b.computed("lo.pad.ratio", voltage_divider_ratio(r_se, r_par, ("lo.pad.r_series", "lo.pad.r_par")))
            s21_pad = b.computed("lo_pad.s21", voltage_ratio_to_db(ratio, ("lo.pad.ratio",)))
            pad_tol = b.choice("lo.pad.tol", 0.2, "dB", "tolerance of the pad's S21 row")
            s11_max = b.choice("lo.pad.s11_max", -20.0, "dB", "largest S11 of the matched pad")
            lin = b.choice("lo.sweep.lin", "lin", None, "linear ac sweep variation")
            pts = b.choice("lo.sweep.points", 401, None, "points of each tank's sweep (rows are read at their own point analyses, the sweep is the figure)")
            f_lo = b.choice("lo.pad.sweep.fstart", 300e6, "Hz", "start of the LO pad sweep")
            f_hi = b.choice("lo.pad.sweep.fstop", 600e6, "Hz", "stop of the LO pad sweep")
            psweep = ac_sweep(b, "lo_pad_sweep", lin, pts, f_lo, f_hi, "LO pad: 300-600 MHz")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        u = b.part("rf_amp", "U50", "PHA-1", "LO1 buffer (gain block to about +13 dBm before the pad) [UNVERIFIED: Mini-Circuits PHA-1 datasheet]", serves_fc)
        exclude(b, "U50", "PHA-1 has no SPICE model: excluded; its ports are model.pha1.port_r in the fixtures")
        lch = b.part("ind_0603", "L52", part_value(l_ch.value), "PHA-1 bias choke")
        cd = b.part("cap_0402", "C55", part_value(c_dec.value), "RX_5V decoupling at the PHA-1 choke")
        co = b.part("cap_0402", "C56", part_value(c_out.value), "PHA-1 output DC block")
        for placed, device, value in ((lch, SpiceDevice.L, l_ch), (cd, SpiceDevice.C, c_dec), (co, SpiceDevice.C, c_out)):
            b.bind(placed.ref, two_terminal(b, placed, device, value, "at its value"))
        pads = {ref: b.part("res_0402", ref, part_value(v.value), desc) for ref, v, desc in (
            ("R50", r_sh, "LO pad: input shunt resistor"), ("R51", r_se, "LO pad: series resistor"), ("R52", r_sh, "LO pad: output shunt resistor"))}
        pad_bind = {ref: two_terminal(b, placed, SpiceDevice.R, r_se if ref == "R51" else r_sh, "pad at its calculator value") for ref, placed in pads.items()}
        for ref, binding in pad_bind.items():
            b.bind(ref, binding)
        nb.add("LO1_RAW", NetKind.RF, u.at("RF_IN"), "PHA-1 input: the LO output band-pass's output tap (the chain block's)")
        nb.add("LO1_BUF_OUT", NetKind.RF, [*u.at("RF_OUT"), ("L52", lch.pin("2")), ("C56", co.pin("1"))], "PHA-1 output with its bias choke")
        nb.add("RX_5V", NetKind.POWER, [("L52", lch.pin("1")), ("C55", cd.pin("1"))], "RX_5V rail (PHA-1 bias)")
        nb.add("LO_PAD_IN", NetKind.RF, [("C56", co.pin("2")), ("R50", pads["R50"].pin("1")), ("R51", pads["R51"].pin("1"))], "LO pad input")
        nb.add("LO1_MIX", NetKind.RF, [("R51", pads["R51"].pin("2")), ("R52", pads["R52"].pin("1"))], "LO drive to the ADEX-10's LO port")
        nb.add(GROUND_NET, NetKind.GROUND, [*u.at("GND"), ("C55", cd.pin("2")), ("R50", pads["R50"].pin("2")), ("R52", pads["R52"].pin("2"))])
        nb.declare(b)
        b.result.networks.append(RFNetwork(
            id="lo_pad", block=self.id, members=list(pads), ports=[fixture_port("pad_in", "LO_PAD_IN", pha1[1], "in"), fixture_port("lo1_mix", "LO1_MIX", adex[1], "out")],
            sweep=[psweep], expectations=[row("s21_lo1", "s21_db", "pad_in", "lo1_mix", f3, s21_pad, tol=pad_tol),
                                          row("s11_lo1", "s11_db", "pad_in", "pad_in", f3, s11_max, bound="at_most")],
        ))
        b.result.ports = [
            RFPort(name="LO1_RAW", net="LO1_RAW", kind="port", z0_ohm=pha1[1], frequency_hz=f3, direction="in"),
            RFPort(name="LO1_MIX", net="LO1_MIX", kind="port", z0_ohm=adex[1], frequency_hz=p.lo1[1], direction="out"),
            RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=rail5, direction="in"),
        ]
        b.result.chain = ["U50", "L52", "C56", "R50", "R51", "R52"]
        b.result.lab_items.append(LabItem(
            id="lo_level", block=self.id, what=f"LO drive at the ADEX-10's LO port (+7 dBm wanted) through the {float(a_db.value):g} dB pad",
            instruments=["power meter", "spectrum analyser"], reason="the PHA-1's output level and the multipliers' conversion are not modelled [UNVERIFIED]"))
        del rail5_key
        return b.done()


__all__ = [
    "BUFFER_ID",
    "BUFFER_INTERFACE",
    "BUFFER_NETWORKS",
    "CHAIN_ID",
    "CHAIN_INTERFACE",
    "CHAIN_NETWORKS",
    "PHA1_PORT",
    "STAGE_MULTIPLIERS",
    "LoBufferBlock",
    "LoChainBlock",
    "LoPlan",
    "lo_plan",
]
