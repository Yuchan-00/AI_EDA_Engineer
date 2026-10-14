"""The PTT sequencer and TX interlock block of the KR 447 MHz family: settle delay, low-pack TX inhibit, PA supply switch, optional time-out.

Invariant: the PA is enabled only when PTT is pressed AND the settle delay
has run AND the pack is above the confirmed cut-off - the default is "no
transmit" - and every threshold is a divider or RC the design deck simulates;
the comparators, the AND gate and the time-out counter have no SPICE model
and are excluded, so what is verified is their *input networks* and the PA
supply switch they drive, under the generic ``model.*`` cards, never the ICs.

The circuit (kr447 design §2.1, decisions 3A; local references, re-based by
200 on a board: ``R1`` -> ``R201``):

* SW1 (PTT push button) pulls ``PTT_N`` low while pressed (R12 pulls it up
  to ``V_SYS``); the power block's inverters turn it into ``PTT_ACTIVE`` and
  switch the rails.
* Settle delay: R1 from ``V_TX`` and R2 to GND charge C1 at ``DLY`` (a
  Thevenin RC: the TX rail itself starts the delay); U1 (LMV331 on
  ``TX_3V3``) compares ``DLY`` with the threshold ``DLY_TH`` of R3 / R4 and
  releases its open-collector output ``DLY_OK`` (R5 pull-up) once ``DLY`` is
  above it. D1 discharges C1 into ``PTT_ACTIVE`` the moment PTT is released
  (Q103 pulls ``PTT_ACTIVE`` low), so the PA is disabled at once. The delay
  comparator has no hysteresis: its input is a monotonic RC ramp (reset by D1)
  and a feedback resistor would load the timing node (a deviation from the
  design's "with hysteresis", stated in the template's notes).
* Low-pack TX inhibit (decision 3A: it enforces ``power.pack_cutoff_v``): U4
  compares the ``V_TX`` divider R6 / R7 (``UV_SENSE``) with the ``TX_3V3``
  divider R8 / R9 (``UV_REF``); its open-collector output ``LOWBAT_N`` (R10
  pull-up) feeds back through R11 (hysteresis ``power.uvlo_hyst_v``) and,
  when low, clamps ``DLY`` through D2, so ``DLY_OK`` never rises below the
  cut-off. The resistor values are choices; the trip (falling, V_TX = 6.40 V)
  and release (rising, 6.60 V) points they give are shown by the theory
  report with the formula, never computed into the IR, and the real points are
  the lab item ``rf.lab.uvlo``.
* Interlock: U2 (74LVC1G08 on ``TX_3V3``) ANDs ``PTT_LOGIC`` (``PTT_ACTIVE``
  halved by R13 / R14 - ``PTT_ACTIVE`` sits at ``V_SYS``, up to 8.4 V, and the
  gate's input must stay within its 5.5 V tolerance [UNVERIFIED: 74LVC1G08
  datasheet]) with ``DLY_OK``: ``PA_ON``. Q2 (driven through R15) pulls the
  gate of the PA supply switch Q1 (``TX_5V`` -> ``PA_5V``, pulled up off by
  R16); Q3 (R17, R18) inverts ``PA_ON`` into ``PA_PD`` for the PA's
  POWER_DOWN pin (``pa.power_down_polarity``: high = powered down, from the
  library pin name [UNVERIFIED: NXP MMZ09332B datasheet]). On a board with
  the PA, ``PA_PD`` also drives the active discharge Q5 (R25 to its base,
  R24 from ``PA_5V`` to its collector; R18 is then 1 kohm): at release the
  PA powers down and draws nothing, so Q5 empties ``PA_5V`` - the PA's draw
  never does. What the deck cannot see (its rails are ideal sources) is the
  lab item ``rail_sequencing``: Q1's body diode ties ``PA_5V`` to at most
  ``TX_5V`` + a diode drop, and ``PA_PD`` is pulled up to ``TX_3V3``, so
  POWER_DOWN deasserts once ``TX_3V3`` collapses - by then Q5 has emptied
  ``PA_5V``.
* With ``tx_timeout`` stated (``tot=True``): U3 (4060) counts its RC
  oscillator (R20 / C6 / R21) from a power-on clear (C7 / R23) and after
  2^13 periods Q14 turns Q4 on (R22), which clamps ``DLY``; D3 stops the
  oscillator so Q14 stays high. The time-out is ``calc.rf.tot.period`` with
  the 4060's RC constant an UNVERIFIED choice: lab item ``rf.lab.tot``.

Simulation: ``PTT_N`` is the PWL stimulus ``VPTT`` (released at the pack
voltage, pressed 0 V; SW1 has no switch model) and ``PA_ON`` - the excluded
AND gate's output - the PWL ``VPAON``, high from the latest time the
interlock may enable until the release (a modelling choice: the gate is not
simulated). ``tran_ptt`` (0-120 ms, :data:`PTT_PLAN_S`) checks the settle delay against the
threshold (held off after ``ptt.t_settle``, enabled by ``ptt.t_delay_max``,
off 1 ms after release), the PA supply (on while enabled, below 0.5 V 1 ms
after release - through a stand-in load R19 on the bench board, which has no
PA, and through the active discharge Q5 on a board with the PA, whose own
draw the deck switches off with POWER_DOWN), and the inhibit's input (``UV_SENSE`` above ``UV_REF`` at the stated
pack voltage). :func:`ptt_timing` declares the test plan every block that
checks a PTT event uses (the power block's rail checks), with identical rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import AnalysisSpec, NetKind, Reduce, Stimulus, StimulusKind, Traced
from ai_eda.ir.rf import LabItem
from ai_eda.tools.calc.basic import parallel_resistance, rc_step_response, rc_time_constant, voltage_divider_output
from ai_eda.tools.calc.radio import tot_period
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.rf import models
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.power import (
    V_IN_KEY,
    capacitor,
    card_bind,
    choice_once,
    computed_once,
    exclude,
    expectation,
    input_copy,
    net,
    pack_cutoff,
    op_analysis,
    pack_voltage,
    rail_level,
    rail_port,
    resistor,
    serving,
)

#: the PTT transient's analysis id (the power block's rail checks run on it too)
TRAN_PTT = "tran_ptt"
#: the requirement that switches the time-out block on
TX_TIMEOUT_KEY = "tx_timeout"
#: the time-out range the 4060 RC network is designed for here (s): R_t stays between about 5 kohm and 320 kohm with C_t 100 nF
TOT_RANGE_S: tuple[float, float] = (10.0, 600.0)
#: the 4060 counts 2^13 oscillator periods until Q14 first rises
TOT_Q14_PERIODS = 8192
#: E24 mantissas x 10 (integers: the values are exact floats)
E24_X10: tuple[int, ...] = (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91)
#: the PTT test plan (s): press, release, the end of ``tran_ptt`` and the check times - compressed from the design's 300 ms run
#: because the deck's 1 kHz audio test tones keep ngspice's step near 35 us through every transient (the run-time budget)
PTT_PLAN_S: dict[str, float] = {
    "t_press": 0.01,
    "check_on": 0.03,
    "check_enable": 0.06,
    "check_pa_on": 0.08,
    "t_release": 0.09,
    "check_release": 0.091,
    "check_off": 0.11,
    "stop": 0.12,
}


def _ms(key: str) -> str:
    return f"{PTT_PLAN_S[key] * 1e3:.12g} ms"


#: the hysteresis choice of the low-pack inhibit (decision 3A)
UVLO_HYST_KEY = "power.uvlo_hyst_v"


@dataclass(frozen=True)
class PttTiming:
    """The PTT test plan: when PTT is pressed / released in ``tran_ptt`` and when each rail / delay is checked (all confirmed choices)."""

    analysis: str
    t_press: Traced
    t_release: Traced
    t_settle: Traced
    t_delay_max: Traced
    check_on: Traced
    check_enable: Traced
    check_pa_on: Traced
    check_release: Traced
    check_off: Traced
    rail_on_min: Traced
    rail_off_max: Traced
    edge: Traced
    stop: Traced


def ptt_timing(b: BlockBuilder) -> PttTiming:
    """Declare (once per block) the PTT test plan, the ``tran_ptt`` analysis and the ``VPTT`` stimulus on ``PTT_N``; identical in every block."""
    v_in = pack_voltage(b)
    c = lambda key, value, unit, text: choice_once(b, key, value, unit, text)  # noqa: E731
    ms = _ms
    t_press = c("ptt.t_press", PTT_PLAN_S["t_press"], "s", f"tran_ptt: PTT is pressed at {ms('t_press')} (the rails start from the operating point with PTT released)")
    t_release = c("ptt.t_release", PTT_PLAN_S["t_release"], "s", f"tran_ptt: PTT is released at {ms('t_release')}")
    edge = c("ptt.t_edge", 1e-5, "s", "tran_ptt: the PTT and PA_ON stimulus edges take 10 us")
    stop = c("ptt.tran_stop", PTT_PLAN_S["stop"], "s", (
        f"tran_ptt runs from 0 to {ms('stop')} (short, because the audio test tones of the same deck keep ngspice's step small: the run-time budget)"))
    step = c("ptt.tran_step", 5e-5, "s", "tran_ptt step 50 us (ngspice's print step; its internal step is at most this)")
    t_settle = c("ptt.t_settle", 0.02, "s", (
        "the PA stays disabled for at least 20 ms after PTT is pressed: TCXO start-up plus chain settling [UNVERIFIED: Kyocera KT2520K datasheet]"))
    t_delay_max = c("ptt.t_delay_max", 0.05, "s", "the PA is enabled at most 50 ms after PTT is pressed (the settle delay's upper design limit)")
    check_on = c("ptt.t_check_on", PTT_PLAN_S["check_on"], "s", f"the rails and the hold-off are checked at {ms('check_on')} (the press + t_settle 20 ms)")
    check_enable = c("ptt.t_check_enable", PTT_PLAN_S["check_enable"], "s", f"the enable is checked at {ms('check_enable')} (the press + t_delay_max 50 ms)")
    check_pa_on = c("ptt.t_check_pa_on", PTT_PLAN_S["check_pa_on"], "s", f"the PA supply is checked on at {ms('check_pa_on')} (enabled, PTT still pressed)")
    check_release = c("ptt.t_check_release", PTT_PLAN_S["check_release"], "s",
                      f"the PA supply and the delay are checked off at {ms('check_release')}, 1 ms after the release")
    check_off = c("ptt.t_check_off", PTT_PLAN_S["check_off"], "s", f"the TX rail is checked off at {ms('check_off')}, 20 ms after the release")
    rail_on_min = c("ptt.rail_on_min", 5.4, "V", "tx_rail_on: V_TX at least 5.4 V while PTT is pressed (the LM1117's input needs V_out + dropout; 5.4 V is the check's floor)")
    rail_off_max = c("ptt.rail_off_max", 0.2, "V", "a switched rail counts as off below 0.2 V")
    v = float(v_in.value)
    tp, tr, te, ts = (float(x.value) for x in (t_press, t_release, edge, stop))
    points = [[0.0, v], [tp, v], [tp + te, 0.0], [tr, 0.0], [tr + te, v], [ts, v]]
    pwl = c("ptt.ptt_n_stimulus", points, None, (
        "the PTT switch SW201 has no model: PTT_N is driven by this PWL (s, V) - the pack voltage (input_voltage, the pull-up level) while released, "
        f"0 V while pressed from {ms('t_press')} to {ms('t_release')}, 10 us edges"))
    if not any(s.id == "VPTT" for s in b.result.stimuli):
        b.result.stimuli.append(Stimulus(id="VPTT", source="voltage", net="PTT_N", reference_net="GND", kind=StimulusKind.PWL, params={"points": pwl},
                                         provenance=b.ctx.provenance("PTT_N driven by the PTT stimulus (SW201 has no model)")))
    if not any(a.id == TRAN_PTT for a in b.result.analyses):
        b.result.analyses.append(AnalysisSpec(id=TRAN_PTT, kind=SpiceAnalysis.TRAN, params={"step": step, "stop": stop},
                                              provenance=b.ctx.provenance(f"PTT sequencing transient: press at {ms('t_press')}, release at {ms('t_release')}")))
    return PttTiming(TRAN_PTT, t_press, t_release, t_settle, t_delay_max, check_on, check_enable, check_pa_on, check_release, check_off, rail_on_min, rail_off_max, edge, stop)


#: every E24 value from 1 ohm to 9.1 Mohm, as exact floats (ascending)
E24_VALUES: tuple[float, ...] = tuple(sorted(float(m * 10**e) / 10.0 for e in range(0, 8) for m in E24_X10))


def e24_at_most(x: float) -> float:
    """The largest E24 value not above ``x``; ``ValueError`` below 1 ohm."""
    below = [v for v in E24_VALUES if v <= x]
    if not below:
        raise ValueError(f"no E24 value at or below {x!r}")
    return below[-1]


def e24_nearest(x: float) -> float:
    """The E24 value nearest to ``x`` (> 0) by ratio (the lower one on a tie)."""
    if not x > 0:
        raise ValueError(f"no E24 value near {x!r}")
    return min(E24_VALUES, key=lambda v: (abs(math.log(v / x)), v))


class PttBlock(Block):
    """PTT sequencer, settle delay, low-pack TX inhibit, TX interlock, PA supply switch and the optional time-out (module docstring).

    ``pa_stand_in``: a load resistor on ``PA_5V`` standing in for the PA (a
    board without a PA - the stage-1 bench board; the design deck's
    ``pa_supply_off_first`` needs a discharge path). Without it (a board
    with the PA) the block adds the active discharge Q205 / R224 / R225: the
    PA powers down at the release edge that opens the PA switch, so its own
    draw cannot empty ``PA_5V``. ``tot``: build the 4060 time-out (only when
    ``tx_timeout`` is stated).
    """

    id = "ptt"
    title = "PTT sequencer, settle delay, low-pack TX inhibit and TX interlock"
    interface_nets = ("V_SYS", "V_TX", "TX_5V", "TX_3V3", "PA_5V", "PTT_N", "PTT_ACTIVE", "DLY", "DLY_OK", "PA_ON", "PA_PD")

    def __init__(self, *, pa_stand_in: bool = True, tot: bool = False) -> None:
        self.pa_stand_in = pa_stand_in
        self.tot = tot

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        c = lambda key, value, unit, text: choice_once(b, key, value, unit, text)  # noqa: E731
        v_in = pack_voltage(b)
        t = ptt_timing(b)
        tx3 = rail_level(b, "TX_3V3")
        tx5 = rail_level(b, "TX_5V")
        pack_cutoff(b)
        c(UVLO_HYST_KEY, 0.2, "V", "hysteresis of the low-pack TX inhibit (decision 3A): the pack sags under the TX load, so the release point is 0.2 V above the trip")
        r_pu = c("ptt.r_ptt_pullup", 10e3, "ohm", "R212: PTT_N pull-up to V_SYS (SW201 pulls PTT_N to GND while pressed)")
        r_dt = c("ptt.r_dly_top", 100e3, "ohm", "R201: settle-delay charging resistor from V_TX to DLY")
        r_db = c("ptt.r_dly_bottom", 39e3, "ohm", "R202: DLY to GND (with R201 it keeps DLY below the comparator's input range at 8.4 V)")
        c_d = c("ptt.c_dly", 1.2e-6, "F", "C201: settle-delay capacitor on DLY")
        r_tt = c("ptt.r_th_top", 18e3, "ohm", "R203: delay threshold divider from TX_3V3")
        r_tb = c("ptt.r_th_bottom", 10e3, "ohm", "R204: delay threshold divider to GND")
        r_ok = c("ptt.r_dly_ok_pullup", 10e3, "ohm", "R205: DLY_OK pull-up to TX_3V3 (U201 has an open-collector output)")
        r_ut = c("ptt.r_uv_top", 100e3, "ohm", "R206: low-pack sense divider from V_TX (UV_SENSE)")
        r_ub = c("ptt.r_uv_bottom", 34e3, "ohm", "R207: low-pack sense divider to GND (with R206 and R211 the trip is at V_TX = 6.40 V falling, 6.60 V rising)")
        r_rt = c("ptt.r_uvref_top", 100e3, "ohm", "R208: low-pack reference divider from TX_3V3 (UV_REF)")
        r_rb = c("ptt.r_uvref_bottom", 100e3, "ohm", "R209: low-pack reference divider to GND")
        r_lb = c("ptt.r_lowbat_pullup", 10e3, "ohm", "R210: LOWBAT_N pull-up to TX_3V3 (U204 has an open-collector output)")
        r_hy = c("ptt.r_uv_hyst", 1.65e6, "ohm", "R211: hysteresis from LOWBAT_N to UV_SENSE: 3.3 V x R206 / R211 = 0.2 V of V_TX (power.uvlo_hyst_v)")
        r_lt = c("ptt.r_logic_top", 220e3, "ohm", "R213: PTT_ACTIVE to the AND gate's input (PTT_ACTIVE is at V_SYS, up to 8.4 V)")
        r_lg = c("ptt.r_logic_bottom", 220e3, "ohm", "R214: the AND gate's input to GND: PTT_LOGIC = PTT_ACTIVE / 2 (at most 4.2 V, within the gate's 5.5 V input tolerance [UNVERIFIED: 74LVC1G08 datasheet])")
        r_pb = c("ptt.r_pa_base", 22e3, "ohm", "R215 / R217: base resistors of the PA-switch driver Q202 and the POWER_DOWN inverter Q203 from PA_ON")
        r_pg = c("ptt.r_pa_gate", 10e3, "ohm", "R216: Q201's gate pull-up to TX_5V (the PA supply is off by default)")
        if self.pa_stand_in:
            r_pd = c("ptt.r_pa_pd_pullup", 10e3, "ohm", "R218: PA_PD pull-up to TX_3V3 (PA_PD high = PA powered down)")
        else:
            r_pd = c("ptt.r_pa_pd_pullup", 1e3, "ohm", (
                "R218: PA_PD pull-up to TX_3V3 (PA_PD high = PA powered down); 1 kohm because it also drives the PA_5V discharge transistor Q205's "
                "base through R225 (about 1.7 mA; Q203 sinks 3.3 mA while the PA is on)"))
        c_pa = c("ptt.c_pa", 1e-6, "F", "C202: PA_5V decoupling capacitor at the switch")
        c_dec = c("ptt.c_decouple", 100e-9, "F", "supply decoupling of U201, U202, U204 (and U203) on TX_3V3")
        c("pa.power_down_polarity", "high = powered down", None, (
            "the PA's POWER_DOWN pin (library pin name): PA_PD high powers the PA down, so Q203 pulls PA_PD low only while PA_ON is high "
            "[UNVERIFIED: NXP MMZ09332B datasheet]"))
        pa_off_max = c("ptt.pa_off_max", 0.5, "V", "pa_supply_off_first: PA_5V below 0.5 V 1 ms after the release (the PA is unpowered before the T/R switch loses its bias)")
        pa_on_min = c("ptt.pa_on_min", 4.5, "V", "pa_supply_on: PA_5V at least 4.5 V while the interlock enables the PA")
        tol_rel = c("ptt.tol_rel", 0.02, None, "relative tolerance of the uvlo_ref divider check (2 %: ideal resistors, the formula reproduces)")
        # the calculators: thresholds and the settle delay at the stated pack voltage (informative; the deck judges)
        v_th = computed_once(b, "ptt.v_th", lambda: voltage_divider_output(tx3, r_tt, r_tb, ("power.tx_3v3", "ptt.r_th_top", "ptt.r_th_bottom")))
        v_ref = computed_once(b, "ptt.v_uv_ref", lambda: voltage_divider_output(tx3, r_rt, r_rb, ("power.tx_3v3", "ptt.r_uvref_top", "ptt.r_uvref_bottom")))
        r_dth = computed_once(b, "ptt.r_dly_thevenin", lambda: parallel_resistance(r_dt, r_db, ("ptt.r_dly_top", "ptt.r_dly_bottom")))
        tau = computed_once(b, "ptt.tau_dly", lambda: rc_time_constant(r_dth, c_d, ("ptt.r_dly_thevenin", "ptt.c_dly")))
        v_fin = computed_once(b, "ptt.v_dly_final", lambda: voltage_divider_output(v_in, r_dt, r_db, (V_IN_KEY, "ptt.r_dly_top", "ptt.r_dly_bottom")))
        computed_once(b, "ptt.v_dly_settle", lambda: rc_step_response(v_fin, t.t_settle, tau, ("ptt.v_dly_final", "ptt.t_settle", "ptt.tau_dly")))
        computed_once(b, "ptt.v_dly_enable", lambda: rc_step_response(v_fin, t.t_delay_max, tau, ("ptt.v_dly_final", "ptt.t_delay_max", "ptt.tau_dly")))
        pmos, npn, diode = models.pmos_card(), models.npn_card(), models.diode_card()

        sw1 = b.part("sw_push", "SW1", "PTT", "PTT push button (pulls PTT_N low while pressed)")
        exclude(b, "SW1", "switch without a model: PTT_N is the PWL stimulus VPTT")
        r12 = resistor(b, "R12", r_pu, "PTT_N pull-up")
        r1 = resistor(b, "R1", r_dt, "settle-delay charging resistor")
        r2 = resistor(b, "R2", r_db, "settle-delay divider to GND")
        c1 = capacitor(b, "C1", c_d, "settle-delay capacitor")
        d1 = b.part("diode_sw", "D1", "1N4148WS", "discharges the settle-delay capacitor into PTT_ACTIVE at release")
        u1 = b.part("comparator", "U1", "LMV331", "settle-delay comparator (supplied from TX_3V3)")
        r3 = resistor(b, "R3", r_tt, "delay threshold divider (top)")
        r4 = resistor(b, "R4", r_tb, "delay threshold divider (bottom)")
        r5 = resistor(b, "R5", r_ok, "DLY_OK pull-up")
        u2 = b.part("and_gate", "U2", "74LVC1G08", "TX interlock AND: PTT_LOGIC and DLY_OK -> PA_ON (supplied from TX_3V3)")
        r13 = resistor(b, "R13", r_lt, "PTT_ACTIVE divider to the AND gate (top)")
        r14 = resistor(b, "R14", r_lg, "PTT_ACTIVE divider to the AND gate (bottom)")
        u4 = b.part("comparator", "U4", "LMV331", "low-pack TX inhibit comparator (enforces power.pack_cutoff_v; supplied from TX_3V3)")
        r6 = resistor(b, "R6", r_ut, "low-pack sense divider (top)")
        r7 = resistor(b, "R7", r_ub, "low-pack sense divider (bottom)")
        r8 = resistor(b, "R8", r_rt, "low-pack reference divider (top)")
        r9 = resistor(b, "R9", r_rb, "low-pack reference divider (bottom)")
        r10 = resistor(b, "R10", r_lb, "LOWBAT_N pull-up")
        r11 = resistor(b, "R11", r_hy, "low-pack inhibit hysteresis")
        d2 = b.part("diode_sw", "D2", "1N4148WS", "clamps DLY low while LOWBAT_N is low (the pack is below the cut-off)")
        q1 = b.part("pfet", "Q1", "AO3401A", "PA supply switch (TX_5V -> PA_5V; off by default)")
        q2 = b.part("npn_small", "Q2", "MMBT3904", "PA supply switch driver (from PA_ON)")
        q3 = b.part("npn_small", "Q3", "MMBT3904", "POWER_DOWN inverter (PA_ON -> PA_PD)")
        r15 = resistor(b, "R15", r_pb, "Q202 base resistor")
        r16 = resistor(b, "R16", r_pg, "Q201 gate pull-up")
        r17 = resistor(b, "R17", r_pb, "Q203 base resistor")
        r18 = resistor(b, "R18", r_pd, "PA_PD pull-up")
        c2 = capacitor(b, "C2", c_pa, "PA_5V decoupling")
        c3 = capacitor(b, "C3", c_dec, "U201 supply decoupling")
        c4 = capacitor(b, "C4", c_dec, "U202 supply decoupling")
        c5 = capacitor(b, "C5", c_dec, "U204 supply decoupling")
        for ref in ("U1", "U4"):
            exclude(b, ref, "LMV331 has no model: its input networks are simulated, its output is not")
        exclude(b, "U2", "74LVC1G08 has no model: PA_ON is the PWL stimulus VPAON")
        for ref in ("D1", "D2"):
            card_bind(b, ref, diode)
        card_bind(b, "Q1", pmos)
        for ref in ("Q2", "Q3"):
            card_bind(b, ref, npn)
        in_a, in_b = u2.pins("IN")
        members: dict[str, list[tuple[str, str]]] = {
            "PTT_N": [*sw1.at("1"), *r12.at("2")],
            "V_SYS": r12.at("1"),
            "V_TX": [*r1.at("1"), *r6.at("1")],
            "DLY": [*r1.at("2"), *r2.at("1"), *c1.at("1"), *d1.at("A"), *u1.at("+"), *d2.at("A")],
            "PTT_ACTIVE": [*d1.at("K"), *r13.at("1")],
            "DLY_TH": [*r3.at("2"), *r4.at("1"), *u1.at("-")],
            "DLY_OK": [*u1.at("OUT"), *r5.at("2"), ("U2", in_b)],
            "PTT_LOGIC": [*r13.at("2"), *r14.at("1"), ("U2", in_a)],
            "PA_ON": [*u2.at("OUT"), *r15.at("1"), *r17.at("1")],
            "UV_SENSE": [*r6.at("2"), *r7.at("1"), *r11.at("2"), *u4.at("+")],
            "UV_REF": [*r8.at("2"), *r9.at("1"), *u4.at("-")],
            "LOWBAT_N": [*u4.at("OUT"), *r10.at("2"), *r11.at("1"), *d2.at("K")],
            "PA_DRV_B": [*r15.at("2"), *q2.at("B")],
            "PA_G": [*q2.at("C"), *r16.at("2"), *q1.at("G")],
            "PA_PD_B": [*r17.at("2"), *q3.at("B")],
            "PA_PD": [*q3.at("C"), *r18.at("2")],
            "TX_5V": [*q1.at("S"), *r16.at("1")],
            "PA_5V": [*q1.at("D"), *c2.at("1")],
            "TX_3V3": [*r3.at("1"), *r5.at("1"), *r8.at("1"), *r10.at("1"), *r18.at("1"), *u1.at("V+"), *u2.at("VCC"), *u4.at("V+"), *c3.at("1"), *c4.at("1"), *c5.at("1")],
            "GND": [*sw1.at("2"), *r2.at("2"), *c1.at("2"), *r4.at("2"), *r7.at("2"), *r9.at("2"), *r14.at("2"), *u1.at("V-"), *u2.at("GND"), *u4.at("V-"),
                    *q2.at("E"), *q3.at("E"), *c2.at("2"), *c3.at("2"), *c4.at("2"), *c5.at("2")],
        }
        chain = ["SW1", "R1", "C1", "U1", "U4", "U2", "Q2", "Q1", "Q3"]
        if self.pa_stand_in:
            r_si = c("ptt.r_pa_stand_in", 330.0, "ohm", (
                "R219: a load on PA_5V standing in for the PA, which this board does not have (15 mA at 5 V, 76 mW): it discharges PA_5V when Q201 "
                "turns off"))
            r19 = resistor(b, "R19", r_si, "PA stand-in load (the board has no PA)")
            members["PA_5V"] += r19.at("1")
            members["GND"] += r19.at("2")
        else:
            # the PA's own draw stops at the same release edge (PA_PD rises: the PA powers down), so PA_5V needs an active discharge
            r_dis = c("ptt.r_pa_discharge", 27.0, "ohm", (
                "R224: PA_5V active-discharge resistor into Q205 (with about 11 uF on PA_5V, tau about 0.3 ms: PA_5V below 0.5 V within 1 ms of the "
                "release; about 185 mA for a moment, the MMBT3904's pulse rating [UNVERIFIED: MMBT3904 datasheet])"))
            r_disb = c("ptt.r_pa_discharge_base", 470.0, "ohm", (
                "R225: Q205's base resistor from PA_PD: Q205 conducts exactly while PA_PD is high (the PA powered down), never while Q203 holds PA_PD low"))
            q5 = b.part("npn_small", "Q5", "MMBT3904", "PA_5V active discharge (on while PA_PD is high: the PA is off and its supply is emptied)")
            r24 = resistor(b, "R24", r_dis, "PA_5V discharge resistor")
            r25 = resistor(b, "R25", r_disb, "Q205 base resistor from PA_PD")
            card_bind(b, "Q5", npn)
            members["PA_5V"] += r24.at("1")
            members["PA_DIS_C"] = [*r24.at("2"), *q5.at("C")]
            members["PA_PD"] += r25.at("1")
            members["PA_DIS_B"] = [*r25.at("2"), *q5.at("B")]
            members["GND"] += q5.at("E")
            chain.append("Q5")
        if self.tot:
            chain += self._time_out(b, members)
        # the PA_ON stimulus: the AND gate's output as the interlock is designed to give it
        on_level = float(tx3.value)
        points = [[0.0, 0.0], [float(t.check_enable.value), 0.0], [float(t.check_enable.value) + float(t.edge.value), on_level],
                  [float(t.t_release.value), on_level], [float(t.t_release.value) + float(t.edge.value), 0.0], [float(t.stop.value), 0.0]]
        pa_on = c("ptt.pa_on_stimulus", points, None, (
            f"U202 has no model: PA_ON is driven by this PWL (s, V) - low until the latest time the interlock may enable ({_ms('check_enable')}: "
            f"press + t_delay_max), TX_3V3 level until the release at {_ms('t_release')}, low after; it checks the PA supply switch, never the AND gate"))
        b.result.stimuli.append(Stimulus(id="VPAON", source="voltage", net="PA_ON", reference_net="GND", kind=StimulusKind.PWL, params={"points": pa_on},
                                         provenance=ctx.provenance("PA_ON driven by the interlock stimulus (U202 has no model)")))
        op_analysis(b)
        expectation(b, "pa_held_off_while_settling", t.analysis, "v(DLY)", Reduce.AT, v_th, "the settle delay holds the PA off for t_settle after PTT is pressed",
                    at=t.check_on, bound="at_most")
        expectation(b, "pa_enabled_after_settle", t.analysis, "v(DLY)", Reduce.AT, v_th, "the settle delay has run by t_delay_max after PTT is pressed",
                    at=t.check_enable, bound="at_least")
        expectation(b, "pa_off_fast", t.analysis, "v(DLY)", Reduce.AT, v_th, "D201 discharges the delay 1 ms after the release: the PA is disabled at once",
                    at=t.check_release, bound="at_most")
        expectation(b, "pa_supply_on", t.analysis, "v(PA_5V)", Reduce.AT, pa_on_min, "the PA supply switch Q201 is on while the interlock enables the PA",
                    at=t.check_pa_on, bound="at_least")
        expectation(b, "pa_supply_off_first", t.analysis, "v(PA_5V)", Reduce.AT, pa_off_max,
                    "the PA supply is off 1 ms after the release (before the T/R switch loses its bias on the transceiver)", at=t.check_release, bound="at_most")
        expectation(b, "uvlo_ref", "op", "v(UV_REF)", Reduce.VALUE, v_ref, "the low-pack inhibit's reference divider from TX_3V3", tol_rel=tol_rel)
        expectation(b, "uvlo_sense_released", t.analysis, "v(UV_SENSE)", Reduce.AT, v_ref,
                    "at the stated pack voltage the inhibit's sense node is above its reference (the inhibit releases); the trip itself is lab item uvlo",
                    at=t.check_on, bound="at_least")
        for name, pins in members.items():
            net(b, name, pins, _NOTES.get(name, f"{name} (ptt block)"), NetKind.GROUND if name == "GND" else None)
        b.result.ports = [rail_port("tx_3v3", "TX_3V3", tx3, "in"), rail_port("tx_5v", "TX_5V", tx5, "in"), rail_port("pa_5v", "PA_5V", tx5, "out")]
        b.result.chain = chain
        b.result.lab_items = [
            LabItem(id="rail_sequencing", block=self.id,
                    what=("PTT rail sequencing, the settle delay and the PA supply switching on the real parts (and, on the transceiver, the PIN bias hold-up "
                          "against PA_5V)" + ("" if self.pa_stand_in else
                                              "; PA_5V's discharge by Q205 as TX_5V / TX_3V3 decay (Q201's body diode ties PA_5V to at most TX_5V + a "
                                              "diode drop; POWER_DOWN, pulled up to TX_3V3, deasserts once TX_3V3 collapses)")),
                    instruments=["oscilloscope"], reason="the comparators, the AND gate and the regulators are not simulated; the deck proves only their input networks"),
            LabItem(id="uvlo", block=self.id, what="the low-pack TX inhibit's real trip and release points against the 6.4 V pack cut-off, under the TX load",
                    instruments=["adjustable supply", "DMM"], reason="U204 is not simulated; its trip points follow from the resistor choices by the formula only"),
        ]
        if self.tot:
            b.result.lab_items.append(LabItem(id="tot", block=self.id, what="the 4060 transmit time-out's real duration",
                                              instruments=["oscilloscope", "stopwatch"], reason="the 4060's RC-oscillator constant is an UNVERIFIED choice and the counter is not simulated"))
        return b.done()

    def _time_out(self, b: BlockBuilder, members: dict[str, list[tuple[str, str]]]) -> list[str]:
        """The optional 4060 time-out (module docstring); returns its chain."""
        ctx = b.ctx
        t_req = input_copy(b, TX_TIMEOUT_KEY, "ptt.tx_timeout")
        k_rc = choice_once(b, "ptt.tot_k_rc", 2.3, None, (
            "the 4060 RC oscillator's period constant: T_osc = k_RC R_t C_t with k_RC 2.3 [UNVERIFIED: 4060 datasheet]"))
        c_t = choice_once(b, "ptt.tot_c", 100e-9, "F", "C206: the time-out oscillator's timing capacitor")
        target = float(t_req.value) / (TOT_Q14_PERIODS * float(k_rc.value) * float(c_t.value))
        r_t_value = e24_at_most(target)
        r_t = choice_once(b, "ptt.tot_r", r_t_value, "ohm", (
            f"R220: the time-out oscillator's timing resistor, the largest E24 value whose time-out 2^13 k_RC R_t C_t is at most the stated "
            f"tx_timeout ({float(t_req.value):.6g} s)"))
        r_s = choice_once(b, "ptt.tot_rs", e24_nearest(10.0 * r_t_value), "ohm", "R221: the oscillator's input resistor, about 10 x R_t [UNVERIFIED: 4060 datasheet]")
        computed_once(b, "ptt.tot_period", lambda: tot_period(k_rc, r_t, c_t, ("ptt.tot_k_rc", "ptt.tot_r", "ptt.tot_c")))
        r_q = choice_once(b, "ptt.tot_r_base", 22e3, "ohm", "R222: Q204's base resistor from the 4060's Q14")
        c_clr = choice_once(b, "ptt.tot_c_clr", 100e-9, "F", "C207: power-on clear of the 4060 (a pulse on CLR when TX_3V3 comes up, i.e. at every PTT press)")
        r_clr = choice_once(b, "ptt.tot_r_clr", 100e3, "ohm", "R223: CLR to GND (ends the power-on clear pulse)")
        c_dec = b.result.params["ptt.c_decouple"]
        u3 = b.part("counter_4060", "U3", "4060", "transmit time-out counter (RC oscillator + 14-stage counter; Q14 after 2^13 periods)")
        r20 = resistor(b, "R20", r_t, "time-out timing resistor", serves=serving(ctx, TX_TIMEOUT_KEY))
        c6 = capacitor(b, "C6", c_t, "time-out timing capacitor", serves=serving(ctx, TX_TIMEOUT_KEY))
        r21 = resistor(b, "R21", r_s, "time-out oscillator input resistor")
        d3 = b.part("diode_sw", "D3", "1N4148WS", "stops the time-out oscillator once Q14 is high")
        r22 = resistor(b, "R22", r_q, "Q204 base resistor")
        q4 = b.part("npn_small", "Q4", "MMBT3904", "clamps DLY once the time-out has run")
        c7 = capacitor(b, "C7", c_clr, "4060 power-on clear capacitor")
        r23 = resistor(b, "R23", r_clr, "4060 power-on clear resistor")
        c8 = capacitor(b, "C8", c_dec, "U203 supply decoupling")
        exclude(b, "U3", "the 4060 has no model: the time-out is calc.rf.tot.period, a lab item")
        exclude(b, "Q4", "its base is driven by the unmodelled 4060's Q14: it would change nothing the deck computes (Q14 is not simulated)")
        exclude(b, "D3", "between the unmodelled 4060's Q14 and its oscillator input")
        for fn in ("Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q12", "Q13"):
            b.leave_open("U3", fn, "an unused counter output")
        members["TOT_RT"] = [*u3.at("Φ0"), *r20.at("1")]
        members["TOT_CT"] = [*u3.at("~{Φ0}"), *c6.at("1")]
        members["TOT_RC"] = [*r20.at("2"), *c6.at("2"), *r21.at("1")]
        members["TOT_RS"] = [*r21.at("2"), *u3.at("~{Φ1}"), *d3.at("K")]
        members["TOT_Q14"] = [*u3.at("Q14"), *d3.at("A"), *r22.at("1")]
        members["TOT_B"] = [*r22.at("2"), *q4.at("B")]
        members["TOT_CLR"] = [*u3.at("CLR"), *c7.at("2"), *r23.at("1")]
        members["DLY"] += q4.at("C")
        members["TX_3V3"] += [*u3.at("VDD"), *c7.at("1"), *c8.at("1")]
        members["GND"] += [*u3.at("VSS"), *q4.at("E"), *r23.at("2"), *c8.at("2")]
        return ["U3", "Q4"]


_NOTES: dict[str, str] = {
    "PTT_N": "the PTT switch (low while pressed) with its pull-up; driven by VPTT in the deck",
    "DLY": "the settle-delay node: RC from V_TX, D201 to PTT_ACTIVE, D202 to LOWBAT_N, the delay comparator's input",
    "DLY_TH": "the delay threshold divider from TX_3V3", "DLY_OK": "U201's open-collector output (settled) with its pull-up: the AND gate's second input",
    "PTT_LOGIC": "PTT_ACTIVE halved for the AND gate's first input", "PA_ON": "the interlock AND output: PA enable",
    "UV_SENSE": "the low-pack sense divider from V_TX with its hysteresis", "UV_REF": "the low-pack reference divider from TX_3V3",
    "LOWBAT_N": "U204's open-collector output (low = pack below the cut-off), hysteresis and the DLY clamp diode",
    "PA_DRV_B": "Q202's base", "PA_G": "Q201's gate: pulled up to TX_5V, pulled low by Q202", "PA_PD_B": "Q203's base",
    "PA_DIS_C": "the PA_5V discharge path: R224 into Q205's collector", "PA_DIS_B": "Q205's base: driven from PA_PD through R225",
    "PA_PD": "the PA's POWER_DOWN line (high = powered down)", "PA_5V": "the switched PA supply", "TX_5V": "the TX 5 V rail (PA switch source)",
    "TX_3V3": "the TX logic rail", "V_TX": "the switched TX rail (delay and inhibit dividers)", "V_SYS": "the pack rail (PTT_N pull-up)",
    "PTT_ACTIVE": "high in TX: D201's cathode and the AND gate's divider", "TOT_RT": "the 4060's Φ0 with the timing resistor",
    "TOT_CT": "the 4060's ~Φ0 with the timing capacitor", "TOT_RC": "the time-out RC junction", "TOT_RS": "the 4060's oscillator input (~Φ1)",
    "TOT_Q14": "the 4060's Q14: the time-out", "TOT_B": "Q204's base", "TOT_CLR": "the 4060's clear: a power-on pulse", "GND": "ground",
}


__all__ = [
    "E24_VALUES",
    "E24_X10",
    "TOT_Q14_PERIODS",
    "TOT_RANGE_S",
    "TRAN_PTT",
    "TX_TIMEOUT_KEY",
    "UVLO_HYST_KEY",
    "PttBlock",
    "PttTiming",
    "e24_at_most",
    "e24_nearest",
    "ptt_timing",
]
