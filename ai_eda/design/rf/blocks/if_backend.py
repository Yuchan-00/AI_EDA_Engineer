"""The IF back-end block of the KR 447 MHz FM receiver (kr447 design §2.2, part P10): IF1 21.4 MHz -> crystal ladder -> SA605D -> audio / RSSI.

Invariant: every part comes from the kr447 parts table by library name
(:mod:`ai_eda.design.rf.parts`), every number is a registered calculator's
output over the block's confirmed choices, the ``model.*`` values and the
confirmed requirements, and every passive network the SA605D sees is an RF
fixture (``ir.rf.networks``) whose rows the fixture runner judges on ngspice.
Nothing here says the SA605D, the crystals or the receiver work: the SA605D
has no model (excluded from every netlist; its ports are the
``model.sa605.*`` port resistances), the crystals are Butterworth-Van Dyke
cards of UNVERIFIED model values, and a fixture PASS is "a network verdict
under confirmed model values (not a measured part)" at schematic level.

What it builds (block-local references, re-based by the template: ``+ 500``):

* ``J1`` - the IF1 input U.FL (``input_connector``; the transceiver drives
  the interface net ``IF1`` from the front end instead) and the 50 ohm ->
  R_end low-pass L-match ``L1`` / ``C1`` (``calc.rf.lmatch.lowpass.*``);
* the crystal ladder ``Y1`` .. ``Yn`` (``ifb.lad.n``, 7 by default): a
  lower-sideband Butterworth ladder - crystals in series, shunt coupling
  capacitors, a series tuning capacitor in every mesh but the one(s) with
  the largest coupling sum - designed by the C0-aware Dishal equations of
  ``calc.crystal.ladder.*`` (reactance slope of crystal + C0 at the mesh
  frequency; Matthaei / Young / Jones 8.02, EMRFD ch. 3). The two end meshes'
  tuning capacitors are trimmers ``C_T1`` / ``C_T2`` (simulated at the
  calculator's value: the alignment's mid position);
* the R_end -> ``model.sa605.rf_in_r`` L-match ``L2`` / ``C_`` into the
  SA605D mixer input ``RF_IN``;
* ``U1`` SA605D (pins by name) with its LO2 crystal oscillator (``Y`` +
  the two Colpitts capacitors ``calc.crystal.c_for_load`` of the confirmed
  load capacitance), the two 450 kHz IF2 2-pole top-C filters
  (``calc.rf.resonator.top_c.*``, ``MIXER_OUT -> IF_AMP_IN`` and
  ``IF_AMP_OUT -> LIMITER_IN``, 1.5 kohm model ports), the quadrature tank
  (``LIMITER_OUT -> C_q -> QUADRATURE_IN``, tank ``L // (C_fixed + C_trim) //
  R_p`` returned to ``RX_5V``), the RSSI load R / C (``calc.rc.tau``),
  decoupling, and test points on the limiter input, the quadrature node, RSSI
  and the unmuted audio output.

Interface nets (never prefixed): ``IF1`` (50 ohm IF1 input, a ``port`` of
the block at ``rf.if1``), ``RX_5V`` (the SA605D supply, a ``rail`` of
``ifb.vcc``), ``DISC_OUT`` (muted audio -> the RX audio block), ``RSSI``
(-> the squelch comparator) and ``MUTE`` (<- the squelch comparator), and
``GND``.

The crystal ladder, and why its numbers differ from the design's first cut:

* **C_m.** With the design's first-cut ``model.xtal21.cm`` = 6 fF and C0 = 4 pF (a
  capacitance ratio of 667) no 6- or 7-crystal lower-sideband ladder of
  7.5 kHz exists: ``calc.crystal.ladder`` refuses it with its C0 bound
  f_s C_m / (4 C0 kappa) (4.52 kHz for 6 crystals). This block therefore
  uses ``model.xtal21.cm`` = 16 fF = C0 / 250 (the model table's value since
  the wave-2 merge; the design's first cut had 6 fF), the
  typical capacitance ratio of an AT-cut fundamental crystal
  [UNVERIFIED] - a procurement spec for the ordered crystals (measure C_m / C0
  / R_m of each one, G3UUR, before the ladder is built), never a fitted
  constant: the calculator and the fixture are unchanged.
* **Seven crystals.** With C_m 16 fF no 6-crystal Butterworth ladder holds
  both the passband rows (+/- ``rf.if_bw`` / 2 at least -3.5 dB) and the
  adjacent-channel rows (+/- the channel spacing at most -40 dB): the
  lower-sideband ladder's low skirt is the shallow one (-38.1 dB at best).
  Seven crystals hold both with margin (``ifb.lad.n`` = 7).
* **Pre-distortion.** The motional loss (``model.xtal21.rm`` 25 ohm) narrows
  the realised passband, so the ladder is designed for ``ifb.lad.bw_design``
  (8.75 kHz) and judged at the confirmed ``rf.if_bw`` (7.5 kHz) - the
  calculator's own advice (judge the realised response, or pre-distort the
  target), the targets unchanged.
* **Centre.** The lower-sideband ladder's passband sits above the crystals'
  series resonance (about 11.4 kHz here), so crystals at 21.4 MHz would put
  IF1 on the lower skirt. The crystals are specified at
  f_s = 2 IF1 - f_c(IF1) (the centre the ladder would have with crystals at
  IF1, mirrored about IF1: ``calc.rf.superhet.image``'s 2 LO - f used as
  that reflection), which puts the realised centre ``ifb.lad.f0``
  (``calc.crystal.ladder.center``) within a few hertz of IF1; the fixture row
  ``if1_centre`` judges that IF1 lies in the flat passband.
* **Loss.** With R_m 25 ohm the ladder loses about 4.9 dB at its centre, so
  the design's "s21 at f_0 at least -4 dB" cannot hold; the row takes the
  exact ladder response as its nominal (``calc.crystal.ladder.s21_db``,
  tol_abs 1 dB - the design's rule for loss rows of lossy networks, §2.3);
  the two L-matches' inductor loss (about 0.7 dB at Q 30) is inside that
  tolerance, and the ladder alone (``if1_ladder``, between R_end ports) is
  judged against the calculator within 0.2 dB.

Every frequency a row is read at is a calculator output (``f0 -/+ delta``
through ``calc.rf.superhet.lo``'s f + side delta, harmonics through
``calc.rf.harmonic``). The block writes its keys under ``ifb.`` and the
plan keys ``rf.if1``, ``rf.if2``, ``rf.lo2_side``, ``rf.if_bw``, ``rf.z0``
(unless a composing template already wrote them: :attr:`BlockContext.shared`);
it reads the profile's ``kr447.channel_raster`` / ``kr447.max_deviation``
from :attr:`BlockContext.shared` (the template's profile choices) and the
confirmed ``channel_spacing``, ``frequency_deviation``, ``system_impedance``
and ``frequency_tolerance`` when stated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_eda.ir import AnalysisSpec, NetKind, SpiceBinding, SpiceDevice, Traced
from ai_eda.ir.rf import LabItem, PlanLine, RFExpectation, RFNetwork, RFPort
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import clock_divided, rc_time_constant
from ai_eda.tools.calc.part_value import format_part_value
from ai_eda.tools.calc.rf import crystal_c_for_load, lc_l_for_resonance, lmatch, lmatch_lowpass_c_shunt, lmatch_lowpass_l_series, ppm_offset
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.inputs import canonical_key
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.models import MODEL_VALUES, MODEL_VERDICT, ModelValue, crystal_card, crystal_lm
from ai_eda.design.rf.parts import PlacedPart

#: the block id (``ir.rf.blocks`` / the networks' ``block``)
BLOCK_ID = "if_backend"
#: the interface nets the composition keeps
INTERFACE_NETS: tuple[str, ...] = ("IF1", "RX_5V", "DISC_OUT", "RSSI", "MUTE")
#: the fixture networks this block declares
NETWORKS: tuple[str, ...] = ("if1_ladder", "if1_filter", "if2_bpf_a", "if2_bpf_b", "quad_tank")
#: the profile values the block reads from :attr:`BlockContext.shared`
PROFILE_READS: tuple[str, ...] = ("kr447.channel_raster", "kr447.max_deviation")

#: the crystal's motional capacitance (16 fF = C0 / 250, see the module docstring): the table's row since the wave-2 merge (the design's first cut had 6 fF)
XTAL_CM = MODEL_VALUES["model.xtal21.cm"]
#: the SA605D mixer input and limiter output port resistances (the table's model.sa605.port_r names the mixer output, IF amplifier and limiter input)
SA605_RF_IN = ModelValue("model.sa605.rf_in_r", 1500.0, "ohm", "input resistance of the SA605 mixer input RF_IN (the ladder's load; the SA605 is excluded from the netlist)",
                         "NXP SA605 datasheet")
SA605_LIM_OUT = ModelValue("model.sa605.lim_out_r", 1500.0, "ohm", "output resistance of the SA605 limiter output LIMITER_OUT (the quadrature network's drive)",
                           "NXP SA605 datasheet")

#: every part of the block is in the fixtures only: the design deck leaves it out with this reason
DECK_EXCLUDED = (
    "if_backend: the SA605D has no SPICE model, so none of its networks carries a signal in the design deck; this part is simulated only in the "
    "block's RF fixtures (spice.rf.if1_ladder / if1_filter / if2_bpf_a / if2_bpf_b / quad_tank) with its fixture binding"
)


@dataclass(frozen=True)
class IfBackendDefaults:
    """The block's default choices (every one a confirmation-table row)."""

    if1_hz: float = 21.4e6
    if2_hz: float = 450e3
    lo2_side: float = -1.0
    if_bw_hz: float = 7.5e3
    z0_ohm: float = 50.0
    n_xtal: float = 7.0
    bw_design_hz: float = 8.75e3
    pass_min_db: float = -3.5
    acs_max_db: float = -40.0
    lo2_max_db: float = -60.0
    image2_max_db: float = -70.0
    centre_min_db: float = -0.5
    loss_tol_db: float = 1.0
    exact_tol_db: float = 0.2
    if2_n: float = 2.0
    if2_bw_hz: float = 30e3
    if2_l_h: float = 390e-6
    if2_offset_hz: float = 100e3
    if2_s21_min_db: float = -6.0
    if2_rej_max_db: float = -10.0
    if2_exact_tol_db: float = 1.0
    quad_c_res_f: float = 1e-9
    quad_c_q_f: float = 10e-12
    quad_c_trim_f: float = 10e-12
    quad_r_p_ohm: float = 10e3
    quad_phase_deg: float = 90.0
    quad_phase_tol_deg: float = 5.0
    quad_exact_tol_deg: float = 2.0
    lo2_c_load_f: float = 20e-12
    lo2_c_stray_f: float = 5e-12
    h_below: float = 47.0
    h_above: float = 48.0
    c_dec_f: float = 100e-9
    r_rssi_ohm: float = 91e3
    c_rssi_f: float = 100e-9
    vcc_v: float = 5.0
    lin_points: int = 801
    lad_fstart_hz: float = 21.38e6
    lad_fstop_hz: float = 21.42e6
    wide_points: int = 200
    wide_fstart_hz: float = 1e6
    wide_fstop_hz: float = 100e6
    if2_points: int = 100
    if2_fstart_hz: float = 100e3
    if2_fstop_hz: float = 2e6
    quad_points: int = 2001
    quad_fstart_hz: float = 440e3
    quad_fstop_hz: float = 460e3


DEFAULTS = IfBackendDefaults()


def _value(x: float) -> str:
    return format_part_value(x)


def _natural_ref(ref: str) -> tuple:
    """``C9`` before ``C10``: letters, then the number."""
    m = re.match(r"^([A-Za-z_]+)(\d+)$", ref)
    return (ref, 0) if m is None else (m.group(1), int(m.group(2)))


class IfBackendBlock(Block):
    """The IF back-end (see the module docstring); ``input_connector=False`` leaves out ``J1`` (the transceiver feeds ``IF1`` from its front end)."""

    id = BLOCK_ID
    title = "IF back-end: 21.4 MHz crystal ladder, SA605D mixer / IF / limiter / quadrature detector / RSSI"
    interface_nets = INTERFACE_NETS

    def __init__(self, *, input_connector: bool = True, defaults: IfBackendDefaults = DEFAULTS) -> None:
        self.input_connector = input_connector
        self.d = defaults

    # ------------------------------------------------------------------ numbers

    def _plan_value(self, b: BlockBuilder, key: str, value: float, unit: str | None, description: str) -> Traced:
        """A plan-level value: the composing template's (``ctx.shared``) when it wrote one, else this block's own choice."""
        have = b.ctx.shared.get(key)
        return have if have is not None else b.choice(key, value, unit, description)

    @staticmethod
    def _copy(b: BlockBuilder, key: str, traced: Traced) -> Traced:
        """A confirmed requirement value copied into a parameter (a calculator names parameters, never requirement ids)."""
        return b._param(key, traced)

    def build_local(self, ctx: BlockContext) -> BlockResult:
        d = self.d
        b = BlockBuilder(ctx, BLOCK_ID, self.title, INTERFACE_NETS)
        missing = [k for k in PROFILE_READS if k not in ctx.shared]
        if missing:
            raise TemplateRefusal(f"block {BLOCK_ID} reads the KR 447 profile values {missing} from the composing template (BlockContext.shared), which did not write them")
        raster = ctx.shared["kr447.channel_raster"]
        spacing_in = ctx.inputs.get("channel_spacing")
        if spacing_in is not None:
            if float(spacing_in.traced.value) != float(raster.value):
                raise TemplateRefusal(
                    f"channel_spacing {float(spacing_in.traced.value):.12g} Hz ({spacing_in.requirement.id}) is not the KR 447 raster kr447.channel_raster "
                    f"{float(raster.value):.12g} Hz [UNVERIFIED]: the IF back-end's crystal ladder is designed for that raster's channel; another spacing is another "
                    f"radio class and a human design change")
        # ---- plan values (shared with the other blocks of a composition)
        if1 = self._plan_value(b, "rf.if1", d.if1_hz, "Hz", "first IF: a standard 21.4 MHz crystal-filter frequency (the ladder's centre, LO1 = f_c - IF1)")
        if2 = self._plan_value(b, "rf.if2", d.if2_hz, "Hz", "second IF: 450 kHz (47 x 455 kHz = 21.385 MHz would sit 15 kHz from IF1; the 450 kHz harmonics are 250 / 200 kHz away)")
        side = self._plan_value(b, "rf.lo2_side", d.lo2_side, None, "LO2 below IF1 (low side: sign -1 of calc.rf.superhet.lo)")
        if_bw = self._plan_value(b, "rf.if_bw", d.if_bw_hz, "Hz", "IF passband the ladder rows judge: 7.5 kHz = 0.6 x the 12.5 kHz channel spacing (+/- 3.75 kHz around the centre)")
        z0_in = ctx.inputs.get("system_impedance")
        z0 = self._copy(b, "ifb.z0", z0_in.traced) if z0_in is not None else self._plan_value(
            b, "rf.z0", d.z0_ohm, "ohm", "system impedance of the IF1 port when system_impedance is not stated (50 ohm, the bench generator's)")
        z0_key = "ifb.z0" if z0_in is not None else "rf.z0"
        spacing, spacing_key = (self._copy(b, "ifb.channel_spacing", spacing_in.traced), "ifb.channel_spacing") if spacing_in is not None else (raster, "kr447.channel_raster")
        dev_in = ctx.inputs.get("frequency_deviation")
        dev, dev_key = (self._copy(b, "ifb.deviation", dev_in.traced), "ifb.deviation") if dev_in is not None else (ctx.shared["kr447.max_deviation"], "kr447.max_deviation")
        # ---- choices
        lower = b.choice("ifb.lower", -1.0, None, "the sign of a frequency below a centre (f - delta through calc.rf.superhet.lo's f + side delta): -1")
        upper = b.choice("ifb.upper", 1.0, None, "the sign of a frequency above a centre (f + delta): +1")
        idx = {k: b.choice(f"ifb.idx.{k}", float(k), None, f"element index {k}: a counting number the ladder / top-C calculators take to name one element (not a design value)")
               for k in range(1, int(d.n_xtal) + 1)}
        n = b.choice("ifb.lad.n", d.n_xtal, None, (
            f"crystals in the ladder: a {int(d.n_xtal)}-pole Butterworth (with C_m 16 fF no 6-pole ladder holds the -40 dB adjacent-channel rows beside the "
            f"passband rows - the lower-sideband ladder's low skirt reaches -38.1 dB at best; 7 poles hold both)"))
        bw = b.choice("ifb.lad.bw_design", d.bw_design_hz, "Hz", (
            "the ladder's design bandwidth, pre-distorted above rf.if_bw: the crystals' motional loss (R_m) narrows the realised passband, so the ladder is designed "
            "wider and judged at rf.if_bw by its exact response (calc.crystal.ladder.s21_db / .center with R_m)"))
        pass_min = b.choice("ifb.lad.pass_min", d.pass_min_db, "dB", "passband rows: S21 at f0 -/+ rf.if_bw / 2 at least this far below S21 at f0")
        acs_max = b.choice("ifb.lad.acs_max", d.acs_max_db, "dB", "adjacent-channel rows: S21 at f0 -/+ the channel spacing at most this, relative to f0")
        lo2_max = b.choice("ifb.lad.lo2_max", d.lo2_max_db, "dB", "S21 at LO2 (its leakage back into IF1) at most this, relative to f0")
        image2_max = b.choice("ifb.lad.image2_max", d.image2_max_db, "dB", "S21 at the SA605 mixer's image 2 LO2 - IF1 at most this, relative to f0")
        centre_min = b.choice("ifb.lad.centre_min", d.centre_min_db, "dB", "S21 at IF1 itself at least this, relative to f0: IF1 lies in the flat passband")
        loss_tol = b.choice("ifb.lad.loss_tol", d.loss_tol_db, "dB", (
            "tolerance of the IF1 filter's loss row around the ladder's exact loss (the L-matches' inductor loss, about 0.7 dB at Q 30, lies inside it)"))
        exact_tol = b.choice("ifb.lad.exact_tol", d.exact_tol_db, "dB", "tolerance of the ladder-alone rows against calc.crystal.ladder.s21_db (the netlist realises the designed ladder)")
        # ---- model values
        cm = b.model(XTAL_CM)
        rm = b.model("model.xtal21.rm")
        c0 = b.model("model.xtal21.c0")
        port_r = b.model("model.sa605.port_r")
        rf_in_r = b.model(SA605_RF_IN)
        lim_out_r = b.model(SA605_LIM_OUT)
        q_if1 = b.model("model.l_q.if1")
        q_if2 = b.model("model.l_q.if2")
        # ---- frequency plan
        try:
            lo2 = b.computed("ifb.lo2.f", radio.superhet_lo(if1, if2, side, ("rf.if1", "rf.if2", "rf.lo2_side")))
            image2 = b.computed("ifb.image2", radio.superhet_image(if1, lo2, ("rf.if1", "ifb.lo2.f")))
            h_lo = b.choice("ifb.if2.h_below", d.h_below, None, "the IF2 harmonic order just below IF1 (47 x 450 kHz = 21.15 MHz; IF1 / IF2 = 47.6)")
            h_hi = b.choice("ifb.if2.h_above", d.h_above, None, "the IF2 harmonic order just above IF1 (48 x 450 kHz = 21.6 MHz)")
            f_h_lo = b.computed("ifb.if2.harm_below", radio.harmonic(if2, h_lo, ("rf.if2", "ifb.if2.h_below")))
            f_h_hi = b.computed("ifb.if2.harm_above", radio.harmonic(if2, h_hi, ("rf.if2", "ifb.if2.h_above")))
            # ---- the ladder
            c_at_if1 = b.computed("ifb.lad.center_at_if1", radio.ladder_center(n, bw, if1, cm, c0, rm, ("ifb.lad.n", "ifb.lad.bw_design", "rf.if1", "model.xtal21.cm",
                                                                                                       "model.xtal21.c0", "model.xtal21.rm")))
            fs = b.computed("ifb.lad.xtal_fs", radio.superhet_image(c_at_if1, if1, ("ifb.lad.center_at_if1", "rf.if1")))
            lad_ids = ("ifb.lad.n", "ifb.lad.bw_design", "ifb.lad.xtal_fs", "model.xtal21.cm", "model.xtal21.c0")
            f0 = b.computed("ifb.lad.f0", radio.ladder_center(n, bw, fs, cm, c0, rm, (*lad_ids, "model.xtal21.rm")))
            lm = b.computed("ifb.lad.xtal_lm", crystal_lm(fs, cm, ("ifb.lad.xtal_fs", "model.xtal21.cm")))
            r_end = b.computed("ifb.lad.r_end", radio.ladder_r_end(n, bw, fs, cm, c0, lad_ids))
            order = int(n.value)
            design = radio.crystal_ladder(order, float(bw.value), float(fs.value), float(cm.value), float(c0.value))
            couple = [b.computed(f"ifb.lad.c_couple.{i}", radio.ladder_c_couple(n, idx[i], bw, fs, cm, c0, (lad_ids[0], f"ifb.idx.{i}", *lad_ids[1:])))
                      for i in range(1, order)]
            mesh: dict[int, Traced] = {}
            for i in range(1, order + 1):
                if design.c_mesh[i - 1] is not None:
                    mesh[i] = b.computed(f"ifb.lad.c_mesh.{i}", radio.ladder_mesh_c(n, idx[i], bw, fs, cm, c0, (lad_ids[0], f"ifb.idx.{i}", *lad_ids[1:])))
            two = b.choice("ifb.half", 2.0, None, "the divisor that halves rf.if_bw into the +/- offset of the passband rows (a counting number)")
            half_bw = b.computed("ifb.lad.half_bw", clock_divided(if_bw, two, ("rf.if_bw", "ifb.half")))
            f_pass_lo = b.computed("ifb.lad.f_pass_lo", radio.superhet_lo(f0, half_bw, lower, ("ifb.lad.f0", "ifb.lad.half_bw", "ifb.lower")))
            f_pass_hi = b.computed("ifb.lad.f_pass_hi", radio.superhet_lo(f0, half_bw, upper, ("ifb.lad.f0", "ifb.lad.half_bw", "ifb.upper")))
            f_acs_lo = b.computed("ifb.lad.f_acs_lo", radio.superhet_lo(f0, spacing, lower, ("ifb.lad.f0", spacing_key, "ifb.lower")))
            f_acs_hi = b.computed("ifb.lad.f_acs_hi", radio.superhet_lo(f0, spacing, upper, ("ifb.lad.f0", spacing_key, "ifb.upper")))
            s21_ids = (*lad_ids, "model.xtal21.rm")
            s21 = {name: b.computed(f"ifb.lad.s21.{name}", radio.ladder_s21_db(n, bw, fs, cm, c0, rm, f, (*s21_ids, key)))
                   for name, f, key in (("f0", f0, "ifb.lad.f0"), ("pass_lo", f_pass_lo, "ifb.lad.f_pass_lo"), ("pass_hi", f_pass_hi, "ifb.lad.f_pass_hi"),
                                        ("acs_lo", f_acs_lo, "ifb.lad.f_acs_lo"), ("acs_hi", f_acs_hi, "ifb.lad.f_acs_hi"))}
            # ---- the two L-matches
            m_in = lmatch(float(if1.value), float(z0.value), float(r_end.value), "lowpass")
            l_in = b.computed("ifb.match_in.l", lmatch_lowpass_l_series(if1, z0, r_end, ("rf.if1", z0_key, "ifb.lad.r_end")))
            c_in = b.computed("ifb.match_in.c", lmatch_lowpass_c_shunt(if1, z0, r_end, ("rf.if1", z0_key, "ifb.lad.r_end")))
            m_out = lmatch(float(if1.value), float(r_end.value), float(rf_in_r.value), "lowpass")
            l_out = b.computed("ifb.match_out.l", lmatch_lowpass_l_series(if1, r_end, rf_in_r, ("rf.if1", "ifb.lad.r_end", "model.sa605.rf_in_r")))
            c_out = b.computed("ifb.match_out.c", lmatch_lowpass_c_shunt(if1, r_end, rf_in_r, ("rf.if1", "ifb.lad.r_end", "model.sa605.rf_in_r")))
            # ---- LO2 oscillator
            c_load = b.choice("ifb.lo2.c_load", d.lo2_c_load_f, "F", "load capacitance the LO2 crystal is specified at (its frequency is stated at this load) [UNVERIFIED: crystal datasheet]")
            c_stray = b.choice("ifb.lo2.c_stray", d.lo2_c_stray_f, "F", "stray capacitance of the oscillator pins and traces added to the series load (an estimate, not measured)")
            c_osc = b.computed("ifb.lo2.c_osc", crystal_c_for_load(c_load, c_stray, ("ifb.lo2.c_load", "ifb.lo2.c_stray")))
            tol_in = ctx.inputs.get("frequency_tolerance")
            lo2_df = None
            if tol_in is not None:
                tol = self._copy(b, "ifb.lo2.tolerance", tol_in.traced)
                lo2_df = b.computed("ifb.lo2.df_max", ppm_offset(lo2, tol, ("ifb.lo2.f", "ifb.lo2.tolerance")))
            # ---- IF2 filters
            n2 = b.choice("ifb.if2.n", d.if2_n, None, "resonators of each 450 kHz IF2 filter: a 2-pole Butterworth top-C network (the design's)")
            bw2 = b.choice("ifb.if2.bw", d.if2_bw_hz, "Hz", "bandwidth of the IF2 filters: 30 kHz, wide against the 8 kHz channel (the ladder sets the selectivity; these clean up the mixer products)")
            l2 = b.choice("ifb.if2.l", d.if2_l_h, "H", "resonator inductance of the IF2 filters (1210 package); the taps into 1.5 kohm need R_p = Q_e w0 L above it")
            off2 = b.choice("ifb.if2.offset", d.if2_offset_hz, "Hz", "offset of the IF2 filters' rejection rows from IF2 (+/- 100 kHz)")
            s21_min2 = b.choice("ifb.if2.s21_min", d.if2_s21_min_db, "dB", "IF2 filter rows: S21 at IF2 at least this")
            rej_max2 = b.choice("ifb.if2.rej_max", d.if2_rej_max_db, "dB", "IF2 filter rows: S21 at IF2 -/+ 100 kHz at most this, relative to IF2")
            exact2 = b.choice("ifb.if2.exact_tol", d.if2_exact_tol_db, "dB", "tolerance of the IF2 rows against the exact top-C network (calc.rf.resonator.top_c.*)")
            tc_ids = ("ifb.if2.n", "rf.if2", "ifb.if2.bw", "ifb.if2.l")
            tap = b.computed("ifb.if2.c_tap", radio.top_c_c_tap(n2, if2, bw2, l2, port_r, (*tc_ids, "model.sa605.port_r")))
            cc2 = b.computed("ifb.if2.c_couple", radio.top_c_c_couple(n2, idx[1], if2, bw2, l2, ("ifb.if2.n", "ifb.idx.1", *tc_ids[1:])))
            shunt2 = [b.computed(f"ifb.if2.c_shunt.{i}", radio.top_c_c_shunt(n2, idx[i], if2, bw2, l2, port_r, port_r,
                                                                             ("ifb.if2.n", f"ifb.idx.{i}", *tc_ids[1:], "model.sa605.port_r", "model.sa605.port_r")))
                      for i in (1, 2)]
            f2_lo = b.computed("ifb.if2.f_lo", radio.superhet_lo(if2, off2, lower, ("rf.if2", "ifb.if2.offset", "ifb.lower")))
            f2_hi = b.computed("ifb.if2.f_hi", radio.superhet_lo(if2, off2, upper, ("rf.if2", "ifb.if2.offset", "ifb.upper")))
            tc_s21_ids = (*tc_ids, "model.sa605.port_r", "model.sa605.port_r", "model.l_q.if2")
            s21_2 = b.computed("ifb.if2.s21", radio.top_c_s21_db(n2, if2, bw2, l2, port_r, port_r, q_if2, if2, (*tc_s21_ids, "rf.if2")))
            rel2_lo = b.computed("ifb.if2.rel_lo", radio.top_c_rel_s21_db(n2, if2, bw2, l2, port_r, port_r, q_if2, f2_lo, if2, (*tc_s21_ids, "ifb.if2.f_lo", "rf.if2")))
            rel2_hi = b.computed("ifb.if2.rel_hi", radio.top_c_rel_s21_db(n2, if2, bw2, l2, port_r, port_r, q_if2, f2_hi, if2, (*tc_s21_ids, "ifb.if2.f_hi", "rf.if2")))
            # ---- quadrature tank
            c_res_q = b.choice("ifb.quad.c_res", d.quad_c_res_f, "F", "total capacitance the quadrature tank resonates with at IF2 (1 nF; L follows)")
            c_q = b.choice("ifb.quad.c_q", d.quad_c_q_f, "F", "coupling capacitor LIMITER_OUT -> QUADRATURE_IN (10 pF): it and the tank give the 90 degree phase at IF2")
            c_tq = b.choice("ifb.quad.c_trim", d.quad_c_trim_f, "F", "quadrature trimmer at its mid position (alignment: the discriminator's zero at IF2)")
            r_p = b.choice("ifb.quad.r_p", d.quad_r_p_ohm, "ohm", "the resistor across the quadrature tank: sets its Q (about 28) and so the discriminator's slope and linear range")
            ph_target = b.choice("ifb.quad.phase_target", d.quad_phase_deg, "deg", "phase of V(QUAD) / V(LIMITER_OUT) at IF2 the detector needs (quadrature: its output is zero there)")
            ph_tol = b.choice("ifb.quad.phase_tol", d.quad_phase_tol_deg, "deg", "tolerance of the quadrature row at IF2 (the trimmer aligns the rest)")
            ph_exact = b.choice("ifb.quad.exact_tol", d.quad_exact_tol_deg, "deg", "tolerance of the quadrature rows against calc.rf.quad.phase (the netlist realises the designed network)")
            l_q = b.computed("ifb.quad.l", lc_l_for_resonance(if2, c_res_q, ("rf.if2", "ifb.quad.c_res")))
            c_fix_q = b.computed("ifb.quad.c_fixed", radio.pm_c_fixed(c_res_q, c_q, c_tq, ("ifb.quad.c_res", "ifb.quad.c_q", "ifb.quad.c_trim")))
            q_tank = b.computed("ifb.quad.q", radio.q_parallel(if2, l_q, r_p, ("rf.if2", "ifb.quad.l", "ifb.quad.r_p")))
            fq_lo = b.computed("ifb.quad.f_lo", radio.superhet_lo(if2, dev, lower, ("rf.if2", dev_key, "ifb.lower")))
            fq_hi = b.computed("ifb.quad.f_hi", radio.superhet_lo(if2, dev, upper, ("rf.if2", dev_key, "ifb.upper")))
            qp_ids = ("model.sa605.lim_out_r", "ifb.quad.c_q", "ifb.quad.l", "model.l_q.if2", "rf.if2", "ifb.quad.c_fixed", "ifb.quad.c_trim", "ifb.quad.r_p")
            phases = {name: b.computed(f"ifb.quad.phase.{name}", radio.quad_phase(f, lim_out_r, c_q, l_q, q_if2, if2, c_fix_q, c_tq, r_p, (key, *qp_ids)))
                      for name, f, key in (("lo", fq_lo, "ifb.quad.f_lo"), ("mid", if2, "rf.if2"), ("hi", fq_hi, "ifb.quad.f_hi"))}
            # ---- RSSI, decoupling, supply
            c_dec = b.choice("ifb.c_dec", d.c_dec_f, "F", "SA605 decoupling (VCC, RF_BYPASS, the limiter and IF amplifier decoupling pins): 100 nF ceramic each [UNVERIFIED: NXP SA605 application circuit]")
            r_rssi = b.choice("ifb.rssi.r", d.r_rssi_ohm, "ohm", "RSSI load resistor: RSSI_OUT is a current output, this turns it into the RSSI voltage [UNVERIFIED: NXP SA605 application circuit]")
            c_rssi = b.choice("ifb.rssi.c", d.c_rssi_f, "F", "RSSI filter capacitor: with the load resistor it smooths the RSSI voltage the squelch compares")
            tau_rssi = b.computed("ifb.rssi.tau", rc_time_constant(r_rssi, c_rssi, ("ifb.rssi.r", "ifb.rssi.c")))
            vcc = b.choice("ifb.vcc", d.vcc_v, "V", "the SA605 supply RX_5V (the RX power section's 5 V rail; the SA605's VCC range, about 4.5-8 V, is [UNVERIFIED: NXP SA605])")
        except ValueError as e:
            raise TemplateRefusal(f"block {BLOCK_ID}: {e}") from e
        # ---- sweeps (fixture ac analyses)
        lin = b.choice("ifb.sweep.lin", "lin", None, "linear ac sweep variation")
        dec = b.choice("ifb.sweep.dec", "dec", None, "logarithmic ac sweep variation")
        sweep_pts = {
            "lad_points": b.choice("ifb.sweep.lad_points", d.lin_points, None, "points of the ladder's linear sweep (50 Hz apart over 40 kHz; the rows are read at their own point analyses, the sweep is the figure)"),
            "lad_fstart": b.choice("ifb.sweep.lad_fstart", d.lad_fstart_hz, "Hz", "start of the ladder's linear sweep (IF1 - 20 kHz)"),
            "lad_fstop": b.choice("ifb.sweep.lad_fstop", d.lad_fstop_hz, "Hz", "stop of the ladder's linear sweep (IF1 + 20 kHz)"),
            "wide_points": b.choice("ifb.sweep.wide_points", d.wide_points, None, "points per decade of the IF1 filter's wide sweep"),
            "wide_fstart": b.choice("ifb.sweep.wide_fstart", d.wide_fstart_hz, "Hz", "start of the IF1 filter's wide sweep"),
            "wide_fstop": b.choice("ifb.sweep.wide_fstop", d.wide_fstop_hz, "Hz", "stop of the IF1 filter's wide sweep"),
            "if2_points": b.choice("ifb.sweep.if2_points", d.if2_points, None, "points per decade of the IF2 filters' sweep"),
            "if2_fstart": b.choice("ifb.sweep.if2_fstart", d.if2_fstart_hz, "Hz", "start of the IF2 filters' sweep"),
            "if2_fstop": b.choice("ifb.sweep.if2_fstop", d.if2_fstop_hz, "Hz", "stop of the IF2 filters' sweep"),
            "quad_points": b.choice("ifb.sweep.quad_points", d.quad_points, None, "points of the quadrature tank's linear sweep"),
            "quad_fstart": b.choice("ifb.sweep.quad_fstart", d.quad_fstart_hz, "Hz", "start of the quadrature tank's linear sweep"),
            "quad_fstop": b.choice("ifb.sweep.quad_fstop", d.quad_fstop_hz, "Hz", "stop of the quadrature tank's linear sweep"),
        }

        def ac(aid: str, variation: Traced, points: str, fstart: str, fstop: str, note: str) -> AnalysisSpec:
            return AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": variation, "points": sweep_pts[points], "fstart": sweep_pts[fstart], "fstop": sweep_pts[fstop]},
                                provenance=ctx.provenance(note))

        # ---- parts
        req = ctx.requirement_id
        serves_z0 = [r for r in (req("system_impedance"),) if r]
        serves_spacing = [r for r in (req("channel_spacing"),) if r]
        serves_dev = [r for r in (req("frequency_deviation"),) if r]
        serves_tol = [r for r in (req("frequency_tolerance"),) if r]
        # the SA605D is the FM IF system that makes this board the one radio_build / modulation ask for (categorical keys: no DesignInput)
        serves_fm = [r.id for r in ctx.ir.requirements.requirements
                     if canonical_key(r.key) in ("modulation", "radio_build") and r.value is not None and r.value.provenance.is_authoritative]
        parts: dict[str, PlacedPart] = {}
        counter = {"C": 0}

        def cap(value: float, description: str, *, key: str = "cap_0603") -> str:
            counter["C"] += 1
            ref = f"C{counter['C']}"
            parts[ref] = b.part(key, ref, _value(value), description)
            return ref

        if self.input_connector:
            parts["J1"] = b.part("coax_ufl", "J1", "U.FL", "IF1 21.4 MHz input (signal generator or the front end's IF1 output, 50 ohm)", serves_z0)
        parts["L1"] = b.part("ind_0603", "L1", _value(float(l_in.value)), f"IF1 input L-match series inductor ({float(z0.value):.6g} ohm -> R_end)")
        c_match_in = cap(float(c_in.value), "IF1 input L-match shunt capacitor (at the ladder end)")
        xtal_refs = [f"Y{i}" for i in range(1, order + 1)]
        for i, ref in enumerate(xtal_refs, 1):
            parts[ref] = b.part("crystal", ref, _value(float(fs.value)), (
                f"ladder crystal {i} of {order}: series resonance {float(fs.value):.1f} Hz (a matched set; C_m >= {float(cm.value) * 1e15:.6g} fF with C0 <= "
                f"{float(c0.value) * 1e12:.6g} pF) [UNVERIFIED: crystal datasheet]"), serves_spacing)
        couple_refs = [cap(float(t.value), f"ladder coupling capacitor between crystals {i} and {i + 1} (NP0)") for i, t in enumerate(couple, 1)]
        mesh_refs: dict[int, str] = {}
        trims = iter(("C_T1", "C_T2"))
        for i, t in mesh.items():
            if i in (1, order):
                ref = next(trims)
                parts[ref] = b.part("ctrim", ref, _value(float(t.value)), f"ladder mesh {i} tuning trimmer (set to the calculator's value at alignment)")
                mesh_refs[i] = ref
            else:
                mesh_refs[i] = cap(float(t.value), f"ladder mesh {i} tuning capacitor (NP0)")
        parts["L2"] = b.part("ind_0603", "L2", _value(float(l_out.value)), "ladder output L-match series inductor (R_end -> SA605 RF_IN)")
        c_match_out = cap(float(c_out.value), "ladder output L-match shunt capacitor")
        parts["U1"] = b.part("fm_if", "U1", "SA605D", "FM IF system: mixer, LO2 oscillator, IF amplifier, limiter, quadrature detector, RSSI, mute", serves_fm)
        y_lo2 = f"Y{order + 1}"
        tol_text = f"; tolerance <= {float(tol_in.traced.value):.6g} ppm ({tol_in.requirement.id})" if tol_in is not None else ""
        parts[y_lo2] = b.part("crystal", y_lo2, _value(float(lo2.value)), (
            f"LO2 crystal {float(lo2.value) / 1e6:.6g} MHz at a {float(c_load.value) * 1e12:.6g} pF load{tol_text} [UNVERIFIED: crystal datasheet]"), serves_tol)
        c_osc_b = cap(float(c_osc.value), "LO2 Colpitts capacitor OSC_IN -> OSC_OUT")
        c_osc_e = cap(float(c_osc.value), "LO2 Colpitts capacitor OSC_OUT -> GND")
        if2_refs: dict[str, dict[str, str]] = {}
        l_count = iter(range(3, 7))
        for f_id, what in (("a", "MIXER_OUT -> IF_AMP_IN"), ("b", "IF_AMP_OUT -> LIMITER_IN")):
            r: dict[str, str] = {}
            r["tap_in"] = cap(float(tap.value), f"IF2 filter {f_id} ({what}) input tap capacitor")
            for i in (1, 2):
                lref = f"L{next(l_count)}"
                parts[lref] = b.part("ind_1210", lref, _value(float(l2.value)), f"IF2 filter {f_id} resonator {i} inductor")
                r[f"l{i}"] = lref
                r[f"sh{i}"] = cap(float(shunt2[i - 1].value), f"IF2 filter {f_id} resonator {i} capacitor")
            r["couple"] = cap(float(cc2.value), f"IF2 filter {f_id} coupling capacitor")
            r["tap_out"] = cap(float(tap.value), f"IF2 filter {f_id} output tap capacitor")
            if2_refs[f_id] = r
        c_qref = cap(float(c_q.value), "quadrature coupling capacitor LIMITER_OUT -> QUADRATURE_IN")
        parts["L7"] = b.part("ind_1210", "L7", _value(float(l_q.value)), "quadrature tank inductor (returned to RX_5V)", serves_dev)
        c_qfix = cap(float(c_fix_q.value), "quadrature tank capacitor")
        parts["C_T3"] = b.part("ctrim", "C_T3", _value(float(c_tq.value)), "quadrature tank trimmer (mid position: the discriminator's zero at IF2)")
        parts["R1"] = b.part("res_0603", "R1", _value(float(r_p.value)), "quadrature tank damping resistor (sets the discriminator slope)")
        dec_refs = {name: cap(float(c_dec.value), f"SA605 {name} decoupling") for name in ("VCC", "RF_BYPASS", "LIMITER_DECOUPL 12", "LIMITER_DECOUPL 13",
                                                                                        "IF_AMP_DECOUPL 17", "IF_AMP_DECOUPL 19")}
        parts["R2"] = b.part("res_0603", "R2", _value(float(r_rssi.value)), "RSSI load resistor")
        c_rssi_ref = cap(float(c_rssi.value), "RSSI filter capacitor")
        tps = {"TP1": "IF2 at the limiter input", "TP2": "quadrature node", "TP3": "RSSI voltage", "TP4": "unmuted discriminator output (alignment)"}
        for ref, what in tps.items():
            parts[ref] = b.part("testpoint", ref, "TP", f"alignment test point: {what}")

        def at(ref: str, function: str) -> list[tuple[str, str]]:
            return parts[ref].at(function)

        def one(ref: str, which: int) -> tuple[str, str]:
            """Pin ``which`` (1 / 2) of a symmetric two-terminal part, in library order."""
            return parts[ref].at(str(which))[0]

        u = "U1"
        # ---- nets
        nets: list[tuple[str, NetKind, list[tuple[str, str]], str]] = []
        gnd: list[tuple[str, str]] = []

        def net(name: str, members: list[tuple[str, str]], note: str, kind: NetKind = NetKind.ANALOG) -> None:
            nets.append((name, kind, members, note))

        if1_members = [one("L1", 1)] + (at("J1", "In") if self.input_connector else [])
        if self.input_connector:
            gnd += at("J1", "Ext")
        net("IF1", if1_members, "IF1 21.4 MHz input (50 ohm)", NetKind.RF)
        # the ladder: node 0 = LAD_IN, node n = LAD_OUT, mesh i between nodes i-1 and i; shunt coupling i at node i
        node = ["LAD_IN", *[f"LAD_N{i}" for i in range(1, order)], "LAD_OUT"]
        node_members: dict[str, list[tuple[str, str]]] = {name: [] for name in node}
        node_members["LAD_IN"] += [one("L1", 2), one(c_match_in, 1)]
        gnd.append(one(c_match_in, 2))
        chain_ladder: list[str] = []
        for i in range(1, order + 1):
            y = xtal_refs[i - 1]
            if i in mesh_refs:
                cref = mesh_refs[i]
                m = f"LAD_M{i}"
                node_members[node[i - 1]].append(one(cref, 1))
                net(m, [one(cref, 2), one(y, 1)], f"ladder mesh {i}: tuning capacitor to crystal {i}")
                chain_ladder += [cref, y]
            else:
                node_members[node[i - 1]].append(one(y, 1))
                chain_ladder.append(y)
            node_members[node[i]].append(one(y, 2))
            if i < order:
                cc = couple_refs[i - 1]
                node_members[node[i]].append(one(cc, 1))
                gnd.append(one(cc, 2))
                chain_ladder.append(cc)
        node_members["LAD_OUT"].append(one("L2", 1))
        for name in node:
            net(name, node_members[name], "crystal ladder node" if name.startswith("LAD_N") else f"crystal ladder {'input' if name == 'LAD_IN' else 'output'}")
        net("MIX_IN", [one("L2", 2), one(c_match_out, 1), *at(u, "RF_IN")], "SA605 mixer input (ladder output match)")
        gnd.append(one(c_match_out, 2))
        net("RF_BYP", [*at(u, "RF_BYPASS"), one(dec_refs["RF_BYPASS"], 1)], "SA605 mixer input bypass")
        gnd.append(one(dec_refs["RF_BYPASS"], 2))
        net("OSC_B", [*at(u, "OSC_IN"), one(y_lo2, 1), one(c_osc_b, 1)], "LO2 oscillator base: crystal to GND, Colpitts divider")
        net("OSC_E", [*at(u, "OSC_OUT"), one(c_osc_b, 2), one(c_osc_e, 1)], "LO2 oscillator emitter")
        gnd += [one(y_lo2, 2), one(c_osc_e, 2)]
        # IF2 filters: port node - tap - R1 (L, C) - couple - R2 (L, C) - tap - port node
        for f_id, src, dst in (("a", "MIXER_OUT", "IF_AMP_IN"), ("b", "IF_AMP_OUT", "LIMITER_IN")):
            r = if2_refs[f_id]
            up = f"F2{f_id.upper()}"
            src_net = {"MIXER_OUT": "MIX_OUT", "IF_AMP_OUT": "IFA_OUT"}[src]
            dst_net = {"IF_AMP_IN": "IFA_IN", "LIMITER_IN": "LIM_IN"}[dst]
            net(src_net, [*at(u, src), one(r["tap_in"], 1)], f"SA605 {src} -> IF2 filter {f_id}")
            net(f"{up}_R1", [one(r["tap_in"], 2), one(r["l1"], 1), one(r["sh1"], 1), one(r["couple"], 1)], f"IF2 filter {f_id} resonator 1")
            net(f"{up}_R2", [one(r["couple"], 2), one(r["l2"], 1), one(r["sh2"], 1), one(r["tap_out"], 1)], f"IF2 filter {f_id} resonator 2")
            dst_members = [*at(u, dst), one(r["tap_out"], 2)] + ([parts["TP1"].at("1")[0]] if dst == "LIMITER_IN" else [])
            net(dst_net, dst_members, f"IF2 filter {f_id} -> SA605 {dst}")
            gnd += [one(r["l1"], 2), one(r["sh1"], 2), one(r["l2"], 2), one(r["sh2"], 2)]
        net("LIM_OUT", [*at(u, "LIMITER_OUT"), one(c_qref, 1)], "SA605 limiter output -> quadrature coupling capacitor")
        net("QUAD", [*at(u, "QUADRATURE_IN"), one(c_qref, 2), one("L7", 1), one(c_qfix, 1), one("C_T3", 1), one("R1", 1), parts["TP2"].at("1")[0]],
            "quadrature tank node")
        limdec = parts[u].pins("LIMITER_DECOUPL")
        ifdec = parts[u].pins("IF_AMP_DECOUPL")
        net("LIMDEC1", [(u, limdec[0]), one(dec_refs["LIMITER_DECOUPL 12"], 1)], "SA605 limiter decoupling (first pin)")
        net("LIMDEC2", [(u, limdec[1]), one(dec_refs["LIMITER_DECOUPL 13"], 1)], "SA605 limiter decoupling (second pin)")
        net("IFDEC1", [(u, ifdec[0]), one(dec_refs["IF_AMP_DECOUPL 17"], 1)], "SA605 IF amplifier decoupling (first pin)")
        net("IFDEC2", [(u, ifdec[1]), one(dec_refs["IF_AMP_DECOUPL 19"], 1)], "SA605 IF amplifier decoupling (second pin)")
        gnd += [one(dec_refs[k], 2) for k in ("LIMITER_DECOUPL 12", "LIMITER_DECOUPL 13", "IF_AMP_DECOUPL 17", "IF_AMP_DECOUPL 19")]
        net("RX_5V", [*at(u, "VCC"), one(dec_refs["VCC"], 1), one("L7", 2), one(c_qfix, 2), one("C_T3", 2), one("R1", 2)],
            "SA605 supply (the RX power section's 5 V rail); the quadrature tank returns here", NetKind.POWER)
        gnd.append(one(dec_refs["VCC"], 2))
        net("RSSI", [*at(u, "RSSI_OUT"), one("R2", 1), one(c_rssi_ref, 1), parts["TP3"].at("1")[0]], "RSSI voltage -> the squelch comparator")
        gnd += [one("R2", 2), one(c_rssi_ref, 2)]
        net("DISC_OUT", at(u, "MUTED_AUD_OUTP"), "muted audio output -> the RX audio block (de-emphasis, 300-3000 Hz, volume)")
        net("AUD_UNMUTED", [*at(u, "UNMUTED_AUD_OUTP"), parts["TP4"].at("1")[0]], "unmuted discriminator output (alignment test point)")
        net("MUTE", at(u, "MUTE_INPUT"), "mute input <- the squelch comparator", NetKind.SIGNAL)
        gnd += at(u, "GND")
        net(GROUND_NET, gnd, "ground", NetKind.GROUND)
        for name, kind, members, note in nets:
            serves = serves_z0 if name == "IF1" else []
            b.net(name, kind, members, note, serves)
        # ---- SPICE: every part is a fixture member only
        for ref in parts:
            b.bind(ref, SpiceBinding(exclude=True, exclude_reason=DECK_EXCLUDED, provenance=ctx.provenance(f"{ref}: fixture-only part")))
        # ---- fixture networks
        card = crystal_card(lm, cm, rm, c0)
        card_text = b.card(card)
        xtal_binding = SpiceBinding(device=SpiceDevice.X, model_name=card.name, model_card=card_text, pin_order=["1", "2"],
                                    provenance=ctx.provenance("crystal as its Butterworth-Van Dyke card (model.xtal21)"))

        def val(ref: str, device: SpiceDevice, t: Traced, note: str) -> SpiceBinding:
            return SpiceBinding(device=device, value=t, pin_order=[parts[ref].pin("1"), parts[ref].pin("2")], provenance=ctx.provenance(f"{ref}: {note}"))

        lad_bind: dict[str, SpiceBinding] = {y: xtal_binding for y in xtal_refs}
        for i, ref in enumerate(couple_refs, 1):
            lad_bind[ref] = val(ref, SpiceDevice.C, couple[i - 1], f"coupling capacitor {i} at its calculated value")
        for i, ref in mesh_refs.items():
            lad_bind[ref] = val(ref, SpiceDevice.C, mesh[i], f"mesh {i} tuning capacitor at its calculated value")
        filt_bind = dict(lad_bind)
        filt_bind["L1"] = val("L1", SpiceDevice.L, l_in, "input L-match inductor")
        filt_bind[c_match_in] = val(c_match_in, SpiceDevice.C, c_in, "input L-match capacitor")
        filt_bind["L2"] = val("L2", SpiceDevice.L, l_out, "output L-match inductor")
        filt_bind[c_match_out] = val(c_match_out, SpiceDevice.C, c_out, "output L-match capacitor")

        def port(name: str, net_name: str, z: Traced | None, f: Traced | None, *, kind: str = "port", voltage: Traced | None = None, direction: str = "bidir") -> RFPort:
            return RFPort(name=name, net=net_name, kind=kind, z0_ohm=z, frequency_hz=f, voltage_v=voltage, direction=direction)

        def row(eid: str, quantity: str, drive: str, to: str, f: Traced, nominal: Traced, *, ref_at: Traced | None = None, tol: Traced | None = None,
                bound: str | None = None) -> RFExpectation:
            return RFExpectation(id=eid, quantity=quantity, drive=drive, to=to, at=f, ref_at=ref_at, nominal=nominal, tol_abs=tol, bound=bound)

        lad_ports = [port("LADI", "LAD_IN", r_end, if1), port("LADO", "LAD_OUT", r_end, if1)]
        networks: list[RFNetwork] = [
            RFNetwork(
                id="if1_ladder", block=BLOCK_ID, members=[*xtal_refs, *couple_refs, *mesh_refs.values()], bindings=lad_bind, ports=lad_ports,
                sweep=[ac("lad_lin", lin, "lad_points", "lad_fstart", "lad_fstop", "the ladder's passband and adjacent channels, 50 Hz steps")],
                expectations=[row(f"s21_{name}", "s21_db", "LADI", "LADO", f, s21[name], tol=exact_tol) for name, f in (
                    ("f0", f0), ("pass_lo", f_pass_lo), ("pass_hi", f_pass_hi), ("acs_lo", f_acs_lo), ("acs_hi", f_acs_hi))],
            ),
            RFNetwork(
                id="if1_filter", block=BLOCK_ID, members=["L1", c_match_in, *xtal_refs, *couple_refs, *mesh_refs.values(), "L2", c_match_out],
                bindings=filt_bind, loss_q={"L1": q_if1, "L2": q_if1}, q_ref_hz=if1,
                ports=[port("IF1", "IF1", z0, if1, direction="in"), port("MIX", "MIX_IN", rf_in_r, if1, direction="out")],
                sweep=[ac("lad_lin", lin, "lad_points", "lad_fstart", "lad_fstop", "the IF1 filter's passband and adjacent channels, 50 Hz steps"),
                       ac("wide", dec, "wide_points", "wide_fstart", "wide_fstop", "the IF1 filter from 1 to 100 MHz (LO2, the second image)")],
                expectations=[
                    row("s21_f0", "s21_db", "IF1", "MIX", f0, s21["f0"], tol=loss_tol),
                    row("pass_lo", "rel_s21_db", "IF1", "MIX", f_pass_lo, pass_min, ref_at=f0, bound="at_least"),
                    row("pass_hi", "rel_s21_db", "IF1", "MIX", f_pass_hi, pass_min, ref_at=f0, bound="at_least"),
                    row("acs_lo", "rel_s21_db", "IF1", "MIX", f_acs_lo, acs_max, ref_at=f0, bound="at_most"),
                    row("acs_hi", "rel_s21_db", "IF1", "MIX", f_acs_hi, acs_max, ref_at=f0, bound="at_most"),
                    row("if1_centre", "rel_s21_db", "IF1", "MIX", if1, centre_min, ref_at=f0, bound="at_least"),
                    row("lo2", "rel_s21_db", "IF1", "MIX", lo2, lo2_max, ref_at=f0, bound="at_most"),
                    row("image2", "rel_s21_db", "IF1", "MIX", image2, image2_max, ref_at=f0, bound="at_most"),
                ],
            ),
        ]
        for f_id, src_net, dst_net in (("a", "MIX_OUT", "IFA_IN"), ("b", "IFA_OUT", "LIM_IN")):
            r = if2_refs[f_id]
            bind2 = {r["tap_in"]: val(r["tap_in"], SpiceDevice.C, tap, "input tap"), r["tap_out"]: val(r["tap_out"], SpiceDevice.C, tap, "output tap"),
                     r["couple"]: val(r["couple"], SpiceDevice.C, cc2, "coupling capacitor"),
                     r["l1"]: val(r["l1"], SpiceDevice.L, l2, "resonator 1 inductor"), r["l2"]: val(r["l2"], SpiceDevice.L, l2, "resonator 2 inductor"),
                     r["sh1"]: val(r["sh1"], SpiceDevice.C, shunt2[0], "resonator 1 capacitor"), r["sh2"]: val(r["sh2"], SpiceDevice.C, shunt2[1], "resonator 2 capacitor")}
            networks.append(RFNetwork(
                id=f"if2_bpf_{f_id}", block=BLOCK_ID, members=[r["tap_in"], r["l1"], r["sh1"], r["couple"], r["l2"], r["sh2"], r["tap_out"]], bindings=bind2,
                loss_q={r["l1"]: q_if2, r["l2"]: q_if2}, q_ref_hz=if2,
                ports=[port("P1", src_net, port_r, if2, direction="in"), port("P2", dst_net, port_r, if2, direction="out")],
                sweep=[ac("if2_dec", dec, "if2_points", "if2_fstart", "if2_fstop", "the IF2 filter from 100 kHz to 2 MHz")],
                expectations=[
                    row("s21_if2", "s21_db", "P1", "P2", if2, s21_min2, bound="at_least"),
                    row("rej_lo", "rel_s21_db", "P1", "P2", f2_lo, rej_max2, ref_at=if2, bound="at_most"),
                    row("rej_hi", "rel_s21_db", "P1", "P2", f2_hi, rej_max2, ref_at=if2, bound="at_most"),
                    row("s21_exact", "s21_db", "P1", "P2", if2, s21_2, tol=exact2),
                    row("rel_lo_exact", "rel_s21_db", "P1", "P2", f2_lo, rel2_lo, ref_at=if2, tol=exact2),
                    row("rel_hi_exact", "rel_s21_db", "P1", "P2", f2_hi, rel2_hi, ref_at=if2, tol=exact2),
                ],
            ))
        quad_bind = {c_qref: val(c_qref, SpiceDevice.C, c_q, "quadrature coupling capacitor"), "L7": val("L7", SpiceDevice.L, l_q, "tank inductor"),
                     c_qfix: val(c_qfix, SpiceDevice.C, c_fix_q, "tank capacitor"), "C_T3": val("C_T3", SpiceDevice.C, c_tq, "tank trimmer at mid"),
                     "R1": val("R1", SpiceDevice.R, r_p, "tank resistor")}
        networks.append(RFNetwork(
            id="quad_tank", block=BLOCK_ID, members=[c_qref, "L7", c_qfix, "C_T3", "R1"], bindings=quad_bind, loss_q={"L7": q_if2}, q_ref_hz=if2,
            ports=[port("LIMO", "LIM_OUT", lim_out_r, if2, direction="in"), port("QUADP", "QUAD", None, if2, kind="probe"),
                   port("VCC", "RX_5V", None, None, kind="rail", voltage=vcc)],
            sweep=[ac("quad_lin", lin, "quad_points", "quad_fstart", "quad_fstop", "the quadrature network around IF2")],
            expectations=[
                row("phase_if2", "phase21_deg", "LIMO", "QUADP", if2, ph_target, tol=ph_tol),
                row("phase_lo_exact", "phase21_deg", "LIMO", "QUADP", fq_lo, phases["lo"], tol=ph_exact),
                row("phase_if2_exact", "phase21_deg", "LIMO", "QUADP", if2, phases["mid"], tol=ph_exact),
                row("phase_hi_exact", "phase21_deg", "LIMO", "QUADP", fq_hi, phases["hi"], tol=ph_exact),
            ],
        ))
        res = b.result
        res.networks = networks
        res.ports = [port("IF1", "IF1", z0, if1, direction="in"), port("RX_5V", "RX_5V", None, None, kind="rail", voltage=vcc, direction="in")]
        # the signal path in order (IF1 -> ladder -> SA605) and the LO2 crystal beside the SA605, then the SA605's own parts tallest first (the
        # floorplan's shelf packer fills rows in this order, and a row is as tall as its tallest part)
        signal = [*(["J1"] if self.input_connector else []), "L1", c_match_in, *chain_ladder, "L2", c_match_out, u, y_lo2]
        rest = [r for r in parts if r not in signal]
        rank = {"ctrim": 0, "ind_1210": 1, "cap_0603": 2, "res_0603": 2, "testpoint": 3}
        res.chain = signal + sorted(rest, key=lambda r: (rank.get(parts[r].part.key, 2), _natural_ref(r)))
        # ---- frequency plan
        res.plan_lines = [
            PlanLine(id="ifb_if2_h_below", kind="margin", f_hz=f_h_lo, ref_hz=if1, min_margin_hz=spacing,
                     note="the IF2 harmonic just below IF1 (the limiter's square wave) stays a channel spacing away from IF1, where the ladder rejects it"),
            PlanLine(id="ifb_if2_h_above", kind="margin", f_hz=f_h_hi, ref_hz=if1, min_margin_hz=spacing,
                     note="the IF2 harmonic just above IF1 stays a channel spacing away from IF1"),
            PlanLine(id="ifb_lo2", kind="margin", f_hz=lo2, ref_hz=if1, min_margin_hz=spacing,
                     note="LO2 (its leakage back into the IF1 filter) stays a channel spacing away from IF1; its rejection is the if1_filter row lo2"),
            PlanLine(id="ifb_image2", kind="response", f_hz=image2, ref_hz=if1, points_to=["spice.rf.if1_filter.image2", "rf.lab.ifb_image2_rejection"],
                     note="the SA605 mixer's image 2 LO2 - IF1: rejected by the IF1 filter ahead of the mixer (fixture row image2: a network verdict under "
                          "the model crystals); the built ladder's rejection there is lab item ifb_image2_rejection, like every receiver response"),
        ]
        res.lab_items = [
            LabItem(id="ifb_xtal_motional", block=BLOCK_ID, what="motional C_m, L_m, R_m and C0 of every ladder crystal, and their matching (the model values model.xtal21.*)",
                    instruments=["crystal test fixture (G3UUR)", "VNA"], reason="the ladder rests on unverified model values of crystals not yet ordered"),
            LabItem(id="ifb_ladder_alignment", block=BLOCK_ID, what="ladder alignment (end trimmers), the real IF bandwidth, ripple and centre against IF1",
                    instruments=["signal generator", "spectrum analyser or VNA"], reason="a fixture verdict is a schematic network under model values, not the built filter"),
            LabItem(id="ifb_image2_rejection", block=BLOCK_ID,
                    what=f"rejection of the second image 2 LO2 - IF1 = {float(image2.value) / 1e6:.6g} MHz by the built ladder (and the response it leaves at the "
                         "limiter output)", instruments=["signal generator", "spectrum analyser or VNA", "SINAD meter"],
                    reason="the fixture row image2 judges the schematic ladder under model.xtal21 (UNVERIFIED crystals, no spurious modes, no board "
                           "coupling around the filter); only the built filter says how much it rejects"),
            LabItem(id="ifb_if2_quad_alignment", block=BLOCK_ID, what="IF2 filter tuning (inductor tolerance) and the quadrature tank's zero at IF2",
                    instruments=["signal generator", "oscilloscope"], reason="the resonators are simulated at nominal values; the trimmer covers only a small range"),
            LabItem(id="ifb_lo2_frequency", block=BLOCK_ID, what="LO2 frequency at its load capacitance (the SA605 oscillator with the Colpitts capacitors)",
                    instruments=["frequency counter"], reason="the SA605 oscillator is not simulated"),
            LabItem(id="ifb_sinad", block=BLOCK_ID, what="12 dB SINAD sensitivity at 21.4 MHz, recovered audio level and distortion",
                    instruments=["signal generator", "SINAD meter", "audio analyser"], reason="the SA605 has no model"),
            LabItem(id="ifb_rssi", block=BLOCK_ID, what="RSSI slope against input level (the squelch threshold rests on it)",
                    instruments=["signal generator", "DMM"], reason="RSSI_OUT's current is an SA605 datasheet fact [UNVERIFIED]"),
            LabItem(id="ifb_birdies", block=BLOCK_ID, what="birdies from the IF2 harmonics and LO2 in the IF1 passband",
                    instruments=["signal generator", "SINAD meter", "spectrum analyser"], reason="harmonic levels of the limiter are not modelled"),
        ]
        res.notes.append(
            f"if_backend: {order}-crystal ladder at f_s {float(fs.value):.1f} Hz (centre {float(f0.value):.3f} Hz, R_end {float(r_end.value):.1f} ohm), "
            f"design bandwidth {float(bw.value):.6g} Hz judged at {float(if_bw.value):.6g} Hz; the SA605D is excluded from every netlist ({MODEL_VERDICT} is the most a fixture says)")
        if lo2_df is not None:
            res.notes.append(f"LO2 crystal tolerance {float(tol_in.traced.value):.6g} ppm = {float(lo2_df.value):.6g} Hz at {float(lo2.value):.9g} Hz")
        res.shield_ref = None
        return b.done()


__all__ = [
    "BLOCK_ID",
    "DECK_EXCLUDED",
    "DEFAULTS",
    "INTERFACE_NETS",
    "NETWORKS",
    "PROFILE_READS",
    "SA605_LIM_OUT",
    "SA605_RF_IN",
    "XTAL_CM",
    "IfBackendBlock",
    "IfBackendDefaults",
]
