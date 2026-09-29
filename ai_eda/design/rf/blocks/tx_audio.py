"""The TX audio processor of the KR 447 MHz family: microphone amplifier, pre-emphasis, limiter, 4th-order splatter filter, integrator, PM_DRIVE.

Invariant: the audio chain bounds the transmitter's frequency deviation by
arithmetic a reader can re-derive - every value is a registered calculator's
output over confirmed choices, the confirmed requirements and UNVERIFIED
``model.*`` values - and the design deck checks each stage against its
calculator nominal. Indirect FM (Armstrong): phase modulation of the
*integrated* limited audio is frequency modulation, so with a multiplier N
after the modulator and a modulator slope K_pm the peak deviation is
Delta_f = N K_pm a V_max / (2 pi tau_i) (``rf.deviation``; ``a`` is the
PM_DRIVE-to-varactor factor of the TX exciter). On this block K_pm is the
model constant ``model.k_pm`` unless the board names another parameter
(``k_pm_key``: the TX exciter's network slope), so ``rf.deviation`` stays
NOT_VERIFIED here. Nothing claims the MAX9814 works: it has no model.

The chain (kr447 design §2.1, decision 4B; local references, re-based by 300
on a board; every op-amp an MCP6001-OT single, critic2):

* MK1 electret microphone, biased from U1's MICBIAS through R1 and coupled by
  C1 into U1 (MAX9814, AGC; its straps and timing capacitors are UNVERIFIED
  choices) -> ``MICOUT``. In the design deck ``MICOUT`` is the stimulus
  ``VMIC`` (a 1 kHz sine at ``tx.test_level``, +20 dB over a normal mic
  level - the overdrive test [UNVERIFIED test method] - with ``ac`` 1 V).
* ``VREF_TX`` = TX_3V3 / 2 (R4 / R5, C6): the single-supply signal ground.
* U2: C7 / R6 high-pass at ``tx.hpf_corner`` into the non-inverting input,
  pre-emphasis zero 1 + s R7 C8 (tau ``tx.preemph_tau`` 750 us, 212 Hz) ->
  ``PRE_OUT``. Nominal: ``calc.audio.highpass1.db_at`` +
  ``calc.audio.emphasis.db_at`` (the whole stage; the op-amp's finite GBW
  adds about +0.4 dB at 3 kHz under ``model.opamp``, inside the 0.5 dB row).
* Limiter: C10 / R8 into the anti-parallel clipper D1 / D2 to GND (``CLIP``,
  symmetric about 0 V, R9 bleeds it) -> C11 / R10 into the unity follower U5 ->
  ``LIM_OUT``. ``limiter_level`` checks the clip level ``tx.clip_vd`` on
  ``CLIP`` (a DC-free node: ``LIM_OUT`` sits on ``VREF_TX``, and MAX has no
  reference).
* 4th-order Butterworth splatter filter at ``audio_bandwidth``: U3 (Q 0.5412)
  then U6 (Q 1.3066), unity-gain Sallen-Key, R 10 kohm, C1 / C2 by
  ``calc.audio.sallen_key.*`` -> ``SPLAT_OUT``. Checked as the attenuation
  ``v(LIM_OUT)`` re ``v(SPLAT_OUT)``: 3.01 dB at the corner
  (``calc.rf.lpf.butterworth.attenuation``), at least 21 dB at twice the
  corner and 46 dB at the channel raster; and as harmonics of the clipped
  1 kHz tone (h3 / h5 / h7 at most -10 / -30 / -40 dBc).
* Deviation trim RV1 (a divider from ``VREF_TX`` to ``SPLAT_OUT`` at
  ``model.pot.position``) -> follower U7 -> C20 -> ``INT_IN`` (R15 to GND: a
  DC-free node, so its MAX is the level ``rf.deviation`` multiplies) -> C21 /
  R16 -> the lossy integrator U4 (C22 // R17, reference ``VREF_TX``) ->
  ``PM_DRIVE``. The design level at ``INT_IN`` is V_lim = V_D x trim
  (``calc.rf.limiter.level``); tau_i = N K_pm V_lim / (2 pi Delta_f_design)
  (``calc.rf.fm.pm_integrator_tau``) with Delta_f_design = the deviation /
  ``tx.deviation_headroom`` (``calc.clock.divided``: f / n); R16 = tau_i / C22.
  ``pm_drive_peak``: MAX v(INT_IN) at most ``calc.rf.fm.pm_drive_limit`` (the
  level that gives exactly the deviation requirement). ``integrator_1k``:
  v(PM_DRIVE) re v(INT_IN) at 1 kHz = the lossy integrator
  (``calc.audio.lossy_integrator.db_at``) times the C21 / R16 high-pass
  (``calc.audio.highpass1.db_at``), summed in dB (``calc.rf.db_sum``).

The design's order had the PM_DRIVE buffer after the integrator (U304 A / B);
here the buffer (U7) drives the integrator input and the integrator drives
PM_DRIVE, so the trim sits *before* the level ``rf.deviation`` reads and the
chain's six factors stay the ones it multiplies (a trim after the integrator
would scale the deviation by a factor no check records). Every single-supply
node sits on ``VREF_TX``; ``CLIP`` and ``INT_IN`` are the DC-free nodes the
two MAX checks read.

:func:`audio_plan` declares the test plan both audio blocks use (tones,
point ac analyses ``ac lin 1 f f`` so every dB row reads an exact sample,
``audio_bandwidth``), with identical rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import AnalysisSpec, Reduce, Stimulus, StimulusKind, Traced
from ai_eda.ir.rf import LabItem
from ai_eda.tools.calc.basic import clock_divided, rc_r_for_cutoff, rc_time_constant
from ai_eda.tools.calc.radio import (
    butterworth_q,
    db_sum,
    emphasis_corner,
    emphasis_db_at,
    fm_pm_drive_limit,
    fm_pm_integrator_tau,
    highpass1_db_at,
    limiter_level,
    lossy_integrator_db_at,
    sallen_key_c1,
    sallen_key_c2,
)
from ai_eda.tools.calc.radio import harmonic as harmonic_hz
from ai_eda.tools.calc.rf import butterworth_attenuation
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import models
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.power import (
    capacitor,
    card_bind,
    choice_once,
    computed_once,
    exclude,
    expectation,
    input_copy,
    model_once,
    net,
    rail_level,
    rail_port,
    resistor,
    serving,
)
from ai_eda.design.rf.common import N_MULT, N_MULT_KEY, N_MULT_TEXT
from ai_eda.design.rf.profile import profile_choices

#: the requirement keys the audio blocks read
DEVIATION_KEY, BANDWIDTH_KEY = "frequency_deviation", "audio_bandwidth"
#: the parameters they become (a copy of the requirement, or the default choice)
DEVIATION_PARAM, BANDWIDTH_PARAM = "tx.frequency_deviation", "audio.bandwidth"
#: the default audio bandwidth when the requirement is not stated
DEFAULT_BANDWIDTH_HZ = 3000.0
#: the analyses of the TX audio transient and of the point ac analyses
TRAN_AF = "tran_af"
AC_LOW, AC_REF, AC_BW, AC_2BW, AC_RASTER = "ac_f_low", "ac_f_ref", "ac_f_bw", "ac_f_2bw", "ac_raster"
#: the model-constant K_pm of a board without a PM fixture
MODEL_K_PM = "model.k_pm"


def profile_value(b: BlockBuilder, key: str) -> Traced:
    """A ``kr447.*`` profile placeholder as this block's parameter (the same row the template shows; never grounded)."""
    have = b.result.params.get(key)
    if have is not None:
        return have
    for ch, traced in profile_choices(b.ctx.template_id, b.ctx.confirmed):
        if ch.key == key:
            b.result.choices.append(ch)
            b.result.params[key] = traced
            return traced
    raise TemplateRefusal(f"no profile value {key!r}")


def deviation(b: BlockBuilder) -> Traced:
    """``tx.frequency_deviation``: the confirmed requirement, or - when it is not stated - the profile's ``kr447.max_deviation`` record itself.

    The unstated case copies the placeholder's own traced value (its
    UNVERIFIED choice row is the one the confirmation table shows), so every
    number designed from it - ``tx.tau_i``, ``tx.v_splat_max`` and the deck
    rows judged against them - names the profile in its provenance, and the
    reports list those rows as resting on a placeholder
    (:func:`ai_eda.report.rf_report.profile_dependents`) instead of reading
    a note.
    """
    if DEVIATION_KEY in b.ctx.inputs:
        return input_copy(b, DEVIATION_KEY, DEVIATION_PARAM)
    have = b.result.params.get(DEVIATION_PARAM)
    if have is not None:
        return have
    limit = profile_value(b, "kr447.max_deviation")
    b.result.params[DEVIATION_PARAM] = limit
    return limit


@dataclass(frozen=True)
class AudioPlan:
    """The audio test plan: the tones, the bandwidth, the ac level and the tolerances both audio blocks use."""

    f_low: Traced
    f_ref: Traced
    bandwidth: Traced
    ac_level: Traced
    tol_db: Traced
    r_vref: Traced
    c_vref: Traced
    c_decouple: Traced


def audio_plan(b: BlockBuilder) -> AudioPlan:
    """Declare (once per block) the audio tones, ``audio.bandwidth`` and the point ac analyses at them; identical in both audio blocks."""
    c = lambda key, value, unit, text: choice_once(b, key, value, unit, text)  # noqa: E731
    f_low = c("audio.f_low", 300.0, "Hz", "the low edge of the voice band: the audio rows are checked at 300 Hz")
    f_ref = c("audio.f_ref", 1000.0, "Hz", "the reference tone: 1 kHz (the ac rows and the transient's test tone)")
    if BANDWIDTH_KEY in b.ctx.inputs:
        bw = input_copy(b, BANDWIDTH_KEY, BANDWIDTH_PARAM)
    else:
        bw = c(BANDWIDTH_PARAM, DEFAULT_BANDWIDTH_HZ, "Hz", "audio_bandwidth is not stated: the voice band ends at 3 kHz (the splatter filter's and the RX filter's corner)")
    ac_level = c("audio.ac_level", 1.0, "V", "the ac magnitude of the audio stimuli (the rows are ratios, so it cancels)")
    tol_db = c("audio.tol_db", 0.5, "dB", "tolerance of the audio ac rows (0.5 dB: ideal parts and the model.opamp macro; its finite GBW adds up to about 0.4 dB at 3 kHz)")
    r_vref = c("audio.r_vref", 100e3, "ohm", "the mid-supply reference dividers VREF_TX / VREF_RX (two equal resistors)")
    c_vref = c("audio.c_vref", 10e-6, "F", "the reference's decoupling capacitor (VREF is the signal ground of the single-supply stages)")
    c_dec = c("audio.c_decouple", 100e-9, "F", "supply decoupling of each op-amp")
    variation = c("audio.ac_variation", "lin", None, "every audio ac analysis is a single point 'ac lin 1 f f', so every dB row reads an exact sample")
    points = c("audio.ac_points", 1, None, "one point per ac analysis (at the row's own frequency)")
    for aid, f, what in ((AC_LOW, f_low, "the low voice-band edge"), (AC_REF, f_ref, "the reference tone"), (AC_BW, bw, "the audio bandwidth")):
        ac_point(b, aid, f, variation, points, what)
    return AudioPlan(f_low, f_ref, bw, ac_level, tol_db, r_vref, c_vref, c_dec)


def ac_point(b: BlockBuilder, aid: str, f: Traced, variation: Traced, points: Traced, what: str) -> None:
    if not any(a.id == aid for a in b.result.analyses):
        b.result.analyses.append(AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": variation, "points": points, "fstart": f, "fstop": f},
                                              provenance=b.ctx.provenance(f"single-point ac analysis at {what}")))


def db_row(b: BlockBuilder, eid: str, analysis: str, vector: str, reference: str, at: Traced, nominal: Traced, note: str,
           tol_db: Traced | None = None, bound: str | None = None) -> None:
    kw = {"tol_abs": tol_db} if bound is None else {"bound": bound}
    expectation(b, eid, analysis, vector, Reduce.DB_AT, nominal, note, at=at, reference_vector=reference, **kw)


class TxAudioBlock(Block):
    """Microphone amplifier, pre-emphasis, limiter, 4th-order splatter filter, deviation trim and integrator -> ``PM_DRIVE`` (module docstring).

    ``k_pm_key``: the parameter holding the design K_pm (rad/V). The default
    ``model.k_pm`` is declared here as an UNVERIFIED model value; any other
    key must be in ``ctx.shared`` (written by the block that computes it).
    """

    id = "tx_audio"
    title = "TX audio: microphone amplifier, pre-emphasis, limiter, splatter filter, integrator"
    interface_nets = ("TX_3V3", "MICOUT", "PM_DRIVE")

    def __init__(self, *, k_pm_key: str = MODEL_K_PM) -> None:
        self.k_pm_key = k_pm_key

    def _k_pm(self, b: BlockBuilder) -> Traced:
        if self.k_pm_key == MODEL_K_PM:
            return model_once(b, MODEL_K_PM)
        shared = b.ctx.shared.get(self.k_pm_key)
        if shared is None:
            raise TemplateRefusal(f"block {self.id}: K_pm parameter {self.k_pm_key!r} is not among the board's shared parameters")
        b.result.params[self.k_pm_key] = shared
        return shared

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        c = lambda key, value, unit, text: choice_once(b, key, value, unit, text)  # noqa: E731
        k = lambda key, make: computed_once(b, key, make)  # noqa: E731
        ap = audio_plan(b)
        tx3 = rail_level(b, "TX_3V3")
        dev = deviation(b)
        raster = profile_value(b, "kr447.channel_raster")
        n_mult = b.ctx.shared.get(N_MULT_KEY)  # a composing board's modulator / LO chain wrote it first: the family's one row (ai_eda.design.rf.common)
        if n_mult is None:
            n_mult = c(N_MULT_KEY, N_MULT, None, N_MULT_TEXT)
        k_pm = self._k_pm(b)
        a0, gbw = model_once(b, "model.opamp.a0"), model_once(b, "model.opamp.gbw")
        opamp, diode = models.opamp_card(a0, gbw), models.diode_card()
        pos = model_once(b, "model.pot.position")
        # --- choices
        mic_r = c("tx.mic_bias_r", 2.2e3, "ohm", "R301: electret bias resistor from MICBIAS [UNVERIFIED: Maxim MAX9814 and CUI CMC-4013 datasheets]")
        mic_c = c("tx.mic_couple", 100e-9, "F", "C301: microphone coupling capacitor into MICIN")
        ct = c("tx.agc_ct", 470e-9, "F", "C302: the MAX9814's AGC timing capacitor on CT [UNVERIFIED: Maxim MAX9814 datasheet]")
        cg = c("tx.agc_cg", 2.2e-6, "F", "C303: the MAX9814's CG capacitor [UNVERIFIED: Maxim MAX9814 datasheet]")
        cb = c("tx.bias_c", 470e-9, "F", "C304: the MAX9814's BIAS bypass capacitor [UNVERIFIED: Maxim MAX9814 datasheet]")
        th_t = c("tx.th_top", 100e3, "ohm", "R302: AGC threshold divider from MICBIAS to TH [UNVERIFIED: Maxim MAX9814 datasheet]")
        th_b = c("tx.th_bottom", 47e3, "ohm", "R303: AGC threshold divider from TH to GND [UNVERIFIED: Maxim MAX9814 datasheet]")
        c("tx.max9814_straps", "GAIN to TX_3V3 (40 dB), A/R to GND (1:500), SHDN to TX_3V3 (on)", None,
          "the MAX9814's pin straps [UNVERIFIED: Maxim MAX9814 datasheet]")
        test_level = c("tx.test_level", 1.0, "V", (
            "the design deck's MICOUT: a 1 kHz sine of 1 V peak, about +20 dB over a normal MAX9814 output of about 0.1 V (the overdrive test "
            "[UNVERIFIED: the test method; Maxim MAX9814 datasheet]); the chain must limit it"))
        f_hp = c("tx.hpf_corner", 300.0, "Hz", "the TX high-pass corner (C307 / R306 into U302's input): the voice band starts at 300 Hz")
        c_hp = c("tx.c_hp", 10e-9, "F", "C307: the TX high-pass capacitor (R306 is solved from it)")
        tau_e = c("tx.preemph_tau", 750e-6, "s", "the pre-emphasis time constant 750 us (212 Hz; the LMR convention [UNVERIFIED for KR])")
        sign_e = c("tx.preemph_sign", 1.0, None, "+1: the TX stage pre-emphasises (a rising response, the sign role of calc.audio.emphasis.db_at)")
        c_g = c("tx.c_preemph", 10e-9, "F", "C308: the pre-emphasis capacitor (R307 is solved from it)")
        c_cp = c("tx.c_couple", 1e-6, "F", "C310 / C311 / C320: the limiter's and the integrator's coupling capacitors")
        r_clip = c("tx.r_clip", 10e3, "ohm", "R308: the clipper's series resistor")
        r_bl = c("tx.r_bleed", 100e3, "ohm", "R309 / R310 / R315: the bleed and bias resistors of the coupled nodes (CLIP, LIM_IN, INT_IN)")
        vd = c("tx.clip_vd", 0.6, "V", "the clipper's diode drop: D301 / D302 clip CLIP at about +/- 0.6 V (the 1N4148WS forward drop [UNVERIFIED: 1N4148WS datasheet])")
        lim_tol = c("tx.limiter_tol", 0.1, None, "limiter_level: the clip level within 10 % of tx.clip_vd (the model diode's drop rises with the overdrive current)")
        order = c("splat.order", 4.0, None, "decision 4B: a 4th-order Butterworth splatter filter after the limiter (two unity-gain Sallen-Key sections)")
        p1 = c("splat.section1_pair", 2.0, None, "U303 realises the Butterworth pole pair k = 2 (Q 0.5412, the lower Q first)")
        p2 = c("splat.section2_pair", 1.0, None, "U306 realises the pole pair k = 1 (Q 1.3066)")
        r_sk = c("splat.r", 10e3, "ohm", "the Sallen-Key resistors (R1 = R2 = 10 kohm in each section; C1 / C2 are solved from them)")
        mult = c("audio.bw_multiple", 2.0, None, "the splatter filter's stopband row is checked at twice the audio bandwidth")
        min_2bw = c("tx.splat_min_2bw", 21.0, "dB", "splatter_2bw: at least 21 dB of attenuation at twice the corner (the ideal 4th-order Butterworth gives 24.1 dB)")
        min_raster = c("tx.splat_min_raster", 46.0, "dB", "splatter_raster: at least 46 dB at the channel raster (the ideal filter gives 49.6 dB at 12.5 kHz)")
        h3 = c("tx.splat_h3_max", -10.0, "dBc", "splatter_h3: the 3rd harmonic of the clipped 1 kHz tone at SPLAT_OUT at most -10 dBc (it sits at the corner: in-band distortion)")
        h5 = c("tx.splat_h5_max", -30.0, "dBc", "splatter_h5: the 5th harmonic at most -30 dBc (decision 4B)")
        h7 = c("tx.splat_h7_max", -40.0, "dBc", "splatter_h7: the 7th harmonic at most -40 dBc (decision 4B)")
        r_trim = c("tx.trim_r", 10e3, "ohm", "RV301: the deviation trim's track resistance (a divider from VREF_TX to SPLAT_OUT, set in the lab)")
        headroom = c("tx.deviation_headroom", 1.4, None, (
            "the integrator is designed for the deviation requirement / 1.4: the 4th-order splatter filter overshoots on the hard-clipped tone "
            "(ngspice-42: the peak at INT_IN is 1.25 x V_lim at +20 dB overdrive), so the design's 0.9 margin (1.11) would exceed the requirement"))
        c_int_cp = c("tx.c_int_couple", 10e-6, "F", "C321: the integrator's input coupling capacitor (with R316 a high-pass far below the voice band)")
        c_int = c("tx.c_int", 10e-9, "F", "C322: the integrator capacitor (R316 is solved from tau_i)")
        r_dc = c("tx.r_dc", 330e3, "ohm", "R317: the integrator's DC-limit resistor across C322 (a lossy integrator: DC gain R317 / R316)")
        af_step = c("tx.af_step", 5e-6, "s", "tran_af step 5 us (the 7th harmonic of 1 kHz needs at most 1 / (20 x 7 kHz) = 7.1 us; 5 us keeps the run-time budget)")
        af_stop = c("tx.af_stop", 0.04, "s", "tran_af runs to 40 ms")
        af_save = c("tx.af_save_start", 0.029, "s", "tran_af saves from 29 ms (the chain has settled; the window starts at 30 ms)")
        w0 = c("tx.af_window_start", 0.03, "s", "the harmonic window starts at 30 ms")
        w1 = c("tx.af_window_stop", 0.04, "s", "the harmonic window ends at 40 ms (10 whole periods of 1 kHz)")
        # --- calculators
        r_hp = k("tx.r_hp", lambda: rc_r_for_cutoff(f_hp, c_hp, ("tx.hpf_corner", "tx.c_hp")))
        f_e = k("tx.f_preemph", lambda: emphasis_corner(tau_e, ("tx.preemph_tau",)))
        r_e = k("tx.r_preemph", lambda: rc_r_for_cutoff(f_e, c_g, ("tx.f_preemph", "tx.c_preemph")))
        q1 = k("splat.q1", lambda: butterworth_q(order, p1, ("splat.order", "splat.section1_pair")))
        q2 = k("splat.q2", lambda: butterworth_q(order, p2, ("splat.order", "splat.section2_pair")))
        sk1_c1 = k("splat.c1_1", lambda: sallen_key_c1(q1, ap.bandwidth, r_sk, ("splat.q1", BANDWIDTH_PARAM, "splat.r")))
        sk1_c2 = k("splat.c2_1", lambda: sallen_key_c2(q1, ap.bandwidth, r_sk, ("splat.q1", BANDWIDTH_PARAM, "splat.r")))
        sk2_c1 = k("splat.c1_2", lambda: sallen_key_c1(q2, ap.bandwidth, r_sk, ("splat.q2", BANDWIDTH_PARAM, "splat.r")))
        sk2_c2 = k("splat.c2_2", lambda: sallen_key_c2(q2, ap.bandwidth, r_sk, ("splat.q2", BANDWIDTH_PARAM, "splat.r")))
        v_lim = k("tx.v_lim", lambda: limiter_level(vd, pos, ("tx.clip_vd", "model.pot.position")))
        df = k("tx.delta_f_design", lambda: clock_divided(dev, headroom, (DEVIATION_PARAM, "tx.deviation_headroom")))
        tau_i = k("tx.tau_i", lambda: fm_pm_integrator_tau(n_mult, k_pm, v_lim, df, ("rf.n_mult", self.k_pm_key, "tx.v_lim", "tx.delta_f_design")))
        f_i = k("tx.f_int", lambda: emphasis_corner(tau_i, ("tx.tau_i",)))
        r_in = k("tx.r_int", lambda: rc_r_for_cutoff(f_i, c_int, ("tx.f_int", "tx.c_int")))
        v_max = k("tx.v_splat_max", lambda: fm_pm_drive_limit(n_mult, k_pm, tau_i, dev, ("rf.n_mult", self.k_pm_key, "tx.tau_i", DEVIATION_PARAM)))
        tau_y = k("tx.tau_int_couple", lambda: rc_time_constant(r_in, c_int_cp, ("tx.r_int", "tx.c_int_couple")))
        f_y = k("tx.f_int_couple", lambda: emphasis_corner(tau_y, ("tx.tau_int_couple",)))
        f_2bw = k("audio.f_2bw", lambda: harmonic_hz(ap.bandwidth, mult, (BANDWIDTH_PARAM, "audio.bw_multiple")))
        pre: dict[str, Traced] = {}
        for tag, f, fid in (("low", ap.f_low, "audio.f_low"), ("ref", ap.f_ref, "audio.f_ref"), ("bw", ap.bandwidth, BANDWIDTH_PARAM)):
            hp = k(f"tx.pre_hp_{tag}", lambda f=f, fid=fid: highpass1_db_at(f, f_hp, (fid, "tx.hpf_corner")))
            em = k(f"tx.pre_emph_{tag}", lambda f=f, fid=fid: emphasis_db_at(f, f_e, sign_e, (fid, "tx.f_preemph", "tx.preemph_sign")))
            try:
                pre[tag] = db_sum(hp, em, (f"tx.pre_hp_{tag}", f"tx.pre_emph_{tag}"))
            except ValueError as e:
                raise TemplateRefusal(f"block {self.id}: pre-emphasis nominal at {tag}: {e}") from e
        att_bw = k("splat.att_bw", lambda: butterworth_attenuation(order, ap.bandwidth, ap.bandwidth, ("splat.order", BANDWIDTH_PARAM, BANDWIDTH_PARAM)))
        lossy = k("tx.int_lossy_ref", lambda: lossy_integrator_db_at(ap.f_ref, tau_i, r_in, r_dc, ("audio.f_ref", "tx.tau_i", "tx.r_int", "tx.r_dc")))
        hp_i = k("tx.int_hp_ref", lambda: highpass1_db_at(ap.f_ref, f_y, ("audio.f_ref", "tx.f_int_couple")))
        int_nominal = db_sum(lossy, hp_i, ("tx.int_lossy_ref", "tx.int_hp_ref"))

        # --- parts
        mk1 = b.part("mic", "MK1", "Mic", "electret microphone")
        u1 = b.part("mic_amp", "U1", "MAX9814", "microphone amplifier with AGC")
        r1 = resistor(b, "R1", mic_r, "electret bias resistor")
        c1 = capacitor(b, "C1", mic_c, "microphone coupling capacitor")
        c2 = capacitor(b, "C2", ct, "AGC timing capacitor (CT)")
        c3 = capacitor(b, "C3", cg, "CG capacitor")
        c4 = capacitor(b, "C4", cb, "BIAS bypass capacitor")
        r2 = resistor(b, "R2", th_t, "AGC threshold divider (top)")
        r3 = resistor(b, "R3", th_b, "AGC threshold divider (bottom)")
        c5 = capacitor(b, "C5", ap.c_decouple, "U301 supply decoupling")
        r4 = resistor(b, "R4", ap.r_vref, "VREF_TX divider (top)")
        r5 = resistor(b, "R5", ap.r_vref, "VREF_TX divider (bottom)")
        c6 = capacitor(b, "C6", ap.c_vref, "VREF_TX decoupling")
        u2 = b.part("opamp", "U2", "MCP6001-OT", "high-pass + pre-emphasis stage")
        c7 = capacitor(b, "C7", c_hp, "TX high-pass capacitor")
        r6 = resistor(b, "R6", r_hp, "TX high-pass resistor to VREF_TX")
        r7 = resistor(b, "R7", r_e, "pre-emphasis feedback resistor")
        c8 = capacitor(b, "C8", c_g, "pre-emphasis capacitor to VREF_TX")
        c9 = capacitor(b, "C9", ap.c_decouple, "U302 supply decoupling")
        c10 = capacitor(b, "C10", c_cp, "clipper coupling capacitor")
        r8 = resistor(b, "R8", r_clip, "clipper series resistor")
        d1 = b.part("diode_sw", "D1", "1N4148WS", "clipper diode (positive half)")
        d2 = b.part("diode_sw", "D2", "1N4148WS", "clipper diode (negative half)")
        r9 = resistor(b, "R9", r_bl, "clipper node bleed resistor to GND")
        u5 = b.part("opamp", "U5", "MCP6001-OT", "limiter output buffer (unity follower)")
        c11_ = capacitor(b, "C11", c_cp, "limiter buffer coupling capacitor")
        r10 = resistor(b, "R10", r_bl, "limiter buffer bias resistor to VREF_TX")
        c12 = capacitor(b, "C12", ap.c_decouple, "U305 supply decoupling")
        u3 = b.part("opamp", "U3", "MCP6001-OT", "splatter filter section 1 (Sallen-Key, Q 0.5412)")
        r11 = resistor(b, "R11", r_sk, "Sallen-Key section 1, R1")
        r12 = resistor(b, "R12", r_sk, "Sallen-Key section 1, R2")
        c13 = capacitor(b, "C13", sk1_c1, "Sallen-Key section 1, C1 (feedback)", serves=serving(ctx, "audio_bandwidth"))
        c14 = capacitor(b, "C14", sk1_c2, "Sallen-Key section 1, C2 (to GND)", serves=serving(ctx, "audio_bandwidth"))
        c15 = capacitor(b, "C15", ap.c_decouple, "U303 supply decoupling")
        u6 = b.part("opamp", "U6", "MCP6001-OT", "splatter filter section 2 (Sallen-Key, Q 1.3066)")
        r13 = resistor(b, "R13", r_sk, "Sallen-Key section 2, R1")
        r14 = resistor(b, "R14", r_sk, "Sallen-Key section 2, R2")
        c16 = capacitor(b, "C16", sk2_c1, "Sallen-Key section 2, C1 (feedback)", serves=serving(ctx, "audio_bandwidth"))
        c17 = capacitor(b, "C17", sk2_c2, "Sallen-Key section 2, C2 (to GND)", serves=serving(ctx, "audio_bandwidth"))
        c18 = capacitor(b, "C18", ap.c_decouple, "U306 supply decoupling")
        rv1 = b.part("trimpot", "RV1", format_value(r_trim), "deviation trim (lab alignment)", serving(ctx, "frequency_deviation"))
        u7 = b.part("opamp", "U7", "MCP6001-OT", "deviation-trim buffer (unity follower)")
        c19 = capacitor(b, "C19", ap.c_decouple, "U307 supply decoupling")
        c20 = capacitor(b, "C20", c_cp, "integrator-input coupling capacitor")
        r15 = resistor(b, "R15", r_bl, "INT_IN bleed resistor to GND")
        c21_ = capacitor(b, "C21", c_int_cp, "integrator input coupling capacitor")
        r16 = resistor(b, "R16", r_in, "integrator input resistor (tau_i = R316 C322)", serves=serving(ctx, "frequency_deviation"))
        u4 = b.part("opamp", "U4", "MCP6001-OT", "lossy integrator driving PM_DRIVE (PM of the integrated audio is FM)", serving(ctx, "modulation"))
        c22_ = capacitor(b, "C22", c_int, "integrator capacitor")
        r17 = resistor(b, "R17", r_dc, "integrator DC-limit resistor")
        c23 = capacitor(b, "C23", ap.c_decouple, "U304 supply decoupling")
        exclude(b, "MK1", "microphone without a model: MICOUT is the stimulus VMIC")
        exclude(b, "U1", "MAX9814 has no model: MICOUT is the stimulus VMIC")
        supply = {"V+": models.OPAMP_SUPPLY_IGNORED, "V-": models.OPAMP_SUPPLY_IGNORED}
        for ref in ("U2", "U3", "U4", "U5", "U6", "U7"):
            card_bind(b, ref, opamp, ignored=supply)
        for ref in ("D1", "D2"):
            card_bind(b, ref, diode)
        card_bind(b, "RV1", models.potentiometer_card(r_trim, pos, key="model.pot.dev", name="POTDEV"))
        members: dict[str, list[tuple[str, str]]] = {
            "MIC_P": [*mk1.at("+"), *r1.at("2"), *c1.at("1")],
            "MIC_IN": [*c1.at("2"), *u1.at("MICIN")],
            "MIC_BIAS": [*u1.at("MICBIAS"), *r1.at("1"), *r2.at("1")],
            "MIC_TH": [*u1.at("TH"), *r2.at("2"), *r3.at("1")],
            "MIC_CT": [*u1.at("CT"), *c2.at("1")],
            "MIC_CG": [*u1.at("CG"), *c3.at("1")],
            "MIC_VB": [*u1.at("BIAS"), *c4.at("1")],
            "MICOUT": [*u1.at("MICOUT"), *c7.at("1")],
            "VREF_TX": [*r4.at("2"), *r5.at("1"), *c6.at("1"), *r6.at("2"), *c8.at("2"), *r10.at("2"), *rv1.at("END1"), *u4.at("+")],
            "PRE_IN": [*c7.at("2"), *r6.at("1"), *u2.at("+")],
            "PRE_FB": [*u2.at("-"), *r7.at("2"), *c8.at("1")],
            "PRE_OUT": [*u2.at("OUT"), *r7.at("1"), *c10.at("1")],
            "CLIP_A": [*c10.at("2"), *r8.at("1")],
            "CLIP": [*r8.at("2"), *d1.at("A"), *d2.at("K"), *r9.at("1"), *c11_.at("1")],
            "LIM_IN": [*c11_.at("2"), *r10.at("1"), *u5.at("+")],
            "LIM_OUT": [*u5.at("OUT"), *u5.at("-"), *r11.at("1")],
            "SK1_A": [*r11.at("2"), *r12.at("1"), *c13.at("1")],
            "SK1_P": [*r12.at("2"), *c14.at("1"), *u3.at("+")],
            "SPLAT_MID": [*u3.at("OUT"), *u3.at("-"), *c13.at("2"), *r13.at("1")],
            "SK2_A": [*r13.at("2"), *r14.at("1"), *c16.at("1")],
            "SK2_P": [*r14.at("2"), *c17.at("1"), *u6.at("+")],
            "SPLAT_OUT": [*u6.at("OUT"), *u6.at("-"), *c16.at("2"), *rv1.at("END3")],
            "DEV_TRIM": [*rv1.at("WIPER"), *u7.at("+")],
            "DEV_BUF": [*u7.at("OUT"), *u7.at("-"), *c20.at("1")],
            "INT_IN": [*c20.at("2"), *r15.at("1"), *c21_.at("1")],
            "INT_A": [*c21_.at("2"), *r16.at("1")],
            "INT_SUM": [*r16.at("2"), *u4.at("-"), *c22_.at("1"), *r17.at("1")],
            "PM_DRIVE": [*u4.at("OUT"), *c22_.at("2"), *r17.at("2")],
            "TX_3V3": [*u1.at("VDD"), *u1.at("~{SHDN}"), *u1.at("GAIN"), *r4.at("1"), *c5.at("1"), *c9.at("1"), *c12.at("1"), *c15.at("1"), *c18.at("1"),
                       *c19.at("1"), *c23.at("1"), *[p for ref in ("U2", "U3", "U4", "U5", "U6", "U7") for p in b.result.placed[ref].at("V+")]],
            "GND": [*mk1.at("-"), *u1.at("GND"), *u1.at("A/R"), *c2.at("2"), *c3.at("2"), *c4.at("2"), *r3.at("2"), *c5.at("2"), *r5.at("2"), *c6.at("2"),
                    *c9.at("2"), *d1.at("K"), *d2.at("A"), *r9.at("2"), *c12.at("2"), *c14.at("2"), *c15.at("2"), *c17.at("2"), *c18.at("2"), *c19.at("2"),
                    *r15.at("2"), *c23.at("2"), *[p for ref in ("U2", "U3", "U4", "U5", "U6", "U7") for p in b.result.placed[ref].at("V-")]],
        }
        for name, pins in members.items():
            net(b, name, pins, _NOTES.get(name, f"{name} (TX audio)"))

        # --- simulation
        zero = Traced(value=0.0, unit="V", provenance=test_level.provenance)  # the sine's offset: part of the same confirmed test stimulus
        b.result.stimuli.append(Stimulus(id="VMIC", source="voltage", net="MICOUT", reference_net="GND", kind=StimulusKind.SINE,
                                         params={"vo": zero, "va": test_level, "freq": ap.f_ref, "ac": ap.ac_level},
                                         provenance=ctx.provenance("MICOUT driven by the overdrive test tone (the MAX9814 is not simulated)")))
        b.result.analyses.append(AnalysisSpec(id=TRAN_AF, kind=SpiceAnalysis.TRAN, params={"step": af_step, "stop": af_stop, "start": af_save},
                                              provenance=ctx.provenance("TX audio transient: the clipped 1 kHz tone through the splatter filter and integrator")))
        var, pts = b.result.params["audio.ac_variation"], b.result.params["audio.ac_points"]
        ac_point(b, AC_2BW, f_2bw, var, pts, "twice the audio bandwidth")
        ac_point(b, AC_RASTER, raster, var, pts, "the channel raster")
        expectation(b, "limiter_level", TRAN_AF, "v(CLIP)", Reduce.MAX, vd, "the clipper clips at the diode drop tx.clip_vd (the model diode)", tol_rel=lim_tol)
        expectation(b, "pm_drive_peak", TRAN_AF, "v(INT_IN)", Reduce.MAX, v_max,
                    "the limited, filtered and trimmed level at the integrator input stays below the level that gives the deviation requirement", bound="at_most")
        window = {"f0": ap.f_ref, "t_start": w0, "t_stop": w1}
        for eid, n, lim in (("splatter_h3", 3, h3), ("splatter_h5", 5, h5), ("splatter_h7", 7, h7)):
            kk = Traced(value=n, provenance=lim.provenance)  # the harmonic number is part of the same confirmed row
            expectation(b, eid, TRAN_AF, "v(SPLAT_OUT)", Reduce.HARMONIC_DBC, lim, f"harmonic {n} of the clipped 1 kHz tone after the splatter filter",
                        params={**window, "k": kk}, bound="at_most")
        for tag, aid, f in (("300", AC_LOW, ap.f_low), ("1k", AC_REF, ap.f_ref), ("bw", AC_BW, ap.bandwidth)):
            key = {"300": "low", "1k": "ref", "bw": "bw"}[tag]
            db_row(b, f"preemph_{tag}", aid, "v(PRE_OUT)", "v(MICOUT)", f, pre[key], "the high-pass + pre-emphasis stage (calc nominal of the whole stage)", tol_db=ap.tol_db)
        db_row(b, "splatter_bw", AC_BW, "v(LIM_OUT)", "v(SPLAT_OUT)", ap.bandwidth, att_bw, "the splatter filter's attenuation at its corner (10 log10 2)", tol_db=ap.tol_db)
        db_row(b, "splatter_2bw", AC_2BW, "v(LIM_OUT)", "v(SPLAT_OUT)", f_2bw, min_2bw, "the splatter filter's attenuation at twice its corner", bound="at_least")
        db_row(b, "splatter_raster", AC_RASTER, "v(LIM_OUT)", "v(SPLAT_OUT)", raster, min_raster, "the splatter filter's attenuation at the channel raster", bound="at_least")
        db_row(b, "integrator_1k", AC_REF, "v(PM_DRIVE)", "v(INT_IN)", ap.f_ref, int_nominal,
               "the lossy integrator's gain at 1 kHz (checks tau_i, which rf.deviation divides by)", tol_db=ap.tol_db)
        b.result.ports = [rail_port("tx_3v3", "TX_3V3", tx3, "in")]
        b.result.chain = ["MK1", "U1", "U2", "D1", "U5", "U3", "U6", "RV1", "U7", "U4"]
        b.result.lab_items = [
            LabItem(id="mic_agc", block=self.id, what="the MAX9814's AGC output level and attack / release with the fitted microphone",
                    instruments=["audio analyser", "oscilloscope"], reason="the MAX9814 has no model and its straps are UNVERIFIED choices"),
            LabItem(id="splatter_spectrum", block=self.id, what="the splatter filter's output spectrum and distortion at +20 dB overdrive on the real op-amps",
                    instruments=["audio analyser"], reason="the op-amps are a generic single-pole macro without rail limits"),
        ]
        return b.done()


def format_value(t: Traced) -> str:
    from ai_eda.tools.calc.part_value import format_part_value

    return format_part_value(float(t.value))


_NOTES: dict[str, str] = {
    "MIC_P": "the electret's + terminal: bias resistor and coupling capacitor", "MIC_IN": "the MAX9814's microphone input",
    "MIC_BIAS": "the MAX9814's microphone bias output", "MIC_TH": "the AGC threshold divider", "MIC_CT": "the AGC timing capacitor",
    "MIC_CG": "the MAX9814's CG capacitor", "MIC_VB": "the MAX9814's BIAS bypass", "MICOUT": "the MAX9814's output (the stimulus VMIC in the deck)",
    "VREF_TX": "the TX audio reference: TX_3V3 / 2", "PRE_IN": "the high-pass node at U302's input", "PRE_FB": "U302's feedback node (pre-emphasis zero)",
    "PRE_OUT": "the pre-emphasised audio", "CLIP_A": "between the clipper's coupling capacitor and series resistor",
    "CLIP": "the clipper node: +/- one diode drop about 0 V", "LIM_IN": "the limiter buffer's input (biased to VREF_TX)", "LIM_OUT": "the limited audio",
    "SK1_A": "Sallen-Key section 1, the R1-R2 node", "SK1_P": "Sallen-Key section 1, the op-amp input", "SPLAT_MID": "between the two filter sections",
    "SK2_A": "Sallen-Key section 2, the R1-R2 node", "SK2_P": "Sallen-Key section 2, the op-amp input", "SPLAT_OUT": "the splatter filter's output",
    "DEV_TRIM": "the deviation trim's wiper", "DEV_BUF": "the trim buffer's output", "INT_IN": "the integrator's DC-free input level (pm_drive_peak)",
    "INT_A": "between the integrator's coupling capacitor and input resistor", "INT_SUM": "the integrator's summing node",
    "PM_DRIVE": "the integrated audio to the phase modulator's bias node", "TX_3V3": "the TX logic / audio rail", "GND": "ground",
}


__all__ = [
    "AC_2BW",
    "AC_BW",
    "AC_LOW",
    "AC_RASTER",
    "AC_REF",
    "BANDWIDTH_KEY",
    "BANDWIDTH_PARAM",
    "DEFAULT_BANDWIDTH_HZ",
    "DEVIATION_KEY",
    "DEVIATION_PARAM",
    "MODEL_K_PM",
    "TRAN_AF",
    "AudioPlan",
    "TxAudioBlock",
    "ac_point",
    "audio_plan",
    "db_row",
    "deviation",
    "profile_value",
]
