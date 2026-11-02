"""The RX audio block of the KR 447 MHz family: de-emphasis, 300-3000 Hz band-pass, volume, LM386 speaker amplifier, squelch comparator.

Invariant: every filter corner is a registered calculator's output over
confirmed choices and the confirmed ``audio_bandwidth``, and the design deck
checks the chain from ``DISC_OUT`` to ``VOL_IN`` against the product of its
first-order terms (``calc.audio.highpass1.db_at`` x
``calc.audio.emphasis.db_at`` (sign -1) x ``calc.audio.lowpass1.db_at``,
summed in dB by ``calc.rf.db_sum``) - exact for this topology, not an
approximation. The LM386, the squelch comparator, the speaker and the
volume / squelch pots' mechanics have no model: the amplifier and the
comparator are excluded, and the squelch threshold is checked as its
divider (``sq_threshold``). Nothing claims the receiver's audio quality.

The chain (kr447 design §2.1; local references, re-based by 400 on a board):

* ``DISC_OUT`` (the FM detector's audio, the SA605's MUTED_AUD_OUTP on the
  receiver; the stimulus ``VDISC`` in the deck) -> U1, an inverting stage
  whose input branch C3 + R3 is the high-pass and whose feedback R4 // C4 is
  the de-emphasis pole (tau ``rx.deemph_tau`` 750 us): H = -(s R C3 / (1 + s R
  C3)) (1 / (1 + s R C4)) with R3 = R4 = R, exactly. R is solved from the
  de-emphasis corner and C4 (``calc.rc.r_for_cutoff``); the high-pass corner
  follows from R and C3 (312 Hz with 6.8 nF).
* R5 / C5 first-order low-pass at ``audio_bandwidth`` into the follower U4 ->
  ``VOL_IN``.
* C6 -> RV1 (volume, RK09K) -> U2 LM386 (gain pins open: 20 [UNVERIFIED:
  TI LM386 datasheet]) on ``V_RX`` -> C2 (220 uF electrolytic, + at the
  amplifier) -> ``SPK+`` -> J1 (speaker); Zobel R6 / C8.
* Squelch: U3 (LMV331 on ``RX_3V3``) compares the threshold ``SQ_SET`` (RV2's
  wiper on the ``SQ_REF`` divider R10 / RV2) with ``RSSI`` (through R7); its
  open-collector output ``MUTE`` (R9 pull-up, R8 hysteresis to the threshold)
  is high - muted - while the RSSI is below the threshold
  (``rx.mute_polarity``: the SA605's MUTE_INPUT high mutes [UNVERIFIED: NXP
  SA605 datasheet]).

``VREF_RX`` = RX_5V / 2 (R1 / R2, C1) is the signal ground of U1 / U4.
"""

from __future__ import annotations

from ai_eda.ir import Reduce, Stimulus, StimulusKind, Traced
from ai_eda.ir.rf import LabItem
from ai_eda.tools.calc.basic import rc_r_for_cutoff, rc_time_constant, voltage_divider_output
from ai_eda.tools.calc.radio import db_sum, emphasis_corner, emphasis_db_at, highpass1_db_at, lowpass1_db_at

from ai_eda.design.rf import models
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.blocks.power import (
    capacitor,
    card_bind,
    choice_once,
    computed_once,
    cp_polarity,
    exclude,
    expectation,
    model_once,
    net,
    op_analysis,
    passive,
    rail_level,
    rail_port,
    resistor,
    serving,
)
from ai_eda.design.rf.blocks.tx_audio import AC_BW, AC_LOW, AC_REF, BANDWIDTH_PARAM, audio_plan, db_row


class RxAudioBlock(Block):
    """De-emphasis + 300-3000 Hz band-pass, volume, LM386 speaker amplifier and squelch comparator (module docstring)."""

    id = "rx_audio"
    title = "RX audio: de-emphasis, band-pass, volume, speaker amplifier, squelch"
    interface_nets = ("V_RX", "RX_5V", "RX_3V3", "DISC_OUT", "RSSI", "MUTE", "SPK+")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        c = lambda key, value, unit, text: choice_once(b, key, value, unit, text)  # noqa: E731
        k = lambda key, make: computed_once(b, key, make)  # noqa: E731
        ap = audio_plan(b)
        rx5, rx3 = rail_level(b, "RX_5V"), rail_level(b, "RX_3V3")
        a0, gbw = model_once(b, "model.opamp.a0"), model_once(b, "model.opamp.gbw")
        opamp = models.opamp_card(a0, gbw)
        pos = model_once(b, "model.pot.position")
        # --- choices
        test_level = c("rx.test_level", 0.1, "V", "the design deck's DISC_OUT: a 1 kHz sine of 0.1 V peak (the detector's audio level [UNVERIFIED: NXP SA605 datasheet])")
        c_hp = c("rx.c_hp", 6.8e-9, "F", "C403: the RX high-pass capacitor (with R403 a 312 Hz corner)")
        tau_d = c("rx.deemph_tau", 750e-6, "s", "the de-emphasis time constant 750 us (212 Hz), the TX pre-emphasis's inverse")
        sign_d = c("rx.deemph_sign", -1.0, None, "-1: the RX stage de-emphasises (a falling response, the sign role of calc.audio.emphasis.db_at)")
        c_de = c("rx.c_deemph", 10e-9, "F", "C404: the de-emphasis capacitor across R404 (R403 = R404 is solved from it)")
        c_lp = c("rx.c_lp", 1e-9, "F", "C405: the RX low-pass capacitor (R405 is solved from the audio bandwidth)")
        c_vol = c("rx.c_vol", 1e-6, "F", "C406: the volume pot's coupling capacitor (no DC through the pot)")
        vol_r = c("rx.vol_r", 10e3, "ohm", "RV401: the volume pot's track resistance")
        c_by = c("rx.c_bypass", 10e-6, "F", "C407: the LM386's BYPASS capacitor [UNVERIFIED: TI LM386 datasheet]")
        r_zo = c("rx.r_zobel", 10.0, "ohm", "R406: the Zobel network's resistor at the LM386 output [UNVERIFIED: TI LM386 datasheet]")
        c_zo = c("rx.c_zobel", 47e-9, "F", "C408: the Zobel network's capacitor")
        c_out = c("rx.c_out", 220e-6, "F", "C402: the speaker coupling electrolytic (220 uF; with an 8 ohm speaker a 90 Hz corner)")
        c("rx.lm386_gain", "GAIN pins open: gain 20", None, "the LM386's gain strap [UNVERIFIED: TI LM386 datasheet]")
        cp_polarity(b)  # C402: pin 1 = + on the LM386 output, pin 2 = - on the speaker
        r_rssi = c("rx.r_rssi", 100e3, "ohm", "R407: RSSI into the squelch comparator")
        r_hy = c("rx.r_sq_hyst", 1e6, "ohm", "R408: squelch hysteresis from MUTE to the threshold")
        r_mu = c("rx.r_mute_pullup", 10e3, "ohm", "R409: MUTE pull-up to RX_3V3 (U403 has an open-collector output)")
        r_sq = c("rx.r_sq_top", 10e3, "ohm", "R410: the squelch range divider from RX_3V3 to SQ_REF (the top of RV402)")
        sq_r = c("rx.sq_r", 10e3, "ohm", "RV402: the squelch pot's track resistance (SQ_REF to GND; its wiper sets the threshold)")
        c("rx.mute_polarity", "MUTE high = muted", None, (
            "the squelch output: high (muted) while RSSI is below the threshold; the SA605's MUTE_INPUT is taken as 'high mutes' [UNVERIFIED: NXP SA605 datasheet]"))
        sq_tol = c("rx.sq_tol", 0.02, None, "sq_threshold: SQ_REF within 2 % of the divider (ideal resistors, the formula reproduces; R408's current shifts it slightly)")
        # --- calculators
        f_de = k("rx.f_deemph", lambda: emphasis_corner(tau_d, ("rx.deemph_tau",)))
        r_st = k("rx.r_stage", lambda: rc_r_for_cutoff(f_de, c_de, ("rx.f_deemph", "rx.c_deemph")))
        tau_h = k("rx.tau_hp", lambda: rc_time_constant(r_st, c_hp, ("rx.r_stage", "rx.c_hp")))
        f_hp = k("rx.f_hp", lambda: emphasis_corner(tau_h, ("rx.tau_hp",)))
        r_lp = k("rx.r_lp", lambda: rc_r_for_cutoff(ap.bandwidth, c_lp, (BANDWIDTH_PARAM, "rx.c_lp")))
        v_sq = k("rx.v_sq_ref", lambda: voltage_divider_output(rx3, r_sq, sq_r, ("power.rx_3v3", "rx.r_sq_top", "rx.sq_r")))
        nominal: dict[str, Traced] = {}
        for tag, f, fid in (("low", ap.f_low, "audio.f_low"), ("ref", ap.f_ref, "audio.f_ref"), ("bw", ap.bandwidth, BANDWIDTH_PARAM)):
            hp = k(f"rx.hp_{tag}", lambda f=f, fid=fid: highpass1_db_at(f, f_hp, (fid, "rx.f_hp")))
            de = k(f"rx.de_{tag}", lambda f=f, fid=fid: emphasis_db_at(f, f_de, sign_d, (fid, "rx.f_deemph", "rx.deemph_sign")))
            hd = k(f"rx.hpde_{tag}", lambda hp=hp, de=de, tag=tag: db_sum(hp, de, (f"rx.hp_{tag}", f"rx.de_{tag}")))
            lp = k(f"rx.lp_{tag}", lambda f=f, fid=fid: lowpass1_db_at(f, ap.bandwidth, (fid, BANDWIDTH_PARAM)))
            nominal[tag] = db_sum(hd, lp, (f"rx.hpde_{tag}", f"rx.lp_{tag}"))

        # --- parts
        r1 = resistor(b, "R1", ap.r_vref, "VREF_RX divider (top)")
        r2 = resistor(b, "R2", ap.r_vref, "VREF_RX divider (bottom)")
        c1 = capacitor(b, "C1", ap.c_vref, "VREF_RX decoupling")
        u1 = b.part("opamp", "U1", "MCP6001-OT", "high-pass + de-emphasis stage (inverting)", serving(ctx, "modulation"))
        c3 = capacitor(b, "C3", c_hp, "RX high-pass capacitor")
        r3 = resistor(b, "R3", r_st, "RX input resistor (R = the de-emphasis resistor)")
        r4 = resistor(b, "R4", r_st, "RX feedback resistor")
        c4 = capacitor(b, "C4", c_de, "de-emphasis capacitor")
        c9 = capacitor(b, "C9", ap.c_decouple, "U401 supply decoupling")
        r5 = resistor(b, "R5", r_lp, "RX low-pass resistor", serves=serving(ctx, "audio_bandwidth"))
        c5 = capacitor(b, "C5", c_lp, "RX low-pass capacitor")
        u4 = b.part("opamp", "U4", "MCP6001-OT", "RX low-pass buffer (unity follower)")
        c10 = capacitor(b, "C10", ap.c_decouple, "U404 supply decoupling")
        c6 = capacitor(b, "C6", c_vol, "volume coupling capacitor")
        rv1 = b.part("pot", "RV1", "10k", "volume")
        u2 = b.part("audio_amp", "U2", "LM386", "speaker amplifier (on V_RX)")
        c7 = capacitor(b, "C7", c_by, "LM386 bypass capacitor")
        r6 = resistor(b, "R6", r_zo, "Zobel resistor")
        c8 = capacitor(b, "C8", c_zo, "Zobel capacitor")
        c2 = passive(b, "cap_polarized", "C2", c_out, "speaker coupling electrolytic (pin 1 = +, choice cp_polarity)")
        c11 = capacitor(b, "C11", ap.c_decouple, "LM386 supply decoupling (V_RX)")
        j1 = b.part("conn_2_jst", "J1", "Speaker", "speaker connector")
        u3 = b.part("comparator", "U3", "LMV331", "squelch comparator (supplied from RX_3V3)")
        r7 = resistor(b, "R7", r_rssi, "RSSI input resistor")
        r8 = resistor(b, "R8", r_hy, "squelch hysteresis")
        r9 = resistor(b, "R9", r_mu, "MUTE pull-up")
        r10 = resistor(b, "R10", r_sq, "squelch range divider (top)")
        rv2 = b.part("pot", "RV2", "10k", "squelch threshold")
        c12 = capacitor(b, "C12", ap.c_decouple, "U403 supply decoupling")
        supply = {"V+": models.OPAMP_SUPPLY_IGNORED, "V-": models.OPAMP_SUPPLY_IGNORED}
        for ref in ("U1", "U4"):
            card_bind(b, ref, opamp, ignored=supply)
        mount = {"MOUNT": models.POT_MOUNT_IGNORED}
        card_bind(b, "RV1", models.potentiometer_card(vol_r, pos, key="model.pot.vol", name="POTVOL"), ignored=mount)
        card_bind(b, "RV2", models.potentiometer_card(sq_r, pos, key="model.pot.sql", name="POTSQL"), ignored=mount)
        exclude(b, "U2", "LM386 has no model: the speaker path is structural only")
        exclude(b, "U3", "LMV331 has no model: its threshold divider is simulated, its output is not")
        exclude(b, "J1", "connector: the speaker is not simulated")
        b.leave_open("U2", "GAIN", "the LM386's gain pins are left open: gain 20 [UNVERIFIED: TI LM386 datasheet]")
        members: dict[str, list[tuple[str, str]]] = {
            "VREF_RX": [*r1.at("2"), *r2.at("1"), *c1.at("1"), *u1.at("+")],
            "DISC_OUT": c3.at("1"),
            "RX_A": [*c3.at("2"), *r3.at("1")],
            "RX_SUM": [*r3.at("2"), *u1.at("-"), *r4.at("2"), *c4.at("2")],
            "RX1": [*u1.at("OUT"), *r4.at("1"), *c4.at("1"), *r5.at("1")],
            "RX_LP": [*r5.at("2"), *c5.at("1"), *u4.at("+")],
            "VOL_IN": [*u4.at("OUT"), *u4.at("-"), *c6.at("1")],
            "VOL_TOP": [*c6.at("2"), *rv1.at("END3")],
            "VOL_W": [*rv1.at("WIPER"), *u2.at("+")],
            "LM_BYP": [*u2.at("BYPASS"), *c7.at("1")],
            "SPK_OUT": [*u2.at("OUT"), *r6.at("1"), *c2.at("+")],
            "ZOBEL": [*r6.at("2"), *c8.at("1")],
            "SPK+": [*c2.at("-"), *j1.at("Pin_1")],
            "V_RX": [*u2.at("V+"), *c11.at("1")],
            "RSSI": r7.at("1"),
            "SQ_IN": [*r7.at("2"), *u3.at("-")],
            "SQ_REF": [*r10.at("2"), *rv2.at("END3")],
            "SQ_SET": [*rv2.at("WIPER"), *u3.at("+"), *r8.at("2")],
            "MUTE": [*u3.at("OUT"), *r9.at("2"), *r8.at("1")],
            "RX_5V": [*r1.at("1"), *u1.at("V+"), *u4.at("V+"), *c9.at("1"), *c10.at("1")],
            "RX_3V3": [*r9.at("1"), *r10.at("1"), *u3.at("V+"), *c12.at("1")],
            "GND": [*r2.at("2"), *c1.at("2"), *u1.at("V-"), *u4.at("V-"), *c9.at("2"), *c10.at("2"), *c5.at("2"), *rv1.at("END1"), *rv1.at("MOUNT"),
                    *u2.at("-"), *u2.at("GND"), *c7.at("2"), *c8.at("2"), *c11.at("2"), *j1.at("Pin_2"), *u3.at("V-"), *rv2.at("END1"), *rv2.at("MOUNT"),
                    *c12.at("2")],
        }
        for name, pins in members.items():
            net(b, name, pins, _NOTES.get(name, f"{name} (RX audio)"))

        # --- simulation
        zero = Traced(value=0.0, unit="V", provenance=test_level.provenance)  # the sine's offset: part of the same confirmed test stimulus
        b.result.stimuli.append(Stimulus(id="VDISC", source="voltage", net="DISC_OUT", reference_net="GND", kind=StimulusKind.SINE,
                                         params={"vo": zero, "va": test_level, "freq": ap.f_ref, "ac": ap.ac_level},
                                         provenance=ctx.provenance("DISC_OUT driven by the detector-level test tone (the SA605 is not on this board)")))
        op_analysis(b)
        note = "the RX chain DISC_OUT -> VOL_IN: high-pass x de-emphasis x low-pass (calc nominal, exact for this topology)"
        db_row(b, "rx_bpf_300", AC_LOW, "v(VOL_IN)", "v(DISC_OUT)", ap.f_low, nominal["low"], note, tol_db=ap.tol_db)
        db_row(b, "deemph_1k", AC_REF, "v(VOL_IN)", "v(DISC_OUT)", ap.f_ref, nominal["ref"], note, tol_db=ap.tol_db)
        db_row(b, "rx_bpf_bw", AC_BW, "v(VOL_IN)", "v(DISC_OUT)", ap.bandwidth, nominal["bw"], note, tol_db=ap.tol_db)
        expectation(b, "sq_threshold", "op", "v(SQ_REF)", Reduce.VALUE, v_sq, "the squelch range divider from RX_3V3 (the top of the threshold pot)", tol_rel=sq_tol)
        b.result.ports = [rail_port("rx_5v", "RX_5V", rx5, "in"), rail_port("rx_3v3", "RX_3V3", rx3, "in")]
        b.result.chain = ["U1", "U4", "RV1", "U2", "C2", "J1", "U3", "RV2"]
        b.result.lab_items = [
            LabItem(id="speaker_audio", block=self.id, what="the LM386's output power and distortion into the speaker, and the recovered audio's level",
                    instruments=["audio analyser", "oscilloscope"], reason="the LM386 has no model"),
            LabItem(id="squelch", block=self.id, what="the squelch threshold and hysteresis against the real RSSI, and the mute polarity",
                    instruments=["signal generator", "SINAD meter"], reason="the comparator has no model and the SA605's RSSI slope and MUTE polarity are UNVERIFIED"),
        ]
        return b.done()


_NOTES: dict[str, str] = {
    "VREF_RX": "the RX audio reference: RX_5V / 2", "DISC_OUT": "the detector's audio (the stimulus VDISC in the deck)",
    "RX_A": "between the RX high-pass capacitor and the input resistor", "RX_SUM": "U401's summing node", "RX1": "the de-emphasised, high-passed audio",
    "RX_LP": "the RX low-pass node at U404's input", "VOL_IN": "the band-limited audio at the volume control", "VOL_TOP": "the volume pot's top (AC-coupled)",
    "VOL_W": "the volume pot's wiper: the LM386's input", "LM_BYP": "the LM386's bypass", "SPK_OUT": "the LM386's output", "ZOBEL": "the Zobel network's node",
    "SPK+": "the speaker's + terminal (after the coupling electrolytic)", "V_RX": "the switched RX rail (LM386 supply)", "RSSI": "the receiver's signal strength",
    "SQ_IN": "the squelch comparator's RSSI input", "SQ_REF": "the top of the squelch pot (the threshold range)",
    "SQ_SET": "the squelch threshold (RV402's wiper) with the hysteresis", "MUTE": "the squelch output (high = muted)",
    "RX_5V": "the RX 5 V rail (op-amp supply, VREF_RX)", "RX_3V3": "the RX 3.3 V rail (squelch)", "GND": "ground",
}


__all__ = ["RxAudioBlock"]
