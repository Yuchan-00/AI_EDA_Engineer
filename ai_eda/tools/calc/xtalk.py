"""Matched-termination crosstalk closed forms of a symmetric edge-coupled pair: NEXT, FEXT, saturation length, Maxwell L / C matrices.

Invariant: these are plain functions, not registered calculators
(``CALC_VERSION`` is unchanged): their numbers live only in the details of
``si.crosstalk`` (:mod:`ai_eda.validation.si`), never in the IR, and they
**rank and explain, never judge**. Every form below assumes both ends of the
victim terminated in its characteristic impedance (and a matched aggressor
source), a lossless, uniform, weakly coupled section over a reference plane,
and a linear-ramp aggressor edge; an ai_eda victim is driven through
``r_drive_ohm`` into ``c_load_f``, which no opened closed form bounds, so a
verdict needs a coupled-line simulation. The modal inputs (``z_even``,
``z_odd``, ``e_eff_even``, ``e_eff_odd``) come from
:func:`ai_eda.tools.calc.tline.edge_coupled_microstrip` (Kirschning & Jansen
1984, static; cited there). Units: impedances in ohm, lengths along a track
in mm, rise times in s, delays per length in s/m, Maxwell matrices per metre
(H/m, F/m). Results are relative to the launched aggressor step.

Sources (opened 2026-11-13 by the crosstalk research step recorded with the
plan, ``crosstalk.json``; quotes are that step's):

* B. Simonovich, "Coupled Transmission Lines and Crosstalk", Signal
  Integrity Journal, 2022-08-09 (opened): Eq. 2 Z0 = sqrt(Z_ev Z_odd);
  Eq. 4 L_sat = (RT/2) c / sqrt(Dk_eff) ("When TD is equal to one-half of
  the linear risetime, the NEXT voltage becomes saturated"); Eq. 6
  V_b = V_a Kb (L / L_sat) below saturation; Eq. 8
  Kb = (Z_ev - Z_odd) / (Z_ev + Z_odd + 2 Z_ev Z_odd / Z_term). His Eq. 11
  (FEXT) prints no coupled-length factor and is **not** used.
* Macnica, "Backward crosstalk factor Kb and forward crosstalk factor Kf",
  2018-02-26, article 127093 (figure images opened): Fig. 1 Kb = xi / 2 with
  xi = (Z_C - Z_D) / (Z_C + Z_D); Fig. 4 with all four ends at
  R = sqrt(Z_C Z_D): when t_r < tau_C - tau_D the far end saturates at -1/4
  of the source EMF, i.e. -1/2 of the launched step (the EMF halves into a
  matched line).
* Macnica, "Line constant of two lines, why is the capacitance Cm between
  lines negative?", 2018-01-31, article 126649 (opened): Fig. 6
  Z_C = sqrt((L + Lm) / (C + Cm)), td_C = sqrt((L + Lm)(C + Cm)), Z_D and
  td_D the same with L - Lm and C - Cm (Maxwell form, Cm < 0); example
  L 356.6 nH/m, Lm 91.5 nH/m, C 126.6 pF/m, Cm -27.8 pF/m -> Z_C 67.3 ohm,
  Z_D 41.4 ohm, td 6.7 and 6.4 ns/m.
* Polar Instruments AP8164, "An introduction to forward and reverse
  crosstalk" (equation images opened): Kb = 1/4 (Cm/C + Lm/L) (a cross-check
  only) and Kf = (1/2)(1/v_odd - 1/v_even), FEXT = (coupling length / rise
  time) Kf - the image glyph reads "1/Z", taken as 1/2 because only 1/2
  agrees with Macnica's two-mode superposition and LaMeres. Polar's other
  form (1/V)(Cm/C - Lm/L) lacks the 1/2 and is **not** used.
* B. LaMeres, EELE 461/561 Module #5 "Crosstalk" (Montana State University
  course notes; opened): slides 21 and 31, FEXT = (length / (v t_rise)) (1/2)
  (C_M/C_L - L_M/L_L) - the 1/2 above; slide 30, Kb = 1/4 (C_M/C_L + L_M/L_L).
  Slide 33 saturates NEXT at v t_r, 2x off Simonovich Eq. 4 / 6 and his own
  slide 10 plateau of 2 TD: **not** used.

Choices (labelled as such, not source statements): the saturation length uses
the slower mode's delay per length (the even mode in microstrip): it is the
shorter L_sat of the two, so the sub-saturated NEXT errs high. The rise time
is used as the 0-100 % linear ramp, as ``spice.si``'s PULSE uses it; a
datasheet 10-90 % t_rise used that way makes the ramp faster, so both NEXT
(shorter L_sat) and FEXT (1/t_r) err high (the research step's reasoning,
not a source statement).
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ai_eda.tools.calc.tline import C0, ROUND_TRIP_FRACTION, critical_length_mm

#: the saturated far-end crosstalk as a fraction of the launched aggressor step (Macnica 127093 Fig. 4: -1/4 of the EMF)
FEXT_CAP = 0.5


def _pos(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return v


def _nonneg(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
    return v


def kb_eq8(z_even: float, z_odd: float, z_term: float) -> float:
    """The saturated near-end coefficient Kb = V_NE / V_A (Simonovich 2022 Eq. 8, quoted in the module docstring).

    ``z_term`` is the victim's near-end termination; matched means
    ``z_term = sqrt(z_even z_odd)`` (Eq. 2), where Eq. 8 reduces to
    (sqrt(Z_ev) - sqrt(Z_odd)) / (sqrt(Z_ev) + sqrt(Z_odd)). Cross-checks
    (``tests/test_xtalk_calc.py``): Macnica 127093 Fig. 1's xi / 2 and the
    weak-coupling 1/4 (Cm/C + Lm/L) of Polar AP8164 / LaMeres slide 30.
    Positive for a rising aggressor edge.
    """
    ze, zo, zt = _pos(z_even, "z_even"), _pos(z_odd, "z_odd"), _pos(z_term, "z_term")
    return (ze - zo) / (ze + zo + 2.0 * ze * zo / zt)


def l_sat_mm(t_r: float, t_pd_even: float) -> float:
    """The NEXT saturation length L_sat = t_r / (2 t_pd) in mm (Simonovich 2022 Eq. 4: the plateau is full when 2 TD reaches the ramp).

    This is :func:`~ai_eda.tools.calc.tline.critical_length_mm` with the
    round-trip fraction 1/2. ``t_pd_even`` is the even (slower) mode's
    delay per length sqrt(e_eff_even) / c0 - a choice (module docstring):
    the shorter L_sat, so the estimate errs high.
    """
    return critical_length_mm(t_r, t_pd_even, ROUND_TRIP_FRACTION)


def next_rel(kb: float, length_mm: float, l_sat_mm: float) -> float:
    """NEXT / V_A = Kb min(1, L / L_sat) (Simonovich 2022 Eq. 6 below saturation, Kb itself at or above it)."""
    if not math.isfinite(float(kb)):
        raise ValueError(f"kb must be finite, got {kb!r}")
    return float(kb) * min(1.0, _nonneg(length_mm, "length_mm") / _pos(l_sat_mm, "l_sat_mm"))


def fext_rel(length_mm: float, e_eff_even: float, e_eff_odd: float, t_r: float) -> float:
    """FEXT / V_A = -min(L |sqrt(e_eff_even) - sqrt(e_eff_odd)| / (2 c0 t_r), 1/2) (module docstring).

    The velocity form is Polar AP8164's Kf = (1/2)(1/v_odd - 1/v_even) times
    L / t_r (the 1/2 from LaMeres slides 21 and 31), with v = c0 / sqrt(e_eff).
    The cap 1/2 is Macnica 127093 Fig. 4: once t_r is shorter than the mode
    delay difference the far end saturates at -1/4 of the EMF, -1/2 of the
    launched step. Negative in microstrip (the even mode is slower: the odd
    mode's negative half arrives first).
    """
    length = _nonneg(length_mm, "length_mm") / 1000.0
    ee, eo = _pos(e_eff_even, "e_eff_even"), _pos(e_eff_odd, "e_eff_odd")
    return -min(length * abs(math.sqrt(ee) - math.sqrt(eo)) / (2.0 * C0 * _pos(t_r, "t_r")), FEXT_CAP)


class MaxwellLC(NamedTuple):
    """Per-metre Maxwell inductance and capacitance matrices of a symmetric pair: L11 = L22, L12; C11 = C22, C12 (C12 < 0 in this form)."""

    l11: float
    l12: float
    c11: float
    c12: float


def maxwell_from_modes(z_even: float, z_odd: float, e_eff_even: float, e_eff_odd: float) -> MaxwellLC:
    """The Maxwell L / C matrices per metre from the even / odd values: the algebraic inverse of Macnica 126649 Fig. 6.

    With td_mode = sqrt(e_eff_mode) / c0 (s/m), Fig. 6's Z_C = sqrt((L + Lm) / (C + Cm))
    and td_C = sqrt((L + Lm)(C + Cm)) give L + Lm = Z_C td_C and
    C + Cm = td_C / Z_C (the same for the D mode with L - Lm, C - Cm), so

        L11 = (Z_e td_e + Z_o td_o) / 2,   L12 = (Z_e td_e - Z_o td_o) / 2
        C11 = (td_e / Z_e + td_o / Z_o) / 2,   C12 = (td_e / Z_e - td_o / Z_o) / 2

    C12 is the Maxwell mutual term (negative: the even mode carries less
    charge per volt). The C11 here is Macnica's C (self term of the Maxwell
    matrix, not the line-to-ground capacitance). :func:`modes_from_maxwell`
    is the forward form.
    """
    ze, zo = _pos(z_even, "z_even"), _pos(z_odd, "z_odd")
    td_e = math.sqrt(_pos(e_eff_even, "e_eff_even")) / C0
    td_o = math.sqrt(_pos(e_eff_odd, "e_eff_odd")) / C0
    return MaxwellLC(l11=(ze * td_e + zo * td_o) / 2.0, l12=(ze * td_e - zo * td_o) / 2.0,
                     c11=(td_e / ze + td_o / zo) / 2.0, c12=(td_e / ze - td_o / zo) / 2.0)


def modes_from_maxwell(l11: float, l12: float, c11: float, c12: float) -> tuple[float, float, float, float]:
    """``(Z_C, Z_D, td_C, td_D)`` (ohm, ohm, s/m, s/m) from the Maxwell terms: Macnica 126649 Fig. 6 as printed (Lm = l12, Cm = c12)."""
    le, lo = _pos(l11 + l12, "L + Lm"), _pos(l11 - l12, "L - Lm")
    ce, co = _pos(c11 + c12, "C + Cm"), _pos(c11 - c12, "C - Cm")
    return math.sqrt(le / ce), math.sqrt(lo / co), math.sqrt(le * ce), math.sqrt(lo * co)


__all__ = ["FEXT_CAP", "MaxwellLC", "fext_rel", "kb_eq8", "l_sat_mm", "maxwell_from_modes", "modes_from_maxwell", "next_rel"]
