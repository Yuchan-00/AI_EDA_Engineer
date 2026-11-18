"""The matched-termination crosstalk closed forms (:mod:`ai_eda.tools.calc.xtalk`) against the numbers their opened sources print.

Sources (as cited in the module): Simonovich, SIJ 2022, Eq. 2 / 4 / 6 / 8 and
his worked example (Kb 0.055, L_sat 363 mil); Macnica 126649 Fig. 6 (the
modal values of a printed Maxwell matrix) and 127093 Fig. 1 (Kb = xi / 2);
Polar AP8164 / LaMeres slide 30 (Kb = 1/4 (Cm/C + Lm/L), a cross-check);
Bogatin EDN Rule of Thumb #21 (FEXT ~ 0.5 % x Len[in] / RT[ns] for 50-ohm FR4
microstrip at s = w, a sanity check). The modal inputs are
:func:`ai_eda.tools.calc.tline.edge_coupled_microstrip` on the 4-layer
generic F.Cu (h 0.2 mm, t 35 um, er 4.5) at the 50-ohm width.
"""

from __future__ import annotations

import math

import pytest

from ai_eda.tools.calc.tline import C0, critical_length_mm, edge_coupled_microstrip, propagation_delay, solve_microstrip_width
from ai_eda.tools.calc.xtalk import FEXT_CAP, fext_rel, kb_eq8, l_sat_mm, maxwell_from_modes, modes_from_maxwell, next_rel

H, T, ER = 0.2, 35.0, 4.5


def _modes(s_over_h: float, er: float = ER):
    w = solve_microstrip_width(50.0, H, T, er)
    return w, edge_coupled_microstrip(w, s_over_h * H, H, T, er)


def _kb(r) -> float:
    return kb_eq8(r.z_even, r.z_odd, math.sqrt(r.z_even * r.z_odd))


def test_kb_eq8_at_s_over_h_1_5_and_its_cross_checks():
    w, r = _modes(1.5)
    assert w == pytest.approx(0.3464, abs=1e-4)
    kb = _kb(r)
    assert kb == pytest.approx(0.04534, rel=1e-3)
    # Eq. 8 with the matched Z_term = sqrt(Ze Zo) (Eq. 2) is (sqrt Ze - sqrt Zo) / (sqrt Ze + sqrt Zo); with an open near end it is Eq. 9
    assert kb == pytest.approx((math.sqrt(r.z_even) - math.sqrt(r.z_odd)) / (math.sqrt(r.z_even) + math.sqrt(r.z_odd)), rel=1e-12)
    assert kb_eq8(r.z_even, r.z_odd, 1e30) == pytest.approx((r.z_even - r.z_odd) / (r.z_even + r.z_odd), rel=1e-12)
    # the weak-coupling LC form 1/4 (Cm/C + Lm/L) (Polar AP8164, LaMeres slide 30) with the Maxwell terms, and Macnica's xi / 2
    m = maxwell_from_modes(r.z_even, r.z_odd, r.e_eff_even, r.e_eff_odd)
    lc = 0.25 * (-m.c12 / m.c11 + m.l12 / m.l11)
    assert lc == pytest.approx(0.04520, rel=1e-3) and abs(lc / kb - 1.0) <= 0.015
    xi_half = (r.z_even - r.z_odd) / (r.z_even + r.z_odd) / 2.0
    assert abs(xi_half / kb - 1.0) <= 0.015


def test_the_20_mm_run_at_1_ns():
    _, r = _modes(1.5)
    kb = _kb(r)
    t_r = 1e-9
    ls = l_sat_mm(t_r, propagation_delay(r.e_eff_even))
    assert ls == pytest.approx(critical_length_mm(t_r, math.sqrt(r.e_eff_even) / C0, 0.5), rel=1e-15)
    # the even (slower) mode, as the spec's formula states: 78.86 mm, NEXT 1.150 %
    assert ls == pytest.approx(78.86, abs=0.01)
    assert next_rel(kb, 20.0, ls) == pytest.approx(0.01150, rel=1e-3)
    assert fext_rel(20.0, r.e_eff_even, r.e_eff_odd, t_r) == pytest.approx(-0.00427, rel=1e-3)
    # the spec's example numbers (L_sat 81.6 mm, NEXT 1.112 %) come from the mean of the two modes' e_eff in the research script,
    # not from the even mode its own formula names: pinned here so the difference stays visible
    mean = critical_length_mm(t_r, math.sqrt((r.e_eff_even + r.e_eff_odd) / 2.0) / C0, 0.5)
    assert mean == pytest.approx(81.6, abs=0.05) and kb * 20.0 / mean == pytest.approx(0.01112, rel=1e-3)
    assert ls < mean  # the even mode gives the shorter L_sat: the estimate errs high


def test_kb_and_fext_do_not_grow_with_the_gap():
    kbs, fexts = [], []
    for g in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0):
        _, r = _modes(g)
        kbs.append(_kb(r))
        fexts.append(abs(fext_rel(20.0, r.e_eff_even, r.e_eff_odd, 1e-9)))
    assert all(a >= b for a, b in zip(kbs, kbs[1:])) and all(a >= b for a, b in zip(fexts, fexts[1:]))
    assert kbs[0] > 0.10 and kbs[-1] < 0.005  # 11.1 % at s/h 0.5 down to 0.28 % at 10 (the research step's samples)


def test_fext_against_bogatin_rule_of_thumb_21():
    """50-ohm FR4 (er 4.3) microstrip, h 0.2 mm, s = w: 0.479 %/(in/ns) computed against the rule's 0.5 % x Len[in] / RT[ns]."""
    w = solve_microstrip_width(50.0, H, T, 4.3)
    r = edge_coupled_microstrip(w, w, H, T, 4.3)
    f = abs(fext_rel(25.4, r.e_eff_even, r.e_eff_odd, 1e-9))
    assert f == pytest.approx(0.00479, abs=5e-6) and abs(f / 0.005 - 1.0) <= 0.10


def test_simonovich_example_and_saturation():
    assert next_rel(0.055, 100.0, 363.0) == pytest.approx(0.01515, rel=1e-3)  # 100 mil of a 363 mil L_sat (lengths in one unit)
    assert next_rel(0.055, 363.0, 363.0) == next_rel(0.055, 1000.0, 363.0) == 0.055  # at or above L_sat: Kb itself


def test_fext_saturates_at_half_the_launched_step_and_is_negative():
    _, r = _modes(1.5)
    assert fext_rel(20.0, r.e_eff_even, r.e_eff_odd, 1e-9) < 0
    assert fext_rel(1e6, r.e_eff_even, r.e_eff_odd, 1e-12) == -FEXT_CAP == -0.5
    assert fext_rel(20.0, 3.0, 3.0, 1e-9) == 0.0  # equal mode velocities (a homogeneous medium): no far-end term
    assert fext_rel(0.0, r.e_eff_even, r.e_eff_odd, 1e-9) == 0.0


def test_macnica_fig6_matrices_and_the_inverse_round_trip():
    z_c, z_d, td_c, td_d = modes_from_maxwell(356.6e-9, 91.5e-9, 126.6e-12, -27.8e-12)
    assert round(z_c, 1) == 67.3 and round(z_d, 1) == 41.4 and round(td_c * 1e9, 1) == 6.7 and round(td_d * 1e9, 1) == 6.4
    m = maxwell_from_modes(z_c, z_d, (td_c * C0) ** 2, (td_d * C0) ** 2)
    assert m.l11 == pytest.approx(356.6e-9, rel=1e-12) and m.l12 == pytest.approx(91.5e-9, rel=1e-12)
    assert m.c11 == pytest.approx(126.6e-12, rel=1e-12) and m.c12 == pytest.approx(-27.8e-12, rel=1e-12)


def test_maxwell_golden_values_at_s_over_h_1_5():
    _, r = _modes(1.5)
    m = maxwell_from_modes(r.z_even, r.z_odd, r.e_eff_even, r.e_eff_odd)
    assert m.l11 == pytest.approx(3.0538e-7, rel=1e-4) and m.l12 == pytest.approx(3.8164e-8, rel=1e-4)
    assert m.c11 == pytest.approx(1.2394e-10, rel=1e-4) and m.c12 == pytest.approx(-6.9173e-12, rel=1e-4)
    # positive definite (the CPL element refuses anything else): L11^2 - L12^2 = (Ze td_e)(Zo td_o) > 0, C11^2 - C12^2 > 0
    assert m.l11 ** 2 - m.l12 ** 2 > 0 and m.c11 ** 2 - m.c12 ** 2 > 0 and m.c12 < 0
    z_c, z_d, td_c, td_d = modes_from_maxwell(*m)
    assert (z_c, z_d) == pytest.approx((r.z_even, r.z_odd), rel=1e-12)
    assert (td_c, td_d) == pytest.approx((math.sqrt(r.e_eff_even) / C0, math.sqrt(r.e_eff_odd) / C0), rel=1e-12)


def test_the_forms_refuse_what_they_cannot_mean():
    for call in (lambda: kb_eq8(0.0, 40.0, 45.0), lambda: kb_eq8(50.0, 40.0, -1.0), lambda: l_sat_mm(0.0, 6e-9), lambda: next_rel(0.05, -1.0, 80.0),
                 lambda: next_rel(0.05, 10.0, 0.0), lambda: fext_rel(10.0, 0.0, 3.0, 1e-9), lambda: fext_rel(10.0, 3.6, 3.1, math.nan),
                 lambda: maxwell_from_modes(50.0, 40.0, -3.0, 3.0), lambda: modes_from_maxwell(1e-7, 2e-7, 1e-10, -1e-11)):
        with pytest.raises(ValueError):
            call()
