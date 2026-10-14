"""Transmission-line calculators (``ai_eda.tools.calc.tline``, CALC_VERSION 0.8): literature cross-checks, inverses, refusals.

What is proven here, without any tool:

* Hammerstad & Jensen's microstrip agrees with Wheeler's independent 1977
  closed form within 2 % (measured: 1 %) over 0.1 <= u <= 10, 2.2 <= er <= 10,
  at t = 0 and with the thickness corrections of both; a wide strip tends to
  the parallel-plate limit, and agrees with the fringing-corrected wide-strip
  form (Pozar, Microwave Engineering, eq. for W/d >= 1) within 0.5 %.
* The zero-thickness values of the single and the coupled microstrip equal
  the numbers KiCad 10's own transmission-line calculator prints for the
  same geometry (an independent transcription of the same published
  equations, compiled from KiCad's source and run at 1 Hz with no cover) to
  1e-6; the coupled-line model agrees with Hammerstad & Jensen's own
  coupled-line equations (1980) within 1 % (even) / 2 % (odd) over the
  Kirschning-Jansen validity range, both modes tend to the single line as
  the gap grows (also with thickness), and Z_odd < Z0 < Z_even.
* Cohn's exact stripline and Wheeler's 1978 closed form agree within 0.5 %
  at t -> 0; with thickness they agree with KiCad's thick-strip closed forms
  within 1 %.
* Every inverse (width for Z0, gap / width for Z_diff) round-trips, is
  deterministic and refuses a target outside its formula's range; every
  forward formula refuses inputs outside its stated range.
* The traced wrappers write role maps and units, and ``calc.recompute``
  re-derives them - including from stackup ids (``pcb.stackup...``).
"""

from __future__ import annotations

import math

import pytest

from ai_eda.ir import BoardOutline, CircuitIR, PCBDesign, ProjectMeta, ProvenanceKind, SolderMask, ValidationStatus as S, assumption, user_requirement
from ai_eda.design.stackup import generic_stackup
from ai_eda.tools.calc import CALC_VERSION, recompute_parameters
from ai_eda.tools.calc.recompute import CALCULATORS
from ai_eda.tools.calc.tline import (
    C0,
    ETA0,
    LUMPED_FRACTION,
    NO_REFERENCE_PLANE,
    NO_STACKUP,
    ROUND_TRIP_FRACTION,
    SOLVE_REL_TOL,
    TLineRangeError,
    critical_length,
    critical_length_mm,
    edge_coupled_microstrip,
    edge_coupled_z_diff,
    line_delay,
    line_geometry,
    microstrip,
    microstrip_e_eff,
    microstrip_z0,
    propagation_delay,
    solve_coupled_spacing,
    solve_coupled_width,
    solve_microstrip_width,
    solve_stripline_width,
    spacing_for_zdiff,
    stripline_impedance,
    tpd,
    width_for_z0,
    width_for_z0_microstrip,
)


# --------------------------------------------------------------------------- independent reference formulas (test-only)


def wheeler_1977(w: float, h: float, t: float, er: float) -> float:
    """H. A. Wheeler, IEEE Trans. MTT-25(8), 1977: Z0 of a strip of width w, thickness t on a sheet of height h (any length unit)."""
    if t > 0:
        dw = t / math.pi * math.log(4.0 * math.e / math.sqrt((t / h) ** 2 + ((1.0 / math.pi) / (w / t + 1.1)) ** 2))
        w = w + dw * (1.0 + 1.0 / er) / 2.0
    k = (14.0 + 8.0 / er) / 11.0
    x = 4.0 * h / w
    return ETA0 / (2.0 * math.pi * math.sqrt(2.0) * math.sqrt(er + 1.0)) * math.log(1.0 + x * (k * x + math.sqrt((k * x) ** 2 + math.pi ** 2 * (1.0 + 1.0 / er) / 2.0)))


def hammerstad_jensen_coupled(u: float, g: float, er: float) -> tuple[float, float]:
    """E. Hammerstad and O. Jensen, 1980, coupled microstrip (Qucs Technical Documentation eqs. 11.156-11.174): ``(Z_even, Z_odd)`` at t = 0.

    The homogeneous (air) mode impedance Z01 / (1 - Z01 Phi_e,o / eta0), divided by the mode's sqrt(e_eff).
    """
    def ab(x: float) -> float:
        a = 1 + math.log((x ** 4 + (x / 52) ** 2) / (x ** 4 + 0.432)) / 49 + math.log(1 + (x / 18.1) ** 3) / 18.7
        return a * 0.564 * ((er - 0.9) / (er + 3)) ** 0.053

    m = 0.2175 + (4.113 + (20.36 / g) ** 6) ** -0.251 + math.log(g ** 10 / (1 + (g / 13.8) ** 10)) / 323
    alpha = 0.5 * math.exp(-g)
    psi = 1 + g / 1.45 + g ** 2.09 / 3.95
    phi_e = 0.8645 * u ** 0.172 / (psi * (alpha * u ** m + (1 - alpha) * u ** -m))
    n = (1 / 17.7 + math.exp(-6.424 - 0.76 * math.log(g) - (g / 0.23) ** 5)) * math.log((10 + 68.3 * g * g) / (1 + 32.5 * g ** 3.093))
    beta = 0.2306 + math.log(g ** 10 / (1 + (g / 3.73) ** 10)) / 301.8 + math.log(1 + 0.646 * g ** 1.175) / 5.3
    theta = 1.729 + 1.175 * math.log(1 + 0.627 / (g + 0.327 * g ** 2.17))
    phi_o = phi_e - theta / psi * math.exp(beta * u ** -n * math.log(u))
    r = 1 + 0.15 * (1 - math.exp(1 - (er - 1) ** 2 / 8.2) / (1 + g ** -6))
    fo1 = 1 - math.exp(-0.179 * g ** 0.15 - 0.328 * g ** r / math.log(math.e + (g / 7) ** 2.8))
    fo = fo1 * math.exp(math.exp(-0.745 * g ** 0.295) / math.cosh(g ** 0.68) * math.log(u) + math.exp(-1.366 - g) * math.sin(math.pi * math.log10(u)))
    mu = g * math.exp(-g) + u * (20 + g * g) / (10 + g * g)
    e_e = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / mu) ** -ab(mu)
    e_o = (er + 1) / 2 + (er - 1) / 2 * fo * (1 + 10 / u) ** -ab(u)
    f = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / u) ** 0.7528))
    z01 = ETA0 / (2 * math.pi) * math.log(f / u + math.sqrt(1 + 4 / (u * u)))
    return z01 / (1 - z01 * phi_e / ETA0) / math.sqrt(e_e), z01 / (1 - z01 * phi_o / ETA0) / math.sqrt(e_o)


def grid(lo: float, hi: float, n: int) -> list[float]:
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# --------------------------------------------------------------------------- microstrip


def test_hammerstad_jensen_agrees_with_wheeler_1977_within_2_percent():
    worst = 0.0
    for er in (2.2, 3.0, 4.5, 6.0, 8.0, 10.0):
        for u in grid(0.1, 10.0, 41):
            hj = microstrip(u, 1.0, 0.0, er).z0_ohm
            worst = max(worst, abs(hj - wheeler_1977(u, 1.0, 0.0, er)) / hj)
    assert worst < 0.02 and worst < 0.011  # measured 0.97 %
    # with the strip's thickness (both formulas' own corrections), PCB-like t/h up to 0.2
    worst = 0.0
    for er in (2.2, 4.5, 10.0):
        for h in (0.2, 0.5, 1.6):
            for t_um in (17.5, 35.0):
                for u in grid(0.5, 10.0, 21):
                    w = u * h
                    hj = microstrip(w, h, t_um, er).z0_ohm
                    worst = max(worst, abs(hj - wheeler_1977(w, h, t_um / 1000.0, er)) / hj)
    assert worst < 0.02


def test_a_wide_strip_tends_to_the_parallel_plate_limit():
    ratios = []
    for u in (10.0, 20.0, 50.0, 100.0):
        r = microstrip(u, 1.0, 0.0, 4.5)
        parallel_plate = ETA0 / (u * math.sqrt(r.e_eff))
        ratios.append(parallel_plate / r.z0_ohm)
        # the fringing-corrected wide-strip form (Hammerstad 1975, as in Pozar's Microwave Engineering for W/d >= 1)
        wide = ETA0 / (math.sqrt(r.e_eff) * (u + 1.393 + 0.667 * math.log(u + 1.444)))
        assert r.z0_ohm == pytest.approx(wide, rel=0.005)
    assert all(a > b > 1.0 for a, b in zip(ratios, ratios[1:]))  # fringing shrinks as the strip widens
    assert ratios[-1] < 1.05
    # the field is ever more inside the dielectric: e_eff -> er
    assert microstrip(1.0, 1.0, 0, 4.5).e_eff < microstrip(10.0, 1.0, 0, 4.5).e_eff < microstrip(100.0, 1.0, 0, 4.5).e_eff < 4.5
    assert microstrip(100.0, 1.0, 0, 4.5).e_eff > 0.97 * 4.5


#: KiCad 10's transmission-line calculator (common/transline_calculations, compiled from its source), 1 Hz, no cover, t = 0:
#: (er, h mm, w mm) -> (Z0, e_eff)
KICAD_MICROSTRIP = {
    (4.5, 1.0, 1.0): (70.33218165, 3.231096652),
    (4.5, 0.2, 0.35): (52.19318064, 3.373441215),
    (10.0, 1.0, 0.1): (106.9122746, 6.040295365),
    (2.2, 1.0, 5.0): (35.47002128, 1.937164616),
}


def test_zero_thickness_microstrip_equals_kicads_transcription():
    for (er, h, w), (z0, e_eff) in KICAD_MICROSTRIP.items():
        r = microstrip(w, h, 0.0, er)
        assert r.z0_ohm == pytest.approx(z0, rel=1e-6) and r.e_eff == pytest.approx(e_eff, rel=1e-6)


def test_thickness_lowers_z0_and_e_eff():
    thin, thick = microstrip(0.35, 0.2, 0.0, 4.5), microstrip(0.35, 0.2, 35.0, 4.5)
    assert thick.z0_ohm < thin.z0_ohm and thick.e_eff < thin.e_eff and thick.u_r > thick.u
    assert thick.z0_ohm == pytest.approx(49.7124, abs=1e-3)  # the generic 4-layer outer layer, 0.35 mm


def test_microstrip_refuses_inputs_outside_the_stated_range():
    with pytest.raises(TLineRangeError, match=r"u = w/h"):
        microstrip(0.001, 0.2, 0.0, 4.5)  # u = 0.005
    with pytest.raises(TLineRangeError, match=r"u = w/h"):
        microstrip(25.0, 0.2, 0.0, 4.5)  # u = 125
    with pytest.raises(TLineRangeError, match="er"):
        microstrip(1.0, 1.0, 0.0, 128.0)
    with pytest.raises(TLineRangeError, match="er"):
        microstrip(1.0, 1.0, 0.0, 0.9)
    with pytest.raises(TLineRangeError, match="not thinner than the substrate"):
        microstrip(1.0, 0.03, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="not thinner than the strip"):
        microstrip(0.03, 0.2, 35.0, 4.5)
    for bad in ((0.0, 0.2, 0.0, 4.5), (-1.0, 0.2, 0.0, 4.5), (1.0, 0.0, 0.0, 4.5), (1.0, 0.2, -1.0, 4.5), (math.nan, 0.2, 0.0, 4.5), (1.0, 0.2, 0.0, math.inf)):
        with pytest.raises(ValueError):
            microstrip(*bad)
    # the bounds are the stated decimal numbers, not one ULP off
    assert microstrip(0.002, 0.2, 0.0, 4.5).u == pytest.approx(0.01) and microstrip(20.0, 0.2, 0.0, 4.5).u == pytest.approx(100.0)


# --------------------------------------------------------------------------- stripline


def test_cohn_exact_and_wheeler_1978_agree_at_zero_thickness():
    worst = 0.0
    for wb in grid(0.02, 5.0, 101):
        exact = stripline_impedance(wb, 1.0, 0.0, 1.0)
        # Wheeler's closed form, evaluated at a vanishing thickness (the formula's own t -> 0 limit)
        wheeler = stripline_impedance(wb, 1.0, 1e-6, 1.0)
        worst = max(worst, abs(exact - wheeler) / exact)
    assert worst < 0.005
    # 1/sqrt(er) scaling of a homogeneous line, and a known exact point: w/b -> the K(k)/K(k') identity at k = k' gives eta0/4
    assert stripline_impedance(0.3, 1.0, 0.0, 4.0) == pytest.approx(stripline_impedance(0.3, 1.0, 0.0, 1.0) / 2.0, rel=1e-12)
    w_sym = 2.0 / math.pi * math.asinh(1.0)  # sech(pi w / 2b) = tanh(pi w / 2b)  <=>  sinh(pi w / 2b) = 1
    assert stripline_impedance(w_sym, 1.0, 0.0, 1.0) == pytest.approx(ETA0 / 4.0, rel=1e-12)


def test_thick_stripline_agrees_with_kicads_thick_strip_forms():
    # KiCad 10 (Cohn's thick-strip / narrow-strip closed forms), centred strip: (er, b mm, w mm, t um) -> Z0
    for (er, b, w, t), z0 in {(4.5, 1.0, 0.3, 17.5): 57.23351382, (2.2, 2.0, 1.5, 35.0): 51.3283655}.items():
        assert stripline_impedance(w, b, t, er) == pytest.approx(z0, rel=0.01)
    assert stripline_impedance(0.3, 1.0, 35.0, 4.5) < stripline_impedance(0.3, 1.0, 17.5, 4.5) < stripline_impedance(0.3, 1.0, 0.0, 4.5)


def test_stripline_refuses_outside_the_checked_range():
    with pytest.raises(TLineRangeError, match="w/b"):
        stripline_impedance(0.01, 1.0, 0.0, 4.5)
    with pytest.raises(TLineRangeError, match="w/b"):
        stripline_impedance(6.0, 1.0, 0.0, 4.5)
    with pytest.raises(TLineRangeError, match="t/b"):
        stripline_impedance(0.3, 0.1, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="below 1"):
        stripline_impedance(0.3, 1.0, 0.0, 0.5)


# --------------------------------------------------------------------------- edge-coupled microstrip


#: KiCad 10's coupled-microstrip calculator (Kirschning-Jansen), 1 Hz, no cover, t = 0: (er, h, w, s mm) -> (Z_even, Z_odd, e_even, e_odd)
KICAD_COUPLED = {
    (4.5, 1.0, 1.0, 1.0): (80.08727951, 59.97166668, 3.451733784, 2.941933216),
    (4.3, 0.2, 0.15, 0.2): (93.85459435, 69.22342238, 3.246350797, 2.792534919),
    (10.0, 1.0, 0.5, 0.2): (89.49262438, 38.40036653, 6.793436074, 5.597414345),
    (2.2, 1.0, 3.0, 5.0): (51.92478841, 50.04518564, 1.907876157, 1.84548949),
    (9.8, 0.635, 0.635, 0.254): (60.7608141, 35.62877482, 7.12423239, 5.690873751),
}


def test_kirschning_jansen_reproduces_kicads_transcription_at_zero_thickness():
    for (er, h, w, s), (z_e, z_o, e_e, e_o) in KICAD_COUPLED.items():
        r = edge_coupled_microstrip(w, s, h, 0.0, er)
        assert (r.z_even, r.z_odd, r.e_eff_even, r.e_eff_odd) == (pytest.approx(z_e, rel=1e-6), pytest.approx(z_o, rel=1e-6), pytest.approx(e_e, rel=1e-6), pytest.approx(e_o, rel=1e-6))
        assert r.z_diff == 2.0 * r.z_odd


def test_kirschning_jansen_agrees_with_hammerstad_jensens_coupled_model():
    worst_e = worst_o = 0.0
    for er in (1.0, 2.2, 4.5, 10.0, 18.0):
        for u in grid(0.1, 10.0, 21):
            for g in grid(0.1, 10.0, 21):
                r = edge_coupled_microstrip(u, g, 1.0, 0.0, er)
                z_e, z_o = hammerstad_jensen_coupled(u, g, er)
                worst_e, worst_o = max(worst_e, abs(r.z_even - z_e) / r.z_even), max(worst_o, abs(r.z_odd - z_o) / r.z_odd)
                assert r.z_odd < microstrip(u, 1.0, 0.0, er).z0_ohm < r.z_even
    assert worst_e < 0.01 and worst_o < 0.02  # measured 0.74 % / 1.83 % (the corner u = g = 10)


def test_both_modes_tend_to_the_single_line_as_the_gap_grows():
    for w, h, t, er in ((0.35, 0.2, 35.0, 4.5), (0.15, 0.2, 35.0, 4.3), (1.0, 1.0, 35.0, 4.5), (0.3, 0.2, 0.0, 4.5)):
        z0 = microstrip(w, h, t, er).z0_ohm
        near, far = edge_coupled_microstrip(w, 0.5 * h, h, t, er), edge_coupled_microstrip(w, 10.0 * h, h, t, er)
        assert far.z_even == pytest.approx(z0, rel=0.015) and far.z_odd == pytest.approx(z0, rel=0.015)
        assert near.z_even - near.z_odd > far.z_even - far.z_odd > 0.0
    thin, thick = edge_coupled_microstrip(0.15, 0.2, 0.2, 0.0, 4.3), edge_coupled_microstrip(0.15, 0.2, 0.2, 35.0, 4.3)
    assert thick.z_odd < thin.z_odd and thick.z_even < thin.z_even and thick.u_odd > thick.u_even > 0.75


def test_z_diff_is_monotonic_in_gap_and_width():
    for t in (0.0, 35.0):
        values = [edge_coupled_microstrip(0.2, s, 0.2, t, 4.5).z_diff for s in grid(0.08, 2.0, 60)]
        assert all(b > a for a, b in zip(values, values[1:]))
        values = [edge_coupled_microstrip(w, 0.2, 0.2, t, 4.5).z_diff for w in grid(0.04, 2.0, 60)]
        assert all(b < a for a, b in zip(values, values[1:]))


def test_coupled_microstrip_refuses_outside_its_validity_range():
    for args, what in (((0.01, 0.2, 0.2, 0.0, 4.5), "u = w/h"), ((3.0, 0.2, 0.2, 0.0, 4.5), "u = w/h"), ((0.2, 0.01, 0.2, 0.0, 4.5), "g = s/h"),
                       ((0.2, 3.0, 0.2, 0.0, 4.5), "g = s/h"), ((0.2, 0.2, 0.2, 0.0, 19.0), "er"), ((0.2, 0.06, 0.2, 35.0, 4.5), "s >> 2t")):
        with pytest.raises(TLineRangeError, match=what):
            edge_coupled_microstrip(*args)


# --------------------------------------------------------------------------- inverses


def test_width_for_z0_round_trips_and_is_deterministic():
    for z0, h, t, er in ((50.0, 0.2, 35.0, 4.5), (75.0, 1.6, 35.0, 4.5), (50.0, 1.53, 35.0, 4.5), (100.0, 0.1, 17.5, 3.66), (28.0, 0.2, 0.0, 10.0)):
        w = solve_microstrip_width(z0, h, t, er)
        assert microstrip(w, h, t, er).z0_ohm == pytest.approx(z0, rel=10 * SOLVE_REL_TOL)
        assert solve_microstrip_width(z0, h, t, er) == w  # the same float, every time
    assert solve_microstrip_width(50.0, 0.2, 35.0, 4.5) == pytest.approx(0.346357, abs=1e-5)  # the generic 4-layer outer layer
    w = solve_stripline_width(50.0, 1.0, 17.5, 4.5)
    assert stripline_impedance(w, 1.0, 17.5, 4.5) == pytest.approx(50.0, rel=10 * SOLVE_REL_TOL)
    s = solve_coupled_spacing(100.0, 0.2, 0.2, 35.0, 4.5)
    assert edge_coupled_microstrip(0.2, s, 0.2, 35.0, 4.5).z_diff == pytest.approx(100.0, rel=10 * SOLVE_REL_TOL)
    w = solve_coupled_width(100.0, 0.2, 0.2, 35.0, 4.5)
    assert edge_coupled_microstrip(w, 0.2, 0.2, 35.0, 4.5).z_diff == pytest.approx(100.0, rel=10 * SOLVE_REL_TOL)


def test_inverses_refuse_targets_outside_the_formulas_range():
    with pytest.raises(TLineRangeError, match="outside what the formula reaches"):
        solve_microstrip_width(500.0, 0.2, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="outside what the formula reaches"):
        solve_microstrip_width(0.5, 0.2, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="outside what the formula reaches"):
        solve_stripline_width(5.0, 1.0, 0.0, 4.5)
    with pytest.raises(TLineRangeError, match="outside what the formula reaches"):
        solve_coupled_spacing(400.0, 0.2, 0.2, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="outside what the formula reaches"):
        solve_coupled_width(20.0, 0.2, 0.2, 35.0, 4.5)
    with pytest.raises(TLineRangeError, match="no valid width"):
        solve_microstrip_width(50.0, 0.0003, 35.0, 4.5)  # the copper is thicker than 100 h: no strip fits the formula


# --------------------------------------------------------------------------- delay and critical length


def test_propagation_delay_and_the_critical_length():
    assert propagation_delay(1.0) == pytest.approx(1.0 / C0) and propagation_delay(4.0) == pytest.approx(2.0 / C0)
    t_pd = propagation_delay(3.24)  # sqrt 1.8
    assert critical_length_mm(1e-9, t_pd, ROUND_TRIP_FRACTION) == pytest.approx(0.5 * 1e-9 * C0 / 1.8 * 1000.0)  # 83.3 mm
    assert critical_length_mm(1e-9, t_pd, LUMPED_FRACTION) == pytest.approx(critical_length_mm(1e-9, t_pd, 0.5) / 3.0)
    for bad in ((0.0, t_pd, 0.5), (1e-9, 0.0, 0.5), (1e-9, t_pd, 0.0), (1e-9, t_pd, 1.5), (1e-9, t_pd, math.nan)):
        with pytest.raises(ValueError):
            critical_length_mm(*bad)
    with pytest.raises(ValueError, match="at least 1"):
        propagation_delay(0.9)


# --------------------------------------------------------------------------- traced wrappers and recompute


def test_traced_wrappers_write_roles_units_and_the_calc_version():
    w, h, t, er = user_requirement(0.35, "mm"), user_requirement(0.2, "mm"), user_requirement(35.0, "um"), user_requirement(4.5)
    z = microstrip_z0(w, h, t, er, ("w_sig", "h_pp", "t_cu", "er_pp"))
    assert z.unit == "ohm" and z.provenance.kind is ProvenanceKind.DERIVED and z.provenance.tool == "calc.tline.microstrip.z0" and z.provenance.tool_version == CALC_VERSION == "0.11"
    assert z.provenance.inputs == {"w": "w_sig", "h": "h_pp", "t": "t_cu", "er": "er_pp"} and "Hammerstad & Jensen 1980" in (z.provenance.note or "")
    e = microstrip_e_eff(w, h, t, er)
    assert e.unit is None and e.value == pytest.approx(microstrip(0.35, 0.2, 35.0, 4.5).e_eff)
    t_pd = tpd(e)
    assert t_pd.unit == "s/m" and t_pd.value == pytest.approx(math.sqrt(e.value) / C0)
    d = line_delay(user_requirement(100.0, "mm"), t_pd)
    assert d.unit == "s" and d.value == pytest.approx(0.1 * t_pd.value)
    l_crit = critical_length(user_requirement(1e-9, "s"), t_pd, user_requirement(0.5))
    assert l_crit.unit == "mm" and l_crit.value == pytest.approx(0.5e-9 / t_pd.value * 1000.0)
    w50 = width_for_z0(user_requirement(50.0, "ohm"), h, t, er, "microstrip")
    assert w50.provenance.tool == "calc.tline.width_for_z0.microstrip" and w50.unit == "mm" and w50.value == width_for_z0_microstrip(user_requirement(50.0, "ohm"), h, t, er).value
    with pytest.raises(ValueError, match="kind"):
        width_for_z0(user_requirement(50.0, "ohm"), h, t, er, "coax")
    zd = edge_coupled_z_diff(user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3))
    assert zd.value == pytest.approx(2 * 69.22342238, rel=1e-6)
    s = spacing_for_zdiff(user_requirement(100.0, "ohm"), user_requirement(0.2, "mm"), h, t, user_requirement(4.5))
    assert s.provenance.tool == "calc.tline.edge_coupled_microstrip.s_for_zdiff" and s.unit == "mm"
    with pytest.raises(TLineRangeError):
        microstrip_z0(user_requirement(0.001), h, user_requirement(0.0), er)


def _stackup_ir() -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="tl", name="tl"))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=50.0, height_mm=40.0), stackup=generic_stackup(4, "fixture", confirmed=True))
    return ir


def test_recompute_rederives_tline_values_from_stackup_ids():
    ir = _stackup_ir()
    geo, why = line_geometry(ir.pcb.stackup, "F.Cu")
    assert why is None
    ir.parameters["z0_target"] = user_requirement(50.0, "ohm")
    ir.parameters["w_50"] = width_for_z0_microstrip(ir.parameters["z0_target"], geo.h, geo.t, geo.er, ("z0_target", *geo.ids))
    ir.parameters["z0_50"] = microstrip_z0(ir.parameters["w_50"], geo.h, geo.t, geo.er, ("w_50", *geo.ids))
    ir.parameters["e_eff_50"] = microstrip_e_eff(ir.parameters["w_50"], geo.h, geo.t, geo.er, ("w_50", *geo.ids))
    ir.parameters["t_pd_50"] = tpd(ir.parameters["e_eff_50"], ("e_eff_50",))
    ir.parameters["t_r"] = user_requirement(1e-9, "s")
    ir.parameters["crit_fraction"] = user_requirement(0.5)
    ir.parameters["l_crit"] = critical_length(ir.parameters["t_r"], ir.parameters["t_pd_50"], ir.parameters["crit_fraction"], ("t_r", "t_pd_50", "crit_fraction"))
    assert ir.parameters["w_50"].provenance.inputs == {"z0": "z0_target", "h": "pcb.stackup.dielectrics[0].thickness_mm", "t": "pcb.stackup.copper[F.Cu].thickness_um", "er": "pcb.stackup.dielectrics[0].er"}
    res = recompute_parameters(ir)
    assert res.status is S.PASS, res.message
    assert set(res.details["parameters"]) == {"w_50", "z0_50", "e_eff_50", "t_pd_50", "l_crit"}
    # an edited stack number no longer gives the stored width: FAIL, both numbers named
    ir.pcb.stackup.dielectrics[0].er = user_requirement(4.2)
    res = recompute_parameters(ir)
    assert res.status is S.FAIL and "w_50" in res.message
    # a stack number that is not there any more: NOT_VERIFIED, not a guess
    ir.pcb.stackup = None
    res = recompute_parameters(ir)
    assert res.status is S.NOT_VERIFIED and "not found in ir.parameters / ir.si / the stackup / ir.requirements" in res.message
    # a unit that does not fit the role is refused (um copper given in mm)
    ir = _stackup_ir()
    ir.parameters["t_mm"] = user_requirement(0.035, "mm")
    ir.parameters["h"] = user_requirement(0.2, "mm")
    ir.parameters["er"] = user_requirement(4.5)
    ir.parameters["w"] = user_requirement(0.35, "mm")
    ir.parameters["z"] = microstrip_z0(ir.parameters["w"], ir.parameters["h"], user_requirement(35.0, "um"), ir.parameters["er"], ("w", "h", "t_mm", "er"))
    res = recompute_parameters(ir)
    assert res.status is S.NOT_VERIFIED and "expects 'um'" in res.message
    assert all(tool in CALCULATORS for tool in ("calc.tline.microstrip.z0", "calc.tline.critical_length", "calc.tline.delay"))


# --------------------------------------------------------------------------- the geometry a stackup gives a layer


def test_line_geometry_reads_the_microstrip_over_the_adjacent_plane():
    four = generic_stackup(4, "fixture", confirmed=False, ground_net="GND", power_net="+3V3")
    top, why = line_geometry(four, "F.Cu")
    assert why is None and top.kind == "microstrip" and (top.reference_layer, top.reference_net) == ("In1.Cu", "GND")
    assert (top.h.value, top.t.value, top.er.value) == (0.2, 35.0, 4.5) and top.ids == ("pcb.stackup.dielectrics[0].thickness_mm", "pcb.stackup.copper[F.Cu].thickness_um", "pcb.stackup.dielectrics[0].er")
    assert any("1e+06 Hz" in n for n in top.notes)
    bottom, _ = line_geometry(four, "B.Cu")
    assert (bottom.reference_layer, bottom.reference_net, bottom.h_id) == ("In2.Cu", "+3V3", "pcb.stackup.dielectrics[2].thickness_mm")
    assert line_geometry(four, "In1.Cu") == (None, "In1.Cu is a plane layer (GND); no signal is routed on it")
    two = generic_stackup(2, "fixture", confirmed=False)
    geo, why = line_geometry(two, "F.Cu")
    assert geo is None and why.startswith(NO_REFERENCE_PLANE)
    assert line_geometry(None, "F.Cu") == (None, NO_STACKUP)
    assert "not in the stackup" in line_geometry(two, "In1.Cu")[1]
    masked = four.model_copy(update={"solder_mask": SolderMask(thickness_um=assumption(20.0, "generic", "um"), er=assumption(3.8, "generic"))})
    assert any("solder mask" in n for n in line_geometry(masked, "F.Cu")[0].notes)
    no_freq = four.model_copy(deep=True)
    no_freq.dielectrics[0].er_frequency_hz = None
    assert any("not recorded" in n for n in line_geometry(no_freq, "F.Cu")[0].notes)
