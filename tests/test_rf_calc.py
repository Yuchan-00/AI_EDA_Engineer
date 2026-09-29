"""The RF calculators (``ai_eda.tools.calc.rf``, ``calc.rf.*`` and ``calc.crystal.c_for_load``).

Every expected number here is computed independently in the test (its own
arithmetic from the textbook formula) or is a published table value / an
anchor the RF foundation design measured with ngspice-42 (the L-match and the
5th-order Butterworth low-pass of ``rf_gap``) - never a call of the helper
under test. Refusals are pinned by their sentence, and the recompute test
tampers a pF and an nH value: under the former absolute floor of
``calc.recompute._same`` both tampered values compared equal.
"""

from __future__ import annotations

import math

import pytest

from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, ValidationStatus, derived, user_requirement
from ai_eda.tools.calc import CALC_VERSION, CALCULATORS, ROLE_UNITS, ROLES, crystal_load_capacitance, lc_cutoff, recompute_parameters
from ai_eda.tools.calc import rf
from ai_eda.tools.calc.recompute import REL_TOL, _same
from ai_eda.tools.calc.tline import propagation_delay

C0 = 299_792_458.0
KB = 1.380649e-23


def U(value: float, unit: str | None = None):
    return user_requirement(value, unit)


# --------------------------------------------------------------------------- registration and provenance


def test_every_rf_calculator_is_registered_with_roles_units_and_the_new_version() -> None:
    assert CALC_VERSION == "0.10"
    assert len(rf.RF_CALCULATORS) == 57
    for tool, fn in rf.RF_CALCULATORS.items():
        assert tool.startswith("calc.rf.") or tool == "calc.crystal.c_for_load"
        assert CALCULATORS[tool] == (fn, ROLES[tool]) and len(ROLE_UNITS[tool]) == len(ROLES[tool])
    # the dB family stays distinct under the recompute's lower-casing
    assert ROLE_UNITS["calc.rf.eirp"] == ("dBm", "dBi", "dB") and ROLE_UNITS["calc.rf.dbuvm_to_vm"] == ("dBuV/m",)
    assert ROLE_UNITS["calc.rf.am.index_from_depth"] == ("percent",) and ROLE_UNITS["calc.rf.ppm_offset"] == ("Hz", "ppm")


def test_a_traced_rf_value_records_its_tool_roles_and_version() -> None:
    p = rf.dbm_to_w(U(27.0, "dBm"), ("req.tx_power",))
    assert p.value == pytest.approx(1e-3 * 10 ** 2.7, rel=1e-15) and p.value == pytest.approx(0.501187233627, rel=1e-12)
    assert p.unit == "W" and p.provenance.kind is ProvenanceKind.DERIVED and p.provenance.tool == "calc.rf.dbm_to_w"
    assert p.provenance.inputs == {"p_dbm": "req.tx_power"} and p.provenance.derived_from == ["req.tx_power"] and p.provenance.tool_version == "0.10"
    c = rf.lmatch_lowpass_c_shunt(U(900e6, "Hz"), U(50.0, "ohm"), U(12.5, "ohm"), ("f0", "rs", "rl"))
    assert c.provenance.inputs == {"f": "f0", "r_source": "rs", "r_load": "rl"} and c.unit == "F"
    with pytest.raises(ValueError, match="takes 1 input ids"):
        rf.dbm_to_w(U(27.0, "dBm"), ("a", "b"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a number"):
        rf.dbm_to_w(user_requirement("27", "dBm"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- levels


def test_levels_convert_both_ways_and_refuse_what_has_no_finite_value() -> None:
    assert rf.w_to_dbm(U(1.0, "W")).value == pytest.approx(30.0, abs=1e-12) and rf.w_to_dbm(U(1.0, "W")).unit == "dBm"
    assert rf.dbw_to_w(U(-3.0, "dBW")).value == pytest.approx(10 ** -0.3, rel=1e-15)
    assert rf.w_to_dbw(U(0.5, "W")).value == pytest.approx(10 * math.log10(0.5), rel=1e-15) and rf.w_to_dbw(U(0.5, "W")).unit == "dBW"
    assert rf.db_to_power_ratio(U(3.0, "dB")).value == pytest.approx(1.9952623149688795, rel=1e-15)
    assert rf.power_ratio_to_db(U(2.0)).value == pytest.approx(3.0102999566398116, rel=1e-15)
    assert rf.db_to_voltage_ratio(U(6.0, "dB")).value == pytest.approx(10 ** 0.3, rel=1e-15)
    assert rf.voltage_ratio_to_db(U(2.0)).value == pytest.approx(6.020599913279624, rel=1e-15)
    # 10 dBm into 50 ohm: P = 10 mW, V_pk = sqrt(2 * 50 * 0.01) = 1 V
    v = rf.dbm_to_vpeak(U(10.0, "dBm"), U(50.0, "ohm"))
    assert v.value == pytest.approx(1.0, rel=1e-15) and v.unit == "V"
    with pytest.raises(ValueError, match="overflows"):
        rf.dbm_to_w(U(4000.0, "dBm"))
    with pytest.raises(ValueError, match="underflows"):
        rf.dbm_to_w(U(-4000.0, "dBm"))
    with pytest.raises(ValueError, match="the power must be positive"):
        rf.w_to_dbm(U(0.0, "W"))
    with pytest.raises(ValueError, match="a voltage ratio must be positive"):
        rf.voltage_ratio_to_db(U(-1.0))


# --------------------------------------------------------------------------- reflection


def test_reflection_vswr_return_and_mismatch_loss() -> None:
    g = rf.gamma_mag(U(25.0, "ohm"), U(0.0, "ohm"), U(50.0, "ohm"))
    assert g.value == pytest.approx(1 / 3, rel=1e-15) and g.unit is None
    z = complex(30.0, 40.0)
    assert rf.gamma_mag(U(30.0, "ohm"), U(40.0, "ohm"), U(50.0, "ohm")).value == pytest.approx(abs((z - 50) / (z + 50)), rel=1e-15)
    assert rf.gamma_mag(U(0.0, "ohm"), U(0.0, "ohm"), U(50.0, "ohm")).value == 1.0  # a short
    assert rf.vswr(U(1 / 3)).value == pytest.approx(2.0, rel=1e-15)
    assert rf.return_loss(U(1 / 3)).value == pytest.approx(20 * math.log10(3), rel=1e-15) and rf.return_loss(U(1 / 3)).unit == "dB"
    assert rf.mismatch_loss(U(1 / 3)).value == pytest.approx(-10 * math.log10(8 / 9), rel=1e-15)
    assert rf.mismatch_loss(U(0.0)).value == 0.0 and rf.vswr(U(0.0)).value == 1.0
    with pytest.raises(ValueError, match="total reflection: the VSWR is infinite"):
        rf.vswr(U(1.0))
    with pytest.raises(ValueError, match="perfect match: the return loss is infinite"):
        rf.return_loss(U(0.0))
    with pytest.raises(ValueError, match="Z0 must be positive"):
        rf.gamma_mag(U(25.0), U(0.0), U(0.0))
    with pytest.raises(ValueError, match="the load resistance must not be negative"):
        rf.gamma_mag(U(-1.0), U(0.0), U(50.0))


# --------------------------------------------------------------------------- matching networks


def _parallel(a: complex, b: complex) -> complex:
    return a * b / (a + b)


def test_lmatch_50_to_12_5_ohm_at_900_mhz_matches_the_measured_anchor() -> None:
    """rf_gap/lmatch.cir (ngspice-42): L 3.8287 nH series to the 12.5 ohm load, C 6.1259 pF across the 50 ohm side -> Zin 50 ohm."""
    f, w = 900e6, 2 * math.pi * 900e6
    q = math.sqrt(50 / 12.5 - 1)
    args = (U(f, "Hz"), U(50.0, "ohm"), U(12.5, "ohm"))
    assert rf.lmatch_q(U(50.0, "ohm"), U(12.5, "ohm")).value == pytest.approx(math.sqrt(3), rel=1e-15)
    l_s, c_p = rf.lmatch_lowpass_l_series(*args), rf.lmatch_lowpass_c_shunt(*args)
    assert l_s.value == pytest.approx(q * 12.5 / w, rel=1e-14) and l_s.value == pytest.approx(3.8287e-9, rel=1e-4) and l_s.unit == "H"
    assert c_p.value == pytest.approx(q / (w * 50), rel=1e-14) and c_p.value == pytest.approx(6.1259e-12, rel=1e-4) and c_p.unit == "F"
    zin = _parallel(12.5 + 1j * w * l_s.value, 1 / (1j * w * c_p.value))
    assert zin.real == pytest.approx(50.0, rel=1e-12) and abs(zin.imag) < 1e-9
    c_s, l_p = rf.lmatch_highpass_c_series(*args), rf.lmatch_highpass_l_shunt(*args)
    assert c_s.value == pytest.approx(1 / (w * q * 12.5), rel=1e-14) and l_p.value == pytest.approx(50 / (w * q), rel=1e-14)
    zin_hp = _parallel(12.5 + 1 / (1j * w * c_s.value), 1j * w * l_p.value)
    assert zin_hp.real == pytest.approx(50.0, rel=1e-12) and abs(zin_hp.imag) < 1e-9
    # the element values do not depend on which side is the source (the shunt element sits across the higher R)
    swapped = (U(f, "Hz"), U(12.5, "ohm"), U(50.0, "ohm"))
    assert rf.lmatch_lowpass_l_series(*swapped).value == l_s.value and rf.lmatch(f, 12.5, 50.0, "lowpass").shunt_at == "load"
    with pytest.raises(ValueError, match="equal resistances need no L-match"):
        rf.lmatch_q(U(50.0, "ohm"), U(50.0, "ohm"))
    with pytest.raises(ValueError, match="the frequency must be positive"):
        rf.lmatch_lowpass_l_series(U(0.0, "Hz"), U(50.0), U(12.5))


def test_pi_match_q5_50_to_12_5_ohm_at_900_mhz() -> None:
    """Bowick's virtual-resistance method, hand values: C_source 17.68 pF, L 2.498 nH, C_load 33.18 pF."""
    f, w = 900e6, 2 * math.pi * 900e6
    rv = 50 / (5 ** 2 + 1)
    q2 = math.sqrt(12.5 / rv - 1)
    args = (U(f, "Hz"), U(50.0, "ohm"), U(12.5, "ohm"), U(5.0))
    cs, l, cl = rf.pimatch_c_source(*args), rf.pimatch_l_series(*args), rf.pimatch_c_load(*args)
    assert cs.value == pytest.approx(1 / (w * 50 / 5), rel=1e-14) and cs.value == pytest.approx(17.68e-12, rel=2e-4)
    assert l.value == pytest.approx((5 + q2) * rv / w, rel=1e-14) and l.value == pytest.approx(2.498e-9, rel=2e-4)
    assert cl.value == pytest.approx(q2 / (w * 12.5), rel=1e-14) and cl.value == pytest.approx(33.18e-12, rel=2e-4)
    # the network transforms the 12.5 ohm load to 50 ohm at f
    zin = _parallel(_parallel(12.5, 1 / (1j * w * cl.value)) + 1j * w * l.value, 1 / (1j * w * cs.value))
    assert zin.real == pytest.approx(50.0, rel=1e-12) and abs(zin.imag) < 1e-9
    with pytest.raises(ValueError, match=r"Q must exceed sqrt\(R_hi/R_lo - 1\) = 1.73205"):
        rf.pimatch_l_series(U(f), U(50.0), U(12.5), U(1.7))


# --------------------------------------------------------------------------- low-pass prototypes


def test_butterworth_and_chebyshev_prototypes_reproduce_the_tables() -> None:
    golden = (1 + math.sqrt(5)) / 2
    bw = [rf.butterworth_g(U(5.0), U(float(k))).value for k in range(1, 7)]
    assert bw == pytest.approx([golden - 1, golden, 2.0, golden, golden - 1, 1.0], rel=1e-14)
    # Pozar 4th ed. table 8.4 / Matthaei 4.05: only the exact 40 / ln 10 reproduces the fourth decimal
    tables = {
        (3, 0.5): [1.5963, 1.0967, 1.5963, 1.0],
        (5, 0.5): [1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0],
        (3, 3.0): [3.3487, 0.7117, 3.3487, 1.0],
        (4, 0.5): [1.6703, 1.1926, 2.3661, 0.8419, 1.9841],
    }
    for (n, ripple), expected in tables.items():
        got = [rf.chebyshev_g(U(float(n)), U(float(k)), U(ripple, "dB")).value for k in range(1, n + 2)]
        assert [round(x, 4) for x in got] == expected, (n, ripple)
    assert rf.CHEBYSHEV_RIPPLE_CONSTANT == pytest.approx(17.3718, abs=1e-4)
    for bad, match in (((5.0, 7.0), "the element index k must lie in 1..6"), ((0.0, 1.0), "the filter order n must lie in 1..20"),
                       ((21.0, 1.0), "the filter order n must lie in 1..20"), ((2.5, 1.0), "the filter order n must be an integer")):
        with pytest.raises(ValueError, match=match):
            rf.butterworth_g(U(bad[0]), U(bad[1]))
    with pytest.raises(ValueError, match="the passband ripple"):
        rf.chebyshev_g(U(3.0), U(1.0), U(0.0, "dB"))


def test_the_5th_order_butterworth_at_1_05_ghz_matches_the_measured_filter() -> None:
    """rf_gap/lpf_run.cir: C1 = C5 1.87358 pF, L2 = L4 12.2628 nH, C3 6.06305 pF (50 ohm, f_c 1.05 GHz); 23.43 dB at 1.8 GHz."""
    w = 2 * math.pi * 1.05e9
    g = [2 * math.sin((2 * k - 1) * math.pi / 10) for k in range(1, 6)]
    c1 = rf.lpf_shunt_c(U(g[0]), U(1.05e9, "Hz"), U(50.0, "ohm"))
    l2 = rf.lpf_series_l(U(g[1]), U(1.05e9, "Hz"), U(50.0, "ohm"))
    c3 = rf.lpf_shunt_c(U(g[2]), U(1.05e9, "Hz"), U(50.0, "ohm"))
    assert c1.value == pytest.approx(g[0] / (w * 50), rel=1e-14) and c1.value == pytest.approx(1.87358e-12, rel=1e-5)
    assert l2.value == pytest.approx(g[1] * 50 / w, rel=1e-14) and l2.value == pytest.approx(12.2628e-9, rel=1e-5)
    assert c3.value == pytest.approx(6.06305e-12, rel=1e-5)
    a = rf.butterworth_attenuation(U(5.0), U(1.8e9, "Hz"), U(1.05e9, "Hz"))
    assert a.value == pytest.approx(10 * math.log10(1 + (1.8 / 1.05) ** 10), rel=1e-13) and round(a.value, 2) == 23.43 and a.unit == "dB"
    assert rf.butterworth_attenuation(U(5.0), U(1e9), U(1e9)).value == pytest.approx(10 * math.log10(2), rel=1e-14)  # -3 dB at f_c
    assert rf.butterworth_attenuation(U(5.0), U(0.0), U(1e9)).value == 0.0
    # far above the corner the value stays finite (computed in the log domain): 20 n log10(f / f_c)
    assert rf.butterworth_attenuation(U(20.0), U(1e300), U(1.0)).value == pytest.approx(20 * 20 * 300, rel=1e-12)


def test_the_chebyshev_attenuation_is_the_ripple_at_the_band_edge_and_42_db_at_twice_it() -> None:
    eps2 = 10 ** 0.05 - 1
    t5_2 = 16 * 2 ** 5 - 20 * 2 ** 3 + 5 * 2  # T5(2) = 362
    a = rf.chebyshev_attenuation(U(5.0), U(0.5, "dB"), U(2e9, "Hz"), U(1e9, "Hz"))
    assert a.value == pytest.approx(10 * math.log10(1 + eps2 * t5_2 ** 2), rel=1e-12) and round(a.value, 2) == 42.04
    assert rf.chebyshev_attenuation(U(5.0), U(0.5), U(1e9), U(1e9)).value == pytest.approx(0.5, rel=1e-12)  # the ripple at the edge
    assert rf.chebyshev_attenuation(U(5.0), U(0.5), U(0.0), U(1e9)).value == pytest.approx(0.0, abs=1e-15)  # n odd: 0 dB at DC
    assert rf.chebyshev_attenuation(U(4.0), U(0.5), U(0.0), U(1e9)).value == pytest.approx(0.5, rel=1e-12)  # n even: the ripple at DC
    assert math.isfinite(rf.chebyshev_attenuation(U(20.0), U(0.5), U(1e300), U(1.0)).value)


def _ladder_loss_db(elements: list[tuple[str, float]], f: float, r_source: float, r_load: float) -> float:
    """The insertion loss P_avail / P_load of a lossless ladder by its ABCD cascade (own arithmetic, no helper)."""
    w = 2 * math.pi * f
    a, b, c, d = 1 + 0j, 0j, 0j, 1 + 0j
    for kind, value in elements:
        if kind == "C":  # shunt admittance jwC
            y = 1j * w * value
            a, b, c, d = a + b * y, b, c + d * y, d
        else:  # series impedance jwL
            z = 1j * w * value
            a, b, c, d = a, a * z + b, c, c * z + d
    num = a * r_load + b + r_source * (c * r_load + d)
    return 10 * math.log10(abs(num) ** 2 / (4 * r_source * r_load))


def test_an_even_order_chebyshev_on_the_shunt_c_first_ladder_ends_in_z0_over_g_n_plus_1() -> None:
    """Pozar 8.3: g_(n+1) is a load conductance after a series L; the module docstring once said the load was g_(n+1) Z0.

    The shunt-C-first ladder of ``lpf_shunt_c`` / ``lpf_series_l`` with n even ends in a series L, so only
    R_L = Z0 / g_(n+1) reproduces ``chebyshev_attenuation`` (n = 4, 0.5 dB: 0.0625 dB at 0.3 f_c, 0.5 dB at f_c,
    as ngspice-42 measured); the load g_(n+1) Z0 loses 1.565 / 3.821 dB there.
    """
    z0, f_c = 50.0, 1e9
    for n in (2, 4, 6):
        for ripple in (0.5, 3.0):
            g = [rf.chebyshev_g(U(float(n)), U(float(k)), U(ripple, "dB")).value for k in range(1, n + 2)]
            ladder = [("C", rf.lpf_shunt_c(U(gk), U(f_c, "Hz"), U(z0, "ohm")).value) if k % 2 == 0
                      else ("L", rf.lpf_series_l(U(gk), U(f_c, "Hz"), U(z0, "ohm")).value) for k, gk in enumerate(g[:n])]
            assert ladder[-1][0] == "L" and g[n] != pytest.approx(1.0)
            for f in (0.3e9, 0.7e9, 1e9, 2e9):
                analytic = rf.chebyshev_attenuation(U(float(n)), U(ripple, "dB"), U(f, "Hz"), U(f_c, "Hz")).value
                assert _ladder_loss_db(ladder, f, z0, z0 / g[n]) == pytest.approx(analytic, abs=1e-9), (n, ripple, f)
            wrong = _ladder_loss_db(ladder, f_c, z0, g[n] * z0)
            assert abs(wrong - ripple) > 1.0, (n, ripple, wrong)
    g4 = [rf.chebyshev_g(U(4.0), U(float(k)), U(0.5, "dB")).value for k in range(1, 6)]
    ladder4 = [("C" if k % 2 == 0 else "L", rf.lpf_shunt_c(U(v), U(f_c), U(z0)).value if k % 2 == 0 else rf.lpf_series_l(U(v), U(f_c), U(z0)).value)
               for k, v in enumerate(g4[:4])]
    assert round(_ladder_loss_db(ladder4, 0.3e9, z0, z0 / g4[4]), 4) == 0.0625 and round(_ladder_loss_db(ladder4, f_c, z0, z0 / g4[4]), 4) == 0.5
    assert round(_ladder_loss_db(ladder4, 0.3e9, z0, g4[4] * z0), 3) == 1.565 and round(_ladder_loss_db(ladder4, f_c, z0, g4[4] * z0), 3) == 3.821
    # an odd order ends in a shunt C with g_(n+1) = 1: the plain Z0 load is right there
    g5 = [rf.chebyshev_g(U(5.0), U(float(k)), U(0.5, "dB")).value for k in range(1, 7)]
    ladder5 = [("C" if k % 2 == 0 else "L", rf.lpf_shunt_c(U(v), U(f_c), U(z0)).value if k % 2 == 0 else rf.lpf_series_l(U(v), U(f_c), U(z0)).value)
               for k, v in enumerate(g5[:5])]
    assert g5[5] == pytest.approx(1.0) and _ladder_loss_db(ladder5, 2e9, z0, z0) == pytest.approx(
        rf.chebyshev_attenuation(U(5.0), U(0.5), U(2e9), U(f_c)).value, abs=1e-9)


# --------------------------------------------------------------------------- wavelengths, link budget, noise


def test_wavelengths_and_the_quarter_wave_monopole() -> None:
    lam = rf.wavelength(U(447e6, "Hz"))
    assert lam.value == pytest.approx(C0 / 447e6, rel=1e-15) and lam.unit == "m"
    assert rf.quarter_wave(U(447e6, "Hz"), U(0.95)).value == pytest.approx(0.95 * C0 / (4 * 447e6), rel=1e-15)
    lg = rf.lambda_g(U(900e6, "Hz"), U(4.5))
    assert lg.unit == "mm" and lg.value == pytest.approx(1000 * C0 / (900e6 * math.sqrt(4.5)), rel=1e-15)
    assert round(lg.value / 10, 2) == 15.70  # lambda_g / 10 at 900 MHz with the no-plane bound sqrt(4.5) (the F12 case)
    # independent of calc.tline, and consistent with its propagation delay
    for e in (1.0, 3.2, 4.5):
        assert rf.lambda_g(U(900e6), U(e)).value == pytest.approx(1000 / (900e6 * propagation_delay(e)), rel=1e-13)
    with pytest.raises(ValueError, match=r"velocity factor must lie in \(0, 1\]"):
        rf.quarter_wave(U(447e6), U(1.2))
    with pytest.raises(ValueError, match="an effective permittivity is at least 1"):
        rf.lambda_g(U(900e6), U(0.9))


def test_free_space_path_loss_and_the_link_budget() -> None:
    pl = rf.fspl(U(915e6, "Hz"), U(1000.0, "m"))
    assert pl.value == pytest.approx(20 * math.log10(4 * math.pi * 1000 * 915e6 / C0), rel=1e-15) and round(pl.value, 2) == 91.68
    with pytest.raises(ValueError, match="far field"):
        rf.fspl(U(915e6), U(0.01))  # inside lambda / (4 pi) = 2.6 cm
    e = rf.eirp(U(27.0, "dBm"), U(2.15, "dBi"), U(0.5, "dB"))
    assert e.value == pytest.approx(28.65, abs=1e-12) and e.unit == "dBm"
    prx = rf.received_power(U(28.65, "dBm"), U(91.68, "dB"), U(2.15, "dBi"), U(0.5, "dB"))
    assert prx.value == pytest.approx(28.65 - 91.68 + 2.15 - 0.5, abs=1e-12)
    assert rf.link_margin(U(-61.38, "dBm"), U(-120.0, "dBm")).value == pytest.approx(58.62, abs=1e-12)
    with pytest.raises(ValueError, match="the feed loss must not be negative"):
        rf.eirp(U(27.0), U(0.0), U(-1.0))


def test_erp_and_eirp_are_bridged_by_the_half_wave_dipole_gain() -> None:
    """An ``erp`` limit (PMR446: 500 mW e.r.p.) is comparable with a computed EIRP only through a registered calculator."""
    assert rf.DIPOLE_GAIN_DBI == 2.15
    # the thin half-wave dipole's directivity 4 / Cin(2 pi) (Cin(2 pi) = 2.43765, Balanis 4.6); regulators round it to 2.15 dB
    cin = sum((1 - math.cos(t)) / t for t in ((i + 0.5) * 2 * math.pi / 20000 for i in range(20000))) * 2 * math.pi / 20000
    assert round(4 / cin, 4) == 1.6409 and round(10 * math.log10(4 / cin), 2) == rf.DIPOLE_GAIN_DBI
    e = rf.erp_to_eirp(U(27.0, "dBm"), ("req.erp",))
    assert e.value == pytest.approx(29.15, abs=1e-12) and e.unit == "dBm" and e.provenance.tool == "calc.rf.erp_to_eirp"
    assert e.provenance.inputs == {"erp": "req.erp"}
    p500 = 10 * math.log10(500.0)  # 500 mW = 26.9897 dBm
    eirp500 = rf.erp_to_eirp(U(p500, "dBm")).value
    assert round(eirp500, 2) == 29.14 and round(1e-3 * 10 ** (eirp500 / 10) * 1e3, 1) == 820.3  # mW EIRP (x 10^0.215 = 1.6406)
    assert rf.eirp_to_erp(U(29.15, "dBm")).value == pytest.approx(27.0, abs=1e-12)
    erp = rf.erp(U(27.0, "dBm"), U(2.15, "dBi"), U(0.5, "dB"))
    assert erp.value == pytest.approx(27.0 + 2.15 - 0.5 - 2.15, abs=1e-12) and erp.unit == "dBm"
    assert ROLES["calc.rf.erp_to_eirp"] == ("erp",) and ROLE_UNITS["calc.rf.erp"] == ("dBm", "dBi", "dB")
    with pytest.raises(ValueError, match="the feed loss must not be negative"):
        rf.erp(U(27.0), U(0.0), U(-1.0))
    with pytest.raises(ValueError, match="the ERP must be a finite number"):
        rf.eirp_dbm_from_erp(float("inf"))
    with pytest.raises(ValueError, match="the EIRP must be a finite number"):
        rf.erp_dbm_from_eirp(float("nan"))
    # the recompute re-derives a stored EIRP from an ERP requirement, and catches a tampered one
    ir = CircuitIR(project=ProjectMeta(id="erp", name="erp"))
    ir.parameters["erp_limit"] = U(27.0, "dBm")
    ir.parameters["eirp_limit"] = rf.erp_to_eirp(ir.parameters["erp_limit"], ("erp_limit",))
    assert recompute_parameters(ir).status is ValidationStatus.PASS
    ir.parameters["eirp_limit"] = derived(27.0, tool="calc.rf.erp_to_eirp", inputs={"erp": "erp_limit"}, unit="dBm", tool_version=CALC_VERSION)
    assert recompute_parameters(ir).status is ValidationStatus.FAIL


def test_thermal_noise_noise_floor_friis_and_sensitivity() -> None:
    ktb = rf.thermal_noise(U(290.0, "K"), U(1.0, "Hz"))
    assert ktb.value == pytest.approx(10 * math.log10(KB * 290 / 1e-3), rel=1e-15) and round(ktb.value, 2) == -173.98
    b = 12.5e3
    floor = 10 * math.log10(KB * 290 * b / 1e-3) + 6.0
    assert rf.noise_floor(U(290.0, "K"), U(b, "Hz"), U(6.0, "dB")).value == pytest.approx(floor, rel=1e-14)
    assert rf.sensitivity(U(290.0), U(b), U(6.0), U(12.0)).value == pytest.approx(floor + 12.0, rel=1e-14)
    f_total = 10 ** 0.15 + (10 ** 0.8 - 1) / 10 ** 1.5  # NF1 1.5 dB, G1 15 dB, NF_rest 8 dB
    assert rf.friis_nf(U(1.5, "dB"), U(15.0, "dB"), U(8.0, "dB")).value == pytest.approx(10 * math.log10(f_total), rel=1e-14)
    # a lossy first stage (G1 < 1) makes the rest count more
    assert rf.friis_nf(U(3.0), U(-3.0), U(3.0)).value > 5.9
    with pytest.raises(ValueError, match="the noise temperature must be positive"):
        rf.thermal_noise(U(0.0, "K"), U(1.0))
    with pytest.raises(ValueError, match="the noise figure must not be negative"):
        rf.noise_floor(U(290.0), U(1.0), U(-1.0))


# --------------------------------------------------------------------------- modulation, field strength


def test_am_and_fm_bandwidth_power_and_indices() -> None:
    assert rf.am_bandwidth(U(3e3, "Hz")).value == 6e3
    assert rf.am_total_power(U(0.1, "W"), U(0.8)).value == pytest.approx(0.1 * (1 + 0.64 / 2), rel=1e-15)
    assert rf.am_sideband_dbc(U(1.0)).value == pytest.approx(20 * math.log10(0.5), rel=1e-15) and rf.am_sideband_dbc(U(1.0)).unit == "dBc"
    m = rf.am_index_from_depth(U(80.0, "percent"))
    assert m.value == pytest.approx(0.8, rel=1e-15) and m.unit is None and m.provenance.inputs == {"depth": "depth"}
    assert rf.fm_carson_bandwidth(U(2.5e3, "Hz"), U(3e3, "Hz")).value == 11e3
    assert rf.fm_modulation_index(U(2.5e3, "Hz"), U(3e3, "Hz")).value == pytest.approx(2.5 / 3, rel=1e-15)
    with pytest.raises(ValueError, match="over-modulated"):
        rf.am_total_power(U(0.1), U(1.2))
    with pytest.raises(ValueError, match="at m = 0 there is no sideband"):
        rf.am_sideband_dbc(U(0.0))
    with pytest.raises(ValueError, match=r"depth must lie in \(0, 100\] percent"):
        rf.am_index_from_depth(U(120.0, "percent"))


def test_field_strength_and_its_inverse_and_dbuv_per_metre() -> None:
    e = rf.field_strength(U(0.75e-3, "W"), U(3.0, "m"))
    assert e.value == pytest.approx(math.sqrt(30 * 0.75e-3) / 3, rel=1e-15) and e.value == pytest.approx(0.05, rel=1e-12) and e.unit == "V/m"
    assert rf.eirp_from_field(U(0.05, "V/m"), U(3.0, "m")).value == pytest.approx((0.05 * 3) ** 2 / 30, rel=1e-15)
    assert rf.dbuvm_to_vm(U(94.0, "dBuV/m")).value == pytest.approx(1e-6 * 10 ** 4.7, rel=1e-15) and round(rf.dbuvm_to_vm(U(94.0)).value, 7) == 0.0501187
    assert rf.vm_to_dbuvm(U(0.05, "V/m")).value == pytest.approx(20 * math.log10(0.05 / 1e-6), rel=1e-15)
    with pytest.raises(ValueError, match="the distance must be positive"):
        rf.field_strength(U(1e-3), U(0.0))


# --------------------------------------------------------------------------- resonance, Q, ppm, crystal load


def test_resonance_q_bandwidth_ppm_and_the_crystal_load_pair() -> None:
    w = 2 * math.pi * 455e3
    l = rf.lc_l_for_resonance(U(455e3, "Hz"), U(1e-9, "F"))
    assert l.value == pytest.approx(1 / (w * w * 1e-9), rel=1e-15) and l.unit == "H"
    # the existing resonance calculator gives the frequency back
    assert lc_cutoff(U(l.value, "H"), U(1e-9, "F")).value == pytest.approx(455e3, rel=1e-13)
    assert rf.lc_c_for_resonance(U(455e3, "Hz"), U(100e-6, "H")).value == pytest.approx(1 / (w * w * 100e-6), rel=1e-15)
    assert rf.q_series(U(447e6, "Hz"), U(47e-9, "H"), U(1.5, "ohm")).value == pytest.approx(2 * math.pi * 447e6 * 47e-9 / 1.5, rel=1e-15)
    assert rf.bandwidth_from_q(U(10.7e6, "Hz"), U(50.0)).value == pytest.approx(214e3, rel=1e-15)
    assert rf.ppm_offset(U(447e6, "Hz"), U(2.5, "ppm")).value == pytest.approx(1117.5, rel=1e-15)
    c = rf.crystal_c_for_load(U(18e-12, "F"), U(4e-12, "F"))
    assert c.value == pytest.approx(28e-12, rel=1e-15) and c.provenance.tool == "calc.crystal.c_for_load"
    assert crystal_load_capacitance(c, c, U(4e-12, "F")).value == pytest.approx(18e-12, rel=1e-14)  # the inverse gives C_L back
    with pytest.raises(ValueError, match="already reaches the load capacitance"):
        rf.crystal_c_for_load(U(4e-12), U(5e-12))


# --------------------------------------------------------------------------- the recompute


def _filter_ir() -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    p = ir.parameters
    p["n"], p["k1"], p["k2"] = U(5.0), U(1.0), U(2.0)
    p["f_c"], p["z0"] = U(1.05e9, "Hz"), U(50.0, "ohm")
    p["g1"] = rf.butterworth_g(p["n"], p["k1"], ("n", "k1"))
    p["g2"] = rf.butterworth_g(p["n"], p["k2"], ("n", "k2"))
    p["c1"] = rf.lpf_shunt_c(p["g1"], p["f_c"], p["z0"], ("g1", "f_c", "z0"))
    p["l2"] = rf.lpf_series_l(p["g2"], p["f_c"], p["z0"], ("g2", "f_c", "z0"))
    p["tx_dbm"] = U(27.0, "dBm")
    p["tx_w"] = rf.dbm_to_w(p["tx_dbm"], ("tx_dbm",))
    return ir


def test_recompute_passes_rf_values_and_fails_a_tampered_pf_and_nh_value() -> None:
    ir = _filter_ir()
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.PASS and res.message == "5 value(s) recomputed"
    # the anchors differ by a factor of 3.2 / 1.2 - under the former absolute floor (1e-9 below 1.0) both compared equal
    for stored, other in ((1.87358e-12, 6.06305e-12), (3.8287e-9, 4.5e-9)):
        assert abs(stored - other) <= REL_TOL * max(1.0, stored, other)  # what the old rule said
        assert not _same(stored, other)  # what the purely relative rule says
    assert _same(0.0, 0.0) and _same(1e-12, 1e-12 * (1 + 1e-10)) and not _same(0.0, 1e-300)
    for key, bad in (("c1", 6.06305e-12), ("l2", 4.5e-9)):
        tampered = _filter_ir()
        t = tampered.parameters[key]
        tampered.parameters[key] = derived(bad, tool=t.provenance.tool, inputs=dict(t.provenance.inputs), unit=t.unit, tool_version=t.provenance.tool_version)
        res = recompute_parameters(tampered)
        assert res.status is ValidationStatus.FAIL and res.details["repair"] == "human"
        assert [m.split(":")[0] for m in res.details["mismatches"]] == [key]


def test_the_recompute_keeps_dbm_and_db_apart() -> None:
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    ir.parameters["p"] = U(27.0, "dB")  # a ratio where a dBm level is expected
    ir.parameters["g"] = U(2.15, "dBi")
    ir.parameters["l"] = U(0.5, "dB")
    ir.parameters["eirp"] = rf.eirp(U(27.0, "dBm"), ir.parameters["g"], ir.parameters["l"], ("p", "g", "l"))
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED
    assert "p_tx='p' carries unit 'dB', calc.rf.eirp expects 'dBm'" in res.details["parameters"]["eirp"]["reason"]
    ir.parameters["p"] = U(27.0, "dBm")
    assert recompute_parameters(ir).status is ValidationStatus.PASS


def test_a_match_takes_positive_real_resistances_only() -> None:
    """The matches take real resistances (a reactive load must be resonated first, module docstring); zero is refused."""
    with pytest.raises(ValueError, match="the load resistance must be positive"):
        rf.lmatch_lowpass_l_series(U(900e6), U(50.0), U(0.0))
    with pytest.raises(ValueError, match="the source resistance must be positive"):
        rf.pimatch_c_source(U(900e6), U(-50.0), U(12.5), U(5.0))
