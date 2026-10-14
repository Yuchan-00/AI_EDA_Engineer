"""The radio-design calculators (``ai_eda.tools.calc.radio``, ``CALC_VERSION`` 0.10).

Every expected number is either computed independently in the test (its own
arithmetic: a separate nodal solve by Gaussian elimination for the top-C
networks, the phase-modulator network written out as complex arithmetic, the
Bessel power series, the textbook closed forms) or one of the KR447 design's
pinned numbers (``kr447/decided/`` and ``kr447/critic2/``, measured on
ngspice-42 when the design was written: the Q_e 20 tanks 4.21 dB and
48.9 / 25.1, 28.8 / 17.8, 14.2 / 9.3 dB at -/+ f_T, the final BPF
29.59 / 22.30 dB, the PM tank -21.43 / +1.39 / +18.21 deg). Only the
ngspice-gated tests at the end run a deck (through
:class:`~ai_eda.tools.spice.NgspiceShared`; skipped without it); they build
the network from the calculators' own element values and compare ngspice
with the calculator's exact response.

The crystal ladder is checked against the fact the KR447 design ran into: the
classical EMRFD equations, with the design's crystal model (C_m 6 fF, C0
4 pF), give 3.4 kHz for a 7.5 kHz target (ngspice-42 measured 3.44 kHz) - so
the C0-aware design refuses that target with the bound, and meets the target
within a few percent where the crystals allow it.
"""

from __future__ import annotations

import cmath
import dataclasses
import inspect
import math
from pathlib import Path

import pytest

from ai_eda.ir import CircuitIR, ProjectMeta, ValidationStatus, derived, user_requirement
from ai_eda.tools.calc import CALC_VERSION, CALCULATORS, ROLE_UNITS, ROLES, format_spice_number, recompute_parameters
from ai_eda.tools.calc import radio as R
from ai_eda.tools.calc import rf
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis

# the KR447 frequency plan (decision 1B: N = 12 = 3 x 2 x 2; low-side LO1 and LO2)
FC = 447.5625e6
IF1 = 21.4e6
IF2 = 450e3
N = 12
FT = FC / N  # 37.296875 MHz
LO1 = FC - IF1  # 426.1625 MHz
FR = LO1 / N  # 35.5135417 MHz
QU = 40.0


def U(value: float, unit: str | None = None):
    return user_requirement(value, unit)


def w(f: float) -> float:
    return 2 * math.pi * f


# --------------------------------------------------------------------------- registration


def test_every_radio_calculator_is_registered_with_its_roles_units_and_the_new_version() -> None:
    assert CALC_VERSION == "0.11"
    assert len(R.RADIO_CALCULATORS) == 62
    for tool, fn in R.RADIO_CALCULATORS.items():
        assert CALCULATORS[tool] == (fn, ROLES[tool])
        params = [p for p in inspect.signature(fn).parameters if p != "ids"]
        # the traced calculator's own parameter order is its roles (the recompute rebuilds the call by role)
        assert tuple(params) == ROLES[tool], tool
        assert len(ROLE_UNITS[tool]) == len(ROLES[tool]), tool
        assert inspect.signature(fn).parameters["ids"].default == ROLES[tool], tool
    # the new ids never shadow an existing calculator
    assert not set(R.RADIO_CALCULATORS) & {"calc.power.P", "calc.power.P_VR", "calc.regulator.p_dissipation", "calc.crystal.c_for_load"}


def test_a_traced_radio_value_records_its_tool_roles_and_version() -> None:
    lo = R.superhet_lo(U(FC, "Hz"), U(IF1, "Hz"), U(-1.0), ("req.carrier_frequency", "rf.if1", "rf.lo1_side"))
    assert lo.value == pytest.approx(426.1625e6, abs=1e-6) and lo.unit == "Hz"
    assert lo.provenance.tool == "calc.rf.superhet.lo" and lo.provenance.tool_version == "0.11"
    assert lo.provenance.inputs == {"f": "req.carrier_frequency", "f_if": "rf.if1", "side": "rf.lo1_side"}
    assert lo.provenance.derived_from == ["req.carrier_frequency", "rf.if1", "rf.lo1_side"]
    assert "LO = f + side IF" in (lo.provenance.note or "")
    with pytest.raises(ValueError, match="takes 3 input ids"):
        R.superhet_lo(U(FC), U(IF1), U(-1.0), ("a", "b"))


# --------------------------------------------------------------------------- frequency plan


def test_the_n12_frequency_plan_reproduces_the_design() -> None:
    s1 = R.mult_stage(U(FT, "Hz"), U(3.0))
    s2 = R.mult_stage(s1, U(2.0))
    s3 = R.mult_stage(s2, U(2.0))
    assert (s1.value, s2.value, s3.value) == (111.890625e6, 223.78125e6, FC)  # exact binary fractions
    rx = [FR * 3, FR * 6, FR * 12]
    assert [round(R.multiplier_stage_hz(FR, 3) / 1e6, 6), round(R.multiplier_stage_hz(FR * 3, 2) / 1e6, 6)] == [106.540625, 213.08125]
    assert rx[2] == pytest.approx(LO1, rel=1e-15)
    assert R.superhet_lo_hz(FC, IF1, -1) == pytest.approx(426.1625e6, abs=1e-6)
    assert R.superhet_lo_hz(IF1, IF2, -1) == pytest.approx(20.95e6, abs=1e-6)  # LO2 = IF1 - IF2
    assert R.superhet_lo_hz(FC, IF1, 1) == pytest.approx(468.9625e6, abs=1e-6)
    assert R.superhet_image_hz(FC, LO1) == pytest.approx(404.7625e6, abs=1e-6)
    assert R.superhet_image_hz(FC, FC + IF1) == pytest.approx(490.3625e6, abs=1e-6)  # the rejected high-side image
    assert R.superhet_half_if_hz(FC, LO1) == pytest.approx(436.8625e6, abs=1e-6)
    assert R.superhet_second_image_hz(FC, IF2) == pytest.approx(446.6625e6, abs=1e-6)
    # the IF-level second image is the second mixer's own image: 2 LO2 - IF1 = 20.5 MHz (the ladder's 2nd-image row)
    assert R.superhet_image_hz(IF1, IF1 - IF2) == pytest.approx(20.5e6, abs=1e-6)
    # LO1 spur responses LO1 + k f_R +/- IF1, nearest first (design 1.4): 440.276, 475.790, 412.049 / 483.076, 376.535 / 518.590, 369.249 MHz
    rows = {(1, -1): 440.276042, (2, -1): 475.789583, (-1, 1): 412.048958, (1, 1): 483.076042, (-2, 1): 376.535417, (2, 1): 518.589583, (-1, -1): 369.248958}
    for (k, s), mhz in rows.items():
        assert R.superhet_lo_spur_response_hz(LO1, FR, k, IF1, s) / 1e6 == pytest.approx(mhz, abs=1e-6)
    # TX close-in products f_c -/+ k f_T and the harmonics
    spurs = [R.multiplier_spur_hz(FC, FT, k) / 1e6 for k in (-1, 1, -2, 2, -3, 3)]
    assert [round(x, 3) for x in spurs] == [410.266, 484.859, 372.969, 522.156, 335.672, 559.453]
    assert [R.harmonic_hz(FC, k) / 1e6 for k in (2, 3, 4)] == [895.125, 1342.6875, 1790.25]


def test_the_frequency_plan_refuses_what_is_no_frequency() -> None:
    with pytest.raises(ValueError, match="the LO side .* must be -1 or \\+1"):
        R.superhet_lo_hz(FC, IF1, 0)
    with pytest.raises(ValueError, match="must be -1 or \\+1"):
        R.superhet_lo_hz(FC, IF1, 0.5)
    with pytest.raises(ValueError, match="not a positive frequency"):
        R.superhet_lo_hz(10e6, 21.4e6, -1)
    with pytest.raises(ValueError, match="an LO at the signal frequency converts to 0 Hz"):
        R.superhet_image_hz(FC, FC)
    with pytest.raises(ValueError, match="not a positive frequency"):
        R.superhet_image_hz(100e6, 40e6)
    with pytest.raises(ValueError, match="the LO spur order k must not be 0"):
        R.superhet_lo_spur_response_hz(LO1, FR, 0, IF1, 1)
    with pytest.raises(ValueError, match="the spur order k must not be 0"):
        R.multiplier_spur_hz(FC, FT, 0)
    with pytest.raises(ValueError, match="not a positive frequency"):
        R.multiplier_spur_hz(FC, FT, -12)  # f_c - 12 f_T = 0 Hz
    with pytest.raises(ValueError, match="must be an integer"):
        R.multiplier_stage_hz(FT, 2.5)
    with pytest.raises(ValueError, match="must lie in 1"):
        R.harmonic_hz(FC, 0)
    with pytest.raises(ValueError, match="the second IF must be positive"):
        R.superhet_second_image_hz(FC, 0.0)
    with pytest.raises(ValueError, match="not a positive frequency"):
        R.superhet_second_image_hz(1e6, 1e6)


# --------------------------------------------------------------------------- FM and the PM-to-FM integrator


def _bessel_series(n: int, x: float) -> float:
    return sum((-1) ** k / (math.factorial(k) * math.factorial(k + n)) * (x / 2) ** (2 * k + n) for k in range(60))


def test_bessel_values_match_the_power_series_and_the_power_sum_is_one() -> None:
    for beta in (0.05, 0.5, 2.5, 5.0, 8.0):
        j = R.bessel_j_values(beta, 12)
        for n in range(13):
            assert j[n] == pytest.approx(_bessel_series(n, beta), abs=1e-13), (beta, n)
    # the identity J0^2 + 2 sum J_k^2 = 1 holds where the power series would have lost its digits
    for beta in (50.0, 300.0):
        j = R.bessel_j_values(beta, int(beta) + 60)
        assert j[0] ** 2 + 2 * sum(v * v for v in j[1:]) == pytest.approx(1.0, abs=1e-12)
    assert R.bessel_j_values(0.0, 3) == [1.0, 0.0, 0.0, 0.0]
    assert R.bessel_j_values(2.5, 1)[0] == pytest.approx(-0.04838377646819, abs=1e-13)  # A&S table 9.1


def _obw_reference(delta_f: float, f_m: float) -> float:
    beta = delta_f / f_m
    tot, n = _bessel_series(0, beta) ** 2, 0
    while tot < 0.99:
        n += 1
        tot += 2 * _bessel_series(n, beta) ** 2
    return 2 * n * f_m


def test_the_99_percent_bandwidth_of_single_tone_fm() -> None:
    # design 1.4: 2.5 kHz deviation at 1 kHz -> 8.0 kHz (Carson 7 kHz); at 3 kHz -> 12.0 kHz (Carson 11 kHz)
    assert R.fm_obw99(U(2.5e3, "Hz"), U(1e3, "Hz")).value == 8000.0
    assert R.fm_obw99_hz(2.5e3, 3e3) == 12000.0
    for df, fm in ((1.5e3, 1e3), (2.5e3, 1.25e3), (2.5e3, 2.5e3), (5e3, 1e3), (2.5e3, 300.0)):
        assert R.fm_obw99_hz(df, fm) == _obw_reference(df, fm)
    assert R.fm_obw99_hz(1e3, 1e4) == 0.0  # beta 0.1: the carrier alone holds 99 %
    assert 2 * (100e3 + 1e3) <= R.fm_obw99_hz(100e3, 1e3) <= 2 * (100e3 + 3e3)  # wideband: about Carson
    with pytest.raises(ValueError, match="the modulating frequency must be positive"):
        R.fm_obw99_hz(2.5e3, 0.0)
    with pytest.raises(ValueError, match="above the 1000 this sideband count evaluates"):
        R.fm_obw99_hz(2e6, 1e3)


def test_the_pm_integrator_and_its_drive_limit() -> None:
    k_pm_tank = 10 * 0.5 * (20e-12 / math.sqrt(1 + 2 / 0.7)) / (0.7 + 2) / 30e-12  # Q |dC/dV| / C_tot at 2 V: 0.6286 rad/V
    tau = R.fm_pm_integrator_tau(U(12.0), U(2 * k_pm_tank, "rad/V"), U(1.0, "V"), U(2.5e3, "Hz"))
    assert tau.value == pytest.approx(12 * 2 * k_pm_tank / (2 * math.pi * 2.5e3), rel=1e-14) and round(tau.value * 1e3, 3) == 0.960  # design 2.1 (1B)
    assert tau.unit == "s"
    assert round(R.pm_integrator_tau_s(36, k_pm_tank, 1.0, 2.5e3) * 1e3, 4) == 1.4407  # the N 36 example
    v = R.pm_drive_limit_v(12, 2 * k_pm_tank, tau.value, 2.5e3)
    assert v == pytest.approx(1.0, rel=1e-14)
    with pytest.raises(ValueError, match="the multiplication factor N must be an integer"):
        R.pm_integrator_tau_s(12.5, 1.0, 1.0, 2.5e3)
    with pytest.raises(ValueError, match="the peak deviation must be positive"):
        R.pm_drive_limit_v(12, 1.0, 1e-3, 0.0)


# --------------------------------------------------------------------------- varactor and the phase-modulator tank


def test_the_varactor_law_and_its_slope() -> None:
    c = R.varactor_c_at_bias(U(20e-12, "F"), U(0.7, "V"), U(0.5), U(2.0, "V"))
    assert c.value == pytest.approx(20e-12 / (1 + 2 / 0.7) ** 0.5, rel=1e-15) and round(c.value * 1e12, 3) == 10.184 and c.unit == "F"
    s = R.varactor_dc_dv(U(20e-12, "F"), U(0.7, "V"), U(0.5), U(2.0, "V"))
    assert round(s.value * 1e12, 4) == -1.8858 and s.unit == "F/V"
    h = 1e-6
    fd = (R.varactor_c_f(20e-12, 0.7, 0.5, 2 + h) - R.varactor_c_f(20e-12, 0.7, 0.5, 2 - h)) / (2 * h)
    assert s.value == pytest.approx(fd, rel=1e-7)
    with pytest.raises(ValueError, match="the reverse bias must not be negative"):
        R.varactor_c_f(20e-12, 0.7, 0.5, -0.1)
    with pytest.raises(ValueError, match="the junction potential VJ must be positive"):
        R.varactor_c_f(20e-12, 0.0, 0.5, 1.0)


def _pm_network_phase(v: float, *, c_fixed: float, l: float, r_s: float, bypass: float = 10e-9, feed: float = 10e3, load: float | None = None) -> float:
    """The pm_n12.cir network written out: 50 ohm, 1 nF, R_s, tank (C_fixed + C_var // L + wL/40 + bypass // feed)."""
    s = 1j * w(FT)
    zl = s * l + w(FT) * l / QU
    zret = 1 / (s * bypass + 1 / feed)
    cv = 20e-12 / (1 + v / 0.7) ** 0.5
    y = 1 / (zl + zret) + s * (c_fixed + cv) + (0 if load is None else 1 / load)
    ztank = 1 / y
    h = ztank / (50 + 1 / (s * 1e-9) + r_s + ztank)
    return math.degrees(cmath.phase(h))


def test_the_pm_tank_design_values_and_its_exact_network_phase() -> None:
    c_tot = 30e-12
    l = rf.l_for_resonance_h(FT, c_tot)  # calc.rf.lc.l_for_resonance
    assert l == pytest.approx(1 / (w(FT) ** 2 * c_tot), rel=1e-15) and round(l * 1e9, 2) == 606.98  # design 2.4: L 607.0 nH
    c_var = R.varactor_c_f(20e-12, 0.7, 0.5, 2.0)
    c_fixed = R.pm_c_fixed(U(c_tot, "F"), U(c_var, "F"), U(0.0, "F")).value
    assert round(c_fixed * 1e12, 3) == 19.816  # C801 + C_T801 = 19.82 pF
    r_s = R.pm_source_r(U(FT, "Hz"), U(l, "H"), U(10.0), U(QU), U(50.0, "ohm")).value
    assert r_s == pytest.approx(1 / (1 / (10 * w(FT) * l) - 1 / (QU * w(FT) * l)) - 50, rel=1e-14) and round(r_s, 1) == 1846.6  # pm_n12.cir 1846.554
    k_pm = R.pm_k_pm(U(10.0), U(R.varactor_slope_f_per_v(20e-12, 0.7, 0.5, 2.0), "F/V"), U(c_tot, "F"))
    assert round(k_pm.value, 4) == 0.6286 and k_pm.unit == "rad/V"  # formula 0.629 per tank
    phases = {}
    for v in (1.44, 2.0, 2.56):
        c_v = R.varactor_c_f(20e-12, 0.7, 0.5, v)
        got = R.pm_tank_phase(U(FT, "Hz"), U(FT, "Hz"), U(l, "H"), U(QU), U(c_fixed, "F"), U(0.0, "F"), U(c_v, "F"), U(50.0, "ohm"), U(1e-9, "F"),
                              U(r_s, "ohm"), U(10e-9, "F"), U(10e3, "ohm")).value
        assert got == pytest.approx(_pm_network_phase(v, c_fixed=c_fixed, l=l, r_s=r_s), abs=1e-9)
        phases[v] = got
    # critic2 (pm_check.py): the network gives -21.46 / +1.39 / +18.23 deg (ngspice-42: -21.43 / +1.39 / +18.21)
    assert [round(phases[v], 2) for v in (1.44, 2.0, 2.56)] == [-21.46, 1.39, 18.23]
    for v, spice in ((1.44, -21.43), (2.0, 1.39), (2.56, 18.21)):
        assert phases[v] == pytest.approx(spice, abs=0.05)
    # the ideal formula -atan(Q (f/f0 - f0/f)) is 0.83 / 1.39 / 1.01 deg off the network: it would FAIL tol 1 deg
    for v, off in ((1.44, 0.83), (2.0, 1.39), (2.56, 1.01)):
        f0 = 1 / (2 * math.pi * math.sqrt(l * (c_fixed + R.varactor_c_f(20e-12, 0.7, 0.5, v))))
        ideal = -math.degrees(math.atan(10 * (FT / f0 - f0 / FT)))
        assert round(phases[v] - ideal, 2) == off
    chord = math.radians(phases[2.56] - phases[1.44]) / 1.12
    assert round(chord, 4) == 0.6186  # the network chord slope rf.deviation takes (formula 0.6157)
    # a load at the tank node: the loaded variant; a huge load resistance is the probe
    c_v = R.varactor_c_f(20e-12, 0.7, 0.5, 2.0)
    args = (FT, FT, l, QU, c_fixed, 0.0, c_v, 50.0, 1e-9, r_s, 10e-9, 10e3)
    assert R.pm_tank_phase_loaded_deg(*args, 1e13) == pytest.approx(R.pm_tank_phase_deg(*args), abs=1e-6)
    assert R.pm_tank_phase_loaded_deg(*args, 10e3) == pytest.approx(_pm_network_phase(2.0, c_fixed=c_fixed, l=l, r_s=r_s, load=10e3), abs=1e-9)
    r_s_loaded = R.pm_source_r_loaded_ohm(FT, l, 10.0, QU, 50.0, 10e3)
    assert r_s_loaded == pytest.approx(1 / (1 / (10 * w(FT) * l) - 1 / (QU * w(FT) * l) - 1 / 10e3) - 50, rel=1e-14)
    with pytest.raises(ValueError, match="the loaded Q 40 must stay below the inductor's unloaded Q 40"):
        R.pm_source_r_ohm(FT, l, 40.0, 40.0, 50.0)
    with pytest.raises(ValueError, match="the port resistance 5000 ohm alone loads the tank"):
        R.pm_source_r_ohm(FT, l, 10.0, QU, 5000.0)
    with pytest.raises(ValueError, match="already exceed C_tot"):
        R.pm_c_fixed_f(10e-12, 10.18e-12, 0.0)
    with pytest.raises(ValueError, match="a varactor slope dC/dV of 0 modulates nothing"):
        R.pm_k_pm_rad_per_v(10.0, 0.0, 30e-12)


# --------------------------------------------------------------------------- top-C coupled-resonator networks


def _nodal_s21_db(net: R.TopCNetwork, q_u: float, f: float) -> float:
    """An independent nodal solve (Gaussian elimination, as kr447/critic2/topc_nodal.py) of the same network."""
    n, s = net.n, 1j * w(f)
    m = n + 2
    y = [[0j] * m for _ in range(m)]

    def add(a: int, b: int | None, adm: complex) -> None:
        y[a][a] += adm
        if b is not None:
            y[b][b] += adm
            y[a][b] -= adm
            y[b][a] -= adm

    rq = w(net.f0) * net.l / q_u
    add(0, None, 1 / net.r_source)
    add(0, 1, s * net.c_tap_source)
    for i in range(1, n + 1):
        add(i, None, s * net.c_shunt[i - 1] + 1 / (rq + s * net.l))
        if i < n:
            add(i, i + 1, s * net.c_couple[i - 1])
    add(n, n + 1, s * net.c_tap_load)
    add(n + 1, None, 1 / net.r_load)
    rhs = [0j] * m
    rhs[0] = 1 / net.r_source
    a = [row[:] + [rhs[i]] for i, row in enumerate(y)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(c + 1, m):
            f_ = a[r][c] / a[c][c]
            for k in range(c, m + 1):
                a[r][k] -= f_ * a[c][k]
    v = [0j] * m
    for r in range(m - 1, -1, -1):
        v[r] = (a[r][m] - sum(a[r][k] * v[k] for k in range(r + 1, m))) / a[r][r]
    return 20 * math.log10(abs(2 * v[n + 1] * math.sqrt(net.r_source / net.r_load)))


def _tank(f0: float, rs: float = 50.0, rl: float = 50.0) -> tuple[float, float, float]:
    l = 1 / (w(f0) ** 2 * 30e-12) if f0 < 300e6 else 4.7e-9  # C_res 30 pF below 300 MHz, L 4.7 nH above (design 2.7 (b))
    return f0, R.top_c_bw_for_qe_hz(2, f0, 20.0), l


def test_the_q_e_20_multiplier_tanks_reproduce_the_ngspice_numbers_and_are_asymmetric() -> None:
    # decided/tank_explore.py (ngspice-42, Q_u 40, 50 ohm): s21 -4.21 dB; -f_T / +f_T 48.9 / 25.1, 28.8 / 17.8, 14.2 / 9.3 dB
    pins = {3: (48.9, 25.1), 6: (28.8, 17.8), 12: (14.2, 9.3)}
    for mult, (lo_side, hi_side) in pins.items():
        f0, bw, l = _tank(FT * mult)
        assert bw == pytest.approx(math.sqrt(2) * f0 / 20, rel=1e-15)
        s21 = R.top_c_s21_db(U(2.0), U(f0, "Hz"), U(bw, "Hz"), U(l, "H"), U(50.0, "ohm"), U(50.0, "ohm"), U(QU), U(f0, "Hz"))
        assert round(s21.value, 2) == -4.21 and s21.unit == "dB"
        rel_m = R.top_c_rel_s21_db_value(2, f0, bw, l, 50, 50, QU, f0 - FT, f0)
        rel_p = R.top_c_rel_s21_db(U(2.0), U(f0), U(bw), U(l), U(50.0), U(50.0), U(QU), U(f0 + FT), U(f0)).value
        assert rel_m == pytest.approx(-lo_side, abs=0.1) and rel_p == pytest.approx(-hi_side, abs=0.1)
        net = R.top_c_network(2, f0, bw, l, 50, 50)
        for f in (f0, f0 - FT, f0 + FT):
            assert R.top_c_s21_db_value(2, f0, bw, l, 50, 50, QU, f) == pytest.approx(_nodal_s21_db(net, QU, f), abs=1e-9)
        # the symmetric narrowband formula (36.7 / 25.7 / 14.4 dB) is up to 11.6 dB off the network's + side
        sym = R.double_tuned_rejection_db(20.0, f0, f0 + FT)
        assert abs(sym - hi_side) > 4.0
        # Cohn's loss for n = 2 at Q_e 20: 4.343 dB (network 4.21)
        cohn = R.bpf_dissipation_loss(U(2.0), U(f0), U(bw), U(QU)).value
        assert cohn == pytest.approx(10 / math.log(10) * 2 * math.sqrt(2) * 20 / (math.sqrt(2) * QU), rel=1e-14) and round(cohn, 3) == 4.343
    assert [round(R.double_tuned_rejection_db(20.0, FT * m, FT * (m + 1)), 1) for m in (3, 6, 12)] == [36.7, 25.7, 14.4]
    # the stage ports of design 2.7 (c): 500 ohm at both ends of the 111.89 MHz tank moves the asymmetry to 46.0 / 27.4 dB
    f0, bw, l = _tank(3 * FT)
    assert round(-R.top_c_rel_s21_db_value(2, f0, bw, l, 500, 500, QU, f0 - FT, f0), 1) == 46.0
    assert round(-R.top_c_rel_s21_db_value(2, f0, bw, l, 500, 500, QU, f0 + FT, f0), 1) == 27.4


def test_the_final_bpf_front_end_and_lo_filters_reproduce_the_design() -> None:
    # tx_bpf: 3-pole, 20 MHz, L 4.7 nH, 50 ohm, Q_u 40: s21 -9.30 dB; f_c -/+ f_T 29.59 / 22.30 dB; 2 f_c 60.98 dB
    assert round(R.top_c_s21_db_value(3, FC, 20e6, 4.7e-9, 50, 50, QU, FC), 2) == -9.30
    rel = [-R.top_c_rel_s21_db_value(3, FC, 20e6, 4.7e-9, 50, 50, QU, f, FC) for f in (FC - FT, FC + FT, FC - 2 * FT, FC + 2 * FT, 1.5 * FC, 2 * FC)]
    assert [round(x, 2) for x in rel] == [29.59, 22.30, 51.39, 36.25, 54.03, 60.98]
    # fe_bpf2 (2-pole, 40 MHz) and fe_bpf3 at the image: 13.56 + 33.57 dB (the design's 47 dB total image rejection)
    assert round(-R.top_c_rel_s21_db_value(2, FC, 40e6, 4.7e-9, 50, 50, QU, LO1 - IF1, FC), 2) == 13.56
    assert round(-R.top_c_rel_s21_db_value(3, FC, 20e6, 4.7e-9, 50, 50, QU, LO1 - IF1, FC), 2) == 33.57
    assert round(R.top_c_s21_db_value(2, FC, 40e6, 4.7e-9, 50, 50, QU, FC), 2) == -3.37
    # the 440.28 MHz response (image + f_R) is rejected by only 2.48 dB by both front-end filters together
    f_resp = R.superhet_lo_spur_response_hz(LO1, FR, 1, IF1, -1)
    both = R.top_c_rel_s21_db_value(2, FC, 40e6, 4.7e-9, 50, 50, QU, f_resp, FC) + R.top_c_rel_s21_db_value(3, FC, 20e6, 4.7e-9, 50, 50, QU, f_resp, FC)
    assert round(-both, 2) == 2.48
    # lo_bpf: 2-pole, 40 MHz at LO1: -3.21 dB; LO1 -/+ f_R 10.49 / 6.12 dB; Cohn 3.27 dB
    assert round(R.top_c_s21_db_value(2, LO1, 40e6, 4.7e-9, 50, 50, QU, LO1), 2) == -3.21
    assert round(-R.top_c_rel_s21_db_value(2, LO1, 40e6, 4.7e-9, 50, 50, QU, LO1 - FR, LO1), 2) == 10.49
    assert round(-R.top_c_rel_s21_db_value(2, LO1, 40e6, 4.7e-9, 50, 50, QU, LO1 + FR, LO1), 2) == 6.12
    assert round(R.dissipation_loss_db(2, LO1, 40e6, QU), 2) == 3.27
    # Cohn against the network: fe_bpf2 3.44 vs 3.37, fe_bpf3 9.72 vs 9.30 dB (within 0.5 dB, the design's check)
    for n, bw, cohn in ((2, 40e6, 3.44), (3, 20e6, 9.72)):
        assert round(R.dissipation_loss_db(n, FC, bw, QU), 2) == cohn
        assert abs(R.dissipation_loss_db(n, FC, bw, QU) + R.top_c_s21_db_value(n, FC, bw, 4.7e-9, 50, 50, QU, FC)) < 0.5


def test_the_top_c_element_values_and_a_lossless_network() -> None:
    f0, bw, l = FC, 20e6, 4.7e-9
    c_res = 1 / (w(f0) ** 2 * l)
    g = [1.0, 2.0, 1.0]  # Butterworth n = 3
    net = R.top_c_network(3, f0, bw, l, 50, 50)
    assert net.c_res == pytest.approx(c_res, rel=1e-15) and net.q_e == pytest.approx(g[0] * f0 / bw, rel=1e-15)
    for i in (1, 2):
        cc = R.top_c_c_couple(U(3.0), U(float(i)), U(f0, "Hz"), U(bw, "Hz"), U(l, "H"))
        assert cc.value == pytest.approx(c_res * (bw / f0) / math.sqrt(g[i - 1] * g[i]), rel=1e-14) == net.c_couple[i - 1]
    rp = net.q_e * w(f0) * l
    cs = R.top_c_c_tap(U(3.0), U(f0), U(bw), U(l), U(50.0)).value
    assert cs == pytest.approx(1 / (w(f0) * math.sqrt(rp * 50 - 50 * 50)), rel=1e-14) == net.c_tap_source
    ceq = cs / (1 + (w(f0) * cs * 50) ** 2)
    shunt = [R.top_c_c_shunt(U(3.0), U(float(i)), U(f0), U(bw), U(l), U(50.0), U(50.0)).value for i in (1, 2, 3)]
    assert shunt[0] == pytest.approx(c_res - net.c_couple[0] - ceq, rel=1e-13)
    assert shunt[1] == pytest.approx(c_res - net.c_couple[0] - net.c_couple[1], rel=1e-13) and shunt == list(net.c_shunt)
    # lossless and matched at f0 (the taps transform exactly at f0), about -3 dB at the band edges (narrowband design)
    assert R.top_c_s21_db_value(3, f0, bw, l, 50, 50, 1e9, f0) == pytest.approx(0.0, abs=1e-5)
    assert -3.5 < R.top_c_s21_db_value(3, f0, bw, l, 50, 50, 1e9, f0 - bw / 2) < -2.5
    assert -3.5 < R.top_c_s21_db_value(3, f0, bw, l, 50, 50, 1e9, f0 + bw / 2) < -2.5
    assert R.q_parallel(U(450e3, "Hz"), U(125e-6, "H"), U(10e3, "ohm")).value == pytest.approx(10e3 / (w(450e3) * 125e-6), rel=1e-15)


def test_the_top_c_design_refuses_a_tap_that_cannot_transform_and_a_negative_resonator() -> None:
    f0, bw, l = _tank(3 * FT)
    # the design's own BFR92 port models (1 kohm / 500 ohm) cannot be tapped at L 67.4 nH: R_p = 948 ohm (critic2 2.7 (c))
    with pytest.raises(ValueError, match="the source termination 1000 ohm is not below the end resonator's R_p = Q_e w0 L = 948.277 ohm: a capacitive tap only transforms down - choose a larger L or another tap"):
        R.top_c_s21_db_value(2, f0, bw, l, 1000, 500, QU, f0)
    with pytest.raises(ValueError, match="resonator 1's shunt capacitor would be .* choose a smaller L"):
        R.top_c_network(2, 100e6, 50e6, 100e-9, 10, 10)
    with pytest.raises(ValueError, match="a single resonator has no coupling capacitor"):
        R.top_c_c_couple(U(1.0), U(1.0), U(f0), U(bw), U(l))
    with pytest.raises(ValueError, match="is 60 % of the centre: the coupled-resonator design is a narrowband method \\(at most 50 %\\)"):
        R.top_c_network(2, 100e6, 60e6, 10e-9, 50, 50)
    with pytest.raises(ValueError, match="the inductor's Q_u must be positive"):
        R.top_c_s21_db_value(2, f0, bw, l, 50, 50, 0.0, f0)


def _nodal_ported_s21_db(p: R.PortedTopC, q_u: float, f: float) -> float:
    """An independent nodal solve of a ported top-C network: the source port R_port with its shunt choke (series loss at f0) to ground, the
    network, the load port R_load with the extra shunt conductance; S21 normalised to the two port resistances."""
    net, s = p.net, 1j * w(f)
    n = net.n
    m = n + 2
    y = [[0j] * m for _ in range(m)]

    def add(a: int, b: int | None, adm: complex) -> None:
        y[a][a] += adm
        if b is not None:
            y[b][b] += adm
            y[a][b] -= adm
            y[b][a] -= adm

    add(0, None, 1 / p.r_port + 1 / (w(net.f0) * p.l_port / p.q_port + s * p.l_port))
    add(0, 1, s * net.c_tap_source)
    rq = w(net.f0) * net.l / q_u
    for i in range(1, n + 1):
        add(i, None, s * net.c_shunt[i - 1] + 1 / (rq + s * net.l))
        if i < n:
            add(i, i + 1, s * net.c_couple[i - 1])
    add(n, n + 1, s * net.c_tap_load)
    add(n + 1, None, 1 / p.r_load + p.g_extra)
    a = [row[:] + [1 / p.r_port if i == 0 else 0j] for i, row in enumerate(y)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        for r in range(c + 1, m):
            f_ = a[r][c] / a[c][c]
            for k in range(c, m + 1):
                a[r][k] -= f_ * a[c][k]
    v = [0j] * m
    for r in range(m - 1, -1, -1):
        v[r] = (a[r][m] - sum(a[r][k] * v[k] for k in range(r + 1, m))) / a[r][r]
    return 20 * math.log10(abs(2 * v[n + 1] * math.sqrt(p.r_port / p.r_load)))


def test_the_ported_top_c_network_absorbs_the_collector_choke_and_loads_the_next_divider() -> None:
    """kr447 wave-2 review finding 2: a tank between a collector with its feed choke and a base with its divider is designed for those loads."""
    f0, l, l_ch, q = 223.78125e6, 68e-9, 1e-6, 40.0  # the x6 tank: 68 nH resonators, the 1 uH collector choke, Q 40 everywhere
    bw = R.top_c_bw_for_qe_hz(2, f0, 20.0)
    r_eff = 1 / (1 / 500 + 1 / 4700 + 1 / 2200)  # the next stage's 500 ohm port model // its base divider
    z = 1000 * complex(w(f0) * l_ch / q, w(f0) * l_ch) / (1000 + complex(w(f0) * l_ch / q, w(f0) * l_ch))
    ids = ("r_port", "l_port", "q_port", "f0")
    pr = R.top_c_port_r(U(1000.0, "ohm"), U(l_ch, "H"), U(q), U(f0, "Hz"), ids)
    px = R.top_c_port_x(U(1000.0, "ohm"), U(l_ch, "H"), U(q), U(f0, "Hz"), ids)
    assert (pr.value, px.value) == pytest.approx((z.real, z.imag), rel=1e-12) and pr.unit == px.unit == "ohm" and pr.provenance.tool_version == "0.11"
    assert pr.value == pytest.approx(660.44, abs=0.01) and px.value == pytest.approx(461.22, abs=0.01)  # the choke is of the port's order
    p = R.top_c_ported_network(2, f0, bw, l, 1000, l_ch, q, 500, r_eff)
    for f in (f0, f0 - FT, f0 + FT):
        assert R.top_c_ported_s21_db_value(2, f0, bw, l, 1000, l_ch, q, 500, r_eff, q, f) == pytest.approx(_nodal_ported_s21_db(p, q, f), abs=1e-9)
    s21 = R.top_c_ported_s21_db(U(2.0), U(f0, "Hz"), U(bw, "Hz"), U(l, "H"), U(1000.0, "ohm"), U(l_ch, "H"), U(q), U(500.0, "ohm"), U(r_eff, "ohm"), U(q),
                                U(f0, "Hz"))
    assert s21.value == pytest.approx(-5.5417, abs=1e-4) and s21.provenance.tool == "calc.rf.resonator.top_c.ported_s21_db"
    rel_p = R.top_c_ported_rel_s21_db(U(2.0), U(f0), U(bw), U(l), U(1000.0), U(l_ch), U(q), U(500.0), U(r_eff), U(q), U(f0 + FT), U(f0))
    assert rel_p.value == pytest.approx(-17.6727, abs=1e-4)
    # the tap into R' + j X': the resistive tap's formula with X' absorbed
    c_s = R.top_c_c_tap_reactive(U(2.0), U(f0), U(bw), U(l), pr, px).value
    r_p = math.sqrt(2) * f0 / bw * w(f0) * l  # Q_e w0 L with g_1 = sqrt(2)
    assert c_s == pytest.approx(1 / (w(f0) * (px.value + math.sqrt(pr.value * (r_p - pr.value)))), rel=1e-12) == p.net.c_tap_source
    # the choke is not negligible: the plain network between the bare port models is another network (the review's 2-3 dB on one side)
    plain = R.top_c_rel_s21_db_value(2, f0, bw, l, 660.4436, r_eff, q, f0 + FT, f0)
    assert abs(plain - rel_p.value) > 0.1
    # with a choke far above the port and no extra load the ported network is the plain one
    far = R.top_c_ported_s21_db_value(2, f0, bw, l, 500, 1.0, 1e9, 500, 500, q, f0 + FT)
    assert far == pytest.approx(R.top_c_s21_db_value(2, f0, bw, l, 500, 500, q, f0 + FT), abs=1e-4)
    with pytest.raises(ValueError, match="the load the network sees .* cannot exceed it"):
        R.top_c_ported_network(2, f0, bw, l, 1000, l_ch, q, 500, 600)


def test_the_symmetric_tuned_circuit_formulas_and_the_single_resonator_loss() -> None:
    x = 20 * (484.859375 / 447.5625 - 447.5625 / 484.859375)
    assert R.single_tuned_rejection(U(20.0), U(FC, "Hz"), U(FC + FT, "Hz")).value == pytest.approx(10 * math.log10(1 + x * x), rel=1e-14)
    assert R.double_tuned_rejection(U(20.0), U(FC), U(FC + FT)).value == pytest.approx(10 * math.log10(1 + x ** 4 / 4), rel=1e-14)
    assert R.single_tuned_rejection_db(20.0, FC, FC) == 0.0
    il = R.single_tuned_insertion_loss(U(20.0), U(40.0))
    assert il.value == pytest.approx(20 * math.log10(2), rel=1e-15) and round(il.value, 2) == 6.02 and il.unit == "dB"
    with pytest.raises(ValueError, match="the loaded Q 40 must stay below the unloaded Q 40"):
        R.single_tuned_insertion_loss_db(40.0, 40.0)


# --------------------------------------------------------------------------- crystal ladder


def _bw3(lad: R.CrystalLadder, r_m: float) -> float:
    """The half-power bandwidth of a ladder's exact response, by a 1 Hz scan (independent of calc.crystal.ladder.center)."""
    fs = [lad.f_mesh - 15e3 + i * 1.0 for i in range(30001)]
    db = [20 * math.log10(abs(R.crystal_ladder_s21(lad, r_m, f))) for f in fs]
    pk = max(db)
    inside = [f for f, d in zip(fs, db) if d >= pk - 10 * math.log10(2)]
    return inside[-1] - inside[0]


def test_the_ladder_without_c0_is_the_classical_emrfd_design() -> None:
    # 6-pole Butterworth, 7.5 kHz, C_m 6 fF at 21.4 MHz: k 1.1688 / 0.6050 / 0.5176, q 0.5176, C_ij 14.65 / 28.30 / 33.07 pF, R_end 839 ohm
    assert [round(R.ladder_k(U(6.0), U(float(i))).value, 4) for i in (1, 2, 3)] == [1.1688, 0.605, 0.5176]
    assert round(R.ladder_q(U(6.0)).value, 4) == 0.5176
    lad = R.crystal_ladder(6, 7.5e3, 21.4e6, 6e-15, 0.0)
    lm = 1 / (w(21.4e6) ** 2 * 6e-15)
    assert lad.l_m == pytest.approx(lm, rel=1e-15) and round(lm * 1e3, 3) == 9.219
    classical = [6e-15 * 21.4e6 / (k * 7.5e3) for k in R.ladder_k_values(6)]
    assert [c * 1e12 for c in lad.c_couple] == pytest.approx([c * 1e12 for c in classical], rel=1e-3)
    assert [round(c * 1e12, 2) for c in classical] == [14.65, 28.3, 33.07, 28.3, 14.65]
    assert lad.r_end == pytest.approx(w(7.5e3) * lm / 0.517638, rel=1e-3) and round(lad.r_end) == 839
    assert lad.c_mesh[1] is None and lad.c_mesh[4] is None and all(c is not None for i, c in enumerate(lad.c_mesh) if i not in (1, 4))
    # without C0 the design meets the bandwidth exactly and centres on the mesh frequency
    assert _bw3(lad, 0.0) == pytest.approx(7500.0, abs=2.0)
    assert R.crystal_ladder_center_hz(6, 7.5e3, 21.4e6, 6e-15, 0.0, 0.0) == pytest.approx(lad.f_mesh, abs=1.0)


def test_the_classical_ladder_with_c0_is_far_too_narrow_and_the_c0_aware_one_refuses_the_target() -> None:
    lad = R.crystal_ladder(6, 7.5e3, 21.4e6, 6e-15, 0.0)
    with_c0 = dataclasses.replace(lad, c0=4e-12)
    # the design's measurement (discrete/sim/ladder6.cir, ngspice-42): 3.44 kHz instead of 7.5 kHz with R_m 25 ohm
    assert _bw3(with_c0, 25.0) == pytest.approx(3440.0, abs=60.0)
    msg = ("no 6-crystal Butterworth ladder of 7500 Hz exists with C_m 6e-15 F and C0 4e-12 F at 2.14e+07 Hz: C0 limits this lower-sideband "
           "ladder to about f_s C_m / (4 C0 kappa) = 4524.26 Hz")
    with pytest.raises(ValueError, match=msg.replace("(", "\\(").replace(")", "\\)").replace("+", "\\+")):
        R.ladder_r_end(U(6.0), U(7.5e3, "Hz"), U(21.4e6, "Hz"), U(6e-15, "F"), U(4e-12, "F"))
    # below the bound the lossless C0-aware design meets its target within a few percent (the classical one gave -49 % lossless, -54 % with R_m 25)
    lad3 = R.crystal_ladder(6, 3.5e3, 21.4e6, 6e-15, 4e-12)
    assert _bw3(lad3, 0.0) == pytest.approx(3500.0, rel=0.05)
    # ... but only lossless: with the design's R_m 25 ohm the same ladder is about 9 % narrower (the module docstring's numbers)
    assert _bw3(lad3, 25.0) == pytest.approx(3177.0, abs=15.0) and _bw3(lad3, 25.0) < 0.95 * 3500.0


def test_the_c0_aware_ladder_with_a_larger_motional_capacitance() -> None:
    lad = R.crystal_ladder(6, 7.5e3, 21.4e6, 18e-15, 4e-12)
    assert _bw3(lad, 0.0) == pytest.approx(7500.0, rel=0.03)
    # every element is the traced calculator's output
    args = (U(6.0), U(7.5e3, "Hz"), U(21.4e6, "Hz"), U(18e-15, "F"), U(4e-12, "F"))
    assert R.ladder_r_end(*args).value == lad.r_end and R.ladder_r_end(*args).unit == "ohm"
    assert R.ladder_c_couple(U(6.0), U(3.0), *args[1:]).value == lad.c_couple[2]
    assert R.ladder_mesh_c(U(6.0), U(1.0), *args[1:]).value == lad.c_mesh[0]
    with pytest.raises(ValueError, match="mesh 2 carries the largest coupling reactance and sets the mesh frequency: it needs no tuning capacitor"):
        R.ladder_mesh_c(U(6.0), U(2.0), *args[1:])
    # C0 skews the response: the half-power centre sits below the mesh frequency, and the calculator finds it
    centre = R.ladder_center(*args, U(0.0, "ohm")).value
    assert -600.0 < centre - lad.f_mesh < -100.0
    lo = R.crystal_ladder_s21_db_value(6, 7.5e3, 21.4e6, 18e-15, 4e-12, 0.0, centre - 3750.0)
    hi = R.crystal_ladder_s21_db_value(6, 7.5e3, 21.4e6, 18e-15, 4e-12, 0.0, centre + 3750.0)
    assert -3.6 < lo < -2.5 and -3.6 < hi < -2.5
    # the lower skirt is the weak one (the C0 zeros lie above the passband)
    s = R.ladder_s21_db(*args, U(25.0, "ohm"), U(centre - 12.5e3, "Hz")).value - R.ladder_s21_db(*args, U(25.0, "ohm"), U(centre, "Hz")).value
    t = R.ladder_s21_db(*args, U(25.0, "ohm"), U(centre + 12.5e3, "Hz")).value - R.ladder_s21_db(*args, U(25.0, "ohm"), U(centre, "Hz")).value
    assert s > t
    with pytest.raises(ValueError, match="the crystal count n must lie in 2"):
        R.crystal_ladder(1, 7.5e3, 21.4e6, 18e-15, 4e-12)


# --------------------------------------------------------------------------- pads, matches, detector, bias


def test_pi_pad_load_line_and_the_lumped_quarter_wave() -> None:
    rsh = R.attenuator_pi_r_shunt(U(50.0, "ohm"), U(6.0, "dB")).value
    rse = R.attenuator_pi_r_series(U(50.0, "ohm"), U(6.0, "dB")).value
    assert round(rsh, 2) == 150.48 and round(rse, 2) == 37.35
    # independent check: the pad terminated in 50 ohm presents 50 ohm and passes 10^(-6/20) of the voltage
    z_load = 1 / (1 / rsh + 1 / 50)
    z_mid = rse + z_load
    z_in = 1 / (1 / rsh + 1 / z_mid)
    assert z_in == pytest.approx(50.0, rel=1e-12)
    assert z_load / z_mid == pytest.approx(10 ** (-6 / 20), rel=1e-12)
    with pytest.raises(ValueError, match="the attenuation \\(dB\\) must be positive"):
        R.pi_pad_r_shunt_ohm(50.0, 0.0)
    assert R.pa_load_line_r(U(5.0, "V"), U(0.5, "V"), U(0.5, "W")).value == 20.25
    with pytest.raises(ValueError, match="leaves no swing"):
        R.pa_load_line_ohm(5.0, 5.0, 0.5)
    l = R.quarter_wave_lumped_l(U(FC, "Hz"), U(50.0, "ohm")).value
    c = R.quarter_wave_lumped_c(U(FC, "Hz"), U(50.0, "ohm")).value
    assert round(l * 1e9, 3) == 17.780 and round(c * 1e12, 4) == 7.1121
    # shunt C, series L, shunt C at f is the lambda/4 inverter: ABCD = [[0, j Z0], [j / Z0, 0]]
    yc, zl = 1j * w(FC) * c, 1j * w(FC) * l
    a, b = 1 + zl * yc, zl
    cc, d = yc * (1 + zl * yc) + yc, 1 + zl * yc
    assert abs(a) < 1e-12 and abs(d) < 1e-12 and b == pytest.approx(50j, rel=1e-12) and cc == pytest.approx(1j / 50, rel=1e-12)


def _quad_reference(f: float) -> float:
    s = 1j * w(f)
    y = s * 1e-9 + 1 / 10e3 + 1 / (s * 125e-6 + w(450e3) * 125e-6 / 50)
    z_tank = 1 / y
    return math.degrees(cmath.phase(z_tank / (1500 + 1 / (s * 10e-12) + z_tank)))


def test_the_quadrature_phase_is_the_exact_network() -> None:
    args = lambda f: (U(f, "Hz"), U(1500.0, "ohm"), U(10e-12, "F"), U(125e-6, "H"), U(50.0), U(450e3, "Hz"), U(1e-9, "F"), U(0.0, "F"), U(10e3, "ohm"))
    for f in (447.5e3, 450e3, 452.5e3):
        assert R.quad_phase(*args(f)).value == pytest.approx(_quad_reference(f), abs=1e-9)
    # with the design's 10 pF / 1.5 kohm the phase at 450 kHz is about 78 deg, not the textbook 90, and the slope
    # over +/- 2.5 kHz is about 11 deg, not 17.4: a fixture nominal must come from here
    at = R.quad_phase_deg(450e3, 1500, 10e-12, 125e-6, 50, 450e3, 1e-9, 0.0, 10e3)
    assert 75.0 < at < 80.0
    assert 9.0 < R.quad_phase_deg(447.5e3, 1500, 10e-12, 125e-6, 50, 450e3, 1e-9, 0.0, 10e3) - at < 13.0


def test_bias_limiter_and_db_sum() -> None:
    assert R.bjt_bias_ic(U(1.7, "V"), U(0.7, "V"), U(100.0, "ohm")).value == pytest.approx(0.01, rel=1e-14)
    with pytest.raises(ValueError, match="the transistor is off"):
        R.bjt_bias_ic_a(0.6, 0.7, 100.0)
    assert R.limiter_level(U(0.6, "V"), U(1.5)).value == pytest.approx(0.9, rel=1e-15)
    s = R.db_sum(U(6.0, "dB"), U(-3.0, "dB"))
    assert s.value == 3.0 and s.unit == "dB"


# --------------------------------------------------------------------------- audio, time-out, power


def test_emphasis_first_order_terms_and_the_lossy_integrator() -> None:
    fc = R.emphasis_corner(U(750e-6, "s")).value
    assert fc == pytest.approx(1 / (2 * math.pi * 750e-6), rel=1e-15) and round(fc, 2) == 212.21
    pre = R.emphasis_db_at(U(1e3, "Hz"), U(fc, "Hz"), U(1.0)).value
    assert pre == pytest.approx(10 * math.log10(1 + (1e3 / fc) ** 2), rel=1e-14) and round(pre, 2) == 13.66
    assert R.emphasis_db(1e3, fc, -1) == -pre
    with pytest.raises(ValueError, match="the emphasis sign"):
        R.emphasis_db(1e3, fc, 0)
    assert R.highpass1_db_at(U(300.0), U(300.0)).value == pytest.approx(-10 * math.log10(2), rel=1e-14)
    assert R.lowpass1_db_at(U(3e3), U(3e3)).value == pytest.approx(-10 * math.log10(2), rel=1e-14)
    assert R.lowpass1_db(0.0, 3e3) == 0.0
    with pytest.raises(ValueError, match="the frequency must be positive"):
        R.highpass1_db(0.0, 300.0)
    # the whole pre-emphasis stage at 300 Hz is the sum of its terms (critic: the emphasis term alone is 3 dB off there)
    stage = R.db_sum_value(R.highpass1_db(300.0, 300.0), R.emphasis_db(300.0, fc, 1))
    assert stage == pytest.approx(-10 * math.log10(2) + 10 * math.log10(1 + (300 / fc) ** 2), rel=1e-14)
    tau = 0.96e-3
    ideal = -20 * math.log10(w(1e3) * tau)
    assert R.lossy_integrator_db(1e3, tau, 10e3, 1e12) == pytest.approx(ideal, abs=1e-9)
    ratio = 1e6 / 10e3
    lossy = R.lossy_integrator_db_at(U(1e3, "Hz"), U(tau, "s"), U(10e3, "ohm"), U(1e6, "ohm")).value
    assert lossy == pytest.approx(20 * math.log10(ratio) - 10 * math.log10(1 + (w(1e3) * tau * ratio) ** 2), rel=1e-14)
    assert ideal - 0.01 < lossy < ideal  # the DC-limit resistor lowers the gain at 1 kHz only slightly


def test_butterworth_pole_q_and_the_sallen_key_values_of_the_4th_order_splatter_filter() -> None:
    q1, q2 = R.butterworth_q(U(4.0), U(1.0)).value, R.butterworth_q(U(4.0), U(2.0)).value
    assert (round(q1, 4), round(q2, 4)) == (1.3066, 0.5412)
    assert q1 == pytest.approx(1 / (2 * math.sin(math.pi / 8)), rel=1e-15)
    assert R.butterworth_pair_q(2, 1) == pytest.approx(1 / math.sqrt(2), rel=1e-15)
    with pytest.raises(ValueError, match="the pole-pair index k must lie in 1..2"):
        R.butterworth_pair_q(5, 3)  # n = 5: two pairs and a real pole
    # design 2.1 (4B): Q 0.5412 -> 5.742 / 4.901 nF, Q 1.3066 -> 13.864 / 2.030 nF with R 10 kohm at 3 kHz
    for q, (c1_nf, c2_nf) in ((0.5412, (5.742, 4.901)), (1.3066, (13.864, 2.030))):
        c1 = R.sallen_key_c1(U(q), U(3e3, "Hz"), U(1e4, "ohm")).value
        c2 = R.sallen_key_c2(U(q), U(3e3, "Hz"), U(1e4, "ohm")).value
        # (the design rounded 13.8631 nF to 13.864)
        assert c1 * 1e9 == pytest.approx(c1_nf, abs=0.0015) and c2 * 1e9 == pytest.approx(c2_nf, abs=0.0015)
        # the unity-gain equal-R section realises w0 = 1 / (R sqrt(C1 C2)) and Q = sqrt(C1 / C2) / 2
        assert 1 / (1e4 * math.sqrt(c1 * c2)) == pytest.approx(w(3e3), rel=1e-12)
        assert math.sqrt(c1 / c2) / 2 == pytest.approx(q, rel=1e-12)


def test_the_time_out_rail_budget_and_headroom() -> None:
    t = R.tot_period(U(2.3), U(100e3, "ohm"), U(10e-9, "F"))
    assert t.value == pytest.approx(8192 * 2.3 * 100e3 * 10e-9, rel=1e-15) and t.unit == "s"
    # design 6 risk 10 (3A): RX_5V 86-161 mA against LP38693DT-5.0's 500 mA; an LP2985-5.0 (150 mA) would be over
    assert R.power_rail_budget(U(0.161, "A"), U(0.5, "A")).value == pytest.approx(0.339, rel=1e-12)
    assert R.rail_budget_a(0.161, 0.150) == pytest.approx(-0.011, rel=1e-9)  # a number, not a refusal: the check FAILs on it
    # LM1117-5.0 at the 6.4 V cut-off: 441 mA, 170 mohm, 1.2 V dropout -> +0.125 V; at 6.0 V it is 0.275 V short
    head = R.regulator_headroom(U(6.4, "V"), U(5.0, "V"), U(1.2, "V"), U(0.441, "A"), U(0.17, "ohm"))
    assert head.value == pytest.approx(6.4 - 0.441 * 0.17 - 1.2 - 5.0, rel=1e-12) and round(head.value, 3) == 0.125 and head.unit == "V"
    assert round(R.regulator_headroom_v(6.0, 5.0, 1.2, 0.441, 0.17), 3) == -0.275
    with pytest.raises(ValueError, match="the rail's load current must not be negative"):
        R.rail_budget_a(-0.1, 0.5)


# --------------------------------------------------------------------------- the recompute re-derives radio values


def _radio_ir() -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="radio", name="radio"))
    p = ir.parameters
    p["carrier_frequency"] = U(FC, "Hz")
    p["rf.if1"] = U(IF1, "Hz")
    p["rf.lo1_side"] = U(-1.0)
    p["rf.lo1"] = R.superhet_lo(p["carrier_frequency"], p["rf.if1"], p["rf.lo1_side"], ("carrier_frequency", "rf.if1", "rf.lo1_side"))
    p["rf.image"] = R.superhet_image(p["carrier_frequency"], p["rf.lo1"], ("carrier_frequency", "rf.lo1"))
    p["bpf.n"] = U(3.0)
    p["bpf.bw"] = U(20e6, "Hz")
    p["bpf.l"] = U(4.7e-9, "H")
    p["z0"] = U(50.0, "ohm")
    p["model.l_q"] = U(QU)
    p["bpf.rej_image"] = R.top_c_rel_s21_db(p["bpf.n"], p["carrier_frequency"], p["bpf.bw"], p["bpf.l"], p["z0"], p["z0"], p["model.l_q"], p["rf.image"], p["carrier_frequency"],
                                             ("bpf.n", "carrier_frequency", "bpf.bw", "bpf.l", "z0", "z0", "model.l_q", "rf.image", "carrier_frequency"))
    p["bpf.c_shunt_2"] = R.top_c_c_shunt(p["bpf.n"], U(2.0), p["carrier_frequency"], p["bpf.bw"], p["bpf.l"], p["z0"], p["z0"], ("bpf.n", "bpf.i2", "carrier_frequency", "bpf.bw", "bpf.l", "z0", "z0"))
    p["bpf.i2"] = U(2.0)
    return ir


def test_the_recompute_re_derives_radio_values_and_catches_a_tampered_one() -> None:
    ir = _radio_ir()
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.PASS, res.message
    assert res.details["parameters"]["bpf.rej_image"]["recomputed"] == pytest.approx(-33.57, abs=0.005)
    stored = ir.parameters["bpf.c_shunt_2"]
    ir.parameters["bpf.c_shunt_2"] = derived(stored.value * (1 + 1e-6), tool=stored.provenance.tool, inputs=dict(stored.provenance.inputs), unit="F", tool_version=CALC_VERSION)
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.FAIL and "bpf.c_shunt_2" in res.message
    # a sign role that is not -1 / +1 is refused by the calculator: NOT_VERIFIED, never a guessed side
    ir = _radio_ir()
    ir.parameters["rf.lo1_side"] = U(0.0)
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED and "must be -1 or +1" in res.message
    # a unit that does not fit its role (an inductance given in F) is NOT_VERIFIED
    ir = _radio_ir()
    ir.parameters["bpf.l"] = U(4.7e-9, "F")
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED and "carries unit 'F'" in res.message


def test_absurd_inputs_are_refused_with_a_sentence_never_an_overflow_error() -> None:
    """A float ** raises OverflowError where a product gives inf: every radio calculator refuses (ValueError with a sentence), and the
    recompute of an edited value is NOT_VERIFIED 'refused the inputs' - the CALCULATION stage no longer aborts on it."""
    for call, match in (
        (lambda: R.varactor_c_f(1e-12, 0.7, 5000, 10), "calc.rf.varactor.c_at_bias underflows"),
        (lambda: R.double_tuned_rejection_db(1e80, 1, 2), "double_tuned.rejection overflows"),
        (lambda: R.double_tuned_rejection(U(1e80), U(1.0, "Hz"), U(2.0, "Hz")), "overflows"),
        (lambda: R.crystal_ladder(6, 1000, 1e160, 6e-15, 4e-12), "motional inductance L_m .* underflows"),
        (lambda: R.pa_load_line_ohm(1e200, 0.5, 1), "calc.rf.pa.load_line_r overflows"),
        (lambda: R.top_c_c_couple(U(2), U(1), U(1e160, "Hz"), U(1e150, "Hz"), U(1e-9, "H")), "c_couple underflows"),
    ):
        with pytest.raises(ValueError, match=match):
            call()
    assert R.varactor_c_f(20e-12, 0.7, 0.5, 2.0) == pytest.approx(20e-12 / (1 + 2.0 / 0.7) ** 0.5, rel=1e-15)  # the same law, without **
    # an edited (absurd) supply under a derived load line: the recompute refuses it, it does not raise
    ir = CircuitIR(project=ProjectMeta(id="pa", name="pa"))
    p = ir.parameters
    p["pa.v_cc"], p["pa.v_sat"], p["pa.p"] = U(7.4, "V"), U(0.5, "V"), U(0.5, "W")
    p["pa.r_load"] = R.pa_load_line_r(p["pa.v_cc"], p["pa.v_sat"], p["pa.p"], ("pa.v_cc", "pa.v_sat", "pa.p"))
    assert recompute_parameters(ir).status is ValidationStatus.PASS
    p["pa.v_cc"] = U(1e200, "V")
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED and "refused the inputs" in res.message and "overflows" in res.message, res.message
    # a varactor grading coefficient of 5000 likewise
    ir = CircuitIR(project=ProjectMeta(id="var", name="var"))
    p = ir.parameters
    p["model.cjo"], p["model.vj"], p["model.m"], p["pm.v_bias"] = U(20e-12, "F"), U(0.7, "V"), U(0.5), U(2.0, "V")
    p["pm.c_var"] = R.varactor_c_at_bias(p["model.cjo"], p["model.vj"], p["model.m"], p["pm.v_bias"], ("model.cjo", "model.vj", "model.m", "pm.v_bias"))
    assert recompute_parameters(ir).status is ValidationStatus.PASS
    p["model.m"] = U(5000.0)
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED and "refused the inputs" in res.message, res.message


# --------------------------------------------------------------------------- ngspice cross-checks (skipped without ngspice)

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")


def _n(x: float) -> str:
    return format_spice_number(x)


def _run_ac(tmp_path: Path, name: str, lines: list[str], f_lo: float, f_hi: float, points: int):
    deck = tmp_path / f"{name}.cir"
    deck.write_text("\n".join([name, *lines, ".end"]) + "\n", encoding="utf-8", newline="\n")
    res = runner.run(deck, SpiceAnalysis.AC, tmp_path / name, f"ac lin {points} {_n(f_lo)} {_n(f_hi)}")
    assert res.succeeded, res.errors
    return res


def _top_c_deck(net: R.TopCNetwork, q_u: float) -> list[str]:
    rq = w(net.f0) * net.l / q_u
    lines = ["VS src 0 DC 0 AC 1", f"RS src a {_n(net.r_source)}", f"CIN a r1 {_n(net.c_tap_source)}"]
    for i in range(1, net.n + 1):
        lines += [f"L{i} r{i} x{i} {_n(net.l)}", f"RQ{i} x{i} 0 {_n(rq)}", f"C{i} r{i} 0 {_n(net.c_shunt[i - 1])}"]
        if i < net.n:
            lines.append(f"CK{i} r{i} r{i + 1} {_n(net.c_couple[i - 1])}")
    return [*lines, f"COUT r{net.n} out {_n(net.c_tap_load)}", f"RL out 0 {_n(net.r_load)}"]


@needs_ngspice
def test_ngspice_measures_the_top_c_networks_as_calculated(tmp_path: Path) -> None:
    cases = [(2, *_tank(6 * FT)), (3, FC, 20e6, 4.7e-9)]
    for n, f0, bw, l in cases:
        net = R.top_c_network(n, f0, bw, l, 50, 50)
        for f in (f0, f0 - FT, f0 + FT):
            res = _run_ac(tmp_path, f"topc{n}_{int(f)}", _top_c_deck(net, QU), f, f * (1 + 1e-9), 2)
            measured = 20 * math.log10(2 * res.vectors["out"][0])
            assert measured == pytest.approx(R.top_c_s21_db_value(n, f0, bw, l, 50, 50, QU, f), abs=0.02)


@needs_ngspice
def test_ngspice_measures_the_pm_tank_phases_of_the_pm_n12_deck(tmp_path: Path) -> None:
    # kr447/decided/pm_n12.cir: 50 ohm port, 1 nF DC block, R_s 1846.554, L 606.98 nH + w L / 40, 10 nF bypass, 10 kohm feed,
    # C_fixed 19.8165 pF, DVAR, 1 Gohm probe
    l = 1 / (w(FT) ** 2 * 30e-12)
    c_fixed = 30e-12 - R.varactor_c_f(20e-12, 0.7, 0.5, 2.0)
    r_s = R.pm_source_r_ohm(FT, l, 10.0, QU, 50.0)
    for v, pin in ((1.44, -21.43), (2.0, 1.39), (2.56, 18.21)):
        lines = [".model DVAR D (CJO=20p VJ=0.7 M=0.5)", "VS s 0 DC 0 AC 1", "RP s a 50", "CD a b 1n", f"RSR b t {_n(r_s)}", f"LT t x {_n(l)}",
                 f"RLS x lb {_n(w(FT) * l / QU)}", "CBP lb 0 10n", "RB vb lb 10k", f"VB vb 0 DC {_n(v)}", f"CF t 0 {_n(c_fixed)}", "D1 0 t DVAR", "RPR t 0 1e9"]
        res = _run_ac(tmp_path, f"pm_{int(v * 100)}", lines, FT, FT * (1 + 1e-9), 2)
        measured = res.vectors["t.phase_deg"][0]
        calc = R.pm_tank_phase_deg(FT, FT, l, QU, c_fixed, 0.0, R.varactor_c_f(20e-12, 0.7, 0.5, v), 50.0, 1e-9, r_s, 10e-9, 10e3)
        assert measured == pytest.approx(calc, abs=0.05) and measured == pytest.approx(pin, abs=0.05)


def _ladder_deck(lad: R.CrystalLadder, r_m: float) -> list[str]:
    lines = ["VS src 0 DC 0 AC 1", f"RS src j0 {_n(lad.r_end)}"]
    node = "j0"
    for i in range(1, lad.n + 1):
        a = node
        if lad.c_mesh[i - 1] is not None:
            lines.append(f"CT{i} {a} t{i} {_n(lad.c_mesh[i - 1])}")
            a = f"t{i}"
        b = f"j{i}"
        lines += [f"LM{i} {a} x{i} {_n(lad.l_m)}", f"CM{i} x{i} y{i} {_n(lad.c_m)}", f"RM{i} y{i} {b} {_n(r_m)}"]
        if lad.c0 > 0:
            lines.append(f"CO{i} {a} {b} {_n(lad.c0)}")
        if i < lad.n:
            lines.append(f"CC{i} {b} 0 {_n(lad.c_couple[i - 1])}")
        node = b
    # the crystals' terminals float at DC: 1 Tohm to ground gives the operating point a path. Only on the terminals (t, j):
    # measured on ngspice-42, the same resistor on a crystal's internal node between L_m and C_m (whose admittances cancel
    # near series resonance) moves the AC result by 0.04-0.25 dB - roundoff, the physical effect is 1e-6 - while on the
    # terminals even 1 Gohm leaves it exact
    floating = sorted({tok for ln in lines for tok in ln.split()[1:3] if tok[0] in "tj" and tok != "j0"})
    return [*lines, f"RL {node} 0 {_n(lad.r_end)}", *[f"RDC{k} {nd} 0 1e12" for k, nd in enumerate(floating)]]


@needs_ngspice
def test_ngspice_measures_the_c0_aware_ladder_bandwidth_and_centre(tmp_path: Path) -> None:
    # (a motional R_m of 0 would put a zero-ohm arm between two DC-floating nodes: ngspice's operating point is then singular)
    for n, bw, c_m, r_m in ((6, 7.5e3, 18e-15, 25.0), (6, 3.5e3, 6e-15, 25.0), (4, 7.5e3, 12e-15, 10.0)):
        lad = R.crystal_ladder(n, bw, 21.4e6, c_m, 4e-12)
        f_lo, f_hi, pts = lad.f_mesh - 15e3, lad.f_mesh + 15e3, 3001  # 10 Hz steps
        res = _run_ac(tmp_path, f"ladder{n}_{int(bw)}_{int(c_m * 1e18)}", _ladder_deck(lad, r_m), f_lo, f_hi, pts)
        freqs = res.vectors["frequency"]
        db = [20 * math.log10(2 * v) for v in res.vectors[f"j{n}"]]
        # the exact response of the calculator is the simulated one, passband, skirts and the C0 zeros alike
        for k in range(0, pts, 50):
            assert db[k] == pytest.approx(R.crystal_ladder_s21_db_value(n, bw, 21.4e6, c_m, 4e-12, r_m, freqs[k]), abs=0.01), freqs[k] - lad.f_mesh
        level = max(db) - 10 * math.log10(2)
        edges = [freqs[k] + (freqs[k + 1] - freqs[k]) * (level - db[k]) / (db[k + 1] - db[k]) for k in range(pts - 1) if (db[k] - level) * (db[k + 1] - level) < 0]
        assert len(edges) == 2  # one half-power edge on each side of a Butterworth passband
        assert sum(edges) / 2 == pytest.approx(R.crystal_ladder_center_hz(n, bw, 21.4e6, c_m, 4e-12, r_m), abs=2.0)


@needs_ngspice
def test_ngspice_measures_the_quadrature_phase(tmp_path: Path) -> None:
    lines = ["VS s 0 DC 0 AC 1", "RS s a 1500", "CQ a q 10p", "LQ q x 125u", f"RLQ x 0 {_n(w(450e3) * 125e-6 / 50)}", "CP q 0 1n", "RP q 0 10k"]
    for f in (447.5e3, 450e3, 452.5e3):
        res = _run_ac(tmp_path, f"quad_{int(f)}", lines, f, f * (1 + 1e-9), 2)
        assert res.vectors["q.phase_deg"][0] == pytest.approx(R.quad_phase_deg(f, 1500, 10e-12, 125e-6, 50, 450e3, 1e-9, 0.0, 10e3), abs=0.01)
