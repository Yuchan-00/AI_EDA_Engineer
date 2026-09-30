"""The window measurements of the RF / audio reductions: RMS, harmonic level in dBc, AM depth - and the stage's reduction of each.

Nothing here needs ngspice: :mod:`ai_eda.tools.spice.measure` is stdlib
arithmetic on sample lists and :func:`reduce_expectation` is fed hand-built
:class:`SpiceResult` objects. What is proved:

* ``window_rms`` is the exact RMS of the piecewise-linear interpolant: a
  triangle wave sampled at its vertices on a non-uniform grid gives
  A / sqrt(3) to rounding, a ramp over a window whose edges fall between
  samples gives the closed form, a finely sampled sine over whole periods
  1 / sqrt(2);
* ``harmonic_level`` reads a -20 dBc third harmonic within 0.01 dB, a
  square wave's third harmonic at 20 log10(1/3), flags a harmonic below the
  engine's resolution, and refuses a grid coarser than 1 / (20 k f0), a
  window that is not whole periods (or only one), a waveform without a
  fundamental and one that does not repeat at f0 (detuned by 1 % / 2.5 %, an
  unsettled start); its bracket holds the grid attenuation;
* ``am_depth``'s bracket holds the true depth at ratio 20 (50 % and 50.70 %),
  and ``window_rms`` with ``f_max`` refuses 8 samples per period;
* ``am_depth`` reads a synthetic 50 % AM wave (ratio 100) within 0.5
  percentage points with its audit numbers, and refuses a wave whose
  carrier is too slow (ratio 10), a window shorter than one modulation
  period, and a grid that has 16 samples per carrier period *by count* but
  a gap wider than 1 / (16 f_carrier) at a crest (the count-only guard
  would have passed it with a wrong amplitude);
* all of it is deterministic and never mutates its inputs;
* the stage's ``db_at`` / ``db_rms`` / ``rms`` / ``harmonic_dbc`` /
  ``am_depth`` reductions give the same numbers, a dB bracket for ``db_at``,
  their audit data in ``extra``, the reason when there is no number, and an
  ``unresolved`` reason for a harmonic at the engine's floor.
"""

from __future__ import annotations

import copy
import math

import pytest

from ai_eda.ir import Expectation, Reduce, user_requirement
from ai_eda.tools.spice import SpiceAnalysis, SpiceResult
from ai_eda.tools.spice.measure import (
    HARMONIC_STEP_FACTOR,
    MIN_CARRIER_TO_MOD_RATIO,
    MIN_SAMPLES_PER_CARRIER_PERIOD,
    am_depth,
    harmonic_level,
    window_rms,
)
from ai_eda.tools.spice.stage import judge, reduce_expectation
from ai_eda.ir import ValidationStatus as S
from tests.fixtures_kicad import USER


def u(value, unit=None):
    return user_requirement(value, unit)


def grid(n: int, stop: float) -> list[float]:
    return [i * stop / n for i in range(n + 1)]


def am_wave(times: list[float], va: float = 1.0, vo: float = 2.0, fm: float = 1e3, fc: float = 100e3) -> list[float]:
    """ngspice's AM(VA VO MF FC 0): VA (VO + sin(2 pi fm t)) sin(2 pi fc t); depth 1 / VO."""
    return [va * (vo + math.sin(2 * math.pi * fm * t)) * math.sin(2 * math.pi * fc * t) for t in times]


# --------------------------------------------------------------------------- RMS


def test_rms_of_a_triangle_on_a_non_uniform_grid_is_exact():
    # a +/-2 V triangle of period 1 ms sampled at its vertices plus uneven points in between: the interpolant *is* the triangle
    times, values = [], []
    for p in range(4):
        base = p * 1e-3
        for frac, v in ((0.0, -2.0), (0.1, -1.2), (0.13, -0.96), (0.5, 2.0), (0.77, -0.16), (0.9, -1.2)):
            times.append(base + frac * 1e-3)
            values.append(v)
    times.append(4e-3)
    values.append(-2.0)
    r = window_rms(times, values, 0.0, 4e-3)
    assert r.problem is None and r.value == pytest.approx(2.0 / math.sqrt(3.0), rel=1e-12)
    assert r.samples == len(times) and r.max_step == pytest.approx(0.37e-3)


def test_rms_edges_are_interpolated_between_samples():
    times = grid(10, 1.0)
    ramp = list(times)  # v = t: linear, so the interpolant is exact
    a, b = 0.23, 0.87  # neither is a sample
    r = window_rms(times, ramp, a, b)
    assert r.value == pytest.approx(math.sqrt((b ** 3 - a ** 3) / (3.0 * (b - a))), rel=1e-12)
    assert r.samples == 2 + 6  # the interpolated edges and the samples 0.3 .. 0.8


def test_rms_of_a_sampled_sine_over_whole_periods():
    times = grid(20000, 4e-3)
    sine = [math.sin(2 * math.pi * 1e3 * t) for t in times]
    r = window_rms(times, sine, 0.0, 4e-3)
    # the linear interpolant of 5000 samples per period reads 1 / sqrt(2) within 2e-7 relative
    assert r.value == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-6) and r.value < 1.0 / math.sqrt(2.0)


def test_rms_refuses_what_it_cannot_measure():
    t = grid(10, 1.0)
    assert window_rms(t, t[:-1], 0.0, 1.0).problem == "scale and vector lengths differ (11 vs 10 samples)"
    assert window_rms(t, [*t[:-1], math.nan], 0.0, 1.0).problem.startswith("vector contains non-finite samples")
    assert window_rms([0.0, 0.5, 0.4, 1.0], [0.0] * 4, 0.0, 1.0).problem == "the scale vector steps backwards (not a transient time axis)"
    assert "is not inside the saved samples [0, 1] s" in window_rms(t, t, 0.5, 1.5).problem
    assert "is not an interval" in window_rms(t, t, 0.6, 0.6).problem


# --------------------------------------------------------------------------- harmonics


def test_a_minus_20_dbc_third_harmonic_is_read_within_0_01_db():
    times = grid(20000, 4e-3)
    v = [math.sin(2 * math.pi * 1e3 * t) + 0.1 * math.sin(2 * math.pi * 3e3 * t + 0.3) for t in times]
    h = harmonic_level(times, v, 1e3, 3, 0.0, 4e-3)
    assert h.problem is None and h.dbc == pytest.approx(-20.0, abs=0.01)
    assert h.a1 == pytest.approx(1.0, rel=1e-4) and h.ak == pytest.approx(0.1, rel=1e-4) and h.periods == pytest.approx(4.0)
    assert h.step_limit == pytest.approx(1.0 / (HARMONIC_STEP_FACTOR * 3 * 1e3)) and h.max_step <= h.step_limit and not h.ak_below_floor


def test_a_square_wave_third_harmonic_and_a_harmonic_below_the_floor():
    times = sorted({*grid(4000, 4e-3), *(k * 0.5e-3 for k in range(9))})
    # ideal edges: two samples at each switching time (a breakpoint), low then high
    t2, v2 = [], []
    for t in times:
        phase = (t * 1e3) % 1.0
        if abs(phase) < 1e-9 or abs(phase - 0.5) < 1e-9 or abs(phase - 1.0) < 1e-9:
            lo_first = abs(phase - 0.5) >= 1e-9  # a rising edge at whole periods, a falling edge at half periods
            t2 += [t, t]
            v2 += [0.0, 1.0] if lo_first else [1.0, 0.0]
        else:
            t2.append(t)
            v2.append(1.0 if phase < 0.5 else 0.0)
    h3 = harmonic_level(t2, v2, 1e3, 3, 0.0, 4e-3)
    assert h3.problem is None and h3.dbc == pytest.approx(20.0 * math.log10(1.0 / 3.0), abs=1e-3)
    assert h3.a1 == pytest.approx(2.0 / math.pi, rel=1e-4)
    h2 = harmonic_level(t2, v2, 1e3, 2, 0.0, 4e-3)
    assert h2.ak_below_floor and h2.ak < 1e-9 and h2.floor == pytest.approx(1e-3 + 1e-6)  # even harmonics of a 50 % square wave vanish


def test_harmonic_refusals():
    times = grid(200, 4e-3)  # 20 us steps: too coarse for k = 3 of 1 kHz (limit 16.7 us)
    v = [math.sin(2 * math.pi * 1e3 * t) for t in times]
    h = harmonic_level(times, v, 1e3, 3, 0.0, 4e-3)
    assert h.dbc is None and h.problem.startswith("time grid too coarse for harmonic 3: largest step 2e-05 s > 1.66667e-05 s; lower the tran step")
    assert harmonic_level(times, v, 1e3, 2, 0.0, 4e-3).problem is None  # k = 2: the limit is 25 us
    fine = grid(4000, 4e-3)
    sine = [math.sin(2 * math.pi * 1e3 * t) for t in fine]
    assert "not a whole number of them" in harmonic_level(fine, sine, 1e3, 3, 0.0, 3.5e-3).problem
    flat = [1e-7 * math.sin(2 * math.pi * 1e3 * t) for t in fine]
    assert harmonic_level(fine, flat, 1e3, 3, 0.0, 4e-3).problem.startswith("no fundamental: the amplitude at f0 = 1000 Hz")
    for bad_k in (1, 2.5, True, 0):
        with pytest.raises(ValueError, match="k must be an integer >= 2"):
            harmonic_level(fine, sine, 1e3, bad_k, 0.0, 4e-3)
    with pytest.raises(ValueError, match="f0 must be a positive finite number"):
        harmonic_level(fine, sine, 0.0, 3, 0.0, 4e-3)


def test_a_detuned_fundamental_or_an_unsettled_start_is_refused_never_read():
    """Regression: a -20 dBc third harmonic of a wave at 1.01 f0 read -20.76 dBc (1.025 f0: -21.43) with no problem."""
    times = grid(40000, 4e-3)
    for f in (1010.0, 1025.0):
        v = [math.sin(2 * math.pi * f * t) + 0.1 * math.sin(2 * math.pi * 3 * f * t) for t in times]
        h = harmonic_level(times, v, 1e3, 3, 0.0, 4e-3)
        assert h.dbc is None and h.problem.startswith("the waveform does not repeat at f0 = 1000 Hz over the window"), f
        assert h.f_measured == pytest.approx(f, rel=1e-6) and f"its rising edges give {f:.9g} Hz" in h.problem
        assert h.periodicity_residual_rel > 0.02
    # an oscillator's start-up (the amplitude still growing) does not repeat either
    grow = [(1.0 - math.exp(-t / 1e-3)) * math.sin(2 * math.pi * 1e3 * t) for t in times]
    h = harmonic_level(times, grow, 1e3, 3, 0.0, 4e-3)
    assert h.dbc is None and "does not repeat at f0" in h.problem and "start the window in the steady state" in h.problem
    # a window of one period cannot show that the waveform repeats
    v = [math.sin(2 * math.pi * 1e3 * t) for t in times]
    assert "at least 2 are needed" in harmonic_level(times, v, 1e3, 3, 0.0, 1e-3).problem
    # a small detuning is measured, and the bracket it leaves contains the true -20 dBc
    v = [math.sin(2 * math.pi * 1001.0 * t) + 0.1 * math.sin(2 * math.pi * 3003.0 * t) for t in times]
    h = harmonic_level(times, v, 1e3, 3, 0.0, 4e-3)
    assert h.problem is None and h.dbc_low < -20.0 < h.dbc_high and h.detuning_bound_rel >= 1e-3
    assert judge(h.dbc, _exp(Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=0.05), None, (h.dbc_low, h.dbc_high))[0] is S.UNRESOLVED


def test_the_harmonic_bracket_holds_the_grid_attenuation():
    """On a grid at the guard the third harmonic reads low by up to the sinc^2 of its step: the bracket's upper edge covers it."""
    n = 241  # 16.6 us steps over 4 ms, not aligned with the 1 ms period
    times = [i * 4e-3 / n for i in range(n + 1)]
    v = [math.sin(2 * math.pi * 1e3 * t) + 0.1 * math.sin(2 * math.pi * 3e3 * t) for t in times]
    h = harmonic_level(times, v, 1e3, 3, 0.0, 4e-3)
    assert h.problem is None and h.dbc < -20.03 and h.dbc_low <= h.dbc and h.dbc_high >= -20.0
    x = math.pi * 3e3 * h.max_step
    assert h.grid_attenuation_k_db == pytest.approx(-20 * math.log10((math.sin(x) / x) ** 2), rel=1e-9)


# --------------------------------------------------------------------------- AM depth


def test_am_depth_of_a_synthetic_50_percent_wave():
    times = grid(400_000, 2e-3)  # 5 ns steps, 2000 per carrier period
    d = am_depth(times, am_wave(times), 100e3, 1e3, 0.0, 2e-3)
    assert d.problem is None and d.depth == pytest.approx(50.0, abs=0.5)
    assert d.a_max == pytest.approx(3.0, rel=1e-3) and d.a_min == pytest.approx(1.0, rel=1e-3) and d.cycles == 200
    assert d.min_samples_per_cycle >= MIN_SAMPLES_PER_CARRIER_PERIOD and d.max_step <= d.step_limit
    # the reported bounds: the crest read low by 1 - cos(pi fc h), the envelope smear at ratio 100
    assert d.crest_bias_bound_rel == pytest.approx(1.0 - math.cos(math.pi * 100e3 * 5e-9), rel=1e-6)
    assert d.envelope_bias_bound_rel == pytest.approx(1.0 - math.cos(math.pi / 100) * math.cos(math.pi / 200), rel=1e-12)
    # the envelope smear is the only visible bias here: A_max low, A_min high by about VA x the bound
    assert 3.0 - d.a_max <= d.envelope_bias_bound_rel + 1e-6 and d.a_min - 1.0 <= d.envelope_bias_bound_rel + 1e-6
    # a DC offset does not change the per-period amplitude
    shifted = [x + 0.7 for x in am_wave(times)]
    assert am_depth(times, shifted, 100e3, 1e3, 0.0, 2e-3).depth == pytest.approx(d.depth, abs=1e-9)


def test_the_am_depth_bracket_holds_the_true_depth_at_ratio_20():
    """Regression: the envelope smear reads a depth low by up to e x depth; the verdict must be judged on the bracket, not the number."""
    times = grid(400_000, 2e-3)
    for vo in (2.0, 1.9724):  # true depth 50 % and 50.70 %
        wave = am_wave(times, vo=vo, fc=20e3)
        d = am_depth(times, wave, 20e3, 1e3, 0.0, 2e-3)
        true = 100.0 / vo
        assert d.problem is None and d.depth < true - 0.5 and d.depth_low <= d.depth and d.depth_low <= true <= d.depth_high, vo
        status = judge(d.depth, _exp(Reduce.AM_DEPTH, 50.0, "percent", tol_abs=0.2), None, (d.depth_low, d.depth_high))[0]
        assert status is S.UNRESOLVED  # formerly FAIL for the correct 50 % wave and PASS for the 50.70 % one
    # a bracket entirely outside the band is still a FAIL, entirely inside a PASS
    assert judge(40.0, _exp(Reduce.AM_DEPTH, 50.0, "percent", tol_abs=0.2), None, (39.9, 40.5))[0] is S.FAIL
    assert judge(50.0, _exp(Reduce.AM_DEPTH, 50.0, "percent", tol_abs=0.2), None, (49.95, 50.1))[0] is S.PASS
    assert judge(-30.0, _exp(Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=1.0), None, (None, -29.0))[0] is S.FAIL  # unbounded below, all below the band
    assert judge(-20.0, _exp(Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=1.0), None, (None, -19.5))[0] is S.UNRESOLVED  # never PASS unbounded


def test_an_rms_grid_coarser_than_f_max_allows_gives_no_number():
    """Regression: a sine at 8 samples per period read 0.672 of its peak (true 0.7071) with no problem."""
    times = grid(32, 4e-3)  # 8 samples per 1 kHz period
    v = [math.sin(2 * math.pi * 1e3 * t) for t in times]
    assert window_rms(times, v, 0.0, 4e-3).value == pytest.approx(math.sqrt((2 + math.cos(2 * math.pi / 8)) / 3) / math.sqrt(2), rel=1e-9)  # no f_max: no guard
    r = window_rms(times, v, 0.0, 4e-3, 1e3)
    assert r.value is None and r.problem.startswith("time grid too coarse for an RMS up to f_max = 1000 Hz: largest step 0.000125 s > 5e-05 s")
    assert r.bias_bound_rel == pytest.approx(1 - math.sqrt((2 + math.cos(2 * math.pi / 8)) / 3), rel=1e-12)
    fine = grid(400, 4e-3)  # 100 samples per period: inside the guard, the bracket [rms, rms / (1 - b)] holds 1 / sqrt(2)
    r = window_rms(fine, [math.sin(2 * math.pi * 1e3 * t) for t in fine], 0.0, 4e-3, 1e3)
    assert r.problem is None and r.value <= 1 / math.sqrt(2) <= r.value / (1 - r.bias_bound_rel) * (1 + 1e-12)
    with pytest.raises(ValueError, match="f_max must be a positive finite number"):
        window_rms(fine, fine, 0.0, 4e-3, 0.0)


def test_am_depth_refusals():
    slow = grid(20_000, 2e-3)
    ratio10 = am_wave(slow, fc=10e3)
    d = am_depth(slow, ratio10, 10e3, 1e3, 0.0, 2e-3)
    assert d.depth is None and d.problem.startswith(f"carrier too slow for the envelope: f_carrier / f_mod = 10 < {MIN_CARRIER_TO_MOD_RATIO}")
    times = grid(400_000, 2e-3)
    wave = am_wave(times)
    assert am_depth(times, wave, 100e3, 1e3, 0.0, 0.5e-3).problem.startswith("window shorter than one modulation period")
    with pytest.raises(ValueError, match="must exceed f_mod"):
        am_depth(times, wave, 1e3, 1e3, 0.0, 2e-3)


def test_a_gap_at_a_crest_is_refused_although_every_period_has_16_samples():
    """20 samples per carrier period by count, but one period's samples cluster away from its crest: the step guard refuses it."""
    fc, fm = 100e3, 1e3
    period = 1.0 / fc
    times: list[float] = []
    for i in range(200):
        base = i * period
        if i == 50:
            # 20 samples squeezed into the first 30 % of the period, then one gap across the positive and negative crests
            times += [base + k * 0.3 * period / 20 for k in range(20)]
        else:
            times += [base + k * period / 20 for k in range(20)]
    times.append(200 * period)
    wave = am_wave(times, fm=fm, fc=fc)
    counts = [sum(1 for t in times if i * period <= t < (i + 1) * period) for i in range(200)]
    assert min(counts) >= MIN_SAMPLES_PER_CARRIER_PERIOD  # a count-only guard would pass this grid
    d = am_depth(times, wave, fc, fm, 0.0, 200 * period)
    assert d.depth is None and d.problem.startswith("time grid too coarse for the carrier: largest step")
    assert "in the carrier period starting at 0.0005 s" in d.problem


def test_measurements_are_deterministic_and_leave_their_inputs_alone():
    times = grid(40_000, 2e-3)
    wave = am_wave(times, fc=50e3)
    t_copy, w_copy = copy.deepcopy(times), copy.deepcopy(wave)
    a = (window_rms(times, wave, 0.0, 2e-3), harmonic_level(times, wave, 1e3, 3, 0.0, 2e-3), am_depth(times, wave, 50e3, 1e3, 0.0, 2e-3))
    b = (window_rms(times, wave, 0.0, 2e-3), harmonic_level(times, wave, 1e3, 3, 0.0, 2e-3), am_depth(times, wave, 50e3, 1e3, 0.0, 2e-3))
    assert a == b and times == t_copy and wave == w_copy


# --------------------------------------------------------------------------- the stage's reductions


def _tran(times: list[float], **vectors: list[float]) -> SpiceResult:
    return SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.TRAN, command="tran 1u 4m",
                       vectors={"time": times, **vectors}, vector_types={"time": "time"}, scale="time", n_points=len(times), succeeded=True)


def _exp(reduce: Reduce, nominal: float, unit: str | None, *, tol_abs: float | None = None, reference: str | None = None, at=None, **params) -> Expectation:
    return Expectation(id="e", analysis_id="a", vector="v(OUT)", reduce=reduce, nominal=u(nominal, unit), tol_abs=None if tol_abs is None else u(tol_abs, unit),
                       reference_vector=reference, at=at, params={k: u(v) for k, v in params.items()}, provenance=USER)


def test_stage_window_reductions():
    times = grid(20000, 4e-3)
    out = [0.5 * math.sin(2 * math.pi * 1e3 * t) + 0.05 * math.sin(2 * math.pi * 3e3 * t) for t in times]
    inp = [math.sin(2 * math.pi * 1e3 * t) for t in times]
    res = _tran(times, out=out, inp=inp)
    rms = reduce_expectation(res, _exp(Reduce.RMS, 0.355, "V", tol_abs=0.01, t_start=0.0, t_stop=4e-3, f_max=3e3), "out")
    assert rms.problem is None and rms.measured == pytest.approx(math.sqrt(0.5 ** 2 / 2 + 0.05 ** 2 / 2), rel=1e-6)
    assert rms.extra["t_start"] == 0.0 and rms.extra["samples"] == len(times)
    db = reduce_expectation(res, _exp(Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, reference="v(INP)", t_start=0.0, t_stop=4e-3, f_max=3e3), "out", "inp")
    assert db.measured == pytest.approx(20 * math.log10(rms.measured / (1 / math.sqrt(2))), abs=1e-5) and db.extra["reference"] == "v(INP)"
    db_ref = reduce_expectation(res, _exp(Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, t_start=0.0, t_stop=4e-3, ref=1.0, f_max=3e3), "out")
    assert db_ref.measured == pytest.approx(20 * math.log10(rms.measured), abs=1e-9) and db_ref.extra["reference"] == "ref 1"
    zero = _tran(times, out=out, inp=[0.0] * len(times))
    assert reduce_expectation(zero, _exp(Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, reference="v(INP)", t_start=0.0, t_stop=4e-3, f_max=3e3), "out", "inp").problem == "reference RMS is zero: no level in dB"
    gone = reduce_expectation(_tran(times, out=out), _exp(Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, reference="v(INP)", t_start=0.0, t_stop=4e-3, f_max=3e3), "out", "inp")
    assert gone.problem.startswith("reference vector not produced: 'inp' is not in the tran plot")
    h = reduce_expectation(res, _exp(Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=0.05, f0=1e3, k=3, t_start=0.0, t_stop=4e-3), "out")
    assert h.measured == pytest.approx(-20.0, abs=0.01) and h.unresolved is None and h.extra["k"] == 3 and h.extra["a1"] == pytest.approx(0.5, rel=1e-4)
    h2 = reduce_expectation(res, _exp(Reduce.HARMONIC_DBC, -120.0, "dBc", tol_abs=100.0, f0=1e3, k=2, t_start=0.0, t_stop=4e-3), "out")
    assert h2.measured is not None and h2.measured < -100.0 and h2.extra["ak_below_floor"] and "is at or below the engine's resolution" in h2.unresolved
    coarse = reduce_expectation(res, _exp(Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=0.05, f0=1e3, k=300, t_start=0.0, t_stop=4e-3), "out")
    assert coarse.measured is None and coarse.problem.startswith("time grid too coarse for harmonic 300") and coarse.extra["step_limit"] == pytest.approx(1 / (20 * 300 * 1e3))
    am_t = grid(200_000, 2e-3)
    am = reduce_expectation(_tran(am_t, out=am_wave(am_t)), _exp(Reduce.AM_DEPTH, 50.0, "percent", tol_abs=0.5, f_carrier=100e3, f_mod=1e3, t_start=0.0, t_stop=2e-3), "out")
    assert am.measured == pytest.approx(50.0, abs=0.5) and am.extra["cycles"] == 200 and am.extra["depth_percent"] == am.measured
    ac = SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.AC, command="ac lin 3 1 3",
                     vectors={"frequency": [1.0, 2.0, 3.0], "out": [1.0, 0.5, 0.25]}, scale="frequency", n_points=3, succeeded=True)
    wrong = reduce_expectation(ac, _exp(Reduce.RMS, 1.0, "V", tol_abs=0.1, t_start=0.0, t_stop=1.0, f_max=1.0), "out")
    assert wrong.problem == "reduce=rms needs a tran result, this is ac"
    no_fmax = reduce_expectation(res, _exp(Reduce.RMS, 0.355, "V", tol_abs=0.01, t_start=0.0, t_stop=4e-3), "out")
    assert no_fmax.measured is None and no_fmax.problem.startswith("reduce=rms needs params['f_max']")


def test_stage_db_at_reads_both_magnitudes_between_the_same_two_samples():
    freqs = [100.0, 1000.0, 10000.0]
    ac = SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.AC, command="ac dec 1 100 10k",
                     vectors={"frequency": freqs, "out": [0.99, 0.70710678, 0.0995], "inp": [1.0, 1.0, 1.0]}, scale="frequency", n_points=3, succeeded=True)
    exact = reduce_expectation(ac, _exp(Reduce.DB_AT, -3.0103, "dB", tol_abs=0.01, reference="v(INP)", at=u(1000.0, "Hz")), "out", "inp")
    assert exact.problem is None and exact.interpolation.exact and exact.measured == pytest.approx(20 * math.log10(0.70710678), abs=1e-9)
    assert exact.extra["magnitude"] == 0.70710678 and exact.extra["reference_magnitude"] == 1.0
    between = _exp(Reduce.DB_AT, -10.0, "dB", tol_abs=0.01, at=u(3000.0, "Hz"), ref=1.0)
    r = reduce_expectation(ac, between, "out")
    assert not r.interpolation.exact and r.interpolation.y0 == pytest.approx(20 * math.log10(0.70710678)) and r.interpolation.y1 == pytest.approx(20 * math.log10(0.0995))
    # the dB bracket straddles far more than the tolerance: the grid cannot judge it
    status, _limit, _dev = judge(r.measured, between, r.interpolation)
    assert status is S.UNRESOLVED
    zero = SpiceResult(**{**ac.model_dump(), "vectors": {"frequency": freqs, "out": [0.99, 0.0, 0.0995], "inp": [1.0, 1.0, 1.0]}})
    assert reduce_expectation(zero, _exp(Reduce.DB_AT, -3.0, "dB", tol_abs=0.01, reference="v(INP)", at=u(1000.0, "Hz")), "out", "inp").problem.startswith(
        "a magnitude of 0 (or not finite) has no level in dB")
