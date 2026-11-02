"""``Reduce.FREQUENCY``: the pure rising-edge detector and the stage's reduction of a synthetic transient result.

Nothing here needs ngspice: :func:`ai_eda.tools.spice.measure.rising_edge_frequency`
is stdlib arithmetic on sample lists, and :func:`reduce_expectation` is fed a
hand-built :class:`SpiceResult` (``tran``, ``scale="time"``). The ngspice-gated
proof that a real astable multivibrator's frequency is measured is in
``tests/test_simulation_stage.py``.
"""

from __future__ import annotations

import math

import pytest

from ai_eda.ir import Expectation, Reduce, user_requirement
from ai_eda.tools.spice import EdgeFrequency, SpiceAnalysis, SpiceResult, rising_edge_frequency
from ai_eda.tools.spice.measure import HIGH_FRACTION, LOW_FRACTION, MID_FRACTION, MIN_EDGES
from ai_eda.tools.spice.stage import reduce_expectation, reduce_result
from tests.fixtures_kicad import USER


def _times(n: int, dt: float = 1e-5) -> list[float]:
    return [i * dt for i in range(n)]


def _square(times: list[float], f: float, lo: float = 0.0, hi: float = 5.0, duty: float = 0.5) -> list[float]:
    return [hi if (t * f) % 1.0 < duty else lo for t in times]


# --------------------------------------------------------------------------- the detector


def test_ideal_square_wave_frequency_and_thresholds():
    t = _times(3000)  # 30 ms window
    r = rising_edge_frequency(t, _square(t, 1000.0))
    assert isinstance(r, EdgeFrequency) and r.problem is None
    assert r.frequency == pytest.approx(1000.0, rel=1e-9)
    assert len(r.edges) == 29 and all(b > a for a, b in zip(r.edges, r.edges[1:]))
    assert (r.vmin, r.vmax) == (0.0, 5.0)
    assert (r.low, r.mid, r.high) == (LOW_FRACTION * 5.0, MID_FRACTION * 5.0, HIGH_FRACTION * 5.0)
    # the mean over the window: (N - 1) periods between the first and the last edge
    assert r.frequency == pytest.approx((len(r.edges) - 1) / (r.edges[-1] - r.edges[0]))


def test_sine_wave_crossings_are_interpolated_at_the_mid_level():
    t = _times(2000, 1e-6)  # 2 ms
    r = rising_edge_frequency(t, [2.5 + 2.5 * math.sin(2 * math.pi * 5000.0 * x) for x in t])
    assert r.problem is None and r.frequency == pytest.approx(5000.0, rel=1e-3)
    # a rising zero crossing of the sine sits at k / 5000 s; the linear interpolation lands within a sample of it
    assert all(abs(e * 5000.0 - round(e * 5000.0)) < 5000.0 * 1e-6 for e in r.edges)


def test_dc_and_flat_waveforms_have_no_edges():
    t = _times(100)
    r = rising_edge_frequency(t, [3.3] * 100)
    assert r.frequency is None and r.edges == [] and "swing 0" in r.problem and r.vmin == r.vmax == 3.3
    ramp = rising_edge_frequency(t, [0.05 * i for i in range(100)])  # monotonic: crosses the mid level once
    assert ramp.frequency is None and len(ramp.edges) == 1 and "1 rising edge(s)" in ramp.problem and f"{MIN_EDGES} are needed" in ramp.problem


def test_two_edges_are_not_a_frequency():
    t = _times(150)
    v = _square(t, 1000.0)  # 1.5 ms: rising edges at 0 (never armed: starts high), 1.0 ms -> only one counted ...
    r = rising_edge_frequency(t, v)
    assert r.frequency is None and len(r.edges) == 1
    v = _square(t, 2000.0)  # 1.5 ms of 2 kHz: edges at 0.5 and 1.0 ms -> two, still not enough
    r = rising_edge_frequency(t, v)
    assert r.frequency is None and len(r.edges) == 2 and "2 rising edge(s)" in r.problem
    r = rising_edge_frequency(t[:110], _square(t[:110], 3000.0))  # 1.1 ms of 3 kHz: three edges (1/3, 2/3, 1 ms) is the minimum
    assert len(r.edges) == 3 and r.frequency == pytest.approx(3000.0, rel=2e-2)  # a 33.3-sample period quantises the edges


def test_chatter_around_the_mid_level_is_counted_once():
    # a 1 kHz square wave whose every rising edge dithers across the mid level several times before settling high:
    # without hysteresis each dither would be another "rising edge"
    t = _times(3000)
    v = _square(t, 1000.0)
    n_chatter = 0
    for i in range(1, len(v)):
        if v[i - 1] == 0.0 and v[i] == 5.0:
            for k, level in enumerate((2.4, 2.7, 2.3, 2.8, 2.45, 3.0)):  # all between low (1.25) and high (3.75)
                if i + k < len(v):
                    v[i + k] = level
            n_chatter += 1
    r = rising_edge_frequency(t, v)
    assert n_chatter == 29 and len(r.edges) == 29 and r.frequency == pytest.approx(1000.0, rel=2e-3)
    # falling-edge chatter does not arm a new edge either
    for i in range(1, len(v)):
        if v[i - 1] == 5.0 and v[i] == 0.0:
            for k, level in enumerate((2.6, 2.2, 2.9)):
                if i + k < len(v):
                    v[i + k] = level
    r = rising_edge_frequency(t, v)
    assert len(r.edges) == 29 and r.frequency == pytest.approx(1000.0, rel=2e-3)


def test_non_finite_and_mismatched_inputs_are_problems_not_numbers():
    t = _times(3000)
    v = _square(t, 1000.0)
    v[1500] = math.nan
    assert "non-finite" in rising_edge_frequency(t, v).problem
    v[1500] = math.inf
    assert "non-finite" in rising_edge_frequency(t, v).problem
    v[1500] = 5.0
    t2 = list(t)
    t2[10] = math.nan
    assert "scale vector contains non-finite" in rising_edge_frequency(t2, v).problem
    t2[10] = t2[9]  # an equal time (an ngspice breakpoint) is accepted ...
    assert rising_edge_frequency(t2, v).frequency == pytest.approx(1000.0, rel=1e-6)
    t2[10] = t2[8]  # ... a step backwards is not
    assert "steps backwards" in rising_edge_frequency(t2, v).problem
    assert "lengths differ" in rising_edge_frequency(t[:-1], v).problem
    assert "no waveform" in rising_edge_frequency([0.0], [1.0]).problem
    assert rising_edge_frequency([], []).frequency is None


def test_the_detector_is_deterministic_and_immutable_on_its_inputs():
    t = _times(1000)
    v = _square(t, 2000.0)
    before = (list(t), list(v))
    a, b = rising_edge_frequency(t, v), rising_edge_frequency(t, v)
    assert a == b and (t, v) == before


# --------------------------------------------------------------------------- the stage's reduction (synthetic result)


def _tran_result(times: list[float], out: list[float], **kw) -> SpiceResult:
    args = dict(
        engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.TRAN, command="tran 1e-5 30m",
        vectors={"time": times, "out": out}, vector_types={"time": "time", "out": "voltage"}, scale="time", n_points=len(times), succeeded=True,
    )
    args.update(kw)
    return SpiceResult(**args)


def _f_osc(nominal: float = 1000.0) -> Expectation:
    return Expectation(id="f_osc", analysis_id="tran", vector="v(OUT)", reduce=Reduce.FREQUENCY, nominal=user_requirement(nominal, "Hz"), tol_rel=user_requirement(0.05), provenance=USER)


def test_reduce_frequency_on_a_synthetic_transient_result():
    t = _times(3000)
    res = _tran_result(t, _square(t, 1000.0))
    r = reduce_expectation(res, _f_osc(), "out")
    assert r.problem is None and r.measured == pytest.approx(1000.0, rel=1e-9) and r.interpolation is None
    # the sampled step sits between two samples: the mid-level crossing is interpolated half a sample before it
    assert r.extra["edges"] == 29 and r.extra["first_edge_s"] == pytest.approx(1e-3 - 5e-6) and r.extra["last_edge_s"] == pytest.approx(29e-3 - 5e-6)
    assert {k: r.extra[k] for k in ("low", "mid", "high", "vmin", "vmax")} == {"low": 1.25, "mid": 2.5, "high": 3.75, "vmin": 0.0, "vmax": 5.0}
    assert reduce_result(res, _f_osc(), "out") == (pytest.approx(1000.0), None)
    # the lookup is the same as for the other reductions: rawfile form and case do not matter
    assert reduce_result(res, _f_osc(), "v(OUT)")[0] == pytest.approx(1000.0)


def test_reduce_frequency_reports_no_oscillation_as_a_problem_with_the_audit_trail():
    t = _times(3000)
    res = _tran_result(t, [5.0] * 3000)
    r = reduce_expectation(res, _f_osc(), "out")
    assert r.measured is None and r.problem.startswith("no oscillation detected") and r.extra["edges"] == 0 and r.extra["vmin"] == r.extra["vmax"] == 5.0
    res = _tran_result(t, [0.0] * 1500 + [5.0] * 1500)  # one step: one edge
    r = reduce_expectation(res, _f_osc(), "out")
    assert r.measured is None and "1 rising edge(s)" in r.problem and r.extra["edges"] == 1 and r.extra["first_edge_s"] == r.extra["last_edge_s"]
    res = _tran_result(t, _square(t, 1000.0))
    assert reduce_expectation(res, _f_osc(), "missing").problem.startswith("vector not produced")
    v = _square(t, 1000.0)
    v[7] = math.nan
    assert "non-finite" in reduce_expectation(_tran_result(t, v), _f_osc(), "out").problem


def test_reduce_frequency_refuses_a_result_that_is_not_a_transient():
    res = SpiceResult(
        engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.DC, command="dc v 0 2 1",
        vectors={"v-sweep": [0.0, 1.0, 2.0], "out": [0.0, 5.0, 0.0]}, scale="v-sweep", n_points=3, succeeded=True,
    )
    r = reduce_expectation(res, _f_osc(), "out")
    assert r.measured is None and r.problem == "reduce=frequency needs a tran result, this is dc"
    op = SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.OP, command="op", vectors={"out": [5.0]}, n_points=1, succeeded=True)
    assert "needs a tran result, this is op" in reduce_expectation(op, _f_osc(), "out").problem
