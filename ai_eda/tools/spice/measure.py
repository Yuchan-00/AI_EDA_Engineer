"""Pure waveform measurements on simulation vectors (no engine, no IR).

Invariant: **a frequency, a level or a depth is measured from the samples,
never assumed** - and a measurement the samples cannot support gives no
number, with the reason. :func:`rising_edge_frequency` reduces a transient vector to the mean
frequency of its rising mid-level crossings and returns *no number* whenever
the samples do not show an oscillation it can count: fewer than three
crossings, a flat waveform, a non-finite sample, a scale that
steps backwards (equal consecutive times, which ngspice's breakpoints can
produce, are accepted: the interpolation divides by the voltage difference,
never by the time step), or a length mismatch. The caller (the SPICE stage) reports
that as a failed expectation - "no oscillation detected" is a verdict about
the design, never a PASS by default.

**Flat means below the engine's own resolution, not exactly constant.** The
thresholds scale with the swing, so without a floor a numerical ripple of
1e-12 V around a DC level would be counted as a full-scale oscillation. A
waveform is flat when ``swing <= RELTOL * max(|vmin|, |vmax|) + abs_floor``:
ngspice's default convergence tolerances (``reltol`` 1e-3, ``vntol`` 1e-6 V
for a voltage, ``abstol`` 1e-12 A for a current) are the smallest change the
engine resolves, so a swing inside them is noise by the engine's own
definition and no frequency is measured from it (the floor used is
reported in :attr:`EdgeFrequency.floor`).

The detector is deterministic and stdlib-only:

* thresholds from the saved window: ``low = vmin + 0.25 * swing``,
  ``mid = vmin + 0.5 * swing``, ``high = vmin + 0.75 * swing`` with
  ``swing = vmax - vmin`` (above the floor, so the three thresholds are
  distinct numbers and the bracketing samples of a crossing always differ);
* hysteresis: a rising edge is *armed* once a sample is at or below ``low``
  and *counted* at the first later sample at or above ``high`` - chatter
  around the mid level between the two thresholds never counts twice;
* the crossing time is the linear interpolation of the mid level between the
  last sample below ``mid`` and the first sample at or above it after the
  edge was armed;
* ``f = (N - 1) / (t_N - t_1)`` over the ``N >= 3`` counted edges (the mean
  period of the saved window, so a start-up transient that was cut off by
  the analysis' ``start`` does not enter).

**Window measurements** (``Reduce.RMS`` / ``DB_RMS`` / ``HARMONIC_DBC`` /
``AM_DEPTH``) read the *piecewise-linear interpolant* of the samples ngspice
wrote over a window ``[t_start, t_stop]`` that must lie inside the saved
samples (the window's two edges are interpolated; an edge that falls on a
breakpoint with two samples takes the value inside the window). Every
integral is exact for that interpolant - never a resampling onto a grid of
this module's choosing - and every result carries the largest sample step
inside the window, so the grid it rests on is auditable:

* :func:`window_rms` - ``sqrt(1/T sum h (v0^2 + v0 v1 + v1^2) / 3)``, the exact
  RMS of the interpolant (T = t_stop - t_start). On a grid of step h the
  interpolant of a sine at f has the RMS ``A / sqrt 2 x sqrt((2 + cos(2 pi f h)) / 3)``
  - always low, and aliased at one sample per period - so with ``f_max`` (the
  highest frequency the RMS must include; the stage's ``params["f_max"]``,
  required) a window whose largest step exceeds ``1 / (20 f_max)``
  (:data:`RMS_STEP_FACTOR`) gives no number, and the result carries the
  read-low bound ``1 - sqrt((2 + cos(2 pi f_max h_max)) / 3)`` (at most
  0.82 %, -0.071 dB, at the guard) that the stage judges as the bracket
  ``[RMS, RMS / (1 - b)]``.
* :func:`harmonic_level` - the Fourier coefficient of the interpolant at
  ``n f0``, integrated in closed form segment by segment (over a segment of
  length h with mid-point value vm, slope step dv and x = w h / 2:
  ``e^(-j w t_m) h (vm sinc(x) - j (dv / 2) (sin x - x cos x) / x^2)``), which
  equals a DFT of an infinitely fine resampling of the interpolant;
  ``A_n = (2 / T) |integral|`` and ``dBc = 20 log10(A_k / A_1)``. The window
  must hold whole periods of ``f0`` (within :data:`PERIOD_REL_TOL`), at least
  :data:`MIN_HARMONIC_PERIODS`. Linear interpolation attenuates a component
  at frequency f by sinc^2(pi f h), so the grid is refused when its largest
  step exceeds 1 / (20 k f0) (:data:`HARMONIC_STEP_FACTOR`): harmonic k is
  then attenuated by at most 0.072 dB, the fundamental by at most 0.072 / k^2
  dB. **The fundamental is measured, not assumed:** the bins sit at ``f0``
  and ``k f0`` only, so a waveform that does not repeat at ``f0`` (an
  oscillator whose own frequency is not the nominal one, an unsettled
  start-up, a modulation) leaks between them. The *periodicity residual*
  ``r = RMS(v(t + 1/f0) - v(t))`` over the window less one period (exact for
  the two interpolants, :func:`periodicity_residual`) shows it: a
  fundamental detuned by eps reads ``sqrt(2) pi |eps| A_1``. Above
  :data:`PERIODICITY_TOL` x A_1 (2 %: eps = 0.45 %) there is no number - the
  problem names the residual and the frequency the window's rising edges
  give (:func:`rising_edge_frequency`; ``f_measured`` is recorded, never
  substituted for ``f0``: the IR says which frequency is judged). Below it
  the result carries the bracket ``[dbc_low, dbc_high]`` the true level lies
  in: each bin may be off by the leakage bound ``L = 1.095 r``
  (:data:`LEAKAGE_PER_RESIDUAL`: a component detuned by delta from harmonic j
  leaks at most ``c |delta| (1 / |j - k| + 1 / (j + k))`` into bin k and
  shows ``r_j >= 2 sqrt 2 c |delta|``; summed over j by Cauchy-Schwarz with
  ``sum 1/m^2 = pi^2 / 6``), a detuning ``eps <= r / (sqrt 2 pi A_1)`` lowers
  the bins by ``sinc(pi N eps)`` / ``sinc(pi N k eps)`` over N periods
  (refused when ``N k eps > 1/2``), and the grid attenuation reads each low -
  so ``A_k in [A_k - L, (A_k + L) / (sinc^2 sinc)]`` and likewise A_1. The
  stage judges that bracket, never the bare number. A fundamental at or
  below the engine's resolution (:func:`flat_floor`) is "no fundamental" - no
  number; a harmonic at or below it is measured but flagged
  ``ak_below_floor`` (ngspice's ``reltol`` bounds what such a small level
  means), and the stage never lets such a number PASS. The grid bound is
  the uniform-grid sinc^2 at the largest step; it covers the interpolation
  of a sinusoid, not aliasing (which the grid guard keeps out).
* :func:`am_depth` - per carrier period ``[t_start + i / fc, t_start + (i + 1) / fc)``
  the amplitude ``(max - min) / 2`` (free of a DC offset), ``A_max`` / ``A_min``
  over the whole periods in the window, depth ``100 (A_max - A_min) / (A_max + A_min)``
  in **percent**. Guards: ``f_carrier >= 20 f_mod`` (:data:`MIN_CARRIER_TO_MOD_RATIO`),
  a window of at least one modulation period, and the largest sample step
  inside every carrier period at most ``1 / (16 f_carrier)``
  (:data:`MIN_SAMPLES_PER_CARRIER_PERIOD`; a *count* of samples bounds
  nothing on ngspice's non-uniform grid, a step does). The biases are not
  corrected but bounded, and the stage judges the bracket
  ``[depth_low, depth_high]`` (:func:`am_depth_bracket`), never the bare
  number: the sampled crest is within half a step of the true one, so each
  amplitude reads low by at most ``1 - cos(pi f_carrier h_max)`` (1.92 % at
  the limit); the A_max period's crests sit up to 3/4 of a carrier period
  from the envelope's maximum, so ``A_max`` reads low by at most
  ``VA (1 - cos(pi f_mod / f_carrier) cos(pi f_mod / (2 f_carrier)))`` of the
  envelope's modulation amplitude VA (1.54 % at ratio 20, 0.062 % at 100);
  and ``A_min`` reads high by at most ``VA (1 - cos(2 pi f_mod / f_carrier))``
  (4.89 % at ratio 20, 0.197 % at 100: a period's extreme samples need not
  sit at the carrier's crests, only inside the period, whose envelope rises
  by at most that much within one carrier period of its minimum - the
  smaller crest-only figure is exceeded by a fourth-order term, measured on
  a synthetic ratio-20 wave). At ratio 20 a correct 50 % wave reads 49.24 %
  with the bracket [49.2, 51.3] %.

A result without a number always says why (``problem``); the SPICE stage
turns that into a FAIL for a human, never a PASS.
"""

from __future__ import annotations

import bisect
import cmath
import math
from dataclasses import dataclass, field
from typing import Sequence

#: fraction of the swing above ``vmin`` at which a rising edge is armed / counted
LOW_FRACTION = 0.25
HIGH_FRACTION = 0.75
#: fraction of the swing at which the crossing time is interpolated
MID_FRACTION = 0.5
#: edges needed for a frequency (two edges give one period, which is not a mean)
MIN_EDGES = 3
#: ngspice's default convergence tolerances: a swing inside ``RELTOL * max(|v|) + <abs>`` is not resolved by the
#: engine, so it is flat, not an oscillation
RELTOL = 1e-3
VNTOL = 1e-6
ABSTOL = 1e-12


@dataclass
class EdgeFrequency:
    """What :func:`rising_edge_frequency` measured: the number, or why there is none, and the thresholds it used."""

    frequency: float | None
    #: interpolated mid-level crossing times of the counted rising edges (seconds, increasing)
    edges: list[float] = field(default_factory=list)
    low: float | None = None
    mid: float | None = None
    high: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    #: the swing at or below which the waveform is flat (``RELTOL * max(|vmin|, |vmax|) + abs_floor``)
    floor: float | None = None
    problem: str | None = None


def flat_floor(vmin: float, vmax: float, abs_floor: float = VNTOL) -> float:
    """The swing at or below which a waveform between ``vmin`` and ``vmax`` is flat: ``RELTOL * max(|vmin|, |vmax|) + abs_floor``."""
    return RELTOL * max(abs(vmin), abs(vmax)) + abs_floor


def rising_edge_frequency(times: Sequence[float], values: Sequence[float], *, abs_floor: float = VNTOL) -> EdgeFrequency:
    """The mean frequency of the rising mid-level crossings of ``values`` over ``times``, or why there is none.

    ``times`` must never step backwards and be as long as ``values``; both
    must be finite. ``abs_floor`` is the absolute part of the flatness floor
    (:data:`VNTOL` for a voltage, :data:`ABSTOL` for a current). See the
    module docstring for the detector.
    """
    if not (math.isfinite(abs_floor) and abs_floor >= 0.0):
        raise ValueError(f"abs_floor must be a finite non-negative number, got {abs_floor!r}")
    n = len(values)
    if n != len(times):
        return EdgeFrequency(None, problem=f"scale and vector lengths differ ({len(times)} vs {n} samples)")
    if n < 2:
        return EdgeFrequency(None, problem=f"{n} sample(s): no waveform to measure")
    if any(not math.isfinite(v) for v in values):
        return EdgeFrequency(None, problem="vector contains non-finite samples (the simulation did not produce a usable waveform)")
    if any(not math.isfinite(t) for t in times):
        return EdgeFrequency(None, problem="the scale vector contains non-finite samples")
    if any(times[i] < times[i - 1] for i in range(1, n)):
        return EdgeFrequency(None, problem="the scale vector steps backwards (not a transient time axis)")
    vmin, vmax = min(values), max(values)
    swing = vmax - vmin
    floor = flat_floor(vmin, vmax, abs_floor)
    if swing <= floor:
        return EdgeFrequency(
            None, vmin=vmin, vmax=vmax, floor=floor,
            problem=f"no oscillation detected: the waveform is flat at {vmin:.6g} .. {vmax:.6g} (swing {swing:.3g} is within the engine's resolution {floor:.3g})",
        )
    low = vmin + LOW_FRACTION * swing
    mid = vmin + MID_FRACTION * swing
    high = vmin + HIGH_FRACTION * swing

    edges: list[float] = []
    armed = False
    below_mid_index: int | None = None  # last sample strictly below mid since the edge was armed
    for i, v in enumerate(values):
        if armed:
            if v < mid:
                below_mid_index = i
            elif below_mid_index is not None and v >= high:
                j = below_mid_index
                # the first sample at or above mid after j brackets the crossing with j
                k = j + 1
                while values[k] < mid:
                    k += 1
                t0, t1, v0, v1 = times[j], times[k], values[j], values[k]
                # v0 < mid <= v1 by construction (the floor keeps mid strictly above vmin); the guard is so that no
                # input can ever divide by zero here
                edges.append(t0 + (mid - v0) * (t1 - t0) / (v1 - v0) if v1 > v0 else t1)
                armed = False
                below_mid_index = None
        if not armed and v <= low:
            armed = True
            below_mid_index = i
    base = EdgeFrequency(None, edges=edges, low=low, mid=mid, high=high, vmin=vmin, vmax=vmax, floor=floor)
    if len(edges) < MIN_EDGES:
        base.problem = (
            f"no oscillation detected: {len(edges)} rising edge(s) through the mid level {mid:.6g} in the saved window, "
            f"{MIN_EDGES} are needed for a frequency (swing {vmin:.6g} .. {vmax:.6g})"
        )
        return base
    span = edges[-1] - edges[0]
    if span <= 0.0:
        base.problem = f"no oscillation detected: the {len(edges)} rising edges all sit at t = {edges[0]:.6g} (the time axis does not advance)"
        return base
    base.frequency = (len(edges) - 1) / span
    return base


# --------------------------------------------------------------------------- window measurements

#: the largest sample step inside a carrier period is at most 1 / (this x f_carrier) for an AM depth
MIN_SAMPLES_PER_CARRIER_PERIOD = 16
#: an AM depth needs f_carrier >= this x f_mod (the per-period amplitude averages the envelope over a carrier period)
MIN_CARRIER_TO_MOD_RATIO = 20
#: the largest sample step in a harmonic's window is at most 1 / (this x k x f0)
HARMONIC_STEP_FACTOR = 20
#: the largest sample step in an RMS window is at most 1 / (this x f_max)
RMS_STEP_FACTOR = 20
#: relative slack of "whole periods" (a window of N / f0 written as decimal seconds is not exactly N / f0)
PERIOD_REL_TOL = 1e-6
#: a harmonic's window must hold at least this many periods of f0: the periodicity residual compares one with the next
MIN_HARMONIC_PERIODS = 2
#: the periodicity residual (RMS of v(t + 1/f0) - v(t)) above which the samples do not repeat at f0, as a fraction of
#: the fundamental's amplitude: a fundamental detuned by eps alone reads sqrt(2) pi |eps| a1, so 0.02 is eps = 0.45 %
PERIODICITY_TOL = 0.02
#: the bound on the leakage into a harmonic's bin per unit of periodicity residual (module docstring):
#: (sqrt(pi^2 / 3) + sqrt(pi^2 / 6)) / (2 sqrt 2) = 1.0954
LEAKAGE_PER_RESIDUAL = (math.pi / math.sqrt(3.0) + math.pi / math.sqrt(6.0)) / (2.0 * math.sqrt(2.0))
#: a detuning that moves harmonic k by this fraction of a bin over the window makes the bin loss unbounded
MAX_BIN_SHIFT = 0.5


@dataclass(frozen=True)
class WindowRms:
    """What :func:`window_rms` measured: the RMS (``None`` with ``problem``), the points it integrated, the grid and its bias."""

    value: float | None
    problem: str | None
    #: points of the interpolant integrated (the samples strictly inside the window plus its two interpolated edges)
    samples: int = 0
    max_step: float | None = None
    #: the highest frequency the RMS must include (``None``: not given, no grid guard - a caller that judges gives it)
    f_max: float | None = None
    #: the largest step the guard admits, 1 / (RMS_STEP_FACTOR f_max)
    step_limit: float | None = None
    #: the relative read-low bound of a sine at f_max on the largest step, 1 - sqrt((2 + cos(2 pi f_max h)) / 3)
    bias_bound_rel: float | None = None


@dataclass(frozen=True)
class Harmonic:
    """What :func:`harmonic_level` measured: harmonic ``k`` in dBc (``None`` with ``problem``), its bias bracket and the numbers behind it."""

    dbc: float | None
    problem: str | None
    #: amplitudes of the fundamental and of harmonic k (the vector's unit)
    a1: float | None = None
    ak: float | None = None
    #: periods of f0 in the window
    periods: float = 0.0
    max_step: float | None = None
    #: the largest step the guard admits, 1 / (HARMONIC_STEP_FACTOR k f0)
    step_limit: float = 0.0
    #: the engine's resolution for this waveform (:func:`flat_floor`)
    floor: float | None = None
    #: harmonic k is at or below ``floor``: measured, but not a resolved level
    ak_below_floor: bool = False
    #: RMS of v(t + 1/f0) - v(t) over the window less one period, relative to a1 (0 for a waveform that repeats at f0)
    periodicity_residual_rel: float | None = None
    #: the fundamental measured from the window's rising edges (``None`` when fewer than three are countable)
    f_measured: float | None = None
    #: the fundamental's relative detuning the residual allows, residual / (sqrt(2) pi a1)
    detuning_bound_rel: float | None = None
    #: the leakage bound into each bin, LEAKAGE_PER_RESIDUAL x residual (the vector's unit)
    leakage_bound: float | None = None
    #: the grid attenuation of harmonic k and of the fundamental on the largest step, dB (>= 0)
    grid_attenuation_k_db: float | None = None
    grid_attenuation_1_db: float | None = None
    #: where the true level lies given those bounds: [dbc_low, dbc_high] (``dbc_low`` ``None``: unbounded below)
    dbc_low: float | None = None
    dbc_high: float | None = None


@dataclass(frozen=True)
class AmDepth:
    """What :func:`am_depth` measured: the depth in percent (``None`` with ``problem``), its bias bracket and its audit numbers."""

    depth: float | None
    problem: str | None
    a_max: float | None = None
    a_min: float | None = None
    #: whole carrier periods in the window
    cycles: int = 0
    #: the fewest interpolant segments in one carrier period
    min_samples_per_cycle: int = 0
    max_step: float | None = None
    #: the largest step the guard admits, 1 / (MIN_SAMPLES_PER_CARRIER_PERIOD f_carrier)
    step_limit: float = 0.0
    #: relative read-low bound of a sampled crest, 1 - cos(pi f_carrier max_step)
    crest_bias_bound_rel: float | None = None
    #: the envelope smear bound on A_max (read low), as a fraction of the envelope's modulation amplitude VA (module docstring)
    envelope_bias_bound_rel: float = 0.0
    #: the bound on A_min (read high), 1 - cos(2 pi f_mod / f_carrier) of VA: the envelope's rise within one carrier
    #: period either side of its minimum (a period's extreme samples need not sit at the carrier's crests)
    envelope_min_bias_bound_rel: float = 0.0
    floor: float | None = None
    #: where the true depth lies given the bias bounds (module docstring), percent
    depth_low: float | None = None
    depth_high: float | None = None


def _series_problem(times: Sequence[float], values: Sequence[float]) -> str | None:
    """The FREQUENCY rules for a usable transient series: equal lengths, two samples, finite, a time axis that never steps back."""
    n = len(values)
    if n != len(times):
        return f"scale and vector lengths differ ({len(times)} vs {n} samples)"
    if n < 2:
        return f"{n} sample(s): no waveform to measure"
    if any(not math.isfinite(v) for v in values):
        return "vector contains non-finite samples (the simulation did not produce a usable waveform)"
    if any(not math.isfinite(t) for t in times):
        return "the scale vector contains non-finite samples"
    if any(times[i] < times[i - 1] for i in range(1, n)):
        return "the scale vector steps backwards (not a transient time axis)"
    return None


def _positive_arg(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return v


def _value_at(times: Sequence[float], values: Sequence[float], x: float, after: bool) -> float:
    """The interpolant at ``x`` (inside the samples): just after ``x`` (the last sample there when a breakpoint repeats it) or just before it (the first)."""
    if after:
        i = bisect.bisect_right(times, x) - 1
        if times[i] == x or i + 1 >= len(times):
            return float(values[i])
        j = i + 1
    else:
        j = bisect.bisect_left(times, x)
        if j >= len(times) or times[j] == x or j == 0:
            return float(values[min(j, len(times) - 1)])
        i = j - 1
    t0, t1 = times[i], times[j]
    return float(values[i]) + (float(values[j]) - float(values[i])) * (x - t0) / (t1 - t0)


def _window(times: Sequence[float], values: Sequence[float], t_start: float, t_stop: float) -> tuple[list[tuple[float, float]], str | None]:
    """The interpolant's points on ``[t_start, t_stop]``: the interpolated start, every sample strictly inside, the interpolated stop.

    The start takes the value just after ``t_start`` (the last sample at that
    time when a breakpoint repeats it), the stop the value just before
    ``t_stop`` (the first one): the values inside the window.
    """
    a, b = float(t_start), float(t_stop)
    if not (math.isfinite(a) and math.isfinite(b)) or not a < b:
        return [], f"the window [{t_start!r}, {t_stop!r}] s is not an interval (t_start < t_stop, both finite)"
    if a < times[0] or b > times[-1]:
        return [], f"the window [{a:.6g}, {b:.6g}] s is not inside the saved samples [{times[0]:.6g}, {times[-1]:.6g}] s"
    lo, hi = bisect.bisect_right(times, a), bisect.bisect_left(times, b)
    points = [(a, _value_at(times, values, a, True))] + [(float(times[i]), float(values[i])) for i in range(lo, hi)] + [(b, _value_at(times, values, b, False))]
    return points, None


def _max_step(points: list[tuple[float, float]]) -> float:
    return max(t1 - t0 for (t0, _), (t1, _) in zip(points, points[1:]))


def _sinc(x: float) -> float:
    if abs(x) < 1e-2:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return math.sin(x) / x


def rms_bias_bound_rel(f_max: float, step: float) -> float:
    """1 - sqrt((2 + cos(2 pi f h)) / 3): how far low the interpolant's RMS reads a sine at ``f_max`` sampled every ``step``."""
    return 1.0 - math.sqrt(max(0.0, (2.0 + math.cos(2.0 * math.pi * f_max * step)) / 3.0))


def window_rms(times: Sequence[float], values: Sequence[float], t_start: float, t_stop: float, f_max: float | None = None) -> WindowRms:
    """The exact RMS of the piecewise-linear interpolant of ``values`` over ``[t_start, t_stop]`` (module docstring), or why there is none.

    With ``f_max`` (the highest frequency the RMS must include, Hz) a window
    whose largest step exceeds ``1 / (RMS_STEP_FACTOR f_max)`` gives no number,
    and the result carries the read-low bound :func:`rms_bias_bound_rel` of the
    largest step; the SPICE stage always gives it (``params["f_max"]``).
    """
    limit = None if f_max is None else 1.0 / (RMS_STEP_FACTOR * _positive_arg(f_max, "f_max"))
    problem = _series_problem(times, values)
    if problem is not None:
        return WindowRms(None, problem, f_max=f_max, step_limit=limit)
    points, problem = _window(times, values, t_start, t_stop)
    if problem is not None:
        return WindowRms(None, problem, f_max=f_max, step_limit=limit)
    step = _max_step(points)
    bias = None if f_max is None else rms_bias_bound_rel(f_max, step)
    if limit is not None and step > limit * (1.0 + PERIOD_REL_TOL):
        return WindowRms(None, f"time grid too coarse for an RMS up to f_max = {f_max:.6g} Hz: largest step {step:.6g} s > {limit:.6g} s "
                               f"(a sine at f_max would read up to {100.0 * bias:.3g} % low, and above it aliases); lower the tran step",  # type: ignore[operator]
                         samples=len(points), max_step=step, f_max=f_max, step_limit=limit, bias_bound_rel=bias)
    integral = 0.0
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        integral += (t1 - t0) * (v0 * v0 + v0 * v1 + v1 * v1) / 3.0
    span = points[-1][0] - points[0][0]
    return WindowRms(math.sqrt(max(integral, 0.0) / span), None, samples=len(points), max_step=step, f_max=f_max, step_limit=limit, bias_bound_rel=bias)


def _odd_term(x: float) -> float:
    """(sin x - x cos x) / x^2, by its series near 0 (where the difference cancels)."""
    if abs(x) < 1e-2:
        x2 = x * x
        return x / 3.0 - x * x2 / 30.0 + x * x2 * x2 / 840.0
    return (math.sin(x) - x * math.cos(x)) / (x * x)


def _fourier(points: list[tuple[float, float]], f: float) -> complex:
    """The integral of the interpolant times e^(-j 2 pi f (t - t_start)) over the window, exact segment by segment."""
    w = 2.0 * math.pi * f
    origin = points[0][0]
    total = 0j
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        h = t1 - t0
        if h <= 0.0:
            continue  # a repeated breakpoint: no length, no contribution
        x = w * h / 2.0
        tm = (t0 + t1) / 2.0 - origin
        total += cmath.exp(-1j * w * tm) * h * ((v0 + v1) / 2.0 * _sinc(x) - 0.5j * (v1 - v0) * _odd_term(x))
    return total


def periodicity_residual(times: Sequence[float], values: Sequence[float], t_start: float, t_stop: float, period: float) -> float:
    """RMS over ``[t_start, t_stop - period]`` of ``v(t + period) - v(t)``, exact for the piecewise-linear interpolant.

    The breakpoints of both interpolants (the samples and the samples shifted
    back by ``period``) split the interval into pieces on which the difference
    is linear, so each piece integrates exactly. 0 for a waveform that repeats
    with ``period``; the window must hold more than one period.
    """
    a, b = float(t_start), float(t_stop) - float(period)
    if not a < b:
        raise ValueError(f"the window [{t_start!r}, {t_stop!r}] s holds no more than one period {period!r} s")
    lo, hi = bisect.bisect_right(times, a), bisect.bisect_left(times, b)
    cuts = {a, b, *(float(times[i]) for i in range(lo, hi))}
    lo2, hi2 = bisect.bisect_right(times, a + period), bisect.bisect_left(times, b + period)
    cuts.update(float(times[i]) - period for i in range(lo2, hi2))
    grid = sorted(x for x in cuts if a <= x <= b)
    integral = 0.0
    for x0, x1 in zip(grid, grid[1:]):
        h = x1 - x0
        if h <= 0.0:
            continue
        d0 = _value_at(times, values, min(x0 + period, times[-1]), True) - _value_at(times, values, x0, True)
        d1 = _value_at(times, values, min(x1 + period, times[-1]), False) - _value_at(times, values, x1, False)
        integral += h * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
    return math.sqrt(max(integral, 0.0) / (b - a))


def harmonic_level(times: Sequence[float], values: Sequence[float], f0: float, k: int, t_start: float, t_stop: float,
                   *, abs_floor: float = VNTOL) -> Harmonic:
    """The level of harmonic ``k`` of ``f0`` relative to the fundamental, in dBc, over whole periods (module docstring), or why there is none.

    The number is the Fourier ratio at ``k f0`` and ``f0``; the result also
    carries the bracket ``[dbc_low, dbc_high]`` the true level lies in given
    the grid attenuation and the leakage the periodicity residual allows, and
    no number at all when the samples do not repeat at ``f0`` (the waveform's
    own fundamental is not ``f0``, a start-up transient, a modulation).
    ``ValueError`` for arguments that are not a measurement's (``f0 <= 0``,
    ``k`` not an integer >= 2); every problem of the samples is a ``problem``.
    ``abs_floor`` is the absolute part of the engine's resolution
    (:data:`VNTOL` for a voltage, :data:`ABSTOL` for a current).
    """
    f0 = _positive_arg(f0, "f0")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or int(k) != k or int(k) < 2:
        raise ValueError(f"k must be an integer >= 2, got {k!r}")
    k = int(k)
    limit = 1.0 / (HARMONIC_STEP_FACTOR * k * f0)
    problem = _series_problem(times, values)
    if problem is not None:
        return Harmonic(None, problem, step_limit=limit)
    points, problem = _window(times, values, t_start, t_stop)
    if problem is not None:
        return Harmonic(None, problem, step_limit=limit)
    span = points[-1][0] - points[0][0]
    periods = span * f0
    whole = round(periods)
    if whole < 1 or abs(periods - whole) > PERIOD_REL_TOL * max(1.0, periods):
        return Harmonic(None, f"the window holds {periods:.9g} periods of f0 = {f0:.6g} Hz, not a whole number of them", periods=periods, step_limit=limit)
    if whole < MIN_HARMONIC_PERIODS:
        return Harmonic(None, f"the window holds {whole} period of f0 = {f0:.6g} Hz: at least {MIN_HARMONIC_PERIODS} are needed to show that the "
                              "waveform repeats at f0", periods=periods, step_limit=limit)
    step = _max_step(points)
    if step > limit:
        return Harmonic(None, f"time grid too coarse for harmonic {k}: largest step {step:.6g} s > {limit:.6g} s; lower the tran step",
                        periods=periods, max_step=step, step_limit=limit)
    vs = [v for _, v in points]
    floor = flat_floor(min(vs), max(vs), abs_floor)
    a1 = 2.0 / span * abs(_fourier(points, f0))
    ak = 2.0 / span * abs(_fourier(points, k * f0))
    base: dict = dict(a1=a1, ak=ak, periods=periods, max_step=step, step_limit=limit, floor=floor, ak_below_floor=ak <= floor)
    if a1 <= floor:
        return Harmonic(None, f"no fundamental: the amplitude at f0 = {f0:.6g} Hz ({a1:.3g}) is within the engine's resolution {floor:.3g}", **base)
    residual = periodicity_residual(times, values, points[0][0], points[-1][0], 1.0 / f0)
    in_window = [i for i, t in enumerate(times) if points[0][0] <= t <= points[-1][0]]
    edge = rising_edge_frequency([times[i] for i in in_window], [values[i] for i in in_window], abs_floor=abs_floor) if len(in_window) >= 2 else None
    f_measured = None if edge is None else edge.frequency
    eps = residual / (math.sqrt(2.0) * math.pi * a1)
    leak = LEAKAGE_PER_RESIDUAL * residual
    base.update(periodicity_residual_rel=residual / a1, f_measured=f_measured, detuning_bound_rel=eps, leakage_bound=leak)
    if residual > PERIODICITY_TOL * a1:
        seen = f"; its rising edges give {f_measured:.9g} Hz" if f_measured is not None else ""
        return Harmonic(None, f"the waveform does not repeat at f0 = {f0:.6g} Hz over the window: the periodicity residual RMS(v(t + 1/f0) - v(t)) is "
                              f"{100.0 * residual / a1:.3g} % of the fundamental (limit {100.0 * PERIODICITY_TOL:g} %){seen} - its fundamental is not f0, "
                              "or the window holds a start-up or a modulation, and the Fourier bins of f0 leak; set f0 to the fundamental the circuit "
                              "produces (reduce=frequency measures it) or start the window in the steady state", **base)
    if whole * k * eps > MAX_BIN_SHIFT:
        return Harmonic(None, f"the window of {whole} periods resolves a detuning of {eps:.3g} (from the periodicity residual) into more than "
                              f"{MAX_BIN_SHIFT} of a bin at harmonic {k}: shorten the window", **base)
    if ak <= 0.0:
        return Harmonic(None, f"harmonic {k} has amplitude 0 in the interpolant: no level in dB", **base)
    g1 = _sinc(math.pi * f0 * step) ** 2
    gk = _sinc(math.pi * k * f0 * step) ** 2
    s1 = _sinc(math.pi * whole * eps)
    sk = _sinc(math.pi * whole * k * eps)
    a1_lo, a1_hi = a1 - leak, (a1 + leak) / (g1 * s1)
    ak_lo, ak_hi = ak - leak, (ak + leak) / (gk * sk)
    dbc = 20.0 * math.log10(ak / a1)
    high = max(dbc, 20.0 * math.log10(ak_hi / a1_lo))
    low = min(dbc, 20.0 * math.log10(ak_lo / a1_hi)) if ak_lo > 0.0 else None
    base.update(grid_attenuation_k_db=-20.0 * math.log10(gk), grid_attenuation_1_db=-20.0 * math.log10(g1), dbc_low=low, dbc_high=high)
    return Harmonic(dbc, None, **base)


def am_depth_bracket(a_max: float, a_min: float, crest: float, envelope: float, envelope_min: float) -> tuple[float, float]:
    """``(low, high)`` percent: where the true depth lies given the measured amplitudes and the bias bounds (module docstring).

    A_max reads low by at most ``crest`` (relative) and ``envelope`` x M, A_min
    high by at most ``envelope_min`` x M and low by at most ``crest``, with M the
    envelope's modulation amplitude, bounded by
    ``(A_max / (1 - crest) - A_min) / (2 - envelope - envelope_min)``.
    """
    def depth(a: float, b: float) -> float:
        return 100.0 * (a - b) / (a + b) if a + b > 0.0 else 0.0

    c = min(max(crest, 0.0), 0.5)
    e = min(max(envelope, 0.0), 0.5)
    e2 = min(max(envelope_min, 0.0), 0.5)
    m_ub = max(0.0, (a_max / (1.0 - c) - a_min) / (2.0 - e - e2))
    high = depth(a_max / (1.0 - c) + e * m_ub, max(0.0, a_min - e2 * m_ub))
    low = max(0.0, depth(a_max, a_min / (1.0 - c)))
    measured = depth(a_max, a_min)
    return min(low, measured), max(high, measured)


def am_depth(times: Sequence[float], values: Sequence[float], f_carrier: float, f_mod: float, t_start: float, t_stop: float,
             *, abs_floor: float = VNTOL) -> AmDepth:
    """The AM modulation depth in percent from the per-carrier-period amplitude (module docstring), or why there is none.

    ``ValueError`` unless ``f_carrier > f_mod > 0``; every other problem is a ``problem``.
    The result carries the bracket ``[depth_low, depth_high]`` of :func:`am_depth_bracket`.
    """
    fc = _positive_arg(f_carrier, "f_carrier")
    fm = _positive_arg(f_mod, "f_mod")
    if not fc > fm:
        raise ValueError(f"f_carrier ({f_carrier!r}) must exceed f_mod ({f_mod!r})")
    limit = 1.0 / (MIN_SAMPLES_PER_CARRIER_PERIOD * fc)
    envelope = 1.0 - math.cos(math.pi * fm / fc) * math.cos(math.pi * fm / (2.0 * fc))
    envelope_min = 1.0 - math.cos(2.0 * math.pi * fm / fc)
    base: dict = dict(step_limit=limit, envelope_bias_bound_rel=envelope, envelope_min_bias_bound_rel=envelope_min)
    if fc < MIN_CARRIER_TO_MOD_RATIO * fm:
        return AmDepth(None, f"carrier too slow for the envelope: f_carrier / f_mod = {fc / fm:.6g} < {MIN_CARRIER_TO_MOD_RATIO} "
                             "(each carrier period's amplitude averages the envelope over that period)", **base)
    problem = _series_problem(times, values)
    if problem is not None:
        return AmDepth(None, problem, **base)
    span = float(t_stop) - float(t_start)
    if math.isfinite(span) and span * fm < 1.0 - PERIOD_REL_TOL:
        return AmDepth(None, f"window shorter than one modulation period ({span:.6g} s < 1 / f_mod = {1.0 / fm:.6g} s)", **base)
    points, problem = _window(times, values, t_start, t_stop)
    if problem is not None:
        return AmDepth(None, problem, **base)
    a = points[0][0]
    cycles = int(math.floor(span * fc * (1.0 + PERIOD_REL_TOL)))
    if cycles < 1:
        return AmDepth(None, "no carrier cycle in the window", **base)
    amplitudes: list[float] = []
    fewest: int | None = None
    worst_step = 0.0
    for i in range(cycles):
        lo, hi = a + i / fc, min(a + (i + 1) / fc, float(t_stop))
        piece, problem = _window(times, values, lo, hi)
        if problem is not None:
            return AmDepth(None, problem, cycles=cycles, **base)
        step = _max_step(piece)
        worst_step = max(worst_step, step)
        if step > limit * (1.0 + PERIOD_REL_TOL):
            return AmDepth(None, f"time grid too coarse for the carrier: largest step {step:.6g} s > {limit:.6g} s in the carrier period starting at "
                                 f"{lo:.9g} s; lower the tran step", cycles=cycles, max_step=step, **base)
        segments = len(piece) - 1
        fewest = segments if fewest is None else min(fewest, segments)
        vs = [v for _, v in piece]
        amplitudes.append((max(vs) - min(vs)) / 2.0)
    vs = [v for _, v in points]
    floor = flat_floor(min(vs), max(vs), abs_floor)
    a_max, a_min = max(amplitudes), min(amplitudes)
    crest = 1.0 - math.cos(math.pi * fc * worst_step)
    audit = dict(a_max=a_max, a_min=a_min, cycles=cycles, min_samples_per_cycle=fewest or 0, max_step=worst_step,
                 crest_bias_bound_rel=crest, floor=floor, **base)
    if a_max + a_min <= floor:
        return AmDepth(None, f"no carrier: the amplitude ({a_min:.3g} .. {a_max:.3g}) is within the engine's resolution {floor:.3g}", **audit)
    low, high = am_depth_bracket(a_max, a_min, crest, envelope, envelope_min)
    return AmDepth(100.0 * (a_max - a_min) / (a_max + a_min), None, depth_low=low, depth_high=high, **audit)


__all__ = [
    "ABSTOL",
    "AmDepth",
    "EdgeFrequency",
    "HARMONIC_STEP_FACTOR",
    "HIGH_FRACTION",
    "Harmonic",
    "LEAKAGE_PER_RESIDUAL",
    "LOW_FRACTION",
    "MIN_CARRIER_TO_MOD_RATIO",
    "MIN_EDGES",
    "MIN_HARMONIC_PERIODS",
    "MIN_SAMPLES_PER_CARRIER_PERIOD",
    "MID_FRACTION",
    "PERIODICITY_TOL",
    "PERIOD_REL_TOL",
    "RELTOL",
    "RMS_STEP_FACTOR",
    "VNTOL",
    "WindowRms",
    "am_depth",
    "am_depth_bracket",
    "flat_floor",
    "harmonic_level",
    "periodicity_residual",
    "rising_edge_frequency",
    "rms_bias_bound_rel",
    "window_rms",
]
