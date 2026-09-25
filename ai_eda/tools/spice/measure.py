"""Pure waveform measurements on simulation vectors (no engine, no IR).

Invariant: **a frequency is measured from the samples, never assumed.**
:func:`rising_edge_frequency` reduces a transient vector to the mean
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
"""

from __future__ import annotations

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


__all__ = ["ABSTOL", "EdgeFrequency", "HIGH_FRACTION", "LOW_FRACTION", "MID_FRACTION", "MIN_EDGES", "RELTOL", "VNTOL", "flat_floor", "rising_edge_frequency"]
