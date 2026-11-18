"""IEC 60063 preferred values (E series): the values resistors, capacitors and inductors are actually sold in.

A calculator that solves a formula returns a real number (``R = 1 / (2 pi f_c C)
= 1591.549... ohm``); no part has that value. A template therefore keeps the
solved number as its own parameter (``<key>_ideal``) and places the part with
the series value one of the calculators below derives from it
(``calc.e_series.<series>.<mode>``). The placed value is a ``derived`` number
like any other: :mod:`ai_eda.tools.calc.recompute` re-runs the calculator, so a
part value that is not the series value of its ideal fails the calculation
check.

Series: ``e6`` (20 %), ``e12`` (10 %), ``e24`` (5 %; also sold at 1 %), ``e96``
(1 %) - the mantissas of IEC 60063:2015, as integers (two digits up to E24,
three for E96). A value is ``mantissa x 10^k`` for any decade ``k``, built from
its decimal spelling so that it is the double a part value text parses back to
(``4.7e-08``, never ``4.7 * 10 ** -8``).

Modes: ``nearest`` (by ratio, so 9.5 goes up to 10 and not down to 9.1; the
lower value on a tie), ``at_most`` and ``at_least`` (a safe direction: a series
resistor that must not pass more current than stated is ``at_least``). A value
within :data:`REL_TOL` of a series value is that series value in every mode, so
float round-off in the solved number never moves it to the neighbour.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ai_eda.ir import Traced
from ai_eda.tools.calc.basic import _derived

#: series name -> mantissas (IEC 60063:2015), ascending; two significant digits up to E24, three for E96
SERIES: dict[str, tuple[int, ...]] = {
    "e6": (10, 15, 22, 33, 47, 68),
    "e12": (10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82),
    "e24": (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91),
    "e96": (
        100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140, 143, 147, 150, 154, 158, 162, 165, 169, 174,
        178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255, 261, 267, 274, 280, 287, 294, 301, 309,
        316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412, 422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549,
        562, 576, 590, 604, 619, 634, 649, 665, 681, 698, 715, 732, 750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976,
    ),
}
#: the tolerance class a series is named for, as a fraction (what a part of that series is sold at, at the loosest)
SERIES_TOLERANCE: dict[str, float] = {"e6": 0.20, "e12": 0.10, "e24": 0.05, "e96": 0.01}
MODES: tuple[str, ...] = ("nearest", "at_most", "at_least")
#: two numbers this close (relative) are the same value: float round-off of a solved number, not a different part
REL_TOL = 1e-9


def _check(value: float, series: str) -> float:
    if series not in SERIES:
        raise ValueError(f"unknown E series {series!r} (known: {', '.join(SERIES)})")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"an E series value needs a positive finite number, got {value!r}")
    return float(value)


def _around(value: float, series: str) -> list[float]:
    """The series values of the decade of ``value`` and its two neighbours, ascending."""
    digits = len(str(SERIES[series][0])) - 1  # 1 for 10..91, 2 for 100..976
    exp = math.floor(math.log10(value))
    return sorted(float(f"{m}e{e - digits}") for e in (exp - 1, exp, exp + 1) for m in SERIES[series])


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def nearest(value: float, series: str) -> float:
    """The ``series`` value nearest to ``value`` by ratio; the lower one on a tie."""
    value = _check(value, series)
    return min(_around(value, series), key=lambda v: (abs(math.log(v / value)), v))


def at_most(value: float, series: str) -> float:
    """The largest ``series`` value not above ``value`` (a value within :data:`REL_TOL` of ``value`` counts as equal)."""
    value = _check(value, series)
    return max(v for v in _around(value, series) if v <= value or _same(v, value))


def at_least(value: float, series: str) -> float:
    """The smallest ``series`` value not below ``value`` (a value within :data:`REL_TOL` of ``value`` counts as equal)."""
    value = _check(value, series)
    return min(v for v in _around(value, series) if v >= value or _same(v, value))


def values_between(lo: float, hi: float, series: str) -> list[float]:
    """Every ``series`` value from ``lo`` to ``hi`` (both included, within :data:`REL_TOL`), ascending."""
    lo, hi = _check(lo, series), _check(hi, series)
    digits = len(str(SERIES[series][0])) - 1
    decades = range(math.floor(math.log10(lo)) - 1, math.floor(math.log10(hi)) + 2)
    values = sorted({float(f"{m}e{e - digits}") for e in decades for m in SERIES[series]})
    return [v for v in values if (lo <= v or _same(v, lo)) and (v <= hi or _same(v, hi))]


def is_standard(value: float, series: str) -> bool:
    """Whether ``value`` is a value of ``series`` (within :data:`REL_TOL`); ``False`` for a non-positive or non-finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return False
    return _same(nearest(value, series), float(value))


def series_of(value: float, ladder: tuple[str, ...] = ("e6", "e12", "e24", "e96")) -> str | None:
    """The first series of ``ladder`` that holds ``value``, or ``None`` when none does."""
    return next((s for s in ladder if is_standard(value, s)), None)


_PICK: dict[str, Callable[[float, str], float]] = {"nearest": nearest, "at_most": at_most, "at_least": at_least}
_WORDS = {"nearest": "nearest to", "at_most": "largest not above", "at_least": "smallest not below"}


def tool_id(series: str, mode: str) -> str:
    """The calculator id of ``series`` / ``mode`` (``calc.e_series.e24.nearest``)."""
    if series not in SERIES or mode not in MODES:
        raise ValueError(f"no E series calculator {series!r} / {mode!r}")
    return f"calc.e_series.{series}.{mode}"


def _calculator(series: str, mode: str) -> Callable[..., Traced[float]]:
    def standard_value(x: Traced[float], ids: tuple[str] = ("x",)) -> Traced[float]:
        value = _PICK[mode](x.value, series)
        return _derived(value, tool_id(series, mode), ids, x.unit, f"the {series.upper()} value (IEC 60063) {_WORDS[mode]} the solved value")

    standard_value.__name__ = f"{series}_{mode}"
    return standard_value


#: tool id -> calculator, for the recompute registry (roles in :data:`ai_eda.tools.calc.basic.ROLES`: one input, ``x``)
E_SERIES_CALCULATORS: dict[str, Callable[..., Traced[float]]] = {tool_id(s, m): _calculator(s, m) for s in SERIES for m in MODES}


def standard_value(
    ideal: Traced[float], ideal_id: str, ladder: tuple[str, ...], mode: str = "nearest", accept: Callable[[Traced[float]], bool] | None = None
) -> Traced[float]:
    """The part value for the solved number ``ideal`` (the parameter ``ideal_id``): the first series of ``ladder`` that ``accept`` takes.

    ``ladder`` goes from the coarsest series to the finest (``("e24", "e96")``
    for a resistor). ``accept`` judges the candidate - typically "the result
    the circuit reaches with this value is within the stated allowance";
    without it the first series is taken. When no series is accepted the
    finest one is returned: the design is still built with a real part, and
    the check that compares the result with the requirement says by how much
    it misses (nothing is widened here).
    """
    if not ladder:
        raise ValueError("an E series ladder names at least one series")
    candidate = None
    for series in ladder:
        candidate = E_SERIES_CALCULATORS[tool_id(series, mode)](ideal, (ideal_id,))
        if accept is None or accept(candidate):
            return candidate
    assert candidate is not None
    return candidate


__all__ = [
    "E_SERIES_CALCULATORS",
    "MODES",
    "REL_TOL",
    "SERIES",
    "SERIES_TOLERANCE",
    "at_least",
    "at_most",
    "is_standard",
    "nearest",
    "series_of",
    "standard_value",
    "tool_id",
    "values_between",
]
