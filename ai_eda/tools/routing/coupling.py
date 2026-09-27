"""The coupled and uncoupled length of a differential pair's copper - one definition for the router and for ``si.diff``.

Invariant: a pair's *coupled* length is geometry only - the overlap of the
two nets' tracks where a P segment and an N segment run parallel on one
layer with an edge-to-edge gap of at most :data:`COUPLED_GAP_FACTOR` track
widths (beyond three widths the coupling is weak - the 3W crosstalk rule of
thumb, stated here, not a verdict). A net's *uncoupled* length is its track
length minus the coupled length: its breakouts at both ends, the mitred
corners, the compensation bumps - everything that does not run beside its
partner. ``pair_uncoupled_max_mm`` bounds exactly this number per net: the
router (:mod:`ai_eda.tools.routing.maze`) refuses a pair whose emitted
copper breaks it, and ``si.diff`` (:mod:`ai_eda.validation.si`) judges the
IR copper with the same function, so a pair the router applied never fails
the check that owns the budget on this count.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol

#: a P / N segment pair is *coupled* when parallel on one layer with an edge gap of at most this many track widths
COUPLED_GAP_FACTOR = 3.0
_TOL = 1e-9


class _Seg(Protocol):
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float


def coupled_pieces(tp: Iterable[_Seg], tn: Iterable[_Seg], factor: float = COUPLED_GAP_FACTOR) -> list[dict]:
    """Parallel P / N segment pairs on one layer whose edge gap is at most ``factor`` widths: their overlap length, width and gap."""
    tn = list(tn)
    out: list[dict] = []
    for a in tp:
        ax, ay = a.end[0] - a.start[0], a.end[1] - a.start[1]
        la = math.hypot(ax, ay)
        if la <= _TOL:
            continue
        ux, uy = ax / la, ay / la
        for b in tn:
            if b.layer != a.layer:
                continue
            bx, by = b.end[0] - b.start[0], b.end[1] - b.start[1]
            lb = math.hypot(bx, by)
            if lb <= _TOL or abs(ux * by - uy * bx) / lb > 1e-6:
                continue  # not parallel
            # perpendicular distance of b's start from a's line, and the overlap of the projections
            dx, dy = b.start[0] - a.start[0], b.start[1] - a.start[1]
            d = abs(ux * dy - uy * dx)
            s0, s1 = sorted((ux * dx + uy * dy, ux * (b.end[0] - a.start[0]) + uy * (b.end[1] - a.start[1])))
            overlap = min(la, s1) - max(0.0, s0)
            if overlap <= 1e-6:
                continue
            width = (float(a.width_mm) + float(b.width_mm)) / 2.0
            gap = d - width
            if gap <= 0 or gap > factor * width + _TOL:
                continue
            out.append({"layer": a.layer, "length_mm": overlap, "width_mm": width, "gap_mm": gap})
    return out


def uncoupled_lengths(tp: Iterable[_Seg], tn: Iterable[_Seg], factor: float = COUPLED_GAP_FACTOR) -> tuple[float, float, float, list[dict]]:
    """``(coupled, uncoupled P, uncoupled N, pieces)`` in mm (module docstring)."""
    tp, tn = list(tp), list(tn)
    pieces = coupled_pieces(tp, tn, factor)
    coupled = sum(x["length_mm"] for x in pieces)

    def length(ts: list[_Seg]) -> float:
        return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in ts)

    return coupled, max(0.0, length(tp) - coupled), max(0.0, length(tn) - coupled), pieces


__all__ = ["COUPLED_GAP_FACTOR", "coupled_pieces", "uncoupled_lengths"]
