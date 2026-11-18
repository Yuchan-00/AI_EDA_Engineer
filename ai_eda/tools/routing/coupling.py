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

:func:`parallel_runs` / :class:`NeighbourIndex` are the crosstalk neighbour
search of ``si.crosstalk``: every other net's copper beside a victim within
an edge gap of a stated number of dielectric heights, parallel runs with
their overlap and end points, and the neighbours no formula covers
(non-parallel, touching, broadside, zones) as named records. They share the
parallel test with :func:`coupled_pieces` and leave it unchanged.

**Curved copper.** A track may be a circular arc (``Track.mid``,
:mod:`ai_eda.ir.arc`; read here with ``getattr``, so a stand-in without the
attribute is a straight segment). Everything this module computes for a
coupled line is about two straight parallel conductors, so an arc is never a
coupled piece and never a parallel run, and it is never read as the straight
line between its ends:

* :func:`coupled_pieces` takes no piece from an arc on either net, and
  :func:`uncoupled_lengths` counts an arc at its arc length
  (:func:`~ai_eda.ir.track_length`) as uncoupled copper of its net - the
  class the module already gives corners and bumps, whether or not the
  partner bends beside it. The router's corner pass leaves the nets of a
  pair alone (a pair keeps the mitres its own emission makes), so only
  hand-made copper brings an arc here.
* the neighbour search takes an arc - as the victim's track and as another
  net's - through its exact box (the grid cells: its bulge leaves the box of
  its two ends) and the chords of its chain
  (:func:`~ai_eda.ir.track_points`, each point of the arc within
  :data:`~ai_eda.ir.arc.ARC_SAGITTA_MM` of a chord and the reverse). Another
  net's track within the window of an arc, or an arc within the window of a
  straight victim track, is one record per (victim track, neighbour track)
  of kind ``non_parallel`` with ``"curved": True`` - never ``parallel``,
  never a broadside overlap. Its ``gap_mm`` is the smallest gap between the
  chords, lowered by that bound once per arc: the copper is never nearer
  than it says. The window is tested on that value, so a neighbour inside
  the window is never left out, and one that only the lowering brings inside
  is named too (the reporting side). It is ``contact`` only when the copper
  surely touches (the chords' gap raised by the bound is still <= 0). A zone
  beside an arc victim is measured the same way. Records without an arc are
  exactly what they were (they carry no ``curved`` key).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Protocol

from ai_eda.ir.arc import ARC_SAGITTA_MM
from ai_eda.ir.pcb import track_bbox, track_length, track_points

#: a P / N segment pair is *coupled* when parallel on one layer with an edge gap of at most this many track widths
COUPLED_GAP_FACTOR = 3.0
_TOL = 1e-9


class _Seg(Protocol):
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float


def _is_arc(t: _Seg) -> bool:
    """Whether ``t`` is an arc: a track with a ``mid`` point (module docstring; a stand-in without the attribute is straight)."""
    return getattr(t, "mid", None) is not None


def coupled_pieces(tp: Iterable[_Seg], tn: Iterable[_Seg], factor: float = COUPLED_GAP_FACTOR) -> list[dict]:
    """Parallel P / N segment pairs on one layer whose edge gap is at most ``factor`` widths: their overlap length, width and gap.

    An arc on either net gives no piece (module docstring: curved copper is never a coupled piece).
    """
    tn = [b for b in tn if not _is_arc(b)]
    out: list[dict] = []
    for a in tp:
        if _is_arc(a):
            continue
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
    """``(coupled, uncoupled P, uncoupled N, pieces)`` in mm (module docstring). A net's track length counts an arc at its arc length,
    all of it uncoupled."""
    tp, tn = list(tp), list(tn)
    pieces = coupled_pieces(tp, tn, factor)
    coupled = sum(x["length_mm"] for x in pieces)

    def length(ts: list[_Seg]) -> float:
        return sum(track_length(t) for t in ts)

    return coupled, max(0.0, length(tp) - coupled), max(0.0, length(tn) - coupled), pieces


# --------------------------------------------------------------------------- crosstalk neighbours (si.crosstalk)


class _NetSeg(_Seg, Protocol):
    net: str


class _NetZone(Protocol):
    net: str
    layer: str
    polygon: list[tuple[float, float]]


#: smallest grid cell (mm) of the neighbour index
_CELL_MIN = 0.5
#: the parallel test of :func:`coupled_pieces`: |u_a x d_b| / |d_b| at most this
_PARALLEL_TOL = 1e-6
#: a projected overlap must exceed this (mm) to be a parallel run
_OVERLAP_MIN = 1e-6


def _pt_seg(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    ll = dx * dx + dy * dy
    t = 0.0 if ll <= 0.0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ll))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _seg_dist(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> float:
    """Centreline distance of segments a-b and c-d (0 when they cross)."""
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if orient(a, b, c) * orient(a, b, d) < 0.0 and orient(c, d, a) * orient(c, d, b) < 0.0:
        return 0.0
    return min(_pt_seg(c, a, b), _pt_seg(d, a, b), _pt_seg(a, c, d), _pt_seg(b, c, d))


def _poly_seg_dist(a: tuple[float, float], b: tuple[float, float], pts: list[tuple[float, float]]) -> float:
    """Distance from segment a-b to a polygon area (0 when it enters or touches it)."""
    def inside(p) -> bool:
        x, y = p
        hit = False
        for i in range(len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
            if (y0 > y) != (y1 > y) and x0 + (y - y0) * (x1 - x0) / (y1 - y0) > x:
                hit = not hit
        return hit

    if not pts:
        return math.inf
    if inside(a) or inside(b):
        return 0.0
    return min(_seg_dist(a, b, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


_Chain = tuple[list[tuple[float, float]], tuple[float, float, float, float]]


def _chain(t: _Seg) -> _Chain:
    """The chords a curved record measures ``t`` on - an arc's chain, a straight track's two ends - and the box that holds them."""
    pts = track_points(t)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return pts, (min(xs), min(ys), max(xs), max(ys))


def _chain_dist(pa: list[tuple[float, float]], pb: list[tuple[float, float]]) -> float:
    """The smallest centreline distance between the chords of two chains (0 where two chords cross): :func:`_seg_dist` over every pair
    of chords, in one pass.

    Two segments that do not cross are nearest at an end of one of them, so the smallest distance is the smallest of every point of
    one chain to every chord of the other; chords can only cross where that is no more than the longest chord, and only then are the
    pairs tested for a crossing.
    """
    sa = [(x0, y0, x1, y1, x1 - x0, y1 - y0, (x1 - x0) ** 2 + (y1 - y0) ** 2) for (x0, y0), (x1, y1) in zip(pa, pa[1:])]
    sb = [(x0, y0, x1, y1, x1 - x0, y1 - y0, (x1 - x0) ** 2 + (y1 - y0) ** 2) for (x0, y0), (x1, y1) in zip(pb, pb[1:])]
    best = math.inf  # squared
    for pts, segs in ((pa, sb), (pb, sa)):
        for px, py in pts:
            for x0, y0, _, _, dx, dy, ll in segs:
                t = 0.0 if ll <= 0.0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / ll))
                ex, ey = px - x0 - t * dx, py - y0 - t * dy
                d2 = ex * ex + ey * ey
                if d2 < best:
                    best = d2
    if best <= max(s[6] for s in (*sa, *sb)):
        for ax0, ay0, ax1, ay1, adx, ady, _ in sa:
            for cx0, cy0, cx1, cy1, cdx, cdy, _ in sb:
                if ((adx * (cy0 - ay0) - ady * (cx0 - ax0)) * (adx * (cy1 - ay0) - ady * (cx1 - ax0)) < 0.0
                        and (cdx * (ay0 - cy0) - cdy * (ax0 - cx0)) * (cdx * (ay1 - cy0) - cdy * (ax1 - cx0)) < 0.0):
                    return 0.0
    return math.sqrt(best)


class NeighbourIndex:
    """Other nets' tracks (and zones) bucketed per layer on a uniform grid, built once; :meth:`runs` is :func:`parallel_runs`.

    ``window`` maps a layer to the height ``h`` (mm) its edge gaps are
    measured against; a neighbour counts when its edge gap is at most
    ``g_max`` h. ``adjacent`` maps a layer to ``{other layer: h}`` for an
    adjacent signal layer with no plane between them (broadside neighbours,
    window ``g_max`` times that dielectric). Candidates are visited in the
    order ``others`` gave them, so the output is deterministic. An arc is
    bucketed by its exact box and measured on its chords (module docstring).
    """

    def __init__(self, others: Iterable[_NetSeg], window: Mapping[str, float], *, g_max: float,
                 adjacent: Mapping[str, Mapping[str, float]] | None = None, zones: Iterable[_NetZone] = ()) -> None:
        self.others = list(others)
        self.zones = list(zones)
        self.window = {k: float(v) for k, v in window.items()}
        self.g_max = float(g_max)
        self.adjacent = {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in (adjacent or {}).items()}
        self._wmax = max((float(t.width_mm) for t in self.others), default=0.0)
        reach: dict[str, float] = {layer: self.g_max * h for layer, h in self.window.items()}
        for row in self.adjacent.values():
            for layer, h in row.items():
                reach[layer] = max(reach.get(layer, 0.0), self.g_max * h)
        self._cell = {layer: max(_CELL_MIN, r) for layer, r in reach.items()}
        self._grid: dict[str, dict[tuple[int, int], list[int]]] = {}
        self._chains: dict[int, _Chain] = {}  # of the tracks a curved record was measured on, by their index (:meth:`_chain_of`)
        for i, t in enumerate(self.others):
            if t.layer not in self._cell:
                continue
            grid = self._grid.setdefault(t.layer, {})
            pad = float(t.width_mm) / 2.0
            if _is_arc(t):  # its gaps are measured on its chords, lowered by the chain bound: the cells cover that (and the rounding of its chain) too
                pad += 2.0 * ARC_SAGITTA_MM
            for key in self._keys(t.layer, t, pad):
                grid.setdefault(key, []).append(i)

    def _keys(self, layer: str, t: _Seg, pad: float) -> list[tuple[int, int]]:
        c = self._cell[layer]
        if _is_arc(t):  # the arc's exact box: its bulge leaves the box of its two ends
            x0, y0, x1, y1 = track_bbox(t)
        else:
            x0, x1 = sorted((t.start[0], t.end[0]))
            y0, y1 = sorted((t.start[1], t.end[1]))
        return [(i, j) for i in range(math.floor((x0 - pad) / c), math.floor((x1 + pad) / c) + 1)
                for j in range(math.floor((y0 - pad) / c), math.floor((y1 + pad) / c) + 1)]

    def _near(self, layer: str, t: _Seg, reach: float) -> list[int]:
        grid = self._grid.get(layer)
        if not grid:
            return []
        found: set[int] = set()
        for key in self._keys(layer, t, reach):
            found.update(grid.get(key, ()))
        return sorted(found)

    def _chain_of(self, i: int) -> _Chain:
        """:func:`_chain` of the indexed track ``i``, made once."""
        if i not in self._chains:
            self._chains[i] = _chain(self.others[i])
        return self._chains[i]

    def runs(self, victim: Iterable[_NetSeg]) -> list[dict]:
        """:func:`parallel_runs` of ``victim`` against the indexed copper (other tracks of the victim's own net are skipped)."""
        out: list[dict] = []
        g = self.g_max
        for ia, a in enumerate(victim):
            if _is_arc(a):
                out.extend(self._arc_runs(ia, a))
                continue
            ax, ay = a.end[0] - a.start[0], a.end[1] - a.start[1]
            la = math.hypot(ax, ay)
            if la <= _TOL:
                continue
            ux, uy = ax / la, ay / la
            wa = float(a.width_mm)
            h = self.window.get(a.layer)
            if h is not None:
                for ib in self._near(a.layer, a, g * h + (wa + self._wmax) / 2.0):
                    b = self.others[ib]
                    if b.net == a.net:
                        continue
                    rec = _curved(a, b, h, g, _chain(a), self._chain_of(ib)) if _is_arc(b) else _pair(a, b, ux, uy, la, h, g)
                    if rec is not None:
                        out.append({**rec, "net": b.net, "layer": a.layer, "h_mm": h, "victim_index": ia, "aggressor_index": ib})
                for iz, z in enumerate(self.zones):
                    if z.layer != a.layer or z.net == a.net:
                        continue
                    gap = _poly_seg_dist(tuple(a.start), tuple(a.end), [tuple(p) for p in z.polygon]) - wa / 2.0
                    if gap <= g * h + _TOL:
                        out.append({"kind": "zone", "parallel": False, "net": z.net, "layer": a.layer, "gap_mm": gap, "s_over_h": gap / h, "h_mm": h,
                                    "victim_index": ia, "zone_index": iz})
            for layer2, h2 in self.adjacent.get(a.layer, {}).items():
                for ib in self._near(layer2, a, g * h2 + (wa + self._wmax) / 2.0):
                    b = self.others[ib]
                    if b.net == a.net or _is_arc(b):  # an arc is no parallel overlap: no broadside neighbour
                        continue
                    rec = _pair(a, b, ux, uy, la, h2, g, broadside=True)
                    if rec is not None:
                        out.append({**rec, "net": b.net, "layer": layer2, "victim_layer": a.layer, "h_mm": h2, "victim_index": ia, "aggressor_index": ib})
        return out

    def _arc_runs(self, ia: int, a: _NetSeg) -> list[dict]:
        """The records of the victim's arc track ``a`` (module docstring: curved copper), in the order :meth:`runs` gives a straight one's.

        Every other net's track on its layer within the window is one curved record (:func:`_curved`); a zone's gap is measured from
        the arc's chords and lowered by the chain bound (its record carries ``"curved": True`` too). An adjacent layer gives nothing:
        only an exactly parallel overlap is a broadside record, and an arc is parallel to no track.
        """
        out: list[dict] = []
        g = self.g_max
        wa = float(a.width_mm)
        h = self.window.get(a.layer)
        if h is None:
            return out
        mine = _chain(a)
        # the lookup reaches as far as a straight victim's, plus the chain bound of this arc and of an arc beside it
        for ib in self._near(a.layer, a, g * h + (wa + self._wmax) / 2.0 + 2.0 * ARC_SAGITTA_MM):
            b = self.others[ib]
            if b.net == a.net or (not _is_arc(b) and math.hypot(b.end[0] - b.start[0], b.end[1] - b.start[1]) <= _TOL):
                continue  # its own net; a straight track without length is no neighbour (as beside a straight victim)
            rec = _curved(a, b, h, g, mine, self._chain_of(ib))
            if rec is not None:
                out.append({**rec, "net": b.net, "layer": a.layer, "h_mm": h, "victim_index": ia, "aggressor_index": ib})
        chain = mine[0]
        for iz, z in enumerate(self.zones):
            if z.layer != a.layer or z.net == a.net:
                continue
            polygon = [tuple(p) for p in z.polygon]
            gap = min(_poly_seg_dist(p, q, polygon) for p, q in zip(chain, chain[1:])) - wa / 2.0 - ARC_SAGITTA_MM
            if gap <= g * h + _TOL:
                out.append({"kind": "zone", "parallel": False, "net": z.net, "layer": a.layer, "gap_mm": gap, "s_over_h": gap / h, "h_mm": h,
                            "victim_index": ia, "zone_index": iz, "curved": True})
        return out


def _curved(a: _Seg, b: _Seg, h: float, g: float, ca: _Chain, cb: _Chain) -> dict | None:
    """The record of other-net track ``b`` beside victim track ``a`` when either is an arc (module docstring), or ``None`` outside the
    window; ``ca`` / ``cb`` are their chords (:func:`_chain`).

    The distance is the smallest between the chords of the two chains (a straight track is its own chord). Each arc lies within
    :data:`~ai_eda.ir.arc.ARC_SAGITTA_MM` of its chain, so the true centreline distance differs from that by at most the bound once
    per arc: the gap is reported at its smallest (the chord gap lowered by the bound), the window is tested on that value, and the
    record is ``contact`` only when even the largest possible gap is <= 0 - otherwise ``non_parallel``, whatever the directions of
    the chords: curved copper is never a parallel run. Chains whose boxes alone are beyond the window are not measured.
    """
    (pa, (ax1, ay1, ax2, ay2)), (pb, (bx1, by1, bx2, by2)) = ca, cb
    half = (float(a.width_mm) + float(b.width_mm)) / 2.0
    bound = ARC_SAGITTA_MM * (int(_is_arc(a)) + int(_is_arc(b)))
    reach = g * h + _TOL
    if math.hypot(max(bx1 - ax2, ax1 - bx2, 0.0), max(by1 - ay2, ay1 - by2, 0.0)) - half - bound > reach:
        return None
    chords = _chain_dist(pa, pb) - half
    gap = chords - bound
    if gap > reach:
        return None
    kind = "contact" if chords + bound <= 0.0 else "non_parallel"
    return {"kind": kind, "parallel": False, "gap_mm": gap, "s_over_h": gap / h, "curved": True}


def _pair(a: _Seg, b: _Seg, ux: float, uy: float, la: float, h: float, g: float, broadside: bool = False) -> dict | None:
    """The record of other-net track ``b`` beside victim track ``a`` (unit direction ``ux, uy``, length ``la``), or ``None`` outside the
    window - both straight: :meth:`NeighbourIndex.runs` takes a pair with an arc to :func:`_curved`."""
    bx, by = b.end[0] - b.start[0], b.end[1] - b.start[1]
    lb = math.hypot(bx, by)
    if lb <= _TOL:
        return None
    wa, wb = float(a.width_mm), float(b.width_mm)
    reach = g * h + _TOL
    parallel = abs(ux * by - uy * bx) / lb <= _PARALLEL_TOL
    if parallel:
        dx, dy = b.start[0] - a.start[0], b.start[1] - a.start[1]
        along = ux * dx + uy * dy
        d = abs(ux * dy - uy * dx)
        s0, s1 = sorted((along, ux * (b.end[0] - a.start[0]) + uy * (b.end[1] - a.start[1])))
        t0, t1 = max(0.0, s0), min(la, s1)
        if t1 - t0 > _OVERLAP_MIN:
            gap = d - (wa + wb) / 2.0
            if gap > reach:
                return None
            if broadside:  # stacked copper (gap <= 0) is the strongest broadside case, not a short: another layer
                return {"kind": "broadside", "parallel": True, "length_mm": t1 - t0, "lateral_gap_mm": gap}
            if gap <= 0.0:
                return {"kind": "contact", "parallel": True, "gap_mm": gap, "s_over_h": gap / h}
            ox, oy = dx - ux * along, dy - uy * along  # b's line relative to a's
            v0 = (a.start[0] + ux * t0, a.start[1] + uy * t0)
            v1 = (a.start[0] + ux * t1, a.start[1] + uy * t1)
            return {"kind": "parallel", "parallel": True, "length_mm": t1 - t0, "victim_width_mm": wa, "aggressor_width_mm": wb, "gap_mm": gap,
                    "s_over_h": gap / h, "victim_span": [v0, v1], "aggressor_span": [(v0[0] + ox, v0[1] + oy), (v1[0] + ox, v1[1] + oy)]}
    if broadside:
        return None
    gap = _seg_dist(tuple(a.start), tuple(a.end), tuple(b.start), tuple(b.end)) - (wa + wb) / 2.0
    if gap > reach:
        return None
    if gap <= 0.0:
        return {"kind": "contact", "parallel": parallel, "gap_mm": gap, "s_over_h": gap / h}
    return {"kind": "non_parallel", "parallel": parallel, "gap_mm": gap, "s_over_h": gap / h}


def parallel_runs(victim: Iterable[_NetSeg], others: Iterable[_NetSeg], window: Mapping[str, float], *, g_max: float,
                  adjacent: Mapping[str, Mapping[str, float]] | None = None, zones: Iterable[_NetZone] = ()) -> list[dict]:
    """Every other net's copper beside the victim's tracks within an edge gap of ``g_max`` h, one record per (victim track, neighbour).

    Unlike :func:`coupled_pieces` (whose output ``si.diff`` copies and which
    stays as it is) this is the crosstalk neighbour search of
    ``si.crosstalk``. ``window`` maps each layer the victim is searched on to
    its height h (mm); a layer missing from it is not searched (the caller
    names that copper). ``victim_index`` / ``aggressor_index`` /
    ``zone_index`` index the given sequences. Record ``kind``:

    * ``parallel`` - exactly parallel (the :func:`coupled_pieces` test),
      projected overlap > 1e-6 mm, edge gap s = centre distance -
      (w_v + w_a) / 2 in (0, g_max h]: ``length_mm`` (the overlap), both
      widths, ``gap_mm``, ``s_over_h``, ``h_mm`` and the overlap's end points
      on both tracks (``victim_span`` / ``aggressor_span``);
    * ``non_parallel`` - another net's track on the layer whose nearest edge
      lies within g_max h but that is not a parallel overlap (``parallel``
      says whether it runs in the same direction without overlapping);
    * ``contact`` - its copper touches or crosses the victim (edge gap <= 0);
    * ``broadside`` - a parallel overlap on an ``adjacent`` layer (no plane
      between), lateral edge gap at most g_max times that dielectric
      (``lateral_gap_mm`` <= 0: stacked);
    * ``zone`` - a zone of another net on the layer within g_max h.

    A record whose victim track or neighbour track is an arc carries
    ``"curved": True`` (module docstring): ``non_parallel`` with
    ``parallel`` ``False``, ``contact`` when the copper surely touches, or
    ``zone`` for a zone beside an arc victim - never ``parallel`` and never
    ``broadside`` (only an exactly parallel overlap is a broadside record,
    and an arc is parallel to nothing, as copper that crosses on the other
    layer is none). Its ``gap_mm`` is the gap between the chords lowered by
    the chain bound: the smallest the copper can have. Every other record is
    what it was, without that key.

    Beyond the window nothing is returned. With many victims build one
    :class:`NeighbourIndex` and call its :meth:`~NeighbourIndex.runs`.
    """
    return NeighbourIndex(others, window, g_max=g_max, adjacent=adjacent, zones=zones).runs(victim)


__all__ = ["COUPLED_GAP_FACTOR", "NeighbourIndex", "coupled_pieces", "parallel_runs", "uncoupled_lengths"]
