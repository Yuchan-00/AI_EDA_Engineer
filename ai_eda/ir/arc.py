"""Circular track arcs: the geometry of an arc given by three points (pure; board frame, mm, Y down).

A :class:`~ai_eda.ir.Track` with a ``mid`` point is the circular arc from ``start`` through ``mid`` to ``end`` - the form KiCad
stores a track arc in (``(arc (start ..) (mid ..) (end ..) ...)``). The three points are the stored ones, at the file's 1 nm
resolution; everything about the arc is computed from them, here and nowhere else: its centre, radius and signed sweep
(:func:`arc_geometry`), its length, its exact box, the exact distance of a point from it and where along it that point lies,
the point at a fraction of its length, a split in two, and the tangent arc that rounds a corner (:func:`fillet`).

**The chain.** Code that measures copper as straight segments (a clearance audit, a keep-out distance, a drawing) reads an arc as
its *chain* (:func:`arc_points`): the arc's own ``start``, points on the arc in equal angular steps, and its own ``end``, the
interior points rounded to the file resolution (:data:`ARC_ROUNDING_MM`). The steps are chosen so that the rule below holds for
the chain as it is returned, rounding included.

**The chain rule** (two-sided): every point of the arc lies within ``max_sagitta`` of its chain, and every point of the chain
within ``max_sagitta`` of the arc. (The chords lie on the centre's side of the arc, up to the rounding.) So a reader that needs
"far enough" judges the chords with its half-width raised by ``max_sagitta``, and a reader that needs "touches" uses the exact
:func:`point_arc_distance`, the shared end points (the chain starts and ends on the arc's own points), or the chords with the
half-width lowered by ``max_sagitta``. :data:`ARC_SAGITTA_MM`, the default, is 1 um - a stated choice: three orders below the
clearances it is added to, and about 7 to 20 chords for the quarter arcs a rounded corner gets.

**Angles.** ``a0`` and ``sweep`` are ``atan2`` angles in the stored coordinates (Y down), in radians: the point at angle ``a`` is
``(cx + r cos a, cy + r sin a)``, and a positive sweep runs toward larger angles (clockwise on screen, the direction SVG calls
positive). The sweep is the one from ``start`` to ``end`` that passes ``mid``; its size is below a full turn.

**What is no arc** (:func:`arc_problem`): a non-finite coordinate, a start and end that coincide (a full circle has no
start / mid / end form), a mid on an end, or a mid on the line through the ends - each within :data:`ARC_COLLINEAR_MM`, the file
resolution: nearer than that, the three stored points define no circle that the file could tell from a straight line - and a
circle of a radius above :data:`ARC_RADIUS_LIMIT_MM`, which double precision no longer places to that resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ARC_COLLINEAR_MM",
    "ARC_RADIUS_LIMIT_MM",
    "ARC_ROUNDING_MM",
    "ARC_SAGITTA_MM",
    "ArcGeometry",
    "Point",
    "arc_bbox",
    "arc_geometry",
    "arc_length",
    "arc_point_at",
    "arc_points",
    "arc_problem",
    "arc_split",
    "fillet",
    "point_arc_distance",
    "point_arc_fraction",
]

Point = tuple[float, float]

#: the largest distance between an arc and its chain (module docstring: the chain rule) - a stated choice, 1 um
ARC_SAGITTA_MM = 0.001
#: KiCad's file resolution, 1 nm: a chain point rounded to it moves at most 0.71 nm, which the chain's steps leave room for
ARC_ROUNDING_MM = 1e-6
#: three points nearer than this to coinciding or to one line define no arc (the file resolution, 1 nm)
ARC_COLLINEAR_MM = 1e-6
#: the largest radius of an arc this module computes (a stated choice, 1 km): a double holds a point of such a circle to about
#: 2e-10 mm (2.2e-16 of the radius), four orders below the file resolution; a larger circle is refused, not computed loosely
ARC_RADIUS_LIMIT_MM = 1e6

_TAU = 2.0 * math.pi
_DECIMALS = 6  # 1 nm in mm
#: the sine of the angle between two legs below which they count as parallel: no corner to round (a numeric guard)
_PARALLEL_SIN = 1e-9


def _round(v: float) -> float:
    """``v`` at the file resolution, ``-0.0`` as ``0.0``."""
    return round(v, _DECIMALS) + 0.0


@dataclass(frozen=True, slots=True)
class ArcGeometry:
    """The circle and direction of an arc (module docstring: angles)."""

    cx: float
    cy: float
    r: float
    #: the angle of ``start`` seen from the centre
    a0: float
    #: signed, from ``start`` to ``end`` through ``mid``; 0 < abs(sweep) < 2 pi
    sweep: float

    @property
    def length(self) -> float:
        return self.r * abs(self.sweep)


def arc_problem(start: Point, mid: Point, end: Point) -> str | None:
    """Why ``start`` / ``mid`` / ``end`` are no arc (module docstring: what is no arc), or ``None`` for an arc."""
    for name, p in (("start", start), ("mid", mid), ("end", end)):
        if not (math.isfinite(p[0]) and math.isfinite(p[1])):  # first: every comparison with a NaN is false
            return f"its {name} point ({p[0]}, {p[1]}) is not finite"
    chord = math.hypot(end[0] - start[0], end[1] - start[1])
    if chord < ARC_COLLINEAR_MM:
        return "its start and end coincide (a full circle has no start / mid / end form)"
    if math.hypot(mid[0] - start[0], mid[1] - start[1]) < ARC_COLLINEAR_MM:
        return "its mid point coincides with its start"
    if math.hypot(mid[0] - end[0], mid[1] - end[1]) < ARC_COLLINEAR_MM:
        return "its mid point coincides with its end"
    # the true distance of mid from the line through start and end
    off = abs((end[0] - start[0]) * (mid[1] - start[1]) - (end[1] - start[1]) * (mid[0] - start[0])) / chord
    if off < ARC_COLLINEAR_MM:
        return f"its mid point lies on the line through its start and end (within {ARC_COLLINEAR_MM:g} mm): the three points define no circle"
    # the circumradius of the triangle: (the product of its sides) / (4 x its area) = start-mid x mid-end / (2 x off)
    radius = math.hypot(mid[0] - start[0], mid[1] - start[1]) * math.hypot(end[0] - mid[0], end[1] - mid[1]) / (2.0 * off)
    if not math.isfinite(radius):
        return "its circle cannot be computed (its coordinates are too large to multiply)"
    if radius > ARC_RADIUS_LIMIT_MM:
        return f"its radius ({radius:g} mm) is above {ARC_RADIUS_LIMIT_MM:g} mm, beyond which its points cannot be computed to the file resolution"
    return None


def arc_geometry(start: Point, mid: Point, end: Point) -> ArcGeometry:
    """The circle through the three points and the sweep from ``start`` to ``end`` that passes ``mid``; ``ValueError`` for no arc."""
    problem = arc_problem(start, mid, end)
    if problem is not None:
        raise ValueError(f"no arc from ({start[0]}, {start[1]}) through ({mid[0]}, {mid[1]}) to ({end[0]}, {end[1]}): {problem}")
    # the circumcentre, relative to start (keeps its precision wherever the arc lies on the board)
    bx, by = mid[0] - start[0], mid[1] - start[1]
    cx, cy = end[0] - start[0], end[1] - start[1]
    d = 2.0 * (bx * cy - by * cx)
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    ux, uy = (cy * b2 - by * c2) / d, (bx * c2 - cx * b2) / d
    ox, oy = start[0] + ux, start[1] + uy
    a0 = math.atan2(-uy, -ux)
    total = (math.atan2(end[1] - oy, end[0] - ox) - a0) % _TAU  # start to end toward larger angles
    via = (math.atan2(mid[1] - oy, mid[0] - ox) - a0) % _TAU
    return ArcGeometry(cx=ox, cy=oy, r=math.hypot(ux, uy), a0=a0, sweep=total if via < total else total - _TAU)


def arc_length(start: Point, mid: Point, end: Point) -> float:
    """The length of the arc's centreline: radius times the size of its sweep."""
    return arc_geometry(start, mid, end).length


def _within(g: ArcGeometry, angle: float) -> float | None:
    """How far along the sweep (radians from ``start``, 0 to abs(sweep)) the direction ``angle`` lies, or ``None`` outside it."""
    t = ((angle - g.a0) * (1.0 if g.sweep > 0.0 else -1.0)) % _TAU
    return t if t <= abs(g.sweep) else None


def arc_bbox(start: Point, mid: Point, end: Point) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` of the arc's centreline, exact: its two ends and every axis extreme of the circle its sweep passes."""
    g = arc_geometry(start, mid, end)
    xs, ys = [start[0], end[0]], [start[1], end[1]]
    for angle, x, y in ((0.0, g.cx + g.r, g.cy), (math.pi / 2.0, g.cx, g.cy + g.r), (math.pi, g.cx - g.r, g.cy), (-math.pi / 2.0, g.cx, g.cy - g.r)):
        if _within(g, angle) is not None:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def arc_point_at(start: Point, mid: Point, end: Point, fraction: float) -> Point:
    """The point at ``fraction`` of the arc's length from ``start`` (0: ``start`` itself, 1: ``end`` itself; between them not rounded)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction {fraction!r} is not in 0 .. 1")
    if fraction == 0.0:
        return (start[0], start[1])
    if fraction == 1.0:
        return (end[0], end[1])
    g = arc_geometry(start, mid, end)
    a = g.a0 + g.sweep * fraction
    return (g.cx + g.r * math.cos(a), g.cy + g.r * math.sin(a))


def arc_points(start: Point, mid: Point, end: Point, max_sagitta: float = ARC_SAGITTA_MM) -> list[Point]:
    """The arc's chain (module docstring): ``start`` itself, points on the arc in equal angular steps rounded to 1 nm, ``end`` itself.

    A chord over the angle ``2 h`` of a circle of radius ``r`` has the sagitta ``r (1 - cos h)``. The step is the largest whose
    sagitta is ``max_sagitta - ARC_ROUNDING_MM``, so the rounded chain keeps the two-sided rule with ``max_sagitta``; where that
    budget is at least the radius a chord may span up to a half turn (its sagitta is then at most the radius), never more. On a
    large circle (the budget below a millionth of the radius) ``h`` is taken as ``sqrt(2 budget / r)``, which is just below the
    exact ``acos(1 - budget / r)`` and keeps its digits where ``1 - budget / r`` would lose them. ``max_sagitta`` must be finite
    and above ``2 * ARC_ROUNDING_MM``.
    """
    if not (math.isfinite(max_sagitta) and max_sagitta > 2.0 * ARC_ROUNDING_MM):
        raise ValueError(f"max_sagitta {max_sagitta!r} must be finite and above {2.0 * ARC_ROUNDING_MM:g} mm (twice the file resolution)")
    g = arc_geometry(start, mid, end)
    budget = max_sagitta - ARC_ROUNDING_MM
    x = budget / g.r
    step = math.pi if x >= 1.0 else 2.0 * (math.acos(1.0 - x) if x >= 1e-6 else math.sqrt(2.0 * x))
    n = max(1, math.ceil(abs(g.sweep) / step))
    out: list[Point] = [(start[0], start[1])]
    for i in range(1, n):
        a = g.a0 + g.sweep * i / n
        out.append((_round(g.cx + g.r * math.cos(a)), _round(g.cy + g.r * math.sin(a))))
    out.append((end[0], end[1]))
    return out


def point_arc_fraction(p: Point, start: Point, mid: Point, end: Point) -> tuple[float, float]:
    """The exact distance of ``p`` from the arc's centreline, and the fraction of the arc's length from ``start`` at the nearest point.

    Where ``p``'s direction from the centre lies inside the sweep the nearest point is the one in that direction (distance:
    the difference between ``p``'s distance from the centre and the radius); otherwise it is the nearer end. The centre itself is
    one radius from every point of the arc: fraction 0.
    """
    g = arc_geometry(start, mid, end)
    dx, dy = p[0] - g.cx, p[1] - g.cy
    rho = math.hypot(dx, dy)
    if rho > 0.0:
        t = _within(g, math.atan2(dy, dx))
        if t is not None:
            return abs(rho - g.r), t / abs(g.sweep)
    elif rho == 0.0:
        return g.r, 0.0
    d0 = math.hypot(p[0] - start[0], p[1] - start[1])
    d1 = math.hypot(p[0] - end[0], p[1] - end[1])
    return (d0, 0.0) if d0 <= d1 else (d1, 1.0)


def point_arc_distance(p: Point, start: Point, mid: Point, end: Point) -> float:
    """The exact distance of ``p`` from the arc's centreline (:func:`point_arc_fraction`)."""
    return point_arc_fraction(p, start, mid, end)[0]


def arc_split(start: Point, mid: Point, end: Point, fraction: float) -> tuple[tuple[Point, Point | None, Point], tuple[Point, Point | None, Point]]:
    """The arc cut at ``fraction`` of its length (0 < fraction < 1): two parts ``(start, m1, p)`` and ``(p, m2, end)``.

    ``p`` and the two new mid points are points of this arc rounded to the file resolution, like every stored arc point (each
    ``m`` in the middle of its part). A part too flat to be an arc at that resolution (:func:`arc_problem`: its mid within 1 nm
    of its chord) is the straight segment the file would hold: its ``m`` is ``None``. The parts' lengths are ``fraction`` and
    ``1 - fraction`` of the arc's length; measure them so, not from the rounded parts. ``ValueError`` outside 0 .. 1 and where
    the cut falls on an end at the file resolution (there is nothing to cut).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction {fraction!r} is not between 0 and 1")
    g = arc_geometry(start, mid, end)

    def at(f: float) -> Point:
        a = g.a0 + g.sweep * f
        return (_round(g.cx + g.r * math.cos(a)), _round(g.cy + g.r * math.sin(a)))

    p = at(fraction)
    s, e = (start[0], start[1]), (end[0], end[1])
    if math.hypot(p[0] - s[0], p[1] - s[1]) < ARC_COLLINEAR_MM or math.hypot(p[0] - e[0], p[1] - e[1]) < ARC_COLLINEAR_MM:
        raise ValueError(f"the cut at fraction {fraction!r} falls on an end of the arc at the file resolution")
    m1, m2 = at(fraction / 2.0), at((1.0 + fraction) / 2.0)
    return (s, m1 if arc_problem(s, m1, p) is None else None, p), (p, m2 if arc_problem(p, m2, e) is None else None, e)


def fillet(corner: Point, toward1: Point, toward2: Point, d: float) -> tuple[Point, Point, Point, float]:
    """The arc that rounds a corner: tangent to its two legs at the distance ``d`` from the corner. Returns ``(a, mid, b, radius)``.

    ``toward1`` / ``toward2`` are any points on the two legs (they give the directions). ``a`` and ``b`` are the points at ``d``
    along the legs; with the interior angle ``alpha`` between the legs the radius is ``d tan(alpha / 2)`` (``d`` at a right
    angle), the centre lies on the bisector at ``d / cos(alpha / 2)`` from the corner, and ``mid`` on the bisector one radius
    nearer. ``a``, ``mid`` and ``b`` are rounded to the file resolution; the radius is the exact one. ``ValueError`` for a
    ``d`` that is not positive and finite, a leg without length, or legs that are parallel (no corner: no fillet).
    """
    if not (math.isfinite(d) and d > 0.0):
        raise ValueError(f"the tangent length {d!r} must be positive and finite")
    units = []
    for toward in (toward1, toward2):
        vx, vy = toward[0] - corner[0], toward[1] - corner[1]
        n = math.hypot(vx, vy)
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError(f"a leg from ({corner[0]}, {corner[1]}) toward ({toward[0]}, {toward[1]}) has no length")
        units.append((vx / n, vy / n))
    (u1x, u1y), (u2x, u2y) = units
    if abs(u1x * u2y - u1y * u2x) <= _PARALLEL_SIN:
        raise ValueError("the two legs are parallel: there is no corner to round")
    alpha = math.acos(max(-1.0, min(1.0, u1x * u2x + u1y * u2y)))
    radius = d * math.tan(alpha / 2.0)
    bx, by = u1x + u2x, u1y + u2y
    bn = math.hypot(bx, by)
    bx, by = bx / bn, by / bn
    reach = d / math.cos(alpha / 2.0) - radius  # from the corner to mid, along the bisector
    a = (_round(corner[0] + u1x * d), _round(corner[1] + u1y * d))
    b = (_round(corner[0] + u2x * d), _round(corner[1] + u2y * d))
    mid = (_round(corner[0] + bx * reach), _round(corner[1] + by * reach))
    return a, mid, b, radius
