"""Plane connectivity of routed copper: which of a plane net's vias and through-hole pads the plane's fill still joins.

Pure geometry for the router's stack mode (``routing.maze`` 0.9, module docstring of :mod:`ai_eda.tools.routing.maze`: the plane
web); this module knows nothing of the maze router. The rule it checks is TI SCAA082A ("PCB Design Guidelines For Reduced EMI",
sec. 2.5; research_synthesis.json, opened): closely spaced vias must not cut slots in a plane - here, a plane via (or through-hole
pad) of the plane's net must still be joined to the rest of its plane by the copper KiCad pours.

The fill model (KiCad's zone filler, GPL - its behaviour as the compiler's zones get it, no code): a plane zone is filled with its
polygon minus a *knockout* around every copper item of another net on its layer - the item's copper grown by the zone clearance
(the compiler writes :data:`~ai_eda.compilers.pcb.ZONE_CLEARANCE_MM` for a zone without its own) - and the result is pruned at the
zone's minimum thickness (:data:`~ai_eda.compilers.pcb.ZONE_MIN_THICKNESS_MM`): shrunk by half of it, every island of the shrunk
fill whose extent (the larger side of its bounding box) is below the minimum thickness deleted, and grown back, so a web narrower
than the minimum thickness and a pocket smaller than it disappear (the island step as KiCad's ``zone_filler.cpp``,
``fillCopperZone``, does it per the review of 2026-11-13, not opened here; measured with kicad-cli 10.0.6 the same day: a GND via
whose copper reached only such a pocket between four foreign vias was ``via_dangling`` although a second GND via touched the
pocket and the plane - ``test_stack_drc``'s canary). Each zone is filled on its own; two zones of one net on one layer are one
copper where their fills overlap or touch along an outline edge both share (kicad-cli 10.0.6 reported no unconnected item for a
plane split round a keep-out into touching rectangles, measured 2026-11-13). Items of the plane's own net are not knocked out (the
compiler writes a solid pad connection); one is joined where the pruned fill overlaps its copper. Measured 2026-11-13 on the
STM32F407 example's 6-layer board (``kicad-cli`` 10.0.6 ``pcb drc --refill-zones --save-board``): the GND fill of In1.Cu kept a
109 mm2 island inside the LQFP-100's ring of fan-out vias, joined to the rest of the plane only by one GND via whose copper
overlaps both, and two GND vias between fan-out vias 0.825 / 1.0 mm away touched no fill at all - DRC reported them
``via_dangling`` and their pads ``unconnected_items``.

The check (:func:`plane_web`) is conservative - a via it calls joined is joined in such a fill: each plane layer is cut into rows
``step`` apart; on each row a zone's free intervals are its polygon's (shrunk by half the minimum thickness from its outline) minus
every knockout grown by half the minimum thickness, everything grown by :data:`WEB_TOLERANCE_MM` more; two intervals of
neighbouring rows that overlap belong together where a vertical segment between them keeps clear of every knockout (only a
knockout widest between the two rows can cut it). A part of a zone's intervals spanning less than the minimum thickness in x and
in y is deleted as KiCad deletes a small island (the rows see less of a part than there is and the tolerance shrinks it, so the
check deletes every island KiCad may delete, and perhaps one KiCad keeps: the safe side). Two zones' parts join where a row's
intervals of both overlap, or where both reach a seam (:func:`_seams`, an axis-parallel stretch of outline the two share) over at
least a step - their intervals ending half the minimum thickness and the tolerance from it on two neighbouring rows with a free
vertical segment between (a vertical seam), or the free intervals of each side's line that far from it, joined to the zone's
nearest row, overlapping over a step (a horizontal one). An item of the plane's net is joined to an interval that comes within its
radius plus half the minimum thickness, less the tolerance. The knockouts are the vias and through-hole pads of other nets and
their tracks on the plane layer (the maze router puts none there: a stack's plane layers are no routing layers); the plane net's
own tracks are left out - copper they would add is not counted, which keeps the check on the safe side. Items are joined across
the net's plane layers (a through via on two planes of one net joins them). The plane's *main* part is the items' group with the
most fill (then the most items, then the first item's); every item outside it, or touching no interval at all, is cut off - KiCad
would leave it dangling or in a separate copper island.

Stricter than KiCad 10 (a cut may be a false one, never a join): before it grows the fill back, KiCad also links parts of the
shrunk fill that lie closer together than the minimum thickness (``connect_nearby_polys`` in ``zone_filler.cpp`` per the review of
2026-11-13, not opened here), so its fill keeps some necks narrower than the minimum thickness that this check cuts - measured with
kicad-cli 10.0.6: a GND via among four foreign vias, cut off here at every tolerance and step tried, lay inside the main In1.Cu fill,
joined through a neck about 0.235 mm wide. The check has no such link step; the router's pass then moves a net KiCad would not
need moved, or names a via "still cut off" that KiCad joins.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: added to every knockout and taken from every reach (mm): KiCad approximates arcs by segments within a few micrometres (its
#: default maximum error, 0.005 mm), so a margin of twice that keeps the check on the safe side of the fill - a stated choice
WEB_TOLERANCE_MM = 0.01
#: the distance between the check's rows (mm): a stated choice - fine against the 0.25 mm minimum thickness, coarse enough to run
#: in about a second on a 126 mm board
WEB_STEP_MM = 0.025
#: what the check leaves out of KiCad 10's fill (module docstring), for the router's record
WEB_STRICTER = ("no link of fill parts closer than the minimum thickness (KiCad 10 links them): an item cut off here may be joined in "
                "KiCad's fill, never the other way round")


@dataclass(frozen=True, slots=True)
class PlaneItem:
    """One copper item on copper layers of the board: a via (``kind="via"``, a disc of radius ``r`` at ``(x, y)``), a through-hole
    pad (``kind="pad"``: ``box`` is its copper's bounding box - its knockout - and ``r`` the radius of its inscribed circle at
    ``(x, y)`` - the copper it surely has) or a track (``kind="track"``: from ``(x, y)`` to ``end``, ``r`` half its width; a
    knockout only). A track item is straight: a track that is an arc (a :class:`~ai_eda.ir.Track` with ``mid``) is listed as the
    chords of its chain (:func:`~ai_eda.ir.track_points`), one track item each, with ``r`` raised by the chain's bound
    (:data:`~ai_eda.ir.arc.ARC_SAGITTA_MM`), so that the knockout covers the arc - never as one item from its start to its end.
    ``net`` is ``None`` for a pad of no net (an NPTH hole: a knockout only); ``layers`` are the copper layers it is on."""

    kind: str
    label: str
    net: str | None
    x: float
    y: float
    r: float
    box: tuple[float, float, float, float]
    layers: frozenset[str]
    end: tuple[float, float] | None = None


@dataclass(slots=True)
class PlaneWeb:
    """What :func:`plane_web` found: per plane net, the indices (into the items given) of its items on its planes, of those cut off
    from the main part, and how many separate groups its items fell into; ``rows`` counts the rows checked."""

    items: dict[str, list[int]] = field(default_factory=dict)
    cut: dict[str, list[int]] = field(default_factory=dict)
    parts: dict[str, int] = field(default_factory=dict)
    rows: int = 0


def _poly_row(poly: Sequence[tuple[float, float]], y: float) -> list[tuple[float, float]]:
    """The x-intervals of the polygon's interior on the line ``y`` (even-odd rule, half-open edges)."""
    xs: list[float] = []
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 <= y < y1) or (y1 <= y < y0):
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def _disc_chord(cx: float, cy: float, radius: float, y: float) -> tuple[float, float] | None:
    dy = y - cy
    if abs(dy) >= radius:
        return None
    h = math.sqrt(radius * radius - dy * dy)
    return (cx - h, cx + h)


def _box_chord(box: tuple[float, float, float, float], radius: float, y: float) -> tuple[float, float] | None:
    """The x-range of the points of line ``y`` nearer than ``radius`` to the box (the box grown by ``radius``, rounded corners)."""
    x0, y0, x1, y1 = box
    if y0 <= y <= y1:
        return (x0 - radius, x1 + radius)
    dy = y0 - y if y < y0 else y - y1
    if dy >= radius:
        return None
    h = math.sqrt(radius * radius - dy * dy)
    return (x0 - h, x1 + h)


def _capsule_chord(a: tuple[float, float], b: tuple[float, float], radius: float, y: float) -> tuple[float, float] | None:
    """The x-range of the points of line ``y`` nearer than ``radius`` to the segment ``ab`` (a convex capsule: the interval between
    the extreme crossings of its two end discs and its two offset sides)."""
    xs: list[float] = []
    for c in (a, b):
        chord = _disc_chord(c[0], c[1], radius, y)
        if chord is not None:
            xs.extend(chord)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length > 0:
        nx, ny = -dy / length * radius, dx / length * radius
        for s in (1.0, -1.0):
            p0, p1 = (a[0] + s * nx, a[1] + s * ny), (b[0] + s * nx, b[1] + s * ny)
            if p0[1] != p1[1] and min(p0[1], p1[1]) <= y <= max(p0[1], p1[1]):
                xs.append(p0[0] + (y - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1]))
    return (min(xs), max(xs)) if xs else None


def _subtract(base: list[tuple[float, float]], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``base`` (disjoint, sorted) minus the union of ``cuts``; intervals shorter than a nanometre are dropped."""
    if not cuts:
        return base
    cuts.sort()
    merged: list[list[float]] = []
    for lo, hi in cuts:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    out: list[tuple[float, float]] = []
    for lo, hi in base:
        x = lo
        for c0, c1 in merged:
            if c1 <= x:
                continue
            if c0 >= hi:
                break
            if c0 > x:
                out.append((x, c0))
            x = max(x, c1)
        if hi > x:
            out.append((x, hi))
    return [(lo, hi) for lo, hi in out if hi - lo > 1e-6]


def _pt_seg(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _seg_seg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> float:
    """The distance between segments ``ab`` and ``cd`` (0 when they cross)."""
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
        return 0.0
    return min(_pt_seg(a, c, d), _pt_seg(b, c, d), _pt_seg(c, a, b), _pt_seg(d, a, b))


def _clear_between(x0: float, x1: float, ya: float, yb: float, band: Sequence[tuple[str, object, float]]) -> bool:
    """Whether a vertical segment from ``ya`` to ``yb`` at the middle (else a quarter) of the overlap ``[x0, x1]`` of two rows'
    intervals keeps clear of every knockout widest between the rows (``band``); every other knockout cannot cut it: its section on
    that vertical is an interval reaching a row line, where the intervals already exclude it."""
    if not band:
        return True
    for x in ((x0 + x1) / 2.0, x0 + (x1 - x0) / 4.0, x1 - (x1 - x0) / 4.0):
        ok = True
        for kind, geom, radius in band:
            if kind == "disc":
                cx, cy = geom  # type: ignore[misc]
                gap = math.hypot(x - cx, min(max(cy, ya), yb) - cy)
            elif kind == "box":
                bx0, by0, bx1, by1 = geom  # type: ignore[misc]
                gap = math.hypot(max(bx0 - x, 0.0, x - bx1), max(by0 - yb, 0.0, ya - by1))
            else:
                ea, eb = geom  # type: ignore[misc]
                gap = _seg_seg((x, ya), (x, yb), ea, eb)
            if gap < radius:
                ok = False
                break
        if ok:
            return True
    return False


def _crosses(item: PlaneItem, layer: str) -> bool:
    return layer in item.layers


def _free(base: list[tuple[float, float]], knockouts: Sequence[tuple[str, object, float]], y: float) -> list[tuple[float, float]]:
    """The free intervals of the line ``y``: ``base`` (disjoint, sorted) minus the chord of every knockout given."""
    cuts: list[tuple[float, float]] = []
    for kind, geom, radius in knockouts:
        if kind == "disc":
            chord = _disc_chord(geom[0], geom[1], radius, y)  # type: ignore[index]
        elif kind == "box":
            chord = _box_chord(geom, radius, y)  # type: ignore[arg-type]
        else:
            chord = _capsule_chord(geom[0], geom[1], radius, y)  # type: ignore[index]
        if chord is not None:
            cuts.append(chord)
    return _subtract(base, cuts)


def _inside(poly: Sequence[tuple[float, float]], x: float, y: float) -> bool:
    return any(lo < x < hi for lo, hi in _poly_row(poly, y))


def _seams(polys: Sequence[Sequence[tuple[float, float]]]) -> list[tuple[str, float, float, float, int, int]]:
    """The axis-parallel stretches of outline two of ``polys`` share with their insides on either side (a plane split round a keep-out
    into touching rectangles, :func:`~ai_eda.tools.keepout.rect_difference`): ``(axis, c, t0, t1, low, high)`` - on ``x = c`` from
    ``y = t0`` to ``t1`` (``axis="x"``) or on ``y = c`` from ``x = t0`` to ``t1`` (``"y"``), ``low`` the polygon on the side of the
    smaller coordinate, ``high`` the other. Any other contact between them is no seam (the safe side: nothing joins there)."""
    edges: list[tuple[str, float, float, float, int]] = []
    for q, poly in enumerate(polys):
        for k in range(len(poly)):
            (x0, y0), (x1, y1) = poly[k], poly[(k + 1) % len(poly)]
            if abs(x0 - x1) <= 1e-9 < abs(y0 - y1):
                edges.append(("x", x0, min(y0, y1), max(y0, y1), q))
            elif abs(y0 - y1) <= 1e-9 < abs(x0 - x1):
                edges.append(("y", y0, min(x0, x1), max(x0, x1), q))
    out: list[tuple[str, float, float, float, int, int]] = []
    for m, (axis, c, t0, t1, q) in enumerate(edges):
        for axis2, c2, u0, u1, q2 in edges[m + 1:]:
            lo, hi = max(t0, u0), min(t1, u1)
            if axis2 != axis or q2 == q or abs(c2 - c) > 1e-9 or hi - lo <= 1e-9:
                continue
            mid = (lo + hi) / 2.0

            def side(poly: Sequence[tuple[float, float]], s: float) -> bool:
                return _inside(poly, c + s, mid) if axis == "x" else _inside(poly, mid, c + s)

            pa, pb = polys[q], polys[q2]
            if side(pa, -1e-6) and not side(pa, 1e-6) and side(pb, 1e-6) and not side(pb, -1e-6):
                out.append((axis, c, lo, hi, q, q2))
            elif side(pb, -1e-6) and not side(pb, 1e-6) and side(pa, 1e-6) and not side(pa, -1e-6):
                out.append((axis, c, lo, hi, q2, q))
    return out


def plane_web(
    planes: Mapping[str, Sequence[tuple[str, Sequence[tuple[float, float]]]]], items: Sequence[PlaneItem], *, clearance: float,
    min_thickness: float, step: float = WEB_STEP_MM, tolerance: float = WEB_TOLERANCE_MM,
    clearances: Mapping[str, Sequence[float]] | None = None,
    holes: Mapping[str, Sequence[Sequence[Sequence[tuple[float, float]]]]] | None = None,
) -> PlaneWeb:
    """Which items of each plane net its fill joins (module docstring): ``planes`` maps a net to the ``(layer, polygon)`` of each of
    its plane zones, ``items`` are every via and through-hole pad of the board (any net), ``clearance`` / ``min_thickness`` the
    zones' clearance and minimum thickness (mm); ``clearances`` gives a net's zones their own clearances, one per zone in the order
    of ``planes[net]`` (``clearance`` for a net it leaves out). Pure and deterministic: the items keep their given order, the nets
    are taken in name order. ``holes`` gives each zone's interior contours in that same order; their bounding boxes are
    excluded conservatively, without adding a foreign-copper clearance to the cut-out boundary."""
    half = min_thickness / 2.0
    grow = half + tolerance
    out = PlaneWeb()
    parent: list[int] = []
    length: list[float] = []  # per node: the interval's length (an item: 0) - a group's summed length times the step is its fill

    def node(size: float = 0.0) -> int:
        parent.append(len(parent))
        length.append(size)
        return len(parent) - 1

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def at(row: Sequence[tuple[float, float, int]], x: float) -> int | None:
        return next((nd for lo, hi, nd in row if lo < x < hi), None)

    for net in sorted(planes):
        entries = list(planes[net])
        zone_clearance = list(clearances[net]) if clearances is not None and net in clearances else [clearance] * len(entries)
        if len(zone_clearance) != len(entries):
            raise ValueError(f"plane net {net!r}: {len(zone_clearance)} clearance(s) for {len(entries)} zone(s)")
        zone_holes = list(holes[net]) if holes is not None and net in holes else [[] for _ in entries]
        if len(zone_holes) != len(entries):
            raise ValueError(f"plane net {net!r}: cut-out count does not match zone count")
        layers = list(dict.fromkeys(layer for layer, _poly in entries))
        own = [i for i, it in enumerate(items) if it.net == net and it.kind != "track" and any(_crosses(it, layer) for layer in layers)]
        out.items[net] = own
        if not own:
            out.cut[net], out.parts[net] = [], 0
            continue
        parent.clear()
        length.clear()
        item_node = {i: node() for i in own}
        touched: set[int] = set()
        for layer in layers:
            zones = [(list(poly), zone_clearance[k]) for k, (pl, poly) in enumerate(entries) if pl == layer]
            polys = [poly for poly, _c in zones]
            foreign = [it for it in items if it.net != net and _crosses(it, layer)]
            mine = [(i, items[i]) for i in own if _crosses(items[i], layer)]
            y_lo = min(y for poly in polys for _x, y in poly)
            y_hi = max(y for poly in polys for _x, y in poly)
            rows = max(0, int(math.ceil((y_hi - y_lo) / step)))
            out.rows += rows
            ys = [y_lo + (j + 0.5) * step for j in range(rows)]
            # KiCad fills each zone on its own: per zone, its free intervals on every row (``grid``), and each knockout and outline
            # edge on the rows it can reach (``reach``) and (``crit``) on the band between two rows that holds the y at which it is
            # widest - only there can it cut the segment between two rows' overlapping intervals (_clear_between)
            grid: list[list[list[tuple[float, float, int]]]] = []
            reach: list[list[list[tuple[str, object, float]]]] = []
            layer_holes = [zone_holes[k] for k, (pl, _poly) in enumerate(entries) if pl == layer]
            for (poly, zc), cutouts in zip(zones, layer_holes, strict=True):
                reach_rows: list[list[tuple[str, object, float]]] = [[] for _ in range(rows)]
                crit: dict[int, list[tuple[str, object, float]]] = {}

                def put(kind: str, geom: object, radius: float, y0: float, y1: float, widest: Sequence[float]) -> None:
                    j0 = max(0, int(math.floor((y0 - y_lo) / step - 0.5)))
                    j1 = min(rows - 1, int(math.ceil((y1 - y_lo) / step - 0.5)))
                    for j in range(j0, j1 + 1):
                        reach_rows[j].append((kind, geom, radius))
                    for yw in widest:
                        crit.setdefault(int(math.floor((yw - y_lo) / step - 0.5)), []).append((kind, geom, radius))

                for hole in cutouts:
                    hx, hy = [p[0] for p in hole], [p[1] for p in hole]
                    box = (min(hx), min(hy), max(hx), max(hy))
                    put("box", box, grow, box[1] - grow, box[3] + grow, ())
                for it in foreign:
                    if it.kind == "via":
                        radius = it.r + zc + grow
                        put("disc", (it.x, it.y), radius, it.y - radius, it.y + radius, (it.y,))
                    elif it.kind == "track":  # a capsule, widest at its ends' heights (as an outline edge below)
                        a, b = (it.x, it.y), it.end or (it.x, it.y)
                        radius = it.r + zc + grow
                        put("edge", (a, b), radius, min(a[1], b[1]) - radius, max(a[1], b[1]) + radius, (a[1], b[1]))
                    else:  # a grown box is widest over its own height, which a row always crosses (it is far taller than a step)
                        radius = zc + grow
                        put("box", it.box, radius, it.box[1] - radius, it.box[3] + radius, ())
                for k in range(len(poly)):
                    a, b = poly[k], poly[(k + 1) % len(poly)]
                    put("edge", (a, b), grow, min(a[1], b[1]) - grow, max(a[1], b[1]) + grow, (a[1], b[1]))
                zone_rows: list[list[tuple[float, float, int]]] = []
                prev: list[tuple[float, float, int]] = []
                for j in range(rows):
                    y = ys[j]
                    row = [(lo, hi, node(hi - lo)) for lo, hi in _free(_poly_row(poly, y), reach_rows[j], y)]
                    band = crit.get(j - 1, ())
                    a = 0
                    for lo, hi, nd in row:  # neighbouring rows' overlapping intervals belong together where the segment between is free
                        while a < len(prev) and prev[a][1] <= lo:
                            a += 1
                        b = a
                        while b < len(prev) and prev[b][0] < hi:
                            x0, x1 = max(lo, prev[b][0]), min(hi, prev[b][1])
                            if x1 - x0 > 1e-6 and _clear_between(x0, x1, y - step, y, band):
                                union(nd, prev[b][2])
                            b += 1
                    zone_rows.append(row)
                    prev = row
                grid.append(zone_rows)
                reach.append(reach_rows)
            # KiCad deletes each zone's pruned islands whose extent is below the minimum thickness (module docstring): a part whose
            # rows span less than that in x and in y is no fill - nothing joins it, its length counts for nothing
            box: dict[int, list[float]] = {}
            for zone_rows in grid:
                for j, row in enumerate(zone_rows):
                    for lo, hi, nd in row:
                        ext = box.setdefault(find(nd), [lo, hi, j, j])
                        ext[0], ext[1], ext[3] = min(ext[0], lo), max(ext[1], hi), j
            small = {root for root, (x0, x1, j0, j1) in box.items() if max(x1 - x0, (j1 - j0) * step) < min_thickness}
            if small:
                for zone_rows in grid:
                    for j, row in enumerate(zone_rows):
                        for _lo, _hi, nd in row:
                            if find(nd) in small:
                                length[nd] = 0.0
                        zone_rows[j] = [(lo, hi, nd) for lo, hi, nd in row if find(nd) not in small]
            if len(zones) > 1:
                # zones of the net on this layer: where two fills overlap they are one copper (a row's intervals of two zones overlap)
                for j in range(rows):
                    top, top_nd = -math.inf, -1
                    for lo, hi, nd in sorted(iv for zone_rows in grid for iv in zone_rows[j]):
                        if top_nd >= 0 and lo < top - 1e-9:
                            union(nd, top_nd)
                        if hi > top:
                            top, top_nd = hi, nd
                # ... and where both reach a seam they share over at least a step (the fill reaches an outline edge where the pruned
                # fill comes within half the minimum thickness of it, as an item's copper - the touching zones KiCad 10 joined)
                for axis, c, t0, t1, qa, qb in _seams(polys):
                    if axis == "x":  # the pruned fill on both sides of x = c over two neighbouring rows, by a free vertical segment
                        xa, xb = c - grow - 1e-6, c + grow + 1e-6
                        for j in range(rows - 1):
                            if ys[j] < t0 or ys[j + 1] > t1:
                                continue
                            ends = [at(grid[q][jj], x) for q, x in ((qa, xa), (qb, xb)) for jj in (j, j + 1)]
                            if (None not in ends and _clear_between(xa, xa, ys[j], ys[j + 1], reach[qa][j] + reach[qa][j + 1])
                                    and _clear_between(xb, xb, ys[j], ys[j + 1], reach[qb][j] + reach[qb][j + 1])):
                                for nd in ends[1:]:
                                    union(ends[0], nd)  # type: ignore[arg-type]
                        continue
                    # y = c: the free intervals of each side's line half the minimum thickness (and the tolerance) from it, each joined
                    # to its zone's nearest row by a free vertical segment; two of them overlapping over a step join the zones
                    ja = int(math.floor((c - grow - 1e-6 - y_lo) / step - 0.5))
                    jb = int(math.ceil((c + grow + 1e-6 - y_lo) / step - 0.5))
                    if ja < 0 or jb >= rows or jb < 1 or ja + 1 >= rows:
                        continue
                    lines: list[list[tuple[float, float, int]]] = []
                    for q, yl, jr, near in ((qa, c - grow - 1e-6, ja, (ja, ja + 1)), (qb, c + grow + 1e-6, jb, (jb - 1, jb))):
                        kos = reach[q][near[0]] + reach[q][near[1]]
                        found: list[tuple[float, float, int]] = []
                        for lo, hi in _free(_poly_row(polys[q], yl), kos, yl):
                            lo, hi = max(lo, t0), min(hi, t1)
                            if hi - lo <= 1e-6:
                                continue
                            joined = [nd for rlo, rhi, nd in grid[q][jr] if min(hi, rhi) - max(lo, rlo) > 1e-6
                                      and _clear_between(max(lo, rlo), min(hi, rhi), min(ys[jr], yl), max(ys[jr], yl), kos)]
                            if joined:
                                for nd in joined[1:]:
                                    union(joined[0], nd)
                                found.append((lo, hi, joined[0]))
                        lines.append(found)
                    for lo, hi, nd in lines[0]:
                        for lo2, hi2, nd2 in lines[1]:
                            if min(hi, hi2) - max(lo, lo2) >= step:
                                union(nd, nd2)
            for i, it in mine:
                radius = it.r + half - tolerance
                j0 = max(0, int(math.floor((it.y - radius - y_lo) / step - 0.5)) - 1)  # a row to spare each way: the distance decides
                j1 = min(rows - 1, int(math.ceil((it.y + radius - y_lo) / step - 0.5)) + 1)
                for j in range(j0, j1 + 1):
                    y = ys[j]
                    if abs(y - it.y) > radius:
                        continue
                    for zone_rows in grid:
                        for lo, hi, nd in zone_rows[j]:
                            dx = lo - it.x if it.x < lo else (it.x - hi if it.x > hi else 0.0)
                            if math.hypot(dx, y - it.y) <= radius:
                                union(item_node[i], nd)
                                touched.add(i)
        fill: dict[int, float] = {}
        for nd in range(len(parent)):
            root = find(nd)
            fill[root] = fill.get(root, 0.0) + length[nd]
        groups: dict[int, list[int]] = {}
        for i in own:
            groups.setdefault(find(item_node[i]), []).append(i)
        main = max(groups, key=lambda root: (fill[root], len(groups[root]), -groups[root][0]))
        out.cut[net] = [i for i in own if find(item_node[i]) != main or i not in touched]
        out.parts[net] = len(groups)
    return out


__all__ = ["WEB_STEP_MM", "WEB_STRICTER", "WEB_TOLERANCE_MM", "PlaneItem", "PlaneWeb", "plane_web"]
