"""Keep-out geometry: the area of an IR keep-out, what it forbids where, and exact distance / overlap tests (board frame, mm, Y down).

Invariant: every answer here is computed from the keep-out's own recorded
numbers (``PCBDesign.keepouts``: ``rect`` ``[x, y, w, h]`` or ``polygon``
``[[x, y], ...]``, both ``Traced`` in mm; ``layers``; ``forbids``;
``allowed_refs`` / ``allowed_nets``) and the copper / footprint geometry the
caller passes - nothing is estimated. A question this module cannot answer
exactly (the overlap area of two non-convex polygons) returns ``None``, and
the caller says so instead of guessing.

The placer (:mod:`ai_eda.tools.placement.rf_floorplan`), the router
(:mod:`ai_eda.tools.routing.maze`, only when the caller hands it keep-outs),
the SI measurements (:mod:`ai_eda.tools.si.measure`), the PCB compiler (KiCad
rule areas) and the ``pcb.keepout`` check (:mod:`ai_eda.validation.layout`)
all read keep-outs through these functions, so one keep-out means the same
area to each of them. The functions read a keep-out by attribute (a value
may be ``Traced`` or plain), so they take the IR model as it is.

Conventions:

* ``layers`` names copper layers or is ``["*.Cu"]`` (every copper layer of
  the board). An exception applies only to the items it names: a ref
  (``allowed_refs``) is exempt from a ``footprints`` / ``pads`` ban, a net
  (``allowed_nets``) from a ``tracks`` / ``vias`` / ``zones`` / ``pads`` ban.
* An area *contains* its boundary: a point on an edge is inside
  (:func:`point_in_polygon`), so copper touching the area from outside is at
  distance 0 there. Two shapes *overlap* only with a positive common area
  (touching edges do not overlap); a track's copper overlaps the area when
  its centreline comes closer than half its width, a via's disc when its
  centre comes closer than its radius.
* :func:`rect_difference` subtracts boxes from a rectangle on the compressed
  grid of their edges (exact for axis-aligned boxes) and returns simple
  polygons without holes: one traced outline per connected part, or, for a
  part with a hole or a pinch point (which one simple outline cannot hold),
  that part's maximal rectangles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "ALL_COPPER",
    "KEEPOUT_ITEMS",
    "Box",
    "Point",
    "allowed_nets",
    "allowed_refs",
    "area_bbox",
    "area_is_rect",
    "area_points",
    "box_area_overlap",
    "boxes_overlap",
    "clip_convex",
    "covered_layers",
    "covers_layer",
    "forbids",
    "is_convex",
    "keepout_id",
    "keepouts_of",
    "point_area_distance",
    "point_in_polygon",
    "polygon_area",
    "polygon_overlap_area",
    "rect_difference",
    "rect_pieces",
    "segment_area_distance",
]

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2

#: what a keep-out may forbid (a KiCad rule area calls the ``zones`` ban ``copperpour``)
KEEPOUT_ITEMS: tuple[str, ...] = ("tracks", "vias", "pads", "zones", "footprints")
#: the layer name that means every copper layer of the board
ALL_COPPER = "*.Cu"
#: coordinates are KiCad's 1 nm; a distance or area this small is zero
_TOL = 1e-6
_AREA_TOL = 1e-9


def _value(x: Any) -> Any:
    """The plain value of a ``Traced`` (or the value itself)."""
    return getattr(x, "value", x)


def keepouts_of(pcb: Any) -> list[Any]:
    """``pcb.keepouts`` as a list (empty when the board has none or ``pcb`` is ``None``)."""
    if pcb is None:
        return []
    return list(getattr(pcb, "keepouts", None) or [])


def keepout_id(k: Any) -> str:
    return str(getattr(k, "id", "?"))


def area_points(k: Any) -> list[Point]:
    """The keep-out's corner points in mm (a rect as its four corners from ``(x, y)`` clockwise on screen)."""
    rect = getattr(k, "rect", None)
    if rect is not None:
        x, y, w, h = (float(v) for v in _value(rect))
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    poly = getattr(k, "polygon", None)
    if poly is None:
        raise ValueError(f"keep-out {keepout_id(k)} has neither rect nor polygon")
    return [(float(p[0]), float(p[1])) for p in _value(poly)]


def area_bbox(k_or_points: Any) -> Box:
    """``(x1, y1, x2, y2)`` of a keep-out's area (or of a point list)."""
    pts = k_or_points if isinstance(k_or_points, list) else area_points(k_or_points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def area_is_rect(k_or_points: Any) -> bool:
    """Whether the area is an axis-aligned rectangle (a ``rect``, or a 4-point polygon with axis-parallel edges)."""
    if not isinstance(k_or_points, list) and getattr(k_or_points, "rect", None) is not None:
        return True
    pts = k_or_points if isinstance(k_or_points, list) else area_points(k_or_points)
    if len(pts) != 4:
        return False
    for a, b in zip(pts, pts[1:] + pts[:1]):
        if abs(a[0] - b[0]) > _TOL and abs(a[1] - b[1]) > _TOL:
            return False
    x1, y1, x2, y2 = area_bbox(pts)
    return abs(polygon_area(pts)) >= (x2 - x1) * (y2 - y1) - _AREA_TOL


def covers_layer(k: Any, layer: str) -> bool:
    """Whether the keep-out applies on copper layer ``layer`` (``*.Cu`` covers every copper layer)."""
    layers = list(getattr(k, "layers", []) or [])
    if ALL_COPPER in layers:
        return layer.endswith(".Cu")
    return layer in layers


def covered_layers(k: Any, board_layers: Iterable[str]) -> list[str]:
    """The board's copper layers the keep-out applies on, in board order."""
    return [name for name in board_layers if name.endswith(".Cu") and covers_layer(k, name)]


def forbids(k: Any, item: str) -> bool:
    return item in (getattr(k, "forbids", None) or ())


def allowed_refs(k: Any) -> tuple[str, ...]:
    return tuple(getattr(k, "allowed_refs", None) or ())


def allowed_nets(k: Any) -> tuple[str, ...]:
    return tuple(getattr(k, "allowed_nets", None) or ())


# --------------------------------------------------------------------------- point / segment / polygon


def polygon_area(pts: Sequence[Point]) -> float:
    """The signed area of the polygon (shoelace; positive when clockwise on screen, Y down)."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, list(pts[1:]) + [pts[0]])) / 2.0


def _pt_seg(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _orient(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _seg_seg(a: Point, b: Point, c: Point, d: Point) -> float:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return 0.0
    return min(_pt_seg(c, a, b), _pt_seg(d, a, b), _pt_seg(a, c, d), _pt_seg(b, c, d))


def point_in_polygon(p: Point, pts: Sequence[Point]) -> bool:
    """Whether ``p`` lies inside the polygon or on its boundary (ray casting; the boundary within 1 nm)."""
    n = len(pts)
    for i in range(n):
        if _pt_seg(p, pts[i], pts[(i + 1) % n]) <= _TOL:
            return True
    inside = False
    x, y = p
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xc > x:
                inside = not inside
    return inside


def point_area_distance(p: Point, pts: Sequence[Point]) -> float:
    """Distance from ``p`` to the area (0 inside or on the boundary)."""
    if point_in_polygon(p, pts):
        return 0.0
    n = len(pts)
    return min(_pt_seg(p, pts[i], pts[(i + 1) % n]) for i in range(n))


def segment_area_distance(a: Point, b: Point, pts: Sequence[Point]) -> float:
    """Distance from the segment ``a``-``b`` to the area (0 when it enters, crosses or touches it)."""
    if point_in_polygon(a, pts) or point_in_polygon(b, pts):
        return 0.0
    n = len(pts)
    return min(_seg_seg(a, b, pts[i], pts[(i + 1) % n]) for i in range(n))


def boxes_overlap(a: Box, b: Box) -> bool:
    """Whether two boxes share a positive area (touching edges do not overlap)."""
    return a[0] < b[2] - _TOL and b[0] < a[2] - _TOL and a[1] < b[3] - _TOL and b[1] < a[3] - _TOL


def is_convex(pts: Sequence[Point]) -> bool:
    """Whether the polygon is convex (collinear points allowed)."""
    n = len(pts)
    sign = 0
    for i in range(n):
        o = _orient(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
        if abs(o) <= _AREA_TOL:
            continue
        s = 1 if o > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def clip_convex(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Sutherland-Hodgman: the part of ``subject`` inside the convex polygon ``clip`` (either orientation)."""
    orientation = 1.0 if polygon_area(clip) >= 0.0 else -1.0
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            break
        a, b = clip[i], clip[(i + 1) % n]

        def inside(p: Point) -> bool:
            return _orient(a, b, p) * orientation >= -_AREA_TOL

        def cross(p: Point, q: Point) -> Point:
            d1, d2 = _orient(a, b, p), _orient(a, b, q)
            t = d1 / (d1 - d2)
            return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

        src, out = out, []
        for j in range(len(src)):
            p, q = src[j], src[(j + 1) % len(src)]
            if inside(q):
                if not inside(p):
                    out.append(cross(p, q))
                out.append(q)
            elif inside(p):
                out.append(cross(p, q))
    return out


def box_area_overlap(box: Box, pts: Sequence[Point]) -> float:
    """The common area (mm^2) of an axis-aligned box and the area polygon (0 when they only touch)."""
    x1, y1, x2, y2 = box
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        return 0.0
    clipped = clip_convex(list(pts), [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    area = abs(polygon_area(clipped)) if len(clipped) >= 3 else 0.0
    return area if area > _AREA_TOL else 0.0


def polygon_overlap_area(p: Sequence[Point], q: Sequence[Point]) -> float | None:
    """The common area of two simple polygons, or ``None`` when neither is convex (not computed here)."""
    if is_convex(q):
        clipped = clip_convex(list(p), list(q))
    elif is_convex(p):
        clipped = clip_convex(list(q), list(p))
    else:
        return None
    area = abs(polygon_area(clipped)) if len(clipped) >= 3 else 0.0
    return area if area > _AREA_TOL else 0.0


# --------------------------------------------------------------------------- rectangle minus boxes


def _grid(rect: Box, holes: Sequence[Box]) -> tuple[list[float], list[float], list[list[bool]]]:
    """The compressed grid of ``rect`` cut by ``holes`` (each clipped to it): x / y breaks and ``inside[j][i]``."""
    x1, y1, x2, y2 = rect
    cut = [(max(h[0], x1), max(h[1], y1), min(h[2], x2), min(h[3], y2)) for h in holes]
    cut = [h for h in cut if h[2] - h[0] > _TOL and h[3] - h[1] > _TOL]
    xs = sorted({x1, x2, *(h[0] for h in cut), *(h[2] for h in cut)})
    ys = sorted({y1, y2, *(h[1] for h in cut), *(h[3] for h in cut)})
    xs = [x for i, x in enumerate(xs) if i == 0 or x - xs[i - 1] > _TOL]
    ys = [y for j, y in enumerate(ys) if j == 0 or y - ys[j - 1] > _TOL]
    inside = []
    for j in range(len(ys) - 1):
        cy = (ys[j] + ys[j + 1]) / 2.0
        row = []
        for i in range(len(xs) - 1):
            cx = (xs[i] + xs[i + 1]) / 2.0
            row.append(not any(h[0] < cx < h[2] and h[1] < cy < h[3] for h in cut))
        inside.append(row)
    return xs, ys, inside


def _components(inside: list[list[bool]]) -> list[list[tuple[int, int]]]:
    """4-connected parts of the inside cells, each as ``(i, j)`` cells, in row-major order of their first cell."""
    ny = len(inside)
    nx = len(inside[0]) if ny else 0
    seen = [[False] * nx for _ in range(ny)]
    parts: list[list[tuple[int, int]]] = []
    for j in range(ny):
        for i in range(nx):
            if not inside[j][i] or seen[j][i]:
                continue
            stack, part = [(i, j)], []
            seen[j][i] = True
            while stack:
                a, b = stack.pop()
                part.append((a, b))
                for da, db in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    c, d = a + da, b + db
                    if 0 <= c < nx and 0 <= d < ny and inside[d][c] and not seen[d][c]:
                        seen[d][c] = True
                        stack.append((c, d))
            parts.append(sorted(part, key=lambda t: (t[1], t[0])))
    return parts


def _rectangles(cells: set[tuple[int, int]], xs: list[float], ys: list[float]) -> list[Box]:
    """Maximal row runs of ``cells`` merged downwards where consecutive rows have the same run (deterministic)."""
    ny = len(ys) - 1
    runs: dict[int, set[tuple[int, int]]] = {}
    for j in range(ny):
        row = sorted(i for i, jj in cells if jj == j)
        out: set[tuple[int, int]] = set()
        if row:
            start = prev = row[0]
            for i in row[1:]:
                if i != prev + 1:
                    out.add((start, prev + 1))
                    start = i
                prev = i
            out.add((start, prev + 1))
        runs[j] = out
    boxes: list[Box] = []
    open_: dict[tuple[int, int], int] = {}  # run -> the row it started in
    for j in range(ny):
        for run in sorted(open_):
            if run not in runs[j]:
                boxes.append((xs[run[0]], ys[open_.pop(run)], xs[run[1]], ys[j]))
        for run in sorted(runs[j]):
            open_.setdefault(run, j)
    for run in sorted(open_):
        boxes.append((xs[run[0]], ys[open_[run]], xs[run[1]], ys[ny]))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _outline(cells: set[tuple[int, int]], xs: list[float], ys: list[float]) -> list[Point] | None:
    """The one boundary loop of a part without holes or pinch points (clockwise on screen), or ``None`` when it has more."""
    nxt: dict[tuple[int, int], list[tuple[int, int]]] = {}
    edges = 0
    for i, j in cells:
        for (di, dj), a, b in (
            ((0, -1), (i, j), (i + 1, j)),
            ((1, 0), (i + 1, j), (i + 1, j + 1)),
            ((0, 1), (i + 1, j + 1), (i, j + 1)),
            ((-1, 0), (i, j + 1), (i, j)),
        ):
            if (i + di, j + dj) not in cells:
                nxt.setdefault(a, []).append(b)
                edges += 1
    if any(len(v) != 1 for v in nxt.values()):
        return None  # a pinch point: two boundary edges leave one vertex
    start = min(nxt, key=lambda v: (v[1], v[0]))
    loop = [start]
    v = nxt[start][0]
    while v != start:
        loop.append(v)
        v = nxt[v][0]
        if len(loop) > edges:
            return None
    if len(loop) != edges:
        return None  # another loop (a hole) holds the remaining edges
    pts = [(xs[a], ys[b]) for a, b in loop]
    keep = []
    for k, p in enumerate(pts):
        before, after = pts[k - 1], pts[(k + 1) % len(pts)]
        if abs(_orient(before, p, after)) > _AREA_TOL:
            keep.append(p)
    return keep


def rect_pieces(rect: Box, holes: Sequence[Box]) -> list[Box]:
    """``rect`` minus ``holes`` as disjoint rectangles (row runs merged downwards), top to bottom, left to right."""
    xs, ys, inside = _grid(rect, holes)
    cells = {(i, j) for j, row in enumerate(inside) for i, v in enumerate(row) if v}
    return _rectangles(cells, xs, ys) if cells else []


def rect_difference(rect: Box, holes: Sequence[Box]) -> list[list[Point]]:
    """``rect`` minus ``holes`` as simple polygons without holes (module docstring), in row-major order of their parts."""
    xs, ys, inside = _grid(rect, holes)
    out: list[list[Point]] = []
    for part in _components(inside):
        cells = set(part)
        loop = _outline(cells, xs, ys)
        if loop is not None:
            out.append(loop)
        else:
            out.extend([[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])] for b in _rectangles(cells, xs, ys)])
    return out
