"""The router: wires between the pins of each net on the 1.27 mm grid.

A* over an occupancy index. Cost: 1 per cell, 4 per bend, 1000 per crossing; ties break by (estimate, cost, x, y,
direction). A search stays inside a window (the bounding box of start and goals plus :data:`WINDOW` cells, doubled up to
the sheet bounds) and expands at most ``budget`` states; it returns None when it finds no path, never raises.

The index answers each step in constant time: cells strictly inside bodies (and inside the body boxes inflated by the
halo), cells strictly inside text boxes, pin points with their nets, every grid point of every wire, the unit runs of the
axis-parallel wires per net, and the bounding-box cells of the other wires (diagonals) for an exact crossing test.

:data:`RULES_09` are exactly the rules of the 0.9 router (kept as its regression oracle). :data:`RULES_10` are the
production rules: (a) a 1-cell halo around foreign bodies, except a pin of the net and its first 2 escape cells; (b) text
boxes (in :func:`route` field text is grown by :data:`TEXT_CLEAR`, so no routed wire skims or turns at a text); (c)
foreign pins and every foreign wire point; (d) a crossing only as a straight 2-cell hop over an interior point of a
perpendicular foreign wire, in the second pass; (e) no unit step alongside a foreign wire of the same axis 1 cell away;
(f) pins left and entered along their own outward axis, so a pin's escape cells (and the runs beside them) are kept for
its net: a wire routed earlier never seals a pin routed later.

:func:`route` grows each net as a nearest-group tree: recipe hints first (already drawn), then nets with power symbols
(ground and glyph stubs), nets inside one drawing (fewest points first), nets across drawings; a power symbol or a pin /
pair label joins its group by name, so such groups are roots, never wired to each other. A crossing-free search comes
first, crossings second. A connection between drawings is a wire only when it is at most 1.5 x Manhattan + 8 cells long
and within the crossing budget. A connection that is not made leaves its net in ``Wiring.failed``; the name labels join
the stranded groups, and the notes say so. A crossing-free connection whose path must step between two rows (or columns)
and starts or ends at a pin has the step moved next to that pin (:func:`settle_step`): the wire runs straight and drops
into the pin, instead of stepping part way along where an obstacle happened to end. A test point
(:func:`~ai_eda.compilers.textbook.model.is_test_point`) is joined to its net by the label or the power symbol on its stub
alone: its pin, its stub, that label and that symbol are obstacles here, never points or wires of the net to join (a wire
across the sheet to a pad, a pad's label or stub as a group that strands the net's other groups), and they stay out of the
sheet bounds; the stubs are written as the recipe drew them.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, insort
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ai_eda.compilers.schematic_layout import GRID_MM, natural_ref_key
from ai_eda.compilers.textbook.glyphs import generated_pin, generated_side, pin_cell, pin_side, to_cells
from ai_eda.compilers.textbook.model import OUTWARD, BoxMM, Kind, LayoutFailure, Pt, Seg, Sheet, Side, Wiring, is_test_point

BEND = 4
CROSSING = 1000
BUDGET = 60_000  # expansions per search
WINDOW = 16  # cells around start and goals, doubled up to the sheet bounds
MARGIN = 12  # cells of sheet bounds beyond the content
ESCAPE = 2  # escape cells out of a pin that the halo never blocks
LINK_RATIO, LINK_SLACK = 1.5, 8
TEXT_CLEAR = 0.5  # mm a routed wire keeps from field text (a wire on a cell line 1.27 mm from text is 0.77 mm clear)
HUB_PINS = 20  # a board with an IC of this many pins gets no extra crossings for links
STEPS: tuple[Pt, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))  # direction index order (a tie-break)
TRACK = 2  # cells between fan-out tracks
STEP = 4  # cells: a run this short between two runs going on the same way is a step (see settle_step)
FLOOD = 4096  # cells a reverse flood from the goals may visit to prove them walled in (once a search has spent as many
#               expansions, or must grow its window)


@dataclass(frozen=True, slots=True)
class Rules:
    halo: int  # cells around a foreign body that are illegal (0: only strict body interiors)
    parallel: bool  # a unit step alongside a foreign wire of its axis 1 cell away is illegal
    pin_axis: bool  # pins are left and entered along their outward axis; other pins and their escape cells are never passed
    perpendicular: bool = False  # a crossing hops only over a wire perpendicular to it (never a diagonal)


RULES_09 = Rules(halo=0, parallel=False, pin_axis=False)
RULES_10 = Rules(halo=1, parallel=True, pin_axis=True, perpendicular=True)


def on(a: Pt, b: Pt, p: Pt) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) == (b[1] - a[1]) * (p[0] - a[0])
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def raster(seg: Seg) -> list[Pt]:
    (ax, ay), (bx, by) = seg
    n = math.gcd(abs(bx - ax), abs(by - ay))
    return [(ax + i * (bx - ax) // n, ay + i * (by - ay) // n) for i in range(n + 1)] if n else [seg[0]]


def intersects(a: Pt, b: Pt, c: Pt, d: Pt) -> bool:
    def cross(p: Pt, q: Pt, r: Pt) -> int:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)
            or (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0))


class _Nearest:
    """Exact Manhattan distance from a point to a growing point set (rows of sorted x)."""

    def __init__(self, points: Iterable[Pt] = ()) -> None:
        self.rows: dict[int, list[int]] = {}
        self.ys: list[int] = []
        for p in points:
            self.add(p)

    def add(self, p: Pt) -> None:
        xs = self.rows.get(p[1])
        if xs is None:
            self.rows[p[1]] = [p[0]]
            insort(self.ys, p[1])
        else:
            i = bisect_left(xs, p[0])
            if i == len(xs) or xs[i] != p[0]:
                xs.insert(i, p[0])

    def distance(self, p: Pt) -> float:
        best, ys = math.inf, self.ys
        hi = bisect_left(ys, p[1])
        lo = hi - 1
        while lo >= 0 or hi < len(ys):
            if hi >= len(ys) or (lo >= 0 and p[1] - ys[lo] <= ys[hi] - p[1]):
                y, lo = ys[lo], lo - 1
            else:
                y, hi = ys[hi], hi + 1
            dy = abs(p[1] - y)
            if dy >= best:
                break
            xs = self.rows[y]
            j = bisect_left(xs, p[0])
            dx = min(xs[j] - p[0] if j < len(xs) else math.inf, p[0] - xs[j - 1] if j else math.inf)
            best = min(best, dy + dx)
        return best


class Occupancy:
    """What a wire of one net may not touch, indexed by grid point (see the module text)."""

    def __init__(self) -> None:
        self.cover: dict[Pt, list[tuple[str | None, bool]]] = {}  # cell -> (body key, or None for text; strictly in the box)
        self.nets_of: dict[str, frozenset[str]] = {}
        self.pins: dict[Pt, frozenset[str]] = {}
        self.side: dict[Pt, Side] = {}
        self.escape: dict[Pt, set[str]] = {}
        self.seg_at: list[Seg] = []
        self.seg_net: list[str] = []
        self.wire: dict[Pt, list[int]] = {}  # grid point -> ids of the wires through it
        self.diag: dict[Pt, list[int]] = {}  # bounding-box cell -> ids of the non-axis wires
        self.busy: set[int] = set()  # keys of every point in wire or diag
        self.beside: dict[int, set[str]] = {}  # unit run (point key * 2 + axis) -> nets running alongside it 1 cell away

    @classmethod
    def build(cls, bodies: Mapping[str, BoxMM], texts: Iterable[BoxMM], pins: Mapping[Pt, str | None | Iterable[str]],
              escapes: Mapping[Pt, Side], *, nets_of: Mapping[str, Iterable[str]] | None = None, halo: int = 1,
              scale: float = GRID_MM) -> Occupancy:
        """``bodies`` keyed by glyph key (``nets_of`` gives each key's nets; a body without nets is foreign to all),
        ``texts`` boxes, ``pins`` point -> net (None: unconnected), ``escapes`` pin point -> side. Boxes are in mm
        (``scale`` mm per cell; 1.0 for boxes already in cells). ``halo`` is the widest halo a search may then use."""
        occ = cls()
        for key in sorted(bodies):
            occ._cover(key, bodies[key], halo, scale)
        for box in texts:
            occ._cover(None, box, 0, scale)
        occ.nets_of = {k: frozenset(v) for k, v in (nets_of or {}).items()}
        for pt, nets in pins.items():
            occ.pins[pt] = frozenset(() if nets is None else (nets,) if isinstance(nets, str) else nets)
        for pt, side in escapes.items():
            occ.side[pt] = side
            ox, oy = OUTWARD[side]
            nets = occ.pins.get(pt, frozenset())
            for k in range(ESCAPE + 1):
                occ.escape.setdefault((pt[0] + ox * k, pt[1] + oy * k), set()).update(nets)
            if nets:  # the escape run counts as a wire of the pin's nets for the parallel rule (no foreign run beside it)
                axis = 0 if oy == 0 else 1
                for k in range(ESCAPE):
                    x, y = min((pt[0] + ox * k, pt[1] + oy * k), (pt[0] + ox * (k + 1), pt[1] + oy * (k + 1)))
                    for nb in (((x, y - 1), (x, y + 1)) if axis == 0 else ((x - 1, y), (x + 1, y))):
                        occ.beside.setdefault(_key(nb) << 1 | axis, set()).update(nets)
        return occ

    def _cover(self, key: str | None, box: BoxMM, h: int, scale: float) -> None:
        x0, y0, x1, y1 = (v / scale for v in box)
        for x in range(math.floor(x0 - h) + 1, math.ceil(x1 + h)):
            inside = x0 < x < x1
            for y in range(math.floor(y0 - h) + 1, math.ceil(y1 + h)):
                self.cover.setdefault((x, y), []).append((key, inside and y0 < y < y1))

    def add_segment(self, net: str, seg: Seg) -> None:
        (ax, ay), (bx, by) = seg
        i = len(self.seg_at)
        self.seg_at.append(seg)
        self.seg_net.append(net)
        pts = raster(seg)
        for p in pts:
            self.wire.setdefault(p, []).append(i)
        self.busy.update(_key(p) for p in pts)
        if ax != bx and ay != by:
            for x in range(min(ax, bx), max(ax, bx) + 1):
                for y in range(min(ay, by), max(ay, by) + 1):
                    self.diag.setdefault((x, y), []).append(i)
                    self.busy.add(_key((x, y)))
            return
        axis = 0 if ay == by else 1
        for u, v in zip(pts, pts[1:]):
            x, y = min(u, v)
            for nb in (((x, y - 1), (x, y + 1)) if axis == 0 else ((x - 1, y), (x + 1, y))):
                self.beside.setdefault(_key(nb) << 1 | axis, set()).add(net)

    def free(self, q: Pt, net: str, rules: Rules) -> bool:
        """May a wire of ``net`` pass grid point ``q`` (not a goal)?"""
        nets = self.pins.get(q)
        if nets is not None and (rules.pin_axis or net not in nets):
            return False
        escape = self.escape.get(q)
        if escape and rules.pin_axis and net not in escape:
            return False  # a pin is left and entered only along its axis: its escape cells are kept for its own net
        for key, inside in self.cover.get(q, ()):
            if key is None:
                return False
            own = net in self.nets_of.get(key, ())
            exempt = escape is not None and net in escape
            if (inside and not (own and exempt)) or (not inside and rules.halo and not own and not exempt):
                return False
        return True

    def step(self, p: Pt, q: Pt, mid: Pt | None, net: str, perpendicular: bool = False) -> int:
        """-1 when the step p -> q (a 2-cell hop over ``mid``) is illegal for ``net``; else the foreign wires it crosses
        (``perpendicular``: only wires square to the hop may be crossed)."""
        w, d = self.wire, self.diag
        pts = (p, q) if mid is None else (p, mid, q)
        if not any(pt in w or pt in d for pt in pts):
            return 0 if mid is None else -1
        ids = set()
        for pt in pts:
            ids.update(i for i in w.get(pt, ()) if self.seg_net[i] != net)
            ids.update(i for i in d.get(pt, ()) if self.seg_net[i] != net and intersects(p, q, *self.seg_at[i]))
        if mid is None:
            return -1 if ids else 0
        for i in ids:
            a, b = self.seg_at[i]
            if mid in (a, b) or not on(a, b, mid) or on(a, b, p) or on(a, b, q):
                return -1
            if perpendicular and (a[0] - b[0]) * (p[0] - q[0]) + (a[1] - b[1]) * (p[1] - q[1]) != 0:
                return -1
        return len(ids) or -1

    def alongside(self, p: Pt, direction: int, length: int, net: str) -> bool:
        """Does a unit run of the step lie 1 cell beside a foreign wire of the same axis?"""
        dx, dy = STEPS[direction]
        for k in range(length):
            a = (p[0] + dx * k, p[1] + dy * k)
            nets = self.beside.get(_key(min(a, (a[0] + dx, a[1] + dy))) << 1 | (dy != 0), ())
            if any(n != net for n in nets):
                return True
        return False


_Y, _O = 20, 1 << 19  # bits of y in a point key, and the offset that keeps both coordinates non-negative


def _key(p: Pt) -> int:
    return ((p[0] + _O) << _Y) | (p[1] + _O)


def _point(k: int) -> Pt:
    return (k >> _Y) - _O, (k & ((1 << _Y) - 1)) - _O


def _windows(start: Pt, goals: Iterable[Pt], bounds: tuple[int, int, int, int], margin: int | None):
    if margin is None:
        yield bounds
        return
    xs = [start[0], *(g[0] for g in goals)]
    ys = [start[1], *(g[1] for g in goals)]
    m = margin
    while True:
        w = (max(bounds[0], min(xs) - m), max(bounds[1], min(ys) - m), min(bounds[2], max(xs) + m), min(bounds[3], max(ys) + m))
        yield w
        if w == bounds:
            return
        m *= 2


def search(start: Pt, goals: Iterable[Pt], occ: Occupancy, net: str, rules: Rules, *, window: tuple[int, int, int, int],
           budget: int | None = BUDGET, allow_crossings: bool = False, margin: int | None = WINDOW) -> list[Pt] | None:
    """The cheapest path from ``start`` to any goal (cells, hop ends only) inside ``window`` = (x0, y0, x1, y1); None when
    there is none or ``budget`` expansions are spent. The search starts in the bounding box of start and goals plus
    ``margin`` cells (None: the whole window) and, when that is exhausted with moves cut off by its edge, doubles the
    margin and carries on from the states those moves left. Goals are exempt from the cell rules."""
    goals = goals if isinstance(goals, (set, frozenset)) else set(goals)
    if start in goals:
        return [start]
    if not goals:
        return None
    near = _Nearest(goals)
    free, step, busy, side, pins, beside = occ.free, occ.step, occ.busy, occ.side, occ.pins, occ.beside
    # points as ints (x, y order kept), states as point * 8 + direction + 1: (estimate, cost, state) ties as (x, y, direction)
    targets = {_key(g) for g in goals}
    moves = []  # (direction, length, point offset, middle offset, dx, dy, unit-run offsets, axis)
    for d, (dx, dy) in enumerate(STEPS):
        half = (dx << _Y) + dy
        for n in (1, 2) if allow_crossings else (1,):
            runs = tuple(half * i if dx + dy > 0 else half * (i + 1) for i in range(n))
            moves.append((d, n, half * n, half, dx, dy, runs, int(dy != 0)))
    first = STEPS.index(OUTWARD[side[start]]) if rules.pin_axis and start in side and net in pins.get(start, ()) else None
    # a step back the way it came is never cheaper than not having come (a U-turn costs a bend and two cells)
    moves_after = {came: [m for m in moves if m[0] != (came + 2) % 4] for came in range(4)}
    moves_after[-1] = [m for m in moves if first is None or m[0] == first]
    pin_axis, parallel, perpendicular, inf, push, pop = rules.pin_axis, rules.parallel, rules.perpendicular, math.inf, heapq.heappush, heapq.heappop
    windows = _windows(start, goals, window, margin)
    x0, y0, x1, y1 = next(windows)
    bx0, by0, bx1, by1 = window
    legal: dict[Pt, bool] = {}  # cell rules, window aside
    estimate: dict[int, float] = {}
    info: dict[int, float] = {}  # point -> estimate (0 for a goal); -1 never usable, -2 outside the current window only
    outside: list[int] = []
    edge: set[int] = set()  # states with a move the current window cut off

    def look(k: int) -> float:
        q = _point(k)
        if k not in targets:
            if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1):
                if bx0 <= q[0] <= bx1 and by0 <= q[1] <= by1:
                    outside.append(k)
                    return -2
                return -1
            ok = legal.get(q)
            if ok is None:
                ok = legal[q] = free(q, net, rules)
            if not ok:
                return -1
        e = estimate.get(k)
        if e is None:
            e = estimate[k] = near.distance(q)
        return e

    initial = _key(start) << 3
    queue = [(near.distance(start), 0, initial)]
    costs, parent = {initial: 0}, {}
    info_get, cost_get = info.get, costs.get
    used = 0
    flooded = False
    while True:
        while queue:
            _, cost, state = pop(queue)
            if cost != costs[state]:
                continue
            pk, came = state >> 3, (state & 7) - 1
            if pk in targets:
                path = [_point(pk)]
                while state in parent:
                    state = parent[state]
                    path.append(_point(state >> 3))
                return path[::-1]
            used += 1
            if budget is not None and used > budget:
                return None
            if used == FLOOD and not flooded:  # a long search: are the goals walled in?
                flooded = True
                if _sealed(start, goals, occ, net, rules, window, allow_crossings, first):
                    return None
            p_busy = pk in busy
            for direction, length, delta, half, dx, dy, runs, axis in moves_after[came]:
                if length == 2 and pk + half not in busy:
                    continue
                qk = pk + delta
                e = info_get(qk)
                if e is None:
                    e = info[qk] = look(qk)
                if e < 0:
                    if e == -2:
                        edge.add(state)
                    continue
                if e == 0 and pin_axis:
                    q = _point(qk)
                    if q in side and net in pins.get(q, ()) and OUTWARD[side[q]] != (-dx, -dy):
                        continue
                if length == 1:
                    hits = 0
                    if (p_busy or qk in busy) and step(_point(pk), _point(qk), None, net):
                        continue
                else:
                    mid = _point(pk + half)
                    hits = step(_point(pk), _point(qk), mid, net, perpendicular)
                    if hits < 0:
                        continue
                    ok = legal.get(mid)
                    if ok is None:
                        ok = legal[mid] = free(mid, net, rules)
                    if not ok:
                        continue
                if parallel:
                    nets = beside.get((pk + runs[0]) << 1 | axis)
                    if nets and (net not in nets or len(nets) > 1):
                        continue
                    if length == 2:
                        nets = beside.get((pk + runs[1]) << 1 | axis)
                        if nets and (net not in nets or len(nets) > 1):
                            continue
                nxt = (qk << 3) | (direction + 1)
                value = cost + length + CROSSING * hits + (BEND if came != direction and came != -1 else 0)
                if value < cost_get(nxt, inf):
                    costs[nxt] = value
                    parent[nxt] = state
                    push(queue, (value + e, value, nxt))
        if not edge:
            return None  # exhausted, and a larger window reaches nothing more
        if not flooded:
            flooded = True
            if _sealed(start, goals, occ, net, rules, window, allow_crossings, first):
                return None
        x0, y0, x1, y1 = next(windows)
        for k in outside:
            info.pop(k, None)
        outside.clear()
        for state in sorted(edge):
            push(queue, (costs[state] + estimate.get(state >> 3, 0) if state != initial else near.distance(start), costs[state], state))
        edge.clear()


def _sealed(start: Pt, goals: set[Pt], occ: Occupancy, net: str, rules: Rules, window: tuple[int, int, int, int],
            crossings: bool, first: int | None) -> bool:
    """True when no move sequence leads from ``start`` into the goals: a flood backward from the goals, under the same
    step rules, ends without meeting the start and within :data:`FLOOD` cells (the answer does not depend on the order)."""
    x0, y0, x1, y1 = window
    seen = set(goals)
    todo = sorted(goals)
    while todo:
        q = todo.pop()
        entry = OUTWARD[occ.side[q]] if rules.pin_axis and q in goals and q in occ.side and net in occ.pins.get(q, ()) else None
        for d, (dx, dy) in enumerate(STEPS):
            if entry is not None and entry != (-dx, -dy):
                continue
            for n in (1, 2) if crossings else (1,):
                p = (q[0] - dx * n, q[1] - dy * n)  # the move p -> q goes in direction d
                if p in seen:
                    continue
                if p == start:
                    if first is not None and d != first:
                        continue
                elif not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and occ.free(p, net, rules)):
                    continue
                mid = (p[0] + dx, p[1] + dy) if n == 2 else None
                hits = occ.step(p, q, mid, net, rules.perpendicular)
                if hits != 0 if mid is None else (hits < 0 or not occ.free(mid, net, rules)):
                    continue
                if rules.parallel and occ.alongside(p, d, n, net):
                    continue
                if p == start:
                    return False
                seen.add(p)
                todo.append(p)
                if len(seen) > FLOOD:
                    return False
    return True


def simplify(path: list[Pt]) -> list[Seg]:
    """The path's straight runs as segments."""
    out = []
    a = path[0]
    for i in range(1, len(path) - 1):
        if not on(a, path[i + 1], path[i]):
            out.append((a, path[i]))
            a = path[i]
    if a != path[-1]:
        out.append((a, path[-1]))
    return out


def _pin_end(p: Pt, occ: Occupancy, net: str) -> bool:
    return p in occ.side and net in occ.pins.get(p, ())


def _legal(path: list[Pt], goals: set[Pt], occ: Occupancy, net: str, rules: Rules) -> bool:
    """Does ``path`` (unit moves only) obey the search's rules: free cells between its ends, no foreign wire touched, no
    unit run beside a foreign wire of its axis, pins left and entered along their axis, no goal before the last point?"""
    for i, (p, q) in enumerate(zip(path, path[1:])):
        d = (q[0] - p[0], q[1] - p[1])
        if abs(d[0]) + abs(d[1]) != 1:
            return False
        last = i == len(path) - 2
        if not last and (q in goals or not occ.free(q, net, rules)):
            return False
        if rules.pin_axis and ((i == 0 and _pin_end(p, occ, net) and OUTWARD[occ.side[p]] != d)
                               or (last and _pin_end(q, occ, net) and OUTWARD[occ.side[q]] != (-d[0], -d[1]))):
            return False
        if occ.step(p, q, None, net) != 0:
            return False
        if rules.parallel and occ.alongside(p, STEPS.index(d), 1, net):
            return False
    return True


def settle_step(path: list[Pt], goals: set[Pt], occ: Occupancy, net: str, rules: Rules) -> list[Pt]:
    """``path`` with a step it cannot avoid moved to its pin. A step is a run of at most :data:`STEP` cells between two
    runs that go on the same way (a Z); when exactly one of those two runs ends at a pin of the net (the path's first or
    last run), the step slides along it to the pin, as close as the rules allow (1 cell from the pin end at best): the
    wire then runs straight and turns into the pin there, like a drop from a line to an input, instead of stepping
    part way along. Same length, same bends, so the search's cost is unchanged; any other path is returned as is."""
    corners = [path[0], *(b for _, b in simplify(path)[:-1]), path[-1]]
    if len(corners) < 4:
        return path
    for reverse in (False, True):
        c = corners[::-1] if reverse else corners
        a, s, b = (c[0], c[1]), (c[1], c[2]), (c[2], c[3])
        u = (_sign(a[1][0] - a[0][0]), _sign(a[1][1] - a[0][1]))
        if (_sign(b[1][0] - b[0][0]), _sign(b[1][1] - b[0][1])) != u or _length(s) > STEP or u[0] * (s[1][0] - s[0][0]) + u[1] * (s[1][1] - s[0][1]):
            continue  # not a Z step: the runs either side go different ways, or the middle run is long
        if not _pin_end(c[0], occ, net) or (len(c) == 4 and _pin_end(c[3], occ, net)):
            continue  # its first run does not start at a pin, or the step lies between two pins (no end to prefer)
        step = (s[1][0] - s[0][0], s[1][1] - s[0][1])
        for k in range(1, _length(a)):
            c1 = (c[0][0] + u[0] * k, c[0][1] + u[1] * k)
            moved = [c[0], c1, (c1[0] + step[0], c1[1] + step[1]), *c[3:]]
            cells = [p for seg in zip(moved, moved[1:]) for p in raster(seg)[1:]]
            new = [moved[0], *cells]
            new = new[::-1] if reverse else new
            if _legal(new, goals, occ, net, rules):
                return new
        return path
    return path


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _length(seg: Seg) -> int:
    return abs(seg[1][0] - seg[0][0]) + abs(seg[1][1] - seg[0][1])


def normalize(segments: Iterable[Seg], pin_points: Iterable[Pt] = ()) -> tuple[list[Seg], list[Pt]]:
    """One net's wires split at its own T joints, each piece once (sorted); junction dots where 3 or more pieces meet, or
    2 or more at a pin point."""
    segments = list(segments)
    ends = {p for seg in segments for p in seg}
    pieces = set()
    for a, b in segments:
        cuts = sorted((p for p in ends if on(a, b, p)), key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
        pieces.update(tuple(sorted((p, q))) for p, q in zip(cuts, cuts[1:]) if p != q)
    degree: dict[Pt, set[Pt]] = {}
    for a, b in pieces:
        degree.setdefault(a, set()).add(b)
        degree.setdefault(b, set()).add(a)
    pins = set(pin_points)
    return sorted(pieces), sorted(p for p, adjacent in degree.items() if len(adjacent) >= 3 or (p in pins and len(adjacent) >= 2))


def channel_fanout(pairs: Iterable[tuple[str, int, int]], x0: int, x1: int) -> dict[str, list[Seg]]:
    """Wires for a monotone matching ``(net, y at x0, y at x1)`` across the channel x0 < x < x1: straight when level,
    else across, along a track, across. Tracks are :data:`TRACK` cells apart from x1 - 2 leftward; a wire takes the first
    track left of every same-direction wire whose vertical run it meets (downward wires top first, upward wires bottom
    first), so no two wires cross or touch; with both sides' rows on the 2-cell pin pitch no two nets run 1 cell apart.
    LayoutFailure when the matching is not monotone or the channel is too narrow."""
    pairs = sorted(pairs, key=lambda t: t[1])
    if [t[2] for t in pairs] != sorted(t[2] for t in pairs) or len({t[2] for t in pairs}) != len(pairs) or len({t[1] for t in pairs}) != len(pairs):
        raise LayoutFailure("fan-out: the matching is not monotone")
    track: dict[int, int] = {}
    for down in (True, False):
        order = [i for i, t in enumerate(pairs) if (t[2] > t[1]) == down and t[1] != t[2]]
        placed: list[tuple[int, int, int]] = []  # (lo, hi, track)
        for i in (order if down else order[::-1]):
            lo, hi = sorted(pairs[i][1:])
            track[i] = 1 + max((k for a, b, k in placed if a <= hi + 1 and lo <= b + 1), default=-1)
            placed.append((lo, hi, track[i]))
    need = 1 + max(track.values(), default=-1)
    if x1 - TRACK * need < x0 + TRACK:
        raise LayoutFailure(f"fan-out: {need} track(s) do not fit between x {x0} and {x1}")
    out: dict[str, list[Seg]] = {}
    for i, (net, ya, yb) in enumerate(pairs):
        if ya == yb:
            out.setdefault(net, []).append(((x0, ya), (x1, ya)))
            continue
        t = x1 - TRACK * (track[i] + 1)
        out.setdefault(net, []).extend((((x0, ya), (t, ya)), ((t, ya), (t, yb)), ((t, yb), (x1, yb))))
    return out


# --------------------------------------------------------------------------- the sheet router


def _groups(segs: list[Seg], points: Iterable[Pt]) -> list[set[Pt]]:
    """Point sets joined by shared grid points (each wire's points, each pin / glyph / label point)."""
    merged: list[set[Pt]] = []
    for group in [set(raster(s)) for s in segs] + [{p} for p in sorted(points)]:
        while (touching := next((other for other in merged if group & other), None)) is not None:
            group |= touching
            merged.remove(touching)
        merged.append(group)
    return merged


def route(ctx, sheet: Sheet, obstacles: list[tuple[str, BoxMM]]) -> Wiring:
    """Draw the hints and wire links of ``sheet``, then route every net that is not joined yet (see the module text).
    ``obstacles`` are the sheet's measured boxes (``body:`` / ``glyph:`` bodies, the rest text)."""
    model = ctx.model
    budget = getattr(ctx, "budget", BUDGET)
    places = sheet.places()
    scope_of = sheet.drawing_of()
    pins: dict[Pt, set[str]] = {}
    escapes: dict[Pt, Side] = {}
    points: dict[str, dict[Pt, set[int]]] = {}  # net -> point -> drawings
    named: dict[str, set[Pt]] = {}  # points that join their group by name
    symbols: dict[str, set[Pt]] = {}  # the power symbol pins among them
    nets_of: dict[str, set[str]] = {}

    def point(at: Pt, net: str | None, side: Side | None, scope: int | None) -> None:
        pins.setdefault(at, set())
        if side is not None:
            escapes.setdefault(at, side)
        if net is not None:
            pins[at].add(net)
            points.setdefault(net, {}).setdefault(at, set()).update(() if scope is None else (scope,))

    pads: set[Pt] = set()  # the test points' pin points: joined by their labels, obstacles only (module text)
    for key in sorted(places, key=natural_ref_key):
        g, pl = model.glyphs[key], places[key]
        nets_of[key] = set(g.net.values())
        pad = is_test_point(g)
        for p in g.pins:
            at = pin_cell(g, p, (pl.x, pl.y), pl.orient)
            if pad:
                pads.add(at)
            point(at, None if pad else g.net.get(p.number), pin_side(g, p, pl.orient), scope_of.get(key))
    # the test points' drawings: their stubs and power symbols are drawn as the recipe left them, never wired (module text)
    pad_drawings = {i for i, d in enumerate(sheet.drawings) if d.place and all(is_test_point(model.glyphs[k]) for k in d.place)}
    pad_wires = [(net, ((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy))) for i, (d, (dx, dy)) in enumerate(zip(sheet.drawings, sheet.offset))
                 if i in pad_drawings for net, segs in d.hints.items() for a, b in segs]
    pad_symbols = {gen.ref for gen in sheet.generated() if scope_of.get(gen.ref) in pad_drawings}
    for gen in sorted(sheet.generated(), key=lambda gen: natural_ref_key(gen.ref)):
        nets_of[gen.ref] = {gen.net}
        if gen.ref in pad_symbols:
            point(generated_pin(gen), None, generated_side(gen), None)
            continue
        point(generated_pin(gen), gen.net, generated_side(gen), scope_of.get(gen.ref))
        if gen.names_net:
            named.setdefault(gen.net, set()).add(generated_pin(gen))
            symbols.setdefault(gen.net, set()).add(generated_pin(gen))
    stubs: dict[str, list[Seg]] = {}
    pad_stubs: list[tuple[str, Seg]] = []
    for lab in sheet.labels():
        if lab.stub is not None and lab.stub[0] in pads:
            pad_stubs.append((lab.net, lab.stub))
            continue
        point(lab.at, lab.net, None, None)
        if lab.kind in ("pin", "pair"):
            named.setdefault(lab.net, set()).add(lab.at)
        if lab.stub is not None and lab.stub[0] != lab.stub[1]:
            stubs.setdefault(lab.net, []).append(lab.stub)
    # a pad's stub and its symbol's are drawn as they are (a recipe's stub is also a hint of its drawing): no wire of the net
    # to join, or the router takes each for a group of its own and wires it across the sheet or strands the net's other groups
    pad_wire_set = set(pad_wires)
    drawn = {net: [s for s in segs if (net, s) not in pad_wire_set] for net, segs in sorted(sheet.hints().items())}
    drawn = {net: segs for net, segs in drawn.items() if segs}
    bodies = {tag.split(":", 1)[1]: box for tag, box in obstacles if tag.startswith(("body:", "glyph:"))}
    # field text (a part's or a power symbol's) keeps TEXT_CLEAR mm of clearance, a hard rule like the body halo; a label's
    # box is kept as is (its own wire ends at its anchor)
    texts = [_grown(box, TEXT_CLEAR) if tag.startswith(("field:", "gfield:")) else box
             for tag, box in obstacles if not tag.startswith(("body:", "glyph:"))]
    occ = Occupancy.build(bodies, texts, pins, escapes, nets_of=nets_of, halo=RULES_10.halo)
    for net in sorted(set(drawn) | set(stubs)):
        for s in drawn.get(net, []) + stubs.get(net, []):
            occ.add_segment(net, s)
    for net, s in pad_stubs + pad_wires:
        occ.add_segment(net, s)
    # the sheet bounds hold what is to be wired: a test point's body, fields, pin and label are obstacles, not content (its strip
    # under the drawing would widen every search window that reaches the bounds)
    pad_keys = {key for key in places if is_test_point(model.glyphs[key])}
    pad_labels = {f"label:{net}@{s[1][0]},{s[1][1]}" for net, s in pad_stubs}
    content = ([to_cells(box) for tag, box in obstacles if tag.split(":", 1)[1] not in pad_keys | pad_symbols and tag not in pad_labels]
               + [(*p, *p) for p in pins if p not in pads] + [(*p, *p) for segs in drawn.values() for s in segs for p in s])
    bounds = (max(1, min(b[0] for b in content) - MARGIN), max(1, min(b[1] for b in content) - MARGIN),
              max(b[2] for b in content) + MARGIN, max(b[3] for b in content) + MARGIN) if content else (1, 1, 1, 1)
    hub = any(len(g.pins) >= HUB_PINS and g.kind is Kind.IC for g in model.glyphs.values())
    state = {"crossings": 0, "link_crossings": 0, "link_budget": 0 if hub else 1}
    failed: dict[str, tuple[frozenset[Pt], ...]] = {}

    glyph_nets = {gen.net for gen in sheet.generated()}

    def order(net: str) -> tuple:
        scopes = set().union(*points.get(net, {}).values())
        return (0 if net in glyph_nets else 1 if len(scopes) <= 1 else 2, len(points.get(net, ())), net)

    for net in sorted(set(points) | set(drawn), key=order):
        own = points.get(net, {})
        groups = _groups(drawn.get(net, []) + stubs.get(net, []), own)
        roots = [g for g in groups if g & named.get(net, set())]
        rest = sorted((g for g in groups if not g & named.get(net, set())), key=min)
        if not rest or (not roots and len(rest) == 1):
            continue

        def scope(g: set[Pt]) -> set[int]:
            return set().union(*(own.get(p, ()) for p in g))

        tree_groups = roots or [rest.pop(0)]
        tree: set[Pt] = set().union(*tree_groups)
        at = {p: scope(g) for g in tree_groups for p in g}
        # a group named by a label only takes wires from its own drawing (power symbols take any)
        closed = {p: scope(g) for g in roots if not g & symbols.get(net, set()) for p in g}
        stranded, reason = _grow(net, tree, at, [(g, scope(g)) for g in rest], occ, bounds, budget, drawn, state, closed)
        if stranded:
            comps = stranded + ([] if roots else [tree])
            failed[net] = tuple(frozenset(c) for c in comps)
            ctx.notes.append(f"net {net}: {len(comps)} group(s) joined by labels ({reason})")
    for link in sorted((lk for lk in sheet.links if lk.kind == "wire" and not lk.segs), key=lambda lk: (lk.net, lk.a, lk.b)):
        groups = _groups(drawn.get(link.net, []) + stubs.get(link.net, []), [link.a, link.b, *points.get(link.net, {})])
        ga = next(g for g in groups if link.a in g)
        if link.b in ga:
            continue
        gb = next(g for g in groups if link.b in g)
        stranded, reason = _grow(link.net, set(ga), {p: {0} for p in ga}, [(gb, {1})], occ, bounds, budget, drawn, state)
        if stranded:
            known = failed.get(link.net, ())  # a group the net's own routing stranded is recorded (and noted) once
            failed[link.net] = (*known, *(g for g in (frozenset(ga), frozenset(gb)) if g not in known))
            ctx.notes.append(f"net {link.net}: 2 group(s) joined by labels (link {link.a}-{link.b}: {reason})")
    segments: dict[str, list[Seg]] = {}
    junctions: dict[str, list[Pt]] = {}
    pad_stubs_of: dict[str, list[Seg]] = {}
    for net, s in pad_stubs:
        pad_stubs_of.setdefault(net, []).append(s)
    for net, segs in sorted(drawn.items()):
        keep = set(points.get(net, {})) | {p for s in stubs.get(net, ()) for p in raster(s)}
        pieces, dots = normalize(_trim(normalize(segs)[0], keep), points.get(net, {}))
        # a piece inside a label stub is written with the label (whole), so it is not repeated here
        segments[net] = [s for s in pieces if not any(on(a, b, s[0]) and on(a, b, s[1]) for a, b in stubs.get(net, ()))]
        if dots:
            junctions[net] = dots
    for net, s in pad_wires:  # drawn as the recipe left them (a label's stub is written with its label)
        if not any(on(a, b, s[0]) and on(a, b, s[1]) for a, b in pad_stubs_of.get(net, ())):
            segments.setdefault(net, []).append(s)
    return Wiring(segments=segments, junctions=junctions, failed=failed, crossings=state["crossings"])


def _grown(box: BoxMM, by: float) -> BoxMM:
    return box[0] - by, box[1] - by, box[2] + by, box[3] + by


def _trim(pieces: list[Seg], keep: set[Pt]) -> list[Seg]:
    """Drop, repeatedly, a piece with an end that meets no other piece and no point of ``keep`` (a dangling wire end;
    for example the rest of a pin stub that a wire joined part way along)."""
    while True:
        ends: dict[Pt, int] = {}
        for s in pieces:
            for p in s:
                ends[p] = ends.get(p, 0) + 1
        live = [s for s in pieces if all(ends[p] > 1 or p in keep for p in s)]
        if len(live) == len(pieces):
            return pieces
        pieces = live


def _grow(net: str, tree: set[Pt], at: dict[Pt, set[int]], pending: list[tuple[set[Pt], set[int]]], occ: Occupancy,
          bounds: tuple[int, int, int, int], budget: int, drawn: dict[str, list[Seg]], state: dict,
          closed: dict[Pt, set[int]] | None = None) -> tuple[list[set[Pt]], str]:
    """Attach each pending group to ``tree``, nearest first: crossing-free first, then with crossings; a connection
    between drawings only within the link rule; a ``closed`` point (of a label-named group) only from its drawings.
    Returns the stranded groups and why."""
    near = _Nearest(tree)
    stranded: list[set[Pt]] = []
    reasons: list[str] = []
    while pending:
        _, start, i = min((near.distance(p), p, i) for i, (g, _) in enumerate(pending) for p in g)
        group, scope = pending.pop(i)
        goals = {p for p in tree if p not in closed or closed[p] & scope} if closed else tree
        path = search(start, goals, occ, net, RULES_10, window=bounds, budget=budget) if goals else None
        if path is None and goals:
            path = search(start, goals, occ, net, RULES_10, window=bounds, budget=budget, allow_crossings=True)
        reason = "no clear path" if goals else "named by a label in another block"
        hops = 0 if path is None else sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2 for a, b in zip(path, path[1:]))
        if path is not None and scope and at.get(path[-1]) and not scope & at[path[-1]]:
            length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(path, path[1:]))
            manhattan = abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1])
            if length > LINK_RATIO * manhattan + LINK_SLACK or state["link_crossings"] + hops > state["link_budget"]:
                path, reason = None, "link longer than 1.5 x Manhattan + 8 cells or over the crossing budget"
            else:
                state["link_crossings"] += hops
        if path is None:
            stranded.append(group)
            reasons += [reason] if reason not in reasons else []
            continue
        if not hops:
            path = settle_step(path, goals, occ, net, RULES_10)
        state["crossings"] += hops
        for s in simplify(path):
            occ.add_segment(net, s)
            drawn.setdefault(net, []).append(s)
        joined = scope | at.get(path[-1], set())  # the new wire belongs to the drawings of both ends
        for p in [*group, *path]:
            if p not in tree:
                tree.add(p)
                near.add(p)
                at[p] = scope if p in group else joined
        keep = []
        for g, s in pending:
            if not g & tree:
                keep.append((g, s))
                continue
            for p in g - tree:
                tree.add(p)
                near.add(p)
                at[p] = s
        pending = keep
    return stranded, "; ".join(reasons)
