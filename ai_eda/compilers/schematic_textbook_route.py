"""Grid wiring for the textbook policy; refuse blocked routes and accidental shorts.

Only the column motif introduces diagonal cross-coupling. General wiring
tries without crossings first, then permits straight interior crossings
at a high cost. Search bounds and tie breaks are fixed; exhaustion is a
compile error, never a label fallback.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from itertools import count
from typing import TYPE_CHECKING

from ai_eda.compilers.schematic_layout import GRID_MM, natural_ref_key
from ai_eda.errors import CompileError

if TYPE_CHECKING:
    from ai_eda.compilers.schematic_textbook import Drawing

Point = tuple[int, int]
Segment = tuple[Point, Point]
Box = tuple[float, float, float, float]


def on(a: Point, b: Point, p: Point) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) == (b[1] - a[1]) * (p[0] - a[0])
            and all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in (0, 1)))


def points(segment: Segment) -> set[Point]:
    a, b = segment
    n = math.gcd(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return {(a[0] + i * (b[0] - a[0]) // n, a[1] + i * (b[1] - a[1]) // n) for i in range(n + 1)} if n else {a}


def intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    def cross(p: Point, q: Point, r: Point) -> int:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b)
            or (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0))


def search(start: Point, goals: set[Point], boxes: list[Box], foreign: list[Segment],
           blocked: set[Point], bounds: tuple[int, int], *, allow_crossings: bool = False) -> list[Point]:
    """A* with length, bend cost, then coordinate/direction ordering."""
    if start in goals:
        return [start]
    def heuristic(p: Point) -> int:
        return min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in goals)
    serial = count()
    initial = (*start, -1)
    queue = [(heuristic(start), 0, initial, next(serial))]
    costs, parent = {initial: 0}, {}
    legal_cache = {}
    while queue:
        _, cost, state, _ = heapq.heappop(queue)
        if cost != costs[state]:
            continue
        p = state[:2]
        if p in goals:
            path = [p]
            while state in parent:
                state = parent[state]
                path.append(state[:2])
            return list(reversed(path))
        steps = [(direction, dx * length, dy * length, length) for direction, (dx, dy) in
                 enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))) for length in ((1, 2) if allow_crossings else (1,))]
        for direction, dx, dy, length in steps:
            q = p[0] + dx, p[1] + dy
            if q not in legal_cache:
                legal_cache[q] = (2 <= q[0] <= bounds[0] and 2 <= q[1] <= bounds[1]
                                  and q not in blocked and not any(x1 < q[0] < x2 and y1 < q[1] < y2 for x1, y1, x2, y2 in boxes))
            if not legal_cache[q] and q not in goals:
                continue
            hits = [(a, b) for a, b in foreign if intersects(p, q, a, b)]
            if length == 2:
                mid = (p[0] + dx // 2, p[1] + dy // 2)
                if (not hits or mid in blocked or any(x1 < mid[0] < x2 and y1 < mid[1] < y2 for x1, y1, x2, y2 in boxes)
                        or any(mid in (a, b) or not on(a, b, mid) or on(a, b, p) or on(a, b, q) for a, b in hits)):
                    continue
            elif hits:
                continue
            nxt = (*q, direction)
            value = cost + length + 1000 * len(hits) + (4 if state[2] not in (-1, direction) else 0)
            if value < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = value, state
                heapq.heappush(queue, (value + heuristic(q), value, nxt, next(serial)))
    raise CompileError(f"textbook: no clear wire from {start} to {min(goals)} within {bounds}")


def simplify(path: list[Point]) -> list[Segment]:
    result = []
    a = path[0]
    for i in range(1, len(path) - 1):
        if not on(a, path[i + 1], path[i]):
            result.append((a, path[i]))
            a = path[i]
    if a != path[-1]:
        result.append((a, path[-1]))
    return result


def normalize(segments: list[Segment]) -> tuple[list[Segment], set[Point]]:
    """Split at same-net T joints, remove duplicate pieces, dot degree >= 3."""
    ends = {p for seg in segments for p in seg}
    pieces = set()
    for a, b in segments:
        cuts = sorted((p for p in ends if on(a, b, p)), key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
        pieces.update(tuple(sorted((p, q))) for p, q in zip(cuts, cuts[1:]) if p != q)
    neighbours = defaultdict(set)
    for a, b in pieces:
        neighbours[a].add(b)
        neighbours[b].add(a)
    return sorted(pieces), {p for p, adjacent in neighbours.items() if len(adjacent) >= 3}


def route(drawing: Drawing, boxes: list[Box]) -> tuple[dict[str, list[Segment]], dict[str, set[Point]]]:
    d = drawing
    obstacles = [tuple(v / GRID_MM for v in box) for box in boxes]
    positions = {k: d.pin(k) for k in d.owner}
    all_pins = {d.pin((ref, p.number)) for ref, ps in d.placed.items() for p in ps.symbol.pins}
    bounds = (math.ceil(max(b[2] for b in obstacles)) + 24, math.ceil(max(b[3] for b in obstacles)) + 24)
    def connect(net: str, keys: list[tuple[str, str]], seeded: bool = False) -> None:
        groups = [{positions[k]} for k in keys]
        if seeded:
            groups = [points(s) for s in d.segments[net]] + groups
        merged = []
        for group in groups:
            while (touching := next((other for other in merged if group & other), None)) is not None:
                group |= touching
                merged.remove(touching)
            merged.append(group)
        tree = merged.pop(0)
        while merged:
            _, start, index = min((min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in tree), p, i)
                                  for i, group in enumerate(merged) for p in group)
            foreign = [s for n, segs in d.segments.items() if n != net for s in segs]
            blocked = all_pins - {positions[k] for k in d.owner if d.owner[k] == net}
            try:
                path = search(start, tree, obstacles, foreign, blocked, bounds)
            except CompileError:
                path = search(start, tree, obstacles, foreign, blocked, bounds, allow_crossings=True)
            d.segments[net].extend(simplify(path))
            tree.update(path)
            tree.update(merged.pop(index))
            for group in list(merged):
                if tree & group:
                    tree.update(group)
                    merged.remove(group)
    # Ground symbols are local returns. They connect by the library's global
    # power semantics; a redundant bottom bus would obscure the branches.
    for key, ground in d.ground_pairs:
        connect(d.owner[key], [key, ground])
    has_ic = any(len(d.placed[r].symbol.pins) > 3 for r in d.parts)
    for net in sorted(set(d.owner.values()) - d.grounds, key=lambda n: ((n in d.supplies) if has_ic else (n not in d.supplies), n)):
        keys = sorted((k for k in d.owner if d.owner[k] == net), key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1])))
        connect(net, keys, seeded=True)
    segments, junctions = {}, {}
    for net, segs in sorted(d.segments.items()):
        segments[net], junctions[net] = normalize(segs)
    return segments, junctions
