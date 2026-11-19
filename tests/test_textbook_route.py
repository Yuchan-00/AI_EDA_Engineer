"""Textbook 1.0 router: the occupancy search under the 0.9 rules reproduces the 0.9 router (frozen routes recorded from the
0.9 code before it was removed), each 1.0 rule, budget and window, channel fan-out, normalize, determinism, and the router
in the 1.0 pipeline on every corpus design."""

from __future__ import annotations

import heapq
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.schematic_layout import GRID_MM, natural_ref_key
from ai_eda.compilers.textbook import build, route
from ai_eda.compilers.textbook.emit import pin_points
from ai_eda.compilers.textbook.glyphs import generated, generated_pin, glyph_boxes, pin_cell
from ai_eda.compilers.textbook.labels import components
from ai_eda.compilers.textbook.model import Drawing, Link, Orient, Place, Sheet, load_model
from ai_eda.compilers.textbook.recipes import Ctx
from ai_eda.compilers.textbook.route import RULES_09, RULES_10, Occupancy, channel_fanout, intersects, normalize, search
from ai_eda.errors import CompileError
from ai_eda.tools.kicad import KicadLibrary
from tests.textbook_corpus import CORPUS, corpus_ir

LIB = KicadLibrary()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "data" / "route09_golden.json"
CASES09 = ("astable", "divider", "led", "rc_lowpass", "lc_lowpass", "bjt_lowside_switch", "i2c_pullup", "amplifier")


def search09(start, goals, boxes, foreign, blocked, bounds, *, allow_crossings=False):
    """The 0.9 ``search`` signature on the occupancy search under RULES_09 (no window, no budget, as 0.9)."""
    occ = Occupancy.build({}, boxes, {p: None for p in blocked}, {}, halo=0, scale=1.0)
    for seg in foreign:
        occ.add_segment("~foreign", seg)
    path = search(start, goals, occ, "~net", RULES_09, window=(2, 2, *bounds), budget=None, allow_crossings=allow_crossings, margin=None)
    if path is None:
        raise CompileError("textbook: no clear wire")
    return path


def replay(inputs: dict) -> dict:
    """The 0.9 driver (ground pairs, then each net seeded with its drawn wires, nearest group first) on frozen inputs, with
    the occupancy search under RULES_09; independent of the 0.9 code."""
    owner = {(r, n): net for r, n, _, _, net in inputs["pins"]}
    at = {(r, n): (x, y) for r, n, x, y, _ in inputs["pins"]}
    every_pin = set(at.values()) | {tuple(p) for p in inputs["unconnected"]}
    boxes = [tuple(b) for b in inputs["boxes"]]
    bounds = (math.ceil(max(b[2] for b in boxes)) + 24, math.ceil(max(b[3] for b in boxes)) + 24)
    drawn = defaultdict(list, {net: [((s[0], s[1]), (s[2], s[3])) for s in segs] for net, segs in inputs["segments"].items()})

    def connect(net, keys, seeded=False):
        groups = ([set(route.raster(s)) for s in drawn[net]] if seeded else []) + [{at[k]} for k in keys]
        merged = []
        for group in groups:
            while (touching := next((o for o in merged if group & o), None)) is not None:
                group |= touching
                merged.remove(touching)
            merged.append(group)
        tree = merged.pop(0)
        while merged:
            _, start, index = min((min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in tree), p, i) for i, g in enumerate(merged) for p in g)
            foreign = [s for n, segs in drawn.items() if n != net for s in segs]
            blocked = every_pin - {at[k] for k in owner if owner[k] == net}
            try:
                path = search09(start, tree, boxes, foreign, blocked, bounds)
            except CompileError:
                path = search09(start, tree, boxes, foreign, blocked, bounds, allow_crossings=True)
            drawn[net].extend(route.simplify(path))
            tree.update(path)
            tree.update(merged.pop(index))
            for group in list(merged):
                if tree & group:
                    tree.update(group)
                    merged.remove(group)

    for r, n, gr, gn in inputs["ground_pairs"]:
        connect(owner[(r, n)], [(r, n), (gr, gn)])
    grounds, supplies = set(inputs["grounds"]), set(inputs["supplies"])
    for net in sorted(set(owner.values()) - grounds, key=lambda n: ((n in supplies) if inputs["has_ic"] else (n not in supplies), n)):
        connect(net, sorted((k for k in owner if owner[k] == net), key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1]))), seeded=True)
    out = {}
    for net, segs in sorted(drawn.items()):
        pieces, dots = normalize(segs)
        out[net] = {"wires": [[*a, *b] for a, b in pieces], "junctions": [list(p) for p in dots]}
    return out


# --------------------------------------------------------------------------- RULES_09 = the 0.9 router


# The live comparisons with the 0.9 code (its bytes, and its routes under the occupancy search) were retired with that
# code: the frozen routes in route09_golden.json were recorded from it and replay below without it.


def test_no_test_waits_on_the_removed_09_modules():
    """A test that imports (or importorskips) a removed 0.9 module would skip forever and hide its coverage."""
    import importlib.util

    removed = ("ai_eda.compilers.schematic_textbook_route", "ai_eda.compilers.schematic_textbook")
    assert all(importlib.util.find_spec(name) is None for name in removed)
    mention = re.compile(r"(?<!\w)schematic_textbook(?:_route)?\b")
    waiting = [path.name for path in sorted((ROOT / "tests").glob("*.py")) if path.name != Path(__file__).name
               and mention.search(path.read_text(encoding="utf-8"))]
    assert waiting == []


@pytest.mark.parametrize("name", CASES09)
def test_frozen_09_routes_replay_without_the_09_code(name):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    assert replay(golden["inputs"]) == golden["routes"]


# --------------------------------------------------------------------------- RULES_10, one rule at a time


def occupancy(bodies=None, texts=(), pins=None, escapes=None, nets_of=None, wires=()):
    occ = Occupancy.build({k: tuple(v * GRID_MM for v in b) for k, b in (bodies or {}).items()}, [tuple(v * GRID_MM for v in b) for b in texts],
                          pins or {}, escapes or {}, nets_of=nets_of)
    for net, seg in wires:
        occ.add_segment(net, seg)
    return occ


def cells(path):
    return {p for a, b in zip(path, path[1:]) for p in route.raster((a, b))}


def test_no_path_inside_the_halo_of_a_foreign_body_but_a_pin_and_its_escape_cells_are_legal():
    body = {"U": (10.0, 0.0, 20.5, 20.0)}  # cells; the halo reaches x < 21.5 and y < 21
    occ = occupancy(bodies=body, nets_of={"U": {"B"}})
    assert search((5, 20), {(30, 20)}, occ, "A", RULES_09, window=(0, 0, 40, 40)) == [(x, 20) for x in range(5, 31)]  # along the edge
    path = search((5, 20), {(30, 20)}, occ, "A", RULES_10, window=(0, 0, 40, 40))
    assert path and not any(9 < x < 21.5 and y < 21 for x, y in cells(path))
    own = occupancy(bodies=body, nets_of={"U": {"A", "B"}})  # no halo around a body with a pin on the net
    assert search((5, 20), {(30, 20)}, own, "A", RULES_10, window=(0, 0, 40, 40)) == [(x, 20) for x in range(5, 31)]
    path = search((5, 10), {(30, 10)}, own, "A", RULES_10, window=(0, 0, 40, 40))  # but never through it
    assert path and not any(10 < x < 20.5 and 0 < y < 20 for x, y in cells(path))
    occ = occupancy(bodies=body, nets_of={"U": {"B"}}, pins={(21, 10): "A"}, escapes={(21, 10): "T"})  # a pin facing up inside the halo
    path = search((21, 10), {(30, 4)}, occ, "A", RULES_10, window=(0, 0, 40, 40))
    assert path[:3] == [(21, 10), (21, 9), (21, 8)] and path[3] == (22, 8)


def test_a_parallel_neighbour_one_cell_away_is_refused_and_two_cells_away_accepted():
    occ = occupancy(wires=[("B", ((0, 10), (40, 10)))])
    assert occ.alongside((5, 11), 0, 1, "A") and not occ.alongside((5, 12), 0, 1, "A") and not occ.alongside((5, 11), 0, 1, "B")
    path10 = search((5, 11), {(35, 11)}, occ, "A", RULES_10, window=(0, 0, 40, 40))
    path09 = search((5, 11), {(35, 11)}, occ, "A", RULES_09, window=(0, 0, 40, 40))
    assert path09 == [(x, 11) for x in range(5, 36)]
    runs = [(a, b) for a, b in zip(path10, path10[1:]) if a[1] == b[1]]
    assert runs and all(a[1] == 12 for a, _ in runs)
    between = occupancy(wires=[("A", ((0, 10), (40, 10))), ("B", ((0, 12), (40, 12)))])  # its own wire on one side does not excuse the other
    path = search((5, 11), {(35, 11)}, between, "A", RULES_10, window=(0, 0, 40, 40))
    assert path and not any(a[1] == b[1] == 11 for a, b in zip(path, path[1:]))


def test_a_crossing_is_a_straight_hop_over_an_interior_point_never_a_node():
    wall = [("B", ((10, 0), (10, 40)))]
    assert search((5, 20), {(15, 20)}, occupancy(wires=wall), "A", RULES_10, window=(0, 0, 40, 40)) is None
    path = search((5, 20), {(15, 20)}, occupancy(wires=wall), "A", RULES_10, window=(0, 0, 40, 40), allow_crossings=True)
    assert path == [(5, 20), (6, 20), (7, 20), (8, 20), (9, 20), (11, 20), (12, 20), (13, 20), (14, 20), (15, 20)]
    split = [("B", ((10, 0), (10, 20))), ("B", ((10, 20), (10, 40))), ("C", ((10, 21), (30, 21)))]  # wire ends at 20 and at 21
    path = search((5, 20), {(15, 20)}, occupancy(wires=split), "A", RULES_10, window=(0, 0, 40, 40), allow_crossings=True)
    hops = [((a[0] + b[0]) // 2, a[1]) for a, b in zip(path, path[1:]) if abs(a[0] - b[0]) == 2]
    assert len(hops) == 1 and hops[0][1] not in (20, 21) and hops[0][0] == 10
    beads = [("B", ((10, y), (10, y + 1))) for y in range(40)]  # every point of the wall is a wire end: nowhere to cross
    assert search((5, 20), {(15, 20)}, occupancy(wires=beads), "A", RULES_10, window=(0, 0, 40, 40), allow_crossings=True) is None
    assert search((9, 10), {(11, 10)}, occupancy(wires=wall), "A", RULES_10, window=(9, 9, 11, 11), allow_crossings=True) == [(9, 10), (11, 10)]
    tee = wall + [("B", ((10, 11), (30, 11)))]  # a branch 1 cell below: the hop would land running alongside it
    assert search((9, 10), {(11, 10)}, occupancy(wires=tee), "A", RULES_10, window=(9, 9, 11, 11), allow_crossings=True) is None
    diagonal = [("B", ((0, 10), (30, 40)))]  # the 0.9 rules hop a diagonal at a grid point; the 1.0 rules never do
    assert search((5, 20), {(15, 20)}, occupancy(wires=diagonal), "A", RULES_09, window=(0, 0, 40, 40), allow_crossings=True)
    assert search((5, 20), {(15, 20)}, occupancy(wires=diagonal), "A", RULES_10, window=(0, 0, 40, 40), allow_crossings=True) is None
    shallow = occupancy(wires=[("B", ((0, 10), (40, 30)))])  # between its grid points a unit step would cross it unseen
    assert search((20, 25), {(20, 15)}, shallow, "A", RULES_09, window=(0, 0, 40, 40)) is None


def test_pins_are_left_and_entered_along_their_own_axis():
    pins = {(10, 10): "A", (30, 30): "A"}
    occ = occupancy(pins=pins, escapes={(10, 10): "T", (30, 30): "L"})
    path = search((10, 10), {(30, 30)}, occ, "A", RULES_10, window=(0, 0, 40, 40))
    assert path[1] == (10, 9) and path[-2] == (29, 30)
    assert search((10, 10), {(30, 30)}, occ, "A", RULES_09, window=(0, 0, 40, 40))[1] != (10, 9)
    occ = occupancy(pins={(20, 10): "A"}, escapes={(20, 10): "T"})  # a pin of the net on the way is passed by 0.9, never by 1.0
    assert search((10, 10), {(30, 10)}, occ, "A", RULES_09, window=(0, 0, 40, 40)) == [(x, 10) for x in range(10, 31)]
    path = search((10, 10), {(30, 10)}, occ, "A", RULES_10, window=(0, 0, 40, 40))
    assert path and (20, 10) not in cells(path)


def test_the_cells_a_pin_leaves_by_are_kept_for_its_net():
    occ = occupancy(pins={(20, 20): "B"}, escapes={(20, 20): "T"})  # B's pin leaves upward through (20, 19) and (20, 18)
    assert search((10, 19), {(30, 19)}, occ, "A", RULES_09, window=(0, 0, 40, 40)) == [(x, 19) for x in range(10, 31)]
    path = search((10, 19), {(30, 19)}, occ, "A", RULES_10, window=(0, 0, 40, 40))  # not across them
    assert path and not cells(path) & {(20, 19), (20, 18)}
    path = search((19, 10), {(19, 30)}, occ, "A", RULES_10, window=(0, 0, 40, 40))  # nor alongside them
    assert path and not any(a[0] == b[0] in (19, 21) and min(a[1], b[1]) in (18, 19) for a, b in zip(path, path[1:]))
    assert search((20, 20), {(20, 10)}, occ, "B", RULES_10, window=(0, 0, 40, 40)) == [(20, y) for y in range(20, 9, -1)]


def path_cost(path):
    bends = sum((b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]) for a, b, c in zip(path, path[1:], path[2:]))
    return len(path) - 1 + 4 * bends


def cheapest(start, goal, blocked, size):
    """Exhaustive search over (cell, direction): 1 per cell, 4 per change of direction."""
    best = {(start, -1): 0}
    queue = [(0, start, -1)]
    while queue:
        cost, p, came = heapq.heappop(queue)
        if cost != best[(p, came)]:
            continue
        if p == goal:
            return cost
        for d, (dx, dy) in enumerate(route.STEPS):
            q = (p[0] + dx, p[1] + dy)
            if 0 <= q[0] <= size and 0 <= q[1] <= size and (q not in blocked or q == goal):
                value = cost + 1 + (4 if came not in (-1, d) else 0)
                if value < best.get((q, d), value + 1):
                    best[(q, d)] = value
                    heapq.heappush(queue, (value, q, d))
    return None


def test_the_path_costs_one_per_cell_and_four_per_bend_and_is_the_cheapest():
    rng = random.Random(3)
    for _ in range(60):
        blocked = {(rng.randrange(16), rng.randrange(16)) for _ in range(70)}
        start, goal = rng.sample(sorted({(x, y) for x in range(16) for y in range(16)} - blocked), 2)
        occ = occupancy(texts=[(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x, y in blocked])  # one blocked cell each
        best = cheapest(start, goal, blocked, 15)
        for rules in (RULES_09, RULES_10):
            path = search(start, {goal}, occ, "A", rules, window=(0, 0, 15, 15))
            assert (path is None) == (best is None) and (path is None or path_cost(path) == best), (start, goal, rules)


def test_four_per_bend_decides_between_a_staircase_and_a_detour():
    stairs = [(1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (3, 8), (4, 8)]  # (0, 5) to (4, 9): 8 cells, 5 bends
    for depth, way in ((14, "detour"), (16, "stairs")):  # the detour: 2 bends, 18 cells (cheaper) or 22 cells (dearer)
        detour = [(0, y) for y in range(6, depth + 1)] + [(x, depth) for x in range(1, 5)] + [(4, y) for y in range(10, depth)]
        free = {(0, 5), (4, 9), *stairs, *detour}
        occ = occupancy(texts=[(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x in range(21) for y in range(21) if (x, y) not in free])
        path = search((0, 5), {(4, 9)}, occ, "A", RULES_10, window=(0, 0, 20, 20))
        assert set(path[1:-1]) == set(stairs if way == "stairs" else detour), way


# --------------------------------------------------------------------------- budget and window


def test_a_spent_budget_ends_the_search_without_an_exception():
    assert search((5, 20), {(30, 20)}, occupancy(), "A", RULES_10, window=(0, 0, 40, 40), budget=1) is None
    assert search((5, 20), {(30, 20)}, occupancy(), "A", RULES_10, window=(0, 0, 40, 40), budget=100) is not None
    assert search((5, 20), {(8, 20)}, occupancy(), "A", RULES_10, window=(0, 0, 40, 40), budget=2) is None  # 3 expansions needed
    assert search((5, 20), {(8, 20)}, occupancy(), "A", RULES_10, window=(0, 0, 40, 40), budget=3) == [(x, 20) for x in range(5, 9)]


def test_the_search_stays_in_its_first_window_while_that_holds_a_path_and_grows_it_when_not(monkeypatch):
    occ = occupancy(wires=[("B", ((51, 36), (51, 200)))])  # the cheapest way goes over the wall end, inside the first window
    looked = []
    real = occ.free
    monkeypatch.setattr(occ, "free", lambda q, net, rules: looked.append(q) or real(q, net, rules))
    path = search((49, 50), {(53, 50)}, occ, "A", RULES_09, window=(0, 0, 200, 200))
    assert path == [(49, y) for y in range(50, 34, -1)] + [(x, 35) for x in range(50, 54)] + [(53, y) for y in range(36, 51)]
    assert min(y for _, y in looked) >= 50 - route.WINDOW
    looked.clear()
    assert search((49, 50), {(53, 50)}, occ, "A", RULES_09, window=(0, 0, 200, 200), margin=None) == path
    assert min(y for _, y in looked) < 50 - route.WINDOW  # the same search without a window looks past it
    wall = [("B", ((55, 20), (55, 80)))]  # reaches past the first window both ways: only a larger one holds a path
    path = search((50, 50), {(60, 50)}, occupancy(wires=wall), "A", RULES_10, window=(0, 0, 200, 200))
    assert path and (min(y for _, y in path) < 20 or max(y for _, y in path) > 80)


@needs_libs
def test_connections_the_budget_does_not_allow_are_failure_records_joined_by_labels(monkeypatch, tmp_path):
    real = route.search
    monkeypatch.setattr(route, "search", lambda *a, **k: real(*a, **{**k, "budget": 1}))
    wirings = []
    real_route = route.route
    monkeypatch.setattr(route, "route", lambda *a: wirings.append(real_route(*a)) or wirings[-1])
    # changed deliberately with the motif recipes: the op-amp stage draws every wire of the amplifier itself, so the
    # stage is switched off here and the columns recipe leaves the router connections to refuse
    from ai_eda.compilers.textbook import motifs

    monkeypatch.setattr(motifs.OPAMP_STAGE, "claim", lambda *a: None)
    result = build(SchematicCompiler(), corpus_ir("amplifier", tmp_path, LIB), LIB)
    failed = [n for n in result.notes if n.endswith("group(s) joined by labels (no clear path)")]
    # changed deliberately with the general placer: pins that face each other for a wire stand 4-6 cells apart, so with
    # nothing wired their name labels may collide and the ladder redraws that block as grid (noted) - never the table
    assert failed and wirings[-1].failed and "whole-sheet table" not in result.fallbacks and result.score[0] == 0
    assert all(f.startswith("block grid for ") for f in result.fallbacks), result.fallbacks
    assert {n.split(":")[0][4:] for n in failed} == set(wirings[-1].failed)


def test_an_unreachable_goal_on_an_a2_sheet_ends_within_the_cap(monkeypatch):
    ring = [("B", ((100, 100), (220, 100))), ("B", ((220, 100), (220, 220))), ("B", ((220, 220), (100, 220))), ("B", ((100, 220), (100, 100)))]
    occ = occupancy(wires=ring)  # the goal sits in a 120 x 120 cell room with no door; the start outside
    pops = [0]
    real = route.heapq.heappop

    def counted(queue):
        pops[0] += 1
        return real(queue)

    monkeypatch.setattr(route.heapq, "heappop", counted)
    t = time.perf_counter()
    assert search((20, 20), {(160, 160)}, occ, "A", RULES_10, window=(1, 1, 467, 330), budget=route.BUDGET) is None
    assert pops[0] <= 4 * route.BUDGET and time.perf_counter() - t < 30
    pops[0] = 0
    assert search((20, 20), {(160, 160)}, occ, "A", RULES_10, window=(1, 1, 467, 330), budget=500) is None
    assert pops[0] <= 4 * 500
    corners = [(150, 150), (170, 150), (170, 170), (150, 170)]
    sealed = occupancy(wires=[("B", (c, corners[(i + 1) % 4])) for i, c in enumerate(corners)])  # a 20 x 20 cell room
    pops[0] = 0
    assert search((20, 20), {(160, 160)}, sealed, "A", RULES_10, window=(1, 1, 467, 330)) is None and pops[0] < 20_000
    path = search((140, 160), {(160, 160)}, sealed, "A", RULES_10, window=(1, 1, 467, 330), allow_crossings=True)  # a hop is a door
    assert path and sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2 for a, b in zip(path, path[1:])) == 1


def test_one_hundred_searches_on_an_a2_sheet_with_600_foreign_wires():
    rng = random.Random(7)
    occ = Occupancy.build({}, [], {}, {})
    for i in range(600):
        x, y, n = rng.randrange(10, 417), rng.randrange(10, 280), rng.randrange(2, 17)
        occ.add_segment(f"N{i % 50}", ((x, y), (x + n, y)) if rng.random() < 0.5 else ((x, y), (x, y + n)))
    pairs = []
    while len(pairs) < 100:
        a = (rng.randrange(5, 462), rng.randrange(5, 325))
        b = (a[0] + rng.randrange(-60, 61), a[1] + rng.randrange(-60, 61))
        if 5 <= b[0] < 462 and 5 <= b[1] < 325 and route._key(a) not in occ.busy and route._key(b) not in occ.busy and a != b:
            pairs.append((a, b))
    t = time.process_time()
    a2 = (1, 1, 467, 330)
    found = sum((search(a, {b}, occ, "X", RULES_10, window=a2) or search(a, {b}, occ, "X", RULES_10, window=a2, allow_crossings=True)) is not None
                for a, b in pairs)
    assert found >= 99 and time.process_time() - t < 10  # measured about 2 s; the bound only catches a lost index


# --------------------------------------------------------------------------- fan-out and normalize


def test_channel_fanout_has_no_crossing_on_500_monotone_matchings():
    rng = random.Random(1)
    for _ in range(500):
        k = rng.randrange(1, 12)
        left = sorted(rng.sample(range(0, 60, 2), k))
        right = sorted(rng.sample(range(0, 60, 2), k))
        wires = channel_fanout([(f"n{i}", a, b) for i, (a, b) in enumerate(zip(left, right))], 0, 40)
        for i, (a, b) in enumerate(zip(left, right)):
            ends = {p for s in wires[f"n{i}"] for p in s}
            assert (0, a) in ends and (40, b) in ends
        segs = [(net, s) for net, ss in wires.items() for s in ss]
        for i, (n1, s1) in enumerate(segs):
            for n2, s2 in segs[i + 1:]:
                assert n1 == n2 or not intersects(*s1, *s2), (left, right, n1, n2)


def test_channel_fanout_refuses_a_matching_that_is_not_monotone():
    from ai_eda.compilers.textbook.model import LayoutFailure

    with pytest.raises(LayoutFailure):
        channel_fanout([("a", 0, 4), ("b", 2, 0)], 0, 40)


def test_normalize_splits_tees_and_dots_a_pin_with_two_wires():
    assert normalize([((0, 0), (10, 0)), ((5, 0), (5, 5))]) == ([((0, 0), (5, 0)), ((5, 0), (5, 5)), ((5, 0), (10, 0))], [(5, 0)])
    corner = [((0, 0), (5, 0)), ((0, 0), (0, 5)), ((0, 5), (0, 0))]
    assert normalize(corner) == ([((0, 0), (0, 5)), ((0, 0), (5, 0))], [])
    assert normalize(corner, {(0, 0)})[1] == [(0, 0)]


# --------------------------------------------------------------------------- the router in the 1.0 pipeline


@needs_libs
def test_every_corpus_design_routes_with_no_spacing_finding_and_no_hard_finding(tmp_path, monkeypatch):
    seen = []
    real_route = route.route
    monkeypatch.setattr(route, "route", lambda ctx, sheet, boxes: seen.append((ctx, sheet, real_route(ctx, sheet, boxes))) or seen[-1][2])
    for name in CORPUS:
        result = build(SchematicCompiler(), corpus_ir(name, tmp_path / name, LIB), LIB)
        assert result.fallbacks == () and result.score[:2] == (0, 0), (name, result.notes[0])
        assert not any(n.startswith("spacing:") for n in result.notes)
        ctx, sheet, wiring = seen[-1]  # every wire end meets another wire, a pin, a glyph or a label (no dangling stub tail)
        anchors = set(pin_points(ctx.model, sheet)) | {p for lab in sheet.labels() for p in ((lab.at,) if lab.stub is None else route.raster(lab.stub))}
        for net, segs in wiring.segments.items():
            ends = defaultdict(int)
            for s in segs:
                ends[s[0]] += 1
                ends[s[1]] += 1
            assert all(ends[p] > 1 or p in anchors for s in segs for p in s), (name, net)
        for note in result.notes:
            if "joined by labels" in note:
                assert note.startswith("net ") and "group(s) joined by labels (" in note, note


def test_a_connection_between_drawings_is_a_wire_only_within_the_length_rule_and_the_crossing_budget():
    def grow(wall, scope, link_budget):
        state = {"crossings": 0, "link_crossings": 0, "link_budget": link_budget}
        stranded, reason = route._grow("A", {(0, 0)}, {(0, 0): {0}}, [({(20, 0)}, scope)], occupancy(wires=[("B", wall)]), (-60, -60, 80, 60),
                                       route.BUDGET, {}, state)
        return bool(stranded), reason, state["crossings"]

    detour = ((10, -30), (10, 30))  # the way round is over 80 cells, more than 1.5 x 20 + 8
    assert grow(detour, {0}, 1) == (False, "", 0)  # inside one drawing any length goes
    assert grow(detour, {1}, 1) == (True, "link longer than 1.5 x Manhattan + 8 cells or over the crossing budget", 0)
    wall = ((10, -60), (10, 60))  # no way round: one crossing
    assert grow(wall, {1}, 1) == (False, "", 1)
    assert grow(wall, {1}, 0)[:2] == (True, "link longer than 1.5 x Manhattan + 8 cells or over the crossing budget")


def grown(occ, tree_seg, start, bounds=(0, 0, 50, 40)):
    """The segments ``route._grow`` draws to join the pin ``start`` to the wire ``tree_seg`` (net A, one drawing)."""
    occ.add_segment("A", tree_seg)
    tree = set(route.raster(tree_seg))
    drawn: dict = {}
    state = {"crossings": 0, "link_crossings": 0, "link_budget": 1}
    stranded, _ = route._grow("A", set(tree), {p: {0} for p in tree}, [({start}, {0})], occ, bounds, route.BUDGET, drawn, state)
    assert not stranded
    return [s for s in drawn["A"] if s != tree_seg]


def test_a_step_a_path_cannot_avoid_stands_at_its_pin():
    """A body across the pin's row forces the wire from the pin (30, 10), facing left, down 2 rows to reach the wire at
    x = 2 from row 12 down. The step goes 1 cell from the pin, so the wire runs straight to there and turns into the pin
    (a drop to an input), not part way along beside the body."""
    def occ():
        return occupancy(bodies={"G": (10, 8, 12, 11)}, nets_of={"G": {"X"}}, pins={(30, 10): "A"}, escapes={(30, 10): "L"})
    assert grown(occ(), ((2, 12), (2, 20)), (30, 10)) == [((30, 10), (29, 10)), ((29, 10), (29, 12)), ((29, 12), (2, 12))]
    # as close to the pin as the rules allow: a text box on (29, 11) puts the step a cell further out
    blocked = occupancy(bodies={"G": (10, 8, 12, 11)}, nets_of={"G": {"X"}}, pins={(30, 10): "A"}, escapes={(30, 10): "L"},
                        texts=[(28.5, 10.5, 29.5, 11.5)])
    assert grown(blocked, ((2, 12), (2, 20)), (30, 10)) == [((30, 10), (28, 10)), ((28, 10), (28, 12)), ((28, 12), (2, 12))]


def test_a_step_between_two_pins_or_a_long_one_is_left_where_the_search_put_it(monkeypatch):
    """Only a step of at most route.STEP cells next to exactly one pin moves: between two pins neither end is the one to
    drop at, and a longer run is a detour of its own."""
    def both(tree_seg, start, **kwargs):
        def occ():
            return occupancy(nets_of={"G": {"X"}}, **kwargs)
        settled = grown(occ(), tree_seg, start)
        monkeypatch.setattr(route, "settle_step", lambda path, *a: path)
        raw = grown(occ(), tree_seg, start)
        monkeypatch.undo()
        return settled, raw

    pins = {(30, 10): "A", (2, 12): "A"}
    settled, raw = both(((2, 12), (1, 12)), (30, 10), bodies={"G": (10, 8, 12, 11)}, pins=pins, escapes={(30, 10): "L", (2, 12): "R"})
    assert settled == raw and len(raw) == 3 and raw[1][0][0] not in (29, 3)  # the search's own place, not beside either pin
    settled, raw = both(((2, 15), (2, 20)), (30, 10), bodies={"G": (10, 8, 12, 14)}, pins={(30, 10): "A"}, escapes={(30, 10): "L"})
    assert settled == raw and abs(raw[1][0][1] - raw[1][1][1]) == 5 > route.STEP


@needs_libs
def test_power_symbols_join_their_groups_by_name_and_are_never_wired_to_each_other(tmp_path):
    model = load_model(SchematicCompiler(), corpus_ir("divider", tmp_path, LIB), LIB)
    place = {"R1": Place(40, 40), "R2": Place(60, 40), "J1": Place(100, 40)}
    gnd = next(n for n, info in model.nets.items() if {k for k, _ in info.pins} == {"R2", "J1"})
    glyphs = tuple(generated(LIB, "GND", gnd, at, Orient(), ref) for at, ref in (((60, 50), "#PWR01"), ((92, 50), "#PWR02")))
    sheet = Sheet((Drawing("t", "test", place),), ((0, 0),), glyphs=glyphs)
    wiring = route.route(Ctx(SchematicCompiler(), LIB, model, []), sheet, glyph_boxes(model, sheet))
    roots = [generated_pin(g) for g in glyphs]
    pins = [pin_cell(model.glyphs[k], p, (place[k].x, place[k].y), Orient()) for k, p in model.nets[gnd].pins]
    groups = components(wiring.segments[gnd], pins + roots)
    assert not wiring.failed and len(groups) == 2 and all(sum(r in pts for r in roots) == 1 for _, pts in groups)


@needs_libs
def test_a_net_routed_first_never_seals_the_pin_of_a_net_routed_later(tmp_path):
    model = load_model(SchematicCompiler(), corpus_ir("divider", tmp_path, LIB), LIB)
    place = {"R1": Place(40, 40), "R2": Place(50, 46), "J1": Place(100, 40)}  # R2's top pin faces up, 3 cells above its ground glyph's row
    gnd = next(n for n, info in model.nets.items() if {k for k, _ in info.pins} == {"R2", "J1"})
    sheet = Sheet((Drawing("t", "test", place),), ((0, 0),), glyphs=(generated(LIB, "GND", gnd, (40, 48), Orient(), "#PWR01"),))
    ctx = Ctx(SchematicCompiler(), LIB, model, [])
    wiring = route.route(ctx, sheet, glyph_boxes(model, sheet))
    assert wiring.failed == {} and ctx.notes == [] and wiring.crossings == 0


@needs_libs
def test_nets_are_routed_power_symbols_first_then_inside_one_drawing_then_across_drawings(tmp_path, monkeypatch):
    model = load_model(SchematicCompiler(), corpus_ir("divider", tmp_path, LIB), LIB)
    nets = {frozenset(k for k, _ in info.pins): n for n, info in model.nets.items()}
    vin, vout, gnd = nets[frozenset({"R1", "J1"})], nets[frozenset({"R1", "R2", "J1"})], nets[frozenset({"R2", "J1"})]
    one, two = {"R1": Place(40, 40), "J1": Place(70, 40)}, {"R2": Place(120, 40)}
    sheet = Sheet((Drawing("a", "test", one), Drawing("b", "test", two)), ((0, 0), (0, 0)), glyphs=(generated(LIB, "GND", gnd, (120, 50), Orient(), "#PWR01"),))
    order = []
    real = route._grow
    monkeypatch.setattr(route, "_grow", lambda net, *a, **k: order.append(net) or real(net, *a, **k))
    route.route(Ctx(SchematicCompiler(), LIB, model, []), sheet, glyph_boxes(model, sheet))
    assert order == [gnd, vin, vout]


@needs_libs
def test_a_wire_link_between_drawings_is_routed_or_noted_like_any_failed_connection(tmp_path):
    model = load_model(SchematicCompiler(), corpus_ir("divider", tmp_path, LIB), LIB)
    place = {"R1": Place(40, 40), "R2": Place(60, 40), "J1": Place(100, 40)}
    net = next(n for n, info in model.nets.items() if {k for k, _ in info.pins} == {"R1", "J1"})
    a, b = (pin_cell(model.glyphs[k], p, (place[k].x, place[k].y), Orient()) for k, p in sorted(model.nets[net].pins, key=lambda kp: kp[0] != "R1"))
    sheet = Sheet((Drawing("a", "test", {k: place[k] for k in ("R1", "R2")}), Drawing("b", "test", {"J1": place["J1"]})), ((0, 0), (0, 0)),
                  links=(Link(net, a, b, "wire"),))
    ctx = Ctx(SchematicCompiler(), LIB, model, [])
    assert route.route(ctx, sheet, glyph_boxes(model, sheet)).failed == {} and ctx.notes == []
    ctx = Ctx(SchematicCompiler(), LIB, model, [], budget=1)  # the net's own routing and then the link both fail
    groups = route.route(ctx, sheet, glyph_boxes(model, sheet)).failed[net]
    assert len(groups) == len(set(groups)) == 2 and {a, b} <= set().union(*groups)  # each stranded group recorded once
    assert all(re.fullmatch(r"net \S+: \d+ group\(s\) joined by labels \(.+\)", note) for note in ctx.notes), ctx.notes
    assert any(f"link {a}-{b}" in note for note in ctx.notes)


@needs_libs
def test_identical_routes_under_any_hash_seed(tmp_path):
    script = (
        "import hashlib, os, sys; sys.path.insert(0, os.getcwd())\n"
        "from pathlib import Path\n"
        "from ai_eda.compilers import SchematicCompiler\n"
        "from ai_eda.compilers.textbook import route\n"
        "from ai_eda.tools.kicad import KicadLibrary\n"
        "from tests.textbook_corpus import corpus_ir\n"
        "lib, seen = KicadLibrary(), []\n"
        "real = route.route\n"
        "route.route = lambda *a: seen.append(real(*a)) or seen[-1]\n"
        "for name in ('analog_piano', 'astable'):\n"
        "    SchematicCompiler().build(corpus_ir(name, Path(sys.argv[1]) / name, lib), lib)\n"
        "    print(name, hashlib.sha256(repr(sorted(seen[-1].segments.items())).encode()).hexdigest())\n"
    )
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, "-c", script, str(tmp_path / seed)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] and outputs[0].count("\n") == 2
