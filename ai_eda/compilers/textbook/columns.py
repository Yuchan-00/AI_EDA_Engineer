"""The general columns recipe: the layered engine run per block.

Steps (the numbers are the design's):

- C1 potentials: breadth-first distances dS / dG from the supply and ground rails over DC paths (a two-terminal part that
  is not a capacitor, the through pair of a three-terminal part); ``pot = dG / (dS + dG)``, rails 1 and 0, unreachable 0.5.
- C2 orientation of every glyph that is not two-terminal: the transform with the lowest cost - 10 per ground net on a top
  pin, 10 per supply on a bottom pin, 4 per top / bottom pair whose top is lower in potential, minus the flow vote (side
  pins facing upstream left and downstream right); ties keep transform order. Transistors (B / C / E, G / D / S) try four
  transforms, other three-pin parts all eight, amplifiers and ICs only their own.
- C3 path cover: parallel bundles first (parts on the same two nets side by side, connectors last), then repeatedly the
  shortest supply-to-ground path over two-terminal parts and through pairs (a capacitor only first or last; ties by the
  first part's reference) as one column; then single parts from a rail to a signal (hanging) and from a signal to ground
  (standing).
- C4 rows: items are nets and glyphs; hard edges run top net -> glyph -> bottom net; a part in series between two
  supplies is merged into the rail row; a two-terminal part across the left and right side nets of one glyph (feedback)
  gets its own row above that glyph; every other signal-to-signal part is vertical when one of its nets already lies above
  the other (or a connector stacks them), else merged into one row - a merge (or edge) that would close a cycle is
  refused, the part gets a jog and a note. A signal net that 3 or more remaining parts end on is a summing bus: each of
  them stands from its own net down to it, under the one part that feeds that net. A transistor whose control net is
  only the tap between a standing part above and one below takes the tap's row. Longest path, then a pull-down pass;
  free side nets take their glyph's row.
- C5 columns: the path-cover columns whole; for each other net the first part above it joined with the first below.
- C6 order of slots (columns, horizontal parts, connectors): a horizontal part between its nets' spines in flow
  direction, input connectors first and output connectors last; up to 8 barycenter sweeps, re-sorted by Kahn's algorithm
  with a heap; then up to 6 passes of greedy adjacent switch on the sketch (each net a trunk on its row plus a drop per
  attachment) that never raise its (crossings, length). A transistor's control pin then turns toward its net.
- Mirror: the largest closed involution of the block's glyphs grown from two like glyphs (not two-terminal) by
  simultaneous breadth-first search (pins matched by library name and position, rails fixed, at least 4 glyphs, the
  halves coupled: the seeds share a signal net or a path of nets the mirror does not fix joins them). Paired slots become
  two halves: the left halves in order and the right halves reversed, the other slots before, between or after them,
  chosen by (the block's output on the right, couplings between their nodes, columns outboard, sketch cost); this order
  is precedence for the greedy switch. The right half takes the mirrored orientation, fields face outward, slot x is
  symmetric about one axis (the slots between the halves centred on it), and the control nets of a mirrored transistor
  pair facing each other cross once between the halves (the X, as recipe wires). A mirrored drawing that fails is
  drawn again without the mirror, noted.
- C7 coordinates from measured extents (body, fields at fixed anchors, local power symbols): horizontal parts whose slot
  ranges overlap in one row get a row each; rows y(k) = y(k-1) + down + 2 + up, 4 under a row of signal nets (room for a
  name label beside the wire); slot x by longest path, 2 cells clear of every earlier slot it shares height with (6 when
  a signal wire runs between them), a pin facing right 4 cells left of what it joins (a pin facing left, right of it); a
  horizontal part centred between its spines and at the height of the one side pin it is tied to; connectors with their
  first pin on its net's height, a connector bundled with a standing part centred on it with a wire from each end.
- C8 ports from :func:`~ai_eda.compilers.textbook.blocks.ports`; rail spans for the sheet's rail lines.

Supply pins: a top pin under the rail row (or a bottom pin over it) is the rail line's; a side pin of a part in series
sits on the rail row; amplifier, IC and power-unit supply pins, and supply pins of a block without its rail row, get a
power symbol of their own; ground and negative-rail pins always do (one symbol per pin); a symbol with no room beside a
neighbour pin becomes a label at that pin.

An IC of 20 pins or more and the glyphs on its signal nets are left to the block grid (which names the IC's pins by
labels), with a note; a free set larger than 24 glyphs is split at nets with more than 6 pins (fewer, repeatedly, when
needed) before it is claimed. Variants: as computed, and with the field sides of standing parts flipped.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, replace

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook import blocks as blocks_mod
from ai_eda.compilers.textbook import rails as rails_mod
from ai_eda.compilers.textbook.glyphs import TRANSFORMS, body_box, drawing_boxes, field_anchors, generated_boxes, label_box, measure, mm, orient_two, overlaps, pin_cell, pin_side, to_cells, union
from ai_eda.compilers.textbook.labels import stub_label
from ai_eda.compilers.textbook.model import OPPOSITE, BlockInfo, BoxMM, Drawing, Generated, Glyph, Kind, Label, LayoutFailure, Model, Orient, Place, Pt, RailSpan, Role, Seg

GAP = 2  # cells between measured boxes
HUB_PINS = 20  # an IC with this many pins is left to the block grid
SPLIT_GLYPHS = 24
SPLIT_PINS = 6
SWEEPS = 8
PASSES = 6
REPAIRS = 3  # extra-gap rounds when the drawn boxes overlap
FIELD_GAP = 1  # cells between a field and its own pin wires
WIRE_GAP = 6  # cells between two glyphs a signal wire runs between (room for its name label)
NET_GAP = 4  # cells from a row of signal nets down to the glyphs under it
FACING = 4  # cells from a side pin to the pin it faces (its 2 escape cells stay clear of the other wire)
BUNDLE_GAP = 4  # extra cells between a standing part and the connector bundled with it (room for both its wires)


def _nat(key: str) -> tuple:
    return natural_ref_key(key)


def _hub(g: Glyph) -> bool:
    return g.kind is Kind.IC and len(g.pins) >= HUB_PINS


def _transistor(g: Glyph) -> bool:
    return g.kind is Kind.THRU3 and {p.name for p in g.pins} in ({"B", "C", "E"}, {"G", "D", "S"})


# --------------------------------------------------------------------------- claim


def _groups(model: Model, keys: list[str], limit: int) -> list[list[str]]:
    """Connected groups of ``keys`` over nets that are not ground and have at most ``limit`` pins among them."""
    inside = set(keys)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for net, info in sorted(model.nets.items()):
        members = sorted({k for k, _ in info.pins if k in inside}, key=_nat)
        if info.role is Role.GROUND or len(members) < 2 or len([1 for k, _ in info.pins if k in inside]) > limit:
            continue
        for k in members[1:]:
            a, b = find(members[0]), find(k)
            if a != b:
                parent[max(a, b, key=_nat)] = min(a, b, key=_nat)
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    return sorted((sorted(g, key=_nat) for g in groups.values()), key=lambda g: _nat(g[0]))


def sub_blocks(model: Model, keys: list[str]) -> list[list[str]]:
    """The glyphs in groups joined by their non-ground nets; a group larger than :data:`SPLIT_GLYPHS` is split at nets
    with more than :data:`SPLIT_PINS` pins, then fewer, and at last in natural order."""
    out = []
    for group in _groups(model, keys, 10**6):
        pending = [group]
        while pending:
            g = pending.pop(0)
            if len(g) <= SPLIT_GLYPHS:
                out.append(g)
                continue
            for limit in range(SPLIT_PINS, 1, -1):
                parts = _groups(model, g, limit)
                if len(parts) > 1:
                    pending = parts + pending
                    break
            else:
                pending = [g[i:i + SPLIT_GLYPHS] for i in range(0, len(g), SPLIT_GLYPHS)] + pending
    return sorted(out, key=lambda g: _nat(g[0]))


class ColumnsRecipe:
    name = "columns"

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        """The first group (see :func:`sub_blocks`) of the free glyphs, without large ICs and the glyphs on their signal
        nets: the block grid names an IC's pins by labels, so those glyphs go with it (noted once per block)."""
        keys = sorted(free, key=_nat)
        hubs = [g for g in model.glyphs.values() if _hub(g)]
        nets = {n for g in hubs for n in g.net.values() if model.nets[n].role is Role.SIGNAL}
        left = [k for k in keys if _hub(model.glyphs[k]) or set(model.glyphs[k].net.values()) & nets]
        if left:
            big = ", ".join(f"{g.key} ({len(g.pins)} pins)" for g in hubs)
            note = f"block {block.id}: {', '.join(left)} left to the block grid (large IC {big} and the glyphs on its signal nets)"
            if note not in model.notes:
                model.notes.append(note)
        keys = [k for k in keys if k not in left]
        if not keys:
            return None
        return frozenset(sub_blocks(model, keys)[0])

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        layout = Layout(ctx, block, members)
        try:
            out = layout.drawings()
        except LayoutFailure:
            if not layout.half:
                raise
            layout = Layout(ctx, block, members, mirror=False)  # the mirror-off variant, only when the mirrored one fails
            out = layout.drawings()
            layout.notes.append("columns: the mirrored halves did not fit; drawn without the mirror")
        for note in layout.notes:  # jogs, dropped edges, extra rows: into the compile notes, once
            if f"block {block.id}: {note}" not in ctx.notes:
                ctx.notes.append(f"block {block.id}: {note}")
        return out


COLUMNS = ColumnsRecipe()


# --------------------------------------------------------------------------- potentials and ranks


def potentials(model: Model) -> dict[str, float]:
    """C1: ``pot = dG / (dS + dG)`` per net over DC paths; supplies 1, ground and negative rails 0, unreachable 0.5."""
    adj: dict[str, set[str]] = defaultdict(set)
    for g in model.glyphs.values():
        pair = None
        if g.kind is Kind.TWO and not g.dc_blocking and len(set(g.net.values())) == 2:
            pair = tuple(g.net.values())
        elif g.kind is Kind.THRU3 and g.through() is not None:
            a, b = g.through()
            if a.number in g.net and b.number in g.net:
                pair = (g.net[a.number], g.net[b.number])
        if pair:
            adj[pair[0]].add(pair[1])
            adj[pair[1]].add(pair[0])
    high = {n for n, i in model.nets.items() if i.role is Role.SUPPLY}
    low = {n for n, i in model.nets.items() if i.role in (Role.GROUND, Role.NEGATIVE)}

    def bfs(seeds: set[str]) -> dict[str, int]:
        dist = {n: 0 for n in seeds}
        queue = deque(sorted(seeds))
        while queue:
            n = queue.popleft()
            for m in sorted(adj[n]):
                if m not in dist and m not in high and m not in low:
                    dist[m] = dist[n] + 1
                    queue.append(m)
        return dist

    ds, dg = bfs(high), bfs(low)
    pot = {}
    for n in model.nets:
        if n in high:
            pot[n] = 1.0
        elif n in low:
            pot[n] = 0.0
        else:
            a, b = ds.get(n), dg.get(n)
            pot[n] = 0.5 if a is None and b is None else 1.0 if b is None else 0.0 if a is None else b / (a + b)
    return pot


# --------------------------------------------------------------------------- the layout of one block


@dataclass
class Cover:
    """C3: the path-cover columns (glyph keys top to bottom), hanging and standing singles, bundles (representative ->
    partners) and the arcs each column item was taken along (key -> (top net, bottom net))."""

    columns: list[list[str]]
    hanging: list[str]
    standing: list[str]
    bundles: dict[str, list[str]]
    arcs: dict[str, tuple[str, str]]


class _UF:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            lo, hi = sorted((a, b))
            self.parent[hi] = lo


class Layout:
    """The columns layout of one block (or sub-block): every step of the module text, in order."""

    def __init__(self, ctx, block: BlockInfo, members: frozenset[str], mirror: bool = True) -> None:
        self.ctx = ctx
        self.model: Model = ctx.model
        self.block = block
        self.keys = sorted(members, key=_nat)
        self.notes: list[str] = []
        m = self.model
        self.g = {k: m.glyphs[k] for k in self.keys}
        self.role = {n: info.role for n, info in m.nets.items()}
        self.pins = {k: [(p, n) for p, n in sorted(self.g[k].net.items(), key=lambda kv: _nat(kv[0])) if (self.g[k].ref, p) not in m.stacked]
                     for k in self.keys}
        self.conn = [k for k in self.keys if self.g[k].kind is Kind.CONN]
        self.cells = [k for k in self.keys if k not in self.conn]
        self.signals = {n for k in self.keys for n in self.g[k].net.values() if m.nets[n].role is Role.SIGNAL}
        self.hub_nets = {n for g in m.glyphs.values() if _hub(g) for n in g.net.values() if m.nets[n].role is Role.SIGNAL}
        self.two = [k for k in self.cells if self.g[k].kind is Kind.TWO and len({n for _, n in self.pins[k]}) == 2 and len(self.pins[k]) == 2]
        self.rail_index = {n: i for i, n in enumerate(m.rails or rails_mod.rail_order(m))}
        self.pot = potentials(m)
        self.rank = self._ranks()
        self.orient: dict[str, Orient] = {}
        self.terminals: dict[str, list[Label]] = {}
        self.half: dict[str, str] = {}  # glyph key -> "L" / "R": the mirrored halves as drawn
        self.sym: list[tuple[str, str]] = []  # (left slot, right slot) of each mirrored pair
        self.pair = self._involution() if mirror else {}
        self._orient_cells()
        self.cover = self.path_cover()
        self._rows()
        self._columns()
        self._order()
        self._turn_controls()
        self._mirror_orient()

    # -- helpers

    def high(self, n: str) -> bool:
        return self.role[n] is Role.SUPPLY

    def low(self, n: str) -> bool:
        return self.role[n] in (Role.GROUND, Role.NEGATIVE)

    def signal(self, n: str) -> bool:
        return self.role[n] is Role.SIGNAL

    def side(self, key: str, number: str, o: Orient | None = None) -> str:
        return pin_side(self.g[key], number, o if o is not None else self.orient[key])

    def nets(self, key: str) -> list[str]:
        return [n for _, n in self.pins[key]]

    def _ranks(self) -> dict[str, int]:
        """Signal flow inside the block: distance from its inputs minus distance to its outputs (missing: 50)."""
        nets_of = {k: [n for n in self.nets(k) if not self.low(n)] for k in self.keys}
        cells_of: dict[str, list[str]] = defaultdict(list)
        for k in self.keys:
            for n in nets_of[k]:
                cells_of[n].append(k)
        present = set(cells_of)
        ins = [n for n in self.block.inputs if n in present]
        outs = [n for n in self.block.outputs if n in present]
        if not ins and not outs:
            driven = {n for k in self.keys for p, n in self.pins[k] if self.g[k].pin(p).etype in ("output", "power_out")}
            outs = sorted(n for n in present if n in driven and self.signal(n))
            ins = sorted(n for k in self.conn for n in nets_of[k] if self.signal(n) and n not in driven)

        def bfs(seeds: list[str]) -> dict[str, int]:
            dist = {n: 0 for n in seeds}
            queue = deque(sorted(seeds))
            while queue:
                n = queue.popleft()
                if self.high(n) and dist[n] > 0:
                    continue
                for k in sorted(cells_of[n], key=_nat):
                    for m in nets_of[k]:
                        if m not in dist:
                            dist[m] = dist[n] + 1
                            queue.append(m)
            return dist

        din, dout = bfs(ins), bfs(outs)
        return {n: din.get(n, 50) - dout.get(n, 50) for n in sorted(present)}

    # -- C2 orientation

    def candidates(self, key: str) -> list[Orient]:
        g = self.g[key]
        if g.kind in (Kind.IC, Kind.PWRUNIT) or g.kind is Kind.AMP and g.power_pins():
            return [Orient()]
        if g.kind is Kind.AMP:
            return [Orient(), Orient(0, "x")]
        if _transistor(g):
            return [Orient(), Orient(0, "y"), Orient(0, "x"), Orient(180)]
        return list(TRANSFORMS)

    def cost(self, key: str, o: Orient) -> float:
        sides = {p: self.side(key, p, o) for p, _ in self.pins[key]}
        tops = [n for p, n in self.pins[key] if sides[p] == "T"]
        bots = [n for p, n in self.pins[key] if sides[p] == "B"]
        c = 10.0 * sum(self.low(n) for n in tops) + 10.0 * sum(self.high(n) for n in bots)
        c += 4.0 * sum(1 for a in tops for b in bots if self.pot[a] < self.pot[b])
        if sum(self.high(n) for n in self.nets(key)) < 2:  # a lone rail pin belongs on top or bottom, not on a side
            c += 2.0 * sum(1 for p, n in self.pins[key] if sides[p] in ("L", "R") and not self.signal(n))
        return c - self.vote(key, sides)

    def vote(self, key: str, sides: dict[str, str]) -> float:
        """Side pins facing upstream left and downstream right: by flow rank among signal nets, by rail order on a part
        whose pins are all on supplies."""
        nets = self.nets(key)
        if nets and all(self.high(n) for n in nets):
            key_of = {n: self.rail_index.get(n, 0) for n in nets}
        else:
            key_of = {n: self.rank.get(n, 0) for n in nets if self.signal(n)}
        votes = 0.0
        for p, n in self.pins[key]:
            if n not in key_of or sides[p] not in ("L", "R"):
                continue
            others = [key_of[m] for q, m in self.pins[key] if m in key_of and m != n]
            if not others:
                continue
            mean = sum(others) / len(others)
            if key_of[n] < mean:
                votes += 1 if sides[p] == "L" else -1
            elif key_of[n] > mean:
                votes += 1 if sides[p] == "R" else -1
        return votes

    def _orient_cells(self) -> None:
        for k in self.cells:
            if k in self.two:
                continue
            cands = self.candidates(k)
            self.orient[k] = min(cands, key=lambda o: (self.cost(k, o), cands.index(o)))

    # -- C3 path cover

    def path_cover(self) -> Cover:
        items = [k for k in self.keys if (k in self.two) or (self.g[k].kind is Kind.CONN and len(self.pins[k]) == 2 and len(set(self.nets(k))) == 2)]
        bundles: dict[frozenset[str], list[str]] = defaultdict(list)
        for k in items:
            bundles[frozenset(self.nets(k))].append(k)
        rep: dict[str, list[str]] = {}
        arcs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # net -> [(item, far net)]
        for nets, ks in sorted(bundles.items(), key=lambda kv: _nat(min(kv[1], key=_nat))):
            ks = sorted(ks, key=lambda k: (self.g[k].kind is Kind.CONN, _nat(k)))
            rep[ks[0]] = ks[1:]
            a, b = sorted(nets)
            arcs[a].append((ks[0], b))
            arcs[b].append((ks[0], a))
        for k in self.cells:
            if self.g[k].kind is Kind.THRU3 and k not in self.two:
                top = [n for p, n in self.pins[k] if self.side(k, p) == "T"]
                bottom = [n for p, n in self.pins[k] if self.side(k, p) == "B"]
                if len(top) == 1 and len(bottom) == 1 and top[0] != bottom[0]:
                    arcs[top[0]].append((k, bottom[0]))
        for n in arcs:
            arcs[n].sort(key=lambda t: (_nat(t[0]), t[1]))
        used: set[str] = set()
        columns: list[list[str]] = []
        taken: dict[str, tuple[str, str]] = {}
        while True:
            path = self._shortest(arcs, used)
            if path is None:
                break
            keys = [k for k, _, _ in path]
            used.update(keys)
            columns.append(keys)
            for k, a, b in path:
                taken[k] = (a, b)
        hanging, standing = [], []
        for k in rep:
            if k in used or self.g[k].kind is Kind.CONN:
                continue
            a, b = self.nets(k)
            if self.high(a) + self.high(b) == 1 and (self.signal(a) or self.signal(b)):
                hanging.append(k)
            elif self.low(a) + self.low(b) == 1 and (self.signal(a) or self.signal(b)):
                standing.append(k)
        return Cover(columns, hanging, standing, {k: v for k, v in rep.items() if v}, taken)

    def _shortest(self, arcs: dict[str, list[tuple[str, str]]], used: set[str]) -> list[tuple[str, str, str]] | None:
        """The shortest unused supply-to-ground path (ties: the natural order of its parts), or None."""
        starts = sorted(n for n in arcs if self.high(n))
        best: dict[str, tuple[tuple, tuple]] = {n: ((), ()) for n in starts}
        layer = starts
        while layer:
            found, nxt = [], {}
            for n in layer:
                keys, path = best[n]
                for item, far in arcs[n]:
                    if item in used or any(item == k for k, _, _ in path) or far in best and not self.low(far):
                        continue
                    if self.g[item].dc_blocking and path and not self.low(far):
                        continue
                    k2 = keys + (_nat(item),)
                    step = path + ((item, n, far),)
                    if self.low(far):
                        found.append((k2, step))
                    elif self.signal(far) and (far not in nxt or k2 < nxt[far][0]):
                        nxt[far] = (k2, step)
            if found:
                return list(min(found)[1])
            for far in sorted(nxt):
                best[far] = nxt[far]
            layer = sorted(nxt)
        return None

    # -- supply pins: a rail line above (or below) the glyph, or a local power symbol

    def _attach(self) -> None:
        """For every pin on a rail: 'line' (a top pin under the rail row, a bottom pin over it), 'row' (a side pin of a
        part in series between supplies, on the rail row), 'local' (its own power symbol) or 'route' (the router joins it
        to the rail line of its block)."""
        self.attach: dict[tuple[str, str], str] = {}
        rowed: set[str] = set()
        for k in self.cells:
            g = self.g[k]
            for p, n in self.pins[k]:
                if self.low(n):
                    self.attach[(k, p)] = "local"
                elif self.high(n):
                    if g.kind in (Kind.AMP, Kind.IC, Kind.PWRUNIT):
                        self.attach[(k, p)] = "local"
                    elif k in self.series:
                        self.attach[(k, p)] = "row"
                        rowed.add(n)
                    elif self._pin_side(k, p) in ("T", "B"):
                        self.attach[(k, p)] = "line"
                        rowed.add(n)
        for k in self.keys:
            for p, n in self.pins[k]:
                if self.high(n) and (k, p) not in self.attach:
                    self.attach[(k, p)] = "route" if n in rowed else "label" if self._crowded(k, p, -1) else "local"
                elif self.low(n):
                    self.attach[(k, p)] = "label" if self._crowded(k, p, 1) else "local"
                elif k in self.conn and n in self.hub_nets:
                    self.attach[(k, p)] = "label"  # the block grid names the large IC's pins: a label pair, not a long wire
        self.rowed = rowed

    def _crowded(self, key: str, number: str, toward: int) -> bool:
        """A connector pin whose power symbol (above it: ``toward`` -1, below: 1) would stand in front of the next pin."""
        if key not in self.conn:
            return False
        o = Orient()
        g = self.g[key]
        at, side = pin_cell(g, number, (0, 0), o), pin_side(g, number, o)
        return any(pin_side(g, p, o) == side and 0 < (pin_cell(g, p, (0, 0), o)[1] - at[1]) * toward <= 2 for p, _ in self.pins[key] if p != number)

    def _pin_side(self, key: str, number: str) -> str:
        if key in self.two:
            top, _ = self.two_nets(key)
            if not self.vertical.get(key):
                return "L" if self.g[key].net[number] == top else "R"
            return "T" if self.g[key].net[number] == top else "B"
        return self.side(key, number)

    def two_nets(self, key: str) -> tuple[str, str]:
        """(top or left net, bottom or right net) of a two-terminal part as drawn."""
        return self.first[key], next(n for n in self.nets(key) if n != self.first[key])

    # -- C4 rows

    def _rows(self) -> None:
        cover = self.cover
        self.vertical: dict[str, bool] = {}
        self.first: dict[str, str] = {}
        partners = {p for ps in cover.bundles.values() for p in ps}
        for k, (a, b) in cover.arcs.items():
            if k in self.two:
                self.vertical[k], self.first[k] = True, a
        for k in cover.hanging:
            self.vertical[k], self.first[k] = True, next(n for n in self.nets(k) if self.high(n))
        for k in cover.standing:
            self.vertical[k], self.first[k] = True, next(n for n in self.nets(k) if not self.low(n))
        for k in sorted(partners, key=_nat):
            if k in self.two:
                rep = next(r for r, ps in cover.bundles.items() if k in ps)
                if rep in self.vertical:
                    self.vertical[k], self.first[k] = self.vertical[rep], (self.first[rep] if self.first[rep] in self.nets(k) else self.nets(k)[0])
        # parts in series between supplies are merged into the rail row
        self.series: set[str] = set()
        for k in self.cells:
            nets = self.nets(k)
            if k in self.vertical:
                continue
            if k in self.two and all(self.high(n) for n in nets):
                self.series.add(k)
                a, b = sorted(nets, key=lambda n: (self.rail_index.get(n, 0), n))
                self.vertical[k], self.first[k] = False, a
            elif k not in self.two and sum(1 for p, n in self.pins[k] if self.high(n) and self.side(k, p) in ("L", "R")) >= 2 and \
                    not any(self.signal(n) for p, n in self.pins[k] if self.side(k, p) in ("L", "R")):
                self.series.add(k)
        self.uf = _UF()
        self.edges: set[tuple[str, str]] = set()
        self.bridge: dict[str, str] = {}
        self.jog: set[str] = set()
        uf = self.uf
        for k in sorted(self.series, key=_nat):
            for p, n in self.pins[k]:
                if self.high(n) and (k in self.two or self.side(k, p) in ("L", "R")):
                    uf.union("c:" + k, "n:" + n)
        self._attach()

        def row_net(n: str) -> bool:
            return self.signal(n) or n in self.rowed

        # hard edges from pin sides of glyphs that are not two-terminal
        for k in self.cells:
            if k in self.two:
                continue
            for p, n in self.pins[k]:
                if not row_net(n) or self.attach.get((k, p)) in ("local", "route", "row"):
                    continue
                s = self.side(k, p)
                if s == "T":
                    self._edge("n:" + n, "c:" + k, k)
                elif s == "B":
                    self._edge("c:" + k, "n:" + n, k)
        # vertical two-terminal parts: path columns, bundles, hanging, standing
        for k in [k for col in cover.columns for k in col] + sorted(partners, key=_nat) + cover.hanging + cover.standing:
            if k not in self.two or not self.vertical.get(k):
                continue
            top, bottom = self.two_nets(k)
            if row_net(top) and not self._edge("n:" + top, "c:" + k, k):
                continue
            if row_net(bottom):
                self._edge("c:" + k, "n:" + bottom, k)
        # feedback bridges: across the left and right side nets of one glyph
        rest = [k for k in self.two if k not in self.vertical and k not in self.series]
        sidenet: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for c in self.cells:
            if c in self.two:
                continue
            for p, n in self.pins[c]:
                if self.signal(n) and self.side(c, p) in ("L", "R"):
                    sidenet[n].add((c, self.side(c, p)))
        for r in list(rest):
            a, b = self.nets(r)
            if not (self.signal(a) and self.signal(b)):
                continue
            owner = next((c for c, sa in sorted(sidenet[a], key=lambda t: (_nat(t[0]), t[1])) if (c, OPPOSITE[sa]) in sidenet[b]), None)
            if owner is None:
                continue
            preds = sorted(u for u, v in self.edges if v == "c:" + owner)
            if not self._edge("c:" + r, "c:" + owner, r):
                continue
            for u in preds:
                self._edge(u, "c:" + r, r)
            left = a if (owner, "L") in sidenet[a] else b
            self.bridge[r] = owner
            self.vertical[r], self.first[r] = False, left
            rest.remove(r)
        # a summing bus: a signal net that 3 or more of the remaining parts end on is a row under them; each of those parts
        # stands from its own net down to the bus, under the one part (if any) that feeds that net
        on_two: dict[str, list[str]] = defaultdict(list)
        for r in rest:
            for n in sorted(set(self.nets(r))):
                on_two[n].append(r)
        count: dict[str, int] = defaultdict(int)
        for k in self.keys:
            for _, n in self.pins[k]:
                count[n] += 1
        buses = sorted(n for n, rs in on_two.items() if len(rs) >= 3 and self.signal(n) and n not in self.hub_nets)
        for bus in buses:
            for r in sorted(on_two[bus], key=_nat):
                m = next((x for x in self.nets(r) if x != bus), None)
                if r not in rest or m is None or not self.signal(m) or m in buses or not self._through("n:" + m, "c:" + r, "n:" + bus, r):
                    continue
                self.vertical[r], self.first[r] = True, m
                rest.remove(r)
                feeders = [f for f in on_two[m] if f != r and f in rest]
                if len(feeders) == 1 and count[m] == 2:
                    f = feeders[0]
                    top = next((x for x in self.nets(f) if x != m), None)
                    if top is not None and self.signal(top) and self._through("n:" + top, "c:" + f, "n:" + m, f):
                        self.vertical[f], self.first[f] = True, top
                        rest.remove(f)
        # every other two-terminal part: vertical when one net already lies above the other, else merged into one row
        stack = set()
        for j in self.conn:
            ps = [(p, n) for p, n in self.pins[j] if self.signal(n)]
            if 2 <= len(self.pins[j]) <= 3:
                for (_, a), (_, b) in zip(ps, ps[1:]):
                    stack.add((a, b))
        for r in sorted(rest, key=lambda r: (min(self.rank.get(n, 0) for n in self.nets(r)), _nat(r))):
            a, b = sorted(self.nets(r), key=lambda n: (self.rank.get(n, 0), n))
            if not row_net(a) or not row_net(b):
                self.vertical[r], self.first[r] = False, a
                continue
            A, B = "n:" + a, "n:" + b
            if not self.g[r].dc_blocking and (self._reach(A, B) or (a, b) in stack) and self._through(A, "c:" + r, B, r):
                self.vertical[r], self.first[r] = True, a
                continue
            if not self.g[r].dc_blocking and (self._reach(B, A) or (b, a) in stack) and self._through(B, "c:" + r, A, r):
                self.vertical[r], self.first[r] = True, b
                continue
            self.vertical[r], self.first[r] = False, a
            if self._reach(A, B) or self._reach(B, A):
                uf.union("c:" + r, A)
                self.jog.add(r)
                self.notes.append(f"columns: {r} would close a cycle in one row; drawn horizontal with a jog")
            else:
                uf.union(A, B)
                uf.union("c:" + r, A)
        for k in self.two:
            if k not in self.first:
                self.vertical[k], self.first[k] = False, self.nets(k)[0]
        # a transistor whose control net is only the tap between a standing part above and one below takes the tap's
        # row, so the control wire leaves the tap straight
        for k in self.cells:
            g = self.g[k]
            ctrl = g.control() if g.kind is Kind.THRU3 and k not in self.two and k not in self.series else None
            if ctrl is None or ctrl.number not in g.net or not self.signal(g.net[ctrl.number]) or self.side(k, ctrl.number) not in ("L", "R"):
                continue
            n = g.net[ctrl.number]
            on = [m for m in self.keys if m != k and n in self.nets(m)]
            if len(on) == 2 and all(m in self.two and self.vertical.get(m) and m not in self.bridge for m in on):
                a, c = "n:" + n, "c:" + k
                if uf.find(a) != uf.find(c) and not self._reach(a, c) and not self._reach(c, a):
                    uf.union(a, c)
        self._solve()

    def _classes(self) -> dict[str, set[str]]:
        succ: dict[str, set[str]] = defaultdict(set)
        for u, v in self.edges:
            a, b = self.uf.find(u), self.uf.find(v)
            if a != b:
                succ[a].add(b)
        return succ

    def _reach(self, a: str, b: str) -> bool:
        a, b = self.uf.find(a), self.uf.find(b)
        if a == b:
            return True
        succ = self._classes()
        seen, stack = {a}, [a]
        while stack:
            x = stack.pop()
            for y in succ.get(x, ()):
                if y == b:
                    return True
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return False

    def _edge(self, u: str, v: str, owner: str, quiet: bool = False) -> bool:
        """Add the hard edge u above v unless it would close a cycle (then a note, unless ``quiet``, and False)."""
        if (u, v) in self.edges:
            return True
        if self._reach(v, u):
            if not quiet:
                self.notes.append(f"columns: {owner}: {u[2:]} above {v[2:]} would close a cycle; dropped")
            return False
        self.edges.add((u, v))
        return True

    def _through(self, u: str, m: str, v: str, owner: str) -> bool:
        """u above m above v, both edges or neither."""
        fresh = (u, m) not in self.edges
        if not self._edge(u, m, owner, quiet=True):
            return False
        if self._edge(m, v, owner, quiet=True):
            return True
        if fresh:
            self.edges.discard((u, m))
        return False

    def _solve(self) -> None:
        """Longest path (sorted Kahn) and a pull-down pass; classes with no hard edge that hold a side net of a glyph take
        that glyph's row (at most 4 rounds)."""
        items = {"c:" + k for k in self.cells}
        for k in self.keys:
            for p, n in self.pins[k]:
                if self.signal(n) or n in self.rowed:
                    items.add("n:" + n)
        items |= {u for e in self.edges for u in e}
        uf = self.uf
        classes = sorted({uf.find(i) for i in items})
        succ = self._classes()
        linked = set(succ) | {b for s in succ.values() for b in s}
        side_cells: dict[str, set[str]] = defaultdict(set)
        for k in self.cells:
            if k in self.two:
                continue
            for p, n in self.pins[k]:
                if self.signal(n) and self.side(k, p) in ("L", "R"):
                    side_cells[uf.find("n:" + n)].add("c:" + k)

        def layer(minrow: dict[str, int]) -> dict[str, int]:
            indeg = {c: 0 for c in classes}
            for a in succ:
                for b in succ[a]:
                    indeg[b] += 1
            row = {c: minrow.get(c, 0) for c in classes}
            ready = [c for c in classes if not indeg[c]]
            heapq.heapify(ready)
            topo = []
            while ready:
                c = heapq.heappop(ready)
                topo.append(c)
                for b in sorted(succ.get(c, ())):
                    row[b] = max(row[b], row[c] + 1)
                    indeg[b] -= 1
                    if not indeg[b]:
                        heapq.heappush(ready, b)
            for c in reversed(topo):
                if succ.get(c) and not any(self.high(i[2:]) for i in items if i.startswith("n:") and uf.find(i) == c):
                    row[c] = max(row[c], min(row[b] for b in succ[c]) - 1)
            return row

        minrow: dict[str, int] = {}
        row = layer(minrow)
        for _ in range(4):
            want = {c: max(row[uf.find(x)] for x in side_cells[c]) for c in classes if c not in linked and side_cells.get(c)}
            if all(row[c] >= v for c, v in want.items()):
                break
            minrow.update(want)
            row = layer(minrow)
        self.row = {i: row[uf.find(i)] for i in items}

    # -- C5 columns

    def _columns(self) -> None:
        cover = self.cover
        col_uf = _UF()
        in_path = set()
        for col in cover.columns:
            ks = [k for k in col if self.g[k].kind is not Kind.CONN]
            in_path.update(ks)
            for k in ks[1:]:
                col_uf.union(ks[0], k)
        ups: dict[str, list[str]] = defaultdict(list)
        downs: dict[str, list[str]] = defaultdict(list)
        for k in self.cells:
            if k in self.bridge or k in self.two and not self.vertical.get(k):
                continue
            for p, n in self.pins[k]:
                if not self.signal(n):
                    continue
                s = self._pin_side(k, p)
                if s == "B":
                    ups[n].append(k)
                elif s == "T":
                    downs[n].append(k)
        rows_of: dict[str, set[int]] = defaultdict(set)
        for k in self.cells:
            rows_of[col_uf.find(k)].add(self.row["c:" + k])
        for n in sorted(set(ups) | set(downs)):
            u = sorted(ups[n], key=_nat)
            d = sorted(downs[n], key=_nat)
            if not u or not d or u[0] in in_path or d[0] in in_path:
                continue
            a, b = col_uf.find(u[0]), col_uf.find(d[0])
            if a != b and not rows_of[a] & rows_of[b] and max(rows_of[a]) < min(rows_of[b]):
                col_uf.union(a, b)
                rows_of[col_uf.find(a)] = rows_of.pop(a, set()) | rows_of.pop(b, set())
        cols: dict[str, list[str]] = defaultdict(list)
        self.hslots: list[str] = []
        for k in self.cells:
            if k in self.bridge:
                continue
            if k in self.two and not self.vertical.get(k):
                self.hslots.append(k)
                continue
            cols[col_uf.find(k)].append(k)
        for r, owner in self.bridge.items():
            cols[col_uf.find(owner)].append(r)
        self.cols: dict[str, list[str]] = {}
        for members in cols.values():
            ms = sorted(members, key=lambda k: (self.row["c:" + k], _nat(k)))
            self.cols["col:" + min(ms, key=_nat)] = ms

    # -- C6 order

    def _slot_members(self) -> dict[str, list[str]]:
        out = dict(self.cols)
        out.update({"h:" + k: [k] for k in self.hslots})
        out.update({"j:" + k: [k] for k in self.conn})
        return out

    def _order(self) -> None:
        members = self._slot_members()
        self.members = members
        slots = sorted(members, key=lambda s: _nat(members[s][0]))
        slot_of = {m: s for s, ms in members.items() for m in ms}
        self.slot_of = slot_of
        ins, outs = set(self.block.inputs), set(self.block.outputs)
        self.conn_side = {}
        only = not self.cells
        for j in self.conn:
            nets = set(self.nets(j))
            if only:  # a block of connectors: a source faces right, a sink left
                self.conn_side[j] = "in" if nets & outs or not nets & ins else "out"
            else:
                self.conn_side[j] = "out" if nets & outs and not nets & ins else "in"
        # spines: the first glyph with a vertical attachment on a net, else with a side attachment, else a connector
        spine: dict[str, str] = {}
        for pass_ in ("TB", "LR", "conn"):
            for k in self.keys:
                if k in self.bridge or k in self.two and not self.vertical.get(k) and pass_ != "conn" or k in self.series and pass_ == "LR":
                    continue
                if (pass_ == "conn") != (k in self.conn):
                    continue
                for p, n in self.pins[k]:
                    s = self._pin_side(k, p) if k not in self.conn else "L"
                    if n not in spine and (pass_ == "conn" or s in pass_):
                        spine[n] = slot_of[k]
        self.spine = spine
        prec: set[tuple[str, str]] = set()

        def flow(n: str) -> tuple:
            return (self.rail_index.get(n, -1) if self.high(n) else 100, self.rank.get(n, 0), n)

        for k in self.hslots:
            a, b = sorted(self.nets(k), key=flow)
            h = "h:" + k
            if a in spine and spine[a] != h:
                prec.add((spine[a], h))
            if b in spine and spine[b] != h:
                prec.add((h, spine[b]))
        for rep, partners in self.cover.bundles.items():
            for p in partners:
                if p not in self.conn and slot_of.get(rep) and slot_of.get(p) and slot_of[rep] != slot_of[p]:
                    prec.add((slot_of[rep], slot_of[p]))
        for j in self.conn:
            for s in slots:
                if s == "j:" + j or s.startswith("j:") and self.conn_side[s[2:]] == self.conn_side[j]:
                    continue
                prec.add(("j:" + j, s) if self.conn_side[j] == "in" else (s, "j:" + j))
        self.prec = self._acyclic(slots, prec)
        nbr: dict[str, list[str]] = defaultdict(list)
        on_net: dict[str, set[str]] = defaultdict(set)
        for s, ms in members.items():
            for m in ms:
                for n in self.nets(m):
                    if self.signal(n):
                        on_net[n].add(s)
        for n, ss in sorted(on_net.items()):
            for s in sorted(ss):
                nbr[s] += sorted(t for t in ss if t != s)

        def init(s: str) -> tuple:
            rails = [self.rail_index.get(n, 9) for m in members[s] for n in self.nets(m) if self.high(n)]
            ranks = [self.rank.get(n, 0) for m in members[s] for n in self.nets(m) if self.signal(n)]
            return (min(rails) if rails else 9, min(ranks) if ranks else 0)

        first = self._kahn(slots, {s: (*init(s), _nat(members[s][0])) for s in slots})
        order = first
        for _ in range(SWEEPS):
            pos = {s: i for i, s in enumerate(order)}
            key = {}
            for s in slots:
                bary = sum(pos[t] for t in nbr[s]) / len(nbr[s]) if nbr[s] else pos[s]
                key[s] = (0.5 * bary + 0.5 * pos[s], pos[s])
            new = self._kahn(slots, key)
            if new == order:
                break
            order = new
        if self.pair:
            order = self._symmetric(order)
        self.before = self.sketch_cost(order)
        self.order = self.greedy(order)
        self.sketch = self.sketch_cost(self.order)

    def _acyclic(self, slots: list[str], prec: set[tuple[str, str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        succ: dict[str, set[str]] = defaultdict(set)
        for a, b in sorted(prec, key=lambda e: (not e[0].startswith("j:") and not e[1].startswith("j:"), e)):
            seen, stack, cyc = {b}, [b], False
            while stack:
                x = stack.pop()
                if x == a:
                    cyc = True
                    break
                for y in succ[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if cyc:
                self.notes.append(f"columns: order {a[2:]} before {b[2:]} would close a cycle; dropped")
                continue
            out.add((a, b))
            succ[a].add(b)
        return out

    def _kahn(self, slots: list[str], key: dict[str, tuple]) -> list[str]:
        indeg = {s: 0 for s in slots}
        succ: dict[str, list[str]] = defaultdict(list)
        for a, b in sorted(self.prec):
            succ[a].append(b)
            indeg[b] += 1
        heap = [(key[s], _nat(self.members[s][0]), s) for s in slots if not indeg[s]]
        heapq.heapify(heap)
        out = []
        while heap:
            _, _, s = heapq.heappop(heap)
            out.append(s)
            for t in succ[s]:
                indeg[t] -= 1
                if not indeg[t]:
                    heapq.heappush(heap, (key[t], _nat(self.members[t][0]), t))
        return out

    def legal(self, order: list[str]) -> bool:
        pos = {s: i for i, s in enumerate(order)}
        return all(pos[a] < pos[b] for a, b in self.prec)

    def _sketch(self, order: list[str], only: set[str] | None = None) -> dict[str, list[tuple[float, float, float, float]]]:
        """Per signal net: its trunk on its row plus a drop per attachment, x = slot index (segments x0, y0, x1, y1)."""
        pos = {s: i for i, s in enumerate(order)}
        att: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for s, ms in self.members.items():
            x0 = pos[s]
            for m in ms:
                for p, n in self.pins[m]:
                    if not self.signal(n) or only is not None and n not in only:
                        continue
                    if m in self.conn:
                        att[n].append((x0, self.row.get("n:" + n, 0)))
                        continue
                    k = self.row["c:" + m]
                    side = self._pin_side(m, p)
                    dx = {"L": -0.4, "R": 0.4}.get(side, 0.0)
                    dy = {"T": -0.5, "B": 0.5}.get(side, 0.0)
                    att[n].append((x0 + dx, k + dy))
        out: dict[str, list[tuple[float, float, float, float]]] = {}
        for n, pts in att.items():
            if len(pts) < 2:
                continue
            y = self.row.get("n:" + n, pts[0][1])
            xs = [p[0] for p in pts]
            segs = [(min(xs), y, max(xs), y)]
            segs += [(px, y, px, py) for px, py in pts if abs(py - y) > 1e-9]
            out[n] = segs
        return out

    @staticmethod
    def _crossings(a: list, b: list) -> int:
        """Crossings between two nets' sketch segments; trunks of both on one row over a common stretch count as one."""
        c = 0
        for s in a:
            for t in b:
                if s[1] == s[3] == t[1] == t[3] and min(max(s[0], s[2]), max(t[0], t[2])) - max(min(s[0], s[2]), min(t[0], t[2])) > 1e-9:
                    c += 1
                    continue
                hs, vs = (s, t) if s[1] == s[3] and t[0] == t[2] else (t, s) if t[1] == t[3] and s[0] == s[2] else (None, None)
                if hs is None or hs[1] != hs[3] or vs[0] != vs[2] or hs[0] == hs[2] or vs[1] == vs[3]:
                    continue
                x, y = vs[0], hs[1]
                if min(hs[0], hs[2]) < x < max(hs[0], hs[2]) and min(vs[1], vs[3]) < y < max(vs[1], vs[3]):
                    c += 1
        return c

    def sketch_cost(self, order: list[str]) -> tuple[int, float]:
        segs = self._sketch(order)
        nets = sorted(segs)
        cross = sum(self._crossings(segs[a], segs[b]) for i, a in enumerate(nets) for b in nets[i + 1:])
        length = sum(abs(s[2] - s[0]) + abs(s[3] - s[1]) for ss in segs.values() for s in ss)
        return cross, round(length, 3)

    def greedy(self, order: list[str]) -> list[str]:
        """Up to :data:`PASSES` passes of adjacent switches that keep precedence and lower the sketch cost; only the nets of
        the two switched slots are re-evaluated."""
        segs = self._sketch(order)
        nets_of = {s: {n for m in ms for n in self.nets(m) if self.signal(n)} for s, ms in self.members.items()}

        def part(segs: dict, touched: set[str]) -> tuple[int, float]:
            cross = 0
            for a in sorted(touched & set(segs)):
                for b, sb in segs.items():
                    if b != a and (b not in touched or b > a):
                        cross += self._crossings(segs[a], sb)
            return cross, sum(abs(s[2] - s[0]) + abs(s[3] - s[1]) for n in touched if n in segs for s in segs[n])

        for _ in range(PASSES):
            improved = False
            for i in range(len(order) - 1):
                cand = order[:i] + [order[i + 1], order[i]] + order[i + 2:]
                if not self.legal(cand):
                    continue
                touched = nets_of[order[i]] | nets_of[order[i + 1]]
                old = part(segs, touched)
                new_segs = dict(segs)
                for n in touched:
                    new_segs.pop(n, None)
                new_segs.update(self._sketch(cand, touched))
                new = part(new_segs, touched)
                if (new[0], round(new[1], 3)) < (old[0], round(old[1], 3)):
                    order, segs, improved = cand, new_segs, True
            if not improved:
                break
        return order

    def _turn_controls(self) -> None:
        """A transistor's (or pot's) control pin faces the mean slot of the rest of its net (mirror y)."""
        pos = {s: i for i, s in enumerate(self.order)}
        for k in self.cells:
            g = self.g[k]
            if g.kind is not Kind.THRU3 or k in self.series:
                continue
            ctrl = g.control()
            if ctrl is None or ctrl.number not in g.net or not self.signal(g.net[ctrl.number]):
                continue
            n = g.net[ctrl.number]
            others = [pos[self.slot_of[m]] for m in self.keys for _, q in self.pins[m] if q == n and m != k]
            if not others:
                continue
            here, mean = pos[self.slot_of[k]], sum(others) / len(others)
            side = self.side(k, ctrl.number)
            want = "L" if mean < here else "R" if mean > here else side
            if side in ("L", "R") and side != want:
                flipped = self._mirror_y(k)
                if flipped is not None:
                    self.orient[k] = flipped

    def _mirror_y(self, key: str) -> Orient | None:
        swap = {"L": "R", "R": "L", "T": "T", "B": "B"}
        want = {p.number: swap[self.side(key, p.number)] for p in self.g[key].pins}
        return next((o for o in TRANSFORMS if all(self.side(key, p, o) == s for p, s in want.items())), None)


    # -- the closed-involution mirror

    def _pin_ids(self, key: str) -> list[tuple[tuple[str, int, int], str]]:
        g = self.g[key]
        return sorted(((g.pin(p).name, g.pin(p).dx, g.pin(p).dy), n) for p, n in self.pins[key])

    def _match(self, x: str, y: str, nmap: dict[str, str]) -> list[tuple[str, str]] | None:
        """The net pairs that map glyph x onto glyph y (same library symbol, pins matched by name and position; a
        two-terminal part with like-named pins either way round), consistent with ``nmap``; rails are fixed points."""
        gx, gy = self.g[x], self.g[y]
        if x == y or gx.symbol.lib_id != gy.symbol.lib_id or gx.kind is not gy.kind or x in self.conn or y in self.conn:
            return None
        px, py = self._pin_ids(x), self._pin_ids(y)
        if [i for i, _ in px] != [i for i, _ in py]:
            return None
        ways = [list(zip([n for _, n in px], [n for _, n in py]))]
        if x in self.two and len({i[0] for i, _ in px}) == 1:
            ways.append(list(zip([n for _, n in px], [n for _, n in reversed(py)])))
        for way in ways:
            seen = dict(nmap)
            ok = True
            for n, m in way:
                if not self.signal(n) or not self.signal(m):
                    ok = n == m
                elif seen.get(n, m) != m or seen.get(m, n) != n:
                    ok = False
                else:
                    seen[n], seen[m] = m, n
                if not ok:
                    break
            if ok:
                return way
        return None

    def _grow(self, a: str, b: str) -> dict[str, str]:
        """The glyph involution grown from the seed pair (a, b) by simultaneous breadth-first search over mapped nets."""
        nmap: dict[str, str] = {}
        gmap: dict[str, str] = {}
        on: dict[str, list[str]] = defaultdict(list)
        for k in self.cells:
            for n in sorted(set(self.nets(k))):
                on[n].append(k)

        def add(x: str, y: str, way: list[tuple[str, str]]) -> list[tuple[str, str]]:
            gmap[x], gmap[y] = y, x
            new = []
            for n, m in way:
                if self.signal(n) and n not in nmap:
                    nmap[n], nmap[m] = m, n
                    new.append((n, m))
            return new

        way = self._match(a, b, nmap)
        if way is None:
            return {}
        queue = deque(add(a, b, way))
        while queue:
            n, m = queue.popleft()
            if n == m:
                continue
            for x in on[n]:
                if x in gmap:
                    continue
                for y in on[m]:
                    if y in gmap:
                        continue
                    way = self._match(x, y, nmap)
                    if way is not None:
                        queue.extend(add(x, y, way))
                        break
        return gmap

    def _involution(self) -> dict[str, str]:
        """The largest closed involution of the block's glyphs seeded by two like glyphs that are not two-terminal (at
        least 4 glyphs); rails are fixed points. Empty when there is none."""
        seeds = [k for k in self.cells if k not in self.two]
        best: dict[str, str] = {}
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                if self.g[a].symbol.lib_id != self.g[b].symbol.lib_id:
                    continue
                m = self._grow(a, b)
                if len(m) > len(best) and self._coupled(a, b, m):
                    best = m
        return best if len(best) >= 4 else {}

    def _coupled(self, a: str, b: str, gmap: dict[str, str]) -> bool:
        """The two halves work together: the seeds share a signal net, or a path of signal nets the mirror does not fix
        joins them (a cross-coupling). Two like circuits that only share a rail or a common reference are not mirrored."""
        fixed = set()
        for x, y in gmap.items():
            for (_, n), (_, m) in zip(self._pin_ids(x), self._pin_ids(y)):
                if n == m:
                    fixed.add(n)
        if {n for n in self.nets(a) if self.signal(n)} & set(self.nets(b)):
            return True
        on: dict[str, list[str]] = defaultdict(list)
        for k in gmap:
            for n in set(self.nets(k)):
                if self.signal(n) and n not in fixed:
                    on[n].append(k)
        seen, stack = {a}, [a]
        while stack:
            k = stack.pop()
            if k == b:
                return True
            for n in set(self.nets(k)):
                for m in on.get(n, ()):
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
        return False

    def _symmetric(self, order: list[str]) -> list[str]:
        """C6 with the mirror: the paired slots as two mirrored halves (left halves in order, the right halves reversed),
        the other slots before, between or after them; the order of least sketch cost that keeps precedence, a mirrored
        horizontal part between its nets' spines either way. The halves become precedence for the greedy switch."""
        members, slot_of = self.members, self.slot_of
        pos0 = {s: i for i, s in enumerate(order)}
        sp: dict[str, str] = {}
        for s, ms in members.items():
            images = [self.pair.get(m) for m in ms]
            if s.startswith("j:") or None in images:
                continue
            ts = {slot_of.get(i) for i in images}  # type: ignore[arg-type]
            t = ts.pop() if len(ts) == 1 else None
            if t is None or t == s or len(members[t]) != len(ms):
                continue
            if any(self.row["c:" + m] != self.row["c:" + self.pair[m]] or self.vertical.get(m) != self.vertical.get(self.pair[m]) for m in ms):
                continue
            sp[s] = t
        sp = {s: t for s, t in sp.items() if sp.get(t) == s}
        if sum(len(members[s]) for s in sp) < 4:
            self.pair = {}
            return order
        pairs = sorted({tuple(sorted((s, t), key=lambda x: _nat(members[x][0]))) for s, t in sp.items()}, key=lambda p: _nat(members[p[0]][0]))
        hs = {s for s in sp if s.startswith("h:")}
        base = {(a, b) for a, b in self.prec if a not in hs and b not in hs}
        others = [s for s in order if s not in sp]
        group_nets = {n for s in sp for m in members[s] for n in self.nets(m) if self.signal(n)}
        near = [o for o in others if any(n in group_nets for m in members[o] for n in self.nets(m))]
        outs = set(self.block.outputs)

        def legal(seq: list[str]) -> bool:
            pos = {s: i for i, s in enumerate(seq)}
            if any(pos[a] >= pos[b] for a, b in base):
                return False
            for h in hs:
                ends = [pos[self.spine[n]] for n in self.nets(h[2:]) if n in self.spine and self.spine[n] != h]
                if len(ends) == 2 and not min(ends) < pos[h] < max(ends):
                    return False
            return True

        def firsts(seq: list[str]) -> dict[str, str]:
            pos = {s: i for i, s in enumerate(seq)}
            out = {}
            for h in hs:
                a, b = self.nets(h[2:])
                pa, pb = pos.get(self.spine.get(a, "")), pos.get(self.spine.get(b, ""))
                if pa is not None and pb is not None and pa != pb:
                    out[h[2:]] = a if pa < pb else b
            return out

        def cost(seq: list[str]) -> tuple[int, float]:
            saved = dict(self.first)
            self.first.update(firsts(seq))
            try:
                return self.sketch_cost(seq)
            finally:
                self.first.clear()
                self.first.update(saved)

        def hspan(seq: list[str]) -> int:
            """How far apart the nodes each mirrored coupling joins stand: a coupling sits between its two nodes."""
            pos = {s: i for i, s in enumerate(seq)}
            return sum(abs(pos[self.spine[a]] - pos[self.spine[b]]) for h in hs for a, b in [self.nets(h[2:])] if a in self.spine and b in self.spine)

        def inner(perm: tuple[str, ...]) -> int:
            """Larger slots (supply-to-ground columns) outboard, single parts inboard."""
            return sum(i * len(members[s]) for i, s in enumerate(perm))

        def build(lefts: list[str], zone: dict[str, int]) -> list[str]:
            return ([o for o in others if zone[o] == 0] + lefts + [o for o in others if zone[o] == 1]
                    + [sp[s] for s in reversed(lefts)] + [o for o in others if zone[o] == 2])

        mean = sum(pos0[s] for s in sp) / len(sp)
        start = {o: 0 if pos0[o] < mean else 2 for o in others}
        if len(pairs) <= 5:
            choices = [[p[(bits >> i) & 1] for i, p in enumerate(pairs)] for bits in range(2 ** len(pairs))]
        else:
            choices = [[min(p, key=lambda s: pos0[s]) for p in pairs]]
        best = None
        for c, lefts in enumerate(choices):
            lefts = sorted(lefts, key=lambda s: pos0[s])
            outward = sum(1 for s in lefts for m in members[s] for n in self.nets(m) if n in outs)
            inward = sum(1 for s in lefts for m in members[sp[s]] for n in self.nets(m) if n in outs)
            for perm in (itertools.permutations(lefts) if len(lefts) <= 3 else [tuple(lefts)]):
                zone = dict(start)
                found = None
                for _ in range(2):
                    for o in [None, *near]:
                        for z in ((None,) if o is None else (0, 1, 2)):
                            trial = dict(zone)
                            if o is not None:
                                trial[o] = z
                            seq = build(list(perm), trial)
                            if not legal(seq):
                                continue
                            key = (int(outward > inward), hspan(seq), inner(perm), *cost(seq))
                            if found is None or key < found[0]:
                                found, zone = (key, seq, list(perm), dict(trial)), dict(trial)
                if found is not None and (best is None or found[0] < best[0]):
                    best = found
        plain = self.sketch_cost(order)
        if best is None or best[0][3] > plain[0] + 1:
            why = "no order keeps precedence" if best is None else f"{best[0][3]} sketch crossings against {plain[0]}"
            self.notes.append(f"columns: mirror of {', '.join(f'{a[2:]}|{b[2:]}' for a, b in pairs)} not drawn ({why})")
            self.pair = {}
            return order
        _, seq, lefts, zone = best
        self.first.update(firsts(seq))
        rights = [sp[s] for s in lefts]
        self.half = {m: "L" for s in lefts for m in members[s]} | {m: "R" for s in rights for m in members[s]}
        self.pair = {m: self.pair[m] for m in self.half}
        self.sym = list(zip(lefts, rights))
        group = lefts + [o for o in others if zone[o] == 1] + list(reversed(rights))
        prec = set(base) | set(zip(group, group[1:]))
        prec |= {(o, group[0]) for o in others if zone[o] == 0} | {(group[-1], o) for o in others if zone[o] == 2}
        self.prec = prec
        halves = ", ".join(f"{'+'.join(members[a])}|{'+'.join(members[b])}" for a, b in self.sym)
        self.notes.append(f"columns: mirrored halves {halves}")
        return seq

    def _mirror_orient(self) -> None:
        """The right half's glyphs (not two-terminal) take the mirror image of their left partner's orientation."""
        swap = {"L": "R", "R": "L", "T": "T", "B": "B"}
        for x in sorted(self.half, key=_nat):
            if self.half[x] != "L" or x in self.two or x in self.conn:
                continue
            y = self.pair[x]
            ix = {(p.name, p.dx, p.dy): p.number for p in self.g[x].pins if p.number in self.g[x].net}
            iy = {(p.name, p.dx, p.dy): p.number for p in self.g[y].pins if p.number in self.g[y].net}
            want = {iy[i]: swap[self.side(x, n)] for i, n in ix.items() if i in iy}
            o = next((o for o in TRANSFORMS if all(self.side(y, p, o) == s for p, s in want.items())), None)
            if o is not None:
                self.orient[y] = o

    def _symmetric_x(self, xs: dict[str, int]) -> dict[str, int]:
        """Slot x mirrored about one axis: each pair as far from it as the wider of its two halves needs, the slots
        between the inner pair centred on it, the slots after the halves moved right by as much (no gap shrinks)."""
        order = self.order
        lefts = [a for a, _ in sorted(self.sym, key=lambda p: order.index(p[0]))]
        rights = [dict(self.sym)[a] for a in lefts]
        k = len(lefts) - 1
        d = [0] * len(lefts)
        span = xs[rights[k]] - xs[lefts[k]]
        i0, i1 = order.index(lefts[k]), order.index(rights[k])
        mids = order[i0 + 1:i1]
        d[k] = (span + 1) // 2
        if mids:  # room to centre the slots between the inner pair on the axis
            w = xs[mids[-1]] - xs[mids[0]]
            d[k] = max(d[k], (w + 1) // 2 + max(xs[mids[0]] - xs[lefts[k]], xs[rights[k]] - xs[mids[-1]]))
        d[k] += d[k] % 2
        for j in range(k - 1, -1, -1):
            d[j] = d[j + 1] + max(xs[lefts[j + 1]] - xs[lefts[j]], xs[rights[j]] - xs[rights[j + 1]])
        axis = xs[lefts[0]] + d[0]
        new = dict(xs)
        for j in range(len(lefts)):
            new[lefts[j]], new[rights[j]] = axis - d[j], axis + d[j]
        if mids:
            w = xs[mids[-1]] - xs[mids[0]]
            lo = new[lefts[k]] + xs[mids[0]] - xs[lefts[k]]
            hi = new[rights[k]] - (xs[rights[k]] - xs[mids[-1]]) - w
            at = axis - w // 2
            at -= at % 2
            at = min(max(at, lo), hi)
            for m in mids:
                new[m] = at + xs[m] - xs[mids[0]]
        delta = new[rights[0]] - xs[rights[0]]
        for s in order[order.index(rights[0]) + 1:]:
            new[s] = xs[s] + delta
        return new


    def _x_hints(self, d: Drawing) -> dict[str, list[Seg]]:
        """The cross-coupling of a mirrored transistor pair whose control pins face each other: each control net leaves
        its node on the other side of the axis straight down, runs 2 cells above (for the right glyph) or below (for the
        left one) the control row and enters its pin along the pin's axis, so the two runs cross once between the halves.
        Kept only where they pass no foreign pin, body, text or wire; the router draws the rest."""
        boxes = drawing_boxes(self.model, d)
        pts: dict[Pt, set[str]] = defaultdict(set)
        sides: dict[Pt, str] = {}
        whose: dict[Pt, set[str]] = defaultdict(set)
        for k, pl in d.place.items():
            g = self.g[k]
            for p in g.pins:
                q = pin_cell(g, p, (pl.x, pl.y), pl.orient)
                pts[q].add(g.net.get(p.number, ""))
                sides[q] = pin_side(g, p, pl.orient)
                whose[q].add(k)
        out: dict[str, list[Seg]] = defaultdict(list)

        def on(q: Pt, a: Pt, b: Pt) -> bool:
            return min(a[0], b[0]) <= q[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= q[1] <= max(a[1], b[1])

        def clear(net: str, segs: list[Seg]) -> bool:
            ends = whose[segs[0][0]] | whose[segs[-1][1]]  # the glyphs the run starts and ends at
            for a, b in segs:
                box = (mm(min(a[0], b[0])), mm(min(a[1], b[1])), mm(max(a[0], b[0])), mm(max(a[1], b[1])))
                if any(on(q, a, b) and net not in ns for q, ns in pts.items()):
                    return False
                for tag, bx in boxes:
                    pad = mm(1) if tag.startswith(("body:", "glyph:")) and tag.split(":", 1)[1] not in ends else 0.0
                    if box[0] < bx[2] + pad - 1e-9 and box[2] > bx[0] - pad + 1e-9 and box[1] < bx[3] + pad - 1e-9 and box[3] > bx[1] - pad + 1e-9:
                        return False
                for n, ss in list(d.hints.items()) + list(out.items()):
                    if n != net and any(on(c, a, b) or on(e, a, b) or on(a, c, e) or on(b, c, e) for c, e in ss):
                        return False
            return True

        for left, _ in self.sym:
            for x in self.members[left]:
                y = self.pair.get(x)
                gx = self.g[x]
                if y is None or gx.kind is not Kind.THRU3:
                    continue
                gy = self.g[y]
                cx, cy = gx.control(), gy.control()
                if cx is None or cy is None or cx.number not in gx.net or cy.number not in gy.net:
                    continue
                nx, ny = gx.net[cx.number], gy.net[cy.number]
                plx, ply = d.place[x], d.place[y]
                if nx == ny or not (self.signal(nx) and self.signal(ny)) or pin_side(gx, cx, plx.orient) != "R" or pin_side(gy, cy, ply.orient) != "L":
                    continue
                px, py = pin_cell(gx, cx, (plx.x, plx.y), plx.orient), pin_cell(gy, cy, (ply.x, ply.y), ply.orient)
                if px[1] != py[1]:
                    continue
                yb, twice = px[1], px[0] + py[0]

                def anchor(net: str, left_side: bool) -> Pt | None:
                    cands = [q for q, ns in pts.items() if net in ns and q not in (px, py) and q[1] < yb - 2
                             and (2 * q[0] < twice if left_side else 2 * q[0] > twice)]
                    return min(cands, key=lambda q: (sides.get(q) != "B", abs(2 * q[0] - twice), q)) if cands else None

                a, b = anchor(ny, True), anchor(nx, False)
                if a is None or b is None or not (px[0] + 4 <= a[0] and b[0] <= py[0] - 4):
                    continue
                leg_y = [(a, (a[0], yb - 2)), ((a[0], yb - 2), (py[0] - 2, yb - 2)), ((py[0] - 2, yb - 2), (py[0] - 2, yb)), ((py[0] - 2, yb), py)]
                leg_x = [(b, (b[0], yb + 2)), ((b[0], yb + 2), (px[0] + 2, yb + 2)), ((px[0] + 2, yb + 2), (px[0] + 2, yb)), ((px[0] + 2, yb), px)]
                if clear(ny, leg_y) and clear(nx, leg_x):
                    out[ny] += leg_y
                    out[nx] += leg_x
        # a connector bundled with a standing part: each pin's wire turns 2 cells out of the pin and meets the part's
        # node wire 2 cells beyond the part's end
        for rep, partners in sorted(self.cover.bundles.items()):
            if rep not in self.two or not self.vertical.get(rep) or rep not in d.place:
                continue
            pr = d.place[rep]
            ends = {self.g[rep].net[p.number]: pin_cell(self.g[rep], p, (pr.x, pr.y), pr.orient) for p in self.g[rep].pins if p.number in self.g[rep].net}
            for j in partners:
                if j not in self.conn or j not in d.place:
                    continue
                pj = d.place[j]
                runs = []
                for p, n in self.pins[j]:
                    if n not in ends:
                        continue
                    q = pin_cell(self.g[j], p, (pj.x, pj.y), pj.orient)
                    side = pin_side(self.g[j], p, pj.orient)
                    e = ends[n]
                    step = -2 if side == "L" else 2 if side == "R" else 0
                    top = e[1] < sum(v[1] for v in ends.values()) / len(ends)
                    y = e[1] - 2 if top else e[1] + 2
                    if not step or (q[0] - e[0]) * step >= 0 or abs(q[0] + step - e[0]) < 4:
                        runs = []
                        break
                    runs.append((n, [(q, (q[0] + step, q[1])), ((q[0] + step, q[1]), (q[0] + step, y)), ((q[0] + step, y), (e[0], y))]))
                if len(runs) == 2 and all(clear(n, segs) for n, segs in runs):
                    for n, segs in runs:
                        out[n] += [s_ for s_ in segs if s_[0] != s_[1]]
        return out

    # -- C7 coordinates

    def _reaches(self) -> dict[str, tuple[int, int]]:
        """Per horizontal part: the slot range its nets span (every slot with a pin on one of them)."""
        pos = {s: i for i, s in enumerate(self.order)}
        out = {}
        for k in self.hslots:
            if k in self.half:  # a mirrored half's own coupling: the X leaves it, no other part runs along its row there
                here = pos["h:" + k]
                out[k] = (here, here)
                continue
            nets = set(self.nets(k))
            xs = [pos[self.slot_of[m]] for m in self.keys for _, n in self.pins[m] if n in nets]
            out[k] = (min(xs), max(xs))
        return out

    def _lanes(self) -> dict[str, int]:
        """The rows, with an extra row for each class of horizontal parts whose slot range overlaps another's in the same
        row (the later one moves just below)."""
        reach = self._reaches()
        uf = self.uf
        span: dict[str, tuple[int, int]] = {}
        for k in self.hslots:
            c = uf.find("c:" + k)
            lo, hi = span.get(c, reach[k])
            span[c] = (min(lo, reach[k][0]), max(hi, reach[k][1]))
        lane: dict[str, int] = {}
        by_row: dict[int, list[str]] = defaultdict(list)
        for c in sorted(span, key=lambda c: (span[c], c)):
            by_row[self.row[c]].append(c)
        width = {}
        for r, cs in by_row.items():
            ends: list[int] = []
            for c in cs:
                lo, hi = span[c]
                k = next((i for i, e in enumerate(ends) if e <= lo), None)
                if k is None:
                    ends.append(hi)
                    k = len(ends) - 1
                else:
                    ends[k] = hi
                lane[c] = k
            width[r] = len(ends)
        extra, shift = 0, {}
        for r in range(max(self.row.values(), default=0) + 1):
            shift[r] = extra
            extra += max(width.get(r, 1), 1) - 1
        if extra:
            note = f"columns: {extra} extra row(s) for horizontal parts that would share a row over the same slots"
            if note not in self.notes:
                self.notes.append(note)
        return {i: r + shift[r] + lane.get(uf.find(i), 0) for i, r in self.row.items()}


    def drawings(self) -> list[Drawing]:
        out = []
        for flip in (False, True):
            try:
                out.append(self.draw(flip))
            except LayoutFailure:
                if not flip:
                    raise
        return out

    def _field_sides(self, k: str, flip: bool) -> list[str]:
        g = self.g[k]
        if k in self.conn:
            return ["L", "R", "T", "B"] if self.conn_side[k] == "in" else ["R", "L", "T", "B"]
        if k in self.two:
            if self.vertical.get(k):
                if k in self.half:  # a mirrored half: fields face outward
                    return ["L", "R"] if self.half[k] == "L" else ["R", "L"]
                return ["L", "R"] if flip else ["R", "L"]
            return ["T", "B"]
        if g.kind is Kind.THRU3:
            ctrl = g.control()
            s = self.side(k, ctrl.number) if ctrl is not None else "L"
            other = OPPOSITE[s]  # type: ignore[index]
            return [other] + [x for x in ("R", "L", "T", "B") if x != other]
        return ["R", "T", "L", "B"]

    def _place(self, k: str, flip: bool) -> tuple[Place, Pt, tuple[Generated, ...], tuple[Seg, ...], BoxMM, tuple[Pt, Pt] | None]:
        """The glyph at a trial origin: its place, anchor (cells), local power symbols with their stubs, and its measured
        box including fields and symbols; the field side is the first that clears its own pin wires."""
        g = self.g[k]
        if k in self.two:
            o = orient_two(g, self.first[k], bool(self.vertical.get(k)))
        elif k in self.conn:
            sides = [pin_side(g, p.number, Orient()) for p in g.pins]
            want = "R" if self.conn_side[k] == "in" else "L"
            o = Orient() if sides.count(want) >= sides.count(OPPOSITE[want]) else Orient(0, "y")  # type: ignore[index]
            self.orient[k] = o
        else:
            o = self.orient[k]
        self.orient[k] = o
        pts = {p: pin_cell(g, p, (0, 0), o) for p, _ in self.pins[k]}
        if k in self.two:
            a, b = (pts[p] for p, _ in self.pins[k])
            anchor = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        elif g.kind is Kind.THRU3 and g.through() is not None and pin_side(g, g.through()[0].number, o) in ("T", "B"):
            t = pin_cell(g, g.through()[0].number, (0, 0), o)
            anchor = (t[0], 0)
        else:
            anchor = (0, 0)
        glyphs, stubs, taken = [], [], [body_box(g, Place(0, 0, o))]
        self.terminals[k] = [stub_label(n, pts[p], pin_side(g, p, o)) for p, n in self.pins[k] if self.attach.get((k, p)) == "label"]
        taken += [label_box(lab) for lab in self.terminals[k]]
        for p, n in self.pins[k]:
            if self.attach.get((k, p)) != "local":
                continue
            kind = "GND" if self.role[n] is Role.GROUND else "NEG" if self.role[n] is Role.NEGATIVE else "VCC"
            gen, segs = rails_mod.local_glyph(self.ctx.library, kind, n, pts[p], pin_side(g, p, o), f"#PWR{len(glyphs) + 1:02d}")
            boxes = [generated_boxes(gen)[0], *generated_boxes(gen)[1]]
            if any(self._hit(a, b) for a in boxes for b in taken):  # no room beside its neighbours: a label names the pin
                lab = stub_label(n, pts[p], pin_side(g, p, o))
                self.attach[(k, p)] = "label"
                self.terminals[k].append(lab)
                taken.append(label_box(lab))
                continue
            glyphs.append(gen)
            stubs += segs
            taken += boxes
        own = taken[1:]
        escape = []
        for p, n in self.pins[k]:
            q, s = pts[p], pin_side(g, p, o)
            d = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}[s]
            far = (q[0] + d[0] * (2 + FIELD_GAP), q[1] + d[1] * (2 + FIELD_GAP))
            pad = mm(FIELD_GAP)
            escape.append((min(mm(q[0]), mm(far[0])) - (pad if s in "TB" else 0), min(mm(q[1]), mm(far[1])) - (pad if s in "LR" else 0),
                           max(mm(q[0]), mm(far[0])) + (pad if s in "TB" else 0), max(mm(q[1]), mm(far[1])) + (pad if s in "LR" else 0)))
        keep = own + escape
        place, anchors = None, None
        for side in self._field_sides(k, flip):
            trial = Place(0, 0, o, side)  # type: ignore[arg-type]
            m = measure(g, trial)
            if not any(self._hit(f, b) for f in m.fields for b in keep) and not overlaps(list(m.fields)):
                place = trial
                break
        if place is None:  # no side is clear of the glyph's own pin wires: fields above, shifted right until they are
            place = Place(0, 0, o, "T")
            base = field_anchors(measure(g, place).field_nodes)
            for shift in range(1, 16):
                trial_anchors = ((base[0][0] + shift, base[0][1]), (base[1][0] + shift, base[1][1]))
                m = measure(g, place, anchors=trial_anchors)
                if not any(self._hit(f, b) for f in m.fields for b in keep) and not overlaps(list(m.fields)):
                    anchors = trial_anchors
                    break
        if anchors is None:  # fixed here: the grid snap of a field depends on where the glyph stands
            anchors = field_anchors(measure(g, place).field_nodes)
        m = measure(g, place, anchors=anchors)
        box = union([m.body, *m.fields, *own, *(rails_mod._seg_box(*s) for s in stubs)])
        return place, anchor, tuple(glyphs), tuple(stubs), box, anchors

    @staticmethod
    def _hit(a: BoxMM, b: BoxMM) -> bool:
        return min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9

    def draw(self, flip: bool, extra: int = 0) -> Drawing:
        for attempt in range(REPAIRS + 1):
            d = self._coordinates(flip, GAP + attempt * 2 + extra)
            boxes = drawing_boxes(self.model, d)
            pairs = overlaps([b for _, b in boxes])
            if not pairs:
                return d
        a, b = pairs[0]
        raise LayoutFailure(f"columns: {boxes[a][0]} overlaps {boxes[b][0]}", self.block.id)

    def _coordinates(self, flip: bool, gap: int) -> Drawing:
        cells = {k: self._place(k, flip) for k in self.keys}
        ext = {}
        for k, (place, anchor, _, _, box, _) in cells.items():
            ax, ay = anchor
            x0, y0, x1, y1 = to_cells(box)
            ext[k] = (ax - x0, x1 - ax, ay - y0, y1 - ay)  # left, right, up, down
        row = self._lanes()
        rows = max(row.values(), default=0) + 1
        up, down = [0] * rows, [0] * rows
        for k in self.cells:
            r = row["c:" + k]
            up[r] = max(up[r], ext[k][2])
            down[r] = max(down[r], ext[k][3])
        ys = [0] * rows
        cell_rows = {row["c:" + k] for k in self.cells}
        net_rows = {r for i, r in row.items() if i.startswith("n:") and self.signal(i[2:])}
        for r in range(rows):
            # a cell row under a row of signal nets only: 4 cells of wire down to it, room for a name label beside
            step = NET_GAP if r - 1 in net_rows and r - 1 not in cell_rows and r in cell_rows else gap
            y = up[r] if r == 0 else ys[r - 1] + down[r - 1] + step + up[r]
            ys[r] = y + (y % 2)
        cy = {k: ys[row["c:" + k]] for k in self.cells}
        # a horizontal part tied to exactly one side pin takes that pin's height, unless another horizontal part already
        # runs there over the same slots
        side_pin_y: dict[str, list[int]] = defaultdict(list)
        for k in self.cells:
            if k in self.two:
                continue
            place, anchor = cells[k][0], cells[k][1]
            for p, n in self.pins[k]:
                if self.signal(n) and pin_side(self.g[k], p, place.orient) in ("L", "R"):
                    q = pin_cell(self.g[k], p, (0, 0), place.orient)
                    side_pin_y[n].append(cy[k] + q[1] - anchor[1])
        reach = self._reaches()
        taken: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for k in sorted(self.hslots, key=lambda k: self.order.index("h:" + k)):
            tied = [side_pin_y[n] for n in self.nets(k) if len(side_pin_y.get(n, ())) == 1]
            lo, hi = reach[k]
            if k not in self.half and len(tied) == 1 and not any(lo < b and a < hi for a, b in taken[tied[0][0]]):
                cy[k] = tied[0][0]
            taken[cy[k]].append((lo, hi))
        # net heights: the pins of horizontal parts on the net, else side pins, else the row
        net_y: dict[str, int] = {}
        for k in self.hslots:
            for n in self.nets(k):
                net_y.setdefault(n, cy[k])
        for n, ys_ in side_pin_y.items():
            net_y.setdefault(n, ys_[0])
        for item, r in row.items():
            if item.startswith("n:"):
                net_y.setdefault(item[2:], ys[r])
        for j in self.conn:
            place, anchor = cells[j][0], cells[j][1]
            rep = next((r for r, ps in sorted(self.cover.bundles.items()) if j in ps and r in self.two and self.vertical.get(r)), None)
            qs = [pin_cell(self.g[j], p, (0, 0), place.orient)[1] for p, n in self.pins[j] if rep is not None and n in self.nets(rep)]
            if len(qs) >= 2:  # a connector bundled with a standing part: centred on it, a wire from each end of the part
                cy[j] = cy[rep] - (min(qs) + max(qs)) // 2
                continue
            for p, n in self.pins[j]:
                if n in net_y:
                    q = pin_cell(self.g[j], p, (0, 0), place.orient)
                    cy[j] = net_y[n] - q[1]
                    break
            else:
                cy[j] = ys[0]
        spans = {k: (cy[k] - ext[k][2], cy[k] + ext[k][3]) for k in self.keys}
        # slot x by longest path over measured extents; a pin facing right is left of what it joins, a pin facing left right
        off: dict[str, dict[str, tuple[int, str]]] = {}
        for k in self.keys:
            pl, anchor = cells[k][0], cells[k][1]
            off[k] = {}
            for p, n in self.pins[k]:
                if not self.low(n) and self.attach.get((k, p)) not in ("local", "label"):
                    off[k].setdefault(n, (pin_cell(self.g[k], p, (0, 0), pl.orient)[0] - anchor[0], pin_side(self.g[k], p, pl.orient)))

        def facing(a: str, b: str) -> int:
            """How far right of a's anchor b's anchor must be for the pins they share to face each other."""
            need = -10**9
            extra = BUNDLE_GAP if b in self.cover.bundles.get(a, ()) and b in self.conn else 0
            for n in set(off[a]) & set(off[b]):
                (xa, sa), (xb, sb) = off[a][n], off[b][n]
                if sa == "R" or sb == "L":
                    need = max(need, xa + FACING + extra - xb)
            return need

        xs: dict[str, int] = {}
        x_of: dict[str, int] = {}
        for i, s in enumerate(self.order):
            lo = max(ext[m][0] for m in self.members[s])
            for t in self.order[:i]:
                lo = max(lo, xs[t] + 2)
                for a in self.members[t]:
                    for b in self.members[s]:
                        lo = max(lo, xs[t] + facing(a, b))
                        if spans[a][0] - gap < spans[b][1] and spans[b][0] - gap < spans[a][1]:
                            wire = WIRE_GAP if set(self.nets(a)) & set(self.nets(b)) & self.signals else gap
                            lo = max(lo, xs[t] + ext[a][1] + max(gap, wire) + ext[b][0])
            xs[s] = lo + (lo % 2)
        # a horizontal part centred between its nets' spines when that keeps clear of its neighbours
        for i, s in enumerate(self.order):
            if not s.startswith("h:") or s[2:] in self.half:
                continue
            k = s[2:]
            ends = [xs[self.spine[n]] for n in self.nets(k) if n in self.spine and self.spine[n] != s]
            if len(ends) != 2:
                continue
            target = (ends[0] + ends[1]) // 2
            lo = xs[s]
            hi = min([xs[t] - 2 for t in self.order[i + 1:]] + [10**9])
            for t in self.order[i + 1:]:
                for a in self.members[t]:
                    hi = min(hi, xs[t] - facing(k, a))
                    if spans[a][0] - gap < spans[k][1] and spans[k][0] - gap < spans[a][1]:
                        wire = WIRE_GAP if set(self.nets(a)) & set(self.nets(k)) & self.signals else gap
                        hi = min(hi, xs[t] - ext[a][0] - max(gap, wire) - ext[k][1])
            if lo <= target <= hi:
                xs[s] = target - (target % 2) if target - (target % 2) >= lo else target
        if self.half:
            xs = self._symmetric_x(xs)
        for s, ms in self.members.items():
            for m in ms:
                x_of[m] = xs[s]
        place: dict[str, Place] = {}
        fields: dict[str, tuple[Pt, Pt]] = {}
        glyphs: list[Generated] = []
        labels: list[Label] = []
        hints: dict[str, list[Seg]] = defaultdict(list)
        for k in self.keys:
            pl, anchor, gens, stubs, _, anchors = cells[k]
            dx, dy = x_of[k] - anchor[0], cy[k] - anchor[1]
            place[k] = pl.moved(dx, dy)
            labels += [lab.moved(dx, dy) for lab in self.terminals.get(k, ())]
            if anchors is not None:
                fields[k] = ((anchors[0][0] + dx, anchors[0][1] + dy), (anchors[1][0] + dx, anchors[1][1] + dy))
            for gen, (a, b) in zip(gens, stubs):
                glyphs.append(replace(gen.moved(dx, dy), ref=f"#PWR{len(glyphs) + 1:02d}"))
                hints[gen.net].append(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)))
        rails = self._rail_spans(place, ys, row)
        d = Drawing(block=self.block.id, recipe="columns", place=place, hints={n: tuple(s) for n, s in sorted(hints.items())},
                    rails=tuple(rails), notes=tuple(self.notes), sketch=(self.sketch[0], round(self.sketch[1] * 10)), glyphs=tuple(glyphs),
                    fields=fields, labels=tuple(labels))
        cross = self._x_hints(d)
        if cross:
            merged = {n: list(ss) for n, ss in d.hints.items()}
            for n, ss in cross.items():
                merged.setdefault(n, []).extend(ss)
            d = replace(d, hints={n: tuple(ss) for n, ss in sorted(merged.items())})
        boxes = [b for _, b in drawing_boxes(self.model, d)]
        segs = [s for ss in d.hints.values() for s in ss]
        boxes += [rails_mod._seg_box(*s) for s in segs]
        d = replace(d, box=to_cells(union(boxes)))
        return replace(d, ports=blocks_mod.ports(self.model, self.block, d))

    def _rail_spans(self, place: dict[str, Place], ys: list[int], row: dict[str, int]) -> list[RailSpan]:
        """Where each supply's line runs in this drawing: at the pin of a part in series on it (the line leaves that pin),
        else on its row; from the leftmost to the rightmost pin that faces it."""
        out = []
        for n in sorted(self.rowed, key=lambda n: (self.rail_index.get(n, 0), n)):
            pts, y, source = [], None, None
            for k in self.cells:
                for p, m in self.pins[k]:
                    if m != n or self.attach.get((k, p)) not in ("line", "row"):
                        continue
                    pl = place[k]
                    q = pin_cell(self.g[k], p, (pl.x, pl.y), pl.orient)
                    pts.append(q[0])
                    if self.attach[(k, p)] == "row" and y is None:
                        y = q[1]
                    if self.attach[(k, p)] == "line" and pin_side(self.g[k], p, pl.orient) == "B" and source is None:
                        source = f"{self.g[k].ref}.{p}"
            if not pts:
                continue
            if y is None:
                y = ys[row["n:" + n]]
            out.append(RailSpan(n, y, min(pts), max(pts), source))
        return out
