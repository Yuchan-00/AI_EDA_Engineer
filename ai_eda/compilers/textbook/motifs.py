"""Motif recipes: op-amp stage, cross-coupled pair, repeated-cell array and supply chain.

Every recipe detects its motif by library geometry, pin types and names and connectivity only (never a template, block
or topology name; values only for the 'verified swap' note), claims the glyphs of the motif, and draws block-relative
variants with wire hints. A recipe whose own check fails (boxes overlap, a hint wire meets a foreign pin or runs through
a text it does not own) raises LayoutFailure, and the block falls to the next recipe with a note. A port where another
block joins a net sits on the net's trunk (a wire node before a part's pin end), so the neighbour lines up with it.

- OPAMP_STAGE, one stage per amplifier unit. The feedback input is the input whose net holds a two-terminal part to the
  unit's output net (or is the output net: a follower). A unit with power pins stays at rotation 0; a unit without is
  mirrored (x) when that puts the feedback input on top. Feedback parts lie on rows above the unit's box (fields and
  supply symbols included; the unit's fields stand beside its supply symbol when they fit), between a riser from the
  feedback node and one from the output node, resistors nearest. Two-terminal parts from the feedback input to a rail
  stand left of the unit (a shunt column with its own ground symbol); parts to another signal lie on the feedback-input
  row to the left (inputs; a second and third input stacked under the first, all meeting one riser). The other input:
  on ground, a 2-cell stub with a ground symbol hanging below it (the feedback riser passes outside the stub end and its
  symbol); on a supply, its own symbol; one part to a rail (a bias), on that input's row with the rail symbol at its far
  end; the tap of a private series chain (a potentiometer wiper, the node between two parts, a divider between the
  rails with its supply end left), the chain on a row on that input's side of the unit (under it, or over it when that
  input is on top) with the part toward the output on the right and the far ends of every stage joined by a bus line
  through them; anything else is left to the router. When the other input sits above the feedback input and is biased,
  tapped from a chain or wired straight in from a connector, the stage's rule puts the feedback below the unit; the
  recipe draws the block with every feedback above too and keeps the better by score (crossings, then wire), above on a
  tie, and notes the choice. A feedback node driven by followers whose control pins sit on the output (a push-pull inside
  the loop) is drawn with them: the output riser feeding both control pins, the upper follower on the supply and the
  lower on ground, the bias parts between the two nodes, the feedback wire over the top, the load in line to its
  connector; when the followers' node is not the feedback input but joined to it by two-terminal parts (an inverting
  stage around a push-pull), those parts lie on rows over the top in place of the wire. Every stage of a block is drawn in one row (natural order), the units at one height, room kept for an
  output's name before the next stage; a package's power unit stands right of its last working stage (under the unit of a
  summing bus whose output runs on to another block, so that output runs straight on) and spare units (an
  output no other part uses) follow in a strip at the end; the outputs of a bank of three or more stages that leave the
  block are named by label pairs where they leave. A connector whose signal nets are the stages' inputs and outputs
  stays on the input side: the first input pin level with its input row (its name above the wire, or under it when a
  pin sits right above), the other inputs and the outputs named by label pairs (at the pin and at the input part or the
  end of the output trunk), rail pins by their symbol (a ground on a stub past the stubs and names of the pins below
  when the next pin is in the way) or a label.
- ARRAY. A summing bus: three or more branches of the same library symbols in series (private middle nets), each from
  its own input net to one common signal net (a part to the output of an amplifier with an input on that net is its
  feedback, never a branch; a part joining two branch inputs makes a ladder, not a bus), drawn as columns in natural
  order at one pitch over the bus line, the amplifier whose feedback input is the bus (if any) at its right end as an
  op-amp stage, a connector of the branch inputs on the left; branch inputs that leave the block or come from that
  connector are named by label pairs at the branches. A chain ladder: three or more two-terminal cells of one library
  symbol between one common net and their own taps, the taps chained by links of one library symbol (an end part allowed
  on the first or last tap), ordered along the chain from the end whose cell has the lowest reference; it is drawn by the
  columns engine (the common line above, the cells hanging, the links on the tap row), named array. A series ladder: a
  path of two-terminal parts from a block input to a block output (a potentiometer with both ends on two different nets
  may start it, entered at one end and left at its wiper, its other end a tail to a rail; a mounting pin on a rail gets
  its own symbol), every other part a shunt from
  a node of the path; drawn on one row with the shunts hanging to their rails, or standing up to a net that leaves.
- CROSS_COUPLED. Two like three-terminal transistors whose top nets each drive the other's control net through one
  coupling part of one library symbol (load and bias slots optional, emitters on a rail or one part to it): four columns
  at one pitch p (outer loads, inner bias nodes), the couplings on the node row, the transistors under the outer columns
  facing inward, and the X as two 45-degree diagonals (dx = dy = p) from the inner nodes to the opposite control pins,
  the only non-orthogonal wires; the half whose top net is a block output on the right. A block holding more than the
  pair is drawn by the columns engine's mirror instead, its orthogonal X turned into diagonals where they clear
  everything (noted). The note says 'verified swap' when the mirrored parts' values match.
- SUPPLY_CHAIN. A block whose every glyph has a pin on a supply rail or on a private net next to one, with at least one
  rail derivation element or source connector and no amplifier or IC. When its rails form one derivation path with one
  horizontal bridge between neighbours (a part, a switch, a regulator) and every other glyph is a shunt to ground (one
  part or two in series) or the source connector, it is one line left to right: the connector, then per rail its shunts
  hanging and its supply symbol on a stub up from the line, then the bridge in the line; the connector's pins of one
  other rail on one side tied over their stub ends to one symbol. A bridge may be a series pass: two three-terminal
  parts whose through pairs lead from one rail over a signal net to the next (back-to-back MOSFETs of a reverse
  protection switch), in the line with their control pins down. Their control network (every net of it the net between
  them, a control net, a rail of the line, ground or private) hangs under them on one gate line joining the control
  pins: the parts from the net between them to a control net hang in the gap between them, side by side; a path from
  that net through private nets to a control net (a resistor, a switch with its throw on ground, a resistor) hangs
  beside them, each part outward of the pin it hangs from; parts from a control net to the downstream rail hang from
  the line after the pass; anything else is drawn by the columns engine under the gate line. Otherwise (a follower
  bridge, a pot) the columns engine's rail rows, named supply_chain, with a note.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook import blocks as blocks_mod
from ai_eda.compilers.textbook import rails as rails_mod
from ai_eda.compilers.textbook.glyphs import TRANSFORMS, body_box, drawing_box, drawing_boxes, field_anchors, field_plan, generated_boxes, measure, mm, orient_two, overlaps, pin_cell, pin_side, to_cells, union
from ai_eda.compilers.textbook.labels import stub_label
from ai_eda.compilers.textbook.model import BlockInfo, Drawing, Generated, Glyph, Kind, Label, LayoutFailure, Model, Orient, Place, Pt, RailSpan, Role, Seg, Sheet, Side, Wiring, mounting

GAP = 4  # cells of wire between two parts in a row (room for a name label)
FB_GAP = 2  # cells from the top of the unit's box (fields and supply symbols included) up to its feedback row
CHAIN_DROP = 8  # cells from the bottom of the unit's box down to its chain row (room for the chain parts' fields)
STAGE_GAP = 6  # cells between the boxes of two stages in a row
BUS_DROP = 2  # cells from a summing branch's foot down to the bus line
MAX_CHAIN = 4  # parts on each side of a chain's tap
SUM_BRANCHES = 3  # branches that make a summing bus
SUM_KEEP = 3  # a feedback input with this many input parts is a summing bus (left to the array recipe)


def _nat(key: str) -> tuple:
    return natural_ref_key(key)


def _nets(g: Glyph) -> list[str]:
    return sorted(set(g.net.values()))


def _two(g: Glyph) -> bool:
    return g.kind is Kind.TWO and len(g.net) == 2 and len(set(g.net.values())) == 2


def _rail(model: Model, net: str) -> bool:
    return model.nets[net].role is not Role.SIGNAL


def _other(g: Glyph, net: str) -> str:
    return next(n for n in g.net.values() if n != net)


# --------------------------------------------------------------------------- the pen: one block-relative drawing


class Pen:
    """A drawing under construction: places with fixed field anchors, hint wires, local power symbols and labels."""

    def __init__(self, ctx, block: BlockInfo, recipe: str) -> None:
        self.ctx = ctx
        self.model: Model = ctx.model
        self.block = block
        self.recipe = recipe
        self.place: dict[str, Place] = {}
        self.fields: dict[str, tuple[Pt, Pt]] = {}
        self.hints: dict[str, list[Seg]] = defaultdict(list)
        self.glyphs: list[Generated] = []
        self.labels: list[Label] = []
        self.rails: list = []  # RailSpan: rail lines the sheet draws over this drawing's top pins

    def g(self, key: str) -> Glyph:
        return self.model.glyphs[key]

    def put(self, key: str, at: Pt, o: Orient, side: Side = "R", anchors: tuple[Pt, Pt] | None = None) -> None:
        place = Place(at[0], at[1], o, side)
        self.place[key] = place
        self.fields[key] = anchors if anchors is not None else field_anchors(measure(self.g(key), place).field_nodes)

    def put_two(self, key: str, first: str, mid: Pt, vertical: bool, side: Side) -> None:
        """A two-terminal part with the pin on net ``first`` above (vertical) or left of the other, centred on ``mid``."""
        g = self.g(key)
        o = orient_two(g, first, vertical)
        a, b = (pin_cell(g, p, (0, 0), o) for p in sorted(g.net))
        self.put(key, (mid[0] - (a[0] + b[0]) // 2, mid[1] - (a[1] + b[1]) // 2), o, side)

    def pin(self, key: str, number: str) -> Pt:
        pl = self.place[key]
        return pin_cell(self.g(key), number, (pl.x, pl.y), pl.orient)

    def on(self, key: str, net: str) -> Pt:
        g = self.g(key)
        return self.pin(key, next(p for p, n in sorted(g.net.items(), key=lambda kv: _nat(kv[0])) if n == net))

    def side(self, key: str, number: str) -> Side:
        return pin_side(self.g(key), number, self.place[key].orient)

    def wire(self, net: str, *pts: Pt) -> None:
        for a, b in zip(pts, pts[1:]):
            if a != b:
                self.hints[net].append((a, b))

    def local(self, key: str, number: str) -> Generated:
        """The power symbol of one rail pin: ground / negative hanging, supply standing (see rails.local_glyph)."""
        g = self.g(key)
        net = g.net[number]
        role = self.model.nets[net].role
        kind = "GND" if role is Role.GROUND else "NEG" if role is Role.NEGATIVE else "VCC"
        gen, stubs = rails_mod.local_glyph(self.ctx.library, kind, net, self.pin(key, number), self.side(key, number), f"#PWR{len(self.glyphs) + 1:02d}")
        self.glyphs.append(gen)
        self.hints[net] += list(stubs)
        return gen

    def absorb(self, other: Pen, dx: int, dy: int) -> None:
        for key, pl in other.place.items():
            self.place[key] = pl.moved(dx, dy)
            (rx, ry), (vx, vy) = other.fields[key]
            self.fields[key] = ((rx + dx, ry + dy), (vx + dx, vy + dy))
        for net, segs in other.hints.items():
            self.hints[net] += [((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in segs]
        self.glyphs += [gen.moved(dx, dy) for gen in other.glyphs]
        self.labels += [lab.moved(dx, dy) for lab in other.labels]

    def boxes(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        return drawing_boxes(self.model, self._raw())

    def cells(self) -> tuple[int, int, int, int]:
        """The cell box of everything drawn so far (bodies, fields, symbols, labels, wires)."""
        boxes = [b for _, b in self.boxes()]
        boxes += [rails_mod._seg_box(*s) for segs in self.hints.values() for s in segs]
        return to_cells(union(boxes))

    def _raw(self) -> Drawing:
        glyphs = tuple(replace(gen, ref=f"#PWR{i + 1:02d}") for i, gen in enumerate(self.glyphs))
        return Drawing(block=self.block.id, recipe=self.recipe, place=dict(self.place), hints={n: tuple(s) for n, s in sorted(self.hints.items()) if s},
                       glyphs=glyphs, labels=tuple(self.labels), fields=dict(self.fields))

    def drawing(self, notes: tuple[str, ...] = (), sketch: tuple[int, int] | None = None, record: bool = True) -> Drawing:
        """The finished drawing, checked: no two boxes overlap and no hint wire meets a foreign pin or wire end (else
        LayoutFailure, and the block goes to the next recipe)."""
        from ai_eda.compilers.textbook.emit import guard

        d = replace(self._raw(), notes=notes, sketch=self.sketch() if sketch is None else sketch, rails=tuple(self.rails))
        boxes = drawing_boxes(self.model, d)
        pairs = overlaps([b for _, b in boxes])
        if pairs:
            i, j = pairs[0]
            raise LayoutFailure(f"{self.recipe}: {boxes[i][0]} overlaps {boxes[j][0]}", self.block.id)
        through = _wire_through_text(d, boxes)
        if through:
            raise LayoutFailure(f"{self.recipe}: {through}", self.block.id)
        alone = Sheet((d,), ((0, 0),))
        hard = [h for h in guard(replace(self.ctx, notes=[]), alone, Wiring(segments={n: list(s) for n, s in d.hints.items()}), list(d.labels), boxes) if h.hard]
        if hard:
            raise LayoutFailure(f"{self.recipe}: {hard[0]}", self.block.id)
        d = replace(d, box=self.cells())
        for note in notes if record else ():  # into the compile notes, once
            if f"block {self.block.id}: {note}" not in self.ctx.notes:
                self.ctx.notes.append(f"block {self.block.id}: {note}")
        return replace(d, ports=tuple(self._port(q) for q in blocks_mod.ports(self.model, self.block, d)))

    def sketch(self) -> tuple[int, int]:
        """(crossings, length) of the hint wires: perpendicular wires of two nets that cross inside both, and the cells
        of wire."""
        segs = [(n, a, b) for n, ss in sorted(self.hints.items()) for a, b in ss]
        length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for _, a, b in segs)
        cross = 0
        for i, (n, a, b) in enumerate(segs):
            if a[1] != b[1] or a[0] == b[0]:
                continue
            x0, x1 = sorted((a[0], b[0]))
            for m, c, e in segs:
                if m == n or c[0] != e[0] or c[1] == e[1]:
                    continue
                y0, y1 = sorted((c[1], e[1]))
                if x0 < c[0] < x1 and y0 < a[1] < y1:
                    cross += 1
        return cross, length

    def _port(self, port):
        """An input or output port at the height where its net's wiring reaches farthest out (leftmost for an input,
        rightmost for an output), so the block it meets lines up with the drawn wire, not with the net's first pin. Among
        the farthest points the net's trunk wins: a wire node (a bend, a T or a bus corner) before a part's pin end, so the
        neighbour lines up with the node row, never one cell off it."""
        if port.role not in ("in", "out"):
            return port
        degree: dict[Pt, int] = defaultdict(int)
        for a, b in self.hints.get(port.net, ()):
            degree[a] += 1
            degree[b] += 1
        pins = set()
        for key in self.place:
            g = self.g(key)
            pins |= {self.pin(key, n) for n, m in g.net.items() if m == port.net}
        pts = set(degree) | pins
        if not pts:
            return port
        sign = 1 if port.role == "in" else -1
        far = min(pts, key=lambda p: (sign * p[0], p in pins, -degree[p], p[1]))
        return replace(port, at=(port.at[0], far[1]))


def _wire_through_text(d: Drawing, boxes) -> str | None:
    """The first hint wire that runs through a text it does not own (a part's fields, another net's symbol text or label),
    as a reason; None when every text is clear."""
    net_of = {gen.ref: gen.net for gen in d.glyphs}
    texts = []
    for tag, box in boxes:
        kind, owner = tag.split(":", 1)
        if kind == "field":
            texts.append((tag, None, box))
        elif kind == "gfield":
            texts.append((tag, net_of.get(owner), box))
        elif kind == "label":
            texts.append((tag, owner.split("@", 1)[0], box))
    for net, segs in sorted(d.hints.items()):
        for a, b in segs:
            x0, x1 = sorted((mm(a[0]), mm(b[0])))
            y0, y1 = sorted((mm(a[1]), mm(b[1])))
            for tag, owner, (bx0, by0, bx1, by1) in texts:
                if owner == net:
                    continue
                if x0 < bx1 - 0.05 and x1 > bx0 + 0.05 and y0 < by1 - 0.05 and y1 > by0 + 0.05:
                    return f"wire of {net} {a}-{b} runs through {tag}"
    return None


# --------------------------------------------------------------------------- op-amp stage analysis


@dataclass
class Stage:
    """One amplifier unit and the parts drawn with it. ``chain`` runs left to right, ``chain_nets[i]`` / ``[i + 1]`` the
    nets left / right of ``chain[i]``; ``tap`` is the index of the potentiometer whose wiper is the other input
    (``tap_kind`` 'pot') or of the net that is (``tap_kind`` 'node')."""

    unit: str
    out: str  # pin numbers
    fpin: str
    ppin: str
    O: str  # nets
    F: str
    P: str
    fb: list[str] = field(default_factory=list)
    shunts: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    pmode: str = "signal"  # ground | rail | chain | signal
    chain: list[str] = field(default_factory=list)
    chain_nets: list[str] = field(default_factory=list)
    tap: int = -1
    tap_kind: str = ""
    buffer: list[str] = field(default_factory=list)  # followers inside the loop (bases on the output, emitters on the feedback node)
    load: list[str] = field(default_factory=list)  # parts in series from the feedback node to a connector, and the connector
    loop: str = ""  # the followers' node when feedback parts join it to the feedback input ("": the feedback input itself)
    back: list[str] = field(default_factory=list)  # those feedback parts (from the followers' node to the feedback input)

    @property
    def node(self) -> str:
        """The net the followers drive: the loop node, or the feedback input itself."""
        return self.loop or self.F

    @property
    def members(self) -> set[str]:
        return {self.unit, *self.fb, *self.shunts, *self.inputs, *self.chain, *self.buffer, *self.load, *self.back}


def stage_of(model: Model, key: str, free: frozenset[str] | set[str], taken: set[str]) -> Stage | None:
    """The op-amp stage of amplifier unit ``key`` among the glyphs ``free`` (``taken`` already belong to other stages), or
    None when no input is a feedback input."""
    g = model.glyphs[key]
    io = g.amp_io()
    if io is None or io[0].number not in g.net:
        return None
    out, upper, lower = io
    if upper.number not in g.net or lower.number not in g.net:
        return None
    O = g.net[out.number]
    avail = [k for k in sorted(free, key=_nat) if k not in taken and k != key]
    twos = [k for k in avail if _two(model.glyphs[k])]

    def fb_parts(n: str) -> list[str]:
        return [k for k in twos if set(model.glyphs[k].net.values()) == {n, O}]

    cands = [p for p in (upper, lower) if fb_parts(g.net[p.number]) or g.net[p.number] == O]
    if not cands:
        return _looped(model, key, set(avail))
    if len(cands) == 2:  # both: the inverting input ('-') is the feedback input
        cands = [p for p in cands if p.name.strip() in ("-", "~")] or cands[:1]
    fp = cands[0]
    pp = lower if fp is upper else upper
    s = Stage(key, out.number, fp.number, pp.number, O, g.net[fp.number], g.net[pp.number])
    if s.P in (s.O, s.F):
        return None
    s.fb = sorted(fb_parts(s.F), key=lambda k: (model.glyphs[k].dc_blocking, _nat(k))) if s.F != O else []  # resistors nearest
    drivers = sorted({m for m, _ in model.nets[s.F].pins if m != key and not _two(model.glyphs[m])}, key=_nat)
    if drivers:  # the feedback node is driven by another glyph: a follower buffer inside the loop, or not a stage
        if s.F == O or not _buffer(model, s, drivers, set(avail)):
            return None
        return s
    used = set(s.fb)
    for k in twos:
        if k in used or s.F == O or s.F not in model.glyphs[k].net.values():
            continue
        far = _other(model.glyphs[k], s.F)
        if _rail(model, far):
            s.shunts.append(k)
        elif far not in (s.O, s.P):
            s.inputs.append(k)
    used |= set(s.shunts) | set(s.inputs)
    role = model.nets[s.P].role
    if role is Role.GROUND:
        s.pmode = "ground"
    elif role in (Role.SUPPLY, Role.NEGATIVE):
        s.pmode = "rail"
    else:
        _chain(model, s, [k for k in avail if k not in used])
        others = [k for k, _ in model.nets[s.P].pins if k != key]
        if s.pmode == "signal" and len(others) == 1 and others[0] in avail and others[0] not in used and _two(model.glyphs[others[0]]) \
                and _rail(model, _other(model.glyphs[others[0]], s.P)):
            s.pmode, s.chain = "bias", [others[0]]
    return s


def _looped(model: Model, key: str, avail: set[str]) -> Stage | None:
    """A stage whose loop closes through followers and feedback parts from their node back to an input (see
    :func:`~ai_eda.compilers.textbook.blocks.follower_loop`), all of them in ``avail``; None otherwise."""
    loop = blocks_mod.follower_loop(model, key)
    if loop is None or not loop[1] or not set(loop[0]) | set(loop[1]) <= avail:
        return None
    followers, back = loop
    g = model.glyphs[key]
    out, upper, lower = g.amp_io()
    F = set(model.glyphs[back[0]].net.values())
    fp = next(p for p in (upper, lower) if g.net[p.number] in F and all(g.net[p.number] in model.glyphs[k].net.values() for k in back))
    pp = lower if fp is upper else upper
    s = Stage(key, out.number, fp.number, pp.number, g.net[out.number], g.net[fp.number], g.net[pp.number])
    if s.P in (s.O, s.F):
        return None
    s.loop, s.back = _other(model.glyphs[back[0]], s.F), list(back)
    if not _buffer(model, s, followers, avail):
        return None
    return s


def _buffer(model: Model, s: Stage, drivers: list[str], avail: set[str]) -> bool:
    """Followers inside the loop: each driver is a transistor with its control pin on the output and its through pair
    between the feedback node and a rail; the parts from the output to the feedback node bias them (drawn between the
    nodes); the feedback node's other parts run in series to a connector (the load)."""
    for k in drivers:
        g = model.glyphs[k]
        pair, ctrl = g.through(), g.control()
        if k not in avail or g.kind is not Kind.THRU3 or pair is None or ctrl is None or g.net.get(ctrl.number) != s.O:
            return False
        ends = {g.net.get(p.number) for p in pair}
        if s.node not in ends or not all(n is not None and (n == s.node or _rail(model, n)) for n in ends):
            return False
    s.buffer = drivers
    s.fb = [k for k in sorted(avail, key=_nat) if _two(model.glyphs[k]) and set(model.glyphs[k].net.values()) == {s.node, s.O}]
    rest = [k for k, _ in model.nets[s.node].pins if k not in (s.unit, *drivers, *s.fb, *s.back) and k in avail]
    if len(rest) == 1 and rest[0] in avail and _two(model.glyphs[rest[0]]):
        load, n = [rest[0]], _other(model.glyphs[rest[0]], s.node)
        while len(load) < MAX_CHAIN:
            info = model.nets[n]
            if info.role is not Role.SIGNAL or len(info.pins) != 2:
                break
            nxt = next(m for m, _ in info.pins if m != load[-1])
            g = model.glyphs[nxt]
            if nxt not in avail:
                break
            if g.kind is Kind.CONN and all(m == n or _rail(model, m) for m in g.net.values()):
                load.append(nxt)
                s.load = load
                break
            if not _two(g):
                break
            load.append(nxt)
            n = _other(g, n)
    s.pmode = "signal" if model.nets[s.P].role is Role.SIGNAL else "ground" if model.nets[s.P].role is Role.GROUND else "rail"
    return True


def _chain(model: Model, s: Stage, avail: list[str]) -> None:
    """The private series chain the other input taps, if there is one (see :class:`Stage`)."""
    pins_on_p = model.nets[s.P].pins
    others = [k for k, _ in pins_on_p if k != s.unit]
    if len(others) + 1 != len(pins_on_p) or any(k not in avail for k in others):
        return
    pool = set(avail)

    def private(n: str, part: str) -> str | None:
        """The next part on net ``n`` (the only other pin of a two-pin net), or None when ``n`` ends the chain."""
        info = model.nets[n]
        if n in (s.O, s.F) or info.role is not Role.SIGNAL or len(info.pins) != 2:
            return None
        nxt = next(k for k, _ in info.pins if k != part)
        return nxt if nxt in pool and _two(model.glyphs[nxt]) else None

    def walk(part: str, n: str) -> tuple[list[str], list[str]]:
        """Parts and nets from ``part`` outward over ``n`` (the net past ``part``): ([part, ...], [n, ...])."""
        parts, nets = [part], [n]
        while len(parts) < MAX_CHAIN:
            nxt = private(nets[-1], parts[-1])
            if nxt is None or nxt in parts:
                break
            parts.append(nxt)
            nets.append(_other(model.glyphs[nxt], nets[-1]))
        return parts, nets

    if len(others) == 1 and model.glyphs[others[0]].kind is Kind.THRU3:
        pot = model.glyphs[others[0]]
        pair, ctrl = pot.through(), pot.control()
        if pair is None or ctrl is None or pot.net.get(ctrl.number) != s.P or any(p.number not in pot.net for p in pair):
            return
        ends = [pot.net[p.number] for p in pair]
        if ends[0] == ends[1] or s.P in ends:
            return
        sides = []
        for n in ends:
            nxt = private(n, pot.key)
            sides.append(walk(nxt, _other(model.glyphs[nxt], n)) if nxt is not None else ([], []))
        a, b = sides
        left_parts, left_nets = a
        right_parts, right_nets = b
        if (left_nets[-1] if left_nets else ends[0]) == s.O:  # the side toward the output goes right
            (left_parts, left_nets), (right_parts, right_nets), ends = (right_parts, right_nets), (left_parts, left_nets), ends[::-1]
        s.chain = [*left_parts[::-1], pot.key, *right_parts]
        s.chain_nets = [*left_nets[::-1], ends[0], ends[1], *right_nets]
        s.tap, s.tap_kind = len(left_parts), "pot"
    elif len(others) == 2 and all(_two(model.glyphs[k]) for k in others):
        sides = [walk(k, _other(model.glyphs[k], s.P)) for k in others]
        (lp, ln), (rp, rn) = sides
        if ln[-1] == s.O:
            (lp, ln), (rp, rn) = (rp, rn), (lp, ln)
        if _rail(model, ln[-1]) and _rail(model, rn[-1]) and model.nets[rn[-1]].role is Role.SUPPLY:
            (lp, ln), (rp, rn) = (rp, rn), (lp, ln)  # a divider between two rails: the supply end left, standing up
        s.chain = [*lp[::-1], *rp]
        s.chain_nets = [*ln[::-1], s.P, *rn]
        s.tap, s.tap_kind = len(lp), "node"
    else:
        return
    s.pmode = "chain"


def stages_of(model: Model, free: frozenset[str] | set[str], summing: bool = False) -> list[Stage]:
    """The op-amp stages among ``free``, in natural order of their units; a unit whose feedback input has
    :data:`SUM_KEEP` or more input parts and that :func:`summing_bus` takes as the summer of a bus is skipped (the array
    recipe draws it) unless ``summing``."""
    out: list[Stage] = []
    taken: set[str] = set()
    for key in sorted((k for k in free if model.glyphs[k].kind is Kind.AMP), key=_nat):
        s = stage_of(model, key, free, taken)
        if s is None:
            continue
        if len(s.inputs) >= SUM_KEEP and not summing:  # left to the array recipe when it draws this unit as a summer
            bus = summing_bus(model, set(free) - taken)
            if bus is not None and bus.summer is not None and bus.summer.unit == key:
                continue
        out.append(s)
        taken |= s.members
    return out


def _power_units(model: Model, free, stages: list[Stage]) -> list[str]:
    refs = {model.glyphs[s.unit].ref for s in stages}
    return sorted((k for k in free if model.glyphs[k].kind is Kind.PWRUNIT and model.glyphs[k].ref in refs), key=_nat)


def _io_connectors(model: Model, free, stages: list[Stage]) -> list[str]:
    """Connectors whose signal nets are all inputs (input parts' far nets, other inputs left to the router) or outputs of
    the stages: drawn with them, on the input side."""
    ins = {_other(model.glyphs[k], s.F) for s in stages for k in s.inputs} | {s.P for s in stages if s.pmode == "signal"}
    outs = {s.O for s in stages}
    out = []
    for k in sorted(free, key=_nat):
        g = model.glyphs[k]
        if g.kind is not Kind.CONN:
            continue
        sig = {n for n in g.net.values() if model.nets[n].role is Role.SIGNAL}
        if sig and sig <= ins | outs and sig & ins:
            out.append(k)
    return out


# --------------------------------------------------------------------------- op-amp stage drawing


@dataclass
class Drawn:
    """Where a stage cell's nodes are (cells, cell-relative)."""

    fy: int
    py: int
    oy: int
    lx: int
    xl: int  # feedback node riser
    xr: int  # output node riser
    far: list[tuple[str, Pt]] = field(default_factory=list)  # chain far ends left to the drawing's bus
    below: bool = False  # the feedback drawn under the unit


def stage_note(model: Model, s: Stage, d: Drawn) -> str:
    """What the compile notes say about one drawn stage."""
    g = model.glyphs[s.unit]
    name = g.pin(s.fpin).name.strip() or s.fpin
    if s.buffer:
        back = f"feedback {'+'.join(s.back)} over the top from {s.loop}" if s.back else "feedback wire over the top"
        return (f"opamp_stage: {g.shown_ref} feedback input {name} ({s.F}), followers {'/'.join(s.buffer)} inside the loop, "
                f"bias {'+'.join(s.fb) or 'none'} between the nodes, {back}")
    top = name if d.fy < d.py else (g.pin(s.ppin).name.strip() or s.ppin)
    return (f"opamp_stage: {g.shown_ref} feedback input {name} ({s.F}), {top} on top, "
            f"feedback {'+'.join(s.fb) or 'wire'} {'below' if d.below else 'above'}")


def _orient_unit(g: Glyph, s: Stage) -> Orient:
    if g.power_pins():
        return Orient()
    _, upper, _ = g.amp_io()
    return Orient() if s.fpin == upper.number else Orient(0, "x")


def _unit_fields(pen: Pen, key: str, around_extra: list, right: int | None = None) -> None:
    """Reference over Value above the unit; with supply symbols on top, beside them (right of the topmost, left of the
    output riser at ``right``) when they fit there, else above them."""
    from ai_eda.gui import schematic_render as render

    g = pen.g(key)
    pl = replace(pen.place[key], field_side="T")
    body = body_box(g, pl)
    cands = [field_anchors(field_plan(g, pl, "T", around=body))]
    if around_extra:
        sym = union(around_extra)
        w = max(render._text_width(g.shown_ref, 1.27), render._text_width(g.component.value, 1.27))
        x = to_cells(sym)[2] + 1 + math.ceil(w / mm(1) / 2)
        y = to_cells(body)[1] - 1
        cands.append(((x, y - 2), (x, y)))
        cands.append(field_anchors(field_plan(g, pl, "T", around=union([body, *around_extra]))))
    for anchors in cands:
        fields = measure(g, pl, anchors=anchors).fields
        clear = not any(_hit(f, b) for f in fields for b in [*around_extra, *(() if anchors == cands[0] else (body,))])
        if clear and (right is None or all(f[2] < mm(right) - 0.5 for f in fields)):
            pen.put(key, (pl.x, pl.y), pl.orient, "T", anchors)
            return
    pen.put(key, (pl.x, pl.y), pl.orient, "T", cands[-1])


def _hit(a, b) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9


def draw_buffered(pen: Pen, s: Stage) -> Drawn:
    """A stage whose loop closes through followers: the unit at the origin, its output riser feeding the followers'
    control pins (the upper one on the supply, the lower on ground or the negative rail), their outputs meeting at the
    feedback node to the right with the bias parts between the two nodes, the feedback wire over the top back to the
    feedback input (or the feedback parts from the followers' node, on rows over the top), and the load in line to the
    right ending at its connector."""
    model = pen.model
    g = model.glyphs[s.unit]
    o = _orient_unit(g, s)
    pen.put(s.unit, (0, 0), o, "T")
    F_at, O_at = pen.pin(s.unit, s.fpin), pen.pin(s.unit, s.out)
    supply = [pen.local(s.unit, p.number) for p in g.power_pins() if p.number in g.net]
    _unit_fields(pen, s.unit, [b for gen in supply for b in (generated_boxes(gen)[0], *generated_boxes(gen)[1])])
    if s.pmode in ("ground", "rail"):
        pen.local(s.unit, s.ppin)
    xb = O_at[0] + 4
    pen.wire(s.O, O_at, (xb, O_at[1]))
    emitters = []
    for k in s.buffer:
        q = model.glyphs[k]
        pair = q.through()
        rail_pin = next(p for p in pair if q.net[p.number] != s.node)
        high = model.nets[q.net[rail_pin.number]].role is Role.SUPPLY
        oq = None
        for cand in TRANSFORMS:
            if pin_side(q, q.control().number, cand) == "L" and pin_side(q, rail_pin.number, cand) == ("T" if high else "B"):
                oq = cand
                break
        if oq is None:
            raise LayoutFailure(f"{k}: no turn puts its rail pin outward")
        c = pin_cell(q, q.control().number, (0, 0), oq)
        e = pin_cell(q, next(p for p in pair if p is not rail_pin).number, (0, 0), oq)
        ey = -6 if high else 6
        pen.put(k, (xb + 2 - c[0], O_at[1] + ey - e[1]), oq, "R")
        base = pen.pin(k, q.control().number)
        pen.wire(s.O, (xb, O_at[1]), (xb, base[1]), base)
        emitters.append(pen.on(k, s.node))
        pen.local(k, rail_pin.number)
    xs = emitters[0][0]
    if any(e[0] != xs for e in emitters):
        raise LayoutFailure("opamp_stage: the followers' outputs are not in one column")
    node = (xs, O_at[1])
    for e in emitters:
        pen.wire(s.node, e, node)
    for k in s.fb:
        pen.put_two(k, s.O, ((xb + xs) // 2, O_at[1]), False, "B")
        pen.wire(s.O, (xb, O_at[1]), pen.on(k, s.O))
        pen.wire(s.node, pen.on(k, s.node), node)
    xf = max(pen.cells()[2] + 2, xs + GAP)
    top = pen.cells()[1] - 2
    xl = F_at[0] - 2
    if s.back:  # the feedback parts on rows over the top, between a riser from the node and one from the feedback input
        y, rows = pen.cells()[1] - FB_GAP, []
        for k in s.back:
            pen.put_two(k, s.F, ((xl + xf) // 2, y), False, "T")
            rows.append(y)
            pen.wire(s.F, (xl, y), pen.on(k, s.F))
            pen.wire(s.node, pen.on(k, s.node), (xf, y))
            y = to_cells(union([b for tag, b in pen.boxes() if tag.endswith(f":{k}")]))[1] - 3
        pen.wire(s.node, node, (xf, O_at[1]), (xf, min(rows)))
        pen.wire(s.F, F_at, (xl, F_at[1]), (xl, min(rows)))
    else:
        pen.wire(s.F, node, (xf, O_at[1]), (xf, top), (xl, top), (xl, F_at[1]), F_at)
    x = xf
    net = s.node
    for k in s.load:
        q = model.glyphs[k]
        if q.kind is Kind.CONN:
            o2 = None
            for cand in (Orient(), Orient(0, "y")):
                pin = next(p for p, n in q.net.items() if n == net)
                if pin_side(q, pin, cand) == "L":
                    o2 = cand
                    break
            if o2 is None:
                raise LayoutFailure(f"{k}: no turn faces the load")
            at = pin_cell(q, pin, (0, 0), o2)
            pen.put(k, (x + GAP + 2 - at[0], O_at[1] - at[1]), o2, "R")
            pen.wire(net, (x, O_at[1]), pen.pin(k, pin))
            pen.labels.append(Label(net, (x + 3, O_at[1]), 90, "left", "name"))  # above the wire, clear of the connector's symbols
            for p, n in sorted(q.net.items()):
                if p != pin and _rail(model, n):
                    pen.local(k, p)
            break
        w = abs(pin_cell(q, sorted(q.net)[0], (0, 0), orient_two(q, net, False))[0] - pin_cell(q, sorted(q.net)[1], (0, 0), orient_two(q, net, False))[0])
        pen.put_two(k, net, (x + GAP + w // 2, O_at[1]), False, "T")
        pen.wire(net, (x, O_at[1]), pen.on(k, net))
        net = _other(q, net)
        x = pen.on(k, net)[0]
    return Drawn(F_at[1], pen.pin(s.unit, s.ppin)[1], O_at[1], F_at[0], xl, xf)


def _width(model: Model, k: str) -> int:
    """The pin span of a two-terminal part lying horizontally."""
    g = model.glyphs[k]
    o = orient_two(g, sorted(g.net.values())[0], False)
    a, b = (pin_cell(g, p, (0, 0), o) for p in g.net)
    return abs(a[0] - b[0])


def draw_stage(pen: Pen, s: Stage, bus_from: int | None = None, straight: bool = False, force: str | None = None) -> Drawn:
    """One stage with its unit at the origin (see the module text). ``bus_from``: the feedback-input row already comes in
    from the left (a summing bus) and starts at this x. ``straight``: the other input is wired in from the left (a
    connector pin level with it), so a feedback above would cross that wire and the stage's rule puts it below the unit.
    ``force`` 'above' or 'below' overrides that rule (the recipe draws both and keeps the better)."""
    if s.buffer:
        return draw_buffered(pen, s)
    model = pen.model
    g = model.glyphs[s.unit]
    o = _orient_unit(g, s)
    pen.put(s.unit, (0, 0), o, "T")
    F_at, P_at, O_at = pen.pin(s.unit, s.fpin), pen.pin(s.unit, s.ppin), pen.pin(s.unit, s.out)
    lx, ox = F_at[0], O_at[0]
    fy, py, oy = F_at[1], P_at[1], O_at[1]
    supply = [pen.local(s.unit, p.number) for p in g.power_pins() if p.number in g.net]
    own = [b for gen in supply for b in (generated_boxes(gen)[0], *generated_boxes(gen)[1])]
    _unit_fields(pen, s.unit, own, ox + 2)
    xl, xr = lx - 2, ox + 2
    if s.pmode == "ground":
        gen = pen.local(s.unit, s.ppin)
        if py < fy:  # the riser from the feedback input passes outside the stub end and its symbol
            xl = lx - 4
    elif s.pmode == "rail":
        pen.local(s.unit, s.ppin)
    elif s.pmode == "bias":  # one part from the other input to a rail, on that input's row, its symbol at the far end
        k = s.chain[0]
        rail = _other(pen.g(k), s.P)
        w = _width(model, k)
        pen.put_two(k, rail, (lx - 4 - w // 2, py), False, "T")
        pen.wire(s.P, pen.on(k, s.P), P_at)
        pen.labels.append(Label(s.P, (lx - 2, py), 90, "left", "name"))  # above its short wire, clear of the unit
        far = pen.on(k, rail)
        pen.local(k, next(q for q, n in pen.g(k).net.items() if n == rail))
    # the other input wired in from the left, or biased on its row, above the feedback input: a feedback above would
    # have to pass outside it (longer, wider); the feedback goes below the unit instead
    rule = py < fy and (straight and s.pmode == "signal" or s.pmode in ("bias", "chain"))
    below = rule if force is None else force == "below"
    if rule and not below and s.pmode == "bias":  # above: the riser passes outside the bias part and its symbol
        xl = min(xl, far[0] - 3)
    unit_boxes = [b for tag, b in pen.boxes() if tag.endswith(f":{s.unit}") or tag.startswith(("glyph:", "gfield:"))]
    top, bottom = to_cells(union(unit_boxes))[1], to_cells(union(unit_boxes))[3]
    y = bottom + FB_GAP + 1 if below else top - FB_GAP
    step = 7 if below else -7
    rows = []
    for k in s.fb:
        pen.put_two(k, s.F, ((xl + xr) // 2, y), False, "B" if below else "T")
        rows.append(y)
        a, b = pen.on(k, s.F), pen.on(k, s.O)
        pen.wire(s.F, (xl, y), a)
        pen.wire(s.O, b, (xr, y))
        mine = to_cells(union([bx for tag, bx in pen.boxes() if tag.endswith(f":{k}")]))
        y = mine[3] + 3 if below else mine[1] - 3  # the next parallel part clear of this one's fields
    if not s.fb:  # a follower: the output wired straight back
        y = bottom + 2 if below else top - 2
        rows.append(y)
        pen.wire(s.O, (xl, y), (xr, y))
        pen.wire(s.O, F_at, (xl, fy), (xl, y))
        pen.wire(s.O, (xr, y), (xr, oy), O_at)
    else:
        far_row = max(rows) if below else min(rows)
        pen.wire(s.F, F_at, (xl, fy), (xl, far_row))
        pen.wire(s.O, O_at, (xr, oy), (xr, far_row))
    d = Drawn(fy, py, oy, lx, xl, xr, below=below)
    own = to_cells(union([b for tag, b in pen.boxes() if tag.endswith(f":{s.unit}") or tag.startswith(("glyph:", "gfield:"))]))
    left = xl
    if s.pmode == "chain":  # on the other input's side of the unit: under it, or over it when that input is on top
        left = min(left, _draw_chain(pen, s, d, own[3], own[1] if py < fy and below else None))
    elif s.pmode == "bias" and not below:
        left = min(left, pen.on(s.chain[0], _other(pen.g(s.chain[0]), s.P))[0] - 2)
    # shunts from the feedback input: columns left of everything under the feedback row
    for k in s.shunts:
        x = min(left, pen.cells()[0]) - GAP
        pen.put_two(k, s.F, (x, fy + 5), True, "L")
        pen.wire(s.F, (x, fy), pen.on(k, s.F))
        pen.local(k, next(p for p, n in pen.g(k).net.items() if n != s.F))
        left = x
    if s.shunts or s.inputs or bus_from is not None:
        pen.wire(s.F, (left, fy), (xl, fy))
    # input parts: the first on the feedback-input row, the others stacked under it, all meeting one riser at x0
    x0, y, low = left, fy, None
    for i, k in enumerate(s.inputs):
        far = _other(pen.g(k), s.F)
        side = "B" if (s.pmode == "bias" and below) or i else "T"
        pen.put_two(k, far, (x0 - GAP - 3, y), False, side)
        if low is not None:  # clear of the part above and its fields
            mine = to_cells(union([b for tag, b in pen.boxes() if tag.endswith(f":{k}")]))
            y += max(0, low + 2 - mine[1])
            pen.put_two(k, far, (x0 - GAP - 3, y), False, side)
        pen.wire(s.F, pen.on(k, s.F), (x0, y))
        if i:
            pen.wire(s.F, (x0, prev), (x0, y))
        prev = y
        low = to_cells(union([b for tag, b in pen.boxes() if tag.endswith(f":{k}")]))[3]
    if bus_from is not None:
        pen.wire(s.F, (bus_from, fy), (left, fy))
    return d


def _chain_orient(g: Glyph, s: Stage, left_net: str, right_net: str, wiper_side: Side = "T") -> Orient:
    """A potentiometer turned so its wiper points to ``wiper_side`` (toward the unit) and its pin on ``right_net`` lies
    right of the one on ``left_net``."""
    by_net = {n: p for p, n in g.net.items()}
    wiper = g.control().number
    for o in TRANSFORMS:
        a, b = pin_cell(g, by_net[left_net], (0, 0), o), pin_cell(g, by_net[right_net], (0, 0), o)
        if pin_side(g, wiper, o) == wiper_side and a[1] == b[1] and a[0] < b[0]:
            return o
    raise LayoutFailure(f"{g.key}: no turn puts its wiper toward the unit")


def _draw_chain(pen: Pen, s: Stage, d: Drawn, bottom: int, top: int | None = None) -> int:
    """The chain under the unit (see the module text), or over it when ``top`` (the unit box's top) is given: the other
    input on top, its feedback below; returns the leftmost x it uses."""
    model = pen.model
    up = top is not None
    yc = top - CHAIN_DROP if up else bottom + CHAIN_DROP
    nets = s.chain_nets
    right_parts = s.chain[s.tap + (1 if s.tap_kind == "pot" else 0):]
    left_parts = s.chain[:s.tap]

    def width(k: str) -> int:
        g = model.glyphs[k]
        o = orient_two(g, sorted(g.net.values())[0], False)
        a, b = (pin_cell(g, p, (0, 0), o) for p in g.net)
        return abs(a[0] - b[0])

    need = sum(width(k) + GAP for k in right_parts) + (3 if s.tap_kind == "pot" else 2)
    t = min(d.lx - 4, d.xr - need)
    if t > d.lx - 4:
        t = d.lx - 4
    # the tap
    if s.tap_kind == "pot":
        pot = s.chain[s.tap]
        g = model.glyphs[pot]
        o = _chain_orient(g, s, nets[s.tap], nets[s.tap + 1], "B" if up else "T")
        w = pin_cell(g, g.control().number, (0, 0), o)
        e = pin_cell(g, next(p for p, n in g.net.items() if n == nets[s.tap]), (0, 0), o)
        pen.put(pot, (t - w[0], yc - e[1]), o, "T" if up else "B")
        wiper = pen.pin(pot, g.control().number)
        pen.wire(s.P, wiper, (t, d.py), (d.lx, d.py))
        right_x = pen.on(pot, nets[s.tap + 1])[0]
        left_x = pen.on(pot, nets[s.tap])[0]
        rnet_index = s.tap + 1
        lnet_index = s.tap
    else:
        pen.wire(s.P, (t, yc), (t, d.py), (d.lx, d.py))
        right_x = left_x = t
        rnet_index = lnet_index = s.tap
    # right of the tap, toward the output
    x = right_x
    for i, k in enumerate(right_parts):
        net = nets[rnet_index + i]
        w = width(k)
        pen.put_two(k, net, (x + GAP + w // 2, yc), False, "T")
        pen.wire(net, (x, yc), pen.on(k, net))
        x = pen.on(k, nets[rnet_index + i + 1])[0]
    end = nets[-1]
    if end == s.O:
        pen.wire(end, (x, yc), (d.xr, yc), (d.xr, d.oy))
    elif _rail(model, end):
        last = right_parts[-1] if right_parts else s.chain[s.tap]
        pen.local(last, next(p for p, n in pen.g(last).net.items() if n == end))
    elif not right_parts and s.tap_kind == "node":
        pass
    else:
        d.far.append((end, (x, yc)))
    # left of the tap: down from a 2-cell jog
    start = nets[0]
    if left_parts and (up or model.nets[start].role is Role.SUPPLY):  # on the chain row, the far end's symbol at the left end
        x = left_x
        for i, k in enumerate(reversed(left_parts)):
            net, outer = nets[lnet_index - i], nets[lnet_index - i - 1]
            w = width(k)
            pen.put_two(k, outer, (x - GAP - w // 2, yc), False, "T")
            pen.wire(net, pen.on(k, net), (x, yc))
            x = pen.on(k, outer)[0]
        last = left_parts[0]
        if _rail(model, start):
            pen.local(last, next(p for p, n in pen.g(last).net.items() if n == start))
        else:
            d.far.append((start, (x, yc)))
        return x - 4
    if left_parts:
        xcol = left_x - 2
        net = nets[lnet_index]
        pen.wire(net, (left_x, yc), (xcol, yc))
        y = yc
        for i, k in enumerate(reversed(left_parts)):
            net = nets[lnet_index - i]
            w = width(k)
            pen.put_two(k, net, (xcol, y + 2 + w // 2), True, "L")
            pen.wire(net, (xcol, y), pen.on(k, net))
            y = pen.on(k, nets[lnet_index - i - 1])[1]
        last = left_parts[0]
        if _rail(model, start):
            pen.local(last, next(p for p, n in pen.g(last).net.items() if n == start))
        else:
            d.far.append((start, (xcol, y)))
        return min(xcol, left_x) - 2
    if _rail(model, start):
        pot = s.chain[s.tap]
        pen.local(pot, next(p for p, n in pen.g(pot).net.items() if n == start))
    else:
        d.far.append((start, (left_x, yc)))
    return left_x


def _buses(pen: Pen, far: list[tuple[str, Pt]]) -> None:
    """The chains' far ends on one net joined by a bus line through the lowest of them (a tee at each of those pin ends,
    so the line itself is the net's trunk where other blocks join it; higher ends drop to it). A body or text the line
    would pass through (a combination's second part under a chain) sends it below that, the ends dropping to it."""
    by_net: dict[str, list[Pt]] = defaultdict(list)
    for net, p in far:
        by_net[net].append(p)
    for net, pts in sorted(by_net.items()):
        if len(pts) < 2:
            continue
        y = max(p[1] for p in pts)
        lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
        boxes = [to_cells(b) for tag, b in pen.boxes() if tag.startswith(("body:", "field:", "glyph:", "gfield:"))]
        moved = True
        while moved:
            moved = False
            for x0, y0, x1, y1 in boxes:
                if x0 < hi and x1 > lo and y0 < y < y1:
                    y, moved = y1 + 1, True
        for p in pts:
            pen.wire(net, p, (p[0], y))
        xs = sorted(p[0] for p in pts)
        x0 = xs[0]
        if _leaves(pen.model, pen.block, net):  # coming in from another block: the line starts at the drawing's left edge,
            x0 = min(x0, pen.cells()[0])  # clear of the first column's fields, so the wire from outside meets it level
        for a, b in zip([x0, *xs], xs):  # one piece per span, so an unused lead-in is trimmed alone
            pen.wire(net, (a, y), (b, y))


def _power_unit(pen: Pen, key: str, x: int) -> None:
    g = pen.g(key)
    pen.put(key, (0, 0), Orient(), "R")
    box = to_cells(body_box(g, pen.place[key]))
    pen.put(key, (x - box[0], 0), Orient(), "R")
    for p in g.power_pins():
        if p.number in g.net:
            pen.local(key, p.number)


def _power_unit_under(pen: Pen, key: str, unit: str) -> None:
    """The power unit centred under amplifier unit ``unit``, 2 cells under everything drawn so far, so the amplifier's
    output can run on to the right."""
    sub = Pen(pen.ctx, pen.block, pen.recipe)
    _power_unit(sub, key, 0)
    box, ub = sub.cells(), to_cells(body_box(pen.g(unit), pen.place[unit]))
    pen.absorb(sub, (ub[0] + ub[2]) // 2 - (box[0] + box[2]) // 2, pen.cells()[3] + 2 - box[1])


def _connector(pen: Pen, key: str, stages: list[Stage], drawn: dict[str, Drawn]) -> None:
    """A connector on the input side: facing right, the first pin whose net is an input part's far net level with that
    part's row and wired straight to it; output pins and other signal pins named by label pairs; rail pins by their
    symbol (a label when the symbol would stand in front of the next pin)."""
    model = pen.model
    g = pen.g(key)
    sides = [pin_side(g, p.number, Orient()) for p in g.pins]
    o = Orient() if sides.count("R") >= sides.count("L") else Orient(0, "y")
    feeds = {}
    for s in stages:
        for k in s.inputs:
            feeds[_other(model.glyphs[k], s.F)] = (k, s)
        if s.pmode == "signal":
            feeds.setdefault(s.P, (s.unit, s))
    numbers = sorted(g.net, key=_nat)
    anchor = next((p for p in numbers if g.net[p] in feeds), None)
    for p in numbers:  # the other input parts fed from this connector: their pair labels at the parts' far ends
        net = g.net[p]
        if p != anchor and net in feeds and feeds[net][0] != feeds[net][1].unit and net != g.net.get(anchor):
            k = feeds[net][0]
            pen.labels.append(stub_label(net, pen.on(k, net), pen.side(k, next(q for q, m in pen.g(k).net.items() if m == net)), "pair"))
    left = pen.cells()[0]
    pen.put(key, (0, 0), o, "L")
    box = to_cells(body_box(g, pen.place[key]))
    if anchor is not None:
        k, s = feeds[g.net[anchor]]
        target = pen.pin(k, s.ppin) if k == s.unit else pen.on(k, g.net[anchor])
        at = pin_cell(g, anchor, (0, 0), o)
        x = min(left - 8 - box[2], target[0] - GAP - 2 - at[0])
        pen.put(key, (x, target[1] - at[1]), o, "L")
        pen.wire(g.net[anchor], pen.pin(key, anchor), target)
        at = pen.pin(key, anchor)
        side = pen.side(key, anchor)
        near = {(pen.pin(key, q)[1] - at[1]) for q in numbers if q != anchor and pin_side(g, q, o) == side and abs(pen.pin(key, q)[1] - at[1]) <= 2}
        if not any(dy < 0 for dy in near):  # above its wire, clear of the pins below
            pen.labels.append(Label(g.net[anchor], (at[0] + 3, at[1]), 90, "left", "name"))
        elif not any(dy > 0 for dy in near):  # a pin right above: hanging under the wire instead
            pen.labels.append(Label(g.net[anchor], (at[0] + 3, at[1]), 270, "right", "name"))
        # pins on both sides: the net is named on its wire by the sheet's labels
    else:
        pen.put(key, (left - 8 - box[2], 0), o, "L")
    outs = {s.O: s for s in stages}
    late = []
    for p in numbers:
        net = g.net[p]
        if p == anchor or (g.ref, p) in model.stacked:
            continue
        at, side = pen.pin(key, p), pen.side(key, p)
        if model.nets[net].role is Role.SIGNAL:
            pen.labels.append(stub_label(net, at, side, "pair"))
            s = outs.get(net)
            if s is not None:
                dn = drawn[s.unit]
                pen.labels.append(stub_label(net, (dn.xr, dn.oy), "R", "pair", length=4))
            continue
        toward = 1 if model.nets[net].role is not Role.SUPPLY else -1
        crowded = any(pin_side(g, q, o) == side and 0 < (pen.pin(key, q)[1] - at[1]) * toward <= 2 for q in numbers if q != p)
        if not crowded:
            pen.local(key, p)
        elif toward == 1 and side == "R" and model.nets[net].role is Role.GROUND:  # a ground symbol (no text) on a longer stub
            late.append((p, net, at))
        else:
            pen.labels.append(stub_label(net, at, side, "pin"))
    for p, net, at in late:  # hanging past the stubs, symbols and labels of the pins below it
        reach = at[0] + 4
        for tag, box in pen.boxes():
            kind = tag.split(":", 1)[0]
            if kind in ("label", "glyph", "gfield") and mm(at[1]) - 0.5 < box[3] and box[1] < mm(at[1] + 5):
                if mm(at[0]) - 0.5 <= box[0] < mm(at[0] + 4):  # the connector's own pin stubs only
                    reach = max(reach, to_cells(box)[2] + 2)
        gen = pen.local(key, p)
        pen.glyphs[-1] = gen.moved(reach - at[0] - rails_mod.STUB, 0)
        pen.hints[net][-1] = (at, (reach, at[1]))


class OpampStageRecipe:
    """OPAMP_STAGE (see the module text)."""

    name = "opamp_stage"

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        stages = stages_of(model, free)
        if not stages:
            return None
        members = {k for s in stages for k in s.members}
        members |= set(_power_units(model, free, stages)) | set(_io_connectors(model, free, stages))
        return frozenset(members)

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        model = ctx.model
        stages = stages_of(model, members)
        if not stages or {k for s in stages for k in s.members} - members:
            raise LayoutFailure("opamp_stage: the stages changed", block.id)
        from ai_eda.compilers.textbook.pipeline import rank

        conns = _io_connectors(model, members, stages)
        fed = {n for k in conns for n in model.glyphs[k].net.values()}
        straight = {s.P for s in stages if s.pmode == "signal" and s.P in fed and not s.inputs} if len(stages) == 1 else set()
        tried = []  # feedback above first (it wins a tie), then each stage's own rule (feedback below where it applies)
        error = None
        for force in ("above", None):
            pen = Pen(ctx, block, self.name)
            try:
                drawn = place_stages(pen, stages, _power_units(model, members, stages), straight, force)
                for k in conns:
                    _connector(pen, k, stages, drawn)
                left = {k for k in members if k not in pen.place}
                if left:
                    raise LayoutFailure(f"opamp_stage: {', '.join(sorted(left, key=_nat))} not drawn", block.id)
                notes = tuple(stage_note(model, s, drawn[s.unit]) for s in stages)
                spare = [model.glyphs[s.unit].shown_ref for s in stages if _spare(model, s)]
                if spare and len(spare) < len(stages):
                    notes += (f"opamp_stage: spare unit(s) {', '.join(spare)} in a strip at the end, drawn as the IR ties them",)
                d = pen.drawing(notes, record=False)
            except LayoutFailure as exc:
                error = error or exc
                continue
            if not any(t[1].hints == d.hints and t[1].place == d.place for t in tried):
                tried.append((rank(ctx, d), d, notes))
        if not tried:
            raise error
        (key, d, notes), *rest = sorted(tried, key=lambda t: t[0])
        if rest:
            other = rest[0][0]
            notes += (f"opamp_stage: feedback {'below' if any(n.endswith('below') for n in notes) else 'above'} kept over the "
                      f"other variant (sketch crossings {key[2]} vs {other[2]}, wire {key[3]} vs {other[3]} cells)",)
        for note in notes:
            if f"block {block.id}: {note}" not in ctx.notes:
                ctx.notes.append(f"block {block.id}: {note}")
        return [replace(d, notes=notes)]


def _leaves(model: Model, block: BlockInfo, net: str) -> bool:
    inside = set(block.members)
    return any(k not in inside for k, _ in model.nets[net].pins)


def place_stages(pen: Pen, stages: list[Stage], power: list[str], straight: set[str] = frozenset(), force: str | None = None) -> dict[str, Drawn]:
    """Every stage in a row, units at one height, natural order; a package's power unit right of its last working stage;
    spare units (an output no other part uses) in a strip at the end, past the power units; the chains' far ends joined by
    buses. A stage whose output is named where it leaves keeps room for the name before the next stage."""
    from ai_eda.gui import schematic_render as render

    model = pen.model
    drawn: dict[str, Drawn] = {}
    far: list[tuple[str, Pt]] = []
    x = 0
    spare = [s for s in stages if _spare(model, s)]
    work = [s for s in stages if s not in spare] or spare
    spare = [s for s in spare if s not in work]
    last_of = {}
    for s in work:
        last_of[model.glyphs[s.unit].ref] = s.unit
    for s in spare:
        last_of.setdefault(model.glyphs[s.unit].ref, s.unit)
    for s in [*work, *spare]:
        cell = Pen(pen.ctx, pen.block, pen.recipe)
        d = draw_stage(cell, s, straight=s.P in straight, force=force)
        if len(stages) >= SUM_BRANCHES and _leaves(model, pen.block, s.O):  # a bank's outputs: named where they leave
            cell.labels.append(stub_label(s.O, (d.xr, d.oy), "R", "pair"))
        box = cell.cells()
        dx = x - box[0] if pen.place else 0
        pen.absorb(cell, dx, 0)
        d = Drawn(d.fy, d.py, d.oy, d.lx + dx, d.xl + dx, d.xr + dx, [(n, (p[0] + dx, p[1])) for n, p in d.far], d.below)
        drawn[s.unit] = d
        far += d.far
        x = box[2] + dx + STAGE_GAP
        if any(k not in s.members for k, _ in model.nets[s.O].pins):  # room for the output's name before the next stage
            x = max(x, d.xr + 6 + math.ceil(render._text_width(s.O, 1.27) / mm(1)) + 2)
        ref = model.glyphs[s.unit].ref
        if last_of[ref] == s.unit:
            for k in power:
                if model.glyphs[k].ref == ref:
                    _power_unit(pen, k, x)
                    x = pen.cells()[2] + STAGE_GAP
    _buses(pen, far)
    return drawn


def _spare(model: Model, s: Stage) -> bool:
    """A unit no other part listens to: its output net holds only its own pins and its feedback parts."""
    return not s.inputs and not s.chain and not s.buffer and all(k in (s.unit, *s.fb) for k, _ in model.nets[s.O].pins)


OPAMP_STAGE = OpampStageRecipe()


# --------------------------------------------------------------------------- array: summing bus and chain ladder


@dataclass
class Bus:
    net: str
    branches: list[list[str]]  # each: parts from the input end to the bus
    summer: Stage | None = None


def summing_bus(model: Model, free) -> Bus | None:
    """The first signal net (by name) that :data:`SUM_BRANCHES` or more branches end on (see the module text)."""
    pool = {k for k in free if _two(model.glyphs[k])}
    amps = sorted((k for k in free if model.glyphs[k].kind is Kind.AMP), key=_nat)
    for net in sorted(model.nets):
        if model.nets[net].role is not Role.SIGNAL:
            continue
        # the output of an amplifier with an input on this net: a part to it is that amplifier's feedback, not a branch
        fed_back = set()
        for key in amps:
            g = model.glyphs[key]
            io = g.amp_io()
            if io is not None and io[0].number in g.net and net in (g.net.get(io[1].number), g.net.get(io[2].number)):
                fed_back.add(g.net[io[0].number])
        branches = []
        for k in sorted((k for k, _ in model.nets[net].pins if k in pool and _other(model.glyphs[k], net) not in fed_back), key=_nat):
            parts, n = [k], _other(model.glyphs[k], net)
            while len(parts) < 3:
                info = model.nets[n]
                if info.role is not Role.SIGNAL or len(info.pins) != 2:
                    break
                nxt = next(m for m, _ in info.pins if m != parts[-1])
                if nxt not in pool or nxt in parts:
                    break
                parts.append(nxt)
                n = _other(model.glyphs[nxt], n)
            if model.nets[n].role is Role.SIGNAL and n != net:
                branches.append(parts[::-1])
        by_kind: dict[tuple[str, ...], list[list[str]]] = defaultdict(list)
        for b in branches:  # repeated cells: the same library symbols in the same order
            by_kind[tuple(model.glyphs[k].symbol.lib_id for k in b)].append(b)
        kind = max(by_kind, key=lambda s: (len(by_kind[s]), len(s), s), default=())
        branches = sorted(by_kind.get(kind, []), key=lambda b: _nat(b[0]))
        if len(branches) < SUM_BRANCHES:
            continue
        # each branch's input end is its own: a part joining two of them makes a chain ladder, not a summing bus
        ends = {_branch_in(model, b, net) for b in branches}
        inside = {k for b in branches for k in b}
        if any(len(set(model.glyphs[k].net.values()) & ends) == 2 for k in pool - inside):
            continue
        bus = Bus(net, branches)
        taken = {k for b in branches for k in b}
        for key in sorted((k for k in free if model.glyphs[k].kind is Kind.AMP), key=_nat):
            s = stage_of(model, key, set(free) - taken, set())
            if s is not None and s.F == net and set(s.inputs) <= taken:
                s.inputs = []
                bus.summer = s
                break
        return bus
    return None


def _branch_in(model: Model, branch: list[str], net: str) -> str:
    """The input-end net of a summing branch (parts from the input end to the bus ``net``)."""
    n = net
    for k in reversed(branch):
        n = _other(model.glyphs[k], n)
    return n


def _bus_members(model: Model, free, bus: Bus) -> set[str]:
    members = {k for b in bus.branches for k in b}
    if bus.summer is not None:
        members |= bus.summer.members | set(_power_units(model, free, [bus.summer]))
    return members | set(_bus_connectors(model, free, bus, members))


def _bus_connectors(model: Model, free, bus: Bus, members: set[str]) -> list[str]:
    """Connectors whose signal nets are all branch inputs or the summer's output (at least one branch input): drawn with
    the bus, on its input side."""
    ins = {_branch_in(model, b, bus.net) for b in bus.branches}
    outs = {bus.summer.O} if bus.summer is not None else set()
    out = []
    for k in sorted(free, key=_nat):
        g = model.glyphs[k]
        if k in members or g.kind is not Kind.CONN:
            continue
        sig = {n for n in g.net.values() if model.nets[n].role is Role.SIGNAL}
        if sig and sig <= ins | outs and sig & ins:
            out.append(k)
    return out


def ladder(model: Model, free) -> dict | None:
    """A chain ladder among ``free``: {'common', 'cells' (in chain order), 'taps', 'links', 'end'} or None."""
    pool = sorted((k for k in free if _two(model.glyphs[k])), key=_nat)
    for common in sorted(model.nets):
        cells = [k for k in pool if common in model.glyphs[k].net.values()]
        by_lib: dict[str, list[str]] = defaultdict(list)
        for k in cells:
            by_lib[model.glyphs[k].symbol.lib_id].append(k)
        for lib, ks in sorted(by_lib.items()):
            if len(ks) < 3:
                continue
            tap_of = {k: _other(model.glyphs[k], common) for k in ks}
            taps = set(tap_of.values())
            if len(taps) != len(ks) or any(model.nets[t].role is not Role.SIGNAL for t in taps):
                continue
            links = [k for k in pool if k not in ks and set(model.glyphs[k].net.values()) <= taps]
            if len(links) != len(ks) - 1 or len({model.glyphs[k].symbol.lib_id for k in links}) != 1:
                continue
            adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for k in links:
                a, b = _nets(model.glyphs[k])
                adj[a].append((k, b))
                adj[b].append((k, a))
            ends = [t for t in taps if len(adj[t]) == 1]
            if len(ends) != 2 or any(len(adj[t]) > 2 for t in taps):
                continue
            cell_of = {t: k for k, t in tap_of.items()}
            start = min(ends, key=lambda t: _nat(cell_of[t]))
            order, seen, t = [start], set(), start
            while True:
                nxt = [(k, b) for k, b in adj[t] if k not in seen]
                if not nxt:
                    break
                k, t = nxt[0]
                seen.add(k)
                order.append(t)
            if len(order) != len(taps):
                continue
            end = [k for k in pool if k not in ks and k not in links and set(model.glyphs[k].net.values()) & {order[0], order[-1]}
                   and not set(model.glyphs[k].net.values()) <= taps]
            link_of = {frozenset(model.glyphs[k].net.values()): k for k in links}
            return {"common": common, "cells": [cell_of[t] for t in order], "taps": order,
                    "links": [link_of[frozenset((a, b))] for a, b in zip(order, order[1:])], "end": end[:1]}
    return None


def series_ladder(model: Model, block: BlockInfo, free) -> dict | None:
    """A series ladder among ``free``: a path of two-terminal parts from a block input to a block output (a potentiometer
    may start it: entered at one end, left at its wiper, its other end a tail to a rail), every other part a shunt from a
    node of the path - to a rail, up to a net that leaves the group, or two parts in series down to a rail. None otherwise.
    {'path': [(part, net in, net out)], 'shunts': {node: [[parts]]}, 'tail': [parts], 'pot': key | None}"""
    keys = set(free)
    if not keys or any(model.glyphs[k].kind not in (Kind.TWO, Kind.THRU3) for k in keys):
        return None
    pots = [k for k in keys if model.glyphs[k].kind is Kind.THRU3]
    if len(pots) > 1 or any({p.name for p in model.glyphs[k].pins} & {"B", "C", "E", "G", "D", "S"} for k in pots):
        return None

    def outside(n: str) -> bool:
        return any(k not in keys for k, _ in model.nets[n].pins)

    def on(n: str, used: set[str]) -> list[str]:
        return sorted({k for k, _ in model.nets[n].pins if k in keys and k not in used}, key=_nat)

    starts = [n for n in block.inputs if model.nets.get(n) is not None and model.nets[n].role is Role.SIGNAL and len(on(n, set())) == 1]
    for start in starts:
        used: set[str] = set()
        path, shunts, tail, pot = [], {}, [], None
        n = start
        first = on(n, used)[0]
        if first in pots:
            g = model.glyphs[first]
            pair, ctrl = g.through(), g.control()
            ends = {g.net.get(q.number) for q in pair} if pair else set()
            if ctrl is None or ctrl.number not in g.net or n not in ends or None in ends or len(ends) != 2:
                continue  # a rheostat (one end open) or a shorted track is no volume pot
            pot = first
            used.add(first)
            other = next(e for e in ends if e != n)
            if not _rail(model, other):
                t_ = [k for k in on(other, used)]
                if len(t_) != 1 or not _two(model.glyphs[t_[0]]) or not _rail(model, _other(model.glyphs[t_[0]], other)) or len(model.nets[other].pins) != 2:
                    continue
                tail = t_
                used |= set(t_)
            path.append((first, n, g.net[ctrl.number]))
            n = g.net[ctrl.number]
        ok = True
        while ok:
            here = on(n, used)
            node_shunts, cont, leaving = [], [], []
            for k in here:
                g = model.glyphs[k]
                if not _two(g):
                    ok = False
                    break
                m = _other(g, n)
                if _rail(model, m):
                    node_shunts.append([k])
                elif outside(m) and not [x for x in on(m, used) if x != k]:
                    leaving.append([k])  # up to a net that leaves, unless the path ends there
                elif len(model.nets[m].pins) == 2 and (nxt := next((x for x, _ in model.nets[m].pins if x != k), None)) in keys \
                        and _two(model.glyphs[nxt]) and _rail(model, _other(model.glyphs[nxt], m)) and nxt not in used:
                    node_shunts.append([k, nxt])
                else:
                    cont.append(k)
            if not cont and len(leaving) == 1 and not node_shunts:
                cont = leaving.pop()
            node_shunts += leaving
            if not ok or len(cont) > 1:
                ok = False
                break
            if node_shunts:
                shunts[n] = node_shunts
                used |= {k for s in node_shunts for k in s}
            if not cont:
                break
            k = cont[0]
            used.add(k)
            m = _other(model.glyphs[k], n)
            path.append((k, n, m))
            n = m
        if ok and used == keys and len(path) >= 2:
            return {"path": path, "shunts": shunts, "tail": tail, "pot": pot}
    return None


def draw_ladder(ctx, block: BlockInfo, plan: dict) -> Drawing:
    """The series ladder on one row left to right: a starting potentiometer standing with its wiper on the row and its
    tail below it, the series parts on the row with GAP-cell wires between, each node's shunts hanging to their rail
    (or standing up to the net that leaves) from the node."""
    model = ctx.model
    pen = Pen(ctx, block, "array")
    y, x = 0, 0
    for k, n_in, n_out in plan["path"]:
        g = model.glyphs[k]
        if k == plan["pot"]:
            ctrl = g.control()
            o = next((o for o in TRANSFORMS if pin_side(g, ctrl.number, o) == "R"
                      and pin_cell(g, next(q.number for q in g.through() if g.net.get(q.number) == n_in), (0, 0), o)[1] < 0), None)
            if o is None:
                raise LayoutFailure(f"array: {k} cannot stand with its wiper right")
            w = pin_cell(g, ctrl.number, (0, 0), o)
            pen.put(k, (x - w[0], y - w[1]), o, "L")
            low = next(q.number for q in g.through() if g.net.get(q.number) != n_in)
            if plan["tail"]:
                c = plan["tail"][0]
                tn = g.net[low]
                bot = pen.pin(k, low)
                h = _width(model, c)
                pen.put_two(c, tn, (bot[0], bot[1] + 4 + h // 2), True, "L")
                pen.wire(tn, bot, pen.on(c, tn))
                pen.labels.append(Label(tn, (bot[0], bot[1] + 2), 180, "right", "name"))  # left of its short wire
                pen.local(c, next(q for q, m in model.glyphs[c].net.items() if m != tn))
            else:
                pen.local(k, low)
            tabs = [pen.local(k, q.number) for q in g.pins if mounting(q) and q.number in g.net and _rail(model, g.net[q.number])]
            if tabs:  # a mounting pin on a rail has its own symbol; the fields stand beside it too
                pl = pen.place[k]
                around = union([body_box(g, pl), *(b for gen in tabs for b in (generated_boxes(gen)[0], *generated_boxes(gen)[1]))])
                pen.put(k, (pl.x, pl.y), pl.orient, "L", field_anchors(measure(g, pl, around=around).field_nodes))
            x = pen.pin(k, ctrl.number)[0]
            continue
        w = _width(model, k)
        pen.put_two(k, n_in, (x + GAP + w // 2, y), False, "T")
        pen.wire(n_in, (x, y), pen.on(k, n_in))
        x = pen.on(k, n_out)[0]
        shunts = plan["shunts"].get(n_out, [])
        if shunts or n_out != plan["path"][-1][2]:
            node = x + GAP // 2
            pen.wire(n_out, (x, y), (node, y))
            x = node
        for i, col in enumerate(shunts):
            first = model.glyphs[col[0]]
            far = _other(first, n_out)
            if i:  # the next shunt of the node further right, the node wire running on to it
                nx = pen.cells()[2] + 3
                pen.wire(n_out, (x, y), (nx, y))
                x = nx
            up = not _rail(model, far) or model.nets[far].role is Role.SUPPLY
            h = _width(model, col[0])
            if up:
                pen.put_two(col[0], far, (x, y - 2 - h // 2), True, "R")
                pen.wire(n_out, (x, y), pen.on(col[0], n_out))
                if _rail(model, far):
                    pen.local(col[0], next(q for q, m in first.net.items() if m == far))
                continue
            pen.put_two(col[0], n_out, (x, y + 2 + h // 2), True, "R")
            pen.wire(n_out, (x, y), pen.on(col[0], n_out))
            low, last = far, col[0]
            if len(col) == 2:
                c = col[1]
                pin = pen.on(col[0], far)
                h2 = _width(model, c)
                pen.put_two(c, far, (x, pin[1] + 4 + h2 // 2), True, "R")
                pen.wire(far, pin, pen.on(c, far))
                low, last = _other(model.glyphs[c], far), c
            pen.local(last, next(q for q, m in model.glyphs[last].net.items() if m == low))
        if shunts and n_out != plan["path"][-1][2]:  # the next series part clear of this node's shunts and their fields
            nx = max(x, pen.cells()[2] + 2 - GAP)
            pen.wire(n_out, (x, y), (nx, y))
            x = nx
    names = [k for k, _, _ in plan["path"]]
    return pen.drawing((f"array: series ladder {' - '.join(names)} with {sum(len(v) for v in plan['shunts'].values())} shunt(s)",))


class ArrayRecipe:
    """ARRAY (see the module text)."""

    name = "array"

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        bus = summing_bus(model, free)
        if bus is not None:
            return frozenset(_bus_members(model, free, bus))
        from ai_eda.compilers.textbook.columns import COLUMNS

        group = COLUMNS.claim(model, block, free)
        lad = ladder(model, free)
        if lad is not None and group is not None and {*lad["cells"], *lad["links"], *lad["end"]} <= group:
            return group
        if group is not None and series_ladder(model, block, group) is not None:
            return group
        return None

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        model = ctx.model
        bus = summing_bus(model, members)
        if bus is not None and _bus_members(model, members, bus) == set(members):
            return [draw_bus(ctx, block, bus, members)]
        lad = ladder(model, members)
        if lad is None:
            plan = series_ladder(model, block, members)
            if plan is None:
                raise LayoutFailure("array: no repeated cells", block.id)
            return [draw_ladder(ctx, block, plan)]
        from ai_eda.compilers.textbook.columns import COLUMNS

        out = [replace(d, recipe=self.name) for d in COLUMNS.draw(ctx, block, members)]
        refs = [model.glyphs[k].ref for k in lad["cells"]]
        note = f"array: chain ladder {lad['cells'][0]}..{lad['cells'][-1]} on {lad['common']}, links {', '.join(lad['links'])}" + \
            (f", end part {lad['end'][0]}" if lad["end"] else "")
        if refs != sorted(refs, key=_nat):
            note += "; the chain order differs from the reference order (chain order kept)"
        ctx.notes.append(f"block {block.id}: {note}")
        return out


def draw_bus(ctx, block: BlockInfo, bus: Bus, members: frozenset[str]) -> Drawing:
    """The branches as columns over the bus line, the summer (if any) at its right end."""
    model = ctx.model
    pen = Pen(ctx, block, "array")
    fy = 0
    lx = 0
    if bus.summer is not None:
        cell = Pen(ctx, block, "array")
        d = draw_stage(cell, bus.summer, bus_from=None)
        x = cell.cells()[2] + STAGE_GAP
        power = [k for k in _power_units(model, members, [bus.summer])]
        for k in power:
            if _leaves(model, block, bus.summer.O):  # the output runs on to the right: the power unit stands under the unit
                _power_unit_under(cell, k, bus.summer.unit)
            else:
                _power_unit(cell, k, x)
        pen.absorb(cell, 0, 0)
        fy, lx = d.fy, d.xl
    right = (min(pen.cells()[0], lx) - GAP - 2) if pen.place else 0
    conns = [k for k in sorted(members, key=_nat) if model.glyphs[k].kind is Kind.CONN and k not in {x for b in bus.branches for x in b}
             and (bus.summer is None or k not in bus.summer.members)]
    fed = {n for k in conns for n in model.glyphs[k].net.values()}
    columns = []
    for branch in bus.branches:
        col = Pen(ctx, block, "array")
        y = fy - BUS_DROP
        nets = [bus.net]
        for k in reversed(branch):
            top = _other(model.glyphs[k], nets[-1])
            g = model.glyphs[k]
            o = orient_two(g, top, True)
            a, b = (pin_cell(g, p, (0, 0), o) for p in g.net)
            h = abs(a[1] - b[1])
            col.put_two(k, top, (0, y - h // 2), True, "L" if k == branch[0] else "R")
            col.wire(nets[-1], col.on(k, nets[-1]), (0, y))
            y = col.on(k, top)[1] - (6 if k != branch[0] else 0)
            if k != branch[0]:
                col.wire(top, col.on(k, top), (0, y))
            nets.append(top)
        col.wire(bus.net, (0, fy - BUS_DROP), (0, fy))
        first = branch[0]
        if _leaves(model, block, nets[-1]) or nets[-1] in fed:  # many inputs from elsewhere: each named at its branch, a pair
            col.labels.append(stub_label(nets[-1], col.on(first, nets[-1]), col.side(first, next(p for p, n in col.g(first).net.items() if n == nets[-1])), "pair"))
        columns.append(col)
    width = max(c.cells()[2] - c.cells()[0] for c in columns) + 2
    x = right
    xs = []
    for col in reversed(columns):
        box = col.cells()
        dx = x - box[2]
        pen.absorb(col, dx, 0)
        xs.append(dx)
        x -= width
    xs = sorted(xs)
    end = lx if bus.summer is not None else xs[-1]
    pen.wire(bus.net, (xs[0], fy), (end, fy))
    for k in conns:  # on the input side, every signal pin named by a label pair, rails by their symbols
        _connector(pen, k, [bus.summer] if bus.summer is not None else [], {bus.summer.unit: d} if bus.summer is not None else {})
    note = f"array: summing bus {bus.net}, {len(bus.branches)} branches {bus.branches[0][0]}..{bus.branches[-1][0]}" + \
        (f" into {model.glyphs[bus.summer.unit].shown_ref}" if bus.summer is not None else "")
    return pen.drawing((note,))


ARRAY = ArrayRecipe()


# --------------------------------------------------------------------------- cross-coupled pair and supply chain


def cross_coupled(model: Model, free) -> dict | None:
    """The skeleton of a cross-coupled pair among ``free``: {'q': (a, b), 'coupling': (top(a)-control(b), top(b)-control(a)),
    'loads', 'biases'} or None."""
    def transistor(g: Glyph) -> bool:
        return g.kind is Kind.THRU3 and {p.name for p in g.pins} in ({"B", "C", "E"}, {"G", "D", "S"})

    qs = sorted((k for k in free if transistor(model.glyphs[k])), key=_nat)
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            ga, gb = model.glyphs[a], model.glyphs[b]
            if ga.symbol.lib_id != gb.symbol.lib_id:
                continue

            def top(g: Glyph) -> str | None:
                pin = next((p for p in g.pins if p.name in ("C", "D")), None)
                return g.net.get(pin.number) if pin else None

            def ctrl(g: Glyph) -> str | None:
                c = g.control()
                return g.net.get(c.number) if c else None

            nets = (top(ga), ctrl(gb), top(gb), ctrl(ga))
            if None in nets or len(set(nets)) != 4:
                continue
            cpl = []
            for x, y in ((nets[0], nets[1]), (nets[2], nets[3])):
                found = [k for k in free if _two(model.glyphs[k]) and set(model.glyphs[k].net.values()) == {x, y}]
                if len(found) != 1:
                    break
                cpl.append(found[0])
            if len(cpl) != 2 or model.glyphs[cpl[0]].symbol.lib_id != model.glyphs[cpl[1]].symbol.lib_id:
                continue

            def slot(net: str) -> list[str]:
                return sorted((k for k in free if _two(model.glyphs[k]) and net in model.glyphs[k].net.values()
                               and _rail(model, _other(model.glyphs[k], net))), key=_nat)

            return {"q": (a, b), "coupling": tuple(cpl), "loads": (slot(nets[0]), slot(nets[2])), "biases": (slot(nets[3]), slot(nets[1]))}
    return None


class DelegatedRecipe:
    """A motif whose geometry the columns engine draws (its mirror, rail rows or chain): the motif claims the columns
    group that holds it and names the drawing."""

    def __init__(self, name: str, detect) -> None:
        self.name = name
        self.detect = detect

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        from ai_eda.compilers.textbook.columns import COLUMNS

        group = COLUMNS.claim(model, block, free)
        if group is None or self.detect(model, group) is None:
            return None
        return group

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        from ai_eda.compilers.textbook.columns import COLUMNS

        found = self.detect(ctx.model, members)
        if found is None:
            raise LayoutFailure(f"{self.name}: not found", block.id)
        if self.name == "cross_coupled":
            try:
                return [draw_cross(ctx, block, members, found)]
            except LayoutFailure as exc:
                ctx.notes.append(f"block {block.id}: cross_coupled geometry not drawn ({exc.reason}); the columns mirror instead")
        out = [replace(d, recipe=self.name) for d in COLUMNS.draw(ctx, block, members)]
        if self.name == "cross_coupled":
            out = [diagonal_x(ctx, d, found) for d in out]
            m = ctx.model
            a, b = found["q"]
            same = all(m.glyphs[x].component.value == m.glyphs[y].component.value
                       for x, y in [(a, b), found["coupling"], *zip(*found["loads"])])
            ctx.notes.append(f"block {block.id}: cross_coupled {a}|{b}, coupling {found['coupling'][0]}|{found['coupling'][1]}"
                             f"{'; verified swap' if same else '; mirrored by symbol (values differ)'}")
        return out


def draw_cross(ctx, block: BlockInfo, members, found: dict) -> Drawing:
    """The cross-coupled pair in four columns at one pitch (left load, left inner node, right inner node, right load): the
    loads and biases standing from the rail line, the couplings on the node row between the outer and inner columns, the
    transistors under the outer columns facing each other, and the X: each inner node runs as a 45-degree diagonal
    (dx = dy = pitch) down toward the opposite transistor and into its control pin horizontally. The half whose top net
    leaves the block as an output is on the right. LayoutFailure when the block holds anything else."""
    model = ctx.model
    a, b = found["q"]
    ga, gb = model.glyphs[a], model.glyphs[b]

    def top(g: Glyph) -> str:
        return g.net[next(p for p in g.pins if p.name in ("C", "D")).number]

    def ctrl(g: Glyph) -> str:
        return g.net[g.control().number]

    loads, biases, cpl = list(found["loads"]), list(found["biases"]), list(found["coupling"])
    if top(ga) in block.outputs and top(gb) not in block.outputs:  # the output half goes right
        a, b, ga, gb = b, a, gb, ga
        loads, biases, cpl = loads[::-1], biases[::-1], cpl[::-1]
    # cpl[0]: top(a)-ctrl(b) (left), cpl[1]: top(b)-ctrl(a) (right); inner left node ctrl(b), inner right ctrl(a)
    tails = []
    for q in (a, b):
        g = model.glyphs[q]
        e = next(p for p in g.pins if p.name in ("E", "S"))
        n = g.net.get(e.number)
        if n is None:
            raise LayoutFailure(f"cross_coupled: {q} has no emitter net")
        tail = [] if _rail(model, n) else [k for k in members if _two(model.glyphs[k]) and n in model.glyphs[k].net.values()]
        if not _rail(model, n) and (len(tail) != 1 or not _rail(model, _other(model.glyphs[tail[0]], n))):
            raise LayoutFailure(f"cross_coupled: {q}'s emitter is not on a rail or one part to it")
        tails.append(tail)
    used = {a, b, *cpl, *(k for s in loads for k in s), *(k for s in biases for k in s), *(k for t_ in tails for k in t_)}
    if used != set(members) or any(len(s) > 1 for s in loads + biases):
        raise LayoutFailure("cross_coupled: the block holds more than the pair")
    rails = {_other(model.glyphs[k], n) for s, n in zip(loads + biases, (top(ga), top(gb), ctrl(ga), ctrl(gb))) for k in s}
    if len(rails) > 1 or any(model.nets[r].role is not Role.SUPPLY for r in rails):
        raise LayoutFailure("cross_coupled: loads and biases are not on one supply")
    pen = Pen(ctx, block, "cross_coupled")
    widths = []
    for k in [*(k for s in loads + biases for k in s), *cpl]:
        trial = Pen(ctx, block, "cross_coupled")
        trial.put_two(k, sorted(model.glyphs[k].net.values())[0], (0, 0), k not in cpl, "B" if k in cpl else "L")
        box = trial.cells()
        widths.append(box[2] - box[0])
    p = max(10, max(widths) + 4)
    p += p % 2
    xs = (0, p, 2 * p, 3 * p)
    yn = 0
    # transistors: the left one mirrored, control pins facing inward, collectors under the outer columns
    rows = {}
    for q, x, side in ((a, xs[0], "R"), (b, xs[3], "L")):
        g = model.glyphs[q]
        c_pin = next(pp for pp in g.pins if pp.name in ("C", "D"))
        o = next((o for o in TRANSFORMS if pin_side(g, g.control().number, o) == side and pin_side(g, c_pin.number, o) == "T"), None)
        if o is None:
            raise LayoutFailure(f"cross_coupled: {q} cannot face its control pin {side}")
        cc, bb = pin_cell(g, c_pin, (0, 0), o), pin_cell(g, g.control(), (0, 0), o)
        pen.put(q, (x - cc[0], yn + p - bb[1]), o, "L" if side == "R" else "R")
        pen.wire(top(g), (x, yn), pen.pin(q, c_pin.number))
        rows[q] = pen.pin(q, g.control().number)
    # couplings on the node row
    for k, (x0, x1), first in ((cpl[0], (xs[0], xs[1]), top(ga)), (cpl[1], (xs[2], xs[3]), ctrl(ga))):
        pen.put_two(k, first, ((x0 + x1) // 2, yn), False, "B")
        g = model.glyphs[k]
        pen.wire(first, (x0, yn), pen.on(k, first))
        pen.wire(_other(g, first), pen.on(k, _other(g, first)), (x1, yn))
    # loads and biases standing from the rail line
    tops = []
    for slot, x, node, side in ((loads[0], xs[0], top(ga), "L"), (biases[1], xs[1], ctrl(gb), "L"),
                                (biases[0], xs[2], ctrl(ga), "R"), (loads[1], xs[3], top(gb), "R")):
        for k in slot:
            g = model.glyphs[k]
            rail = _other(g, node)
            h = abs(pin_cell(g, sorted(g.net)[0], (0, 0), orient_two(g, rail, True))[1] - pin_cell(g, sorted(g.net)[1], (0, 0), orient_two(g, rail, True))[1])
            pen.put_two(k, rail, (x, yn - 4 - h // 2), True, side)
            pen.wire(node, pen.on(k, node), (x, yn))
            tops.append((rail, pen.on(k, rail)))
    if tops:
        y = min(q[1] for _, q in tops) - 2
        for rail, q in tops:
            if q[1] != y + 2:
                raise LayoutFailure("cross_coupled: the standing parts differ in height")
        pen.rails.append(RailSpan(tops[0][0], y, min(q[0] for _, q in tops), max(q[0] for _, q in tops)))
    # the X
    for node_x, net, q in ((xs[1], ctrl(gb), b), (xs[2], ctrl(ga), a)):
        base = rows[q]
        d = base[1] - yn
        x_end = node_x + d if base[0] > node_x else node_x - d
        if (base[0] - x_end) * (1 if base[0] > node_x else -1) < 2:
            raise LayoutFailure("cross_coupled: the pitch leaves no room for the X")
        pen.wire(net, (node_x, yn), (x_end, base[1]), base)
        mid = ((x_end + base[0]) // 2, base[1])
        pen.labels.append(Label(net, mid, 270, "right", "name"))  # under its run into the control pin
    # emitters: ground symbols, or the tail part down to its rail
    for q, tail in zip((a, b), tails):
        g = model.glyphs[q]
        e = next(pp for pp in g.pins if pp.name in ("E", "S"))
        if not tail:
            pen.local(q, e.number)
            continue
        k = tail[0]
        n = g.net[e.number]
        ep = pen.pin(q, e.number)
        h = _width(model, k)
        pen.put_two(k, n, (ep[0], ep[1] + 2 + h // 2), True, "L" if q == a else "R")
        pen.wire(n, ep, pen.on(k, n))
        pen.local(k, next(pp for pp, m in model.glyphs[k].net.items() if m != n))
    # collector names on the outer drops
    for x, net, angle, just in ((xs[0], top(ga), 180, "right"), (xs[3], top(gb), 0, "left")):
        pen.labels.append(Label(net, (x, yn + 3), angle, just, "name"))
    same = all(model.glyphs[x].component.value == model.glyphs[y].component.value for x, y in [(a, b), tuple(cpl), *zip(*loads)])
    note = f"cross_coupled: {a}|{b} mirrored, coupling {cpl[0]}|{cpl[1]}, pitch {p} cells, the X as two diagonals" + \
        ("; verified swap" if same else "; mirrored by symbol (values differ)")
    return pen.drawing((note,))


def diagonal_x(ctx, d: Drawing, found: dict) -> Drawing:
    """The cross-coupling of a mirrored pair as the textbook X: each control net leaves its node on the other side of the
    axis as a 45-degree diagonal down toward the opposite transistor (a vertical drop first when the node sits higher than
    the run is wide) and enters the control pin horizontally; the two diagonals cross once between the halves. The
    columns engine's orthogonal runs are kept when the diagonals would meet a foreign pin, wire end or box."""
    from ai_eda.compilers.textbook.emit import guard

    model = ctx.model
    hints = {n: list(s) for n, s in d.hints.items()}
    changed = False
    for q in found["q"]:
        g, pl = model.glyphs[q], d.place.get(q)
        ctrl = g.control()
        if pl is None or ctrl is None or ctrl.number not in g.net:
            continue
        net = g.net[ctrl.number]
        p = pin_cell(g, ctrl, (pl.x, pl.y), pl.orient)
        side = pin_side(g, ctrl, pl.orient)
        step = -1 if side == "L" else 1 if side == "R" else 0
        segs = hints.get(net, [])
        # the columns run: anchor -> (anchor.x, y2) -> (p.x + 2 step, y2) -> (p.x + 2 step, p.y) -> p
        knee = (p[0] + 2 * step, p[1])
        s4 = next((s for s in segs if s == (knee, p)), None)
        s3 = next((s for s in segs if s[1] == knee and s[0][0] == knee[0] and s[0] != knee), None)
        if not step or s4 is None or s3 is None:
            continue
        y2 = s3[0][1]
        s2 = next((s for s in segs if s[1] == s3[0] and s[0][1] == y2), None)
        s1 = next((s for s in segs if s2 is not None and s[1] == s2[0] and s[0][0] == s2[0][0]), None)
        if s2 is None or s1 is None:
            continue
        a = s1[0]
        run = abs(knee[0] - a[0])
        drop = p[1] - a[1]
        if drop <= 0 or run <= 0:
            continue
        if run >= drop:
            path = [a, (a[0] + (drop if knee[0] > a[0] else -drop), p[1]), p]
        else:
            path = [a, (a[0], p[1] - run), knee, p]
        new = [s for s in segs if s not in (s1, s2, s3, s4)]
        new += [(u, v) for u, v in zip(path, path[1:]) if u != v]
        hints[net] = new
        changed = True
    if not changed:
        return d
    trial = replace(d, hints={n: tuple(s) for n, s in sorted(hints.items())})
    boxes = drawing_boxes(model, trial)
    alone = Sheet((trial,), ((0, 0),))
    hits = [h for h in guard(replace(ctx, notes=[]), alone, Wiring(segments={n: list(s) for n, s in trial.hints.items()}), list(trial.labels), boxes) if h.hard]
    return d if hits else trial


def supply_chain(model: Model, free) -> dict | None:
    """A power block: every glyph has a supply pin or a private signal net shared with a glyph that has one; at least one
    rail derivation element or source connector; no amplifier or IC."""
    keys = sorted(free, key=_nat)
    if not keys or any(model.glyphs[k].kind in (Kind.AMP, Kind.IC, Kind.PWRUNIT) for k in keys):
        return None
    supplied = {k for k in keys if any(model.nets[n].role is Role.SUPPLY for n in model.glyphs[k].net.values())}
    inside = set(keys)
    series = series_bridges(model, inside)
    reach = set(supplied)
    if series:  # the control network of a series pass: reached from a supplied glyph over signal nets of the block
        grew = True
        while grew:
            grew = False
            for k in keys:
                sig = [n for n in model.glyphs[k].net.values() if model.nets[n].role is Role.SIGNAL]
                if k not in reach and any(any(m in reach for m, _ in model.nets[n].pins) for n in sig):
                    reach.add(k)
                    grew = True
    for k in keys:
        if k in supplied:
            continue
        sig = [n for n in model.glyphs[k].net.values() if model.nets[n].role is Role.SIGNAL]
        if not sig or not all(all(m in inside for m, _ in model.nets[n].pins) and any(m in reach for m, _ in model.nets[n].pins) for n in sig):
            return None
    bridges = {key for _, _, key, _ in rails_mod._bridges(model)} & inside | {k for _, _, ks, _ in series for k in ks}
    sources = {k for k in keys if model.glyphs[k].kind is Kind.CONN and all(model.nets[n].role is not Role.SIGNAL for n in model.glyphs[k].net.values())}
    if not bridges and not sources:
        return None
    return {"bridges": sorted(bridges, key=_nat), "sources": sorted(sources, key=_nat)}


def _step(step) -> tuple[str, ...]:
    """A bridge of the supply line as its parts: one key, or the keys of a series pass."""
    return (step,) if isinstance(step, str) else tuple(step)


def series_bridges(model: Model, inside) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    """(upstream rail, downstream rail, glyphs, nets between them): two or three three-terminal parts of the block whose
    through pairs lead from one supply rail over signal nets to another, a series pass (for example two back-to-back
    MOSFETs of a reverse-protection switch). Upstream is the rail fed from outside (a connector or power-output pin), then
    the one with more pins, then the name, as for a one-part bridge."""
    supply = {n for n, i in model.nets.items() if i.role is Role.SUPPLY}
    sourced = {n for n in supply for k, p in model.nets[n].pins
               if model.glyphs[k].kind is Kind.CONN or model.glyphs[k].pin(p).etype == "power_out"}
    thru = sorted((k for k in inside if model.glyphs[k].through() is not None), key=_nat)

    def ends(k: str) -> tuple[str | None, str | None]:
        a, b = model.glyphs[k].through()
        return model.glyphs[k].net.get(a.number), model.glyphs[k].net.get(b.number)

    def weight(n: str) -> tuple:
        return (n not in sourced, -len(model.nets[n].pins), n)

    found: dict[tuple, tuple] = {}
    for start in sorted(supply):
        stack: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [(start, (), ())]
        while stack:
            net, ks, ns = stack.pop()
            for k in thru:
                if k in ks or net not in ends(k):
                    continue
                a, b = ends(k)
                nxt = b if a == net else a
                if nxt is None or nxt == net:
                    continue
                if nxt in supply:
                    if ks and nxt != start and weight(start) < weight(nxt):
                        found.setdefault((start, nxt, ks + (k,)), (start, nxt, ks + (k,), ns))
                elif model.nets[nxt].role is Role.SIGNAL and nxt not in ns and len(ks) < 2:
                    stack.append((nxt, ks + (k,), ns + (nxt,)))
    return [found[key] for key in sorted(found)]


def chain_plan(model: Model, members) -> dict | None:
    """The supply chain as one line: the rails of the block in derivation order, each joined to the next by one
    horizontal bridge (a two-terminal part, a switch, a regulator, or a series pass of three-terminal parts), every other
    glyph a shunt from a rail of the line to ground (one part, or two in series over a private net), a source connector
    or a part of a series pass's control network ("control"); None when the block holds anything else (a follower, a
    pot): the columns engine draws it then."""
    keys = sorted(members, key=_nat)
    inside = set(keys)
    gnd = {n for n, i in model.nets.items() if i.role is Role.GROUND}
    rails = [n for n in rails_mod.rail_order(model) if model.nets[n].role is Role.SUPPLY and any(k in inside for k, _ in model.nets[n].pins)]
    every = rails_mod._bridges(model)
    if any(vertical for _, _, k, vertical in every if k in inside):
        return None
    series = series_bridges(model, inside)
    bridges = [(a, b, k) for a, b, k, _ in every if k in inside] + [(a, b, ks) for a, b, ks, _ in series]
    child = {a: (b, k) for a, b, k in bridges}
    if len(child) != len(bridges):
        return None
    heads = [n for n in rails if not any(b == n for _, b, _ in bridges)]
    if len(heads) != 1:
        return None
    path, steps = [heads[0]], []
    while path[-1] in child:
        b, k = child[path[-1]]
        steps.append(k)
        path.append(b)
    if set(path) != set(rails):
        return None
    sources = [k for k in keys if model.glyphs[k].kind is Kind.CONN and all(model.nets[n].role is not Role.SIGNAL for n in model.glyphs[k].net.values())]
    if len(sources) > 1 or (sources and path[0] not in model.glyphs[sources[0]].net.values()):
        return None
    shunts: dict[str, list[list[str]]] = defaultdict(list)
    used = {k for step in steps for k in _step(step)} | set(sources)
    for k in keys:
        if k in used:
            continue
        g = model.glyphs[k]
        if not _two(g):
            if series:
                continue
            return None
        nets = set(g.net.values())
        rail = next((n for n in path if n in nets), None)
        if rail is not None and nets - {rail} <= gnd:
            shunts[rail].append([k])
            used.add(k)
    for k in keys:
        if k in used:
            continue
        g = model.glyphs[k]
        nets = set(g.net.values())
        rail = next((n for n in path if n in nets), None)
        if rail is None:
            continue
        if not _two(g):
            continue
        mid = _other(g, rail)
        info = model.nets[mid]
        if info.role is not Role.SIGNAL or len(info.pins) != 2:
            if series:
                continue
            return None
        low = next(m for m, _ in info.pins if m != k)
        if low in used or low not in inside or not _two(model.glyphs[low]) or _other(model.glyphs[low], mid) not in gnd:
            if series:
                continue
            return None
        shunts[rail].append([k, low])
        used |= {k, low}
    control = sorted(inside - used, key=_nat)
    if control:
        # the control network of a series pass: every net it touches is a net of the pass (between its parts or a
        # control pin), a rail of the line, ground, or private to the network
        passing = {n for _, _, ks, ns in series if ks in steps for n in ns}
        passing |= {model.glyphs[k].net.get(model.glyphs[k].control().number) for _, _, ks, _ in series if ks in steps for k in ks}
        if not passing:
            return None
        for k in control:
            for n in model.glyphs[k].net.values():
                if n in passing or n in path or n in gnd:
                    continue
                if not all(m in control for m, _ in model.nets[n].pins):
                    return None
    for rail in shunts:
        shunts[rail].sort(key=lambda col: _nat(col[0]))
    return {"rails": path, "bridges": steps, "source": sources[0] if sources else None, "shunts": dict(shunts), "control": control}


def _turn(g: Glyph, left_net: str, right_net: str, control_down: bool = False) -> Orient:
    """The first turn of ``g`` (identity first) with a pin on ``left_net`` facing left and one on ``right_net`` right
    (``control_down``: and its control pin facing down, toward the network under a series pass, when a turn allows)."""
    ctrl = g.control().number if control_down and g.control() is not None else None
    for want in ((True, False) if ctrl else (False,)):
        for o in TRANSFORMS:
            sides = {p.number: pin_side(g, p.number, o) for p in g.pins}
            if want and sides[ctrl] != "B":
                continue
            if any(sides[p] == "L" and n == left_net for p, n in g.net.items()) and any(sides[p] == "R" and n == right_net for p, n in g.net.items()):
                return o
    raise LayoutFailure(f"{g.key}: no turn puts {left_net} left of {right_net}")


def _source_pins(pen: Pen, key: str, first: str) -> None:
    """The source connector's other rail pins: each its own power symbol, except pins of one net on one side, tied by a
    wire over their stub ends to one symbol (a ground at the lowest, a supply at the highest), so no symbol stands on a
    neighbour's stub."""
    g = pen.g(key)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p, n in sorted(g.net.items(), key=lambda kv: _nat(kv[0])):
        if p != first and (g.ref, p) not in pen.model.stacked:
            groups[(n, pen.side(key, p))].append(p)
    for (net, side), pins in sorted(groups.items()):
        if len(pins) == 1 or side not in ("L", "R"):
            for p in pins:
                pen.local(key, p)
            continue
        d = rails_mod.outward(side)
        ends = sorted(((pen.pin(key, p)[0] + d[0] * rails_mod.STUB, pen.pin(key, p)[1]), p) for p in pins)
        ends.sort(key=lambda e: e[0][1])
        for end, p in ends:
            pen.wire(net, pen.pin(key, p), end)
        pen.wire(net, ends[0][0], ends[-1][0])
        hang = pen.model.nets[net].role is not Role.SUPPLY
        end, p = ends[-1] if hang else ends[0]
        pin_at = pen.pin(key, p)
        gen, stubs = rails_mod.local_glyph(pen.ctx.library, _kind_of(pen, net), net, pin_at, side, f"#PWR{len(pen.glyphs) + 1:02d}")
        pen.glyphs.append(gen)


def _kind_of(pen: Pen, net: str) -> str:
    role = pen.model.nets[net].role
    return "GND" if role is Role.GROUND else "NEG" if role is Role.NEGATIVE else "VCC"


def draw_chain_line(ctx, block: BlockInfo, plan: dict) -> Drawing:
    """The supply line left to right: the source connector, then per rail its shunts hanging to ground and its supply
    symbol on a stub up from the line, then the bridge to the next rail in the line."""
    from ai_eda.compilers.textbook.glyphs import generated

    model = ctx.model
    pen = Pen(ctx, block, "supply_chain")
    y, x = 0, 0
    src = plan["source"]
    if src is not None:
        g = model.glyphs[src]
        sides = [pin_side(g, p.number, Orient()) for p in g.pins]
        o = Orient() if sides.count("R") >= sides.count("L") else Orient(0, "y")
        first = next(p for p, n in sorted(g.net.items(), key=lambda kv: _nat(kv[0])) if n == plan["rails"][0])
        at = pin_cell(g, first, (0, 0), o)
        pen.put(src, (-at[0], -at[1]), o, "L")
        _source_pins(pen, src, first)
    rails, steps = plan["rails"], plan["bridges"]
    drives: dict[str, list[str]] = {}
    for i, rail in enumerate(rails):
        for col in plan["shunts"].get(rail, []):
            sub = Pen(ctx, block, "supply_chain")
            top, yy = rail, y + 2
            for j, k in enumerate(col):
                g = model.glyphs[k]
                o = orient_two(g, top, True)
                a, b = (pin_cell(g, p, (0, 0), o) for p in g.net)
                h = abs(a[1] - b[1])
                sub.put_two(k, top, (0, yy + h // 2), True, "R")
                sub.wire(top, (0, y if j == 0 else yy - 6), sub.on(k, top))
                low = _other(g, top)
                if j == len(col) - 1:
                    sub.local(k, next(p for p, n in g.net.items() if n == low))
                top, yy = low, sub.on(k, low)[1] + 6
            box = sub.cells()
            cx = max(x + GAP, pen.cells()[2] + 2 - box[0]) if pen.place else x + GAP
            pen.absorb(sub, cx, 0)
            pen.wire(rail, (x, y), (cx, y))
            x = cx
        gx = max(x + GAP, pen.cells()[2] + 2) if pen.place and plan["shunts"].get(rail) else x + GAP
        gen = generated(ctx.library, "VCC", rail, (gx, y - rails_mod.GLYPH_STUB), Orient(), f"#PWR{len(pen.glyphs) + 1:02d}")
        pen.glyphs.append(gen)
        pen.wire(rail, (x, y), (gx, y), (gx, y - rails_mod.GLYPH_STUB))
        x = gx
        if i == len(steps):
            break
        step = _step(steps[i])
        nets = [rail, *_between(model, step, rail), rails[i + 1]]
        drive = _drive_plan(model, step, nets, plan.get("control") or []) if len(step) > 1 else None
        hung: list[tuple[str, str]] = []  # (part, net of its lower pin): the gate drive's parts under the line
        for j, k in enumerate(step):
            g = model.glyphs[k]
            left, right = nets[j], nets[j + 1]
            o = orient_two(g, left, False) if _two(g) else _turn(g, left, right, len(step) > 1)
            pin_in = next(p for p, n in sorted(g.net.items(), key=lambda kv: _nat(kv[0])) if n == left and pin_side(g, p, o) == "L")
            pin_out = next(p for p, n in sorted(g.net.items(), key=lambda kv: _nat(kv[0])) if n == right and pin_side(g, p, o) == "R")
            at = pin_cell(g, pin_in, (0, 0), o)
            pen.put(k, (x + GAP - at[0], y - at[1]), o, "T")
            pen.wire(left, (x, y), pen.pin(k, pin_in))
            for p, n in g.net.items():
                if model.nets[n].role is Role.GROUND:
                    pen.local(k, p)
            x, y = pen.pin(k, pin_out)
            if drive is not None:
                parts = drive["gap"][j] if j < len(step) - 1 else drive["rail"]
                x = _hang(pen, parts, right, x, y, hung)
                if drive["path"] and drive["path"][0] == right:
                    x = _hang_path(pen, drive["path"][1], right, x, y, hung)
        if drive is not None:
            _gate_line(pen, step, drive, hung)
            if drive["rest"] and not drive["path"]:
                _control(ctx, block, pen, drive["rest"], step)
                drives.setdefault("columns", []).extend(drive["rest"])
    named = ", ".join("+".join(_step(step)) for step in steps) or "nothing"
    notes = [f"supply_chain: {' -> '.join(rails)} through {named}"]
    if plan.get("control"):
        rest = [k for k in plan["control"] if k in drives.get("columns", ())]
        notes.append(f"supply_chain: control network {', '.join(plan['control'])} under the series pass, on its gate line"
                     + (f" ({', '.join(rest)} by the columns engine)" if rest else ""))
    return pen.drawing(tuple(notes))


def _drive_plan(model: Model, step: tuple[str, ...], nets: list[str], control: list[str]) -> dict:
    """The gate drive of a series pass: per gap between two pass parts, the two-terminal parts from the net between them
    to a control net (drawn hanging in that gap, side by side); the two-terminal parts from a control net to the
    downstream rail (hanging from the line after the pass); the rest, as one path (:func:`_drive_path`) hanging in the
    gap after those parts, or else drawn by the columns engine under the gate line."""
    gates = {model.glyphs[k].net.get(model.glyphs[k].control().number) for k in step} - {None}
    gap: list[list[str]] = [[] for _ in step[:-1]]
    rail: list[str] = []
    rest: list[str] = []
    for k in control:
        g = model.glyphs[k]
        ns = set(g.net.values())
        if _two(g) and len(ns) == 2 and ns & gates:
            other = next(iter(ns - gates), None)
            if other in nets[1:-1]:
                gap[nets.index(other) - 1].append(k)
                continue
            if other == nets[-1]:
                rail.append(k)
                continue
        rest.append(k)
    return {"gates": sorted(gates), "gap": gap, "rail": rail, "rest": rest, "path": _drive_path(model, rest, nets[1:-1], gates)}


def _drive_path(model: Model, rest: list[str], between: list[str], gates: set[str]) -> tuple[str, list[tuple[str, str, str]]] | None:
    """The rest of a gate drive as one path from a net between pass parts to a gate net, over nets private to it: each
    part with exactly two nets that are not ground (where it enters and leaves), any other pin on ground (a switch's
    throw). (between net, [(part, upper net, lower net), ...]) or None."""
    gnd = {n for n, i in model.nets.items() if i.role is Role.GROUND}
    if not rest:
        return None
    for start in between:
        left, net, out = set(rest), start, []
        while left:
            nxt = [k for k in sorted(left, key=_nat) if net in model.glyphs[k].net.values()]
            if len(nxt) != 1:
                break
            k = nxt[0]
            ns = set(model.glyphs[k].net.values()) - gnd
            if len(ns) != 2 or net not in ns:
                break
            low = next(iter(ns - {net}))
            out.append((k, net, low))
            left.discard(k)
            net = low
            if low in gates or (low not in between and any(m not in rest for m, _ in model.nets[low].pins)):
                break
        if not left and out and net in gates:
            return start, out
    return None


def _hang_path(pen: Pen, path: list[tuple[str, str, str]], top: str, x: int, y: int, hung: list[tuple[str, str]]) -> int:
    """See :func:`_path_pen`; the fields of a part with a third pin go on the first side (above, right, below, left)
    where no two boxes of the path overlap."""
    for fields in ("T", "R", "B", "L"):
        sub, at, lower = _path_pen(pen, path, y, fields)
        if not overlaps([b for _, b in sub.boxes()]):
            break
    box = sub.cells()
    dx = x + 2 - box[0]
    pen.absorb(sub, dx, 0)
    hung.append((f"@{at[0] + dx},{at[1]}", lower))
    return box[2] + dx + 2


def _path_pen(pen: Pen, path: list[tuple[str, str, str]], y: int, fields: Side) -> tuple[Pen, Pt, str]:
    """A gate-drive path hanging from the line (net ``top``) down toward the gate line, in the gap after ``x``: a
    two-terminal part upright, a part with a third (ground) pin turned so the pin it enters by faces up or out, the one
    it leaves by down or out and no ground pin up, each next part beside the pin it hangs from, outward of it; the whole
    path 2 cells clear of ``x``. Returns the x where the line goes on."""
    model = pen.model
    gnd = {n for n, i in model.nets.items() if i.role is Role.GROUND}
    sub = Pen(pen.ctx, pen.block, pen.recipe)
    at, side = (0, y), "B"  # where the path goes on from, and the side that point leaves by
    for i, (k, upper, lower) in enumerate(path):
        g = sub.g(k)
        if _two(g):
            o = orient_two(g, upper, True)
            a = pin_cell(g, next(p for p, n in g.net.items() if n == upper), (0, 0), o)
            cx = at[0] if side == "B" else at[0] + 2 * (1 if side == "R" else -1)
            top_y = at[1] + (3 if i == 0 else 2)
            sub.put(k, (cx - a[0], top_y - a[1]), o, "R")
            sub.wire(upper, at, (cx, at[1]), sub.on(k, upper))
            at, side = sub.on(k, lower), "B"
        else:
            best = None
            for o in TRANSFORMS:
                sides = {p: pin_side(g, p, o) for p in g.net}
                up = next(p for p, n in g.net.items() if n == upper)
                lo = next(p for p, n in g.net.items() if n == lower)
                cells = {p: pin_cell(g, p, (0, 0), o) for p in g.net}
                cost = (10 * sum(1 for p, n in g.net.items() if n in gnd and sides[p] == "T") + 4 * (sides[up] == "B") + 4 * (sides[lo] == "T")
                        + 2 * sum(1 for p, n in g.net.items() if n in gnd and cells[p][1] < cells[up][1])
                        + (sides[up] not in ("R", "T")) + (sides[lo] not in ("L", "B")))
                if best is None or cost < best[0]:
                    best = (cost, o, up, lo)
            _, o, up, lo = best
            u = pin_cell(g, up, (0, 0), o)
            su = pin_side(g, up, o)
            d = rails_mod.outward(su)
            want = (at[0] - 2 * d[0], at[1] + 4) if su in ("L", "R") else (at[0], at[1] + 2)
            sub.put(k, (want[0] - u[0], want[1] - u[1]), o, fields)
            pin_at = sub.pin(k, up)
            sub.wire(upper, at, (at[0], pin_at[1]), pin_at)
            for p, n in g.net.items():
                if n in gnd:
                    sub.local(k, p)
            at, side = sub.pin(k, lo), pin_side(g, lo, o)
    lower = path[-1][2]
    if side != "B":
        end = (at[0] + 2 * (1 if side == "R" else -1), at[1])
        sub.wire(lower, at, end)
        at = end
    return sub, at, lower


def _hang(pen: Pen, parts: list[str], top: str, x: int, y: int, hung: list[tuple[str, str]]) -> int:
    """Two-terminal parts hanging side by side from the line (net ``top``) after ``x``: each 2 cells clear of the last
    box, its upper pin on ``top`` joined up to the line, its fields on its right. Returns the x where the line goes on."""
    for k in parts:
        g = pen.g(k)
        o = orient_two(g, top, True)
        a, b = (pin_cell(g, p, (0, 0), o) for p in sorted(g.net))
        h = abs(a[1] - b[1])
        sub = Pen(pen.ctx, pen.block, pen.recipe)
        sub.put_two(k, top, (0, 0), True, "R")
        box = sub.cells()
        cx = x + 2 - box[0]
        pen.put_two(k, top, (cx, y + 3 + h // 2), True, "R")
        pen.wire(top, (cx, y), pen.on(k, top))
        hung.append((k, _other(g, top)))
        x = cx + box[2] + 2
    return x


def _gate_line(pen: Pen, step: tuple[str, ...], drive: dict, hung: list[tuple[str, str]]) -> None:
    """The gate line: one row under the pass parts and the hanging drive parts, joining each control pin (straight down)
    and each hanging part's lower pin of that net."""
    model = pen.model
    for gate in drive["gates"]:
        drops = [pen.pin(k, model.glyphs[k].control().number) for k in step if model.glyphs[k].net.get(model.glyphs[k].control().number) == gate]
        drops += [_point(k) if k.startswith("@") else pen.on(k, gate) for k, low in hung if low == gate]
        if len(drops) < 2:
            continue
        low = max(p[1] for p in drops) + 2
        for p in sorted(drops):
            pen.wire(gate, p, (p[0], low))
        xs = sorted(p[0] for p in drops)
        pen.wire(gate, (xs[0], low), (xs[-1], low))


def _point(tag: str) -> Pt:
    x, y = tag[1:].split(",")
    return int(x), int(y)


def _between(model: Model, step: tuple[str, ...], rail: str) -> list[str]:
    """The nets between the parts of a series pass, from ``rail`` on."""
    out, net = [], rail
    for k in step[:-1]:
        a, b = model.glyphs[k].through()
        ends = (model.glyphs[k].net.get(a.number), model.glyphs[k].net.get(b.number))
        net = ends[1] if ends[0] == net else ends[0]
        out.append(net)
    return out


def _control(ctx, block: BlockInfo, pen: Pen, control: list[str], step: tuple[str, ...]) -> None:
    """The control network of a series pass, drawn by the columns engine as one group and set under the pass parts (its
    left edge at the first part's input pin, its top 4 cells under everything drawn so far); the router joins it to the
    control pins and the nets between the parts, the rail pins to the line."""
    from ai_eda.compilers.textbook.columns import COLUMNS
    from ai_eda.compilers.textbook.pipeline import rank

    sub = replace(block, members=tuple(control))
    variants = COLUMNS.draw(ctx, sub, frozenset(control))
    d = min((v if v.box is not None else replace(v, box=drawing_box(ctx.model, v)) for v in variants), key=lambda v: rank(ctx, v))
    box = d.box
    first = min(pen.place[k].x for k in step)
    lo = pen.cells()[3]
    dx, dy = first - box[0], lo + 4 - box[1]
    for key, pl in d.place.items():
        pen.place[key] = pl.moved(dx, dy)
        (rx, ry), (vx, vy) = d.fields[key]
        pen.fields[key] = ((rx + dx, ry + dy), (vx + dx, vy + dy))
    for net, segs in d.hints.items():
        pen.hints[net] += [((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in segs]
    pen.glyphs += [gen.moved(dx, dy) for gen in d.glyphs]
    pen.labels += [lab.moved(dx, dy) for lab in d.labels]


class SupplyChainRecipe(DelegatedRecipe):
    """SUPPLY_CHAIN: one line when :func:`chain_plan` finds one, else the columns engine's rail rows."""

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        plan = chain_plan(ctx.model, members)
        if plan is not None:
            try:
                return [draw_chain_line(ctx, block, plan)]
            except LayoutFailure as exc:
                ctx.notes.append(f"block {block.id}: supply_chain line not drawn ({exc.reason}); columns rail rows instead")
        else:
            ctx.notes.append(f"block {block.id}: supply_chain: not one line (a follower or potentiometer bridge, or a part "
                             f"off the line); columns rail rows instead")
        return super().draw(ctx, block, members)


CROSS_COUPLED = DelegatedRecipe("cross_coupled", cross_coupled)
SUPPLY_CHAIN = SupplyChainRecipe("supply_chain", supply_chain)
