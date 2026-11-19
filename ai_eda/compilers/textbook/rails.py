"""Rails: supply derivation, rail lines and power symbols.

:func:`derivation` finds which supply rail feeds which: rail B derives from rail A when one glyph joins them in series - a
two-terminal part or a switch between two supplies, a three-terminal part whose through pair sits on two supplies (a
follower) or a regulator from its input to its output pin. Direction: a power-output pin is downstream, a diode conducts
anode to cathode, a follower's top pin (at rotation 0) is upstream; otherwise the rail fed from outside (a connector or
power-output pin) is upstream, then the one with more pins, then the name. :func:`rail_order` is the breadth-first order
from the rails with no parent (ties by name); a horizontal bridge keeps the level, a vertical one adds one.

:func:`local_glyph` is the power symbol of one pin: a ground (or negative-rail) symbol 2 cells below a bottom pin, or
hanging under the end of a 2-cell outward stub from a side pin; a supply symbol 2 cells above a top pin, or standing on
such a stub. :func:`draw_rails` draws, per band, one line per supply at the rail row its drawings share: vertical drops to
the pins facing it, a supply symbol (Value = the rail name) on a 3-cell stub up from the line, a line broken where another
rail's drop crosses it (each piece gets its own symbol). A drop or line that would pass a foreign pin, body, text or wire
is not drawn, with a note (the router joins those pins). PWR_FLAGs go on the rail line, or on a short wire beside one of
the net's power symbols, by a free-spot search: for the existing type rule, and for every net a power symbol of the layout
sits on that nothing drives (the symbol's pin is a power input, so the same rule applies to it).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook.glyphs import generated, generated_boxes, generated_pin, generated_side, glyph_boxes, mm, outward, pin_cell, pin_side
from ai_eda.compilers.textbook.model import BoxMM, Generated, Kind, Link, Model, Orient, Pt, RailSpan, Role, Seg, Sheet, Side, is_test_point

FLAG_PITCH = 8  # cells between PWR_FLAGs in the fallback row under the content
FLAG_GAP = 6  # cells from the content to that row
STUB = 2  # cells from a pin to its local power symbol
GLYPH_STUB = 3  # cells from a rail line up to its supply symbol
NEAR_MM = 1.27  # a wire never comes closer than this to a body that has no pin on its net
TEXT_MM = 0.5  # nor this close to text
FLAG_OFFSETS = (4, -4, 6, -6, 8, -8, 10, -10)


# --------------------------------------------------------------------------- derivation


def _bridges(model: Model) -> list[tuple[str, str, str, bool]]:
    """(parent rail, child rail, glyph key, vertical) for every glyph that joins two supply rails in series."""
    supply = {n for n, info in model.nets.items() if info.role is Role.SUPPLY}
    sourced = {n for n in supply for k, p in model.nets[n].pins
               if model.glyphs[k].kind is Kind.CONN or model.glyphs[k].pin(p).etype == "power_out"}
    out = []
    for key in sorted(model.glyphs, key=natural_ref_key):
        g = model.glyphs[key]
        on = {p: n for p, n in g.net.items() if n in supply}
        rails = sorted(set(on.values()))
        if len(rails) != 2 or g.kind in (Kind.CONN, Kind.AMP, Kind.IC, Kind.PWRUNIT):
            continue
        if g.kind is Kind.THRU3:
            pair = g.through()
            if pair is None or {pair[0].number, pair[1].number} != set(on):
                continue
            outs = [p for p in pair if p.etype == "power_out"]
            child = on[outs[0].number] if outs else on[pair[1].number]
            out.append((rails[0] if child == rails[1] else rails[1], child, key, pair[0].side == "T"))
            continue
        names = {g.pin(p).name: n for p, n in on.items()}
        if {"A", "K"} <= set(names) and names["A"] != names["K"]:
            out.append((names["A"], names["K"], key, False))
            continue

        def weight(n: str) -> tuple:
            return (n not in sourced, -len(model.nets[n].pins), n)

        a, b = sorted(rails, key=weight)
        out.append((a, b, key, False))
    return out


def derivation(model: Model) -> dict[str, tuple[str | None, str | None, int]]:
    """Supply rail -> (parent rail, bridging glyph key, level), breadth first from the rails with no parent."""
    rails = sorted(n for n, info in model.nets.items() if info.role in (Role.SUPPLY, Role.NEGATIVE))
    bridges = _bridges(model)
    children: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    has_parent = set()
    for parent, child, key, vertical in bridges:
        children[parent].append((child, key, vertical))
        has_parent.add(child)
    out: dict[str, tuple[str | None, str | None, int]] = {}
    queue = [n for n in rails if n not in has_parent]
    for n in queue:
        out[n] = (None, None, 0)
    while queue:
        nxt = []
        for n in queue:
            for child, key, vertical in sorted(children[n]):
                if child not in out:
                    out[child] = (n, key, out[n][2] + (1 if vertical else 0))
                    nxt.append(child)
        queue = nxt
    for n in rails:  # rails only reachable through a cycle of bridges
        out.setdefault(n, (None, None, 0))
    return out


def rail_order(model: Model) -> tuple[str, ...]:
    """The supply rails in derivation order: breadth first from the rails with no parent, ties by name."""
    tree = derivation(model)
    order: list[str] = []
    level = sorted(n for n, (parent, _, _) in tree.items() if parent is None)
    while level:
        order += level
        level = sorted(n for n, (parent, _, _) in tree.items() if parent in level and n not in order)
    return tuple(order)


# --------------------------------------------------------------------------- local power symbols


def _orient_body(library, kind: str, net: str, at: Pt, want: Side, ref: str) -> Generated:
    """The power symbol at ``at`` turned so its pin leaves the body on side ``want`` (T: body below the pin)."""
    for o in (Orient(), Orient(180), Orient(90), Orient(270)):
        gen = generated(library, kind, net, at, o, ref)
        if generated_side(gen) == want:
            return gen
    return generated(library, kind, net, at, Orient(), ref)


def local_glyph(library, kind: str, net: str, pin_at: Pt, side: Side, ref: str) -> tuple[Generated, tuple[Seg, ...]]:
    """The power symbol of one pin at ``pin_at`` leaving its body on ``side``, and the stub to it. GND / NEG hang below
    (from a bottom pin, or below the end of an outward stub from a side pin); VCC stands above."""
    d = outward(side)
    end = (pin_at[0] + d[0] * STUB, pin_at[1] + d[1] * STUB)
    hang = kind in ("GND", "NEG")
    if side == "T" and hang or side == "B" and not hang:
        gen = _orient_body(library, kind, net, end, "B" if hang else "T", ref)  # the wrong way round: body continues outward
    else:
        gen = _orient_body(library, kind, net, end, "T" if hang else "B", ref)
    return gen, ((pin_at, end),)


# --------------------------------------------------------------------------- free space


def _seg_box(a: Pt, b: Pt) -> BoxMM:
    return mm(min(a[0], b[0])), mm(min(a[1], b[1])), mm(max(a[0], b[0])), mm(max(a[1], b[1]))


def _gap(a: BoxMM, b: BoxMM) -> float:
    dx, dy = max(a[0] - b[2], b[0] - a[2]), max(a[1] - b[3], b[1] - a[3])
    return max(dx, dy) if dx <= 0 and dy <= 0 else math.hypot(max(dx, 0.0), max(dy, 0.0))


class Space:
    """The sheet as drawn so far, for clearance tests of new wires and symbols: bodies (and the nets their pins are on),
    text boxes, pin points with their nets, every wire point per net."""

    def __init__(self, model: Model, sheet: Sheet) -> None:
        self.bodies: list[tuple[BoxMM, set[str]]] = []
        self.texts: list[BoxMM] = []
        nets_of: dict[str, set[str]] = {key: set(model.glyphs[key].net.values()) for key in sheet.places()}
        nets_of.update({gen.ref: {gen.net} for gen in sheet.generated()})
        for tag, box in glyph_boxes(model, sheet):
            kind, owner = tag.split(":", 1)
            if kind in ("body", "glyph"):
                self.bodies.append((box, nets_of.get(owner, set())))
            else:
                self.texts.append(box)
        self.pins: dict[Pt, set[str | None]] = defaultdict(set)
        self.escapes: dict[Pt, set[str]] = defaultdict(set)  # the 2 cells a connected pin leaves by, kept for its net
        for key, place in sheet.places().items():
            g = model.glyphs[key]
            for p in g.pins:
                at = pin_cell(g, p, (place.x, place.y), place.orient)
                self.pins[at].add(g.net.get(p.number))
                if p.number in g.net:
                    d = outward(pin_side(g, p, place.orient))
                    for i in (1, 2):
                        self.escapes[(at[0] + d[0] * i, at[1] + d[1] * i)].add(g.net[p.number])
        for gen in sheet.generated():
            self.pins[generated_pin(gen)].add(gen.net)
        for lab in sheet.labels():
            self.pins[lab.at].add(lab.net)
        self.points: dict[Pt, set[str]] = defaultdict(set)
        self.runs: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
        for net, segs in sheet.hints().items():
            for s in segs:
                self.add_wire(net, s)
        for lab in sheet.labels():
            if lab.stub is not None:
                self.add_wire(lab.net, lab.stub)

    def add_wire(self, net: str, seg: Seg) -> None:
        (ax, ay), (bx, by) = seg
        n = max(abs(bx - ax), abs(by - ay))
        for i in range(n + 1):
            self.points[(ax + (bx - ax) * i // max(n, 1), ay + (by - ay) * i // max(n, 1))].add(net)
        if ay == by and ax != bx:
            self.runs[("h", ay)].append((min(ax, bx), max(ax, bx), net))
        elif ax == bx and ay != by:
            self.runs[("v", ax)].append((min(ay, by), max(ay, by), net))

    def add_glyph(self, gen: Generated) -> None:
        body, fields = generated_boxes(gen)
        self.bodies.append((body, {gen.net}))
        self.texts += list(fields)
        self.pins[generated_pin(gen)].add(gen.net)

    def wire_clear(self, net: str, seg: Seg, ends_ok: set[Pt] = frozenset()) -> bool:  # type: ignore[assignment]
        """An axis-parallel wire of ``net``: no foreign pin or wire point on it (its own pins only at its ends or where it
        joins), no foreign body within 1.27 mm, no text crossed, no foreign wire 1 cell beside it."""
        (ax, ay), (bx, by) = seg
        n = max(abs(bx - ax), abs(by - ay))
        for i in range(n + 1):
            p = (ax + (bx - ax) * i // max(n, 1), ay + (by - ay) * i // max(n, 1))
            if any(o != net for o in self.pins.get(p, ())) or any(o != net for o in self.points.get(p, ())) or self.escapes.get(p, {net}) != {net}:
                return False
        box = _seg_box(seg[0], seg[1])
        for body, nets in self.bodies:
            if net not in nets and _gap(box, body) < NEAR_MM - 1e-9:
                return False
        if any(_gap(box, text) < TEXT_MM - 1e-9 for text in self.texts):
            return False
        axis, level, lo, hi = ("h", ay, min(ax, bx), max(ax, bx)) if ay == by else ("v", ax, min(ay, by), max(ay, by))
        for side in (level - 1, level + 1):
            for a, b, other in self.runs.get((axis, side), ()):
                if other != net and min(hi, b) - max(lo, a) > 0:
                    return False
        return True

    def glyph_clear(self, gen: Generated) -> bool:
        """The symbol's body and shown text overlap nothing and keep 0.5 mm from text; no foreign wire runs through it."""
        body, fields = generated_boxes(gen)
        for box in (body, *fields):
            if any(_gap(box, other) < TEXT_MM - 1e-9 for other, _ in self.bodies) or any(_gap(box, t) < TEXT_MM - 1e-9 for t in self.texts):
                return False
            for p, nets in self.points.items():
                if (nets - {gen.net}) and box[0] - NEAR_MM < mm(p[0]) < box[2] + NEAR_MM and box[1] - NEAR_MM < mm(p[1]) < box[3] + NEAR_MM:
                    return False
            for p, nets in self.pins.items():
                if (nets - {gen.net}) and box[0] - NEAR_MM < mm(p[0]) < box[2] + NEAR_MM and box[1] - NEAR_MM < mm(p[1]) < box[3] + NEAR_MM:
                    return False
        return True


# --------------------------------------------------------------------------- sheet rails


def _renumber(sheet: Sheet) -> Sheet:
    """The drawings' power symbols numbered #PWR01.. across the sheet, in drawing and symbol order."""
    n = 0
    drawings = []
    for d in sheet.drawings:
        glyphs = []
        for gen in d.glyphs:
            if gen.kind == "FLAG":
                glyphs.append(gen)
                continue
            n += 1
            glyphs.append(replace(gen, ref=f"#PWR{n:02d}"))
        drawings.append(replace(d, glyphs=tuple(glyphs)))
    return replace(sheet, drawings=tuple(drawings))


def _attachments(model: Model, sheet: Sheet, band: tuple[int, ...], net: str, y: int) -> list[tuple[Pt, Pt, str]]:
    """(point on the line, pin point, pin side) of every pin of ``net`` in the band's drawings that faces a line at ``y``:
    a top pin below it, a bottom pin above it, a side pin on it."""
    out = []
    for i in band:
        d, (dx, dy) = sheet.drawings[i], sheet.offset[i]
        if d.recipe == "grid":
            continue  # the block grid names its pins by labels
        for key in sorted(d.place, key=natural_ref_key):
            g, pl = model.glyphs[key], d.place[key].moved(dx, dy)
            if g.kind in (Kind.AMP, Kind.IC, Kind.PWRUNIT, Kind.CONN):
                continue
            for number, n in sorted(g.net.items(), key=lambda kv: natural_ref_key(kv[0])):
                if n != net or (g.ref, number) in model.stacked:
                    continue
                p = pin_cell(g, number, (pl.x, pl.y), pl.orient)
                side = pin_side(g, number, pl.orient)
                if side == "T" and p[1] > y or side == "B" and p[1] < y or side in ("L", "R") and p[1] == y:
                    out.append(((p[0], y), p, side))
    return out


def draw_rails(ctx, sheet: Sheet) -> Sheet:
    """The rail lines, their supply symbols and the PWR_FLAGs of ``sheet`` (sheet level, absolute), see the module text."""
    model: Model = ctx.model
    for note in sheet.notes:  # the arrangement's notes go with the sheet variant being finished
        if note not in ctx.notes:
            ctx.notes.append(note)
    sheet = _renumber(replace(sheet, glyphs=(), links=(), rails=()))
    space = Space(model, sheet)
    count = sum(1 for d in sheet.drawings for g in d.glyphs if g.kind != "FLAG")
    glyphs: list[Generated] = []
    links: list[Link] = []
    spans: list[RailSpan] = []
    bands = sheet.bands or (tuple(range(len(sheet.drawings))),)
    order = [n for n in rail_order(model) if model.nets[n].role is Role.SUPPLY]
    for band in bands:
        drops_of: list[tuple[str, int, int, int]] = []  # (net, x, y top, y bottom) of the drops drawn in this band
        for net in order:
            ys = [s.y + sheet.offset[i][1] for i in band for s in sheet.drawings[i].rails if s.net == net]
            if not ys:
                continue
            y = min(ys)
            atts = _attachments(model, sheet, band, net, y)
            breaks = sorted({x for other, x, top, bottom in drops_of if other != net and top < y < bottom})
            pieces: list[list[tuple[Pt, Pt, str]]] = [[]]
            for a in sorted(atts):
                while len(pieces) <= sum(1 for b in breaks if b < a[0][0]):
                    pieces.append([])
                pieces[sum(1 for b in breaks if b < a[0][0])].append(a)
            if len(pieces) > 1 and sum(1 for p in pieces if p) > 1:
                ctx.notes.append(f"rail {net}: line broken where another rail's drop crosses it; each piece has its own symbol")
            for piece in (p for p in pieces if p):
                count, drawn = _piece(ctx, space, net, y, piece, count, glyphs, links)
                if drawn:
                    spans.append(RailSpan(net, y, min(a[0][0] for a in piece), max(a[0][0] for a in piece)))
                    drops_of += [(net, a[0][0], min(a[0][1], a[1][1]), max(a[0][1], a[1][1])) for a in piece]
    # a test point's power symbol names its pad's net and nothing more: no PWR_FLAG goes on it or beside it (the pads hang in
    # a row of their own under the content, and a flag there would be wired up into the drawing)
    pads = {i for i, d in enumerate(sheet.drawings) if d.place and all(is_test_point(model.glyphs[k]) for k in d.place)}
    symbols = [gen.moved(*sheet.offset[i]) for i, d in enumerate(sheet.drawings) if i not in pads for gen in d.glyphs] + list(sheet.glyphs) + glyphs
    flags, flag_links = _flags(ctx, symbols, space, links, list(model.flags) + _symbol_flags(ctx, symbols))
    return replace(sheet, glyphs=tuple(glyphs) + tuple(flags), links=tuple(links) + tuple(flag_links), rails=tuple(spans))


def _symbol_flags(ctx, symbols: list[Generated]) -> list[tuple[str, str]]:
    """(net, flag ref) for each net a power symbol of the layout sits on that nothing drives: the symbol's pin is a power
    input, so the existing type rule applies to it too (no power-output pin, every pin a power input, input or passive).
    A net with a pin of another type gets no flag and a note."""
    model: Model = ctx.model
    flagged = {net for net, _ in model.flags}
    used = [int(ref[4:]) for _, ref in model.flags if ref[4:].isdigit()]
    out = []
    for net in sorted({gen.net for gen in symbols if gen.kind != "FLAG"} - flagged):
        types = {model.glyphs[k].pin(p).etype for k, p in model.nets[net].pins}
        if "power_out" in types:
            continue
        if types - {"power_in", "input", "passive"}:
            ctx.notes.append(f"power flags: none on {net} (a pin there is neither power-input, input nor passive)")
            continue
        n = max(used, default=0) + 1
        used.append(n)
        out.append((net, f"#FLG{n:02d}"))
    if out:
        ctx.notes.append(f"power flags: PWR_FLAG on {len(out)} net(s) carrying a power symbol and no power-output pin "
                         f"({', '.join(net for net, _ in out)})")
    return out


def _piece(ctx, space: Space, net: str, y: int, piece: list[tuple[Pt, Pt, str]], count: int, glyphs: list[Generated],
           links: list[Link]) -> tuple[int, bool]:
    """One rail line (or a single attachment) with its drops and its supply symbol; False when it is not drawn."""
    model = ctx.model
    drops = [(a, p) for a, p, side in piece if a != p]
    x0, x1 = min(a[0] for a, _, _ in piece), max(a[0] for a, _, _ in piece)
    segs: list[Seg] = [((x0, y), (x1, y))] if x1 > x0 else []
    segs += [(a, p) if a[1] < p[1] else (p, a) for a, p in drops]
    sides = {a: side for a, _, side in piece if side in ("L", "R")}
    if any(side == "L" and a[0] != x1 or side == "R" and a[0] != x0 for a, side in sides.items()):
        ctx.notes.append(f"rail {net}: a side pin on the line does not face it; its pins are joined by the router")
        return count, False
    if not all(space.wire_clear(net, s) for s in segs):
        ctx.notes.append(f"rail {net}: a line at y {y} would pass a foreign pin, body, text or wire; its pins are joined by the router")
        return count, False
    for s in segs:
        space.add_wire(net, s)
    taken = {a[0] for a, _ in drops}
    candidates = sorted((x for x in range(x1, x0 - 1, -1) if (x, y) not in sides), key=lambda x: (x in taken and x not in (x0, x1), -x))
    if (x1, y) not in sides or sides[(x1, y)] != "L":
        candidates += [x1 + k for k in range(2, 11)]  # the line continues past its right end to the symbol
    if (x0, y) not in sides or sides[(x0, y)] != "R":
        candidates += [x0 - k for k in range(2, 11)]
    gen, extra = None, []
    lone = len(piece) == 1 and len(drops) == 1 and drops[0][0][1] < drops[0][1][1]  # one pin below: its drop is the stub
    for x in candidates:
        top = (x, y - (0 if lone and x == x0 else GLYPH_STUB))
        if any(p[1] < y and abs(p[0] - x) < 2 for _, p in drops):
            continue  # a source pin rises into the line here
        reach = [((x1, y), (x, y))] if x > x1 else [((x, y), (x0, y))] if x < x0 else []
        stub = [((x, y), top)] if top != (x, y) else []
        trial = generated(ctx.library, "VCC", net, top, Orient(), f"#PWR{count + 1:02d}")
        if all(space.wire_clear(net, s) for s in reach + stub) and space.glyph_clear(trial):
            gen, extra = trial, reach + stub
            break
    if gen is None:
        ctx.notes.append(f"rail {net}: no clear spot for its symbol on the line at y {y}; the net is named by a label")
    else:
        count += 1
        glyphs.append(gen)
        for s in extra:
            space.add_wire(net, s)
        space.add_glyph(gen)
        segs += extra
    if segs:
        links.append(Link(net, (x0, y), (x1, y), "wire", tuple(segs)))
    return count, True


def _flags(ctx, symbols: list[Generated], space: Space, links: list[Link], wanted: list[tuple[str, str]]) -> tuple[list[Generated], list[Link]]:
    """A PWR_FLAG for each (net, ref) of ``wanted``: on a rail line of the net (body above it), else on a short wire
    beside one of its power symbol pins; else in a row under the content, joined to its net by its name label."""
    model: Model = ctx.model
    flags: list[Generated] = []
    out: list[Link] = []
    fallback = 0
    lines = defaultdict(list)
    for link in links:
        lines[link.net] += [s for s in link.segs if s[0][1] == s[1][1]]
    for net, ref in wanted:
        placed = None
        for (a, b) in sorted(lines.get(net, ()), key=lambda s: -max(s[0][0], s[1][0])):
            for x in range(max(a[0], b[0]) - 1, min(a[0], b[0]), -1):
                trial = generated(ctx.library, "FLAG", net, (x, a[1]), Orient(), ref)
                if space.glyph_clear(trial) and not any(o != net for o in space.points.get((x, a[1] - 1), ())):
                    placed = (trial, ())
                    break
            if placed:
                break
        if placed is None:
            own = [gen for gen in symbols if gen.net == net and gen.kind != "FLAG"]
            # the lowest ground symbol (the highest supply symbol) first: the edge of the drawing has room
            points = sorted({generated_pin(gen) for gen in own}, key=lambda p: (p[1] if any(g.kind == "VCC" for g in own) else -p[1], p[0]))
            for p in points:
                for off in FLAG_OFFSETS:
                    spot = (p[0] + off, p[1])
                    wire = (p, spot) if off > 0 else (spot, p)
                    trial = generated(ctx.library, "FLAG", net, spot, Orient(), ref)
                    if space.wire_clear(net, wire) and space.glyph_clear(trial):
                        placed = (trial, (wire,))
                        break
                if placed:
                    break
        if placed is None:
            boxes = [b for b, _ in space.bodies] + space.texts
            left = min((b[0] for b in boxes), default=mm(20))
            y = round(max((b[3] for b in boxes), default=mm(20)) / mm(1)) + FLAG_GAP
            flag = generated(ctx.library, "FLAG", net, (round(left / mm(1)) + 2 + fallback * FLAG_PITCH, y), Orient(), ref)
            pin = generated_pin(flag)
            placed = (flag, ((pin, (pin[0], pin[1] + 2)),))
            fallback += 1
            ctx.notes.append(f"PWR_FLAG {ref} on {net}: no clear spot beside the net; placed under the content, joined by its name")
        flag, segs = placed
        flags.append(flag)
        space.add_glyph(flag)
        for s in segs:
            space.add_wire(net, s)
        if segs:
            out.append(Link(net, segs[0][0], segs[-1][1], "wire", tuple(segs)))
    return flags, out
