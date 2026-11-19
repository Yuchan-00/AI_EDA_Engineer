"""Net names on the sheet: one global label (shape passive) per wire component that nothing else names.

Global labels keep every IR net name (DRC ``--schematic-parity``); a local
label on the root sheet would name its net ``/NAME``. A component is named
when it holds a power symbol of its net (the symbol's Value is the net name)
or a label already. Candidates for a new label, in order: a horizontal
segment (the lowest first, the label hanging below the wire), a vertical
segment, a 2-cell stub from a free point of a segment, a pin stub. A
candidate is refused when its box comes within 0.5 mm of a symbol, field or
label box, or within 1.27 mm of another net's wire, and when a wire of its own net
runs through its box (a label sits on the wire it ends, never along one). When only pins are left
and none is clear (a dense pin row), the label goes on the first pin stub
whose box overlaps nothing, and the notes count it.
"""

from __future__ import annotations

import math

from ai_eda.compilers.schematic_layout import label_orientation, natural_ref_key
from ai_eda.compilers.textbook.glyphs import generated_pin, generated_side, label_box, mm, outward, pin_cell, pin_side
from ai_eda.compilers.textbook.model import BoxMM, Label, Model, Place, Pt, Seg, Sheet, Side, Wiring

BOX_CLEARANCE_MM = 0.5
WIRE_CLEARANCE_MM = 1.27
STUB = 2


def stub_label(net: str, pin_at: Pt, side: Side, kind: str = "pin", length: int = STUB) -> Label:
    """A label at the end of a ``length``-cell stub leaving ``pin_at`` outward on ``side``, its text continuing outward."""
    d = outward(side)
    end = (pin_at[0] + d[0] * length, pin_at[1] + d[1] * length)
    angle, justify = label_orientation((-d[0], -d[1]))
    return Label(net, end, angle, justify, kind, (pin_at, end))  # type: ignore[arg-type]


def terminal(net: str, pin_at: Pt, side: Side) -> Label:
    """The named terminal of a connector pin (Vin / Vout): its label doubles as the net name."""
    return stub_label(net, pin_at, side, "terminal")


def pin_labels(model: Model, places: dict[str, Place], keys: list[tuple[str, str]], kind: str = "pin") -> list[Label]:
    """A stub label at every ``(glyph key, pin number)`` of ``keys`` (glyphs at ``places``), named after the pin's net."""
    out = []
    for key, number in keys:
        g, p = model.glyphs[key], places[key]
        out.append(stub_label(g.net[number], pin_cell(g, number, (p.x, p.y), p.orient), pin_side(g, number, p.orient), kind))
    return out


def on_segment(p: Pt, a: Pt, b: Pt) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) == (b[1] - a[1]) * (p[0] - a[0])
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def components(segments: list[Seg], pins: list[Pt]) -> list[tuple[list[Seg], list[Pt]]]:
    """Connected groups of one net's segments and pin points as KiCad joins them: a segment end on another segment, a pin
    or label point on a segment. Ordered by their first pin (or segment)."""
    items: list[tuple[str, object]] = [("s", s) for s in segments] + [("p", p) for p in sorted(set(pins))]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (ki, a) in enumerate(items):
        for j in range(i):
            kj, b = items[j]
            if ki == "s" and kj == "s":
                joined = any(on_segment(p, *b) for p in a) or any(on_segment(p, *a) for p in b)  # type: ignore[misc]
            elif ki == "p" and kj == "p":
                joined = a == b
            else:
                seg, pt = (a, b) if ki == "s" else (b, a)
                joined = on_segment(pt, *seg)  # type: ignore[misc]
            if joined:
                parent[find(i)] = find(j)
    groups: dict[int, tuple[list[Seg], list[Pt]]] = {}
    for i, (k, v) in enumerate(items):
        segs, pts = groups.setdefault(find(i), ([], []))
        (segs if k == "s" else pts).append(v)  # type: ignore[arg-type]
    return sorted(groups.values(), key=lambda g: (min(g[1]) if g[1] else min(min(s) for s in g[0])))


def _gap(a: BoxMM, b: BoxMM) -> float:
    """Distance between two boxes; zero or less when they touch or overlap."""
    dx, dy = max(a[0] - b[2], b[0] - a[2]), max(a[1] - b[3], b[1] - a[3])
    return max(dx, dy) if dx <= 0 and dy <= 0 else math.hypot(max(dx, 0.0), max(dy, 0.0))


def _seg_box(s: Seg) -> BoxMM:
    return mm(min(s[0][0], s[1][0])), mm(min(s[0][1], s[1][1])), mm(max(s[0][0], s[1][0])), mm(max(s[0][1], s[1][1]))


class _Space:
    """The sheet's text / body boxes and every net's wires, for clearance tests of label candidates."""

    def __init__(self, boxes: list[tuple[str, BoxMM]], wires: dict[str, list[Seg]]) -> None:
        self.boxes = [b for _, b in boxes]
        self.wires = [(net, _seg_box(s)) for net, segs in sorted(wires.items()) for s in segs]

    def clear(self, net: str, box: BoxMM, box_gap: float, wire_gap: float) -> bool:
        if any(_gap(box, b) < box_gap - 1e-9 for b in self.boxes):
            return False
        return not any(n != net and _gap(box, b) < wire_gap - 1e-9 for n, b in self.wires)

    def add(self, net: str, label: Label) -> None:
        self.boxes.append(label_box(label))
        if label.stub is not None:
            self.wires.append((net, _seg_box(label.stub)))


def _on_wire(label: Label, segs: list[Seg]) -> bool:
    """A segment of the label's own net runs through its box (beyond the anchor it ends at): the label would lie along a
    wire, and a stub there would double it."""
    x0, y0, x1, y1 = label_box(label)
    x0, y0, x1, y1 = x0 + 0.2, y0 + 0.2, x1 - 0.2, y1 - 0.2
    for a, b in segs:
        sx0, sy0, sx1, sy1 = _seg_box((a, b))
        if sx0 < x1 and sx1 > x0 and sy0 < y1 and sy1 > y0:
            return True
    return False


def _interior(seg: Seg, avoid: set[Pt]) -> list[Pt]:
    (ax, ay), (bx, by) = seg
    n = math.gcd(abs(bx - ax), abs(by - ay))
    pts = [(ax + i * (bx - ax) // n, ay + i * (by - ay) // n) for i in range(n + 1)] if n else []
    return [p for p in pts if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= STUB for q in avoid)]


def _candidates(net: str, segs: list[Seg], pins: list[tuple[Pt, Side]], ends: set[Pt]) -> tuple[list[Label], list[Label]]:
    """(wire candidates 1-3, pin-stub candidates 4) for one wire component."""
    avoid = ends | {p for p, _ in pins}
    horizontal = sorted((s for s in segs if s[0][1] == s[1][1] and s[0] != s[1]), key=lambda s: (-s[0][1], min(s[0][0], s[1][0])))
    vertical = sorted((s for s in segs if s[0][0] == s[1][0] and s[0] != s[1]), key=lambda s: (min(s[0][0], s[1][0]), min(s[0][1], s[1][1])))
    wire = [Label(net, p, 270, "right", "name") for s in horizontal for p in _interior(s, avoid)]
    wire += [Label(net, p, 0, "left", "name") for s in vertical for p in _interior(s, avoid)]
    wire += [stub_label(net, p, "B", "name") for s in horizontal for p in _interior(s, avoid)]
    wire += [stub_label(net, p, "R", "name") for s in vertical for p in _interior(s, avoid)]
    stubs = []
    for at, side in pins:
        d = outward(side)
        end = (at[0] + d[0] * STUB, at[1] + d[1] * STUB)
        existing = (at, end) in segs or (end, at) in segs
        label = stub_label(net, at, side, "name")
        stubs.append(Label(net, end, label.angle, label.justify, "name", None) if existing else label)
    return wire, stubs


def _pins_of(model: Model, sheet: Sheet) -> dict[str, list[tuple[Pt, Side]]]:
    """Every connected, drawn pin point per net (stacked pins share their representative's point), and power symbol pins."""
    out: dict[str, list[tuple[Pt, Side]]] = {}
    for key, place in sorted(sheet.places().items(), key=lambda kv: natural_ref_key(kv[0])):
        g = model.glyphs[key]
        for number in sorted(g.net, key=natural_ref_key):
            if (g.ref, number) in model.stacked:
                continue
            at = pin_cell(g, number, (place.x, place.y), place.orient)
            out.setdefault(g.net[number], []).append((at, pin_side(g, number, place.orient)))
    for gen in sheet.generated():
        out.setdefault(gen.net, []).append((generated_pin(gen), generated_side(gen)))
    return out


def name_nets(ctx, sheet: Sheet, wiring: Wiring, boxes: list[tuple[str, BoxMM]]) -> list[Label]:
    """One global label per wire component that no power symbol or existing label names, by the candidates and
    clearances of the module text. ``boxes`` are the sheet's measured bodies, fields, labels and glyphs;
    ``wiring.segments`` every drawn wire."""
    model: Model = ctx.model
    placed = sheet.labels()
    named_points: dict[str, set[Pt]] = {}
    for lab in placed:
        named_points.setdefault(lab.net, set()).add(lab.at)
    for gen in sheet.generated():
        if gen.names_net:
            named_points.setdefault(gen.net, set()).add(generated_pin(gen))
    wires = {net: list(segs) for net, segs in wiring.segments.items()}
    for lab in placed:
        if lab.stub is not None:
            wires.setdefault(lab.net, []).append(lab.stub)
    space = _Space(boxes, wires)
    pins_by_net = _pins_of(model, sheet)
    out: list[Label] = []
    tight = 0
    for net in sorted(set(pins_by_net) | set(wires)):
        pins = pins_by_net.get(net, [])
        side_of = {}
        for at, side in pins:
            side_of.setdefault(at, side)
        for segs, pts in components(wires.get(net, []), [p for p, _ in pins] + sorted(named_points.get(net, ()))):
            if not pts or any(p in named_points.get(net, ()) for p in pts):
                continue
            ends = {p for s in segs for p in s}
            wire, stubs = _candidates(net, segs, [(p, side_of[p]) for p in sorted(pts) if p in side_of], ends)
            own = wires.get(net, [])
            choice = next((c for c in wire + stubs if not _on_wire(c, own) and space.clear(net, label_box(c), BOX_CLEARANCE_MM, WIRE_CLEARANCE_MM)), None)
            if choice is None:
                choice = next((c for c in stubs if not _on_wire(c, own) and space.clear(net, label_box(c), 0.0, WIRE_CLEARANCE_MM / 2)), None)
                tight += choice is not None
            if choice is None and (stubs or wire):
                choice = next((c for c in stubs + wire if not _on_wire(c, own)), (stubs or wire)[0])
                ctx.notes.append(f"net {net}: no clear spot for its name label; placed at {choice.at}")
            if choice is not None:
                out.append(choice)
                space.add(net, choice)
    if tight:
        ctx.notes.append(f"labels: {tight} name label(s) on a dense pin row sit closer than {BOX_CLEARANCE_MM} mm to a neighbour (no overlap)")
    return out
