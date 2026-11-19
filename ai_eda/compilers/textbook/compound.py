"""Combinations: two like resistors that make one value, drawn as one element.

A value the parts at hand do not have is often made of two resistors: in series (one after the other, the net between
them joins nothing else) or in parallel (both across the same two nets). The layout draws such a pair as one element:
:func:`contract` replaces the pair, before any block is drawn, by one two-terminal glyph whose symbol is the pair's outline
(in series: the two bodies in line, pin to pin, with room for the name of the net between them on one side; in parallel:
the two bodies side by side with room for the inner one's fields between them, joined at both ends), so every recipe
places it like one resistor. :func:`expand` puts the two real parts back where the outline was, their fields beside
their own bodies, the net between a series pair named by a label on a stub from the point where the two pins meet, the
ends of a parallel pair joined by wires. A series pair's fields stand in the outline's field row, each centred on its own part's
body (beside it for a column), so the middle net's name, on its stub from the point where the parts meet, stands between
the two parts' texts and never under one of them.

A pair is two glyphs whose library default Reference is ``R`` (the library's own field), of one library symbol, in the
same declared block (or both in none), with no stacked pins; each glyph is in at most one pair (natural order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ai_eda.compilers.schematic_layout import natural_ref_key, pin_position
from ai_eda.compilers.textbook.glyphs import FONT_MM, TRANSFORMS, cell, mm, pin_cell, symbol_box
from ai_eda.compilers.textbook.labels import stub_label
from ai_eda.compilers.textbook.model import Glyph, Kind, Label, LayoutFailure, Model, NetInfo, Orient, PinGeo, Place, Pt, Role, Sheet, Wiring, _side
from ai_eda.gui import schematic_render as render
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import SymbolDef, SymbolPin

STUB = 2  # cells of the stub that carries a series pair's middle net name
LABEL_PAD_MM = 2.54  # a global label's outline beyond its text


@dataclass(frozen=True)
class Member:
    key: str
    at: tuple[float, float]  # library mm (y up) of the member's origin in the outline's frame
    rot: int  # 0 or 180 in the outline's frame
    pins: tuple[tuple[str, str], ...]  # (member pin, outline pin "1" / "2" or "" for the middle)


@dataclass(frozen=True)
class Compound:
    key: str  # the outline glyph's key: the first member's
    kind: str  # "series" | "parallel"
    members: tuple[Member, Member]
    middle: str | None  # the net between a series pair
    label_at: tuple[float, float] | None  # library mm of the point the middle net's stub leaves from
    label_side: tuple[float, float] | None  # its direction (library mm, y up)


def _resistor(g: Glyph) -> bool:
    return g.kind is Kind.TWO and g.symbol.properties.get("Reference") == "R" and len(set(g.net.values())) == 2


def pairs(model: Model) -> list[tuple[str, str, str, str | None]]:
    """(kind, first key, second key, middle net) of every combination pair, natural order."""
    block_of: dict[str, str] = {}
    ir = model.ir
    if ir is not None and ir.topology:
        for b in ir.topology.blocks:
            for ref in b.component_refs:
                block_of.setdefault(ref, b.id)
    keys = sorted((k for k, g in model.glyphs.items() if _resistor(g) and not any((g.ref, p) in model.stacked for p in g.net)),
                  key=natural_ref_key)
    used: set[str] = set()
    out = []
    for i, a in enumerate(keys):
        if a in used:
            continue
        ga = model.glyphs[a]
        for b in keys[i + 1:]:
            gb = model.glyphs[b]
            if b in used or gb.symbol.lib_id != ga.symbol.lib_id or block_of.get(ga.ref) != block_of.get(gb.ref):
                continue
            na, nb = set(ga.net.values()), set(gb.net.values())
            if na == nb:
                out.append(("parallel", a, b, None))
                used |= {a, b}
                break
            mid = na & nb
            if len(mid) == 1:
                m = next(iter(mid))
                info = model.nets[m]
                if info.role is Role.SIGNAL and {k for k, _ in info.pins} == {a, b} and len(info.pins) == 2 and len(na | nb) == 3:
                    out.append(("series", a, b, m))
                    used |= {a, b}
                    break
    return out


def _lib_xy(p: SymbolPin, at: tuple[float, float], rot: int) -> tuple[float, float]:
    x, y = (p.x, p.y) if rot == 0 else (-p.x, -p.y)
    return round(x + at[0], 4), round(y + at[1], 4)


def _pin_node(number: str, x: float, y: float, angle: float, length: float) -> list:
    font = ["effects", ["font", ["size", 1.27, 1.27]]]
    return ["pin", "passive", "line", ["at", x, y, angle], ["length", length], ["name", sexpr.QStr("~"), font], ["number", sexpr.QStr(number), font]]


def _outline(g: Glyph, name: str, box: tuple[float, float, float, float], pins: list[SymbolPin]) -> SymbolDef:
    """A symbol of a rectangle ``box`` (library mm, y up) and two passive pins, for measurement and placement only."""
    x0, y0, x1, y1 = box
    hidden = ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]]
    node = ["symbol", sexpr.QStr(name), ["pin_numbers", ["hide", "yes"]], ["pin_names", ["offset", 0]],
            ["property", sexpr.QStr("Reference"), sexpr.QStr("R"), ["at", 0, 0, 0], hidden],
            ["property", sexpr.QStr("Value"), sexpr.QStr("R"), ["at", 0, 0, 0], hidden],
            ["symbol", sexpr.QStr(f"{name}_0_1"), ["rectangle", ["start", x0, y0], ["end", x1, y1], ["stroke", ["width", 0], ["type", "default"]],
                                                   ["fill", ["type", "none"]]]],
            ["symbol", sexpr.QStr(f"{name}_1_1"), *(_pin_node(p.number, p.x, p.y, p.angle, p.length) for p in pins)]]
    lib = g.symbol.lib_id.partition(":")[0]
    return SymbolDef(lib_id=f"{lib}:{name}", name=name, node=node, pins=pins, units=[1], is_power=False,
                     properties={"Reference": "R", "Value": "R"})


def _box_lib(g: Glyph, at: tuple[float, float], rot: int) -> tuple[float, float, float, float]:
    """The member's body box in library mm (y up) at ``at`` turned by ``rot``."""
    x0, y0, x1, y1 = symbol_box(g.symbol, (0, 0), Orient(rot))
    return x0 + at[0], -y1 + at[1], x1 + at[0], -y0 + at[1]


def contract(model: Model) -> tuple[Model, dict[str, Compound]]:
    """The model with every combination pair as one outline glyph, and the pairs by outline key (see the module text)."""
    found = pairs(model)
    if not found:
        return model, {}
    glyphs = dict(model.glyphs)
    nets = {n: list(info.pins) for n, info in model.nets.items()}
    compounds: dict[str, Compound] = {}
    aliases = dict(getattr(model, "aliases", {}) or {})
    outlines = dict(model.outlines)
    for kind, a, b, mid in found:
        ga, gb = model.glyphs[a], model.glyphs[b]
        pa = {p.number: p for p in ga.symbol.pins if p.unit in (0, ga.unit)}
        pb = {p.number: p for p in gb.symbol.pins if p.unit in (0, gb.unit)}
        if len(pa) != 2 or len(pb) != 2:
            continue
        values = (ga.component.value, gb.component.value)
        if kind == "series":
            a_out = next(p for p, n in ga.net.items() if n != mid)
            a_mid = next(p for p, n in ga.net.items() if n == mid)
            b_mid = next(p for p, n in gb.net.items() if n == mid)
            b_out = next(p for p, n in gb.net.items() if n != mid)
            # B continues A's line: its middle pin on A's middle pin, turned so it goes on the way A came in
            mx, my = pa[a_mid].x, pa[a_mid].y
            ox, oy = pa[a_out].x, pa[a_out].y
            rot = 0 if (pb[b_mid].x - pb[b_out].x, pb[b_mid].y - pb[b_out].y) == (ox - mx, oy - my) else 180
            bx, by = _lib_xy(pb[b_mid], (0, 0), rot)
            at_b = (round(mx - bx, 4), round(my - by, 4))
            far = _lib_xy(pb[b_out], at_b, rot)
            axis = (mx - ox, my - oy)
            perp = (-axis[1], axis[0])  # the stub side: to the left of the line's direction
            norm = math.hypot(*perp) or 1.0
            side = (round(perp[0] / norm), round(perp[1] / norm))
            text = render._text_width(mid, FONT_MM) + LABEL_PAD_MM
            reach = (STUB * 1.27 + text)
            ba, bb = _box_lib(ga, (0, 0), 0), _box_lib(gb, at_b, rot)
            box = [min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])]
            lx, ly = mx + side[0] * reach, my + side[1] * reach
            box = [min(box[0], lx, mx - abs(side[1]) * 1.27), min(box[1], ly, my - abs(side[0]) * 1.27),
                   max(box[2], lx, mx + abs(side[1]) * 1.27), max(box[3], ly, my + abs(side[0]) * 1.27)]
            p1 = replace(pa[a_out], number="1")
            p2 = SymbolPin("2", "~", pb[b_out].electrical_type, far[0], far[1], (pb[b_out].angle + rot) % 360, pb[b_out].length, 1, False)
            members = (Member(a, (0.0, 0.0), 0, ((a_out, "1"), (a_mid, ""))), Member(b, at_b, rot, ((b_mid, ""), (b_out, "2"))))
            outline_nets = {"1": ga.net[a_out], "2": gb.net[b_out]}
            label = ((mx, my), (side[0], side[1]))
            value = max(values, key=lambda v: render._text_width(v, FONT_MM))
        else:
            (pa1, pa2) = sorted(pa)
            b_on = {gb.net[p]: p for p in pb}
            b1, b2 = b_on[ga.net[pa1]], b_on[ga.net[pa2]]
            rot = 0 if (pb[b1].x, pb[b1].y) == (pa[pa1].x, pa[pa1].y) else 180
            axis = (pa[pa2].x - pa[pa1].x, pa[pa2].y - pa[pa1].y)
            perp = (abs(axis[1]) > 0, abs(axis[0]) > 0)  # B beside A, across the axis
            fields = max(render._text_width(t, FONT_MM) for t in (gb.ref, *values))
            ba0 = _box_lib(ga, (0, 0), 0)
            width = ba0[2] - ba0[0] if perp[0] else ba0[3] - ba0[1]
            step = mm(cell(width + 2.54 + fields + 2.54))
            at_b = (step, 0.0) if perp[0] else (0.0, -step)
            ba, bb = ba0, _box_lib(gb, at_b, rot)
            ends = [(pa[pa1].x, pa[pa1].y), (pa[pa2].x, pa[pa2].y)]
            xs = [ba[0], bb[0], ba[2], bb[2], *(e[0] for e in ends), *(e[0] + at_b[0] for e in ends)]
            ys = [ba[1], bb[1], ba[3], bb[3], *(e[1] for e in ends), *(e[1] + at_b[1] for e in ends)]
            box = [min(xs), min(ys), max(xs), max(ys)]
            p1, p2 = replace(pa[pa1], number="1"), replace(pa[pa2], number="2")
            members = (Member(a, (0.0, 0.0), 0, ((pa1, "1"), (pa2, "2"))), Member(b, at_b, rot, ((b1, "1"), (b2, "2"))))
            outline_nets = {"1": ga.net[pa1], "2": ga.net[pa2]}
            label = None
            value = max(values, key=lambda v: render._text_width(v, FONT_MM))
        sym = _outline(ga, f"{ga.symbol.name}~{kind}~{ga.ref}", tuple(box), [p1, p2])
        geo = tuple(PinGeo(p.number, p.name, p.electrical_type, 1, False, _side(p), round(p.x / 1.27), -round(p.y / 1.27)) for p in (p1, p2))
        glyphs[a] = Glyph(key=a, ref=ga.ref, unit=1, component=ga.component.model_copy(update={"value": value}), symbol=sym,
                          footprint_id=ga.footprint_id, kind=Kind.TWO, pins=geo, net=outline_nets, dc_blocking=False)
        del glyphs[b]
        aliases[gb.ref] = a
        outlines[a] = ga.symbol.lib_id
        for n in list(nets):
            pins = [(k, p) for k, p in nets[n] if k not in (a, b)]
            if len(pins) != len(nets[n]):
                pins += [(a, q) for q, m in outline_nets.items() if m == n]
            nets[n] = pins
        if mid is not None:
            del nets[mid]
        compounds[a] = Compound(a, kind, members, mid, label[0] if label else None, label[1] if label else None)
    infos = {n: NetInfo(n, model.nets[n].role, tuple(sorted(p, key=lambda kp: (natural_ref_key(kp[0]), natural_ref_key(kp[1])))))
             for n, p in nets.items()}
    out = replace(model, glyphs=glyphs, nets=infos, aliases=aliases, outlines=outlines)
    names = ", ".join(f"{model.glyphs[c.members[0].key].ref}{'+' if c.kind == 'series' else '||'}{model.glyphs[c.members[1].key].ref}"
                      for c in compounds.values())
    model.notes.append(f"combinations drawn as one element: {names}")
    return out, compounds


def _absolute(place: Place, x: float, y: float) -> Pt:
    probe = SymbolPin("~", "~", "passive", x, y, 0.0, 0.0, 1, False)
    ax, ay = pin_position(mm(place.x), mm(place.y), place.orient.rot, place.orient.mirror, probe)
    return cell(ax), cell(ay)


def _fit(g: Glyph, targets: dict[str, Pt]) -> tuple[Pt, Orient]:
    """The origin and turn that put each pin of ``g`` on its target cell (the first transform that does)."""
    first, second = sorted(targets)
    for o in TRANSFORMS:
        a = pin_cell(g, first, (0, 0), o)
        origin = (targets[first][0] - a[0], targets[first][1] - a[1])
        if pin_cell(g, second, origin, o) == targets[second]:
            return origin, o
    raise LayoutFailure(f"combination: no turn puts {g.key}'s pins on {targets}")


def expand(original: Model, compounds: dict[str, Compound], sheet: Sheet, wiring: Wiring, labels: list[Label]) -> tuple[Sheet, Wiring, list[Label]]:
    """The sheet with each outline replaced by its two parts (see the module text)."""
    if not compounds:
        return sheet, wiring, labels
    drawings = list(sheet.drawings)
    segments = {n: list(s) for n, s in wiring.segments.items()}
    labels = list(labels)
    of = sheet.drawing_of()
    for key in sorted(compounds, key=natural_ref_key):
        c = compounds[key]
        i = of.get(key)
        if i is None:
            continue
        d = drawings[i]
        dx, dy = sheet.offset[i]
        outline = d.place[key].moved(dx, dy)
        anchors = d.fields.get(key)
        place, fields = dict(d.place), dict(d.fields)
        del place[key]
        fields.pop(key, None)
        centres: dict[str, tuple[float, float]] = {}
        spots: dict[str, Place] = {}
        for m in c.members:
            g = original.glyphs[m.key]
            lib = {p.number: p for p in g.symbol.pins if p.unit in (0, g.unit)}
            targets = {num: _absolute(outline, *_lib_xy(lib[num], m.at, m.rot)) for num, _ in m.pins}
            origin, o = _fit(g, targets)
            spots[m.key] = Place(origin[0], origin[1], o, outline.field_side)
            pts = list(targets.values())
            centres[m.key] = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        if c.kind == "series":  # each part's fields beside its own body, on the outline's field side
            ref_point = None
        else:  # the part on the outline's field side keeps the outline's fields; the other's stand between the two
            d_out = {"R": (1, 0), "L": (-1, 0), "T": (0, -1), "B": (0, 1)}[outline.field_side]
            near = max(c.members, key=lambda m: (centres[m.key][0] * d_out[0] + centres[m.key][1] * d_out[1], m.key == c.members[0].key))
            ref_point = centres[near.key]
        for m in c.members:
            place[m.key] = spots[m.key].moved(-dx, -dy)
            if ref_point is None and anchors is not None:  # the outline's field row, centred on this part's own body
                (rx, ry), (vx, vy) = anchors
                cx, cy = round(centres[m.key][0]) - dx, round(centres[m.key][1]) - dy
                if outline.field_side in ("T", "B"):
                    fields[m.key] = ((cx, ry), (cx, vy))
                else:
                    mid = (ry + vy) // 2
                    fields[m.key] = ((rx, ry - mid + cy), (vx, vy - mid + cy))
            elif anchors is not None:
                sx, sy = round(centres[m.key][0] - ref_point[0]), round(centres[m.key][1] - ref_point[1])
                (rx, ry), (vx, vy) = anchors
                fields[m.key] = ((rx + sx, ry + sy), (vx + sx, vy + sy))
        drawings[i] = replace(d, place=place, fields=fields)
        if c.kind == "series":
            at = _absolute(outline, *c.label_at)
            tip = _absolute(outline, c.label_at[0] + c.label_side[0] * 1.27, c.label_at[1] + c.label_side[1] * 1.27)
            side = {(1, 0): "R", (-1, 0): "L", (0, 1): "B", (0, -1): "T"}[(tip[0] - at[0], tip[1] - at[1])]
            labels.append(stub_label(c.middle, at, side, "name", STUB))
        else:
            a, b = c.members
            ga, gb = original.glyphs[a.key], original.glyphs[b.key]
            for num_a, outer in a.pins:
                num_b = next(nb for nb, ob in b.pins if ob == outer)
                pa = pin_cell(ga, num_a, (spots[a.key].x, spots[a.key].y), spots[a.key].orient)
                pb = pin_cell(gb, num_b, (spots[b.key].x, spots[b.key].y), spots[b.key].orient)
                segments.setdefault(ga.net[num_a], []).append((pa, pb))
    return replace(sheet, drawings=tuple(drawings)), replace(wiring, segments=segments), labels
