"""Glyph geometry: pins under the eight transforms, measured boxes, upright fields and generated power symbols.

Boxes are millimetres in the schematic frame, measured by the preview renderer
(``gui.schematic_render``): a body box holds the drawn graphics, the pins and
their name / number text; a field or label box its text outline.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ai_eda.compilers.schematic_layout import GRID_MM, pin_body_direction, pin_position, snap
from ai_eda.compilers.textbook.model import BoxMM, Drawing, Generated, Glyph, Label, LayoutFailure, Model, Orient, PinGeo, Place, Pt, Sheet, Side
from ai_eda.errors import CompileError
from ai_eda.gui import schematic_render as render
from ai_eda.ir import LibraryRef
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary, SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import Node, Q, S

#: every orientation a glyph may take: four rotations, unmirrored then mirrored about the vertical axis (fixed order)
TRANSFORMS: tuple[Orient, ...] = tuple(Orient(rot, mirror) for mirror in (None, "y") for rot in (0, 90, 180, 270))
FONT_MM = 1.27
#: generated symbols by kind: (library, name)
GENERATED: dict[str, tuple[str, str]] = {"GND": ("power", "GND"), "VCC": ("power", "VCC"), "NEG": ("power", "VCC"), "FLAG": ("power", "PWR_FLAG")}
_SIDE_OF_BODY: dict[tuple[float, float], Side] = {(1.0, 0.0): "L", (-1.0, 0.0): "R", (0.0, 1.0): "T", (0.0, -1.0): "B"}


class Measured(NamedTuple):
    body: BoxMM
    fields: tuple[BoxMM, ...]
    field_nodes: tuple[Node, ...]


def mm(v: int) -> float:
    return snap(v * GRID_MM)


def cell(v: float) -> int:
    return round(v / GRID_MM)


def _lib_pin(g: Glyph, pin: str | PinGeo) -> SymbolPin:
    return g.lib_pin(pin.number if isinstance(pin, PinGeo) else pin)


def pin_cell(g: Glyph, pin: str | PinGeo, at: Pt, o: Orient) -> Pt:
    """The pin's connection point in cells for the glyph at ``at`` turned by ``o``; CompileError for a library pin off the
    1.27 mm grid (KiCad connects by exact coordinates, so such a pin could never meet a wire on the grid)."""
    lp = _lib_pin(g, pin)
    actual = pin_position(mm(at[0]), mm(at[1]), o.rot, o.mirror, lp)
    point = (cell(actual[0]), cell(actual[1]))
    if any(abs(mm(c) - v) > 1e-6 for c, v in zip(point, actual)):
        raise CompileError(f"textbook: library pin {g.ref}.{lp.number} of {g.symbol.lib_id!r} at {actual} is off the 1.27 mm grid")
    return point


def pin_side(g: Glyph, pin: str | PinGeo, o: Orient) -> Side:
    """The side of the body the pin leaves from once the glyph is turned by ``o``."""
    body = pin_body_direction(o.rot, o.mirror, _lib_pin(g, pin).angle)
    return _SIDE_OF_BODY[(round(body[0]), round(body[1]))]


def outward(side: Side) -> Pt:
    return {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}[side]


def orient_two(g: Glyph, first_net: str, vertical: bool) -> Orient:
    """The rotation that puts the pin on ``first_net`` above (``vertical``) or left of the other pin: the first of 0, 90,
    180, 270 that does (the 0.9 two-terminal rule). LayoutFailure when no rotation does."""
    numbers = [p.number for p in g.symbol.pins if p.unit in (0, g.unit)]
    first = next((n for n in numbers if g.net.get(n) == first_net), None)
    if first is None or len(numbers) < 2:
        raise LayoutFailure(f"{g.key} has no pin on {first_net}")
    other = next(n for n in numbers if n != first)
    for rot in (0, 90, 180, 270):
        o = Orient(rot)
        a, b = pin_cell(g, first, (0, 0), o), pin_cell(g, other, (0, 0), o)
        if (a[0] == b[0] and a[1] < b[1]) if vertical else (a[1] == b[1] and a[0] < b[0]):
            return o
    raise LayoutFailure(f"{g.key} does not have opposed two-terminal pins")


def _instance(lib_id: str, x: float, y: float, o: Orient, unit: int) -> Node:
    return S("symbol", S("lib_id", Q(lib_id)), S("at", x, y, o.rot), S("mirror", o.mirror) if o.mirror else None, S("unit", unit))


def _box(out: render._Out) -> BoxMM | None:
    return (min(out.xs), min(out.ys), max(out.xs), max(out.ys)) if out.xs else None


def symbol_box(symbol: SymbolDef, at: Pt, o: Orient, unit: int = 1) -> BoxMM:
    """Body box (graphics, pins, pin text) of ``symbol`` at ``at`` turned by ``o``."""
    out = render._Out()
    render._draw_symbol(out, _instance(symbol.lib_id, mm(at[0]), mm(at[1]), o, unit), {symbol.lib_id: symbol.lib_symbols_entry()})
    box = _box(out)
    x, y = mm(at[0]), mm(at[1])
    return box if box is not None else (x, y, x, y)


def body_box(g: Glyph, place: Place) -> BoxMM:
    return symbol_box(g.symbol, (place.x, place.y), place.orient, g.unit)


def field_plan(g: Glyph, place: Place, side: Side, ref_text: str | None = None, around: BoxMM | None = None) -> tuple[Node, Node]:
    """Reference over Value, upright, on the grid, on ``side`` of the body (or of ``around``, a larger box such as the body
    plus its pin labels): beside it centred on the origin row (L / R), or above / below it centred on the origin column."""
    from ai_eda.compilers.schematic import _property

    x0, y0, x1, y1 = around if around is not None else body_box(g, place)
    px, py = mm(place.x), mm(place.y)
    nodes = []
    for key, text in (("Reference", ref_text or g.ref), ("Value", g.component.value)):
        width = render._text_width(g.shown_ref if key == "Reference" and ref_text is None else text, FONT_MM)
        first = key == "Reference"
        if side == "R":
            x, y = x1 + 2.54 + width / 2, py + (-1.27 if first else 1.27)
        elif side == "L":
            x, y = x0 - 2.54 - width / 2, py + (-1.27 if first else 1.27)
        elif side == "T":
            x, y = px, y0 - (5.08 if first else 2.54)
        else:
            x, y = px, y1 + (2.54 if first else 5.08)
        nodes.append(_property(key, text, mm(cell(x)), mm(cell(y)), rot=place.orient.rot % 180))
    return nodes[0], nodes[1]


def field_nodes(g: Glyph, place: Place, anchors: tuple[Pt, Pt] | None = None, ref_text: str | None = None) -> tuple[Node, Node]:
    """The Reference and Value nodes as written: at ``anchors`` (cells) when a recipe fixed them, else :func:`field_plan` on
    ``place.field_side``."""
    from ai_eda.compilers.schematic import _property

    if anchors is None:
        return field_plan(g, place, place.field_side, ref_text)
    rot = place.orient.rot % 180
    (rx, ry), (vx, vy) = anchors
    return (_property("Reference", ref_text or g.ref, mm(rx), mm(ry), rot=rot), _property("Value", g.component.value, mm(vx), mm(vy), rot=rot))


def field_anchors(nodes: tuple[Node, Node]) -> tuple[Pt, Pt]:
    """The (Reference, Value) anchor cells of two field nodes."""
    def at(node: Node) -> Pt:
        a = sexpr.find(node, "at")
        return cell(sexpr.to_float(a[1])), cell(sexpr.to_float(a[2]))
    return at(nodes[0]), at(nodes[1])


def field_boxes(nodes: tuple[Node, ...] | list[Node], at: Pt, o: Orient) -> tuple[BoxMM, ...]:
    inst = render._Instance(mm(at[0]), mm(at[1]), o.rot, o.mirror)
    boxes = []
    for node in nodes:
        out = render._Out()
        render._draw_fields(out, ["symbol", node], inst)
        if (box := _box(out)) is not None:
            boxes.append(box)
    return tuple(boxes)


def measure(g: Glyph, place: Place, ref_text: str | None = None, around: BoxMM | None = None, anchors: tuple[Pt, Pt] | None = None) -> Measured:
    """The body box, the field boxes and the field nodes of ``g`` at ``place``: fields at ``anchors`` when given, else on
    ``place.field_side`` of the body (or of ``around``)."""
    body = body_box(g, place)
    if anchors is not None:
        nodes = field_nodes(g, place, anchors, ref_text)
    else:
        nodes = field_plan(g, place, place.field_side, ref_text, around if around is not None else body)
    return Measured(body, field_boxes(_as_shown(g, nodes), (place.x, place.y), place.orient), nodes)


def _as_shown(g: Glyph, nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    """The field nodes as KiCad draws them, for measuring: a unit of a multi-unit symbol shows its unit letter after the
    Reference (``U1A``); the file keeps the ref."""
    if g.shown_ref == g.ref:
        return nodes
    return tuple([*n[:2], Q(g.shown_ref), *n[3:]] if str(n[1]) == "Reference" and str(n[2]) == g.ref else n for n in nodes)


def generated(library: KicadLibrary, kind: str, net: str, at: Pt, orient: Orient, ref: str) -> Generated:
    """A ``#PWR`` / ``#FLG`` glyph: GND, VCC and NEG carry Value = the net name (the power symbol names its net), FLAG is
    ``power:PWR_FLAG``. NEG uses the library power symbol named like the net when there is one. LayoutFailure when the
    installed libraries lack the symbol."""
    lib, name = GENERATED[kind]
    if kind == "NEG" and library.resolve_symbol(LibraryRef(library="power", name=net)).verified:
        name = net
    libref = library.resolve_symbol(LibraryRef(library=lib, name=name))
    if not libref.verified:
        raise LayoutFailure(f"{lib}:{name} is not in the installed KiCad libraries")
    symbol = library.load_symbol(libref)
    value = (symbol.properties.get("Value") or name) if kind == "FLAG" else net
    return Generated(ref=ref, lib=symbol.lib_id, net=net, at=at, orient=orient, symbol=symbol, value=value, kind=kind)  # type: ignore[arg-type]


def generated_pin(gen: Generated) -> Pt:
    p = gen.symbol.pins[0]
    actual = pin_position(mm(gen.at[0]), mm(gen.at[1]), gen.orient.rot, gen.orient.mirror, p)
    return cell(actual[0]), cell(actual[1])


def generated_side(gen: Generated) -> Side:
    body = pin_body_direction(gen.orient.rot, gen.orient.mirror, gen.symbol.pins[0].angle)
    return _SIDE_OF_BODY[(round(body[0]), round(body[1]))]


def generated_fields(gen: Generated) -> tuple[Node, Node]:
    """Reference hidden; Value shown for a supply glyph (VCC / NEG), on the far side of the body from the pin."""
    from ai_eda.compilers.schematic import _property

    box = symbol_box(gen.symbol, gen.at, gen.orient)
    x, y = mm(gen.at[0]), mm(gen.at[1])
    above = (box[1] + box[3]) / 2 <= y
    vy = mm(cell(box[1] - 2.54)) if above else mm(cell(box[3] + 2.54))
    return (_property("Reference", gen.ref, x, vy, hide=True),
            _property("Value", gen.value, x, vy, hide=gen.kind not in ("VCC", "NEG")))


def generated_boxes(gen: Generated) -> tuple[BoxMM, tuple[BoxMM, ...]]:
    return symbol_box(gen.symbol, gen.at, gen.orient), field_boxes(generated_fields(gen), gen.at, gen.orient)


def label_node(label: Label, uuid: str = "00000000-0000-0000-0000-000000000000") -> Node:
    from ai_eda.compilers.schematic import _global_label

    return _global_label(label.net, (mm(label.at[0]), mm(label.at[1])), label.angle, label.justify, uuid)


#: advance widths of KiCad's stroke font in 1/21 of the text height, by character (printable ASCII; anything else is
#: counted as the widest, 28), measured from kicad-cli 10.0.6 SVG exports of global labels
_STROKE_WIDTHS: dict[int, str] = {
    8: "`", 10: "!',.:;Iij", 11: "l", 12: "^ft", 13: "r", 14: "()[]{}", 15: "~", 16: "*JT_vy ", 17: "Lksxz", 18: "?AFVYce",
    19: "Eabdghnopqu", 20: "$0123456789SXZ|", 21: "#BCDGKPR", 22: "/HNOQUw", 24: "%MW", 26: "&+-<=>", 27: "@", 28: "m",
}
_STROKE_WIDTH: dict[str, int] = {ch: w for w, chars in _STROKE_WIDTHS.items() for ch in chars}
#: KiCad's passive global label outline, per text height: the text advance gains 0.16 height and the outline 0.8905
#: height more along the label (1.27 mm text: 0.2032 mm and 1.1309 mm); across, it is 2.5408 mm; its stroke is 0.1524 mm
_LABEL_TEXT_EXTRA, _LABEL_LENGTH_EXTRA = 0.16, 1.1309 / 1.27
_LABEL_STROKE_MM = 0.1524
_LABEL_DIRECTION: dict[int, Pt] = {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}


def label_box(label: Label) -> BoxMM:
    """The box KiCad draws for the global label (shape passive, text 1.27 mm): along the label's angle, its outline from
    the anchor with the stroke at both ends, text widths from KiCad's stroke-font advances, so a wire kept out of the box
    never runs inside the drawn label. Across, the box is one cell (2.54 mm) wide: the outline is 2.5408 mm plus its
    stroke, and KiCad draws labels on neighbouring pins of a 2.54 mm row edge to edge, which is not an overlap."""
    size = FONT_MM
    text = sum(_STROKE_WIDTH.get(ch, 28) for ch in label.net) / 21 * size + _LABEL_TEXT_EXTRA * size
    length = text + _LABEL_LENGTH_EXTRA * size
    half = GRID_MM
    pad = _LABEL_STROKE_MM / 2
    ex, ey = _LABEL_DIRECTION.get(label.angle % 360, (1, 0))
    ax, ay = mm(label.at[0]), mm(label.at[1])
    xs = (ax + ex * -pad, ax + ex * (length + pad))
    ys = (ay + ey * -pad, ay + ey * (length + pad))
    if ex:
        return min(xs), ay - half, max(xs), ay + half
    return ax - half, min(ys), ax + half, max(ys)


def overlaps(boxes: list[BoxMM] | tuple[BoxMM, ...]) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j``, of boxes that share a positive area (touching edges do not count). Never raises."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0], i))
    out: list[tuple[int, int]] = []
    active: list[int] = []
    for i in order:
        x0 = boxes[i][0]
        active = [j for j in active if boxes[j][2] - x0 > 1e-9]
        for j in active:
            a, b = boxes[i], boxes[j]
            if min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9:
                out.append((min(i, j), max(i, j)))
        active.append(i)
    return sorted(out)


def union(boxes: list[BoxMM]) -> BoxMM:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def to_cells(box: BoxMM) -> tuple[int, int, int, int]:
    """The smallest cell box that holds ``box``."""
    return (math.floor(box[0] / GRID_MM + 1e-9), math.floor(box[1] / GRID_MM + 1e-9),
            math.ceil(box[2] / GRID_MM - 1e-9), math.ceil(box[3] / GRID_MM - 1e-9))


def _boxes(model: Model, places: dict[str, Place], anchors: dict[str, tuple[Pt, Pt]], gens: list[Generated], labels: list[Label]) -> list[tuple[str, BoxMM]]:
    out: list[tuple[str, BoxMM]] = []
    for key in sorted(places, key=_natural):
        m = measure(model.glyphs[key], places[key], anchors=anchors.get(key))
        out.append((f"body:{key}", m.body))
        out += [(f"field:{key}", b) for b in m.fields]
    for gen in sorted(gens, key=lambda gen: _natural(gen.ref)):
        body, fields = generated_boxes(gen)
        out.append((f"glyph:{gen.ref}", body))
        out += [(f"gfield:{gen.ref}", b) for b in fields]
    for lab in labels:
        out.append((f"label:{lab.net}@{lab.at[0]},{lab.at[1]}", label_box(lab)))
    return out


def drawing_boxes(model: Model, drawing: Drawing) -> list[tuple[str, BoxMM]]:
    """Every measured box of a drawing, block-relative, tagged ``body:KEY``, ``field:KEY``, ``glyph:REF`` (a generated
    symbol's body), ``gfield:REF``, ``label:NET@x,y``."""
    return _boxes(model, drawing.place, drawing.fields, list(drawing.glyphs), list(drawing.labels))


def glyph_boxes(model: Model, sheet: Sheet) -> list[tuple[str, BoxMM]]:
    """:func:`drawing_boxes` of the whole sheet, absolute: every drawing's and the sheet's own glyphs, and the drawings' labels."""
    return _boxes(model, sheet.places(), sheet.fields(), sheet.generated(), sheet.labels())


def drawing_box(model: Model, drawing: Drawing) -> tuple[int, int, int, int]:
    """The cell box of a drawing: its measured boxes and its hint wires."""
    boxes = [b for _, b in drawing_boxes(model, drawing)]
    boxes += [(mm(min(a[0], b[0])), mm(min(a[1], b[1])), mm(max(a[0], b[0])), mm(max(a[1], b[1]))) for segs in drawing.hints.values() for a, b in segs]
    return to_cells(union(boxes)) if boxes else (0, 0, 0, 0)


def _natural(key: str) -> tuple:
    from ai_eda.compilers.schematic_layout import natural_ref_key

    return natural_ref_key(key)
