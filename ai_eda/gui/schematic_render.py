"""An SVG preview of a ``.kicad_sch`` file, drawn from the file's own ``lib_symbols``.

Invariant: this is **a preview of the file the compiler wrote, drawn from its
own lib_symbols; not a KiCad render and no ERC is implied**. It reads one
parsed schematic tree and returns text: it computes no status, touches no
library on disk, writes nothing, and the same tree always gives the same
bytes (document order, fixed-precision numbers, no randomness, no clock, no
path). Every text that reaches the markup - references, values, net names,
pin names, lib ids - is XML-escaped (:func:`esc`), so a value such as
``<b>&"x"`` is shown as typed and cannot inject markup.

Geometry follows the file exactly:

* ``(paper ...)`` sets the viewBox (:data:`PAPER_SIZES_MM`, KiCad's page
  table, landscape unless ``portrait``; ``"User" W H`` as written); an
  unknown size falls back to the bounding box of the drawn content with a
  :data:`CONTENT_MARGIN_MM` margin. That box (``x y w h``, mm) is always on
  the root as ``data-content-box``, so a viewer can fit the drawing rather
  than the sheet.
* Every ``(symbol (lib_id ..) (at x y rot) [(mirror x|y)] (unit n) ...)``
  instance is drawn from the ``lib_symbols`` entry of that name (its
  ``lib_name`` first when present): the unit sub-symbols ``Name_<unit>_<style>``
  whose unit is 0 (common) or the instance's unit, body style 1 (and 0,
  items common to every body style - the pin set the compiler places, see
  :func:`ai_eda.tools.kicad.library.KicadLibrary.load_symbol`). The library
  frame is Y-up, the sheet Y-down; every library point goes through
  :func:`ai_eda.compilers.schematic_layout.transform_offset` - the compiler's
  own transform (rotate, then mirror in sheet coordinates: KiCad's
  ``SCH_SYMBOL`` transform) - and every pin end through
  :func:`~ai_eda.compilers.schematic_layout.pin_position` /
  :func:`~ai_eda.compilers.schematic_layout.pin_body_direction`, so there is
  one convention only and the drawn pin ends are the points the compiler
  connected its wire stubs to.
* Symbol graphics: ``rectangle``, ``polyline``, ``bezier`` (flattened into
  :data:`BEZIER_SEGMENTS` segments), ``circle``, ``arc`` (three-point arc ->
  an SVG arc on the circumcircle, sweep and large-arc flags from start / mid /
  end) and ``text`` (angle in tenths of a degree, as libraries store it).
  Fill ``background`` is the light body fill, ``outline`` is filled with the
  outline colour (KiCad's meaning: the solid arrow head of a transistor),
  ``color`` its own colour, ``none`` no fill. A stroke width of 0 means the
  default line (:data:`LINE_MM`).
* Pins: a line of ``length`` from the connection point toward the body; the
  number beside it and the name at the inner end (inside the body when the
  symbol's ``pin_names`` offset is > 0, above the line otherwise, the number
  then below), unless the symbol says ``(pin_names hide)`` /
  ``(pin_numbers hide)`` (either spelling: bare ``hide`` or ``(hide yes)``) or
  the pin itself is hidden - a hidden pin is drawn dashed without texts, so a
  wire stub to it does not look dangling. Every pin is a straight line: the
  preview does not draw clock wedges, inversion bubbles or other
  ``graphic_style`` decorations.
* Texts keep KiCad's readability rule: a text is drawn at 0 or 90 degrees
  (reading left-to-right or bottom-to-top), and a field of a rotated or
  mirrored symbol is re-justified so that its box lies where the symbol
  transform puts it (KiCad's ``SCH_FIELD`` draw rotation and justification
  flip). Visible properties are drawn at their own ``(at ..)`` with their
  ``justify``; hidden ones are skipped. KiCad's text markup is reduced to
  an overline (``~{..}``) and sub / superscripts (``_{..}`` / ``^{..}``).
* Sheet items: ``wire`` (a line), ``bus``, ``global_label`` (the text inside
  the label outline of its ``shape``, pointing along its angle), ``label``
  (the text above the wire), ``text``, ``no_connect`` (an X of 1.27 mm) and
  ``junction`` (a filled dot). Sheets, hierarchical labels, bus entries,
  images and text boxes are not drawn.

Classes the page and the tests rely on: one ``<g class="symbol">`` per
symbol instance (``data-ref``, ``data-lib-id``, ``data-unit``;
``data-missing="yes"`` when ``lib_symbols`` lacks the entry, drawn as a
dashed box), one ``<line class="pin">`` per pin (``x1``/``y1`` = the
connection point, ``x2``/``y2`` = the inner end, ``data-number``,
``data-hidden="yes"`` for a hidden pin), ``<text class="pin-name">`` /
``<text class="pin-number">``, ``<text class="field">`` (``data-field``),
one ``<line class="wire">`` per wire, ``<g class="global-label">``
(``data-net``), ``<g class="label">``, ``<circle class="junction">`` and
``<path class="no-connect">``.

A malformed tree (not a ``kicad_sch``, an instance without ``(at x y)``, a
rotation other than 0/90/180/270, a non-finite coordinate) is a
:class:`ValueError`, never a partial picture.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eda.compilers.schematic_layout import Vec, pin_body_direction, pin_position, snap, transform_offset
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import SymbolPin
from ai_eda.tools.kicad.sexpr import Node

__all__ = [
    "BEZIER_SEGMENTS",
    "CONTENT_MARGIN_MM",
    "FONT_SCALE",
    "LABEL_COLOUR",
    "LINE_MM",
    "NO_CONNECT_COLOUR",
    "PAPER_SIZES_MM",
    "PIN_COLOUR",
    "PIN_NAME_COLOUR",
    "REFERENCE_COLOUR",
    "SYMBOL_FILL",
    "SYMBOL_STROKE",
    "VALUE_COLOUR",
    "WIRE_COLOUR",
    "arc_path",
    "esc",
    "paper_size",
    "render_kicad_sch",
    "render_kicad_sch_file",
]

#: KiCad's page table (mm, landscape): ISO A5-A0, ANSI A-E, US Letter / Legal / Ledger
PAPER_SIZES_MM: dict[str, tuple[float, float]] = {
    "A5": (210.0, 148.0),
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
    "A2": (594.0, 420.0),
    "A1": (841.0, 594.0),
    "A0": (1189.0, 841.0),
    "A": (279.4, 215.9),
    "B": (431.8, 279.4),
    "C": (558.8, 431.8),
    "D": (863.6, 558.8),
    "E": (1117.6, 863.6),
    "USLetter": (279.4, 215.9),
    "USLegal": (355.6, 215.9),
    "USLedger": (431.8, 279.4),
}
#: margin around the drawn content when the paper size is unknown
CONTENT_MARGIN_MM = 10.0
#: segments a cubic bezier is flattened into
BEZIER_SEGMENTS = 16

SYMBOL_STROKE = "#8b0000"
SYMBOL_FILL = "#fff8dc"
PIN_COLOUR = "#8b0000"
PIN_NAME_COLOUR = "#006464"
WIRE_COLOUR = "#006400"
BUS_COLOUR = "#00008b"
LABEL_COLOUR = "#008080"
REFERENCE_COLOUR = "#0b0b0b"
VALUE_COLOUR = "#52514e"
NO_CONNECT_COLOUR = "#0000c8"
PAPER_COLOUR = "#ffffff"
PAPER_EDGE = "#b9b6ab"
#: KiCad's default line width (6 mil), used where a stroke says width 0
LINE_MM = 0.15
BUS_MM = 0.3
#: default font height (mm) when an item carries no (effects (font (size ..)))
DEFAULT_TEXT_MM = 1.27
#: SVG font-size per KiCad text height: KiCad's size is about the cap height, a sans font's cap height is ~0.72 em
FONT_SCALE = 1.3
#: gap between a pin line and its number / name text
PIN_TEXT_MARGIN_MM = 0.254
#: KiCad's default pin name offset (20 mil) when a symbol has no (pin_names (offset ..))
DEFAULT_PIN_NAME_OFFSET_MM = 0.508
#: KiCad's default junction dot (36 mil) when (diameter 0)
JUNCTION_MM = 0.9144
#: half the size of the no-connect X (1.27 mm across)
NO_CONNECT_HALF_MM = 0.635
#: global label box expansion as a fraction of the text height (KiCad's default label margin ratio)
LABEL_MARGIN_RATIO = 0.375
#: local label text offset above the wire as a fraction of the text height (KiCad's default)
LABEL_OFFSET_RATIO = 0.15
#: global label text lifted off the outline's centre line (fraction of the text height) so descenders stay inside the outline,
#: like KiCad centring label text on the "E" centre line
LABEL_TEXT_LIFT_RATIO = 0.1
#: line pitch of multi-line text as a multiple of the text height
LINE_PITCH_RATIO = 1.6
_FONT_FAMILY = "'DejaVu Sans', 'Noto Sans', 'Liberation Sans', Arial, 'Noto Sans CJK KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"
_DASHES = {"dash": "1 0.5", "dot": "0.2 0.4", "dash_dot": "1 0.4 0.2 0.4", "dash_dot_dot": "1 0.4 0.2 0.4 0.2 0.4"}
_UNIT_NAME_RE = re.compile(r"^(?P<base>.+)_(?P<unit>\d+)_(?P<style>\d+)$")
_INVALID_XML_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
_MARKUP_RE = re.compile(r"([~_^])\{([^{}]*)\}")
_EPS = 1e-9


# --------------------------------------------------------------------------- text and numbers


def esc(text: object) -> str:
    """XML-escape ``text`` for element content and attribute values; characters XML 1.0 cannot carry are dropped."""
    s = _INVALID_XML_RE.sub("", str(text))
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


def _n(value: float) -> str:
    """A coordinate in mm: at most 4 decimals (KiCad's schematic precision), no trailing zeros, never ``-0``; non-finite -> ValueError."""
    return sexpr.fmt_num(float(value), 4)


def _plain(text: str) -> str:
    """``text`` with KiCad's markup reduced to its characters (for width estimates)."""
    return _MARKUP_RE.sub(lambda m: m.group(2), text)


def _markup(text: str) -> str:
    """Escaped SVG text content; ``~{x}`` becomes an overlined run, ``_{x}`` / ``^{x}`` sub / superscripts."""
    out: list[str] = []
    pos = 0
    for m in _MARKUP_RE.finditer(text):
        out.append(esc(text[pos : m.start()]))
        kind, body = m.group(1), esc(m.group(2))
        if kind == "~":
            out.append(f'<tspan text-decoration="overline">{body}</tspan>')
        else:
            shift = "sub" if kind == "_" else "super"
            out.append(f'<tspan baseline-shift="{shift}" font-size="70%">{body}</tspan>')
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def _char_em(ch: str) -> float:
    if ord(ch) > 0x2E7F:
        return 1.0
    if ch in " iljI.,:;'|!`":
        return 0.3
    if ch in "mwMW":
        return 0.9
    if ch.isupper():
        return 0.7
    if ch.isdigit():
        return 0.62
    return 0.58


def _text_width(text: str, size: float) -> float:
    """Estimated drawn width (mm) of one line of ``text`` at KiCad height ``size`` - for label outlines and the content box."""
    return sum(_char_em(ch) for ch in _plain(text)) * size * FONT_SCALE


# --------------------------------------------------------------------------- tree helpers


def _num(atom: Any, what: str) -> float:
    try:
        value = sexpr.to_float(atom)
    except sexpr.SExprError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{what}: {atom!r} is not a finite number")
    return value


def _xy(node: Node | None, what: str) -> Vec:
    if node is None or len(node) < 3 or isinstance(node[1], list) or isinstance(node[2], list):
        raise ValueError(f"{what}: expected ({(node or ['?'])[0]} x y)")
    return _num(node[1], what), _num(node[2], what)


def _is_yes(node: Node | None, name: str) -> bool:
    """``(name yes)`` (KiCad 8+; ``True`` in a tree built in Python, as :func:`~ai_eda.tools.kicad.sexpr.S` writes it) or a bare ``name`` atom (KiCad 6/7)."""
    if node is None:
        return False
    value = sexpr.get(node, name)
    # a flag is a *bare* atom: a quoted value that happens to read "hide" (a property's value) is text, not a flag
    return value is True or value == "yes" or any(a == name and not isinstance(a, sexpr.QStr) for a in sexpr.args(node))


def _hidden(node: Node | None) -> bool:
    """``(hide yes)`` or a bare ``hide``."""
    return _is_yes(node, "hide")


@dataclass(frozen=True, slots=True)
class _Font:
    size: float = DEFAULT_TEXT_MM
    bold: bool = False
    italic: bool = False
    h: str = "center"  # left | right | center (the item's own justify)
    v: str = "center"  # top | bottom | center
    hidden: bool = False


def _font(effects: Node | None) -> _Font:
    if effects is None:
        return _Font()
    font = sexpr.find(effects, "font")
    size = DEFAULT_TEXT_MM
    bold = italic = False
    if font is not None:
        size_node = sexpr.find(font, "size")
        if size_node is not None and len(size_node) >= 2:
            size = _num(size_node[1], "font size")
        bold, italic = _is_yes(font, "bold"), _is_yes(font, "italic")
    justify = sexpr.find(effects, "justify")
    tokens = [str(t) for t in sexpr.args(justify)] if justify is not None else []
    h = "left" if "left" in tokens else "right" if "right" in tokens else "center"
    v = "top" if "top" in tokens else "bottom" if "bottom" in tokens else "center"
    return _Font(size=size, bold=bold, italic=italic, h=h, v=v, hidden=_hidden(effects))


# --------------------------------------------------------------------------- output


@dataclass(slots=True)
class _Out:
    """Markup per layer plus the bounding box of everything drawn (for an unknown paper size)."""

    body: list[str] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    def grow(self, *points: Vec) -> None:
        for x, y in points:
            self.xs.append(x)
            self.ys.append(y)


def _frame(theta: float) -> tuple[Vec, Vec]:
    """Reading direction and glyph-down direction (sheet frame, Y down) of text drawn at ``theta`` degrees (CCW on screen)."""
    rad = math.radians(theta)
    c, s = snap(math.cos(rad), 6), snap(math.sin(rad), 6)
    return (c, -s), (s, c)


def _readable(reading: Vec) -> float:
    """KiCad's readability rule: the draw angle in (-90, 90] for a text whose reading direction is ``reading`` (0 = left-to-right, 90 = bottom-to-top)."""
    theta = math.degrees(math.atan2(-reading[1], reading[0]))
    theta = ((theta + 90.0) % 180.0) - 90.0
    if theta <= -90.0 + _EPS:
        theta = 90.0
    return snap(theta, 6)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _oriented(local_angle: float, h: str, v: str, to_sheet: Callable[[Vec], Vec]) -> tuple[float, str, str]:
    """Draw angle and effective (h, v) justification of a text defined in a local frame mapped by ``to_sheet``.

    The text's box is carried by the transform (KiCad transforms a field's
    bounding box with its symbol), then the text is redrawn readable and
    justified so that it fills that box.
    """
    rad = math.radians(local_angle)
    r_loc: Vec = (math.cos(rad), -math.sin(rad))
    d_loc: Vec = (math.sin(rad), math.cos(rad))
    theta = _readable(to_sheet(r_loc))
    r_draw, d_draw = _frame(theta)
    hs = {"left": 1.0, "right": -1.0}.get(h, 0.0)
    vs = {"top": 1.0, "bottom": -1.0}.get(v, 0.0)
    ext_r = _dot(to_sheet((r_loc[0] * hs, r_loc[1] * hs)), r_draw)
    ext_d = _dot(to_sheet((d_loc[0] * vs, d_loc[1] * vs)), d_draw)
    h_eff = "start" if ext_r > _EPS else "end" if ext_r < -_EPS else "middle"
    v_eff = "top" if ext_d > _EPS else "bottom" if ext_d < -_EPS else "center"
    return theta, h_eff, v_eff


def _identity(v: Vec) -> Vec:
    return v


def _text(
    out: _Out,
    text: str,
    anchor: Vec,
    *,
    size: float,
    theta: float,
    h: str,
    v: str,
    colour: str,
    cls: str,
    font: _Font | None = None,
    attrs: str = "",
) -> None:
    """One text item at ``anchor``: ``h`` = start | middle | end along the reading direction, ``v`` = top | center | bottom."""
    lines = text.split("\n")
    r, d = _frame(theta)
    pitch = size * LINE_PITCH_RATIO
    block = size + (len(lines) - 1) * pitch
    first = {"top": size, "center": size - block / 2.0, "bottom": size - block}[v]
    weight = ' font-weight="bold"' if font is not None and font.bold else ""
    style = ' font-style="italic"' if font is not None and font.italic else ""
    for i, line in enumerate(lines):
        if not line:
            continue
        off = first + i * pitch
        bx, by = anchor[0] + d[0] * off, anchor[1] + d[1] * off
        rotate = f' transform="rotate({_n(-theta)} {_n(bx)} {_n(by)})"' if theta else ""
        out.body.append(
            f'<text class="{cls}"{attrs} x="{_n(bx)}" y="{_n(by)}" font-size="{_n(size * FONT_SCALE)}" '
            f'text-anchor="{h}" fill="{colour}"{weight}{style}{rotate}>{_markup(line)}</text>'
        )
        width = _text_width(line, size)
        lo = {"start": 0.0, "middle": -width / 2.0, "end": -width}[h]
        for along in (lo, lo + width):
            for across in (off - size, off + 0.25 * size):
                out.grow((anchor[0] + r[0] * along + d[0] * across, anchor[1] + r[1] * along + d[1] * across))


# --------------------------------------------------------------------------- symbol instances


@dataclass(frozen=True, slots=True)
class _Instance:
    """A placed symbol: ``(at x y rotation)`` and ``(mirror ..)`` exactly as the file states them."""

    x: float
    y: float
    rotation: int
    mirror: str | None

    def point(self, lx: float, ly: float) -> Vec:
        """Library point (Y up) -> sheet point: the compiler's transform, then the compiler's rounding."""
        dx, dy = transform_offset(lx, ly, self.rotation, self.mirror)
        return snap(self.x + dx), snap(self.y + dy)

    def vector(self, v: Vec) -> Vec:
        """A local direction in the sheet's Y-down sense, carried by the symbol transform."""
        return transform_offset(v[0], -v[1], self.rotation, self.mirror)


def _rotation(atom: Any) -> int:
    value = _num(atom, "symbol rotation")
    rot = int(round(value)) % 360
    if abs(value - round(value)) > _EPS or rot not in (0, 90, 180, 270):
        raise ValueError(f"symbol rotation must be 0, 90, 180 or 270, got {atom!r}")
    return rot


def _stroke_attrs(item: Node, colour: str) -> str:
    stroke = sexpr.find(item, "stroke")
    width = 0.0
    dash = ""
    if stroke is not None:
        w = sexpr.get(stroke, "width")
        if w is not None:
            width = _num(w, "stroke width")
        dash_type = sexpr.get(stroke, "type")
        if dash_type is not None and str(dash_type) in _DASHES:
            dash = f' stroke-dasharray="{_DASHES[str(dash_type)]}"'
    if width <= 0:
        width = LINE_MM
    return f'stroke="{colour}" stroke-width="{_n(width)}"{dash}'


def _fill_attr(item: Node) -> str:
    fill = sexpr.find(item, "fill")
    kind = str(sexpr.get(fill, "type", 1, "none")) if fill is not None else "none"
    if kind == "background":
        return f'fill="{SYMBOL_FILL}"'
    if kind == "outline":
        return f'fill="{SYMBOL_STROKE}"'
    if kind == "color" and fill is not None:
        colour = sexpr.find(fill, "color")
        if colour is not None and len(colour) >= 5:
            r, g, b = (max(0, min(255, int(_num(colour[i], "fill colour")))) for i in (1, 2, 3))
            alpha = max(0.0, min(1.0, _num(colour[4], "fill colour")))
            if alpha > 0:
                return f'fill="#{r:02x}{g:02x}{b:02x}" fill-opacity="{_n(alpha)}"'
    return 'fill="none"'


def _points_attr(points: list[Vec]) -> str:
    return " ".join(f"{_n(x)},{_n(y)}" for x, y in points)


def _bezier(points: list[Vec], segments: int = BEZIER_SEGMENTS) -> list[Vec]:
    """A cubic bezier (4 control points) flattened into ``segments`` straight pieces; any other count is drawn as given."""
    if len(points) != 4:
        return points
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    out: list[Vec] = []
    for i in range(segments + 1):
        t = i / segments
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        out.append((a * x0 + b * x1 + c * x2 + d * x3, a * y0 + b * y1 + c * y2 + d * y3))
    return out


def arc_path(start: Vec, mid: Vec, end: Vec) -> tuple[str, Vec | None, float]:
    """SVG path data for the three-point arc start -> mid -> end (sheet frame), its circumcentre and radius.

    The sweep flag follows the direction that reaches ``mid`` before ``end``;
    the large-arc flag is set when that arc spans more than 180 degrees.
    Collinear points (no circle) give the two straight segments and ``None``.
    """
    (ax, ay), (bx, by), (cx, cy) = start, mid, end
    det = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(det) < 1e-12:
        return f"M {_n(ax)} {_n(ay)} L {_n(bx)} {_n(by)} L {_n(cx)} {_n(cy)}", None, 0.0
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / det
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / det
    radius = math.hypot(ax - ux, ay - uy)
    tau = 2.0 * math.pi
    a0 = math.atan2(ay - uy, ax - ux)
    d_mid = (math.atan2(by - uy, bx - ux) - a0) % tau
    d_end = (math.atan2(cy - uy, cx - ux) - a0) % tau
    if d_mid < d_end:
        sweep, span = 1, d_end  # increasing angle: clockwise on the Y-down sheet
    else:
        sweep, span = 0, tau - d_end
    large = 1 if span > math.pi + 1e-9 else 0
    return f"M {_n(ax)} {_n(ay)} A {_n(radius)} {_n(radius)} 0 {large} {sweep} {_n(cx)} {_n(cy)}", (ux, uy), radius


def _unit_items(lib_symbol: Node, unit: int) -> list[Node]:
    """Drawable children of the unit sub-symbols shown for ``unit`` (unit 0 = common; body style 1 or 0 = common), in file order."""
    items: list[Node] = []
    for sub in sexpr.find_all(lib_symbol, "symbol"):
        if len(sub) < 2 or isinstance(sub[1], list):
            continue
        m = _UNIT_NAME_RE.match(str(sub[1]))
        if m is None:
            continue
        if int(m.group("unit")) not in (0, unit) or int(m.group("style")) > 1:
            continue
        items.extend(it for it in sub[2:] if isinstance(it, list))
    return items


def _draw_graphic(out: _Out, item: Node, inst: _Instance) -> None:
    kind = sexpr.head(item)
    stroke = _stroke_attrs(item, SYMBOL_STROKE)
    fill = _fill_attr(item)
    if kind == "rectangle":
        p0 = inst.point(*_xy(sexpr.find(item, "start"), "rectangle start"))
        p1 = inst.point(*_xy(sexpr.find(item, "end"), "rectangle end"))
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        out.body.append(f'<rect class="body" x="{_n(x0)}" y="{_n(y0)}" width="{_n(x1 - x0)}" height="{_n(y1 - y0)}" {fill} {stroke}/>')
        out.grow((x0, y0), (x1, y1))
    elif kind in ("polyline", "bezier"):
        pts_node = sexpr.find(item, "pts")
        local = [_xy(p, f"{kind} point") for p in sexpr.find_all(pts_node, "xy")] if pts_node is not None else []
        if kind == "bezier":
            local = _bezier(local)
        points = [inst.point(x, y) for x, y in local]
        if len(points) >= 2:
            out.body.append(f'<polyline class="body {kind}" points="{_points_attr(points)}" {fill} {stroke}/>')
            out.grow(*points)
    elif kind == "circle":
        cx, cy = inst.point(*_xy(sexpr.find(item, "center"), "circle center"))
        r = _num(sexpr.get(item, "radius", 1, "0"), "circle radius")
        out.body.append(f'<circle class="body" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" {fill} {stroke}/>')
        out.grow((cx - r, cy - r), (cx + r, cy + r))
    elif kind == "arc":
        s = inst.point(*_xy(sexpr.find(item, "start"), "arc start"))
        m = inst.point(*_xy(sexpr.find(item, "mid"), "arc mid"))
        e = inst.point(*_xy(sexpr.find(item, "end"), "arc end"))
        d, _, _ = arc_path(s, m, e)
        out.body.append(f'<path class="body arc" d="{d}" {fill} {stroke}/>')
        out.grow(s, m, e)
    elif kind == "text":
        _draw_symbol_text(out, item, inst)


def _draw_symbol_text(out: _Out, item: Node, inst: _Instance) -> None:
    effects = sexpr.find(item, "effects")
    font = _font(effects)
    if font.hidden or _hidden(item) or len(item) < 2 or isinstance(item[1], list):
        return
    at = sexpr.find(item, "at")
    lx, ly = _xy(at, "symbol text")
    angle = _num(at[3], "symbol text angle") / 10.0 if at is not None and len(at) > 3 else 0.0  # libraries store tenths of a degree
    theta, h, v = _oriented(angle, font.h, font.v, inst.vector)
    _text(out, str(item[1]), inst.point(lx, ly), size=font.size, theta=theta, h=h, v=v, colour=SYMBOL_STROKE, cls="symbol-text", font=font)


@dataclass(frozen=True, slots=True)
class _PinStyle:
    names_hidden: bool
    numbers_hidden: bool
    name_offset: float


def _pin_style(lib_symbol: Node) -> _PinStyle:
    names = sexpr.find(lib_symbol, "pin_names")
    numbers = sexpr.find(lib_symbol, "pin_numbers")
    offset = DEFAULT_PIN_NAME_OFFSET_MM
    if names is not None and sexpr.get(names, "offset") is not None:
        offset = _num(sexpr.get(names, "offset"), "pin_names offset")
    return _PinStyle(names_hidden=_hidden(names), numbers_hidden=_hidden(numbers), name_offset=offset)


def _pin_def(item: Node, unit: int) -> tuple[SymbolPin, Node | None, Node | None]:
    """The pin as the compiler reads it (:class:`SymbolPin`), plus its name / number nodes."""
    at = sexpr.find(item, "at")
    if at is None or len(at) < 4:
        raise ValueError("pin without (at x y angle)")
    x, y = _xy(at, "pin")
    number = sexpr.find(item, "number")
    name = sexpr.find(item, "name")
    pin = SymbolPin(
        number=str(number[1]) if number is not None and len(number) > 1 else "",
        name=str(name[1]) if name is not None and len(name) > 1 else "",
        electrical_type=str(item[1]) if len(item) > 1 and not isinstance(item[1], list) else "",
        x=x,
        y=y,
        angle=_num(at[3], "pin angle"),
        length=_num(sexpr.get(item, "length", 1, "0"), "pin length"),
        unit=unit,
        hidden=_hidden(item),
        graphic_style=str(item[2]) if len(item) > 2 and not isinstance(item[2], list) else "line",
    )
    return pin, name, number


def _draw_pin(out: _Out, item: Node, inst: _Instance, style: _PinStyle, unit: int) -> None:
    pin, name_node, number_node = _pin_def(item, unit)
    conn = pin_position(inst.x, inst.y, inst.rotation, inst.mirror, pin)
    u = pin_body_direction(inst.rotation, inst.mirror, pin.angle)
    inner = (snap(conn[0] + pin.length * u[0]), snap(conn[1] + pin.length * u[1]))
    hidden = ' data-hidden="yes" stroke-dasharray="0.4 0.3"' if pin.hidden else ""
    out.body.append(
        f'<line class="pin" data-number="{esc(pin.number)}" x1="{_n(conn[0])}" y1="{_n(conn[1])}" x2="{_n(inner[0])}" y2="{_n(inner[1])}" '
        f'stroke="{PIN_COLOUR}" stroke-width="{_n(LINE_MM)}"{hidden}/>'
    )
    out.grow(conn, inner)
    if pin.hidden:
        return
    name_font = _font(sexpr.find(name_node, "effects") if name_node is not None else None)
    number_font = _font(sexpr.find(number_node, "effects") if number_node is not None else None)
    show_name = not style.names_hidden and not name_font.hidden and pin.name not in ("", "~")
    show_number = not style.numbers_hidden and not number_font.hidden and pin.number != ""
    theta = 0.0 if abs(u[0]) >= abs(u[1]) else 90.0
    r, d = _frame(theta)
    up = (-d[0], -d[1])
    mid = ((conn[0] + inner[0]) / 2.0, (conn[1] + inner[1]) / 2.0)
    margin = PIN_TEXT_MARGIN_MM
    data = f' data-number="{esc(pin.number)}"'
    if style.name_offset > 0:
        if show_name:
            at = (inner[0] + u[0] * style.name_offset, inner[1] + u[1] * style.name_offset)
            h = "start" if _dot(u, r) > 0 else "end"
            _text(out, pin.name, at, size=name_font.size, theta=theta, h=h, v="center", colour=PIN_NAME_COLOUR, cls="pin-name", attrs=data)
        if show_number:
            at = (mid[0] + up[0] * margin, mid[1] + up[1] * margin)
            _text(out, pin.number, at, size=number_font.size, theta=theta, h="middle", v="bottom", colour=PIN_COLOUR, cls="pin-number", attrs=data)
    else:
        if show_name:
            at = (mid[0] + up[0] * margin, mid[1] + up[1] * margin)
            _text(out, pin.name, at, size=name_font.size, theta=theta, h="middle", v="bottom", colour=PIN_NAME_COLOUR, cls="pin-name", attrs=data)
        if show_number:
            at = (mid[0] - up[0] * margin, mid[1] - up[1] * margin)
            _text(out, pin.number, at, size=number_font.size, theta=theta, h="middle", v="top", colour=PIN_COLOUR, cls="pin-number", attrs=data)


def _draw_fields(out: _Out, instance: Node, inst: _Instance) -> None:
    """Every visible property of the instance at its own ``(at ..)``; Reference in ink, the others secondary."""
    for prop in sexpr.find_all(instance, "property"):
        if len(prop) < 3 or isinstance(prop[1], list) or isinstance(prop[2], list):
            continue
        key, value = str(prop[1]), str(prop[2])
        font = _font(sexpr.find(prop, "effects"))
        if _hidden(prop) or font.hidden or not value:
            continue
        at = sexpr.find(prop, "at")
        pos = _xy(at, f"property {key!r}")
        angle = _num(at[3], f"property {key!r} angle") if at is not None and len(at) > 3 else 0.0
        text = f"{key}: {value}" if _is_yes(prop, "show_name") else value
        theta, h, v = _oriented(angle, font.h, font.v, inst.vector)
        colour = REFERENCE_COLOUR if key == "Reference" else VALUE_COLOUR
        _text(out, text, pos, size=font.size, theta=theta, h=h, v=v, colour=colour, cls="field", font=font, attrs=f' data-field="{esc(key)}"')


def _draw_symbol(out: _Out, instance: Node, lib: dict[str, Node]) -> None:
    lib_id = str(sexpr.get(instance, "lib_id", 1, ""))
    lib_name = sexpr.get(instance, "lib_name")
    at = sexpr.find(instance, "at")
    x, y = _xy(at, f"symbol {lib_id!r}")
    rotation = _rotation(at[3]) if at is not None and len(at) > 3 else 0
    mirror_atom = sexpr.get(instance, "mirror")
    mirror = str(mirror_atom) if mirror_atom is not None else None
    if mirror not in (None, "x", "y"):
        raise ValueError(f"symbol {lib_id!r}: mirror must be x or y, got {mirror!r}")
    unit = int(_num(sexpr.get(instance, "unit", 1, "1"), "symbol unit"))
    inst = _Instance(x=x, y=y, rotation=rotation, mirror=mirror)
    ref = next((str(p[2]) for p in sexpr.find_all(instance, "property") if len(p) > 2 and p[1] == "Reference" and not isinstance(p[2], list)), "")
    lib_symbol = lib.get(str(lib_name)) if lib_name is not None else None
    if lib_symbol is None:
        lib_symbol = lib.get(lib_id)
    missing = ' data-missing="yes"' if lib_symbol is None else ""
    out.body.append(f'<g class="symbol" data-ref="{esc(ref)}" data-lib-id="{esc(lib_id)}" data-unit="{unit}"{missing}>')
    if lib_symbol is None:
        half = 2.54
        out.body.append(
            f'<rect class="body" x="{_n(x - half)}" y="{_n(y - half)}" width="{_n(2 * half)}" height="{_n(2 * half)}" fill="none" '
            f'stroke="{SYMBOL_STROKE}" stroke-width="{_n(LINE_MM)}" stroke-dasharray="0.5 0.5"/>'
        )
        out.grow((x - half, y - half), (x + half, y + half))
        _text(out, "??", (x, y), size=DEFAULT_TEXT_MM, theta=0.0, h="middle", v="center", colour=SYMBOL_STROKE, cls="symbol-text")
    else:
        items = _unit_items(lib_symbol, unit)
        for item in items:
            if sexpr.head(item) != "pin":
                _draw_graphic(out, item, inst)
        style = _pin_style(lib_symbol)
        for item in items:
            if sexpr.head(item) == "pin":
                _draw_pin(out, item, inst, style, unit)
    _draw_fields(out, instance, inst)
    out.body.append("</g>")


# --------------------------------------------------------------------------- sheet items


def _pts(item: Node, what: str) -> list[Vec]:
    pts_node = sexpr.find(item, "pts")
    return [_xy(p, what) for p in sexpr.find_all(pts_node, "xy")] if pts_node is not None else []


def _draw_line(out: _Out, points: list[Vec], cls: str, colour: str, width: float) -> None:
    if len(points) < 2:
        return
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        out.body.append(f'<line class="{cls}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" stroke="{colour}" stroke-width="{_n(width)}"/>')
    else:
        out.body.append(f'<polyline class="{cls}" points="{_points_attr(points)}" fill="none" stroke="{colour}" stroke-width="{_n(width)}"/>')
    out.grow(*points)


def _label_direction(angle: float) -> Vec:
    """The direction a label's text extends from its anchor: 0 -> +x, 90 -> up, 180 -> -x, 270 -> down."""
    rad = math.radians(angle)
    return snap(math.cos(rad), 6), snap(-math.sin(rad), 6)


def _label_at(item: Node, what: str) -> tuple[Vec, float]:
    at = sexpr.find(item, "at")
    pos = _xy(at, what)
    angle = _num(at[3], f"{what} angle") if at is not None and len(at) > 3 else 0.0
    return pos, angle


def _draw_global_label(out: _Out, item: Node) -> None:
    """The label outline of its ``shape`` from the anchor along its angle, the text inside (KiCad's global label geometry)."""
    name = str(item[1]) if len(item) > 1 and not isinstance(item[1], list) else ""
    (ax, ay), angle = _label_at(item, f"global_label {name!r}")
    font = _font(sexpr.find(item, "effects"))
    shape = str(sexpr.get(item, "shape", 1, "passive"))
    size = font.size
    e = _label_direction(angle)
    n = (-e[1], e[0])
    margin = LABEL_MARGIN_RATIO * size
    half = size / 2.0 + margin
    length = _text_width(name, size) + 2.0 * margin
    tip_in = shape in ("input", "bidirectional", "tri_state")
    tip_out = shape in ("output", "bidirectional", "tri_state")
    start = half if tip_in else 0.0  # an input / bidirectional tip sits on the anchor
    end = start + length
    # (along the label, across it): a box from the anchor, with a tip at either end for the directional shapes
    outline: list[tuple[float, float]] = [(0.0, 0.0), (start, half)] if tip_in else [(0.0, half)]
    outline.append((end, half))
    if tip_out:
        outline.append((end + half, 0.0))
    outline += [(end, -half), (start, -half)]
    points = [(snap(ax + e[0] * a + n[0] * b), snap(ay + e[1] * a + n[1] * b)) for a, b in outline]
    theta = _readable(e)
    r, d = _frame(theta)
    h = "start" if _dot(e, r) > 0 else "end"
    text_off = margin + (0.75 * size if tip_in else 0.0)
    lift = LABEL_TEXT_LIFT_RATIO * size
    anchor = (ax + e[0] * text_off - d[0] * lift, ay + e[1] * text_off - d[1] * lift)
    out.body.append(f'<g class="global-label" data-net="{esc(name)}" data-shape="{esc(shape)}">')
    out.body.append(f'<polygon points="{_points_attr(points)}" fill="none" stroke="{LABEL_COLOUR}" stroke-width="{_n(LINE_MM)}"/>')
    out.grow(*points)
    _text(out, name, anchor, size=size, theta=theta, h=h, v="center", colour=LABEL_COLOUR, cls="label-text", font=font)
    out.body.append("</g>")


def _draw_local_label(out: _Out, item: Node) -> None:
    name = str(item[1]) if len(item) > 1 and not isinstance(item[1], list) else ""
    (ax, ay), angle = _label_at(item, f"label {name!r}")
    font = _font(sexpr.find(item, "effects"))
    e = _label_direction(angle)
    theta = _readable(e)
    r, d = _frame(theta)
    lift = LABEL_OFFSET_RATIO * font.size + LINE_MM
    h = "start" if _dot(e, r) > 0 else "end"
    out.body.append(f'<g class="label" data-net="{esc(name)}">')
    _text(out, name, (ax - d[0] * lift, ay - d[1] * lift), size=font.size, theta=theta, h=h, v="bottom", colour=REFERENCE_COLOUR, cls="label-text", font=font)
    out.body.append("</g>")


def _draw_sheet_text(out: _Out, item: Node) -> None:
    if len(item) < 2 or isinstance(item[1], list):
        return
    font = _font(sexpr.find(item, "effects"))
    if font.hidden:
        return
    pos, angle = _label_at(item, "text")
    theta, h, v = _oriented(angle, font.h, font.v, _identity)
    _text(out, str(item[1]), pos, size=font.size, theta=theta, h=h, v=v, colour=VALUE_COLOUR, cls="sheet-text", font=font)


def _draw_junction(out: _Out, item: Node) -> None:
    cx, cy = _xy(sexpr.find(item, "at"), "junction")
    diameter = _num(sexpr.get(item, "diameter", 1, "0"), "junction diameter")
    r = (diameter if diameter > 0 else JUNCTION_MM) / 2.0
    out.body.append(f'<circle class="junction" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="{WIRE_COLOUR}"/>')
    out.grow((cx - r, cy - r), (cx + r, cy + r))


def _draw_no_connect(out: _Out, item: Node) -> None:
    x, y = _xy(sexpr.find(item, "at"), "no_connect")
    k = NO_CONNECT_HALF_MM
    out.body.append(
        f'<path class="no-connect" d="M {_n(x - k)} {_n(y - k)} L {_n(x + k)} {_n(y + k)} M {_n(x - k)} {_n(y + k)} L {_n(x + k)} {_n(y - k)}" '
        f'fill="none" stroke="{NO_CONNECT_COLOUR}" stroke-width="{_n(LINE_MM)}"/>'
    )
    out.grow((x - k, y - k), (x + k, y + k))


# --------------------------------------------------------------------------- the page


def paper_size(root: Node) -> tuple[float, float] | None:
    """``(width, height)`` in mm of the file's ``(paper ..)``: KiCad's table, ``portrait`` swapped, ``"User" W H`` as written; None when unknown."""
    paper = sexpr.find(root, "paper")
    if paper is None or len(paper) < 2 or isinstance(paper[1], list):
        return None
    name = str(paper[1])
    atoms = [str(a) for a in sexpr.args(paper)[1:]]
    size: tuple[float, float] | None = PAPER_SIZES_MM.get(name)
    if name == "User" and len(atoms) >= 2:
        try:
            w, h = _num(atoms[0], "paper width"), _num(atoms[1], "paper height")
        except ValueError:
            return None
        size = (w, h) if w > 0 and h > 0 else None
    if size is None:
        return None
    if "portrait" in atoms:
        size = (size[1], size[0])
    return size


def render_kicad_sch(node: Node) -> str:
    """The SVG preview of a parsed ``(kicad_sch ...)`` tree (see the module docstring for what is drawn and how)."""
    if sexpr.head(node) != "kicad_sch":
        raise ValueError(f"not a KiCad schematic: root node is ({sexpr.head(node) or '?'} ...)")
    lib: dict[str, Node] = {}
    lib_node = sexpr.find(node, "lib_symbols")
    if lib_node is not None:
        for sym in sexpr.find_all(lib_node, "symbol"):
            if len(sym) > 1 and not isinstance(sym[1], list):
                lib.setdefault(str(sym[1]), sym)

    wires, symbols, marks, labels = _Out(), _Out(), _Out(), _Out()
    for item in node[1:]:
        kind = sexpr.head(item)
        if kind == "wire":
            _draw_line(wires, _pts(item, "wire"), "wire", WIRE_COLOUR, LINE_MM)
        elif kind == "bus":
            _draw_line(wires, _pts(item, "bus"), "bus", BUS_COLOUR, BUS_MM)
        elif kind == "symbol":
            _draw_symbol(symbols, item, lib)
        elif kind == "junction":
            _draw_junction(marks, item)
        elif kind == "no_connect":
            _draw_no_connect(marks, item)
        elif kind == "global_label":
            _draw_global_label(labels, item)
        elif kind == "label":
            _draw_local_label(labels, item)
        elif kind == "text":
            _draw_sheet_text(labels, item)

    xs = wires.xs + symbols.xs + marks.xs + labels.xs or [0.0]
    ys = wires.ys + symbols.ys + marks.ys + labels.ys or [0.0]
    content = (min(xs) - CONTENT_MARGIN_MM, min(ys) - CONTENT_MARGIN_MM, max(xs) - min(xs) + 2 * CONTENT_MARGIN_MM, max(ys) - min(ys) + 2 * CONTENT_MARGIN_MM)
    size = paper_size(node)
    vx, vy, vw, vh = (0.0, 0.0, size[0], size[1]) if size is not None else content
    title = sexpr.find(node, "title_block")
    title_text = str(sexpr.get(title, "title", 1, "")) if title is not None else ""
    label = f"KiCad schematic preview: {title_text}" if title_text else "KiCad schematic preview"
    view = f"{_n(vx)} {_n(vy)} {_n(vw)} {_n(vh)}"
    content_box = " ".join(_n(v) for v in content)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="kicad-sch-preview" viewBox="{view}" width="{_n(vw)}mm" height="{_n(vh)}mm" '
        f'data-content-box="{content_box}" role="img" aria-label="{esc(label)}" font-family="{esc(_FONT_FAMILY)}" '
        'stroke-linecap="round" stroke-linejoin="round">',
        f"<title>{esc(label)}</title>",
        f'<rect class="paper" x="{_n(vx)}" y="{_n(vy)}" width="{_n(vw)}" height="{_n(vh)}" fill="{PAPER_COLOUR}" '
        f'stroke="{PAPER_EDGE if size is not None else "none"}" stroke-width="0.3"/>',
        '<g class="wires">',
        *wires.body,
        "</g>",
        '<g class="symbols">',
        *symbols.body,
        "</g>",
        '<g class="marks">',
        *marks.body,
        "</g>",
        '<g class="labels">',
        *labels.body,
        "</g>",
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def render_kicad_sch_file(path: Path | str) -> str:
    """Parse a ``.kicad_sch`` with :func:`ai_eda.tools.kicad.sexpr.parse_file` and render it (:func:`render_kicad_sch`)."""
    return render_kicad_sch(sexpr.parse_file(path))
