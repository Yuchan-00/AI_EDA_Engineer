"""Silkscreen geometry (pure): the text-box estimate, footprint silk graphics and pad copper as shapes in the board frame.

Invariant: every shape here is derived from the IR (placements, silk texts)
and the footprint read from a KiCad library on disk
(:class:`~ai_eda.tools.kicad.library.FootprintDef`: pads, ``F.SilkS`` /
``B.SilkS`` graphics and texts, the ``Reference`` property) through the
board-frame transforms of :mod:`ai_eda.tools.kicad.geometry` (mm, Y down,
bottom side mirrored). Nothing comes from model memory. The placer
(:mod:`ai_eda.tools.silkscreen.place`) and the ``pcb.silk.*`` checks
(:mod:`ai_eda.validation.layout`) read the board through this module; the
checks re-measure every final item against every keep-out, they never reuse
the placer's candidate bookkeeping.

**Shape model.** A :class:`Shape` is a point set grown by a radius ``r``
(the Minkowski sum): one point = a disc, two points = a capsule (a stroked
segment with round caps), three or more = a filled simple polygon (rounded
by ``r``). :func:`shape_distance` is the exact distance between two shapes
(0 when they touch or overlap).

* **Pad copper** (:func:`pad_copper`): the convex pad shapes are exact -
  ``rect`` a rectangle, ``roundrect`` the rectangle shrunk by the corner
  radius (``roundrect_rratio`` x the smaller side, KiCad's rule; 0.25 when the
  file gives none) grown by that radius, ``circle`` a disc, ``oval`` a
  capsule; each rotated by the stored pad angle and centred where KiCad puts
  the copper - the pad's ``(at)`` plus its ``(drill (offset x y))`` rotated
  by the pad angle (``PAD::ShapePos``; the hole stays at ``(at)``). A chamfer only removes
  copper, so the rounded rectangle stays an outer bound. A ``custom`` pad's
  primitives and a ``trapezoid``'s ``rect_delta`` are not read by the library
  reader: such a pad has ``shape=None`` (copper unknown) and every user says
  so instead of guessing. A pad is on the front / back silk side when any of
  its layers is a copper or mask layer of that side (``*.Cu`` / ``*.Mask``
  and through holes: both).
* **Footprint silk** (:func:`footprint_silk`): ``fp_line`` a capsule of half
  its stroke width; ``fp_rect`` / ``fp_poly`` / ``fp_circle`` their outline
  as capsules, or the filled area (``(fill yes|solid)``) grown by half the
  stroke; ``fp_arc`` and circles as chords whose capsule radius adds the
  chord's sagitta, so the chords cover the true arc; ``fp_curve`` the filled
  hull of its four control points (the Bezier curve lies inside it). Line
  style (dashes) is ignored: the solid stroke covers every dash. Visible
  ``fp_text`` / ``fp_text_box`` and visible properties on a silk layer are
  text items (``${REFERENCE}`` / ``${VALUE}`` expanded); the ``Reference``
  property is returned separately (:attr:`FootprintSilk.reference`): a
  designed ``reference`` :class:`~ai_eda.ir.SilkText` replaces it in the
  compiled board, otherwise the compiler writes it where the library put it
  (or at the footprint origin on ``F.SilkS``, 1.0 / 0.15 mm, when the library
  has none - :func:`default_reference_template`, the compiler's fallback).
* **Text box** (:func:`text_box`): KiCad's stroke font is proportional; this
  is a conservative *estimate*, not KiCad's font metrics: width = characters
  x size x :data:`TEXT_WIDTH_FACTOR` + thickness, height = size x
  :data:`TEXT_HEIGHT_FACTOR` + thickness, placed by the horizontal
  justification (``left``: the box starts at the anchor, ``right``: ends
  there, ``center``), vertically centred on the anchor (``top`` / ``bottom``
  for library texts), mirrored in the text's own x on the bottom side (a
  mirrored left-justified text runs to the left, as KiCad's
  ``EDA_TEXT::GetTextBox`` places it) and rotated by the text angle
  (counter-clockwise on screen). Whether KiCad's own glyph boxes stay inside
  it is not measured; KiCad's DRC (``silk_over_copper`` / ``silk_overlap``)
  is the judge of the compiled board.

Every coordinate is rounded to KiCad's 1e-6 mm resolution where it is
produced; a non-finite library coordinate is refused by the library loader
and a non-finite IR number by the compiler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from ai_eda.ir import BoardSide, Placement, SilkKind, SilkText
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import _q, mirrored_layer, pad_angle, pad_copper_center, pad_layers, rotate, text_angle, to_board
from ai_eda.tools.kicad.library import FootprintDef

__all__ = [
    "Point",
    "Box",
    "Shape",
    "PadCopper",
    "SilkItem",
    "FootprintSilk",
    "TEXT_WIDTH_FACTOR",
    "TEXT_HEIGHT_FACTOR",
    "KICAD_TEXT_SIZE_MM",
    "KICAD_TEXT_THICKNESS_MM",
    "SILK_LAYERS",
    "FAB_LAYERS",
    "CONVEX_PAD_SHAPES",
    "ARC_SAGITTA_MM",
    "side_of_layer",
    "text_extent",
    "text_box",
    "shape_distance",
    "shape_bbox",
    "inside_box",
    "pad_copper",
    "footprint_silk",
    "default_reference_template",
    "expand_text",
    "TextSpec",
    "silk_text_problems",
    "ir_text_box",
]

Point = tuple[float, float]
#: ``(x1, y1, x2, y2)``
Box = tuple[float, float, float, float]

#: the text-box estimate (module docstring): width per character and height, as multiples of the text size
TEXT_WIDTH_FACTOR = 0.9
TEXT_HEIGHT_FACTOR = 1.2
#: KiCad's default silk text: 1.0 mm high, 0.15 mm stroke (the compiler's fallback ``Reference`` property uses them)
KICAD_TEXT_SIZE_MM = 1.0
KICAD_TEXT_THICKNESS_MM = 0.15
SILK_LAYERS = frozenset({"F.SilkS", "B.SilkS"})
FAB_LAYERS = frozenset({"F.Fab", "B.Fab"})
#: pad shapes whose copper this module models exactly (the rest: copper unknown)
CONVEX_PAD_SHAPES = frozenset({"circle", "rect", "oval", "roundrect"})
#: the largest sagitta a chord of an arc / circle may have; the chord capsule's radius adds the actual sagitta
ARC_SAGITTA_MM = 0.005
#: KiCad's default corner ratio of a ``roundrect`` pad that does not state one
DEFAULT_RRATIO = 0.25

_GRAPHIC_HEADS = frozenset({"fp_line", "fp_rect", "fp_circle", "fp_arc", "fp_poly", "fp_curve"})


# --------------------------------------------------------------------------- shapes


@dataclass(frozen=True, slots=True)
class Shape:
    """A point set (1 = disc, 2 = capsule, >= 3 = filled simple polygon) grown by ``r`` (module docstring)."""

    points: tuple[Point, ...]
    r: float = 0.0

    def bbox(self) -> Box:
        return shape_bbox(self)


def shape_bbox(s: Shape) -> Box:
    xs = [p[0] for p in s.points]
    ys = [p[1] for p in s.points]
    return (min(xs) - s.r, min(ys) - s.r, max(xs) + s.r, max(ys) + s.r)


def inside_box(s: Shape, box: Box) -> bool:
    """Whether the whole shape lies inside ``box`` (exact: a convex hull of its points grown by ``r``)."""
    x1, y1, x2, y2 = shape_bbox(s)
    return _q(x1) >= _q(box[0]) and _q(y1) >= _q(box[1]) and _q(x2) <= _q(box[2]) and _q(y2) <= _q(box[3])


def _seg_point(a: Point, b: Point, p: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _orient(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _seg_seg(a: Point, b: Point, c: Point, d: Point) -> float:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return 0.0
    return min(_seg_point(a, b, c), _seg_point(a, b, d), _seg_point(c, d, a), _seg_point(c, d, b))


def _edges(points: tuple[Point, ...]) -> list[tuple[Point, Point]]:
    if len(points) == 1:
        return [(points[0], points[0])]
    if len(points) == 2:
        return [(points[0], points[1])]
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _point_in_polygon(p: Point, poly: tuple[Point, ...]) -> bool:
    """Ray casting (a point on the boundary may go either way; the edge distance is 0 there anyway)."""
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]):
            x = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
            if p[0] < x:
                inside = not inside
    return inside


def _core_distance(a: tuple[Point, ...], b: tuple[Point, ...]) -> float:
    """Distance between the two point sets (0 when they meet): polygons are filled."""
    if len(a) >= 3 and any(_point_in_polygon(p, a) for p in b):
        return 0.0
    if len(b) >= 3 and any(_point_in_polygon(p, b) for p in a):
        return 0.0
    return min(_seg_seg(p, q, r, s) for p, q in _edges(a) for r, s in _edges(b))


def shape_distance(a: Shape, b: Shape) -> float:
    """Exact distance between two shapes (0 when they touch or overlap), rounded to 1e-6 mm."""
    return max(_q(_core_distance(a.points, b.points) - a.r - b.r), 0.0)


# --------------------------------------------------------------------------- text boxes


def side_of_layer(layer: str) -> str:
    """``"F"`` / ``"B"`` for a front / back layer name."""
    return "B" if layer.startswith("B.") else "F"


def text_extent(text: str, size: float, thickness: float) -> tuple[float, float]:
    """``(width, height)`` of the conservative text-box estimate (module docstring)."""
    return len(text) * size * TEXT_WIDTH_FACTOR + thickness, size * TEXT_HEIGHT_FACTOR + thickness


def text_box(
    text: str, x: float, y: float, rotation: float, size: float, thickness: float, justify: str = "center", *,
    mirrored: bool = False, vjustify: str = "center",
) -> Shape:
    """The estimated box of a text anchored at ``(x, y)`` as a 4-point polygon (board frame)."""
    w, h = text_extent(text, size, thickness)
    if justify == "left":
        lx = (0.0, w)
    elif justify == "right":
        lx = (-w, 0.0)
    else:
        lx = (-w / 2.0, w / 2.0)
    if mirrored:
        lx = (-lx[1], -lx[0])
    if vjustify == "top":
        ly = (0.0, h)
    elif vjustify == "bottom":
        ly = (-h, 0.0)
    else:
        ly = (-h / 2.0, h / 2.0)
    corners = [(lx[0], ly[0]), (lx[1], ly[0]), (lx[1], ly[1]), (lx[0], ly[1])]
    pts = []
    for cx, cy in corners:
        rx, ry = rotate(cx, cy, rotation)
        pts.append((_q(x + rx), _q(y + ry)))
    return Shape(tuple(pts))


def expand_text(text: str, ref: str, value: str) -> str:
    """``${REFERENCE}`` / ``${VALUE}`` as KiCad expands them on the board (other variables stay as written)."""
    return text.replace("${REFERENCE}", ref).replace("${VALUE}", value)


# --------------------------------------------------------------------------- pads


@dataclass(frozen=True, slots=True)
class PadCopper:
    """One pad's copper in the board frame; ``shape`` is ``None`` when it is not modelled (custom / trapezoid)."""

    label: str  # "R1.2"
    ref: str
    number: str
    sides: frozenset[str]  # {"F"}, {"B"} or both
    shape: Shape | None
    pad_shape: str


def _pad_sides(layers: list[str], pad_type: str) -> frozenset[str]:
    if pad_type in ("thru_hole", "np_thru_hole"):
        return frozenset({"F", "B"})
    sides: set[str] = set()
    for layer in layers:
        if layer in ("*.Cu", "*.Mask", "F&B.Cu"):
            sides |= {"F", "B"}
        elif layer in ("F.Cu", "F.Mask"):
            sides.add("F")
        elif layer in ("B.Cu", "B.Mask"):
            sides.add("B")
    return frozenset(sides)


def pad_copper(ref: str, fp: FootprintDef, placement: Placement) -> list[PadCopper]:
    """Every pad of ``fp`` placed at ``placement`` as copper shapes (module docstring)."""
    out: list[PadCopper] = []
    for pad in fp.pads:
        label = f"{ref}.{pad.number}" if pad.number else f"{ref}.(unnumbered)"
        sides = _pad_sides(pad_layers(placement, pad), pad.pad_type)
        if pad.shape not in CONVEX_PAD_SHAPES:
            out.append(PadCopper(label, ref, pad.number, sides, None, pad.shape))
            continue
        w, h = pad.size_w, pad.size_h
        if pad.shape == "rect":
            r = 0.0
        elif pad.shape == "roundrect":
            r = (pad.roundrect_rratio if pad.roundrect_rratio is not None else DEFAULT_RRATIO) * min(w, h)
        else:  # circle / oval: the full half of the smaller side
            r = min(w, h) / 2.0
        hx, hy = max(w / 2.0 - r, 0.0), max(h / 2.0 - r, 0.0)
        cx, cy = pad_copper_center(placement, pad)  # the drill offset moves the copper, not the hole
        angle = pad_angle(placement, pad)
        if hx > 0.0 and hy > 0.0:
            local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        elif hx > 0.0 or hy > 0.0:
            local = [(-hx, -hy), (hx, hy)]
        else:
            local = [(0.0, 0.0)]
        pts = []
        for px, py in local:
            rx, ry = rotate(px, py, angle)
            pts.append((_q(cx + rx), _q(cy + ry)))
        out.append(PadCopper(label, ref, pad.number, sides, Shape(tuple(pts), _q(r)), pad.shape))
    return out


# --------------------------------------------------------------------------- footprint silk


@dataclass(frozen=True, slots=True)
class SilkItem:
    """One silk graphic or text of a footprint (board frame) on ``layer`` (``F.SilkS`` / ``B.SilkS``)."""

    label: str  # "U1:fp_line[3]", "U1:text[${VALUE}]"
    ref: str
    layer: str
    kind: str  # "graphic" | "text"
    shape: Shape | None  # None: a construct this module does not read (reported, never guessed)
    width: float | None  # stroke width (graphics) / thickness (texts); None for a fill without a stroke
    size: float | None = None  # text height
    text: str | None = None
    why: str = ""  # why ``shape`` is None


@dataclass(frozen=True, slots=True)
class TextSpec:
    """A text as KiCad would draw it on the board: position, angle, size, stroke, justification, layer."""

    text: str
    x: float
    y: float
    rotation: float
    size: float
    thickness: float
    justify: str
    vjustify: str
    layer: str
    hidden: bool

    def box(self) -> Shape:
        return text_box(self.text, self.x, self.y, self.rotation, self.size, self.thickness, self.justify,
                        mirrored=self.layer.startswith("B."), vjustify=self.vjustify)


@dataclass(slots=True)
class FootprintSilk:
    """The silk of one placed footprint: its graphics and texts, and where its ``Reference`` would be by default."""

    ref: str
    items: list[SilkItem] = field(default_factory=list)
    reference: TextSpec | None = None  # the library default (hidden or not), None only when not readable


def default_reference_template() -> list:
    """The ``Reference`` property the compiler writes for a footprint whose library entry has none."""
    return sexpr.S("property", sexpr.Q("Reference"), sexpr.Q(""), sexpr.S("at", 0, 0, 0), sexpr.S("layer", sexpr.Q("F.SilkS")),
                   sexpr.S("effects", sexpr.S("font", sexpr.S("size", KICAD_TEXT_SIZE_MM, KICAD_TEXT_SIZE_MM), sexpr.S("thickness", KICAD_TEXT_THICKNESS_MM))))


def _hidden(node: list) -> bool:
    if sexpr.get(node, "hide") == "yes" or "hide" in sexpr.args(node):
        return True
    effects = sexpr.find(node, "effects")
    return effects is not None and (sexpr.get(effects, "hide") == "yes" or "hide" in sexpr.args(effects))


def _stroke_width(node: list) -> float:
    stroke = sexpr.find(node, "stroke")
    if stroke is not None:
        w = sexpr.get(stroke, "width")
        if w is not None:
            return float(sexpr.to_float(w))
    w = sexpr.get(node, "width")
    return float(sexpr.to_float(w)) if w is not None else 0.0


def _filled(node: list) -> bool:
    fill = sexpr.find(node, "fill")
    if fill is None:
        return False
    args = [str(a) for a in sexpr.args(fill)]
    if any(a in ("yes", "solid") for a in args):
        return True
    ftype = sexpr.get(fill, "type")
    return ftype is not None and str(ftype) not in ("none", "no")


def _pt(node: list | None) -> Point | None:
    if node is None or len(node) < 3:
        return None
    return float(sexpr.to_float(node[1])), float(sexpr.to_float(node[2]))


def _arc_points(start: Point, mid: Point, end: Point) -> tuple[list[Point], float] | None:
    """Chord points of the arc start -> mid -> end (library frame) and the largest chord sagitta; None when collinear."""
    ax, ay = start
    bx, by = mid
    cx, cy = end
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    radius = math.hypot(ax - ux, ay - uy)
    a0 = math.atan2(ay - uy, ax - ux)
    am = math.atan2(by - uy, bx - ux)
    a1 = math.atan2(cy - uy, cx - ux)

    def ccw(a: float, b: float) -> float:
        return (b - a) % (2.0 * math.pi)

    sweep = ccw(a0, a1)
    if ccw(a0, am) > sweep:  # the mid point is on the other way round
        sweep = sweep - 2.0 * math.pi
    return _circle_points(ux, uy, radius, a0, sweep)


def _circle_points(ux: float, uy: float, radius: float, a0: float, sweep: float) -> tuple[list[Point], float]:
    """Chords over ``sweep`` radians from ``a0``: enough that the sagitta stays below :data:`ARC_SAGITTA_MM`."""
    if radius <= ARC_SAGITTA_MM:
        n = 1
    else:
        step = 2.0 * math.acos(max(-1.0, 1.0 - ARC_SAGITTA_MM / radius))
        n = max(1, math.ceil(abs(sweep) / step))
    sagitta = radius * (1.0 - math.cos(abs(sweep) / n / 2.0))
    pts = [(ux + radius * math.cos(a0 + sweep * i / n), uy + radius * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]
    return pts, sagitta


def _hull(points: list[Point]) -> list[Point]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _outline_shapes(points: list[Point], closed: bool, r: float) -> list[Shape]:
    segs = list(zip(points, points[1:]))
    if closed and len(points) > 2:
        segs.append((points[-1], points[0]))
    return [Shape((a, b), r) for a, b in segs] if segs else [Shape((points[0],), r)] if points else []


def _graphic_shapes(node: list, placement: Placement) -> tuple[list[Shape] | None, str]:
    """Board-frame shapes of one library silk graphic (``None`` + why for a construct not read)."""
    head = sexpr.head(node)
    half = _stroke_width(node) / 2.0
    filled = _filled(node)

    def board(pts: list[Point], extra: float = 0.0, closed: bool = False, fill: bool = False) -> list[Shape]:
        placed = [to_board(placement, x, y) for x, y in pts]
        shapes = _outline_shapes(placed, closed, _q(half + extra))
        if fill and len(placed) >= 3:
            shapes.append(Shape(tuple(placed), _q(half)))
        return shapes

    if head == "fp_line":
        a, b = _pt(sexpr.find(node, "start")), _pt(sexpr.find(node, "end"))
        if a is None or b is None:
            return None, "fp_line without start / end"
        return board([a, b]), ""
    if head == "fp_rect":
        a, b = _pt(sexpr.find(node, "start")), _pt(sexpr.find(node, "end"))
        if a is None or b is None:
            return None, "fp_rect without start / end"
        corners = [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])]
        return board(corners, closed=True, fill=filled), ""
    if head == "fp_circle":
        c, e = _pt(sexpr.find(node, "center")), _pt(sexpr.find(node, "end"))
        if c is None or e is None:
            return None, "fp_circle without center / end"
        radius = math.hypot(e[0] - c[0], e[1] - c[1])
        if filled:
            cx, cy = to_board(placement, c[0], c[1])
            return [Shape(((cx, cy),), _q(radius + half))], ""
        pts, sag = _circle_points(c[0], c[1], radius, 0.0, 2.0 * math.pi)
        return board(pts, extra=sag), ""
    if head == "fp_arc":
        a, m, b = _pt(sexpr.find(node, "start")), _pt(sexpr.find(node, "mid")), _pt(sexpr.find(node, "end"))
        if a is None or m is None or b is None:
            return None, "fp_arc without start / mid / end (only the three-point form is read)"
        arc = _arc_points(a, m, b)
        if arc is None:
            return board([a, b]), ""
        pts, sag = arc
        return board(pts, extra=sag), ""
    if head == "fp_poly":
        pts_node = sexpr.find(node, "pts")
        if pts_node is None:
            return None, "fp_poly without pts"
        pts: list[Point] = []
        extra = 0.0
        for child in pts_node[1:]:
            if sexpr.head(child) == "xy":
                p = _pt(child)
                if p is not None:
                    pts.append(p)
            elif sexpr.head(child) == "arc":
                a, m, b = _pt(sexpr.find(child, "start")), _pt(sexpr.find(child, "mid")), _pt(sexpr.find(child, "end"))
                if a is None or m is None or b is None:
                    return None, "fp_poly arc without start / mid / end"
                arc = _arc_points(a, m, b)
                if arc is None:
                    pts += [a, b]
                else:
                    pts += arc[0]
                    extra = max(extra, arc[1])
        if not pts:
            return None, "fp_poly without points"
        return board(pts, extra=extra, closed=True, fill=filled), ""
    if head == "fp_curve":
        pts_node = sexpr.find(node, "pts")
        pts = [p for p in (_pt(c) for c in (pts_node[1:] if pts_node is not None else []) if sexpr.head(c) == "xy") if p is not None]
        if len(pts) < 2:
            return None, "fp_curve without control points"
        placed = [to_board(placement, x, y) for x, y in pts]
        hull = _hull(placed)
        return [Shape(tuple(hull), _q(half))], ""
    return None, f"({head} ...) is not a silk graphic this module reads"


def _text_spec(node: list, placement: Placement, text: str) -> TextSpec | None:
    """Where KiCad draws a footprint text / property: the library offset transformed, the stored angle, the library font."""
    at = sexpr.find(node, "at")
    layer = sexpr.get(node, "layer")
    if at is None or len(at) < 3 or layer is None:
        return None
    lx, ly = float(sexpr.to_float(at[1])), float(sexpr.to_float(at[2]))
    lib_angle = float(sexpr.to_float(at[3])) if len(at) > 3 else 0.0
    x, y = to_board(placement, lx, ly)
    effects = sexpr.find(node, "effects")
    font = sexpr.find(effects, "font") if effects is not None else None
    size_node = sexpr.find(font, "size") if font is not None else None
    size = float(sexpr.to_float(size_node[1])) if size_node is not None and len(size_node) > 1 else KICAD_TEXT_SIZE_MM
    thick = sexpr.get(font, "thickness") if font is not None else None
    thickness = float(sexpr.to_float(thick)) if thick is not None else _q(size * 0.15)
    justify = sexpr.find(effects, "justify") if effects is not None else None
    jargs = [str(a) for a in sexpr.args(justify)] if justify is not None else []
    h = "left" if "left" in jargs else "right" if "right" in jargs else "center"
    v = "top" if "top" in jargs else "bottom" if "bottom" in jargs else "center"
    return TextSpec(text, x, y, text_angle(placement, lib_angle), size, thickness, h, v, mirrored_layer(str(layer), placement.side), _hidden(node))


def footprint_silk(ref: str, fp: FootprintDef, placement: Placement, values: Mapping[str, str] | None = None) -> FootprintSilk:
    """The silk of ``fp`` placed at ``placement`` (module docstring): graphics, visible texts, and the default ``Reference``.

    ``values`` are the property values the compiler writes from the IR
    (``Value``, ``Datasheet``, ``Description``); a property it does not name
    keeps the library's text, as the compiler keeps it.
    """
    values = dict(values or {})
    value = values.get("Value", "")
    out = FootprintSilk(ref=ref)
    counters: dict[str, int] = {}
    reference_node: list | None = None
    for node in fp.node:
        head = sexpr.head(node)
        if head is None:
            continue
        if head == "property" and len(node) > 2 and str(node[1]) == "Reference":
            if reference_node is None:
                reference_node = node
            continue
        layer = sexpr.get(node, "layer")
        if layer is None:
            continue
        board_layer = mirrored_layer(str(layer), placement.side)
        if board_layer not in SILK_LAYERS:
            continue
        index = counters.get(head, 0)
        counters[head] = index + 1
        label = f"{ref}:{head}[{index}]"
        if head in _GRAPHIC_HEADS:
            shapes, why = _graphic_shapes(node, placement)
            width = _stroke_width(node)
            if shapes is None:
                out.items.append(SilkItem(label, ref, board_layer, "graphic", None, width or None, why=why))
                continue
            for k, shape in enumerate(shapes):
                out.items.append(SilkItem(label if len(shapes) == 1 else f"{label}.{k}", ref, board_layer, "graphic", shape, width if width > 0 else None))
        elif head in ("fp_text", "property"):
            if _hidden(node):
                continue
            raw = str(node[2]) if len(node) > 2 else ""  # (fp_text user "TEXT" ...) / (property "KEY" "VALUE" ...)
            if head == "property" and str(node[1]) in values:
                raw = values[str(node[1])]
            text = expand_text(raw, ref, value)
            spec = _text_spec(node, placement, text)
            if spec is None:
                out.items.append(SilkItem(label, ref, board_layer, "text", None, None, why=f"({head} ...) without (at) / (layer)"))
                continue
            out.items.append(SilkItem(f"{ref}:text[{raw}]", ref, board_layer, "text", spec.box(), spec.thickness, spec.size, text))
        elif head == "fp_text_box":
            if _hidden(node):
                continue
            a, b = _pt(sexpr.find(node, "start")), _pt(sexpr.find(node, "end"))
            pts_node = sexpr.find(node, "pts")
            corners: list[Point] = []
            if a is not None and b is not None:
                corners = [(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])]
            elif pts_node is not None:
                corners = [p for p in (_pt(c) for c in pts_node[1:] if sexpr.head(c) == "xy") if p is not None]
            if len(corners) < 3:
                out.items.append(SilkItem(label, ref, board_layer, "text", None, None, why="fp_text_box without start / end or pts"))
                continue
            placed = tuple(to_board(placement, x, y) for x, y in corners)
            out.items.append(SilkItem(label, ref, board_layer, "text", Shape(placed), _stroke_width(node) or None, None, str(node[1]) if len(node) > 1 else ""))
    node = reference_node if reference_node is not None else default_reference_template()
    out.reference = _text_spec(node, placement, ref)
    return out



# --------------------------------------------------------------------------- IR silk texts


def ir_text_box(t: SilkText) -> Shape:
    """The estimated box of an IR :class:`~ai_eda.ir.SilkText` (mirrored on a ``B.*`` layer)."""
    return text_box(t.text, t.x_mm, t.y_mm, t.rotation_deg, t.size_mm, t.thickness_mm, t.justify, mirrored=t.layer.startswith("B."))


def silk_text_problems(texts: list[SilkText], sides: Mapping[str, BoardSide | None]) -> dict[int, str]:
    """``{index: why}`` for every IR silk text the PCB compiler refuses; the ``pcb.silk.*`` checks report the same.

    ``sides`` maps every IR component reference to its placement side
    (``None`` when it has no placement). A text is refused when its text is
    empty or holds a control character, a number is not finite, its size or
    stroke is not positive, it is on a layer that is neither a silkscreen nor
    a fab layer, a non-reference text is on a fab layer, or it names a
    component that does not exist; a ``reference`` text also when its text
    is not its component's reference, it is the component's second reference
    text, or its layer is on the other side than the footprint.
    """
    out: dict[int, str] = {}
    seen_refs: dict[str, int] = {}
    for i, t in enumerate(texts):
        numbers = (t.x_mm, t.y_mm, t.rotation_deg, t.size_mm, t.thickness_mm)
        if not t.text or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in t.text):
            out[i] = f"silk text {t.text!r} is empty or holds a control character (one line of text only)"
        elif not all(math.isfinite(v) for v in numbers):
            out[i] = f"silk text {t.text!r} has a non-finite number (at ({t.x_mm}, {t.y_mm}), angle {t.rotation_deg}, size {t.size_mm}, stroke {t.thickness_mm})"
        elif t.size_mm <= 0.0 or t.thickness_mm <= 0.0:
            out[i] = f"silk text {t.text!r} has a non-positive size {t.size_mm:g} mm or stroke {t.thickness_mm:g} mm"
        elif t.layer not in SILK_LAYERS and t.layer not in FAB_LAYERS:
            out[i] = f"silk text {t.text!r} is on layer {t.layer!r}, which is not a silkscreen or fab layer of the board ({sorted(SILK_LAYERS | FAB_LAYERS)})"
        elif t.kind != SilkKind.REFERENCE and t.layer in FAB_LAYERS:
            out[i] = f"{t.kind} text {t.text!r} is on the fab layer {t.layer!r}; only a reference goes there"
        elif t.component_ref is not None and t.component_ref not in sides:
            out[i] = f"silk text {t.text!r} names component {t.component_ref!r}, which is not in the IR"
        elif t.kind == SilkKind.REFERENCE:
            if t.component_ref is None:
                out[i] = f"reference text {t.text!r} names no component"
            elif t.text != t.component_ref:
                out[i] = f"reference text {t.text!r} is not the reference of its component {t.component_ref!r}"
            elif t.component_ref in seen_refs:
                out[i] = f"second reference text for {t.component_ref!r} (the first is silkscreen[{seen_refs[t.component_ref]}])"
            elif sides[t.component_ref] is None:
                out[i] = f"reference text of {t.component_ref!r}, which has no placement"
            elif (sides[t.component_ref] == BoardSide.BOTTOM) != t.layer.startswith("B."):
                out[i] = f"reference text of {t.component_ref!r} is on {t.layer!r}, the other side than its footprint ({sides[t.component_ref]})"
            if t.component_ref is not None and t.component_ref not in seen_refs:
                seen_refs[t.component_ref] = i
    return out
