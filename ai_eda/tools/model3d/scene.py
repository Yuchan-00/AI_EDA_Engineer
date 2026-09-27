"""A 3D preview scene of the placed board, built from the IR and the KiCad libraries on disk (no status, no guess).

Invariant: every solid comes from the IR (outline, placements, tracks, vias)
or from a library file on disk (pad shapes and silkscreen strokes from the
``.kicad_mod``, body heights from the STEP file the footprint's ``(model ...)``
names, :mod:`ai_eda.tools.model3d.step_bbox`). The scene is a *picture*: it
claims nothing about the board and computes no status. The same IR, library
and 3D files give the same scene (IR order, library order, coordinates
rounded to 1e-6 mm), so the writers built on it are byte-deterministic.

What is drawn (board frame: millimetres, y down; z up, board bottom at 0,
top at the thickness ``T``):

* the board slab: the outline x ``T`` - ``board_thickness_mm`` from
  ``ir.pcb.manufacturing`` when its provenance is authoritative (the rule the
  PCB compiler uses for ``(general (thickness ..))``), else 1.6 mm, recorded
  as an assumption (``가정``) in :attr:`Scene.notes`;
* a solder-mask sheet on each side (translucent; the board's vias are tented,
  so via copper stays under it, as in ``compilers/pcb.py`` ``_setup``);
* the **inner planes** of a multilayer board: every zone on an inner copper
  layer (``In1.Cu`` ...) - the plane zones the PCB agent draws from the
  stackup - as a translucent copper sheet (material ``plane``, a top and a
  bottom face) at the layer's depth in ``ir.pcb.stackup`` (copper and
  dielectric thicknesses from the top, scaled to the drawn thickness when the
  stack's own total differs), over the zone's polygon. The fill (the
  clearances KiCad cuts around other nets' pads and vias) is KiCad's
  computation and is not drawn. So the sheets can be seen, the slab of such a
  board is drawn translucent (material ``board_clear``); a board without an
  inner plane keeps the opaque slab and gives exactly the scene it gave
  before. An inner zone without a stackup has no depth and is not drawn
  (said in the notes);
* copper: tracks on F.Cu / B.Cu (inner layers are hidden in the board and not
  drawn), via rings on the outer layers they reach, pads from the library
  footprint on every outer copper layer they are on (above the mask when the
  pad opens the mask, under it otherwise), centred on the copper (the pad's
  ``(at)`` plus its drill offset; the drill stays at ``(at)``); circle / oval /
  rect / roundrect are drawn as such, a trapezoid by KiCad's corner formula
  from its ``rect_delta``, a ``custom`` pad as the rectangle around its
  anchor and every copper primitive (each grown by half its width; counted
  in the notes as 외접 사각형), and a pad whose copper this module cannot
  read (an unknown primitive or shape) as its ``(size)`` box, said in the
  notes to be possibly smaller than the copper;
* drills (pads and vias) as dark 8-sided cylinders through the board; oval
  drills are drawn round;
* silkscreen **strokes** of the footprints' own library graphics
  (``F.SilkS`` / ``B.SilkS`` lines, arcs, circles, rectangles, polygons,
  curves, mirrored with the footprint). **Texts are omitted in 3D** - the
  footprint's text items and the IR's silk texts alike (said in the notes);
* per component a **body box**: x/y = the box around the footprint's
  ``F.Fab`` graphics (else its courtyard), z = 0 .. the highest z of the STEP
  envelope after the model's scale / rotate / offset (all visible models of
  the footprint together), placed with the footprint's rotation and side. A
  component whose STEP file is not found (or unreadable, or lies wholly
  below the surface) gets a flat outline of that box instead, with the reason
  in :attr:`BodyInfo.reason`. The box is not the part's shape
  (:data:`BODY_CAPTION`).

Nothing thinner than 0.1 mm (copper, mask, silk) is a closed solid: it is a
*decal* - only its outer face (``Solid.faces`` ``"top"`` / ``"bottom"``) - at
fixed drawing heights (:data:`COPPER_MM` ...) that are picture values, not
measurements.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import EllipsisType

from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, Placement
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import mirrored_layer, pad_angle, pad_center, pad_copper_center, pad_layers, rotate, to_board
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, Pad
from ai_eda.tools.model3d.models import ModelRef, find_3dmodel_dir, footprint_models, resolve_model_path, transformed_box
from ai_eda.tools.model3d.step_bbox import Box3, StepEnvelope, read_step_envelope

__all__ = [
    "SCENE_VERSION",
    "DEFAULT_BOARD_THICKNESS_MM",
    "BODY_CAPTION",
    "GROUP_LABELS",
    "Material",
    "MATERIALS",
    "Solid",
    "BodyInfo",
    "Scene",
    "SceneError",
    "Face",
    "build_scene",
    "solid_faces",
    "triangulate",
    "scene_caption",
    "GRAPHIC_HEADS",
    "graphic_paths",
    "stroke_width",
]

SCENE_VERSION = "0.1"
#: what the PCB compiler writes when the IR carries no grounded thickness (``ai_eda.compilers.pcb.DEFAULT_BOARD_THICKNESS_MM``)
DEFAULT_BOARD_THICKNESS_MM = 1.6
#: the caption every view of the scene carries: what a body box is
BODY_CAPTION = "부품 = F.Fab 외곽 × STEP 최대 높이의 상자, 실제 모양 아님"

# drawing heights above the surface of each side (picture values, not measurements)
COPPER_MM = 0.035
MASK_MM = 0.045
PAD_MM = 0.05
SILK_MM = 0.06
OUTLINE_MM = 0.065
DRILL_OVER_MM = 0.07
OUTLINE_STROKE_MM = 0.12
_DRILL_SIDES = 8
_VIA_SIDES = 12
_CIRCLE_SIDES = 16
_CAP_SEGMENTS = 4
_CORNER_SEGMENTS = 3
_ARC_STEP_RAD = math.pi / 12
_BEZIER_SEGMENTS = 8
_DECAL_MAX_MM = 0.1
_DECIMALS = 6

#: an inner copper layer (``In1.Cu`` ...): its zones are the planes drawn as sheets (the router's ``INNER_LAYER_RE``)
INNER_LAYER_RE = re.compile(r"^In[1-9][0-9]*\.Cu$")

GROUP_LABELS: dict[str, str] = {"board": "보드", "copper": "구리", "silk": "실크", "parts": "부품"}


@dataclass(frozen=True, slots=True)
class Material:
    key: str
    rgba: tuple[float, float, float, float]
    group: str  # a key of GROUP_LABELS (the viewer's layer toggles)
    metallic: float = 0.0
    roughness: float = 0.8


#: every material in the order the writers emit them
MATERIALS: tuple[Material, ...] = (
    Material("board", (0.45, 0.41, 0.25, 1.0), "board"),
    # a board with inner planes: the slab translucent so the plane sheets inside it show (drawn before the sheets and the mask)
    Material("board_clear", (0.45, 0.41, 0.25, 0.55), "board"),
    Material("plane", (0.42, 0.24, 0.12, 0.5), "copper", 0.6, 0.4),
    Material("mask", (0.06, 0.36, 0.18, 0.72), "board"),
    Material("drill", (0.10, 0.10, 0.10, 1.0), "board"),
    Material("copper", (0.95, 0.66, 0.35, 1.0), "copper", 0.6, 0.4),
    Material("pad", (0.85, 0.72, 0.36, 1.0), "copper", 0.7, 0.35),
    Material("silk", (0.95, 0.95, 0.93, 1.0), "silk"),
    Material("body", (0.26, 0.28, 0.31, 1.0), "parts", 0.0, 0.6),
    Material("outline", (0.93, 0.62, 0.20, 1.0), "parts"),
)
_MATERIAL_KEYS = frozenset(m.key for m in MATERIALS)


class SceneError(CompileError, ValueError):
    """The IR cannot be pictured without guessing (no outline, a component without a placement or a footprint on disk ...)."""


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Solid:
    """A vertical prism: ``polygon`` (board frame, mm, y down; counter-clockwise seen from the top, i.e. in KiCad's y-up 3D frame) from ``z0`` to ``z1``.

    ``faces``: ``"all"`` for a closed solid, ``"top"`` / ``"bottom"`` for a
    decal (only the face at ``z1`` / ``z0`` exists).
    """

    kind: str  # slab | mask | plane | track | via | pad | drill | silk | body | outline
    material: str
    polygon: tuple[Point, ...]
    z0: float
    z1: float
    faces: str
    label: str


@dataclass(frozen=True, slots=True)
class BodyInfo:
    """The body of one component: its box height from the STEP envelope, or why there is none."""

    ref: str
    lib_id: str
    side: str
    outline_source: str  # "F.Fab" | "courtyard" | "" (neither: nothing drawn)
    models: tuple[str, ...]  # the (model ...) references as written in the footprint
    height_mm: float | None  # None: flat outline only
    envelopes: tuple[tuple[str, Box3], ...] = ()  # (model reference, envelope after offset / scale / rotate)
    reason: str = ""


@dataclass
class Scene:
    project_id: str
    outline: tuple[float, float, float, float]  # origin x, origin y, width, height (mm)
    thickness_mm: float
    thickness_grounded: bool
    solids: list[Solid] = field(default_factory=list)
    bodies: list[BodyInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model_dir_found: bool = False
    version: str = SCENE_VERSION

    def counts(self) -> dict[str, int]:
        """Solids per kind, sorted by kind."""
        out: dict[str, int] = {}
        for s in self.solids:
            out[s.kind] = out.get(s.kind, 0) + 1
        return dict(sorted(out.items()))

    @property
    def bodies_with_step(self) -> list[BodyInfo]:
        return [b for b in self.bodies if b.height_mm is not None]

    @property
    def bodies_without_step(self) -> list[BodyInfo]:
        return [b for b in self.bodies if b.height_mm is None]


@dataclass(frozen=True, slots=True)
class Face:
    """One planar face in KiCad's 3D frame (x, **y up** = -board y, z); ``points`` counter-clockwise seen from outside."""

    normal: tuple[float, float, float]
    points: tuple[tuple[float, float, float], ...]
    which: str  # "top" | "bottom" | "side"


# --------------------------------------------------------------------------- 2-D helpers


def _q(v: float) -> float:
    return round(v, _DECIMALS) + 0.0


def _signed_area(poly: list[Point]) -> float:
    """Area in the y-up frame (positive = counter-clockwise seen from the top)."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * (-y1) - x1 * (-y0)
    return s / 2.0


def _clean(poly: list[Point]) -> tuple[Point, ...] | None:
    """Rounded, without repeated or collinear points, counter-clockwise; ``None`` when nothing with an area is left."""
    pts: list[Point] = []
    for x, y in poly:
        p = (_q(x), _q(y))
        if not (math.isfinite(p[0]) and math.isfinite(p[1])):
            raise SceneError(f"non-finite coordinate {p!r} in a scene polygon")
        if not pts or pts[-1] != p:
            pts.append(p)
    while len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-12:
                pts.pop(i)
                changed = True
                break
    if len(pts) < 3:
        return None
    area = _signed_area(pts)
    if abs(area) < 1e-12:
        return None
    return tuple(pts if area > 0 else reversed(pts))


def _circle(cx: float, cy: float, r: float, n: int) -> list[Point]:
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _capsule(a: Point, b: Point, r: float, n: int = _CAP_SEGMENTS) -> list[Point]:
    """The outline of a segment ``a``-``b`` of half-width ``r`` with round caps (``n`` segments per cap)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return _circle(a[0], a[1], r, 2 * n + 2)
    base = math.atan2(dy, dx)
    pts: list[Point] = []
    for k in range(n + 1):  # cap around b: from base-90 to base+90
        t = base - math.pi / 2 + math.pi * k / n
        pts.append((b[0] + r * math.cos(t), b[1] + r * math.sin(t)))
    for k in range(n + 1):  # cap around a: from base+90 to base+270
        t = base + math.pi / 2 + math.pi * k / n
        pts.append((a[0] + r * math.cos(t), a[1] + r * math.sin(t)))
    return pts


def _bar(a: Point, b: Point, w: float) -> list[Point]:
    """A stroke of width ``w`` from ``a`` to ``b`` with square caps (half a width beyond each end)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    h = w / 2.0
    if length < 1e-9:
        return [(a[0] - h, a[1] - h), (a[0] + h, a[1] - h), (a[0] + h, a[1] + h), (a[0] - h, a[1] + h)]
    ux, uy = dx / length * h, dy / length * h
    nx, ny = -uy, ux
    return [(a[0] - ux + nx, a[1] - uy + ny), (b[0] + ux + nx, b[1] + uy + ny), (b[0] + ux - nx, b[1] + uy - ny), (a[0] - ux - nx, a[1] - uy - ny)]


def _rounded_rect(w: float, h: float, r: float, n: int = _CORNER_SEGMENTS) -> list[Point]:
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    if r <= 1e-9:
        return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts: list[Point] = []
    corners = ((w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, 90.0), (-w / 2 + r, -h / 2 + r, 180.0), (w / 2 - r, -h / 2 + r, 270.0))
    for cx, cy, start in corners:
        for k in range(n + 1):
            t = math.radians(start + 90.0 * k / n)
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


#: custom-pad primitives that are copper (``gr_bbox`` / ``gr_vector`` are editor annotations, not copper)
_PAD_PRIMITIVE_HEADS = frozenset({"gr_poly", "gr_line", "gr_rect", "gr_circle", "gr_arc", "gr_curve"})
_PAD_ANNOTATION_HEADS = frozenset({"gr_bbox", "gr_vector"})


def _arc_extremes(start: Point, mid: Point, end: Point) -> list[Point]:
    """The points that bound the arc ``start`` -> ``mid`` -> ``end``: its ends and every axis extreme of its circle on the arc."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [start, mid, end]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, am, a3 = math.atan2(y1 - uy, x1 - ux), math.atan2(y2 - uy, x2 - ux), math.atan2(y3 - uy, x3 - ux)
    tau = 2 * math.pi
    sweep = (a3 - a1) % tau
    ccw = (am - a1) % tau <= sweep  # mid on the counter-clockwise way (in this frame's angles) from start to end
    out = [start, mid, end]
    for k in range(4):
        t = k * math.pi / 2
        on = (t - a1) % tau <= sweep if ccw else (a1 - t) % tau <= tau - sweep
        if on:
            out.append((ux + r * math.cos(t), uy + r * math.sin(t)))
    return out


def _custom_pad_box(pad: Pad, node: list) -> tuple[list[Point], bool]:
    """The rectangle (pad frame) around a ``custom`` pad's anchor and every copper primitive, each grown by half its width, and
    whether every primitive was read (an unknown primitive head leaves its copper out of the box)."""
    hx, hy = pad.size_w / 2.0, pad.size_h / 2.0
    options = sexpr.find(node, "options")
    if options is not None and str(sexpr.get(options, "anchor", 1, "rect")) == "circle":
        hy = hx  # a circle anchor's diameter is the size's x
    xs, ys = [-hx, hx], [-hy, hy]
    complete = True
    prims = sexpr.find(node, "primitives")
    for prim in (prims[1:] if prims is not None else []):
        head = sexpr.head(prim)
        if head is None or head in _PAD_ANNOTATION_HEADS:
            continue
        if head not in _PAD_PRIMITIVE_HEADS:
            complete = False
            continue
        half = _stroke_width(prim) / 2.0
        pts: list[Point] = []
        if head in ("gr_poly", "gr_curve"):
            pts_node = sexpr.find(prim, "pts")
            for child in (pts_node[1:] if pts_node is not None else []):
                if sexpr.head(child) == "xy":
                    q = _xy(child)
                    if q:
                        pts.append(q)
                elif sexpr.head(child) == "arc":
                    a, m, b = _xy(sexpr.find(child, "start")), _xy(sexpr.find(child, "mid")), _xy(sexpr.find(child, "end"))
                    if a and m and b:
                        pts += _arc_extremes(a, m, b)
        elif head in ("gr_line", "gr_rect"):
            pts = [q for q in (_xy(sexpr.find(prim, "start")), _xy(sexpr.find(prim, "end"))) if q]
        elif head == "gr_circle":
            c, e = _xy(sexpr.find(prim, "center")), _xy(sexpr.find(prim, "end"))
            if c and e:
                r = math.hypot(e[0] - c[0], e[1] - c[1])
                pts = [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]
        else:  # gr_arc
            a, m, b = _xy(sexpr.find(prim, "start")), _xy(sexpr.find(prim, "mid")), _xy(sexpr.find(prim, "end"))
            if a and m and b:
                pts = _arc_extremes(a, m, b)
        if not pts:
            complete = False
            continue
        xs += [q[0] - half for q in pts] + [q[0] + half for q in pts]
        ys += [q[1] - half for q in pts] + [q[1] + half for q in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], complete


def _pad_outline(pad: Pad, node: list | None = None) -> tuple[list[Point], str]:
    """The pad's copper outline in its own frame (library orientation, y down) and how it is drawn: ``""`` exactly (circle / oval /
    rect / roundrect as such; a trapezoid by KiCad's corner formula from ``rect_delta``), ``"bound"`` for a ``custom`` pad (the
    rectangle around its anchor and every copper primitive), ``"anchor"`` when the copper is not known beyond the ``(size)`` box
    (a custom pad with a primitive this module does not read, an unknown shape)."""
    w, h = pad.size_w, pad.size_h
    if pad.shape == "circle":
        return _circle(0.0, 0.0, w / 2.0, _CIRCLE_SIDES), ""
    if pad.shape == "oval":
        return _rounded_rect(w, h, min(w, h) / 2.0, _CAP_SEGMENTS), ""
    if pad.shape == "roundrect":
        ratio = pad.roundrect_rratio if pad.roundrect_rratio is not None else 0.25
        return _rounded_rect(w, h, ratio * min(w, h)), ""
    rect = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    if pad.shape == "rect":
        return rect, ""
    if pad.shape == "trapezoid" and node is not None:
        delta = sexpr.find(node, "rect_delta")
        dx = sexpr.to_float(delta[1]) / 2.0 if delta is not None and len(delta) > 2 else 0.0
        dy = sexpr.to_float(delta[2]) / 2.0 if delta is not None and len(delta) > 2 else 0.0
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise SceneError(f"pad {pad.number!r} has a non-finite rect_delta")
        # KiCad's trapezoid corners (PAD polygon): the delta's x widens / narrows the y sides, its y the x sides
        hx, hy = w / 2.0, h / 2.0
        return [(-hx - dy, hy + dx), (-hx + dy, -hy - dx), (hx - dy, -hy + dx), (hx + dy, hy - dx)], ""
    if pad.shape == "custom" and node is not None:
        box, complete = _custom_pad_box(pad, node)
        return box, "bound" if complete else "anchor"
    return rect, "anchor"


def _arc_points(start: Point, mid: Point, end: Point) -> list[Point]:
    """Points along the circular arc through ``start``, ``mid``, ``end`` (a straight line when they are collinear)."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [start, end]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    am = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)
    sweep = (a3 - a1) % (2 * math.pi)
    if (am - a1) % (2 * math.pi) > sweep:  # mid is not on the counter-clockwise way: go the other way round
        sweep -= 2 * math.pi
    n = max(2, math.ceil(abs(sweep) / _ARC_STEP_RAD))
    return [(ux + r * math.cos(a1 + sweep * k / n), uy + r * math.sin(a1 + sweep * k / n)) for k in range(n + 1)]


def _xy(node: list | None) -> Point | None:
    if node is None or len(node) < 3:
        return None
    x, y = sexpr.to_float(node[1]), sexpr.to_float(node[2])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise SceneError(f"non-finite ({node[0]} ...) in a footprint graphic")
    return (x, y)


def _graphic_paths(item: list) -> tuple[list[list[Point]], list[list[Point]]]:
    """``(open/closed polylines, filled polygons)`` of a footprint graphic, in the footprint frame (y down)."""
    kind = sexpr.head(item)
    fill = str(sexpr.get(item, "fill", 1, "no")) in ("yes", "solid")
    lines: list[list[Point]] = []
    fills: list[list[Point]] = []
    if kind == "fp_line":
        a, b = _xy(sexpr.find(item, "start")), _xy(sexpr.find(item, "end"))
        if a and b:
            lines.append([a, b])
    elif kind == "fp_rect":
        a, b = _xy(sexpr.find(item, "start")), _xy(sexpr.find(item, "end"))
        if a and b:
            ring = [a, (b[0], a[1]), b, (a[0], b[1]), a]
            lines.append(ring)
            if fill:
                fills.append(ring[:-1])
    elif kind == "fp_arc":
        a, m, b = _xy(sexpr.find(item, "start")), _xy(sexpr.find(item, "mid")), _xy(sexpr.find(item, "end"))
        if a and m and b:
            lines.append(_arc_points(a, m, b))
    elif kind == "fp_circle":
        c, e = _xy(sexpr.find(item, "center")), _xy(sexpr.find(item, "end"))
        if c and e:
            r = math.hypot(e[0] - c[0], e[1] - c[1])
            n = max(12, math.ceil(2 * math.pi / _ARC_STEP_RAD))
            ring = _circle(c[0], c[1], r, n)
            lines.append(ring + [ring[0]])
            if fill:
                fills.append(ring)
    elif kind == "fp_poly":
        pts_node = sexpr.find(item, "pts")
        ring: list[Point] = []
        if pts_node is not None:
            for child in pts_node[1:]:
                h = sexpr.head(child)
                if h == "xy":
                    p = _xy(child)
                    if p:
                        ring.append(p)
                elif h == "arc":
                    a, m, b = _xy(sexpr.find(child, "start")), _xy(sexpr.find(child, "mid")), _xy(sexpr.find(child, "end"))
                    if a and m and b:
                        ring.extend(_arc_points(a, m, b))
        if len(ring) >= 2:
            lines.append(ring + [ring[0]])
            if fill and len(ring) >= 3:
                fills.append(ring)
    elif kind == "fp_curve":
        pts_node = sexpr.find(item, "pts")
        ctrl = [p for p in (_xy(c) for c in sexpr.find_all(pts_node, "xy")) if p] if pts_node is not None else []
        if len(ctrl) == 4:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = ctrl
            curve = []
            for k in range(_BEZIER_SEGMENTS + 1):
                t = k / _BEZIER_SEGMENTS
                u = 1 - t
                curve.append((u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3, u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3))
            lines.append(curve)
    return lines, fills


_GRAPHIC_HEADS = ("fp_line", "fp_rect", "fp_arc", "fp_circle", "fp_poly", "fp_curve")
#: the footprint graphics read as drawable paths (the report's board figure draws the same silk strokes as the scene)
GRAPHIC_HEADS = _GRAPHIC_HEADS
graphic_paths = _graphic_paths


def _stroke_width(item: list) -> float:
    stroke = sexpr.find(item, "stroke")
    width = sexpr.get(stroke, "width") if stroke is not None else sexpr.get(item, "width")
    try:
        value = sexpr.to_float(width) if width is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


stroke_width = _stroke_width


def _layer_box(fp: FootprintDef, layer: str) -> tuple[float, float, float, float] | None:
    """The box (footprint frame) around every graphic on ``layer`` (texts excluded)."""
    xs: list[float] = []
    ys: list[float] = []
    for item in fp.node:
        if sexpr.head(item) not in _GRAPHIC_HEADS or sexpr.get(item, "layer") != layer:
            continue
        lines, fills = _graphic_paths(item)
        for path in lines + fills:
            for x, y in path:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _has_area(box: tuple[float, float, float, float] | None) -> bool:
    return box is not None and box[2] - box[0] > 1e-6 and box[3] - box[1] > 1e-6


# --------------------------------------------------------------------------- the builder


class _Builder:
    def __init__(self, ir: CircuitIR, library: KicadLibrary, model_dir: Path | None, step_cache: dict[Path, StepEnvelope]) -> None:
        pcb = ir.pcb
        if pcb is None or pcb.outline is None:
            raise SceneError("ir.pcb.outline is None: a 3D scene needs the board outline the PCB compiler needs")
        o = pcb.outline
        if not (o.width_mm > 0 and o.height_mm > 0) or not all(math.isfinite(v) for v in (o.width_mm, o.height_mm, o.origin_x_mm, o.origin_y_mm)):
            raise SceneError(f"ir.pcb.outline must have a positive, finite size (got {o.width_mm} x {o.height_mm})")
        traced = pcb.manufacturing.board_thickness_mm
        grounded = traced is not None and traced.provenance.is_authoritative
        thickness = float(traced.value) if grounded and traced is not None else DEFAULT_BOARD_THICKNESS_MM
        if not (math.isfinite(thickness) and thickness > 0):
            raise SceneError(f"board thickness {thickness!r} is not a positive finite number")
        self.ir = ir
        self.library = library
        self.model_dir = model_dir
        self.cache = step_cache
        self.T = thickness
        self.scene = Scene(ir.project.id, (o.origin_x_mm, o.origin_y_mm, o.width_mm, o.height_mm), thickness, grounded, model_dir_found=model_dir is not None)
        self.approximated_pads = 0  # custom pads drawn as the rectangle around their anchor and primitives
        self.anchor_only_pads = 0  # pads whose copper is not known beyond their (size) box

    # --- emitters ---------------------------------------------------------------------------------

    def solid(self, kind: str, material: str, polygon: list[Point], z0: float, z1: float, faces: str, label: str) -> None:
        assert material in _MATERIAL_KEYS
        clean = _clean(polygon)
        if clean is not None:
            self.scene.solids.append(Solid(kind, material, clean, _q(z0), _q(z1), faces, label))

    def decal(self, kind: str, material: str, polygon: list[Point], top: bool, level: float, label: str) -> None:
        if top:
            self.solid(kind, material, polygon, self.T, self.T + level, "top", label)
        else:
            self.solid(kind, material, polygon, -level, 0.0, "bottom", label)

    def drill(self, cx: float, cy: float, diameter: float, label: str) -> None:
        if diameter > 0 and math.isfinite(diameter):
            self.solid("drill", "drill", _circle(cx, cy, diameter / 2.0, _DRILL_SIDES), -DRILL_OVER_MM, self.T + DRILL_OVER_MM, "all", label)

    # --- board --------------------------------------------------------------------------------------

    def board(self) -> None:
        x, y, w, h = self.scene.outline
        rect = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        drawn = self.planes()
        self.solid("slab", "board_clear" if drawn else "board", rect, 0.0, self.T, "all", "board")
        self.decal("mask", "mask", rect, True, MASK_MM, "F.Mask")
        self.decal("mask", "mask", rect, False, MASK_MM, "B.Mask")

    def planes(self) -> bool:
        """The inner-layer zones as translucent sheets at their stackup depth (module docstring); whether any was drawn."""
        pcb = self.ir.pcb
        assert pcb is not None
        inner = [z for z in pcb.zones if INNER_LAYER_RE.match(z.layer)]
        if not inner:
            return False
        stack = pcb.stackup
        depth: dict[str, tuple[float, float]] = {}  # layer -> (top, bottom) depth below the top face, mm
        if stack is not None:
            d = 0.0
            for i, c in enumerate(stack.copper):
                t = float(c.thickness_um.value) / 1000.0
                depth[c.name] = (d, d + t)
                d += t + (float(stack.dielectrics[i].thickness_mm.value) if i < len(stack.dielectrics) else 0.0)
            total = stack.board_thickness_mm()
            k = self.T / total if total > 0 else 1.0
            depth = {name: (a * k, b * k) for name, (a, b) in depth.items()}
        drawn: list[str] = []
        skipped: list[str] = []
        for z in inner:
            span = depth.get(z.layer)
            if span is None:
                skipped.append(f"{z.net} ({z.layer})")
                continue
            z_top, z_bottom = self.T - span[0], self.T - span[1]
            poly = [(float(px), float(py)) for px, py in z.polygon]
            label = f"{z.net} {z.layer}"
            self.solid("plane", "plane", poly, z_bottom, z_top, "top", label)
            self.solid("plane", "plane", poly, z_bottom, z_top, "bottom", label)
            drawn.append(f"{z.layer} {z.net}")
        if drawn:
            scaled = "" if stack is None or math.isclose(stack.board_thickness_mm(), self.T, abs_tol=1e-6) else f"; 적층 두께 {stack.board_thickness_mm():g} mm 를 보드 두께 {self.T:g} mm 에 맞춰 비례"
            self.scene.notes.append(
                f"내층 평면 {', '.join(drawn)}: 영역 외곽을 적층의 깊이에 반투명 구리 판으로 그림 (채움과 다른 넷 둘레의 빈틈은 KiCad 가 계산하므로 그리지 않음{scaled}); "
                "내층이 보이도록 보드 판을 반투명으로 그림"
            )
        if skipped:
            self.scene.notes.append(f"내층 영역 {', '.join(skipped)} 은 적층(ir.pcb.stackup)이 없어 깊이를 몰라 그리지 않음")
        return bool(drawn)

    def copper(self) -> None:
        pcb = self.ir.pcb
        assert pcb is not None
        inner = 0
        for t in pcb.tracks:
            if t.layer not in ("F.Cu", "B.Cu"):
                inner += 1
                continue
            if not (t.width_mm > 0 and math.isfinite(t.width_mm)):
                raise SceneError(f"track of {t.net!r} has a non-positive width {t.width_mm!r}")
            self.decal("track", "copper", _capsule(t.start, t.end, t.width_mm / 2.0), t.layer == "F.Cu", COPPER_MM, t.net)
        for v in pcb.vias:
            for layer in ("F.Cu", "B.Cu"):
                if layer in v.layers:
                    self.decal("via", "copper", _circle(v.x_mm, v.y_mm, v.diameter_mm / 2.0, _VIA_SIDES), layer == "F.Cu", COPPER_MM, v.net)
            self.drill(v.x_mm, v.y_mm, v.drill_mm, v.net)
        if inner:
            self.scene.notes.append(f"안쪽 층 트랙 {inner}개는 보드 속에 있어 그리지 않음")
        outer = [z for z in pcb.zones if not INNER_LAYER_RE.match(z.layer)]
        if outer:
            self.scene.notes.append(f"구리 영역(zone) {len(outer)}개는 채움 계산이 KiCad 의 몫이라 그리지 않음")

    # --- components -------------------------------------------------------------------------------

    def components(self) -> None:
        pcb = self.ir.pcb
        assert pcb is not None
        for comp in self.ir.components:
            if comp.footprint is None:
                raise SceneError(f"component {comp.ref!r} has no footprint; the 3D scene never picks one")
            if not self.library.resolve_footprint(comp.footprint).verified:
                raise SceneError(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} was not found in a KiCad library")
            placement = pcb.placement(comp.ref)
            if placement is None:
                raise SceneError(f"component {comp.ref!r} has no placement in ir.pcb.placements")
            fp = self.library.load_footprint(comp.footprint)
            self.pads(comp.ref, placement, fp)
            self.silk(comp.ref, placement, fp)
            self.body(comp.ref, placement, fp)
        if self.approximated_pads:
            self.scene.notes.append(f"사용자 정의 모양 패드 {self.approximated_pads}개는 앵커와 모든 구리 도형을 감싸는 사각형(외접 사각형)으로 그림")
        if self.anchor_only_pads:
            self.scene.notes.append(f"구리 모양을 읽지 못한 패드 {self.anchor_only_pads}개는 (size) 사각형으로만 그림 - 실제 구리는 더 넓을 수 있음")

    def pads(self, ref: str, placement: Placement, fp: FootprintDef) -> None:
        nodes = sexpr.find_all(fp.node, "pad")  # the library reader builds fp.pads from these nodes, in this order
        mirror = placement.side == BoardSide.BOTTOM  # a flipped pad's own shape mirrors in y (PAD::Flip: rect_delta / primitives)
        for pad, node in zip(fp.pads, nodes):
            cx, cy = pad_center(placement, pad)
            layers = pad_layers(placement, pad)
            if pad.drill:
                self.drill(cx, cy, pad.drill, f"{ref}.{pad.number}")
            if pad.pad_type == "np_thru_hole":
                continue
            cx, cy = pad_copper_center(placement, pad)  # the copper sits at the drill offset; the hole stays at (at)
            local, drawn_as = _pad_outline(pad, node)
            if mirror:
                local = [(u, -v) for u, v in local]
            angle = pad_angle(placement, pad)
            outline = [(cx + dx, cy + dy) for dx, dy in (rotate(u, v, angle) for u, v in local)]
            drawn = False
            for top, cu, mask in ((True, "F.Cu", "F.Mask"), (False, "B.Cu", "B.Mask")):
                if cu in layers or "*.Cu" in layers:
                    exposed = mask in layers or "*.Mask" in layers
                    self.decal("pad", "pad" if exposed else "copper", outline, top, PAD_MM if exposed else COPPER_MM, f"{ref}.{pad.number}")
                    drawn = True
            if drawn and drawn_as == "bound":
                self.approximated_pads += 1
            elif drawn and drawn_as == "anchor":
                self.anchor_only_pads += 1

    def silk(self, ref: str, placement: Placement, fp: FootprintDef) -> None:
        for item in fp.node:
            if sexpr.head(item) not in _GRAPHIC_HEADS:
                continue
            layer = sexpr.get(item, "layer")
            if layer not in ("F.SilkS", "B.SilkS"):
                continue
            top = mirrored_layer(str(layer), placement.side) == "F.SilkS"
            width = _stroke_width(item)
            lines, fills = _graphic_paths(item)
            for poly in fills:
                self.decal("silk", "silk", [to_board(placement, x, y) for x, y in poly], top, SILK_MM, ref)
            if width <= 0:
                continue
            for path in lines:
                pts = [to_board(placement, x, y) for x, y in path]
                for a, b in zip(pts, pts[1:]):
                    self.decal("silk", "silk", _bar(a, b, width), top, SILK_MM, ref)

    def envelope(self, model: ModelRef) -> tuple[Box3 | None, str]:
        if not model.is_step:
            return None, "STEP 모델이 아님"
        path, why = resolve_model_path(model.path, self.model_dir)
        if path is None:
            return None, why
        env = self.cache.get(path)
        if env is None:
            env = read_step_envelope(path)
            self.cache[path] = env
        if env.box is None:
            return None, env.reason
        return transformed_box(model, env.box), ""

    def body(self, ref: str, placement: Placement, fp: FootprintDef) -> None:
        side = "bottom" if placement.side == BoardSide.BOTTOM else "top"
        box: tuple[float, float, float, float] | None = _layer_box(fp, "F.Fab")
        source = "F.Fab"
        if not _has_area(box) and fp.courtyard is not None:
            c = fp.courtyard
            box, source = (c.x1, c.y1, c.x2, c.y2), "courtyard"
        if not _has_area(box):
            box = None
        try:
            models = footprint_models(fp)
        except ValueError as exc:
            raise SceneError(str(exc)) from exc
        visible = [m for m in models if not m.hidden]
        written = tuple(m.path for m in models)
        if box is None:
            self.scene.bodies.append(BodyInfo(ref, fp.lib_id, side, "", written, None, (), "면적 있는 F.Fab 도 코트야드도 없어 상자를 만들 수 없음"))
            return
        envelopes: list[tuple[str, Box3]] = []
        reasons: list[str] = []
        for m in visible:
            env, why = self.envelope(m)
            if env is None:
                reasons.append(why)
            else:
                envelopes.append((m.path, env))
        height = max((e.z2 for _, e in envelopes), default=None)
        reason = ""
        if not visible:
            reason = "풋프린트에 (보이는) 3D 모델 참조가 없음" if not models else "3D 모델이 모두 숨김(hide)"
        elif height is None:
            reason = "; ".join(dict.fromkeys(reasons))
        elif height <= 0:
            reason, height = "STEP 외곽 상자가 보드 면 위로 올라오지 않음", None
        x1, y1, x2, y2 = box
        corners = [to_board(placement, x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        top = side == "top"
        if height is not None:
            h = _q(height)
            self.solid("body", "body", corners, self.T if top else -h, self.T + h if top else 0.0, "all", ref)
        else:
            for a, b in zip(corners, corners[1:] + corners[:1]):
                self.decal("outline", "outline", _bar(a, b, OUTLINE_STROKE_MM), top, OUTLINE_MM, ref)
        self.scene.bodies.append(BodyInfo(ref, fp.lib_id, side, source, written, _q(height) if height is not None else None, tuple(envelopes), reason))

    def notes(self) -> None:
        notes = self.scene.notes
        if self.scene.thickness_grounded:
            head = f"보드 두께 {self.T:g} mm (제조 조건)"
        else:
            head = f"보드 두께 {self.T:g} mm (가정: 제조 조건에 확정된 두께 없음)"
        assert self.ir.pcb is not None
        # the IR's designed silk texts (ir.pcb.silkscreen) are counted, never drawn: their positions are the 2-D views' business
        silk_texts = [t for t in self.ir.pcb.silkscreen if t.layer in ("F.SilkS", "B.SilkS")]
        text_note = "실크 문자는 3D 에서 생략 (풋프린트의 문자" + (f", IR 실크 문자 {len(silk_texts)}개" if silk_texts else "") + "); 실크 선은 풋프린트 라이브러리의 것"
        missing = self.scene.bodies_without_step
        lines = [head, BODY_CAPTION, text_note, "구리·마스크·실크 높이는 그림용 값 (측정 아님)"]
        if missing:
            by_reason: dict[str, list[str]] = {}
            for b in missing:
                by_reason.setdefault(b.reason, []).append(b.ref)
            for why, refs in by_reason.items():
                lines.append(f"STEP 높이 없음 ({why}): {', '.join(refs)} - 평면 외곽선만")
        self.scene.notes = lines + notes


def build_scene(ir: CircuitIR, library: KicadLibrary, *, model_dir: Path | None | EllipsisType = ..., step_cache: dict[Path, StepEnvelope] | None = None) -> Scene:
    """The 3D preview scene of ``ir`` (module docstring). ``model_dir`` defaults to :func:`~ai_eda.tools.model3d.models.find_3dmodel_dir`
    (pass ``None`` for no 3D library: every body is a flat outline); ``step_cache`` lets a caller reuse parsed STEP files between builds.

    Refuses (:class:`SceneError`) what the PCB compiler refuses: no outline, a component without a footprint on disk or a placement,
    a non-finite coordinate, a non-positive track width. Builds nothing into the IR.
    """
    directory = find_3dmodel_dir(library) if model_dir is ... else model_dir
    builder = _Builder(ir, library, directory, step_cache if step_cache is not None else {})
    builder.board()
    builder.copper()
    builder.components()
    builder.notes()
    return builder.scene


def scene_caption(scene: Scene) -> str:
    """The Korean caption of any view of ``scene``: size, body counts, the body rule and the assumptions (no path, no time)."""
    _, _, w, h = scene.outline
    with_step, without = len(scene.bodies_with_step), len(scene.bodies_without_step)
    parts = [f"{scene.project_id}: 보드 {w:g} × {h:g} × {scene.thickness_mm:g} mm", f"부품 상자 {with_step}개 (STEP 높이), 평면 외곽선 {without}개"]
    parts += scene.notes
    return "; ".join(parts) + ". 그림은 보드의 유효성을 판정하지 않습니다."


# --------------------------------------------------------------------------- faces (shared by the writers)


def triangulate(polygon: tuple[Point, ...] | list[Point]) -> list[tuple[int, int, int]]:
    """Ear clipping of a simple counter-clockwise (y-up) polygon given in board coordinates (y down); triangles counter-clockwise, deterministic."""
    pts = [(x, -y) for x, y in polygon]
    idx = list(range(len(pts)))
    tris: list[tuple[int, int, int]] = []

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12 and cross(c, a, p) >= -1e-12

    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-15:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:  # not simple (self-touching): a fan of what is left keeps every point drawn
            tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1)]
            return tris
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def solid_faces(solid: Solid, *, sides: bool = True) -> list[Face]:
    """The faces of ``solid`` in KiCad's 3D frame (x, y up, z); a decal has only its outer face; ``sides=False`` leaves the side walls out."""
    ring = [(x, -y) for x, y in solid.polygon]
    faces: list[Face] = []
    if solid.faces in ("all", "top"):
        faces.append(Face((0.0, 0.0, 1.0), tuple((x, y, solid.z1) for x, y in ring), "top"))
    if solid.faces in ("all", "bottom"):
        faces.append(Face((0.0, 0.0, -1.0), tuple((x, y, solid.z0) for x, y in reversed(ring)), "bottom"))
    if solid.faces == "all" and sides:
        n = len(ring)
        for i in range(n):
            (ax, ay), (bx, by) = ring[i], ring[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            normal = (dy / length, -dx / length, 0.0)
            faces.append(Face(normal, ((ax, ay, solid.z0), (bx, by, solid.z0), (bx, by, solid.z1), (ax, ay, solid.z1)), "side"))
    return faces
