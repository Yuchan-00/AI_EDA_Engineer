"""Board-frame geometry of placed footprints (pure functions, no I/O).

Invariant: every number produced here is *derived* from a footprint that was
read from a KiCad library (:class:`~ai_eda.tools.kicad.library.FootprintDef`
/ :class:`~ai_eda.tools.kicad.library.Pad`) plus an IR
:class:`~ai_eda.ir.Placement`. Nothing is estimated.

KiCad conventions encoded here (verified against kicad-cli 10.0.6 DRC runs
and KiCad-saved demo boards):

* The board frame is millimetres, **Y down**. A positive footprint rotation
  ``a`` is counter-clockwise on screen, so a footprint-relative offset
  ``(px, py)`` lands at ``x = fx + px*cos(a) + py*sin(a)``,
  ``y = fy - px*sin(a) + py*cos(a)`` (a pad below the centre moves to the
  right at +90 degrees).
* A ``.kicad_pcb`` stores pad / graphic coordinates **relative and unrotated**
  (the library values); only angles are combined with the footprint rotation.
* **Bottom side** (``layer "B.Cu"``): KiCad flips a footprint by mirroring it
  about its own X axis, so the stored relative ``y`` is negated, every
  ``F.*`` layer becomes ``B.*`` (and vice versa; ``*.Cu`` / ``*.Mask`` stay),
  the stored pad angle is ``rotation - library pad angle`` (``PAD::Flip``
  negates the pad orientation), text angles become
  ``180 - library angle + rotation`` (``PCB_TEXT::Flip``) with
  ``(justify mirror)``. The loader applies no further mirroring, so the
  absolute-position formula above applies unchanged to the *stored* offsets.

All results are rounded to 1e-6 mm, KiCad's internal resolution.

A box is only ever built from finite points: ``min`` / ``max`` drop a NaN
silently and an infinite coordinate makes a box that is everywhere, so a
non-finite pad or courtyard coordinate (the library loader refuses them at
the source; a hand-built :class:`FootprintDef` could still carry one) is a
:class:`~ai_eda.errors.CompileError` naming the footprint, never a smaller
extent.

**Custom pads** (:func:`custom_pad_parts`): the one extent every user of a
``custom`` pad's copper sees - the router, the ``pcb.routing.*`` /
``pcb.keepout`` checks, the silkscreen, the SI path extraction, the 3D scene
and the board figures. The copper is bounded by a list of axis-aligned
board-frame boxes (:class:`PadPart`): first the **anchor** (the ``(size)``
rectangle - its rotated corners' box - or the circle of diameter ``size`` x,
at the pad's copper centre :func:`pad_copper_center`), then one box per
copper primitive (the library's :class:`~ai_eda.tools.kicad.library.PadPrimitive`,
placed like KiCad places it: relative to the copper centre, mirrored in its
own y on the bottom side, rotated by :func:`pad_angle`) around its defining
points - a polygon's vertices (and the extremes of each ``arc`` piece of its
outline), a line's ends, a rectangle's four corners, a circle's centre +/-
its radius, an arc's ends and every axis extreme of its circle it passes, a
Bezier curve's four control points (the curve lies in their hull) - grown by
half its stroke width. Every box contains the copper it stands for, so the
union is an outer bound: never smaller than the copper. A part is ``exact``
only when its box *is* copper (a rect anchor, or a filled zero-width
axis-aligned rectangle - ``gr_rect`` or a four-vertex ``gr_poly`` - at a
quarter turn), and ``inscribed_r`` is the radius of a disc around the box
centre that is copper for sure (a rect anchor ``min(w, h)/2``, a circle
anchor its radius, an exact part ``min`` of its half sides, a filled circle
primitive its radius plus half its stroke; otherwise 0: nothing about the
inside is claimed). A pad whose primitives were not all read
(:attr:`~ai_eda.tools.kicad.library.Pad.unread_primitives`) is refused
(:class:`CompileError` naming them): its copper is unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, Placement
from ai_eda.tools.kicad.library import BBox, FootprintDef, Pad

__all__ = [
    "PadPart",
    "custom_pad_parts",
    "arc_extremes",
    "RESOLUTION_DECIMALS",
    "normalize_angle",
    "footprint_angle",
    "rotate",
    "stored_offset",
    "to_board",
    "pad_center",
    "pad_copper_center",
    "pad_angle",
    "text_angle",
    "mirrored_layer",
    "pad_layers",
    "courtyard_bbox",
    "pads_bbox",
    "footprint_bbox",
    "finite_bbox",
]

#: KiCad's internal unit is 1 nm = 1e-6 mm.
RESOLUTION_DECIMALS = 6


def _q(v: float) -> float:
    """Round to KiCad resolution and normalise ``-0.0`` to ``0.0``."""
    return round(v, RESOLUTION_DECIMALS) + 0.0


def normalize_angle(deg: float) -> float:
    """Angle in ``[0, 360)`` as KiCad stores orientations (``-90`` -> ``270``)."""
    a = math.fmod(deg, 360.0)
    if a < 0:
        a += 360.0
    a = _q(a)
    return 0.0 if a >= 360.0 else a


def footprint_angle(deg: float) -> float:
    """Footprint orientation as KiCad stores it in ``(at x y rot)``: ``(-180, 180]``.

    ``FOOTPRINT::SetOrientation`` normalises to this range (270 is written
    ``-90``); pads and texts use :func:`normalize_angle` (``[0, 360)``) instead.
    """
    a = normalize_angle(deg)
    return a - 360.0 if a > 180.0 else a


def rotate(px: float, py: float, angle_deg: float) -> tuple[float, float]:
    """Rotate a board-frame (y-down) offset by ``angle_deg`` counter-clockwise on screen."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return _q(px * c + py * s), _q(-px * s + py * c)


def stored_offset(px: float, py: float, side: BoardSide) -> tuple[float, float]:
    """The relative offset as a ``.kicad_pcb`` stores it: ``y`` negated on the bottom side."""
    return (_q(px), _q(-py)) if side == BoardSide.BOTTOM else (_q(px), _q(py))


def to_board(placement: Placement, px: float, py: float) -> tuple[float, float]:
    """Absolute board position of a library-frame offset ``(px, py)`` of ``placement``."""
    sx, sy = stored_offset(px, py, placement.side)
    rx, ry = rotate(sx, sy, placement.rotation_deg)
    return _q(placement.x_mm + rx), _q(placement.y_mm + ry)


def pad_center(placement: Placement, pad: Pad) -> tuple[float, float]:
    """Absolute position of ``pad`` - its ``(at)``, where the hole is - (library frame, y down) once its footprint is placed."""
    return to_board(placement, pad.x, pad.y)


def pad_copper_center(placement: Placement, pad: Pad) -> tuple[float, float]:
    """Absolute centre of ``pad``'s copper shape: its ``(at)`` plus the ``(drill (offset x y))`` rotated by the library pad angle
    (KiCad's ``PAD::ShapePos``; mirrored with the footprint on the bottom side, as ``PAD::Flip`` mirrors the offset). The same
    point as :func:`pad_center` for a pad without an offset."""
    if not pad.offset_x and not pad.offset_y:
        return pad_center(placement, pad)
    ox, oy = rotate(pad.offset_x, pad.offset_y, pad.rotation)
    return to_board(placement, pad.x + ox, pad.y + oy)


def pad_angle(placement: Placement, pad: Pad) -> float:
    """Pad orientation as stored in the board file (``[0, 360)``)."""
    if placement.side == BoardSide.BOTTOM:
        return normalize_angle(placement.rotation_deg - pad.rotation)
    return normalize_angle(placement.rotation_deg + pad.rotation)


def text_angle(placement: Placement, library_angle: float) -> float:
    """Orientation of a footprint text / property as stored in the board file."""
    if placement.side == BoardSide.BOTTOM:
        return normalize_angle(180.0 - library_angle + placement.rotation_deg)
    return normalize_angle(library_angle + placement.rotation_deg)


def mirrored_layer(layer: str, side: BoardSide) -> str:
    """``F.SilkS`` <-> ``B.SilkS`` on the bottom side; wildcard layers (``*.Cu``) unchanged."""
    if side != BoardSide.BOTTOM:
        return layer
    if layer.startswith("F."):
        return "B." + layer[2:]
    if layer.startswith("B."):
        return "F." + layer[2:]
    return layer


def pad_layers(placement: Placement, pad: Pad) -> list[str]:
    return [mirrored_layer(layer, placement.side) for layer in pad.layers]


def _bbox_of_points(points: list[tuple[float, float]], what: str) -> BBox | None:
    """Axis-aligned box around ``points``; ``None`` for no points; :class:`CompileError` when a coordinate is not finite (module docstring)."""
    if not points:
        return None
    bad = [pt for pt in points if not (math.isfinite(pt[0]) and math.isfinite(pt[1]))]
    if bad:
        raise CompileError(f"{what} has a non-finite coordinate ({bad[0][0]!r}, {bad[0][1]!r}); its extent cannot be measured")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(_q(min(xs)), _q(min(ys)), _q(max(xs)), _q(max(ys)))


def finite_bbox(box: BBox, what: str) -> BBox:
    """``box`` when all four coordinates are finite, else :class:`CompileError` naming ``what``."""
    if not all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2)):
        raise CompileError(f"{what} has a non-finite extent ({box}); refusing to use it")
    return box


def courtyard_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Axis-aligned board-frame box around the placed courtyard (None when the footprint has none).

    Exact for rotations that are multiples of 90 degrees; for other angles it
    is the box around the rotated rectangle's corners (conservative).
    """
    cy = fp.courtyard
    if cy is None:
        return None
    corners = [(cy.x1, cy.y1), (cy.x2, cy.y1), (cy.x2, cy.y2), (cy.x1, cy.y2)]
    return _bbox_of_points([to_board(placement, x, y) for x, y in corners], f"footprint {fp.lib_id} courtyard")


def pads_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Board-frame box around all pad copper (exact for pads at multiples of 90 degrees, else conservative); a pad's copper sits at
    :func:`pad_copper_center` (its drill offset applied). A ``custom`` pad whose primitives were all read counts with every
    :func:`custom_pad_parts` box (its anchor and primitives); one with an unread primitive only with its ``(size)`` box, as before."""
    points: list[tuple[float, float]] = []
    for pad in fp.pads:
        if pad.shape == "custom" and not pad.unread_primitives:
            for part in custom_pad_parts(placement, pad):
                points += [(part.box.x1, part.box.y1), (part.box.x2, part.box.y2)]
            continue
        cx, cy = pad_copper_center(placement, pad)
        angle = pad_angle(placement, pad)
        if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):
            w, h = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
            hx, hy = w / 2.0, h / 2.0
        else:
            hx = hy = math.hypot(pad.size_w, pad.size_h) / 2.0
        points += [(cx - hx, cy - hy), (cx + hx, cy + hy)]
    return _bbox_of_points(points, f"footprint {fp.lib_id} pads")


def footprint_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Union of :func:`courtyard_bbox` and :func:`pads_bbox` (None when neither exists); never a box with a non-finite coordinate."""
    boxes = [b for b in (courtyard_bbox(placement, fp), pads_bbox(placement, fp)) if b is not None]
    if not boxes:
        return None
    return finite_bbox(BBox(min(b.x1 for b in boxes), min(b.y1 for b in boxes), max(b.x2 for b in boxes), max(b.y2 for b in boxes)), f"footprint {fp.lib_id}")


# --------------------------------------------------------------------------- custom pads


@dataclass(frozen=True, slots=True)
class PadPart:
    """One box of a pad's copper bound in the board frame (module docstring: custom pads).

    ``anchor``: the pad's anchor, where a track may land; ``exact``: the box is copper exactly; ``inscribed_r``: a disc of this
    radius around the box centre is copper for sure (0: nothing about the inside is claimed); ``what``: ``"anchor rect"`` /
    ``"anchor circle"`` / ``"gr_poly[0]"`` ... (the primitive's kind and index in the file); ``disc``: ``(x, y, r)`` of a disc that
    bounds the part's copper more tightly than its box - a circle anchor, a ``gr_circle`` grown by half its stroke (a ring's disc
    covers its hole too) - for a user that measures exact shapes (the silkscreen; ``pcb.routing``'s clearance and connectivity, where copper
    clear of the disc is clear of the part's copper); ``None`` otherwise."""

    box: BBox
    anchor: bool
    exact: bool
    inscribed_r: float
    what: str
    disc: tuple[float, float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.box.x1 + self.box.x2) / 2.0, (self.box.y1 + self.box.y2) / 2.0


def arc_extremes(start: tuple[float, float], mid: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
    """The points that bound the arc ``start`` -> ``mid`` -> ``end``: its ends, its mid point and every axis extreme of its circle the
    arc passes (the three points alone when they are collinear: a straight piece)."""
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


def _axis_rectangle(points: tuple[tuple[float, float], ...]) -> bool:
    """Whether four points are the corners of an axis-aligned rectangle with an area."""
    xs, ys = sorted({p[0] for p in points}), sorted({p[1] for p in points})
    return len(points) == 4 and len(xs) == 2 and len(ys) == 2 and set(points) == {(x, y) for x in xs for y in ys}


def custom_pad_parts(placement: Placement, pad: Pad) -> list[PadPart]:
    """The boxes that bound a ``custom`` pad's copper in the board frame, the anchor first (module docstring: custom pads).

    Raises :class:`CompileError` for a pad that is not ``custom``, one with a primitive the library reader did not read, or one
    whose geometry is not finite."""
    if pad.shape != "custom":
        raise CompileError(f"pad {pad.number!r} has shape {pad.shape!r}, not 'custom'")
    if pad.unread_primitives:
        raise CompileError(
            f"custom pad {pad.number or '(unnumbered)'!r} has primitive(s) {', '.join(pad.unread_primitives)} the library reader does not read: "
            "its copper is unknown"
        )
    cx, cy = pad_copper_center(placement, pad)
    angle = pad_angle(placement, pad)
    quarter = math.isclose(angle % 90.0, 0.0, abs_tol=1e-9)
    mirror = placement.side == BoardSide.BOTTOM  # PAD::Flip mirrors the pad's own shape (its primitives) in its y

    def board(u: float, v: float) -> tuple[float, float]:
        rx, ry = rotate(u, -v if mirror else v, angle)
        return _q(cx + rx), _q(cy + ry)

    def box_of(points: list[tuple[float, float]], grow: float, what: str) -> BBox:
        got = _bbox_of_points(points, f"custom pad {pad.number!r} {what}")
        assert got is not None
        return BBox(_q(got.x1 - grow), _q(got.y1 - grow), _q(got.x2 + grow), _q(got.y2 + grow))

    parts: list[PadPart] = []
    if pad.anchor == "circle":
        r = pad.size_w / 2.0  # KiCad: a circle anchor's diameter is the size's x
        parts.append(PadPart(box_of([(cx, cy)], r, "anchor"), True, False, r, "anchor circle", (cx, cy, r)))
    else:
        hw, hh = pad.size_w / 2.0, pad.size_h / 2.0
        corners = [board(u, v) for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        parts.append(PadPart(box_of(corners, 0.0, "anchor"), True, quarter, min(hw, hh), "anchor rect"))
    for k, prim in enumerate(pad.primitives):
        what = f"gr_{prim.kind}[{k}]"
        half = prim.width / 2.0
        exact = False
        inscribed = 0.0
        disc = None
        if prim.kind == "circle":
            (ux, uy), (ex, ey) = prim.points
            r = math.hypot(ex - ux, ey - uy)
            centre = board(ux, uy)
            box = box_of([centre], r + half, what)
            inscribed = r + half if prim.fill else 0.0
            disc = (centre[0], centre[1], _q(r + half))
        elif prim.kind == "arc":
            box = box_of(arc_extremes(*(board(u, v) for u, v in prim.points)), half, what)  # type: ignore[arg-type]
        elif prim.kind == "rect":
            (sx, sy), (ex, ey) = prim.points
            box = box_of([board(u, v) for u, v in ((sx, sy), (ex, sy), (ex, ey), (sx, ey))], half, what)
            exact = prim.fill and prim.width == 0.0 and quarter and sx != ex and sy != ey
        else:  # poly / line / curve: the vertices (a curve inside the hull of its control points), a polygon's arc pieces by extremes
            pts = [board(u, v) for u, v in prim.points]
            for a, m, b in prim.arcs:
                pts += arc_extremes(board(*a), board(*m), board(*b))
            box = box_of(pts, half, what)
            exact = prim.kind == "poly" and prim.fill and prim.width == 0.0 and quarter and not prim.arcs and _axis_rectangle(prim.points)
        if exact:
            inscribed = min(box.width, box.height) / 2.0
        parts.append(PadPart(box, False, exact, inscribed, what, disc))
    return parts
