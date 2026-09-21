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
"""

from __future__ import annotations

import math

from ai_eda.ir import BoardSide, Placement
from ai_eda.tools.kicad.library import BBox, FootprintDef, Pad

__all__ = [
    "RESOLUTION_DECIMALS",
    "normalize_angle",
    "footprint_angle",
    "rotate",
    "stored_offset",
    "to_board",
    "pad_center",
    "pad_angle",
    "text_angle",
    "mirrored_layer",
    "pad_layers",
    "courtyard_bbox",
    "pads_bbox",
    "footprint_bbox",
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
    """Absolute centre of ``pad`` (library frame, y down) once its footprint is placed."""
    return to_board(placement, pad.x, pad.y)


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


def _bbox_of_points(points: list[tuple[float, float]]) -> BBox | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(_q(min(xs)), _q(min(ys)), _q(max(xs)), _q(max(ys)))


def courtyard_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Axis-aligned board-frame box around the placed courtyard (None when the footprint has none).

    Exact for rotations that are multiples of 90 degrees; for other angles it
    is the box around the rotated rectangle's corners (conservative).
    """
    cy = fp.courtyard
    if cy is None:
        return None
    corners = [(cy.x1, cy.y1), (cy.x2, cy.y1), (cy.x2, cy.y2), (cy.x1, cy.y2)]
    return _bbox_of_points([to_board(placement, x, y) for x, y in corners])


def pads_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Board-frame box around all pad copper (exact for pads at multiples of 90 degrees, else conservative)."""
    points: list[tuple[float, float]] = []
    for pad in fp.pads:
        cx, cy = pad_center(placement, pad)
        angle = pad_angle(placement, pad)
        if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):
            w, h = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
            hx, hy = w / 2.0, h / 2.0
        else:
            hx = hy = math.hypot(pad.size_w, pad.size_h) / 2.0
        points += [(cx - hx, cy - hy), (cx + hx, cy + hy)]
    return _bbox_of_points(points)


def footprint_bbox(placement: Placement, fp: FootprintDef) -> BBox | None:
    """Union of :func:`courtyard_bbox` and :func:`pads_bbox` (None when neither exists)."""
    boxes = [b for b in (courtyard_bbox(placement, fp), pads_bbox(placement, fp)) if b is not None]
    if not boxes:
        return None
    return BBox(min(b.x1 for b in boxes), min(b.y1 for b in boxes), max(b.x2 for b in boxes), max(b.y2 for b in boxes))
