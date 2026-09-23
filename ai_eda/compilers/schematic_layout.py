"""Pure geometry for the schematic compiler.

Invariant: every function here is a deterministic function of its arguments
(no I/O, no randomness), so the schematic compiler stays byte-reproducible.

Coordinate frames
-----------------
* Library symbol frame (``SymbolPin.x/y``): mm, **Y up**; ``(x, y)`` is the pin's
  connection point (outer end) and ``angle`` is the direction from that point
  toward the symbol body (0 = +x, 90 = +y/up, 180 = -x, 270 = -y/down).
* Schematic frame: mm, **Y down**. A symbol instance is ``(at X Y ROT)`` with an
  optional ``(mirror x|y)``.

The instance transform is KiCad's ``SCH_SYMBOL`` TRANSFORM: rotate first, then
mirror in schematic coordinates. The matrices below were verified against three
KiCad demo schematics and confirmed experimentally for every rotation / mirror
combination via ``kicad-cli`` netlist export (pin -> net assignment).

KiCad connects by *exact* coordinate equality, so every connection point must
land on the 1.27 mm grid and be formatted identically wherever it is reused.
For the same reason two symbols must never be placed so that a pin, a wire
stub end or a label anchor of one lands on one of the other's: KiCad would
silently merge the nets. :func:`symbol_extent` measures each symbol
(graphics, pins, stubs, label text) from the library data and
:func:`layout_positions` spaces the grid so extents cannot touch; the compiler
additionally refuses any coinciding connection points.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import SymbolDef, SymbolPin

__all__ = [
    "GRID_MM",
    "SYMBOL_PITCH_MM",
    "LAYOUT_ORIGIN_MM",
    "LAYOUT_COLUMNS",
    "LAYOUT_GAP_MM",
    "STUB_LENGTH_MM",
    "LABEL_CHAR_WIDTH_MM",
    "LABEL_MARGIN_MM",
    "ROTATIONS",
    "Vec",
    "Extent",
    "snap",
    "transform_offset",
    "pin_position",
    "pin_body_direction",
    "stub_end",
    "label_orientation",
    "natural_ref_key",
    "grid_positions",
    "symbol_graphic_points",
    "symbol_extent",
    "layout_pitch",
    "layout_positions",
    "point_on_segment",
]

Vec = tuple[float, float]

#: KiCad's default schematic connection grid (50 mil)
GRID_MM = 1.27
#: minimum distance between symbol origins on the placement grid (20 grid steps)
SYMBOL_PITCH_MM = 25.4
#: origin of the first symbol (40 grid steps in from the top-left corner)
LAYOUT_ORIGIN_MM: Vec = (50.8, 50.8)
#: symbols per row before wrapping
LAYOUT_COLUMNS = 4
#: free space kept between the extents of neighbouring symbols (4 grid steps)
LAYOUT_GAP_MM = 5.08
#: wire stub from the pin end outward to the global label (2 grid steps)
STUB_LENGTH_MM = 2.54
#: conservative width reserved per character of a global label (1.27 mm font)
LABEL_CHAR_WIDTH_MM = 1.27
#: the label's arrow/box shape plus half the text height, reserved beyond the text on every side
LABEL_MARGIN_MM = 2.54

#: rotation -> 2x2 matrix mapping library (x, y_up) offsets to schematic (dx, dy_down)
ROTATIONS: dict[int, tuple[Vec, Vec]] = {
    0: ((1.0, 0.0), (0.0, -1.0)),
    90: ((0.0, -1.0), (-1.0, 0.0)),
    180: ((-1.0, 0.0), (0.0, 1.0)),
    270: ((0.0, 1.0), (1.0, 0.0)),
}


def snap(value: float, decimals: int = 4) -> float:
    """Round to the precision KiCad stores (<= 4 decimals) and kill ``-0.0``."""
    v = round(value, decimals)
    return 0.0 if v == 0 else v


def transform_offset(px: float, py: float, rotation: int, mirror: str | None = None) -> Vec:
    """Library-frame offset (Y up) -> schematic-frame offset (Y down) for ``(at X Y rotation) (mirror m)``."""
    if rotation not in ROTATIONS:
        raise ValueError(f"symbol rotation must be one of {sorted(ROTATIONS)}, got {rotation!r}")
    if mirror not in (None, "x", "y"):
        raise ValueError(f"mirror must be None, 'x' or 'y', got {mirror!r}")
    (m00, m01), (m10, m11) = ROTATIONS[rotation]
    dx = m00 * px + m01 * py
    dy = m10 * px + m11 * py
    if mirror == "y":  # horizontal flip
        dx = -dx
    elif mirror == "x":  # vertical flip
        dy = -dy
    return snap(dx), snap(dy)


def pin_position(sym_x: float, sym_y: float, rotation: int, mirror: str | None, pin: SymbolPin) -> Vec:
    """Absolute schematic position of a library pin's connection point."""
    dx, dy = transform_offset(pin.x, pin.y, rotation, mirror)
    return snap(sym_x + dx), snap(sym_y + dy)


def pin_body_direction(rotation: int, mirror: str | None, pin_angle: float) -> Vec:
    """Unit vector (schematic frame) from the pin's connection point toward the symbol body."""
    rad = math.radians(pin_angle)
    ux, uy = transform_offset(math.cos(rad), math.sin(rad), rotation, mirror)
    return snap(ux), snap(uy)


def stub_end(pin_pos: Vec, body_dir: Vec, length: float = STUB_LENGTH_MM) -> Vec:
    """Far end of a wire stub that leaves the pin *away* from the body."""
    return snap(pin_pos[0] - length * body_dir[0]), snap(pin_pos[1] - length * body_dir[1])


def label_orientation(body_dir: Vec) -> tuple[int, str]:
    """``(angle, justify)`` for a global label anchored at the stub end, text extending away from the body.

    KiCad: angle 0 -> text extends toward +x (justify left), 180 -> -x (right),
    90 -> up (left), 270 -> down (right).
    """
    ax, ay = -body_dir[0], -body_dir[1]  # direction away from the body
    if abs(ax) >= abs(ay):
        return (0, "left") if ax > 0 else (180, "right")
    return (270, "right") if ay > 0 else (90, "left")


_REF_SPLIT = re.compile(r"(\d+)")


def natural_ref_key(ref: str) -> tuple:
    """Sort key so that ``R2 < R10`` and ``J1 < R1``: alternating text / integer chunks.

    Every chunk is a ``(0, int)`` or ``(1, str)`` pair, so a list that mixes
    numeric and alphabetic pin numbers (``1``, ``2``, ``CD``, ``SH``, ``A1``)
    sorts instead of raising ``TypeError``.
    """
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in _REF_SPLIT.split(ref) if part)


def grid_positions(
    refs: Iterable[str],
    origin: Vec = LAYOUT_ORIGIN_MM,
    pitch: float | Vec = SYMBOL_PITCH_MM,
    columns: int = LAYOUT_COLUMNS,
) -> dict[str, Vec]:
    """Deterministic symbol origins: refs in natural order, row-major on a ``pitch`` grid.

    ``pitch`` is one value or ``(x pitch, y pitch)``. With the defaults every
    origin is a multiple of 2.54 mm, which keeps all library pin ends
    (multiples of 1.27 mm) on KiCad's connection grid.
    """
    if columns < 1:
        raise ValueError("columns must be >= 1")
    px, py = (pitch, pitch) if isinstance(pitch, (int, float)) else pitch
    out: dict[str, Vec] = {}
    for i, ref in enumerate(sorted(set(refs), key=natural_ref_key)):
        row, col = divmod(i, columns)
        out[ref] = (snap(origin[0] + col * px), snap(origin[1] + row * py))
    return out


# --------------------------------------------------------------------------- extents


@dataclass(frozen=True, slots=True)
class Extent:
    """Axis-aligned box in the schematic frame (Y down), relative to a symbol origin unless stated."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def shifted(self, dx: float, dy: float) -> "Extent":
        return Extent(snap(self.xmin + dx), snap(self.ymin + dy), snap(self.xmax + dx), snap(self.ymax + dy))

    def overlaps(self, other: "Extent") -> bool:
        """True when the two boxes share interior area or touch (touching wires would connect)."""
        return not (self.xmax < other.xmin or other.xmax < self.xmin or self.ymax < other.ymin or other.ymax < self.ymin)


def symbol_graphic_points(symbol: SymbolDef) -> list[Vec]:
    """Library-frame (Y up) vertices of every drawn item in the symbol's unit sub-symbols.

    Rectangles, polylines, beziers, circles (as a box around the radius),
    arcs (start / mid / end) and text anchors are included; the body of a
    symbol is therefore covered by these points plus its pins.
    """
    pts: list[Vec] = []

    def xy(node: list | None) -> None:
        if node is not None and len(node) >= 3 and not isinstance(node[1], list) and not isinstance(node[2], list):
            pts.append((sexpr.to_float(node[1]), sexpr.to_float(node[2])))

    for unit in sexpr.find_all(symbol.node, "symbol"):
        for item in unit:
            h = sexpr.head(item)
            if h in ("rectangle",):
                xy(sexpr.find(item, "start"))
                xy(sexpr.find(item, "end"))
            elif h in ("polyline", "bezier"):
                p = sexpr.find(item, "pts")
                if p is not None:
                    for v in sexpr.find_all(p, "xy"):
                        xy(v)
            elif h == "circle":
                c = sexpr.find(item, "center")
                r = sexpr.get(item, "radius")
                if c is not None and r is not None and len(c) >= 3:
                    cx, cy, rr = sexpr.to_float(c[1]), sexpr.to_float(c[2]), sexpr.to_float(r)
                    pts += [(cx - rr, cy - rr), (cx + rr, cy + rr)]
            elif h == "arc":
                for name in ("start", "mid", "end"):
                    xy(sexpr.find(item, name))
            elif h == "text":
                xy(sexpr.find(item, "at"))
    return pts


def symbol_extent(
    symbol: SymbolDef,
    label_lengths: Mapping[str, int],
    rotation: int = 0,
    mirror: str | None = None,
    stub: float = STUB_LENGTH_MM,
    margin: float = GRID_MM,
) -> Extent:
    """Everything a placed symbol occupies, relative to its origin, in the schematic frame.

    Covers the drawn body, every pin (connection point and inner end), and for
    each pin listed in ``label_lengths`` (pin number -> characters of its net
    name) the wire stub plus a conservative box for the global label text.
    ``margin`` is added on all sides.
    """
    points: list[Vec] = [transform_offset(x, y, rotation, mirror) for x, y in symbol_graphic_points(symbol)]
    for pin in symbol.pins:
        conn = transform_offset(pin.x, pin.y, rotation, mirror)
        rad = math.radians(pin.angle)
        inner = transform_offset(pin.x + pin.length * math.cos(rad), pin.y + pin.length * math.sin(rad), rotation, mirror)
        points += [conn, inner]
        chars = label_lengths.get(pin.number)
        if chars is None:
            continue
        body = pin_body_direction(rotation, mirror, pin.angle)
        end = stub_end(conn, body, stub)
        reach = chars * LABEL_CHAR_WIDTH_MM + LABEL_MARGIN_MM
        far = (snap(end[0] - reach * body[0]), snap(end[1] - reach * body[1]))
        # the label's text box also extends LABEL_MARGIN_MM sideways from the stub axis
        side = (-body[1], body[0])
        for p in (end, far):
            points += [
                (snap(p[0] + LABEL_MARGIN_MM * side[0]), snap(p[1] + LABEL_MARGIN_MM * side[1])),
                (snap(p[0] - LABEL_MARGIN_MM * side[0]), snap(p[1] - LABEL_MARGIN_MM * side[1])),
            ]
    if not points:
        points = [(0.0, 0.0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return Extent(snap(min(xs) - margin), snap(min(ys) - margin), snap(max(xs) + margin), snap(max(ys) + margin))


def _ceil_grid(value: float, step: float = 2 * GRID_MM) -> float:
    """Smallest multiple of ``step`` (2.54 mm by default) that is >= ``value``."""
    return snap(math.ceil(round(value / step, 6)) * step)


def layout_pitch(extents: Iterable[Extent], gap: float = LAYOUT_GAP_MM, min_pitch: float = SYMBOL_PITCH_MM) -> Vec:
    """``(x pitch, y pitch)`` such that no two extents on the grid can touch, whatever their pairing.

    The pitch is the span from the most negative edge to the most positive
    edge over *all* extents plus ``gap`` (so an asymmetric symbol next to
    another asymmetric one still has ``gap`` between them), rounded up to
    2.54 mm, never below ``min_pitch``.
    """
    boxes = list(extents)
    if not boxes:
        return (min_pitch, min_pitch)
    px = max(min_pitch, _ceil_grid(max(e.xmax for e in boxes) - min(e.xmin for e in boxes) + gap))
    py = max(min_pitch, _ceil_grid(max(e.ymax for e in boxes) - min(e.ymin for e in boxes) + gap))
    return (px, py)


def layout_positions(
    extents: Mapping[str, Extent],
    origin: Vec = LAYOUT_ORIGIN_MM,
    columns: int = LAYOUT_COLUMNS,
    gap: float = LAYOUT_GAP_MM,
    min_pitch: float = SYMBOL_PITCH_MM,
) -> dict[str, Vec]:
    """Symbol origins on a grid whose pitch is derived from the symbols' measured extents."""
    return grid_positions(extents.keys(), origin, layout_pitch(extents.values(), gap, min_pitch), columns)


def point_on_segment(p: Vec, a: Vec, b: Vec, tol: float = 1e-6) -> bool:
    """True when ``p`` lies on segment ``a``-``b`` (endpoints included)."""
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > tol:
        return False
    return min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
