"""Read-only view of a compiled ``.kicad_pcb`` for the independent reviewer.

The reviewer must not trust the IR's word for what is on the board; it reads
the board file itself (the same file kicad-cli checked, identified by its
hash) and compares the footprints it finds with the BOM and CPL rows. Only
what a footprint *is* and *where* it is are extracted: reference, value,
library id, anchor position, orientation, side, and its pads' positions as
the file stores them (relative to the anchor, in the footprint's frame - a
bottom-side footprint's already flipped), from which :attr:`BoardFootprint.centre`
places the centre of their box on the board the way KiCad does. Nothing here
writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ai_eda.tools.kicad import sexpr

__all__ = ["BoardFootprint", "read_board_footprints", "board_side"]


@dataclass(frozen=True, slots=True)
class BoardFootprint:
    ref: str
    value: str
    lib_id: str  # "Resistor_SMD:R_0603_1608Metric"
    x: float  # footprint anchor, board frame (mm, Y down)
    y: float
    rotation: float  # as stored: (-180, 180]
    layer: str  # "F.Cu" | "B.Cu"
    #: every pad's ``(at x y)`` as the file stores it: relative to the anchor, in the footprint's frame
    pads: tuple[tuple[float, float], ...] = ()
    #: every token of the footprint's ``(attr ...)`` (``through_hole``, ``exclude_from_bom``, ...), in file order
    attributes: tuple[str, ...] = ()

    @property
    def side(self) -> str:
        return board_side(self.layer)

    @property
    def centre(self) -> tuple[float, float]:
        """The centre of the box around the pads, in the board frame (each pad's offset turned by the footprint's orientation,
        counter-clockwise on screen, Y down, and added to the anchor); the anchor for a footprint without pads."""
        if not self.pads:
            return self.x, self.y
        a = math.radians(self.rotation)
        c, s = math.cos(a), math.sin(a)
        xs = [self.x + px * c + py * s for px, py in self.pads]
        ys = [self.y - px * s + py * c for px, py in self.pads]
        return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def board_side(layer: str) -> str:
    """``F.Cu`` -> ``Top``, ``B.Cu`` -> ``Bottom`` (the CPL spelling)."""
    if layer == "F.Cu":
        return "Top"
    if layer == "B.Cu":
        return "Bottom"
    raise ValueError(f"footprint on {layer!r} is neither top nor bottom")


def read_board_footprints(path: Path) -> list[BoardFootprint]:
    """Every ``(footprint ...)`` of the board at ``path`` (document order). Raises on a malformed file."""
    tree = sexpr.parse_file(path)
    if sexpr.head(tree) != "kicad_pcb":
        raise ValueError(f"{path} is not a (kicad_pcb ...) file")
    out: list[BoardFootprint] = []
    for fp in sexpr.find_all(tree, "footprint"):
        if len(fp) < 2 or isinstance(fp[1], list):
            raise ValueError(f"{path}: footprint without a library id")
        props = {str(p[1]): str(p[2]) for p in sexpr.find_all(fp, "property") if len(p) > 2 and not isinstance(p[2], list)}
        at = sexpr.find(fp, "at")
        layer = sexpr.get(fp, "layer")
        if at is None or len(at) < 3 or layer is None:
            raise ValueError(f"{path}: footprint {props.get('Reference', '?')} lacks (at x y) or (layer ..)")
        out.append(
            BoardFootprint(
                ref=props.get("Reference", ""),
                value=props.get("Value", ""),
                lib_id=str(fp[1]),
                x=sexpr.to_float(at[1]),
                y=sexpr.to_float(at[2]),
                rotation=sexpr.to_float(at[3]) if len(at) > 3 else 0.0,
                layer=str(layer),
                pads=tuple((sexpr.to_float(pat[1]), sexpr.to_float(pat[2])) for pad in sexpr.find_all(fp, "pad")
                           if (pat := sexpr.find(pad, "at")) is not None and len(pat) >= 3),
                attributes=tuple(str(a) for a in sexpr.args(attr)) if (attr := sexpr.find(fp, "attr")) is not None else (),
            )
        )
    return out
