"""Bounded rectangle search for functional placement (dimensions in mm).

CHOICE: 40% utilization is a search seed, 1 mm steps, at most 160 mm per
side and aspect ratio 2. Candidates sort by area, aspect ratio, width, height.
Feasibility is decided by the caller, never by the area estimate.
"""

from __future__ import annotations

import math

from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline

UTILIZATION = 0.40
MAX_SIDE_MM = 160
MAX_ASPECT = 2.0


def rectangles(inflated_area: float, mechanical_area: float, *, fixed: BoardOutline | None = None,
               min_width: float = 1, min_height: float = 1, max_side: int = MAX_SIDE_MM):
    """Yield the fixed outline verbatim, or every bounded candidate above the seed."""
    values = (inflated_area, mechanical_area, min_width, min_height, max_side)
    if not all(math.isfinite(v) and v >= 0 for v in values) or max_side < 1:
        raise CompileError("functional outline: areas and bounds must be finite and nonnegative")
    if fixed is not None:
        if not all(math.isfinite(v) for v in (fixed.width_mm, fixed.height_mm, fixed.origin_x_mm, fixed.origin_y_mm)) or min(fixed.width_mm, fixed.height_mm) <= 0:
            raise CompileError("functional outline: fixed outline must have finite coordinates and positive dimensions")
        yield fixed
        return
    seed = inflated_area / UTILIZATION + mechanical_area
    sizes = [(w, h) for w in range(max(1, math.ceil(min_width)), max_side + 1)
             for h in range(max(1, math.ceil(min_height)), max_side + 1)
             if w * h >= seed and max(w, h) / min(w, h) <= MAX_ASPECT]
    for w, h in sorted(sizes, key=lambda s: (s[0] * s[1], max(s) / min(s), s[0], s[1])):
        yield BoardOutline(width_mm=w, height_mm=h)
