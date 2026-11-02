"""Silkscreen tools.

:mod:`ai_eda.tools.silkscreen.place` is the deterministic silkscreen placer
(``silkscreen.place``): reference designators beside their footprints,
connector pin labels and the board title, kept clear of pad copper, the
board edge, the footprints' own silk and each other on an estimated text box.
:mod:`ai_eda.tools.silkscreen.geometry` reads the board as shapes (pad copper,
library silk graphics and texts, the text-box estimate) for the placer and
for the ``pcb.silk.*`` checks. Whether the compiled board's silkscreen is
valid is decided by KiCad's DRC (``silk_over_copper`` / ``silk_overlap``),
never here.
"""

from ai_eda.tools.silkscreen.geometry import (
    FAB_LAYERS,
    SILK_LAYERS,
    TEXT_HEIGHT_FACTOR,
    TEXT_WIDTH_FACTOR,
    Shape,
    footprint_silk,
    pad_copper,
    shape_distance,
    text_box,
    text_extent,
)
from ai_eda.tools.silkscreen.place import PLACER_ID, PLACER_VERSION, SilkParams, SilkPlacement, place_silkscreen, stroke_for

__all__ = [
    "FAB_LAYERS",
    "PLACER_ID",
    "PLACER_VERSION",
    "SILK_LAYERS",
    "TEXT_HEIGHT_FACTOR",
    "TEXT_WIDTH_FACTOR",
    "Shape",
    "SilkParams",
    "SilkPlacement",
    "footprint_silk",
    "pad_copper",
    "place_silkscreen",
    "shape_distance",
    "stroke_for",
    "text_box",
    "text_extent",
]
