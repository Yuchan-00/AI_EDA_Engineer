"""Placement tools.

Only :mod:`ai_eda.tools.placement.grid` exists today and it is a
**placeholder**: a deterministic row-major grid built from the extents of
the verified library footprints, with a guard that refuses touching extents
or a part outside the outline. Whether the resulting board is valid is
decided exclusively by real ``kicad-cli`` DRC
(:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), never by the placer.
"""

from ai_eda.tools.placement.grid import (
    COLUMNS,
    MARGIN_MM,
    PLACER_ID,
    PLACER_VERSION,
    SPACING_MM,
    GridPlacement,
    footprint_extent,
    grid_pitch,
    grid_placement,
    placement_provenance,
)

__all__ = [
    "COLUMNS",
    "MARGIN_MM",
    "PLACER_ID",
    "PLACER_VERSION",
    "SPACING_MM",
    "GridPlacement",
    "footprint_extent",
    "grid_pitch",
    "grid_placement",
    "placement_provenance",
]
