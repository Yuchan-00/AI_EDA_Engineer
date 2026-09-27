"""Placement tools.

Two deterministic placers, both built from the extents of the verified
library footprints and guarded against touching extents or a part outside
the outline:

* :mod:`ai_eda.tools.placement.grid` - a **placeholder** row-major grid in
  natural ref order; the PCB agent uses it for every board without a
  many-pad part.
* :mod:`ai_eda.tools.placement.core_ring` - the part with the most pads in
  the centre, the parts wired only to it on an inner ring ordered by the
  direction of the pads they connect to, everything else on an outer ring at
  the board edge; the PCB agent uses it when a part has at least
  :data:`~ai_eda.tools.placement.core_ring.CORE_MIN_PADS` pads.

Whether the resulting board is valid is decided exclusively by real
``kicad-cli`` DRC (:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), never
by a placer.
"""

from ai_eda.tools.placement.core_ring import CORE_MIN_PADS, RingPlacement, core_ring_placement
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
    "CORE_MIN_PADS",
    "MARGIN_MM",
    "PLACER_ID",
    "PLACER_VERSION",
    "SPACING_MM",
    "GridPlacement",
    "RingPlacement",
    "core_ring_placement",
    "footprint_extent",
    "grid_pitch",
    "grid_placement",
    "placement_provenance",
]
