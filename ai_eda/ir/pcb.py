"""PCB model.

Geometry here is authored in the IR and compiled to ``.kicad_pcb``. Pad
geometry is *not* stored here - it comes from the verified KiCad footprint at
compile time.

Traceability (spec 26): every layout item (``Placement``, ``Track``, ``Via``,
``Zone``) carries a :class:`~ai_eda.ir.provenance.Provenance` so copper can be
traced to the tool run, the user or the model that produced it. An item that
was constructed without saying where it came from gets
:data:`UNRECORDED_ORIGIN` - an *assumption* that ``needs_verification`` - and
the reviewer reports the board as NOT_VERIFIED until someone records the
origin. A tool that generates layout (``ai_eda.tools.routing``) stamps
``derived`` provenance with its id and version.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Provenance, ProvenanceKind, Traced

#: note of the placeholder provenance a layout item gets when nobody said where it came from
UNRECORDED_ORIGIN = "origin not recorded"


def unrecorded_origin() -> Provenance:
    """Placeholder provenance for layout authored without saying by whom / what (needs verification)."""
    return Provenance(kind=ProvenanceKind.ASSUMPTION, note=UNRECORDED_ORIGIN)


class BoardSide(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


class Layer(BaseModel):
    name: str  # KiCad layer name: "F.Cu", "B.Cu", "In1.Cu", ...
    kind: str  # "signal", "power", "mixed"


class BoardOutline(BaseModel):
    """Simple rectangular outline; extend with polygon points later."""

    width_mm: float
    height_mm: float
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0


class Placement(BaseModel):
    component_ref: str
    x_mm: float
    y_mm: float
    rotation_deg: float = 0.0
    side: BoardSide = BoardSide.TOP
    #: who / what decided this position (user, placement tool, model); unrecorded = assumption
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Track(BaseModel):
    net: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float
    #: the router / user / model that drew this segment; ``derived_from`` names the net and placements
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Via(BaseModel):
    net: str
    x_mm: float
    y_mm: float
    drill_mm: float
    diameter_mm: float
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Zone(BaseModel):
    net: str
    layer: str
    polygon: list[tuple[float, float]]
    clearance_mm: float | None = None
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class ManufacturingConstraints(BaseModel):
    """Fab limits. Every value is Traced so a JLCPCB limit that was never
    confirmed stays visibly NOT_VERIFIED instead of silently becoming a rule."""

    fab: str | None = None  # "JLCPCB"
    min_track_width_mm: Traced[float] | None = None
    min_clearance_mm: Traced[float] | None = None
    min_via_drill_mm: Traced[float] | None = None
    min_via_diameter_mm: Traced[float] | None = None
    min_hole_to_edge_mm: Traced[float] | None = None
    layer_count_options: Traced[list[int]] | None = None
    copper_weight_oz: Traced[float] | None = None
    board_thickness_mm: Traced[float] | None = None


class PCBDesign(BaseModel):
    layers: list[Layer] = Field(default_factory=lambda: [Layer(name="F.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")])
    outline: BoardOutline | None = None
    placements: list[Placement] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    vias: list[Via] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    manufacturing: ManufacturingConstraints = Field(default_factory=ManufacturingConstraints)

    def placement(self, ref: str) -> Placement | None:
        for p in self.placements:
            if p.component_ref == ref:
                return p
        return None

    def layout_items(self) -> list[tuple[str, Provenance]]:
        """``(label, provenance)`` of every placement / track / via / zone, for traceability review."""
        out: list[tuple[str, Provenance]] = [(f"placement[{p.component_ref}]", p.provenance) for p in self.placements]
        out += [(f"track[{i}:{t.net}]", t.provenance) for i, t in enumerate(self.tracks)]
        out += [(f"via[{i}:{v.net}]", v.provenance) for i, v in enumerate(self.vias)]
        out += [(f"zone[{i}:{z.net}]", z.provenance) for i, z in enumerate(self.zones)]
        return out
