"""Deterministic grid placement from library footprint extents (pure, no I/O beyond the library).

Invariant: every coordinate produced here is a function of the IR's
component list, the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`) and the three grid
parameters (:data:`SPACING_MM`, :data:`MARGIN_MM`, :data:`COLUMNS`). Nothing
is estimated and nothing is guessed: a component without a footprint, a
footprint that is not on disk, or one whose extent cannot be measured
(neither courtyard nor pads) raises :class:`~ai_eda.errors.CompileError`
instead of getting a default size.

What this is: a *placeholder* placer that puts every part on its own cell of
a row-major grid, ordered by :func:`~ai_eda.compilers.schematic_layout.natural_ref_key`
(``R2 < R10``, ``J1 < R1``), rotation 0, top side. The cell pitch is the
largest footprint extent plus :data:`SPACING_MM` in each axis, so no two
extents can touch; each part is anchored so the top-left corner of its own
extent sits at its cell's top-left corner, which keeps the rule valid for
asymmetric footprints. Before returning, :func:`grid_placement` re-measures
the placed extents with :func:`~ai_eda.tools.kicad.geometry.footprint_bbox`
and refuses (``CompileError``) any pair that touches or any part outside the
outline - the guard behind the arithmetic, as the schematic compiler's
extent check guards its pitch.

What this is not: a layout. Whether the board is valid is decided only by
``kicad-cli pcb drc`` on the compiled board. A placed-but-unrouted board with
nets FAILs DRC with ``unconnected_items``; nothing here claims otherwise.

Outline: without a user outline the board is the grid's bounding box plus
:data:`MARGIN_MM` on every side, at origin ``(0, 0)``. A user outline
(``ir.pcb.outline``) is kept verbatim - never resized - and the grid is
anchored at ``(origin_x + margin, origin_y + margin)``; when the parts do not
fit inside it the guard refuses.

Traceability: every :class:`~ai_eda.ir.Placement` carries ``derived``
provenance naming this tool (:data:`PLACER_ID` / :data:`PLACER_VERSION`),
the footprint and the grid parameters in ``derived_from``.
``Provenance.inputs`` stays empty: that field is the calculator role map
:func:`ai_eda.tools.calc.recompute_parameters` rebuilds calls from, and a
placement is not a calculator output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, Placement, Provenance, ProvenanceKind
from ai_eda.tools.kicad.geometry import RESOLUTION_DECIMALS, finite_bbox, footprint_bbox
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary

__all__ = [
    "PLACER_ID",
    "PLACER_VERSION",
    "SPACING_MM",
    "MARGIN_MM",
    "COLUMNS",
    "GridPlacement",
    "footprint_extent",
    "grid_pitch",
    "placement_provenance",
    "grid_placement",
]

#: provenance ``tool`` / ``tool_version`` stamped on every placement
PLACER_ID = "placement.grid"
PLACER_VERSION = "0.1"
#: free space kept between the extents of neighbouring parts (an engineering assumption, recorded in ``derived_from``)
SPACING_MM = 1.0
#: free space between the grid and the board edge (idem)
MARGIN_MM = 2.0
#: parts per row before wrapping
COLUMNS = 4


def _q(v: float) -> float:
    """Round to KiCad resolution (1e-6 mm) and normalise ``-0.0`` to ``0.0``."""
    return round(v, RESOLUTION_DECIMALS) + 0.0


@dataclass(frozen=True, slots=True)
class GridPlacement:
    """What :func:`grid_placement` produced: the outline, the placements and the measured extents (board frame)."""

    outline: BoardOutline
    placements: list[Placement]
    pitch: tuple[float, float]
    extents: dict[str, BBox]


def footprint_extent(fp: FootprintDef) -> BBox:
    """The footprint's extent (courtyard union pads) around its own origin at rotation 0 on the top side.

    Raises :class:`CompileError` when the footprint has neither a courtyard
    nor pads (a size for it would be a guess) or when the measured box has a
    non-finite coordinate (a NaN pad would otherwise vanish from ``min`` /
    ``max`` and an infinite one is everywhere; the library loader refuses
    such numbers at the source and this is the guard behind it).
    """
    origin = Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=0.0, side=BoardSide.TOP)
    box = footprint_bbox(origin, fp)
    if box is None:
        raise CompileError(f"footprint {fp.lib_id} has neither a courtyard nor pads; its extent cannot be measured")
    return finite_bbox(box, f"footprint {fp.lib_id}")


def grid_pitch(extents: Mapping[str, BBox], spacing: float) -> tuple[float, float]:
    """Cell pitch ``(x, y)``: the largest extent span in each axis plus ``spacing``."""
    if not extents:
        raise CompileError("no extents: nothing to space")
    return _q(max(b.width for b in extents.values()) + spacing), _q(max(b.height for b in extents.values()) + spacing)


def placement_provenance(footprint_id: str, spacing: float, margin: float, columns: int) -> Provenance:
    """``derived`` provenance of a placement this tool made from ``footprint_id`` and the grid parameters."""
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=PLACER_ID,
        tool_version=PLACER_VERSION,
        derived_from=[f"footprint:{footprint_id}", f"spacing_mm:{spacing}", f"margin_mm:{margin}", f"columns:{columns}"],
        note="row-major grid from library footprint extents; validity is decided by kicad-cli DRC only",
    )


def _disjoint(a: BBox, b: BBox) -> bool:
    """Strictly apart: touching edges count as an overlap (as the schematic compiler's extents do)."""
    return a.x2 < b.x1 or b.x2 < a.x1 or a.y2 < b.y1 or b.y2 < a.y1


def _inside(box: BBox, outline: BoardOutline) -> bool:
    return (
        outline.origin_x_mm <= box.x1
        and box.x2 <= outline.origin_x_mm + outline.width_mm
        and outline.origin_y_mm <= box.y1
        and box.y2 <= outline.origin_y_mm + outline.height_mm
    )


def _resolve(ir: CircuitIR, library: KicadLibrary) -> list[tuple[str, FootprintDef]]:
    """``(ref, footprint)`` for every component in natural order; refuses instead of guessing.

    ``load_footprint`` may raise ``LibraryFormatError`` (a corrupt
    ``.kicad_mod``) or ``LibraryLookupError`` (the file vanished between
    resolve and load); both propagate to the caller.
    """
    if not ir.components:
        raise CompileError("no components to place")
    out: list[tuple[str, FootprintDef]] = []
    for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        if comp.footprint is None:
            raise CompileError(f"component {comp.ref!r} has no footprint; the placer never picks one")
        resolved = library.resolve_footprint(comp.footprint)
        if not resolved.verified:
            raise CompileError(
                f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} was not found in a KiCad "
                f"library (searched {[str(r) for r in library.roots]}); refusing to guess"
            )
        out.append((comp.ref, library.load_footprint(comp.footprint)))
    return out


def grid_placement(
    ir: CircuitIR,
    library: KicadLibrary,
    *,
    spacing: float = SPACING_MM,
    margin: float = MARGIN_MM,
    columns: int = COLUMNS,
    outline: BoardOutline | None = None,
) -> GridPlacement:
    """Place every component of ``ir`` on a row-major grid; pure (same IR + library -> same result).

    ``outline`` (the user's, when the IR has one) is kept verbatim and the
    grid is anchored at ``(origin + margin)``; without one the outline is the
    grid's bounding box plus ``margin`` on every side at ``(0, 0)``. Raises
    :class:`CompileError` for: no components, a component without a
    footprint, a footprint not on disk, an unmeasurable extent, a parameter
    that makes no sense, or a result that violates the guard (two placed
    extents touching, or a part outside the outline - which is how a too
    small user outline is refused).
    """
    if columns < 1:
        raise CompileError("columns must be >= 1")
    if spacing < 0 or margin < 0:
        raise CompileError("spacing and margin must be >= 0")
    parts = _resolve(ir, library)
    extents = {ref: footprint_extent(fp) for ref, fp in parts}
    px, py = grid_pitch(extents, spacing)
    ox = outline.origin_x_mm if outline is not None else 0.0
    oy = outline.origin_y_mm if outline is not None else 0.0
    placements: list[Placement] = []
    placed: dict[str, BBox] = {}
    for i, (ref, fp) in enumerate(parts):
        row, col = divmod(i, columns)
        ext = extents[ref]
        # the part's own extent corner sits at the cell corner, so the rule holds for asymmetric footprints
        placement = Placement(
            component_ref=ref,
            x_mm=_q(ox + margin + col * px - ext.x1),
            y_mm=_q(oy + margin + row * py - ext.y1),
            rotation_deg=0.0,
            side=BoardSide.TOP,
            provenance=placement_provenance(fp.lib_id, spacing, margin, columns),
        )
        box = footprint_bbox(placement, fp)
        assert box is not None  # the extent was measurable (and finite) a moment ago from the same footprint
        placements.append(placement)
        placed[ref] = box
    if outline is None:
        outline = BoardOutline(
            width_mm=_q(max(b.x2 for b in placed.values()) + margin - ox),
            height_mm=_q(max(b.y2 for b in placed.values()) + margin - oy),
            origin_x_mm=ox,
            origin_y_mm=oy,
        )
    refs = [p.component_ref for p in placements]
    for a_i, a in enumerate(refs):
        for b in refs[a_i + 1:]:
            if not _disjoint(placed[a], placed[b]):
                raise CompileError(f"placed extents of {a!r} and {b!r} touch or overlap ({placed[a]} vs {placed[b]}); refusing the grid")
    outside = [r for r in refs if not _inside(placed[r], outline)]
    if outside:
        raise CompileError(
            f"component(s) {outside} do not fit inside the {outline.width_mm} x {outline.height_mm} mm outline at "
            f"({outline.origin_x_mm}, {outline.origin_y_mm}) with margin {margin} mm; the outline is kept as given, not resized"
        )
    return GridPlacement(outline=outline, placements=placements, pitch=(px, py), extents=placed)
