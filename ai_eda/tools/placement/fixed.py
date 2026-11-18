"""Deterministic placement of a board whose design fixes parts and names pack regions (``placement.fixed`` 0.1; pure, no I/O beyond the library).

Invariant: every coordinate produced here is a function of the IR's
components, the board the design gave (``ir.pcb``: ``outline``, ``fixed``,
``pack_regions``, ``keepouts`` and the silk texts already on it), the
footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: extents and pads, never
model memory) and the two parameters :data:`SPACING_MM` and
:data:`MARGIN_MM` - the grid placer's own (``placement.grid``, the free space
kept between neighbouring extents and between the parts and the board edge:
engineering choices recorded in ``derived_from``). Nothing is estimated and
nothing is guessed: every case below that cannot be placed without moving a
part the design fixed, putting a part outside its region or overlapping two
parts raises :class:`~ai_eda.errors.CompileError`, naming the parts - there
is no fallback to another placer.

What this is: the ``placement.fixed`` placer :class:`~ai_eda.ir.PCBDesign`
documents for ``fixed`` and ``pack_regions`` (a keyboard's keys in one row in
pitch order, the parts of each circuit block in the rectangle the design
reserved for it). The PCB agent uses it for every board with fixed parts or
pack regions:

1. **Outline.** The board's outline (``ir.pcb.outline``) is kept verbatim;
   there is none to infer - the fixed parts and the regions are positions on
   it, so a board without one is refused (a template that proposes fixed
   parts proposes the outline with them).
2. **Fixed parts** (``ir.pcb.fixed``) are copied into the placements as
   given: position, rotation, side and the provenance of whoever fixed them
   (a template, the user). Their extents are measured from the library
   footprint at that position; a fixed part outside the outline, or two
   fixed parts whose extents touch or overlap, are refused. A fixed part
   needs no margin from the edge: it is where the design put it.
3. **Pack regions** (``ir.pcb.pack_regions``), in list order: a region's refs
   are packed into its rectangle (kept :data:`MARGIN_MM` inside the board
   edge) **in the region's own list order** - rows from the rectangle's
   top-left corner, left to right - around everything placed before. A region
   that leaves the outline, and a region whose parts do not all fit, are
   refused. This version packs the top side only (a region whose ``side`` is
   ``bottom`` is refused).
4. **Every other part** (in no region and not fixed) is packed the same way
   into the free board area - the outline :data:`MARGIN_MM` inside its edge,
   every region an obstacle - in natural ref order
   (:func:`~ai_eda.compilers.schematic_layout.natural_ref_key`). A part that
   finds no room there is refused: a component never ends up nowhere.
5. **Obstacles** while packing: every fixed part's and every packed part's
   extent grown by :data:`SPACING_MM`, the estimated box of every text the
   board's silk layers already carry (a template's note names, a title:
   :func:`~ai_eda.tools.silkscreen.geometry.ir_text_box`) grown by
   :data:`SPACING_MM`, and every keep-out area that forbids ``footprints`` -
   or ``pads`` on a layer the part's pads are on - unless it allows the ref.
   The packing itself is the RF floorplan's shelf packer
   (:func:`~ai_eda.tools.placement.rf_floorplan._shelf_pack`: rotation 0 where
   the extent fits at the cursor, else 90; a part that meets an obstacle
   moves right past it; a row that no part could use moves down past what
   blocked it), so both placers pack by one rule.

Before returning, the placed extents are re-measured with
:func:`~ai_eda.tools.kicad.geometry.footprint_bbox` and checked by the RF
floorplan's guard (:func:`~ai_eda.tools.placement.rf_floorplan._guard`):
two extents that touch, a part outside the outline or a packed part outside
its region is a :class:`CompileError`; so is a part (a fixed one included)
in a keep-out that bans it, judged on the part's own side as the
``pcb.keepout`` check judges it: a ``footprints`` ban on that side's copper
(``F.Cu`` / ``B.Cu``), a ``pads`` ban on a layer its pads are on there (a
bottom-side part's ``F.Cu`` pads are on ``B.Cu``). Extents are compared
whatever side a part is on (a conservative rule of this version: a
bottom-side fixed part may not sit under a top-side part).

Also refused: a ref in ``fixed`` or in a region that is no component of the
IR, a ref fixed twice or both fixed and packed or packed in two regions (the
IR model refuses these too; the placer does not rely on it), and an IR with
RF blocks (``ir.rf.blocks``): ``placement.rf_floorplan`` does not read fixed
parts and this placer does not read block regions, chains or shield cans,
so neither is used silently in place of the other. Fine-pitch parts get no
fan-out room here (``placement.rf_floorplan`` 0.2 keeps one; this version
does not).

What this is not: a layout judgement. Whether the board is valid is decided
only by ``kicad-cli pcb drc`` on the compiled board; the silk texts kept clear
here are judged by the ``pcb.silk.*`` checks.

Traceability: a fixed part keeps the provenance it was fixed with. Every
packed :class:`~ai_eda.ir.Placement` carries ``derived`` provenance naming
this tool (:data:`PLACER_ID` / :data:`PLACER_VERSION`), the footprint, the
region (its id and rectangle, or the free board area), the row, the
rotation, the fixed parts it was packed around and the parameters in
``derived_from``. ``Provenance.inputs`` stays empty: that field is the
calculator role map, and a placement is not a calculator output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, Placement, Provenance, ProvenanceKind
from ai_eda.tools.keepout import allowed_refs, area_bbox, area_points, box_area_overlap, covers_layer, forbids, keepout_id, keepouts_of
from ai_eda.tools.kicad.geometry import finite_bbox, footprint_bbox, mirrored_layer
from ai_eda.tools.kicad.library import BBox, KicadLibrary
from ai_eda.tools.placement.grid import MARGIN_MM, SPACING_MM, _disjoint, _inside, _resolve, footprint_extent
from ai_eda.tools.placement.rf_floorplan import Box, _clip, _grow, _guard, _Obstacles, _pad_layers, _shelf_pack, _touch, rf_blocks
from ai_eda.tools.silkscreen.geometry import FAB_LAYERS, ir_text_box

__all__ = [
    "PLACER_ID",
    "PLACER_VERSION",
    "SPACING_MM",
    "MARGIN_MM",
    "FREE_AREA",
    "FixedPlacement",
    "fixed_placement",
]

#: provenance ``tool`` / ``tool_version`` stamped on every packed placement
PLACER_ID = "placement.fixed"
PLACER_VERSION = "0.1"
#: what :attr:`FixedPlacement.region_of` says for a fixed part (no region id can be spelt so: an id is a plain identifier)
FIXED = "(fixed)"
#: what :attr:`FixedPlacement.region_of` says for a part packed in the free board area
FREE_AREA = ""
_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class FixedPlacement:
    """What :func:`fixed_placement` produced: the outline, the placements, the placed extents and where each part went."""

    outline: BoardOutline
    #: every component's placement, in natural ref order (the fixed ones exactly as the design gave them)
    placements: list[Placement]
    extents: dict[str, BBox]
    #: ref -> :data:`FIXED`, the pack region's id, or :data:`FREE_AREA` (``""``, the free board area)
    region_of: dict[str, str]
    #: the pack region ids in the order they were packed
    regions: list[str] = field(default_factory=list)
    #: the keep-out ids the packer honoured (footprint / pad bans)
    keepouts: list[str] = field(default_factory=list)
    #: how many silk texts already on the board the packed parts were kept clear of
    texts: int = 0

    def description(self, origin: str, spacing: float, margin: float) -> str:
        o = self.outline
        fixed = sum(1 for r in self.region_of.values() if r == FIXED)
        free = sum(1 for r in self.region_of.values() if r == FREE_AREA)
        per = [f"{rid} {sum(1 for r in self.region_of.values() if r == rid)}" for rid in self.regions]
        text = (
            f"{PLACER_ID} {PLACER_VERSION}: {len(self.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm {origin} outline at "
            f"({o.origin_x_mm}, {o.origin_y_mm}), {fixed} fixed part(s) where the design put them, {len(self.regions)} pack region(s)"
            + (f" ({', '.join(per)})" if per else "")
            + f" packed in their list order, {free} part(s) in the free board area in natural ref order; shelf packing by each part's own "
            f"extent around the fixed parts, spacing {spacing} mm, margin {margin} mm"
        )
        if self.texts:
            text += f"; {self.texts} silk text(s) on the board kept clear"
        if self.keepouts:
            text += f"; keep-outs {', '.join(self.keepouts)} honoured"
        return text


def _provenance(lib_id: str, region: str, box: Box | None, row: int, rotation: float, spacing: float, margin: float, fixed: list[str]) -> Provenance:
    where = f"region:{region}:" + ",".join(f"{v:g}" for v in (box[0], box[1], box[2] - box[0], box[3] - box[1])) if box is not None else "region:(free board area)"
    derived = [f"footprint:{lib_id}", where, f"row:{row}", f"rotation:{rotation:g}", f"around_fixed:{','.join(fixed) or '-'}",
               f"spacing_mm:{spacing}", f"margin_mm:{margin}"]
    return Provenance(
        kind=ProvenanceKind.DERIVED, tool=PLACER_ID, tool_version=PLACER_VERSION, derived_from=derived,
        note="shelf packing around the design's fixed parts from library footprint extents (pack regions in their list order, the rest in "
             "natural ref order); validity is decided by kicad-cli DRC only",
    )


def _keepout_ban(e: BBox, ref: str, side: BoardSide, pad_layers: frozenset[str], areas: list) -> str | None:
    """``"<keep-out id>: <what>"`` of the first keep-out of ``areas`` that bans the part ``ref`` (extent ``e``) on ``side``, or ``None``.

    The part's own side decides (module docstring): its footprint is on that
    side's copper, its pads on their library layers mirrored to that side
    (``pad_layers``: the layers at the top side). On the top side this is the
    rule of :meth:`~ai_eda.tools.placement.rf_floorplan._Obstacles.hit`.
    """
    copper = "F.Cu" if side == BoardSide.TOP else "B.Cu"
    pads = sorted({mirrored_layer(layer, side) for layer in pad_layers})
    box = (e.x1, e.y1, e.x2, e.y2)
    for kid, pts, bb, refs, _f, _p, ko in areas:
        if ref in refs or not (_touch(box, bb) and box_area_overlap(box, pts) > 0.0):
            continue
        if forbids(ko, "footprints") and covers_layer(ko, copper):
            return f"{kid}: footprints on {copper}, the {side} side"
        hit = [layer for layer in pads if covers_layer(ko, layer)] if forbids(ko, "pads") else []
        if hit:
            return f"{kid}: pads on {', '.join(hit)}"
    return None


def fixed_placement(ir: CircuitIR, library: KicadLibrary, *, spacing: float = SPACING_MM, margin: float = MARGIN_MM) -> FixedPlacement:
    """Place every component of ``ir`` around the board's fixed parts and in its pack regions (module docstring); pure.

    Reads ``ir.pcb`` (outline, fixed, pack_regions, keepouts, silkscreen);
    ``ir`` is never modified. Raises :class:`CompileError` for every case the
    module docstring lists, naming the parts / region.
    """
    if not spacing > 0 or not margin >= 0:
        raise CompileError("spacing must be > 0 and margin >= 0")
    pcb = ir.pcb
    if pcb is None or not (pcb.fixed or pcb.pack_regions):
        raise CompileError("no fixed parts and no pack regions (ir.pcb.fixed / ir.pcb.pack_regions): placement.fixed has nothing to keep")
    if pcb.outline is None:
        raise CompileError("no board outline (ir.pcb.outline): placement.fixed needs one - the fixed parts and pack regions are positions on it "
                           "(a template that proposes fixed parts proposes the outline with them)")
    if rf_blocks(ir):
        raise CompileError("the IR has RF blocks (ir.rf.blocks) and fixed parts / pack regions: placement.rf_floorplan does not read fixed parts "
                           "and placement.fixed does not read block regions, chains or shield cans; refusing rather than ignoring either")
    parts = _resolve(ir, library)  # natural ref order; refuses a missing footprint
    fps = dict(parts)
    order = [ref for ref, _ in parts]
    owner: dict[str, str] = {}
    for p in pcb.fixed:
        if p.component_ref in owner:
            raise CompileError(f"{p.component_ref} is fixed twice")
        owner[p.component_ref] = FIXED
    seen_regions: set[str] = set()
    for r in pcb.pack_regions:
        if r.id in seen_regions:
            raise CompileError(f"pack region id {r.id!r} is used twice")
        seen_regions.add(r.id)
        for ref in r.refs:
            if ref in owner:
                raise CompileError(f"{ref} is both {'fixed' if owner[ref] == FIXED else 'in pack region ' + owner[ref]} and in pack region {r.id}")
            owner[ref] = r.id
    stray_fixed = [p.component_ref for p in pcb.fixed if p.component_ref not in fps]
    if stray_fixed:
        raise CompileError(f"fixed part(s) {stray_fixed} are not components of the IR")
    for r in pcb.pack_regions:
        stray = [ref for ref in r.refs if ref not in fps]
        if stray:
            raise CompileError(f"pack region {r.id} names {stray}, which are not components of the IR")
        if r.side != BoardSide.TOP:
            raise CompileError(f"pack region {r.id}: side {r.side} - placement.fixed {PLACER_VERSION} packs the top side only")
    outline = pcb.outline
    ox, oy = float(outline.origin_x_mm), float(outline.origin_y_mm)
    edge: Box = (ox, oy, ox + float(outline.width_mm), oy + float(outline.height_mm))
    size = f"the {outline.width_mm} x {outline.height_mm} mm outline at ({outline.origin_x_mm}, {outline.origin_y_mm})"

    # the fixed parts, as given (module docstring, step 2)
    fixed_refs = sorted((p.component_ref for p in pcb.fixed), key=natural_ref_key)
    given = {p.component_ref: p for p in pcb.fixed}
    placed: dict[str, BBox] = {}
    for ref in fixed_refs:
        fp = fps[ref]
        footprint_extent(fp)  # refuses a footprint whose extent cannot be measured, as every placer does
        box = footprint_bbox(given[ref], fp)
        assert box is not None  # measurable a moment ago
        placed[ref] = finite_bbox(box, f"fixed part {ref} at ({given[ref].x_mm}, {given[ref].y_mm})")
    outside = [r for r in fixed_refs if not _inside(placed[r], outline)]
    if outside:
        raise CompileError(f"fixed part(s) {outside} lie outside {size} ({'; '.join(f'{r} {placed[r]}' for r in outside)}); a fixed part is never moved")
    clashes = [(a, b) for i, a in enumerate(fixed_refs) for b in fixed_refs[i + 1:] if not _disjoint(placed[a], placed[b])]
    if clashes:
        raise CompileError("fixed parts overlap: " + "; ".join(f"{a} {placed[a]} and {b} {placed[b]}" for a, b in clashes)
                           + " (touching extents count); a fixed part is never moved")

    # the regions must lie on the board (step 3)
    for r in pcb.pack_regions:
        x1, y1, x2, y2 = r.box()
        if x1 < edge[0] - _TOL or y1 < edge[1] - _TOL or x2 > edge[2] + _TOL or y2 > edge[3] + _TOL:
            raise CompileError(f"pack region {r.id} ({x1:g}, {y1:g})-({x2:g}, {y2:g}) leaves {size}")

    # the obstacles (step 5): the fixed parts, the board's silk texts, the keep-outs that ban footprints / pads
    board_layers = [layer.name for layer in pcb.layers]
    areas = []
    honoured: list[str] = []
    for ko in keepouts_of(pcb):
        if forbids(ko, "footprints") or forbids(ko, "pads"):
            pts = area_points(ko)
            areas.append((keepout_id(ko), pts, area_bbox(pts), frozenset(allowed_refs(ko)), forbids(ko, "footprints"), forbids(ko, "pads"), ko))
            honoured.append(keepout_id(ko))
    obstacles = _Obstacles(boxes=[(_grow((b.x1, b.y1, b.x2, b.y2), spacing), f"fixed part {ref}") for ref, b in placed.items()], areas=list(areas))
    texts = [t for t in pcb.silkscreen if t.layer not in FAB_LAYERS]
    for t in texts:
        obstacles.boxes.append((_grow(ir_text_box(t).bbox(), spacing), f"silk text {t.text!r}"))
    inner_edge: Box = (edge[0] + margin, edge[1] + margin, edge[2] - margin, edge[3] - margin)

    # each region's parts in its list order, around everything placed before (step 3; the shared obstacles keep the spacing between regions)
    anchors: dict[str, tuple[float, float, float, int]] = {}
    region_box: dict[str, Box] = {}
    for r in pcb.pack_regions:
        box = r.box()
        what = f"pack region {r.id} ({box[2] - box[0]:g} x {box[3] - box[1]:g} mm at ({box[0]:g}, {box[1]:g}))"
        anchors.update(_shelf_pack([(ref, fps[ref]) for ref in r.refs], _clip(box, inner_edge), obstacles, spacing, what, board_layers))
        for ref in r.refs:
            region_box[ref] = box

    # everything else into the free board area, every region an obstacle (step 4)
    rest = [ref for ref in order if ref not in owner]
    if rest:
        obstacles.boxes.extend((r.box(), f"pack region {r.id}") for r in pcb.pack_regions)
        anchors.update(_shelf_pack([(ref, fps[ref]) for ref in rest], inner_edge, obstacles, spacing,
                                   "the free board area (the parts neither fixed nor in a pack region)", board_layers))

    placements: list[Placement] = []
    region_of: dict[str, str] = {}
    for ref in order:
        if ref in given:
            p = given[ref].model_copy(deep=True)  # verbatim: position, rotation, side and its provenance
            region_of[ref] = FIXED
        elif ref in anchors:
            x, y, rot, row = anchors[ref]
            fp = fps[ref]
            box = region_box.get(ref)
            p = Placement(component_ref=ref, x_mm=x, y_mm=y, rotation_deg=rot, side=BoardSide.TOP,
                          provenance=_provenance(fp.lib_id, owner.get(ref, ""), box, row, rot, spacing, margin, fixed_refs))
            got = footprint_bbox(p, fp)
            assert got is not None
            placed[ref] = got
            region_of[ref] = owner.get(ref, FREE_AREA)
        else:
            continue
        placements.append(p)
    nowhere = [ref for ref in order if ref not in region_of]
    if nowhere:  # the packer refuses before this can happen; the guard behind it
        raise CompileError(f"component(s) {nowhere} ended up nowhere on {size}; refusing a board without them")
    _guard(order, placed, outline, region_box, {}, {}, [], fps, board_layers)  # its keep-out rule is the top side's: the keep-outs below
    for ref in order:
        ban = _keepout_ban(placed[ref], ref, given[ref].side if ref in given else BoardSide.TOP, _pad_layers(fps[ref], board_layers), areas)
        if ban is not None:
            raise CompileError(f"placed extent of {ref!r} ({placed[ref]}) lies in a keep-out that bans it ({ban}); refusing the placement")
    for t in texts:  # the packer kept the texts clear; the guard behind it
        tb = ir_text_box(t).bbox()
        hit = [ref for ref in order if ref not in given and _touch(tb, (placed[ref].x1, placed[ref].y1, placed[ref].x2, placed[ref].y2))]
        if hit:
            raise CompileError(f"packed part(s) {hit} lie on the silk text {t.text!r}; refusing the placement")
    return FixedPlacement(outline=outline, placements=placements, extents={ref: placed[ref] for ref in order}, region_of=region_of,
                          regions=[r.id for r in pcb.pack_regions], keepouts=honoured, texts=len(texts))
