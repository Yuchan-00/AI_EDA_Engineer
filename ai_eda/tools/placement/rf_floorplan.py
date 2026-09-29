"""Deterministic RF floorplan placement: block regions, chain order, shield cans first, keep-outs, per-region shelf packing (pure, no I/O beyond the library).

Invariant: every coordinate produced here is a function of the IR's
components, its RF blocks (``ir.rf.blocks``: ``refs``, ``chain``,
``shield_ref``, ``region`` ``x`` / ``y`` / ``w`` / ``h`` in mm), its
keep-outs (``ir.pcb.keepouts``), the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: extents and pads, never
model memory) and the parameters :data:`SPACING_MM`, :data:`MARGIN_MM` and
:data:`RING_MM`. Nothing is estimated and nothing is guessed: a component
without a footprint, a footprint that is not on disk or one whose extent
cannot be measured, a block naming a ref the IR lacks, a region outside the
outline or overlapping another, a shielded block without a region (its can
has nowhere to be centred), a can whose pads close no fence, and every
part that does not fit where it belongs raise
:class:`~ai_eda.errors.CompileError` - there is no fallback to overlapping
parts and no fallback to the grid placer (whose single cell, the board's
largest extent, cannot hold a board of this size).

What this is: the placement the PCB agent uses for every IR with RF blocks
(``ir.rf.blocks``, every kr447 board, audio_ptt included):

1. **Outline.** A user outline (``ir.pcb.outline``) is kept verbatim. Without
   one, the board is the bounding box of the block regions from ``(0, 0)``
   (regions are in the board frame, Y down, from the outline's top-left
   corner). Every region must lie inside the outline and no two regions may
   overlap.
2. **Blocks, in ``ir.rf.blocks`` order.** A block's parts are packed into
   its region, shrunk by ``SPACING_MM / 2`` on every side (so parts of
   neighbouring regions keep ``SPACING_MM``) and kept ``MARGIN_MM`` inside
   the board edge. The order is the block's ``chain`` (the signal path,
   row by row) and then its other refs in natural ref order.
3. **Shield cans.** A block with a ``shield_ref`` places its can first,
   rotation 0, centred in the region (a can that does not fit, or that meets
   a keep-out, is refused). The can's *fence* is read from its pads (the
   library footprint): each pad bounds the inner area on the side it lies
   on (left / right / top / bottom of the pads' box by the larger normalised
   offset from its centre); a side without a pad means no closed fence and
   is refused. Every other part of the block is packed inside the fence
   shrunk by :data:`RING_MM` (the ring kept free inside the fence). A can's
   extent covers its contents by design, so a part inside its own block's
   can is compared with the can's fence, not with its extent.
4. **Parts outside the regions** (in no block, or in a block without a
   region - which may not have a ``shield_ref``: that is refused before
   anything is packed) are packed the same way into the outline (``MARGIN_MM`` inside
   its edge), with every region (grown by ``SPACING_MM / 2``) as an
   obstacle, in natural ref order.
5. **Shelf packing** (every group above): rows from the top-left corner of
   the target box; each part in turn goes on the current row at the cursor,
   rotation 0 if its extent fits there, else rotation 90; a part that meets
   an obstacle (a placed part grown by ``SPACING_MM``, a keep-out area that
   forbids ``footprints`` - or ``pads`` - on a layer the part occupies and
   does not allow its ref, a region) moves right past it; a part that does
   not fit the row starts the next row below the row's tallest part (+
   ``SPACING_MM``); a row that no part could use moves down past the
   obstacle that blocked it. A part that fits nowhere in its box is refused
   with the box's size and the part's extent.

Before returning, the placed extents are re-measured with
:func:`~ai_eda.tools.kicad.geometry.footprint_bbox`: two extents that touch
(except a part inside its own can's fence), a part outside the outline, a
block part outside its region or a part in a keep-out it may not enter is a
:class:`CompileError` - the guard behind the arithmetic, as in the grid and
core-ring placers.

What this is not: an RF layout judgement. Whether the board is valid is
decided only by ``kicad-cli pcb drc`` on the compiled board (a can's
courtyard around its contents is DRC's to judge, not measured here); the
``pcb.keepout`` check judges keep-outs, regions and fences on the IR
geometry. Placement is always the top side.

Traceability: every :class:`~ai_eda.ir.Placement` carries ``derived``
provenance naming this tool (:data:`PLACER_ID` / :data:`PLACER_VERSION`),
the footprint, the block and its region, the row, the rotation, the can and
the parameters in ``derived_from``. ``Provenance.inputs`` stays empty: that
field is the calculator role map, and a placement is not a calculator
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, Placement, Provenance, ProvenanceKind
from ai_eda.tools.keepout import allowed_refs, area_bbox, area_points, box_area_overlap, covers_layer, forbids, keepout_id, keepouts_of
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_copper_center, to_board
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary
from ai_eda.tools.placement.grid import MARGIN_MM, SPACING_MM, _disjoint, _inside, _q, _resolve, footprint_extent

__all__ = [
    "PLACER_ID",
    "PLACER_VERSION",
    "RING_MM",
    "SPACING_MM",
    "MARGIN_MM",
    "FloorplanPlacement",
    "fence_box",
    "placed_fence_box",
    "region_box",
    "rf_blocks",
    "rf_floorplan_placement",
]

#: provenance ``tool`` / ``tool_version`` stamped on every placement
PLACER_ID = "placement.rf_floorplan"
PLACER_VERSION = "0.1"
#: the ring (mm) kept free inside a shield can's fence: the can's parts stay this far inside its pads
RING_MM = 1.0
_TOL = 1e-6
#: the footprint frame as a placement (rotation 0, top side, at the origin)
_ORIGIN = Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=0.0, side=BoardSide.TOP)

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FloorplanPlacement:
    """What :func:`rf_floorplan_placement` produced: the outline, the placements, the placed extents and where each part went."""

    outline: BoardOutline
    placements: list[Placement]
    extents: dict[str, BBox]
    #: ref -> the block id whose region holds it ("" for a part placed outside the regions)
    block_of: dict[str, str]
    #: block id -> its shield can's ref (only blocks with a can)
    cans: dict[str, str] = field(default_factory=dict)
    #: the keep-out ids the packer honoured (footprint / pad bans)
    keepouts: list[str] = field(default_factory=list)

    def description(self, origin: str, spacing: float, margin: float, ring: float) -> str:
        o = self.outline
        regions = sorted({b for b in self.block_of.values() if b})
        rest = sum(1 for b in self.block_of.values() if not b)
        text = (
            f"{PLACER_ID} {PLACER_VERSION}: {len(self.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm {origin} outline at "
            f"({o.origin_x_mm}, {o.origin_y_mm}), {len(regions)} block region(s) ({', '.join(regions)}), {len(self.cans)} shield can(s) placed first "
            f"with their parts inside the fence less a {ring} mm ring, {rest} part(s) outside the regions; shelf packing by each part's own extent "
            f"in chain order, spacing {spacing} mm, margin {margin} mm"
        )
        if self.keepouts:
            text += f"; keep-outs {', '.join(self.keepouts)} honoured"
        return text


# --------------------------------------------------------------------------- IR access (by the documented names)


def rf_blocks(ir: CircuitIR) -> list[Any]:
    """``ir.rf.blocks`` (empty when the IR carries no RF design)."""
    rf = getattr(ir, "rf", None)
    return list(getattr(rf, "blocks", None) or []) if rf is not None else []


def region_box(block: Any) -> Box | None:
    """The block's region as ``(x1, y1, x2, y2)`` mm, or ``None`` when it has none."""
    region = getattr(block, "region", None)
    if region is None:
        return None
    x, y, w, h = (float(getattr(getattr(region, name), "value", getattr(region, name))) for name in ("x", "y", "w", "h"))
    return x, y, x + w, y + h


# --------------------------------------------------------------------------- the fence of a can


def fence_box(fp: FootprintDef) -> BBox | None:
    """The inner area of a shield can's pad fence in the footprint frame (module docstring, step 3), or ``None`` when the pads close none."""
    copper = [p for p in fp.pads if p.pad_type != "np_thru_hole" and any(layer.endswith(".Cu") or layer == "*.Cu" for layer in p.layers)]
    if not copper:
        return None
    boxes: list[Box] = []
    for p in copper:
        turn = p.rotation % 180.0
        if abs(turn) < 1e-9:
            hw, hh = p.size_w / 2.0, p.size_h / 2.0
        elif abs(turn - 90.0) < 1e-9:
            hw, hh = p.size_h / 2.0, p.size_w / 2.0
        else:
            return None  # a pad at an odd angle: its box is not exact, the fence is not read
        x, y = pad_copper_center(_ORIGIN, p)
        boxes.append((x - hw, y - hh, x + hw, y + hh))
    X1, Y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    X2, Y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cx, cy, W, H = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0, (X2 - X1) / 2.0, (Y2 - Y1) / 2.0
    if W <= 0.0 or H <= 0.0:
        return None
    x1, y1, x2, y2 = -float("inf"), -float("inf"), float("inf"), float("inf")
    sides = set()
    for b in boxes:
        dx, dy = ((b[0] + b[2]) / 2.0 - cx) / W, ((b[1] + b[3]) / 2.0 - cy) / H
        if abs(dx) >= abs(dy):
            if dx < 0:
                x1, side = max(x1, b[2]), "left"
            else:
                x2, side = min(x2, b[0]), "right"
        else:
            if dy < 0:
                y1, side = max(y1, b[3]), "top"
            else:
                y2, side = min(y2, b[1]), "bottom"
        sides.add(side)
    if sides != {"left", "right", "top", "bottom"} or not (x2 > x1 and y2 > y1):
        return None
    return BBox(x1, y1, x2, y2)


def placed_fence_box(placement: Placement, fp: FootprintDef) -> BBox | None:
    """:func:`fence_box` in the board frame (exact for rotations that are multiples of 90 degrees; ``None`` otherwise)."""
    box = fence_box(fp)
    if box is None or abs(float(placement.rotation_deg) % 90.0) > 1e-9:
        return None
    corners = [to_board(placement, x, y) for x, y in ((box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2))]
    return BBox(min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners), max(c[1] for c in corners))


# --------------------------------------------------------------------------- packing


def _extent_at(fp: FootprintDef, rotation: float) -> BBox:
    """The footprint's extent around its own origin at ``rotation`` on the top side (exact at multiples of 90 degrees)."""
    if rotation == 0.0:
        return footprint_extent(fp)
    box = footprint_bbox(Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=rotation, side=BoardSide.TOP), fp)
    assert box is not None  # the extent at 0 was measurable
    return box


def _grow(b: Box, d: float) -> Box:
    return b[0] - d, b[1] - d, b[2] + d, b[3] + d


def _touch(a: Box, b: Box) -> bool:
    """Whether two boxes share a positive area."""
    return a[0] < b[2] - _TOL and b[0] < a[2] - _TOL and a[1] < b[3] - _TOL and b[1] < a[3] - _TOL


@dataclass
class _Obstacles:
    """What a part may not overlap: other parts (grown by the spacing), regions, and keep-out areas that ban it."""

    boxes: list[tuple[Box, str]] = field(default_factory=list)
    #: (keep-out id, area points, area bbox, the refs it allows, the layers it bans footprints on, the layers it bans pads on)
    areas: list[tuple[str, list[tuple[float, float]], Box, frozenset[str], bool, bool, Any]] = field(default_factory=list)

    def hit(self, box: Box, ref: str, pad_layers: frozenset[str]) -> Box | None:
        """The bbox of the first obstacle ``box`` overlaps (for ``ref``), or ``None``."""
        for b, _what in self.boxes:
            if _touch(box, b):
                return b
        for _kid, pts, bb, refs, _f, _p, ko in self.areas:
            if ref in refs:
                continue
            banned = (forbids(ko, "footprints") and covers_layer(ko, "F.Cu")) or (forbids(ko, "pads") and any(covers_layer(ko, layer) for layer in pad_layers))
            if banned and _touch(box, bb) and box_area_overlap(box, pts) > 0.0:
                return bb
        return None


def _pad_layers(fp: FootprintDef, board_layers: list[str]) -> frozenset[str]:
    """The copper layers a top-side part's pads are on (a through-hole or ``*.Cu`` pad: every board copper layer)."""
    out: set[str] = set()
    for p in fp.pads:
        if p.pad_type in ("thru_hole", "np_thru_hole") or "*.Cu" in p.layers:
            out.update(board_layers)
        else:
            out.update(layer for layer in p.layers if layer.endswith(".Cu"))
    return frozenset(out)


def _shelf_pack(
    entries: list[tuple[str, FootprintDef]], target: Box, obstacles: _Obstacles, spacing: float, what: str, board_layers: list[str],
) -> dict[str, tuple[float, float, float, int]]:
    """Pack ``entries`` into ``target`` (module docstring, step 5): ``ref -> (x, y, rotation, row)`` of each placement anchor."""
    x1, y1, x2, y2 = target
    if not (x2 - x1 > _TOL and y2 - y1 > _TOL):
        raise CompileError(f"{what}: no room at all ({x2 - x1:.3f} x {y2 - y1:.3f} mm after the margin and spacing)")
    out: dict[str, tuple[float, float, float, int]] = {}
    cy, cx, shelf_h, row = y1, x1, 0.0, 0
    for ref, fp in entries:
        options = [(0.0, _extent_at(fp, 0.0)), (90.0, _extent_at(fp, 90.0))]
        layers = _pad_layers(fp, board_layers)
        min_h = min(e.height for _, e in options)
        if all(e.width > x2 - x1 + _TOL or e.height > y2 - y1 + _TOL for _, e in options):
            e = options[0][1]
            raise CompileError(
                f"{what}: {ref} ({fp.lib_id}, {e.width:.3f} x {e.height:.3f} mm) is larger than the box it belongs in "
                f"({x2 - x1:.3f} x {y2 - y1:.3f} mm after the margin and spacing) in both orientations; the region is refused, parts never overlap"
            )
        while True:
            if cy + min_h > y2 + _TOL:
                raise CompileError(
                    f"{what}: {ref} ({fp.lib_id}) does not fit - the {x2 - x1:.3f} x {y2 - y1:.3f} mm box is full after {len(out)} part(s); "
                    "the region is refused, parts never overlap"
                )
            spot = None
            for rot, e in options:
                x = cx
                while x + e.width <= x2 + _TOL and cy + e.height <= y2 + _TOL:
                    box = (x, cy, x + e.width, cy + e.height)
                    blocker = obstacles.hit(box, ref, layers)
                    if blocker is None:
                        spot = (rot, e, box)
                        break
                    x = max(x + _TOL * 10, blocker[2])
                if spot is not None:
                    break
            if spot is not None:
                rot, e, box = spot
                out[ref] = (_q(box[0] - e.x1), _q(box[1] - e.y1), rot, row)
                obstacles.boxes.append((_grow(box, spacing), ref))
                shelf_h = max(shelf_h, e.height)
                cx = box[2] + spacing
                break
            # next row: below this row's parts, or past whatever blocks an empty row
            if shelf_h > 0.0:
                cy = cy + shelf_h + spacing
            else:
                band = (x1, cy, x2, cy + min_h)
                below = [b[3] for b, _ in obstacles.boxes if _touch(band, b) and b[3] > cy + _TOL]
                below += [bb[3] for _k, pts, bb, refs, _f, _p, _ko in obstacles.areas if ref not in refs and _touch(band, bb) and bb[3] > cy + _TOL]
                if not below:
                    raise CompileError(f"{what}: {ref} ({fp.lib_id}) fits no row of the {x2 - x1:.3f} x {y2 - y1:.3f} mm box")
                cy = min(below)
            cx, shelf_h, row = x1, 0.0, row + 1
    return out


def _provenance(fp: FootprintDef, block: str, region: Box | None, row: int, rotation: float, spacing: float, margin: float,
                ring: float, can: str | None) -> Provenance:
    derived = [f"footprint:{fp.lib_id}", f"block:{block or '(outside the regions)'}"]
    if region is not None:
        derived.append("region:" + ",".join(f"{v:g}" for v in (region[0], region[1], region[2] - region[0], region[3] - region[1])))
    derived += [f"row:{row}", f"rotation:{rotation:g}", f"spacing_mm:{spacing}", f"margin_mm:{margin}"]
    if can is not None:
        derived.append(f"inside_can:{can},ring_mm:{ring}")
    return Provenance(
        kind=ProvenanceKind.DERIVED, tool=PLACER_ID, tool_version=PLACER_VERSION, derived_from=derived,
        note="shelf packing in the RF block regions from library footprint extents; validity is decided by kicad-cli DRC only",
    )


def rf_floorplan_placement(
    ir: CircuitIR,
    library: KicadLibrary,
    *,
    spacing: float = SPACING_MM,
    margin: float = MARGIN_MM,
    ring: float = RING_MM,
    outline: BoardOutline | None = None,
    keepouts: list[Any] | None = None,
) -> FloorplanPlacement:
    """Place every component of ``ir`` by its RF block's region (module docstring); pure (same IR + library + parameters -> same result).

    ``outline`` (the user's) is kept verbatim; ``keepouts`` default to
    ``ir.pcb.keepouts``. Raises :class:`CompileError` for every case the
    module docstring lists.
    """
    if spacing <= 0 or margin < 0 or ring < 0:
        raise CompileError("spacing must be > 0, margin and ring >= 0")
    parts = _resolve(ir, library)
    fps = dict(parts)
    order = [ref for ref, _ in parts]
    blocks = rf_blocks(ir)
    if not blocks:
        raise CompileError("no RF blocks (ir.rf.blocks): the floorplan placer has no regions to pack")
    kos = list(keepouts) if keepouts is not None else keepouts_of(ir.pcb)
    block_of: dict[str, str] = {}
    for b in blocks:
        bid = str(getattr(b, "id", "?"))
        for ref in getattr(b, "refs", []) or []:
            if ref not in fps:
                raise CompileError(f"block {bid} names {ref!r}, which is not a component of the IR")
            if ref in block_of:
                raise CompileError(f"{ref} is in two blocks ({block_of[ref]}, {bid})")
            block_of[ref] = bid
        shield = getattr(b, "shield_ref", None)
        if shield is not None and shield not in (getattr(b, "refs", []) or []):
            raise CompileError(f"block {bid}: shield_ref {shield!r} is not one of its refs")
        if shield is not None and region_box(b) is None:
            raise CompileError(f"block {bid}: shield can {shield} but no region - a shielded block needs a region (its can is placed first, centred "
                               "in it, and its parts inside the fence); packed outside the regions the can would sit beside its own parts")
        stray = [r for r in getattr(b, "chain", []) or [] if r not in (getattr(b, "refs", []) or [])]
        if stray:
            raise CompileError(f"block {bid}: chain names {stray}, which are not refs of the block")
    regions = [(str(getattr(b, "id", "?")), b, region_box(b)) for b in blocks]
    with_region = [(bid, b, box) for bid, b, box in regions if box is not None]
    for bid, _b, box in with_region:
        if not (box[2] - box[0] > 0 and box[3] - box[1] > 0):
            raise CompileError(f"block {bid}: region {box} has no area")
    if outline is None:
        if not with_region:
            raise CompileError("no board outline and no block region: nothing sizes the board")
        outline = BoardOutline(width_mm=_q(max(r[2] for _, _, r in with_region)), height_mm=_q(max(r[3] for _, _, r in with_region)),
                               origin_x_mm=0.0, origin_y_mm=0.0)
    ox, oy = float(outline.origin_x_mm), float(outline.origin_y_mm)
    edge = (ox, oy, ox + float(outline.width_mm), oy + float(outline.height_mm))
    for bid, _b, box in with_region:
        if box[0] < edge[0] - _TOL or box[1] < edge[1] - _TOL or box[2] > edge[2] + _TOL or box[3] > edge[3] + _TOL:
            raise CompileError(f"block {bid}: region ({box[0]:g}, {box[1]:g})-({box[2]:g}, {box[3]:g}) leaves the {outline.width_mm} x {outline.height_mm} mm outline")
    for i, (a, _ba, ra) in enumerate(with_region):
        for c, _bc, rc in with_region[i + 1:]:
            if _touch(ra, rc):
                raise CompileError(f"the regions of blocks {a} and {c} overlap")
    inner_edge = (edge[0] + margin, edge[1] + margin, edge[2] - margin, edge[3] - margin)
    board_layers = [layer.name for layer in ir.pcb.layers] if ir.pcb is not None else ["F.Cu", "B.Cu"]
    honoured: list[str] = []
    areas = []
    for ko in kos:
        if forbids(ko, "footprints") or forbids(ko, "pads"):
            pts = area_points(ko)
            areas.append((keepout_id(ko), pts, area_bbox(pts), frozenset(allowed_refs(ko)), forbids(ko, "footprints"), forbids(ko, "pads"), ko))
            honoured.append(keepout_id(ko))
    anchors: dict[str, tuple[float, float, float, int]] = {}
    region_of: dict[str, Box] = {}
    can_of: dict[str, str] = {}
    cans: dict[str, str] = {}
    fence_of: dict[str, BBox] = {}
    for bid, b, box in with_region:
        refs = list(getattr(b, "refs", []) or [])
        if not refs:
            continue
        chain = list(getattr(b, "chain", []) or [])
        shield = getattr(b, "shield_ref", None)
        rest = sorted((r for r in refs if r not in chain and r != shield), key=natural_ref_key)
        members = [r for r in chain if r != shield] + rest
        target = _clip(_grow(box, -spacing / 2.0), inner_edge)
        obstacles = _Obstacles(areas=list(areas))
        what = f"block {bid} (region {box[2] - box[0]:g} x {box[3] - box[1]:g} mm at ({box[0]:g}, {box[1]:g}))"
        if shield is not None:
            fp = fps[shield]
            e = footprint_extent(fp)
            if e.width > target[2] - target[0] + _TOL or e.height > target[3] - target[1] + _TOL:
                raise CompileError(
                    f"{what}: the shield can {shield} ({fp.lib_id}, {e.width:.3f} x {e.height:.3f} mm) does not fit its region "
                    f"({target[2] - target[0]:.3f} x {target[3] - target[1]:.3f} mm after the margin and spacing); the region is refused"
                )
            left = (target[0] + target[2]) / 2.0 - e.width / 2.0
            top = (target[1] + target[3]) / 2.0 - e.height / 2.0
            can_box = (left, top, left + e.width, top + e.height)
            if obstacles.hit(can_box, shield, _pad_layers(fp, board_layers)) is not None:
                raise CompileError(f"{what}: the shield can {shield} centred in its region meets a keep-out that bans it")
            anchors[shield] = (_q(left - e.x1), _q(top - e.y1), 0.0, 0)
            fence = fence_box(fp)
            if fence is None:
                raise CompileError(f"{what}: the pads of the shield can {shield} ({fp.lib_id}) close no fence on four sides; its inner area is not read")
            ax, ay = anchors[shield][0], anchors[shield][1]
            inner = (ax + fence.x1 + ring, ay + fence.y1 + ring, ax + fence.x2 - ring, ay + fence.y2 - ring)
            fence_of[shield] = BBox(ax + fence.x1, ay + fence.y1, ax + fence.x2, ay + fence.y2)
            cans[bid] = shield
            region_of[shield] = box
            got = _shelf_pack([(r, fps[r]) for r in members], inner, obstacles, spacing, f"{what}, inside the fence of {shield} less {ring} mm", board_layers)
            for r in members:
                can_of[r] = shield
        else:
            got = _shelf_pack([(r, fps[r]) for r in members], target, obstacles, spacing, what, board_layers)
        anchors.update(got)
        for r in members:
            region_of[r] = box
    outside = [r for r in order if r not in region_of]
    if outside:
        obstacles = _Obstacles(boxes=[(_grow(box, spacing / 2.0), f"region {bid}") for bid, _b, box in with_region], areas=list(areas))
        got = _shelf_pack([(r, fps[r]) for r in outside], inner_edge, obstacles, spacing, "the parts outside the block regions", board_layers)
        anchors.update(got)
    placements: list[Placement] = []
    placed: dict[str, BBox] = {}
    for ref in order:
        x, y, rot, row = anchors[ref]
        fp = fps[ref]
        region = region_of.get(ref)
        p = Placement(
            component_ref=ref, x_mm=x, y_mm=y, rotation_deg=rot, side=BoardSide.TOP,
            provenance=_provenance(fp, block_of.get(ref, "") if region is not None else "", region, row, rot, spacing, margin, ring, can_of.get(ref)),
        )
        box = footprint_bbox(p, fp)
        assert box is not None
        placements.append(p)
        placed[ref] = box
    _guard(order, placed, outline, region_of, can_of, fence_of, areas, fps, board_layers)
    return FloorplanPlacement(
        outline=outline, placements=placements, extents=placed, block_of={r: (block_of.get(r, "") if r in region_of else "") for r in order},
        cans=cans, keepouts=honoured,
    )


def _clip(a: Box, b: Box) -> Box:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _guard(
    order: list[str], placed: dict[str, BBox], outline: BoardOutline, region_of: dict[str, Box], can_of: dict[str, str],
    fence_of: dict[str, BBox], areas: list, fps: dict[str, FootprintDef], board_layers: list[str],
) -> None:
    """The re-measured extents against every rule of the module docstring; a violation is a :class:`CompileError`."""
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if can_of.get(b) == a or can_of.get(a) == b:
                inner, can = (a, b) if can_of.get(a) == b else (b, a)
                f = fence_of[can]
                e = placed[inner]
                if not (f.x1 + _TOL < e.x1 and e.x2 < f.x2 - _TOL and f.y1 + _TOL < e.y1 and e.y2 < f.y2 - _TOL):
                    raise CompileError(f"placed extent of {inner!r} ({e}) leaves the fence of its can {can!r} ({f}); refusing the floorplan")
                continue
            if not _disjoint(placed[a], placed[b]):
                raise CompileError(f"placed extents of {a!r} and {b!r} touch or overlap ({placed[a]} vs {placed[b]}); refusing the floorplan")
    outside = [r for r in order if not _inside(placed[r], outline)]
    if outside:
        raise CompileError(
            f"component(s) {outside} do not fit inside the {outline.width_mm} x {outline.height_mm} mm outline at "
            f"({outline.origin_x_mm}, {outline.origin_y_mm}); the outline is kept as given, not resized"
        )
    for ref, box in region_of.items():
        e = placed[ref]
        if e.x1 < box[0] - _TOL or e.y1 < box[1] - _TOL or e.x2 > box[2] + _TOL or e.y2 > box[3] + _TOL:
            raise CompileError(f"placed extent of {ref!r} ({e}) leaves its block's region {box}; refusing the floorplan")
    obstacles = _Obstacles(areas=list(areas))
    for ref in order:
        e = placed[ref]
        if obstacles.hit((e.x1, e.y1, e.x2, e.y2), ref, _pad_layers(fps[ref], board_layers)) is not None:
            raise CompileError(f"placed extent of {ref!r} ({e}) lies in a keep-out that bans it; refusing the floorplan")
