"""Opt-in functional placement, version 1.

Library geometry and declared terminal associations are the only pin facts.
CHOICE: decoupling pad-edge target 3 mm, oscillator pad-edge target
5 mm, reset/boot target 10 mm, 1 mm part spacing, 2 mm vertical access clearance.
Oscillator distance is a placement target; routed copper still needs measuring.
The router's effective fan-out reservation is a hard obstacle. Missed distance
targets are recorded. Packing is deterministic and bounded, not exhaustive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import (BoardOutline, CircuitIR, Component, Keepout, LibraryRef, NetKind,
                       Placement, Provenance, ProvenanceKind, SpiceBinding, Traced)
from ai_eda.tools.functional_intent import resolve_intent
from ai_eda.tools.keepout import allowed_refs, box_area_overlap, covers_layer, forbids, area_points, keepouts_of
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center, pads_bbox, rotate
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary
from ai_eda.tools.placement.core_ring import find_core
from ai_eda.tools.placement.grid import _disjoint, _inside, _q, _resolve, footprint_extent
from ai_eda.tools.placement.outline import MAX_SIDE_MM, UTILIZATION, rectangles
from ai_eda.tools.placement.rf_floorplan import FANOUT_PITCH_MM
from ai_eda.tools.routing.maze import RoutingParams, effective_params, fanout_margin, footprint_pad_pitch

PLACER_ID = "placement.functional"
PLACER_VERSION = "1"
ACCESS_MM = 2.0
HOLE_FOOTPRINT = LibraryRef(library="MountingHole", name="MountingHole_3.2mm_M3")
HOLE_NOTE = "CHOICE: four corner M3 clearance holes; size and hardware/access envelope UNVERIFIED pending mechanical requirements"
HOLE_MIN_AREA_MM2 = 400
HOLE_RULE = "CHOICE: automatic M3 holes only when the feasible outline without holes is at least 400 mm^2"
RULES = ["CHOICE:decoupling_pad_edge_mm=3", "CHOICE:oscillator_connection_mm=5;placement_bound_only",
         "CHOICE:reset_boot_pad_edge_mm=10", "CHOICE:utilization=0.4;outline_step_mm=1;max_side_mm=160;max_aspect=2",
         "CHOICE:vertical_access_mm=2", HOLE_RULE, "anchors:holes,edge_connectors,power,core,support,blocks"]


def provenance(*entries: str) -> Provenance:
    return Provenance(kind=ProvenanceKind.DERIVED, tool=PLACER_ID, tool_version=PLACER_VERSION,
                      derived_from=[*RULES, *entries], created_at=datetime(1970, 1, 1, tzinfo=timezone.utc))


def grow(b: BBox, mm: float) -> BBox:
    return BBox(b.x1 - mm, b.y1 - mm, b.x2 + mm, b.y2 + mm)


def centre(b: BBox) -> tuple[float, float]:
    return (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2


def at(ref: str, fp: FootprintDef, x: float, y: float, angle: int = 0) -> Placement:
    p = Placement(component_ref=ref, x_mm=0, y_mm=0, rotation_deg=angle)
    cx, cy = centre(footprint_bbox(p, fp))
    return p.model_copy(update={"x_mm": _q(x - cx), "y_mm": _q(y - cy)})


def _pad_shape(p, pad):
    """A quarter-turned rounded rectangle as an inner rectangle plus a radius."""
    if pad.shape not in ("rect", "roundrect", "oval", "circle") or pad.offset_x or pad.offset_y or (p.rotation_deg + pad.rotation) % 90:
        raise CompileError(f"functional pad-distance: unsupported geometry at {p.component_ref}.{pad.number}")
    radius = min(pad.size_w, pad.size_h) * (pad.roundrect_rratio or 0) if pad.shape == "roundrect" else min(pad.size_w, pad.size_h) / 2 if pad.shape in ("circle", "oval") else 0
    w, h = pad.size_w / 2 - radius, pad.size_h / 2 - radius
    if (p.rotation_deg + pad.rotation) % 180:
        w, h = h, w
    x, y = pad_center(p, pad)
    return (x - w, y - h, x + w, y + h), radius


def connection_distance(p: Placement, fp: FootprintDef, number: str, target: Placement, target_fp: FootprintDef,
                        target_number: str) -> float:
    """Exact closest copper edges for quarter-turned rect/roundrect/oval/circle pads.

    This is a necessary placement bound, never a claim about routed length.
    """
    aa = [_pad_shape(p, pad) for pad in fp.pads if pad.number == number]
    bb = [_pad_shape(target, pad) for pad in target_fp.pads if pad.number == target_number]
    if not aa or not bb:
        raise CompileError(f"functional association: library pad missing: {p.component_ref}.{number} -> {target.component_ref}.{target_number}")
    return min(max(0, math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0)) - ra - rb)
               for a, ra in aa for b, rb in bb)


def association_limit(a) -> tuple[float, bool]:
    role = a.role.lower()
    if any(word in role for word in ("crystal", "oscillator", "clock")):
        return 5.0, True
    if any(word in role for word in ("decoupl", "supply", "regulator capacitor")):
        return 3.0, False
    return 10.0, False


def pin_side(fp: FootprintDef, number: str) -> str:
    pad = fp.pad(number)
    if pad is None:
        raise CompileError(f"functional: {fp.lib_id} has no pad {number}")
    b = footprint_extent(fp)
    cx, cy = centre(b)
    dx, dy = (pad.x - cx) / max(b.width, 1e-6), (pad.y - cy) / max(b.height, 1e-6)
    return ("east" if dx >= 0 else "west") if abs(dx) >= abs(dy) else ("south" if dy >= 0 else "north")


NORMAL = {"west": (-1, 0), "east": (1, 0), "north": (0, -1), "south": (0, 1)}


@dataclass
class FunctionalPlacement:
    outline: BoardOutline
    placements: list[Placement]
    components: list[Component]
    keepouts: list[Keepout]
    notes: list[str]
    extents: dict[str, BBox]

    def description(self) -> str:
        return f"{PLACER_ID} {PLACER_VERSION}: {len(self.placements)} footprints on {self.outline.width_mm:g} x {self.outline.height_mm:g} mm; bounded area/aspect rectangle search"


def mounting_components(ir: CircuitIR, library: KicadLibrary) -> list[Component]:
    """Add real, unnumbered-hole library parts; never reuse an occupied reference."""
    fp = library.resolve_footprint(HOLE_FOOTPRINT)
    symbol = library.resolve_symbol(LibraryRef(library="Mechanical", name="MountingHole"))
    if not fp.verified or not symbol.verified:
        raise CompileError("functional mounting holes: MountingHole_3.2mm_M3 footprint and Mechanical:MountingHole symbol must exist in the library")
    refs = {c.ref for c in ir.components}
    out = []
    i = 1
    while len(out) < 4:
        ref = f"H{i}"
        i += 1
        if ref in refs:
            continue
        out.append(Component(ref=ref, value="MountingHole", description=HOLE_NOTE, footprint=fp, symbol=symbol, board_only=True,
                             provenance=provenance(HOLE_NOTE), spice=SpiceBinding(exclude=True, exclude_reason="mechanical mounting hole", provenance=provenance(HOLE_NOTE))))
    return out


def functional_placement(ir: CircuitIR, library: KicadLibrary, *, spacing: float = 1.0, margin: float = 2.0,
                         outline: BoardOutline | None = None, routing: RoutingParams | None = None) -> FunctionalPlacement:
    """Choose automatic mounting holes from the feasible outline before their reservations are added."""
    fixed = outline if outline is not None else (ir.pcb.outline if ir.pcb else None)
    lower_area = (fixed.width_mm * fixed.height_mm if fixed is not None else
                  sum((footprint_extent(fp).width + spacing) * (footprint_extent(fp).height + spacing)
                      for _, fp in _resolve(ir, library)) / UTILIZATION)
    if lower_area >= HOLE_MIN_AREA_MM2:
        return _functional_placement(ir, library, spacing=spacing, margin=margin, outline=outline, routing=routing, mounting=True)
    bare = _functional_placement(ir, library, spacing=spacing, margin=margin, outline=outline, routing=routing, mounting=False)
    if bare.outline.width_mm * bare.outline.height_mm < HOLE_MIN_AREA_MM2:
        bare.notes.append(f"{HOLE_RULE}; omitted")
        return bare
    return _functional_placement(ir, library, spacing=spacing, margin=margin, outline=outline, routing=routing, mounting=True)


def _functional_placement(ir: CircuitIR, library: KicadLibrary, *, spacing: float, margin: float,
                          outline: BoardOutline | None, routing: RoutingParams | None, mounting: bool) -> FunctionalPlacement:
    if not all(math.isfinite(v) and v >= 0 for v in (spacing, margin)):
        raise CompileError("functional: spacing and margin must be finite and nonnegative")
    try:
        intent = resolve_intent(ir)
    except ValueError as e:
        raise CompileError(str(e)) from e
    base = ir.pcb
    if base and (base.pack_regions or base.fixed):
        raise CompileError("functional: fixed parts / pack_regions require placement.fixed; no position or region is silently ignored")
    outline = outline if outline is not None else (base.outline if base else None)
    added = mounting_components(ir, library) if mounting else []
    working = ir.model_copy(update={"components": [*ir.components, *added]})
    fps = dict(_resolve(working, library))
    parts = _resolve(ir, library)
    core, core_fp, _ = find_core([(r, fp) for r, fp in parts if not r.startswith(("J", "P", "H"))] or parts)
    params = effective_params(ir, routing or RoutingParams.for_board(ir, library))[0]
    fan = {ref: fanout_margin(params) for ref, fp in fps.items()
           if (pitch := footprint_pad_pitch(fp)) is not None and pitch < FANOUT_PITCH_MM}
    fan_stamp = [f"fanout:{r}={m:g};{params.derived_from_entry()}" for r, m in sorted(fan.items())]
    associations = {}
    for a in intent.pin_associations:
        if fps[a.terminal.component_ref].pad(a.terminal.pin_number) is None or fps[a.ic_pin.component_ref].pad(a.ic_pin.pin_number) is None:
            raise CompileError(f"functional association: library pad missing: {a.terminal} -> {a.ic_pin}")
        associations.setdefault(a.terminal.component_ref, []).append(a)
    for rows in associations.values():
        rows.sort(key=lambda a: (a.ic_pin.component_ref, a.ic_pin.pin_number, a.terminal.pin_number, a.role))
    components = {c.ref: c for c in ir.components}
    blocks = {r: b for b in intent.blocks for r in b.component_refs}
    power = {r for b in intent.blocks if b.domain.value == "power" or "regulat" in b.function.lower() for r in b.component_refs}
    connectors = [r for r in fps if r in components and r != core and r.startswith(("J", "P")) and r not in associations]
    # A support header (boot strap) stays beside its associated pin, not on the perimeter.
    sides, links = {}, {}
    for ref in connectors:
        pairs = []
        for n in sorted(ir.nets, key=lambda n: n.name):
            if n.kind in (NetKind.POWER, NetKind.GROUND):
                continue
            pairs.extend((p.pin_number, q.pin_number) for p in n.pins if p.component_ref == ref
                         for q in n.pins if q.component_ref == core)
        links[ref] = pairs
        counts = {s: sum(pin_side(core_fp, q) == s for _, q in pairs) for s in NORMAL}
        sides[ref] = max(NORMAL, key=lambda s: counts[s]) if pairs else "north"
    # Precompute relative core/support geometry once; translations preserve every constraint.
    local: dict[str, Placement] = {core: at(core, core_fp, 0, 0)}
    local_boxes = {core: footprint_bbox(local[core], core_fp)}
    existing_keepouts = list(keepouts_of(base)) if base else []

    @lru_cache(maxsize=16384)
    def bounds(ref, x, y, angle):
        p = Placement(component_ref=ref, x_mm=x, y_mm=y, rotation_deg=angle)
        return footprint_bbox(p, fps[ref]), grow(pads_bbox(p, fps[ref]), fan[ref] - 1e-6) if ref in fan else None

    def boxes(p):
        return bounds(p.component_ref, p.x_mm, p.y_mm, p.rotation_deg)

    def room(p, placed):
        box, reservation = boxes(p)
        for ref, q in placed.items():
            other, other_reservation = boxes(q)
            # The physical extent and each direction of the fan-out rule both matter.
            if not _disjoint(box, grow(other, max(spacing - 1e-6, 0))):
                return False
            if other_reservation is not None and not _disjoint(box, other_reservation):
                return False
            if reservation is not None and not _disjoint(other, reservation):
                return False
        return True

    pending = set(associations) - {core}
    while pending:
        ready = [r for r in pending if all(a.ic_pin.component_ref in local for a in associations[r])]
        # Regulator support is placed with the power block after its anchor exists.
        if not ready:
            break
        ref = min(ready, key=lambda r: (0 if any(association_limit(a)[1] for a in associations[r]) else 1 if min(association_limit(a)[0] for a in associations[r]) == 3 else 2,
                                       -footprint_extent(fps[r]).width * footprint_extent(fps[r]).height, natural_ref_key(r)))
        local[ref] = _support(ref, fps, associations[ref], local, room)
        local_boxes[ref] = footprint_bbox(local[ref], fps[ref])
        pending.remove(ref)
    if any(a.ic_pin.component_ref == core for r in pending for a in associations[r]):
        raise CompileError("functional: unresolved support association dependency")
    local_extent = BBox(min(b.x1 for b in local_boxes.values()), min(b.y1 for b in local_boxes.values()),
                        max(b.x2 for b in local_boxes.values()), max(b.y2 for b in local_boxes.values()))
    inflated = sum((footprint_extent(fp).width + spacing) * (footprint_extent(fp).height + spacing)
                   for r, fp in fps.items() if r not in {c.ref for c in added})
    hole_half = max([4.0, *(max(footprint_extent(fps[c.ref]).width, footprint_extent(fps[c.ref]).height) / 2 for c in added)]) if added else 0.0
    mechanical_area = 4 * (2 * hole_half) ** 2
    min_w, min_h = local_extent.width + 2 * margin, local_extent.height + 2 * margin
    for side in NORMAL:
        members = [r for r in connectors if sides[r] == side]
        length = sum((min if "Vertical" in fps[r].name or "HDR" in fps[r].name else max)(
            footprint_extent(fps[r]).width, footprint_extent(fps[r]).height) for r in members)
        length += max(spacing, ACCESS_MM) * max(0, len(members) - 1) + 4 * hole_half + 2 * margin + 2 * ACCESS_MM
        if side in ("west", "east"):
            min_h = max(min_h, length)
        else:
            min_w = max(min_w, length)
    last = "no rectangle within the search bounds"
    for candidate in rectangles(inflated, mechanical_area, fixed=outline, min_width=min_w, min_height=min_h):
        try:
            placed, keepouts = _pack(candidate, fps, core, local, connectors, sides, links, power, pending,
                                     associations, blocks, added, hole_half, existing_keepouts, room, spacing, margin, boxes)
        except CompileError as e:
            last = str(e)
            continue
        notes = [*([HOLE_NOTE] if added else []), *intent.unassociated,
                 "CHOICE: seed = inflated footprint area / 0.4 + exclusive corner reservations; first feasible bounded rectangle",
                 "UNVERIFIED: oscillator routed lengths and ground return paths require routed-copper measurement"]
        for block in intent.blocks:
            if any(word in block.function.lower() for word in ("clock", "crystal", "reset", "boot")):
                missing = [r for r in block.component_refs if r != core and r not in associations]
                if missing:
                    notes.append(f"UNVERIFIED: block {block.id} lacks pin associations for {', '.join(missing)}; pin-distance constraints cannot be checked")
        for ref, p in placed.items():
            entries = [f"footprint:{fps[ref].lib_id}", f"spacing_mm={spacing:g};margin_mm={margin:g}", *fan_stamp]
            if ref in sides:
                vertical = "Vertical" in fps[ref].name or "HDR" in fps[ref].name
                direction = fps[ref].properties.get("mating_direction")
                entries.append(f"edge:{sides[ref]};pin_order:net_pad_distance;access={'vertical' if vertical else direction or 'UNVERIFIED mating direction'}")
                if not vertical and not direction:
                    notes.append(f"UNVERIFIED: {ref} mating direction requires mechanical intent; edge placement alone does not establish the opening")
            entries.extend(f"association:{a.terminal}->{a.ic_pin};role={a.role};limit_mm={association_limit(a)[0]:g}" for a in associations.get(ref, []))
            for a in associations.get(ref, []):
                distance = connection_distance(p, fps[ref], a.terminal.pin_number, placed[a.ic_pin.component_ref],
                                               fps[a.ic_pin.component_ref], a.ic_pin.pin_number)
                limit = association_limit(a)[0]
                if distance > limit + 1e-6:
                    note = (f"constraint not met: {ref} {distance:.3f} mm > {limit:g} mm "
                            f"(FANOUT reservation, anchor extents or spacing; {a.terminal}->{a.ic_pin})")
                    entries.append(note)
                    notes.append(note)
            if ref in blocks:
                entries.append(f"block:{blocks[ref].id};inferred={blocks[ref].inferred}")
            if ref in {c.ref for c in added}:
                entries.append(HOLE_NOTE)
            placed[ref] = p.model_copy(update={"provenance": provenance(*entries)})
        ordered = [placed[r] for r in sorted(placed, key=natural_ref_key)]
        return FunctionalPlacement(candidate, ordered, added, keepouts, notes,
                                   {r: footprint_bbox(p, fps[r]) for r, p in placed.items()})
    raise CompileError(f"functional constraints infeasible in {'fixed outline' if outline else f'bounded search <= {MAX_SIDE_MM} mm'}: {last}")


def _support(ref, fps, associations, placed, room):
    a = associations[0]
    target = placed[a.ic_pin.component_ref]
    target_fp = fps[a.ic_pin.component_ref]
    px, py = pad_center(target, target_fp.pad(a.ic_pin.pin_number))
    nx, ny = NORMAL[pin_side(target_fp, a.ic_pin.pin_number)]
    # Quarter-mm candidates, fixed order; board-frame rotation follows the IC.
    nx, ny = rotate(nx, ny, target.rotation_deg)
    zero = at(ref, fps[ref], 0, 0)
    support_radius = max(math.hypot(*pad_center(zero, pad)) + math.hypot(pad.size_w, pad.size_h) / 2
                         for pad in fps[ref].pads if pad.number == a.terminal.pin_number)
    target_radius = max(math.dist((px, py), pad_center(target, pad)) + math.hypot(pad.size_w, pad.size_h) / 2
                        for pad in target_fp.pads if pad.number == a.ic_pin.pin_number)
    best = None
    # Expand until no unsearched centre can improve the pad-edge distance.
    for reach in (12, 24, 48, 96, MAX_SIDE_MM):
        choices = []
        for angle in (0, 90, 180, 270):
            origin = at(ref, fps[ref], 0, 0, angle)
            for normal in range(4, reach * 4 + 1):
                for tangent in range(-reach * 4, reach * 4 + 1):
                    x = px + nx * normal / 4 - ny * tangent / 4
                    y = py + ny * normal / 4 + nx * tangent / 4
                    p = origin.model_copy(update={"x_mm": _q(origin.x_mm + x), "y_mm": _q(origin.y_mm + y)})
                    distances = [connection_distance(p, fps[ref], b.terminal.pin_number, placed[b.ic_pin.component_ref],
                                 fps[b.ic_pin.component_ref], b.ic_pin.pin_number) for b in associations]
                    missed = any(d > association_limit(b)[0] + 1e-6 for d, b in zip(distances, associations))
                    score = (missed, max(distances), sum(distances), abs(tangent), normal, angle, tangent)
                    if best is not None and score >= best[0]:
                        continue
                    if reach > 12:
                        if room(p, placed):
                            best = score, p
                    else:
                        choices.append((score, p))
        for score, p in sorted(choices, key=lambda c: c[0]):
            if room(p, placed):
                best = score, p
                break
        if best is not None:
            frontier = reach - support_radius - target_radius - 0.001
            if frontier > max(best[0][1], *(association_limit(b)[0] for b in associations)) or reach == MAX_SIDE_MM:
                return best[1]
    names = ", ".join(f"{a.terminal}->{a.ic_pin} <= {association_limit(a)[0]:g} mm" for a in associations)
    raise CompileError(f"support {ref}: no feasible geometry on assigned pin side in bounded search ({names})")


def _pack(o, fps, core, local, connectors, sides, links, power, pending, associations, blocks, holes, hole_half,
          existing_keepouts, room, spacing, margin, boxes):
    ox, oy = o.origin_x_mm, o.origin_y_mm
    board_centre = (ox + o.width_mm / 2, oy + o.height_mm / 2)
    placed, keepouts = {}, []
    obstacles = list(existing_keepouts)
    def bbox(p):
        return boxes(p)[0]

    def legal(p):
        b = bbox(p)
        if not _inside(grow(b, margin), o) or not room(p, placed):
            return False
        for k in obstacles:
            if p.component_ref not in allowed_refs(k) and (forbids(k, "footprints") or forbids(k, "pads")) and any(covers_layer(k, layer) for layer in ("F.Cu", "B.Cu")):
                if box_area_overlap((b.x1, b.y1, b.x2, b.y2), area_points(k)):
                    return False
        return True

    for c, (sx, sy) in zip(holes, ((0, 0), (1, 0), (0, 1), (1, 1))):
        x = ox + (o.width_mm - hole_half - margin if sx else hole_half + margin)
        y = oy + (o.height_mm - hole_half - margin if sy else hole_half + margin)
        p = at(c.ref, fps[c.ref], x, y)
        if not legal(p):
            raise CompileError(f"corner mounting reservation {c.ref} does not fit")
        placed[c.ref] = p
        k = Keepout(id=f"functional_{c.ref}", layers=["*.Cu"], rect=Traced(value=[x - hole_half, y - hole_half, hole_half * 2, hole_half * 2], unit="mm", provenance=provenance(HOLE_NOTE)),
                    forbids=["tracks", "vias", "pads", "zones", "footprints"], allowed_refs=[c.ref], reason=HOLE_NOTE, provenance=provenance(HOLE_NOTE))
        keepouts.append(k)
        obstacles.append(k)
    # Reserve the core/support cluster while packing the perimeter anchors.
    cluster = {r: p.model_copy(update={"x_mm": _q(p.x_mm + board_centre[0]), "y_mm": _q(p.y_mm + board_centre[1])}) for r, p in local.items()}
    for side in NORMAL:
        refs = [r for r in connectors if sides[r] == side]
        refs.sort(key=lambda r: (sum((fps[core].pad(pin).y if side in ("west", "east") else fps[core].pad(pin).x)
                                    for _, pin in links[r]) / max(1, len(links[r])), natural_ref_key(r)))
        cursor = margin + 2 * hole_half + ACCESS_MM
        for index, ref in enumerate(refs):
            options = []
            vertical = "Vertical" in fps[ref].name or "HDR" in fps[ref].name
            for angle in (0, 90, 180, 270):
                zero = at(ref, fps[ref], 0, 0, angle)
                b = footprint_bbox(zero, fps[ref])
                # Only a side-mating connector needs its long edge along the board edge.
                if not vertical and (b.height < b.width if side in ("west", "east") else b.width < b.height):
                    continue
                direction = fps[ref].properties.get("mating_direction")
                if direction is not None:
                    if direction not in NORMAL:
                        raise CompileError(f"{ref}: library mating_direction must be west/east/north/south")
                    if rotate(*NORMAL[direction], angle) != NORMAL[side]:
                        continue
                along = o.height_mm if side in ("west", "east") else o.width_mm
                length = b.height if side in ("west", "east") else b.width
                tail = sum((min if "Vertical" in fps[r].name or "HDR" in fps[r].name else max)(
                    footprint_extent(fps[r]).width, footprint_extent(fps[r]).height) + max(spacing, ACCESS_MM) for r in refs[index + 1:])
                low, high = cursor + length / 2, along - margin - 2 * hole_half - ACCESS_MM - tail - length / 2
                positions = sorted({low, high, *(step / 2 for step in range(math.ceil(low * 2), math.floor(high * 2) + 1))}) if low <= high else []
                for t in positions:
                    x = ox + (margin + b.width / 2 if side == "west" else o.width_mm - margin - b.width / 2) if side in ("west", "east") else ox + t
                    y = oy + (margin + b.height / 2 if side == "north" else o.height_mm - margin - b.height / 2) if side in ("north", "south") else oy + t
                    p = zero.model_copy(update={"x_mm": _q(zero.x_mm + x), "y_mm": _q(zero.y_mm + y)})
                    extent = bbox(p)
                    if not legal(p) or not room(p, cluster):
                        continue
                    if vertical and any(not _disjoint(grow(extent, ACCESS_MM - 1e-6), bbox(q)) for q in [*placed.values(), *cluster.values()]):
                        continue
                    cost = sum(math.dist(pad_center(p, fps[ref].pad(pin)), pad_center(cluster[core], fps[core].pad(ic))) for pin, ic in links[ref])
                    # Compact each pin-ordered edge run before minimizing its airwires.
                    options.append(((t, cost, angle), p))
            if not options:
                raise CompileError(f"edge connector {ref} cannot fit on its {side} pin side")
            placed[ref] = min(options, key=lambda row: row[0])[1]
            b = footprint_bbox(placed[ref], fps[ref])
            cursor = (b.y2 - oy if side in ("west", "east") else b.x2 - ox) + max(spacing, ACCESS_MM)
            if "Vertical" in fps[ref].name or "HDR" in fps[ref].name:
                b = grow(footprint_bbox(placed[ref], fps[ref]), ACCESS_MM)
                k = Keepout(id=f"access_{ref}", layers=["F.Cu"], rect=Traced(value=[b.x1, b.y1, b.width, b.height], unit="mm", provenance=provenance("CHOICE:vertical access clearance=2 mm")),
                            forbids=["footprints"], allowed_refs=[ref], reason="CHOICE: vertical header assembly access; no horizontal cable direction inferred", provenance=provenance("CHOICE:vertical access clearance=2 mm"))
                obstacles.append(k)
                keepouts.append(k)
    for ref, p in cluster.items():
        if not legal(p):
            raise CompileError(f"central support cluster {ref} does not fit")
        placed[ref] = p
    # Power entry is an edge anchor; regulator and its capacitors follow it.
    remaining = [r for r in fps if r not in placed]
    remaining.sort(key=lambda r: (r not in power, r in pending, blocks[r].id if r in blocks else "", natural_ref_key(r)))
    while remaining:
        ref = next((r for r in remaining if r not in associations or all(a.ic_pin.component_ref in placed for a in associations[r])), None)
        if ref is None:
            raise CompileError(f"support association cycle or missing anchor: {remaining}")
        if ref in associations:
            p = _support(ref, fps, associations[ref], placed, lambda p, _: legal(p))
        else:
            neighbours = [q for r, q in placed.items() if r in power] if ref in power else [q for r, q in placed.items() if r in blocks and ref in blocks and blocks[r].id == blocks[ref].id]
            tx, ty = centre(footprint_bbox(neighbours[0], fps[neighbours[0].component_ref])) if neighbours else board_centre
            candidates = []
            for angle in (0, 90):
                for y in range(math.ceil(oy + margin), math.floor(oy + o.height_mm - margin) + 1):
                    for x in range(math.ceil(ox + margin), math.floor(ox + o.width_mm - margin) + 1):
                        candidates.append(((math.hypot(x - tx, y - ty), angle, x, y), (x, y, angle)))
            p = next((p for _, (x, y, a) in sorted(candidates) if legal(p := at(ref, fps[ref], x, y, a))), None)
            if p is None:
                raise CompileError(f"block packing: {ref} cannot fit")
        placed[ref] = p
        remaining.remove(ref)
    return placed, keepouts
