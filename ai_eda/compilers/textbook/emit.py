"""Guards against connections the IR does not state, and the ``(kicad_sch ...)`` tree of a textbook 1.0 sheet.

KiCad connects by exact coordinates, so a wire through a foreign pin or a
foreign wire end (diagonals included) would merge nets silently; so would two
parts' pins of different nets (or an unconnected pin) meeting at one point,
and a label anchor off its own net, or on another. Overlapping boxes are hard
faults too. Wires within 1.27 mm of a symbol that has no pin on their net,
and parallel wires of different nets 1.27 mm apart, are spacing findings:
scored, not refused.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ai_eda.compilers.ids import net_item_uuid, sheet_uuid
from ai_eda.compilers.schematic_layout import Extent, natural_ref_key
from ai_eda.compilers.textbook.glyphs import (
    field_nodes,
    generated_fields,
    generated_pin,
    glyph_boxes,
    label_box,
    label_node,
    mm,
    overlaps,
    pin_cell,
)
from ai_eda.compilers.textbook.model import BoxMM, Hit, Label, Model, Pt, Seg, Sheet, Wiring
from ai_eda.ir import Component, LibraryRef, Pin, Provenance, ProvenanceKind
from ai_eda.tools.kicad.library import fabrication_exclusions
from ai_eda.tools.kicad.sexpr import Node, Q, S

NEAR_MM = 1.27  # a wire closer than this to a symbol with no pin on its net is a spacing finding
PARALLEL_CELLS = 1  # different nets running parallel this many cells apart (1.27 mm) is a spacing finding


def raster(seg: Seg) -> list[Pt]:
    """The grid points on a segment (diagonals included), ends first to last."""
    (ax, ay), (bx, by) = seg
    n = math.gcd(abs(bx - ax), abs(by - ay))
    return [(ax + i * (bx - ax) // n, ay + i * (by - ay) // n) for i in range(n + 1)] if n else [seg[0]]


def all_segments(wiring: Wiring, labels: list[Label]) -> dict[str, list[Seg]]:
    """Every wire of the sheet per net: the wiring plus the labels' stubs, each piece once, in a fixed order."""
    out: dict[str, set[Seg]] = defaultdict(set)
    for net, segs in wiring.segments.items():
        out[net].update(tuple(sorted(s)) for s in segs if s[0] != s[1])  # type: ignore[misc]
    for lab in labels:
        if lab.stub is not None and lab.stub[0] != lab.stub[1]:
            out[lab.net].add(tuple(sorted(lab.stub)))  # type: ignore[arg-type]
    return {net: sorted(segs) for net, segs in sorted(out.items())}


def pin_points(model: Model, sheet: Sheet) -> dict[Pt, list[tuple[str, str | None, str]]]:
    """Every pin point of the sheet: ``[(owner key, net or None, "ref.pin")]`` (unconnected and hidden pins included)."""
    out: dict[Pt, list[tuple[str, str | None, str]]] = defaultdict(list)
    for key, place in sorted(sheet.places().items(), key=lambda kv: natural_ref_key(kv[0])):
        g = model.glyphs[key]
        for p in g.pins:
            out[pin_cell(g, p, (place.x, place.y), place.orient)].append((key, g.net.get(p.number), f"{g.ref}.{p.number}"))
    for gen in sorted(sheet.generated(), key=lambda gen: natural_ref_key(gen.ref)):
        out[generated_pin(gen)].append((gen.ref, gen.net, gen.ref))
    return out


def _seg_box_gap(a: tuple[float, float], b: tuple[float, float], box: BoxMM) -> float:
    """Distance from segment a-b (mm) to an axis-aligned box; 0 when they meet."""
    x0, y0, x1, y1 = box
    # Liang-Barsky: does the segment enter the box?
    t0, t1, dx, dy = 0.0, 1.0, b[0] - a[0], b[1] - a[1]
    inside = True
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                inside = False
                break
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                inside = False
                break
    if inside:
        return 0.0

    def point_box(p: tuple[float, float]) -> float:
        return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))

    def point_seg(p: tuple[float, float]) -> float:
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length))
        return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)

    return min(point_box(a), point_box(b), *(point_seg(c) for c in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))))


def spacing(model: Model, sheet: Sheet, segments: dict[str, list[Seg]], bodies: list[tuple[str, BoxMM]]) -> list[Hit]:
    """Spacing findings: a wire within 1.27 mm of a body that has no pin on its net; different nets parallel 1 cell apart."""
    nets_of: dict[str, set[str]] = {key: set(model.glyphs[key].net.values()) for key in sheet.places()}
    nets_of.update({gen.ref: {gen.net} for gen in sheet.generated()})
    hits: list[Hit] = []
    for net, segs in segments.items():
        for a, b in segs:
            am, bm = (mm(a[0]), mm(a[1])), (mm(b[0]), mm(b[1]))
            for owner, box in bodies:
                if net not in nets_of.get(owner, set()) and _seg_box_gap(am, bm, box) < NEAR_MM - 1e-9:
                    hits.append(Hit(f"spacing: wire of {net} {a}-{b} within {NEAR_MM} mm of {owner}", (owner,), hard=False))
    rows: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
    for net, segs in segments.items():
        for a, b in segs:
            if a[1] == b[1]:
                rows[("h", a[1])].append((min(a[0], b[0]), max(a[0], b[0]), net))
            elif a[0] == b[0]:
                rows[("v", a[0])].append((min(a[1], b[1]), max(a[1], b[1]), net))
    for (axis, level), runs in sorted(rows.items()):
        for lo, hi, net in runs:
            for lo2, hi2, net2 in rows.get((axis, level + PARALLEL_CELLS), []):
                if net2 != net and min(hi, hi2) - max(lo, lo2) > 0:
                    hits.append(Hit(f"spacing: {net} and {net2} run parallel 1.27 mm apart ({axis} {level})", (), hard=False))
    return hits


def guard(ctx, sheet: Sheet, wiring: Wiring, labels: list[Label], boxes: list[tuple[str, BoxMM]] | None = None) -> list[Hit]:
    """Every finding on the sheet as drawn (``labels`` = all of its labels). Hard: a wire on a foreign pin or a foreign wire
    end, pins of two parts that meet at one point with different nets (or one unconnected), a label off its net or on
    another, overlapping boxes. Soft: :func:`spacing`."""
    model: Model = ctx.model
    segments = all_segments(wiring, labels)
    pins = pin_points(model, sheet)
    ends: dict[Pt, set[str]] = defaultdict(set)
    points: dict[str, set[Pt]] = {}
    for net, segs in segments.items():
        points[net] = {p for s in segs for p in raster(s)}
        for s in segs:
            ends[s[0]].add(net)
            ends[s[1]].add(net)
    hits: list[Hit] = []
    for net, segs in segments.items():
        for s in segs:
            for p in raster(s):
                for owner, pnet, what in pins.get(p, ()):
                    if pnet != net:
                        hits.append(Hit(f"wire of {net} {s[0]}-{s[1]} touches pin {what} ({pnet or 'no net'}) at {p}", (owner,)))
                for other in sorted(ends.get(p, ())):
                    if other != net:
                        hits.append(Hit(f"wire of {net} {s[0]}-{s[1]} touches a wire end of {other} at {p}"))
    for p, items in sorted(pins.items()):
        owners = tuple(sorted({owner for owner, _, _ in items}, key=natural_ref_key))
        nets = {pnet for _, pnet, _ in items}
        if len(owners) > 1 and (len(nets) > 1 or None in nets):  # one part's stacked pins are checked by load_model
            met = ", ".join(f"{what} ({pnet or 'no net'})" for _, pnet, what in items)
            hits.append(Hit(f"pins {met} meet at {p}", owners))
    for lab in labels:
        own = lab.at in points.get(lab.net, ()) or any(pnet == lab.net for _, pnet, _ in pins.get(lab.at, ()))
        foreign = [n for n, pts in points.items() if n != lab.net and lab.at in pts] + [w for _, pnet, w in pins.get(lab.at, ()) if pnet != lab.net]
        if not own:
            hits.append(Hit(f"label {lab.net} at {lab.at} is on no wire or pin of its net"))
        if foreign:
            hits.append(Hit(f"label {lab.net} at {lab.at} touches {', '.join(foreign)}"))
    if boxes is None:
        boxes = glyph_boxes(model, sheet)
    known = {tag for tag, _ in boxes}
    tagged = boxes + [(tag, label_box(lab)) for lab in labels if (tag := f"label:{lab.net}@{lab.at[0]},{lab.at[1]}") not in known]
    for i, j in overlaps([b for _, b in tagged]):
        hits.append(Hit(f"overlap: {tagged[i][0]} and {tagged[j][0]}", (_owner(tagged[i][0]), _owner(tagged[j][0]))))
    bodies = [(_owner(tag), box) for tag, box in boxes if tag.startswith(("body:", "glyph:"))]
    return hits + spacing(model, sheet, segments, bodies)


def _owner(tag: str) -> str:
    return tag.split(":", 1)[1] if not tag.startswith("label:") else tag


def junctions(segments: dict[str, list[Seg]], pins: dict[Pt, list[tuple[str, str | None, str]]]) -> dict[str, list[Pt]]:
    """Junction dots per net: where three or more wire ends meet or a wire end lands on another wire's interior (degree
    3), or where two or more wires meet at a pin point (a T at a pin)."""
    out: dict[str, list[Pt]] = {}
    for net, segs in segments.items():
        degree: dict[Pt, int] = defaultdict(int)
        for s in segs:
            pts = raster(s)
            degree[pts[0]] += 1
            degree[pts[-1]] += 1
            for p in pts[1:-1]:
                degree[p] += 2
        ends = {p for s in segs for p in s}
        own_pins = {p for p, items in pins.items() if any(n == net for _, n, _ in items)}
        dots = [p for p, d in degree.items() if (p in ends and d >= 3) or (p in own_pins and d >= 2)]
        if dots:
            out[net] = sorted(dots)
    return out


def emit(ctx, sheet: Sheet, wiring: Wiring, labels: list[Label], paper: tuple[str, float, float], comment: str) -> Node:
    """The ``(kicad_sch ...)`` tree: symbols from :mod:`ai_eda.compilers.ids` UUIDs, wires / junctions / labels from
    :func:`~ai_eda.compilers.ids.net_item_uuid`, every list in a fixed order."""
    from ai_eda.compilers.schematic import GENERATOR, GENERATOR_VERSION, SCH_FORMAT_VERSION, _junction, _PlacedSymbol, _wire

    model: Model = ctx.model
    compiler = ctx.compiler
    ir = model.ir
    pid = model.project_id
    root = sheet_uuid(pid)
    places = sheet.places()
    anchors = sheet.fields()
    placed: dict[str, _PlacedSymbol] = {}
    symbols: list[Node] = []
    for key in sorted(places, key=natural_ref_key):
        g, place = model.glyphs[key], places[key]
        ps = _PlacedSymbol(g.component, g.symbol, g.footprint_id, mm(place.x), mm(place.y), Extent(0, 0, 0, 0), place.orient.rot, place.orient.mirror,
                           g.unit)
        placed[key] = ps  # one instance per glyph: a multi-unit part writes every unit
        symbols.append(_with_fields(compiler._symbol_node(pid, root, ps, fabrication_exclusions(ctx.library, g.component.footprint)),
                                    field_nodes(g, place, anchors.get(key))))
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool=compiler.id, tool_version="1.0")
    for gen in sorted(sheet.generated(), key=lambda gen: natural_ref_key(gen.ref)):
        lib, name = gen.lib.split(":", 1)
        c = Component(ref=gen.ref, value=gen.value, symbol=LibraryRef(library=lib, name=name, verified=True), provenance=prov,
                      pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in gen.symbol.pins])
        ps = _PlacedSymbol(c, gen.symbol, "", mm(gen.at[0]), mm(gen.at[1]), Extent(0, 0, 0, 0), gen.orient.rot, gen.orient.mirror)
        symbols.append(_with_fields(compiler._symbol_node(pid, root, ps), generated_fields(gen)))
    pins_by_key = compiler._pin_map(placed)
    no_connects, _, _ = compiler._connectivity(ir, placed, pins_by_key, model.stacked)
    segments = all_segments(wiring, labels)
    wires = [_wire((mm(a[0]), mm(a[1])), (mm(b[0]), mm(b[1])), net_item_uuid(pid, "textbook10-wire", net, *a, *b))
             for net, segs in segments.items() for a, b in segs]
    dots = junctions(segments, pin_points(model, sheet))
    junction_nodes = [_junction((mm(p[0]), mm(p[1])), net_item_uuid(pid, "textbook10-junction", net, *p)) for net, pts in dots.items() for p in pts]
    label_nodes = [label_node(lab, net_item_uuid(pid, "textbook10-label", lab.net, *lab.at))
                   for lab in sorted(labels, key=lambda lab: (lab.net, lab.at, lab.angle))]
    lib = {g.symbol.lib_id: g.symbol for g in model.glyphs.values() if g.key in places}
    lib.update({gen.symbol.lib_id: gen.symbol for gen in sheet.generated()})
    name, w, h = paper
    return S(
        "kicad_sch",
        S("version", SCH_FORMAT_VERSION),
        S("generator", Q(GENERATOR)),
        S("generator_version", Q(GENERATOR_VERSION)),
        S("uuid", Q(root)),
        S("paper", Q(name)) if name != "User" else S("paper", Q(name), w, h),
        S("title_block", S("title", Q(ir.project.name)), S("comment", 1, Q(comment)),
          S("comment", 2, Q(ir.project.description)) if ir.project.description else None),
        S("lib_symbols", *(lib[k].lib_symbols_entry() for k in sorted(lib))),
        *no_connects,
        *wires,
        *junction_nodes,
        *label_nodes,
        *symbols,
        S("sheet_instances", S("path", Q("/"), S("page", Q("1")))),
        S("embedded_fonts", False),
    )


def _with_fields(node: Node, fields: tuple[Node, Node]) -> Node:
    """``node`` with its Reference and Value properties replaced by ``fields`` (same place in the list)."""
    by_key = {str(f[1]): f for f in fields}
    return [by_key.get(str(item[1]), item) if isinstance(item, list) and item and item[0] == "property" and str(item[1]) in by_key else item for item in node]
