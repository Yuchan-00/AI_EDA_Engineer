"""Deterministic layout measurements, never electrical verification.

Lengths are millimetres, areas square millimetres, rounded to six decimals.
Schematic overlaps count pairs of axis-aligned bounds with positive shared
area. Symbol bounds include graphics, pins and pin text; visible fields,
labels (including their outline) and sheet text are separate text objects.
Text uses the preview's documented font estimate, not KiCad font metrics.
Crossings are distinct interior/interior intersections without a junction;
bends are degree-two, non-collinear wire vertices (T joints are not bends).
Only the supplied sheet is measured; hierarchical sheets must be measured
separately. PCB bounds are courtyard union pads in the placed board frame.
Airwires are Euclidean MSTs of pad centres, with no obstacles or via cost.
Routed track length is the sum of the tracks' centreline lengths: the
straight distance between a track's ends, or the arc length of an arc
track (``Track.mid``), never the chord between an arc's ends.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from ai_eda.gui import schematic_render as render
from ai_eda.ir import CircuitIR, track_length
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center
from ai_eda.tools.kicad.library import KicadLibrary

METRICS_VERSION = "1"
EPS = 1e-9
Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _point(p: Point) -> Point:
    return round(p[0], 6), round(p[1], 6)


def _overlap(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > EPS and min(a[3], b[3]) - max(a[1], b[1]) > EPS


def _box(out: render._Out) -> Box | None:
    return (min(out.xs), min(out.ys), max(out.xs), max(out.ys)) if out.xs else None


def schematic_metrics(path: str | Path) -> dict:
    """Measure a compiled schematic using its embedded symbols, never installed replacements."""
    root = sexpr.parse(Path(path).read_text(encoding="utf-8"))
    if sexpr.head(root) != "kicad_sch":
        raise ValueError("metrics.schematic: expected kicad_sch")
    lib_node = sexpr.find(root, "lib_symbols")
    lib = {str(s[1]): s for s in sexpr.find_all(lib_node, "symbol")} if lib_node else {}
    symbols, texts, segments = [], [], []
    junctions = {_point(render._xy(sexpr.find(j, "at"), "junction")) for j in sexpr.find_all(root, "junction")}
    for item in root:
        kind = sexpr.head(item)
        out = render._Out()
        if kind == "symbol":
            lib_id = str(sexpr.get(item, "lib_id", 1, ""))
            if lib_id not in lib and str(sexpr.get(item, "lib_name")) not in lib:
                raise ValueError(f"metrics.symbol: embedded symbol missing: {lib_id}")
            body = [n for n in item if sexpr.head(n) != "property"]
            render._draw_symbol(out, body, lib)
            if (box := _box(out)) is not None:
                symbols.append(box)
            at = sexpr.find(item, "at")
            x, y = render._xy(at, "symbol")
            inst = render._Instance(x, y, render._rotation(at[3]) if len(at) > 3 else 0, sexpr.get(item, "mirror"))
            for prop in sexpr.find_all(item, "property"):
                field_out = render._Out()
                render._draw_fields(field_out, ["symbol", prop], inst)
                if (box := _box(field_out)) is not None:
                    texts.append(box)
        elif kind in ("global_label", "label", "text"):
            {"global_label": render._draw_global_label, "label": render._draw_local_label, "text": render._draw_sheet_text}[kind](out, item)
            if (box := _box(out)) is not None:
                texts.append(box)
        elif kind == "wire":
            pts = render._pts(item, "wire")
            if len(pts) < 2:
                raise ValueError("metrics.wire: fewer than two points")
            segments.extend((a, b) for a, b in zip(pts, pts[1:]))
    crossings = set()
    neighbours = defaultdict(set)
    for a, b in segments:
        if a != b:
            neighbours[a].add(b)
            neighbours[b].add(a)
    for (a, b), (c, d) in combinations(segments, 2):
        u, v, offset = _sub(b, a), _sub(d, c), _sub(c, a)
        det = _cross(u, v)
        if abs(det) <= EPS:
            continue
        t, s = _cross(offset, v) / det, _cross(offset, u) / det
        if EPS < t < 1 - EPS and EPS < s < 1 - EPS:
            p = _point((a[0] + t * u[0], a[1] + t * u[1]))
            if p not in junctions:
                crossings.add(p)
    bends = 0
    for p, adjacent in neighbours.items():
        if len(adjacent) != 2 or _point(p) in junctions:
            continue
        a, b = sorted(adjacent)
        if abs(_cross(_sub(a, p), _sub(b, p))) <= EPS:
            continue
        # A segment passing through the vertex makes a junction, not a bend.
        if any(p != a and p != b and abs(_cross(_sub(p, a), _sub(b, a))) <= EPS
               and min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
               and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS for a, b in segments):
            continue
        bends += 1
    ss = sum(_overlap(a, b) for a, b in combinations(symbols, 2))
    st = sum(_overlap(a, b) for a in symbols for b in texts)
    tt = sum(_overlap(a, b) for a, b in combinations(texts, 2))
    return {
        "symbol_symbol_overlaps": ss, "symbol_text_overlaps": st, "text_text_overlaps": tt,
        "overlaps": ss + st + tt,
        "global_label_count": len(sexpr.find_all(root, "global_label")),
        "local_label_count": len(sexpr.find_all(root, "label")),
        "wire_segment_count": len(segments),
        "wire_length_mm": round(math.fsum(sorted(math.dist(a, b) for a, b in segments)), 6),
        "wire_crossings": len(crossings), "bends": bends,
    }


def mst_length(points: list[tuple[float, float]]) -> float:
    """Euclidean Prim tree length; duplicates collapse, equal edges use coordinate order."""
    points = sorted(set(points))
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("metrics.pad: non-finite coordinate")
    if not points:
        return 0.0
    remaining = {p: math.dist(points[0], p) for p in points[1:]}
    edges = []
    while remaining:
        p = min(remaining, key=lambda p: (remaining[p], p))
        edges.append(remaining.pop(p))
        for other in remaining:
            remaining[other] = min(remaining[other], math.dist(p, other))
    return math.fsum(edges)


def pcb_metrics(ir: CircuitIR, library: KicadLibrary) -> dict:
    """Measure placed IR geometry; missing outlines, placements or pads are refused, not zeroed.

    Plane nets are those named by a stackup plane or a copper zone. Their MSTs
    are reported separately as well as in the total, even when no tracks join them.
    """
    pcb = ir.pcb
    if pcb is None or pcb.outline is None:
        raise ValueError("metrics.outline: board outline missing")
    if not all(math.isfinite(v) and v > 0 for v in (pcb.outline.width_mm, pcb.outline.height_mm)):
        raise ValueError("metrics.outline: dimensions must be finite and positive")
    placements = {p.component_ref: p for p in pcb.placements}
    if len(placements) != len(pcb.placements):
        raise ValueError("metrics.placement: duplicate reference")
    pads = defaultdict(list)
    areas = []
    for c in sorted(ir.components, key=lambda c: c.ref):
        if c.ref not in placements or c.footprint is None:
            raise ValueError(f"metrics.placement: {c.ref} has no placement or footprint")
        p = placements[c.ref]
        fp = library.load_footprint(c.footprint)
        box = footprint_bbox(p, fp)
        if box is None:
            raise ValueError(f"metrics.footprint: {c.ref} has no measurable geometry")
        areas.append((box.x2 - box.x1) * (box.y2 - box.y1))
        for pad in fp.pads:
            if pad.number:
                pads[c.ref, pad.number].append(pad_center(p, pad))
    planes = {z.net for z in pcb.zones}
    if pcb.stackup:
        planes.update(c.plane_net.value for c in pcb.stackup.copper if c.plane_net is not None)
    per_net = {}
    for net in sorted(ir.nets, key=lambda n: n.name):
        points = []
        for pin in net.pins:
            key = pin.component_ref, pin.pin_number
            if key not in pads:
                raise ValueError(f"metrics.pad: net {net.name} has no placed pad for {pin}")
            points.extend(pads[key])
        per_net[net.name] = mst_length(points)
    try:
        lengths = [track_length(t) for t in pcb.tracks]
    except ValueError as exc:  # an arc whose three points are no arc (refused by the Track model; assigned afterwards)
        raise ValueError(f"metrics.track: {exc}") from exc
    if not all(math.isfinite(v) for v in lengths):
        raise ValueError("metrics.track: non-finite length")
    return {
        "board_area_mm2": round(pcb.outline.width_mm * pcb.outline.height_mm, 6),
        "footprint_bbox_area_mm2": round(math.fsum(areas), 6),
        "airwire_mst_mm": round(math.fsum(per_net.values()), 6),
        "signal_airwire_mst_mm": round(math.fsum(v for n, v in per_net.items() if n not in planes), 6),
        "plane_airwire_mst_mm": round(math.fsum(v for n, v in per_net.items() if n in planes), 6),
        "airwire_mst_per_net_mm": {n: round(v, 6) for n, v in per_net.items()},
        "plane_nets": sorted(planes),
        "routed_track_length_mm": round(math.fsum(sorted(lengths)), 6),
        "via_count": len(pcb.vias),
        "layer_count": sum(layer.name.endswith(".Cu") for layer in pcb.layers),
    }
