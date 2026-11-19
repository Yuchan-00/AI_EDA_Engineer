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

Schematic gates (:func:`schematic_gates`, metrics version 2) read nets from
the sheet itself: a wire end on a wire, a pin or a label anchor on a wire,
pins at one point, labels of one name and power-input pins of power symbols
(named by their Value) join. A pin is *reached by a label* when the wires
and power symbols alone do not join it to every other pin of its net.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from ai_eda.gui import schematic_render as render
from ai_eda.ir import CircuitIR, track_length
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center
from ai_eda.tools.kicad.library import KicadLibrary

METRICS_VERSION = "2"
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


def rectilinear_mst_length(points: list[tuple[float, float]]) -> float:
    """Manhattan Prim tree length; duplicates collapse, equal edges use coordinate order."""
    points = sorted(set(points))
    if len(points) < 2:
        return 0.0
    remaining = {p: abs(points[0][0] - p[0]) + abs(points[0][1] - p[1]) for p in points[1:]}
    edges = []
    while remaining:
        p = min(remaining, key=lambda q: (remaining[q], q))
        edges.append(remaining.pop(p))
        for other in remaining:
            remaining[other] = min(remaining[other], abs(p[0] - other[0]) + abs(p[1] - other[1]))
    return math.fsum(edges)


def _on(p: Point, a: Point, b: Point) -> bool:
    return (abs(_cross(_sub(b, a), _sub(p, a))) <= 1e-6 and min(a[0], b[0]) - 1e-6 <= p[0] <= max(a[0], b[0]) + 1e-6
            and min(a[1], b[1]) - 1e-6 <= p[1] <= max(a[1], b[1]) + 1e-6)


def _segment_box_distance(a: Point, b: Point, box: Box) -> float:
    """Distance from segment a-b to an axis-aligned box (0 when they meet): the minimum over the segment's ends against the
    box and the box corners against the segment, unless the segment enters the box."""
    x0, y0, x1, y1 = box
    lo, hi = 0.0, 1.0
    for p, q in ((a[0] - b[0], a[0] - x0), (b[0] - a[0], x1 - a[0]), (a[1] - b[1], a[1] - y0), (b[1] - a[1], y1 - a[1])):
        if abs(p) <= EPS:
            if q < -EPS:
                lo, hi = 1.0, 0.0
                break
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
    if lo <= hi + EPS:
        return 0.0
    d = _sub(b, a)
    length = d[0] * d[0] + d[1] * d[1]

    def to_segment(p: Point) -> float:
        t = 0.0 if length <= EPS else max(0.0, min(1.0, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / length))
        return math.dist(p, (a[0] + t * d[0], a[1] + t * d[1]))

    def to_box(p: Point) -> float:
        return math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1))

    return min(to_box(a), to_box(b), *(to_segment(c) for c in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))))


class _Union:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


#: net names a sheet without an IR treats as ground
_GROUND_NAMES = frozenset({"GND", "GNDA", "GNDD", "AGND", "DGND", "GNDPWR", "0"})
NEAR_SYMBOL_MM = 1.27
PARALLEL_MM = 2.54


def schematic_gates(path: str | Path, ir: CircuitIR | None = None) -> dict:
    """The textbook layout gates of a compiled sheet, read from the file alone; ``ir`` adds net kinds
    and declared blocks. Lengths mm, rounded to six decimals; a percentage to one."""
    root = sexpr.parse(Path(path).read_text(encoding="utf-8"))
    if sexpr.head(root) != "kicad_sch":
        raise ValueError("metrics.schematic: expected kicad_sch")
    lib_node = sexpr.find(root, "lib_symbols")
    lib = {str(s[1]): s for s in sexpr.find_all(lib_node, "symbol")} if lib_node else {}
    symbols: list[dict] = []
    texts: list[Box] = []
    for item in root:
        if sexpr.head(item) != "symbol":
            continue
        lib_id = str(sexpr.get(item, "lib_id", 1, ""))
        if lib_id not in lib:
            raise ValueError(f"metrics.symbol: embedded symbol missing: {lib_id}")
        at = sexpr.find(item, "at")
        x, y = render._xy(at, "symbol")
        mirror = sexpr.get(item, "mirror")
        inst = render._Instance(x, y, render._rotation(at[3]) if len(at) > 3 else 0, str(mirror) if mirror is not None else None)
        unit = int(render._num(sexpr.get(item, "unit", 1, "1"), "symbol unit"))
        props = {str(p[1]): str(p[2]) for p in sexpr.find_all(item, "property") if len(p) > 2}
        out = render._Out()
        render._draw_symbol(out, [n for n in item if sexpr.head(n) != "property"], lib)
        power = sexpr.find(lib[lib_id], "power") is not None
        pins = []
        for unit_item in render._unit_items(lib[lib_id], unit):
            if sexpr.head(unit_item) == "pin":
                pin, _, _ = render._pin_def(unit_item, unit)
                pins.append((pin.number, _point(inst.point(pin.x, pin.y)), pin.electrical_type))
        for prop in sexpr.find_all(item, "property"):
            field_out = render._Out()
            render._draw_fields(field_out, ["symbol", prop], inst)
            if (box := _box(field_out)) is not None:
                texts.append(box)
        symbols.append({"ref": props.get("Reference", ""), "value": props.get("Value", ""), "lib_id": lib_id, "power": power,
                        "box": _box(out) or (x, y, x, y), "pins": pins})
    segments = [(_point(a), _point(b)) for w in sexpr.find_all(root, "wire") for a, b in zip(render._pts(w, "wire"), render._pts(w, "wire")[1:]) if a != b]
    labels = []
    for kind in ("global_label", "label"):
        for item in sexpr.find_all(root, kind):
            pos, _ = render._label_at(item, kind)
            labels.append((str(item[1]), _point(pos)))
            out = render._Out()
            (render._draw_global_label if kind == "global_label" else render._draw_local_label)(out, item)
            if (box := _box(out)) is not None:
                texts.append(box)
    # connectivity: wires, pins and labels by position; labels and power-input pins of power symbols by name
    full, wired = _Union(), _Union()
    points: dict[Point, list] = defaultdict(list)
    for i, sym in enumerate(symbols):
        for number, p, etype in sym["pins"]:
            points[p].append(("p", i, number))
            if sym["power"] and etype == "power_in":
                for u in (full, wired):
                    u.join(("p", i, number), ("n", sym["value"]))
    for j, (name, p) in enumerate(labels):
        points[p].append(("l", j))
        full.join(("l", j), ("n", name))
    for k, (a, b) in enumerate(segments):
        points[a].append(("s", k))
        points[b].append(("s", k))
    by_x, by_y = defaultdict(list), defaultdict(list)
    for p in points:
        by_x[p[0]].append(p)
        by_y[p[1]].append(p)
    for k, (a, b) in enumerate(segments):
        near = by_x[a[0]] if a[0] == b[0] else by_y[a[1]] if a[1] == b[1] else list(points)
        for p in near:
            if _on(p, a, b):
                for key in points[p]:
                    for u in (full, wired):
                        u.join(("s", k), key)
    for p, keys in points.items():
        for key in keys[1:]:
            for u in (full, wired):
                u.join(keys[0], key)
    # net names and kinds
    owner = {(p.component_ref, p.pin_number): n for n in (ir.nets if ir else []) for p in n.pins}
    group_name: dict = {}
    for i, sym in enumerate(symbols):
        for number, _, _ in sym["pins"]:
            net = owner.get((sym["ref"], number))
            if net is not None:
                group_name.setdefault(full.find(("p", i, number)), net.name)
    names_in: dict = defaultdict(set)
    for key in list(full.parent):
        if key[0] == "n":
            names_in[full.find(key)].add(key[1])
    kinds = {n.name: str(n.kind.value if hasattr(n.kind, "value") else n.kind) for n in (ir.nets if ir else [])}

    memo: dict = {}

    def kind(group) -> str:
        if group not in memo:
            memo[group] = _kind(group)
        return memo[group]

    def _kind(group) -> str:
        if ir is not None:
            return kinds.get(group_name.get(group, ""), "signal")
        names = {n.upper() for n in names_in.get(group, ())}
        if names & _GROUND_NAMES:
            return "ground"
        powered = any(sym["power"] and full.find(("p", i, n)) == group for i, sym in enumerate(symbols) for n, _, t in sym["pins"] if t == "power_in")
        return "power" if powered else "signal"

    physical = [i for i, sym in enumerate(symbols) if not sym["ref"].startswith("#")]
    net_pins: dict = defaultdict(list)
    for i in physical:
        for number, p, _ in symbols[i]["pins"]:
            net_pins[full.find(("p", i, number))].append((i, number, p))
    seg_group = [full.find(("s", k)) for k in range(len(segments))]
    groups_of_symbol = [{full.find(("p", i, n)) for n, _, _ in sym["pins"]} for i, sym in enumerate(symbols)]
    near = sum(1 for k, (a, b) in enumerate(segments) for i, sym in enumerate(symbols)
               if seg_group[k] not in groups_of_symbol[i] and _segment_box_distance(a, b, sym["box"]) < NEAR_SYMBOL_MM - 1e-6)
    parallel = 0
    for k, l in combinations(range(len(segments)), 2):
        if seg_group[k] == seg_group[l]:
            continue
        (a, b), (c, d) = segments[k], segments[l]
        u, length = _sub(b, a), math.dist(a, b)
        if abs(_cross(u, _sub(d, c))) > 1e-6 or abs(_cross(u, _sub(c, a))) / length >= PARALLEL_MM - 1e-6:
            continue
        t = sorted(((c[0] - a[0]) * u[0] + (c[1] - a[1]) * u[1]) / length for c in (c, d))
        if min(length, t[1]) - max(0.0, t[0]) > 1e-6:
            parallel += 1
    labels_per_net: dict[str, int] = defaultdict(int)
    for name, _ in labels:
        labels_per_net[name] += 1
    reached = counted = joined_by_labels = 0
    for group, pins in net_pins.items():
        if len(pins) < 2:
            continue
        parts = {wired.find(("p", i, n)) for i, n, _ in pins}
        joined_by_labels += len(parts) > 1
        for i, n, _ in pins:
            counted += 1
            reached += sum(wired.find(("p", j, m)) == wired.find(("p", i, n)) for j, m, _ in pins) < len(pins)
    ground = {g for g in set(seg_group) | set(net_pins) if kind(g) == "ground"}
    wire = math.fsum(sorted(math.dist(a, b) for k, (a, b) in enumerate(segments) if seg_group[k] not in ground))
    rmst = math.fsum(sorted(rectilinear_mst_length([p for _, _, p in pins]) for g, pins in net_pins.items() if g not in ground))
    centres = {i: ((symbols[i]["box"][0] + symbols[i]["box"][2]) / 2, (symbols[i]["box"][1] + symbols[i]["box"][3]) / 2) for i in physical}
    pairs = set()
    for g, pins in net_pins.items():
        if kind(g) in ("power", "ground"):
            continue
        refs = sorted({i for i, _, _ in pins})
        pairs.update(combinations(refs, 2))
    distances = sorted(math.dist(centres[i], centres[j]) for i, j in pairs)
    negative = {g for g in net_pins if kind(g) == "power" and any(n.startswith("-") or "VEE" in n.upper() or "NEG" in n.upper()
                                                                     for n in {group_name.get(g, "")} | names_in.get(g, set()))}
    checked = ok = 0
    for i in physical:
        ys = defaultdict(list)
        for number, p, _ in symbols[i]["pins"]:
            g = full.find(("p", i, number))
            ys["supply" if kind(g) == "power" and g not in negative else kind(g)].append(p[1])
        if ys["supply"] and ys["ground"]:
            checked += 1
            ok += max(ys["supply"]) < min(ys["ground"])
    flow = None
    if ir is not None and ir.topology is not None and ir.topology.blocks:
        xs = {(symbols[i]["ref"], n): p[0] for i in physical for n, p, _ in symbols[i]["pins"]}
        flow = {"ok": 0, "of": 0}
        for block in ir.topology.blocks:
            members = set(block.component_refs)
            xi = [x for (ref, n), x in xs.items() if ref in members and owner.get((ref, n)) is not None and owner[(ref, n)].name in block.input_nets]
            xo = [x for (ref, n), x in xs.items() if ref in members and owner.get((ref, n)) is not None and owner[(ref, n)].name in block.output_nets]
            if xi and xo:
                flow["of"] += 1
                flow["ok"] += sum(xi) / len(xi) < sum(xo) / len(xo)
    content = [s["box"] for s in symbols] + texts + [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])) for a, b in segments]
    paper = sexpr.find(root, "paper")
    return {
        "near_symbol_hits": near,
        "parallel_hits": parallel,
        "crossings": schematic_metrics(path)["wire_crossings"],
        "labels_per_net": dict(sorted(labels_per_net.items())),
        "label_joined_nets": joined_by_labels,
        "pins_reached_by_label_pct": round(100.0 * reached / counted, 1) if counted else 0.0,
        "wire_mm": round(wire, 6),
        "wire_rmst_ratio": round(wire / rmst, 6) if rmst > EPS else None,
        "median_connected_mm": round(statistics.median(distances), 6) if distances else None,
        "supply_above_ground": {"ok": ok, "of": checked},
        "blocks_left_to_right": flow,
        "paper": str(paper[1]) if paper is not None and len(paper) > 1 else None,
        "content_box": tuple(round(v, 6) for v in (min(b[0] for b in content), min(b[1] for b in content), max(b[2] for b in content), max(b[3] for b in content))) if content else None,
    }


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
