"""Pad-to-pad paths through a net's routed copper: the longest real line the critical-length rule and ``spice.si`` measure.

Invariant: a path length is read from the IR copper (``ir.pcb.tracks`` /
``ir.pcb.vias``) and the pad geometry of the KiCad footprints on disk (the
same pad boxes :mod:`ai_eda.validation.layout` reads), never estimated. A
net's *total* copper (:attr:`~ai_eda.tools.si.measure.NetMeasure.length_mm`)
is only an upper bound of any line through a tree: summed branches are not a
line, so a rule that finds a net *long* must use a line that exists. What
this module cannot extract it says, and the caller uses the total only in the
direction a bound is valid (short when even the total is short).

The graph of one net (board frame, mm, coordinates rounded to KiCad's 1e-6 mm):

* **nodes** - every track end, split where another track end or a via
  centre of the net lies on a track's interior on the same layer (a
  T-junction); every via is two nodes, its ``F.Cu`` and ``B.Cu`` ends, joined
  by its barrel (``Stackup.span_mm("F.Cu", "B.Cu")``, the length the router's
  budget and :mod:`ai_eda.tools.si.measure` count per via; no stackup: 0 mm
  and no delay), a track end within the via's radius on those layers joining
  that end;
* **pads** - every logical pad of the net (one per ``(ref, pin)``; pads that
  repeat a number are one pad) *contracts* the nodes inside its copper box on
  a layer it is on: copper inside a pad is the pad, its own length (the
  router's stub to the pad centre, a through-hole pad's barrel) is not
  counted;
* **edges** - the track pieces (length, layer, width) and the via barrels.

The path is extracted only when every pad of the net is in one connected
part of that graph and the part is a tree (``edges == nodes - 1``): then the
pad-to-pad path is unique, and the longest one (by length; ties by the pad
labels) is the net's *line* - its length, its delay (the sum of each piece's
``t_pd`` from :func:`~ai_eda.tools.si.measure.line_model`, a via's the
barrel bound), whether any piece's delay is the no-plane upper bound, and the
width with the most copper along it. Otherwise the reason is returned: a pad
the copper does not reach, a loop (the path is not unique), a track on an
inner layer (the via barrel between layers is not split here), a pad whose
footprint is not on disk or whose shape is not bounded by its ``(size)`` box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Stackup, Track, Via
from ai_eda.tools.kicad.geometry import _q, pad_angle, pad_copper_center, pad_layers
from ai_eda.tools.kicad.library import KicadLibrary, LibraryLookupError

#: coordinates are rounded to KiCad's 1 nm; a point this close to a segment or a box edge touches it
PATH_TOL_MM = 1e-6
#: pad shapes whose copper lies inside their ``(size)`` box (the only ones this module models)
CONVEX_PAD_SHAPES = frozenset({"circle", "rect", "oval", "roundrect"})
#: the only layers a via end is joined on (the compiler writes through vias; the router routes these two)
OUTER_LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")


@dataclass(frozen=True)
class PadBox:
    """One copper pad of a logical pad (``label`` ``"U1.20"``): the box around its copper centre, the copper layers it is on."""

    label: str
    cx: float
    cy: float
    hw: float
    hh: float
    layers: frozenset[str]

    def contains(self, x: float, y: float) -> bool:
        return abs(x - self.cx) <= self.hw + PATH_TOL_MM and abs(y - self.cy) <= self.hh + PATH_TOL_MM


@dataclass
class NetPads:
    """Every net's logical pads (label -> its copper boxes) and, per net, why its pads are not all known."""

    pads: dict[str, dict[str, list[PadBox]]] = field(default_factory=dict)
    problems: dict[str, list[str]] = field(default_factory=dict)


def net_pads(ir: CircuitIR, library: KicadLibrary) -> NetPads:
    """The pad boxes of every net's pins (the placements and the footprints on disk); a net whose pad cannot be read gets a problem."""
    out = NetPads()
    pcb = ir.pcb
    copper = [layer.name for layer in pcb.layers] if pcb is not None else list(OUTER_LAYERS)
    pin_net: dict[tuple[str, str], str] = {}
    for net in ir.nets:
        out.pads[net.name] = {}
        for pin in net.pins:
            pin_net[(pin.component_ref, pin.pin_number)] = net.name
    comp_problem: dict[str, str] = {}
    for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        placement = pcb.placement(comp.ref) if pcb is not None else None
        if placement is None:
            comp_problem[comp.ref] = f"{comp.ref} has no placement"
            continue
        if comp.footprint is None:
            comp_problem[comp.ref] = f"{comp.ref} has no footprint"
            continue
        try:
            fp = library.load_footprint(comp.footprint)
        except LibraryLookupError:
            comp_problem[comp.ref] = f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref} was not found in a KiCad library"
            continue
        except CompileError as e:  # a malformed .kicad_mod is a reason, not a crash
            comp_problem[comp.ref] = f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref}: {e}"
            continue
        for pad in fp.pads:
            net = pin_net.get((comp.ref, pad.number)) if pad.number else None
            if net is None or pad.pad_type == "np_thru_hole":
                continue
            label = f"{comp.ref}.{pad.number}"
            if pad.shape not in CONVEX_PAD_SHAPES:
                out.problems.setdefault(net, []).append(f"pad {label} has shape {pad.shape!r}, whose copper is not bounded by its (size) box")
                continue
            cx, cy = pad_copper_center(placement, pad)
            angle = pad_angle(placement, pad)
            if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):
                w, h = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
                hw, hh = w / 2.0, h / 2.0
            else:
                hw = hh = math.hypot(pad.size_w, pad.size_h) / 2.0
            layers = pad_layers(placement, pad)
            if pad.pad_type == "thru_hole" or "*.Cu" in layers:
                on = frozenset(copper)
            else:
                on = frozenset(name for name in layers if name.endswith(".Cu"))
            out.pads[net].setdefault(label, []).append(PadBox(label, cx, cy, hw, hh, on))
    for net in ir.nets:
        for pin in net.pins:
            label = f"{pin.component_ref}.{pin.pin_number}"
            if pin.component_ref in comp_problem:
                out.problems.setdefault(net.name, []).append(comp_problem[pin.component_ref])
            elif label not in out.pads[net.name] and not any(p.startswith(f"pad {label} ") for p in out.problems.get(net.name, [])):
                out.problems.setdefault(net.name, []).append(f"pin {label} has no copper pad in its footprint")
    return out


@dataclass(frozen=True)
class NetPath:
    """The longest pad-to-pad path of a net (module docstring). ``delay_s`` is ``None`` when a piece has no line model."""

    ends: tuple[str, str]
    length_mm: float
    track_length_mm: float
    vias: int
    delay_s: float | None
    bound: bool
    #: ``(layer, width_mm)`` with the most track length along the path
    main: tuple[str, float] | None

    @property
    def label(self) -> str:
        return f"{self.ends[0]}-{self.ends[1]}"


def _pt_seg_t(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """``(distance, parameter)`` of point ``p`` to segment ``a``-``b`` (parameter in [0, 1] along it)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    ll = dx * dx + dy * dy
    if ll <= 0.0:
        return math.hypot(px - a[0], py - a[1]), 0.0
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / ll))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy)), t


def longest_path(
    net: str, tracks: list[Track], vias: list[Via], pads: dict[str, list[PadBox]], stackup: Stackup | None,
    t_pd: dict[tuple[str, float], tuple[float | None, bool]], via_t_pd: float | None,
) -> tuple[NetPath | None, str | None]:
    """The longest pad-to-pad path of ``net`` (module docstring), or ``(None, why not)``.

    ``t_pd`` maps ``(layer, width)`` to ``(delay per metre or None, bound)``
    (the caller's line models); ``via_t_pd`` is the barrel's delay per metre.
    """
    if len(pads) < 2:
        return None, "fewer than two pads"
    inner = sorted({t.layer for t in tracks if t.layer not in OUTER_LAYERS})
    if inner:
        return None, f"track(s) on inner layer(s) {inner}: the via barrel between layers is not split in this model"
    barrel = stackup.span_mm("F.Cu", "B.Cu") if stackup is not None else 0.0

    def key(layer: str, pt: tuple[float, float]) -> tuple[str, float, float]:
        return (layer, _q(pt[0]), _q(pt[1]))

    # split points per layer: every track end and via centre of the net
    points: dict[str, set[tuple[float, float]]] = {layer: set() for layer in OUTER_LAYERS}
    for t in tracks:
        points[t.layer].add((_q(t.start[0]), _q(t.start[1])))
        points[t.layer].add((_q(t.end[0]), _q(t.end[1])))
    for v in vias:
        for layer in OUTER_LAYERS:
            points[layer].add((_q(v.x_mm), _q(v.y_mm)))
    # edges: (node a, node b, length, (layer, width) or None for a barrel)
    edges: list[tuple[tuple, tuple, float, tuple[str, float] | None]] = []
    for t in tracks:
        a, b = (_q(t.start[0]), _q(t.start[1])), (_q(t.end[0]), _q(t.end[1]))
        if a == b:
            continue
        cuts = sorted({tt for pt in points[t.layer] if pt not in (a, b) for d, tt in [_pt_seg_t(pt[0], pt[1], a, b)] if d <= PATH_TOL_MM and 0.0 < tt < 1.0}
                      | {0.0, 1.0})
        stops = [(_q(a[0] + s * (b[0] - a[0])), _q(a[1] + s * (b[1] - a[1]))) for s in cuts]
        stops[0], stops[-1] = a, b
        for p0, p1 in zip(stops, stops[1:]):
            if p0 != p1:
                edges.append((key(t.layer, p0), key(t.layer, p1), math.hypot(p1[0] - p0[0], p1[1] - p0[1]), (t.layer, float(t.width_mm))))
    nodes: set[tuple] = set()
    for a, b, _, _ in edges:
        nodes.update((a, b))
    for i, v in enumerate(vias):
        top, bottom = ("via", i, "F.Cu"), ("via", i, "B.Cu")
        edges.append((top, bottom, barrel, None))
        nodes.update((top, bottom))
    # union-find: a via end joins the track ends within its radius; a pad contracts the nodes inside its box
    parent: dict[tuple, tuple] = {n: n for n in nodes}

    def find(n: tuple) -> tuple:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: tuple, b: tuple) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=repr)] = min(ra, rb, key=repr)

    track_nodes = [n for n in nodes if n[0] in OUTER_LAYERS]
    for i, v in enumerate(vias):
        r = float(v.diameter_mm) / 2.0 + PATH_TOL_MM
        for n in track_nodes:
            if math.hypot(n[1] - v.x_mm, n[2] - v.y_mm) <= r:
                union(("via", i, n[0]), n)
    pad_node: dict[str, tuple] = {}
    for label in sorted(pads, key=natural_ref_key):
        me = ("pad", label)
        parent[me] = me
        pad_node[label] = me
        for box in pads[label]:
            for n in track_nodes:
                if n[0] in box.layers and box.contains(n[1], n[2]):
                    union(me, n)
            for i, v in enumerate(vias):
                for layer in OUTER_LAYERS:
                    if layer in box.layers and box.contains(v.x_mm, v.y_mm):
                        union(me, ("via", i, layer))
    # the contracted multigraph
    adj: dict[tuple, list[tuple[tuple, float, tuple[str, float] | None]]] = {}
    for a, b, length, what in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue  # copper inside one pad (or a via whose both ends one pad holds)
        adj.setdefault(ra, []).append((rb, length, what))
        adj.setdefault(rb, []).append((ra, length, what))
    reps = {label: find(n) for label, n in pad_node.items()}
    labels = sorted(reps, key=natural_ref_key)
    start = reps[labels[0]]
    seen = {start}
    stack = [start]
    while stack:
        n = stack.pop()
        for m, _, _ in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    left = sorted(label for label, r in reps.items() if r not in seen)
    if left:
        return None, f"the copper does not join pad(s) {', '.join(left)} to the others"
    n_edges = sum(len(adj.get(n, [])) for n in seen) // 2
    if n_edges != len(seen) - 1:
        return None, f"the copper forms a loop ({n_edges} pieces on {len(seen)} nodes): the pad-to-pad path is not unique"
    # the tree: from every pad, the distance to every node; the longest pad-to-pad pair (ties: the pad labels)
    best: tuple[float, str, str] | None = None
    parents: dict[str, dict[tuple, tuple[tuple, float, tuple[str, float] | None] | None]] = {}
    for i, a in enumerate(labels):
        dist = {reps[a]: 0.0}
        par: dict[tuple, tuple[tuple, float, tuple[str, float] | None] | None] = {reps[a]: None}
        order = [reps[a]]
        while order:
            n = order.pop()
            for m, length, what in adj.get(n, []):
                if m not in dist:
                    dist[m] = dist[n] + length
                    par[m] = (n, length, what)
                    order.append(m)
        parents[a] = par
        for b in labels[i + 1:]:
            if reps[b] == reps[a]:
                continue
            d = dist[reps[b]]
            if best is None or d > best[0] + PATH_TOL_MM:  # ties keep the first pair in natural pad order
                best = (d, a, b)
    if best is None:
        return None, "every pad of the net is joined inside one pad's copper: no line between pads"
    _, a, b = best
    par = parents[a]
    n = reps[b]
    tracks_len = 0.0
    n_vias = 0
    delay: float | None = 0.0
    bound = False
    per_width: dict[tuple[str, float], float] = {}
    while par[n] is not None:
        prev, length, what = par[n]  # type: ignore[misc]
        if what is None:
            n_vias += 1
            if via_t_pd is None or stackup is None:
                delay = None
            elif delay is not None:
                delay += length / 1000.0 * via_t_pd
        else:
            tracks_len += length
            per_width[what] = per_width.get(what, 0.0) + length
            tp, is_bound = t_pd.get(what, (None, False))
            bound = bound or is_bound
            if tp is None:
                delay = None
            elif delay is not None:
                delay += length / 1000.0 * tp
        n = prev
    main = max(per_width.items(), key=lambda kv: (kv[1], -kv[0][1], kv[0][0]))[0] if per_width else None
    return NetPath(ends=(a, b), length_mm=tracks_len + n_vias * barrel, track_length_mm=tracks_len, vias=n_vias, delay_s=delay, bound=bound, main=main), None


__all__ = ["CONVEX_PAD_SHAPES", "PATH_TOL_MM", "NetPads", "NetPath", "PadBox", "longest_path", "net_pads"]
