"""Wire routing for the schematic compiler (B11): trunks in the gutters of the symbol table, one branch per pin.

Pure geometry, deterministic (a function of its arguments only), so the compiler stays byte-reproducible.

Model
-----
* Every routed wire starts at a pin's **stub end** - the point where the stub leaves the pin and where the
  net's global label is anchored. The label stays: the net name, the exported netlist and ERC do not change,
  the wires are purely additive (a reader follows copper instead of matching label names).
* A net of 2 .. :data:`MAX_WIRED_PINS` pins gets one **trunk** on a track inside a channel (the gutter between
  two table columns or rows, or the sheet margin outside the table) and a **branch** from each stub end to the
  trunk: straight when the pin faces the trunk, an L (stub axis -> a channel on the pin's side -> trunk)
  otherwise. Where three or more wire ends meet a :class:`Routed.junctions` entry marks the point.
* **Nothing may connect by accident.** KiCad joins wire ends that coincide, a wire end lying on another wire,
  and a wire passing through a pin or label anchor. A candidate path is refused when any segment crosses a
  symbol's extent (body, pins, stubs, label text), passes through a foreign connection point, ends on a foreign
  wire or has a foreign wire end on it, or overlaps a foreign wire collinearly. Crossing another net's wire
  mid-segment does not connect in KiCad and is allowed. The only segment allowed inside an extent is a pin's
  first branch segment inside its *own* symbol's extent (it runs out through the label text).
* **All or nothing per net**: when no candidate path is clear the net keeps stubs + labels only and the caller
  records why; a half-routed wire would dangle.
* Two nets never share a track: each channel hands out tracks 0, +1, -1, +2, ... (1.27 mm apart) in the order
  the nets are routed. A channel that ran out (:func:`channel_full`) is a matter of the table's geometry, not of
  this module: the compiler widens that gutter and routes the sheet again (B11 W3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ai_eda.compilers.schematic_layout import GRID_MM, Extent, Vec, point_on_segment, snap

__all__ = ["MAX_WIRED_PINS", "TRACK_STEP_MM", "CHANNEL_CLEARANCE_MM", "RoutePin", "Channels", "Routed", "route_net", "foreign_point_on_wire", "channel_full"]

Segment = tuple[Vec, Vec]

#: nets with more pins than this (a board's GND / VCC) keep labels only: a trunk with ten branches hides the circuit
MAX_WIRED_PINS = 4
#: distance between two nets' tracks in one channel
TRACK_STEP_MM = GRID_MM
#: a track keeps this much from the channel's walls (the walls are symbol extents, which already carry a 1.27 mm margin)
CHANNEL_CLEARANCE_MM = GRID_MM


@dataclass(frozen=True, slots=True)
class RoutePin:
    ref: str  # the symbol the pin belongs to (its extent is the one the first branch segment may cross)
    end: Vec  # stub end = label anchor: where the wire starts
    away: Vec  # unit vector along the stub, away from the symbol body


@dataclass(frozen=True, slots=True)
class Channels:
    """Free corridors of the symbol table: ``vertical`` = (xmin, xmax) of each gutter between columns, ``horizontal`` = (ymin, ymax)
    between rows; both include the margin outside the table on each side. Order = left to right / top to bottom."""

    vertical: tuple[tuple[float, float], ...]
    horizontal: tuple[tuple[float, float], ...]


@dataclass(slots=True)
class Routed:
    segments: list[Segment]
    junctions: list[Vec]
    length: float = 0.0
    tracks_used: list[tuple[str, int]] = field(default_factory=list)  # ("v" | "h", channel index) per track taken


def _norm(seg: Segment) -> Segment:
    a, b = seg
    return (a, b) if a <= b else (b, a)


def _horizontal(seg: Segment) -> bool:
    return seg[0][1] == seg[1][1]


def _length(seg: Segment) -> float:
    return abs(seg[1][0] - seg[0][0]) + abs(seg[1][1] - seg[0][1])


def _hits_extent(seg: Segment, ext: Extent) -> bool:
    """An axis-aligned segment touches or crosses the box (touching counts: the box edge is a margin, not free space)."""
    (ax, ay), (bx, by) = _norm(seg)
    return not (bx < ext.xmin or ax > ext.xmax or by < ext.ymin or ay > ext.ymax)


def _on(p: Vec, seg: Segment) -> bool:
    return point_on_segment(p, seg[0], seg[1])


def _collinear_overlap(a: Segment, b: Segment) -> bool:
    if _horizontal(a) != _horizontal(b):
        return False
    if _horizontal(a):
        if a[0][1] != b[0][1]:
            return False
        (a0, _), (a1, _) = _norm(a)
        (b0, _), (b1, _) = _norm(b)
    else:
        if a[0][0] != b[0][0]:
            return False
        (_, a0), (_, a1) = _norm(a)
        (_, b0), (_, b1) = _norm(b)
    return max(a0, b0) < min(a1, b1)  # a shared interior; touching at one point is an endpoint case, judged elsewhere


def foreign_point_on_wire(segments: Sequence[Segment], points: Mapping[Vec, str]) -> str | None:
    """``"<what> at <point> lies on wire <a>-<b>"`` for the first foreign connection point on a routed segment, else ``None``.

    The router never produces such a segment; the compiler calls this once more on the final wires as the
    guard that turns any slip into a :class:`~ai_eda.errors.CompileError` instead of a silently merged net.
    """
    for seg in segments:
        for point, what in points.items():
            if _on(point, seg):
                return f"{what} at {point} lies on wire {seg[0]}-{seg[1]}"
    return None


def _clear(
    segments: Sequence[tuple[Segment, str | None]],
    extents: Mapping[str, Extent],
    points: Mapping[Vec, str],
    foreign: Sequence[Segment],
) -> bool:
    """True when every candidate segment is free of symbols, foreign connection points and foreign wires."""
    foreign_ends = {p for seg in foreign for p in seg}
    own_ends = {p for seg, _ in segments for p in seg}
    for seg, allowed_ref in segments:
        if seg[0] == seg[1]:
            continue
        for ref, ext in extents.items():
            if ref != allowed_ref and _hits_extent(seg, ext):
                return False
        if any(_on(p, seg) for p in points):
            return False
        if any(_on(p, seg) for p in foreign_ends):
            return False
        for other in foreign:
            if _collinear_overlap(seg, other):
                return False
    for p in own_ends:
        if any(_on(p, other) for other in foreign):
            return False
    return True


def _track(channel: tuple[float, float], index: int) -> float | None:
    """The ``index``-th track of a channel (0, +1, -1, +2, -2 ... steps from its grid-snapped centre), or None when the channel is full."""
    lo, hi = channel
    centre = snap(round(((lo + hi) / 2) / GRID_MM) * GRID_MM)
    step = (index + 1) // 2 * (1 if index % 2 else -1) if index else 0
    value = snap(centre + step * TRACK_STEP_MM)
    if value < lo + CHANNEL_CLEARANCE_MM or value > hi - CHANNEL_CLEARANCE_MM:
        return None
    return value


def channel_full(channel: tuple[float, float], taken: int) -> bool:
    """Whether a channel that handed out ``taken`` tracks has none left (the compiler widens such a gutter and routes again)."""
    return _track(channel, taken) is None


def _normalise(segments: list[Segment]) -> tuple[list[Segment], list[Vec]]:
    """Merge collinear overlapping pieces of one net, split every span at the points where other pieces end, and list the
    points where three or more wire ends meet (junctions). Deterministic: sorted everywhere."""
    pieces = [_norm(s) for s in segments if s[0] != s[1]]
    spans: list[Segment] = []
    for horizontal in (True, False):
        lines: dict[float, list[tuple[float, float]]] = {}
        for seg in pieces:
            if _horizontal(seg) != horizontal:
                continue
            if horizontal:
                lines.setdefault(seg[0][1], []).append((seg[0][0], seg[1][0]))
            else:
                lines.setdefault(seg[0][0], []).append((seg[0][1], seg[1][1]))
        for coord in sorted(lines):
            merged: list[list[float]] = []
            for lo, hi in sorted(lines[coord]):
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                spans.append(((lo, coord), (hi, coord)) if horizontal else ((coord, lo), (coord, hi)))
    ends: set[Vec] = {p for seg in spans for p in seg}
    out: list[Segment] = []
    for seg in spans:
        cuts = sorted({p for p in ends if _on(p, seg)} | set(seg), key=lambda p: (p[0], p[1]))
        out += [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    degree: dict[Vec, int] = {}
    for seg in out:
        for p in seg:
            degree[p] = degree.get(p, 0) + 1
    junctions = sorted(p for p, n in degree.items() if n >= 3)
    return sorted(out), junctions


def _branch(
    pin: RoutePin, trunk_axis: str, trunk_at: float, channels: Channels, used: Mapping[tuple[str, int], int]
) -> tuple[list[tuple[Segment, str | None]], list[tuple[str, int]]] | None:
    """Segments from a pin's stub end to the trunk line, or None when the pin cannot reach it.

    A pin whose stub points along the trunk's normal goes straight; one whose stub runs parallel to the trunk
    leaves along its stub into the nearest channel on its away side and turns there.
    """
    ex, ey = pin.end
    ax, ay = pin.away
    if trunk_axis == "h":  # the trunk is the line y = trunk_at
        if ay:
            if (trunk_at - ey) * ay <= 0:
                return None
            return [(((ex, ey), (ex, trunk_at)), pin.ref)], []
        side = [i for i, (lo, hi) in enumerate(channels.vertical) if ((lo + hi) / 2 - ex) * ax > 0]
        if not side:
            return None
        i = min(side, key=lambda k: abs((channels.vertical[k][0] + channels.vertical[k][1]) / 2 - ex))
        x = _track(channels.vertical[i], used.get(("v", i), 0))
        if x is None:
            return None
        return [(((ex, ey), (x, ey)), pin.ref), (((x, ey), (x, trunk_at)), None)], [("v", i)]
    if ax:  # the trunk is the line x = trunk_at
        if (trunk_at - ex) * ax <= 0:
            return None
        return [(((ex, ey), (trunk_at, ey)), pin.ref)], []
    side = [i for i, (lo, hi) in enumerate(channels.horizontal) if ((lo + hi) / 2 - ey) * ay > 0]
    if not side:
        return None
    i = min(side, key=lambda k: abs((channels.horizontal[k][0] + channels.horizontal[k][1]) / 2 - ey))
    y = _track(channels.horizontal[i], used.get(("h", i), 0))
    if y is None:
        return None
    return [(((ex, ey), (ex, y)), pin.ref), (((ex, y), (trunk_at, y)), None)], [("h", i)]


def route_net(
    pins: Sequence[RoutePin],
    channels: Channels,
    extents: Mapping[str, Extent],
    points: Mapping[Vec, str],
    foreign: Sequence[Segment],
    used: Mapping[tuple[str, int], int],
) -> Routed | None:
    """The shortest clear trunk-and-branches route for one net, or None (keep labels only).

    ``extents``: every placed symbol's box by ref; ``points``: every connection point that is not one of this
    net's stub ends or pins (label anchors, pin ends, routed wire ends of other nets) with a description;
    ``foreign``: wires routed so far; ``used``: tracks already handed out per channel. The result's
    ``tracks_used`` tells the caller which counters to advance.
    """
    if not 2 <= len(pins) <= MAX_WIRED_PINS:
        return None
    best: Routed | None = None
    for axis, chans in (("h", channels.horizontal), ("v", channels.vertical)):
        for ci, chan in enumerate(chans):
            at = _track(chan, used.get((axis, ci), 0))
            if at is None:
                continue
            tagged: list[tuple[Segment, str | None]] = []
            tracks = [(axis, ci)]
            ok = True
            for pin in pins:
                branch = _branch(pin, axis, at, channels, used)
                if branch is None:
                    ok = False
                    break
                tagged += branch[0]
                tracks += branch[1]
            if not ok:
                continue
            # the feet: where each branch meets the trunk line; the trunk spans the outermost ones
            feet = sorted({seg[1][0] if axis == "h" else seg[1][1] for seg, _ in tagged if (seg[1][1] if axis == "h" else seg[1][0]) == at})
            if len(feet) >= 2:
                trunk = ((feet[0], at), (feet[-1], at)) if axis == "h" else ((at, feet[0]), (at, feet[-1]))
                tagged.append((trunk, None))
            if not _clear(tagged, extents, points, foreign):
                continue
            segments, junctions = _normalise([seg for seg, _ in tagged])
            length = sum(_length(s) for s in segments)
            if best is None or length < best.length:
                best = Routed(segments=segments, junctions=junctions, length=length, tracks_used=tracks)
    return best
