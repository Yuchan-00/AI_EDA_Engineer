"""Corner style ``arc`` (routing.corners 0.2): the router's corners of 90 degrees or sharper rounded into tangent arcs after routing.

The project owner's corner-style rule (2026-11-13) - a geometry convention, not an electrical requirement - on synthetic boards,
the twin of ``tests/test_routing_corners.py`` (the 45-degree style, routing.corners 0.1):

* the pass itself (:func:`ai_eda.tools.routing.corners.mitre_corners` with ``"arc"`` on hand-drawn copper over a real
  :class:`_Board`): an L rounded at the cap radius, tangent to both tracks; short runs that give the room's radius, and ones too
  short for the smallest arc; a U (a half circle) and a Z (an S-curve) on a used-up track; a chamfer beside an arc, which leaves the
  arc's end on its track; an acute corner; the nested corners of a bus, the same radius whatever their order; copper at exactly the
  limit, where only the chain's bound decides; an arc made earlier, which stands in the audit with its bulge and its bound; another
  net's via inside the corner that shrinks the arc, a pad that leaves only the chamfer, a hole that leaves the corner; a keep-out
  and the edge, each judged on the whole arc; the larger of two nets' clearances; a blind via, in the arc's way on the layers of
  its span only; the arc never joining its own net where the corner did not - judged on its true curve - and never losing a
  contact the corner had with a pad, a via or a track; every skip of the chamfer; the entries, the note, the stats and the lengths,
  a rule net's length rows among them; the net's connectivity before and after;
* the styles that existed: ``"miter45"`` and ``"orthogonal"`` give byte for byte what they gave before the arc style, and a result
  that already holds an arc is refused under every style;
* the router (``route_board(..., corner_style="arc")``; its own default stays ``"orthogonal"``) and the PCB agent, whose default
  style it is (``pcb.corners=arc`` gives the same, ``pcb.corners=miter45`` the 45-degree corners).

An arc is checked here on its own: its circle against the corner (tangent to both tracks within what the 1 nm the points are stored
at allows), and its clearance by distances computed in this file, not by the pass's audit.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from pathlib import Path

import pytest

from ai_eda.agents.keys import CORNERS_KEY
from ai_eda.agents.pcb import PCBAgent
from ai_eda.errors import CompileError
from ai_eda.ir import Track, ValidationStatus, Via, straight_pieces, track_length
from ai_eda.ir.arc import ARC_SAGITTA_MM, arc_geometry, arc_points, fillet, point_arc_distance
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.corners import (
    ARC,
    ARC_RADIUS_WIDTHS,
    CORNER_BASIS,
    CORNER_ENTRY_PREFIX,
    CORNER_RULE,
    CORNER_STYLES,
    CORNERS_ARC_VERSION,
    CORNERS_ID,
    CORNERS_VERSION,
    DEFAULT_CORNER_STYLE,
    MITER45,
    ORTHOGONAL,
    check_style,
    mitre_corners,
)
from ai_eda.tools.routing.maze import NetRule, Routing, RoutingParams, _Board, route_board
from ai_eda.tools.si.paths import longest_path, net_pads
from tests.test_pcb_agent import _ctx
from tests.test_rf_floorplan import keepout
from tests.test_routing import _connected, _routing_checks, board_ir, fixture_library
from tests.test_routing_corners import HAND, NETS, PARTS, _l_board, _poly
from tests.test_routing_layers import _n_stack
from tests.test_routing_rules import canonical

P = RoutingParams()  # grid 0.25, width 0.4, clearance 0.25, edge 0.3
W = P.track_width_mm
#: the cap of a 0.4 mm track's arcs: 4 widths, 1.6 mm (the smallest radius is 1 width, 0.4 mm)
CAP = ARC_RADIUS_WIDTHS[0] * W
L_SHAPE = ((2.0, 2.0), (10.0, 2.0), (10.0, 10.0))


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _pass(tmp_path: Path, lib: KicadLibrary, tracks: list[Track], *, style: str = ARC, parts=None, nets=None, vias=(), rules=None,
          keepouts=None) -> Routing:
    """``mitre_corners`` in ``style`` on hand-drawn copper over the router's board of a 20 x 20 mm IR (as ``_mitre`` of the 0.1 tests)."""
    ir = board_ir(tmp_path, lib, parts or PARTS, nets or NETS, (20.0, 20.0))
    board = _Board(ir, lib, P, keepouts=keepouts, ko_width=P.track_width_mm)
    lengths: dict[str, float] = {}
    for t in tracks:
        lengths[t.net] = lengths.get(t.net, 0.0) + track_length(t)
    r = Routing(params=P, tracks=list(tracks), vias=list(vias), rules=dict(rules or {}),
                stats={"net_length_mm": {k: round(v, 6) for k, v in lengths.items()}, "total_length_mm": round(sum(lengths.values()), 6),
                       "track_count": len(tracks)})
    mitre_corners(r, board, style)
    return r


def _shape(r: Routing, net: str = "A") -> list[tuple]:
    """The net's tracks as ``(start, end)``, an arc as ``(start, mid, end)``."""
    return [(t.start, t.end) if t.mid is None else (t.start, t.mid, t.end) for t in r.tracks if t.net == net]


def _arcs(r: Routing, net: str | None = None) -> list[Track]:
    return [t for t in r.tracks if t.mid is not None and (net is None or t.net == net)]


def _rows(r: Routing) -> list[dict]:
    return r.stats["corners"]["corners"]


def _row(r: Routing, at: tuple[float, float]) -> dict:
    return next(row for row in _rows(r) if tuple(row["at"]) == at)


def _leaving(t: Track, point: tuple[float, float]) -> tuple[float, float]:
    """The unit direction in which a track leaves one of its ends: along a straight track, along an arc's tangent there (from its circle)."""
    at_start = t.start == point
    if t.mid is None:
        other = t.end if at_start else t.start
        n = math.hypot(other[0] - point[0], other[1] - point[1])
        return (other[0] - point[0]) / n, (other[1] - point[1]) / n
    g = arc_geometry(t.start, t.mid, t.end)
    angle = g.a0 if at_start else g.a0 + g.sweep
    sign = (1.0 if g.sweep > 0.0 else -1.0) * (1.0 if at_start else -1.0)
    return -math.sin(angle) * sign, math.cos(angle) * sign


def _off_tangent(t1: Track, t2: Track) -> float:
    """By how much (radians) two tracks that share an end fail to continue each other there: 0 where they are tangent."""
    shared = next(p for p in (t1.start, t1.end) if p in (t2.start, t2.end))
    u, v = _leaving(t1, shared), _leaving(t2, shared)
    return abs(math.atan2(u[0] * v[1] - u[1] * v[0], -(u[0] * v[0] + u[1] * v[1])))


def _tangent_everywhere(r: Routing, net: str = "A") -> None:
    """Every arc of the net continues the one track it shares each end with. The three points of an arc are stored at 1 nm, which
    turns the tangent of an arc of radius ``r`` mm by a few 1e-6 / r radians at most (on the boards measured: below 2e-6 / r); 5e-6 / r
    is asserted."""
    tracks = [t for t in r.tracks if t.net == net]
    for arc in _arcs(r, net):
        radius = arc_geometry(arc.start, arc.mid, arc.end).r
        for point in (arc.start, arc.end):
            joined = [t for t in tracks if t is not arc and point in (t.start, t.end)]
            assert len(joined) == 1, (arc, joined)
            assert _off_tangent(arc, joined[0]) < 5e-6 / radius, (arc, joined[0], _off_tangent(arc, joined[0]))


def _grouping(tracks: list[Track], points: set[tuple]) -> set[frozenset]:
    """How the copper joins ``points`` (``(net, layer, point)`` of track ends): the groups connected through tracks that share end
    points exactly - an arc joins its two ends as a straight track does."""
    parent: dict[tuple, tuple] = {}

    def find(n: tuple) -> tuple:
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for t in tracks:
        parent[find((t.net, t.layer, t.start))] = find((t.net, t.layer, t.end))
    groups: dict[tuple, set[tuple]] = {}
    for point in points:
        groups.setdefault(find(point), set()).add(point)
    return {frozenset(group) for group in groups.values()}


def _track_ends(tracks: list[Track]) -> set[tuple]:
    return {(t.net, t.layer, point) for t in tracks for point in (t.start, t.end)}


def _point_segment(q: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(q[0] - a[0] - t * dx, q[1] - a[1] - t * dy)


def _point_box(q: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    return math.hypot(max(box[0] - q[0], 0.0, q[0] - box[2]), max(box[1] - q[1], 0.0, q[1] - box[3]))


def _copper_gap_to_box(r: Routing, box: tuple[float, float, float, float], net: str = "A") -> float:
    """The smallest gap between the net's copper and a box of another net, measured here: every track as its chords (an arc's chain),
    each chord sampled every 1 um (as fine as the chain's own bound), less half the width and - for an arc - less the chain's bound,
    by which the arc may lie nearer than its chords. So for an arc this is a lower bound, at most 2 um below the true gap."""
    best = math.inf
    for t in r.tracks:
        if t.net != net:
            continue
        bound = ARC_SAGITTA_MM if t.mid is not None else 0.0
        for piece in straight_pieces([t]):
            a, b = piece.start, piece.end
            steps = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.001))
            for k in range(steps + 1):
                q = (a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
                best = min(best, _point_box(q, box) - t.width_mm / 2.0 - bound)
    return best


def _copper_gap_to_disc(r: Routing, centre: tuple[float, float], radius: float, net: str = "A") -> float:
    """The smallest gap between the net's copper and a disc of another net, exact: a straight track by the distance of the centre from
    its segment, an arc by the distance of the centre from the arc itself."""
    return min((point_arc_distance(centre, t.start, t.mid, t.end) if t.mid is not None else _point_segment(centre, t.start, t.end))
               - t.width_mm / 2.0 - radius for t in r.tracks if t.net == net)


def _on_arc(start: tuple[float, float], mid: tuple[float, float], end: tuple[float, float], n: int = 2000) -> list[tuple[float, float]]:
    """``n + 1`` points of the arc through the three points, on its own circle, its two ends included: what a distance to the arc
    itself - not to its chords - is measured on in this file."""
    g = arc_geometry(start, mid, end)
    return [(g.cx + g.r * math.cos(g.a0 + g.sweep * k / n), g.cy + g.r * math.sin(g.a0 + g.sweep * k / n)) for k in range(n + 1)]


def _chain_gap(p: list[tuple[float, float]], q: list[tuple[float, float]]) -> float:
    """The smallest distance between two chains of chords that do not cross: a point of one against a chord of the other."""
    return min(min(_point_segment(v, a, b) for v in p for a, b in zip(q, q[1:])), min(_point_segment(v, a, b) for v in q for a, b in zip(p, p[1:])))


# --------------------------------------------------------------------------- the arc


def test_an_l_becomes_one_arc_of_the_cap_radius_tangent_to_both_tracks(tmp_path: Path, lib: KicadLibrary):
    """Runs of 8 and 4 mm allow the cap, 4 track widths (1.6 mm): the corner becomes one arc between the points 1.6 mm along the two
    tracks, on the circle that touches both tracks' lines, bulging toward the old corner; the ends stay where they were, the tracks are
    stamped, and the lengths, the track count and the stats are measured again - an arc by its own length."""
    r = _pass(tmp_path, lib, _poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0)))
    assert [t.mid is not None for t in r.tracks] == [False, True, False]
    first, arc, last = r.tracks
    assert (first.start, first.end, arc.start, arc.end, last.start, last.end) == ((4.0, 4.0), (10.4, 4.0), (10.4, 4.0), (12.0, 5.6), (12.0, 5.6), (12.0, 8.0))
    # its three points lie on the circle of radius 1.6 around (10.4, 5.6), the one tangent to both tracks 1.6 mm from the corner
    centre = (12.0 - CAP, 4.0 + CAP)
    for point in (arc.start, arc.mid, arc.end):
        assert math.hypot(point[0] - centre[0], point[1] - centre[1]) == pytest.approx(CAP, abs=1e-6)
    g = arc_geometry(arc.start, arc.mid, arc.end)
    assert (g.cx, g.cy, g.r) == pytest.approx((*centre, CAP), abs=2e-6) and math.degrees(g.sweep) == pytest.approx(90.0, abs=1e-4)
    # the middle point lies on the bisector, between the chord and the old corner
    assert arc.mid == (pytest.approx(12.0 - CAP * (1.0 - math.sqrt(0.5)), abs=1e-6), pytest.approx(4.0 + CAP * (1.0 - math.sqrt(0.5)), abs=1e-6))
    # tangent at both ends: the arc's own tangent is the direction of the track it joins
    assert _off_tangent(first, arc) < 1e-6 and _off_tangent(arc, last) < 1e-6
    assert _leaving(arc, arc.start) == pytest.approx((1.0, 0.0), abs=1e-6) and _leaving(arc, arc.end) == pytest.approx((0.0, -1.0), abs=1e-6)
    entry = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {CORNERS_ARC_VERSION}:{ARC}:at=12.0,4.0:tangent=1.6:radius=1.6"
    assert entry == "corner:routing.corners 0.2:arc:at=12.0,4.0:tangent=1.6:radius=1.6"
    for t in r.tracks:
        assert t.provenance.derived_from == [entry] and t.width_mm == 0.4 and t.layer == "F.Cu"
        note = t.provenance.note
        assert "routing.corners 0.2" in note and "rounded into a tangent arc, or mitred into a 45-degree chamfer where no arc fits" in note
        assert "a geometry convention, not an electrical requirement: no signal-integrity or manufacturing benefit is claimed" in note
    s = r.stats
    length = round(6.4 + CAP * math.pi / 2.0 + 2.4, 6)
    assert s["net_length_mm"] == {"A": pytest.approx(length, abs=2e-6)} and s["total_length_mm"] == s["net_length_mm"]["A"]
    assert s["net_length_mm"]["A"] == round(sum(track_length(t) for t in r.tracks), 6) and s["track_count"] == 3
    c = s["corners"]
    assert (c["tool"], c["version"], c["style"], c["rule"], c["basis"]) == (CORNERS_ID, "0.2", "arc", CORNER_RULE, CORNER_BASIS)
    assert (c["found"], c["arcs"], c["mitred"], c["left"], c["skipped"]) == (1, 1, 0, 0, 0)
    assert c["corners"] == [{"net": "A", "layer": "F.Cu", "at": [12.0, 4.0], "angle_deg": 90.0, "outcome": "arc", "radius_mm": 1.6, "tangent_mm": 1.6}]
    assert "4, 2 and 1 track widths, largest first" in c["arc"] and f"{ARC_SAGITTA_MM:g} mm bound on the refusing side" in c["arc"]
    assert "a halving below 0.3 x the track's width tries that smallest chamfer instead, the last try" in c["chamfer"]


def test_short_runs_give_the_radius_their_room_allows_and_too_short_ones_the_chamfer_or_nothing(tmp_path: Path, lib: KicadLibrary):
    """An arc takes half of each run: a 2 mm run leaves a tangent length of 1 mm (radius 1 mm at a right angle, between the cap and
    the smallest). A 0.5 mm run leaves 0.25 mm, below the smallest radius (1 width, 0.4 mm): no arc, the chamfer of 0.1 instead, and
    the row says why. A 0.2 mm run leaves no room for the smallest chamfer either: the corner stays, with both reasons."""
    r = _pass(tmp_path, lib, _poly("A", (8.0, 2.0), (10.0, 2.0), (10.0, 8.0)))
    assert _shape(r) == [((8.0, 2.0), (9.0, 2.0)), ((9.0, 2.0), (9.707107, 2.292893), (10.0, 3.0)), ((10.0, 3.0), (10.0, 8.0))]
    assert (_row(r, (10.0, 2.0))["radius_mm"], _row(r, (10.0, 2.0))["tangent_mm"]) == (1.0, 1.0)
    _tangent_everywhere(r)
    r = _pass(tmp_path / "jog", lib, _poly("A", (9.5, 2.0), (10.0, 2.0), (10.0, 8.0)))
    assert _shape(r) == [((9.5, 2.0), (9.75, 2.0)), ((9.75, 2.0), (10.0, 2.25)), ((10.0, 2.25), (10.0, 8.0))]
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 0.25, "no_room")
    assert row["no_arc_why"] == "the tracks leave room for an arc of radius 0.2500 mm, below the smallest (0.4 mm: 1 x the track's width)"
    assert r.tracks[1].provenance.derived_from == ["corner:routing.corners 0.2:miter45:at=10.0,2.0:chamfer=0.25"]
    assert (r.stats["corners"]["arcs"], r.stats["corners"]["mitred"]) == (0, 1)
    short = _poly("A", (9.8, 2.0), (10.0, 2.0), (10.0, 8.0))
    r = _pass(tmp_path / "short", lib, short)
    row = _row(r, (10.0, 2.0))
    assert r.tracks == short and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "no_room", "no_room")
    assert row["why"] == ("the tracks leave room for an arc of radius 0.1000 mm, below the smallest (0.4 mm: 1 x the track's width); the tracks "
                          "leave room for a chamfer of 0.1000 mm, below the smallest (0.12 mm: 0.3 x the track's width)")


def test_a_u_becomes_a_half_circle_and_a_z_an_s_curve(tmp_path: Path, lib: KicadLibrary):
    """Two corners share a 3 mm track. Each arc takes half of it, whichever way the next corner turns: the track is used up and the two
    arcs meet at its middle, both tangent to its line - a U's two quarter arcs of 1.5 mm are a half circle (one centre, the same turn),
    a Z's an S-curve (opposite turns, a common tangent). The chamfer keeps a third of the track in a U; the arc needs none."""
    u = _pass(tmp_path, lib, _poly("A", (4.0, 10.0), (4.0, 4.0), (7.0, 4.0), (7.0, 10.0)))
    assert _shape(u) == [((4.0, 10.0), (4.0, 5.5)), ((4.0, 5.5), (4.43934, 4.43934), (5.5, 4.0)), ((5.5, 4.0), (6.56066, 4.43934), (7.0, 5.5)),
                         ((7.0, 5.5), (7.0, 10.0))]
    a1, a2 = _arcs(u)
    g1, g2 = arc_geometry(a1.start, a1.mid, a1.end), arc_geometry(a2.start, a2.mid, a2.end)
    assert (g1.cx, g1.cy, g1.r) == pytest.approx((5.5, 5.5, 1.5), abs=2e-6) and (g2.cx, g2.cy, g2.r) == pytest.approx((5.5, 5.5, 1.5), abs=2e-6)
    assert math.degrees(g1.sweep) == pytest.approx(90.0, abs=1e-4) and math.degrees(g2.sweep) == pytest.approx(90.0, abs=1e-4)  # the same turn
    assert a1.end == a2.start == (5.5, 4.0) and _off_tangent(a1, a2) < 1e-6
    _tangent_everywhere(u)
    assert [row["radius_mm"] for row in _rows(u)] == [1.5, 1.5] and u.stats["track_count"] == 4
    assert u.stats["net_length_mm"]["A"] == pytest.approx(4.5 + math.pi * 1.5 + 4.5, abs=2e-6)
    z = _pass(tmp_path / "z", lib, _poly("A", (2.0, 4.0), (6.0, 4.0), (6.0, 7.0), (10.0, 7.0)))
    assert _shape(z) == [((2.0, 4.0), (4.5, 4.0)), ((4.5, 4.0), (5.56066, 4.43934), (6.0, 5.5)), ((6.0, 5.5), (6.43934, 6.56066), (7.5, 7.0)),
                         ((7.5, 7.0), (10.0, 7.0))]
    a1, a2 = _arcs(z)
    g1, g2 = arc_geometry(a1.start, a1.mid, a1.end), arc_geometry(a2.start, a2.mid, a2.end)
    assert math.degrees(g1.sweep) == pytest.approx(90.0, abs=1e-4) and math.degrees(g2.sweep) == pytest.approx(-90.0, abs=1e-4)  # opposite turns
    assert (g1.cx, g1.cy) == pytest.approx((4.5, 5.5), abs=2e-6) and (g2.cx, g2.cy) == pytest.approx((7.5, 5.5), abs=2e-6)
    assert a1.end == a2.start == (6.0, 5.5) and _off_tangent(a1, a2) < 1e-6
    _tangent_everywhere(z)
    assert z.stats["track_count"] == 4 and z.stats["corners"]["arcs"] == 2
    # a longer shared track keeps a straight piece between two arcs of the cap radius
    wide = _pass(tmp_path / "wide", lib, _poly("A", (4.0, 10.0), (4.0, 4.0), (9.0, 4.0), (9.0, 10.0)))
    assert [len(piece) for piece in _shape(wide)] == [2, 3, 2, 3, 2] and _shape(wide)[2] == ((5.6, 4.0), (7.4, 4.0))
    _tangent_everywhere(wide)


def test_a_chamfer_beside_an_arc_keeps_the_arcs_end_on_its_track(tmp_path: Path, lib: KicadLibrary):
    """A right-angle corner and a 45-degree corner share a 1.6 mm track. The first is rounded with half of it (0.8 mm). The second has
    no room for the smallest arc (at 45 degrees a tangent length of 0.8 mm gives a radius of 0.33 mm), so it gets the chamfer - which
    may take only half of what is left of the track up to the arc's end: 0.4 mm of track stays between them, and the arc still meets
    its track in the track's direction. (The 45-degree style, with two chamfers, uses the track up as it always did.)"""
    tracks = _poly("A", (2.0, 4.0), (6.0, 4.0), (6.0, 5.6), (9.0, 2.6))
    r = _pass(tmp_path, lib, tracks)
    assert _shape(r) == [((2.0, 4.0), (5.2, 4.0)), ((5.2, 4.0), (5.765685, 4.234315), (6.0, 4.8)), ((6.0, 4.8), (6.0, 5.2)),
                         ((6.0, 5.2), (6.282843, 5.317157)), ((6.282843, 5.317157), (9.0, 2.6))]
    assert (_row(r, (6.0, 4.0))["outcome"], _row(r, (6.0, 4.0))["radius_mm"]) == ("arc", 0.8)
    row = _row(r, (6.0, 5.6))
    assert (row["outcome"], row["angle_deg"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 45.0, 0.4, "no_room")
    assert row["no_arc_why"] == "the tracks leave room for an arc of radius 0.3314 mm, below the smallest (0.4 mm: 1 x the track's width)"
    _tangent_everywhere(r)
    assert [t.provenance.derived_from for t in r.tracks[2:4]] == [
        ["corner:routing.corners 0.2:arc:at=6.0,4.0:tangent=0.8:radius=0.8", "corner:routing.corners 0.2:miter45:at=6.0,5.6:chamfer=0.4"],
        ["corner:routing.corners 0.2:miter45:at=6.0,5.6:chamfer=0.4"]]
    assert "routing.corners 0.2: 2 corner(s) of 90 degrees or sharper of this net rounded into a tangent arc, or mitred" in r.tracks[2].provenance.note
    old = _pass(tmp_path / "miter45", lib, tracks, style=MITER45)
    assert _shape(old) == [((2.0, 4.0), (5.2, 4.0)), ((5.2, 4.0), (6.0, 4.8)), ((6.0, 4.8), (6.565685, 5.034315)), ((6.565685, 5.034315), (9.0, 2.6))]


def test_an_acute_corner_gets_the_arc_of_its_angle(tmp_path: Path, lib: KicadLibrary):
    """A 60-degree corner: the arc of tangent length ``d`` has the radius ``d tan(30 degrees)`` and sweeps 120 degrees. The cap of 1.6 mm
    asks for d = 1.6 / tan(30 degrees) = 2.771281 mm, which runs of 10 and 8 mm allow; an obtuse bend is no corner, as for the chamfer."""
    far = (12.0 - 8.0 * math.cos(math.radians(60.0)), 2.0 + 8.0 * math.sin(math.radians(60.0)))
    r = _pass(tmp_path, lib, _poly("A", (2.0, 2.0), (12.0, 2.0), far))
    row = _row(r, (12.0, 2.0))
    assert (row["outcome"], row["angle_deg"], row["radius_mm"]) == ("arc", 60.0, 1.6)
    assert row["tangent_mm"] == round(CAP / math.tan(math.radians(30.0)), 6) == 2.771281
    arc, = _arcs(r)
    g = arc_geometry(arc.start, arc.mid, arc.end)
    assert g.r == pytest.approx(row["tangent_mm"] * math.tan(math.radians(30.0)), abs=2e-6) and abs(math.degrees(g.sweep)) == pytest.approx(120.0, abs=1e-4)
    assert arc.start == (round(12.0 - row["tangent_mm"], 6), 2.0) and math.hypot(arc.end[0] - 12.0, arc.end[1] - 2.0) == pytest.approx(row["tangent_mm"], abs=2e-6)
    _tangent_everywhere(r)
    assert arc.provenance.derived_from == ["corner:routing.corners 0.2:arc:at=12.0,2.0:tangent=2.771281:radius=1.6"]
    obtuse = _poly("A", (2.0, 2.0), (6.0, 2.0), (10.0, 6.0))
    r = _pass(tmp_path / "o", lib, obtuse)
    assert r.tracks == obtuse and r.stats["corners"]["found"] == 0


#: three nets side by side through one bend: the outer net A, then B and C inside it
BUS_PARTS = [*PARTS, ("R5", "SMD1", 8.0, 18.0), ("R6", "SMD1", 10.0, 18.0)]
BUS_NETS = {"A": NETS["A"], "B": NETS["B"], "C": [("R5", "1"), ("R6", "1")]}


def _bus(pitch: float, order: str) -> list[Track]:
    """Three L-shaped nets ``pitch`` apart, their corners nested on one diagonal; the tracks in the order of ``order``."""
    nets = {name: _poly(name, (2.0, 4.0 + k * pitch), (12.0 - k * pitch, 4.0 + k * pitch), (12.0 - k * pitch, 14.0)) for k, name in enumerate("ABC")}
    return [t for name in order for t in nets[name]]


@pytest.mark.parametrize("pitch", [0.75, 0.652])
def test_nested_bus_corners_get_the_same_radius_whatever_their_order(tmp_path: Path, lib: KicadLibrary, pitch: float):
    """Three nets around one bend, 0.75 mm apart (the router's grid pitch for a 0.4 mm track at 0.25 mm clearance) and 0.652 mm apart
    (2 um more than width + clearance: twice the chain's bound). The outer corner cannot be rounded at 1.6 mm while the corner inside
    it is square (the arc would pass too near that corner's point), so with the outer net first the round at 4 widths is swept
    again: the inner corner passes first, then the middle one against the rounded inner one, then the outer one. All three get the
    cap, with the tracks drawn outer first and inner first - the same copper per net - and every arc keeps the clearance from its
    neighbours (measured here on the arcs' own circles)."""
    results = {order: _pass(tmp_path / order, lib, _bus(pitch, order), parts=BUS_PARTS, nets=BUS_NETS) for order in ("ABC", "CBA")}
    for order, r in results.items():
        assert {row["net"]: (row["outcome"], row["radius_mm"]) for row in _rows(r)} == {name: ("arc", CAP) for name in "ABC"}, order
        assert [row["net"] for row in _rows(r)] == list(order)
        for name in "ABC":
            _tangent_everywhere(r, name)
    for name in "ABC":
        assert _shape(results["ABC"], name) == _shape(results["CBA"], name)
    r = results["ABC"]
    centre = {t.net: arc_geometry(t.start, t.mid, t.end) for t in _arcs(r)}
    for outer, inner in (("A", "B"), ("B", "C")):
        # the same radius on centres one pitch apart on the diagonal: the two arcs are nowhere nearer than the pitch
        go, gi = centre[outer], centre[inner]
        assert (gi.cx - go.cx, gi.cy - go.cy) == pytest.approx((-pitch, pitch), abs=3e-6)
        samples = [(go.cx + go.r * math.cos(go.a0 + go.sweep * k / 200.0), go.cy + go.r * math.sin(go.a0 + go.sweep * k / 200.0)) for k in range(201)]
        inner_arc = next(t for t in _arcs(r) if t.net == inner)
        assert min(point_arc_distance(q, inner_arc.start, inner_arc.mid, inner_arc.end) for q in samples) - W >= P.clearance_mm


def test_where_copper_runs_at_exactly_the_limit_only_the_bound_would_decide_and_no_arc_is_made(tmp_path: Path, lib: KicadLibrary):
    """An arc is judged on its chords with their 1 um bound on the refusing side. Where a track runs at exactly the limit - width +
    clearance from its neighbour, or the edge clearance from the outline - the arc's first chord is as near as the track was, and
    with the bound it is within the limit: the true arc is legal, but only the bound would say so, so it is not made. The chamfer,
    whose audit is exact, is made instead (the row names the bound). 2 um more room and the arc is made."""
    r = _pass(tmp_path, lib, _bus(0.65, "CBA"), parts=BUS_PARTS, nets=BUS_NETS)
    rows = {row["net"]: row for row in _rows(r)}
    for name, neighbour in (("C", "B"), ("B", "A")):  # each has the next net's track 0.65 mm beside its own
        assert (rows[name]["outcome"], rows[name]["no_arc_reason"]) == ("mitred", "clearance"), rows[name]
        assert rows[name]["no_arc_why"].startswith("no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (")
        assert (f"judged on its chords with the half-width raised by their 0.001 mm bound, comes 0.2490 mm from the track of {neighbour} "
                "(needs 0.25 mm)") in rows[name]["no_arc_why"]
    assert rows["A"]["outcome"] == "arc"  # nothing runs beside the outer net's track on its outer side
    edge = _pass(tmp_path / "edge", lib, _poly("A", (2.0, 0.5), (10.0, 0.5), (10.0, 8.0)))  # 0.5 mm: the edge clearance + half the width
    row = _row(edge, (10.0, 0.5))
    assert (row["outcome"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 3.75, "edge")
    assert row["no_arc_why"].endswith("the arc at (10, 0.5) would come nearer the board edge than 0.3 mm, or within its chords' 0.001 mm bound of that")
    clear = _pass(tmp_path / "clear", lib, _poly("A", (2.0, 0.502), (10.0, 0.502), (10.0, 8.0)))
    assert _row(clear, (10.0, 0.502))["outcome"] == "arc"


def _nest(pitch: float, order: str, inner_run: float = 1.6) -> list[Track]:
    """Net A through a bend with long runs and net B's corner ``pitch`` inside it on the diagonal, B's two runs ``inner_run`` long
    (half of a run is the tangent length its arc may take); the tracks in the order of ``order``."""
    outer = _poly("A", (2.0, 4.0), (12.0, 4.0), (12.0, 14.0))
    x, y = round(12.0 - pitch, 6), round(4.0 + pitch, 6)
    inner = _poly("B", (round(x - inner_run, 6), y), (x, y), (x, round(y + inner_run, 6)))
    return [t for name in order for t in (outer if name == "A" else inner)]


def _cap_arc_gaps(r: Routing) -> tuple[float, float, float]:
    """Net A's arc of the cap radius at (12, 4) - the arc itself, whether the pass made it or not - against net B's copper in ``r``:
    its smallest copper gap to B's arc (on the two circles), to B's two tracks, and to the straight line between the two ends of
    B's arc. Measured here, not by the pass's audit."""
    a, mid, b, _radius = fillet((12.0, 4.0), (2.0, 4.0), (12.0, 14.0), CAP)
    samples = _on_arc(a, mid, b)
    inner, = _arcs(r, "B")
    tracks = [t for t in r.tracks if t.net == "B" and t.mid is None]
    return (min(point_arc_distance(q, inner.start, inner.mid, inner.end) for q in samples) - W,
            min(_point_segment(q, t.start, t.end) for q in samples for t in tracks) - W,
            min(_point_segment(q, inner.start, inner.end) for q in samples) - W)


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_an_arc_made_earlier_stands_in_the_audit_with_its_bulge_and_its_bound(tmp_path: Path, lib: KicadLibrary, order: str):
    """An arc that is made is copper for every corner tried after it - its chords, each with the chain's bound - not the straight line
    between its two ends. Net B's corner, inside net A's, has runs of 1.6 mm and is rounded at 0.8 mm; net A's corner then asks for
    the cap, a flatter arc that comes nearest B's arc in the middle of the bend:

    * 0.69 mm apart, A's 1.6 mm arc would pass 0.244 mm from B's arc (measured here on the two circles) - yet 0.283 mm from B's two
      tracks and from the straight line between the ends of B's arc. Only the bulge of B's arc is in its way: the cap is refused and
      0.8 mm is made;
    * 0.6957 mm apart the two arcs would be 0.2525 mm apart - legal by 2.5 um. The audit measures the two chains (the outer arc's
      chord lies 0.9 um inside its arc where the two are nearest) and takes the bound of each chain, 1 um, off: it reads 0.2496 mm.
      The distance lies within what the bounds leave open, so the cap is not made here either;
    * 0.70 mm apart (0.2586 mm) the cap is made.

    The same whichever net's tracks come first."""
    gaps = {}
    for pitch, radius in ((0.69, 0.8), (0.6957, 0.8), (0.70, CAP)):
        r = _pass(tmp_path / f"{pitch}", lib, _nest(pitch, order))
        rows = {row["net"]: row for row in _rows(r)}
        assert (rows["B"]["outcome"], rows["B"]["radius_mm"]) == ("arc", 0.8), (pitch, rows["B"])
        assert (rows["A"]["outcome"], rows["A"]["radius_mm"]) == ("arc", radius), (pitch, rows["A"])
        for name in "AB":
            _tangent_everywhere(r, name)
        gaps[pitch] = _cap_arc_gaps(r)
        if pitch == 0.6957:  # what the audit reads there: the two chains, less the bound of each
            inner, = _arcs(r, "B")
            chains = _chain_gap(arc_points(*fillet((12.0, 4.0), (2.0, 4.0), (12.0, 14.0), CAP)[:3]), inner.points()) - W
            assert chains == pytest.approx(0.25157, abs=1e-5) and chains - 2.0 * ARC_SAGITTA_MM < P.clearance_mm < chains
    to_arc, to_tracks, to_chord = gaps[0.69]
    assert to_arc == pytest.approx(0.24444, abs=1e-5) and to_tracks == pytest.approx(0.28338, abs=1e-5)
    assert to_arc < P.clearance_mm < min(to_tracks, to_chord)  # the refused cap arc breaks the clearance against the bulge only
    assert gaps[0.6957][0] == pytest.approx(0.25250, abs=1e-5)
    assert P.clearance_mm < gaps[0.6957][0] < P.clearance_mm + 3.0 * ARC_SAGITTA_MM  # legal, by less than the bounds take
    assert gaps[0.70][0] == pytest.approx(0.25858, abs=1e-5)


# --------------------------------------------------------------------------- the audit


def test_another_nets_via_inside_the_corner_shrinks_the_arc(tmp_path: Path, lib: KicadLibrary):
    """B's via at (9, 3), inside the triangle of the corner and the two cut points of the 1.6 mm arc. The 4 mm chamfer of the 45-degree
    style passes it on its far side; the 1.6 mm arc would run 0.15 mm from it, so the arc has its own audit: 1.6 mm is refused, 0.8 mm
    (2 widths) is made and keeps 0.42 mm - measured here by the exact distance of the via's centre from the arc."""
    via = Via(net="B", x_mm=9.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    r = _pass(tmp_path, lib, _poly("A", *L_SHAPE), vias=(via,))
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["radius_mm"], row["tangent_mm"]) == ("arc", 0.8, 0.8)
    assert _shape(r) == [((2.0, 2.0), (9.2, 2.0)), ((9.2, 2.0), (9.765685, 2.234315), (10.0, 2.8)), ((10.0, 2.8), (10.0, 10.0))]
    assert _copper_gap_to_disc(r, (9.0, 3.0), 0.4) >= P.clearance_mm  # the whole net: its tracks pass 0.4 mm from the via, the arc further
    arc, = _arcs(r)
    assert point_arc_distance((9.0, 3.0), arc.start, arc.mid, arc.end) - 0.6 == pytest.approx(math.hypot(0.2, 1.0) - 0.6, abs=1e-6)
    # the arc that was refused: its circle passes 1.6 - |(9, 3) - (8.4, 3.6)| = 0.751 mm from the via's centre, 0.151 mm of gap
    assert point_arc_distance((9.0, 3.0), (8.4, 2.0), (9.531371, 2.468629), (10.0, 3.6)) - 0.6 == pytest.approx(0.1515, abs=1e-4)
    chamfered = _pass(tmp_path / "miter45", lib, _poly("A", *L_SHAPE), vias=(via,), style=MITER45)
    assert _row(chamfered, (10.0, 2.0))["chamfer_mm"] == 4.0


def test_another_nets_pad_inside_the_corner_leaves_the_chamfer_and_a_hole_leaves_the_corner(tmp_path: Path, lib: KicadLibrary):
    """A 1.0 mm rule track with B's pad 0.25 mm from both tracks' copper inside the corner (the fixture of the 0.1 test): the arcs of
    2 mm (the room) and 1 mm (the smallest, 1 width) come too near it, so the corner falls back to the chamfer, whose last try (0.3
    widths) fits. A pad without a net beside the corner refuses every arc and every chamfer: the corner stays, with both reasons.
    Whatever the outcome, the copper keeps the clearance (measured here)."""
    rules = {"A": NetRule(width_mm=1.0, clearance_mm=0.25)}
    parts = [*PARTS, ("R5", "SMD1", 8.75, 3.25)]  # box 8.25..9.25 x 2.75..3.75
    nets = {"A": NETS["A"], "B": [("R3", "1"), ("R5", "1")]}
    r = _pass(tmp_path, lib, _poly("A", (6.0, 2.0), (10.0, 2.0), (10.0, 6.0), width=1.0), parts=parts, nets=nets, rules=rules)
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 0.3, "clearance")
    assert row["no_arc_why"] == ("no arc from a radius of 2.0000 mm down to 1.0000 mm keeps every rule: the arc at (10, 2), judged on its chords "
                                 "with the half-width raised by their 0.001 mm bound, comes 0.2452 mm from the pad R5.1 (needs 0.25 mm)")
    assert _shape(r) == [((6.0, 2.0), (9.7, 2.0)), ((9.7, 2.0), (10.0, 2.3)), ((10.0, 2.3), (10.0, 6.0))]
    assert _copper_gap_to_box(r, (8.25, 2.75, 9.25, 3.75)) >= 0.25 - 1e-6
    # B's pad beside a 0.4 mm track (the 0.1 test's, where chamfers of 4 and 2 mm touch it): the arc of the cap passes round it, its
    # circle 1.6 - |(9, 3) - (8.4, 3.6)| = 0.751 mm from the box's corner - 0.55 mm of gap, measured here
    wide = _pass(tmp_path / "pad", lib, _poly("A", *L_SHAPE), parts=[*PARTS, ("R5", "SMD1", 8.5, 3.5)], nets=nets)
    assert _row(wide, (10.0, 2.0))["radius_mm"] == CAP
    assert _copper_gap_to_box(wide, (8.0, 3.0, 9.0, 4.0)) == pytest.approx(CAP - math.hypot(0.6, 0.6) - W / 2.0, abs=2.1e-3)
    hole = _poly("A", *L_SHAPE)
    r = _pass(tmp_path / "hole", lib, hole, parts=[*PARTS, ("W1", "VBAR", 9.3, 4.45)])
    row = _row(r, (10.0, 2.0))
    assert r.tracks == hole and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "clearance", "clearance")
    assert row["why"].startswith("no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (10, 2), judged on its chords")
    assert "; no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at (10, 2) comes" in row["why"]
    assert row["why"].count("pad W1.(unnumbered) (needs 0.375 mm)") == 2 and r.stats["corners"]["left_reasons"] == {"clearance": 1}
    assert r.stats["net_length_mm"] == {"A": 16.0}


def test_a_keep_out_and_the_edge_refuse_the_arc_as_they_refuse_the_chamfer(tmp_path: Path, lib: KicadLibrary):
    """A track keep-out touching both tracks' copper inside the corner refuses every arc and every chamfer (unless it allows the net);
    a corner on copper nearer the edge than the edge clearance cannot be rounded either."""
    area = keepout("K1", [6.8, 2.2, 3.0, 3.0], forbids=["tracks"], layers=["F.Cu"])
    l_shape = _poly("A", *L_SHAPE)
    r = _pass(tmp_path, lib, l_shape, keepouts=[area])
    row = _row(r, (10.0, 2.0))
    assert r.tracks == l_shape and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "keepout", "keepout")
    assert "the arc at (10, 2) would enter track keep-out K1, or come within its chords' 0.001 mm bound of it; no chamfer from" in row["why"]
    allowed = keepout("K1", [6.8, 2.2, 3.0, 3.0], forbids=["tracks"], layers=["F.Cu"], nets=["A"])
    assert _row(_pass(tmp_path / "allowed", lib, l_shape, keepouts=[allowed]), (10.0, 2.0))["radius_mm"] == CAP
    edge = _poly("A", (2.0, 0.3), (10.0, 0.3), (10.0, 8.0))
    row = _row(_pass(tmp_path / "edge", lib, edge), (10.0, 0.3))
    assert (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "edge", "edge") and row["why"].count("board edge") == 2


def test_a_keep_out_and_the_edge_are_judged_on_the_whole_arc_with_its_bound(tmp_path: Path, lib: KicadLibrary):
    """The arc's audit reads every chord of its chain - not the straight line between its two ends - and keeps the chain's bound:

    * a 0.2 mm keep-out square inside the corner, 0.25 mm from both tracks' centrelines: the 1.6 mm arc would run 0.026 mm from the
      square's inner corner, its copper through it, while the straight line between the arc's ends passes 0.49 mm away; the 0.8 mm
      arc would run 0.022 mm from its outer corner, and the copper of the 0.4 mm arc would overlap it by 0.012 mm. No arc is made;
      the 4 mm chamfer, far from the square, is;
    * a keep-out whose edge lies exactly half a track width from the first track's centreline, on the outer side of the corner (the
      track's copper touches it, which is legal): every arc starts on that track, so its first chord is within the chain's bound of
      the limit and no arc is made; the chamfer, whose audit is exact, is. With 2 um between the copper and the keep-out the cap
      arc is made;
    * a corner whose point lies 0.3 mm from the board edge, between two tracks that leave the edge at 45 degrees (its copper is
      0.1 mm from the edge where 0.3 mm is the rule; the runs leave room for a radius of 0.45 mm): the two ends of the arc would lie
      0.62 mm from the edge, but its middle bulges toward the corner's point, to 0.487 mm where 0.5 mm is the least. No arc is
      made; the chamfer between the same two points is."""
    l_shape = _poly("A", *L_SHAPE)
    square = keepout("K1", [9.55, 2.25, 0.2, 0.2], forbids=["tracks"], layers=["F.Cu"])
    r = _pass(tmp_path, lib, l_shape, keepouts=[square])
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 4.0, "keepout")
    assert row["no_arc_why"] == ("no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (10, 2) would enter track "
                                 "keep-out K1, or come within its chords' 0.001 mm bound of it")
    # the three arcs that were refused, against the square's two corners on the diagonal - each nearer than half the track's width -,
    # and the straight line between the ends of the largest, which is not
    arcs = {d: fillet((10.0, 2.0), (2.0, 2.0), (10.0, 10.0), d)[:3] for d in (1.6, 0.8, 0.4)}
    assert point_arc_distance((9.55, 2.45), *arcs[1.6]) == pytest.approx(0.0263, abs=1e-4)
    assert point_arc_distance((9.75, 2.25), *arcs[0.8]) == pytest.approx(0.0222, abs=1e-4)
    assert point_arc_distance((9.75, 2.25), *arcs[0.4]) == pytest.approx(W / 2.0 - 0.0121, abs=1e-4)
    assert _point_segment((9.55, 2.45), arcs[1.6][0], arcs[1.6][2]) == pytest.approx(0.4950, abs=1e-4)

    for top, outcome in ((1.8, "mitred"), (1.798, "arc")):
        beside = keepout("K1", [4.0, round(top - 0.5, 6), 6.3, 0.5], forbids=["tracks"], layers=["F.Cu"])  # y from top - 0.5 to top
        row = _row(_pass(tmp_path / f"{top}", lib, l_shape, keepouts=[beside]), (10.0, 2.0))
        assert row["outcome"] == outcome, (top, row)
        if outcome == "mitred":
            assert (row["chamfer_mm"], row["no_arc_reason"]) == (4.0, "keepout")
        else:
            assert row["radius_mm"] == CAP

    pointed = _poly("A", (9.36, 0.94), (10.0, 0.3), (10.64, 0.94))
    r = _pass(tmp_path / "edge", lib, pointed)
    row = _row(r, (10.0, 0.3))
    assert (row["outcome"], row["chamfer_mm"], row["no_arc_reason"]) == ("mitred", 0.452548, "edge")
    assert _shape(r) == [((9.36, 0.94), (9.68, 0.62)), ((9.68, 0.62), (10.32, 0.62)), ((10.32, 0.62), (10.64, 0.94))]
    a, mid, b, radius = fillet((10.0, 0.3), (9.36, 0.94), (10.64, 0.94), 0.452548)
    assert (a, b) == ((9.68, 0.62), (10.32, 0.62)) and radius == pytest.approx(0.452548, abs=1e-6)
    limit = P.edge_clearance_mm + W / 2.0
    assert mid[1] == pytest.approx(0.48745, abs=1e-5) and mid[1] < limit < a[1]  # the refused arc: its ends legal, its middle not


def test_an_arc_keeps_the_larger_of_the_two_nets_clearances(tmp_path: Path, lib: KicadLibrary):
    """The audit's clearance is the larger of the two nets', for an arc as for every other copper of the router.

    * B's via at (8.9, 3.1), its disc 0.5 mm from both tracks' copper: the 1.6 mm arc passes 0.293 mm from it - made at the board's
      0.25 mm; with a rule clearance of 0.5 mm on the arc's own net, or on the via's net, it is refused and the 0.8 mm arc, which
      keeps 0.54 mm, is made;
    * an arc that is made carries its net's clearance for the corners tried after it: net B's corner 0.96 mm inside net A's (runs of
      0.8 mm: rounded at 0.4 mm). A's 1.6 mm arc would pass 0.461 mm from B's arc and 0.516 mm from B's tracks - made at the board's
      clearance, refused once B's rule asks for 0.5 mm (B's arc alone is nearer than that), in either order of the tracks."""
    via = Via(net="B", x_mm=8.9, y_mm=3.1, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    for name, rules, radius, gap in (("board", {}, CAP, 0.2929), ("own", {"A": NetRule(clearance_mm=0.5)}, 0.8, 0.5402),
                                     ("other", {"B": NetRule(clearance_mm=0.5)}, 0.8, 0.5402)):
        r = _pass(tmp_path / name, lib, _poly("A", *L_SHAPE), vias=(via,), rules=rules)
        assert (_row(r, (10.0, 2.0))["outcome"], _row(r, (10.0, 2.0))["radius_mm"]) == ("arc", radius), name
        arc, = _arcs(r)
        assert point_arc_distance((8.9, 3.1), arc.start, arc.mid, arc.end) - W / 2.0 - 0.4 == pytest.approx(gap, abs=1e-4), name
    for name, rules, radius in (("board", {}, CAP), ("rule", {"B": NetRule(clearance_mm=0.5)}, 0.8)):
        for order in ("AB", "BA"):
            r = _pass(tmp_path / name / order, lib, _nest(0.96, order, inner_run=0.8), rules=rules)
            rows = {row["net"]: row for row in _rows(r)}
            assert (rows["B"]["outcome"], rows["B"]["radius_mm"]) == ("arc", 0.4), (name, order)
            assert (rows["A"]["outcome"], rows["A"]["radius_mm"]) == ("arc", radius), (name, order, rows["A"])
            to_arc, to_tracks, _to_chord = _cap_arc_gaps(r)
            assert to_arc == pytest.approx(0.4606, abs=1e-4) and to_tracks == pytest.approx(0.5165, abs=1e-4)


def test_a_typed_via_is_in_the_arcs_way_only_on_the_layers_of_its_span(tmp_path: Path, lib: KicadLibrary):
    """On a board with a stack (6 layers: the routing layers F.Cu, In2.Cu, In3.Cu, B.Cu) a blind or buried via is copper on the layers
    of its span only, for an arc as for every other copper. Net B's blind via F.Cu - In3.Cu at (9, 3), inside the corner: on F.Cu,
    In2.Cu and In3.Cu the 1.6 mm arc would pass 0.15 mm from it and the 0.8 mm arc is made, exactly as beside a through via; on
    B.Cu, which the via does not reach, the cap is made. A buried via In2.Cu - In3.Cu of the corner's own net at the corner covers
    the corner on those two layers (skipped) and is no copper on F.Cu or B.Cu, where the corner is rounded."""
    ir = board_ir(tmp_path, lib, PARTS, NETS, (20.0, 20.0))
    board = _Board(ir, lib, P, stack=_n_stack(ir))
    assert board.layer_names == ("F.Cu", "B.Cu", "In2.Cu", "In3.Cu")

    def rounded(layer: str, via: Via) -> dict:
        tracks = _poly("A", *L_SHAPE, layer=layer)
        r = Routing(params=P, tracks=tracks, vias=[via], stats={"net_length_mm": {"A": 16.0}, "total_length_mm": 16.0, "track_count": 2})
        mitre_corners(r, board, ARC)
        for arc in _arcs(r):
            assert arc.layer == layer
        return _row(r, (10.0, 2.0))

    blind = Via(net="B", x_mm=9.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, layers=("F.Cu", "In3.Cu"), kind="blind", provenance=HAND)
    assert {layer: rounded(layer, blind)["radius_mm"] for layer in board.layer_names} == {"F.Cu": 0.8, "In2.Cu": 0.8, "In3.Cu": 0.8, "B.Cu": CAP}
    through = Via(net="B", x_mm=9.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    assert {layer: rounded(layer, through)["radius_mm"] for layer in board.layer_names} == dict.fromkeys(board.layer_names, 0.8)
    buried = Via(net="A", x_mm=10.0, y_mm=2.0, drill_mm=0.4, diameter_mm=0.8, layers=("In2.Cu", "In3.Cu"), kind="buried", provenance=HAND)
    rows = {layer: rounded(layer, buried) for layer in board.layer_names}
    assert {layer: (row["outcome"], row.get("reason"), row.get("radius_mm")) for layer, row in rows.items()} == {
        "F.Cu": ("arc", None, CAP), "B.Cu": ("arc", None, CAP), "In2.Cu": ("skipped", "via", None), "In3.Cu": ("skipped", "via", None)}


# --------------------------------------------------------------------------- the net's own copper


def test_an_arc_never_joins_its_own_net_where_the_corner_did_not(tmp_path: Path, lib: KicadLibrary):
    """The audit ignores the net's own copper, so the arc is guarded as the chamfer is, on its true curve:

    * A's own via at (9.3, 2.7), clear of both tracks' copper: the copper of the 1.6 mm arc would reach its disc - refused, 0.8 mm made;
    * A's own via with its centre on the 1.6 mm arc (its disc already under the tracks' copper, so reaching it is nothing new): the arc
      would pass through the via's centre, a junction the corner did not have - refused by the exact distance, 0.8 mm made;
    * an A track along the bisector ending 0.05 mm short of the corner: every arc and every chamfer crosses it - the corner stays."""
    l_shape = _poly("A", *L_SHAPE)
    own = lambda x, y: (Via(net="A", x_mm=x, y_mm=y, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),)  # noqa: E731
    for k, (x, y) in enumerate(((9.3, 2.7), (9.531371, 2.468629))):
        r = _pass(tmp_path / f"via{k}", lib, l_shape, vias=own(x, y))
        assert _row(r, (10.0, 2.0))["radius_mm"] == 0.8, (x, y)
    assert point_arc_distance((9.531371, 2.468629), (8.4, 2.0), (9.531371, 2.468629), (10.0, 3.6)) <= 1e-6
    bisector = _poly("A", (6.0, 6.0), (9.95, 2.05))
    r = _pass(tmp_path / "cross", lib, [*l_shape, *bisector])
    row = _row(r, (10.0, 2.0))
    assert r.tracks == [*l_shape, *bisector] and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "own_copper", "own_copper")
    assert row["why"] == ("no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (10, 2) would meet its own net's track "
                          "(6, 6)-(9.95, 2.05) (a new junction); no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at "
                          "(10, 2) would meet its own net's track (6, 6)-(9.95, 2.05) (a new junction)")
    # a branch on a track limits the arc as it limits the chamfer (half of the 1 mm to the branch point) and stays on the track
    branch = _poly("A", (9.0, 2.0), (9.0, 0.8))
    r = _pass(tmp_path / "branch", lib, [*l_shape, *branch])
    assert _row(r, (10.0, 2.0))["radius_mm"] == 0.5 and r.tracks[3:] == branch and r.tracks[0].end == (9.5, 2.0)


def test_an_arc_is_held_against_its_own_nets_copper_on_its_true_curve(tmp_path: Path, lib: KicadLibrary):
    """What an arc may reach of its own net is judged on the arc - its chain of chords - and never on the straight line between its
    two ends; an arc the pass made earlier is read the same way.

    * a short piece of A's own copper inside the corner, 0.03 mm clear of both tracks' copper (its middle 0.45 mm from both
      centrelines): the 1.6 mm arc would run 0.026 mm from its middle - while the straight line between that arc's ends passes
      0.49 mm away, too far to touch it - and the 0.8 mm arc 0.305 mm: both would join it, a contact the corner did not have. The
      0.4 mm arc keeps 0.036 mm of gap to it, joins nothing and is made;
    * a 0.2 mm pad of A at the same place (its box 0.15 mm clear of both tracks' copper): the 1.6 mm arc would cross it, the 0.8 mm
      arc's copper would overlap it by 0.036 mm; the 0.4 mm arc keeps 0.13 mm and is made;
    * two pieces of A that do not touch: a bend with long runs, and inside it, 0.5 mm away, a second bend with runs of 1.6 mm (the
      copper of the two 0.1 mm apart). The inner corner is rounded at 0.8 mm. The outer one at the cap would then run 0.376 mm from
      the inner arc - nearer than the two half widths, joining the two pieces through the bulge of that arc alone (it stays 0.46 mm
      from the inner piece's two tracks and from the straight line between the ends of its arc) - so the cap is refused and both
      corners get 0.8 mm, the two pieces still apart. In either order of the tracks."""
    l_shape = _poly("A", *L_SHAPE)
    island = _poly("A", (9.53, 2.43), (9.57, 2.47))
    r = _pass(tmp_path, lib, [*l_shape, *island])
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["radius_mm"]) == ("arc", 0.4) and r.tracks[3:] == island
    arcs = {d: fillet((10.0, 2.0), (2.0, 2.0), (10.0, 10.0), d)[:3] for d in (1.6, 0.8, 0.4)}
    reach = W  # two half widths: nearer than this, the copper of two 0.4 mm tracks touches
    assert [round(point_arc_distance((9.55, 2.45), *arcs[d]), 3) for d in (1.6, 0.8)] == [0.026, 0.305]  # the two arcs refused
    assert point_arc_distance((9.55, 2.45), *arcs[1.6]) < reach < _point_segment((9.55, 2.45), arcs[1.6][0], arcs[1.6][2])
    assert _point_segment((9.55, 2.45), arcs[1.6][0], arcs[1.6][2]) == pytest.approx(0.4950, abs=1e-4)
    made, = _arcs(r)
    assert (made.start, made.mid, made.end) == arcs[0.4]
    apart = min(_point_segment(q, island[0].start, island[0].end) for q in _on_arc(made.start, made.mid, made.end))
    assert apart == pytest.approx(reach + 0.0357, abs=1e-4)  # the arc that was made, measured on its own circle

    nets = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]}
    r = _pass(tmp_path / "pad", lib, l_shape, parts=[*PARTS, ("R5", "TINY", 9.55, 2.45)], nets=nets)  # box 9.45..9.65 x 2.35..2.55
    row = _row(r, (10.0, 2.0))
    assert (row["outcome"], row["radius_mm"]) == ("arc", 0.4)
    assert _copper_gap_to_box(r, (9.45, 2.35, 9.65, 2.55)) == pytest.approx(0.129, abs=2.1e-3)

    outer = _poly("A", (2.0, 4.0), (12.0, 4.0), (12.0, 14.0))
    inner = _poly("A", (9.9, 4.5), (11.5, 4.5), (11.5, 6.1))
    for order, tracks in (("outer first", [*outer, *inner]), ("inner first", [*inner, *outer])):
        r = _pass(tmp_path / order, lib, tracks)
        assert {tuple(row["at"]): (row["outcome"], row["radius_mm"]) for row in _rows(r)} == {(12.0, 4.0): ("arc", 0.8), (11.5, 4.5): ("arc", 0.8)}, order
        a_outer = next(t for t in _arcs(r) if (11.2, 4.0) in (t.start, t.end))
        a_inner = next(t for t in _arcs(r) if (10.7, 4.5) in (t.start, t.end))
        # the two pieces are still apart: their arcs, and each arc against the other piece's tracks
        samples = _on_arc(a_outer.start, a_outer.mid, a_outer.end)
        assert min(point_arc_distance(q, a_inner.start, a_inner.mid, a_inner.end) for q in samples) - W == pytest.approx(0.1928, abs=1e-4)
        inner_tracks = [t for t in r.tracks if t.mid is None and (t.start in ((9.9, 4.5), (11.5, 6.1)) or t.end in ((9.9, 4.5), (11.5, 6.1)))]
        assert len(inner_tracks) == 2 and min(_point_segment(q, t.start, t.end) for q in samples for t in inner_tracks) - W > 0.1 - 1e-9
        # the cap arc that was refused: nearer the inner arc than two half widths, clear of its two tracks and of its chord
        cap = _on_arc(*fillet((12.0, 4.0), (2.0, 4.0), (12.0, 14.0), CAP)[:3])
        to_arc = min(point_arc_distance(q, a_inner.start, a_inner.mid, a_inner.end) for q in cap)
        to_tracks = min(_point_segment(q, t.start, t.end) for q in cap for t in inner_tracks)
        to_chord = min(_point_segment(q, a_inner.start, a_inner.end) for q in cap)
        assert (to_arc, to_tracks, to_chord) == pytest.approx((0.3757, 0.4598, 0.4598), abs=1e-4) and to_arc < reach < min(to_tracks, to_chord)


def test_no_contact_of_the_net_is_lost(tmp_path: Path, lib: KicadLibrary):
    """New in routing.corners 0.2, for an arc and for the chamfers made under the arc style: copper of the net that touched the two
    track ends the cut removes must still touch what the cut leaves.

    * A's pad R5 lies diagonally outside the corner, its box 0.14 mm from the corner point: only the round end of the corner's copper
      touches it. Every arc and every chamfer would take that copper away - the corner stays ("contact_lost"). The 45-degree style
      (routing.corners 0.1) has no such guard and cuts the corner as it always did.
    * R5 beside the first track, touching its copper from 8.5 to 9.5 mm: the 1.6 mm arc would leave the track ending at 8.4 mm, clear
      of the pad - refused; the 0.8 mm arc leaves it ending at 9.2 mm, still on the pad - made."""
    l_shape = _poly("A", *L_SHAPE)
    nets = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]}
    corner_pad = [*PARTS, ("R5", "SMD1", 10.6, 1.4)]  # box 10.1..11.1 x 0.9..1.9
    r = _pass(tmp_path, lib, l_shape, parts=corner_pad, nets=nets)
    row = _row(r, (10.0, 2.0))
    assert r.tracks == l_shape and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "contact_lost", "contact_lost")
    words = "would cut its own net's contact with pad R5.1: the copper it removes touched it, the copper it leaves does not"
    assert row["why"] == (f"no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (10, 2) {words}; "
                          f"no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at (10, 2) {words}")
    assert r.stats["corners"]["left_reasons"] == {"contact_lost": 1}
    old = _pass(tmp_path / "miter45", lib, l_shape, parts=corner_pad, nets=nets, style=MITER45)
    assert _row(old, (10.0, 2.0)) == {"net": "A", "layer": "F.Cu", "at": [10.0, 2.0], "angle_deg": 90.0, "outcome": "mitred", "chamfer_mm": 4.0}
    beside = _pass(tmp_path / "beside", lib, l_shape, parts=[*PARTS, ("R5", "SMD1", 9.0, 1.3)], nets=nets)  # box 8.5..9.5 x 0.8..1.8
    assert _row(beside, (10.0, 2.0))["radius_mm"] == 0.8 and beside.tracks[0].end == (9.2, 2.0)


def test_no_contact_with_a_via_or_a_track_of_the_net_is_lost_either(tmp_path: Path, lib: KicadLibrary):
    """The same guard for the two other kinds of copper a net has, each touching only the round end of the corner's copper from
    diagonally outside the corner:

    * A's via at (10.4, 1.6), its centre 0.566 mm from the corner point (the disc and the track's end overlap by 0.034 mm; the
      corner is not inside the via);
    * a track of A that ends at (10.25, 1.75), 0.354 mm from the corner point (the two round ends overlap by 0.046 mm), and leads
      away from the corner.

    Every arc and every chamfer would take the corner's copper away from them: the corner stays, the row naming what would be cut.
    (The 45-degree style has no such guard and cuts both, as it always did.)"""
    l_shape = _poly("A", *L_SHAPE)
    via = Via(net="A", x_mm=10.4, y_mm=1.6, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    away = _poly("A", (10.25, 1.75), (11.5, 0.5))
    cases = (("via", l_shape, {"vias": (via,)}, "the via at (10.4, 1.6)"),
             ("track", [*l_shape, *away], {}, "its track (10.25, 1.75)-(11.5, 0.5)"))
    for name, tracks, extra, what in cases:
        r = _pass(tmp_path / name, lib, tracks, **extra)
        row = _row(r, (10.0, 2.0))
        assert r.tracks == tracks and (row["outcome"], row["reason"], row["no_arc_reason"]) == ("left", "contact_lost", "contact_lost"), name
        words = f"would cut its own net's contact with {what}: the copper it removes touched it, the copper it leaves does not"
        assert row["why"] == (f"no arc from a radius of 1.6000 mm down to 0.4000 mm keeps every rule: the arc at (10, 2) {words}; "
                              f"no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at (10, 2) {words}"), name
        assert r.stats["corners"]["left_reasons"] == {"contact_lost": 1} and r.stats["net_length_mm"]["A"] == round(sum(track_length(t) for t in tracks), 6)
        old = _pass(tmp_path / name / "miter45", lib, tracks, style=MITER45, **extra)
        assert (_row(old, (10.0, 2.0))["outcome"], _row(old, (10.0, 2.0))["chamfer_mm"]) == ("mitred", 4.0), name


def test_what_the_chamfer_skips_the_arc_skips(tmp_path: Path, lib: KicadLibrary):
    """Copper another mechanism fixed keeps its corners under the arc style too - a differential pair, a length-matched group (in the
    arc's words: a rounded corner shortens the net), an escape stub, a width change - and so does a corner at a pad centre, inside a
    pad, at or inside a via, or on another track of its net: each skipped with its reason, the copper untouched."""
    l_shape = _poly("A", *L_SHAPE)
    pair = {"A": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="B", pair_spacing_mm=0.25),
            "B": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="A", pair_spacing_mm=0.25)}
    group = {"A": NetRule(match_group="BUS", max_skew_mm=0.5)}
    escape = HAND.model_copy(update={"derived_from": ["net:A", "escape:R1.1:stub:width=0.4:points=2.0,2.0;10.0,2.0"]})
    via = lambda x, y: (Via(net="A", x_mm=x, y_mm=y, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),)  # noqa: E731
    r5 = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]}
    cases = (
        ("pair", l_shape, {"rules": pair}, "a differential pair's track"),
        ("match_group", l_shape, {"rules": group}, "a net of length-matched group BUS: a rounded corner shortens it and changes the group's skew"),
        ("escape", [*_poly("A", (2.0, 2.0), (10.0, 2.0), prov=escape), l_shape[1]], {}, "a segment of an escape stub"),
        ("width", [l_shape[0], *_poly("A", (10.0, 2.0), (10.0, 10.0), width=0.3)], {}, "the two tracks differ in width (0.4 / 0.3 mm"),
        ("pad_centre", l_shape, {"parts": [*PARTS, ("R5", "SMD1", 10.0, 2.0)], "nets": r5}, "at a pad centre of its net"),
        ("in_pad", l_shape, {"parts": [*PARTS, ("R5", "SMD1", 10.2, 2.2)], "nets": r5}, "inside the box of pad R5.1 of its net"),
        ("via", l_shape, {"vias": via(10.0, 2.0)}, "at or inside a via of its net"),
        ("via", l_shape, {"vias": via(10.2, 2.2)}, "at or inside a via of its net"),
        ("junction", [*l_shape, *_poly("A", (9.0, 1.0), (11.0, 3.0))], {}, "on another track of its net (a junction)"),
    )
    for k, (reason, tracks, extra, words) in enumerate(cases):
        r = _pass(tmp_path / f"{k}", lib, tracks, **extra)
        row = _row(r, (10.0, 2.0))
        assert r.tracks == tracks and (row["outcome"], row["reason"]) == ("skipped", reason) and row["why"].startswith(words), row
        assert r.stats["corners"]["skipped_reasons"] == {reason: 1} and r.stats["corners"]["arcs"] == 0
        assert set(row) == {"net", "layer", "at", "angle_deg", "outcome", "reason", "why"}


# --------------------------------------------------------------------------- the records


def test_the_counts_add_up_and_entries_notes_and_lengths_follow_the_copper(tmp_path: Path, lib: KicadLibrary):
    """One board with every outcome: an L that is rounded, a jog whose 0.5 mm run leaves only the chamfer, a 0.2 mm jog that stays, a
    corner at a via that is skipped. found = arcs + mitred + left + skipped; each track the pass made or shortened carries one entry
    per corner in the format of its outcome (floats at 1 nm), untouched tracks are the same objects; the lengths are the sums of the
    tracks' own lengths, an arc counted as an arc."""
    via = Via(net="A", x_mm=6.0, y_mm=12.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    rounded = _poly("A", *L_SHAPE)
    chamfered = _poly("B", (13.5, 12.0), (14.0, 12.0), (14.0, 16.0))
    kept = _poly("B", (15.8, 3.0), (16.0, 3.0), (16.0, 8.0))
    at_via = _poly("A", (2.0, 12.0), (6.0, 12.0), (6.0, 16.0))
    r = _pass(tmp_path, lib, [*rounded, *chamfered, *kept, *at_via], vias=(via,))
    c = r.stats["corners"]
    assert (c["version"], c["style"]) == (CORNERS_ARC_VERSION, ARC) == ("0.2", "arc")
    assert (c["found"], c["arcs"], c["mitred"], c["left"], c["skipped"]) == (4, 1, 1, 1, 1) and c["found"] == c["arcs"] + c["mitred"] + c["left"] + c["skipped"]
    assert [row["outcome"] for row in c["corners"]] == ["arc", "skipped", "mitred", "left"]
    assert (c["left_reasons"], c["skipped_reasons"]) == ({"no_room": 1}, {"via": 1})
    # the keys the arc style had, then the ones its straightening adds (tests/test_routing_straighten.py; no sidestep on this board)
    assert list(c) == ["tool", "version", "style", "rule", "basis", "ideas", "arc", "chamfer", "found", "arcs", "mitred", "left", "skipped",
                       "left_reasons", "skipped_reasons", "corners",
                       "straighten", "sidesteps", "straightened", "sidesteps_left", "sidestep_reasons", "sidestep_rows"]
    assert (c["sidesteps"], c["straightened"], c["sidesteps_left"], c["sidestep_reasons"], c["sidestep_rows"]) == (0, 0, 0, {}, [])
    arc_entry = re.compile(r"corner:routing\.corners 0\.2:arc:at=-?\d+\.\d+,-?\d+\.\d+:tangent=\d+\.\d+:radius=\d+\.\d+")
    chamfer_entry = re.compile(r"corner:routing\.corners 0\.2:miter45:at=-?\d+\.\d+,-?\d+\.\d+:chamfer=\d+\.\d+")
    entries = [e for t in r.tracks for e in t.provenance.derived_from]
    assert sorted(set(entries)) == ["corner:routing.corners 0.2:arc:at=10.0,2.0:tangent=1.6:radius=1.6", "corner:routing.corners 0.2:miter45:at=14.0,12.0:chamfer=0.25"]
    assert arc_entry.fullmatch(entries[0]) and chamfer_entry.fullmatch("corner:routing.corners 0.2:miter45:at=14.0,12.0:chamfer=0.25")
    assert len(entries) == 6  # the arc and its two tracks, the chamfer and its two tracks
    assert [t for t in r.tracks if not t.provenance.derived_from] == [*kept, *at_via] and all(t is u for t, u in zip(r.tracks[6:], [*kept, *at_via]))
    for t in r.tracks:
        assert (t.mid is not None) == (t.net == "A" and t.start == (8.4, 2.0))
        if t.provenance.derived_from:
            assert "routing.corners 0.2: 1 corner(s) of 90 degrees or sharper of this net rounded into a tangent arc" in t.provenance.note
    for net in "AB":
        assert r.stats["net_length_mm"][net] == round(sum(track_length(t) for t in r.tracks if t.net == net), 6)
    assert r.stats["total_length_mm"] == round(sum(r.stats["net_length_mm"].values()), 6) and r.stats["track_count"] == len(r.tracks) == 10
    assert r.stats["net_length_mm"]["A"] == pytest.approx(6.4 + CAP * math.pi / 2.0 + 6.4 + 8.0, abs=2e-6)
    again = _pass(tmp_path / "again", lib, [*rounded, *chamfered, *kept, *at_via], vias=(via,))
    assert canonical(again) == canonical(r)


def test_a_rule_nets_length_rows_are_measured_again_on_the_arcs(tmp_path: Path, lib: KicadLibrary):
    """The lengths a rule net is judged by follow the rounded copper, an arc counted by its own length:

    * the routed L (11 mm) under a length budget of 12 mm: with the corner rounded at the cap, the budget row of ``stats["rules"]``
      reads 4.4 + 1.6 pi / 2 + 3.4 = 10.313 mm, the net's length - shorter than routed, so the budget still holds;
    * hand-drawn copper of a net with a neck-down rule: an L drawn at the neck-down width (0.25 mm; runs of 8 and 4 mm) and 4 mm at
      the full width. The corner is rounded at 4 neck-down widths (1 mm), and the row's neck-down length is the neck-down copper
      as it now is, 7 + pi / 2 + 3 mm - the arc is neck-down copper too."""
    ir = _l_board(tmp_path, lib)
    rules = {"A": NetRule(max_length_mm=12.0)}
    plain, r = route_board(ir, lib, rules=rules), route_board(ir, lib, rules=rules, corner_style=ARC)
    assert plain.stats["rules"]["A"]["length_mm"] == 11.0 and len(_arcs(r)) == 1
    length = round(sum(track_length(t) for t in r.tracks), 6)
    assert r.stats["rules"]["A"]["length_mm"] == r.stats["net_length_mm"]["A"] == r.stats["total_length_mm"] == length
    assert length == pytest.approx(4.4 + CAP * math.pi / 2.0 + 3.4, abs=2e-6) and length < 11.0

    board = _Board(board_ir(tmp_path / "neck", lib, PARTS, NETS, (20.0, 20.0)), lib, P)
    tracks = [*_poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0), width=0.25), *_poly("A", (12.0, 8.0), (12.0, 12.0), width=0.6)]
    row = {"routed": True, "neckdown_pads": ["R1.1"], "neckdown_length_mm": 12.0, "length_mm": 16.0}
    r = Routing(params=P, tracks=tracks, rules={"A": NetRule(width_mm=0.6, neckdown_width_mm=0.25)},
                stats={"net_length_mm": {"A": 16.0}, "total_length_mm": 16.0, "track_count": 3, "rules": {"A": row}})
    mitre_corners(r, board, ARC)
    assert (_row(r, (12.0, 4.0))["outcome"], _row(r, (12.0, 4.0))["radius_mm"]) == ("arc", 1.0)
    arc, = _arcs(r)
    assert arc.width_mm == 0.25 and (arc.start, arc.end) == ((11.0, 4.0), (12.0, 5.0))
    neck = round(sum(track_length(t) for t in r.tracks if t.width_mm == 0.25), 6)
    assert r.stats["rules"]["A"]["neckdown_length_mm"] == neck == pytest.approx(7.0 + math.pi / 2.0 + 3.0, abs=2e-6)
    assert r.stats["rules"]["A"]["length_mm"] == r.stats["net_length_mm"]["A"] == round(neck + 4.0, 6)


#: two nets whose hand-drawn copper joins their pads: A in a staircase of three corners, B in a U
JOINED_PARTS = [("R1", "SMD1", 3.0, 3.0), ("R2", "SMD1", 15.0, 15.0), ("R3", "SMD1", 3.0, 17.0), ("R4", "SMD1", 10.0, 17.0)]
JOINED_NETS = {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}


def _joined() -> list[Track]:
    return [*_poly("A", (3.0, 3.0), (9.0, 3.0), (9.0, 9.0), (15.0, 9.0), (15.0, 15.0)), *_poly("B", (3.0, 17.0), (3.0, 13.0), (10.0, 13.0), (10.0, 17.0))]


@pytest.mark.parametrize("style", [ARC, MITER45])
def test_the_pads_a_net_joins_are_the_same_before_and_after(tmp_path: Path, lib: KicadLibrary, style: str):
    """Rounding changes no connection: each net's longest pad-to-pad path (:mod:`ai_eda.tools.si.paths`, the arcs as the chords of
    their chains, which share their end points exactly) joins the same two pads before and after, shorter by what the corners lost;
    the ``pcb.routing`` checks find the copper connected and clear, the arcs again as their chords."""
    ir = board_ir(tmp_path, lib, JOINED_PARTS, JOINED_NETS, (20.0, 20.0))
    board = _Board(ir, lib, P)
    before = Routing(params=P, tracks=_joined(), stats={"net_length_mm": {"A": 24.0, "B": 15.0}, "total_length_mm": 39.0, "track_count": 7})
    after = Routing(params=P, tracks=_joined(), stats={"net_length_mm": {"A": 24.0, "B": 15.0}, "total_length_mm": 39.0, "track_count": 7})
    mitre_corners(after, board, style)
    c = after.stats["corners"]
    assert (c["found"], c["left"], c["skipped"]) == (5, 0, 0) and (c["arcs"] if style == ARC else c["mitred"]) == 5
    pads = net_pads(ir, lib)
    centres = {"A": [(3.0, 3.0), (15.0, 15.0)], "B": [(3.0, 17.0), (10.0, 17.0)]}
    for net, ends in (("A", ("R1.1", "R2.1")), ("B", ("R3.1", "R4.1"))):
        paths = []
        for r in (before, after):
            path, why = longest_path(net, straight_pieces([t for t in r.tracks if t.net == net]), [], pads.pads[net], None, {}, None)
            assert path is not None and path.ends == ends, (net, why)
            paths.append(path)
        assert paths[1].length_mm < paths[0].length_mm == before.stats["net_length_mm"][net]
        # the chords of a chain are at most 1 um inside the arc: the path through them is the net's own length within their bound
        assert paths[1].length_mm == pytest.approx(after.stats["net_length_mm"][net], abs=2e-3)
        assert _connected([t for t in after.tracks if t.net == net], centres[net])
    for r in (before, after):
        chords = Routing(params=P, tracks=straight_pieces(r.tracks), vias=[])
        checks = _routing_checks(ir, chords, lib, tmp_path)
        assert checks["pcb.routing.connectivity"].status == ValidationStatus.PASS and checks["pcb.routing.clearance"].status == ValidationStatus.PASS, checks


# --------------------------------------------------------------------------- the styles that existed


#: SHA-256 of :func:`tests.test_routing_rules.canonical` (copper with its provenance, stats, parameters) of ``"miter45"`` on the
#: fixtures of this file, produced by routing.corners 0.1 before the arc style existed (never regenerated)
MITER45_BEFORE_ARCS: dict[str, str] = {
    "l": "5a7efe0b1caddd188f871e440195e4499f3de276ee558ab2e47c54bef7df44aa",
    "u": "1fdea4b1fc4392f0bc31a5858cdda478f97da659b51efba4251dd89b68209349",
    "z": "387f496f702d3afc7b1969a658e6f297622b84f4f2365e5ea4affa0c7b48306b",
    "acute": "b9901a983af749881e91a048eaf65ea8252033378fe5ad7c1d20f43d1359b193",
    "jog": "e8f814dbc7dd7ada9f6571a6084c32da73d5a4467dce10560207695b73c9dd9f",
    "chamfer_beside_arc": "cd2b80e04cb61eb163b00391c5d01b6053310257daaa1de05212b5bc18752fa3",
    "bus_abc": "ccfd08b44d7f7c7c4315881a93cc2c2c2b6eab74cb31dbea54f5237bdf9389a4",
    "bus_cba": "f9388fc6997544afaea328d5b221fd9dc20b1398d9a94f92451c3b7515043ec8",
    "bus_at_the_limit": "c0e668b077d48a63c275e6caff3bba6c16a4b30c8035ba8809a1b5c45f4c236d",
    "foreign_via": "a5ff7e5706baf54e6fa6df75dc3083d173ef301dda2b9bb3c226c792bd42ea2d",
    "foreign_pad": "926a6b3c835fd9e2a7414c476283e2d5dc51ad75b3ba9bb69971cb48dfcd9630",
    "hole": "9c133e4b150405738e722fa131827e5c9432a2c041ff731ea23d7d6cc43bb2a3",
    "keepout": "72be8e2d81a08102baec3e157535b4f8686333e51da01942904b7489a210dfde",
    "own_via": "a2a60466289e86ff2f812668cf1ad65934e9ad1caa4aa0a671600c186e18b9d7",
    "own_track": "ad9f6f22063ff47738e47d94bca4e4a6be6d6eecc90cabbd904002632636ffe5",
    "own_pad_at_the_corner": "06f0e0d61409fa8dff36f6e137867f22a2bb08e81609e6ff6ec2f75479e78827",
    "own_pad_beside": "06f0e0d61409fa8dff36f6e137867f22a2bb08e81609e6ff6ec2f75479e78827",  # the same chamfer: the 45-degree style reads no contact
}


def _fixtures(tmp_path: Path, lib: KicadLibrary, style: str) -> dict[str, Routing]:
    """The hand-drawn fixtures of this file, each through the pass in ``style``."""
    via_b = Via(net="B", x_mm=9.0, y_mm=3.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    via_a = Via(net="A", x_mm=9.3, y_mm=2.7, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    r5 = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]}
    b5 = {"A": NETS["A"], "B": [("R3", "1"), ("R5", "1")]}
    far = (12.0 - 8.0 * math.cos(math.radians(60.0)), 2.0 + 8.0 * math.sin(math.radians(60.0)))
    cases: dict[str, tuple[list[Track], dict]] = {
        "l": (_poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0)), {}),
        "u": (_poly("A", (4.0, 10.0), (4.0, 4.0), (7.0, 4.0), (7.0, 10.0)), {}),
        "z": (_poly("A", (2.0, 4.0), (6.0, 4.0), (6.0, 7.0), (10.0, 7.0)), {}),
        "acute": (_poly("A", (2.0, 2.0), (12.0, 2.0), far), {}),
        "jog": (_poly("A", (9.5, 2.0), (10.0, 2.0), (10.0, 8.0)), {}),
        "chamfer_beside_arc": (_poly("A", (2.0, 4.0), (6.0, 4.0), (6.0, 5.6), (9.0, 2.6)), {}),
        "bus_abc": (_bus(0.75, "ABC"), {"parts": BUS_PARTS, "nets": BUS_NETS}),
        "bus_cba": (_bus(0.75, "CBA"), {"parts": BUS_PARTS, "nets": BUS_NETS}),
        "bus_at_the_limit": (_bus(0.65, "CBA"), {"parts": BUS_PARTS, "nets": BUS_NETS}),
        "foreign_via": (_poly("A", *L_SHAPE), {"vias": (via_b,)}),
        "foreign_pad": (_poly("A", (6.0, 2.0), (10.0, 2.0), (10.0, 6.0), width=1.0),
                        {"parts": [*PARTS, ("R5", "SMD1", 8.75, 3.25)], "nets": b5, "rules": {"A": NetRule(width_mm=1.0, clearance_mm=0.25)}}),
        "hole": (_poly("A", *L_SHAPE), {"parts": [*PARTS, ("W1", "VBAR", 9.3, 4.45)]}),
        "keepout": (_poly("A", *L_SHAPE), {"keepouts": [keepout("K1", [6.8, 2.2, 3.0, 3.0], forbids=["tracks"], layers=["F.Cu"])]}),
        "own_via": (_poly("A", *L_SHAPE), {"vias": (via_a,)}),
        "own_track": ([*_poly("A", *L_SHAPE), *_poly("A", (6.0, 6.0), (9.95, 2.05))], {}),
        "own_pad_at_the_corner": (_poly("A", *L_SHAPE), {"parts": [*PARTS, ("R5", "SMD1", 10.6, 1.4)], "nets": r5}),
        "own_pad_beside": (_poly("A", *L_SHAPE), {"parts": [*PARTS, ("R5", "SMD1", 9.0, 1.3)], "nets": r5}),
    }
    return {name: _pass(tmp_path / style / name, lib, tracks, style=style, **extra) for name, (tracks, extra) in cases.items()}


def test_miter45_gives_what_it_gave_before_the_arc_style_and_orthogonal_nothing(tmp_path: Path, lib: KicadLibrary):
    """The version is chosen by the style: ``"miter45"`` is routing.corners 0.1 byte for byte - copper, entries, notes, stats (no arc
    key, no arc track) - on every fixture of this file (:data:`MITER45_BEFORE_ARCS`, and the ends ``tests/test_routing_corners.py``
    pins); ``"orthogonal"`` leaves a result untouched."""
    results = _fixtures(tmp_path, lib, MITER45)
    assert {name: hashlib.sha256(canonical(r)).hexdigest() for name, r in results.items()} == MITER45_BEFORE_ARCS
    for name, r in results.items():
        c = r.stats["corners"]
        assert (c["version"], c["style"]) == (CORNERS_VERSION, MITER45) == ("0.1", "miter45") and "arcs" not in c and "arc" not in c, name
        assert not _arcs(r) and all("no_arc_reason" not in row for row in c["corners"]), name
        assert all(e.startswith("corner:routing.corners 0.1:miter45:at=") for t in r.tracks for e in t.provenance.derived_from), name
    assert _shape(results["l"]) == [((4.0, 4.0), (10.0, 4.0)), ((10.0, 4.0), (12.0, 6.0)), ((12.0, 6.0), (12.0, 8.0))]
    assert _shape(results["u"]) == [((4.0, 10.0), (4.0, 5.0)), ((4.0, 5.0), (5.0, 4.0)), ((5.0, 4.0), (6.0, 4.0)), ((6.0, 4.0), (7.0, 5.0)), ((7.0, 5.0), (7.0, 10.0))]
    assert _shape(results["z"]) == [((2.0, 4.0), (4.5, 4.0)), ((4.5, 4.0), (6.0, 5.5)), ((6.0, 5.5), (7.5, 7.0)), ((7.5, 7.0), (10.0, 7.0))]
    tracks = _poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0))
    untouched = _pass(tmp_path / "orthogonal", lib, tracks, style=ORTHOGONAL)
    assert all(t is u for t, u in zip(untouched.tracks, tracks)) and "corners" not in untouched.stats
    assert untouched.stats == {"net_length_mm": {"A": 12.0}, "total_length_mm": 12.0, "track_count": 2}


def test_the_arc_style_is_deterministic_tangent_and_complete_on_every_fixture(tmp_path: Path, lib: KicadLibrary):
    """Two runs of the arc style give the same bytes on every fixture of this file; every arc continues the tracks it joins; the
    counts add up; a rounded or mitred corner shortens its net (a length budget still holds); and the copper joins the points the
    pass kept - every track end that is still one - exactly as it joined them before (no connection made, none cut)."""
    first, second = _fixtures(tmp_path / "1", lib, ARC), _fixtures(tmp_path / "2", lib, ARC)
    before = _fixtures(tmp_path / "0", lib, ORTHOGONAL)
    for name, r in first.items():
        assert canonical(r) == canonical(second[name]), name
        c = r.stats["corners"]
        assert c["found"] == c["arcs"] + c["mitred"] + c["left"] + c["skipped"] and c["arcs"] == len(_arcs(r)), name
        for net in {t.net for t in r.tracks}:
            _tangent_everywhere(r, net)
            assert r.stats["net_length_mm"][net] <= before[name].stats["net_length_mm"][net], (name, net)
        kept = _track_ends(before[name].tracks) & _track_ends(r.tracks)
        assert kept and _grouping(r.tracks, kept) == _grouping(before[name].tracks, kept), name
    assert {name: r.stats["corners"]["arcs"] for name, r in first.items() if r.stats["corners"]["arcs"]} == {
        "l": 1, "u": 2, "z": 2, "acute": 1, "chamfer_beside_arc": 1, "bus_abc": 3, "bus_cba": 3, "bus_at_the_limit": 1, "foreign_via": 1, "own_via": 1,
        "own_pad_beside": 1}


def test_a_result_that_already_holds_an_arc_is_refused_under_every_style(tmp_path: Path, lib: KicadLibrary):
    """The pass runs once, on the router's own copper, which is straight (no router code draws an arc): a track with ``mid`` at entry
    is refused by name, under every style - also one that would leave the copper as it is."""
    ir = board_ir(tmp_path, lib, PARTS, NETS, (20.0, 20.0))
    board = _Board(ir, lib, P)
    rounded = _pass(tmp_path / "once", lib, _poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0)))
    assert len(_arcs(rounded)) == 1
    for style in CORNER_STYLES:
        again = Routing(params=P, tracks=list(rounded.tracks), stats={"track_count": 3})
        with pytest.raises(CompileError, match=r"routing\.corners: track 1 of net 'A' on F\.Cu, from \(10\.4, 4\) through \(11\.5314, 4\.46863\) to "
                                               r"\(12, 5\.6\), is already an arc - the corner pass runs once, on the router's own copper"):
            mitre_corners(again, board, style)
        assert again.tracks == rounded.tracks and "corners" not in again.stats


# --------------------------------------------------------------------------- the router and the PCB agent


def test_the_styles_are_arc_miter45_and_orthogonal_and_the_agents_default_is_arc():
    assert CORNER_STYLES == ("arc", "miter45", "orthogonal") == (ARC, MITER45, ORTHOGONAL) and DEFAULT_CORNER_STYLE == ARC
    assert inspect.signature(route_board).parameters["corner_style"].default == ORTHOGONAL  # the router alone keeps its grid corners
    assert (CORNERS_VERSION, CORNERS_ARC_VERSION) == ("0.1", "0.2") and ARC_RADIUS_WIDTHS == (4.0, 2.0, 1.0)
    for style in CORNER_STYLES:
        check_style(style)
    with pytest.raises(CompileError, match=r"corner_style 'round' is not one of \['arc', 'miter45', 'orthogonal'\] \(routing\.corners: 'arc' is "
                                           r"version 0\.2, 'miter45' version 0\.1\)"):
        check_style("round")


def test_the_router_rounds_on_request_deterministically(tmp_path: Path, lib: KicadLibrary):
    """``route_board(..., corner_style="arc")``: the routed L (runs of 6 and 5 mm) gets one arc of the cap radius; the router's own
    provenance stays (tool, version, every entry) and the corner entry follows; the copper joins the two pads and passes the
    ``pcb.routing`` checks (the arc as the chords of its chain); two runs give the same bytes."""
    ir = _l_board(tmp_path, lib)
    plain = route_board(ir, lib)
    a, b = route_board(ir, lib, corner_style=ARC), route_board(ir, lib, corner_style=ARC)
    assert canonical(a) == canonical(b) and canonical(route_board(ir, lib, corner_style=ORTHOGONAL)) == canonical(plain)
    assert [(t.start, t.end) for t in plain.tracks] == [((3.0, 3.0), (9.0, 3.0)), ((9.0, 3.0), (9.0, 8.0))]
    assert _shape(a) == [((3.0, 3.0), (7.4, 3.0)), ((7.4, 3.0), (8.531371, 3.468629), (9.0, 4.6)), ((9.0, 4.6), (9.0, 8.0))]
    for t in a.tracks:
        assert t.provenance.tool == "routing.maze" and t.provenance.tool_version == plain.version
        assert t.provenance.derived_from[:-1] == plain.tracks[0].provenance.derived_from
        assert t.provenance.derived_from[-1] == "corner:routing.corners 0.2:arc:at=9.0,3.0:tangent=1.6:radius=1.6"
    c = a.stats["corners"]
    assert (c["version"], c["style"], c["found"], c["arcs"], c["mitred"], c["left"], c["skipped"]) == ("0.2", "arc", 1, 1, 0, 0, 0)
    mitred = route_board(ir, lib, corner_style=MITER45)
    assert mitred.stats["net_length_mm"]["A"] < a.stats["net_length_mm"]["A"] < plain.stats["net_length_mm"]["A"]  # the 0.1 chamfer is larger
    assert a.stats["net_length_mm"]["A"] == round(sum(track_length(t) for t in a.tracks), 6) == pytest.approx(4.4 + CAP * math.pi / 2.0 + 3.4, abs=2e-6)
    assert _connected(a.tracks, [(3.0, 3.0), (9.0, 8.0)])
    _tangent_everywhere(a)
    chords = Routing(params=a.params, tracks=straight_pieces(a.tracks), vias=list(a.vias))
    checks = _routing_checks(ir, chords, lib, tmp_path)
    assert checks["pcb.routing.clearance"].status == ValidationStatus.PASS and checks["pcb.routing.connectivity"].status == ValidationStatus.PASS, checks


def test_the_pcb_agent_rounds_by_default_and_on_the_arc_answer_and_mitres_on_request(tmp_path: Path, lib: KicadLibrary):
    """The agent's default style is the arc: without an answer, as with ``pcb.corners=arc``, it gives ``route_board(...,
    corner_style="arc")``'s copper, the arc with its ``mid``, and the routing description counts the corners rounded and the corners
    mitred; ``pcb.corners=miter45`` gives the 45-degree corners of routing.corners 0.1, and an answer it does not understand names the
    three styles and rounds as by default."""
    ir = _l_board(tmp_path, lib)
    unrouted = ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"tracks": [], "vias": []})})
    rounded = route_board(unrouted, lib, corner_style=ARC)
    by_default = PCBAgent().run(ir, _ctx(tmp_path, lib))
    for res in (by_default, PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: " Arc"}))):
        payload = res.proposals[0].payload
        assert [(t.start, t.mid, t.end) for t in payload.tracks] == [(t.start, t.mid, t.end) for t in rounded.tracks]
        assert [t.mid for t in payload.tracks] == [None, (8.531371, 3.468629), None]
        assert any("corners: 1 of 1 corner(s) of 90 degrees or sharper rounded into tangent arcs and 0 mitred into 45-degree chamfers where no arc "
                   "fits (routing.corners 0.2, the project owner's corner-style rule - a geometry convention, not an electrical requirement), 0 "
                   "left as routed, 0 skipped" in n for n in res.notes), res.notes
        assert not any("not understood" in n for n in res.notes)
        assert res.notes == by_default.notes
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: "miter45"}))
    mitred = route_board(unrouted, lib, corner_style=MITER45)
    assert [(t.start, t.mid, t.end) for t in res.proposals[0].payload.tracks] == [(t.start, t.mid, t.end) for t in mitred.tracks]
    assert all(t.mid is None for t in res.proposals[0].payload.tracks)
    assert any("corners: 1 of 1 corner(s) of 90 degrees or sharper mitred into 45-degree chamfers (routing.corners 0.1" in n for n in res.notes), res.notes
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: "rounded"}))
    assert any(n == "pcb.corners='rounded' not understood (the only answers are 'arc', 'miter45' and 'orthogonal'); corners are rounded as usual "
               "(arc)" for n in res.notes), res.notes
    assert [(t.start, t.mid, t.end) for t in res.proposals[0].payload.tracks] == [(t.start, t.mid, t.end) for t in rounded.tracks]
