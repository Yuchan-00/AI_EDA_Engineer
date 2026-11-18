"""Arc tracks (``Track.mid``) in the report figures, the stage reports, the GUI stylesheet, the 3D preview and the layout metrics.

Hand-built boards on the synthetic libraries of the neighbouring test modules, no KiCad. What is proved:

* the board figure draws an arc track as one ``<path class="track">`` with a straight track's classes, data attributes and
  stroke, and the path is the arc: the SVG arc is evaluated here, by the endpoint-to-centre conversion of the SVG specification,
  and compared with the true arc both ways (it passes the stored ``mid``; the chord between the ends is not what is drawn), for
  both directions, a mirrored drawing, a half turn, more than a half turn, an arc that nearly closes and one too flat to tell from a
  line; an arc starts and ends on the very coordinates of the straight tracks it joins; a board without an arc draws lines only,
  and a straight track keeps its markup letter for letter, beside an arc too;
* every length a report prints counts an arc by its arc length: the figure's caption, the per-net table and its total, the
  longest net, the copper bars;
* the keep-out section reads both corner records of ``routing.corners``: the 0.1 sentence is the one it always was, a 0.2 board
  gets the count of rounded corners with their radii and of the corners mitred where no arc went in, and an entry in another
  form is counted as unread;
* the GUI's net-class rules address an arc's ``<path>`` exactly as a straight track's ``<line>``;
* the 3D scene draws one round-ended bar per chord of the arc's chain, on the arc's own side of the board at a straight track's
  height, and says so in its notes;
* the layout metric ``routed_track_length_mm`` counts the arc length, and a frozen baseline keeps its number;
* an arc broken after the model checked it is refused by name, never drawn or measured as its chord.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ai_eda.report.stages as stages
from ai_eda.gui.preview import board_svg
from ai_eda.ir import (
    ARC_SAGITTA_MM, BoardOutline, CircuitIR, Layer, Net, PCBDesign, PinRef, Placement, Provenance, ProvenanceKind, Track,
    arc_length, arc_point_at, arc_problem, fillet, point_arc_distance, track_points,
)
from ai_eda.report.figures import _ARC_FLAT_PX, CLASS_COLOURS, COPPER_COLOURS, NET_CLASS_DEFAULT, _arc_path, board_figure
from ai_eda.report.stages import circuit_report, net_routing_stats
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.layout_metrics import pcb_metrics
from ai_eda.tools.model3d import SceneError, build_scene, write_glb
from ai_eda.tools.model3d.scene import SCENE_VERSION
from tests.layout_baseline_data import DATA, load_geometry
from tests.test_model3d import USER, _ir as scene_ir, _library as scene_library, _only, _place
from tests.test_parts_existence import synthetic_library
from tests.test_pcb_agent import parts_ir
from tests.test_report_figures import _board, _by_class, _parse
from tests.test_routing import fixture_library
from tests.test_si_checks import default_classes, long_board, si_of

SVG_NS = "{http://www.w3.org/2000/svg}"
#: the figures write every coordinate to 0.01 px: a drawn arc and the true one are nearer than that
PX = 0.01
Point = tuple[float, float]

# --------------------------------------------------------------------------- an SVG path arc, evaluated here


def _svg_arc(x1: float, y1: float, r: float, large: int, sweep: int, x2: float, y2: float) -> tuple[float, float, float, float, float]:
    """Centre, radius, start angle and signed sweep of the SVG path arc ``A r r 0 large sweep x2 y2`` drawn from ``(x1, y1)``.

    The endpoint-to-centre conversion of the SVG specification (implementation notes on elliptical arcs) for equal radii and no
    rotation, written out here so the drawing is not checked with its own geometry: a radius too small for the chord is raised to
    half of it; the centre lies on the chord's perpendicular bisector, to the right of the way from start to end as seen on screen
    (y down) when the two flags differ, to the left when they are equal; a set sweep flag runs toward larger angles.
    """
    hx, hy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    assert hx or hy, "identical end points: SVG draws no arc at all"
    r = max(r, math.hypot(hx, hy))
    k = math.sqrt(max(0.0, r * r / (hx * hx + hy * hy) - 1.0)) * (-1.0 if large == sweep else 1.0)
    cx, cy = k * hy + (x1 + x2) / 2.0, -k * hx + (y1 + y2) / 2.0
    t1 = math.atan2(y1 - cy, x1 - cx)
    dt = (math.atan2(y2 - cy, x2 - cx) - t1) % (2.0 * math.pi)
    return cx, cy, r, t1, dt if sweep else dt - 2.0 * math.pi


def _path(d: str) -> list[tuple]:
    """The pieces of a path of ``M`` / ``A`` / ``L`` commands: ``("L", from, to)`` or ``("A", from, to, cx, cy, r, t1, dt)``."""
    tok = d.split()
    assert tok[0] == "M"
    here = (float(tok[1]), float(tok[2]))
    out: list[tuple] = []
    i = 3
    while i < len(tok):
        if tok[i] == "L":
            there = (float(tok[i + 1]), float(tok[i + 2]))
            out.append(("L", here, there))
            i += 3
        else:
            assert tok[i] == "A" and tok[i + 1] == tok[i + 2] and tok[i + 3] == "0" and tok[i + 4] in ("0", "1") and tok[i + 5] in ("0", "1"), d
            there = (float(tok[i + 6]), float(tok[i + 7]))
            out.append(("A", here, there, *_svg_arc(*here, float(tok[i + 1]), int(tok[i + 4]), int(tok[i + 5]), *there)))
            i += 8
        here = there
    return out


def _to_path(p: Point, pieces: list[tuple]) -> float:
    """The distance of ``p`` from the drawn path."""
    best = math.inf
    for piece in pieces:
        a, b = piece[1], piece[2]
        if piece[0] == "L":
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = 0.0 if not (dx or dy) else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
            best = min(best, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))
            continue
        cx, cy, r, t1, dt = piece[3:]
        along = ((math.atan2(p[1] - cy, p[0] - cx) - t1) * (1.0 if dt > 0 else -1.0)) % (2.0 * math.pi)
        if along <= abs(dt):
            best = min(best, abs(math.hypot(p[0] - cx, p[1] - cy) - r))
        else:
            best = min(best, math.hypot(p[0] - a[0], p[1] - a[1]), math.hypot(p[0] - b[0], p[1] - b[1]))
    return best


def _path_points(pieces: list[tuple], n: int = 64) -> list[Point]:
    out: list[Point] = []
    for piece in pieces:
        if piece[0] == "L":
            (ax, ay), (bx, by) = piece[1], piece[2]
            out += [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n + 1)]
        else:
            cx, cy, r, t1, dt = piece[3:]
            out += [(cx + r * math.cos(t1 + dt * k / n), cy + r * math.sin(t1 + dt * k / n)) for k in range(n + 1)]
    return out


def _apart(d: str, start: Point, mid: Point, end: Point, to_px) -> float:
    """How far the drawn path and the true arc are apart, in px: the largest distance of a point of the arc from the path, and of a
    point of the path from the arc (the exact point-to-arc distance, on the three points as drawn)."""
    pieces = _path(d)
    from_arc = max(_to_path(to_px(arc_point_at(start, mid, end, k / 128.0)), pieces) for k in range(129))
    from_path = max(point_arc_distance(q, to_px(start), to_px(mid), to_px(end)) for q in _path_points(pieces))
    return max(from_arc, from_path)


def _figure_map(root: ET.Element, ir: CircuitIR):
    """The figure's own map from a board point (mm) to its coordinates, read from the outline rectangle it drew."""
    (rect,) = [e for e in root.iter(f"{SVG_NS}rect") if e.get("class") == "outline"]
    o = ir.pcb.outline
    scale = float(rect.get("width")) / o.width_mm
    x0, y0 = float(rect.get("x")), float(rect.get("y"))
    return lambda p: (x0 + (p[0] - o.origin_x_mm) * scale, y0 + (p[1] - o.origin_y_mm) * scale)


def _plain(p: Point) -> Point:
    """A figure of 18 px per mm with a 40 px margin."""
    return (40.0 + 18.0 * p[0], 40.0 + 18.0 * p[1])


def _mirrored(p: Point) -> Point:
    """The same figure seen from the other side: x runs the other way."""
    return (700.0 - 18.0 * p[0], 40.0 + 18.0 * p[1])


def _arc(cx: float, cy: float, r: float, a0_deg: float, sweep_deg: float, mid_at: float = 0.5) -> tuple[Point, Point, Point]:
    """``start``, ``mid`` and ``end`` of an arc given by its circle, at the 1 nm a stored arc has (``mid`` at ``mid_at`` of the sweep)."""
    def at(f: float) -> Point:
        a = math.radians(a0_deg + sweep_deg * f)
        return (round(cx + r * math.cos(a), 6), round(cy + r * math.sin(a), 6))

    points = (at(0.0), at(mid_at), at(1.0))
    assert arc_problem(*points) is None
    return points


# --------------------------------------------------------------------------- the board figure

#: the corner of a VIN track that runs right along y = 6 and turns down at x = 14, rounded with a tangent length of 2 mm (radius 2 mm)
CORNER, TANGENT = (14.0, 6.0), 2.0
PARAMS = ("params:grid=0.25,width=0.25,clearance=0.25,via=0.8/0.4,edge=0.3,via_cost=12.0,bend_cost=0.6,base_cost=1.0,history_cost=1.0,"
          "present_cost=0.5,present_growth=2.0,max_iterations=40,window=10.0")


def _routed(net: str, *entries: str) -> Provenance:
    return Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.2", derived_from=[f"net:{net}", PARAMS, *entries], note="routed")


def _arc_board(tmp_path: Path, *, reverse: bool = False) -> tuple[CircuitIR, KicadLibrary, Track]:
    """The divider board with VIN routed round one rounded corner (straight 7 mm, a quarter arc of radius 2 mm, straight 2 mm) and a
    straight VOUT track; ``reverse`` stores the arc from its other end. The tracks carry the router's provenance with the corner's
    ``routing.corners`` 0.2 entry."""
    ir, lib = _board(tmp_path)
    a, mid, b, radius = fillet(CORNER, (5.0, 6.0), (14.0, 10.0), TANGENT)
    assert (a, b) == ((12.0, 6.0), (14.0, 8.0)) and radius == pytest.approx(2.0)
    vin = _routed("VIN", f"corner:routing.corners 0.2:arc:at=14.0,6.0:tangent=2.0:radius={round(radius, 6)!r}")
    start, end = (b, a) if reverse else (a, b)
    arc = Track(net="VIN", layer="F.Cu", start=start, mid=mid, end=end, width_mm=0.25, provenance=vin)
    ir.pcb.tracks = [
        Track(net="VIN", layer="F.Cu", start=(5.0, 6.0), end=a, width_mm=0.25, provenance=vin),
        arc,
        Track(net="VIN", layer="F.Cu", start=b, end=(14.0, 10.0), width_mm=0.25, provenance=vin),
        Track(net="VOUT", layer="B.Cu", start=(5.0, 8.54), end=(10.0, 8.54), width_mm=0.3, provenance=_routed("VOUT")),
    ]
    return ir, lib, arc


@pytest.mark.parametrize("reverse", [False, True])
def test_the_board_figure_draws_an_arc_as_one_path_through_its_mid_point(tmp_path: Path, reverse: bool):
    ir, lib, arc = _arc_board(tmp_path, reverse=reverse)
    before = ir.model_dump_json()
    fig = board_figure(ir, lib)
    assert ir.model_dump_json() == before
    root = _parse(fig.svg)
    tracks = _by_class(root, "track")
    assert len(tracks) == len(ir.pcb.tracks) == 4  # one element per track, arc or not
    assert [e.tag.removeprefix(SVG_NS) for e in tracks] == ["line", "line", "path", "line"]  # B.Cu first, then F.Cu in track order
    first, path, last = tracks[1:]
    # a straight track's classes, data attributes and stroke; never filled, and nothing the GUI's SVG import would strip
    for name in ("class", "data-net", "data-layer", "stroke", "stroke-width", "stroke-linecap"):
        assert path.get(name) == first.get(name), name
    assert (path.get("data-net"), path.get("data-layer"), path.get("stroke"), path.get("stroke-width")) == ("VIN", "F.Cu", COPPER_COLOURS["F.Cu"], "4.50")
    assert path.get("fill") == "none" and path.get("style") is None
    (group,) = [g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "layer-F_Cu"]
    assert path in list(group)  # inside its layer group: the layer toggle hides it with the rest
    # one arc command: it starts and ends on the very coordinates the neighbouring lines end on
    d = path.get("d")
    tok = d.split()
    assert len(tok) == 11 and tok[0] == "M" and tok[3] == "A" and tok[4] == tok[5] and tok[6] == "0"
    joint_a, joint_b = (first.get("x2"), first.get("y2")), (last.get("x1"), last.get("y1"))
    assert ((tok[1], tok[2]), (tok[9], tok[10])) == ((joint_b, joint_a) if reverse else (joint_a, joint_b))
    assert float(tok[4]) == pytest.approx(2.0 * 18.0, abs=1e-3)  # the radius to scale
    # a quarter turn, clockwise on screen from the horizontal track to the vertical one: the sweep flag is set that way round only
    assert (tok[7], tok[8]) == ("0", "0" if reverse else "1")
    to_px = _figure_map(root, ir)
    pieces = _path(d)
    assert _to_path(to_px(arc.mid), pieces) < PX  # the drawn arc passes the stored mid point
    assert _apart(d, arc.start, arc.mid, arc.end, to_px) < PX  # and is the arc, all along
    # not the chord: the middle of the straight line between the ends lies r (1 - cos 45 deg) = 0.586 mm inside the drawn arc
    chord_middle = ((arc.start[0] + arc.end[0]) / 2.0, (arc.start[1] + arc.end[1]) / 2.0)
    assert _to_path(to_px(chord_middle), pieces) == pytest.approx((2.0 - math.sqrt(2.0)) * 18.0, abs=PX)
    assert fig.svg == board_figure(ir, lib).svg  # deterministic
    # a wider board is drawn at a smaller scale, and the arc with it
    ir.pcb.outline.width_mm = 200.0
    small = _parse(board_figure(ir, lib).svg)
    (path,) = [e for e in _by_class(small, "track") if e.tag == f"{SVG_NS}path"]
    to_px = _figure_map(small, ir)
    scale = to_px((1.0, 0.0))[0] - to_px((0.0, 0.0))[0]
    assert scale < 3.0 and float(path.get("d").split()[4]) == pytest.approx(2.0 * scale, abs=1e-3)
    assert _apart(path.get("d"), arc.start, arc.mid, arc.end, to_px) < PX


def test_a_board_without_an_arc_draws_lines_only(tmp_path: Path):
    ir, lib = _board(tmp_path)
    assert not any(t.is_arc for t in ir.pcb.tracks)
    svg = board_figure(ir, lib).svg
    tracks = _by_class(_parse(svg), "track")
    assert len(tracks) == len(ir.pcb.tracks) and {e.tag for e in tracks} == {f"{SVG_NS}line"}
    assert '<path class="track"' not in svg and '<path class="track"' not in board_figure(ir, lib, copper=False).svg


#: the two tracks of that board as the figure wrote a straight track before a track could be an arc, letter for letter
LINE_VOUT = '<line class="track" data-net="VOUT" data-layer="B.Cu" x1="130.00" y1="193.72" x2="220.00" y2="193.72" stroke="#2a78d6" stroke-width="5.40" stroke-linecap="round"/>'
LINE_VIN = '<line class="track" data-net="VIN" data-layer="F.Cu" x1="130.00" y1="148.00" x2="292.00" y2="148.00" stroke="#e34948" stroke-width="4.50" stroke-linecap="round"/>'


def test_a_straight_track_keeps_its_markup_letter_for_letter_beside_an_arc_too(tmp_path: Path):
    ir, lib = _board(tmp_path / "square")
    svg = board_figure(ir, lib).svg
    assert svg.count('class="track"') == 2 and LINE_VOUT in svg and LINE_VIN in svg
    # on the board with the rounded corner VOUT is the same track, and VIN's first leg ends where the arc starts (x = 12 mm: 256 px)
    arc_ir, arc_lib, _arc_track = _arc_board(tmp_path / "rounded")
    arc_svg = board_figure(arc_ir, arc_lib).svg
    assert arc_svg.count('<line class="track"') == 3 and arc_svg.count('<path class="track"') == 1
    assert LINE_VOUT in arc_svg and LINE_VIN.replace('x2="292.00"', 'x2="256.00"') in arc_svg


def test_an_arc_ends_on_the_very_coordinates_of_the_straight_tracks_it_joins(tmp_path: Path):
    # the arc's end (6.0225, 7.6) maps onto a rounding boundary of the figure's 0.01 px (40 + 18 x 6.0225 = 148.405, written 148.41): the
    # path's end is written from the stored point, as the straight track that starts there is - the same point taken from the arc's own
    # circle (centre + radius at the end angle) would be written 148.40, 0.01 px off the track it joins
    ir, lib = _board(tmp_path)
    a, mid, b, _radius = fillet((6.0225, 6.0), (1.0, 6.0), (6.0225, 10.0), 1.6)
    assert (a, b) == ((4.4225, 6.0), (6.0225, 7.6))
    ir.pcb.tracks = [
        Track(net="VIN", layer="F.Cu", start=(1.0, 6.0), end=a, width_mm=0.25),
        Track(net="VIN", layer="F.Cu", start=a, mid=mid, end=b, width_mm=0.25),
        Track(net="VIN", layer="F.Cu", start=b, end=(6.0225, 10.0), width_mm=0.25),
    ]
    first, path, last = _by_class(_parse(board_figure(ir, lib).svg), "track")
    tok = path.get("d").split()
    assert (tok[1], tok[2]) == (first.get("x2"), first.get("y2")) == ("119.61", "148.00")
    assert (tok[-2], tok[-1]) == (last.get("x1"), last.get("y1")) == ("148.41", "176.80")


def test_the_arc_flags_are_read_from_the_drawing_so_a_mirror_turns_the_arc():
    start, mid, end = _arc(12.0, 8.0, 2.0, -90.0, 90.0)  # the corner of the board above: positive sweep, clockwise on screen
    plain, mirrored = _arc_path(start, mid, end, _plain), _arc_path(start, mid, end, _mirrored)
    assert plain.split()[7:9] == ["0", "1"] and mirrored.split()[7:9] == ["0", "0"]
    for d, to_px in ((plain, _plain), (mirrored, _mirrored)):
        assert _to_path(to_px(mid), _path(d)) < PX and _apart(d, start, mid, end, to_px) < PX
    # the board's own direction would bulge to the other side of the chord in the mirrored drawing: nowhere nearer the mid point
    # than the chord is (the arc's rise, 10.5 px)
    wrong = mirrored.split()
    wrong[8] = "1"
    assert _to_path(_mirrored(mid), _path(" ".join(wrong))) > 2.0 * 18.0 * (1.0 - math.cos(math.pi / 4.0))


@pytest.mark.parametrize(
    ("arc", "commands"),
    [
        (_arc(12.0, 8.0, 2.0, 0.0, -90.0), "A"),  # a quarter turn the other way
        (_arc(12.0, 8.0, 2.0, 30.0, 120.0, mid_at=0.1), "A"),  # the stored mid need not be the middle
        (_arc(12.0, 5.0, 2.0, 180.0, 180.0), "A"),  # a half turn: its middle is the most sensitive to the radius
        (_arc(12.0, 5.0, 2.0, 0.0, -180.0), "A"),
        (_arc(12.0, 5.0, 5.5, 90.0, 180.0), "A"),  # a large half turn
        # oblique half turns of radius 270 px and 720 px (wider than a figure): nine digits of the radius would miss by 0.016 and 0.022 px
        (((10.020557, 24.214961), (25.785039, 10.020557), (39.979443, 25.785039)), "A"),
        (_arc(50.0, 50.0, 40.0, 59.0, 180.0), "A"),
        (_arc(12.0, 5.0, 5.5, 200.0, 181.0), "AA"),  # just beyond a half turn: two arcs through the middle
        (_arc(12.0, 5.0, 2.0, 0.0, 270.0), "AA"),
        (_arc(12.0, 5.0, 2.0, 45.0, -200.0), "AA"),
        (_arc(12.0, 5.0, 2.0, 0.0, 359.9), "AA"),  # nearly closed: the ends are 3.5 um apart
        (_arc(12.0, 5.0, 2.0, 10.0, -359.99, mid_at=0.3), "AA"),
        (_arc(12.0, 5.0, 0.05, 0.0, 90.0), "A"),  # a radius under one px
        (_arc(12.0, 5.0, 0.01, 0.0, 300.0), "AA"),
    ],
)
def test_an_arc_of_any_sweep_is_drawn_true(arc, commands: str):
    start, mid, end = arc
    for to_px in (_plain, _mirrored):
        d = _arc_path(start, mid, end, to_px)
        assert "".join(t for t in d.split() if t in ("A", "L")) == commands, d
        tok = d.split()  # the ends as a straight track's are written: the stored points at the figure's 0.01 px
        assert (tok[1], tok[2]) == tuple(f"{v:.2f}" for v in to_px(start)) and (tok[-2], tok[-1]) == tuple(f"{v:.2f}" for v in to_px(end)), d
        if commands == "AA":  # beyond a half turn: two arcs that meet in the arc's middle
            assert (tok[9], tok[10]) == tuple(f"{v:.2f}" for v in to_px(arc_point_at(start, mid, end, 0.5))), d
        assert _to_path(to_px(mid), _path(d)) < PX and _apart(d, start, mid, end, to_px) < PX


def test_a_half_turn_drawn_through_ends_written_nearer_than_its_diameter_sets_the_large_arc_flag():
    # a half turn of radius 125.93 px whose two ends are written 0.009 px nearer to each other than they lie: the circle through the
    # written ends and the arc's middle has its centre 0.009 px on the middle's side of the chord, so the drawn arc sweeps a little
    # more than a half turn and the path must say so (large-arc flag 1). With the flag cleared the same radius is the other arc
    # through the two ends - the one that stays short of a half turn - and passes the middle 0.019 px away.
    start, mid, end = (22.555834, -1.151894), (15.559478, 5.844462), (22.555834, 12.840818)
    d = _arc_path(start, mid, end, _plain)
    tok = d.split()
    assert len(tok) == 11 and tok[3] == "A" and (tok[7], tok[8]) == ("1", "0")
    assert _to_path(_plain(mid), _path(d)) < PX and _apart(d, start, mid, end, _plain) < PX
    cleared = " ".join([*tok[:7], "0", *tok[8:]])
    assert _to_path(_plain(mid), _path(cleared)) == pytest.approx(0.019, abs=0.002)
    # the same arc seen from the other side: the flag stays (the circle is the mirrored one), the direction turns
    mirrored = _arc_path(start, mid, end, _mirrored).split()
    assert (mirrored[7], mirrored[8]) == ("1", "1") and _to_path(_mirrored(mid), _path(" ".join(mirrored))) < PX


def test_an_arc_too_flat_or_small_for_the_figure_is_the_line_it_is_there():
    # a chord of 1 mm on a radius of 500 mm rises 0.25 um = 0.0045 px: below the figure's resolution, a straight line
    flat = _arc(12.0, 505.0, 500.0, -90.0, 0.11459156)
    d = _arc_path(*flat, _plain)
    assert d.split()[3] == "L" and len(d.split()) == 6
    assert _apart(d, *flat, _plain) < _ARC_FLAT_PX + PX
    # the same arc at twenty times the scale is an arc again
    assert _arc_path(*flat, lambda p: (360.0 * p[0], 360.0 * p[1])).split()[3] == "A"
    # a loop of radius 0.5 um is a dot at this scale: two strokes shorter than the figure's resolution, under one round cap
    dot = _arc(12.0, 5.0, 0.0005, 0.0, 300.0)
    d = _arc_path(*dot, _plain)
    assert "".join(t for t in d.split() if t in ("A", "L")) == "LL" and _apart(d, *dot, _plain) < _ARC_FLAT_PX + PX


# --------------------------------------------------------------------------- lengths


def test_every_length_counts_the_arc_by_its_arc_length(tmp_path: Path):
    ir, lib, arc = _arc_board(tmp_path)
    quarter = arc_length(arc.start, arc.mid, arc.end)
    assert quarter == pytest.approx(math.pi, abs=1e-5) and arc.length_mm == quarter  # radius 2 mm: pi mm; its chord is 2.83 mm
    vin, total = 7.0 + quarter + 2.0, 7.0 + quarter + 2.0 + 5.0
    stats = {s["net"]: s for s in net_routing_stats(ir)}
    assert stats["VIN"]["segments"] == 3 and stats["VIN"]["length_mm"] == pytest.approx(vin, abs=1e-9)
    assert stats["VOUT"]["length_mm"] == 5.0
    fig = board_figure(ir, lib)
    assert f"{total:.4g}" == "17.14" and "트랙 4개 (동박 17.14 mm)" in fig.caption and "트랙 4개 (동박 17.14 mm)" in " ".join(_parse(fig.svg).itertext())
    text = circuit_report(ir, lib, None)
    section = text.split("\n## 배선")[1].split("\n## ")[0]
    assert "| `VIN` | 3 | 12.142 | F.Cu | 0 | 0.25 |" in section and "| **합계** | 4 | 17.142 | B.Cu, F.Cu | 1 | - |" in section
    assert "세그먼트 수에는 원호 트랙 1개가 들어 있고, 원호의 길이는 양 끝을 잇는 직선(현)이 아니라 호의 길이로 쟀습니다." in section
    assert "- 가장 긴 넷 `VIN` (12.142 mm, 폭 0.25 mm) 의 구리 저항:" in section and "넷별 동박 길이: VIN 12.14 mm, VOUT 5 mm, GND 0 mm." in section
    # what the chord between the arc's ends would give: 9 + 2.828 mm for VIN, 16.83 mm in all
    assert "| `VIN` | 3 | 11.828 |" not in text and "(동박 16.83 mm)" not in text and "(동박 16.83 mm)" not in fig.caption
    # the same copper with the corner as the router drew it: no word about arcs
    square = ir.model_copy(deep=True)
    square.pcb.tracks = [Track(net="VIN", layer="F.Cu", start=(5.0, 6.0), end=CORNER, width_mm=0.25, provenance=_routed("VIN")),
                         Track(net="VIN", layer="F.Cu", start=CORNER, end=(14.0, 10.0), width_mm=0.25, provenance=_routed("VIN")), ir.pcb.tracks[3]]
    plain = circuit_report(square, lib, None)
    assert "원호 트랙" not in plain and "| `VIN` | 2 | 13.000 | F.Cu | 0 | 0.25 |" in plain


# --------------------------------------------------------------------------- the corner sentences


def _cornered(net: str, layer: str, *entries: str) -> Track:
    return Track(net=net, layer=layer, start=(0.0, 0.0), end=(10.0, 0.0), width_mm=0.25, provenance=_routed(net, *entries))


MITRE_01_A, MITRE_01_B = "corner:routing.corners 0.1:miter45:at=12.0,4.0:chamfer=2.0", "corner:routing.corners 0.1:miter45:at=20.0,4.0:chamfer=1.0"
ARC_02_A, ARC_02_B = "corner:routing.corners 0.2:arc:at=12.0,4.0:tangent=1.6:radius=1.6", "corner:routing.corners 0.2:arc:at=20.0,4.0:tangent=0.692820:radius=0.4"
MITRE_02 = "corner:routing.corners 0.2:miter45:at=30.0,4.0:chamfer=0.12"
#: the sentence of a board mitred by routing.corners 0.1, as it has always been printed (three corners; c = g = 0.25 mm)
SENTENCE_01 = (
    "이 보드의 구리에는 `routing.corners` 0.1 의 45도 모따기가 3 곳 있습니다 (트랙 provenance 의 `corner:` 항목; 프로젝트 소유자의 모서리 규칙 — 전기적 요구가 아닌 기하 관례). "
    "위 반경은 격자 위의 구리에만 해당합니다. 모따기 구간은 격자 여유 g/2 없이 정확한 기하로 검사했습니다: 다른 넷의 트랙·비아·패드 상자와 구리 사이 간격 c = 0.25 mm "
    "(두 넷 중 넷 규칙의 간격이 더 크면 그 값; 넷이 없는 패드·구멍과는 c + g/2 = 0.375 mm), 중심선에서 트랙 keep-out 까지 w/2, 가장자리까지 e + w/2. "
    "그래서 모따기 근처의 구리는 위 반경보다 가까울 수 있으며, 그 판정도 kicad-cli DRC 가 합니다."
)
SENTENCE_02 = (
    "이 보드의 구리에는 `routing.corners` 0.2 의 둥근 모서리(원호 트랙)가 3 곳 (반지름 0.4–1.6 mm), 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 1 곳 있습니다 "
    "(트랙 provenance 의 `corner:` 항목; 프로젝트 소유자의 모서리 규칙 — 전기적 요구가 아닌 기하 관례). "
    "위 반경은 격자 위의 구리에만 해당합니다. 원호와 모따기 구간은 격자 여유 g/2 없이 정확한 기하로 검사했습니다: 다른 넷의 트랙·비아·패드 상자와 구리 사이 간격 c = 0.25 mm "
    "(두 넷 중 넷 규칙의 간격이 더 크면 그 값; 넷이 없는 패드·구멍과는 c + g/2 = 0.375 mm), 중심선에서 트랙 keep-out 까지 w/2, 가장자리까지 e + w/2. "
    "원호는 그 위의 점들을 잇는 짧은 직선(현)들의 사슬로 검사했고, 사슬과 원호의 차이 한계 0.001 mm 만큼 구리의 반폭을 키워 쟀습니다. "
    "그래서 원호·모따기 근처의 구리는 위 반경보다 가까울 수 있으며, 그 판정도 kicad-cli DRC 가 합니다. "
    "그대로 두었거나 건너뛴 모서리는 트랙에 항목을 남기지 않아 여기서 세지 않습니다 (라우터의 배선 메모가 그 수를 적었다면 아래 단계 기록에 있습니다)."
)


def test_the_corner_sentence_of_a_mitred_board_is_the_one_it_was():
    # one corner: one distinct entry per net and layer (a corner's entry is on every track it touched)
    tracks = [_cornered("A", "F.Cu", MITRE_01_A), _cornered("A", "F.Cu", MITRE_01_A, MITRE_01_B), _cornered("B", "B.Cu", MITRE_01_A)]
    assert stages._corner_lines(tracks, 0.25, 0.25) == [SENTENCE_01, ""]
    assert stages._corner_lines([_cornered("A", "F.Cu")], 0.25, 0.25) == []  # grid copper: nothing to say


def test_the_corner_sentence_of_a_rounded_board_counts_arcs_with_their_radii_and_the_mitres():
    assert ARC_SAGITTA_MM == 0.001
    tracks = [_cornered("A", "F.Cu", ARC_02_A), _cornered("A", "F.Cu", ARC_02_A, ARC_02_B), _cornered("B", "B.Cu", ARC_02_A), _cornered("B", "B.Cu", MITRE_02)]
    assert stages._corner_lines(tracks, 0.25, 0.25) == [SENTENCE_02, ""]
    # one radius is no range; no chamfer is said as none
    (one, _blank) = stages._corner_lines([_cornered("A", "F.Cu", ARC_02_A)], 0.25, 0.25)
    assert "둥근 모서리(원호 트랙)가 1 곳 (반지름 1.6 mm), 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 0 곳 있습니다" in one
    assert "전기적 요구가 아닌 기하 관례" in one and "45도 모따기가" not in one  # never the 0.1 sentence for a 0.2 board
    # chamfers only, made under the arc style
    (only, _blank) = stages._corner_lines([_cornered("A", "F.Cu", MITRE_02)], 0.25, 0.25)
    assert "`routing.corners` 0.2 의 둥근 모서리(원호 트랙)가 0 곳, 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 1 곳 있습니다" in only
    # copper holding both records gets both sentences, each with its own count
    both = stages._corner_lines([_cornered("A", "F.Cu", MITRE_01_A), _cornered("B", "F.Cu", ARC_02_B)], 0.25, 0.25)
    assert len(both) == 4 and "`routing.corners` 0.1 의 45도 모따기가 1 곳 있습니다" in both[0] and "0.2 의 둥근 모서리(원호 트랙)가 1 곳 (반지름 0.4 mm)" in both[2]


def test_copper_holding_both_records_counts_each_chamfer_under_its_own_version():
    # two corners mitred by 0.1, and on other nets one corner 0.2 mitred and one it rounded: the 0.1 sentence counts its own two only
    tracks = [_cornered("A", "F.Cu", MITRE_01_A, MITRE_01_B), _cornered("B", "F.Cu", MITRE_02), _cornered("C", "B.Cu", ARC_02_A)]
    lines = stages._corner_lines(tracks, 0.25, 0.25)
    assert len(lines) == 4 and lines[1] == lines[3] == ""
    assert lines[0].startswith("이 보드의 구리에는 `routing.corners` 0.1 의 45도 모따기가 2 곳 있습니다 (")
    assert lines[2].startswith("이 보드의 구리에는 `routing.corners` 0.2 의 둥근 모서리(원호 트랙)가 1 곳 (반지름 1.6 mm), 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 1 곳 있습니다 (")
    # the same chamfer record under the two versions is two records: one corner each
    (old, _b1, new, _b2) = stages._corner_lines([_cornered("A", "F.Cu", MITRE_01_A, MITRE_01_A.replace(" 0.1:", " 0.2:"))], 0.25, 0.25)
    assert "0.1 의 45도 모따기가 1 곳 있습니다" in old and "둥근 모서리(원호 트랙)가 0 곳, 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 1 곳 있습니다" in new


def test_a_record_of_a_later_version_or_without_a_finite_number_is_unread_not_counted():
    later = "corner:routing.corners 0.3:miter45:at=1.0,2.0:chamfer=1.0"  # a chamfer of a version this report does not know
    endless = [
        "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=1.0:radius=inf",
        "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=inf:radius=1.0",
        "corner:routing.corners 0.2:arc:at=nan,2.0:tangent=1.0:radius=1.0",
        "corner:routing.corners 0.2:arc:at=1.0,inf:tangent=1.0:radius=1.0",
        "corner:routing.corners 0.1:miter45:at=1.0,2.0:chamfer=inf",
        "corner:routing.corners 0.1:miter45:at=-inf,2.0:chamfer=1.0",
    ]
    assert [stages._corner_record(e) for e in (later, *endless)] == [None] * 7
    lines = stages._corner_lines([_cornered("A", "F.Cu", MITRE_01_A, ARC_02_A, later, *endless)], 0.25, 0.25)
    assert len(lines) == 6 and "0.1 의 45도 모따기가 1 곳 있습니다" in lines[0] and "둥근 모서리(원호 트랙)가 1 곳 (반지름 1.6 mm)" in lines[2]
    assert lines[4] == (
        "이 보드의 트랙 provenance 에는 이 보고서가 형식을 읽지 못하는 `corner:` 항목이 7 곳 있습니다 (`corner:routing.corners 0.1:miter45`, "
        "`corner:routing.corners 0.2:arc`, `corner:routing.corners 0.3:miter45`): 어떤 모서리인지 알 수 없어 위 설명에 넣지 않았습니다."
    )
    assert "inf" not in "".join(lines)  # no radius without an end in the range of radii


def test_a_corner_entry_in_an_unknown_form_is_counted_as_unread_never_called_a_chamfer():
    unknown = ["corner:routing.corners 0.3:spline:at=1.0,2.0:tension=0.5", "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=1.0",
               "corner:other.tool 0.1:miter45:at=1.0,2.0:chamfer=1.0", "corner:"]
    lines = stages._corner_lines([_cornered("A", "F.Cu", MITRE_01_A, *unknown)], 0.25, 0.25)
    assert len(lines) == 4 and "`routing.corners` 0.1 의 45도 모따기가 1 곳 있습니다" in lines[0]
    assert lines[2] == (
        "이 보드의 트랙 provenance 에는 이 보고서가 형식을 읽지 못하는 `corner:` 항목이 4 곳 있습니다 (`corner:`, `corner:other.tool 0.1:miter45`, "
        "`corner:routing.corners 0.2:arc`, `corner:routing.corners 0.3:spline`): 어떤 모서리인지 알 수 없어 위 설명에 넣지 않았습니다."
    )
    (alone, _blank) = stages._corner_lines([_cornered("A", "F.Cu", unknown[0])], 0.25, 0.25)
    assert "모따기" not in alone and "원호" not in alone
    # a record is read field for field: one with a field missing, repeated, added or without a number in it is none of the forms
    malformed = [
        "corner:routing.corners 0.2:arc:radius=1.0",  # no corner, no tangent
        "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=1.0:radius=1.0:radius=2.0",
        "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=1.0:radius=-1.0",
        "corner:routing.corners 0.2:arc:at=1.0,2.0:tangent=nan:radius=1.0",
        "corner:routing.corners 0.2:arc:at=1.0:tangent=1.0:radius=1.0",
        "corner:routing.corners 0.1:arc:at=1.0,2.0:tangent=1.0:radius=1.0",  # 0.1 rounds nothing
        "corner:routing.corners 0.1:miter45:chamfer=1.0",
        "corner:routing.corners 0.1:miter45:at=1.0,2.0:chamfer=wide",
        "corner:routing.corners 0.1:miter45:at=1.0,2.0:chamfer=0.0",
        "corner:routing.corners 0.2:miter45:at=1.0,2.0:chamfer=1.0:radius=1.0",
    ]
    assert [stages._corner_record(e) for e in malformed] == [None] * len(malformed)
    assert stages._corner_record(MITRE_01_A) == ("0.1", "miter45", None) and stages._corner_record(MITRE_02) == ("0.2", "miter45", None)
    assert stages._corner_record(ARC_02_B) == ("0.2", "arc", 0.4) and stages._corner_record("corner:routing.corners 0.2:arc:at=-1e-05,2.0:tangent=1.0:radius=1.5") == ("0.2", "arc", 1.5)
    (lines_malformed, _blank) = stages._corner_lines([_cornered("A", "F.Cu", *malformed)], 0.25, 0.25)
    assert "형식을 읽지 못하는 `corner:` 항목이 10 곳 있습니다" in lines_malformed and "모따기" not in lines_malformed and "원호" not in lines_malformed


def test_the_circuit_report_words_the_rounded_corner_in_its_keep_out_section(tmp_path: Path):
    ir, lib, _arc_track = _arc_board(tmp_path)
    text = circuit_report(ir, lib, None)
    section = text.split("### 라우터의 keep-out 규칙")[1].split("###")[0]
    assert "`routing.corners` 0.2 의 둥근 모서리(원호 트랙)가 1 곳 (반지름 2 mm), 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 0 곳 있습니다" in section
    assert "전기적 요구가 아닌 기하 관례" in section and "사슬과 원호의 차이 한계 0.001 mm 만큼" in section and "45도 모따기가" not in section
    assert circuit_report(ir, lib, None) == text


# --------------------------------------------------------------------------- the GUI stylesheet


def test_the_gui_class_colours_reach_an_arc_path_like_a_line():
    from ai_eda.gui.page import APP_CSS, APP_JS, NETCLASS_CSS

    rules = NETCLASS_CSS.splitlines()
    for_lines = [r for r in rules if "line.track" in r]
    assert len(for_lines) == len(CLASS_COLOURS) + 4  # one per class hue, the default and the overflow class; B.Cu lighter; no class gray
    for rule in for_lines:
        assert rule.replace("line.track", "path.track") in rules, rule
    assert [r for r in rules if ".track" in r and "line.track" not in r and "path.track" not in r] == []  # no rule for one element only
    # the arc rules are the line rules again, in their order, after every other rule: nothing that was in the sheet moved or changed
    assert rules[-len(for_lines):] == [r.replace("line.track", "path.track") for r in for_lines] and rules[-len(for_lines) - 1].startswith("/*")
    assert not any("path.track" in r for r in rules[:-len(for_lines)]) and APP_CSS.endswith(NETCLASS_CSS)
    assert f'.board-view.by-class path.track[data-nc="nc-0"] {{ stroke: {CLASS_COLOURS[0]}; }}' in rules
    assert f'.board-view.by-class path.track[data-nc="nc-default"] {{ stroke: {NET_CLASS_DEFAULT}; }}' in rules
    assert '.board-view.by-class path.track[data-layer="B.Cu"] { stroke-opacity: 0.5; }' in rules
    assert ".board-view.by-class path.track:not([data-nc]) { stroke: #c3c2b7; }" in rules
    assert NETCLASS_CSS in APP_CSS and "url(" not in APP_CSS
    # the page's script toggles layer groups and imports the SVG without scripts or style attributes: it names no single track, and a
    # path element is not among the elements it removes
    assert "line.track" not in APP_JS and "path.track" not in APP_JS
    assert "root.querySelectorAll('script, foreignObject, iframe, image, use')" in APP_JS


def test_the_class_view_draws_an_arc_in_its_class_colour(tmp_path: Path):
    lib = fixture_library(tmp_path / "kicad")
    ir = long_board(tmp_path, lib, None, length=30.0)
    ir.si = si_of(*default_classes())
    a, mid, b, _radius = fillet((10.0, 15.0), (5.0, 15.0), (10.0, 10.0), 1.0)
    ir.pcb.tracks = [
        Track(net="SHORT", layer="B.Cu", start=(5.0, 15.0), end=a, width_mm=0.4),
        Track(net="SHORT", layer="B.Cu", start=a, mid=mid, end=b, width_mm=0.4),
        Track(net="LONG", layer="F.Cu", start=(5.0, 5.0), end=(35.0, 5.0), width_mm=0.4),
    ]
    # the layer view - the GUI's board preview - tags the arc with its net class like every track (what the GUI's rules select on) ...
    assert board_svg(ir, lib) == board_figure(ir, lib).svg
    root = _parse(board_figure(ir, lib).svg)
    (path,) = [e for e in _by_class(root, "track") if e.tag == f"{SVG_NS}path"]
    assert (path.get("data-net"), path.get("data-layer"), path.get("data-class"), path.get("data-nc")) == ("SHORT", "B.Cu", "DEFAULT", "nc-default")
    assert path.get("stroke") == COPPER_COLOURS["B.Cu"] and path.get("stroke-opacity") is None
    # ... and the class view draws it in the class's colour, lighter on B.Cu, as it draws a line
    croot = _parse(board_figure(ir, lib, colour_by="class", fig_id="si_board").svg)
    (cpath,) = [e for e in _by_class(croot, "track") if e.tag == f"{SVG_NS}path"]
    (cline,) = [e for e in _by_class(croot, "track") if e.tag == f"{SVG_NS}line" and e.get("data-net") == "SHORT"]
    for name in ("class", "data-net", "data-layer", "data-class", "data-nc", "stroke", "stroke-opacity", "stroke-width", "stroke-linecap"):
        assert cpath.get(name) == cline.get(name), name
    assert cpath.get("stroke") == NET_CLASS_DEFAULT and cpath.get("stroke-opacity") == "0.5" and cpath.get("fill") == "none"
    to_px = _figure_map(croot, ir)  # the class view sits lower (its legend): the arc is drawn where that view puts the board
    assert _apart(cpath.get("d"), a, mid, b, to_px) < PX
    # the straight tracks beside the arc are written as they were before a track could be an arc, letter for letter: the class
    # attributes in the layer view, the class colour (lighter on B.Cu, solid on F.Cu) in the class view
    assert ('<line class="track" data-net="SHORT" data-layer="B.Cu" data-class="DEFAULT" data-nc="nc-default" x1="114.00" y1="262.00" x2="173.20" y2="262.00" '
            'stroke="#2a78d6" stroke-width="5.92" stroke-linecap="round"/>') in board_figure(ir, lib).svg
    class_svg = board_figure(ir, lib, colour_by="class", fig_id="si_board").svg
    assert ('<line class="track" data-net="SHORT" data-layer="B.Cu" data-class="DEFAULT" data-nc="nc-default" x1="114.00" y1="268.00" x2="173.20" y2="268.00" '
            'stroke="#8a8983" stroke-opacity="0.5" stroke-width="5.92" stroke-linecap="round"/>') in class_svg
    assert ('<line class="track" data-net="LONG" data-layer="F.Cu" data-class="DEFAULT" data-nc="nc-default" x1="114.00" y1="120.00" x2="558.00" y2="120.00" '
            'stroke="#8a8983" stroke-width="5.92" stroke-linecap="round"/>') in class_svg


# --------------------------------------------------------------------------- the 3D preview


def test_the_3d_scene_draws_one_bar_per_chord_of_the_arc(tmp_path: Path):
    lib = scene_library(tmp_path / "kicad")
    a, mid, b, _radius = fillet((6.0, 2.0), (2.0, 2.0), (6.0, 6.0), 1.0)
    arc = Track(net="A", layer="F.Cu", start=a, mid=mid, end=b, width_mm=0.25, provenance=USER)
    straight = Track(net="A", layer="F.Cu", start=(2.0, 2.0), end=a, width_mm=0.25, provenance=USER)
    hidden = Track(net="A", layer="In1.Cu", start=b, mid=mid, end=a, width_mm=0.25, provenance=USER)
    ir = scene_ir([("U1", "BOX", _place("U1", x=20.0))], tracks=[straight, arc, hidden])
    scene = build_scene(ir, lib, model_dir=None)
    chain = track_points(arc)
    assert len(chain) - 1 == 18 and chain[0] == a and chain[-1] == b  # a quarter arc of radius 1 mm: 18 chords at the 1 um bound
    solids = _only(scene, "track", "A")
    assert len(solids) == 1 + 18  # the straight track's bar, then one bar per chord (the inner-layer arc is not drawn)
    half = 0.125
    for solid, (p, q) in zip(solids[1:], zip(chain, chain[1:])):
        assert (solid.material, solid.faces) == ("copper", "top")
        # a round-ended bar about exactly this chord: every corner of its outline is half a width from the chord
        for x, y in solid.polygon:
            dx, dy = q[0] - p[0], q[1] - p[1]
            t = max(0.0, min(1.0, ((x - p[0]) * dx + (y - p[1]) * dy) / (dx * dx + dy * dy)))
            assert math.hypot(x - p[0] - t * dx, y - p[1] - t * dy) == pytest.approx(half, abs=2e-6)
        xs, ys = [v[0] for v in solid.polygon], [v[1] for v in solid.polygon]
        assert min(xs) < min(p[0], q[0]) and max(xs) > max(p[0], q[0]) and min(ys) < min(p[1], q[1]) and max(ys) > max(p[1], q[1])
    # the bars follow the arc, not its chord: none covers the middle of the straight line between its ends
    middle = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    assert point_arc_distance(middle, a, mid, b) > half + 0.1
    assert all(min(math.hypot(middle[0] - x, middle[1] - y) for x, y in s.polygon) > 0.1 for s in solids[1:])
    assert "원호 트랙 1개는 현 18개를 이은 사슬로 그림 (사슬과 원호의 차이 0.001 mm 이내)" in scene.notes
    assert "안쪽 층 트랙 1개는 보드 속에 있어 그리지 않음" in scene.notes
    assert scene.version == SCENE_VERSION == "0.1" and scene.counts()["track"] == 19
    assert write_glb(scene) == write_glb(build_scene(ir, lib, model_dir=None))  # the writers take the chain as they take any bar
    # a board without an arc: one bar per track, no word about arcs
    plain = build_scene(scene_ir([("U1", "BOX", _place("U1", x=20.0))], tracks=[straight]), lib, model_dir=None)
    assert len(_only(plain, "track")) == 1 and not any("원호 트랙" in n for n in plain.notes)


def test_an_arc_on_the_bottom_layer_is_drawn_under_the_board_at_a_straight_track_s_height(tmp_path: Path):
    lib = scene_library(tmp_path / "kicad")
    a, mid, b, _radius = fillet((6.0, 2.0), (2.0, 2.0), (6.0, 6.0), 1.0)
    tracks = [
        Track(net="TOP", layer="F.Cu", start=(2.0, 2.0), end=a, width_mm=0.25, provenance=USER),
        Track(net="TOP", layer="F.Cu", start=a, mid=mid, end=b, width_mm=0.25, provenance=USER),
        Track(net="BOTTOM", layer="B.Cu", start=(2.0, 2.0), end=a, width_mm=0.4, provenance=USER),
        Track(net="BOTTOM", layer="B.Cu", start=b, mid=mid, end=a, width_mm=0.4, provenance=USER),  # the same corner, stored from its other end
    ]
    scene = build_scene(scene_ir([("U1", "BOX", _place("U1", x=20.0))], tracks=tracks), lib, model_dir=None)
    for net, faces, half in (("TOP", "top", 0.125), ("BOTTOM", "bottom", 0.2)):
        straight, *bars = _only(scene, "track", net)
        assert len(bars) == 18 and {s.faces for s in bars} == {faces} == {straight.faces}  # each side's arc on its own side of the board
        assert {(s.material, s.z0, s.z1) for s in bars} == {(straight.material, straight.z0, straight.z1)}  # at the straight track's height
        # the arc's own width: the corners of a bar lie half a width round the chain's points, which lie on the arc - so the bars
        # reach half a width from the arc's centreline and no further
        reach = max(point_arc_distance(p, a, mid, b) for s in bars for p in s.polygon)
        assert reach == pytest.approx(half, abs=2e-6)
    top, bottom = _only(scene, "track", "TOP")[1:], _only(scene, "track", "BOTTOM")[1:]
    assert top[0].z0 > bottom[0].z1  # above the board, and under it
    assert "원호 트랙 2개는 현 36개를 이은 사슬로 그림 (사슬과 원호의 차이 0.001 mm 이내)" in scene.notes


# --------------------------------------------------------------------------- the layout metrics


def test_the_routed_length_metric_counts_the_arc_length_and_a_frozen_baseline_keeps_its_number(tmp_path: Path):
    lib = synthetic_library(tmp_path / "library")
    ir = parts_ir(tmp_path, lib, ("R1", "R2"), nets=False)
    prov = ir.components[0].provenance
    ir.nets = [Net(name="SIGNAL", pins=[PinRef(component_ref=r, pin_number="1") for r in ("R1", "R2")], provenance=prov)]
    start, mid, end = _arc(5.0, 4.0, 2.0, 180.0, 90.0)  # from (3, 4) round to (5, 2): a quarter of radius 2 mm
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=20, height_mm=10),
                       placements=[Placement(component_ref="R1", x_mm=2, y_mm=2), Placement(component_ref="R2", x_mm=5, y_mm=6)],
                       layers=[Layer(name="F.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")],
                       tracks=[Track(net="SIGNAL", layer="F.Cu", start=(0, 0), end=(3, 4), width_mm=0.25),
                               Track(net="SIGNAL", layer="F.Cu", start=start, mid=mid, end=end, width_mm=0.25)])
    metrics = pcb_metrics(ir, lib)
    assert metrics["routed_track_length_mm"] == round(5.0 + arc_length(start, mid, end), 6) == pytest.approx(5.0 + math.pi, abs=1e-5)
    assert metrics["routed_track_length_mm"] != round(5.0 + math.dist(start, end), 6)  # the chord: 2.83 mm
    # a baseline frozen before arcs existed: no arc in it, and the number recorded then
    recorded = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))["boards"]["divider"]["pcb"]
    frozen, footprints = load_geometry("divider")
    assert not any(t.is_arc for t in frozen.pcb.tracks)
    assert pcb_metrics(frozen, footprints) == recorded and recorded["routed_track_length_mm"] == 29.165464


# --------------------------------------------------------------------------- an arc that is none


def test_an_arc_broken_after_the_model_checked_it_is_refused_by_name(tmp_path: Path):
    ir, lib, arc = _arc_board(tmp_path)
    metric_lib = synthetic_library(tmp_path / "library")
    board = parts_ir(tmp_path, metric_lib, ("R1",), nets=False)
    board.pcb = PCBDesign(outline=BoardOutline(width_mm=20, height_mm=10), placements=[Placement(component_ref="R1", x_mm=2, y_mm=2)], tracks=[arc])
    assert pcb_metrics(board, metric_lib)["routed_track_length_mm"] == round(arc.length_mm, 6)
    arc.mid = (13.0, 7.0)  # on the straight line between its ends: no arc (an assignment is not checked by the model)
    with pytest.raises(ValueError, match=r"track of 'VIN' on F\.Cu: start \(12\.0, 6\.0\) / mid \(13\.0, 7\.0\) / end \(14\.0, 8\.0\) is no arc: its mid point lies on the line"):
        board_figure(ir, lib)
    with pytest.raises(ValueError, match="no arc from .* its mid point lies on the line"):
        net_routing_stats(ir)
    with pytest.raises(SceneError, match=r"track of 'VIN' on F\.Cu: no arc from .* its mid point lies on the line"):
        build_scene(ir, lib, model_dir=None)
    with pytest.raises(ValueError, match="no arc from"):
        _arc_path((12.0, 6.0), (13.0, 7.0), (14.0, 8.0), _plain)
    assert board.pcb.tracks[0] is arc  # the board measured above holds the same, now broken, track
    with pytest.raises(ValueError, match="metrics.track: no arc from .* its mid point lies on the line"):
        pcb_metrics(board, metric_lib)
