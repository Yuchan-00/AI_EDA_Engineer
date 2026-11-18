"""Track arcs in the IR: the arc geometry (``ai_eda.ir.arc``), ``Track.mid`` and the helpers every reader of tracks uses.

A track with ``mid`` is the circular arc from ``start`` through ``mid`` to ``end`` (KiCad's own form). These tests pin the
geometry against hand-computed circles, the two-sided chain rule against dense samples of the true arc, and the promise that a
track without ``mid`` is serialised exactly as before the field existed.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ai_eda.ir import Provenance, ProvenanceKind, Track, straight_pieces, track_bbox, track_length, track_points
from ai_eda.ir.arc import (
    ARC_COLLINEAR_MM,
    ARC_ROUNDING_MM,
    ARC_SAGITTA_MM,
    arc_bbox,
    arc_geometry,
    arc_length,
    arc_point_at,
    arc_points,
    arc_problem,
    arc_split,
    fillet,
    point_arc_distance,
    point_arc_fraction,
)
from ai_eda.ir.provenance import design_data
from ai_eda.tools.keepout import segment_area_distance, track_area_distance

S2 = math.sqrt(0.5)
#: a quarter of the unit circle from angle 0 to angle 90 degrees (toward larger angles: a positive sweep)
QUARTER = ((1.0, 0.0), (S2, S2), (0.0, 1.0))


def _on_circle(cx: float, cy: float, r: float, a0: float, sweep: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Start, mid and end of the arc of that circle, each rounded to the file resolution as a stored arc is."""
    def at(a: float) -> tuple[float, float]:
        return (round(cx + r * math.cos(a), 6), round(cy + r * math.sin(a), 6))
    return at(a0), at(a0 + sweep / 2.0), at(a0 + sweep)


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


# --------------------------------------------------------------------------- the circle


def test_a_quarter_arc_has_its_centre_radius_sweep_and_length_in_both_directions():
    g = arc_geometry(*QUARTER)
    assert (g.cx, g.cy) == pytest.approx((0.0, 0.0), abs=1e-12) and g.r == pytest.approx(1.0)
    assert g.a0 == pytest.approx(0.0, abs=1e-12) and g.sweep == pytest.approx(math.pi / 2.0) and g.length == pytest.approx(math.pi / 2.0)
    back = arc_geometry(*reversed(QUARTER))  # the same copper drawn from the other end: the sweep changes its sign
    assert back.a0 == pytest.approx(math.pi / 2.0) and back.sweep == pytest.approx(-math.pi / 2.0) and back.length == pytest.approx(g.length)
    assert arc_length(*QUARTER) == g.length


def test_the_sweep_is_the_way_round_that_passes_mid():
    # a half circle of radius 2 from (-2, 0) over (0, 2) to (2, 0): from angle 180 over 90 to 0 degrees, toward smaller angles
    g = arc_geometry((-2.0, 0.0), (0.0, 2.0), (2.0, 0.0))
    assert (g.cx, g.cy, g.r) == pytest.approx((0.0, 0.0, 2.0)) and g.sweep == pytest.approx(-math.pi) and g.length == pytest.approx(2.0 * math.pi)
    # the same two ends through the other side: toward larger angles
    assert arc_geometry((-2.0, 0.0), (0.0, -2.0), (2.0, 0.0)).sweep == pytest.approx(math.pi)
    # three quarters of a turn: from angle 0 over 180 to 270 degrees
    three = arc_geometry((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0))
    assert three.sweep == pytest.approx(1.5 * math.pi) and three.length == pytest.approx(1.5 * math.pi)
    # and the quarter between the same ends, the short way
    assert arc_geometry((1.0, 0.0), (S2, -S2), (0.0, -1.0)).sweep == pytest.approx(-math.pi / 2.0)


def test_the_centre_keeps_its_precision_far_from_the_origin():
    pts = _on_circle(1234.5, -987.25, 0.3, 0.4, 1.9)
    g = arc_geometry(*pts)
    assert (g.cx, g.cy, g.r) == pytest.approx((1234.5, -987.25, 0.3), abs=2e-6) and g.sweep == pytest.approx(1.9, abs=1e-5)


def test_the_box_is_exact_and_takes_the_axis_extremes_the_sweep_passes():
    assert arc_bbox(*QUARTER) == pytest.approx((0.0, 0.0, 1.0, 1.0))
    # three quarters of a turn from (1, 0) over (-1, 0) to (0, -1) pass the extremes (0, 1) and (-1, 0), not (0, -1) .. (1, 0) below the ends
    assert arc_bbox((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)) == pytest.approx((-1.0, -1.0, 1.0, 1.0))
    # a bulge between its ends: the chord from (-S2, S2) to (S2, S2) over the top (0, 1)
    assert arc_bbox((-S2, S2), (0.0, 1.0), (S2, S2)) == pytest.approx((-S2, S2, S2, 1.0))
    t = Track(net="N", layer="F.Cu", start=(-S2, S2), mid=(0.0, 1.0), end=(S2, S2), width_mm=0.2)
    assert track_bbox(t) == arc_bbox((-S2, S2), (0.0, 1.0), (S2, S2))
    assert track_bbox(SimpleNamespace(start=(3.0, 1.0), end=(1.0, 2.0))) == (1.0, 1.0, 3.0, 2.0)


# --------------------------------------------------------------------------- the chain


@pytest.mark.parametrize("radius", [0.05, 0.15, 0.4, 1.0, 7.5, 50.0])
@pytest.mark.parametrize("sweep_deg", [5.0, 60.0, 90.0, 179.0, 181.0, 270.0, 350.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_chain_rule_holds_both_ways_for_the_rounded_chain(radius: float, sweep_deg: float, sign: float):
    start, mid, end = _on_circle(12.3456, -7.891, radius, 0.7, sign * math.radians(sweep_deg))
    g = arc_geometry(start, mid, end)
    for sag in (ARC_SAGITTA_MM, 0.01, 0.0001):
        chain = arc_points(start, mid, end, sag)
        assert chain[0] == start and chain[-1] == end  # the arc's own ends, not recomputed
        assert all(p == (round(p[0], 6), round(p[1], 6)) for p in chain[1:-1])
        k = len(chain) - 1
        # every point of the arc within sag of the chain (20 samples per chord, each against its own chord and the two beside it)
        for i in range(20 * k + 1):
            a = g.a0 + g.sweep * i / (20 * k)
            p = (g.cx + g.r * math.cos(a), g.cy + g.r * math.sin(a))
            j = min(k - 1, i // 20)
            assert min(_seg_dist(p, chain[m], chain[m + 1]) for m in range(max(0, j - 1), min(k, j + 2))) <= sag + 1e-12
        # every vertex and every chord middle of the chain within sag of the arc
        for a, b in zip(chain, chain[1:]):
            assert point_arc_distance(a, start, mid, end) <= sag + 1e-12
            assert point_arc_distance(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), start, mid, end) <= sag + 1e-12


def test_the_chain_is_as_coarse_as_its_bound_allows_and_refuses_a_bound_below_the_file_resolution():
    # a quarter of radius 1 mm at 1 um: a chord may span 2 acos(1 - 0.000999) = 0.0894 rad, so 18 chords
    assert len(arc_points(*QUARTER)) - 1 == math.ceil((math.pi / 2.0) / (2.0 * math.acos(1.0 - (ARC_SAGITTA_MM - ARC_ROUNDING_MM))))
    assert len(arc_points(*QUARTER)) - 1 == 18
    assert len(arc_points(*QUARTER, 0.0001)) > len(arc_points(*QUARTER)) > len(arc_points(*QUARTER, 0.01))
    # a bound of the radius or more: one chord per started half turn, never a chord over more than a half turn
    assert len(arc_points(*QUARTER, 5.0)) == 2
    assert len(arc_points((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), 5.0)) == 3
    for bad in (0.0, -0.001, 2.0 * ARC_ROUNDING_MM, math.nan, math.inf):
        with pytest.raises(ValueError, match="max_sagitta"):
            arc_points(*QUARTER, bad)


# --------------------------------------------------------------------------- points on and beside the arc


def test_the_distance_of_a_point_is_exact_inside_and_outside_the_sweep():
    assert point_arc_distance((2.0 * S2, 2.0 * S2), *QUARTER) == pytest.approx(1.0)  # outside the circle, in the sweep
    assert point_arc_distance((0.5 * S2, 0.5 * S2), *QUARTER) == pytest.approx(0.5)  # inside the circle, in the sweep
    assert point_arc_distance((0.0, 0.0), *QUARTER) == pytest.approx(1.0)  # the centre: one radius from every point
    assert point_arc_distance((1.0, -1.0), *QUARTER) == pytest.approx(1.0)  # beyond the start: the start is nearest
    assert point_arc_distance((-3.0, 1.0), *QUARTER) == pytest.approx(3.0)  # beyond the end
    # a point right under the chord's middle is far from the arc though it is on the chord
    assert point_arc_distance((0.5, 0.5), *QUARTER) == pytest.approx(1.0 - S2)
    d, f = point_arc_fraction((2.0 * math.cos(math.radians(30.0)), 2.0 * math.sin(math.radians(30.0))), *QUARTER)
    assert d == pytest.approx(1.0) and f == pytest.approx(1.0 / 3.0)
    assert point_arc_fraction((1.0, -1.0), *QUARTER) == (pytest.approx(1.0), 0.0) and point_arc_fraction((-3.0, 1.0), *QUARTER)[1] == 1.0
    assert point_arc_fraction((0.0, 0.0), *QUARTER) == (pytest.approx(1.0), 0.0)
    # the other direction of the same copper: fractions are counted from its own start
    assert point_arc_fraction((2.0 * math.cos(math.radians(30.0)), 2.0 * math.sin(math.radians(30.0))), *reversed(QUARTER))[1] == pytest.approx(2.0 / 3.0)


def test_the_point_at_a_fraction_and_the_split():
    assert arc_point_at(*QUARTER, 0.0) == QUARTER[0] and arc_point_at(*QUARTER, 1.0) == QUARTER[2]
    assert arc_point_at(*QUARTER, 0.5) == pytest.approx(QUARTER[1])
    assert arc_point_at(*QUARTER, 1.0 / 3.0) == pytest.approx((math.cos(math.radians(30.0)), math.sin(math.radians(30.0))))
    for bad in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="fraction"):
            arc_point_at(*QUARTER, bad)
    (s1, m1, e1), (s2, m2, e2) = arc_split(*QUARTER, 1.0 / 3.0)
    assert s1 == QUARTER[0] and e2 == QUARTER[2] and e1 == s2 == (round(math.cos(math.radians(30.0)), 6), 0.5)
    g1, g2 = arc_geometry(s1, m1, e1), arc_geometry(s2, m2, e2)
    for g in (g1, g2):  # the same circle, as well as three points rounded to 1 nm on a short arc give it back
        assert (g.cx, g.cy, g.r) == pytest.approx((0.0, 0.0, 1.0), abs=5e-5)
    assert g1.length == pytest.approx(math.pi / 6.0, abs=5e-6) and g2.length == pytest.approx(math.pi / 3.0, abs=5e-6)
    # a part too flat to be an arc at the file resolution is the straight segment the file would hold
    big = _on_circle(0.0, 0.0, 50.0, 0.0, math.radians(20.0))
    (_s, flat, _e), (_s2, rest, _e2) = arc_split(*big, 0.0005)
    assert flat is None and rest is not None
    for bad in (0.0, 1.0, -0.2, 1e-12):  # nothing to cut at an end (1e-12 of this arc is below the file resolution)
        with pytest.raises(ValueError):
            arc_split(*QUARTER, bad)


# --------------------------------------------------------------------------- the fillet


def test_a_right_angle_gets_a_fillet_of_its_tangent_length():
    # the corner at (9, 3) of a track coming from the left and leaving downwards (Y down), tangent length 2.5
    a, mid, b, r = fillet((9.0, 3.0), (3.0, 3.0), (9.0, 8.0), 2.5)
    assert a == (6.5, 3.0) and b == (9.0, 5.5) and r == pytest.approx(2.5)
    g = arc_geometry(a, mid, b)
    assert (g.cx, g.cy, g.r) == pytest.approx((6.5, 5.5, 2.5), abs=2e-6) and abs(g.sweep) == pytest.approx(math.pi / 2.0, abs=1e-6)
    # mid lies on the bisector, between the corner and the chord
    assert mid == pytest.approx((6.5 + 2.5 * S2, 5.5 - 2.5 * S2), abs=1e-6)
    # tangent: the radius at each cut point is perpendicular to its leg
    assert (a[0] - g.cx) * (3.0 - 9.0) + (a[1] - g.cy) * 0.0 == pytest.approx(0.0, abs=1e-5)
    assert (b[0] - g.cx) * 0.0 + (b[1] - g.cy) * (8.0 - 3.0) == pytest.approx(0.0, abs=1e-5)
    # the arc is shorter than the corner it replaces by (2 - pi / 2) radii
    assert 2.0 * 2.5 - arc_length(a, mid, b) == pytest.approx((2.0 - math.pi / 2.0) * 2.5, abs=1e-5)


@pytest.mark.parametrize("angle_deg", [30.0, 60.0, 90.0, 120.0, 150.0])
def test_the_fillet_radius_is_d_tan_half_the_angle_and_the_arc_is_tangent(angle_deg: float):
    al = math.radians(angle_deg)
    d = 1.2
    corner, toward1, toward2 = (2.0, -1.0), (7.0, -1.0), (2.0 + 4.0 * math.cos(al), -1.0 + 4.0 * math.sin(al))
    a, mid, b, r = fillet(corner, toward1, toward2, d)
    assert r == pytest.approx(d * math.tan(al / 2.0))
    assert math.hypot(a[0] - corner[0], a[1] - corner[1]) == pytest.approx(d, abs=2e-6) and math.hypot(b[0] - corner[0], b[1] - corner[1]) == pytest.approx(d, abs=2e-6)
    g = arc_geometry(a, mid, b)
    assert g.r == pytest.approx(r, abs=1e-4) and abs(g.sweep) == pytest.approx(math.pi - al, abs=1e-4)
    for cut, toward in ((a, toward1), (b, toward2)):
        rx, ry, lx, ly = cut[0] - g.cx, cut[1] - g.cy, toward[0] - corner[0], toward[1] - corner[1]
        assert abs(rx * lx + ry * ly) / (math.hypot(rx, ry) * math.hypot(lx, ly)) < 1e-4
    # the arc bulges toward the corner
    assert math.hypot(mid[0] - corner[0], mid[1] - corner[1]) < math.hypot((a[0] + b[0]) / 2.0 - corner[0], (a[1] + b[1]) / 2.0 - corner[1])


def test_the_fillet_refuses_what_is_no_corner():
    with pytest.raises(ValueError, match="tangent length"):
        fillet((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.0)
    with pytest.raises(ValueError, match="tangent length"):
        fillet((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), math.nan)
    with pytest.raises(ValueError, match="no length"):
        fillet((0.0, 0.0), (0.0, 0.0), (0.0, 1.0), 0.5)
    with pytest.raises(ValueError, match="parallel"):
        fillet((0.0, 0.0), (1.0, 0.0), (-3.0, 0.0), 0.5)  # straight on
    with pytest.raises(ValueError, match="parallel"):
        fillet((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), 0.5)  # back over itself


# --------------------------------------------------------------------------- what is no arc


def test_three_points_that_are_no_arc_are_named():
    assert arc_problem(*QUARTER) is None
    assert "coincide" in arc_problem((0.0, 0.0), (1.0, 1.0), (0.0, 0.0))
    assert "mid point coincides with its start" in arc_problem((0.0, 0.0), (0.0, 0.0), (1.0, 1.0))
    assert "mid point coincides with its end" in arc_problem((0.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    assert "on the line" in arc_problem((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    assert "on the line" in arc_problem((0.0, 0.0), (1.0, ARC_COLLINEAR_MM / 2.0), (2.0, 0.0))
    assert arc_problem((0.0, 0.0), (1.0, 2.0 * ARC_COLLINEAR_MM), (2.0, 0.0)) is None
    for k in range(3):  # a NaN or an infinity anywhere, in each of the three points (comparisons with a NaN are false: tested first)
        for bad in (math.nan, math.inf):
            pts = [list(p) for p in QUARTER]
            pts[k][1] = bad
            assert "is not finite" in arc_problem(*[tuple(p) for p in pts])
    with pytest.raises(ValueError, match="no arc from"):
        arc_geometry((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))


def test_a_circle_too_large_to_compute_to_the_file_resolution_is_no_arc_and_a_large_one_keeps_its_bound():
    # a 20 m chord with a mid 2 nm off it: a radius of 2.5e13 mm, where a double places a point to millimetres at best
    assert "its radius (2.5e+13 mm) is above 1e+06 mm" in arc_problem((0.0, 0.0), (10000.0, 2e-06), (20000.0, 0.0))
    assert "its circle cannot be computed" in arc_problem((0.0, 0.0), (1e200, 1e200), (2e200, 0.0))  # the sides cannot even be multiplied
    with pytest.raises(ValidationError, match="its radius"):
        Track(net="N", layer="F.Cu", start=(0.0, 0.0), mid=(10000.0, 2e-06), end=(20000.0, 0.0), width_mm=0.25)
    # just inside the limit: a 300 mm chord of a circle of radius 9e5 mm - the chain is computed, and its bound holds
    r, half = 9.0e5, 150.0
    sag = r - math.sqrt(r * r - half * half)  # 0.0125 mm: the arc's middle above its chord
    start, mid, end = (-half, 0.0), (0.0, round(sag, 6)), (half, 0.0)
    assert arc_problem(start, mid, end) is None and arc_geometry(start, mid, end).r == pytest.approx(r, rel=1e-3)
    chain = arc_points(start, mid, end)
    assert chain[0] == start and chain[-1] == end and len(chain) - 1 == 4  # 4 chords: each 75 mm chord's sagitta is 0.78 um
    for a, b in zip(chain, chain[1:]):
        assert point_arc_distance(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), start, mid, end) <= ARC_SAGITTA_MM + 1e-12


# --------------------------------------------------------------------------- Track.mid


PROV = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.9", derived_from=["net:N"],
                  created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))


def test_a_track_without_mid_serialises_exactly_as_before_the_field_existed():
    """The strings were printed by the model before ``mid`` existed, for the same track."""
    t = Track(net="N", layer="F.Cu", start=(1.25, -2.5), end=(3.0, 4.5), width_mm=0.25, provenance=PROV)
    assert t.mid is None and not t.is_arc
    assert t.model_dump_json() == (
        '{"net":"N","layer":"F.Cu","start":[1.25,-2.5],"end":[3.0,4.5],"width_mm":0.25,"provenance":{"kind":"derived","source":null,'
        '"derived_from":["net:N"],"inputs":{},"tool":"routing.maze","tool_version":"0.9","note":null,"created_at":"2026-01-02T03:04:05Z"}}'
    )
    assert design_data(t) == {"net": "N", "layer": "F.Cu", "start": [1.25, -2.5], "end": [3.0, 4.5], "width_mm": 0.25,
                              "provenance": {"kind": "derived", "source": None, "derived_from": ["net:N"], "inputs": {}, "tool": "routing.maze",
                                             "tool_version": "0.9", "note": None}}
    assert "mid" not in t.model_dump() and "mid" not in t.model_dump(mode="json")
    assert Track.model_validate_json(t.model_dump_json()) == t


def test_a_straight_track_stays_as_permissive_as_it_was():
    # a non-finite end or a width that is not positive is refused by the compiler and the checks, not by the model
    assert Track(net="N", layer="F.Cu", start=(math.nan, 0.0), end=(1.0, 1.0), width_mm=0.0).width_mm == 0.0
    assert Track(net="N", layer="F.Cu", start=(0.0, 0.0), end=(0.0, 0.0), width_mm=-1.0).start == (0.0, 0.0)


def test_an_arc_track_dumps_its_mid_loads_back_and_measures_as_an_arc():
    arc = Track(net="N", layer="F.Cu", start=QUARTER[0], mid=QUARTER[1], end=QUARTER[2], width_mm=0.25, provenance=PROV)
    assert arc.is_arc and arc.mid == QUARTER[1]
    assert f'"end":[0.0,1.0],"mid":[{S2!r},{S2!r}],"width_mm":0.25' in arc.model_dump_json() and design_data(arc)["mid"] == [S2, S2]
    assert Track.model_validate_json(arc.model_dump_json()) == arc
    assert arc.length_mm == pytest.approx(math.pi / 2.0) and track_length(arc) == arc.length_mm
    assert arc.points() == arc_points(*QUARTER) == track_points(arc) and len(arc.points(0.01)) < len(arc.points())
    straight = Track(net="N", layer="F.Cu", start=(0.0, 0.0), end=(3.0, 4.0), width_mm=0.25)
    assert straight.length_mm == 5.0 and straight.points() == [(0.0, 0.0), (3.0, 4.0)]
    # the helpers take anything shaped like a track; a straight length is exactly the hypot of its ends
    assert track_length(SimpleNamespace(start=(0.1, 0.2), end=(0.7, 1.9))) == math.hypot(0.7 - 0.1, 1.9 - 0.2)
    assert track_points(SimpleNamespace(start=[1.0, 2.0], end=[3.0, 4.0])) == [(1.0, 2.0), (3.0, 4.0)]


def test_three_points_that_are_no_arc_are_refused_by_the_model():
    with pytest.raises(ValidationError, match="is no arc: its mid point lies on the line"):
        Track(net="N", layer="F.Cu", start=(0.0, 0.0), mid=(1.0, 0.0), end=(2.0, 0.0), width_mm=0.25)
    with pytest.raises(ValidationError, match="is no arc: its start and end coincide"):
        Track(net="N", layer="F.Cu", start=(0.0, 0.0), mid=(1.0, 1.0), end=(0.0, 0.0), width_mm=0.25)
    with pytest.raises(ValidationError, match="is not finite"):
        Track(net="N", layer="F.Cu", start=(0.0, 0.0), mid=(1.0, math.nan), end=(2.0, 0.0), width_mm=0.25)


def test_straight_pieces_keep_straight_tracks_and_turn_an_arc_into_its_chords():
    straight = Track(net="A", layer="F.Cu", start=(0.0, 0.0), end=(1.0, 0.0), width_mm=0.25, provenance=PROV)
    arc = Track(net="A", layer="B.Cu", start=QUARTER[0], mid=QUARTER[1], end=QUARTER[2], width_mm=0.25, provenance=PROV)
    out = straight_pieces([straight, arc, straight])
    chain = arc_points(*QUARTER)
    assert out[0] is straight and out[-1] is straight and len(out) == 2 + len(chain) - 1
    chords = out[1:-1]
    assert [(c.start, c.end) for c in chords] == list(zip(chain, chain[1:]))
    assert all(c.mid is None and c.net == "A" and c.layer == "B.Cu" and c.width_mm == 0.25 and c.provenance is arc.provenance for c in chords)
    wide = straight_pieces([arc], inflate=True)
    assert all(c.width_mm == pytest.approx(0.25 + 2.0 * ARC_SAGITTA_MM) for c in wide)
    coarse = straight_pieces([arc], inflate=True, max_sagitta=0.01)
    assert len(coarse) < len(wide) and all(c.width_mm == pytest.approx(0.27) for c in coarse)
    assert straight_pieces([]) == []


def test_the_distance_of_an_arc_from_an_area_is_measured_on_its_chain():
    area = [(0.8, 0.8), (2.0, 0.8), (2.0, 2.0), (0.8, 2.0)]  # a square beyond the quarter whose corner (0.8, 0.8) faces its bulge
    straight = SimpleNamespace(start=QUARTER[0], end=QUARTER[2])
    assert track_area_distance(straight, area) == segment_area_distance(QUARTER[0], QUARTER[2], area)
    arc = Track(net="N", layer="F.Cu", start=QUARTER[0], mid=QUARTER[1], end=QUARTER[2], width_mm=0.25)
    chord_distance = segment_area_distance(QUARTER[0], QUARTER[2], area)
    arc_distance = track_area_distance(arc, area)
    exact = point_arc_distance((0.8, 0.8), *QUARTER)  # the corner is the area's nearest point: 0.8 sqrt(2) - 1 = 0.1314 beyond the arc
    assert exact == pytest.approx(0.8 * math.sqrt(2.0) - 1.0) and chord_distance == pytest.approx(0.6 * S2)
    assert arc_distance < chord_distance - 0.05  # the bulge reaches toward the area; the chord does not
    assert exact - ARC_SAGITTA_MM - 1e-12 <= arc_distance <= exact + ARC_SAGITTA_MM + 1e-12  # the chain is within its bound of the arc
    inside = [(0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.5, 0.9)]  # the arc crosses this square (its mid lies in it); the chord only touches a corner
    assert track_area_distance(arc, inside) == 0.0
