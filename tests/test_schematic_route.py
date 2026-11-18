"""The schematic wire router (B11): pure geometry, no libraries.

Three rules are pinned here: a net whose every path is blocked is not wired at all (no half wire), a routed wire
that would touch a foreign connection point is a CompileError in the compiler's guard, and two nets crossing
mid-segment is allowed (KiCad does not connect crossing wires).
"""

from __future__ import annotations

import pytest

from ai_eda.compilers.schematic import SchematicCompiler
from ai_eda.compilers.schematic_layout import Extent, layout_channels, layout_positions
from ai_eda.compilers.schematic_route import MAX_WIRED_PINS, Channels, RoutePin, foreign_point_on_wire, route_net
from ai_eda.errors import CompileError

# one row of two resistors (pins top / bottom), channels above, between and below them
EXTENTS = {"R1": Extent(45.0, 40.0, 56.0, 62.0), "R2": Extent(70.0, 40.0, 81.0, 62.0)}
CHANNELS = Channels(vertical=((30.0, 44.0), (57.0, 69.0), (82.0, 95.0)), horizontal=((25.0, 39.0), (63.0, 77.0)))
UP = (0.0, -1.0)
DOWN = (0.0, 1.0)


def test_two_pins_facing_the_same_channel_get_a_u_shaped_wire_on_the_channel_centre_track():
    pins = [RoutePin("R1", (50.8, 41.91), UP), RoutePin("R2", (76.2, 41.91), UP)]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, [], {})
    assert routed is not None and routed.junctions == [] and routed.tracks_used == [("h", 0)]
    assert routed.segments == [((50.8, 31.75), (50.8, 41.91)), ((50.8, 31.75), (76.2, 31.75)), ((76.2, 31.75), (76.2, 41.91))]
    assert routed.length == pytest.approx(45.72)
    # deterministic: the same call gives the same wires
    again = route_net(pins, CHANNELS, EXTENTS, {}, [], {})
    assert again is not None and again.segments == routed.segments


def test_three_pins_share_a_trunk_and_the_meeting_point_is_a_junction():
    pins = [RoutePin("R1", (50.8, 41.91), UP), RoutePin("R2", (76.2, 41.91), UP), RoutePin("R2", (76.2, 60.09), DOWN)]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, [], {})
    assert routed is not None and len(routed.junctions) == 1
    ends = [p for seg in routed.segments for p in seg]
    assert ends.count(routed.junctions[0]) == 3  # three wire ends meet there
    assert all(a[0] == b[0] or a[1] == b[1] for a, b in routed.segments)  # orthogonal only


def test_a_net_with_too_many_pins_or_one_pin_is_not_wired():
    one = [RoutePin("R1", (50.8, 41.91), UP)]
    assert route_net(one, CHANNELS, EXTENTS, {}, [], {}) is None
    many = [RoutePin("R1", (50.8 + i, 41.91), UP) for i in range(MAX_WIRED_PINS + 1)]
    assert route_net(many, CHANNELS, EXTENTS, {}, [], {}) is None


def test_a_blocked_net_keeps_its_labels_rather_than_a_half_wire():
    # a foreign label anchor sits on the only straight branch of R1's top pin, and every channel is full: nothing is drawn
    pins = [RoutePin("R1", (50.8, 41.91), UP), RoutePin("R2", (76.2, 41.91), UP)]
    blocked_points = {(50.8, 35.0): "label anchor of R9.1"}
    full = {("h", 0): 99, ("h", 1): 99, ("v", 0): 99, ("v", 1): 99, ("v", 2): 99}
    assert route_net(pins, CHANNELS, EXTENTS, blocked_points, [], full) is None
    # with free channels the foreign anchor on the top path still forbids that path; the router finds another or gives up,
    # but never returns a wire through the anchor
    routed = route_net(pins, CHANNELS, EXTENTS, blocked_points, [], {})
    assert routed is None or foreign_point_on_wire(routed.segments, blocked_points) is None


def test_a_foreign_wire_end_on_the_path_blocks_it_but_a_mid_segment_crossing_does_not():
    pins = [RoutePin("R1", (50.8, 41.91), UP), RoutePin("R2", (76.2, 41.91), UP)]
    # another net's wire ends exactly on the top-channel trunk line: that candidate is refused
    foreign_end_on_trunk = [((60.0, 31.75), (60.0, 20.0))]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, foreign_end_on_trunk, {})
    assert routed is None or ((50.8, 31.75), (76.2, 31.75)) not in routed.segments
    # another net's wire crossing the trunk mid-segment (its ends well above and below the trunk) is fine
    crossing = [((60.0, 10.0), (60.0, 38.0))]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, crossing, {})
    assert routed is not None and ((50.8, 31.75), (76.2, 31.75)) in routed.segments


def test_the_compiler_guard_refuses_a_wire_through_a_foreign_point():
    tagged = [(((10.0, 10.0), (30.0, 10.0)), "VIN")]
    points = {(20.0, 10.0): "pin R9.1"}
    with pytest.raises(CompileError, match="touches pin R9.1"):
        SchematicCompiler._check_wires(tagged, points, {(20.0, 10.0): "GND"})
    # the same point belonging to the wire's own net is allowed (that is where its branches start)
    SchematicCompiler._check_wires(tagged, points, {(20.0, 10.0): "VIN"})
    # a wire end of another net on the segment is refused too
    with pytest.raises(CompileError, match="wire end of net 'GND'"):
        SchematicCompiler._check_wires(tagged + [(((20.0, 10.0), (20.0, 0.0)), "GND")], {}, {})


def test_layout_channels_follow_the_table_order_and_include_the_margins():
    extents = {"A": Extent(-5.0, -5.0, 5.0, 5.0), "B": Extent(-5.0, -5.0, 5.0, 5.0), "C": Extent(-5.0, -5.0, 5.0, 5.0)}
    order = ["C", "A", "B"]
    positions = layout_positions(extents, columns=2, order=order)
    assert list(positions) == ["C", "A", "B"] and positions["C"][0] < positions["A"][0] and positions["B"][1] > positions["C"][1]
    placed = {ref: extents[ref].shifted(*positions[ref]) for ref in order}
    vertical, horizontal = layout_channels(placed, order, 2)
    assert len(vertical) == 3 and len(horizontal) == 3  # left margin, one gutter, right margin / top, one gutter, bottom
    assert vertical[1][0] == placed["C"].xmax and vertical[1][1] == placed["A"].xmin
    assert horizontal[1][0] == max(placed["C"].ymax, placed["A"].ymax) and horizontal[1][1] == placed["B"].ymin
    with pytest.raises(ValueError):
        layout_positions(extents, order=["A", "B"])
    # B11 W3: a gutter that ran out of tracks is widened - only what lies beyond it moves; a margin corridor only gets wider
    wide = layout_positions(extents, columns=2, order=order, extra_columns=[20.32])
    assert wide["C"] == positions["C"] and wide["B"] == positions["B"] and wide["A"] == pytest.approx((positions["A"][0] + 20.32, positions["A"][1]))
    lower = layout_positions(extents, columns=2, order=order, extra_rows=[20.32, 99.0])  # an entry beyond the last gutter is ignored
    assert lower["C"] == positions["C"] and lower["A"] == positions["A"] and lower["B"] == pytest.approx((positions["B"][0], positions["B"][1] + 20.32))
    v_wide, h_wide = layout_channels(placed, order, 2, margins=(38.1, 12.7, 12.7, 38.1))
    assert v_wide[0] == pytest.approx((vertical[0][1] - 38.1, vertical[0][1])) and v_wide[1:] == vertical[1:]
    assert h_wide[-1] == pytest.approx((horizontal[-1][0], horizontal[-1][0] + 38.1)) and h_wide[:-1] == horizontal[:-1]


def test_collinear_touching_and_a_corner_on_a_foreign_wire_are_refused():
    # two more cases: what the first tests did not pin down
    pins = [RoutePin("R1", (50.8, 41.91), UP), RoutePin("R2", (76.2, 41.91), UP)]
    # (a) a foreign wire that lies on the trunk line and merely touches the trunk at one point: its end is on our trunk
    touching = [((76.2, 31.75), (90.0, 31.75))]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, touching, {})
    assert routed is None or ((50.8, 31.75), (76.2, 31.75)) not in routed.segments
    # (b) a foreign wire whose interior passes exactly through where our branch would turn (the corner at the trunk)
    corner = [((50.8, 20.0), (50.8, 35.0))]
    routed = route_net(pins, CHANNELS, EXTENTS, {}, corner, {})
    assert routed is None or all(not any(seg[0] == (50.8, 31.75) or seg[1] == (50.8, 31.75) for seg in routed.segments) for _ in [0])
    # the final guard sees both as well
    with pytest.raises(CompileError):
        SchematicCompiler._check_wires([(((50.8, 31.75), (76.2, 31.75)), "A"), (((76.2, 31.75), (90.0, 31.75)), "B")], {}, {})
    with pytest.raises(CompileError):
        SchematicCompiler._check_wires([(((50.8, 31.75), (76.2, 31.75)), "A")], {(60.0, 31.75): "pin R9.2"}, {(60.0, 31.75): "B"})
