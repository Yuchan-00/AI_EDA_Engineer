"""Corner style (routing.corners 0.1): the router's right-angle corners mitred into 45-degree chamfers after routing.

The project owner's corner-style rule (2026-11-13) - a geometry convention, not an electrical requirement - on synthetic boards:

* the pass itself (:func:`ai_eda.tools.routing.corners.mitre_corners` on hand-drawn copper over a real :class:`_Board`): an L, a U, a
  Z, an acute corner, an obtuse one left alone; a corner beside another net's pad that must shrink, one beside a hole that stays,
  one whose only legal chamfer is the smallest (the last try); a corner at a pad centre, inside a pad, at a via, on another track
  (a junction), a branch that limits the chamfer, a chamfer that would join its own net where the corner did not - at a cut
  point, or between its ends (crossing or reaching its own net's pad, track or via); a diff pair, a
  length-matched net, an escape stub and a width change skipped; a keep-out, the board edge and no room; the stamps and the stats;
* the router: ``corner_style="orthogonal"`` (the default) is byte for byte the router alone on every golden fixture, an unknown
  style is refused, ``"miter45"`` is deterministic and its copper passes the ``pcb.routing`` checks; where kicad-cli exists, the
  compiled mitred board has no clearance, edge or connection finding;
* the PCB agent: mitres on ``pcb.corners=miter45`` (its default is the arc style, routing.corners 0.2: tests/test_routing_arcs.py),
  keeps the grid corners on ``pcb.corners=orthogonal``, notes an answer it does not understand; docs/ARCHITECTURE.md names the pass
  and every ``pcb.*`` control key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ai_eda.agents.keys import CONTROL_KEYS, CORNERS_KEY
from ai_eda.agents.pcb import PCBAgent
from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.errors import CompileError
from ai_eda.ir import Provenance, ProvenanceKind, Track, ValidationStatus, Via
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.corners import (
    CORNER_BASIS,
    CORNER_ENTRY_PREFIX,
    CORNER_RULE,
    CORNERS_ID,
    CORNERS_VERSION,
    DEFAULT_CORNER_STYLE,
    MITER45,
    ORTHOGONAL,
    mitre_corners,
)
from ai_eda.tools.routing.maze import NetRule, Routing, RoutingParams, _Board, route_board
from tests.test_pcb_agent import _ctx
from tests.test_rf_floorplan import keepout
from tests.test_routing import _connected, _routing_checks, board_ir, fixture_library
from tests.test_routing_rules import FIXTURES, GOLDEN_0_2, GOLDEN_0_6, canonical, digest

P = RoutingParams()  # grid 0.25, width 0.4, clearance 0.25, edge 0.3
HAND = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand for the test")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not kicad.available(), reason="kicad-cli not installed")

#: two nets far from where the copper is drawn (their pads only make the nets exist)
PARTS = [("R1", "SMD1", 2.0, 18.0), ("R2", "SMD1", 4.0, 18.0), ("R3", "SMD1", 16.0, 18.0), ("R4", "SMD1", 18.0, 18.0)]
NETS = {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _poly(net: str, *pts: tuple[float, float], width: float = 0.4, layer: str = "F.Cu", prov: Provenance = HAND) -> list[Track]:
    return [Track(net=net, layer=layer, start=a, end=b, width_mm=width, provenance=prov) for a, b in zip(pts, pts[1:])]


def _length(tracks: list[Track]) -> float:
    return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in tracks)


def _mitre(tmp_path: Path, lib: KicadLibrary, tracks: list[Track], *, parts=None, nets=None, vias=(), rules=None, keepouts=None) -> Routing:
    """``mitre_corners`` on hand-drawn copper over the router's board of a 20 x 20 mm IR (its pads, keep-outs, outline, default params)."""
    ir = board_ir(tmp_path, lib, parts or PARTS, nets or NETS, (20.0, 20.0))
    board = _Board(ir, lib, P, keepouts=keepouts, ko_width=P.track_width_mm)
    lengths = {}
    for t in tracks:
        lengths[t.net] = lengths.get(t.net, 0.0) + math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
    r = Routing(params=P, tracks=list(tracks), vias=list(vias), rules=dict(rules or {}),
                stats={"net_length_mm": {k: round(v, 6) for k, v in lengths.items()}, "total_length_mm": round(sum(lengths.values()), 6),
                       "track_count": len(tracks)})
    mitre_corners(r, board, MITER45)
    return r


def _ends(r: Routing, net: str = "A") -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(t.start, t.end) for t in r.tracks if t.net == net]


def _angle_at(t1: Track, t2: Track) -> float:
    """The angle between two tracks at the end they share (degrees)."""
    shared = next(p for p in (t1.start, t1.end) if p in (t2.start, t2.end))
    a = t1.end if t1.start == shared else t1.start
    b = t2.end if t2.start == shared else t2.start
    v1, v2 = (a[0] - shared[0], a[1] - shared[1]), (b[0] - shared[0], b[1] - shared[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _row(r: Routing, at: tuple[float, float]) -> dict:
    return next(row for row in r.stats["corners"]["corners"] if tuple(row["at"]) == at)


# --------------------------------------------------------------------------- the chamfer


def test_an_l_gets_one_45_degree_chamfer_of_half_its_shorter_run(tmp_path: Path, lib: KicadLibrary):
    """Runs of 8 and 4 mm: half the shorter (2 mm) on each - a 45-degree segment, two 135-degree corners, the ends where they were;
    the tracks are stamped, the lengths, the track count and the stats measured again."""
    r = _mitre(tmp_path, lib, _poly("A", (4.0, 4.0), (12.0, 4.0), (12.0, 8.0)))
    assert _ends(r) == [((4.0, 4.0), (10.0, 4.0)), ((10.0, 4.0), (12.0, 6.0)), ((12.0, 6.0), (12.0, 8.0))]
    assert [round(_angle_at(a, b), 6) for a, b in zip(r.tracks, r.tracks[1:])] == [135.0, 135.0]
    entry = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {CORNERS_VERSION}:{MITER45}:at=12.0,4.0:chamfer=2.0"
    for t in r.tracks:
        assert t.provenance.derived_from == [entry] and t.width_mm == 0.4 and t.layer == "F.Cu"
        assert "routing.corners 0.1" in t.provenance.note and "a geometry convention, not an electrical requirement" in t.provenance.note
    s = r.stats
    assert s["net_length_mm"] == {"A": round(6.0 + 2.0 * math.sqrt(2.0) + 2.0, 6)} and s["total_length_mm"] == s["net_length_mm"]["A"]
    assert s["track_count"] == 3
    c = s["corners"]
    assert (c["tool"], c["version"], c["style"], c["rule"], c["basis"]) == (CORNERS_ID, CORNERS_VERSION, MITER45, CORNER_RULE, CORNER_BASIS)
    assert (c["found"], c["mitred"], c["left"], c["skipped"]) == (1, 1, 0, 0)
    assert c["corners"] == [{"net": "A", "layer": "F.Cu", "at": [12.0, 4.0], "angle_deg": 90.0, "outcome": "mitred", "chamfer_mm": 2.0}]
    assert "no noticeable effect on a 5 mil trace below about 70 GHz" in CORNER_BASIS and "no signal-integrity or manufacturing benefit" in CORNER_BASIS


def test_a_u_keeps_a_third_between_its_chamfers_and_a_z_lines_them_up(tmp_path: Path, lib: KicadLibrary):
    """A U's two corners turn the same way: each takes a third of the 3 mm bottom (two halves would meet at a right angle), so every
    corner left is 135 degrees. A Z's turn opposite ways: each takes half of the shared 3 mm, which is used up, and the two chamfers
    line up into one 45-degree run."""
    u = _mitre(tmp_path, lib, _poly("A", (4.0, 10.0), (4.0, 4.0), (7.0, 4.0), (7.0, 10.0)))
    assert _ends(u) == [((4.0, 10.0), (4.0, 5.0)), ((4.0, 5.0), (5.0, 4.0)), ((5.0, 4.0), (6.0, 4.0)), ((6.0, 4.0), (7.0, 5.0)), ((7.0, 5.0), (7.0, 10.0))]
    assert [round(_angle_at(a, b), 6) for a, b in zip(u.tracks, u.tracks[1:])] == [135.0] * 4
    assert [row["chamfer_mm"] for row in u.stats["corners"]["corners"]] == [1.0, 1.0]
    z = _mitre(tmp_path / "z", lib, _poly("A", (2.0, 4.0), (6.0, 4.0), (6.0, 7.0), (10.0, 7.0)))
    assert _ends(z) == [((2.0, 4.0), (4.5, 4.0)), ((4.5, 4.0), (6.0, 5.5)), ((6.0, 5.5), (7.5, 7.0)), ((7.5, 7.0), (10.0, 7.0))]
    assert [round(_angle_at(a, b), 6) for a, b in zip(z.tracks, z.tracks[1:])] == [135.0, 180.0, 135.0]
    assert z.stats["track_count"] == 4 and z.stats["corners"]["mitred"] == 2


def test_an_acute_corner_gets_two_obtuse_ones_and_an_obtuse_corner_is_no_corner(tmp_path: Path, lib: KicadLibrary):
    """A 36.87-degree corner (runs 8 and 5): the chamfer of 2.5 leaves two corners of (180 + 36.87) / 2 degrees; a 135-degree bend is
    not a corner of 90 degrees or sharper and is left alone (nothing found)."""
    r = _mitre(tmp_path, lib, _poly("A", (2.0, 2.0), (10.0, 2.0), (6.0, 5.0)))
    assert _ends(r) == [((2.0, 2.0), (7.5, 2.0)), ((7.5, 2.0), (8.0, 3.5)), ((8.0, 3.5), (6.0, 5.0))]
    acute = math.degrees(math.acos(0.8))
    assert _row(r, (10.0, 2.0))["angle_deg"] == round(acute, 3)
    assert [round(_angle_at(a, b), 6) for a, b in zip(r.tracks, r.tracks[1:])] == [round((180.0 + acute) / 2.0, 6)] * 2
    obtuse = _poly("A", (2.0, 2.0), (6.0, 2.0), (10.0, 6.0))
    r = _mitre(tmp_path / "o", lib, obtuse)
    assert r.tracks == obtuse and r.stats["corners"]["found"] == 0 and r.stats["track_count"] == 2


def test_a_corner_beside_another_nets_pad_shrinks_and_one_beside_a_hole_stays(tmp_path: Path, lib: KicadLibrary):
    """B's 1 x 1 mm pad inside the corner: the chamfers of 4 and 2 mm touch it, 1 mm keeps the 0.25 mm clearance (exact audit). A pad
    without a net (a 0.5 x 4 mm bar, 0.25 mm from both tracks' copper - legal) needs the clearance plus half a grid step from a chamfer,
    which no chamfer down to 0.3 track widths keeps (4, 2, ... 0.125 and last the smallest, 0.12 mm): the corner stays, with the
    reason."""
    parts = [*PARTS, ("R5", "SMD1", 8.5, 3.5)]
    nets = {"A": NETS["A"], "B": [("R3", "1"), ("R5", "1")]}
    r = _mitre(tmp_path, lib, _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0)), parts=parts, nets=nets)
    assert _ends(r) == [((2.0, 2.0), (9.0, 2.0)), ((9.0, 2.0), (10.0, 3.0)), ((10.0, 3.0), (10.0, 10.0))]
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 1.0
    hole = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    r = _mitre(tmp_path / "hole", lib, hole, parts=[*PARTS, ("W1", "VBAR", 9.3, 4.45)])
    assert r.tracks == hole and r.stats["net_length_mm"] == {"A": 16.0}
    row = _row(r, (10.0, 2.0))
    assert row["outcome"] == "left" and row["reason"] == "clearance", row
    assert row["why"].startswith("no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at (10, 2) comes")
    assert "pad W1.(unnumbered) (needs 0.375 mm)" in row["why"]
    assert r.stats["corners"]["left_reasons"] == {"clearance": 1}


def test_the_last_try_is_the_smallest_chamfer_itself(tmp_path: Path, lib: KicadLibrary):
    """A 1.0 mm rule track (clearance 0.25) with B's pad exactly 0.25 mm from both tracks' copper inside the corner: the chamfers of
    2, 1 and 0.5 mm come too near it and the next halving (0.25) is below the smallest (0.3 x 1.0 mm), so the smallest is tried - it
    keeps the clearance (a chamfer of 0.3 track widths clears what is legal against the tracks) and the corner is mitred, not left."""
    rules = {"A": NetRule(width_mm=1.0, clearance_mm=0.25)}
    parts = [*PARTS, ("R5", "SMD1", 8.75, 3.25)]  # box 8.25..9.25 x 2.75..3.75: 0.25 mm from both tracks' copper
    nets = {"A": NETS["A"], "B": [("R3", "1"), ("R5", "1")]}
    r = _mitre(tmp_path, lib, _poly("A", (6.0, 2.0), (10.0, 2.0), (10.0, 6.0), width=1.0), parts=parts, nets=nets, rules=rules)
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 0.3
    assert _ends(r) == [((6.0, 2.0), (9.7, 2.0)), ((9.7, 2.0), (10.0, 2.3)), ((10.0, 2.3), (10.0, 6.0))]
    assert "a halving below 0.3 x the track's width tries that smallest chamfer instead, the last try" in r.stats["corners"]["chamfer"]


def test_corners_at_a_pad_centre_inside_a_pad_at_a_via_or_on_a_track_are_kept(tmp_path: Path, lib: KicadLibrary):
    """Moving any of them would cut a connection or change nothing visible (a via's or a pad's copper covers it): each is skipped with
    its reason, the copper untouched."""
    l_shape = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    for parts, vias, extra, reason in (
        ([*PARTS, ("R5", "SMD1", 10.0, 2.0)], (), [], "pad_centre"),
        ([*PARTS, ("R5", "SMD1", 10.2, 2.2)], (), [], "in_pad"),
        (PARTS, (Via(net="A", x_mm=10.0, y_mm=2.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),), [], "via"),
        (PARTS, (Via(net="A", x_mm=10.2, y_mm=2.2, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),), [], "via"),
        (PARTS, (), _poly("A", (9.0, 1.0), (11.0, 3.0)), "junction"),
    ):
        nets = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]} if any(p[0] == "R5" for p in parts) else NETS
        tracks = [*l_shape, *extra]
        r = _mitre(tmp_path / f"{reason}{len(vias) and vias[0].x_mm}", lib, tracks, parts=parts, nets=nets, vias=vias)
        assert r.tracks == tracks, reason
        row = _row(r, (10.0, 2.0))
        assert (row["outcome"], row["reason"]) == ("skipped", reason), row
        if reason == "in_pad":
            assert row["why"] == "inside the box of pad R5.1 of its net: the pad's copper covers the corner"


def test_a_branch_on_a_track_limits_the_chamfer_and_stays_on_it(tmp_path: Path, lib: KicadLibrary):
    """A track of the net ends 1 mm before the corner (a T-junction): the chamfer takes half of that, 0.5 mm, and the junction point
    stays on the shortened track; a T-junction where three tracks end is no corner at all."""
    branch = _poly("A", (9.0, 2.0), (9.0, 0.8))
    r = _mitre(tmp_path, lib, [*_poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0)), *branch])
    assert _ends(r)[:3] == [((2.0, 2.0), (9.5, 2.0)), ((9.5, 2.0), (10.0, 2.5)), ((10.0, 2.5), (10.0, 10.0))]
    assert r.tracks[3:] == branch and _row(r, (10.0, 2.0))["chamfer_mm"] == 0.5
    tee = _poly("A", (2.0, 2.0), (6.0, 2.0)) + _poly("A", (6.0, 2.0), (10.0, 2.0)) + _poly("A", (6.0, 2.0), (6.0, 8.0))
    r = _mitre(tmp_path / "tee", lib, tee)
    assert r.tracks == tee and r.stats["corners"]["found"] == 0


def test_a_chamfer_never_joins_its_own_net_where_the_corner_did_not(tmp_path: Path, lib: KicadLibrary):
    """The audit ignores the net's own copper, so the net's copper graph is guarded on its own: the 4 mm chamfer's cut point would land
    in the box of A's own pad R5 (the track passes over it, its far end does not), and a 4 mm chamfer would pass through the end of
    another A track - each refused ("own_copper"), the 2 mm chamfer taken."""
    l_shape = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    parts = [*PARTS, ("R5", "SMD1", 6.0, 2.3)]
    r = _mitre(tmp_path, lib, l_shape, parts=parts, nets={"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]})
    assert _ends(r) == [((2.0, 2.0), (8.0, 2.0)), ((8.0, 2.0), (10.0, 4.0)), ((10.0, 4.0), (10.0, 10.0))]
    stub = _poly("A", (8.0, 4.0), (8.0, 6.0))
    r = _mitre(tmp_path / "end", lib, [*l_shape, *stub])
    assert _ends(r) == [((2.0, 2.0), (8.0, 2.0)), ((8.0, 2.0), (10.0, 4.0)), ((10.0, 4.0), (10.0, 10.0)), ((8.0, 4.0), (8.0, 6.0))]
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 2.0


def _diagonal(r: Routing, at: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    entry = f"at={at[0]!r},{at[1]!r}:"
    return next((t.start, t.end) for t in r.tracks if t.start[0] != t.end[0] and t.start[1] != t.end[1]
                and any(e.startswith(CORNER_ENTRY_PREFIX) and entry in e for e in t.provenance.derived_from))


def test_a_chamfer_never_crosses_or_reaches_its_own_nets_copper_between_its_ends(tmp_path: Path, lib: KicadLibrary):
    """The chamfer's interior is guarded too, not only its two cut points: the copper would make a loop the SI path model (nodes at
    track ends only) does not see.

    * A's pad R5 lies inside the corner, clear of both tracks' copper, and connected by its own track that tees into the L: the 3 mm
      chamfer's centreline runs through its box (both cut points outside it) - refused, 1.5 mm taken.
    * A's branch from the L down to its pad R5 at (8.5, 3): the 3 mm chamfer crosses the branch's track and the 1.5 mm one runs
      through R5's box corner - both refused, 0.75 mm taken.
    * R5 at (9.15, 3) beside an L of runs 8 and 6 mm: the 3 mm and 0.75 mm chamfers pass 0.11 and 0.07 mm from its box (inside half
      the chamfer's width, the centreline outside) and the 1.5 mm one crosses it - all refused, 0.375 mm taken.
    * an A track along the corner's bisector ending 0.05 mm short of the corner: every chamfer from 4 mm down to the smallest
      crosses it - the corner is left ("own_copper"), the copper as drawn."""
    l_shape = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    nets = {"A": [("R1", "1"), ("R5", "1")], "B": NETS["B"]}
    tee = _poly("A", (8.2, 3.4), (4.0, 3.4), (4.0, 2.0))
    r = _mitre(tmp_path / "pad", lib, [*l_shape, *tee], parts=[*PARTS, ("R5", "SMD1", 8.2, 3.4)], nets=nets)
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 1.5 and _diagonal(r, (10.0, 2.0)) == ((8.5, 2.0), (10.0, 3.5))
    branch = _poly("A", (10.0, 8.0), (8.5, 8.0), (8.5, 3.0))
    r = _mitre(tmp_path / "branch", lib, [*l_shape, *branch], parts=[*PARTS, ("R5", "SMD1", 8.5, 3.0)], nets=nets)
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 0.75 and _diagonal(r, (10.0, 2.0)) == ((9.25, 2.0), (10.0, 2.75))
    r = _mitre(tmp_path / "reach", lib, _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 8.0)), parts=[*PARTS, ("R5", "SMD1", 9.15, 3.0)], nets=nets)
    assert _row(r, (10.0, 2.0))["chamfer_mm"] == 0.375
    bisector = _poly("A", (6.0, 6.0), (9.95, 2.05))
    r = _mitre(tmp_path / "cross", lib, [*l_shape, *bisector])
    row = _row(r, (10.0, 2.0))
    assert r.tracks == [*l_shape, *bisector] and (row["outcome"], row["reason"]) == ("left", "own_copper"), row
    assert row["why"] == ("no chamfer from 4.0000 mm down to 0.1200 mm keeps every rule: the chamfer at (10, 2) would meet its own net's track "
                          "(6, 6)-(9.95, 2.05) (a new junction)")


def test_pairs_matched_nets_escape_stubs_and_width_changes_are_skipped(tmp_path: Path, lib: KicadLibrary):
    """Copper another mechanism fixed keeps its corners: the pair emission's mitres and coupling, a length-matched group's skew, an
    escape stub's recorded points, a neck-down's width step."""
    l_shape = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    pair = {"A": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="B", pair_spacing_mm=0.25),
            "B": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="A", pair_spacing_mm=0.25)}
    group = {"A": NetRule(match_group="BUS", max_skew_mm=0.5)}
    escape = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.6", note="escape",
                        derived_from=["net:A", "escape:R1.1:stub:width=0.4:points=2.0,2.0;10.0,2.0"])
    cases = (
        (l_shape, pair, "pair", "a differential pair's track"),
        (l_shape, group, "match_group", "a net of length-matched group BUS"),
        ([*_poly("A", (2.0, 2.0), (10.0, 2.0), prov=escape), l_shape[1]], None, "escape", "a segment of an escape stub"),
        ([l_shape[0], *_poly("A", (10.0, 2.0), (10.0, 10.0), width=0.3)], None, "width", "the two tracks differ in width (0.4 / 0.3 mm"),
    )
    for tracks, rules, reason, words in cases:
        r = _mitre(tmp_path / reason, lib, tracks, rules=rules)
        row = _row(r, (10.0, 2.0))
        assert r.tracks == tracks and (row["outcome"], row["reason"]) == ("skipped", reason) and row["why"].startswith(words), row
        assert r.stats["corners"]["skipped_reasons"] == {reason: 1}


def test_a_keep_out_the_edge_and_short_tracks_leave_the_corner(tmp_path: Path, lib: KicadLibrary):
    """A track keep-out touching both tracks' copper inside the corner refuses every chamfer (unless it allows the net); a corner on
    copper already at the edge clearance's limit cannot cut nearer; a run whose half is below 0.3 track widths leaves no room."""
    area = keepout("K1", [6.8, 2.2, 3.0, 3.0], forbids=["tracks"], layers=["F.Cu"])
    l_shape = _poly("A", (2.0, 2.0), (10.0, 2.0), (10.0, 10.0))
    r = _mitre(tmp_path, lib, l_shape, keepouts=[area])
    row = _row(r, (10.0, 2.0))
    assert r.tracks == l_shape and (row["outcome"], row["reason"]) == ("left", "keepout") and "track keep-out K1" in row["why"]
    allowed = keepout("K1", [6.8, 2.2, 3.0, 3.0], forbids=["tracks"], layers=["F.Cu"], nets=["A"])
    assert _row(_mitre(tmp_path / "allowed", lib, l_shape, keepouts=[allowed]), (10.0, 2.0))["outcome"] == "mitred"
    edge = _poly("A", (2.0, 0.3), (10.0, 0.3), (10.0, 8.0))
    row = _row(_mitre(tmp_path / "edge", lib, edge), (10.0, 0.3))
    assert (row["outcome"], row["reason"]) == ("left", "edge") and "board edge" in row["why"]
    short = _poly("A", (9.8, 2.0), (10.0, 2.0), (10.0, 8.0))
    row = _row(_mitre(tmp_path / "short", lib, short), (10.0, 2.0))
    assert (row["outcome"], row["reason"]) == ("left", "no_room") and "0.1000 mm, below the smallest (0.12 mm: 0.3 x the track's width)" in row["why"]
    jog = _poly("A", (9.75, 2.0), (10.0, 2.0), (10.0, 8.0))  # half of 0.25 mm is 0.125 mm: just above the smallest
    assert _row(_mitre(tmp_path / "jog", lib, jog), (10.0, 2.0))["chamfer_mm"] == 0.125


# --------------------------------------------------------------------------- the router


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_orthogonal_is_the_router_alone_on_every_golden_fixture(tmp_path: Path, name: str):
    """``corner_style="orthogonal"`` (the default) is routing.maze's copper, provenance and stats byte for byte (GOLDEN_0_2 / 0_6)."""
    ir, flib, params = FIXTURES[name](tmp_path)
    assert digest(route_board(ir, flib, params, corner_style=ORTHOGONAL)) == GOLDEN_0_6.get(name, GOLDEN_0_2[name]), name


def _l_board(tmp_path: Path, lib: KicadLibrary):
    """Two pads diagonally apart, B.Cu covered by a net-less plane pad: one net routed as an L on F.Cu."""
    return board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 8.0), ("Z1", "BPLANE", 6.0, 5.0)],
                    {"A": [("R1", "1"), ("R2", "1")]}, (12.0, 11.0))


def test_the_router_mitres_on_request_deterministically_and_refuses_an_unknown_style(tmp_path: Path, lib: KicadLibrary):
    ir = _l_board(tmp_path, lib)
    plain = route_board(ir, lib)
    assert canonical(route_board(ir, lib, corner_style=ORTHOGONAL)) == canonical(plain) and "corners" not in plain.stats
    a, b = route_board(ir, lib, corner_style=MITER45), route_board(ir, lib, corner_style=MITER45)
    assert canonical(a) == canonical(b)
    assert [(t.start, t.end) for t in plain.tracks] == [((3.0, 3.0), (9.0, 3.0)), ((9.0, 3.0), (9.0, 8.0))]
    assert [(t.start, t.end) for t in a.tracks] == [((3.0, 3.0), (6.5, 3.0)), ((6.5, 3.0), (9.0, 5.5)), ((9.0, 5.5), (9.0, 8.0))]
    # the router's own provenance stays (tool, version, every entry), the corner entry follows
    for t in a.tracks:
        assert t.provenance.tool == "routing.maze" and t.provenance.tool_version == plain.version
        assert t.provenance.derived_from[:-1] == plain.tracks[0].provenance.derived_from and t.provenance.derived_from[-1].startswith(CORNER_ENTRY_PREFIX)
    assert a.stats["net_length_mm"]["A"] < plain.stats["net_length_mm"]["A"] and a.stats["corners"]["mitred"] == 1
    assert _connected(a.tracks, [(3.0, 3.0), (9.0, 8.0)])
    checks = _routing_checks(ir, a, lib, tmp_path)
    assert checks["pcb.routing.clearance"].status == ValidationStatus.PASS and checks["pcb.routing.connectivity"].status == ValidationStatus.PASS, checks
    with pytest.raises(CompileError, match="corner_style 'round' is not one of"):
        route_board(ir, lib, corner_style="round")


@needs_kicad
def test_kicad_finds_no_clearance_edge_or_connection_problem_on_a_mitred_board(tmp_path: Path, lib: KicadLibrary):
    """Canary (kicad-cli): two two-pad parts, two nets routed with mitred corners and compiled - DRC reports no clearance, edge, hole,
    short or connection finding."""
    ir = board_ir(tmp_path, lib, [("C1", "SMD2", 4.0, 3.0), ("R1", "SMD2", 10.0, 9.0)],
                  {"A": [("C1", "2"), ("R1", "1")], "B": [("C1", "1"), ("R1", "2")]}, (14.0, 12.0))
    r = route_board(ir, lib, corner_style=MITER45)
    assert r.unrouted == {} and r.stats["corners"]["mitred"] >= 2, r.stats["corners"]
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    out = tmp_path / "out"
    art = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": lib}))
    kicad.run_drc(Path(art.path), out / "drc.json")
    report = json.loads((out / "drc.json").read_text(encoding="utf-8"))
    bad = {"clearance", "copper_edge_clearance", "hole_clearance", "shorting_items", "track_dangling", "tracks_crossing"}
    assert [v for v in report["violations"] if v["type"] in bad] == [] and report["unconnected_items"] == []


# --------------------------------------------------------------------------- the PCB agent


def test_the_pcb_agent_mitres_on_request_and_keeps_grid_corners_on_request(tmp_path: Path, lib: KicadLibrary):
    """``pcb.corners=miter45`` (a control key) gives ``route_board(..., corner_style="miter45")``'s copper and the routing description
    counts the corners; ``pcb.corners=orthogonal`` gives the router's grid corners with a note; another answer is noted as not
    understood and the agent's default style (the owner's rule, rounded corners since routing.corners 0.2: tests/test_routing_arcs.py)
    applies."""
    ir = _l_board(tmp_path, lib)
    unrouted = ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"tracks": [], "vias": []})})
    assert CORNERS_KEY == "pcb.corners" and CORNERS_KEY in CONTROL_KEYS
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: " Miter45"}))
    payload = res.proposals[0].payload
    mitred = route_board(unrouted, lib, corner_style=MITER45)
    assert [(t.start, t.end) for t in payload.tracks] == [(t.start, t.end) for t in mitred.tracks] and all(t.mid is None for t in payload.tracks)
    assert any("corners: 1 of 1 corner(s) of 90 degrees or sharper mitred into 45-degree chamfers (routing.corners 0.1" in n for n in res.notes), res.notes
    assert not any("not understood" in n for n in res.notes)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: "Orthogonal "}))
    plain = route_board(unrouted, lib)
    assert [(t.start, t.end) for t in res.proposals[0].payload.tracks] == [(t.start, t.end) for t in plain.tracks]
    assert "corners: pcb.corners=orthogonal - the router's right-angle grid corners are kept (routing.corners does not run)" in res.notes
    assert not any("mitred into 45-degree chamfers" in n for n in res.notes)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: "round"}))
    assert any(n.startswith("pcb.corners='round' not understood (the only answers are 'arc', 'miter45' and 'orthogonal')") for n in res.notes), res.notes
    default = route_board(unrouted, lib, corner_style=DEFAULT_CORNER_STYLE)
    assert [(t.start, t.mid, t.end) for t in res.proposals[0].payload.tracks] == [(t.start, t.mid, t.end) for t in default.tracks]


def test_the_architecture_document_names_the_corner_pass_and_every_pcb_control_key():
    """docs/ARCHITECTURE.md (where CLAUDE.md sends a reader for the invariants) describes the agent's corner pass and its off switch:
    every ``pcb.*`` control key with its answer, routing.corners 0.1 in the PCB agent's paragraph and in its flow line - there as the
    ``pcb.corners=miter45`` answer, after the default arc style (routing.corners 0.2)."""
    doc = (Path(__file__).resolve().parents[1] / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    pcb_keys = sorted(k for k in CONTROL_KEYS if k.startswith("pcb."))
    assert pcb_keys == ["pcb.corners", "pcb.placement", "pcb.placement_policy", "pcb.routing", "pcb.silkscreen"]
    assert [k for k in pcb_keys if f"`{k}=" not in doc] == []
    assert "`pcb.corners=orthogonal`" in doc and "`pcb.corners=miter45`" in doc and "`routing.corners` 0.1(`tools/routing/corners.py`)" in doc
    assert "routing.maze + 마지막에 routing.corners 0.2 arc" in doc and "pcb.corners=miter45이면 routing.corners 0.1 miter45의 45도 모따기" in doc
    assert "기하 관례이지 전기적 요구가 아님" in doc
