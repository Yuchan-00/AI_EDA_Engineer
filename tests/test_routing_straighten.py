"""Corner style ``arc`` (routing.corners 0.2), the straightening: the router's sidesteps removed before the corners are rounded.

A sidestep is a short track between two right-angle corners - what a grid search leaves, not a pattern anyone asked for. Under the
arc style the pass slides it along one of its neighbours to that neighbour's far corner and joins it with the track that leaves
there, wherever the copper may legally run so. On hand-drawn copper over a real :class:`_Board` (as ``tests/test_routing_arcs.py``
draws it):

* the move: a sidestep beside a turn, in both directions and drawn from either pad; a sidestep that may slide either way (fewer
  corners, then less copper, then the track order decide); a bump that becomes a straight line, one with unequal steps, a
  staircase that collapses; a U that slides as far as its shorter side; the length that makes a track a sidestep;
* what stops it: a point the pass keeps (a pad - its centre or a corner inside its box -, a via, a junction, a track end on the
  copper it would take away), another net's via or track (the result is measured again here; also copper another net's move laid
  or took away just before), a keep-out, the board edge, the net's own pad or via it would newly touch or leave, a via it would
  newly pass through; pairs, length-matched nets, escape stubs, width changes and neck-down copper stay as routed;
* the records: the entries (also of a straightened track the rounding then uses up), the notes, the stats (which add up), the
  lengths, the track order, the bound on the sweeps;
* what must not change: ``"miter45"`` and ``"orthogonal"`` do not straighten, and the arc style on copper without a sidestep it
  can move gives the bytes it gave before the step existed (:data:`ARC_BEFORE_STRAIGHTENING`);
* the stage report's sentence, the PCB agent's description and the architecture document;
* where kicad-cli exists: the compiled board of straightened copper has no clearance, edge or connection finding.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from ai_eda.agents.keys import CORNERS_KEY
from ai_eda.agents.pcb import PCBAgent
from ai_eda.compilers import CompileContext, PCBCompiler
from ai_eda.ir import Track, ValidationStatus, Via, straight_pieces, track_length
from ai_eda.ir.arc import ARC_SAGITTA_MM, fillet
from ai_eda.report import stages
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import corners
from ai_eda.tools.routing.corners import (
    ARC,
    ARC_RADIUS_WIDTHS,
    CORNERS_ARC_VERSION,
    MITER45,
    ORTHOGONAL,
    SIDESTEP_SWEEPS,
    SIDESTEP_WIDTHS,
    STRAIGHTEN,
)
from ai_eda.tools.routing.maze import NetRule, Routing, route_board
from ai_eda.tools.si.paths import longest_path, net_pads
from tests.test_arc_reports import ARC_02_A, ARC_02_B, MITRE_02, SENTENCE_02, _arc_board, _cornered
from tests.test_pcb_agent import _ctx
from tests.test_rf_floorplan import keepout
from tests.test_routing import _routing_checks, board_ir, fixture_library
from tests.test_routing_arcs import CAP, P, W, _arcs, _copper_gap_to_disc, _fixtures, _grouping, _pass, _rows, _shape, _tangent_everywhere, _track_ends
from tests.test_routing_corners import HAND, _l_board, _poly, kicad, needs_kicad
from tests.test_routing_rules import canonical

Point = tuple[float, float]

#: the two nets of every board here: A's copper is drawn between its pads R1 and R2, B's pads lie in a corner out of the way
STEP_NETS = {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}
#: the keys the straightening adds to ``stats["corners"]``, after the ones the arc style had
ADDED = ["straighten", "sidesteps", "straightened", "sidesteps_left", "sidestep_reasons", "sidestep_rows"]
#: a sidestep beside a turn: from R1 3 mm up, 0.2 mm aside, 7 mm up, then the turn and 9.8 mm along to R2
BESIDE_A_TURN: tuple[Point, ...] = ((4.0, 16.0), (4.0, 13.0), (4.2, 13.0), (4.2, 6.0), (14.0, 6.0))


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _parts(start: Point, end: Point, *more: tuple) -> list[tuple]:
    """R1 at the copper's first point and R2 at its last (1 x 1 mm pads), B's two pads in a corner, then ``more``."""
    return [("R1", "SMD1", *start), ("R2", "SMD1", *end), ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8), *more]


def _run(tmp_path: Path, lib: KicadLibrary, pts: tuple[Point, ...], *more: Track, style: str = ARC, parts: tuple = (), **extra) -> Routing:
    """The pass in ``style`` on net A's copper along ``pts`` (pad R1 at the first point, pad R2 at the last) and the tracks ``more``."""
    return _pass(tmp_path, lib, [*_poly("A", *pts), *more], style=style, parts=_parts(pts[0], pts[-1], *parts), nets=extra.pop("nets", STEP_NETS), **extra)


def _watch(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Point, Point]]:
    """Make the pass show the copper as the straightening leaves it, before any corner is rounded: ``(net, start, end)`` of every
    track, in track order, read at the point where the step hands the copper on (the list is filled by the next run of the pass)."""
    seen: list[tuple[str, Point, Point]] = []
    real = corners._straighten

    def watching(groups, *rest):
        out = real(groups, *rest)
        seen[:] = [(net, a, b) for _order, net, a, b in sorted((s.order, s.track.net, s.a, s.b) for group in groups.values() for s in group if s.alive)]
        return out

    monkeypatch.setattr(corners, "_straighten", watching)
    return seen


def _path(seen: list[tuple[str, Point, Point]], net: str = "A") -> list[tuple[Point, Point]]:
    return [(a, b) for name, a, b in seen if name == net]


def _steps(r: Routing) -> list[dict]:
    return r.stats["corners"]["sidestep_rows"]


def _counts(r: Routing) -> tuple[int, int, int, dict]:
    c = r.stats["corners"]
    return c["sidesteps"], c["straightened"], c["sidesteps_left"], c["sidestep_reasons"]


def _entry(at: Point, step: float) -> str:
    return f"corner:routing.corners 0.2:straighten:at={at[0]!r},{at[1]!r}:step={step!r}"


def _point_segment(q: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(q[0] - a[0] - t * dx, q[1] - a[1] - t * dy)


def _gap_to_track(r: Routing, other: Track, net: str = "A") -> float:
    """The smallest copper gap between the net's tracks and a straight track of another net, measured here: every track as its chords
    (an arc's chain, the chain's bound taken off), the ends of each against the other - exact for segments that do not cross."""
    best = math.inf
    for t in r.tracks:
        if t.net != net:
            continue
        bound = ARC_SAGITTA_MM if t.mid is not None else 0.0
        for piece in straight_pieces([t]):
            d = min(_point_segment(piece.start, other.start, other.end), _point_segment(piece.end, other.start, other.end),
                    _point_segment(other.start, piece.start, piece.end), _point_segment(other.end, piece.start, piece.end))
            best = min(best, d - t.width_mm / 2.0 - other.width_mm / 2.0 - bound)
    return best


# --------------------------------------------------------------------------- the move


def test_a_sidestep_beside_a_turn_is_slid_to_the_turn_and_the_one_corner_left_gets_the_cap_radius(tmp_path: Path, lib: KicadLibrary,
                                                                                                    monkeypatch: pytest.MonkeyPatch):
    """Pad - 3 mm up - 0.2 mm aside - 7 mm up - turn - 9.8 mm along - pad. The sidestep cannot slide toward the pad (the pass keeps
    it), so it slides along the 7 mm track to the turn and joins the track that leaves it: pad - 10 mm up - turn - 10 mm along - pad,
    three corners become one, the copper is as long as it was, and the one corner is then rounded at the cap (4 widths) - where the
    sidestep's two corners, 0.2 mm apart, had room for neither an arc nor a chamfer."""
    seen = _watch(monkeypatch)
    r = _run(tmp_path, lib, BESIDE_A_TURN)
    assert _path(seen) == [((4.0, 16.0), (4.0, 6.0)), ((4.0, 6.0), (14.0, 6.0))]
    a, mid, b, radius = fillet((4.0, 6.0), (4.0, 16.0), (14.0, 6.0), CAP)
    assert _shape(r) == [((4.0, 16.0), a), (a, mid, b), (b, (14.0, 6.0))] and (a, b) == ((4.0, 7.6), (5.6, 6.0)) and radius == pytest.approx(CAP)
    _tangent_everywhere(r)
    c = r.stats["corners"]
    assert _counts(r) == (1, 1, 0, {}) and (c["found"], c["arcs"], c["mitred"], c["left"], c["skipped"]) == (1, 1, 0, 0, 0)
    assert _steps(r) == [{"net": "A", "layer": "F.Cu", "at": [4.1, 13.0], "step_mm": 0.2, "outcome": "straightened", "slid_to": [4.2, 6.0],
                          "corners_removed": 2, "shorter_mm": 0.0}]
    assert _rows(r) == [{"net": "A", "layer": "F.Cu", "at": [4.0, 6.0], "angle_deg": 90.0, "outcome": "arc", "radius_mm": 1.6, "tangent_mm": 1.6}]
    # the entries: the move on the two tracks it changed, then the corner's; the arc itself was made after the move
    move, arc = _entry((4.1, 13.0), 0.2), "corner:routing.corners 0.2:arc:at=4.0,6.0:tangent=1.6:radius=1.6"
    assert move == f"corner:routing.corners {CORNERS_ARC_VERSION}:{STRAIGHTEN}:at=4.1,13.0:step=0.2"
    assert [t.provenance.derived_from for t in r.tracks] == [[move, arc], [arc], [move, arc]]
    for t in (r.tracks[0], r.tracks[2]):
        assert ("routing.corners 0.2: 1 sidestep(s) of the router on this net straightened, then 1 corner(s) of 90 degrees or sharper of this net "
                "rounded into a tangent arc, or mitred into a 45-degree chamfer where no arc fits") in t.provenance.note
        assert t.provenance.note.endswith("a geometry convention, not an electrical requirement: no signal-integrity or manufacturing benefit is claimed")
    assert "sidestep" not in r.tracks[1].provenance.note
    # a Z keeps the net's length: 20 mm as routed and as straightened; the arc then shortens it
    assert sum(track_length(t) for t in _poly("A", *BESIDE_A_TURN)) == pytest.approx(20.0)
    assert r.stats["net_length_mm"]["A"] == round(sum(track_length(t) for t in r.tracks), 6) == pytest.approx(8.4 + CAP * math.pi / 2.0 + 8.4, abs=2e-6)
    assert r.stats["track_count"] == 3 and r.stats["total_length_mm"] == r.stats["net_length_mm"]["A"]


#: a sidestep beside a turn in its four shapes: the points as routed, the points as straightened, how much copper the move removes
#: (a turn that continues the sidestep keeps the length; one that runs back over it loses the sidestep twice)
TURNS: dict[str, tuple[tuple[Point, ...], tuple[Point, ...], float]] = {
    "aside to the right, turn to the right": (BESIDE_A_TURN, ((4.0, 16.0), (4.0, 6.0), (14.0, 6.0)), 0.0),
    "aside to the right, turn to the left": (((10.0, 16.0), (10.0, 13.0), (10.2, 13.0), (10.2, 6.0), (3.0, 6.0)), ((10.0, 16.0), (10.0, 6.0), (3.0, 6.0)), 0.4),
    "aside to the left, turn to the left": (((10.2, 16.0), (10.2, 13.0), (10.0, 13.0), (10.0, 6.0), (3.0, 6.0)), ((10.2, 16.0), (10.2, 6.0), (3.0, 6.0)), 0.0),
    "aside to the left, turn to the right": (((4.2, 16.0), (4.2, 13.0), (4.0, 13.0), (4.0, 6.0), (14.0, 6.0)), ((4.2, 16.0), (4.2, 6.0), (14.0, 6.0)), 0.4),
}


@pytest.mark.parametrize("shape", sorted(TURNS))
@pytest.mark.parametrize("drawn", ["from the pad below", "from the pad at the end of the run"])
def test_the_move_is_the_same_in_every_direction_and_from_either_end(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch, shape: str,
                                                                    drawn: str):
    """The sidestep to the right or to the left, the turn continuing it or running back over it, and the copper drawn from either
    pad (which swaps the roles of the two neighbours: the mirrored move). Each time the sidestep ends at the turn: one corner
    left, at the cap radius, the copper never longer - by twice the sidestep shorter where the track after the turn ran back over
    it."""
    routed, straightened, shorter = TURNS[shape]
    if drawn != "from the pad below":
        routed, straightened = routed[::-1], straightened[::-1]
    seen = _watch(monkeypatch)
    r = _run(tmp_path, lib, routed)
    assert _path(seen) == list(zip(straightened, straightened[1:])), (shape, drawn)
    assert _counts(r) == (1, 1, 0, {})
    row, = _steps(r)
    assert (row["outcome"], row["step_mm"], row["corners_removed"], row["shorter_mm"]) == ("straightened", 0.2, 2, shorter)
    corner = straightened[1]
    a, mid, b, _radius = fillet(corner, straightened[0], straightened[2], CAP)
    assert _shape(r) == [(straightened[0], a), (a, mid, b), (b, straightened[2])]
    assert _rows(r) == [{"net": "A", "layer": "F.Cu", "at": [corner[0], corner[1]], "angle_deg": 90.0, "outcome": "arc", "radius_mm": 1.6, "tangent_mm": 1.6}]
    _tangent_everywhere(r)
    before = sum(track_length(t) for t in _poly("A", *routed))
    assert sum(track_length(t) for t in _poly("A", *straightened)) == pytest.approx(before - shorter)


def test_a_sidestep_that_may_slide_either_way_takes_the_fewer_corners_then_the_less_copper_then_the_track_order(tmp_path: Path, lib: KicadLibrary,
                                                                                                                monkeypatch: pytest.MonkeyPatch):
    """A sidestep between two turns may slide to either. Three boards, the copper drawn pad - along - turn - 2 mm up - 0.2 mm aside
    - 4 mm up - turn - along - pad:

    * both tracks beyond the turns continue the sidestep: either move leaves two corners and the same copper, so the track order
      decides - the sidestep slides along the neighbour drawn first (the 2 mm one), back to the first turn;
    * the track beyond the second turn runs back over the sidestep: sliding there leaves 0.4 mm less copper, so the sidestep slides
      along the 4 mm neighbour although that one was drawn later;
    * on a third board the copper steps back by the same 0.2 mm, 2 mm beyond the sidestep, and runs on to its pad (a bump): sliding
      there removes four corners, sliding to the first turn only two - the fewer corners win, before copper and order are looked
      at."""
    seen = _watch(monkeypatch)
    order = _run(tmp_path, lib, BOARDS["either_way"])
    assert _path(seen) == [((3.0, 10.0), (6.2, 10.0)), ((6.2, 10.0), (6.2, 4.0)), ((6.2, 4.0), (12.0, 4.0))]
    assert _steps(order) == [{"net": "A", "layer": "F.Cu", "at": [6.1, 8.0], "step_mm": 0.2, "outcome": "straightened", "slid_to": [6.0, 10.0],
                              "corners_removed": 2, "shorter_mm": 0.0}]
    assert [row["radius_mm"] for row in _rows(order)] == [CAP, CAP]  # a Z again, but one with 6 mm between its corners
    copper = _run(tmp_path / "copper", lib, ((3.0, 10.0), (6.0, 10.0), (6.0, 8.0), (6.2, 8.0), (6.2, 4.0), (2.0, 4.0)))
    assert _path(seen) == [((3.0, 10.0), (6.0, 10.0)), ((6.0, 10.0), (6.0, 4.0)), ((6.0, 4.0), (2.0, 4.0))]
    row, = _steps(copper)
    assert (row["slid_to"], row["corners_removed"], row["shorter_mm"]) == ([6.2, 4.0], 2, 0.4)
    fewer = _run(tmp_path / "fewer", lib, ((3.0, 10.0), (6.0, 10.0), (6.0, 8.0), (6.2, 8.0), (6.2, 6.0), (6.0, 6.0), (6.0, 2.0)))
    assert _path(seen) == [((3.0, 10.0), (6.0, 10.0)), ((6.0, 10.0), (6.0, 2.0))]
    row, = _steps(fewer)
    assert (row["slid_to"], row["corners_removed"], row["shorter_mm"]) == ([6.2, 6.0], 4, 0.4) and _counts(fewer) == (1, 1, 0, {})


def test_a_bump_becomes_a_straight_track(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Aside by 0.2 mm, 1 mm on, and back by the same 0.2 mm: the first sidestep slides along the bump's top to the second, the two
    cancel, and the tracks before and after the bump - now on one line - are one track from pad to pad. Four corners and 0.4 mm of
    copper are gone with one move; the second sidestep disappears with it and has no row of its own. Nothing is left to round; the
    one track keeps the place and the direction of the first, and its note names the move alone."""
    seen = _watch(monkeypatch)
    routed = ((10.0, 16.0), (10.0, 12.0), (10.2, 12.0), (10.2, 11.0), (10.0, 11.0), (10.0, 5.0))
    r = _run(tmp_path, lib, routed)
    assert _path(seen) == [((10.0, 16.0), (10.0, 5.0))] == _shape(r)
    assert _counts(r) == (1, 1, 0, {}) and r.stats["corners"]["found"] == 0 and r.stats["track_count"] == 1
    assert _steps(r) == [{"net": "A", "layer": "F.Cu", "at": [10.1, 12.0], "step_mm": 0.2, "outcome": "straightened", "slid_to": [10.2, 11.0],
                          "corners_removed": 4, "shorter_mm": 0.4}]
    track, = r.tracks
    assert track.provenance.derived_from == [_entry((10.1, 12.0), 0.2)] and track.mid is None and track.width_mm == 0.4
    assert track.provenance.note == ("drawn by hand for the test; routing.corners 0.2: 1 sidestep(s) of the router on this net straightened (every clearance "
                                     "checked exactly; corner entries in derived_from) - the project owner's corner-style rule (2026-11-13), a geometry "
                                     "convention, not an electrical requirement: no signal-integrity or manufacturing benefit is claimed")
    assert r.stats["net_length_mm"] == {"A": 11.0} and sum(track_length(t) for t in _poly("A", *routed)) == pytest.approx(11.4)


def test_a_bump_with_unequal_steps_becomes_one_sidestep_or_is_merged_away(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Aside by 0.2 mm and back by 0.5 mm. The shorter sidestep slides along the bump's top onto the longer one, which runs back over
    it: their difference, 0.3 mm, is left - one sidestep. Between two pads it has nowhere to slide and stays, counted with its
    reason; beside a turn it slides on, in the next sweep, and is merged away."""
    seen = _watch(monkeypatch)
    bump = ((10.0, 16.0), (10.0, 12.0), (10.2, 12.0), (10.2, 11.0), (9.7, 11.0))
    r = _run(tmp_path, lib, (*bump, (9.7, 5.0)))
    assert _path(seen) == [((10.0, 16.0), (10.0, 11.0)), ((10.0, 11.0), (9.7, 11.0)), ((9.7, 11.0), (9.7, 5.0))]
    assert _counts(r) == (2, 1, 1, {"kept_point": 1})
    slid, stays = _steps(r)
    assert (slid["at"], slid["step_mm"], slid["outcome"], slid["corners_removed"], slid["shorter_mm"]) == ([10.1, 12.0], 0.2, "straightened", 2, 0.4)
    assert (stays["at"], stays["step_mm"], stays["outcome"], stays["reason"]) == ([9.85, 11.0], 0.3, "left", "kept_point")
    assert stays["why"] == ("toward (9.7, 5): the pass keeps that point: the centre of pad R2.1 of its net; "
                            "toward (10, 16): the pass keeps that point: the centre of pad R1.1 of its net")
    r = _run(tmp_path / "turn", lib, (*bump, (9.7, 5.0), (3.0, 5.0)))
    assert _path(seen) == [((10.0, 16.0), (10.0, 5.0)), ((10.0, 5.0), (3.0, 5.0))]
    assert _counts(r) == (2, 2, 0, {}) and [(row["at"], row["step_mm"], row["slid_to"]) for row in _steps(r)] == [
        ([10.1, 12.0], 0.2, [10.2, 11.0]), ([9.85, 11.0], 0.3, [9.7, 5.0])]
    assert _rows(r)[0]["radius_mm"] == CAP and len(_rows(r)) == 1


def test_a_staircase_of_three_sidesteps_toward_one_turn_collapses(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Three 0.2 mm sidesteps, 1 mm apart, on the way to one turn. The first slides onto the second (one 0.4 mm track); the third may
    slide to the turn or back onto that track - the same corners and copper either way, so along the neighbour drawn first: one
    0.6 mm track; in the second sweep that one slides to the turn. Three moves, and the copper is pad - up - turn - along - pad
    with one corner at the cap radius. Every move's entry is still on the copper: on the track it lengthened, and passed on by the
    tracks that were merged or used up."""
    seen = _watch(monkeypatch)
    stairs = ((4.0, 16.0), (4.0, 13.0), (4.2, 13.0), (4.2, 12.0), (4.4, 12.0), (4.4, 11.0), (4.6, 11.0), (4.6, 6.0), (14.0, 6.0))
    r = _run(tmp_path, lib, stairs)
    assert _path(seen) == [((4.0, 16.0), (4.0, 6.0)), ((4.0, 6.0), (14.0, 6.0))]
    assert _counts(r) == (3, 3, 0, {})
    assert [(row["at"], row["step_mm"], row["slid_to"], row["corners_removed"], row["shorter_mm"]) for row in _steps(r)] == [
        ([4.1, 13.0], 0.2, [4.2, 12.0], 2, 0.0), ([4.5, 11.0], 0.2, [4.4, 12.0], 2, 0.0), ([4.3, 12.0], 0.6, [4.6, 6.0], 2, 0.0)]
    moves = [_entry((4.1, 13.0), 0.2), _entry((4.5, 11.0), 0.2), _entry((4.3, 12.0), 0.6)]
    arc = "corner:routing.corners 0.2:arc:at=4.0,6.0:tangent=1.6:radius=1.6"
    assert [t.provenance.derived_from for t in r.tracks] == [[moves[0], moves[2], arc], [arc], [*moves, arc]]
    assert "routing.corners 0.2: 3 sidestep(s) of the router on this net straightened, then 1 corner(s)" in r.tracks[2].provenance.note
    assert _shape(r)[0] == ((4.0, 16.0), (4.0, 7.6)) and _shape(r)[2] == ((5.6, 6.0), (14.0, 6.0)) and len(_arcs(r)) == 1
    assert sum(track_length(t) for t in _poly("A", *stairs)) == pytest.approx(20.0)  # as long as routed: three Zs


def test_a_u_slides_as_far_as_its_shorter_side(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """A track between two corners that turn the same way closes a detour: 10 mm up, 1 mm across, 3 mm back down, then on. It slides
    along its shorter side to that side's far corner and joins the track that leaves there: the detour's 3 mm are gone twice. Along
    the longer side it cannot slide (it would pass the end of the shorter one): where the shorter side ends on a point the pass
    keeps, the U stays. A detour whose two sides are equally long - 5 mm out, 0.4 mm across, 5 mm back - is cut off whole: both sides
    are used up, and the tracks before and after it, on one line, become one track with the 0.4 mm between them."""
    seen = _watch(monkeypatch)
    r = _run(tmp_path, lib, ((4.0, 16.0), (4.0, 6.0), (5.0, 6.0), (5.0, 9.0), (12.0, 9.0)))
    assert _path(seen) == [((4.0, 16.0), (4.0, 9.0)), ((4.0, 9.0), (12.0, 9.0))]
    assert _steps(r) == [{"net": "A", "layer": "F.Cu", "at": [4.5, 6.0], "step_mm": 1.0, "outcome": "straightened", "slid_to": [5.0, 9.0],
                          "corners_removed": 2, "shorter_mm": 6.0}]
    assert _rows(r)[0]["radius_mm"] == CAP and _shape(r)[0] == ((4.0, 16.0), (4.0, 10.6)) and _shape(r)[2] == ((5.6, 9.0), (12.0, 9.0))
    assert r.stats["net_length_mm"]["A"] == pytest.approx(5.4 + CAP * math.pi / 2.0 + 6.4, abs=2e-6)  # 21 mm as routed, 15 mm straightened
    # the shorter side leads to the pad: nowhere to slide
    stays = _run(tmp_path / "pad", lib, ((4.0, 8.0), (4.0, 6.0), (5.0, 6.0), (5.0, 9.0), (12.0, 9.0)))
    assert _path(seen) == [((4.0, 8.0), (4.0, 6.0)), ((4.0, 6.0), (5.0, 6.0)), ((5.0, 6.0), (5.0, 9.0)), ((5.0, 9.0), (12.0, 9.0))]
    first, second = _steps(stays)
    assert _counts(stays) == (2, 0, 2, {"kept_point": 2}) and (first["at"], second["at"]) == ([4.5, 6.0], [5.0, 7.5])
    assert first["why"] == ("toward (5, 9): the sidestep would pass (4, 8), the end of its shorter neighbour (a U slides only as far as the shorter of "
                            "its two neighbours); toward (4, 8): the pass keeps that point: the centre of pad R1.1 of its net")
    whole = _run(tmp_path / "whole", lib, ((3.0, 10.0), (6.0, 10.0), (6.0, 5.0), (6.4, 5.0), (6.4, 10.0), (12.0, 10.0)))
    assert _path(seen) == [((3.0, 10.0), (12.0, 10.0))] == _shape(whole) and whole.stats["corners"]["found"] == 0
    assert _steps(whole) == [{"net": "A", "layer": "F.Cu", "at": [6.2, 5.0], "step_mm": 0.4, "outcome": "straightened", "slid_to": [6.0, 10.0],
                              "corners_removed": 4, "shorter_mm": 10.0}]


def test_a_junction_on_the_side_a_u_shortens_stays_on_the_copper_or_stops_the_move(tmp_path: Path, lib: KicadLibrary,
                                                                                      monkeypatch: pytest.MonkeyPatch):
    """A move never slides across a point the pass keeps. The U above shortens its long side from (4, 6) back to (4, 9); a branch of
    the net that ends on that side

    * at (4, 12), on the part that stays: its end lies on the side's centreline before and after - the U is slid away;
    * at (4, 8), on the part the move takes away: the branch would be left hanging - the U move is refused (since 2026-11-19 the
      branch's T is a lane change, which takes the U's copper instead);
    * at (4, 9), exactly where the slid track would turn, but along the line the slid copper takes there (since 2026-11-19): the
      copper passes straight through that junction - a T, the line y = 9 unbroken - and the U is slid away;
    * at (4, 9) running off at 45 degrees, along no line the copper takes: a corner on a junction is no plain corner - the U stays."""
    seen = _watch(monkeypatch)
    detour = ((4.0, 16.0), (4.0, 6.0), (5.0, 6.0), (5.0, 9.0), (12.0, 9.0))
    r = _run(tmp_path, lib, detour, *_poly("A", (4.0, 12.0), (2.0, 12.0)))
    assert _path(seen) == [((4.0, 16.0), (4.0, 9.0)), ((4.0, 9.0), (12.0, 9.0)), ((4.0, 12.0), (2.0, 12.0))] and _counts(r) == (2, 1, 1, {"kept_point": 1})
    # since 2026-11-19 the slid U leaves a lane change at a T (the branch west at y 12, the run east at y 9, 3 mm apart along the side
    # between them): a step at a T, which stays - both lanes end on points the pass keeps
    assert _steps(r)[1]["junction"] == [4.0, 12.0] and _steps(r)[1]["reason"] == "kept_point"
    r = _run(tmp_path / "through", lib, detour, *_poly("A", (4.0, 9.0), (2.0, 9.0)))
    # since 2026-11-19 the branch's T on the U's long side is a lane change (west at y 9 there, east at y 9 beyond the U): the same
    # copper, the branch carried through the T to the far side of the U and joined with the run there - one track along y = 9
    assert _path(seen) == [((4.0, 16.0), (4.0, 9.0)), ((2.0, 9.0), (12.0, 9.0))] and _counts(r) == (1, 1, 0, {})
    assert _steps(r)[0]["junction"] == [4.0, 9.0] and _steps(r)[0]["corners_removed"] == 4
    # the branch at (4, 8): the U itself cannot be slid (the branch's end lies on copper the move would take away), but since
    # 2026-11-19 the branch's T, 2 mm above the U's corner, is a lane change: the branch is carried through the T to the U's far
    # side and the U's copper goes - the run comes down to the T and leaves along the branch's line, 4 mm shorter, and a 1 mm step
    # down to the run east is left between two points the pass keeps
    r = _run(tmp_path / "8.0", lib, detour, *_poly("A", (4.0, 8.0), (2.0, 8.0)))
    assert _path(seen) == [((4.0, 16.0), (4.0, 8.0)), ((5.0, 8.0), (5.0, 9.0)), ((5.0, 9.0), (12.0, 9.0)), ((5.0, 8.0), (2.0, 8.0))], _path(seen)
    assert [(row["outcome"], row.get("junction"), row.get("shorter_mm")) for row in _steps(r)] == [("straightened", [4.0, 8.0], 4.0), ("left", None, None)]
    for y, branch, words in ((9.0, ((4.0, 9.0), (2.0, 11.0)), "the slid copper would turn at (4, 9), on its own net's track (4, 9)-(2, 11) (a junction)"),):
        r = _run(tmp_path / f"{y}", lib, detour, *_poly("A", *branch))
        assert _path(seen)[:4] == list(zip(detour, detour[1:])) and r.stats["corners"]["straightened"] == 0, y
        row = _steps(r)[0]
        assert (row["at"], row["reason"]) == ([4.5, 6.0], "kept_point"), (y, row)
        assert row["why"] == f"toward (5, 9): {words}; toward (4, 16): the pass keeps that point: the centre of pad R1.1 of its net", y


def test_a_track_of_eight_widths_is_no_sidestep(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """A track between two corners is a sidestep below 8 track widths - the room two arcs of the cap radius need (4 widths of tangent
    length each): at 3.2 mm (8 x 0.4 mm) it holds both arcs - they use it up and meet in its middle - and stays where it was
    routed; 0.2 mm shorter it is slid away."""
    assert SIDESTEP_WIDTHS == 2.0 * ARC_RADIUS_WIDTHS[0] == 8.0 and SIDESTEP_WIDTHS * W == pytest.approx(2.0 * CAP) and STRAIGHTEN == "straighten"
    seen = _watch(monkeypatch)
    at_the_limit = ((4.0, 17.0), (4.0, 13.0), (7.2, 13.0), (7.2, 6.0), (14.0, 6.0))
    r = _run(tmp_path, lib, at_the_limit)
    assert _path(seen) == list(zip(at_the_limit, at_the_limit[1:])) and _counts(r) == (0, 0, 0, {}) and _steps(r) == []
    assert [row["radius_mm"] for row in _rows(r)] == [CAP, CAP, CAP]  # 3.2 mm between two corners: an arc of 1.6 mm at each end of it
    assert [len(piece) for piece in _shape(r)] == [2, 3, 3, 2, 3, 2] and _shape(r)[1][2] == _shape(r)[2][0] == (5.6, 13.0)
    r = _run(tmp_path / "below", lib, ((4.0, 16.0), (4.0, 13.0), (7.0, 13.0), (7.0, 6.0), (14.0, 6.0)))
    assert _path(seen) == [((4.0, 16.0), (4.0, 6.0)), ((4.0, 6.0), (14.0, 6.0))] and _counts(r) == (1, 1, 0, {})
    words = r.stats["corners"]["straighten"]
    assert "a track shorter than 8 track widths (the room two arcs of the largest radius need) between two right-angle corners" in words
    assert "neither of them on a track at its net's neck-down width" in words and "a move never lengthens the net" in words
    # the length rule says what it measures: the router's straight copper, not the rounded one (a move takes away corners whose
    # rounding would have shortened the net - the 9 V piano board's rev 3 SPK and GND F.Cu came out 0.27 / 0.22 mm longer that way)
    assert "the net's copper as the router laid it (before the rounding; the finished copper is never longer than the router's" in words


#: routing.corners 0.2, since 2026-11-19: net A of the junction boards - R1 east, R2 at the end of a branch north of the junction, R5 south-west
JUNCTION_NETS = {"A": [("R1", "1"), ("R2", "1"), ("R5", "1")], "B": [("R3", "1"), ("R4", "1")]}


def test_a_step_at_a_junction_is_slid_so_the_run_passes_straight_through_it(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The 9 V piano board's MIX run came from the east to a junction at (40.5, 42), stepped 0.25 mm down there and ran on west 0.25 mm
    off its own line (a kink after the rounding): the step leaves a junction, so it was no sidestep. routing.corners 0.2, since 2026-11-19: a short step
    between a junction and a plain corner, where a track of the junction arrives along the line the step's neighbour runs on, is slid
    along that neighbour like a sidestep - the arriving track is carried through the junction, which stays where it was (a T with an
    unbroken line); the row names the junction. Without a track arriving along that line the step is a plain branch: no candidate."""
    seen = _watch(monkeypatch)
    east, branch, step = _poly("A", (16.0, 6.0), (10.0, 6.0)), _poly("A", (10.0, 6.0), (10.0, 2.0)), _poly("A", (10.0, 6.0), (10.0, 6.25), (3.0, 6.25), (3.0, 14.0))
    parts = [("R1", "SMD1", 16.0, 6.0), ("R2", "SMD1", 10.0, 2.0), ("R5", "SMD1", 3.0, 14.0), ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    r = _pass(tmp_path, lib, [*east, *branch, *step], parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((16.0, 6.0), (3.0, 6.0)), ((10.0, 6.0), (10.0, 2.0)), ((3.0, 6.0), (3.0, 14.0))], _path(seen)
    row = _steps(r)[0]
    assert _counts(r) == (1, 1, 0, {}) and row["junction"] == [10.0, 6.0] and row["outcome"] == "straightened" and row["corners_removed"] == 2
    assert row["shorter_mm"] == 0.0  # a Z joined to the track that continues it keeps the length
    # R1 reached at 45 degrees instead: no track arrives at (10, 6) heading west, so the step is a branch of the junction and stays
    parts[0] = ("R1", "SMD1", 14.0, 2.0)
    r = _pass(tmp_path / "branch", lib, [*_poly("A", (14.0, 2.0), (10.0, 6.0)), *branch, *step], parts=parts, nets=JUNCTION_NETS)
    assert _counts(r) == (0, 0, 0, {}) and _path(seen)[:3] == [((14.0, 2.0), (10.0, 6.0)), ((10.0, 6.0), (10.0, 2.0)), ((10.0, 6.0), (10.0, 6.25))]


def test_a_bump_between_two_junctions_becomes_their_straight_line(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The piano board's VREF bus ran at y 111.5 from one resistor column's junction, stepped 0.25 mm down, ran 15 mm at y 111.75 and
    stepped back up onto the next column's junction (the stub up to that resistor's pad ends on the line there). The bump is slid onto
    the line: the copper arrives at the second junction along the line the track beyond it leaves on, so it passes straight through
    that junction too, and the bus is one line through both junctions, 0.5 mm shorter."""
    seen = _watch(monkeypatch)
    nets = {"A": [("R1", "1"), ("R2", "1"), ("R5", "1"), ("R6", "1")], "B": [("R3", "1"), ("R4", "1")]}
    parts = [("R1", "SMD1", 2.0, 6.0), ("R2", "SMD1", 18.0, 6.0), ("R5", "SMD1", 6.0, 2.0), ("R6", "SMD1", 14.0, 2.0),
             ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    tracks = [*_poly("A", (2.0, 6.0), (6.0, 6.0)), *_poly("A", (6.0, 6.0), (6.0, 2.0)), *_poly("A", (6.0, 6.0), (6.0, 6.25), (14.0, 6.25), (14.0, 2.0)),
              *_poly("A", (14.0, 6.0), (18.0, 6.0))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=nets)
    assert _path(seen) == [((2.0, 6.0), (14.0, 6.0)), ((6.0, 6.0), (6.0, 2.0)), ((14.0, 6.0), (14.0, 2.0)), ((14.0, 6.0), (18.0, 6.0))], _path(seen)
    assert _counts(r) == (1, 1, 0, {}) and _steps(r)[0]["junction"] == [6.0, 6.0] and _steps(r)[0]["shorter_mm"] == 0.5


def test_a_u_may_end_its_run_at_a_junction_where_a_track_continues_its_new_line(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """routing.corners 0.2, since 2026-11-19 (module docstring: the straightening, a junction the run may pass through, (b)): a move may
    meet the end of its run at a junction from a new direction when a straight track of the net continues the slid copper's line on the
    far side of the junction. The junction (4, 10): R1's track arrives from the north heading south, a branch runs west to R5, and
    the run leaves east as an equal-armed U - 4 mm east, 0.5 mm down, 4 mm back west - and goes on south to R2. Slid, the U's copper
    leaves the junction heading south, the line R1's track arrives on: the net becomes one straight line from R1 to R2 through the
    junction, which stays a T. Where no track arrives at the junction along that line (R1 reached from the west and R5 at 45
    degrees), the junction keeps the direction the copper meets it in and the U stays."""
    seen = _watch(monkeypatch)
    u = _poly("A", (4.0, 10.0), (8.0, 10.0), (8.0, 10.5), (4.0, 10.5), (4.0, 16.0))
    parts = [("R1", "SMD1", 4.0, 4.0), ("R2", "SMD1", 4.0, 16.0), ("R5", "SMD1", 1.0, 10.0), ("R3", "SMD1", 18.8, 1.2), ("R4", "SMD1", 18.8, 18.8)]
    r = _pass(tmp_path, lib, [*_poly("A", (4.0, 4.0), (4.0, 10.0)), *_poly("A", (4.0, 10.0), (1.0, 10.0)), *u], parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((4.0, 4.0), (4.0, 10.0)), ((4.0, 10.0), (1.0, 10.0)), ((4.0, 10.0), (4.0, 16.0))], _path(seen)
    assert _counts(r) == (1, 1, 0, {}) and _steps(r)[0]["outcome"] == "straightened"
    # no straight track continues the new line beyond the junction: the U stays as routed
    parts[0], parts[2] = ("R1", "SMD1", 1.0, 10.0), ("R5", "SMD1", 1.0, 7.0)
    r = _pass(tmp_path / "no_line", lib, [*_poly("A", (1.0, 10.0), (4.0, 10.0)), *_poly("A", (1.0, 7.0), (4.0, 10.0)), *u], parts=parts,
              nets=JUNCTION_NETS)
    assert _counts(r) == (1, 0, 1, {"kept_point": 1}) and _path(seen)[2:] == [((4.0, 10.0), (8.0, 10.0)), ((8.0, 10.0), (8.0, 10.5)),
                                                                               ((8.0, 10.5), (4.0, 10.5)), ((4.0, 10.5), (4.0, 16.0))]
    assert "the copper would meet (4, 10) from a new direction" in _steps(r)[0]["why"]


# --------------------------------------------------------------------------- what stops a move


def test_a_sidestep_between_two_points_the_pass_keeps_stays(tmp_path: Path, lib: KicadLibrary):
    """The pass keeps pads, vias, junctions and track ends where they are, in the direction the copper meets them. A sidestep whose
    two neighbours both end on such a point stays as routed - the tracks untouched, the reason ``kept_point`` with both points:

    * a pad below and a via of the net above;
    * a junction at the turn (a third track of the net ends there);
    * another track of the net ending on the neighbour the sidestep would slide along (the move would take the copper away from
      under that end), and a via of the net on it;
    * a pad that the copper left of the move would meet from the other side: the track after the turn runs back over the slid
      sidestep and ends, shorter than it, on its pad - what is left of the two would enter that pad from where the track came."""
    via = Via(net="A", x_mm=10.2, y_mm=8.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    between = _poly("A", (10.0, 16.0), (10.0, 13.0), (10.2, 13.0), (10.2, 8.0))
    r = _pass(tmp_path, lib, between, vias=(via,), parts=_parts((10.0, 16.0), (16.0, 4.0)), nets=STEP_NETS)
    assert all(t is u for t, u in zip(r.tracks, between)) and len(r.tracks) == 3
    assert _counts(r) == (1, 0, 1, {"kept_point": 1})
    assert _steps(r) == [{"net": "A", "layer": "F.Cu", "at": [10.1, 13.0], "step_mm": 0.2, "outcome": "left", "reason": "kept_point",
                          "why": "toward (10.2, 8): the pass keeps that point: at or inside the via of its net at (10.2, 8); "
                                 "toward (10, 16): the pass keeps that point: the centre of pad R1.1 of its net"}]
    cases = (
        (_poly("A", (4.2, 6.0), (4.2, 3.0)), (), "toward (4.2, 6): the pass keeps that point: a junction of 3 track ends of its net; "),
        (_poly("A", (4.2, 9.0), (7.0, 9.0)), (), "toward (4.2, 6): an end of track (4.2, 9)-(7, 9) of its net at (4.2, 9) lies on copper the move would take away; "),
        ([], (Via(net="A", x_mm=4.2, y_mm=9.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),),
         "toward (4.2, 6): a via of its net at (4.2, 9) lies on copper the move would take away; "),
    )
    for k, (more, vias, words) in enumerate(cases):
        r = _run(tmp_path / f"{k}", lib, BESIDE_A_TURN, *more, vias=vias)
        row, = [row for row in _steps(r) if row["at"] == [4.1, 13.0]]
        assert (row["outcome"], row["reason"]) == ("left", "kept_point"), (k, row)
        assert row["why"] == words + "toward (4, 16): the pass keeps that point: the centre of pad R1.1 of its net", k
        assert ((4.0, 13.0), (4.2, 13.0)) in _shape(r) and r.stats["corners"]["straightened"] == 0, k
        assert not any(STRAIGHTEN in e for t in r.tracks for e in t.provenance.derived_from), k
    hook = ((4.0, 16.0), (4.0, 13.0), (5.0, 13.0), (5.0, 9.0), (4.4, 9.0))
    r = _run(tmp_path / "hook", lib, hook)
    assert _counts(r) == (1, 0, 1, {"kept_point": 1}) and not any(STRAIGHTEN in e for t in r.tracks for e in t.provenance.derived_from)
    assert _steps(r)[0]["why"] == ("toward (5, 9): the copper would meet (4.4, 9) from a new direction, and the pass keeps that point: the centre of pad "
                                   "R2.1 of its net; toward (4, 16): the pass keeps that point: the centre of pad R1.1 of its net")


def test_a_corner_inside_a_pad_box_is_a_point_the_pass_keeps(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The router's terminal cells lie in their pads, so a run often turns inside a pad's box before it reaches the pad's centre. Such
    a corner is no plain corner - the pad's copper covers it - and a sidestep may not slide into it. Pad - 0.3 mm aside inside the pad
    - 3 mm up - 0.2 mm back - 5 mm up - a via of the net: the sidestep stays as routed, the pad box and the via its two kept points."""
    seen = _watch(monkeypatch)
    pts = ((4.0, 16.0), (4.3, 16.0), (4.3, 13.0), (4.1, 13.0), (4.1, 8.0))
    via = Via(net="A", x_mm=4.1, y_mm=8.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    r = _pass(tmp_path, lib, _poly("A", *pts), vias=(via,), parts=_parts((4.0, 16.0), (14.0, 4.0)), nets=STEP_NETS)
    assert _path(seen) == list(zip(pts, pts[1:])) and _counts(r) == (1, 0, 1, {"kept_point": 1})
    assert _steps(r) == [{"net": "A", "layer": "F.Cu", "at": [4.2, 13.0], "step_mm": 0.2, "outcome": "left", "reason": "kept_point",
                          "why": "toward (4.1, 8): the pass keeps that point: at or inside the via of its net at (4.1, 8); "
                                 "toward (4.3, 16): the pass keeps that point: inside the box of pad R1.1 of its net"}]


def test_a_sidestep_another_nets_copper_needs_stays_and_the_result_is_measured_again(tmp_path: Path, lib: KicadLibrary):
    """The copper a move adds passes the router's exact audit. With the sidestep beside a turn, the straight run would lie at x = 4 mm
    where the routed one lies at x = 4.2 mm:

    * net B's via at (3.3, 10) is legal beside the routed copper (0.3 mm of gap, measured here) and would be 0.1 mm from the straight
      one: the sidestep stays, the reason ``clearance`` with the audit's words; at (3.1, 10) the straight copper keeps 0.3 mm and
      the sidestep is slid away;
    * net B's track along x = 3.4 mm, 0.4 mm from the routed copper, would be 0.2 mm from the straight one: the sidestep stays;
      along x = 3.3 mm (0.3 mm from the straight copper) it is slid away.

    Whatever the outcome, the copper keeps the clearance: the gap is measured here, and the ``pcb.routing`` checks pass."""
    ir = board_ir(tmp_path / "board", lib, _parts(BESIDE_A_TURN[0], BESIDE_A_TURN[-1]), STEP_NETS, (20.0, 20.0))
    for x, slid, gap in ((3.3, False, 0.3), (3.1, True, 0.3)):
        via = Via(net="B", x_mm=x, y_mm=10.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
        r = _run(tmp_path / f"via{x}", lib, BESIDE_A_TURN, vias=(via,))
        assert _counts(r) == ((1, 1, 0, {}) if slid else (1, 0, 1, {"clearance": 1})), x
        assert _copper_gap_to_disc(r, (x, 10.0), 0.4) == pytest.approx(gap, abs=1e-9) and gap >= P.clearance_mm
        if not slid:
            row, = _steps(r)
            assert row["why"] == ("toward (4.2, 6): the straightened copper of the sidestep at (4.1, 13) comes 0.1000 mm from the via of B (needs "
                                  "0.25 mm); toward (4, 16): the pass keeps that point: the centre of pad R1.1 of its net")
            assert ((4.0, 13.0), (4.2, 13.0)) in _shape(r)
        checks = _routing_checks(ir, Routing(params=P, tracks=straight_pieces(r.tracks), vias=[via]), lib, tmp_path)
        assert checks["pcb.routing.clearance"].status == ValidationStatus.PASS, (x, checks)
    for x, slid, gap in ((3.4, False, 0.4), (3.3, True, 0.3)):
        other, = _poly("B", (x, 12.0), (x, 8.0))
        r = _run(tmp_path / f"track{x}", lib, BESIDE_A_TURN, other)
        assert _counts(r) == ((1, 1, 0, {}) if slid else (1, 0, 1, {"clearance": 1})), x
        assert _gap_to_track(r, other) == pytest.approx(gap, abs=1e-9) and gap >= P.clearance_mm
        if not slid:
            assert "comes 0.2000 mm from the track of B (needs 0.25 mm)" in _steps(r)[0]["why"]
        checks = _routing_checks(ir, Routing(params=P, tracks=straight_pieces(r.tracks), vias=[]), lib, tmp_path)
        assert checks["pcb.routing.clearance"].status == ValidationStatus.PASS, (x, checks)


def test_a_move_is_judged_against_the_copper_as_the_moves_before_it_left_it(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """A move takes the copper it changes out of the audit and puts the new copper in at once, so a later move of another net is judged
    against the copper as it stands. Net A's sidestep beside a turn moves first (as short as the others, drawn first) and leaves its
    run at x = 4 mm instead of 4.2 mm. Net B's sidestep on its right would lay copper along x = 4.7 mm - 0.5 mm from A's run as routed,
    0.7 mm from it straightened (0.65 mm is the least: the clearance and two half-widths); net C's sidestep on its left would lay
    copper along x = 3.5 mm - 0.7 mm from A's run as routed, 0.5 mm from it straightened. So B is slid and C stays; where a branch of
    A pins A's sidestep (A stays as routed) it is the other way round. Every gap between two nets, measured here on the copper the
    straightening leaves, keeps the clearance."""
    seen = _watch(monkeypatch)
    b = _poly("B", (4.7, 14.5), (4.7, 13.5), (4.9, 13.5), (4.9, 9.0), (8.0, 9.0))
    c = _poly("C", (3.5, 4.0), (3.5, 5.0), (3.3, 5.0), (3.3, 9.0), (1.5, 9.0))
    nets = {**STEP_NETS, "C": [("R5", "1"), ("R6", "1")]}
    parts = _parts(BESIDE_A_TURN[0], BESIDE_A_TURN[-1], ("R5", "SMD1", 18.8, 1.2), ("R6", "SMD1", 18.8, 18.8))
    for pinned in (False, True):
        branch = _poly("A", (4.2, 10.0), (3.0, 10.0)) if pinned else []
        r = _pass(tmp_path / f"{pinned}", lib, [*_poly("A", *BESIDE_A_TURN), *branch, *b, *c], parts=parts, nets=nets)
        rows = {row["net"]: row for row in _steps(r)}
        assert {net: row["outcome"] for net, row in rows.items()} == (
            {"A": "left", "B": "left", "C": "straightened"} if pinned else {"A": "straightened", "B": "straightened", "C": "left"}), pinned
        stays = rows["B" if pinned else "C"]
        assert stays["reason"] == "clearance" and "comes 0.1000 mm from the track of A (needs 0.25 mm)" in stays["why"], (pinned, stays)
        assert (rows["A"]["reason"] == "kept_point") if pinned else (_path(seen, "A") == [((4.0, 16.0), (4.0, 6.0)), ((4.0, 6.0), (14.0, 6.0))])
        # two straight tracks that do not cross are nearest at an end of one of them
        gap = min(min(_point_segment(a1, a2, b2), _point_segment(b1, a2, b2), _point_segment(a2, a1, b1), _point_segment(b2, a1, b1))
                  for n1, a1, b1 in seen for n2, a2, b2 in seen if n1 < n2) - W
        assert gap == pytest.approx(0.3) and gap >= P.clearance_mm, (pinned, gap)


def test_a_keep_out_and_the_edge_stop_a_move_as_they_stop_a_chamfer(tmp_path: Path, lib: KicadLibrary):
    """A track keep-out the straight copper would enter - 0.1 mm from its centreline, where half the width is the least - stops the
    move unless it allows the net; so does the board edge, where the straight copper would run 0.4 mm from the outline and 0.5 mm
    (the edge clearance and half the width) is the least."""
    area = keepout("K1", [3.0, 7.0, 0.9, 4.0], forbids=["tracks"], layers=["F.Cu"])  # x 3.0 .. 3.9, beside the straight run at x = 4
    r = _run(tmp_path, lib, BESIDE_A_TURN, keepouts=[area])
    assert _counts(r) == (1, 0, 1, {"keepout": 1})
    assert _steps(r)[0]["why"].startswith("toward (4.2, 6): the straightened copper of the sidestep at (4.1, 13) would enter track keep-out K1; ")
    allowed = keepout("K1", [3.0, 7.0, 0.9, 4.0], forbids=["tracks"], layers=["F.Cu"], nets=["A"])
    assert _counts(_run(tmp_path / "allowed", lib, BESIDE_A_TURN, keepouts=[allowed])) == (1, 1, 0, {})
    # the routed run 0.6 mm from the left edge, the run from the pad 0.4 mm: straight, the whole run would lie 0.4 mm from the edge
    at_the_edge = ((0.4, 16.0), (0.4, 13.0), (0.6, 13.0), (0.6, 6.0), (10.0, 6.0))
    r = _run(tmp_path / "edge", lib, at_the_edge)
    assert _counts(r) == (1, 0, 1, {"edge": 1})
    assert _steps(r)[0]["why"].startswith("toward (0.6, 6): the straightened copper of the sidestep at (0.5, 13) would come nearer the board edge than 0.3 mm; ")


def test_a_move_changes_no_contact_within_its_net(tmp_path: Path, lib: KicadLibrary):
    """The audit ignores the net's own copper, so a move is held to it separately: what of the net touched the copper of the tracks it
    changes must still touch it, and nothing else may. With the sidestep beside a turn (routed run at x = 4.2 mm, straight run at
    x = 4 mm), each of these stops the move with the reason ``own_copper``:

    * the net's via at (3.5, 10), clear of the routed copper: the straight copper would reach its disc (a new contact);
    * the net's pad R5 with its box up to x = 3.9: the straight copper would reach it (a new contact);
    * the net's via at (4.75, 10), whose disc the routed copper overlaps: the straight copper would leave it (a contact lost - it
      may have been that via's only one); the net's pad R5 with its box from x = 4.4, which the routed copper touches: the same;
    * the net's via at (4, 10): its disc already lies under the routed copper, so no contact changes - but the straight copper
      would pass through its centre, a junction the copper did not have."""
    def own(x: float) -> tuple[Via, ...]:
        return (Via(net="A", x_mm=x, y_mm=10.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND),)

    what = "the straightened copper of the sidestep at (4.1, 13)"
    cases = (
        ({"vias": own(3.5)}, f"{what} would reach the via of its own net at (3.5, 10), which the copper it replaces did not reach (a new contact)"),
        ({"parts": (("R5", "SMD1", 3.4, 10.0),), "nets": {"A": [("R1", "1"), ("R2", "1"), ("R5", "1")], "B": STEP_NETS["B"]}},
         f"{what} would reach pad R5.1 of its own net, which the copper it replaces did not reach (a new contact)"),
        ({"vias": own(4.75)}, f"{what} would leave the via of its own net at (4.75, 10), which the copper it replaces reached (a contact lost)"),
        ({"parts": (("R5", "SMD1", 4.9, 10.0),), "nets": {"A": [("R1", "1"), ("R2", "1"), ("R5", "1")], "B": STEP_NETS["B"]}},
         f"{what} would leave pad R5.1 of its own net, which the copper it replaces reached (a contact lost)"),
        ({"vias": own(4.0)}, f"{what} would pass through a via of its own net at (4, 10) (a new junction)"),
    )
    for k, (extra, words) in enumerate(cases):
        r = _run(tmp_path / f"{k}", lib, BESIDE_A_TURN, **extra)
        row, = _steps(r)
        assert _counts(r) == (1, 0, 1, {"own_copper": 1}) and (row["outcome"], row["reason"]) == ("left", "own_copper"), (k, row)
        assert row["why"] == f"toward (4.2, 6): {words}; toward (4, 16): the pass keeps that point: the centre of pad R1.1 of its net", k
        assert ((4.0, 13.0), (4.2, 13.0)) in _shape(r), k
    # the same via further away touches neither the routed copper nor the straight one: the sidestep is slid away
    assert _counts(_run(tmp_path / "clear", lib, BESIDE_A_TURN, vias=own(3.2))) == (1, 1, 0, {})


def test_copper_another_mechanism_pinned_is_not_straightened(tmp_path: Path, lib: KicadLibrary):
    """What the corner pass skips, the straightening does not move: the nets of a differential pair and of a length-matched group are
    not looked at (a meander is a deliberate bump), and a short track beside an escape stub or a width change is no sidestep - its
    corner there is one the pass would skip. The tracks stay the same objects and no sidestep is counted. One more for this step
    alone: copper at its net's neck-down width stays where the router necked it down (``si.impedance`` reads a narrower track as a
    neck-down only near a pad) - its corners are rounded as before -, while the same copper at the net's own width is
    straightened."""
    pair = {"A": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="B", pair_spacing_mm=0.25),
            "B": NetRule(width_mm=0.4, clearance_mm=0.25, pair_partner="A", pair_spacing_mm=0.25)}
    group = {"A": NetRule(match_group="BUS", max_skew_mm=0.5)}
    escape = HAND.model_copy(update={"derived_from": ["net:A", "escape:R1.1:stub:width=0.4:points=4.0,16.0;4.0,13.0"]})
    routed = _poly("A", *BESIDE_A_TURN)
    stub = [*_poly("A", (4.0, 16.0), (4.0, 13.0), prov=escape), *routed[1:]]
    wide = [*routed[:2], *_poly("A", (4.2, 13.0), (4.2, 6.0), width=0.3), *routed[3:]]
    for name, tracks, rules in (("pair", routed, pair), ("match_group", routed, group), ("escape", stub, {}), ("width", wide, {})):
        r = _pass(tmp_path / name, lib, tracks, parts=_parts(BESIDE_A_TURN[0], BESIDE_A_TURN[-1]), nets=STEP_NETS, rules=rules)
        assert _counts(r) == (0, 0, 0, {}) and _steps(r) == [], name
        assert [(t.start, t.end) for t in r.tracks if t.mid is None][:2] == [((4.0, 16.0), (4.0, 13.0)), ((4.0, 13.0), (4.2, 13.0))], name
        assert all(t is u for t, u in zip(r.tracks, tracks[:2])), name
        assert not any(STRAIGHTEN in e for t in r.tracks for e in t.provenance.derived_from), name
    necked = {"A": NetRule(width_mm=0.6, neckdown_width_mm=0.25)}
    parts = _parts(BESIDE_A_TURN[0], BESIDE_A_TURN[-1])
    r = _pass(tmp_path / "neck", lib, _poly("A", *BESIDE_A_TURN, width=0.25), parts=parts, nets=STEP_NETS, rules=necked)
    c = r.stats["corners"]  # the sidestep stays: its two corners get 0.1 mm chamfers (one 45-degree run), the turn its arc
    assert _counts(r) == (0, 0, 0, {}) and _shape(r)[1:3] == [((4.0, 13.1), (4.1, 13.0)), ((4.1, 13.0), (4.2, 12.9))] and (c["arcs"], c["mitred"]) == (1, 2)
    r = _pass(tmp_path / "full", lib, _poly("A", *BESIDE_A_TURN, width=0.6), parts=parts, nets=STEP_NETS, rules=necked)
    assert _counts(r) == (1, 1, 0, {}) and _rows(r)[0]["radius_mm"] == 2.4  # 4 widths of 0.6 mm


def test_a_ring_of_plain_corners_has_no_point_to_slide_to(tmp_path: Path, lib: KicadLibrary):
    """Copper that closes on itself through plain corners only - no pad, via, junction or end on it - has no point the pass keeps; its
    short tracks are counted and left. A ring that hangs on a junction is kept by that junction: sliding one of its sides onto the
    opposite one would leave nothing of the ring but its two other sides run over each other, and the junction would be met from
    a new direction."""
    ring = _poly("A", (8.0, 8.0), (10.0, 8.0), (10.0, 10.0), (8.0, 10.0), (8.0, 8.0))
    r = _pass(tmp_path, lib, ring, parts=_parts((4.0, 16.0), (16.0, 4.0)), nets=STEP_NETS)
    assert _counts(r) == (4, 0, 4, {"kept_point": 4})
    assert {row["why"] for row in _steps(r)} == {"its copper closes on itself through plain corners: the pass keeps no point of it to slide the sidestep to"}
    hung = [*_poly("A", (8.0, 12.0), (8.0, 10.0)), *_poly("A", (8.0, 10.0), (10.0, 10.0), (10.0, 8.0), (8.0, 8.0), (8.0, 10.0))]
    r = _pass(tmp_path / "hung", lib, hung, parts=_parts((8.0, 12.0), (16.0, 4.0)), nets=STEP_NETS)
    # routing.corners 0.2, since 2026-11-19: the top side also leaves the junction (8, 10), where the stem arrives along the line the right side runs
    # on - a step that could pass through the junction; sliding it would fold the ring onto itself just the same
    assert _counts(r) == (3, 0, 3, {"kept_point": 3}) and [row["at"] for row in _steps(r)] == [[9.0, 10.0], [10.0, 9.0], [9.0, 8.0]]
    assert _steps(r)[0]["junction"] == [8.0, 10.0] and "the copper would meet (8, 10) from a new direction" in _steps(r)[0]["why"]
    assert _steps(r)[1]["why"] == ("toward (8, 8): the copper would meet (8, 10) from a new direction, and the pass keeps that point: a junction of 3 track "
                                   "ends of its net; toward (8, 10): the pass keeps that point: a junction of 3 track ends of its net")


# --------------------------------------------------------------------------- the records


#: the hand-drawn copper of this file's boards, by name: net A's points (pad R1 at the first, pad R2 at the last)
BOARDS: dict[str, tuple[Point, ...]] = {
    "beside_a_turn": BESIDE_A_TURN,
    "turn_back": TURNS["aside to the right, turn to the left"][0],
    "either_way": ((3.0, 10.0), (6.0, 10.0), (6.0, 8.0), (6.2, 8.0), (6.2, 4.0), (12.0, 4.0)),
    "bump": ((10.0, 16.0), (10.0, 12.0), (10.2, 12.0), (10.2, 11.0), (10.0, 11.0), (10.0, 5.0)),
    "unequal_bump": ((10.0, 16.0), (10.0, 12.0), (10.2, 12.0), (10.2, 11.0), (9.7, 11.0), (9.7, 5.0)),
    "staircase": ((4.0, 16.0), (4.0, 13.0), (4.2, 13.0), (4.2, 12.0), (4.4, 12.0), (4.4, 11.0), (4.6, 11.0), (4.6, 6.0), (14.0, 6.0)),
    "u": ((4.0, 16.0), (4.0, 6.0), (5.0, 6.0), (5.0, 9.0), (12.0, 9.0)),
    "between_pads": ((10.0, 16.0), (10.0, 13.0), (10.2, 13.0), (10.2, 8.0)),
}


def test_the_stats_add_up_and_two_runs_give_the_same_bytes(tmp_path: Path, lib: KicadLibrary):
    """On every board of this file: ``sidesteps == straightened + sidesteps_left``, one row each (the ones slid first, in the order
    of the moves, then the ones left); the reasons add up to the ones left; every move's entry is on the copper once per track it
    changed; ``found`` counts the corners of the straightened copper; the net is never longer than routed, and its stats are the
    sums of its tracks; the keys follow the ones the arc style had; and a second run gives the same bytes."""
    for name, pts in BOARDS.items():
        r, again = _run(tmp_path / name, lib, pts), _run(tmp_path / name / "again", lib, pts)
        assert canonical(r) == canonical(again), name
        c = r.stats["corners"]
        assert list(c) == ["tool", "version", "style", "rule", "basis", "ideas", "arc", "chamfer", "found", "arcs", "mitred", "left", "skipped",
                           "left_reasons", "skipped_reasons", "corners", *ADDED], name
        rows = c["sidestep_rows"]
        assert c["sidesteps"] == c["straightened"] + c["sidesteps_left"] == len(rows), name
        assert [row["outcome"] for row in rows] == ["straightened"] * c["straightened"] + ["left"] * c["sidesteps_left"], name
        assert sum(c["sidestep_reasons"].values()) == c["sidesteps_left"] and list(c["sidestep_reasons"]) == sorted(c["sidestep_reasons"]), name
        for row in rows:
            keys = {"net", "layer", "at", "step_mm", "outcome"} | ({"slid_to", "corners_removed", "shorter_mm"} if row["outcome"] == "straightened"
                                                                    else {"reason", "why"})
            assert set(row) == keys and row["step_mm"] < SIDESTEP_WIDTHS * W, (name, row)
        entries = {e for t in r.tracks for e in t.provenance.derived_from if f":{STRAIGHTEN}:" in e}
        assert entries == {_entry(tuple(row["at"]), row["step_mm"]) for row in rows if row["outcome"] == "straightened"}, name
        assert c["found"] == c["arcs"] + c["mitred"] + c["left"] + c["skipped"] == len(c["corners"]), name
        routed = sum(track_length(t) for t in _poly("A", *pts))
        assert r.stats["net_length_mm"]["A"] == round(sum(track_length(t) for t in r.tracks), 6) <= round(routed, 6), name
        assert r.stats["track_count"] == len(r.tracks) and r.stats["total_length_mm"] == r.stats["net_length_mm"]["A"], name
        _tangent_everywhere(r)
    assert {name: _counts(_run(tmp_path / "counts" / name, lib, pts))[:3] for name, pts in BOARDS.items()} == {
        "beside_a_turn": (1, 1, 0), "turn_back": (1, 1, 0), "either_way": (1, 1, 0), "bump": (1, 1, 0), "unequal_bump": (2, 1, 1),
        "staircase": (3, 3, 0), "u": (1, 1, 0), "between_pads": (1, 0, 1)}


def test_the_pads_a_net_joins_are_the_same_before_and_after(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Straightening changes no connection. On every board whose copper joins its two pads: the net's pad-to-pad path
    (:mod:`ai_eda.tools.si.paths`) joins the same two pads as routed, as straightened and as rounded, never longer; the points the
    pass kept are joined exactly as they were; and the ``pcb.routing`` checks find the copper connected and clear."""
    seen = _watch(monkeypatch)
    joined = _poly("B", (1.2, 1.2), (1.2, 18.8))  # net B's two pads, joined out of the way: the checks read the whole board
    for name, pts in BOARDS.items():
        if name == "between_pads":
            continue  # its copper ends on a via, not on the second pad
        ir = board_ir(tmp_path / name / "board", lib, _parts(pts[0], pts[-1]), STEP_NETS, (20.0, 20.0))
        pads = net_pads(ir, lib)
        routed = [*_poly("A", *pts), *joined]
        r = _run(tmp_path / name, lib, pts, *joined)
        straightened = [Track(net=net, layer="F.Cu", start=a, end=b, width_mm=0.4, provenance=HAND) for net, a, b in seen]
        assert r.tracks[-1] is joined[0] and seen[-1] == ("B", (1.2, 1.2), (1.2, 18.8)), name
        lengths = []
        for copper in (routed, straightened, r.tracks):
            path, why = longest_path("A", [t for t in copper if t.net == "A"], [], pads.pads["A"], None, {}, None)
            assert path is not None and path.ends == ("R1.1", "R2.1"), (name, why)
            lengths.append(path.length_mm)
            kept = _track_ends(routed) & _track_ends(copper)
            assert {("A", "F.Cu", pts[0]), ("A", "F.Cu", pts[-1])} <= kept and _grouping(copper, kept) == _grouping(routed, kept), name
            checks = _routing_checks(ir, Routing(params=P, tracks=straight_pieces(copper), vias=[]), lib, tmp_path)
            assert checks["pcb.routing.connectivity"].status == ValidationStatus.PASS and checks["pcb.routing.clearance"].status == ValidationStatus.PASS, name
        assert lengths[2] <= lengths[1] <= lengths[0] + 1e-9, (name, lengths)


def test_a_merged_track_keeps_the_place_and_the_direction_of_the_earlier_one(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The track order is stable: a track joined from two keeps the place of the one drawn first, running the way that one ran, and
    the other nets' tracks stay where they were between them. Net B's track is drawn between the sidestep and the turn; net A's
    copper is drawn from the far pad, so the long track beyond the turn comes first and takes the slid sidestep in."""
    seen = _watch(monkeypatch)
    other, = _poly("B", (16.0, 12.0), (16.0, 16.0))
    drawn = _poly("A", *BESIDE_A_TURN[::-1])  # (14, 6) -> (4.2, 6) -> (4.2, 13) -> (4, 13) -> (4, 16)
    r = _pass(tmp_path, lib, [*drawn[:2], other, *drawn[2:]], parts=_parts((14.0, 6.0), (4.0, 16.0)), nets=STEP_NETS)
    assert seen == [("A", (14.0, 6.0), (4.0, 6.0)), ("B", (16.0, 12.0), (16.0, 16.0)), ("A", (4.0, 6.0), (4.0, 16.0))]
    assert [t.net for t in r.tracks] == ["A", "A", "B", "A"] and r.tracks[2] is other  # the arc follows the first track of its corner
    assert (r.tracks[0].start, r.tracks[3].end) == ((14.0, 6.0), (4.0, 16.0))


def test_a_move_whose_track_the_rounding_uses_up_keeps_its_entry_on_the_copper(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The rounding may use a straightened track up: two arcs that meet on it take it whole. Its straightening entries then pass to
    the arc that used it up, so every move stays on the copper and the stage report counts what the stats count. Pad - 6.7 mm up -
    2 mm aside - 4.2 mm down - 1 mm back - 2.6 mm down - 2.2 mm aside - 3.2 mm down - pad, net B's via beside the run: the 1 mm
    track slides to the bottom (the way up passes 0.2 mm from the via), the 1.2 mm track it leaves there slides back up the 6.8 mm
    side and joins the 2 mm track - one 3.2 mm track between two turns, which the two cap-radius arcs then use up. The first
    move's entry was only on that track (the second move's is also on the long one beside it)."""
    seen = _watch(monkeypatch)
    pts = ((4.0, 2.0), (4.0, 8.7), (6.0, 8.7), (6.0, 12.9), (5.0, 12.9), (5.0, 15.5), (7.2, 15.5), (7.2, 18.7))
    via = Via(net="B", x_mm=4.2, y_mm=11.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    r = _run(tmp_path, lib, pts, vias=(via,))
    assert _path(seen) == [((4.0, 2.0), (4.0, 8.7)), ((4.0, 8.7), (7.2, 8.7)), ((7.2, 8.7), (7.2, 18.7))]
    first, second = _entry((5.5, 12.9), 1.0), _entry((6.6, 15.5), 1.2)
    assert [(row["at"], row["step_mm"], row["outcome"]) for row in _steps(r)] == [([5.5, 12.9], 1.0, "straightened"), ([6.6, 15.5], 1.2, "straightened")]
    assert [row["tangent_mm"] for row in _rows(r)] == [CAP, CAP] and len(_shape(r)) == 4  # the 3.2 mm track between the two arcs is gone
    low, high = "corner:routing.corners 0.2:arc:at=4.0,8.7:tangent=1.6:radius=1.6", "corner:routing.corners 0.2:arc:at=7.2,8.7:tangent=1.6:radius=1.6"
    assert [[e for e in t.provenance.derived_from if e.startswith("corner:")] for t in r.tracks] == [[low], [low], [first, second, high], [second, high]]
    assert "routing.corners 0.2: 2 sidestep(s) of the router on this net straightened, then 1 corner(s)" in r.tracks[2].provenance.note
    entries = {e for t in r.tracks for e in t.provenance.derived_from if f":{STRAIGHTEN}:" in e}
    assert entries == {first, second} and r.stats["corners"]["straightened"] == len(entries)
    assert any("옆걸음(sidestep: 두 직각 모서리 사이의 짧은 트랙) 2 곳을 곧게 폈습니다 (길이 1–1.2 mm; " in line for line in stages._corner_lines(r.tracks, 0.25, 0.25))


def test_the_rounding_reads_the_ends_a_move_gave_a_track_not_the_ends_it_was_routed_with(tmp_path: Path, lib: KicadLibrary,
                                                                                       monkeypatch: pytest.MonkeyPatch):
    """The rounding never uses a track up to an end the pass keeps, and it reads those ends from the track as the straightening left
    it. Pad - 6 mm up - 1 mm aside - 0.4 mm down - 1 mm aside - 5.6 mm down - pad, the copper drawn so that the 0.4 mm sidestep
    slides to the second turn (the 1 mm track beyond it comes first in the track order): the first 1 mm track is lengthened to
    2 mm, from (4, 10) to (6, 10) - where it was routed it ended at (5, 10), its middle now. The two turns then get one arc each,
    1 mm (half of the 2 mm) at both, and the two arcs meet in the middle of the track they use up: a half circle. Read with its
    routed ends, the second arc would have stopped short of (5, 10) as if the pass kept that point."""
    seen = _watch(monkeypatch)
    up, aside, step, back, down = (_poly("A", *pts)[0] for pts in (((4.0, 4.0), (4.0, 10.0)), ((4.0, 10.0), (5.0, 10.0)), ((5.0, 10.0), (5.0, 10.4)),
                                                                    ((5.0, 10.4), (6.0, 10.4)), ((6.0, 10.4), (6.0, 16.0))))
    r = _pass(tmp_path, lib, [down, back, step, aside, up], parts=_parts((4.0, 4.0), (6.0, 16.0)), nets=STEP_NETS)
    assert seen == [("A", (6.0, 10.0), (6.0, 16.0)), ("A", (4.0, 10.0), (6.0, 10.0)), ("A", (4.0, 4.0), (4.0, 10.0))]
    assert [(row["at"], row["outcome"], row.get("slid_to")) for row in _steps(r)] == [([5.0, 10.2], "straightened", [6.0, 10.4]), ([5.0, 10.0], "left", None)]
    assert [(row["at"], row["outcome"], row["radius_mm"]) for row in _rows(r)] == [([6.0, 10.0], "arc", 1.0), ([4.0, 10.0], "arc", 1.0)]
    assert [len(piece) for piece in _shape(r)] == [2, 3, 3, 2] and _shape(r)[1][2] == _shape(r)[2][0] == (5.0, 10.0)
    _tangent_everywhere(r)


def test_the_sweeps_are_bounded_and_the_stats_say_when_the_bound_cut_the_work_short(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Every move removes at least two corners, so the sweeps end by themselves; :data:`SIDESTEP_SWEEPS` only bounds them. The
    staircase needs two sweeps that move something. With the bound at one sweep, the sidestep that could still be slid is left and
    counted with the reason ``sweep_cap``; the copper is legal either way."""
    assert SIDESTEP_SWEEPS == 32 and f"(at most {SIDESTEP_SWEEPS} sweeps)" in _run(tmp_path, lib, BOARDS["staircase"]).stats["corners"]["straighten"]
    seen = _watch(monkeypatch)
    monkeypatch.setattr(corners, "SIDESTEP_SWEEPS", 1)
    r = _run(tmp_path / "one", lib, BOARDS["staircase"])
    assert _counts(r) == (3, 2, 1, {"sweep_cap": 1})
    assert _path(seen) == [((4.0, 16.0), (4.0, 12.0)), ((4.0, 12.0), (4.6, 12.0)), ((4.6, 12.0), (4.6, 6.0)), ((4.6, 6.0), (14.0, 6.0))]
    assert _steps(r)[2] == {"net": "A", "layer": "F.Cu", "at": [4.3, 12.0], "step_mm": 0.6, "outcome": "left", "reason": "sweep_cap",
                            "why": "it could still be slid, but the step stops after 1 sweeps"}
    monkeypatch.setattr(corners, "SIDESTEP_SWEEPS", 2)
    assert _counts(_run(tmp_path / "two", lib, BOARDS["staircase"])) == (3, 3, 0, {})


@needs_kicad
def test_kicad_finds_no_clearance_edge_or_connection_problem_on_straightened_copper(tmp_path: Path, lib: KicadLibrary):
    """Canary (kicad-cli): two two-pad parts, net A drawn as the staircase between their first pads - three sidesteps slid away, the
    one corner rounded - and net B joining their second pads. Compiled, DRC reports no clearance, edge, hole, short or connection
    finding: the straightened copper is legal to KiCad too, and still joins what it joined."""
    parts = [("R1", "SMD2", 5.5, 16.0), ("R2", "SMD2", 15.5, 6.0)]  # pads 1 at (4, 16) and (14, 6), pads 2 at (7, 16) and (17, 6)
    nets = {"A": [("R1", "1"), ("R2", "1")], "B": [("R1", "2"), ("R2", "2")]}
    drawn = [*_poly("A", *BOARDS["staircase"]), *_poly("B", (7.0, 16.0), (7.0, 8.0), (17.0, 8.0), (17.0, 6.0))]
    r = _pass(tmp_path, lib, drawn, parts=parts, nets=nets)
    assert _counts(r) == (3, 3, 0, {}) and _shape(r)[0] == ((4.0, 16.0), (4.0, 7.6)) and len(_arcs(r, "A")) == 1 and len(_arcs(r, "B")) == 2
    ir = board_ir(tmp_path / "board", lib, parts, nets, (20.0, 20.0))
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), []
    out = tmp_path / "out"
    art = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": lib}))
    kicad.run_drc(Path(art.path), out / "drc.json")
    report = json.loads((out / "drc.json").read_text(encoding="utf-8"))
    bad = {"clearance", "copper_edge_clearance", "hole_clearance", "shorting_items", "track_dangling", "tracks_crossing"}
    assert [v for v in report["violations"] if v["type"] in bad] == [] and report["unconnected_items"] == []


# --------------------------------------------------------------------------- what must not change


#: SHA-256 of :func:`tests.test_routing_rules.canonical` (copper with its provenance, stats, parameters) of the arc style on the
#: hand-drawn fixtures of ``tests/test_routing_arcs.py``, produced by routing.corners 0.2 before the straightening existed (never
#: regenerated). None of that copper holds a sidestep the step can move, so the step must leave every byte - the keys it adds to
#: ``stats["corners"]`` apart, which are taken out before the bytes are compared
ARC_BEFORE_STRAIGHTENING: dict[str, str] = {
    "l": "9fa62e5d3ee6d6d84e2699b17867479561b58c689ff409926e06ab9474dfffe8",
    "u": "0f90197dd67630432245a88eb92a1beb52b3c3e5f976141de755615f455a4c39",
    "z": "43bea567c43ba61834e9c05b3bc347fe7b4f5972815b814af2e284d5d7e52298",
    "acute": "9424ee70d6f49a4efce5e611cc510eb7a15ba4e0baa425cb8903266b0db50c53",
    "jog": "22f39382c9f047b6958d699bbca09d9bcace704d687009da7fd1cd333791e6bf",
    "chamfer_beside_arc": "7cd8115db0f147f7c2fc076c9ae0ca9c83cb14e8df9441e4b7cd40eaeb3b2403",
    "bus_abc": "d8179139a48df51b06148a656f12ff86b8d7f9c5328eec01751352cb27a0c4df",
    "bus_cba": "0545cd5488129efbea780a9155a160610cab93de7e18995ee8434262cb68253b",
    "bus_at_the_limit": "f34ede41bb05a3caa4b455702e57cb6aee7f1f6df9ee9215b13124d099c95e18",
    "foreign_via": "3fcf2ca3cf71fcf449afd2c5e2fd1367876db578c5592d8dff158df0b76593e8",
    "foreign_pad": "13ebdf66ce477224417cbfd97d975bd53b442fd177e7075044f69d6856ee1c38",
    "hole": "ef7af1dfb8adf0c4cabd025518d93895d8ab5f792d12fc39651446fc61308c34",
    "keepout": "f305b66c5d61184c42520896d71c98031c3a341704c686fa4d5102683e5bd4b7",
    "own_via": "279e5d844da879f887d29a652e5ecbe482f29ddde43e3cb22dd756b4290b7a44",
    "own_track": "c63c062a18de0ee75716c4a4950036e493d4e322daf22dceb059e5b7457fb085",
    "own_pad_at_the_corner": "c89985303b604512342d9c3c5a0e150749517929d9da24b440b30d4c4c40f6ca",
    "own_pad_beside": "e086156feff6298d9697ff3ac782a2e391e64663ff3ae8336d0b1ca2586527d2",
}
#: the same for ``"miter45"`` on the boards of this file (:data:`BOARDS`), produced by routing.corners 0.1 on the code before the
#: straightening existed (never regenerated): the 45-degree style does not straighten
MITER45_BEFORE_STRAIGHTENING: dict[str, str] = {
    "beside_a_turn": "62ffd3d231777f0362cd56f6681b7fddc550df4bc67ea25ade590f385224be30",
    "turn_back": "580d43645a080cff40e8564aa0f644a8effc1a82d423f4c619a8b5873db0990c",
    "either_way": "cc546f29d4817d5f314ae4d87dd1cf940ec538801518d98f8ac3396e790f6722",
    "bump": "a3f571ea90b74fc73d65261a121bc9c37b042878754f8bf3c866f71499c5373a",
    "unequal_bump": "898a6d5e073d296026fcfa3b4fc6d96cdfda2b2a36eddbc582129653e61ebd63",
    "staircase": "b87ffa3d73d175365fe4a59fdb711270678afed98cbc36cb49ada6cac5b850ce",
    "u": "5d7c5460f358f59625e3e9fd3915af480e7cd271e848f775f322ae1b4bf6a4af",
    "between_pads": "a26f909949ffc1c852ad97b36c0367a97826d219742187105ddcae365fc99bb8",
}


def test_the_arc_style_on_copper_without_a_sidestep_it_can_move_gives_what_it_gave_before(tmp_path: Path, lib: KicadLibrary):
    """Every fixture of ``tests/test_routing_arcs.py`` through the arc style: the copper, its entries and notes, the corner rows and
    counts, the lengths - byte for byte what routing.corners 0.2 gave before it straightened (:data:`ARC_BEFORE_STRAIGHTENING`),
    once the keys the step adds are taken out of ``stats["corners"]``. The step moved nothing there: the U and the Z of those
    fixtures have a 3 mm track between their corners - a sidestep by its length - whose neighbours end on track ends the pass
    keeps."""
    results = _fixtures(tmp_path, lib, ARC)
    counts = {}
    for name, r in results.items():
        c = r.stats["corners"]
        counts[name] = _counts(r)
        assert list(c)[-len(ADDED):] == ADDED and c["straightened"] == 0, name
        r.stats["corners"] = {key: value for key, value in c.items() if key not in ADDED}
    assert {name: hashlib.sha256(canonical(r)).hexdigest() for name, r in results.items()} == ARC_BEFORE_STRAIGHTENING
    assert {name: found for name, found in counts.items() if found[0]} == {"u": (1, 0, 1, {"kept_point": 1}), "z": (1, 0, 1, {"kept_point": 1})}


def test_miter45_and_orthogonal_do_not_straighten(tmp_path: Path, lib: KicadLibrary):
    """The straightening belongs to the arc style (routing.corners 0.2). ``"miter45"`` is routing.corners 0.1 byte for byte on the
    boards of this file (:data:`MITER45_BEFORE_STRAIGHTENING`): the sidestep stays where the router drew it, no key and no entry of
    the step appears. ``"orthogonal"`` leaves the result untouched."""
    digests = {}
    for name, pts in BOARDS.items():
        r = _run(tmp_path / name, lib, pts, style=MITER45)
        c = r.stats["corners"]
        assert (c["version"], c["style"]) == ("0.1", "miter45") and not set(ADDED) & set(c), name
        assert not any(STRAIGHTEN in e for t in r.tracks for e in t.provenance.derived_from), name
        digests[name] = hashlib.sha256(canonical(r)).hexdigest()
        tracks = _poly("A", *pts)
        untouched = _pass(tmp_path / name / "orthogonal", lib, tracks, style=ORTHOGONAL, parts=_parts(pts[0], pts[-1]), nets=STEP_NETS)
        assert all(t is u for t, u in zip(untouched.tracks, tracks)) and "corners" not in untouched.stats, name
    assert digests == MITER45_BEFORE_STRAIGHTENING
    mitred = _run(tmp_path / "shape", lib, BESIDE_A_TURN, style=MITER45)
    assert _shape(mitred)[:3] == [((4.0, 16.0), (4.0, 13.0)), ((4.0, 13.0), (4.2, 13.0)), ((4.2, 13.0), (4.2, 9.5))]  # the sidestep, as routed


# --------------------------------------------------------------------------- the report, the agent, the document


def test_the_stage_report_counts_a_straightened_sidestep_as_what_it_is(tmp_path: Path):
    """A ``straighten`` record is a sidestep removed, not a corner rounded or mitred: the keep-out section's corner lines give it a
    sentence of its own - how many, how long, how the new copper was checked - and leave the counts of arcs and chamfers alone; a
    record of that kind the report cannot read is unread, as every other."""
    step, longer = _entry((49.3, 35.2), 0.2), _entry((38.6, 47.9), 0.6)
    assert stages._corner_record(step) == ("0.2", "straighten", 0.2) and stages._corner_record(longer) == ("0.2", STRAIGHTEN, 0.6)
    sentence = (
        "이 보드의 구리에서 `routing.corners` 0.2 는 모서리를 둥글리기 전에 라우터가 남긴 옆걸음(sidestep: 두 직각 모서리 사이의 짧은 트랙) 2 곳을 곧게 폈습니다 "
        "(길이 0.2–0.6 mm; 트랙 provenance 의 `corner:` 항목 중 `straighten`; 프로젝트 소유자의 모서리 규칙 — 전기적 요구가 아닌 기하 관례). "
        "옆걸음은 의도한 패턴이나 길이 맞춤이 아니라 격자 탐색이 남긴 것입니다. 그 짧은 트랙을 이웃 트랙을 따라 다음 모서리까지 밀어 거기서 나가는 트랙과 하나로 합쳤고, "
        "패드·비아는 그대로이고 접합점은 제 직선 위에서만 움직이며, 넷의 길이는 라우터가 놓은 직선 구리보다 늘지 않습니다(없앤 모서리의 둥글리기 몫만큼은 줄지 않으므로, 펴지 않고 둥글린 구리보다는 조금 길 수 있습니다). 새로 놓인 구리는 격자 여유 g/2 없이 정확한 기하로 검사했습니다: 다른 넷의 트랙·비아·패드 상자와 구리 사이 "
        "간격 c = 0.25 mm (두 넷 중 넷 규칙의 간격이 더 크면 그 값; 넷이 없는 패드·구멍과는 c + g/2 = 0.375 mm), 중심선에서 트랙 keep-out 까지 w/2, 가장자리까지 e + w/2. "
        "그래서 그 구리는 위 반경보다 가까울 수 있으며, 그 판정도 kicad-cli DRC 가 합니다. 곧게 펴지 못한 옆걸음은 트랙에 항목을 남기지 않아 여기서 세지 않습니다 "
        "(라우터의 배선 메모가 그 수를 적었다면 아래 단계 기록에 있습니다)."
    )
    # the move's entry is on every track it changed: one sidestep per distinct entry of a net and layer
    alone = stages._corner_lines([_cornered("A", "F.Cu", step), _cornered("A", "F.Cu", step, longer)], 0.25, 0.25)
    assert alone == [sentence, ""] and "원호" not in sentence and "모따기" not in sentence
    one, _blank = stages._corner_lines([_cornered("A", "F.Cu", step)], 0.25, 0.25)
    assert "옆걸음(sidestep: 두 직각 모서리 사이의 짧은 트랙) 1 곳을 곧게 폈습니다 (길이 0.2 mm; " in one
    # beside rounded and mitred corners: their sentence is the one it was, with their own counts
    tracks = [_cornered("A", "F.Cu", ARC_02_A), _cornered("A", "F.Cu", ARC_02_A, ARC_02_B), _cornered("B", "B.Cu", ARC_02_A), _cornered("B", "B.Cu", MITRE_02),
              _cornered("B", "B.Cu", step, longer)]
    assert stages._corner_lines(tracks, 0.25, 0.25) == [SENTENCE_02, "", sentence, ""]
    unread = ["corner:routing.corners 0.1:straighten:at=1.0,2.0:step=0.2", "corner:routing.corners 0.2:straighten:at=1.0,2.0:step=0.0",
              "corner:routing.corners 0.2:straighten:at=1.0,2.0:step=inf", "corner:routing.corners 0.2:straighten:at=1.0,2.0",
              "corner:routing.corners 0.2:straighten:at=1.0,2.0:step=0.2:radius=1.0"]
    assert [stages._corner_record(e) for e in unread] == [None] * len(unread)
    line, _blank = stages._corner_lines([_cornered("A", "F.Cu", *unread)], 0.25, 0.25)
    assert "형식을 읽지 못하는 `corner:` 항목이 5 곳 있습니다" in line and "곧게" not in line
    # in the circuit report itself: a board with one rounded corner, one of whose tracks a move had lengthened
    ir, lib, _arc = _arc_board(tmp_path)
    first = ir.pcb.tracks[0]
    ir.pcb.tracks[0] = first.model_copy(update={"provenance": first.provenance.model_copy(update={"derived_from": [*first.provenance.derived_from, step]})})
    section = stages.circuit_report(ir, lib, None).split("### 라우터의 keep-out 규칙")[1].split("###")[0]
    assert "둥근 모서리(원호 트랙)가 1 곳 (반지름 2 mm), 원호를 넣지 못해 45도 모따기로 바꾼 모서리가 0 곳 있습니다" in section
    assert "옆걸음(sidestep: 두 직각 모서리 사이의 짧은 트랙) 1 곳을 곧게 폈습니다 (길이 0.2 mm; 트랙 provenance 의 `corner:` 항목 중 `straighten`;" in section


def test_the_pcb_agent_says_how_many_sidesteps_were_straightened(tmp_path: Path, lib: KicadLibrary):
    """Under ``pcb.corners=arc`` - the agent's default style - the routing description counts the sidesteps after the corners; the
    ``pcb.corners=miter45`` description is the one it was. The counts are the pass's: a description written from a result whose copper
    held sidesteps names them."""
    ir = _l_board(tmp_path, lib)
    for answers in ({CORNERS_KEY: "arc"}, {}):
        res = PCBAgent().run(ir, _ctx(tmp_path, lib, **answers))
        assert any(n.endswith("0 left as routed, 0 skipped; sidesteps: 0 of 0 sidestep(s) of the router (a short track between two right-angle "
                              "corners) straightened before the corners were rounded, 0 left as routed") for n in res.notes), res.notes
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{CORNERS_KEY: "miter45"}))
    assert not any("sidestep" in n for n in res.notes) and any(n.endswith("0 left as routed, 0 skipped") for n in res.notes), res.notes
    unrouted = ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"tracks": [], "vias": []})})
    routing = route_board(unrouted, lib, corner_style=ARC)
    routing.stats["corners"] = _run(tmp_path / "hand", lib, BOARDS["unequal_bump"]).stats["corners"]
    assert PCBAgent._routing_description(routing).endswith(
        "; sidesteps: 1 of 2 sidestep(s) of the router (a short track between two right-angle corners) straightened before the corners were rounded, "
        "1 left as routed")


def test_the_architecture_document_names_the_straightening():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "둥글리기 전에 라우터가 남긴 옆걸음" in doc and "`straighten` 항목" in doc and "`sidesteps`·`straightened`·`sidesteps_left`·`sidestep_reasons`" in doc


# --------------------------------------------------------------------------- a run end that takes the sidestep


def test_a_junction_on_a_straight_line_takes_the_sidestep_and_moves_along_that_line(tmp_path: Path, lib: KicadLibrary,
                                                                                     monkeypatch: pytest.MonkeyPatch):
    """The 9 V piano board's GND sidestep at (67, 40): a run from a pad steps 0.2 mm aside and ends on a straight track of its net
    that runs along the sidestep's line (a T). Sliding the sidestep onto that track lays nothing new there - the track already
    covers it -, so the junction moves 0.2 mm along the track and the run is one straight track. Where the line's copper does not
    cover the slid sidestep whole (the track ends short of it), the junction stays a point the pass keeps, as before."""
    seen = _watch(monkeypatch)
    nets = {"A": [("R1", "1"), ("R2", "1"), ("R5", "1")], "B": STEP_NETS["B"]}
    line = _poly("A", (2.0, 6.0), (14.0, 6.0))
    r = _pass(tmp_path, lib, [*_poly("A", (4.0, 16.0), (4.0, 13.0), (4.2, 13.0), (4.2, 6.0)), *line],
              parts=_parts((4.0, 16.0), (14.0, 6.0), ("R5", "SMD1", 2.0, 6.0)), nets=nets)
    assert _path(seen) == [((4.0, 16.0), (4.0, 6.0)), ((2.0, 6.0), (14.0, 6.0))]
    assert _counts(r) == (1, 1, 0, {})
    row, = _steps(r)
    assert (row["outcome"], row["at"], row["slid_to"], row["corners_removed"], row["shorter_mm"]) == ("straightened", [4.1, 13.0], [4.2, 6.0], 2, 0.2)
    assert any(_entry((4.1, 13.0), 0.2) in t.provenance.derived_from for t in r.tracks)
    short = _poly("A", (4.1, 6.0), (14.0, 6.0))  # the line ends at x = 4.1: the slid sidestep (4 .. 4.2) would leave its copper
    r = _pass(tmp_path / "short", lib, [*_poly("A", (4.0, 16.0), (4.0, 13.0), (4.2, 13.0), (4.2, 6.0)), *short],
              parts=_parts((4.0, 16.0), (14.0, 6.0)), nets=STEP_NETS)
    assert _counts(r) == (1, 0, 1, {"kept_point": 1}) and ((4.0, 13.0), (4.2, 13.0)) in _shape(r)


def test_a_through_hole_pad_takes_the_sidestep_into_its_ring_or_is_met_from_the_side(tmp_path: Path, lib: KicadLibrary,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    """The 9 V piano board's VREF sidestep at (41.5, 51.4): a run leaves a through-hole pad, steps aside and runs on; its other way
    is blocked (a via of the net on the copper the move would take away). A plated hole is soldered through its barrel, so where
    the track meets its ring does not matter: a 0.3 mm step slides into the pad - it lies within the pad's inscribed circle (0.8 mm)
    and goes, the run now leaves the pad 0.3 mm off its centre, straight; a 1.0 mm step is longer than that, so the slid sidestep
    stays and meets the pad from its side (one corner fewer). A surface-mount pad in the same place stays a point the pass keeps."""
    seen = _watch(monkeypatch)
    via = Via(net="A", x_mm=4.3, y_mm=9.0, drill_mm=0.4, diameter_mm=0.8, provenance=HAND)
    pts = ((4.0, 16.0), (4.0, 13.0), (4.3, 13.0), (4.3, 6.0), (14.0, 6.0))
    tht = [("R1", "PAD1", 4.0, 16.0), ("R2", "SMD1", 14.0, 6.0), ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    r = _pass(tmp_path, lib, _poly("A", *pts), vias=(via,), parts=tht, nets=STEP_NETS)
    assert _path(seen) == [((4.3, 16.0), (4.3, 6.0)), ((4.3, 6.0), (14.0, 6.0))] and _counts(r) == (1, 1, 0, {})
    assert _steps(r)[0]["slid_to"] == [4.0, 16.0]
    side = ((4.0, 16.0), (4.0, 13.0), (5.0, 13.0), (5.0, 6.0), (14.0, 6.0))
    r = _pass(tmp_path / "side", lib, _poly("A", *side), vias=(via.model_copy(update={"x_mm": 5.0}),), parts=tht, nets=STEP_NETS)
    assert _path(seen) == [((4.0, 16.0), (5.0, 16.0)), ((5.0, 16.0), (5.0, 6.0)), ((5.0, 6.0), (14.0, 6.0))] and _counts(r) == (1, 1, 0, {})
    smd = [("R1", "SMD1", 4.0, 16.0), *tht[1:]]
    r = _pass(tmp_path / "smd", lib, _poly("A", *pts), vias=(via,), parts=smd, nets=STEP_NETS)
    assert _counts(r) == (1, 0, 1, {"kept_point": 1}) and _path(seen) == list(zip(pts, pts[1:]))


# --------------------------------------------------------------------------- a lane change at a junction (since 2026-11-19)


def test_a_lane_change_at_a_t_on_a_track_is_slid_so_the_lane_runs_straight_through(tmp_path: Path, lib: KicadLibrary,
                                                                                   monkeypatch: pytest.MonkeyPatch):
    """The 9 V piano board's VS run (rev 3, at R50.1): the run came from the east at y 26.75 and ended on the stub down to the pad,
    0.25 mm before that stub's corner; the run went on west from the corner at y 27 - a lane change at a T on a track's interior, no
    junction of track ends, so no candidate. Now the part of the stub between the T and its corner is a step: the arriving track is
    carried through the T along the corner's neighbour (the stub is cut back to the T), and the lane is one line."""
    seen = _watch(monkeypatch)
    parts = [("R1", "SMD1", 16.0, 6.0), ("R2", "SMD1", 10.0, 2.0), ("R5", "SMD1", 3.0, 14.0), ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    tracks = [*_poly("A", (16.0, 6.0), (10.0, 6.0)), *_poly("A", (10.0, 6.25), (10.0, 2.0)), *_poly("A", (10.0, 6.25), (3.0, 6.25), (3.0, 14.0))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((16.0, 6.0), (3.0, 6.0)), ((10.0, 6.0), (10.0, 2.0)), ((3.0, 6.0), (3.0, 14.0))], _path(seen)
    row, = _steps(r)
    assert _counts(r) == (1, 1, 0, {}) and (row["at"], row["step_mm"], row["junction"], row["corners_removed"], row["shorter_mm"]) == (
        [10.0, 6.125], 0.25, [10.0, 6.0], 2, 0.0), row  # a Z: the step slid to the west corner continues the track there


def test_a_lane_change_at_a_t_may_move_the_arriving_lane_onto_the_corners_line(tmp_path: Path, lib: KicadLibrary,
                                                                              monkeypatch: pytest.MonkeyPatch):
    """The piano board's OSC4 / OSC8 runs (rev 3, at R24.1 / R28.1): the lane from the north at x 46.25 ended on the stub to the pad
    0.25 mm east of the stub's corner, where the lane on south left at x 46.0; the southern lane cannot move (its far end is a pad).
    The northern lane is slid onto the corner's line instead - the step between the T and the corner stays, as the stub it is, and
    the corner becomes the T: the lane is one line, 0.25 mm shorter."""
    seen = _watch(monkeypatch)
    parts = [("R1", "SMD1", 1.0, 3.0), ("R2", "SMD1", 7.0, 10.0), ("R5", "SMD1", 4.0, 16.0), ("R3", "SMD1", 18.8, 1.2), ("R4", "SMD1", 18.8, 18.8)]
    tracks = [*_poly("A", (1.0, 3.0), (4.25, 3.0), (4.25, 10.0)), *_poly("A", (4.0, 10.0), (7.0, 10.0)), *_poly("A", (4.0, 10.0), (4.0, 16.0))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((1.0, 3.0), (4.0, 3.0)), ((4.0, 10.0), (7.0, 10.0)), ((4.0, 3.0), (4.0, 16.0))], _path(seen)
    row, = _steps(r)
    assert _counts(r) == (1, 1, 0, {}) and (row["junction"], row["slid_to"], row["shorter_mm"]) == ([4.25, 10.0], [4.25, 3.0], 0.25), row


def test_a_junction_and_a_corner_in_a_through_hole_pad_let_the_run_pass_straight_through(tmp_path: Path, lib: KicadLibrary,
                                                                                         monkeypatch: pytest.MonkeyPatch):
    """The piano board's GND feed past Q2.1 (rev 3): the run came up at x 10.0 into the pad's box, met the stub to the pad's centre
    there and went on north 0.25 mm aside, at x 9.75 - the junction and the corner both inside the pad's box, so both points the pass
    kept. A plated hole is soldered through its barrel (as for a run end in such a pad): the junction may lie in its box and so may
    the step's corner; the run is carried straight through and the step goes. A surface-mount pad there stays a point the pass keeps."""
    seen = _watch(monkeypatch)
    parts = [("R1", "SMD1", 10.1, 16.0), ("R2", "PAD1", 10.0, 10.05), ("R5", "SMD1", 16.0, 4.0), ("R3", "SMD1", 1.2, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    tracks = [*_poly("A", (10.1, 16.0), (10.1, 10.0)), *_poly("A", (10.1, 10.0), (10.0, 10.05)),
              *_poly("A", (10.1, 10.0), (9.85, 10.0), (9.85, 4.0), (16.0, 4.0))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((10.1, 16.0), (10.1, 4.0)), ((10.1, 10.0), (10.0, 10.05)), ((10.1, 4.0), (16.0, 4.0))], _path(seen)
    assert _counts(r) == (1, 1, 0, {}) and _steps(r)[0]["junction"] == [10.1, 10.0]
    smd = [*parts[:1], ("R2", "SMD1", 10.0, 10.05), *parts[2:]]
    r = _pass(tmp_path / "smd", lib, tracks, parts=smd, nets=JUNCTION_NETS)
    assert _counts(r) == (0, 0, 0, {}) and _path(seen)[2:4] == [((10.1, 10.0), (9.85, 10.0)), ((9.85, 10.0), (9.85, 4.0))]


def test_a_lane_change_at_a_junction_may_slide_the_junctions_lane_where_the_step_goes_on_as_a_branch(tmp_path: Path, lib: KicadLibrary,
                                                                                                     monkeypatch: pytest.MonkeyPatch):
    """The piano board's VBAT run (rev 3, the T at (57.25, 28.75)): the lane from the west at y 28.75 met a branch going north, stepped
    0.75 mm south along that branch's line and ran on east at y 29.5 to a via. The step could not slide east (the via is a point the
    pass keeps); now the junction's lane may be slid onto the corner's line instead, the step staying as the branch's copper - here
    0.5 mm shorter, the bend at the west end taking the difference. Where the west end continues the lane's direction (a Z) the move
    would lengthen the net by the step: the step stays, with the reason ``length`` (the pass never lengthens a net)."""
    seen = _watch(monkeypatch)
    parts = [("R1", "SMD1", 2.0, 14.0), ("R2", "SMD1", 10.0, 2.0), ("R5", "SMD1", 16.0, 6.5), ("R3", "SMD1", 18.8, 1.2), ("R4", "SMD1", 18.8, 18.8)]
    tracks = [*_poly("A", (2.0, 14.0), (2.0, 6.0), (10.0, 6.0)), *_poly("A", (10.0, 6.0), (10.0, 2.0)), *_poly("A", (10.0, 6.0), (10.0, 6.5), (16.0, 6.5))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=JUNCTION_NETS)
    assert _path(seen) == [((2.0, 14.0), (2.0, 6.5)), ((10.0, 6.0), (10.0, 2.0)), ((10.0, 6.0), (10.0, 6.5)), ((2.0, 6.5), (16.0, 6.5))], _path(seen)
    row, = _steps(r)
    assert _counts(r) == (1, 1, 0, {}) and (row["junction"], row["slid_to"], row["shorter_mm"]) == ([10.0, 6.0], [2.0, 6.0], 0.5), row
    parts[0] = ("R1", "SMD1", 2.0, 2.0)
    z = [*_poly("A", (2.0, 2.0), (2.0, 6.0), (10.0, 6.0)), *tracks[2:]]
    r = _pass(tmp_path / "z", lib, z, parts=parts, nets=JUNCTION_NETS)
    assert _counts(r) == (1, 0, 1, {"length": 1}) and _path(seen) == [(t.start, t.end) for t in z]
    assert "would lengthen its net by 0.5 mm (the step stays as the branch's copper)" in _steps(r)[0]["why"]


def test_two_lane_changes_at_a_through_hole_pad_and_a_t_beyond_it_become_one_line(tmp_path: Path, lib: KicadLibrary,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    """The piano board's GND run past C24.2 (rev 3): from the north at x 10 into the pad's box, 0.5 mm east to the stub to the pad's
    centre, on south at x 10.5 to a T on a line from the west, then south again at x 10.5 below it. First the southern lane is slid onto
    the corner's line (the step stays as the stub's copper); its end merges into the line from the west, and the junction's other
    track there, the lane on south, keeps its contact through that line. That leaves a lane change at the T on the line, slid as a
    step at a T: the run is one line from the north to the bend at the bottom."""
    seen = _watch(monkeypatch)
    nets = {"A": [("R1", "1"), ("R2", "1"), ("R5", "1"), ("R6", "1")], "B": [("R3", "1"), ("R4", "1")]}
    parts = [("R1", "SMD1", 10.0, 1.5), ("R2", "PAD1", 10.55, 6.0), ("R5", "SMD1", 4.0, 9.0), ("R6", "SMD1", 16.0, 14.0),
             ("R3", "SMD1", 18.8, 1.2), ("R4", "SMD1", 1.2, 18.8)]
    tracks = [*_poly("A", (10.0, 1.5), (10.0, 6.0), (10.5, 6.0)), *_poly("A", (10.5, 6.0), (10.55, 6.0)), *_poly("A", (10.5, 6.0), (10.5, 9.0)),
              *_poly("A", (4.0, 9.0), (10.5, 9.0)), *_poly("A", (10.5, 9.0), (10.5, 14.0), (16.0, 14.0))]
    r = _pass(tmp_path, lib, tracks, parts=parts, nets=nets)
    assert _path(seen) == [((10.0, 1.5), (10.0, 14.0)), ((10.0, 6.0), (10.5, 6.0)), ((10.5, 6.0), (10.55, 6.0)), ((4.0, 9.0), (10.0, 9.0)),
                           ((10.0, 14.0), (16.0, 14.0))], _path(seen)
    assert _counts(r) == (2, 2, 0, {}) and [row["junction"] for row in _steps(r)] == [[10.5, 6.0], [10.0, 9.0]]
