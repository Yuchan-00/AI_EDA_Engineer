"""Deterministic two-layer grid maze router with negotiated congestion and per-net rules (pure, no I/O beyond the KiCad library).

Invariant: every track and via produced here is a function of the IR's
placements, the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: pad positions, sizes,
layers - never model memory), the :class:`RoutingParams` and the caller's
:class:`NetRule` s. Nothing is estimated, nothing is guessed: a component
without a placement or footprint, a footprint that is not on disk, a net pin
without a pad, a copper layer other than ``F.Cu`` / ``B.Cu`` (inner
``In<k>.Cu`` layers only when the caller opts in), a pad too small for the
grid or a rule that makes no sense raises :class:`~ai_eda.errors.CompileError`
instead of getting a default.

What this is: a Lee / A* maze router on a square grid over the board
outline, ``F.Cu`` and ``B.Cu`` only, with through vias between them, inside a
PathFinder-style negotiated-congestion loop (rip-up and reroute). Inner
copper layers (``In1.Cu`` ... of a 4-layer board) are refused as 0.2 did,
unless the caller passes ``inner_layers=True``: then they are accepted and
never routed - they hold the planes (zones) the caller adds, a via crosses
them as a through via, and ``stats["inner_layers"]`` names them.

* **The search** (unchanged from 0.1 in kind): A* over ``(layer, cell,
  direction)`` states, so the bend cost is exact; a 4-neighbour step costs
  the cell's node cost, a change of direction adds ``bend_cost``, a layer
  change adds the via cost; the heuristic is ``base_cost`` times the
  Manhattan distance to the target (a lower bound: no step costs less than
  ``base_cost``); heap ties are broken by a monotonically increasing push
  counter, so the result is a pure function of its inputs. The search runs
  in a window around the net's terminals (``window_mm`` beyond their
  bounding box), doubled on failure until it covers the board.
* **Multi-terminal nets** grow a Steiner tree: from the terminal nearest the
  centroid of the net's pad centres, each step adds the unconnected terminal
  nearest (Manhattan, grid cells) to the tree so far - one A* from every
  tree cell to that terminal - until all terminals hang on the tree.
* **Negotiation**: iteration 1 routes every net in ``(pad count, name)``
  order, each seeing the others' copper not as an obstacle but as a cost:
  a cell's node cost is ``base_cost * (1 + history) * (1 + present *
  occupancy)``, occupancy being the number of other nets whose clearance
  halo covers the cell, and a via's cost ``(via_cost + base_cost) * (1 +
  history) * (1 + present * via occupancy) - base_cost`` plus the landing
  cell's congestion. After each iteration every over-used cell (a net's
  copper inside another net's halo) gains ``history_cost``; the present
  factor starts at ``present_cost`` and is multiplied by ``present_growth``
  per iteration; the next iteration rips up and reroutes, in the same fixed
  order, every net that owns an over-used cell. The loop stops when no cell
  is over-used (``stats["legal"]``) or after ``max_iterations``.
* **What is emitted**: only mutually legal copper. When the cap is reached
  with conflicts left, the net with the most conflict partners (then the
  fewer pads, then the name) is ripped up, repeatedly, until the rest is
  legal; each such net is then tried once more against the legal copper as
  a hard obstacle (0.1's rule: no cell of it - its terminal cells, where the
  search starts and ends, included - inside another net's halo) and kept
  only when that succeeds. A net that
  still has no legal route - or a terminal no search can reach even with
  every other net's copper ignored - is listed in :attr:`Routing.unrouted`
  with the reason and **no copper at all**: a half-routed net is never
  emitted. Whether a board with unrouted nets is applied is the caller's
  decision (:class:`~ai_eda.agents.pcb.PCBAgent`: all-or-nothing unless the
  user answered ``pcb.routing=partial``).

Obstacle model (board frame, mm, Y down; every emitted coordinate is rounded
with :func:`ai_eda.tools.kicad.geometry._q`). The static part is 0.1's:

* an *owner map* per layer says for every grid cell which net may use it
  (free, one net, or :data:`BLOCKED`). A cell closer than
  ``clearance + width/2 + grid/2`` to a pad's box (the box is exact for pads
  at multiples of 90 degrees and the circumscribed square otherwise, the
  ``grid/2`` covers a box edge that falls between two grid columns) is owned
  by the pad's net; a cell claimed by two nets, or near a pad without a net
  (unnumbered pads, NPTH holes, pins no net names), is BLOCKED. A cell closer
  than ``edge_clearance + width/2`` to the outline is BLOCKED on both layers.
  These cells are never entered by a foreign net, whatever the costs.
* the copper of a routed net is its *halo*, counted, not owned: every path
  cell covers the cells within ``width + clearance + grid/2`` on its layer
  (a foreign track centre there would come closer than ``clearance``) and
  the cells within ``via_diameter/2 + clearance + width/2`` in the via plane
  (a foreign via centre there would); every via covers the cells within
  ``via_diameter/2 + clearance + width/2`` on both layers and within
  ``via_diameter + clearance`` in the via plane. Every one of these relations
  is symmetric, so "no net's copper lies in another net's halo" is exactly
  "every copper pair of different nets keeps ``clearance``" - a legal result
  keeps every clearance of 0.1's model (a via's hard pad / edge rules below
  included).
* a via may sit where every cell within ``via_diameter/2 + clearance +
  width/2`` on *both* layers is free or the net's own on the static map, at
  least ``edge_clearance + via_diameter/2`` from the outline, and where its
  copper disc (radius ``via_diameter/2``) overlaps no pad box at all - the
  net's own pads included: a via is drilled next to a pad, never through it
  (no via-in-pad).
* a pad's terminal is the grid cell nearest to its centre; it must lie inside
  the pad's inscribed circle, so the track ending there overlaps the pad
  copper, and one stub segment joins it to the exact centre. The terminal
  cell obeys the owner map like every other cell: on a layer where it is
  BLOCKED (a foreign pad's keep-out or the board edge reaches it) or owned by
  another net it is neither a seed nor a target, and a terminal with no
  usable layer leaves the net unrouted with that reason - the router never
  emits copper that breaks its own clearance at a pad.
* pad shapes: ``circle`` / ``rect`` / ``oval`` / ``roundrect`` are convex and
  lie inside their ``(size)`` box, so the box is a conservative obstacle and
  the inscribed circle a safe terminal. A ``custom`` pad (its ``primitives``
  may reach beyond the anchor's size) or a ``trapezoid`` (``rect_delta``
  extends one side beyond the box) is refused (:class:`CompileError` naming
  the pad): the library reader keeps neither, and an obstacle box that is
  smaller than the copper would be a guess.

**Net rules (routing.maze 0.3).** :func:`route_board` takes ``rules``, net
name -> :class:`NetRule`, supplied by the caller (the IR's net classes are
mapped onto it elsewhere; this module reads no IR net class). A rule whose
fields other than ``net_class`` are all ``None`` is no rule. None of the
following runs for a board without rules: its copper, its stats, its
provenance and its refusals are routing.maze 0.2's byte for byte, stamped
:data:`ROUTER_VERSION` ``"0.2"``. A board
routed with at least one rule is stamped :data:`ROUTER_RULES_VERSION`
``"0.3"`` on every track and via; their ``derived_from`` records the 0.3
parameters (:meth:`RoutingParams.derived_from_entry` with ``rules=True``)
and, for a net with a rule, the effective rule (:meth:`NetRule.derived_from_entry`),
and ``stats["rules"]`` / ``["match_groups"]`` / ``["pairs"]`` say what each rule did.

* **Per-net width and clearance** inside the negotiation. Every distinct
  ``(width, clearance)`` is a *profile* with its own halo counts: in the
  plane of profile ``Q`` a path cell of width ``w`` / clearance ``c`` covers
  the cells within ``w/2 + w_Q/2 + max(c, c_Q) + grid/2``, a via of a net with
  clearance ``c`` the cells within ``via_diameter/2 + max(c, c_Q) + w_Q/2``;
  via planes are kept per clearance (``via_diameter/2 + max(c, c_Q) + w/2``
  from a path cell, ``via_diameter + max(c, c_Q)`` from a via). Each relation
  is symmetric in the two nets (half widths plus the larger clearance), so a
  legal result keeps every pair of nets at the larger of their clearances;
  with one profile these are 0.2's radii. A rule net's larger clearance also
  grows the owner map around its pads (every other net keeps it) and keeps
  every other net's via ``via_diameter/2 + c`` from them; its own track keeps
  ``max(c, c_pad) + w/2 + grid/2`` from every foreign pad box and
  ``edge_clearance + w/2`` from the outline (the net's *fence*), its vias
  ``via_diameter/2 + max(c, c_pad)``. A rule width (or neck-down width) below
  ``ir.pcb.manufacturing.min_track_width_mm`` and a rule clearance below the
  board's effective clearance are raised and recorded (``stats["rules"]``).
* **Neck-down**: a net with ``neckdown_width_mm`` narrows to it (at the
  board's clearance) within ``neckdown_radius_mm`` (default
  :attr:`RoutingParams.neckdown_radius_mm`) of the box of each of its pads
  whose terminal cell the full width's keep-out reaches - a foreign pad or
  width can use: the pad pitch cannot take the width there. A unit step with
  an end in that zone is emitted at the neck-down width, so the narrow copper
  reaches up to one grid step beyond the radius (the step that leaves the
  zone; each track's note says so); the neck-down pads and the length at the
  neck-down width are recorded per net.
* **Length budget**: ``max_length_mm`` bounds the net's routed length
  (tracks plus ``via_length_mm`` per via - the caller converts a delay budget
  and a via barrel into length). It is a hard bound in the search: a state
  whose length so far plus the Manhattan distance to its target exceeds what
  is left of the budget (after the stubs) is not expanded; when the
  congestion-costed search finds nothing within the budget, the shortest
  legal path (lengths only) is taken if it fits, and otherwise the net is
  unrouted with the shortest length named. A multi-pad net whose tree breaks
  the budget is grown again from each of its other pads, and the first tree
  within the budget is taken. Because the tree grows at the current
  negotiation costs, a reroute in a later iteration may find no tree within
  the budget where an earlier one did: the earlier route (within every rule)
  is then kept, still in conflict, and rerouted again or ripped up like any
  other conflicting net - a net that once had a route within its budget is
  never lost to one iteration's costs. The emitted copper is measured again
  and a net over its budget is unrouted, never emitted.
* **Length matching**: after the negotiation every routed net of a
  ``match_group`` shorter than the group's longest net by more than
  ``max_skew_mm`` gets square serpentine meanders on its longest straight
  run (legs :attr:`RoutingParams.meander_pitch_mm` apart, each bump at most
  :attr:`RoutingParams.meander_amplitude_mm` high, both grid multiples, the
  total the smallest multiple of ``2 * grid`` that reaches the window
  ``[longest - max_skew, longest]``). Every new cell is free on the owner
  map - never in a pad keep-out, the net's own included, so a bump is never
  drawn on (or against) its own pad, which would short it and add no
  electrical length - keeps the net's own width + clearance from its other
  copper (via_diameter/2 + clearance + width/2 from its vias), and obeys the
  net's fence and every other net's halo (legal against the other copper as
  an obstacle); the board is re-checked by the negotiation's over-use test
  and the meandered net's emitted copper by an exact geometric clearance
  audit (segment / box / disc distances, not the grid), and a meander that
  fails either is taken back. A group that cannot be matched is reported in
  ``stats["match_groups"]`` with the reason; its nets are still applied.
* **Coupled differential pairs**, only for pairs a rule declares
  (``pair_partner`` naming each other, both 2-pad nets): the pair is one
  virtual net in the negotiation whose centreline is routed on one layer as
  a track of width ``2w + s`` (the pair's copper envelope) at the pair's
  clearance, then offset into the two tracks at ``+/-(w + s)/2`` - the outer
  track of every corner mitred (chamfered between the two offset lines'
  feet at the corner cell), the inner one meeting at the offset lines'
  intersection. At each end the breakout is uncoupled: one straight segment
  from each pad centre to its offset track's start at a *launch* cell found
  by stepping from the pads' midpoint outward (up to ``window_mm``) in each
  of the four directions until the breakouts keep every clearance (pads,
  other nets' terminal cells, the edge, each other, the other track's first
  step) and ``pair_uncoupled_max_mm`` (a launch is kept only when some launch
  at the other end leaves each net's two breakouts within it); the centreline leaves and enters its
  launch cells straight along the launch direction, and both ends keep the
  same track on the same side. The breakouts are part of the virtual net's
  halo, so the negotiation keeps other nets off them. The length difference
  of the two tracks (corners, breakouts) above ``pair_max_skew_mm`` is
  compensated by small rectangular bumps outward on the shorter track
  (at most :attr:`RoutingParams.pair_bump_amplitude_mm` high,
  :attr:`RoutingParams.pair_bump_length_mm` long, on its longest straight
  coupled segment that holds them legally). The pair's final copper passes
  the exact clearance audit (the two tracks keep ``min(s, clearance)`` from
  each other) and each net's uncoupled length - its copper outside the
  coupled overlap, breakouts, mitred corners and bumps included, measured by
  :func:`ai_eda.tools.routing.coupling.uncoupled_lengths` exactly as
  ``si.diff`` measures it - is within ``pair_uncoupled_max_mm``, or neither
  net gets copper. The launch points, directions, breakout, coupled and
  uncoupled lengths and the skew before / after are in ``stats["pairs"]``.

What this is not: a DRC. The clearances above are the router's own
parameters; whether the board is valid is decided by ``kicad-cli pcb drc``
on the compiled board (:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), and
the ``pcb.routing.*`` validators only check the IR geometry against the
limits recorded in ``ir.pcb.manufacturing``. :func:`effective_params` raises
the width / clearance / via sizes / edge clearance to those limits when they
exist (whatever their provenance - a limit is a limit; the capability check
judges grounding) and :attr:`Routing.stats` says which were raised.
Impedance, delay and skew are judged by the SI checks, not here: the router
only honours the widths, spacings and lengths it is given.

Traceability: every :class:`~ai_eda.ir.Track` / :class:`~ai_eda.ir.Via`
carries ``derived`` provenance naming this router (:data:`ROUTER_ID` /
:data:`ROUTER_VERSION` or :data:`ROUTER_RULES_VERSION`), the net, the
placements of the net's components and every parameter in
``derived_from``; the note names the iteration count and how the net's route
was obtained. ``Provenance.inputs`` stays empty: it is the calculator role
map, and a track is not a calculator output.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Net, Provenance, ProvenanceKind, Track, Via
from ai_eda.tools.kicad.geometry import _q, pad_angle, pad_center, pad_layers
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, Pad
from ai_eda.tools.routing.coupling import uncoupled_lengths

__all__ = [
    "ROUTER_ID",
    "ROUTER_VERSION",
    "ROUTER_RULES_VERSION",
    "LAYERS",
    "BLOCKED",
    "CONVEX_PAD_SHAPES",
    "FINE_PITCH_MM",
    "FINE_RULES",
    "NetRule",
    "RoutingParams",
    "Routing",
    "effective_params",
    "finest_pad_pitch",
    "route_board",
]

#: provenance ``tool`` stamped on every track and via
ROUTER_ID = "routing.maze"
#: ``tool_version`` of copper routed without net rules: routing.maze 0.3 routes such a board exactly as 0.2 did (byte for
#: byte), so the stamp - part of every track's provenance, hence of the design hash - stays
ROUTER_VERSION = "0.2"
#: ``tool_version`` of every track and via of a board routed with at least one :class:`NetRule`
ROUTER_RULES_VERSION = "0.3"
#: the only copper layers this router routes (index 0 / 1 in the owner maps)
LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")
#: inner copper layers a board may list: accepted, never routed (planes / zones live there)
INNER_LAYER_RE = re.compile(r"^In[1-9][0-9]*\.Cu$")
#: owner-map value of a cell no net may use
BLOCKED = -1
#: flat static-map value of a cell no pad or edge claims (the owner maps say ``None``)
_FREE = -2
#: net index of a coupled pair's virtual net (it owns no pad and never reads the owner map)
_VIRTUAL = -3

#: ``(track parameter, ir.pcb.manufacturing limit)`` pairs a fab minimum can raise
_FAB_MINIMUMS: tuple[tuple[str, str], ...] = (
    ("track_width_mm", "min_track_width_mm"),
    ("clearance_mm", "min_clearance_mm"),
    ("via_drill_mm", "min_via_drill_mm"),
    ("via_diameter_mm", "min_via_diameter_mm"),
)
#: pad shapes whose copper lies inside the ``(size)`` box (convex): the only ones this router models
CONVEX_PAD_SHAPES = frozenset({"circle", "rect", "oval", "roundrect"})
#: direction index of a state that has no direction yet (a seed or the far end of a via)
_NO_DIR = 4
_STATES_PER_CELL = 5
_EPS = 1e-9
_INF = math.inf
#: exact-geometry tolerance (mm): coordinates are rounded to KiCad's 1 nm, so a distance short by less than this is rounding
_TOL = 1e-6
#: the search's direction indices as unit vectors (board frame, Y down): east, south, west, north
_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
_DIR_NAMES = ("east", "south", "west", "north")


def _left(d: int) -> tuple[int, int]:
    """The unit normal on the left of travel direction ``d`` (Y down: heading east, left is north)."""
    ux, uy = _DIRS[d]
    return (uy, -ux)


@dataclass(frozen=True, slots=True)
class RoutingParams:
    """The router's knobs (mm; the costs in grid steps; the iteration cap a count). Every one lands in the provenance.

    Geometry: ``grid_mm``, ``track_width_mm``, ``clearance_mm``,
    ``via_diameter_mm`` / ``via_drill_mm``, ``edge_clearance_mm``. Search
    costs: ``base_cost`` (one grid step on a free cell), ``via_cost``,
    ``bend_cost``. Negotiation: ``history_cost`` (added to an over-used
    cell's history after each iteration), ``present_cost`` (the present-
    congestion factor of iteration 1) and ``present_growth`` (its factor per
    iteration), ``max_iterations``; ``window_mm`` is the search window's
    margin around a net's terminals (doubled on failure) and how far a
    coupled pair's launch cell is searched from its pads.

    ``rules`` / ``pad_pitch_mm`` / ``pitch_footprint`` say why the values
    were chosen when :meth:`for_board` picked the fine rules (``None`` for
    values given directly or the defaults); they change nothing in the search
    and are appended to :meth:`derived_from_entry` only when set.

    The net-rule knobs (0.3; used only for nets a :class:`NetRule` names, and
    recorded in the provenance only when a board has rules):
    ``neckdown_radius_mm`` (the neck-down zone around a pad when the rule
    gives none), ``meander_amplitude_mm`` / ``meander_pitch_mm`` (the largest
    bump height and the leg spacing of a length-matching meander, rounded to
    whole grid steps) and ``pair_bump_amplitude_mm`` / ``pair_bump_length_mm``
    (the largest height and the length of one intra-pair compensation bump).
    """

    grid_mm: float = 0.25
    track_width_mm: float = 0.4
    clearance_mm: float = 0.25
    via_diameter_mm: float = 0.8
    via_drill_mm: float = 0.4
    edge_clearance_mm: float = 0.3
    via_cost: float = 12.0
    bend_cost: float = 0.6
    base_cost: float = 1.0
    history_cost: float = 1.0
    present_cost: float = 0.5
    present_growth: float = 2.0
    max_iterations: int = 40
    window_mm: float = 10.0
    rules: str | None = None
    pad_pitch_mm: float | None = None
    pitch_footprint: str | None = None
    neckdown_radius_mm: float = 1.0
    meander_amplitude_mm: float = 1.0
    meander_pitch_mm: float = 1.0
    pair_bump_amplitude_mm: float = 0.25
    pair_bump_length_mm: float = 0.5

    def check(self) -> None:
        """Refuse parameters that make no sense (:class:`CompileError`)."""
        for name in (
            "grid_mm", "track_width_mm", "via_drill_mm", "via_diameter_mm", "base_cost", "window_mm",
            "neckdown_radius_mm", "meander_amplitude_mm", "meander_pitch_mm", "pair_bump_amplitude_mm", "pair_bump_length_mm",
        ):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise CompileError(f"routing parameter {name} must be > 0 (got {value!r})")
        for name in ("clearance_mm", "edge_clearance_mm", "via_cost", "bend_cost", "history_cost", "present_cost"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise CompileError(f"routing parameter {name} must be >= 0 (got {value!r})")
        if not (isinstance(self.present_growth, (int, float)) and math.isfinite(self.present_growth) and self.present_growth >= 1):
            raise CompileError(f"routing parameter present_growth must be >= 1 (got {self.present_growth!r})")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise CompileError(f"routing parameter max_iterations must be an integer >= 1 (got {self.max_iterations!r})")
        if self.via_diameter_mm <= self.via_drill_mm:
            raise CompileError(f"via diameter {self.via_diameter_mm} mm must exceed the via drill {self.via_drill_mm} mm")
        if self.pad_pitch_mm is not None and not (math.isfinite(self.pad_pitch_mm) and self.pad_pitch_mm > 0):
            raise CompileError(f"routing parameter pad_pitch_mm must be a finite number > 0 (got {self.pad_pitch_mm!r})")

    def derived_from_entry(self, rules: bool = False) -> str:
        """The ``derived_from`` entry that records every parameter (and, when set, the rule set and the pitch that chose it).

        ``rules=True`` (a board routed with net rules) appends the 0.3 knobs;
        without rules the entry is 0.2's, character for character.
        """
        entry = (
            f"params:grid={self.grid_mm},width={self.track_width_mm},clearance={self.clearance_mm},"
            f"via={self.via_diameter_mm}/{self.via_drill_mm},edge={self.edge_clearance_mm},"
            f"via_cost={self.via_cost},bend_cost={self.bend_cost},base_cost={self.base_cost},"
            f"history_cost={self.history_cost},present_cost={self.present_cost},present_growth={self.present_growth},"
            f"max_iterations={self.max_iterations},window={self.window_mm}"
        )
        if self.rules is not None:
            entry += f",rules={self.rules},pad_pitch={self.pad_pitch_mm},pitch_footprint={self.pitch_footprint}"
        if rules:
            entry += (
                f",neckdown_radius={self.neckdown_radius_mm},meander_amplitude={self.meander_amplitude_mm},"
                f"meander_pitch={self.meander_pitch_mm},pair_bump_amplitude={self.pair_bump_amplitude_mm},"
                f"pair_bump_length={self.pair_bump_length_mm}"
            )
        return entry

    @classmethod
    def for_board(cls, ir: CircuitIR, library: KicadLibrary) -> RoutingParams:
        """The fine rules (:data:`FINE_RULES`) when the finest pad pitch on the board is below :data:`FINE_PITCH_MM`, else the defaults.

        The pitch is :func:`finest_pad_pitch` (library pad positions). The
        default instance is returned as ``cls()`` - no ``rules`` recorded.
        Fab minimums in ``ir.pcb.manufacturing`` still raise either set when
        the router runs (:func:`effective_params`).
        """
        pitch = finest_pad_pitch(ir, library)
        if pitch is None or pitch[0] >= FINE_PITCH_MM:
            return cls()
        return cls(**FINE_RULES, rules="fine", pad_pitch_mm=pitch[0], pitch_footprint=pitch[1])


#: a board whose finest centre-to-centre pad pitch is below this (mm) is routed with :data:`FINE_RULES`
FINE_PITCH_MM = 1.0
#: the fine rules: a 0.2 mm grid puts a terminal cell inside the inscribed circle of a 0.45 mm wide 0.8 mm-pitch pad
#: (worst case 0.2 * sqrt(2) / 2 = 0.141 mm from its centre < 0.225 mm) and a 0.25 mm track at 0.2 mm clearance
#: leaves such a pad along its axis; the via shrinks with it
FINE_RULES: dict[str, float] = {
    "grid_mm": 0.2,
    "track_width_mm": 0.25,
    "clearance_mm": 0.2,
    "via_diameter_mm": 0.6,
    "via_drill_mm": 0.3,
    "edge_clearance_mm": 0.3,
}


@dataclass(frozen=True, slots=True)
class NetRule:
    """One net's routing rule (mm), supplied by the caller; ``None`` = the board's value / not constrained.

    * ``net_class``: the name the caller mapped the rule from (recorded only).
    * ``width_mm`` / ``clearance_mm``: the net's track width and its clearance
      to every other net's copper (the larger of two nets' clearances applies).
    * ``neckdown_width_mm`` (below ``width_mm``) / ``neckdown_radius_mm``: the
      narrower width near the net's pads the full width cannot reach (module
      docstring), and how far from those pads' boxes it applies.
    * ``max_length_mm``: hard bound on the routed length, tracks plus
      ``via_length_mm`` per via (the caller converts a delay to a length and
      gives the via barrel; ``None`` counts a via as no length).
    * ``match_group`` / ``max_skew_mm`` (together): nets of one group are
      length-matched to within ``max_skew_mm`` by meanders (the members must
      agree on it).
    * ``pair_partner`` / ``pair_spacing_mm`` / ``pair_uncoupled_max_mm`` /
      ``pair_max_skew_mm``: a coupled differential pair with the named net
      (both rules name each other and agree on width, clearance, spacing and
      budgets); ``pair_spacing_mm`` is the edge-to-edge gap ``s`` (at least
      the board clearance), ``pair_uncoupled_max_mm`` bounds each net's
      uncoupled length (its copper outside the coupled overlap: the breakouts
      at both ends, the mitred corners, the compensation bumps -
      :mod:`ai_eda.tools.routing.coupling`, the definition ``si.diff``
      judges), ``pair_max_skew_mm`` the length difference left
      uncompensated.
    """

    net_class: str | None = None
    width_mm: float | None = None
    clearance_mm: float | None = None
    neckdown_width_mm: float | None = None
    neckdown_radius_mm: float | None = None
    max_length_mm: float | None = None
    via_length_mm: float | None = None
    match_group: str | None = None
    max_skew_mm: float | None = None
    pair_partner: str | None = None
    pair_spacing_mm: float | None = None
    pair_uncoupled_max_mm: float | None = None
    pair_max_skew_mm: float | None = None

    def is_empty(self) -> bool:
        """Whether the rule constrains nothing (every field but ``net_class`` is ``None``): such a rule is no rule."""
        return all(getattr(self, f.name) is None for f in fields(self) if f.name != "net_class")

    def derived_from_entry(self) -> str:
        """``rule:class=...,width=...,...``: every field of the (effective) rule, recorded in each track's provenance."""
        parts = []
        for f in fields(self):
            key = "class" if f.name == "net_class" else f.name.removesuffix("_mm")
            parts.append(f"{key}={getattr(self, f.name)}")
        return "rule:" + ",".join(parts)


#: rule fields that must be finite and > 0 / >= 0 when given, and the text fields
_RULE_POSITIVE = ("width_mm", "neckdown_width_mm", "neckdown_radius_mm", "max_length_mm", "pair_spacing_mm")
_RULE_NONNEGATIVE = ("clearance_mm", "via_length_mm", "max_skew_mm", "pair_uncoupled_max_mm", "pair_max_skew_mm")
_RULE_TEXT = ("net_class", "match_group", "pair_partner")
#: what the two nets of a coupled pair must agree on
_PAIR_SHARED = ("width_mm", "clearance_mm", "pair_spacing_mm", "pair_uncoupled_max_mm", "pair_max_skew_mm", "via_length_mm", "max_length_mm")


def _copper(pad: Pad) -> bool:
    return pad.pad_type != "np_thru_hole" and any(layer.endswith(".Cu") for layer in pad.layers)


def finest_pad_pitch(ir: CircuitIR, library: KicadLibrary) -> tuple[float, str] | None:
    """``(pitch, footprint lib id)``: the smallest centre-to-centre distance between two copper pads of one footprint on the board.

    Only pads that can carry different nets count: two pads with the same
    non-empty number are one logical pad (a split thermal pad, a switch's
    duplicated pins) and are skipped, as are pads without copper (paste-only
    apertures, NPTH holes) and coincident centres. Distances are taken in the
    footprint's own frame (a rotation does not change them) and rounded to
    KiCad's resolution; ties go to the first footprint in natural ref order.
    Components without a footprint, or whose footprint is not on disk, are
    not measured here: :func:`route_board` refuses such a board anyway, so no
    copper is ever routed at rules chosen without them. ``None`` when no
    footprint has two such pads.
    """
    best: tuple[float, str] | None = None
    seen: dict[str, float | None] = {}
    for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        if comp.footprint is None or not library.resolve_footprint(comp.footprint).verified:
            continue
        fp = library.load_footprint(comp.footprint)
        if fp.lib_id not in seen:
            pads = [p for p in fp.pads if _copper(p)]
            pitch: float | None = None
            for i, a in enumerate(pads):
                for b in pads[i + 1:]:
                    if a.number and a.number == b.number:
                        continue
                    d = _q(math.hypot(a.x - b.x, a.y - b.y))
                    if d > 0 and (pitch is None or d < pitch):
                        pitch = d
            seen[fp.lib_id] = pitch
        pitch = seen[fp.lib_id]
        if pitch is not None and (best is None or pitch < best[0]):
            best = (pitch, fp.lib_id)
    return best


@dataclass(slots=True)
class Routing:
    """What :func:`route_board` produced. ``unrouted`` maps a net name to why it has no copper.

    ``version`` is the ``tool_version`` stamped on the copper (:data:`ROUTER_VERSION`
    without net rules, :data:`ROUTER_RULES_VERSION` with), ``rules`` the
    effective rules (after the fab / board raises and the defaults).
    """

    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    unrouted: dict[str, str] = field(default_factory=dict)
    params: RoutingParams = field(default_factory=RoutingParams)
    #: routed_nets, unrouted_nets, skipped_nets (< 2 pads), net_length_mm, total_length_mm, track_count, via_count,
    #: grid (nx, ny, cells per layer), raised (parameter -> [requested, effective]), iterations, legal (the negotiation
    #: ended without an over-used cell), history (per iteration: rerouted nets, over-used cells, conflicting nets),
    #: dropped (nets ripped up after the cap), recovered (dropped nets routed again against the legal copper), net_order;
    #: inner_layers when the board has inner copper layers; with net rules also rules / match_groups / pairs
    stats: dict[str, Any] = field(default_factory=dict)
    version: str = ROUTER_VERSION
    rules: dict[str, NetRule] = field(default_factory=dict)


def effective_params(ir: CircuitIR, params: RoutingParams | None = None) -> tuple[RoutingParams, dict[str, tuple[float, float]]]:
    """``params`` (default :class:`RoutingParams`) with each value raised to the IR's fab minimum when one exists.

    Returns the effective parameters and ``{parameter: (requested, effective)}``
    for the ones that were raised. ``min_track_width_mm`` / ``min_clearance_mm``
    / ``min_via_drill_mm`` / ``min_via_diameter_mm`` map to their parameter;
    ``min_hole_to_edge_mm`` raises ``edge_clearance_mm`` so that a via's hole
    edge (``via position - drill/2``) keeps that distance from the outline.
    Any provenance counts: a limit is a limit, and whether it is grounded is
    the capability check's question. A raise that leaves the via drill at or
    above its diameter is refused (:class:`CompileError`) rather than guessed
    around.
    """
    base = params if params is not None else RoutingParams()
    base.check()
    mfg = ir.pcb.manufacturing if ir.pcb is not None else None
    raised: dict[str, tuple[float, float]] = {}
    values: dict[str, float] = {}
    if mfg is not None:
        for name, limit_name in _FAB_MINIMUMS:
            limit = getattr(mfg, limit_name)
            requested = float(getattr(base, name))
            if limit is not None and float(limit.value) > requested:
                values[name] = float(limit.value)
                raised[name] = (requested, float(limit.value))
        drill = values.get("via_drill_mm", base.via_drill_mm)
        diameter = values.get("via_diameter_mm", base.via_diameter_mm)
        if mfg.min_hole_to_edge_mm is not None:
            needed = float(mfg.min_hole_to_edge_mm.value) - (diameter - drill) / 2.0
            if needed > base.edge_clearance_mm:
                values["edge_clearance_mm"] = needed
                raised["edge_clearance_mm"] = (float(base.edge_clearance_mm), needed)
    eff = replace(base, **values)
    if eff.via_diameter_mm <= eff.via_drill_mm:
        raise CompileError(
            f"ir.pcb.manufacturing raises the via drill to {eff.via_drill_mm} mm, which is not below the via diameter "
            f"{eff.via_diameter_mm} mm; set min_via_diameter_mm or a larger via_diameter_mm instead of guessing an annular ring"
        )
    return eff, raised


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _effective_rules(
    ir: CircuitIR, p: RoutingParams, rules: Mapping[str, NetRule] | None,
) -> tuple[dict[str, NetRule], dict[str, dict[str, list[float | None]]]]:
    """The non-empty rules with every number filled in, and ``{net: {field: [given, effective]}}`` for the raised ones.

    Refuses (:class:`CompileError`) a rule for a net the IR lacks, a number
    that is not finite or out of range, a neck-down not below the width,
    ``match_group`` without ``max_skew_mm`` (or the reverse) or a group whose
    members disagree on it, pair fields without a partner, a partner that
    does not name the net back or disagrees on the shared values, a pair
    spacing below the board's clearance (raising it would change the pair's
    impedance: the caller re-solves), and a pair with a neck-down or in a
    match group (not implemented in 0.3). Raises (recorded): a width or
    neck-down width below ``min_track_width_mm``, a clearance below the
    board's effective clearance. A neck-down that the raise brings to the
    width is dropped (recorded with ``None``).
    """
    if rules is None:
        return {}, {}
    if not isinstance(rules, Mapping):
        raise CompileError(f"net rules must be a mapping of net name to NetRule (got {type(rules).__name__})")
    names = {net.name for net in ir.nets}
    given: dict[str, NetRule] = {}
    for name in sorted(rules, key=str):
        rule = rules[name]
        if not isinstance(name, str) or not isinstance(rule, NetRule):
            raise CompileError(f"net rule {name!r}: the key must be a net name and the value a NetRule (got {type(rule).__name__})")
        if name not in names:
            raise CompileError(f"net rule for {name!r}: the IR has no such net")
        for attr in _RULE_POSITIVE:
            value = getattr(rule, attr)
            if value is not None and not (_finite(value) and value > 0):
                raise CompileError(f"net rule for {name!r}: {attr} must be a finite number > 0 (got {value!r})")
        for attr in _RULE_NONNEGATIVE:
            value = getattr(rule, attr)
            if value is not None and not (_finite(value) and value >= 0):
                raise CompileError(f"net rule for {name!r}: {attr} must be a finite number >= 0 (got {value!r})")
        for attr in _RULE_TEXT:
            value = getattr(rule, attr)
            if value is not None and not (isinstance(value, str) and value):
                raise CompileError(f"net rule for {name!r}: {attr} must be a non-empty name (got {value!r})")
        if not rule.is_empty():
            given[name] = rule
    mfg = ir.pcb.manufacturing if ir.pcb is not None else None
    min_width = float(mfg.min_track_width_mm.value) if mfg is not None and mfg.min_track_width_mm is not None else None
    eff: dict[str, NetRule] = {}
    raised: dict[str, dict[str, list[float | None]]] = {}
    groups: dict[str, set[float]] = {}
    for name, rule in given.items():
        if rule.neckdown_width_mm is not None:
            if rule.width_mm is None:
                raise CompileError(f"net rule for {name!r}: neckdown_width_mm needs width_mm (the width it narrows from)")
            if rule.neckdown_width_mm >= rule.width_mm:
                raise CompileError(f"net rule for {name!r}: neckdown_width_mm {rule.neckdown_width_mm} must be below width_mm {rule.width_mm}")
        if rule.neckdown_radius_mm is not None and rule.neckdown_width_mm is None:
            raise CompileError(f"net rule for {name!r}: neckdown_radius_mm without neckdown_width_mm")
        if (rule.match_group is None) != (rule.max_skew_mm is None):
            raise CompileError(f"net rule for {name!r}: match_group and max_skew_mm go together (got {rule.match_group!r} / {rule.max_skew_mm!r})")
        if rule.match_group is not None:
            groups.setdefault(rule.match_group, set()).add(float(rule.max_skew_mm))
        pair_fields = ("pair_spacing_mm", "pair_uncoupled_max_mm", "pair_max_skew_mm")
        if rule.pair_partner is None:
            extra = [attr for attr in pair_fields if getattr(rule, attr) is not None]
            if extra:
                raise CompileError(f"net rule for {name!r}: {extra} without pair_partner")
        else:
            partner = rule.pair_partner
            other = given.get(partner)
            if partner == name or other is None or other.pair_partner != name:
                raise CompileError(f"net rule for {name!r}: pair_partner {partner!r} must be another net whose rule names {name!r} back")
            if rule.width_mm is None or rule.pair_spacing_mm is None:
                raise CompileError(f"net rule for {name!r}: a coupled pair needs width_mm and pair_spacing_mm")
            for attr in _PAIR_SHARED:
                if getattr(rule, attr) != getattr(other, attr):
                    raise CompileError(f"net rule for {name!r}: {attr} {getattr(rule, attr)!r} differs from its partner {partner!r} ({getattr(other, attr)!r})")
            if rule.neckdown_width_mm is not None:
                raise CompileError(f"net rule for {name!r}: a neck-down of a coupled pair's breakout is not implemented in routing.maze {ROUTER_RULES_VERSION}")
            if rule.match_group is not None:
                raise CompileError(f"net rule for {name!r}: a coupled pair in a match group is not implemented in routing.maze {ROUTER_RULES_VERSION}")
            if rule.pair_spacing_mm < p.clearance_mm - _EPS:
                raise CompileError(
                    f"net rule for {name!r}: pair_spacing_mm {rule.pair_spacing_mm} is below the board clearance {p.clearance_mm} mm; "
                    "raising it would change the pair's impedance - solve the width and spacing for a legal gap instead"
                )
        up: dict[str, list[float | None]] = {}
        width = p.track_width_mm if rule.width_mm is None else float(rule.width_mm)
        if rule.width_mm is not None and min_width is not None and width < min_width:
            up["width_mm"] = [width, min_width]
            width = min_width
        clearance = p.clearance_mm if rule.clearance_mm is None else float(rule.clearance_mm)
        if clearance < p.clearance_mm:
            up["clearance_mm"] = [clearance, p.clearance_mm]
            clearance = p.clearance_mm
        neck = None if rule.neckdown_width_mm is None else float(rule.neckdown_width_mm)
        if neck is not None and min_width is not None and neck < min_width:
            up["neckdown_width_mm"] = [neck, min_width]
            neck = min_width
        if neck is not None and neck >= width:
            up["neckdown_width_mm"] = [float(rule.neckdown_width_mm), None]  # the raise left nothing to narrow
            neck = None
        radius = None
        if neck is not None:
            radius = p.neckdown_radius_mm if rule.neckdown_radius_mm is None else float(rule.neckdown_radius_mm)
        eff[name] = replace(
            rule, width_mm=width, clearance_mm=clearance, neckdown_width_mm=neck, neckdown_radius_mm=radius,
            via_length_mm=0.0 if rule.via_length_mm is None else float(rule.via_length_mm),
        )
        if up:
            raised[name] = up
    for group, skews in sorted(groups.items()):
        if len(skews) != 1:
            raise CompileError(f"match group {group!r}: its nets give different max_skew_mm {sorted(skews)}")
    return eff, raised


# --------------------------------------------------------------------------- exact geometry (the audits)


def _pt_seg(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from the point to the segment a-b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _segs_cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    """Whether the closed segments a-b and c-d share a point (collinear overlaps included)."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True

    def on(p, q, r):  # r on segment p-q, given collinear
        return min(p[0], q[0]) - _EPS <= r[0] <= max(p[0], q[0]) + _EPS and min(p[1], q[1]) - _EPS <= r[1] <= max(p[1], q[1]) + _EPS

    return (o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b))


def _seg_seg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> float:
    """Distance between the segments a-b and c-d (0 when they touch or cross)."""
    if _segs_cross(a, b, c, d):
        return 0.0
    return min(_pt_seg(a[0], a[1], c, d), _pt_seg(b[0], b[1], c, d), _pt_seg(c[0], c[1], a, b), _pt_seg(d[0], d[1], a, b))


def _pt_box(px: float, py: float, box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - px, 0.0, px - x2), max(y1 - py, 0.0, py - y2))


def _seg_box(a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    """Distance from the segment a-b to the axis-aligned box (0 when it touches or enters it); both convex, so a vertex of one is nearest."""
    x1, y1, x2, y2 = box
    if _pt_box(a[0], a[1], box) == 0.0 or _pt_box(b[0], b[1], box) == 0.0:
        return 0.0
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    for k in range(4):
        if _segs_cross(a, b, corners[k], corners[(k + 1) % 4]):
            return 0.0
    return min(_pt_box(a[0], a[1], box), _pt_box(b[0], b[1], box), *(_pt_seg(cx, cy, a, b) for cx, cy in corners))


# --------------------------------------------------------------------------- geometry helpers


@dataclass(frozen=True, slots=True)
class _PadGeom:
    """One placed pad in the board frame: centre, half extents of its obstacle box, copper layers, net index."""

    ref: str
    number: str
    cx: float
    cy: float
    hw: float
    hh: float
    inscribed_r: float
    layers: tuple[int, ...]  # indices into LAYERS
    net: int  # net index or BLOCKED

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.hw, self.cy - self.hh, self.cx + self.hw, self.cy + self.hh)


@dataclass(frozen=True, slots=True)
class _Terminal:
    pad: _PadGeom
    cell: int  # cell index k = j * nx + i
    label: str


def _pad_copper_layers(placement, pad: Pad) -> tuple[int, ...]:
    """Indices of the copper layers the pad is on: through-hole and ``*.Cu`` pads on both, SMD pads where they say."""
    layers = pad_layers(placement, pad)
    if pad.pad_type in ("thru_hole", "np_thru_hole") or "*.Cu" in layers:
        return (0, 1)
    return tuple(k for k, name in enumerate(LAYERS) if name in layers)


def _pad_box(placement, pad: Pad) -> tuple[float, float, float, float, float]:
    """``(cx, cy, hw, hh, inscribed radius)``; exact for multiples of 90 degrees, the circumscribed square otherwise."""
    cx, cy = pad_center(placement, pad)
    angle = pad_angle(placement, pad)
    if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):
        w, h = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
        hw, hh = w / 2.0, h / 2.0
    else:
        hw = hh = math.hypot(pad.size_w, pad.size_h) / 2.0
    return cx, cy, hw, hh, min(pad.size_w, pad.size_h) / 2.0


def _disc(radius: float, grid: float) -> list[tuple[int, int]]:
    """Grid offsets ``(a, b)`` with ``hypot(a, b) * grid < radius`` (the cell itself included)."""
    k = int(radius / grid) + 1
    return [(a, b) for a in range(-k, k + 1) for b in range(-k, k + 1) if math.hypot(a * grid, b * grid) < radius - _EPS]


class _Halo:
    """A disc of grid offsets as flat index offsets, with its reach for the in-bounds fast path."""

    __slots__ = ("offsets", "flat", "reach")

    def __init__(self, offsets: list[tuple[int, int]], nx: int) -> None:
        self.offsets = offsets
        self.flat = [a + b * nx for a, b in offsets]
        self.reach = max((max(abs(a), abs(b)) for a, b in offsets), default=0)


# --------------------------------------------------------------------------- the board model


class _Board:
    """The grid, the static owner maps and the terminals of one IR (built once per :func:`route_board`).

    ``inner_layers`` accepts inner ``In<k>.Cu`` layers in ``ir.pcb.layers``
    (never routed); without it they are refused, as 0.2 refused them.

    ``pad_clearance`` (net index -> clearance, only for nets whose rule
    clearance exceeds ``p.clearance_mm``) grows the owner map around those
    nets' pads and keeps every other net's via ``via_diameter/2 + c`` from
    them (:attr:`via_keep`); without it the maps are 0.2's.
    """

    def __init__(
        self, ir: CircuitIR, library: KicadLibrary, p: RoutingParams, pad_clearance: Mapping[int, float] | None = None, inner_layers: bool = False,
    ) -> None:
        if ir.pcb is None:
            raise CompileError("cannot route: ir.pcb is None")
        if ir.pcb.outline is None:
            raise CompileError("cannot route: ir.pcb.outline is None; the grid needs a board outline")
        names = [layer.name for layer in ir.pcb.layers]
        extra = [n for n in names if n not in LAYERS]
        if extra and not inner_layers:
            raise CompileError(f"cannot route: this router knows only {list(LAYERS)}, ir.pcb.layers also has {extra}")
        extra = [n for n in extra if not INNER_LAYER_RE.match(n)]
        if extra:
            raise CompileError(
                f"cannot route: this router knows only {list(LAYERS)} (and inner In<k>.Cu layers, which it never routes), "
                f"ir.pcb.layers also has {extra}"
            )
        missing = [n for n in LAYERS if n not in names]
        if missing:
            raise CompileError(f"cannot route: ir.pcb.layers lacks {missing}")
        #: the inner copper layers the board lists (accepted, never routed)
        self.inner_layers = [n for n in names if INNER_LAYER_RE.match(n)]
        self.p = p
        self.pad_clear: dict[int, float] = {k: float(v) for k, v in (pad_clearance or {}).items() if float(v) > p.clearance_mm}
        o = ir.pcb.outline
        if not (o.width_mm > 0 and o.height_mm > 0):
            raise CompileError(f"cannot route: outline {o.width_mm} x {o.height_mm} mm has no area")
        self.ox, self.oy, self.w, self.h = float(o.origin_x_mm), float(o.origin_y_mm), float(o.width_mm), float(o.height_mm)
        g = p.grid_mm
        self.nx = int(self.w / g + _EPS) + 1
        self.ny = int(self.h / g + _EPS) + 1
        self.n = self.nx * self.ny
        # static owner maps (pads and the board edge): one flat list per layer, None = free
        self.owner: list[list[int | None]] = [[None] * self.n for _ in LAYERS]
        self.via_edge_ok: list[bool] = [False] * self.n
        #: False where a via's copper disc would overlap a pad box (any net, any layer): no via-in-pad
        self.via_pad_ok: list[bool] = [True] * self.n
        edge_track = p.edge_clearance_mm + p.track_width_mm / 2.0
        edge_via = p.edge_clearance_mm + p.via_diameter_mm / 2.0
        for k in range(self.n):
            d = self._edge_distance(k)
            if d < edge_track - _EPS:
                self.owner[0][k] = BLOCKED
                self.owner[1][k] = BLOCKED
            self.via_edge_ok[k] = d >= edge_via - _EPS
        self.pad_radius = p.clearance_mm + p.track_width_mm / 2.0 + g / 2.0
        #: track-to-track: a foreign track centre closer than this to a path cell breaks the clearance (0.1's path mark)
        self.track_halo = _Halo(_disc(p.track_width_mm + p.clearance_mm + g / 2.0, g), self.nx)
        #: track-to-via: a via centre closer than this to a track centre (and a via's static pad / edge check, 0.1's via mark)
        self.via_disc = _disc(p.via_diameter_mm / 2.0 + p.clearance_mm + p.track_width_mm / 2.0, g)
        self.via_track_halo = _Halo(self.via_disc, self.nx)
        #: via-to-via: centres at least via_diameter + clearance apart
        self.via_via_halo = _Halo(_disc(p.via_diameter_mm + p.clearance_mm, g), self.nx)
        #: halo discs by radius (the negotiation's per-profile radii; 0.2's three are among them)
        self._halos: dict[float, _Halo] = {}
        self.net_index = {net.name: k for k, net in enumerate(ir.nets)}
        self.terminals: dict[str, list[_Terminal]] = {net.name: [] for net in ir.nets}
        #: every placed pad (the rule fences and the exact audits read them)
        self.pads: list[_PadGeom] = []
        self._load_pads(ir, library)
        #: the static owner maps as one flat list over ``layer * n + k`` with :data:`_FREE` for free cells (the search reads this)
        self.stat: list[int] = [_FREE if v is None else v for v in self.owner[0]] + [_FREE if v is None else v for v in self.owner[1]]
        #: per cell, lazily: BLOCKED / _FREE / the one net whose pads the via's keep-out touches (see :meth:`via_static_at`)
        self.via_static: list[int | None] = [None] * self.n
        #: per cell: _FREE, BLOCKED or the one net whose larger-clearance pads a via of any other net would come too near
        self.via_keep: list[int] | None = None
        if self.pad_clear:
            self.via_keep = [_FREE] * self.n
            for geom in self.pads:
                c = self.pad_clear.get(geom.net)
                if c is None:
                    continue
                for k, dist in self.cells_near_box(geom, p.via_diameter_mm / 2.0 + c):
                    if dist < p.via_diameter_mm / 2.0 + c - _EPS:
                        cur = self.via_keep[k]
                        self.via_keep[k] = geom.net if cur in (_FREE, geom.net) else BLOCKED

    # --- coordinates ---------------------------------------------------------

    def pos(self, k: int) -> tuple[float, float]:
        j, i = divmod(k, self.nx)
        return _q(self.ox + i * self.p.grid_mm), _q(self.oy + j * self.p.grid_mm)

    def xy(self, k: int) -> tuple[float, float]:
        """The unrounded board position of cell ``k`` (the pair geometry offsets from it)."""
        j, i = divmod(k, self.nx)
        return self.ox + i * self.p.grid_mm, self.oy + j * self.p.grid_mm

    def _edge_distance(self, k: int) -> float:
        j, i = divmod(k, self.nx)
        x, y = self.ox + i * self.p.grid_mm, self.oy + j * self.p.grid_mm
        return min(x - self.ox, self.ox + self.w - x, y - self.oy, self.oy + self.h - y)

    def edge_distance_xy(self, x: float, y: float) -> float:
        return min(x - self.ox, self.ox + self.w - x, y - self.oy, self.oy + self.h - y)

    def _nearest_cell(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.ox) / self.p.grid_mm), round((y - self.oy) / self.p.grid_mm)

    def halo_for(self, radius: float) -> _Halo:
        """The disc of grid offsets closer than ``radius`` (memoised by radius)."""
        halo = self._halos.get(radius)
        if halo is None:
            halo = _Halo(_disc(radius, self.p.grid_mm), self.nx)
            self._halos[radius] = halo
        return halo

    def cells_near_box(self, geom: _PadGeom, r: float):
        """``(k, distance to the box)`` for every cell of the bounding square ``r`` around the pad box (callers compare the distance)."""
        i0, j0 = self._nearest_cell(geom.cx - geom.hw - r, geom.cy - geom.hh - r)
        i1, j1 = self._nearest_cell(geom.cx + geom.hw + r, geom.cy + geom.hh + r)
        g = self.p.grid_mm
        for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
            dy = max(abs(self.oy + j * g - geom.cy) - geom.hh, 0.0)
            for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                dx = max(abs(self.ox + i * g - geom.cx) - geom.hw, 0.0)
                yield j * self.nx + i, math.hypot(dx, dy)

    # --- pads ----------------------------------------------------------------

    def _load_pads(self, ir: CircuitIR, library: KicadLibrary) -> None:
        pin_net: dict[tuple[str, str], str] = {}
        for net in ir.nets:
            for pin in net.pins:
                comp = ir.component(pin.component_ref)
                if comp is None:
                    raise CompileError(f"net {net.name!r} references unknown component {pin.component_ref!r}")
                pin_net[(pin.component_ref, pin.pin_number)] = net.name
        placed = {pl.component_ref for pl in ir.pcb.placements}
        refs = {c.ref for c in ir.components}
        stray = sorted(placed - refs)
        if stray:
            raise CompileError(f"cannot route: placement(s) of unknown component(s) {stray}")
        for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
            placement = ir.pcb.placement(comp.ref)
            if placement is None:
                raise CompileError(f"cannot route: component {comp.ref!r} has no placement")
            if comp.footprint is None:
                raise CompileError(f"cannot route: component {comp.ref!r} has no footprint")
            resolved = library.resolve_footprint(comp.footprint)
            if not resolved.verified:
                raise CompileError(
                    f"cannot route: footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} "
                    f"was not found in a KiCad library"
                )
            fp = library.load_footprint(comp.footprint)
            self._check_net_pads(ir, comp.ref, fp)
            for pad in fp.pads:
                if pad.shape not in CONVEX_PAD_SHAPES:
                    raise CompileError(
                        f"cannot route: pad {comp.ref}.{pad.number or '(unnumbered)'} of footprint {fp.lib_id} has shape {pad.shape!r}; "
                        f"its copper is not bounded by its (size) box (custom primitives / trapezoid rect_delta are not read), "
                        f"so this router models only {sorted(CONVEX_PAD_SHAPES)}"
                    )
                net_name = pin_net.get((comp.ref, pad.number)) if pad.number else None
                cx, cy, hw, hh, r_in = _pad_box(placement, pad)
                layers = _pad_copper_layers(placement, pad)
                geom = _PadGeom(comp.ref, pad.number, cx, cy, hw, hh, r_in, layers, self.net_index[net_name] if net_name else BLOCKED)
                self.pads.append(geom)
                for layer in layers:
                    self._mark_box(layer, geom)
                self._forbid_vias_in(geom)
                if net_name is None:
                    continue
                if not layers:
                    raise CompileError(f"cannot route net {net_name!r}: pad {comp.ref}.{pad.number} is on no copper layer ({pad.layers})")
                self.terminals[net_name].append(self._terminal(geom))
        for terms in self.terminals.values():
            terms.sort(key=lambda t: (natural_ref_key(t.pad.ref), natural_ref_key(t.pad.number)))

    @staticmethod
    def _check_net_pads(ir: CircuitIR, ref: str, fp: FootprintDef) -> None:
        for net in ir.nets:
            for pin in net.pins:
                if pin.component_ref == ref and fp.pad(pin.pin_number) is None:
                    raise CompileError(f"net {net.name!r} references {ref}.{pin.pin_number} but footprint {fp.lib_id} has no pad {pin.pin_number!r}")

    def _terminal(self, geom: _PadGeom) -> _Terminal:
        i, j = self._nearest_cell(geom.cx, geom.cy)
        label = f"{geom.ref}.{geom.number}"
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            inside = self.ox <= geom.cx <= self.ox + self.w and self.oy <= geom.cy <= self.oy + self.h
            if not inside:
                raise CompileError(f"cannot route: pad {label} at ({geom.cx}, {geom.cy}) lies outside the board outline")
            raise CompileError(
                f"cannot route: pad {label} at ({geom.cx}, {geom.cy}) is inside the {self.w:g} x {self.h:g} mm outline but nearer than half a "
                f"{self.p.grid_mm} mm grid step to its far edge: no grid cell exists there (and it is inside the edge keep-out anyway)"
            )
        x, y = self.pos(j * self.nx + i)
        d = math.hypot(x - geom.cx, y - geom.cy)
        if d > geom.inscribed_r - 1e-6:
            raise CompileError(
                f"pad {label} ({2 * geom.hw:g} x {2 * geom.hh:g} mm) is too small for the {self.p.grid_mm} mm routing grid: "
                f"the nearest grid point is {d:.4f} mm from its centre, outside its inscribed circle (r={geom.inscribed_r:g} mm)"
            )
        return _Terminal(pad=geom, cell=j * self.nx + i, label=label)

    def _mark_box(self, layer: int, geom: _PadGeom) -> None:
        """Own every cell within the pad's keep-out radius of the box for ``geom.net`` (BLOCKED when another net has it).

        The radius is ``clearance + width/2 + grid/2`` (:attr:`pad_radius`),
        with the pad's net's own clearance when a rule makes it larger.
        """
        r = self.pad_radius
        c = self.pad_clear.get(geom.net) if self.pad_clear else None
        if c is not None:
            r = c + self.p.track_width_mm / 2.0 + self.p.grid_mm / 2.0
        i0, j0 = self._nearest_cell(geom.cx - geom.hw - r, geom.cy - geom.hh - r)
        i1, j1 = self._nearest_cell(geom.cx + geom.hw + r, geom.cy + geom.hh + r)
        owner = self.owner[layer]
        g = self.p.grid_mm
        for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
            y = self.oy + j * g
            dy = max(abs(y - geom.cy) - geom.hh, 0.0)
            for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                x = self.ox + i * g
                dx = max(abs(x - geom.cx) - geom.hw, 0.0)
                if math.hypot(dx, dy) < r - _EPS:
                    k = j * self.nx + i
                    cur = owner[k]
                    if cur is None:
                        owner[k] = geom.net
                    elif cur != geom.net:
                        owner[k] = BLOCKED

    def _forbid_vias_in(self, geom: _PadGeom) -> None:
        """Clear ``via_pad_ok`` where a via disc (radius ``via_diameter/2``) would overlap the pad box (any layer: a via spans both)."""
        r = self.p.via_diameter_mm / 2.0
        i0, j0 = self._nearest_cell(geom.cx - geom.hw - r, geom.cy - geom.hh - r)
        i1, j1 = self._nearest_cell(geom.cx + geom.hw + r, geom.cy + geom.hh + r)
        g = self.p.grid_mm
        for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
            dy = max(abs(self.oy + j * g - geom.cy) - geom.hh, 0.0)
            for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                dx = max(abs(self.ox + i * g - geom.cx) - geom.hw, 0.0)
                if math.hypot(dx, dy) < r - _EPS:
                    self.via_pad_ok[j * self.nx + i] = False

    def usable_layers(self, terminal: _Terminal, net: int) -> tuple[int, ...]:
        """The pad's copper layers on which the terminal cell is free or the net's own (never BLOCKED / another net's)."""
        return tuple(layer for layer in terminal.pad.layers if self.owner[layer][terminal.cell] in (None, net))

    def via_static_at(self, k: int) -> int:
        """:data:`BLOCKED`, :data:`_FREE` or the one net that may drill a via at cell ``k`` given the pads and the edge (memoised).

        BLOCKED when the via would be too near the edge, overlap any pad box,
        leave the grid, or when a cell within ``via_diameter/2 + clearance +
        width/2`` is BLOCKED or owned by two nets on either layer; the owning
        net when some of those cells belong to one net's pads; free otherwise.
        With larger rule clearances, :attr:`via_keep` narrows it further.
        """
        cached = self.via_static[k]
        if cached is not None:
            return cached
        value = self._via_static(k)
        if self.via_keep is not None and value != BLOCKED:
            keep = self.via_keep[k]
            if keep == BLOCKED or (keep != _FREE and value != _FREE and value != keep):
                value = BLOCKED
            elif keep != _FREE:
                value = keep
        self.via_static[k] = value
        return value

    def _via_static(self, k: int) -> int:
        if not self.via_edge_ok[k] or not self.via_pad_ok[k]:
            return BLOCKED
        j, i = divmod(k, self.nx)
        stat, n = self.stat, self.n
        owner = _FREE
        for a, b in self.via_disc:
            ii, jj = i + a, j + b
            if not (0 <= ii < self.nx and 0 <= jj < self.ny):
                return BLOCKED
            kk = jj * self.nx + ii
            for o in (stat[kk], stat[n + kk]):
                if o == _FREE:
                    continue
                if o == BLOCKED or (owner != _FREE and owner != o):
                    return BLOCKED
                owner = o
        return owner

    def via_allowed(self, k: int, net: int) -> bool:
        """Whether the static map (pads, edge) lets ``net`` drill a via at cell ``k`` (other nets' copper is negotiated, not checked here)."""
        v = self.via_static_at(k)
        return v == _FREE or v == net

    def halo_cells(self, halo: _Halo, k: int, base: int, out: set[int]) -> None:
        """Add ``base + cell`` for every cell of ``halo`` around cell ``k`` that lies on the grid."""
        j, i = divmod(k, self.nx)
        r = halo.reach
        if r <= i < self.nx - r and r <= j < self.ny - r:
            b = base + k
            out.update([b + f for f in halo.flat])
            return
        for a, bb in halo.offsets:
            ii, jj = i + a, j + bb
            if 0 <= ii < self.nx and 0 <= jj < self.ny:
                out.add(base + jj * self.nx + ii)

    # --- rule fences -----------------------------------------------------------

    def rule_blocked(self, k: int, layer: int, net: int, width: float, clearance: float, clear_of: Callable[[int], float]) -> bool:
        """Whether a track of ``net`` at cell ``k`` on ``layer`` with this width / clearance comes too near a foreign pad or the edge."""
        g = self.p.grid_mm
        if self._edge_distance(k) < self.p.edge_clearance_mm + width / 2.0 - _EPS:
            return True
        x, y = self.xy(k)
        for geom in self.pads:
            if geom.net == net or layer not in geom.layers:
                continue
            if _pt_box(x, y, geom.box) < max(clearance, clear_of(geom.net)) + width / 2.0 + g / 2.0 - _EPS:
                return True
        return False

    def rule_fence(
        self, net: int, width: float, clearance: float, zone: bytearray | None, neck: float | None, neck_clearance: float,
        clear_of: Callable[[int], float],
    ) -> bytearray:
        """``layer * n + k`` -> 1 where the net's track (neck-down width inside ``zone``) breaks its rule at a foreign pad or the edge."""
        n, g, e = self.n, self.p.grid_mm, self.p.edge_clearance_mm
        fence = bytearray(2 * n)
        for k in range(n):
            w = neck if zone is not None and zone[k] else width
            if self._edge_distance(k) < e + w / 2.0 - _EPS:
                fence[k] = 1
                fence[n + k] = 1
        for geom in self.pads:
            if geom.net == net:
                continue
            cy = clear_of(geom.net)
            r_main = max(clearance, cy) + width / 2.0 + g / 2.0
            r_neck = max(neck_clearance, cy) + neck / 2.0 + g / 2.0 if (zone is not None and neck is not None) else 0.0
            for k, dist in self.cells_near_box(geom, max(r_main, r_neck)):
                r = r_neck if zone is not None and zone[k] else r_main
                if dist < r - _EPS:
                    for layer in geom.layers:
                        fence[layer * n + k] = 1
        return fence

    def rule_via_fence(self, net: int, clearance: float, clear_of: Callable[[int], float]) -> bytearray:
        """``k`` -> 1 where a via of ``net`` would come closer than ``via_diameter/2 + max(c, c_pad)`` to a foreign pad box."""
        fence = bytearray(self.n)
        r0 = self.p.via_diameter_mm / 2.0
        for geom in self.pads:
            if geom.net == net:
                continue
            r = r0 + max(clearance, clear_of(geom.net))
            for k, dist in self.cells_near_box(geom, r):
                if dist < r - _EPS:
                    fence[k] = 1
        return fence


# --------------------------------------------------------------------------- one net's route

#: a profile key: (track width, clearance) of the copper a halo plane is counted for
_Key = tuple[float, float]


@dataclass(slots=True)
class _NetRoute:
    """The route of one net: the ordered steps (stubs and paths), its cells, and the halo it covers."""

    steps: list[tuple[str, Any, int]] = field(default_factory=list)  # ("stub", terminal, layer) | ("path", path, 0) | ("pair", data, 0)
    path_cells: set[int] = field(default_factory=set)  # layer * n + k
    via_cells: list[int] = field(default_factory=list)  # k, in path order
    cover_t: dict[_Key, set[int]] = field(default_factory=dict)  # per track profile: layer * n + k
    cover_v: dict[float, set[int]] = field(default_factory=dict)  # per via clearance: k
    how: str = "negotiated"  # "negotiated" or "recovered" (routed again against the legal copper after the cap)
    #: the route's path cells grouped by the profile of their copper, and the clearance of its vias
    keyed: list[tuple[_Key, set[int]]] = field(default_factory=list)
    via_key: float = 0.0
    #: covers that are no path cells (a coupled pair's breakouts), merged into cover_t / cover_v
    extra_t: dict[_Key, set[int]] = field(default_factory=dict)
    extra_v: dict[float, set[int]] = field(default_factory=dict)


@dataclass(slots=True)
class _RuleNet:
    """A net routed under a :class:`NetRule` (or a coupled pair's virtual net): its profiles, fences and budget."""

    name: str
    idx: int
    rule: NetRule | None
    key: _Key
    via_key: float
    nd_key: _Key | None = None
    fence: bytearray | None = None  # layer * n + k -> 1: the rule's own keep-out beyond the owner map
    zone: bytearray | None = None  # k -> 1: the neck-down zone
    vfence: bytearray | None = None  # k -> 1: no via of this net here
    nd_pads: list[str] = field(default_factory=list)
    budget: float | None = None
    via_len: float = 0.0
    use_stat: bool = True
    allow_vias: bool = True

    def key_at(self, k: int) -> _Key:
        return self.nd_key if self.zone is not None and self.zone[k] else self.key


class _Negotiation:
    """The present / history costs and the halo counts of every routed net (PathFinder state).

    One halo count / node cost array per track profile (``track_keys``, the
    board's own ``(width, clearance)`` first) and one via count / cost array
    per clearance (``via_keys``); the board profile's arrays are also
    ``halo`` / ``ncost`` / ``vhalo`` / ``vcost`` (what the 0.2 search reads).
    Without rules there is exactly one of each: 0.2's state.
    """

    def __init__(self, board: _Board, p: RoutingParams, track_keys: list[_Key] | None = None, via_keys: list[float] | None = None) -> None:
        self.board = board
        self.p = p
        n = board.n
        self.key0: _Key = (p.track_width_mm, p.clearance_mm)
        self.vkey0 = p.clearance_mm
        self.track_keys = list(track_keys) if track_keys else [self.key0]
        self.via_keys = list(via_keys) if via_keys else [self.vkey0]
        self.t_halo = {key: [0] * (2 * n) for key in self.track_keys}
        self.t_ncost = {key: [p.base_cost] * (2 * n) for key in self.track_keys}
        self.v_halo = {c: [0] * n for c in self.via_keys}
        self.v_cost = {c: [p.via_cost] * n for c in self.via_keys}
        self.hist = [0.0] * (2 * n)
        self.vhist = [0.0] * n
        self.pres = p.present_cost
        #: the search's per-state best cost and predecessor (``state = (layer * n + k) * 5 + direction``), reset after every search
        self.g_best = [_INF] * (2 * n * _STATES_PER_CELL)
        self.prev = [-1] * (2 * n * _STATES_PER_CELL)
        self._glen: list[float] | None = None
        self._alias()

    def _alias(self) -> None:
        self.halo = self.t_halo[self.key0]
        self.ncost = self.t_ncost[self.key0]
        self.vhalo = self.v_halo[self.vkey0]
        self.vcost = self.v_cost[self.vkey0]

    def lengths(self) -> list[float]:
        """The rule search's per-state path length (mm), allocated on first use."""
        if self._glen is None:
            self._glen = [0.0] * len(self.g_best)
        return self._glen

    # --- costs -----------------------------------------------------------------

    def recompute(self) -> None:
        """Every node / via cost from the current present factor, history and counts (after the factor or the history changed)."""
        base, pres = self.p.base_cost, self.pres
        for key in self.track_keys:
            self.t_ncost[key] = [base * (1.0 + h) * (1.0 + pres * o) for h, o in zip(self.hist, self.t_halo[key])]
        vb = self.p.via_cost + base
        for c in self.via_keys:
            self.v_cost[c] = [vb * (1.0 + h) * (1.0 + pres * o) - base for h, o in zip(self.vhist, self.v_halo[c])]
        self._alias()

    # --- occupancy -------------------------------------------------------------

    def _count(self, route: _NetRoute, delta: int) -> None:
        p = self.p
        base, pres, hist, vhist = p.base_cost, self.pres, self.hist, self.vhist
        for key, cells in route.cover_t.items():
            halo, ncost = self.t_halo[key], self.t_ncost[key]
            for c in cells:
                halo[c] += delta
                ncost[c] = base * (1.0 + hist[c]) * (1.0 + pres * halo[c])
        vb = p.via_cost + base
        for ck, cells in route.cover_v.items():
            vhalo, vcost = self.v_halo[ck], self.v_cost[ck]
            for k in cells:
                vhalo[k] += delta
                vcost[k] = vb * (1.0 + vhist[k]) * (1.0 + pres * vhalo[k]) - base

    def add(self, route: _NetRoute) -> None:
        self._count(route, 1)

    def remove(self, route: _NetRoute) -> None:
        self._count(route, -1)

    def overused(self, route: _NetRoute) -> tuple[list[int], list[int]]:
        """``(track cells, via cells)`` of ``route`` that lie inside another net's halo (its own halo always covers them)."""
        t_halo = self.t_halo
        track = [c for key, cells in route.keyed for c in cells if t_halo[key][c] > 1]
        vhalo = self.v_halo[route.via_key]
        return track, [k for k in route.via_cells if vhalo[k] > 1]

    def cover(self, route: _NetRoute) -> None:
        """Fill ``route.cover_t`` / ``cover_v`` from its keyed path cells, its vias and its extra covers (module docstring: the radii).

        With one profile these are 0.2's radii, in 0.2's arithmetic order:
        ``w/2 + w/2 + max(c, c) + g/2 == w + c + g/2`` bit for bit.
        """
        board, p = self.board, self.p
        n, g, dv = board.n, p.grid_mm, p.via_diameter_mm
        cover_t: dict[_Key, set[int]] = {key: set() for key in self.track_keys}
        cover_v: dict[float, set[int]] = {c: set() for c in self.via_keys}
        for (wc, cc), cells in route.keyed:
            if not cells:
                continue
            for key in self.track_keys:
                halo = board.halo_for(wc / 2.0 + key[0] / 2.0 + max(cc, key[1]) + g / 2.0)
                out = cover_t[key]
                for c in cells:
                    layer_base = n if c >= n else 0
                    board.halo_cells(halo, c - layer_base, layer_base, out)
            for ck in self.via_keys:
                halo = board.halo_for(dv / 2.0 + max(cc, ck) + wc / 2.0)
                out = cover_v[ck]
                for c in cells:
                    board.halo_cells(halo, c - n if c >= n else c, 0, out)
        vias = set(route.via_cells)
        if vias:
            cv = route.via_key
            for key in self.track_keys:
                halo = board.halo_for(dv / 2.0 + max(cv, key[1]) + key[0] / 2.0)
                out = cover_t[key]
                for k in vias:
                    board.halo_cells(halo, k, 0, out)
                    board.halo_cells(halo, k, n, out)
            for ck in self.via_keys:
                halo = board.halo_for(dv + max(cv, ck))
                out = cover_v[ck]
                for k in vias:
                    board.halo_cells(halo, k, 0, out)
        for key, cells in route.extra_t.items():
            cover_t[key] |= cells
        for ck, cells in route.extra_v.items():
            cover_v[ck] |= cells
        route.cover_t = cover_t
        route.cover_v = cover_v


# --------------------------------------------------------------------------- the search


def _search(
    neg: _Negotiation, net: int, seeds: list[int], targets: set[int], target_ij: tuple[int, int], window: tuple[int, int, int, int], strict: bool,
) -> list[int] | None:
    """Cheapest path (``layer * n + cell`` ids) from any seed to any target cell inside ``window``; ``None`` when none is reachable.

    ``strict`` makes every cell another net's halo covers an obstacle (0.1's
    rule; used only to recover a dropped net against legal copper), else such
    cells cost their negotiated node cost. Only stepped-into cells are
    checked here: the caller (:func:`_route_net`) keeps strict seeds and
    targets outside the halos. The per-state cost / predecessor
    arrays are the negotiation's, shared by every search and reset after
    each one (only the states this search touched). This is 0.2's search,
    used for every net without a rule (the board profile's arrays).
    """
    board = neg.board
    n, nx = board.n, board.nx
    stat = board.stat
    via_static = board.via_static
    ncost, vcost, halo, vhalo = neg.ncost, neg.vcost, neg.halo, neg.vhalo
    base, bend = neg.p.base_cost, neg.p.bend_cost
    i0, j0, i1, j1 = window
    tj, ti = target_ij
    # the heuristic, split by axis: base * |i - ti| + base * |j - tj|
    hx = [base * abs(i - ti) for i in range(nx)]
    hy = [base * abs(j - tj) for j in range(board.ny)]
    dirs = ((1, 1, 0), (nx, 0, 1), (-1, -1, 0), (-nx, 0, -1))  # (flat step, di, dj): east, south, west, north
    g_best = neg.g_best
    prev = neg.prev
    touched: list[int] = []
    heap: list[tuple[float, int, float, int]] = []
    counter = 0
    for c in seeds:
        s = c * _STATES_PER_CELL + _NO_DIR
        if g_best[s] == 0.0:
            continue
        g_best[s] = 0.0
        prev[s] = -1
        touched.append(s)
        j, i = divmod(c - n if c >= n else c, nx)
        heap.append((hx[i] + hy[j], counter, 0.0, s))
        counter += 1
    heapq.heapify(heap)
    heappush, heappop = heapq.heappush, heapq.heappop
    found = -1
    while heap:
        _, _, g, s = heappop(heap)
        if g > g_best[s]:
            continue
        c = s // _STATES_PER_CELL
        d = s - c * _STATES_PER_CELL
        if c in targets:
            found = s
            break
        if c < n:
            other = c + n
            k = c
        else:
            other = c - n
            k = other
        j = k // nx
        i = k - j * nx
        back = (d + 2) & 3 if d != _NO_DIR else -1
        for nd in range(4):
            if nd == back:
                continue  # no reversal onto the cell we came from
            step, di, dj = dirs[nd]
            ii = i + di
            jj = j + dj
            if ii < i0 or ii > i1 or jj < j0 or jj > j1:
                continue
            c2 = c + step
            o = stat[c2]
            if o != _FREE and o != net:
                continue  # BLOCKED or another net's pad keep-out: never entered, a target cell included
            if strict and halo[c2]:
                continue
            ng = g + ncost[c2]
            if d != _NO_DIR and nd != d:
                ng += bend
            s2 = c2 * _STATES_PER_CELL + nd
            if ng < g_best[s2]:
                if g_best[s2] == _INF:
                    touched.append(s2)
                g_best[s2] = ng
                prev[s2] = s
                heappush(heap, (ng + hx[ii] + hy[jj], counter, ng, s2))
                counter += 1
        v = via_static[k]
        if v is None:
            v = board.via_static_at(k)
        if (v == _FREE or v == net) and not (strict and (vhalo[k] or halo[other])):
            ng = g + vcost[k] + ncost[other] - base
            s2 = other * _STATES_PER_CELL + _NO_DIR
            if ng < g_best[s2]:
                if g_best[s2] == _INF:
                    touched.append(s2)
                g_best[s2] = ng
                prev[s2] = s
                heappush(heap, (ng + hx[i] + hy[j], counter, ng, s2))
                counter += 1
    path: list[int] = []
    if found >= 0:
        s = found
        while s >= 0:
            c = s // _STATES_PER_CELL
            if not path or path[-1] != c:
                path.append(c)
            s = prev[s]
        path.reverse()
    for s in touched:
        g_best[s] = _INF
    return path if found >= 0 else None


def _search_rules(
    neg: _Negotiation, rn: _RuleNet, seed_states: list[int], target_cells: set[int] | None, target_states: set[int] | None,
    target_ijs: list[tuple[int, int]], window: tuple[int, int, int, int], strict: bool, budget: float | None = None, length_mode: bool = False,
) -> tuple[list[int], float, int, int] | None:
    """The search for a rule net or a pair's centreline: ``(path, cost, first state, last state)`` or ``None``.

    As :func:`_search`, plus: the net's fence and via fence are obstacles;
    a cell in the neck-down zone costs (and, ``strict``, is judged by) the
    neck-down profile's plane; ``budget`` (mm) prunes every state whose
    length so far plus ``grid`` times the Manhattan distance to the nearest
    target exceeds it; ``length_mode`` makes the cost the length itself (a
    grid step ``grid``, a via ``via_length`` plus 1e-6 mm as a tie-break, no
    bend or congestion cost) - the shortest legal path. A seed state with a
    direction (a pair's launch) may only continue straight; targets are
    cells, or states when the arrival direction matters (a pair's far launch).
    """
    board = neg.board
    n, nx, ny = board.n, board.nx, board.ny
    p = neg.p
    stat = board.stat
    use_stat, fence, zone, vfence = rn.use_stat, rn.fence, rn.zone, rn.vfence
    nc_main, hal_main = neg.t_ncost[rn.key], neg.t_halo[rn.key]
    if rn.nd_key is not None:
        nc_nd, hal_nd = neg.t_ncost[rn.nd_key], neg.t_halo[rn.nd_key]
    else:
        nc_nd, hal_nd = nc_main, hal_main
    allow_vias = rn.allow_vias
    vcost = neg.v_cost[rn.via_key] if allow_vias else None
    vhal = neg.v_halo[rn.via_key] if allow_vias else None
    base, bend, g_mm = p.base_cost, p.bend_cost, p.grid_mm
    via_len = rn.via_len
    hbase = g_mm if length_mode else base
    i0, j0, i1, j1 = window
    if len(target_ijs) == 1:
        tj0, ti0 = target_ijs[0]
        hx = [abs(i - ti0) for i in range(nx)]
        hy = [abs(j - tj0) for j in range(ny)]

        def man(i: int, j: int) -> int:
            return hx[i] + hy[j]
    else:
        def man(i: int, j: int) -> int:
            return min(abs(i - ti) + abs(j - tj) for tj, ti in target_ijs)

    dirs = ((1, 1, 0), (nx, 0, 1), (-1, -1, 0), (-nx, 0, -1))
    g_best, prev = neg.g_best, neg.prev
    glen = neg.lengths()
    touched: list[int] = []
    heap: list[tuple[float, int, float, int]] = []
    counter = 0
    for s in seed_states:
        if g_best[s] == 0.0:
            continue
        g_best[s] = 0.0
        prev[s] = -1
        glen[s] = 0.0
        touched.append(s)
        c = s // _STATES_PER_CELL
        j, i = divmod(c - n if c >= n else c, nx)
        heap.append((hbase * man(i, j), counter, 0.0, s))
        counter += 1
    heapq.heapify(heap)
    heappush, heappop = heapq.heappush, heapq.heappop
    found = -1
    limit = None if budget is None else budget + _EPS
    while heap:
        _, _, g, s = heappop(heap)
        if g > g_best[s]:
            continue
        c = s // _STATES_PER_CELL
        d = s - c * _STATES_PER_CELL
        if (target_states is not None and s in target_states) or (target_cells is not None and c in target_cells):
            found = s
            break
        if c < n:
            other, k = c + n, c
        else:
            other, k = c - n, c - n
        j = k // nx
        i = k - j * nx
        back = (d + 2) & 3 if d != _NO_DIR else -1
        straight_only = d != _NO_DIR and prev[s] == -1
        l0 = glen[s]
        for nd in range(4):
            if nd == back or (straight_only and nd != d):
                continue
            step, di, dj = dirs[nd]
            ii = i + di
            jj = j + dj
            if ii < i0 or ii > i1 or jj < j0 or jj > j1:
                continue
            c2 = c + step
            if use_stat:
                o = stat[c2]
                if o != _FREE and o != rn.idx:
                    continue
            if fence is not None and fence[c2]:
                continue
            inz = zone is not None and zone[k + step]
            if strict and (hal_nd if inz else hal_main)[c2]:
                continue
            nl = l0 + g_mm
            if limit is not None and nl + g_mm * man(ii, jj) > limit:
                continue
            if length_mode:
                ng = g + g_mm
            else:
                ng = g + (nc_nd if inz else nc_main)[c2]
                if d != _NO_DIR and nd != d:
                    ng += bend
            s2 = c2 * _STATES_PER_CELL + nd
            if ng < g_best[s2]:
                if g_best[s2] == _INF:
                    touched.append(s2)
                g_best[s2] = ng
                prev[s2] = s
                glen[s2] = nl
                heappush(heap, (ng + hbase * man(ii, jj), counter, ng, s2))
                counter += 1
        if allow_vias:
            v = board.via_static_at(k)
            if (v == _FREE or v == rn.idx) and (vfence is None or not vfence[k]) and (fence is None or not fence[other]):
                inz = zone is not None and zone[k]
                if not (strict and (vhal[k] or (hal_nd if inz else hal_main)[other])):
                    nl = l0 + via_len
                    if limit is None or nl + g_mm * man(i, j) <= limit:
                        ng = g + (via_len + 1e-6 if length_mode else vcost[k] + (nc_nd if inz else nc_main)[other] - base)
                        s2 = other * _STATES_PER_CELL + _NO_DIR
                        if ng < g_best[s2]:
                            if g_best[s2] == _INF:
                                touched.append(s2)
                            g_best[s2] = ng
                            prev[s2] = s
                            glen[s2] = nl
                            heappush(heap, (ng + hbase * man(i, j), counter, ng, s2))
                            counter += 1
    result = None
    if found >= 0:
        path: list[int] = []
        s = found
        first = found
        while s >= 0:
            c = s // _STATES_PER_CELL
            if not path or path[-1] != c:
                path.append(c)
            first = s
            s = prev[s]
        path.reverse()
        result = (path, g_best[found], first, found)
    for s in touched:
        g_best[s] = _INF
    return result


def _path_length(board: _Board, path: list[int], via_len: float) -> float:
    """Grid length of a path (mm): ``grid`` per step on one layer, ``via_len`` per layer change."""
    n, g = board.n, board.p.grid_mm
    return sum(via_len if a // n != b // n else g for a, b in zip(path, path[1:]))


def _stub_length(board: _Board, t: _Terminal) -> float:
    """Length of the stub from the terminal cell to the pad centre, as :class:`_NetCopper` emits it."""
    x, y = board.pos(t.cell)
    return math.hypot(_q(t.pad.cx) - x, _q(t.pad.cy) - y)


def _route_net(
    neg: _Negotiation, net: Net, idx: int, terms: list[_Terminal], strict: bool = False, rn: _RuleNet | None = None, start: int | None = None,
) -> _NetRoute | str:
    """Grow the net's Steiner tree (module docstring) at the current costs; the reason when a terminal is unreachable.

    Terminals are tracked by their position in ``terms``, never by label: a
    footprint may repeat a pad number (a switch's paired pins), and each such
    pad is a terminal of its own that gets its own path end and stub. ``rn``
    (a net with a rule) adds its fence, neck-down profile and length budget
    (module docstring); without it this is 0.2's routine. A tree that breaks
    its length budget is grown again from each other terminal in turn (the
    search reads the negotiation, never changes it), and the first that fits
    is taken; the reason given is the first tree's. ``start`` fixes the
    first terminal (the retries).
    """
    board = neg.board
    n, nx = board.n, board.nx
    usable = [board.usable_layers(t, idx) for t in terms]
    # a terminal cell inside a keep-out (a foreign pad, a net-less pad, the board edge) is no seed and no target on that
    # layer; a terminal with no usable layer cannot be connected without breaking the router's own clearance
    fenced = [t for t, layers in zip(terms, usable) if not layers]
    if fenced:
        x, y = board.pos(fenced[0].cell)
        return (
            f"{fenced[0].label} terminal cell ({x:g}, {y:g}) is inside a keep-out on every copper layer of the pad "
            f"(a foreign pad, a pad without a net or the board edge is within clearance {board.p.clearance_mm:g} + width/2 of it)"
        )
    if rn is not None and rn.fence is not None:
        fence = rn.fence
        usable = [tuple(layer for layer in layers if not fence[layer * n + t.cell]) for t, layers in zip(terms, usable)]
        fenced = [t for t, layers in zip(terms, usable) if not layers]
        if fenced:
            x, y = board.pos(fenced[0].cell)
            w = rn.nd_key[0] if rn.zone is not None and rn.zone[fenced[0].cell] else rn.key[0]
            return (
                f"{fenced[0].label} terminal cell ({x:g}, {y:g}) is inside its net rule's keep-out on every copper layer of the pad "
                f"(a foreign pad or the board edge is within clearance {rn.key[1]:g} + width {w:g}/2 of it: the pad pitch cannot take that width)"
            )
    if strict:
        # against legal copper a terminal cell inside another net's halo is no seed and no target either: the search checks the
        # halo only on the cells it steps into, and a seed is never stepped into (negotiated mode counts it as a path cell instead)
        if rn is None:
            usable = [tuple(layer for layer in layers if not neg.halo[layer * n + t.cell]) for t, layers in zip(terms, usable)]
        else:
            usable = [tuple(layer for layer in layers if not neg.t_halo[rn.key_at(t.cell)][layer * n + t.cell]) for t, layers in zip(terms, usable)]
        fenced = [t for t, layers in zip(terms, usable) if not layers]
        if fenced:
            x, y = board.pos(fenced[0].cell)
            return f"{fenced[0].label} terminal cell ({x:g}, {y:g}) is inside another net's clearance halo on every copper layer of the pad"
    # the start: the terminal nearest the centroid of the pad centres (ties: natural ref order)
    mx = sum(t.pad.cx for t in terms) / len(terms)
    my = sum(t.pad.cy for t in terms) / len(terms)
    first = min(range(len(terms)), key=lambda q: math.hypot(terms[q].pad.cx - mx, terms[q].pad.cy - my))
    if rn is not None and rn.budget is not None and start is None and len(terms) > 2:
        got = _route_net(neg, net, idx, terms, strict, rn, first)
        if not isinstance(got, str) or not got.startswith("no route within max_length_mm"):
            return got
        for q in range(len(terms)):
            if q != first:
                again = _route_net(neg, net, idx, terms, strict, rn, q)
                if not isinstance(again, str):
                    return again
        return got
    if start is not None:
        first = start
    ij = [divmod(t.cell, nx) for t in terms]  # (j, i)
    margin = max(1, int(round(board.p.window_mm / board.p.grid_mm)))
    box = (min(i for _, i in ij), min(j for j, _ in ij), max(i for _, i in ij), max(j for j, _ in ij))
    full = (0, 0, board.nx - 1, board.ny - 1)
    route = _NetRoute()
    stubbed: set[int] = set()
    at_terminal: dict[int, int] = {}  # tree cell -> terminal index, for the terminals' own cells
    tree: list[int] = []
    tree_set: set[int] = set()
    remaining = [q for q in range(len(terms)) if q != first]
    dmin = {q: abs(ij[q][1] - ij[first][1]) + abs(ij[q][0] - ij[first][0]) for q in remaining}
    budget = rn.budget if rn is not None else None
    stubs = sum(_stub_length(board, t) for t in terms) if budget is not None else 0.0
    used = 0.0
    if budget is not None and stubs > budget + _EPS:
        return f"max_length_mm {budget:g}: the stubs from the terminal cells to the pad centres alone are {stubs:.3f} mm"

    def cells_of(q: int) -> list[int]:
        return [layer * n + terms[q].cell for layer in usable[q]]

    def stub(q: int, layer: int) -> None:
        if q not in stubbed:
            stubbed.add(q)
            route.steps.append(("stub", terms[q], layer))

    def grow(cells: list[int]) -> None:
        for c in cells:
            if c in tree_set:
                continue
            tree_set.add(c)
            tree.append(c)
            j, i = divmod(c - n if c >= n else c, nx)
            for q in remaining:
                tj, ti = ij[q]
                dd = abs(i - ti) + abs(j - tj)
                if dd < dmin[q]:
                    dmin[q] = dd

    def attach(q: int) -> None:
        for c in cells_of(q):
            at_terminal.setdefault(c, q)
        grow(cells_of(q))

    attach(first)
    while remaining:
        already = [q for q in remaining if any(c in tree_set for c in cells_of(q))]
        if already:
            q = already[0]
            stub(q, next(c for c in cells_of(q) if c in tree_set) // n)
            remaining.remove(q)
            attach(q)
            continue
        target = min(remaining, key=lambda q: dmin[q])  # ties: the first in natural ref order
        targets = set(cells_of(target))
        grown = margin
        left = None if budget is None else budget - stubs - used
        while True:
            window = (max(0, box[0] - grown), max(0, box[1] - grown), min(board.nx - 1, box[2] + grown), min(board.ny - 1, box[3] + grown))
            if rn is None:
                path = _search(neg, idx, tree, targets, ij[target], window, strict)
            else:
                got = _search_rules(neg, rn, [c * _STATES_PER_CELL + _NO_DIR for c in tree], targets, None, [ij[target]], window, strict, left)
                path = None if got is None else got[0]
            if path is not None or window == full:
                break
            grown *= 2
        if path is None and left is not None:
            # nothing within the budget at the negotiated costs: the shortest legal path, if it fits
            got = _search_rules(neg, rn, [c * _STATES_PER_CELL + _NO_DIR for c in tree], targets, None, [ij[target]], full, strict, None, length_mode=True)
            if got is None:
                return ", ".join(terms[q].label for q in remaining) + " unreachable from the routed part of the net"
            shortest = _path_length(board, got[0], rn.via_len)
            if shortest > left + _EPS:
                return (
                    f"no route within max_length_mm {budget:g}: the shortest legal branch to {terms[target].label} is {shortest:.3f} mm "
                    f"with {left:.3f} mm of the budget left (stubs {stubs:.3f} mm, {used:.3f} mm already routed)"
                )
            path = got[0]
        if path is None:
            return ", ".join(terms[q].label for q in remaining) + " unreachable from the routed part of the net"
        seed = at_terminal.get(path[0])
        if seed is not None:
            stub(seed, path[0] // n)
        route.steps.append(("path", path, 0))
        for a, b in zip(path, path[1:]):
            if a // n != b // n:
                route.via_cells.append(a % n)
        route.path_cells.update(path)
        if budget is not None:
            used += _path_length(board, path, rn.via_len)
        stub(target, path[-1] // n)
        remaining.remove(target)
        grow(path)
        attach(target)
    if budget is not None and used + stubs > budget + _EPS:
        return f"no route within max_length_mm {budget:g}: the routed tree is {used + stubs:.3f} mm"
    _key_route(neg, route, rn)
    neg.cover(route)
    return route


def _key_route(neg: _Negotiation, route: _NetRoute, rn: _RuleNet | None) -> None:
    """Group the route's path cells by the profile of their copper (neck-down zone or not) and set its via clearance."""
    if rn is None:
        route.keyed = [(neg.key0, route.path_cells)]
        route.via_key = neg.vkey0
        return
    route.via_key = rn.via_key
    if rn.zone is None:
        route.keyed = [(rn.key, route.path_cells)]
        return
    n, zone = neg.board.n, rn.zone
    neck = {c for c in route.path_cells if zone[c - n if c >= n else c]}
    route.keyed = [(rn.key, route.path_cells - neck), (rn.nd_key, neck)]


# --------------------------------------------------------------------------- coupled pairs


@dataclass(slots=True)
class _Launch:
    """One end of a coupled pair: the launch cell, the outward direction, which side net ``a`` takes, and the two breakouts.

    ``sigma`` is net ``a``'s side relative to the direction of travel along
    the centreline (+1 left, -1 right): at the first end travel is the launch
    direction, at the second its reverse.
    """

    end: int
    layer: int
    cell: int
    d: int
    sigma: int
    seg_a: tuple[tuple[float, float], tuple[float, float]]  # pad centre -> start of the offset track
    seg_b: tuple[tuple[float, float], tuple[float, float]]
    cover_t: dict[_Key, set[int]] | None = None
    cover_v: dict[float, set[int]] | None = None


@dataclass(slots=True)
class _Pair:
    """A declared coupled pair (nets ``a`` < ``b`` by name) and its virtual net."""

    name: str
    a: str
    b: str
    ia: int
    ib: int
    rule: NetRule
    w: float
    s: float
    c: float
    net: Net
    rn: _RuleNet
    ends: list[tuple[_Terminal, _Terminal]] = field(default_factory=list)
    launches: list[list[_Launch]] = field(default_factory=list)
    reason: str | None = None

    @property
    def h(self) -> float:
        """Offset of each track from the centreline: (w + s) / 2."""
        return (self.w + self.s) / 2.0

    @property
    def width(self) -> float:
        """The virtual centreline track's width: 2w + s (the pair's copper envelope)."""
        return 2.0 * self.w + self.s

    @property
    def req(self) -> float:
        """The distance the two tracks keep from each other: the spacing, or the pair's clearance if smaller."""
        return min(self.s, self.c)


def _setup_pair(board: _Board, pr: _Pair, clear_of: Callable[[int], float], track_of: Callable[[int], tuple[float, float]]) -> None:
    """The pair's fence, the matching of its pad ends and the launch candidates at each end (module docstring)."""
    p, n, g = board.p, board.n, board.p.grid_mm
    W, h = pr.width, pr.h
    fence = bytearray(2 * n)
    for k in range(n):
        if board._edge_distance(k) < p.edge_clearance_mm + W / 2.0 - _EPS:
            fence[k] = 1
            fence[n + k] = 1
    for geom in board.pads:
        r = max(pr.c, clear_of(geom.net)) + W / 2.0 + g / 2.0
        for k, dist in board.cells_near_box(geom, r):
            if dist < r - _EPS:
                for layer in geom.layers:
                    fence[layer * n + k] = 1
    pr.rn.fence = fence
    ta, tb = board.terminals[pr.a], board.terminals[pr.b]

    def dist(t: _Terminal, u: _Terminal) -> float:
        return math.hypot(t.pad.cx - u.pad.cx, t.pad.cy - u.pad.cy)

    if dist(ta[0], tb[0]) + dist(ta[1], tb[1]) <= dist(ta[0], tb[1]) + dist(ta[1], tb[0]) + _EPS:
        pr.ends = [(ta[0], tb[0]), (ta[1], tb[1])]
    else:
        pr.ends = [(ta[0], tb[1]), (ta[1], tb[0])]
    steps = max(1, int(round(p.window_mm / g)))
    foreign_terms = [(board.net_index[name], t) for name, terms in board.terminals.items() if name not in (pr.a, pr.b) for t in terms]
    pr.launches = []
    for end, (t_a, t_b) in enumerate(pr.ends):
        found: list[_Launch] = []
        first: str | None = None  # the problem of the nearest candidate tried: the most telling reason when none launches
        pa = (_q(t_a.pad.cx), _q(t_a.pad.cy))
        pb = (_q(t_b.pad.cx), _q(t_b.pad.cy))
        mx, my = (t_a.pad.cx + t_b.pad.cx) / 2.0, (t_a.pad.cy + t_b.pad.cy) / 2.0
        i0, j0 = board._nearest_cell(mx, my)
        i0, j0 = min(max(i0, 0), board.nx - 1), min(max(j0, 0), board.ny - 1)
        for layer in (0, 1):
            if layer not in t_a.pad.layers or layer not in t_b.pad.layers:
                continue
            for d in range(4):
                dx, dy = _DIRS[d]
                lx, ly = _left(d)
                proj = (t_a.pad.cx - mx) * lx + (t_a.pad.cy - my) * ly
                if abs(proj) < 1e-9:
                    first = first or f"the pads lie along the {_DIR_NAMES[d]} direction"
                    continue
                side = 1 if proj > 0 else -1
                for step in range(steps + 1):
                    i, j = i0 + step * dx, j0 + step * dy
                    if not (0 <= i < board.nx and 0 <= j < board.ny):
                        break
                    k = j * board.nx + i
                    if fence[layer * n + k]:
                        continue
                    x, y = board.xy(k)
                    sa = (_q(x + side * h * lx), _q(y + side * h * ly))
                    sb = (_q(x - side * h * lx), _q(y - side * h * ly))
                    fa = (sa, (_q(sa[0] + dx * g), _q(sa[1] + dy * g)))
                    fb = (sb, (_q(sb[0] + dx * g), _q(sb[1] + dy * g)))
                    why = _breakout_problem(board, pr, layer, (pa, sa), (pb, sb), fa, fb, clear_of, track_of, foreign_terms)
                    if why is None:
                        found.append(_Launch(end, layer, k, d, side if end == 0 else -side, (pa, sa), (pb, sb)))
                        break
                    first = first or why
        if not found and pr.reason is None:
            why = first or "no layer carries both pads"
            pr.reason = f"no launch for the coupled pair {pr.a}/{pr.b} at {t_a.label}/{t_b.label} within {p.window_mm:g} mm: {why}"
        pr.launches.append(found)


def _breakout_problem(
    board: _Board, pr: _Pair, layer: int, seg_a: tuple, seg_b: tuple, first_a: tuple, first_b: tuple,
    clear_of: Callable[[int], float], track_of: Callable[[int], tuple[float, float]], foreign_terms: list[tuple[int, _Terminal]],
) -> str | None:
    """Why the two straight breakouts of one launch candidate break a clearance, or ``None`` (exact geometry)."""
    p, g, w = board.p, board.p.grid_mm, pr.w
    for seg, own, other, name in ((seg_a, pr.ia, pr.ib, pr.a), (seg_b, pr.ib, pr.ia, pr.b)):
        length = math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])
        if pr.rule.pair_uncoupled_max_mm is not None and length > pr.rule.pair_uncoupled_max_mm + _EPS:
            return f"the breakout of {name} would be {length:.3f} mm, over pair_uncoupled_max_mm {pr.rule.pair_uncoupled_max_mm:g}"
        x, y = seg[1]
        if board.edge_distance_xy(x, y) < p.edge_clearance_mm + w / 2.0 - _EPS:
            return f"the breakout of {name} ends within edge_clearance + width/2 of the outline"
        for geom in board.pads:
            if geom.net == own or layer not in geom.layers:
                continue
            req = pr.req if geom.net == other else max(pr.c, clear_of(geom.net))
            gap = _seg_box(seg[0], seg[1], geom.box) - w / 2.0
            if gap < req - _EPS:
                return f"the breakout of {name} passes {geom.ref}.{geom.number or '(unnumbered)'} at {max(gap, 0.0):.3f} mm < {req:g} mm"
        for idx, t in foreign_terms:
            if layer not in t.pad.layers:
                continue
            wx, cx = track_of(idx)
            tx, ty = board.xy(t.cell)
            if _pt_seg(tx, ty, seg[0], seg[1]) < w / 2.0 + wx / 2.0 + max(pr.c, cx) + g / 2.0 - _EPS:
                return f"the breakout of {name} passes the terminal of {t.label} too near for its track"
    if _seg_seg(*seg_a, *seg_b) - w < pr.req - _EPS:
        return "the two breakouts come closer than the pair's spacing"
    if _seg_seg(*seg_a, *first_b) - w < pr.req - _EPS or _seg_seg(*seg_b, *first_a) - w < pr.req - _EPS:
        return "a breakout crosses the other track's first coupled step"
    return None


def _launch_cover(neg: _Negotiation, pr: _Pair, lc: _Launch) -> None:
    """The halo the launch's two breakout segments cover in every profile plane (exact distance from the segment, + grid/2 on tracks)."""
    if lc.cover_t is not None:
        return
    board, p = neg.board, neg.p
    n, g, dv, w = board.n, p.grid_mm, p.via_diameter_mm, pr.w
    lc.cover_t = {key: set() for key in neg.track_keys}
    lc.cover_v = {c: set() for c in neg.via_keys}
    for seg in (lc.seg_a, lc.seg_b):
        radii_t = {key: w / 2.0 + key[0] / 2.0 + max(pr.c, key[1]) + g / 2.0 for key in neg.track_keys}
        radii_v = {c: dv / 2.0 + max(pr.c, c) + w / 2.0 for c in neg.via_keys}
        reach = max([*radii_t.values(), *radii_v.values()])
        (ax, ay), (bx, by) = seg
        i0, j0 = board._nearest_cell(min(ax, bx) - reach, min(ay, by) - reach)
        i1, j1 = board._nearest_cell(max(ax, bx) + reach, max(ay, by) + reach)
        for j in range(max(0, j0 - 1), min(board.ny - 1, j1 + 1) + 1):
            for i in range(max(0, i0 - 1), min(board.nx - 1, i1 + 1) + 1):
                k = j * board.nx + i
                x, y = board.xy(k)
                d = _pt_seg(x, y, seg[0], seg[1])
                for key, r in radii_t.items():
                    if d < r - _EPS:
                        lc.cover_t[key].add(lc.layer * n + k)
                for c, r in radii_v.items():
                    if d < r - _EPS:
                        lc.cover_v[c].add(k)


def _launch_conflicts(pr: _Pair, lc: _Launch, routes: Mapping[str, _NetRoute]) -> bool:
    """Whether the launch's breakouts touch another routed net's copper (the strict recovery keeps off it)."""
    for name, r in routes.items():
        if name == pr.name:
            continue
        for key, cells in r.keyed:
            if not cells.isdisjoint(lc.cover_t[key]):
                return True
        if r.via_cells and not lc.cover_v[r.via_key].isdisjoint(r.via_cells):
            return True
    return False


def _route_pair(neg: _Negotiation, pr: _Pair, strict: bool = False, routes: Mapping[str, _NetRoute] | None = None) -> _NetRoute | str:
    """Route the pair's centreline as its virtual net between two launches (same layer, same side); the reason when none exists."""
    if pr.reason is not None:
        return pr.reason
    board = neg.board
    n, nx = board.n, board.nx
    launches = [list(end) for end in pr.launches]
    for end in launches:
        for lc in end:
            _launch_cover(neg, pr, lc)
    if strict:
        launches = [[lc for lc in end if not _launch_conflicts(pr, lc, routes or {})] for end in launches]
        if not launches[0] or not launches[1]:
            return f"coupled pair {pr.a}/{pr.b}: every launch's breakout touches the legal copper"
    budget = pr.rule.pair_uncoupled_max_mm
    if budget is not None:  # a net's two breakouts are uncoupled copper: keep only launches some opposite launch fits the budget with
        def fits(x: _Launch, y: _Launch) -> bool:
            return _seg_len(x.seg_a) + _seg_len(y.seg_a) <= budget + _EPS and _seg_len(x.seg_b) + _seg_len(y.seg_b) <= budget + _EPS

        launches = [[lc for lc in launches[0] if any(fits(lc, o) for o in launches[1])], [lc for lc in launches[1] if any(fits(o, lc) for o in launches[0])]]
        if not launches[0] or not launches[1]:
            return (f"coupled pair {pr.a}/{pr.b}: no two launches keep each net's breakouts within pair_uncoupled_max_mm {budget:g} "
                    "(a net's uncoupled length counts the breakouts at both ends)")
    margin = max(1, int(round(board.p.window_mm / board.p.grid_mm)))
    full = (0, 0, board.nx - 1, board.ny - 1)
    best: tuple[float, int, int, list[int], _Launch, _Launch] | None = None
    for layer in (0, 1):
        for sigma in (1, -1):
            seeds = {(layer * n + lc.cell) * _STATES_PER_CELL + lc.d: lc for lc in launches[0] if lc.layer == layer and lc.sigma == sigma}
            targets = {(layer * n + lc.cell) * _STATES_PER_CELL + ((lc.d + 2) & 3): lc for lc in launches[1] if lc.layer == layer and lc.sigma == sigma}
            if not seeds or not targets:
                continue
            cells = [divmod(s // _STATES_PER_CELL - layer * n, nx) for s in (*seeds, *targets)]  # (j, i)
            box = (min(i for _, i in cells), min(j for j, _ in cells), max(i for _, i in cells), max(j for j, _ in cells))
            target_ijs = [divmod(s // _STATES_PER_CELL - layer * n, nx) for s in targets]
            grown = margin
            while True:
                window = (max(0, box[0] - grown), max(0, box[1] - grown), min(board.nx - 1, box[2] + grown), min(board.ny - 1, box[3] + grown))
                got = _search_rules(neg, pr.rn, list(seeds), None, set(targets), target_ijs, window, strict)
                if got is not None or window == full:
                    break
                grown *= 2
            if got is None:
                continue
            path, cost, s0, s1 = got
            if len(path) < 2:
                continue
            if best is None or cost < best[0]:
                best = (cost, layer, sigma, path, seeds[s0], targets[s1])
    if best is None:
        return f"coupled pair {pr.a}/{pr.b}: no centreline of width {pr.width:g} mm (2w + s) joins the launches on one layer"
    _, layer, sigma, path, la, lb = best
    route = _NetRoute()
    route.steps.append(("pair", (layer, path, la, lb, sigma), 0))
    route.path_cells.update(path)
    route.keyed = [(pr.rn.key, route.path_cells)]
    route.via_key = pr.c
    route.extra_t = {key: la.cover_t[key] | lb.cover_t[key] for key in neg.track_keys}
    route.extra_v = {c: la.cover_v[c] | lb.cover_v[c] for c in neg.via_keys}
    neg.cover(route)
    return route


def _pair_polylines(board: _Board, pr: _Pair, data: tuple) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[list[int]]]:
    """The two tracks as point lists (pad, breakout end, offset centreline with mitred corners, breakout start, pad) and the
    indices of each list's points that start an axis-aligned coupled segment."""
    layer, path, la, lb, sigma = data
    n, nx = board.n, board.nx
    cells = [c - layer * n for c in path]
    xy = [board.xy(k) for k in cells]
    dirs = []
    for a, b in zip(cells, cells[1:]):
        dirs.append({1: 0, nx: 1, -1: 2, -nx: 3}[b - a])
    verts = [xy[0]]
    segd = [dirs[0]]
    for t in range(1, len(dirs)):
        if dirs[t] != dirs[t - 1]:
            verts.append(xy[t])
            segd.append(dirs[t])
    verts.append(xy[-1])
    h = pr.h

    def shift(pt: tuple[float, float], amount: float, normal: tuple[int, int]) -> tuple[float, float]:
        return (pt[0] + amount * normal[0], pt[1] + amount * normal[1])

    def offset(side: int) -> tuple[list[tuple[float, float]], list[int]]:
        pts = [shift(verts[0], side * h, _left(segd[0]))]
        straight = [0]  # index into pts of the start of each segment parallel to the centreline
        for i in range(1, len(verts) - 1):
            u0, u1 = _DIRS[segd[i - 1]], _DIRS[segd[i]]
            turn = u0[0] * u1[1] - u0[1] * u1[0]
            l0, l1 = _left(segd[i - 1]), _left(segd[i])
            if side * turn > 0:  # the outer track: a chamfer between the two offset lines' feet at the corner
                pts.append(shift(verts[i], side * h, l0))
                pts.append(shift(verts[i], side * h, l1))
            else:  # the inner track: the offset lines' intersection
                pts.append((verts[i][0] + side * h * (l0[0] + l1[0]), verts[i][1] + side * h * (l0[1] + l1[1])))
            straight.append(len(pts) - 1)
        pts.append(shift(verts[-1], side * h, _left(segd[-1])))
        return pts, straight

    off_a, straight_a = offset(sigma)
    off_b, straight_b = offset(-sigma)
    pts_a = [la.seg_a[0], *[(_q(x), _q(y)) for x, y in off_a], lb.seg_a[0]]
    pts_b = [la.seg_b[0], *[(_q(x), _q(y)) for x, y in off_b], lb.seg_b[0]]
    return pts_a, pts_b, [straight_a, straight_b]


def _polyline_length(pts: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _seg_len(seg: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])


# --------------------------------------------------------------------------- exact audit


class _World:
    """The emitted copper (tracks as segments, vias as discs) and every pad (boxes), bucketed, for exact clearance queries.

    The distance two items of different nets must keep is the larger of the
    nets' clearances (``clear_of``; a track at a neck-down width carries the
    board clearance), except the two nets of a coupled pair, which keep the
    pair's ``min(spacing, clearance)`` (``partner``).
    """

    _CELL = 2.0

    def __init__(self, board: _Board, clear_of: Callable[[int], float], partner: Mapping[int, tuple[int, float]]) -> None:
        self.board = board
        self.clear_of = clear_of
        self.partner = partner
        self.buckets: dict[tuple[int, int], list[tuple]] = {}
        self.cmax = board.p.clearance_mm
        for geom in board.pads:
            c = clear_of(geom.net)
            self._add(("box", geom.net, frozenset(geom.layers), geom.box, 0.0, c, f"pad {geom.ref}.{geom.number or '(unnumbered)'}"), geom.box)

    def _add(self, item: tuple, bbox: tuple[float, float, float, float]) -> None:
        self.cmax = max(self.cmax, item[5])
        s = self._CELL
        for bi in range(math.floor(bbox[0] / s), math.floor(bbox[2] / s) + 1):
            for bj in range(math.floor(bbox[1] / s), math.floor(bbox[3] / s) + 1):
                self.buckets.setdefault((bi, bj), []).append(item)

    def add_track(self, net: int, t: Track, c: float) -> None:
        layer = LAYERS.index(t.layer)
        r = t.width_mm / 2.0
        (ax, ay), (bx, by) = t.start, t.end
        self._add(("seg", net, frozenset((layer,)), (t.start, t.end), r, c, f"track of {t.net}"), (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r))

    def add_via(self, net: int, v: Via, c: float) -> None:
        r = v.diameter_mm / 2.0
        self._add(("disc", net, frozenset((0, 1)), (v.x_mm, v.y_mm), r, c, f"via of {v.net}"), (v.x_mm - r, v.y_mm - r, v.x_mm + r, v.y_mm + r))

    def seg_problem(self, net: int, layer: int, a: tuple[float, float], b: tuple[float, float], w: float, c: float, what: str) -> str | None:
        """Why a segment of ``net`` breaks a clearance to a foreign item, or ``None``."""
        r = w / 2.0
        m = r + self.cmax + max(self.board.p.via_diameter_mm, w) + _TOL
        s = self._CELL
        seen: set[int] = set()
        partner = self.partner.get(net)
        for bi in range(math.floor((min(a[0], b[0]) - m) / s), math.floor((max(a[0], b[0]) + m) / s) + 1):
            for bj in range(math.floor((min(a[1], b[1]) - m) / s), math.floor((max(a[1], b[1]) + m) / s) + 1):
                for item in self.buckets.get((bi, bj), ()):
                    if id(item) in seen:
                        continue
                    seen.add(id(item))
                    kind, other, layers, geom, half, oc, label = item
                    if other == net or layer not in layers:
                        continue
                    req = partner[1] if partner is not None and other == partner[0] else max(c, oc)
                    if kind == "box":
                        gap = _seg_box(a, b, geom) - r
                    elif kind == "seg":
                        gap = _seg_seg(a, b, geom[0], geom[1]) - r - half
                    else:
                        gap = _pt_seg(geom[0], geom[1], a, b) - r - half
                    if gap < req - _TOL:
                        return f"{what} comes {max(gap, 0.0):.4f} mm from the {label} (needs {req:g} mm)"
        return None


# --------------------------------------------------------------------------- emission


def _provenance(
    net: Net, p: RoutingParams, how: str, iterations: int, legal: bool, *, version: str = ROUTER_VERSION, rules_active: bool = False,
    rule: NetRule | None = None, extra_refs: list[str] | None = None, story: str | None = None, rule_note: str | None = None,
) -> Provenance:
    refs = sorted({pin.component_ref for pin in net.pins} | set(extra_refs or []))
    if story is None:
        if how == "recovered":
            story = f"routed against the legal copper as an obstacle after {iterations} negotiation iteration(s) left it in conflict"
        else:
            story = f"negotiated-congestion route, {iterations} iteration(s)" + ("" if legal else ", kept after the conflicting nets were ripped up")
    derived = [f"net:{net.name}", *(f"placement:{r}" for r in refs), p.derived_from_entry(rules=rules_active)]
    if rule is not None:
        derived.append(rule.derived_from_entry())
    rule_text = f"; {rule_note}" if rule_note else ""
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=ROUTER_ID,
        tool_version=version,
        derived_from=derived,
        note=f"grid maze route on F.Cu/B.Cu ({story}){rule_text}; validity is decided by kicad-cli DRC (pcb.routing checks the IR geometry only)",
    )


class _NetCopper:
    """The tracks and vias of one routed net, built from its recorded steps (a unit step touching the neck-down zone at the neck-down width)."""

    def __init__(self, board: _Board, net: Net, prov: Provenance, width: float | None = None, neck: float | None = None, zone: bytearray | None = None) -> None:
        self.board = board
        self.net = net
        self.prov = prov
        self.p = board.p
        self.width = self.p.track_width_mm if width is None else width
        self.neck = neck
        self.zone = zone if neck is not None else None
        self.tracks: list[Track] = []
        self.vias: list[Via] = []

    def _track(self, layer: int, a: int, b: int, width: float) -> None:
        start, end = self.board.pos(a), self.board.pos(b)
        if start == end:
            return
        self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=width, provenance=self.prov))

    def add_path(self, path: list[int]) -> None:
        """Merge the unit steps of ``path`` (``layer * n + cell`` ids) into tracks and vias."""
        n, nx = self.board.n, self.board.nx
        zone = self.zone
        run_start: int | None = None
        run_dir: tuple[int, int] | None = None
        run_w = self.width
        for a, b in zip(path, path[1:]):
            la, ka = divmod(a, n)
            lb, kb = divmod(b, n)
            if la != lb:
                if run_start is not None:
                    self._track(la, run_start, ka, run_w)
                    run_start = run_dir = None
                x, y = self.board.pos(ka)
                self.vias.append(
                    Via(net=self.net.name, x_mm=x, y_mm=y, drill_mm=self.p.via_drill_mm, diameter_mm=self.p.via_diameter_mm, layers=LAYERS, provenance=self.prov)
                )
                continue
            step = (kb % nx - ka % nx, kb // nx - ka // nx)
            w = self.neck if zone is not None and (zone[ka] or zone[kb]) else self.width
            if run_start is None:
                run_start, run_dir, run_w = ka, step, w
            elif step != run_dir or w != run_w:
                self._track(la, run_start, ka, run_w)
                run_start, run_dir, run_w = ka, step, w
        if run_start is not None:
            self._track(path[-1] // n, run_start, path[-1] % n, run_w)

    def add_stub(self, terminal: _Terminal, layer: int) -> None:
        """One segment from the terminal cell to the exact pad centre (nothing when the cell is the centre)."""
        start = self.board.pos(terminal.cell)
        end = (_q(terminal.pad.cx), _q(terminal.pad.cy))
        width = self.neck if self.zone is not None and self.zone[terminal.cell] else self.width
        if start != end:
            self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=width, provenance=self.prov))

    def length_mm(self) -> float:
        return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in self.tracks)


def _emit(board: _Board, net: Net, route: _NetRoute, prov: Provenance, rn: _RuleNet | None) -> _NetCopper:
    if rn is None or rn.rule is None:
        copper = _NetCopper(board, net, prov)
    else:
        copper = _NetCopper(board, net, prov, rn.key[0], rn.nd_key[0] if rn.nd_key is not None else None, rn.zone)
    for kind, item, layer in route.steps:
        if kind == "stub":
            copper.add_stub(item, layer)
        else:
            copper.add_path(item)
    return copper


# --------------------------------------------------------------------------- length matching


def _straight_runs(board: _Board, route: _NetRoute) -> list[tuple[int, int, int, int]]:
    """``(step index, first path index, steps, direction)`` of every maximal straight single-layer run of the route's paths."""
    n, nx = board.n, board.nx
    code = {1: 0, nx: 1, -1: 2, -nx: 3}
    runs = []
    for si, (kind, path, _) in enumerate(route.steps):
        if kind != "path":
            continue
        t = 0
        while t < len(path) - 1:
            if path[t] // n != path[t + 1] // n:
                t += 1
                continue
            d = code[path[t + 1] - path[t]]
            u = t + 1
            while u + 1 < len(path) and path[u] // n == path[u + 1] // n and code[path[u + 1] - path[u]] == d:
                u += 1
            runs.append((si, t, u - t, d))
            t = u
    return runs


def _meander(neg: _Negotiation, rn: _RuleNet, route: _NetRoute, cells_needed: int) -> tuple[_NetRoute, dict[str, Any]] | str:
    """A copy of ``route`` with square serpentine bumps adding ``cells_needed`` grid steps up and as many down, or why not.

    The caller has removed ``route`` from the negotiation, so the halo counts
    are the other nets' copper only: every new cell must be free of them.
    """
    board, p = neg.board, neg.p
    n, nx, ny, g = board.n, board.nx, board.ny, p.grid_mm
    amp = max(1, int(p.meander_amplitude_mm / g + _EPS))
    pitch = max(1, int(round(p.meander_pitch_mm / g)))
    if pitch * g - rn.key[0] < rn.key[1] - _EPS:
        return f"meander legs {pitch * g:g} mm apart would not keep the net's own width {rn.key[0]:g} + clearance {rn.key[1]:g} between them"
    bumps = -(-cells_needed // amp)
    # the height spread as evenly as whole grid steps allow (the first bumps one step taller when it does not divide)
    heights = [cells_needed // bumps + (1 if b < cells_needed % bumps else 0) for b in range(bumps)]
    span = (2 * bumps - 1) * pitch
    runs = sorted(_straight_runs(board, route), key=lambda r: (-r[2], r[0], r[1]))
    halo = neg.t_halo[rn.key]
    stat, fence, zone = board.stat, rn.fence, rn.zone
    # a bump over the net's own copper would be shorted by it (no electrical length): every bump cell keeps the net's own
    # width + clearance from its other path cells and via_diameter/2 + clearance + width/2 from its vias
    w_own, c_own = rn.key
    r_path = int(math.ceil((w_own + c_own) / g - _EPS))
    r_via = int(math.ceil((p.via_diameter_mm / 2.0 + c_own + w_own / 2.0) / g - _EPS))
    for si, t0, steps, d in runs:
        if steps < span + 2 * pitch:
            continue
        path = route.steps[si][1]
        layer = path[t0] // n
        lb = layer * n
        start = path[t0] - lb
        du = {0: 1, 1: nx, 2: -1, 3: -nx}[d]
        offset = pitch + ((steps - 2 * pitch) - span) // 2
        run = set(path[t0: t0 + steps + 1])
        own = [c - lb for c in route.path_cells if c // n == layer and c not in run]
        own_vias = list(route.via_cells)

        def near_own(c: int) -> bool:
            j, i = divmod(c - lb, nx)
            for o in own:
                oj, oi = divmod(o, nx)
                if (oi - i) ** 2 + (oj - j) ** 2 < r_path * r_path:
                    return True
            for o in own_vias:
                oj, oi = divmod(o % n, nx)
                if (oi - i) ** 2 + (oj - j) ** 2 < r_via * r_via:
                    return True
            return False

        for side in (1, -1):
            lx, ly = _left(d)
            vi, vj = side * lx, side * ly
            new_segments: list[list[int]] = []
            ok = True
            for b, height in enumerate(heights):
                o = offset + 2 * b * pitch
                seg: list[int] = []
                for q in range(pitch + 1):
                    base_k = start + (o + q) * du
                    j0, i0 = divmod(base_k, nx)
                    tops = range(1, height + 1) if q == 0 else ([height] if q < pitch else range(height, 0, -1))
                    for up in tops:
                        i, j = i0 + up * vi, j0 + up * vj
                        if not (0 <= i < nx and 0 <= j < ny):
                            ok = False
                            break
                        c = lb + j * nx + i
                        # _FREE only: a cell of the net's own pad keep-out would put the bump on (or against) its own pad
                        if stat[c] != _FREE or (fence is not None and fence[c]) or (zone is not None and zone[c - lb]) or halo[c] or c in route.path_cells or near_own(c):
                            ok = False
                            break
                        seg.append(c)
                    if not ok:
                        break
                if not ok:
                    break
                new_segments.append(seg)
            if not ok:
                continue
            # the new path: each bump replaces the run cells strictly between its two legs
            new_path = list(path[: t0 + offset + 1])
            for b in range(bumps):
                o = offset + 2 * b * pitch
                new_path.extend(new_segments[b])
                new_path.append(path[t0 + o + pitch])
                if b + 1 < bumps:
                    new_path.extend(path[t0 + o + pitch + 1: t0 + o + 2 * pitch + 1])
            new_path.extend(path[t0 + offset + span + 1:])
            fresh = _NetRoute(steps=list(route.steps), via_cells=list(route.via_cells), how=route.how)
            fresh.steps[si] = ("path", new_path, 0)
            for kind, item, _ in fresh.steps:
                if kind == "path":
                    fresh.path_cells.update(item)
            _key_route(neg, fresh, rn)
            neg.cover(fresh)
            a, b = board.pos(path[t0] - lb), board.pos(path[t0 + steps] - lb)
            info = {
                "layer": LAYERS[layer], "run": [a, b], "side": "left" if side == 1 else "right", "bumps": bumps,
                "amplitude_mm": [_q(hgt * g) for hgt in heights], "leg_pitch_mm": _q(pitch * g), "added_mm": _q(2 * cells_needed * g),
            }
            return fresh, info
    return (
        f"no straight run holds {bumps} bump(s) of up to {amp * g:g} mm at a {pitch * g:g} mm leg pitch "
        f"({span + 2 * pitch} grid steps needed) clear of the other copper"
    )


# --------------------------------------------------------------------------- the router


def route_board(
    ir: CircuitIR, library: KicadLibrary, params: RoutingParams | None = None, *,
    rules: Mapping[str, NetRule] | None = None, inner_layers: bool = False, progress: Callable[[dict[str, Any]], None] | None = None,
) -> Routing:
    """Route every net of the placed board on ``F.Cu`` / ``B.Cu``; pure (same IR + library + params + rules -> same result).

    ``ir`` is not mutated and its existing copper (``ir.pcb.tracks`` / ``vias``
    / ``zones``) is neither an obstacle nor reused: the caller decides what to
    do with the result (the PCB agent never routes a board that already has
    copper). Nets are negotiated in ``(pad count, name)`` order (module
    docstring); a net with fewer than two pads is skipped (nothing to
    connect; listed in ``stats["skipped_nets"]``). A net without a legal
    route is listed in :attr:`Routing.unrouted` with no copper at all. Tracks
    come in routing order (``stats["net_order"]``), then path order.
    ``rules`` maps net names to :class:`NetRule` (module docstring: net
    rules); without a non-empty one the result is routing.maze 0.2's.
    ``inner_layers=True`` routes a board whose layers include inner
    ``In<k>.Cu`` layers (a 4-layer board's planes) on its outer layers; by
    default such a board is refused, as 0.2 refused it.
    ``progress``, when given, is called after every negotiation iteration
    with that iteration's row of ``stats["history"]`` (for a caller that
    reports progress; it never changes the result). Raises
    :class:`CompileError` for anything that would need a guess (module
    docstring).
    """
    p, raised = effective_params(ir, params)
    eff, rule_raised = _effective_rules(ir, p, rules)
    index = {net.name: k for k, net in enumerate(ir.nets)}
    board = _Board(ir, library, p, {index[name]: r.clearance_mm for name, r in eff.items()}, inner_layers)
    clearance = {index[name]: r.clearance_mm for name, r in eff.items()}

    def clear_of(idx: int) -> float:
        return clearance.get(idx, p.clearance_mm)

    def track_of(idx: int) -> tuple[float, float]:
        r = eff.get(ir.nets[idx].name) if idx >= 0 else None
        return (r.width_mm, r.clearance_mm) if r is not None else (p.track_width_mm, p.clearance_mm)

    ordered = sorted(ir.nets, key=lambda net: (len(board.terminals[net.name]), net.name))
    skipped = [net.name for net in ordered if len(board.terminals[net.name]) < 2]
    order = [net for net in ordered if len(board.terminals[net.name]) >= 2]
    # --- the rule nets, the pairs and the profiles
    pairs: dict[str, _Pair] = {}  # virtual name -> pair
    pair_of: dict[str, _Pair] = {}  # real net name -> pair
    for name in sorted(eff):
        rule = eff[name]
        if rule.pair_partner is None or name > rule.pair_partner:
            continue
        a, b = name, rule.pair_partner
        for x in (a, b):
            if len(board.terminals[x]) != 2:
                raise CompileError(f"coupled pair {a}/{b}: net {x!r} has {len(board.terminals[x])} pad(s); a coupled pair joins two 2-pad nets")
        vname = f"pair:{a}/{b}"
        if vname in index:
            raise CompileError(f"coupled pair {a}/{b}: the IR already has a net named {vname!r}")
        net_a = ir.nets[index[a]]
        c = rule.clearance_mm
        w, s = rule.width_mm, rule.pair_spacing_mm
        rn = _RuleNet(vname, _VIRTUAL, rule, (2.0 * w + s, c), c, use_stat=False, allow_vias=False)
        vnet = Net(name=vname, pins=[*net_a.pins, *ir.nets[index[b]].pins], provenance=net_a.provenance)
        pr = _Pair(vname, a, b, index[a], index[b], rule, w, s, c, vnet, rn)
        pairs[vname] = pr
        pair_of[a] = pair_of[b] = pr
    rule_nets: dict[str, _RuleNet] = {}
    track_keys: set[_Key] = set()
    via_keys: set[float] = set()
    for net in order:
        rule = eff.get(net.name)
        if rule is None or net.name in pair_of:
            continue
        rule_nets[net.name] = _setup_rule_net(board, net.name, rule, clear_of)
        track_keys.add(rule_nets[net.name].key)
        via_keys.add(rule.clearance_mm)
        if rule_nets[net.name].nd_key is not None:
            track_keys.add(rule_nets[net.name].nd_key)
    for pr in pairs.values():
        track_keys.add(pr.rn.key)
        via_keys.add(pr.c)
        _setup_pair(board, pr, clear_of, track_of)
    key0: _Key = (p.track_width_mm, p.clearance_mm)
    neg = _Negotiation(board, p, [key0, *sorted(track_keys - {key0})], [p.clearance_mm, *sorted(via_keys - {p.clearance_mm})])
    if pairs:  # the pair's virtual net takes the place of the first of its two nets in the order
        placed: set[str] = set()
        new_order: list[Net] = []
        for net in order:
            pr = pair_of.get(net.name)
            if pr is None:
                new_order.append(net)
            elif pr.name not in placed:
                placed.add(pr.name)
                new_order.append(pr.net)
        order = new_order
    routes: dict[str, _NetRoute] = {}
    unrouted: dict[str, str] = {}

    def route_one(net: Net, strict: bool = False) -> _NetRoute | str:
        pr = pairs.get(net.name)
        if pr is not None:
            return _route_pair(neg, pr, strict, routes if strict else None)
        return _route_net(neg, net, board.net_index[net.name], board.terminals[net.name], strict=strict, rn=rule_nets.get(net.name))

    history: list[dict[str, Any]] = []
    legal = not order
    iterations = 0
    todo = list(order)
    while todo and iterations < p.max_iterations:
        iterations += 1
        for net in todo:
            old = routes.pop(net.name, None)
            if old is not None:
                neg.remove(old)
            got = route_one(net)
            if isinstance(got, str):
                if old is not None:
                    # a net rule (a length budget grown at this iteration's costs) can refuse what an earlier iteration found: the
                    # earlier route meets every rule and stays - still in conflict, so it is rerouted again or ripped up later
                    routes[net.name] = old
                    neg.add(old)
                    continue
                unrouted[net.name] = got  # unreachable even with the other nets' copper as a mere cost: a static fence
                continue
            unrouted.pop(net.name, None)
            routes[net.name] = got
            neg.add(got)
        over_t: set[int] = set()
        over_v: set[int] = set()
        conflicting: list[str] = []
        for net in order:
            route = routes.get(net.name)
            if route is None:
                continue
            ct, cv = neg.overused(route)
            if ct or cv:
                conflicting.append(net.name)
                over_t.update(ct)
                over_v.update(cv)
        row = {"iteration": iterations, "rerouted": len(todo), "overused_cells": len(over_t) + len(over_v), "conflicting": list(conflicting)}
        history.append(row)
        if progress is not None:
            progress(dict(row))
        if not conflicting:
            legal = True
            break
        for c in over_t:
            neg.hist[c] += p.history_cost
        for k in over_v:
            neg.vhist[k] += p.history_cost
        neg.pres *= p.present_growth
        neg.recompute()
        names = set(conflicting)
        todo = [net for net in order if net.name in names]
    dropped: list[str] = []
    recovered: list[str] = []
    if not legal:
        dropped = _drop_conflicts(neg, order, routes, iterations, unrouted)
        recovered = _recover(neg, order, routes, dropped, unrouted, route_one)
    if not eff:
        return _result_0_2(board, p, raised, order, routes, unrouted, skipped, iterations, legal, history, dropped, recovered)
    return _result_rules(
        ir, board, neg, p, raised, eff, rule_raised, rule_nets, pairs, order, routes, unrouted, skipped, iterations, legal, history, dropped, recovered, clear_of,
    )


def _setup_rule_net(board: _Board, name: str, rule: NetRule, clear_of: Callable[[int], float]) -> _RuleNet:
    """The net's profiles, its neck-down pads and zone, its fence and via fence (module docstring: net rules)."""
    p, n = board.p, board.n
    idx = board.net_index[name]
    w, c = rule.width_mm, rule.clearance_mm
    rn = _RuleNet(name, idx, rule, (w, c), c, budget=rule.max_length_mm, via_len=rule.via_length_mm or 0.0)
    neck = rule.neckdown_width_mm
    if neck is not None:
        nd_terms = [
            t for t in board.terminals[name]
            if any(board.rule_blocked(t.cell, layer, idx, w, c, clear_of) for layer in board.usable_layers(t, idx))
        ]
        if nd_terms:
            zone = bytearray(n)
            radius = rule.neckdown_radius_mm
            for t in nd_terms:
                for k, dist in board.cells_near_box(t.pad, radius):
                    if dist <= radius + _EPS:
                        zone[k] = 1
            rn.zone = zone
            rn.nd_key = (neck, p.clearance_mm)
            rn.nd_pads = [t.label for t in nd_terms]
    if w > p.track_width_mm or c > p.clearance_mm or (rn.zone is not None and neck > p.track_width_mm):
        rn.fence = board.rule_fence(idx, w, c, rn.zone, neck if rn.zone is not None else None, p.clearance_mm, clear_of)
    if c > p.clearance_mm:
        rn.vfence = board.rule_via_fence(idx, c, clear_of)
    return rn


def _result_0_2(
    board: _Board, p: RoutingParams, raised: dict, order: list[Net], routes: dict[str, _NetRoute], unrouted: dict[str, str], skipped: list[str],
    iterations: int, legal: bool, history: list, dropped: list[str], recovered: list[str],
) -> Routing:
    """The result of a board without rules: 0.2's emission and stats (plus ``inner_layers`` when the board has them)."""
    result = Routing(params=p, unrouted={net.name: unrouted[net.name] for net in order if net.name in unrouted})
    lengths: dict[str, float] = {}
    for net in order:
        route = routes.get(net.name)
        if route is None:
            continue
        copper = _emit(board, net, route, _provenance(net, p, route.how, iterations, legal), None)
        result.tracks.extend(copper.tracks)
        result.vias.extend(copper.vias)
        lengths[net.name] = _q(copper.length_mm())
    result.stats = _base_stats(board, result, lengths, skipped, raised, iterations, legal, history, dropped, recovered, [net.name for net in order])
    return result


def _base_stats(
    board: _Board, result: Routing, lengths: dict[str, float], skipped: list[str], raised: dict, iterations: int, legal: bool,
    history: list, dropped: list[str], recovered: list[str], net_order: list[str],
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "routed_nets": len(lengths),
        "unrouted_nets": len(result.unrouted),
        "skipped_nets": skipped,
        "net_length_mm": lengths,
        "total_length_mm": _q(sum(lengths.values())),
        "track_count": len(result.tracks),
        "via_count": len(result.vias),
        "grid": (board.nx, board.ny, board.n),
        "raised": {name: [requested, effective] for name, (requested, effective) in raised.items()},
        "iterations": iterations,
        "legal": legal,
        "history": history,
        "dropped": dropped,
        "recovered": recovered,
        "net_order": net_order,
    }
    if board.inner_layers:
        stats["inner_layers"] = list(board.inner_layers)
    return stats


def _rule_note(rn: _RuleNet, meander: dict[str, Any] | None, rn_grid: float) -> str:
    r = rn.rule
    parts = [f"net rule{f' {r.net_class}' if r.net_class else ''}: width {r.width_mm:g} mm, clearance {r.clearance_mm:g} mm"]
    if rn.nd_pads:
        parts.append(f"neck-down to {r.neckdown_width_mm:g} mm within {r.neckdown_radius_mm:g} mm (+ the one {rn_grid:g} mm grid step that leaves that zone) "
                     f"of {', '.join(rn.nd_pads)}")
    if r.max_length_mm is not None:
        parts.append(f"length budget {r.max_length_mm:g} mm (vias {r.via_length_mm:g} mm each)")
    if r.match_group is not None:
        parts.append(f"match group {r.match_group} (skew {r.max_skew_mm:g} mm)")
    if meander is not None:
        parts.append(f"meander +{meander['added_mm']:g} mm in {meander['bumps']} bump(s)")
    return ", ".join(parts)


def _result_rules(
    ir: CircuitIR, board: _Board, neg: _Negotiation, p: RoutingParams, raised: dict, eff: dict[str, NetRule], rule_raised: dict,
    rule_nets: dict[str, _RuleNet], pairs: dict[str, _Pair], order: list[Net], routes: dict[str, _NetRoute], unrouted: dict[str, str],
    skipped: list[str], iterations: int, legal: bool, history: list, dropped: list[str], recovered: list[str], clear_of: Callable[[int], float],
) -> Routing:
    """Match groups, emission, pairs, the audits and the stats of a board routed with rules (stamped :data:`ROUTER_RULES_VERSION`)."""
    version = ROUTER_RULES_VERSION
    # --- length matching on the grid (before emission: the meanders are checked against every other net's halo)
    meanders: dict[str, dict[str, Any]] = {}
    pre_meander: dict[str, _NetRoute] = {}
    group_stats: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    for name, rn in rule_nets.items():
        if rn.rule.match_group is not None:
            groups.setdefault(rn.rule.match_group, []).append(name)
    for name, r in eff.items():  # members that were skipped (< 2 pads) are group members without copper
        if r.match_group is not None and name not in rule_nets:
            groups.setdefault(r.match_group, []).append(name)
    net_of = {net.name: net for net in ir.nets}

    def measured(name: str, route: _NetRoute) -> float:
        rn = rule_nets[name]
        copper = _emit(board, net_of[name], route, _provenance(net_of[name], p, route.how, iterations, legal), rn)
        return copper.length_mm() + len(copper.vias) * rn.via_len

    for group in sorted(groups):
        members = sorted(groups[group])
        skew = float(eff[members[0]].max_skew_mm)
        info: dict[str, Any] = {"nets": members, "max_skew_mm": skew, "reasons": {}}
        routed = [m for m in members if m in routes]
        for m in members:
            if m not in routes:
                info["reasons"][m] = "no copper: " + unrouted.get(m, "fewer than 2 pads")
        before = {m: measured(m, routes[m]) for m in routed}
        after = dict(before)
        target = max(before.values()) if before else 0.0
        g = p.grid_mm
        for m in routed:
            short = target - after[m]
            if short <= skew + _TOL:
                continue
            needed = -(-(short - skew - _TOL) // (2 * g))  # the smallest whole number of grid steps up (and as many down)
            cells = int(needed)
            if 2 * g * cells > short + _TOL:
                info["reasons"][m] = f"the skew budget {skew:g} mm is narrower than the meander quantum 2 x grid = {2 * g:g} mm at this length difference"
                continue
            budget = rule_nets[m].budget
            if budget is not None and after[m] + 2 * g * cells > budget + _TOL:
                info["reasons"][m] = f"meanders of {2 * g * cells:g} mm would exceed max_length_mm {budget:g}"
                continue
            old = routes[m]
            neg.remove(old)
            got = _meander(neg, rule_nets[m], old, cells)
            if isinstance(got, str):
                neg.add(old)
                info["reasons"][m] = got
                continue
            fresh, minfo = got
            neg.add(fresh)
            routes[m] = fresh
            pre_meander[m] = old
            meanders[m] = minfo
            after[m] = measured(m, fresh)
        info["target_mm"] = _q(target)
        info["before_mm"] = {m: _q(v) for m, v in before.items()}
        info["after_mm"] = {m: _q(v) for m, v in after.items()}
        group_stats[group] = info
    # re-check: the meanders must leave every route free of over-used cells (a meander in conflict is taken back)
    if meanders:
        bad = [name for name in routes if any(neg.overused(routes[name]))]
        involved = set(bad) | {x for b in bad for x in _partners(neg, routes[b], routes, b)}
        for name in sorted(meanders):
            if name in involved:
                neg.remove(routes[name])
                routes[name] = pre_meander.pop(name)
                neg.add(routes[name])
                meanders.pop(name)
                group_stats[eff[name].match_group]["reasons"][name] = "the meander left an over-used cell and was taken back"
    # --- emission of every net but the pairs
    emitted: dict[str, _NetCopper] = {}

    def emit(net: Net) -> _NetCopper:
        route = routes[net.name]
        rn = rule_nets.get(net.name)
        prov = _provenance(
            net, p, route.how, iterations, legal, version=version, rules_active=True, rule=None if rn is None else rn.rule,
            rule_note=None if rn is None else _rule_note(rn, meanders.get(net.name), p.grid_mm),
        )
        return _emit(board, net, route, prov, rn)

    for net in order:
        if net.name in routes and net.name not in pairs:
            emitted[net.name] = emit(net)
    # --- the length budgets, measured on the emitted copper
    for name, rn in rule_nets.items():
        if rn.budget is None or name not in emitted:
            continue
        copper = emitted[name]
        total = copper.length_mm() + len(copper.vias) * rn.via_len
        if total > rn.budget + _TOL:
            emitted.pop(name)
            unrouted[name] = f"the routed copper is {total:.3f} mm, over max_length_mm {rn.budget:g}"
    partner: dict[int, tuple[int, float]] = {}
    for pr in pairs.values():
        partner[pr.ia] = (pr.ib, pr.req)
        partner[pr.ib] = (pr.ia, pr.req)

    def track_clearance(name: str, t: Track) -> float:
        rn = rule_nets.get(name)
        if rn is None:
            return clear_of(board.net_index[name])
        return p.clearance_mm if rn.nd_key is not None and t.width_mm == rn.nd_key[0] else rn.key[1]

    def build_world() -> _World:
        world = _World(board, clear_of, partner)
        for name, copper in emitted.items():
            idx = board.net_index[name]
            for t in copper.tracks:
                world.add_track(idx, t, track_clearance(name, t))
            for v in copper.vias:
                world.add_via(idx, v, clear_of(idx))
        return world

    # --- the exact audit of the meandered nets (a failure takes the meander back)
    world = build_world()
    reverted = False
    for name in sorted(meanders):
        idx = board.net_index[name]
        problem = None
        for t in emitted[name].tracks:
            problem = world.seg_problem(idx, LAYERS.index(t.layer), t.start, t.end, t.width_mm, track_clearance(name, t), f"the meandered track of {name}")
            if problem:
                break
        if problem:
            routes[name] = pre_meander.pop(name)
            meanders.pop(name)
            emitted[name] = emit(net_of[name])
            group_stats[eff[name].match_group]["reasons"][name] = f"exact clearance audit: {problem}; the meander was taken back"
            reverted = True
    if reverted:
        world = build_world()
    for group, info in group_stats.items():
        lengths = {}
        for m in info["nets"]:
            if m in emitted:
                rn = rule_nets[m]
                lengths[m] = emitted[m].length_mm() + len(emitted[m].vias) * rn.via_len
        info["after_mm"] = {m: _q(v) for m, v in lengths.items()}
        info["meanders"] = {m: meanders[m] for m in info["nets"] if m in meanders}
        info["skew_mm"] = _q(max(lengths.values()) - min(lengths.values())) if lengths else None
        info["matched"] = len(lengths) == len(info["nets"]) and info["skew_mm"] is not None and info["skew_mm"] <= info["max_skew_mm"] + _TOL
    # --- the coupled pairs: offset tracks, compensation, exact audit
    pair_stats: dict[str, dict[str, Any]] = {}
    pair_tracks: dict[str, list[Track]] = {}
    for pr in pairs.values():
        key = f"{pr.a}/{pr.b}"
        st: dict[str, Any] = {
            "nets": [pr.a, pr.b], "width_mm": pr.w, "spacing_mm": pr.s, "clearance_mm": pr.c, "centreline_width_mm": _q(pr.width), "offset_mm": _q(pr.h),
        }
        pair_stats[key] = st
        route = routes.get(pr.name)
        if route is None:
            st["reason"] = unrouted.get(pr.name, pr.reason)
            continue
        problem, tracks = _emit_pair(board, p, pr, route, world, st, iterations, legal, net_of, eff)
        if problem is not None:
            unrouted[pr.name] = problem
            st["reason"] = problem
            continue
        st["reason"] = None
        pair_tracks[pr.name] = tracks
        for t in tracks:
            world.add_track(board.net_index[t.net], t, pr.c)
    # --- assemble in routing order
    result = Routing(params=p, version=version, rules=dict(eff))
    lengths: dict[str, float] = {}
    for net in order:
        pr = pairs.get(net.name)
        if pr is not None:
            tracks = pair_tracks.get(pr.name)
            if tracks is None:
                reason = unrouted.get(pr.name, pr.reason or "not routed")
                result.unrouted[pr.a] = result.unrouted[pr.b] = reason
                continue
            result.tracks.extend(tracks)
            for x in (pr.a, pr.b):
                lengths[x] = _q(sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in tracks if t.net == x))
            continue
        copper = emitted.get(net.name)
        if copper is None:
            result.unrouted[net.name] = unrouted.get(net.name, "not routed")
            continue
        result.tracks.extend(copper.tracks)
        result.vias.extend(copper.vias)
        lengths[net.name] = _q(copper.length_mm())
    result.stats = _base_stats(board, result, lengths, skipped, raised, iterations, legal, history, dropped, recovered, [net.name for net in order])
    rule_stats: dict[str, dict[str, Any]] = {}
    for name in sorted(eff):
        r = eff[name]
        rn = rule_nets.get(name)
        copper = emitted.get(name)
        entry: dict[str, Any] = {"rule": r.derived_from_entry(), "raised": rule_raised.get(name, {}), "routed": name in lengths}
        if rn is not None:
            entry["neckdown_pads"] = list(rn.nd_pads)
            neck = 0.0
            if copper is not None and rn.nd_key is not None:
                neck = sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in copper.tracks if t.width_mm == rn.nd_key[0])
            entry["neckdown_length_mm"] = _q(neck)
            entry["length_mm"] = None if copper is None else _q(copper.length_mm() + len(copper.vias) * rn.via_len)
        elif name in _pair_names(pairs):
            entry["length_mm"] = lengths.get(name)
        rule_stats[name] = entry
    result.stats["rules"] = rule_stats
    result.stats["match_groups"] = group_stats
    result.stats["pairs"] = pair_stats
    result.stats["unrouted_nets"] = len(result.unrouted)
    return result


@dataclass(frozen=True, slots=True)
class _PairSeg:
    """A pair track segment before it is a :class:`~ai_eda.ir.Track` (what :mod:`ai_eda.tools.routing.coupling` reads)."""

    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float


def _pair_names(pairs: Mapping[str, _Pair]) -> set[str]:
    """The real net names of the pairs."""
    return {x for pr in pairs.values() for x in (pr.a, pr.b)}


def _emit_pair(
    board: _Board, p: RoutingParams, pr: _Pair, route: _NetRoute, world: _World, st: dict[str, Any], iterations: int, legal: bool,
    net_of: Mapping[str, Net], eff: Mapping[str, NetRule],
) -> tuple[str | None, list[Track]]:
    """The pair's two tracks (breakouts + offset centreline + compensation), audited exactly; ``(problem, tracks)``."""
    data = route.steps[0][1]
    layer, _, la, lb, sigma = data
    pts_a, pts_b, straight = _pair_polylines(board, pr, data)
    lengths = {pr.a: _polyline_length(pts_a), pr.b: _polyline_length(pts_b)}
    coupled = {pr.a: _polyline_length(pts_a[1:-1]), pr.b: _polyline_length(pts_b[1:-1])}
    st["layer"] = LAYERS[layer]
    st["launch"] = []
    for lc, (t_a, t_b) in ((la, pr.ends[0]), (lb, pr.ends[1])):
        seg_len = {pr.a: math.hypot(lc.seg_a[1][0] - lc.seg_a[0][0], lc.seg_a[1][1] - lc.seg_a[0][1]),
                   pr.b: math.hypot(lc.seg_b[1][0] - lc.seg_b[0][0], lc.seg_b[1][1] - lc.seg_b[0][1])}
        st["launch"].append({
            "pads": [t_a.label, t_b.label], "at": board.pos(lc.cell), "direction": _DIR_NAMES[lc.d],
            "breakout_mm": {x: _q(v) for x, v in seg_len.items()},
        })
    st["coupled_mm"] = {x: _q(v) for x, v in coupled.items()}
    st["breakouts_mm"] = {x: _q(lengths[x] - coupled[x]) for x in (pr.a, pr.b)}
    st["skew_before_mm"] = _q(abs(lengths[pr.a] - lengths[pr.b]))
    st["compensation"] = None
    diff = lengths[pr.a] - lengths[pr.b]
    tol = pr.rule.pair_max_skew_mm if pr.rule.pair_max_skew_mm is not None else _TOL
    polys = {pr.a: pts_a, pr.b: pts_b}
    sides = {pr.a: sigma, pr.b: -sigma}
    idx = {pr.a: pr.ia, pr.b: pr.ib}
    if abs(diff) > tol:
        short = pr.b if diff > 0 else pr.a
        other = pr.a if short == pr.b else pr.b
        placed = _compensate(board, p, pr, layer, polys[short], straight[0 if short == pr.a else 1], sides[short], abs(diff), idx[short], world, polys[other])
        if isinstance(placed, str):
            st["compensation"] = {"net": short, "needed_mm": _q(abs(diff)), "reason": placed}
        else:
            polys[short], comp = placed
            comp["net"] = short
            st["compensation"] = comp
    lengths = {x: _polyline_length(polys[x]) for x in (pr.a, pr.b)}
    st["length_mm"] = {x: _q(v) for x, v in lengths.items()}
    st["skew_mm"] = _q(abs(lengths[pr.a] - lengths[pr.b]))
    if pr.rule.max_length_mm is not None:
        over = [x for x in (pr.a, pr.b) if lengths[x] > pr.rule.max_length_mm + _TOL]
        if over:
            return f"coupled pair {pr.a}/{pr.b}: {over[0]} is {lengths[over[0]]:.3f} mm, over max_length_mm {pr.rule.max_length_mm:g}", []
    # the uncoupled length as si.diff measures it (ai_eda.tools.routing.coupling): breakouts, mitred corners, compensation bumps
    segs = {x: [_PairSeg(LAYERS[layer], a, b, pr.w) for a, b in zip(polys[x], polys[x][1:]) if a != b] for x in (pr.a, pr.b)}
    _, unc_a, unc_b, _ = uncoupled_lengths(segs[pr.a], segs[pr.b])
    st["uncoupled_mm"] = {pr.a: _q(unc_a), pr.b: _q(unc_b)}
    budget = pr.rule.pair_uncoupled_max_mm
    if budget is not None:
        over = [(x, u) for x, u in ((pr.a, unc_a), (pr.b, unc_b)) if u > budget + _TOL]
        if over:
            x, u = over[0]
            return (f"coupled pair {pr.a}/{pr.b}: {x} would have {u:.3f} mm of uncoupled copper (breakouts, mitred corners, compensation), "
                    f"over pair_uncoupled_max_mm {budget:g}"), []
    # the exact audit: every segment of both tracks against the other copper, the pads, the edge and the partner
    pad_centres = {pt for end in pr.ends for t in end for pt in [(_q(t.pad.cx), _q(t.pad.cy))]}
    for x in (pr.a, pr.b):
        partner_pts = polys[pr.b if x == pr.a else pr.a]
        for a, b in zip(polys[x], polys[x][1:]):
            if a == b:
                continue
            problem = world.seg_problem(idx[x], layer, a, b, pr.w, pr.c, f"the track of {x}")
            if problem is None:
                for c, d in zip(partner_pts, partner_pts[1:]):
                    if c != d and _seg_seg(a, b, c, d) - pr.w < pr.req - _TOL:
                        problem = f"the track of {x} comes {max(_seg_seg(a, b, c, d) - pr.w, 0.0):.4f} mm from its partner (needs {pr.req:g} mm)"
                        break
            if problem is None:
                for pt in (a, b):
                    if pt not in pad_centres and board.edge_distance_xy(*pt) < p.edge_clearance_mm + pr.w / 2.0 - _TOL:
                        problem = f"the track of {x} comes within edge_clearance + width/2 of the outline"
            if problem is not None:
                return f"coupled pair {pr.a}/{pr.b}: exact clearance audit: {problem}", []
    refs = sorted({pin.component_ref for pin in pr.net.pins})
    story = (
        f"coupled differential pair {pr.a}/{pr.b}: centreline of width 2w + s = {_q(pr.width):g} mm negotiated as one net, "
        f"{iterations} iteration(s){'' if legal else ', kept after the conflicting nets were ripped up'}"
        + (", routed against the legal copper after the cap" if route.how == "recovered" else "")
        + f", offset by +/-(w + s)/2 = {_q(pr.h):g} mm with mitred corners; straight uncoupled breakouts at both ends"
    )
    tracks: list[Track] = []
    for x in (pr.a, pr.b):
        comp = st["compensation"]
        note = f"pair rule{f' {pr.rule.net_class}' if pr.rule.net_class else ''}: width {pr.w:g} mm, spacing {pr.s:g} mm"
        if comp is not None and comp.get("net") == x and "bumps" in comp:
            note += f", {comp['bumps']} compensation bump(s) of {comp['amplitude_mm']:g} mm"
        prov = _provenance(
            net_of[x], p, route.how, iterations, legal, version=ROUTER_RULES_VERSION, rules_active=True, rule=eff[x],
            extra_refs=refs, story=story, rule_note=note,
        )
        for a, b in zip(polys[x], polys[x][1:]):
            if a != b:
                tracks.append(Track(net=x, layer=LAYERS[layer], start=a, end=b, width_mm=pr.w, provenance=prov))
    return None, tracks


def _compensate(
    board: _Board, p: RoutingParams, pr: _Pair, layer: int, pts: list[tuple[float, float]], straight: list[int], side: int, needed: float,
    net: int, world: _World, partner_pts: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], dict[str, Any]] | str:
    """Rectangular bumps outward on the shorter track adding ``needed`` mm, on its longest coupled straight segment that holds them."""
    bumps = max(1, math.ceil(needed / (2.0 * p.pair_bump_amplitude_mm) - _EPS))
    height = needed / (2.0 * bumps)
    b = p.pair_bump_length_mm
    span = (2 * bumps - 1) * b
    margin = b + pr.h
    # pts[0] is the pad; the offset polyline starts at pts[1]: a straight coupled segment starts at pts[1 + s] for s in straight
    candidates = []
    for s in straight:
        t = 1 + s
        (ax, ay), (bx, by) = pts[t], pts[t + 1]
        length = math.hypot(bx - ax, by - ay)
        if length >= span + 2 * margin - _EPS and (ax == bx or ay == by):
            candidates.append((-length, t))
    last = "no straight coupled segment is long enough"
    for _, t in sorted(candidates):
        (ax, ay), (bx, by) = pts[t], pts[t + 1]
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        vx, vy = side * uy, side * -ux  # outward: the track's own side of the centreline (away from the partner)
        t0 = (length - span) / 2.0
        new: list[tuple[float, float]] = []
        for j in range(bumps):
            s0 = t0 + 2 * j * b
            p1 = (_q(ax + s0 * ux), _q(ay + s0 * uy))
            p2 = (_q(p1[0] + height * vx), _q(p1[1] + height * vy))
            p4 = (_q(ax + (s0 + b) * ux), _q(ay + (s0 + b) * uy))
            p3 = (_q(p4[0] + height * vx), _q(p4[1] + height * vy))
            new += [p1, p2, p3, p4]
        problem = None
        for j in range(bumps):
            p1, p2, p3, p4 = new[4 * j: 4 * j + 4]
            for a, c in ((p1, p2), (p2, p3), (p3, p4)):
                problem = world.seg_problem(net, layer, a, c, pr.w, pr.c, "a compensation bump")
                if problem is None:
                    for e, f in zip(partner_pts, partner_pts[1:]):
                        if e != f and _seg_seg(a, c, e, f) - pr.w < pr.req - _TOL:
                            problem = "a compensation bump comes too near the partner"
                            break
                if problem is None and board.edge_distance_xy(*c) < p.edge_clearance_mm + pr.w / 2.0 - _TOL:
                    problem = "a compensation bump comes within edge_clearance + width/2 of the outline"
                if problem:
                    break
            if problem:
                break
        if problem:
            last = problem
            continue
        out = pts[: t + 1] + new + pts[t + 1:]
        return out, {"bumps": bumps, "amplitude_mm": _q(height), "length_mm": b, "at": [pts[t], pts[t + 1]], "added_mm": _q(2 * bumps * height)}
    return f"the length difference {needed:.3f} mm was not compensated: {last}"


def _partners(neg: _Negotiation, route: _NetRoute, routes: dict[str, _NetRoute], name: str) -> list[str]:
    """The other nets whose halo holds some of ``route``'s copper (sorted by name)."""
    out = []
    for other, r in routes.items():
        if other == name:
            continue
        if any(not cells.isdisjoint(r.cover_t[key]) for key, cells in route.keyed) or not r.cover_v[route.via_key].isdisjoint(route.via_cells):
            out.append(other)
    return sorted(out)


def _drop_conflicts(neg: _Negotiation, order: list[Net], routes: dict[str, _NetRoute], iterations: int, unrouted: dict[str, str]) -> list[str]:
    """Rip up conflicting nets (most partners first, then fewer pads, then name) until the rest is legal; the dropped names in order."""
    pads = {net.name: len(net.pins) for net in order}
    dropped: list[str] = []
    while True:
        bad = [net.name for net in order if net.name in routes and any(neg.overused(routes[net.name]))]
        if not bad:
            return dropped
        partners = {name: _partners(neg, routes[name], routes, name) for name in bad}
        victim = min(bad, key=lambda name: (-len(partners[name]), pads[name], name))
        neg.remove(routes.pop(victim))
        dropped.append(victim)
        unrouted[victim] = (
            f"no legal route after {iterations} negotiation iteration(s): its copper still broke the clearance of "
            f"{', '.join(partners[victim])}, and a route against the legal copper as an obstacle was not found"
        )


def _recover(
    neg: _Negotiation, order: list[Net], routes: dict[str, _NetRoute], dropped: list[str], unrouted: dict[str, str],
    route_one: Callable[..., _NetRoute | str] | None = None,
) -> list[str]:
    """Route each dropped net once more with every other net's halo as an obstacle, terminal cells included (legal by construction); the recovered names."""
    board = neg.board
    recovered: list[str] = []
    names = set(dropped)
    for net in order:
        if net.name not in names:
            continue
        if route_one is None:
            got = _route_net(neg, net, board.net_index[net.name], board.terminals[net.name], strict=True)
        else:
            got = route_one(net, True)
        if isinstance(got, str):
            continue
        got.how = "recovered"
        routes[net.name] = got
        neg.add(got)
        unrouted.pop(net.name, None)
        recovered.append(net.name)
    return recovered
