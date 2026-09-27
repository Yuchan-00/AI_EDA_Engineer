"""Deterministic two-layer grid maze router with negotiated congestion (pure, no I/O beyond the KiCad library).

Invariant: every track and via produced here is a function of the IR's
placements, the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: pad positions, sizes,
layers - never model memory) and the :class:`RoutingParams`. Nothing is
estimated, nothing is guessed: a component without a placement or footprint,
a footprint that is not on disk, a net pin without a pad, an inner copper
layer or a pad too small for the grid raises
:class:`~ai_eda.errors.CompileError` instead of getting a default.

What this is: a Lee / A* maze router on a square grid over the board
outline, ``F.Cu`` and ``B.Cu`` only, with vias between them, one track width
for every net, inside a PathFinder-style negotiated-congestion loop
(rip-up and reroute):

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

What this is not: a DRC. The clearances above are the router's own
parameters; whether the board is valid is decided by ``kicad-cli pcb drc``
on the compiled board (:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), and
the ``pcb.routing.*`` validators only check the IR geometry against the
limits recorded in ``ir.pcb.manufacturing``. :func:`effective_params` raises
the width / clearance / via sizes / edge clearance to those limits when they
exist (whatever their provenance - a limit is a limit; the capability check
judges grounding) and :attr:`Routing.stats` says which were raised.

Traceability: every :class:`~ai_eda.ir.Track` / :class:`~ai_eda.ir.Via`
carries ``derived`` provenance naming this router (:data:`ROUTER_ID` /
:data:`ROUTER_VERSION`), the net, the placements of the net's components and
every parameter in ``derived_from``; the note names the iteration count and
how the net's route was obtained. ``Provenance.inputs`` stays empty: it is
the calculator role map, and a track is not a calculator output.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Net, Provenance, ProvenanceKind, Track, Via
from ai_eda.tools.kicad.geometry import _q, pad_angle, pad_center, pad_layers
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, Pad

__all__ = [
    "ROUTER_ID",
    "ROUTER_VERSION",
    "LAYERS",
    "BLOCKED",
    "CONVEX_PAD_SHAPES",
    "FINE_PITCH_MM",
    "FINE_RULES",
    "RoutingParams",
    "Routing",
    "effective_params",
    "finest_pad_pitch",
    "route_board",
]

#: provenance ``tool`` / ``tool_version`` stamped on every track and via
ROUTER_ID = "routing.maze"
ROUTER_VERSION = "0.2"
#: the only copper layers this router knows (index 0 / 1 in the owner maps)
LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")
#: owner-map value of a cell no net may use
BLOCKED = -1
#: flat static-map value of a cell no pad or edge claims (the owner maps say ``None``)
_FREE = -2

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
    margin around a net's terminals (doubled on failure).

    ``rules`` / ``pad_pitch_mm`` / ``pitch_footprint`` say why the values
    were chosen when :meth:`for_board` picked the fine rules (``None`` for
    values given directly or the defaults); they change nothing in the search
    and are appended to :meth:`derived_from_entry` only when set.
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

    def check(self) -> None:
        """Refuse parameters that make no sense (:class:`CompileError`)."""
        for name in ("grid_mm", "track_width_mm", "via_drill_mm", "via_diameter_mm", "base_cost", "window_mm"):
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

    def derived_from_entry(self) -> str:
        """The ``derived_from`` entry that records every parameter (and, when set, the rule set and the pitch that chose it)."""
        entry = (
            f"params:grid={self.grid_mm},width={self.track_width_mm},clearance={self.clearance_mm},"
            f"via={self.via_diameter_mm}/{self.via_drill_mm},edge={self.edge_clearance_mm},"
            f"via_cost={self.via_cost},bend_cost={self.bend_cost},base_cost={self.base_cost},"
            f"history_cost={self.history_cost},present_cost={self.present_cost},present_growth={self.present_growth},"
            f"max_iterations={self.max_iterations},window={self.window_mm}"
        )
        if self.rules is not None:
            entry += f",rules={self.rules},pad_pitch={self.pad_pitch_mm},pitch_footprint={self.pitch_footprint}"
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
    """What :func:`route_board` produced. ``unrouted`` maps a net name to why it has no copper."""

    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    unrouted: dict[str, str] = field(default_factory=dict)
    params: RoutingParams = field(default_factory=RoutingParams)
    #: routed_nets, unrouted_nets, skipped_nets (< 2 pads), net_length_mm, total_length_mm, track_count, via_count,
    #: grid (nx, ny, cells per layer), raised (parameter -> [requested, effective]), iterations, legal (the negotiation
    #: ended without an over-used cell), history (per iteration: rerouted nets, over-used cells, conflicting nets),
    #: dropped (nets ripped up after the cap), recovered (dropped nets routed again against the legal copper), net_order
    stats: dict[str, Any] = field(default_factory=dict)


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
    """The grid, the static owner maps and the terminals of one IR (built once per :func:`route_board`)."""

    def __init__(self, ir: CircuitIR, library: KicadLibrary, p: RoutingParams) -> None:
        if ir.pcb is None:
            raise CompileError("cannot route: ir.pcb is None")
        if ir.pcb.outline is None:
            raise CompileError("cannot route: ir.pcb.outline is None; the grid needs a board outline")
        names = [layer.name for layer in ir.pcb.layers]
        extra = [n for n in names if n not in LAYERS]
        if extra:
            raise CompileError(f"cannot route: this router knows only {list(LAYERS)}, ir.pcb.layers also has {extra}")
        missing = [n for n in LAYERS if n not in names]
        if missing:
            raise CompileError(f"cannot route: ir.pcb.layers lacks {missing}")
        self.p = p
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
        self.net_index = {net.name: k for k, net in enumerate(ir.nets)}
        self.terminals: dict[str, list[_Terminal]] = {net.name: [] for net in ir.nets}
        self._load_pads(ir, library)
        #: the static owner maps as one flat list over ``layer * n + k`` with :data:`_FREE` for free cells (the search reads this)
        self.stat: list[int] = [_FREE if v is None else v for v in self.owner[0]] + [_FREE if v is None else v for v in self.owner[1]]
        #: per cell, lazily: BLOCKED / _FREE / the one net whose pads the via's keep-out touches (see :meth:`via_static_at`)
        self.via_static: list[int | None] = [None] * self.n

    # --- coordinates ---------------------------------------------------------

    def pos(self, k: int) -> tuple[float, float]:
        j, i = divmod(k, self.nx)
        return _q(self.ox + i * self.p.grid_mm), _q(self.oy + j * self.p.grid_mm)

    def _edge_distance(self, k: int) -> float:
        j, i = divmod(k, self.nx)
        x, y = self.ox + i * self.p.grid_mm, self.oy + j * self.p.grid_mm
        return min(x - self.ox, self.ox + self.w - x, y - self.oy, self.oy + self.h - y)

    def _nearest_cell(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.ox) / self.p.grid_mm), round((y - self.oy) / self.p.grid_mm)

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
        """Own every cell within ``pad_radius`` of the box for ``geom.net`` (BLOCKED when another net has it)."""
        r = self.pad_radius
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
        """
        cached = self.via_static[k]
        if cached is not None:
            return cached
        value = self._via_static(k)
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


# --------------------------------------------------------------------------- one net's route


@dataclass(slots=True)
class _NetRoute:
    """The route of one net: the ordered steps (stubs and paths), its cells, and the halo it covers."""

    steps: list[tuple[str, Any, int]] = field(default_factory=list)  # ("stub", terminal, layer) | ("path", path, 0)
    path_cells: set[int] = field(default_factory=set)  # layer * n + k
    via_cells: list[int] = field(default_factory=list)  # k, in path order
    cover_t: set[int] = field(default_factory=set)  # track plane: layer * n + k
    cover_v: set[int] = field(default_factory=set)  # via plane: k
    how: str = "negotiated"  # "negotiated" or "recovered" (routed again against the legal copper after the cap)


class _Negotiation:
    """The present / history costs and the halo counts of every routed net (PathFinder state)."""

    def __init__(self, board: _Board, p: RoutingParams) -> None:
        self.board = board
        self.p = p
        n = board.n
        self.halo = [0] * (2 * n)
        self.vhalo = [0] * n
        self.hist = [0.0] * (2 * n)
        self.vhist = [0.0] * n
        self.pres = p.present_cost
        self.ncost = [p.base_cost] * (2 * n)
        self.vcost = [p.via_cost] * n
        #: the search's per-state best cost and predecessor (``state = (layer * n + k) * 5 + direction``), reset after every search
        self.g_best = [_INF] * (2 * n * _STATES_PER_CELL)
        self.prev = [-1] * (2 * n * _STATES_PER_CELL)

    # --- costs -----------------------------------------------------------------

    def _node(self, c: int) -> float:
        return self.p.base_cost * (1.0 + self.hist[c]) * (1.0 + self.pres * self.halo[c])

    def _via(self, k: int) -> float:
        p = self.p
        return (p.via_cost + p.base_cost) * (1.0 + self.vhist[k]) * (1.0 + self.pres * self.vhalo[k]) - p.base_cost

    def recompute(self) -> None:
        """Every node / via cost from the current present factor, history and counts (after the factor or the history changed)."""
        base, pres = self.p.base_cost, self.pres
        self.ncost = [base * (1.0 + h) * (1.0 + pres * o) for h, o in zip(self.hist, self.halo)]
        vb = self.p.via_cost + base
        self.vcost = [vb * (1.0 + h) * (1.0 + pres * o) - base for h, o in zip(self.vhist, self.vhalo)]

    # --- occupancy -------------------------------------------------------------

    def add(self, route: _NetRoute) -> None:
        halo, ncost = self.halo, self.ncost
        for c in route.cover_t:
            halo[c] += 1
            ncost[c] = self._node(c)
        vhalo, vcost = self.vhalo, self.vcost
        for k in route.cover_v:
            vhalo[k] += 1
            vcost[k] = self._via(k)

    def remove(self, route: _NetRoute) -> None:
        halo, ncost = self.halo, self.ncost
        for c in route.cover_t:
            halo[c] -= 1
            ncost[c] = self._node(c)
        vhalo, vcost = self.vhalo, self.vcost
        for k in route.cover_v:
            vhalo[k] -= 1
            vcost[k] = self._via(k)

    def overused(self, route: _NetRoute) -> tuple[list[int], list[int]]:
        """``(track cells, via cells)`` of ``route`` that lie inside another net's halo (its own halo always covers them)."""
        halo, vhalo = self.halo, self.vhalo
        return [c for c in route.path_cells if halo[c] > 1], [k for k in route.via_cells if vhalo[k] > 1]


def _cover(board: _Board, route: _NetRoute) -> None:
    """Fill ``route.cover_t`` / ``cover_v`` from its path and via cells (module docstring: the halo radii)."""
    n = board.n
    cover_t: set[int] = set()
    cover_v: set[int] = set()
    for c in route.path_cells:
        layer_base = n if c >= n else 0
        k = c - layer_base
        board.halo_cells(board.track_halo, k, layer_base, cover_t)
        board.halo_cells(board.via_track_halo, k, 0, cover_v)
    for k in set(route.via_cells):
        board.halo_cells(board.via_track_halo, k, 0, cover_t)
        board.halo_cells(board.via_track_halo, k, n, cover_t)
        board.halo_cells(board.via_via_halo, k, 0, cover_v)
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
    each one (only the states this search touched).
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


def _route_net(neg: _Negotiation, net: Net, idx: int, terms: list[_Terminal], strict: bool = False) -> _NetRoute | str:
    """Grow the net's Steiner tree (module docstring) at the current costs; the reason when a terminal is unreachable.

    Terminals are tracked by their position in ``terms``, never by label: a
    footprint may repeat a pad number (a switch's paired pins), and each such
    pad is a terminal of its own that gets its own path end and stub.
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
    if strict:
        # against legal copper a terminal cell inside another net's halo is no seed and no target either: the search checks the
        # halo only on the cells it steps into, and a seed is never stepped into (negotiated mode counts it as a path cell instead)
        usable = [tuple(layer for layer in layers if not neg.halo[layer * n + t.cell]) for t, layers in zip(terms, usable)]
        fenced = [t for t, layers in zip(terms, usable) if not layers]
        if fenced:
            x, y = board.pos(fenced[0].cell)
            return f"{fenced[0].label} terminal cell ({x:g}, {y:g}) is inside another net's clearance halo on every copper layer of the pad"
    # the start: the terminal nearest the centroid of the pad centres (ties: natural ref order)
    mx = sum(t.pad.cx for t in terms) / len(terms)
    my = sum(t.pad.cy for t in terms) / len(terms)
    first = min(range(len(terms)), key=lambda q: math.hypot(terms[q].pad.cx - mx, terms[q].pad.cy - my))
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
        while True:
            window = (max(0, box[0] - grown), max(0, box[1] - grown), min(board.nx - 1, box[2] + grown), min(board.ny - 1, box[3] + grown))
            path = _search(neg, idx, tree, targets, ij[target], window, strict)
            if path is not None or window == full:
                break
            grown *= 2
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
        stub(target, path[-1] // n)
        remaining.remove(target)
        grow(path)
        attach(target)
    _cover(board, route)
    return route


# --------------------------------------------------------------------------- emission


def _provenance(net: Net, p: RoutingParams, how: str, iterations: int, legal: bool) -> Provenance:
    refs = sorted({pin.component_ref for pin in net.pins})
    if how == "recovered":
        story = f"routed against the legal copper as an obstacle after {iterations} negotiation iteration(s) left it in conflict"
    else:
        story = f"negotiated-congestion route, {iterations} iteration(s)" + ("" if legal else ", kept after the conflicting nets were ripped up")
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=ROUTER_ID,
        tool_version=ROUTER_VERSION,
        derived_from=[f"net:{net.name}", *(f"placement:{r}" for r in refs), p.derived_from_entry()],
        note=f"grid maze route on F.Cu/B.Cu ({story}); validity is decided by kicad-cli DRC (pcb.routing checks the IR geometry only)",
    )


class _NetCopper:
    """The tracks and vias of one routed net, built from its recorded steps."""

    def __init__(self, board: _Board, net: Net, prov: Provenance) -> None:
        self.board = board
        self.net = net
        self.prov = prov
        self.p = board.p
        self.tracks: list[Track] = []
        self.vias: list[Via] = []

    def _track(self, layer: int, a: int, b: int) -> None:
        start, end = self.board.pos(a), self.board.pos(b)
        if start == end:
            return
        self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=self.p.track_width_mm, provenance=self.prov))

    def add_path(self, path: list[int]) -> None:
        """Merge the unit steps of ``path`` (``layer * n + cell`` ids) into tracks and vias."""
        n, nx = self.board.n, self.board.nx
        run_start: int | None = None
        run_dir: tuple[int, int] | None = None
        for a, b in zip(path, path[1:]):
            la, ka = divmod(a, n)
            lb, kb = divmod(b, n)
            if la != lb:
                if run_start is not None:
                    self._track(la, run_start, ka)
                    run_start = run_dir = None
                x, y = self.board.pos(ka)
                self.vias.append(
                    Via(net=self.net.name, x_mm=x, y_mm=y, drill_mm=self.p.via_drill_mm, diameter_mm=self.p.via_diameter_mm, layers=LAYERS, provenance=self.prov)
                )
                continue
            step = (kb % nx - ka % nx, kb // nx - ka // nx)
            if run_start is None:
                run_start, run_dir = ka, step
            elif step != run_dir:
                self._track(la, run_start, ka)
                run_start, run_dir = ka, step
        if run_start is not None:
            self._track(path[-1] // n, run_start, path[-1] % n)

    def add_stub(self, terminal: _Terminal, layer: int) -> None:
        """One segment from the terminal cell to the exact pad centre (nothing when the cell is the centre)."""
        start = self.board.pos(terminal.cell)
        end = (_q(terminal.pad.cx), _q(terminal.pad.cy))
        if start != end:
            self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=self.p.track_width_mm, provenance=self.prov))

    def length_mm(self) -> float:
        return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in self.tracks)


# --------------------------------------------------------------------------- the router


def route_board(
    ir: CircuitIR, library: KicadLibrary, params: RoutingParams | None = None, *, progress: Callable[[dict[str, Any]], None] | None = None,
) -> Routing:
    """Route every net of the placed board on ``F.Cu`` / ``B.Cu``; pure (same IR + library + params -> same result).

    ``ir`` is not mutated and its existing copper (``ir.pcb.tracks`` / ``vias``
    / ``zones``) is neither an obstacle nor reused: the caller decides what to
    do with the result (the PCB agent never routes a board that already has
    copper). Nets are negotiated in ``(pad count, name)`` order (module
    docstring); a net with fewer than two pads is skipped (nothing to
    connect; listed in ``stats["skipped_nets"]``). A net without a legal
    route is listed in :attr:`Routing.unrouted` with no copper at all. Tracks
    come in routing order (``stats["net_order"]``), then path order.
    ``progress``, when given, is called after every negotiation iteration
    with that iteration's row of ``stats["history"]`` (for a caller that
    reports progress; it never changes the result). Raises
    :class:`CompileError` for anything that would need a guess (module
    docstring).
    """
    p, raised = effective_params(ir, params)
    board = _Board(ir, library, p)
    ordered = sorted(ir.nets, key=lambda net: (len(board.terminals[net.name]), net.name))
    skipped = [net.name for net in ordered if len(board.terminals[net.name]) < 2]
    order = [net for net in ordered if len(board.terminals[net.name]) >= 2]
    neg = _Negotiation(board, p)
    routes: dict[str, _NetRoute] = {}
    unrouted: dict[str, str] = {}
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
            got = _route_net(neg, net, board.net_index[net.name], board.terminals[net.name])
            if isinstance(got, str):  # unreachable even with the other nets' copper as a mere cost: a static fence
                unrouted[net.name] = got
                continue
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
        recovered = _recover(neg, order, routes, dropped, unrouted)
    result = Routing(params=p, unrouted={net.name: unrouted[net.name] for net in order if net.name in unrouted})
    lengths: dict[str, float] = {}
    for net in order:
        route = routes.get(net.name)
        if route is None:
            continue
        copper = _NetCopper(board, net, _provenance(net, p, route.how, iterations, legal))
        for kind, item, layer in route.steps:
            if kind == "stub":
                copper.add_stub(item, layer)
            else:
                copper.add_path(item)
        result.tracks.extend(copper.tracks)
        result.vias.extend(copper.vias)
        lengths[net.name] = _q(copper.length_mm())
    result.stats = {
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
        "net_order": [net.name for net in order],
    }
    return result


def _partners(neg: _Negotiation, route: _NetRoute, routes: dict[str, _NetRoute], name: str) -> list[str]:
    """The other nets whose halo holds some of ``route``'s copper (sorted by name)."""
    out = []
    for other, r in routes.items():
        if other == name:
            continue
        if not route.path_cells.isdisjoint(r.cover_t) or not r.cover_v.isdisjoint(route.via_cells):
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


def _recover(neg: _Negotiation, order: list[Net], routes: dict[str, _NetRoute], dropped: list[str], unrouted: dict[str, str]) -> list[str]:
    """Route each dropped net once more with every other net's halo as an obstacle, terminal cells included (legal by construction); the recovered names."""
    board = neg.board
    recovered: list[str] = []
    names = set(dropped)
    for net in order:
        if net.name not in names:
            continue
        got = _route_net(neg, net, board.net_index[net.name], board.terminals[net.name], strict=True)
        if isinstance(got, str):
            continue
        got.how = "recovered"
        routes[net.name] = got
        neg.add(got)
        unrouted.pop(net.name, None)
        recovered.append(net.name)
    return recovered
