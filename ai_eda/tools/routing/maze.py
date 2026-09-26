"""Deterministic two-layer grid maze router (pure, no I/O beyond the KiCad library).

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
for every net, nets routed one after another (no rip-up). Each net grows a
tree from its first terminal: an A* search (4-neighbour steps cost 1, a
change of direction adds ``bend_cost``, a layer change adds ``via_cost``;
Manhattan heuristic to the nearest remaining terminal; heap ties broken by a
monotonically increasing counter, so the result is a pure function of its
inputs) from every cell of the tree to the nearest remaining terminal, until
all terminals hang on the tree. A net whose terminal cannot be reached is
recorded in :attr:`Routing.unrouted` with the reason and *all* of its copper
is discarded - a half-routed net is never emitted - and routing continues
with the next net. Nets are routed in ``(pad count, name)`` order; when that
order leaves a net unrouted, one second pass routes the board again from
scratch with the starved nets first (no rip-up: a whole new attempt in a
different order, taken only when it leaves fewer nets unrouted, so a board
that routes in one pass is unchanged; ``stats["passes"]`` / ``stats["net_order"]``
say what happened).

Obstacle model (board frame, mm, Y down; every emitted coordinate is rounded
with :func:`ai_eda.tools.kicad.geometry._q`):

* an *owner map* per layer says for every grid cell which net may use it
  (free, one net, or :data:`BLOCKED`). A cell closer than
  ``clearance + width/2 + grid/2`` to a pad's box (the box is exact for pads
  at multiples of 90 degrees and the circumscribed square otherwise, the
  ``grid/2`` covers a box edge that falls between two grid columns) is owned
  by the pad's net; a cell claimed by two nets, or near a pad without a net
  (unnumbered pads, NPTH holes, pins no net names), is BLOCKED. A cell closer
  than ``edge_clearance + width/2`` to the outline is BLOCKED on both layers.
* a routed path is marked with radius ``width + clearance + grid/2`` around
  every path cell on its layer and a via with radius
  ``via_diameter/2 + clearance + width/2`` on both layers, so the next net
  keeps ``clearance`` from this copper. Because the marks of a net are
  neither an obstacle nor a shortcut for the net itself, they are written
  once the whole net has routed (which is what lets a failed net be
  discarded without a trace).
* a via may sit where every cell within ``via_diameter/2 + clearance +
  width/2`` on *both* layers is free or the net's own, at least
  ``edge_clearance + via_diameter/2`` from the outline, and where its copper
  disc (radius ``via_diameter/2``) overlaps no pad box at all - the net's own
  pads included: a via is drilled next to a pad, never through it
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
every parameter in ``derived_from``. ``Provenance.inputs`` stays empty: it is
the calculator role map, and a track is not a calculator output.
"""

from __future__ import annotations

import heapq
import math
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
    "RoutingParams",
    "Routing",
    "effective_params",
    "route_board",
]

#: provenance ``tool`` / ``tool_version`` stamped on every track and via
ROUTER_ID = "routing.maze"
ROUTER_VERSION = "0.1"
#: the only copper layers this router knows (index 0 / 1 in the owner maps)
LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")
#: owner-map value of a cell no net may use
BLOCKED = -1

#: ``(track parameter, ir.pcb.manufacturing limit)`` pairs a fab minimum can raise
_FAB_MINIMUMS: tuple[tuple[str, str], ...] = (
    ("track_width_mm", "min_track_width_mm"),
    ("clearance_mm", "min_clearance_mm"),
    ("via_drill_mm", "min_via_drill_mm"),
    ("via_diameter_mm", "min_via_diameter_mm"),
)
#: pad shapes whose copper lies inside the ``(size)`` box (convex): the only ones this router models
CONVEX_PAD_SHAPES = frozenset({"circle", "rect", "oval", "roundrect"})
#: 4-neighbour steps: east, south, west, north (board frame, Y down)
_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
#: direction index of a state that has no direction yet (a seed or the far end of a via)
_NO_DIR = 4
_STATES_PER_CELL = 5
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class RoutingParams:
    """The router's knobs (mm, except the two costs in grid steps). Every one lands in the provenance."""

    grid_mm: float = 0.25
    track_width_mm: float = 0.4
    clearance_mm: float = 0.25
    via_diameter_mm: float = 0.8
    via_drill_mm: float = 0.4
    edge_clearance_mm: float = 0.3
    via_cost: float = 12.0
    bend_cost: float = 0.6

    def check(self) -> None:
        """Refuse parameters that make no sense (:class:`CompileError`)."""
        for name in ("grid_mm", "track_width_mm", "via_drill_mm", "via_diameter_mm"):
            if not (getattr(self, name) > 0):
                raise CompileError(f"routing parameter {name} must be > 0 (got {getattr(self, name)!r})")
        for name in ("clearance_mm", "edge_clearance_mm", "via_cost", "bend_cost"):
            if not (getattr(self, name) >= 0):
                raise CompileError(f"routing parameter {name} must be >= 0 (got {getattr(self, name)!r})")
        if self.via_diameter_mm <= self.via_drill_mm:
            raise CompileError(f"via diameter {self.via_diameter_mm} mm must exceed the via drill {self.via_drill_mm} mm")

    def derived_from_entry(self) -> str:
        """The ``derived_from`` entry that records every parameter."""
        return (
            f"params:grid={self.grid_mm},width={self.track_width_mm},clearance={self.clearance_mm},"
            f"via={self.via_diameter_mm}/{self.via_drill_mm},edge={self.edge_clearance_mm},"
            f"via_cost={self.via_cost},bend_cost={self.bend_cost}"
        )


@dataclass(slots=True)
class Routing:
    """What :func:`route_board` produced. ``unrouted`` maps a net name to why it has no copper."""

    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    unrouted: dict[str, str] = field(default_factory=dict)
    params: RoutingParams = field(default_factory=RoutingParams)
    #: routed_nets, unrouted_nets, skipped_nets (< 2 pads), net_length_mm, total_length_mm, track_count,
    #: via_count, grid (nx, ny, cells per layer), raised (parameter -> [requested, effective])
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


@dataclass(slots=True)
class _Terminal:
    pad: _PadGeom
    cell: int  # cell index k = j * nx + i
    label: str
    stub_layer: int | None = None  # the layer the stub was emitted on once the terminal is connected


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


# --------------------------------------------------------------------------- the board model


class _Board:
    """The grid, the owner maps and the terminals of one IR (built once per :func:`route_board`)."""

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
        # owner maps: one flat list per layer, None = free
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
        self.path_disc = _disc(p.track_width_mm + p.clearance_mm + g / 2.0, g)
        self.via_disc = _disc(p.via_diameter_mm / 2.0 + p.clearance_mm + p.track_width_mm / 2.0, g)
        self.net_index = {net.name: k for k, net in enumerate(ir.nets)}
        self.terminals: dict[str, list[_Terminal]] = {net.name: [] for net in ir.nets}
        self._load_pads(ir, library)

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
        for name, terms in self.terminals.items():
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
                    self._own(owner, j * self.nx + i, geom.net)

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

    @staticmethod
    def _own(owner: list[int | None], k: int, net: int) -> None:
        cur = owner[k]
        if cur is None:
            owner[k] = net
        elif cur != net:
            owner[k] = BLOCKED

    def mark_disc(self, layer: int, k: int, disc: list[tuple[int, int]], net: int) -> None:
        j, i = divmod(k, self.nx)
        owner = self.owner[layer]
        for a, b in disc:
            ii, jj = i + a, j + b
            if 0 <= ii < self.nx and 0 <= jj < self.ny:
                self._own(owner, jj * self.nx + ii, net)

    def via_allowed(self, k: int, net: int) -> bool:
        if not self.via_edge_ok[k] or not self.via_pad_ok[k]:
            return False
        j, i = divmod(k, self.nx)
        for a, b in self.via_disc:
            ii, jj = i + a, j + b
            if not (0 <= ii < self.nx and 0 <= jj < self.ny):
                return False
            kk = jj * self.nx + ii
            for owner in self.owner:
                cur = owner[kk]
                if cur is not None and cur != net:
                    return False
        return True


# --------------------------------------------------------------------------- the search


def _astar(board: _Board, net: int, tree: list[int], tree_set: set[int], targets: dict[int, _Terminal]) -> list[int] | None:
    """Cheapest path (as ``layer * n + cell`` ids) from any tree cell to any target cell; ``None`` when none is reachable.

    States are ``(layer, cell, direction)`` so the bend cost is exact; the
    heuristic is the Manhattan distance to the nearest target cell (a lower
    bound: every step costs at least 1 and bends / vias only add). Ties in
    the heap are broken by the push counter, never by memory addresses.
    """
    p = board.p
    nx, n = board.nx, board.n
    owner = board.owner
    bend, via_cost = p.bend_cost, p.via_cost
    target_ij = [divmod(c % n, nx) for c in targets]  # (j, i)

    def h(c: int) -> float:
        j, i = divmod(c % n, nx)
        return min(abs(i - ti) + abs(j - tj) for tj, ti in target_ij)

    g_cost: dict[int, float] = {}
    prev: dict[int, int] = {}
    heap: list[tuple[float, int, float, int]] = []
    counter = 0
    for c in tree:
        s = c * _STATES_PER_CELL + _NO_DIR
        if s not in g_cost:
            g_cost[s] = 0.0
            heap.append((h(c), counter, 0.0, s))
            counter += 1
    heapq.heapify(heap)
    via_memo: dict[int, bool] = {}
    found: int | None = None
    while heap:
        _, _, g, s = heapq.heappop(heap)
        if g > g_cost.get(s, math.inf):
            continue
        c, d = divmod(s, _STATES_PER_CELL)
        if c in targets and c not in tree_set:
            found = s
            break
        layer, k = divmod(c, n)
        j, i = divmod(k, nx)
        for nd, (di, dj) in enumerate(_DIRS):
            if d != _NO_DIR and nd == (d + 2) % 4:
                continue  # no reversal onto the cell we came from
            ii, jj = i + di, j + dj
            if not (0 <= ii < nx and 0 <= jj < board.ny):
                continue
            kk = jj * nx + ii
            nc = layer * n + kk
            cur = owner[layer][kk]
            if cur is not None and cur != net:
                continue  # BLOCKED or another net's: never entered, a target cell included
            ng = g + 1.0 + (bend if d != _NO_DIR and nd != d else 0.0)
            ns = nc * _STATES_PER_CELL + nd
            if ng < g_cost.get(ns, math.inf):
                g_cost[ns] = ng
                prev[ns] = s
                heapq.heappush(heap, (ng + h(nc), counter, ng, ns))
                counter += 1
        ok = via_memo.get(k)
        if ok is None:
            ok = via_memo[k] = board.via_allowed(k, net)
        if ok:
            nc = (1 - layer) * n + k
            ns = nc * _STATES_PER_CELL + _NO_DIR
            ng = g + via_cost
            if ng < g_cost.get(ns, math.inf):
                g_cost[ns] = ng
                prev[ns] = s
                heapq.heappush(heap, (ng + h(nc), counter, ng, ns))
                counter += 1
    if found is None:
        return None
    path: list[int] = []
    s = found
    while True:
        c = s // _STATES_PER_CELL
        if not path or path[-1] != c:
            path.append(c)
        if s not in prev:
            break
        s = prev[s]
    path.reverse()
    return path


# --------------------------------------------------------------------------- emission


def _provenance(net: Net, p: RoutingParams) -> Provenance:
    refs = sorted({pin.component_ref for pin in net.pins})
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=ROUTER_ID,
        tool_version=ROUTER_VERSION,
        derived_from=[f"net:{net.name}", *(f"placement:{r}" for r in refs), p.derived_from_entry()],
        note="grid maze route on F.Cu/B.Cu; validity is decided by kicad-cli DRC (pcb.routing checks the IR geometry only)",
    )


class _NetCopper:
    """The copper of one net while it is being routed (discarded whole when a terminal is unreachable)."""

    def __init__(self, board: _Board, net: Net, prov: Provenance) -> None:
        self.board = board
        self.net = net
        self.prov = prov
        self.p = board.p
        self.tracks: list[Track] = []
        self.vias: list[Via] = []
        self.path_cells: list[tuple[int, int]] = []  # (layer, k)
        self.via_cells: list[int] = []

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
                self.via_cells.append(ka)
                continue
            step = (kb % nx - ka % nx, kb // nx - ka // nx)
            if run_start is None:
                run_start, run_dir = ka, step
            elif step != run_dir:
                self._track(la, run_start, ka)
                run_start, run_dir = ka, step
        if run_start is not None:
            self._track(path[-1] // n, run_start, path[-1] % n)
        for c in path:
            self.path_cells.append(divmod(c, n))

    def add_stub(self, terminal: _Terminal, layer: int) -> None:
        """One segment from the terminal cell to the exact pad centre (nothing when the cell is the centre)."""
        if terminal.stub_layer is not None:
            return
        terminal.stub_layer = layer
        start = self.board.pos(terminal.cell)
        end = (_q(terminal.pad.cx), _q(terminal.pad.cy))
        if start != end:
            self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=self.p.track_width_mm, provenance=self.prov))

    def commit(self) -> None:
        """Write the net's marks on the owner maps (only once the whole net has routed)."""
        idx = self.board.net_index[self.net.name]
        for layer, k in self.path_cells:
            self.board.mark_disc(layer, k, self.board.path_disc, idx)
        for k in self.via_cells:
            for layer in range(len(LAYERS)):
                self.board.mark_disc(layer, k, self.board.via_disc, idx)

    def length_mm(self) -> float:
        return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in self.tracks)


# --------------------------------------------------------------------------- the router


def _route_net(board: _Board, net: Net, copper: _NetCopper) -> str | None:
    """Grow the net's tree until every terminal hangs on it; the reason when a terminal is unreachable."""
    n = board.n
    idx = board.net_index[net.name]
    terms = board.terminals[net.name]
    # a terminal cell inside a keep-out (a foreign pad, a net-less pad, the board edge) is no seed and no target on that
    # layer; a terminal with no usable layer cannot be connected without breaking the router's own clearance
    usable = {t.label: board.usable_layers(t, idx) for t in terms}
    fenced = [t for t in terms if not usable[t.label]]
    if fenced:
        x, y = board.pos(fenced[0].cell)
        return (
            f"{fenced[0].label} terminal cell ({x:g}, {y:g}) is inside a keep-out on every copper layer of the pad "
            f"(a foreign pad, a pad without a net or the board edge is within clearance {board.p.clearance_mm:g} + width/2 of it)"
        )
    first, remaining = terms[0], list(terms[1:])
    tree: list[int] = [layer * n + first.cell for layer in usable[first.label]]
    tree_set = set(tree)
    seeds = {c: first for c in tree}
    while remaining:
        targets: dict[int, _Terminal] = {}
        for t in remaining:
            for layer in usable[t.label]:
                targets.setdefault(layer * n + t.cell, t)
        already = [t for t in remaining if any(layer * n + t.cell in tree_set for layer in usable[t.label])]
        if already:
            t = already[0]
            layer = next(layer for layer in usable[t.label] if layer * n + t.cell in tree_set)
            copper.add_stub(t, layer)
            remaining.remove(t)
            continue
        path = _astar(board, idx, tree, tree_set, targets)
        if path is None:
            return ", ".join(t.label for t in remaining) + " unreachable from the routed part of the net"
        seed = seeds.get(path[0])
        if seed is not None:
            copper.add_stub(seed, path[0] // n)
        copper.add_path(path)
        reached = targets[path[-1]]
        copper.add_stub(reached, path[-1] // n)
        remaining.remove(reached)
        for c in path:
            if c not in tree_set:
                tree_set.add(c)
                tree.append(c)
        for layer in usable[reached.label]:
            c = layer * n + reached.cell
            if c not in tree_set:
                tree_set.add(c)
                tree.append(c)
    return None


def route_board(ir: CircuitIR, library: KicadLibrary, params: RoutingParams | None = None) -> Routing:
    """Route every net of the placed board on ``F.Cu`` / ``B.Cu``; pure (same IR + library + params -> same result).

    ``ir`` is not mutated and its existing copper (``ir.pcb.tracks`` / ``vias``
    / ``zones``) is neither an obstacle nor reused: the caller decides what to
    do with the result (the PCB agent never routes a board that already has
    copper). Nets are routed in ``(pad count, name)`` order, with one second
    pass from scratch that puts the nets the first pass could not route first
    (module docstring); a net with fewer than two pads is skipped (nothing to
    connect; listed in ``stats["skipped_nets"]``). A net whose terminals
    cannot all be reached in either pass is listed in :attr:`Routing.unrouted`
    with no copper at all. Tracks come in routing order (``stats["net_order"]``),
    then path order. Raises :class:`CompileError` for anything that would
    need a guess (module docstring).
    """
    p, raised = effective_params(ir, params)
    board = _Board(ir, library, p)
    ordered = sorted(ir.nets, key=lambda net: (len(board.terminals[net.name]), net.name))
    skipped = [net.name for net in ordered if len(board.terminals[net.name]) < 2]
    order = [net for net in ordered if len(board.terminals[net.name]) >= 2]
    result, lengths = _route_in_order(board, order, p)
    passes = 1
    if result.unrouted:
        # second pass, from scratch: the nets the first order starved go first (their relative order kept); it is
        # taken only when it leaves fewer nets unrouted, so a board that routes in one pass is unchanged
        retry = [net for net in order if net.name in result.unrouted] + [net for net in order if net.name not in result.unrouted]
        again, again_lengths = _route_in_order(_Board(ir, library, p), retry, p)
        if len(again.unrouted) < len(result.unrouted):
            result, lengths, order, passes = again, again_lengths, retry, 2
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
        "passes": passes,
        "net_order": [net.name for net in order],
    }
    return result


def _route_in_order(board: _Board, order: list[Net], p: RoutingParams) -> tuple[Routing, dict[str, float]]:
    """Route ``order`` on a fresh ``board``; ``(result without stats, routed net -> copper length)``."""
    result = Routing(params=p)
    lengths: dict[str, float] = {}
    for net in order:
        copper = _NetCopper(board, net, _provenance(net, p))
        reason = _route_net(board, net, copper)
        if reason is not None:
            result.unrouted[net.name] = reason
            continue
        copper.commit()
        result.tracks.extend(copper.tracks)
        result.vias.extend(copper.vias)
        lengths[net.name] = _q(copper.length_mm())
    return result, lengths
