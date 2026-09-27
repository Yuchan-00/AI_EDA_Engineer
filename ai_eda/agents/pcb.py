"""PCB Agent: proposes a deterministic placement and a maze-routed board into ``ir.pcb``; DRC is done by KiCad, not here.

Invariants this agent keeps:

* It only *proposes*. It never mutates ``ir`` (the content hash before and
  after :meth:`PCBAgent.run` is the same); the orchestrator applies the
  proposal through ``Orchestrator.apply_proposals`` like any other.
* No LLM. The placement is
  :func:`ai_eda.tools.placement.core_ring.core_ring_placement` when a part
  has at least :data:`~ai_eda.tools.placement.core_ring.CORE_MIN_PADS` pads
  (the many-pad part in the centre, the parts wired only to it on an inner
  ring, the rest at the edge) and
  :func:`ai_eda.tools.placement.grid.grid_placement` otherwise (every
  template before the 64-pin microcontroller board, unchanged); the copper
  is :func:`ai_eda.tools.routing.maze.route_board`. All of them run on the
  footprints a :class:`~ai_eda.tools.kicad.library.KicadLibrary`
  (``ctx.tools["kicad_library"]``) found on disk; every placement carries
  ``derived`` / ``placement.core_ring`` or ``placement.grid`` provenance and
  every track / via ``derived`` / ``routing.maze`` provenance naming the
  net, the placements and the router's parameters.
* The routing rules come from the board, not from a guess: unless the agent
  was built with explicit :class:`~ai_eda.tools.routing.maze.RoutingParams`,
  it routes with :meth:`RoutingParams.for_board
  <ai_eda.tools.routing.maze.RoutingParams.for_board>` - the fine rules
  (0.2 mm grid, 0.25 mm track, 0.2 mm clearance, 0.6 / 0.3 mm via) when a
  footprint's pads are closer than 1.0 mm centre to centre, the defaults
  otherwise. The fine rules and the pitch that chose them are in every
  track's ``params:`` provenance entry and in a ``fine rules: ...`` note; a
  default-rule board's notes and copper are exactly what they were before
  the selection existed.
* It never guesses a footprint, never resizes a user outline, never places
  on top of existing copper and never replaces copper: it places only when
  ``ir.pcb`` is ``None`` or has neither placements nor copper, and it routes
  only a board (the one it just placed, or the user's placements) that has
  no tracks / vias / zones. Otherwise it says why in a note starting
  ``not placed:`` / ``not routed:``.
* Place then route is **one** proposal, ``target="pcb"``,
  ``operation="set"``, whose payload is the whole
  :class:`~ai_eda.ir.PCBDesign` (the existing one copied with ``outline``
  + ``placements`` + ``tracks`` + ``vias`` filled, so layers and
  manufacturing constraints survive). It never emits a dotted ``pcb.*``
  target: ``apply_proposals`` resolves every target's parent *before* any
  step runs, so a second ``pcb.placements`` proposal in the same result would
  land on the old (replaced) object or raise on ``None``.
* A half-routed board is never proposed by default: when the router leaves
  any net unrouted (:attr:`~ai_eda.tools.routing.maze.Routing.unrouted`),
  refuses (``CompileError``, a library error) or is skipped, the proposal
  carries the placement only (nothing at all when the board was already
  placed) and one ``not routed: <net>: <reason>`` note per net, after a
  ``not applied:`` note with what the router did connect (nets, tracks,
  vias, copper length, negotiation iterations) when it connected any -
  reported, never proposed. Only the opt-in answer ``--answer
  pcb.routing=partial`` (:data:`PARTIAL_ANSWER`) proposes the nets the
  router did connect - each net whole, the router never emits part of one -
  with a ``partial:`` note naming every net left without copper for manual
  routing in KiCad (``pcb.routing.connectivity`` then FAILs naming them at
  IR_BUILD: the board is honestly unfinished). Anything else propagates: it
  is a defect, not a verdict.
* It asks no question. The only steering is the two control keys
  :data:`PLACEMENT_KEY` (``--answer pcb.placement=skip`` proposes nothing)
  and :data:`ROUTING_KEY` (``--answer pcb.routing=skip`` proposes the
  placement without copper, ``--answer pcb.routing=partial`` applies the
  routed nets of a board the router could not finish); any other value is
  noted as not understood and the work proceeds; the keys never become
  requirements (``CONTROL_KEYS`` in :mod:`ai_eda.agents.keys`).

The PLACEMENT stage runs before IR_BUILD, so this agent may see an
inconsistent IR (a net naming a component that does not exist). The placer
walks ``ir.components`` only, so such an IR is still placed; the router
refuses it (``not routed:`` note) and IR_BUILD then reports the
inconsistency exactly as before. The stage outcome is never PASS: a
proposal is not evidence. What the IR copper proves is judged by the
``pcb.routing.*`` validators at IR_BUILD (IR geometry, not DRC); a board
left unrouted FAILs ``pcb.routing.connectivity`` there and DRC with
``unconnected_items`` later. The router raises its width / clearance / via
sizes to the limits already in ``ir.pcb.manufacturing`` when it runs; limits
the FAB_CAPABILITY stage records *after* this stage in the same run reach
the copper only through ``mfg.capability`` / DRC, never by a silent re-route.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.keys import PLACEMENT_KEY, ROUTING_KEY
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, PCBDesign
from ai_eda.llm.router import TaskKind
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.core_ring import CORE_MIN_PADS, RingPlacement, core_ring_placement, find_core
from ai_eda.tools.placement.core_ring import PLACER_ID as RING_PLACER_ID
from ai_eda.tools.placement.core_ring import PLACER_VERSION as RING_PLACER_VERSION
from ai_eda.tools.placement.grid import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, _resolve, grid_placement
from ai_eda.tools.routing.maze import FINE_PITCH_MM, ROUTER_ID, ROUTER_VERSION, Routing, RoutingParams, route_board

#: ``--answer pcb.placement=skip`` makes the agent propose nothing, ``--answer pcb.routing=skip`` leaves the placement
#: without copper (``PLACEMENT_KEY`` / ``ROUTING_KEY``: control keys, never requirements, defined in
#: :mod:`ai_eda.agents.keys` and imported above)
SKIP_ANSWER = "skip"
#: ``--answer pcb.routing=partial``: when the router leaves nets unrouted, apply the nets it did route (each whole)
#: instead of the placement alone; the unrouted nets are named for manual routing
PARTIAL_ANSWER = "partial"
#: the note a board left without copper gets: what DRC will say about it
UNROUTED_NOTE = "unrouted: DRC will report unconnected_items until routed"
#: the rationale of a proposal whose placement is the grid / the core ring
GRID_RATIONALE = "row-major grid from verified library footprint extents; placed extents are pairwise disjoint and inside the outline"
RING_RATIONALE = (
    "core-and-ring placement from verified library footprint extents and the IR nets: the part with the most pads in the centre, "
    "the parts wired only to it on an inner ring by pull angle, the rest on an outer ring at the board edge; "
    "placed extents are pairwise disjoint and inside the outline"
)


class PCBAgent(Agent):
    name = "pcb"
    task = TaskKind.CIRCUIT_DESIGN

    def __init__(
        self, *, spacing_mm: float = SPACING_MM, margin_mm: float = MARGIN_MM, columns: int = COLUMNS, routing: RoutingParams | None = None,
    ) -> None:
        self.spacing_mm = spacing_mm
        self.margin_mm = margin_mm
        self.columns = columns
        self.routing = routing

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        notes: list[str] = []
        answer = ctx.answers.get(PLACEMENT_KEY)
        if answer is not None:
            if answer.strip().lower() == SKIP_ANSWER:
                return self._result(notes=["placement skipped by answer"])
            notes.append(f"{PLACEMENT_KEY}={answer!r} not understood (the only answer is '{SKIP_ANSWER}'); placing as usual")
        reason = self._refusal(ir, ctx)
        if reason is not None:
            notes.append(f"not placed: {reason}")
            if ir.pcb is None or not ir.pcb.placements:
                return self._result(notes=notes)
            # the user's (or an earlier run's) placements: route them, never move them
            return self._route(ir, ctx, ir.pcb, notes, placed=False, description="")
        library: KicadLibrary = ctx.tools["kicad_library"]
        base = ir.pcb if ir.pcb is not None else PCBDesign()
        origin = "user" if base.outline is not None else "generated"
        try:
            core, _fp, core_pads = find_core(_resolve(ir, library))
            if core_pads >= CORE_MIN_PADS:
                ring = core_ring_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, outline=base.outline)
                placed = base.model_copy(update={"outline": ring.outline, "placements": list(ring.placements)})
                description = self._ring_description(ring, origin)
                basis = RING_RATIONALE
            else:
                grid = grid_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, columns=self.columns, outline=base.outline)
                placed = base.model_copy(update={"outline": grid.outline, "placements": list(grid.placements)})
                o = grid.outline
                description = (
                    f"{PLACER_ID} {PLACER_VERSION}: {len(grid.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm "
                    f"{origin} outline at ({o.origin_x_mm}, {o.origin_y_mm}), "
                    f"pitch {grid.pitch[0]} x {grid.pitch[1]} mm, {self.columns} column(s)"
                )
                basis = GRID_RATIONALE
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return self._result(notes=[*notes, f"not placed: {e}"])
        return self._route(ir, ctx, placed, notes, placed=True, description=description, basis=basis)

    def _ring_description(self, ring: RingPlacement, origin: str) -> str:
        o = ring.outline
        centre = next(p for p in ring.placements if p.component_ref == ring.core)
        inner = sum(1 for r in ring.rings.values() if r == "inner")
        outer = sum(1 for r in ring.rings.values() if r == "outer")
        return (
            f"{RING_PLACER_ID} {RING_PLACER_VERSION}: {len(ring.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm "
            f"{origin} outline at ({o.origin_x_mm}, {o.origin_y_mm}), core {ring.core} ({ring.core_pads} pads) at ({centre.x_mm}, {centre.y_mm}), "
            f"{inner} part(s) on the inner ring, {outer} on the outer ring, spacing {self.spacing_mm} mm, margin {self.margin_mm} mm"
        )

    def _route(
        self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign, notes: list[str], *, placed: bool, description: str, basis: str = GRID_RATIONALE,
    ) -> AgentResult:
        """Route ``board`` (placements, no copper) and build the one proposal; ``placed`` says whether this run placed it (``basis``: how)."""
        answer = ctx.answers.get(ROUTING_KEY)
        partial = False
        if answer is not None:
            if answer.strip().lower() == SKIP_ANSWER:
                return self._unrouted(board, notes, ["routing skipped by answer"], placed=placed, description=description, basis=basis)
            if answer.strip().lower() == PARTIAL_ANSWER:
                partial = True
            else:
                notes.append(
                    f"{ROUTING_KEY}={answer!r} not understood (the only answers are '{SKIP_ANSWER}' and '{PARTIAL_ANSWER}'); routing as usual"
                )
        if board.tracks or board.vias or board.zones:
            reason = f"not routed: ir.pcb already has copper ({len(board.tracks)} track(s), {len(board.vias)} via(s), {len(board.zones)} zone(s)); the agent never replaces copper"
            return self._unrouted(board, notes, [reason], placed=placed, description=description, basis=basis)
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return self._unrouted(
                board, notes, ["not routed: no KiCad library in ctx.tools['kicad_library']; pad geometry cannot be read"], placed=placed, description=description, basis=basis,
            )
        board_ir = ir.model_copy(update={"pcb": board})  # a shallow copy: ir itself is never touched
        params: RoutingParams | None = None
        try:
            params = self.routing if self.routing is not None else RoutingParams.for_board(board_ir, library)
            routing = route_board(board_ir, library, params)
        except (CompileError, LibraryLookupError, LibraryFormatError) as e:
            rules = [self._rules_note(params)] if params is not None and params.rules is not None else []
            return self._unrouted(board, notes, [*rules, f"not routed: {e}"], placed=placed, description=description, basis=basis)
        reasons = [f"not routed: {net}: {why}" for net, why in routing.unrouted.items()]
        if routing.unrouted and not (partial and routing.stats.get("routed_nets")):
            rules = [self._rules_note(routing.params, routing.stats.get("raised"))] if routing.params.rules is not None else []
            dropped = [self._dropped_note(routing)] if routing.stats.get("routed_nets") else []
            return self._unrouted(board, notes, [*rules, *dropped, *reasons], placed=placed, description=description, basis=basis)
        payload = board.model_copy(update={"tracks": list(routing.tracks), "vias": list(routing.vias)})
        routed = self._routing_description(routing)
        which = "the nets listed as routed" if routing.unrouted else "every net"
        if placed:
            notes.append(description)
            full = f"{description}; {routed}"
            rationale = (
                f"{basis}; {which} maze-routed on F.Cu/B.Cu from the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity"
            )
        else:
            full = routed
            rationale = f"{which} maze-routed on F.Cu/B.Cu from the existing placements and the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity"
        notes.append(routed)
        if routing.unrouted:  # the opt-in partial board: whole nets only, the rest named
            left = self._partial_note(routing)
            full = f"{full}; {left}"
            notes += [left, *reasons]
        proposal = IRProposal(description=full, target="pcb", operation="set", payload=payload, rationale=rationale)
        return self._result(proposals=[proposal], notes=notes)

    def _unrouted(self, board: PCBDesign, notes: list[str], reasons: list[str], *, placed: bool, description: str, basis: str = GRID_RATIONALE) -> AgentResult:
        """The placement alone (or nothing when the board was already placed) plus why there is no copper."""
        if not placed:
            return self._result(notes=[*notes, *reasons])
        proposal = IRProposal(description=description, target="pcb", operation="set", payload=board, rationale=f"{basis}; DRC decides validity")
        return self._result(proposals=[proposal], notes=[*notes, f"{description}; {UNROUTED_NOTE}", *reasons])

    @staticmethod
    def _rules_note(p: RoutingParams, raised: dict | None = None) -> str:
        """``fine rules: pad pitch 0.8 mm ...``: why the board was routed at the fine rules, with the values the router used."""
        text = (
            f"{p.rules} rules: pad pitch {p.pad_pitch_mm} mm ({p.pitch_footprint}) is below {FINE_PITCH_MM} mm: grid {p.grid_mm} mm, "
            f"width {p.track_width_mm} mm, clearance {p.clearance_mm} mm, via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
        if raised:
            text += " (raised to ir.pcb.manufacturing minimums: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in raised.items()) + ")"
        return text

    @staticmethod
    def _dropped_note(routing: Routing) -> str:
        """What the router did connect on a board it could not finish - reported, never proposed (no half-routed board)."""
        s = routing.stats
        total = s["routed_nets"] + s["unrouted_nets"]
        return (
            f"not applied: {ROUTER_ID} {ROUTER_VERSION} connected {s['routed_nets']} of {total} net(s) ({s['track_count']} track(s), {s['via_count']} via(s), "
            f"{s['total_length_mm']} mm of copper, {s['iterations']} iteration(s)) but not the {s['unrouted_nets']} below; a half-routed board is never proposed, "
            f"so the proposal carries the placement only (--answer {ROUTING_KEY}={PARTIAL_ANSWER} applies the routed nets, each whole)"
        )

    @staticmethod
    def _partial_note(routing: Routing) -> str:
        """The opt-in partial board: which nets were left without copper, for manual routing."""
        s = routing.stats
        total = s["routed_nets"] + s["unrouted_nets"]
        return (
            f"partial: {ROUTING_KEY}={PARTIAL_ANSWER} applies the {s['routed_nets']} of {total} net(s) the router connected, each whole; "
            f"{s['unrouted_nets']} net(s) have no copper and need manual routing in KiCad: {', '.join(routing.unrouted)} "
            "(pcb.routing.connectivity judges the IR copper, DRC the board)"
        )

    @staticmethod
    def _routing_description(routing: Routing) -> str:
        p = routing.params
        s = routing.stats
        text = (
            f"{ROUTER_ID} {ROUTER_VERSION}: {s['routed_nets']} net(s) routed on F.Cu/B.Cu, {s['track_count']} track(s), {s['via_count']} via(s), "
            f"{s['total_length_mm']} mm of copper, {s['iterations']} negotiation iteration(s); grid {p.grid_mm} mm, width {p.track_width_mm} mm, "
            f"clearance {p.clearance_mm} mm, via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
        if p.rules is not None:
            text += f"; {p.rules} rules: pad pitch {p.pad_pitch_mm} mm ({p.pitch_footprint}) is below {FINE_PITCH_MM} mm"
        if s["skipped_nets"]:
            text += f"; {len(s['skipped_nets'])} net(s) with fewer than 2 pads skipped ({', '.join(s['skipped_nets'])})"
        if s["raised"]:
            text += "; raised to ir.pcb.manufacturing minimums: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in s["raised"].items())
        return text

    @staticmethod
    def _refusal(ir: CircuitIR, ctx: AgentContext) -> str | None:
        """Why nothing is placed, or ``None`` when the agent may place."""
        if ir.pcb is not None and ir.pcb.placements:
            return f"ir.pcb already has {len(ir.pcb.placements)} placement(s); the agent never replaces a layout"
        if ir.pcb is not None and (ir.pcb.tracks or ir.pcb.vias or ir.pcb.zones):
            return (
                f"ir.pcb has copper without placements ({len(ir.pcb.tracks)} track(s), {len(ir.pcb.vias)} via(s), "
                f"{len(ir.pcb.zones)} zone(s)); placing under existing copper would be a guess"
            )
        if not ir.components:
            return "no components to place"
        if not isinstance(ctx.tools.get("kicad_library"), KicadLibrary):
            return "no KiCad library in ctx.tools['kicad_library']; footprint extents cannot be read"
        return None


__all__ = ["GRID_RATIONALE", "PARTIAL_ANSWER", "PCBAgent", "PLACEMENT_KEY", "RING_RATIONALE", "ROUTING_KEY", "SKIP_ANSWER", "UNROUTED_NOTE"]
