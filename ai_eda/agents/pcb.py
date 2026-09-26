"""PCB Agent: proposes a deterministic grid placement and a maze-routed board into ``ir.pcb``; DRC is done by KiCad, not here.

Invariants this agent keeps:

* It only *proposes*. It never mutates ``ir`` (the content hash before and
  after :meth:`PCBAgent.run` is the same); the orchestrator applies the
  proposal through ``Orchestrator.apply_proposals`` like any other.
* No LLM. The placement is :func:`ai_eda.tools.placement.grid.grid_placement`
  and the copper is :func:`ai_eda.tools.routing.maze.route_board`, both run
  on the footprints a :class:`~ai_eda.tools.kicad.library.KicadLibrary`
  (``ctx.tools["kicad_library"]``) found on disk; every placement carries
  ``derived`` / ``placement.grid`` provenance and every track / via
  ``derived`` / ``routing.maze`` provenance naming the net, the placements
  and the router's parameters.
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
* A half-routed board is never proposed: when the router leaves any net
  unrouted (:attr:`~ai_eda.tools.routing.maze.Routing.unrouted`), refuses
  (``CompileError``, a library error) or is skipped, the proposal carries
  the placement only (nothing at all when the board was already placed)
  and one ``not routed: <net>: <reason>`` note per net. Anything else
  propagates: it is a defect, not a verdict.
* It asks no question. The only steering is the two control keys
  :data:`PLACEMENT_KEY` (``--answer pcb.placement=skip`` proposes nothing)
  and :data:`ROUTING_KEY` (``--answer pcb.routing=skip`` proposes the
  placement without copper); any other value is noted as not understood and
  the work proceeds; the keys never become requirements (``CONTROL_KEYS`` in
  :mod:`ai_eda.agents.keys`).

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
from ai_eda.tools.placement.grid import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, grid_placement
from ai_eda.tools.routing.maze import ROUTER_ID, ROUTER_VERSION, Routing, RoutingParams, route_board

#: ``--answer pcb.placement=skip`` makes the agent propose nothing, ``--answer pcb.routing=skip`` leaves the placement
#: without copper (``PLACEMENT_KEY`` / ``ROUTING_KEY``: control keys, never requirements, defined in
#: :mod:`ai_eda.agents.keys` and imported above)
SKIP_ANSWER = "skip"
#: the note a board left without copper gets: what DRC will say about it
UNROUTED_NOTE = "unrouted: DRC will report unconnected_items until routed"


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
        try:
            grid = grid_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, columns=self.columns, outline=base.outline)
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return self._result(notes=[*notes, f"not placed: {e}"])
        placed = base.model_copy(update={"outline": grid.outline, "placements": list(grid.placements)})
        o = grid.outline
        description = (
            f"{PLACER_ID} {PLACER_VERSION}: {len(grid.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm "
            f"{'user' if base.outline is not None else 'generated'} outline at ({o.origin_x_mm}, {o.origin_y_mm}), "
            f"pitch {grid.pitch[0]} x {grid.pitch[1]} mm, {self.columns} column(s)"
        )
        return self._route(ir, ctx, placed, notes, placed=True, description=description)

    def _route(self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign, notes: list[str], *, placed: bool, description: str) -> AgentResult:
        """Route ``board`` (placements, no copper) and build the one proposal; ``placed`` says whether this run placed it."""
        answer = ctx.answers.get(ROUTING_KEY)
        if answer is not None:
            if answer.strip().lower() == SKIP_ANSWER:
                return self._unrouted(board, notes, ["routing skipped by answer"], placed=placed, description=description)
            notes.append(f"{ROUTING_KEY}={answer!r} not understood (the only answer is '{SKIP_ANSWER}'); routing as usual")
        if board.tracks or board.vias or board.zones:
            reason = f"not routed: ir.pcb already has copper ({len(board.tracks)} track(s), {len(board.vias)} via(s), {len(board.zones)} zone(s)); the agent never replaces copper"
            return self._unrouted(board, notes, [reason], placed=placed, description=description)
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return self._unrouted(board, notes, ["not routed: no KiCad library in ctx.tools['kicad_library']; pad geometry cannot be read"], placed=placed, description=description)
        try:
            routing = route_board(ir.model_copy(update={"pcb": board}), library, self.routing)  # a shallow copy: ir itself is never touched
        except (CompileError, LibraryLookupError, LibraryFormatError) as e:
            return self._unrouted(board, notes, [f"not routed: {e}"], placed=placed, description=description)
        if routing.unrouted:
            reasons = [f"not routed: {net}: {why}" for net, why in routing.unrouted.items()]
            return self._unrouted(board, notes, reasons, placed=placed, description=description)
        payload = board.model_copy(update={"tracks": list(routing.tracks), "vias": list(routing.vias)})
        routed = self._routing_description(routing)
        if placed:
            notes.append(description)
            full = f"{description}; {routed}"
            rationale = (
                "row-major grid from verified library footprint extents; placed extents are pairwise disjoint and inside the outline; "
                "every net maze-routed on F.Cu/B.Cu from the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity"
            )
        else:
            full = routed
            rationale = "every net maze-routed on F.Cu/B.Cu from the existing placements and the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity"
        notes.append(routed)
        proposal = IRProposal(description=full, target="pcb", operation="set", payload=payload, rationale=rationale)
        return self._result(proposals=[proposal], notes=notes)

    def _unrouted(self, board: PCBDesign, notes: list[str], reasons: list[str], *, placed: bool, description: str) -> AgentResult:
        """The placement alone (or nothing when the board was already placed) plus why there is no copper."""
        if not placed:
            return self._result(notes=[*notes, *reasons])
        proposal = IRProposal(
            description=description,
            target="pcb",
            operation="set",
            payload=board,
            rationale="row-major grid from verified library footprint extents; placed extents are pairwise disjoint and inside the outline; DRC decides validity",
        )
        return self._result(proposals=[proposal], notes=[*notes, f"{description}; {UNROUTED_NOTE}", *reasons])

    @staticmethod
    def _routing_description(routing: Routing) -> str:
        p = routing.params
        s = routing.stats
        text = (
            f"{ROUTER_ID} {ROUTER_VERSION}: {s['routed_nets']} net(s) routed on F.Cu/B.Cu, {s['track_count']} track(s), {s['via_count']} via(s), "
            f"{s['total_length_mm']} mm of copper; grid {p.grid_mm} mm, width {p.track_width_mm} mm, clearance {p.clearance_mm} mm, "
            f"via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
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


__all__ = ["PCBAgent", "PLACEMENT_KEY", "ROUTING_KEY", "SKIP_ANSWER", "UNROUTED_NOTE"]
