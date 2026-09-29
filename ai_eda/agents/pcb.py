"""PCB Agent: proposes a deterministic placement, a maze-routed board and its silkscreen into ``ir.pcb``; DRC is done by KiCad, not here.

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
* Place, route and silkscreen are **one** proposal, ``target="pcb"``,
  ``operation="set"``, whose payload is the whole
  :class:`~ai_eda.ir.PCBDesign` (the existing one copied with ``outline``
  + ``placements`` + ``tracks`` + ``vias`` + ``silkscreen`` filled, so
  layers and manufacturing constraints survive). It never emits a dotted ``pcb.*``
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
* Silkscreen comes last, in the **same** proposal:
  :func:`ai_eda.tools.silkscreen.place.place_silkscreen` (``silkscreen.place``)
  puts the reference designators, connector pin labels and the board title
  on the placed board (the one it just placed and routed, or the user's
  placements) - only when ``ir.pcb.silkscreen`` is empty: existing
  silkscreen is never replaced (``silkscreen not placed: ...`` note). On a
  board the agent neither places nor routes (already placed, already routed
  or left unrouted) and whose silkscreen is empty, the proposal carries the
  existing board unchanged plus the silkscreen. The placer's description is
  the last note and is appended to the proposal description; a placer that
  cannot read the board leaves the silkscreen out with a
  ``silkscreen not placed: <reason>`` note, never a guess. Every text
  carries ``derived`` / ``silkscreen.place`` provenance.
* Signal integrity is need-driven and happens inside the same proposal
  (``ir.si`` net classes, :mod:`ai_eda.tools.si`): the router gets the
  per-net rules the classes map to (:func:`ai_eda.tools.si.rules.net_rules`:
  a controlled-impedance width from ``calc.tline.width_for_z0`` over the
  stackup's plane, a minimum width, length / delay budgets, declared pairs);
  a board whose classes constrain nothing is routed with no rule - routing.maze
  0.2's copper, byte for byte (once any net has a rule, every net is routed
  in the same negotiation, stamped 0.3, and a net without a rule may take
  another path than 0.2 gave it). After the routing pass every routed net's
  line - its longest pad-to-pad path, extracted from the copper and the
  library's pad boxes (:mod:`ai_eda.tools.si.paths`), or a 2-pad net's copper -
  is measured and the critical-length rule
  (:func:`ai_eda.tools.si.promote.promote`) promotes each electrically long
  net of a class that names ``promote_to`` into that controlled class (a net
  whose whole copper exceeds l_crit but whose path was not extracted is only
  possibly long and is never promoted); the
  promotions are a second proposal (``target="si"``, the whole
  :class:`~ai_eda.ir.SIConstraints` with ``derived`` promotion records), and
  when they change the rules the board is re-routed **once** with them. A
  re-route that leaves a net unrouted which the first pass routed keeps the
  first pass's copper (a note says so; ``si.impedance`` then judges the
  promoted nets at the width they have). On a board without a reference
  plane a promoted net gets no controlled width, so nothing is re-routed and
  ``si.impedance`` / ``spice.si`` say why. A 4-layer board (a stackup whose
  inner layers are planes, listed in ``ir.pcb.layers``) is routed on its outer
  layers (``inner_layers=True``) and gets its plane zones
  (:func:`ai_eda.design.stackup.plane_zones`, the ``plane_edge_clearance``
  parameter) in the same ``pcb`` proposal, after the routing and before the
  silkscreen; the planes are return paths and impedance references, never a
  reason to route a net with fewer tracks, and they are not "existing
  copper" for a later run. The silkscreen is placed on the final copper.
* An IR with RF blocks (``ir.rf.blocks``, every kr447 board) is placed by
  :func:`ai_eda.tools.placement.rf_floorplan.rf_floorplan_placement`
  (``placement.rf_floorplan``: block regions, chain order, shield cans first
  with their parts inside the fence ring, keep-outs, a per-region shelf
  packer - never the grid for "the rest"); every other IR keeps the choice
  above. A board with keep-outs (``ir.pcb.keepouts``) or RF blocks hands the
  router its keep-outs and its plane nets (routing.maze 0.4: keep-out
  obstacles, each plane net's SMD pads joined to the plane by a via each,
  never by tracks - the nets of the plane zones the board has or gets here),
  and its plane zones are clipped by every keep-out that forbids zones on
  their layer (the rectangle minus the keep-outs' boxes, a polygon keep-out
  by its bounding box - conservative; the parts as simple polygons, same
  provenance tool so they stay the stackup's planes, the keep-outs named in
  ``derived_from``). Every other board is routed and planed exactly as
  before.
* It asks no question. The only steering is the three control keys
  :data:`PLACEMENT_KEY` (``--answer pcb.placement=skip`` proposes nothing),
  :data:`ROUTING_KEY` (``--answer pcb.routing=skip`` proposes the
  placement without copper, ``--answer pcb.routing=partial`` applies the
  routed nets of a board the router could not finish) and
  :data:`SILKSCREEN_KEY` (``--answer pcb.silkscreen=skip`` proposes no
  silkscreen); any other value is noted as not understood and the work
  proceeds; the keys never become requirements (``CONTROL_KEYS`` in
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
from ai_eda.agents.keys import PLACEMENT_KEY, ROUTING_KEY, SILKSCREEN_KEY
from ai_eda.design.board import PLANE_CLEARANCE_KEY
from ai_eda.design.stackup import STACKUP_TOOL, plane_zones
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, PCBDesign, SIConstraints, Zone
from ai_eda.llm.router import TaskKind
from ai_eda.tools.keepout import allowed_nets, area_bbox, area_is_rect, area_points, covers_layer, forbids, keepout_id, keepouts_of, rect_difference
from ai_eda.tools.kicad.geometry import _q
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.core_ring import CORE_MIN_PADS, RingPlacement, core_ring_placement, find_core
from ai_eda.tools.placement.core_ring import PLACER_ID as RING_PLACER_ID
from ai_eda.tools.placement.core_ring import PLACER_VERSION as RING_PLACER_VERSION
from ai_eda.tools.placement.grid import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, _resolve, grid_placement
from ai_eda.tools.placement.rf_floorplan import RING_MM, rf_blocks, rf_floorplan_placement
from ai_eda.tools.routing.maze import FINE_PITCH_MM, INNER_LAYER_RE, ROUTER_ID, Routing, RoutingParams, route_board
from ai_eda.tools.si.promote import promote
from ai_eda.tools.si.rules import SIRules, net_rules
from ai_eda.tools.silkscreen.place import SilkParams, place_silkscreen

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
#: the rationale part of a proposal that carries the silkscreen
SILK_RATIONALE = (
    "silkscreen placed on estimated text boxes clear of pad copper + 0.15 mm, the outline inset 0.3 mm, the footprints' own silk and each other "
    "(tracks under solder mask and tented vias are not keep-outs); KiCad DRC decides silk_over_copper / silk_overlap"
)
RING_RATIONALE = (
    "core-and-ring placement from verified library footprint extents and the IR nets: the part with the most pads in the centre, "
    "the parts wired only to it on an inner ring by pull angle, the rest on an outer ring at the board edge; "
    "placed extents are pairwise disjoint and inside the outline"
)
#: the rationale of a proposal whose placement is the RF floorplan
RF_RATIONALE = (
    "RF floorplan placement from verified library footprint extents: each block shelf-packed in its region in chain order, "
    "shield cans first with their block's parts inside the fence less the ring, keep-outs that ban footprints / pads honoured; "
    "placed extents are pairwise disjoint (a can's own parts inside its fence), inside their regions and inside the outline"
)


class PCBAgent(Agent):
    name = "pcb"
    task = TaskKind.CIRCUIT_DESIGN

    def __init__(
        self, *, spacing_mm: float = SPACING_MM, margin_mm: float = MARGIN_MM, columns: int = COLUMNS, routing: RoutingParams | None = None,
        silk: SilkParams | None = None,
    ) -> None:
        self.spacing_mm = spacing_mm
        self.margin_mm = margin_mm
        self.columns = columns
        self.routing = routing
        self.silk = silk

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
            if rf_blocks(ir):
                plan = rf_floorplan_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, outline=base.outline, keepouts=keepouts_of(base))
                rf_placed = base.model_copy(update={"outline": plan.outline, "placements": list(plan.placements)})
                rf_description = plan.description(origin, self.spacing_mm, self.margin_mm, RING_MM)
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return self._result(notes=[*notes, f"not placed: {e}"])
        if rf_blocks(ir):
            return self._route(ir, ctx, rf_placed, notes, placed=True, description=rf_description, basis=RF_RATIONALE)
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
                return self._unrouted(ir, ctx, board, notes, ["routing skipped by answer"], placed=placed, description=description, basis=basis)
            if answer.strip().lower() == PARTIAL_ANSWER:
                partial = True
            else:
                notes.append(
                    f"{ROUTING_KEY}={answer!r} not understood (the only answers are '{SKIP_ANSWER}' and '{PARTIAL_ANSWER}'); routing as usual"
                )
        foreign = [z for z in board.zones if not _is_plane(z)]
        if board.tracks or board.vias or foreign:
            reason = f"not routed: ir.pcb already has copper ({len(board.tracks)} track(s), {len(board.vias)} via(s), {len(foreign)} zone(s)); the agent never replaces copper"
            return self._unrouted(ir, ctx, board, notes, [reason], placed=placed, description=description, basis=basis)
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return self._unrouted(
                ir, ctx, board, notes, ["not routed: no KiCad library in ctx.tools['kicad_library']; pad geometry cannot be read"],
                placed=placed, description=description, basis=basis,
            )
        board_ir = ir.model_copy(update={"pcb": board})  # a shallow copy: ir itself is never touched
        inner = _routes_around_planes(board)
        router_extra = self._router_extras(ir, board)
        params: RoutingParams | None = None
        si_rules = SIRules()
        try:
            params = self.routing if self.routing is not None else RoutingParams.for_board(board_ir, library)
            si_rules = net_rules(board_ir, params)
            routing = route_board(board_ir, library, params, rules=si_rules.rules or None, inner_layers=inner, **router_extra)
        except (CompileError, LibraryLookupError, LibraryFormatError) as e:
            rules = [self._rules_note(params)] if params is not None and params.rules is not None else []
            return self._unrouted(ir, ctx, board, notes, [*rules, *self._si_rule_notes(si_rules), f"not routed: {e}"],
                                  placed=placed, description=description, basis=basis)
        si_notes = self._si_rule_notes(si_rules)
        reasons = [f"not routed: {net}: {why}" for net, why in routing.unrouted.items()]
        if routing.unrouted and not (partial and routing.stats.get("routed_nets")):
            rules = [self._rules_note(routing.params, routing.stats.get("raised"))] if routing.params.rules is not None else []
            dropped = [self._dropped_note(routing)] if routing.stats.get("routed_nets") else []
            return self._unrouted(ir, ctx, board, notes, [*rules, *si_notes, *dropped, *reasons],
                                  placed=placed, description=description, basis=basis)
        # --- the critical-length rule on the routed copper, and one re-route with the promoted rules
        extra: list[IRProposal] = []
        routed_ir = ir.model_copy(update={"pcb": board.model_copy(update={"tracks": list(routing.tracks), "vias": list(routing.vias)})})
        new_si, promotions, promo_notes = promote(routed_ir, library=library)
        si_notes += [f"si promotion: {n}" for n in promo_notes]
        if new_si is not None:
            extra.append(IRProposal(
                description=f"si: {len(promotions)} electrically long net(s) promoted ({', '.join(p.net for p in promotions)})", target="si", operation="set",
                payload=new_si, rationale="critical-length rule on the routed delays (derived; the numbers are in each promotion's provenance)",
            ))
            routing, rerouted = self._reroute(board_ir, library, params, si_rules, new_si, inner, routing, router_extra)
            si_notes.append(rerouted)
            reasons = [f"not routed: {net}: {why}" for net, why in routing.unrouted.items()]
        payload = board.model_copy(update={"tracks": list(routing.tracks), "vias": list(routing.vias)})
        payload = self._with_planes(ir, payload, si_notes)
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
        notes.extend(si_notes)
        if routing.unrouted:  # the opt-in partial board: whole nets only, the rest named
            left = self._partial_note(routing)
            full = f"{full}; {left}"
            notes += [left, *reasons]
        proposal = IRProposal(description=full, target="pcb", operation="set", payload=payload, rationale=rationale)
        return self._with_silk(ir, ctx, payload, proposal, notes, extra)

    def _reroute(
        self, board_ir: CircuitIR, library: KicadLibrary, params: RoutingParams | None, before: SIRules, new_si: SIConstraints, inner: bool,
        first: Routing, extra: dict | None = None,
    ) -> tuple[Routing, str]:
        """Route once more with the promoted classes' rules; ``(routing to apply, note)`` - the first pass's when nothing changed or the re-route is worse."""
        promoted_ir = board_ir.model_copy(update={"si": new_si})
        try:
            after = net_rules(promoted_ir, params)
        except CompileError as e:
            return first, f"si re-route: not attempted ({e}); the first pass's copper is kept"
        if after.signature() == before.signature():
            missing = "; ".join(after.notes) or "the promoted nets' class adds no routing rule"
            return first, f"si re-route: not needed - the promotions change no routing rule ({missing})"
        try:
            second = route_board(promoted_ir, library, params, rules=after.rules or None, inner_layers=inner, **(extra or {}))
        except (CompileError, LibraryLookupError, LibraryFormatError) as e:
            return first, f"si re-route refused ({e}); the first pass's copper is kept (si.impedance judges the promoted nets at the width they have)"
        lost = sorted(set(second.unrouted) - set(first.unrouted))
        if lost:  # never trade a routed net for a controlled width, not even on an opt-in partial board
            why = "; ".join(f"{n}: {second.unrouted[n]}" for n in lost)
            return first, f"si re-route left {len(lost)} net(s) unrouted ({why}); the first pass's copper is kept (si.impedance judges the promoted nets at the width they have)"
        widths = ", ".join(f"{n} {r.width_mm:g} mm" for n, r in sorted(after.rules.items()) if r.width_mm is not None and before.rules.get(n) != r)
        return second, f"si re-route: routed once more with the promoted rules ({widths or 'new budgets'}): {self._routing_description(second)}"

    def _with_planes(self, ir: CircuitIR, board: PCBDesign, notes: list[str]) -> PCBDesign:
        """``board`` with its stackup's plane zones (a 4-layer board the agent routes), unless it has them already (module docstring)."""
        zones, plan_notes = self._plane_plan(ir, board)
        if zones is None:
            notes.extend(plan_notes)
            return board
        clearance = ir.parameters.get(PLANE_CLEARANCE_KEY)
        notes.append("plane zones: " + ", ".join(dict.fromkeys(f"{z.net} on {z.layer}" for z in zones))
                     + f" (outline inset {float(clearance.value):g} mm; KiCad fills them)")
        notes.extend(plan_notes)
        return board.model_copy(update={"zones": [*board.zones, *zones]})

    @staticmethod
    def _plane_plan(ir: CircuitIR, board: PCBDesign) -> tuple[list[Zone] | None, list[str]]:
        """The plane zones :meth:`_with_planes` adds (clipped by the keep-outs), or ``None`` and why not; nothing is changed here."""
        stack = board.stackup
        if stack is None or not stack.plane_layers() or board.outline is None or any(_is_plane(z) for z in board.zones):
            return None, []
        names = {layer.name for layer in board.layers}
        missing = [c.name for c in stack.plane_layers() if c.name not in names]
        if missing:
            return None, [f"plane zones not added: ir.pcb.layers does not list the plane layer(s) {missing}"]
        clearance = ir.parameters.get(PLANE_CLEARANCE_KEY)
        if clearance is None:
            return None, [f"plane zones not added: no {PLANE_CLEARANCE_KEY} parameter says how far inside the board edge they end"]
        try:
            zones = plane_zones(stack, board.outline, clearance)
        except (ValueError, TypeError) as e:
            return None, [f"plane zones not added: {e}"]
        kos = keepouts_of(board)
        if not kos:
            return zones, []
        notes: list[str] = []
        clipped = _clip_planes(zones, kos, notes)
        return (clipped or None), notes

    @classmethod
    def _router_extras(cls, ir: CircuitIR, board: PCBDesign) -> dict:
        """The router's keep-outs and plane nets for a board with keep-outs or RF blocks (module docstring); ``{}`` for every other board."""
        kos = keepouts_of(board)
        if not kos and not rf_blocks(ir):
            return {}
        zones = [z for z in board.zones if _is_plane(z)] or (cls._plane_plan(ir, board)[0] or [])
        nets = {n.name for n in ir.nets}
        planes: dict[str, list[tuple[str, list[tuple[float, float]]]]] = {}
        for z in zones:
            if z.net in nets:
                planes.setdefault(z.net, []).append((z.layer, [(float(x), float(y)) for x, y in z.polygon]))
        return {"keepouts": kos, "plane_nets": dict(sorted(planes.items()))}

    @staticmethod
    def _si_rule_notes(si_rules: SIRules) -> list[str]:
        out = [f"si: {n}" for n in si_rules.notes]
        if si_rules.rules:
            per: dict[str, list[str]] = {}
            for net, rule in sorted(si_rules.rules.items()):
                per.setdefault(rule.net_class or "?", []).append(net)
            out.insert(0, "si rules: " + "; ".join(
                f"{cls} ({len(nets)} net(s)): " + " / ".join(si_rules.classes[cls].derivation) if cls in si_rules.classes else f"{cls}: {len(nets)} net(s)"
                for cls, nets in per.items()
            ))
        return out

    def _unrouted(
        self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign, notes: list[str], reasons: list[str], *, placed: bool, description: str, basis: str = GRID_RATIONALE,
    ) -> AgentResult:
        """The placement alone (or nothing when the board was already placed) plus why there is no copper; then the silkscreen.

        A board this run placed gets its stackup's plane zones here too (they are not routed copper).
        """
        if not placed:
            return self._with_silk(ir, ctx, board, None, [*notes, *reasons])
        plane_notes: list[str] = []
        board = self._with_planes(ir, board, plane_notes)
        proposal = IRProposal(description=description, target="pcb", operation="set", payload=board, rationale=f"{basis}; DRC decides validity")
        return self._with_silk(ir, ctx, board, proposal, [*notes, f"{description}; {UNROUTED_NOTE}", *reasons, *plane_notes])

    def _with_silk(
        self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign, proposal: IRProposal | None, notes: list[str], extra: list[IRProposal] | None = None,
    ) -> AgentResult:
        """Add the silkscreen to the one ``pcb`` proposal, or propose ``board`` + silkscreen alone when nothing else is proposed (module docstring).

        ``extra`` (the ``si`` promotion proposal) follows the ``pcb`` one.
        """
        extra = list(extra or [])
        silk_notes, texts, silk_description = self._silkscreen(ir, ctx, board)
        notes = [*notes, *silk_notes]
        if texts is None:
            return self._result(proposals=[*([proposal] if proposal is not None else []), *extra], notes=notes)
        payload = board.model_copy(update={"silkscreen": texts})
        if proposal is None:
            proposal = IRProposal(description=silk_description, target="pcb", operation="set", payload=payload, rationale=SILK_RATIONALE)
        else:
            proposal = proposal.model_copy(update={
                "payload": payload, "description": f"{proposal.description}; {silk_description}", "rationale": f"{proposal.rationale}; {SILK_RATIONALE}",
            })
        return self._result(proposals=[proposal, *extra], notes=notes)

    def _silkscreen(self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign) -> tuple[list[str], list | None, str]:
        """``(notes, texts or None, description)`` of the silkscreen for ``board``; ``None`` = propose none (the notes say why)."""
        notes: list[str] = []
        answer = ctx.answers.get(SILKSCREEN_KEY)
        if answer is not None:
            if answer.strip().lower() == SKIP_ANSWER:
                return (["silkscreen skipped by answer"] if board.placements else []), None, ""
            notes.append(f"{SILKSCREEN_KEY}={answer!r} not understood (the only answer is '{SKIP_ANSWER}'); placing the silkscreen as usual")
        if not board.placements:
            return notes, None, ""
        if board.silkscreen:
            return [*notes, f"silkscreen not placed: ir.pcb already has {len(board.silkscreen)} silkscreen text(s); the agent never replaces silkscreen"], None, ""
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return [*notes, "silkscreen not placed: no KiCad library in ctx.tools['kicad_library']; pad and silk geometry cannot be read"], None, ""
        try:
            placed = place_silkscreen(ir.model_copy(update={"pcb": board}), library, self.silk)
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return [*notes, f"silkscreen not placed: {e}"], None, ""
        description = placed.description()
        return [*notes, description, *placed.notes], list(placed.texts), description

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
            f"not applied: {ROUTER_ID} {routing.version} connected {s['routed_nets']} of {total} net(s) ({s['track_count']} track(s), {s['via_count']} via(s), "
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
            f"{ROUTER_ID} {routing.version}: {s['routed_nets']} net(s) routed on F.Cu/B.Cu, {s['track_count']} track(s), {s['via_count']} via(s), "
            f"{s['total_length_mm']} mm of copper, {s['iterations']} negotiation iteration(s); grid {p.grid_mm} mm, width {p.track_width_mm} mm, "
            f"clearance {p.clearance_mm} mm, via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
        if p.rules is not None:
            text += f"; {p.rules} rules: pad pitch {p.pad_pitch_mm} mm ({p.pitch_footprint}) is below {FINE_PITCH_MM} mm"
        if s["skipped_nets"]:
            text += f"; {len(s['skipped_nets'])} net(s) with fewer than 2 pads skipped ({', '.join(s['skipped_nets'])})"
        if s["raised"]:
            text += "; raised to ir.pcb.manufacturing minimums: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in s["raised"].items())
        if routing.rules:
            text += f"; {len(routing.rules)} net(s) with net-class rules"
        if s.get("inner_layers"):
            text += f"; inner layers {', '.join(s['inner_layers'])} carry planes, not tracks"
        if s.get("keepouts"):
            text += f"; keep-outs {', '.join(k['id'] for k in s['keepouts'])} are obstacles"
        joined = {n: r for n, r in (s.get("plane_nets") or {}).items() if "vias" in r}
        if joined:
            text += "; plane net(s) " + ", ".join(
                f"{n} ({r['vias']} pad via(s), {len(r['through_hole'])} through-hole pad(s)) to {'+'.join(r['planes'])}" for n, r in joined.items()
            ) + " - joined through the plane fill, not by tracks"
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


def _is_plane(zone) -> bool:
    """Whether ``zone`` is a stackup plane this agent drew (:func:`ai_eda.design.stackup.plane_zones`), not routed copper."""
    return zone.provenance.tool == STACKUP_TOOL


def _clip_planes(zones: list[Zone], keepouts: list, notes: list[str]) -> list[Zone]:
    """Each plane zone minus the keep-outs that forbid zones on its layer and do not allow its net (module docstring).

    The zone's rectangle minus each keep-out's bounding box (exact for a
    rectangle keep-out, larger than a polygon keep-out - the plane loses
    more, never less) as simple polygons
    (:func:`~ai_eda.tools.keepout.rect_difference`); every part keeps the
    stackup provenance tool (it is still that plane) with the keep-outs
    named in ``derived_from`` and the note.
    """
    out: list[Zone] = []
    for z in zones:
        cuts = [k for k in keepouts if forbids(k, "zones") and covers_layer(k, z.layer) and z.net not in allowed_nets(k)]
        if not cuts:
            out.append(z)
            continue
        ids = [keepout_id(k) for k in cuts]
        xs, ys = [p[0] for p in z.polygon], [p[1] for p in z.polygon]
        parts = rect_difference((min(xs), min(ys), max(xs), max(ys)), [area_bbox(area_points(k)) for k in cuts])
        by_box = [kid for kid, k in zip(ids, cuts) if not area_is_rect(k)]
        words = f"clipped by keep-out(s) {', '.join(ids)}" + (f" ({', '.join(by_box)} by its bounding box)" if by_box else "")
        if not parts:
            notes.append(f"plane zone {z.net} on {z.layer} not added: keep-out(s) {', '.join(ids)} forbid all of it")
            continue
        prov = z.provenance.model_copy(update={
            "derived_from": [*z.provenance.derived_from, *(f"pcb.keepouts[{kid}]" for kid in ids)],
            "note": f"{z.provenance.note}; {words}",
        })
        out.extend(z.model_copy(update={"polygon": [(_q(x), _q(y)) for x, y in poly], "provenance": prov}) for poly in parts)
        notes.append(f"plane zone {z.net} on {z.layer} {words}: {len(parts)} polygon(s)")
    return out


def _routes_around_planes(board: PCBDesign) -> bool:
    """Whether ``board``'s inner layers are exactly its stackup's plane layers: then the router may accept them (it never routes them)."""
    inner = [layer.name for layer in board.layers if INNER_LAYER_RE.match(layer.name)]
    if not inner or board.stackup is None:
        return False
    planes = {c.name for c in board.stackup.plane_layers()}
    return set(inner) <= planes


__all__ = [
    "GRID_RATIONALE", "PARTIAL_ANSWER", "PCBAgent", "PLACEMENT_KEY", "RING_RATIONALE", "ROUTING_KEY", "SILKSCREEN_KEY", "SILK_RATIONALE", "SKIP_ANSWER",
    "UNROUTED_NOTE",
]
