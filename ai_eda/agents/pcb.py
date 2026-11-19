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
  otherwise, and a board with more than ``LARGE_BOARD_NETS`` nets to route
  negotiates up to ``LARGE_BOARD_ITERATIONS`` iterations instead of 40. The
  fine rules and the pitch that chose them, and a large board's net count,
  are in every track's ``params:`` provenance entry and in the notes; a
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
  silkscreen is never replaced (``silkscreen not placed: ...`` note). The
  one exception is a board whose every silk text is a template's board text
  (:func:`ai_eda.design.board.template_silk_text`: a ``user`` / ``title``
  text stamped ``design.template.<id>``, a key's note name) or a design
  file's (:func:`ai_eda.design.board.design_file_silk_text`: its ``board``
  section's texts, stamped as every value of the file): those texts
  stay where the template (the file) put them, ahead of the placed ones, and are
  keep-outs for the references, pin labels and title the placer adds
  (``silkscreen.place`` 0.2; a template title means no second title; the
  notes say ``design file`` instead of ``template`` for a file's texts). On a
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
  packer - never the grid for "the rest"; a part whose pad pitch is below
  ``FANOUT_PITCH_MM`` keeps every other part the router's escape room away
  from its pads, computed from the routing parameters this agent routes
  with: ``placement.rf_floorplan`` 0.2 - parameters the router refuses are
  still a ``not routed:`` note under the placement, never ``not placed:``);
  every other IR keeps the choice
  above. A board whose design fixes parts or names pack regions
  (``ir.pcb.fixed`` / ``ir.pcb.pack_regions``, a template's keyboard row) is
  placed by :func:`ai_eda.tools.placement.fixed.fixed_placement` before any
  of the choices above (``placement.fixed``: the fixed parts copied as given,
  each region's parts packed in its list order, the rest in the free board
  area; it refuses rather than moves a fixed part, and an IR with RF blocks
  too is refused, ``not placed:``). On such a board the answer
  ``pcb.placement_policy=functional`` keeps its meaning (the functional placer
  refuses fixed parts and pack regions: ``not placed:``), and a stated
  ``pcb.placement_policy=legacy`` - the grid / core-ring / RF placers, none of
  which reads them - is refused with that reason (``not placed:``) instead of
  dropping the design's positions; ``pcb.placement=skip`` still proposes
  nothing. Every board without fixed parts or pack regions is placed exactly
  as before. A board with keep-outs (``ir.pcb.keepouts``) or RF blocks hands the
  router its keep-outs and its plane nets (routing.maze 0.4: keep-out
  obstacles, each plane net's SMD pads joined to the plane by a via each,
  never by tracks - the nets of the plane zones the board has or gets here),
  and its plane zones are clipped by every keep-out that forbids zones on
  their layer (the rectangle minus the keep-outs' boxes, a polygon keep-out
  by its bounding box - conservative; one outline with cut-outs per connected part, same
  provenance tool so they stay the stackup's planes, the keep-outs named in
  ``derived_from``). Every other board is routed and planed exactly as
  before.
* A board whose stack has one copper layer on a metal base (an aluminium
  IMS, :mod:`ai_eda.design.materials`) gets keep-out ``ims_bcu`` over its
  outline on ``B.Cu`` (tracks, vias, pads, zones;
  :func:`~ai_eda.design.materials.single_layer_keepout`, ``derived`` from the
  stack's metal base and the outline) before it is routed, unless it has it
  already: the KiCad board keeps ``B.Cu`` (an even copper count) and the
  routing.maze 0.4 keep-outs make the router use ``F.Cu`` only. Every other
  board is unchanged.
* **Layer escalation** (``design.stackup.escalation`` 0.1, the project owner's
  rule :data:`ESCALATION_RULE`). When the router leaves nets unrouted and at
  least one was ripped up in negotiation (congestion: a statically fenced pad
  never escalates), the agent routes the same placement again at each count of
  :data:`~ai_eda.design.stackup.LAYER_ESCALATION` above the board's (4, 6, 8,
  10, 12; the escalation stacks, :func:`~ai_eda.design.stackup.escalation_stackup`,
  with their plane zones, routing.maze 0.9 with typed vias) and keeps the
  first at which every net routes: the payload carries that stack, its layers
  and zones, and the stack's provenance (``derived`` by the escalation tool)
  quotes the rule and every count's record - routed (its router version,
  result and iterations), skipped or refused (its reason). The stage notes
  carry one ``layer escalation:`` line per count, with its wall time (read
  from :data:`_clock`; never in the IR), right after the routing description;
  when no count routes every net the board's own result stands (all or
  nothing, or ``pcb.routing=partial``) and the lines follow the per-net
  reasons. The board's own count is routed exactly as before (no stack), so a
  board that routes keeps its copper and notes byte for byte. Which boards
  climb: a board with no stack and the default ``F.Cu`` / ``B.Cu`` layers (a
  design file, a bare IR) whose requirements state no material but FR-4, or
  with the stack its template chose, whose layer
  count no requirement states, with exactly one net of kind ground (the ground
  planes; the power planes carry the power net with the most pins, else the
  ground net); a restricting template policy keeps its counts; a material
  other than FR-4 climbs only its own stacks
  (:data:`~ai_eda.design.materials.MATERIAL_STACKS`: RO4350B 2 -> 4) and
  stops past them; ``ir.pcb.manufacturing.layer_count_options`` skips the
  counts it does not list. The escalated stack serves the requirements the
  template's stack served (its material keys stay traced). A stated count is
  never escalated: the note names
  the next count and how to ask for it. A board whose parts need more layers
  than it has starts at the first such count: a part's
  ``Component.min_pcb_layers`` with its source document, and the escape depth
  of an area array (:data:`NXP_ESCAPE_SOURCE` under the channel check
  :data:`TI_CHANNEL_SOURCE`) mapped through :data:`LADDER_SIGNALS`; the reason
  and the source are in the notes and the provenance. When no count from there
  routes every net, the first routed one is the result - all or nothing: only
  ``pcb.routing=partial`` applies it, otherwise the board's own placement is
  proposed (never a stack without its copper) and the partial answer repeats
  the climb; when none from there can be routed at all, the board's own count
  is routed and climbs like any board (its kept count's record says it is
  below the parts' minimum). The trigger is the
  owner's - any net left unrouted - not In-Circuit Design's 85 % completion
  rule (AN2011_2, research_synthesis.json), which is stated nowhere in code.
  It runs here only, never in REPAIR.
* **Package start** (:mod:`ai_eda.design.package_layers` 0.1, the owner's rule
  :data:`~ai_eda.design.package_layers.PACKAGE_RULE`). The board's parts are
  matched against the sourced package / part table: each part's start (the
  convention) and floor (the manufacturer's recommendation or minimum), the
  board's the largest of them, the P6 minimums above (a sourced
  ``min_pcb_layers``, the escape-depth estimate) as floors, rounded up to the
  ladder. Only a board some part of which matches a row with a start above 2
  or a floor, whose layer count no requirement states, which may climb at
  all (the rules above: :meth:`PCBAgent._ladder_plan`) and whose ladder
  reaches the floor, is searched this way (a start above the ladder's last
  count - a material's last stack, a template's last count - is capped
  there, with a note); every other board is routed exactly as before, the
  P6 start included, and a board with no notes from this module keeps its
  notes too. The search routes the start; when nets stay unrouted it climbs
  one count at a time as above; when every net routes at the first count
  routed from the start it removes layers - the next lower count of the
  ladder (a material: its own stacks), again while every net routes and the
  count stays at or above the floor - and keeps the fewest that routes every
  net. When no count from the start up can be built (skipped by
  ``layer_count_options`` or refused), the search starts at the highest count
  below the start, at or above the floor, that can be. Every count, the
  board's own included, is checked against ``layer_count_options``. The
  board's own count is routed by today's router exactly (no stack), so a
  board that ends there keeps today's copper and stack byte for byte; only
  its notes carry the search. A kept stack is stamped
  ``design.package_layers`` 0.1 and its provenance names the start's and the
  floor's rows with their sources and every count tried, up or down; the
  notes carry one ``layer escalation:`` line per count (``fewer layers:`` on
  the way down), the rows, and the conditional rows skipped because the
  requirements do not state their condition. A stated count is never
  searched: the note names the floor's source and quote when the stated count
  is below it. When no count at or above the floor can be built at all, or
  the ladder does not reach the floor, the board is routed as without the
  search: the P6 start above, then its own count, which may climb as above.
* **Corner style** (the project owner's rule of 2026-11-13,
  :mod:`ai_eda.tools.routing.corners`): every routing this agent runs - the
  board's own count, each count of the layer escalation and the package
  search, the SI re-route - passes ``corner_style="arc"``
  (:data:`~ai_eda.tools.routing.corners.DEFAULT_CORNER_STYLE`), so after
  the router has settled its sidesteps (a short track between two
  right-angle corners) are straightened where the copper may legally run
  without them, and each right-angle corner is rounded into an arc tangent
  to its two tracks - a track with ``mid`` - or, where no arc fits, mitred
  into a 45-degree chamfer, wherever every clearance allows
  (routing.corners 0.2: a geometry convention, not an electrical
  requirement; a sidestep or corner it cannot change legally stays as
  routed, so it never leaves a net unrouted). The routing description
  counts the corners rounded, the corners mitred and the sidesteps
  straightened. A board that holds an arc is judged by the arc-aware
  checks, each stamped with its own arc version (``pcb.routing`` /
  ``pcb.keepout`` 0.3, ``si`` 0.3, ``si.crosstalk`` 0.3, ``spice.xtalk``
  decks 0.2). ``--answer pcb.corners=miter45`` passes
  ``corner_style="miter45"`` instead - every corner mitred into a
  45-degree chamfer, the 0.1 copper (routing.corners 0.1, the same
  convention; the description then counts the corners mitred) - and
  ``--answer pcb.corners=orthogonal`` keeps the router's grid corners (the
  copper before the rule). The "exactly as before" / "byte for byte"
  comparisons above hold for one corner style.
* It asks no question. The only steering is the four control keys
  :data:`PLACEMENT_KEY` (``--answer pcb.placement=skip`` proposes nothing),
  :data:`ROUTING_KEY` (``--answer pcb.routing=skip`` proposes the
  placement without copper, ``--answer pcb.routing=partial`` applies the
  routed nets of a board the router could not finish),
  :data:`SILKSCREEN_KEY` (``--answer pcb.silkscreen=skip`` proposes no
  silkscreen) and :data:`CORNERS_KEY` (``--answer pcb.corners=orthogonal``
  keeps right-angle corners, ``--answer pcb.corners=miter45`` mitres them, ``--answer pcb.corners=arc`` - the default - rounds them); any
  other value is noted as not understood and the work
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

import math
import time
from dataclasses import dataclass, field

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.agents.keys import CORNERS_KEY, PACKING_KEY, PLACEMENT_KEY, PLACEMENT_POLICY_KEY, ROUTING_KEY, SILKSCREEN_KEY
from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.design.base import TOOL_ID as TEMPLATE_TOOL_ID
from ai_eda.design.base import leave_out_answer
from ai_eda.design.board import PLANE_CLEARANCE_KEY, design_file_silk_text, proposed_silk_text
from ai_eda.design.inputs import LAYER_COUNT_KEY, LAYER_COUNT_OPTIONS, MATERIAL_KEY, LayerCountInput, read_layer_count
from ai_eda.design.materials import (
    DEFAULT_MATERIAL,
    MATERIAL_STACKS,
    MATERIALS,
    MATERIALS_TOOL,
    MATERIALS_VERSION,
    MaterialDecision,
    material_escalation_stackup,
    material_stack_description,
    select_material,
    single_layer_keepout,
)
from ai_eda.design.package_layers import PACKAGE_LAYERS_TOOL, PACKAGE_LAYERS_VERSION, PACKAGE_RULE, BoardLayers, plan_for_ir
from ai_eda.design.stackup import (
    ESCALATION_TOOL,
    ESCALATION_VERSION,
    LAYER_ESCALATION,
    PLANE_EDGE_CLEARANCE_MM,
    STACKUP_TOOL,
    board_layers,
    escalation_description,
    escalation_refusal,
    escalation_stack,
    escalation_stackup,
    plane_description,
    plane_zones,
    via_structures,
)
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, NetKind, PCBDesign, Provenance, ProvenanceKind, SIConstraints, Traced, Zone
from ai_eda.llm.router import TaskKind
from ai_eda.tools.keepout import allowed_nets, area_bbox, area_is_rect, area_points, covers_layer, forbids, keepout_id, keepouts_of, rect_contours
from ai_eda.tools.kicad.geometry import _q
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.core_ring import CORE_MIN_PADS, RingPlacement, core_ring_placement, find_core
from ai_eda.tools.placement.core_ring import PLACER_ID as RING_PLACER_ID
from ai_eda.tools.placement.core_ring import PLACER_FANOUT_VERSION as RING_FANOUT_VERSION
from ai_eda.tools.placement.core_ring import PLACER_VERSION as RING_PLACER_VERSION
from ai_eda.tools.placement.grid import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, _resolve, grid_placement
from ai_eda.tools.placement.fixed import PLACER_ID as FIXED_PLACER_ID
from ai_eda.tools.placement.fixed import fixed_placement
from ai_eda.tools.placement.functional import functional_placement
from ai_eda.tools.placement.rf_floorplan import RING_MM, rf_blocks, rf_floorplan_placement
from ai_eda.tools.routing.corners import ARC, CORNER_STYLES, CORNERS_ID, DEFAULT_CORNER_STYLE, ORTHOGONAL
from ai_eda.tools.routing.maze import (
    FINE_PITCH_MM,
    INNER_LAYER_RE,
    LARGE_BOARD_NETS,
    LAYERS,
    ROUTER_ID,
    Routing,
    RoutingParams,
    effective_params,
    footprint_pad_pitch,
    route_board,
)
from ai_eda.tools.routing.stack import RouterStack
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
#: the rationale of a proposal whose placement keeps the design's fixed parts and pack regions
FIXED_RATIONALE = (
    "the design's fixed parts copied as given, each pack region's parts shelf-packed in its list order and the rest in the free board area, "
    "from verified library footprint extents; placed extents are pairwise disjoint, inside their regions and inside the outline"
)
#: the rationale of a proposal whose placement is the RF floorplan
RF_RATIONALE = (
    "RF floorplan placement from verified library footprint extents: each block shelf-packed in its region in chain order, "
    "shield cans first with their block's parts inside the fence less the ring, keep-outs that ban footprints / pads honoured; "
    "placed extents are pairwise disjoint (a can's own parts inside its fence), inside their regions and inside the outline"
)

# --------------------------------------------------------------------------- the layer escalation (module docstring)

#: the project owner's layer-escalation rule as every escalated stack's provenance quotes it
ESCALATION_RULE = (
    "2026-11-12: when nets stay unrouted, try the board again at 4, 6, 8, 10 and 12 copper layers - up to 10-12 layers - and keep the first "
    "count at which every net routes; a count the user stated is never escalated; 2026-11-13: a board whose parts need 6 or more layers "
    "starts at that count"
)
#: the escape rule the area-array estimate applies (its premise: one trace between adjacent lands)
NXP_ESCAPE_SOURCE = (
    "NXP AN10778 Rev. 2 (15 Apr 2011) sec. 5.2: with one trace between adjacent lands the outer two rows escape on the mounting layer and "
    "each further signal layer escapes two more rows"
)
#: the channel check that decides whether that premise holds
TI_CHANNEL_SOURCE = "TI SPRU811A (May 2005) p. 21, 24-27: one trace fits between two lands or via pads when g = pitch - d >= 2s + w"
#: whether the two documents above were opened
ESCAPE_SOURCES_OPENED = "both opened for the multilayer plan's research (research_synthesis.json rules 5 and 6), not re-opened here"
#: how far an area-array pad centre may lie off its lattice point (mm): KiCad's 1 nm resolution
LATTICE_TOLERANCE_MM = 1e-6
#: the plane zones' inset on a board the escalation gives planes, unless the IR already has a ``plane_edge_clearance`` parameter
ESCALATION_PLANE_CLEARANCE = Traced(
    value=PLANE_EDGE_CLEARANCE_MM, unit="mm",
    provenance=Provenance(kind=ProvenanceKind.ASSUMPTION, note=(
        "plane zones end this far inside the board edge: the generic stacks' plane edge clearance, chosen by the layer-escalation rule")),
)
#: the signal layers each count of the ladder has (2: F.Cu / B.Cu; 4: the same two, the inner layers are planes; 6 - 12: the escalation stacks')
LADDER_SIGNALS: dict[int, int] = {n: 2 if n == 2 else len(escalation_stack(n).signal_layers()) for n in LAYER_ESCALATION}
#: the wall clock each rung's routing time is read from - stage notes only, never the IR (the copper is a pure function of its inputs)
_clock = time.perf_counter


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
        policy = ctx.answers.get(PLACEMENT_POLICY_KEY, "legacy").strip().lower()
        if policy not in ("legacy", "functional"):
            return self._result(notes=[*notes, f"not placed: unknown {PLACEMENT_POLICY_KEY}={policy!r}; choose legacy or functional"])
        if policy == "functional":
            try:
                plan = functional_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, routing=self.routing)
            except (CompileError, LibraryLookupError) as e:
                return self._result(notes=[*notes, f"not placed: {e}"])
            expanded = ir.model_copy(update={"components": [*ir.components, *plan.components]})
            placed = base.model_copy(update={"outline": plan.outline, "placements": plan.placements,
                                             "keepouts": [*base.keepouts, *plan.keepouts]})
            result = self._route(expanded, ctx, placed, [*notes, *plan.notes], placed=True,
                                 description=plan.description(), basis="functional pin associations, hard fan-out reservations and bounded outline search")
            additions = [IRProposal(description=f"corner mounting hole {c.ref} (size UNVERIFIED)", target="components", operation="append",
                                    payload=c, rationale=c.description) for c in plan.components]
            return result.model_copy(update={"proposals": [*additions, *result.proposals]})
        if base.fixed or base.pack_regions:  # the design's own positions (module docstring): placement.fixed, before every other choice
            if PLACEMENT_POLICY_KEY in ctx.answers:  # only a stated 'legacy' reaches here
                return self._result(notes=[*notes, (
                    f"not placed: {PLACEMENT_POLICY_KEY}={policy!r} places by the grid / core-ring / RF placers, which do not read the "
                    f"{len(base.fixed)} fixed part(s) and {len(base.pack_regions)} pack region(s) in ir.pcb; {FIXED_PLACER_ID} places them - leave "
                    f"{PLACEMENT_POLICY_KEY} unanswered")])
            packing = ctx.answers.get(PACKING_KEY, "left").strip().lower()
            if packing not in ("left", "spread"):
                notes.append(f"{PACKING_KEY}={packing!r} not understood (left or spread); packing from the left as usual")
            try:
                fixed = fixed_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, spread=packing == "spread")
            except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
                return self._result(notes=[*notes, f"not placed: {e}"])
            placed = base.model_copy(update={"outline": fixed.outline, "placements": list(fixed.placements)})
            return self._route(ir, ctx, placed, notes, placed=True, description=fixed.description(origin, self.spacing_mm, self.margin_mm),
                               basis=FIXED_RATIONALE)
        try:
            if rf_blocks(ir):
                plan = rf_floorplan_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, outline=base.outline, keepouts=keepouts_of(base),
                                              routing=self.routing)
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
            f"{RING_PLACER_ID} {RING_PLACER_VERSION if ring.fanout is None else RING_FANOUT_VERSION}: {len(ring.placements)} component(s) on a "
            f"{o.width_mm} x {o.height_mm} mm "
            f"{origin} outline at ({o.origin_x_mm}, {o.origin_y_mm}), core {ring.core} ({ring.core_pads} pads) at ({centre.x_mm}, {centre.y_mm}), "
            f"{inner} part(s) on the inner ring, {outer} on the outer ring, spacing {self.spacing_mm} mm, margin {self.margin_mm} mm"
            + (f"; no other part within {ring.fanout[0]:g} mm of {ring.core}'s pads (its fan-out room)" if ring.fanout is not None else "")
        )

    def _route(
        self, ir: CircuitIR, ctx: AgentContext, board: PCBDesign, notes: list[str], *, placed: bool, description: str, basis: str = GRID_RATIONALE,
    ) -> AgentResult:
        """Route ``board`` (placements, no copper) and build the one proposal; ``placed`` says whether this run placed it (``basis``: how)."""
        board = self._with_single_layer_keepout(board, notes)
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
        corners = self._corner_style(ctx, notes)
        board_ir = ir.model_copy(update={"pcb": board})  # a shallow copy: ir itself is never touched
        inner = _routes_around_planes(board)
        router_extra = {**self._router_extras(ir, board), "corner_style": corners}
        params: RoutingParams | None = None
        si_rules = SIRules()
        climb: _Climb | None = None
        esc_notes: list[str] = []
        routing: Routing | None = None

        def own_route() -> tuple[Routing, float]:
            """The board at its own count, routed exactly as before any escalation (no stack), and its wall time."""
            started = _clock()
            done = route_board(board_ir, library, params, rules=si_rules.rules or None, inner_layers=inner, **router_extra)
            return done, _clock() - started

        try:
            params = self.routing if self.routing is not None else RoutingParams.for_board(board_ir, library)
            si_rules = net_rules(board_ir, params)
            # the layer escalation (module docstring): the parts' minimum count first - a board whose parts need more layers than it has starts higher
            minimum = _parts_minimum(board_ir, library, params)
            esc_notes = [f"layer escalation: {n}" for n in minimum.notes]
            # the package start (module docstring): only a board whose parts match a row with a start above 2 or a floor; a plan that does
            # not apply says nothing (the board's notes stay as they were)
            package = plan_for_ir(board_ir, library, minimum.floors())
            searched = False  # the package search settled the board's count; otherwise the P6 start below runs as before
            if package.applies:
                esc_notes += [f"layer escalation: {n}" for n in package.notes]
                if package.refusal is not None:
                    esc_notes.append(f"layer escalation: the package start is not applied - {package.refusal}")
            if package.applies and package.refusal is None and package.start > _copper_count(board):
                esc_notes += _package_lines(package)
                plan, why, silent = self._ladder_plan(ir, board)
                if plan is not None and package.floor <= max(plan.rungs):  # the floor is reachable; a start above the ladder is capped
                    search = self._search(ir, board, library, params, plan, package, own_route, corners=corners)
                    if search.own is not None:  # every net routes on the board's own count: today's routing, the search in the notes
                        routing, seconds = search.own, search.own_seconds
                        esc_notes += search.notes
                        searched = True
                    elif search.rung is None:  # no count at or above the floor could be built: the board is routed as without the search
                        esc_notes += search.notes
                    else:
                        climb, searched = search, True
                else:
                    esc_notes += _package_not_applied(ir, package, why if plan is None else f"no count of the ladder ({', '.join(map(str, plan.rungs))}) "
                                                      f"reaches the package floor of {package.floor}")
            if not searched and minimum.layers > _copper_count(board):
                plan, why, silent = self._ladder_plan(ir, board)
                if plan is not None and any(n >= minimum.layers for n in plan.rungs):
                    climb = self._climb(ir, board, library, params, plan, None, 0.0, minimum, corners=corners)
                    if climb.rung is None:  # no count from the minimum up was routed: the board's own count is routed, and may still climb
                        esc_notes += climb.notes
                        climb = None
                else:
                    esc_notes.append(f"layer escalation: the parts need at least {minimum.layers} copper layers ({'; '.join(minimum.reasons)}); not applied - "
                                     + (why if plan is None else f"no count of the ladder ({', '.join(map(str, plan.rungs))}) has that many"))
            if climb is None and routing is None:
                routing, seconds = own_route()
        except (CompileError, LibraryLookupError, LibraryFormatError) as e:
            rules = [self._rules_note(params)] if params is not None and params.rules is not None else []
            return self._unrouted(ir, ctx, board, notes, [*rules, *self._si_rule_notes(si_rules), f"not routed: {e}", *esc_notes],
                                  placed=placed, description=description, basis=basis)
        if climb is None and routing.unrouted and set(routing.unrouted) & set(routing.stats.get("dropped") or ()):
            # nets stay unrouted and at least one was ripped up in negotiation (congestion; a static fence never escalates)
            plan, why, silent = self._ladder_plan(ir, board)
            if plan is not None:
                climb = self._climb(ir, board, library, params, plan, routing, seconds, minimum, corners=corners)
            elif not silent:
                esc_notes.append(f"layer escalation: not attempted - {why}")
        escalated: int | None = None
        chosen_by = "the project owner's layer-escalation rule"
        if climb is not None:
            chosen_by = climb.rule
            esc_notes += climb.notes
            r = climb.rung
            if r is not None and (not r.routing.unrouted or (partial and r.routing.stats.get("routed_nets"))):
                # the kept count, or the parts' first count when none routes every net and pcb.routing=partial applies its nets: its board,
                # copper and rules
                board, board_ir, routing, si_rules, router_extra, inner, escalated = r.board, r.board_ir, r.routing, r.si_rules, r.extra, False, r.n
                esc_notes += climb.board_notes
            elif r is not None:
                # the parts' first count left nets unrouted and nothing of it applies: its result is reported under the board's own
                # placement - its stack, layers and planes never enter the IR without its copper (the partial answer repeats the climb)
                routing, si_rules = r.routing, r.si_rules
        si_notes = self._si_rule_notes(si_rules)
        reasons = [f"not routed: {net}: {why}" for net, why in routing.unrouted.items()]
        if routing.unrouted and not (partial and routing.stats.get("routed_nets")):
            rules = ([self._rules_note(routing.params, routing.stats.get("raised"), (routing.stats.get("kicad_minimums") or {}).get("raised"))]
                     if routing.params.rules is not None else [])
            dropped = [self._dropped_note(routing)] if routing.stats.get("routed_nets") else []
            return self._unrouted(ir, ctx, board, notes, [*rules, *si_notes, *dropped, *reasons, *esc_notes],
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
        on = "/".join(routing.stats["stack"]["signal_layers"]) if "stack" in routing.stats else "F.Cu/B.Cu"
        # a board the layer escalation gave its count says so (one routed without a stack keeps its rationale byte for byte)
        chosen = "" if escalated is None else f"; layer count {escalated} by {chosen_by} (see ir.pcb.stackup provenance)"
        if placed:
            notes.append(description)
            full = f"{description}; {routed}"
            rationale = (
                f"{basis}; {which} maze-routed on {on} from the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity{chosen}"
            )
        else:
            full = routed
            rationale = f"{which} maze-routed on {on} from the existing placements and the library pad geometry at the recorded width / clearance / via sizes; DRC decides validity{chosen}"
        notes.append(routed)
        if corners == ORTHOGONAL:  # routing.corners did not run (module docstring: corner style)
            notes.append(f"corners: {CORNERS_KEY}={ORTHOGONAL} - the router's right-angle grid corners are kept ({CORNERS_ID} does not run)")
        if not routing.unrouted:  # the escalation's record follows the routing description
            notes.extend(esc_notes)
        notes.extend(si_notes)
        if routing.unrouted:  # the opt-in partial board: whole nets only, the rest named
            left = self._partial_note(routing)
            full = f"{full}; {left}"
            notes += [left, *reasons, *esc_notes]
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

    # ----------------------------------------------------------------------- the layer escalation (module docstring)

    @staticmethod
    def _ladder_plan(ir: CircuitIR, board: PCBDesign) -> tuple[_LadderPlan | None, str, bool]:
        """``(plan, "", False)`` when the escalation may climb from ``board``; else ``(None, why, silent)`` - ``silent``: no note on a failure.

        In this order: a board without a stack must have the default
        ``F.Cu`` / ``B.Cu`` layers, and its confirmed requirements must state
        no material but FR-4 (:func:`~ai_eda.design.materials.select_material`
        untouched or FR-4: no stage builds a material stack for a board no
        template made, and no FR-4 stack is swapped in for the material); a
        board with a stack must carry the one its template chose (silent
        otherwise: the IR's own stack is never replaced); the ladder is the
        FR-4 counts of :data:`LAYER_ESCALATION` above the board's, or its
        material's own stacks (:data:`~ai_eda.design.materials.MATERIAL_STACKS`),
        within a restricting template policy's counts; a count the user stated
        is never escalated (the note names the next count and how to ask for
        it); the board needs exactly one net of kind ground (its planes); the
        power planes take the power net with the most pins (ties: the name),
        or the ground net. The plan carries the requirements the board's stack
        serves (:meth:`~ai_eda.ir.Stackup.served_requirements`): the escalated
        stack serves them too.
        """
        stack = board.stackup
        own = _copper_count(board)
        template = None
        if stack is None:
            names = [layer.name for layer in board.layers]
            if names != list(LAYERS):
                return None, f"the board's copper layers {names} are not the default F.Cu/B.Cu and it has no stack", True
            stated = _stated_material(select_material(ir))
            if stated is not None:
                return None, stated, False
        else:
            template = _template_of(ir)
            if template is None or f"template {template.id} v" not in (stack.provenance.note or ""):
                return None, "the board carries its own stack (no template chose it), which the escalation never replaces", True
        material = stack.material.id if stack is not None and stack.material is not None else DEFAULT_MATERIAL
        if material == DEFAULT_MATERIAL:
            rungs = [n for n in LAYER_ESCALATION if n > own]
        else:
            rungs = sorted(n for m, n in MATERIAL_STACKS if m == material and n > own)
        policy = template.layer_policy if template is not None else None
        if policy is not None and policy.restricts:
            rungs = [n for n in rungs if n in policy.allowed]
        layers, unusable = read_layer_count(ir)
        if layers is None or not layers.is_default:
            return None, _stated_count_note(layers, unusable, own, rungs, template is not None), False
        if not rungs:
            if material != DEFAULT_MATERIAL:
                counts = sorted(n for m, n in MATERIAL_STACKS if m == material)
                return None, (f"the {MATERIALS[material].name} board has no stack of its material above {own} copper layer(s) (materials.py: {counts}); "
                              "no FR-4 stack is swapped in"), False
            if policy is not None and policy.restricts:
                return None, f"template {template.id} builds only {policy.allowed_text()} layers: {policy.reason.strip()}", False  # type: ignore[union-attr]
            return None, f"the board already has {own} copper layers, the ladder's last count", False
        grounds = [net.name for net in ir.nets if net.kind == NetKind.GROUND]
        if len(grounds) != 1:
            return None, f"{len(grounds)} net(s) of kind ground ({', '.join(grounds) or 'none'}); the plane layers need exactly one", False
        powers = sorted((net for net in ir.nets if net.kind == NetKind.POWER), key=lambda net: (-len(net.pins), net.name))
        reason = stack.material.reason if stack is not None and stack.material is not None else ""
        served = tuple(stack.served_requirements()) if stack is not None else ()
        return _LadderPlan(own, tuple(rungs), grounds[0], powers[0].name if powers else None, material, reason, served), "", False

    def _climb(
        self, ir: CircuitIR, board: PCBDesign, library: KicadLibrary, params: RoutingParams, plan: _LadderPlan, first: Routing | None, seconds: float,
        minimum: _Minimum, *, corners: str = DEFAULT_CORNER_STYLE,
    ) -> _Climb:
        """Route ``board`` at each count of ``plan`` in turn and keep the first at which every net routes (module docstring: layer escalation).

        ``first`` is the board's own routing (``seconds`` its wall time), or
        ``None`` when the parts' ``minimum`` skips the board's own count - then
        the counts below the minimum are skipped too; a climb from the board's
        own count (``first`` given: no count from the minimum up was routed, or
        the ladder has none) tries every count above it.
        Every count is recorded - routed (its result and time), skipped (below
        the parts' minimum, not in ``layer_count_options``, refused by
        :func:`~ai_eda.design.stackup.escalation_refusal`) or not routed (a
        refusal, its reason) - in the notes and in the kept stack's provenance.
        """
        attempts: list[str] = []
        notes: list[str] = []

        def record(text: str, took: float | None = None) -> None:
            attempts.append(text)
            notes.append(f"layer escalation: {text}" + (f" ({took:.1f} s wall clock)" if took is not None else ""))

        started = first is None
        below = f"skipped - below the parts' minimum of {minimum.layers} copper layers"
        if first is not None:
            record(_attempt(plan.own, board, None, first), seconds)
        else:
            record(f"{plan.own} copper layers: {below}")
        opts = board.manufacturing.layer_count_options
        clearance = ir.parameters.get(PLANE_CLEARANCE_KEY) or ESCALATION_PLANE_CLEARANCE
        stands: _Rung | None = None
        for n in plan.rungs:
            if started and n < minimum.layers:
                record(f"{n} copper layers: {below}")
                continue
            skip = escalation_refusal(n) if plan.material == DEFAULT_MATERIAL else None
            if skip is None and opts is not None and n not in list(opts.value):
                skip = f"{n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options {list(opts.value)}"
            if skip is not None:
                record(skip)
                continue
            try:
                rung = self._rung(ir, board, library, params, plan, n, clearance, corners=corners)
            except (CompileError, LibraryLookupError, LibraryFormatError, ValueError, TypeError) as e:  # TypeError: as _plane_plan
                record(f"{n} copper layers: not routed - {e}")
                continue
            record(_attempt(n, rung.board, rung.extra["stack"], rung.routing), rung.seconds)
            if not rung.routing.unrouted:
                return self._settle(plan, rung, attempts, notes, minimum, clearance, kept=True, started=started)
            if started and stands is None:
                stands = rung
        if stands is not None:  # the parts' minimum skipped the board's own count: the first count it routed stands
            return self._settle(plan, stands, attempts, notes, minimum, clearance, kept=False, started=True)
        if first is None:
            notes.append(f"layer escalation: no count from {minimum.layers} up was routed; the board is routed at its own {plan.own} copper layers")
            return _Climb(None, notes)
        last = max(plan.rungs)
        past = ("" if plan.material == DEFAULT_MATERIAL else
                f"; past {last} the {MATERIALS[plan.material].name} board has no stack of its material and no FR-4 stack is swapped in")
        notes.append(f"layer escalation: no count up to {last} routes every net; the {plan.own}-layer result stands{past}")
        return _Climb(None, notes)

    def _rung(
        self, ir: CircuitIR, board: PCBDesign, library: KicadLibrary, params: RoutingParams, plan: _LadderPlan, n: int, clearance: Traced, *,
        corners: str = DEFAULT_CORNER_STYLE,
    ) -> _Rung:
        """``board`` at ``n`` copper layers - the stack, its layers and plane zones (clipped by the keep-outs) - routed with it (routing.maze 0.9).

        ``ValueError`` from the stack or the zones (a board without an outline
        has no plane area), the router's refusals (:class:`CompileError`, the
        library errors) propagate: the caller records them as the count's reason.
        """
        if board.outline is None:  # route_board's own words for it; the plane zones are drawn from the outline first
            raise ValueError("cannot route: ir.pcb.outline is None; the grid needs a board outline")
        if plan.material == DEFAULT_MATERIAL:
            g = escalation_stack(n)
            stack = escalation_stackup(n, ground_net=plan.ground, power_net=plan.power)
            vias, title, directions = g.vias, g.title, g.directions
            description = escalation_description(n, plan.ground, plan.power)
        else:
            ms = MATERIAL_STACKS[(plan.material, n)]
            stack = material_escalation_stackup(ms, plan.reason, ground_net=plan.ground, power_net=plan.power)
            vias, title, directions = via_structures(ms.generic()), ms.title, ms.directions
            description = (f"{material_stack_description(ms)}; planes: {plane_description(plan.ground, plan.power or plan.ground)}; via structures "
                           + ", ".join(["through F.Cu-B.Cu", *(f"{kind} {top}-{bottom}" for kind, top, bottom in vias)]))
        router_stack = RouterStack.from_stackup(stack, vias, title, directions=directions)
        zones = plane_zones(stack, board.outline, clearance)  # type: ignore[arg-type]
        clip: list[str] = []
        kos = keepouts_of(board)
        if kos:
            zones = _clip_planes(zones, kos, clip)
        board_n = board.model_copy(update={"layers": board_layers(stack), "stackup": stack, "zones": [*(z for z in board.zones if not _is_plane(z)), *zones]})
        board_ir = ir.model_copy(update={"pcb": board_n})
        planes: dict[str, list[tuple[str, list[tuple[float, float]]]]] = {}
        for z in zones:
            planes.setdefault(z.net, []).append((z.layer, [(float(x), float(y)) for x, y in z.polygon]))
        extra = {"keepouts": kos or None, "plane_nets": dict(sorted(planes.items())), "stack": router_stack, "corner_style": corners}
        si = net_rules(board_ir, params)
        started = _clock()
        routing = route_board(board_ir, library, params, rules=si.rules or None, **extra)
        return _Rung(n, board_n, board_ir, routing, si, extra, clip, description, _clock() - started)

    @staticmethod
    def _settle(plan: _LadderPlan, rung: _Rung, attempts: list[str], notes: list[str], minimum: _Minimum, clearance: Traced, *, kept: bool,
                started: bool) -> _Climb:
        """The climb's result at ``rung``: its stack's provenance (the owner's rule, every count's record, the stack) and the closing notes.

        ``started``: the climb began at the parts' minimum (the board's own
        count skipped). The stack keeps serving the requirements the board's
        own stack served (``plan.served``, first in ``derived_from``). What the
        board's planes and keep-out clips are goes to ``board_notes``: they are
        said only when this board is applied.
        """
        n = rung.n
        ladder = ", ".join(map(str, (plan.own, *plan.rungs)))
        reasons = "; ".join(minimum.reasons)
        if started:
            parts = f"; started at the parts' minimum of {minimum.layers} copper layers: {reasons}"
        elif n < minimum.layers:  # climbed from the board's own count: no count from the minimum up was routed, or the ladder has none
            parts = f"; below the parts' minimum of {minimum.layers} copper layers ({reasons}): no count at or above it was routed"
        else:
            parts = ""
        if kept:
            head = f"layer count {n}: chosen by the project owner's layer-escalation rule ({ESCALATION_RULE})"
            notes.append(f"layer escalation: {n} layers kept - the first count of {ladder} at which every net routes (the project owner's rule"
                         + (f", started at the parts' minimum of {minimum.layers}" if started else "")
                         + (f"; below the parts' minimum of {minimum.layers}, at or above which no count was routed" if parts and not started else "") + ")")
        else:
            head = (f"layer count {n}: the first count the project owner's layer-escalation rule ({ESCALATION_RULE}) routed at or above the parts' "
                    f"minimum; no count up to {max(plan.rungs)} routes every net, so this {n}-layer result stands")
            notes.append(f"layer escalation: no count up to {max(plan.rungs)} routes every net; the {n}-layer result stands")
        derived = [*plan.served, "pcb.placements", f"layer_escalation:{','.join(map(str, LAYER_ESCALATION))}"]
        if plan.material != DEFAULT_MATERIAL:
            derived.append(f"material:{plan.material}")
        if started:
            derived += minimum.derived
        prov = Provenance(kind=ProvenanceKind.DERIVED, tool=ESCALATION_TOOL, tool_version=ESCALATION_VERSION, derived_from=derived,
                          note=f"{head}{parts} - " + "; ".join(attempts) + f"; {rung.description}")
        return _stamped(rung, prov, notes, clearance)

    def _search(
        self, ir: CircuitIR, board: PCBDesign, library: KicadLibrary, params: RoutingParams, plan: _LadderPlan, package: BoardLayers, own_route, *,
        corners: str = DEFAULT_CORNER_STYLE,
    ) -> _Climb:
        """The package start search (module docstring: package start), from ``package.start`` up or down.

        The start is capped at the ladder's last count (``max(plan.rungs)``;
        the caller runs the search only when ``package.floor`` is at or below
        it). Up: each count of ``plan.rungs`` from the start in turn (skipped /
        refused / routed, recorded as :meth:`_climb` records them) until one
        routes every net. When no count from the start up can be built at all,
        the search starts at the highest count of ``plan.rungs`` below it, at
        or above the floor, that can be. When the count the search starts at
        routes every net, layers are removed: each lower count of the ladder
        (``plan.own`` and the rungs between it and that count, never below
        ``package.floor``), from the top, while every net routes; a count
        skipped or refused there (``plan.own`` too, when
        ``layer_count_options`` does not offer it) is passed over (it says
        nothing about routability), the first that leaves a net unrouted stops
        the removal. ``plan.own`` is routed by ``own_route`` - today's router
        on the board as it is, no stack: when it routes every net the result is
        that routing (``_Climb.own``), the board untouched. When the start
        leaves nets unrouted the board climbs and keeps the first count that
        routes every net (nothing is removed below a count that failed); when
        none does, the first routed count stands (all or nothing, as
        :meth:`_climb`), as does a count below the start that leaves nets
        unrouted (nothing above it could be built); when no count at or above
        the floor (above the board's own) could be built at all,
        ``_Climb(None, notes)``: the caller routes the board as without the
        package search (the P6 start, then its own count).
        """
        attempts: list[str] = []
        notes: list[str] = []

        def record(text: str, took: float | None = None) -> None:
            attempts.append(text)
            notes.append(f"layer escalation: {text}" + (f" ({took:.1f} s wall clock)" if took is not None else ""))

        opts = board.manufacturing.layer_count_options
        clearance = ir.parameters.get(PLANE_CLEARANCE_KEY) or ESCALATION_PLANE_CLEARANCE
        down = "fewer layers: "

        def not_offered(n: int) -> str | None:
            if opts is None or n in list(opts.value):
                return None
            return f"{n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options {list(opts.value)}"

        def build(n: int, prefix: str = "") -> _Rung | None:
            skip = escalation_refusal(n) if plan.material == DEFAULT_MATERIAL else None
            if skip is None:
                skip = not_offered(n)
            if skip is not None:
                record(prefix + skip)
                return None
            try:
                rung = self._rung(ir, board, library, params, plan, n, clearance, corners=corners)
            except (CompileError, LibraryLookupError, LibraryFormatError, ValueError, TypeError) as e:  # TypeError: as _plane_plan
                record(f"{prefix}{n} copper layers: not routed - {e}")
                return None
            record(prefix + _attempt(n, rung.board, rung.extra["stack"], rung.routing), rung.seconds)
            return rung

        floor = package.floor
        ladder = ", ".join(map(str, (plan.own, *plan.rungs)))
        start = min(package.start, max(plan.rungs))
        if start < package.start:
            notes.append(f"layer escalation: the package start of {package.start} copper layers is above the ladder's last count ({ladder}); the "
                         f"search starts at {start}")
        first: _Rung | None = None  # the first count routed from the start, when it left nets unrouted
        kept: _Rung | None = None
        for n in plan.rungs:
            if n < start:
                continue
            rung = build(n)
            if rung is None:
                continue
            if not rung.routing.unrouted:
                kept = rung
                break
            if first is None:
                first = rung
        lower = sorted((m for m in plan.rungs if floor <= m < start), reverse=True)
        if kept is None and first is None and lower:
            # no count from the start up could be built: the search starts at the highest count below it, at or above the floor, that can be
            notes.append(f"layer escalation: no count from the package start {start} up could be built; the search starts at the highest count "
                         f"below it{f' (down to the package floor of {floor})' if floor else ''} that can be")
            for n in lower:
                rung = build(n, down)
                if rung is None:
                    continue
                if rung.routing.unrouted:  # nothing above it could be built: nothing to climb to
                    notes.append(f"layer escalation: no count above {n} could be built and {n} copper layers leave nets unrouted; the {n}-layer "
                                 "result stands")
                    return self._settle_package(plan, rung, attempts, notes, package, clearance, "below", start=start)
                kept = rung
                break
        if kept is None:
            if first is not None:
                notes.append(f"layer escalation: no count up to {max(plan.rungs)} routes every net; the {first.n}-layer result stands")
                return self._settle_package(plan, first, attempts, notes, package, clearance, "stands", start=start)
            nor = (f", nor below it down to the package floor of {floor}," if floor > plan.own else ", nor below it,") if lower else ""
            below = f" - below the package floor of {floor}" if floor > plan.own else ""
            notes.append(f"layer escalation: no count from the package start {start} up{nor} could be built; the board is routed as without the "
                         f"package search{below}")
            return _Climb(None, notes)
        if first is not None:
            notes.append(f"layer escalation: {kept.n} layers kept - the first count of {ladder} from the package start {start} up at which every "
                         "net routes (the project owner's rule)")
            return self._settle_package(plan, kept, attempts, notes, package, clearance, "climbed", start=start)
        span = f"from the package start {start}" + (f" down to the package floor {floor}" if floor else "")
        stop: str | None = None
        for n in sorted((m for m in (plan.own, *plan.rungs) if floor <= m < kept.n), reverse=True):
            if n == plan.own:
                skip = not_offered(n)
                if skip is not None:  # as build() does for every other count: a count the fab does not offer is passed over
                    record(down + skip)
                    continue
                try:
                    routing, seconds = own_route()
                except (CompileError, LibraryLookupError, LibraryFormatError) as e:
                    record(f"{down}{n} copper layers: not routed - {e}")
                    stop = f"the board's own {n} copper layers were not routed"
                    break
                record(down + _attempt(n, board, None, routing), seconds)
                if routing.unrouted:
                    stop = f"{n} copper layers left {len(routing.unrouted)} net(s) unrouted"
                    break
                notes.append(f"layer escalation: {n} layers kept - every net routes on the board's own {n} copper layers (the router as before, "
                             f"no stack): the fewest {span} at which every net routes (the project owner's rule)")
                return _Climb(None, notes, own=routing, own_seconds=seconds, rule=PACKAGE_CHOSEN_BY)
            rung = build(n, down)
            if rung is None:
                continue
            if rung.routing.unrouted:
                stop = f"{n} copper layers left {len(rung.routing.unrouted)} net(s) unrouted"
                break
            kept = rung
        if stop is None:  # no lower count left a net unrouted: the floor was reached, or every count above it was skipped or refused
            stop = (f"the package floor of {floor} copper layers allows no fewer" if kept.n <= floor
                    else f"no count below {kept.n}{f' down to the package floor of {floor}' if floor else ''} could be routed")
        notes.append(f"layer escalation: {kept.n} layers kept - the fewest copper layers {span} at which every net routes ({stop}; the project "
                     "owner's rule)")
        return self._settle_package(plan, kept, attempts, notes, package, clearance, "removed", stop, start=start)

    @staticmethod
    def _settle_package(plan: _LadderPlan, rung: _Rung, attempts: list[str], notes: list[str], package: BoardLayers, clearance: Traced, how: str,
                        stop: str = "", *, start: int) -> _Climb:
        """The package search's result at ``rung`` (``how``: ``removed`` / ``climbed`` / ``stands`` / ``below``; ``start``: the count the
        search started from, ``package.start`` capped at the ladder): its stack stamped ``design.package_layers`` with the rules, the
        start's and the floor's rows (with their sources), every count tried and the stack."""
        n = rung.n
        head = {
            "removed": f"layer count {n}: the fewest copper layers from the package start at which every net routes ({stop})",
            "climbed": f"layer count {n}: the package start left nets unrouted; the first count above it at which every net routes",
            "stands": (f"layer count {n}: the first count routed from the package start; no count up to {max(plan.rungs)} routes every net, so "
                       f"this {n}-layer result stands"),
            "below": (f"layer count {n}: no count from the package start {start} up could be built; {n}, the highest count below it that could "
                      f"be, leaves nets unrouted and nothing above it could be built, so this {n}-layer result stands"),
        }[how]
        floor = f"package floor {package.floor}: " + "; ".join(package.floor_why) if package.floor else "package floor: none"
        capped = f" (the search started at {start}, the ladder's last count)" if start < package.start else ""
        derived = [*plan.served, "pcb.placements", f"layer_escalation:{','.join(map(str, LAYER_ESCALATION))}", *package.derived]
        if plan.material != DEFAULT_MATERIAL:
            derived.append(f"material:{plan.material}")
        prov = Provenance(kind=ProvenanceKind.DERIVED, tool=PACKAGE_LAYERS_TOOL, tool_version=PACKAGE_LAYERS_VERSION, derived_from=derived,
                          note=(f"{head} - the package rule ({PACKAGE_RULE}) with the project owner's layer-escalation rule ({ESCALATION_RULE}); "
                                f"package start {package.start}{capped}: {'; '.join(package.start_why)}; {floor} - " + "; ".join(attempts)
                                + f"; {rung.description}"))
        climb = _stamped(rung, prov, notes, clearance)
        climb.rule = PACKAGE_CHOSEN_BY
        return climb

    @staticmethod
    def _with_single_layer_keepout(board: PCBDesign, notes: list[str]) -> PCBDesign:
        """``board`` with keep-out ``ims_bcu`` (no copper on B.Cu) when its stack has one copper layer on a metal base; every other board unchanged."""
        ko = single_layer_keepout(board.stackup, board.outline)
        if ko is None or board.keepout(ko.id) is not None:
            return board
        notes.append(f"keep-out {ko.id}: {ko.reason} - B.Cu kept out over the outline (tracks, vias, pads, zones), so the router routes F.Cu only "
                     f"({MATERIALS_TOOL} {MATERIALS_VERSION})")
        return board.model_copy(update={"keepouts": [*board.keepouts, ko]})

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
        # a template's (a design file's) board texts (module docstring) stay and are placed around; any other text means the silkscreen is
        # designed already
        keep = list(board.silkscreen) if board.silkscreen and all(proposed_silk_text(t) for t in board.silkscreen) else None
        if board.silkscreen and keep is None:
            return [*notes, f"silkscreen not placed: ir.pcb already has {len(board.silkscreen)} silkscreen text(s); the agent never replaces silkscreen"], None, ""
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return [*notes, "silkscreen not placed: no KiCad library in ctx.tools['kicad_library']; pad and silk geometry cannot be read"], None, ""
        try:
            kept_by = "design file" if keep and all(design_file_silk_text(t) for t in keep) else "template"
            placed = place_silkscreen(ir.model_copy(update={"pcb": board}), library, self.silk, keep=keep, kept_by=kept_by)  # keep=None: as before
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return [*notes, f"silkscreen not placed: {e}"], None, ""
        description = placed.description()
        return [*notes, description, *placed.notes], list(placed.texts), description

    @staticmethod
    def _rules_note(p: RoutingParams, raised: dict | None = None, floored: dict | None = None) -> str:
        """``fine rules: pad pitch 0.8 mm ...``: why the board was routed at the fine rules, with the values the router used - and
        where they come from when a fab minimum (``raised``) or, on a stack board, the minimums kicad-cli DRC checks (``floored``,
        routing.maze 0.9's ``stats["kicad_minimums"]["raised"]``) raised them."""
        text = (
            f"{p.rules} rules: pad pitch {p.pad_pitch_mm} mm ({p.pitch_footprint}) is below {FINE_PITCH_MM} mm: grid {p.grid_mm} mm, "
            f"width {p.track_width_mm} mm, clearance {p.clearance_mm} mm, via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
        if raised:
            text += " (raised to ir.pcb.manufacturing minimums: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in raised.items()) + ")"
        if floored:
            text += (" (raised to the minimums kicad-cli DRC checks: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in floored.items())
                     + ")")
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
    def _corner_style(ctx: AgentContext, notes: list[str]) -> str:
        """The corner style every routing of this run passes (module docstring: corner style): ``arc``
        (:data:`~ai_eda.tools.routing.corners.DEFAULT_CORNER_STYLE`) unless ``pcb.corners`` answers ``miter45`` or ``orthogonal`` (the
        latter said after the routing description); any other answer is noted as not understood and the default applies."""
        answer = ctx.answers.get(CORNERS_KEY)
        if answer is None:
            return DEFAULT_CORNER_STYLE
        style = answer.strip().lower()
        if style not in CORNER_STYLES:
            answers = ", ".join(repr(s) for s in CORNER_STYLES[:-1]) + f" and {CORNER_STYLES[-1]!r}"
            notes.append(f"{CORNERS_KEY}={answer!r} not understood (the only answers are {answers}); "
                         f"corners are rounded as usual ({DEFAULT_CORNER_STYLE})")
            return DEFAULT_CORNER_STYLE
        return style

    @staticmethod
    def _routing_description(routing: Routing) -> str:
        p = routing.params
        s = routing.stats
        on = "/".join(s["stack"]["signal_layers"]) if "stack" in s else "F.Cu/B.Cu"  # routing.maze 0.9: the stack's signal layers
        text = (
            f"{ROUTER_ID} {routing.version}: {s['routed_nets']} net(s) routed on {on}, {s['track_count']} track(s), {s['via_count']} via(s), "
            f"{s['total_length_mm']} mm of copper, {s['iterations']} negotiation iteration(s); grid {p.grid_mm} mm, width {p.track_width_mm} mm, "
            f"clearance {p.clearance_mm} mm, via {p.via_diameter_mm}/{p.via_drill_mm} mm, edge {p.edge_clearance_mm} mm"
        )
        if p.rules is not None:
            text += f"; {p.rules} rules: pad pitch {p.pad_pitch_mm} mm ({p.pitch_footprint}) is below {FINE_PITCH_MM} mm"
        if p.board_nets is not None:  # a large board: its negotiation cap (RoutingParams.for_board)
            text += f"; large board: {p.board_nets} nets to route (more than {LARGE_BOARD_NETS}), up to {p.max_iterations} negotiation iterations"
        if s["skipped_nets"]:
            text += f"; {len(s['skipped_nets'])} net(s) with fewer than 2 pads skipped ({', '.join(s['skipped_nets'])})"
        if s["raised"]:
            text += "; raised to ir.pcb.manufacturing minimums: " + ", ".join(f"{name} {req} -> {eff}" for name, (req, eff) in s["raised"].items())
        if (s.get("kicad_minimums") or {}).get("raised"):  # routing.maze 0.9, a stack board: what the minimums kicad-cli DRC checks raised
            text += "; raised to the minimums kicad-cli DRC checks: " + ", ".join(
                f"{name} {req} -> {eff}" for name, (req, eff) in s["kicad_minimums"]["raised"].items())
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
        if s.get("escapes"):  # routing.maze 0.6: fine-pitch / off-grid pads joined to the grid by escape stubs
            text += (f"; {len(s['escapes'])} pad(s) reach the grid by an escape stub or cell ({', '.join(e['pad'] for e in s['escapes'])})"
                     + (f", none reaches {', '.join(s['escape_refused'])}" if s.get("escape_refused") else ""))
        if s.get("via_fanout"):  # routing.maze 0.8: every other pad of a fine-pitch row left by a stub to a via, on the other layer
            fan = s["via_fanout"]
            text += f"; via fan-out of {', '.join(fan['footprints'])}: {len(fan['pads'])} pad(s) leave by a stub to a via and continue on the other layer"
        if "stack" in s:  # routing.maze 0.9: how many vias of each kind
            kinds: dict[str, int] = {}
            for row in s["stack"]["via_types"]:
                kinds[row["kind"]] = kinds.get(row["kind"], 0) + row["count"]
            text += "; via types: " + ", ".join(f"{kind} {count}" for kind, count in kinds.items())
        if "corners" in s:  # routing.corners 0.1: the project owner's corner-style rule (module docstring: corner style)
            c = s["corners"]
            # routing.corners 0.2: the corners rounded into arcs and the ones mitred where no arc fits, then the router's sidesteps
            # straightened before that
            if c.get("style") == ARC:
                text += (f"; corners: {c['arcs']} of {c['found']} corner(s) of 90 degrees or sharper rounded into tangent arcs and {c['mitred']} "
                         f"mitred into 45-degree chamfers where no arc fits ({c['tool']} {c['version']}, the project owner's corner-style rule - "
                         f"a geometry convention, not an electrical requirement), {c['left']} left as routed, {c['skipped']} skipped; "
                         f"sidesteps: {c['straightened']} of {c['sidesteps']} sidestep(s) of the router (a short track between two "
                         f"right-angle corners) straightened before the corners were rounded, {c['sidesteps_left']} left as routed")
            else:
                text += (f"; corners: {c['mitred']} of {c['found']} corner(s) of 90 degrees or sharper mitred into 45-degree chamfers ({c['tool']} "
                         f"{c['version']}, the project owner's corner-style rule - a geometry convention, not an electrical requirement), "
                         f"{c['left']} left as routed, {c['skipped']} skipped")
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
    more, never less) as connected polygons with cut-outs
    (:func:`~ai_eda.tools.keepout.rect_contours`); every part keeps the
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
        boxes = [area_bbox(area_points(k)) for k in cuts]
        try:
            parts = rect_contours((min(xs), min(ys), max(xs), max(ys)), boxes)
        except ValueError as e:
            raise CompileError(str(e)) from e
        by_box = [kid for kid, k in zip(ids, cuts) if not area_is_rect(k)]
        words = f"clipped by keep-out(s) {', '.join(ids)}" + (f" ({', '.join(by_box)} by its bounding box)" if by_box else "")
        if not parts:
            notes.append(f"plane zone {z.net} on {z.layer} not added: keep-out(s) {', '.join(ids)} forbid all of it")
            continue
        prov = z.provenance.model_copy(update={
            "derived_from": [*z.provenance.derived_from, *(f"pcb.keepouts[{kid}]" for kid in ids)],
            "note": f"{z.provenance.note}; {words}",
        })
        out.extend(z.model_copy(update={"polygon": [(_q(x), _q(y)) for x, y in poly],
            "holes": [[(_q(x), _q(y)) for x, y in hole] for hole in holes], "provenance": prov}) for poly, holes in parts)
        notes.append(f"plane zone {z.net} on {z.layer} {words}: {len(parts)} polygon(s)"
                     + (f" with {sum(len(holes) for _, holes in parts)} cut-out(s)" if any(holes for _, holes in parts) else ""))
    return out


def _routes_around_planes(board: PCBDesign) -> bool:
    """Whether ``board``'s inner layers are exactly its stackup's plane layers: then the router may accept them (it never routes them)."""
    inner = [layer.name for layer in board.layers if INNER_LAYER_RE.match(layer.name)]
    if not inner or board.stackup is None:
        return False
    planes = {c.name for c in board.stackup.plane_layers()}
    return set(inner) <= planes


# --------------------------------------------------------------------------- the layer escalation's records (module docstring)


@dataclass
class _LadderPlan:
    """The counts the escalation may try above the board's own (``own``), in order, its plane nets and its material's stacks."""

    own: int
    rungs: tuple[int, ...]
    ground: str
    power: str | None
    #: :data:`~ai_eda.design.materials.DEFAULT_MATERIAL` (the FR-4 escalation stacks) or the board's material id (its own stacks)
    material: str
    #: the board's material decision, carried into a material stack's record ("" for FR-4)
    reason: str
    #: the requirement ids the board's own stack serves (``Stackup.served_requirements``: the material keys ``add_board`` read); the
    #: escalated stack serves them too, so the reviewer still finds them traced. Empty for a board without a stack
    served: tuple[str, ...] = ()


@dataclass
class _Minimum:
    """The copper layer count the board's parts need (0: none stated or estimated), why (with each source), and parts not counted."""

    layers: int = 0
    reasons: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: each reason's own count (``layers`` is their largest)
    counts: list[int] = field(default_factory=list)

    def add(self, layers: int, reason: str, derived: str) -> None:
        self.layers = max(self.layers, layers)
        self.reasons.append(reason)
        self.derived.append(derived)
        self.counts.append(layers)

    def floors(self) -> list[tuple[int, str, str]]:
        """``(layers, reason, derived)`` per minimum: the package search's extra floors (module docstring: package start)."""
        return list(zip(self.counts, self.reasons, self.derived))


@dataclass
class _Rung:
    """One count the escalation routed: the board at that count (stack, layers, plane zones), its IR copy, copper, rules and router keywords."""

    n: int
    board: PCBDesign
    board_ir: CircuitIR
    routing: Routing
    si_rules: SIRules
    extra: dict
    clip_notes: list[str]
    #: the stack's one-line description, for the provenance
    description: str
    seconds: float


@dataclass
class _Climb:
    """What the escalation leaves: the rung whose board and copper replace the board's own (``None``: the board's own stands) and its notes."""

    rung: _Rung | None
    notes: list[str]
    #: what the rung's board adds (its plane zones, their keep-out clips): noted only when that board is applied
    board_notes: list[str] = field(default_factory=list)
    #: the package search kept the board's own count: its routing (today's router, no stack) and wall time
    own: Routing | None = None
    own_seconds: float = 0.0
    #: what chose the kept count, for the proposal's rationale
    rule: str = "the project owner's layer-escalation rule"


#: what chose a count the package search kept, for the proposal's rationale
PACKAGE_CHOSEN_BY = "the package-start rule and the project owner's layer-escalation rule"


def _stamped(rung: _Rung, prov: Provenance, notes: list[str], clearance: Traced) -> _Climb:
    """``rung`` with its stack's provenance replaced by ``prov``, and what its board adds (its plane zones, their keep-out clips)."""
    stack = rung.board.stackup.model_copy(update={"provenance": prov})  # type: ignore[union-attr]
    board = rung.board.model_copy(update={"stackup": stack})
    zones = [z for z in board.zones if _is_plane(z)]
    board_notes: list[str] = []
    if zones:  # what _with_planes says of the planes it adds (it adds none here: the board has them)
        board_notes.append("plane zones: " + ", ".join(dict.fromkeys(f"{z.net} on {z.layer}" for z in zones))
                           + f" (outline inset {float(clearance.value):g} mm; KiCad fills them)")
    board_notes.extend(rung.clip_notes)
    settled = _Rung(rung.n, board, rung.board_ir.model_copy(update={"pcb": board}), rung.routing, rung.si_rules, rung.extra, rung.clip_notes,
                    rung.description, rung.seconds)
    return _Climb(settled, notes, board_notes)


def _package_lines(package: BoardLayers) -> list[str]:
    """The notes that say where the package search starts and how far down it may go, each row with its source (the floor's with its quote)."""
    lines = [f"layer escalation: package start {package.start} copper layers ({PACKAGE_LAYERS_TOOL} {PACKAGE_LAYERS_VERSION}) - "
             + "; ".join(package.start_why)]
    if package.floor:
        lines.append(f"layer escalation: package floor {package.floor} copper layers - " + "; ".join(package.floor_why))
    else:
        lines.append("layer escalation: package floor: none - no matched row states a recommended, required or minimum count, so layers may be "
                     "removed down to the board's own count")
    return lines


def _package_not_applied(ir: CircuitIR, package: BoardLayers, why: str) -> list[str]:
    """Why the package search does not run on this board; a stated count below the floor names the floor's sources and quotes."""
    lines = [f"layer escalation: the package start of {package.start} copper layers is not applied - {why}"]
    layers, _unusable = read_layer_count(ir)
    if package.floor and layers is not None and layers.requirement is not None and layers.value < package.floor:
        lines.append(f"layer escalation: {layers.requirement.id} states {layers.value} copper layer(s), below the package floor of {package.floor} - "
                     + "; ".join(package.floor_why))
    return lines


def _copper_count(board: PCBDesign) -> int:
    """The board's copper layer count: its stack's (an IMS has one, though the KiCad board keeps ``B.Cu``), else its layer list's."""
    return board.stackup.layer_count if board.stackup is not None else len(board.layers)


def _template_of(ir: CircuitIR):
    """The registered template whose structural provenance (``design.template.<id>`` on the topology or a component) built ``ir``; ``None``.

    The rule of ``ai_eda.report.stages.template_for``, read here so the agent
    does not import the report package.
    """
    from ai_eda.design.templates import all_templates  # the templates import the agents' neighbours: read only when a board has a stack

    prefix = f"{TEMPLATE_TOOL_ID}."
    provenances = ([ir.topology.provenance] if ir.topology is not None else []) + [c.provenance for c in ir.components]
    for p in provenances:
        if p.tool and p.tool.startswith(prefix):
            tid = p.tool[len(prefix):]
            return next((t for t in all_templates() if t.id == tid), None)
    return None


def _stated_material(decision: MaterialDecision) -> str | None:
    """Why a board without a stack does not climb the FR-4 stacks against what its requirements say of its material, or ``None``.

    ``None`` when nothing is stated or the decision is FR-4 (the escalation
    stacks' material); else the note names the requirements, the material (or
    the question, or why they are unusable) and that no FR-4 stack is swapped
    in: no stage builds a material stack for a board no template made.
    """
    ids = ", ".join(r.id for r in decision.read)
    if decision.refusal is not None:
        head = f"the material requirements are unusable ({decision.refusal})"
    elif decision.question is not None:
        head = f"{ids}: the material rule asks {MATERIAL_KEY} first ({decision.question.rationale})"
    elif decision.material != DEFAULT_MATERIAL:
        head = f"{ids}: the board's base material is {MATERIALS[decision.material].name} ({decision.reason})"
    else:
        return None
    return f"{head}; this board has no stack and no stage builds a material stack for a board no template made, so no FR-4 stack is swapped in"


def _stated_count_note(layers: LayerCountInput | None, unusable: str | None, own: int, rungs: list[int], template: bool) -> str:
    """Why a board whose layer count the user stated is not escalated, the next count of the ladder and how to ask for it."""
    if layers is None or layers.requirement is None:
        return f"the layer count is stated but unusable ({unusable}); a stated count is never escalated"
    key = layers.requirement.key
    head = f"{layers.requirement.id} states {layers.value} copper layer(s), the user's choice, which is never escalated"
    if layers.value != own:
        return (f"{head}; this board has {own} (no stage builds a stack from a stated count for a board no template made) - leave the count out "
                f"({leave_out_answer([key])}) and the PCB stage climbs the ladder from {own}")
    if not rungs:
        return f"{head}; no count of the ladder lies above it"
    nxt = rungs[0]
    if template and nxt in LAYER_COUNT_OPTIONS:
        return f"{head}; the next count of the ladder is {nxt} layers: answer --answer {LAYER_COUNT_KEY}={nxt} to ask for it"
    return (f"{head}; the next count of the ladder is {nxt} layers, which this version reaches only by escalation ({LAYER_COUNT_KEY} takes "
            f"{' or '.join(map(str, LAYER_COUNT_OPTIONS))}) - leave the count out ({leave_out_answer([key])}) and the PCB stage climbs the ladder")


def _attempt(n: int, board: PCBDesign, stack: RouterStack | None, routing: Routing) -> str:
    """One count's record: its signal and plane layers, the router's version and result, the negotiation iterations."""
    if stack is not None:
        signals, planes = list(stack.signal), [f"{layer} {net}" for layer, net in stack.planes]
    elif board.stackup is not None:
        signals = board.stackup.signal_layers()
        planes = [f"{c.name} {c.plane_net.value}" for c in board.stackup.plane_layers()]  # type: ignore[union-attr]
    else:
        signals, planes = [layer.name for layer in board.layers], []
    s = routing.stats
    if routing.unrouted:
        ripped = len(set(routing.unrouted) & set(s.get("dropped") or ()))
        result = f"left {len(routing.unrouted)} of {s['routed_nets'] + s['unrouted_nets']} net(s) unrouted ({ripped} ripped up in negotiation)"
    else:
        result = "routed every net"
    return (f"{n} copper layers (signals {', '.join(signals)}; planes {', '.join(planes) or 'none'}): {ROUTER_ID} {routing.version} {result} "
            f"after {s['iterations']} iteration(s)")


def _area_array(fp: FootprintDef, numbers: set[str]) -> tuple[int, float, float] | None:
    """``(rows deep, pitch, land)`` of an area-array footprint, or ``None``.

    An area array here: every copper pad one land (shape and size) and every
    centre on one square lattice of the footprint's pad pitch
    (:func:`~ai_eda.tools.routing.maze.footprint_pad_pitch`, within
    :data:`LATTICE_TOLERANCE_MM`) - a BGA / LGA grid. A pad's ring is how many
    rows lie between it and the lattice's edge; the depth is the deepest ring
    of a pad on a net (``numbers``: an unconnected land needs no escape) plus
    one. A perimeter package (QFP / QFN / SOIC) has depth 1, or is no lattice
    at all; a staggered array is none either (not estimated).
    """
    pads = [pad for pad in fp.pads if pad.pad_type != "np_thru_hole" and any(layer.endswith(".Cu") for layer in pad.layers)]
    if len(pads) < 2 or len({(pad.shape, pad.size_w, pad.size_h) for pad in pads}) != 1:
        return None
    pitch = footprint_pad_pitch(fp)
    if pitch is None:
        return None
    x0, y0 = min(pad.x for pad in pads), min(pad.y for pad in pads)
    cells: list[tuple[str, int, int]] = []
    for pad in pads:
        i, j = round((pad.x - x0) / pitch), round((pad.y - y0) / pitch)
        if abs(pad.x - x0 - i * pitch) > LATTICE_TOLERANCE_MM or abs(pad.y - y0 - j * pitch) > LATTICE_TOLERANCE_MM:
            return None
        cells.append((pad.number, i, j))
    imax, jmax = max(i for _, i, _ in cells), max(j for _, _, j in cells)
    depth = max((min(i, imax - i, j, jmax - j) + 1 for number, i, j in cells if number in numbers), default=0)
    return depth, pitch, max(pads[0].size_w, pads[0].size_h)


def _parts_minimum(ir: CircuitIR, library: KicadLibrary, params: RoutingParams) -> _Minimum:
    """The copper layer count the board's parts need, in natural ref order (module docstring: layer escalation).

    * A part's ``min_pcb_layers`` counts only with its source document in its
      provenance (a part fact comes from the manufacturer's datasheet, the
      document and page cited); one without is noted and not counted.
    * An area-array footprint (:func:`_area_array`) ``d`` rows deep needs
      ``ceil(d / 2)`` signal layers (:data:`NXP_ESCAPE_SOURCE`) when one trace
      fits between its lands and between the via pads of its escape
      (:data:`TI_CHANNEL_SOURCE`, ``d`` = the larger of the land and the via
      diameter, ``s`` / ``w`` the router's clearance and width after the fab
      raise); that is the first count of the ladder with as many signal layers
      (:data:`LADDER_SIGNALS`). When one trace does not fit the rule's premise
      fails and nothing is estimated (a note says so). An array the board's
      two signal layers escape (4 rows or fewer) adds nothing.

    Every pad on a net is counted, a ground or power ball too (the estimate
    does not know which nets the planes will carry): a stated choice.
    """
    out = _Minimum()
    p, _raised = effective_params(ir, params)
    connected: dict[str, set[str]] = {}
    for net in ir.nets:
        for pin in net.pins:
            connected.setdefault(pin.component_ref, set()).add(pin.pin_number)
    for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        stated = comp.min_pcb_layers
        if stated is not None:
            source = stated.provenance.source
            if source is None or not source.title.strip():
                out.notes.append(f"{comp.ref} min_pcb_layers {stated.value} names no source document - not counted (a part fact is read from its "
                                 "manufacturer's datasheet, the document and page cited)")
            else:
                where = source.title.strip() + (f", {source.section}" if source.section else "")
                out.add(int(stated.value), f"{comp.ref} needs at least {int(stated.value)} copper layers ({where})", f"components[{comp.ref}].min_pcb_layers")
        if comp.footprint is None or not library.resolve_footprint(comp.footprint).verified:
            continue
        fp = library.load_footprint(comp.footprint)
        array = _area_array(fp, connected.get(comp.ref, set()))
        if array is None:
            continue
        depth, pitch, land = array
        signals = math.ceil(depth / 2)
        if signals <= 2:
            continue
        need = 2.0 * p.clearance_mm + p.track_width_mm
        d = max(land, p.via_diameter_mm)
        g = pitch - d
        what = f"{comp.ref} ({fp.lib_id}): pads {depth} rows deep in a {pitch:g} mm area array"
        if g < need - 1e-9:
            out.notes.append(f"{what} - no escape-depth estimate: one trace between lands needs g >= 2s + w = {_q(need):g} mm and g = pitch - d = "
                             f"{_q(g):g} mm (d = the larger of the {land:g} mm land and the {p.via_diameter_mm:g} mm via pad), so the premise of "
                             f"{NXP_ESCAPE_SOURCE} does not hold ({TI_CHANNEL_SOURCE}; {ESCAPE_SOURCES_OPENED})")
            continue
        count = next((n for n in LAYER_ESCALATION if LADDER_SIGNALS.get(n, 0) >= signals), None)
        if count is None:
            out.notes.append(f"{what} need {signals} signal layers ({NXP_ESCAPE_SOURCE}) - more than any count of the ladder has "
                             f"({max(LADDER_SIGNALS.values())}); not counted")
            continue
        out.add(count, (f"{what} need {signals} signal layers ({NXP_ESCAPE_SOURCE}; one trace fits: g = {pitch:g} - {d:g} = {_q(g):g} mm >= 2s + w = "
                        f"{_q(need):g} mm, {TI_CHANNEL_SOURCE}; {ESCAPE_SOURCES_OPENED}) - {count} copper layers is the first count of the ladder "
                        f"with {LADDER_SIGNALS[count]}"), f"escape_depth:{comp.ref}={depth}")
    return out


__all__ = [
    "CORNERS_KEY", "ESCALATION_PLANE_CLEARANCE", "ESCALATION_RULE", "ESCAPE_SOURCES_OPENED", "GRID_RATIONALE", "LADDER_SIGNALS", "NXP_ESCAPE_SOURCE", "PARTIAL_ANSWER",
    "PCBAgent", "PLACEMENT_KEY", "RING_RATIONALE", "ROUTING_KEY", "SILKSCREEN_KEY", "SILK_RATIONALE", "SKIP_ANSWER", "TI_CHANNEL_SOURCE", "UNROUTED_NOTE",
]
