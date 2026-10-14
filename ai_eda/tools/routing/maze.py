"""Deterministic two-layer grid maze router with negotiated congestion and per-net rules (pure, no I/O beyond the KiCad library).

Invariant: every track and via produced here is a function of the IR's
placements, the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: pad positions, sizes,
layers - never model memory), the :class:`RoutingParams` and the caller's
:class:`NetRule` s. Nothing is estimated, nothing is guessed: a component
without a placement or footprint, a footprint that is not on disk, a net pin
without a pad, a copper layer other than ``F.Cu`` / ``B.Cu`` (inner
``In<k>.Cu`` layers only when the caller opts in) or a rule that makes no
sense raises :class:`~ai_eda.errors.CompileError` instead of getting a
default; a pad too small for the grid that no escape stub reaches (below:
routing.maze 0.6) leaves its net unrouted with that reason.

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
  copper, and one stub segment joins it to the exact centre (a pad whose
  nearest cell is outside that circle, or fenced as below, gets an escape
  instead: routing.maze 0.6, below). The terminal
  cell obeys the owner map like every other cell: on a layer where it is
  BLOCKED (a foreign pad's keep-out or the board edge reaches it) or owned by
  another net it is neither a seed nor a target, and a terminal with no
  usable layer leaves the net unrouted with that reason - the router never
  emits copper that breaks its own clearance at a pad.
* pad shapes: ``circle`` / ``rect`` / ``oval`` / ``roundrect`` are convex and
  lie inside their ``(size)`` box, so the box is a conservative obstacle and
  the inscribed circle a safe terminal. A ``custom`` pad is several boxes
  (below: routing.maze 0.5). A ``trapezoid`` (``rect_delta`` extends one side
  beyond the box) is refused (:class:`CompileError` naming the pad): the
  library reader does not keep ``rect_delta``, and an obstacle box that is
  smaller than the copper would be a guess; so is a ``custom`` pad with a
  primitive the library reader does not read.

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
  ``via_diameter/2 + max(c, c_pad)``; another net's escape stub or escape
  via (routing.maze 0.6, below) counts by its exact copper, as the escape
  pass and the owner map measure it - never by its box, whose corners a bent
  stub does not fill (a 0.4 plane via or stub keeps its box). A rule width (or neck-down width) below
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

**Keep-outs and plane nets (routing.maze 0.4).** :func:`route_board` takes
``keepouts`` (the IR's :class:`~ai_eda.ir.pcb.Keepout` s, read through
:mod:`ai_eda.tools.keepout`) and ``plane_nets`` (net name -> the layer and
polygon of each plane zone the caller gives the net), both supplied by the
caller - the PCB agent passes them only for a board with keep-outs or RF
blocks. Without both the
result is exactly 0.2's / 0.3's (no line of the paths above changes); with
either, every track and via is stamped :data:`ROUTER_KEEPOUT_VERSION`
``"0.4"`` and names the keep-outs / plane nets in ``derived_from``.

* A keep-out that forbids **tracks** on ``F.Cu`` / ``B.Cu`` joins the static
  owner map: every cell closer than ``w_max/2 + grid/2`` to its area (the
  widest copper any net of the board may draw - the board width, the rule
  widths, a pair's envelope, the plane stubs - so no net's copper reaches
  into the area from a cell outside it) is BLOCKED, or owned by the
  keep-out's one allowed net (a cell holds one net: a keep-out allowing two
  or more nets lets none of them through, and ``stats["keepouts"]`` says
  so). A pair's fence and its breakouts keep out of such areas too.
* A keep-out that forbids **vias** on any copper layer (a through via is on
  all of them) forbids a via whose disc (``via_diameter/2``) would reach its
  area, except for its allowed nets.
* A **plane net**'s pads are never joined by tracks: each SMD pad gets its
  own via beside it - the nearest grid cell (along one axis from the pad's
  terminal cell, steps east / south / west / north in that order, up to
  :data:`PLANE_VIA_REACH_MM` beyond the point where a via disc first clears
  the pad's own copper box on that axis: a 6.4 x 5.8 mm TO-252 tab is
  crossed first) where a via is legal (the via rules above - never inside a
  pad box, its own included: a tab's paste apertures would wick the solder -,
  the keep-outs, ``via_diameter + clearance`` between the centres of any two
  plane vias, and its whole disc inside one of the net's plane zones, so the
  fill can surround it) and the straight run to it is free on the pad's
  layer - joined by one stub; a through-hole pad needs none (its barrel
  reaches the inner planes), and neither does an SMD pad on which a
  through-hole pad of the same footprint, pad number and net sits (a
  ``..._ThermalVias`` exposed pad: the footprint's own vias join it to the
  plane; listed as ``joined_by_footprint_vias``). The pad vias and stubs are
  claimed on the static maps like pads (every other net keeps its clearance
  from them) before any net is negotiated. A plane net one of whose SMD pads
  has no via site is unrouted with that reason and gets no copper at all.
  Whether the plane's fill really joins the vias is KiCad's fill and DRC;
  ``pcb.routing.connectivity`` reports such a net NOT_VERIFIED.

**Custom pads (routing.maze 0.5).** A ``custom`` pad's copper is bounded by
:func:`ai_eda.tools.kicad.geometry.custom_pad_parts` - the one extent the
``pcb.routing.*`` / ``pcb.keepout`` checks, the silkscreen, the SI paths, the
3D scene and the figures also read: the box of its anchor (the ``(size)``
rectangle or circle at the pad's copper centre) and one box around each
copper primitive (``gr_poly`` / ``gr_line`` / ``gr_rect`` / ``gr_circle`` /
``gr_arc`` / ``gr_curve``, grown by half its stroke), each an outer bound of
its copper. Each box is a pad entry of its own for the owner maps, the
no-via-in-pad rule, the rule fences, the pair breakouts and the exact audit,
so every clearance rule above holds against the whole bound. A track lands
on the anchor: the terminal is the grid cell nearest the anchor's centre
inside its inscribed circle (else inside the inscribed circle of the first
part known to be copper there - an exact filled rectangle, a filled disc -,
else the anchor's refusal); a plane net's pad-via walk starts beyond the
pad's whole copper in each direction. A box larger than the copper (a ring's
square, a circle's corners) keeps other copper further away than needed and
may enclose another pad (the ``CUI_CMC-4013-SMT`` microphone's ring pad 1
around its pad 2: pad 2's terminal is then inside pad 1's keep-out and its
net is unrouted with that reason - on the real copper too, a closed ring
leaves no way out on its layer). A board with at least one custom pad is
stamped :data:`ROUTER_CUSTOM_PAD_VERSION` ``"0.5"`` on every track and via
(whatever else it has: rules, keep-outs, plane nets), ``derived_from`` names
the pads (``custom_pads:...;model=anchor+primitive_boxes``) and
``stats["custom_pads"]`` lists every part's box. Such a board was refused
before, so no saved copper changes; a board without a custom pad is exactly
0.2's / 0.3's / 0.4's.

**Escape stubs (routing.maze 0.6).** Two pad situations used to end a net
before it was searched, and neither is a fact about the copper: a pad whose
nearest grid point lies outside its inscribed circle (a 0.25 mm-wide pad
0.125 mm off the 0.2 mm grid: 0.5 refused the whole board), and a pad whose
grid cell lies inside its *neighbours'* keep-out on every layer of the pad.
The second is the fine-pitch case, and its cause is a rule applied where it
does not belong: the owner map keeps a cell ``clearance + width/2 + grid/2``
from every foreign pad box because a full-width *track* centred there must
keep the clearance (and its unit steps too), and at a 0.4 mm pitch (a
0.25 mm pad, a 0.15 mm gap) or a 0.5 mm one (0.25 mm pads) every cell of the
pad's own centreline is within that radius of the pads beside it - so the
rule for a track was fencing the pad's own copper, where no full-width track
has to lie and the clearance of a narrower one can be measured exactly. The
rule is now: such a pad (and a plane pad whose 0.4 via walk finds no site)
is joined to the grid by an *escape*, and only when none exists is it
refused. Nothing changes for a pad whose grid cell is usable: every board
without such a pad routes exactly as 0.2 / 0.3 / 0.4 / 0.5 did (byte for
byte, stamps included).

* **The stub** runs from the pad's centre along the axis of its larger half
  extent (a square or round pad: the axis nearer the direction from its
  footprint's copper centroid to the pad; all four directions, east, south,
  west, north, when that is undecided), pointing away from the centroid, to
  the pad's outer edge ``O``; from there straight on for ``a`` (a multiple
  of ``grid/2``, 0 first) and then one straight segment to a grid cell ``E``
  in front of the edge (``(E - O) . d >= 0``, both offsets at most
  :attr:`RoutingParams.escape_reach_mm`). Its width is the smaller of the
  pad's narrow side and the net's track width (a plane pad's: its stubs'
  width) when that keeps every clearance, else the widest multiple of
  :attr:`RoutingParams.escape_width_step_mm` that does, and at least
  :attr:`RoutingParams.escape_min_width_mm` (raised to the fab's
  ``min_track_width_mm``: recorded in ``stats["raised"]`` and the params
  entry) - or that candidate is illegal. A signal pad whose
  grid cell is only fenced first tries another grid point inside its
  inscribed circle (``kind="cell"``) from which the ordinary stub to the
  centre, at the net's width, keeps every clearance below exactly.
* **Legality, exact**: every segment keeps ``max(c_net, c_other)`` from
  every foreign pad box, every other net's escape, plane stub and plane via
  and every other net's ordinary stub (terminal cell to pad centre, at that
  net's width) on its layer (exact segment / box / disc distances - the part
  inside its own pad included, which is why a 0.4 mm-pitch stub is narrower
  than its pad), no copper inside a track keep-out that does not let the net
  through, ``edge_clearance + width/2`` from the outline; ``E`` is free or
  the net's own on the static owner map and outside the net rule's fence (a
  track of the net may start there); the stub's claim - every cell within
  ``clearance + track_width/2 + grid/2`` of its copper, owned by the net
  like a pad's - takes no cell reserved for another net: a usable ordinary
  terminal cell, an earlier escape's ``E`` or its *way out*, a cell a fan
  holds (a refusal names which); its copper (stub and via, exactly) puts no
  usable terminal cell of another net whose rule is wider or has a larger
  clearance than the board's - an ordinary one, or an earlier escape's
  ``E`` - inside that net's fence (the neck-down width within its radius of
  the pad), so whether such a terminal stays usable does not depend on which
  pad escaped first (the reserved cell alone keeps only the board's width
  clear); and ``E`` has a
  way out - the shortest 4-neighbour chain of cells free or the net's own
  (and reserved for no other net) from ``E`` to the first cell outside the
  footprint's escape area (its pads' box grown by the reach and a grid
  step), which is then reserved for the net so no later escape closes it.
* **Plane nets**: an SMD pad of a plane net whose grid cell is fenced or off
  the grid, or whose axis walk (0.4's rule, unchanged when it succeeds)
  finds no via site, waits - its net pending, its other vias claimed - for
  an escape stub to a *via* at a grid cell in front of its edge: the via
  obeys the plane-via rules exactly (its disc keeps ``max(c_net, c_other)``
  from every foreign pad, stub and via, overlaps no pad box - its own
  included -, keeps ``via_diameter + clearance`` from every other plane via,
  lies wholly inside one of the net's plane zones, outside every via
  keep-out that does not allow the net, ``edge_clearance + via_diameter/2``
  from the outline) and both claims keep every other net's reserved cells.
  A pending net one of whose pads gets none is unrouted with that pad's
  reason and no copper - and since its other vias were claimed while it
  waited, it is doomed (below).
* **The choice**: one footprint at a time (natural ref order, each seeing
  the escapes before it), its waiting pads - signal and plane - escaped
  greedily in each order of ``_ESCAPE_TRIALS`` in turn (every claim undone
  after each), and the first order with the fewest plane pads left without
  an escape (one leaves its whole plane net unrouted), then the fewest
  signal pads, then the shortest total stub, is applied. An order is the
  pads' sequence - ``fan`` (below, then the rest nearest the centroid
  first), nearest the footprint's copper centroid first (a row fans out from
  its middle), farthest first, pad order - and the candidates - by distance
  from ``O`` to ``E``, then the smaller lateral offset, then the side away
  from the centroid, then the cell index (``near``), or straight ones first
  (``straight``); ``a`` from 0 up, the first legal one taken - and whether
  the plane pads' vias or the signal pads go first. A *fan* gives the pads
  of one row (one escape direction, one edge line) cells on one line in
  front of it, as far apart as their claims need (the next cell outside a
  signal stub's claim, ``via_diameter + clearance`` between two vias),
  centred on the row, at the shallowest depth where the most of them are
  legal, the straightest pads first: a QFN's side leaves as a fan the
  nearest-first greedy cannot find. All of it is deterministic; none of it
  is optimal.
* **Doomed nets**: a pass - the plane vias, then the escape pass - can doom
  a net: a plane net one of whose pads no escape reaches (its other vias
  were claimed while it waited), or a signal net with such a pad that holds
  an escape (a stub, a reserved way out, that the unrouted net will never
  use). Then the pass is thrown away and run again from the maps as the
  pads and the keep-outs left them, without that net - no via, stub, claim
  or reservation of it -, so the other nets' via walks, escape choices and
  claims are made as on a board where it is never routed (as 0.4 undid a
  failing plane net before the next one walked), and a net that failed only
  beside the doomed net's claims is tried again. One net per pass: the
  first such plane net in name order (the order they walk in), else the
  signal net whose pad was refused first. A doomed net is never taken back
  (a greedy pass is not monotone), so there is at most one more pass per
  net, and a board where no pass dooms a net runs once. A doomed net keeps
  the refusals of the pass that doomed it, and each of its pads that pass
  escaped is listed as withdrawn (``stats["escape_refused"]``).
* The pad's terminal becomes ``E`` on the stub's layer (the search starts
  and ends there) and the stub replaces the one-segment stub to the pad
  centre; its segments are ordinary tracks of the net in the IR at the stub
  width (so ``pcb.routing.*`` and the SI paths see them), each naming the
  stub in its ``derived_from`` (``escape:<pad>:<stub|via>:width=<w>:points=
  x,y;...``, read back by :func:`escape_entry`: ``si.impedance`` names such
  a segment of a controlled net like a neck-down) and its geometry in the
  note, and their length counts toward a length budget.
* A pad no escape reaches keeps its refusal, naming it, with why no escape
  exists appended (the nearest candidate's problem): its net is unrouted
  with no copper at all (an off-grid pad no longer refuses the whole board -
  the refusal is the net's, like a fenced terminal's - except a pad of a
  coupled pair, which keeps 0.3's board refusal). Pads of nets that are
  never routed (fewer than two pads) get no escape.

A board with a pad off the grid (0.5 refused such a board), or with an
escape or a plane pad that waited for one in the last pass, is stamped
:data:`ROUTER_ESCAPE_VERSION` ``"0.6"`` on every track and via (in place of
0.2 - 0.5); its params entry records the three escape parameters,
``derived_from`` names the escaped and the refused pads
(``escapes:...;refused=...;model=centre+edge+cell``) and ``stats["escapes"]``
/ ``["escape_refused"]`` list every escape (pad, net, kind, why, layer,
cell, width, points, length) and every pad none reached. A board whose only
escape news is pads none reached - a fenced signal pad, a fenced plane pad
or one without a via site, and with them the doomed nets' withdrawn
escapes - keeps its 0.2 - 0.5 copper and stamp: those nets were unrouted
before too, now with the reason extended.

**Fan-out room (routing.maze 0.7).** 0.6 lays each footprint's escapes
greedily among the copper already on the maps, and every plane net's vias
come first: on the kr447 boards a neighbouring part's ground pad took its
0.4 via (the walk's first legal site, east first) inside the MAX9814's
escape area, and the lanes and ways out its fan needed were gone. The
*fan-out room* of a footprint is its pads' box grown by
:func:`fanout_margin` (the escape area, ``escape_reach_mm + grid_mm``, plus
the farthest foreign copper reaches into it: the owner map's pad keep-out
``clearance + track_width/2 + grid/2`` or a via disc's ``via_diameter/2 +
clearance`` - 2.2 mm at the fine rules; the RF floorplan keeps every other
part's extent that far from a fine-pitch part's pads). A board on which
0.6's static phase leaves a pad without an escape runs the static phase
once more, from the same maps, with the fan-out rooms of the footprints
0.6 left a pad of:

* **kept clear of other footprints' plane vias**: a 0.4 via walk of another
  footprint's pad closes a direction at the first site whose via claim
  (its disc grown by the pad keep-out) would enter a room (the pad is listed
  as kept out), and a plane escape's via there is illegal ("the via would
  lie in the fan-out room of ...");
* **ways out to the room's edge**: the way out of such a footprint's escape
  runs to the first cell outside its room, not its escape area, and is
  reserved as before - a way out that ended at the escape area's edge could
  end in a pocket between the next part's pads and the footprint's own
  escape claims (measured: MIC_VB on the transceiver, statically
  unreachable).

The second run's result is kept only when it leaves fewer nets (then fewer
pads) unroutable; then every track and via is stamped
:data:`ROUTER_FANOUT_VERSION` ``"0.7"``, its params entry records the escape
knobs, ``derived_from`` names the rooms, their margin and the pads whose via
walk met one (``fanout_room:<refs>;margin=<mm>;kept_out=<pads>;model=
pads_box+margin``) and ``stats["fanout_room"]`` gives each room's box.
Otherwise the board is 0.6's, byte for byte; a board 0.6 escapes
completely, or that needs no escape, never runs it.

**Large boards.** :meth:`RoutingParams.for_board` gives a board with more
than :data:`LARGE_BOARD_NETS` nets to route (:func:`routed_net_count`) a
negotiation cap of :data:`LARGE_BOARD_ITERATIONS` in place of 40, recorded
in every track's params entry with the net count
(``large_board_nets=<n>``); every other board keeps 40 and its copper.

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
:data:`ROUTER_VERSION`, :data:`ROUTER_RULES_VERSION`, :data:`ROUTER_KEEPOUT_VERSION`,
:data:`ROUTER_CUSTOM_PAD_VERSION`, :data:`ROUTER_ESCAPE_VERSION` or
:data:`ROUTER_FANOUT_VERSION`), the net, the
placements of the net's components and every parameter in
``derived_from``; the note names the iteration count and how the net's route
was obtained. ``Provenance.inputs`` stays empty: it is the calculator role
map, and a track is not a calculator output.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Net, Provenance, ProvenanceKind, Track, Via
from ai_eda.tools.kicad.geometry import PadPart, _q, custom_pad_parts, pad_angle, pad_center, pad_layers
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, Pad
from ai_eda.tools.keepout import allowed_nets as keepout_allowed_nets
from ai_eda.tools.keepout import area_bbox, area_points, covers_layer, forbids, keepout_id, point_area_distance, segment_area_distance
from ai_eda.tools.routing.coupling import uncoupled_lengths

__all__ = [
    "ROUTER_ID",
    "ROUTER_VERSION",
    "ROUTER_RULES_VERSION",
    "ROUTER_KEEPOUT_VERSION",
    "ROUTER_CUSTOM_PAD_VERSION",
    "ROUTER_ESCAPE_VERSION",
    "ROUTER_FANOUT_VERSION",
    "ESCAPE_ENTRY_PREFIX",
    "PLANE_VIA_REACH_MM",
    "LAYERS",
    "BLOCKED",
    "CONVEX_PAD_SHAPES",
    "FINE_PITCH_MM",
    "FINE_RULES",
    "LARGE_BOARD_ITERATIONS",
    "LARGE_BOARD_NETS",
    "NetRule",
    "RoutingParams",
    "Routing",
    "effective_params",
    "escape_entry",
    "fanout_margin",
    "finest_pad_pitch",
    "footprint_pad_pitch",
    "route_board",
    "routed_net_count",
]

#: provenance ``tool`` stamped on every track and via
ROUTER_ID = "routing.maze"
#: ``tool_version`` of copper routed without net rules: routing.maze 0.3 routes such a board exactly as 0.2 did (byte for
#: byte), so the stamp - part of every track's provenance, hence of the design hash - stays
ROUTER_VERSION = "0.2"
#: ``tool_version`` of every track and via of a board routed with at least one :class:`NetRule`
ROUTER_RULES_VERSION = "0.3"
#: the version stamped on every track and via of a board routed with keep-outs or plane nets (module docstring)
ROUTER_KEEPOUT_VERSION = "0.4"
#: the version stamped on every track and via of a board with at least one ``custom`` pad (module docstring: custom pads); it takes
#: the place of 0.2 / 0.3 / 0.4 on such a board, whose copper was refused before
ROUTER_CUSTOM_PAD_VERSION = "0.5"
#: the version stamped on every track and via of a board with at least one escape stub (module docstring: escape stubs); it takes the
#: place of 0.2 - 0.5 on such a board, whose escaped pads were refused before
ROUTER_ESCAPE_VERSION = "0.6"
#: the version stamped on every track and via of a board whose static phase ran under routing.maze 0.7's rules (module docstring:
#: fan-out room); it takes the place of 0.6 on such a board, where 0.6 left a pad without an escape
ROUTER_FANOUT_VERSION = "0.7"
#: how far (mm, along one axis from the terminal cell) a plane net's SMD pad looks for the site of its via
PLANE_VIA_REACH_MM = 3.0
#: the entries of the board's pad list a net rule's fence measures by their exact copper (module docstring: net rules), as the escape pass
#: and the owner map do: the escape stubs and escape vias (routing.maze 0.6); pads and the 0.4 plane vias / stubs keep their boxes
_EXACT_FENCE_KINDS = frozenset({"escape", "escape-via"})
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
#: an escape candidate that is not one at all (a cell another net's pad claims, a via site off the zone): skipped without a reason
_SKIP = "skip"
#: the orders one footprint's escapes are tried in (routing.maze 0.6, module docstring: escape stubs): (pad order, candidate order,
#: which pads go first - the plane pads' vias or the signal pads' stubs)
_ESCAPE_TRIALS: tuple[tuple[str, str, str], ...] = (
    ("fan", "near", "plane"), ("fan", "near", "signal"), ("middle", "near", "plane"), ("middle", "near", "signal"), ("middle", "straight", "plane"), ("middle", "straight", "signal"),
    ("ends", "near", "plane"), ("ends", "near", "signal"), ("natural", "near", "plane"), ("natural", "near", "signal"),
)

#: what a reserved cell of the escape pass is (routing.maze 0.6), as a refusal names it - after "it is" and after "would take": an
#: ordinary or escape terminal cell, a cell on the way out reserved for an escape, a cell a fan holds for one of its pads while it is tried
_RESERVED_WORDS: dict[str, tuple[str, str]] = {
    "terminal": ("the terminal cell of", "the terminal cell of"),
    "exit": ("on the way out reserved for an escape of", "a cell on the way out reserved for an escape of"),
    "fan": ("the cell a fan holds for", "the cell a fan holds for"),
}

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
    ``board_nets`` says why ``max_iterations`` is :data:`LARGE_BOARD_ITERATIONS`:
    :meth:`for_board` counted that many nets to route, more than
    :data:`LARGE_BOARD_NETS` (``None`` otherwise; recorded only when set).

    The net-rule knobs (0.3; used only for nets a :class:`NetRule` names, and
    recorded in the provenance only when a board has rules):
    ``neckdown_radius_mm`` (the neck-down zone around a pad when the rule
    gives none), ``meander_amplitude_mm`` / ``meander_pitch_mm`` (the largest
    bump height and the leg spacing of a length-matching meander, rounded to
    whole grid steps) and ``pair_bump_amplitude_mm`` / ``pair_bump_length_mm``
    (the largest height and the length of one intra-pair compensation bump).

    The escape knobs (0.6; used only for a pad whose grid cell is off its
    copper or fenced, module docstring: escape stubs, and recorded in the
    provenance only when a board has an escape stub):
    ``escape_reach_mm`` (how far in front of the pad's edge, along and
    across its axis, an escape cell is looked for), ``escape_min_width_mm``
    (the narrowest stub; the fab's ``min_track_width_mm`` raises it, and
    :func:`route_board` records the raise on a board with escapes) and
    ``escape_width_step_mm`` (a stub narrower than the pad's narrow side and
    the net's width is a whole multiple of it: the widest that keeps every
    clearance).
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
    escape_reach_mm: float = 1.5
    escape_min_width_mm: float = 0.1
    escape_width_step_mm: float = 0.01
    board_nets: int | None = None

    def check(self) -> None:
        """Refuse parameters that make no sense (:class:`CompileError`)."""
        for name in (
            "grid_mm", "track_width_mm", "via_drill_mm", "via_diameter_mm", "base_cost", "window_mm",
            "neckdown_radius_mm", "meander_amplitude_mm", "meander_pitch_mm", "pair_bump_amplitude_mm", "pair_bump_length_mm",
            "escape_reach_mm", "escape_min_width_mm", "escape_width_step_mm",
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
        if self.board_nets is not None and (isinstance(self.board_nets, bool) or not isinstance(self.board_nets, int) or self.board_nets < 1):
            raise CompileError(f"routing parameter board_nets must be an integer >= 1 (got {self.board_nets!r})")

    def derived_from_entry(self, rules: bool = False, escapes: bool = False) -> str:
        """The ``derived_from`` entry that records every parameter (and, when set, the rule set and the pitch that chose it).

        ``rules=True`` (a board routed with net rules) appends the 0.3 knobs,
        ``escapes=True`` (a board with an escape stub) the 0.6 knobs; without
        either the entry is 0.2's, character for character.
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
        if self.board_nets is not None:
            entry += f",large_board_nets={self.board_nets}"
        if rules:
            entry += (
                f",neckdown_radius={self.neckdown_radius_mm},meander_amplitude={self.meander_amplitude_mm},"
                f"meander_pitch={self.meander_pitch_mm},pair_bump_amplitude={self.pair_bump_amplitude_mm},"
                f"pair_bump_length={self.pair_bump_length_mm}"
            )
        if escapes:
            entry += (
                f",escape_reach={self.escape_reach_mm},escape_min_width={self.escape_min_width_mm},"
                f"escape_width_step={self.escape_width_step_mm}"
            )
        return entry

    @classmethod
    def for_board(cls, ir: CircuitIR, library: KicadLibrary) -> RoutingParams:
        """The fine rules (:data:`FINE_RULES`) when the finest pad pitch on the board is below :data:`FINE_PITCH_MM`, else the defaults;
        a large board (more than :data:`LARGE_BOARD_NETS` nets to route) negotiates up to :data:`LARGE_BOARD_ITERATIONS` iterations.

        The pitch is :func:`finest_pad_pitch` (library pad positions); the
        net count is :func:`routed_net_count` (the IR's nets with two or more
        pins), recorded in ``board_nets``. A board with neither is returned
        as ``cls()`` - no ``rules`` recorded. Fab minimums in
        ``ir.pcb.manufacturing`` still raise either set when the router runs
        (:func:`effective_params`).
        """
        pitch = finest_pad_pitch(ir, library)
        base = cls() if pitch is None or pitch[0] >= FINE_PITCH_MM else cls(**FINE_RULES, rules="fine", pad_pitch_mm=pitch[0], pitch_footprint=pitch[1])
        nets = routed_net_count(ir)
        if nets > LARGE_BOARD_NETS:
            return replace(base, max_iterations=LARGE_BOARD_ITERATIONS, board_nets=nets)
        return base


def routed_net_count(ir: CircuitIR) -> int:
    """How many of the IR's nets name two or more distinct pins - the nets a router has to connect (:meth:`RoutingParams.for_board`)."""
    return sum(1 for net in ir.nets if len({(pin.component_ref, pin.pin_number) for pin in net.pins}) >= 2)


#: a board with more nets to route than this (:func:`routed_net_count`) is a large board: its negotiation may take up to
#: :data:`LARGE_BOARD_ITERATIONS` iterations (every other board keeps :attr:`RoutingParams.max_iterations`, 40, and its copper byte for byte)
LARGE_BOARD_NETS = 100
#: the negotiation cap of a large board (:data:`LARGE_BOARD_NETS`), recorded in every track's ``params:`` entry with the net count; chosen
#: above the measured need of the kr447 boards with the MAX9814's fan-out room (2026-09-30: tx_exciter, 108 nets, legal after 93
#: iterations; both transceiver builds, 211 nets, after 123 - at 40 they had 4 / 5 / 6 conflicting nets left)
LARGE_BOARD_ITERATIONS = 150


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


def footprint_pad_pitch(fp: FootprintDef) -> float | None:
    """The smallest centre-to-centre distance between two copper pads of ``fp`` that can carry different nets (``None``: no such pair).

    Two pads with the same non-empty number are one logical pad (a split
    thermal pad, a switch's duplicated pins) and are skipped, as are pads
    without copper (paste-only apertures, NPTH holes) and coincident centres;
    distances are taken in the footprint's own frame (a rotation does not
    change them) and rounded to KiCad's resolution. :func:`finest_pad_pitch`
    takes the board's smallest; the RF floorplan placer reads it per
    footprint (its fan-out room, :mod:`ai_eda.tools.placement.rf_floorplan`).
    """
    pads = [p for p in fp.pads if _copper(p)]
    pitch: float | None = None
    for i, a in enumerate(pads):
        for b in pads[i + 1:]:
            if a.number and a.number == b.number:
                continue
            d = _q(math.hypot(a.x - b.x, a.y - b.y))
            if d > 0 and (pitch is None or d < pitch):
                pitch = d
    return pitch


def fanout_margin(p: RoutingParams) -> float:
    """The fan-out room (mm) around a fine-pitch footprint's pads' box, from the router's own parameters (routing.maze 0.7, module
    docstring: fan-out room; the RF floorplan keeps every other part's extent this far from such a part's pads): the escape area
    (``escape_reach_mm + grid_mm``, where every escape cell, plane-via stub and way out lies) plus the farthest foreign copper reaches
    into it (the owner map's pad keep-out ``clearance + track_width/2 + grid/2``, or a via disc's ``via_diameter/2 + clearance``);
    rounded to KiCad's resolution. 2.2 mm at the fine rules (1.5 + 0.2 + max(0.425, 0.5))."""
    reach = max(p.clearance_mm + p.track_width_mm / 2.0 + p.grid_mm / 2.0, p.via_diameter_mm / 2.0 + p.clearance_mm)
    return _q(p.escape_reach_mm + p.grid_mm + reach)


def finest_pad_pitch(ir: CircuitIR, library: KicadLibrary) -> tuple[float, str] | None:
    """``(pitch, footprint lib id)``: the smallest centre-to-centre distance between two copper pads of one footprint on the board.

    Each footprint's pitch is :func:`footprint_pad_pitch` (pads that can
    carry different nets, copper only, the footprint's own frame, KiCad's
    resolution); ties go to the first footprint in natural ref order.
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
            seen[fp.lib_id] = footprint_pad_pitch(fp)
        pitch = seen[fp.lib_id]
        if pitch is not None and (best is None or pitch < best[0]):
            best = (pitch, fp.lib_id)
    return best


@dataclass(slots=True)
class Routing:
    """What :func:`route_board` produced. ``unrouted`` maps a net name to why it has no copper.

    ``version`` is the ``tool_version`` stamped on the copper (:data:`ROUTER_VERSION`
    without net rules, :data:`ROUTER_RULES_VERSION` with; :data:`ROUTER_KEEPOUT_VERSION`,
    :data:`ROUTER_CUSTOM_PAD_VERSION`, :data:`ROUTER_ESCAPE_VERSION` or
    :data:`ROUTER_FANOUT_VERSION` on the boards the module docstring names), ``rules`` the effective rules (after
    the fab / board raises and the defaults).
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


def _edge_gap(p: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    """Distance from ``p`` to the polygon's boundary."""
    n = len(poly)
    return min(_pt_seg(p[0], p[1], poly[i], poly[(i + 1) % n]) for i in range(n))


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
    """One placed pad in the board frame: centre, half extents of its obstacle box, copper layers, net index.

    A ``custom`` pad is several of these (one per :class:`~ai_eda.tools.kicad.geometry.PadPart`, sharing ``ref`` / ``number`` /
    ``net``): the part a track lands on carries ``reach`` - how far the pad's whole copper extends east, south, west and north of its
    centre (the plane-via walk starts beyond it); ``None`` for a convex pad (its box is its whole copper)."""

    ref: str
    number: str
    cx: float
    cy: float
    hw: float
    hh: float
    inscribed_r: float
    layers: tuple[int, ...]  # indices into LAYERS
    net: int  # net index or BLOCKED
    reach: tuple[float, float, float, float] | None = None
    #: what the entry is: a footprint ``"pad"`` (or part of one), a plane net's ``"plane-via"`` / ``"plane-stub"`` (0.4) or an
    #: ``"escape"`` segment / ``"escape-via"`` (0.6); every consumer before 0.6 reads only :attr:`box`
    kind: str = "pad"
    #: the exact copper of a via or stub entry, which the 0.6 escape checks measure instead of the box: ``("seg", a, b, half width)``
    #: or ``("disc", centre, radius)``; ``None`` for a pad (its box is its bound)
    shape: tuple | None = None

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.hw, self.cy - self.hh, self.cx + self.hw, self.cy + self.hh)


@dataclass(frozen=True, slots=True)
class _Escape:
    """How a pad whose grid cell is off its copper or fenced reaches the grid (module docstring: escape stubs, routing.maze 0.6).

    ``kind``: ``"cell"`` - another grid point inside the pad's inscribed circle is usable, so the ordinary one-segment stub to the pad
    centre is kept and only the terminal cell moves; ``"stub"`` - a stub (``points``, from the pad centre, at ``width``) to the grid cell
    ``cell`` where the net's track starts; ``"via"`` - a plane net's stub to its via at ``cell``. ``why``: ``"off-grid"``, ``"fenced"``
    or ``"no via site"`` (a plane pad whose axis walk found none)."""

    kind: str
    why: str
    layer: int
    cell: int
    width: float | None = None
    points: tuple[tuple[float, float], ...] = ()
    #: a signal stub's way out: the grid cells (on its layer) from ``cell`` to the first one outside the footprint's escape area,
    #: every one free or the net's own - reserved for the net, so no later escape closes it
    exit: tuple[int, ...] = ()

    def length_mm(self) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:]))


@dataclass(frozen=True, slots=True)
class _Terminal:
    pad: _PadGeom
    cell: int  # cell index k = j * nx + i
    label: str
    #: routing.maze 0.6: how the pad reaches the grid when its own cell does not (``cell`` is then the escape's cell), else ``None``
    escape: _Escape | None = None
    #: the refusal of a pad whose nearest grid point lies outside its inscribed circle (0.5 refused the board with it)
    offgrid: str | None = None
    #: why the pad cannot be reached when it needed an escape and none exists: its net is unrouted with this reason
    refusal: str | None = None


#: one pad waiting for an escape (routing.maze 0.6): (``"signal"`` / ``"plane"``, net, index in the net's terminals or slot in the
#: plane net's links, terminal, pad layer of a plane pad, why - ``"off-grid"`` / ``"fenced"`` / ``"no via site"`` -, the plane walk's reason)
_Need = tuple[str, str, int, "_Terminal", int, str, "str | None"]


@dataclass(slots=True)
class _PendingPlane:
    """A plane net one of whose SMD pads waits for an escape (routing.maze 0.6): its walk links so far (``None`` where a pad waits),
    its claimed entries, its through-hole / footprint-joined pads, the waiting pads ``(slot, terminal, layer, why, walk reason)`` and,
    once the escape pass is done, the reason of the first pad none reached."""

    idx: int
    width: float
    areas: list[list[tuple[float, float]]]
    links: list[tuple[_Terminal, int, int] | None]
    geoms: list[_PadGeom]
    tht: list[str]
    joined: list[str]
    deferred: list[tuple[int, _Terminal, int, str, str]]
    failed: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Doomed:
    """A net a pass of the static phase doomed (routing.maze 0.6, :meth:`_Board._static_phase`): a plane net one of whose pads no escape
    reaches, or a signal net with such a pad that held an escape. ``reason`` is the net's (its first refused pad's refusal), ``refused``
    the refused pads (terminal index - ``None`` for a plane pad -, label, refusal) as the dooming pass left them, ``withdrawn`` the pads
    whose escape that pass made and the later passes do not (the net gets no copper): terminal index, label and, for a plane pad, why it
    waited (a signal pad's is its off-grid sentence or what fences it in the last pass)."""

    kind: str
    reason: str
    refused: list[tuple[int | None, str, str]]
    withdrawn: list[tuple[int | None, str, str | None]]


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

    Escapes (routing.maze 0.6, module docstring: escape stubs) are found
    here, after the pads, the keep-outs and the plane vias: for every pad of
    a routed signal net (two or more pads, not a plane net, not in
    ``no_escape`` - the coupled pairs, whose off-grid pads keep 0.3's board
    refusal) whose grid cell is off its copper or fenced on every layer of
    the pad, and for every plane pad the plane via walk could not serve
    (:meth:`_static_phase`: :meth:`_plane_vias` and :meth:`_escape_all`,
    repeated without each net a pass dooms, then :meth:`_finish_planes`;
    :attr:`doomed` / :attr:`static_passes` say what happened). ``net_rules`` (net
    index -> width, clearance, neck-down width and radius) keeps an escape
    cell out of a rule net's fence. :attr:`escapes` lists the escapes,
    :attr:`escape_refused` the pads none reached, :attr:`escape_ran` says
    whether the board is routing.maze 0.6's, :attr:`rules07` whether its
    static phase is routing.maze 0.7's (the fan-out rooms, :attr:`fanout_rooms`,
    and the plane pads whose via walk met one, :attr:`fanout_kept`).
    """

    def __init__(
        self, ir: CircuitIR, library: KicadLibrary, p: RoutingParams, pad_clearance: Mapping[int, float] | None = None, inner_layers: bool = False,
        *, keepouts: Sequence[Any] | None = None, ko_width: float | None = None, plane: Mapping[str, float] | None = None,
        plane_areas: Mapping[str, list[list[tuple[float, float]]]] | None = None, no_escape: Sequence[str] | None = None,
        net_rules: Mapping[int, tuple[float, float, float | None, float | None]] | None = None,
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
        #: every placed pad (the rule fences and the exact audits read them); a custom pad is one entry per part
        self.pads: list[_PadGeom] = []
        #: the custom pads of the board (``REF.NUMBER``, natural ref order) and their parts (routing.maze 0.5, module docstring)
        self.custom_pads: list[str] = []
        self.custom_pad_stats: list[dict[str, Any]] = []
        #: each placed component's copper-pad centroid (the "footprint body" an escape stub points away from, routing.maze 0.6)
        self.fp_centroid: dict[str, tuple[float, float]] = {}
        #: routing.maze 0.6 (module docstring): the escape stubs made (one row each), the pads none reached, the cells no other net's claim
        #: may take (``layer * n + k`` -> net) and what each of them is (a key of :data:`_RESERVED_WORDS`)
        self.escapes: list[dict[str, Any]] = []
        self.escape_refused: dict[str, str] = {}
        self._reserved: dict[int, int] | None = None
        self._reserved_why: dict[int, str] = {}
        #: the ordinary one-segment stubs (terminal cell -> pad centre) of the usable terminals, as copper an escape keeps clear of
        self._bands: list[_PadGeom] = []
        #: the usable terminal cells of the nets whose rule is stricter than the owner map (``(layer * n + k, net, pad)``, ordinary and
        #: escaped): no later escape may put one inside its net rule's fence (:meth:`_guard_hit`), whatever the escape order
        self._rule_guard: list[tuple[int, int, _PadGeom]] = []
        #: plane nets waiting for an escape of one of their pads (routing.maze 0.6) and every plane via so far
        self._pending: dict[str, _PendingPlane] = {}
        self._plane_placed: list[tuple[float, float]] = []
        #: the nets the static phase doomed (:meth:`_static_phase`): net -> what it left behind, in the order they were doomed
        self.doomed: dict[str, _Doomed] = {}
        #: how many times the static phase ran (1 unless a pass doomed a net)
        self.static_passes = 0
        #: each footprint's escape area (its pads' box grown by the escape reach): an escape's way out ends outside it
        self._escape_area: dict[str, tuple[float, float, float, float]] = {}
        #: routing.maze 0.6 ran on this board: an escape was made, or a plane net waited for one (its copper may differ from 0.5's)
        self.escape_ran = False
        #: net index -> (width, clearance, neck-down width, neck-down radius) of the nets a rule names (an escape cell obeys the rule)
        self.net_rules: dict[int, tuple[float, float, float | None, float | None]] = dict(net_rules or {})
        mfg = ir.pcb.manufacturing
        min_track = float(mfg.min_track_width_mm.value) if mfg is not None and mfg.min_track_width_mm is not None else 0.0
        #: the narrowest escape stub: ``escape_min_width_mm``, raised to the fab's ``min_track_width_mm`` (:func:`route_board` records the
        #: raise in ``stats["raised"]`` and the params entry of a 0.6 board)
        self.escape_min_width = max(p.escape_min_width_mm, min_track)
        self._load_pads(ir, library)
        blocked_escape = set(no_escape or ())
        for name in sorted(blocked_escape):  # a coupled pair's pads keep 0.3's refusal of a pad off the grid
            for t in self.terminals.get(name, []):
                if t.offgrid is not None:
                    raise CompileError(t.offgrid)
        if any(t.offgrid is not None for terms in self.terminals.values() for t in terms):
            self.escape_ran = True  # 0.5 refused a board with a pad off the grid: whatever routes it is 0.6
        self._eligible: list[str] = [
            net.name for net in ir.nets
            if len(self.terminals[net.name]) >= 2 and net.name not in blocked_escape and net.name not in (plane or {})
        ]
        #: routing.maze 0.4 (module docstring): what each keep-out blocked, the via keep-outs (cell -> the nets allowed a via
        #: there), the track keep-outs as areas (the pair breakouts check them), and each plane net's pad vias or why not
        self.keepout_stats: list[dict[str, Any]] = []
        self.via_ko: dict[int, frozenset[int]] | None = None
        self.track_keepouts: list[tuple[str, list[tuple[float, float]], frozenset[int], frozenset[int]]] = []
        self.plane_links: dict[str, list[tuple[_Terminal, int, int]]] = {}
        self.plane_tht: dict[str, list[str]] = {}
        #: SMD pads of a plane net joined to it by their own footprint's same-numbered through-hole pads (no stub, no via of ours)
        self.plane_joined: dict[str, list[str]] = {}
        self.plane_problems: dict[str, str] = {}
        #: routing.maze 0.7 (module docstring: fan-out room) is in force - the static phase's second run - and the escape areas it keeps
        #: clear of every other footprint's plane vias (the footprints 0.6 left a pad of), and the plane pads whose via it kept out
        self.rules07 = False
        self.fanout_rooms: dict[str, tuple[float, float, float, float]] = {}
        self.fanout_kept: list[str] = []
        if keepouts:
            self._apply_keepouts(keepouts, p.track_width_mm if ko_width is None else max(p.track_width_mm, ko_width))
        self._static_phase(plane or {}, plane_areas or {})
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
            centres = [pad_center(placement, pad) for pad in fp.pads if _copper(pad)]
            if centres:
                self.fp_centroid[comp.ref] = (sum(x for x, _ in centres) / len(centres), sum(y for _, y in centres) / len(centres))
            for pad in fp.pads:
                parts: list[PadPart] | None = None
                if pad.shape == "custom":
                    try:
                        parts = custom_pad_parts(placement, pad)
                    except CompileError as e:
                        raise CompileError(f"cannot route: pad {comp.ref}.{pad.number or '(unnumbered)'} of footprint {fp.lib_id}: {e}") from e
                elif pad.shape not in CONVEX_PAD_SHAPES:
                    raise CompileError(
                        f"cannot route: pad {comp.ref}.{pad.number or '(unnumbered)'} of footprint {fp.lib_id} has shape {pad.shape!r}; "
                        f"its copper is not bounded by its (size) box (a trapezoid's rect_delta is not read), "
                        f"so this router models only {sorted(CONVEX_PAD_SHAPES)} and custom pads (by their primitives)"
                    )
                net_name = pin_net.get((comp.ref, pad.number)) if pad.number else None
                layers = _pad_copper_layers(placement, pad)
                net_idx = self.net_index[net_name] if net_name else BLOCKED
                if parts is None:
                    cx, cy, hw, hh, r_in = _pad_box(placement, pad)
                    geoms = [_PadGeom(comp.ref, pad.number, cx, cy, hw, hh, r_in, layers, net_idx)]
                else:
                    geoms = self._custom_geoms(comp.ref, pad.number, parts, layers, net_idx)
                for geom in geoms:
                    self.pads.append(geom)
                    for layer in layers:
                        self._mark_box(layer, geom)
                    self._forbid_vias_in(geom)
                if net_name is None:
                    continue
                if not layers:
                    raise CompileError(f"cannot route net {net_name!r}: pad {comp.ref}.{pad.number} is on no copper layer ({pad.layers})")
                self.terminals[net_name].append(self._terminal(geoms[0]) if parts is None else self._custom_terminal(geoms))
        for terms in self.terminals.values():
            terms.sort(key=lambda t: (natural_ref_key(t.pad.ref), natural_ref_key(t.pad.number)))

    def _custom_geoms(self, ref: str, number: str, parts: list[PadPart], layers: tuple[int, ...], net: int) -> list[_PadGeom]:
        """A ``custom`` pad as one obstacle box per part (its anchor first), each knowing how far the whole pad reaches from its centre;
        recorded in :attr:`custom_pads` / :attr:`custom_pad_stats`."""
        x1, y1 = min(p.box.x1 for p in parts), min(p.box.y1 for p in parts)
        x2, y2 = max(p.box.x2 for p in parts), max(p.box.y2 for p in parts)
        geoms: list[_PadGeom] = []
        for part in parts:
            cx, cy = part.center
            hw, hh = part.box.width / 2.0, part.box.height / 2.0
            geoms.append(_PadGeom(ref, number, cx, cy, hw, hh, part.inscribed_r, layers, net, (x2 - cx, y2 - cy, cx - x1, cy - y1)))
        label = f"{ref}.{number or '(unnumbered)'}"
        self.custom_pads.append(label)
        self.custom_pad_stats.append({"pad": label, "parts": [
            {"part": part.what, "box": [part.box.x1, part.box.y1, part.box.x2, part.box.y2], "exact": part.exact, "inscribed_r": _q(part.inscribed_r)}
            for part in parts
        ]})
        return geoms

    def _custom_terminal(self, geoms: list[_PadGeom]) -> _Terminal:
        """The terminal of a ``custom`` pad: on its anchor when the nearest grid point lies inside the anchor's inscribed circle, else on
        the first part whose inscribed circle (copper for sure: an exact rectangle, a filled disc) holds its nearest grid point; else the
        first part's failure - raised when it is an error, the off-grid terminal (whose escape routing.maze 0.6 looks for) otherwise."""
        first: CompileError | _Terminal | None = None
        for geom in geoms:
            if geom.inscribed_r <= 0.0:
                continue
            try:
                t = self._terminal(geom)
            except CompileError as e:
                first = first or e
                continue
            if t.offgrid is None:
                return t
            first = first or t
        if isinstance(first, _Terminal):
            return first
        raise first or CompileError(f"cannot route: custom pad {geoms[0].ref}.{geoms[0].number} has no part a track can land on")

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
            # 0.5 refused the board here; 0.6 looks for an escape stub for a pad of a routed net and refuses only that net
            return _Terminal(pad=geom, cell=j * self.nx + i, label=label, offgrid=(
                f"pad {label} ({2 * geom.hw:g} x {2 * geom.hh:g} mm) is too small for the {self.p.grid_mm} mm routing grid: "
                f"the nearest grid point is {d:.4f} mm from its centre, outside its inscribed circle (r={geom.inscribed_r:g} mm)"
            ))
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

    # --- keep-outs and plane nets (routing.maze 0.4) -----------------------------

    def _cells_near_area(self, pts: list[tuple[float, float]], r: float) -> list[int]:
        """Every cell closer than ``r`` to the area ``pts`` (inside it included), in cell order."""
        x1, y1, x2, y2 = area_bbox(pts)
        i0, j0 = self._nearest_cell(x1 - r, y1 - r)
        i1, j1 = self._nearest_cell(x2 + r, y2 + r)
        out: list[int] = []
        g = self.p.grid_mm
        for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
            for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                if point_area_distance((self.ox + i * g, self.oy + j * g), pts) < r - _EPS:
                    out.append(j * self.nx + i)
        return out

    def _apply_keepouts(self, keepouts: Sequence[Any], width: float) -> None:
        """Fold the track / via keep-outs into the static maps (module docstring); ``width`` is the widest copper of the board."""
        r_track = width / 2.0 + self.p.grid_mm / 2.0
        r_via = self.p.via_diameter_mm / 2.0
        layers_all = [*LAYERS, *self.inner_layers]
        for ko in keepouts:
            pts = area_points(ko)
            names = keepout_allowed_nets(ko)
            allowed = frozenset(self.net_index[n] for n in names if n in self.net_index)
            row: dict[str, Any] = {"id": keepout_id(ko), "allowed_nets": list(names), "track_cells": {}, "via_cells": 0}
            if forbids(ko, "tracks"):
                layers = [layer for layer, name in enumerate(LAYERS) if covers_layer(ko, name)]
                if layers:
                    self.track_keepouts.append((keepout_id(ko), pts, frozenset(layers), allowed))
                cells = self._cells_near_area(pts, r_track) if layers else []
                for layer in layers:
                    owner = self.owner[layer]
                    for k in cells:
                        cur = owner[k]
                        if len(allowed) == 1:
                            (only,) = allowed
                            owner[k] = only if cur in (None, only) else BLOCKED
                        else:
                            owner[k] = BLOCKED
                    row["track_cells"][LAYERS[layer]] = len(cells)
                if len(allowed) > 1 and layers:
                    row["note"] = (f"allows {len(allowed)} nets ({', '.join(names)}); a grid cell holds one net, so none of them is routed "
                                   "through the area")
            if forbids(ko, "vias") and any(covers_layer(ko, name) for name in layers_all):
                if self.via_ko is None:
                    self.via_ko = {}
                cells = self._cells_near_area(pts, r_via)
                for k in cells:
                    prev = self.via_ko.get(k)
                    self.via_ko[k] = allowed if prev is None else prev & allowed
                row["via_cells"] = len(cells)
            self.keepout_stats.append(row)

    def _claim(self, layer_set: tuple[int, ...], geom: _PadGeom, log: list[tuple[str, int, int, Any]]) -> None:
        """:meth:`_mark_box` + :meth:`_forbid_vias_in` for a plane net's own via / stub, recording every change in ``log`` (undo)."""
        r = self.pad_radius
        for layer in layer_set:
            owner = self.owner[layer]
            for k, dist in self.cells_near_box(geom, r):
                if dist < r - _EPS:
                    cur = owner[k]
                    new = geom.net if cur in (None, geom.net) else BLOCKED
                    if new != cur:
                        log.append(("owner", layer, k, cur))
                        owner[k] = new
        rv = self.p.via_diameter_mm / 2.0
        for k, dist in self.cells_near_box(geom, rv):
            if dist < rv - _EPS:
                if self.via_pad_ok[k]:
                    log.append(("via", 0, k, True))
                    self.via_pad_ok[k] = False

    def _wide_run_ok(self, k: int, layer: int, net: int, width: float) -> bool:
        """For a plane stub wider than the board's tracks: the cell keeps ``clearance + width/2 + grid/2`` from every foreign pad box."""
        if width <= self.p.track_width_mm + _EPS:
            return True
        x, y = self.xy(k)
        r = self.p.clearance_mm + width / 2.0 + self.p.grid_mm / 2.0
        return all(_pt_box(x, y, geom.box) >= r - _EPS for geom in self.pads if geom.net != net and layer in geom.layers)

    def _undo(self, log: list[tuple[str, int, int, Any]]) -> None:
        for kind, layer, k, old in reversed(log):
            if kind == "owner":
                self.owner[layer][k] = old
            else:
                self.via_pad_ok[k] = old

    def _plane_via_ok(self, k: int, net: int, placed: list[tuple[float, float]], areas: list[list[tuple[float, float]]]) -> bool:
        """Whether a via of plane net ``net`` may sit at cell ``k``: edge, pads, keep-outs, both layers' owners, plane-via spacing, and its
        whole disc inside one of the net's plane zones (``areas``), so the fill can surround it."""
        if not self.via_edge_ok[k] or not self.via_pad_ok[k]:
            return False
        x, y = self.xy(k)
        r = self.p.via_diameter_mm / 2.0
        if not any(point_area_distance((x, y), poly) == 0.0 and _edge_gap((x, y), poly) >= r - _EPS for poly in areas):
            return False
        if self.via_ko is not None and k in self.via_ko and net not in self.via_ko[k]:
            return False
        j, i = divmod(k, self.nx)
        for a, b in self.via_disc:
            ii, jj = i + a, j + b
            if not (0 <= ii < self.nx and 0 <= jj < self.ny):
                return False
            kk = jj * self.nx + ii
            for layer in (0, 1):
                if self.owner[layer][kk] not in (None, net):
                    return False
        spacing = self.p.via_diameter_mm + self.p.clearance_mm
        return all(math.hypot(x - vx, y - vy) >= spacing - _EPS for vx, vy in placed)

    def _plane_vias(self, plane: Mapping[str, float], areas: Mapping[str, list[list[tuple[float, float]]]]) -> None:
        """Each plane net's SMD pads get a via beside them, claimed on the static maps (module docstring); a net that cannot is undone.

        routing.maze 0.6: a pad whose terminal cell is off its copper or fenced, or whose walk finds no via site, is not a failure here:
        it waits for its escape (:meth:`_escape_all`), its net pending - its other vias claimed and seen by the nets after it while the
        net may still succeed: a pending net one of whose pads gets no escape is left out of the next pass of :meth:`_static_phase`, so
        no decision that stands was made beside its vias - until :meth:`_finish_planes`."""
        g = self.p.grid_mm
        reach = max(1, int(PLANE_VIA_REACH_MM / g + _EPS))
        placed = self._plane_placed
        for name in sorted(plane):
            idx = self.net_index[name]
            width = float(plane[name])
            log: list[tuple[str, int, int, Any]] = []
            links: list[tuple[_Terminal, int, int] | None] = []
            geoms: list[_PadGeom] = []
            tht: list[str] = []
            joined: list[str] = []
            mine: list[tuple[float, float]] = []
            deferred: list[tuple[int, _Terminal, int, str, str]] = []  # (slot in links, terminal, layer, why, the walk's reason)
            barrels = [t.pad for t in self.terminals[name] if len(t.pad.layers) != 1]
            for t in self.terminals[name]:
                if len(t.pad.layers) != 1:
                    tht.append(t.label)  # on both outer layers: a through-hole pad, whose barrel reaches the inner planes
                    continue
                if any(b.ref == t.pad.ref and b.number == t.pad.number and abs(b.cx - t.pad.cx) <= t.pad.hw + _EPS
                       and abs(b.cy - t.pad.cy) <= t.pad.hh + _EPS for b in barrels):
                    if t.label not in joined:  # the F.Cu and B.Cu copies of one exposed pad are one logical pad
                        joined.append(t.label)  # the footprint's own thermal vias (same pad number, on this pad's copper) reach the plane
                    continue
                layer = t.pad.layers[0]
                site: tuple[int, int] | None = None
                if t.offgrid is not None:
                    why, problem = "off-grid", t.offgrid
                elif self.owner[layer][t.cell] not in (None, idx):
                    why = "fenced"
                    problem = f"pad {t.label}: its terminal cell is inside a keep-out on {LAYERS[layer]} (a foreign pad, a keep-out area or the edge)"
                else:
                    site = self._plane_walk(t, idx, layer, width, reach, placed + mine, areas.get(name, []))
                    why = "no via site"
                    problem = (f"pad {t.label}: no legal via site within {PLANE_VIA_REACH_MM:g} mm beyond its copper along the grid axes from its "
                               f"terminal cell (pads, keep-outs, the edge, the other plane vias and the extent of its plane zone leave none)")
                if site is None:  # routing.maze 0.6 (module docstring: escape stubs): the escape pass looks for a stub to a via
                    deferred.append((len(links), t, layer, why, problem))
                    links.append(None)
                    continue
                k, d = site
                (x0, y0), (x1, y1) = self.xy(t.cell), self.xy(k)
                hv = self.p.via_diameter_mm / 2.0
                via = _PadGeom(t.pad.ref, f"{t.pad.number}:plane-via", x1, y1, hv, hv, hv, (0, 1), idx, kind="plane-via", shape=("disc", (x1, y1), hv))
                run = _PadGeom(t.pad.ref, f"{t.pad.number}:plane-stub", (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                               abs(x1 - x0) / 2.0 + width / 2.0, abs(y1 - y0) / 2.0 + width / 2.0, width / 2.0, (layer,), idx,
                               kind="plane-stub", shape=("seg", (x0, y0), (x1, y1), width / 2.0))
                self._claim((0, 1), via, log)
                self._claim((layer,), run, log)
                geoms += [via, run]
                links.append((t, layer, k))
                mine.append((x1, y1))
            if deferred:
                self.pads.extend(geoms)
                placed.extend(mine)
                self._pending[name] = _PendingPlane(idx, width, list(areas.get(name, [])), links, geoms, tht, joined, deferred)
                self.escape_ran = True
                continue
            self.pads.extend(geoms)
            placed.extend(mine)
            self.plane_links[name] = links
            self.plane_tht[name] = tht
            self.plane_joined[name] = joined

    def _static_phase(self, plane: Mapping[str, float], areas: Mapping[str, list[list[tuple[float, float]]]]) -> None:
        """The plane vias and the escape pass (module docstring: escape stubs), repeated while a pass dooms a net.

        A pass dooms a plane net one of whose pads no escape reaches (its other vias were claimed while it waited), and a signal net with
        a pad no escape reaches that holds an escape (a stub, or a cell reserved for it, that the unrouted net will never use). Each such
        pass dooms one net - the first such plane net in name order (the order they walk in: their vias come before every escape), else
        the signal net whose pad was refused first (footprints in natural ref order) - and the next pass starts again from the maps as
        the pads and the keep-outs left them, every doomed net left out: no via, stub, claim or reservation of it, so the others' via
        walks, escape choices and claims are made as on a board where it is never routed (as 0.4 undid a failing plane net before the
        next one walked) and a net that failed only beside a doomed net's claims is tried again. A doomed net is never taken back (a
        greedy pass is not monotone), so the passes end - at most one more per net -, and a board where no pass dooms a net runs once,
        exactly as before. The doomed nets are refused with the reasons of the pass that doomed them (:attr:`doomed`): a plane net in
        :attr:`plane_problems`, a signal net on its refused terminals, and every pad of theirs that pass escaped is listed in
        :attr:`escape_refused` as withdrawn. :attr:`escape_ran` is the last pass's (a board whose only escape news is pads none reached
        keeps its 0.2 - 0.5 maps and stamp).

        routing.maze 0.7 (module docstring: fan-out room): when these passes leave a pad without an escape, they run once more from the
        same maps with :attr:`rules07` set - the fan-out rooms (:attr:`fanout_rooms`) of the footprints left a pad of kept clear of every
        other footprint's plane vias, their escapes' ways out running to the rooms' edges - and that result is kept (:attr:`rules07` stays
        set) only when it leaves fewer nets, then fewer pads, unroutable (:meth:`_left`); otherwise the first result is restored."""
        base = self._capture()
        self._passes(plane, areas)
        if not self.escape_refused:
            return
        # routing.maze 0.7 (module docstring: fan-out room): 0.6 left a pad without an escape - the static phase runs again from the same
        # maps with the escape areas of those pads' footprints kept clear of every other footprint's plane vias, and its result is kept
        # only when it leaves fewer nets (then pads) unroutable
        first, left = self._capture(fresh=False), self._left()
        self._restore(base)
        self.rules07 = True
        m = fanout_margin(self.p)
        self.fanout_rooms = {ref: self._room_of(ref, m) for ref in sorted({label.split(".", 1)[0] for label in first["escape_refused"]}, key=natural_ref_key)}
        self._passes(plane, areas)
        if self._left() >= left:
            self._restore(first)
            self.rules07, self.fanout_rooms = False, {}

    #: the board state the static phase changes (:meth:`_capture` / :meth:`_restore`)
    _STATIC_STATE = (
        "owner", "via_pad_ok", "pads", "terminals", "escape_ran", "_eligible", "escapes", "escape_refused", "_reserved", "_reserved_why", "_bands",
        "_rule_guard", "_pending", "_plane_placed", "plane_links", "plane_tht", "plane_joined", "plane_problems", "fanout_kept", "doomed", "static_passes",
    )

    def _capture(self, fresh: bool = True) -> dict[str, Any]:
        """The static-phase state: copies (``fresh``: the state before the first pass, to start from again - the maps, the pad entries
        and the terminals copied, the rest still empty) or the objects themselves (a finished result)."""
        state = {name: getattr(self, name) for name in self._STATIC_STATE}
        if fresh:
            state = self._copied(state)
        return state

    @staticmethod
    def _copied(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["owner"] = [list(layer) for layer in state["owner"]]
        out["via_pad_ok"] = list(state["via_pad_ok"])
        out["pads"] = list(state["pads"])
        out["terminals"] = {name: list(terms) for name, terms in state["terminals"].items()}
        for name in ("_eligible", "escapes", "_bands", "_rule_guard", "_plane_placed", "fanout_kept"):
            out[name] = list(state[name])
        for name in ("escape_refused", "_reserved_why", "_pending", "plane_links", "plane_tht", "plane_joined", "plane_problems", "doomed"):
            out[name] = dict(state[name])
        out["_reserved"] = None if state["_reserved"] is None else dict(state["_reserved"])
        return out

    def _restore(self, state: dict[str, Any]) -> None:
        for name, value in self._copied(state).items():
            setattr(self, name, value)

    def _left(self) -> tuple[int, int]:
        """``(nets, pads)`` the static phase leaves unroutable: plane nets with a problem, doomed nets and nets with a refused pad."""
        nets = {name for name in self._eligible if any(t.refusal is not None for t in self.terminals[name])} | set(self.doomed) | set(self.plane_problems)
        return len(nets), len(self.escape_refused)

    def _area_of(self, ref: str) -> tuple[float, float, float, float]:
        """The escape area of footprint ``ref``: its pads' box grown by ``escape_reach_mm`` + a grid step (memoised)."""
        box = self._escape_area.get(ref)
        if box is None:
            geoms = [geom for geom in self.pads if geom.ref == ref and geom.kind == "pad"]
            grow = self.p.escape_reach_mm + self.p.grid_mm
            box = (min(g.cx - g.hw for g in geoms) - grow, min(g.cy - g.hh for g in geoms) - grow,
                   max(g.cx + g.hw for g in geoms) + grow, max(g.cy + g.hh for g in geoms) + grow)
            self._escape_area[ref] = box
        return box

    def _room_of(self, ref: str, margin: float) -> tuple[float, float, float, float]:
        """The fan-out room of footprint ``ref``: its pads' box grown by ``margin`` (:func:`fanout_margin`)."""
        geoms = [geom for geom in self.pads if geom.ref == ref and geom.kind == "pad"]
        return (min(g.cx - g.hw for g in geoms) - margin, min(g.cy - g.hh for g in geoms) - margin,
                max(g.cx + g.hw for g in geoms) + margin, max(g.cy + g.hh for g in geoms) + margin)

    def _in_fanout(self, ref: str, box: tuple[float, float, float, float]) -> str | None:
        """routing.maze 0.7: the footprint (not ``ref``) whose fan-out room the claim of copper with bounding ``box`` would enter - the box
        grown by the owner map's pad keep-out overlaps that room - else ``None``."""
        r = self.pad_radius
        for other, (x1, y1, x2, y2) in self.fanout_rooms.items():
            if other != ref and box[0] - r < x2 and x1 < box[2] + r and box[1] - r < y2 and y1 < box[3] + r:
                return other
        return None

    def _passes(self, plane: Mapping[str, float], areas: Mapping[str, list[list[tuple[float, float]]]]) -> None:
        """The passes of :meth:`_static_phase` under the rules in force (0.6's, or 0.7's with :attr:`rules07`)."""
        owner0 = [list(layer) for layer in self.owner]
        via0 = list(self.via_pad_ok)
        pads0 = len(self.pads)
        terms0 = {name: list(terms) for name, terms in self.terminals.items()}
        ran0 = self.escape_ran
        eligible0 = list(self._eligible)
        while True:
            self.static_passes += 1
            if plane:
                self._plane_vias({name: width for name, width in plane.items() if name not in self.doomed}, areas)
            self._escape_all()
            new = self._next_doomed()
            if new is None:
                break
            self.doomed[new[0]] = new[1]
            self.owner = [list(layer) for layer in owner0]
            self.via_pad_ok = list(via0)
            del self.pads[pads0:]
            self.terminals = {name: list(terms) for name, terms in terms0.items()}
            self.escape_ran = ran0
            self._eligible = [name for name in eligible0 if name not in self.doomed]
            self.escapes, self.escape_refused = [], {}
            self._reserved, self._reserved_why, self._bands, self._rule_guard = None, {}, [], []
            self._pending, self._plane_placed = {}, []
            self.plane_links, self.plane_tht, self.plane_joined, self.plane_problems = {}, {}, {}, {}
            self.fanout_kept = []
        self._finish_planes()
        for name, d in self.doomed.items():
            for q, label, why in d.refused:
                self.escape_refused[label] = why
                if q is not None:
                    self.terminals[name][q] = replace(self.terminals[name][q], refusal=why)
            first = d.refused[0][1]
            for q, label, head in d.withdrawn:
                if head is None:
                    t = self.terminals[name][q]
                    head = t.offgrid if t.offgrid is not None else self._fenced_words(t, self.net_index[name])
                self.escape_refused.setdefault(label, f"{head}; its escape is withdrawn: net {name} stays unrouted, since no escape reaches {first}")
            if d.kind == "plane":
                self.plane_problems[name] = d.reason

    def _next_doomed(self) -> tuple[str, _Doomed] | None:
        """The net this pass of the static phase dooms (:meth:`_static_phase`), or ``None``: the first pending plane net (name order) with
        a pad none reached, else the signal net with a refused pad and an escape whose first refused pad was refused first."""
        for name in sorted(self._pending):
            pend = self._pending[name]
            if pend.failed:
                refused = [(None, t.label, self.escape_refused[t.label]) for _slot, t, _l, _w, _pr in pend.deferred if t.label in self.escape_refused]
                problems = {slot: problem for slot, _t, _l, _w, problem in pend.deferred}
                withdrawn = [(None, link[0].label, problems[slot]) for slot, link in enumerate(pend.links)
                             if link is not None and link[0].escape is not None]
                return name, _Doomed("plane", pend.failed[0], refused, withdrawn)
        order = {label: i for i, label in enumerate(self.escape_refused)}
        best: tuple[int, str, _Doomed] | None = None
        for name in self._eligible:
            terms = self.terminals[name]
            refused = [(q, t.label, t.refusal) for q, t in enumerate(terms) if t.refusal is not None]
            withdrawn = [(q, t.label, None) for q, t in enumerate(terms) if t.escape is not None]
            if refused and withdrawn:
                when = min(order[label] for _q, label, _why in refused)
                if best is None or when < best[0]:
                    best = (when, name, _Doomed("signal", refused[0][2], refused, withdrawn))
        return None if best is None else (best[1], best[2])

    def _finish_planes(self) -> None:
        """Close the pending plane nets of the last pass of :meth:`_static_phase` (routing.maze 0.6): every deferred pad of each has its
        escape (a net with a pad none reached was doomed and left out of this pass), so each gets its links and its escapes are recorded."""
        for name in sorted(self._pending):
            pend = self._pending[name]
            self.plane_links[name] = [link for link in pend.links if link is not None]
            self.plane_tht[name] = pend.tht
            self.plane_joined[name] = pend.joined
            for link in pend.links:
                if link is not None and link[0].escape is not None:
                    self._record_escape(name, link[0])

    def _plane_walk(
        self, t: _Terminal, idx: int, layer: int, width: float, reach: int, placed: list[tuple[float, float]], areas: list[list[tuple[float, float]]],
    ) -> tuple[int, int] | None:
        """0.4's via site walk for one plane pad along the grid axes from its terminal cell: ``(cell, direction)`` or ``None``."""
        g = self.p.grid_mm
        rv = self.p.via_diameter_mm / 2.0
        j0, i0 = divmod(t.cell, self.nx)
        # the walk crosses the pad's own copper first: the reach counts from where a via disc clears the pad box on that axis
        # (a custom pad's whole copper, every part, in each direction)
        if t.pad.reach is None:
            clear = (math.ceil((t.pad.hw + rv) / g - _EPS) + 1, math.ceil((t.pad.hh + rv) / g - _EPS) + 1)
            limits = [reach + (clear[0] if di else clear[1]) for di, _dj in _DIRS]
        else:
            limits = [reach + math.ceil((t.pad.reach[d] + rv) / g - _EPS) + 1 for d in range(4)]
        open_dirs = [True, True, True, True]
        for step in range(1, max(limits) + 1):
            for d, (di, dj) in enumerate(_DIRS):
                if not open_dirs[d]:
                    continue
                if step > limits[d]:
                    open_dirs[d] = False
                    continue
                i, j = i0 + di * step, j0 + dj * step
                if not (0 <= i < self.nx and 0 <= j < self.ny):
                    open_dirs[d] = False
                    continue
                k = j * self.nx + i
                if self.owner[layer][k] not in (None, idx) or not self._wide_run_ok(k, layer, idx, width):
                    open_dirs[d] = False
                    continue
                if self.fanout_rooms:  # routing.maze 0.7: the via (and the run to it) stays out of another footprint's fan-out room
                    x, y = self.xy(k)
                    if self._in_fanout(t.pad.ref, (x - rv, y - rv, x + rv, y + rv)) is not None:
                        open_dirs[d] = False
                        if t.label not in self.fanout_kept:
                            self.fanout_kept.append(t.label)
                        continue
                if self._plane_via_ok(k, idx, placed, areas):
                    return (k, d)
            if not any(open_dirs):
                break
        return None

    # --- escape stubs (routing.maze 0.6) ---------------------------------------

    def _clear_of(self, net: int) -> float:
        return self.pad_clear.get(net, self.p.clearance_mm)

    def _net_width(self, idx: int) -> float:
        """The net's track width: its rule's, else the board's."""
        rule = self.net_rules.get(idx)
        return self.p.track_width_mm if rule is None else float(rule[0])

    def _reserve(self) -> dict[int, int]:
        """The cells no escape claim of another net may take (``layer * n + k`` -> net), built on first use with every usable terminal
        cell of a routed net (the escapes add theirs and their ways out, a fan the cells it holds; :attr:`_reserved_why` says which),
        together with those terminals' one-segment stubs to the pad centre (:attr:`_bands`, copper an escape keeps its clearance from);
        a usable terminal cell of a net whose rule is stricter than the owner map, and not fenced by it, joins :attr:`_rule_guard`."""
        if self._reserved is None:
            self._reserved = {}
            for name in self._eligible:
                idx = self.net_index[name]
                strict = self._strict_rule(idx)
                for t in self.terminals[name]:
                    if t.offgrid is not None:
                        continue
                    layers = tuple(layer for layer in t.pad.layers if self.owner[layer][t.cell] in (None, idx))
                    for layer in layers:
                        c = layer * self.n + t.cell
                        if c not in self._reserved:
                            self._reserved[c] = idx
                            self._reserved_why[c] = "terminal"
                        if strict and not self._rule_fenced(t.cell, layer, idx, t.pad):
                            self._rule_guard.append((c, idx, t.pad))
                    a, b = self.pos(t.cell), (_q(t.pad.cx), _q(t.pad.cy))
                    if layers and a != b:
                        half = self._net_width(idx) / 2.0
                        self._bands.append(_PadGeom(
                            t.pad.ref, f"{t.pad.number}:stub", (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, abs(b[0] - a[0]) / 2.0 + half,
                            abs(b[1] - a[1]) / 2.0 + half, half, layers, idx, kind="stub", shape=("seg", a, b, half),
                        ))
        return self._reserved

    def _centroid_distance(self, geom: _PadGeom) -> float:
        fx, fy = self.fp_centroid.get(geom.ref, (geom.cx, geom.cy))
        return round(math.hypot(geom.cx - fx, geom.cy - fy), 6)

    def _escape_all(self) -> None:
        """The escape pass (module docstring: escape stubs): every pad of a routed signal net (not a doomed one, :meth:`_static_phase`)
        whose grid cell is off its copper or fenced on every pad layer, and every waiting plane pad, gets an escape or a refusal.
        Footprints in natural ref order, each seeing the escapes before it; one footprint's pads are escaped greedily in each order of
        :data:`_ESCAPE_TRIALS` in turn (undone after each) and the first order with the fewest pads left without one - plane pads first,
        since one such pad leaves its whole plane net unrouted -, then the shortest total stub, is applied."""
        need: list[_Need] = []
        for name in self._eligible:
            idx = self.net_index[name]
            for q, t in enumerate(self.terminals[name]):
                if t.offgrid is not None or not any(self.owner[layer][t.cell] in (None, idx) for layer in t.pad.layers):
                    need.append(("signal", name, q, t, -1, "off-grid" if t.offgrid is not None else "fenced", None))
        for name, pend in self._pending.items():
            for slot, t, layer, why, problem in pend.deferred:
                need.append(("plane", name, slot, t, layer, why, problem))
        if not need:
            return
        self._reserve()
        groups: dict[str, list[_Need]] = {}
        for row in need:
            groups.setdefault(row[3].pad.ref, []).append(row)
        for ref in sorted(groups, key=natural_ref_key):
            rows = groups[ref]
            best: tuple[tuple[int, int, float, int], tuple[str, str, str]] | None = None
            trials = _ESCAPE_TRIALS if len(rows) > 1 else tuple(tr for tr in _ESCAPE_TRIALS if tr[0] == "middle" and tr[2] == "plane")
            # one pad: its candidate orders only (no row to fan, nothing to go first)
            for n_trial, trial in enumerate(trials):
                left_plane, left_signal, total = self._escape_group(rows, trial, commit=False)
                key = (left_plane, left_signal, round(total, 6), n_trial)
                if best is None or key < best[0]:
                    best = (key, trial)
                if left_plane == 0 and left_signal == 0:
                    break
            self._escape_group(rows, best[1], commit=True)

    def _escape_group(self, rows: list[_Need], trial: tuple[str, str, str], *, commit: bool) -> tuple[int, int, float]:
        """Escape one footprint's pads by the trial - pad order ``"middle"`` (nearest its copper centroid first: a row fans out from its
        middle), ``"ends"`` (farthest first), ``"natural"`` (pad order) or ``"fan"`` (:meth:`_fan` for each row of pads first, the rest
        in the middle order); candidates ``"near"`` or ``"straight"`` (:meth:`_find_escape`); the plane pads (vias) or the signal pads
        first - and return ``(plane pads left, signal pads left, total stub length)``. ``commit=False`` undoes every claim afterwards;
        ``commit=True`` applies them and records each escape or refusal."""
        order, cands, first_kind = trial

        def key(i: int) -> tuple:
            row = rows[i]
            d = self._centroid_distance(row[3].pad)
            where = -d if order == "ends" else 0.0 if order == "natural" else d
            return (0 if row[0] == first_kind else 1, where, natural_ref_key(row[3].pad.number), row[1], row[2])

        log: list[tuple[str, int, int, Any]] | None = None if commit else []
        snap = self._snapshot()
        left_plane = left_signal = 0
        total = 0.0
        fanned = self._fan(rows, cands, log, snap[2]) if order == "fan" else {}
        for i, got in sorted(fanned.items()):
            total += got.length_mm()
            if commit:
                self._keep(rows[i], got)
        for i in sorted((i for i in range(len(rows)) if i not in fanned), key=key):
            row = rows[i]
            got = self._find_row_escape(row, cands)
            if isinstance(got, str):
                kind, name, q, t, _layer, _why, problem = row
                if kind == "plane":
                    left_plane += 1
                    if commit:
                        reason = f"{problem}; {got}"
                        self._pending[name].failed.append(reason)
                        self.escape_refused[t.label] = reason
                else:
                    left_signal += 1
                    if commit:
                        head = t.offgrid if t.offgrid is not None else self._fenced_words(t, self.net_index[name])
                        refusal = f"{head}; {got}"
                        self.terminals[name][q] = replace(t, refusal=refusal)
                        self.escape_refused[t.label] = refusal
                continue
            total += got.length_mm()
            self._take(row, got, log, snap[2])
            if commit:
                self._keep(row, got)
        if log is not None:
            self._rollback(snap, log)
        return left_plane, left_signal, total

    def _snapshot(self) -> tuple[int, int, list[int], int]:
        """What :meth:`_rollback` restores: the lengths of :attr:`pads` and of the plane vias, a list for the reserved cells added
        after it, and the length of :attr:`_rule_guard`."""
        return (len(self.pads), len(self._plane_placed), [], len(self._rule_guard))

    def _rollback(self, snap: tuple[int, int, list[int], int], log: list[tuple[str, int, int, Any]]) -> None:
        self._undo(log)
        del self.pads[snap[0]:]
        del self._plane_placed[snap[1]:]
        del self._rule_guard[snap[3]:]
        for c in snap[2]:
            self._reserved.pop(c, None)
            self._reserved_why.pop(c, None)
        snap[2].clear()

    def _find_row_escape(self, row: _Need, cands: str, only: int | None = None) -> _Escape | str:
        kind, name, _q, t, _layer, why, _problem = row
        idx = self.net_index[name]
        if kind == "plane":
            pend = self._pending[name]
            return self._find_escape(t, idx, why, via=True, w_cap=pend.width, areas=pend.areas, placed=self._plane_placed, cands=cands, only=only)
        return self._find_escape(t, idx, why, cands=cands, only=only)

    def _take(self, row: _Need, got: _Escape, log: list[tuple[str, int, int, Any]] | None, added: list[int]) -> None:
        """Claim one escape (its stub and via), reserve a signal escape's cell for its net (``added`` collects the new reservations)."""
        kind, name, _q, t, _layer, _why, _problem = row
        idx = self.net_index[name]
        if got.kind != "cell":
            self._claim_escape(got, t, idx, log)
        if kind == "plane":
            self._plane_placed.append(self.xy(got.cell))
            return
        for k in (got.cell, *got.exit):
            c = got.layer * self.n + k
            if c not in self._reserved:
                added.append(c)
                self._reserved[c] = idx
                self._reserved_why[c] = "terminal" if k == got.cell else "exit"
            elif k == got.cell and c in added:  # a fan's held cell, now the escape's terminal cell (rolled back with it)
                self._reserved_why[c] = "terminal"
        if self._strict_rule(idx):  # the escape's cell obeyed the net's rule (_rule_fenced): no later escape may fence it
            self._rule_guard.append((got.layer * self.n + got.cell, idx, t.pad))

    def _keep(self, row: _Need, got: _Escape) -> None:
        """Record a committed escape: the signal pad's terminal moves to the escape's cell, a plane pad's link waits in its net."""
        kind, name, q, t, layer, _why, _problem = row
        self.escape_ran = True
        t2 = replace(t, cell=got.cell, escape=got)
        if kind == "plane":
            self._pending[name].links[q] = (t2, layer, got.cell)
            return
        self.terminals[name][q] = t2
        self._record_escape(name, t2)

    def _fan(self, rows: list[_Need], cands: str, log: list[tuple[str, int, int, Any]] | None, added: list[int]) -> dict[int, _Escape]:
        """Fan each row of the footprint's waiting pads out (module docstring: escape stubs): pads that leave in one direction from one
        edge line get escape cells in one line in front of it, as far apart as their claims need (a signal stub's cell keeps
        ``c + width/2 + grid/2`` plus its neighbour's half copper from it, a via's cell ``via_diameter + c`` from another via), centred
        on the row and at the shallowest depth where the most of them are legal; the straightest pads go first. Returns the escapes
        made (row index -> escape), claimed through ``log``."""
        p, g = self.p, self.p.grid_mm
        lines: dict[tuple[int, int, float], list[int]] = {}
        for i, row in enumerate(rows):
            geom = row[3].pad
            dirs = self._escape_dirs(geom)
            if len(dirs) != 1 or len(geom.layers) != 1:
                continue
            dx, dy = _DIRS[dirs[0]]
            ext = geom.hw if dx else geom.hh
            lines.setdefault((dirs[0], geom.layers[0], round((geom.cx + dx * ext) * dx + (geom.cy + dy * ext) * dy, 4)), []).append(i)
        out: dict[int, _Escape] = {}
        for line_key in sorted(lines):
            members = lines[line_key]
            if len(members) < 2:
                continue
            d, _layer, edge = line_key
            dx, dy = _DIRS[d]
            along_x = dx != 0

            def lateral(i: int) -> float:
                geom = rows[i][3].pad
                return ((geom.cy - self.oy) if along_x else (geom.cx - self.ox)) / g

            members.sort(key=lambda i: (lateral(i), i))
            halves: list[float] = []
            clears: list[float] = []
            for i in members:
                kind, name, _q, t, _l, _w, _pr = rows[i]
                idx = self.net_index[name]
                halves.append(p.via_diameter_mm / 2.0 if kind == "plane" else min(2.0 * t.pad.inscribed_r, self._net_width(idx)) / 2.0)
                clears.append(self._clear_of(idx))
            gaps: list[int] = []
            for k in range(len(members) - 1):
                c = max(clears[k], clears[k + 1])
                need = c + p.track_width_mm / 2.0 + g / 2.0 + max(halves[k], halves[k + 1])
                if rows[members[k]][0] == "plane" and rows[members[k + 1]][0] == "plane":
                    need = max(need, p.via_diameter_mm + c)
                gaps.append(math.ceil(need / g - 1e-9))
            span = sum(gaps)
            j0 = math.floor((lateral(members[0]) + lateral(members[-1])) / 2.0 - span / 2.0 + 0.5)
            lanes = [j0 + sum(gaps[:k]) for k in range(len(members))]
            n_axis, n_lat = (self.nx, self.ny) if along_x else (self.ny, self.nx)
            origin = self.ox if along_x else self.oy
            sign = dx if along_x else dy
            depths = sorted((f, a) for a in range(n_axis) for f in [(origin + a * g) * sign - edge] if -_EPS <= f <= p.escape_reach_mm + _EPS)
            best: tuple[int, int] | None = None
            for _f, a in depths:
                trial_snap = self._snapshot()
                sub: list[tuple[str, int, int, Any]] = []
                got = self._fan_line(rows, members, lanes, a, along_x, n_lat, cands, sub, trial_snap[2], lateral)
                self._rollback(trial_snap, sub)
                if best is None or len(got) > best[0]:
                    best = (len(got), a)
                if len(got) == len(members):
                    break
            if best is not None and best[0] > 0:
                out.update(self._fan_line(rows, members, lanes, best[1], along_x, n_lat, cands, log, added, lateral))
        return out

    def _fan_line(
        self, rows: list[_Need], members: list[int], lanes: list[int], depth: int, along_x: bool, n_lat: int, cands: str,
        log: list[tuple[str, int, int, Any]] | None, added: list[int], lateral: Callable[[int], float],
    ) -> dict[int, _Escape]:
        """One row's fan at one depth: every signal cell reserved for its net first, then each pad (the straightest first) takes the
        stub to its own cell if one is legal; a pad that gets none gives its reservation back."""
        cells: dict[int, int] = {}
        for k, i in enumerate(members):
            if 0 <= lanes[k] < n_lat:
                cells[i] = lanes[k] * self.nx + depth if along_x else depth * self.nx + lanes[k]
        pre: dict[int, int] = {}
        for i, cell in cells.items():
            kind, name, _q, t, _l, _w, _pr = rows[i]
            if kind == "plane":
                continue
            for layer in t.pad.layers:
                c = layer * self.n + cell
                if c not in self._reserved:
                    self._reserved[c] = self.net_index[name]
                    self._reserved_why[c] = "fan"
                    pre[i] = c
                    added.append(c)
        got_map: dict[int, _Escape] = {}
        order = sorted(cells, key=lambda i: (round(abs(lanes[members.index(i)] - lateral(i)), 6), lateral(i)))
        for i in order:
            got = self._find_row_escape(rows[i], cands, only=cells[i])
            if isinstance(got, str):
                continue
            self._take(rows[i], got, log, added)
            got_map[i] = got
        for i, c in pre.items():
            if i not in got_map:
                self._reserved.pop(c, None)
                self._reserved_why.pop(c, None)
                if c in added:
                    added.remove(c)
        return got_map

    def _fenced_words(self, t: _Terminal, idx: int) -> str:
        """Why the terminal cell of ``t`` is no seed on any layer of its pad (0.2's sentence, naming what reaches it)."""
        x, y = self.pos(t.cell)
        xr, yr = self.xy(t.cell)
        g, w = self.p.grid_mm, self.p.track_width_mm
        names: list[str] = []
        for geom in self.pads:
            if geom.net == idx or not set(geom.layers) & set(t.pad.layers):
                continue
            c = self.pad_clear.get(geom.net)
            r = self.pad_radius if c is None else c + w / 2.0 + g / 2.0
            if _pt_box(xr, yr, geom.box) < r - _EPS:
                number = geom.number.split(":", 1)[0] or "(unnumbered)"
                label = f"{geom.ref}.{number}" if geom.kind == "pad" else f"the {geom.kind} of {geom.ref}.{number}"
                if label not in names:
                    names.append(label)
        if self._edge_distance(t.cell) < self.p.edge_clearance_mm + w / 2.0 - _EPS:
            names.append("the board edge")
        what = ", ".join(names) if names else "a keep-out area"
        return (f"{t.label} terminal cell ({x:g}, {y:g}) is inside a keep-out on every copper layer of the pad ({what} within clearance "
                f"{self.p.clearance_mm:g} + width/2 of it)")

    def _record_escape(self, name: str, t: _Terminal) -> None:
        esc = t.escape
        row: dict[str, Any] = {"pad": t.label, "net": name, "kind": esc.kind, "why": esc.why, "layer": LAYERS[esc.layer], "cell": list(self.pos(esc.cell))}
        if esc.kind != "cell":
            row.update(width_mm=esc.width, points=[list(pt) for pt in esc.points], length_mm=_q(esc.length_mm()))
        self.escapes.append(row)

    def _escape_dirs(self, geom: _PadGeom) -> list[int]:
        """The directions (indices into :data:`_DIRS`) a stub may leave ``geom`` in: its long axis, pointing away from its footprint's
        copper centroid; a square or round pad takes the axis nearer that direction; both senses, or all four, when it is undecided."""
        fx, fy = self.fp_centroid.get(geom.ref, (geom.cx, geom.cy))
        vx, vy = geom.cx - fx, geom.cy - fy
        tol = 1e-6
        if geom.hw > geom.hh + tol:
            axis = "x"
        elif geom.hh > geom.hw + tol:
            axis = "y"
        elif abs(vx) > abs(vy) + tol:
            axis = "x"
        elif abs(vy) > abs(vx) + tol:
            axis = "y"
        else:
            return [0, 1, 2, 3]
        if axis == "x":
            return [0] if vx > tol else [2] if vx < -tol else [0, 2]
        return [1] if vy > tol else [3] if vy < -tol else [1, 3]

    def _in_pad_cell(self, t: _Terminal, idx: int, local: list[_PadGeom]) -> tuple[int, int] | None:
        """``(cell, layer)``: another grid point strictly inside the pad's inscribed circle that is free or the net's own on a layer of the
        pad, obeys the net's rule, is no other net's reserved cell, and from which the ordinary stub to the pad centre at the net's width
        keeps every clearance exactly (nearest to the centre first, then cell order, then layer); ``None`` when there is none."""
        geom, g = t.pad, self.p.grid_mm
        centre = (_q(geom.cx), _q(geom.cy))
        half = self._net_width(idx) / 2.0
        r = geom.inscribed_r
        i0, j0 = self._nearest_cell(geom.cx, geom.cy)
        span = int(r / g) + 1
        reserved = self._reserve()
        best: tuple[float, int] | None = None
        for j in range(max(0, j0 - span), min(self.ny - 1, j0 + span) + 1):
            for i in range(max(0, i0 - span), min(self.nx - 1, i0 + span) + 1):
                k = j * self.nx + i
                if k == t.cell:
                    continue
                x, y = self.pos(k)
                d = math.hypot(x - geom.cx, y - geom.cy)
                if d > r - 1e-6:
                    continue
                for layer in geom.layers:
                    if (self.owner[layer][k] in (None, idx) and reserved.get(layer * self.n + k, idx) == idx and not self._rule_fenced(k, layer, idx, geom)
                            and self._stub_room([(x, y), centre], layer, idx, local)[0] >= half - _EPS):
                        key = (round(d, 9), k, layer)
                        if best is None or key < best:
                            best = key
                        break
        return None if best is None else (best[1], best[2])

    def _strict_rule(self, idx: int) -> bool:
        """Whether net ``idx`` has a rule whose track is wider or whose clearance is larger than the board's (else the owner map is its
        fence)."""
        rule = self.net_rules.get(idx)
        return rule is not None and (rule[0] > self.p.track_width_mm + _EPS or rule[1] > self.p.clearance_mm + _EPS)

    def _rule_track(self, k: int, idx: int, geom: _PadGeom) -> tuple[float, float]:
        """``(width, clearance)`` of the net rule's track at cell ``k`` of a terminal of pad ``geom``: the neck-down width and the board's
        clearance within the neck-down radius of the pad, else the rule's (module docstring: net rules)."""
        w, c, neck, radius = self.net_rules[idx]
        x, y = self.xy(k)
        if neck is not None and radius is not None and _pt_box(x, y, geom.box) <= radius + _EPS:
            return neck, self.p.clearance_mm
        return w, c

    def _rule_fenced(self, k: int, layer: int, idx: int, geom: _PadGeom) -> bool:
        """Whether a net rule's wider track or larger clearance keeps the net's own track off cell ``k`` (its fence, module docstring:
        net rules; the neck-down width within its radius of the pad) - an escape cell must not be one."""
        if not self._strict_rule(idx):
            return False  # the owner map is the net's fence
        w, c = self._rule_track(k, idx, geom)
        return self.rule_blocked(k, layer, idx, w, c, self._clear_of)

    def _guard_hit(self, pts: list[tuple[float, float]], half: float, layer: int, via: tuple[float, float] | None, idx: int) -> str | None:
        """Why a candidate escape of net ``idx`` (a stub along ``pts`` of half width ``half`` on ``layer``, and a via at ``via`` on both
        layers) may not be claimed: its copper would put a guarded terminal cell of another net (:attr:`_rule_guard`) inside that net's
        rule fence, measured as :meth:`rule_blocker` measures an escape (exact copper); ``None`` when it puts none. So whether a rule net's
        terminal stays usable does not depend on which footprint or pad escaped first (module docstring: escape stubs)."""
        if not self._rule_guard:
            return None
        g, c_new, rv = self.p.grid_mm, self._clear_of(idx), self.p.via_diameter_mm / 2.0
        segs = list(zip(pts, pts[1:]))
        for c, other, pad in self._rule_guard:
            if other == idx:
                continue
            lay, k = divmod(c, self.n)
            if lay != layer and via is None:
                continue
            x, y = self.xy(k)
            w, cl = self._rule_track(k, other, pad)
            need = max(cl, c_new) + w / 2.0 + g / 2.0 - _EPS
            gaps = [_pt_seg(x, y, a, b) - half for a, b in segs] if lay == layer else []
            if via is not None:
                gaps.append(math.hypot(x - via[0], y - via[1]) - rv)
            if gaps and min(gaps) < need:
                px, py = self.pos(k)
                return (f"its copper would put the terminal cell ({px:g}, {py:g}) of {self._label(pad)} (net {self._net_name(other)}) inside that "
                        f"net rule's keep-out on {LAYERS[lay]}")
        return None

    def _near(self, x1: float, y1: float, x2: float, y2: float) -> list[_PadGeom]:
        """The entries of :attr:`pads` and of the ordinary stubs (:attr:`_bands`) whose box meets the given window."""
        return [geom for geom in (*self.pads, *self._bands)
                if geom.cx + geom.hw >= x1 and geom.cx - geom.hw <= x2 and geom.cy + geom.hh >= y1 and geom.cy - geom.hh <= y2]

    @staticmethod
    def _seg_gap(a: tuple[float, float], b: tuple[float, float], geom: _PadGeom) -> float:
        """Distance from the segment a-b to the copper of ``geom`` (its exact shape when it has one, else its box)."""
        shape = geom.shape
        if shape is None:
            return _seg_box(a, b, geom.box)
        if shape[0] == "seg":
            return _seg_seg(a, b, shape[1], shape[2]) - shape[3]
        return _pt_seg(shape[1][0], shape[1][1], a, b) - shape[2]

    @staticmethod
    def _pt_gap(x: float, y: float, geom: _PadGeom) -> float:
        """Distance from the point to the copper of ``geom`` (exact shape or box)."""
        shape = geom.shape
        if shape is None:
            return _pt_box(x, y, geom.box)
        if shape[0] == "seg":
            return _pt_seg(x, y, shape[1], shape[2]) - shape[3]
        return math.hypot(x - shape[1][0], y - shape[1][1]) - shape[2]

    @staticmethod
    def _label(geom: _PadGeom) -> str:
        number = geom.number.split(":", 1)[0] or "(unnumbered)"
        if geom.kind == "pad":
            return f"pad {geom.ref}.{number}"
        return {"stub": "the stub of pad", "escape": "the escape stub of pad", "escape-via": "the escape via of pad",
                "plane-stub": "the plane stub of pad", "plane-via": "the plane via of pad"}.get(geom.kind, geom.kind) + f" {geom.ref}.{number}"

    def _stub_room(self, pts: list[tuple[float, float]], layer: int, idx: int, local: list[_PadGeom]) -> tuple[float, str]:
        """The largest half width a stub along ``pts`` may have on ``layer`` (exact geometry) and what limits it."""
        c_net = self._clear_of(idx)
        room, what = _INF, ""
        segs = list(zip(pts, pts[1:]))
        for geom in local:
            if geom.net == idx or layer not in geom.layers:
                continue
            req = max(c_net, self._clear_of(geom.net) if geom.net >= 0 else self.p.clearance_mm)
            for a, b in segs:
                h = self._seg_gap(a, b, geom) - req
                if h < room:
                    room, what = h, f"{self._label(geom)} (needs {req:g} mm)"
        for ko, area, layers, allowed in self.track_keepouts:
            if layer not in layers or (len(allowed) == 1 and idx in allowed):
                continue
            for a, b in segs:
                h = segment_area_distance(a, b, area)
                if h < room:
                    room, what = h, f"keep-out {ko}"
        for pt in pts:  # the outline is a rectangle: a segment is nearest the edge at an end
            h = self.edge_distance_xy(*pt) - self.p.edge_clearance_mm
            if h < room:
                room, what = h, "the board edge"
        return room, what

    def _claim_cells(self, pts: list[tuple[float, float]], width: float, layers: tuple[int, ...], idx: int, via: tuple[float, float] | None) -> list[int]:
        """The cells (``layer * n + k``) a stub along ``pts`` of ``width`` on ``layers`` (and a via at ``via``, on both layers) claims for
        ``idx``: within ``c + track_width/2 + grid/2`` of their copper, as :meth:`_mark_box` claims around a pad."""
        g = self.p.grid_mm
        r = self._clear_of(idx) + self.p.track_width_mm / 2.0 + g / 2.0
        out: set[int] = set()
        shapes: list[tuple[tuple[float, float], tuple[float, float], float, tuple[int, ...]]] = [(a, b, width / 2.0, layers) for a, b in zip(pts, pts[1:])]
        if via is not None:
            shapes.append((via, via, self.p.via_diameter_mm / 2.0, (0, 1)))
        for a, b, half, lays in shapes:
            reach = r + half
            i0, j0 = self._nearest_cell(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach)
            i1, j1 = self._nearest_cell(max(a[0], b[0]) + reach, max(a[1], b[1]) + reach)
            for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
                for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                    k = j * self.nx + i
                    x, y = self.xy(k)
                    if _pt_seg(x, y, a, b) < reach - _EPS:
                        for layer in lays:
                            out.add(layer * self.n + k)
        return sorted(out)

    def _find_escape(
        self, t: _Terminal, idx: int, why: str, *, via: bool = False, w_cap: float | None = None,
        areas: Sequence[list[tuple[float, float]]] = (), placed: Sequence[tuple[float, float]] = (), cands: str = "near",
        only: int | None = None,
    ) -> _Escape | str:
        """The escape of one pad (module docstring: escape stubs), or why none exists (a sentence naming the nearest candidate's problem).

        A signal pad first tries another grid point inside its inscribed circle (``kind="cell"``), then a stub to a free grid cell with a
        way out; a plane pad (``via=True``) a stub to a via obeying the plane-via rules exactly (``areas``: the net's plane zones,
        ``placed``: the plane vias so far, ``w_cap``: the plane stubs' width). ``cands`` orders the candidates (``"near"`` /
        ``"straight"``), ``only`` restricts them to one cell (a fan's)."""
        p, g, n = self.p, self.p.grid_mm, self.n
        geom = t.pad
        w_net = (p.track_width_mm if w_cap is None else w_cap) if via else self._net_width(idx)
        w_top = min(2.0 * geom.inscribed_r, w_net)
        w_min = self.escape_min_width
        step = p.escape_width_step_mm
        reach = p.escape_reach_mm
        if w_top < w_min - _EPS:
            return f"no escape stub: its narrow side ({2.0 * geom.inscribed_r:g} mm) is below the narrowest stub {w_min:g} mm"
        rv = p.via_diameter_mm / 2.0
        reserved = self._reserve()
        c_max = max(p.clearance_mm, *self.pad_clear.values()) if self.pad_clear else p.clearance_mm
        margin = max(geom.hw, geom.hh) + 1.5 * reach + max(w_top, w_net) + 2.0 * c_max + 2.0 * rv + g
        local = self._near(geom.cx - margin, geom.cy - margin, geom.cx + margin, geom.cy + margin)
        if not via and t.offgrid is None and only is None:
            inside = self._in_pad_cell(t, idx, local)
            if inside is not None:
                return _Escape("cell", why, inside[1], inside[0])
        centre = (_q(geom.cx), _q(geom.cy))
        noun = "via site" if via else "cell"
        first: str | None = None
        free_seen = False
        for layer in geom.layers:
            for d in self._escape_dirs(geom):
                dx, dy = _DIRS[d]
                lx, ly = -dy, dx
                ext = geom.hw if dx else geom.hh
                ox, oy = geom.cx + dx * ext, geom.cy + dy * ext
                fx, fy = self.fp_centroid.get(geom.ref, (geom.cx, geom.cy))
                side = (geom.cx - fx) * lx + (geom.cy - fy) * ly
                i0, j0 = self._nearest_cell(ox - reach, oy - reach)
                i1, j1 = self._nearest_cell(ox + reach, oy + reach)
                found: list[tuple[tuple, int, float, float]] = []
                for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
                    for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                        k = j * self.nx + i
                        if only is not None and k != only:
                            continue
                        x, y = self.xy(k)
                        f = (x - ox) * dx + (y - oy) * dy
                        lat = (x - ox) * lx + (y - oy) * ly
                        if f < -_EPS or f > reach + _EPS or abs(lat) > reach + _EPS:
                            continue
                        away = 0 if abs(lat) < _EPS or lat * side > 0 else 1
                        near = (round(math.hypot(f, lat), 9), round(abs(lat), 9), away, k)
                        found.append(((0 if abs(lat) < _EPS else 1, *near) if cands == "straight" else near, k, f, lat))
                found.sort()
                for _key, k, f, lat in found:
                    ex, ey = self.pos(k)
                    if via:
                        problem = self._via_site_problem(k, idx, areas, placed, local, geom.ref)
                    else:
                        problem = None if self.owner[layer][k] in (None, idx) else _SKIP
                        if problem is None and reserved.get(layer * n + k, idx) != idx:
                            problem = f"it is {self._reserved_words(layer * n + k, False)}"
                        if problem is None and self._rule_fenced(k, layer, idx, geom):
                            problem = "the net rule's wider track or larger clearance keeps its track off it"
                    if problem == _SKIP:
                        continue
                    free_seen = True
                    if problem is not None:
                        first = first or f"the nearest candidate {noun} ({ex:g}, {ey:g}): {problem}"
                        continue
                    a_steps = [0] if abs(lat) < _EPS else list(range(0, int(f / (g / 2.0) + _EPS) + 1))
                    for a_i in a_steps:
                        a = a_i * g / 2.0
                        if abs(lat) < _EPS:
                            pts = [centre, (ex, ey)]
                        else:
                            bend = (_q(ox + dx * a), _q(oy + dy * a))
                            pts = [centre, bend, (ex, ey)] if bend != (ex, ey) else [centre, (ex, ey)]
                        room, what = self._stub_room(pts, layer, idx, local)
                        width = _q(min(w_top, math.floor((2.0 * room + 1e-9) / step) * step))
                        if width < w_min - _EPS:
                            first = first or (f"the nearest candidate {noun} ({ex:g}, {ey:g}): a stub keeping its clearance from {what} could be at "
                                              f"most {max(2.0 * room, 0.0):.3f} mm wide, below the narrowest stub {w_min:g} mm")
                            continue
                        hit = self._guard_hit(pts, width / 2.0, layer, (ex, ey) if via else None, idx)
                        if hit is not None:
                            first = first or f"the nearest candidate {noun} ({ex:g}, {ey:g}): {hit}"
                            continue
                        cells = self._claim_cells(pts, width, (layer,), idx, (ex, ey) if via else None)
                        taken = next((c for c in cells if reserved.get(c, idx) != idx), None)
                        if taken is not None:
                            first = first or (f"the nearest candidate {noun} ({ex:g}, {ey:g}): its clearance would take "
                                              f"{self._reserved_words(taken, True)}")
                            continue
                        if via:
                            return _Escape("via", why, layer, k, width, tuple(pts))
                        way = self._exit_path(k, layer, idx, geom.ref)
                        if way is None:
                            first = first or (f"the nearest candidate cell ({ex:g}, {ey:g}): no way out of the footprint's escape area from it on "
                                              f"{LAYERS[layer]} (every path meets another net's pad, stub or edge clearance)")
                            break  # another bend does not open a way out of this cell
                        return _Escape("stub", why, layer, k, width, tuple(pts), tuple(way))
        where = f"within {reach:g} mm in front of its edge"
        if not free_seen:
            if via:
                return (f"no escape stub: no via site {where} lies inside the net's plane zone clear of the edge, every pad and the via "
                        "keep-outs")
            return (f"no escape stub: every grid cell {where} is inside a foreign pad's, the edge's or a keep-out's clearance, or another "
                    "escape's")
        return f"no escape stub {where}: {first or 'no candidate is legal'}"

    def _exit_path(self, start: int, layer: int, idx: int, ref: str) -> list[int] | None:
        """The shortest 4-neighbour way (cells on ``layer``; ties in the order east, south, west, north) from ``start`` to the first cell
        outside the escape area of footprint ``ref`` (its pads' box grown by ``escape_reach_mm`` + a grid step; under routing.maze 0.7
        its fan-out room when it has one) through cells free or the net's own and reserved for no other net (the candidate's own claim
        takes only free cells, so it closes none); ``None`` when there is none."""
        # routing.maze 0.7 (module docstring: fan-out room): the way out of a footprint in a fan-out room runs to that room's edge
        x1, y1, x2, y2 = self.fanout_rooms.get(ref) or self._area_of(ref)
        owner, reserved, base, nx, ny = self.owner[layer], self._reserved, layer * self.n, self.nx, self.ny

        def ok(k: int) -> bool:
            return owner[k] in (None, idx) and reserved.get(base + k, idx) == idx

        prev: dict[int, int] = {start: -1}
        queue = deque([start])
        while queue:
            k = queue.popleft()
            x, y = self.xy(k)
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                path = [k]
                while prev[path[-1]] != -1:
                    path.append(prev[path[-1]])
                return path[::-1]
            j, i = divmod(k, nx)
            for di, dj in _DIRS:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    kk = jj * nx + ii
                    if kk not in prev and ok(kk):
                        prev[kk] = k
                        queue.append(kk)
        return None

    def _reserved_words(self, c: int, taken: bool) -> str:
        """What the reserved cell ``c`` (``layer * n + k``) is, as a refusal names it (:data:`_RESERVED_WORDS`; ``taken``: the form after
        "would take")."""
        return f"{_RESERVED_WORDS[self._reserved_why[c]][taken]} net {self._net_name(self._reserved[c])}"

    def _net_name(self, idx: int) -> str:
        for name, k in self.net_index.items():
            if k == idx:
                return name
        return "?"

    def _via_site_problem(
        self, k: int, idx: int, areas: Sequence[list[tuple[float, float]]], placed: Sequence[tuple[float, float]], local: list[_PadGeom],
        ref: str = "",
    ) -> str | None:
        """Why a plane escape's via may not sit at cell ``k`` (exact rules, module docstring), :data:`_SKIP` when the cell is not even a
        candidate (the edge, a pad under its disc, a via keep-out, outside the plane zones), ``None`` when it may."""
        p = self.p
        rv = p.via_diameter_mm / 2.0
        if not self.via_edge_ok[k] or not self.via_pad_ok[k]:
            return _SKIP
        x, y = self.xy(k)
        if not any(point_area_distance((x, y), poly) == 0.0 and _edge_gap((x, y), poly) >= rv - _EPS for poly in areas):
            return _SKIP
        if self.via_ko is not None and k in self.via_ko and idx not in self.via_ko[k]:
            return _SKIP
        c_net = self._clear_of(idx)
        for other in local:
            if other.net == idx:
                continue
            req = max(c_net, self._clear_of(other.net) if other.net >= 0 else p.clearance_mm)
            gap = self._pt_gap(x, y, other) - rv
            if gap < req - _EPS:
                return f"the via would come {max(gap, 0.0):.3f} mm from {self._label(other)} (needs {req:g} mm)"
        spacing = p.via_diameter_mm + p.clearance_mm
        if any(math.hypot(x - vx, y - vy) < spacing - _EPS for vx, vy in placed):
            return f"the via would come nearer than {spacing:g} mm to another plane via"
        other = self._in_fanout(ref, (x - rv, y - rv, x + rv, y + rv)) if self.fanout_rooms else None
        if other is not None:  # routing.maze 0.7 (module docstring: fan-out room)
            return f"the via would lie in the fan-out room of {other}"
        return None

    def _claim_escape(self, esc: _Escape, t: _Terminal, idx: int, log: list[tuple[str, int, int, Any]] | None) -> list[_PadGeom]:
        """Claim the escape's cells on the owner maps (and, for a via, keep other vias off it), add its entries to :attr:`pads` and
        return them; ``log`` (a trial) records every change for the undo."""
        pts = list(esc.points)
        via_xy = self.xy(esc.cell) if esc.kind == "via" else None
        for c in self._claim_cells(pts, esc.width, (esc.layer,), idx, via_xy):
            layer, k = divmod(c, self.n)
            cur = self.owner[layer][k]
            new = idx if cur in (None, idx) else BLOCKED
            if new != cur:
                if log is not None:
                    log.append(("owner", layer, k, cur))
                self.owner[layer][k] = new
        geoms: list[_PadGeom] = []
        half = esc.width / 2.0
        for a, b in zip(pts, pts[1:]):
            geoms.append(_PadGeom(
                t.pad.ref, f"{t.pad.number}:escape", (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, abs(b[0] - a[0]) / 2.0 + half, abs(b[1] - a[1]) / 2.0 + half,
                half, (esc.layer,), idx, kind="escape", shape=("seg", a, b, half),
            ))
        if via_xy is not None:
            rv = self.p.via_diameter_mm / 2.0
            geoms.append(_PadGeom(t.pad.ref, f"{t.pad.number}:escape-via", via_xy[0], via_xy[1], rv, rv, rv, (0, 1), idx, kind="escape-via",
                                  shape=("disc", via_xy, rv)))
            i0, j0 = self._nearest_cell(via_xy[0] - 2 * rv, via_xy[1] - 2 * rv)
            i1, j1 = self._nearest_cell(via_xy[0] + 2 * rv, via_xy[1] + 2 * rv)
            for j in range(max(0, j0 - 1), min(self.ny - 1, j1 + 1) + 1):
                for i in range(max(0, i0 - 1), min(self.nx - 1, i1 + 1) + 1):
                    k = j * self.nx + i
                    x, y = self.xy(k)
                    if math.hypot(x - via_xy[0], y - via_xy[1]) < 2 * rv - _EPS:
                        if self.via_pad_ok[k]:
                            if log is not None:
                                log.append(("via", 0, k, True))
                            self.via_pad_ok[k] = False
        self.pads.extend(geoms)
        return geoms

    def usable_layers(self, terminal: _Terminal, net: int) -> tuple[int, ...]:
        """The pad's copper layers on which the terminal cell is free or the net's own (never BLOCKED / another net's); an escape stub's
        own layer only (routing.maze 0.6)."""
        esc = terminal.escape
        layers = terminal.pad.layers if esc is None or esc.kind == "cell" else (esc.layer,)
        return tuple(layer for layer in layers if self.owner[layer][terminal.cell] in (None, net))

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
        if self.via_ko is not None and value != BLOCKED and k in self.via_ko:
            allowed = self.via_ko[k]  # a via keep-out: only its allowed nets, and a free cell only when exactly one is allowed
            if value == _FREE:
                value = next(iter(allowed)) if len(allowed) == 1 else BLOCKED
            elif value not in allowed:
                value = BLOCKED
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

    @staticmethod
    def _fence_gap(x: float, y: float, geom: _PadGeom) -> float:
        """The distance a net rule's fence measures from the point to a foreign entry of :attr:`pads`: an escape stub or escape via
        (routing.maze 0.6, :data:`_EXACT_FENCE_KINDS`) by its exact copper, as the escape pass and the owner map measure it; a footprint
        pad, a 0.4 plane via or plane stub by its box, as before."""
        return _Board._pt_gap(x, y, geom) if geom.kind in _EXACT_FENCE_KINDS else _pt_box(x, y, geom.box)

    def rule_blocker(self, k: int, layer: int, net: int, width: float, clearance: float, clear_of: Callable[[int], float]) -> str | None:
        """What keeps a track of ``net`` at cell ``k`` on ``layer`` with this width / clearance off it: ``"the board edge"``, or the
        label of the first foreign entry of :attr:`pads` (:meth:`_fence_gap`) within ``max(clearance, its net's) + width/2 + grid/2`` of
        the cell; ``None`` when nothing does."""
        g = self.p.grid_mm
        if self._edge_distance(k) < self.p.edge_clearance_mm + width / 2.0 - _EPS:
            return "the board edge"
        x, y = self.xy(k)
        for geom in self.pads:
            if geom.net == net or layer not in geom.layers:
                continue
            if self._fence_gap(x, y, geom) < max(clearance, clear_of(geom.net)) + width / 2.0 + g / 2.0 - _EPS:
                return self._label(geom)
        return None

    def rule_blocked(self, k: int, layer: int, net: int, width: float, clearance: float, clear_of: Callable[[int], float]) -> bool:
        """Whether a track of ``net`` at cell ``k`` on ``layer`` with this width / clearance comes too near a foreign pad, escape or the
        edge (:meth:`rule_blocker`)."""
        return self.rule_blocker(k, layer, net, width, clearance, clear_of) is not None

    def rule_fence(
        self, net: int, width: float, clearance: float, zone: bytearray | None, neck: float | None, neck_clearance: float,
        clear_of: Callable[[int], float],
    ) -> bytearray:
        """``layer * n + k`` -> 1 where the net's track (neck-down width inside ``zone``) breaks its rule at a foreign pad, escape or the
        edge (each entry measured as :meth:`_fence_gap` measures it)."""
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
            exact = geom.kind in _EXACT_FENCE_KINDS
            for k, dist in self.cells_near_box(geom, max(r_main, r_neck)):
                if exact:  # the box's cells are a superset of the exact copper's
                    dist = self._fence_gap(*self.xy(k), geom)
                r = r_neck if zone is not None and zone[k] else r_main
                if dist < r - _EPS:
                    for layer in geom.layers:
                        fence[layer * n + k] = 1
        return fence

    def rule_via_fence(self, net: int, clearance: float, clear_of: Callable[[int], float]) -> bytearray:
        """``k`` -> 1 where a via of ``net`` would come closer than ``via_diameter/2 + max(c, c_pad)`` to a foreign pad box (an escape
        stub or via: its exact copper, :meth:`_fence_gap`)."""
        fence = bytearray(self.n)
        r0 = self.p.via_diameter_mm / 2.0
        for geom in self.pads:
            if geom.net == net:
                continue
            r = r0 + max(clearance, clear_of(geom.net))
            exact = geom.kind in _EXACT_FENCE_KINDS
            for k, dist in self.cells_near_box(geom, r):
                if exact:
                    dist = self._fence_gap(*self.xy(k), geom)
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
    """Length of the stub from the terminal cell to the pad centre, as :class:`_NetCopper` emits it (an escape stub's polyline, 0.6)."""
    if t.escape is not None and t.escape.kind != "cell":
        return t.escape.length_mm()
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
    refused = [t for t in terms if t.refusal is not None]
    if refused:  # routing.maze 0.6: a pad that needed an escape stub and has none (its refusal names it and why)
        return refused[0].refusal
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
        before = usable
        usable = [tuple(layer for layer in layers if not fence[layer * n + t.cell]) for t, layers in zip(terms, usable)]
        fenced = [q for q, layers in enumerate(usable) if not layers]
        if fenced:
            t = terms[fenced[0]]
            x, y = board.pos(t.cell)
            w, c = rn.nd_key if rn.zone is not None and rn.zone[t.cell] else rn.key
            # what fences the cell on each layer the owner map left it (as the fence measures it: an escape by its exact copper)
            what = {layer: board.rule_blocker(t.cell, layer, idx, w, c, board._clear_of) or "its fence" for layer in before[fenced[0]]}
            names = list(dict.fromkeys(what.values()))
            where = names[0] if len(names) == 1 else "; ".join(f"{LAYERS[layer]}: {name}" for layer, name in what.items())
            pitch = ": the pad pitch cannot take that width" if all(name.startswith("pad ") for name in names) else ""
            return (
                f"{t.label} terminal cell ({x:g}, {y:g}) is inside its net rule's keep-out on every copper layer of the pad "
                f"({where} {'is' if len(names) == 1 else 'are'} within clearance {c:g} + width {w:g}/2 of it{pitch})"
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
    for _ko, pts, layers, _allowed in board.track_keepouts:  # routing.maze 0.4: the envelope never enters a track keep-out
        for k in board._cells_near_area(pts, W / 2.0 + g / 2.0):
            for layer in layers:
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
        for ko, pts, layers, allowed in board.track_keepouts:  # routing.maze 0.4
            if layer in layers and own not in allowed and segment_area_distance(seg[0], seg[1], pts) < w / 2.0 - _EPS:
                return f"the breakout of {name} enters keep-out {ko}"
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
            label = f"pad {geom.ref}.{geom.number or '(unnumbered)'}"
            if geom.kind in ("escape", "escape-via"):  # routing.maze 0.6: an escape stub / via is audited as its exact copper
                if geom.shape[0] == "seg":
                    self._add(("seg", geom.net, frozenset(geom.layers), (geom.shape[1], geom.shape[2]), geom.shape[3], c, f"escape stub of {label}"), geom.box)
                else:
                    self._add(("disc", geom.net, frozenset(geom.layers), geom.shape[1], geom.shape[2], c, f"escape via of {label}"), geom.box)
                continue
            self._add(("box", geom.net, frozenset(geom.layers), geom.box, 0.0, c, label), geom.box)

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


@dataclass(frozen=True, slots=True)
class _Plan:
    """routing.maze 0.4 (module docstring): the stamp, the extra ``derived_from`` entries and the note words of a board with keep-outs / plane nets."""

    version: str
    derived: tuple[str, ...]
    note: str
    #: routing.maze 0.6: the board has an escape stub, so the params entry records the escape knobs
    escape: bool = False


def _provenance(
    net: Net, p: RoutingParams, how: str, iterations: int, legal: bool, *, version: str = ROUTER_VERSION, rules_active: bool = False,
    rule: NetRule | None = None, extra_refs: list[str] | None = None, story: str | None = None, rule_note: str | None = None,
    plan: _Plan | None = None,
) -> Provenance:
    refs = sorted({pin.component_ref for pin in net.pins} | set(extra_refs or []))
    if story is None:
        if how == "recovered":
            story = f"routed against the legal copper as an obstacle after {iterations} negotiation iteration(s) left it in conflict"
        else:
            story = f"negotiated-congestion route, {iterations} iteration(s)" + ("" if legal else ", kept after the conflicting nets were ripped up")
    derived = [f"net:{net.name}", *(f"placement:{r}" for r in refs), p.derived_from_entry(rules=rules_active, escapes=plan is not None and plan.escape)]
    if rule is not None:
        derived.append(rule.derived_from_entry())
    rule_text = f"; {rule_note}" if rule_note else ""
    if plan is not None:
        version = plan.version
        derived.extend(plan.derived)
        rule_text += f"; {plan.note}"
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
        """One segment from the terminal cell to the exact pad centre (nothing when the cell is the centre); an escape stub's segments
        (routing.maze 0.6) from its cell back to the pad centre, at its own width, each naming the escape in its provenance."""
        esc = terminal.escape
        if esc is not None and esc.kind != "cell":
            prov = _escape_provenance(self.prov, terminal, self.board.p)
            pts = list(reversed(esc.points))
            for a, b in zip(pts, pts[1:]):
                if a != b:
                    self.tracks.append(Track(net=self.net.name, layer=LAYERS[esc.layer], start=a, end=b, width_mm=esc.width, provenance=prov))
            return
        start = self.board.pos(terminal.cell)
        end = (_q(terminal.pad.cx), _q(terminal.pad.cy))
        width = self.neck if self.zone is not None and self.zone[terminal.cell] else self.width
        if start != end:
            self.tracks.append(Track(net=self.net.name, layer=LAYERS[layer], start=start, end=end, width_mm=width, provenance=self.prov))

    def length_mm(self) -> float:
        return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in self.tracks)


#: the ``derived_from`` prefix of an escape stub's segments (routing.maze 0.6): ``escape:<pad>:<stub|via>:width=<w>:points=x,y;x,y;...``
ESCAPE_ENTRY_PREFIX = "escape:"


def escape_entry(track: Track) -> tuple[str, float, list[tuple[float, float]]] | None:
    """``(pad label, width, points)`` of the escape stub a track is a segment of (its provenance's :data:`ESCAPE_ENTRY_PREFIX` entry),
    or ``None`` for any other track."""
    for entry in track.provenance.derived_from:
        if not entry.startswith(ESCAPE_ENTRY_PREFIX):
            continue
        try:
            label, _kind, width, points = entry[len(ESCAPE_ENTRY_PREFIX):].rsplit(":", 3)
            pts = [(float(x), float(y)) for x, y in (pt.split(",") for pt in points.removeprefix("points=").split(";"))]
            return label, float(width.removeprefix("width=")), pts
        except ValueError:
            return None
    return None


def _escape_provenance(prov: Provenance, t: _Terminal, p: RoutingParams) -> Provenance:
    """The net's provenance for the segments of ``t``'s escape stub: an ``escape:<pad>:...`` entry in ``derived_from`` (its kind, width and
    points, :func:`escape_entry`), the stub's geometry in the note."""
    esc = t.escape
    why = {"off-grid": "no grid point lies inside its inscribed circle", "fenced": "its grid cell is inside its neighbours' keep-out on every layer",
           "no via site": "the axis walk found no via site"}.get(esc.why, esc.why)
    path = " -> ".join(f"({x:g}, {y:g})" for x, y in esc.points)
    target = "its via" if esc.kind == "via" else "the grid cell the track starts from"
    points = ";".join(f"{x!r},{y!r}" for x, y in esc.points)
    return prov.model_copy(update={
        "derived_from": [*prov.derived_from, f"{ESCAPE_ENTRY_PREFIX}{t.label}:{esc.kind}:width={esc.width}:points={points}"],
        "note": (f"{prov.note}; escape stub of pad {t.label} (routing.maze {ROUTER_ESCAPE_VERSION}: {why}): {path} on {LAYERS[esc.layer]} at "
                 f"{esc.width:g} mm, from the pad centre out of its edge to {target} (every clearance checked exactly; "
                 f"escape_reach {p.escape_reach_mm:g} mm)"),
    })


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
    keepouts: Sequence[Any] | None = None, plane_nets: Mapping[str, Sequence[tuple[str, Sequence[tuple[float, float]]]]] | None = None,
) -> Routing:
    """Route every net of the placed board on ``F.Cu`` / ``B.Cu``; pure (same IR + library + params + rules + keep-outs -> same result).

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
    reports progress; it never changes the result). ``keepouts`` (keep-out
    areas, read through :mod:`ai_eda.tools.keepout`) and ``plane_nets`` (net
    name -> ``(layer, polygon)`` of each plane zone the caller gives the net;
    a pad via must lie wholly inside one of them) make it
    routing.maze 0.4 (module docstring: keep-outs and plane nets); a plane
    net's rule width becomes its pad stubs' width and the rest of its rule
    does not apply (no track joins its pads; ``stats["plane_nets"]`` says
    so), and a plane net in a coupled pair is refused. Raises
    :class:`CompileError` for anything that would need a guess (module
    docstring).
    """
    p, raised = effective_params(ir, params)
    eff, rule_raised = _effective_rules(ir, p, rules)
    index = {net.name: k for k, net in enumerate(ir.nets)}
    plan: _Plan | None = None
    plane_width: dict[str, float] = {}
    plane_rules: dict[str, NetRule] = {}
    plane_layers: dict[str, list[str]] = {}
    plane_areas: dict[str, list[list[tuple[float, float]]]] = {}
    if keepouts or plane_nets:
        for name in sorted(plane_nets or {}):
            if name not in index:
                raise CompileError(f"plane net {name!r}: the IR has no such net")
            partner = eff[name].pair_partner if name in eff else None
            if partner is not None or any(r.pair_partner == name for r in eff.values()):
                raise CompileError(f"plane net {name!r} is in a coupled pair: a plane net's pads are joined by vias, never by a coupled track")
            entries = list(plane_nets[name])
            if not entries:
                raise CompileError(f"plane net {name!r}: no plane zone given (a pad via must land inside one)")
            plane_layers[name] = list(dict.fromkeys(str(layer) for layer, _ in entries))
            plane_areas[name] = [[(float(x), float(y)) for x, y in poly] for _, poly in entries]
            rule = eff.pop(name, None)
            if rule is not None:
                plane_rules[name] = rule
            plane_width[name] = p.track_width_mm if rule is None else max(p.track_width_mm, float(rule.width_mm))
        ids = [keepout_id(k) for k in keepouts or ()]
        derived = []
        if ids:
            derived.append("keepouts:" + ",".join(ids))
        if plane_width:
            derived.append("plane_nets:" + ",".join(f"{n}={'+'.join(plane_layers[n])}" for n in sorted(plane_width)) + f";reach={PLANE_VIA_REACH_MM}")
        words = []
        if ids:
            words.append(f"keep-outs {', '.join(ids)} are obstacles")
        if plane_width:
            words.append(f"plane net(s) {', '.join(sorted(plane_width))} join their pads by vias to the plane, never by tracks")
        plan = _Plan(ROUTER_KEEPOUT_VERSION, tuple(derived), f"routing.maze {ROUTER_KEEPOUT_VERSION}: " + "; ".join(words))
    widths = [p.track_width_mm, *(float(r.width_mm) for r in eff.values()), *plane_width.values()]
    widths += [2.0 * float(r.width_mm) + float(r.pair_spacing_mm) for r in eff.values() if r.pair_partner is not None]
    pair_nets = sorted({name for name, r in eff.items() if r.pair_partner is not None})
    board = _Board(
        ir, library, p, {index[name]: r.clearance_mm for name, r in eff.items()}, inner_layers,
        keepouts=list(keepouts or ()) or None, ko_width=max(widths), plane=plane_width or None, plane_areas=plane_areas or None,
        no_escape=pair_nets,
        net_rules={index[name]: (float(r.width_mm), float(r.clearance_mm), r.neckdown_width_mm, r.neckdown_radius_mm) for name, r in eff.items()},
    )
    if board.escape_ran and board.escape_min_width > p.escape_min_width_mm + _EPS:
        # routing.maze 0.6: the fab's min_track_width_mm raised the narrowest escape stub - recorded like every other raise, and only on a
        # board with escapes (the knob is in no other board's provenance)
        raised["escape_min_width_mm"] = (p.escape_min_width_mm, board.escape_min_width)
        p = replace(p, escape_min_width_mm=board.escape_min_width)
    if board.custom_pads:  # routing.maze 0.5 (module docstring: custom pads): its own stamp and the pads it modelled, on every item
        entry = "custom_pads:" + ",".join(board.custom_pads) + ";model=anchor+primitive_boxes"
        words = (f"custom pad(s) {', '.join(board.custom_pads)} are obstacles as the box of their anchor plus one box around each copper "
                 "primitive; a track lands on the anchor")
        head = f"routing.maze {ROUTER_CUSTOM_PAD_VERSION}: "
        if plan is None:
            plan = _Plan(ROUTER_CUSTOM_PAD_VERSION, (entry,), head + words)
        else:
            plan = _Plan(ROUTER_CUSTOM_PAD_VERSION, (*plan.derived, entry), head + plan.note.removeprefix(f"routing.maze {ROUTER_KEEPOUT_VERSION}: ") + "; " + words)
    planes_or_pads = plan is not None  # a 0.4 / 0.5 board: its plane copper and stats are added after the nets (_add_planes)
    if board.escape_ran:  # routing.maze 0.6 (module docstring: escape stubs): its own stamp, the escape knobs and the escaped pads, on every item
        labels = [row["pad"] for row in board.escapes]
        refused = sorted(board.escape_refused, key=lambda label: [natural_ref_key(part) for part in label.split(".", 1)])
        entry = "escapes:" + ",".join(labels) + (";refused=" + ",".join(refused) if refused else "") + ";model=centre+edge+cell"
        words = (f"pad(s) {', '.join(labels) or 'none'} reach the grid by an escape (another grid point inside the pad, or a stub from the "
                 "pad centre out of its edge to a grid cell or a plane via, every clearance checked exactly) because their own grid cell is "
                 "off their copper or inside their neighbours' keep-out" + (f"; no escape reaches {', '.join(refused)}" if refused else ""))
        head = f"routing.maze {ROUTER_ESCAPE_VERSION}: "
        if plan is None:
            plan = _Plan(ROUTER_ESCAPE_VERSION, (entry,), head + words, escape=True)
        else:
            plan = _Plan(ROUTER_ESCAPE_VERSION, (*plan.derived, entry), head + plan.note.removeprefix(f"routing.maze {plan.version}: ") + "; " + words, escape=True)
    if board.rules07:  # routing.maze 0.7 (module docstring: fan-out room): its own stamp, the kept areas and the vias kept out, on every item
        margin = fanout_margin(board.p)
        entry = (f"fanout_room:{','.join(board.fanout_rooms)};margin={margin}" + (";kept_out=" + ",".join(board.fanout_kept) if board.fanout_kept else "")
                 + ";model=pads_box+margin")
        words = (f"the fan-out rooms of {', '.join(board.fanout_rooms)} (their pads' box grown by {margin:g} mm; 0.6 left a pad of each without an "
                 "escape) are kept clear of every other footprint's plane vias" + (f" ({', '.join(board.fanout_kept)} took another site)" if board.fanout_kept else "")
                 + " and their escapes' ways out run to the room's edge")
        head = f"routing.maze {ROUTER_FANOUT_VERSION}: "
        if plan is None:
            plan = _Plan(ROUTER_FANOUT_VERSION, (entry,), head + words, escape=True)
        else:
            plan = _Plan(ROUTER_FANOUT_VERSION, (*plan.derived, entry), head + plan.note.removeprefix(f"routing.maze {plan.version}: ") + "; " + words,
                         escape=True)
    clearance = {index[name]: r.clearance_mm for name, r in eff.items()}

    def clear_of(idx: int) -> float:
        return clearance.get(idx, p.clearance_mm)

    def track_of(idx: int) -> tuple[float, float]:
        r = eff.get(ir.nets[idx].name) if idx >= 0 else None
        return (r.width_mm, r.clearance_mm) if r is not None else (p.track_width_mm, p.clearance_mm)

    nets = [net for net in ir.nets if net.name not in plane_width] if plane_width else ir.nets
    ordered = sorted(nets, key=lambda net: (len(board.terminals[net.name]), net.name))
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
    if plan is None:
        if not eff:
            result = _result_0_2(board, p, raised, order, routes, unrouted, skipped, iterations, legal, history, dropped, recovered)
        else:
            result = _result_rules(
                ir, board, neg, p, raised, eff, rule_raised, rule_nets, pairs, order, routes, unrouted, skipped, iterations, legal, history, dropped,
                recovered, clear_of,
            )
    else:
        if not eff:
            result = _result_0_2(board, p, raised, order, routes, unrouted, skipped, iterations, legal, history, dropped, recovered, plan)
        else:
            result = _result_rules(
                ir, board, neg, p, raised, eff, rule_raised, rule_nets, pairs, order, routes, unrouted, skipped, iterations, legal, history, dropped,
                recovered, clear_of, plan,
            )
        if planes_or_pads:
            _add_planes(ir, board, p, result, plan, plane_width, plane_rules, plane_layers, iterations, legal)
        if board.custom_pads:
            result.stats["custom_pads"] = list(board.custom_pad_stats)
    if board.escapes or board.escape_refused:  # routing.maze 0.6: every stub and every pad none reached
        result.stats["escapes"] = list(board.escapes)
        result.stats["escape_refused"] = dict(board.escape_refused)
    if board.rules07:  # routing.maze 0.7: the fan-out rooms kept and the plane pads whose via they kept out
        result.stats["fanout_room"] = {"rooms": {ref: [_q(v) for v in box] for ref, box in board.fanout_rooms.items()}, "kept_out": list(board.fanout_kept)}
    return result


def _add_planes(
    ir: CircuitIR, board: _Board, p: RoutingParams, result: Routing, plan: _Plan, widths: Mapping[str, float], rules: Mapping[str, NetRule],
    plane_nets: Mapping[str, Sequence[str]], iterations: int, legal: bool,
) -> None:
    """Append each plane net's pad stubs and vias (IR net order) and the 0.4 stats: ``keepouts`` and ``plane_nets`` (module docstring)."""
    lengths: dict[str, float] = result.stats.setdefault("net_length_mm", {})
    rows: dict[str, dict[str, Any]] = {}
    for net in ir.nets:
        name = net.name
        if name not in widths:
            continue
        row: dict[str, Any] = {"planes": list(plane_nets.get(name, ())), "width_mm": widths[name]}
        if name in rules:
            row["rule"] = rules[name].derived_from_entry()
            row["rule_note"] = "the rule's width is the pad stubs' width; nothing else of it applies (no track joins a plane net's pads)"
        rows[name] = row
        problem = board.plane_problems.get(name)
        if problem is not None:
            row["reason"] = problem
            result.unrouted[name] = f"plane net: {problem}"
            continue
        links = board.plane_links.get(name, [])
        planes = "+".join(plane_nets.get(name, ())) or "the plane"
        joined = list(board.plane_joined.get(name, []))
        story = (f"plane net: {len(links)} SMD pad(s) each joined by one stub to its own via to the {planes} plane, "
                 f"{len(board.plane_tht.get(name, []))} through-hole pad(s) reach it by their barrel"
                 + (f", {len(joined)} SMD pad(s) by their footprint's own same-numbered through-hole pads" if joined else "")
                 + "; no track joins two pads")
        prov = _provenance(net, p, "plane", iterations, legal, story=story, plan=plan)
        copper = _NetCopper(board, net, prov, widths[name])
        for t, layer, k in links:
            copper.add_stub(t, layer)
            if t.escape is not None:  # routing.maze 0.6: the escape stub ends on its via
                copper.add_path([layer * board.n + k, (1 - layer) * board.n + k])
                continue
            copper.add_path([layer * board.n + t.cell, layer * board.n + k, (1 - layer) * board.n + k])
        result.tracks.extend(copper.tracks)
        result.vias.extend(copper.vias)
        length = _q(copper.length_mm())
        lengths[name] = length
        row.update(vias=len(copper.vias), pads=[t.label for t, _, _ in links], through_hole=list(board.plane_tht.get(name, [])), length_mm=length)
        if joined:
            row["joined_by_footprint_vias"] = joined
    ids = [k["id"] for k in board.keepout_stats]
    if ids:  # say that the keep-outs were obstacles wherever a net found no route
        for name, why in list(result.unrouted.items()):
            if name not in widths:
                result.unrouted[name] = f"{why} (keep-out(s) {', '.join(ids)} are obstacles on this board)"
    s = result.stats
    s["routed_nets"] = len(lengths)
    s["unrouted_nets"] = len(result.unrouted)
    s["total_length_mm"] = _q(sum(lengths.values()))
    s["track_count"] = len(result.tracks)
    s["via_count"] = len(result.vias)
    s["keepouts"] = list(board.keepout_stats)
    s["plane_nets"] = rows


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
    iterations: int, legal: bool, history: list, dropped: list[str], recovered: list[str], plan: _Plan | None = None,
) -> Routing:
    """The result of a board without rules: 0.2's emission and stats (plus ``inner_layers`` when the board has them); ``plan`` stamps 0.4."""
    result = Routing(params=p, unrouted={net.name: unrouted[net.name] for net in order if net.name in unrouted})
    if plan is not None:
        result.version = plan.version
    lengths: dict[str, float] = {}
    for net in order:
        route = routes.get(net.name)
        if route is None:
            continue
        copper = _emit(board, net, route, _provenance(net, p, route.how, iterations, legal, plan=plan), None)
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
    plan: _Plan | None = None,
) -> Routing:
    """Match groups, emission, pairs, the audits and the stats of a board routed with rules (stamped :data:`ROUTER_RULES_VERSION`, 0.4 with ``plan``)."""
    version = ROUTER_RULES_VERSION if plan is None else plan.version
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
            rule_note=None if rn is None else _rule_note(rn, meanders.get(net.name), p.grid_mm), plan=plan,
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
        problem, tracks = _emit_pair(board, p, pr, route, world, st, iterations, legal, net_of, eff, plan)
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
    net_of: Mapping[str, Net], eff: Mapping[str, NetRule], plan: _Plan | None = None,
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
            extra_refs=refs, story=story, rule_note=note, plan=plan,
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
