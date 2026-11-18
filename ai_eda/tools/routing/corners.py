"""Corner style (routing.corners 0.1 / 0.2): the router's right-angle track corners mitred into 45-degree corners, or rounded into
tangent arcs, after routing (pure).

**The rule.** The project owner, 2026-11-13: "배선 규칙에 직각 배선을 최대한 피하고 45도 배선 혹은 라운드 패턴 배선을 위주로
사용하게 해줘" - avoid right-angle track corners as far as possible and route mainly with 45-degree or rounded corners
(:data:`CORNER_RULE`). This is the owner's corner-style rule, a geometry convention - **not an electrical requirement**, and
nothing here claims a signal-integrity or manufacturing benefit (:data:`CORNER_BASIS`): "a 90-degree corner causes reflections or
EMI at normal edge rates" is a myth - a corner adds about 1.7 fF per mil of track width, no noticeable effect on a 5 mil trace below
about 70 GHz (E. Bogatin, Signal Integrity Journal, 13 Apr 2021; K. Knack, Altium, "Slaying the PCB 90-degree Right-Angle Bend
Dragon", 2019, updated 2026) - and so is "right or acute corners create acid traps" (Knack, same article); both opened for the
router's research (``research_synthesis.json``), not re-opened here. So the pass never blocks routing and never makes a routable
board fail: it runs on copper that is already routed and legal, changes no endpoint at a pad, via or junction, and a corner it
cannot mitre legally stays as the router drew it, counted with the reason.

**When it runs.** :func:`ai_eda.tools.routing.maze.route_board` takes ``corner_style`` (:data:`CORNER_STYLES`): ``"orthogonal"``,
its default, returns the router's copper exactly as before (nothing here changes it - every board's copper, provenance and stats byte
for byte); ``"miter45"`` runs :func:`mitre_corners` last, after the negotiation, the end game, the drop and recovery, the plane-web
repair, the meanders and the pair emission have settled and every net's copper has been emitted, on the :class:`Routing` result
itself (routing.corners 0.1: everything below up to the records); ``"arc"`` runs it at the same point, straightens the router's
sidesteps and rounds the corners (routing.corners 0.2: the arc and the straightening, below). The PCB agent routes with
:data:`DEFAULT_CORNER_STYLE`, ``"arc"``, by default (the owner's rule applies to every board the pipeline routes from 2026-11-13 on,
with 45-degree corners until 2026-11-18 and rounded ones since - kicad-cli 10.0.6 DRC with schematic parity and refilled zones gave 0
errors, 0 warnings and 0 unconnected items on the divider, the ATmega128 2-layer and the STM32F407 6-layer boards routed with it);
``--answer pcb.corners=miter45`` gives the 45-degree corners of routing.corners 0.1 and ``--answer pcb.corners=orthogonal`` the
router's grid corners (:mod:`ai_eda.agents.pcb`). The pass runs
once, on the router's own copper, which is straight: a result that already holds an arc track is refused
(:class:`~ai_eda.errors.CompileError`), whatever the style.

**A corner** is a point where exactly two tracks of one net end on one layer and the angle between them is 90 degrees or sharper
(the grid router's bends are all right angles; a stub to a pad centre may meet a track at any angle). A point where three or more
tracks end is a junction, not a corner; an obtuse bend is left alone. Each corner found is mitred, left or skipped (under the arc
style it may also be rounded: the arc, below):

* **skipped** - its geometry is pinned by another mechanism: a differential pair's track (the pair emission mitres its own
  corners and fixes its coupling and skew), a net of a length-matched group (a chamfer shortens it and changes the group's skew), a
  segment of an escape stub (its points are recorded in its ``escape:`` entry, which ``si.impedance`` reads back), two tracks of
  different widths (a neck-down or stub transition), a corner at a pad centre of its net or on another track of its net (a
  junction: moving it would cut that connection), at or inside a via of its net or inside the box of a pad of its net (their
  copper covers the corner - the router's terminal cells lie in their pads - and the SI paths count copper in a pad box or via
  disc as the pad or via), or where the two tracks run back over each other (a 0-degree turn: no chamfer lies between them);
* **mitred** - a 45-degree chamfer cuts the corner: the point ``d`` along each of the two tracks becomes the end of a new straight
  segment between them (on an axis-aligned right angle a 45-degree segment and two 135-degree corners; on a corner of ``a`` < 90
  degrees two corners of ``(180 + a) / 2`` degrees each, obtuse but not 135);
* **left** - no chamfer from the largest down to the smallest keeps every rule below (or the tracks leave no room for the smallest).

**The chamfer** (stated choices). Each of the two tracks offers its *run* - from the corner to the first point on it the pass
keeps: its other end, or a point on it where another track of the net ends, a via or a pad centre of the net lies. The chamfer may
take half of each run, so the kept point still meets the track in its own direction and no angle there changes (a track never
enters a pad, via or junction at a new angle); when the run ends at another corner the pass mitres, the two corners share it -
half each when they turn opposite ways (a Z: the two chamfers line up into one 45-degree run), a third each when they turn the same
way (a U: two half chamfers would meet at a right angle, so a third of the track stays between them). It starts at the smaller of
the two and is halved (FreeRouting's 45-degree pull-tight bisects the chamfer the same way, :data:`CORNER_IDEAS`) until the new
segment passes the audit below; a halving that would fall below the smallest chamfer tries the smallest itself, and that is the
last try. The smallest chamfer is :data:`MIN_CHAMFER_WIDTHS` times the track's width: a chamfer of
``d`` on a right angle of width ``w`` passes ``(w - d) / sqrt(2)`` from the inside corner of the two tracks' copper, so from ``d =
(1 - 1/sqrt(2)) w`` (0.293 widths) on its copper covers that corner and the copper outline keeps no right angle (the round track
ends already round the outside); 0.3 is that bound rounded up. Every point is rounded to KiCad's 1 nm
(:func:`~ai_eda.tools.kicad.geometry._q`).

**The audit** is the exact one the router's meander audit uses (:class:`~ai_eda.tools.routing.maze._World` ``seg_problem``:
segment / box / disc distances, not the grid): the new segment keeps the larger of the two nets' clearances from every other net's
track, via, pad box, escape stub and plane via or stub (a rule net's clearance, the board's at a neck-down width), and from a pad
without a net (a mounting or NPTH hole, an unnumbered pad) the board clearance plus half a grid step - the margin the router's grid
copper keeps from a pad box (its owner map's ``clearance + width/2 + grid/2`` from the track centre), because kicad-cli judges a
hole by its hole clearance, not the copper clearance; it stays outside every track keep-out that does not allow its net (its
copper keeps half its width from the area) and ``edge_clearance + width/2`` from the outline. The two shortened tracks are parts of
the old ones, so they need no check. The other nets' copper is read as it stands when the corner's turn comes (corners in track
order, each net's mitres in the audit's copper as soon as they are made). No zone lies on a routing layer of a board the router
routes (plane zones are on plane layers; the PCB agent routes no board with other zones), so a chamfer meets none.

**Connectivity** is unchanged: every track end at a pad, via or junction stays where it was, in its direction; the new segment
shares its two ends exactly with the two shortened tracks (on a Z the shared track is used up and the two chamfers meet where it
was cut in half). The audit ignores the net's own copper (no clearance applies within a net), so a chamfer is also refused where
it would join its own net where the corner did not - a cut point on a pad box or via disc of the net that its track's far end is
not on, a via centre of the net on the new segment, the new segment's centreline meeting another track of the net anywhere but at
a shared end (a track end on it, or a crossing), or the new segment's copper (half its width around it) reaching a pad box, a via
disc or another track's copper of the net that the copper of the two tracks, as they stood, did not reach: the net's copper graph
(:mod:`ai_eda.tools.si.paths` - nodes at track ends and on track interiors, pads contracting the nodes in their boxes, vias the
nodes in their discs) keeps its shape, and the copper itself makes no new contact that model would not see, so a tree stays a
tree.

**Records.** Every track the pass creates or shortens keeps its net's provenance (``routing.maze``, its version) with one
:data:`CORNER_ENTRY_PREFIX` entry per corner in ``derived_from`` (``corner:routing.corners 0.1:miter45:at=x,y:chamfer=d``) and one
sentence in the note; ``stats["corners"]`` gives the rule, the basis, the counts (found, mitred, left, skipped, each reason) and one
row per corner; ``net_length_mm``, ``total_length_mm``, ``track_count`` and the rule / plane-net length rows are measured again on
the mitred copper (a chamfer only shortens a net, so a length budget still holds).

**The arc** (routing.corners 0.2, style ``"arc"``; stated choices). The owner also named rounded corners, and a
:class:`~ai_eda.ir.Track` with a ``mid`` point is a circular arc (KiCad's own track arc; :mod:`ai_eda.ir.arc` holds its geometry).
Under this style a corner is **rounded**: the arc tangent to its two tracks replaces it. The corners, the skips and the kept points
are those above; the cut points at ``d`` along the two tracks are the ones a chamfer of ``d`` has, and they are the arc's two ends;
for a corner of ``a`` degrees its radius is ``d tan(a / 2)`` (``d`` itself at a right angle) and its ``mid`` lies on the bisector
(:func:`~ai_eda.ir.arc.fillet`). Like the chamfer it is a geometry convention: no signal-integrity or manufacturing benefit is
claimed for a rounded corner either.

* *Radius* (:data:`ARC_RADIUS_WIDTHS`): 4, 2 and 1 track widths, largest first. 4 widths is the cap - the rounding stays local, and
  neighbouring corners of a bus reach the same radius. 1 width is the smallest - from a centreline radius above half a width on, the
  inside edge of the copper is an arc and no longer a corner, and 1 width leaves that edge a radius of half a width.
* *Room*: an arc may take half of each track's run to the next kept point, also where the next corner turns the same way - two
  arcs that meet on a used-up track are both tangent to it, so no corner appears where they meet (a U of two quarter arcs is a half
  circle, a Z an S-curve; the chamfer keeps its third in a U). A step asks for the tangent length that gives its radius; a corner
  with less room tries the radius its room gives, never one below the smallest: a corner whose room is below that gets no arc.
* *Rounds*: one round per radius step, largest first. In a round every corner not yet rounded tries its size, in corner order, and
  the round is repeated while its last sweep rounded at least one corner - so the result does not depend on which of two nested
  corners comes first: the inner corner of a bus passes first, the outer one then passes against the rounded inner one, and both get
  the same radius. A corner is not tried again at a size while no arc was made since its last try (the result would be the same).
* *The audit of an arc* is its own, because an object may lie inside the triangle the corner and the two cut points span, so a
  legal chamfer does not imply a legal arc. It is made on the arc's chain of chords (:func:`~ai_eda.ir.arc.arc_points`; every point
  of the arc lies within :data:`~ai_eda.ir.arc.ARC_SAGITTA_MM`, 1 um, of the chain and every point of the chain within that of the
  arc), **the bound on the refusing side each time**: every chord passes the exact audit above with its width raised by twice the
  bound; every chord keeps half the width plus the bound from every track keep-out that does not allow the net; every chain point
  keeps ``edge_clearance + width/2`` plus the bound from the outline (points, as the chamfer's two ends are judged: exact for the
  convex outline the router's board has, where a chord is nearest the edge at one of its ends). So an arc is only made where every
  reader that measures it on the same chain with the same bound finds it legal; where the true distance lies within the bound of a
  limit the arc is not made at that size - in particular where a track runs at exactly the limit beside another net's copper, a
  keep-out or the edge, the chords at the arc's end are within the bound of that limit, and the corner gets the chamfer (whose
  audit is exact) instead. An arc that is made enters the audit's copper as its chords, each with half the width plus the bound,
  for everything tried after it.
* *Its own net*: the guard of the chamfer, on the arc - no new contact. The cut-point rule is the same; a via centre of the net on
  the arc is judged by the exact distance (:func:`~ai_eda.ir.arc.point_arc_distance`), as is the arc's copper reaching a via disc;
  the arc meeting another track or arc of the net is judged on the two chains (nearer than their bounds anywhere but at one shared
  end - two pieces meet at an end only when neither far end lies on the other, exactly, they leave the shared point in directions at
  least a right angle apart, and beyond their first chords the chains stay further apart than their bounds); its copper reaching a
  pad box or another track's copper is judged on the chords with the half-width raised by the bound, against what the two tracks'
  copper did not surely reach. New in 0.2, for an arc and for the chamfers made under this style - no contact lost: an item of the
  net on the layer (a pad box, a via disc, another track or arc) whose copper touched one of the two removed track ends must still
  touch the new piece or what is left of the two tracks (an arc judged with its half-width lowered by the bound, the sure side);
  otherwise the piece is refused, because the copper the cut removes may have been that item's only contact with the net.
* *After the rounds* every corner still square tries the 45-degree chamfer as above (largest first, down to 0.3 widths), with two
  additions under this style: the chamfer may not lose a contact either, and the end of an arc is one more point the pass keeps - a
  chamfer takes at most half of what is left of a track up to an arc's end, so the arc still meets that track in its own direction
  (two chamfers may still use a track up between them, as on a Z). So every arc is tangent to the copper it joins at both ends: the
  track it was cut from, or the arc of the next corner. A corner neither an arc nor a chamfer fits is left as routed, with both
  reasons.
* *Not rounded in 0.2*: obtuse corners stay; a differential pair stays as the pair emission mitres it; escape stubs stay (their
  recorded points are straight); the nets of a length-matched group stay as routed - rounding shortens a net by ``(2 - pi/2)``,
  0.43, radii per right angle, the group's skew budget is judged on delay by the SI check, and rounding them needs meanders that
  plan for the loss (the follow-up).
* *Records*: the entries are ``corner:routing.corners 0.2:arc:at=x,y:tangent=d:radius=r`` on the arc and the two shortened tracks
  of a rounded corner, and ``corner:routing.corners 0.2:miter45:at=x,y:chamfer=d`` for a chamfer made under this style; the note
  names the arc; ``stats["corners"]`` has ``version`` 0.2, ``style`` ``"arc"``, every key above, ``arcs`` (how many corners were
  rounded) and ``arc`` (this rule in words), a rounded corner's row has ``outcome`` ``"arc"`` with ``radius_mm`` and
  ``tangent_mm``, and the row of a corner that got a chamfer or stayed says why it has no arc (``no_arc_reason``, ``no_arc_why``).
  The lengths are measured again with the arcs' own lengths (:func:`~ai_eda.ir.track_length`; an arc is shorter than the corner it
  replaces, so a length budget still holds).

**The straightening** (routing.corners 0.2, style ``"arc"`` only; stated choices). The project owner, 2026-11-18, on a board routed
with arcs: "중간에 의도적인 패턴이 아닌 0.5그리드 수준으로 꾸불거리는 패턴은 왜 들어가는거야? 좀 더 일체감이 있거나 타이밍을 맞추기 위해
돌아가게 하는 배선이면 더 의도적인 배선이라는 느낌이 있었으면 좋겠어" - why are there wiggles of about half a grid step that are no
intended pattern; the routing should look of one piece, and a detour should look deliberate. Such a wiggle is a **sidestep**: a
short track between two right-angle corners, which shifts a run by a grid step or two (the corners turn opposite ways: a Z) or
closes a small bump (they turn the same way: a U). It is no timing detour - the copper of a length-matched group is not touched
here - but what the router's negotiated grid search can leave: a bend costs little there (``bend_cost``: by default 0.6 of a step
on a free cell), while a cell costs more as long as another net's clearance halo covers it and as long as the negotiation's
history cost stays on it - so a net steps aside where the cells straight on were dear when it was last routed, and keeps the step
after the other net has moved. A sidestep shorter than two track widths cannot hold two arcs of even the smallest radius (each
takes half of it), so its corners get small chamfers or stay square: a kink. So under the arc style, before the corners are found,
the pass removes the sidesteps wherever the copper may legally run without them. Like the rounding it is a geometry convention -
nothing electrical is claimed - and it never makes a routed board fail: a sidestep that cannot be removed legally stays as routed,
counted with the reason.

* *A plain corner* is a corner the pass would not skip, between a track along the X axis and a track along the Y axis: exactly two
  track ends of the net on the layer, the same width, neither an escape stub, not at or inside a via of the net, not at a pad
  centre, not inside a pad box of the net, not on another track of the net. Only rectilinear copper is straightened, on its stored
  coordinates, exactly: every coordinate of the straightened copper is one the router's copper had. The nets the pass skips
  entirely - a differential pair's, a length-matched group's - stay as routed (their copper is pinned by the pair emission and by
  the meanders: a meander is a deliberate bump). One more track is pinned for this step only: a track at its net's neck-down width
  is no part of a plain corner, because ``si.impedance`` takes a narrower track for a neck-down only within the rule's radius of
  a pad, and a slide - unlike a chamfer or an arc, which stay inside the corner they cut - may carry copper out of that zone.
* *A sidestep* (:data:`SIDESTEP_WIDTHS`) is a track between two plain corners shorter than 8 track widths: the room two arcs of the
  cap radius need (a tangent length of 4 widths each), so a track the rounding could not round at its standard radius.
* *The move*. With the points ``P0``-``P1`` (the neighbour ``n1``), ``P1``-``P2`` (the sidestep ``m``) and ``P2``-``P3`` (the
  neighbour ``n2``), ``m`` slides along ``n2`` to its far end: ``m`` becomes ``P1'``-``P3`` with ``P1' = P1 + (P3 - P2)``, ``n2`` is
  used up, and ``n1`` becomes ``P0``-``P1'`` - longer where ``n1`` and ``n2`` run the same way (a Z), shorter where ``n2`` runs back
  (a U, which slides only as far as the shorter of its two neighbours). ``P3`` must be a plain corner too, so the track ``n3`` that
  leaves it lies on the line of the slid ``m``, and the two are joined: into one track where ``n3`` continues ``m`` (two corners
  fewer, the same length); where ``n3`` runs back over ``m`` only their difference is left - nothing when they are equally long, and
  then ``n1`` and the track after ``n3`` meet at ``P1'`` and are joined in the same way (a bump has become a straight line). The
  mirrored move slides ``m`` along ``n1`` to ``P0``. Both are tried; of two legal moves the one that leaves fewer corners is made,
  then the one that leaves less copper, then the one along the neighbour that comes first in the track order. A sidestep neither
  move may slide stays: with the reason of the first of the two moves that got as far as the audit below, or ``kept_point`` where
  kept points stop both; its row says in words what stopped each.
* *Kept points*. A move never slides into a point the pass keeps (a pad, a via, a junction, a track end with nothing else: no plain
  corner is one of these) and never across one: a via centre, a pad centre or another track's end of the net that lies on the
  centreline of the tracks a move changes must still lie on it afterwards; the point where the slid copper turns must be fit for
  a plain corner itself; and at both ends of its run of plain corners the copper leaves the kept point in the direction it had.
* *The audit of a move*, in this order, the first problem kept as the reason. The new copper is what the changed tracks cover and
  did not cover before - straight pieces, so there are no chords and no bound. (1) Other nets: every new piece passes the exact
  audit of a chamfer, with the track's width and clearance. (2) Every new piece stays outside the track keep-outs that do not allow
  the net and keeps the edge clearance from the outline, as a chamfer does. (3) The net's own copper: no via centre, pad centre or
  track end of the net comes to lie on the centreline of the changed tracks, and every pad box, via disc and other track of the
  net on the layer touches the copper of the changed tracks afterwards exactly when it did before (copper touches at a centreline
  distance of at most the half-widths) - no contact made, none lost. (4) Length: a move never lengthens the net - a Z joined to a
  track that continues the sidestep keeps the length, every other case shortens it - so a length budget still holds, and none is
  checked again.
* *Order*: sweep after sweep over the sidesteps as they stand, shortest first, then in track order. A move changes the audit's
  copper at once, so every later move and the rounding see it. Every move removes at least two corners, so the sweeps end by
  themselves; a sweep that moves nothing ends the step, and :data:`SIDESTEP_SWEEPS` bounds them (a sidestep that only this bound
  kept from moving is counted with the reason ``sweep_cap``).
* *What the rounding sees*: the straightened copper. A track a move used up or merged away is gone before the corners are found, a
  merged track keeps the place of the earlier of the two in the track order, and ``found`` counts the corners found after the step.
* *Records*: every track a move changed carries one entry per move, ``corner:routing.corners 0.2:straighten:at=x,y:step=s`` (the
  middle of the sidestep where it lay, and its length; a track that is merged into another or used up passes its entries on - also
  one the rounding then uses up, whose entries go to the arc or chamfer that took it, so every move stays on the copper) and one
  sentence in its note. ``stats["corners"]`` gains ``straighten`` (this rule in words), ``sidesteps`` (how many the step dealt
  with: each one it slid - a track slid twice, further each time, counts twice - and each one still in the copper at its end),
  ``straightened`` (the moves made), ``sidesteps_left`` with ``sidestep_reasons`` (``kept_point``, ``clearance``, ``keepout``,
  ``edge``, ``own_copper``, ``sweep_cap``) and ``sidestep_rows``, one row per sidestep (where it lay, its length, slid - where to,
  the corners and the copper it removed - or left, and why): ``sidesteps == straightened + sidesteps_left`` (a second sidestep that
  disappears with a move, as the far side of a bump does, has no row of its own). The lengths are measured again as for the corners.

Ideas used, no code (:data:`CORNER_IDEAS`): FreeRouting's 45-degree pull-tight (``TraceTightener45``: each corner of 90 degrees or
sharper replaced by two 45-degree corners, the chamfer bisected under a clearance re-check) and KiCad PNS's 45-degree / arc corner
modes - both GPL, both read for the router's research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ai_eda.errors import CompileError
from ai_eda.ir import Track, track_length
from ai_eda.ir.arc import ARC_SAGITTA_MM, arc_points, arc_problem, fillet, point_arc_distance
from ai_eda.tools.keepout import segment_area_distance
from ai_eda.tools.kicad.geometry import _q
from ai_eda.tools.routing.maze import Routing, _Board, _PadGeom, _pt_seg, _seg_box, _seg_seg, _World, escape_entry

__all__ = [
    "ARC",
    "ARC_RADIUS_WIDTHS",
    "CORNERS_ARC_VERSION",
    "CORNERS_ID",
    "CORNERS_VERSION",
    "CORNER_BASIS",
    "CORNER_ENTRY_PREFIX",
    "CORNER_IDEAS",
    "CORNER_RULE",
    "CORNER_STYLES",
    "DEFAULT_CORNER_STYLE",
    "MIN_CHAMFER_WIDTHS",
    "MITER45",
    "ORTHOGONAL",
    "SIDESTEP_SWEEPS",
    "SIDESTEP_WIDTHS",
    "STRAIGHTEN",
    "check_style",
    "mitre_corners",
]

#: the pass's id and version, in every corner entry and in ``stats["corners"]`` (the version of the 45-degree style)
CORNERS_ID = "routing.corners"
CORNERS_VERSION = "0.1"
#: the version of the arc style (module docstring: the arc): its entries, its note and its ``stats["corners"]``
CORNERS_ARC_VERSION = "0.2"
#: the corner styles :func:`~ai_eda.tools.routing.maze.route_board` takes: tangent arcs (routing.corners 0.2), 45-degree mitres
#: (routing.corners 0.1), or the router's grid corners as they are
ARC = "arc"
MITER45 = "miter45"
ORTHOGONAL = "orthogonal"
CORNER_STYLES: tuple[str, ...] = (ARC, MITER45, ORTHOGONAL)
#: the PCB agent's style unless ``pcb.corners`` answers otherwise (the owner's rule; ``"arc"`` since routing.corners 0.2 - module
#: docstring: when it runs); ``route_board`` itself defaults to ``"orthogonal"``
DEFAULT_CORNER_STYLE = ARC
#: the owner's rule, as every record quotes it
CORNER_RULE = (
    "the project owner's corner-style rule (2026-11-13): avoid right-angle track corners as far as possible and route mainly with "
    "45-degree or rounded corners"
)
#: why it is a convention and not an electrical rule (module docstring)
CORNER_BASIS = (
    "a geometry convention, not an electrical requirement: a right-angle corner adds about 1.7 fF per mil of track width, no noticeable "
    "effect on a 5 mil trace below about 70 GHz (E. Bogatin, Signal Integrity Journal, 13 Apr 2021; K. Knack, Altium, 'Slaying the PCB "
    "90-degree Right-Angle Bend Dragon', 2019, updated 2026), and right or acute corners create no acid traps (Knack, same article); no "
    "signal-integrity or manufacturing benefit is claimed"
)
#: where the chamfer comes from (ideas only; both projects are GPL, no code was taken)
CORNER_IDEAS = (
    "FreeRouting's 45-degree pull-tight (TraceTightener45: each corner of 90 degrees or sharper replaced by two 45-degree corners, the "
    "chamfer bisected under a clearance re-check) and KiCad PNS's 45-degree corner mode - GPL, ideas only, no code"
)
#: the smallest chamfer tried, in track widths: (1 - 1/sqrt(2)) = 0.293 rounded up - from there the new segment's copper covers the
#: inside corner of the two tracks (module docstring: the chamfer)
MIN_CHAMFER_WIDTHS = 0.3
#: routing.corners 0.2: the radii an arc is tried at, in track widths, largest first - stated choices (module docstring: the arc).
#: 4 is the cap: the rounding stays local, and neighbouring corners of a bus reach the same radius. 1 is the smallest: from a
#: centreline radius above half a width on, the inside edge of the copper is an arc and no longer a corner, and 1 width leaves that
#: edge a radius of half a width. 2 is the step between them.
ARC_RADIUS_WIDTHS: tuple[float, ...] = (4.0, 2.0, 1.0)
#: routing.corners 0.2 (module docstring: the straightening): a track between two plain corners is a sidestep when it is shorter than
#: this many track widths - a stated choice: the room two arcs of the cap radius need (a tangent length of ARC_RADIUS_WIDTHS[0] widths
#: each at a right angle), so a shorter track could not be rounded at the standard radius at both of its ends
SIDESTEP_WIDTHS = 2.0 * ARC_RADIUS_WIDTHS[0]
#: routing.corners 0.2: the most sweeps the straightening makes over the sidesteps - a stated bound on the running time, not a rule
#: of the geometry: every move removes at least two corners, so the sweeps end by themselves, and a sweep that moves nothing ends
#: the step (measured on the ATmega128 2-layer example board, 0.2 mm grid: 29 moves in the first sweep, 1 in the second, none in
#: the third)
SIDESTEP_SWEEPS = 32
#: routing.corners 0.2: the word of a straightened sidestep in its corner entry (``corner:routing.corners 0.2:straighten:at=x,y:step=s``)
STRAIGHTEN = "straighten"
#: the ``derived_from`` prefix of a mitred corner on a track it created or shortened
CORNER_ENTRY_PREFIX = "corner:"
#: routing.corners 0.2: how a straightening entry begins (module docstring: the straightening, records)
_STRAIGHTEN_ENTRY = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {CORNERS_ARC_VERSION}:{STRAIGHTEN}:"

_TOL = 1e-6
#: a corner of 90 degrees or sharper: the cosine of its angle is at least minus this
_COS_TOL = 1e-9

#: why a corner is skipped (module docstring), by key
_SKIP_WORDS = {
    "pair": "a differential pair's track: the pair emission mitres its corners and fixes its coupling and skew",
    "match_group": "a net of length-matched group {group}: a chamfer shortens it and changes the group's skew",
    "escape": "a segment of an escape stub: its points are recorded in its escape entry, which si.impedance reads back",
    "width": "the two tracks differ in width ({w1:g} / {w2:g} mm: a neck-down or stub transition)",
    "via": "at or inside a via of its net: the via's copper covers the corner",
    "pad_centre": "at a pad centre of its net",
    "junction": "on another track of its net (a junction)",
    "in_pad": "inside the box of pad {pad} of its net: the pad's copper covers the corner",
    "overlap": "the two tracks run back over each other (a 0-degree turn): no chamfer lies between them",
}
#: routing.corners 0.2: the same reasons under the arc style, in its words where they differ
_ARC_SKIP_WORDS = {
    **_SKIP_WORDS,
    "match_group": "a net of length-matched group {group}: a rounded corner shortens it and changes the group's skew",
    "overlap": "the two tracks run back over each other (a 0-degree turn): no arc or chamfer lies between them",
}

Point = tuple[float, float]


def check_style(style: str) -> None:
    """Refuse a corner style the pass does not know (:class:`CompileError` naming the styles)."""
    if style not in CORNER_STYLES:
        raise CompileError(f"corner_style {style!r} is not one of {list(CORNER_STYLES)} ({CORNERS_ID}: {ARC!r} is version "
                           f"{CORNERS_ARC_VERSION}, {MITER45!r} version {CORNERS_VERSION})")


@dataclass(eq=False)
class _Seg:
    """One track as the pass edits it: its original :class:`Track` (net, layer, width, provenance), its ends now, its corner entries.
    routing.corners 0.2: an arc the pass made has ``mid``, its chain of chords (:func:`~ai_eda.ir.arc.arc_points`) and the chain's
    box; it is never a corner's track, so it is never shortened. A track the straightening changed (module docstring: the
    straightening) has ``track`` with the ends that step left it: the ends the rounding keeps."""

    order: int
    track: Track
    a: Point
    b: Point
    escape: bool
    entries: list[str] = field(default_factory=list)
    alive: bool = True
    mid: Point | None = None
    chain: list[Point] | None = None
    box: tuple[float, float, float, float] | None = None


@dataclass(eq=False)
class _Corner:
    net: str
    layer: str
    li: int
    at: Point
    #: the two tracks and which of their ends (0: ``a``, 1: ``b``) lies at the corner, the earlier track (in emission order) first
    legs: tuple[tuple[_Seg, int], tuple[_Seg, int]]
    angle: float
    skip: str | None = None
    why: str = ""
    dmax: float = 0.0
    #: routing.corners 0.2: the largest tangent length an arc may take (half of each run), the tangent of half the corner's angle
    #: (the radius of the arc of tangent length ``d`` is ``d * tan_half``), and what each of the two tracks allows a chamfer (``dmax``
    #: is the smaller)
    dmax_arc: float = 0.0
    tan_half: float = 0.0
    limits: tuple[float, ...] = ()


class _Copper(_World):
    """The router's exact audit (:class:`_World`) over tracks that may be taken out again (a shortened track is replaced)."""

    def __init__(self, board: _Board, clear_of) -> None:
        super().__init__(board, clear_of, {})
        self._held: dict[int, list[tuple[tuple, tuple[float, float, float, float]]]] = {}

    def put(self, seg: _Seg, net: int, c: float) -> None:
        """A straight track is one item. routing.corners 0.2: an arc is one item per chord of its chain, each with half the width
        plus the chain's bound, so its copper lies inside theirs (module docstring: the audit of an arc)."""
        r = seg.track.width_mm / 2.0
        layers = frozenset((self.board.layer_index(seg.track.layer),))
        if seg.mid is None:
            pieces, label = [(seg.a, seg.b)], f"track of {seg.track.net}"
        else:
            r += ARC_SAGITTA_MM
            pieces, label = list(zip(seg.chain, seg.chain[1:])), f"arc of {seg.track.net} (a chord of it, with its {ARC_SAGITTA_MM:g} mm bound)"
        held = []
        for a, b in pieces:
            (ax, ay), (bx, by) = a, b
            item = ("seg", net, layers, (a, b), r, c, label)
            bbox = (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)
            self._add(item, bbox)
            held.append((item, bbox))
        self._held[id(seg)] = held

    def take(self, seg: _Seg) -> None:
        s = self._CELL
        for item, bbox in self._held.pop(id(seg)):
            for bi in range(math.floor(bbox[0] / s), math.floor(bbox[2] / s) + 1):
                for bj in range(math.floor(bbox[1] / s), math.floor(bbox[3] / s) + 1):
                    bucket = self.buckets.get((bi, bj))
                    if bucket is not None:
                        bucket[:] = [x for x in bucket if x is not item]


def _far(seg: _Seg, end: int) -> Point:
    return seg.b if end == 0 else seg.a


def _orient(p: Point, q: Point, r: Point) -> float:
    """Which side of the line ``p``-``q`` the point ``r`` lies on (the sign; 0 on the line)."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _near(p: Point, q: Point) -> bool:
    return math.hypot(p[0] - q[0], p[1] - q[1]) <= _TOL


def mitre_corners(result: Routing, board: _Board, style: str) -> None:
    """Mitre ``result``'s corners of 90 degrees or sharper into 45-degree chamfers (``"miter45"``) or - after the router's sidesteps
    are straightened - round them into tangent arcs (``"arc"``), in place (module docstring); ``"orthogonal"`` leaves it untouched.
    ``board`` is the router's board for that result (its pads, keep-outs, outline and parameters). The pass runs once, on the
    router's own copper: a track that is already an arc is refused under every style."""
    check_style(style)
    for k, t in enumerate(result.tracks):
        if t.mid is not None:
            raise CompileError(
                f"{CORNERS_ID}: track {k} of net {t.net!r} on {t.layer}, from ({t.start[0]:g}, {t.start[1]:g}) through ({t.mid[0]:g}, "
                f"{t.mid[1]:g}) to ({t.end[0]:g}, {t.end[1]:g}), is already an arc - the corner pass runs once, on the router's own copper")
    if style == ORTHOGONAL:
        return
    arcs = style == ARC  # routing.corners 0.2 (module docstring: the arc); otherwise routing.corners 0.1, as it was
    version = CORNERS_ARC_VERSION if arcs else CORNERS_VERSION
    skip_words = _ARC_SKIP_WORDS if arcs else _SKIP_WORDS
    p = board.p
    rules = result.rules
    idx_of = board.net_index
    # the router's effective rules (a field it leaves None is the board's value)
    clearance = {idx_of[name]: float(r.clearance_mm) for name, r in rules.items() if name in idx_of and r.clearance_mm is not None}
    hole = p.clearance_mm + p.grid_mm / 2.0

    def clear_of(idx: int) -> float:
        # a pad without a net (a hole, an unnumbered pad): the router's grid margin (module docstring: the audit)
        return hole if idx < 0 else clearance.get(idx, p.clearance_mm)

    def track_clear(t: Track) -> float:
        r = rules.get(t.net)
        if r is None or r.clearance_mm is None:
            return p.clearance_mm
        neck, width = r.neckdown_width_mm, r.width_mm if r.width_mm is not None else p.track_width_mm
        if neck is not None and abs(float(t.width_mm) - float(neck)) <= _TOL and abs(float(neck) - float(width)) > _TOL:
            return p.clearance_mm  # a neck-down track keeps the board's clearance, as the router's audit has it
        return float(r.clearance_mm)

    world = _Copper(board, clear_of)
    segs = [_Seg(k, t, tuple(t.start), tuple(t.end), escape_entry(t) is not None) for k, t in enumerate(result.tracks)]
    for s in segs:
        world.put(s, idx_of[s.track.net], track_clear(s.track))
    for v in result.vias:
        world.add_via(idx_of[v.net], v, clear_of(idx_of[v.net]))
    pads_of: dict[int, list[_PadGeom]] = {}
    for g in board.pads:
        if g.kind == "pad" and g.net >= 0:
            pads_of.setdefault(g.net, []).append(g)
    groups: dict[tuple[str, str], list[_Seg]] = {}
    for s in segs:
        groups.setdefault((s.track.net, s.track.layer), []).append(s)
    sidesteps: dict[str, Any] = {}
    if arcs:
        # routing.corners 0.2 (module docstring: the straightening): the router's sidesteps go first, on these same tracks, so the
        # corners found below, their room and their audit are those of the straightened copper; a track a move used up or merged
        # away is gone from here on
        sidesteps = _straighten(groups, world, board, result, pads_of, idx_of, track_clear)
        groups = {key: [s for s in group if s.alive] for key, group in groups.items()}
    #: per net and layer: its pad boxes, its vias (centre, radius) and its tracks as they stand (the own-copper guard below)
    own: dict[tuple[str, str], tuple[list[_PadGeom], list[tuple[Point, float]], list[_Seg]]] = {}

    corners: list[_Corner] = []
    for (net, layer), group in groups.items():
        li = board.layer_index(layer)
        idx = idx_of[net]
        discs = [((v.x_mm, v.y_mm), v.diameter_mm / 2.0) for v in result.vias if v.net == net and li in board.via_layers_of(v)]
        vias = [centre for centre, _r in discs]
        geoms = [g for g in pads_of.get(idx, ()) if li in g.layers]
        centres = [(_q(g.cx), _q(g.cy)) for g in geoms]
        own[(net, layer)] = (geoms, discs, list(group))
        ends: dict[Point, list[tuple[_Seg, int]]] = {}
        for s in group:
            ends.setdefault(s.a, []).append((s, 0))
            ends.setdefault(s.b, []).append((s, 1))
        here: list[_Corner] = []
        for pt, at in ends.items():
            if len(at) != 2 or at[0][0] is at[1][0]:
                continue
            (s1, e1), (s2, e2) = sorted(at, key=lambda leg: (leg[0].order, leg[1]))
            f1, f2 = _far(s1, e1), _far(s2, e2)
            v1, v2 = (f1[0] - pt[0], f1[1] - pt[1]), (f2[0] - pt[0], f2[1] - pt[1])
            l1, l2 = math.hypot(*v1), math.hypot(*v2)
            if l1 <= _TOL or l2 <= _TOL:
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            if cos < -_COS_TOL:
                continue  # obtuse or straight on: not a corner of 90 degrees or sharper
            c = _Corner(net, layer, li, pt, ((s1, e1), (s2, e2)), math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
            c.tan_half = math.sqrt(max(0.0, 1.0 - cos) / (1.0 + cos))  # tan(a / 2) from cos(a): exactly 1 at a right angle
            rule = rules.get(net)
            w1, w2 = float(s1.track.width_mm), float(s2.track.width_mm)
            inside = next((g for g in geoms if _in_box(pt, g.box)), None)
            if rule is not None and rule.pair_partner is not None:
                c.skip = "pair"
            elif rule is not None and rule.match_group is not None:
                c.skip = "match_group"
            elif s1.escape or s2.escape:
                c.skip = "escape"
            elif abs(w1 - w2) > _TOL:
                c.skip = "width"
            elif any(math.hypot(pt[0] - q[0], pt[1] - q[1]) <= r + _TOL for q, r in discs):
                c.skip = "via"
            elif any(_near(pt, q) for q in centres):
                c.skip = "pad_centre"
            elif any(_pt_seg(pt[0], pt[1], s.a, s.b) <= _TOL for s in group if s is not s1 and s is not s2):
                c.skip = "junction"
            elif inside is not None:
                c.skip = "in_pad"
            elif cos > 1.0 - _COS_TOL:
                c.skip = "overlap"
            if c.skip is not None:
                c.why = skip_words[c.skip].format(group=getattr(rule, "match_group", None), w1=w1, w2=w2,
                                                  pad=f"{inside.ref}.{inside.number}" if inside is not None else "")
            here.append(c)
        # the largest chamfer each corner may take (module docstring: the chamfer), and the arc's room (module docstring: the arc)
        by_point = {c.at: c for c in here if c.skip is None}
        pins = [*vias, *centres, *(q for s in group for q in (s.a, s.b))]
        for c in here:
            if c.skip is not None:
                continue
            limits = []
            room = []
            for k, (s, e) in enumerate(c.legs):
                far = _far(s, e)
                run = math.hypot(far[0] - c.at[0], far[1] - c.at[1])
                inner = min((math.hypot(q[0] - c.at[0], q[1] - c.at[1]) for q in pins
                             if not _near(q, s.a) and not _near(q, s.b) and _pt_seg(q[0], q[1], s.a, s.b) <= _TOL), default=run)
                room.append(min(inner, run) / 2.0)  # an arc: half of the run, whichever way the next corner turns
                other = by_point.get(far) if inner >= run - _TOL else None
                if other is None:  # the run ends at a point the pass keeps: half of it, so that point keeps its angle
                    limits.append(min(inner, run) / 2.0)
                    continue
                mine = _far(*c.legs[1 - k])
                theirs = next(_far(t, f) for t, f in other.legs if t is not s)
                same_side = _orient(c.at, far, mine) * _orient(c.at, far, theirs) > 0.0
                limits.append(run / 3.0 if same_side else run / 2.0)  # a U leaves a third between the chamfers; a Z lines them up
            c.dmax = min(limits)
            c.dmax_arc = min(room)
            c.limits = tuple(limits)
        corners.extend(here)

    inserted: dict[int, list[_Seg]] = {}
    outcome: dict[int, tuple[str, str | None, str, float | None]] = {}  # id(corner) -> (outcome, reason, words, chamfer)
    rounded: dict[int, tuple[float, float]] = {}  # routing.corners 0.2: id(corner) -> (tangent length, radius) of its arc
    arc_ends: set[tuple[int, int]] = set()  # routing.corners 0.2: (id(track), end) of every track end that is now an arc's end
    no_arc: dict[int, tuple[str, str]] = {}  # routing.corners 0.2: id(corner) -> (reason, words) of a corner without an arc

    def cut_points(c: _Corner, d: float) -> tuple[Point, Point] | None:
        """The points ``d`` along the corner's two tracks; ``None`` where a track would be used up to an end the pass keeps."""
        cut = []
        for s, e in c.legs:
            far = _far(s, e)
            full = math.hypot(far[0] - c.at[0], far[1] - c.at[1])
            if d >= full - _TOL:
                # a track is used up only up to a cut point this pass made (a Z's shared track), never up to an end it keeps; the
                # limits above never ask for that - this is the guard
                if _near(far, tuple(s.track.end) if e == 0 else tuple(s.track.start)):
                    return None
                cut.append(far)
            else:
                cut.append((_q(c.at[0] + (far[0] - c.at[0]) * d / full), _q(c.at[1] + (far[1] - c.at[1]) * d / full)))
        return cut[0], cut[1]

    def attempt(c: _Corner) -> tuple[str, str | None, str, float | None]:
        s1 = c.legs[0][0]
        w = float(s1.track.width_mm)
        cl = track_clear(s1.track)
        idx = idx_of[c.net]
        smallest = MIN_CHAMFER_WIDTHS * w
        if c.dmax < smallest - _TOL:
            return ("left", "no_room", f"the tracks leave room for a chamfer of {c.dmax:.4f} mm, below the smallest ({smallest:g} mm: "
                    f"{MIN_CHAMFER_WIDTHS:g} x the track's width)", None)
        d: float | None = c.dmax
        last: tuple[str, str] = ("clearance", "")
        tried = c.dmax
        while d is not None:
            tried = d
            # the next size: half of this one, or the smallest itself where half would fall below it; none after the smallest
            after = None if d <= smallest + _TOL else max(d / 2.0, smallest)
            cut = cut_points(c, d)
            if cut is None:
                d = after
                continue
            a, b = cut
            problem = world.seg_problem(idx, c.li, a, b, w, cl, f"the chamfer at ({c.at[0]:g}, {c.at[1]:g})")
            if problem is not None:
                last = ("clearance", problem)
            else:
                problem = next((f"the chamfer at ({c.at[0]:g}, {c.at[1]:g}) would enter track keep-out {kid}"
                                for kid, pts, layers, allowed in board.track_keepouts
                                if c.li in layers and idx not in allowed and segment_area_distance(a, b, pts) < w / 2.0 - _TOL), None)
                if problem is not None:
                    last = ("keepout", problem)
                elif min(board.edge_distance_xy(*a), board.edge_distance_xy(*b)) < p.edge_clearance_mm + w / 2.0 - _TOL:
                    problem = f"the chamfer at ({c.at[0]:g}, {c.at[1]:g}) would come nearer the board edge than {p.edge_clearance_mm:g} mm"
                    last = ("edge", problem)
            if problem is None:
                problem = _own_problem(c, a, None, b, [a, b], w, *own[(c.net, c.layer)])
                if problem is not None:
                    last = ("own_copper", problem)
            if problem is None and arcs:  # routing.corners 0.2: a chamfer made under the arc style loses no contact either
                problem = _lost_contact(c, a, None, b, [a, b], w, *own[(c.net, c.layer)])
                if problem is not None:
                    last = ("contact_lost", problem)
            if problem is None:
                own[(c.net, c.layer)][2].append(_apply(c, a, None, b, d, None, version, world, idx_of, track_clear, inserted))
                return ("mitred", None, "", _q(d))
            d = after
        return ("left", last[0], f"no chamfer from {c.dmax:.4f} mm down to {tried:.4f} mm keeps every rule: {last[1]}", None)

    def round_corner(c: _Corner, d: float) -> tuple[str, str] | None:
        """routing.corners 0.2: make the arc of tangent length ``d`` at the corner (``None``), or say the first rule it breaks, in the
        order of the module docstring (the audit of an arc, its own net): ``(reason, words)``."""
        s1 = c.legs[0][0]
        w = float(s1.track.width_mm)
        cl = track_clear(s1.track)
        idx = idx_of[c.net]
        where = f"the arc at ({c.at[0]:g}, {c.at[1]:g})"
        cut = cut_points(c, d)
        if cut is None:
            return ("geometry", f"{where} would use a track up to an end the pass keeps")
        a, b = cut
        try:
            _a, mid, _b, radius = fillet(c.at, a, b, d)
        except ValueError as exc:
            return ("geometry", f"{where} has no tangent arc: {exc}")
        flaw = arc_problem(a, mid, b)
        if flaw is not None:
            return ("geometry", f"{where} is no arc at the file's resolution: {flaw}")
        chain = arc_points(a, mid, b)
        chords = list(zip(chain, chain[1:]))
        sag = ARC_SAGITTA_MM
        what = f"{where}, judged on its chords with the half-width raised by their {sag:g} mm bound,"
        for p1, p2 in chords:
            problem = world.seg_problem(idx, c.li, p1, p2, w + 2.0 * sag, cl, what)
            if problem is not None:
                return ("clearance", problem)
        for kid, pts, layers, allowed in board.track_keepouts:
            if c.li in layers and idx not in allowed and any(segment_area_distance(p1, p2, pts) < w / 2.0 + sag - _TOL for p1, p2 in chords):
                return ("keepout", f"{where} would enter track keep-out {kid}, or come within its chords' {sag:g} mm bound of it")
        # points, as for a chamfer's two ends: the edge distance of the router's rectangular board is concave, so a chord is nearest the
        # edge at one of its ends, and the arc lies within the bound of its chords
        if min(board.edge_distance_xy(*pt) for pt in chain) < p.edge_clearance_mm + w / 2.0 + sag - _TOL:
            return ("edge", f"{where} would come nearer the board edge than {p.edge_clearance_mm:g} mm, or within its chords' {sag:g} mm "
                    "bound of that")
        geoms, discs, tracks = own[(c.net, c.layer)]
        problem = _own_problem(c, a, mid, b, chain, w, geoms, discs, tracks)
        if problem is not None:
            return ("own_copper", problem)
        problem = _lost_contact(c, a, mid, b, chain, w, geoms, discs, tracks)
        if problem is not None:
            return ("contact_lost", problem)
        tracks.append(_apply(c, a, mid, b, d, radius, version, world, idx_of, track_clear, inserted, chain))
        rounded[id(c)] = (_q(d), _q(radius))
        arc_ends.update((id(s), e) for s, e in c.legs)
        return None

    if arcs:
        # routing.corners 0.2 (module docstring: the arc - radius, room, rounds): one round per radius step, largest first, each
        # repeated while its last sweep rounded a corner
        made = 0  # the arcs made so far: a corner is tried again at a size only after this moved
        tried_at: dict[int, tuple[float, int]] = {}
        radii: dict[int, tuple[float, float]] = {}  # id(corner) -> the first and the last radius tried
        for step in ARC_RADIUS_WIDTHS:
            sweep = True
            while sweep:
                sweep = False
                for c in corners:
                    if c.skip is not None or id(c) in rounded:
                        continue
                    w = float(c.legs[0][0].track.width_mm)
                    smallest = ARC_RADIUS_WIDTHS[-1] * w
                    d = min(step * w / c.tan_half, c.dmax_arc)
                    radius = d * c.tan_half
                    if radius < smallest - _TOL:
                        no_arc[id(c)] = ("no_room", f"the tracks leave room for an arc of radius {radius:.4f} mm, below the smallest "
                                         f"({smallest:g} mm: {ARC_RADIUS_WIDTHS[-1]:g} x the track's width)")
                        continue
                    if tried_at.get(id(c)) == (d, made):
                        continue  # no arc was made since this size failed here: the result would be the same
                    tried_at[id(c)] = (d, made)
                    radii[id(c)] = (radii.get(id(c), (radius, radius))[0], radius)
                    problem = round_corner(c, d)
                    if problem is None:
                        made += 1
                        sweep = True
                        no_arc.pop(id(c), None)
                    else:
                        first, last_r = radii[id(c)]
                        no_arc[id(c)] = (problem[0], f"no arc from a radius of {first:.4f} mm down to {last_r:.4f} mm keeps every rule: {problem[1]}")

    for c in corners:
        if c.skip is None and id(c) not in rounded:
            if arcs:
                # routing.corners 0.2: the end of an arc is one more point the pass keeps - a chamfer takes at most half of what is left of
                # a track up to it, so the arc still meets that track in its own direction (every arc stays tangent to what it joins)
                limits = []
                for (s, e), limit in zip(c.legs, c.limits):
                    far = _far(s, e)
                    if (id(s), 1 - e) in arc_ends:
                        limit = min(limit, math.hypot(far[0] - c.at[0], far[1] - c.at[1]) / 2.0)
                    limits.append(limit)
                c.dmax = min(limits)
            outcome[id(c)] = attempt(c)

    rows: list[dict[str, Any]] = []
    left: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for c in corners:
        row: dict[str, Any] = {"net": c.net, "layer": c.layer, "at": [c.at[0], c.at[1]], "angle_deg": round(c.angle, 3)}
        if c.skip is not None:
            row.update(outcome="skipped", reason=c.skip, why=c.why)
            skipped[c.skip] = skipped.get(c.skip, 0) + 1
        elif id(c) in rounded:  # routing.corners 0.2
            d, radius = rounded[id(c)]
            row.update(outcome="arc", radius_mm=radius, tangent_mm=d)
        else:
            what, reason, words, d = outcome[id(c)]
            row["outcome"] = what
            if what == "mitred":
                row["chamfer_mm"] = d
            else:
                if arcs:  # routing.corners 0.2: both reasons - why no arc, why no chamfer
                    words = f"{no_arc[id(c)][1]}; {words}"
                row.update(reason=reason, why=words)
                left[reason] = left.get(reason, 0) + 1
            if arcs:  # routing.corners 0.2: why this corner has no arc
                row.update(no_arc_reason=no_arc[id(c)][0], no_arc_why=no_arc[id(c)][1])
        rows.append(row)

    changed: set[str] = set()
    tracks: list[Track] = []
    for s in segs:
        if s.alive:
            tracks.append(_stamped(s, arcs))
        tracks.extend(_stamped(d, arcs) for d in inserted.get(s.order, ()))
        if s.entries or not s.alive:
            changed.add(s.track.net)
    result.tracks = tracks
    _measure_again(result, changed)
    mitred = sum(1 for row in rows if row["outcome"] == "mitred")
    chamfer = (f"half of each track's run to the next point kept (its end, a pad centre, a via or a track end on it; a third when the "
               f"next corner turns the same way), the smaller of the two, halved until every rule holds; a halving below "
               f"{MIN_CHAMFER_WIDTHS:g} x the track's width tries that smallest chamfer instead, the last try")
    if arcs:  # routing.corners 0.2: every key of 0.1, the arcs' count and the arc's rule, then what the straightening did
        result.stats["corners"] = {
            "tool": CORNERS_ID, "version": CORNERS_ARC_VERSION, "style": style, "rule": CORNER_RULE, "basis": CORNER_BASIS, "ideas": CORNER_IDEAS,
            "arc": _arc_words(), "chamfer": chamfer,
            "found": len(rows), "arcs": len(rounded), "mitred": mitred, "left": sum(left.values()), "skipped": sum(skipped.values()),
            "left_reasons": dict(sorted(left.items())), "skipped_reasons": dict(sorted(skipped.items())), "corners": rows,
            **sidesteps,
        }
        return
    result.stats["corners"] = {
        "tool": CORNERS_ID, "version": CORNERS_VERSION, "style": style, "rule": CORNER_RULE, "basis": CORNER_BASIS, "ideas": CORNER_IDEAS,
        "chamfer": chamfer,
        "found": len(rows), "mitred": mitred, "left": sum(left.values()), "skipped": sum(skipped.values()),
        "left_reasons": dict(sorted(left.items())), "skipped_reasons": dict(sorted(skipped.items())), "corners": rows,
    }


def _arc_words() -> str:
    """routing.corners 0.2: the arc's rule in words, for ``stats["corners"]["arc"]`` (module docstring: the arc)."""
    steps = ", ".join(f"{step:g}" for step in ARC_RADIUS_WIDTHS[:-1]) + f" and {ARC_RADIUS_WIDTHS[-1]:g}"
    return (f"a tangent arc of radius {steps} track widths, largest first - or, where the tracks leave less room, of the radius that room "
            f"gives, never below {ARC_RADIUS_WIDTHS[-1]:g} width; the room is half of each track's run to the next point kept (its end, a pad "
            "centre, a via or a track end on it), the smaller of the two; one round per radius, repeated while it rounded a corner; an arc is "
            f"judged on its chain of chords with the chain's {ARC_SAGITTA_MM:g} mm bound on the refusing side (every clearance, track keep-out "
            "and the board edge) and may neither make nor lose a contact within its net; a corner no arc fits tries the 45-degree chamfer, "
            "which keeps the end of an arc as one more point (half of what is left of the track up to it) and may not lose a contact "
            "either, and stays as routed where neither fits")


def _apply(c: _Corner, a: Point, mid: Point | None, b: Point, d: float, radius: float | None, version: str, world: _Copper,
           idx_of: dict[str, int], track_clear, inserted: dict[int, list[_Seg]], chain: list[Point] | None = None) -> _Seg:
    """Cut the corner: each track's end at the corner moves to its cut point (a track used up is dropped), the new piece - the segment
    ``a``-``b``, or with ``mid`` (routing.corners 0.2) the arc through it, whose ``chain`` is given - follows the earlier track
    (returned); the audit's copper is updated and every track touched carries the corner's entry. routing.corners 0.2: a track used
    up passes its straightening entries (module docstring: the straightening, records) to the new piece, before the corner's own,
    so the record of every move stays on the copper (a track without such an entry passes nothing: the copper of a board on which
    nothing was straightened, and every 0.1 result, is the one it was)."""
    if mid is None:
        entry = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {version}:{MITER45}:at={c.at[0]!r},{c.at[1]!r}:chamfer={_q(d)!r}"
    else:
        entry = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {version}:{ARC}:at={c.at[0]!r},{c.at[1]!r}:tangent={_q(d)!r}:radius={_q(radius)!r}"
    idx = idx_of[c.net]
    moves: list[str] = []  # routing.corners 0.2: the straightening's entries of a track this cut uses up
    for (s, e), point in zip(c.legs, (a, b)):
        world.take(s)
        far = _far(s, e)
        if _near(point, far):
            s.alive = False
            moves.extend(x for x in s.entries if x.startswith(_STRAIGHTEN_ENTRY) and x not in moves)
            continue
        if e == 0:
            s.a = point
        else:
            s.b = point
        s.entries.append(entry)
        world.put(s, idx, track_clear(s.track))
    first = c.legs[0][0]
    piece = _Seg(first.order, first.track, a, b, False, [*moves, entry], mid=mid, chain=chain if mid is not None else None,
                 box=_box_of(chain) if mid is not None else None)
    inserted.setdefault(first.order, []).append(piece)
    world.put(piece, idx, track_clear(first.track))
    return piece


def _in_box(p: Point, box: tuple[float, float, float, float]) -> bool:
    return box[0] - _TOL <= p[0] <= box[2] + _TOL and box[1] - _TOL <= p[1] <= box[3] + _TOL


# --------------------------------------------------------------------------- a piece: a straight segment, or an arc on its chain


def _pts(s: _Seg) -> list[Point]:
    """A track of the pass as the points of its centreline: its two ends, or (routing.corners 0.2) an arc's chain."""
    return [s.a, s.b] if s.mid is None else s.chain


def _bound(mid: Point | None) -> float:
    """How far a piece may lie from its points' chords: nothing for a straight one, the chain's bound for an arc."""
    return 0.0 if mid is None else ARC_SAGITTA_MM


def _box_of(pts: list[Point]) -> tuple[float, float, float, float]:
    xs, ys = [x for x, _y in pts], [y for _x, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _apart(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    """The distance between two boxes: no two points, one in each, are nearer."""
    return math.hypot(max(b1[0] - b2[2], b2[0] - b1[2], 0.0), max(b1[1] - b2[3], b2[1] - b1[3], 0.0))


def _pt_piece(q: Point, a: Point, mid: Point | None, b: Point) -> float:
    """The exact distance of a point from a piece's centreline: the segment ``a``-``b``, or the arc through ``mid``."""
    return _pt_seg(q[0], q[1], a, b) if mid is None else point_arc_distance(q, a, mid, b)


def _chain_box(pts: list[Point], box: tuple[float, float, float, float]) -> float:
    """The distance of a chain of chords from a box (a straight piece is one chord)."""
    if len(pts) == 2:
        return _seg_box(pts[0], pts[1], box)
    return min(_seg_box(p, q, box) for p, q in zip(pts, pts[1:]))


def _chain_gap(p: list[Point], q: list[Point]) -> float:
    """The distance between two chains of chords, 0 where they touch or cross (a straight piece is one chord)."""
    if len(p) == 2 and len(q) == 2:
        return _seg_seg(p[0], p[1], q[0], q[1])
    return min(_seg_seg(a, b, c, d) for a, b in zip(p, p[1:]) for c, d in zip(q, q[1:]))


def _name(t: _Seg) -> str:
    return f"{'track' if t.mid is None else 'arc'} ({t.a[0]:g}, {t.a[1]:g})-({t.b[0]:g}, {t.b[1]:g})"


def _own_problem(c: _Corner, a: Point, mid: Point | None, b: Point, pts: list[Point], w: float, geoms: list[_PadGeom],
                 discs: list[tuple[Point, float]], tracks: list[_Seg]) -> str | None:
    """Why the new piece from ``a`` to ``b`` (width ``w``; a segment, or with ``mid`` the arc whose chain is ``pts``) would join its
    own net's copper where the corner did not (module docstring: connectivity): a cut point on a pad box or via disc of the net that
    its track's far end is not on; a via centre of the net on the new piece; its centreline meeting another live track of the net
    anywhere but at a shared end; or its copper reaching a pad box, a via disc or another live track's copper of the net that neither
    of the two tracks' copper (as they stand) reached; else ``None``.

    routing.corners 0.2 (module docstring: the arc, its own net): a point against an arc is the exact distance; an arc against a
    box or another track, and anything against an arc the pass made earlier, is judged on the chains with their bounds on the
    refusing side - the new piece reaches what it may reach, the two tracks reached only what they surely reached."""
    where = f"the {'chamfer' if mid is None else 'arc'} at ({c.at[0]:g}, {c.at[1]:g})"
    legs = [s for s, _e in c.legs]
    half = w / 2.0
    sag = _bound(mid)
    for (s, e), x in zip(c.legs, (a, b)):
        far = _far(s, e)
        for g in geoms:
            if _in_box(x, g.box) and not _in_box(far, g.box):
                return f"{where} would end on pad {g.ref}.{g.number} of its own net, which its track did not reach (a new junction)"
        for centre, r in discs:
            if math.hypot(x[0] - centre[0], x[1] - centre[1]) <= r + _TOL and not _near(far, centre):
                return f"{where} would end on a via of its own net at ({centre[0]:g}, {centre[1]:g}) (a new junction)"
    for q, _r in discs:
        if not _near(q, a) and not _near(q, b) and _pt_piece(q, a, mid, b) <= _TOL:
            return f"{where} would pass through the via of its own net at ({q[0]:g}, {q[1]:g}) (a new junction)"
    for g in geoms:
        if _chain_box(pts, g.box) <= half + sag + _TOL and all(_seg_box(s.a, s.b, g.box) > half + _TOL for s in legs):
            return f"{where} would reach pad {g.ref}.{g.number} of its own net, which its tracks' copper did not reach (a new junction)"
    for q, r in discs:
        if _pt_piece(q, a, mid, b) <= r + half + _TOL and all(_pt_seg(q[0], q[1], s.a, s.b) > r + half + _TOL for s in legs):
            return f"{where} would reach the via of its own net at ({q[0]:g}, {q[1]:g}), which its tracks' copper did not reach (a new junction)"
    box = _box_of(pts)
    for t in tracks:
        if not t.alive or any(t is s for s in legs):
            continue
        tp = _pts(t)
        other = _bound(t.mid)
        bound = sag + other
        reach = half + float(t.track.width_mm) / 2.0 + _TOL
        if bound and _apart(box, t.box if t.box is not None else _box_of(tp)) > reach + bound:
            continue  # an arc is in play and the two are further apart than any test below reaches
        gap = _chain_gap(pts, tp)
        if gap <= bound + _TOL and not _meets_at_end(a, mid, b, pts, t):
            return f"{where} would meet its own net's {_name(t)} (a new junction)"
        if gap <= reach + bound and all(_chain_gap([s.a, s.b], tp) > reach - other for s in legs):
            return f"{where} would reach its own net's {_name(t)}, which its tracks' copper did not reach (a new junction)"
    return None


def _meets_at_end(a: Point, mid: Point | None, b: Point, pts: list[Point], t: _Seg) -> bool:
    """Whether the new piece (``a`` to ``b``, its points ``pts``) and the track ``t`` meet only at one end they share (a chamfer joining
    a chain at its cut point - a Z's two chamfers): one of ``t``'s ends is ``a`` or ``b`` and neither runs along the other.

    routing.corners 0.2: with an arc on either side (two arcs meeting on a used-up track, an arc and a chamfer) the far ends are
    judged by the exact distance, and, because a curve may come back to the other piece, the two must also leave the shared end in
    directions at least a right angle apart and stay further apart than their chains' bounds beyond their first chords."""
    for i, (q, o) in enumerate(((a, b), (b, a))):
        for j, (e, f) in enumerate(((t.a, t.b), (t.b, t.a))):
            if _near(e, q):
                if not (_pt_piece(f, a, mid, b) > _TOL and _pt_piece(o, t.a, t.mid, t.b) > _TOL):
                    return False
                if mid is None and t.mid is None:
                    return True  # two straight segments from one point meet again only by running along each other
                # both chains from the shared end outward
                mine = pts if i == 0 else pts[::-1]
                theirs = _pts(t) if j == 0 else _pts(t)[::-1]
                (m0, m1), (t0, t1) = mine[:2], theirs[:2]
                if (m1[0] - m0[0]) * (t1[0] - t0[0]) + (m1[1] - m0[1]) * (t1[1] - t0[1]) > 0.0:
                    return False
                bound = _bound(mid) + _bound(t.mid) + _TOL
                return all(_seg_seg(p1, p2, q1, q2) > bound
                           for m, (p1, p2) in enumerate(zip(mine, mine[1:])) for n, (q1, q2) in enumerate(zip(theirs, theirs[1:])) if m or n)
    return False


def _lost_contact(c: _Corner, a: Point, mid: Point | None, b: Point, pts: list[Point], w: float, geoms: list[_PadGeom],
                  discs: list[tuple[Point, float]], tracks: list[_Seg]) -> str | None:
    """routing.corners 0.2 (module docstring: the arc, its own net): why the new piece from ``a`` to ``b`` would take the net's copper
    away from something of the net it touched, or ``None``. The cut removes the two track ends between the corner and the cut
    points; an item of the net on the layer - a pad box, a via disc, another live track or arc - whose copper touched one of them
    must still touch the new piece or what is left of the two tracks, because the removed copper may have been its only contact.

    The bounds are on the refusing side: an item touched the removed copper when it may have (an earlier arc with its chain's bound
    added), and still touches when it surely does (an arc's chain with the bound taken off; a point against an arc exactly)."""
    where = f"the {'chamfer' if mid is None else 'arc'} at ({c.at[0]:g}, {c.at[1]:g})"
    half = w / 2.0
    sag = _bound(mid)
    removed = [(c.at, x) for x in (a, b)]
    kept = [(x, _far(s, e)) for (s, e), x in zip(c.legs, (a, b)) if not _near(x, _far(s, e))]
    words = "the copper it removes touched it, the copper it leaves does not"
    for g in geoms:
        if (any(_seg_box(p, q, g.box) <= half + _TOL for p, q in removed) and _chain_box(pts, g.box) > half - sag + _TOL
                and all(_seg_box(p, q, g.box) > half + _TOL for p, q in kept)):
            return f"{where} would cut its own net's contact with pad {g.ref}.{g.number}: {words}"
    for q, r in discs:
        reach = r + half + _TOL
        if (any(_pt_seg(q[0], q[1], p1, p2) <= reach for p1, p2 in removed) and _pt_piece(q, a, mid, b) > reach
                and all(_pt_seg(q[0], q[1], p1, p2) > reach for p1, p2 in kept)):
            return f"{where} would cut its own net's contact with the via at ({q[0]:g}, {q[1]:g}): {words}"
    legs = [s for s, _e in c.legs]
    box = _box_of([c.at, a, b])
    for t in tracks:
        if not t.alive or any(t is s for s in legs):
            continue
        tp = _pts(t)
        other = _bound(t.mid)
        reach = half + float(t.track.width_mm) / 2.0 + _TOL
        if _apart(box, t.box if t.box is not None else _box_of(tp)) > reach + other:
            continue  # too far from the corner's triangle to have touched what the cut removes
        if (any(_chain_gap([p1, p2], tp) <= reach + other for p1, p2 in removed) and _chain_gap(pts, tp) > reach - sag - other
                and all(_chain_gap([p1, p2], tp) > reach - other for p1, p2 in kept)):
            return f"{where} would cut its own net's contact with its {_name(t)}: {words}"
    return None


# --------------------------------------------------------------------------- routing.corners 0.2: the straightening


def _axis(a: Point, b: Point) -> int | None:
    """The axis a segment runs along - 0: X, 1: Y - or ``None`` (at another angle, or without length). Exact, on the stored
    coordinates: the straightening moves copper only along the lines it already lies on."""
    if a[1] == b[1] and a[0] != b[0]:
        return 0
    if a[0] == b[0] and a[1] != b[1]:
        return 1
    return None


def _heading(a: Point, b: Point) -> tuple[int, int]:
    """The direction from ``a`` to ``b`` as the signs of its two components."""
    return (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])


def _on(q: Point, segs: list[tuple[Point, Point]]) -> bool:
    """Whether a point lies on the centreline of one of the segments."""
    return any(_pt_seg(q[0], q[1], a, b) <= _TOL for a, b in segs)


def _minus(seg: tuple[Point, Point], cover: list[tuple[Point, Point]]) -> list[tuple[Point, Point]]:
    """The parts of a segment along an axis that lie on none of the segments ``cover`` (only those on its own line take anything
    away), each longer than the pass's tolerance."""
    a, b = seg
    ax = _axis(a, b)
    line = a[1 - ax]
    lo, hi = min(a[ax], b[ax]), max(a[ax], b[ax])
    free, at = [], lo
    for start, end in sorted((min(c[ax], d[ax]), max(c[ax], d[ax])) for c, d in cover if c[1 - ax] == line and d[1 - ax] == line):
        if start - at > _TOL:
            free.append((at, min(start, hi)))
        at = max(at, end)
        if at >= hi:
            break
    if hi - at > _TOL:
        free.append((at, hi))
    return [((v1, line), (v2, line)) if ax == 0 else ((line, v1), (line, v2)) for v1, v2 in free]


@dataclass(eq=False)
class _OwnCopper:
    """routing.corners 0.2 (module docstring: the straightening): one net's copper on one layer as the step reads it - the net's pad
    boxes there with their centres, its vias (centre, radius), its tracks (the step marks the ones a move uses up or merges away),
    which live track ends meet where (``ends``, built again after a move) and what the pass keeps at each such point (``known``,
    worked out when asked and forgotten where a move changed the copper). ``neck`` is the net's neck-down width where its rule has
    one that differs from its width (a track at that width is the neck-down rule's: the step leaves it where it is)."""

    net: str
    layer: str
    li: int
    idx: int
    geoms: list[_PadGeom]
    centres: list[Point]
    discs: list[tuple[Point, float]]
    tracks: list[_Seg]
    neck: float | None = None
    ends: dict[Point, list[tuple[_Seg, int]]] = field(default_factory=dict)
    known: dict[Point, str | None] = field(default_factory=dict)

    def refresh(self, moved: list[tuple[Point, Point]] | None = None) -> None:
        """Read the tracks again. After a move, ``moved`` is the copper it took and the copper it laid (segments): what the pass
        keeps at a point can only have changed where the point lies on one of them - a track that ends there or passes there is
        one the move changed - so only those points are forgotten."""
        self.ends = {}
        for s in self.tracks:
            if s.alive:
                self.ends.setdefault(s.a, []).append((s, 0))
                self.ends.setdefault(s.b, []).append((s, 1))
        self.known = {} if moved is None else {pt: words for pt, words in self.known.items() if not _on(pt, moved)}

    def pinned(self, pt: Point, but: list[_Seg]) -> str | None:
        """What of the net pins a point, whatever tracks end there - a via it is at or inside, a pad centre, a pad box, another live
        track it lies on (none of ``but``) - in words, or ``None``: the tests by which the pass skips a corner."""
        for q, r in self.discs:
            if math.hypot(pt[0] - q[0], pt[1] - q[1]) <= r + _TOL:
                return f"at or inside the via of its net at ({q[0]:g}, {q[1]:g})"
        for g, centre in zip(self.geoms, self.centres):
            if _near(pt, centre):
                return f"the centre of pad {g.ref}.{g.number} of its net"
        for g in self.geoms:
            if _in_box(pt, g.box):
                return f"inside the box of pad {g.ref}.{g.number} of its net"
        for s in self.tracks:
            if s.alive and all(s is not t for t in but) and _pt_seg(pt[0], pt[1], s.a, s.b) <= _TOL:
                return f"on its own net's {_name(s)} (a junction)"
        return None

    def kept(self, pt: Point) -> str | None:
        """What the pass keeps at a point where track ends meet, in words - or ``None`` at a plain corner (module docstring: the
        straightening): two tracks, one along X and one along Y, of one width, neither an escape stub nor a neck-down, and nothing
        pinning it."""
        if pt not in self.known:
            at = self.ends[pt]
            here = [s for s, _e in at]
            words = self.pinned(pt, here)
            if words is None and len(at) == 1:
                words = "where its copper ends (no other track ends there)"
            elif words is None and (len(at) != 2 or here[0] is here[1]):
                words = f"a junction of {len(at)} track ends of its net"
            elif words is None:
                s1, s2 = here
                w1, w2 = float(s1.track.width_mm), float(s2.track.width_mm)
                if s1.escape or s2.escape:
                    words = "the end of an escape stub"
                elif abs(w1 - w2) > _TOL:
                    words = f"a width change ({w1:g} / {w2:g} mm)"
                elif self.neck is not None and abs(w1 - self.neck) <= _TOL:
                    words = f"the end of a neck-down track ({w1:g} mm: its place is the neck-down rule's, within the rule's radius of a pad)"
                elif {_axis(s1.a, s1.b), _axis(s2.a, s2.b)} != {0, 1}:
                    words = "no right angle between a track along X and a track along Y"
            self.known[pt] = words
        return self.known[pt]

    def sidestep(self, s: _Seg) -> bool:
        """Whether a track is a sidestep as the copper stands: alive, along an axis, shorter than :data:`SIDESTEP_WIDTHS` track
        widths, between two plain corners."""
        if not s.alive or _axis(s.a, s.b) is None:
            return False
        length = abs(s.b[0] - s.a[0]) + abs(s.b[1] - s.a[1])
        return length < SIDESTEP_WIDTHS * float(s.track.width_mm) - _TOL and self.kept(s.a) is None and self.kept(s.b) is None

    def run(self, m: _Seg) -> tuple[list[_Seg], list[Point], int] | None:
        """The run of tracks through plain corners that holds the sidestep ``m``, from one point the pass keeps to the next: its
        tracks in path order, their points (track ``i`` joins point ``i`` and point ``i + 1``) and ``m``'s place among the tracks.
        ``None`` where the run closes on itself (a ring of plain corners: the pass keeps no point of it)."""
        sides: list[tuple[list[_Seg], list[Point]]] = []
        for e in (0, 1):
            cur, pt = m, (m.a if e == 0 else m.b)
            tracks: list[_Seg] = []
            pts = [pt]
            while self.kept(pt) is None:
                (s1, e1), (s2, e2) = self.ends[pt]
                cur, at = (s2, e2) if s1 is cur else (s1, e1)
                if cur is m:
                    return None
                pt = _far(cur, at)
                tracks.append(cur)
                pts.append(pt)
            sides.append((tracks, pts))
        (before, pts_before), (after, pts_after) = sides
        return [*before[::-1], m, *after], [*pts_before[::-1], *pts_after], len(before)


@dataclass(eq=False)
class _Slide:
    """routing.corners 0.2: a legal move of the straightening (module docstring: the straightening, the move). ``pieces`` is the run
    after it, in path order - each ``[start, end, the tracks joined into it, the tracks whose entries pass to it]`` -, ``changed``
    the pieces that are not a track of the run as it lay, ``involved`` the tracks those replace, ``lay`` where each track of the
    run lay (its two points in path order), ``to`` the corner the sidestep slides to and ``rank`` what decides between two legal
    moves: the tracks the run keeps (one more than its corners), its copper, the place of the neighbour used up in the track order."""

    pieces: list[list]
    changed: list[list]
    involved: list[_Seg]
    lay: dict[int, tuple[Point, Point]]
    to: Point
    rank: tuple[int, float, int]


def _fold(pieces: list[list]) -> None:
    """Join what a slide left on one line, until nothing is: a piece without length goes (the entries of its tracks pass to the piece
    before it, or to the one after), and two pieces on one line that share an end become one - from the start of the first to the
    end of the second: their sum where the second continues the first, their difference where it runs back over it."""
    while True:
        gone = next((i for i, piece in enumerate(pieces) if piece[0] == piece[1]), None)
        if gone is not None:
            _a, _b, joined, passed = pieces.pop(gone)
            if pieces:
                pieces[gone - 1 if gone else 0][3].extend([*joined, *passed])
            continue
        for i in range(len(pieces) - 1):
            (a, b, joined, passed), (_b, c, joined2, passed2) = pieces[i], pieces[i + 1]
            if (a[0] == b[0] and b[0] == c[0]) or (a[1] == b[1] and b[1] == c[1]):
                pieces[i:i + 2] = [[a, c, [*joined, *joined2], [*passed, *passed2]]]
                break
        else:
            return


def _slide(own: _OwnCopper, world: _Copper, board: _Board, tracks: list[_Seg], pts: list[Point], k: int, cl: float,
           what: str) -> _Slide | tuple[str, str]:
    """routing.corners 0.2 (module docstring: the straightening - the move, kept points, the audit of a move): slide the sidestep
    ``tracks[k]`` of a run along the next track, ``tracks[k + 1]``, to that track's far end. A :class:`_Slide` where the move is
    legal (nothing is changed here), else ``(reason, words)`` of the first rule it breaks."""
    m = tracks[k]
    w = float(m.track.width_mm)
    half = w / 2.0
    p0, p1, p2, p3 = pts[k - 1], pts[k], pts[k + 1], pts[k + 2]
    if k + 2 == len(tracks):  # the far end of the neighbour is the end of the run
        return "kept_point", f"the pass keeps that point: {own.kept(p3)}"
    ax = _axis(p2, p3)  # the axis the neighbour runs along; the sidestep runs along the other one
    p1n = (p3[0], p1[1]) if ax == 0 else (p1[0], p3[1])  # P1 + (P3 - P2), on the stored coordinates
    side = p0[ax] - p1[ax]
    if side * (p3[ax] - p2[ax]) > 0.0 and (p3[ax] - p0[ax]) * side > 0.0:  # a U whose other neighbour is the shorter one
        return "kept_point", (f"the sidestep would pass ({p0[0]:g}, {p0[1]:g}), the end of its shorter neighbour (a U slides only as far as "
                              "the shorter of its two neighbours)")
    pieces = [[pts[i], pts[i + 1], [t], []] for i, t in enumerate(tracks)]
    pieces[k - 1][1] = p1n
    pieces[k][0], pieces[k][1] = p1n, p3
    pieces[k][3].append(tracks[k + 1])  # the neighbour is used up: its entries pass to the slid track
    del pieces[k + 1]
    _fold(pieces)
    for point, toward, piece in ((pts[0], pts[1], pieces[:1]), (pts[-1], pts[-2], pieces[-1:])):
        # the two ends of the run are points the pass keeps: the copper still leaves each in the direction it had
        leaves = [(a, b) if a == point else (b, a) for a, b, _joined, _passed in piece]
        if not leaves or leaves[0][0] != point or _heading(*leaves[0]) != _heading(point, toward):
            return "kept_point", (f"the copper would meet ({point[0]:g}, {point[1]:g}) from a new direction, and the pass keeps that point: "
                                  f"{own.kept(point)}")
    lay = {id(t): (pts[i], pts[i + 1]) for i, t in enumerate(tracks)}
    changed = [piece for piece in pieces if piece[3] or len(piece[2]) != 1 or (piece[0], piece[1]) != lay[id(piece[2][0])]]
    involved = [t for _a, _b, joined, passed in changed for t in (*joined, *passed)]
    old = [lay[id(t)] for t in involved]
    new = [(a, b) for a, b, _joined, _passed in changed]
    others = [t for t in own.tracks if t.alive and all(t is not s for s in involved)]
    pins = [*((q, "a via") for q, _r in own.discs),
            *((centre, f"the centre of pad {g.ref}.{g.number}") for g, centre in zip(own.geoms, own.centres)),
            *((q, f"an end of {_name(t)}") for t in others for q in (t.a, t.b))]
    onto: tuple[Point, str] | None = None
    for q, name in pins:
        before, after = _on(q, old), _on(q, new)
        if before and not after:
            return "kept_point", f"{name} of its net at ({q[0]:g}, {q[1]:g}) lies on copper the move would take away"
        if after and not before and onto is None:
            onto = (q, name)
    if any(piece[0] == p1n for piece in pieces[1:]):  # the slid copper still turns there: the point must be fit for a plain corner
        pinned = own.pinned(p1n, involved)
        if pinned is not None:
            return "kept_point", f"the slid copper would turn at ({p1n[0]:g}, {p1n[1]:g}), {pinned}"
    # the audit of a move, in the module docstring's order: other nets, keep-outs and the edge, the net's own copper
    fresh = [part for seg in new for part in _minus(seg, old)]
    for a, b in fresh:
        problem = world.seg_problem(own.idx, own.li, a, b, w, cl, what)
        if problem is not None:
            return "clearance", problem
    for kid, area, layers, allowed in board.track_keepouts:
        if own.li in layers and own.idx not in allowed and any(segment_area_distance(a, b, area) < half - _TOL for a, b in fresh):
            return "keepout", f"{what} would enter track keep-out {kid}"
    edge = board.p.edge_clearance_mm
    if any(min(board.edge_distance_xy(*a), board.edge_distance_xy(*b)) < edge + half - _TOL for a, b in fresh):
        return "edge", f"{what} would come nearer the board edge than {edge:g} mm"
    if onto is not None:
        return "own_copper", f"{what} would pass through {onto[1]} of its own net at ({onto[0][0]:g}, {onto[0][1]:g}) (a new junction)"
    both = [*old, *new]
    touched = [(f"pad {g.ref}.{g.number} of its own net", [_seg_box(a, b, g.box) <= half + _TOL for a, b in both]) for g in own.geoms]
    touched += [(f"the via of its own net at ({q[0]:g}, {q[1]:g})", [_pt_seg(q[0], q[1], a, b) <= r + half + _TOL for a, b in both])
                for q, r in own.discs]
    touched += [(f"its own net's {_name(t)}", [_seg_seg(a, b, t.a, t.b) <= half + float(t.track.width_mm) / 2.0 + _TOL for a, b in both])
                for t in others]
    for name, hits in touched:
        before, after = any(hits[:len(old)]), any(hits[len(old):])
        if after and not before:
            return "own_copper", f"{what} would reach {name}, which the copper it replaces did not reach (a new contact)"
        if before and not after:
            return "own_copper", f"{what} would leave {name}, which the copper it replaces reached (a contact lost)"
    length = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b, _joined, _passed in pieces)
    return _Slide(pieces, changed, involved, lay, p3, (len(pieces), _q(length), tracks[k + 1].order))


def _straighten(groups: dict[tuple[str, str], list[_Seg]], world: _Copper, board: _Board, result: Routing, pads_of: dict[int, list[_PadGeom]],
                idx_of: dict[str, int], track_clear) -> dict[str, Any]:
    """routing.corners 0.2 (module docstring: the straightening): remove the router's sidesteps wherever the copper may legally run
    without them, on the pass's own tracks - a track a move changes gets its new ends and the move's entry, one it uses up or merges
    away is marked dead, and the audit's copper follows every move at once. Returns what ``stats["corners"]`` gains."""
    owns: list[_OwnCopper] = []
    for (net, layer), group in groups.items():
        rule = result.rules.get(net)
        if rule is not None and (rule.pair_partner is not None or rule.match_group is not None):
            continue  # a net the pass skips entirely stays as routed
        li = board.layer_index(layer)
        geoms = [g for g in pads_of.get(idx_of[net], ()) if li in g.layers]
        discs = [((v.x_mm, v.y_mm), v.diameter_mm / 2.0) for v in result.vias if v.net == net and li in board.via_layers_of(v)]
        neck = None  # the net's neck-down width, where its rule has one that is not the net's width (as the pass's clearance reads it)
        if rule is not None and rule.neckdown_width_mm is not None:
            width = rule.width_mm if rule.width_mm is not None else board.p.track_width_mm
            if abs(float(rule.neckdown_width_mm) - float(width)) > _TOL:
                neck = float(rule.neckdown_width_mm)
        own = _OwnCopper(net, layer, li, idx_of[net], geoms, [(_q(g.cx), _q(g.cy)) for g in geoms], discs, group, neck)
        own.refresh()
        owns.append(own)

    def step(s: _Seg) -> float:
        return _q(abs(s.b[0] - s.a[0]) + abs(s.b[1] - s.a[1]))

    rows: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    for sweep in range(SIDESTEP_SWEEPS + 1):
        last = sweep == SIDESTEP_SWEEPS  # the bound is reached: one more look, which moves nothing and says what it kept from moving
        left = []
        moved = False
        found = [(own, s) for own in owns for s in own.tracks if own.sidestep(s)]
        for own, m in sorted(found, key=lambda f: (step(f[1]), f[1].order)):  # shortest first, then in track order
            if not own.sidestep(m):
                continue  # a move of this sweep took it, lengthened it or changed its corners
            at = (_q((m.a[0] + m.b[0]) / 2.0), _q((m.a[1] + m.b[1]) / 2.0))
            row: dict[str, Any] = {"net": own.net, "layer": own.layer, "at": [at[0], at[1]], "step_mm": step(m)}
            run = own.run(m)
            if run is None:
                left.append({**row, "outcome": "left", "reason": "kept_point",
                             "why": "its copper closes on itself through plain corners: the pass keeps no point of it to slide the sidestep to"})
                continue
            tracks, pts, k = run
            what = f"the straightened copper of the sidestep at ({at[0]:g}, {at[1]:g})"
            cl = track_clear(m.track)
            # the move along the neighbour at the sidestep's second end, and the mirrored one (the run read the other way round)
            tries = [(pts[k + 2], _slide(own, world, board, tracks, pts, k, cl, what)),
                     (pts[k - 1], _slide(own, world, board, tracks[::-1], pts[::-1], len(tracks) - 1 - k, cl, what))]
            legal = [slide for _to, slide in tries if isinstance(slide, _Slide)]
            if not legal:
                reason = next((slide[0] for _to, slide in tries if slide[0] != "kept_point"), "kept_point")
                left.append({**row, "outcome": "left", "reason": reason,
                             "why": "; ".join(f"toward ({to[0]:g}, {to[1]:g}): {slide[1]}" for to, slide in tries)})
                continue
            if last:
                left.append({**row, "outcome": "left", "reason": "sweep_cap",
                             "why": f"it could still be slid, but the step stops after {SIDESTEP_SWEEPS} sweeps"})
                continue
            best = min(legal, key=lambda slide: slide.rank)
            copper = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:]))
            entry = f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {CORNERS_ARC_VERSION}:{STRAIGHTEN}:at={at[0]!r},{at[1]!r}:step={step(m)!r}"
            _make(own, world, best, entry, track_clear)
            rows.append({**row, "outcome": "straightened", "slid_to": [best.to[0], best.to[1]], "corners_removed": len(tracks) - best.rank[0],
                         "shorter_mm": _q(copper - best.rank[1])})
            moved = True
        if not moved:
            break
    reasons: dict[str, int] = {}
    for row in left:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {"straighten": _straighten_words(), "sidesteps": len(rows) + len(left), "straightened": len(rows), "sidesteps_left": len(left),
            "sidestep_reasons": dict(sorted(reasons.items())), "sidestep_rows": [*rows, *left]}


def _make(own: _OwnCopper, world: _Copper, slide: _Slide, entry: str, track_clear) -> None:
    """Make a legal move of the straightening: the tracks it changes leave the audit's copper; each changed piece becomes the
    earliest of the tracks joined into it (the track order keeps its place), which keeps its own direction along the run, takes the
    entries of the tracks joined into it or passed to it and then the move's own, and enters the audit's copper again; the other
    tracks are gone. From here on such a track is the pass's track as if the router had drawn it there: its ends are ends the pass
    keeps."""
    for t in slide.involved:
        world.take(t)
    stays: list[_Seg] = []
    for a, b, joined, passed in slide.changed:
        s = min(joined, key=lambda t: t.order)
        entries: list[str] = []
        for t in sorted([*joined, *passed], key=lambda t: t.order):
            entries.extend(e for e in t.entries if e not in entries)
        s.a, s.b = (a, b) if slide.lay[id(s)][0] == s.a else (b, a)
        s.entries = [*entries, entry]
        s.track = s.track.model_copy(update={"start": s.a, "end": s.b})
        stays.append(s)
    for t in slide.involved:
        if any(t is s for s in stays):
            world.put(t, own.idx, track_clear(t.track))
        else:
            t.alive = False
    own.refresh([*(slide.lay[id(t)] for t in slide.involved), *((a, b) for a, b, _joined, _passed in slide.changed)])


def _straighten_words() -> str:
    """routing.corners 0.2: the straightening's rule in words, for ``stats["corners"]["straighten"]`` (module docstring: the
    straightening)."""
    return (f"before the corners are found, a sidestep of the router - a track shorter than {SIDESTEP_WIDTHS:g} track widths (the room two arcs "
            "of the largest radius need) between two right-angle corners the pass would not skip, neither of them on a track at its net's "
            "neck-down width - is slid along one of its two neighbours to that neighbour's far corner and joined with the track that leaves "
            "it there (one track where that one continues it, their difference where it runs back over it); never into or across a point the "
            "pass keeps (a pad, a via, a junction, a track end); the copper a move adds passes the exact audit of a chamfer (every clearance, "
            "track keep-out and the board edge), and the move may neither make nor lose a contact within its net; a move never lengthens the "
            "net; of two legal moves the one that leaves fewer corners, then less copper; shortest sidestep first, sweep after sweep until "
            f"none moves (at most {SIDESTEP_SWEEPS} sweeps); a sidestep that cannot be slid legally stays as routed, with the reason")


def _stamped(s: _Seg, arcs: bool = False) -> Track:
    """The track as the pass leaves it: unchanged (the same object) without a corner entry, else its new ends and its stamp (under the
    arc style, routing.corners 0.2, the arc's ``mid`` and that style's note, which also counts the sidesteps straightened along the
    track)."""
    if not s.entries:
        return s.track
    prov = s.track.provenance
    if arcs:
        # the straightening's entries (module docstring: the straightening) and the corners': each is said in its own words
        slid = sum(1 for e in s.entries if e.startswith(f"{CORNER_ENTRY_PREFIX}{CORNERS_ID} {CORNERS_ARC_VERSION}:{STRAIGHTEN}:"))
        cut = len(s.entries) - slid
        did = (f"{cut} corner(s) of 90 degrees or sharper of this net rounded into a tangent arc, or mitred into a 45-degree chamfer where no "
               f"arc fits (every clearance checked exactly, an arc on its chords with their {ARC_SAGITTA_MM:g} mm bound on the refusing side; "
               "corner entries in derived_from)")
        if slid:
            straightened = f"{slid} sidestep(s) of the router on this net straightened"
            did = f"{straightened}, then {did}" if cut else f"{straightened} (every clearance checked exactly; corner entries in derived_from)"
        note = (f"{prov.note}; {CORNERS_ID} {CORNERS_ARC_VERSION}: {did} - the project owner's corner-style rule (2026-11-13), a geometry "
                "convention, not an electrical requirement: no signal-integrity or manufacturing benefit is claimed")
    else:
        note = (f"{prov.note}; {CORNERS_ID} {CORNERS_VERSION}: {len(s.entries)} corner(s) of 90 degrees or sharper of this net mitred into a "
                "45-degree chamfer (every clearance checked exactly; corner entries in derived_from) - the project owner's corner-style rule "
                "(2026-11-13), a geometry convention, not an electrical requirement: no signal-integrity or manufacturing benefit is claimed")
    prov = prov.model_copy(update={"derived_from": [*prov.derived_from, *s.entries], "note": note})
    if s.mid is not None:
        return Track(net=s.track.net, layer=s.track.layer, start=s.a, mid=s.mid, end=s.b, width_mm=s.track.width_mm, provenance=prov)
    return Track(net=s.track.net, layer=s.track.layer, start=s.a, end=s.b, width_mm=s.track.width_mm, provenance=prov)


def _measure_again(result: Routing, changed: set[str]) -> None:
    """The stats measured on the copper (net lengths and their total, the track count, the rule and plane-net rows) for the nets the
    pass changed, as the router measured them (an arc by its own length: :func:`~ai_eda.ir.track_length`)."""
    s = result.stats
    s["track_count"] = len(result.tracks)
    if not changed:
        return
    length = {name: 0.0 for name in changed}
    for t in result.tracks:
        if t.net in length:
            length[t.net] += track_length(t)
    lengths = s.get("net_length_mm")
    if isinstance(lengths, dict):
        for name in sorted(changed):
            if name in lengths:
                lengths[name] = _q(length[name])
        s["total_length_mm"] = _q(sum(lengths.values()))
    for name, row in (s.get("rules") or {}).items():
        if name not in changed or row.get("length_mm") is None or "neckdown_pads" not in row:
            continue
        rule = result.rules[name]
        vias = sum(1 for v in result.vias if v.net == name)
        row["length_mm"] = _q(length[name] + vias * float(rule.via_length_mm or 0.0))
        if row["neckdown_pads"] and rule.neckdown_width_mm is not None:
            neck = float(rule.neckdown_width_mm)
            row["neckdown_length_mm"] = _q(sum(track_length(t) for t in result.tracks if t.net == name and t.width_mm == neck))
    for name, row in (s.get("plane_nets") or {}).items():
        if name in changed and "length_mm" in row:
            row["length_mm"] = _q(length[name])
