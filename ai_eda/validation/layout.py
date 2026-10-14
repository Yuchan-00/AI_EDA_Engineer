"""IR-geometry checks of the board: ``pcb.routing.connectivity`` / ``pcb.routing.clearance`` and ``pcb.silk.*``.

Invariant: these results judge the *IR geometry* - the tracks, vias and
silkscreen texts in ``ir.pcb``, the placements, and the pad / silk geometry
read from the KiCad footprints on disk (``ctx.tools["kicad_library"]``,
never model memory) - and nothing else. They are not ERC / DRC and every
message says so:
KiCad's verdict on the compiled board comes only from
:mod:`ai_eda.tools.kicad.cli` (CLAUDE.md invariant 3), and whether the
fab's limits are grounded is ``mfg.capability``'s question. The geometry
here is computed independently of the router (:mod:`ai_eda.tools.routing.maze`)
on purpose: a check that reused the router's own obstacle model would
inherit its mistakes.

Both checks are conservative, so a claim is real:

* **connectivity** - two copper items of a net are *connected* only when
  their copper overlaps: track-track on one layer when the segment distance
  is below the sum of the half widths; track-pad when the segment comes
  closer than ``width/2 + r`` to the pad centre, ``r`` being the radius of
  the pad's **inscribed** circle (inside any convex pad), on a layer the pad
  is on; via-track / via-pad likewise with the via radius on a shared layer.
  Union-find over the items: a net with two or more pads whose pads do not
  all end in one set is a FAIL row naming the pads left out; a track or via
  naming a net the IR does not have, or a layer ``ir.pcb.layers`` does not
  list, is a FAIL row, and so is a track or via that cannot be copper at
  all (a non-finite coordinate, a width / drill / diameter that is not
  positive - the compiler refuses the same items): it joins nothing and is
  compared with nothing. Pads that repeat one number inside a footprint
  (split thermal / mounting pads) are one logical pad, as KiCad treats them:
  copper reaching any of them reaches the pad. A net whose pads a copper
  pour (``ir.pcb.zones``) may join is NOT_VERIFIED, never FAIL: whether the
  fill reaches a pad is decided by KiCad's fill and DRC, not by the polygon.
  A net whose pours all lie on inner layers - a plane, whose pads the router
  joins by a via each (routing.maze 0.4, a board with keep-outs or RF
  blocks) - says so: connected only through a plane fill the IR does not
  measure. A copper set of such a net that no pour can reach at all (no via
  in it, no pad or track on a pour layer: an SMD pad on ``F.Cu`` without a
  via can never meet an ``In1.Cu`` plane) is a FAIL row naming its pads.
  Nets with fewer than two pads have nothing to connect (NOT_APPLICABLE
  rows).
* **clearance** - judged only against ``ir.pcb.manufacturing.min_clearance_mm``
  (any provenance: a limit is a limit); without one the result is
  NOT_VERIFIED, because the router's own clearance is a parameter, not a
  rule. Every IR copper item (track, via) is compared with every item of a
  *different* net - tracks, vias and pads, a net-less pad counting as
  foreign to everything - on a shared layer; the copper distance is the
  centreline / centre distance minus the half widths, pads taken as their
  **bounding** box. Copper of two nets that overlaps or touches is a short
  and a FAIL row whatever the limit (a 0 mm limit does not make a short
  legal). Pad-to-pad distances are footprint / placement geometry, not
  routing, and are left to DRC (the result says so); so are zone fills,
  which KiCad computes around the copper already there. Tracks or vias
  outside the outline, and via holes closer to it than
  ``min_hole_to_edge_mm`` when that limit exists, are FAIL rows whatever the
  clearance limit. Without IR copper there is nothing to compare
  (NOT_APPLICABLE, never a vacuous PASS).

A via is modelled as a **through** via: the compiler writes it without a via
type, which KiCad reads as spanning every copper layer, so its copper is on
every layer of ``ir.pcb.layers`` (the two names in ``Via.layers`` are only
checked against that list), and a track of another net on an inner layer
under it is a short.

Pad geometry (board frame, mm, Y down, as :mod:`ai_eda.tools.kicad.geometry`
defines it): the centre from ``pad_center``, the box exact for pads at
multiples of 90 degrees and the circumscribed square otherwise, the
inscribed radius ``min(w, h)/2``; a through-hole pad or one listing
``*.Cu`` is on every copper layer of the board, an SMD pad on the copper
layers it lists (mirrored on the bottom side). Only the convex shapes
(``circle`` / ``rect`` / ``oval`` / ``roundrect``) lie inside their
``(size)`` box. A ``custom`` pad is read as the boxes of
:func:`ai_eda.tools.kicad.geometry.custom_pad_parts` - its anchor and one box
around each copper primitive, the extent the router and every other user
reads -, one item per box under the pad's label (one logical pad): clearance
and ``pcb.keepout`` compare every box (an outer bound, like a round pad's
box); a part with a bounding disc (``PadPart.disc``: a circle anchor, a
``gr_circle`` grown by half its stroke - a ring's disc covers its hole) is
measured clear by copper at least the limit from that disc, however close it
comes to the box (a ring's box corner); copper closer than the limit only to
the box / disc that merely bounds its part's copper (a ring, a line's box -
the part's known copper, the disc of radius ``r_in`` around its centre,
keeping the limit) is a NOT_VERIFIED pair under
``details["custom_pad_bounds"]``, never a FAIL, and ``pcb.keepout`` FAILs
only on a part's known copper (an exact box, the disc of ``r_in``);
connectivity joins copper to the pad only through a part known to be copper
around its centre (the anchor's inscribed circle, an exact filled rectangle,
a filled disc), and a net whose pads are left apart while its copper meets a
custom pad inside a merely bounding box (and its bounding disc, when the
part has one: a ring, a line's box) is a NOT_VERIFIED row naming the pad
(``bounded_pads``), never a FAIL: that contact is neither measured nor
excluded. A ``trapezoid``'s
``rect_delta`` reaches beyond the box and the library reader does not keep
it, and a custom pad with a primitive the reader did not read is not bounded
either: such a pad's copper is unknown, like every pad of a component
without a placement or footprint, or whose footprint is not on disk.
Neither result is then ever
PASS (a check that skipped that copper would claim more than it measured):
connectivity makes each net holding an unknown pad a NOT_VERIFIED row naming
it (never the "no such pad" FAIL), still judges every other net - its union
joins only the net's own copper, so copper of another net cannot join its
pads, and an open between known pads is a measured FAIL - and is FAIL when
such a net is, NOT_VERIFIED naming the unknown pads otherwise (a placement-only
board whose router refused a pad FAILs on its open nets, like any other
unrouted board); clearance stays NOT_APPLICABLE without IR copper
(pad-to-pad is DRC's anyway), FAILs on a malformed item, an outline row or a
violation between known items, and is NOT_VERIFIED naming the unknown pads
otherwise. Only a missing KiCad library makes both NOT_VERIFIED for the whole
board. Every distance is rounded
to KiCad's 1e-6 mm resolution before it is compared, and the derived
numbers stay in ``details``, never in the IR.

Silkscreen (:class:`SilkscreenValidator`, tool ``pcb.silk``) - three checks
on what the compiled board's silk layers will carry: every IR
:class:`~ai_eda.ir.SilkText` on ``F.SilkS`` / ``B.SilkS``, the library
default ``Reference`` of every footprint without a designed reference text
(where the compiler leaves it), and every footprint's library silk graphics
and visible silk texts, all read as shapes by
:mod:`ai_eda.tools.silkscreen.geometry` (texts as the documented
conservative text-box *estimate*, not KiCad's font metrics). References on
``F.Fab`` / ``B.Fab`` are not silk and are only counted.

* ``pcb.silk.clearance`` - every IR silk text and every library-default
  ``Reference`` keeps :data:`SILK_TO_PAD_MM` (0.15) from every pad's copper
  on its side and lies inside the outline inset by :data:`SILK_TO_EDGE_MM`
  (0.3) - the placer's own margins, stated in the message, not a fab rule;
  every library silk graphic must not touch pad copper and must lie inside
  the outline. A library item (graphic or the footprint's own visible
  ``fp_text``) closer than the margins is listed under
  ``details["below_margin"]``, not failed: it is the footprint's own design
  and the design cannot move it - KiCad 10's ``BarrelJack_Horizontal`` has a
  silk line 0.09 mm from its pad 1, ``AMASS_XT60PW-M`` its ``+`` / ``-``
  0.12 mm from its pads. A library text whose *estimated* box meets pad
  copper or leaves the outline is not a violation either way (the estimate
  is conservative by design; a real glyph is narrower): it is listed under
  ``details["library_text_estimates"]`` and makes the result NOT_VERIFIED -
  KiCad's ``silk_over_copper`` decides. So does a silk item that meets a
  custom pad's shape that only *bounds* its copper (a ring's disc, a line's
  box: ``PadCopper.bound``), or a designed / default text within its margin
  of one - listed under ``details["custom_pad_bounds"]``, never a FAIL: the
  copper inside that bound is not known. A bound is an outer limit of the
  copper, so a library silk graphic clear of it touches no copper of that
  part (judged; at least that far away, listed under ``below_margin`` when
  closer than the text margin). FAIL on a violation; PASS only when
  every item was checked (a footprint that cannot be read, a ``trapezoid``
  pad or a custom pad with an unread primitive - copper unknown -, or a
  missing outline make it
  NOT_VERIFIED). Tracks are not compared (silk over mask-covered copper is
  normal) and neither are vias (the compiled board tents them).
* ``pcb.silk.overlap`` - silk texts pairwise, and silk texts against the
  footprints' silk graphics, on one side: FAIL when two touch or overlap -
  except a footprint's own library text against that same footprint's own
  library silk (``WS2812B-Mini``'s pin ``1`` beside its outline: the
  footprint's design, judged only by the estimated box), which is listed
  under ``details["library_own_overlaps"]``. A pair with an IR text, a
  library-default ``Reference`` or another footprint's silk still FAILs.
* ``pcb.silk.size`` - every silk text's height against
  ``ir.pcb.manufacturing.min_silk_text_height_mm`` and every text stroke and
  library silk line width against ``min_silk_line_width_mm`` (any
  provenance: a limit is a limit, like ``min_clearance_mm`` above; the
  message names the provenance). A key without a limit in the IR is a
  NOT_VERIFIED row naming it; the result is the worst row.

Keep-outs (:class:`KeepoutValidator`, tool ``pcb.keepout``, only for a
board with ``ir.pcb.keepouts`` or RF blocks with a region or a shield can):
nothing a keep-out forbids lies in it (footprints, pads, tracks, vias,
zones - each with its named exceptions), every block part lies inside its
region and every part under a can inside the can's fence. The compiled
KiCad rule areas cannot carry the exceptions (they are cut out of the rule
area's polygon), so only this check knows them.

An IR silk text the compiler refuses
(:func:`~ai_eda.tools.silkscreen.geometry.silk_text_problems`: a layer the
board lacks, a non-finite number, a reference to no component ...) FAILs all
three. KiCad's DRC ``silk_over_copper`` / ``silk_overlap`` /
``text_height`` stay kicad-cli's (``kicad.drc``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, PCBDesign, Track, ValidationResult, ValidationStatus, Via
from ai_eda.tools.kicad.geometry import _q, custom_pad_parts, pad_angle, pad_center, pad_layers
from ai_eda.tools.kicad.library import KicadLibrary, LibraryLookupError
from ai_eda.tools.silkscreen.geometry import (
    FAB_LAYERS,
    SILK_LAYERS,
    PadCopper,
    Shape,
    footprint_silk,
    inside_box,
    ir_text_box,
    pad_copper,
    shape_distance,
    side_of_layer,
    silk_text_problems,
)
from ai_eda.tools.silkscreen.place import SilkParams
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry

__all__ = [
    "TOOL_ID",
    "TOOL_VERSION",
    "CONNECTIVITY_CHECK",
    "CLEARANCE_CHECK",
    "RoutingValidator",
    "connectivity_rows",
    "clearance_rows",
    "SILK_TOOL_ID",
    "SILK_TOOL_VERSION",
    "SILK_CLEARANCE_CHECK",
    "SILK_OVERLAP_CHECK",
    "SILK_SIZE_CHECK",
    "SILK_TO_PAD_MM",
    "SILK_TO_EDGE_MM",
    "SilkscreenValidator",
    "KEEPOUT_CHECK",
    "KEEPOUT_TOOL_ID",
    "KEEPOUT_TOOL_VERSION",
    "KeepoutValidator",
]

TOOL_ID = "pcb.routing"
TOOL_VERSION = "0.1"
CONNECTIVITY_CHECK = "pcb.routing.connectivity"
CLEARANCE_CHECK = "pcb.routing.clearance"
#: details["kind"] of both results: an IR geometry check, never ERC / DRC
KIND = "ir_geometry"
NOT_DRC = "IR geometry, not DRC"
#: what the clearance check leaves to DRC
NOT_COMPARED = ["pad-to-pad (footprint / placement geometry; DRC judges it)"]
#: added to ``not_compared`` when ``ir.pcb.zones`` is not empty
ZONES_NOT_COMPARED = "zone fills (KiCad fills a zone around the copper already there; DRC judges the fill)"
#: pad shapes whose copper lies inside the ``(size)`` box; any other shape is unknown copper (module docstring)
CONVEX_PAD_SHAPES = frozenset({"circle", "rect", "oval", "roundrect"})

Point = tuple[float, float]
Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- items


@dataclass(frozen=True, slots=True)
class _PadItem:
    label: str  # "R1.2"
    net: str | None  # None: a pad no net names (an obstacle for everyone)
    cx: float
    cy: float
    hw: float  # half extents of the bounding box
    hh: float
    r_in: float  # radius of the inscribed circle
    layers: frozenset[str]
    #: the box is the pad's copper exactly (a ``rect`` pad at a multiple of 90 degrees)
    exact: bool = False
    #: copper reaching the disc of radius ``r_in`` around the centre reaches the pad; ``False`` for a part of a ``custom`` pad whose
    #: box only bounds its copper (a ring, a line, an arc ...): nothing inside that box is known to be copper
    joins: bool = True
    #: the part of a ``custom`` pad this item is (``"anchor rect"``, ``"gr_poly[0]"`` ...); ``None`` for every other pad
    part: str | None = None
    #: ``(x, y, r)`` of a disc that bounds the part's copper more tightly than its box (``PadPart.disc``: a circle anchor, a
    #: ``gr_circle`` grown by half its stroke - a ring's disc covers its hole); ``None`` otherwise
    disc: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class _TrackItem:
    label: str  # "track[3:N]"
    net: str
    layer: str
    a: Point
    b: Point
    w: float


@dataclass(frozen=True, slots=True)
class _ViaItem:
    label: str  # "via[0:N]"
    net: str
    x: float
    y: float
    d: float
    drill: float
    layers: frozenset[str]  # every copper layer of the board: a through via (module docstring)
    named: tuple[str, str]  # the two names ``Via.layers`` carries, checked against ``ir.pcb.layers``


_Item = _PadItem | _TrackItem | _ViaItem


# --------------------------------------------------------------------------- exact geometry (mm)


def _seg_point_distance(a: Point, b: Point, p: Point) -> float:
    """Distance from ``p`` to the segment ``a``-``b``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _orient(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _seg_seg_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    """Distance between the segments ``a``-``b`` and ``c``-``d`` (0 when they cross or touch)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return 0.0  # a proper crossing; every touching / collinear case is an endpoint at distance 0 below
    return min(_seg_point_distance(a, b, c), _seg_point_distance(a, b, d), _seg_point_distance(c, d, a), _seg_point_distance(c, d, b))


def _point_box_distance(p: Point, box: Box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - p[0], 0.0, p[0] - x2), max(y1 - p[1], 0.0, p[1] - y2))


def _inside_box(p: Point, box: Box) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


def _seg_box_distance(a: Point, b: Point, box: Box) -> float:
    """Distance from the segment ``a``-``b`` to the axis-aligned box (0 when it enters or crosses the box)."""
    if _inside_box(a, box) or _inside_box(b, box):
        return 0.0
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return min(_seg_seg_distance(a, b, corners[i], corners[(i + 1) % 4]) for i in range(4))


def _pad_box(pad: _PadItem) -> Box:
    return (pad.cx - pad.hw, pad.cy - pad.hh, pad.cx + pad.hw, pad.cy + pad.hh)


def _item_box(item: _Item) -> Box:
    """Axis-aligned box around the item's copper (for the pair prefilter)."""
    if isinstance(item, _PadItem):
        return _pad_box(item)
    if isinstance(item, _ViaItem):
        r = item.d / 2.0
        return (item.x - r, item.y - r, item.x + r, item.y + r)
    r = item.w / 2.0
    return (min(item.a[0], item.b[0]) - r, min(item.a[1], item.b[1]) - r, max(item.a[0], item.b[0]) + r, max(item.a[1], item.b[1]) + r)


def _item_layers(item: _Item) -> frozenset[str]:
    return frozenset({item.layer}) if isinstance(item, _TrackItem) else item.layers


def _copper_distance(a: _Item, b: _Item) -> float:
    """Distance between the copper of two items (0 when they overlap); pads as their bounding box, vias as circles."""
    if isinstance(a, _PadItem):
        a, b = b, a
    if isinstance(a, _TrackItem):
        if isinstance(b, _TrackItem):
            return _seg_seg_distance(a.a, a.b, b.a, b.b) - (a.w + b.w) / 2.0
        if isinstance(b, _ViaItem):
            return _seg_point_distance(a.a, a.b, (b.x, b.y)) - a.w / 2.0 - b.d / 2.0
        return _seg_box_distance(a.a, a.b, _pad_box(b)) - a.w / 2.0
    assert isinstance(a, _ViaItem)
    if isinstance(b, _TrackItem):
        return _seg_point_distance(b.a, b.b, (a.x, a.y)) - b.w / 2.0 - a.d / 2.0
    if isinstance(b, _ViaItem):
        return math.hypot(a.x - b.x, a.y - b.y) - (a.d + b.d) / 2.0
    return _point_box_distance((a.x, a.y), _pad_box(b)) - a.d / 2.0


def _connected(a: _Item, b: _Item) -> bool:
    """Whether the copper of two items of one net overlaps (conservative: touching is not connected)."""
    if isinstance(a, _PadItem):
        a, b = b, a
    if isinstance(a, _PadItem):
        return False  # pad-to-pad: not IR copper
    if isinstance(b, _PadItem) and not b.joins:
        return False  # a custom pad's bounding part: nothing inside its box is known to be copper
    shared = _item_layers(a) & _item_layers(b)
    if not shared:
        return False
    if isinstance(a, _TrackItem):
        if isinstance(b, _TrackItem):
            return _q(_seg_seg_distance(a.a, a.b, b.a, b.b)) < _q((a.w + b.w) / 2.0)
        if isinstance(b, _ViaItem):
            return _q(_seg_point_distance(a.a, a.b, (b.x, b.y))) < _q(a.w / 2.0 + b.d / 2.0)
        return _q(_seg_point_distance(a.a, a.b, (b.cx, b.cy))) < _q(a.w / 2.0 + b.r_in)
    assert isinstance(a, _ViaItem)
    if isinstance(b, _TrackItem):
        return _q(_seg_point_distance(b.a, b.b, (a.x, a.y))) < _q(b.w / 2.0 + a.d / 2.0)
    if isinstance(b, _ViaItem):
        return _q(math.hypot(a.x - b.x, a.y - b.y)) < _q((a.d + b.d) / 2.0)
    return _q(math.hypot(a.x - b.cx, a.y - b.cy)) < _q(a.d / 2.0 + b.r_in)


# --------------------------------------------------------------------------- loading the board


class _Board:
    """The pads (from the library), tracks and vias of ``ir.pcb`` as geometry items; ``unknown`` says why pads are missing."""

    def __init__(self, ir: CircuitIR, library: KicadLibrary) -> None:
        pcb: PCBDesign = ir.pcb  # the validator checked it exists
        self.copper_layers = [layer.name for layer in pcb.layers]
        self.pin_net: dict[tuple[str, str], str] = {}
        for net in ir.nets:
            for pin in net.pins:
                self.pin_net[(pin.component_ref, pin.pin_number)] = net.name
        self.pads: list[_PadItem] = []
        self.pad_labels: set[str] = set()
        self.unknown: list[str] = []
        #: the components none of whose pads is known (no placement / footprint, or a footprint that cannot be read) and the
        #: ``REF.NUMBER`` labels of single pads whose copper is unknown (a trapezoid, a custom pad with an unread primitive): a net holding
        #: one is never judged
        self.unknown_refs: set[str] = set()
        self.unknown_pads: set[str] = set()
        #: the ``REF.NUMBER`` labels of the custom pads read as parts (module docstring)
        self.custom_labels: set[str] = set()
        for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
            placement = pcb.placement(comp.ref)
            if placement is None:
                self.unknown.append(f"{comp.ref} has no placement")
                self.unknown_refs.add(comp.ref)
                continue
            if comp.footprint is None:
                self.unknown.append(f"{comp.ref} has no footprint")
                self.unknown_refs.add(comp.ref)
                continue
            try:
                fp = library.load_footprint(comp.footprint)
            except LibraryLookupError:
                self.unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref} was not found in a KiCad library")
                self.unknown_refs.add(comp.ref)
                continue
            except CompileError as e:  # LibraryFormatError: a malformed .kicad_mod is a reason, not a crash
                self.unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref}: {e}")
                self.unknown_refs.add(comp.ref)
                continue
            for pad in fp.pads:
                if pad.shape == "custom":
                    self._custom(comp.ref, f"{comp.footprint.library}:{comp.footprint.name}", placement, pad)
                    continue
                if pad.shape not in CONVEX_PAD_SHAPES:
                    self.unknown.append(
                        f"pad {comp.ref}.{pad.number or '(unnumbered)'} of footprint {comp.footprint.library}:{comp.footprint.name} has shape "
                        f"{pad.shape!r}, whose copper is not bounded by its (size) box (a trapezoid's rect_delta is not read)"
                    )
                    if pad.number:
                        self.unknown_pads.add(f"{comp.ref}.{pad.number}")
                    continue
                cx, cy = pad_center(placement, pad)
                angle = pad_angle(placement, pad)
                if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):
                    w, h = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
                    hw, hh = w / 2.0, h / 2.0
                else:
                    hw = hh = math.hypot(pad.size_w, pad.size_h) / 2.0
                layers = pad_layers(placement, pad)
                if pad.pad_type in ("thru_hole", "np_thru_hole") or "*.Cu" in layers:
                    copper = frozenset(self.copper_layers)
                else:
                    copper = frozenset(name for name in layers if name.endswith(".Cu"))
                net = self.pin_net.get((comp.ref, pad.number)) if pad.number else None
                label = f"{comp.ref}.{pad.number}" if pad.number else f"{comp.ref}.(unnumbered)"
                if pad.number:
                    self.pad_labels.add(label)
                exact = pad.shape == "rect" and math.isclose(angle % 90.0, 0.0, abs_tol=1e-9)
                self.pads.append(_PadItem(label, net, cx, cy, hw, hh, min(pad.size_w, pad.size_h) / 2.0, copper, exact))
        self.track_total, self.via_total = len(pcb.tracks), len(pcb.vias)
        #: FAIL rows for tracks / vias that cannot be copper (non-finite coordinate, non-positive width / drill / diameter);
        #: such an item is in neither ``tracks`` nor ``vias``: it joins nothing and is compared with nothing
        self.malformed: list[dict[str, Any]] = []
        self.tracks: list[_TrackItem] = []
        self.vias: list[_ViaItem] = []
        for i, t in enumerate(pcb.tracks):
            item = _track_item(i, t)
            why = _malformed_track(item)
            if why is None:
                self.tracks.append(item)
            else:
                self.malformed.append({"item": item.label, "status": str(ValidationStatus.FAIL), "message": f"{item.label} {why}"})
        for i, v in enumerate(pcb.vias):
            item = _via_item(i, v, self.copper_layers)
            why = _malformed_via(item)
            if why is None:
                self.vias.append(item)
            else:
                self.malformed.append({"item": item.label, "status": str(ValidationStatus.FAIL), "message": f"{item.label} {why}"})
        self.zones = [(f"zone[{i}:{z.net}]", z.net, z.layer) for i, z in enumerate(pcb.zones)]

    def _custom(self, ref: str, lib_id: str, placement: Any, pad: Any) -> None:
        """A ``custom`` pad as one item per :func:`~ai_eda.tools.kicad.geometry.custom_pad_parts` box (module docstring); unknown when
        a primitive was not read."""
        label = f"{ref}.{pad.number}" if pad.number else f"{ref}.(unnumbered)"
        try:
            parts = custom_pad_parts(placement, pad)
        except CompileError as e:
            self.unknown.append(f"pad {label} of footprint {lib_id}: {e}")
            if pad.number:
                self.unknown_pads.add(label)
            return
        layers = pad_layers(placement, pad)
        if pad.pad_type in ("thru_hole", "np_thru_hole") or "*.Cu" in layers:
            copper = frozenset(self.copper_layers)
        else:
            copper = frozenset(name for name in layers if name.endswith(".Cu"))
        net = self.pin_net.get((ref, pad.number)) if pad.number else None
        if pad.number:
            self.pad_labels.add(label)
        self.custom_labels.add(label)
        for part in parts:
            cx, cy = part.center
            self.pads.append(_PadItem(label, net, cx, cy, part.box.width / 2.0, part.box.height / 2.0, part.inscribed_r, copper, part.exact,
                                      part.inscribed_r > 0.0, part.what, part.disc))

    def is_unknown(self, ref: str, pin: str) -> bool:
        """Whether the copper of pin ``ref.pin`` is unknown (its component unread, or the pad itself a trapezoid or a custom pad with a
        primitive the library reader did not read)."""
        return ref in self.unknown_refs or f"{ref}.{pin}" in self.unknown_pads


def _track_item(i: int, t: Track) -> _TrackItem:
    return _TrackItem(f"track[{i}:{t.net}]", t.net, t.layer, (float(t.start[0]), float(t.start[1])), (float(t.end[0]), float(t.end[1])), float(t.width_mm))


def _via_item(i: int, v: Via, copper_layers: list[str]) -> _ViaItem:
    return _ViaItem(f"via[{i}:{v.net}]", v.net, float(v.x_mm), float(v.y_mm), float(v.diameter_mm), float(v.drill_mm), frozenset(copper_layers), (str(v.layers[0]), str(v.layers[1])))


def _malformed_track(t: _TrackItem) -> str | None:
    """Why the track cannot be copper (the compiler refuses the same), or ``None``."""
    if not all(math.isfinite(v) for v in (*t.a, *t.b, t.w)):
        return f"has a non-finite coordinate or width (start {t.a}, end {t.b}, width {t.w}): its copper cannot be measured"
    if t.w <= 0.0:
        return f"has non-positive width {t.w:g} mm: no copper (the compiler refuses it too)"
    return None


def _malformed_via(v: _ViaItem) -> str | None:
    """Why the via cannot be copper (the compiler refuses the same), or ``None``."""
    if not all(math.isfinite(x) for x in (v.x, v.y, v.d, v.drill)):
        return f"has a non-finite coordinate or size (at ({v.x}, {v.y}), diameter {v.d}, drill {v.drill}): its copper cannot be measured"
    if v.drill <= 0.0 or v.d <= v.drill:
        return f"needs diameter {v.d:g} mm > drill {v.drill:g} mm > 0: no copper (the compiler refuses it too)"
    return None


# --------------------------------------------------------------------------- connectivity


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def connectivity_rows(ir: CircuitIR, board: _Board) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(net rows, item rows)``: one row per IR net, and one per track / via on an unknown net or layer."""
    net_names = {net.name for net in ir.nets}
    layer_names = set(board.copper_layers)
    item_rows: list[dict[str, Any]] = []
    for t in board.tracks:
        if t.net not in net_names:
            item_rows.append({"item": t.label, "status": str(ValidationStatus.FAIL), "message": f"{t.label} names net {t.net!r}, which is not in the IR"})
        if t.layer not in layer_names:
            item_rows.append({"item": t.label, "status": str(ValidationStatus.FAIL), "message": f"{t.label} is on layer {t.layer!r}, which ir.pcb.layers does not list"})
    for v in board.vias:
        if v.net not in net_names:
            item_rows.append({"item": v.label, "status": str(ValidationStatus.FAIL), "message": f"{v.label} names net {v.net!r}, which is not in the IR"})
        for layer in v.named:
            if layer not in layer_names:
                item_rows.append({"item": v.label, "status": str(ValidationStatus.FAIL), "message": f"{v.label} is on layer {layer!r}, which ir.pcb.layers does not list"})
    item_rows.extend(board.malformed)
    rows: list[dict[str, Any]] = []
    for net in ir.nets:
        pads = [p for p in board.pads if p.net == net.name]
        labels = list(dict.fromkeys(p.label for p in pads))  # same-numbered pads of one footprint: one logical pad
        unknown = list(dict.fromkeys(f"{pin.component_ref}.{pin.pin_number}" for pin in net.pins if board.is_unknown(pin.component_ref, pin.pin_number)))
        missing = [f"{pin.component_ref}.{pin.pin_number}" for pin in net.pins
                   if f"{pin.component_ref}.{pin.pin_number}" not in board.pad_labels and not board.is_unknown(pin.component_ref, pin.pin_number)]
        items: list[_Item] = [*pads, *(t for t in board.tracks if t.net == net.name), *(v for v in board.vias if v.net == net.name)]
        zones = [label for label, zone_net, _ in board.zones if zone_net == net.name]
        row: dict[str, Any] = {
            "net": net.name,
            "pads": labels,
            "tracks": sum(1 for t in board.tracks if t.net == net.name),
            "vias": sum(1 for v in board.vias if v.net == net.name),
        }
        if zones:
            row["zones"] = zones
        if missing:
            row.update(status=str(ValidationStatus.FAIL), unconnected=missing, message=f"{', '.join(missing)}: no such pad in the placed footprint")
            rows.append(row)
            continue
        if unknown:  # copper this check cannot see may join (or be) the net's pads: never judged, never a FAIL for the pad it cannot read
            row.update(status=str(ValidationStatus.NOT_VERIFIED), unknown_pads=unknown,
                       message=f"the copper of {', '.join(unknown)} is unknown (see details['unknown']): the net is not judged")
            rows.append(row)
            continue
        if len(labels) < 2:
            row.update(status=str(ValidationStatus.NOT_APPLICABLE), message=f"{len(labels)} pad(s): nothing to connect")
            rows.append(row)
            continue
        uf = _UnionFind(len(items))
        first_of: dict[str, int] = {}
        for k, p in enumerate(pads):
            if p.label in first_of:
                uf.union(k, first_of[p.label])  # internally connected in KiCad: one logical pad
            else:
                first_of[p.label] = k
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                if _connected(a, items[j]):
                    uf.union(i, j)
        root = uf.find(0)
        left_out = list(dict.fromkeys(p.label for k, p in enumerate(pads) if uf.find(k) != root))
        maybe = _maybe_joined(pads, items, uf) if left_out else []
        cut_off = _unreachable_by_pour(items, uf, len(pads), [layer for _, zone_net, layer in board.zones if zone_net == net.name]) if left_out and zones else []
        if maybe:  # copper meets a custom pad's bound where its copper is not known: the open is neither measured nor excluded
            row.update(
                status=str(ValidationStatus.NOT_VERIFIED), unconnected=left_out, bounded_pads=maybe,
                message=(f"{', '.join(left_out)} not joined to {labels[0]} through known copper; copper of the net reaches the bounding box of "
                         f"custom pad(s) {', '.join(maybe)} where the pad's copper is not known (a ring, a line, an arc ... is bounded by its box): "
                         "whether it touches the pad is not measured here"),
            )
        elif cut_off:
            row.update(
                status=str(ValidationStatus.FAIL), unconnected=cut_off,
                message=(f"{', '.join(cut_off)} not connected: no track / via of the net reaches them and their copper is on no layer of "
                         f"its pour {', '.join(zones)} (only a via or a through-hole barrel reaches an inner plane)"),
            )
        elif left_out and zones and all(_INNER_RE.match(layer) for _, zone_net, layer in board.zones if zone_net == net.name):
            row.update(
                status=str(ValidationStatus.NOT_VERIFIED), unconnected=left_out, plane=True,
                message=(f"{', '.join(left_out)} not joined to {labels[0]} by tracks; each reaches the plane {', '.join(zones)} by a via or a "
                         "through-hole barrel: connected only through a plane fill the IR does not measure (KiCad's fill and DRC decide)"),
            )
        elif left_out and zones:
            row.update(
                status=str(ValidationStatus.NOT_VERIFIED), unconnected=left_out,
                message=f"{', '.join(left_out)} not joined to {labels[0]} by tracks / vias; whether the copper pour {', '.join(zones)} reaches them is decided by KiCad's fill and DRC, not by the polygon",
            )
        elif left_out:
            row.update(status=str(ValidationStatus.FAIL), unconnected=left_out, message=f"{', '.join(left_out)} not connected to {labels[0]}")
        else:
            row.update(status=str(ValidationStatus.PASS), message=f"{len(labels)} pad(s) in one copper set")
        rows.append(row)
    return rows, item_rows


#: an inner copper layer (a plane layer of a 4-layer board)
_INNER_RE = re.compile(r"^In[1-9][0-9]*\.Cu$")


def _disc_distance(a: _Item, pad: _PadItem) -> float | None:
    """Distance from the copper of a track / via ``a`` to the disc that bounds ``pad``'s copper (``None``: the part has no disc)."""
    if pad.disc is None or isinstance(a, _PadItem):
        return None
    x, y, r = pad.disc
    if isinstance(a, _TrackItem):
        return _seg_point_distance(a.a, a.b, (x, y)) - a.w / 2.0 - r
    return math.hypot(a.x - x, a.y - y) - a.d / 2.0 - r


def _maybe_joined(pads: list[_PadItem], items: list[_Item], uf: _UnionFind) -> list[str]:
    """The custom pads a track / via of the net meets only inside a bounding part (``joins`` False) while not joined to that pad through
    known copper: the contact is neither measured nor excluded (module docstring). Copper that meets the part's box but not its bounding
    disc (a ring's box corner) cannot meet its copper: no contact."""
    out: list[str] = []
    for k, p in enumerate(pads):
        if p.joins or p.label in out:
            continue
        for j in range(len(pads), len(items)):
            c = items[j]
            if uf.find(j) == uf.find(k) or not (_item_layers(c) & p.layers):
                continue
            disc = _disc_distance(c, p)
            if _q(_copper_distance(c, p)) <= 0.0 and (disc is None or _q(disc) <= 0.0):
                out.append(p.label)
                break
    return out


def _unreachable_by_pour(items: list[_Item], uf: _UnionFind, n_pads: int, zone_layers: list[str]) -> list[str]:
    """The pads of every copper set that no pour of the net can reach: none of its items is a via, a pad or a track on a pour layer."""
    layers = set(zone_layers)
    reach: dict[int, bool] = {}
    for k, item in enumerate(items):
        root = uf.find(k)
        ok = isinstance(item, _ViaItem) or bool(_item_layers(item) & layers)
        reach[root] = reach.get(root, False) or ok
    if len({uf.find(k) for k in range(n_pads)}) < 2:
        return []
    return list(dict.fromkeys(items[k].label for k in range(n_pads) if not reach[uf.find(k)]))


# --------------------------------------------------------------------------- clearance


def _bound_only(a: _Item, pad: _PadItem, limit: float) -> bool:
    """Whether ``a`` comes within ``limit`` of ``pad`` only through a custom pad's box (or bounding disc) that merely bounds its copper:
    the part's known copper (the disc of radius ``r_in`` around its centre; none when ``r_in`` is 0) keeps the limit, so the violation is
    not measured."""
    if pad.part is None or pad.exact:
        return False
    if pad.r_in <= 0.0:
        return True
    if isinstance(a, _TrackItem):
        known = _seg_point_distance(a.a, a.b, (pad.cx, pad.cy)) - a.w / 2.0 - pad.r_in
    else:
        known = math.hypot(a.x - pad.cx, a.y - pad.cy) - a.d / 2.0 - pad.r_in
    return _q(known) >= limit


def clearance_rows(board: _Board, limit: float) -> tuple[int, list[dict[str, Any]]]:
    """``(pairs compared, violation rows)``: every IR copper item against every item of another net on a shared layer.

    A custom pad part with a bounding disc (``_PadItem.disc``) is measured to that disc, which bounds its copper more tightly than its
    box: a pair at least ``limit`` from it is clear, however close it comes to the part's box (a ring's box corner). A pair closer than
    ``limit`` only to the box / disc that merely bounds the part's copper (:func:`_bound_only`) is a NOT_VERIFIED row (``bound`` True),
    never a FAIL: the copper inside that bound is not known."""
    copper: list[_Item] = [*board.tracks, *board.vias]
    others: list[_Item] = [*board.tracks, *board.vias, *board.pads]
    boxes = {id(item): _item_box(item) for item in others}
    compared = 0
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for a in copper:
        box_a = boxes[id(a)]
        for b in others:
            if a is b or (id(b), id(a)) in seen:
                continue
            if a.net == b.net and b.net is not None:
                continue
            shared = _item_layers(a) & _item_layers(b)
            if not shared:
                continue
            seen.add((id(a), id(b)))
            compared += 1
            box_b = boxes[id(b)]
            # prefilter: copper boxes further apart than the limit on either axis cannot violate it
            if box_a[0] - limit > box_b[2] or box_b[0] - limit > box_a[2] or box_a[1] - limit > box_b[3] or box_b[1] - limit > box_a[3]:
                continue
            raw = _q(_copper_distance(a, b))
            disc = _disc_distance(a, b) if isinstance(b, _PadItem) else None
            if disc is not None:  # the disc bounds the part's copper more tightly than its box (a ring's box corner is not copper)
                raw = max(raw, _q(disc))
            d = max(raw, 0.0)
            if d < limit or raw <= 0.0:  # overlapping or touching copper of two nets is a short whatever the limit
                where = ",".join(sorted(shared))
                if isinstance(b, _PadItem) and _bound_only(a, b, limit):
                    bound = "disc" if disc is not None else "box"
                    rows.append({"a": a.label, "b": b.label, "layer": where, "distance_mm": d, "limit_mm": limit, "bound": True,
                                 "status": str(ValidationStatus.NOT_VERIFIED),
                                 "message": (f"{a.label} vs {b.label} on {where}: {d:g} mm from the {bound} that bounds the copper of custom pad "
                                             f"{b.label} ({b.part}; its copper inside that {bound} is not known)")})
                    continue
                message = f"{a.label} vs {b.label} on {where}: {d:g} mm < {limit:g} mm" if d < limit else f"{a.label} vs {b.label} on {where}: copper overlaps (a short, whatever the {limit:g} mm limit)"
                rows.append({"a": a.label, "b": b.label, "layer": where, "distance_mm": d, "limit_mm": limit, "status": str(ValidationStatus.FAIL), "message": message})
    return compared, rows


def _outline_rows(pcb: PCBDesign, board: _Board) -> list[dict[str, Any]]:
    """Tracks / vias whose copper leaves the outline, and via holes closer to it than ``min_hole_to_edge_mm``."""
    o = pcb.outline
    rows: list[dict[str, Any]] = []
    if o is None:
        return rows
    x1, y1, x2, y2 = float(o.origin_x_mm), float(o.origin_y_mm), float(o.origin_x_mm + o.width_mm), float(o.origin_y_mm + o.height_mm)
    for item in [*board.tracks, *board.vias]:
        bx1, by1, bx2, by2 = _item_box(item)
        if _q(bx1) < _q(x1) or _q(by1) < _q(y1) or _q(bx2) > _q(x2) or _q(by2) > _q(y2):
            rows.append({"item": item.label, "status": str(ValidationStatus.FAIL), "message": f"{item.label} copper leaves the {o.width_mm:g} x {o.height_mm:g} mm outline at ({o.origin_x_mm:g}, {o.origin_y_mm:g})"})
    hole_limit = pcb.manufacturing.min_hole_to_edge_mm
    if hole_limit is not None:
        lim = float(hole_limit.value)
        for v in board.vias:
            edge = _q(min(v.x - x1, x2 - v.x, v.y - y1, y2 - v.y) - v.drill / 2.0)
            if edge < lim:
                rows.append({"item": v.label, "status": str(ValidationStatus.FAIL), "distance_mm": edge, "limit_mm": lim,
                             "message": f"{v.label} hole edge {edge:g} mm from the outline < min_hole_to_edge_mm {lim:g}"})
    return rows


# --------------------------------------------------------------------------- the validator


class RoutingValidator(Validator):
    id = TOOL_ID
    description = "IR copper connects every net's pads and keeps the recorded clearance from foreign copper (IR geometry, not DRC)"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        pcb = ir.pcb
        if pcb is None:
            return self._both(ValidationStatus.NOT_APPLICABLE, "no ir.pcb: nothing routed to check")
        if not pcb.placements:
            return self._both(ValidationStatus.NOT_APPLICABLE, "ir.pcb has no placements: no pads to connect")
        if not ir.nets:
            return self._both(ValidationStatus.NOT_APPLICABLE, "no nets: nothing to connect")
        library = ctx.tools.get("kicad_library")
        if not isinstance(library, KicadLibrary):
            return self._both(ValidationStatus.NOT_VERIFIED, "no KiCad library: pad geometry unknown")
        board = _Board(ir, library)
        return [self._connectivity(ir, board), self._clearance(pcb, board)]

    @staticmethod
    def _unknown_part(board: _Board) -> str:
        return "pad geometry unknown: " + "; ".join(board.unknown)

    @staticmethod
    def _malformed_part(board: _Board) -> str:
        return f"{len(board.malformed)} track(s) / via(s) that cannot be copper: " + "; ".join(r["message"] for r in board.malformed)

    def _both(self, status: ValidationStatus, message: str, **details: Any) -> list[ValidationResult]:
        return [self._result(check, status, f"{message} ({NOT_DRC})", **details) for check in (CONNECTIVITY_CHECK, CLEARANCE_CHECK)]

    @staticmethod
    def _result(check_id: str, status: ValidationStatus, message: str, **details: Any) -> ValidationResult:
        return ValidationResult(check_id=check_id, status=status, message=message, tool=TOOL_ID, tool_version=TOOL_VERSION, details={"kind": KIND, **details})

    def _connectivity(self, ir: CircuitIR, board: _Board) -> ValidationResult:
        rows, item_rows = connectivity_rows(ir, board)
        failed = [r["net"] for r in rows if r["status"] == str(ValidationStatus.FAIL)]
        unread = [r["net"] for r in rows if r.get("unknown_pads")]
        bounded = [r["net"] for r in rows if r.get("bounded_pads")]
        pour = [r["net"] for r in rows if r["status"] == str(ValidationStatus.NOT_VERIFIED) and not r.get("unknown_pads") and not r.get("bounded_pads")]
        connected = [r["net"] for r in rows if r["status"] == str(ValidationStatus.PASS)]
        details = {"nets": rows, "items": item_rows, "tracks": board.track_total, "vias": board.via_total, "zones": len(board.zones), "unknown": board.unknown}
        if board.custom_labels:  # read as the boxes of their anchor and primitives (module docstring)
            details["custom_pads"] = sorted(board.custom_labels, key=natural_ref_key)
        # a FAIL row is a net whose every pad is known and whose own copper leaves a pad out: copper this check cannot read belongs to
        # other nets (a union joins only the net's own items), so it cannot join these pads - a measured open, whatever else is unknown
        if failed or item_rows:
            parts = []
            if failed:
                parts.append(f"{len(failed)} net(s) not connected through IR copper: " + "; ".join(f"{r['net']}: {r['message']}" for r in rows if r["net"] in failed))
            stray = [r for r in item_rows if r not in board.malformed]
            if stray:
                parts.append(f"{len(stray)} track(s) / via(s) on a net or layer the IR does not have: " + "; ".join(r["message"] for r in stray))
            if board.malformed:
                parts.append(self._malformed_part(board))
            if board.unknown:
                parts.append(f"{len(unread)} net(s) not judged because of pads whose geometry is unknown ({'; '.join(board.unknown)})")
            if bounded:
                parts.append(f"{len(bounded)} net(s) not judged where copper meets a custom pad's bounding box only ({', '.join(bounded)})")
            return self._result(CONNECTIVITY_CHECK, ValidationStatus.FAIL, "; ".join(parts) + f" ({NOT_DRC})", **details)
        if board.unknown:  # never a PASS while a pad's copper is unknown: it may join or cut what the check measured
            tail = f"; {len(unread)} net(s) holding such a pad not judged" + (f", {len(connected)} other net(s) connected through IR copper" if connected else "")
            return self._result(CONNECTIVITY_CHECK, ValidationStatus.NOT_VERIFIED, self._unknown_part(board) + tail + f" ({NOT_DRC})", **details)
        if bounded:
            return self._result(
                CONNECTIVITY_CHECK, ValidationStatus.NOT_VERIFIED,
                f"{len(bounded)} net(s) whose copper meets a custom pad only inside the box that bounds its copper, where the copper itself is not "
                "known: " + "; ".join(f"{r['net']}: {r['message']}" for r in rows if r["net"] in bounded) + f" ({NOT_DRC})", **details,
            )
        if pour:
            return self._result(
                CONNECTIVITY_CHECK, ValidationStatus.NOT_VERIFIED,
                f"{len(pour)} net(s) joined only by a copper pour, which this check cannot judge: " + "; ".join(f"{r['net']}: {r['message']}" for r in rows if r["net"] in pour)
                + f" ({NOT_DRC})", **details,
            )
        if not connected:
            return self._result(CONNECTIVITY_CHECK, ValidationStatus.NOT_APPLICABLE, f"no net with two or more pads: nothing to connect ({NOT_DRC})", **details)
        return self._result(
            CONNECTIVITY_CHECK, ValidationStatus.PASS,
            f"{len(connected)} net(s) connected through IR copper ({len(board.tracks)} track(s), {len(board.vias)} via(s)) (IR geometry check, not DRC)", **details,
        )

    def _clearance(self, pcb: PCBDesign, board: _Board) -> ValidationResult:
        limit = pcb.manufacturing.min_clearance_mm
        outline_rows = _outline_rows(pcb, board)
        hole = pcb.manufacturing.min_hole_to_edge_mm
        details: dict[str, Any] = {
            "limit_mm": None if limit is None else float(limit.value),
            "hole_to_edge_limit_mm": None if hole is None else float(hole.value),
            "outline": outline_rows,
            "malformed": board.malformed,
            "not_compared": [*NOT_COMPARED, ZONES_NOT_COMPARED] if board.zones else NOT_COMPARED,
            "tracks": board.track_total,
            "vias": board.via_total,
        }
        if board.unknown:
            details["unknown"] = board.unknown
        if board.custom_labels:
            details["custom_pads"] = sorted(board.custom_labels, key=natural_ref_key)
        if board.malformed:  # copper the compiler refuses: FAIL whatever the limit, nothing else is claimed about the board
            return self._result(CLEARANCE_CHECK, ValidationStatus.FAIL, self._malformed_part(board) + f" ({NOT_DRC})", pairs_compared=0, violations=[], **details)
        if not board.tracks and not board.vias:  # pad-to-pad is DRC's even when every pad is known: nothing here to compare
            return self._result(CLEARANCE_CHECK, ValidationStatus.NOT_APPLICABLE, f"no IR copper (tracks / vias) to compare ({NOT_DRC})", pairs_compared=0, violations=[], **details)
        if limit is None:
            if outline_rows:
                return self._result(CLEARANCE_CHECK, ValidationStatus.FAIL, f"{len(outline_rows)} track(s) / via(s) outside the outline or too close to it: "
                                    + "; ".join(r["message"] for r in outline_rows) + f" ({NOT_DRC})", pairs_compared=0, violations=[], **details)
            if board.unknown:
                return self._result(CLEARANCE_CHECK, ValidationStatus.NOT_VERIFIED, self._unknown_part(board) + f" ({NOT_DRC})", pairs_compared=0, violations=[], **details)
            return self._result(
                CLEARANCE_CHECK, ValidationStatus.NOT_VERIFIED,
                f"no clearance limit in ir.pcb.manufacturing; the router's own clearance is a parameter, not a rule; DRC with the fab rules decides ({NOT_DRC})",
                pairs_compared=0, violations=[], **details,
            )
        lim = float(limit.value)
        compared, rows = clearance_rows(board, lim)
        violations = [r for r in rows if r["status"] == str(ValidationStatus.FAIL)]
        bounded = [r for r in rows if r.get("bound")]
        details.update(pairs_compared=compared, violations=violations)
        if bounded:
            details["custom_pad_bounds"] = bounded[:_MAX_ROWS]
        if violations or outline_rows:
            parts = []
            if violations:
                parts.append(f"{len(violations)} copper pair(s) of different nets closer than {lim:g} mm: " + "; ".join(r["message"] for r in violations))
            if outline_rows:
                parts.append(f"{len(outline_rows)} track(s) / via(s) outside the outline or too close to it: " + "; ".join(r["message"] for r in outline_rows))
            return self._result(CLEARANCE_CHECK, ValidationStatus.FAIL, "; ".join(parts) + f" ({NOT_DRC})", **details)
        if board.unknown:  # the copper nobody read may be closer than the limit (or a short): no PASS
            return self._result(CLEARANCE_CHECK, ValidationStatus.NOT_VERIFIED,
                                self._unknown_part(board) + f"; {compared} copper pair(s) of known items keep >= {lim:g} mm ({NOT_DRC})", **details)
        if bounded:  # closer than the limit only to a box that merely bounds a custom pad's copper: not measured, no PASS, no FAIL
            return self._result(
                CLEARANCE_CHECK, ValidationStatus.NOT_VERIFIED,
                f"{len(bounded)} copper pair(s) closer than {lim:g} mm only to the box / disc that bounds a custom pad's copper, where the copper is not "
                "known: " + "; ".join(r["message"] for r in bounded[:5]) + (" ..." if len(bounded) > 5 else "") + f" ({NOT_DRC})", **details,
            )
        edge = "" if pcb.outline is None else f", {len(board.tracks)} track(s) / {len(board.vias)} via(s) inside the outline"
        if pcb.outline is None:
            return self._result(
                CLEARANCE_CHECK, ValidationStatus.NOT_VERIFIED,
                f"{compared} copper pair(s) of different nets keep >= {lim:g} mm on a shared layer, but ir.pcb has no outline: edge distances not judged ({NOT_DRC})", **details,
            )
        return self._result(
            CLEARANCE_CHECK, ValidationStatus.PASS,
            f"{compared} copper pair(s) of different nets keep >= {lim:g} mm on a shared layer{edge} "
            f"(IR geometry with pads as bounding boxes, pad-to-pad left to DRC; not DRC)", **details,
        )


default_registry.register(RoutingValidator())


# --------------------------------------------------------------------------- silkscreen

SILK_TOOL_ID = "pcb.silk"
SILK_TOOL_VERSION = "0.1"
SILK_CLEARANCE_CHECK = "pcb.silk.clearance"
SILK_OVERLAP_CHECK = "pcb.silk.overlap"
SILK_SIZE_CHECK = "pcb.silk.size"
#: the margins the check holds silk texts to: the silkscreen placer's own (not a fab rule; the message says so)
SILK_TO_PAD_MM = SilkParams().silk_to_pad_mm
SILK_TO_EDGE_MM = SilkParams().silk_to_edge_mm
SILK_SIZE_KEYS = ("min_silk_text_height_mm", "min_silk_line_width_mm")
#: rows listed per result at most (the counts are complete)
_MAX_ROWS = 200


@dataclass(frozen=True, slots=True)
class _SilkEntry:
    label: str
    side: str  # "F" / "B"
    kind: str  # "text" | "graphic"
    source: str  # "ir" | "library default" | "library"
    shape: Shape
    box: tuple[float, float, float, float]
    width: float | None  # text stroke / graphic stroke width (None: a fill without a stroke)
    size: float | None  # text height
    ref: str | None = None  # the footprint a library item belongs to / the component an IR text names


class _SilkBoard:
    """Everything the compiled board's silk layers will carry, the pad copper, and what could not be read."""

    def __init__(self, ir: CircuitIR, library: KicadLibrary | None) -> None:
        pcb: PCBDesign = ir.pcb  # the validator checked it exists
        o = pcb.outline
        self.outline = None if o is None else (o.origin_x_mm, o.origin_y_mm, o.origin_x_mm + o.width_mm, o.origin_y_mm + o.height_mm)
        self.entries: list[_SilkEntry] = []
        self.pads: list[PadCopper] = []
        self.unknown: list[str] = []  # footprints / silk constructs that could not be read
        self.unknown_copper: list[str] = []  # pads whose copper is not modelled
        self.malformed: list[dict[str, Any]] = []
        self.on_fab: list[str] = []
        sides = {c.ref: (p.side if (p := pcb.placement(c.ref)) is not None else None) for c in ir.components}
        problems = silk_text_problems(pcb.silkscreen, sides)
        designed: set[str] = set()
        for i, t in enumerate(pcb.silkscreen):
            label = f"silk[{i}:{t.kind}:{t.text}]"
            if i in problems:
                self.malformed.append({"item": label, "status": str(ValidationStatus.FAIL), "message": f"{label}: {problems[i]} (the compiler refuses it)"})
                continue
            if t.kind == "reference" and t.component_ref is not None:
                designed.add(t.component_ref)
            if t.layer in FAB_LAYERS:
                self.on_fab.append(t.component_ref or t.text)
                continue
            shape = ir_text_box(t)
            self.entries.append(_SilkEntry(label, side_of_layer(t.layer), "text", "ir", shape, shape.bbox(), t.thickness_mm, t.size_mm, t.component_ref))
        if not pcb.placements:
            return
        if library is None:
            self.unknown.append("no KiCad library: footprint silk and pad geometry unknown")
            return
        for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
            placement = pcb.placement(comp.ref)
            if placement is None:
                self.unknown.append(f"{comp.ref} has no placement")
                continue
            if comp.footprint is None:
                self.unknown.append(f"{comp.ref} has no footprint")
                continue
            try:
                fp = library.load_footprint(comp.footprint)
            except LibraryLookupError:
                self.unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref} was not found in a KiCad library")
                continue
            except CompileError as e:
                self.unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref}: {e}")
                continue
            for pad in pad_copper(comp.ref, fp, placement):
                if pad.shape is None:
                    self.unknown_copper.append(f"pad {pad.label} of {fp.lib_id} has shape {pad.pad_shape!r}, whose copper is not read (a trapezoid's rect_delta, or a custom pad's unread primitive)")
                else:
                    self.pads.append(pad)
            values = {"Value": comp.value, "Datasheet": (comp.datasheet.url or "") if comp.datasheet is not None else ""}
            silk = footprint_silk(comp.ref, fp, placement, values)
            for item in silk.items:
                if item.shape is None:
                    self.unknown.append(f"{item.label} of {fp.lib_id}: {item.why}")
                    continue
                self.entries.append(_SilkEntry(item.label, side_of_layer(item.layer), item.kind, "library", item.shape, item.shape.bbox(), item.width, item.size, comp.ref))
            ref = silk.reference
            if comp.ref not in designed and ref is not None and not ref.hidden and ref.layer in SILK_LAYERS:
                shape = ref.box()
                self.entries.append(_SilkEntry(f"{comp.ref}:Reference (library position)", side_of_layer(ref.layer), "text", "library default", shape, shape.bbox(), ref.thickness, ref.size, comp.ref))


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float) -> bool:
    return not (a[0] - margin > b[2] or b[0] - margin > a[2] or a[1] - margin > b[3] or b[1] - margin > a[3])


class SilkscreenValidator(Validator):
    """``pcb.silk.clearance`` / ``pcb.silk.overlap`` / ``pcb.silk.size`` (module docstring): IR geometry, not DRC."""

    id = SILK_TOOL_ID
    description = "silkscreen texts and footprint silk keep clear of pad copper, the board edge and each other, at the fab's minimum sizes (IR geometry, not DRC)"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        pcb = ir.pcb
        if pcb is None:
            return self._all(ValidationStatus.NOT_APPLICABLE, "no ir.pcb: no silkscreen to check")
        if not pcb.placements and not pcb.silkscreen:
            return self._all(ValidationStatus.NOT_APPLICABLE, "ir.pcb has no placements and no silkscreen texts: no silkscreen to check")
        library = ctx.tools.get("kicad_library")
        board = _SilkBoard(ir, library if isinstance(library, KicadLibrary) else None)
        return [self._clearance(board), self._overlap(board), self._size(pcb, board)]

    def _all(self, status: ValidationStatus, message: str) -> list[ValidationResult]:
        return [self._result(check, status, f"{message} ({NOT_DRC})") for check in (SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK)]

    @staticmethod
    def _result(check_id: str, status: ValidationStatus, message: str, **details: Any) -> ValidationResult:
        return ValidationResult(check_id=check_id, status=status, message=message, tool=SILK_TOOL_ID, tool_version=SILK_TOOL_VERSION, details={"kind": KIND, **details})

    @staticmethod
    def _common(board: _SilkBoard) -> dict[str, Any]:
        return {
            "texts": sum(1 for e in board.entries if e.kind == "text"),
            "graphics": sum(1 for e in board.entries if e.kind == "graphic"),
            "designed_texts": sum(1 for e in board.entries if e.source == "ir"),
            "library_default_references": [e.label.split(":")[0] for e in board.entries if e.source == "library default"],
            "on_fab": board.on_fab,
            "malformed": board.malformed,
            "unknown": board.unknown,
            "estimate": "text boxes are a conservative estimate (characters x size x 0.9 + stroke by size x 1.2 + stroke), not KiCad's font metrics",
        }

    def _verdict(self, check_id: str, board: _SilkBoard, rows: list[dict[str, Any]], what: str, passed: str, unknown: list[str], **details: Any) -> ValidationResult:
        details = {**self._common(board), "violations": rows[:_MAX_ROWS], "violation_count": len(rows), **details}
        if board.malformed or rows:
            parts = []
            if board.malformed:
                parts.append(f"{len(board.malformed)} silk text(s) the compiler refuses: " + "; ".join(r["message"] for r in board.malformed[:5]))
            if rows:
                parts.append(f"{len(rows)} {what}: " + "; ".join(r["message"] for r in rows[:5]) + (" ..." if len(rows) > 5 else ""))
            return self._result(check_id, ValidationStatus.FAIL, "; ".join(parts) + f" ({NOT_DRC})", **details)
        if unknown:
            return self._result(check_id, ValidationStatus.NOT_VERIFIED, "not every silk item could be checked: " + "; ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "") + f" ({NOT_DRC})", **details)
        if not board.entries:
            return self._result(check_id, ValidationStatus.NOT_APPLICABLE, f"nothing on the silk layers ({len(board.on_fab)} reference(s) on the fab layer) ({NOT_DRC})", **details)
        return self._result(check_id, ValidationStatus.PASS, f"{passed} (IR geometry with estimated text boxes, not DRC)", **details)

    def _clearance(self, board: _SilkBoard) -> ValidationResult:
        rows: list[dict[str, Any]] = []
        below: list[dict[str, Any]] = []
        estimated: list[str] = []  # library texts whose over-estimated box meets copper / leaves the outline: not a proof either way
        bounded: list[str] = []  # items meeting a custom pad's box that only bounds its copper (a ring's square ...): not a proof either way
        compared = 0
        for e in board.entries:
            margined = e.kind == "text" and e.source != "library"  # designed / default-position texts: the placer's margins
            library_text = e.kind == "text" and e.source == "library"
            limit = SILK_TO_PAD_MM if margined else 0.0
            for pad in board.pads:
                if e.side not in pad.sides or pad.shape is None:
                    continue
                compared += 1
                if not _near(e.box, pad.shape.bbox(), SILK_TO_PAD_MM):
                    continue
                d = shape_distance(e.shape, pad.shape)
                # a bound (a ring's disc, a line's box) is an outer limit of the copper: an item clear of it is clear of the copper, so only an
                # item meeting it - or a margined text within its margin of it - is not measured
                if pad.bound and (d <= 0.0 or (margined and d < _q(limit))):
                    bounded.append(f"{e.label} is {max(d, 0.0):g} mm from the shape that bounds the copper of custom pad {pad.label} (its copper inside "
                                   "that bound is not known: KiCad's silk_over_copper decides)")
                elif library_text and d <= 0.0:
                    estimated.append(f"the estimated box of library text {e.label} meets pad {pad.label} copper (the estimate is conservative; KiCad's silk_over_copper decides)")
                elif d <= 0.0 or (margined and d < _q(limit)):
                    what = "overlaps" if d <= 0.0 else f"is {d:g} mm from"
                    rows.append({"a": e.label, "b": f"pad {pad.label}", "distance_mm": d, "limit_mm": limit, "status": str(ValidationStatus.FAIL),
                                 "message": f"{e.label} {what} pad {pad.label} copper" + ("" if d <= 0.0 else f" (< {limit:g} mm)")})
                elif d < _q(SILK_TO_PAD_MM):
                    at = f"at least {d:g} mm from pad {pad.label} (measured to the shape that bounds its copper)" if pad.bound else f"{d:g} mm from pad {pad.label}"
                    below.append({"a": e.label, "b": f"pad {pad.label}", "distance_mm": d, "message": f"library silk {e.label} is {at} (the footprint's own design; below the {SILK_TO_PAD_MM:g} mm text margin)"})
            if board.outline is not None:
                inset = (board.outline[0] + SILK_TO_EDGE_MM, board.outline[1] + SILK_TO_EDGE_MM, board.outline[2] - SILK_TO_EDGE_MM, board.outline[3] - SILK_TO_EDGE_MM)
                if margined and not inside_box(e.shape, inset):
                    rows.append({"a": e.label, "b": "board edge", "status": str(ValidationStatus.FAIL), "message": f"{e.label} is not inside the outline inset by {SILK_TO_EDGE_MM:g} mm"})
                elif library_text and not inside_box(e.shape, board.outline):
                    estimated.append(f"the estimated box of library text {e.label} leaves the outline (the estimate is conservative; KiCad's DRC decides)")
                elif not margined and not library_text and not inside_box(e.shape, board.outline):
                    rows.append({"a": e.label, "b": "board edge", "status": str(ValidationStatus.FAIL), "message": f"{e.label} leaves the outline"})
                elif not margined and not inside_box(e.shape, inset):
                    below.append({"a": e.label, "b": "board edge", "message": f"library silk {e.label} is within {SILK_TO_EDGE_MM:g} mm of the outline (the footprint's own design)"})
        unknown = [*board.unknown, *board.unknown_copper, *estimated, *bounded] + ([] if board.outline is not None else ["ir.pcb has no outline: edge distances not judged"])
        passed = (
            f"{sum(1 for e in board.entries if e.kind == 'text' and e.source != 'library')} silk text(s) keep >= {SILK_TO_PAD_MM:g} mm from pad copper and >= {SILK_TO_EDGE_MM:g} mm inside the outline "
            f"(the silkscreen placer's margins, not a fab rule), {sum(1 for e in board.entries if e.source == 'library')} footprint silk graphic(s) / library text(s) touch no pad copper and stay inside the outline"
            + (f"; {len(below)} library item(s) closer than the text margins, listed" if below else "")
        )
        return self._verdict(SILK_CLEARANCE_CHECK, board, rows, "silk item(s) too close to pad copper or the board edge", passed, unknown,
                             pairs_compared=compared, silk_to_pad_mm=SILK_TO_PAD_MM, silk_to_edge_mm=SILK_TO_EDGE_MM, below_margin=below[:_MAX_ROWS],
                             library_text_estimates=estimated[:_MAX_ROWS],
                             not_compared=["tracks (silk over mask-covered copper is normal)", "vias (tented by the compiled board)"],
                             **({"custom_pad_bounds": bounded[:_MAX_ROWS]} if bounded else {}))

    def _overlap(self, board: _SilkBoard) -> ValidationResult:
        rows: list[dict[str, Any]] = []
        own: list[dict[str, Any]] = []  # a footprint's library text against its own library silk: its own design, listed
        texts = [e for e in board.entries if e.kind == "text"]
        graphics = [e for e in board.entries if e.kind == "graphic"]
        compared = 0
        for i, a in enumerate(texts):
            for b in [*texts[i + 1:], *graphics]:
                if a.side != b.side:
                    continue
                compared += 1
                if not _near(a.box, b.box, 0.0):
                    continue
                if shape_distance(a.shape, b.shape) <= 0.0:
                    if a.source == b.source == "library" and a.ref is not None and a.ref == b.ref:
                        own.append({"a": a.label, "b": b.label, "message": f"library text {a.label} meets {b.label} of the same footprint (the footprint's own design, judged by the estimated text box)"})
                        continue
                    rows.append({"a": a.label, "b": b.label, "status": str(ValidationStatus.FAIL), "message": f"{a.label} touches or overlaps {b.label}"})
        passed = (f"{len(texts)} silk text(s) overlap no other silk text and no footprint silk graphic ({compared} pair(s) compared)"
                  + (f"; {len(own)} library text(s) meeting their own footprint's silk, listed" if own else ""))
        return self._verdict(SILK_OVERLAP_CHECK, board, rows, "silk overlap(s)", passed, list(board.unknown), pairs_compared=compared, library_own_overlaps=own[:_MAX_ROWS])

    def _size(self, pcb: PCBDesign, board: _SilkBoard) -> ValidationResult:
        mfg = pcb.manufacturing
        rows: list[dict[str, Any]] = []
        limits: dict[str, Any] = {}
        missing: list[str] = []
        for key in SILK_SIZE_KEYS:
            traced = getattr(mfg, key)
            if traced is None:
                missing.append(f"no {key} in ir.pcb.manufacturing (ground it with --fab-capability): not judged")
                limits[key] = None
                continue
            lim = float(traced.value)
            limits[key] = {"value_mm": lim, "provenance": str(traced.provenance.kind)}
            for e in board.entries:
                if key == "min_silk_text_height_mm" and e.kind == "text" and e.size is not None and _q(e.size) < _q(lim):
                    rows.append({"item": e.label, "key": key, "value_mm": e.size, "limit_mm": lim, "status": str(ValidationStatus.FAIL),
                                 "message": f"{e.label} text height {e.size:g} mm < {key} {lim:g}"})
                if key == "min_silk_line_width_mm" and e.width is not None and _q(e.width) < _q(lim):
                    what = "stroke" if e.kind == "text" else "line width"
                    rows.append({"item": e.label, "key": key, "value_mm": e.width, "limit_mm": lim, "status": str(ValidationStatus.FAIL),
                                 "message": f"{e.label} {what} {e.width:g} mm < {key} {lim:g}"})
        grounded = ", ".join(f"{k} {v['value_mm']:g} mm ({v['provenance']})" for k, v in limits.items() if v is not None)
        passed = f"every silk text and line meets {grounded}"
        missing = missing if board.entries else []  # nothing on the silk layers: nothing to size, whatever the limits
        result = self._verdict(SILK_SIZE_CHECK, board, rows, "silk item(s) below the fab's minimum size", passed, [*missing, *board.unknown], limits=limits)
        if result.status is ValidationStatus.NOT_VERIFIED and missing and not board.unknown:
            checked = f"; checked: {grounded}" if grounded else ""
            result.message = "; ".join(missing) + checked + f" ({NOT_DRC})"
        return result


default_registry.register(SilkscreenValidator())


# --------------------------------------------------------------------------- keep-outs, block regions, shield-can fences

KEEPOUT_TOOL_ID = "pcb.keepout"
KEEPOUT_TOOL_VERSION = "0.1"
KEEPOUT_CHECK = "pcb.keepout"
#: what the keep-out check says about KiCad
KEEPOUT_NOT_DRC = "IR geometry, not DRC; KiCad's DRC reads the compiled rule areas, which know no exception"


class KeepoutValidator(Validator):
    """``pcb.keepout``: nothing a keep-out forbids lies in it, every block part lies in its region, every part under a can inside its fence.

    Applies only to a board with keep-outs (``ir.pcb.keepouts``) or RF blocks
    with a region or a shield can (``ir.rf.blocks``); every other design gets
    no row. IR geometry (the keep-out areas through
    :mod:`ai_eda.tools.keepout`, the pads as in the routing checks above),
    never DRC:

    * **footprints** - a placed part's extent (courtyard union pads, the box
      the placers measure) sharing area with the keep-out on its side's
      copper layer, unless its ref is in ``allowed_refs``;
    * **pads** - a pad on a covered copper layer whose copper shares area with
      it (a ``rect`` pad at a multiple of 90 degrees by its box, any other
      shape by its inscribed circle for a FAIL; a pad whose bounding box
      reaches the area but whose inscribed circle does not is NOT_VERIFIED),
      unless its ref is in ``allowed_refs`` or its net in ``allowed_nets``;
    * **tracks** - a track on a covered layer whose centreline comes closer
      than half its width; **vias** - a through via (on every copper layer)
      whose disc reaches the area when any board layer is covered;
      **zones** - a zone on a covered layer whose polygon shares area with it
      (two non-convex polygons: NOT_VERIFIED, not computed) - each unless
      its net is in ``allowed_nets``;
    * **regions** - every ref of a block with a region has its placed extent
      inside the region; **fences** - every other part of a block with a
      shield can lies strictly inside the can's fence (read from its pads,
      :func:`~ai_eda.tools.placement.rf_floorplan.placed_fence_box`; the
      least distance to the fence is recorded as ``ring_mm``).

    FAIL on any violation; NOT_VERIFIED when something could not be judged
    (no KiCad library, a footprint not on disk, a pad whose copper is not
    bounded by its box, a fence not read, an unplaced block part); PASS only
    when every item was compared; NOT_APPLICABLE when nothing is placed or
    routed yet.
    """

    id = KEEPOUT_CHECK
    description = "Keep-outs, RF block regions and shield-can fences on the IR geometry (not DRC)"

    def applies_to(self, ir: CircuitIR) -> bool:
        from ai_eda.tools.keepout import keepouts_of
        from ai_eda.tools.placement.rf_floorplan import region_box, rf_blocks

        return bool(keepouts_of(ir.pcb)) or any(region_box(b) is not None or getattr(b, "shield_ref", None) for b in rf_blocks(ir))

    def _result(self, status: ValidationStatus, message: str, **details: Any) -> ValidationResult:
        return ValidationResult(check_id=self.id, status=status, message=f"{message} ({KEEPOUT_NOT_DRC})", tool=KEEPOUT_TOOL_ID,
                                tool_version=KEEPOUT_TOOL_VERSION, details={"kind": KIND, **details})

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        from ai_eda.tools import keepout as ko_geom
        from ai_eda.tools.kicad.geometry import footprint_bbox
        from ai_eda.tools.placement.rf_floorplan import placed_fence_box, region_box, rf_blocks

        pcb = ir.pcb
        if pcb is None or not (pcb.placements or pcb.tracks or pcb.vias or pcb.zones):
            return [self._result(ValidationStatus.NOT_APPLICABLE, "nothing placed or routed yet: no footprint, pad or copper to compare")]
        library = ctx.tools.get("kicad_library")
        unknown: list[str] = []
        parts: dict[str, tuple[Any, Any]] = {}  # ref -> (placement, footprint)
        board: _Board | None = None
        if isinstance(library, KicadLibrary):
            for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
                placement = pcb.placement(comp.ref)
                if placement is None or comp.footprint is None:
                    continue
                try:
                    parts[comp.ref] = (placement, library.load_footprint(comp.footprint))
                except LibraryLookupError:
                    unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref} was not found in a KiCad library")
                except CompileError as e:
                    unknown.append(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref}: {e}")
            board = _Board(ir, library)
            unknown.extend(u for u in board.unknown if u not in unknown and "has no placement" not in u)
        else:
            unknown.append("no KiCad library: footprint extents and pads unknown")
        copper_layers = [layer.name for layer in pcb.layers]
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        def row(status: ValidationStatus, what: str, message: str, **extra: Any) -> None:
            rows.append({"item": what, "status": str(status), "message": message, **extra})

        for k in ko_geom.keepouts_of(pcb):
            kid = ko_geom.keepout_id(k)
            pts = ko_geom.area_points(k)
            bbox = ko_geom.area_bbox(pts)
            refs, nets = set(ko_geom.allowed_refs(k)), set(ko_geom.allowed_nets(k))
            layers = ko_geom.covered_layers(k, copper_layers)
            info: dict[str, Any] = {"id": kid, "layers": layers, "forbids": [i for i in ko_geom.KEEPOUT_ITEMS if ko_geom.forbids(k, i)],
                                    "allowed_refs": sorted(refs), "allowed_nets": sorted(nets), "reason": str(getattr(k, "reason", "") or ""),
                                    "compared": 0, "exempt": []}
            summaries.append(info)
            at = f"keep-out {kid}"

            def near(box: tuple[float, float, float, float], bbox: tuple[float, float, float, float] = bbox) -> bool:
                return box[0] < bbox[2] and bbox[0] < box[2] and box[1] < bbox[3] and bbox[1] < box[3]

            if ko_geom.forbids(k, "footprints"):
                for ref, (placement, fp) in parts.items():
                    side = "F.Cu" if placement.side == BoardSide.TOP else "B.Cu"
                    if side not in layers:
                        continue
                    e = footprint_bbox(placement, fp)
                    if e is None:
                        continue
                    info["compared"] += 1
                    box = (e.x1, e.y1, e.x2, e.y2)
                    if near(box) and ko_geom.box_area_overlap(box, pts) > 0.0:
                        if ref in refs:
                            info["exempt"].append(f"footprint {ref}")
                        else:
                            row(ValidationStatus.FAIL, f"footprint {ref}", f"{at} forbids footprints; the extent of {ref} lies in it", keepout=kid)
            if ko_geom.forbids(k, "pads") and board is not None:
                for pad in board.pads:
                    if not (pad.layers & set(layers)):
                        continue
                    info["compared"] += 1
                    box = _pad_box(pad)
                    if not near(box) or ko_geom.box_area_overlap(box, pts) <= 0.0:
                        continue
                    ref = pad.label.split(".", 1)[0]
                    if ref in refs or (pad.net is not None and pad.net in nets):
                        info["exempt"].append(f"pad {pad.label}")
                    elif pad.exact or ko_geom.point_area_distance((pad.cx, pad.cy), pts) < pad.r_in:
                        row(ValidationStatus.FAIL, f"pad {pad.label}", f"{at} forbids pads; the copper of {pad.label} lies in it", keepout=kid)
                    else:
                        row(ValidationStatus.NOT_VERIFIED, f"pad {pad.label}",
                            f"{at} forbids pads; the bounding box of {pad.label} reaches it but its copper shape is not judged here", keepout=kid)
            if ko_geom.forbids(k, "tracks") and board is not None:
                for t in board.tracks:
                    if t.layer not in layers:
                        continue
                    info["compared"] += 1
                    if ko_geom.segment_area_distance(t.a, t.b, pts) < t.w / 2.0 - 1e-9:
                        if t.net in nets:
                            info["exempt"].append(t.label)
                        else:
                            row(ValidationStatus.FAIL, t.label, f"{at} forbids tracks on {t.layer}; {t.label} ({t.net}) reaches into it", keepout=kid)
            if ko_geom.forbids(k, "vias") and board is not None and layers:
                for v in board.vias:
                    info["compared"] += 1
                    if ko_geom.point_area_distance((v.x, v.y), pts) < v.d / 2.0 - 1e-9:
                        if v.net in nets:
                            info["exempt"].append(v.label)
                        else:
                            row(ValidationStatus.FAIL, v.label, f"{at} forbids vias; {v.label} ({v.net}) reaches into it", keepout=kid)
            if ko_geom.forbids(k, "zones"):
                for i, z in enumerate(pcb.zones):
                    if z.layer not in layers or len(z.polygon) < 3:
                        continue
                    info["compared"] += 1
                    label = f"zone[{i}:{z.net}]"
                    area = ko_geom.polygon_overlap_area([(float(x), float(y)) for x, y in z.polygon], pts)
                    if area is None:
                        row(ValidationStatus.NOT_VERIFIED, label,
                            f"{at} forbids zones; {label} and the keep-out are both non-convex: their overlap is not computed here", keepout=kid)
                    elif area > 0.0:
                        if z.net in nets:
                            info["exempt"].append(label)
                        else:
                            row(ValidationStatus.FAIL, label, f"{at} forbids zones on {z.layer}; {label} covers {area:.3f} mm^2 of it", keepout=kid)
        regions: list[dict[str, Any]] = []
        for b in rf_blocks(ir):
            bid = str(getattr(b, "id", "?"))
            box = region_box(b)
            members = list(getattr(b, "refs", []) or [])
            if box is not None:
                region: dict[str, Any] = {"block": bid, "region": list(box), "refs": members, "outside": []}
                regions.append(region)
                for ref in members:
                    if ref not in parts:
                        if pcb.placements:
                            row(ValidationStatus.NOT_VERIFIED, f"block {bid}: {ref}", f"{ref} of block {bid} is not placed (or its footprint is unknown)", block=bid)
                        continue
                    e = footprint_bbox(*parts[ref])
                    if e is not None and (e.x1 < box[0] - 1e-6 or e.y1 < box[1] - 1e-6 or e.x2 > box[2] + 1e-6 or e.y2 > box[3] + 1e-6):
                        region["outside"].append(ref)
                        row(ValidationStatus.FAIL, f"block {bid}: {ref}",
                            f"the extent of {ref} ({e.x1:g}, {e.y1:g})-({e.x2:g}, {e.y2:g}) leaves the region of block {bid} "
                            f"({box[0]:g}, {box[1]:g})-({box[2]:g}, {box[3]:g})", block=bid)
            can = getattr(b, "shield_ref", None)
            if can is None:
                continue
            if can not in parts:
                if pcb.placements:
                    row(ValidationStatus.NOT_VERIFIED, f"can {can}", f"the shield can {can} of block {bid} is not placed (or its footprint is unknown)", block=bid)
                continue
            fence = placed_fence_box(*parts[can])
            if fence is None:
                row(ValidationStatus.NOT_VERIFIED, f"can {can}", f"the fence of the shield can {can} could not be read from its pads", block=bid)
                continue
            ring: float | None = None
            for ref in members:
                if ref == can or ref not in parts:
                    continue
                e = footprint_bbox(*parts[ref])
                if e is None:
                    continue
                gap = min(e.x1 - fence.x1, fence.x2 - e.x2, e.y1 - fence.y1, fence.y2 - e.y2)
                ring = gap if ring is None else min(ring, gap)
                if gap <= 1e-6:
                    row(ValidationStatus.FAIL, f"can {can}: {ref}", f"{ref} of block {bid} is not inside the fence of its can {can} (by {gap:.3f} mm)", block=bid)
            regions.append({"block": bid, "can": can, "fence": [fence.x1, fence.y1, fence.x2, fence.y2], "ring_mm": None if ring is None else _q(ring)})
        details: dict[str, Any] = {"keepouts": summaries, "regions": regions, "rows": rows, "unknown": unknown}
        failed = [r for r in rows if r["status"] == str(ValidationStatus.FAIL)]
        open_rows = [r for r in rows if r["status"] == str(ValidationStatus.NOT_VERIFIED)]
        if failed:
            return [self._result(ValidationStatus.FAIL, f"{len(failed)} violation(s): " + "; ".join(r["message"] for r in failed), **details, repair="human")]
        if open_rows or unknown:
            return [self._result(ValidationStatus.NOT_VERIFIED, "not every item could be judged: " + "; ".join([*(r["message"] for r in open_rows), *unknown]),
                                 **details)]
        n_ko, n_reg = len(summaries), sum(1 for r in regions if "region" in r)
        n_can = sum(1 for r in regions if "can" in r)
        exempt = sum(len(s["exempt"]) for s in summaries)
        return [self._result(
            ValidationStatus.PASS,
            f"{n_ko} keep-out(s): nothing they forbid lies in them ({exempt} allowed item(s) named as exceptions); "
            f"{n_reg} block region(s) hold their parts; {n_can} can fence(s) hold their block's parts", **details,
        )]


default_registry.register(KeepoutValidator())
