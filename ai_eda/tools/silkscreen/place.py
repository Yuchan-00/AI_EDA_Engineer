"""Deterministic silkscreen placer (``silkscreen.place`` 0.1): reference designators, connector pin labels, the board title.

Invariant: every text produced here is a function of the IR (components,
nets, placements, outline, project name), the footprints read from a KiCad
library on disk (pads, silk graphics - :mod:`ai_eda.tools.silkscreen.geometry`)
and :class:`SilkParams`; nothing is estimated beyond the documented text-box
estimate, nothing comes from a model. The same IR + library gives the same
texts (tested). It is run by :class:`~ai_eda.agents.pcb.PCBAgent` after
placement and routing, only when ``ir.pcb.silkscreen`` is empty - existing
silkscreen is never replaced - and it only *returns* texts; the agent
proposes them and the orchestrator applies the proposal.

Keep-outs (board frame, mm, per silk side):

* every pad's copper (library geometry, transformed: exact rectangles /
  rounded rectangles / discs / capsules) grown by ``silk_to_pad_mm``
  (0.15) - a pad counts on the side of its copper / mask layers, a through
  hole on both. A ``custom`` / ``trapezoid`` pad's copper is not read by the
  library reader: its footprint's courtyard stands in for it (a footprint
  with such a pad and no courtyard is refused, :class:`CompileError`);
* the board edge: every text lies inside the outline inset by
  ``silk_to_edge_mm`` (0.3);
* every footprint's own silkscreen graphics and visible silk texts (library
  lines, arcs, circles, polygons, texts - so a reference never covers a
  pin-1 or polarity mark) and every text placed before, each kept
  ``text_gap_mm`` (0.1) away;
* **not** tracks: silk over mask-covered copper is normal; **not** vias: the
  compiled board tents every via (``compilers/pcb.py`` ``_setup``: ``(tenting
  (front yes) (back yes))``), so a via is under solder mask like a track.

Placement, in this order (each text is a keep-out for the next):

1. **References**, natural ref order: ten candidates around the footprint's
   courtyard (widened to the pads' box grown by ``silk_to_pad - text_gap``
   where the pads reach beyond it, and itself grown by ``silk_to_pad -
   text_gap`` when it stands in for a ``custom`` / ``trapezoid`` pad's
   copper; the pads' box alone without a courtyard) - above, below, left at 0 then
   90 degrees, right at 0 then 90 degrees, then the four corners (the box
   diagonally outside each courtyard corner) - at 1.0 mm (0.15 mm stroke,
   KiCad's default silk text), then the same ten at 0.8 mm; the first
   candidate whose estimated box touches no keep-out and stays ``text_gap``
   clear of every *other* footprint's courtyard on its side (their pads' box
   without one) wins - a designator inside a neighbour's courtyard reads as
   the neighbour's. Only when none of the twenty is outside every other
   courtyard, the first one that merely touches no keep-out wins and the
   note names the courtyard. A connector that
   will get pin labels tries the side(s) its labels use (step 2) last, so its
   own reference does not take their place. None free: the reference goes to
   ``F.Fab`` / ``B.Fab`` at the courtyard centre (KiCad's convention for a
   reference that does not fit: on the fab drawing, not on the silk) and the
   note names it.
2. **Connector pin labels**: for every component whose library symbol is in a
   ``Connector*`` library, each pad's net name (unnamed pads, ``Net-(...)`` /
   ``unconnected-...`` nets and pads without a net are not labelled) as a
   0.8 mm text beside the pad, outside the courtyard, perpendicular to the pad
   row (a vertical row gets horizontal labels, a horizontal row vertical
   ones), justified to run away from the footprint; a pad off the row's
   centre line goes on its own side (a double-row header's outer sides), the
   pads of a single row all on the one side where more labels fit (a tie: the
   side facing the board centre). A label that collides is skipped and named
   in the note - never overlapped. A net name that is not printable ASCII is
   skipped too (the text-box estimate assumes Latin glyphs).
3. **Title**: ``ir.project.name`` (printable ASCII only; other characters
   dropped with a note - KiCad's stroke font is used and the estimate assumes
   Latin glyphs), 1.5 mm, in the board corner with the most room for it (the
   largest scale of the title's box, anchored at the corner inset by the edge
   rule, that touches no keep-out; ties: top-left, top-right, bottom-left,
   bottom-right), left-justified in the left corners and right-justified in
   the right ones. When no corner is free, the title slides from each corner
   along the top / bottom edge towards the middle in 0.25 mm steps and takes
   the free position nearest to a corner (ties: the corner order); the note
   says how far. No date, no hash (determinism; a hash would be circular).
   No free position: no title, and the note says so.

Every coordinate is rounded to 0.001 mm before its box is tested, so the
texts written are the texts tested. Provenance: ``derived`` /
``silkscreen.place`` with the component / pad / net, the candidate index and
name, the text size and every parameter in ``derived_from``, and the
keep-out rules in the note. Whether KiCad agrees is decided by its DRC
(``silk_over_copper`` / ``silk_overlap``) on the compiled board; the
``pcb.silk.*`` checks (:mod:`ai_eda.validation.layout`) re-measure the IR
geometry independently of this placer's bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, Component, Placement, Provenance, ProvenanceKind, SilkKind, SilkText
from ai_eda.tools.kicad.geometry import _q, courtyard_bbox, pad_copper_center, pads_bbox
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary, LibraryLookupError
from ai_eda.tools.silkscreen.geometry import (
    CONVEX_PAD_SHAPES,
    Box,
    Shape,
    footprint_silk,
    inside_box,
    pad_copper,
    shape_distance,
    text_box,
    text_extent,
)

__all__ = [
    "PLACER_ID",
    "PLACER_VERSION",
    "SilkParams",
    "SilkPlacement",
    "stroke_for",
    "ascii_text",
    "is_named_net",
    "place_silkscreen",
]

PLACER_ID = "silkscreen.place"
PLACER_VERSION = "0.1"
#: KiCad's default silk text stroke; texts above 1 mm get 15 % of their size
MIN_STROKE_MM = 0.15
#: coordinates of placed texts are rounded to this many decimals (1 um) before they are tested
COORD_DECIMALS = 3
#: the reference candidates, in the order they are tried: (name, rotation)
REFERENCE_CANDIDATES: tuple[tuple[str, float], ...] = (
    ("above", 0.0), ("below", 0.0), ("left", 0.0), ("left", 90.0), ("right", 0.0), ("right", 90.0),
    ("top-left", 0.0), ("top-right", 0.0), ("bottom-left", 0.0), ("bottom-right", 0.0),
)
#: the title corners, in the order that breaks a tie
CORNERS: tuple[str, ...] = ("top-left", "top-right", "bottom-left", "bottom-right")
#: symbol libraries whose parts get pin labels
CONNECTOR_LIBRARY_PREFIX = "Connector"
#: the title's slide step along an edge when no corner is free
TITLE_SLIDE_STEP_MM = 0.25
KEEPOUT_NOTE = (
    "keep-outs: pad copper + silk_to_pad, the outline inset by silk_to_edge, footprint silk graphics / texts and placed texts + gap; "
    "tracks are not keep-outs (silk over mask-covered copper is normal), vias are tented; estimated text box "
    "(characters x size x 0.9 + stroke by size x 1.2 + stroke), not KiCad's font metrics; KiCad DRC decides silk_over_copper / silk_overlap"
)


@dataclass(frozen=True, slots=True)
class SilkParams:
    """The placer's margins and text sizes (recorded in every text's provenance)."""

    silk_to_pad_mm: float = 0.15
    silk_to_edge_mm: float = 0.3
    text_gap_mm: float = 0.1
    reference_sizes_mm: tuple[float, ...] = (1.0, 0.8)
    pin_label_size_mm: float = 0.8
    title_size_mm: float = 1.5

    def tag(self) -> str:
        sizes = "/".join(f"{s:g}" for s in self.reference_sizes_mm)
        return (
            f"params:silk_to_pad={self.silk_to_pad_mm:g},silk_to_edge={self.silk_to_edge_mm:g},gap={self.text_gap_mm:g},"
            f"reference_sizes={sizes},pin_label={self.pin_label_size_mm:g},title={self.title_size_mm:g}"
        )


@dataclass(slots=True)
class SilkPlacement:
    """What :func:`place_silkscreen` produced: the texts and what happened to each reference / label / the title."""

    texts: list[SilkText]
    params: SilkParams
    references_on_silk: list[str] = field(default_factory=list)
    references_on_fab: list[str] = field(default_factory=list)
    #: references on the silk only beside another footprint's courtyard (no candidate outside every other courtyard): "J2: within ..."
    references_in_courtyard: list[str] = field(default_factory=list)
    labels_placed: list[str] = field(default_factory=list)  # "J1.1 VIN"
    labels_skipped: list[tuple[str, str]] = field(default_factory=list)  # ("J1.3 VOUT", why)
    title: str | None = None
    title_corner: str | None = None  # "at the top-left corner" / "on the bottom edge, 5 mm from the bottom-left corner"
    notes: list[str] = field(default_factory=list)

    def description(self) -> str:
        """One line for the agent's note and the proposal description."""
        text = (
            f"{PLACER_ID} {PLACER_VERSION}: {len(self.references_on_silk)} reference(s) on the silkscreen, "
            f"{len(self.references_on_fab)} on the fab layer"
            + (f" ({', '.join(self.references_on_fab)}: no free silk position at {'/'.join(f'{s:g}' for s in self.params.reference_sizes_mm)} mm)" if self.references_on_fab else "")
            + f", {len(self.labels_placed)} connector pin label(s) placed, {len(self.labels_skipped)} skipped"
        )
        if self.title is not None:
            text += f", title {self.title!r} {self.title_corner}"
        else:
            text += ", no title"
        return text


def stroke_for(size: float) -> float:
    """The stroke width of a text of ``size`` mm: KiCad's default 0.15 mm, 15 % of the size above 1 mm."""
    return max(MIN_STROKE_MM, round(0.15 * size, 3))


def ascii_text(text: str) -> tuple[str, bool]:
    """``(printable ASCII characters of text, whether any character was dropped)``, stripped."""
    kept = "".join(ch for ch in text if 0x20 <= ord(ch) <= 0x7E)
    return kept.strip(), kept != text


def is_named_net(name: str) -> bool:
    """A net a person named: not empty, not KiCad's automatic ``Net-(...)`` / ``unconnected-...``."""
    return bool(name) and not name.startswith("Net-(") and not name.startswith("unconnected-")


def _r(v: float) -> float:
    return _q(round(v, COORD_DECIMALS))


def _r_toward(v: float, *, up: bool) -> float:
    """``v`` rounded to :data:`COORD_DECIMALS` upwards (``up``) or downwards."""
    scale = 10 ** COORD_DECIMALS
    return _q((math.ceil(v * scale - 1e-6) if up else math.floor(v * scale + 1e-6)) / scale)


# --------------------------------------------------------------------------- the board as keep-outs


def _body(placement: Placement, fp: FootprintDef, params: SilkParams) -> BBox | None:
    """What texts are placed beside: the courtyard, widened where the pads' keep-out reaches beyond it (a courtyard flush with its
    pads), so a candidate ``text_gap`` outside it is also ``silk_to_pad`` clear of those pads; the pads alone without a courtyard.

    A footprint with a ``custom`` / ``trapezoid`` pad has its courtyard as that pad's keep-out (:class:`_Board`), so the courtyard is
    widened the same way - otherwise every candidate ``text_gap`` outside it would sit ``text_gap`` < ``silk_to_pad`` from that
    keep-out and the reference would go to the fab layer even alone on an empty board."""
    boxes = []
    grow = max(params.silk_to_pad_mm - params.text_gap_mm, 0.0)
    court = courtyard_bbox(placement, fp)
    if court is not None:
        if any(pad.shape not in CONVEX_PAD_SHAPES for pad in fp.pads):
            court = BBox(court.x1 - grow, court.y1 - grow, court.x2 + grow, court.y2 + grow)
        boxes.append(court)
    pads = pads_bbox(placement, fp)
    if pads is not None:
        boxes.append(BBox(pads.x1 - grow, pads.y1 - grow, pads.x2 + grow, pads.y2 + grow))
    if not boxes:
        return None
    return BBox(_q(min(b.x1 for b in boxes)), _q(min(b.y1 for b in boxes)), _q(max(b.x2 for b in boxes)), _q(max(b.y2 for b in boxes)))


@dataclass(slots=True)
class _Obstacle:
    label: str
    shape: Shape
    box: Box
    margin: float


class _Board:
    """Keep-outs per side (``"F"`` / ``"B"``) and the placed footprints; refuses what it cannot bound."""

    def __init__(self, ir: CircuitIR, library: KicadLibrary, params: SilkParams) -> None:
        pcb = ir.pcb
        if pcb is None or not pcb.placements:
            raise CompileError("no placements: nothing to put silkscreen beside")
        if pcb.outline is None:
            raise CompileError("ir.pcb.outline is None: the edge keep-out cannot be measured")
        o = pcb.outline
        self.outline: Box = (o.origin_x_mm, o.origin_y_mm, o.origin_x_mm + o.width_mm, o.origin_y_mm + o.height_mm)
        e = params.silk_to_edge_mm
        self.inset: Box = (self.outline[0] + e, self.outline[1] + e, self.outline[2] - e, self.outline[3] - e)
        self.params = params
        self.obstacles: dict[str, list[_Obstacle]] = {"F": [], "B": []}
        self.parts: list[tuple[Component, FootprintDef, Placement, BBox]] = []
        #: per side, every footprint's courtyard box (its pads' box without one): a reference prefers a spot outside the others'
        self.courtyards: dict[str, list[tuple[str, _Obstacle]]] = {"F": [], "B": []}
        for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
            placement = pcb.placement(comp.ref)
            if placement is None:
                raise CompileError(f"component {comp.ref!r} has no placement; its pads cannot be kept clear")
            if comp.footprint is None:
                raise CompileError(f"component {comp.ref!r} has no footprint; its pads and silk are unknown")
            try:
                fp = library.load_footprint(comp.footprint)
            except LibraryLookupError as e:
                raise CompileError(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} was not found in a KiCad library: {e}") from e
            body = _body(placement, fp, params)
            if body is None:
                raise CompileError(f"footprint {fp.lib_id} of {comp.ref!r} has neither a courtyard nor pads; there is nothing to place its reference beside")
            self.parts.append((comp, fp, placement, body))
            own = courtyard_bbox(placement, fp) or pads_bbox(placement, fp) or body
            own_shape = Shape(((own.x1, own.y1), (own.x2, own.y1), (own.x2, own.y2), (own.x1, own.y2)))
            self.courtyards["B" if placement.side == BoardSide.BOTTOM else "F"].append(
                (comp.ref, _Obstacle(f"the courtyard of {comp.ref}", own_shape, own_shape.bbox(), params.text_gap_mm)))
            for pad in pad_copper(comp.ref, fp, placement):
                if pad.shape is None:
                    cy = courtyard_bbox(placement, fp)
                    if cy is None:
                        raise CompileError(
                            f"pad {pad.label} of footprint {fp.lib_id} has shape {pad.pad_shape!r} (custom primitives / trapezoid rect_delta are not read) "
                            "and the footprint has no courtyard to bound it"
                        )
                    shape = Shape(((cy.x1, cy.y1), (cy.x2, cy.y1), (cy.x2, cy.y2), (cy.x1, cy.y2)))
                    label = f"{pad.label} ({pad.pad_shape} pad: the courtyard of {comp.ref} stands in for its copper)"
                else:
                    shape, label = pad.shape, f"pad {pad.label}"
                for side in sorted(pad.sides):
                    self._add(side, label, shape, params.silk_to_pad_mm)
            values = {"Value": comp.value, "Datasheet": (comp.datasheet.url or "") if comp.datasheet is not None else ""}
            for item in footprint_silk(comp.ref, fp, placement, values).items:
                if item.shape is None:
                    raise CompileError(f"{item.label} of footprint {fp.lib_id}: {item.why}; its silk cannot be kept clear")
                self._add("B" if item.layer.startswith("B.") else "F", f"silk {item.label}", item.shape, params.text_gap_mm)

    def _add(self, side: str, label: str, shape: Shape, margin: float) -> None:
        self.obstacles[side].append(_Obstacle(label, shape, shape.bbox(), margin))

    def add_text(self, side: str, label: str, shape: Shape) -> None:
        self._add(side, f"text {label}", shape, self.params.text_gap_mm)

    def blocked(self, shape: Shape, side: str, extra: list[_Obstacle] | None = None) -> str | None:
        """Why ``shape`` cannot go on ``side`` (the first collision, in a fixed order), or ``None`` when it is free."""
        if not inside_box(shape, self.inset):
            return f"within {self.params.silk_to_edge_mm:g} mm of the board edge"
        x1, y1, x2, y2 = shape.bbox()
        for ob in [*self.obstacles[side], *(extra or [])]:
            b, m = ob.box, ob.margin
            if x1 - m > b[2] or b[0] - m > x2 or y1 - m > b[3] or b[1] - m > y2:
                continue
            if shape_distance(shape, ob.shape) < _q(m):
                return f"within {m:g} mm of {ob.label}"
        return None


# --------------------------------------------------------------------------- the placer


def _provenance(derived_from: list[str], note: str, params: SilkParams) -> Provenance:
    return Provenance(
        kind=ProvenanceKind.DERIVED, tool=PLACER_ID, tool_version=PLACER_VERSION,
        derived_from=[*derived_from, params.tag()], note=f"{note}; {KEEPOUT_NOTE}",
    )


def _reference_candidates(body: BBox, w: float, h: float, gap: float) -> list[tuple[int, str, float, float, float]]:
    """``(index, name, x, y, rotation)`` of the ten candidates for a ``w`` x ``h`` box (at rotation 0) around ``body``."""
    cx, cy = (body.x1 + body.x2) / 2.0, (body.y1 + body.y2) / 2.0
    spots = {
        ("above", 0.0): (cx, body.y1 - gap - h / 2.0),
        ("below", 0.0): (cx, body.y2 + gap + h / 2.0),
        ("left", 0.0): (body.x1 - gap - w / 2.0, cy),
        ("left", 90.0): (body.x1 - gap - h / 2.0, cy),
        ("right", 0.0): (body.x2 + gap + w / 2.0, cy),
        ("right", 90.0): (body.x2 + gap + h / 2.0, cy),
        ("top-left", 0.0): (body.x1 - w / 2.0, body.y1 - gap - h / 2.0),
        ("top-right", 0.0): (body.x2 + w / 2.0, body.y1 - gap - h / 2.0),
        ("bottom-left", 0.0): (body.x1 - w / 2.0, body.y2 + gap + h / 2.0),
        ("bottom-right", 0.0): (body.x2 + w / 2.0, body.y2 + gap + h / 2.0),
    }
    return [(i, name, _r(spots[(name, rot)][0]), _r(spots[(name, rot)][1]), rot) for i, (name, rot) in enumerate(REFERENCE_CANDIDATES)]


@dataclass(frozen=True, slots=True)
class _ConnectorPads:
    """The labelled pads of a connector: ``(number, net, centre)``, the row direction and the label side(s)."""

    pads: list[tuple[str, str, tuple[float, float]]]
    vertical_row: bool
    toward_centre: bool  # the positive side (right / below) faces the board centre
    fixed: list[tuple[str, str, tuple[float, float], bool]]  # off the centre line: (number, net, centre, positive side)
    centre_line: list[tuple[str, str, tuple[float, float]]]

    def label_sides(self) -> set[str]:
        """The reference candidate names the labels will occupy (the centre-line pads' side predicted as the one facing the board centre)."""
        positive = {positive for *_, positive in self.fixed}
        if self.centre_line:
            positive.add(self.toward_centre)
        names = ("right", "left") if self.vertical_row else ("below", "above")
        return {names[0] if flag else names[1] for flag in positive}


def _connector_pads(comp: Component, fp: FootprintDef, placement: Placement, body: BBox, pin_net: dict[tuple[str, str], str], board_centre: tuple[float, float]) -> _ConnectorPads | None:
    """The pads of a ``Connector*`` part that get a label (a named net), or ``None`` when it is not one / has none."""
    if comp.symbol is None or not comp.symbol.library.startswith(CONNECTOR_LIBRARY_PREFIX):
        return None
    pads: list[tuple[str, str, tuple[float, float]]] = []
    seen: set[str] = set()
    for pad in sorted(fp.pads, key=lambda q: natural_ref_key(q.number)):
        if not pad.number or pad.number in seen:
            continue
        seen.add(pad.number)
        net = pin_net.get((comp.ref, pad.number))
        if net is None or not is_named_net(net):
            continue
        pads.append((pad.number, net, pad_copper_center(placement, pad)))
    if not pads:
        return None
    xs, ys = [c[0] for _, _, c in pads], [c[1] for _, _, c in pads]
    vertical_row = (max(ys) - min(ys)) >= (max(xs) - min(xs))
    bx, by = (body.x1 + body.x2) / 2.0, (body.y1 + body.y2) / 2.0
    toward_centre = (board_centre[0] >= bx) if vertical_row else (board_centre[1] >= by)
    fixed: list[tuple[str, str, tuple[float, float], bool]] = []
    centre_line: list[tuple[str, str, tuple[float, float]]] = []
    for number, net, xy in pads:
        offset = (xy[0] - bx) if vertical_row else (xy[1] - by)
        if abs(offset) > 0.01:
            fixed.append((number, net, xy, offset > 0))
        else:
            centre_line.append((number, net, xy))
    return _ConnectorPads(pads, vertical_row, toward_centre, fixed, centre_line)


def _place_references(board: _Board, out: SilkPlacement, connectors: dict[str, _ConnectorPads]) -> None:
    p = board.params
    for comp, fp, placement, body in board.parts:
        side = "B" if placement.side == BoardSide.BOTTOM else "F"
        mirrored = side == "B"
        label_sides = connectors[comp.ref].label_sides() if comp.ref in connectors else set()
        others = [ob for ref, ob in board.courtyards[side] if ref != comp.ref]
        tried = 0
        found: tuple[int, str, float, float, float, float, float, Shape, str | None] | None = None
        # first a spot outside every other footprint's courtyard (a designator there reads as that part's); only when none of the
        # candidates at any size has one, the first spot that is merely free of the keep-outs
        for strict in (True, False):
            tried = 0
            for size in p.reference_sizes_mm:
                thickness = stroke_for(size)
                w, h = text_extent(comp.ref, size, thickness)
                candidates = _reference_candidates(body, w, h, p.text_gap_mm)
                candidates = [c for c in candidates if c[1] not in label_sides] + [c for c in candidates if c[1] in label_sides]
                for index, name, x, y, rot in candidates:
                    tried += 1
                    shape = text_box(comp.ref, x, y, rot, size, thickness, "center", mirrored=mirrored)
                    if board.blocked(shape, side) is not None:
                        continue
                    crowded = board.blocked(shape, side, others) if others else None
                    if strict and crowded is not None:
                        continue
                    found = (index, name, x, y, rot, size, thickness, shape, crowded)
                    break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            index, name, x, y, rot, size, thickness, shape, crowded = found
            where = f"reference beside the courtyard of {comp.ref} ({name}, candidate {index} of {len(REFERENCE_CANDIDATES)} at {size:g} mm)"
            if crowded is not None:
                where += f"; {crowded}: no candidate outside every other footprint's courtyard"
                out.references_in_courtyard.append(f"{comp.ref}: {crowded}")
            out.texts.append(SilkText(
                text=comp.ref, x_mm=x, y_mm=y, rotation_deg=rot, layer=f"{side}.SilkS", size_mm=size, thickness_mm=thickness,
                justify="center", kind=SilkKind.REFERENCE, component_ref=comp.ref,
                provenance=_provenance(
                    [f"component:{comp.ref}", f"footprint:{fp.lib_id}", f"placement:{placement.x_mm:g},{placement.y_mm:g},{placement.rotation_deg:g},{placement.side}",
                     f"candidate:{index}:{name}@{rot:g}", f"size_mm:{size:g}"],
                    where, p,
                ),
            ))
            board.add_text(side, comp.ref, shape)
            out.references_on_silk.append(comp.ref)
            continue
        cx, cy = _r((body.x1 + body.x2) / 2.0), _r((body.y1 + body.y2) / 2.0)
        size = p.reference_sizes_mm[0]
        out.texts.append(SilkText(
            text=comp.ref, x_mm=cx, y_mm=cy, rotation_deg=0.0, layer=f"{side}.Fab", size_mm=size, thickness_mm=stroke_for(size),
            justify="center", kind=SilkKind.REFERENCE, component_ref=comp.ref,
            provenance=_provenance(
                [f"component:{comp.ref}", f"footprint:{fp.lib_id}", f"placement:{placement.x_mm:g},{placement.y_mm:g},{placement.rotation_deg:g},{placement.side}",
                 f"candidates_tried:{tried}"],
                f"no free silk position for {comp.ref} among {tried} candidate(s): the reference is on the fab layer at the courtyard centre (hidden on the silk)", p,
            ),
        ))
        out.references_on_fab.append(comp.ref)


def _label_spot(body: BBox, pad_xy: tuple[float, float], vertical_row: bool, positive: bool, gap: float, mirrored: bool) -> tuple[float, float, float, str]:
    """``(x, y, rotation, justify)`` of a pin label beside a pad, outside ``body``, running away from it."""
    if vertical_row:  # labels horizontal, left / right of the body
        x = body.x2 + gap if positive else body.x1 - gap
        y, rot = pad_xy[1], 0.0
    else:  # labels vertical (reading upwards), above / below the body
        y = body.y2 + gap if positive else body.y1 - gap
        x, rot = pad_xy[0], 90.0
        positive = not positive  # at 90 degrees the text's own +x points to -y (up)
    justify = "left" if positive != mirrored else "right"
    return _r(x), _r(y), rot, justify


def _connectors(ir: CircuitIR, board: _Board) -> dict[str, _ConnectorPads]:
    pin_net = {(pin.component_ref, pin.pin_number): net.name for net in ir.nets for pin in net.pins}
    centre = ((board.outline[0] + board.outline[2]) / 2.0, (board.outline[1] + board.outline[3]) / 2.0)
    out: dict[str, _ConnectorPads] = {}
    for comp, fp, placement, body in board.parts:
        found = _connector_pads(comp, fp, placement, body, pin_net, centre)
        if found is not None:
            out[comp.ref] = found
    return out


def _place_pin_labels(board: _Board, out: SilkPlacement, connectors: dict[str, _ConnectorPads]) -> None:
    p = board.params
    size = p.pin_label_size_mm
    thickness = stroke_for(size)
    for comp, fp, placement, body in board.parts:
        conn = connectors.get(comp.ref)
        if conn is None:
            continue
        side = "B" if placement.side == BoardSide.BOTTOM else "F"
        mirrored = side == "B"
        vertical_row, toward_centre, fixed, centre_line = conn.vertical_row, conn.toward_centre, conn.fixed, conn.centre_line

        def attempt(items: list[tuple[str, str, tuple[float, float], bool]]) -> list[tuple[str, str, SilkText | None, Shape | None, str]]:
            """Place ``items`` in order (each placed one a keep-out for the next, nothing committed): ``(label, net, text, shape, why)``."""
            trial: list[_Obstacle] = []
            result = []
            for number, net, xy, positive in items:
                label = f"{comp.ref}.{number} {net}"
                text, dropped = ascii_text(net)
                if dropped or not text:
                    result.append((label, net, None, None, "net name is not printable ASCII (the text-box estimate assumes Latin glyphs)"))
                    continue
                x, y, rot, justify = _label_spot(body, xy, vertical_row, positive, p.text_gap_mm, mirrored)
                shape = text_box(text, x, y, rot, size, thickness, justify, mirrored=mirrored)
                why = board.blocked(shape, side, trial)
                if why is not None:
                    result.append((label, net, None, None, why))
                    continue
                silk = SilkText(
                    text=text, x_mm=x, y_mm=y, rotation_deg=rot, layer=f"{side}.SilkS", size_mm=size, thickness_mm=thickness, justify=justify,
                    kind=SilkKind.PIN_LABEL, component_ref=comp.ref,
                    provenance=_provenance(
                        [f"component:{comp.ref}", f"pad:{comp.ref}.{number}", f"net:{net}", f"footprint:{fp.lib_id}", f"size_mm:{size:g}"],
                        f"net name of pad {comp.ref}.{number} beside the pad, outside the courtyard, {'right' if positive else 'left'} of the row" if vertical_row
                        else f"net name of pad {comp.ref}.{number} beside the pad, outside the courtyard, {'below' if positive else 'above'} the row", p,
                    ),
                )
                trial.append(_Obstacle(f"text {label}", shape, shape.bbox(), p.text_gap_mm))
                result.append((label, net, silk, shape, ""))
            return result

        chosen = attempt(fixed)
        if centre_line:
            first, second = (True, False) if toward_centre else (False, True)
            a = attempt([(n, net, xy, first) for n, net, xy in centre_line])
            b = attempt([(n, net, xy, second) for n, net, xy in centre_line])
            best = a if sum(1 for r in a if r[2] is not None) >= sum(1 for r in b if r[2] is not None) else b
            # the fixed-side labels were tried without the centre-line ones: re-run both together so every commit is checked against the others
            chosen = attempt([*fixed, *[(n, net, xy, first if best is a else second) for n, net, xy in centre_line]])
        for label, _net, silk, shape, why in chosen:
            if silk is None or shape is None:
                out.labels_skipped.append((label, why))
                continue
            out.texts.append(silk)
            board.add_text(side, label, shape)
            out.labels_placed.append(label)


def _place_title(ir: CircuitIR, board: _Board, out: SilkPlacement) -> None:
    p = board.params
    text, dropped = ascii_text(ir.project.name or "")
    if dropped:
        out.notes.append(f"silkscreen title: characters outside printable ASCII dropped from {ir.project.name!r} (KiCad's stroke font; the text-box estimate assumes Latin glyphs)")
    if not text:
        out.notes.append("silkscreen title not placed: the project name has no printable ASCII character")
        return
    size = p.title_size_mm
    thickness = stroke_for(size)
    w, h = text_extent(text, size, thickness)
    x1, y1, x2, y2 = board.inset
    if x2 - x1 < w or y2 - y1 < h:
        out.notes.append(f"silkscreen title not placed: the {w:.2f} x {h:.2f} mm title box is larger than the board inside the edge rule")
        return

    def box_at(corner: str, scale: float) -> Shape:
        bw, bh = w * scale, h * scale
        left = corner.endswith("left")
        top = corner.startswith("top")
        bx1 = x1 if left else x2 - bw
        by1 = y1 if top else y2 - bh
        return Shape(((bx1, by1), (bx1 + bw, by1), (bx1 + bw, by1 + bh), (bx1, by1 + bh)))

    best: tuple[float, str] | None = None
    for corner in CORNERS:
        if board.blocked(box_at(corner, 1.0), "F") is not None:
            continue
        lo, hi = 1.0, min((x2 - x1) / w, (y2 - y1) / h)
        for _ in range(24):  # the largest free scale, to 1e-4 of the title box
            mid = (lo + hi) / 2.0
            if board.blocked(box_at(corner, mid), "F") is None:
                lo = mid
            else:
                hi = mid
        if best is None or lo > best[0] + 1e-9:
            best = (lo, corner)
    slide = 0.0
    if best is not None:
        scale, corner = best
        where = f"the {corner} corner, the corner with the most room for it (x{scale:.2f} of its box)"
        tags = [f"corner:{corner}", f"free_scale:{scale:.3f}"]
    else:
        # no corner is free: slide along the top / bottom edge towards the middle, nearest free position wins
        steps = int(((x2 - x1) - w) / 2.0 / TITLE_SLIDE_STEP_MM)
        found: tuple[int, int, str] | None = None
        for order, corner in enumerate(CORNERS):
            sign = 1.0 if corner.endswith("left") else -1.0
            for k in range(1, steps + 1):
                if found is not None and k >= found[0]:
                    break
                shape = box_at(corner, 1.0)
                moved = Shape(tuple((px + sign * k * TITLE_SLIDE_STEP_MM, py) for px, py in shape.points))
                if board.blocked(moved, "F") is None:
                    found = (k, order, corner)
                    break
        if found is None:
            out.notes.append(f"silkscreen title not placed: no board corner has room for the {w:.2f} x {h:.2f} mm title box, nor any position along the top / bottom edge")
            return
        k, _order, corner = found
        slide = k * TITLE_SLIDE_STEP_MM
        where = f"the {corner.split('-')[0]} edge, {slide:g} mm from the {corner} corner"
        tags = [f"corner:{corner}", f"slide_mm:{slide:g}"]
    left = corner.endswith("left")
    # rounded towards the board's inside, so the rounding never moves the box across the edge rule
    x = _r_toward(x1 + slide, up=True) if left else _r_toward(x2 - slide, up=False)
    y = _r_toward(y1 + h / 2.0, up=True) if corner.startswith("top") else _r_toward(y2 - h / 2.0, up=False)
    justify = "left" if left else "right"
    shape = text_box(text, x, y, 0.0, size, thickness, justify)
    if board.blocked(shape, "F") is not None:  # the rounding moved it into a keep-out: nothing is written that was not tested
        out.notes.append(f"silkscreen title not placed: the rounded position at the {corner} corner touches a keep-out")
        return
    out.texts.append(SilkText(
        text=text, x_mm=x, y_mm=y, rotation_deg=0.0, layer="F.SilkS", size_mm=size, thickness_mm=thickness, justify=justify, kind=SilkKind.TITLE,
        provenance=_provenance([f"project.name:{ir.project.name}", *tags, f"size_mm:{size:g}"], f"board title on {where}", p),
    ))
    board.add_text("F", "title", shape)
    out.title, out.title_corner = text, f"at the {corner} corner" if not slide else f"on {where}"
    if slide:
        out.notes.append(f"silkscreen title: no corner was free; placed on {where}")


def place_silkscreen(ir: CircuitIR, library: KicadLibrary, params: SilkParams | None = None) -> SilkPlacement:
    """Place references, connector pin labels and the title on the placed board of ``ir`` (module docstring); pure.

    Raises :class:`CompileError` when the board cannot be read: no
    placements or outline, a component without a placement / footprint, a
    footprint not on disk or malformed, a non-convex pad in a footprint
    without a courtyard, a silk construct the geometry module does not read.
    ``ir`` is never modified.
    """
    params = params or SilkParams()
    board = _Board(ir, library, params)
    out = SilkPlacement(texts=[], params=params)
    connectors = _connectors(ir, board)
    _place_references(board, out, connectors)
    _place_pin_labels(board, out, connectors)
    _place_title(ir, board, out)
    if out.references_in_courtyard:
        out.notes.append("silkscreen: reference(s) at another footprint's courtyard (no candidate outside every other courtyard): " + "; ".join(out.references_in_courtyard))
    if out.references_on_fab:
        out.notes.append(f"silkscreen: reference(s) on the fab layer (no free silk position): {', '.join(out.references_on_fab)}")
    if out.labels_skipped:
        out.notes.append("silkscreen: pin label(s) skipped (never overlapped): " + "; ".join(f"{label}: {why}" for label, why in out.labels_skipped))
    return out
