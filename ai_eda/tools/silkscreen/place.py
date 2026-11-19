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
  hole on both. A ``custom`` pad counts with the boxes of its anchor and each
  copper primitive (:func:`ai_eda.tools.kicad.geometry.custom_pad_parts`, the
  extent the router reads). A ``trapezoid``'s copper (``rect_delta``) and a
  custom pad with a primitive the library reader does not read are unknown:
  the footprint's courtyard stands in for them (a footprint with such a pad
  and no courtyard is refused, :class:`CompileError`);
* the board edge: every text lies inside the outline inset by
  ``silk_to_edge_mm`` (0.3);
* every footprint's own silkscreen graphics and visible silk texts (library
  lines, arcs, circles, polygons, texts - so a reference never covers a
  pin-1 or polarity mark) and every text placed before, each kept
  ``text_gap_mm`` (0.1) away;
* **not** tracks: silk over mask-covered copper is normal - but a reference
  prefers a spot clear of the tracks on its side's outer copper (step 1);
* every via's disc + ``text_gap``, on each outer side its hole reaches (both
  for a through via). The compiled board tents every via (``compilers/pcb.py``
  ``_setup``: ``(tenting (front yes) (back yes))``), so the copper is under
  mask like a track's - but a via is a drilled hole, no silk prints over a
  hole, and a hole in a designator breaks its glyphs (the 9 V piano board of
  2026-11-18 had via INM6 inside ``C16``). The silk placer keeps the texts
  off the vias rather than the router keeping vias off the texts: the router
  runs first and the texts go where the copper leaves room, so this is the
  one place that knows both (a design file's kept texts are not moved either
  way; the ``pcb.silk.*`` checks judge them).

Placement, in this order (each text is a keep-out for the next):

1. **References**, natural ref order: ten candidates around the footprint's
   courtyard (widened to the pads' box grown by ``silk_to_pad - text_gap``
   where the pads reach beyond it, and itself grown by ``silk_to_pad -
   text_gap`` when it stands in for a pad's unknown copper - a trapezoid, a
   custom pad with an unread primitive; the pads' box alone without a
   courtyard) - above, below, left at 0 then
   90 degrees, right at 0 then 90 degrees, then the four corners (the box
   diagonally outside each courtyard corner) at 1.0 mm (0.15 mm stroke,
   KiCad's default silk text) and at 0.8 mm. They are tried horizontal
   first: the four sides (above, below, left, right) at 1.0 mm, then each of
   them slid along its side in :data:`OFFSET_STEP_MM` steps (the smaller
   shift first, as long as the text's centre stays within the body's extent
   along that side), then the same at 0.8 mm; then the four corners at 1.0
   and 0.8 mm; and only then the two turned candidates (left and right at 90
   degrees, also slid) at each size - a turned reference is named in the
   notes with why no horizontal one was taken. A candidate is taken only
   when its estimated box touches no keep-out; among those, in this order of
   preference: (a) one that stays ``text_gap`` clear of every *other*
   footprint's courtyard on its side (their pads' box without one) - a
   designator inside a neighbour's courtyard reads as the neighbour's; only
   when none is, the first one (in the same order) that merely touches no
   keep-out wins and the note names the courtyard; (b) one not on a pin-label
   side (below); (c) one that reads as its own part's: every other courtyard
   at least :data:`OWN_MARGIN_MM` farther from the box than the part's own
   body (a designator in the gap between two trimmers or two resistor columns,
   as near the neighbour as its own part, names either - the 9 V piano board of
   2026-11-18 had ``RV5`` off the corner of ``RV5`` beside ``RV6``); only when
   none does, the first that does not (the note names the nearer courtyard);
   (d) one at least :data:`PREFERRED_EDGE_MM` from the board edge (the hard
   rule stays ``silk_to_edge``; the rev 3 piano board had R18 / R38 0.55 mm
   from the edge; the note gives the gap of one nearer); (e) the standard
   size before the smaller one - a turned 1 mm text before a horizontal
   0.8 mm one: 0.8 mm is the last resort, and the note says why the standard
   size was not free (the rev 3 board had eight 0.8 mm references where a
   1 mm one fitted turned); (f) one at least :data:`CLUSTER_GAP_MM` from every
   other part's reference (two designators closer read as one cluster - the
   rev 3 board's 'C11' and 'C12' 0.93 mm apart with 'R11' 0.6 mm above them;
   the note names the nearer one); (g) the horizontal tiers before the turned
   one, as above - a turned reference says why (its free horizontal spots
   read as another part's, sit near the edge or another reference, or are not
   free in every repeat of its cell); (h) a spot whose box touches no track on
   its side's outer copper before one over a track (the note says when none
   was clear). **Repeated cells** (since 2026-11-19): pack regions
   (``ir.pcb.pack_regions``) of two parts or more whose parts are the same
   footprints at the same rotation and side, one region a translation of the
   other (within 1 um), are repeats of one cell - the rev 3 board's eight voice
   columns, whose references were placed differently in each (RV2/3/6/7/8
   above, RV1/RV4 right, RV5 left; C11/C16 right and the other C1x left). The
   parts at one cell position are placed together, at one candidate (the same
   offset from every body): the first in the order above that is free in
   every repeat - a repeat's preference counts as met when it is met in every
   repeat (a candidate is ranked by how many repeats miss each preference),
   and the texts of the repeats keep ``text_gap`` from each other. The notes
   give the spot of every cell position and, where most repeats alone would
   have taken another spot, which repeats it was not free at and what blocked
   it (a design file's text, a via, a pad: what a designer may move). Where no
   candidate is free in every repeat, the one free in the most repeats is
   theirs and the others are placed one by one, named in the notes. A
   connector with pin labels is in no cell. A connector that
   will get pin labels tries the side(s) its labels use (step 2) last - after
   every other spot, turned ones included -, so its own reference does not
   take their place; a part that will get a polarity mark (step 2b) tries the
   side its mark takes first last, for the same reason. None free: the reference goes to
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
   skipped too (the text-box estimate assumes Latin glyphs). A pad whose net
   name a kept text (a template's or design file's, below) already prints for
   that part - its ``component_ref`` is the part, its text the net name, as a
   design file labels a test point - gets no second label: the pad is left
   out of the part's labels (a part with none left places its reference as
   any other part) and the note names it.
2b. **Polarity marks**: a two-pin part whose symbol names its pins ``A`` and
   ``K`` (a diode, an LED) gets a ``K`` beside the pad of its ``K`` pin, one
   whose symbol names them ``+`` and ``-`` a ``+`` beside the ``+`` pad - the
   pin names are the library symbol's, never a recollection, and the pin's
   number must be a pad of the footprint (else the part is named in the
   notes). A footprint whose own silk already carries the mark as a text gets
   none. Its body's graphics are no evidence: a "flat" on an LED outline is a
   KiCad drawing convention the part may lack (the TLLR4400's collar is a
   full circle, Vishay doc. 83029 p.7). The mark is 0.8 mm, outside the pad's
   keep-out, on the side away from the part's other pads first, then the two
   sides across, then the side toward them, and outside every other
   courtyard - the first such spot clear of the tracks on the side's outer
   copper, else the first over a track (its note says so); a mark that finds
   no free spot is left out and named in the notes, never overlapped. It is a ``pin_label`` text of its part.
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

**Kept texts (``silkscreen.place`` 0.2).** A template - or a design file's
``board`` section - may put board texts on the silk before the board is
placed (a keyboard's note names, a title: ``keep``, the ``user`` / ``title``
texts the PCB agent recognises as a template's or a design file's; the notes
name which, ``kept_by``). They stay exactly where they were put, ahead of
the placed texts in :attr:`SilkPlacement.texts`, and each is a keep-out for
every text placed here, before step 1: its estimated box grown by one
character of it (:data:`KEPT_TEXT_GAP_CHARS` x its size x the estimate's
width factor, at least ``text_gap``) - a placed text closer than that reads
as one word with it (the 9 V piano board's pin label ``GND`` 0.35 mm after
the design text ``SPK`` read ``SPKGND``), so such a label is skipped and
named in the notes;
when one of them is a ``title`` text, step 3 places no second title. A kept
text the compiler would refuse (:func:`~ai_eda.tools.silkscreen.geometry.silk_text_problems`)
or of another kind is refused (:class:`CompileError`). Whether the kept
texts themselves keep clear of pads, the edge and the footprints' silk is
the ``pcb.silk.*`` checks' to judge, not this placer's. Every text placed on
such a board is stamped 0.2 and names the kept texts' count in
``derived_from`` (``kept_texts:<n>``); every other board is placed exactly as
0.1 did (byte for byte, stamp included).

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

from ai_eda.compilers.schematic import datasheet_field
from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, Component, Placement, Provenance, ProvenanceKind, SilkKind, SilkText, straight_pieces
from ai_eda.tools.kicad.geometry import _q, courtyard_bbox, pad_copper_center, pads_bbox
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary, LibraryLookupError
from ai_eda.tools.silkscreen.geometry import (
    CONVEX_PAD_SHAPES,
    TEXT_WIDTH_FACTOR,
    Box,
    Shape,
    footprint_silk,
    inside_box,
    ir_text_box,
    pad_copper,
    shape_distance,
    side_of_layer,
    silk_text_problems,
    text_box,
    text_extent,
)

__all__ = [
    "KEPT_KINDS",
    "PLACER_ID",
    "PLACER_KEEP_VERSION",
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
#: ``tool_version`` of every text placed on a board with kept template texts (module docstring); every other board keeps 0.1
PLACER_KEEP_VERSION = "0.2"
#: the silk text kinds a template may put on the board before it is placed (the placer adds references and pin labels itself)
KEPT_KINDS: frozenset[SilkKind] = frozenset({SilkKind.USER, SilkKind.TITLE})
#: KiCad's default silk text stroke; texts above 1 mm get 15 % of their size
MIN_STROKE_MM = 0.15
#: coordinates of placed texts are rounded to this many decimals (1 um) before they are tested
COORD_DECIMALS = 3
#: the reference candidates, in the order they are tried: (name, rotation)
REFERENCE_CANDIDATES: tuple[tuple[str, float], ...] = (
    ("above", 0.0), ("below", 0.0), ("left", 0.0), ("left", 90.0), ("right", 0.0), ("right", 90.0),
    ("top-left", 0.0), ("top-right", 0.0), ("bottom-left", 0.0), ("bottom-right", 0.0),
)
#: the horizontal reference candidates beside a side of the body - above, below, left, right (indices into
#: :data:`REFERENCE_CANDIDATES`) -, each also slid along its side: tried first, at every size
SIDE_CANDIDATES: tuple[int, ...] = tuple(i for i, (name, rot) in enumerate(REFERENCE_CANDIDATES) if rot == 0.0 and "-" not in name)
#: the horizontal candidates diagonally off a corner of the body: next (a corner text sits as near another part's corner)
CORNER_CANDIDATES: tuple[int, ...] = tuple(i for i, (name, rot) in enumerate(REFERENCE_CANDIDATES) if rot == 0.0 and "-" in name)
#: the reference candidates turned by 90 degrees (each slid along its side too): only when no horizontal one is free
ROTATED_CANDIDATES: tuple[int, ...] = tuple(i for i, (_name, rot) in enumerate(REFERENCE_CANDIDATES) if rot != 0.0)
#: the step by which the above / below / left / right candidates are slid along the side they face (module docstring, step 1)
OFFSET_STEP_MM = 0.25
#: a reference reads as its own part's when every other footprint's courtyard is at least this much farther from its box than its
#: own body is (module docstring, step 1): a designator as near a neighbour as its own part - in the gap between two trimmers, between
#: two resistor columns - names either
OWN_MARGIN_MM = 0.5
#: how the notes say a reference was turned because its free horizontal spots read as another part's
OWNED_TURN = "every free horizontal position reads as another part's"
#: a reference prefers a box this far from the board edge (module docstring, step 1) - a preference over the hard ``silk_to_edge``
#: (0.3 mm, the margin ``pcb.silk.clearance`` holds every silk item to): a designator printed at the very edge of the board reads
#: badly and is the first thing a depanelled edge chips; where no free spot keeps it, the nearest one is taken and named in the notes
PREFERRED_EDGE_MM = 1.0
#: a reference prefers a box this far from every other part's reference (module docstring, step 1): two designators closer than
#: about a character's width (1 mm text: 0.9 mm) read as one cluster ("C11 C12" between two bodies) and neither as its own part's
CLUSTER_GAP_MM = 1.0
#: how the notes say a reference was turned, or kept at a smaller size, for the other preferences of step 1
EDGE_TURN = f"every free horizontal position at its size is within {PREFERRED_EDGE_MM:g} mm of the board edge"
CLUSTER_TURN = f"every free horizontal position at its size is within {CLUSTER_GAP_MM:g} mm of another part's reference"
COMMON_TURN = "no horizontal position is free in every repeat of its cell"
OWNED_SMALL = "every free spot at the standard size reads as another part's"
EDGE_SMALL = f"every free spot at the standard size is within {PREFERRED_EDGE_MM:g} mm of the board edge"
COMMON_SMALL = "no spot at the standard size is free in every repeat of its cell"
#: the gap a placed text keeps from a kept (template or design file) text, in characters of the kept text: one character's width of
#: the text-box estimate (size x :data:`~ai_eda.tools.silkscreen.geometry.TEXT_WIDTH_FACTOR`), so the two never read as one word
KEPT_TEXT_GAP_CHARS = 1.0
#: the polarity mark a two-pin part gets: a diode / LED (symbol pins named ``A`` and ``K``) a ``K`` beside its cathode pad, a polarized
#: part whose symbol names its pins ``+`` and ``-`` a ``+`` beside its positive pad - ``(the two pin names, the marked pin, the mark)``
POLARITY_MARKS: tuple[tuple[frozenset[str], str, str], ...] = ((frozenset({"A", "K"}), "K", "K"), (frozenset({"+", "-"}), "+", "+"))
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
#: what every text placed on a board with copper (tracks or vias) adds to :data:`KEEPOUT_NOTE`
COPPER_NOTE = (
    "every via's disc + gap is a keep-out too (a tented via is still a drilled hole, and a hole inside a designator breaks its glyphs); a "
    "reference prefers a spot that reads as its own part's (every other courtyard at least 0.5 mm farther than its own body), then one "
    "1 mm from the board edge, then the 1 mm size to 0.8 mm, then one 1 mm from every other part's reference, then a "
    "horizontal spot (each side, slid along it, then the corners) to a turned one, and a spot clear of the tracks to one over a track; "
    "the repeats of a cell take one spot"
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
    #: references turned by 90 degrees because no horizontal position was free: "RV4 (why the first horizontal one was not taken)"
    references_rotated: list[str] = field(default_factory=list)
    #: references placed as near another part's courtyard as their own (no free spot read as their own part's): "R1: as near ..."
    references_ambiguous: list[str] = field(default_factory=list)
    #: references nearer the board edge than PREFERRED_EDGE_MM (no free spot kept it): "R38: 0.48 mm"
    references_near_edge: list[str] = field(default_factory=list)
    #: references within CLUSTER_GAP_MM of another part's reference (no free spot kept clear): "C12: the reference of C11"
    references_clustered: list[str] = field(default_factory=list)
    #: references below the standard (first) size, with why: "R23 (no position is free at 1 mm ...)"
    references_reduced: list[str] = field(default_factory=list)
    #: one line per repeated cell position (module docstring, step 1): its references, the common spot, what kept another one
    cells: list[str] = field(default_factory=list)
    #: polarity marks placed beside a part's cathode / positive pad ("D1.1 K"), and the ones left out with why
    marks_placed: list[str] = field(default_factory=list)
    marks_skipped: list[tuple[str, str]] = field(default_factory=list)
    labels_placed: list[str] = field(default_factory=list)  # "J1.1 VIN"
    labels_skipped: list[tuple[str, str]] = field(default_factory=list)  # ("J1.3 VOUT", why)
    #: pads whose net name a kept text already prints for the part (step 2): "TP1.1 VOUT"
    labels_on_kept: list[str] = field(default_factory=list)
    title: str | None = None
    title_corner: str | None = None  # "at the top-left corner" / "on the bottom edge, 5 mm from the bottom-left corner"
    notes: list[str] = field(default_factory=list)
    #: the template texts kept where they were (also the first entries of ``texts``); empty on every other board
    kept: list[SilkText] = field(default_factory=list)
    #: who put the kept texts on the board, as the notes name it: ``template``, or ``design file`` (a design file's board texts)
    kept_by: str = "template"

    def description(self) -> str:
        """One line for the agent's note and the proposal description."""
        text = (
            f"{PLACER_ID} {PLACER_KEEP_VERSION if self.kept else PLACER_VERSION}: "
            + (f"{len(self.kept)} {self.kept_by} text(s) kept where the {self.kept_by} put them, " if self.kept else "")
            + f"{len(self.references_on_silk)} reference(s) on the silkscreen, "
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


def _r_xy(xy: tuple[float, float]) -> tuple[float, float]:
    return _r(xy[0]), _r(xy[1])


def _r_toward(v: float, *, up: bool) -> float:
    """``v`` rounded to :data:`COORD_DECIMALS` upwards (``up``) or downwards."""
    scale = 10 ** COORD_DECIMALS
    return _q((math.ceil(v * scale - 1e-6) if up else math.floor(v * scale + 1e-6)) / scale)


# --------------------------------------------------------------------------- the board as keep-outs


def _body(placement: Placement, fp: FootprintDef, params: SilkParams) -> BBox | None:
    """What texts are placed beside: the courtyard, widened where the pads' keep-out reaches beyond it (a courtyard flush with its
    pads), so a candidate ``text_gap`` outside it is also ``silk_to_pad`` clear of those pads; the pads alone without a courtyard.

    A footprint with a pad whose copper is unknown (a trapezoid, a custom pad with an unread primitive) has its courtyard as that pad's
    keep-out (:class:`_Board`), so the courtyard is
    widened the same way - otherwise every candidate ``text_gap`` outside it would sit ``text_gap`` < ``silk_to_pad`` from that
    keep-out and the reference would go to the fab layer even alone on an empty board."""
    boxes = []
    grow = max(params.silk_to_pad_mm - params.text_gap_mm, 0.0)
    court = courtyard_bbox(placement, fp)
    if court is not None:
        if any(pad.shape not in CONVEX_PAD_SHAPES and (pad.shape != "custom" or pad.unread_primitives) for pad in fp.pads):
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
        f = max(PREFERRED_EDGE_MM, e)
        #: what a reference prefers to stay inside (PREFERRED_EDGE_MM)
        self.preferred: Box = (self.outline[0] + f, self.outline[1] + f, self.outline[2] - f, self.outline[3] - f)
        #: per side, the references placed so far: ``(ref, box shape, bbox)`` (a reference prefers to keep CLUSTER_GAP_MM from them)
        self.references: dict[str, list[tuple[str, Shape, Box]]] = {"F": [], "B": []}
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
                            f"pad {pad.label} of footprint {fp.lib_id} has shape {pad.pad_shape!r} (a trapezoid's rect_delta / a custom pad's unread primitive) "
                            "and the footprint has no courtyard to bound it"
                        )
                    shape = Shape(((cy.x1, cy.y1), (cy.x2, cy.y1), (cy.x2, cy.y2), (cy.x1, cy.y2)))
                    label = f"{pad.label} ({pad.pad_shape} pad: the courtyard of {comp.ref} stands in for its copper)"
                else:
                    shape, label = pad.shape, f"pad {pad.label}"
                for side in sorted(pad.sides):
                    self._add(side, label, shape, params.silk_to_pad_mm)
            values = {"Value": comp.value, "Datasheet": datasheet_field(comp)}  # the text the footprint carries (pcb.py)
            for item in footprint_silk(comp.ref, fp, placement, values).items:
                if item.shape is None:
                    raise CompileError(f"{item.label} of footprint {fp.lib_id}: {item.why}; its silk cannot be kept clear")
                self._add("B" if item.layer.startswith("B.") else "F", f"silk {item.label}", item.shape, params.text_gap_mm)
        # the routed copper (module docstring): every via's disc is a keep-out on each outer side it reaches, every
        # track on an outer layer is what a reference prefers to stay clear of
        self.copper = bool(pcb.vias or pcb.tracks)
        for via in pcb.vias:
            disc = Shape(((via.x_mm, via.y_mm),), via.diameter_mm / 2.0)
            for side in _via_sides(via):
                self._add(side, f"the via of {via.net} at ({via.x_mm:g}, {via.y_mm:g})", disc, params.text_gap_mm)
        #: per side, the tracks on that side's outer copper as capsules (a reference prefers a spot clear of them)
        self.tracks: dict[str, list[tuple[Shape, Box, str]]] = {"F": [], "B": []}
        for piece in straight_pieces(pcb.tracks):
            side = {"F.Cu": "F", "B.Cu": "B"}.get(piece.layer)
            if side is not None:
                capsule = Shape((piece.start, piece.end), float(piece.width_mm) / 2.0)
                self.tracks[side].append((capsule, capsule.bbox(), piece.net))

    def _add(self, side: str, label: str, shape: Shape, margin: float) -> None:
        self.obstacles[side].append(_Obstacle(label, shape, shape.bbox(), margin))

    def add_text(self, side: str, label: str, shape: Shape, margin: float | None = None) -> None:
        self._add(side, f"text {label}", shape, self.params.text_gap_mm if margin is None else margin)

    def on_track(self, shape: Shape, side: str, pool: list[tuple[Shape, Box, str]] | None = None) -> str | None:
        """The net of the first track on ``side``'s outer copper that ``shape`` touches, or ``None``; ``pool``, when given, stands in for
        the side's tracks (those near an area holding the shape)."""
        x1, y1, x2, y2 = shape.bbox()
        for capsule, b, net in (self.tracks[side] if pool is None else pool):
            if x1 > b[2] or b[0] > x2 or y1 > b[3] or b[1] > y2:
                continue
            if shape_distance(shape, capsule) <= 0.0:
                return net
        return None

    def near(self, side: str, area: Box) -> list[_Obstacle]:
        """The keep-outs on ``side`` (in their order) whose box, grown by their margin, meets ``area``: all that a shape inside ``area``
        can touch."""
        return [ob for ob in self.obstacles[side]
                if not (ob.box[0] - ob.margin > area[2] or area[0] > ob.box[2] + ob.margin or ob.box[1] - ob.margin > area[3] or area[1] > ob.box[3] + ob.margin)]

    def clustered(self, shape: Shape, side: str, ref: str) -> str | None:
        """``the reference of X`` for the first placed reference of another part within CLUSTER_GAP_MM of ``shape``, or ``None``."""
        x1, y1, x2, y2 = shape.bbox()
        g = CLUSTER_GAP_MM
        for other, other_shape, b in self.references[side]:
            if other == ref or x1 - g > b[2] or b[0] - g > x2 or y1 - g > b[3] or b[1] - g > y2:
                continue
            if shape_distance(shape, other_shape) < g - 1e-9:
                return f"the reference of {other}"
        return None

    def blocked(self, shape: Shape, side: str, extra: list[_Obstacle] | None = None, pool: list[_Obstacle] | None = None) -> str | None:
        """Why ``shape`` cannot go on ``side`` (the first collision, in a fixed order), or ``None`` when it is free; ``pool``, when
        given, stands in for the side's keep-outs (those of :meth:`near` an area holding the shape)."""
        if not inside_box(shape, self.inset):
            return f"within {self.params.silk_to_edge_mm:g} mm of the board edge"
        x1, y1, x2, y2 = shape.bbox()
        for ob in [*(self.obstacles[side] if pool is None else pool), *(extra or [])]:
            b, m = ob.box, ob.margin
            if x1 - m > b[2] or b[0] - m > x2 or y1 - m > b[3] or b[1] - m > y2:
                continue
            if shape_distance(shape, ob.shape) < _q(m):
                return f"within {m:g} mm of {ob.label}"
        return None


def _via_sides(via) -> list[str]:
    """The silk sides a via's hole reaches: both for a through via, the outer side(s) of a typed via's span."""
    if via.kind is None:
        return ["F", "B"]
    top, bottom = via.layers[0], via.layers[-1]
    return [side for side, layer in (("F", "F.Cu"), ("B", "B.Cu")) if layer in (top, bottom)]


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


def _nearer_part(shape: Shape, own: Shape, others: list[_Obstacle]) -> str | None:
    """The label of the first other courtyard (in ``others``' order) less than :data:`OWN_MARGIN_MM` farther from ``shape`` than the
    part's own body ``own`` - the text would read as that part's as much as its own -, or ``None`` when it reads as its own part's."""
    reach = shape_distance(shape, own) + OWN_MARGIN_MM
    x1, y1, x2, y2 = shape.bbox()
    for ob in others:
        b = ob.box
        if x1 - reach > b[2] or b[0] - reach > x2 or y1 - reach > b[3] or b[1] - reach > y2:
            continue
        if shape_distance(shape, ob.shape) < reach - 1e-9:
            return ob.label
    return None


def _shifted(body: BBox, candidates: list[tuple[int, str, float, float, float]],
             indices: tuple[int, ...]) -> list[tuple[int, str, float, float, float, float, float]]:
    """``(index, name, x, y, rotation, dx, dy)``: first the candidates ``indices`` as they are, then each ``above`` / ``below`` one slid
    along X and each ``left`` / ``right`` one slid along Y by :data:`OFFSET_STEP_MM` steps, the smaller shift first and the negative
    one before the positive one, as long as the text still faces the side of the body it names (its centre within the body's extent
    along that side); the corners are not slid."""
    out = [(i, name, x, y, rot, 0.0, 0.0) for i, name, x, y, rot in candidates if i in indices]
    slides: list[tuple[float, int, int, int, str, float, float, float, float, float]] = []
    for i, name, x, y, rot in candidates:
        if i not in indices or name not in ("above", "below", "left", "right"):
            continue
        along_x = name in ("above", "below")
        reach = (body.x2 - body.x1) / 2.0 if along_x else (body.y2 - body.y1) / 2.0
        k = 1
        while k * OFFSET_STEP_MM <= reach + 1e-9:
            for sign in (-1.0, 1.0):
                d = sign * k * OFFSET_STEP_MM
                dx, dy = (d, 0.0) if along_x else (0.0, d)
                slides.append((k * OFFSET_STEP_MM, indices.index(i), 0 if sign < 0 else 1, i, name, _r(x + dx), _r(y + dy), rot, dx, dy))
            k += 1
    slides.sort(key=lambda t: t[:3])
    out += [(i, name, x, y, rot, dx, dy) for _d, _o, _s, i, name, x, y, rot, dx, dy in slides]
    return out


@dataclass(slots=True)
class _Verdict:
    """What one candidate of one reference meets on the board as it stands (each measured once)."""

    shape: Shape
    blocked: str | None  # a keep-out it touches: not free
    crowded: str | None  # another footprint's courtyard it comes within text_gap of
    nearer: str | None  # another courtyard less than OWN_MARGIN_MM farther than its own body
    edge: bool  # within PREFERRED_EDGE_MM of the board edge
    cluster: str | None  # another part's reference within CLUSTER_GAP_MM
    track: str | None  # the net of a track on its side's outer copper under it


class _RefSearch:
    """The reference candidates of one part, in a fixed order, each judged once against the board as it stands (module docstring,
    step 1). A candidate's id - ``(on a pin-label side, tier, size index, position in its list)`` - is the same for every part with the
    same footprint, rotation, side and pin-label sides, so the repeats of a cell are judged at the same relative spot."""

    def __init__(self, board: _Board, comp: Component, fp: FootprintDef, placement: Placement, body: BBox, label_sides: set[str]) -> None:
        p = board.params
        self.board, self.comp, self.fp, self.placement, self.body = board, comp, fp, placement, body
        self.side = "B" if placement.side == BoardSide.BOTTOM else "F"
        self.mirrored = self.side == "B"
        self.others = [ob for ref, ob in board.courtyards[self.side] if ref != comp.ref]
        self.own = Shape(((body.x1, body.y1), (body.x2, body.y1), (body.x2, body.y2), (body.x1, body.y2)))
        self.cands: list[tuple[tuple[bool, int, int, int], float, float, tuple]] = []
        reach = 0.0
        for on_labels in (False, True):
            if on_labels and not label_sides:
                break
            for tier, indices in enumerate((SIDE_CANDIDATES, CORNER_CANDIDATES, ROTATED_CANDIDATES)):
                for k, size in enumerate(p.reference_sizes_mm):
                    thickness = stroke_for(size)
                    w, h = text_extent(comp.ref, size, thickness)
                    reach = max(reach, w, h)
                    listed = [c for c in _shifted(body, _reference_candidates(body, w, h, p.text_gap_mm), indices) if (c[1] in label_sides) == on_labels]
                    self.cands += [((on_labels, tier, k, pos), size, thickness, c) for pos, c in enumerate(listed)]
        self.index = {cid: i for i, (cid, *_rest) in enumerate(self.cands)}
        # what can touch a candidate: every candidate lies within the body grown by the widest text and the gap
        grow = reach + 2.0 * p.text_gap_mm + max(p.silk_to_pad_mm, p.text_gap_mm, CLUSTER_GAP_MM) + 0.01
        self.area: Box = (body.x1 - grow, body.y1 - grow, body.x2 + grow, body.y2 + grow)
        self.pool = board.near(self.side, self.area)
        a = self.area
        self.tracks = [t for t in board.tracks[self.side] if not (t[1][0] > a[2] or a[0] > t[1][2] or t[1][1] > a[3] or a[1] > t[1][3])]
        # a candidate's box is a gap from its body, so a courtyard that can be as near it as the body (+ OWN_MARGIN_MM) lies in the area
        self.others = [ob for ob in self.others if not (ob.box[0] > a[2] or a[0] > ob.box[2] or ob.box[1] > a[3] or a[1] > ob.box[3])]
        self._seen: dict[int, _Verdict] = {}

    def judge(self, i: int) -> _Verdict:
        if i not in self._seen:
            _cid, size, thickness, cand = self.cands[i]
            _index, _name, x, y, rot, _dx, _dy = cand
            board, side = self.board, self.side
            shape = text_box(self.comp.ref, x, y, rot, size, thickness, "center", mirrored=self.mirrored)
            blocked = board.blocked(shape, side, pool=self.pool)
            crowded = nearer = cluster = track = None
            edge = False
            if blocked is None:
                crowded = board.blocked(shape, side, self.others, pool=[]) if self.others else None
                nearer = _nearer_part(shape, self.own, self.others)
                edge = not inside_box(shape, board.preferred)
                cluster = board.clustered(shape, side, self.comp.ref)
                track = board.on_track(shape, side, pool=self.tracks)
            self._seen[i] = _Verdict(shape, blocked, crowded, nearer, edge, cluster, track)
        return self._seen[i]


def _joint(searches: list[_RefSearch], cid: tuple[bool, int, int, int], gap: float) -> tuple[tuple, list[_Verdict], list[str | None]] | None:
    """``(key, verdicts, sibling clusters)`` of the candidate ``cid`` taken by every part of ``searches`` at once, or ``None`` when it is
    not free for one of them (a keep-out, or the text of another of them). The key orders the candidates (module docstring, step 1):
    the number of parts whose text is in another courtyard, on a pin-label side, as near another part as its own, near the edge, the
    size, the number near another reference, the tier, the number over a track, the position."""
    verdicts: list[_Verdict] = []
    for s in searches:
        i = s.index.get(cid)
        if i is None:
            return None
        v = s.judge(i)
        if v.blocked is not None:
            return None
        verdicts.append(v)
    siblings: list[str | None] = [None] * len(searches)
    for a in range(len(searches)):
        ba = verdicts[a].shape.bbox()
        for b in range(a + 1, len(searches)):
            bb = verdicts[b].shape.bbox()
            if ba[0] - CLUSTER_GAP_MM > bb[2] or bb[0] - CLUSTER_GAP_MM > ba[2] or ba[1] - CLUSTER_GAP_MM > bb[3] or bb[1] - CLUSTER_GAP_MM > ba[3]:
                continue
            d = shape_distance(verdicts[a].shape, verdicts[b].shape)
            if d < _q(gap):
                return None
            if d < CLUSTER_GAP_MM - 1e-9:
                siblings[a] = siblings[a] or f"the reference of {searches[b].comp.ref}"
                siblings[b] = siblings[b] or f"the reference of {searches[a].comp.ref}"
    on_labels, tier, k, pos = cid
    key = (
        sum(v.crowded is not None for v in verdicts), on_labels, sum(v.nearer is not None for v in verdicts), sum(v.edge for v in verdicts),
        k, sum(v.cluster is not None or c is not None for v, c in zip(verdicts, siblings)), tier, sum(v.track is not None for v in verdicts), pos,
    )
    return key, verdicts, siblings


def _best(searches: list[_RefSearch], gap: float) -> tuple[tuple, tuple, list[_Verdict], list[str | None]] | None:
    """``(cid, key, verdicts, sibling clusters)`` of the candidate with the smallest key free for every part of ``searches``."""
    best = None
    for cid, *_rest in searches[0].cands:
        got = _joint(searches, cid, gap)
        if got is not None and (best is None or got[0] < best[1]):
            best = (cid, *got)
    return best


def _why(s: _RefSearch, cid: tuple[bool, int, int, int], *, turned: bool, common: bool) -> str:
    """Why ``s`` took the turned (``turned``) or reduced candidate ``cid`` instead of a horizontal one at its size / a 1 mm one: the
    first such spot (not on a pin-label side) that was free, outside every other courtyard, and what it met."""
    p = s.board.params
    _on, _tier, k, _pos = cid
    first: _Verdict | None = None
    for i, (other, size, _thickness, cand) in enumerate(s.cands):
        o_on, o_tier, o_k, _o_pos = other
        if o_on or (turned and (o_tier == 2 or o_k != k)) or (not turned and o_k != 0):
            continue
        v = s.judge(i)
        if first is None:
            first = v
        if v.blocked is not None or v.crowded is not None:
            continue
        spot = f"the first, {cand[1]} at {size:g} mm"
        if v.nearer is not None:
            what = OWNED_TURN if turned else OWNED_SMALL
            return f"{what} ({spot}, is as near {v.nearer} as its own body)"
        if v.edge:
            return f"{EDGE_TURN if turned else EDGE_SMALL} ({spot})"
        if turned and v.cluster is not None:
            return f"{CLUSTER_TURN} ({spot}, is within {CLUSTER_GAP_MM:g} mm of {v.cluster})"
        if common:
            return COMMON_TURN if turned else COMMON_SMALL
    size0 = p.reference_sizes_mm[k if turned else 0]
    reason = (first.blocked or first.crowded) if first is not None else None
    head = f"no horizontal position is free at {size0:g} mm" if turned else f"no position is free at {size0:g} mm"
    return f"{head} (the first, above: {reason})" if reason else head


def _repeated_cells(ir: CircuitIR, board: _Board, connectors: dict[str, _ConnectorPads]) -> dict[str, tuple[tuple[str, ...], str]]:
    """``ref -> (the refs at its position in every repeat, natural order; the regions)``: the repeated cells of the board (module
    docstring, step 1) - pack regions (``ir.pcb.pack_regions``) of two parts or more whose parts are the same footprints at the same
    rotation and side in the same arrangement, one region a translation of the other (within 1 um); a connector that gets pin labels is
    left out (its labels decide its reference's sides)."""
    pcb = ir.pcb
    parts = {comp.ref: (fp.lib_id, placement) for comp, fp, placement, _body in board.parts}

    def sig(ref: str) -> tuple:
        lib_id, pl = parts[ref]
        return lib_id, round(pl.rotation_deg % 360.0, 6), pl.side.value

    regions = [r for r in (pcb.pack_regions if pcb is not None else []) if len(r.refs) >= 2 and all(ref in parts for ref in r.refs)]
    classes: list[tuple[list[str], list[tuple[str, dict[str, str]]]]] = []  # (representative's refs, [(region id, rep ref -> ref)])
    for region in regions:
        refs = sorted(region.refs, key=lambda ref: (sig(ref), parts[ref][1].x_mm, parts[ref][1].y_mm))
        for rep, members in classes:
            mapping = _translate(rep, refs, parts, sig)
            if mapping is not None:
                members.append((region.id, mapping))
                break
        else:
            classes.append((refs, [(region.id, {ref: ref for ref in refs})]))
    out: dict[str, tuple[tuple[str, ...], str]] = {}
    for rep, members in classes:
        if len(members) < 2:
            continue
        where = "pack regions " + ", ".join(rid for rid, _m in members)
        for ref in rep:
            group = tuple(sorted((m[ref] for _rid, m in members), key=natural_ref_key))
            if any(r in connectors for r in group):
                continue
            for r in group:
                out[r] = (group, where)
    return out


def _translate(rep: list[str], refs: list[str], parts: dict, sig) -> dict[str, str] | None:
    """``rep ref -> ref`` when ``refs`` is ``rep`` moved by one translation (the same signatures, every part within 1 um), else ``None``."""
    if len(rep) != len(refs) or sorted(sig(r) for r in rep) != sorted(sig(r) for r in refs):
        return None
    a = parts[rep[0]][1]
    for anchor in refs:
        if sig(anchor) != sig(rep[0]):
            continue
        b = parts[anchor][1]
        tx, ty = b.x_mm - a.x_mm, b.y_mm - a.y_mm
        mapping: dict[str, str] = {}
        free = set(refs)
        for r in rep:
            pr = parts[r][1]
            hit = next((q for q in refs if q in free and sig(q) == sig(r)
                        and abs(parts[q][1].x_mm - pr.x_mm - tx) <= 1e-3 and abs(parts[q][1].y_mm - pr.y_mm - ty) <= 1e-3), None)
            if hit is None:
                break
            mapping[r] = hit
            free.discard(hit)
        else:
            return mapping
    return None


def _place_references(ir: CircuitIR, board: _Board, out: SilkPlacement, connectors: dict[str, _ConnectorPads]) -> None:
    p = board.params
    cells = _repeated_cells(ir, board, connectors)
    by_ref = {part[0].ref: part for part in board.parts}
    done: set[str] = set()

    def search(ref: str) -> _RefSearch:
        comp, fp, placement, body = by_ref[ref]
        label_sides = connectors[ref].label_sides() if ref in connectors else _mark_sides(comp, fp, placement, body)
        return _RefSearch(board, comp, fp, placement, body, label_sides)

    for comp, _fp, _placement, _body in board.parts:
        if comp.ref in done:
            continue
        group, where = cells.get(comp.ref, ((comp.ref,), ""))
        done.update(group)
        if len(group) == 1:
            s = search(comp.ref)
            _put(s, _best([s], p.text_gap_mm), out, cell=None)
            continue
        searches = [search(ref) for ref in group]
        found = _best(searches, p.text_gap_mm)
        alone: list[_RefSearch] = []
        if found is None:
            # no spot free in every repeat: the spot free in the most of them (the first such, in candidate order) is theirs, the
            # others are placed one by one
            counts = []
            for cid, *_rest in searches[0].cands:
                free = [s for s in searches if s.index.get(cid) is not None and s.judge(s.index[cid]).blocked is None]
                counts.append((len(free), cid, free))
            most = max((c[0] for c in counts), default=0)
            subset = next((free for n, _cid, free in counts if n == most), []) if most >= 2 else []
            found = _best(subset, p.text_gap_mm) if subset else None
            while found is None and len(subset) > 2:  # their texts collide with each other at that spot: fewer of them
                subset = subset[:-1]
                found = _best(subset, p.text_gap_mm)
            if found is None:
                subset = []
            alone = [s for s in searches if s not in subset]
            searches = subset
        refs = [s.comp.ref for s in searches]
        if searches:
            cid, _key, verdicts, siblings = found
            name = searches[0].cands[searches[0].index[cid]][3][1]
            line = (f"{_stem(refs)} ({', '.join(refs)}, {where}): {name}"
                    f"{' turned' if cid[1] == 2 else ''} at {p.reference_sizes_mm[cid[2]]:g} mm in every repeat")
            note = _blocked_spot(searches, cid)
            if note:
                line += f"; {note}"
            if alone:
                line += f"; placed one by one (no spot is free in every repeat): {', '.join(s.comp.ref for s in alone)}"
            out.cells.append(line)
            for s, v, sib in zip(searches, verdicts, siblings):
                _put(s, (cid, None, [v], [sib]), out, cell=(refs, where))
        elif alone:
            out.cells.append(f"{_stem([s.comp.ref for s in alone])} ({', '.join(s.comp.ref for s in alone)}, {where}): no spot is free in "
                             "two repeats or more: placed one by one")
        for s in alone:
            s = search(s.comp.ref)  # against the texts placed since
            _put(s, _best([s], p.text_gap_mm), out, cell=None)


def _stem(refs: list[str]) -> str:
    """``R11`` ... ``R18`` -> ``R1x``: what the references of a cell position have in common, for naming it."""
    head = refs[0]
    for ref in refs[1:]:
        while not ref.startswith(head):
            head = head[:-1]
    return f"{head}x"


def _blocked_spot(searches: list[_RefSearch], cid: tuple) -> str:
    """When the common spot is not the one most repeats would take alone, which repeats that one is not free at and why (a design's
    text, a via, a pad), so a designer can free it."""
    alone = []
    for s in searches:
        got = _best([s], s.board.params.text_gap_mm)
        if got is not None:
            alone.append(got[0])
    if not alone:
        return ""
    preferred = max(set(alone), key=lambda c: (alone.count(c), [-x for x in (c[0], c[1], c[2], c[3])]))
    if preferred == cid:
        return ""
    misses, worse = [], []
    for s in searches:
        i = s.index.get(preferred)
        v = s.judge(i) if i is not None else None
        if v is None or v.blocked is not None:
            misses.append(f"{s.comp.ref} ({v.blocked if v is not None else 'no such spot'})")
            continue
        issues = [what for what, bad in (
            (f"in {v.crowded}", v.crowded), (f"as near {v.nearer} as its own body", v.nearer),
            (f"within {PREFERRED_EDGE_MM:g} mm of the board edge", v.edge), (f"within {CLUSTER_GAP_MM:g} mm of {v.cluster}", v.cluster),
            (f"over a track of {v.track}", v.track)) if bad]
        if issues:
            worse.append(f"{s.comp.ref} ({', '.join(issues)})")
    name = searches[0].cands[searches[0].index[preferred]][3][1] + (" turned" if preferred[1] == 2 else "")
    size = searches[0].cands[searches[0].index[preferred]][1]
    head = f"{name} at {size:g} mm, which {alone.count(preferred)} of {len(searches)} repeats would take alone,"
    if misses:
        return f"{head} is not free at " + "; ".join(misses)
    if worse:
        return f"{head} is " + "; ".join(worse)
    return f"{head} is not free for all of them at once (their texts would meet)"


def _put(s: _RefSearch, found: tuple | None, out: SilkPlacement, cell: tuple[list[str], str] | None) -> None:
    """Write the reference of ``s`` at the candidate ``found`` (``(cid, key, [verdict], [sibling cluster])``), or on the fab layer when
    ``found`` is ``None``, and record it (the notes, the lists of :class:`SilkPlacement`)."""
    board, comp, fp, placement, body = s.board, s.comp, s.fp, s.placement, s.body
    p, side = board.params, s.side
    if found is None:
        cx, cy = _r((body.x1 + body.x2) / 2.0), _r((body.y1 + body.y2) / 2.0)
        size = p.reference_sizes_mm[0]
        tried = len(REFERENCE_CANDIDATES) * len(p.reference_sizes_mm)
        out.texts.append(SilkText(
            text=comp.ref, x_mm=cx, y_mm=cy, rotation_deg=0.0, layer=f"{side}.Fab", size_mm=size, thickness_mm=stroke_for(size),
            justify="center", kind=SilkKind.REFERENCE, component_ref=comp.ref,
            provenance=_provenance(
                [f"component:{comp.ref}", f"footprint:{fp.lib_id}", f"placement:{placement.x_mm:g},{placement.y_mm:g},{placement.rotation_deg:g},{placement.side}",
                 f"candidates_tried:{tried}"],
                f"no free silk position for {comp.ref} among {tried} candidate(s) (nor any of them slid along its side): the reference is "
                "on the fab layer at the courtyard centre (hidden on the silk)", p,
            ),
        ))
        out.references_on_fab.append(comp.ref)
        return
    cid, _key, (v,), (sibling,) = found
    _cid, size, thickness, cand = s.cands[s.index[cid]]
    index, name, x, y, rot, dx, dy = cand
    shape, crowded, nearer, track = v.shape, v.crowded, v.nearer, v.track
    cluster = v.cluster or sibling
    common = cell is not None
    where = f"reference beside the courtyard of {comp.ref} ({name}, candidate {index} of {len(REFERENCE_CANDIDATES)} at {size:g} mm)"
    derived = [f"component:{comp.ref}", f"footprint:{fp.lib_id}", f"placement:{placement.x_mm:g},{placement.y_mm:g},{placement.rotation_deg:g},{placement.side}",
               f"candidate:{index}:{name}@{rot:g}", f"size_mm:{size:g}"]
    if dx or dy:
        where += f", slid by ({dx:g}, {dy:g}) mm along that side"
        derived.append(f"offset_mm:{dx:g},{dy:g}")
    if common:
        refs, regions = cell
        where += f"; the same spot as every repeat of its cell ({', '.join(refs)}: {regions})"
        derived.append(f"repeated_cell:{','.join(refs)}")
    if track is not None:
        where += f"; over a track of {track} (no spot clear of the tracks among the spots preferred to it)"
    if nearer is not None and crowded is None:
        where += f"; as near {nearer} as its own body (no free spot reads as its own part's)"
        out.references_ambiguous.append(f"{comp.ref}: as near {nearer}")
    if v.edge:
        gap = _edge_gap(shape, board)
        where += f"; {gap:g} mm from the board edge (no free spot keeps {PREFERRED_EDGE_MM:g} mm)"
        out.references_near_edge.append(f"{comp.ref}: {gap:g} mm")
    if cluster is not None:
        where += f"; within {CLUSTER_GAP_MM:g} mm of {cluster} (no free spot keeps clear of the other references)"
        out.references_clustered.append(f"{comp.ref}: {cluster}")
    if cid[2] > 0:
        why_small = _why(s, cid, turned=False, common=common)
        where += f"; at {size:g} mm, not {p.reference_sizes_mm[0]:g} mm: {why_small}"
        out.references_reduced.append(f"{comp.ref} ({why_small})")
    if rot:
        why_turned = _why(s, cid, turned=True, common=common)
        where += f"; turned by {rot:g} degrees: {why_turned}"
        out.references_rotated.append(f"{comp.ref} ({why_turned})")
    if crowded is not None:
        where += f"; {crowded}: no candidate outside every other footprint's courtyard"
        out.references_in_courtyard.append(f"{comp.ref}: {crowded}")
    note = f"{where}; {COPPER_NOTE}" if board.copper else where
    out.texts.append(SilkText(
        text=comp.ref, x_mm=x, y_mm=y, rotation_deg=rot, layer=f"{side}.SilkS", size_mm=size, thickness_mm=thickness,
        justify="center", kind=SilkKind.REFERENCE, component_ref=comp.ref,
        provenance=_provenance(derived, note, p),
    ))
    board.add_text(side, comp.ref, shape)
    board.references[side].append((comp.ref, shape, shape.bbox()))
    out.references_on_silk.append(comp.ref)


def _edge_gap(shape: Shape, board: _Board) -> float:
    """The gap between a text's box and the board edge, mm (rounded to 0.01)."""
    x1, y1, x2, y2 = shape.bbox()
    o = board.outline
    return round(min(x1 - o[0], y1 - o[1], o[2] - x2, o[3] - y2), 2)


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


def _connectors(ir: CircuitIR, board: _Board, keep: list[SilkText] | None = None, placement: SilkPlacement | None = None) -> dict[str, _ConnectorPads]:
    pin_net = {(pin.component_ref, pin.pin_number): net.name for net in ir.nets for pin in net.pins}
    # a pad whose net name a kept text already prints for its part gets no second label (module docstring, step 2)
    printed = {(t.component_ref, t.text) for t in keep or () if t.component_ref}
    for (ref, number), net in sorted(pin_net.items()):
        if (ref, net) in printed:
            del pin_net[(ref, number)]
            if placement is not None:
                placement.labels_on_kept.append(f"{ref}.{number} {net}")
    centre = ((board.outline[0] + board.outline[2]) / 2.0, (board.outline[1] + board.outline[3]) / 2.0)
    out: dict[str, _ConnectorPads] = {}
    for comp, fp, part_placement, body in board.parts:
        found = _connector_pads(comp, fp, part_placement, body, pin_net, centre)
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


def _polarity_pad(comp: Component, fp: FootprintDef) -> tuple[str, str, str] | str | None:
    """``(mark, pad number, pin name)`` of a two-pin part whose symbol names its pins ``A`` / ``K`` or ``+`` / ``-`` (:data:`POLARITY_MARKS`),
    checked against the footprint: the marked pin's number must be a pad of the footprint, and the footprint's own silk must not
    already carry the mark as a text; ``None`` for any other part (or one with the mark already there), else why it gets none."""
    names = sorted(pin.name for pin in comp.pins)
    for pair, pin_name, mark in POLARITY_MARKS:
        if names != sorted(pair):
            continue
        number = next(pin.number for pin in comp.pins if pin.name == pin_name)
        if not any(pad.number == number for pad in fp.pads):
            return f"its symbol's pin {pin_name!r} is number {number}, and footprint {fp.lib_id} has no pad {number}"
        own = {str(node[2]).strip() for node in fp.node if isinstance(node, list) and node and node[0] == "fp_text" and len(node) > 2
               and "SilkS" in str(next((c[1] for c in node if isinstance(c, list) and c and c[0] == "layer"), ""))}
        if mark in own:
            return None
        return mark, number, pin_name
    return None


def _mark_order(comp: Component, fp: FootprintDef, placement: Placement, body: BBox) -> list[str]:
    """The sides of the polarity mark's pad in the order step 2b tries them (away from the part's other pads first, then the two
    across, then toward them); empty for a part that gets no mark or whose mark pad's copper is not known."""
    got = _polarity_pad(comp, fp)
    if got is None or isinstance(got, str):
        return []
    _mark, number, _pin_name = got
    coppers = [c for c in pad_copper(comp.ref, fp, placement) if c.number == number and c.shape is not None]
    rest = [c for c in pad_copper(comp.ref, fp, placement) if c.number != number and c.shape is not None]
    if not coppers:
        return []
    cx = (min(c.shape.bbox()[0] for c in coppers) + max(c.shape.bbox()[2] for c in coppers)) / 2.0
    cy = (min(c.shape.bbox()[1] for c in coppers) + max(c.shape.bbox()[3] for c in coppers)) / 2.0
    if rest:
        ox = sum((c.shape.bbox()[0] + c.shape.bbox()[2]) / 2.0 for c in rest) / len(rest)
        oy = sum((c.shape.bbox()[1] + c.shape.bbox()[3]) / 2.0 for c in rest) / len(rest)
    else:
        ox, oy = (body.x1 + body.x2) / 2.0, (body.y1 + body.y2) / 2.0
    away_x = "left" if ox >= cx else "right"
    away_y = "above" if oy >= cy else "below"
    toward = {"left": "right", "right": "left", "above": "below", "below": "above"}
    if abs(ox - cx) >= abs(oy - cy):
        return [away_x, away_y, toward[away_y], toward[away_x]]
    return [away_y, away_x, toward[away_x], toward[away_y]]


def _mark_sides(comp: Component, fp: FootprintDef, placement: Placement, body: BBox) -> set[str]:
    """The reference candidate side the part's polarity mark (step 2b) takes first: its reference tries that side last, as a
    connector's reference does its pin labels' sides, so the designator does not take the mark's place (a mark finds no spot
    once a designator sits beside the pad: the 9 V piano board's D1 lost its K to its own reference, 2026-11-19)."""
    order = _mark_order(comp, fp, placement, body)
    return {order[0]} if order else set()


def _place_polarity_marks(board: _Board, out: SilkPlacement) -> None:
    """Step 2b (module docstring): a part whose symbol names its cathode (``K``) or positive (``+``) pin gets that mark beside the pad of
    that number, outside the pad's keep-out - first on the side away from the part's other pads, then the two sides across, then the
    side toward them -, at the pin label size, outside every other footprint's courtyard; a mark that finds no free spot is left out
    and named in the notes, never overlapped."""
    p = board.params
    size = p.pin_label_size_mm
    thickness = stroke_for(size)
    for comp, fp, placement, body in board.parts:
        got = _polarity_pad(comp, fp)
        if got is None:
            continue
        if isinstance(got, str):
            out.marks_skipped.append((comp.ref, got))
            continue
        mark, number, pin_name = got
        side = "B" if placement.side == BoardSide.BOTTOM else "F"
        mirrored = side == "B"
        others = [ob for ref, ob in board.courtyards[side] if ref != comp.ref]
        coppers = [c for c in pad_copper(comp.ref, fp, placement) if c.number == number and c.shape is not None]
        label = f"{comp.ref}.{number} {mark}"
        if not coppers:
            out.marks_skipped.append((label, f"pad {number}'s copper is not known exactly"))
            continue
        x1 = min(c.shape.bbox()[0] for c in coppers)
        y1 = min(c.shape.bbox()[1] for c in coppers)
        x2 = max(c.shape.bbox()[2] for c in coppers)
        y2 = max(c.shape.bbox()[3] for c in coppers)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = text_extent(mark, size, thickness)
        reach = p.silk_to_pad_mm + p.text_gap_mm
        spots = {"left": (x1 - reach - w / 2.0, cy), "right": (x2 + reach + w / 2.0, cy),
                 "above": (cx, y1 - reach - h / 2.0), "below": (cx, y2 + reach + h / 2.0)}
        order = _mark_order(comp, fp, placement, body)
        why = ""
        placed = False
        # a spot clear of the tracks on the side's outer copper first, in the same order; then the first free one over a track
        tried = [(name, *_r_xy(spots[name])) for name in order]
        free = []
        for name, x, y in tried:
            shape = text_box(mark, x, y, 0.0, size, thickness, "center", mirrored=mirrored)
            why = board.blocked(shape, side) or (board.blocked(shape, side, others) if others else None) or ""
            if not why:
                free.append((board.on_track(shape, side) is not None, name, x, y, shape))
        for over, name, x, y, shape in sorted(free, key=lambda f: f[0])[:1]:
            out.texts.append(SilkText(
                text=mark, x_mm=x, y_mm=y, rotation_deg=0.0, layer=f"{side}.SilkS", size_mm=size, thickness_mm=thickness, justify="center",
                kind=SilkKind.PIN_LABEL, component_ref=comp.ref,
                provenance=_provenance(
                    [f"component:{comp.ref}", f"pad:{comp.ref}.{number}", f"pin_name:{pin_name}",
                     f"symbol:{comp.symbol.library}:{comp.symbol.name}" if comp.symbol is not None else "symbol:-", f"footprint:{fp.lib_id}",
                     f"side:{name}", f"size_mm:{size:g}"],
                    f"polarity mark {mark!r} {name} pad {comp.ref}.{number}, the pin the symbol names {pin_name!r}: the part's polarity on the "
                    "board whatever its body or the footprint's graphics show"
                    + ("; over a track (no free spot beside the pad is clear of the tracks)" if over else ""), p,
                ),
            ))
            board.add_text(side, label, shape)
            out.marks_placed.append(label)
            placed = True
        if not placed:
            out.marks_skipped.append((label, f"no free spot beside the pad (the last tried: {why})"))


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


def _kept(ir: CircuitIR, board: _Board, keep: list[SilkText]) -> SilkText | None:
    """Make every kept template text a keep-out (module docstring, kept texts); the first kept ``title`` text, or ``None``."""
    sides = {c.ref: (p.side if (p := ir.pcb.placement(c.ref)) is not None else None) for c in ir.components}  # type: ignore[union-attr]
    problems = silk_text_problems(keep, sides)
    for i, t in enumerate(keep):
        if t.kind not in KEPT_KINDS:
            raise CompileError(f"kept silk text {t.text!r} is a {t.kind} text; only {' / '.join(sorted(KEPT_KINDS))} texts are kept (the placer adds "
                               "references and pin labels itself)")
        if i in problems:
            raise CompileError(f"kept silk text refused: {problems[i]}")
        # a placed text keeps a character's width of this text from it (module docstring, kept texts): never read as one word with it
        gap = max(board.params.text_gap_mm, KEPT_TEXT_GAP_CHARS * t.size_mm * TEXT_WIDTH_FACTOR)
        board.add_text(side_of_layer(t.layer), f"{t.kind} {t.text!r} (kept)", ir_text_box(t), gap)
    return next((t for t in keep if t.kind == SilkKind.TITLE), None)


def _restamped(t: SilkText, kept: int, kept_by: str = "template") -> SilkText:
    """A text placed on a board with ``kept`` template texts, stamped :data:`PLACER_KEEP_VERSION` (module docstring)."""
    prov = t.provenance.model_copy(update={
        "tool_version": PLACER_KEEP_VERSION, "derived_from": [*t.provenance.derived_from, f"kept_texts:{kept}"],
        "note": f"{t.provenance.note}; the {kept_by}'s {kept} kept text(s) are keep-outs too",
    })
    return t.model_copy(update={"provenance": prov})


def place_silkscreen(ir: CircuitIR, library: KicadLibrary, params: SilkParams | None = None, *, keep: list[SilkText] | None = None,
                     kept_by: str = "template") -> SilkPlacement:
    """Place references, connector pin labels and the title on the placed board of ``ir`` (module docstring); pure.

    ``keep`` (a template's ``user`` / ``title`` texts already on the board)
    stay as they are, first in ``texts``, and every placed text keeps clear of
    them (module docstring, kept texts); without it the placer works exactly
    as 0.1 did. ``kept_by`` names who put them there in the notes and the
    placed texts' provenance notes (``template``; ``design file`` for a design
    file's board texts) - it changes no position. Raises
    :class:`CompileError` when the board cannot be read: no placements or
    outline, a component without a placement / footprint, a
    footprint not on disk or malformed, a pad whose copper is unknown (a
    trapezoid, a custom pad with an unread primitive) in a footprint without
    a courtyard, a silk construct the geometry module does not read, a kept
    text that is not a ``user`` / ``title`` text or that the compiler would
    refuse. ``ir`` is never modified.
    """
    params = params or SilkParams()
    board = _Board(ir, library, params)
    out = SilkPlacement(texts=[], params=params)
    kept_title = _kept(ir, board, list(keep)) if keep else None
    connectors = _connectors(ir, board, keep, out)
    _place_references(ir, board, out, connectors)
    _place_pin_labels(board, out, connectors)
    _place_polarity_marks(board, out)
    if kept_title is None:
        _place_title(ir, board, out)
    else:
        out.title, out.title_corner = kept_title.text, f"kept where the {kept_by} put it"
        out.notes.append(f"silkscreen title: the {kept_by}'s title {kept_title.text!r} is kept where the {kept_by} put it; no second title placed")
    if keep:
        out.kept, out.kept_by = list(keep), kept_by
        out.texts = [*keep, *(_restamped(t, len(keep), kept_by) for t in out.texts)]
    if out.cells:
        out.notes.append("silkscreen: repeated cells (pack regions with the same parts in the same arrangement) - one reference spot per "
                         "cell position, the same in every repeat: " + "; ".join(out.cells))
    blocked_turns = [r for r in out.references_rotated if r.split(" (", 1)[1].startswith("no horizontal position is free at")]
    owned_turns = [r for r in out.references_rotated if r.split(" (", 1)[1].startswith(OWNED_TURN)]
    other_turns = [r for r in out.references_rotated if r not in blocked_turns and r not in owned_turns]
    if blocked_turns:
        out.notes.append("silkscreen: reference(s) turned by 90 degrees (no horizontal position free): " + "; ".join(blocked_turns))
    if owned_turns:
        out.notes.append("silkscreen: reference(s) turned by 90 degrees (every free horizontal position reads as another part's): "
                         + "; ".join(owned_turns))
    if other_turns:
        out.notes.append("silkscreen: reference(s) turned by 90 degrees (every free horizontal position is near the board edge or another "
                         "reference, or not free in every repeat of its cell): " + "; ".join(other_turns))
    if out.references_reduced:
        out.notes.append(f"silkscreen: reference(s) below {params.reference_sizes_mm[0]:g} mm (the last resort): " + "; ".join(out.references_reduced))
    if out.references_near_edge:
        out.notes.append(f"silkscreen: reference(s) nearer the board edge than {PREFERRED_EDGE_MM:g} mm (no free spot farther in): "
                         + "; ".join(out.references_near_edge))
    if out.references_clustered:
        out.notes.append(f"silkscreen: reference(s) within {CLUSTER_GAP_MM:g} mm of another part's reference (no free spot clear of them): "
                         + "; ".join(out.references_clustered))
    if out.marks_placed:
        out.notes.append("silkscreen: polarity mark(s) placed beside the pad the symbol's pin names: " + "; ".join(out.marks_placed))
    if out.marks_skipped:
        out.notes.append("silkscreen: polarity mark(s) left out (never overlapped): " + "; ".join(f"{label}: {why}" for label, why in out.marks_skipped))
    if out.references_ambiguous:
        out.notes.append("silkscreen: reference(s) as near another part as their own (no free spot reads as their own part's): "
                         + "; ".join(out.references_ambiguous))
    if out.references_in_courtyard:
        out.notes.append("silkscreen: reference(s) at another footprint's courtyard (no candidate outside every other courtyard): " + "; ".join(out.references_in_courtyard))
    if out.references_on_fab:
        out.notes.append(f"silkscreen: reference(s) on the fab layer (no free silk position): {', '.join(out.references_on_fab)}")
    if out.labels_on_kept:
        out.notes.append(f"silkscreen: pin label(s) not placed, the {kept_by}'s text already prints the net name for the part: "
                         + "; ".join(out.labels_on_kept))
    if out.labels_skipped:
        out.notes.append("silkscreen: pin label(s) skipped (never overlapped): " + "; ".join(f"{label}: {why}" for label, why in out.labels_skipped))
    return out
