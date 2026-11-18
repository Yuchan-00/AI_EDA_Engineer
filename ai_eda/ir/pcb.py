"""PCB model.

Geometry here is authored in the IR and compiled to ``.kicad_pcb``. Pad
geometry is *not* stored here - it comes from the verified KiCad footprint at
compile time.

Traceability (spec 26): every layout item (``Placement``, ``Track``, ``Via``,
``Zone``, ``SilkText``, ``Keepout``) carries a :class:`~ai_eda.ir.provenance.Provenance` so
copper and silkscreen can be traced to the tool run, the user or the model
that produced it. An item that was constructed without saying where it came
from gets :data:`UNRECORDED_ORIGIN` - an *assumption* that
``needs_verification`` - and the reviewer reports the board as NOT_VERIFIED
until someone records the origin. A tool that generates layout
(``ai_eda.tools.routing``, ``ai_eda.tools.silkscreen``) stamps ``derived``
provenance with its id and version.

Silkscreen (:class:`SilkText`, ``PCBDesign.silkscreen``) is designed content:
where a footprint's reference designator goes (a ``reference`` text replaces
the position / layer / size of that footprint's ``Reference`` property in the
compiled board - on ``F.Fab`` / ``B.Fab`` when it could not be placed on the
silk without a collision) and the board texts (``title`` / ``pin_label`` /
``user``, compiled to ``gr_text`` on ``F.SilkS`` / ``B.SilkS``). The
footprints' own silk graphics come from the library, never from here. Fields
added after IRs were saved (``PCBDesign.silkscreen``, the two silk limits of
:class:`ManufacturingConstraints`) are left out of the design view while
empty (:func:`~ai_eda.ir.provenance.drop_empty_in_design_view`), so an IR
saved before they existed keeps its ``content_hash``.

The layer stack (:class:`Stackup`, ``PCBDesign.stackup``) is design content
added the same way (out of the design view while ``None``): copper layers
top to bottom (thickness, and the net of a reference plane on a plane
layer), the dielectric between each pair (thickness, relative permittivity
at a stated frequency, optional loss tangent) and an optional solder mask,
every number ``Traced``. Impedance and propagation delay are only defined
over a reference plane; :mod:`ai_eda.tools.calc.tline` reads the geometry of
a routed layer from the stack (``line_geometry``) and says why when it
cannot.

A stack built from a named base material other than the generic FR-4
(:mod:`ai_eda.design.materials`: a high-frequency laminate, polyimide flex,
an aluminium insulated metal substrate) also carries that material
(:class:`StackMaterial`, ``Stackup.material``), each dielectric's product
name and thermal conductivity (``StackupDielectric.material`` /
``thermal_conductivity_w_mk``) and, for an insulated metal substrate, the
metal base under its one dielectric (:class:`MetalBase`,
``Stackup.metal_base``; such a stack has exactly one copper layer, ``F.Cu``,
and no through via - :meth:`Stackup.through_via_span_mm` is ``None``).
These fields were added after IRs were saved: each is left out of every
dump while ``None`` (``exclude_if``, like ``Via.kind``), so every stack
without them - every generic FR-4 stack - serialises byte for byte as
before and keeps its ``content_hash``.

Keep-outs (:class:`Keepout`, ``PCBDesign.keepouts``) are design content added
the same way (out of the design view while empty): an area of one or more
copper layers (a rectangle or a polygon, mm, every number ``Traced``) where
tracks, vias, pads, zones and / or footprints are forbidden, with the refs /
nets that are the named exceptions (an antenna's feed pad and its feed
track in an otherwise empty band) and the reason. A keep-out is a rule for
the placer, the router, the plane zones and the ``pcb.keepout`` geometry
check; it carries a provenance like every other layout item.

A track is a straight segment, or - with ``Track.mid`` - a circular arc from
``start`` through ``mid`` to ``end``, the form KiCad stores a track arc in
(:mod:`ai_eda.ir.arc` holds its geometry). ``mid`` was added after IRs were
saved: it is left out of every dump while ``None`` (``exclude_if``, like
``Via.kind``), so every board without an arc serialises byte for byte as
before and keeps its ``content_hash``. Whoever reads tracks measures an arc
as an arc - its length (:func:`track_length`), its chain of chords
(:func:`track_points`, :func:`straight_pieces`) with the stated bound, its
exact box (:func:`track_bbox`) - and never as the straight line between its
ends.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ai_eda.ir.arc import ARC_SAGITTA_MM, Point, arc_bbox, arc_length, arc_points, arc_problem
from ai_eda.ir.provenance import Provenance, ProvenanceKind, Traced, drop_empty_in_design_view

#: note of the placeholder provenance a layout item gets when nobody said where it came from
UNRECORDED_ORIGIN = "origin not recorded"


def unrecorded_origin() -> Provenance:
    """Placeholder provenance for layout authored without saying by whom / what (needs verification)."""
    return Provenance(kind=ProvenanceKind.ASSUMPTION, note=UNRECORDED_ORIGIN)


class BoardSide(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


class Layer(BaseModel):
    name: str  # KiCad layer name: "F.Cu", "B.Cu", "In1.Cu", ...
    kind: str  # "signal", "power", "mixed"


class BoardOutline(BaseModel):
    """Simple rectangular outline; extend with polygon points later."""

    width_mm: float
    height_mm: float
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0


class Placement(BaseModel):
    component_ref: str
    x_mm: float
    y_mm: float
    rotation_deg: float = 0.0
    side: BoardSide = BoardSide.TOP
    #: who / what decided this position (user, placement tool, model); unrecorded = assumption
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Track(BaseModel):
    """A track of one net on one copper layer (board frame, mm, Y down): a straight segment from ``start`` to ``end``, or - with
    ``mid`` - the circular arc from ``start`` through ``mid`` to ``end`` (KiCad's own track arc; :mod:`ai_eda.ir.arc`).

    ``mid`` ``None`` (the default) is a straight segment, and the field is then left out of every dump (``exclude_if``, like
    ``Via.kind``): a track without it serialises, hashes and compares exactly as before the field existed. The model stays as
    permissive as it was for a straight track (a non-finite end or a width that is not positive is a design error the compiler
    and the checks refuse, not a parse error); three points that are no arc (:func:`~ai_eda.ir.arc.arc_problem`: non-finite,
    coincident or on one line) are refused here, because no reader could measure them.
    """

    net: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    #: the arc's point between ``start`` and ``end``; ``None``: a straight segment
    mid: tuple[float, float] | None = Field(default=None, exclude_if=lambda v: v is None)
    width_mm: float
    #: the router / user / model that drew this segment; ``derived_from`` names the net and placements
    provenance: Provenance = Field(default_factory=unrecorded_origin)

    @model_validator(mode="after")
    def _check_arc(self) -> Track:
        if self.mid is not None:
            problem = arc_problem(self.start, self.mid, self.end)
            if problem is not None:
                raise ValueError(f"track of {self.net!r} on {self.layer}: start {self.start} / mid {self.mid} / end {self.end} is no arc: {problem}")
        return self

    @property
    def is_arc(self) -> bool:
        return self.mid is not None

    @property
    def length_mm(self) -> float:
        """The length of the centreline (:func:`track_length`)."""
        return track_length(self)

    def points(self, max_sagitta: float = ARC_SAGITTA_MM) -> list[Point]:
        """The centreline as points (:func:`track_points`): the two ends, or an arc's chain."""
        return track_points(self, max_sagitta)


def track_length(t) -> float:
    """The centreline length of a track, or of anything shaped like one (``start``, ``end`` and maybe ``mid``): the straight
    distance between its ends, or the arc's length."""
    mid = getattr(t, "mid", None)
    if mid is None:
        return math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
    return arc_length(t.start, mid, t.end)


def track_points(t, max_sagitta: float = ARC_SAGITTA_MM) -> list[Point]:
    """The centreline of a track (or of anything shaped like one) as points: its two ends, or an arc's chain
    (:func:`~ai_eda.ir.arc.arc_points`: every point of the arc within ``max_sagitta`` of the chain, and the reverse)."""
    mid = getattr(t, "mid", None)
    if mid is None:
        return [(t.start[0], t.start[1]), (t.end[0], t.end[1])]
    return arc_points(t.start, mid, t.end, max_sagitta)


def track_bbox(t) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` of a track's centreline (or of anything shaped like one), exact for an arc."""
    mid = getattr(t, "mid", None)
    if mid is None:
        return min(t.start[0], t.end[0]), min(t.start[1], t.end[1]), max(t.start[0], t.end[0]), max(t.start[1], t.end[1])
    return arc_bbox(t.start, mid, t.end)


def straight_pieces(tracks: Iterable[Track], *, inflate: bool = False, max_sagitta: float = ARC_SAGITTA_MM) -> list[Track]:
    """``tracks`` with every arc replaced by the straight chords of its chain, in order: a track without ``mid`` is returned as
    it is (the same object); an arc's chords keep its net, layer and provenance.

    With ``inflate`` the chords are ``2 * max_sagitta`` wider than the arc, so their copper covers the arc's copper: a clearance
    that holds for the chords holds for the arc. Without it they have the arc's width (the chain itself, for a drawing or for
    following connections through shared end points).
    """
    out: list[Track] = []
    for t in tracks:
        if getattr(t, "mid", None) is None:
            out.append(t)
            continue
        pts = arc_points(t.start, t.mid, t.end, max_sagitta)
        width = t.width_mm + 2.0 * max_sagitta if inflate else t.width_mm
        out.extend(Track(net=t.net, layer=t.layer, start=a, end=b, width_mm=width, provenance=t.provenance) for a, b in zip(pts, pts[1:]))
    return out


class Via(BaseModel):
    """A plated via (board frame, mm, Y down): a through via, or a typed via with a layer span.

    ``kind`` ``None`` (the default) is a **through** via: the compiler writes
    it without a type token, which KiCad builds as ``F.Cu``-``B.Cu`` whatever
    ``layers`` say (measured on kicad-cli 10.0.6), so the compiler refuses a
    ``kind``-less via whose ``layers`` are not exactly ``F.Cu`` / ``B.Cu``.
    The model itself stays permissive there: a ``kind``-less via on other
    layers is a design error the compiler refuses (``pcb.routing`` models it
    as the through via KiCad would build), not a parse error.

    ``kind`` ``blind`` / ``buried`` / ``micro`` is a typed via whose copper
    spans ``layers[0]`` (the upper end) to ``layers[1]`` (the lower end) and
    every copper layer between them (:func:`via_span`); the compiler writes
    KiCad 10's token after ``via`` (``(via blind ...``), the one KiCad 10.0.6
    keeps on load (measured). A typed via must name a legal span
    (:func:`via_span_problem`, checked here without the board's layer list
    and again by the compiler with it). ``kind`` was added after IRs were
    saved: it is left out of every dump while ``None`` (``exclude_if``), so
    an IR without a typed via serialises byte for byte as before and keeps
    its ``content_hash``.
    """

    net: str
    x_mm: float
    y_mm: float
    drill_mm: float
    diameter_mm: float
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    #: ``None`` = a through via (no KiCad type token); else KiCad 10's via type with the span ``layers`` (class docstring)
    kind: Literal["blind", "buried", "micro"] | None = Field(default=None, exclude_if=lambda v: v is None)
    provenance: Provenance = Field(default_factory=unrecorded_origin)

    @model_validator(mode="after")
    def _legal_span(self) -> Via:
        if self.kind is not None:
            top, bottom = self.layers
            problem = via_span_problem(self.kind, top, bottom)
            if problem is not None:
                raise ValueError(f"via {self.kind} {top}-{bottom}: {problem}")
        return self


#: the via kinds a :class:`Via` may carry (KiCad 10's type tokens); a via without one is a through via
VIA_KINDS: tuple[str, ...] = ("blind", "buried", "micro")
#: the outer copper layers (a blind via has exactly one of them as an end, a buried via none)
OUTER_COPPER: frozenset[str] = frozenset({"F.Cu", "B.Cu"})
#: :func:`copper_position` of ``B.Cu`` when the board's layer list is not given: above every inner index
UNKNOWN_BOTTOM_POSITION = sys.maxsize
#: the micro via rule's refusal (a stated rule of this version, see :func:`via_span_problem`)
MICRO_VIA_ADJACENT = "a micro via joins two adjacent copper layers (a stated rule of this version; KiCad 10.0.6 DRC does not check it)"


def copper_position(name: str, copper: Sequence[str] | None = None) -> int:
    """Stack position of copper layer ``name``: ``F.Cu`` 0, ``In<k>.Cu`` k, ``B.Cu`` ``len(copper) - 1``.

    Without ``copper`` (the board's copper layer names) the position of
    ``B.Cu`` is not known and is :data:`UNKNOWN_BOTTOM_POSITION`, a sentinel
    above every inner index, so the order top-before-bottom still holds.
    ``ValueError`` for a name that is not a copper layer name.
    """
    if not _COPPER_LAYER_RE.match(name):
        raise ValueError(f"{name!r} is not a copper layer name")
    if name == "F.Cu":
        return 0
    if name == "B.Cu":
        return len(copper) - 1 if copper is not None else UNKNOWN_BOTTOM_POSITION
    return int(name[2:-3])


def copper_stack_order(names: Iterable[str]) -> list[str]:
    """The copper layer names among ``names`` top to bottom (``F.Cu``, ``In1.Cu`` .. ``In<n>.Cu``, ``B.Cu``); other names are dropped."""
    copper = [n for n in dict.fromkeys(names) if _COPPER_LAYER_RE.match(n)]
    return sorted(copper, key=lambda n: copper_position(n))


def via_span_problem(kind: str, top: str, bottom: str, copper: Sequence[str] | None = None) -> str | None:
    """Why a ``kind`` via from ``top`` to ``bottom`` is not a legal span, or ``None``.

    ``copper`` is the board's copper layer names (any order); without it the
    rules that need the board are checked as far as the names allow. The
    rules, each a stated rule of this version:

    * both ends are copper layer names (``F.Cu``, ``In<k>.Cu``, ``B.Cu``), two
      different layers, the upper one first - KiCad writes a via's layers top
      first (kicad-cli 10.0.6 ``pcb upgrade`` rewrote ``In2.Cu F.Cu`` as
      ``F.Cu In2.Cu``, measured);
    * ``blind``: exactly one outer end (``F.Cu`` or ``B.Cu``); ``buried``: no
      outer end - the usual definitions (Eurocircuits, "Blind and Buried
      Vias", D. Stans, 28 Sep 2016, opened for the plan). KiCad 10.0.6 does
      not enforce them: it kept a ``blind`` token on an ``In1.Cu``-``In3.Cu``
      via (measured probe), so this version refuses what KiCad would load;
    * ``micro``: the two ends are adjacent copper layers - a laser microvia
      spans exactly one dielectric (IPC-T-50M definition as quoted by Kusiak &
      Bowles, IPC APEX 2022, and NCAB "Microvias"; the IPC text was not
      opened). Checked when ``copper`` is given, or when both ends are known
      without it (``F.Cu``-``In1.Cu``, ``In<k>.Cu``-``In<k+1>.Cu``).
      KiCad 10.0.6 DRC raised nothing on a ``micro`` via ``F.Cu``-``In2.Cu``
      (measured probe), so the rule is this version's, not KiCad's;
    * with ``copper``: both ends are copper layers of the board.

    Not checked here: whether a span is buildable by a fab's lamination
    sequence (Eurocircuits, same article: a span cuts whole cores; spans
    nest or are disjoint) - that needs the stack's core / prepreg map and
    a fab's process, and ``mfg.capability`` says the via process is not
    verified; aspect ratios and laser drill sizes - no size is assumed.
    """
    if kind not in VIA_KINDS:
        return f"{kind!r} is not a via kind ({', '.join(VIA_KINDS)})"
    for name in (top, bottom):
        if not _COPPER_LAYER_RE.match(name):
            return f"{name!r} is not a copper layer name"
    if top == bottom:
        return "both ends are the same layer"
    if copper is not None:
        for name in (top, bottom):
            if name not in copper:
                return f"{name!r} is not a copper layer of the board"
        stack = copper_stack_order(copper)
        pos_top, pos_bottom = stack.index(top), stack.index(bottom)
    else:
        pos_top, pos_bottom = copper_position(top), copper_position(bottom)
    if pos_top > pos_bottom:
        return "the upper layer comes first (KiCad writes it so)"
    outer = (top in OUTER_COPPER) + (bottom in OUTER_COPPER)
    if kind == "blind" and outer != 1:
        return "a blind via has exactly one outer end"
    if kind == "buried" and outer != 0:
        return "a buried via has no outer end"
    if kind == "micro" and (copper is not None or bottom != "B.Cu") and pos_bottom - pos_top != 1:
        return MICRO_VIA_ADJACENT
    return None


def via_span(via: Via, copper: Sequence[str]) -> list[str]:
    """The copper layers ``via``'s copper is on: every one of ``copper`` for a through via, else its span top to bottom.

    ``copper`` is the board's copper layer names **top to bottom**
    (:func:`copper_stack_order`); the span of a typed via is ``copper`` from
    its upper end to its lower end, both included (on a board with every
    inner layer, ``copper[position(top) : position(bottom) + 1]``). A span
    whose ends the board does not hold (:func:`via_span_problem` with
    ``copper``) is the caller's to refuse first; here it is a ``ValueError``.
    """
    if via.kind is None:
        return list(copper)
    top, bottom = via.layers
    return list(copper[list(copper).index(top):list(copper).index(bottom) + 1])


class Zone(BaseModel):
    net: str
    layer: str
    polygon: list[tuple[float, float]]
    #: Interior contours; absent from older IR dumps and hashes while empty.
    holes: list[list[tuple[float, float]]] = Field(default_factory=list, exclude_if=lambda v: not v)
    clearance_mm: float | None = None
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class SilkKind(StrEnum):
    """What a :class:`SilkText` is: a footprint's reference designator, the board title, a connector pin label, anything else."""

    REFERENCE = "reference"
    TITLE = "title"
    PIN_LABEL = "pin_label"
    USER = "user"


class SilkText(BaseModel):
    """One designed silkscreen text (board frame, mm, Y down).

    ``x_mm`` / ``y_mm`` is the text position KiCad stores (the anchor the
    justification refers to; vertically the text is centred on it),
    ``rotation_deg`` the stored text angle (counter-clockwise on screen),
    ``size_mm`` the glyph height = width, ``thickness_mm`` the stroke width
    (KiCad's stroke font). A text on a ``B.*`` layer is written mirrored.
    ``kind="reference"`` names the footprint in ``component_ref`` and its
    ``text`` is that reference; its layer is the silk or fab layer of the
    footprint's side (``F.Fab`` = placed on the fab drawing because no silk
    position was free). ``pin_label`` names the connector in
    ``component_ref``. The compiler refuses anything else (a layer the board
    lacks, a non-finite number, a reference to no component).
    """

    text: str
    x_mm: float
    y_mm: float
    rotation_deg: float = 0.0
    layer: str = "F.SilkS"
    size_mm: float = 1.0
    thickness_mm: float = 0.15
    justify: Literal["center", "left", "right"] = "center"
    kind: SilkKind = SilkKind.USER
    component_ref: str | None = None
    #: the placer / user / model that put the text here; unrecorded = assumption
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class ManufacturingConstraints(BaseModel):
    """Fab limits. Every value is Traced so a JLCPCB limit that was never
    confirmed stays visibly NOT_VERIFIED instead of silently becoming a rule.

    ``min_silk_text_height_mm`` / ``min_silk_line_width_mm`` (judged by
    ``pcb.silk.size``, IR geometry) were added after IRs were saved: they are
    left out of the design view while unset, so an existing IR keeps its hash.
    """

    fab: str | None = None  # "JLCPCB"
    min_track_width_mm: Traced[float] | None = None
    min_clearance_mm: Traced[float] | None = None
    min_via_drill_mm: Traced[float] | None = None
    min_via_diameter_mm: Traced[float] | None = None
    min_hole_to_edge_mm: Traced[float] | None = None
    layer_count_options: Traced[list[int]] | None = None
    copper_weight_oz: Traced[float] | None = None
    board_thickness_mm: Traced[float] | None = None
    min_silk_text_height_mm: Traced[float] | None = None
    min_silk_line_width_mm: Traced[float] | None = None

    _design = drop_empty_in_design_view("min_silk_text_height_mm", "min_silk_line_width_mm")


# --------------------------------------------------------------------------- stackup


class DielectricKind(StrEnum):
    """What a dielectric layer of the stackup is made of: a laminated core or a prepreg (bonding) sheet."""

    CORE = "core"
    PREPREG = "prepreg"


class CopperRole(StrEnum):
    """What a copper layer of the stackup carries: routed signals, or one net's reference plane."""

    SIGNAL = "signal"
    PLANE = "plane"


#: the unit each dimensional stackup number must carry (a number in another unit would be misread, so it is refused)
STACKUP_UNITS: dict[str, str] = {"copper.thickness_um": "um", "dielectric.thickness_mm": "mm", "dielectric.er_frequency_hz": "Hz", "solder_mask.thickness_um": "um",
                                 "dielectric.thermal_conductivity_w_mk": "W/mK", "metal_base.thickness_mm": "mm"}
#: the kinds of base material a stack may name (:class:`StackMaterial`): a rigid laminate, a flexible one, an insulated metal substrate
MATERIAL_KINDS: tuple[str, ...] = ("rigid", "flex", "ims")


def _absent(v: object) -> bool:
    """``exclude_if`` of the stack fields added after IRs were saved: left out of every dump while ``None``."""
    return v is None


def _check_traced(traced: Traced | None, what: str, *, unit: str | None, minimum: float, strict: bool, upper: float | None = None) -> None:
    """``ValueError`` when a stackup number carries the wrong unit or lies outside its physical range."""
    if traced is None:
        return
    if traced.unit != unit:
        want = f"unit {unit!r}" if unit is not None else "no unit (a dimensionless ratio)"
        raise ValueError(f"{what} must carry {want}, got {traced.unit!r}")
    v = traced.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {v!r}")
    if (strict and not v > minimum) or (not strict and not v >= minimum):
        raise ValueError(f"{what} must be {'>' if strict else '>='} {minimum:g}, got {v!r}")
    if upper is not None and not v < upper:
        raise ValueError(f"{what} must be < {upper:g}, got {v!r}")


class StackupCopper(BaseModel):
    """One copper layer of the stackup, top (``F.Cu``) to bottom (``B.Cu``).

    ``thickness_um`` is the finished copper thickness in micrometres.
    ``plane_net`` names the net of a reference plane on this layer (a plane
    layer: the PCB agent fills it with a zone of that net, the router does not
    route on it); ``None`` is a signal layer. The plane assignment is a design
    decision, never a fab fact: a stackup grounded on a fab page carries no
    plane until the design adds one.
    """

    name: str
    thickness_um: Traced[float]
    plane_net: Traced[str] | None = None

    @property
    def role(self) -> CopperRole:
        return CopperRole.PLANE if self.plane_net is not None else CopperRole.SIGNAL


class StackupDielectric(BaseModel):
    """One dielectric layer between two copper layers.

    ``er`` is the relative permittivity (Dk) at ``er_frequency_hz`` when the
    source states the frequency (a Dk without its frequency is kept, and a
    report says the frequency is not recorded); ``loss_tangent`` is optional
    (the lossless calculators do not use it). ``material`` names the product
    the numbers were read from (``"Rogers RO4350B"``) and
    ``thermal_conductivity_w_mk`` its through-plane thermal conductivity
    (W/mK), both only on a stack built from a named material
    (:mod:`ai_eda.design.materials`); left out of every dump while ``None``.
    """

    kind: DielectricKind
    thickness_mm: Traced[float]
    er: Traced[float]
    er_frequency_hz: Traced[float] | None = None
    loss_tangent: Traced[float] | None = None
    material: str | None = Field(default=None, exclude_if=_absent)
    thermal_conductivity_w_mk: Traced[float] | None = Field(default=None, exclude_if=_absent)


class SolderMask(BaseModel):
    """The solder mask over the outer layers (thickness over the copper, relative permittivity)."""

    thickness_um: Traced[float]
    er: Traced[float]


class MetalBase(BaseModel):
    """The metal plate under the one dielectric of an insulated metal substrate (IMS): what it is and how thick.

    ``material`` names the metal as its data sheet does
    (``"aluminium 5052 / 6061"``); ``thickness_mm`` is traced like every
    stack number. It is not a copper layer and carries no net: nothing in this
    version connects it, so it is no reference plane.
    """

    material: str
    thickness_mm: Traced[float]


class StackMaterial(BaseModel):
    """The named base material a stack was built from, when it is not the generic FR-4 (``None``: the generic FR-4, as before).

    ``id`` is the value the user answers under ``pcb_material``
    (:data:`ai_eda.design.materials.MATERIAL_IDS`), ``name`` the product,
    ``kind`` one of :data:`MATERIAL_KINDS`, ``reason`` why this material
    (the user's answer, or the selection rule with its source), ``sources``
    the documents its numbers were read from (title, revision, URL, page /
    table) and ``facts`` the data-sheet statements no calculator reads (Tg,
    breakdown voltage), each with its source. ``min_bend_ratio`` (a flex
    material only) is the minimum bend radius over the flex thickness, traced
    with the guide that states it.
    """

    id: str
    name: str
    kind: Literal["rigid", "flex", "ims"]
    reason: str
    sources: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    min_bend_ratio: Traced[float] | None = None


class Stackup(BaseModel):
    """The board's layer stack: copper layers top to bottom with the dielectric between each pair.

    ``dielectrics[i]`` lies between ``copper[i]`` and ``copper[i + 1]``.
    Every number is :class:`~ai_eda.ir.provenance.Traced`: grounded on a fab
    page (``--fab-capability`` with a ``stackup`` block,
    :mod:`ai_eda.tools.manufacturing.capability_file`), a template's
    confirmed choice (:mod:`ai_eda.design.stackup`) or the user's. The model
    refuses what it cannot mean: fewer than two copper layers, names other
    than ``F.Cu``, ``In1.Cu`` .. ``In<n>.Cu``, ``B.Cu`` in that order, a
    dielectric count that is not the copper count minus one, a dimensional
    number without its unit (:data:`STACKUP_UNITS`), a thickness that is not
    positive, a permittivity below 1 or a loss tangent outside [0, 1).
    ``provenance`` records who decided the stack as a whole (layer count,
    plane assignment).

    ``material`` (:class:`StackMaterial`) names the base material of a stack
    built from one; ``metal_base`` (:class:`MetalBase`) is the plate of an
    insulated metal substrate: such a stack has exactly one copper layer
    (``F.Cu``) and one dielectric, between it and the base (``dielectrics[0]``
    lies between ``copper[0]`` and the base). Both are left out of every dump
    while ``None``.

    Every traced number has an id (:meth:`traced_items`), e.g.
    ``pcb.stackup.dielectrics[0].er`` or
    ``pcb.stackup.copper[F.Cu].thickness_um``: a calculator that reads a
    stackup number records that id as its input, and
    :func:`~ai_eda.tools.calc.recompute.recompute_parameters` resolves it
    through :meth:`lookup`.
    """

    copper: list[StackupCopper]
    dielectrics: list[StackupDielectric]
    solder_mask: SolderMask | None = None
    provenance: Provenance = Field(default_factory=unrecorded_origin)
    material: StackMaterial | None = Field(default=None, exclude_if=_absent)
    metal_base: MetalBase | None = Field(default=None, exclude_if=_absent)

    @model_validator(mode="after")
    def _consistent(self) -> Stackup:
        names = [c.name for c in self.copper]
        if self.metal_base is not None:
            if names != ["F.Cu"]:
                raise ValueError(f"a stack on a metal base (an insulated metal substrate) has exactly one copper layer, F.Cu, in this version, got {names}")
            if len(self.dielectrics) != 1:
                raise ValueError(f"a stack on a metal base has one dielectric, between F.Cu and the base, got {len(self.dielectrics)}")
            _check_traced(self.metal_base.thickness_mm, "metal_base thickness_mm", unit="mm", minimum=0.0, strict=True)
            if not self.metal_base.material.strip():
                raise ValueError("metal_base: material must name the metal")
        else:
            if len(names) < 2:
                raise ValueError(f"a stackup needs at least two copper layers, got {names}")
            expected = ["F.Cu", *[f"In{i}.Cu" for i in range(1, len(names) - 1)], "B.Cu"]
            if names != expected:
                raise ValueError(f"copper layers must be {expected} top to bottom, got {names}")
            if len(self.dielectrics) != len(self.copper) - 1:
                raise ValueError(f"{len(self.copper)} copper layers need {len(self.copper) - 1} dielectric(s) between them, got {len(self.dielectrics)}")
        if self.material is not None:
            if self.material.kind == "ims" and self.metal_base is None:
                raise ValueError(f"material {self.material.id}: an insulated metal substrate names its metal base")
            if self.material.kind != "ims" and self.metal_base is not None:
                raise ValueError(f"material {self.material.id} ({self.material.kind}): only an insulated metal substrate has a metal base")
            _check_traced(self.material.min_bend_ratio, "material min_bend_ratio", unit=None, minimum=0.0, strict=True)
        for c in self.copper:
            _check_traced(c.thickness_um, f"copper {c.name} thickness_um", unit="um", minimum=0.0, strict=True)
            if c.plane_net is not None and (not isinstance(c.plane_net.value, str) or not c.plane_net.value.strip()):
                raise ValueError(f"copper {c.name}: plane_net must name a net, got {c.plane_net.value!r}")
        for i, d in enumerate(self.dielectrics):
            below = self.copper[i + 1].name if i + 1 < len(self.copper) else "metal base"
            what = f"dielectric {i} ({self.copper[i].name} / {below})"
            _check_traced(d.thickness_mm, f"{what} thickness_mm", unit="mm", minimum=0.0, strict=True)
            _check_traced(d.er, f"{what} er", unit=None, minimum=1.0, strict=False)
            _check_traced(d.er_frequency_hz, f"{what} er_frequency_hz", unit="Hz", minimum=0.0, strict=True)
            _check_traced(d.loss_tangent, f"{what} loss_tangent", unit=None, minimum=0.0, strict=False, upper=1.0)
            _check_traced(d.thermal_conductivity_w_mk, f"{what} thermal_conductivity_w_mk", unit="W/mK", minimum=0.0, strict=True)
            if d.material is not None and not d.material.strip():
                raise ValueError(f"{what}: material must name a product, got {d.material!r}")
        if self.solder_mask is not None:
            _check_traced(self.solder_mask.thickness_um, "solder_mask thickness_um", unit="um", minimum=0.0, strict=True)
            _check_traced(self.solder_mask.er, "solder_mask er", unit=None, minimum=1.0, strict=False)
        return self

    @property
    def layer_count(self) -> int:
        return len(self.copper)

    def copper_names(self) -> list[str]:
        return [c.name for c in self.copper]

    def copper_layer(self, name: str) -> StackupCopper | None:
        for c in self.copper:
            if c.name == name:
                return c
        return None

    def index(self, name: str) -> int:
        """Position of copper layer ``name`` (0 = ``F.Cu``); ``ValueError`` for a layer the stack does not have."""
        for i, c in enumerate(self.copper):
            if c.name == name:
                return i
        raise ValueError(f"copper layer {name!r} is not in the stackup {self.copper_names()}")

    def plane_layers(self) -> list[StackupCopper]:
        return [c for c in self.copper if c.plane_net is not None]

    def signal_layers(self) -> list[str]:
        return [c.name for c in self.copper if c.plane_net is None]

    def board_thickness_mm(self) -> float:
        """Copper plus dielectric thickness, outer copper face to outer copper face (the solder mask is not counted).

        On a metal-base stack: the copper face to the base's face (the base is
        not counted; :meth:`total_thickness_mm` adds it).
        """
        return sum(float(d.thickness_mm.value) for d in self.dielectrics) + sum(float(c.thickness_um.value) for c in self.copper) / 1000.0

    def total_thickness_mm(self) -> float:
        """:meth:`board_thickness_mm` plus the metal base of an insulated metal substrate (the same number without one)."""
        base = float(self.metal_base.thickness_mm.value) if self.metal_base is not None else 0.0
        return self.board_thickness_mm() + base

    def through_via_span_mm(self) -> float | None:
        """The barrel of a through via, ``F.Cu`` to ``B.Cu`` (:meth:`span_mm`); ``None`` on a stack with no ``B.Cu`` (one copper layer on a metal base: no through via)."""
        if self.copper[-1].name != "B.Cu":
            return None
        return self.span_mm("F.Cu", "B.Cu")

    def span_mm(self, upper: str, lower: str) -> float:
        """Distance from the top face of copper layer ``upper`` to the bottom face of ``lower`` (the plated barrel a via between them spans).

        For ``F.Cu`` to ``B.Cu`` that is :meth:`board_thickness_mm`. The two
        names may be given in either order; the same layer twice gives its own
        copper thickness.
        """
        a, b = sorted((self.index(upper), self.index(lower)))
        copper = sum(float(c.thickness_um.value) for c in self.copper[a:b + 1]) / 1000.0
        return copper + sum(float(d.thickness_mm.value) for d in self.dielectrics[a:b])

    def traced_items(self, prefix: str = "pcb.stackup") -> list[tuple[str, Traced]]:
        """``(id, traced)`` of every traced value of the stack, in stack order (the ids :meth:`lookup` resolves)."""
        out: list[tuple[str, Traced]] = []
        for c in self.copper:
            out.append((f"{prefix}.copper[{c.name}].thickness_um", c.thickness_um))
            if c.plane_net is not None:
                out.append((f"{prefix}.copper[{c.name}].plane_net", c.plane_net))
        for i, d in enumerate(self.dielectrics):
            for field in ("thickness_mm", "er", "er_frequency_hz", "loss_tangent", "thermal_conductivity_w_mk"):
                t = getattr(d, field)
                if t is not None:
                    out.append((f"{prefix}.dielectrics[{i}].{field}", t))
        if self.solder_mask is not None:
            out.append((f"{prefix}.solder_mask.thickness_um", self.solder_mask.thickness_um))
            out.append((f"{prefix}.solder_mask.er", self.solder_mask.er))
        if self.metal_base is not None:
            out.append((f"{prefix}.metal_base.thickness_mm", self.metal_base.thickness_mm))
        if self.material is not None and self.material.min_bend_ratio is not None:
            out.append((f"{prefix}.material.min_bend_ratio", self.material.min_bend_ratio))
        return out

    def served_requirements(self) -> list[str]:
        """The requirement ids the stack as a whole serves (``provenance.derived_from``: a layer count the user stated, ``req.pcb_layers``)."""
        return [rid for rid in self.provenance.derived_from if rid.startswith("req.")]

    def lookup(self, key: str, prefix: str = "pcb.stackup") -> Traced | None:
        """The traced value with id ``key`` (see :meth:`traced_items`), else ``None``."""
        if not key.startswith(prefix + "."):
            return None
        for k, t in self.traced_items(prefix):
            if k == key:
                return t
        return None


# --------------------------------------------------------------------------- keep-outs


#: what a :class:`Keepout` may forbid (a KiCad rule area calls the ``zones`` ban ``copperpour``)
KEEPOUT_ITEMS: tuple[str, ...] = ("tracks", "vias", "pads", "zones", "footprints")
#: the items a ref exception applies to, and the items a net exception applies to
_REF_ITEMS = frozenset({"footprints", "pads"})
_NET_ITEMS = frozenset({"tracks", "vias", "zones", "pads"})
#: every copper layer of the board
ALL_COPPER = "*.Cu"
_COPPER_LAYER_RE = re.compile(r"^(F\.Cu|B\.Cu|In[1-9][0-9]*\.Cu)$")
_KEEPOUT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class Keepout(BaseModel):
    """An area of copper layers where tracks / vias / pads / zones / footprints are forbidden (board frame, mm, Y down).

    ``layers`` names copper layers (``F.Cu``, ``In1.Cu`` .. ``In<n>.Cu``,
    ``B.Cu``) or is exactly ``["*.Cu"]`` (every copper layer of the board).
    The area is exactly one of ``rect`` - ``[x, y, w, h]`` in mm, the corner
    with the smallest coordinates first, ``w`` / ``h`` > 0 - and ``polygon``
    - at least three ``[x, y]`` points in mm enclosing a non-zero area; the
    traced value carries unit ``mm``. ``forbids`` lists what may not lie in
    the area (:data:`KEEPOUT_ITEMS`, no repeats). ``allowed_refs`` names the
    footprints exempt from a ``footprints`` / ``pads`` ban (the antenna in a
    no-parts band) and ``allowed_nets`` the nets exempt from a ``tracks`` /
    ``vias`` / ``zones`` / ``pads`` ban (its feed track); an exception for an
    item the keep-out does not forbid is refused. ``reason`` says why the
    area is kept out. ``id`` is a plain identifier (it names the check row
    and the compiled rule area).
    """

    id: str
    layers: list[str]
    rect: Traced[list[float]] | None = None
    polygon: Traced[list[list[float]]] | None = None
    forbids: list[Literal["tracks", "vias", "pads", "zones", "footprints"]]
    allowed_refs: list[str] = Field(default_factory=list)
    allowed_nets: list[str] = Field(default_factory=list)
    reason: str
    #: the template / user / tool that decided the keep-out; unrecorded = assumption
    provenance: Provenance = Field(default_factory=unrecorded_origin)

    @model_validator(mode="after")
    def _consistent(self) -> Keepout:
        what = f"keep-out {self.id}"
        if not _KEEPOUT_ID_RE.match(self.id):
            raise ValueError(f"keep-out id {self.id!r} must be a plain identifier (a letter, then letters, digits or _)")
        if not self.layers:
            raise ValueError(f"{what}: needs at least one copper layer")
        if ALL_COPPER in self.layers and self.layers != [ALL_COPPER]:
            raise ValueError(f"{what}: {ALL_COPPER!r} means every copper layer and stands alone, got {self.layers}")
        for layer in self.layers:
            if layer != ALL_COPPER and not _COPPER_LAYER_RE.match(layer):
                raise ValueError(f"{what}: {layer!r} is not a copper layer (F.Cu, In<n>.Cu, B.Cu or {ALL_COPPER})")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"{what}: a layer is listed twice in {self.layers}")
        if (self.rect is None) == (self.polygon is None):
            raise ValueError(f"{what}: give exactly one of rect ([x, y, w, h] mm) or polygon ([[x, y], ...] mm)")
        area = self.rect if self.rect is not None else self.polygon
        if area.unit != "mm":  # type: ignore[union-attr]
            raise ValueError(f"{what}: the area must carry unit 'mm', got {area.unit!r}")  # type: ignore[union-attr]
        if self.rect is not None:
            r = self.rect.value
            if len(r) != 4 or any(not math.isfinite(float(v)) for v in r):
                raise ValueError(f"{what}: rect must be [x, y, w, h] (four finite numbers), got {r!r}")
            if not (r[2] > 0 and r[3] > 0):
                raise ValueError(f"{what}: rect width and height must be > 0, got {r[2]!r} x {r[3]!r}")
        else:
            pts = self.polygon.value  # type: ignore[union-attr]
            if len(pts) < 3 or any(len(p) != 2 or any(not math.isfinite(float(v)) for v in p) for p in pts):
                raise ValueError(f"{what}: polygon needs at least three [x, y] points of finite numbers, got {pts!r}")
            if _shoelace([(float(p[0]), float(p[1])) for p in pts]) == 0.0:
                raise ValueError(f"{what}: polygon encloses no area")
        if not self.forbids:
            raise ValueError(f"{what}: forbids nothing ({list(KEEPOUT_ITEMS)})")
        if len(set(self.forbids)) != len(self.forbids):
            raise ValueError(f"{what}: an item is listed twice in forbids {self.forbids}")
        for label, names, items in (("allowed_refs", self.allowed_refs, _REF_ITEMS), ("allowed_nets", self.allowed_nets, _NET_ITEMS)):
            if any(not isinstance(n, str) or not n.strip() for n in names):
                raise ValueError(f"{what}: {label} entries must name something, got {names!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"{what}: a name is listed twice in {label} {names}")
            if names and not (set(self.forbids) & items):
                raise ValueError(f"{what}: {label} are exceptions to a {' / '.join(sorted(items))} ban, which this keep-out does not state (forbids {self.forbids})")
        if not self.reason.strip():
            raise ValueError(f"{what}: reason must say why the area is kept out")
        return self

    def outline(self) -> list[tuple[float, float]]:
        """The area's corner points in mm (a rect as its four corners, from ``(x, y)`` clockwise on screen)."""
        if self.rect is not None:
            x, y, w, h = (float(v) for v in self.rect.value)
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return [(float(p[0]), float(p[1])) for p in self.polygon.value]  # type: ignore[union-attr]

    def bbox(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the area in mm."""
        pts = self.outline()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def covers_layer(self, layer: str) -> bool:
        """Whether the keep-out applies on copper layer ``layer`` (``*.Cu`` covers every copper layer)."""
        if self.layers == [ALL_COPPER]:
            return bool(_COPPER_LAYER_RE.match(layer))
        return layer in self.layers


def _shoelace(pts: list[tuple[float, float]]) -> float:
    """Twice the signed area of the polygon ``pts``."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


class PackRegion(BaseModel):
    """A rectangle of the board (board frame, mm, Y down) a placer packs the named parts into.

    ``rect`` is ``[x, y, w, h]`` (traced, unit ``mm``, ``w`` / ``h`` > 0),
    ``refs`` the parts packed there (no ref twice), ``side`` the side they go
    on. Used with :attr:`PCBDesign.fixed`: the fixed parts are where the
    design put them and the parts named here are packed around them by the
    ``placement.fixed`` placer, never moved outside the region.
    """

    id: str
    rect: Traced[list[float]]
    refs: list[str]
    side: BoardSide = BoardSide.TOP
    #: the template / user that decided the region; unrecorded = assumption
    provenance: Provenance = Field(default_factory=unrecorded_origin)

    @model_validator(mode="after")
    def _consistent(self) -> PackRegion:
        what = f"pack region {self.id}"
        if not _KEEPOUT_ID_RE.match(self.id):
            raise ValueError(f"pack region id {self.id!r} must be a plain identifier (a letter, then letters, digits or _)")
        if self.rect.unit != "mm":
            raise ValueError(f"{what}: rect must carry unit 'mm', got {self.rect.unit!r}")
        r = self.rect.value
        if len(r) != 4 or any(not math.isfinite(float(v)) for v in r):
            raise ValueError(f"{what}: rect must be [x, y, w, h] (four finite numbers), got {r!r}")
        if not (r[2] > 0 and r[3] > 0):
            raise ValueError(f"{what}: rect width and height must be > 0, got {r[2]!r} x {r[3]!r}")
        if not self.refs:
            raise ValueError(f"{what}: names no part")
        if len(set(self.refs)) != len(self.refs):
            raise ValueError(f"{what}: a ref is listed twice in {self.refs}")
        return self

    def box(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the region in mm."""
        x, y, w, h = (float(v) for v in self.rect.value)
        return x, y, x + w, y + h


class PCBDesign(BaseModel):
    layers: list[Layer] = Field(default_factory=lambda: [Layer(name="F.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")])
    outline: BoardOutline | None = None
    placements: list[Placement] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    vias: list[Via] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    manufacturing: ManufacturingConstraints = Field(default_factory=ManufacturingConstraints)
    #: designed silkscreen texts (references, title, pin labels, user texts); empty = the library default positions
    silkscreen: list[SilkText] = Field(default_factory=list)
    #: the layer stack (thicknesses, permittivities, plane layers); ``None`` = not stated, and every impedance / delay
    #: number is NOT_VERIFIED "no stackup"
    stackup: Stackup | None = None
    #: areas where tracks / vias / pads / zones / footprints are forbidden (:class:`Keepout`); empty = none
    keepouts: list[Keepout] = Field(default_factory=list)
    #: parts whose position, rotation and side the design fixes (a keyboard's keys and their diodes, edge connectors):
    #: the ``placement.fixed`` placer copies them into ``placements`` verbatim; empty = no part is fixed
    fixed: list[Placement] = Field(default_factory=list)
    #: rectangles the ``placement.fixed`` placer packs the named non-fixed parts into (:class:`PackRegion`)
    pack_regions: list[PackRegion] = Field(default_factory=list)

    _design = drop_empty_in_design_view("silkscreen", "stackup", "keepouts", "fixed", "pack_regions")

    @model_validator(mode="after")
    def _unique_keepouts(self) -> PCBDesign:
        ids = [k.id for k in self.keepouts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"keep-out ids must be unique, got {ids}")
        fixed = [p.component_ref for p in self.fixed]
        if len(set(fixed)) != len(fixed):
            raise ValueError(f"a part is fixed twice: {sorted({r for r in fixed if fixed.count(r) > 1})}")
        region_ids = [r.id for r in self.pack_regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError(f"pack region ids must be unique, got {region_ids}")
        packed: dict[str, str] = {}
        for r in self.pack_regions:
            for ref in r.refs:
                if ref in packed:
                    raise ValueError(f"ref {ref!r} is in two pack regions ({packed[ref]}, {r.id})")
                if ref in fixed:
                    raise ValueError(f"ref {ref!r} is both fixed and packed in region {r.id}")
                packed[ref] = r.id
        return self

    def keepout(self, id: str) -> Keepout | None:
        return next((k for k in self.keepouts if k.id == id), None)

    def placement(self, ref: str) -> Placement | None:
        for p in self.placements:
            if p.component_ref == ref:
                return p
        return None

    def layout_items(self) -> list[tuple[str, Provenance]]:
        """``(label, provenance)`` of every placement / track / via / zone / silkscreen text / keep-out, for traceability review."""
        out: list[tuple[str, Provenance]] = [(f"placement[{p.component_ref}]", p.provenance) for p in self.placements]
        out += [(f"track[{i}:{t.net}]", t.provenance) for i, t in enumerate(self.tracks)]
        out += [(f"via[{i}:{v.net}]", v.provenance) for i, v in enumerate(self.vias)]
        out += [(f"zone[{i}:{z.net}]", z.provenance) for i, z in enumerate(self.zones)]
        out += [(f"silk[{i}:{t.kind}:{t.text}]", t.provenance) for i, t in enumerate(self.silkscreen)]
        out += [(f"keepout[{i}:{k.id}]", k.provenance) for i, k in enumerate(self.keepouts)]
        out += [(f"fixed[{p.component_ref}]", p.provenance) for p in self.fixed]
        out += [(f"pack_region[{i}:{r.id}]", r.provenance) for i, r in enumerate(self.pack_regions)]
        return out
