"""PCB model.

Geometry here is authored in the IR and compiled to ``.kicad_pcb``. Pad
geometry is *not* stored here - it comes from the verified KiCad footprint at
compile time.

Traceability (spec 26): every layout item (``Placement``, ``Track``, ``Via``,
``Zone``, ``SilkText``) carries a :class:`~ai_eda.ir.provenance.Provenance` so
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
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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
    net: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float
    #: the router / user / model that drew this segment; ``derived_from`` names the net and placements
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Via(BaseModel):
    net: str
    x_mm: float
    y_mm: float
    drill_mm: float
    diameter_mm: float
    layers: tuple[str, str] = ("F.Cu", "B.Cu")
    provenance: Provenance = Field(default_factory=unrecorded_origin)


class Zone(BaseModel):
    net: str
    layer: str
    polygon: list[tuple[float, float]]
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
STACKUP_UNITS: dict[str, str] = {"copper.thickness_um": "um", "dielectric.thickness_mm": "mm", "dielectric.er_frequency_hz": "Hz", "solder_mask.thickness_um": "um"}


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
    (the lossless calculators do not use it).
    """

    kind: DielectricKind
    thickness_mm: Traced[float]
    er: Traced[float]
    er_frequency_hz: Traced[float] | None = None
    loss_tangent: Traced[float] | None = None


class SolderMask(BaseModel):
    """The solder mask over the outer layers (thickness over the copper, relative permittivity)."""

    thickness_um: Traced[float]
    er: Traced[float]


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

    @model_validator(mode="after")
    def _consistent(self) -> Stackup:
        names = [c.name for c in self.copper]
        if len(names) < 2:
            raise ValueError(f"a stackup needs at least two copper layers, got {names}")
        expected = ["F.Cu", *[f"In{i}.Cu" for i in range(1, len(names) - 1)], "B.Cu"]
        if names != expected:
            raise ValueError(f"copper layers must be {expected} top to bottom, got {names}")
        if len(self.dielectrics) != len(self.copper) - 1:
            raise ValueError(f"{len(self.copper)} copper layers need {len(self.copper) - 1} dielectric(s) between them, got {len(self.dielectrics)}")
        for c in self.copper:
            _check_traced(c.thickness_um, f"copper {c.name} thickness_um", unit="um", minimum=0.0, strict=True)
            if c.plane_net is not None and (not isinstance(c.plane_net.value, str) or not c.plane_net.value.strip()):
                raise ValueError(f"copper {c.name}: plane_net must name a net, got {c.plane_net.value!r}")
        for i, d in enumerate(self.dielectrics):
            what = f"dielectric {i} ({self.copper[i].name} / {self.copper[i + 1].name})"
            _check_traced(d.thickness_mm, f"{what} thickness_mm", unit="mm", minimum=0.0, strict=True)
            _check_traced(d.er, f"{what} er", unit=None, minimum=1.0, strict=False)
            _check_traced(d.er_frequency_hz, f"{what} er_frequency_hz", unit="Hz", minimum=0.0, strict=True)
            _check_traced(d.loss_tangent, f"{what} loss_tangent", unit=None, minimum=0.0, strict=False, upper=1.0)
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
        """Copper plus dielectric thickness, outer copper face to outer copper face (the solder mask is not counted)."""
        return sum(float(d.thickness_mm.value) for d in self.dielectrics) + sum(float(c.thickness_um.value) for c in self.copper) / 1000.0

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
            for field in ("thickness_mm", "er", "er_frequency_hz", "loss_tangent"):
                t = getattr(d, field)
                if t is not None:
                    out.append((f"{prefix}.dielectrics[{i}].{field}", t))
        if self.solder_mask is not None:
            out.append((f"{prefix}.solder_mask.thickness_um", self.solder_mask.thickness_um))
            out.append((f"{prefix}.solder_mask.er", self.solder_mask.er))
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

    _design = drop_empty_in_design_view("silkscreen", "stackup")

    def placement(self, ref: str) -> Placement | None:
        for p in self.placements:
            if p.component_ref == ref:
                return p
        return None

    def layout_items(self) -> list[tuple[str, Provenance]]:
        """``(label, provenance)`` of every placement / track / via / zone / silkscreen text, for traceability review."""
        out: list[tuple[str, Provenance]] = [(f"placement[{p.component_ref}]", p.provenance) for p in self.placements]
        out += [(f"track[{i}:{t.net}]", t.provenance) for i, t in enumerate(self.tracks)]
        out += [(f"via[{i}:{v.net}]", v.provenance) for i, v in enumerate(self.vias)]
        out += [(f"zone[{i}:{z.net}]", z.provenance) for i, z in enumerate(self.zones)]
        out += [(f"silk[{i}:{t.kind}:{t.text}]", t.provenance) for i, t in enumerate(self.silkscreen)]
        return out
