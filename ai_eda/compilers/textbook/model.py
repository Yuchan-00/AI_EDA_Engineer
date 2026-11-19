"""Data model of the textbook 1.0 schematic layout, and :func:`load_model`.

Every coordinate inside the layout is an integer number of 1.27 mm cells,
schematic frame (y down); millimetres appear only when a file is written.
Pin sides come from library pin angles, pin kinds from library pin types and
the pin count; nothing here reads a template, block or topology name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from ai_eda.compilers.pins import load_verified_symbol, pad_pin_types, unit_pins
from ai_eda.compilers.schematic_layout import Extent, natural_ref_key
from ai_eda.ir import CircuitIR, Component, NetKind
from ai_eda.tools.kicad.library import KicadLibrary, SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import Node

if TYPE_CHECKING:
    from ai_eda.compilers.schematic import SchematicCompiler

Cell = int
Pt = tuple[int, int]
Seg = tuple[Pt, Pt]
BoxMM = tuple[float, float, float, float]
Side = Literal["L", "R", "T", "B"]
#: ``(hard, spacing, crossings, extra_labels, bends, wire_cells, area_cells)``, compared lexicographically; the first variant wins a tie
Score = tuple[int, int, int, int, int, int, int]

#: the side a pin sits on at rotation 0, from its library angle (the direction from the pin end toward the body)
SIDE_OF_ANGLE: dict[int, Side] = {0: "L", 180: "R", 270: "T", 90: "B"}
#: unit cell step away from the body for a pin on each side
OUTWARD: dict[str, Pt] = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}
OPPOSITE: dict[str, Side] = {"L": "R", "R": "L", "T": "B", "B": "T"}


class Kind(str, Enum):
    TWO = "two"  # exactly two pins
    THRU3 = "thru3"  # two pins on opposite sides (the through pair) and one perpendicular (the control)
    AMP = "amp"  # one output, two inputs on the opposite side, every other pin on top / bottom
    PWRUNIT = "pwrunit"  # every pin is power_in
    IC = "ic"  # any other glyph with more than three pins
    CONN = "conn"  # library name starts with Connector
    OTHER = "other"


class Role(str, Enum):
    SUPPLY = "supply"  # a power net that is not NEGATIVE
    NEGATIVE = "negative"  # a power net whose amplifier / IC pins are all bottom-side power inputs
    GROUND = "ground"
    SIGNAL = "signal"


class LayoutFailure(Exception):
    """A layout reason (no room, no clear path, a recipe that does not fit). Internal: it sends the drawing one rung down
    the fallback ladder and never leaves :func:`ai_eda.compilers.textbook.pipeline.build`. IR and library faults stay
    :class:`~ai_eda.errors.CompileError`."""

    def __init__(self, reason: str, block: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.block = block


class Hit(str):
    """One guard finding: its text, the glyph keys / label tags it involves, and whether it is a hard fault (a spacing
    finding is scored, not refused)."""

    owners: tuple[str, ...]
    hard: bool

    def __new__(cls, text: str, owners: tuple[str, ...] = (), hard: bool = True) -> Hit:
        hit = super().__new__(cls, text)
        hit.owners = tuple(owners)
        hit.hard = hard
        return hit


@dataclass(frozen=True, slots=True)
class PinGeo:
    """A library pin of one unit: ``side`` and the offset ``(dx, dy)`` from the symbol origin in cells, at rotation 0."""

    number: str
    name: str
    etype: str
    unit: int
    hidden: bool
    side: Side
    dx: int
    dy: int


@dataclass(frozen=True, slots=True)
class Orient:
    rot: int = 0
    mirror: str | None = None  # None | "x" | "y", applied after the rotation (KiCad's order)


@dataclass(frozen=True, slots=True)
class Place:
    x: Cell
    y: Cell
    orient: Orient = Orient()
    field_side: Side = "R"

    def moved(self, dx: int, dy: int) -> Place:
        return Place(self.x + dx, self.y + dy, self.orient, self.field_side)


@dataclass(frozen=True, eq=False)
class Glyph:
    """One drawn unit of one part. ``key`` is the ref (``R1``) or ``ref:unit`` (``U1:2``) for a multi-unit symbol."""

    key: str
    ref: str
    unit: int
    component: Component
    symbol: SymbolDef
    footprint_id: str
    kind: Kind
    pins: tuple[PinGeo, ...]
    net: dict[str, str]  # pin number -> net, the unit's connected pins only
    dc_blocking: bool  # the library's default Reference is C: no DC path through it

    def pin(self, number: str) -> PinGeo:
        for p in self.pins:
            if p.number == number:
                return p
        raise KeyError(f"{self.key} has no pin {number!r}")

    def lib_pin(self, number: str) -> SymbolPin:
        for p in self.symbol.pins:
            if p.number == number and p.unit in (0, self.unit):
                return p
        raise KeyError(f"{self.key} has no library pin {number!r}")

    def nets(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.net.values())))

    def _on(self, *sides: str) -> list[PinGeo]:
        return [p for p in self.pins if p.side in sides and not mounting(p)]

    def through(self) -> tuple[PinGeo, PinGeo] | None:
        """THRU3: the two pins on opposite sides, top (or left) first."""
        if self.kind is not Kind.THRU3:
            return None
        for a, b in (("T", "B"), ("L", "R")):
            pa, pb = self._on(a), self._on(b)
            if len(pa) == 1 and len(pb) == 1:
                return pa[0], pb[0]
        return None

    def control(self) -> PinGeo | None:
        """THRU3: the pin perpendicular to the through pair."""
        pair = self.through()
        return None if pair is None else next(p for p in self.pins if p not in pair and not mounting(p))

    def amp_io(self) -> tuple[PinGeo, PinGeo, PinGeo] | None:
        """AMP: ``(output, upper input, lower input)`` at rotation 0."""
        if self.kind is not Kind.AMP:
            return None
        out = next(p for p in self.pins if p.etype == "output")
        ins = sorted((p for p in self.pins if p.etype == "input" and p.side == OPPOSITE[out.side]), key=lambda p: (p.dy, p.dx))
        return out, ins[0], ins[1]

    def power_pins(self) -> tuple[PinGeo, ...]:
        return tuple(p for p in self.pins if p.etype in ("power_in", "power_out"))

    @property
    def shown_ref(self) -> str:
        """The Reference as KiCad draws it: the ref with the unit letter for a unit of a multi-unit symbol (``U1A``, ``U1B``),
        the ref alone otherwise. The file stores the ref; only measurement uses this."""
        return self.ref if len(self.symbol.units) == 1 else self.ref + unit_letter(self.unit)


MOUNTING = frozenset({"MountPin"})  # library pin names of a mechanical mounting pin (a potentiometer's or connector's tab)


def mounting(pin: PinGeo) -> bool:
    """A mechanical mounting pin: it is wired like any pin (often to ground) but takes no part in the symbol's kind or in
    its through pair and control pin."""
    return pin.name in MOUNTING


def unit_letter(unit: int) -> str:
    """The unit suffix KiCad shows by default: 1 -> A, 2 -> B .. 26 -> Z (A to C checked in a kicad-cli render); past Z the
    letters go on as AA, AB .. (not checked)."""
    out = ""
    while unit > 0:
        unit, rest = divmod(unit - 1, 26)
        out = chr(ord("A") + rest) + out
    return out


@dataclass(frozen=True, slots=True)
class NetInfo:
    name: str
    role: Role
    pins: tuple[tuple[str, str], ...]  # (glyph key, pin number), natural order


@dataclass(frozen=True, slots=True)
class BlockInfo:
    id: str
    members: tuple[str, ...]  # glyph keys, natural order
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    declared: bool = False
    inferred: bool = False
    parent: str | None = None
    layer: int = 0


@dataclass(frozen=True, eq=False)
class Model:
    """The validated design as the layout sees it. ``owner`` and ``stacked`` are keyed by ``(ref, pin number)`` like the
    compiler's own maps; ``notes`` is the one list every stage appends to."""

    project_id: str
    glyphs: dict[str, Glyph]
    nets: dict[str, NetInfo]
    owner: dict[tuple[str, str], str]
    stacked: dict[tuple[str, str], tuple[str, str]]
    blocks: tuple[BlockInfo, ...] = ()
    rails: tuple[str, ...] = ()
    ir: CircuitIR | None = None
    flags: tuple[tuple[str, str], ...] = ()  # (net, flag ref): nets the existing PWR_FLAG rule powers from outside the sheet
    notes: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # part ref -> the glyph key that draws it (a combination's outline)
    outlines: dict[str, str] = field(default_factory=dict)  # a combination's outline glyph key -> its parts' library symbol

    def glyph_of(self, ref: str, number: str) -> Glyph:
        """The glyph (unit) that draws pin ``number`` of part ``ref``."""
        for g in self.glyphs.values():
            if g.ref == ref and any(p.number == number for p in g.pins):
                return g
        raise KeyError(f"{ref}.{number}")


@dataclass(frozen=True, slots=True)
class Port:
    net: str
    side: Side
    at: Pt
    role: Literal["in", "out", "slot", "rail"]


@dataclass(frozen=True, slots=True)
class RailSpan:
    net: str
    y: Cell
    x0: Cell
    x1: Cell
    source: str | None = None  # "ref.pin" a derived rail starts at


@dataclass(frozen=True, eq=False)
class Generated:
    """A glyph the layout adds: ``#PWR`` power symbols (``GND`` / ``VCC`` / ``NEG``, Value = the net name) and ``#FLG``."""

    ref: str
    lib: str  # "power:GND"
    net: str
    at: Pt
    orient: Orient
    symbol: SymbolDef
    value: str
    kind: Literal["GND", "VCC", "NEG", "FLAG"]

    @property
    def names_net(self) -> bool:
        """A power symbol names its net by its Value; PWR_FLAG does not."""
        return self.kind != "FLAG"

    def moved(self, dx: int, dy: int) -> Generated:
        return Generated(self.ref, self.lib, self.net, (self.at[0] + dx, self.at[1] + dy), self.orient, self.symbol, self.value, self.kind)


@dataclass(frozen=True, slots=True)
class Label:
    """A global label (shape passive). ``stub`` is the 2-cell wire from a pin or wire point to ``at`` it adds, if any."""

    net: str
    at: Pt
    angle: int
    justify: str
    kind: Literal["name", "terminal", "pair", "pin"]
    stub: Seg | None = None

    def moved(self, dx: int, dy: int) -> Label:
        stub = None if self.stub is None else tuple((p[0] + dx, p[1] + dy) for p in self.stub)
        return Label(self.net, (self.at[0] + dx, self.at[1] + dy), self.angle, self.justify, self.kind, stub)


@dataclass(frozen=True)
class Drawing:
    """One block (or sub-block) as a recipe drew it, block-relative. ``hints`` are wires the recipe wants as drawn (stubs,
    rail lines, the cross-coupling X); ``box`` (cells) holds every measured body, field, label and hint."""

    block: str
    recipe: str
    place: dict[str, Place]
    hints: dict[str, tuple[Seg, ...]] = field(default_factory=dict)
    rails: tuple[RailSpan, ...] = ()
    ports: tuple[Port, ...] = ()
    notes: tuple[str, ...] = ()
    sketch: tuple[int, int] = (0, 0)  # (crossings, length) of the recipe's own wiring estimate
    glyphs: tuple[Generated, ...] = ()
    labels: tuple[Label, ...] = ()
    box: tuple[int, int, int, int] | None = None
    fields: dict[str, tuple[Pt, Pt]] = field(default_factory=dict)  # glyph key -> (Reference, Value) anchors; else Place.field_side


@dataclass(frozen=True, slots=True)
class Link:
    net: str
    a: Pt
    b: Pt
    kind: Literal["wire", "labels"]
    segs: tuple[Seg, ...] = ()


@dataclass(frozen=True)
class Sheet:
    """Drawings tiled on one sheet: drawing ``i`` is shifted by ``offset[i]``. ``glyphs`` / ``links`` / ``rails`` are sheet-level
    (absolute). ``paper`` is ``(name, width, height)`` once known."""

    drawings: tuple[Drawing, ...]
    offset: tuple[Pt, ...]
    bands: tuple[tuple[int, ...], ...] = ()
    rails: tuple[RailSpan, ...] = ()
    glyphs: tuple[Generated, ...] = ()
    links: tuple[Link, ...] = ()
    paper: tuple[str, float, float] | None = None
    notes: tuple[str, ...] = ()

    def places(self) -> dict[str, Place]:
        out: dict[str, Place] = {}
        for d, (dx, dy) in zip(self.drawings, self.offset):
            for key, p in d.place.items():
                out[key] = p.moved(dx, dy)
        return out

    def hints(self) -> dict[str, list[Seg]]:
        out: dict[str, list[Seg]] = {}
        for d, (dx, dy) in zip(self.drawings, self.offset):
            for net, segs in d.hints.items():
                out.setdefault(net, []).extend(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in segs)
        for link in self.links:
            if link.kind == "wire":
                out.setdefault(link.net, []).extend(link.segs)
        return out

    def generated(self) -> list[Generated]:
        out = [g.moved(dx, dy) for d, (dx, dy) in zip(self.drawings, self.offset) for g in d.glyphs]
        return out + list(self.glyphs)

    def labels(self) -> list[Label]:
        return [lab.moved(dx, dy) for d, (dx, dy) in zip(self.drawings, self.offset) for lab in d.labels]

    def fields(self) -> dict[str, tuple[Pt, Pt]]:
        out: dict[str, tuple[Pt, Pt]] = {}
        for d, (dx, dy) in zip(self.drawings, self.offset):
            for key, (r, v) in d.fields.items():
                out[key] = ((r[0] + dx, r[1] + dy), (v[0] + dx, v[1] + dy))
        return out

    def drawing_of(self) -> dict[str, int]:
        """Glyph key / generated ref -> index of the drawing that holds it (sheet-level glyphs are absent)."""
        out: dict[str, int] = {}
        for i, d in enumerate(self.drawings):
            out.update({k: i for k in d.place})
            out.update({g.ref: i for g in d.glyphs})
        return out


@dataclass(slots=True)
class Wiring:
    segments: dict[str, list[Seg]] = field(default_factory=dict)
    junctions: dict[str, list[Pt]] = field(default_factory=dict)
    failed: dict[str, tuple[frozenset[Pt], ...]] = field(default_factory=dict)  # net -> its stranded groups of points
    crossings: int = 0


@dataclass(frozen=True, slots=True)
class DrawResult:
    node: Node
    notes: list[str]
    fallbacks: tuple[str, ...]
    score: Score | None  # None when the whole-sheet table was written


# --------------------------------------------------------------------------- loading


def _load_symbol(c: Component, library: KicadLibrary) -> SymbolDef:
    """The one place the layout loads a part's library symbol: multi-unit symbols are accepted (one glyph per unit), a
    multi-unit symbol with common (unit 0) pins is refused with the library reason."""
    return load_verified_symbol(c, library, multi_unit=True)


def _units(symbol: SymbolDef) -> list[int]:
    return list(symbol.units)


def _side(pin: SymbolPin) -> Side:
    angle = int(round(pin.angle)) % 360
    if angle not in SIDE_OF_ANGLE:
        raise ValueError(f"pin {pin.number} angle {pin.angle}")
    return SIDE_OF_ANGLE[angle]


def classify(symbol: SymbolDef, pins: tuple[PinGeo, ...], library_name: str) -> Kind:
    """Kind of one unit from its library name prefix, pin count, pin sides and pin types. An amplifier unit of a multi-unit
    symbol (MCP6002 / LM358 units A and B: output, '-', '+') has three pins and is AMP, never THRU3 or OTHER. A mounting
    pin (library pin name ``MountPin``) does not count: a potentiometer with one is THRU3."""
    if library_name.startswith("Connector"):
        return Kind.CONN
    electrical = tuple(p for p in pins if not mounting(p))
    if len(electrical) >= 2 and len(electrical) < len(pins):
        pins = electrical  # a part with a mounting pin is the kind of its electrical pins
    if pins and all(p.etype == "power_in" for p in pins):
        return Kind.PWRUNIT
    if len(pins) == 2:
        return Kind.TWO
    outputs = [p for p in pins if p.etype == "output"]
    if len(outputs) == 1 and outputs[0].side in ("L", "R"):
        out = outputs[0]
        ins = [p for p in pins if p.etype == "input" and p.side == OPPOSITE[out.side]]
        rest = [p for p in pins if p is not out and p not in ins]
        if len(ins) == 2 and all(p.side in ("T", "B") for p in rest):
            return Kind.AMP
    if len(pins) == 3:
        sides = [p.side for p in pins]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if OPPOSITE[sides[i]] == sides[j]:
                rest = sides[3 - i - j]
                if rest not in (sides[i], sides[j]) and sum(OPPOSITE[s] == t for s in sides for t in sides) == 2:
                    return Kind.THRU3
        return Kind.OTHER
    return Kind.IC if len(pins) > 3 else Kind.OTHER


def is_test_point(g: Glyph) -> bool:
    """A test point: a ``Connector*`` library's ``TestPoint*`` symbol with one pin (KiCad's ``Connector:TestPoint``). The layout
    leaves its pin out of the net's pins (:attr:`NetInfo.pins`): a pad hung on a net is no part of the stage the net belongs
    to, and every motif that reads a net's pins (a supply chain's line, an amplifier's follower loop) would see a stray part.
    It is drawn in a block of its own on a label or power symbol of its net (:mod:`~ai_eda.compilers.textbook.blocks`)."""
    sym = g.component.symbol
    return (g.kind is Kind.CONN and sym is not None and sym.name.startswith("TestPoint")
            and len({p.number for p in g.symbol.pins}) == 1)


def load_model(compiler: SchematicCompiler, ir: CircuitIR, library: KicadLibrary) -> Model:
    """Validate ``ir`` exactly as the legacy compiler does (components, library symbols and pin types, stacked pins, nets)
    and describe it for the layout. IR and library faults raise CompileError."""
    from ai_eda.compilers.schematic import _PlacedSymbol

    placed: dict[str, _PlacedSymbol] = {}  # one instance per unit, keyed like the glyphs
    for c in compiler._checked_components(ir):
        symbol = _load_symbol(c, library)
        pad_pin_types(c, symbol)  # pin numbers + electrical types must be the library's
        fp = "" if symbol.is_power else compiler._footprint_id(c, library)
        units = _units(symbol)
        for unit in units:
            placed[c.ref if len(units) == 1 else f"{c.ref}:{unit}"] = _PlacedSymbol(c, symbol, fp, 0.0, 0.0, Extent(0, 0, 0, 0), unit=unit)
    owner = compiler._pin_net_names(ir)
    pins = compiler._pin_map(placed)
    stacked = compiler._stacks(placed, pins, owner)  # per unit: pins of different units never stack
    compiler._connectivity(ir, placed, pins, stacked)  # nets: names, unknown parts / pins, a pin in two nets, open pins
    glyphs: dict[str, Glyph] = {}
    for key, ps in placed.items():
        ref, unit = ps.component.ref, ps.unit
        geo = tuple(
            PinGeo(p.number, p.name, p.electrical_type, p.unit, p.hidden, _side(p), round(p.x / 1.27), -round(p.y / 1.27))
            for p in sorted(unit_pins(ps.symbol, unit), key=lambda p: natural_ref_key(p.number))
        )
        glyphs[key] = Glyph(
            key=key, ref=ref, unit=unit, component=ps.component, symbol=ps.symbol, footprint_id=ps.footprint_id,
            kind=classify(ps.symbol, geo, ps.component.symbol.library),
            pins=geo, net={p.number: owner[(ref, p.number)] for p in geo if (ref, p.number) in owner},
            dc_blocking=ps.symbol.properties.get("Reference") == "C",
        )
    key_of = {(g.ref, p.number): g.key for g in glyphs.values() for p in g.pins}
    pads = {g.ref for g in glyphs.values() if is_test_point(g)}  # drawn on a label / power symbol of their net, not one of its pins (is_test_point)
    nets: dict[str, NetInfo] = {}
    for net in sorted(ir.nets, key=lambda n: n.name):
        members = sorted({(key_of[(p.component_ref, p.pin_number)], p.pin_number) for p in net.pins if p.component_ref not in pads},
                         key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1])))
        nets[net.name] = NetInfo(net.name, _role(net.kind, members, glyphs), tuple(members))
    drawn, flag_parts, flag_notes = compiler._power_flags(ir, library)
    flagged = {p.component_ref: n.name for n in drawn.nets for p in n.pins if p.component_ref.startswith("#")}
    flags = tuple((flagged[c.ref], c.ref) for c in flag_parts)
    return Model(project_id=ir.project.id, glyphs=glyphs, nets=nets, owner=owner, stacked=stacked, ir=ir, flags=flags,
                 notes=list(flag_notes))


def _role(kind: NetKind, members: list[tuple[str, str]], glyphs: dict[str, Glyph]) -> Role:
    if kind == NetKind.GROUND:
        return Role.GROUND
    if kind != NetKind.POWER:
        return Role.SIGNAL
    big = [glyphs[k].pin(n) for k, n in members if glyphs[k].kind in (Kind.AMP, Kind.IC, Kind.PWRUNIT)]
    if big and all(p.etype == "power_in" and p.side == "B" for p in big):
        return Role.NEGATIVE
    return Role.SUPPLY
