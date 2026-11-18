"""IR -> ``.kicad_sch`` (KiCad 10, flat single sheet) compiler.

Invariants this module enforces (CLAUDE.md #1 and #2):

* **Nothing is guessed.** Every ``Component.symbol`` is resolved against the
  installed KiCad libraries at compile time; a symbol that is not on disk, an
  IR pin set that differs from the library pin set *in number or electrical
  type* (:mod:`ai_eda.compilers.pins`, shared with the PCB compiler), a net
  that names an unknown pin, or a pin left out of every net (unless it is
  ``no_connect``) raises :class:`CompileError` instead of producing a
  plausible-looking file.
* **Byte-determinism.** Every UUID comes from :mod:`ai_eda.compilers.ids`,
  components / nets / pins are emitted in sorted order, placement is a pure
  function of the reference designators, the library geometry and the net
  names, and no timestamp or path is written. The same IR always compiles to
  the same bytes, so ``ArtifactRef.content_hash`` is meaningful and
  "regenerate" is a real repair.
* **Library truth is embedded verbatim.** ``lib_symbols`` holds the library's
  own ``(symbol ...)`` tree renamed ``"Lib:Name"`` (what KiCad itself embeds);
  kicad-cli compares it with the global library and warns on any deviation.
* **No accidental connections.** KiCad connects by exact coordinate
  equality, so the column widths and row heights of the symbol table are
  derived from the measured extent of every symbol (body, pins, stubs,
  label text - :func:`~ai_eda.compilers.schematic_layout.symbol_extent`,
  :func:`~ai_eda.compilers.schematic_layout.layout_positions`), the sheet
  is the smallest ISO size that holds them
  (:func:`~ai_eda.compilers.schematic_layout.layout_paper`), and the compiler
  refuses a layout in which any two connection points (pin ends, stub ends)
  coincide, a connection point lies on another stub, or two extents touch.
  Stacked pins (two pins of one symbol at the same point, like the hidden
  VCC / GND pins of the ATmega128) are accepted only when the IR puts every
  pin at that point in the same net and they point the same way; the point
  then gets one stub and one label. Stacked pins in different nets are
  refused like any other coinciding points.
* **Fields where the library puts them.** Each instance's Reference and
  Value field is written at the library symbol's own field position turned
  by the instance's rotation / mirror, with the library's angle and
  justification - what KiCad does when it places a symbol (the
  ATmega128-16A's value below its body, not across its pin names). Only a
  library field without a position falls back to a fixed offset right of
  the origin. Fields are text, not connection points; they are not part of
  the measured extents.

Connectivity model: each pin of each net gets a 2.54 mm wire stub leaving the
pin away from the symbol body and a global label at the stub end named after
the net. Same-named global labels form one net, so the exported netlist
reproduces the IR nets exactly whether or not a wire is drawn (tested against
kicad-cli with tall connectors that would overlap on a fixed 25.4 mm grid).
Labels use shape ``passive`` so they add no electrical semantics of their
own: ERC pin-to-pin / driver checks are decided by the real pins only.

Wires (B11, additive): symbols are placed in topology order (``ir.topology``
blocks in sequence, their members in natural ref order, the rest after) so
connected symbols sit in neighbouring cells, and every net of 2 .. 4 pins is
also drawn as a trunk in a table gutter with a branch to each stub end and
junctions where branches meet (:mod:`ai_eda.compilers.schematic_route`). A
net whose path is blocked keeps stubs + labels only and the artifact note
says so; the labels stay on every pin, so a reader can follow the copper and
the netlist, ERC and the hash-based staleness rules are unchanged. A wire can
never create a connection the IR does not state: the router refuses any
segment through a foreign pin, label anchor or wire end, and
``_check_wires`` re-checks the final wires (CompileError, never a silent merge).
Two nets never share a track, so a gutter holds only a few wires (three in the
default 5.08 mm): when nets stay label-only while channels have no track left,
the compiler widens exactly those gutters and margin corridors, draws the
sheet a second time and keeps it when more nets are wired (B11 W3; the note
says so). A sheet whose wires fit is drawn once, on the same coordinates as
before.

Power flags (B12): a net that has a power-input pin and otherwise only
power-input, input and passive pins (a supply that enters through a connector,
a rail behind a ferrite) gets one ``power:PWR_FLAG`` - the library's own
symbol, resolved like every other - so ERC's ``power_pin_not_driven`` is
answered the way KiCad expects. A net with any other pin type is left alone:
there KiCad reports the flag's power output against that pin (``pin_to_pin``,
measured with an ``output`` pin on kicad-cli 10.0.6). Each flag is
a one-pin cell after the design's symbols with its own stub and global label
(never routed), its ``#FLG`` reference keeps it out of the netlist, the BOM
and the board, and the artifact note names the nets, because a flag is the
compiler's statement that the net is powered from outside the sheet.

The opt-in textbook policy (compiler 0.9) is dispatched separately for small
circuits; see :mod:`ai_eda.compilers.schematic_textbook`. The default legacy
policy and its output remain version 0.8.

What the legacy policy does *not* carry yet: multi-unit symbols, ``power:*`` rail
symbols (``GND``, ``+5V``), and sheet hierarchy (each refused with
:class:`CompileError` or simply absent).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.compilers.ids import net_item_uuid, pin_uuid, sheet_uuid, symbol_uuid
from ai_eda.compilers.pins import load_verified_symbol, pad_pin_types
from ai_eda.compilers.schematic_layout import (
    LAYOUT_MARGIN_CHANNEL_MM,
    LAYOUT_ORIGIN_MM,
    LAYOUT_WIDE_GUTTER_EXTRA_MM,
    LAYOUT_WIDE_MARGIN_CHANNEL_MM,
    Extent,
    Vec,
    label_orientation,
    layout_channels,
    layout_columns,
    layout_paper,
    layout_positions,
    natural_ref_key,
    pin_body_direction,
    pin_position,
    point_on_segment,
    snap,
    stub_end,
    symbol_extent,
    transform_offset,
)
from ai_eda.compilers.schematic_route import MAX_WIRED_PINS, Channels, RoutePin, channel_full, foreign_point_on_wire, route_net
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Component, LibraryRef, Pin, PinElectricalType, PinRef, Provenance, ProvenanceKind
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary, SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import Node, Q, S

__all__ = ["SchematicCompiler", "SCH_FORMAT_VERSION", "GLOBAL_LABEL_SHAPE", "POWER_FLAG"]

log = logging.getLogger(__name__)

#: what KiCad 10.0 writes (kicad-cli 10.0.6 also accepts 20250114 / 20250610; 20270101 is rejected)
SCH_FORMAT_VERSION = 20260101
#: kicad-cli accepts any generator string (verified), but the KiCad GUI may key format migrations on
#: generator_version, so the header states the *format* the file follows (eeschema 10.0). The real
#: producer is recorded in the title_block comment and in ArtifactRef.generator / generator_version.
GENERATOR = "eeschema"
GENERATOR_VERSION = "10.0"
FONT_SIZE_MM = 1.27
#: neutral label shape: adds no driver / direction semantics to ERC
GLOBAL_LABEL_SHAPE = "passive"
#: the library symbol that tells ERC a net is powered from outside the sheet (one power-output pin, reference ``#FLG``)
POWER_FLAG = ("power", "PWR_FLAG")
#: the pin types a flag's power output meets without an ERC violation (measured with kicad-cli 10.0.6); any other type on the net: no flag
_FLAG_NEIGHBOURS = frozenset({PinElectricalType.POWER_IN, PinElectricalType.INPUT, PinElectricalType.PASSIVE})

_REF_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*[0-9]+")


@dataclass(frozen=True, slots=True)
class _PlacedPin:
    ref: str
    pin: SymbolPin
    position: Vec
    body_dir: Vec


@dataclass(slots=True)
class _PlacedSymbol:
    component: Component
    symbol: SymbolDef
    footprint_id: str  # "" when the component has no verified footprint
    x: float
    y: float
    extent: Extent  # absolute schematic-frame box of body + pins + stubs + labels
    rotation: int = 0
    mirror: str | None = None


@dataclass(frozen=True, slots=True)
class _Widen:
    """Which gutters of the symbol table are wider than the default (B11 W3): the extra width after each column / row and
    the widths of the (left, right, top, bottom) margin corridors. The default instance is the table as it always was."""

    columns: tuple[float, ...] = ()
    rows: tuple[float, ...] = ()
    margins: tuple[float, float, float, float] = (LAYOUT_MARGIN_CHANNEL_MM,) * 4

    @classmethod
    def of(cls, full: frozenset[tuple[str, int]], n_vertical: int, n_horizontal: int) -> _Widen:
        """The widening for the channels that ran out of tracks: ``("v" | "h", index)`` into the channel lists, whose first and
        last entries are the margin corridors and whose entry ``i`` between them is the gutter after column / row ``i - 1``."""
        def inner(axis: str, n: int) -> tuple[float, ...]:
            return tuple(LAYOUT_WIDE_GUTTER_EXTRA_MM if (axis, i) in full else 0.0 for i in range(1, n - 1))

        def margin(axis: str, i: int) -> float:
            return LAYOUT_WIDE_MARGIN_CHANNEL_MM if (axis, i) in full else LAYOUT_MARGIN_CHANNEL_MM

        return cls(columns=inner("v", n_vertical), rows=inner("h", n_horizontal),
                   margins=(margin("v", 0), margin("v", n_vertical - 1), margin("h", 0), margin("h", n_horizontal - 1)))

    @property
    def origin(self) -> Vec:
        """The first symbol's origin: moved right / down by what the left / top corridor gained, so its wires stay on the sheet."""
        return (snap(LAYOUT_ORIGIN_MM[0] + self.margins[0] - LAYOUT_MARGIN_CHANNEL_MM), snap(LAYOUT_ORIGIN_MM[1] + self.margins[2] - LAYOUT_MARGIN_CHANNEL_MM))


@dataclass(slots=True)
class _Sheet:
    """One drawing of the placed symbols: the nodes that depend on the positions, and how the routing went."""

    channels: Channels
    no_connects: list[Node]
    wires: list[Node]
    labels: list[Node]
    junctions: list[Node]
    notes: list[str]
    wired: int  # nets drawn as wires
    blocked: int  # nets of 2 .. MAX_WIRED_PINS pins left with labels only for want of a clear path
    full: frozenset[tuple[str, int]]  # channels with no track left after routing


# --------------------------------------------------------------------------- node builders


def _effects(justify: str | tuple[str, ...] | None = None) -> Node:
    tokens = (justify,) if isinstance(justify, str) else tuple(justify or ())
    return S("effects", S("font", S("size", FONT_SIZE_MM, FONT_SIZE_MM)), S("justify", *tokens) if tokens else None)


def _property(key: str, value: str, x: float, y: float, rot: float = 0, hide: bool = False, justify: str | tuple[str, ...] | None = None) -> Node:
    return S("property", Q(key), Q(value), S("at", x, y, rot), S("hide", True) if hide else None, _effects(justify))


#: where a Reference / Value field goes when the library symbol gives it no position: right of the origin, above / below it
_FALLBACK_FIELDS: dict[str, tuple[float, float]] = {"Reference": (2.54, -1.27), "Value": (2.54, 1.27)}


def _library_field(symbol: SymbolDef, key: str) -> tuple[float, float, float, tuple[str, ...], bool] | None:
    """``(x, y, angle, justify tokens, hidden)`` of the library symbol's own ``key`` field (library frame, Y up); ``None`` without a position."""
    for prop in sexpr.find_all(symbol.node, "property"):
        if len(prop) < 3 or isinstance(prop[1], list) or str(prop[1]) != key:
            continue
        at = sexpr.find(prop, "at")
        if at is None or len(at) < 3:
            return None
        try:
            x, y = sexpr.to_float(at[1]), sexpr.to_float(at[2])
            angle = sexpr.to_float(at[3]) if len(at) > 3 else 0.0
        except (TypeError, ValueError):
            return None
        effects = sexpr.find(prop, "effects")
        justify = sexpr.find(effects, "justify") if effects is not None else None
        tokens = tuple(str(t) for t in justify[1:] if not isinstance(t, list)) if justify is not None else ()
        return x, y, angle, tokens, sexpr.get(prop, "hide") == "yes"
    return None


def _field(key: str, value: str, ps: "_PlacedSymbol") -> Node:
    """A Reference / Value field where the library symbol puts it: its ``(at ..)`` offset turned by the instance's rotation / mirror
    (KiCad's own rule when it places a symbol), its angle and justification as the library writes them (KiCad applies the instance's
    transform when it draws); the fixed fallback offset only when the library gives no position."""
    lib = _library_field(ps.symbol, key)
    if lib is None:
        fx, fy = _FALLBACK_FIELDS[key]
        return _property(key, value, snap(ps.x + fx), snap(ps.y + fy), justify="left")
    lx, ly, angle, justify, hidden = lib
    dx, dy = transform_offset(lx, ly, ps.rotation, ps.mirror)
    return _property(key, value, snap(ps.x + dx), snap(ps.y + dy), angle, hide=hidden, justify=justify)


def _wire(a: Vec, b: Vec, uuid: str) -> Node:
    return S(
        "wire",
        S("pts", S("xy", a[0], a[1]), S("xy", b[0], b[1])),
        S("stroke", S("width", 0), S("type", "default")),
        S("uuid", Q(uuid)),
    )


def _global_label(text: str, at: Vec, angle: int, justify: str, uuid: str) -> Node:
    return S(
        "global_label",
        Q(text),
        S("shape", GLOBAL_LABEL_SHAPE),
        S("at", at[0], at[1], angle),
        _effects(justify),
        S("uuid", Q(uuid)),
    )


def _no_connect(at: Vec, uuid: str) -> Node:
    return S("no_connect", S("at", at[0], at[1]), S("uuid", Q(uuid)))


def _junction(at: Vec, uuid: str) -> Node:
    return S("junction", S("at", at[0], at[1]), S("diameter", 0), S("color", 0, 0, 0, 0), S("uuid", Q(uuid)))


# --------------------------------------------------------------------------- compiler


class SchematicCompiler(Compiler):
    id = "compiler.kicad_sch"
    version = "0.8"
    kind = ArtifactKind.SCHEMATIC
    layout: str | None = None
    answer_layout: str | None = None

    def __init__(self, layout: str | None = None, *, answer_layout: str | None = None) -> None:
        self.layout = layout
        self.answer_layout = answer_layout

    def layout_policy(self, ir: CircuitIR) -> str:
        policy = (ir.topology.schematic_layout if ir.topology else None) or self.layout or self.answer_layout or "legacy"
        if policy not in ("legacy", "textbook"):
            raise CompileError(f"unknown schematic layout {policy!r}; expected legacy or textbook")
        return policy

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        library = ctx.tools.get("kicad_library")
        if library is None:
            library = KicadLibrary()
        elif not isinstance(library, KicadLibrary):
            raise CompileError(f"ctx.tools['kicad_library'] is not a KicadLibrary: {type(library).__name__}")
        node, notes = self.build_with_notes(ir, library)
        path = ctx.workdir / f"{ir.project.id}.kicad_sch"
        artifact = self._write(ir, path, sexpr.dumps(node), notes)
        if self.layout_policy(ir) == "textbook":
            from ai_eda.compilers.schematic_textbook import VERSION, eligible
            if eligible(ir):
                artifact.generator_version = VERSION
        return artifact

    # --- pure tree construction (no I/O besides library reads) --------------------

    def build(self, ir: CircuitIR, library: KicadLibrary) -> Node:
        """The complete ``(kicad_sch ...)`` tree for ``ir``. Raises CompileError, never guesses."""
        return self.build_with_notes(ir, library)[0]

    def build_with_notes(self, ir: CircuitIR, library: KicadLibrary) -> tuple[Node, list[str]]:
        """:meth:`build` plus the notes a human should read in ``compile.schematic`` (which nets stayed label-only, and why)."""
        project_id = ir.project.id
        if not project_id or re.search(r'[\\/:*?"<>|\s]', project_id):
            raise CompileError(f"project id {project_id!r} is not usable as a KiCad file stem")
        if not ir.components:
            # an empty schematic passes ERC with zero violations - that would be evidence about nothing
            raise NothingToCompileError("IR has no components: nothing to draw (the schematic stage needs a component list)")
        if self.layout_policy(ir) == "textbook":
            from ai_eda.compilers.schematic_textbook import build, eligible
            if eligible(ir):
                return build(self, ir, library)
        root_uuid = sheet_uuid(project_id)
        drawn, flags, flag_notes = self._power_flags(ir, library)
        components = [*self._ordered(ir, self._checked_components(ir)), *flags]
        pin_nets = self._pin_net_names(drawn)
        placed = self._place(components, library, pin_nets)
        order = [c.ref for c in components]
        widened = _Widen()
        sheet = self._draw(ir, drawn, placed, order, pin_nets, widened)
        if sheet.blocked and sheet.full:
            # B11 W3: nets stayed label-only while channels had no track left - widen exactly those gutters and draw the
            # sheet again; the wider sheet is kept only when it wires more nets (so a sheet whose wires fit keeps its bytes)
            widen = _Widen.of(sheet.full, len(sheet.channels.vertical), len(sheet.channels.horizontal))
            wider_placed = self._moved(placed, order, widen)
            wider = self._draw(ir, drawn, wider_placed, order, pin_nets, widen)
            if wider.wired > sheet.wired:
                wider.notes.append(f"gutters widened: {len(sheet.full)} channel(s) had no wire track left at the default width "
                                   f"({sheet.wired} net(s) wired there, {wider.wired} now)")
                placed.update(wider_placed)  # in place: whoever holds the dict ``_place`` returned sees the sheet that is written
                sheet, widened = wider, widen
        no_connects, wires, labels, junctions = sheet.no_connects, sheet.wires, sheet.labels, sheet.junctions
        notes = sheet.notes + flag_notes

        lib_entries = [placed[ref].symbol for ref in placed]
        lib_symbols = S(
            "lib_symbols",
            *(sym.lib_symbols_entry() for sym in sorted({s.lib_id: s for s in lib_entries}.values(), key=lambda s: s.lib_id)),
        )
        symbols = [self._symbol_node(project_id, root_uuid, ps) for ps in placed.values()]

        return S(
            "kicad_sch",
            S("version", SCH_FORMAT_VERSION),
            S("generator", Q(GENERATOR)),
            S("generator_version", Q(GENERATOR_VERSION)),
            S("uuid", Q(root_uuid)),
            self._paper(placed, widened),
            S(
                "title_block",
                S("title", Q(ir.project.name)),
                S("comment", 1, Q(f"generated by {self.id} {self.version} from IR project {project_id}")),
                S("comment", 2, Q(ir.project.description)) if ir.project.description else None,
            ),
            lib_symbols,
            *no_connects,
            *wires,
            *junctions,
            *labels,
            *symbols,
            S("sheet_instances", S("path", Q("/"), S("page", Q("1")))),
            S("embedded_fonts", False),
        ), notes

    def _draw(
        self, ir: CircuitIR, drawn: CircuitIR, placed: dict[str, _PlacedSymbol], order: list[str], pin_nets: dict[tuple[str, str], str], widen: _Widen,
    ) -> _Sheet:
        """Everything that depends on where the symbols stand: the channels, the geometry guard, stubs + labels and the routed wires."""
        vertical, horizontal = layout_channels({r: placed[r].extent for r in order}, order, layout_columns(len(order)), margins=widen.margins)
        channels = Channels(vertical=vertical, horizontal=horizontal)
        pins_by_key = self._pin_map(placed)
        stacked = self._stacks(placed, pins_by_key, pin_nets)
        self._check_geometry(placed, pins_by_key, stacked)
        no_connects, wires, labels = self._connectivity(drawn, placed, pins_by_key, stacked)
        routed, junctions, notes, wired, blocked, full = self._route(ir, placed, pins_by_key, stacked, channels)  # the IR's own nets: a flag keeps stub + label
        return _Sheet(channels=channels, no_connects=no_connects, wires=wires + routed, labels=labels, junctions=junctions, notes=notes,
                      wired=wired, blocked=blocked, full=full)

    @staticmethod
    def _moved(placed: dict[str, _PlacedSymbol], order: list[str], widen: _Widen) -> dict[str, _PlacedSymbol]:
        """The same symbols on the table with ``widen``'s gutters: positions and extents recomputed, nothing else touched."""
        own = {ref: placed[ref].extent.shifted(-placed[ref].x, -placed[ref].y) for ref in order}
        positions = layout_positions(own, origin=widen.origin, columns=layout_columns(len(order)), order=order,
                                     extra_columns=widen.columns, extra_rows=widen.rows)
        return {ref: replace(placed[ref], x=positions[ref][0], y=positions[ref][1], extent=own[ref].shifted(*positions[ref])) for ref in order}

    @staticmethod
    def _paper(placed: dict[str, _PlacedSymbol], widen: _Widen | None = None) -> Node:
        """``(paper "A4")`` unless the placed extents need a larger sheet (A3 .. A0, else ``"User" W H``). A widened right /
        bottom corridor asks for that much more sheet, so its wires keep the distance from the frame and the title block that
        the default corridor's have."""
        grown = (0.0, 0.0) if widen is None else (widen.margins[1] - LAYOUT_MARGIN_CHANNEL_MM, widen.margins[3] - LAYOUT_MARGIN_CHANNEL_MM)
        name, w, h = layout_paper((ps.extent for ps in placed.values()), grown=grown)
        return S("paper", Q(name)) if name != "User" else S("paper", Q(name), w, h)

    # --- validation -----------------------------------------------------------------

    @staticmethod
    def _checked_components(ir: CircuitIR) -> list[Component]:
        """Annotated, unique references (kicad-cli ERC does not report either fault)."""
        seen: set[str] = set()
        for c in ir.components:
            if not _REF_RE.fullmatch(c.ref):
                raise CompileError(f"component reference {c.ref!r} is not annotated (expected e.g. R1, J3)")
            if c.ref in seen:
                raise CompileError(f"duplicate component reference {c.ref!r}")
            seen.add(c.ref)
        return sorted((c for c in ir.components if not c.board_only), key=lambda c: natural_ref_key(c.ref))

    @staticmethod
    def _pin_net_names(ir: CircuitIR) -> dict[tuple[str, str], str]:
        """``(ref, pin) -> net name`` (first net wins; conflicts are refused later in ``_connectivity``)."""
        out: dict[tuple[str, str], str] = {}
        for net in sorted(ir.nets, key=lambda n: n.name):
            for pref in net.pins:
                out.setdefault((pref.component_ref, pref.pin_number), net.name)
        return out

    @staticmethod
    def _ordered(ir: CircuitIR, components: list[Component]) -> list[Component]:
        """Components in table order (W2): ``ir.topology`` blocks in sequence (each block's members in natural order, a
        component only where it first appears), then every component no block names. Without a topology: natural ref
        order, as ``_checked_components`` returns them. ``_place`` lays the table out in list order."""
        if ir.topology is None or not ir.topology.blocks:
            return components
        by_ref = {c.ref: c for c in components}
        order: list[str] = []
        for block in ir.topology.blocks:
            for ref in sorted(block.component_refs, key=natural_ref_key):
                if ref in by_ref and ref not in order:
                    order.append(ref)
        return [by_ref[r] for r in order] + [c for c in components if c.ref not in order]

    def _power_flags(self, ir: CircuitIR, library: KicadLibrary) -> tuple[CircuitIR, list[Component], list[str]]:
        """B12: one ``power:PWR_FLAG`` for every net that has a power-input pin and nothing but ``_FLAG_NEIGHBOURS`` pins.

        Returns the IR as drawn (each such net with its flag's pin added), the flags as one-pin components for the
        table's last cells (``#FLG01`` .. in net-name order), and the notes. Pin types are the IR's, which ``_place``
        holds to the library's. A net with a power-input pin, no power-output pin and a pin of another type (an
        ``output`` feeding a supply pin) gets no flag and a note: the flag would be a second ERC violation, not a fix.
        A library without the symbol gets no flag and a note that says so: the file is then what it was before flags
        existed, and ERC reports the pins.
        """
        kinds = {(c.ref, p.number): p.electrical_type for c in ir.components for p in c.pins}
        undriven: set[str] = set()
        left: set[str] = set()
        for net in ir.nets:
            types = {kinds.get((p.component_ref, p.pin_number)) for p in net.pins}
            if PinElectricalType.POWER_IN in types and PinElectricalType.POWER_OUT not in types:
                (undriven if types <= _FLAG_NEIGHBOURS else left).add(net.name)
        notes = [f"power flags: none on {', '.join(sorted(left))} (a pin there is neither power-input, input nor passive: KiCad would report "
                 "the flag against it; ERC reports the power-input pins as not driven)"] if left else []
        undriven = sorted(undriven)
        if not undriven:
            return ir, [], notes
        lib_ref = library.resolve_symbol(LibraryRef(library=POWER_FLAG[0], name=POWER_FLAG[1]))
        if not lib_ref.verified:
            return ir, [], [f"power flags: none placed, {':'.join(POWER_FLAG)} is not in the installed KiCad libraries "
                            f"(ERC reports the power-input pins of {', '.join(undriven)} as not driven)", *notes]
        symbol = library.load_symbol(lib_ref)
        prefix = symbol.properties.get("Reference") or "#FLG"
        prov = Provenance(kind=ProvenanceKind.DERIVED, tool=self.id)
        flags = {
            name: Component(
                ref=f"{prefix}{i:02d}", value=symbol.properties.get("Value") or symbol.name, symbol=lib_ref, provenance=prov,
                pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins],
            )
            for i, name in enumerate(undriven, 1)
        }
        nets = [
            n.model_copy(update={"pins": [*n.pins, *(PinRef(component_ref=flags[n.name].ref, pin_number=p.number) for p in flags[n.name].pins)]})
            if n.name in flags else n
            for n in ir.nets
        ]
        note = f"power flags: PWR_FLAG on {len(flags)} net(s) with a power-input pin and no power-output pin ({', '.join(flags)})"
        return ir.model_copy(update={"nets": nets}), list(flags.values()), [note, *notes]

    def _place(
        self, components: list[Component], library: KicadLibrary, pin_nets: dict[tuple[str, str], str]
    ) -> dict[str, _PlacedSymbol]:
        """Resolve every symbol, measure it, and lay the symbols out on a grid the measurements dictate, in the order of
        ``components`` (row-major; see ``_ordered``)."""
        resolved: dict[str, SymbolDef] = {}
        extents: dict[str, Extent] = {}
        for c in components:
            symbol = load_verified_symbol(c, library)
            pad_pin_types(c, symbol)  # pin numbers + electrical types must be the library's
            label_lengths = {p.number: len(pin_nets[(c.ref, p.number)]) for p in symbol.pins if (c.ref, p.number) in pin_nets}
            resolved[c.ref] = symbol
            extents[c.ref] = symbol_extent(symbol, label_lengths)
        order = [c.ref for c in components]
        positions = layout_positions(extents, columns=layout_columns(len(order)), order=order)
        placed: dict[str, _PlacedSymbol] = {}
        for c in components:
            x, y = positions[c.ref]
            placed[c.ref] = _PlacedSymbol(
                component=c,
                symbol=resolved[c.ref],
                footprint_id="" if resolved[c.ref].is_power else self._footprint_id(c, library),  # a power symbol has no footprint
                x=x,
                y=y,
                extent=extents[c.ref].shifted(x, y),
            )
        return placed

    @staticmethod
    def _footprint_id(c: Component, library: KicadLibrary) -> str:
        """``"Lib:Name"`` only for a footprint the IR marks verified *and* the library finds on disk."""
        if c.footprint is None or not c.footprint.verified:
            log.warning("%s: no verified footprint; Footprint property left empty", c.ref)
            return ""
        lib_id = f"{c.footprint.library}:{c.footprint.name}"
        if not library.resolve_footprint(c.footprint).verified:
            raise CompileError(f"{c.ref}: footprint {lib_id!r} is marked verified but is not in the installed KiCad libraries")
        return lib_id

    # --- geometry -------------------------------------------------------------------

    @staticmethod
    def _pin_map(placed: dict[str, _PlacedSymbol]) -> dict[tuple[str, str], _PlacedPin]:
        out: dict[tuple[str, str], _PlacedPin] = {}
        for ref, ps in placed.items():
            for pin in ps.symbol.pins:
                if (ref, pin.number) in out:
                    raise CompileError(f"{ref}: library symbol {ps.symbol.lib_id!r} repeats pin number {pin.number!r} (stacked pins); unsupported - one physical pin per number")
                pos = pin_position(ps.x, ps.y, ps.rotation, ps.mirror, pin)
                out[(ref, pin.number)] = _PlacedPin(ref, pin, pos, pin_body_direction(ps.rotation, ps.mirror, pin.angle))
        return out

    @staticmethod
    def _stacks(
        placed: dict[str, _PlacedSymbol], pins: dict[tuple[str, str], _PlacedPin], pin_nets: dict[tuple[str, str], str],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        """Stacked pins: ``{member: representative}`` for every pin of a symbol instance that sits on another pin of the same instance.

        KiCad connects every pin at one point, so a library symbol may stack
        pins that are one node electrically (the ATmega128's hidden VCC / GND
        pins on the visible ones). That is accepted only when the IR agrees:
        every pin at the point is in the *same* IR net (or none of them is in
        any net, left to the no_connect rule) and they all leave the point in
        the same direction. The representative is the lowest pin number at
        the point; it alone gets the wire stub and the global label (one
        connection point, so no label is stacked on another), the others are
        connected by KiCad through the shared point. Anything else - two nets,
        a pin in a net stacked on one in none, different directions - is a
        :class:`CompileError` (the ``coincides with`` refusal), never a silent
        merge.
        """
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for ref in sorted(placed, key=natural_ref_key):
            at: dict[Vec, list[tuple[str, str]]] = {}
            for key in sorted((k for k in pins if k[0] == ref), key=lambda k: natural_ref_key(k[1])):
                at.setdefault(pins[key].position, []).append(key)
            for point, keys in at.items():
                if len(keys) < 2:
                    continue
                first = keys[0]
                for key in keys[1:]:
                    what = f"pin {ref}.{key[1]} at {point} coincides with pin {ref}.{first[1]}; KiCad would connect them"
                    if pin_nets.get(key) != pin_nets.get(first):
                        nets = (pin_nets.get(key), pin_nets.get(first))
                        raise CompileError(
                            f"{what} (stacked pins of {placed[ref].symbol.lib_id!r}), but the IR puts them in "
                            f"{'no net' if nets[0] is None else repr(nets[0])} and {'no net' if nets[1] is None else repr(nets[1])}"
                        )
                    if pins[key].body_dir != pins[first].body_dir:
                        raise CompileError(f"{what} (stacked pins of {placed[ref].symbol.lib_id!r} pointing in different directions; unsupported)")
                    out[key] = first
        return out

    @staticmethod
    def _check_geometry(
        placed: dict[str, _PlacedSymbol], pins: dict[tuple[str, str], _PlacedPin], stacked: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> None:
        """Refuse any layout in which KiCad could connect things the IR does not connect.

        Every pin end and every stub end is a connection point; two of them at
        the same coordinates, or one lying on another pin's stub, would merge
        nets silently (ERC does not report it). Extents that touch are refused
        as the guard behind the pitch computation. A stacked pin
        (``stacked``, see :meth:`_stacks`) is not a point of its own: it is
        its representative's point, which the IR already puts in the same net.
        """
        stacked = stacked or {}
        refs = sorted(placed, key=natural_ref_key)
        for i, a in enumerate(refs):
            for b in refs[i + 1 :]:
                if placed[a].extent.overlaps(placed[b].extent):
                    raise CompileError(f"symbols {a} and {b} overlap on the schematic grid: {placed[a].extent} vs {placed[b].extent}")
        points: dict[Vec, str] = {}
        stubs: dict[tuple[str, str], tuple[Vec, Vec]] = {}
        for key in sorted(pins, key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1]))):
            if key in stacked:
                continue
            pp = pins[key]
            label = f"{pp.ref}.{pp.pin.number}"
            end = stub_end(pp.position, pp.body_dir)
            for point, what in ((pp.position, f"pin {label}"), (end, f"stub end of {label}")):
                if point in points:
                    raise CompileError(f"{what} at {point} coincides with {points[point]}; KiCad would connect them")
                points[point] = what
            stubs[key] = (pp.position, end)
        for key, (a, b) in stubs.items():
            for point, what in points.items():
                if point in (a, b):
                    continue
                if point_on_segment(point, a, b):
                    raise CompileError(f"{what} at {point} lies on the wire stub of {key[0]}.{key[1]}; KiCad would connect them")

    def _connectivity(
        self, ir: CircuitIR, placed: dict[str, _PlacedSymbol], pins: dict[tuple[str, str], _PlacedPin],
        stacked: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> tuple[list[Node], list[Node], list[Node]]:
        stacked = stacked or {}
        project_id = ir.project.id
        names = [n.name for n in ir.nets]
        if len(set(names)) != len(names):
            raise CompileError(f"duplicate net names: {sorted(n for n in set(names) if names.count(n) > 1)}")
        owner: dict[tuple[str, str], str] = {}
        wires: list[Node] = []
        labels: list[Node] = []
        for net in sorted(ir.nets, key=lambda n: n.name):
            if not net.name.strip():
                raise CompileError("a net has an empty name")
            for pref in sorted(net.pins, key=lambda p: (natural_ref_key(p.component_ref), natural_ref_key(p.pin_number))):
                key = (pref.component_ref, pref.pin_number)
                if pref.component_ref not in placed:
                    raise CompileError(f"net {net.name!r}: unknown component {pref.component_ref!r}")
                if key not in pins:
                    raise CompileError(f"net {net.name!r}: {pref.component_ref} has no pin {pref.pin_number!r}")
                if key in owner:
                    if owner[key] == net.name:
                        continue  # same pin listed twice in one net: harmless
                    raise CompileError(f"pin {pref.component_ref}.{pref.pin_number} is in both nets {owner[key]!r} and {net.name!r}")
                owner[key] = net.name
                if key in stacked:
                    continue  # connected through its representative's point, which carries this net's stub and label
                pp = pins[key]
                end = stub_end(pp.position, pp.body_dir)
                angle, justify = label_orientation(pp.body_dir)
                wires.append(_wire(pp.position, end, net_item_uuid(project_id, "wire", pp.ref, pp.pin.number)))
                labels.append(_global_label(net.name, end, angle, justify, net_item_uuid(project_id, "global_label", net.name, pp.ref, pp.pin.number)))
        no_connects: list[Node] = []
        for key in sorted(pins, key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1]))):
            if key in owner:
                continue
            ref, number = key
            ir_pin = placed[ref].component.pin(number)
            if ir_pin is not None and ir_pin.electrical_type is PinElectricalType.NO_CONNECT:
                if key not in stacked:  # one marker per point: a stacked pin shares its representative's
                    no_connects.append(_no_connect(pins[key].position, net_item_uuid(project_id, "no_connect", ref, number)))
                continue
            raise CompileError(f"pin {ref}.{number} ({(ir_pin.electrical_type if ir_pin else 'unknown')}) is not in any net; "
                               "connect it or mark it no_connect in the IR")
        return no_connects, wires, labels

    def _route(
        self,
        ir: CircuitIR,
        placed: dict[str, _PlacedSymbol],
        pins: dict[tuple[str, str], _PlacedPin],
        stacked: dict[tuple[str, str], tuple[str, str]],
        channels: Channels,
    ) -> tuple[list[Node], list[Node], list[str], int, int, frozenset[tuple[str, int]]]:
        """Wires + junctions drawn between the stub ends of every net of 2 .. MAX_WIRED_PINS pins, the notes, and how it went:
        nets wired, nets left without a clear path, and the channels that have no track left (``("v" | "h", index)``).

        Nets are routed in name order; each sees the wires of the nets before it. Every connection point of the
        sheet that does not belong to the net being routed is forbidden ground for its wires (a stub end is the
        label anchor, a pin end a connection point). The final wires are re-checked by ``_check_wires``.
        """
        project_id = ir.project.id
        extents = {ref: ps.extent for ref, ps in placed.items()}
        all_points: dict[Vec, str] = {}
        stub_of: dict[tuple[str, str], tuple[Vec, Vec]] = {}  # key -> (stub end, away direction)
        for key, pp in pins.items():
            if key in stacked:
                continue
            end = stub_end(pp.position, pp.body_dir)
            stub_of[key] = (end, (snap(-pp.body_dir[0]), snap(-pp.body_dir[1])))
            all_points.setdefault(pp.position, f"pin {pp.ref}.{pp.pin.number}")
            all_points.setdefault(end, f"label anchor of {pp.ref}.{pp.pin.number}")
        owner: dict[tuple[str, str], str] = {}
        for net in ir.nets:
            for p in net.pins:
                owner.setdefault(stacked.get((p.component_ref, p.pin_number), (p.component_ref, p.pin_number)), net.name)
        net_of_point: dict[Vec, str] = {}
        for key, (end, _) in stub_of.items():
            net_of_point[pins[key].position] = owner.get(key, "")
            net_of_point[end] = owner.get(key, "")
        routed: list[Node] = []
        junction_nodes: list[Node] = []
        segments_so_far: list[tuple[Vec, Vec]] = []
        tagged: list[tuple[tuple[Vec, Vec], str]] = []
        used: dict[tuple[str, int], int] = {}
        wired: list[str] = []
        skipped: list[str] = []
        blocked = 0
        for net in sorted(ir.nets, key=lambda n: n.name):
            keys = sorted({stacked.get((p.component_ref, p.pin_number), (p.component_ref, p.pin_number)) for p in net.pins},
                          key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1])))
            if len(keys) < 2:
                continue
            if len(keys) > MAX_WIRED_PINS:
                skipped.append(f"{net.name} ({len(keys)} pins > {MAX_WIRED_PINS}: labels only)")
                continue
            route_pins = [RoutePin(ref=k[0], end=stub_of[k][0], away=stub_of[k][1]) for k in keys]
            own = {stub_of[k][0] for k in keys} | {pins[k].position for k in keys}
            points = {p: what for p, what in all_points.items() if p not in own}
            for seg in segments_so_far:
                for p in seg:
                    points.setdefault(p, "wire end of another net")
            result = route_net(route_pins, channels, extents, points, segments_so_far, used)
            if result is None:
                skipped.append(f"{net.name} (no clear path: labels only)")
                blocked += 1
                continue
            for track in set(result.tracks_used):
                used[track] = used.get(track, 0) + 1
            for i, (a, b) in enumerate(result.segments):
                routed.append(_wire(a, b, net_item_uuid(project_id, "wire", net.name, "route", i)))
            for j, at in enumerate(result.junctions):
                junction_nodes.append(_junction(at, net_item_uuid(project_id, "junction", net.name, j)))
            segments_so_far += result.segments
            tagged += [(seg, net.name) for seg in result.segments]
            wired.append(net.name)
        self._check_wires(tagged, all_points, net_of_point)
        total = len(wired) + len(skipped)
        notes = [f"wiring: {len(wired)} of {total} net(s) drawn as wires (labels kept on every pin)"] if total else []
        if skipped:
            notes.append("not wired: " + "; ".join(skipped))
        full = frozenset(
            (axis, i) for axis, chans in (("v", channels.vertical), ("h", channels.horizontal))
            for i, chan in enumerate(chans) if channel_full(chan, used.get((axis, i), 0))
        )
        return routed, junction_nodes, notes, len(wired), blocked, full

    @staticmethod
    def _check_wires(tagged: list[tuple[tuple[Vec, Vec], str]], points: dict[Vec, str], net_of_point: dict[Vec, str]) -> None:
        """Refuse the file when a routed wire touches a connection point (pin end, label anchor, another net's wire end) that is
        not its own net's. The router never produces one; this is the guard behind it, the rule ``_check_geometry`` applies to
        stubs: KiCad would connect them, silently."""
        for seg, net in tagged:
            for point, what in points.items():
                if point_on_segment(point, seg[0], seg[1]) and net_of_point.get(point) != net:
                    raise CompileError(f"routed wire of net {net!r} {seg[0]}-{seg[1]} touches {what} at {point}; KiCad would connect them")
            for other, other_net in tagged:
                if other_net == net:
                    continue
                for point in other:
                    if point_on_segment(point, seg[0], seg[1]):
                        raise CompileError(f"routed wire of net {net!r} {seg[0]}-{seg[1]} touches a wire end of net {other_net!r} at {point}; KiCad would connect them")

    # --- symbol instance ------------------------------------------------------------

    @staticmethod
    def _symbol_node(project_id: str, root_uuid: str, ps: _PlacedSymbol) -> Node:
        c = ps.component
        x, y = ps.x, ps.y
        datasheet = (c.datasheet.url if c.datasheet and c.datasheet.url else "") or ""
        description = c.description or ps.symbol.properties.get("Description", "")
        # A part with no SPICE binding (or one excluded from it) has no SPICE meaning in the IR; without
        # this flag kicad-cli's SPICE export writes a placeholder line (`J1 __J1`, exit 0, empty stderr)
        # that makes ngspice refuse the whole circuit - measured with kicad-cli 10.0.6. `dnp` does not
        # remove it; `exclude_from_sim` does.
        exclude_from_sim = c.spice is None or c.spice.exclude
        return S(
            "symbol",
            S("lib_id", Q(ps.symbol.lib_id)),
            S("at", x, y, ps.rotation),
            S("mirror", ps.mirror) if ps.mirror else None,
            S("unit", 1),
            S("exclude_from_sim", exclude_from_sim),
            S("in_bom", True),
            S("on_board", True),
            S("dnp", False),
            S("uuid", Q(symbol_uuid(project_id, c.ref))),
            _field("Reference", c.ref, ps),
            _field("Value", c.value, ps),
            _property("Footprint", ps.footprint_id, x, y, hide=True),
            _property("Datasheet", datasheet, x, y, hide=True),
            _property("Description", description, x, y, hide=True),
            *(S("pin", Q(pin.number), S("uuid", Q(pin_uuid(project_id, c.ref, pin.number)))) for pin in ps.symbol.pins),
            S("instances", S("project", Q(project_id), S("path", Q("/" + root_uuid), S("reference", Q(c.ref)), S("unit", 1)))),
        )
