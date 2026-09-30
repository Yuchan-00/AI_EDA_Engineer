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
the net. Same-named global labels form one net, so no wire routing or
junctions are needed, and the exported netlist reproduces the IR nets exactly
(tested against kicad-cli with tall connectors that would overlap on a fixed
25.4 mm grid). Labels use shape ``passive`` so they add no electrical
semantics of their own: ERC pin-to-pin / driver checks are decided by the
real pins only.

What the file does *not* carry yet: multi-unit symbols, power symbols, and
sheet hierarchy (each refused with :class:`CompileError` or simply absent).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.compilers.ids import net_item_uuid, pin_uuid, sheet_uuid, symbol_uuid
from ai_eda.compilers.pins import load_verified_symbol, pad_pin_types
from ai_eda.compilers.schematic_layout import (
    Extent,
    Vec,
    label_orientation,
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
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Component, PinElectricalType
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary, SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import Node, Q, S

__all__ = ["SchematicCompiler", "SCH_FORMAT_VERSION", "GLOBAL_LABEL_SHAPE"]

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


# --------------------------------------------------------------------------- node builders


def _effects(justify: str | tuple[str, ...] | None = None) -> Node:
    tokens = (justify,) if isinstance(justify, str) else tuple(justify or ())
    return S("effects", S("font", S("size", FONT_SIZE_MM, FONT_SIZE_MM)), S("justify", *tokens) if tokens else None)


def _property(key: str, value: str, x: float, y: float, rot: float = 0, hide: bool = False, justify: str | tuple[str, ...] | None = None) -> Node:
    return S("property", Q(key), Q(value), S("at", x, y, rot), S("hide", True) if hide else None, _effects(justify))


#: where a Reference / Value field goes when the library symbol gives it no position: right of the origin, above / below it
_FALLBACK_FIELDS: dict[str, tuple[float, float]] = {"Reference": (2.54, -1.27), "Value": (2.54, 1.27)}


def _library_field(symbol: SymbolDef, key: str) -> tuple[float, float, float, tuple[str, ...]] | None:
    """``(x, y, angle, justify tokens)`` of the library symbol's own ``key`` field (library frame, Y up); ``None`` without a position."""
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
        return x, y, angle, tokens
    return None


def _field(key: str, value: str, ps: "_PlacedSymbol") -> Node:
    """A Reference / Value field where the library symbol puts it: its ``(at ..)`` offset turned by the instance's rotation / mirror
    (KiCad's own rule when it places a symbol), its angle and justification as the library writes them (KiCad applies the instance's
    transform when it draws); the fixed fallback offset only when the library gives no position."""
    lib = _library_field(ps.symbol, key)
    if lib is None:
        fx, fy = _FALLBACK_FIELDS[key]
        return _property(key, value, snap(ps.x + fx), snap(ps.y + fy), justify="left")
    lx, ly, angle, justify = lib
    dx, dy = transform_offset(lx, ly, ps.rotation, ps.mirror)
    return _property(key, value, snap(ps.x + dx), snap(ps.y + dy), angle, justify=justify)


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


# --------------------------------------------------------------------------- compiler


class SchematicCompiler(Compiler):
    id = "compiler.kicad_sch"
    version = "0.5"
    kind = ArtifactKind.SCHEMATIC

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        library = ctx.tools.get("kicad_library")
        if library is None:
            library = KicadLibrary()
        elif not isinstance(library, KicadLibrary):
            raise CompileError(f"ctx.tools['kicad_library'] is not a KicadLibrary: {type(library).__name__}")
        node = self.build(ir, library)
        path = ctx.workdir / f"{ir.project.id}.kicad_sch"
        return self._write(ir, path, sexpr.dumps(node))

    # --- pure tree construction (no I/O besides library reads) --------------------

    def build(self, ir: CircuitIR, library: KicadLibrary) -> Node:
        """The complete ``(kicad_sch ...)`` tree for ``ir``. Raises CompileError, never guesses."""
        project_id = ir.project.id
        if not project_id or re.search(r'[\\/:*?"<>|\s]', project_id):
            raise CompileError(f"project id {project_id!r} is not usable as a KiCad file stem")
        if not ir.components:
            # an empty schematic passes ERC with zero violations - that would be evidence about nothing
            raise NothingToCompileError("IR has no components: nothing to draw (the schematic stage needs a component list)")
        root_uuid = sheet_uuid(project_id)
        components = self._checked_components(ir)
        pin_nets = self._pin_net_names(ir)
        placed = self._place(components, library, pin_nets)
        pins_by_key = self._pin_map(placed)
        stacked = self._stacks(placed, pins_by_key, pin_nets)
        self._check_geometry(placed, pins_by_key, stacked)

        no_connects, wires, labels = self._connectivity(ir, placed, pins_by_key, stacked)

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
            self._paper(placed),
            S(
                "title_block",
                S("title", Q(ir.project.name)),
                S("comment", 1, Q(f"generated by {self.id} {self.version} from IR project {project_id}")),
                S("comment", 2, Q(ir.project.description)) if ir.project.description else None,
            ),
            lib_symbols,
            *no_connects,
            *wires,
            *labels,
            *symbols,
            S("sheet_instances", S("path", Q("/"), S("page", Q("1")))),
            S("embedded_fonts", False),
        )

    @staticmethod
    def _paper(placed: dict[str, _PlacedSymbol]) -> Node:
        """``(paper "A4")`` unless the placed extents need a larger sheet (A3 .. A0, else ``"User" W H``)."""
        name, w, h = layout_paper(ps.extent for ps in placed.values())
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
        return sorted(ir.components, key=lambda c: natural_ref_key(c.ref))

    @staticmethod
    def _pin_net_names(ir: CircuitIR) -> dict[tuple[str, str], str]:
        """``(ref, pin) -> net name`` (first net wins; conflicts are refused later in ``_connectivity``)."""
        out: dict[tuple[str, str], str] = {}
        for net in sorted(ir.nets, key=lambda n: n.name):
            for pref in net.pins:
                out.setdefault((pref.component_ref, pref.pin_number), net.name)
        return out

    def _place(
        self, components: list[Component], library: KicadLibrary, pin_nets: dict[tuple[str, str], str]
    ) -> dict[str, _PlacedSymbol]:
        """Resolve every symbol, measure it, and lay the symbols out on a grid the measurements dictate."""
        resolved: dict[str, SymbolDef] = {}
        extents: dict[str, Extent] = {}
        for c in components:
            symbol = load_verified_symbol(c, library)
            pad_pin_types(c, symbol)  # pin numbers + electrical types must be the library's
            label_lengths = {p.number: len(pin_nets[(c.ref, p.number)]) for p in symbol.pins if (c.ref, p.number) in pin_nets}
            resolved[c.ref] = symbol
            extents[c.ref] = symbol_extent(symbol, label_lengths)
        positions = layout_positions(extents)
        placed: dict[str, _PlacedSymbol] = {}
        for c in components:
            x, y = positions[c.ref]
            placed[c.ref] = _PlacedSymbol(
                component=c,
                symbol=resolved[c.ref],
                footprint_id=self._footprint_id(c, library),
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
