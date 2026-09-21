"""KiCad symbol / footprint library access.

Invariant (CLAUDE.md #2): a compiler never emits a symbol or footprint it did
not find on disk, and pin numbers / pad geometry come from the library file,
never from model memory. This module is the only reader of ``.kicad_sym`` /
``.kicad_mod`` files; it returns *parsed* definitions (:class:`SymbolDef`,
:class:`FootprintDef`) whose ``node`` is the exact library tree, ready to be
embedded in a schematic's ``lib_symbols`` or a board's ``footprint`` list.

* ``resolve_*`` turn an unverified :class:`LibraryRef` into a verified one by
  actually parsing the entry (``verified=False`` when it does not exist).
* ``load_symbol`` resolves ``(extends "PARENT")`` recursively and merges the
  parent's body so the returned tree is self-contained (what KiCad itself
  embeds in a schematic). A malformed library raises :class:`LibraryFormatError`
  instead of being skipped - an unknown node or pin type makes the whole
  library unloadable in KiCad too.
* Parsed library files are cached per :class:`KicadLibrary` instance
  (``Device.kicad_sym`` is 2.5 MB, ``Connector_Generic.kicad_sym`` 7.3 MB).
  Loading one symbol first tries a text-level extraction of just that
  ``(symbol ...)`` block (milliseconds), falling back to the full parse.

Coordinates: symbol pins are in the symbol's own frame with **Y up** (as in
the library); footprint pads are in the footprint frame with **Y down**
(board convention). The schematic compiler must negate pin ``y``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eda.errors import AiEdaError, CompileError
from ai_eda.ir import LibraryRef, PinElectricalType
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.sexpr import QStr

__all__ = [
    "KicadLibrary",
    "LibraryLookupError",
    "LibraryFormatError",
    "SymbolDef",
    "SymbolPin",
    "FootprintDef",
    "Pad",
    "BBox",
    "KICAD_PIN_TYPES",
    "kicad_pin_type_to_ir",
    "ir_pin_type_to_kicad",
]


class LibraryLookupError(LookupError, AiEdaError):
    """The requested library, symbol or footprint does not exist on disk."""


class LibraryFormatError(CompileError):
    """A library file exists but is malformed or uses a construct we refuse to guess about."""


# --------------------------------------------------------------------------- pin type mapping

#: The 12 electrical types KiCad 10 accepts (string-identical to PinElectricalType values).
KICAD_PIN_TYPES: frozenset[str] = frozenset(t.value for t in PinElectricalType)


def kicad_pin_type_to_ir(kicad_type: str) -> PinElectricalType:
    """``"passive"`` -> ``PinElectricalType.PASSIVE``; unknown -> LibraryFormatError."""
    try:
        return PinElectricalType(str(kicad_type))
    except ValueError as exc:
        raise LibraryFormatError(f"unknown KiCad pin electrical type {kicad_type!r}") from exc


def ir_pin_type_to_kicad(pin_type: PinElectricalType | str) -> str:
    """``PinElectricalType.PASSIVE`` -> ``"passive"`` (the bare atom KiCad writes)."""
    return PinElectricalType(pin_type).value


# --------------------------------------------------------------------------- definitions


@dataclass(frozen=True, slots=True)
class SymbolPin:
    """One pin of a library symbol, in library coordinates (mm, Y up)."""

    number: str
    name: str
    electrical_type: str  # one of KICAD_PIN_TYPES
    x: float
    y: float
    angle: float  # 0 = pin line runs +x from the connection point, 90 = +y (up), 180, 270
    length: float
    unit: int  # 0 = common to all units
    hidden: bool
    graphic_style: str = "line"
    body_style: int = 1  # 1 = normal, 2 = De Morgan alternate, 0 = both

    @property
    def ir_type(self) -> PinElectricalType:
        return kicad_pin_type_to_ir(self.electrical_type)


@dataclass(slots=True)
class SymbolDef:
    """A library symbol with ``(extends)`` resolved. ``node`` is read-only shared data."""

    lib_id: str  # "Device:R"
    name: str  # "R"
    node: list  # (symbol "R" ...) - self-contained, no (extends)
    pins: list[SymbolPin]  # body styles 0 and 1, all units
    units: list[int]
    is_power: bool
    properties: dict[str, str]
    extends_from: str | None = None  # parent name when the library entry was derived
    library_path: str | None = None

    def pin(self, number: str) -> SymbolPin | None:
        for p in self.pins:
            if p.number == number:
                return p
        return None

    def lib_symbols_entry(self) -> list:
        """Deep copy renamed ``"Lib:Name"`` - exactly what ``(lib_symbols ...)`` holds.

        Unit sub-symbols keep their bare ``Name_u_b`` names, as KiCad writes them.
        """
        node = sexpr.deep_copy(self.node)
        node[1] = QStr(self.lib_id)
        return node


@dataclass(frozen=True, slots=True)
class Pad:
    """One footprint pad, in footprint coordinates (mm, Y down)."""

    number: str  # "" for unnumbered (e.g. mounting holes)
    pad_type: str  # smd | thru_hole | np_thru_hole | connect
    shape: str  # circle | rect | oval | roundrect | trapezoid | custom
    x: float
    y: float
    rotation: float
    size_w: float
    size_h: float
    drill: float | None
    layers: list[str]
    roundrect_rratio: float | None = None


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(slots=True)
class FootprintDef:
    """A library footprint. ``node`` is the exact ``(footprint "Name" ...)`` tree (read-only)."""

    lib_id: str  # "Resistor_SMD:R_0603_1608Metric"
    name: str
    node: list
    pads: list[Pad]
    attr: str  # smd | through_hole | unspecified
    courtyard: BBox | None  # bounding box of F.CrtYd/B.CrtYd graphics, if any
    properties: dict[str, str] = field(default_factory=dict)
    descr: str = ""
    tags: str = ""
    attributes: list[str] = field(default_factory=list)  # every token of (attr ...)
    library_path: str | None = None

    def pad(self, number: str) -> Pad | None:
        for p in self.pads:
            if p.number == number:
                return p
        return None


# --------------------------------------------------------------------------- roots


def _default_library_roots() -> list[Path]:
    """``share/kicad`` roots to search: ``$KICAD*_SYMBOL_DIR``'s parent, then the installed versions.

    Versions are ordered numerically (``10.0`` before ``9.0``) and the
    installation whose ``kicad-cli`` this project actually runs
    (:func:`~ai_eda.tools.kicad.cli.find_kicad_cli`) comes first, so the
    libraries embedded in our files are the ones that binary compares them
    with (otherwise ERC/DRC report ``lib_*_mismatch``).
    """
    from ai_eda.tools.kicad.cli import find_kicad_cli, install_root, version_dirs

    roots: list[Path] = []
    env = os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    if env:
        roots.append(Path(env).parent)
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "KiCad"
    shares = [ver / "share" / "kicad" for ver in version_dirs(base) if (ver / "share" / "kicad").exists()]
    cli_root = install_root(find_kicad_cli())
    if cli_root is not None:
        cli_share = cli_root / "share" / "kicad"
        shares = [cli_share] + [s for s in shares if s.resolve() != cli_share.resolve()]
    roots.extend(shares)
    return roots


# --------------------------------------------------------------------------- symbol parsing helpers

_UNIT_NAME_RE = re.compile(r"^(?P<base>.+)_(?P<unit>\d+)_(?P<body>\d+)$")


def _qstr_at(node: list, index: int, what: str) -> str:
    if index >= len(node) or isinstance(node[index], list):
        raise LibraryFormatError(f"{what}: missing string at position {index} in ({node[0]} ...)")
    return str(node[index])


def _unit_of(sub_symbol_name: str, symbol_name: str) -> tuple[int, int]:
    """``"Conn_01x03_1_1"`` -> (unit 1, body_style 1)."""
    m = _UNIT_NAME_RE.match(sub_symbol_name)
    if not m:
        raise LibraryFormatError(f"symbol {symbol_name!r}: unit sub-symbol {sub_symbol_name!r} is not named NAME_u_b")
    return int(m.group("unit")), int(m.group("body"))


def _properties_of(node: list) -> dict[str, str]:
    props: dict[str, str] = {}
    for prop in sexpr.find_all(node, "property"):
        if len(prop) < 3 or isinstance(prop[1], list) or isinstance(prop[2], list):
            raise LibraryFormatError(f"malformed property node in {str(node[1]) if len(node) > 1 else '?'}")
        props[str(prop[1])] = str(prop[2])
    return props


def _is_hidden(node: list) -> bool:
    return sexpr.get(node, "hide") == "yes" or "hide" in sexpr.args(node)


def _parse_pins(sym: list) -> list[SymbolPin]:
    name = str(sym[1])
    pins: list[SymbolPin] = []
    for unit_sym in sexpr.find_all(sym, "symbol"):
        unit, body = _unit_of(_qstr_at(unit_sym, 1, name), name)
        if body > 1:
            continue  # De Morgan alternates repeat the same pins with other graphics
        for pin in sexpr.find_all(unit_sym, "pin"):
            etype = _qstr_at(pin, 1, f"{name} pin")
            if etype not in KICAD_PIN_TYPES:
                raise LibraryFormatError(f"symbol {name!r}: unknown pin electrical type {etype!r}")
            style = _qstr_at(pin, 2, f"{name} pin")
            at = sexpr.find(pin, "at")
            number = sexpr.find(pin, "number")
            pname = sexpr.find(pin, "name")
            if at is None or len(at) < 4 or number is None or len(number) < 2:
                raise LibraryFormatError(f"symbol {name!r}: pin without (at x y angle) or (number ...)")
            pins.append(
                SymbolPin(
                    number=str(number[1]),
                    name=str(pname[1]) if pname is not None and len(pname) > 1 else "",
                    electrical_type=etype,
                    x=sexpr.to_float(at[1]),
                    y=sexpr.to_float(at[2]),
                    angle=sexpr.to_float(at[3]),
                    length=sexpr.to_float(sexpr.get(pin, "length", 1, "0")),
                    unit=unit,
                    hidden=_is_hidden(pin),
                    graphic_style=style,
                    body_style=body,
                )
            )
    return pins


def _units_of(sym: list) -> list[int]:
    name = str(sym[1])
    units = sorted({_unit_of(_qstr_at(u, 1, name), name)[0] for u in sexpr.find_all(sym, "symbol")} - {0})
    return units or [1]


def _flatten(child: list, parent: list) -> list:
    """Merge a derived symbol into a deep copy of its (already resolved) parent.

    Mirrors what KiCad's ``LIB_SYMBOL::Flatten`` produces inside ``lib_symbols``:
    the parent's body, pins and unit sub-symbols renamed to the child, the
    child's properties overriding the parent's by key, no ``(extends)``.
    """
    child_name = str(child[1])
    parent_name = str(parent[1])
    node = sexpr.deep_copy(parent)
    node[1] = QStr(child_name)
    # property overrides (replace in place, append new keys after the last property)
    overrides = {str(p[1]): p for p in sexpr.find_all(child, "property")}
    last_prop = 0
    for i, item in enumerate(node):
        if isinstance(item, list) and item and item[0] == "property":
            last_prop = i
            key = str(item[1])
            if key in overrides:
                node[i] = sexpr.deep_copy(overrides.pop(key))
    for prop in overrides.values():
        last_prop += 1
        node.insert(last_prop, sexpr.deep_copy(prop))
    # unit sub-symbols PARENT_u_b -> CHILD_u_b
    prefix = parent_name + "_"
    for unit_sym in sexpr.find_all(node, "symbol"):
        uname = str(unit_sym[1])
        if uname.startswith(prefix):
            unit_sym[1] = QStr(child_name + uname[len(parent_name):])
    node[:] = [it for it in node if not (isinstance(it, list) and it and it[0] == "extends")]
    return node


# --------------------------------------------------------------------------- footprint parsing helpers


def _parse_pads(fp: list) -> list[Pad]:
    name = str(fp[1])
    pads: list[Pad] = []
    for pad in sexpr.find_all(fp, "pad"):
        if len(pad) < 4 or any(isinstance(pad[i], list) for i in (1, 2, 3)):
            raise LibraryFormatError(f"footprint {name!r}: malformed (pad ...) node")
        at = sexpr.find(pad, "at")
        size = sexpr.find(pad, "size")
        layers = sexpr.find(pad, "layers")
        if at is None or len(at) < 3 or size is None or len(size) < 3 or layers is None:
            raise LibraryFormatError(f"footprint {name!r}: pad {str(pad[1])!r} lacks (at)/(size)/(layers)")
        drill: float | None = None
        drill_node = sexpr.find(pad, "drill")
        if drill_node is not None:
            for atom in sexpr.args(drill_node):
                try:
                    drill = float(atom)
                    break
                except ValueError:
                    continue  # "oval" etc.
        rr = sexpr.get(pad, "roundrect_rratio")
        pads.append(
            Pad(
                number=str(pad[1]),
                pad_type=str(pad[2]),
                shape=str(pad[3]),
                x=sexpr.to_float(at[1]),
                y=sexpr.to_float(at[2]),
                rotation=sexpr.to_float(at[3]) if len(at) > 3 else 0.0,
                size_w=sexpr.to_float(size[1]),
                size_h=sexpr.to_float(size[2]),
                drill=drill,
                layers=[str(layer) for layer in sexpr.args(layers)],
                roundrect_rratio=sexpr.to_float(rr) if rr is not None else None,
            )
        )
    return pads


_COURTYARD_LAYERS = frozenset({"F.CrtYd", "B.CrtYd"})


def _courtyard_bbox(fp: list) -> BBox | None:
    xs: list[float] = []
    ys: list[float] = []

    def add(node: list | None) -> None:
        if node is not None and len(node) >= 3:
            xs.append(sexpr.to_float(node[1]))
            ys.append(sexpr.to_float(node[2]))

    for item in fp:
        if not isinstance(item, list) or not item or sexpr.get(item, "layer") not in _COURTYARD_LAYERS:
            continue
        kind = item[0]
        if kind in ("fp_line", "fp_rect"):
            add(sexpr.find(item, "start"))
            add(sexpr.find(item, "end"))
        elif kind == "fp_arc":
            for h in ("start", "mid", "end"):
                add(sexpr.find(item, h))
        elif kind == "fp_circle":
            center = sexpr.find(item, "center")
            end = sexpr.find(item, "end")
            if center is not None and end is not None and len(center) >= 3 and len(end) >= 3:
                cx, cy = sexpr.to_float(center[1]), sexpr.to_float(center[2])
                r = ((sexpr.to_float(end[1]) - cx) ** 2 + (sexpr.to_float(end[2]) - cy) ** 2) ** 0.5
                xs += [cx - r, cx + r]
                ys += [cy - r, cy + r]
        elif kind == "fp_poly":
            pts = sexpr.find(item, "pts")
            if pts is not None:
                for xy in sexpr.find_all(pts, "xy"):
                    add(xy)
    if not xs:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- the library


class KicadLibrary:
    """Reads the installed KiCad libraries. Instances cache parsed files."""

    def __init__(self, roots: list[Path] | None = None) -> None:
        self.roots = roots or _default_library_roots()
        self._lib_cache: dict[Path, list] = {}
        self._symbol_cache: dict[tuple[str, str], SymbolDef] = {}
        self._footprint_cache: dict[tuple[str, str], FootprintDef] = {}

    # --- symbols -------------------------------------------------------------

    def symbol_file(self, library: str) -> Path | None:
        for root in self.roots:
            p = root / "symbols" / f"{library}.kicad_sym"
            if p.exists():
                return p
        return None

    def symbol_library(self, library: str) -> list:
        """The full parsed ``(kicad_symbol_lib ...)`` tree (cached; read-only)."""
        path = self.symbol_file(library)
        if path is None:
            raise LibraryLookupError(f"symbol library {library!r} not found under {[str(r) for r in self.roots]}")
        return self._parsed(path)

    def symbol_names(self, library: str) -> list[str]:
        lib = self.symbol_library(library)
        return [str(s[1]) for s in sexpr.find_all(lib, "symbol") if len(s) > 1]

    def resolve_symbol(self, ref: LibraryRef) -> LibraryRef:
        """Verify by parsing the entry. Unknown library/symbol -> ``verified=False``."""
        try:
            sym = self.load_symbol(ref)
        except LibraryLookupError:
            return ref.model_copy(update={"verified": False, "library_path": None})
        return ref.model_copy(update={"verified": True, "library_path": sym.library_path})

    def load_symbol(self, ref: LibraryRef) -> SymbolDef:
        key = (ref.library, ref.name)
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        path = self.symbol_file(ref.library)
        if path is None:
            raise LibraryLookupError(f"symbol library {ref.library!r} not found under {[str(r) for r in self.roots]}")
        node, extends_from = self._resolved_symbol_node(path, ref.name, chain=())
        try:
            sym = SymbolDef(
                lib_id=f"{ref.library}:{ref.name}",
                name=ref.name,
                node=node,
                pins=_parse_pins(node),
                units=_units_of(node),
                is_power=sexpr.find(node, "power") is not None,
                properties=_properties_of(node),
                extends_from=extends_from,
                library_path=str(path),
            )
        except sexpr.SExprError as exc:
            raise LibraryFormatError(f"{path}: symbol {ref.name!r}: {exc}") from exc
        self._symbol_cache[key] = sym
        return sym

    def symbol_pins(self, ref: LibraryRef) -> list[SymbolPin]:
        """Pins read from the library file (never from memory)."""
        return self.load_symbol(ref).pins

    # --- footprints ----------------------------------------------------------

    def footprint_file(self, library: str, name: str) -> Path | None:
        for root in self.roots:
            p = root / "footprints" / f"{library}.pretty" / f"{name}.kicad_mod"
            if p.exists():
                return p
        return None

    def resolve_footprint(self, ref: LibraryRef) -> LibraryRef:
        try:
            fp = self.load_footprint(ref)
        except LibraryLookupError:
            return ref.model_copy(update={"verified": False, "library_path": None})
        return ref.model_copy(update={"verified": True, "library_path": fp.library_path})

    def load_footprint(self, ref: LibraryRef) -> FootprintDef:
        key = (ref.library, ref.name)
        cached = self._footprint_cache.get(key)
        if cached is not None:
            return cached
        path = self.footprint_file(ref.library, ref.name)
        if path is None:
            raise LibraryLookupError(f"footprint {ref.library}:{ref.name} not found under {[str(r) for r in self.roots]}")
        node = self._parsed(path)
        if sexpr.head(node) != "footprint" or len(node) < 2 or isinstance(node[1], list):
            raise LibraryFormatError(f"{path}: root node is not (footprint \"NAME\" ...)")
        if str(node[1]) != ref.name:
            raise LibraryFormatError(f"{path}: footprint is named {str(node[1])!r}, expected {ref.name!r}")
        attr_node = sexpr.find(node, "attr")
        attributes = [str(a) for a in sexpr.args(attr_node)] if attr_node is not None else []
        attr = next((a for a in attributes if a in ("smd", "through_hole")), "unspecified")
        try:
            fp = FootprintDef(
                lib_id=f"{ref.library}:{ref.name}",
                name=ref.name,
                node=node,
                pads=_parse_pads(node),
                attr=attr,
                courtyard=_courtyard_bbox(node),
                properties=_properties_of(node),
                descr=str(sexpr.get(node, "descr", 1, "")),
                tags=str(sexpr.get(node, "tags", 1, "")),
                attributes=attributes,
                library_path=str(path),
            )
        except sexpr.SExprError as exc:
            raise LibraryFormatError(f"{path}: {exc}") from exc
        self._footprint_cache[key] = fp
        return fp

    def footprint_pads(self, ref: LibraryRef) -> list[Pad]:
        """Pad geometry read from the ``.kicad_mod`` file (never from memory)."""
        return self.load_footprint(ref).pads

    # --- internals -----------------------------------------------------------

    def _parsed(self, path: Path) -> list:
        node = self._lib_cache.get(path)
        if node is None:
            try:
                node = sexpr.parse_file(path)
            except (sexpr.SExprError, UnicodeDecodeError) as exc:
                raise LibraryFormatError(f"{path}: {exc}") from exc
            self._lib_cache[path] = node
        return node

    def _symbol_node(self, path: Path, name: str) -> list | None:
        """The raw ``(symbol "name" ...)`` node from ``path`` (fast text extraction, then full parse)."""
        if path in self._lib_cache:
            return self._find_symbol(self._lib_cache[path], name)
        block = _extract_symbol_block(path, name)
        if block is not None:
            try:
                node = sexpr.parse(block)
            except sexpr.SExprError:
                node = None
            if node is not None and sexpr.head(node) == "symbol" and len(node) > 1 and node[1] == name:
                return node
        return self._find_symbol(self._parsed(path), name)

    @staticmethod
    def _find_symbol(lib: list, name: str) -> list | None:
        if sexpr.head(lib) != "kicad_symbol_lib":
            raise LibraryFormatError("root node is not (kicad_symbol_lib ...)")
        for s in sexpr.find_all(lib, "symbol"):
            if len(s) > 1 and s[1] == name:
                return s
        return None

    def _resolved_symbol_node(self, path: Path, name: str, chain: tuple[str, ...]) -> tuple[list, str | None]:
        """Symbol ``name`` with ``(extends)`` flattened recursively. Returns (node, direct parent or None)."""
        if name in chain:
            raise LibraryFormatError(f"{path}: circular (extends) chain {' -> '.join((*chain, name))}")
        node = self._symbol_node(path, name)
        if node is None:
            if chain:
                raise LibraryFormatError(f"{path}: {chain[-1]!r} extends missing symbol {name!r}")
            raise LibraryLookupError(f"symbol {name!r} not found in {path}")
        parent_name = sexpr.get(node, "extends")
        if parent_name is None:
            return node, None
        parent, _ = self._resolved_symbol_node(path, str(parent_name), (*chain, name))
        return _flatten(node, parent), str(parent_name)


_BLOCK_CLOSE = "\n\t)"


def _extract_symbol_block(path: Path, name: str) -> str | None:
    """Cut the top-level ``(symbol "name" ...)`` block out of a KiCad-formatted file.

    Relies on KiCad's layout (top-level symbols open at ``\\n\\t(symbol "..."`` and
    close at ``\\n\\t)``); any deviation returns None and the caller falls back
    to a full parse. Quoted strings never contain raw newlines, so the first
    depth-1 close after the opener is the symbol's own.
    """
    text = path.read_bytes().decode("utf-8")
    opener = re.compile(r"\n\t\(symbol " + re.escape('"' + sexpr.escape(name) + '"') + r"\r?\n")
    m = opener.search(text)
    if m is None:
        return None
    end = text.find(_BLOCK_CLOSE, m.end())
    if end == -1:
        return None
    return text[m.start() + 1 : end + len(_BLOCK_CLOSE)]
