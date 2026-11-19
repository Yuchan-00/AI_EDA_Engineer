"""Library truth for pins, shared by the schematic and the PCB compiler.

Invariant (CLAUDE.md #2): pin numbers *and* electrical types come from the
KiCad symbol on disk. The IR may repeat them, but it may not disagree with
them: a component whose IR pins differ from the library symbol's pins in
number or electrical type is a :class:`~ai_eda.errors.CompileError` in both
compilers, so the ``.kicad_sch`` (which embeds the library symbol) and the
``.kicad_pcb`` (which writes ``(pintype ..)`` on every pad) can never carry
different pin types for one IR.

The one IR-side statement that is allowed to differ is
``PinElectricalType.NO_CONNECT``: it is a *design* decision ("this pin is
deliberately left open"), not a claim about the library. KiCad records it the
same way on the board - ``(pintype "passive+no_connect")`` - which is what
:func:`pad_pin_types` returns for such pins.

Multi-unit symbols (a dual op-amp's units A, B and the power unit C) are one
component with one pin list: the IR names every pin of every unit, and the
schematic draws one instance per unit, each with its own pins
(:func:`unit_pins`). A multi-unit symbol with common (unit 0) pins is refused:
those pins would belong to every unit instance at once.

Pads pair with pins by number (:func:`match_pads`, the one rule the PCB
compiler, the design file and the RF parts table apply): every pin needs a
pad, and every numbered plated pad must be a pin - except a mounting pad
``MP`` (:data:`MOUNTING_PAD`) on a symbol without an ``MP`` pin, which is
mechanical copper and compiled without a net, as an unnumbered pad always
was. ``Device:R_Potentiometer_MountingPin`` has the pin (``MP``,
"MountPin"), so its mounting pads keep the net of that pin.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ai_eda.compilers.schematic_layout import natural_ref_key as natural_pin_key
from ai_eda.errors import CompileError
from ai_eda.ir import Component, PinElectricalType
from ai_eda.tools.kicad.library import KicadLibrary, LibraryLookupError, Pad, SymbolDef, SymbolPin

__all__ = ["load_verified_symbol", "match_pads", "pad_pin_types", "unit_pins", "MOUNTING_PAD", "NO_CONNECT_SUFFIX", "PadMatch"]

#: suffix KiCad appends to a pad's pintype when the schematic pin carries a no-connect flag
NO_CONNECT_SUFFIX = "+no_connect"

#: the KiCad library's pad and pin number of a part's mechanical mounting pins (``Device:R_Potentiometer_MountingPin``'s pin
#: "MountPin", the ``Connector_Generic_MountingPin`` connectors' pin "MountPin"; the potentiometer footprints' two pads)
MOUNTING_PAD = "MP"


@dataclass(frozen=True, slots=True)
class PadMatch:
    """How a footprint's pads pair with a symbol's pin numbers (:func:`match_pads`)."""

    #: pin numbers no pad carries: refused
    without_pad: frozenset[str]
    #: numbered plated pads that are no pin and no mechanical mounting pad: refused
    extra: frozenset[str]
    #: plated :data:`MOUNTING_PAD` pads of a symbol without that pin: mechanical, compiled without a net
    mounting: int
    #: unnumbered copper pads (a slide switch's mounting lugs): no pin can name them, compiled without a net
    unnumbered: int

    def note(self, ref: str) -> str | None:
        """The one note naming ``ref``'s pads that are left without a net, or ``None`` when every copper pad is a pin."""
        parts = ([f"mounting pad(s) {MOUNTING_PAD}"] if self.mounting else []) + ([f"{self.unnumbered} unnumbered pad(s)"] if self.unnumbered else [])
        return f"{ref}: {' and '.join(parts)} left without a net (mechanical)" if parts else None


def match_pads(pins: Iterable[str], pads: Iterable[Pad]) -> PadMatch:
    """Pair the pin numbers of a symbol (or an IR component) with the pads of its footprint (module docstring).

    A pad with a number is a pin's pad when the numbers are equal; a numbered NPTH hole counts as a pad but is never
    copper, so it need not be a pin. A plated :data:`MOUNTING_PAD` pad is a pin's pad only when the pins have that
    number; otherwise it is mechanical (``mounting``). A pad without a number is never a pin's: it is counted when it is
    copper (``unnumbered``), not when it is a paste-only aperture.
    """
    pins = set(pins)
    pads = list(pads)
    plated = [p for p in pads if p.pad_type != "np_thru_hole"]
    numbered = {p.number for p in plated if p.number}
    return PadMatch(
        without_pad=frozenset(pins - {p.number for p in pads if p.number}),
        extra=frozenset(numbered - pins - {MOUNTING_PAD}),
        mounting=0 if MOUNTING_PAD in pins else sum(1 for p in plated if p.number == MOUNTING_PAD),
        unnumbered=sum(1 for p in plated if not p.number and any(layer.endswith(".Cu") for layer in p.layers)),
    )


def load_verified_symbol(c: Component, library: KicadLibrary, *, multi_unit: bool = False) -> SymbolDef:
    """The library symbol of ``c``, or CompileError (no symbol, unverified, not on disk; multi-unit unless ``multi_unit``,
    and with ``multi_unit`` a multi-unit symbol with common (unit 0) pins)."""
    if c.symbol is None:
        raise CompileError(f"{c.ref}: no KiCad symbol assigned")
    lib_id = f"{c.symbol.library}:{c.symbol.name}"
    if not c.symbol.verified:
        raise CompileError(f"{c.ref}: symbol {lib_id!r} is not verified against a KiCad library (resolve it first)")
    try:
        symbol = library.load_symbol(c.symbol)
    except LibraryLookupError as exc:
        raise CompileError(f"{c.ref}: symbol {lib_id!r} not found in the installed KiCad libraries: {exc}") from exc
    if symbol.units != [1]:
        if not multi_unit:
            raise CompileError(f"{c.ref}: symbol {lib_id!r} has units {symbol.units}; multi-unit symbols are not supported yet")
        common = sorted({p.number for p in symbol.pins if p.unit == 0}, key=natural_pin_key)
        if common:
            raise CompileError(f"{c.ref}: {lib_id} has common (unit 0) pins {common}; multi-unit symbols with common pins are not supported")
    return symbol


def unit_pins(symbol: SymbolDef, unit: int) -> list[SymbolPin]:
    """The library pins one instance of unit ``unit`` carries: its own and the common (unit 0) ones, in library order.
    For a single-unit symbol (unit 1) that is every pin."""
    return [p for p in symbol.pins if p.unit in (0, unit)]


def pad_pin_types(c: Component, symbol: SymbolDef) -> dict[str, str]:
    """``{pin number: KiCad pintype atom}`` for ``c``, verified against ``symbol``.

    Raises CompileError when the IR pin numbers are not exactly the library's,
    or when an IR pin's electrical type differs from the library's (except the
    IR marking a pin ``no_connect``, which yields ``"<library type>+no_connect"``).
    """
    numbers = [p.number for p in c.pins]
    if len(set(numbers)) != len(numbers):
        raise CompileError(f"{c.ref}: duplicate pin numbers in IR: {sorted(numbers, key=natural_pin_key)}")
    ir_pins = set(numbers)
    lib_numbers = [p.number for p in symbol.pins]
    if len(set(lib_numbers)) != len(lib_numbers):
        repeated = sorted({n for n in lib_numbers if lib_numbers.count(n) > 1}, key=natural_pin_key)
        raise CompileError(f"{c.ref}: library symbol {symbol.lib_id!r} repeats pin number(s) {repeated} (stacked pins); unsupported - one physical pin per number")
    lib_pins = set(lib_numbers)
    if ir_pins != lib_pins:
        raise CompileError(
            f"{c.ref}: IR pins {sorted(ir_pins, key=natural_pin_key)} do not match library symbol {symbol.lib_id!r} "
            f"pins {sorted(lib_pins, key=natural_pin_key)} (missing in IR: {sorted(lib_pins - ir_pins, key=natural_pin_key)}, "
            f"unknown to library: {sorted(ir_pins - lib_pins, key=natural_pin_key)})"
        )
    lib_types = {p.number: p.electrical_type for p in symbol.pins}
    types: dict[str, str] = {}
    mismatches: list[str] = []
    for pin in c.pins:
        lib_type = lib_types[pin.number]
        ir_type = PinElectricalType(pin.electrical_type).value
        if ir_type == lib_type:
            types[pin.number] = lib_type
        elif pin.electrical_type is PinElectricalType.NO_CONNECT:
            types[pin.number] = lib_type + NO_CONNECT_SUFFIX
        else:
            mismatches.append(f"pin {pin.number}: IR {ir_type} vs library {lib_type}")
    if mismatches:
        raise CompileError(
            f"{c.ref}: IR pin electrical types differ from library symbol {symbol.lib_id!r} ({'; '.join(mismatches)}); "
            "pin types come from the library, fix the IR"
        )
    return types
