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
"""

from __future__ import annotations

from ai_eda.compilers.schematic_layout import natural_ref_key as natural_pin_key
from ai_eda.errors import CompileError
from ai_eda.ir import Component, PinElectricalType
from ai_eda.tools.kicad.library import KicadLibrary, LibraryLookupError, SymbolDef

__all__ = ["load_verified_symbol", "pad_pin_types", "NO_CONNECT_SUFFIX"]

#: suffix KiCad appends to a pad's pintype when the schematic pin carries a no-connect flag
NO_CONNECT_SUFFIX = "+no_connect"


def load_verified_symbol(c: Component, library: KicadLibrary) -> SymbolDef:
    """The library symbol of ``c``, or CompileError (no symbol, unverified, not on disk, multi-unit)."""
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
        raise CompileError(f"{c.ref}: symbol {lib_id!r} has units {symbol.units}; multi-unit symbols are not supported yet")
    return symbol


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
