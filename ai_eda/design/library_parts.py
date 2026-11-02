"""Parts a template instantiates: symbol, footprint and pins come from the KiCad library on disk, never from memory.

Invariant: :func:`library_component` refuses (:class:`TemplateRefusal`)
when the symbol or footprint is not found, and copies the pins - number,
name, electrical type - from the parsed ``.kicad_sym`` with authoritative
provenance whose :class:`~ai_eda.ir.SourceRef` carries the file's sha256, so
a reviewer can prove which library file the pins came from. A template that
needs a pin by name (an LED's anode ``A`` / cathode ``K``) asks
:func:`pin_by_name` and refuses when the library spells it differently: the
failure mode is *no design*, never a guessed netlist.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.ir import Component, LibraryRef, Pin, Provenance, ProvenanceKind, SourceRef
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError


class TemplateRefusal(Exception):
    """The template cannot be instantiated from what is on disk; the agent reports the reason and proposes nothing."""


def library_component(
    library: KicadLibrary,
    ref: str,
    value: str,
    description: str,
    symbol: tuple[str, str],
    footprint: tuple[str, str],
    provenance: Provenance,
    serves: list[str],
) -> Component:
    """A component whose symbol / footprint are verified in ``library`` and whose pins are the symbol's."""
    sym_ref = library.resolve_symbol(LibraryRef(library=symbol[0], name=symbol[1]))
    if not sym_ref.verified:
        raise TemplateRefusal(f"symbol {symbol[0]}:{symbol[1]} not found in the KiCad libraries {[str(r) for r in library.roots]}")
    fp_ref = library.resolve_footprint(LibraryRef(library=footprint[0], name=footprint[1]))
    if not fp_ref.verified:
        raise TemplateRefusal(f"footprint {footprint[0]}:{footprint[1]} not found in the KiCad libraries {[str(r) for r in library.roots]}")
    try:
        sym = library.load_symbol(sym_ref)
    except LibraryFormatError as e:
        raise TemplateRefusal(str(e)) from e
    path = Path(sym.library_path) if sym.library_path else None
    source = SourceRef(
        title=f"KiCad symbol {sym.lib_id}",
        authority="KiCad library",
        document_path=sym.library_path,
        content_hash=SourceRef.hash_bytes(path.read_bytes()) if path is not None and path.is_file() else None,
    )
    pins: list[Pin] = []
    seen: set[str] = set()
    for p in sym.pins:
        if p.number in seen:
            continue  # an alternate body style repeats a pin; the number is the pad
        seen.add(p.number)
        pins.append(Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=source)))
    if not pins:
        raise TemplateRefusal(f"symbol {sym.lib_id} has no pins")
    return Component(
        ref=ref, value=value, description=description, pins=pins, symbol=sym_ref, footprint=fp_ref, provenance=provenance, serves_requirements=list(serves),
    )


def pin_by_name(component: Component, name: str) -> str | None:
    """The pin number carrying ``name`` (exactly), or ``None``."""
    for p in component.pins:
        if p.name == name:
            return p.number
    return None


def two_terminals(component: Component) -> tuple[str, str]:
    """The two pin numbers of a two-terminal part in library order; refuses any other pin count."""
    if len(component.pins) != 2:
        raise TemplateRefusal(f"{component.ref} ({component.symbol.library}:{component.symbol.name}) has {len(component.pins)} pin(s), a two-terminal part was expected")  # type: ignore[union-attr]
    return component.pins[0].number, component.pins[1].number


def require_pins(component: Component, numbers: tuple[str, ...]) -> None:
    """Refuse unless the component has every pin number listed."""
    missing = [n for n in numbers if component.pin(n) is None]
    if missing:
        raise TemplateRefusal(f"{component.ref} ({component.symbol.library}:{component.symbol.name}) has no pin(s) {missing}; it has {[p.number for p in component.pins]}")  # type: ignore[union-attr]


__all__ = ["TemplateRefusal", "library_component", "pin_by_name", "require_pins", "two_terminals"]
