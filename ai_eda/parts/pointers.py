"""Datasheet pointers: where the system will look for a component's datasheet, and where that pointer came from.

Invariant: a pointer is evidence of *where the system looked*, never a fact
about the part. Three origins, tried in this order, and nothing else:

1. ``user`` - the user named the URL for this reference designator
   (``--datasheet-url REF=URL``, kept in
   :attr:`~ai_eda.tools.sources.policy.NetworkPolicy.user_urls`). Trusted by
   rule (c) of the network policy: exactly that URL.
2. ``ir`` - the IR's own :class:`~ai_eda.ir.SourceRef`
   (``Component.datasheet``) when it points at something: a URL, an archived
   copy (``document_path`` / ``content_hash``). A bare title is not a pointer.
   Its host is trusted only when the KiCad symbol's ``Datasheet`` field names
   the same host (rule (a)) or the policy already trusts it; otherwise the
   archive refuses the fetch and the check reports that.
3. ``kicad_symbol`` - the ``Datasheet`` property of the component's KiCad
   library symbol, read from the library file on disk by
   :class:`~ai_eda.tools.kicad.library.KicadLibrary` (derived ``extends``
   symbols inherit it; never from model memory). Trusted by rule (a): the
   host of that URL. ``""`` (what KiCad 10 writes) and the legacy ``~`` mean
   "no datasheet".

A URL proposed by a model is never a pointer. A URL that cannot be used
(``hhttps://`` typo, ``ftp:``, no host) is reported with
:func:`~ai_eda.tools.sources.policy.normalise_url`'s reason and never repaired
by guessing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ai_eda.ir import Component, SourceRef
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.sources.policy import host_key, host_of, normalise_url

PointerOrigin = Literal["user", "ir", "kicad_symbol"]

#: the KiCad symbol property that carries the manufacturer's datasheet URL
DATASHEET_PROPERTY = "Datasheet"
#: property values that mean "no datasheet" (KiCad 10 writes the empty string; ``~`` is the legacy marker)
NO_DATASHEET_VALUES: frozenset[str] = frozenset({"", "~"})


class DatasheetPointer(BaseModel):
    """Where a datasheet is expected to be, with the origin of that expectation."""

    ref: SourceRef
    origin: PointerOrigin
    #: where the pointer came from, for a human ("Datasheet property of Regulator_Linear:LM7805_TO220 in Regulator_Linear.kicad_sym")
    note: str
    #: the https URL the archive may attempt (``None`` when the pointer names only an archived copy or its URL is unusable)
    url: str | None = None
    #: what normalisation changed about the raw URL (http upgraded, scheme added)
    url_note: str | None = None
    #: why the raw URL is unusable (reported, never repaired)
    url_error: str | None = None
    #: host key of ``url`` (``ti.com`` for ``www.ti.com``)
    host: str | None = None
    #: why the archive may trust ``host`` - rule (a) for a KiCad Datasheet host, rule (c) for a user URL; ``None`` when the policy alone decides
    trust_reason: str | None = None

    @property
    def fetchable(self) -> bool:
        return self.url is not None


class PointerSearch(BaseModel):
    """The outcome of :func:`locate_datasheet`: the pointer found (if any) and why none was found otherwise."""

    pointer: DatasheetPointer | None = None
    #: why there is no pointer (``None`` when one was found)
    reason: str | None = None
    #: the raw KiCad ``Datasheet`` property value, when the symbol could be read (``None`` otherwise)
    kicad_value: str | None = None
    #: what happened while looking (a KiCad symbol that could not be read, an IR reference without a pointer, ...)
    notes: list[str] = []


def _normalised(raw: str | None) -> tuple[str | None, str | None, str | None]:
    """``(https url, note, error)`` for a raw URL; ``(None, None, None)`` when there is no URL at all."""
    if raw is None or raw.strip().strip('"\'').strip() in NO_DATASHEET_VALUES:
        return None, None, None
    try:
        url, note = normalise_url(raw)
    except ValueError as e:
        return None, None, str(e)
    return url, note, None


def kicad_datasheet_property(component: Component, library: KicadLibrary | None) -> tuple[str | None, str | None, str | None]:
    """``(raw Datasheet value, library file, reason)``: the symbol's property as written in the library file, or why it could not be read.

    The value is returned as it stands in the file (``""`` for none, a
    ``www.`` scheme-less URL, a typo) so the caller can report it verbatim;
    ``reason`` is set when the symbol could not be read at all.
    """
    if library is None:
        return None, None, "no KiCad library available"
    if component.symbol is None:
        return None, None, "component has no symbol reference"
    lib_id = f"{component.symbol.library}:{component.symbol.name}"
    try:
        sym = library.load_symbol(component.symbol)
    except LibraryLookupError:
        return None, None, f"symbol {lib_id} not found in the KiCad libraries"
    except LibraryFormatError as e:
        return None, None, f"symbol {lib_id} is unreadable: {e}"
    return sym.properties.get(DATASHEET_PROPERTY, ""), sym.library_path, None


def _is_pointer(ref: SourceRef | None) -> bool:
    return ref is not None and bool(ref.url or ref.document_path or ref.content_hash)


def locate_datasheet(component: Component, library: KicadLibrary | None = None, user_urls: dict[str, str] | None = None) -> PointerSearch:
    """Find the component's datasheet pointer in the order user URL, IR reference, KiCad ``Datasheet`` property (module docstring)."""
    notes: list[str] = []
    kicad_raw, kicad_file, kicad_reason = kicad_datasheet_property(component, library)
    kicad_url, kicad_note, kicad_error = _normalised(kicad_raw)
    lib_id = f"{component.symbol.library}:{component.symbol.name}" if component.symbol is not None else "?"
    kicad_trust = f"KiCad Datasheet field of {lib_id}" + (f" ({kicad_file})" if kicad_file else "")
    if kicad_reason:
        notes.append(f"KiCad Datasheet property not read: {kicad_reason}")
    elif kicad_error:
        notes.append(f"KiCad Datasheet property of {lib_id} is unusable: {kicad_raw!r}: {kicad_error}")

    # 1. the user's URL for this reference designator
    raw_user = (user_urls or {}).get(component.ref)
    if raw_user is not None:
        url, note, error = _normalised(raw_user)
        ref = SourceRef(title=f"Datasheet of {component.ref} (URL supplied by the user)", url=raw_user)
        pointer = DatasheetPointer(
            ref=ref, origin="user", note=f"URL given by the user for {component.ref}: {raw_user}", url=url, url_note=note, url_error=error,
            host=host_key(host_of(url)) if url else None, trust_reason=f"URL supplied by the user for {component.ref}" if url else None,
        )
        return PointerSearch(pointer=pointer, kicad_value=kicad_raw, notes=notes)

    # 2. the IR's own reference, when it points at something
    ir_ref = component.datasheet
    if _is_pointer(ir_ref):
        assert ir_ref is not None
        url, note, error = _normalised(ir_ref.url)
        host = host_key(host_of(url)) if url else None
        trust = kicad_trust if (host and kicad_url and host == host_key(host_of(kicad_url))) else None
        what = []
        if ir_ref.url:
            what.append(f"url {ir_ref.url}")
        if ir_ref.content_hash:
            what.append(f"archived copy {ir_ref.content_hash}")
        pointer = DatasheetPointer(
            ref=ir_ref, origin="ir", note=f"IR datasheet reference of {component.ref} ({', '.join(what)})", url=url, url_note=note, url_error=error,
            host=host, trust_reason=trust,
        )
        return PointerSearch(pointer=pointer, kicad_value=kicad_raw, notes=notes)
    if ir_ref is not None:
        notes.append(f"IR datasheet reference of {component.ref} names no URL and no archived copy ({ir_ref.title!r}); not a pointer")

    # 3. the KiCad symbol's Datasheet property
    if kicad_reason:
        return PointerSearch(reason=f"IR has no datasheet pointer and the KiCad symbol could not be read ({kicad_reason})", kicad_value=kicad_raw, notes=notes)
    if kicad_url is None and kicad_error is None:
        return PointerSearch(reason=f"IR has no datasheet pointer and the KiCad symbol {lib_id} has an empty Datasheet property", kicad_value=kicad_raw, notes=notes)
    library_name = component.symbol.library if component.symbol is not None else "?"
    ref = SourceRef(title=f"Datasheet of {lib_id} (KiCad Datasheet property)", url=kicad_raw, authority=f"KiCad symbol library {library_name}")
    pointer = DatasheetPointer(
        ref=ref, origin="kicad_symbol", note=f"Datasheet property of KiCad symbol {lib_id}" + (f" in {kicad_file}" if kicad_file else ""),
        url=kicad_url, url_note=kicad_note, url_error=kicad_error, host=host_key(host_of(kicad_url)) if kicad_url else None,
        trust_reason=kicad_trust if kicad_url else None,
    )
    return PointerSearch(pointer=pointer, kicad_value=kicad_raw, notes=notes)


def datasheet_pointer(component: Component, library: KicadLibrary | None = None, user_urls: dict[str, str] | None = None) -> SourceRef | None:
    """The :class:`~ai_eda.ir.SourceRef` the system will look at for this component's datasheet, or ``None`` (see :func:`locate_datasheet`)."""
    found = locate_datasheet(component, library, user_urls).pointer
    return found.ref if found is not None else None


__all__ = [
    "DATASHEET_PROPERTY",
    "NO_DATASHEET_VALUES",
    "DatasheetPointer",
    "PointerOrigin",
    "PointerSearch",
    "datasheet_pointer",
    "kicad_datasheet_property",
    "locate_datasheet",
]
