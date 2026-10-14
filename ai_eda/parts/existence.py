"""Component existence check: does this part exist in the KiCad libraries, in an archived manufacturer datasheet, and in the catalog?

Invariant: the check reports what deterministic tools could confirm and
never guesses. It produces one ``component.existence.<ref>``
:class:`~ai_eda.ir.ValidationResult` (``tool="parts.existence"``) whose
status is the worst of six sub-checks, each listed in ``details.checks``:

1. ``symbol`` - the IR's symbol reference parsed from a library file on disk
   (:meth:`~ai_eda.tools.kicad.library.KicadLibrary.resolve_symbol`); FAIL
   when the library entry does not exist or is malformed (the design
   references something that is not there), NOT_VERIFIED when there is no
   reference or no library. The IR is not mutated - the reference's own
   ``verified`` flag is reported, not repaired.
2. ``footprint`` - the same for the footprint.
3. ``datasheet_pointer`` - where a datasheet is expected
   (:func:`~ai_eda.parts.pointers.locate_datasheet`: user URL, IR reference,
   KiCad ``Datasheet`` property), with its origin recorded.
4. ``datasheet_archived`` - an archived, hash-verified copy: the IR's own
   archived copy re-hashed (``tampered`` -> NOT_VERIFIED), else a copy an
   earlier run fetched from the same URL, else one fetch through the
   :class:`~ai_eda.tools.sources.DocumentArchive` when the user opened an
   online session (the policy decides trust; a KiCad Datasheet host is
   trusted by rule (a) here). ``blocked`` / ``missing`` / ``refused`` /
   ``error`` outcomes and "not fetched (offline)" are NOT_VERIFIED with the
   reason - never FAIL, never a guess. **A datasheet is a PDF.** A pointer
   that resolves to an HTML / XML / text document (a manufacturer's product
   page, a viewer shell, a note somebody wrote) is archived as evidence of
   what the pointer led to, but the sub-check is NOT_VERIFIED
   ("archived html document is not a datasheet") and neither the MPN nor any
   fact is grounded on it - a product page repeats the part number for
   every product and proves nothing about the part. The document kind is in
   the message and the details.
5. ``mpn_in_datasheet`` - the IR's MPN found verbatim in the archived text
   (ASCII case ignored, whitespace ignored, token boundary; control
   characters mapped to spaces first). Not found -> NOT_VERIFIED "MPN not
   found in datasheet" - never FAIL and never a shorter number that *is*
   found. Text not extractable (an AES-encrypted PDF) -> NOT_VERIFIED
   "datasheet text not extractable".
6. ``catalog`` - the MPN row in a loaded :class:`~ai_eda.parts.catalog.CatalogSource`
   (NOT_APPLICABLE when no catalog is loaded, so a design is not marked
   unverified for lacking a distributor file; NOT_VERIFIED "not in catalog").
   When the IR names a manufacturer or a package, the row's must agree
   (ASCII case and whitespace ignored); a row for the same MPN from another
   brand or in another package is NOT_VERIFIED with both values named and
   backs no sourcing.

Evidence: the archived datasheet (path, final URL, sha256) and the catalog
file (path, sha256). ``artifact_hash`` is the datasheet's sha256 - the
document the tool actually read.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ai_eda.errors import ApprovalRequiredError
from ai_eda.ir import Component, Evidence, LibraryRef, SourcingInfo, ValidationResult, ValidationStatus, worst_status
from ai_eda.parts.catalog import CatalogRow, CatalogSource
from ai_eda.parts.datasheet_facts import searchable_document
from ai_eda.parts.pointers import DatasheetPointer, PointerSearch, locate_datasheet
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError
from ai_eda.tools.sources import ArchiveError, ArchivedDocument, DocumentArchive, FetchOutcome, QuoteHit

TOOL = "parts.existence"
TOOL_VERSION = "0.2"
CHECK_PREFIX = "component.existence."

S = ValidationStatus


class SubCheck(BaseModel):
    name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ExistenceReport(BaseModel):
    """Everything :func:`examine_component` found; :meth:`result` is the ValidationResult, the other fields feed the agent's proposals."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: str
    mpn: str | None = None
    checks: list[SubCheck] = Field(default_factory=list)
    pointer: DatasheetPointer | None = None
    pointer_search: PointerSearch | None = None
    document: ArchivedDocument | None = None
    fetch: FetchOutcome | None = None
    mpn_hit: QuoteHit | None = None
    catalog_row: CatalogRow | None = None
    sourcing: SourcingInfo | None = None

    @property
    def status(self) -> ValidationStatus:
        return worst_status(c.status for c in self.checks)

    def check(self, name: str) -> SubCheck | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def result(self, catalog: CatalogSource | None = None) -> ValidationResult:
        evidence: list[Evidence] = []
        if self.document is not None:
            evidence.append(Evidence(
                description=f"archived datasheet of {self.ref}", path=str(self.document.path),
                url=self.document.final_url or self.document.url, content_hash=self.document.sha256,
            ))
        if self.catalog_row is not None and catalog is not None:
            evidence.append(Evidence(description=f"{catalog.supplier} catalog export (row {self.catalog_row.line})", path=str(catalog.path), content_hash=catalog.sha256))
        summary = ", ".join(f"{c.name} {c.status}" for c in self.checks)
        problems = [f"{c.name}: {c.message}" for c in self.checks if c.status not in (S.PASS, S.NOT_APPLICABLE)]
        message = summary + ("; " + "; ".join(problems) if problems else "")
        details: dict[str, Any] = {
            "ref": self.ref,
            "mpn": self.mpn,
            "checks": [c.model_dump(mode="json") for c in self.checks],
            "datasheet": {
                "pointer": self.pointer.model_dump(mode="json", exclude={"ref"}) if self.pointer is not None else None,
                "pointer_notes": list(self.pointer_search.notes) if self.pointer_search is not None else [],
                "archived": None if self.document is None else {
                    "sha256": self.document.sha256, "path": str(self.document.path), "url": self.document.url, "final_url": self.document.final_url,
                    "retrieved_at": self.document.meta.get("retrieved_at"), "pages": self.document.page_count, "text_available": self.document.text_available,
                    "extractor": self.document.meta.get("extractor"), "extractor_version": self.document.meta.get("extractor_version"),
                },
                "fetch": self.fetch.log_record() if self.fetch is not None else None,
                "mpn_hit": self.mpn_hit.model_dump(mode="json") if self.mpn_hit is not None else None,
            },
            "catalog": None if catalog is None else {
                "path": str(catalog.path), "sha256": catalog.sha256, "retrieved_at": catalog.retrieved_at, "supplier": catalog.supplier,
                "row": self.catalog_row.model_dump(mode="json", exclude={"raw"}) if self.catalog_row is not None else None,
            },
        }
        return ValidationResult(
            check_id=f"{CHECK_PREFIX}{self.ref}", status=self.status, message=message, tool=TOOL, tool_version=TOOL_VERSION,
            artifact_hash=self.document.sha256 if self.document is not None else None, evidence=evidence, details=details,
        )


# --------------------------------------------------------------------------- sub-checks


def _resolve(ref: LibraryRef | None, what: str, library: KicadLibrary | None) -> SubCheck:
    if ref is None:
        return SubCheck(name=what, status=S.NOT_VERIFIED, message=f"no {what} reference in the IR")
    lib_id = f"{ref.library}:{ref.name}"
    if library is None:
        return SubCheck(name=what, status=S.NOT_VERIFIED, message=f"no KiCad library available to resolve {what} {lib_id}", details={"lib_id": lib_id})
    try:
        resolved = library.resolve_symbol(ref) if what == "symbol" else library.resolve_footprint(ref)
    except LibraryFormatError as e:
        return SubCheck(name=what, status=S.FAIL, message=f"{what} {lib_id} exists but is unreadable: {e}", details={"lib_id": lib_id})
    if not resolved.verified:
        return SubCheck(name=what, status=S.FAIL, message=f"{what} {lib_id} does not exist in the KiCad libraries (the design references a library entry that is not there)",
                        details={"lib_id": lib_id, "ir_verified_flag": ref.verified})
    return SubCheck(name=what, status=S.PASS, message=f"{what} {lib_id} found in {resolved.library_path}",
                    details={"lib_id": lib_id, "library_path": resolved.library_path, "ir_verified_flag": ref.verified})


#: the one document kind that counts as a datasheet (module docstring, sub-check 4)
DATASHEET_KINDS: frozenset[str] = frozenset({"pdf"})


def _as_datasheet(doc: ArchivedDocument, outcome: FetchOutcome | None, message: str, details: dict[str, Any]) -> tuple[ArchivedDocument | None, FetchOutcome | None, SubCheck]:
    """PASS with ``doc`` when it is a datasheet by kind, else NOT_VERIFIED naming the kind and no document (nothing is grounded on a web page)."""
    name = "datasheet_archived"
    details = {**details, "kind": doc.kind, "title": doc.title}
    if doc.kind in DATASHEET_KINDS:
        return doc, outcome, SubCheck(name=name, status=S.PASS, message=f"{message} [{doc.kind}]", details=details)
    return None, outcome, SubCheck(
        name=name, status=S.NOT_VERIFIED,
        message=(f"archived {doc.kind} document is not a datasheet ({doc.title or doc.path.name}): a product page / viewer shell / text file proves "
                 f"nothing about the part, so the MPN and no fact are grounded on it; {message}"),
        details=details,
    )


def _archived_document(pointer: DatasheetPointer, archive: DocumentArchive, *, fetch: bool, purpose: str) -> tuple[ArchivedDocument | None, FetchOutcome | None, SubCheck]:
    """The hash-verified archived datasheet behind ``pointer`` (module docstring, sub-check 4)."""
    name = "datasheet_archived"
    ref = pointer.ref
    prefix = ""
    if ref.content_hash:
        verdict = archive.verify(ref)
        if verdict == "ok":
            try:
                doc = archive.load(ref.content_hash)
            except ArchiveError as e:  # pragma: no cover - verify() just loaded it
                return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"archived copy unreadable: {e}")
            return _as_datasheet(doc, None, f"archived copy re-hashed and verified: {doc.path} ({doc.sha256})", {"source": "ir_reference", "sha256": doc.sha256})
        if verdict == "tampered":
            return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"tampered: the archived datasheet {ref.document_path or ref.content_hash} no longer hashes to {ref.content_hash}",
                                        details={"verify": verdict})
        prefix = f"archived copy {ref.content_hash} not found in the archive ({verdict}); "
    if pointer.url is None:
        why = f"URL unusable: {pointer.url_error}" if pointer.url_error else "no URL to fetch"
        return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=prefix + why, details={"origin": pointer.origin})
    earlier = archive.lookup(pointer.url)
    if earlier is not None:
        return _as_datasheet(earlier, None, f"{prefix}archived copy from an earlier fetch of {pointer.url}: {earlier.path} ({earlier.sha256})",
                             {"source": "earlier_fetch", "sha256": earlier.sha256, "retrieved_at": earlier.meta.get("retrieved_at")})
    if not fetch:
        return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=prefix + "not fetched (fetching disabled for this check)", details={"url": pointer.url})
    if not archive.online:
        return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=prefix + "not fetched (offline: no online session was opened)", details={"url": pointer.url})
    if pointer.origin in ("kicad_symbol", "ir") and pointer.host and pointer.trust_reason:
        archive.policy.trust_host(pointer.host, pointer.trust_reason)  # rule (a): the KiCad library's own datasheet host (a user URL stays exact: rule (c))
    expect = "pdf" if urlsplit(pointer.url).path.lower().endswith(".pdf") else "any"
    try:
        out = archive.fetch(pointer.url, purpose=purpose, expect=expect)
    except ApprovalRequiredError as e:
        return None, None, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"{prefix}not fetched: {e}", details={"url": pointer.url})
    if out.ok and out.document is not None:
        doc = out.document
        return _as_datasheet(doc, out, f"{prefix}fetched and archived: {out.final_url} -> {doc.path} ({doc.sha256})",
                             {"source": "fetch", "sha256": doc.sha256, "final_url": out.final_url, "redirects": len(out.redirects), "trusted_by": out.trusted_by})
    return None, out, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"{prefix}not archived ({out.status}): {out.reason}",
                               details={"url": pointer.url, "fetch_status": out.status, "http_status": out.http_status, "final_url": out.final_url})


def find_mpn(doc: ArchivedDocument, mpn: str) -> list[QuoteHit]:
    """Every page's first hit of ``mpn`` in the archived text (control characters as spaces, ASCII case ignored, whole identifier).

    Whole identifier: ``LM2596S-5`` is not found in ``LM2596S-5.0/NOPB`` - a
    part number cut before its ``.x`` fraction or ``-suffix`` names another
    (or no) orderable code and must not be grounded on the longer one.
    """
    return searchable_document(doc).find_quote(mpn, ignore_case=True, identifier=True)


def _mpn_check(mpn: str | None, doc: ArchivedDocument | None) -> tuple[QuoteHit | None, SubCheck]:
    name = "mpn_in_datasheet"
    if not mpn:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED, message="no MPN in the IR: nothing to look for")
    if doc is None:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED, message="datasheet not archived: MPN not checked")
    if not doc.text_available:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"datasheet text not extractable ({doc.extraction_error or 'no text'}): MPN not checked",
                              details={"extraction_error": doc.extraction_error})
    hits = find_mpn(doc, mpn)
    if not hits:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED,
                              message=f"MPN {mpn!r} not found in datasheet ({doc.page_count} page(s) searched; whitespace-insensitive, ASCII case-insensitive, whole identifier)",
                              details={"pages_searched": doc.page_count})
    first = hits[0]
    return first, SubCheck(name=name, status=S.PASS, message=f"MPN {mpn!r} found verbatim on page {first.page}: {first.context}",
                           details={"page": first.page, "offset": first.offset, "matched": first.matched, "context": first.context, "pages": [h.page for h in hits]})


def _same_text(a: str, b: str) -> bool:
    return "".join(a.split()).casefold() == "".join(b.split()).casefold()


def _catalog_check(mpn: str | None, catalog: CatalogSource | None, component: Component | None = None) -> tuple[CatalogRow | None, SubCheck]:
    name = "catalog"
    if catalog is None:
        return None, SubCheck(name=name, status=S.NOT_APPLICABLE, message="no catalog loaded")
    if not mpn:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED, message="no MPN in the IR: nothing to look up")
    rows = catalog.rows_for(mpn)
    if not rows:
        return None, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"not in catalog: MPN {mpn!r} has no row in {catalog.path.name} ({len(catalog.rows)} rows, {catalog.supplier})")

    def _mismatches(row: CatalogRow) -> list[str]:
        out: list[str] = []
        if component is not None:
            for label, ir_value, row_value in (("manufacturer", component.manufacturer, row.manufacturer), ("package", component.package, row.package)):
                if ir_value is not None and ir_value.value not in (None, "") and row_value.strip() and not _same_text(str(ir_value.value), row_value):
                    out.append(f"catalog {label} {row_value!r} differs from IR {label} {ir_value.value!r} ({ir_value.provenance.kind})")
        return out

    # a community dump lists one MPN under several brands / packages: the row that agrees with the IR backs the
    # sourcing, whatever its position; none agreeing is said with every row's disagreement
    judged = [(row, _mismatches(row)) for row in rows]
    row, mismatches = next(((r, m) for r, m in judged if not m), judged[0])
    details = {"line": row.line, "supplier_part_number": row.supplier_part_number, "stock": row.stock, "unit_price": row.unit_price, "currency": row.currency,
               "assembly_class": row.assembly_class, "manufacturer": row.manufacturer, "package": row.package, "rows": [r.line for r in rows]}
    if mismatches:
        others = "; ".join(f"row {r.line}: " + ", ".join(m) for r, m in judged)
        return row, SubCheck(name=name, status=S.NOT_VERIFIED, message=f"MPN {mpn!r} found in {catalog.supplier} catalog ({len(rows)} row(s)) but none agrees with the IR: {others}"
                                                                        + " - a same-numbered part of another brand / package backs no sourcing", details={**details, "mismatches": mismatches})
    return row, SubCheck(name=name, status=S.PASS, message=f"MPN {mpn!r} found in {catalog.supplier} catalog row {row.line} (supplier part {row.supplier_part_number or '-'}, stock {row.stock if row.stock is not None else '-'})",
                         details=details)


# --------------------------------------------------------------------------- the check


def examine_component(
    component: Component,
    library: KicadLibrary | None = None,
    archive: DocumentArchive | None = None,
    catalog: CatalogSource | None = None,
    *,
    fetch: bool = True,
    user_urls: dict[str, str] | None = None,
    purpose: str | None = None,
) -> ExistenceReport:
    """Run the sub-checks of the module docstring and return everything found (the IR is not touched).

    ``fetch=False`` forbids a network attempt even in an online session;
    ``user_urls`` are the user's ``--datasheet-url`` pointers (the policy must
    also carry them for the fetch to be allowed).
    """
    mpn = str(component.mpn.value) if component.mpn is not None and component.mpn.value is not None else None
    report = ExistenceReport(ref=component.ref, mpn=mpn)
    report.checks.append(_resolve(component.symbol, "symbol", library))
    report.checks.append(_resolve(component.footprint, "footprint", library))
    search = locate_datasheet(component, library, user_urls)
    report.pointer_search = search
    report.pointer = search.pointer
    if search.pointer is None:
        report.checks.append(SubCheck(name="datasheet_pointer", status=S.NOT_VERIFIED, message=f"no datasheet pointer: {search.reason}", details={"notes": search.notes}))
        report.checks.append(SubCheck(name="datasheet_archived", status=S.NOT_VERIFIED, message="no datasheet pointer: nothing to archive"))
    else:
        p = search.pointer
        report.checks.append(SubCheck(name="datasheet_pointer", status=S.PASS, message=p.note,
                                      details={"origin": p.origin, "url": p.url, "url_note": p.url_note, "url_error": p.url_error, "host": p.host, "trust_reason": p.trust_reason, "notes": search.notes}))
        if archive is None:
            report.checks.append(SubCheck(name="datasheet_archived", status=S.NOT_VERIFIED, message="no document archive available: nothing fetched, nothing verified", details={"url": p.url}))
        else:
            doc, outcome, check = _archived_document(p, archive, fetch=fetch, purpose=purpose or f"datasheet of {component.ref}")
            report.document, report.fetch = doc, outcome
            report.checks.append(check)
    hit, check = _mpn_check(mpn, report.document)
    report.mpn_hit = hit
    report.checks.append(check)
    row, check = _catalog_check(mpn, catalog, component)
    report.catalog_row = row
    report.checks.append(check)
    if row is not None and catalog is not None and check.status is S.PASS:
        report.sourcing = catalog.sourcing_info(row)
    return report


def check_component_existence(
    component: Component,
    library: KicadLibrary | None = None,
    archive: DocumentArchive | None = None,
    catalog: CatalogSource | None = None,
    *,
    fetch: bool = True,
    user_urls: dict[str, str] | None = None,
) -> ValidationResult:
    """The ``component.existence.<ref>`` result (see the module docstring); ``details.checks`` lists every sub-check."""
    return examine_component(component, library, archive, catalog, fetch=fetch, user_urls=user_urls).result(catalog)


__all__ = [
    "CHECK_PREFIX",
    "DATASHEET_KINDS",
    "TOOL",
    "TOOL_VERSION",
    "ExistenceReport",
    "SubCheck",
    "check_component_existence",
    "examine_component",
    "find_mpn",
]
