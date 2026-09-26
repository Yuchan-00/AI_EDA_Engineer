"""The user's fab capability file and its grounding on the archived vendor page.

Invariants (the trust model of :mod:`ai_eda.parts.datasheet_facts`, moved to
the fab):

* The file (``--fab-capability FILE``) is the *user's* claim: which vendor
  page states which limit, on which page, in which words. Nothing in it is
  design truth until :func:`ground_capability` finds every quote verbatim on
  the claimed page of the *archived* copy (hash-named, extractor-stamped) and
  re-reads the number from the document's own text. Unknown keys are refused
  by the strict schema, never mapped.
* A millimetre limit (:data:`MM_KEYS`) must quote one whole length of the
  page (the request-side rule :func:`~ai_eda.llm.extraction._request_quantity`,
  reused: ``0.127mm`` cut out of ``0.127mm(5mil)`` or ``0.15mm/0.25mm`` is
  not one quantity), carry a length unit the parser knows (``mil`` is not one),
  and agree with the user's ``value``/``unit``. The value stored is the
  document's own decimal token converted exactly (:func:`mm_from_token`,
  ``Decimal``): ``0.09 mm`` is stored as ``0.09``, never as the float drift
  ``9e-05 * 1000``, so ``design_rules`` writes the page's number and the IR
  comparisons in :mod:`ai_eda.tools.manufacturing.capability` judge equality
  exactly.
* ``layer_count_options`` is grounded only when the integer tokens of the
  quote, in order, equal the option list; ``copper_weight_oz`` only when the
  quote holds exactly one ``<n> oz`` token and no other number. The quantity
  parser does not know ``oz`` or layer counts and is not extended for them.
* An accepted limit becomes an ``authoritative`` :class:`~ai_eda.ir.Traced`
  whose :class:`~ai_eda.ir.SourceRef` names the archived page by hash and
  ``page N``, and whose note keeps the quote as it stands in the document
  (``quote: <repr>``; :func:`quote_from_note` reads it back, for the
  reviewer's and the agent's re-verification), the parse and the extractor
  stamp. A rejected limit never enters the IR; the reason is reported.
* What grounding can not check is the *key* the user attached to a quote
  (``min_clearance_mm`` with the quote of the via drill would ground): the
  key attribution is the user's assertion, shown with quote and page in
  ``mfg.capability_source`` details so a person can see it.
* Re-verification (:func:`relocate_limits`, the agent without a file and the
  reviewer) re-runs the *same* grounding step on the re-located quote and
  compares the number and unit it reads with the ``Traced`` the IR stores:
  a limit whose value was edited after grounding (note, page hash and quote
  intact) is ``value_mismatch``, never ``ok``. :func:`~ai_eda.tools.manufacturing.capability.check_capability`
  reads the IR's provenance as it stands; the agent's and the reviewer's
  verdicts are the gate that catches an edited number.
* Numbers are finite: a page token that overflows (``1e400 mm``) or a
  non-finite file value is rejected with the reason, and a capability file
  that does not parse (including one nested past the JSON parser's depth)
  is a :class:`CapabilityFileError`, never a traceback.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ai_eda.errors import AiEdaError
from ai_eda.ir import Evidence, ManufacturingConstraints, ProvenanceKind, SourceRef, Traced, ValidationResult, ValidationStatus, authoritative
from ai_eda.llm.extraction import _mismatch, _model_quantity, _request_quantity, find_directive
from ai_eda.parts.datasheet_facts import searchable_document
from ai_eda.tools.calc.quantity import PREFIX_EXPONENTS, Quantity, format_quantity
from ai_eda.tools.sources import ArchivedDocument, DocumentArchive, QuoteHit

#: bumped when the grounding rules or the file schema change
CAPABILITY_FILE_VERSION = "0.1"
GROUNDING_TOOL = "mfg.capability_grounding"
SOURCE_CHECK_ID = "mfg.capability_source"

#: the eight ``ManufacturingConstraints`` fields a file may set (``fab`` is the file's own field)
CAPABILITY_KEYS: frozenset[str] = frozenset(k for k in ManufacturingConstraints.model_fields if k != "fab")
#: limits stated in millimetres (grounded through the quantity parser, stored in mm)
MM_KEYS: frozenset[str] = frozenset(k for k in CAPABILITY_KEYS if k.endswith("_mm"))
LAYER_KEY = "layer_count_options"
OZ_KEY = "copper_weight_oz"

QUOTE_NOTE_PREFIX = "quote: "
#: a Python string literal as ``repr`` writes it (the quote in a provenance note)
_STRING_LITERAL_RE = re.compile("'(?:[^'\\\\]|\\\\.)*'|\"(?:[^\"\\\\]|\\\\.)*\"")
_LENGTH_TOKEN_RE = re.compile(
    r"^\s*(?P<num>[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?|\.\d+)\s*"
    r"(?P<prefix>[" + "".join(re.escape(p) for p in PREFIX_EXPONENTS) + r"]?)\s*"
    r"(?P<unit>m|meter|meters|metre|metres|미터)\s*$"
)
_OZ_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*oz(?![A-Za-z0-9])")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_PAGE_SECTION_RE = re.compile(r"^page (\d+)$")


class CapabilityFileError(AiEdaError):
    """The capability file does not validate; the message names the entry."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapabilityLimit(_Strict):
    """One limit the user claims the vendor page states."""

    key: str = Field(description="a ManufacturingConstraints field: min_track_width_mm, min_clearance_mm, min_via_drill_mm, min_via_diameter_mm, "
                                 "min_hole_to_edge_mm, board_thickness_mm, layer_count_options, copper_weight_oz")
    value: float | list[int] = Field(description="the number as written in the quote (mm keys: any length unit, stored in mm), or the layer options")
    unit: str | None = Field(default=None, description="the unit as written in the quote (mm, um, ...; oz); null for layer_count_options")
    page: int = Field(description="1-based page of the archived document the quote stands on (1 for an HTML page)")
    quote: str = Field(description="verbatim phrase from that page containing exactly this value and its unit")


class CapabilitySource(_Strict):
    """Where the page comes from: exactly one of ``url`` (fetched through the archive) or ``file`` (a saved page, with the user's ``retrieved_at``)."""

    url: str | None = None
    file: str | None = None
    retrieved_at: str | None = None
    title: str | None = None
    authority: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> CapabilitySource:
        if bool(self.url) == bool(self.file):
            raise ValueError("source needs exactly one of 'url' or 'file'")
        if self.file and not self.retrieved_at:
            raise ValueError("source.file needs source.retrieved_at (ISO 8601): when you saved the page - the system never invents a date")
        return self


class FabCapabilityFile(_Strict):
    fab: str
    source: CapabilitySource
    limits: list[CapabilityLimit]
    #: where the file was read from and what it hashed to (bookkeeping, not part of the schema the user writes)
    path: Path
    sha256: str

    def describe(self) -> dict[str, Any]:
        return {"fab": self.fab, "path": str(self.path), "sha256": self.sha256, "source": self.source.model_dump(mode="json"), "limits": [lim.key for lim in self.limits]}


def load_capability_file(path: Path | str) -> FabCapabilityFile:
    """The user's JSON capability file; :class:`CapabilityFileError` names the entry that does not validate.

    Layout: ``{"fab": "...", "source": {"url": ...} | {"file": ..., "retrieved_at": ...},
    "limits": [{"key", "value", "unit", "page", "quote"}, ...]}``. Keys the
    schema does not declare are refused (a misspelled key would otherwise be
    a silently dropped limit).
    """
    p = Path(path)
    try:
        raw_bytes = p.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
    except OSError as e:
        raise CapabilityFileError(f"capability file {p} unreadable: {e}") from e
    except (ValueError, UnicodeDecodeError, RecursionError) as e:
        raise CapabilityFileError(f"capability file {p} is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise CapabilityFileError(f"capability file {p}: expected an object with fab / source / limits")
    if "path" in data or "sha256" in data:
        raise CapabilityFileError(f"capability file {p}: 'path' and 'sha256' are recorded by the loader, not written in the file")
    try:
        loaded = FabCapabilityFile.model_validate({**data, "path": p, "sha256": "sha256:" + hashlib.sha256(raw_bytes).hexdigest()})
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ())) or "file"
        raise CapabilityFileError(f"capability file {p}: {where}: {first.get('msg', e)}") from e
    if not loaded.fab.strip():
        raise CapabilityFileError(f"capability file {p}: fab: must name the fab")
    if not loaded.limits:
        raise CapabilityFileError(f"capability file {p}: limits: at least one limit is needed")
    for i, lim in enumerate(loaded.limits):
        # the value's shape follows the key; an unknown key is left to grounding, which rejects it with the reason
        if lim.key in MM_KEYS or lim.key == OZ_KEY:
            if not _is_number(lim.value) or not math.isfinite(lim.value):
                raise CapabilityFileError(f"capability file {p}: limits[{i}] ({lim.key}): value must be a finite number, got {lim.value!r}")
        elif lim.key == LAYER_KEY and not isinstance(lim.value, list):
            raise CapabilityFileError(f"capability file {p}: limits[{i}] ({lim.key}): value must be a list of layer counts, got {lim.value!r}")
    return loaded


# --------------------------------------------------------------------------- tokens


def mm_from_token(original: str) -> float:
    """The exact millimetre value of a length token as the document writes it (``0.09 mm`` -> 0.09, ``90um`` -> 0.09).

    ``Decimal`` arithmetic on the document's own digits: the canonical
    ``value * 1000`` of the quantity parser would give ``0.09000000000000001``.
    ``ValueError`` when the token is not one length.
    """
    m = _LENGTH_TOKEN_RE.match(original)
    if m is None:
        raise ValueError(f"not a length token: {original!r}")
    num = m.group("num").replace(",", "").replace("−", "-")
    exponent = PREFIX_EXPONENTS.get(m.group("prefix") or "", 0) + 3  # metres -> millimetres
    try:
        value = Decimal(num).scaleb(exponent)
    except InvalidOperation as e:
        raise ValueError(f"not a number: {num!r}") from e
    return float(value)


def quote_from_note(note: str | None) -> str | None:
    """The quote a grounding note starts with (``quote: '<repr>'; ...``), else ``None``.

    Shared by fact and capability grounding and by their re-verification
    (agent and reviewer): one format, one reader, pinned by a test.
    """
    if not note or not note.startswith(QUOTE_NOTE_PREFIX):
        return None
    m = _STRING_LITERAL_RE.match(note, len(QUOTE_NOTE_PREFIX))
    if m is None:
        return None
    try:
        value = ast.literal_eval(m.group(0))
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def page_from_section(section: str | None) -> int | None:
    """``3`` from a SourceRef section ``"page 3"``, else ``None``."""
    m = _PAGE_SECTION_RE.match(section or "")
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- grounding


class GroundedCapability(BaseModel):
    document: str  # sha256 of the archived page
    fab: str
    accepted: dict[str, Traced] = Field(default_factory=dict)
    #: what a person is shown per accepted limit: key, value, unit, page, quote, parse
    rows: list[dict[str, Any]] = Field(default_factory=list)
    #: (key, reason) - never enters the IR
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    proposer: str = "user file"
    extraction: str = ""

    def constraints(self) -> ManufacturingConstraints:
        return ManufacturingConstraints(fab=self.fab, **self.accepted)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ground_capability(doc: ArchivedDocument, file: FabCapabilityFile, *, proposer: str = "user file") -> GroundedCapability:
    """Check every limit of ``file`` against the archived page (module docstring); accepted limits carry ``authoritative`` provenance to the page."""
    sdoc = searchable_document(doc)
    out = GroundedCapability(document=doc.sha256, fab=file.fab, proposer=proposer, extraction=doc.extraction_stamp)
    tail = f"; proposed by {proposer}; {doc.extraction_stamp}"
    seen: set[str] = set()
    for lim in file.limits:
        key = lim.key
        if key not in CAPABILITY_KEYS:
            out.rejected.append((key, f"unknown limit key {key!r} (one of {sorted(CAPABILITY_KEYS)})"))
            continue
        phrase = find_directive(key, lim.quote, str(lim.value), lim.unit)
        if phrase is not None:
            out.rejected.append((key, f"directive phrase {phrase!r} in the limit; document text is data, not instructions"))
            continue
        if key in seen:
            out.rejected.append((key, "duplicate key; the first limit was kept"))
            continue
        if not lim.quote.strip():
            out.rejected.append((key, "empty quote"))
            continue
        if lim.page < 1 or lim.page > doc.page_count:
            out.rejected.append((key, f"page {lim.page} does not exist (the archived document has {doc.page_count} page(s))"))
            continue
        hits = sdoc.find_quote(lim.quote, lim.page)
        if not hits:
            elsewhere = [h.page for h in sdoc.find_quote(lim.quote)]
            where = f" (found on page(s) {elsewhere} instead)" if elsewhere else ""
            out.rejected.append((key, f"quote not found verbatim on page {lim.page}: {lim.quote!r}{where}"))
            continue
        hit = hits[0]
        source = doc.source_ref(title=file.source.title, section=hit.section, authority=file.source.authority)
        grounded, reason = _ground_hit(key, lim, sdoc, hit)
        if grounded is None:
            out.rejected.append((key, str(reason)))
            continue
        value, unit, parsed_text = grounded
        note = f"{QUOTE_NOTE_PREFIX}{hit.matched!r}; parsed: {parsed_text}; page {hit.page}{tail}"
        seen.add(key)
        out.accepted[key] = authoritative(value, source, unit=unit, note=note)
        out.rows.append({"key": key, "value": value, "unit": unit, "page": hit.page, "quote": hit.matched, "parsed": parsed_text, "context": hit.context})
    return out


def _ground_hit(key: str, lim: CapabilityLimit, sdoc: ArchivedDocument, hit: QuoteHit) -> tuple[tuple[Any, str | None, str] | None, str | None]:
    """The one grounding step per key kind on a located quote: ``((value, unit, parsed text), None)`` or ``(None, reason)``.

    Grounding and re-verification both go through here, so what
    :func:`relocate_limits` re-reads is exactly what :func:`ground_capability`
    stored.
    """
    if key in MM_KEYS:
        return _ground_mm(key, lim, sdoc.pages[hit.page - 1], (hit.offset, hit.offset + len(hit.matched)))
    if key == LAYER_KEY:
        return _ground_layers(lim, hit.matched)
    return _ground_oz(lim, hit.matched)


def _ground_mm(key: str, lim: CapabilityLimit, page_text: str, span: tuple[int, int]) -> tuple[tuple[Any, str | None, str] | None, str | None]:
    if not _is_number(lim.value):
        return None, f"{key} is a length; got {lim.value!r}"
    if not math.isfinite(lim.value):
        return None, f"{key} must be a finite length; got {lim.value!r}"
    if not lim.unit or not lim.unit.strip():
        return None, f"{key} needs the unit as written in the quote"
    model_q = _model_quantity(float(lim.value), lim.unit, None)
    if model_q is None:
        return None, f"unit not recognised: {lim.unit!r} (value {lim.value!r}); lengths must be written in a metre unit (mm, um, ...) - mil is not one"
    if model_q.unit != "m":
        return None, f"unit mismatch: key {key!r} expects a length, got {lim.unit!r} ({format_quantity(model_q)})"
    parsed, reason = _request_quantity(page_text, span)
    if parsed is None:
        return None, str(reason).replace("in the request", "on the page")
    if not isinstance(parsed, Quantity) or parsed.plus_minus:
        return None, f"quote states a range or tolerance ({format_quantity(parsed)}), not one limit"
    if parsed.unit != "m":
        return None, f"the page states {format_quantity(parsed)} there, which is not a length"
    if not math.isfinite(parsed.value):
        return None, f"the page states a non-finite length ({parsed.original!r})"
    reason = _mismatch(model_q, parsed)
    if reason is not None:
        return None, reason.replace("model", "file")
    try:
        mm = mm_from_token(parsed.original)
    except ValueError as e:
        return None, str(e)
    if not math.isfinite(mm):
        return None, f"the page states a non-finite length ({parsed.original!r})"
    return (mm, "mm", f"{mm:g} mm"), None


def _ground_layers(lim: CapabilityLimit, matched: str) -> tuple[tuple[Any, str | None, str] | None, str | None]:
    value = lim.value
    if not isinstance(value, list) or not value or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value):
        return None, f"{LAYER_KEY} must be a non-empty list of positive layer counts; got {value!r}"
    if lim.unit:
        return None, f"{LAYER_KEY} has no unit; got {lim.unit!r}"
    tokens = [int(t) for t in re.findall(r"\d+", matched)]
    if tokens != list(value):
        return None, f"the quote's integer tokens {tokens} do not equal the options {list(value)}"
    return (list(value), None, ", ".join(str(v) for v in value) + " layers"), None


def _ground_oz(lim: CapabilityLimit, matched: str) -> tuple[tuple[Any, str | None, str] | None, str | None]:
    if not _is_number(lim.value):
        return None, f"{OZ_KEY} is a number; got {lim.value!r}"
    if (lim.unit or "").strip() != "oz":
        return None, f"{OZ_KEY} needs unit 'oz'; got {lim.unit!r}"
    oz = _OZ_TOKEN_RE.findall(matched)
    if len(oz) != 1:
        return None, f"the quote must hold exactly one '<n> oz' token; found {len(oz)} in {matched!r}"
    if _NUMBER_RE.findall(matched) != oz:
        return None, f"the quote holds other numbers besides {oz[0]} oz: {matched!r}"
    token = float(Decimal(oz[0]))
    if abs(token - float(lim.value)) > 1e-9 * max(1.0, abs(token)):
        return None, f"number mismatch: file {lim.value!r} oz vs quote {oz[0]} oz"
    return (token, "oz", f"{token:g} oz"), None


# --------------------------------------------------------------------------- results


def capability_source_result(
    file: FabCapabilityFile,
    doc: ArchivedDocument | None,
    grounded: GroundedCapability | None,
    *,
    status: ValidationStatus | None = None,
    reason: str | None = None,
) -> ValidationResult:
    """``mfg.capability_source``: PASS when the page was obtained and every limit of the file grounded; NOT_VERIFIED otherwise, never FAIL.

    A wrong quote is the user's file, not the design. ``status`` overrides the
    rule (the agent passes NOT_VERIFIED when there is no board to record the
    limits in); ``reason`` is appended to the message (or is the message when
    no document was obtained).
    """
    details: dict[str, Any] = {"fab": file.fab, "file": str(file.path), "file_sha256": file.sha256, "source": file.source.model_dump(mode="json"),
                               "limits_in_file": [lim.key for lim in file.limits]}
    if doc is None or grounded is None:
        details["reason"] = reason
        return ValidationResult(check_id=SOURCE_CHECK_ID, status=status or ValidationStatus.NOT_VERIFIED, message=reason or "vendor page not obtained",
                                tool=GROUNDING_TOOL, tool_version=CAPABILITY_FILE_VERSION, details=details)
    n_ok, n_bad = len(grounded.accepted), len(grounded.rejected)
    if status is None:
        status = ValidationStatus.PASS if n_ok and not n_bad and n_ok == len(file.limits) else ValidationStatus.NOT_VERIFIED
    message = f"{n_ok} limit(s) grounded verbatim on the archived vendor page, {n_bad} rejected (proposed by {grounded.proposer})"
    if grounded.rejected:
        message += ": " + "; ".join(f"{k}: {why}" for k, why in grounded.rejected[:3]) + (" ..." if n_bad > 3 else "")
    if reason:
        message += f"; {reason}"
    details.update({
        "document": doc.sha256, "url": doc.final_url or doc.url, "proposer": grounded.proposer,
        "accepted": list(grounded.rows), "rejected": [{"key": k, "reason": why} for k, why in grounded.rejected],
        "extraction": grounded.extraction, "text_matches_meta": doc.text_matches_meta, "reason": reason,
    })
    return ValidationResult(
        check_id=SOURCE_CHECK_ID, status=status, message=message, tool=GROUNDING_TOOL, tool_version=CAPABILITY_FILE_VERSION,
        artifact_hash=doc.sha256,
        evidence=[Evidence(description=f"archived capability page of {file.fab}", path=str(doc.path), url=doc.final_url or doc.url, content_hash=doc.sha256)],
        details=details,
    )


# --------------------------------------------------------------------------- re-verification (agent without a file, reviewer)


class LimitSourceCheck(BaseModel):
    key: str
    status: str  # ok | unarchived | missing | tampered | quote_missing | value_mismatch
    reason: str
    document: str | None = None
    page: int | None = None
    quote: str | None = None


def relocate_limits(constraints: ManufacturingConstraints, archive: DocumentArchive | None) -> list[LimitSourceCheck]:
    """Re-verify every ``authoritative`` limit: its archived page re-hashed, its quote re-located on the recorded page and its number re-read.

    ``ok`` only when the archive holds the page under the recorded hash, the
    quote from the note stands on ``page N`` of the re-extracted text **and**
    the grounding step run again on that quote (:func:`_ground_hit`) reads
    exactly the value and unit the IR stores; ``unarchived`` / ``missing`` /
    ``tampered`` as :func:`~ai_eda.parts.identity.locate_source` reports;
    ``quote_missing`` when the page is there but the quote is not;
    ``value_mismatch`` (both numbers in the reason) when the quote is there
    but does not state the stored value - an edited ``ir.json``.
    """
    from ai_eda.parts.identity import locate_source

    out: list[LimitSourceCheck] = []
    cache: dict[str, tuple[str, str, ArchivedDocument | None]] = {}
    for key in sorted(CAPABILITY_KEYS):
        traced = getattr(constraints, key)
        if traced is None or traced.provenance.kind is not ProvenanceKind.AUTHORITATIVE:
            continue
        ref: SourceRef | None = traced.provenance.source
        quote = quote_from_note(traced.provenance.note)
        page = page_from_section(ref.section if ref is not None else None)
        cache_key = f"{ref.content_hash}|{ref.document_path}|{ref.retrieved_at}" if ref is not None else ""
        if cache_key not in cache:
            cache[cache_key] = locate_source(ref, archive)
        status, reason, doc = cache[cache_key]
        if status != "ok" or doc is None:
            out.append(LimitSourceCheck(key=key, status=status, reason=reason, document=ref.content_hash if ref else None, page=page, quote=quote))
            continue
        if quote is None or page is None:
            out.append(LimitSourceCheck(key=key, status="quote_missing", reason="the limit's note records no quote / page to re-locate", document=doc.sha256, page=page, quote=quote))
            continue
        sdoc = searchable_document(doc)
        hits = sdoc.find_quote(quote, page)
        if not hits:
            out.append(LimitSourceCheck(key=key, status="quote_missing", reason=f"quote {quote!r} not found on page {page} of the archived page {doc.sha256[:19]}… at review time",
                                        document=doc.sha256, page=page, quote=quote))
            continue
        mismatch = _stored_value_mismatch(key, traced, sdoc, hits[0])
        if mismatch is not None:
            out.append(LimitSourceCheck(key=key, status="value_mismatch", reason=mismatch, document=doc.sha256, page=page, quote=quote))
            continue
        out.append(LimitSourceCheck(key=key, status="ok", reason=f"{reason}; quote re-read as {_stored_text(traced)}", document=doc.sha256, page=page, quote=quote))
    return out


def _stored_text(traced: Traced) -> str:
    return f"{traced.value!r}" + (f" {traced.unit}" if traced.unit else "")


def _stored_value_mismatch(key: str, traced: Traced, sdoc: ArchivedDocument, hit: QuoteHit) -> str | None:
    """The reason the stored ``Traced`` is not what the re-located quote states, else ``None`` (the same grounding step as :func:`ground_capability`)."""
    stored = _stored_text(traced)
    try:
        lim = CapabilityLimit(key=key, value=traced.value, unit=traced.unit, page=hit.page, quote=hit.matched)
    except ValidationError as e:
        return f"stored value {stored} has a shape the grounding rules do not accept: {e.errors()[0].get('msg', e) if e.errors() else e}"
    grounded, reason = _ground_hit(key, lim, sdoc, hit)
    if grounded is None:
        states = _quote_states(key, sdoc, hit)
        return f"stored value {stored} but the quote {hit.matched!r} states {states}: {reason}" if states else f"stored value {stored} does not re-ground on the quote {hit.matched!r}: {reason}"
    value, unit, parsed_text = grounded
    if value != traced.value or (unit or None) != (traced.unit or None):
        return f"stored value {stored} but the quote {hit.matched!r} re-reads as {parsed_text} ({value!r} {unit or ''})".rstrip()
    return None


def _quote_states(key: str, sdoc: ArchivedDocument, hit: QuoteHit) -> str | None:
    """What the located quote states, in the words grounding would have stored (``0.09 mm``, ``1, 2, 4, 6 layers``, ``1 oz``); ``None`` when it does not read as one number."""
    if key in MM_KEYS:
        parsed, _ = _request_quantity(sdoc.pages[hit.page - 1], (hit.offset, hit.offset + len(hit.matched)))
        if not isinstance(parsed, Quantity) or parsed.plus_minus or parsed.unit != "m" or not math.isfinite(parsed.value):
            return None
        try:
            return f"{mm_from_token(parsed.original):g} mm"
        except ValueError:
            return None
    if key == LAYER_KEY:
        tokens = re.findall(r"\d+", hit.matched)
        return ", ".join(tokens) + " layers" if tokens else None
    oz = _OZ_TOKEN_RE.findall(hit.matched)
    return f"{oz[0]} oz" if len(oz) == 1 else None


__all__ = [
    "CAPABILITY_FILE_VERSION",
    "CAPABILITY_KEYS",
    "GROUNDING_TOOL",
    "LAYER_KEY",
    "MM_KEYS",
    "OZ_KEY",
    "QUOTE_NOTE_PREFIX",
    "SOURCE_CHECK_ID",
    "CapabilityFileError",
    "CapabilityLimit",
    "CapabilitySource",
    "FabCapabilityFile",
    "GroundedCapability",
    "LimitSourceCheck",
    "capability_source_result",
    "ground_capability",
    "load_capability_file",
    "mm_from_token",
    "page_from_section",
    "quote_from_note",
    "relocate_limits",
]
