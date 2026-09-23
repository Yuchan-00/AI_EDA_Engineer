"""Distributor catalog: a user-supplied CSV export that may back ``SourcingInfo``, never a part's identity.

Invariants:

* **Catalog data is distributor data.** A row found for an MPN yields
  :class:`~ai_eda.ir.SourcingInfo` values (supplier part number, stock, unit
  price, assembly class) with ``authoritative`` provenance whose
  :class:`~ai_eda.ir.SourceRef` names the catalog *file* (path, sha256 of its
  bytes, the retrieval date the user declared). It never makes an MPN or a
  manufacturer authoritative - that needs the manufacturer's datasheet
  (:mod:`ai_eda.parts.existence`).
* **No official export exists** (JLCPCB / LCSC have none; probed 2026-09-23),
  so the file is a community dump or a manual export. The importer is
  alias-driven: the canonical columns are :data:`REQUIRED_COLUMNS`
  (``mpn``, ``manufacturer``, ``package``) plus :data:`OPTIONAL_COLUMNS`, and
  :data:`HEADER_ALIASES` maps the header spellings seen in the wild
  (``MFR.Part #``, ``LCSC Part #``, ``brand``, ``model``, ``Type``,
  ``Price (USD)`` ...) onto them. A file whose headers do not cover the
  required columns is refused (:class:`CatalogError`) with the columns it
  lacks; a user maps other exports to these headers.
* **Values are read, not guessed.** Stock must be an integer, a price a plain
  number or the first tier of a quantity-break string
  (``1-499:0.0099,500-1499:0.008`` - the tier is recorded in the row's
  notes); anything else stays ``None``. Cells are stripped (a community file
  carries ``0402\\t``); an all-digit LCSC number under an LCSC header gets its
  ``C`` prefix (recorded as a note). MPN lookup ignores ASCII case and
  whitespace and nothing else.
* **A community dump is untrusted text.** A cell that spreadsheet software
  would execute (starts with ``=``, ``+``, ``-``, ``@`` or carries a control
  character - CSV formula injection, ``=HYPERLINK(...)``, ``-2+3|cmd``)
  skips the whole row with a note (:func:`unsafe_cell`); a supplier part
  number must match :data:`SUPPLIER_PN_RE` (letters, digits, ``.-_/#+``).
  Nothing from a row is ever written verbatim into the BOM without this
  gate (the BOM compiler refuses such cells on its own too). The header
  ``mfr`` means the MPN only in dumps that also carry a separate
  ``manufacturer`` column (the cdfer database); on its own it is ambiguous
  and is left unmapped rather than guessed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_eda.errors import AiEdaError
from ai_eda.ir import SourceRef, SourcingInfo, Traced, authoritative

REQUIRED_COLUMNS: tuple[str, ...] = ("mpn", "manufacturer", "package")
OPTIONAL_COLUMNS: tuple[str, ...] = ("supplier_part_number", "stock", "unit_price", "currency", "assembly_class", "description", "datasheet_url")

#: canonical column -> accepted header spellings (compared lower-cased, whitespace collapsed); first alias wins on a clash.
#: ``mfr`` (last) is the MPN only next to a separate manufacturer column - see :func:`map_headers`
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "mpn": ("mpn", "mfr.part #", "mfr part #", "mfr. part #", "mfr.part#", "manufacturer part number", "manufacturer part", "model", "part number", "part_number", "mfr"),
    "manufacturer": ("manufacturer", "brand", "mfr name", "manufacturer name", "vendor"),
    "package": ("package", "footprint", "case"),
    "supplier_part_number": ("supplier_part_number", "supplier part number", "supplier pn", "lcsc part #", "lcsc part number", "lcsc", "lcsc#", "code", "jlcpcb part #", "jlcpcb part number"),
    "stock": ("stock", "qty", "quantity", "in stock"),
    "unit_price": ("unit_price", "unit price", "price (usd)", "price", "price usd"),
    "currency": ("currency",),
    "assembly_class": ("assembly_class", "assembly class", "library_type", "library type", "library", "type", "assembly", "basic"),
    "description": ("description", "describe", "desc"),
    "datasheet_url": ("datasheet_url", "datasheet", "datasheet url", "file", "url"),
}
#: headers under which an all-digit supplier number is an LCSC C-number without its prefix
_LCSC_HEADERS: frozenset[str] = frozenset({"lcsc part #", "lcsc part number", "lcsc", "lcsc#", "code", "jlcpcb part #", "jlcpcb part number"})
#: headers whose name states the currency
_CURRENCY_IN_HEADER: dict[str, str] = {"price (usd)": "USD", "price usd": "USD"}

_TIER_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+|\+)?|\+)?\s*:\s*([0-9]*\.?[0-9]+)")
_NUMBER_RE = re.compile(r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$")
#: characters a spreadsheet reads as the start of a formula / command
FORMULA_PREFIXES = "=+-@"
#: what a supplier part number may look like (LCSC ``C25792``, Digi-Key ``296-1234-1-ND``, Mouser ``595-LM2596SX-5.0/NOPB``)
SUPPLIER_PN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/#+\-]{0,63}$")
#: text columns the injection gate applies to (numbers are parsed, never copied)
_TEXT_COLUMNS: tuple[str, ...] = ("mpn", "manufacturer", "package", "supplier_part_number", "currency", "assembly_class", "description", "datasheet_url")


def unsafe_cell(value: str | None) -> str | None:
    """Why a (stripped) cell must not be copied anywhere: a formula / command prefix or a control character; ``None`` when it is plain text."""
    if not value:
        return None
    if value[0] in FORMULA_PREFIXES:
        return f"starts with {value[0]!r} (a spreadsheet would read it as a formula or command)"
    if any(ord(ch) < 0x20 or ch == "\x7f" for ch in value):
        return "contains a control character"
    return None


class CatalogError(AiEdaError):
    """The catalog file cannot be used (unreadable, headers do not cover the required columns, bad date)."""


def mpn_key(mpn: str) -> str:
    """Lookup key of a part number: whitespace removed, ASCII case folded (nothing else - ``LM2931AZ-5.0/NOPB`` stays itself)."""
    return "".join(str(mpn).split()).casefold()


def normalise_header(header: str) -> str:
    return " ".join((header or "").replace("﻿", "").strip().lower().split())


def map_headers(headers: list[str]) -> tuple[dict[str, int], list[str]]:
    """``(canonical column -> index, notes)`` for a header row; unmapped headers and clashes are noted, never guessed."""
    mapping: dict[str, int] = {}
    mapped_from: dict[str, str] = {}
    notes: list[str] = []
    normalised = [normalise_header(h) for h in headers]
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            for i, h in enumerate(normalised):
                if h == alias and i not in mapping.values():
                    if canonical in mapping:
                        notes.append(f"header {headers[i]!r} also means {canonical}; {mapped_from[canonical]!r} kept")
                        continue
                    mapping[canonical] = i
                    mapped_from[canonical] = headers[i]
    if "mpn" in mapping and normalised[mapping["mpn"]] == "mfr" and "manufacturer" not in mapping:
        # 'mfr' alone is as likely the manufacturer as the part number: never guessed
        notes.append(f"header {mapped_from['mpn']!r} is ambiguous (manufacturer or MPN) without a separate manufacturer column; not mapped")
        del mapping["mpn"]
    for i, h in enumerate(headers):
        if i not in mapping.values():
            notes.append(f"header {h!r} not mapped (ignored)")
    return mapping, notes


def parse_stock(text: str | None) -> tuple[int | None, str | None]:
    if text is None or not text.strip():
        return None, None
    s = text.strip().replace(",", "")
    if s.isdigit():
        return int(s), None
    return None, f"stock {text!r} is not an integer"


def parse_price(text: str | None) -> tuple[float | None, str | None]:
    """A plain number, or the first tier of a quantity-break string (tier recorded); ``None`` with a note otherwise."""
    if text is None or not text.strip():
        return None, None
    s = text.strip()
    if _NUMBER_RE.match(s):
        return float(s), None
    m = _TIER_RE.match(s)
    if m:
        hi = m.group(2) or "+"
        return float(m.group(3)), f"price taken from the first quantity break {m.group(1)}-{hi}: {m.group(3)}"
    return None, f"price {text!r} not parseable"


class CatalogRow(BaseModel):
    """One catalog line, cells stripped, numbers parsed or ``None``."""

    line: int  # 1-based line in the file (header is line 1)
    mpn: str
    manufacturer: str
    package: str
    supplier_part_number: str | None = None
    stock: int | None = None
    unit_price: float | None = None
    currency: str | None = None
    assembly_class: str | None = None
    description: str | None = None
    datasheet_url: str | None = None
    #: what the importer changed or could not read about this row
    notes: list[str] = Field(default_factory=list)
    #: the cells as they stand in the file, by canonical column
    raw: dict[str, str] = Field(default_factory=dict)


def _decode(data: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace"), "cp1252-replace"


class CatalogSource:
    """A loaded catalog file: rows by MPN key, and the :class:`~ai_eda.ir.SourceRef` every value points at."""

    def __init__(
        self,
        path: Path,
        sha256: str,
        retrieved_at: str,
        authority: str,
        supplier: str,
        rows: list[CatalogRow],
        columns: dict[str, int],
        notes: list[str],
        encoding: str,
    ) -> None:
        self.path = path
        self.sha256 = sha256
        self.retrieved_at = retrieved_at
        self.authority = authority
        self.supplier = supplier
        self.rows = rows
        self.columns = columns
        self.notes = notes
        self.encoding = encoding
        self._by_key: dict[str, CatalogRow] = {}
        self.duplicates: dict[str, int] = {}
        for row in rows:
            key = mpn_key(row.mpn)
            if key in self._by_key:
                self.duplicates[key] = self.duplicates.get(key, 1) + 1
                continue
            self._by_key[key] = row

    # ------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: Path | str, retrieved_at: datetime | str, authority: str, *, supplier: str | None = None) -> CatalogSource:
        """Read a CSV export; ``retrieved_at`` is the user's declaration of when it was exported (ISO 8601).

        ``authority`` names who produced the data (``"JLCPCB export"``,
        ``"LCSC export"``); ``supplier`` is the :attr:`SourcingInfo.supplier`
        label (defaults to ``authority``). Raises :class:`CatalogError` when
        the file is unreadable, the date is not ISO 8601 or the headers do not
        cover :data:`REQUIRED_COLUMNS`.
        """
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as e:
            raise CatalogError(f"cannot read catalog {p}: {e}") from e
        when = retrieved_at.isoformat() if isinstance(retrieved_at, datetime) else str(retrieved_at)
        try:
            datetime.fromisoformat(when)
        except ValueError as e:
            raise CatalogError(f"retrieved_at must be an ISO 8601 date/time, got {retrieved_at!r}") from e
        text, encoding = _decode(data)
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            headers = next(reader)
        except StopIteration as e:
            raise CatalogError(f"catalog {p} is empty") from e
        columns, notes = map_headers(headers)
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise CatalogError(
                f"catalog {p.name} lacks the required column(s) {missing}; headers found: {headers}. "
                f"Map the export to the headers mpn, manufacturer, package (optional: {', '.join(OPTIONAL_COLUMNS)})"
            )
        header_currency = None
        if "unit_price" in columns:
            header_currency = _CURRENCY_IN_HEADER.get(normalise_header(headers[columns["unit_price"]]))
        lcsc_column = "supplier_part_number" in columns and normalise_header(headers[columns["supplier_part_number"]]) in _LCSC_HEADERS
        rows: list[CatalogRow] = []
        for n, cells in enumerate(reader, start=2):
            if not any(c.strip() for c in cells):
                continue

            def cell(col: str) -> str | None:
                i = columns.get(col)
                if i is None or i >= len(cells):
                    return None
                return cells[i].strip()

            mpn = cell("mpn") or ""
            if not mpn:
                notes.append(f"line {n}: empty mpn; skipped")
                continue
            unsafe = [(col, why) for col in _TEXT_COLUMNS if (why := unsafe_cell(cell(col))) is not None]
            if unsafe:
                notes.append(f"line {n}: skipped - " + "; ".join(f"{col} cell {cell(col)!r} {why}" for col, why in unsafe))
                continue
            row_notes: list[str] = []
            stock, note = parse_stock(cell("stock"))
            if note:
                row_notes.append(note)
            price, note = parse_price(cell("unit_price"))
            if note:
                row_notes.append(note)
            currency = cell("currency") or (header_currency if price is not None else None)
            spn = cell("supplier_part_number") or None
            if spn and lcsc_column and spn.isdigit():
                row_notes.append(f"LCSC number {spn} given without its C prefix; normalised to C{spn}")
                spn = f"C{spn}"
            if spn and not SUPPLIER_PN_RE.match(spn):
                notes.append(f"line {n}: skipped - supplier part number {spn!r} is not a part-number-like token ({SUPPLIER_PN_RE.pattern})")
                continue
            raw = {col: (cells[i] if i < len(cells) else "") for col, i in columns.items()}
            rows.append(CatalogRow(
                line=n, mpn=mpn, manufacturer=cell("manufacturer") or "", package=cell("package") or "", supplier_part_number=spn,
                stock=stock, unit_price=price, currency=currency, assembly_class=cell("assembly_class") or None,
                description=cell("description") or None, datasheet_url=cell("datasheet_url") or None, notes=row_notes, raw=raw,
            ))
        sha = "sha256:" + hashlib.sha256(data).hexdigest()
        return cls(p, sha, when, authority, supplier or authority, rows, columns, notes, encoding)

    # ------------------------------------------------------------ lookup

    @property
    def source_ref(self) -> SourceRef:
        return SourceRef(
            title=f"{self.supplier} catalog export {self.path.name}",
            authority=self.authority,
            document_path=str(self.path),
            content_hash=self.sha256,
            retrieved_at=datetime.fromisoformat(self.retrieved_at),
        )

    def lookup(self, mpn: str | None) -> CatalogRow | None:
        """The first row whose MPN equals ``mpn`` ignoring ASCII case and whitespace, else ``None``."""
        if not mpn or not str(mpn).strip():
            return None
        return self._by_key.get(mpn_key(mpn))

    def sourcing_info(self, row: CatalogRow) -> SourcingInfo:
        """``SourcingInfo`` for a row: every value ``authoritative`` with this file as its source (row noted)."""
        section = f"line {row.line}"
        src = self.source_ref.model_copy(update={"section": section})
        note = f"catalog row {row.line} for MPN {row.mpn!r}"
        if row.notes:
            note += "; " + "; ".join(row.notes)

        def traced(value: Any, unit: str | None = None) -> Traced:
            return authoritative(value, src, unit=unit, note=note)

        return SourcingInfo(
            supplier=self.supplier,
            supplier_part_number=traced(row.supplier_part_number) if row.supplier_part_number else None,
            stock=traced(row.stock) if row.stock is not None else None,
            unit_price=traced(row.unit_price, row.currency) if row.unit_price is not None else None,
            currency=row.currency,
            assembly_class=traced(row.assembly_class) if row.assembly_class else None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "sha256": self.sha256, "retrieved_at": self.retrieved_at, "authority": self.authority,
            "supplier": self.supplier, "rows": len(self.rows), "columns": dict(self.columns), "encoding": self.encoding,
            "duplicate_mpns": len(self.duplicates), "notes": list(self.notes),
        }


__all__ = [
    "FORMULA_PREFIXES",
    "HEADER_ALIASES",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUPPLIER_PN_RE",
    "CatalogError",
    "CatalogRow",
    "CatalogSource",
    "map_headers",
    "mpn_key",
    "normalise_header",
    "parse_price",
    "parse_stock",
    "unsafe_cell",
]
