"""A minimal, deterministic PDF generator for tests (pure Python, no library).

:func:`build_pdf` takes pages as lists of text lines and returns the bytes of a
PDF 1.4 file: one Helvetica (standard-14, WinAnsiEncoding) ``BT … Tj ET`` text
object per line at an absolute position, so pypdf extracts each line as its
own run, top to bottom, verbatim. Same input -> byte-identical output (no
dates, no ids, no compression), so tests can assert on sha256 names.

Limits, by design: WinAnsi text only (other characters become ``?``), no
embedded fonts, no images. Measured on 2026-09-23 against pypdf 6.19.0: every
line of a three-page fixture came back verbatim in both plain and layout
extraction modes.
"""

from __future__ import annotations

PAGE_W, PAGE_H = 612, 792  # US Letter, points
MARGIN_L, TOP_Y, LEADING, FONT_SIZE = 72, 720, 14, 11


def _pdf_string(s: str) -> bytes:
    """Literal PDF string in WinAnsiEncoding (cp1252) with ``(``, ``)`` and ``\\`` escaped."""
    raw = s.encode("cp1252", errors="replace")
    raw = raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + raw + b")"


def _content_stream(lines: list[str]) -> bytes:
    parts = []
    y = TOP_Y
    for line in lines:
        parts.append(b"BT /F1 %d Tf %d %d Td " % (FONT_SIZE, MARGIN_L, y) + _pdf_string(line) + b" Tj ET")
        y -= LEADING
    return b"\n".join(parts) + b"\n"


def build_pdf(pages: list[list[str]]) -> bytes:
    """The bytes of a PDF with ``len(pages)`` pages, each drawing its lines in Helvetica."""
    if not pages:
        raise ValueError("a PDF needs at least one page")
    objects: list[bytes] = []  # objects[i] is object number i+1 (body without "N 0 obj"/"endobj")
    n_pages = len(pages)
    page_obj_nums = [4 + 2 * i for i in range(n_pages)]
    kids = b" ".join(b"%d 0 R" % n for n in page_obj_nums)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % n_pages)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        page_num = page_obj_nums[i]
        content_num = page_num + 1
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>"
            % (PAGE_W, PAGE_H, content_num)
        )
        stream = _content_stream(lines)
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_pos)
    return bytes(out)


#: a datasheet-like fixture: family number on page 1, orderable code in a later addendum, a value with a unit
DATASHEET_PAGES: list[list[str]] = [
    ["LM2931-N Series Low Dropout Regulators", "Fixed 5 V and adjustable output", "Output voltage 5 V +/- 3.8 %", "Dropout voltage 0.6 V at 100 mA"],
    ["Electrical Characteristics", "Quiescent current 0.4 mA (typ)", "Operating junction temperature -40 to 125 degC"],
    ["Package Option Addendum", "Orderable device: LM2931AZ-5.0/NOPB  TO-92  Tube", "Orderable device: LM2931AM-5.0/NOPB  SOIC-8  Tape and Reel"],
]


def datasheet_pdf() -> bytes:
    return build_pdf(DATASHEET_PAGES)


__all__ = ["DATASHEET_PAGES", "build_pdf", "datasheet_pdf"]
