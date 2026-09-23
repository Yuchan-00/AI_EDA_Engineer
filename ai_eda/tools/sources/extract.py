"""Deterministic text extraction for archived documents.

Invariant: the text a quote is grounded against is produced by a named
extractor at a named version, and both are recorded in the document's
``meta.json`` next to the sha256 of the text. Same bytes + same extractor
version -> same pages; a changed version is visible, never silent.

Extractors (all deterministic, no network, no model):

* ``pypdf`` - one entry of ``pages`` per PDF page (``page.extract_text()``),
  followed by :func:`normalise_pdf_text`: line endings unified, soft hyphens
  (U+00AD) dropped, Latin ligature code points (ﬀ ﬁ ﬂ ﬃ ﬄ) spelled out, NBSP
  made a plain space. Nothing else is touched - no NFKC (it would turn ``mm²``
  into ``mm2`` and change the meaning of a quantity). An encrypted PDF that
  pypdf can not open without the optional ``cryptography`` package (AES-
  protected Microchip / JST datasheets) yields no pages and an ``error``; the
  bytes are still archived, the caller reports "text not extractable".
* ``html.parser`` - the stdlib parser; ``script`` / ``style`` / ``noscript``
  / ``template`` / ``svg`` content is dropped, every block-level element
  (paragraph, heading, list item, table cell, ``<br>``, the ``<title>`` …)
  stands on its own line, runs of spaces collapse, no blank lines. One
  page. The ``<title>`` text is also kept in the meta and used for
  interstitial detection.
* ``xml.etree`` - for XML documents (the Korean 법제처 open API serves
  statutes as XML with Korean element names and CDATA bodies, which
  ``html.parser`` does not treat as markup): every element's text and tail
  on its own line. One page.
* ``text`` - ``text/plain`` and JSON as one page, decoded with
  :func:`decode_text`.

:func:`sniff_kind` decides which extractor applies from the *bytes first*
(``%PDF-`` within the first :data:`PDF_HEADER_WINDOW` bytes is a PDF whatever
the ``Content-Type`` says - the PDF specification allows leading bytes before
the header, and servers prepend CR/LF or a BOM; pypdf 6 reads such files) and
the content type second; a body that is neither is ``binary`` and is not
archived as a document. The HTML stripper keeps the text of ``<noscript>``
elements apart (``ExtractedText.hidden``): it is not document text, but it is
where script-rendered shells say "please enable JavaScript" or "access
denied", so the archive scans it for bot-protection markers.
"""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from typing import Literal
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

Kind = Literal["pdf", "html", "xml", "text", "json", "binary"]

PDF_EXTRACTOR = "pypdf"
#: bumped when :func:`extract_pdf` / :func:`normalise_pdf_text` change what they produce for the same bytes
PDF_EXTRACTOR_VERSION = "1"
HTML_EXTRACTOR = "html.parser"
HTML_EXTRACTOR_VERSION = "1"
XML_EXTRACTOR = "xml.etree"
XML_EXTRACTOR_VERSION = "1"
TEXT_EXTRACTOR = "text"
TEXT_EXTRACTOR_VERSION = "1"

_EXTENSIONS: dict[str, str] = {"pdf": "pdf", "html": "html", "xml": "xml", "text": "txt", "json": "json", "binary": "bin"}

_BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "nav", "main", "aside", "title", "blockquote", "pre", "hr",
    "dt", "dd", "dl", "tbody", "thead", "tfoot", "caption", "form", "fieldset", "legend", "option", "label",
    "address", "figure", "figcaption", "body", "html", "head", "iframe",
})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
#: the PDF header may be preceded by this many bytes of junk (PDF 32000-1 section 7.5.2 implementation note)
PDF_HEADER_WINDOW = 1024

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))


class ExtractedText(BaseModel):
    """What an extractor produced for one document."""

    pages: list[str] = Field(default_factory=list)
    extractor: str
    extractor_version: str
    #: version of the third-party library used (pypdf), ``None`` for stdlib extractors
    library_version: str | None = None
    #: text encoding used to decode a text document, ``None`` for PDFs
    encoding: str | None = None
    #: the HTML ``<title>`` (whitespace collapsed) when there is one
    title: str | None = None
    #: text of ``<noscript>`` elements (HTML only): not document text, but scanned for bot-protection markers
    hidden: str | None = None
    #: why extraction failed (pages then empty) or was partial (a page that raised is empty)
    error: str | None = None

    @property
    def text_available(self) -> bool:
        return any(p.strip() for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)


def extension_for(kind: str) -> str:
    """File extension the archive stores a document of ``kind`` under."""
    return _EXTENSIONS.get(kind, "bin")


def _main_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def pdf_header_offset(body: bytes) -> int | None:
    """Offset of ``%PDF-`` when it stands within the first :data:`PDF_HEADER_WINDOW` bytes and no markup precedes it, else ``None``.

    Leading whitespace, a BOM or a few junk bytes before the header are what
    the PDF specification tolerates; a ``<`` before it means the bytes are
    markup that merely mentions ``%PDF-`` (an HTML viewer page), not a PDF.
    """
    idx = body[:PDF_HEADER_WINDOW].find(b"%PDF-")
    if idx < 0 or b"<" in body[:idx]:
        return None
    return idx


def sniff_kind(body: bytes, content_type: str | None = None, name: str | None = None) -> Kind:
    """Which extractor applies: the bytes decide first, the content type second, the file name last."""
    head = body.lstrip()[:64]
    if pdf_header_offset(body) is not None:
        return "pdf"
    ct = _main_type(content_type)
    if ct in ("text/html", "application/xhtml+xml"):
        return "html"
    if ct in ("application/xml", "text/xml") or ct.endswith("+xml"):
        return "xml"
    if ct == "application/json" or ct.endswith("+json"):
        return "json"
    if ct.startswith("text/"):
        return "text"
    lowered = head.lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    if lowered.startswith(b"<?xml"):
        return "xml"
    suffix = (name or "").rsplit(".", 1)[-1].lower() if name and "." in name else ""
    if suffix in ("html", "htm"):
        return "html"
    if suffix == "xml":
        return "xml"
    if suffix == "json":
        return "json"
    if suffix in ("txt", "md", "csv", "text"):
        return "text"
    if suffix == "pdf":
        return "binary"  # named .pdf but not a PDF: never treat it as one
    if not ct and _decodes(body, "utf-8"):
        return "text"
    return "binary"


def _decodes(body: bytes, encoding: str) -> bool:
    try:
        body.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return False
    return True


_CHARSET_RE = re.compile(r"charset=[\"']?\s*([\w.-]+)", re.I)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset=[\"']?\s*([\w.-]+)", re.I)
_XML_DECL_RE = re.compile(rb"<\?xml[^>]*encoding=[\"']([\w.-]+)[\"']", re.I)


def decode_text(body: bytes, content_type: str | None = None) -> tuple[str, str]:
    """``(text, encoding)``: the declared charset, a BOM, an in-document declaration, then strict utf-8, strict cp949, utf-8 with replacement."""
    candidates: list[str] = []
    m = _CHARSET_RE.search(content_type or "")
    if m:
        candidates.append(m.group(1))
    if body.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")
    elif body.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.append("utf-16")
    head = body[:4096]
    for rx in (_META_CHARSET_RE, _XML_DECL_RE):
        dm = rx.search(head)
        if dm:
            candidates.append(dm.group(1).decode("ascii", "ignore"))
    candidates += ["utf-8", "cp949"]
    seen: set[str] = set()
    for enc in candidates:
        key = enc.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            return body.decode(enc), key
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", errors="replace"), "utf-8-replace"


class _TextStripper(HTMLParser):
    """Drops script/style, emits newlines at block tags, remembers the title and keeps ``<noscript>`` text apart."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_parts: list[str] = []
        self._skip = 0
        self._in_noscript = 0
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag == "noscript":
            self._in_noscript += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "noscript" and self._in_noscript:
            self._in_noscript -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            if self._in_noscript:
                # a <noscript> may hold markup of its own (html.parser hands it over as data); keep only its words
                self.hidden_parts.append(re.sub(r"<[^>]*>", " ", data))
            return
        if self._in_title:
            t = " ".join(data.split())
            if t:
                self.title = t if self.title is None else f"{self.title} {t}"
        self.parts.append(data)

    def unknown_decl(self, data: str) -> None:
        # <![CDATA[...]]> arrives here; its content is text
        if data.startswith("CDATA[") and not self._skip:
            self.parts.append(data[6:])


def collapse_text(text: str) -> str:
    """NBSP to space, horizontal whitespace runs to one space, every block on exactly one line (newline runs collapsed)."""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def strip_html(html: str) -> tuple[str, str | None]:
    """``(text, title)`` of an HTML document: scripts, styles and noscript dropped, block elements on their own lines."""
    text, title, _hidden = strip_html_with_hidden(html)
    return text, title


def strip_html_with_hidden(html: str) -> tuple[str, str | None, str | None]:
    """``(text, title, noscript text)``: as :func:`strip_html`, plus the ``<noscript>`` words kept apart (``None`` when there are none)."""
    p = _TextStripper()
    p.feed(html)
    p.close()
    hidden = collapse_text(" ".join(p.hidden_parts)) or None
    return collapse_text("".join(p.parts)), p.title, hidden


def normalise_pdf_text(text: str) -> str:
    """The documented, minimal normalisation of pypdf output (see the module docstring)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "").replace("\xa0", " ")
    return _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group(0)], text)


def extract_pdf(body: bytes) -> ExtractedText:
    """One page of text per PDF page via pypdf; a page that fails is empty and named in ``error``."""
    try:
        import pypdf
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - pypdf is an optional dependency
        return ExtractedText(extractor=PDF_EXTRACTOR, extractor_version=PDF_EXTRACTOR_VERSION, error=f"pypdf not installed: {e}")
    lib = getattr(pypdf, "__version__", None)
    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            reader.decrypt("")
        n = len(reader.pages)
    except Exception as e:  # pypdf raises many types (PdfReadError, DependencyError, ...)
        return ExtractedText(extractor=PDF_EXTRACTOR, extractor_version=PDF_EXTRACTOR_VERSION, library_version=lib,
                             error=f"{type(e).__name__}: {e}")
    pages: list[str] = []
    failures: list[str] = []
    for i in range(n):
        try:
            pages.append(normalise_pdf_text(reader.pages[i].extract_text() or ""))
        except Exception as e:
            pages.append("")
            failures.append(f"page {i + 1}: {type(e).__name__}: {e}")
    return ExtractedText(
        pages=pages, extractor=PDF_EXTRACTOR, extractor_version=PDF_EXTRACTOR_VERSION, library_version=lib,
        error="; ".join(failures) or None,
    )


def extract_html(body: bytes, content_type: str | None = None) -> ExtractedText:
    text, enc = decode_text(body, content_type)
    stripped, title, hidden = strip_html_with_hidden(text)
    return ExtractedText(pages=[stripped], extractor=HTML_EXTRACTOR, extractor_version=HTML_EXTRACTOR_VERSION, encoding=enc, title=title, hidden=hidden)


def extract_xml(body: bytes, content_type: str | None = None) -> ExtractedText:
    """Every element's text and tail on its own line; falls back to the HTML stripper when the XML is not well-formed."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        out = extract_html(body, content_type)
        out.error = f"not well-formed XML ({e}); stripped as HTML"
        return out
    parts: list[str] = []
    for el in root.iter():
        if el.text and el.text.strip():
            parts.append(el.text)
        if el.tail and el.tail.strip():
            parts.append(el.tail)
    enc = None
    m = _XML_DECL_RE.search(body[:4096])
    if m:
        enc = m.group(1).decode("ascii", "ignore").lower()
    return ExtractedText(pages=[collapse_text("\n".join(parts))], extractor=XML_EXTRACTOR, extractor_version=XML_EXTRACTOR_VERSION, encoding=enc or "utf-8")


def extract_plain(body: bytes, content_type: str | None = None) -> ExtractedText:
    text, enc = decode_text(body, content_type)
    return ExtractedText(pages=[text.replace("\r\n", "\n").replace("\r", "\n")], extractor=TEXT_EXTRACTOR, extractor_version=TEXT_EXTRACTOR_VERSION, encoding=enc)


def extract_document(body: bytes, kind: str, content_type: str | None = None) -> ExtractedText:
    """Dispatch on :func:`sniff_kind`'s answer; ``binary`` raises ``ValueError`` (it is not a document)."""
    if kind == "pdf":
        return extract_pdf(body)
    if kind == "html":
        return extract_html(body, content_type)
    if kind == "xml":
        return extract_xml(body, content_type)
    if kind in ("text", "json"):
        return extract_plain(body, content_type)
    raise ValueError(f"no text extractor for kind {kind!r}")


__all__ = [
    "HTML_EXTRACTOR",
    "HTML_EXTRACTOR_VERSION",
    "PDF_EXTRACTOR",
    "PDF_EXTRACTOR_VERSION",
    "PDF_HEADER_WINDOW",
    "TEXT_EXTRACTOR",
    "TEXT_EXTRACTOR_VERSION",
    "XML_EXTRACTOR",
    "XML_EXTRACTOR_VERSION",
    "ExtractedText",
    "Kind",
    "collapse_text",
    "decode_text",
    "extension_for",
    "extract_document",
    "extract_html",
    "extract_pdf",
    "extract_plain",
    "extract_xml",
    "normalise_pdf_text",
    "pdf_header_offset",
    "sniff_kind",
    "strip_html",
    "strip_html_with_hidden",
]
