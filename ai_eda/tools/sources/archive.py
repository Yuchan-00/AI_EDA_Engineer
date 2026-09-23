"""Document archive - the only way an external document enters the system, and the only module that opens a socket.

Invariants this module enforces:

* **A fetch is an outward action.** :meth:`DocumentArchive.fetch` calls
  :meth:`~ai_eda.tools.sources.policy.NetworkPolicy.require_online` before
  building a request; without the user's online session it raises
  :class:`~ai_eda.errors.ApprovalRequiredError` and no socket is opened.
  Only URLs the policy trusts are attempted (status ``refused`` otherwise),
  redirects are followed one hop at a time and each target is checked
  against the policy before it is contacted, only ``https`` is used, one
  attempt is made per URL per run (a second call returns the recorded
  outcome), every request carries a User-Agent naming the project and a
  20 s timeout, and nothing is ever retried in a loop.
* **What is archived is exactly what was received.** A successful fetch is
  stored as ``<root>/<sha256>.<ext>`` next to ``<sha256>.meta.json`` (URL,
  final URL, redirect chain, retrieval time, HTTP status, content type, size,
  sha256, extractor + version, text sha256, ...). The file name *is* the
  content hash, so the same bytes archived twice land in one file, and
  :meth:`DocumentArchive.load` re-hashes the file and raises
  :class:`TamperedDocumentError` when it no longer matches - a
  :class:`~ai_eda.ir.SourceRef` that points at such a file is ``tampered``
  (:meth:`DocumentArchive.verify`) and whatever it supported is
  ``NOT_VERIFIED``.
* **A response is a document only when it is one.** A 404/410 is
  ``missing``; 401/403/407/429/503, a body below :data:`MIN_BODY_BYTES`, an
  HTML page where a PDF was expected, HTML whose readable text is below
  :data:`MIN_HTML_TEXT_CHARS` (a script-rendered shell) or whose title / head
  carries a :data:`BLOCK_MARKERS` phrase ("Just a moment", "Access Denied",
  captcha, ...) is ``blocked``; transport failures, timeouts, other HTTP
  statuses and oversize bodies are ``error``. None of these is archived as a
  document; all of them are recorded in ``<root>/fetch_log.jsonl`` with the
  body's hash so a human can see what came back.
* **Reading a document is not compliance and not verification.** The
  archive proves *what text a document contains*; :meth:`ArchivedDocument.find_quote`
  locates a claimed phrase with the same normalisation the requirement
  stage uses (:func:`ai_eda.llm.extraction.find_quote`: exact characters,
  whitespace free, token boundaries) and reports the page and a context
  snippet. Deciding what the text *means* is the caller's job, and never a
  model's.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, Field

from ai_eda import __version__
from ai_eda.errors import AiEdaError
from ai_eda.ir.provenance import SourceRef
from ai_eda.llm.extraction import find_quote as _find_quote_span
from ai_eda.llm.extraction import quote_context
from ai_eda.tools.sources.extract import PDF_HEADER_WINDOW, ExtractedText, extension_for, extract_document, sniff_kind
from ai_eda.tools.sources.policy import NetworkPolicy, normalise_url

if TYPE_CHECKING:  # httpx is imported lazily so an offline archive (add_file / load / verify) needs no network client installed
    import httpx

log = logging.getLogger(__name__)

#: sent with every request; names the project and what it does
USER_AGENT = f"ai-eda-engineer/{__version__} (verification-first EDA; archives datasheets and official texts for provenance)"
#: seconds for connect / read / write / pool on every request
DEFAULT_TIMEOUT_S = 20.0
#: longest redirect chain followed (LCSC mirrors chain three 301s, Vishay a 301 then a 307)
MAX_REDIRECTS = 10
#: a body above this is not read to the end and the fetch is an ``error``
MAX_BODY_BYTES = 64 * 1024 * 1024
#: a 200 body below this is not a document (empty shells, "OK")
MIN_BODY_BYTES = 64
#: an HTML page whose stripped text is shorter than this is a script-rendered shell or an interstitial, not a document
MIN_HTML_TEXT_CHARS = 200
#: characters of document text shown on each side of a quote hit
QUOTE_CONTEXT_CHARS = 60
#: phrases (case-insensitive, in the title or the first :data:`BLOCK_SCAN_CHARS` of the text) that mark bot protection / interstitials
BLOCK_MARKERS: tuple[str, ...] = (
    "just a moment",
    "access denied",
    "captcha",
    "unblock",
    "enable javascript",
    "attention required",
    "checking your browser",
    "verify you are human",
    "bot detection",
    "request blocked",
    "cf-chl",
    "please enable cookies",
)
BLOCK_SCAN_CHARS = 2000
BLOCKED_HTTP_STATUSES = frozenset({401, 403, 407, 429, 503})
MISSING_HTTP_STATUSES = frozenset({404, 410})
META_SCHEMA = 1

FetchStatus = Literal["ok", "blocked", "missing", "refused", "error"]
Expect = Literal["pdf", "html", "any"]
VerifyStatus = Literal["ok", "missing", "tampered", "unarchived"]

#: ASCII-only lower-casing (length-preserving, unlike ``str.lower`` on ``İ`` or ``str.casefold`` on ``ß``)
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

_ACCEPT: dict[str, str] = {
    "pdf": "application/pdf,*/*;q=0.8",
    "html": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    "any": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class ArchiveError(AiEdaError):
    """The archive can not do what was asked (no such document, unusable file)."""


class DocumentMissingError(ArchiveError):
    """No archived document with that hash."""


class TamperedDocumentError(ArchiveError):
    """The archived file no longer hashes to its name: whatever it supported is NOT_VERIFIED."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(data: bytes) -> str:
    """``sha256:<hex>`` - the form every ``content_hash`` in the IR uses."""
    return SourceRef.hash_bytes(data)


def sha256_hex(sha: str) -> str:
    """The bare hex of ``sha256:<hex>`` or of a bare hex string (``ValueError`` otherwise)."""
    h = sha.strip().lower()
    if h.startswith("sha256:"):
        h = h[7:]
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"not a sha256: {sha!r}")
    return h


def text_hash(pages: list[str]) -> str:
    """sha256 of the extracted pages (joined by form feed) - ties a quote hit to one extraction."""
    return sha256_of("\x0c".join(pages).encode("utf-8"))


def find_block_marker(title: str | None, text: str, hidden: str | None = None) -> str | None:
    """The first :data:`BLOCK_MARKERS` phrase in the title, the head of the text or the ``<noscript>`` text, else ``None``.

    ``hidden`` is what the HTML stripper keeps apart from the document text
    (:attr:`~ai_eda.tools.sources.extract.ExtractedText.hidden`): a
    script-rendered shell says "please enable JavaScript" or "access denied"
    there, invisible in the stripped page.
    """
    hay = ((title or "") + "\n" + text[:BLOCK_SCAN_CHARS] + "\n" + (hidden or "")[:BLOCK_SCAN_CHARS]).lower()
    for marker in BLOCK_MARKERS:
        if marker in hay:
            return marker
    return None


class QuoteHit(BaseModel):
    """Where a quote stands in an archived document."""

    page: int  # 1-based
    offset: int  # character offset in that page's text
    #: the document's own text at the hit (whitespace as it stands there)
    matched: str
    #: the text around the hit with the match in brackets
    context: str

    @property
    def section(self) -> str:
        return f"page {self.page}"


class ArchivedDocument(BaseModel):
    """A document on disk whose bytes hash to ``sha256`` and whose ``pages`` came from the recorded extractor."""

    sha256: str
    path: Path
    meta: dict[str, Any]
    pages: list[str] = Field(default_factory=list)
    text_sha256: str
    extraction_error: str | None = None
    #: the extractor that produced ``pages`` *now* (``meta`` records the one used at archive time; they differ after an upgrade)
    extractor: str | None = None
    extractor_version: str | None = None
    library_version: str | None = None

    # ------------------------------------------------------------ meta accessors

    @property
    def url(self) -> str | None:
        return self.meta.get("url")

    @property
    def final_url(self) -> str | None:
        return self.meta.get("final_url")

    @property
    def title(self) -> str | None:
        return self.meta.get("title")

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind", "binary"))

    @property
    def content_type(self) -> str | None:
        return self.meta.get("content_type")

    @property
    def retrieved_at(self) -> datetime | None:
        raw = self.meta.get("retrieved_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_available(self) -> bool:
        return any(p.strip() for p in self.pages)

    @property
    def text_matches_meta(self) -> bool:
        """Whether the current extraction hashes to what ``meta.json`` recorded (false after an extractor change)."""
        return self.meta.get("text_sha256") == self.text_sha256

    @property
    def extraction_stamp(self) -> str:
        """Which extractor produced the text a claim was grounded against, and that text's hash - for provenance notes.

        ``extractor pypdf 1 (lib 6.19.0); text sha256:...``: a later reader
        can tell whether the text it re-extracts is the text the claim rests
        on (:attr:`text_matches_meta` says the same against the meta).
        """
        name = self.extractor or self.meta.get("extractor") or "unknown"
        version = self.extractor_version or self.meta.get("extractor_version") or "?"
        lib = self.library_version or self.meta.get("library_version")
        return f"extractor {name} {version}" + (f" (lib {lib})" if lib else "") + f"; text {self.text_sha256}"

    # ------------------------------------------------------------ grounding

    def find_quote(self, quote: str | None, page: int | None = None, *, ignore_case: bool = False) -> list[QuoteHit]:
        """The first hit of ``quote`` on every page (or on ``page`` only), with the requirement stage's normalisation.

        Characters must match exactly, whitespace may differ, and the match
        must sit at a token boundary (``LM2931`` is not found inside
        ``LM2931AZ``). ``ignore_case=True`` lowers ASCII letters on both sides
        (length-preserving, so offsets stay exact) - for part numbers, which
        datasheets set in either case; ``m``/``M`` in a quantity still differ
        only when the caller keeps the default. A ``page`` outside the
        document gives no hit; the caller reports the claim as ungrounded.
        """
        if not quote or not quote.strip():
            return []
        if page is not None:
            if page < 1 or page > len(self.pages):
                return []
            candidates = [(page, self.pages[page - 1])]
        else:
            candidates = list(enumerate(self.pages, start=1))
        needle = quote.translate(_ASCII_LOWER) if ignore_case else quote
        hits: list[QuoteHit] = []
        for n, text in candidates:
            span = _find_quote_span(needle, text.translate(_ASCII_LOWER) if ignore_case else text)
            if span is None:
                continue
            hits.append(QuoteHit(page=n, offset=span[0], matched=text[span[0]:span[1]], context=quote_context(text, span, QUOTE_CONTEXT_CHARS)))
        return hits

    def source_ref(self, title: str | None = None, section: str | None = None, authority: str | None = None) -> SourceRef:
        """A :class:`~ai_eda.ir.SourceRef` that names this archived file by hash (``document_path`` + ``content_hash`` + ``retrieved_at``)."""
        return SourceRef(
            title=title or self.title or self.final_url or self.url or self.path.name,
            url=self.final_url or self.url,
            authority=authority if authority is not None else self.meta.get("authority"),
            section=section,
            document_path=str(self.path),
            content_hash=self.sha256,
            retrieved_at=self.retrieved_at,
        )


class FetchOutcome(BaseModel):
    """What one attempt at one URL produced. ``document`` is set only when ``status == "ok"``."""

    status: FetchStatus
    url: str
    normalised_url: str | None = None
    final_url: str | None = None
    document: ArchivedDocument | None = None
    reason: str | None = None
    #: one entry per hop followed: ``{"status", "url", "location"}``
    redirects: list[dict[str, Any]] = Field(default_factory=list)
    http_status: int | None = None
    content_type: str | None = None
    size: int | None = None
    #: hash of the body that came back, also for responses that were not archived
    body_sha256: str | None = None
    #: first characters of the response's text as a human would read it (page 1 for a PDF), also for responses that were not archived
    text_head: str | None = None
    purpose: str = ""
    expect: str = "any"
    retrieved_at: str = Field(default_factory=_now_iso)
    elapsed_s: float = 0.0
    #: why the URL was allowed (KiCad datasheet host, official domain, user URL)
    trusted_by: str | None = None
    #: what :func:`~ai_eda.tools.sources.policy.normalise_url` changed about the URL
    url_note: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.document is not None

    def log_record(self) -> dict[str, Any]:
        rec = self.model_dump(mode="json", exclude={"document"})
        rec["sha256"] = self.document.sha256 if self.document is not None else None
        rec["archived_path"] = str(self.document.path) if self.document is not None else None
        return rec


class DocumentArchive:
    """See the module docstring for the invariants."""

    def __init__(
        self,
        root: Path | str,
        policy: NetworkPolicy,
        client: httpx.Client | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self._client = client
        self._owns_client = client is None
        #: normalised URL -> the outcome of this run's one attempt
        self.attempts: dict[str, FetchOutcome] = {}
        self.log_path = self.root / "fetch_log.jsonl"

    # ------------------------------------------------------------ lifecycle

    @property
    def online(self) -> bool:
        """Whether fetches may be attempted at all (the user opened an online session)."""
        return self.policy.approved

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(follow_redirects=False, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> DocumentArchive:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ fetching

    def fetch(self, url: str, purpose: str, expect: Expect = "any") -> FetchOutcome:
        """Fetch ``url`` once (see the module docstring); never raises for a network or content problem, only for a missing approval.

        ``expect="pdf"`` requires a PDF (by its bytes, whatever the content
        type says) - anything else is ``blocked`` as an interstitial or viewer
        shell; ``"html"`` and ``"any"`` accept any document kind (PDF, HTML,
        XML, text) and differ only in the ``Accept`` header sent. ``purpose``
        is free text for the log and the meta (``"datasheet of U1"``).
        """
        started = time.monotonic()
        try:
            norm, note = normalise_url(url)
        except ValueError as e:
            return self._done(FetchOutcome(status="refused", url=url, reason=f"unusable URL: {e}", purpose=purpose, expect=expect), started)
        prior = self.attempts.get(norm)
        if prior is not None:
            log.info("archive: %s already attempted in this run (%s); not fetching again", norm, prior.status)
            return prior
        self.policy.require_online()  # raises before any socket use
        origin, why = self.policy.trusted_origin(norm)
        if origin is None:
            return self._done(FetchOutcome(status="refused", url=url, normalised_url=norm, reason=why, purpose=purpose, expect=expect, url_note=note), started)
        outcome = self._fetch_trusted(url, norm, origin, purpose, expect)
        outcome.trusted_by = why
        outcome.url_note = note
        return self._done(outcome, started)

    def _done(self, outcome: FetchOutcome, started: float) -> FetchOutcome:
        outcome.elapsed_s = round(time.monotonic() - started, 3)
        if outcome.normalised_url:
            self.attempts[outcome.normalised_url] = outcome
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(outcome.log_record(), ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as e:  # pragma: no cover - the log is best effort
            log.warning("archive: could not append to %s: %s", self.log_path, e)
        log.info("archive: %s -> %s%s", outcome.url, outcome.status, f" ({outcome.reason})" if outcome.reason else "")
        return outcome

    def _headers(self, expect: str) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": _ACCEPT.get(expect, _ACCEPT["any"])}

    def _fetch_trusted(self, url: str, norm: str, origin: str, purpose: str, expect: str) -> FetchOutcome:
        import httpx

        client = self._http()
        redirects: list[dict[str, Any]] = []
        visited = {norm}
        current = norm
        for _hop in range(MAX_REDIRECTS + 1):
            request = client.build_request("GET", current, headers=self._headers(expect), timeout=self.timeout)
            try:
                response = client.send(request, stream=True)
            except httpx.TimeoutException as e:
                return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, purpose=purpose, expect=expect,
                                    reason=f"timed out after {self.timeout:g} s ({type(e).__name__})")
            except httpx.HTTPError as e:
                return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, purpose=purpose, expect=expect,
                                    reason=f"transport error: {type(e).__name__}: {e}")
            try:
                status = response.status_code
                location = response.headers.get("location")
                if 300 <= status < 400 and location:
                    target = urljoin(current, location)
                    redirects.append({"status": status, "url": current, "location": location, "target": target})
                    try:
                        target_norm, _ = normalise_url(target)
                    except ValueError as e:
                        return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, http_status=status,
                                            purpose=purpose, expect=expect, reason=f"unusable redirect target {target!r}: {e}")
                    refusal = self.policy.redirect_allowed(origin, target_norm)
                    if refusal is not None:
                        return FetchOutcome(status="refused", url=url, normalised_url=norm, final_url=current, redirects=redirects, http_status=status,
                                            purpose=purpose, expect=expect, reason=refusal)
                    if target_norm in visited:
                        return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, http_status=status,
                                            purpose=purpose, expect=expect, reason=f"redirect loop back to {target_norm}")
                    visited.add(target_norm)
                    current = target_norm
                    continue
                try:
                    body, truncated = self._read_body(response)
                except httpx.TimeoutException as e:
                    return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, http_status=status,
                                        purpose=purpose, expect=expect, reason=f"timed out reading the body after {self.timeout:g} s ({type(e).__name__})")
                except httpx.HTTPError as e:
                    return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, http_status=status,
                                        purpose=purpose, expect=expect, reason=f"transport error while reading: {type(e).__name__}: {e}")
                content_type = response.headers.get("content-type")
            finally:
                response.close()
            return self._classify(url, norm, current, redirects, status, content_type, body, truncated, purpose, expect)
        return FetchOutcome(status="error", url=url, normalised_url=norm, final_url=current, redirects=redirects, purpose=purpose, expect=expect,
                            reason=f"more than {MAX_REDIRECTS} redirects")

    @staticmethod
    def _read_body(response: Any) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return b"".join(chunks), True
        return b"".join(chunks), False

    def _classify(
        self, url: str, norm: str, final_url: str, redirects: list[dict[str, Any]], status: int, content_type: str | None,
        body: bytes, truncated: bool, purpose: str, expect: str,
    ) -> FetchOutcome:
        kind = sniff_kind(body, content_type, final_url)
        base = dict(url=url, normalised_url=norm, final_url=final_url, redirects=redirects, http_status=status, content_type=content_type,
                    size=len(body), body_sha256=sha256_of(body), purpose=purpose, expect=expect)
        # extract once: text kinds always (the head of a 404 page is evidence too), a PDF only when it may be archived
        extracted: ExtractedText | None = None
        if kind != "binary" and not truncated and (status == 200 or kind != "pdf"):
            extracted = extract_document(body, kind, content_type)
        first_page = extracted.pages[0] if extracted is not None and extracted.pages else ""
        title = extracted.title if extracted is not None else None
        hidden = extracted.hidden if extracted is not None else None
        text_head = " ".join(first_page.split())[:200] or None
        if status in MISSING_HTTP_STATUSES:
            return FetchOutcome(status="missing", reason=f"HTTP {status}: the document is not at this URL", text_head=text_head, **base)
        if status in BLOCKED_HTTP_STATUSES:
            marker = find_block_marker(title, text_head or "", hidden)
            why = f" ({marker!r} page)" if marker else ""
            return FetchOutcome(status="blocked", reason=f"HTTP {status}: access denied, rate limited or bot protection{why}", text_head=text_head, **base)
        if status != 200:
            return FetchOutcome(status="error", reason=f"HTTP {status}", text_head=text_head, **base)
        if truncated:
            return FetchOutcome(status="error", reason=f"body exceeds {MAX_BODY_BYTES} bytes; not read to the end", **base)
        if len(body) < MIN_BODY_BYTES:
            return FetchOutcome(status="blocked", reason=f"body is {len(body)} bytes: not a document", text_head=text_head, **base)
        if expect == "pdf" and kind != "pdf":
            if kind == "binary":
                # a distinct reason: these bytes are simply not a PDF (no %PDF- header in the first 1024 bytes) - not a bot wall
                return FetchOutcome(status="blocked", reason=f"expected a PDF but received binary content ({content_type or 'no content type'}): not a PDF by its bytes "
                                                             f"(no %PDF- header within the first {PDF_HEADER_WINDOW} bytes)", text_head=text_head, **base)
            marker = find_block_marker(title, text_head or "", hidden)
            why = f"; {marker!r} page" if marker else ""
            return FetchOutcome(status="blocked", reason=f"expected a PDF but received {kind} ({content_type or 'no content type'}): interstitial or viewer shell{why}",
                                text_head=text_head, **base)
        if kind == "binary" or extracted is None:
            return FetchOutcome(status="error", reason=f"unsupported content ({content_type or 'no content type'}, not a PDF, HTML, XML or text document)", **base)
        if kind == "html":
            marker = find_block_marker(title, first_page, hidden)
            if marker is not None:
                where = "noscript text" if marker not in ((title or "") + "\n" + first_page[:BLOCK_SCAN_CHARS]).lower() else "title or text"
                return FetchOutcome(status="blocked", reason=f"interstitial / bot-protection page ({marker!r} in {where})", text_head=text_head, **base)
            if len(first_page) < MIN_HTML_TEXT_CHARS:
                return FetchOutcome(status="blocked", reason=f"HTML with {len(first_page)} characters of readable text: script-rendered shell or interstitial, not the document",
                                    text_head=text_head, **base)
        meta = {
            "source": "network", "url": url, "normalised_url": norm, "final_url": final_url, "redirects": redirects, "status": "ok",
            "http_status": status, "content_type": content_type, "purpose": purpose, "expect": expect, "user_agent": self.user_agent,
            "retrieved_at": _now_iso(),
        }
        doc = self._store(body, kind, extracted, meta)
        reason = f"text extraction failed: {extracted.error}" if extracted.error and not extracted.text_available else (
            f"partial text extraction: {extracted.error}" if extracted.error else None)
        return FetchOutcome(status="ok", document=doc, reason=reason, text_head=text_head, **base)

    # ------------------------------------------------------------ storage

    def _store(self, body: bytes, kind: str, extracted: ExtractedText, meta: dict[str, Any]) -> ArchivedDocument:
        sha = sha256_of(body)
        hexd = sha256_hex(sha)
        ext = extension_for(kind)
        path = self.root / f"{hexd}.{ext}"
        if path.exists() and sha256_of(path.read_bytes()) != sha:
            log.warning("archive: %s did not hash to its name; overwriting with the verified bytes", path)
        if not path.exists() or sha256_of(path.read_bytes()) != sha:
            path.write_bytes(body)
        meta_path = self.root / f"{hexd}.meta.json"
        history: list[dict[str, Any]] = []
        old: dict[str, Any] | None = None
        if meta_path.exists():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                history = list(old.get("history") or [])
            except (OSError, ValueError):
                old, history = None, []
        pages_hash = text_hash(extracted.pages)
        event_keys = ("source", "url", "final_url", "retrieved_at", "purpose", "original_path", "title", "authority", "archived_at")
        if old is not None and old.get("source") == "network" and old.get("status") == "ok" and meta.get("source") == "user_file":
            # the same bytes were fetched from the network earlier: that provenance (URL, redirects, server time) is the stronger
            # one and is kept; the user's file is recorded as an event, never as a downgrade of the source
            history.append({k: meta.get(k) for k in event_keys if meta.get(k) is not None} | {"note": "same bytes supplied as a user file; network provenance kept"})
            full = dict(old)
            full["history"] = history
            if not full.get("title") and meta.get("title"):
                full["title"] = meta["title"]
            if not full.get("authority") and meta.get("authority"):
                full["authority"] = meta["authority"]
            meta_path.write_text(json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            return ArchivedDocument(sha256=sha, path=path, meta=full, pages=list(extracted.pages), text_sha256=pages_hash, extraction_error=extracted.error,
                                    extractor=extracted.extractor, extractor_version=extracted.extractor_version, library_version=extracted.library_version)
        if old is not None:
            history.append({k: old.get(k) for k in ("source", "url", "final_url", "retrieved_at", "purpose", "original_path") if old.get(k) is not None})
        full = {
            "schema": META_SCHEMA, "sha256": sha, "ext": ext, "kind": kind, "size": len(body),
            "extractor": extracted.extractor, "extractor_version": extracted.extractor_version, "library_version": extracted.library_version,
            "encoding": extracted.encoding, "pages": len(extracted.pages), "text_sha256": pages_hash, "extraction_error": extracted.error,
            "history": history,
        }
        full.update(meta)
        if not full.get("title"):
            full["title"] = extracted.title  # the HTML <title> when there is one, else null (the key is always present)
        meta_path.write_text(json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return ArchivedDocument(sha256=sha, path=path, meta=full, pages=list(extracted.pages), text_sha256=pages_hash, extraction_error=extracted.error,
                                extractor=extracted.extractor, extractor_version=extracted.extractor_version, library_version=extracted.library_version)

    def add_file(self, path: Path | str, title: str, retrieved_at: datetime | str, authority: str | None = None) -> ArchivedDocument:
        """Archive a user-supplied file (a datasheet the user downloaded, an official text on disk).

        ``retrieved_at`` is the user's own declaration of when the document was
        obtained; the archive records it as such (``source: "user_file"``) and
        never invents one. A file that is not a PDF, HTML, XML or text document
        raises :class:`ArchiveError`.
        """
        p = Path(path)
        try:
            body = p.read_bytes()
        except OSError as e:
            raise ArchiveError(f"cannot read {p}: {e}") from e
        kind = sniff_kind(body, None, p.name)
        if kind == "binary":
            raise ArchiveError(f"{p} is not a PDF, HTML, XML or text document (nothing to ground a quote in)")
        when = retrieved_at.isoformat() if isinstance(retrieved_at, datetime) else str(retrieved_at)
        try:
            datetime.fromisoformat(when)
        except ValueError as e:
            raise ArchiveError(f"retrieved_at must be an ISO 8601 date/time, got {retrieved_at!r}") from e
        extracted = extract_document(body, kind, None)
        meta = {
            "source": "user_file", "url": None, "normalised_url": None, "final_url": None, "redirects": [], "status": "ok", "http_status": None,
            "content_type": None, "purpose": "user-supplied file", "expect": "any", "title": title, "authority": authority,
            "original_path": str(p.resolve()), "retrieved_at": when, "archived_at": _now_iso(),
        }
        return self._store(body, kind, extracted, meta)

    def load(self, sha256: str) -> ArchivedDocument:
        """The archived document with that hash, re-hashed from disk and re-extracted with the current extractors.

        Raises :class:`DocumentMissingError` when there is no such file and
        :class:`TamperedDocumentError` when the file's bytes no longer hash to
        its name. ``doc.text_matches_meta`` says whether the fresh extraction
        equals the one recorded at archive time.
        """
        hexd = sha256_hex(sha256)
        meta_path = self.root / f"{hexd}.meta.json"
        if not meta_path.exists():
            raise DocumentMissingError(f"no archived document {hexd[:12]}… under {self.root}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise DocumentMissingError(f"{meta_path} is unreadable: {e}") from e
        ext = meta.get("ext") or extension_for(str(meta.get("kind", "binary")))
        path = self.root / f"{hexd}.{ext}"
        if not path.exists():
            raise DocumentMissingError(f"{path} is missing (meta present)")
        body = path.read_bytes()
        actual = sha256_of(body)
        if sha256_hex(actual) != hexd:
            raise TamperedDocumentError(f"{path} hashes to {actual}, not to its name sha256:{hexd}: the archived document was altered")
        kind = str(meta.get("kind") or sniff_kind(body, meta.get("content_type"), path.name))
        extracted = extract_document(body, kind, meta.get("content_type"))
        return ArchivedDocument(sha256=actual, path=path, meta=meta, pages=list(extracted.pages), text_sha256=text_hash(extracted.pages),
                                extraction_error=extracted.error, extractor=extracted.extractor, extractor_version=extracted.extractor_version,
                                library_version=extracted.library_version)

    def verify(self, ref: SourceRef) -> VerifyStatus:
        """``ok`` / ``missing`` / ``tampered`` for a :class:`~ai_eda.ir.SourceRef`, ``unarchived`` when it carries no content hash."""
        if not ref.content_hash:
            return "unarchived"
        try:
            self.load(ref.content_hash)
        except TamperedDocumentError:
            return "tampered"
        except (DocumentMissingError, ValueError):
            return "missing"
        return "ok"

    def lookup(self, url: str) -> ArchivedDocument | None:
        """The most recently archived, hash-verified document fetched from ``url`` (any run), else ``None``."""
        try:
            norm, _ = normalise_url(url)
        except ValueError:
            return None
        best: tuple[str, str] | None = None
        for meta_path in self.root.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict) or meta.get("normalised_url") != norm or meta.get("status") != "ok":
                continue
            # the hash is the file's name; a meta whose own ``sha256`` names another document is corrupt or
            # planted and answers for nothing (it would otherwise redirect a URL lookup to a different file)
            named = meta_path.name[: -len(".meta.json")]
            try:
                hexd = sha256_hex(named)
            except ValueError:
                continue
            claimed = meta.get("sha256")
            if claimed is not None:
                try:
                    if sha256_hex(str(claimed)) != hexd:
                        continue
                except ValueError:
                    continue
            when = str(meta.get("retrieved_at") or "")
            if best is None or when > best[0]:
                best = (when, f"sha256:{hexd}")
        if best is None:
            return None
        try:
            return self.load(best[1])
        except (ArchiveError, ValueError):
            return None

    def documents(self) -> list[str]:
        """Hashes (``sha256:<hex>``) of every archived document, sorted."""
        return sorted(f"sha256:{p.name[:-len('.meta.json')]}" for p in self.root.glob("*.meta.json"))

    def copy_into(self, other_root: Path | str) -> None:
        """Copy every archived file and meta into another archive root (a project workdir), keeping names."""
        dst = Path(other_root)
        dst.mkdir(parents=True, exist_ok=True)
        for p in self.root.iterdir():
            if p.is_file() and p.name != self.log_path.name:
                shutil.copyfile(p, dst / p.name)


__all__ = [
    "BLOCKED_HTTP_STATUSES",
    "BLOCK_MARKERS",
    "DEFAULT_TIMEOUT_S",
    "MAX_BODY_BYTES",
    "MAX_REDIRECTS",
    "META_SCHEMA",
    "MIN_BODY_BYTES",
    "MIN_HTML_TEXT_CHARS",
    "MISSING_HTTP_STATUSES",
    "QUOTE_CONTEXT_CHARS",
    "USER_AGENT",
    "ArchiveError",
    "ArchivedDocument",
    "DocumentArchive",
    "DocumentMissingError",
    "Expect",
    "FetchOutcome",
    "FetchStatus",
    "QuoteHit",
    "TamperedDocumentError",
    "VerifyStatus",
    "find_block_marker",
    "sha256_hex",
    "sha256_of",
    "text_hash",
]
