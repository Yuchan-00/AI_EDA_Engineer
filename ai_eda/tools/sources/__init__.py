"""External documents: fetched only with the user's approval, only from trusted origins, archived by content hash.

- :mod:`ai_eda.tools.sources.policy` - :class:`NetworkPolicy`: the user's
  online-session approval (``NETWORK_FETCH``), the trusted-origin rules
  (KiCad datasheet hosts, official domains, user-supplied URLs), https only.
- :mod:`ai_eda.tools.sources.archive` - :class:`DocumentArchive`: the one
  place a socket is opened; stores ``<sha256>.<ext>`` + ``<sha256>.meta.json``,
  classifies bot-protection pages and 404s honestly, re-verifies hashes on
  load, and grounds quotes with the requirement stage's normalisation.
- :mod:`ai_eda.tools.sources.extract` - deterministic, versioned text
  extraction (pypdf per page, html.parser, xml.etree, plain text).

Nothing here decides what a document *means*; reading a regulation is not
compliance and finding an MPN in a datasheet is not a design verdict.
"""

from ai_eda.tools.sources.archive import (
    USER_AGENT,
    ArchiveError,
    ArchivedDocument,
    DocumentArchive,
    DocumentMissingError,
    FetchOutcome,
    QuoteHit,
    TamperedDocumentError,
    sha256_hex,
    sha256_of,
)
from ai_eda.tools.sources.extract import ExtractedText, extract_document, sniff_kind, strip_html
from ai_eda.tools.sources.policy import ONLINE_SESSION_DETAIL, NetworkPolicy, host_key, host_of, normalise_url

__all__ = [
    "ONLINE_SESSION_DETAIL",
    "USER_AGENT",
    "ArchiveError",
    "ArchivedDocument",
    "DocumentArchive",
    "DocumentMissingError",
    "ExtractedText",
    "FetchOutcome",
    "NetworkPolicy",
    "QuoteHit",
    "TamperedDocumentError",
    "extract_document",
    "host_key",
    "host_of",
    "normalise_url",
    "sha256_hex",
    "sha256_of",
    "sniff_kind",
    "strip_html",
]
