"""Identity grounding: what makes a component's MPN count as verified, checked the same way everywhere.

Invariant this module enforces: an MPN is *grounded* only when its
``authoritative`` provenance names an archived datasheet (``document_path`` +
``content_hash`` + ``retrieved_at``) whose bytes still hash to that name
**and whose archived text still contains the MPN** at the recorded section.
The IR_BUILD validator (``ir.component_provenance``), the independent
reviewer (``review.component_provenance``) and the CLI all go through
:func:`mpn_grounding`, so a value that is merely *tagged* authoritative - a
fixture, a hand edit, a record whose file is gone - is downgraded to
``NOT_VERIFIED`` with the reason, a file altered after grounding is
``tampered``, and a tag that points at an archived document which does not
say the MPN (an edited ``ir.json``) is ``not in text``.

What counts as an archived copy is decided by the archive, not by any file
on disk: the recorded ``document_path`` must be an entry of a document
archive - the run's archive (``ctx.tools["archive"]``), the workdir's
``<workdir>/sources`` directory, or the directory the path itself lies in
when that directory holds the entry's ``<sha256>.meta.json`` (a run with
``--sources-dir`` elsewhere). A file somewhere else that happens to hash to
the recorded hash is *not* an archived copy: nothing proves where it came
from. The archive re-extracts the text with the current extractor; when
that text no longer hashes to what ``meta.json`` recorded (an extractor
upgrade), the grounding is ``re-extracted`` - NOT_VERIFIED until the
existence check re-grounds it. Nothing here fetches anything.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel

from ai_eda.ir import Component, ProvenanceKind, SourceRef, ValidationResult, ValidationStatus
from ai_eda.parts.existence import CHECK_PREFIX as EXISTENCE_PREFIX
from ai_eda.parts.existence import find_mpn
from ai_eda.security.approval import ApprovalGate
from ai_eda.tools.sources.archive import ArchivedDocument, ArchiveError, DocumentArchive, sha256_hex
from ai_eda.tools.sources.policy import NetworkPolicy

SourceStatus = Literal["ok", "unarchived", "missing", "tampered"]

#: the sub-directory of a project workdir where the CLI's document archive lives
SOURCES_DIRNAME = "sources"

_PAGE_RE = re.compile(r"page\s+(\d+)")


def _read_only(root: Path) -> DocumentArchive:
    return DocumentArchive(root, NetworkPolicy(approved=False, gate=ApprovalGate()))


def open_archive(tools: Mapping[str, Any] | None, workdir: Path | str | None) -> DocumentArchive | None:
    """The run's archive (``tools["archive"]``), else the workdir's ``sources`` directory opened read-only, else ``None``.

    A read-only archive carries an unapproved policy on a private gate: it can
    re-hash and re-extract what earlier runs archived, and it can never open a
    socket.
    """
    given = (tools or {}).get("archive")
    if isinstance(given, DocumentArchive):
        return given
    if workdir is None:
        return None
    root = Path(workdir) / SOURCES_DIRNAME
    if root.is_dir():
        return _read_only(root)
    return None


def _holding_archive(ref: SourceRef, hexd: str, archive: DocumentArchive | None) -> DocumentArchive | None:
    """The archive that holds the entry ``ref`` names: the given one when its meta is there, else the directory of the recorded path when *it* is an archive entry."""
    if archive is not None and (archive.root / f"{hexd}.meta.json").is_file():
        return archive
    if ref.document_path:
        p = Path(ref.document_path)
        if p.parent.is_dir() and (p.parent / f"{hexd}.meta.json").is_file():
            return _read_only(p.parent)
    return None


def locate_source(ref: SourceRef | None, archive: DocumentArchive | None = None) -> tuple[SourceStatus, str, ArchivedDocument | None]:
    """``(status, reason, document)`` for the archived copy a :class:`~ai_eda.ir.SourceRef` claims to name (module docstring).

    ``unarchived`` when the reference names no archived copy (no
    ``content_hash`` / ``document_path`` / ``retrieved_at``) or its path is
    not an archive entry, ``missing`` when the archive does not hold the
    entry, ``tampered`` when the file no longer hashes to the recorded hash,
    ``ok`` with the loaded (re-hashed, re-extracted) document otherwise.
    """
    if ref is None:
        return "unarchived", "no source reference", None
    if not ref.content_hash:
        return "unarchived", "the source names no archived copy (no content hash)", None
    if not ref.document_path:
        return "unarchived", f"the source names a hash ({ref.content_hash[:19]}…) but no archived file (no document path)", None
    if ref.retrieved_at is None:
        return "unarchived", "the source records no retrieval time", None
    try:
        hexd = sha256_hex(ref.content_hash)
    except ValueError:
        return "unarchived", f"content hash {ref.content_hash!r} is not a sha256", None
    holder = _holding_archive(ref, hexd, archive)
    if holder is None:
        where = f" (not in the archive at {archive.root})" if archive is not None else ""
        if Path(ref.document_path).is_file():
            return "unarchived", (f"{ref.document_path} is not an entry of a document archive (no {hexd[:12]}….meta.json beside it){where}: "
                                  "a file outside the archive proves nothing about where its bytes came from"), None
        return "missing", f"archived copy {ref.document_path} is not on disk{where}", None
    verdict = holder.verify(ref)
    if verdict == "tampered":
        return "tampered", f"tampered: {ref.document_path} no longer hashes to {ref.content_hash} (altered after it was archived)", None
    if verdict != "ok":
        return "missing", f"archived copy {ref.content_hash} ({verdict}) under {holder.root}", None
    try:
        doc = holder.load(ref.content_hash)
    except ArchiveError as e:  # pragma: no cover - verify() just loaded it
        return "missing", f"archived copy unreadable: {e}", None
    return "ok", f"archived copy {ref.content_hash} re-hashed and verified ({ref.document_path})", doc


def verify_source(ref: SourceRef | None, archive: DocumentArchive | None = None) -> tuple[SourceStatus, str]:
    """``(status, reason)`` of :func:`locate_source` - for callers that need no document."""
    status, reason, _doc = locate_source(ref, archive)
    return status, reason


class MpnGrounding(BaseModel):
    """Whether a component's MPN rests on an archived, hash-verified datasheet that states it, and why not otherwise."""

    ref: str
    mpn: str | None = None
    status: ValidationStatus
    #: ``missing`` (no MPN), the provenance kind of a non-authoritative MPN, or ``authoritative, <SourceStatus | re-extracted | not in text | ok>``
    label: str
    reason: str
    content_hash: str | None = None
    document_path: str | None = None
    section: str | None = None
    #: the page the MPN was re-located on at check time
    page: int | None = None
    #: extractor stamp of the text the MPN was re-located in
    extraction: str | None = None

    @property
    def grounded(self) -> bool:
        return self.status is ValidationStatus.PASS


def mpn_grounding(component: Component, archive: DocumentArchive | None = None) -> MpnGrounding:
    """The one rule for an authoritative identity: an ``authoritative`` MPN whose SourceRef names an archived, hash-verified datasheet that states it."""
    ref = component.ref
    if component.mpn is None or component.mpn.value in (None, ""):
        return MpnGrounding(ref=ref, status=ValidationStatus.NOT_VERIFIED, label="missing", reason="no MPN: nobody can source this part")
    mpn = str(component.mpn.value)
    prov = component.mpn.provenance
    if prov.kind is not ProvenanceKind.AUTHORITATIVE:
        why = {
            ProvenanceKind.LLM_GENERATED: "proposed by a model and not yet found in an archived datasheet",
            ProvenanceKind.ASSUMPTION: "an assumption, not backed by a datasheet",
            ProvenanceKind.USER_REQUIREMENT: "typed by the user; its existence has not been confirmed in an archived datasheet",
            ProvenanceKind.DERIVED: "derived, not backed by a datasheet",
        }[prov.kind]
        return MpnGrounding(ref=ref, mpn=mpn, status=ValidationStatus.NOT_VERIFIED, label=str(prov.kind), reason=f"MPN {mpn!r} is {why}")
    src = prov.source
    verdict, reason, doc = locate_source(src, archive)
    base = dict(ref=ref, mpn=mpn, content_hash=src.content_hash if src is not None else None,
                document_path=src.document_path if src is not None else None, section=src.section if src is not None else None)
    if verdict != "ok" or doc is None:
        return MpnGrounding(status=ValidationStatus.NOT_VERIFIED, label=f"authoritative, {verdict}", reason=f"MPN {mpn!r} is tagged authoritative but not grounded: {reason}", **base)
    assert src is not None
    if doc.kind != "pdf":
        return MpnGrounding(status=ValidationStatus.NOT_VERIFIED, label="authoritative, not a datasheet", extraction=doc.extraction_stamp,
                            reason=f"MPN {mpn!r} is tagged from an archived {doc.kind} document ({doc.title or doc.path.name}), not a datasheet: a web page proves nothing about the part", **base)
    if not doc.text_matches_meta:
        return MpnGrounding(status=ValidationStatus.NOT_VERIFIED, label="authoritative, re-extracted", extraction=doc.extraction_stamp,
                            reason=(f"MPN {mpn!r}: the archived bytes still hash to {src.content_hash}, but the current {doc.extraction_stamp} differs from the text "
                                    f"recorded at archive time ({doc.meta.get('extractor')} {doc.meta.get('extractor_version')}, text {doc.meta.get('text_sha256')}); "
                                    "re-run the existence check to re-ground it"), **base)
    if not doc.text_available:
        return MpnGrounding(status=ValidationStatus.NOT_VERIFIED, label="authoritative, no text", extraction=doc.extraction_stamp,
                            reason=f"MPN {mpn!r} is tagged from {src.content_hash} whose text can not be extracted ({doc.extraction_error or 'no text'}); nothing can be re-located in it", **base)
    m = _PAGE_RE.search(src.section or "")
    page = int(m.group(1)) if m else None
    hits = find_mpn(doc, mpn)
    if page is not None:
        hits = [h for h in hits if h.page == page]
    if not hits:
        where = f"on {src.section}" if page is not None else "anywhere"
        return MpnGrounding(status=ValidationStatus.NOT_VERIFIED, label="authoritative, not in text", extraction=doc.extraction_stamp,
                            reason=f"MPN {mpn!r} is tagged from {src.content_hash} but is not found verbatim {where} of the archived text at check time (the tag does not match the document)", **base)
    return MpnGrounding(
        status=ValidationStatus.PASS, label="authoritative, ok", page=hits[0].page, extraction=doc.extraction_stamp,
        reason=f"MPN {mpn!r} grounded in the archived datasheet {src.content_hash} (re-located on page {hits[0].page}); {reason}; {doc.extraction_stamp}", **base,
    )


def latest_existence(ir: Any, ref: str) -> ValidationResult | None:
    """The newest ``component.existence.<ref>`` result in the IR, if the component stage produced one."""
    return ir.validation.latest(f"{EXISTENCE_PREFIX}{ref}")


def existence_conflict(result: ValidationResult | None, grounding: MpnGrounding) -> str | None:
    """Why the latest existence check disagrees with a grounded MPN (``None`` when it corroborates it or is absent).

    The check is corroborating when it read the very document the MPN's
    SourceRef names (``artifact_hash``), looked for the same MPN and found it
    (``mpn_in_datasheet`` PASS). A check that read another document or did
    not find the MPN outranks the IR's tag: the tag is then stale.
    """
    if result is None or not grounding.grounded:
        return None
    if result.status is ValidationStatus.FAIL:
        return f"the latest existence check is FAIL: {result.message}"
    checked_mpn = result.details.get("mpn")
    if checked_mpn is not None and str(checked_mpn) != grounding.mpn:
        return f"the latest existence check looked for MPN {checked_mpn!r}, the IR now says {grounding.mpn!r}"
    if result.artifact_hash and grounding.content_hash and result.artifact_hash != grounding.content_hash:
        return f"the latest existence check read datasheet {result.artifact_hash[:19]}…, the MPN is tagged from {grounding.content_hash[:19]}…"
    for check in result.details.get("checks", []):
        if check.get("name") == "mpn_in_datasheet" and check.get("status") != str(ValidationStatus.PASS):
            return f"the latest existence check did not find the MPN in the datasheet: {check.get('message')}"
    return None


__all__ = ["SOURCES_DIRNAME", "MpnGrounding", "SourceStatus", "existence_conflict", "latest_existence", "locate_source", "mpn_grounding", "open_archive", "verify_source"]
