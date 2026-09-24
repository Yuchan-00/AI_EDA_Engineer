"""Fab capability agent (FAB_CAPABILITY stage) and manufacturing agent (MANUFACTURABILITY stage).

Invariants:

* :class:`FabCapabilityAgent` decides nothing about the design. It obtains
  the vendor page the user's capability file names through the
  :class:`~ai_eda.tools.sources.DocumentArchive` (an online session fetches
  the exact user URL the session registered under ``fab_capability``; an
  offline one reuses the archived copy of that URL; a saved page is added as
  a user file with the user's ``retrieved_at``), grounds every limit
  verbatim on it (:func:`~ai_eda.tools.manufacturing.ground_capability`)
  and reports ``mfg.capability_source`` (tool-backed, evidence = the
  archived page; PASS only when every limit grounded, never FAIL). It then
  proposes **one** ``set pcb.manufacturing`` whose constraints are the
  existing ones *merged* with the grounded ones: keys the file names
  override; ``authoritative`` fields from a *different* page (their
  SourceRef hash differs) are dropped and said so in a note; every other
  existing field (a ``user_requirement`` board thickness) is preserved.
  "Differs" is a design-view comparison (:func:`~ai_eda.ir.provenance.design_data`),
  so a run that re-grounds identical limits proposes nothing and the hash
  stays. Without ``ir.pcb`` the limits are grounded but not recorded
  (NOT_VERIFIED, "nothing to lay out"); rejected limits never enter the IR.
* Without a file, the agent re-verifies the ``authoritative`` limits the IR
  already holds against the archive (page re-hashed, quote re-located, the
  reviewer's own check) and emits a fresh ``mfg.capability_source``, so a
  later run without ``--fab-capability`` re-produces its evidence instead
  of carrying an old PASS; with nothing to verify it notes the missing file.
* The stage sits after PLACEMENT (which may create ``ir.pcb``) and before
  IR_BUILD: the limits are design content, so they must be in the IR before
  any validator hash, project file, board or DRC is produced.
* :class:`ManufacturingAgent` proposes nothing and asks nothing: its
  validation is exactly ``[check_capability(ir)]``
  (:mod:`ai_eda.tools.manufacturing.capability`), stamped with the IR hash
  here because it is a deterministic verdict about the IR as it stands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.ir import CircuitIR, Evidence, ManufacturingConstraints, ProvenanceKind, ValidationResult, ValidationStatus
from ai_eda.ir.provenance import design_data
from ai_eda.llm.router import TaskKind
from ai_eda.tools.manufacturing import check_capability
from ai_eda.tools.manufacturing.capability_file import (
    CAPABILITY_FILE_VERSION,
    CAPABILITY_KEYS,
    GROUNDING_TOOL,
    SOURCE_CHECK_ID,
    FabCapabilityFile,
    GroundedCapability,
    capability_source_result,
    ground_capability,
    relocate_limits,
)
from ai_eda.tools.sources import ArchiveError, ArchivedDocument, DocumentArchive

#: the user-URL key the session registers the capability page under (exact-URL trust)
FAB_CAPABILITY_KEY = "fab_capability"
NO_FILE_NOTE = "no fab capability file (pass --fab-capability FILE to ground the fab limits on the vendor page)"


def obtain_capability_page(file: FabCapabilityFile, archive: DocumentArchive | None) -> tuple[ArchivedDocument | None, str]:
    """``(document, note)`` for the page the file names, through the archive only; ``(None, reason)`` when it can not be had.

    ``source.url``: fetched (``expect="html"``) when the session is online,
    else the most recent archived copy of exactly that URL; ``source.file``:
    archived as a user file with the user's ``retrieved_at`` (relative to the
    capability file's directory). Nothing else opens a socket or reads a
    page.
    """
    if archive is None:
        return None, "no document archive in this session: the vendor page can not be obtained"
    src = file.source
    if src.file:
        path = Path(src.file)
        if not path.is_absolute():
            path = file.path.parent / path
        try:
            doc = archive.add_file(path, title=src.title or path.name, retrieved_at=str(src.retrieved_at), authority=src.authority)
        except ArchiveError as e:
            return None, f"capability page file not usable: {e}"
        return doc, f"vendor page taken from the user's file {path.name} (retrieved {src.retrieved_at})"
    url = str(src.url)
    if archive.online:
        outcome = archive.fetch(url, purpose=f"fab capability page of {file.fab}", expect="html")
        if outcome.ok and outcome.document is not None:
            return outcome.document, f"vendor page fetched: {outcome.final_url or url}"
        return None, f"vendor page not fetched ({outcome.status}): {outcome.reason}"
    doc = archive.lookup(url)
    if doc is None:
        return None, f"not fetched (offline) and no archived copy of {url}; run once with --online"
    return doc, f"vendor page reused from the archive (offline): {doc.sha256} retrieved {doc.meta.get('retrieved_at')}"


def merge_constraints(existing: ManufacturingConstraints, grounded: GroundedCapability) -> tuple[ManufacturingConstraints, list[str]]:
    """The merge policy of the module docstring: ``(merged constraints, notes about dropped fields)``."""
    data: dict[str, Any] = {}
    notes: list[str] = []
    for key in sorted(CAPABILITY_KEYS):
        new = grounded.accepted.get(key)
        if new is not None:
            data[key] = new
            continue
        old = getattr(existing, key)
        if old is None:
            continue
        src = old.provenance.source
        if old.provenance.kind is ProvenanceKind.AUTHORITATIVE and (src is None or src.content_hash != grounded.document):
            notes.append(f"dropped {key}: authoritative limit from another page ({src.content_hash if src is not None and src.content_hash else 'no archived page'}), not on {grounded.document}")
            continue
        data[key] = old
    if existing.fab and existing.fab != grounded.fab:
        notes.append(f"fab {existing.fab!r} replaced by {grounded.fab!r} (the capability file names the fab)")
    return ManufacturingConstraints(fab=grounded.fab, **data), notes


class FabCapabilityAgent(Agent):
    name = "fab_capability"
    task = TaskKind.RESULT_INTERPRETATION

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        file = ctx.tools.get("fab_capability_file")
        archive = ctx.tools.get("archive")
        archive = archive if isinstance(archive, DocumentArchive) else None
        if not isinstance(file, FabCapabilityFile):
            return self._reverify(ir, archive)
        doc, note = obtain_capability_page(file, archive)
        if doc is None:
            return self._result(validation=[capability_source_result(file, None, None, reason=note)], notes=[note])
        grounded = ground_capability(doc, file)
        notes = [note]
        if ir.pcb is None:
            res = capability_source_result(file, doc, grounded, status=ValidationStatus.NOT_VERIFIED, reason="not recorded: nothing to lay out (ir.pcb is None)")
            return self._result(validation=[res], notes=[*notes, "limits grounded but not recorded: ir.pcb is None"])
        res = capability_source_result(file, doc, grounded)
        merged, merge_notes = merge_constraints(ir.pcb.manufacturing, grounded)
        notes.extend(merge_notes)
        proposals: list[IRProposal] = []
        if design_data(merged) != design_data(ir.pcb.manufacturing):
            proposals.append(IRProposal(
                description=f"fab limits of {file.fab} grounded on the archived vendor page ({', '.join(sorted(grounded.accepted)) or 'none'})",
                target="pcb.manufacturing", operation="set", payload=merged,
                rationale=f"every limit quoted verbatim from {doc.final_url or doc.url or doc.path.name} ({doc.sha256}); rejected: {[k for k, _ in grounded.rejected]}",
            ))
        else:
            notes.append("ir.pcb.manufacturing already holds these limits (design view unchanged): nothing proposed")
        return self._result(proposals=proposals, validation=[res], notes=notes)

    def _reverify(self, ir: CircuitIR, archive: DocumentArchive | None) -> AgentResult:
        """No file: re-verify the authoritative limits the IR holds, or note that there is nothing to verify."""
        if ir.pcb is None:
            return self._result(notes=[NO_FILE_NOTE])
        checks = relocate_limits(ir.pcb.manufacturing, archive)
        if not checks:
            return self._result(notes=[NO_FILE_NOTE])
        ok = [c for c in checks if c.status == "ok"]
        bad = [c for c in checks if c.status != "ok"]
        documents = sorted({c.document for c in ok if c.document})
        evidence: list[Evidence] = []
        for c in ok:
            if c.document and all(e.content_hash != c.document for e in evidence):
                doc = None
                if archive is not None:
                    try:
                        doc = archive.load(c.document)
                    except ArchiveError:
                        doc = None
                evidence.append(Evidence(description=f"archived capability page grounding {c.key}", path=str(doc.path) if doc is not None else None, content_hash=c.document))
        status = ValidationStatus.PASS if not bad else ValidationStatus.NOT_VERIFIED
        message = f"{len(ok)} authoritative limit(s) re-verified against the archived vendor page (no capability file this run), {len(bad)} not"
        if bad:
            message += ": " + "; ".join(f"{c.key}: {c.status}: {c.reason}" for c in bad[:3]) + (" ..." if len(bad) > 3 else "")
        res = ValidationResult(
            check_id=SOURCE_CHECK_ID, status=status, message=message, tool=GROUNDING_TOOL, tool_version=CAPABILITY_FILE_VERSION,
            artifact_hash=documents[0] if len(documents) == 1 else None, evidence=evidence,
            details={"fab": ir.pcb.manufacturing.fab, "reverified": [c.model_dump(mode="json") for c in checks], "documents": documents, "file": None},
        )
        return self._result(validation=[res], notes=["no fab capability file: the IR's authoritative limits were re-verified against the archive"])


class ManufacturingAgent(Agent):
    name = "manufacturing"
    task = TaskKind.RESULT_INTERPRETATION

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        res = check_capability(ir)
        res.ir_hash = ir.content_hash()  # a deterministic verdict about this IR version (the agent proposes nothing)
        return self._result(validation=[res])
