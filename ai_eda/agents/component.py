"""Component Agent: existence checks, datasheet grounding and - with an LLM - candidate parts and datasheet facts the user confirms.

Selection order (from the spec): electrical -> safety -> regulatory ->
environment -> reliability -> manufacturability -> sourcing -> cost. **None
of these criteria is evaluated by this version**: the agent checks that a
part exists (library entry, archived datasheet, MPN in it, catalog row) and
grounds facts, it does not judge whether the part *fits* the design. It says
so with one ``component.fit`` result that is always ``NOT_VERIFIED`` and
names the criteria not evaluated - a model's rationale in the candidate
table is prose for a human, never evidence. The agent never decides part
truth; deterministic tools do, and the user chooses.

Without an LLM (``ctx.llm is None``):

1. Every IR component gets a ``component.existence.<ref>`` result from
   :func:`~ai_eda.parts.existence.examine_component`: symbol and footprint
   parsed from the KiCad libraries (``ctx.tools["kicad_library"]``), the
   datasheet pointer located (user URL, IR reference, KiCad ``Datasheet``
   property), the datasheet archived and hash-verified through
   ``ctx.tools["archive"]`` (a :class:`~ai_eda.tools.sources.DocumentArchive`;
   a fetch happens only when the user opened an online session - offline the
   result says "not fetched (offline)"; an HTML product page is archived but
   is not a datasheet and grounds nothing), the MPN found verbatim in it,
   and the catalog row (``ctx.tools["catalog"]``, a
   :class:`~ai_eda.parts.catalog.CatalogSource`; a row whose manufacturer or
   package differs from the IR's backs no sourcing).
2. Proposals (one ``components`` replacement, applied only by the
   orchestrator): the datasheet :class:`~ai_eda.ir.SourceRef` recorded as the
   archived copy (path, sha256, retrieval time) or as the pointer - its
   ``authority`` is the IR's own datasheet authority, else the manufacturer
   only when that value is itself ``authoritative`` / ``user_requirement``,
   else the pointer's host (a model's manufacturer string never becomes the
   authority of a document); the MPN re-tagged ``authoritative`` with the
   page it was found on when - and only when - the archived datasheet states
   it verbatim (the note carries the extractor stamp); ``SourcingInfo`` from
   the catalog row. The agent never invents an MPN, a package or a value.
3. Facts the user wrote (``--answer datasheet_facts_file=<json>``) are
   grounded by :func:`~ai_eda.parts.datasheet_facts.ground_facts`; accepted
   ones become ``authoritative`` Traced values at once (the user read the
   page and chose the key), rejected ones are listed in
   ``component.facts.<ref>`` with the reason and never enter the IR.

With an LLM (``ctx.llm`` is an :class:`~ai_eda.llm.service.LLMService`), in
addition:

* For components that lack an identity (no MPN) the model may propose
  candidates (manufacturer, MPN, KiCad symbol, KiCad footprint, rationale, a
  datasheet URL that is recorded and **never fetched** - only the KiCad
  library's own Datasheet field, the user's URL or the IR's reference are
  pointers). Each candidate is checked deterministically before it enters:
  the symbol and footprint must exist in the installed libraries and may not
  differ from what the design already names (a footprint change is a design
  change for a human); directive phrases are dropped. Accepted candidates
  enter the IR as ``llm_generated`` identity and the user is shown a table
  under the required question ``confirm_parts`` (the model reply is cached
  in ``<workdir>/parts/candidates.json`` under a hash of the component list
  and the prompt/schema fingerprint, so an unchanged design costs no second
  call). ``--answer confirm_parts=yes`` in a *later* run makes the
  candidates the user's **choice** (``Component.provenance`` becomes
  ``user_requirement``) - **row by row**: only a candidate whose exact row
  (ref, manufacturer, MPN, symbol, footprint) stood in a table shown in an
  earlier run is confirmed; a candidate that only became acceptable since
  (a library entry appeared) goes back into the table and waits. The
  identity facts themselves stay ``llm_generated`` until the existence
  check finds the MPN in the archived datasheet (then ``authoritative``) -
  the user chose a part, the datasheet proves it exists.
  ``confirm_parts=no`` removes the candidates. Until confirmed, candidate
  components are not checked (no fetch on a model's say-so).
* ``--answer extract_datasheet_facts=yes`` asks the model (billed) for facts
  from each archived datasheet. They are grounded exactly like the user's
  (verbatim quote on the page, one whole quantity, unit fits the key,
  package row names the MPN), but grounding can not check the *meaning* a
  model gave a number (that ``0.6 V`` is ``v_max`` and not a dropout
  voltage), so the grounded facts are only **shown** - a table under the
  required question ``confirm_facts`` with key, value, page, quote and
  context - and enter the IR as ``authoritative`` (note: confirmed by user)
  only when the user confirms that very table in a later run
  (``--answer confirm_facts=yes``; ``no`` discards it). Nothing of a model's
  facts is in the IR before that. The reply is cached in
  ``<workdir>/parts/facts.json`` under (ref, datasheet hash, keys asked,
  fingerprint); a cached table is replayed without a call and without the
  flag, so the confirmation costs nothing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.ir import (
    CircuitIR,
    Component,
    LibraryRef,
    MissingInformation,
    Provenance,
    ProvenanceKind,
    ValidationResult,
    ValidationStatus,
    authoritative,
    llm_generated,
)
from ai_eda.llm.client import LLMError
from ai_eda.llm.extraction import _strictify, _table, find_directive, is_confirmation, is_rejection  # shared helpers, reused on purpose
from ai_eda.llm.prompts import DATASHEET_FACT_SYSTEM, PART_CANDIDATE_SYSTEM, part_candidate_messages
from ai_eda.llm.router import TaskKind
from ai_eda.llm.service import BudgetExceededError, LLMService, StructuredOutputError
from ai_eda.parts.catalog import CatalogSource
from ai_eda.parts.datasheet_facts import (
    DEFAULT_FACT_KEYS,
    FACTS_VERSION,
    DatasheetFact,
    DatasheetFacts,
    GroundedFacts,
    apply_facts,
    fact_row,
    facts_json_schema,
    facts_result,
    ground_facts,
    llm_fact_proposals,
    load_facts_file,
)
from ai_eda.parts.existence import CHECK_PREFIX as EXISTENCE_PREFIX
from ai_eda.parts.existence import TOOL as EXISTENCE_TOOL
from ai_eda.parts.existence import ExistenceReport, examine_component
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError
from ai_eda.tools.sources import ArchivedDocument, DocumentArchive

#: answer keys the agent reads (never requirements)
CONFIRM_PARTS_KEY = "confirm_parts"
CONFIRM_FACTS_KEY = "confirm_facts"
FACTS_FILE_KEY = "datasheet_facts_file"
EXTRACT_FACTS_KEY = "extract_datasheet_facts"
CANDIDATES_CHECK = "component.candidates"
FIT_CHECK = "component.fit"
#: model replies for candidate parts, keyed by :func:`candidate_key`, under the workdir
CANDIDATE_CACHE_FILE = Path("parts") / "candidates.json"
#: model replies for datasheet facts, keyed by :func:`facts_key`, under the workdir
FACTS_CACHE_FILE = Path("parts") / "facts.json"
#: bumped when the candidate rules, prompt or schema change
CANDIDATES_VERSION = "0.2"
#: how a candidate identity value's note starts; :func:`lacks_identity` recognises unconfirmed candidates by it
CANDIDATE_NOTE_PREFIX = "candidate proposed by "
#: appended to a candidate value's note when the user confirmed the choice
CONFIRMED_MARK = "; confirmed as the user's choice (identity not yet grounded in a datasheet)"
#: how the component provenance of a confirmed choice starts
CHOSEN_NOTE_PREFIX = "part chosen by the user"
#: the selection criteria of the spec that nothing in this version evaluates (see the module docstring)
FIT_CRITERIA: tuple[str, ...] = ("electrical stress", "safety", "regulatory", "environment", "reliability", "manufacturability", "sourcing", "cost")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartCandidate(_Strict):
    ref: str = Field(description="reference designator of the component this candidate is for")
    manufacturer: str
    mpn: str = Field(description="the manufacturer's exact orderable part number")
    kicad_symbol: str = Field(description="'Library:Name' of the KiCad symbol")
    kicad_footprint: str = Field(description="'Library:Name' of the KiCad footprint")
    rationale: str
    datasheet_url: str | None = Field(description="the manufacturer's datasheet URL if known; recorded, never fetched")


class PartCandidates(_Strict):
    candidates: list[PartCandidate]


def candidates_json_schema() -> dict[str, Any]:
    return _strictify(PartCandidates.model_json_schema())


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidates_fingerprint() -> dict[str, str]:
    """What a cached candidate reply depends on besides the component list."""
    return {
        "candidates_version": CANDIDATES_VERSION,
        "prompt_hash": _sha(PART_CANDIDATE_SYSTEM),
        "schema_hash": _sha(json.dumps(candidates_json_schema(), sort_keys=True, ensure_ascii=False)),
    }


def candidate_key(lacking: list[Component]) -> str:
    """Cache key of a candidate request: the identity-lacking components by ref, value and description (stable across the proposal itself)."""
    payload = [{"ref": c.ref, "value": c.value, "description": c.description} for c in sorted(lacking, key=lambda c: c.ref)]
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def facts_fingerprint() -> dict[str, str]:
    """What a cached facts reply depends on besides the part and the document: grounding version, prompt hash, schema hash."""
    return {
        "facts_version": FACTS_VERSION,
        "prompt_hash": _sha(DATASHEET_FACT_SYSTEM),
        "schema_hash": _sha(json.dumps(facts_json_schema(), sort_keys=True, ensure_ascii=False)),
    }


def facts_key(ref: str, document_sha256: str, keys: tuple[str, ...] | list[str] = DEFAULT_FACT_KEYS) -> str:
    """Cache key of a facts request: the part, the archived document it was asked about and the keys asked for."""
    return _sha(json.dumps({"ref": ref, "document": document_sha256, "keys": list(keys)}, sort_keys=True, ensure_ascii=False))


def lacks_identity(c: Component) -> bool:
    """No MPN at all, or an MPN that is an unconfirmed candidate from an earlier run."""
    if c.mpn is None:
        return True
    p = c.mpn.provenance
    note = p.note or ""
    return p.kind == ProvenanceKind.LLM_GENERATED and note.startswith(CANDIDATE_NOTE_PREFIX) and CONFIRMED_MARK not in note


def is_candidate_value(p: Provenance) -> bool:
    return p.kind == ProvenanceKind.LLM_GENERATED and (p.note or "").startswith(CANDIDATE_NOTE_PREFIX)


class GroundedCandidate(BaseModel):
    ref: str
    manufacturer: str
    mpn: str
    symbol: LibraryRef
    footprint: LibraryRef
    rationale: str
    datasheet_url: str | None = None
    #: which of symbol / footprint the component did not have before (set by the candidate)
    assigns: list[str] = Field(default_factory=list)

    def row(self) -> dict[str, str]:
        """What a person sees and confirms: the identity the candidate would give the part."""
        return {"ref": self.ref, "manufacturer": self.manufacturer, "mpn": self.mpn, "symbol": f"{self.symbol.library}:{self.symbol.name}",
                "footprint": f"{self.footprint.library}:{self.footprint.name}"}


def _lib_id(text: str) -> tuple[str, str] | None:
    if text.count(":") != 1:
        return None
    lib, name = (part.strip() for part in text.split(":", 1))
    if not lib or not name:
        return None
    return lib, name


def ground_candidates(parsed: PartCandidates, lacking: list[Component], library: KicadLibrary | None) -> tuple[list[GroundedCandidate], list[tuple[str, str]]]:
    """Deterministic gate on a model's candidates: known ref, no directive, symbol and footprint on disk and not a design change."""
    accepted: list[GroundedCandidate] = []
    rejected: list[tuple[str, str]] = []
    by_ref = {c.ref: c for c in lacking}
    seen: set[str] = set()
    for cand in parsed.candidates:
        ref = cand.ref.strip()
        comp = by_ref.get(ref)
        if comp is None:
            rejected.append((ref or "?", "not a component lacking an identity in this design"))
            continue
        hit = find_directive(cand.ref, cand.manufacturer, cand.mpn, cand.kicad_symbol, cand.kicad_footprint, cand.rationale, cand.datasheet_url)
        if hit is not None:
            rejected.append((ref, f"directive phrase {hit!r} in the candidate; model output is data, not instructions"))
            continue
        if ref in seen:
            rejected.append((ref, "duplicate candidate for this ref; the first was kept"))
            continue
        mpn, manufacturer = cand.mpn.strip(), cand.manufacturer.strip()
        if not mpn or not manufacturer:
            rejected.append((ref, "candidate without a manufacturer or an MPN"))
            continue
        if any(ch in mpn for ch in "*?") or " series" in mpn.lower():
            rejected.append((ref, f"MPN {mpn!r} is a placeholder / family name, not an orderable part number"))
            continue
        if library is None:
            rejected.append((ref, "no KiCad library available to verify the symbol and footprint"))
            continue
        sym_id, fp_id = _lib_id(cand.kicad_symbol), _lib_id(cand.kicad_footprint)
        if sym_id is None or fp_id is None:
            rejected.append((ref, f"symbol {cand.kicad_symbol!r} / footprint {cand.kicad_footprint!r} are not 'Library:Name' identifiers"))
            continue
        try:
            symbol = library.resolve_symbol(LibraryRef(library=sym_id[0], name=sym_id[1]))
            footprint = library.resolve_footprint(LibraryRef(library=fp_id[0], name=fp_id[1]))
        except LibraryFormatError as e:
            rejected.append((ref, f"library entry unreadable: {e}"))
            continue
        if not symbol.verified:
            rejected.append((ref, f"symbol {cand.kicad_symbol} does not exist in the KiCad libraries"))
            continue
        if not footprint.verified:
            rejected.append((ref, f"footprint {cand.kicad_footprint} does not exist in the KiCad libraries"))
            continue
        assigns: list[str] = []
        if comp.symbol is not None:
            if (comp.symbol.library, comp.symbol.name) != sym_id:
                rejected.append((ref, f"candidate names symbol {cand.kicad_symbol} but the design has {comp.symbol.library}:{comp.symbol.name} (a design change needs a human)"))
                continue
        else:
            assigns.append("symbol")
        if comp.footprint is not None:
            if (comp.footprint.library, comp.footprint.name) != fp_id:
                rejected.append((ref, f"candidate names footprint {cand.kicad_footprint} but the design has {comp.footprint.library}:{comp.footprint.name} (a design change needs a human)"))
                continue
        else:
            assigns.append("footprint")
        seen.add(ref)
        accepted.append(GroundedCandidate(ref=ref, manufacturer=manufacturer, mpn=mpn, symbol=symbol, footprint=footprint, rationale=cand.rationale.strip(),
                                          datasheet_url=(cand.datasheet_url or "").strip() or None, assigns=assigns))
    return accepted, rejected


def confirmation_question(accepted: list[GroundedCandidate], rejected: list[tuple[str, str]], model: str, *, fresh: list[str] | None = None) -> MissingInformation:
    """The ``confirm_parts`` table; ``fresh`` names refs whose row was not in any earlier table (they need this table to be seen first)."""
    rows = [[g.ref + (" (new)" if fresh and g.ref in fresh else ""), g.manufacturer, g.mpn, f"{g.symbol.library}:{g.symbol.name}", f"{g.footprint.library}:{g.footprint.name}",
             g.rationale, g.datasheet_url or "-"] for g in accepted]
    parts = [
        f"Candidate parts proposed by {model} for components without a part number. Nothing below is trusted: the symbol and footprint "
        "were found in the installed KiCad libraries, nothing else was checked - the rationale is the model's prose, not an evaluation of fit. "
        "Confirming makes them YOUR choice; each part still has to be found in its manufacturer's datasheet before its identity counts as verified.",
        "",
        _table(["ref", "manufacturer", "mpn", "symbol", "footprint", "rationale", "datasheet_url (model, not fetched)"], rows),
    ]
    if fresh:
        parts += ["", f"Rows marked (new) were not in the table you saw before ({', '.join(fresh)}); only rows already shown can be confirmed."]
    if rejected:
        parts += ["", "Rejected candidates:", *[f"  - {ref}: {why}" for ref, why in rejected]]
    parts += ["", f"Reply yes / y / ok / confirm (네 / 예 / 확인) with --answer {CONFIRM_PARTS_KEY}=yes to use these parts, or no to discard them."]
    return MissingInformation(key=CONFIRM_PARTS_KEY, question="\n".join(parts), required=True, rationale="a model's part proposal becomes a design choice only when the user confirms it")


def facts_confirmation_question(ref: str, doc: ArchivedDocument, grounded: GroundedFacts, model: str) -> MissingInformation:
    """The ``confirm_facts`` table for one component: every grounded fact with its page, quote and context, and the rejected ones."""
    rows = [[a.key, a.target, f"{a.traced.value}" + (f" {a.traced.unit}" if a.traced.unit else ""), str(a.page), a.quote, a.context] for a in grounded.accepted]
    parts = [
        f"Datasheet facts proposed by {model} for {ref} from the archived datasheet {doc.sha256} ({doc.title or doc.path.name}). Each quote was found verbatim "
        "on the page named and each number was re-read from the document's own text; what was NOT checked is the meaning the model gave the number - "
        "that a value is the maximum rating rather than a typical or a dropout figure is decided by the key the model chose. Read the context column "
        "and confirm only if every key fits its quote.",
        "",
        _table(["key", "target", "value", "page", "quote (document text)", "context"], rows),
    ]
    if grounded.rejected:
        parts += ["", "Rejected (never enter the IR):", *[f"  - {k}: {why}" for k, why in grounded.rejected]]
    parts += ["", f"Reply yes / y / ok / confirm (네 / 예 / 확인) with --answer {CONFIRM_FACTS_KEY}=yes to record these facts as the datasheet's values, or no to discard them."]
    return MissingInformation(key=CONFIRM_FACTS_KEY, question="\n".join(parts), required=True,
                              rationale="a model's reading of a datasheet enters the IR only when the user confirms the table it was shown in")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_json(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _billed_cost(llm: LLMService) -> float | None:
    billed = [a for a in llm.last_attempts if a.usage is not None]
    costs = [a.cost_usd for a in billed]
    return None if not costs or any(c is None for c in costs) else float(sum(c for c in costs if c is not None))


class ComponentAgent(Agent):
    name = "component"
    task = TaskKind.COMPONENT_PROPOSAL

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        library = ctx.tools.get("kicad_library")
        library = library if isinstance(library, KicadLibrary) else None
        archive = ctx.tools.get("archive")
        archive = archive if isinstance(archive, DocumentArchive) else None
        catalog = ctx.tools.get("catalog")
        catalog = catalog if isinstance(catalog, CatalogSource) else None
        notes: list[str] = []
        questions: list[MissingInformation] = []
        validation: list[ValidationResult] = []
        if not ir.components:
            return self._result(notes=["no components in the IR: nothing to check"])
        components = [c.model_copy(deep=True) for c in ir.components]
        if library is None:
            notes.append("no KiCad library in the tool context: symbols and footprints cannot be resolved")
        if archive is None:
            notes.append("no document archive in the tool context: datasheets are neither fetched nor verified")
        elif not archive.online:
            notes.append("offline: datasheets are not fetched (open an online session to fetch them)")

        pending: set[str] = set()
        if ctx.llm is not None:
            pending = self._candidates(ir, ctx, components, library, notes, questions, validation)

        facts_by_ref: dict[str, list[DatasheetFact]] = {}
        facts_path = ctx.answers.get(FACTS_FILE_KEY)
        if facts_path:
            facts_by_ref, errors = load_facts_file(facts_path)
            notes.extend(f"{FACTS_FILE_KEY}: {e}" for e in errors)
            for ref in sorted(set(facts_by_ref) - {c.ref for c in components}):
                notes.append(f"{FACTS_FILE_KEY}: facts for {ref!r} ignored (no such component)")
        extract = ctx.llm is not None and is_confirmation(ctx.answers.get(EXTRACT_FACTS_KEY))
        user_urls = archive.policy.user_urls if archive is not None else None
        facts_cache = _load_json(Path(ctx.workdir) / FACTS_CACHE_FILE)
        facts_cache_dirty = False

        updated: list[Component] = []
        for c in components:
            if c.ref in pending:
                validation.append(ValidationResult(
                    check_id=f"{EXISTENCE_PREFIX}{c.ref}", status=ValidationStatus.NOT_VERIFIED, tool=EXISTENCE_TOOL,
                    message=f"candidate part awaiting the user's confirmation ({CONFIRM_PARTS_KEY}); existence not checked and nothing fetched",
                ))
                updated.append(c)
                continue
            report = examine_component(c, library, archive, catalog, user_urls=user_urls, purpose=f"datasheet of {c.ref} ({c.value})")
            validation.append(report.result(catalog))
            c = self._apply_report(c, report, notes)
            doc = report.document
            user_facts = facts_by_ref.get(c.ref, [])
            if doc is not None and doc.text_available:
                title = c.datasheet.title if c.datasheet is not None else None
                authority = c.datasheet.authority if c.datasheet is not None else None
                mpn = str(c.mpn.value) if c.mpn is not None and c.mpn.value is not None else None
                if user_facts:
                    grounded = ground_facts(doc, user_facts, proposer="user file", title=title, authority=authority, mpn=mpn)
                    c = apply_facts(c, grounded.accepted)
                    validation.append(facts_result(c.ref, doc, grounded))
                c, dirty = self._model_facts(ctx, c, doc, title, authority, mpn, extract, facts_cache, notes, questions, validation)
                facts_cache_dirty = facts_cache_dirty or dirty
            elif user_facts:
                notes.append(f"{c.ref}: {len(user_facts)} fact(s) from {FACTS_FILE_KEY} not grounded: datasheet not archived or its text not extractable")
            updated.append(c)
        if facts_cache_dirty:
            _save_json(Path(ctx.workdir) / FACTS_CACHE_FILE, facts_cache)

        proposals: list[IRProposal] = []
        if [c.model_dump(mode="json") for c in updated] != [c.model_dump(mode="json") for c in ir.components]:
            proposals.append(IRProposal(
                description="record datasheet references, grounded identities and catalog sourcing on the components",
                target="components", operation="set", payload=updated,
                rationale="only values found verbatim in an archived datasheet or a hashed catalog file became authoritative; everything else is a recorded pointer or a model proposal",
            ))
        checked = [r for r in validation if r.check_id.startswith(EXISTENCE_PREFIX)]
        by_status: dict[str, int] = {}
        for r in checked:
            by_status[str(r.status)] = by_status.get(str(r.status), 0) + 1
            if r.status is not ValidationStatus.PASS:
                notes.append(f"{r.check_id} {r.status}: {r.message}")
        notes.insert(0, f"existence checked for {len(checked)} component(s): " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
        validation.append(ValidationResult(
            check_id=FIT_CHECK, status=ValidationStatus.NOT_VERIFIED,
            message=("part fit not evaluated: this version checks that each part exists (library entry, archived datasheet, MPN, catalog row) and grounds "
                     "facts, but no criterion of the selection order is compared with the requirements or the simulated operating point ("
                     + ", ".join(FIT_CRITERIA) + "); a model's rationale is prose, not evidence"),
            details={"refs": [c.ref for c in updated], "criteria_not_evaluated": list(FIT_CRITERIA), "criteria_evaluated": []},
        ))
        return self._result(proposals=proposals, questions=questions, validation=validation, notes=notes)

    # ------------------------------------------------------------------ existence -> proposals

    @staticmethod
    def _apply_report(c: Component, report: ExistenceReport, notes: list[str]) -> Component:
        """The component with what the check *found* recorded: the archived datasheet, the grounded MPN, the catalog sourcing."""
        doc = report.document
        pointer = report.pointer
        if doc is not None:
            title = (c.datasheet.title if c.datasheet is not None else None) or (pointer.ref.title if pointer is not None else None)
            trusted_manufacturer = c.manufacturer is not None and c.manufacturer.provenance.is_authoritative and c.manufacturer.value not in (None, "")
            authority = (c.datasheet.authority if c.datasheet is not None else None) or (str(c.manufacturer.value) if trusted_manufacturer else None) \
                or doc.meta.get("authority") or (pointer.host if pointer is not None else None)
            if c.datasheet is None or c.datasheet.content_hash != doc.sha256 or c.datasheet.document_path != str(doc.path):
                c.datasheet = doc.source_ref(title=title, authority=authority)
                notes.append(f"{c.ref}: datasheet reference set to the archived copy {doc.sha256} ({doc.path.name})")
        elif pointer is not None and c.datasheet is None and pointer.origin != "ir":
            c.datasheet = pointer.ref
            notes.append(f"{c.ref}: datasheet pointer recorded from {pointer.origin} ({pointer.ref.url}); not archived")
        hit = report.mpn_hit
        if hit is not None and doc is not None and c.mpn is not None:
            old = c.mpn.provenance
            same = (old.kind == ProvenanceKind.AUTHORITATIVE and old.source is not None and old.source.content_hash == doc.sha256 and old.source.section == hit.section
                    and doc.extraction_stamp in (old.note or ""))
            if not same:
                src = doc.source_ref(title=c.datasheet.title if c.datasheet is not None else None, section=hit.section,
                                     authority=c.datasheet.authority if c.datasheet is not None else None)
                prev = f"; previously {old.kind}" + (f": {old.note}" if old.note else "")
                c.mpn = authoritative(c.mpn.value, src, note=f"MPN found verbatim on page {hit.page} of the archived datasheet: {hit.context}; {doc.extraction_stamp}{prev}")
                notes.append(f"{c.ref}: MPN {c.mpn.value!r} grounded on page {hit.page} of {doc.sha256}: authoritative")
        if report.sourcing is not None:
            c.sourcing = [s for s in c.sourcing if s.supplier != report.sourcing.supplier] + [report.sourcing]
        return c

    # ------------------------------------------------------------------ datasheet facts (LLM) -> confirmed by the user

    @staticmethod
    def _facts_stale(entry: Any) -> str | None:
        if not isinstance(entry, dict):
            return "facts cache entry is not an object"
        for field, want in facts_fingerprint().items():
            if entry.get(field) != want:
                return f"cached datasheet facts were produced under another {field.replace('_', ' ')}; not replayed"
        try:
            DatasheetFacts.model_validate(entry.get("facts"))
        except ValidationError as e:
            return f"cached datasheet facts no longer match the schema ({str(e).splitlines()[0][:200]}); not replayed"
        return None

    def _model_facts(
        self, ctx: AgentContext, c: Component, doc: ArchivedDocument, title: str | None, authority: str | None, mpn: str | None, extract: bool,
        cache: dict[str, Any], notes: list[str], questions: list[MissingInformation], validation: list[ValidationResult],
    ) -> tuple[Component, bool]:
        """Propose (billed, on request) / replay / confirm a model's facts for ``c`` from ``doc``; returns the component and whether the cache changed."""
        key = facts_key(c.ref, doc.sha256, DEFAULT_FACT_KEYS)
        check_id = f"component.facts.{c.ref}.llm"
        entry: dict[str, Any] | None = cache.get(key)
        if entry is not None:
            stale = self._facts_stale(entry)
            if stale is not None:
                notes.append(f"{c.ref}: {stale}")
                entry = None
        answer = ctx.answers.get(CONFIRM_FACTS_KEY)
        wants_confirm, wants_reject = is_confirmation(answer), is_rejection(answer)
        if answer is not None and not wants_confirm and not wants_reject:
            notes.append(f"answer {answer.strip()!r} to {CONFIRM_FACTS_KEY} not understood: reply yes to record the shown facts or no to discard them")
        called = False
        if entry is None:
            if not extract:
                if ctx.llm is not None and wants_confirm:
                    notes.append(f"{c.ref}: no model facts to confirm for the archived datasheet {doc.sha256[:19]}… (ask with --answer {EXTRACT_FACTS_KEY}=yes first)")
                return c, False
            assert ctx.llm is not None
            try:
                proposed, resp, _ = llm_fact_proposals(ctx.llm, doc, c)
            except (LLMError, BudgetExceededError, StructuredOutputError) as e:
                validation.append(ValidationResult(check_id=check_id, status=ValidationStatus.NOT_VERIFIED,
                                                   message=f"model facts not obtained ({type(e).__name__}): {e}", details={"error_type": type(e).__name__}))
                return c, False
            entry = {
                **facts_fingerprint(), "key": key, "ref": c.ref, "document": doc.sha256, "keys": list(DEFAULT_FACT_KEYS),
                "requested_model": resp.model, "model": resp.model_used or resp.model, "facts": proposed.model_dump(mode="json"),
                "usage": resp.usage.model_dump(mode="json"), "cost_usd": _billed_cost(ctx.llm), "created_at": _now(),
                "presented": False, "confirmed": False, "rejected": False, "presented_facts": [], "confirmed_facts": [],
            }
            cache[key] = entry
            called = True
        proposed = DatasheetFacts.model_validate(entry["facts"])
        model = str(entry.get("model"))
        if entry.get("rejected"):
            notes.append(f"{c.ref}: model facts for this datasheet were discarded by the user earlier; not shown again")
            validation.append(ValidationResult(check_id=check_id, status=ValidationStatus.NOT_VERIFIED, message="model facts discarded by the user; nothing entered the IR",
                                               details={"key": key, "model": model, "rejected_by_user": True, "cache_file": str(Path(ctx.workdir) / FACTS_CACHE_FILE)}))
            return c, called
        grounded = ground_facts(doc, proposed, proposer=f"model {model}", title=title, authority=authority, mpn=mpn)
        shown = [fact_row(a) for a in grounded.accepted]
        presented_before, confirmed_before = bool(entry.get("presented")), bool(entry.get("confirmed"))
        dirty = called
        if wants_reject and not confirmed_before:
            entry["rejected"], entry["rejected_at"] = True, _now()
            notes.append(f"{c.ref}: model facts discarded by the user ({len(grounded.accepted)} grounded fact(s) not recorded)")
            validation.append(ValidationResult(check_id=check_id, status=ValidationStatus.NOT_VERIFIED, message="model facts discarded by the user; nothing entered the IR",
                                               details={"key": key, "model": model, "rejected_by_user": True, "discarded": shown}))
            return c, True
        confirm_now = False
        if wants_confirm and not confirmed_before:
            if called or not presented_before:
                notes.append(f"{c.ref}: confirmation ignored - the model facts were (re)generated in this run and have not been shown to you yet; review the table and confirm again")
            elif not grounded.accepted:
                notes.append(f"{c.ref}: confirmation ignored - no grounded model fact to confirm")
            elif shown != list(entry.get("presented_facts") or []):
                notes.append(f"{c.ref}: confirmation ignored - the grounded facts differ from the table you saw (rules or document changed); review the new table and confirm again")
            else:
                confirm_now = True
        confirmed = confirmed_before or confirm_now
        if confirmed:
            confirmed_rows = shown if confirm_now else list(entry.get("confirmed_facts") or [])
            final = ground_facts(doc, proposed, proposer=f"model {model}", title=title, authority=authority, mpn=mpn, confirmed_by=CONFIRM_FACTS_KEY)
            keep = [a for a in final.accepted if fact_row(a) in confirmed_rows]
            c = apply_facts(c, keep)
            if confirm_now:
                entry["confirmed"], entry["confirmed_at"], entry["confirmed_facts"] = True, _now(), shown
                dirty = True
                notes.append(f"{c.ref}: {len(keep)} model fact(s) confirmed by the user and recorded as the datasheet's values ({', '.join(a.key for a in keep)})")
            skipped = [fact_row(a)["key"] for a in final.accepted if fact_row(a) not in confirmed_rows]
            status = ValidationStatus.PASS if keep and not final.rejected and not skipped else ValidationStatus.NOT_VERIFIED
            suffix = f"{len(keep)} confirmed by the user ({CONFIRM_FACTS_KEY})" + (f"; not confirmed (changed since the table): {skipped}" if skipped else "")
            res = facts_result(c.ref, doc, final, check_id=check_id, status=status, message_suffix=suffix)
        elif grounded.accepted:
            questions.append(facts_confirmation_question(c.ref, doc, grounded, model))
            if not presented_before or shown != list(entry.get("presented_facts") or []):
                entry["presented"], entry["presented_at"], entry["presented_facts"] = True, _now(), shown
                dirty = True
            res = facts_result(c.ref, doc, grounded, check_id=check_id, status=ValidationStatus.USER_INPUT_REQUIRED,
                               message_suffix=f"awaiting the user's confirmation ({CONFIRM_FACTS_KEY}); nothing entered the IR")
        else:
            res = facts_result(c.ref, doc, grounded, check_id=check_id, status=ValidationStatus.NOT_VERIFIED, message_suffix="no fact grounded; nothing to confirm")
        res.details.update({
            "key": key, "not_found": list(proposed.not_found), "model": model, "requested_model": entry.get("requested_model"), "cached": not called,
            "presented": bool(entry.get("presented")), "confirmed": confirmed, "cost_usd": entry.get("cost_usd"), "cost_known": entry.get("cost_usd") is not None,
            **{k: entry.get(k) for k in facts_fingerprint()}, "cache_file": str(Path(ctx.workdir) / FACTS_CACHE_FILE),
        })
        validation.append(res)
        return c, dirty

    # ------------------------------------------------------------------ candidates (LLM)

    def _cache_path(self, ctx: AgentContext) -> Path:
        return Path(ctx.workdir) / CANDIDATE_CACHE_FILE

    def _load_cache(self, ctx: AgentContext) -> dict[str, Any]:
        return _load_json(self._cache_path(ctx))

    def _save_cache(self, ctx: AgentContext, cache: dict[str, Any]) -> None:
        _save_json(self._cache_path(ctx), cache)

    @staticmethod
    def _stale(entry: Any) -> str | None:
        if not isinstance(entry, dict):
            return "cache entry is not an object"
        for field, want in candidates_fingerprint().items():
            if entry.get(field) != want:
                return f"cached candidates were produced under another {field.replace('_', ' ')}; asking again"
        try:
            PartCandidates.model_validate(entry.get("candidates"))
        except ValidationError as e:
            return f"cached candidates no longer match the schema ({str(e).splitlines()[0][:200]}); asking again"
        return None

    @staticmethod
    def _propose(llm: LLMService, lacking: list[Component], ir: CircuitIR, key: str) -> dict[str, Any]:
        """One structured call; the returned dict is the cache entry (model output kept verbatim, cost summed over billed attempts)."""
        about = []
        for c in lacking:
            about.append({
                "ref": c.ref, "value": c.value, "description": c.description,
                "kicad_symbol": f"{c.symbol.library}:{c.symbol.name}" if c.symbol is not None else None,
                "kicad_footprint": f"{c.footprint.library}:{c.footprint.name}" if c.footprint is not None else None,
                "electrical": {k: f"{t.value} {t.unit or ''}".strip() for k, t in c.electrical.items()},
            })
        requirements = [r.text for r in ir.requirements.requirements]
        messages = part_candidate_messages(about, requirements)
        parsed, resp = llm.structured(TaskKind.COMPONENT_PROPOSAL, messages, PartCandidates, json_schema=candidates_json_schema())
        billed = [a for a in llm.last_attempts if a.usage is not None]
        return {
            **candidates_fingerprint(),
            "key": key,
            "refs": [c.ref for c in lacking],
            "requested_model": resp.model,
            "model": resp.model_used or resp.model,
            "candidates": parsed.model_dump(mode="json"),
            "usage": resp.usage.model_dump(mode="json"),
            "cost_usd": _billed_cost(llm),
            "billed_attempts": len(billed),
            "created_at": _now(),
            "presented": False,
            "confirmed": False,
            "rejected": False,
            "assigned": {},
            "presented_rows": [],
            "confirmed_rows": [],
        }

    def _candidates(
        self, ir: CircuitIR, ctx: AgentContext, components: list[Component], library: KicadLibrary | None,
        notes: list[str], questions: list[MissingInformation], validation: list[ValidationResult],
    ) -> set[str]:
        """Propose / confirm candidate parts for identity-lacking components; returns the refs still awaiting confirmation."""
        llm = ctx.llm
        assert llm is not None
        lacking = [c for c in components if lacks_identity(c)]
        if not lacking:
            return set()
        key = candidate_key(lacking)
        cache = self._load_cache(ctx)
        entry: dict[str, Any] | None = cache.get(key)
        if entry is not None:
            stale = self._stale(entry)
            if stale is not None:
                notes.append(stale)
                entry = None
        lacking_refs = {c.ref for c in lacking}
        if entry is None:
            # some parts of an earlier table were confirmed and grounded since: the reply made for the larger set still covers the rest
            for other in cache.values():
                if isinstance(other, dict) and lacking_refs <= set(other.get("refs") or []) and self._stale(other) is None and not other.get("rejected"):
                    entry, key = other, str(other.get("key") or key)
                    notes.append(f"reusing the candidate reply made for {other.get('refs')} (the rest were confirmed and grounded since)")
                    break
        answer = ctx.answers.get(CONFIRM_PARTS_KEY)
        wants_confirm = is_confirmation(answer)
        wants_reject = is_rejection(answer)
        if answer is not None and not wants_confirm and not wants_reject:
            notes.append(f"answer {answer.strip()!r} to {CONFIRM_PARTS_KEY} not understood: reply yes to use the candidates or no to discard them")
        if entry is not None and entry.get("rejected"):
            notes.append(f"candidates for {', '.join(entry.get('refs', []))} were rejected by the user earlier; not asking the model again "
                         f"(give the parts in the IR or with --datasheet-url)")
            validation.append(ValidationResult(check_id=CANDIDATES_CHECK, status=ValidationStatus.NOT_VERIFIED, message="candidate parts rejected by the user; components still lack an identity",
                                               details={"key": key, "refs": entry.get("refs", []), "rejected_by_user": True}))
            return set()
        called = False
        if entry is None:
            try:
                entry = self._propose(llm, lacking, ir, key)
            except (LLMError, BudgetExceededError, StructuredOutputError) as e:
                kind = type(e).__name__
                validation.append(ValidationResult(check_id=CANDIDATES_CHECK, status=ValidationStatus.NOT_VERIFIED, message=f"candidate parts not proposed ({kind}): {e}",
                                                   details={"key": key, "refs": [c.ref for c in lacking], "error_type": kind, "error": str(e)}))
                notes.append(f"LLM candidate proposal unavailable ({kind}); components without an identity stay as they are")
                return set()
            called = True
            cache[key] = entry
        parsed = PartCandidates.model_validate(entry["candidates"])
        if set(entry.get("refs") or []) != lacking_refs:
            parsed = PartCandidates(candidates=[c for c in parsed.candidates if c.ref.strip() in lacking_refs])
        accepted, rejected = ground_candidates(parsed, lacking, library)
        model = str(entry.get("model"))
        presented_rows: list[dict[str, str]] = list(entry.get("presented_rows") or [])
        confirmed_rows: list[dict[str, str]] = list(entry.get("confirmed_rows") or [])
        by_ref = {c.ref: c for c in components}

        if wants_reject and not confirmed_rows:
            for ref, assigned in (entry.get("assigned") or {}).items():
                comp = by_ref.get(ref)
                if comp is None:
                    continue
                if "symbol" in assigned:
                    comp.symbol = None
                if "footprint" in assigned:
                    comp.footprint = None
            for comp in lacking:
                if comp.mpn is not None and is_candidate_value(comp.mpn.provenance):
                    comp.mpn = None
                if comp.manufacturer is not None and is_candidate_value(comp.manufacturer.provenance):
                    comp.manufacturer = None
            entry["rejected"] = True
            entry["rejected_at"] = _now()
            self._save_cache(ctx, cache)
            notes.append(f"candidate parts rejected by the user and removed from {[g.ref for g in accepted]}")
            validation.append(ValidationResult(check_id=CANDIDATES_CHECK, status=ValidationStatus.NOT_VERIFIED, message="candidate parts rejected by the user; components still lack an identity",
                                               details={"key": key, "refs": [g.ref for g in accepted], "rejected_by_user": True}))
            return set()
        if wants_reject and confirmed_rows:
            notes.append(f"rejection ignored: candidates were already confirmed as your choice for {[r['ref'] for r in confirmed_rows]} - remove the parts by hand")

        # confirmation is row by row: only a row that stood in a table shown in an earlier run can become the user's choice
        already = [g for g in accepted if g.row() in confirmed_rows]
        confirmable = [g for g in accepted if g.row() not in confirmed_rows and g.row() in presented_rows]
        fresh = [g for g in accepted if g.row() not in confirmed_rows and g.row() not in presented_rows]
        confirm_now: list[GroundedCandidate] = []
        if wants_confirm:
            if called or not presented_rows:
                notes.append("confirmation ignored: the candidates were (re)generated in this run and have not been shown to you yet; review the table and confirm again")
            elif not confirmable:
                notes.append("confirmation ignored: no candidate row you were shown earlier is left to confirm"
                             + (f" ({[g.ref for g in fresh]} appeared only in this run and must be seen first)" if fresh else ""))
            else:
                confirm_now = confirmable
                confirmed_rows.extend(g.row() for g in confirm_now)
                entry["confirmed_rows"] = confirmed_rows
                entry["confirmed_at"] = _now()
                if fresh:
                    notes.append(f"{[g.ref for g in fresh]}: candidate rows that were not in the table you saw are not confirmed; they are shown now")
        confirmed_refs = {g.ref for g in already} | {g.ref for g in confirm_now}
        pending_candidates = [g for g in accepted if g.ref not in confirmed_refs]

        assigned: dict[str, list[str]] = {}
        for g in accepted:
            comp = by_ref[g.ref]
            is_confirmed = g.ref in confirmed_refs
            note = f"{CANDIDATE_NOTE_PREFIX}{model}; rationale: {g.rationale}" + (f"; model datasheet URL (recorded, not fetched): {g.datasheet_url}" if g.datasheet_url else "")
            if is_confirmed:
                note += CONFIRMED_MARK
            comp.mpn = llm_generated(g.mpn, model, note=note)
            comp.manufacturer = llm_generated(g.manufacturer, model, note=note)
            if "symbol" in g.assigns:
                comp.symbol = g.symbol
            if "footprint" in g.assigns:
                comp.footprint = g.footprint
            if g.assigns:
                assigned[g.ref] = list(g.assigns)
            if is_confirmed:
                comp.provenance = Provenance(kind=ProvenanceKind.USER_REQUIREMENT,
                                             note=f"{CHOSEN_NOTE_PREFIX} from candidates proposed by {model}; rationale: {g.rationale}")
        if assigned and not entry.get("assigned"):
            entry["assigned"] = assigned
        if confirm_now:
            notes.append(f"candidate parts confirmed by the user for {[g.ref for g in confirm_now]}: the choice is the user's; identity still needs the datasheet")
        pending: set[str] = {g.ref for g in pending_candidates}
        if pending_candidates:
            questions.append(confirmation_question(pending_candidates, rejected, model, fresh=[g.ref for g in fresh]))
            for g in pending_candidates:
                if g.row() not in presented_rows:
                    presented_rows.append(g.row())
            entry["presented_rows"] = presented_rows
            if not entry.get("presented"):
                entry["presented"] = True
                entry["presented_at"] = _now()
        elif not accepted:
            notes.append("no acceptable candidate part was proposed" + (f" ({len(rejected)} rejected)" if rejected else ""))
        entry["confirmed"] = bool(accepted) and not pending_candidates and bool(confirmed_refs)
        self._save_cache(ctx, cache)
        cost = entry.get("cost_usd")
        if pending:
            status = ValidationStatus.USER_INPUT_REQUIRED
        elif entry["confirmed"]:
            status = ValidationStatus.PASS
        else:
            status = ValidationStatus.NOT_VERIFIED
        message = (
            f"{len(accepted)} candidate(s) accepted by the library gate, {len(rejected)} rejected; model {model}{' (cached)' if not called else ''}; "
            f"cost {'unknown' if cost is None else f'{cost:.6f} USD'}; "
            f"{'confirmed by user' if entry['confirmed'] else f'awaiting user confirmation for {sorted(pending)}' if accepted else 'nothing to confirm'}"
        )
        validation.append(ValidationResult(
            check_id=CANDIDATES_CHECK, status=status, message=message,
            tool=str(entry.get("requested_model") or model), tool_version=model,
            details={
                "key": key, "model": model, "requested_model": entry.get("requested_model"), "cached": not called,
                "presented": bool(entry.get("presented")), "confirmed": entry["confirmed"], "confirmed_refs": sorted(confirmed_refs), "pending_refs": sorted(pending),
                "fresh_refs": [g.ref for g in fresh], "cost_usd": cost, "cost_known": cost is not None,
                "billed_attempts": entry.get("billed_attempts"), **{k: entry.get(k) for k in candidates_fingerprint()},
                "accepted": [g.model_dump(mode="json") for g in accepted],
                "rejected": [{"ref": r, "reason": why} for r, why in rejected],
                "cache_file": str(self._cache_path(ctx)),
            },
        ))
        return pending


__all__ = [
    "CANDIDATES_CHECK",
    "CANDIDATES_VERSION",
    "CANDIDATE_CACHE_FILE",
    "CANDIDATE_NOTE_PREFIX",
    "CHOSEN_NOTE_PREFIX",
    "CONFIRMED_MARK",
    "CONFIRM_FACTS_KEY",
    "CONFIRM_PARTS_KEY",
    "EXTRACT_FACTS_KEY",
    "FACTS_CACHE_FILE",
    "FACTS_FILE_KEY",
    "FIT_CHECK",
    "FIT_CRITERIA",
    "ComponentAgent",
    "GroundedCandidate",
    "PartCandidate",
    "PartCandidates",
    "candidate_key",
    "candidates_fingerprint",
    "candidates_json_schema",
    "confirmation_question",
    "facts_confirmation_question",
    "facts_fingerprint",
    "facts_key",
    "ground_candidates",
    "is_candidate_value",
    "lacks_identity",
]
