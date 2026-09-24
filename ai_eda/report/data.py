"""What the report shows, collected as plain data (no markup, no verdict).

Invariant: :func:`build_report_data` computes no :class:`ValidationStatus`.
Every status it carries is a stored ``ValidationResult.status`` or
``StageOutcome.status`` copied verbatim; the only labels it adds are facts
about hashes and files - whether a result's ``ir_hash`` is the current
design hash (fresh / stale / unstamped), whether an artifact or an evidence
file is still on disk with the recorded hash, and whether ``pipeline.json``
describes the ir.json it sits next to (``ir_file_sha256``). The RELEASE
verdict and its reasons come only from the recorded RELEASE outcome; the
aggregate of the latest results is labelled as not a verdict. Everything is
rendered from ``model_dump(mode="json")`` so timestamps read exactly as the
files store them, and every field is a str / int / bool / list of those, so
the renderer can never print an object repr.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from ai_eda import __version__
from ai_eda.ir.project import ArtifactRef, CircuitIR
from ai_eda.ir.requirements import MissingInformation
from ai_eda.ir.validation import Evidence, ValidationResult, ValidationStatus
from ai_eda.report.pipeline_log import PIPELINE_FILE, PipelineRecord, PipelineRecordError, load_pipeline_record, sha256_of_file
from ai_eda.review.areas import ReviewArea
from ai_eda.workflow.stages import STAGE_ORDER, Stage

#: freshness labels (facts about hashes, never statuses)
FRESH = "fresh"
STALE = "stale"
UNSTAMPED_PRODUCED = "unstamped (produced by the recorded run)"
UNSTAMPED_CARRIED = "unstamped (carried over from an earlier run - vouches for nothing now)"
UNSTAMPED_UNKNOWN = "unstamped (run unknown)"
#: file-state labels
ON_DISK = "on disk"
CHANGED_ON_DISK = "changed on disk"
MISSING_ON_DISK = "missing on disk"
EVIDENCE_OK = "on disk (hash matches)"
EVIDENCE_NO_HASH = "present (no hash recorded)"
EVIDENCE_NO_PATH = "no path"
EVIDENCE_MISSING = "missing"
#: labels about the run log
PIPELINE_DESCRIBES_IR = "pipeline.json describes this ir.json"
PIPELINE_STALE_IR = "ir.json changed since pipeline.json was written"
NO_RECORDED_RUN = "no recorded run: run `ai-eda run` first"
OPINION = "opinion (no tool)"
MODEL_OUTPUT = "model output"
AGGREGATE_NOTE = "aggregate of the latest results - not a release verdict; see RELEASE"
RELEASE_PREFIX = "not releasable: "
#: check-id prefixes of the domain sections (latest results only)
REGULATORY_PREFIXES = ("regulatory.",)
COMPONENT_PREFIXES = ("component.", "ir.component_provenance")
SIMULATION_PREFIXES = ("spice", "domain.analog.", "compile.spice_netlist")


class MetaSection(BaseModel):
    project_id: str
    project_name: str
    description: str
    created_at: str
    workdir: str
    schema_version: str
    ai_eda_version: str
    design_hash: str
    ir_file_path: str
    ir_file_sha256: str
    pipeline_file_path: str | None
    pipeline_file_sha256: str | None
    #: PIPELINE_DESCRIBES_IR / PIPELINE_STALE_IR / why there is no usable record
    pipeline_note: str


class StageRow(BaseModel):
    stage: str
    status: str
    message: str
    at: str
    questions: int
    reached: bool


class StagesSection(BaseModel):
    rows: list[StageRow]
    run_ir_hash: str
    run_hash_label: str
    blocked: bool
    current: str | None
    aborted: str | None
    aborted_stage: str | None
    results_before: int
    ai_eda_version: str
    ir_path: str


class RequirementRow(BaseModel):
    id: str
    text: str
    kind: str
    status: str
    category: str
    value: str
    unit: str
    provenance_kind: str
    needs_verification: bool


class ParameterRow(BaseModel):
    key: str
    value: str
    unit: str
    provenance_kind: str
    tool: str


class ExtractionRow(BaseModel):
    request_hash: str
    confirmed: bool
    presented: bool


class RequirementsSection(BaseModel):
    raw_input: str
    corrections: list[str]
    rows: list[RequirementRow]
    parameters: list[ParameterRow]
    conflicts: list[str]
    #: what a model said (count + flags only): labelled MODEL_OUTPUT, never design data
    extraction: list[ExtractionRow]


class QuestionRow(BaseModel):
    key: str
    question: str
    required: bool
    options: list[str]
    rationale: str
    source: str
    source_label: str
    origin: str
    command: str


class EvidenceRow(BaseModel):
    description: str
    path: str
    url: str
    content_hash: str
    state: str


class ValidationRow(BaseModel):
    index: int
    check_id: str
    status: str
    message: str
    tool: str
    tool_version: str
    opinion: bool
    ir_hash: str
    freshness: str
    artifact_hash: str
    artifact_kind: str
    evidence: list[EvidenceRow]
    details_json: str
    #: MODEL_OUTPUT for a result whose tool is a model id (requirements.extraction), else ""
    note: str
    timestamp: str


class HistoryEntry(BaseModel):
    check_id: str
    index: int
    status: str
    timestamp: str
    message: str


class ValidationSection(BaseModel):
    latest: list[ValidationRow]
    history: list[HistoryEntry]
    aggregate: str
    aggregate_note: str


class ArtifactRow(BaseModel):
    kind: str
    path: str
    files: int
    generator: str
    generator_version: str
    content_hash: str
    generated_from_ir_hash: str
    freshness: str
    disk: str
    notes: list[str]
    created_at: str


class ReviewRow(BaseModel):
    area: str
    status: str
    message: str
    freshness: str
    evidence: int
    timestamp: str


class ReviewSection(BaseModel):
    rows: list[ReviewRow]
    counts: dict[str, int]
    note: str


class RepairSection(BaseModel):
    status: str
    message: str
    freshness: str
    iterations: str
    stopped_reason: str
    actions_json: str
    unresolved_json: str
    final_review_json: str
    timestamp: str


class DomainSection(BaseModel):
    title: str
    prefixes: list[str]
    results: list[ValidationRow]
    ir_json: str
    note: str


class ReleaseSection(BaseModel):
    status: str | None
    message: str
    reasons: list[str]
    at: str
    note: str


class ReportData(BaseModel):
    meta: MetaSection
    stages: StagesSection | None
    stages_reason: str | None
    requirements: RequirementsSection
    questions: list[QuestionRow]
    validation: ValidationSection
    artifacts: list[ArtifactRow]
    review: ReviewSection
    repair: RepairSection | None
    regulatory: DomainSection
    components: DomainSection
    simulation: DomainSection
    release: ReleaseSection


# --- helpers -----------------------------------------------------------------


def _s(value: object) -> str:
    return "" if value is None else str(value)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _traced_fields(traced) -> tuple[str, str, str, str, bool]:
    """``(value, unit, provenance kind, tool, needs_verification)`` of a ``Traced`` (or empty strings for ``None``)."""
    if traced is None:
        return "", "", "", "", False
    prov = traced.provenance
    return _s(traced.value), _s(traced.unit), _s(prov.kind), _s(prov.tool), bool(getattr(traced, "needs_verification", False))


def evidence_state(ev: Evidence) -> str:
    """Where an evidence file stands right now: re-hashed against the recorded hash."""
    if not ev.path:
        return EVIDENCE_NO_PATH
    p = Path(ev.path)
    if not p.is_file():
        return EVIDENCE_MISSING
    if ev.content_hash is None:
        return EVIDENCE_NO_HASH
    return EVIDENCE_OK if sha256_of_file(p) == ev.content_hash else CHANGED_ON_DISK


def artifact_disk_state(art: ArtifactRef) -> str:
    on_disk = art.disk_hash()
    if on_disk is None:
        return MISSING_ON_DISK
    return ON_DISK if on_disk == art.content_hash else CHANGED_ON_DISK


def answer_command(ir_path: Path, q: MissingInformation) -> str:
    """The ``ai-eda run ... --answer key=value`` line that answers ``q``; the value is a placeholder the human fills in."""
    if q.source == "llm" or not q.options:
        value = "<value>"
    else:
        value = "<" + "|".join(q.options) + ">"
    return f"ai-eda run {shlex.quote(str(ir_path))} --answer {q.key}={value}"


class _Freshness:
    """The ``ir_hash`` label of a validation result, given what pipeline.json says about this ir.json."""

    def __init__(self, design_hash: str, results_before: int | None) -> None:
        self.design_hash = design_hash
        self.results_before = results_before  # None: no record describes this ir.json

    def label(self, r: ValidationResult, index: int) -> str:
        if r.ir_hash is None:
            if self.results_before is None:
                return UNSTAMPED_UNKNOWN
            return UNSTAMPED_PRODUCED if index >= self.results_before else UNSTAMPED_CARRIED
        if r.ir_hash == self.design_hash:
            return FRESH
        return f"{STALE} (IR {r.ir_hash[:16]})"


def _validation_row(r: ValidationResult, index: int, fresh: _Freshness, kinds_by_hash: dict[str, str]) -> ValidationRow:
    d = r.model_dump(mode="json")
    return ValidationRow(
        index=index,
        check_id=r.check_id,
        status=d["status"],
        message=r.message,
        tool=_s(r.tool),
        tool_version=_s(r.tool_version),
        opinion=r.status is ValidationStatus.PASS and not r.is_tool_backed,
        ir_hash=_s(r.ir_hash),
        freshness=fresh.label(r, index),
        artifact_hash=_s(r.artifact_hash),
        artifact_kind=kinds_by_hash.get(r.artifact_hash or "", ""),
        evidence=[
            EvidenceRow(description=e.description, path=_s(e.path), url=_s(e.url), content_hash=_s(e.content_hash), state=evidence_state(e))
            for e in r.evidence
        ],
        details_json=_json(d["details"]) if r.details else "",
        note=MODEL_OUTPUT if r.check_id == "requirements.extraction" else "",
        timestamp=d["timestamp"],
    )


# --- sections ------------------------------------------------------------------


def _meta(ir: CircuitIR, ir_path: Path, workdir: Path, record: PipelineRecord | None, record_error: str | None, ir_sha: str) -> MetaSection:
    p = ir.project.model_dump(mode="json")
    pipeline_path = workdir / PIPELINE_FILE
    has_file = pipeline_path.is_file()
    if record is not None:
        note = PIPELINE_DESCRIBES_IR if record.ir_file_sha256 == ir_sha else PIPELINE_STALE_IR
    elif record_error is not None:
        note = record_error
    else:
        note = f"no {PIPELINE_FILE} in {workdir}: run `ai-eda run` to record stage outcomes"
    return MetaSection(
        project_id=ir.project.id,
        project_name=ir.project.name,
        description=ir.project.description,
        created_at=p["created_at"],
        workdir=_s(ir.project.workdir),
        schema_version=ir.schema_version,
        ai_eda_version=__version__,
        design_hash=ir.content_hash(),
        ir_file_path=str(ir_path),
        ir_file_sha256=ir_sha,
        pipeline_file_path=str(pipeline_path) if has_file else None,
        pipeline_file_sha256=sha256_of_file(pipeline_path) if has_file else None,
        pipeline_note=note,
    )


def _stages(record: PipelineRecord, design_hash: str) -> StagesSection:
    dumped = record.model_dump(mode="json")
    outcomes = {o.stage: (o, d) for o, d in zip(record.state.outcomes, dumped["state"]["outcomes"])}
    rows: list[StageRow] = []
    for stage in STAGE_ORDER:
        if stage in outcomes:
            o, d = outcomes[stage]
            rows.append(StageRow(stage=str(stage), status=d["status"], message=o.message, at=d["at"], questions=len(o.questions), reached=True))
        else:
            rows.append(StageRow(stage=str(stage), status="", message="not reached", at="", questions=0, reached=False))
    label = "recorded for the current IR" if record.ir_hash == design_hash else f"recorded for an earlier IR version (IR {record.ir_hash[:16]})"
    return StagesSection(
        rows=rows,
        run_ir_hash=record.ir_hash,
        run_hash_label=label,
        blocked=record.state.blocked,
        current=_s(record.state.current) or None,
        aborted=record.aborted,
        aborted_stage=_s(record.aborted_stage) or None,
        results_before=record.results_before,
        ai_eda_version=record.ai_eda_version,
        ir_path=record.ir_path,
    )


def _requirements(ir: CircuitIR) -> RequirementsSection:
    rs = ir.requirements
    rows: list[RequirementRow] = []
    for req in rs.requirements:
        value, unit, kind, _tool, needs = _traced_fields(req.value)
        rows.append(
            RequirementRow(
                id=req.id, text=req.text, kind=str(req.kind), status=str(req.status), category=req.category,
                value=value, unit=unit, provenance_kind=kind, needs_verification=needs,
            )
        )
    params: list[ParameterRow] = []
    for key in sorted(ir.parameters):
        value, unit, kind, tool, _needs = _traced_fields(ir.parameters[key])
        params.append(ParameterRow(key=key, value=value, unit=unit, provenance_kind=kind, tool=tool))
    extraction = [
        ExtractionRow(request_hash=h, confirmed=bool(entry.get("confirmed", False)), presented=bool(entry.get("presented", False)))
        for h, entry in sorted(rs.extraction_cache.items())
        if isinstance(entry, dict)
    ]
    return RequirementsSection(
        raw_input=rs.raw_input,
        corrections=list(rs.corrections),
        rows=rows,
        parameters=params,
        conflicts=[f"{', '.join(c.requirement_ids)}: {c.description}" for c in rs.conflicts],
        extraction=extraction,
    )


def _question_row(q: MissingInformation, origin: str, ir_path: Path) -> QuestionRow:
    return QuestionRow(
        key=q.key, question=q.question, required=q.required, options=list(q.options), rationale=q.rationale, source=q.source,
        source_label="model question" if q.source == "llm" else "system", origin=origin, command=answer_command(ir_path, q),
    )


def _questions(ir: CircuitIR, record: PipelineRecord | None, ir_path: Path) -> list[QuestionRow]:
    rows = [_question_row(q, "ir.requirements.missing", ir_path) for q in ir.requirements.missing]
    if record is not None:
        for o in record.state.outcomes:
            rows.extend(_question_row(q, f"stage {o.stage}", ir_path) for q in o.questions)
    return rows


def _validation(ir: CircuitIR, fresh: _Freshness, kinds_by_hash: dict[str, str]) -> ValidationSection:
    results = ir.validation.results
    latest_index: dict[str, int] = {}
    for i, r in enumerate(results):
        latest_index[r.check_id] = i
    latest = [_validation_row(results[i], i, fresh, kinds_by_hash) for _cid, i in sorted(latest_index.items())]
    history = [
        HistoryEntry(check_id=r.check_id, index=i, status=str(r.status), timestamp=r.model_dump(mode="json")["timestamp"], message=r.message)
        for i, r in sorted(enumerate(results), key=lambda ir_: (ir_[1].check_id, ir_[0]))
    ]
    return ValidationSection(latest=latest, history=history, aggregate=str(ir.validation.overall()), aggregate_note=AGGREGATE_NOTE)


def _artifacts(ir: CircuitIR, design_hash: str) -> list[ArtifactRow]:
    rows: list[ArtifactRow] = []
    for kind in sorted(ir.artifacts, key=str):
        art = ir.artifacts[kind]
        d = art.model_dump(mode="json")
        rows.append(
            ArtifactRow(
                kind=str(kind), path=art.path, files=len(art.files), generator=_s(art.generator), generator_version=_s(art.generator_version),
                content_hash=_s(art.content_hash), generated_from_ir_hash=_s(art.generated_from_ir_hash),
                freshness=STALE if art.is_stale(design_hash) else FRESH, disk=artifact_disk_state(art), notes=list(art.notes), created_at=d["created_at"],
            )
        )
    return rows


def _review(ir: CircuitIR, fresh: _Freshness) -> ReviewSection:
    latest = ir.validation.latest_by_check()
    index = {r.check_id: i for i, r in enumerate(ir.validation.results)}
    rows: list[ReviewRow] = []
    counts: dict[str, int] = {}
    for area in ReviewArea:
        r = latest.get(str(area))
        if r is None:
            rows.append(ReviewRow(area=str(area), status="", message="not reviewed", freshness="", evidence=0, timestamp=""))
            continue
        status = str(r.status)
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            ReviewRow(area=str(area), status=status, message=r.message, freshness=fresh.label(r, index[r.check_id]), evidence=len(r.evidence),
                      timestamp=r.model_dump(mode="json")["timestamp"])
        )
    return ReviewSection(rows=rows, counts=dict(sorted(counts.items())), note="latest rows are the repair loop's final review when repair ran")


def _repair(ir: CircuitIR, fresh: _Freshness) -> RepairSection | None:
    r = ir.validation.latest("repair.loop")
    if r is None:
        return None
    index = max(i for i, x in enumerate(ir.validation.results) if x.check_id == "repair.loop")
    d = r.model_dump(mode="json")
    details = d["details"]
    return RepairSection(
        status=d["status"], message=r.message, freshness=fresh.label(r, index),
        iterations=_s(details.get("iterations")), stopped_reason=_s(details.get("stopped_reason")),
        actions_json=_json(details.get("actions", [])), unresolved_json=_json(details.get("unresolved", [])),
        final_review_json=_json(details.get("final_review", {})), timestamp=d["timestamp"],
    )


def _domain(title: str, prefixes: tuple[str, ...], latest: list[ValidationRow], ir_part: object, note: str) -> DomainSection:
    matching = [row for row in latest if row.check_id.startswith(prefixes)]
    return DomainSection(title=title, prefixes=list(prefixes), results=matching, ir_json=_json(ir_part), note=note)


def _release(record: PipelineRecord | None) -> ReleaseSection:
    if record is None:
        return ReleaseSection(status=None, message="", reasons=[], at="", note=NO_RECORDED_RUN)
    outcome = record.state.outcome(Stage.RELEASE)
    if outcome is None:
        why = f"last run aborted: {record.aborted}" if record.aborted else "the last run stopped before RELEASE"
        return ReleaseSection(status=None, message="", reasons=[], at="", note=f"{NO_RECORDED_RUN.split(':')[0]} ({why})")
    d = outcome.model_dump(mode="json")
    if outcome.status is ValidationStatus.PASS:
        reasons = [outcome.message]  # 'evidence-backed release', verbatim
    else:
        reasons = [x for x in outcome.message.removeprefix(RELEASE_PREFIX).split("; ") if x]
    return ReleaseSection(status=d["status"], message=outcome.message, reasons=reasons, at=d["at"], note="recorded RELEASE outcome of the last run")


# --- entry point -----------------------------------------------------------------


def build_report_data(ir: CircuitIR, ir_path: Path, workdir: Path) -> ReportData:
    """Everything the report shows about ``ir`` (loaded from ``ir_path``) and ``<workdir>/pipeline.json``; nothing is written."""
    ir_path = Path(ir_path)
    ir_sha = sha256_of_file(ir_path)
    design_hash = ir.content_hash()
    record: PipelineRecord | None = None
    record_error: str | None = None
    try:
        record = load_pipeline_record(workdir)
    except PipelineRecordError as e:
        record_error = str(e)
    describes = record is not None and record.ir_file_sha256 == ir_sha
    fresh = _Freshness(design_hash, record.results_before if describes and record is not None else None)
    kinds_by_hash = {art.content_hash: str(kind) for kind, art in ir.artifacts.items() if art.content_hash}
    validation = _validation(ir, fresh, kinds_by_hash)
    stages = _stages(record, design_hash) if record is not None else None
    if record is None:
        stages_reason = record_error or f"no {PIPELINE_FILE} in {workdir}: run `ai-eda run` to record stage outcomes"
    else:
        stages_reason = None
    return ReportData(
        meta=_meta(ir, ir_path, workdir, record, record_error, ir_sha),
        stages=stages,
        stages_reason=stages_reason,
        requirements=_requirements(ir),
        questions=_questions(ir, record, ir_path),
        validation=validation,
        artifacts=_artifacts(ir, design_hash),
        review=_review(ir, fresh),
        repair=_repair(ir, fresh),
        regulatory=_domain("Regulatory", REGULATORY_PREFIXES, validation.latest, ir.regulatory.model_dump(mode="json"),
                           "latest regulatory.* results; ir.regulatory as stored (a ProposedRegulation is model output until accepted)"),
        components=_domain("Components", COMPONENT_PREFIXES, validation.latest, [c.model_dump(mode="json") for c in ir.components],
                           "latest component.* results; ir.components as stored (a tag such as mpn authoritative / library verified is a claim, "
                           "the component.existence.<ref> row is what was checked)"),
        simulation=_domain("Simulation", SIMULATION_PREFIXES, validation.latest,
                           ir.simulation.model_dump(mode="json") if ir.simulation is not None else None,
                           "latest spice / domain.analog.* results; ir.simulation as stored"),
        release=_release(record),
    )
