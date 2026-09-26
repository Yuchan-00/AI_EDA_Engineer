"""The last run's stage outcomes on disk: ``<workdir>/pipeline.json``.

Invariant: ``pipeline.json`` is a *run log* in the same class as
``ir.validation`` - state about a run, never design content. It is never
registered as an artifact, never enters the design hash, and it is written
only by ``ai-eda run`` (:func:`save_pipeline_record`) after the IR was saved.
A record pins what it describes twice: ``ir_hash`` (the design view the run
ended on) and ``ir_file_sha256`` (the bytes ``ir.save`` wrote), so a report
can tell an outcome recorded for *this* ir.json from one recorded for an
earlier save. ``results_before`` is the index in ``ir.validation.results``
where the recorded run started - the number RELEASE uses to tell a result
the run produced from one carried over from an earlier run.

An aborted run stores only the exception's type name and the stage that was
running: no message, no traceback (an error text can embed a URL or a
header). Loading is as strict as :meth:`ai_eda.ir.CircuitIR.load`: another
``schema_version`` or a key the models would drop is refused with
:class:`PipelineRecordError`, never rendered as a truncated record. The
file is read exactly once: :func:`load_pipeline_record` returns the record
together with the sha256 of the very bytes it parsed, so a report never
labels one version of the file with the hash of another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from ai_eda import __version__
from ai_eda.errors import AiEdaError
from ai_eda.ir.project import CircuitIR, unknown_keys
from ai_eda.workflow.orchestrator import PipelineState
from ai_eda.workflow.stages import Stage

PIPELINE_FILE = "pipeline.json"
PIPELINE_SCHEMA_VERSION = "1"


class PipelineRecordError(AiEdaError):
    """``pipeline.json`` exists but is not a record this code can read faithfully.

    ``file_sha256`` is the hash of the bytes that were refused (``None`` when
    they could not be read at all), so a report can still name the file it
    looked at without reading it a second time.
    """

    def __init__(self, message: str, *, file_sha256: str | None = None) -> None:
        super().__init__(message)
        self.file_sha256 = file_sha256


def sha256_of_bytes(raw: bytes) -> str:
    """``sha256:<hex>`` of ``raw``."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_of_file(path: str | Path) -> str:
    """``sha256:<hex>`` of the bytes at ``path`` (the file must exist)."""
    return sha256_of_bytes(Path(path).read_bytes())


class PipelineRecord(BaseModel):
    """What ``ai-eda run`` recorded about its last run of a project."""

    schema_version: str = PIPELINE_SCHEMA_VERSION
    ai_eda_version: str
    ir_path: str
    #: the design hash the run ended on (``CircuitIR.content_hash`` of the saved IR)
    ir_hash: str
    #: sha256 of the ir.json bytes ``ir.save`` wrote at the end of the run; ``None`` when that save failed
    ir_file_sha256: str | None = None
    #: ``len(ir.validation.results)`` right before the run started: earlier results were carried over
    results_before: int
    state: PipelineState
    #: the exception's type name when the run died before finishing; then ``state`` holds the partial outcomes
    aborted: str | None = None
    #: the stage that was running when the run died
    aborted_stage: Stage | None = None


def save_pipeline_record(
    state: PipelineState,
    ir: CircuitIR,
    ir_path: str | Path,
    workdir: Path,
    *,
    results_before: int,
    ir_file_sha256: str | None,
    aborted: str | None = None,
) -> Path:
    """Write ``workdir / PIPELINE_FILE`` describing ``state`` and return its path."""
    record = PipelineRecord(
        ai_eda_version=__version__,
        ir_path=str(ir_path),
        ir_hash=ir.content_hash(),
        ir_file_sha256=ir_file_sha256,
        results_before=results_before,
        state=state,
        aborted=aborted,
        aborted_stage=state.current if aborted is not None else None,
    )
    path = Path(workdir) / PIPELINE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


class LoadedPipelineRecord(NamedTuple):
    """A record and the ``sha256:<hex>`` of the exact bytes it was parsed from."""

    record: PipelineRecord
    file_sha256: str


def load_pipeline_record(workdir: Path) -> LoadedPipelineRecord | None:
    """The record in ``workdir`` with the hash of the bytes it was parsed from, ``None`` when there is no file,
    :class:`PipelineRecordError` when it cannot be read faithfully (the error carries the hash of the refused bytes)."""
    path = Path(workdir) / PIPELINE_FILE
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError as e:
        raise PipelineRecordError(f"{path}: not readable as JSON: {e}") from e
    file_sha256 = sha256_of_bytes(data)
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as e:
        raise PipelineRecordError(f"{path}: not readable as JSON: {e}", file_sha256=file_sha256) from e
    if not isinstance(raw, dict):
        raise PipelineRecordError(f"{path}: not a pipeline record object", file_sha256=file_sha256)
    version = raw.get("schema_version")
    if version != PIPELINE_SCHEMA_VERSION:
        raise PipelineRecordError(
            f"{path}: schema_version {version!r} is not {PIPELINE_SCHEMA_VERSION!r} (this code reads no other version)",
            file_sha256=file_sha256,
        )
    try:
        record = PipelineRecord.model_validate(raw)
    except ValidationError as e:
        raise PipelineRecordError(
            f"{path}: not a valid pipeline record: {e.errors()[0].get('msg', e) if e.errors() else e}", file_sha256=file_sha256
        ) from e
    unknown = unknown_keys(raw, record.model_dump(mode="json"))
    if unknown:
        raise PipelineRecordError(f"{path}: unknown key(s) the record models would drop: {', '.join(unknown)}", file_sha256=file_sha256)
    return LoadedPipelineRecord(record, file_sha256)
