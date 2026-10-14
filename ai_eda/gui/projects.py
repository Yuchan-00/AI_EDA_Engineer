"""The projects under the GUI's projects root: one folder per project, found by the ``ir.json`` it holds.

Invariant: a project is a folder directly under the root that holds
``ir.json``. It is listed by that folder, and its working directory is that
folder - the one ``ai-eda run`` uses (:func:`ai_eda.workdir.project_workdir`),
so the GUI and the CLI never disagree on where a run writes. A project whose
recorded workdir is another folder, or whose registered artifacts lie
outside its folder (a copied or moved project,
:func:`ai_eda.workdir.workdir_mismatch`), is listed with
``workdir_mismatch`` and a Korean warning naming ``ai-eda relocate``, never
hidden: its previews, files and zip come from its own folder only (never
the recorded one), its project view reads no file through the IR's
locators outside that folder (artifact and evidence states say "outside
this project folder (not read)", the SPICE summary is not checked against
the IR), its report tab is refused as ``ai-eda report`` refuses it (409,
the relocate remedy), its last run is its own folder's pipeline.json, and a
run of it is refused (the CLI refuses it too). The name rule
:data:`PROJECT_NAME_PATTERN` is for the folders the GUI *creates*; an
existing folder opens under any name that is one safe path component
(:func:`safe_part`), since ``ai-eda new`` accepts any name (``osc.v2``,
``발진기``). A folder whose name is not such a component (a Windows device
name, a ``\\`` or ``:``, a control character), that is a symbolic link,
whose ir.json is a symbolic link or cannot be read faithfully
(:meth:`~ai_eda.ir.CircuitIR.load` refuses it), or whose relative recorded
workdir ``project_workdir`` refuses, is listed with ``error`` and nothing else.

The last run is *copied* from ``<workdir>/pipeline.json`` through
:func:`~ai_eda.report.pipeline_log.load_pipeline_record`: whether it
blocked or aborted, the last stage with its recorded status and message,
and the questions exactly as recorded - no status is computed here. The
only labels added are :func:`~ai_eda.report.data.run_hash_label`, a fact
about hashes, and :data:`~ai_eda.report.data.PIPELINE_OTHER_IR` when the
record's ``ir_file`` names another ir.json (a copy's inherited record has the
same design hash, so the hash label alone would call it current).

The one write is :meth:`ProjectsRoot.create`: it claims a new folder and
calls :func:`ai_eda.cli.new_project`, the code path of ``ai-eda new``.
Nothing here edits, renames or deletes a project (deleting would be a
``SYSTEM_DELETE``; there is no such operation in the GUI).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ai_eda.ir.requirements import MissingInformation

#: the file whose presence makes a folder a project
IR_FILE = "ir.json"
#: a project name (the folder name): an ASCII letter or digit, then up to 63 letters, digits, ``_`` or ``-``
PROJECT_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
PROJECT_NAME_RE = re.compile(PROJECT_NAME_PATTERN)
#: the rule in the words the page shows
PROJECT_NAME_RULE = "영문자나 숫자로 시작하고 영문자, 숫자, '_', '-'만 쓰는 64자 이하의 이름"
#: names Windows reserves for devices (a folder cannot carry them there, whatever the case)
WINDOWS_RESERVED_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class ProjectError(ValueError):
    """A project the GUI refuses to create (a bad name, an existing folder, a request that is not text): the server's 400."""


class ProjectNotFoundError(LookupError):
    """No project of that name under the root, or a name no project can have: the server's 404."""


class RunSummary(BaseModel):
    """What ``<workdir>/pipeline.json`` records about the project's last run; every value copied, none computed.

    ``error`` alone is set when the file exists but cannot be read faithfully
    (:class:`~ai_eda.report.pipeline_log.PipelineRecordError`).
    """

    blocked: bool = False
    #: the exception type name of a run that died before finishing
    aborted: str | None = None
    aborted_stage: str | None = None
    #: the last recorded stage and its recorded status / message
    last_stage: str | None = None
    last_status: str | None = None
    last_message: str = ""
    #: how many stage outcomes the record holds
    stages_recorded: int = 0
    #: the required questions (the run stopped on them) and the optional ones, as recorded
    open_questions: list[MissingInformation] = Field(default_factory=list)
    optional_questions: list[MissingInformation] = Field(default_factory=list)
    #: whether the run ended on the current design hash (:func:`ai_eda.report.data.run_hash_label`), or
    #: ``PIPELINE_OTHER_IR`` when the record describes another ir.json
    run_hash_label: str = ""
    #: the ir.json the record describes (its ``ir_file``) when that is not this project's ir.json, else ``None``
    other_ir: str | None = None
    error: str | None = None


class ProjectInfo(BaseModel):
    """One project folder under the root. With ``error`` set nothing but ``name`` / ``folder`` / ``ir_path`` is filled."""

    name: str
    folder: Path
    ir_path: Path
    #: the project's own folder, where ``ai-eda run`` writes (:func:`ai_eda.workdir.project_workdir`) - also for a mismatch
    workdir: Path | None = None
    #: ``project.workdir`` as ir.json records it
    recorded_workdir: str | None = None
    #: the recorded workdir is absolute and names another folder: listed, confined to its own folder, never run
    workdir_mismatch: bool = False
    workdir_exists: bool = False
    #: a Korean sentence when the project is listed with a caveat (a mismatched recorded workdir: names ``ai-eda relocate``)
    warning: str | None = None
    #: ``CircuitIR.content_hash`` of the ir.json as read now
    design_hash: str | None = None
    last_run: RunSummary | None = None
    error: str | None = None


def valid_project_name(name: object) -> bool:
    """Whether ``name`` is a project name the GUI creates (:data:`PROJECT_NAME_PATTERN`, Windows device names excluded)."""
    return isinstance(name, str) and PROJECT_NAME_RE.fullmatch(name) is not None and name.lower() not in WINDOWS_RESERVED_NAMES


def safe_part(part: object) -> bool:
    """Whether ``part`` may be one component of a path: text, not ``.`` / ``..``, at most 255 characters, no separator, drive colon or control character, no Windows device name."""
    if not isinstance(part, str) or not part or part in (".", "..") or len(part) > 255:
        return False
    if any(c in part for c in "/\\:") or _CONTROL_RE.search(part):
        return False
    return part.split(".", 1)[0].strip().lower() not in WINDOWS_RESERVED_NAMES


def openable_project_name(name: object) -> bool:
    """Whether an existing folder called ``name`` opens as a project: one safe path component (:func:`safe_part`), whatever ``ai-eda new`` named it."""
    return safe_part(name)


def _one_line(e: BaseException) -> str:
    """The first line of an error (a pydantic error by its first location and message), for a list entry."""
    if isinstance(e, ValidationError) and e.errors():
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        return f"{loc}: {first.get('msg', '')}" if loc else str(first.get("msg", ""))
    lines = str(e).strip().splitlines()
    return lines[0] if lines else type(e).__name__


def run_summary(workdir: Path, design_hash: str, ir_path: Path | None = None) -> RunSummary | None:
    """The last run recorded in ``workdir`` (``None`` without ``pipeline.json``), copied from the record.

    With ``ir_path``, a record whose ``ir_file`` names another ir.json is
    labelled :data:`~ai_eda.report.data.PIPELINE_OTHER_IR` and carries that path in ``other_ir``.
    """
    from ai_eda.report.data import PIPELINE_OTHER_IR, describes_other_ir, run_hash_label
    from ai_eda.report.pipeline_log import PipelineRecordError, load_pipeline_record

    try:
        loaded = load_pipeline_record(workdir)
    except PipelineRecordError as e:
        return RunSummary(error=_one_line(e))
    if loaded is None:
        return None
    record = loaded.record
    state = record.state
    last = state.outcomes[-1] if state.outcomes else None
    other = ir_path is not None and describes_other_ir(record, ir_path)
    return RunSummary(
        blocked=state.blocked,
        aborted=record.aborted,
        aborted_stage=str(record.aborted_stage) if record.aborted_stage is not None else None,
        last_stage=str(last.stage) if last is not None else None,
        last_status=str(last.status) if last is not None else None,
        last_message=last.message if last is not None else "",
        stages_recorded=len(state.outcomes),
        open_questions=list(state.open_questions),
        optional_questions=list(state.optional_questions),
        run_hash_label=PIPELINE_OTHER_IR if other else run_hash_label(record, design_hash),
        other_ir=record.ir_file if other else None,
    )


class ProjectsRoot:
    """The folder that holds the projects (the CLI default is ``projects`` under the current directory, as for ``new``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def list(self) -> list[ProjectInfo]:
        """Every folder under the root that holds ``ir.json``, by name; a missing root lists nothing."""
        if not self.root.is_dir():
            return []
        projects: list[ProjectInfo] = []
        for entry in sorted(self.root.iterdir(), key=lambda p: p.name):
            if _holds_ir(entry):
                projects.append(self.info(entry))
        return projects

    def get(self, name: str) -> ProjectInfo:
        """The project ``name`` as it is on disk now; :class:`ProjectNotFoundError` for a name no folder under the root can have or no such project."""
        if not openable_project_name(name):
            raise ProjectNotFoundError(f"프로젝트 이름이 아닙니다: {name!r}")
        folder = self.root / name
        if not _holds_ir(folder):
            raise ProjectNotFoundError(f"프로젝트가 없습니다: {name}")
        return self.info(folder)

    def create(self, name: str, request: str | None = "") -> ProjectInfo:
        """Create ``<root>/<name>/ir.json`` through :func:`ai_eda.cli.new_project` (what ``ai-eda new`` writes) and return the new project.

        :class:`ProjectError` for a name that breaks the rule, a folder (or
        file, or link) that already exists, or a request that is not text;
        nothing is written then. The folder is claimed with an exclusive
        ``mkdir`` so two concurrent requests for one name cannot both write.
        """
        from ai_eda.cli import new_project

        if not valid_project_name(name):
            raise ProjectError(f"프로젝트 이름은 {PROJECT_NAME_RULE}이어야 합니다 (받은 값: {name!r})")
        if request is not None and not isinstance(request, str):
            raise ProjectError("요청문은 문자열이어야 합니다")
        folder = self.root / name
        if folder.is_symlink() or folder.exists():
            raise ProjectError(f"이미 있는 폴더입니다: {name} (다른 이름을 쓰십시오; 기존 폴더는 건드리지 않습니다)")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError as e:
            raise ProjectError(f"이미 있는 폴더입니다: {name} (다른 이름을 쓰십시오; 기존 폴더는 건드리지 않습니다)") from e
        new_project(name, request or "", folder)
        return self.info(folder)

    def info(self, folder: Path) -> ProjectInfo:
        """The :class:`ProjectInfo` of one project folder (reads its ir.json and pipeline.json once each; writes nothing).

        A recorded workdir naming another folder is no error: the project's
        workdir is still its own folder (everything shown comes from there),
        ``workdir_mismatch`` is set and the warning names ``ai-eda relocate``.
        """
        from ai_eda.ir import CircuitIR
        from ai_eda.workdir import project_workdir, workdir_mismatch

        folder = Path(folder)
        ir_path = folder / IR_FILE
        base = {"name": folder.name, "folder": folder, "ir_path": ir_path}
        if not openable_project_name(folder.name):
            return ProjectInfo(**base, error="폴더 이름을 경로의 한 칸으로 쓸 수 없어(Windows 장치 이름, '\\', ':', 제어 문자) 열지 않습니다")
        if folder.is_symlink():
            return ProjectInfo(**base, error="프로젝트 폴더가 심볼릭 링크라서 열지 않습니다")
        if ir_path.is_symlink():
            return ProjectInfo(**base, error="ir.json이 심볼릭 링크라서 읽지 않습니다")
        try:
            ir = CircuitIR.load(ir_path)
            design_hash = ir.content_hash()
            mismatch = workdir_mismatch(ir, ir_path)
            # the folder itself for a mismatch (never the recorded one); a refused relative workdir raises: error entry
            workdir = mismatch[1] if mismatch is not None else project_workdir(ir, ir_path)
        except Exception as e:  # noqa: BLE001 - one unreadable project is listed with its reason, never ends the listing
            return ProjectInfo(**base, error=f"ir.json을 읽을 수 없습니다: {_one_line(e)}")
        exists = workdir.is_dir()
        warning = None
        if mismatch is not None:
            warning = mismatch_warning(mismatch[0], folder, ir_path, artifacts=mismatch[0] != ir.project.workdir)
        try:
            last_run = run_summary(workdir, design_hash, ir_path) if exists else None
        except OSError as e:
            last_run = RunSummary(error=_one_line(e))
        return ProjectInfo(
            **base, workdir=workdir, recorded_workdir=ir.project.workdir, workdir_mismatch=mismatch is not None, workdir_exists=exists,
            warning=warning, design_hash=design_hash, last_run=last_run,
        )


def mismatch_warning(recorded: str, folder: Path, ir_path: Path, *, artifacts: bool = False) -> str:
    """The Korean warning of a copied or moved project (its recorded workdir, or its registered artifacts, name another folder): not run, ``ai-eda relocate`` fixes it."""
    what = "ir.json에 등록된 산출물의 폴더" if artifacts else "ir.json에 기록된 작업 폴더"
    return (
        f"{what}({recorded})가 이 프로젝트 폴더({folder})와 다릅니다(복사하거나 옮긴 프로젝트). "
        f"이 상태로는 실행하지 않습니다: 명령줄에서 `ai-eda relocate {ir_path}`로 작업 폴더를 이 폴더로 다시 기록한 뒤 실행하십시오."
    )


def _holds_ir(folder: Path) -> bool:
    """Whether ``folder`` is a directory holding an ``ir.json`` entry (a file or a link; the link is refused later, visibly)."""
    try:
        ir_path = folder / IR_FILE
        return folder.is_dir() and (ir_path.is_file() or ir_path.is_symlink())
    except OSError:
        return False
