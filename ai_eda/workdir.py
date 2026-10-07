"""A project's working folder: where its outputs go, the per-project lock, and relocating a copied or moved project.

Invariant: a project's outputs are written only into the folder that holds
its ir.json; a recorded absolute workdir that names another folder is
refused, never followed and never silently replaced.

* :func:`project_workdir` is the one answer to "where does this project
  write" (``run`` / ``review`` / ``report`` / ``serve`` / ``stage-reports``
  and the GUI): always the ir.json's own directory. ``project.workdir`` is
  a record, not an instruction: when it is absolute and names another
  folder (the project was copied or moved, W1 / W2 of the 900 MHz test
  report) the command refuses with :class:`WorkdirMismatchError` - an
  :class:`~ai_eda.errors.IRSchemaError`, so every existing ``except
  IRSchemaError`` maps it to exit 2 - and nothing is run, read from the
  other folder or created in it. A workdir recorded as absolute under the
  *other* operating system's rules (``C:\\proj`` read on Linux, ``/proj``
  read on Windows - this project is built on Windows and tested on Linux)
  is an absolute other folder too, never "relative". A relative workdir
  keeps its legacy rule: accepted only when the ir.json's directory ends
  with it, else refused as ambiguous. Whatever the recorded workdir says -
  none, a relative one the folder ends with, this very folder - an IR whose
  registered artifacts (``ArtifactRef.path`` / ``files``, always compiled
  into the workdir) lie by absolute path outside the ir.json's folder is a
  copy too (:func:`stray_artifacts`) and is refused the same way: its review
  would read another folder's files. (Library, datasheet and source-document
  locators may legitimately lie outside the project and are not judged.)
* :func:`relocate_project` (``ai-eda relocate``) is the only way the
  recorded workdir changes: it records the ir.json's folder and rebases the
  locator paths under the old folder - the recorded absolute workdir, else
  the folder the artifacts show (:func:`infer_old_folder`: for a relative
  workdir the ancestor of the artifact paths that ends with it, for none
  the one folder holding the root-level artifacts), else the one given with
  ``--from`` - (``ArtifactRef.path`` / ``files``,
  ``Evidence.path``, ``SourceRef.document_path``, ``LibraryRef.library_path``,
  ``RegulatoryProvenance.source_document``) onto the new one - lexically, the
  old folder need not exist. Paths outside the old folder, every
  ``ValidationResult.details`` / message and ``ir.parameters`` are left as
  written (a run log is not rewritten). The ``SPICE_RESULT`` reference is
  dropped whenever the workdir changes (the file stays): ``results.json`` is
  tool output registered with its hash that names its rawfiles by absolute
  path under the old folder - rewriting it would forge evidence, keeping it
  would let readers follow those paths into the other folder - so the next
  ``run`` re-simulates. It refuses (:class:`RelocateError`, exit 2, nothing
  written) when the old folder cannot be told, or when an artifact path
  would still lie outside the new folder after rebasing - never "relocated"
  with evidence left elsewhere. Every field it touches is outside the design view
  (``project.workdir``, ``ir.artifacts``, ``ir.validation``, the locators),
  and the design hash is compared before and after: a difference raises and
  nothing is written. ``pipeline.json`` is never touched (a run log; only
  ``run`` writes it).
* :class:`ProjectLock` is the per-project lock ``run`` / ``review`` /
  ``stage-reports`` / ``relocate`` take on :data:`LOCK_FILE` in the workdir:
  non-blocking only (a second command refuses with "another ai-eda run is
  using this project", it never waits), so no wait cycle can exist - the GUI
  never takes it (it holds only its run log's ``flock``, a different file)
  and its child, the CLI, takes it itself. The lock file is created on first
  use and never deleted (deleting races with a process that opened the old
  inode); it is never an artifact, never hashed, never in the GUI's zip.
  POSIX: ``flock`` on a non-inheritable descriptor opened with
  ``O_NOFOLLOW`` (a symbolic-link lock file is refused), so kicad-cli, the
  browser or ``claude`` started by the command never keep it. Windows:
  ``msvcrt.locking`` on byte 0 (it dies with the process) - NOT measured on
  Windows. A lock file this user cannot open for writing (a read-only
  folder, another user's file) is locked through a read-only descriptor -
  a lock needs no write. A folder where no lock can be taken at all (no
  lock file and none can be created, ``ENOLCK``, ``EOPNOTSUPP``) proceeds
  unlocked with a printed note: a review of a read-only copy must still
  work.
"""

from __future__ import annotations

import errno
import ntpath
import os
import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from ai_eda.errors import AiEdaError, IRSchemaError

try:
    import fcntl
except ImportError:  # Windows: msvcrt below
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]

#: the per-project lock file in the workdir (never an artifact, never hashed, never deleted)
LOCK_FILE = ".ai-eda.lock"
#: the phrase every busy refusal carries
BUSY_PHRASE = "another ai-eda run is using this project"
#: the artifact kinds the compilers write straight into the workdir (the others go into a subfolder: gerbers, spice, 3d)
ROOT_ARTIFACT_KINDS: frozenset[str] = frozenset({"kicad_sch", "kicad_pcb", "kicad_pro", "spice_netlist", "bom", "cpl", "model_3d"})
#: the ``RelocateReport.rebased`` keys, in report order
REBASE_KEYS: tuple[str, ...] = (
    "artifact_paths", "artifact_files", "evidence", "source_documents", "library_paths", "regulatory_documents",
)


class WorkdirMismatchError(IRSchemaError):
    """``project.workdir`` is absolute and names another folder than the ir.json's own: nothing is run there.

    An :class:`~ai_eda.errors.IRSchemaError`, so every command that already
    maps that to exit 2 refuses a copied or moved project the same way.
    """

    def __init__(self, ir_path: str | Path, recorded: str, here: Path, stray: list[str] | None = None) -> None:
        self.ir_path = Path(ir_path)
        self.recorded = recorded
        self.here = here
        #: the artifact paths outside ``here`` when that (not the recorded workdir) is the mismatch
        self.stray = list(stray or [])
        super().__init__(mismatch_message(ir_path, recorded, here, self.stray))


class RelocateError(AiEdaError):
    """``ai-eda relocate`` cannot rebase every artifact into the ir.json's folder (exit 2, nothing written)."""


class ProjectLockError(AiEdaError):
    """The project lock refuses the command (exit 2): another process holds it, or the lock file is a symbolic link."""


class ProjectBusyError(ProjectLockError):
    """Another process holds the project's :data:`LOCK_FILE`."""


def mismatch_message(ir_path: str | Path, recorded: str, here: Path, stray: list[str] | None = None) -> str:
    """The refusal of a copied or moved project: both paths, "Nothing was run" and the ``ai-eda relocate`` remedy."""
    if stray:
        more = f" and {len(stray) - 1} more" if len(stray) > 1 else ""
        return (
            f"{ir_path}: its registered artifacts lie in {recorded}, not in this ir.json's directory ({here}) - e.g. {stray[0]}{more}: "
            f"the project was copied or moved, and a run or review would read or write the other folder's files. Nothing was run. "
            f"Run `ai-eda relocate {ir_path}` to rebase them onto {here} (add `--from <old folder>` if it cannot tell the old folder; "
            "the SPICE results reference is dropped and re-simulated by the next run)."
        )
    return (
        f"{ir_path}: project.workdir {recorded} is not this ir.json's directory ({here}): the project was copied or moved, "
        f"and running it would write into {recorded}. Nothing was run. Run `ai-eda relocate {ir_path}` to record {here} as the "
        "workdir (artifact and evidence paths under the old folder are rebased; the SPICE results reference is dropped and "
        f"re-simulated by the next run), or run the ir.json that is in {recorded}."
    )


# --------------------------------------------------------------------------- the workdir rule


def foreign_absolute(recorded: str) -> bool:
    """Whether ``recorded`` is absolute under the *other* operating system's rules.

    On POSIX a Windows path with a drive (or UNC share) and a root
    (``C:\\proj``, ``\\\\server\\share\\x``); on Windows a rooted path without
    a drive (``/proj``). ``Path(recorded).is_absolute()`` is False for both on
    the host, which would otherwise read them as relative.
    """
    w = PureWindowsPath(recorded)
    if os.name == "nt":
        return not w.drive and bool(w.root)
    return bool(w.drive and w.root)


def _flavour(recorded: str) -> type[PurePath] | None:
    """The pure-path class ``recorded`` is absolute in (the host's or the other OS's), or ``None`` when it is relative."""
    if Path(recorded).is_absolute():
        return type(PurePath(recorded))
    if foreign_absolute(recorded):
        return PurePosixPath if os.name == "nt" else PureWindowsPath
    return None


def _same_folder(recorded: str, here: Path) -> bool:
    """Whether the host-absolute ``recorded`` is the folder ``here`` (resolved paths equal, or the same file when both exist)."""
    try:
        if Path(recorded).resolve() == here:
            return True
    except (OSError, RuntimeError):
        pass
    try:
        return os.path.exists(recorded) and here.exists() and os.path.samefile(recorded, here)
    except OSError:
        return False


def _relative_accepted(recorded: str, here: Path) -> bool:
    """The legacy rule of a relative workdir: accepted when the ir.json's directory ends with it (``.`` and empty parts too)."""
    parts = Path(recorded).parts
    return not parts or here.parts[-len(parts):] == parts


def _recorded_mismatch(recorded: str | None, here: Path) -> bool:
    """Whether the recorded workdir is absolute (host or other OS) and not ``here``."""
    if not recorded or _flavour(recorded) is None:
        return False
    return not (Path(recorded).is_absolute() and _same_folder(recorded, here))


def _artifact_locations(ir: Any) -> list[tuple[str, str, bool]]:
    """``(kind, path, is_the_main_path)`` of every non-empty ``ArtifactRef.path`` / ``files`` entry."""
    out: list[tuple[str, str, bool]] = []
    for kind, art in getattr(ir, "artifacts", {}).items():
        if art.path:
            out.append((str(kind), art.path, True))
        out.extend((str(kind), f, False) for f in art.files if f)
    return out


def _inside(path: str, here: Path) -> bool:
    """Whether an artifact path lies in ``here`` (resolved); a relative path is not judged (True), a foreign-absolute one never does."""
    flavour = _flavour(path)
    if flavour is None:
        return True
    if not Path(path).is_absolute():
        return False  # absolute only under the other OS's rules: never below a host folder
    try:
        return Path(path).resolve().is_relative_to(here)
    except (OSError, RuntimeError, ValueError):
        return Path(os.path.normpath(path)).is_relative_to(here)


def stray_artifacts(ir: Any, here: Path) -> list[str]:
    """Every absolute ``ArtifactRef.path`` / ``files`` entry outside ``here`` (sorted, unique): what a copied project's IR still points at."""
    return sorted({p for _, p, _ in _artifact_locations(ir) if not _inside(p, here)})


def infer_old_folder(ir: Any, here: Path) -> str | None:
    """The folder a copied project's artifacts were compiled into, when the IR shows exactly one; else ``None``.

    The recorded workdir when it is absolute and another folder; for a
    relative one, the nearest ancestor of the stray artifact paths whose
    tail is that relative path (``/r1/projects/demo`` for ``projects/demo``);
    otherwise the parent of the root-level artifacts
    (:data:`ROOT_ARTIFACT_KINDS`: schematic, board, project file, netlist,
    BOM, CPL, 3D preview - compiled straight into the workdir). Several
    candidates, or none, answer ``None``: the caller asks for ``--from``.
    """
    recorded = ir.project.workdir
    if _recorded_mismatch(recorded, here):
        return recorded
    stray = [(kind, p, main) for kind, p, main in _artifact_locations(ir) if not _inside(p, here)]
    if not stray:
        return None
    candidates: set[str] = set()
    if recorded and _flavour(recorded) is None:
        parts = PurePath(recorded).parts
        for _, p, _ in stray:
            flavour = _flavour(p)
            assert flavour is not None
            for parent in flavour(p).parents:
                if parts and parent.parts[-len(parts):] == parts:
                    candidates.add(str(parent))
                    break
    else:
        for kind, p, main in stray:
            if main and kind in ROOT_ARTIFACT_KINDS:
                flavour = _flavour(p)
                assert flavour is not None
                candidates.add(str(flavour(p).parent))
    return candidates.pop() if len(candidates) == 1 else None


def workdir_mismatch(ir: Any, ir_path: str | Path) -> tuple[str, Path] | None:
    """``(recorded or old folder, here)`` when this ir.json belongs to another folder, else ``None``.

    Another folder: ``project.workdir`` is absolute (host or other OS) and is
    not the ir.json's own directory, or - whatever the recorded workdir says
    - a registered artifact lies by absolute path outside it
    (:func:`stray_artifacts`; the first element is then the folder
    :func:`infer_old_folder` finds, or a description when it finds none).
    ``here`` is the ir.json's resolved directory. The same folder is
    ``Path(recorded).resolve() == here``, or ``os.path.samefile`` when both
    exist; a path absolute only under the other OS's rules is never the same
    folder. A relative workdir the folder does not end with is no mismatch
    here (:func:`project_workdir` refuses it as relative).
    """
    here = Path(ir_path).resolve().parent
    recorded = ir.project.workdir
    if _recorded_mismatch(recorded, here):
        return recorded, here
    if stray_artifacts(ir, here):
        return infer_old_folder(ir, here) or "another folder", here
    return None


def project_workdir(ir: Any, ir_path: str | Path) -> Path:
    """The project's working directory: always the ir.json's own directory, or a refusal.

    ``None`` -> the ir.json's directory; a relative workdir the directory
    ends with (``projects/demo`` for ``.../projects/demo/ir.json``, ``.``) ->
    the same (the layout ``new`` wrote before it recorded absolute paths); a
    relative one that does not -> :class:`~ai_eda.errors.IRSchemaError`
    ("is relative ...", never resolved against the caller's cwd); an absolute
    one naming this folder -> this folder; an absolute one naming another
    folder (or absolute under the other OS's rules) ->
    :class:`WorkdirMismatchError`. In every accepted case an artifact
    registered by absolute path outside this folder (:func:`stray_artifacts`)
    -> :class:`WorkdirMismatchError` naming it (a copy whose workdir was
    never recorded, or whose relative workdir still matches the copy's
    folder tail, would otherwise be reviewed on the original's files).
    """
    here = Path(ir_path).resolve().parent
    recorded = ir.project.workdir
    if _recorded_mismatch(recorded, here):
        raise WorkdirMismatchError(ir_path, recorded, here)
    if recorded and _flavour(recorded) is None and not _relative_accepted(recorded, here):
        raise IRSchemaError(
            f"{ir_path}: project.workdir {recorded!r} is relative and does not name this ir.json's directory ({here}); "
            "record an absolute path (ai-eda new does) or remove it to use the ir.json's directory"
            f"; if the project was copied or moved, run ai-eda relocate {ir_path}"
        )
    stray = stray_artifacts(ir, here)
    if stray:
        raise WorkdirMismatchError(ir_path, infer_old_folder(ir, here) or "another folder", here, stray)
    return here


# --------------------------------------------------------------------------- the lock


class ProjectLock:
    """The non-blocking per-project lock on ``<workdir>/`` :data:`LOCK_FILE` (see the module docstring); a context manager.

    :meth:`acquire` raises :class:`ProjectBusyError` when another process
    (or another open of the file in this one) holds it and
    :class:`ProjectLockError` for a symbolic-link lock file; when no lock can
    be taken at all it proceeds and sets :attr:`note` (the caller prints it).
    """

    def __init__(self, workdir: str | Path, command: str = "run") -> None:
        self.workdir = Path(workdir)
        self.command = command
        self.path = self.workdir / LOCK_FILE
        #: why the command runs unlocked (set by :meth:`acquire`), else ``None``
        self.note: str | None = None
        self._fd: int | None = None

    def __enter__(self) -> ProjectLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _unlocked(self, err: BaseException) -> None:
        self.note = f"could not lock {self.path} ({err}): concurrent ai-eda runs of this project are not prevented"

    def _busy(self) -> ProjectBusyError:
        return ProjectBusyError(f"{self.workdir}: {BUSY_PHRASE} ({LOCK_FILE} is locked); wait for it to finish")

    def _link(self) -> ProjectLockError:
        return ProjectLockError(f"{self.workdir}: lock file {self.path} is a symbolic link; refusing to follow it (remove the link)")

    def acquire(self) -> None:
        """Take the lock without waiting (see the class docstring). Calling it while held is a no-op."""
        if self._fd is not None:
            return
        if os.name == "nt" and os.path.islink(self.path):  # no O_NOFOLLOW there
            raise self._link()
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | nofollow, 0o666)
        except OSError as e:
            if e.errno == errno.ELOOP or os.path.islink(self.path):
                raise self._link() from e
            try:  # an existing lock file this user may not write (a read-only folder, another user's file): a lock needs no write
                fd = os.open(self.path, os.O_RDONLY | nofollow)
            except OSError:
                self._unlocked(e)
                return
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                os.close(fd)
                raise self._busy() from e
            except OSError as e:  # ENOLCK / EOPNOTSUPP: a filesystem without flock
                os.close(fd)
                self._unlocked(e)
                return
        elif msvcrt is not None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)  # msvcrt.locking locks from the current position
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # past EOF of the empty file is allowed on Windows
            except OSError as e:
                os.close(fd)
                raise self._busy() from e
        else:
            os.close(fd)
            self._unlocked(OSError("no file locking on this platform"))
            return
        self._fd = fd

    def release(self) -> None:
        """Release the lock (the file stays). Idempotent."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # closing the descriptor below releases it anyway
        finally:
            os.close(fd)


def is_locked(workdir: str | Path) -> bool:
    """Whether another process holds ``<workdir>/`` :data:`LOCK_FILE` now (a probe; POSIX only, ``False`` on Windows).

    The probe takes a shared ``flock`` for an instant, and a command that
    tries its exclusive lock at that very instant refuses as busy. So it is
    called **only** where a user starts a run (``RunManager.start`` - a real
    conflict anyway), never on a listing, status or polling path. On Windows
    it answers ``False``: the CLI child refuses by itself there. An absent
    file is not locked; the probe never creates it.
    """
    if fcntl is None:
        return False
    try:
        fd = os.open(Path(workdir) / LOCK_FILE, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)  # releases the probe's own shared lock
    return False


# --------------------------------------------------------------------------- relocation


@dataclass
class RelocateReport:
    """What :func:`relocate_project` did (or, with ``dry_run``, would do)."""

    ir_path: Path
    old_workdir: str | None
    new_workdir: Path
    #: whether ``project.workdir`` is (or would be) re-recorded
    changed: bool
    #: rebased locator paths by kind (:data:`REBASE_KEYS`)
    rebased: dict[str, int] = field(default_factory=lambda: {k: 0 for k in REBASE_KEYS})
    #: locator paths that were not under the old workdir (left as written)
    untouched: int = 0
    #: ``["spice_result"]`` when that reference was removed from ``ir.artifacts``
    dropped: list[str] = field(default_factory=list)
    #: rebased paths with no file at the new place (informational, not an error)
    missing_on_disk: list[str] = field(default_factory=list)
    #: ``CircuitIR.content_hash`` - equal before and after (asserted)
    design_hash: str = ""
    #: why nothing was rebased although the workdir changes (a relative old workdir cannot be matched)
    notes: list[str] = field(default_factory=list)
    dry_run: bool = False
    #: the folder the locators were rebased from (the recorded absolute workdir, the one the artifacts show, or ``--from``)
    rebased_from: str | None = None

    @property
    def rebased_total(self) -> int:
        return sum(self.rebased.values())


def _rebase_one(value: str, flavour: type[PurePath], old: str, new: Path) -> str | None:
    """``value`` moved from under ``old`` to under ``new`` (host separators), or ``None`` when it is not lexically under ``old``.

    Compared as ``normpath`` strings of ``flavour`` (the old folder may not
    exist, so nothing is resolved); a Windows flavour compares
    case-insensitively.
    """
    norm = ntpath.normpath if flavour is PureWindowsPath else posixpath.normpath
    try:
        base = flavour(norm(old))
        path = flavour(norm(value))
    except (TypeError, ValueError):
        return None
    if not path.is_absolute():
        return None
    if not path.is_relative_to(base):
        return None
    rel = path.relative_to(base)
    return str(new.joinpath(*rel.parts))


def _locators(ir: Any) -> list[tuple[Any, str, int | None, str]]:
    """Every locator slot of the IR object graph: ``(owner, attribute, list index or None, REBASE_KEYS key)``."""
    from pydantic import BaseModel

    from ai_eda.ir.components import LibraryRef
    from ai_eda.ir.project import ArtifactRef
    from ai_eda.ir.provenance import SourceRef
    from ai_eda.ir.regulatory import RegulatoryProvenance
    from ai_eda.ir.validation import Evidence

    slots: list[tuple[Any, str, int | None, str]] = []
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if isinstance(obj, BaseModel):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, ArtifactRef):
                slots.append((obj, "path", None, "artifact_paths"))
                slots.extend((obj, "files", i, "artifact_files") for i in range(len(obj.files)))
            elif isinstance(obj, Evidence):
                slots.append((obj, "path", None, "evidence"))
            elif isinstance(obj, SourceRef):
                slots.append((obj, "document_path", None, "source_documents"))
            elif isinstance(obj, LibraryRef):
                slots.append((obj, "library_path", None, "library_paths"))
            elif isinstance(obj, RegulatoryProvenance):
                slots.append((obj, "source_document", None, "regulatory_documents"))
            for name in type(obj).model_fields:
                visit(getattr(obj, name))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)

    visit(ir)
    return slots


def _slot_get(owner: Any, attr: str, index: int | None) -> Any:
    value = getattr(owner, attr)
    return value[index] if index is not None else value


def _slot_set(owner: Any, attr: str, index: int | None, value: str) -> None:
    if index is not None:
        getattr(owner, attr)[index] = value
    else:
        setattr(owner, attr, value)


def relocate_project(ir_path: str | Path, *, dry_run: bool = False, from_folder: str | None = None) -> RelocateReport:
    """Re-record ``project.workdir`` as the ir.json's own folder and rebase the locators under the old one (see the module docstring).

    Reads ir.json once, here - a caller holding the :class:`ProjectLock`
    therefore works on the file as it is under the lock. The old folder is
    ``from_folder`` (``--from``, absolute) when given, else the recorded
    absolute workdir naming another folder, else the one the artifacts show
    (:func:`infer_old_folder`). Nothing to do (no recorded workdir, a
    relative one the folder ends with, or already this folder - and no
    artifact outside it) -> ``changed`` False and nothing written. A relative
    workdir the folder does not end with and no artifact to follow is
    re-recorded with nothing rebased (``notes`` says so).
    :class:`RelocateError` (nothing written) when artifacts lie outside this
    folder and the old folder cannot be told, or when any would still lie
    outside it after rebasing. With ``dry_run`` nothing is written.
    :class:`RuntimeError` when the design hash would change (nothing
    written); the load's own errors propagate.
    """
    from ai_eda.ir.project import ArtifactKind, CircuitIR

    ir_path = Path(ir_path)
    new = ir_path.resolve().parent
    ir = CircuitIR.loads(ir_path.read_bytes(), source=str(ir_path))
    recorded = ir.project.workdir
    before = ir.content_hash()
    report = RelocateReport(ir_path=ir_path, old_workdir=recorded, new_workdir=new, changed=False, design_hash=before, dry_run=dry_run)
    stray = stray_artifacts(ir, new)
    if from_folder is not None:
        if _flavour(from_folder) is None:
            raise RelocateError(f"--from {from_folder!r} is not an absolute path (the folder the project was copied or moved from)")
        old: str | None = from_folder
    else:
        old = infer_old_folder(ir, new)
    if old is None:
        if stray:
            more = f" and {len(stray) - 1} more" if len(stray) > 1 else ""
            raise RelocateError(
                f"{ir_path}: its artifacts lie outside this folder ({stray[0]}{more}) and the folder they were compiled into cannot be told "
                f"from the IR; run `ai-eda relocate {ir_path} --from <old folder>` naming it. Nothing was written."
            )
        if not recorded or _relative_accepted(recorded, new) or (_flavour(recorded) is not None and not _recorded_mismatch(recorded, new)):
            return report  # nothing to relocate
        report.notes.append(f"the recorded workdir {recorded!r} is relative and no artifact lies outside this folder; nothing was rebased")
    report.changed = True
    report.rebased_from = old
    # results.json names its rawfiles by absolute path under the old folder: the reference goes, the file stays
    if ir.artifacts.pop(ArtifactKind.SPICE_RESULT, None) is not None:
        report.dropped.append(str(ArtifactKind.SPICE_RESULT))
    flavour = None if old is None else _flavour(old)
    missing: dict[str, None] = {}
    for owner, attr, index, key in _locators(ir):
        value = _slot_get(owner, attr, index)
        if not value:
            continue
        moved = _rebase_one(value, flavour, old, new) if flavour is not None and old is not None else None
        if moved is None:
            report.untouched += 1
            continue
        _slot_set(owner, attr, index, moved)
        report.rebased[key] += 1
        if not os.path.exists(moved):
            missing[moved] = None
    report.missing_on_disk = list(missing)
    left = stray_artifacts(ir, new)
    if left:
        more = f" and {len(left) - 1} more" if len(left) > 1 else ""
        raise RelocateError(
            f"{ir_path}: after rebasing from {old} an artifact would still lie outside this folder ({left[0]}{more}); "
            f"run `ai-eda relocate {ir_path} --from <old folder>` naming the folder it was compiled into. Nothing was written."
        )
    ir.project.workdir = str(new)
    after = ir.content_hash()
    if after != before:
        raise RuntimeError(f"relocate would change the design hash ({before} -> {after}); nothing was written")
    if not dry_run:
        ir.save(ir_path)
    return report


__all__ = [
    "BUSY_PHRASE",
    "LOCK_FILE",
    "REBASE_KEYS",
    "ProjectBusyError",
    "ProjectLock",
    "ProjectLockError",
    "ROOT_ARTIFACT_KINDS",
    "RelocateError",
    "RelocateReport",
    "WorkdirMismatchError",
    "foreign_absolute",
    "infer_old_folder",
    "is_locked",
    "mismatch_message",
    "project_workdir",
    "relocate_project",
    "stray_artifacts",
    "workdir_mismatch",
]
