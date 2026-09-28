"""Runs launched from the GUI: one CLI subprocess per project at a time, logged under ``<workdir>/gui/runs/``.

Invariant: every run is a subprocess ``[sys.executable, "-P", "-m",
"ai_eda.cli", <kind>, <ir.json>, *flags]`` - never in-process (ngspice is a
process-wide singleton, a run can take minutes, a crash must not take the GUI
down) - so its approvals are exactly the CLI's. The child never imports from
the project folder: ``-m`` alone would put its working directory (the
workdir) first on ``sys.path``, so an ``ai_eda/cli.py`` or a ``csv.py`` a
project folder holds (an unzipped project from someone else, a scratch
script) would run instead of the installed package; ``-P`` (safe path,
Python 3.11+) leaves the working directory off ``sys.path``, as the
``ai-eda`` console script does. The flags are built only by the CLI's
own :func:`ai_eda.cli.run_option_flags` from the form's fields, which are
the ``RUN_OPTIONS`` dests the subcommand's own parser accepts
(:func:`allowed_options`; answers become ``--answer`` items): there is no
second list of flag names, and a field the user left empty adds nothing.
The budget / ``--llm claude`` / ``--online`` flags are recorded as grants by
the subprocess's own approval gate, never here.

The form is checked before any process starts (:func:`form_args`): the kind
is ``run`` / ``review`` / ``stage-reports``, answer keys match
:data:`ANSWER_KEY_PATTERN`, option names are known to that kind and values
have the option's type, no value holds a control character, and no option
value starts with ``-`` (argparse would read it as a flag). A refusal is
:class:`RunRequestError` (the server's 400) and starts nothing, writes
nothing. Each value is one argv item; there is never a shell.

At most one run per project workdir is active (:class:`RunConflictError`,
the server's 409): the runs this manager started by its own memory, and, on
POSIX, a run another GUI process started - or one whose GUI was stopped
while it ran - by the ``flock`` its log carries, and a command-line
``ai-eda run`` / ``review`` / ``stage-reports`` / ``relocate`` by the
project lock it holds (:func:`ai_eda.workdir.is_locked`, probed only when a
run is started, never while listing or polling). The GUI never takes that
lock itself (only its run log's ``flock``, a different file): its child is
the CLI, which takes it non-blockingly, and no lock in the system waits, so
there is no deadlock. A project whose recorded workdir names another folder
(copied or moved) is refused before anything starts, with the CLI's
``ai-eda relocate`` remedy. The log's open file
description is locked before the child starts and the child inherits it as
its stdout / stderr, so the lock is held exactly as long as the run's
output is open, whatever happens to the GUI: a log with no closing line
whose lock is held is ``running`` (and blocks a new run of that workdir),
one whose lock is free ended without a recorded code. On Windows a
byte-range lock dies with the process that took it, so there the slot is
this GUI process's own (a GUI ended from the task manager while a run goes
on does not know about that run). The child runs with ``cwd`` = the project workdir, stdin
from the null device, and the user's own environment - API keys, the Claude
Code login, PATH, never read or shown here - with these additions:
``PYTHONIOENCODING=utf-8`` (the log is UTF-8 even under a cp949 console)
and ``PYTHONUNBUFFERED=1`` (stdout and stderr reach the log in the order
they were written, while the run is going); on Windows a third,
``NoDefaultCurrentDirectoryInExePath=1``, so the bare tool names the CLI
looks up (``kicad-cli``, ``claude``, a browser; Python 3.12's
``shutil.which`` consults it) are never found in the workdir either (not
measured on Windows yet). The log
``<workdir>/gui/runs/<NNN>-<kind>.log`` holds the command line (``$ ...``;
no secret is ever on it - the CLI takes keys from the environment only),
stdout and stderr merged, and a closing ``# exit code N`` line. It is the
only file this module writes; no run is stopped, deleted or renamed here.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel

try:
    import fcntl
except ImportError:  # Windows: no flock (see the module docstring)
    fcntl = None  # type: ignore[assignment]

from ai_eda.cli import RUN_OPTIONS, RunOption, build_parser, run_option_flags
from ai_eda.gui.projects import ProjectInfo, openable_project_name
from ai_eda.workdir import WorkdirMismatchError, is_locked

#: what a run can be: the CLI subcommands the GUI starts
RUN_KINDS: tuple[str, ...] = ("run", "review", "stage-reports")
#: an answer key (``confirm_design``, ``pcb.placement``, ``highest_rated_voltage``)
ANSWER_KEY_PATTERN = r"^[a-z0-9_.]+$"
ANSWER_KEY_RE = re.compile(ANSWER_KEY_PATTERN)
#: the run option the answers are rendered through (never a form field of its own)
ANSWER_OPTION = "answer"
#: the log folder under a project workdir
LOG_DIR = Path("gui") / "runs"
LOG_NAME_RE = re.compile(r"(\d{3,})-(" + "|".join(re.escape(k) for k in RUN_KINDS) + r")\.log")
#: the first characters of the command line (first line) and of the closing line of a log
COMMAND_PREFIX = "$ "
EXIT_PREFIX = "# exit code "
EXIT_LINE_RE = re.compile(r"# exit code (-?\d+)")
#: how many log lines a status carries, and how many bytes at most are read for them
TAIL_LINES = 200
TAIL_BYTES = 1 << 20
#: what the child's environment gains over the user's own (see the module docstring)
CHILD_ENV_ADDITIONS: dict[str, str] = {
    "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
    **({"NoDefaultCurrentDirectoryInExePath": "1"} if os.name == "nt" else {}),
}
#: what every run starts with: ``-P`` keeps the child's working directory (the workdir) off ``sys.path``
CLI_PYTHON_ARGS: tuple[str, ...] = (sys.executable, "-P", "-m", "ai_eda.cli")


class RunRequestError(ValueError):
    """A run the GUI refuses before starting anything (a bad kind, answer key, option or value; a project that cannot run): the server's 400."""


class RunConflictError(RuntimeError):
    """A run of this project is already active: the server's 409."""


class RunNotFoundError(LookupError):
    """No run log of that id for this project: the server's 404."""


class RunStartError(RuntimeError):
    """The process could not be started (the reason is also the last line of its log): the server's 503."""


class RunHandle(BaseModel):
    """A started run: ``id`` is the log's name without ``.log`` (``001-run``)."""

    id: str
    kind: str
    workdir: Path
    log_path: Path
    #: when the GUI started it (UTC, ISO 8601): GUI state, never written into the IR
    started: str


class RunStatus(BaseModel):
    """A run as its log and the manager know it now; ``exit_code`` is ``None`` while running or when the log has no closing line."""

    id: str
    kind: str
    running: bool
    exit_code: int | None
    #: the last :data:`TAIL_LINES` lines of the log, decoded as UTF-8 with replacement
    log_tail: str
    log_bytes: int
    started: str | None = None


class RunLog(BaseModel):
    """One log file under ``<workdir>/gui/runs``."""

    id: str
    kind: str
    size: int
    running: bool


# --------------------------------------------------------------------------- the form


@lru_cache(maxsize=None)
def _subcommand_dests(kind: str) -> frozenset[str]:
    """The argument dests of the ``ai-eda <kind>`` parser, read from the CLI's own :func:`~ai_eda.cli.build_parser`."""
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse keeps its subparsers here; the CLI parser is the one source
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            sub = action.choices.get(kind)
            if sub is not None:
                return frozenset(a.dest for a in sub._actions)  # noqa: SLF001
    return frozenset()


def allowed_options(kind: str) -> tuple[str, ...]:
    """The form fields of ``kind``: the ``RUN_OPTIONS`` dests the ``ai-eda <kind>`` parser accepts, in table order, ``answer`` excluded.

    ``run`` takes every run option; ``stage-reports`` only ``no_pdf`` and
    ``browser``; ``review`` none.
    """
    if kind not in RUN_KINDS:
        raise RunRequestError(f"알 수 없는 실행 종류: {kind!r} (가능: {', '.join(RUN_KINDS)})")
    dests = _subcommand_dests(kind)
    return tuple(o.dest for o in RUN_OPTIONS if o.dest in dests and o.dest != ANSWER_OPTION)


def _has_control(text: str) -> bool:
    return any(ord(c) < 32 or ord(c) == 127 for c in text)


def _text(field: str, value: Any, *, option: bool = True) -> str | None:
    """A text value as one argv item; ``None`` for an empty field. Refuses a non-string, a control character and (for an option value) a leading ``-``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunRequestError(f"{field}: 문자열이어야 합니다 (받은 값: {value!r})")
    if value == "":
        return None
    if _has_control(value):
        raise RunRequestError(f"{field}: 제어 문자(줄바꿈 등)는 쓸 수 없습니다")
    if option and value.startswith("-"):
        raise RunRequestError(f"{field}: '-'로 시작하는 값은 명령줄 옵션으로 읽히므로 쓸 수 없습니다 ({value!r})")
    return value


def _number(option: RunOption, value: Any) -> float | int | None:
    """A ``float`` / ``int`` option from a JSON number or a numeric string; ``None`` for an empty field."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise RunRequestError(f"{option.dest}: 숫자여야 합니다 (받은 값: {value!r})")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = int(text) if option.kind == "int" else float(text)
        except ValueError:
            raise RunRequestError(f"{option.dest}: 숫자가 아닙니다 ({value!r})") from None
    if option.kind == "int":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise RunRequestError(f"{option.dest}: 정수여야 합니다 (받은 값: {value!r})")
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RunRequestError(f"{option.dest}: 유한한 숫자여야 합니다 (받은 값: {value!r})")
    return float(value)


def _option_value(option: RunOption, value: Any) -> Any:
    """``value`` in the shape :func:`~ai_eda.cli.run_option_flags` takes for ``option``; ``None`` when the field is empty."""
    if option.kind == "flag":
        if value is None:
            return None
        if not isinstance(value, bool):
            raise RunRequestError(f"{option.dest}: 참/거짓 값이어야 합니다 (받은 값: {value!r})")
        return True if value else None
    if option.kind in ("float", "int"):
        return _number(option, value)
    if option.kind == "append":
        if value is None:
            return None
        if isinstance(value, Mapping):
            if "=" not in (option.metavar or ""):
                raise RunRequestError(f"{option.dest}: 목록이어야 합니다 ({option.metavar} 형식이 아닙니다)")
            pairs: dict[str, str] = {}
            for key, item in value.items():
                k = _text(f"{option.dest} 키", key)
                v = _text(f"{option.dest}[{key}]", item, option=False)
                if k is None or "=" in k:
                    raise RunRequestError(f"{option.dest}: 키가 비었거나 '='를 담고 있습니다 ({key!r})")
                if v is not None:
                    pairs[k] = v
            return pairs or None
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list):
            raise RunRequestError(f"{option.dest}: 목록이어야 합니다 (받은 값: {value!r})")
        texts = [t for t in (_text(option.dest, item) for item in items) if t is not None]
        return texts or None
    return _text(option.dest, value)


def _answers(answers: Any) -> dict[str, str]:
    """The answers as ``{key: value}``; an empty value is not an answer (the field was left empty)."""
    if answers is None:
        return {}
    if not isinstance(answers, Mapping):
        raise RunRequestError("answers: {키: 값} 객체여야 합니다")
    out: dict[str, str] = {}
    for key, value in answers.items():
        if not isinstance(key, str) or ANSWER_KEY_RE.fullmatch(key) is None:
            raise RunRequestError(f"답변 키 {key!r}: 영소문자, 숫자, '_', '.'만 쓸 수 있습니다 ({ANSWER_KEY_PATTERN})")
        text = _text(f"답변 {key}", value, option=False)
        if text is not None:
            out[key] = text
    return out


def form_args(kind: str, answers: Mapping[str, Any] | None = None, options: Mapping[str, Any] | None = None) -> list[str]:
    """The ``ai-eda <kind>`` flags of a GUI form, rendered by :func:`ai_eda.cli.run_option_flags`; :class:`RunRequestError` for anything it refuses.

    ``options`` is keyed by run-option dest (:func:`allowed_options`), with
    JSON values: a bool for a flag, a number or a numeric string for a
    budget, a string for a text option, a list of strings (or, for a
    ``KEY=VALUE`` option, an object) for a repeatable one. ``None``, ``""``,
    ``false`` and ``[]`` mean the field is empty and add nothing.
    ``answers`` (``run`` only) become ``--answer key=value`` items, the
    value verbatim - spaces and ``=`` included.
    """
    allowed = allowed_options(kind)
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise RunRequestError("options: {이름: 값} 객체여야 합니다")
    unknown = sorted(str(name) for name in options if name not in allowed)
    if unknown:
        hint = " (답변은 answers로 보냅니다)" if ANSWER_OPTION in unknown else ""
        raise RunRequestError(
            f"{kind}에 없는 실행 옵션: {', '.join(unknown)}{hint} (가능: {', '.join(allowed) or '없음'})"
        )
    by_dest = {o.dest: o for o in RUN_OPTIONS}
    values: dict[str, Any] = {}
    for name, value in options.items():
        rendered = _option_value(by_dest[name], value)
        if rendered is not None:
            values[name] = rendered
    parsed = _answers(answers)
    if parsed:
        if kind != "run":
            raise RunRequestError(f"{kind}는 답변을 받지 않습니다 (답변은 run에서만 씁니다)")
        values[ANSWER_OPTION] = parsed
    try:
        return run_option_flags(**values)
    except TypeError as e:  # the CLI's own refusal of a name or a value shape: reported, nothing started
        raise RunRequestError(str(e)) from e


# --------------------------------------------------------------------------- the process


def child_env() -> dict[str, str]:
    """The run's environment: the user's own, read when the run starts, plus :data:`CHILD_ENV_ADDITIONS`."""
    return {**os.environ, **CHILD_ENV_ADDITIONS}


def command_line(argv: Sequence[str]) -> str:
    """``argv`` as one displayable line (POSIX quoting; the Windows rules on Windows)."""
    return subprocess.list2cmdline(list(argv)) if os.name == "nt" else shlex.join(argv)


def _workdir_of(ir_path: Path) -> Path:
    from ai_eda.ir import CircuitIR
    from ai_eda.workdir import project_workdir

    try:
        return project_workdir(CircuitIR.load(ir_path), ir_path)
    except WorkdirMismatchError as e:  # a copied / moved project: the CLI's own refusal, with its relocate remedy
        raise RunRequestError(str(e)) from e
    except Exception as e:  # noqa: BLE001 - the run cannot be placed; refused before anything starts
        raise RunRequestError(f"{ir_path}: 작업 폴더를 정할 수 없습니다: {e}") from e


def _exit_code_of(lines: list[str]) -> int | None:
    """The code of a log's closing ``# exit code N`` line (``None`` when the log has none: running, or the GUI stopped first)."""
    for line in reversed(lines):
        if line.strip():
            m = EXIT_LINE_RE.fullmatch(line.strip())
            return int(m.group(1)) if m else None
    return None


def _read_tail(path: Path) -> tuple[list[str], int]:
    """The last :data:`TAIL_LINES` lines of ``path`` (UTF-8, replacement for bad bytes) and its size in bytes."""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        start = max(0, size - TAIL_BYTES)
        fh.seek(start)
        data = fh.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # the first line was cut by the byte window
    return lines[-TAIL_LINES:], size


def _hold(log: IO[bytes]) -> None:
    """Lock the log's open file description (POSIX ``flock``) before the child inherits it: the lock then lives as long as the run's output does."""
    if fcntl is None:
        return
    try:
        fcntl.flock(log.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        pass  # a filesystem without flock: the slot is this process's memory only


def _held(path: Path) -> bool:
    """Whether some process still holds the lock of the log ``path`` (its run's output is open); always ``False`` without ``flock``."""
    if fcntl is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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


def _running_elsewhere(path: Path) -> bool:
    """A log this manager did not write whose run is still going: its lock is held and it has no closing ``# exit code`` line."""
    if not _held(path):
        return False
    lines, _size = _read_tail(path)
    return _exit_code_of(lines) is None


class _Run:
    def __init__(self, handle: RunHandle, proc: subprocess.Popen[bytes], log: IO[bytes]) -> None:
        self.handle = handle
        self.proc = proc
        self.log = log
        self.exit_code: int | None = None
        self.done = threading.Event()


class RunManager:
    """Starts and watches the GUI's runs; one active run per project workdir.

    ``python_args`` replaces :data:`CLI_PYTHON_ARGS` (``[sys.executable,
    "-P", "-m", "ai_eda.cli"]``) - the command the kind, the ir.json and the
    flags are appended to - for tests (a sleeping fake process); the GUI
    never passes it.
    """

    def __init__(self, python_args: Sequence[str] | None = None) -> None:
        self.python_args: list[str] = list(python_args) if python_args is not None else list(CLI_PYTHON_ARGS)
        self._lock = threading.Lock()
        self._active: dict[Path, _Run] = {}
        self._runs: dict[tuple[Path, str], _Run] = {}

    # ------------------------------------------------------------------ start

    def start_project(
        self, kind: str, project: ProjectInfo, answers: Mapping[str, Any] | None = None, options: Mapping[str, Any] | None = None,
    ) -> RunHandle:
        """Start ``kind`` for a listed project from the form's ``answers`` / ``options`` (checked by :func:`form_args` first)."""
        if not openable_project_name(project.name):
            raise RunRequestError(f"프로젝트 이름이 아닙니다: {project.name!r}")
        if project.error is not None or project.workdir is None:
            raise RunRequestError(f"{project.name}: 실행할 수 없는 프로젝트입니다: {project.error or '작업 폴더 없음'}")
        if project.workdir_mismatch:
            raise RunRequestError(
                f"{project.name}: 실행하지 않습니다: "
                + (project.warning or f"기록된 작업 폴더가 이 폴더와 다릅니다; `ai-eda relocate {project.ir_path}`를 먼저 실행하십시오")
            )
        args = form_args(kind, answers, options)
        return self.start(kind, project.ir_path, args, workdir=project.workdir)

    def start(self, kind: str, ir_path: str | Path, args: Sequence[str], *, workdir: str | Path | None = None) -> RunHandle:
        """Start ``ai-eda <kind> <ir_path> *args`` in the project's workdir (from ir.json when not given) and return its handle.

        ``args`` are flags :func:`form_args` built. :class:`RunRequestError`
        for an unknown kind, an argument that is not text or holds a NUL,
        or a workdir that is not an existing folder (or whose log folder is
        a link); :class:`RunConflictError` while a run of that workdir is
        active - this manager's, another GUI's, or a command-line one holding
        the project lock; :class:`RunStartError` when the process cannot be started.
        """
        if kind not in RUN_KINDS:
            raise RunRequestError(f"알 수 없는 실행 종류: {kind!r} (가능: {', '.join(RUN_KINDS)})")
        args = list(args)
        if not all(isinstance(a, str) and "\x00" not in a for a in args):
            raise RunRequestError("실행 인자는 NUL이 없는 문자열이어야 합니다")
        ir_file = Path(ir_path).absolute()
        place = Path(workdir) if workdir is not None else _workdir_of(ir_file)
        if not place.is_dir():
            raise RunRequestError(f"작업 폴더가 없습니다: {place}")
        place = place.resolve()
        log_dir = place / LOG_DIR
        if any(p.is_symlink() for p in (place / LOG_DIR.parts[0], log_dir)):
            raise RunRequestError(f"실행 로그 폴더가 심볼릭 링크입니다: {log_dir}")
        argv = [*self.python_args, kind, str(ir_file), *args]
        with self._lock:
            if place in self._active:
                active = self._active[place].handle
                raise RunConflictError(f"이 프로젝트는 이미 실행 중입니다 ({active.id}); 끝난 뒤에 다시 시작하십시오")
            elsewhere = self._foreign_active(place)
            if elsewhere is not None:
                raise RunConflictError(
                    f"이 프로젝트는 이미 실행 중입니다 ({elsewhere}: 다른 GUI 프로세스가, 또는 먼저 끝난 GUI가 시작한 실행); 끝난 뒤에 다시 시작하십시오"
                )
            if is_locked(place):  # a command-line ai-eda holds the project lock (probed here only: a user started this run)
                raise RunConflictError("이 프로젝트는 다른 ai-eda 명령(명령줄 실행)이 쓰고 있습니다 (.ai-eda.lock); 끝난 뒤에 다시 시작하십시오")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                run_id, log_path, log = _open_new_log(log_dir, kind)
            except OSError as e:
                raise RunStartError(f"실행 로그를 만들 수 없습니다: {log_dir}: {e}") from e
            _hold(log)
            try:
                log.write(f"{COMMAND_PREFIX}{command_line(argv)}\n".encode("utf-8"))
                log.flush()
                proc = subprocess.Popen(  # noqa: S603 - a list argv, no shell; the flags come from run_option_flags
                    argv, cwd=str(place), env=child_env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            except (OSError, ValueError) as e:
                try:
                    log.write(f"# could not start: {type(e).__name__}: {e}\n".encode("utf-8", errors="replace"))
                finally:
                    log.close()
                raise RunStartError(f"실행을 시작할 수 없습니다: {type(e).__name__}: {e}") from e
            handle = RunHandle(
                id=run_id, kind=kind, workdir=place, log_path=log_path,
                started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            run = _Run(handle, proc, log)
            self._active[place] = run
            self._runs[(place, run_id)] = run
        threading.Thread(target=self._watch, args=(run,), name=f"ai-eda-gui-run-{run_id}", daemon=True).start()
        return handle

    def _watch(self, run: _Run) -> None:
        """Wait for the process, close the log with its exit code, then free the project's slot."""
        code = run.proc.wait()
        try:
            _append_exit_line(run.handle.log_path, run.log, code)
        except OSError:
            pass  # the status still carries the code; the log only lacks its closing line
        finally:
            run.log.close()
        with self._lock:
            run.exit_code = code
            if self._active.get(run.handle.workdir) is run:
                del self._active[run.handle.workdir]
        run.done.set()

    # ------------------------------------------------------------------ read

    def active(self, workdir: str | Path) -> RunHandle | None:
        """The run of this workdir that this manager started and that is going now, if any."""
        with self._lock:
            run = self._active.get(Path(workdir).resolve())
            return run.handle if run is not None else None

    def active_id(self, workdir: str | Path) -> str | None:
        """The id of the run of this workdir that is going now: this manager's, else (POSIX) one another GUI process started."""
        place = Path(workdir).resolve()
        with self._lock:
            run = self._active.get(place)
            return run.handle.id if run is not None else self._foreign_active(place)

    def _foreign_active(self, place: Path) -> str | None:
        """A run of ``place`` this manager did not start whose output is still open (its log's lock is held; POSIX only)."""
        log_dir = place / LOG_DIR
        if fcntl is None or log_dir.is_symlink() or not log_dir.is_dir():
            return None
        for path in sorted(log_dir.iterdir(), key=lambda p: p.name):
            if LOG_NAME_RE.fullmatch(path.name) is None or path.is_symlink() or not path.is_file():
                continue
            run_id = path.name[: -len(".log")]
            if (place, run_id) not in self._runs and _running_elsewhere(path):
                return run_id
        return None

    def status(self, workdir: str | Path, run_id: str) -> RunStatus:
        """What the run ``run_id`` of ``workdir`` is now: running / exit code / the log's tail. :class:`RunNotFoundError` for an unknown id."""
        place = Path(workdir).resolve()
        m = LOG_NAME_RE.fullmatch(f"{run_id}.log") if isinstance(run_id, str) else None
        if m is None:
            raise RunNotFoundError(f"실행 기록이 없습니다: {run_id!r}")
        with self._lock:  # the manager's state first: a finished run's log already holds its closing line
            run = self._runs.get((place, run_id))
            exit_code = run.exit_code if run is not None else None
            running = run is not None and run.exit_code is None
        path = place / LOG_DIR / f"{run_id}.log"
        if path.is_symlink() or not path.is_file():
            raise RunNotFoundError(f"실행 기록이 없습니다: {run_id}")
        lines, size = _read_tail(path)
        if run is None:  # a run of another GUI process (or of one that stopped first): the log's closing line, else its lock
            exit_code = _exit_code_of(lines)
            running = exit_code is None and _held(path)
        return RunStatus(
            id=run_id, kind=m.group(2), running=running, exit_code=exit_code, log_tail="\n".join(lines), log_bytes=size,
            started=run.handle.started if run is not None else None,
        )

    def list_runs(self, workdir: str | Path) -> list[RunLog]:
        """The run logs of ``workdir`` that exist, oldest first."""
        place = Path(workdir).resolve()
        log_dir = place / LOG_DIR
        if log_dir.is_symlink() or not log_dir.is_dir():
            return []
        found: list[tuple[int, RunLog]] = []
        with self._lock:
            known = {run_id: run.exit_code is None for (wd, run_id), run in self._runs.items() if wd == place}
        for path in log_dir.iterdir():
            m = LOG_NAME_RE.fullmatch(path.name)
            if m is None or path.is_symlink() or not path.is_file():
                continue
            run_id = path.name[: -len(".log")]
            running = known[run_id] if run_id in known else _running_elsewhere(path)
            found.append((int(m.group(1)), RunLog(id=run_id, kind=m.group(2), size=path.stat().st_size, running=running)))
        return [log for _, log in sorted(found, key=lambda t: t[0])]

    def wait(self, handle: RunHandle, timeout: float | None = None) -> int:
        """Block until the run of ``handle`` ended and return its exit code (for tests); ``TimeoutError`` after ``timeout`` seconds."""
        with self._lock:
            run = self._runs.get((handle.workdir, handle.id))
        if run is None:
            raise RunNotFoundError(f"이 관리자가 시작한 실행이 아닙니다: {handle.id}")
        if not run.done.wait(timeout):
            raise TimeoutError(f"{handle.id} is still running after {timeout} s")
        assert run.exit_code is not None
        return run.exit_code


def _open_new_log(log_dir: Path, kind: str) -> tuple[str, Path, IO[bytes]]:
    """Create the next ``<NNN>-<kind>.log`` exclusively, in append mode (the number follows the highest log already there).

    Append mode: the child writes through a duplicate of this handle, and the
    closing line written here afterwards must land after everything it wrote.
    """
    highest = 0
    for path in log_dir.iterdir():
        m = LOG_NAME_RE.fullmatch(path.name)
        if m is not None:
            highest = max(highest, int(m.group(1)))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | getattr(os, "O_BINARY", 0)
    for number in range(highest + 1, highest + 100):
        run_id = f"{number:03d}-{kind}"
        path = log_dir / f"{run_id}.log"
        try:
            fd = os.open(path, flags, 0o644)
        except FileExistsError:
            continue
        return run_id, path, os.fdopen(fd, "ab")
    raise RunStartError(f"새 실행 로그를 만들 수 없습니다: {log_dir}")


def _append_exit_line(path: Path, log: IO[bytes], code: int) -> None:
    """Write ``# exit code N`` on a line of its own at the end of the log."""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        ends_with_newline = True
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            ends_with_newline = fh.read(1) == b"\n"
    log.write((b"" if ends_with_newline else b"\n") + f"{EXIT_PREFIX}{code}\n".encode("utf-8"))
    log.flush()
