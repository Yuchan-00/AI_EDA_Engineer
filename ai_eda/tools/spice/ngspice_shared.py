"""ngspice as a shared library: KiCad's bundled ``ngspice.dll`` driven through ctypes.

Invariant: a :class:`~ai_eda.tools.spice.runner.SpiceResult` from this runner
is ``succeeded`` only when the engine itself reported a clean run - no
``stderr`` line, no ``ControlledExit``, a ``No. of Data Rows`` line, a *new*
plot of the requested analysis whose title is the deck's, every vector of
that plot of that length, and a rawfile written by ngspice that agrees with
the vectors read through the API. Return codes are not evidence
(``ngSpice_Command`` returns 0 for parse errors, aborted analyses and unknown
commands). A simulation failure is a result with ``errors``; only a missing
or unusable DLL raises ``ToolUnavailableError``.

Everything below was measured against KiCad 10.0.6's ``bin\\ngspice.dll``
(ngspice-46, build Apr 14 2026) in the SPICE-stage lab; the code depends on
these facts:

1. ``ngGet_Vec_Info`` returns a pointer to ONE static struct: the data is
   copied immediately (``v_realdata[:n]``) and never held as two views.
2. The DLL is process-global and cannot be unloaded; ``ngSpice_Init`` may be
   repeated (callbacks are replaced). Hence one :class:`_Engine` per process
   and :class:`NgspiceShared` handles that share it.
3. KiCad ships no ``spinit``; the "can't find the initialization file spinit"
   warning is benign. Code models are loaded with ``codemodel '<path>'`` for
   the six ``.cm`` files KiCad loads, only while no circuit is loaded (with a
   circuit present the next ``remcirc`` faults) and again after every
   ``ngSpice_Reset`` (they do not survive it). Success is proven by an XSPICE
   self-test and exposed as :attr:`NgspiceShared.codemodels_loaded`; a
   failure is recorded in the engine log, never hidden.
4. Netlists are loaded with ``ngSpice_Circ`` from the bytes the runner read,
   hashed and validated *once* (so ``netlist_hash`` is provably the hash of
   what ngspice parsed, and the file's path never reaches an ngspice command:
   a ``'``, ``$`` or backtick in a user's folder name cannot fail a run).
   ``ngSpice_Circ`` behaves like ``source``: ngspice prints ``Circuit:
   <title>`` for the first line, ``listing`` works and the plot carries the
   title - both are cross-checked. Only the rawfile is written by path
   (``write '<path>'``: spaces, Korean and emoji work; ``'``, ``$`` and a
   backtick do not), so when the workdir cannot be named the rawfile goes to
   the system temp directory and is copied into the workdir, and when neither
   can be named the run is ``unverifiable`` (NOT_VERIFIED downstream, never a
   FAIL about the design).
5. The analysis is an interactive command (``op``, ``dc v1 0 12 1``,
   ``tran 10u 5m``, ``ac dec 10 1 1meg``) issued through ngspice's ``bg_``
   prefix (``bg_tran 10u 5m``): identical data to the foreground command but
   interruptible with ``bg_halt`` (a foreground command cannot be interrupted
   from Python). ``ngSpice_running()`` is still False right after ``bg_*``
   returns, so the watchdog waits for the BGThreadRunning start event first.
   ngspice lowercases decks but matches device names in commands
   case-sensitively, so commands are lowercased
   (:func:`~ai_eda.tools.spice.runner.normalise_command`).
6. Some decks crash the host process (no non-ground node -> access violation
   in ``op``) or poison it for good (an unbalanced ``'`` -> every later load
   fails, not cured by Reset+Init). :func:`validate_deck` refuses those - and
   analysis cards, ``.control``, ``.include``/``.lib``, non-ASCII, node names
   ngspice mangles - before the DLL sees the file.
7. Silent wrong results exist: ``R1 IN OUT`` (no value) is "ignored!" with a
   warning and the analysis still runs; singular-matrix circuits print data
   rows after a warning; a missing title eats the first element. Any
   ``stderr`` line during load or run is a failure.
8. A failed load leaves the previous circuit and plot current: every run
   starts by unloading everything (``remcirc`` until "there is no circuit
   loaded", then ``destroy all``) and demands a plot that did not exist
   before the command.
9. After a ``ControlledExit`` the DLL "awaits to be reset": recovery is
   ``ngSpice_Reset()`` + ``ngSpice_Init()`` + settings + code models +
   self-test. An access violation (``OSError`` from ctypes) leaves the DLL
   undefined: the engine is marked dead and every later use raises
   ``ToolUnavailableError`` until the process is restarted.
10. Rawfiles carry a ``Date`` line, so they are evidence (path + sha256), not
    deterministic artifacts. ASCII rawfiles have 15 significant digits
    (~1e-15 relative); binary ones are bit-exact.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import tempfile
import threading
import time
from ctypes import CFUNCTYPE, POINTER, Structure, c_bool, c_char_p, c_double, c_int, c_short, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai_eda.errors import ToolExecutionError, ToolUnavailableError
from ai_eda.tools.spice import rawfile
from ai_eda.tools.spice.runner import SpiceAnalysis, SpiceResult, SpiceRunner, normalise_command

ENGINE_ID = "ngspice-shared"
#: the XSPICE code model libraries KiCad loads (``<KiCad>/lib/ngspice/*.cm``)
CODEMODELS: tuple[str, ...] = ("spice2poly.cm", "analog.cm", "digital.cm", "xtradev.cm", "xtraevt.cm", "table.cm")
#: what KiCad sets right after ``ngSpice_Init``
INIT_SETTINGS: tuple[str, ...] = ("unset interactive", "set noaskquit", "set nomoremode")
BENIGN_INIT_STDERR: tuple[str, ...] = ("Warning: can't find the initialization file spinit.",)
#: stderr lines that are information, not errors: ngspice builds other than KiCad's announce their linear solver
#: on stderr before every analysis (Debian/Ubuntu ``libngspice0`` 42: ``Using SPARSE 1.3 as Direct Linear Solver``).
#: They stay in the transcript (``log``) but do not fail a run.
INFORMATIONAL_STDERR_RE = re.compile(r"^Using \S.* as Direct Linear Solver$")
DEFAULT_TIMEOUT_S = 120.0

# ------------------------------------------------------------------------------------------ sharedspice.h


class NgComplex(Structure):
    _fields_ = [("cx_real", c_double), ("cx_imag", c_double)]


class VectorInfo(Structure):
    """``struct vector_info``; x64 offsets 0 name, 8 type, 12 flags, 16 realdata, 24 compdata, 32 length (verified)."""

    _fields_ = [
        ("v_name", c_char_p),
        ("v_type", c_int),
        ("v_flags", c_short),
        ("v_realdata", POINTER(c_double)),
        ("v_compdata", POINTER(NgComplex)),
        ("v_length", c_int),
    ]


class VecValues(Structure):
    _fields_ = [("name", c_char_p), ("creal", c_double), ("cimag", c_double), ("is_scale", c_bool), ("is_complex", c_bool)]


class VecValuesAll(Structure):
    _fields_ = [("veccount", c_int), ("vecindex", c_int), ("vecsa", POINTER(POINTER(VecValues)))]


class VecInfo(Structure):
    _fields_ = [("number", c_int), ("vecname", c_char_p), ("is_real", c_bool), ("pdvec", c_void_p), ("pdvecscale", c_void_p)]


class VecInfoAll(Structure):
    _fields_ = [
        ("name", c_char_p),  # plot type, e.g. 'Transient Analysis'
        ("title", c_char_p),  # deck title
        ("date", c_char_p),
        ("type", c_char_p),  # plot name, e.g. 'tran1'
        ("veccount", c_int),
        ("vecs", POINTER(POINTER(VecInfo))),
    ]


SendChar = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
SendStat = CFUNCTYPE(c_int, c_char_p, c_int, c_void_p)
ControlledExit = CFUNCTYPE(c_int, c_int, c_bool, c_bool, c_int, c_void_p)  # (status, immediate_unload, quit_exit, ident, user)
SendData = CFUNCTYPE(c_int, POINTER(VecValuesAll), c_int, c_int, c_void_p)
SendInitData = CFUNCTYPE(c_int, POINTER(VecInfoAll), c_int, c_void_p)
BGThreadRunning = CFUNCTYPE(c_int, c_bool, c_int, c_void_p)  # noruns: False = thread started, True = finished

#: ``v_type`` words (ngspice ``sim.h``)
SV_TYPE_NAMES: dict[int, str] = {
    0: "notype", 1: "time", 2: "frequency", 3: "voltage", 4: "current", 5: "voltage_density", 6: "current_density",
    7: "sqr_voltage_density", 8: "sqr_current_density", 9: "sqr_voltage", 10: "sqr_current", 11: "pole", 12: "zero",
    13: "s_param", 14: "temperature", 15: "res", 16: "impedance", 17: "admittance", 18: "power", 19: "phase", 20: "db",
    21: "capacitance", 22: "charge",
}
VF_REAL, VF_COMPLEX, VF_PERMANENT = 1, 2, 128

_VERSION_RE = re.compile(r"^\*\* (ngspice-\d+)", re.M)
_BUILD_RE = re.compile(r"^\*\* Creation Date: (.+?)\s*$", re.M)
_ROWS_RE = re.compile(r"No\. of Data Rows\s*:\s*(\d+)")


class EngineDead(RuntimeError):
    """The DLL is unusable in this process (access violation, failed re-init, failed self-test)."""


# ------------------------------------------------------------------------------------------ locating the DLL


def find_ngspice_dll() -> Path | None:
    """``$NGSPICE_DLL`` if set (must exist), else ``ngspice.dll`` next to the ``kicad-cli`` that
    :func:`ai_eda.tools.kicad.cli.find_kicad_cli` finds, else None."""
    env = os.environ.get("NGSPICE_DLL")
    if env:
        p = Path(env)
        return p.resolve() if p.is_file() else None
    from ai_eda.tools.kicad.cli import find_kicad_cli  # lazy: kicad.cli imports ai_eda.ir, which imports this package

    cli = find_kicad_cli()
    if not cli:
        return None
    p = Path(cli).resolve().parent / "ngspice.dll"
    return p if p.is_file() else None


def find_codemodel_dir(dll: Path | None) -> Path | None:
    """``$NGSPICE_CODEMODEL_DIR`` if set, else the code model directory next to the library.

    Two layouts are known: KiCad's ``<root>/lib/ngspice`` for a DLL at
    ``<root>/bin/ngspice.dll``, and Debian/Ubuntu's ``<libdir>/ngspice`` for
    ``<libdir>/libngspice.so.0`` (measured 2026-09-23 on Ubuntu 24.04:
    ``/usr/lib/x86_64-linux-gnu/ngspice/*.cm``).
    """
    env = os.environ.get("NGSPICE_CODEMODEL_DIR")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    if dll is None:
        return None
    for p in (dll.parent.parent / "lib" / "ngspice", dll.parent / "ngspice"):
        if p.is_dir():
            return p
    return None


def check_path_for_command(path: Path) -> None:
    """``ValueError`` when ``path`` cannot be passed inside single quotes to an ngspice command."""
    s = str(path)
    bad = [ch for ch in "'$`" if ch in s]
    if bad:
        raise ValueError(f"path {s!r} contains {bad}: ngspice expands $ and ` even inside single quotes and a ' cannot be escaped")
    if "\n" in s or "\r" in s:
        raise ValueError(f"path {s!r} contains a newline")


# ------------------------------------------------------------------------------------------ deck validation

#: characters ngspice keeps intact in a node name (measured); ``( ) { } = , ; ' "`` and non-ASCII are mangled or fatal
NODE_RE = re.compile(r"[A-Za-z0-9_./+\-:#@\[\]]+")
GROUND_NODES = frozenset({"0", "gnd"})
_ANALYSIS_CARDS = (".op", ".dc", ".ac", ".tran")
#: cards refused because they execute commands, pull in files, change the title or hide vectors
_FORBIDDEN_CARDS: dict[str, str] = {
    ".control": "a .control block executes arbitrary ngspice commands (including shell) when the file is sourced",
    ".endc": "a .control block executes arbitrary ngspice commands (including shell) when the file is sourced",
    ".include": ".include reads a file outside the netlist (the netlist must be self-contained)",
    ".lib": ".lib reads a file outside the netlist (the netlist must be self-contained)",
    ".title": "the first line is the title; a .title card would change what the plot records",
    ".save": ".save limits which vectors ngspice keeps; the runner records every vector of the plot",
}
#: node count per element letter, for the letters where it is fixed (E/G: only the two output nodes are
#: certain - the behavioural forms ``E1 out 0 value={...}`` have no controlling node pair)
_ELEMENT_NODES: dict[str, int] = {
    "R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2, "E": 2, "F": 2, "G": 2, "H": 2, "W": 2,
    "Q": 3, "J": 3, "Z": 3, "M": 4, "S": 4, "T": 4,
}
_NEEDS_VALUE = frozenset("RCL")
_NEEDS_MODEL = frozenset("DQJZM")


def _element_nodes(letter: str, toks: list[str]) -> list[str] | None:
    """The node tokens of an element line, or None when the letter's node list cannot be determined."""
    if letter in _ELEMENT_NODES:
        n = _ELEMENT_NODES[letter]
        return toks[1 : 1 + n]
    if letter == "X":
        body = toks[1:]
        while body and "=" in body[-1]:
            body.pop()
        return body[:-1] if len(body) >= 2 else []
    if letter == "A":
        nodes: list[str] = []
        for t in toks[1:-1]:
            t = t.strip("[]()")
            if t.startswith("%"):
                t = t.split("(", 1)[1] if "(" in t else ""
            if t:
                nodes.append(t)
        return nodes
    return None


def validate_deck(text: str) -> tuple[list[str], dict[str, str]]:
    """Refuse decks that crash, poison or silently mis-simulate in ngspice.dll; returns ``(problems, info)``.

    ``info["title"]`` is the first line. Checked: non-empty title; ``.end`` as
    the last non-empty line; no quote/backtick, control or non-ASCII
    characters; no analysis cards (the runner issues the analysis as a
    command, so a card would be inert and misleading), no ``.control`` /
    ``.include`` / ``.lib`` / ``.title`` / ``.save``; element lines with a
    name, their nodes and a value (R C L) or model (D Q J Z M); node names in
    ``A-Z a-z 0-9 _ . / + - : # @ [ ]`` (:data:`NODE_RE`); no duplicate element names; at least one
    non-ground node; no two node names that collide after lowercasing.
    """
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    problems: list[str] = []
    title = lines[0].strip() if lines else ""
    if not title:
        problems.append("first line (the title) is empty: ngspice would eat the first element instead")
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty or non_empty[-1].strip().lower() != ".end":
        problems.append(".end must be the last non-empty line")
    names: dict[str, int] = {}
    nodes: dict[str, set[str]] = {}
    depth = 0
    in_control = False
    for i, ln in enumerate(lines, 1):
        if any(ch in "'\"`" for ch in ln):
            problems.append(f"line {i}: quote/backtick characters are forbidden (an unbalanced ' poisons ngspice.dll for the rest of the process)")
        if any(ord(ch) > 126 or (ord(ch) < 32 and ch != "\t") for ch in ln):
            problems.append(f"line {i}: non-ASCII or control characters (ngspice turns non-ASCII node names into ______)")
        s = ln.strip()
        if i == 1 or not s or s[0] in "*+":
            continue
        low = s.lower()
        if s[0] == ".":
            card = low.split()[0]
            if card in _FORBIDDEN_CARDS:
                problems.append(f"line {i}: {_FORBIDDEN_CARDS[card]}")
                if card == ".control":
                    in_control = True
                elif card == ".endc":
                    in_control = False
            elif card in _ANALYSIS_CARDS:
                problems.append(f"line {i}: analysis card {s.split()[0]!r}: the runner issues the analysis as a command; remove .op/.dc/.ac/.tran cards")
            elif card == ".subckt":
                depth += 1
            elif card == ".ends":
                depth = max(0, depth - 1)
            continue
        if in_control:
            continue  # already rejected as a whole; its lines are commands, not elements
        toks = s.split()
        if not toks[0][0].isalpha():
            problems.append(f"line {i}: not an element or card line: {s!r}")
            continue
        letter = toks[0][0].upper()
        if depth == 0:
            key = toks[0].lower()
            if key in names:
                problems.append(f"line {i}: element name {toks[0]!r} already used on line {names[key]} (ngspice: 'device already exists, bail out')")
            names[key] = i
        n_nodes = _ELEMENT_NODES.get(letter)
        if n_nodes is not None:
            needed = 1 + n_nodes + (1 if letter in _NEEDS_VALUE or letter in _NEEDS_MODEL else 0)
            if len(toks) < needed:
                what = "a value" if letter in _NEEDS_VALUE else "a model" if letter in _NEEDS_MODEL else ""
                problems.append(
                    f"line {i}: {s!r} needs {n_nodes} nodes{' and ' + what if what else ''} "
                    "(ngspice ignores such a line with only a warning and simulates the rest)"
                )
                continue
        element_nodes = _element_nodes(letter, toks)
        if element_nodes is None:
            continue
        for t in element_nodes:
            if not NODE_RE.fullmatch(t):
                problems.append(f"line {i}: node name {t!r} contains characters ngspice mangles (allowed: A-Z a-z 0-9 _ . / + - : # @ [ ])")
            elif depth == 0:
                nodes.setdefault(t.lower(), set()).add(t)
    if not any(k not in GROUND_NODES for k in nodes):
        problems.append("the deck has no non-ground node: ngspice.dll crashes the host process on such a circuit")
    for k, variants in nodes.items():
        if len(variants) > 1:
            problems.append(f"node names {sorted(variants)} collide after ngspice lowercases them")
    return problems, {"title": title}


# ------------------------------------------------------------------------------------------ the engine


class _Capture:
    """Everything the DLL sends through its callbacks since the last ``clear()``."""

    def __init__(self) -> None:
        self.chars: list[str] = []  # raw SendChar strings: 'stdout ...' / 'stderr ...'
        self.stats: list[str] = []
        self.exits: list[tuple[int, bool, bool]] = []
        self.init_data: list[dict] = []
        self.bg: list[bool] = []

    def stdout(self) -> list[str]:
        return [s[7:] for s in self.chars if s.startswith("stdout ")]

    def stderr(self) -> list[str]:
        """The engine's stderr lines since ``clear()`` minus :data:`INFORMATIONAL_STDERR_RE` (kept in ``chars``)."""
        return [s[7:] for s in self.chars if s.startswith("stderr ") and not INFORMATIONAL_STDERR_RE.match(s[7:].strip())]

    def clear(self) -> None:
        self.chars.clear()
        self.stats.clear()
        self.exits.clear()
        self.init_data.clear()
        self.bg.clear()


@dataclass
class _Vector:
    name: str
    v_type: int
    type_name: str
    v_flags: int
    length: int
    complex: bool
    data: list  # list[float] or list[tuple[float, float]]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def bind_reset(lib) -> bool:
    """Bind ``ngSpice_Reset`` if the library exports it (KiCad's ngspice-46 does; Debian/Ubuntu ``libngspice0`` 42 does not).

    Without it a ControlledExit cannot be recovered from and marks the engine dead for the rest of the process
    (``engine_info()["reset_supported"]`` says which case applies). Returns whether the symbol was bound.
    """
    try:
        reset = lib.ngSpice_Reset
    except AttributeError:
        return False
    reset.argtypes = []
    reset.restype = c_int
    return True


class _Engine:
    """The process-wide ngspice.dll state machine. Use through :class:`NgspiceShared`; hold ``lock`` while running."""

    def __init__(self, dll: Path, codemodel_dir: Path | None) -> None:
        self.dll_path = dll
        self.codemodel_dir = codemodel_dir
        self.lock = threading.RLock()
        self.cap = _Capture()
        self._callbacks: list = []
        self.dead = False
        self.dead_reason = ""
        self.version = ""
        self.build = ""
        self.codemodels_loaded = False
        self.codemodel_errors: list[str] = []
        self.init_log: list[str] = []
        self.settings: list[str] = []
        self.self_test: dict = {}
        self._dll_dir = os.add_dll_directory(str(dll.parent)) if hasattr(os, "add_dll_directory") else None
        try:
            self.lib = ctypes.CDLL(str(dll))
        except OSError as e:
            raise ToolUnavailableError(f"cannot load {dll}: {e}") from e
        lib = self.lib
        try:
            lib.ngSpice_Init.restype = c_int
            lib.ngSpice_Init.argtypes = [SendChar, SendStat, ControlledExit, SendData, SendInitData, BGThreadRunning, c_void_p]
            lib.ngSpice_Command.argtypes = [c_char_p]
            lib.ngSpice_Command.restype = c_int
            lib.ngSpice_Circ.argtypes = [POINTER(c_char_p)]
            lib.ngSpice_Circ.restype = c_int
            lib.ngGet_Vec_Info.argtypes = [c_char_p]
            lib.ngGet_Vec_Info.restype = POINTER(VectorInfo)
            lib.ngSpice_CurPlot.argtypes = []
            lib.ngSpice_CurPlot.restype = c_char_p
            lib.ngSpice_AllPlots.argtypes = []
            lib.ngSpice_AllPlots.restype = POINTER(c_char_p)
            lib.ngSpice_AllVecs.argtypes = [c_char_p]
            lib.ngSpice_AllVecs.restype = POINTER(c_char_p)
            lib.ngSpice_running.argtypes = []
            lib.ngSpice_running.restype = c_bool
        except AttributeError as e:
            raise ToolUnavailableError(f"{dll} does not export the ngspice shared-library API: {e}") from e
        self.reset_supported = bind_reset(lib)
        try:
            self._init_engine()
        except EngineDead as e:
            raise ToolUnavailableError(f"ngspice.dll at {dll} failed to initialise: {e}") from e

    # -- callbacks and low-level calls -------------------------------------------------------------------------

    def _install_callbacks(self) -> int:
        cap = self.cap

        @SendChar
        def _send_char(s, ident, user):
            cap.chars.append((s or b"").decode("utf-8", "replace"))
            return 0

        @SendStat
        def _send_stat(s, ident, user):
            cap.stats.append((s or b"").decode("utf-8", "replace"))
            return 0

        @ControlledExit
        def _controlled_exit(status, immediate, quit_, ident, user):
            cap.exits.append((int(status), bool(immediate), bool(quit_)))
            return 0

        @SendData
        def _send_data(p, n, ident, user):
            return 0

        @SendInitData
        def _send_init_data(p, ident, user):
            a = p.contents
            vecs = []
            for i in range(a.veccount):
                vi = a.vecs[i].contents
                vecs.append(((vi.vecname or b"").decode("utf-8", "replace"), vi.pdvec is not None and vi.pdvec == vi.pdvecscale))
            cap.init_data.append(
                {
                    "plot": (a.type or b"").decode("utf-8", "replace"),
                    "plot_type": (a.name or b"").decode("utf-8", "replace"),
                    "title": (a.title or b"").decode("utf-8", "replace"),
                    "vecs": vecs,
                }
            )
            return 0

        @BGThreadRunning
        def _bg_running(noruns, ident, user):
            cap.bg.append(bool(noruns))
            return 0

        self._callbacks = [_send_char, _send_stat, _controlled_exit, _send_data, _send_init_data, _bg_running]
        return self.lib.ngSpice_Init(*self._callbacks, None)

    def _mark_dead(self, reason: str) -> EngineDead:
        self.dead = True
        self.dead_reason = reason
        return EngineDead(reason)

    def _cmd(self, command: str) -> int:
        if self.dead:
            raise EngineDead(self.dead_reason)
        try:
            return self.lib.ngSpice_Command(command.encode("utf-8"))
        except OSError as e:  # access violation surfaced by ctypes: the DLL state is undefined now
            raise self._mark_dead(f"ngspice.dll faulted during {command!r}: {e}")

    def _circ(self, lines: list[str]) -> int:
        if self.dead:
            raise EngineDead(self.dead_reason)
        arr = (c_char_p * (len(lines) + 1))(*[ln.encode("utf-8") for ln in lines], None)
        try:
            return self.lib.ngSpice_Circ(arr)
        except OSError as e:
            raise self._mark_dead(f"ngspice.dll faulted in ngSpice_Circ: {e}")

    @staticmethod
    def _strings(pp) -> list[str]:
        out: list[str] = []
        if not pp:
            return out
        i = 0
        while pp[i]:
            out.append(pp[i].decode("utf-8", "replace"))
            i += 1
        return out

    def cur_plot(self) -> str:
        p = self.lib.ngSpice_CurPlot()
        return p.decode("utf-8", "replace") if p else ""

    def all_plots(self) -> list[str]:
        return self._strings(self.lib.ngSpice_AllPlots())

    def all_vecs(self, plot: str) -> list[str]:
        return self._strings(self.lib.ngSpice_AllVecs(plot.encode("utf-8")))

    def read_vector(self, qualified_name: str) -> _Vector | None:
        """Copy one vector NOW: ``ngGet_Vec_Info`` returns a pointer to a single static struct."""
        p = self.lib.ngGet_Vec_Info(qualified_name.encode("utf-8"))
        if not p:
            return None
        v = p.contents
        n = v.v_length
        is_complex = bool(v.v_flags & VF_COMPLEX)
        if is_complex:
            data: list = [(v.v_compdata[i].cx_real, v.v_compdata[i].cx_imag) for i in range(n)] if (n and v.v_compdata) else []
        else:
            data = v.v_realdata[:n] if (n and v.v_realdata) else []  # slicing a POINTER(c_double) copies into a list
        name = (v.v_name or b"").decode("utf-8", "replace")
        return _Vector(
            name=name, v_type=v.v_type, type_name=SV_TYPE_NAMES.get(v.v_type, str(v.v_type)), v_flags=v.v_flags,
            length=n, complex=is_complex, data=data,
        )

    def read_plot(self, plot: str) -> dict[str, _Vector | None]:
        return {name: self.read_vector(f"{plot}.{name}") for name in self.all_vecs(plot)}

    # -- lifecycle -----------------------------------------------------------------------------------------------

    def _init_engine(self) -> None:
        self.cap.clear()
        rc = self._install_callbacks()
        if rc != 0:
            raise self._mark_dead(f"ngSpice_Init returned {rc}")
        log = list(self.cap.chars)
        unexpected = [s for s in self.cap.stderr() if s not in BENIGN_INIT_STDERR]
        self.cap.clear()
        for c in INIT_SETTINGS:
            self._cmd(c)
        self._cmd("version -s")
        out = "\n".join(self.cap.stdout())
        m = _VERSION_RE.search(out)
        self.version = m.group(1) if m else "unknown"
        m = _BUILD_RE.search(out)
        self.build = m.group(1).strip() if m else "unknown"
        log += self.cap.chars
        self.cap.clear()
        self._load_codemodels()  # no circuit is loaded here (fresh init / after Reset)
        log += self.cap.chars
        self.cap.clear()
        self._cmd("set")
        self.settings = self.cap.stdout()
        self.cap.clear()
        if unexpected:
            log.append(f"note: unexpected stderr during ngSpice_Init: {unexpected}")
        if self.version == "unknown":
            log.append("note: could not parse the engine version from 'version -s'")
        self.init_log = log
        self._self_test()

    def _load_codemodels(self) -> None:
        self.codemodel_errors = []
        if self.codemodel_dir is None:
            self.codemodel_errors.append("no code model directory (expected <KiCad>/lib/ngspice next to bin/ngspice.dll, or NGSPICE_CODEMODEL_DIR)")
        else:
            for cm in CODEMODELS:
                path = self.codemodel_dir / cm
                if not path.is_file():
                    self.codemodel_errors.append(f"code model {path} not found")
                    continue
                try:
                    check_path_for_command(path)
                except ValueError as e:
                    self.codemodel_errors.append(str(e))
                    continue
                self.cap.clear()
                self._cmd(f"codemodel '{path}'")
                err = self.cap.stderr()
                if err:
                    self.codemodel_errors.append(f"codemodel {cm}: " + " | ".join(err))
        self.codemodels_loaded = not self.codemodel_errors

    def _self_test(self) -> None:
        """Prove the engine works (divider 6.0 V) and, when code models loaded, that XSPICE works (gain 3.0 V)."""
        res: dict = {}
        self.unload_all()
        self.cap.clear()
        self._circ(["selftest divider", "V1 VIN 0 DC 12", "R1 VIN VOUT 10k", "R2 VOUT 0 10k", ".end"])
        self._cmd("op")
        v = self.read_vector(self.cur_plot() + ".vout")
        res["divider_vout"] = v.data[0] if v is not None and v.length else None
        res["divider_stderr"] = self.cap.stderr()
        self.unload_all()
        divider_ok = res["divider_vout"] is not None and abs(res["divider_vout"] - 6.0) < 1e-9 and not res["divider_stderr"]
        if self.codemodels_loaded:
            self.cap.clear()
            self._circ(["selftest gain", "V1 IN 0 DC 1.5", "A1 IN OUT gainblk", ".model gainblk gain(in_offset=0 gain=2.0 out_offset=0)", "R1 OUT 0 1k", ".end"])
            self._cmd("op")
            v = self.read_vector(self.cur_plot() + ".out")
            res["gain_vout"] = v.data[0] if v is not None and v.length else None
            res["gain_stderr"] = self.cap.stderr()
            self.unload_all()
            if res["gain_vout"] is None or abs(res["gain_vout"] - 3.0) > 1e-9 or res["gain_stderr"]:
                self.codemodels_loaded = False
                self.codemodel_errors.append(f"XSPICE self-test failed (gain block): {res}")
        res["ok"] = divider_ok
        self.self_test = res
        if not divider_ok:
            raise self._mark_dead(f"engine self-test failed: {res}")

    def unload_all(self) -> None:
        """Circuits stack (``setcirc`` lists them): remove every one, then free every plot."""
        for _ in range(64):
            self.cap.clear()
            self._cmd("remcirc")
            if any("there is no circuit loaded" in s for s in self.cap.stderr()):
                break
        else:
            raise self._mark_dead("could not unload all circuits (64 remcirc commands did not empty the stack)")
        self.cap.clear()
        self._cmd("destroy all")
        self.cap.clear()

    def recover(self) -> None:
        """After a ControlledExit: ``ngSpice_Reset`` + ``ngSpice_Init`` + settings + code models + self-test."""
        if self.dead:
            raise EngineDead(self.dead_reason)
        if not self.reset_supported:
            raise self._mark_dead(f"{self.dll_path} does not export ngSpice_Reset (ngspice < 44): the engine cannot be recovered after a ControlledExit; restart the process")
        try:
            rc = self.lib.ngSpice_Reset()
        except OSError as e:
            raise self._mark_dead(f"ngSpice_Reset faulted: {e}")
        if rc != 0:
            raise self._mark_dead(f"ngSpice_Reset returned {rc}")
        self._init_engine()

    # -- running ------------------------------------------------------------------------------------------------

    def _run_with_watchdog(self, bg_command: str, timeout_s: float) -> bool:
        """Issue ``bg_<command>``, wait for the BGThreadRunning start then finish event; ``bg_halt`` on timeout.
        Returns True when it timed out."""
        self.cap.bg.clear()
        self._cmd(bg_command)
        t0 = time.perf_counter()
        while not self.cap.bg and time.perf_counter() - t0 < 2.0:  # BGThreadRunning(noruns=False): started
            time.sleep(0.001)
        started = bool(self.cap.bg)
        timed_out = False
        while True:
            if len(self.cap.bg) >= 2:  # BGThreadRunning(noruns=True): finished
                break
            if started and not self.lib.ngSpice_running():
                time.sleep(0.005)
                if len(self.cap.bg) >= 2 or not self.lib.ngSpice_running():
                    break
            if time.perf_counter() - t0 > timeout_s:
                timed_out = True
                self._cmd("bg_halt")
                t1 = time.perf_counter()
                while self.lib.ngSpice_running() and time.perf_counter() - t1 < 10.0:
                    time.sleep(0.005)
                break
            time.sleep(0.002)
        return timed_out

    def write_rawfile(self, path: Path, binary: bool) -> list[str]:
        """``write '<path>'`` of the current plot; returns the stderr lines (empty on success)."""
        check_path_for_command(path)
        self.cap.clear()
        self._cmd("set filetype=binary" if binary else "set filetype=ascii")
        self._cmd(f"write '{path}'")
        self._cmd("unset filetype")
        return self.cap.stderr()

    @staticmethod
    def rawfile_dir(workdir: Path) -> tuple[Path | None, str | None]:
        """Where ngspice can be told to write the rawfile: ``workdir``, else the system temp directory.

        Returns ``(directory, problem)``: ``directory`` is None when neither
        path can be passed to a ``write`` command, and ``problem`` says why.
        """
        problems: list[str] = []
        for candidate in (workdir, Path(tempfile.gettempdir()).resolve()):
            try:
                check_path_for_command(candidate)
            except ValueError as e:
                problems.append(str(e))
                continue
            return candidate, None
        return None, "; ".join(problems)

    def run(
        self,
        netlist_path: Path,
        analysis: SpiceAnalysis,
        command: str,
        workdir: Path,
        *,
        timeout_s: float,
        raw_format: Literal["ascii", "binary"],
    ) -> SpiceResult:
        t_start = time.perf_counter()
        transcript: list[str] = []
        errors: list[str] = []
        # read ONCE: the hash recorded, the text validated and the lines ngspice parses are the same bytes
        data = netlist_path.read_bytes()
        base = dict(
            engine=ENGINE_ID, engine_version=self.version, netlist_path=str(netlist_path),
            netlist_hash="sha256:" + hashlib.sha256(data).hexdigest(), analysis=analysis, command=command,
        )

        def finish(succeeded: bool = False, **extra) -> SpiceResult:
            return SpiceResult(**base, log="\n".join(transcript), errors=errors, elapsed_s=time.perf_counter() - t_start, succeeded=succeeded, **extra)

        def drain() -> tuple[list[str], list[str], list[tuple[int, bool, bool]], list[dict]]:
            transcript.extend(self.cap.chars)
            phase = (self.cap.stdout(), self.cap.stderr(), list(self.cap.exits), list(self.cap.init_data))
            self.cap.clear()
            return phase

        text = data.decode("utf-8", "replace")
        problems, info = validate_deck(text)
        if problems:
            errors.extend(f"netlist rejected before reaching ngspice: {p}" for p in problems)
            return finish()
        title = info["title"]
        lines = [ln.rstrip("\r") for ln in text.split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        raw_dir, raw_problem = self.rawfile_dir(workdir)
        if raw_dir is None:
            errors.append(f"no directory the rawfile could be written to: {raw_problem}")
            return finish(unverifiable=f"the rawfile (the run's evidence) cannot be written: {raw_problem}")
        try:
            self.unload_all()
            self.cap.clear()
            rc = self._circ(lines)
            out, err, exits, _ = drain()
            circuit_seen = any(s.strip().lower() == f"circuit: {title.lower()}" for s in out)
            if rc != 0 or exits or err or not circuit_seen:
                errors.extend(err)
                if rc != 0:
                    errors.append(f"ngSpice_Circ returned {rc}")
                if exits:
                    errors.append(f"ngspice ControlledExit {exits} while loading the netlist")
                if not circuit_seen:
                    errors.append(f"ngspice did not announce 'Circuit: {title}' after loading the netlist {netlist_path}")
                self._after_failure(exits)
                return finish()
            self._cmd("listing")
            listing = self.cap.stdout()
            drain()
            pre_plots = self.all_plots()
            timed_out = self._run_with_watchdog(f"bg_{command}", timeout_s)
            out, err, exits, init_data = drain()
            rows = [int(m.group(1)) for s in out for m in [_ROWS_RE.search(s)] if m]
            plot = self.cur_plot()
            new_plot = plot if (plot and plot not in pre_plots) else None
            plot_type = init_data[-1]["plot_type"] if init_data else None
            plot_title = init_data[-1]["title"] if init_data else None
            errors.extend(err)
            fails: list[str] = []
            if timed_out:
                fails.append(f"timeout: {command!r} did not finish within {timeout_s} s (bg_halt issued, partial data discarded)")
            if exits:
                fails.append(f"ngspice ControlledExit {exits} during {command!r}")
            if err:
                fails.append(f"ngspice reported {len(err)} error/warning line(s) during {command!r}")
            if not rows:
                fails.append("ngspice printed no 'No. of Data Rows' line (analysis did not complete)")
            if new_plot is None or not plot.startswith(analysis.value):
                fails.append(f"no new {analysis.value} plot (current plot {plot!r}, plots before the command {pre_plots})")
            if plot_title is not None and plot_title.lower() != title.lower():
                fails.append(f"plot title {plot_title!r} is not the netlist title {title!r}")
            if fails:
                errors.extend(fails)
                self._after_failure(exits, timed_out)
                return finish(plot_name=new_plot, plot_type=plot_type, listing=listing, timed_out=timed_out)
            n_points = rows[-1]
            vecs = self.read_plot(plot)
            short = [n for n, v in vecs.items() if v is None or v.length != n_points]
            if not vecs or short:
                errors.append(f"plot {plot} vectors missing or not {n_points} long: {short or 'no vectors'}")
                self.unload_all()
                return finish(plot_name=plot, plot_type=plot_type, listing=listing)
            good: dict[str, _Vector] = {n: v for n, v in vecs.items() if v is not None}
            scale = self._scale_name(analysis, init_data, good)
            vectors: dict[str, list[float]] = {}
            vector_types: dict[str, str] = {}
            for name, v in good.items():
                vector_types[name] = v.type_name
                if v.complex:
                    vectors.update(rawfile.complex_convention(name, v.data))
                else:
                    vectors[name] = list(v.data)
            workdir.mkdir(parents=True, exist_ok=True)
            raw = workdir / f"{netlist_path.stem}.{analysis.value}.raw"
            if raw.exists():
                raw.unlink()
            if raw_dir == workdir:
                written = raw
            else:
                # the workdir cannot be named in an ngspice command: write next door and copy the bytes
                written = Path(tempfile.mkdtemp(prefix="ai_eda_raw_", dir=raw_dir)) / raw.name
            err = self.write_rawfile(written, binary=(raw_format == "binary"))
            drain()
            common = dict(plot_name=plot, plot_type=plot_type, vectors=vectors, vector_types=vector_types, scale=scale, n_points=n_points, listing=listing)
            if err or not written.is_file():
                errors.extend(err)
                errors.append(f"ngspice did not write the rawfile {written}")
                self.unload_all()
                return finish(**common)
            if written != raw:
                shutil.copyfile(written, raw)
                shutil.rmtree(written.parent, ignore_errors=True)
                transcript.append(f"note: rawfile written to {written} (workdir cannot be named in an ngspice command) and copied to {raw}")
            raw_hash = _sha256(raw)
            mismatch = self._rawfile_mismatch(raw, good, n_points, plot_type)
            self.unload_all()
            if mismatch:
                errors.append(f"rawfile {raw} disagrees with the vectors read through the API: {mismatch}")
                return finish(**common, raw_output_path=str(raw), raw_output_hash=raw_hash)
            return finish(True, **common, raw_output_path=str(raw), raw_output_hash=raw_hash)
        except EngineDead as e:
            transcript.extend(self.cap.chars)
            errors.append(f"ngspice.dll is unusable in this process: {e}")
            return finish(unverifiable=f"ngspice.dll is unusable in this process: {e}")

    def _after_failure(self, exits: list, timed_out: bool = False) -> None:
        if exits:
            self.recover()
        else:
            self.unload_all()
            if timed_out:
                self._self_test()

    @staticmethod
    def _scale_name(analysis: SpiceAnalysis, init_data: list[dict], vecs: dict[str, _Vector]) -> str | None:
        if analysis == SpiceAnalysis.OP:
            return None
        if init_data:
            flagged = [name for name, is_scale in init_data[-1]["vecs"] if is_scale]
            if len(flagged) == 1 and flagged[0] in vecs:
                return flagged[0]
        fallback = {SpiceAnalysis.TRAN: "time", SpiceAnalysis.AC: "frequency"}.get(analysis)
        if fallback in vecs:
            return fallback
        sweeps = [n for n in vecs if n.endswith("-sweep")]
        return sweeps[0] if len(sweeps) == 1 else None

    @staticmethod
    def _rawfile_mismatch(raw: Path, vecs: dict[str, _Vector], n_points: int, plot_type: str | None) -> str:
        """Compare the rawfile ngspice wrote with the API vectors; '' when consistent (binary exact, ASCII 1e-12 relative)."""
        try:
            rp = rawfile.parse(raw)
        except ValueError as e:
            return f"cannot parse it: {e}"
        if rp.n_points != n_points:
            return f"{rp.n_points} points in the file, {n_points} in the plot"
        if plot_type is not None and rp.plotname != plot_type:
            return f"plot type {rp.plotname!r} in the file, {plot_type!r} in the plot"
        if rp.n_variables != len(vecs):
            return f"{rp.n_variables} variables in the file, {len(vecs)} vectors in the plot"

        def same(a: float, b: float) -> bool:
            if rp.binary:
                return a == b
            return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))

        for name, v in vecs.items():
            raw_name = rawfile.raw_variable_name(name, v.v_type)
            expected = rawfile.complex_convention(raw_name, v.data) if v.complex else {raw_name: list(v.data)}
            for key, api_values in expected.items():
                file_values = rp.vectors.get(key)
                if file_values is None:
                    return f"variable {key!r} (plot vector {name!r}) missing from the file"
                if len(file_values) != len(api_values):
                    return f"variable {key!r} has {len(file_values)} samples in the file, {len(api_values)} in the plot"
                for k, (a, b) in enumerate(zip(file_values, api_values)):
                    if not same(a, b):
                        return f"variable {key!r} sample {k}: file {a!r} vs plot {b!r}"
        return ""


_ENGINE: _Engine | None = None
_ENGINE_ERROR: str | None = None
_ENGINE_LOCK = threading.Lock()


def _get_engine(dll: Path, codemodel_dir: Path | None) -> _Engine:
    """The one engine of this process (ngspice.dll can be loaded and initialised once per process)."""
    global _ENGINE, _ENGINE_ERROR
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            if _ENGINE.dll_path != dll:
                raise ToolUnavailableError(f"ngspice.dll is already loaded from {_ENGINE.dll_path}; a process can host only one ngspice shared library")
            if _ENGINE.dead:
                raise ToolUnavailableError(f"ngspice.dll is unusable in this process ({_ENGINE.dead_reason}); restart the process")
            return _ENGINE
        if _ENGINE_ERROR is not None:
            raise ToolUnavailableError(_ENGINE_ERROR)
        try:
            _ENGINE = _Engine(dll, codemodel_dir)
        except ToolUnavailableError as e:
            _ENGINE_ERROR = str(e)
            raise
        return _ENGINE


class NgspiceShared(SpiceRunner):
    """:class:`SpiceRunner` on KiCad's bundled ``ngspice.dll`` (see the module docstring for the rules).

    A process-wide singleton: ``NgspiceShared()`` always returns the same
    handle (one per explicit ``dll`` path), the DLL is loaded and initialised
    on first use and shared by every handle. ``available()`` only checks that
    the DLL file exists; ``version()``, ``codemodels_loaded`` and ``run()``
    initialise it and raise ``ToolUnavailableError`` when that fails.
    """

    engine = ENGINE_ID
    _instances: dict[str, NgspiceShared] = {}

    def __new__(cls, dll: str | Path | None = None, codemodel_dir: str | Path | None = None) -> NgspiceShared:
        key = str(Path(dll).resolve()) if dll else ""
        inst = cls._instances.get(key)
        if inst is None:
            inst = super().__new__(cls)
            inst._configured = False
            cls._instances[key] = inst
        return inst

    def __init__(self, dll: str | Path | None = None, codemodel_dir: str | Path | None = None) -> None:
        if self._configured:
            return
        self._dll: Path | None = Path(dll).resolve() if dll else find_ngspice_dll()
        self._codemodel_dir: Path | None = Path(codemodel_dir) if codemodel_dir else find_codemodel_dir(self._dll)
        self._configured = True

    @property
    def dll_path(self) -> Path | None:
        return self._dll

    @property
    def codemodel_dir(self) -> Path | None:
        return self._codemodel_dir

    def available(self) -> bool:
        return self._dll is not None and self._dll.is_file()

    def _engine(self) -> _Engine:
        if not self.available():
            raise ToolUnavailableError("ngspice.dll not found (set NGSPICE_DLL, or install KiCad, whose bin/ngspice.dll is used)")
        return _get_engine(self._dll, self._codemodel_dir)  # type: ignore[arg-type]

    def version(self) -> str:
        """``"ngspice-46"`` - parsed from ``version -s`` (also present in the init banner and every rawfile's ``Command:`` line)."""
        return self._engine().version

    def build(self) -> str:
        """The engine's build date string (``"Apr 14 2026   05:15:29"``)."""
        return self._engine().build

    @property
    def codemodels_loaded(self) -> bool:
        """True when the six KiCad code model libraries loaded and the XSPICE self-test passed."""
        return self._engine().codemodels_loaded

    def engine_log(self) -> list[str]:
        """Init banner, version output, code model errors, self-test result and notes - evidence of how the engine was set up."""
        eng = self._engine()
        return [*eng.init_log, *(f"codemodel error: {e}" for e in eng.codemodel_errors), f"self-test: {eng.self_test}"]

    def engine_settings(self) -> list[str]:
        """The ``set`` listing after init (a ``.spiceinit`` in the cwd or ``$SPICE_SCRIPTS/spinit`` would show up here)."""
        return list(self._engine().settings)

    def self_test(self) -> dict:
        return dict(self._engine().self_test)

    def engine_info(self) -> dict:
        """Everything about the engine's state a result depends on, for ``results.json`` and the ``spice`` details.

        ``dll_path``, ``version``, ``build``, ``codemodel_dir``,
        ``codemodels_loaded`` (+ ``codemodel_errors``), the ``set`` listing
        after init (``settings``, where a ``.spiceinit`` in the cwd or
        ``$SPICE_SCRIPTS/spinit`` would show) and its sha256
        (``settings_hash``), the self-test result and the notes of the init
        log - so a PASS can be traced to the engine configuration that
        produced it and two runs on differently configured engines never
        look alike.
        """
        eng = self._engine()
        settings = list(eng.settings)
        return {
            "engine": self.engine,
            "version": eng.version,
            "build": eng.build,
            "dll_path": str(eng.dll_path),
            "reset_supported": eng.reset_supported,
            "codemodel_dir": None if eng.codemodel_dir is None else str(eng.codemodel_dir),
            "codemodels_loaded": eng.codemodels_loaded,
            "codemodel_errors": list(eng.codemodel_errors),
            "settings": settings,
            "settings_hash": "sha256:" + hashlib.sha256("\n".join(settings).encode("utf-8")).hexdigest(),
            "self_test": dict(eng.self_test),
            "init_notes": [ln for ln in eng.init_log if ln.startswith("note:")],
        }

    def run(
        self,
        netlist_path: Path,
        analysis: SpiceAnalysis,
        workdir: Path,
        command: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        raw_format: Literal["ascii", "binary"] = "ascii",
    ) -> SpiceResult:
        """Run one analysis of the netlist file at ``netlist_path``.

        The file's bytes are read once, hashed (``netlist_hash``), validated
        and handed to ``ngSpice_Circ`` - the path itself never reaches an
        ngspice command. ``command`` is the interactive analysis command
        (``None`` = ``"op"``, only for ``SpiceAnalysis.OP``), normalised by
        :func:`~ai_eda.tools.spice.runner.normalise_command` (``ValueError`` for
        a malformed one). The rawfile ngspice writes for the plot goes to
        ``<workdir>/<stem>.<analysis>.raw`` (``raw_format`` ``"ascii"`` by
        default, ``"binary"`` for a bit-exact copy; written via the system
        temp directory when the workdir cannot be named in a ``write``
        command, ``unverifiable`` when neither can). The command runs in
        ngspice's background thread and is halted after ``timeout_s`` (the
        result then has ``timed_out=True`` and ``succeeded=False``).
        Simulation failures come back as ``succeeded=False`` with ``errors``;
        a missing netlist file is ``ToolExecutionError``; a missing or dead
        engine is ``ToolUnavailableError``.
        """
        netlist_path = Path(netlist_path).resolve()
        if not netlist_path.is_file():
            raise ToolExecutionError(f"netlist {netlist_path} does not exist")
        analysis = SpiceAnalysis(analysis)
        cmd = normalise_command(command, analysis)
        if raw_format not in ("ascii", "binary"):
            raise ValueError(f"raw_format must be 'ascii' or 'binary', got {raw_format!r}")
        eng = self._engine()
        with eng.lock:
            if eng.dead:
                raise ToolUnavailableError(f"ngspice.dll is unusable in this process ({eng.dead_reason}); restart the process")
            return eng.run(netlist_path, analysis, cmd, Path(workdir).resolve(), timeout_s=timeout_s, raw_format=raw_format)


__all__ = [
    "CODEMODELS",
    "DEFAULT_TIMEOUT_S",
    "ENGINE_ID",
    "NODE_RE",
    "NgspiceShared",
    "check_path_for_command",
    "find_codemodel_dir",
    "find_ngspice_dll",
    "validate_deck",
]
