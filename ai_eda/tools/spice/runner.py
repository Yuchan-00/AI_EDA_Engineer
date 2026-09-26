"""SPICE runner interface and result model.

Invariants this module enforces:

* A :class:`SpiceResult` exists only because a real engine ran a real netlist
  file: it records the engine, its version, the path and sha256 of the file
  that was loaded, and the exact command that produced the plot.
* ``succeeded`` is never inferred from a return code alone (ngspice's are
  nearly meaningless); each runner decides it from the engine's own messages
  and the presence of a fresh plot with full-length vectors. A failed
  simulation is a result with ``succeeded=False`` and ``errors``, never an
  exception. Exceptions are reserved for a missing engine
  (``ToolUnavailableError``) and caller mistakes (missing netlist file ->
  ``ToolExecutionError``, malformed command -> ``ValueError``).
* Vectors are stored under ngspice's plot vector names (node names
  lowercased: ``vout``, ``v1#branch``, ``time``, ``v-sweep``). Complex (AC)
  vectors use the convention of :mod:`ai_eda.tools.spice.rawfile`: ``<name>``
  is the magnitude, ``<name>.phase_deg`` the phase in degrees and
  ``<name>.real`` / ``<name>.imag`` the components.
* A value *between* samples is an interpolation, and says so:
  :meth:`SpiceResult.interpolate` returns the two bracketing samples and the
  method with the number (linear on the scale; on an ac sweep linear in
  log frequency, and in log magnitude for a magnitude vector, which halves
  and quarters the error of plain linear interpolation on a log-spaced grid
  - measured 1.3 % -> 0.33 % on a 10 points/decade RC low-pass). Whether the
  grid is fine enough for a tolerance is the judge's decision
  (:func:`ai_eda.tools.spice.stage.judge`), never hidden in the number.
* A run the *environment* prevents (the rawfile cannot be written anywhere,
  the engine is unusable) is ``succeeded=False`` with ``unverifiable`` set:
  it is a tool limitation, reported as NOT_VERIFIED by the stage, not a
  verdict about the design.
* Analysis commands go through :func:`normalise_command` before they reach
  an engine: lowercased (ngspice lowercases the deck but matches device
  names in interactive commands case-sensitively, so ``dc VVIN 0 12 1``
  fails where ``dc vvin 0 12 1`` runs), restricted to ``op``/``dc``/``ac``/
  ``tran`` matching the requested analysis, and limited to plain tokens
  (a ``;`` would chain a second ngspice command).
"""

from __future__ import annotations

import builtins
import hashlib
import math
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from ai_eda.errors import ToolExecutionError, ToolUnavailableError
from ai_eda.tools.spice import rawfile


class SpiceAnalysis(StrEnum):
    OP = "op"
    DC = "dc"
    AC = "ac"
    TRAN = "tran"


#: suffixes of the complex-convention parts that are not magnitudes (log-magnitude interpolation does not apply)
_COMPLEX_PART_SUFFIXES = (".phase_deg", ".real", ".imag")


class Interpolation(BaseModel):
    """One number read off a sweep, with the samples it came from.

    ``exact`` - ``x`` hit a sample; ``x0/y0/x1/y1`` are then that sample
    twice. Otherwise the two bracketing samples and ``method``: ``"linear"``,
    ``"log-x"`` (linear in log of the scale) or ``"log-log"`` (also linear
    in log of the value). ``span`` is ``|y1 - y0|``: how far apart the two
    neighbours are, the plain bound on what the interpolation can be trusted
    to when the curve is monotonic between them.
    """

    value: float
    exact: bool
    x0: float
    y0: float
    x1: float
    y1: float
    method: str

    @property
    def span(self) -> float:
        return abs(self.y1 - self.y0)

    @property
    def low(self) -> float:
        return builtins.min(self.y0, self.y1, self.value)

    @property
    def high(self) -> float:
        return builtins.max(self.y0, self.y1, self.value)


#: characters allowed in one token of an interactive analysis command
_COMMAND_TOKEN = re.compile(r"[a-z0-9_.+\-]+")


def normalise_command(command: str | None, analysis: SpiceAnalysis | str) -> str:
    """The command a runner issues for ``analysis``: lowercased, whitespace-collapsed, validated.

    ``None`` means the analysis' default, which exists only for ``op``; the
    other analyses need their arguments (``"dc v1 0 12 1"``, ``"tran 10u 5m"``,
    ``"ac dec 10 1 1meg"``). ``ValueError`` when the command does not start
    with the analysis keyword, ``op`` has arguments, or a token contains
    anything but ``a-z 0-9 _ . + -`` (so no ``;``, quotes, ``$`` or
    backticks reach ngspice).
    """
    analysis = SpiceAnalysis(analysis)
    if command is None:
        if analysis != SpiceAnalysis.OP:
            raise ValueError(f"analysis {analysis.value!r} needs an explicit command (e.g. 'dc v1 0 12 1', 'tran 10u 5m', 'ac dec 10 1 1meg')")
        command = analysis.value
    tokens = command.split()
    if not tokens:
        raise ValueError("empty SPICE command")
    lowered = [t.lower() for t in tokens]
    bad = [t for t in lowered if not _COMMAND_TOKEN.fullmatch(t)]
    if bad:
        raise ValueError(
            f"command {command!r}: token(s) {bad} contain characters that are not allowed "
            "(letters, digits, _ . + - only; ';' would chain a second ngspice command)"
        )
    if lowered[0] != analysis.value:
        raise ValueError(f"command {command!r} does not start with the analysis keyword {analysis.value!r}")
    if analysis == SpiceAnalysis.OP and len(lowered) != 1:
        raise ValueError(f"command {command!r}: 'op' takes no arguments")
    if analysis != SpiceAnalysis.OP and len(lowered) < 2:
        raise ValueError(f"command {command!r}: {analysis.value} needs arguments")
    return " ".join(lowered)


class SpiceResult(BaseModel):
    """What one analysis of one netlist file produced, with the evidence to check it.

    ``vectors`` holds every vector of the plot, keyed by ngspice's plot
    vector name (deck node names lowercased; ``<src>#branch`` for source
    currents; the scale ``time`` / ``frequency`` / ``v-sweep`` ...). Complex
    vectors (AC) are expanded as ``<name>`` (magnitude), ``<name>.phase_deg``,
    ``<name>.real``, ``<name>.imag`` - including the ``frequency`` scale, whose
    magnitude is the frequency. ``vector_types`` gives ngspice's type word
    per plot vector (``voltage``, ``current``, ``time``, ``frequency`` ...).

    ``raw_output_path`` is the rawfile ngspice itself wrote for the plot
    (``write``), ``raw_output_hash`` its sha256; the rawfile carries a date
    line, so it is evidence, not a deterministic artifact. ``log`` is every
    line the engine sent through its output callbacks (``stdout ...`` /
    ``stderr ...`` prefixed) or printed, ``errors`` the lines it flagged plus
    the runner's own findings; ``listing`` is the deck as ngspice parsed it
    (lowercased and numbered) when the load succeeded.
    """

    #: ``"ngspice-shared"`` (KiCad's ngspice.dll through ctypes) or ``"ngspice"`` (batch binary)
    engine: str
    #: e.g. ``"ngspice-46"``
    engine_version: str
    netlist_path: str
    #: ``sha256:<hex>`` of the file that was loaded
    netlist_hash: str
    analysis: SpiceAnalysis
    #: the exact command issued (normalised: lowercase, single spaces), e.g. ``"tran 10u 5m"``
    command: str
    #: ngspice plot name (``op1``, ``dc1``, ``ac1``, ``tran1``); None when no plot was produced
    plot_name: str | None = None
    #: ngspice plot type (``Operating Point``, ``DC transfer characteristic``, ``AC Analysis``, ``Transient Analysis``)
    plot_type: str | None = None
    vectors: dict[str, list[float]] = Field(default_factory=dict)
    vector_types: dict[str, str] = Field(default_factory=dict)
    #: name of the sweep vector (``time``, ``frequency``, ``v-sweep`` ...); None for an operating point
    scale: str | None = None
    n_points: int = 0
    log: str = ""
    errors: list[str] = Field(default_factory=list)
    raw_output_path: str | None = None
    raw_output_hash: str | None = None
    #: the batch runner simulates a copy of the netlist with the analysis card appended: its path and sha256
    #: (``netlist_path`` / ``netlist_hash`` are the original's, which is what the copy is made from)
    deck_path: str | None = None
    deck_hash: str | None = None
    listing: list[str] = Field(default_factory=list)
    elapsed_s: float = 0.0
    timed_out: bool = False
    #: why the *environment* (not the design) prevented a verified result: the rawfile could not be
    #: written anywhere, the engine is dead ... The stage reports such a run as NOT_VERIFIED, not FAIL.
    unverifiable: str | None = None
    succeeded: bool

    # -- reductions for the SimulationAgent ------------------------------------------------------------------

    def vector(self, name: str) -> list[float]:
        """Samples of ``name``: a plot vector name (``vout``, ``v1#branch``, ``out.phase_deg``) or the
        rawfile form (``v(VOUT)``, ``i(V1)``); lookup is case-insensitive. ``KeyError`` when absent."""
        for candidate in (name, name.lower(), rawfile.canonical_name(name.lower())):
            if candidate in self.vectors:
                return self.vectors[candidate]
        raise KeyError(f"no vector {name!r} in this result (available: {sorted(self.vectors)})")

    def scale_values(self) -> list[float]:
        if self.scale is None:
            raise ValueError(f"a {self.analysis.value} result has no sweep scale")
        return self.vector(self.scale)

    def interpolate(self, vector: str, x: float) -> Interpolation:
        """``vector`` at scale value ``x``, with the two samples it was read between.

        An exact scale hit returns that sample (``exact=True``). Otherwise the
        first segment that brackets ``x`` (a non-monotonic scale - nested dc
        sweeps are flattened by ngspice - uses the first one) is interpolated
        linearly; on an ac sweep linearly in ``log(x)``, and for a magnitude
        vector (not ``.phase_deg`` / ``.real`` / ``.imag``) with two positive
        neighbours also in ``log(y)``. ``x`` outside the scale's range is a
        ``ValueError`` (no extrapolation, no clamping).
        """
        xs = self.scale_values()
        ys = self.vector(vector)
        if len(xs) != len(ys):
            raise ValueError(f"{vector!r} has {len(ys)} samples but the scale {self.scale!r} has {len(xs)}")
        for k, xk in enumerate(xs):
            if xk == x:
                return Interpolation(value=ys[k], exact=True, x0=xk, y0=ys[k], x1=xk, y1=ys[k], method="exact")
        for k in range(len(xs) - 1):
            x0, x1 = xs[k], xs[k + 1]
            if not ((x0 < x < x1) or (x1 < x < x0)):
                continue
            y0, y1 = ys[k], ys[k + 1]
            log_x = self.analysis == SpiceAnalysis.AC and x0 > 0 and x1 > 0 and x > 0
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0)) if log_x else (x - x0) / (x1 - x0)
            log_y = log_x and y0 > 0 and y1 > 0 and not vector.lower().endswith(_COMPLEX_PART_SUFFIXES)
            if log_y:
                value = math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * t)
                method = "log-log"
            else:
                value = y0 + (y1 - y0) * t
                method = "log-x" if log_x else "linear"
            return Interpolation(value=value, exact=False, x0=x0, y0=y0, x1=x1, y1=y1, method=method)
        lo, hi = builtins.min(xs), builtins.max(xs)
        raise ValueError(f"{self.scale}={x} is outside the simulated range [{lo}, {hi}]")

    def value_at(self, vector: str, x: float) -> float:
        """The number of :meth:`interpolate` alone (an exact sample, or the interpolation between neighbours)."""
        return self.interpolate(vector, x).value

    def final(self, vector: str) -> float:
        """The last sample (an op result's single value, a tran result's end value)."""
        data = self.vector(vector)
        if not data:
            raise ValueError(f"vector {vector!r} is empty")
        return data[-1]

    def max(self, vector: str) -> float:
        data = self.vector(vector)
        if not data:
            raise ValueError(f"vector {vector!r} is empty")
        return builtins.max(data)

    def min(self, vector: str) -> float:
        data = self.vector(vector)
        if not data:
            raise ValueError(f"vector {vector!r} is empty")
        return builtins.min(data)


class SpiceRunner(ABC):
    engine: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None) -> SpiceResult:
        """Run one analysis of the netlist file. ``command`` is the interactive analysis command
        (``None`` = the analysis' default, only defined for ``op``); the result is one-per-analysis."""

    def engine_info(self) -> dict:
        """How the engine is set up, for the evidence of every run: at least ``engine`` and ``version``.

        Engines with more state (a build date, code models, an init-time
        settings listing where a ``.spiceinit`` / ``spinit`` would show, a
        self-test) add it here; the stage persists the whole dict in
        ``results.json`` and the ``spice`` result details.
        """
        return {"engine": self.engine, "version": self.version()}

    @staticmethod
    def netlist_hash(netlist_path: Path) -> str:
        return "sha256:" + hashlib.sha256(Path(netlist_path).read_bytes()).hexdigest()


def result_from_rawfile(
    path: Path,
    *,
    engine: str,
    engine_version: str,
    netlist_path: Path | str,
    netlist_hash: str,
    analysis: SpiceAnalysis,
    command: str,
    log: str = "",
    errors: list[str] | None = None,
    deck_path: str | None = None,
    deck_hash: str | None = None,
) -> SpiceResult:
    """Build a :class:`SpiceResult` from a rawfile ngspice wrote (batch ``-r`` or ``write``).

    Rawfile variable names are mapped back to plot vector names
    (``v(vout)`` -> ``vout``, ``i(v1)`` -> ``v1#branch``) and complex plots use
    the shared complex convention. The plot type must match ``analysis`` and
    hold at least one point, otherwise ``succeeded`` is False. A rawfile the
    parser rejects is ``ValueError`` (the caller decides how to report it).
    """
    path = Path(path)
    plot = rawfile.parse(path)
    errs = list(errors or [])
    expected = rawfile.PLOTNAMES[SpiceAnalysis(analysis).value]
    if plot.plotname != expected:
        errs.append(f"rawfile {path} holds a {plot.plotname!r} plot, expected {expected!r} for {analysis}")
    if plot.n_points < 1:
        errs.append(f"rawfile {path} holds no data points")
    scale = rawfile.canonical_name(plot.scale) if plot.scale else None
    return SpiceResult(
        engine=engine,
        engine_version=engine_version,
        netlist_path=str(netlist_path),
        deck_path=deck_path,
        deck_hash=deck_hash,
        netlist_hash=netlist_hash,
        analysis=SpiceAnalysis(analysis),
        command=command,
        plot_name=None,
        plot_type=plot.plotname,
        vectors=plot.as_plot_vectors(),
        vector_types={rawfile.canonical_name(name): typ for name, typ in plot.variables},
        scale=scale,
        n_points=plot.n_points,
        log=log,
        errors=errs,
        raw_output_path=str(path),
        raw_output_hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        succeeded=not errs,
    )


_VERSION_RE = re.compile(r"ngspice-\d+")


class NgspiceRunner(SpiceRunner):
    """ngspice *batch binary* runner (``ngspice -b -r <raw> <deck>``); optional, used only when an
    ``ngspice`` executable is on PATH (or given). Prefer :class:`ai_eda.tools.spice.NgspiceShared`.

    Batch mode runs the analysis cards inside the deck, so the analysis command
    is written as a card into a copy of the netlist
    (``<workdir>/<stem>.<analysis>.cir``) and *that* file is what ran:
    ``netlist_path`` / ``netlist_hash`` refer to it. The result is read from the
    rawfile with :func:`result_from_rawfile`.
    """

    engine = "ngspice"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("ngspice")

    def available(self) -> bool:
        return self.binary is not None

    def version(self) -> str:
        if not self.available():
            raise ToolUnavailableError("ngspice not found on PATH")
        out = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
        )
        m = _VERSION_RE.search(out.stdout or "")
        if m:
            return m.group(0)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"

    @staticmethod
    def batch_deck(text: str, command: str) -> str:
        """The deck text with ``.<command>`` inserted before the final ``.end`` (appended with ``.end`` if absent)."""
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().lower() == ".end":
            lines.pop()
        return "\n".join([*lines, f".{command}", ".end"]) + "\n"

    def run(
        self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None, *, timeout_s: float = 600.0
    ) -> SpiceResult:
        if not self.available():
            raise ToolUnavailableError("ngspice not found on PATH")
        netlist_path = Path(netlist_path)
        if not netlist_path.is_file():
            raise ToolExecutionError(f"netlist {netlist_path} does not exist")
        analysis = SpiceAnalysis(analysis)
        cmd = normalise_command(command, analysis)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        deck = workdir / f"{netlist_path.stem}.{analysis.value}.cir"
        deck.write_text(self.batch_deck(netlist_path.read_text(encoding="utf-8"), cmd), encoding="utf-8", newline="\n")
        raw = workdir / f"{netlist_path.stem}.{analysis.value}.raw"
        if raw.exists():
            raw.unlink()
        proc = subprocess.run(
            [self.binary, "-b", "-r", str(raw), str(deck)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            timeout=timeout_s,
        )
        from ai_eda.tools.spice.ngspice_shared import INFORMATIONAL_STDERR_RE  # the same informational lines as the shared runner

        log = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        errors = [ln for ln in proc.stderr.splitlines() if ln.strip() and not INFORMATIONAL_STDERR_RE.match(ln.strip())]
        if proc.returncode != 0:
            errors.append(f"ngspice exited {proc.returncode}")
        base = dict(
            engine=self.engine,
            engine_version=self.version(),
            netlist_path=str(netlist_path),
            netlist_hash=self.netlist_hash(netlist_path),
            deck_path=str(deck),
            deck_hash=self.netlist_hash(deck),
            analysis=analysis,
            command=cmd,
        )
        if errors or not raw.is_file():
            if not raw.is_file():
                errors.append(f"ngspice wrote no rawfile at {raw}")
            return SpiceResult(**base, log=log, errors=errors, succeeded=False)
        try:
            return result_from_rawfile(raw, log=log, **base)
        except ValueError as e:
            return SpiceResult(**base, log=log, errors=[f"unreadable rawfile: {e}"], raw_output_path=str(raw), succeeded=False)


__all__ = [
    "Interpolation",
    "NgspiceRunner",
    "SpiceAnalysis",
    "SpiceResult",
    "SpiceRunner",
    "normalise_command",
    "result_from_rawfile",
]
