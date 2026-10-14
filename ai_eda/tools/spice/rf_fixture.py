"""``spice.rf.<network>[.<state>].<exp>``: S-parameters of the IR's RF fixture networks, measured by an ngspice ac sweep.

Invariant: a fixture verdict is ngspice evidence about a *network of the IR's
own parts under their bound values* - never an opinion, never a statement
about a real part, the board or the radio. Each :class:`~ai_eda.ir.rf.RFNetwork`
of ``ir.rf.networks`` (read by attribute: the names are the wave-1 contract of
the RF IR) is simulated like this, one deck per (network, state, drive port):

* **Only the members.** The deck holds the network's ``members`` (IR
  component refs) and the design nets restricted to their pins, nothing else
  of the design. Each member is simulated with the state's binding, else the
  network's fixture binding, else its own ``Component.spice``; a member
  without any, or whose own binding excludes it from the design netlist
  while the network gives no fixture binding, is refused (the deck would
  silently lose a part). A net that also reaches a non-member is open in the
  fixture unless a port names it; the summary lists such nets.
* **The runner adds exactly four kinds of element**, each listed in the
  summary's ``details["decks"][*]["added"]``: (1) at the drive port an ac
  source of magnitude 1 (DC level = the state's ``port_dc_v`` for that port,
  else 0) behind a series resistor equal to the port's ``z0_ohm``; (2) at
  every other ``"port"`` a load resistor equal to its ``z0_ohm`` (to a DC
  source at the state's level when the state names one); an ideal DC source
  at every ``"rail"`` (the state's level, else the port's ``voltage_v``) and
  every ``"control"`` port (the state's level) - an ideal source is also an
  ac short at that node, which is what a bypassed rail or a bias feed is;
  nothing at a ``"probe"`` (high impedance); (3) for each inductor in
  ``loss_q``, a series resistor ``R = 2 pi q_ref_hz L / Q`` (the loss model
  of the design's ``calc.rf.resonator.top_c.s21_db`` / ``calc.rf.pm.tank_phase``);
  (4) a :data:`DC_PATH_OHM` (1e12 ohm) resistor to ground at every node that
  has no DC path to ground otherwise (a graph walk: R, L, V, D, Q, the
  drain / source / bulk of M and both conductors of a T line conduct; C,
  I and X - whose inside the walk cannot see - do not). ngspice needs a DC
  path to solve the operating point before the ac sweep; each such resistor
  adds 1e-12 S from its node to ground, i.e. it moves a node of impedance
  ``|Z|`` by at most ``|Z|`` x 1e-12 relative (1e-6 even at 1 Mohm) - the
  bound is written into the details beside the list.
* **The same compiler and whitelist.** The deck is an IR compiled by
  :func:`ai_eda.compilers.spice.build` (no ``llm_generated`` value, binding,
  source or analysis; model cards authoritative or user-given; the runner's
  deck check); the network's ``sweep`` specs must all be ``ac`` analyses and
  are issued as the compiler's analysis commands on the engine in
  ``tools["spice"]``, followed by the runner's own **point analyses**: one
  ``ac lin 1 f f`` per distinct ``at`` / ``ref_at`` of the deck's
  expectations (ids ``rf_at_<k>`` - :data:`POINT_PREFIX`, a prefix no
  network sweep may use -, their frequencies in
  ``details["decks"][*]["points"]``, their numbers of the runner's
  ``derived`` provenance), compiled by the same compiler and whitelist
  (the deck text carries no analysis card, so its hash does not change).
  Nothing else: no ``.save`` / ``.meas`` / ``sp`` / ``noise`` card or
  command exists here.
* **S-parameters from complex node voltages**, with real reference
  impedances (power waves): ``S21 = 2 V_to / V_s * sqrt(R_drive / R_to)``,
  ``S11 = 2 V_drive / V_s - 1``, ``V_s`` the source's own ac voltage (read
  from its node, 1 V), every voltage relative to its port's reference net.
  "S21" to a probe (no load, no reference impedance) is the voltage ratio
  ``2 V_to / V_s``: only its phase (``phase21_deg``) and relative level
  (``rel_s21_db``) mean anything, so an absolute ``s21_db`` to a probe is
  refused. ``rel_s21_db`` = ``s21_db(at) - s21_db(ref_at)``. A phase is
  compared on the 360-degree branch nearest its nominal.
* **Judged, never interpreted.** An expectation is read only at its own
  frequency - a sample of its point analysis (a sample within
  :data:`HIT_REL` relative of ``at`` is ``at``: ngspice's number parser is
  not correctly rounded), recorded as ``details["sample_hz"]`` - never
  between two sweep points: the two-sample bracket bounds the true value
  only where the response is monotonic between them, and a resonance, a
  notch or a ripple extremum between two points is not. The exact number
  is judged by :func:`ai_eda.tools.spice.stage.judge`: PASS inside the
  tolerance (or on the passing side of a one-sided ``bound``), FAIL
  outside it. A probe (no verdict) may still be read between sweep points,
  linearly in log frequency (dB and unwrapped phase), with its bracket
  recorded. A dB or degree level needs ``tol_abs`` or a ``bound`` (never
  ``tol_rel``: a relative tolerance on a logarithm or an angle is no
  tolerance). An **exact zero** |S| (a perfect match or isolation in the
  double-precision solve, below about -300 dB: whether it lands on 0.0 or
  on 1e-16 is rounding) is a level of ``-inf`` dB and is judged like any
  other: PASS on an ``at_most`` bound (with a hint: a port shorted to its
  reference reads 0 too), FAIL on an ``at_least`` bound or a ``tol_abs``
  band (outside every finite band); ``details["measured"]`` is then
  ``None`` beside ``details["zero_magnitude"]`` (JSON has no infinity) and
  every other infinity in the details is the text ``"-inf"`` / ``"+inf"``.
  A zero S at a ``rel_s21_db`` row's ``ref_at`` (no level to be relative
  to), the phase of a zero S (no phase), a non-finite magnitude, a
  frequency no analysis sampled, a vector ngspice did not write or a
  malformed row (a probe as the drive, an absolute S21 to a probe ...) is
  FAIL for a human with the reason; a structural mistake in the network
  (unknown member, dangling port, a member the deck would lose ...) FAILs
  its summary and leaves its rows NOT_VERIFIED ("not simulated") - never
  repaired at run time. The port order is free: each row names its drive.
* **An assumption is not evidence.** A PASS that rests on an
  ``assumption``-provenance value (anything the deck compiler lists, or the
  expectation's own numbers, a port impedance, a ``loss_q``) is NOT_VERIFIED
  naming it; ``llm_generated`` numbers are refused. A PASS message always
  says what it is: a network verdict under the confirmed model values, not a
  measured part, and a schematic-level network (no track, via or
  ground-return inductance).
* **Evidence.** Every result carries the deck file and the rawfile(s) it was
  read from as :class:`~ai_eda.ir.Evidence` with their sha256, the deck's hash
  as ``artifact_hash``, the design hash as ``ir_hash``, and the engine; the
  rawfiles carry ngspice's date line, so they are evidence, not
  deterministic artifacts (the deck text is deterministic). Each expectation
  records ``details["measured"]`` (dB or degrees; ``None`` for an exact zero,
  see above), ``details["f_hz"]``,
  ``details["state"]`` and the state's port DC levels
  ``details["port_dc_v"]``; probes are recorded in the network summary
  without a verdict. Decks and run directories live under :data:`RF_DIR` at
  a stem unique to the exact (network, state, drive) key
  (:func:`~ai_eda.tools.spice.si_check.deck_stem`).
* **One summary per network** (``spice.rf.<network>``): the worst of its
  expectations, NOT_VERIFIED without any (a network that states nothing
  verifies nothing), FAIL when the network is malformed, a deck does not
  compile or an analysis fails for a reason other than the environment; no
  engine is NOT_VERIFIED. Every ``spice.rf.*`` result recorded earlier that
  this run does not produce gets a superseding NOT_APPLICABLE, so a removed
  network or expectation never keeps its old verdict.
"""

from __future__ import annotations

import cmath
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ai_eda.compilers.spice import build, build_report
from ai_eda.errors import CompileError, ToolExecutionError, ToolUnavailableError
from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Component,
    Evidence,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    ValidationResult,
    ValidationStatus,
    worst_status,
)
from ai_eda.ir.rf import POINT_ANALYSIS_PREFIX
from ai_eda.tools.spice.runner import Interpolation, SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice.si_check import deck_stem
from ai_eda.tools.spice.stage import NGSPICE_DEFAULT_TEMP_C, RF_CHECK_PREFIX, judge

if TYPE_CHECKING:  # the RF IR type (read by attribute; the module never needs it at run time)
    from ai_eda.ir.rf import RFDesign

#: check id prefix: ``spice.rf.<network>`` (summary) and ``spice.rf.<network>[.<state>].<expectation>``
CHECK_PREFIX = RF_CHECK_PREFIX
#: where the decks and run directories go, under the workdir
RF_DIR = "spice_rf"
#: the DC-path resistor (module docstring): 1e-12 S from a node without a DC path to ground
DC_PATH_OHM = 1e12
#: a sample this close to a row's frequency (relative) is that frequency: ngspice's number parser is not correctly
#: rounded, so the one point of ``ac lin 1 f f`` may come back an ULP or two off ``f`` (about 2e-16 relative measured)
HIT_REL = 1e-12
#: the ids of the runner's single-point analyses (``rf_at_0``, ``rf_at_1`` ...): a network's own sweep may not use the prefix
POINT_PREFIX = POINT_ANALYSIS_PREFIX
#: what an exact zero is, in every message that reports one
ZERO_NOTE = "-inf dB (|S| = 0 in the double-precision solve, i.e. below about -300 dB)"
#: version of this runner (the provenance of every element it adds)
FIXTURE_VERSION = "0.1"
TOOL = "spice.rf"
#: the four quantities an expectation or probe may name
S21_DB, S11_DB, REL_S21_DB, PHASE21_DEG = "s21_db", "s11_db", "rel_s21_db", "phase21_deg"
QUANTITIES: tuple[str, ...] = (S21_DB, S11_DB, REL_S21_DB, PHASE21_DEG)
#: port kinds (module docstring): an RF port with a reference impedance, a high-impedance probe, a rail, a DC control
PORT, PROBE, RAIL, CONTROL = "port", "probe", "rail", "control"
PORT_KINDS: tuple[str, ...] = (PORT, PROBE, RAIL, CONTROL)
BOUNDS: tuple[str, ...] = ("at_least", "at_most")
#: the unit a quantity's nominal may state (lower-cased)
_UNITS: dict[str, frozenset[str]] = {
    S21_DB: frozenset({"db"}), S11_DB: frozenset({"db"}), REL_S21_DB: frozenset({"db"}),
    PHASE21_DEG: frozenset({"deg", "degree", "degrees", "°"}),
}
_UNIT_TEXT = {S21_DB: "dB", S11_DB: "dB", REL_S21_DB: "dB", PHASE21_DEG: "deg"}
#: ids name check ids, deck stems and element names: plain identifiers, no dots
_PLAIN = re.compile(r"^[A-Za-z0-9_]+$")
#: what every PASS says it is (the design's wording rule for fixture verdicts)
SCOPE_NOTE = (
    "network verdict under the confirmed model values (not a measured part; a schematic-level network: "
    "no track, via or ground-return inductance)"
)
DC_PATH_EFFECT = (
    f"each DC-path resistor ({DC_PATH_OHM:g} ohm) adds {1.0 / DC_PATH_OHM:g} S from its node to ground: a node of impedance |Z| "
    f"moves by at most |Z| x {1.0 / DC_PATH_OHM:g} relative (1e-6 even at 1 Mohm)"
)
CONDITIONS: dict[str, Any] = {
    "values": "nominal (each member's bound value in this state; no tolerance corners, no Monte Carlo)",
    "temperature_c": NGSPICE_DEFAULT_TEMP_C,
    "temperature_source": "ngspice default (the fixture deck has no .temp card)",
    "network": "schematic level: the members and the runner's port elements only - no track, via or ground-return inductance",
}

_PROV = Provenance(kind=ProvenanceKind.DERIVED, tool=TOOL, tool_version=FIXTURE_VERSION, note="RF fixture deck element (ai_eda.tools.spice.rf_fixture)")


# --------------------------------------------------------------------------- pure arithmetic


def s_parameter(quantity: str, *, v_s: complex, v_drive: complex, v_to: complex, r_drive: float, r_to: float | None) -> complex:
    """The complex S-parameter a quantity reads (module docstring): ``S11`` for ``s11_db``, else ``S21`` (``r_to`` ``None`` = a probe)."""
    if v_s == 0:
        raise ValueError("the source voltage is zero: no S-parameter")
    if quantity == S11_DB:
        return 2.0 * v_drive / v_s - 1.0
    ratio = 2.0 * v_to / v_s
    return ratio if r_to is None else ratio * math.sqrt(r_drive / r_to)


def level_db(s: complex) -> float:
    """20 log10 |s|: ``-inf`` for an exact zero, ``ValueError`` for a non-finite magnitude.

    An exact zero is what the double-precision solve gives a perfect match or
    a perfect isolation (below about -300 dB, whether it lands on 0.0 or on
    1e-16 is floating-point rounding): a level of ``-inf`` dB, judged like any
    other (on the passing side of every ``at_most`` bound, outside every
    finite band). A nan / inf magnitude is no level at all.
    """
    mag = abs(s)
    if not math.isfinite(mag):
        raise ValueError(f"|S| = {mag!r}: a non-finite magnitude has no level in dB")
    if mag == 0.0:
        return -math.inf
    return 20.0 * math.log10(mag)


def phase_deg(s: complex) -> float:
    """The phase of ``s`` in degrees, in (-180, 180]."""
    if s == 0 or not (math.isfinite(s.real) and math.isfinite(s.imag)):
        raise ValueError(f"S = {s!r} has no phase")
    p = math.degrees(cmath.phase(s))
    return 180.0 if p <= -180.0 else p


def is_hit(sample_hz: float, f_hz: float) -> bool:
    """``True`` when a sample's frequency is ``f_hz`` itself: equal, or within :data:`HIT_REL` of it (ngspice's parser is not correctly rounded)."""
    return sample_hz == f_hz or abs(sample_hz - f_hz) <= HIT_REL * abs(f_hz)


def interpolate_log_f(x0: float, y0: float, x1: float, y1: float, x: float, *, phase: bool = False) -> Interpolation:
    """``y`` at frequency ``x`` between two samples, linear in log frequency (in the phase unwrapped next to ``y0``).

    The same rule as :meth:`~ai_eda.tools.spice.SpiceResult.interpolate` on an
    ac sweep (a level in dB interpolated in log f is the runner's log-log
    magnitude reading). An exact hit (:func:`is_hit`) returns the sample. A
    level of ``-inf`` dB on either side (an exact zero, :func:`level_db`)
    reads ``-inf`` - never ``nan`` - so the bracket is ``(-inf, the other
    sample]``: it passes an ``at_most`` bound only when the finite neighbour
    does too.
    """
    if is_hit(x0, x) or x0 == x1:
        return Interpolation(value=y0, exact=True, x0=x0, y0=y0, x1=x0, y1=y0, method="exact")
    if is_hit(x1, x):
        return Interpolation(value=y1, exact=True, x0=x1, y0=y1, x1=x1, y1=y1, method="exact")
    if not phase and (y0 == -math.inf or y1 == -math.inf):
        return Interpolation(value=-math.inf, exact=False, x0=x0, y0=y0, x1=x1, y1=y1, method="a sample at -inf dB (|S| = 0): the reading is -inf")
    if phase:
        y1 = y0 + ((y1 - y0 + 180.0) % 360.0 - 180.0)
    if x0 > 0 and x1 > 0 and x > 0:
        t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
        method = "log-x"
    else:
        t = (x - x0) / (x1 - x0)
        method = "linear"
    return Interpolation(value=y0 + (y1 - y0) * t, exact=False, x0=x0, y0=y0, x1=x1, y1=y1, method=method + (" (phase unwrapped)" if phase else ""))


def _bracket(xs: list[float], x: float) -> tuple[int, int] | None:
    """Indices of the sample(s) ``x`` is read from: an exact hit ``(k, k)`` (:func:`is_hit`), else the first bracketing pair, else ``None``."""
    for k, xk in enumerate(xs):
        if is_hit(xk, x):
            return k, k
    for k in range(len(xs) - 1):
        if xs[k] < x < xs[k + 1] or xs[k + 1] < x < xs[k]:
            return k, k + 1
    return None


def _conducting_pins(device: SpiceDevice, order: list[str]) -> list[list[str]]:
    """Groups of pins a device joins for DC (module docstring: junctions conduct through ngspice's gmin)."""
    if device in (SpiceDevice.R, SpiceDevice.L, SpiceDevice.V, SpiceDevice.D, SpiceDevice.Q):
        return [list(order)]
    if device == SpiceDevice.M:  # drain gate source bulk: the gate is insulated
        return [[p for i, p in enumerate(order) if i != 1]]
    if device == SpiceDevice.T:  # port 1 +, port 1 -, port 2 +, port 2 -: each conductor is a DC short
        return [[order[0], order[2]], [order[1], order[3]]] if len(order) == 4 else []
    return []  # C, I, X: no DC path the walk can vouch for


def floating_nets(ir: CircuitIR) -> list[str]:
    """Nets that are nodes of the deck (a pin of a simulated element, a source terminal) with no DC path to the GROUND net."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    of_pin = {(p.component_ref, p.pin_number): n.name for n in ir.nets for p in n.pins}
    nodes: set[str] = set()
    for c in ir.components:
        b = c.spice
        if b is None or b.exclude or b.device is None:
            continue
        for pin in b.pin_order:
            net = of_pin.get((c.ref, pin))
            if net is not None:
                nodes.add(net)
                find(net)
        for group in _conducting_pins(b.device, b.pin_order):
            nets = [of_pin[(c.ref, p)] for p in group if (c.ref, p) in of_pin]
            for a, b_ in zip(nets, nets[1:]):
                union(a, b_)
    if ir.simulation is not None:
        for s in ir.simulation.stimuli:
            nodes.update((s.net, s.reference_net))
            if s.source == "voltage":
                union(s.net, s.reference_net)
    grounds = [n.name for n in ir.nets if n.kind == NetKind.GROUND]
    if not grounds:
        return sorted(nodes)
    g = find(grounds[0])
    return [n for n in sorted(nodes) if n != grounds[0] and find(n) != g]


# --------------------------------------------------------------------------- reading the RF IR (by attribute)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return default if value is None else value


def _num(t: Any) -> float | None:
    if t is None:
        return None
    v = getattr(t, "value", t)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _kind(t: Any) -> ProvenanceKind | None:
    prov = getattr(t, "provenance", None)
    return None if prov is None else prov.kind


def _finite(t: Any, what: str, *, positive: bool = False) -> tuple[float | None, str | None]:
    if t is None:
        return None, f"{what} is missing"
    v = _num(t)
    if v is None or not math.isfinite(v):
        return None, f"{what} must be a finite number, got {getattr(t, 'value', t)!r}"
    if positive and v <= 0.0:
        return None, f"{what} must be > 0, got {v!r}"
    if _kind(t) is ProvenanceKind.LLM_GENERATED:
        return None, f"{what} has llm_generated provenance: nothing an LLM proposed reaches a verdict until it is verified"
    return v, None


def _state_key(state: str | None) -> str:
    return "" if state is None else state


@dataclass
class _Deck:
    """One (network, state, drive) deck: its IR, the nodes the quantities read, and what the runner added."""

    key: str
    stem: str
    state: str | None
    drive: str
    ir: CircuitIR | None = None
    text: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    problem: str | None = None
    #: port name -> (node vector name, reference node vector name); ``None`` is the ground node
    nodes: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    source: tuple[str | None, str | None] = (None, None)
    #: port name -> reference impedance (ohm); ``None`` for a probe
    r_port: dict[str, float | None] = field(default_factory=dict)
    added: list[dict[str, Any]] = field(default_factory=list)
    dc_path: list[dict[str, str]] = field(default_factory=list)
    open_nets: list[str] = field(default_factory=list)
    port_dc_v: dict[str, float] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    path: Path | None = None
    hash: str | None = None
    results: dict[str, SpiceResult] = field(default_factory=dict)
    unverifiable: dict[str, str] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)
    #: the runner's single-point analyses: id (``rf_at_<k>``) -> the frequency its one point is at
    points: dict[str, float] = field(default_factory=dict)

    def analysis_ids(self) -> list[str]:
        """Every analysis the deck runs, in run order: the network's sweeps, then the runner's point analyses."""
        return [] if self.ir is None or self.ir.simulation is None else [a.id for a in self.ir.simulation.analyses]

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key, "stem": self.stem, "state": self.state, "drive": self.drive, "path": None if self.path is None else str(self.path),
            "hash": self.hash, "problem": self.problem, "port_dc_v": self.port_dc_v,
            # what a reader of the rawfile needs to recompute the S-parameters: each port's node and reference node
            # (lower-cased vector names; None = ground), the source's, and each port's reference impedance (None = probe)
            "nodes": {k: list(v) for k, v in self.nodes.items()}, "source": list(self.source), "r_port": self.r_port,
            "added": self.added, "dc_path": {"ohm": DC_PATH_OHM, "resistors": self.dc_path, "effect_bound": DC_PATH_EFFECT},
            "open_nets": self.open_nets, "assumptions": self.assumptions, "excluded": self.report.get("excluded", []),
            "points": dict(self.points),
            "analyses": {
                aid: {"command": r.command, "succeeded": r.succeeded, "n_points": r.n_points, "elapsed_s": r.elapsed_s, "timed_out": r.timed_out,
                      "raw_output_path": r.raw_output_path, "raw_output_hash": r.raw_output_hash, "errors": list(r.errors), "unverifiable": r.unverifiable}
                for aid, r in self.results.items()
            },
        }


class _Names:
    """Unique names for what the runner adds (case-insensitive, like ngspice): ``base``, ``base_1``, ``base_2`` ..."""

    def __init__(self, taken: Iterable[str]) -> None:
        self.taken = {t.lower() for t in taken}

    def new(self, base: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", base)
        name, i = base, 0
        while name.lower() in self.taken:
            i += 1
            name = f"{base}_{i}"
        self.taken.add(name.lower())
        return name


def _pins(n: int) -> list[Pin]:
    return [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=_PROV) for i in range(1, n + 1)]


def _traced(value: float, unit: str | None, note: str, *, kind: ProvenanceKind = ProvenanceKind.DERIVED, derived_from: list[str] | None = None) -> Traced:
    prov = _PROV.model_copy(update={"kind": kind, "note": note, "derived_from": list(derived_from or [])})
    return Traced(value=value, unit=unit, provenance=prov)


def _ground(ir: CircuitIR) -> tuple[Net | None, str | None]:
    grounds = [n for n in ir.nets if n.kind == NetKind.GROUND]
    if len(grounds) != 1:
        return None, f"the IR has {len(grounds)} GROUND nets ({[n.name for n in grounds]}): a fixture needs exactly one reference node"
    return grounds[0], None


def _network_problems(ir: CircuitIR, network: Any) -> list[str]:
    """Structural mistakes of a network that make every deck of it meaningless (module docstring)."""
    problems: list[str] = []
    nid = str(_attr(network, "id", ""))
    members = list(_attr(network, "members", []))
    if not members:
        problems.append(f"network {nid} has no members")
    if len(set(members)) != len(members):
        problems.append(f"network {nid} lists a member twice: {members}")
    for ref in members:
        if ir.component(ref) is None:
            problems.append(f"member {ref!r} is not a component of the IR")
    for label, bindings in [("bindings", _attr(network, "bindings", {}))] + [
        (f"state {_attr(s, 'id', '?')} bindings", _attr(s, "bindings", {})) for s in _attr(network, "states", [])
    ]:
        stray = sorted(set(bindings) - set(members))
        if stray:
            problems.append(f"{label} name {stray}, which are not members of {nid}")
    loss_q = _attr(network, "loss_q", {})
    stray = sorted(set(loss_q) - set(members))
    if stray:
        problems.append(f"loss_q names {stray}, which are not members of {nid}")
    if loss_q:
        _, why = _finite(getattr(network, "q_ref_hz", None), f"network {nid} q_ref_hz (the frequency of the loss_q series resistors)", positive=True)
        if why:
            problems.append(why)
        for ref, q in sorted(loss_q.items()):
            _, why = _finite(q, f"loss_q[{ref}]", positive=True)
            if why:
                problems.append(why)
    ports = list(_attr(network, "ports", []))
    if not ports:
        problems.append(f"network {nid} has no ports")
    names = [str(_attr(p, "name", "")) for p in ports]
    if len(set(names)) != len(names):
        problems.append(f"network {nid} names a port twice: {names}")
    port_nets: dict[str, str] = {}
    for p in ports:
        name, kind = str(_attr(p, "name", "")), _attr(p, "kind", PORT)
        if not name:
            problems.append("a port has no name")
        if kind not in PORT_KINDS:
            problems.append(f"port {name}: kind {kind!r} is not one of {list(PORT_KINDS)}")
        if kind == PORT:
            _, why = _finite(getattr(p, "z0_ohm", None), f"port {name} z0_ohm (an RF port needs a reference impedance)", positive=True)
            if why:
                problems.append(why)
            net = str(_attr(p, "net", ""))
            if net in port_nets:
                problems.append(f"ports {port_nets[net]} and {name} both terminate net {net}")
            port_nets[net] = name
        if kind == RAIL and getattr(p, "voltage_v", None) is not None:
            _, why = _finite(getattr(p, "voltage_v"), f"rail port {name} voltage_v")
            if why:
                problems.append(why)
    if ports and not any(_attr(p, "kind", PORT) == PORT for p in ports):
        problems.append(f"network {nid} has no port of kind {PORT!r} (a probe, rail or control port cannot be driven)")
    states = list(_attr(network, "states", []))
    sids = [str(_attr(s, "id", "")) for s in states]
    if len(set(sids)) != len(sids):
        problems.append(f"network {nid} names a state twice: {sids}")
    for sid in sids:
        if not _PLAIN.match(sid):
            problems.append(f"state id {sid!r} must be a plain identifier (it is part of the check id)")
    by_name = {str(_attr(p, "name", "")): p for p in ports}
    for s in states:
        for pname, level in sorted(_attr(s, "port_dc_v", {}).items()):
            port = by_name.get(pname)
            if port is None:
                problems.append(f"state {_attr(s, 'id')} sets a DC level on {pname!r}, which is not a port of {nid}")
            elif _attr(port, "kind", PORT) == PROBE:
                problems.append(f"state {_attr(s, 'id')} sets a DC level on probe {pname}: a probe has no element to carry it")
            _, why = _finite(level, f"state {_attr(s, 'id')} port_dc_v[{pname}]")
            if why:
                problems.append(why)
    sweep = list(_attr(network, "sweep", []))
    if not sweep:
        problems.append(f"network {nid} has no sweep (ac analyses)")
    for spec in sweep:
        if getattr(spec, "kind", None) != SpiceAnalysis.AC:
            problems.append(f"sweep {getattr(spec, 'id', '?')} is a {getattr(spec, 'kind', '?')} analysis: a fixture is measured by ac sweeps only")
        if str(getattr(spec, "id", "")).lower().startswith(POINT_PREFIX):
            problems.append(f"sweep id {getattr(spec, 'id', '')!r} starts with {POINT_PREFIX!r}, the prefix of the runner's own point analyses")
    return problems


def _resolve_binding(ir: CircuitIR, network: Any, state: Any, ref: str) -> tuple[SpiceBinding | None, str, str | None]:
    """``(binding, source, problem)`` of one member: the state's, else the network's fixture binding, else its own."""
    state_b = _attr(state, "bindings", {}) if state is not None else {}
    if ref in state_b:
        return state_b[ref], "state", None
    net_b = _attr(network, "bindings", {})
    if ref in net_b:
        return net_b[ref], "network", None
    comp = ir.component(ref)
    b = None if comp is None else comp.spice
    if b is None:
        return None, "component", f"member {ref} has no SPICE binding and the network gives no fixture binding"
    if b.exclude:
        return None, "component", (f"member {ref} is excluded from the design netlist ({b.exclude_reason or 'no reason given'}) and the network gives no "
                                   "fixture binding: the deck would silently lose it")
    return b, "component", None


def point_analyses(freqs: Iterable[float]) -> list[AnalysisSpec]:
    """The runner's single-point analyses ``ac lin 1 f f``, one per distinct frequency (ascending), ids ``rf_at_0``, ``rf_at_1`` ...

    Their numbers carry the runner's ``derived`` provenance: whether a row's
    ``at`` is an assumption is the row's own business (:func:`_assumed`), not
    every row's of the deck.
    """
    out: list[AnalysisSpec] = []
    for k, f in enumerate(sorted({float(x) for x in freqs})):
        note = f"the RF fixture runner's single-point analysis at {f!r} Hz (an expectation's at / ref_at)"
        params = {
            "variation": Traced(value="lin", provenance=_PROV.model_copy(update={"note": note})),
            "points": Traced(value=1, provenance=_PROV.model_copy(update={"note": note})),
            "fstart": _traced(f, "Hz", note), "fstop": _traced(f, "Hz", note),
        }
        out.append(AnalysisSpec(id=f"{POINT_PREFIX}{k}", kind=SpiceAnalysis.AC, params=params, provenance=_PROV.model_copy(update={"note": note})))
    return out


def build_deck(ir: CircuitIR, network: Any, state: Any, drive: str, points: Iterable[float] = ()) -> _Deck:
    """The deck IR of ``network`` in ``state`` driven at port ``drive`` (module docstring); ``deck.problem`` says why there is none.

    ``points`` are the frequencies the deck's expectations are judged at: each
    gets one of the runner's single-point analyses (:func:`point_analyses`)
    after the network's own sweeps.
    """
    nid = str(_attr(network, "id", ""))
    sid = None if state is None else str(_attr(state, "id", ""))
    key = nid + ("" if sid is None else f".{sid}")
    ports = list(_attr(network, "ports", []))
    # the key names the drive unless it is the network's first RF port (the port order is free: a bias or probe port may come first)
    first_rf = next((str(_attr(p, "name", "")) for p in ports if _attr(p, "kind", PORT) == PORT), None)
    if drive != first_rf:
        key += f"@{drive}"
    deck = _Deck(key=key, stem=deck_stem(key), state=sid, drive=drive)
    sweeps = list(_attr(network, "sweep", []))
    reserved = [str(getattr(a, "id", "")) for a in sweeps if str(getattr(a, "id", "")).lower().startswith(POINT_PREFIX)]
    if reserved:
        deck.problem = f"sweep id(s) {reserved} start with {POINT_PREFIX!r}, the prefix of the runner's own point analyses"
        return deck
    ground, why = _ground(ir)
    if ground is None:
        deck.problem = why
        return deck
    members = list(_attr(network, "members", []))
    member_set = set(members)
    port_by_name = {str(_attr(p, "name", "")): p for p in ports}
    drive_port = port_by_name.get(drive)
    if drive_port is None or _attr(drive_port, "kind", PORT) != PORT:
        deck.problem = f"the drive {drive!r} is not an RF port (kind {PORT!r}) of {nid}"
        return deck
    port_dc: dict[str, Traced] = dict(_attr(state, "port_dc_v", {})) if state is not None else {}
    deck.port_dc_v = {k: float(_num(v) or 0.0) for k, v in sorted(port_dc.items())}

    components: list[Component] = []
    for ref in members:
        comp = ir.component(ref)
        b, _source, problem = _resolve_binding(ir, network, state, ref)
        if problem is not None or comp is None:
            deck.problem = problem or f"member {ref!r} is not a component of the IR"
            return deck
        components.append(comp.model_copy(update={"spice": b}, deep=True))
    nets: dict[str, Net] = {}
    for n in ir.nets:
        pins = [p for p in n.pins if p.component_ref in member_set]
        if pins or n.kind == NetKind.GROUND:
            nets[n.name] = Net(name=n.name, kind=n.kind, pins=[p.model_copy() for p in pins], provenance=n.provenance)
            if any(p.component_ref not in member_set for p in n.pins) and n.kind != NetKind.GROUND:
                deck.open_nets.append(n.name)
    ports_on = {str(_attr(p, "net", "")) for p in ports}
    deck.open_nets = sorted(n for n in deck.open_nets if n not in ports_on)

    def node(net_name: str) -> str | None:
        return None if net_name == ground.name else net_name.lower()

    for p in ports:
        name, kind, net = str(_attr(p, "name", "")), _attr(p, "kind", PORT), str(_attr(p, "net", ""))
        ref_net = str(_attr(p, "reference_net", ground.name))
        if net == ground.name:
            deck.problem = f"port {name} sits on the ground net {net}: it would measure nothing"
            return deck
        if net not in nets or not nets[net].pins:
            deck.problem = f"port {name}'s net {net!r} is not a net of any member of {nid}: the port would be dangling"
            return deck
        if ref_net not in nets:
            deck.problem = f"port {name}'s reference net {ref_net!r} is neither the ground net nor a net of a member"
            return deck
        if ref_net == net:
            deck.problem = f"port {name}'s net and reference net are both {net!r}"
            return deck
        deck.nodes[name] = (node(net), node(ref_net))
        deck.r_port[name] = _num(getattr(p, "z0_ohm", None)) if kind == PORT else None

    element_names = _Names([c.ref for c in components] + [
        (c.ref if c.spice is None or c.spice.device is None or c.ref[:1].upper() == c.spice.device.value else f"{c.spice.device.value}{c.ref}")
        for c in components
    ])
    net_names = _Names(nets)
    added: list[Component] = []
    stimuli: list[Stimulus] = []

    def add_r(base: str, value: Traced, a: str, b: str, role: str) -> str:
        ref = element_names.new(base if base.upper().startswith("R") else f"R{base}")
        added.append(Component(ref=ref, value=f"{float(value.value):g}", pins=_pins(2), provenance=_PROV,
                               spice=SpiceBinding(device=SpiceDevice.R, value=value, provenance=_PROV)))
        nets[a].pins.append(PinRef(component_ref=ref, pin_number="1"))
        nets[b].pins.append(PinRef(component_ref=ref, pin_number="2"))
        deck.added.append({"ref": ref, "role": role, "ohm": float(value.value), "nets": [a, b]})
        return ref

    def new_net(base: str) -> str:
        name = net_names.new(base)
        nets[name] = Net(name=name, kind=NetKind.SIGNAL, pins=[], provenance=_PROV)
        return name

    def add_v(base: str, a: str, b: str, level: Traced, *, ac: bool, role: str) -> None:
        sid_ = element_names.new(f"V{base}")[1:]  # the element name is "V" + id
        params = {"ac": _traced(1.0, "V", "ac source magnitude of the drive port")} if ac else {}
        stimuli.append(Stimulus(id=sid_, source="voltage", net=a, reference_net=b, kind=StimulusKind.DC, value=level, params=params, provenance=_PROV))
        deck.added.append({"ref": f"V{sid_}", "role": role, "dc_v": float(level.value), "ac_v": 1.0 if ac else 0.0, "nets": [a, b]})

    # (3) loss_q: split each listed inductor with its series R
    q_ref = getattr(network, "q_ref_hz", None)
    for ref, q in sorted(_attr(network, "loss_q", {}).items()):
        comp = next(c for c in components if c.ref == ref)
        b = comp.spice
        if b is None or b.device != SpiceDevice.L or b.value is None or len(b.pin_order) != 2:
            deck.problem = f"loss_q names {ref}, which is not bound as an inductor with a value in this deck ({'state ' + sid if sid else 'no state'})"
            return deck
        l_h = _num(b.value)
        if l_h is None or not math.isfinite(l_h) or l_h <= 0.0:
            deck.problem = f"loss_q names {ref}, whose inductance {b.value.value!r} is not a positive number"
            return deck
        for t, what in ((b.value, f"{ref} inductance"), (q, f"loss_q[{ref}]"), (q_ref, "q_ref_hz")):
            if _kind(t) is ProvenanceKind.ASSUMPTION:
                deck.assumptions.append(f"{what} (series-R loss model)")
        f_q, why_f = _finite(q_ref, "q_ref_hz", positive=True)
        q_v, why_q = _finite(q, f"loss_q[{ref}]", positive=True)
        if f_q is None or q_v is None:
            deck.problem = why_f or why_q
            return deck
        r = 2.0 * math.pi * f_q * l_h / q_v
        pin2 = b.pin_order[1]
        home = next((n for n in nets.values() if any(p.component_ref == ref and p.pin_number == pin2 for p in n.pins)), None)
        if home is None:
            deck.problem = f"{ref}.{pin2} is in no net: the loss resistor has nowhere to go"
            return deck
        mid = new_net(f"{ref}_Q")
        home.pins = [p for p in home.pins if not (p.component_ref == ref and p.pin_number == pin2)]
        nets[mid].pins.append(PinRef(component_ref=ref, pin_number=pin2))
        value = _traced(r, "ohm", f"loss of {ref}: 2 pi q_ref_hz L / Q = 2 pi {f_q:g} Hz x {l_h:g} H / {q_v:g}", derived_from=[f"{ref}.spice.value", f"loss_q[{ref}]", "q_ref_hz"])
        add_r(f"RQ_{ref}", value, mid, home.name, f"loss of {ref} (Q {q_v:g} at {f_q:g} Hz)")

    # (1) + (2): sources, loads, rails and controls
    for p in ports:
        name, kind, net = str(_attr(p, "name", "")), _attr(p, "kind", PORT), str(_attr(p, "net", ""))
        ref_net = str(_attr(p, "reference_net", ground.name))
        level = port_dc.get(name)
        if kind == PORT:
            z0 = getattr(p, "z0_ohm")
            if _kind(z0) is ProvenanceKind.ASSUMPTION:
                deck.assumptions.append(f"port {name} z0_ohm")
            if name == drive:
                src = new_net(f"{name}_SRC")
                add_v(f"S_{name}", src, ref_net, level if level is not None else _traced(0.0, "V", "the drive port's DC level (none set)"),
                      ac=True, role=f"drive source of port {name} (ac 1 V)")
                add_r(f"RS_{name}", z0, src, net, f"source impedance of port {name}")
                deck.source = (node(src), node(ref_net))
            elif level is not None:
                dcn = new_net(f"{name}_DC")
                add_r(f"RL_{name}", z0, net, dcn, f"load of port {name}")
                add_v(f"DC_{name}", dcn, ref_net, level, ac=False, role=f"DC level of port {name}'s load")
            else:
                add_r(f"RL_{name}", z0, net, ref_net, f"load of port {name}")
        elif kind in (RAIL, CONTROL):
            if level is None and kind == RAIL and getattr(p, "voltage_v", None) is not None:
                level = getattr(p, "voltage_v")
            if level is None:
                where = f"state {sid}" if sid else "the network (it has no states)"
                deck.problem = f"{kind} port {name} has no DC level in {where}" + (" and no voltage_v" if kind == RAIL else "")
                return deck
            add_v(f"DC_{name}", net, ref_net, level, ac=False, role=f"ideal DC source at {kind} port {name} (an ac short at that node)")

    extra = point_analyses(points)
    deck.points = {a.id: float(a.params["fstart"].value) for a in extra}
    deck_ir = CircuitIR(project=ProjectMeta(id=f"rf_{deck.stem}", name=f"RF fixture {key}"), components=[*components, *added],
                        nets=list(nets.values()),
                        simulation=SimulationSetup(stimuli=stimuli, analyses=[*(s.model_copy(deep=True) for s in sweeps), *extra]))
    # (4) DC-path resistors
    floating = floating_nets(deck_ir)
    for net_name in floating:
        value = _traced(DC_PATH_OHM, "ohm", "DC-path resistor (no DC path to ground otherwise; see ai_eda.tools.spice.rf_fixture)")
        add_r(f"RDC_{net_name}", value, net_name, ground.name, "DC path to ground")
        deck.dc_path.append({"ref": deck.added[-1]["ref"], "net": net_name})
    deck.ir = deck_ir.model_copy(update={"components": [*components, *added], "nets": list(nets.values())})
    try:
        deck.text = build(deck.ir)
        deck.report = build_report(deck.ir)
    except CompileError as e:
        deck.problem = f"the fixture deck does not compile: {e}"
        return deck
    deck.assumptions += list(deck.report.get("assumptions") or [])
    return deck


# --------------------------------------------------------------------------- measuring


@dataclass
class _Reading:
    value: float | None
    problem: str | None = None
    interpolation: Interpolation | None = None
    analysis_id: str | None = None
    command: str | None = None
    raw: Evidence | None = None
    unverifiable: str | None = None
    not_run: str | None = None
    #: the frequencies of the samples read whose |S| is exactly 0 (a level of -inf dB)
    zero_hz: list[float] = field(default_factory=list)


def _complex(res: SpiceResult, node: str | None, k: int) -> complex:
    if node is None:
        return 0j
    return complex(res.vector(f"{node}.real")[k], res.vector(f"{node}.imag")[k])


def _sample(res: SpiceResult, deck: _Deck, quantity: str, drive: str, to: str | None, k: int) -> complex:
    v_s = _complex(res, deck.source[0], k) - _complex(res, deck.source[1], k)
    dn, dr = deck.nodes[drive]
    v_drive = _complex(res, dn, k) - _complex(res, dr, k)
    v_to = 0j
    r_to: float | None = None
    if quantity != S11_DB and to is not None:
        tn, tr = deck.nodes[to]
        v_to = _complex(res, tn, k) - _complex(res, tr, k)
        r_to = deck.r_port.get(to)
    r_drive = deck.r_port.get(drive) or 0.0
    return s_parameter(quantity, v_s=v_s, v_drive=v_drive, v_to=v_to, r_drive=r_drive, r_to=r_to)


def _raw_evidence(res: SpiceResult, key: str, aid: str) -> Evidence | None:
    if not res.raw_output_path:
        return None
    return Evidence(description=f"ngspice rawfile of RF fixture {key}, analysis {aid} ({res.command})", path=res.raw_output_path, content_hash=res.raw_output_hash)


def read_quantity(deck: _Deck, order: list[str], quantity: str, drive: str, to: str | None, f_hz: float, *, exact_only: bool = False) -> _Reading:
    """``quantity`` at ``f_hz`` from the deck's analyses: the one with the closest bracket (an exact hit first), in dB or degrees.

    ``exact_only`` (every judged row): only a sample at ``f_hz`` itself
    (:func:`is_hit`) is read - the runner's point analysis at that frequency
    - never a value between two sweep points, whose bracket bounds the true
    value only where the response is monotonic between them (a resonance or
    a notch between two points is not). An exact zero |S| reads ``-inf`` dB
    and is listed in ``zero_hz``; a phase of a zero S is no phase (``problem``).
    """
    best: tuple[float, int, str, tuple[int, int]] | None = None
    ranges: list[str] = []
    for i, aid in enumerate(order):
        res = deck.results.get(aid)
        if res is None or not res.succeeded or aid in deck.failed:
            continue
        try:
            xs = res.scale_values()
        except (KeyError, ValueError):
            continue
        if xs:
            ranges.append(f"{aid} {min(xs):.12g}..{max(xs):.12g} Hz")
        br = _bracket(xs, f_hz)
        if br is None or (exact_only and br[0] != br[1]):
            continue
        x0, x1 = xs[br[0]], xs[br[1]]
        cost = 0.0 if br[0] == br[1] else abs(math.log(x1 / x0)) if x0 > 0 and x1 > 0 else abs(x1 - x0)
        if best is None or cost < best[0]:
            best = (cost, i, aid, br)
    if best is None:
        not_run = {aid: why for aid, why in deck.unverifiable.items()}
        if not_run:
            return _Reading(None, unverifiable="; ".join(f"analysis {a} could not be run here: {w}" for a, w in not_run.items()))
        if deck.failed:
            return _Reading(None, not_run="; ".join(f"analysis {a} did not succeed: {' | '.join(e[:3])}" for a, e in deck.failed.items()))
        if exact_only:
            return _Reading(None, f"no analysis has a sample at f = {f_hz:.12g} Hz (a row is read only at its own frequency, never between "
                                  f"sweep points; analyses: {'; '.join(ranges) or 'none ran'})")
        return _Reading(None, f"f = {f_hz:g} Hz is outside every sweep of the network ({'; '.join(ranges) or 'none ran'})")
    _, _, aid, (k0, k1) = best
    res = deck.results[aid]
    xs = res.scale_values()
    raw = _raw_evidence(res, deck.key, aid)
    try:
        s0 = _sample(res, deck, quantity, drive, to, k0)
        s1 = s0 if k1 == k0 else _sample(res, deck, quantity, drive, to, k1)
        if quantity == PHASE21_DEG:
            y0, y1 = phase_deg(s0), phase_deg(s1)
        else:
            y0, y1 = level_db(s0), level_db(s1)
    except KeyError as e:
        return _Reading(None, f"vector not produced: {e}", analysis_id=aid, command=res.command, raw=raw)
    except (ValueError, ZeroDivisionError) as e:
        return _Reading(None, f"{quantity} at {xs[k0]:.12g} Hz: {e}", analysis_id=aid, command=res.command, raw=raw)
    if any(math.isnan(y) or y == math.inf for y in (y0, y1)):
        return _Reading(None, f"{quantity} is not finite near {f_hz:g} Hz", analysis_id=aid, command=res.command, raw=raw)
    zero = sorted({xs[k] for k, y in ((k0, y0), (k1, y1)) if y == -math.inf})
    if k0 == k1:  # an exact hit: the sample itself (its own frequency is recorded as x0)
        interp = Interpolation(value=y0, exact=True, x0=xs[k0], y0=y0, x1=xs[k0], y1=y0, method="exact")
    else:
        interp = interpolate_log_f(xs[k0], y0, xs[k1], y1, f_hz, phase=quantity == PHASE21_DEG)
    return _Reading(interp.value, None, interp, aid, res.command, raw, zero_hz=zero)


def _j(y: float | None) -> float | str | None:
    """A number for ``details`` (JSON): a finite float as it is, an infinity as the text ``"-inf"`` / ``"+inf"`` (JSON has none; pydantic would write null)."""
    if y is None or math.isfinite(y):
        return y
    return "-inf" if y < 0 else "+inf" if y > 0 else None


def _bracket_details(r: _Reading) -> dict[str, Any] | None:
    i = r.interpolation
    if i is None:
        return None
    return {"analysis_id": r.analysis_id, "x0": i.x0, "y0": _j(i.y0), "x1": i.x1, "y1": _j(i.y1), "method": i.method, "exact": i.exact}


def _zero_hint(quantity: str, drive: str | None, to: str | None) -> str:
    """What an exact zero means - and what else could produce one (a mis-wired port reads 0 as well)."""
    if quantity == S11_DB:
        return f"|S11| = 0: port {drive} sees exactly its reference impedance"
    return (f"|S21| = 0: nothing of the drive reaches port {to} - check that {to} is not shorted to its reference (a rail or control "
            f"port's ideal source is an ac short) and that it is connected to the drive")


def _on_branch(value: float, low: float, high: float, nominal: float) -> tuple[float, float, float]:
    """A phase and its bracket moved by the multiple of 360 degrees that puts the value nearest ``nominal``."""
    k = round((nominal - value) / 360.0)
    return value + 360.0 * k, low + 360.0 * k, high + 360.0 * k


# --------------------------------------------------------------------------- the stage


def _exp_problems(network: Any, exp: Any, by_name: dict[str, Any], state_ids: set[str], *, probe: bool = False) -> tuple[list[str], str | None, str | None]:
    """``(problems, drive, to)`` of an expectation or probe (module docstring)."""
    problems: list[str] = []
    ports = list(_attr(network, "ports", []))
    eid = str(_attr(exp, "id", ""))
    if not _PLAIN.match(eid):
        problems.append(f"id {eid!r} must be a plain identifier (it is part of the check id)")
    quantity = _attr(exp, "quantity")
    if quantity not in QUANTITIES:
        problems.append(f"quantity {quantity!r} is not one of {list(QUANTITIES)}")
    state = getattr(exp, "state", None)
    if state_ids and state not in state_ids:
        problems.append(f"state {state!r} is not a state of the network ({sorted(state_ids)})")
    if not state_ids and state is not None:
        problems.append(f"state {state!r} named, but the network has no states")
    # the RF IR requires a drive on every row; a stand-in without one is driven at the network's first RF port
    drive = getattr(exp, "drive", None) or next((str(_attr(p, "name", "")) for p in ports if _attr(p, "kind", PORT) == PORT), None)
    dport = by_name.get(drive or "")
    if dport is None or _attr(dport, "kind", PORT) != PORT:
        problems.append(f"drive {drive!r} is not an RF port (kind {PORT!r}) of the network")
    to = getattr(exp, "to", None)
    if quantity == S11_DB:
        if to is not None and to != drive:
            problems.append(f"s11_db is read at the drive port; 'to' ({to!r}) must be empty or the drive")
        to = None
    elif quantity in QUANTITIES:
        others = [n for n, p in by_name.items() if n != drive and _attr(p, "kind", PORT) in (PORT, PROBE)]
        if to is None and len(others) == 1:
            to = others[0]
        tport = by_name.get(to or "")
        if tport is None or _attr(tport, "kind", PORT) not in (PORT, PROBE):
            problems.append(f"'to' {to!r} is not an RF port or probe of the network")
        elif to == drive:
            problems.append(f"'to' is the drive port {drive}: that is s11_db, not {quantity}")
        elif quantity == S21_DB and _attr(tport, "kind", PORT) == PROBE:
            problems.append(f"an absolute s21_db to probe {to} is not a power-wave ratio (a probe has no reference impedance): use rel_s21_db or phase21_deg")
    _, why = _finite(getattr(exp, "at", None), "at (Hz)", positive=True)
    if why:
        problems.append(why)
    ref_at = getattr(exp, "ref_at", None)
    if quantity == REL_S21_DB:
        _, why = _finite(ref_at, "ref_at (Hz, the reference frequency of a relative level)", positive=True)
        if why:
            problems.append(why)
    elif ref_at is not None:
        problems.append(f"ref_at is only for {REL_S21_DB}")
    if probe:
        return problems, drive, to
    nominal = getattr(exp, "nominal", None)
    _, why = _finite(nominal, "nominal")
    if why:
        problems.append(why)
    elif quantity in _UNITS and getattr(nominal, "unit", None) is not None and str(nominal.unit).strip().lower() not in _UNITS[quantity]:
        problems.append(f"nominal carries unit {nominal.unit!r}; {quantity} is in {_UNIT_TEXT[quantity]}")
    bound = getattr(exp, "bound", None)
    tol_abs, tol_rel = getattr(exp, "tol_abs", None), getattr(exp, "tol_rel", None)
    if tol_rel is not None:
        problems.append("tol_rel is given: a relative tolerance on a level in dB or on an angle is no tolerance - use tol_abs or a bound")
    if bound is not None and bound not in BOUNDS:
        problems.append(f"bound {bound!r} is not one of {list(BOUNDS)}")
    if bound is not None and tol_abs is not None:
        problems.append("both a bound and tol_abs are given: an expectation has exactly one of them")
    if bound is None and tol_abs is None:
        problems.append(f"no tolerance: a level in {_UNIT_TEXT.get(quantity, 'dB')} needs tol_abs or a one-sided bound")
    if tol_abs is not None:
        v, why = _finite(tol_abs, "tol_abs")
        if why:
            problems.append(why)
        elif v is not None and v < 0.0:
            problems.append(f"tol_abs must be >= 0, got {v!r}")
    if bound is not None and getattr(exp, "requirement_id", None) is not None:
        problems.append("a one-sided bound cannot claim a requirement yet (requirement_id is set)")
    if _kind(exp) is ProvenanceKind.LLM_GENERATED:
        problems.append("the expectation has llm_generated provenance: nothing an LLM proposed reaches a verdict until it is verified")
    return problems, drive, to


def _assumed(exp: Any) -> list[str]:
    out = [f"expectation {getattr(exp, 'id', '?')} {label}" for label in ("nominal", "tol_abs", "at", "ref_at")
           if _kind(getattr(exp, label, None)) is ProvenanceKind.ASSUMPTION]
    if _kind(exp) is ProvenanceKind.ASSUMPTION:
        out.append(f"expectation {getattr(exp, 'id', '?')}")
    return out


def _check_id(network_id: str, state: str | None, exp_id: str) -> str:
    return f"{CHECK_PREFIX}.{network_id}" + ("" if state is None else f".{state}") + f".{exp_id}"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_deck(deck: _Deck, network: Any, runner: SpiceRunner, dir_: Path) -> None:
    """Write the deck and run every analysis of it - the network's sweeps, then the runner's point analyses (results, failures and environment limits into ``deck``)."""
    assert deck.text is not None
    dir_.mkdir(parents=True, exist_ok=True)
    deck.path = dir_ / f"{deck.stem}.cir"
    deck.path.write_text(deck.text, encoding="utf-8", newline="\n")
    deck.hash = _sha(deck.path)
    commands: dict[str, str] = dict(deck.report.get("analyses") or {})
    for aid in deck.analysis_ids():
        try:
            res = runner.run(deck.path, SpiceAnalysis.AC, dir_ / deck.stem / aid, command=commands[aid])
        except (ToolUnavailableError, ToolExecutionError) as e:
            deck.unverifiable[aid] = f"the SPICE engine could not run the deck: {e}"
            continue
        deck.results[aid] = res
        if res.netlist_hash != deck.hash:
            deck.failed.setdefault(aid, []).append(f"runner loaded a file with hash {res.netlist_hash}, the deck is {deck.hash}")
        elif not res.succeeded:
            if res.unverifiable:
                deck.unverifiable[aid] = res.unverifiable
            else:
                deck.failed[aid] = list(res.errors) or ["no result"]


def _network_results(ir: CircuitIR, network: Any, runner: Any, workdir: Path, ir_hash: str) -> list[ValidationResult]:
    nid = str(_attr(network, "id", ""))
    block = getattr(network, "block", None)
    summary_id = f"{CHECK_PREFIX}.{nid}"
    ports = list(_attr(network, "ports", []))
    by_name = {str(_attr(p, "name", "")): p for p in ports}
    states = {str(_attr(s, "id", "")): s for s in _attr(network, "states", [])}
    problems = _network_problems(ir, network)
    expectations = list(_attr(network, "expectations", []))
    probes = list(_attr(network, "probes", []))
    engine_ok = isinstance(runner, SpiceRunner) and runner.available()
    tool = runner.engine if engine_ok else None
    engine_version = None
    out: list[ValidationResult] = []

    # group expectations and probes by (state, drive): one deck each
    plans: list[tuple[Any, bool, list[str], str | None, str | None]] = []  # (obj, is_probe, problems, drive, to)
    seen_ids: set[tuple[str, str]] = set()
    for obj, is_probe in [(e, False) for e in expectations] + [(p, True) for p in probes]:
        errs, drive, to = _exp_problems(network, obj, by_name, set(states), probe=is_probe)
        key = (_state_key(getattr(obj, "state", None)), str(_attr(obj, "id", "")))
        if not is_probe:
            if key in seen_ids:
                errs.append(f"expectation id {key[1]!r} is used twice in state {key[0] or '(none)'}")
            seen_ids.add(key)
        plans.append((obj, is_probe, errs, drive, to))
    decks: dict[tuple[str | None, str], _Deck] = {}
    if not problems:
        # one deck per (state, drive); every judged row's at / ref_at gets a point analysis in its deck
        points: dict[tuple[str | None, str], list[float]] = {}
        for obj, is_probe, errs, drive, _to in plans:
            if errs or drive is None:
                continue
            key_ = (getattr(obj, "state", None), drive)
            fs = points.setdefault(key_, [])
            if not is_probe:
                fs += [v for v in (_num(getattr(obj, "at", None)), _num(getattr(obj, "ref_at", None))) if v is not None]
        for (sid, drive), fs in points.items():
            decks[(sid, drive)] = build_deck(ir, network, states.get(sid) if sid is not None else None, drive, points=fs)
        if engine_ok:
            for deck in decks.values():
                if deck.problem is None:
                    _run_deck(deck, network, runner, workdir / RF_DIR)
                    if deck.results:
                        engine_version = engine_version or next((r.engine_version for r in deck.results.values() if r.engine_version), None)
    sweep_ids = [str(s.id) for s in _attr(network, "sweep", [])]
    probe_rows: dict[str, Any] = {}
    exp_results: list[ValidationResult] = []
    for obj, is_probe, errs, drive, to in plans:
        eid = str(_attr(obj, "id", ""))
        sid = getattr(obj, "state", None)
        quantity = _attr(obj, "quantity")
        at, ref_at = _num(getattr(obj, "at", None)), _num(getattr(obj, "ref_at", None))
        deck = decks.get((sid, drive or ""))
        port_dc = dict(deck.port_dc_v) if deck is not None else {k: float(_num(v) or 0.0) for k, v in sorted(_attr(states.get(sid), "port_dc_v", {}).items())}
        base: dict[str, Any] = {
            "kind": "rf fixture expectation" if not is_probe else "rf fixture probe", "network": nid, "block": block, "state": sid,
            "port_dc_v": port_dc, "quantity": quantity, "drive": drive, "to": to, "at": at, "f_hz": at, "ref_at": ref_at,
        }
        if not is_probe:
            nominal = getattr(obj, "nominal", None)
            base.update(nominal=_num(nominal), unit=_UNIT_TEXT.get(quantity), tol_abs=_num(getattr(obj, "tol_abs", None)), tol_rel=None,
                        bound=getattr(obj, "bound", None), requirement_id=getattr(obj, "requirement_id", None), conditions=CONDITIONS, scope=SCOPE_NOTE)
        if deck is not None:
            base.update(deck=deck.stem, deck_key=deck.key)
        evidence: list[Evidence] = []
        if deck is not None and deck.path is not None and deck.hash is not None:
            evidence.append(Evidence(description=f"RF fixture deck {deck.key} (compiled from the members' bindings and the port elements)", path=str(deck.path), content_hash=deck.hash))
        stamp: dict[str, Any] = dict(ir_hash=ir_hash)
        if tool is not None and deck is not None and deck.hash is not None and deck.results:
            stamp.update(tool=tool, tool_version=engine_version, artifact_hash=deck.hash)

        def verdict(status: ValidationStatus, message: str, details: dict[str, Any], ev: list[Evidence]) -> None:
            if is_probe:
                probe_rows[f"{sid}.{eid}" if sid else eid] = {**details, "note": message}
                return
            exp_results.append(ValidationResult(check_id=_check_id(nid, sid, eid), status=status, message=message, evidence=ev, details=details, **stamp))

        label = f"{quantity} {drive}" + ("" if quantity == S11_DB else f"->{to}") + (f" at {at:g} Hz" if at is not None else "")
        if quantity == REL_S21_DB and ref_at is not None:
            label += f" re {ref_at:g} Hz"
        if sid is not None:
            label = f"[{sid}] " + label
        if errs:
            verdict(ValidationStatus.FAIL, f"{nid}: {label}: cannot be judged: " + "; ".join(errs), {**base, "problems": errs, "repair": "human"}, evidence)
            continue
        if problems:
            verdict(ValidationStatus.NOT_VERIFIED, f"{nid}: {label}: not simulated - the network is malformed (see {summary_id})", base, evidence)
            continue
        assert deck is not None and at is not None
        if deck.problem is not None:
            verdict(ValidationStatus.NOT_VERIFIED, f"{nid}: {label}: not simulated - {deck.problem}", base, evidence)
            continue
        if not engine_ok:
            verdict(ValidationStatus.NOT_VERIFIED, f"{nid}: {label}: no SPICE engine available (the fixture deck compiled but was not simulated)", base, evidence)
            continue
        # a judged row is read only at its own frequency (the runner's point analysis); a probe may be read between sweep points
        order = [*deck.points, *sweep_ids]
        reading = read_quantity(deck, order, quantity, drive, to, at, exact_only=not is_probe)
        ref_reading = (read_quantity(deck, order, quantity, drive, to, ref_at, exact_only=not is_probe)
                       if quantity == REL_S21_DB and ref_at is not None else None)
        for r in (reading, ref_reading):
            if r is not None and r.raw is not None and all(e.path != r.raw.path for e in evidence):
                evidence.insert(0, r.raw)
        details = {**base, "analysis_id": reading.analysis_id, "command": reading.command}
        blocked = next((r for r in (reading, ref_reading) if r is not None and (r.unverifiable or r.not_run)), None)
        if blocked is not None:
            if blocked.unverifiable:
                verdict(ValidationStatus.NOT_VERIFIED, f"{nid}: {label}: {blocked.unverifiable}", {**details, "unverifiable": blocked.unverifiable}, evidence)
            else:
                verdict(ValidationStatus.NOT_VERIFIED, f"{nid}: {label}: {blocked.not_run}", {**details, "errors": deck.failed}, evidence)
            continue
        bad = next((r for r in (reading, ref_reading) if r is not None and r.problem is not None), None)
        if bad is not None:
            verdict(ValidationStatus.FAIL, f"{nid}: {label}: {bad.problem}", {**details, "repair": "human"}, evidence)
            continue
        assert reading.value is not None and reading.interpolation is not None
        if ref_reading is not None and ref_reading.zero_hz:
            # S21 at the reference frequency is exactly 0: a level relative to -inf dB is no number
            verdict(ValidationStatus.FAIL, f"{nid}: {label}: the reference level at ref_at is {ZERO_NOTE}: a level relative to it is undefined",
                    {**details, "zero_magnitude": {"ref_at": ref_at, "sample_hz": ref_reading.zero_hz}, "repair": "human"}, evidence)
            continue
        measured = reading.value
        low, high = reading.interpolation.low, reading.interpolation.high
        details["bracket"] = _bracket_details(reading)
        details["sample_hz"] = reading.interpolation.x0 if reading.interpolation.exact else None
        if reading.zero_hz:
            details["zero_magnitude"] = {"at": at, "sample_hz": reading.zero_hz, "note": ZERO_NOTE}
        if ref_reading is not None:
            assert ref_reading.value is not None and ref_reading.interpolation is not None
            measured = reading.value - ref_reading.value  # -inf when the reading at 'at' is an exact zero
            low, high = low - ref_reading.interpolation.high, high - ref_reading.interpolation.low
            details["bracket"] = {"at": _bracket_details(reading), "ref_at": _bracket_details(ref_reading)}
            details["ref_analysis_id"] = ref_reading.analysis_id
            details["levels_db"] = {"at": _j(reading.value), "ref_at": _j(ref_reading.value)}
        finite = math.isfinite(measured)
        unit = f" {_UNIT_TEXT.get(quantity)}"
        shown = f"{measured:.6g}{unit}" if finite else ZERO_NOTE
        if is_probe:
            if quantity == PHASE21_DEG:
                measured = phase_deg(cmath.rect(1.0, math.radians(measured)))
                shown = f"{measured:.6g}{unit}"
            if not reading.interpolation.exact and not reading.zero_hz:
                details["probe_bracket"] = {"low": _j(low), "high": _j(high), "why": "a probe may be read between sweep points (linear in log frequency)"}
            verdict(ValidationStatus.NOT_APPLICABLE, f"{label} = {shown} (probe: recorded, no verdict)", {**details, "measured": measured if finite else None}, evidence)
            continue
        nominal_v = float(_num(getattr(obj, "nominal")))  # type: ignore[arg-type]
        if quantity == PHASE21_DEG:
            measured, low, high = _on_branch(measured, low, high, nominal_v)
            shown = f"{measured:.6g}{unit}"
        details["measured"] = measured if finite else None
        # every row is an exact reading (its own point analysis): judged on the number itself; -inf is on the passing side
        # of every at_most bound and outside every finite band
        status, limit, deviation = judge(measured, obj)
        details["tolerance"] = limit
        details["deviation"] = _j(deviation)
        bound = getattr(obj, "bound", None)
        against = f"{bound} {nominal_v:.6g}{unit}" if bound is not None else f"nominal {nominal_v:.6g}{unit} +/- {'?' if limit is None else f'{limit:.3g}'}{unit}"
        if status is ValidationStatus.UNRESOLVED:
            message = f"{nid}: {label} = {shown}: no usable tolerance or bound ({against}), nothing to judge against"
        else:
            message = f"{nid}: {label} = {shown}, " + ("outside every finite band: " if not finite and bound is None else "") + against
            if bound is None and deviation is not None and math.isfinite(deviation):
                message += f" (deviation {deviation:.3g}{unit})"
            if status is ValidationStatus.FAIL:
                details["repair"] = "human"
            elif status is ValidationStatus.PASS:
                if not finite:
                    message += f"; {_zero_hint(quantity, drive, to)}"
                assumptions = [*deck.assumptions, *_assumed(obj)]
                details["assumptions"] = assumptions
                if assumptions:
                    status = ValidationStatus.NOT_VERIFIED
                    message += f"; not evidence: rests on assumption(s) {assumptions} that nobody confirmed"
                else:
                    message += f"; {SCOPE_NOTE}"
        verdict(status, message, details, evidence)

    # the network's summary
    deck_list = [d.describe() for d in decks.values()]
    failed_decks = [d for d in decks.values() if d.problem is not None or d.failed]
    counts: dict[str, int] = {}
    for r in exp_results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    judged = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none"
    summary_details: dict[str, Any] = {
        "kind": "rf fixture network", "network": nid, "block": block, "members": list(_attr(network, "members", [])),
        "ports": [{"name": str(_attr(p, "name", "")), "kind": _attr(p, "kind", PORT), "net": _attr(p, "net"), "reference_net": _attr(p, "reference_net"),
                   "z0_ohm": _num(getattr(p, "z0_ohm", None))} for p in ports],
        "states": {sid: {k: _num(v) for k, v in sorted(_attr(s, "port_dc_v", {}).items())} for sid, s in states.items()},
        "decks": deck_list, "expectations": {r.check_id: r.status.value for r in exp_results}, "probes": probe_rows,
        "conditions": CONDITIONS, "scope": SCOPE_NOTE,
    }
    evidence = [Evidence(description=f"RF fixture deck {d.key}", path=str(d.path), content_hash=d.hash) for d in decks.values() if d.path is not None and d.hash]
    evidence += [ev for d in decks.values() for aid, r in d.results.items() for ev in [_raw_evidence(r, d.key, aid)] if ev is not None]
    stamp = dict(ir_hash=ir_hash)
    ran = [d for d in decks.values() if d.results]
    if tool is not None and ran:
        stamp.update(tool=tool, tool_version=engine_version, artifact_hash=ran[0].hash if len(decks) == 1 else None)
    malformed = problems + [f"probe {_attr(obj, 'id', '?')}: {e}" for obj, is_probe, errs, _d, _t in plans if is_probe for e in errs]
    if malformed:
        status = ValidationStatus.FAIL
        message = f"{nid}: the fixture network is malformed: " + "; ".join(malformed[:5]) + (f" (+{len(malformed) - 5} more)" if len(malformed) > 5 else "")
        summary_details.update(problems=malformed, repair="human")
    elif failed_decks:
        status = ValidationStatus.FAIL
        reasons = [d.problem or "; ".join(f"sweep {a} failed: {' | '.join(e[:3])}" for a, e in d.failed.items()) for d in failed_decks]
        message = f"{nid}: " + "; ".join(f"deck {d.key}: {why}" for d, why in zip(failed_decks, reasons))
        summary_details["repair"] = "human"
    elif not engine_ok and decks:
        status = ValidationStatus.NOT_VERIFIED
        message = f"{nid}: no SPICE engine available: {len(decks)} fixture deck(s) compiled, none simulated"
    elif decks and all(set(d.unverifiable) == set(d.analysis_ids()) for d in decks.values()):
        status = ValidationStatus.NOT_VERIFIED
        message = f"{nid}: the fixture could not be run here: " + "; ".join(f"{d.key}: {w}" for d in decks.values() for w in d.unverifiable.values())
        summary_details["unverifiable"] = {d.key: d.unverifiable for d in decks.values()}
    else:
        status = worst_status(r.status for r in exp_results)
        message = f"{nid}: {len(decks)} deck(s) run, {len(exp_results)} expectation(s): {judged}"
        if probe_rows:
            message += f", {len(probe_rows)} probe(s) recorded"
        if status is ValidationStatus.FAIL:
            summary_details["repair"] = "human"
            message += f"; failed: {[r.check_id for r in exp_results if r.status is ValidationStatus.FAIL]}"
        elif status is ValidationStatus.PASS:
            message += f"; {SCOPE_NOTE}"
        elif not exp_results:
            message += " (a network without expectations verifies nothing)"
    out.append(ValidationResult(check_id=summary_id, status=status, message=message, evidence=evidence, details=summary_details, **stamp))
    out.extend(exp_results)
    return out


def spice_rf_results(ir: CircuitIR, tools: dict[str, Any], workdir: Path | str, *, design: RFDesign | None = None) -> list[ValidationResult]:
    """Every ``spice.rf.*`` result of the IR's RF fixture networks (module docstring), plus superseding NOT_APPLICABLE ones.

    ``design`` is the RF design to run (default ``ir.rf``); ``[]`` when there
    is no network and nothing recorded to supersede. The engine is
    ``tools["spice"]``; without one every deck is compiled and every
    expectation is NOT_VERIFIED.
    """
    rf = design if design is not None else getattr(ir, "rf", None)
    networks = list(_attr(rf, "networks", [])) if rf is not None else []
    runner = tools.get("spice")
    ir_hash = ir.content_hash()
    out: list[ValidationResult] = []
    ids = [str(_attr(n, "id", "")) for n in networks]
    for network, nid in zip(networks, ids):
        if not _PLAIN.match(nid) or ids.count(nid) > 1:
            why = f"network id {nid!r} must be a plain identifier" if not _PLAIN.match(nid) else f"network id {nid!r} is used {ids.count(nid)} times"
            if not any(r.check_id == f"{CHECK_PREFIX}.{nid}" for r in out):
                out.append(ValidationResult(check_id=f"{CHECK_PREFIX}.{nid}", status=ValidationStatus.FAIL, message=f"{why}: its check ids and deck files would collide",
                                            ir_hash=ir_hash, details={"kind": "rf fixture network", "network": nid, "repair": "human"}))
            continue
        out.extend(_network_results(ir, network, runner, Path(workdir), ir_hash))
    keep = {r.check_id for r in out}
    for check_id, last in ir.validation.latest_by_check().items():
        if check_id.startswith(CHECK_PREFIX + ".") and check_id not in keep and last.status is not ValidationStatus.NOT_APPLICABLE:
            out.append(ValidationResult(check_id=check_id, status=ValidationStatus.NOT_APPLICABLE, ir_hash=ir_hash, details={"superseded": last.status.value},
                                        message="the RF fixture network or expectation is no longer in ir.rf (superseded by this run)"))
    return out


def rf_note(results: list[ValidationResult]) -> str | None:
    """The SPICE stage's one-line note about the ``spice.rf`` results (``None`` when there are none)."""
    summaries = [r for r in results if r.details.get("kind") == "rf fixture network"]
    expectations = [r for r in results if r.details.get("kind") == "rf fixture expectation"]
    retired = [r for r in results if r.status is ValidationStatus.NOT_APPLICABLE and "superseded" in r.details]
    if not summaries and not retired:
        return None
    counts: dict[str, int] = {}
    for r in expectations:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    text = f"spice.rf: {len(summaries)} fixture network(s) " + "(" + ", ".join(f"{r.check_id.removeprefix(CHECK_PREFIX + '.')} {r.status.value}" for r in summaries) + ")"
    text += f", {len(expectations)} expectation(s): " + (", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none")
    if retired:
        text += f"; {len(retired)} earlier result(s) superseded"
    return text


__all__ = [
    "CHECK_PREFIX",
    "DC_PATH_OHM",
    "FIXTURE_VERSION",
    "QUANTITIES",
    "RF_DIR",
    "SCOPE_NOTE",
    "build_deck",
    "floating_nets",
    "interpolate_log_f",
    "level_db",
    "phase_deg",
    "read_quantity",
    "rf_note",
    "s_parameter",
    "spice_rf_results",
]
