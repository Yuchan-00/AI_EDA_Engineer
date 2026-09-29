"""``spice.rf``: the RF fixture runner (:mod:`ai_eda.tools.spice.rf_fixture`).

Without ngspice: the S-parameter arithmetic, the log-frequency reading and
its bracket, the DC-path graph walk, the deck built from a network's members
and the runner's port elements (compiled by the SPICE netlist compiler,
byte-identical twice), every refusal (a member the deck would lose, a
dangling port, a probe as the drive, an absolute S21 to a probe, a relative
tolerance on a dB level, an llm_generated nominal ...), the retirement of
superseded results and the SimulationAgent wiring. A fake ac engine that
solves the compiled R / L / C / V deck by complex nodal analysis
(:func:`mna_solve`, independent of the runner) exercises the whole
measurement path: the 7th-order 0.1 dB Chebyshev low-pass against
``calc.rf.lpf.chebyshev.attenuation``, S11 of a lossless network against
1 - |S21|^2, every row read at its own frequency by the runner's point
analysis (never between two sweep points; a probe still may be), an exact
zero |S| judged as -inf dB (PASS at_most, FAIL at_least / tol_abs, a zero
reference or phase refused), rel / phase rows, probes, the lambda/4 T/R
switch in two states driven from different ports, ports in any order (a
bias port or a probe first), the runner's own refusals of what the RF IR
refuses, an assumption turning a PASS into NOT_VERIFIED, and an engine
failure.

With ngspice (skipped otherwise), through the real engine: the lossless
Chebyshev low-pass within 0.05 dB of the calculator (the critic's rule:
lossless deck only - with Q-40 inductors the passband is about 1 dB off the
lossless formula, -1.045 dB measured on ngspice-42), the Q_e 20 double-tuned
top-C tank at 223.78 MHz at Q_u 40 (-4.2147 dB, rel -28.832 / -17.792 dB at
-/+ f_T, within 0.5 dB of Cohn's dissipation loss 4.343 dB and within
0.05 dB of an independent nodal solve), the lambda/4 T/R switch states, a
2-pole crystal ladder of X-card crystals (its middle node has no DC path
and gets the 1e12 ohm resistor) and the PM tank of ``kr447/decided/pm_n12.cir``
in its three bias states (-21.462 / +1.393 / +18.233 deg at 1.44 / 2.00 /
2.56 V on ngspice-42, the exact deck; chord slope 0.6186 rad/V), a series
LC resonance between two points of an ac dec 5 grid (FAIL at f0, never the
old bracket's false PASS) and the 2.5 / 4 / 6 / 10 dB pi pads of
``calc.rf.attenuator.pi.*`` whose |S11| lands on exactly 0 (PASS at_most).

The RF IR types come from ``ai_eda.ir.rf`` when that module exists (the
wave-1 RF IR part); before it is merged the tests build stand-ins with the
contract's attribute names (``types.SimpleNamespace``) - the runner reads the
networks by attribute. The one-sided ``bound`` rows, the design-deck
retirement skip and ``ir.rf`` itself are that part's and skip without it.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from ai_eda.agents.base import AgentContext
from ai_eda.compilers.spice import netlist_elements
from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Component,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SpiceBinding,
    SpiceDevice,
    Traced,
    ValidationResult,
    ValidationStatus as S,
    assumption,
    user_requirement,
)
from ai_eda.tools.calc import rf
from ai_eda.tools.calc.si import format_spice_number, parse_spice_number
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice import rf_fixture as fx
from ai_eda.tools.spice import stage

try:
    from ai_eda.ir import rf as rf_ir
except ImportError:  # the RF IR part is not merged into this tree
    rf_ir = None

engine = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not engine.available(), reason="ngspice shared library not found")
needs_bound = pytest.mark.skipif(not hasattr(stage, "RF_CHECK_PREFIX"), reason="the one-sided bound branch of stage.judge is the RF IR part's (not in this tree)")
needs_rf_field = pytest.mark.skipif("rf" not in CircuitIR.model_fields, reason="CircuitIR.rf is the RF IR part's (not in this tree)")

PROV = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test fixture")
F_EDGE = 480e6  # the Chebyshev ripple-band edge
F_C = 447.5625e6
F_T = 37296875.0

_DEFAULTS: dict[str, dict[str, Any]] = {
    "RFPort": dict(reference_net="GND", kind="port", z0_ohm=None, frequency_hz=None, voltage_v=None, direction="bidir"),
    "RFState": dict(port_dc_v={}, bindings={}),
    "RFExpectation": dict(state=None, drive=None, to=None, ref_at=None, tol_abs=None, tol_rel=None, bound=None, requirement_id=None),
    "RFProbe": dict(state=None, drive=None, to=None, ref_at=None),
    "RFNetwork": dict(block=None, bindings={}, loss_q={}, q_ref_hz=None, states=[], expectations=[], probes=[]),
    "RFDesign": dict(blocks=[], networks=[], frequency_plan=[], lab_items=[], rails=[], model_values=[], profile_keys=[]),
}


def mk(type_name: str, /, **kw: Any) -> Any:
    """An RF IR object: the real type when the RF IR module exists, else a stand-in with the contract's names."""
    if rf_ir is not None:
        cls = getattr(rf_ir, type_name)
        if "provenance" in cls.model_fields and "provenance" not in kw:
            kw["provenance"] = PROV
        if type_name == "RFDesign" and "blocks" not in kw:  # the real design refuses a network naming a block it does not hold
            names = dict.fromkeys(n.block for n in kw.get("networks", []) if n.block is not None)
            kw["blocks"] = [rf_ir.RFBlock(id=b) for b in names]
        return cls(**kw)
    return SimpleNamespace(**{**_DEFAULTS[type_name], **kw})


def maybe(factory: Callable[[], Any]) -> Any:
    """``factory()``, or ``None`` when the RF IR type refuses the object on construction (stricter than the runner, which refuses it too)."""
    try:
        return factory()
    except ValueError:
        return None


def u(value: Any, unit: str | None = None) -> Traced:
    return user_requirement(value, unit)


def port(name: str, net: str, kind: str = "port", z0: float | None = 50.0, **kw: Any) -> Any:
    return mk("RFPort", name=name, net=net, kind=kind, z0_ohm=None if z0 is None or kind != "port" else u(z0, "ohm"), **kw)


def exp(eid: str, quantity: str, at: float, nominal: float | None = None, tol: float | None = 0.05, **kw: Any) -> Any:
    unit = "deg" if quantity == "phase21_deg" else "dB"
    return mk("RFExpectation", id=eid, quantity=quantity, at=u(at, "Hz"), nominal=None if nominal is None else u(nominal, unit),
              tol_abs=None if tol is None else u(tol, unit), **kw)


def lin(aid: str, f: float, half: float = 1e6, points: int = 3) -> AnalysisSpec:
    """An ``ac lin`` sweep whose middle point is exactly ``f``."""
    return AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(points), "fstart": u(f - half, "Hz"), "fstop": u(f + half, "Hz")}, provenance=PROV)


def bind(device: SpiceDevice, value: float | None = None, unit: str | None = None, **kw: Any) -> SpiceBinding:
    return SpiceBinding(device=device, value=None if value is None else u(value, unit), provenance=PROV, **kw)


class Board:
    """A small design IR: two-pin parts between named nets; the net ``GND`` is the ground."""

    def __init__(self, name: str = "rf_test") -> None:
        self.name = name
        self.parts: list[Component] = []
        self.conn: dict[str, list[tuple[str, str]]] = {"GND": []}

    def add(self, ref: str, binding: SpiceBinding | None, *nets: str | None) -> "Board":
        pins = [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=PROV) for i in range(1, len(nets) + 1)]
        self.parts.append(Component(ref=ref, value=ref, pins=pins, provenance=PROV, spice=binding))
        for i, net in enumerate(nets, 1):
            if net is not None:
                self.conn.setdefault(net, []).append((ref, str(i)))
        return self

    def r(self, ref: str, ohm: float, a: str, b: str) -> "Board":
        return self.add(ref, bind(SpiceDevice.R, ohm, "ohm"), a, b)

    def l(self, ref: str, h: float, a: str, b: str) -> "Board":  # noqa: E743 - an inductor
        return self.add(ref, bind(SpiceDevice.L, h, "H"), a, b)

    def c(self, ref: str, f: float, a: str, b: str) -> "Board":
        return self.add(ref, bind(SpiceDevice.C, f, "F"), a, b)

    def ir(self) -> CircuitIR:
        nets = [Net(name=n, kind=NetKind.GROUND if n == "GND" else NetKind.SIGNAL, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pp], provenance=PROV)
                for n, pp in self.conn.items()]
        return CircuitIR(project=ProjectMeta(id=self.name, name=self.name), components=self.parts, nets=nets)


# --------------------------------------------------------------------------- an independent ac solver (the fake engine)


def mna_solve(text: str, f: float) -> dict[str, complex]:
    """Node voltages of an R / L / C / V deck in the compiler's layout at ``f`` (complex nodal analysis; the V sources' ``AC`` magnitude)."""
    elements = netlist_elements(text)
    nodes = sorted({n for _, ns, _ in elements for n in ns} - {"0"})
    idx = {n: i for i, n in enumerate(nodes)}
    sources = [(name, ns, rest) for name, ns, rest in elements if name[0].upper() == "V"]
    size = len(nodes) + len(sources)
    a = [[0j] * size for _ in range(size)]
    z = [0j] * size
    w = 2.0 * math.pi * f

    def stamp(p: str, m: str, y: complex) -> None:
        for n1, n2, sign in ((p, p, 1), (m, m, 1), (p, m, -1), (m, p, -1)):
            if n1 != "0" and n2 != "0":
                a[idx[n1]][idx[n2]] += sign * y

    for name, ns, rest in elements:
        letter = name[0].upper()
        if letter in "RLC":
            x = parse_spice_number(rest.split()[0])
            stamp(ns[0], ns[1], 1.0 / x if letter == "R" else 1.0 / (1j * w * x) if letter == "L" else 1j * w * x)
        elif letter != "V":
            raise NotImplementedError(f"the fake engine solves R / L / C / V decks only, not {name}")
    for k, (name, ns, rest) in enumerate(sources):
        toks = [t.lower() for t in rest.split()]
        row = len(nodes) + k
        for node, sign in ((ns[0], 1), (ns[1], -1)):
            if node != "0":
                a[idx[node]][row] += sign
                a[row][idx[node]] += sign
        z[row] = parse_spice_number(toks[toks.index("ac") + 1]) if "ac" in toks else 0.0
    m = [row[:] + [z[i]] for i, row in enumerate(a)]
    for c in range(size):
        p = max(range(c, size), key=lambda r_: abs(m[r_][c]))
        m[c], m[p] = m[p], m[c]
        for r_ in range(c + 1, size):
            factor = m[r_][c] / m[c][c]
            for j in range(c, size + 1):
                m[r_][j] -= factor * m[c][j]
    x = [0j] * size
    for c in reversed(range(size)):
        x[c] = (m[c][size] - sum(m[c][j] * x[j] for j in range(c + 1, size))) / m[c][c]
    return {n: x[idx[n]] for n in nodes}


class FakeAC(SpiceRunner):
    """An ac 'engine' that solves the deck file it is given with :func:`mna_solve` (no ngspice); ``mode`` simulates failures."""

    engine = "fake-ac"

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.runs: list[str] = []

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "fake-1"

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None) -> SpiceResult:
        path = Path(netlist_path)
        text = path.read_text(encoding="utf-8")
        self.runs.append(command or "")
        base = dict(engine=self.engine, engine_version="fake-1", netlist_path=str(path), netlist_hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                    analysis=analysis, command=command or "")
        if self.mode == "fail":
            return SpiceResult(**base, errors=["singular matrix (fake)"], succeeded=False)
        if self.mode == "unverifiable":
            return SpiceResult(**base, errors=["no rawfile (fake)"], unverifiable="the rawfile could not be written (fake)", succeeded=False)
        toks = (command or "").split()
        variation, n, f1, f2 = toks[1], int(toks[2]), parse_spice_number(toks[3]), parse_spice_number(toks[4])
        if variation == "lin":
            freqs = [f1 + k * (f2 - f1) / (n - 1) for k in range(n)] if n > 1 else [f1]
        else:
            freqs, k = [], 0
            while f1 * 10 ** (k / n) <= f2 * (1 + 1e-12):
                freqs.append(f1 * 10 ** (k / n))
                k += 1
        vectors: dict[str, list[float]] = {"frequency": freqs}
        for f in freqs:
            for node, v in mna_solve(text, f).items():
                key = node.lower()
                vectors.setdefault(key, []).append(abs(v))
                vectors.setdefault(key + ".real", []).append(v.real)
                vectors.setdefault(key + ".imag", []).append(v.imag)
                vectors.setdefault(key + ".phase_deg", []).append(math.degrees(cmath.phase(v)))
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        raw = workdir / f"{path.stem}.ac.raw"
        raw.write_text(json.dumps(vectors, sort_keys=True), encoding="utf-8")
        return SpiceResult(**base, vectors=vectors, scale="frequency", n_points=len(freqs), raw_output_path=str(raw),
                           raw_output_hash="sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest(), succeeded=True)


def run(ir: CircuitIR, networks: list[Any], tmp_path: Path, runner: Any = None, *, validate: bool = True) -> dict[str, ValidationResult]:
    """Every ``spice.rf.*`` result of ``networks`` by check id; ``validate=False`` hands the runner a design the RF IR never validated."""
    if validate or rf_ir is None:
        design = mk("RFDesign", networks=networks)
    else:
        blocks = [rf_ir.RFBlock(id=b) for b in dict.fromkeys(n.block for n in networks if n.block is not None)]
        design = rf_ir.RFDesign.model_construct(blocks=blocks, networks=networks, frequency_plan=[], lab_items=[], rails=[], model_values=[], profile_keys=[])
    results = fx.spice_rf_results(ir, {} if runner is None else {"spice": runner}, tmp_path, design=design)
    return {r.check_id: r for r in results}


# --------------------------------------------------------------------------- the networks


def chebyshev_lpf(board: Board, *, with_neighbours: bool = True) -> list[str]:
    """The 7th-order 0.1 dB Chebyshev low-pass (ripple edge 480 MHz, 50 ohm) between nets IN and OUT; its refs."""
    g = rf.chebyshev_g_values(7, 0.1)
    nodes, node, refs = ["IN", "M1", "M2", "OUT"], 0, []
    for k in range(1, 8):
        ref = f"{'C' if k % 2 else 'L'}{k}"
        if k % 2:
            board.c(ref, rf.lpf_shunt_c_farads(g[k - 1], F_EDGE, 50.0), nodes[node], "GND")
        else:
            board.l(ref, rf.lpf_series_l_henries(g[k - 1], F_EDGE, 50.0), nodes[node], nodes[node + 1])
            node += 1
        refs.append(ref)
    if with_neighbours:  # parts of the design that are not members: a connector on IN, a resistor on M1
        board.add("J1", SpiceBinding(exclude=True, exclude_reason="connector", provenance=PROV), "IN", "GND")
        board.r("R98", 1000.0, "M1", "GND")
    return refs


def lpf_attenuation(f: float) -> float:
    return rf.chebyshev_attenuation_db(7, 0.1, f, F_EDGE)


def lpf_network(refs: list[str], **kw: Any) -> Any:
    freqs = {"f300": 300e6, "fc": F_C, "edge": F_EDGE, "h2": 2 * F_C, "h3": 3 * F_C}
    expectations = [exp(f"s21_{k}", "s21_db", f, -lpf_attenuation(f), drive="P1", to="P2") for k, f in freqs.items()]
    a_fc = lpf_attenuation(F_C)
    expectations.append(exp("s11_fc", "s11_db", F_C, 10 * math.log10(1 - 10 ** (-a_fc / 10)), drive="P1", to="P1"))
    base = dict(id="lpf", block="trx", members=refs, ports=[port("P1", "IN"), port("P2", "OUT")],
                sweep=[lin(k, f) for k, f in freqs.items()], expectations=expectations)
    return mk("RFNetwork", **{**base, **kw})


#: the Q_e 20 double-tuned top-C tank at 6 f_T (kr447/decided/tanks_n12.cir network "d": 50 ohm ends, C_res 30 pF, Q_u 40)
TANK = dict(c_tap=4.883797e-12, l=1.686055e-08, c_shunt=2.413122e-11, c_k=1.5e-12)
F_TANK = 6 * F_T


def tank_board(board: Board) -> list[str]:
    t = TANK
    board.c("C1", t["c_tap"], "A", "R1").l("L1", t["l"], "R1", "GND").c("C2", t["c_shunt"], "R1", "GND").c("C3", t["c_k"], "R1", "R2")
    board.l("L2", t["l"], "R2", "GND").c("C4", t["c_shunt"], "R2", "GND").c("C5", t["c_tap"], "R2", "B")
    return ["C1", "L1", "C2", "C3", "L2", "C4", "C5"]


def tank_s21(f: float, q_u: float = 40.0) -> complex:
    """S21 of the tank by an independent nodal solve (the loss R = w0 L / Q_u in series with each L)."""
    t, w = TANK, 2 * math.pi * f
    zl = 2 * math.pi * F_TANK * t["l"] / q_u + 1j * w * t["l"]
    y = [[1 / 50 + 1j * w * t["c_tap"], -1j * w * t["c_tap"], 0, 0],
         [-1j * w * t["c_tap"], 1j * w * (t["c_tap"] + t["c_shunt"] + t["c_k"]) + 1 / zl, -1j * w * t["c_k"], 0],
         [0, -1j * w * t["c_k"], 1j * w * (t["c_k"] + t["c_shunt"] + t["c_tap"]) + 1 / zl, -1j * w * t["c_tap"]],
         [0, 0, -1j * w * t["c_tap"], 1 / 50 + 1j * w * t["c_tap"]]]
    i = [1 / 50, 0, 0, 0]  # 1 V behind 50 ohm, as a Norton source
    m = [row[:] + [i[k]] for k, row in enumerate(y)]
    for c in range(4):
        for r_ in range(c + 1, 4):
            factor = m[r_][c] / m[c][c]
            for j in range(c, 5):
                m[r_][j] -= factor * m[c][j]
    x = [0j] * 4
    for c in reversed(range(4)):
        x[c] = (m[c][4] - sum(m[c][j] * x[j] for j in range(c + 1, 4))) / m[c][c]
    return 2 * x[3]


def top_c_calc(name: str, *args: Any) -> float:
    """``calc.rf.resonator.top_c.<name>`` (or ``calc.rf.bpf.<name>``) of the radio calculators, by position."""
    from ai_eda.tools.calc.recompute import CALCULATORS

    tool = f"calc.rf.bpf.{name}" if name == "dissipation_loss" else f"calc.rf.resonator.top_c.{name}"
    return float(CALCULATORS[tool][0](*args).value)


def tank_calc_args() -> dict[str, Traced]:
    """The input set of the Q_e 20 tank (``calc.rf.resonator.top_c.s21_db``): 50 ohm ends, C_res 30 pF, Q_u 40 as a series R at f0."""
    bw = u(top_c_calc("bw_for_qe", u(2), u(F_TANK, "Hz"), u(20.0)), "Hz")
    return dict(n=u(2), f0=u(F_TANK, "Hz"), bw=bw, l=u(1.0 / ((2 * math.pi * F_TANK) ** 2 * 30e-12), "H"), r_source=u(50.0, "ohm"),
                r_load=u(50.0, "ohm"), q_u=u(40.0))


def tank_network(refs: list[str], **kw: Any) -> Any:
    s0 = 20 * math.log10(abs(tank_s21(F_TANK)))
    rel = {k: 20 * math.log10(abs(tank_s21(f))) - s0 for k, f in (("minus", F_TANK - F_T), ("plus", F_TANK + F_T))}
    base = dict(id="tank2", block="tx_chain", members=refs, ports=[port("IN", "A"), port("OUT", "B")], loss_q={"L1": u(40.0), "L2": u(40.0)},
                q_ref_hz=u(F_TANK, "Hz"), sweep=[lin("wide", F_TANK, F_T)],
                expectations=[exp("s21_w", "s21_db", F_TANK, s0, drive="IN", to="OUT"),
                              exp("rel_m", "rel_s21_db", F_TANK - F_T, rel["minus"], ref_at=u(F_TANK, "Hz"), drive="IN", to="OUT"),
                              exp("rel_p", "rel_s21_db", F_TANK + F_T, rel["plus"], ref_at=u(F_TANK, "Hz"), drive="IN", to="OUT")])
    return mk("RFNetwork", **{**base, **kw})


def test_the_tank_fixture_is_the_top_c_calculators_network():
    """The Q_e 20 tank's element values and responses are the radio calculators' (``calc.rf.resonator.top_c.*``)."""
    a = tank_calc_args()
    n, f0, bw, l_, rs, rl = (a[k] for k in ("n", "f0", "bw", "l", "r_source", "r_load"))
    assert top_c_calc("c_tap", n, f0, bw, l_, rs) == pytest.approx(TANK["c_tap"], rel=1e-6)
    assert top_c_calc("c_couple", n, u(1), f0, bw, l_) == pytest.approx(TANK["c_k"], rel=1e-6)
    assert top_c_calc("c_shunt", n, u(1), f0, bw, l_, rs, rl) == pytest.approx(TANK["c_shunt"], rel=1e-6)
    assert l_.value == pytest.approx(TANK["l"], rel=1e-6)
    assert top_c_calc("s21_db", *a.values(), u(F_TANK, "Hz")) == pytest.approx(20 * math.log10(abs(tank_s21(F_TANK))), abs=1e-4)
    for f in (F_TANK - F_T, F_TANK + F_T):
        want = 20 * math.log10(abs(tank_s21(f))) - 20 * math.log10(abs(tank_s21(F_TANK)))
        assert top_c_calc("rel_s21_db", *a.values(), u(f, "Hz"), u(F_TANK, "Hz")) == pytest.approx(want, abs=1e-4)


#: the lumped lambda/4 T/R switch at f_c (7.112 pF / 17.780 nH / 7.112 pF) with PIN models R_on 1 ohm / C_off 0.3 pF
L_Q4 = 50.0 / (2 * math.pi * F_C)
C_Q4 = 1.0 / (2 * math.pi * F_C * 50.0)


def switch_board(board: Board) -> list[str]:
    diode = SpiceBinding(device=SpiceDevice.D, model_name="DPIN", provenance=PROV)  # the design's binding; the fixture states replace it
    board.add("D1", diode, "TX", "COM").add("D2", diode, "RXE", "GND")
    board.c("C1", C_Q4, "COM", "GND").l("L1", L_Q4, "COM", "RXE").c("C2", C_Q4, "RXE", "GND")
    return ["D1", "D2", "C1", "L1", "C2"]


def switch_s21(state: str, drive: str, to: str, f: float = F_C) -> complex:
    """S21 of the switch by nodal analysis (TX, COM, RXE ports of 50 ohm; the PIN diodes as R_on or C_off)."""
    w = 2 * math.pi * f
    yd = 1.0 if state == "tx" else 1j * w * 0.3e-12  # 1 ohm on, 0.3 pF off
    nodes = ["TX", "COM", "RXE"]
    y = [[0j] * 3 for _ in range(3)]

    def add(a: str | None, b: str | None, v: complex) -> None:
        ia = None if a is None else nodes.index(a)
        ib = None if b is None else nodes.index(b)
        if ia is not None:
            y[ia][ia] += v
        if ib is not None:
            y[ib][ib] += v
        if ia is not None and ib is not None:
            y[ia][ib] -= v
            y[ib][ia] -= v

    add("TX", "COM", yd)
    add("RXE", None, yd)
    add("COM", None, 1j * w * C_Q4)
    add("RXE", None, 1j * w * C_Q4)
    add("COM", "RXE", 1 / (1j * w * L_Q4))
    for n in nodes:
        add(n, None, 1 / 50)
    node_of = {"TX": "TX", "COM": "COM", "RX": "RXE"}  # port -> net
    i = [0j] * 3
    i[nodes.index(node_of[drive])] = 1 / 50
    m = [row[:] + [i[k]] for k, row in enumerate(y)]
    for c in range(3):
        for r_ in range(c + 1, 3):
            factor = m[r_][c] / m[c][c]
            for j in range(c, 4):
                m[r_][j] -= factor * m[c][j]
    x = [0j] * 3
    for c in reversed(range(3)):
        x[c] = (m[c][3] - sum(m[c][j] * x[j] for j in range(c + 1, 3))) / m[c][c]
    return 2 * x[nodes.index(node_of[to])]


def switch_network(refs: list[str]) -> Any:
    states = [mk("RFState", id="tx", bindings={"D1": bind(SpiceDevice.R, 1.0, "ohm"), "D2": bind(SpiceDevice.R, 1.0, "ohm")}),
              mk("RFState", id="rx", bindings={"D1": bind(SpiceDevice.C, 0.3e-12, "F"), "D2": bind(SpiceDevice.C, 0.3e-12, "F")})]

    def row(eid: str, state: str, drive: str, to: str) -> Any:
        return exp(eid, "s21_db", F_C, 20 * math.log10(abs(switch_s21(state, drive, to))), state=state, drive=drive, to=to)

    return mk("RFNetwork", id="trsw", block="trx", members=refs, ports=[port("TX", "TX"), port("COM", "COM"), port("RX", "RXE")], states=states,
              sweep=[lin("fc", F_C)], expectations=[row("tx_com", "tx", "TX", "COM"), row("tx_rx", "tx", "TX", "RX"),
                                                    row("com_rx", "rx", "COM", "RX"), row("com_tx", "rx", "COM", "TX")])


# --------------------------------------------------------------------------- pure parts


def test_s_parameters_are_power_waves_between_real_references():
    # a matched through: S21 = 1, S11 = 0
    assert fx.s_parameter("s21_db", v_s=1, v_drive=0.5, v_to=0.5, r_drive=50, r_to=50) == pytest.approx(1)
    assert fx.s_parameter("s11_db", v_s=1, v_drive=0.5, v_to=0.5, r_drive=50, r_to=50) == pytest.approx(0)
    # a lossless 50 -> 12.5 ohm match delivers 0.25 V into 12.5 ohm (rf_gap/lmatch.cir): |S21| = 1 with the impedance factor
    assert abs(fx.s_parameter("s21_db", v_s=1, v_drive=0.5, v_to=0.25, r_drive=50, r_to=12.5)) == pytest.approx(1)
    # an open end reflects everything (S11 = +1), a short (S11 = -1)
    assert fx.s_parameter("s11_db", v_s=1, v_drive=1, v_to=0, r_drive=50, r_to=50) == pytest.approx(1)
    assert fx.s_parameter("s11_db", v_s=1, v_drive=0, v_to=0, r_drive=50, r_to=50) == pytest.approx(-1)
    # to a probe the ratio is the voltage ratio 2 V_to / V_s (no reference impedance)
    assert fx.s_parameter("phase21_deg", v_s=1, v_drive=0.5, v_to=0.25j, r_drive=50, r_to=None) == pytest.approx(0.5j)
    with pytest.raises(ValueError, match="source voltage is zero"):
        fx.s_parameter("s21_db", v_s=0, v_drive=0, v_to=0, r_drive=50, r_to=50)
    assert fx.level_db(0.5) == pytest.approx(-6.0206, abs=1e-4)
    # an exact zero is a level of -inf dB (a perfect match / isolation in the double-precision solve); nan / inf are no level
    assert fx.level_db(0) == -math.inf and fx.level_db(0j) == -math.inf
    for bad in (complex(math.nan, 0.0), complex(math.inf, 0.0)):
        with pytest.raises(ValueError, match="non-finite magnitude has no level in dB"):
            fx.level_db(bad)
    with pytest.raises(ValueError, match="has no phase"):
        fx.phase_deg(0j)
    assert fx.phase_deg(-1) == 180.0 and fx.phase_deg(1j) == pytest.approx(90.0) and fx.phase_deg(complex(-1, -1e-300)) == 180.0


def test_readings_between_sweep_points_are_linear_in_log_frequency_with_the_phase_unwrapped():
    exact = fx.interpolate_log_f(1e6, -3.0, 1e7, -23.0, 1e6)
    assert exact.exact and exact.value == -3.0
    mid = fx.interpolate_log_f(1e6, -3.0, 1e7, -23.0, math.sqrt(1e13))  # the geometric middle
    assert not mid.exact and mid.value == pytest.approx(-13.0) and mid.method == "log-x" and (mid.low, mid.high) == (-23.0, -3.0)
    wrap = fx.interpolate_log_f(1e6, 170.0, 1e7, -170.0, math.sqrt(1e13), phase=True)  # -170 is 190 next to 170
    assert wrap.value == pytest.approx(180.0) and wrap.y1 == pytest.approx(190.0) and "unwrapped" in wrap.method
    # a sample within HIT_REL of f is f itself (ngspice's parser reads '21.40375meg' an ULP or two off)
    near = 21403750.0 * (1 + 2.2e-16)
    assert fx.is_hit(near, 21403750.0) and not fx.is_hit(21403750.0 * (1 + 1e-9), 21403750.0)
    assert fx.interpolate_log_f(near, -3.0, 3e7, -9.0, 21403750.0).exact
    assert fx._bracket([1e7, near, 3e7], 21403750.0) == (1, 1)


def test_an_exact_zero_reads_minus_infinity_and_its_bracket_is_never_nan():
    """A sample at -inf dB (|S| = 0) next to a finite one: the reading is -inf, the bracket (-inf, finite] - never nan."""
    one = fx.interpolate_log_f(1e6, -math.inf, 2e6, -300.0, 1.5e6)
    assert one.value == -math.inf and not one.exact and (one.low, one.high) == (-math.inf, -300.0)
    other = fx.interpolate_log_f(1e6, -25.0, 2e6, -math.inf, 1.5e6)
    assert other.value == -math.inf and (other.low, other.high) == (-math.inf, -25.0)
    both = fx.interpolate_log_f(1e6, -math.inf, 2e6, -math.inf, 1.5e6)
    assert both.value == -math.inf and (both.low, both.high) == (-math.inf, -math.inf)
    assert fx.interpolate_log_f(1e6, -math.inf, 2e6, -300.0, 1e6).exact  # an exact hit on the zero sample
    # the bracket judged: at_most -20 passes (the finite neighbour is below it too), a bound between the two straddles
    at_most = exp("m", "s11_db", 1.5e6, -20.0, tol=None, bound="at_most", drive="P1", to="P1")
    at_least = exp("l", "s11_db", 1.5e6, -20.0, tol=None, bound="at_least", drive="P1", to="P1")
    band = exp("b", "s11_db", 1.5e6, -20.0, tol=3.0, drive="P1", to="P1")
    assert stage.judge(one.value, at_most, one)[0] is S.PASS and stage.judge(other.value, at_most, other)[0] is S.PASS
    tight = exp("t", "s11_db", 1.5e6, -30.0, tol=None, bound="at_most", drive="P1", to="P1")
    assert stage.judge(other.value, tight, other)[0] is S.UNRESOLVED  # (-inf, -25] straddles at_most -30: not a PASS on the zero side alone
    assert stage.judge(other.value, at_least, other)[0] is S.FAIL  # the whole bracket (-inf, -25] is below -20 dB
    assert stage.judge(-math.inf, at_most)[0] is S.PASS and stage.judge(-math.inf, at_least)[0] is S.FAIL
    assert stage.judge(-math.inf, band)[0] is S.FAIL  # outside every finite band
    assert fx._j(-math.inf) == "-inf" and fx._j(math.inf) == "+inf" and fx._j(-3.0) == -3.0 and fx._j(None) is None


def test_the_dc_path_walk_finds_nodes_only_capacitors_or_subcircuits_reach():
    card = u(".subckt XT 1 2\nC1 1 2 1p\n.ends")
    b = Board()
    b.r("R1", 50.0, "IN", "GND").c("C1", 1e-12, "IN", "A").c("C2", 1e-12, "A", "GND")  # A: capacitors only
    b.add("X1", bind(SpiceDevice.X, model_name="XT", model_card=card), "A", "B").c("C3", 1e-12, "B", "GND")  # B: a subcircuit and a capacitor
    b.l("L1", 1e-9, "IN", "C").add("D1", bind(SpiceDevice.D, model_name="D1N"), "C", "E")  # E: through a diode (gmin) - a path
    b.add("M1", bind(SpiceDevice.M, model_name="NM", pin_order=["1", "2", "3", "4"]), "E", "G", "GND", "GND").c("C4", 1e-12, "G", "GND")  # G: a MOS gate
    b.add("T1", bind(SpiceDevice.T, pin_order=["1", "2", "3", "4"], params={"z0": u(50.0, "ohm"), "td": u(1e-9, "s")}), "IN", "GND", "F", "GND")  # F: a line
    assert fx.floating_nets(b.ir()) == ["A", "B", "G"]


def test_stems_are_unique_per_exact_key_and_the_check_ids_carry_the_state():
    assert fx.deck_stem("a.b") != fx.deck_stem("a_b") != fx.deck_stem("A.b")
    assert fx._check_id("trsw", "tx", "tx_com") == "spice.rf.trsw.tx.tx_com" and fx._check_id("lpf", None, "s21_fc") == "spice.rf.lpf.s21_fc"
    assert fx.CHECK_PREFIX == "spice.rf"


def test_the_deck_holds_the_members_and_the_runners_port_elements_only(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    net = lpf_network(refs)
    deck = fx.build_deck(ir, net, None, "P1")
    assert deck.problem is None
    elements = {name: (nodes, rest) for name, nodes, rest in netlist_elements(deck.text)}
    assert "J1" not in elements and "R98" not in elements  # not members
    assert elements["VS_P1"] == (["P1_SRC", "0"], "DC 0 AC 1") and elements["RS_P1"] == (["P1_SRC", "IN"], "50") and elements["RL_P2"] == (["OUT", "0"], "50")
    assert deck.open_nets == ["M1"]  # M1 also reaches R98, which is not a member: open in the fixture (IN is a port: J1 there is the port)
    assert deck.dc_path == [] and deck.source == ("p1_src", None) and deck.nodes == {"P1": ("in", None), "P2": ("out", None)}
    assert [a["role"] for a in deck.added] == ["drive source of port P1 (ac 1 V)", "source impedance of port P1", "load of port P2"]
    again = fx.build_deck(ir, net, None, "P1")
    assert again.text == deck.text and again.stem == deck.stem  # deterministic


def test_loss_q_splits_each_inductor_with_its_series_resistor():
    b = Board()
    refs = tank_board(b)
    deck = fx.build_deck(b.ir(), tank_network(refs), None, "IN")
    assert deck.problem is None
    elements = {name: (nodes, rest) for name, nodes, rest in netlist_elements(deck.text)}
    r_q = 2 * math.pi * F_TANK * TANK["l"] / 40.0
    assert elements["L1"][0] == ["R1", "L1_Q"] and elements["RQ_L1"][0] == ["L1_Q", "0"]
    assert parse_spice_number(elements["RQ_L1"][1]) == pytest.approx(r_q, rel=1e-12) and r_q == pytest.approx(0.5926731, rel=1e-6)  # tanks_n12.cir
    assert {a["ref"] for a in deck.added} >= {"RQ_L1", "RQ_L2"}


def test_a_node_without_a_dc_path_gets_the_1e12_resistor_named_with_its_bound():
    b = Board()
    b.c("C1", 1e-12, "A", "MID").c("C2", 1e-12, "MID", "GND").r("R1", 100.0, "A", "GND")
    net = mk("RFNetwork", id="dc", members=["C1", "C2", "R1"], ports=[port("P1", "A"), port("PR", "MID", kind="probe")], sweep=[lin("s", 1e6, 1e3)],
             expectations=[exp("ph", "phase21_deg", 1e6, 0.0, tol=5.0, drive="P1", to="PR")])
    deck = fx.build_deck(b.ir(), net, None, "P1")
    assert deck.dc_path == [{"ref": "RDC_MID", "net": "MID"}]
    rdc = next((ns, r_) for n, ns, r_ in netlist_elements(deck.text) if n == "RDC_MID")
    assert rdc[0] == ["MID", "0"] and parse_spice_number(rdc[1]) == 1e12
    assert "1e-12 S" in deck.describe()["dc_path"]["effect_bound"] and deck.describe()["dc_path"]["ohm"] == 1e12


# --------------------------------------------------------------------------- refusals


def test_a_member_the_deck_would_lose_or_a_malformed_network_is_refused(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    # an excluded member without a fixture binding: the deck would silently lose it
    got = run(ir, [lpf_network([*refs, "J1"])], tmp_path, FakeAC())
    assert got["spice.rf.lpf"].status is S.FAIL and "J1 is excluded from the design netlist (connector)" in got["spice.rf.lpf"].message
    assert got["spice.rf.lpf.s21_fc"].status is S.NOT_VERIFIED and "not simulated" in got["spice.rf.lpf.s21_fc"].message
    # ... unless the network binds it for the fixture
    got = run(ir, [lpf_network([*refs, "J1"], bindings={"J1": bind(SpiceDevice.R, 1e6, "ohm")})], tmp_path, FakeAC())
    assert got["spice.rf.lpf"].status is S.PASS
    # an unknown member, a dangling port, an RF port without an impedance, a non-ac sweep
    for kw, text in (
        (lambda: dict(members=[*refs, "Q9"]), "member 'Q9' is not a component of the IR"),
        (lambda: dict(ports=[port("P1", "IN"), port("P2", "NOWHERE")]), "not a net of any member"),
        (lambda: dict(ports=[port("P1", "IN", z0=None), port("P2", "OUT")]), "z0_ohm (an RF port needs a reference impedance) is missing"),
        (lambda: dict(sweep=[AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=PROV)]), "a fixture is measured by ac sweeps only"),
    ):
        net = maybe(lambda: lpf_network(refs, **kw()))
        if net is None:
            continue  # the RF IR type refuses it itself, before the runner sees it
        got = run(ir, [net], tmp_path, FakeAC())
        assert got["spice.rf.lpf"].status is S.FAIL and text in got["spice.rf.lpf"].message, (text, got["spice.rf.lpf"].message)
        assert got["spice.rf.lpf"].details["repair"] == "human"


def test_an_expectation_that_cannot_be_judged_is_fail_for_a_human_with_the_reason(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    ports = [port("P1", "IN"), port("P2", "OUT"), port("PR", "M1", kind="probe")]
    factories: list[tuple[Callable[[], Any], str]] = [
        (lambda: exp("to_probe", "s21_db", F_C, -1.0, drive="P1", to="PR"), "an absolute s21_db to probe PR is not a power-wave ratio"),
        (lambda: exp("probe_drive", "s21_db", F_C, -1.0, drive="PR", to="P2"), "drive 'PR' is not an RF port"),
        (lambda: exp("rel_tol", "s21_db", F_C, -1.0, drive="P1", to="P2", tol_rel=u(0.1)), "a relative tolerance on a level in dB or on an angle is no tolerance"),
        (lambda: exp("no_tol", "s21_db", F_C, -1.0, tol=None, drive="P1", to="P2"), "no tolerance: a level in dB needs tol_abs or a one-sided bound"),
        (lambda: mk("RFExpectation", id="llm", quantity="s21_db", at=u(F_C, "Hz"), nominal=Traced(value=-1.0, unit="dB", provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED)),
            tol_abs=u(0.1, "dB"), drive="P1", to="P2"), "nominal has llm_generated provenance"),
        (lambda: exp("no_ref", "rel_s21_db", F_C, -1.0, drive="P1", to="P2"), "ref_at (Hz, the reference frequency of a relative level) is missing"),
        (lambda: exp("self", "s21_db", F_C, -1.0, drive="P1", to="P1"), "that is s11_db"),
        (lambda: mk("RFExpectation", id="unit", quantity="phase21_deg", at=u(F_C, "Hz"), nominal=u(10.0, "dB"), tol_abs=u(1.0, "deg"), drive="P1", to="P2"),
         "nominal carries unit 'dB'; phase21_deg is in deg"),
    ]
    rows = [(row, text) for row, text in ((maybe(f), text) for f, text in factories) if row is not None  # the RF IR types may refuse some
            and maybe(lambda: lpf_network(refs, ports=ports, expectations=[row])) is not None]  # themselves, a row or its network
    assert rows
    got = run(ir, [lpf_network(refs, ports=ports, expectations=[r for r, _ in rows])], tmp_path, FakeAC())
    for row, text in rows:
        result = got[f"spice.rf.lpf.{row.id}"]
        assert result.status is S.FAIL and text in result.message and result.details["repair"] == "human", (row.id, result.message)
    assert got["spice.rf.lpf"].status is S.FAIL


def _unvalidated_network(fields: dict[str, Any], match: str) -> Any:
    """The network the RF IR refuses (``ValueError`` matching ``match``), built without validation - the runner's own backstop."""
    if rf_ir is None:
        return mk("RFNetwork", **fields)
    with pytest.raises(ValueError, match=match):
        rf_ir.RFNetwork(**fields)
    return rf_ir.RFNetwork.model_construct(**fields)


def test_a_control_or_rail_port_needs_a_level_in_the_state(tmp_path: Path):
    b = Board()
    b.r("R1", 50.0, "A", "B").r("R2", 1000.0, "B", "VB")
    fields = dict(id="bias", members=["R1", "R2"], ports=[port("P1", "A"), port("P2", "B"), port("VB", "VB", kind="control")],
                  states=[mk("RFState", id="on", port_dc_v={"VB": u(2.0, "V")}), mk("RFState", id="off")], sweep=[lin("s", 1e6, 1e3)],
                  expectations=[exp("t_on", "s21_db", 1e6, 0.0, tol=100.0, state="on", drive="P1", to="P2"),
                                exp("t_off", "s21_db", 1e6, 0.0, tol=100.0, state="off", drive="P1", to="P2")])
    net = _unvalidated_network(fields, r"control port VB needs a DC level in every state .*\['off'\]")  # the RF IR refuses it first
    got = run(b.ir(), [net], tmp_path, FakeAC(), validate=False)
    assert got["spice.rf.bias.on.t_on"].status is S.PASS and got["spice.rf.bias.on.t_on"].details["port_dc_v"] == {"VB": 2.0}
    assert got["spice.rf.bias.off.t_off"].status is S.NOT_VERIFIED and "control port VB has no DC level in state off" in got["spice.rf.bias.off.t_off"].message
    assert got["spice.rf.bias"].status is S.FAIL and "deck bias.off" in got["spice.rf.bias"].message


def test_the_runner_still_refuses_what_the_rf_ir_refuses(tmp_path: Path):
    """A DC level on a probe, a control port in a network without states, a sweep in the runner's point namespace: built without
    validation (``model_construct``), the runner FAILs the summary itself."""
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    base = dict(id="lpf", block="trx", members=refs, sweep=[lin("fc", F_C)], expectations=[exp("s21", "s21_db", F_C, -lpf_attenuation(F_C), drive="P1", to="P2")])
    cases = [
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("PR", "M1", kind="probe")], states=[mk("RFState", id="on", port_dc_v={"PR": u(1.0, "V")})],
              expectations=[exp("s21", "s21_db", F_C, -lpf_attenuation(F_C), drive="P1", to="P2", state="on")]),
         r"port_dc_v on the probe", "sets a DC level on probe PR: a probe has no element to carry it"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("VB", "M1", kind="control")]), "network has no states", "control port VB has no DC level in the network"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT")], sweep=[lin("rf_at_0", F_C)]), "prefix of the fixture runner's own point analyses",
         "starts with 'rf_at_', the prefix of the runner's own point analyses"),
    ]
    for over, ir_match, text in cases:
        net = _unvalidated_network({**base, **over}, ir_match)
        got = run(ir, [net], tmp_path, FakeAC(), validate=False)
        assert got["spice.rf.lpf"].status is S.FAIL and text in got["spice.rf.lpf"].message, (text, got["spice.rf.lpf"].message)


def test_the_port_order_is_free_every_row_names_its_drive(tmp_path: Path):
    """A bias (control) port or a probe listed first is simulated like the same network with it listed last: every row names its drive."""
    z_b = 1.0 / (1.0 / 1000.0 + 1.0 / (1e-3 + 50.0))  # B: R2 (VB is an ac short) || R3 + the 50 ohm load of P2
    s21 = 20 * math.log10(2 * z_b / (100.0 + z_b) * 50.0 / (50.0 + 1e-3))
    rows = [exp("t", "s21_db", 1e6, s21, tol=1e-6, state="on", drive="P1", to="P2"),
            exp("ph", "phase21_deg", 1e6, 0.0, tol=1e-6, state="on", drive="P1", to="PR")]
    measured = {}
    for order in ("control_first", "probe_first", "last"):
        vb, pr, p1, p2 = port("VB", "VB", kind="control"), port("PR", "B", kind="probe"), port("P1", "A"), port("P2", "B2")
        ports = {"control_first": [vb, pr, p1, p2], "probe_first": [pr, p1, vb, p2], "last": [p1, p2, pr, vb]}[order]
        b2 = Board()
        b2.r("R1", 50.0, "A", "B").r("R2", 1000.0, "B", "VB").r("R3", 1e-3, "B", "B2")  # P2 on B2, the probe on B: one net each
        net = mk("RFNetwork", id="n", members=["R1", "R2", "R3"], ports=ports, states=[mk("RFState", id="on", port_dc_v={"VB": u(2.0, "V")})],
                 sweep=[lin("s", 1e6, 1e3)], expectations=rows)
        got = run(b2.ir(), [net], tmp_path / order, FakeAC())
        assert got["spice.rf.n"].status is S.PASS, (order, got["spice.rf.n"].message)
        assert [d["key"] for d in got["spice.rf.n"].details["decks"]] == ["n.on"]  # the first RF port is the default key, whatever comes before it
        measured[order] = got["spice.rf.n.on.t"].details["measured"]
    assert measured["control_first"] == pytest.approx(measured["last"], abs=1e-9) == pytest.approx(measured["probe_first"], abs=1e-9)
    assert measured["last"] == pytest.approx(s21, abs=1e-3)


def test_no_engine_compiles_every_deck_and_verifies_nothing(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    got = run(b.ir(), [lpf_network(refs)], tmp_path)
    assert got["spice.rf.lpf"].status is S.NOT_VERIFIED and "no SPICE engine available" in got["spice.rf.lpf"].message
    assert all(r.status is S.NOT_VERIFIED for k, r in got.items() if k != "spice.rf.lpf")
    assert not (tmp_path / fx.RF_DIR).exists() and got["spice.rf.lpf"].tool is None
    assert got["spice.rf.lpf"].details["decks"][0]["problem"] is None  # the deck compiled


# --------------------------------------------------------------------------- the whole path on the fake engine


def test_the_lossless_chebyshev_lowpass_reads_as_the_calculator_on_an_exact_engine(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    runner = FakeAC()
    got = run(ir, [lpf_network(refs, probes=[mk("RFProbe", id="ph_fc", quantity="phase21_deg", at=u(F_C, "Hz"), drive="P1", to="P2")])], tmp_path, runner)
    summary = got["spice.rf.lpf"]
    assert summary.status is S.PASS and fx.SCOPE_NOTE in summary.message, summary.message
    for k, f in {"f300": 300e6, "fc": F_C, "edge": F_EDGE, "h2": 2 * F_C, "h3": 3 * F_C}.items():
        r = got[f"spice.rf.lpf.s21_{k}"]
        assert r.status is S.PASS and r.details["measured"] == pytest.approx(-lpf_attenuation(f), abs=1e-6), (k, r.message)
        assert r.details["f_hz"] == f and r.details["state"] is None and r.details["port_dc_v"] == {} and r.details["bracket"]["exact"]
    assert round(-got["spice.rf.lpf.s21_h2"].details["measured"], 2) == 52.75  # the design's lossless 2 f_c number
    assert got["spice.rf.lpf.s11_fc"].status is S.PASS  # |S11|^2 = 1 - |S21|^2: the network is lossless
    deck = summary.details["decks"][0]
    r = got["spice.rf.lpf.s21_fc"]
    assert r.tool == "fake-ac" and r.tool_version == "fake-1" and r.artifact_hash == deck["hash"] and r.ir_hash == ir.content_hash()
    paths = {e.path for e in r.evidence}
    # every row is read from the runner's own point analysis at its frequency (after the network's five sweeps), whose rawfile is its evidence
    assert r.details["analysis_id"].startswith(fx.POINT_PREFIX) and deck["points"][r.details["analysis_id"]] == F_C and r.details["sample_hz"] == F_C
    assert deck["path"] in paths and deck["analyses"][r.details["analysis_id"]]["raw_output_path"] in paths
    assert sorted(deck["points"].values()) == sorted({300e6, F_C, F_EDGE, 2 * F_C, 3 * F_C})  # s21_fc and s11_fc share one point
    assert Path(deck["path"]).parent == tmp_path / fx.RF_DIR and Path(deck["path"]).stem == fx.deck_stem("lpf")
    assert len(runner.runs) == 10 and runner.runs[1] == "ac lin 3 446.5625meg 448.5625meg" and runner.runs[6] == "ac lin 1 447.5625meg 447.5625meg"
    probe = summary.details["probes"]["ph_fc"]
    assert "spice.rf.lpf.ph_fc" not in got and -180 < probe["measured"] <= 180 and "no verdict" in probe["note"]


def test_a_row_between_sweep_points_is_read_at_its_own_frequency_never_interpolated(tmp_path: Path):
    """Every row gets the runner's point analysis ``ac lin 1 f f``: a coarse grid's two-sample bracket is never judged (it bounds the
    true value only where the response is monotonic between the samples); a row away from every sweep is read all the same."""
    b = Board()
    refs = chebyshev_lpf(b)
    coarse = AnalysisSpec(id="coarse", kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(3), "fstart": u(300e6, "Hz"), "fstop": u(600e6, "Hz")}, provenance=PROV)
    net = lpf_network(refs, sweep=[coarse], expectations=[
        exp("straddle", "s21_db", 470e6, -8.0, tol=0.5, drive="P1", to="P2"),  # 450 .. 600 MHz spans -0.1 .. -19.8 dB; 470 MHz is in the ripple band
        exp("outside", "s21_db", 470e6, 10.0, tol=0.5, drive="P1", to="P2"),
        exp("exact", "s21_db", 470e6, -lpf_attenuation(470e6), tol=1e-6, drive="P1", to="P2"),
        exp("far", "s21_db", 5e9, -1.0, drive="P1", to="P2")],  # no sweep reaches 5 GHz: its point analysis does
        probes=[mk("RFProbe", id="between", quantity="s21_db", at=u(470e6, "Hz"), drive="P1", to="P2"),
                mk("RFProbe", id="nowhere", quantity="s21_db", at=u(5e9, "Hz"), drive="P1", to="P2")])
    got = run(b.ir(), [net], tmp_path, FakeAC())
    for k in ("straddle", "outside", "exact", "far"):
        r = got[f"spice.rf.lpf.{k}"]
        assert r.details["bracket"]["exact"] and r.details["analysis_id"].startswith(fx.POINT_PREFIX), (k, r.details)
    assert got["spice.rf.lpf.straddle"].status is S.FAIL and "refine the sweep" not in got["spice.rf.lpf.straddle"].message
    assert got["spice.rf.lpf.straddle"].details["measured"] == pytest.approx(-lpf_attenuation(470e6), abs=1e-9)
    assert got["spice.rf.lpf.outside"].status is S.FAIL and got["spice.rf.lpf.exact"].status is S.PASS
    assert got["spice.rf.lpf.far"].status is S.FAIL and got["spice.rf.lpf.far"].details["measured"] == pytest.approx(-lpf_attenuation(5e9), abs=1e-6)
    # a probe (no verdict) is not given a point analysis: it is read between sweep points with its bracket recorded, or not at all
    probes = got["spice.rf.lpf"].details["probes"]
    assert probes["between"]["analysis_id"] == fx.POINT_PREFIX + "0" and probes["between"]["bracket"]["exact"]  # it shares the rows' point at 470 MHz
    assert "outside every sweep" not in probes["nowhere"]["note"] and probes["nowhere"]["bracket"]["exact"]  # ... and the one at 5 GHz
    only_probe = lpf_network(refs, sweep=[coarse], expectations=[], probes=[mk("RFProbe", id="between", quantity="s21_db", at=u(470e6, "Hz"), drive="P1", to="P2"),
                                                                         mk("RFProbe", id="nowhere", quantity="s21_db", at=u(5e9, "Hz"), drive="P1", to="P2")])
    probes = run(b.ir(), [only_probe], tmp_path, FakeAC())["spice.rf.lpf"].details["probes"]
    assert not probes["between"]["bracket"]["exact"] and probes["between"]["probe_bracket"]["low"] < probes["between"]["probe_bracket"]["high"]
    assert "outside every sweep" in probes["nowhere"]["note"]


@needs_ngspice
def test_ngspice_a_resonance_between_sweep_points_is_no_false_pass(tmp_path: Path):
    """A series L 1 uH / C 2.533 pF between two 50 ohm ports (f0 = 100 MHz), a row 'at_most -6 dB' at f0: on an ac dec 5 grid from
    70 MHz the two neighbouring samples read -13.4 / -7.7 dB (the old bracket judge said PASS); at f0 itself S21 is 0 dB - FAIL."""
    f0, l_ = 100e6, 1e-6
    c = 1.0 / ((2 * math.pi * f0) ** 2 * l_)
    b = Board()
    b.l("L1", l_, "IN", "M").c("C1", c, "M", "OUT")
    dec5 = AnalysisSpec(id="dec5", kind=SpiceAnalysis.AC, params={"variation": u("dec"), "points": u(5), "fstart": u(70e6, "Hz"), "fstop": u(1e9, "Hz")},
                        provenance=PROV)
    net = mk("RFNetwork", id="ser", members=["L1", "C1"], ports=[port("P1", "IN"), port("P2", "OUT")], sweep=[dec5],
             expectations=[exp("rej", "s21_db", f0, -6.0, tol=None, bound="at_most", drive="P1", to="P2")])
    got = run(b.ir(), [net], tmp_path, engine)
    rej = got["spice.rf.ser.rej"]
    assert rej.status is S.FAIL and abs(rej.details["measured"]) < 1e-6 and rej.details["analysis_id"] == fx.POINT_PREFIX + "0", rej.message
    assert rej.details["sample_hz"] == pytest.approx(f0, rel=1e-12) and got["spice.rf.ser"].status is S.FAIL


def test_relative_and_phase_rows_of_an_asymmetric_tank(tmp_path: Path):
    b = Board()
    refs = tank_board(b)
    net = tank_network(refs)
    phase = math.degrees(cmath.phase(tank_s21(F_TANK)))
    net_rows = list(net.expectations) + [exp("ph", "phase21_deg", F_TANK, phase + 360.0, tol=0.01, drive="IN", to="OUT")]  # another branch of the same angle
    net = tank_network(refs, expectations=net_rows)
    got = run(b.ir(), [net], tmp_path, FakeAC())
    assert got["spice.rf.tank2"].status is S.PASS, got["spice.rf.tank2"].message
    assert got["spice.rf.tank2.s21_w"].details["measured"] == pytest.approx(-4.2147, abs=1e-3)
    assert got["spice.rf.tank2.rel_m"].details["measured"] == pytest.approx(-28.832, abs=1e-3)  # kr447/decided tanks_n12.cir network d
    assert got["spice.rf.tank2.rel_p"].details["measured"] == pytest.approx(-17.792, abs=1e-3)  # the + side is 11 dB weaker: asymmetric
    assert set(got["spice.rf.tank2.rel_p"].details["levels_db"]) == {"at", "ref_at"}
    assert got["spice.rf.tank2.ph"].details["measured"] == pytest.approx(phase + 360.0, abs=1e-9)


def test_a_phase_between_sweep_points_is_read_exactly_on_the_nominals_branch(tmp_path: Path):
    b = Board()
    refs = tank_board(b)
    f = F_TANK + 12345.0  # between two points of a 20 kHz grid: read from its own point analysis
    want = math.degrees(cmath.phase(tank_s21(f)))
    rows = [exp("ph", "phase21_deg", f, want - 360.0, tol=1.0, drive="IN", to="OUT"), exp("ph_tight", "phase21_deg", f, want, tol=1e-4, drive="IN", to="OUT")]
    got = run(b.ir(), [tank_network(refs, sweep=[lin("fine", F_TANK, 1e6, 101)], expectations=rows)], tmp_path, FakeAC())
    ph = got["spice.rf.tank2.ph"]
    assert ph.status is S.PASS and ph.details["measured"] == pytest.approx(want - 360.0, abs=1e-9), ph.message
    assert ph.details["bracket"]["exact"] and "bias_bracket" not in ph.details
    assert got["spice.rf.tank2.ph_tight"].status is S.PASS  # exact: a 1e-4 degree band is judged on the number itself


def test_bound_rows_under_the_contract_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The runner's side of a one-sided row, under a judge that follows the RF IR part's contract (the real one is tested below)."""

    def contract_judge(measured, exp_, interpolation=None, bias=None):
        if getattr(exp_, "bound", None) is None:
            return stage.judge(measured, exp_, interpolation, bias)
        nominal = float(exp_.nominal.value)
        lo, hi = (bias if bias is not None else (interpolation.low, interpolation.high) if interpolation is not None and not interpolation.exact else (measured, measured))
        lo, hi = min(lo, measured), max(hi, measured)
        ok = (lambda v: v >= nominal) if exp_.bound == "at_least" else (lambda v: v <= nominal)
        status = S.PASS if ok(lo) and ok(hi) else S.FAIL if not ok(lo) and not ok(hi) else S.UNRESOLVED
        return status, None, measured - nominal

    monkeypatch.setattr(fx, "judge", contract_judge)
    b = Board()
    refs = chebyshev_lpf(b)
    coarse = AnalysisSpec(id="coarse", kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(3), "fstart": u(300e6, "Hz"), "fstop": u(600e6, "Hz")}, provenance=PROV)
    rows = [exp("fc_min", "s21_db", F_C, -1.0, tol=None, bound="at_least", drive="P1", to="P2"),
            exp("h2_max", "s21_db", 2 * F_C, -45.0, tol=None, bound="at_most", drive="P1", to="P2"),
            exp("fc_zero", "s21_db", F_C, 0.0, tol=None, bound="at_least", drive="P1", to="P2"),
            exp("between", "s21_db", 470e6, -3.0, tol=None, bound="at_least", drive="P1", to="P2")]
    net = lpf_network(refs, sweep=[*lpf_network(refs).sweep, coarse], expectations=rows)
    got = run(b.ir(), [net], tmp_path, FakeAC())
    assert got["spice.rf.lpf.fc_min"].status is S.PASS and "at_least -1 dB" in got["spice.rf.lpf.fc_min"].message
    assert fx.SCOPE_NOTE in got["spice.rf.lpf.fc_min"].message and got["spice.rf.lpf.fc_min"].details["bound"] == "at_least"
    assert got["spice.rf.lpf.h2_max"].status is S.PASS
    fc_zero = got["spice.rf.lpf.fc_zero"]
    assert fc_zero.status is S.FAIL and fc_zero.details["deviation"] == pytest.approx(-lpf_attenuation(F_C)) and fc_zero.details["repair"] == "human"
    between = got["spice.rf.lpf.between"]  # 450 .. 600 MHz reads -0.1 .. -19.8 dB, but the row is read at 470 MHz itself: in the ripple band
    assert between.status is S.PASS and between.details["analysis_id"].startswith(fx.POINT_PREFIX) and between.details["bracket"]["exact"]
    # a bound with a requirement, or with a tolerance, is refused (by the RF IR type itself, or by the runner)
    req = maybe(lambda: exp("req", "s21_db", F_C, -1.0, tol=None, bound="at_least", requirement_id="req.x", drive="P1", to="P2"))
    both = maybe(lambda: exp("both", "s21_db", F_C, -1.0, bound="at_least", drive="P1", to="P2"))
    got = run(b.ir(), [lpf_network(refs, expectations=[r for r in (req, both) if r is not None])], tmp_path, FakeAC())
    assert req is None or "cannot claim a requirement yet" in got["spice.rf.lpf.req"].message
    assert both is None or "exactly one of them" in got["spice.rf.lpf.both"].message


def _zero_rows_network(f: float) -> tuple[CircuitIR, Any]:
    """A 50 ohm termination on P1 and a separately terminated P2: |S11| and |S21| are exactly 0 in any double-precision solve."""
    b = Board()
    b.r("R1", 50.0, "IN", "GND").r("R2", 50.0, "OUT", "GND")
    rows = [exp("rl", "s11_db", f, -30.0, tol=None, bound="at_most", drive="P1", to="P1"),
            exp("rl_min", "s11_db", f, -30.0, tol=None, bound="at_least", drive="P1", to="P1"),
            exp("rl_band", "s11_db", f, -40.0, tol=10.0, drive="P1", to="P1"),
            exp("iso", "s21_db", f, -30.0, tol=None, bound="at_most", drive="P1", to="P2"),
            exp("rel_at_zero", "rel_s21_db", f, -30.0, tol=None, bound="at_most", ref_at=u(2 * f, "Hz"), drive="P1", to="P2"),
            exp("ph", "phase21_deg", f, 0.0, tol=1.0, drive="P1", to="P2")]
    net = mk("RFNetwork", id="zero", members=["R1", "R2"], ports=[port("P1", "IN"), port("P2", "OUT")], sweep=[lin("f", f, f / 10)], expectations=rows)
    return b.ir(), net


def _assert_zero_rows(got: dict[str, ValidationResult], f: float) -> None:
    rl = got["spice.rf.zero.rl"]
    assert rl.status is S.PASS and fx.ZERO_NOTE in rl.message and "sees exactly its reference impedance" in rl.message, rl.message
    assert rl.details["measured"] is None and rl.details["zero_magnitude"]["sample_hz"] == [pytest.approx(f, rel=1e-12)] and rl.details["deviation"] == "-inf"
    assert json.loads(rl.model_dump_json())["details"]["bracket"]["y0"] == "-inf"  # nothing of the zero is lost on save (JSON has no infinity)
    assert got["spice.rf.zero.rl_min"].status is S.FAIL and got["spice.rf.zero.rl_min"].details["repair"] == "human"
    band = got["spice.rf.zero.rl_band"]
    assert band.status is S.FAIL and "outside every finite band" in band.message and band.details["deviation"] == "+inf"
    iso = got["spice.rf.zero.iso"]
    assert iso.status is S.PASS and "check that P2 is not shorted to its reference" in iso.message  # a mis-wired port reads 0 too
    # a relative level at a zero is -inf (PASS at_most); a zero at ref_at leaves nothing to be relative to; a zero has no phase
    rel = got["spice.rf.zero.rel_at_zero"]
    assert rel.status is S.FAIL and "the reference level at ref_at is -inf dB" in rel.message
    assert got["spice.rf.zero.ph"].status is S.FAIL and "has no phase" in got["spice.rf.zero.ph"].message


def test_an_exact_zero_is_judged_as_minus_infinity_on_the_fake_engine(tmp_path: Path):
    """|S| = 0 is not 'no level': it passes every at_most bound (with a hint), fails at_least and every finite band."""
    ir, net = _zero_rows_network(100e6)
    _assert_zero_rows(run(ir, [net], tmp_path, FakeAC()), 100e6)
    # a relative level whose 'at' reading is the zero and whose reference is not: -inf dB, on the passing side of at_most
    b = Board()
    b.r("R1", 50.0, "IN", "GND").c("C1", 1e-9, "IN", "OUT").r("R2", 50.0, "OUT", "GND")
    rows = [exp("rel", "rel_s21_db", 1e6, -30.0, tol=None, bound="at_most", ref_at=u(1e6, "Hz"), drive="P1", to="P2")]
    net = mk("RFNetwork", id="hp", members=["R1", "C1", "R2"], ports=[port("P1", "IN"), port("P2", "OUT")], sweep=[lin("f", 1e6, 1e3)], expectations=rows)
    assert run(b.ir(), [net], tmp_path, FakeAC())["spice.rf.hp.rel"].details["measured"] == pytest.approx(0.0, abs=1e-12)  # same frequency: 0 dB


@needs_ngspice
def test_ngspice_matched_pads_from_the_calculators_pass_their_at_most_reflection_rows(tmp_path: Path):
    """50 ohm pi pads from ``calc.rf.attenuator.pi.r_shunt`` / ``.r_series`` (the design's lo_pad / drv_pad / pa_pad rows: s21 nominal -A
    tol 0.2 dB, s11 at_most -20 dB) at 447.5625 MHz: on ngspice-42 |S11| of the 2.5 / 4 / 6 / 10 dB pads lands on exactly 0.0 (about half
    of a 0.5 dB grid does - the rest on 1e-16, about -303 to -319 dB): floating-point rounding, never a reason to FAIL a correct pad."""
    from ai_eda.tools.calc.recompute import CALCULATORS

    for a_db in (2.5, 4.0, 6.0, 10.0):
        z0, a = u(50.0, "ohm"), u(a_db, "dB")
        r_sh, r_se = (CALCULATORS[f"calc.rf.attenuator.pi.{k}"][0](z0, a) for k in ("r_shunt", "r_series"))
        b = Board()
        b.add("R1", SpiceBinding(device=SpiceDevice.R, value=r_sh, provenance=PROV), "IN", "GND")
        b.add("R2", SpiceBinding(device=SpiceDevice.R, value=r_se, provenance=PROV), "IN", "OUT")
        b.add("R3", SpiceBinding(device=SpiceDevice.R, value=r_sh, provenance=PROV), "OUT", "GND")
        dec20 = AnalysisSpec(id="dec20", kind=SpiceAnalysis.AC, params={"variation": u("dec"), "points": u(20), "fstart": u(100e6, "Hz"), "fstop": u(1e9, "Hz")},
                             provenance=PROV)
        net = mk("RFNetwork", id="pad", members=["R1", "R2", "R3"], ports=[port("P1", "IN"), port("P2", "OUT")], sweep=[dec20],
                 expectations=[exp("s21", "s21_db", F_C, -a_db, tol=0.2, drive="P1", to="P2"),
                               exp("s11", "s11_db", F_C, -20.0, tol=None, bound="at_most", drive="P1", to="P1")])
        got = run(b.ir(), [net], tmp_path / f"pad{a_db}", engine)
        s11 = got["spice.rf.pad.s11"]
        assert s11.status is S.PASS and got["spice.rf.pad"].status is S.PASS, (a_db, s11.message, got["spice.rf.pad"].message)
        assert s11.details["measured"] is None and fx.ZERO_NOTE in s11.message, (a_db, s11.details.get("measured"))  # the exact zero these pads hit
        assert got["spice.rf.pad.s21"].details["measured"] == pytest.approx(-a_db, abs=1e-9)
    # the same zeros at an exact lin point: a plain termination and an ideal isolation
    ir, net = _zero_rows_network(100e6)
    _assert_zero_rows(run(ir, [net], tmp_path / "zero", engine), 100e6)


def test_the_lambda_quarter_switch_states_are_driven_from_their_own_ports(tmp_path: Path):
    b = Board()
    refs = switch_board(b)
    got = run(b.ir(), [switch_network(refs)], tmp_path, FakeAC())
    assert got["spice.rf.trsw"].status is S.PASS, got["spice.rf.trsw"].message
    tx_rx, com_tx = got["spice.rf.trsw.tx.tx_rx"].details["measured"], got["spice.rf.trsw.rx.com_tx"].details["measured"]
    assert -36 < tx_rx < -33 and -29 < com_tx < -26  # the prototype's 34.3 / 27.5 dB isolation
    assert got["spice.rf.trsw.tx.tx_com"].details["measured"] > -0.5 and got["spice.rf.trsw.rx.com_rx"].details["measured"] > -0.5
    decks = {d["key"]: d for d in got["spice.rf.trsw"].details["decks"]}
    assert set(decks) == {"trsw.tx", "trsw.rx@COM"} and decks["trsw.rx@COM"]["drive"] == "COM"


def test_an_assumed_port_impedance_makes_a_pass_not_evidence(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ports = [port("P1", "IN"), mk("RFPort", name="P2", net="OUT", kind="port", z0_ohm=assumption(50.0, "a guessed load", "ohm"))]
    got = run(b.ir(), [lpf_network(refs, ports=ports)], tmp_path, FakeAC())
    r = got["spice.rf.lpf.s21_fc"]
    assert r.status is S.NOT_VERIFIED and "port P2 z0_ohm" in r.message and "not evidence" in r.message


def test_an_engine_failure_fails_the_network_and_an_environment_limit_does_not(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    got = run(ir, [lpf_network(refs)], tmp_path, FakeAC("fail"))
    assert got["spice.rf.lpf"].status is S.FAIL and "singular matrix (fake)" in got["spice.rf.lpf"].message
    assert got["spice.rf.lpf.s21_fc"].status is S.NOT_VERIFIED and "did not succeed" in got["spice.rf.lpf.s21_fc"].message
    got = run(ir, [lpf_network(refs)], tmp_path, FakeAC("unverifiable"))
    assert got["spice.rf.lpf"].status is S.NOT_VERIFIED and "could not be run here" in got["spice.rf.lpf"].message
    assert got["spice.rf.lpf.s21_fc"].status is S.NOT_VERIFIED


def test_superseded_results_are_retired_and_nothing_else(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    first = run(ir, [lpf_network(refs)], tmp_path, FakeAC())
    ir.validation.results.extend(first.values())
    ir.validation.results.append(ValidationResult(check_id="spice.v_out", status=S.PASS, message="design deck"))
    ir.validation.results.append(ValidationResult(check_id="spice.si.CLK", status=S.PASS, message="si"))
    second = run(ir, [lpf_network(refs, expectations=list(lpf_network(refs).expectations)[:1])], tmp_path, FakeAC())
    retired = {k for k, r in second.items() if r.status is S.NOT_APPLICABLE}
    assert retired == {"spice.rf.lpf.s21_fc", "spice.rf.lpf.s21_edge", "spice.rf.lpf.s21_h2", "spice.rf.lpf.s21_h3", "spice.rf.lpf.s11_fc"}
    assert all(second[k].details["superseded"] == "PASS" for k in retired)
    third = fx.spice_rf_results(ir, {}, tmp_path, design=mk("RFDesign", networks=[]))
    assert {r.check_id for r in third} == set(first) and all(r.status is S.NOT_APPLICABLE for r in third)
    assert fx.spice_rf_results(CircuitIR(project=ProjectMeta(id="x", name="x")), {}, tmp_path) == []


def test_duplicate_network_ids_are_refused_once(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    twice = [lpf_network(refs), lpf_network(refs)]
    if rf_ir is None:
        design = mk("RFDesign", networks=twice)
    else:  # the real type refuses it on construction; the runner still refuses a design built without validation
        with pytest.raises(ValueError, match="network id 'lpf' is used twice"):
            mk("RFDesign", networks=twice)
        design = rf_ir.RFDesign.model_construct(blocks=[rf_ir.RFBlock(id="trx")], networks=twice, frequency_plan=[], lab_items=[], rails=[],
                                                model_values=[], profile_keys=[])
    got = fx.spice_rf_results(b.ir(), {"spice": FakeAC()}, tmp_path, design=design)
    assert [r.check_id for r in got] == ["spice.rf.lpf"] and got[0].status is S.FAIL and "used 2 times" in got[0].message


def test_the_simulation_agent_appends_the_rf_results_after_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ai_eda.agents.simulation as sim

    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    design = mk("RFDesign", networks=[lpf_network(refs)])
    monkeypatch.setattr(sim, "spice_rf_results", lambda ir_, tools, workdir: fx.spice_rf_results(ir_, tools, workdir, design=design))
    res = sim.SimulationAgent().run(ir, AgentContext(workdir=tmp_path, tools={"spice": FakeAC()}))
    ids = [r.check_id for r in res.validation]
    assert ids[0] == "spice" and "spice.rf.lpf" in ids and ids.index("spice.rf.lpf") > 0
    assert any(n.startswith("spice.rf: 1 fixture network(s) (lpf PASS), 6 expectation(s): 6 PASS") for n in res.notes), res.notes
    # the pipeline entry reads ir.rf: an IR without one has no RF result
    assert fx.spice_rf_results(ir, {"spice": FakeAC()}, tmp_path) == []


# --------------------------------------------------------------------------- the RF IR part (skipped before it is merged)


@needs_bound
def test_one_sided_bounds_are_judged_on_their_passing_side(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    rows = [exp("fc_min", "s21_db", F_C, -1.0, tol=None, bound="at_least", drive="P1", to="P2"),
            exp("h2_max", "s21_db", 2 * F_C, -45.0, tol=None, bound="at_most", drive="P1", to="P2"),
            exp("fc_zero", "s21_db", F_C, 0.0, tol=None, bound="at_least", drive="P1", to="P2")]  # the ripple puts f_c at -0.073 dB
    got = run(b.ir(), [lpf_network(refs, expectations=rows)], tmp_path, FakeAC())
    assert got["spice.rf.lpf.fc_min"].status is S.PASS and got["spice.rf.lpf.h2_max"].status is S.PASS
    assert got["spice.rf.lpf.fc_zero"].status is S.FAIL and got["spice.rf.lpf.fc_zero"].details["bound"] == "at_least"


@needs_bound
def test_the_design_deck_retirement_leaves_the_rf_results_alone():
    ir = CircuitIR(project=ProjectMeta(id="x", name="x"))
    ir.validation.results += [ValidationResult(check_id="spice.rf.lpf", status=S.PASS, message="x"), ValidationResult(check_id="spice.rf.lpf.s21_fc", status=S.PASS, message="x")]
    assert stage.retire_expectation_results(ir, set(), S.NOT_APPLICABLE, "gone") == []
    assert stage.RF_CHECK_PREFIX == fx.CHECK_PREFIX


@needs_rf_field
def test_the_pipeline_reads_ir_rf(tmp_path: Path):
    from ai_eda.agents.simulation import SimulationAgent

    b = Board()
    refs = chebyshev_lpf(b)
    ir = b.ir()
    ir.rf = mk("RFDesign", networks=[lpf_network(refs)])
    res = SimulationAgent().run(ir, AgentContext(workdir=tmp_path, tools={"spice": FakeAC()}))
    got = {r.check_id: r for r in res.validation}
    assert got["spice.rf.lpf"].status is S.PASS and got["spice.rf.lpf.s21_fc"].ir_hash == ir.content_hash()


# --------------------------------------------------------------------------- the real engine (skipped without ngspice)


@needs_ngspice
def test_ngspice_lossless_lowpass_matches_the_calculator_within_0_05_db(tmp_path: Path):
    b = Board()
    refs = chebyshev_lpf(b)
    got = run(b.ir(), [lpf_network(refs)], tmp_path, engine)
    assert got["spice.rf.lpf"].status is S.PASS, got["spice.rf.lpf"].message
    for k, f in {"f300": 300e6, "fc": F_C, "edge": F_EDGE, "h2": 2 * F_C, "h3": 3 * F_C}.items():
        r = got[f"spice.rf.lpf.s21_{k}"]
        assert r.status is S.PASS and abs(r.details["measured"] + lpf_attenuation(f)) < 0.05 and r.tool == "ngspice-shared"
    raw = next(e for e in got["spice.rf.lpf.s21_fc"].evidence if e.path.endswith(".raw"))
    assert raw.content_hash == "sha256:" + hashlib.sha256(Path(raw.path).read_bytes()).hexdigest()


@needs_ngspice
def test_ngspice_q40_lowpass_passband_is_a_decibel_below_the_lossless_formula(tmp_path: Path):
    """The critic's rule: with Q-40 inductors the passband is 1.045 dB (lpf7_q40.cir), not the lossless 0.073 dB."""
    b = Board()
    refs = chebyshev_lpf(b)
    rows = [exp("fc", "s21_db", F_C, -1.045, tol=0.01, drive="P1", to="P2"), exp("h2", "s21_db", 2 * F_C, -52.8, tol=0.1, drive="P1", to="P2")]
    net = lpf_network(refs, loss_q={r: u(40.0) for r in refs if r.startswith("L")}, q_ref_hz=u(F_C, "Hz"), expectations=rows)
    got = run(b.ir(), [net], tmp_path, engine)
    assert got["spice.rf.lpf.fc"].status is S.PASS and got["spice.rf.lpf.h2"].status is S.PASS, [got[k].message for k in ("spice.rf.lpf.fc", "spice.rf.lpf.h2")]
    assert got["spice.rf.lpf.fc"].details["measured"] - (-lpf_attenuation(F_C)) < -0.9


@needs_ngspice
def test_ngspice_q40_double_tuned_tank_against_the_network_and_cohn(tmp_path: Path):
    b = Board()
    refs = tank_board(b)
    got = run(b.ir(), [tank_network(refs)], tmp_path, engine)
    assert got["spice.rf.tank2"].status is S.PASS, got["spice.rf.tank2"].message
    loss = -got["spice.rf.tank2.s21_w"].details["measured"]
    g1 = rf.butterworth_g_values(2)[0]
    cohn = 4.343 * 2 * g1 / ((g1 / 20.0) * 40.0)  # S. B. Cohn 1959: 4.343 sum(g) / (w Q_u), w = BW / f0 = g1 / Q_e
    assert loss == pytest.approx(4.2147, abs=1e-3) and abs(loss - cohn) < 0.5
    assert got["spice.rf.tank2.rel_m"].details["measured"] == pytest.approx(-28.832, abs=0.01)
    assert got["spice.rf.tank2.rel_p"].details["measured"] == pytest.approx(-17.792, abs=0.01)
    # the radio calculators: the exact network (the fixture rows' nominal) within 0.01 dB, Cohn's dissipation loss within 0.5 dB
    a = tank_calc_args()
    assert -loss == pytest.approx(top_c_calc("s21_db", *a.values(), u(F_TANK, "Hz")), abs=0.01)
    assert abs(loss - top_c_calc("dissipation_loss", a["n"], a["f0"], a["bw"], a["q_u"])) < 0.5
    for check, f in (("rel_m", F_TANK - F_T), ("rel_p", F_TANK + F_T)):
        want = top_c_calc("rel_s21_db", *a.values(), u(f, "Hz"), u(F_TANK, "Hz"))
        assert got[f"spice.rf.tank2.{check}"].details["measured"] == pytest.approx(want, abs=0.01)


@needs_ngspice
def test_ngspice_lambda_quarter_switch_states(tmp_path: Path):
    b = Board()
    refs = switch_board(b)
    got = run(b.ir(), [switch_network(refs)], tmp_path, engine)
    assert got["spice.rf.trsw"].status is S.PASS, got["spice.rf.trsw"].message
    assert len({d["hash"] for d in got["spice.rf.trsw"].details["decks"]}) == 2


@needs_ngspice
def test_ngspice_two_pole_crystal_ladder_with_a_floating_middle_node(tmp_path: Path):
    cm, rm, c0, fs = 6e-15, 25.0, 4e-12, 21.4e6
    lm = 1.0 / ((2 * math.pi * fs) ** 2 * cm)
    card = u(f".subckt XTAL21 1 2\nLM 1 a {format_spice_number(lm)}\nCM a b {format_spice_number(cm)}\nRM b 2 {format_spice_number(rm)}\nC0 1 2 {format_spice_number(c0)}\n.ends")
    xtal = bind(SpiceDevice.X, model_name="XTAL21", model_card=card)
    b = Board()
    b.add("Y1", xtal, "IN", "MID").add("Y2", xtal, "MID", "OUT").c("C1", 30e-12, "MID", "GND")

    def s21(f: float) -> float:
        w = 2 * math.pi * f
        zx = 1 / (1 / (rm + 1j * w * lm + 1 / (1j * w * cm)) + 1j * w * c0)
        z2 = zx + 50
        zmid = 1 / (1j * w * 30e-12 + 1 / z2)
        v_mid = zmid / (50 + zx + zmid)
        return 20 * math.log10(abs(2 * v_mid * 50 / z2))

    offsets = (-2000.0, 0.0, 2000.0)
    net = mk("RFNetwork", id="ladder", block="if_backend", members=["Y1", "Y2", "C1"], ports=[port("P1", "IN"), port("P2", "OUT")],
             sweep=[lin("f0", fs, 4000.0, 5)], expectations=[exp(f"s{i}", "s21_db", fs + d, s21(fs + d), drive="P1", to="P2") for i, d in enumerate(offsets)])
    got = run(b.ir(), [net], tmp_path, engine)
    assert got["spice.rf.ladder"].status is S.PASS, got["spice.rf.ladder"].message
    assert got["spice.rf.ladder"].details["decks"][0]["dc_path"]["resistors"] == [{"ref": "RDC_MID", "net": "MID"}]


@needs_ngspice
def test_ngspice_pm_tank_phase_in_its_three_bias_states(tmp_path: Path):
    """The exact deck of kr447/decided/pm_n12.cir: 50 ohm port, 1 nF DC block, R_s 1846.554 ohm, L 607.0 nH (series R w L / 40),
    10 nF bypass, 10 kohm bias feed, C_fixed 19.8165 pF, the model varactor, 1 Gohm probe; the bias is a control port's state level."""
    card = u(".model DVAR D (CJO=20p VJ=0.7 M=0.5)")
    b = Board()
    b.c("CD", 1e-9, "A", "B").r("RSR", 1846.554, "B", "T").l("L1", 6.069796e-07, "T", "LB").c("CBP", 10e-9, "LB", "GND").r("RB", 10e3, "VB", "LB")
    b.c("CF", 1.981650e-11, "T", "GND").add("D1", bind(SpiceDevice.D, model_name="DVAR", model_card=card), "GND", "T").r("RPR", 1e9, "T", "GND")
    pinned = {"bias_lo": (1.44, -21.4617), "bias_nom": (2.0, 1.3925), "bias_hi": (2.56, 18.2329)}  # ngspice-42, measured on the exact deck
    states = [mk("RFState", id=sid, port_dc_v={"VB": u(v, "V")}) for sid, (v, _) in pinned.items()]
    rows = [exp(f"ph_{sid}", "phase21_deg", F_T, phi, tol=1.0, state=sid, drive="P", to="TANK") for sid, (_, phi) in pinned.items()]
    net = mk("RFNetwork", id="pm_mod1", block="tx_chain", members=["CD", "RSR", "L1", "CBP", "RB", "CF", "D1", "RPR"],
             ports=[port("P", "A"), port("VB", "VB", kind="control"), port("TANK", "T", kind="probe")], states=states,
             loss_q={"L1": u(40.0)}, q_ref_hz=u(F_T, "Hz"), sweep=[lin("ft", F_T, 37296.875)], expectations=rows)
    got = run(b.ir(), [net], tmp_path, engine)
    assert got["spice.rf.pm_mod1"].status is S.PASS, got["spice.rf.pm_mod1"].message
    phi = {sid: got[f"spice.rf.pm_mod1.{sid}.ph_{sid}"].details["measured"] for sid in pinned}
    for sid, (v, want) in pinned.items():
        assert phi[sid] == pytest.approx(want, abs=0.01) and got[f"spice.rf.pm_mod1.{sid}.ph_{sid}"].details["port_dc_v"] == {"VB": v}
    slope = math.radians(phi["bias_hi"] - phi["bias_lo"]) / (2.56 - 1.44)  # the chord K_pm the deviation check takes
    assert slope == pytest.approx(0.6186, abs=5e-4)
