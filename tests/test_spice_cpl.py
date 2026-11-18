"""The coupled two-conductor line (``SpiceDevice.P``, ngspice's CPL) through the SPICE compiler, the runner's deck check and ngspice.

Without ngspice: the compiler writes ``P<ref> <6 nodes> <ref>_cpl`` and, on the
next line, ``.model <ref>_cpl CPL length=... R=1 0 1 L=... G=0 0 0 C=...``
only for a binding that states exactly ``l11 l12 l22 c11 c22 cm length``
(positive, in H/m, F/m and m) with both GND pins on the ground net and
positive-definite matrices; :func:`validate_deck` refuses a ``P`` line with
another node count, a non-ground reference node, no model name or an instance
length; the rawfile cross-check compares the four branch currents ngspice
writes under one name by position; decks without a ``P`` line compile as they
did before it existed.

With ngspice (KiCad 10.0.6's ``ngspice.dll``, ngspice-46; measured
2026-11-13, :mod:`ai_eda.tools.spice.ngspice_shared` item 12): every deck runs
in a process of its own (the runner loads one CPL deck per process). A
four-end matched pair on the 4-layer generic F.Cu (h 0.2 mm, t 35 um, er 4.5,
the 50-ohm width, s/h = 1.5) is compared with the closed forms of
:mod:`ai_eda.tools.calc.xtalk` (Simonovich Eq. 8 with Eq. 4 / 6 for NEXT, the
velocity-difference FEXT): NEXT / V_A within 5 % at 20 mm and 100 mm, FEXT /
V_A within 10 % at 20 mm, V_A read at the aggressor's input node. A weakly
coupled pair matches two lossless T lines of the same Z0 and delay within
1 %; a pair with no coupling at all is refused by the engine; the
time-step-versus-tau check of a 1 mm line is pinned.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import CPL_R_OHM_PER_M, analysis_command, build, netlist_elements
from ai_eda.errors import CompileError
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
    SimElement,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    user_requirement,
)
from ai_eda.ir.simulation import DEVICE_PARAMS, NODE_COUNTS, PARAM_DEVICES
from ai_eda.tools.calc.tline import edge_coupled_microstrip, propagation_delay, solve_microstrip_width
from ai_eda.tools.calc.xtalk import fext_rel, kb_eq8, l_sat_mm, maxwell_from_modes, next_rel
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, rawfile
from ai_eda.tools.spice.ngspice_shared import _Engine, _Vector, cpl_refusal, validate_deck
from ai_eda.tools.spice.si_check import deck_ir

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")
DERIVED = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
u = user_requirement
needs_ngspice = pytest.mark.skipif(not NgspiceShared().available(), reason="ngspice shared library not found")

#: the 4-layer generic F.Cu, as tests/test_xtalk_calc.py: h (mm), t (um), er
H, T, ER = 0.2, 35.0, 4.5
T_R = 1e-9
EMF = 2.0


def _part(ref: str, n_pins: int, binding: SpiceBinding) -> Component:
    pins = [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=USER) for i in range(1, n_pins + 1)]
    return Component(ref=ref, value=ref, pins=pins, provenance=DERIVED, spice=binding)


def _net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=DERIVED)


ROUND = {"l11": (3e-7, "H/m"), "l12": (4e-8, "H/m"), "l22": (3e-7, "H/m"), "c11": (1.2e-10, "F/m"), "cm": (7e-12, "F/m"), "c22": (1.2e-10, "F/m"), "length": (0.02, "m")}


def line_binding(values: dict[str, tuple[float, str]] | None = None, **over: Traced | None) -> SpiceBinding:
    params: dict[str, Traced | None] = {k: u(v, unit) for k, (v, unit) in (values or ROUND).items()}
    params.update(over)
    return SpiceBinding(device=SpiceDevice.P, pin_order=["1", "2", "3", "4", "5", "6"], params={k: v for k, v in params.items() if v is not None}, provenance=USER)


def pair_ir(line: SpiceBinding | None = None, *, r_term: float = 50.0, step: float = 1e-12, stop: float = 4e-9, project: str = "cpl") -> CircuitIR:
    """A step of EMF volts behind ``r_term`` into line 1 (A1 -> B1) of the pair; every other end (A2, B1, B2) is ``r_term`` to ground."""
    ir = CircuitIR(project=ProjectMeta(id=project, name=project))

    def res(ref: str) -> Component:
        return _part(ref, 2, SpiceBinding(device=SpiceDevice.R, value=u(r_term, "ohm"), provenance=USER))

    ir.components = [res("RS"), res("RV"), _part("TL1", 6, line or line_binding()), res("RA"), res("RB")]
    ir.nets = [_net("S", ("RS", "1")), _net("A1", ("RS", "2"), ("TL1", "1")), _net("A2", ("RV", "1"), ("TL1", "2")),
               _net("B1", ("TL1", "4"), ("RA", "1")), _net("B2", ("TL1", "5"), ("RB", "1")),
               _net("GND", ("RV", "2"), ("TL1", "3"), ("TL1", "6"), ("RA", "2"), ("RB", "2"), kind=NetKind.GROUND)]
    pulse = {k: u(v) for k, v in (("v1", 0.0), ("v2", EMF), ("td", 0.0), ("tr", T_R), ("tf", T_R), ("pw", 100e-9), ("per", 200e-9))}
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="S", reference_net="GND", kind=StimulusKind.PULSE, params=pulse, provenance=USER)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": u(step, "s"), "stop": u(stop, "s")}, provenance=USER)],
    )
    return ir


def _p1_pair(s_over_h: float = 1.5):
    """The P1 geometry: the modal values (Kirschning-Jansen through calc.tline) and their Maxwell matrices."""
    w = solve_microstrip_width(50.0, H, T, ER)
    m = edge_coupled_microstrip(w, s_over_h * H, H, T, ER)
    return m, maxwell_from_modes(m.z_even, m.z_odd, m.e_eff_even, m.e_eff_odd)


def _p1_values(length_m: float) -> dict[str, tuple[float, str]]:
    _, mx = _p1_pair()
    return {"l11": (mx.l11, "H/m"), "l12": (mx.l12, "H/m"), "l22": (mx.l11, "H/m"), "c11": (mx.c11, "F/m"), "cm": (-mx.c12, "F/m"),
            "c22": (mx.c11, "F/m"), "length": (length_m, "m")}


# --------------------------------------------------------------------------- the device and the compiler


def test_the_coupled_line_is_a_six_node_param_device():
    assert SpiceDevice.P in PARAM_DEVICES and NODE_COUNTS[SpiceDevice.P] == (6,)
    assert DEVICE_PARAMS[SpiceDevice.P] == {"c11": "F/m", "c22": "F/m", "cm": "F/m", "l11": "H/m", "l12": "H/m", "l22": "H/m", "length": "m"}
    assert CPL_R_OHM_PER_M == 1.0


GOLDEN = """cpl
RA B1 0 50
RB B2 0 50
RS S A1 50
RV A2 0 50
PTL1 A1 A2 0 B1 B2 0 TL1_cpl
.model TL1_cpl CPL length=20m R=1 0 1 L=3e-7 40n 3e-7 G=0 0 0 C=120p -7p 120p
VVIN S 0 PULSE(0 2 0 1n 1n 1e-7 2e-7)
.end
"""


def test_the_compiler_writes_the_element_and_its_model_line():
    text = build(pair_ir())
    assert text == GOLDEN
    problems, info = validate_deck(text)
    assert problems == [] and info["coupled_lines"] == 1
    elements = {name: (nodes, rest) for name, nodes, rest in netlist_elements(text)}
    assert elements["PTL1"] == (["A1", "A2", "0", "B1", "B2", "0"], "TL1_cpl")
    assert build(pair_ir()) == text  # deterministic


def _refused(binding: SpiceBinding, match: str, ir: CircuitIR | None = None) -> None:
    with pytest.raises(CompileError, match=match):
        build(ir or pair_ir(binding))


def test_the_compiler_refuses_an_incomplete_or_impossible_line():
    _refused(line_binding(l12=u(3e-7, "H/m")), "inductance matrix is not positive definite")
    _refused(line_binding(cm=u(1.3e-10, "F/m")), "capacitance matrix is not positive definite")
    _refused(line_binding(length=None), "missing \\['length'\\]")
    _refused(line_binding(td=u(1e-9, "s")), "not allowed \\['td'\\]")
    _refused(line_binding(length=u(20.0, "mm")), "carries unit 'mm'")
    _refused(line_binding(cm=u(0.0, "F/m")), "cm must be a positive finite number")
    _refused(line_binding(l12=u(-4e-8, "H/m")), "l12 must be a positive finite number")
    _refused(line_binding(l11=Traced(value=3e-7, unit="H/m", provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED, tool="llm"))), "llm_generated")
    with_value = line_binding()
    with_value.value = u(1.0, "ohm")
    _refused(with_value, "not a value or a model")
    with_model = line_binding()
    with_model.model_name = "TL1_cpl"
    _refused(with_model, "not a value or a model")
    lifted = line_binding()
    lifted.pin_order = ["1", "2", "4", "3", "5", "6"]  # GND1 on B1, line 1 out on the ground net
    _refused(lifted, "GND1 and GND2 .* must be on the ground net")
    five = line_binding()
    five.pin_order = ["1", "2", "3", "4", "5"]
    five.ignored_pins = {"6": "not used"}
    _refused(five, "takes 6 nodes")
    clash = pair_ir()
    clash.simulation.elements = [SimElement(id="DX", device=SpiceDevice.D, net="B1", reference_net="GND", model_name="TL1_cpl",
                                            model_card=u(".model TL1_cpl D(is=1e-14)"), reason="fixture", provenance=USER)]
    _refused(line_binding(), "also defined by a model_card", ir=clash)


#: the spice.si deck and the T-line fixture as the compiler wrote them before the coupled line existed (base 64127a1)
SI_DECK_BEFORE = """si_deck
CL B 0 5p
RS S A 25
T1 A 0 B 0 td=450p z0=50
VVDRV S 0 PULSE(0 1 0 1n 1n 41.75n 85.5n)
.end
"""


def test_decks_without_a_coupled_line_compile_as_before():
    ir, _, _, _ = deck_ir("si_deck", z0=50.0, td=0.45e-9, t_r=1e-9, r_drive=25.0, c_load=5e-12)
    text = build(ir)
    assert text == SI_DECK_BEFORE
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "0ea69c2171071f99ca1367ebb6f01f8b9c41d02f3bd7a15e75fa47ac00a832f6"
    assert validate_deck(text)[1] == {"title": "si_deck", "coupled_lines": 0}


# --------------------------------------------------------------------------- the runner's deck check


BASE = ("x\nv1 s 0 pulse(0 1 0 1n 1n 100n 200n)\nr1 s a1 50\nr2 a2 0 50\n{line}\n"
        ".model pl cpl length=0.02 r=1 0 1 l=3e-7 4e-8 3e-7 g=0 0 0 c=1.2e-10 -7e-12 1.2e-10\nr3 b1 0 50\nr4 b2 0 50\n.end\n")


def test_the_runner_refuses_a_coupled_line_ngspice_would_misread():
    for line in ("p1 a1 a2 0 b1 b2 0 pl", "P1 a1 a2 gnd b1 b2 gnd PL"):
        problems, info = validate_deck(BASE.format(line=line))
        assert problems == [] and info["coupled_lines"] == 1, (line, problems)
    for line, expect in (
        ("p1 a1 0 b1 0 pl", "needs 6 nodes and a model"),
        ("p1 a1 a2 0 b1 b2 0", "needs 6 nodes and a model"),
        ("p1 a1 a2 a3 0 b1 b2 b3 0 pl", "exactly 6 nodes"),
        ("p1 a1 a2 x b1 b2 0 pl", "must be ground"),
        ("p1 a1 a2 0 b1 b2 y pl", "must be ground"),
        ("p1 a1 a2 0 b1 b2 0 pl len=0.02", "not on the instance"),
        ("p1 a1 a2 0 b1 b2 0 pl LENGTH=0.02", "not on the instance"),
        ("p1 a1 a2 0 b1 b2 0 len=0.02", "not on the instance"),
        ("p1 a1 a2 0 b1 b2 0 m=1", "needs a model name"),
    ):
        problems = validate_deck(BASE.format(line=line))[0]
        assert any(expect in p for p in problems), (line, problems)


def test_one_coupled_line_deck_per_process():
    assert cpl_refusal(None) is None
    text = cpl_refusal("C:/work/first.cir")
    assert "C:/work/first.cir" in text and "process of its own" in text


#: the library modules the crosstalk change (P1 - P4) touched, under ``ai_eda/``
CHANGED_MODULES = ("agents/simulation.py", "compilers/spice.py", "gui/page.py", "ir/si.py", "ir/simulation.py", "parts/datasheet_facts.py",
                   "report/si_report.py", "tools/calc/xtalk.py", "tools/routing/coupling.py", "tools/si/paths.py", "tools/spice/ngspice_shared.py",
                   "tools/spice/rawfile.py", "tools/spice/stage.py", "tools/spice/xtalk_check.py", "validation/domain.py", "validation/si.py")


def test_the_changed_modules_space_every_equals_sign_alike():
    """An ``=`` has a space on both sides (``x = 1``) or on neither (``f(x=1)``): a review found ``self._dll_dir =os...`` and
    ``lines =[...]`` in ngspice_shared.py, two lines the change only reformatted (spaced at base 22d7895)."""
    root = Path(__file__).resolve().parent.parent / "ai_eda"
    lopsided = []
    for name in CHANGED_MODULES:
        source = (root / name).read_text(encoding="utf-8")
        skip = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)
        toks = [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type not in skip]
        for a, t, b in zip(toks, toks[1:], toks[2:]):
            if t.type == tokenize.OP and t.string == "=" and a.end[0] == t.start[0] == b.start[0] and (t.start[1] > a.end[1]) != (b.start[1] > t.end[1]):
                lopsided.append(f"{name}:{t.start[0]}: {t.line.strip()}")
    assert lopsided == []


def _write_raw(path: Path, names: list[tuple[str, str]], columns: list[list[float]]) -> None:
    lines = ["Title: t", "Date: d", "Plotname: Transient Analysis", "Flags: real", f"No. Variables: {len(names)}", f"No. Points: {len(columns[0])}", "Variables:"]
    lines += [f"\t{i}\t{n}\t{t}" for i, (n, t) in enumerate(names)]
    lines.append("Values:")
    for k in range(len(columns[0])):
        lines.append(f" {k}\t" + "\n\t".join(f"{c[k]:.15e}" for c in columns))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_rawfile_check_compares_alike_names_by_position(tmp_path: Path):
    """ngspice-46 writes a CPL's four branch currents all as ``i(p1)``, in the sorted order of their plot names (measured)."""
    time = [0.0, 1e-12, 2e-12]
    branches = {"p1#branch1_0": [0.0, 1.0, 2.0], "p1#branch1_1": [0.0, -1.0, -2.0], "p1#branch2_0": [0.0, 3.0, 4.0], "p1#branch2_1": [0.0, -3.0, -4.0]}
    raw = tmp_path / "x.tran.raw"
    _write_raw(raw, [("time", "time"), ("v(a1)", "voltage"), *[("i(p1)", "current")] * 4], [time, [0.0, 0.5, 1.0], *branches.values()])
    plot = rawfile.parse(raw)
    assert plot.vectors["i(p1)"] == branches["p1#branch2_1"] and plot.repeated == {"i(p1)": list(branches.values())}

    def vec(name: str, v_type: int, data: list[float]) -> _Vector:
        return _Vector(name=name, v_type=v_type, type_name="", v_flags=1, length=len(data), complex=False, data=data)

    vecs = {"time": vec("time", 1, time), "a1": vec("a1", 3, [0.0, 0.5, 1.0])}
    vecs.update({n: vec(n, 4, d) for n, d in reversed(branches.items())})  # the API's own order is not the file's
    assert _Engine._rawfile_mismatch(raw, vecs, 3, "Transient Analysis") == ""
    vecs["p1#branch1_1"], vecs["p1#branch2_0"] = vec("p1#branch1_1", 4, branches["p1#branch2_0"]), vec("p1#branch2_0", 4, branches["p1#branch1_1"])
    assert "#1 (plot vector 'p1#branch1_1')" in _Engine._rawfile_mismatch(raw, vecs, 3, "Transient Analysis")
    short = tmp_path / "short.tran.raw"  # as many variables as plot vectors, but one i(p1) column fewer
    _write_raw(short, [("time", "time"), ("v(a1)", "voltage"), ("v(a2)", "voltage"), *[("i(p1)", "current")] * 3], [time, [0.0, 0.5, 1.0], time, *list(branches.values())[:3]])
    vecs = {"time": vec("time", 1, time), "a1": vec("a1", 3, [0.0, 0.5, 1.0]), **{n: vec(n, 4, d) for n, d in branches.items()}}
    assert "the file lists that name 3 times" in _Engine._rawfile_mismatch(short, vecs, 3, "Transient Analysis")
    single = tmp_path / "y.tran.raw"
    _write_raw(single, [("time", "time"), ("v(a1)", "voltage")], [time, [0.0, 0.5, 1.0]])
    assert rawfile.parse(single).repeated == {}


# --------------------------------------------------------------------------- the engine, one process per deck

_PROBE = r"""
import json, sys
from pathlib import Path
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis
runner = NgspiceShared()
runs = []
for deck, command in json.loads(sys.argv[1]):
    p = Path(deck)
    res = runner.run(p, SpiceAnalysis.TRAN, p.parent / (p.stem + "_run"), command)
    keep = json.loads(sys.argv[2])
    runs.append({"succeeded": res.succeeded, "errors": res.errors, "unverifiable": res.unverifiable, "n_points": res.n_points,
                 "peak": {k: max(v, key=abs) for k, v in res.vectors.items() if k != "time"},
                 "final": {k: v[-1] for k, v in res.vectors.items()}, "vectors": {k: res.vectors[k] for k in keep if k in res.vectors}})
print(json.dumps({"version": runner.version(), "build": runner.build(), "runs": runs}))
"""


def _engine(tmp_path: Path, decks: list[tuple[Path, str]], keep: tuple[str, ...] = ()) -> dict:
    """Run ``decks`` ((netlist, tran command) pairs) in one new process; its JSON summary."""
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", _PROBE, json.dumps([[str(p), c] for p, c in decks]), json.dumps(list(keep))], cwd=tmp_path, env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["version"].startswith("ngspice-")
    return summary


def _compiled(tmp_path: Path, ir: CircuitIR) -> tuple[Path, str]:
    ref = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path))
    return Path(ref.path), analysis_command(ir.simulation.analyses[0], ir.simulation)


@needs_ngspice
@pytest.mark.parametrize("length_mm", [20.0, 100.0])
def test_a_matched_pair_against_the_closed_forms(tmp_path: Path, length_mm: float):
    """All four ends at sqrt(Ze Zo): NEXT / V_A within 5 % of kb_eq8 min(1, L / L_sat), FEXT / V_A within 10 % of fext_rel (20 mm)."""
    m, _ = _p1_pair()
    r_t = math.sqrt(m.z_even * m.z_odd)
    ir = pair_ir(line_binding(_p1_values(length_mm / 1000.0)), r_term=r_t, step=1e-12, stop=T_R + 4.0 * length_mm / 1000.0 * math.sqrt(m.e_eff_even) / 299792458.0 + 2e-9,
                 project=f"pair{length_mm:g}")
    run = _engine(tmp_path, [_compiled(tmp_path, ir)])["runs"][0]
    assert run["succeeded"], run["errors"]  # the rawfile cross-check included (the four i(ptl1) columns by position)
    v_a = run["final"]["a1"]
    assert v_a == pytest.approx(EMF / 2.0, rel=2e-3)  # the divider, the line's 1 ohm/m being 0.02-0.1 ohm
    kb = kb_eq8(m.z_even, m.z_odd, r_t)
    expect_next = next_rel(kb, length_mm, l_sat_mm(T_R, propagation_delay(m.e_eff_even)))
    assert run["peak"]["a2"] / v_a == pytest.approx(expect_next, rel=0.05)
    if length_mm == 20.0:
        assert run["peak"]["b2"] / v_a == pytest.approx(fext_rel(length_mm, m.e_eff_even, m.e_eff_odd, T_R), rel=0.10)
    else:  # saturated: the plateau is Kb itself
        assert run["peak"]["a2"] / v_a == pytest.approx(kb, rel=0.01)


def _two_t_lines(l12: float, c12: float) -> str:
    """A CPL pair next to two T lines of each CPL line's own Z0 and delay, with the same drivers and loads (20 mm)."""
    z1, td1, z2, td2, length = 50.0, 6e-9, 60.0, 6.5e-9, 0.02
    return f"""cpl vs t
v1 s1 0 pulse(0 1 0 1n 1n 100n 200n)
r1 s1 a1 25
v2 s2 0 pulse(0 1 0.3n 0.5n 0.5n 100n 200n)
r2 s2 a2 40
p1 a1 a2 0 b1 b2 0 pl
.model pl cpl length={length!r} r=1 0 1 l={z1 * td1!r} {l12!r} {z2 * td2!r} g=0 0 0 c={td1 / z1!r} {c12!r} {td2 / z2!r}
c1 b1 0 5p
r3 b2 0 100
v3 t1s 0 pulse(0 1 0 1n 1n 100n 200n)
r4 t1s ta1 25
t1 ta1 0 tb1 0 z0={z1!r} td={td1 * length!r}
c2 tb1 0 5p
v4 t2s 0 pulse(0 1 0.3n 0.5n 0.5n 100n 200n)
r5 t2s ta2 40
t2 ta2 0 tb2 0 z0={z2!r} td={td2 * length!r}
r6 tb2 0 100
.end
"""


@needs_ngspice
def test_a_weakly_coupled_pair_matches_two_lossless_lines(tmp_path: Path):
    """L12 and C12 at 1e-6 of line 1's: each far end within 1 % (of its peak) of a T line with that line's Z0 and delay."""
    deck = tmp_path / "weak.cir"
    deck.write_text(_two_t_lines(1e-6 * 50.0 * 6e-9, -1e-6 * 6e-9 / 50.0), encoding="utf-8", newline="\n")
    run = _engine(tmp_path, [(deck, "tran 1p 6n")], keep=("b1", "tb1", "b2", "tb2"))["runs"][0]
    assert run["succeeded"], run["errors"]
    v = run["vectors"]
    for cpl_end, t_end in (("b1", "tb1"), ("b2", "tb2")):
        peak = max(abs(x) for x in v[t_end])
        assert max(abs(a - b) for a, b in zip(v[cpl_end], v[t_end])) <= 0.01 * peak, cpl_end


@needs_ngspice
def test_the_engine_refuses_a_pair_with_no_coupling(tmp_path: Path):
    """L12 = C12 = 0: "Forbidden combination of model parameters!" and a ControlledExit (measured), not two T lines."""
    deck = tmp_path / "uncoupled.cir"
    deck.write_text(_two_t_lines(0.0, 0.0), encoding="utf-8", newline="\n")
    run = _engine(tmp_path, [(deck, "tran 1p 6n")])["runs"][0]
    assert not run["succeeded"] and any("Forbidden combination of model parameters" in e for e in run["errors"]), run["errors"]


@needs_ngspice
def test_the_tau_check_of_a_1_mm_line(tmp_path: Path):
    """A 1 mm line runs clean at a 1 ps step; at 10 ps ngspice lowers its own tmax and says so on stderr (a failed run here)."""
    m, _ = _p1_pair()
    r_t = math.sqrt(m.z_even * m.z_odd)
    fine = _engine(tmp_path, [_compiled(tmp_path, pair_ir(line_binding(_p1_values(1e-3)), r_term=r_t, step=1e-12, stop=4e-9, project="mm1"))])["runs"][0]
    assert fine["succeeded"], fine["errors"]
    coarse_dir = tmp_path / "coarse"
    coarse_dir.mkdir()
    coarse = _engine(coarse_dir, [_compiled(coarse_dir, pair_ir(line_binding(_p1_values(1e-3)), r_term=r_t, step=10e-12, stop=4e-9, project="mm1"))])["runs"][0]
    assert not coarse["succeeded"] and any("too large for CPL tau" in e for e in coarse["errors"]), coarse["errors"]


@needs_ngspice
def test_the_measured_defect_an_unmatched_pair_at_a_fine_step_settles_off(tmp_path: Path):
    """ngspice_shared item 12d, pinned: 25 ohm driver, 5 pF loads, s/h 0.5, 10 mm, a 2 ns edge. At a 1.27 ps step the run
    succeeds and settles 1.3 % below its DC level (the far end is a capacitor: it must reach the 1 V source); at 10 ps it
    settles within 1e-4. If this fails, the engine changed: measure item 12 again before a verdict rests on a CPL deck."""
    _, mx = _p1_pair(0.5)
    text = (f"cpl unmatched\nv1 s1 0 pulse(0 1 0 2e-9 2e-9 1 2)\nr1 s1 a1 25\nr2 a2 0 25\nc1 b1 0 5e-12\nc2 b2 0 5e-12\np1 a1 a2 0 b1 b2 0 pl\n"
            f".model pl cpl length=0.01 r=1 0 1 l={mx.l11!r} {mx.l12!r} {mx.l11!r} g=0 0 0 c={mx.c11!r} {mx.c12!r} {mx.c11!r}\n.end\n")
    finals = {}
    for step in ("1.271e-12", "1e-11"):
        sub = tmp_path / step
        sub.mkdir()
        deck = sub / "unmatched.cir"
        deck.write_text(text, encoding="utf-8", newline="\n")
        run = _engine(sub, [(deck, f"tran {step} 9.021e-09")])["runs"][0]
        assert run["succeeded"], run["errors"]  # no message marks the wrong run
        finals[step] = run["final"]
    assert abs(finals["1.271e-12"]["b1"] - 1.0) > 0.005 and abs(finals["1.271e-12"]["a1"] - 1.0) > 0.005
    assert abs(finals["1e-11"]["b1"] - 1.0) < 1e-4 and abs(finals["1e-11"]["a1"] - 1.0) < 1e-4


@needs_ngspice
def test_a_second_coupled_line_deck_in_one_process_is_not_run(tmp_path: Path):
    m, _ = _p1_pair()
    r_t = math.sqrt(m.z_even * m.z_odd)
    deck, command = _compiled(tmp_path, pair_ir(line_binding(_p1_values(0.02)), r_term=r_t, step=1e-12, stop=3e-9, project="twice"))
    runs = _engine(tmp_path, [(deck, command), (deck, command)])["runs"]
    assert runs[0]["succeeded"], runs[0]["errors"]
    assert not runs[1]["succeeded"] and "already loaded a coupled-line (CPL) deck" in (runs[1]["unverifiable"] or "")
