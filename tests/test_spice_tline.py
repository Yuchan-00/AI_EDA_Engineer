"""The lossless transmission line (``SpiceDevice.T``) through the SPICE compiler, the runner's deck check and ngspice.

Without ngspice: the compiler writes ``T<ref> p1+ p1- p2+ p2- td=<s> z0=<ohm>``
only for a binding that states exactly ``z0`` and ``td`` (positive, in ohm /
s) and nothing else, :func:`netlist_elements` reads the line back, and the
runner's :func:`validate_deck` refuses a ``T`` line without both (ngspice-42
simulates a line without ``td`` with a delay of its own and no error -
measured). With ngspice: the compiled deck reproduces the case measured on
ngspice-42 (5 V / 1 ns step, 40 ohm driver, 50 ohm / 1 ns line, 5 pF load:
max v(far end) 5.5757 V, 11.5 % overshoot) within 0.5 %, and obeys the
physics a lossless line must: nothing arrives before t_d, the launched step
is the 40 / 50 ohm divider, the line settles to the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import analysis_command, build, build_report, netlist_elements
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
    SimulationSetup,
    SourceRef,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    user_requirement,
)
from ai_eda.ir.simulation import DEVICE_PARAMS, NODE_COUNTS, PARAM_DEVICES
from ai_eda.tools.calc import format_spice_number
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis
from ai_eda.tools.spice.ngspice_shared import TLINE_PARAMS, validate_deck

DS = SourceRef(title="fixture", content_hash="sha256:" + "0" * 64)
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")
DERIVED = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
#: measured on ngspice-42 (Ubuntu libngspice0, 2026-09-27) through NgspiceShared.run: max v(B) of the fixture below
MEASURED_MAX_VB = 5.5757
runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")


def _part(ref: str, n_pins: int, binding: SpiceBinding) -> Component:
    pins = [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=USER) for i in range(1, n_pins + 1)]
    return Component(ref=ref, value=ref, pins=pins, provenance=DERIVED, spice=binding)


def line_binding(**params) -> SpiceBinding:
    values = {"z0": user_requirement(50.0, "ohm"), "td": user_requirement(1e-9, "s")}
    values.update(params)
    return SpiceBinding(device=SpiceDevice.T, pin_order=["1", "2", "3", "4"], params={k: v for k, v in values.items() if v is not None}, provenance=USER)


def tline_ir(tmp_path: Path | None = None, line: SpiceBinding | None = None) -> CircuitIR:
    """A 5 V / 1 ns step through a 40 ohm driver into a 50 ohm / 1 ns lossless line with 5 pF at the far end."""
    ir = CircuitIR(project=ProjectMeta(id="tline", name="tline", workdir=None if tmp_path is None else str(tmp_path)))
    ir.components = [
        _part("RS", 2, SpiceBinding(device=SpiceDevice.R, value=user_requirement(40.0, "ohm"), provenance=USER)),
        _part("T1", 4, line or line_binding()),
        _part("CL", 2, SpiceBinding(device=SpiceDevice.C, value=user_requirement(5e-12, "F"), provenance=USER)),
    ]

    def net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
        return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=DERIVED)

    ir.nets = [net("S", ("RS", "1")), net("A", ("RS", "2"), ("T1", "1")), net("B", ("T1", "3"), ("CL", "1")),
               net("GND", ("T1", "2"), ("T1", "4"), ("CL", "2"), kind=NetKind.GROUND)]
    pulse = {k: user_requirement(v) for k, v in (("v1", 0.0), ("v2", 5.0), ("td", 0.0), ("tr", 1e-9), ("tf", 1e-9), ("pw", 100e-9), ("per", 200e-9))}
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="S", reference_net="GND", kind=StimulusKind.PULSE, params=pulse, provenance=USER)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(10e-12, "s"), "stop": user_requirement(20e-9, "s")}, provenance=USER)],
    )
    return ir


def test_the_line_is_a_four_node_param_device():
    assert SpiceDevice.T in PARAM_DEVICES and NODE_COUNTS[SpiceDevice.T] == (4,) and DEVICE_PARAMS[SpiceDevice.T] == {"td": "s", "z0": "ohm"}
    assert TLINE_PARAMS == frozenset({"z0", "td"})


def test_the_compiler_writes_the_lossless_line_and_reads_it_back():
    text = build(tline_ir())
    line = f"T1 A 0 B 0 td={format_spice_number(1e-9)} z0={format_spice_number(50.0)}"
    assert line in text.splitlines() and text.endswith(".end\n")
    assert validate_deck(text)[0] == []
    elements = {name: (nodes, rest) for name, nodes, rest in netlist_elements(text)}
    assert elements["T1"] == (["A", "0", "B", "0"], f"td={format_spice_number(1e-9)} z0={format_spice_number(50.0)}")
    assert build(tline_ir()) == text  # deterministic
    report = build_report(tline_ir())
    assert "T1" in report["elements"] and report["value_sources"].get("T1") is None


def test_the_compiler_refuses_an_incomplete_or_wrong_line():
    cases = [
        (line_binding(td=None), "missing \\['td'\\]"),
        (line_binding(z0=None), "missing \\['z0'\\]"),
        (line_binding(nl=user_requirement(0.25)), "not allowed \\['nl'\\]"),
        (line_binding(z0=user_requirement(0.0, "ohm")), "z0 must be a positive finite number"),
        (line_binding(td=user_requirement(-1e-9, "s")), "td must be a positive finite number"),
        (line_binding(td=user_requirement(1.0, "ns")), "carries unit 'ns'"),
        (line_binding(z0=user_requirement("50")), "positive finite number"),
    ]
    for binding, match in cases:
        with pytest.raises(CompileError, match=match):
            build(tline_ir(line=binding))
    with_value = line_binding()
    with_value.value = user_requirement(50.0, "ohm")
    with pytest.raises(CompileError, match="not a value or a model"):
        build(tline_ir(line=with_value))
    with_model = line_binding()
    with_model.model_name = "LOSSY"
    with pytest.raises(CompileError, match="not a value or a model"):
        build(tline_ir(line=with_model))
    three = line_binding()
    three.pin_order = ["1", "2", "3"]
    three.ignored_pins = {"4": "not used"}
    with pytest.raises(CompileError, match="takes 4 nodes"):
        build(tline_ir(line=three))


def test_the_runner_refuses_a_line_that_ngspice_would_misread():
    base = "x\nvin s 0 pulse(0 5 0 1n 1n 100n 200n)\nrs s a 40\n{line}\ncl b 0 5p\n.end\n"
    assert validate_deck(base.format(line="t1 a 0 b 0 z0=50 td=1n"))[0] == []
    assert validate_deck(base.format(line="T1 a 0 b 0 TD=1n Z0=50"))[0] == []  # ngspice matches param names case-insensitively
    for line, expect in (("t1 a 0 b 0 z0=50", "needs exactly z0= and td="), ("t1 a 0 b 0 td=1n", "needs exactly z0= and td="),
                         ("t1 a 0 b 0 z0=50 f=1g nl=0.25", "needs exactly z0= and td="), ("t1 a 0 b 0 z0 = 50 td=1n", "takes only z0=<ohm> td=<s>"),
                         ("t1 a 0 b 0 z0=50 td=1n td=2n", "needs exactly z0= and td=")):
        problems = validate_deck(base.format(line=line))[0]
        assert any(expect in p for p in problems), (line, problems)


def _at(time: list[float], values: list[float], t: float) -> float:
    i = min(range(len(time)), key=lambda k: abs(time[k] - t))
    return values[i]


@needs_ngspice
def test_ngspice_reproduces_the_measured_overshoot_of_a_lossless_line(tmp_path: Path):
    ir = tline_ir(tmp_path)
    ref = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path))
    spec = ir.simulation.analyses[0]
    res = runner.run(Path(ref.path), SpiceAnalysis.TRAN, tmp_path / "run", analysis_command(spec, ir.simulation))
    assert res.succeeded, res.errors
    time, va, vb = res.vectors["time"], res.vectors["a"], res.vectors["b"]
    peak = max(vb)
    assert peak == pytest.approx(MEASURED_MAX_VB, rel=0.005)
    assert (peak - 5.0) / 5.0 == pytest.approx(0.115, abs=0.005)
    # the physics of an ideal line: nothing arrives before t_d, the launched step is the 40 / 50 ohm divider until the
    # reflection returns at 2 t_d, and the line settles to the source
    assert abs(_at(time, vb, 0.9e-9)) < 0.05
    assert _at(time, va, 1.5e-9) == pytest.approx(5.0 * 50.0 / 90.0, rel=0.01)
    assert vb[-1] == pytest.approx(5.0, abs=0.05)
    assert {"t1#i1", "t1#i2"} <= set(res.vectors)
