"""The ``i2c_pullup`` catalog template (PLAN D9a, the first user of simulation-only elements): table, refusals, provenance,
determinism, the rise verified by ngspice (and a too-large pull-up shown to fail), a clean ERC on a schematic without the elements.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.spice import build
from ai_eda.design import templates as templates_mod
from ai_eda.design.base import CHOICE_NOTE_PREFIX
from ai_eda.design.catalog.i2c_pullup import I2cPullupTemplate
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, user_requirement
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import Orchestrator, Stage

_lib = KicadLibrary()
HAS_LIBS = _lib.symbol_file("Device") is not None and _lib.symbol_file("Connector_Generic") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not (HAS_LIBS and runner.available()), reason="ngspice shared library / KiCad libraries not found")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (HAS_LIBS and kicad.available()), reason="kicad-cli / KiCad libraries not installed")

BASE = {"application": "sensor bus", "jurisdiction": "EU"}
I2C = {"input_voltage": "3.3 V", "clock_frequency": "400 kHz", "bus_capacitance": "150 pF", "bus_rise_time": "300 ns"}


def test_the_template_is_registered_once():
    assert [t.id for t in templates_mod.all_templates()].count("i2c_pullup") == 1 and any(isinstance(t, I2cPullupTemplate) for t in templates_mod.TEMPLATES)


def _ir(tmp_path: Path, name: str = "bus") -> CircuitIR:
    return CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))


def _present(ir: CircuitIR, tmp_path: Path, answers: dict[str, str]):
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"] and ir.components == [], [q.key for q in state.open_questions]
    return state.open_questions[0].question


def _confirm(ir: CircuitIR, tmp_path: Path, stop_after: Stage, *, spice: bool = False):
    tools = {"kicad_library": _lib, **({"spice": runner} if spice else {})}
    ctx = AgentContext(workdir=tmp_path, tools=tools, answers={"confirm_design": "yes", "pcb.placement": "skip"})
    return Orchestrator(ctx).run(ir, stop_after=stop_after), ctx


@needs_libs
def test_the_table_names_the_window_the_elements_and_the_bound(tmp_path: Path):
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, I2C)  # the ATmega board (clock_frequency + input_voltage) stands back: not ambiguous
    assert "i2c_pullup" in question and "ambiguous" not in question
    # R_max = 300 ns / (0.8473 x 150 pF) = 2360 ohm, R_min = 2.9 V / 3 mA = 966.7 ohm, R = 0.8 R_max = 1888 ohm; 1888 is no part: R1 / R2 are the nearest E24 value inside the window, 1.8k
    for text in ("bus_capacitance", "bus_rise_time", "v_ol", "i_ol", "pullup_fraction", "driver_model", "r_max = 2360", "r_min = 966.6", "r_pullup_ideal = 1888", "r_pullup = 1800 ohm [calc.e_series.e24.nearest from r_pullup_ideal]",
                 "R1 Device:R", "J1 Connector_Generic:Conn_01x03", "SDA (signal): R1.2, J1.2", "GND (ground): " , "simulation-only elements: CBUS = 1.5e-10 F SDA-GND",
                 "expectation held_low: v(SDA) at t_hold = 1.125e-06 s = v_low = 0.66", "expectation rise: v(SDA) at t_check", "at least v_ih = 2.31 V", "not verified: V_OL"):
        assert text in question, text
    # the margin policy never undercuts R_min: 150 pF / 150 ns gives R_max = 1180 ohm, 0.8 R_max = 944 ohm < R_min, so R = R_min
    ir2 = _ir(tmp_path, "edge")
    question = _present(ir2, tmp_path, {**I2C, "bus_rise_time": "150 ns"})
    # ... and the part stays inside the window too: the nearest E24 value to 966.7 ohm is 1 kohm (>= R_min, <= R_max)
    assert "r_max = 1180" in question and "r_pullup_ideal = 966.6" in question and "r_pullup = 1000 ohm [calc.e_series.e24.nearest from r_pullup_ideal]" in question


@needs_libs
def test_a_bus_no_pull_up_can_serve_is_refused_with_both_bounds(tmp_path: Path):
    ir = _ir(tmp_path, "r")
    # 400 pF and 100 ns: R_max = 295 ohm < R_min = 967 ohm
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **I2C, "bus_capacitance": "400 pF", "bus_rise_time": "100 ns"})).run(ir, stop_after=Stage.ARCHITECTURE)
    out = state.outcome(Stage.ARCHITECTURE)
    assert ir.components == [] and "template i2c_pullup not proposed" in out.message and "no pull-up satisfies this bus" in out.message and "R_min = 966.6" in out.message, out.message
    for supply in ("12 V", "1.8 V"):
        ir2 = _ir(tmp_path, "r" + supply[:2])
        state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **I2C, "input_voltage": supply})).run(ir2, stop_after=Stage.ARCHITECTURE)
        assert "outside 2.5..5.5 V" in state.outcome(Stage.ARCHITECTURE).message


@needs_libs
def test_confirmation_applies_the_calculators_and_the_elements_with_provenance_and_two_builds_are_identical(tmp_path: Path):
    hashes = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        ir = _ir(d, "bus")
        _present(ir, d, I2C)
        state, _ = _confirm(ir, d, Stage.CALCULATION)
        assert not state.blocked and [c.ref for c in ir.components] == ["R1", "R2", "J1"]
        r_max, r, v_low, t_30, t_check = (ir.parameters[k] for k in ("r_max", "r_pullup_ideal", "v_low", "t_30", "t_check"))
        placed = ir.parameters["r_pullup"]
        assert placed.value == 1800.0 and placed.provenance.tool == "calc.e_series.e24.nearest" and placed.provenance.inputs == {"x": "r_pullup_ideal"}
        assert {c.ref: c.value for c in ir.components} == {"R1": "1.8k", "R2": "1.8k", "J1": "Conn_01x03"} and ir.component("R1").spice.value.value == 1800.0
        assert v_low.provenance.inputs == {"v_dd": "v_dd", "r": "r_pullup"} and t_30.provenance.inputs["r"] == "r_pullup"  # every later number uses the placed value
        assert r_max.provenance.tool == "calc.i2c.r_max" and r_max.provenance.inputs == {"t_r": "t_r", "c_b": "c_b"} and r_max.value == pytest.approx(2360.2, rel=1e-3)
        assert r.provenance.tool == "calc.i2c.r_pullup" and r.provenance.inputs == {"r_max": "r_max", "r_min": "r_min", "fraction": "pullup_fraction"} and r.value == pytest.approx(0.8 * r_max.value) and r.unit == "ohm"
        assert v_low.provenance.tool == "calc.i2c.v_low_diode" and 0.6 < v_low.value < 0.7
        assert t_30.provenance.tool == "calc.i2c.t_30" and t_30.provenance.inputs["v_low"] == "v_low" and t_check.provenance.inputs == {"t_low": "t_low", "t_30": "t_30", "t_r": "t_r"} and t_check.value == pytest.approx(1.25e-6 + t_30.value + 300e-9)
        sim = ir.simulation
        assert sim.nodes == ["DRV"] and [e.id for e in sim.elements] == ["CBUS", "DOD"]
        cbus, dod = sim.elements
        assert cbus.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT and cbus.value.value == pytest.approx(150e-12) and "req.bus_capacitance" in cbus.serves_requirements
        assert dod.model_card.provenance.kind is ProvenanceKind.USER_REQUIREMENT and dod.model_card.provenance.note.startswith(CHOICE_NOTE_PREFIX)
        held, exp = sim.expectations
        assert held.id == "held_low" and held.nominal.provenance.tool == "calc.i2c.v_low_diode" and held.at.provenance.tool == "calc.i2c.t_hold"
        assert exp.id == "rise" and exp.bound == "at_least" and exp.requirement_id is None and exp.nominal.provenance.tool == "calc.i2c.v_ih" and exp.at.provenance.tool == "calc.i2c.t_check"
        assert ir.validation.latest("calc.recompute").status is S.PASS
        text = build(ir)
        assert "CBUS SDA 0 150p" in text and "DOD SDA DRV DOD" in text and "VDRV DRV 0 PULSE(" in text and ".model DOD D" in text
        hashes.append(ir.content_hash())
    assert hashes[0] == hashes[1]


@needs_dll
def test_the_held_level_and_the_rise_are_verified_by_ngspice_and_a_pull_up_above_r_max_fails(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, I2C)
    state, _ = _confirm(ir, tmp_path, Stage.SPICE, spice=True)
    held, r = ir.validation.latest("spice.held_low"), ir.validation.latest("spice.rise")
    assert state.outcome(Stage.SPICE).status is S.PASS and held.status is S.PASS and r.status is S.PASS, (held.message, r.message)
    assert held.details["measured"] == pytest.approx(ir.parameters["v_low"].value, abs=0.02)  # the computed diode drop is what ngspice holds the line at
    assert r.details["measured"] >= 2.31 and r.details["measured"] < 3.3
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    ir2 = _ir(wrong, "wrong")
    _present(ir2, wrong, I2C)
    _confirm(ir2, wrong, Stage.CALCULATION)
    for ref in ("R1", "R2"):
        part = ir2.component(ref)
        big = user_requirement(1.3 * part.spice.value.value, "ohm", note="a wrong part: 1.04 x R_max")
        part.spice.value = big
        part.electrical["resistance"] = big
    _confirm(ir2, wrong, Stage.SPICE, spice=True)
    r2 = ir2.validation.latest("spice.rise")
    assert r2 is not None and r2.status is S.FAIL, (r2.status, r2.message)


@needs_kicad
def test_the_schematic_has_the_parts_only_and_erc_is_clean(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, I2C)
    _confirm(ir, tmp_path, Stage.CALCULATION)
    ref = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _lib}))
    text = Path(ref.path).read_text(encoding="utf-8")
    assert "CBUS" not in text and "DOD" not in text and "DRV" not in text  # simulation-only: not on the schematic
    result = kicad.run_erc(Path(ref.path), tmp_path / "erc.json")
    assert result.status is S.PASS and result.details["errors"] == [], result.details["errors"]
