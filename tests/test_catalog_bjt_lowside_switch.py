"""The ``bjt_lowside_switch`` catalog template (PLAN D9a): table, refusals, confirmation provenance, determinism, the
saturated ON drop and the flyback clamp verified by ngspice (and under-drive shown to FAIL), a clean ERC on the schematic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.design import templates as templates_mod
from ai_eda.design.base import CHOICE_NOTE_PREFIX
from ai_eda.design.catalog.bjt_lowside_switch import BjtLowsideSwitchTemplate
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, user_requirement
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import Orchestrator, Stage

_lib = KicadLibrary()
HAS_LIBS = all(_lib.symbol_file(name) is not None for name in ("Device", "Connector_Generic", "Transistor_BJT", "Diode"))
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not (HAS_LIBS and runner.available()), reason="ngspice shared library / KiCad libraries not found")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (HAS_LIBS and kicad.available()), reason="kicad-cli / KiCad libraries not installed")

BASE = {"application": "relay driver", "jurisdiction": "EU"}
SWITCH = {"input_voltage": "5 V", "output_voltage": "12 V", "output_current": "50 mA", "max_voltage_drop": "0.3 V"}


def test_the_template_is_registered_once():
    assert [t.id for t in templates_mod.all_templates()].count("bjt_lowside_switch") == 1 and any(isinstance(t, BjtLowsideSwitchTemplate) for t in templates_mod.TEMPLATES)


def _ir(tmp_path: Path, name: str = "sw") -> CircuitIR:
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
def test_the_table_names_the_inputs_choices_computed_values_parts_nets_and_bounds(tmp_path: Path):
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, SWITCH)  # the divider (input_voltage + output_voltage) stands back: not ambiguous
    assert "bjt_lowside_switch" in question and "ambiguous" not in question
    for text in ("max_voltage_drop", "v_be", "beta_forced", "switch_models", "R1 Device:R", "Q1 Transistor_BJT:2N3904", "D1 Diode:1N4001 / Diode_THT:D_DO-41",
                 "SW (signal): Q1.", "VLOAD (power): J2.1, D1.", "expectation on_drop: ON-state level = mean of v(SW) over 0.002..0.003 s at most 0.3 V",
                 "expectation off_clamp: max of v(SW) at most 13 V"):
        assert text in question, text
    # I_B = 50 mA / 10 = 5 mA, R_B = (5 - 0.7) V / 5 mA = 860 ohm; 860 is no part, so R1 is the largest E24 value not above it (820: more base drive, never less)
    assert "i_b = 0.005" in question and "r_b_ideal = 860 ohm [calc.bjt_switch.base_resistor from v_ctl, v_be, i_b]" in question
    assert "r_b = 820 ohm [calc.e_series.e24.at_most from r_b_ideal]" in question


@needs_libs
def test_out_of_range_inputs_are_refused_with_the_reason_named(tmp_path: Path):
    for answers, text in (({**SWITCH, "output_current": "500 mA"}, "outside 0..0.2 A"), ({**SWITCH, "input_voltage": "1 V"}, "below V_BE + 0.5 V"),
                          ({**SWITCH, "output_voltage": "39.5 V"}, "V_load + v_f = 39.5 V + 1 V = 40.5 V")):
        ir = _ir(tmp_path, "r")
        state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
        out = state.outcome(Stage.ARCHITECTURE)
        assert ir.components == [] and "template bjt_lowside_switch not proposed" in out.message and text in out.message, out.message


@needs_libs
def test_confirmation_applies_the_calculators_with_provenance_and_two_builds_are_identical(tmp_path: Path):
    hashes = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        ir = _ir(d, "sw")
        _present(ir, d, SWITCH)
        state, _ = _confirm(ir, d, Stage.CALCULATION)
        assert not state.blocked and [c.ref for c in ir.components] == ["R1", "Q1", "D1", "J1", "J2"]
        i_b, r_b, clamp, beta = (ir.parameters[k] for k in ("i_b", "r_b_ideal", "clamp_max", "beta_forced"))
        placed = ir.parameters["r_b"]
        assert placed.value == 820.0 and placed.unit == "ohm" and placed.provenance.tool == "calc.e_series.e24.at_most" and placed.provenance.inputs == {"x": "r_b_ideal"}
        assert ir.component("R1").value == "820" and ir.component("R1").spice.value.value == 820.0
        assert i_b.provenance.tool == "calc.bjt_switch.base_current" and i_b.provenance.inputs == {"i_c": "i_load", "beta_forced": "beta_forced"} and i_b.value == pytest.approx(5e-3)
        assert r_b.provenance.tool == "calc.bjt_switch.base_resistor" and r_b.provenance.inputs == {"v_ctl": "v_ctl", "v_be": "v_be", "i_b": "i_b"} and r_b.value == pytest.approx(860.0) and r_b.unit == "ohm"
        assert clamp.provenance.tool == "calc.bjt_switch.clamp_max" and clamp.value == pytest.approx(13.0)
        assert beta.provenance.kind is ProvenanceKind.USER_REQUIREMENT and beta.provenance.note.startswith(CHOICE_NOTE_PREFIX)
        on, off = ir.simulation.expectations
        assert on.id == "on_drop" and on.bound == "at_most" and on.requirement_id is None and on.nominal.value == pytest.approx(0.3)
        assert on.nominal.provenance.kind is ProvenanceKind.USER_REQUIREMENT  # the requirement's own value, not a typed number
        assert off.id == "off_clamp" and off.bound == "at_most" and off.nominal.provenance.tool == "calc.bjt_switch.clamp_max"
        assert ir.component("Q1").spice.pin_order == ["3", "2", "1"] and ir.component("D1").spice.pin_order == ["2", "1"]  # C B E / A K by pin name
        assert "req.max_voltage_drop" in ir.net("SW").serves_requirements
        assert ir.validation.latest("calc.recompute").status is S.PASS
        hashes.append(ir.content_hash())
    assert hashes[0] == hashes[1]


@needs_dll
def test_the_saturated_drop_and_the_clamp_are_verified_by_ngspice_and_under_drive_fails(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, SWITCH)
    state, _ = _confirm(ir, tmp_path, Stage.SPICE, spice=True)
    on, off = ir.validation.latest("spice.on_drop"), ir.validation.latest("spice.off_clamp")
    assert state.outcome(Stage.SPICE).status is S.PASS and on.status is S.PASS and off.status is S.PASS, (on.message, off.message)
    assert 0.0 < on.details["measured"] < 0.3 and 12.0 < off.details["measured"] <= 13.0
    # not an identity: a base resistor 100 x the calculators' value (as a user's own part) cannot saturate the switch, and on_drop fails
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    ir2 = _ir(wrong, "wrong")
    _present(ir2, wrong, SWITCH)
    _confirm(ir2, wrong, Stage.CALCULATION)
    r1 = ir2.component("R1")
    big = user_requirement(100.0 * r1.spice.value.value, "ohm", note="a wrong part")
    r1.spice.value = big
    r1.electrical["resistance"] = big
    _confirm(ir2, wrong, Stage.SPICE, spice=True)
    r2 = ir2.validation.latest("spice.on_drop")
    assert r2 is not None and r2.status is S.FAIL, (r2.status, r2.message)


@needs_kicad
def test_the_schematic_compiles_with_wires_and_erc_is_clean(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, SWITCH)
    _confirm(ir, tmp_path, Stage.CALCULATION)
    # the "wiring:" note is the legacy table's: checked with layout="legacy" since textbook 1.0 became the default, which
    # is checked too (drawn with no whole-sheet fallback, ERC clean)
    ref = SchematicCompiler("legacy").compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _lib}))
    assert ref.notes and ref.notes[0].startswith("wiring: "), ref.notes
    result = kicad.run_erc(Path(ref.path), tmp_path / "erc.json")
    assert result.status is S.PASS and result.details["errors"] == [], result.details["errors"]
    assert not [w for w in result.details["warnings"] if w.get("type") in ("lib_symbol_mismatch", "unconnected_wire_endpoint", "endpoint_off_grid", "isolated_pin_label", "global_label_dangling")]
    ref = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "textbook", tools={"kicad_library": _lib}))
    assert ref.generator_version == "1.0" and "fallbacks: none" in ref.notes[0], ref.notes
    result = kicad.run_erc(Path(ref.path), tmp_path / "textbook" / "erc.json")
    assert result.status is S.PASS and result.details["errors"] == [], result.details["errors"]
    assert not [w for w in result.details["warnings"] if w.get("type") in ("lib_symbol_mismatch", "unconnected_wire_endpoint", "endpoint_off_grid", "isolated_pin_label", "global_label_dangling")]
