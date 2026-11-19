"""The ``lc_lowpass`` catalog template (PLAN D9a): table, confirmation provenance, calculators, determinism, refusals,
the corner verified by ngspice (and shown not to be an identity), and a clean ERC on the compiled schematic.

The template is registered in ``TEMPLATES`` (README / GUI counts updated with it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.design import templates as templates_mod
from ai_eda.design.base import CHOICE_NOTE_PREFIX
from ai_eda.design.catalog.lc_lowpass import LcLowpassTemplate
from ai_eda.design.inputs import read_inputs
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, Traced, user_requirement
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc import recompute_parameters
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

BASE = {"application": "bench filter", "jurisdiction": "EU"}
LC = {"cutoff_frequency": "100 kHz", "system_impedance": "50 ohm"}
TEMPLATE = LcLowpassTemplate()


def test_the_template_is_registered_once():
    assert [t.id for t in templates_mod.all_templates()].count("lc_lowpass") == 1 and any(isinstance(t, LcLowpassTemplate) for t in templates_mod.TEMPLATES)


def _ir(tmp_path: Path, name: str = "lc") -> CircuitIR:
    return CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))


def _present(ir: CircuitIR, tmp_path: Path, answers: dict[str, str]):
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"] and ir.components == []
    return state.open_questions[0].question


def _confirm(ir: CircuitIR, tmp_path: Path, stop_after: Stage, *, spice: bool = False):
    tools = {"kicad_library": _lib, **({"spice": runner} if spice else {})}
    ctx = AgentContext(workdir=tmp_path, tools=tools, answers={"confirm_design": "yes", "pcb.placement": "skip"})
    return Orchestrator(ctx).run(ir, stop_after=stop_after), ctx


@needs_libs
def test_the_table_names_every_input_choice_computed_value_part_net_and_simulation_line(tmp_path: Path):
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, LC)
    assert "lc_lowpass" in question
    for text in ("cutoff_frequency", "system_impedance", "tol_db", "ac_probe", "L1 Device:L / Inductor_SMD:L_0805_2012Metric", "R1 Device:R", "R2 Device:R",
                 "OUT (signal): L1.2, C1.1, R2.1, J2.1", "expectation level_fc: v(OUT) at 100000 Hz = -3.01029995664 dB relative to v_pass = 0.5 V"):
        assert text in question, text
    # L = sqrt(2) Z0 / (2 pi f_c) and C = sqrt(2) / (2 pi f_c Z0) for 100 kHz / 50 ohm
    assert "l = 0.00011254" in question and "c = 4.5016e-08" in question or "112.54u" in question


@needs_libs
def test_out_of_range_inputs_are_refused_with_the_bounds_named(tmp_path: Path):
    ir = _ir(tmp_path)
    for answers, text in (({**LC, "cutoff_frequency": "100 Hz"}, "outside 1000..30000000 Hz"), ({**LC, "system_impedance": "2 ohm"}, "outside 10..600 ohm")):
        ir = _ir(tmp_path, "r")
        state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
        out = state.outcome(Stage.ARCHITECTURE)
        assert ir.components == [] and "template lc_lowpass not proposed" in out.message and text in out.message, out.message


@needs_libs
def test_confirmation_applies_the_calculators_with_provenance_and_two_builds_are_identical(tmp_path: Path):
    hashes = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        ir = _ir(d, "lc")  # the same project id in two workdirs: the content hash covers the design, not the folder
        _present(ir, d, LC)
        state, _ = _confirm(ir, d, Stage.CALCULATION)
        assert not state.blocked and [c.ref for c in ir.components] == ["R1", "L1", "C1", "R2", "J1", "J2"]
        l, c, level, v_pass, tol = (ir.parameters[k] for k in ("l", "c", "level_fc", "v_pass", "tol_db"))
        assert l.provenance.tool == "calc.lc.butterworth2.series_l" and l.provenance.inputs == {"f_c": "f_c", "z0": "z0"} and l.unit == "H"
        assert c.provenance.tool == "calc.lc.butterworth2.shunt_c" and c.value == pytest.approx(4.5016e-8, rel=1e-4)
        assert level.provenance.tool == "calc.lc.butterworth2.level_db" and level.value == pytest.approx(-3.0103, abs=1e-4) and level.unit == "dB"
        assert v_pass.provenance.tool == "calc.divider.v_out" and v_pass.value == pytest.approx(0.5)
        assert tol.provenance.kind is ProvenanceKind.USER_REQUIREMENT and tol.provenance.note.startswith(CHOICE_NOTE_PREFIX)
        exp = ir.simulation.expectations[0]
        assert exp.id == "level_fc" and exp.requirement_id == "req.cutoff_frequency" and exp.nominal.provenance.tool == "calc.lc.butterworth2.level_db"
        assert exp.params["ref"].provenance.tool == "calc.divider.v_out"  # the passband level the dB is relative to, not a typed number
        assert ir.validation.latest("calc.recompute").status is S.PASS
        hashes.append(ir.content_hash())
    assert hashes[0] == hashes[1]


@needs_dll
def test_the_corner_is_verified_by_ngspice_and_a_wrong_inductor_fails_it(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, LC)
    state, _ = _confirm(ir, tmp_path, Stage.SPICE, spice=True)
    r = ir.validation.latest("spice.level_fc")
    assert state.outcome(Stage.SPICE).status is S.PASS and r.status is S.PASS, r.message
    assert r.details["measured"] == pytest.approx(-3.0103, abs=0.05) and r.details["nominal"] == pytest.approx(-3.0103, abs=1e-3)
    # not an identity: double the inductor the calculators chose (as a user's own value) and the same expectation no longer passes
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    ir2 = _ir(wrong, "wrong")
    _present(ir2, wrong, LC)
    _confirm(ir2, wrong, Stage.CALCULATION)
    l1 = ir2.component("L1")
    doubled = user_requirement(2.0 * l1.spice.value.value, "H", note="a wrong part")
    l1.spice.value = doubled
    l1.electrical["inductance"] = doubled
    state2, _ = _confirm(ir2, wrong, Stage.SPICE, spice=True)
    r2 = ir2.validation.latest("spice.level_fc")
    assert r2 is not None and r2.status is not S.PASS, (r2.status, r2.message)


@needs_kicad
def test_the_schematic_compiles_with_wires_and_erc_is_clean(tmp_path: Path):
    ir = _ir(tmp_path)
    _present(ir, tmp_path, LC)
    _confirm(ir, tmp_path, Stage.CALCULATION)
    # the "wiring:" note is the legacy table's: checked with layout="legacy" since textbook 1.0 became the default, which
    # is checked too (drawn with no whole-sheet fallback, ERC clean)
    ref = SchematicCompiler("legacy").compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _lib}))
    assert ref.notes and ref.notes[0].startswith("wiring: 4 of 4 net(s)") or ref.notes[0].startswith("wiring: 3 of 4"), ref.notes
    result = kicad.run_erc(Path(ref.path), tmp_path / "erc.json")
    assert result.status is S.PASS and result.details["errors"] == [], result.details["errors"]
    assert not [w for w in result.details["warnings"] if w.get("type") in ("lib_symbol_mismatch", "unconnected_wire_endpoint", "endpoint_off_grid", "isolated_pin_label", "global_label_dangling")]
    ref = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "textbook", tools={"kicad_library": _lib}))
    assert ref.generator_version == "1.0" and "fallbacks: none" in ref.notes[0], ref.notes
    result = kicad.run_erc(Path(ref.path), tmp_path / "textbook" / "erc.json")
    assert result.status is S.PASS and result.details["errors"] == [], result.details["errors"]
    assert not [w for w in result.details["warnings"] if w.get("type") in ("lib_symbol_mismatch", "unconnected_wire_endpoint", "endpoint_off_grid", "isolated_pin_label", "global_label_dangling")]
