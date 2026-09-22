"""Regression tests for the verifier findings on the SPICE stage (one test group per finding).

Tests that need KiCad's bundled ``ngspice.dll`` are marked ``needs_dll`` and
otherwise run against the real DLL; everything else runs everywhere.
"""

from __future__ import annotations

import io
import json
import math
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, SimulationAgent
from ai_eda.cli import main as cli_main, run_exit_code
from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import build, build_report, spice_vector_name
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    ArtifactKind,
    CircuitIR,
    Component,
    Expectation,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    Requirement,
    RequirementKind,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    ValidationResult,
    ValidationStatus,
    assumption,
    authoritative,
    derived,
    llm_generated,
    user_requirement,
)
from ai_eda.ir.project import strip_wall_clock
from ai_eda.repair import RepairLoop
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.calc import (
    current_from_voltage_resistance,
    ngspice_reads,
    rc_lowpass_magnitude,
    rc_lowpass_phase_deg,
    rc_time_constant,
    recompute_parameters,
    voltage_divider_output,
    voltage_divider_ratio,
)
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice.runner import Interpolation
from ai_eda.tools.spice.stage import NGSPICE_DEFAULT_TEMP_C, judge, read_results, reduce_expectation, run_spice_for
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow import Orchestrator, PipelineState, Stage, StageOutcome
from tests.conftest import AUTH, DS
from tests.fixtures_kicad import RESISTOR_DS, USER, divider_with_connector_ir
from tests.fixtures_rc import rc_lowpass_ir
from tests.ngspice_parser_harness import measure_spellings, sample_spellings

S = ValidationStatus
LIB = KicadLibrary()
runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice.dll (KiCad's bundled ngspice shared library) not found")
ANSWERS = {"application": "bench", "jurisdiction": "EU"}


def _context(tmp_path: Path) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": runner}, answers=dict(ANSWERS))


def _spice_stage(ir: CircuitIR, ctx: AgentContext) -> PipelineState:
    return Orchestrator(ctx).run(ir, stop_after=Stage.SPICE)


def _review(ir: CircuitIR, ctx: AgentContext) -> dict[str, ValidationResult]:
    return {r.check_id: r for r in IndependentReviewer(tools=ctx.tools).review(ir, ctx.workdir).results}


def _compile_and_run(ir: CircuitIR, ctx: AgentContext) -> list[ValidationResult]:
    """The SPICE stage without the orchestrator (no IR_BUILD gate): compile, then run_spice_for."""
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=ctx.workdir, tools=ctx.tools))
    results = run_spice_for(ir, ctx.tools, ctx.workdir)
    ir.validation.extend(results)
    return results


def _change_r1(ir: CircuitIR, ohms: float, label: str) -> None:
    r1 = ir.component("R1")
    r1.value = label
    r1.electrical["resistance"] = authoritative(ohms, RESISTOR_DS, "ohm")
    r1.spice.value = r1.electrical["resistance"]
    ir.parameters["r1"] = r1.electrical["resistance"]


def _recalculate_nominals(ir: CircuitIR) -> None:
    p = ir.parameters
    p["v_out"] = voltage_divider_output(p["v_in"], p["r1"], p["r2"], ("v_in", "r1", "r2"))
    p["v_out_mid"] = voltage_divider_output(p["v_in_mid"], p["r1"], p["r2"], ("v_in_mid", "r1", "r2"))
    for exp in ir.simulation.expectations:
        exp.nominal = p["v_out"] if exp.id == "v_out" else p["v_out_mid"]


# --------------------------------------------------------------------------- 1. content_hash is a design hash


def test_content_hash_is_the_same_for_the_same_design_built_twice(tmp_path: Path):
    a = divider_with_connector_ir(tmp_path / "a", LIB)
    time.sleep(0.01)
    b = divider_with_connector_ir(tmp_path / "b", LIB)  # later, in another workdir
    assert a.project.created_at != b.project.created_at and a.project.workdir != b.project.workdir
    assert a.content_hash() == b.content_hash()
    assert "created_at" not in json.dumps(a.design_dict()) and "workdir" not in a.design_dict()["project"]
    loaded = CircuitIR.load(a.save(tmp_path / "a" / "ir.json"))
    assert loaded.content_hash() == a.content_hash()
    b.component("R1").spice.value = authoritative(10_001.0, RESISTOR_DS, "ohm")
    b.component("R1").electrical["resistance"] = b.component("R1").spice.value
    assert b.content_hash() != a.content_hash()  # a design change still changes the hash
    assert strip_wall_clock({"x": {"created_at": 1, "y": [{"created_at": 2, "z": 3}]}}) == {"x": {"y": [{"z": 3}]}}


# --------------------------------------------------------------------------- 2. ai-eda run exit code


def test_run_exit_code_is_nonzero_when_the_pipeline_ends_in_fail(tmp_path: Path):
    ok = PipelineState(outcomes=[StageOutcome(stage=Stage.RELEASE, status=S.NOT_VERIFIED)])
    assert run_exit_code(ok) == 0  # nothing wrong, nothing proven
    assert run_exit_code(PipelineState(outcomes=[StageOutcome(stage=Stage.RELEASE, status=S.FAIL)])) == 1
    assert run_exit_code(PipelineState(outcomes=[StageOutcome(stage=Stage.MISSING_INFORMATION, status=S.USER_INPUT_REQUIRED)], blocked=True)) == 1
    # end to end through the CLI: a stored derived parameter the calculator does not reproduce -> release FAIL -> exit 1
    ir = CircuitIR(project=ProjectMeta(id="bad_calc", name="bad calc", workdir=str(tmp_path / "bad")))
    ir.parameters["v_in"] = user_requirement(12.0, "V")
    ir.parameters["r1"] = authoritative(1e4, DS, "ohm")
    ir.parameters["r2"] = authoritative(1e4, DS, "ohm")
    ir.parameters["v_out"] = derived(5.0, tool="calc.divider.v_out", inputs={"v_in": "v_in", "r1": "r1", "r2": "r2"}, unit="V")
    path = ir.save(tmp_path / "bad" / "ir.json")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["run", str(path), "--answer", "application=bench", "--answer", "jurisdiction=EU"])
    assert code == 1, out.getvalue()
    assert "release" in out.getvalue() and "FAIL" in out.getvalue().split("release")[1]
    # ... and an empty design that is merely unverified still exits 0
    empty = CircuitIR(project=ProjectMeta(id="empty", name="empty", workdir=str(tmp_path / "empty"))).save(tmp_path / "empty" / "ir.json")
    with redirect_stdout(io.StringIO()):
        assert cli_main(["run", str(empty), "--answer", "application=bench", "--answer", "jurisdiction=EU"]) == 0


# --------------------------------------------------------------------------- 3. SpiceBinding.value vs Component.electrical


def test_spice_value_that_disagrees_with_the_electrical_value_is_refused(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    assert build_report(ir)["value_sources"] == {"R1": "electrical.resistance", "R2": "electrical.resistance"}
    r1 = ir.component("R1")
    r1.value = "20k"
    r1.electrical["resistance"] = authoritative(20_000.0, RESISTOR_DS, "ohm")  # the BOM ships 20k, the binding still says 10k
    with pytest.raises(CompileError, match=r"R1: SPICE value 10000.0 differs from electrical\['resistance'\] = 20000.0"):
        build(ir)
    r1.spice.value = authoritative(20_000.0, RESISTOR_DS, "kohm")  # same number, another unit: not comparable either
    with pytest.raises(CompileError, match="unit 'kohm' differs from"):
        build(ir)
    r1.spice.value = r1.electrical["resistance"]
    assert "R1 VIN VOUT 20k" in build(ir)
    # a binding without a matching electrical entry is reported as such, never silently trusted as "the part"
    del r1.electrical["resistance"]
    assert build_report(ir)["value_sources"]["R1"] == "spice binding only (component records no resistance)"


# --------------------------------------------------------------------------- 4 / 14. review.calculations_vs_design


def test_review_calculations_vs_design_recomputes_instead_of_trusting_provenance_shape(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    assert _review(ir, ctx)[ReviewArea.CALCULATIONS_VS_DESIGN].status is S.PASS
    ir.parameters["v_out"].value = 5.0  # provenance shape intact, number wrong
    r = _review(ir, ctx)[ReviewArea.CALCULATIONS_VS_DESIGN]
    assert r.status is S.FAIL and r.details["repair"] == "human"
    assert any(m.startswith("v_out: stored 5.0, calc.divider.v_out recomputes 6.0") for m in r.details["recompute"]["mismatches"])
    ir.parameters["v_out"].value = 6.0
    ir.parameters["v_out"].provenance.tool = "calc.mystery"
    r = _review(ir, ctx)[ReviewArea.CALCULATIONS_VS_DESIGN]
    assert r.status is S.NOT_VERIFIED and "not a registered calculator" in r.message
    # a stored stage result that contradicts the reviewer's own recompute is a finding of its own
    ir.parameters["v_out"].provenance.tool = "calc.divider.v_out"
    stale = recompute_parameters(ir)
    stale.status, stale.ir_hash = S.FAIL, ir.content_hash()
    ir.validation.add(stale)
    r = _review(ir, ctx)[ReviewArea.CALCULATIONS_VS_DESIGN]
    assert r.status is S.FAIL and "stored calc.recompute result says FAIL" in r.message and r.details["stage_result"]["current"] is True


# --------------------------------------------------------------------------- 5. AT between samples


def _interp(y0: float, y1: float, value: float) -> Interpolation:
    return Interpolation(value=value, exact=False, x0=1.0, y0=y0, x1=2.0, y1=y1, method="linear")


def test_judge_judges_the_whole_bracket_of_an_interpolation():
    exp = Expectation(id="e", analysis_id="a", vector="v(X)", reduce=Reduce.AT, at=user_requirement(1.5), nominal=user_requirement(10.0), tol_rel=user_requirement(0.01), provenance=USER)
    assert judge(10.0, exp, _interp(9.95, 10.05, 10.0))[0] is S.PASS  # both neighbours inside the tolerance
    assert judge(10.0, exp, _interp(9.0, 11.0, 10.0))[0] is S.UNRESOLVED  # interpolated value fits, the grid does not resolve 1 %
    assert judge(10.5, exp, _interp(9.0, 12.0, 10.5))[0] is S.UNRESOLVED  # nominal within reach of the bracket
    assert judge(12.5, exp, _interp(12.0, 13.0, 12.5))[0] is S.FAIL  # the whole bracket is outside on one side
    assert judge(10.0, exp, Interpolation(value=10.0, exact=True, x0=1.5, y0=10.0, x1=1.5, y1=10.0, method="exact"))[0] is S.PASS
    assert judge(10.0, exp) == (S.PASS, pytest.approx(0.1), 0.0)


@needs_dll
def test_ac_interpolation_is_log_aware_records_the_bracket_and_never_fails_on_grid_coarseness(tmp_path: Path):
    deck = tmp_path / "rc_ac.cir"
    deck.write_text("rc ac\nV1 IN 0 DC 0 AC 1\nR1 IN OUT 1k\nC1 OUT 0 1u\n.end\n", encoding="utf-8", newline="\n")
    res = runner.run(deck, SpiceAnalysis.AC, tmp_path, "ac dec 10 1 1meg")
    assert res.succeeded, res.errors
    fc = 1 / (2 * math.pi * 1e3 * 1e-6)

    def analytic(f: float) -> float:
        return 1 / math.sqrt(1 + (f / fc) ** 2)

    freqs = res.vectors["frequency"]
    worst = 0.0
    for f0, f1 in zip(freqs, freqs[1:]):
        f = math.sqrt(f0 * f1)
        i = res.interpolate("out", f)
        assert i.method == "log-log" and not i.exact and i.x0 == f0 and i.x1 == f1
        worst = max(worst, abs(i.value - analytic(f)) / analytic(f))
    assert worst < 0.004  # measured 0.33 % worst case (plain linear: 1.33 %)
    assert res.interpolate("out.phase_deg", fc).method == "log-x"
    # an expectation at fc with a 0.5 % tolerance: the grid (ratio 1.26) cannot resolve that -> UNRESOLVED with the bracket, not FAIL
    exp = Expectation(id="hc", analysis_id="ac", vector="v(OUT)", reduce=Reduce.AT, at=user_requirement(fc, "Hz"), nominal=user_requirement(analytic(fc)), tol_rel=user_requirement(0.005), provenance=USER)
    red = reduce_expectation(res, exp, "out")
    assert red.problem is None and red.interpolation is not None and red.interpolation.y0 > analytic(fc) > red.interpolation.y1
    status, limit, deviation = judge(red.measured, exp, red.interpolation)
    assert status is S.UNRESOLVED and deviation < 0.004 * analytic(fc)  # the number is within 0.4 %, the verdict is honest about the grid


# --------------------------------------------------------------------------- 6. orphan spice.<id> results


@needs_dll
def test_removed_expectation_no_longer_blocks_the_release(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].nominal = user_requirement(5.0, "V", note="wrong on purpose")
    ctx = _context(tmp_path)
    _spice_stage(ir, ctx)
    assert ir.validation.latest("spice.v_out").status is S.FAIL and ir.validation.overall() is S.FAIL
    del ir.simulation.expectations[0]  # the user drops the wrong expectation
    _spice_stage(ir, ctx)
    retired = ir.validation.latest("spice.v_out")
    assert retired.status is S.NOT_APPLICABLE and retired.details["superseded"] == "FAIL" and "no longer in the simulation setup" in retired.message
    assert ir.validation.latest("spice").status is S.PASS and ir.validation.latest("spice.v_out_mid").status is S.PASS
    assert "spice.v_out" not in {r.check_id for r in ir.validation.failing()}
    # the same when the whole setup goes away: the agent supersedes every spice.<id> with NOT_VERIFIED
    ir.simulation = None
    res = SimulationAgent().run(ir, ctx)
    assert {r.check_id: r.status for r in res.validation} == {"spice": S.NOT_VERIFIED, "spice.v_out_mid": S.NOT_VERIFIED}


# --------------------------------------------------------------------------- 7. roles instead of positional derived_from


def test_role_swaps_are_named_not_computed_the_wrong_way_round(divider_ir: CircuitIR):
    ir = divider_ir
    ir.parameters["v"] = user_requirement(12.0, "V")
    ir.parameters["r"] = authoritative(1e3, DS, "ohm")
    # scenario A: the caller labels the ids the wrong way round; the calculator records exactly that,
    # and the recompute refuses to feed a resistance into the voltage role
    ir.parameters["i"] = current_from_voltage_resistance(ir.parameters["v"], ir.parameters["r"], ("r", "v"))
    assert ir.parameters["i"].value == pytest.approx(0.012) and ir.parameters["i"].provenance.inputs == {"v": "r", "r": "v"}
    res = recompute_parameters(ir)
    entry = res.details["parameters"]["i"]
    assert entry["status"] == "NOT_VERIFIED" and "unit does not fit its role" in entry["reason"] and "v='r' carries unit 'ohm'" in entry["reason"]
    assert res.status is S.NOT_VERIFIED and res.details["mismatches"] == []
    del ir.parameters["i"]
    # scenario B: a symmetric slip cannot be told apart by units; the recorded chain is false and the recompute says so
    ir.parameters["ra"] = authoritative(1e3, DS, "ohm")
    ir.parameters["rb"] = authoritative(3e3, DS, "ohm")
    ir.parameters["ratio"] = voltage_divider_ratio(ir.parameters["ra"], ir.parameters["rb"], ("rb", "ra"))
    res = recompute_parameters(ir)
    assert res.status is S.FAIL
    assert res.details["mismatches"] == ["ratio: stored 0.75, calc.divider.ratio recomputes 0.25 from {'rb': 3000.0, 'ra': 1000.0}"]
    assert res.details["parameters"]["ratio"]["roles"] == {"r1": "rb", "r2": "ra"}


# --------------------------------------------------------------------------- 8. derived values outside ir.parameters


def test_stale_expectation_nominal_is_recomputed_under_its_path(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    res = recompute_parameters(ir)
    assert res.status is S.PASS and set(res.details["parameters"]) == {"v_out", "v_out_mid", "simulation.expectations[v_out].nominal", "simulation.expectations[v_out_mid].nominal"}
    ir = CircuitIR.load(ir.save(tmp_path / "ir.json"))  # after a round trip the nominal is an independent copy
    _change_r1(ir, 20_000.0, "20k")
    ir.parameters["v_out"] = voltage_divider_output(ir.parameters["v_in"], ir.parameters["r1"], ir.parameters["r2"], ("v_in", "r1", "r2"))
    ir.parameters["v_out_mid"] = voltage_divider_output(ir.parameters["v_in_mid"], ir.parameters["r1"], ir.parameters["r2"], ("v_in_mid", "r1", "r2"))
    res = recompute_parameters(ir)
    assert res.status is S.FAIL
    assert res.details["parameters"]["v_out"]["status"] == "PASS"
    assert res.details["parameters"]["simulation.expectations[v_out].nominal"]["status"] == "FAIL"
    assert any(m.startswith("simulation.expectations[v_out].nominal: stored 6.0, calc.divider.v_out recomputes 4.0") for m in res.details["mismatches"])
    # every other Traced a design number can hide in is walked too
    ir.component("R2").spice.params["m"] = rc_time_constant(ir.parameters["r1"], authoritative(1e-6, DS, "F"), ("r1", "c_missing"))
    ir.simulation.temperature_c = derived(25.0, tool="calc.mystery", inputs={"x": "v_in"}, unit="degC")
    res = recompute_parameters(ir)
    assert "components[R2].spice.params[m]" in res.details["parameters"] and "simulation.temperature_c" in res.details["parameters"]
    assert "['c_missing'] not found" in res.details["parameters"]["components[R2].spice.params[m]"]["reason"]


# --------------------------------------------------------------------------- 9. zero nominal with tol_rel only


def test_zero_nominal_with_only_a_relative_tolerance_is_unresolved_and_refused_at_compile_time(tmp_path: Path):
    exp = Expectation(id="z", analysis_id="op", vector="v(X)", nominal=user_requirement(0.0, "V"), tol_rel=user_requirement(0.01), provenance=USER)
    assert judge(1e-12, exp) == (S.UNRESOLVED, None, 1e-12)  # not FAIL: the tolerance is undefined, not violated
    exp.tol_abs = user_requirement(1e-6, "V")
    assert judge(1e-12, exp)[0] is S.PASS
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].nominal = user_requirement(0.0, "V")
    with pytest.raises(CompileError, match="expectation v_out: nominal is 0 and only tol_rel is given"):
        build(ir)
    ir.simulation.expectations[0].tol_abs = user_requirement(1e-6, "V")
    build(ir)


# --------------------------------------------------------------------------- 10. paths ngspice commands cannot take


@needs_dll
def test_apostrophe_in_the_workdir_does_not_fail_the_run(tmp_path: Path):
    work = tmp_path / "O'Brien" / "$proj"
    work.mkdir(parents=True)
    ir = divider_with_connector_ir(work, LIB)
    ctx = _context(work)
    results = _compile_and_run(ir, ctx)
    summary = results[0]
    assert summary.status is S.PASS, summary.message
    op = summary.details["analyses"]["op"]
    assert op["succeeded"] and Path(op["raw_output_path"]).parent == (work / "spice" / "op").resolve()
    assert abs(ir.validation.latest("spice.v_out").details["measured"] - 6.0) < 1e-9
    data = read_results(ir.artifacts[ArtifactKind.SPICE_RESULT].path)
    assert "note: rawfile written to" in data["analyses"]["op"]["result"]["log"]  # via the temp dir, then copied
    assert data["netlist_hash"] == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash == SpiceRunner.netlist_hash(Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path))


@needs_dll
def test_no_writable_rawfile_location_is_not_verified_not_a_design_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ai_eda.tools.spice import ngspice_shared

    work = tmp_path / "O'Brien"
    work.mkdir()
    monkeypatch.setattr(ngspice_shared.tempfile, "gettempdir", lambda: str(tmp_path / "also'bad"))
    ir = divider_with_connector_ir(work, LIB)
    ctx = _context(work)
    results = _compile_and_run(ir, ctx)
    summary = results[0]
    assert summary.status is S.NOT_VERIFIED and "could not be run here" in summary.message and "repair" not in summary.details
    assert summary.details["unverifiable"]["op"].startswith("the rawfile (the run's evidence) cannot be written")
    assert ir.validation.latest("spice.v_out").status is S.NOT_VERIFIED and ir.validation.latest("spice.v_out").details["unverifiable"]
    assert _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.NOT_VERIFIED


# --------------------------------------------------------------------------- 11 / 18 / 19. engine info, conditions, .temp


@needs_dll
def test_results_record_the_engine_configuration_and_the_run_conditions(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _spice_stage(ir, ctx)
    data = read_results(ir.artifacts[ArtifactKind.SPICE_RESULT].path)
    info = data["engine_info"]
    assert info["version"] == "ngspice-46" and info["build"] == runner.build() and info["dll_path"] == str(runner.dll_path)
    assert info["codemodels_loaded"] is True and info["codemodel_errors"] == [] and info["self_test"]["ok"] is True
    assert "sharedmode" in info["settings"] and info["settings_hash"].startswith("sha256:")
    assert data["conditions"] == {
        "values": "nominal (SpiceBinding.value of every part; no tolerance corners, no Monte Carlo)",
        "corners": "not simulated",
        "temperature_c": NGSPICE_DEFAULT_TEMP_C,
        "temperature_source": "ngspice default (no .temp card)",
        "component_tolerances_recorded": [],
    }
    summary = ir.validation.latest("spice")
    assert summary.details["engine_info"] == info and summary.details["conditions"] == data["conditions"]
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.details["conditions"] == data["conditions"] and v_out.details["engine"]["settings_hash"] == info["settings_hash"]
    assert v_out.details["engine"]["codemodels_loaded"] is True and v_out.details["engine"]["build"] == info["build"]
    assert _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS].details["conditions"] == data["conditions"]


@needs_dll
def test_temp_card_changes_a_temperature_dependent_result_and_is_recorded(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.component("R1").spice.params["tc1"] = authoritative(0.01, RESISTOR_DS, "1/K")  # R1 doubles between 27 and 127 degC
    ir.simulation.temperature_c = user_requirement(127.0, "degC")
    ir.simulation.expectations[0].nominal = user_requirement(4.0, "V", note="12 V * 10k / (20k + 10k) at 127 degC")
    ctx = _context(tmp_path)
    results = _compile_and_run(ir, ctx)
    assert ".temp 127" in Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.PASS and abs(v_out.details["measured"] - 4.0) < 1e-9, v_out.message
    assert v_out.details["conditions"]["temperature_c"] == 127.0 and v_out.details["conditions"]["temperature_source"].startswith("SimulationSetup.temperature_c")
    assert results[0].details["conditions"]["temperature_c"] == 127.0


class _FailingRunner(SpiceRunner):
    """An engine whose code models did not load and whose analyses fail (a stand-in: the real singleton cannot be broken in-process)."""

    engine = "fake-spice"

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "fake-1"

    def engine_info(self) -> dict:
        return {"engine": self.engine, "version": "fake-1", "codemodels_loaded": False, "codemodel_errors": ["code model analog.cm not found"]}

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None) -> SpiceResult:
        return SpiceResult(
            engine=self.engine, engine_version="fake-1", netlist_path=str(netlist_path), netlist_hash=self.netlist_hash(netlist_path),
            analysis=analysis, command=command or "op", errors=["Error: unknown model type gain"], succeeded=False,
        )


def test_analysis_failure_caused_by_missing_code_models_is_not_verified(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": _FailingRunner()})
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools=ctx.tools))
    results = run_spice_for(ir, ctx.tools, tmp_path)
    assert results[0].status is S.FAIL and results[0].details["repair"] == "human"  # no A element: an honest analysis failure
    # the same failure on a deck that needs the code models the engine could not load is the engine's problem
    text = Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    patched = text.replace(".end", "A1 VIN VOUT gainblk\n.model gainblk gain(gain=2)\n.end")
    Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).write_text(patched, encoding="utf-8", newline="\n")
    art = ir.artifacts[ArtifactKind.SPICE_NETLIST]
    art.content_hash = art.disk_hash()
    results = run_spice_for(ir, ctx.tools, tmp_path)
    assert results[0].status is S.NOT_VERIFIED and "code models did not load" in results[0].message and "repair" not in results[0].details
    assert all(r.status is S.NOT_VERIFIED for r in results[1:] if r.check_id.startswith("spice."))


# --------------------------------------------------------------------------- 12. connected pin outside pin_order


def _pot_ir(tmp_path: Path) -> CircuitIR:
    ir = divider_with_connector_ir(tmp_path, LIB)
    pot = Component(
        ref="RV1", value="10k", pins=[Pin(number=str(n), name="", electrical_type=PinElectricalType.PASSIVE, provenance=AUTH) for n in (1, 2, 3)],
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test"),
        spice=SpiceBinding(device=SpiceDevice.R, value=authoritative(10_000.0, DS, "ohm"), pin_order=["1", "3"], provenance=AUTH),
    )
    ir.components.append(pot)
    ir.net("VOUT").pins.append(PinRef(component_ref="RV1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="RV1", pin_number="3"))
    ir.nets.append(Net(name="WIPER", pins=[PinRef(component_ref="RV1", pin_number="2")], provenance=USER))
    return ir


def test_connected_pin_the_element_does_not_use_must_be_declared_ignored(tmp_path: Path):
    ir = _pot_ir(tmp_path)
    with pytest.raises(CompileError, match=r"RV1.2 is in a net but the SPICE binding does not use it .* list it in ignored_pins"):
        build(ir)
    ir.component("RV1").spice.ignored_pins = {"2": ""}
    with pytest.raises(CompileError, match="RV1.2 is in ignored_pins without a reason"):
        build(ir)
    ir.component("RV1").spice.ignored_pins = {"2": "wiper not modelled: the pot is simulated at its end-to-end value"}
    assert "RV1 VOUT 0 10k" in build(ir)
    report = build_report(ir)
    assert report["ignored_pins"] == {"RV1": {"2": "wiper not modelled: the pot is simulated at its end-to-end value"}}
    assert report["nets_touching_only_excluded"] == ["WIPER"]  # RV1.2 is ignored: the net has no node in the netlist
    ir.component("RV1").spice.ignored_pins = {"2": "x", "3": "y"}
    with pytest.raises(CompileError, match=r"pins \['3'\] are in pin_order and in ignored_pins"):
        build(ir)


# --------------------------------------------------------------------------- 13. nominal vs the requirement it claims to verify


@needs_dll
def test_recalculated_nominals_do_not_pass_review_while_the_requirement_is_unchanged(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _spice_stage(ir, ctx)
    _change_r1(ir, 20_000.0, "20k")
    _recalculate_nominals(ir)  # nominals 4 V / 2 V; req.v_out still says 6 V, req.v_out_half 3 V
    assert recompute_parameters(ir).status is S.PASS
    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    assert [a.description for a in outcome.actions] == ["regenerate spice_netlist from IR", "re-run spice"]
    assert outcome.stopped_reason == "no repairable failures remain"
    final = {r.check_id: r for r in outcome.final_review.results}
    r = final[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert r.status is S.FAIL and r.details["repair"] == "human" and r.details["failed"] == []  # ngspice agrees with the nominals ...
    assert r.details["nominal_vs_requirement"] == [
        "v_out: nominal 4 V is not requirement req.v_out's 6 V (+/- 0.06 V)",
        "v_out_mid: nominal 2 V is not requirement req.v_out_half's 3 V (+/- 0.03 V)",
    ]  # ... but the nominals are not the requirements
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.SPICE_VS_REQUIREMENTS]
    assert ir.validation.latest("spice.v_out").status is S.PASS  # the simulation itself is honest: 4 V is what the netlist gives


def test_expectation_traced_to_a_requirement_without_a_value_is_not_compared(tmp_path: Path):
    reviewer = IndependentReviewer()
    exp = Expectation(id="e", analysis_id="op", vector="v(VOUT)", nominal=user_requirement(6.0, "V"), tol_rel=user_requirement(0.01), requirement_id="req.x", provenance=USER)
    req = Requirement(id="req.x", key="x", text="six volts", kind=RequirementKind.EXPLICIT)
    assert reviewer._nominal_vs_requirement(exp, req) == ("requirement req.x has no value to compare the nominal with", False)
    req.value = user_requirement("six", None)
    assert reviewer._nominal_vs_requirement(exp, req)[1] is False
    req.value = user_requirement(6.0, "mV")
    assert reviewer._nominal_vs_requirement(exp, req) == ("nominal unit 'V' is not the requirement's 'mV'", False)
    req.value = user_requirement(6.05, "V")
    assert reviewer._nominal_vs_requirement(exp, req) == (None, True)  # within the expectation's 1 %
    req.value = user_requirement(6.1, "V")
    assert reviewer._nominal_vs_requirement(exp, req) == ("nominal 6 V is not requirement req.x's 6.1 V (+/- 0.061 V)", True)


# --------------------------------------------------------------------------- 15. assumptions in the netlist


@needs_dll
def test_assumed_binding_value_is_surfaced_and_never_evidence(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.component("R1").spice.value = assumption(10_000.0, "guessed", "ohm")
    ir.component("R1").electrical["resistance"] = ir.component("R1").spice.value
    ctx = _context(tmp_path)
    state = _spice_stage(ir, ctx)
    assert state.blocked and state.current == Stage.IR_BUILD  # the structural validator surfaces it before anything runs
    assert ir.validation.latest("ir.assumptions").details["assumptions"] == ["components[R1].spice.value: guessed"]
    # and even when the stage is run directly, an assumed value yields no PASS anywhere
    results = _compile_and_run(ir, ctx)
    summary = results[0]
    assert summary.status is S.NOT_VERIFIED and summary.details["assumptions"] == ["R1 value"] and "not evidence until confirmed" in summary.message
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.NOT_VERIFIED and abs(v_out.details["measured"] - 6.0) < 1e-9 and "assumption" in v_out.message
    bias = default_registry.get("domain.analog.bias").validate(ir, ValidationContext(workdir=tmp_path))[0]
    assert bias.status is S.NOT_VERIFIED and "the spice summary is NOT_VERIFIED" in bias.message
    review = _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert review.status is S.NOT_VERIFIED and "assumption" in review.message and review.details["assumptions"] == ["R1 value"]


def test_assumptions_validator_walks_the_whole_simulation_setup(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.stimuli[0].value = assumption(12.0, "typical bench supply", "V")
    ir.simulation.analyses[1].params["stop"] = assumption(12.0, "sweep to the supply", "V")
    ir.simulation.expectations[1].tol_rel = assumption(0.01, "usual accuracy")
    ir.simulation.expectations[0].provenance = Provenance(kind=ProvenanceKind.ASSUMPTION, note="someone thought op was enough")
    ir.simulation.temperature_c = assumption(25.0, "room", "degC")
    r = default_registry.get("ir.assumptions").validate(ir, ValidationContext(workdir=tmp_path))[0]
    assert r.status is S.USER_INPUT_REQUIRED and r.details["assumptions"] == [
        "simulation.stimuli[VIN].value: typical bench supply",
        "simulation.analyses[dc_vin].params[stop]: sweep to the supply",
        "simulation.expectations[v_out]: someone thought op was enough",
        "simulation.expectations[v_out_mid].tol_rel: usual accuracy",
        "simulation.temperature_c: room",
    ]


# --------------------------------------------------------------------------- 16. llm_generated containers


def test_llm_generated_bindings_stimuli_analyses_and_expectations_are_refused(tmp_path: Path):
    llm = Provenance(kind=ProvenanceKind.LLM_GENERATED, tool="some-model")
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.component("R1").spice.provenance = llm
    with pytest.raises(CompileError, match="R1 SPICE binding has llm_generated provenance"):
        build(ir)
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.component("J1").spice.provenance = llm  # an LLM deciding what is *excluded* is refused too
    with pytest.raises(CompileError, match="J1 SPICE binding has llm_generated provenance"):
        build(ir)
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.stimuli[0].provenance = llm
    with pytest.raises(CompileError, match="stimulus VIN has llm_generated provenance"):
        build(ir)
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.analyses[0].provenance = llm
    with pytest.raises(CompileError, match="analysis op has llm_generated provenance"):
        build(ir)
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].provenance = llm
    with pytest.raises(CompileError, match="expectation v_out has llm_generated provenance"):
        build(ir)
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].provenance = Provenance(kind=ProvenanceKind.ASSUMPTION, note="assumed")
    report = build_report(ir)
    assert "assumption" in report["accepted_provenance_kinds"] and report["assumptions"] == ["expectation v_out"]


# --------------------------------------------------------------------------- 17. domain.analog.bias


@needs_dll
def test_bias_is_not_verified_when_the_op_expectation_fails(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].nominal = user_requirement(5.0, "V", note="wrong on purpose")
    ctx = _context(tmp_path)
    _spice_stage(ir, ctx)
    assert ir.validation.latest("spice").status is S.FAIL
    bias = ir.validation.latest("domain.analog.bias")
    assert bias.status is S.NOT_VERIFIED and "bias correctness is not judged: the spice summary is FAIL" in bias.message
    assert bias.details["voltages"] == pytest.approx({"VIN": 12.0, "VOUT": 6.0}) and bias.details["op_expectations"] == {"spice.v_out": "FAIL"}
    # a PASSing run whose only expectations are on the dc sweep leaves the op unjudged as well
    ir = divider_with_connector_ir(tmp_path / "b", LIB)
    del ir.simulation.expectations[0]
    ctx = _context(tmp_path / "b")
    _spice_stage(ir, ctx)
    assert ir.validation.latest("spice").status is S.PASS
    bias = ir.validation.latest("domain.analog.bias")
    assert bias.status is S.NOT_VERIFIED and "no expectation on analysis op PASSed" in bias.message


# --------------------------------------------------------------------------- 20. ControlledExit recovery


@needs_dll
def test_controlled_exit_is_recovered_and_the_engine_keeps_working(tmp_path: Path):
    eng = runner._engine()
    deck = tmp_path / "divider.cir"
    deck.write_text("divider\nV1 VIN 0 DC 12\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\n.end\n", encoding="utf-8", newline="\n")
    assert runner.run(deck, SpiceAnalysis.OP, tmp_path).succeeded
    with eng.lock:
        eng.unload_all()
        eng.cap.clear()
        eng._cmd("quit")  # the DLL calls ControlledExit(quit_exit=True) and "awaits to be reset"
        exits = list(eng.cap.exits)
        assert exits and exits[0][2] is True, exits
        eng._after_failure(exits)  # the runner's own recovery path: Reset + Init + settings + code models + self-test
        assert not eng.dead and eng.self_test["ok"] and eng.codemodels_loaded
    res = runner.run(deck, SpiceAnalysis.OP, tmp_path)
    assert res.succeeded and abs(res.vectors["vout"][0] - 6.0) < 1e-9
    assert runner.self_test()["ok"] and runner.engine_info()["codemodels_loaded"]


# --------------------------------------------------------------------------- 21. the parser model against the DLL


@needs_dll
def test_ngspice_reads_matches_the_dll_on_a_fresh_sample_of_spellings():
    spellings = sample_spellings(500)
    assert len(set(spellings)) == 500
    measured = measure_spellings(runner, spellings)
    assert set(measured) == set(spellings)
    modelled = {t: v for t, v in measured.items() if ngspice_reads(t) is not None}
    assert len(modelled) >= 400  # the sample stays inside the region the model vouches for
    mismatches = [(t, v, ngspice_reads(t)) for t, v in modelled.items() if ngspice_reads(t) != v]
    assert mismatches == []
    # the pinned values of tests/test_si.py come from this very harness
    assert measure_spellings(runner, ["10u", "3.3", "1M", "1meg"]) == {"10u": 9.999999999999999e-06, "3.3": 3.3000000000000003, "1M": 0.001, "1meg": 1e6}


# --------------------------------------------------------------------------- 23. ac phase / real / imaginary expectations


def _rc_ac_ir(tmp_path: Path) -> CircuitIR:
    """The RC fixture driven by a 1 V ac source, swept ``ac lin 11 100 200`` (150 Hz is an exact grid point)."""
    ir = rc_lowpass_ir(tmp_path, LIB)
    p = ir.parameters
    p["f_probe"] = user_requirement(150.0, "Hz")
    p["mag_150"] = rc_lowpass_magnitude(p["f_probe"], p["tau"], ("f_probe", "tau"))
    p["phase_150"] = rc_lowpass_phase_deg(p["f_probe"], p["tau"], ("f_probe", "tau"))
    ir.simulation.stimuli = [
        Stimulus(id="VIN", source="voltage", net="IN", reference_net="GND", kind=StimulusKind.DC, value=user_requirement(0.0, "V"), params={"ac": user_requirement(1.0, "V")}, provenance=USER)
    ]
    ir.simulation.analyses = [
        AnalysisSpec(id="ac", kind=SpiceAnalysis.AC, params={"variation": user_requirement("lin"), "points": user_requirement(11), "fstart": user_requirement(100.0, "Hz"), "fstop": user_requirement(200.0, "Hz")}, provenance=USER)
    ]
    ir.simulation.expectations = [
        Expectation(id="mag", analysis_id="ac", vector="v(OUT)", reduce=Reduce.AT, at=p["f_probe"], nominal=p["mag_150"], tol_rel=user_requirement(0.01), provenance=USER),
        Expectation(id="phase", analysis_id="ac", vector="vp(OUT)", reduce=Reduce.AT, at=p["f_probe"], nominal=p["phase_150"], tol_abs=user_requirement(0.5, "deg"), provenance=USER),
    ]
    return ir


def test_vector_grammar_names_ac_parts_and_refuses_them_elsewhere(tmp_path: Path):
    ir = _rc_ac_ir(tmp_path)
    assert spice_vector_name("vp(OUT)", ir) == "out.phase_deg" and spice_vector_name("vr(OUT)", ir) == "out.real" and spice_vector_name("vi(out)", ir) == "out.imag"
    assert spice_vector_name("ip(VIN)", ir) == "vvin#branch.phase_deg"
    for bad in ("vdb(OUT)", "v(OUT).phase_deg", "vm(OUT)"):
        with pytest.raises(CompileError, match="not of the form"):
            spice_vector_name(bad, ir)
    assert build_report(ir)["vectors"] == {"mag": "out", "phase": "out.phase_deg"}
    tran = rc_lowpass_ir(tmp_path / "t", LIB)
    tran.simulation.expectations[0].vector = "vp(OUT)"
    with pytest.raises(CompileError, match="names a part of a complex vector, which only an ac analysis produces"):
        build(tran)


@needs_dll
def test_ac_magnitude_and_phase_expectations_pass_on_a_grid_point(tmp_path: Path):
    ir = _rc_ac_ir(tmp_path)
    ctx = _context(tmp_path)
    state = _spice_stage(ir, ctx)
    assert state.outcome(Stage.CALCULATION).status is S.PASS
    assert state.outcome(Stage.SPICE).status is S.PASS, state.outcome(Stage.SPICE).message
    mag, phase = ir.validation.latest("spice.mag"), ir.validation.latest("spice.phase")
    assert mag.status is S.PASS and mag.details["bracket"]["exact"] is True and mag.details["spice_vector"] == "out"
    assert abs(mag.details["measured"] - 1 / math.sqrt(1 + (2 * math.pi * 150 * 1e-3) ** 2)) < 1e-3
    assert phase.status is S.PASS and phase.details["spice_vector"] == "out.phase_deg"
    assert abs(phase.details["measured"] + math.degrees(math.atan(2 * math.pi * 150 * 1e-3))) < 0.5
    assert ir.validation.latest("spice").details["analyses"]["ac"]["command"] == "ac lin 11 100 200"


# --------------------------------------------------------------------------- 24. the title line


def test_title_with_dollar_or_quotes_is_refused_by_the_compiler(tmp_path: Path):
    for bad in ("a$b", "o'brien", 'say "hi"', "tick`", "two\nlines"):
        ir = divider_with_connector_ir(tmp_path, LIB)
        ir.project.id = bad
        with pytest.raises(CompileError, match="cannot be the netlist title line"):
            build(ir)
