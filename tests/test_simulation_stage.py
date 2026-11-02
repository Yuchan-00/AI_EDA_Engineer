"""The SPICE stage end to end against KiCad's bundled ngspice.dll (ngspice-46).

IR (divider / RC fixtures with SPICE bindings and a simulation setup)
-> Orchestrator: CALCULATION recomputes every derived parameter, SPICE
compiles the netlist, runs op / dc / tran through :class:`NgspiceShared`,
judges the expectations against calculator outputs, writes results.json +
rawfiles as evidence and re-validates ``domain.analog.bias``; the
independent reviewer confirms every expectation on the current netlist.

Then the honest failures: a wrong nominal is FAIL and not repairable; a
design change makes the netlist stale, the repair loop regenerates it and
re-runs ngspice, and the expectation fails honestly (4 V vs 6 V) unless the
nominals *and the requirements* were updated - in which case the loop
converges to PASS in two iterations. Hand-edited results are detected and
re-run. Nothing here ever touches the IR to make a check pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, SimulationAgent
from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.errors import ToolExecutionError
from ai_eda.ir import (
    AnalysisSpec,
    ArtifactKind,
    CircuitDomain,
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
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    ValidationResult,
    ValidationStatus,
    authoritative,
    user_requirement,
)
from ai_eda.repair import RepairLoop, RerunTool
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.calc import recompute_parameters, voltage_divider_output
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult
from ai_eda.tools.spice.measure import ABSTOL, RELTOL, VNTOL, rising_edge_frequency
from ai_eda.tools.spice.stage import judge, read_results, reduce_expectation, reduce_result, run_spice_for
from ai_eda.workflow import Orchestrator, PipelineState, Stage
from tests.conftest import make_component
from tests.fixtures_kicad import RESISTOR_DS, USER, divider_with_connector_ir, pin_header
from tests.fixtures_rc import rc_lowpass_ir

S = ValidationStatus
LIB = KicadLibrary()
runner = NgspiceShared()
pytestmark = pytest.mark.skipif(not runner.available(), reason="ngspice.dll (KiCad's bundled ngspice shared library) not found")
HAS_LIBS = LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and LIB.symbol_file("Device") is not None
#: the tests that assert on the WHOLE review report also need the KiCad libraries: without them component.existence.<ref>
#: fails for Device:R / Conn_01x03 and review.component_provenance is a second FAIL beside the SPICE one under test
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed (Device:R / R_0603_1608Metric)")


def _context(tmp_path: Path) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": runner}, answers={"application": "bench", "jurisdiction": "EU"})


def _run(ir: CircuitIR, ctx: AgentContext) -> PipelineState:
    state = Orchestrator(ctx).run(ir, stop_after=Stage.SPICE)
    for o in state.outcomes:
        print(f"{o.stage:<24} {o.status:<14} {o.message}")
    return state


def _review(ir: CircuitIR, ctx: AgentContext) -> dict[str, ValidationResult]:
    report = IndependentReviewer(tools=ctx.tools).review(ir, ctx.workdir)
    for r in report.results:
        print(f"  {r.check_id:<34} {r.status:<14} {r.message}")
    return {r.check_id: r for r in report.results}


def _evidence_is_real(res: ValidationResult) -> None:
    assert res.evidence, res.check_id
    for e in res.evidence:
        p = Path(e.path)
        assert p.is_file(), e
        assert e.content_hash and e.content_hash == "sha256:" + __import__("hashlib").sha256(p.read_bytes()).hexdigest(), e


def _change_r1_to_20k(ir: CircuitIR) -> None:
    """A design change: R1 becomes 20 k (VOUT = 12 * 10k / 30k = 4 V). Parameters follow, expectations do not (yet)."""
    r1 = ir.component("R1")
    r1.value = "20k"
    r1.electrical["resistance"] = authoritative(20_000.0, RESISTOR_DS, "ohm")
    r1.spice.value = r1.electrical["resistance"]
    ir.parameters["r1"] = r1.electrical["resistance"]


def _recalculate_expectations(ir: CircuitIR) -> None:
    """The nominals are calculator outputs: recompute them from the changed parameters and point the expectations at them."""
    p = ir.parameters
    p["v_out"] = voltage_divider_output(p["v_in"], p["r1"], p["r2"], ("v_in", "r1", "r2"))
    p["v_out_mid"] = voltage_divider_output(p["v_in_mid"], p["r1"], p["r2"], ("v_in_mid", "r1", "r2"))
    for exp in ir.simulation.expectations:
        exp.nominal = p["v_out"] if exp.id == "v_out" else p["v_out_mid"]


# --------------------------------------------------------------------------- the stage


def test_divider_spice_stage_passes_with_real_evidence(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    state = _run(ir, ctx)
    assert not state.blocked
    assert state.outcome(Stage.CALCULATION).status is S.PASS and "4 value(s) recomputed" in state.outcome(Stage.CALCULATION).message
    spice_stage = state.outcome(Stage.SPICE)
    assert spice_stage.status is S.PASS, spice_stage.message
    assert "re-validated: " in spice_stage.message and "domain.analog.bias PASS" in spice_stage.message  # component.fit is listed too (registry order)

    netlist = ir.artifacts[ArtifactKind.SPICE_NETLIST]
    results = ir.artifacts[ArtifactKind.SPICE_RESULT]
    ir_hash = ir.content_hash()
    assert netlist.generated_from_ir_hash == ir_hash and netlist.matches_disk()
    assert results.generated_from_ir_hash == ir_hash and results.matches_disk()
    assert Path(netlist.path).read_text(encoding="utf-8") == "divider_conn\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\nVVIN VIN 0 DC 12\n.end\n"
    assert Path(results.path) == tmp_path / "spice" / "results.json"

    summary = ir.validation.latest("spice")
    assert summary.status is S.PASS and summary.tool == "ngspice-shared" and summary.tool_version == runner.version()
    assert summary.artifact_hash == netlist.content_hash and summary.ir_hash == ir_hash
    assert set(summary.details["analyses"]) == {"op", "dc_vin"}
    assert summary.details["analyses"]["op"] == {**summary.details["analyses"]["op"], "command": "op", "succeeded": True, "n_points": 1, "plot_name": "op1"}
    assert summary.details["analyses"]["dc_vin"]["command"] == "dc vvin 0 12 1" and summary.details["analyses"]["dc_vin"]["n_points"] == 13
    assert summary.details["netlist_provenance_kinds"] == ["authoritative", "derived", "user_requirement"]
    assert summary.details["excluded"] == [{"ref": "J1", "reason": "connector, no electrical model"}]
    assert summary.details["expectations"] == {"v_out": "PASS", "v_out_mid": "PASS"}
    _evidence_is_real(summary)
    assert {Path(e.path).name for e in summary.evidence} == {"results.json", "divider_conn.cir", "divider_conn.cir.report.json", "divider_conn.op.raw", "divider_conn.dc.raw"}

    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.PASS and v_out.tool == "ngspice-shared" and v_out.artifact_hash == netlist.content_hash
    assert abs(v_out.details["measured"] - 6.0) < 1e-9 and v_out.details["nominal"] == 6.0 and v_out.details["tol_rel"] == 0.01
    assert v_out.details["tolerance"] == pytest.approx(0.06) and v_out.details["requirement_id"] == "req.v_out"
    assert v_out.details["provenance_kinds_used"] == ["derived", "user_requirement"] and v_out.details["spice_vector"] == "vout"
    _evidence_is_real(v_out)
    assert Path(v_out.evidence[0].path).name == "divider_conn.op.raw"
    mid = ir.validation.latest("spice.v_out_mid")
    assert mid.status is S.PASS and abs(mid.details["measured"] - 3.0) < 1e-9 and mid.details["at"] == 6.0 and mid.details["reduce"] == "at"
    assert Path(mid.evidence[0].path).name == "divider_conn.dc.raw"

    # results.json is the persisted evidence: every SpiceResult, keyed by analysis id, naming the netlist it ran on
    data = read_results(results.path)
    assert data["netlist_hash"] == netlist.content_hash and data["ir_hash"] == ir_hash and data["engine_version"] == runner.version()
    assert data["analyses"]["op"]["result"]["vectors"]["vout"] == pytest.approx([6.0])
    assert len(data["analyses"]["dc_vin"]["result"]["vectors"]["v-sweep"]) == 13
    assert data["netlist_report"]["accepted_provenance_kinds"] == ["authoritative", "derived", "user_requirement"]

    calc = ir.validation.latest("calc.recompute")
    assert calc.status is S.PASS and calc.tool == "calc" and calc.details["parameters"]["v_out"]["recomputed"] == 6.0
    assert calc.details["parameters"]["v_out_mid"]["status"] == "PASS"

    bias = ir.validation.latest("domain.analog.bias")
    assert bias.status is S.PASS and bias.tool == "ngspice-shared" and bias.artifact_hash == netlist.content_hash
    assert bias.details["voltages"] == pytest.approx({"VIN": 12.0, "VOUT": 6.0})
    _evidence_is_real(bias)

    review = _review(ir, ctx)
    spice_review = review[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert spice_review.status is S.PASS, spice_review.message
    assert spice_review.details["verified"] == {"v_out": "req.v_out", "v_out_mid": "req.v_out_half"} and spice_review.details["untraced"] == []
    assert "nominal component values and one temperature only" in spice_review.message
    _evidence_is_real(spice_review)
    assert review[ReviewArea.CALCULATIONS_VS_DESIGN].status is S.PASS
    assert review[ReviewArea.REQUIREMENTS_VS_IR].status is S.PASS


def test_rc_step_response_calculator_and_ngspice_agree(tmp_path: Path):
    ir = rc_lowpass_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    state = _run(ir, ctx)
    assert state.outcome(Stage.CALCULATION).status is S.PASS
    assert state.outcome(Stage.SPICE).status is S.PASS, state.outcome(Stage.SPICE).message
    assert Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8") == "rc_lowpass\nC1 OUT 0 1u\nR1 IN OUT 1k\nVVIN IN 0 PULSE(0 5 0 1n 1n 1 2)\n.end\n"
    exp = ir.validation.latest("spice.v_out_tau")
    nominal = 5.0 * (1 - math.exp(-1))
    assert exp.status is S.PASS and exp.details["nominal"] == pytest.approx(nominal) and exp.details["at"] == pytest.approx(1e-3)
    # ngspice (tran 1u) vs the closed form: the documented bound (README / ARCHITECTURE: "within 1e-6 V"; measured 8.5e-7 V),
    # far inside the 2 % the IR allows - the IR tolerance is the requirement, this is the engine's accuracy claim
    assert abs(exp.details["measured"] - nominal) < 1e-6
    assert exp.details["bracket"]["method"] == "linear" and not exp.details["bracket"]["exact"] and exp.details["bracket"]["x0"] < 1e-3 < exp.details["bracket"]["x1"]
    assert exp.details["command"] == "tran 1u 5m" and exp.details["spice_vector"] == "out"
    _evidence_is_real(exp)
    summary = ir.validation.latest("spice")
    assert summary.details["analyses"]["tran"]["n_points"] > 1000 and summary.details["analyses"]["tran"]["scale"] == "time"
    calc = ir.validation.latest("calc.recompute")
    assert calc.status is S.PASS and calc.details["parameters"]["tau"]["recomputed"] == pytest.approx(1e-3)
    assert calc.details["parameters"]["v_out_at_tau"]["tool"] == "calc.rc.step_response"
    bias = ir.validation.latest("domain.analog.bias")
    assert bias.status is S.NOT_VERIFIED and "no successful operating-point" in bias.message  # no op requested: said, not hidden
    review = _review(ir, ctx)
    assert review[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.PASS
    assert review[ReviewArea.CALCULATIONS_VS_DESIGN].status is S.PASS


def test_second_pipeline_run_is_consistent_and_ir_build_sees_the_bias(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _run(ir, ctx)
    first = ir.validation.latest("spice.v_out").details["measured"]
    state = _run(ir, ctx)  # IR_BUILD now finds the previous run's evidence: bias PASS already at that stage
    assert state.outcome(Stage.SPICE).status is S.PASS
    bias_at_ir_build = [r for r in ir.validation.results if r.check_id == "domain.analog.bias"]
    assert [r.status for r in bias_at_ir_build] == [S.NOT_VERIFIED, S.PASS, S.PASS, S.PASS]
    assert ir.validation.latest("spice.v_out").details["measured"] == first


# --------------------------------------------------------------------------- honest failures


@needs_libs
def test_wrong_nominal_fails_and_is_not_repairable(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].nominal = user_requirement(5.0, "V", note="wrong on purpose")
    ctx = _context(tmp_path)
    state = _run(ir, ctx)
    assert state.outcome(Stage.SPICE).status is S.FAIL
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.FAIL and abs(v_out.details["measured"] - 6.0) < 1e-9 and v_out.details["repair"] == "human"
    assert "deviation 1 V" in v_out.message
    summary = ir.validation.latest("spice")
    assert summary.status is S.FAIL and summary.details["repair"] == "human" and "spice.v_out" in summary.message
    assert ir.validation.latest("spice.v_out_mid").status is S.PASS  # the other expectation is judged on its own
    review = _review(ir, ctx)
    r = review[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert r.status is S.FAIL and r.details["repair"] == "human" and r.details["failed"][0].startswith("v_out:")
    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    print(outcome.summary())
    assert outcome.actions == [] and outcome.stopped_reason == "no repairable failures remain"
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.SPICE_VS_REQUIREMENTS]
    assert "human" in outcome.unresolved[0].message


@needs_libs
def test_design_change_is_resimulated_and_the_old_expectation_fails_honestly(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _run(ir, ctx)
    old_netlist = ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    _change_r1_to_20k(ir)
    changed = ir.content_hash()
    review = _review(ir, ctx)
    r = review[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert r.status is S.FAIL and r.details == {"artifact": ArtifactKind.SPICE_NETLIST, "repair": "regenerate"}

    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    print(outcome.summary())
    for a in outcome.actions:
        print(f"  {a.strategy:<28} {a.description:<40} ok={a.succeeded} {a.error or ''}")
    assert [a.description for a in outcome.actions] == ["regenerate spice_netlist from IR", "re-run spice"]
    assert all(a.succeeded and a.ir_hash_before == changed == a.ir_hash_after for a in outcome.actions)
    # iteration 1 regenerates, 2 re-runs, 3 finds only the human findings left and stops: the stale stored
    # v_out (6 V, derived from the old r1) is caught by the reviewer's own recompute as well as by ngspice
    assert outcome.iterations == 3 and outcome.stopped_reason == "no repairable failures remain"
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.CALCULATIONS_VS_DESIGN, ReviewArea.SPICE_VS_REQUIREMENTS]
    assert all("human" in u.message for u in outcome.unresolved)
    assert ir.content_hash() == changed
    netlist = ir.artifacts[ArtifactKind.SPICE_NETLIST]
    assert netlist.content_hash != old_netlist and netlist.generated_from_ir_hash == changed and "R1 VIN VOUT 20k" in Path(netlist.path).read_text(encoding="utf-8")
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.FAIL and abs(v_out.details["measured"] - 4.0) < 1e-9 and v_out.details["nominal"] == 6.0
    assert v_out.artifact_hash == netlist.content_hash
    mid = ir.validation.latest("spice.v_out_mid")
    assert mid.status is S.FAIL and abs(mid.details["measured"] - 2.0) < 1e-9
    final = {r.check_id: r for r in outcome.final_review.results}
    assert final[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.FAIL and final[ReviewArea.SPICE_VS_REQUIREMENTS].details["repair"] == "human"


@needs_libs
def test_design_change_with_recalculated_nominals_and_requirements_converges_to_pass(tmp_path: Path):
    """R1 -> 20k *and* the user now asks for 4 V: nominals and requirements move together, so the loop converges.

    (Recalculating the nominals while the requirement still says 6 V is refused by the reviewer -
    ``tests/test_spice_findings_regressions.py``.)
    """
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _run(ir, ctx)
    _change_r1_to_20k(ir)
    _recalculate_expectations(ir)
    ir.requirements.get("v_out").value = user_requirement(4.0, "V")
    ir.requirements.get("v_out").text = "4 V output (a third of the input) within 1 %"
    ir.requirements.get("v_out_half").value = user_requirement(2.0, "V")
    assert ir.parameters["v_out"].value == pytest.approx(4.0) and ir.parameters["v_out_mid"].value == pytest.approx(2.0)
    assert recompute_parameters(ir).status is S.PASS  # the calculators stand behind the new nominals
    changed = ir.content_hash()
    assert _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS].details["repair"] == "regenerate"

    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    print(outcome.summary())
    assert [a.description for a in outcome.actions] == ["regenerate spice_netlist from IR", "re-run spice"]
    assert outcome.stopped_reason == "all failures resolved" and outcome.unresolved == [] and outcome.iterations == 2
    assert all(a.succeeded for a in outcome.actions) and ir.content_hash() == changed
    final = {r.check_id: r for r in outcome.final_review.results}
    assert final[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.PASS
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.PASS and abs(v_out.details["measured"] - 4.0) < 1e-9 and v_out.details["nominal"] == pytest.approx(4.0)
    assert ir.artifacts[ArtifactKind.SPICE_RESULT].generated_from_ir_hash == changed and ir.artifacts[ArtifactKind.SPICE_RESULT].matches_disk()


@needs_libs
def test_hand_edited_results_are_detected_and_rerun(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _run(ir, ctx)
    results = Path(ir.artifacts[ArtifactKind.SPICE_RESULT].path)
    data = json.loads(results.read_text(encoding="utf-8"))
    data["analyses"]["op"]["result"]["vectors"]["vout"] = [5.0]
    results.write_text(json.dumps(data), encoding="utf-8")
    review = _review(ir, ctx)
    r = review[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert r.status is S.FAIL and r.details == {"tool_check": "spice", "repair": "rerun_tool"}
    assert ir.validation.latest("domain.analog.bias").status is S.PASS  # recorded before the edit ...
    from ai_eda.validation import ValidationContext, default_registry

    bias_now = default_registry.get("domain.analog.bias").validate(ir, ValidationContext(workdir=tmp_path))[0]
    assert bias_now.status is S.NOT_VERIFIED and "does not match its recorded hash" in bias_now.message  # ... but not now
    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    assert [a.description for a in outcome.actions] == ["re-run spice"] and outcome.unresolved == [] and outcome.iterations == 1
    assert ir.artifacts[ArtifactKind.SPICE_RESULT].matches_disk()
    assert {r.check_id: r.status for r in outcome.final_review.results}[ReviewArea.SPICE_VS_REQUIREMENTS] is S.PASS


def test_rerun_tool_refuses_a_stale_netlist(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    _run(ir, ctx)
    _change_r1_to_20k(ir)
    finding = ValidationResult(check_id=ReviewArea.SPICE_VS_REQUIREMENTS, status=S.FAIL, details={"repair": "rerun_tool", "tool_check": "spice"})
    action = RerunTool().apply(ir, finding, tmp_path, ctx.tools)
    assert not action.succeeded and "regenerate it first" in action.error
    with pytest.raises(ToolExecutionError, match="regenerate it first"):
        run_spice_for(ir, ctx.tools, tmp_path)


# --------------------------------------------------------------------------- what the agent refuses to claim


def test_agent_without_setup_engine_or_bindings_is_not_verified(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ctx = _context(tmp_path)
    Orchestrator(ctx)  # registers the compilers
    ir.simulation = None
    res = SimulationAgent().run(ir, ctx)
    assert [(r.check_id, r.status, r.message) for r in res.validation] == [("spice", S.NOT_VERIFIED, "no simulation setup in IR")]

    ir = divider_with_connector_ir(tmp_path, LIB)
    res = SimulationAgent().run(ir, AgentContext(workdir=tmp_path, tools={"compilers": ctx.tools["compilers"]}))
    assert res.validation[0].status is S.NOT_VERIFIED and res.validation[0].message == "no SPICE engine available"
    assert ArtifactKind.SPICE_NETLIST not in ir.artifacts

    ir = divider_with_connector_ir(tmp_path, LIB)
    for c in ir.components:
        c.spice = None  # the model mapping has not been made: nothing to compile, not a failure
    res = SimulationAgent().run(ir, ctx)
    assert res.validation[0].status is S.NOT_VERIFIED and "no SPICE binding for J1" in res.validation[0].message
    assert res.proposals == []


def test_unbound_component_is_a_fail_not_a_guess(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.components.append(make_component("R3", "1k"))  # no SpiceBinding
    ir.net("VOUT").pins.append(PinRef(component_ref="R3", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="R3", pin_number="2"))
    ctx = _context(tmp_path)
    state = _run(ir, ctx)
    o = state.outcome(Stage.SPICE)
    assert o.status is S.FAIL and "no SPICE binding for R3" in o.message
    spice = ir.validation.latest("spice")
    assert spice.status is S.FAIL and spice.details["repair"] == "human" and spice.tool == "compiler.spice"
    assert ArtifactKind.SPICE_NETLIST not in ir.artifacts and ArtifactKind.SPICE_RESULT not in ir.artifacts
    assert _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.NOT_VERIFIED


def test_expectation_without_tolerance_is_unresolved_and_missing_vector_is_fail(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[0].tol_rel = None  # nothing to judge against
    # a net that touches only an excluded part exists in the IR but has no node in the netlist: no vector
    ir.components.append(pin_header("J2", 2, LIB))
    ir.nets.append(Net(name="SPARE", pins=[PinRef(component_ref="J2", pin_number="1"), PinRef(component_ref="J2", pin_number="2")], provenance=USER))
    ir.simulation.expectations.append(
        Expectation(id="spare", analysis_id="op", vector="v(SPARE)", reduce=Reduce.VALUE, nominal=user_requirement(0.0, "V"), tol_abs=user_requirement(0.1, "V"), provenance=USER)
    )
    ctx = _context(tmp_path)
    state = _run(ir, ctx)
    assert state.outcome(Stage.SPICE).status is S.FAIL
    assert ir.validation.latest("spice.v_out").status is S.UNRESOLVED and ir.validation.latest("spice.v_out").message.startswith("no tolerance")
    spare = ir.validation.latest("spice.spare")
    assert spare.status is S.FAIL and spare.message.startswith("vector not produced: 'spare'") and spare.details["repair"] == "human"
    assert ir.validation.latest("spice").details["nets_touching_only_excluded"] == ["SPARE"]
    review = _review(ir, ctx)
    assert review[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.FAIL and review[ReviewArea.SPICE_VS_REQUIREMENTS].details["untraced"] == ["spare"]
    bias = ir.validation.latest("domain.analog.bias")
    assert bias.status is S.NOT_VERIFIED and bias.details["missing"] == ["SPARE"]


def test_expectation_naming_an_unknown_requirement_is_flagged(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.simulation.expectations[1].requirement_id = "req.does_not_exist"
    ctx = _context(tmp_path)
    _run(ir, ctx)
    assert ir.validation.latest("spice").status is S.PASS  # the simulation itself is fine ...
    r = _review(ir, ctx)[ReviewArea.SPICE_VS_REQUIREMENTS]
    assert r.status is S.FAIL and r.details["unknown_requirements"] == ["v_out_mid -> req.does_not_exist"] and r.details["repair"] == "human"


# --------------------------------------------------------------------------- reductions (pure)


def test_judge_uses_the_looser_of_abs_and_rel_and_reduce_reports_problems():
    exp = Expectation(id="e", analysis_id="a", vector="v(X)", nominal=user_requirement(10.0), tol_abs=user_requirement(0.5), tol_rel=user_requirement(0.01), provenance=USER)
    assert judge(10.4, exp) == (S.PASS, 0.5, pytest.approx(0.4))
    assert judge(10.6, exp)[0] is S.FAIL
    exp.tol_abs = None
    assert judge(10.09, exp) == (S.PASS, pytest.approx(0.1), pytest.approx(0.09))
    exp.tol_rel = None
    assert judge(10.0, exp) == (S.UNRESOLVED, None, 0.0)
    from ai_eda.tools.spice import SpiceAnalysis, SpiceResult

    res = SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.DC, command="dc v 0 2 1", vectors={"v-sweep": [0.0, 1.0, 2.0], "x": [0.0, 1.5, 3.0]}, scale="v-sweep", n_points=3, succeeded=True)
    exp = Expectation(id="e", analysis_id="a", vector="v(X)", reduce=Reduce.AT, at=user_requirement(0.5), nominal=user_requirement(0.75), tol_abs=user_requirement(0.01), provenance=USER)
    assert reduce_result(res, exp, "x") == (0.75, None)
    exp.at = user_requirement(5.0)
    assert reduce_result(res, exp, "x")[1].startswith("v-sweep=5.0 is outside")
    assert reduce_result(res, exp, "missing")[1].startswith("vector not produced")
    for reduce, expected in ((Reduce.FINAL, 3.0), (Reduce.MAX, 3.0), (Reduce.MIN, 0.0)):
        exp.reduce = reduce
        assert reduce_result(res, exp, "x") == (expected, None)


# --------------------------------------------------------------------------- a measured frequency (Reduce.FREQUENCY + tran uic)

#: the astable's timing values: Vcc 5 V, R_c 1 k, R_b 10 k, C 72 nF, V_BE 0.7 V
ASTABLE_VCC, ASTABLE_R_C, ASTABLE_R_B, ASTABLE_C, ASTABLE_V_BE = 5.0, 1_000.0, 10_000.0, 72e-9, 0.7
#: f = 1 / (2 R_b C ln((2 Vcc - V_BE) / (Vcc - V_BE))) = 900.2 Hz - the expression the part-B template's calculator will register
ASTABLE_F_NOMINAL = 1.0 / (2.0 * ASTABLE_R_B * ASTABLE_C * math.log((2.0 * ASTABLE_VCC - ASTABLE_V_BE) / (ASTABLE_VCC - ASTABLE_V_BE)))


def _astable_part(ref: str, value: str, pins: list[tuple[str, str, PinElectricalType]], spice: SpiceBinding, electrical: dict | None = None) -> Component:
    return Component(
        ref=ref, value=value, description=value,
        pins=[Pin(number=n, name=name, electrical_type=kind, provenance=USER) for n, name, kind in pins],
        electrical=dict(electrical or {}), provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="hand-built astable"), spice=spice,
    )


def astable_ir(tmp_path: Path) -> CircuitIR:
    """A collector-coupled BJT astable multivibrator, built by hand (no library refs: only the netlist is under test).

    Q1 / Q2 carry pins 1 = E, 2 = B, 3 = C (the KiCad ``Transistor_BJT:2N3904``
    numbering) on a generic ``.model QNPN NPN`` card; ``C2`` starts at -1 V
    (``ic``) and the transient runs ``uic`` from 10 ms to 30 ms in 5 us steps,
    so the saved window holds ~18 periods of the ~900 Hz oscillation.
    """
    two = [("1", "", PinElectricalType.PASSIVE), ("2", "", PinElectricalType.PASSIVE)]
    npn = [("1", "E", PinElectricalType.PASSIVE), ("2", "B", PinElectricalType.INPUT), ("3", "C", PinElectricalType.PASSIVE)]
    card = user_requirement(".model QNPN NPN", note="generic Gummel-Poon NPN with ngspice's default parameters")
    q_binding = lambda: SpiceBinding(device=SpiceDevice.Q, model_name="QNPN", model_card=card, pin_order=["3", "2", "1"], provenance=USER)  # noqa: E731
    r_c, r_b, c = user_requirement(ASTABLE_R_C, "ohm"), user_requirement(ASTABLE_R_B, "ohm"), user_requirement(ASTABLE_C, "F")
    ir = CircuitIR(project=ProjectMeta(id="astable_hand", name="BJT astable multivibrator (hand-built)", workdir=str(tmp_path)))
    ir.topology = Topology(name="BJT astable multivibrator", domains=[CircuitDomain.ANALOG], provenance=USER)
    ir.components = [
        _astable_part("Q1", "2N3904", npn, q_binding()),
        _astable_part("Q2", "2N3904", npn, q_binding()),
        _astable_part("R1", "1k", two, SpiceBinding(device=SpiceDevice.R, value=r_c, provenance=USER), {"resistance": r_c}),
        _astable_part("R2", "1k", two, SpiceBinding(device=SpiceDevice.R, value=r_c, provenance=USER), {"resistance": r_c}),
        _astable_part("R3", "10k", two, SpiceBinding(device=SpiceDevice.R, value=r_b, provenance=USER), {"resistance": r_b}),
        _astable_part("R4", "10k", two, SpiceBinding(device=SpiceDevice.R, value=r_b, provenance=USER), {"resistance": r_b}),
        _astable_part("C1", "72n", two, SpiceBinding(device=SpiceDevice.C, value=c, provenance=USER), {"capacitance": c}),
        _astable_part(
            "C2", "72n", two,
            SpiceBinding(device=SpiceDevice.C, value=c, params={"ic": user_requirement(-1.0, "V", note="breaks the symmetry so the oscillation starts")}, provenance=USER),
            {"capacitance": c},
        ),
        _astable_part("J1", "Conn_01x03", [("1", "Pin_1", PinElectricalType.PASSIVE), ("2", "Pin_2", PinElectricalType.PASSIVE), ("3", "Pin_3", PinElectricalType.PASSIVE)],
                      SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=USER)),
    ]

    def net(name: str, kind: NetKind, *pins: tuple[str, str]) -> Net:
        return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=USER)

    ir.nets = [
        net("VCC", NetKind.POWER, ("J1", "1"), ("R1", "1"), ("R2", "1"), ("R3", "1"), ("R4", "1")),
        net("Q1_C", NetKind.SIGNAL, ("R1", "2"), ("Q1", "3"), ("C1", "1")),
        net("Q2_B", NetKind.SIGNAL, ("C1", "2"), ("R4", "2"), ("Q2", "2")),
        net("OUT", NetKind.SIGNAL, ("R2", "2"), ("Q2", "3"), ("C2", "1"), ("J1", "2")),
        net("Q1_B", NetKind.SIGNAL, ("C2", "2"), ("R3", "2"), ("Q1", "2")),
        net("GND", NetKind.GROUND, ("J1", "3"), ("Q1", "1"), ("Q2", "1")),
    ]
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="VCC", reference_net="GND", kind=StimulusKind.DC, value=user_requirement(ASTABLE_VCC, "V"), provenance=USER)],
        analyses=[
            AnalysisSpec(
                id="tran", kind=SpiceAnalysis.TRAN,
                params={"step": user_requirement(5e-6, "s"), "stop": user_requirement(30e-3, "s"), "start": user_requirement(10e-3, "s"), "uic": user_requirement(True)},
                provenance=USER,
            )
        ],
        expectations=[
            Expectation(
                id="f_osc", analysis_id="tran", vector="v(OUT)", reduce=Reduce.FREQUENCY, nominal=user_requirement(ASTABLE_F_NOMINAL, "Hz"),
                tol_rel=user_requirement(0.05, note="the expression neglects V_CE(sat) and the model's finite switching"), provenance=USER,
            )
        ],
    )
    return ir


def test_astable_multivibrator_frequency_is_measured_by_ngspice(tmp_path: Path):
    """``Reduce.FREQUENCY`` on a real transient: the astable's ~900 Hz is measured from v(OUT)'s rising edges, not assumed."""
    ir = astable_ir(tmp_path)
    ctx = _context(tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools=ctx.tools))
    text = Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    assert text.startswith("astable_hand\n.model QNPN NPN\n") and text.endswith("VVIN VCC 0 DC 5\n.end\n")
    assert "Q1 Q1_C Q1_B 0 QNPN\n" in text and "Q2 OUT Q2_B 0 QNPN\n" in text  # node order C B E from pin_order ["3", "2", "1"]
    assert "C2 OUT Q1_B 72.0n ic=-1\n" in text and "C1 Q1_C Q2_B 72.0n\n" in text  # format_spice_number: the spelling ngspice reads back exactly
    assert "uic" not in text and ".tran" not in text  # the analysis is a runner command, never a netlist card

    results = run_spice_for(ir, ctx.tools, tmp_path)
    ir.validation.extend(results)
    for r in results:
        print(f"  {r.check_id:<16} {r.status:<14} {r.message}")
    summary = ir.validation.latest("spice")
    assert summary.status is S.PASS, summary.message
    assert summary.details["analyses"]["tran"]["command"] == "tran 5.00u 30m 10m uic" and summary.details["analyses"]["tran"]["succeeded"]
    assert summary.details["analyses"]["tran"]["scale"] == "time"

    f_osc = ir.validation.latest("spice.f_osc")
    assert f_osc.status is S.PASS, f_osc.message
    assert f_osc.details["reduce"] == "frequency" and f_osc.details["spice_vector"] == "out" and f_osc.details["unit"] == "Hz"
    assert f_osc.details["nominal"] == pytest.approx(900.2, abs=0.05)
    assert abs(f_osc.details["measured"] - ASTABLE_F_NOMINAL) <= 0.05 * ASTABLE_F_NOMINAL
    assert f_osc.details["tolerance"] == pytest.approx(0.05 * ASTABLE_F_NOMINAL) and f_osc.details["deviation"] == abs(f_osc.details["measured"] - ASTABLE_F_NOMINAL)
    freq = f_osc.details["frequency"]
    assert freq["edges"] >= 15 and 10e-3 <= freq["first_edge_s"] < freq["last_edge_s"] <= 30e-3
    assert freq["vmin"] < 0.5 and 4.5 < freq["vmax"] <= 5.0  # a saturated collector swings between ~V_CE(sat) and Vcc
    assert freq["low"] < freq["mid"] < freq["high"] and freq["mid"] == pytest.approx((freq["vmin"] + freq["vmax"]) / 2)
    assert freq["floor"] == pytest.approx(RELTOL * freq["vmax"] + VNTOL)  # the flatness floor of a voltage vector, recorded with the verdict
    # the mean over the window is what was judged
    assert f_osc.details["measured"] == pytest.approx((freq["edges"] - 1) / (freq["last_edge_s"] - freq["first_edge_s"]))
    assert f_osc.message.startswith("v(OUT) frequency = ") and " Hz, nominal " in f_osc.message
    assert f_osc.tool == "ngspice-shared" and f_osc.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    _evidence_is_real(f_osc)
    assert Path(f_osc.evidence[0].path).name == "astable_hand.tran.raw"
    data = read_results(ir.artifacts[ArtifactKind.SPICE_RESULT].path)
    assert data["analyses"]["tran"]["command"] == "tran 5.00u 30m 10m uic"
    assert min(data["analyses"]["tran"]["result"]["vectors"]["time"]) >= 10e-3  # the start-up transient before 10 ms is not saved


def test_astable_without_oscillation_fails_honestly(tmp_path: Path):
    """The same circuit with its supply at 0 V (the ``uic`` start and C2's ``ic`` kick kept): a flat v(OUT) is FAIL, never PASS.

    Nothing can oscillate without a supply, so the transient's v(OUT) stays
    within the engine's resolution of 0 V and the stage says "no oscillation
    detected". (The symmetric start without ``uic`` / ``ic`` is not what this
    test runs: on ngspice-42 that circuit sometimes starts from round-off, so
    it is not a deterministic no-oscillation case.)
    """
    ir = astable_ir(tmp_path)
    ir.simulation.stimuli[0].value = user_requirement(0.0, "V")
    ctx = _context(tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools=ctx.tools))
    text = Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    assert "C2 OUT Q1_B 72.0n ic=-1\n" in text and "VVIN VCC 0 DC 0\n" in text  # the kick stays, the supply is what is dead
    results = run_spice_for(ir, ctx.tools, tmp_path)
    ir.validation.extend(results)
    assert ir.validation.latest("spice").details["analyses"]["tran"]["command"] == "tran 5.00u 30m 10m uic"
    f_osc = ir.validation.latest("spice.f_osc")
    assert f_osc.status is S.FAIL and f_osc.message.startswith("no oscillation detected") and f_osc.details["repair"] == "human"
    assert "measured" not in f_osc.details and f_osc.details["frequency"]["edges"] < 3
    freq = f_osc.details["frequency"]
    assert abs(freq["vmin"]) <= freq["floor"] and abs(freq["vmax"]) <= freq["floor"]  # flat at 0 V within the engine's resolution
    assert ir.validation.latest("spice").status is S.FAIL and "spice.f_osc" in ir.validation.latest("spice").message


# --------------------------------------------------------------------------- the measurement's flatness floor and the scale vector


def _tran_result(times: list[float], out: list[float], **kw) -> SpiceResult:
    args = dict(
        engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.TRAN, command="tran 1e-5 30m",
        vectors={"time": times, "out": out}, vector_types={"time": "time", "out": "voltage"}, scale="time", n_points=len(times), succeeded=True,
    )
    args.update(kw)
    return SpiceResult(**args)


def _frequency_expectation(vector: str = "v(OUT)", nominal: float = 1000.0) -> Expectation:
    return Expectation(id="f_osc", analysis_id="tran", vector=vector, reduce=Reduce.FREQUENCY, nominal=user_requirement(nominal, "Hz"), tol_rel=user_requirement(0.05), provenance=USER)


def test_numerical_ripple_is_not_an_oscillation():
    """A ripple within ngspice's own resolution (reltol * level + vntol) is flat: no frequency, "no oscillation detected", never PASS.

    The thresholds scale with the swing, so a swing of 2e-12 V around 2.5 V
    used to be counted as a full-scale 1 kHz square wave and judged PASS
    against a 1 kHz expectation.
    """
    times = [i * 1e-5 for i in range(3000)]
    ripple = [2.5 + 1e-12 * (1 if (i // 50) % 2 else -1) for i in range(3000)]  # a 1 kHz pattern of 2e-12 V swing
    edge = rising_edge_frequency(times, ripple)
    assert edge.frequency is None and edge.edges == [] and edge.problem.startswith("no oscillation detected: the waveform is flat at 2.5")
    assert edge.floor == pytest.approx(RELTOL * 2.500000000001 + VNTOL) and "within the engine's resolution" in edge.problem
    # through the stage: no measurement, so no PASS (and no FAIL claiming a measured frequency)
    r = reduce_expectation(_tran_result(times, ripple), _frequency_expectation(), "out")
    assert r.measured is None and r.problem.startswith("no oscillation detected") and r.extra["edges"] == 0 and r.extra["floor"] == edge.floor
    assert reduce_result(_tran_result(times, ripple), _frequency_expectation(), "out")[0] is None
    # one-ulp representation noise around a level does not even reach the detector's arithmetic
    noise = [2.5 + (4.4e-16 if (i * 7919) % 3 == 0 else 0.0) for i in range(3000)]
    assert rising_edge_frequency(times, noise).problem.startswith("no oscillation detected: the waveform is flat")
    # a swing just above the floor is measured; a real square wave is unaffected
    small = [2.5 + 0.01 * (1 if (i // 50) % 2 else -1) for i in range(3000)]  # 20 mV swing: 8 x the 2.5 mV floor
    assert rising_edge_frequency(times, small).frequency == pytest.approx(1000.0, rel=1e-9)
    square = [5.0 if (i // 50) % 2 else 0.0 for i in range(3000)]
    assert rising_edge_frequency(times, square).frequency == pytest.approx(1000.0, rel=1e-9) and rising_edge_frequency(times, square).floor == pytest.approx(RELTOL * 5.0 + VNTOL)


def test_the_flatness_floor_follows_the_vector_kind():
    """A current vector is measured in amperes: its absolute floor is ``abstol`` (1e-12 A), a voltage's is ``vntol`` (1e-6 V)."""
    times = [i * 1e-5 for i in range(3000)]
    nano = [1e-9 * (1 if (i // 50) % 2 else 0) for i in range(3000)]  # a 1 nA square wave
    assert rising_edge_frequency(times, nano, abs_floor=ABSTOL).frequency == pytest.approx(1000.0, rel=1e-9)
    assert rising_edge_frequency(times, nano).problem.startswith("no oscillation detected")  # the same numbers as volts are within vntol
    res = _tran_result(times, nano, vectors={"time": times, "vvin#branch": nano}, vector_types={"time": "time", "vvin#branch": "current"})
    assert reduce_expectation(res, _frequency_expectation("i(VIN)"), "vvin#branch").measured == pytest.approx(1000.0, rel=1e-9)
    assert reduce_expectation(_tran_result(times, nano), _frequency_expectation(), "out").measured is None
    with pytest.raises(ValueError):
        rising_edge_frequency(times, nano, abs_floor=-1.0)


def test_a_plot_without_its_scale_vector_is_reported_as_the_scale():
    """A result whose scale vector is missing names the scale in the problem, never the expectation's vector (which is present)."""
    res = _tran_result([0.0], [0.0], vectors={"out": [0.0, 1.0, 0.0, 1.0]}, n_points=4)
    r = reduce_expectation(res, _frequency_expectation(), "out")
    assert r.measured is None and r.problem == "scale vector not produced: 'time' is not in the tran plot (vectors: ['out'])"
    dc = SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.DC, command="dc v 0 2 1", vectors={"x": [0.0, 1.5, 3.0]}, scale="v-sweep", n_points=3, succeeded=True)
    at = Expectation(id="a", analysis_id="dc", vector="v(X)", reduce=Reduce.AT, at=user_requirement(1.0, "V"), nominal=user_requirement(1.5, "V"), tol_abs=user_requirement(0.1, "V"), provenance=USER)
    assert reduce_expectation(dc, at, "x").problem == "scale vector not produced: 'v-sweep' is not in the dc plot (vectors: ['x'])"
    # an absent expectation vector is still reported as such
    assert reduce_expectation(_tran_result([0.0, 1.0], [0.0, 1.0]), _frequency_expectation("v(NOPE)"), "nope").problem.startswith("vector not produced: 'nope' is not in the tran plot")

