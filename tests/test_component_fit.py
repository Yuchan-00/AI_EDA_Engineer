"""``component.fit`` (electrical stress at the SPICE op) and ``domain.power.thermal`` (Tj = Ta + P * theta_ja).

Engine-free tests pin the refusal rules on a hand-made results dict (what may
be judged, what is recorded and not judged); the ngspice-backed tests run the
divider fixture through the orchestrator up to SPICE and check the verdicts
against real evidence (netlist hash, rawfile / results.json, freshness) - no
KiCad libraries are needed (the component stage fails on the missing symbols
and the pipeline continues; neither validator reads a KiCad library).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_eda.agents import AgentContext
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitDomain,
    CircuitIR,
    Component,
    LibraryRef,
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
    Topology,
    ValidationResult,
    ValidationStatus,
    assumption,
    authoritative,
    llm_generated,
    user_requirement,
)
from ai_eda.tools.calc import CALC_VERSION, junction_temperature, power_from_voltage_resistance, recompute_parameters
from ai_eda.tools.calc.recompute import unit_key
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.evidence import NO_OP, FreshSpiceRun, fresh_spice_run, pin_net
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation import domain as domain_module
from ai_eda.validation import fit as fit_module
from ai_eda.validation.domain import NO_THERMAL_KEYS, THERMAL_TOOL, THERMAL_VERSION
from ai_eda.validation.fit import DERATING, FIT_CHECK, FIT_CRITERIA, FIT_VERSION, MULTI_TERMINAL, component_stress
from ai_eda.workflow import Orchestrator, PipelineState, Stage
from tests.conftest import DS
from tests.fixtures_kicad import RESISTOR_DS, divider_with_connector_ir
from tests.test_simulation_stage import _change_r1_to_20k, _evidence_is_real, _recalculate_expectations

S = ValidationStatus
LIB = KicadLibrary()
runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
FIT = default_registry.get(FIT_CHECK)
THERMAL = default_registry.get(THERMAL_TOOL)
AUTH = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")


# --------------------------------------------------------------------------- helpers


def _ctx(tmp_path: Path) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": runner}, answers={"application": "bench", "jurisdiction": "EU"})


def _vctx(tmp_path: Path) -> ValidationContext:
    return ValidationContext(workdir=tmp_path)


def _run(ir: CircuitIR, tmp_path: Path) -> PipelineState:
    state = Orchestrator(_ctx(tmp_path)).run(ir, stop_after=Stage.SPICE)
    for o in state.outcomes:
        print(f"{o.stage:<24} {o.status:<14} {o.message[:300]}")
    return state


def _divider(tmp_path: Path, *, connector: bool = True) -> CircuitIR:
    """The divider fixture; without ``connector`` J1 (excluded from SPICE) and its pins are dropped, so every part is simulated."""
    ir = divider_with_connector_ir(tmp_path, LIB)
    if not connector:
        ir.components = [c for c in ir.components if c.ref != "J1"]
        for net in ir.nets:
            net.pins = [p for p in net.pins if p.component_ref != "J1"]
        if ir.pcb is not None:
            ir.pcb.placements = [p for p in ir.pcb.placements if p.component_ref != "J1"]
    return ir


def _rate(ir: CircuitIR, refs: tuple[str, ...] = ("R1", "R2"), *, v_max: float | None = 50.0, power_rating: float | None = 0.1) -> None:
    for ref in refs:
        c = ir.component(ref)
        if v_max is not None:
            c.electrical["v_max"] = authoritative(v_max, RESISTOR_DS, "V")
        if power_rating is not None:
            c.electrical["power_rating"] = authoritative(power_rating, RESISTOR_DS, "W")


def _thermal(ir: CircuitIR, refs: tuple[str, ...] = ("R1", "R2"), *, theta_ja: float = 200.0, t_j_max: float = 155.0, unit: str = "K/W") -> None:
    for ref in refs:
        c = ir.component(ref)
        c.electrical["theta_ja"] = authoritative(theta_ja, RESISTOR_DS, unit)
        c.electrical["t_j_max"] = authoritative(t_j_max, RESISTOR_DS, "degC")


def _pin(n: str) -> Pin:
    return Pin(number=n, name=f"~{n}", electrical_type=PinElectricalType.PASSIVE, provenance=AUTH)


def _part(ref: str, n_pins: int, binding: SpiceBinding, electrical: dict[str, Any] | None = None) -> Component:
    return Component(
        ref=ref, value=ref, description="part", pins=[_pin(str(i)) for i in range(1, n_pins + 1)],
        symbol=LibraryRef(library="Device", name="R", verified=True), footprint=LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric", verified=True),
        electrical=dict(electrical or {}), provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test"), spice=binding,
    )


def _fake_run(ir: CircuitIR, vectors: dict[str, list[float]], *, temperature_c: float = 27.0, spice_status: S = S.PASS, assumptions: list[str] | None = None,
              analyses: dict[str, Any] | None = None) -> FreshSpiceRun:
    """A run object as ``fresh_spice_run`` would return it, from a hand-made results dict (no engine, nothing on disk)."""
    data = {
        "format": "2", "engine": "fake", "engine_version": "fake-0", "engine_info": {"build": "b", "codemodels_loaded": True, "settings_hash": "sha256:0", "dll_path": "x"},
        "conditions": {"temperature_c": temperature_c, "temperature_source": "test"},
        "netlist_hash": "sha256:netlist", "netlist_report": {"assumptions": list(assumptions or [])},
        "analyses": analyses if analyses is not None else {"op": {"kind": "op", "command": "op", "result": {"succeeded": True, "unverifiable": None, "vectors": vectors}}},
    }
    art = ArtifactRef(kind=ArtifactKind.SPICE_NETLIST, path="/nonexistent/x.cir", content_hash="sha256:netlist", generated_from_ir_hash=ir.content_hash())
    res = ArtifactRef(kind=ArtifactKind.SPICE_RESULT, path="/nonexistent/results.json", content_hash="sha256:results", generated_from_ir_hash=ir.content_hash())
    spice = ValidationResult(check_id="spice", status=spice_status, tool="fake", artifact_hash="sha256:netlist")
    return FreshSpiceRun(data=data, netlist=art, results=res, spice=spice)


def _bench_ir(tmp_path: Path, *parts: Component, nets: dict[str, list[tuple[str, str]]], domains=(CircuitDomain.ANALOG,), temperature_c=None) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="bench", name="bench", workdir=str(tmp_path)))
    ir.topology = Topology(name="bench", domains=list(domains), provenance=USER)
    ir.components = list(parts)
    ir.nets = [
        Net(name=name, kind=NetKind.GROUND if name == "GND" else NetKind.SIGNAL, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=USER)
        for name, pins in nets.items()
    ]
    if temperature_c is not None:
        from ai_eda.ir import SimulationSetup

        ir.simulation = SimulationSetup(temperature_c=temperature_c)
    return ir


# --------------------------------------------------------------------------- engine-free: calculators and units


def test_calculators_and_thermal_units():
    p = power_from_voltage_resistance(user_requirement(6.0, "V"), authoritative(1e4, DS, "ohm"), ("R1.v_across", "R1.resistance"))
    assert p.value == pytest.approx(3.6e-3) and p.unit == "W" and p.provenance.tool == "calc.power.P_VR" and p.provenance.tool_version == CALC_VERSION == "0.6"
    assert p.provenance.inputs == {"v": "R1.v_across", "r": "R1.resistance"}
    with pytest.raises(ZeroDivisionError):
        power_from_voltage_resistance(user_requirement(6.0), user_requirement(0.0))
    tj = junction_temperature(user_requirement(85.0, "degC"), p, authoritative(200.0, DS, "K/W"))
    assert tj.value == pytest.approx(85.72) and tj.unit == "degC" and tj.provenance.tool == "calc.thermal.T_j" and list(tj.provenance.inputs) == ["t_a", "p", "theta_ja"]
    for bad in ((user_requirement(85.0), user_requirement(-1.0), user_requirement(200.0)), (user_requirement(85.0), user_requirement(1.0), user_requirement(-200.0))):
        with pytest.raises(ValueError):
            junction_temperature(*bad)
    # a thermal resistance written degC/W is the K/W the calculator role expects (the recompute compares by this key)
    assert unit_key("degC/W") == unit_key("°C/W") == unit_key("℃/W") == unit_key("K/W") == unit_key(" k/w ") == "k/w"
    assert unit_key("°C") == unit_key("℃") == unit_key("deg C") == unit_key("degC") == "degc" and unit_key("W") == "w"


# --------------------------------------------------------------------------- engine-free: the fresh-run helper


def test_no_run_attached_every_op_validator_refuses_with_its_need(divider_ir: CircuitIR, tmp_path: Path):
    assert fresh_spice_run(divider_ir, needs="X needs an op") == "X needs an op; no tool-backed spice result attached"
    [fit] = FIT.validate(divider_ir, _vctx(tmp_path))
    assert fit.status is S.NOT_VERIFIED and fit.check_id == FIT_CHECK and fit.tool == FIT_CHECK and fit.tool_version == FIT_VERSION
    assert fit.message == "component fit needs an operating point from SPICE; no tool-backed spice result attached"
    assert fit.details["criteria_evaluated"] == ["electrical stress"] and fit.details["criteria_not_evaluated"] == list(FIT_CRITERIA[1:])
    assert fit.artifact_hash is None and fit.evidence == [] and fit.ir_hash == divider_ir.content_hash()
    # thermal runs for every design: outside the POWER domain a design without thermal keys claims nothing ...
    assert CircuitDomain.POWER not in divider_ir.topology.domains
    [th] = THERMAL.validate(divider_ir, _vctx(tmp_path))
    assert th.status is S.NOT_APPLICABLE and th.message.startswith(NO_THERMAL_KEYS) and "not in the POWER domain" in th.message
    divider_ir.topology.domains.append(CircuitDomain.POWER)
    [th] = THERMAL.validate(divider_ir, _vctx(tmp_path))
    assert th.status is S.NOT_VERIFIED and th.message == NO_THERMAL_KEYS and th.tool == THERMAL_TOOL and th.tool_version == THERMAL_VERSION
    # ... and with a key present it is judged whatever the domain
    divider_ir.topology.domains.remove(CircuitDomain.POWER)
    _thermal(divider_ir, ("R1",))
    [th] = THERMAL.validate(divider_ir, _vctx(tmp_path))
    assert th.status is S.NOT_VERIFIED and th.message == "thermal analysis needs an operating point from SPICE; no tool-backed spice result attached"
    # nothing to judge is NOT_APPLICABLE, never PASS
    empty = CircuitIR(project=ProjectMeta(id="e", name="e", workdir=str(tmp_path)))
    assert FIT.validate(empty, _vctx(tmp_path))[0].status is S.NOT_APPLICABLE


def test_op_selection_refuses_unsucceeded_and_unverifiable_results(divider_ir: CircuitIR):
    ok = {"kind": "op", "command": "op", "result": {"succeeded": True, "unverifiable": None, "vectors": {"vin": [12.0], "vout": [6.0]}}}
    failed = {**ok, "result": {**ok["result"], "succeeded": False}}
    blocked = {**ok, "result": {**ok["result"], "unverifiable": "no place for the rawfile"}}
    dc = {"kind": "dc", "command": "dc vvin 0 12 1", "result": {"succeeded": True, "unverifiable": None, "vectors": {"vout": [0.0, 6.0]}}}
    run = _fake_run(divider_ir, {}, analyses={"dc": dc, "op_bad": failed, "op_blocked": blocked, "op": ok})
    assert run.op() == ("op", ok["result"])  # the first *usable* op, whatever came before it
    assert run.node_voltage(divider_ir, "VOUT") == 6.0 and run.node_voltage(divider_ir, "GND") == 0.0
    for analyses in ({"op_bad": failed}, {"op_blocked": blocked}, {"dc": dc}, {}):
        r = _fake_run(divider_ir, {}, analyses=analyses)
        assert r.op() == NO_OP and r.node_voltage(divider_ir, "VOUT") == NO_OP
    assert run.temperature_c == 27.0 and run.engine == "fake" and run.assumptions == [] and run.analysis_ids == ["dc", "op", "op_bad", "op_blocked"]
    # a net without a vector, a non-finite sample and an unknown net are reasons, never numbers
    r = _fake_run(divider_ir, {"vin": [12.0], "vout": [float("inf")]})
    assert r.node_voltage(divider_ir, "VIN") == 12.0
    assert "not finite" in r.node_voltage(divider_ir, "VOUT") and "not in the IR" in r.node_voltage(divider_ir, "NOPE")
    assert "no op vector" in _fake_run(divider_ir, {"vin": [12.0]}).node_voltage(divider_ir, "VOUT")
    assert pin_net(divider_ir, "R1", "2").name == "VOUT" and pin_net(divider_ir, "R1", "9") is None and pin_net(divider_ir, "R9", "1") is None
    # evidence: results.json always, the rawfile first when the analysis wrote one
    assert [e.description for e in run.evidence("op")] == ["spice results.json"]
    with_raw = _fake_run(divider_ir, {}, analyses={"op": {**ok, "result": {**ok["result"], "raw_output_path": "/x/op.raw", "raw_output_hash": "sha256:raw"}}})
    assert [e.path for e in with_raw.evidence("op")] == ["/x/op.raw", "/nonexistent/results.json"]


# --------------------------------------------------------------------------- engine-free: what may be judged


def _judge(monkeypatch, ir: CircuitIR, run: FreshSpiceRun, validator=None) -> ValidationResult:
    """Run a validator with ``fresh_spice_run`` replaced by the hand-made run (the freshness chain is tested with the engine)."""
    monkeypatch.setattr(fit_module, "fresh_spice_run", lambda ir, needs: run)
    monkeypatch.setattr(domain_module, "fresh_spice_run", lambda ir, needs: run)
    [res] = (validator or FIT).validate(ir, ValidationContext(workdir=Path(ir.project.workdir)))
    print(res.status, res.message)
    return res


def test_only_two_terminal_r_and_c_are_judged_on_v_max(monkeypatch, tmp_path: Path):
    """A multi-terminal part, an L, a D and a source carry a v_max: none is judged, the voltages are recorded, the reason is stated."""
    v_max = {"v_max": authoritative(50.0, DS, "V")}
    parts = [
        _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=authoritative(1e3, DS, "ohm"), provenance=AUTH), {**v_max, "resistance": authoritative(1e3, DS, "ohm")}),
        _part("C1", 2, SpiceBinding(device=SpiceDevice.C, value=authoritative(1e-6, DS, "F"), provenance=AUTH), v_max),
        _part("L1", 2, SpiceBinding(device=SpiceDevice.L, value=authoritative(1e-3, DS, "H"), provenance=AUTH), v_max),
        _part("D1", 2, SpiceBinding(device=SpiceDevice.D, model_name="dx", provenance=AUTH), v_max),
        _part("V1", 2, SpiceBinding(device=SpiceDevice.V, value=user_requirement(12.0, "V"), provenance=USER), v_max),
        _part("Q1", 3, SpiceBinding(device=SpiceDevice.Q, model_name="qx", pin_order=["1", "2", "3"], provenance=AUTH), v_max),
        _part("U1", 3, SpiceBinding(device=SpiceDevice.X, model_name="sub", pin_order=["1", "2", "3"], provenance=AUTH), v_max),
        _part("J1", 2, SpiceBinding(exclude=True, exclude_reason="connector", provenance=USER), v_max),
    ]
    nets = {"VIN": [("R1", "1"), ("V1", "1"), ("L1", "1"), ("D1", "1"), ("Q1", "1"), ("U1", "1"), ("J1", "1")],
            "VOUT": [("R1", "2"), ("C1", "1"), ("L1", "2"), ("D1", "2"), ("Q1", "2"), ("U1", "2")],
            "VNEG": [("Q1", "3"), ("U1", "3")],
            "GND": [("C1", "2"), ("V1", "2"), ("J1", "2")]}
    ir = _bench_ir(tmp_path, *parts, nets=nets)
    run = _fake_run(ir, {"vin": [40.0], "vout": [-30.0], "vneg": [-40.0]})
    res = _judge(monkeypatch, ir, run)
    comp = res.details["components"]
    # R1 and C1: judged (70 V across R1 > 50 V is a FAIL; 30 V across C1 <= 50 V is a PASS)
    assert comp["R1"]["voltage"]["status"] == "FAIL" and comp["R1"]["voltage"]["reason"] == "70 V > 50 V (v_max)" and comp["R1"]["voltage"]["repair"] == "human"
    assert comp["R1"]["voltage"]["nets"] == {"1": "VIN", "2": "VOUT"} and comp["R1"]["voltage"]["pins"] == {"1": 40.0, "2": -30.0} and comp["R1"]["voltage"]["derating"] == DERATING
    assert comp["C1"]["voltage"] == {**comp["C1"]["voltage"], "status": "PASS", "v": 30.0, "limit": 50.0, "factor": 1.0}
    # everything else: NOT_VERIFIED with the reason and the recorded voltages, never a proxy verdict
    assert comp["L1"]["voltage"]["status"] == "NOT_VERIFIED" and "inductor is 0 by definition" in comp["L1"]["voltage"]["reason"] and comp["L1"]["voltage"]["v_across"] == 70.0
    assert comp["D1"]["voltage"]["status"] == "NOT_VERIFIED" and "reverse rating" in comp["D1"]["voltage"]["reason"]
    assert comp["V1"]["voltage"]["status"] == "NOT_VERIFIED" and "source has no voltage rating" in comp["V1"]["voltage"]["reason"]
    for ref in ("Q1", "U1"):  # |v(pin)| <= 50 for every pin, yet V(1)-V(3) = 80 V: a per-pin proxy would have said PASS
        assert comp[ref]["voltage"]["status"] == "NOT_VERIFIED" and comp[ref]["voltage"]["reason"] == MULTI_TERMINAL
        assert comp[ref]["voltage"]["pins"] == {"1": 40.0, "2": -30.0, "3": -40.0}
    assert comp["J1"]["voltage"] == {"status": "NOT_VERIFIED", "reason": "excluded from SPICE (connector): no operating point"}
    assert comp["J1"]["power"]["reason"] == comp["J1"]["voltage"]["reason"]
    # power: only the plain R gets a dissipation (4.9 W > no rating -> NOT_VERIFIED naming the missing rating, the number kept)
    assert comp["R1"]["power"]["status"] == "NOT_VERIFIED" and comp["R1"]["power"]["reason"].startswith("no power_rating rating") and comp["R1"]["power"]["dissipation"]["value"] == pytest.approx(4.9)
    for ref, dev in (("C1", "C"), ("L1", "L"), ("D1", "D"), ("V1", "V"), ("Q1", "Q"), ("U1", "X")):
        assert comp[ref]["power"]["status"] == "NOT_VERIFIED" and comp[ref]["power"]["reason"] == f"dissipation of a {dev} device is not computed: ngspice records no element currents in the op"
    assert res.status is S.FAIL and res.details["repair"] == "human" and "R1 FAIL (voltage: 70 V > 50 V (v_max)" in res.message
    assert "nominal values, operating point only, one temperature (27 degC)" in res.message and "not evaluated: safety, regulatory" in res.message


def test_power_needs_a_plain_r_binding_and_an_authoritative_resistance(monkeypatch, tmp_path: Path):
    rating = {"power_rating": authoritative(0.25, DS, "W"), "v_max": authoritative(50.0, DS, "V")}
    r = authoritative(1e3, DS, "ohm")
    parts = [
        _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, params={"m": user_requirement(2.0)}, provenance=AUTH), {**rating, "resistance": r}),
        _part("R2", 2, SpiceBinding(device=SpiceDevice.R, value=r, params={"tc1": user_requirement(1e-3), "tc2": user_requirement(0.0)}, provenance=AUTH), {**rating, "resistance": r}),
        _part("R3", 2, SpiceBinding(device=SpiceDevice.R, model_name="rmod", model_card=user_requirement(".model rmod r (rsh=1)"), provenance=AUTH), {**rating, "resistance": r}),
        _part("R4", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), rating),  # no electrical resistance
        _part("R5", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {**rating, "resistance": assumption(1e3, "guessed", "ohm")}),
        _part("R6", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {**rating, "resistance": r}),
        # a value AND a model: ngspice simulates the model's resistance, not the value - the value alone must not make it a plain R
        _part("R7", 2, SpiceBinding(device=SpiceDevice.R, value=r, model_name="rmod", provenance=AUTH), {**rating, "resistance": r}),
        _part("R8", 2, SpiceBinding(device=SpiceDevice.R, value=r, model_name="rmod", model_card=user_requirement(".model rmod r (rsh=1)"), provenance=AUTH), {**rating, "resistance": r}),
        _part("R9", 2, SpiceBinding(device=SpiceDevice.R, value=r, model_card=user_requirement(".model rmod r (rsh=1)"), provenance=AUTH), {**rating, "resistance": r}),
    ]
    nets = {"VIN": [(f"R{i}", "1") for i in range(1, 10)], "GND": [(f"R{i}", "2") for i in range(1, 10)]}
    ir = _bench_ir(tmp_path, *parts, nets=nets)
    res = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}))
    comp = res.details["components"]
    assert comp["R1"]["power"]["status"] == "NOT_VERIFIED" and comp["R1"]["power"]["reason"].startswith("dissipation not computed: R1 carries SPICE params ['m']")
    assert comp["R2"]["power"]["reason"].startswith("dissipation not computed: R2 carries SPICE params ['tc1', 'tc2']") and "not electrical[resistance]" in comp["R2"]["power"]["reason"]
    assert comp["R3"]["power"]["reason"].startswith("dissipation not computed: R3 carries model 'rmod' and no value")
    assert comp["R4"]["power"]["reason"].startswith("electrical[resistance] is missing") and "binding, not the shipped part" in comp["R4"]["power"]["reason"]
    assert comp["R5"]["power"]["reason"] == "resistance is assumption (needs confirmation)"
    assert comp["R6"]["power"]["status"] == "PASS" and comp["R6"]["power"]["p"] == pytest.approx(0.1) and comp["R6"]["power"]["reason"] == "100 mW <= 250 mW (power_rating)"
    assert comp["R6"]["power"]["dissipation"] == {**comp["R6"]["power"]["dissipation"], "tool": "calc.power.P_VR", "tool_version": CALC_VERSION, "unit": "W", "inputs": {"v": "R6.v_across", "r": "R6.resistance"}}
    for ref in ("R7", "R8"):
        assert comp[ref]["power"]["status"] == "NOT_VERIFIED" and "dissipation" not in comp[ref]["power"], ref
        assert comp[ref]["power"]["reason"] == f"dissipation not computed: {ref} carries model 'rmod'; ngspice simulates a resistance that is not electrical[resistance]"
    assert comp["R9"]["power"]["status"] == "NOT_VERIFIED" and comp["R9"]["power"]["reason"].startswith("dissipation not computed: R9 carries model card (without model_name)")
    # the voltage criterion does not care about params: every R sees 10 V <= 50 V
    assert all(comp[f"R{i}"]["voltage"]["status"] == "PASS" for i in range(1, 10)) and comp["R6"]["status"] == "PASS" and res.status is S.NOT_VERIFIED
    # thermal on the same op: R1 (params), R7 / R8 (a model) and R9 (a card) never reach a Tj, R6 does
    ir.topology.domains.append(CircuitDomain.POWER)
    _thermal(ir, ("R1", "R6", "R7", "R8", "R9"), theta_ja=100.0, t_j_max=155.0)
    ir.simulation = __import__("ai_eda.ir", fromlist=["SimulationSetup"]).SimulationSetup(temperature_c=user_requirement(27.0, "degC"))
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    tc = th.details["components"]
    assert tc["R1"]["thermal"]["status"] == "NOT_VERIFIED" and tc["R1"]["thermal"]["reason"].startswith("dissipation not computed: R1 carries SPICE params ['m']")
    assert tc["R6"]["thermal"]["status"] == "PASS" and tc["R6"]["thermal"]["t_j"] == pytest.approx(37.0) and tc["R6"]["thermal"]["calc"]["tool"] == "calc.thermal.T_j"
    for ref in ("R7", "R8"):
        assert tc[ref]["thermal"]["status"] == "NOT_VERIFIED" and tc[ref]["thermal"]["reason"].startswith(f"dissipation not computed: {ref} carries model 'rmod'") and "t_j" not in tc[ref]["thermal"]
    assert tc["R9"]["thermal"]["status"] == "NOT_VERIFIED" and "carries model card (without model_name)" in tc["R9"]["thermal"]["reason"]
    assert tc["R2"]["thermal"]["reason"] == "no theta_ja / t_j_max (datasheet facts: theta_ja in K/W, t_j_max in degC)" and th.status is S.NOT_VERIFIED


def test_a_stress_exactly_at_its_rating_is_within_the_limit(monkeypatch, tmp_path: Path):
    """The comparisons are ``>``: v == v_max and p == power_rating are PASS (100 % of the rating, no derating), one unit more is FAIL."""
    r = authoritative(100.0, DS, "ohm")  # 10 V across 100 ohm: exactly 1 W
    parts = [
        _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": authoritative(10.0, DS, "V"), "power_rating": authoritative(1.0, DS, "W")}),
        _part("R2", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": authoritative(9.999, DS, "V"), "power_rating": authoritative(0.999, DS, "W")}),
    ]
    ir = _bench_ir(tmp_path, *parts, nets={"VIN": [("R1", "1"), ("R2", "1")], "GND": [("R1", "2"), ("R2", "2")]})
    comp = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]})).details["components"]
    assert comp["R1"]["voltage"] == {**comp["R1"]["voltage"], "status": "PASS", "v": 10.0, "limit": 10.0, "reason": "10 V <= 10 V (v_max)"}
    assert comp["R1"]["power"] == {**comp["R1"]["power"], "status": "PASS", "p": 1.0, "limit": 1.0, "reason": "1 W <= 1 W (power_rating)"} and comp["R1"]["status"] == "PASS"
    assert comp["R2"]["voltage"]["status"] == "FAIL" and comp["R2"]["power"]["status"] == "FAIL" and comp["R2"]["power"]["reason"] == "1 W > 999 mW (power_rating)"


def test_ratings_must_be_authoritative_numbers_with_the_expected_unit(monkeypatch, tmp_path: Path):
    r = authoritative(1e3, DS, "ohm")
    parts = [
        _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": assumption(50.0, "guess", "V"), "power_rating": llm_generated(1.0, "m", "W")}),
        _part("R2", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": authoritative(50.0, DS, "A"), "power_rating": authoritative(1.0, DS)}),
        _part("R3", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": user_requirement(50.0, "V"), "power_rating": user_requirement(1.0, "W")}),
    ]
    ir = _bench_ir(tmp_path, *parts, nets={"VIN": [("R1", "1"), ("R2", "1"), ("R3", "1")], "GND": [("R1", "2"), ("R2", "2"), ("R3", "2")]})
    res = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}))
    comp = res.details["components"]
    assert comp["R1"]["voltage"]["reason"] == "v_max is assumption (needs confirmation)" and comp["R1"]["power"]["reason"] == "power_rating is llm_generated (needs confirmation)"
    assert comp["R2"]["voltage"]["reason"] == "v_max carries unit 'A', expected V" and comp["R2"]["power"]["reason"] == "power_rating carries unit None, expected W"
    assert comp["R3"]["status"] == "PASS" and comp["R1"]["status"] == comp["R2"]["status"] == "NOT_VERIFIED" and res.status is S.NOT_VERIFIED
    assert comp["R1"]["voltage"]["v_across"] == 10.0 and comp["R1"]["power"]["dissipation"]["value"] == pytest.approx(0.1)  # recorded, not judged


def test_a_would_be_pass_is_not_verified_on_assumptions_or_a_failed_run_but_a_fail_stays(monkeypatch, tmp_path: Path):
    r = authoritative(1e3, DS, "ohm")
    parts = [
        _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": authoritative(50.0, DS, "V"), "power_rating": authoritative(1.0, DS, "W")}),
        _part("R2", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r, "v_max": authoritative(5.0, DS, "V"), "power_rating": authoritative(1.0, DS, "W")}),
    ]
    ir = _bench_ir(tmp_path, *parts, nets={"VIN": [("R1", "1"), ("R2", "1")], "GND": [("R1", "2"), ("R2", "2")]})
    res = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, assumptions=["R2 value: guessed"]))
    comp = res.details["components"]
    assert comp["R1"]["voltage"]["status"] == "NOT_VERIFIED" and "within limit (10 V <= 50 V (v_max)) but the netlist rests on assumption(s) ['R2 value: guessed']" in comp["R1"]["voltage"]["reason"]
    assert comp["R2"]["voltage"]["status"] == "FAIL" and res.status is S.FAIL
    res = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, spice_status=S.FAIL))
    comp = res.details["components"]
    assert comp["R1"]["voltage"]["status"] == "NOT_VERIFIED" and comp["R1"]["voltage"]["reason"].endswith("but computed on an op of a run whose spice summary is FAIL")
    assert comp["R1"]["power"]["status"] == "NOT_VERIFIED" and comp["R1"]["power"]["p"] == pytest.approx(0.1)  # the numbers stay
    assert comp["R2"]["voltage"]["status"] == "FAIL" and res.status is S.FAIL and res.message.startswith("stress computed on an op of a run whose spice summary is FAIL; ")
    assert res.details["spice_status"] == "FAIL"
    ir.components.pop()  # only the passing part left: the run's FAIL alone makes the whole result NOT_VERIFIED
    res = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, spice_status=S.FAIL))
    assert res.status is S.NOT_VERIFIED and "R1 NOT_VERIFIED" in res.message


def test_thermal_ambient_rules(monkeypatch, tmp_path: Path):
    from ai_eda.ir import SimulationSetup

    r = authoritative(1e3, DS, "ohm")
    part = _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r})
    ir = _bench_ir(tmp_path, part, nets={"VIN": [("R1", "1")], "GND": [("R1", "2")]}, domains=(CircuitDomain.POWER,))
    _thermal(ir, ("R1",), theta_ja=100.0, t_j_max=40.0, unit="degC/W")  # a datasheet spelling of K/W
    # no stated ambient: the op ran at ngspice's default and nothing says that is the design's ambient
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.status is S.NOT_VERIFIED and th.details["components"]["R1"]["thermal"]["reason"].startswith("no ambient: simulation.temperature_c is not set (the op ran at 27 degC")
    assert th.details["components"]["R1"]["thermal"]["p"] == pytest.approx(0.1) and th.details["t_a"] is None
    # an ambient that is not the op's temperature is another op
    ir.simulation = SimulationSetup(temperature_c=user_requirement(85.0, "degC"))
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, temperature_c=27.0), THERMAL)
    assert th.status is S.NOT_VERIFIED and th.details["components"]["R1"]["thermal"]["reason"] == "the op ran at 27 degC, not the stated ambient 85 degC"
    # an assumed ambient is not evidence
    ir.simulation = SimulationSetup(temperature_c=assumption(27.0, "room", "degC"))
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.details["components"]["R1"]["thermal"]["reason"] == "simulation.temperature_c is assumption (needs confirmation)"
    # the stated ambient is the op's: Tj = 27 + 0.1 * 100 = 37 degC <= 40 -> PASS; t_j_max 36 -> FAIL (human)
    ir.simulation = SimulationSetup(temperature_c=user_requirement(27.0, "degC"))
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    t = th.details["components"]["R1"]["thermal"]
    assert th.status is S.PASS and t == {**t, "status": "PASS", "t_a": 27.0, "theta_ja": 100.0, "t_j_max": 40.0} and t["t_j"] == pytest.approx(37.0)
    assert t["calc"]["inputs"] == {"t_a": "simulation.temperature_c", "p": "R1.p", "theta_ja": "R1.theta_ja"} and t["calc"]["tool_version"] == CALC_VERSION
    assert "ambient 27 degC (simulation.temperature_c, the op's temperature)" in th.message and "Tj = Ta + P * theta_ja (calc.thermal.T_j)" in th.message
    ir.component("R1").electrical["t_j_max"] = authoritative(36.0, RESISTOR_DS, "degC")
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.status is S.FAIL and th.details["repair"] == "human" and th.details["components"]["R1"]["thermal"]["reason"] == "Tj 37 degC > t_j_max 36 degC"
    # exactly at the rating (Tj == t_j_max) is within the limit: the comparison is ``>``
    ir.component("R1").electrical["t_j_max"] = authoritative(37.0, RESISTOR_DS, "degC")
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.status is S.PASS and th.details["components"]["R1"]["thermal"]["reason"] == "Tj 37 degC <= t_j_max 37 degC" and th.details["components"]["R1"]["thermal"]["t_j"] == 37.0
    # the same gate as component.fit: a would-be PASS is NOT_VERIFIED when the netlist rests on an assumption ...
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, assumptions=["R1 value: guessed"]), THERMAL)
    t = th.details["components"]["R1"]["thermal"]
    assert th.status is S.NOT_VERIFIED and t["status"] == "NOT_VERIFIED" and t["t_j"] == 37.0  # the numbers stay
    assert t["reason"] == "within limit (Tj 37 degC <= t_j_max 37 degC) but the netlist rests on assumption(s) ['R1 value: guessed'] that nobody confirmed"
    assert th.details["assumptions"] == ["R1 value: guessed"] and "R1 NOT_VERIFIED (thermal: within limit" in th.message and "repair" not in th.details
    # ... or the run's spice summary is FAIL ...
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}, spice_status=S.FAIL), THERMAL)
    t = th.details["components"]["R1"]["thermal"]
    assert th.status is S.NOT_VERIFIED and t["status"] == "NOT_VERIFIED" and t["reason"] == "within limit (Tj 37 degC <= t_j_max 37 degC) but computed on an op of a run whose spice summary is FAIL"
    assert th.message.startswith("computed on an op of a run whose spice summary is FAIL; ") and th.details["spice_status"] == "FAIL"
    # ... while an over-limit Tj stays FAIL whatever the run says
    ir.component("R1").electrical["t_j_max"] = authoritative(36.0, RESISTOR_DS, "degC")
    for run in (_fake_run(ir, {"vin": [10.0]}, spice_status=S.FAIL), _fake_run(ir, {"vin": [10.0]}, assumptions=["R1 value: guessed"])):
        th = _judge(monkeypatch, ir, run, THERMAL)
        assert th.status is S.FAIL and th.details["repair"] == "human" and th.details["components"]["R1"]["thermal"]["reason"] == "Tj 37 degC > t_j_max 36 degC"
    ir.component("R1").electrical["t_j_max"] = authoritative(40.0, RESISTOR_DS, "degC")
    # an ambient in another unit is not judged as degC, even at the op's number
    ir.simulation = SimulationSetup(temperature_c=user_requirement(27.0, "K"))
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    t = th.details["components"]["R1"]["thermal"]
    assert th.status is S.NOT_VERIFIED and t["status"] == "NOT_VERIFIED" and t["reason"] == "simulation.temperature_c carries unit 'K', expected degC" and "t_j" not in t
    assert th.details["t_a"] is None and "expected degC" in th.message
    ir.simulation = SimulationSetup(temperature_c=user_requirement(27.0, "degC"))
    # a thermal key with the wrong unit family or provenance
    ir.component("R1").electrical["theta_ja"] = authoritative(100.0, RESISTOR_DS, "ohm")
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.details["components"]["R1"]["thermal"]["reason"] == "theta_ja carries unit 'ohm', expected K/W"
    ir.component("R1").electrical["theta_ja"] = llm_generated(100.0, "m", "K/W")
    th = _judge(monkeypatch, ir, _fake_run(ir, {"vin": [10.0]}), THERMAL)
    assert th.details["components"]["R1"]["thermal"]["reason"] == "theta_ja is llm_generated (needs confirmation)" and th.status is S.NOT_VERIFIED


# --------------------------------------------------------------------------- with ngspice: the pipeline


@needs_ngspice
def test_pipeline_divider_without_ratings_is_not_verified_with_real_evidence(tmp_path: Path):
    ir = _divider(tmp_path)
    state = _run(ir, tmp_path)
    spice_stage = state.outcome(Stage.SPICE)
    assert spice_stage.status is S.PASS and "re-validated: " in spice_stage.message and "component.fit NOT_VERIFIED" in spice_stage.message
    fits = [r for r in ir.validation.results if r.check_id == FIT_CHECK]
    assert [r.status for r in fits] == [S.NOT_VERIFIED, S.NOT_VERIFIED]  # IR_BUILD (no run yet), then right after SPICE
    assert fits[0].message == "component fit needs an operating point from SPICE; no tool-backed spice result attached" and fits[0].ir_hash == ir.content_hash()
    fit = fits[1]
    assert fit.tool == FIT_CHECK and fit.tool_version == FIT_VERSION and fit.ir_hash == ir.content_hash()
    assert fit.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    _evidence_is_real(fit)
    assert {e.path for e in fit.evidence} == {ir.artifacts[ArtifactKind.SPICE_RESULT].path, ir.validation.latest("spice").details["analyses"]["op"]["raw_output_path"]}
    d = fit.details
    assert d["criteria_evaluated"] == ["electrical stress"] and d["criteria_not_evaluated"] == list(FIT_CRITERIA[1:]) and d["engine"] == "ngspice-shared"
    assert d["engine_version"] == runner.version() and d["analysis"] == "op" and d["spice_status"] == "PASS" and d["assumptions"] == [] and d["derating"] == DERATING
    assert d["conditions"]["temperature_c"] == 27.0 and d["engine_stamp"]["settings_hash"]
    r1 = d["components"]["R1"]
    assert r1["voltage"]["reason"].startswith("no v_max rating (ground it with datasheet_facts_file / confirm_facts)") and r1["voltage"]["v_across"] == pytest.approx(6.0)
    assert r1["voltage"]["pins"] == {"1": pytest.approx(12.0), "2": pytest.approx(6.0)} and r1["voltage"]["nets"] == {"1": "VIN", "2": "VOUT"}
    assert r1["power"]["reason"].startswith("no power_rating rating") and r1["power"]["dissipation"]["value"] == pytest.approx(3.6e-3)
    assert d["components"]["J1"]["voltage"] == {"status": "NOT_VERIFIED", "reason": "excluded from SPICE (connector, no electrical model): no operating point"}
    assert "J1 NOT_VERIFIED (voltage: excluded from SPICE" in fit.message and "one temperature (27 degC)" in fit.message
    # the thermal validator does not run on an ANALOG-only design (registry selection), and nothing was written into the IR
    th = ir.validation.latest(THERMAL_TOOL)  # thermal runs for every design; an ANALOG divider without thermal keys claims nothing
    assert th is not None and th.status is S.NOT_APPLICABLE and "not in the POWER domain" in th.message and th.ir_hash == ir.content_hash()
    assert "component.fit" in ir.validation.latest_by_check() and recompute_parameters(ir).status is S.PASS


@needs_ngspice
def test_pipeline_pass_needs_ratings_on_every_simulated_part_and_an_excluded_part_keeps_it_not_verified(tmp_path: Path):
    ir = _divider(tmp_path)
    _rate(ir)
    _run(ir, tmp_path)
    fit = ir.validation.latest(FIT_CHECK)
    comp = fit.details["components"]
    assert comp["R1"]["status"] == comp["R2"]["status"] == "PASS" and comp["J1"]["status"] == "NOT_VERIFIED"
    assert fit.status is S.NOT_VERIFIED and "J1 NOT_VERIFIED (voltage: excluded from SPICE (connector, no electrical model): no operating point" in fit.message


@needs_ngspice
def test_pipeline_pass_fail_and_round_trip(tmp_path: Path):
    ir = _divider(tmp_path, connector=False)
    _rate(ir)
    state = _run(ir, tmp_path)
    assert state.outcome(Stage.SPICE).status is S.PASS and "component.fit PASS" in state.outcome(Stage.SPICE).message
    fit = ir.validation.latest(FIT_CHECK)
    assert fit.status is S.PASS and fit.ir_hash == ir.content_hash() and fit.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    _evidence_is_real(fit)
    r1 = fit.details["components"]["R1"]
    assert r1["voltage"] == {**r1["voltage"], "status": "PASS", "v_max": 50.0, "limit": 50.0, "factor": 1.0, "derating": DERATING, "nets": {"1": "VIN", "2": "VOUT"}, "analysis": "op"}
    assert r1["voltage"]["v"] == pytest.approx(6.0) and r1["voltage"]["reason"] == "6 V <= 50 V (v_max)"
    assert r1["power"]["p"] == pytest.approx(3.6e-3) and r1["power"]["limit"] == 0.1 and r1["power"]["reason"] == "3.6 mW <= 100 mW (power_rating)"
    assert r1["power"]["dissipation"] == {**r1["power"]["dissipation"], "tool": "calc.power.P_VR", "tool_version": CALC_VERSION, "unit": "W", "inputs": {"v": "R1.v_across", "r": "R1.resistance"}}
    assert "R1 PASS, R2 PASS; nominal values, operating point only, one temperature (27 degC); criteria evaluated: electrical stress; not evaluated: safety" in fit.message
    # the verdict survives the project file (details are JSON) and the IR is untouched by validating
    h = ir.content_hash()
    back = CircuitIR.load(ir.save(tmp_path / "ir.json"))
    assert back.validation.latest(FIT_CHECK).details == fit.details and back.content_hash() == h
    [again] = FIT.validate(ir, _vctx(tmp_path))
    assert again.status is S.PASS and ir.content_hash() == h and recompute_parameters(ir).status is S.PASS
    # FAIL: a 1 mW rating against 3.6 mW, then a 5 V rating against 6 V - human repairs, never a tool
    ir2 = _divider(tmp_path / "f1", connector=False)
    _rate(ir2, power_rating=0.001)
    _run(ir2, tmp_path / "f1")
    fail = ir2.validation.latest(FIT_CHECK)
    assert fail.status is S.FAIL and fail.details["repair"] == "human" and "R1 FAIL (power: 3.6 mW > 1 mW (power_rating))" in fail.message and "R2 FAIL" in fail.message
    assert fail.details["components"]["R1"]["power"]["repair"] == "human" and fail.details["components"]["R1"]["voltage"]["status"] == "PASS"
    ir3 = _divider(tmp_path / "f2", connector=False)
    _rate(ir3, v_max=5.0)
    state3 = _run(ir3, tmp_path / "f2")
    assert ir3.validation.latest(FIT_CHECK).details["components"]["R1"]["voltage"]["reason"] == "6 V > 5 V (v_max)" and state3.outcome(Stage.SPICE).status is S.PASS
    assert "component.fit FAIL" in state3.outcome(Stage.SPICE).message  # the stage keeps the agent's verdict; the re-validation is reported


@needs_ngspice
def test_pipeline_rating_provenance_and_units(tmp_path: Path):
    ir = _divider(tmp_path, connector=False)
    _rate(ir)
    ir.component("R1").electrical["v_max"] = assumption(50.0, "guess", "V")
    ir.component("R2").electrical["v_max"] = llm_generated(50.0, "m", "V")
    ir.component("R2").electrical["power_rating"] = authoritative(0.1, RESISTOR_DS, "A")
    _run(ir, tmp_path)
    comp = ir.validation.latest(FIT_CHECK).details["components"]
    assert comp["R1"]["voltage"]["reason"] == "v_max is assumption (needs confirmation)" and comp["R1"]["power"]["status"] == "PASS"
    assert comp["R2"]["voltage"]["reason"] == "v_max is llm_generated (needs confirmation)" and comp["R2"]["power"]["reason"] == "power_rating carries unit 'A', expected W"
    assert ir.validation.latest(FIT_CHECK).status is S.NOT_VERIFIED


@needs_ngspice
def test_pipeline_freshness_design_change_hand_edit_and_rerun(tmp_path: Path):
    ir = _divider(tmp_path, connector=False)
    _rate(ir)
    _run(ir, tmp_path)
    assert ir.validation.latest(FIT_CHECK).status is S.PASS
    first_hash = ir.validation.latest(FIT_CHECK).artifact_hash
    # a design change: the netlist on disk is about the old IR
    _change_r1_to_20k(ir)
    [stale] = FIT.validate(ir, _vctx(tmp_path))
    assert stale.status is S.NOT_VERIFIED and stale.message == "the SPICE netlist was generated from a different IR version; regenerate and re-simulate"
    # a hand-edited results.json is not evidence (the change makes the expectations right again, so the IR is the one the netlist came from)
    _recalculate_expectations(ir)
    _run(ir, tmp_path)
    assert ir.validation.latest(FIT_CHECK).status is S.PASS and ir.validation.latest(FIT_CHECK).artifact_hash != first_hash
    comp = ir.validation.latest(FIT_CHECK).details["components"]
    assert comp["R1"]["voltage"]["v"] == pytest.approx(8.0) and comp["R1"]["power"]["p"] == pytest.approx(3.2e-3)  # 12 V * 20k / 30k across R1: 8 V, 64 / 20k
    assert comp["R2"]["voltage"]["v"] == pytest.approx(4.0) and comp["R2"]["power"]["p"] == pytest.approx(1.6e-3)
    results = Path(ir.artifacts[ArtifactKind.SPICE_RESULT].path)
    data = json.loads(results.read_text(encoding="utf-8"))
    data["analyses"]["op"]["result"]["vectors"]["vout"] = [0.0]
    results.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    [edited] = FIT.validate(ir, _vctx(tmp_path))
    assert edited.status is S.NOT_VERIFIED and edited.message == "results.json on disk does not match its recorded hash"


@needs_ngspice
def test_pipeline_assumption_in_the_netlist_and_a_failed_run_block_pass(tmp_path: Path):
    from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
    from ai_eda.tools.spice.stage import run_spice_for

    ir = _divider(tmp_path, connector=False)
    _rate(ir)
    ir.component("R2").spice.value = assumption(10_000.0, "guessed from the value field", "ohm")  # same number, unconfirmed
    assert _run(ir, tmp_path).blocked  # ir.assumptions stops the pipeline at IR_BUILD; run the SPICE stage directly, as test_spice_findings_regressions does
    ctx = _ctx(tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools=ctx.tools))
    ir.validation.extend(run_spice_for(ir, ctx.tools, tmp_path))
    [fit] = FIT.validate(ir, _vctx(tmp_path))
    assert fit.status is S.NOT_VERIFIED and fit.details["assumptions"] == ["R2 value"] and ir.validation.latest("spice").status is S.NOT_VERIFIED
    assert "within limit (6 V <= 50 V (v_max)) but the netlist rests on assumption(s)" in fit.details["components"]["R1"]["voltage"]["reason"]
    # a run whose spice summary FAILs (a wrong nominal) gives an op the fit reads but does not PASS on; an exceeded limit still FAILs
    ir2 = _divider(tmp_path / "w", connector=False)
    _rate(ir2)
    ir2.simulation.expectations[0].nominal = user_requirement(5.0, "V", note="wrong on purpose")
    _run(ir2, tmp_path / "w")
    fit2 = ir2.validation.latest(FIT_CHECK)
    assert ir2.validation.latest("spice").status is S.FAIL and fit2.status is S.NOT_VERIFIED and fit2.message.startswith("stress computed on an op of a run whose spice summary is FAIL")
    assert fit2.details["components"]["R1"]["power"]["p"] == pytest.approx(3.6e-3)
    ir3 = _divider(tmp_path / "wf", connector=False)
    _rate(ir3, power_rating=0.001)
    ir3.simulation.expectations[0].nominal = user_requirement(5.0, "V", note="wrong on purpose")
    _run(ir3, tmp_path / "wf")
    assert ir3.validation.latest(FIT_CHECK).status is S.FAIL


@needs_ngspice
def test_pipeline_r_with_spice_params_is_never_judged_on_power(tmp_path: Path):
    ir = _divider(tmp_path, connector=False)
    _rate(ir)
    ir.topology.domains.append(CircuitDomain.POWER)
    ir.simulation.temperature_c = user_requirement(27.0, "degC")
    _thermal(ir)
    ir.component("R1").spice.params = {"m": user_requirement(2.0, note="two in parallel")}
    _run(ir, tmp_path)  # m=2 halves R1: VOUT = 8 V, the 6 V expectation FAILs - and the power of R1 is not what electrical[resistance] says
    fit, th = ir.validation.latest(FIT_CHECK), ir.validation.latest(THERMAL_TOOL)
    assert fit.details["components"]["R1"]["power"]["status"] == "NOT_VERIFIED" and "carries SPICE params ['m']" in fit.details["components"]["R1"]["power"]["reason"]
    assert th.details["components"]["R1"]["thermal"]["status"] == "NOT_VERIFIED" and "carries SPICE params ['m']" in th.details["components"]["R1"]["thermal"]["reason"]
    assert fit.status is not S.PASS and th.status is not S.PASS


@needs_ngspice
def test_pipeline_thermal_pass_fail_and_ambient(tmp_path: Path):
    ir = _divider(tmp_path, connector=False)
    ir.topology.domains.append(CircuitDomain.POWER)
    ir.simulation.temperature_c = user_requirement(85.0, "degC")
    _thermal(ir)
    state = _run(ir, tmp_path)
    assert "domain.power.thermal PASS" in state.outcome(Stage.SPICE).message
    th = ir.validation.latest(THERMAL_TOOL)
    assert th.status is S.PASS and th.tool == THERMAL_TOOL and th.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash and th.ir_hash == ir.content_hash()
    _evidence_is_real(th)
    r1 = th.details["components"]["R1"]["thermal"]
    assert r1["t_a"] == 85.0 and r1["p"] == pytest.approx(3.6e-3) and r1["theta_ja"] == 200.0 and r1["t_j"] == pytest.approx(85.72) and r1["t_j_max"] == 155.0
    assert th.details["conditions"]["temperature_c"] == 85.0 and "ambient 85 degC (simulation.temperature_c, the op's temperature)" in th.message
    # the design view is untouched by validating; the stored derived numbers still recompute
    h = ir.content_hash()
    [again] = THERMAL.validate(ir, _vctx(tmp_path))
    assert again.status is S.PASS and ir.content_hash() == h and recompute_parameters(ir).status is S.PASS
    # FAIL: t_j_max 85.5 < 85.72
    ir2 = _divider(tmp_path / "f", connector=False)
    ir2.topology.domains.append(CircuitDomain.POWER)
    ir2.simulation.temperature_c = user_requirement(85.0, "degC")
    _thermal(ir2, t_j_max=85.5)
    _run(ir2, tmp_path / "f")
    th2 = ir2.validation.latest(THERMAL_TOOL)
    assert th2.status is S.FAIL and th2.details["repair"] == "human" and th2.details["components"]["R1"]["thermal"]["reason"] == "Tj 85.72 degC > t_j_max 85.5 degC"
    # no stated ambient: the op ran at ngspice's default 27 degC and nothing says that is the design's ambient
    ir3 = _divider(tmp_path / "a", connector=False)
    ir3.topology.domains.append(CircuitDomain.POWER)
    _thermal(ir3)
    _run(ir3, tmp_path / "a")
    th3 = ir3.validation.latest(THERMAL_TOOL)
    assert th3.status is S.NOT_VERIFIED and th3.details["components"]["R1"]["thermal"]["reason"].startswith("no ambient: simulation.temperature_c is not set (the op ran at 27 degC, ngspice default")
    # keys on one part only: the other is NOT_VERIFIED naming the keys, and so is the design
    ir4 = _divider(tmp_path / "k", connector=False)
    ir4.topology.domains.append(CircuitDomain.POWER)
    ir4.simulation.temperature_c = user_requirement(27.0, "degC")
    _thermal(ir4, ("R1",))
    _run(ir4, tmp_path / "k")
    th4 = ir4.validation.latest(THERMAL_TOOL)
    assert th4.status is S.NOT_VERIFIED and th4.details["components"]["R1"]["status"] == "PASS" and th4.details["components"]["R2"]["thermal"]["reason"].startswith("no theta_ja / t_j_max")


def test_component_stress_is_pure(monkeypatch, tmp_path: Path):
    """The helper both validators share reads the IR and the run and writes nothing back."""
    r = authoritative(1e3, DS, "ohm")
    part = _part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=r, provenance=AUTH), {"resistance": r})
    ir = _bench_ir(tmp_path, part, nets={"VIN": [("R1", "1")], "GND": [("R1", "2")]})
    h = ir.content_hash()
    st = component_stress(ir, _fake_run(ir, {"vin": [10.0]}), part)
    assert st.v_across == 10.0 and st.dissipation.value == pytest.approx(0.1) and st.plain_r and st.two_terminal and st.excluded is None
    assert st.dissipation.provenance.tool == "calc.power.P_VR" and st.dissipation.provenance.inputs == {"v": "R1.v_across", "r": "R1.resistance"}
    assert ir.content_hash() == h and "v_across" not in part.electrical and list(part.electrical) == ["resistance"]
