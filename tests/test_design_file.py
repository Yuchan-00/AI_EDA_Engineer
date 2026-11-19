"""``ai-eda run --design FILE.json``: a file's circuit goes through the template gate, and a file cannot state provenance or pins."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.cli import main
from ai_eda.design import CONFIRM_DESIGN_KEY, INPUTS_CHECK
from ai_eda.design.base import design_file_hash
from ai_eda.compilers.spice import model_card_hash
from ai_eda.design.design_file import NEEDS_EMPTY_IR, DesignFileError, load_design_file
from ai_eda.ir import CircuitIR, ProvenanceKind, ValidationStatus
from ai_eda.ir.simulation import MODEL_CARD_NO_SOURCE
from ai_eda.report.stages import NO_TEMPLATE, theory_report
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import Orchestrator
from ai_eda.workflow.stages import Stage
from tests.fixtures_rc import C_FARAD, R_OHM, V_STEP
from tests.test_circuit_templates import BASE, _ir, template_library

runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

#: tests/fixtures_rc.py's circuit as a design file: R1 1 k, C1 1 u, J1 header, a 5 V step, v(OUT) at tau = 5 V (1 - 1/e)
RC = {
    "format": "ai-eda-design/1",
    "title": "RC low-pass",
    "topology": {"name": "rc low-pass", "domains": ["analog"]},
    "components": [
        {"ref": "R1", "value": "1k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "serves_requirements": ["req.step_response"], "spice": {"device": "R", "value": {"value": R_OHM, "unit": "ohm"}}},
        {"ref": "C1", "value": "1u", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "serves_requirements": ["req.step_response"], "spice": {"device": "C", "value": {"value": C_FARAD, "unit": "F"}}},
        {"ref": "J1", "value": "Conn_01x02", "symbol": "Connector_Generic:Conn_01x02", "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
         "spice": {"exclude": True, "exclude_reason": "connector"}},
    ],
    "nets": [
        {"name": "IN", "pins": ["J1.1", "R1.1"]},
        {"name": "OUT", "pins": ["R1.2", "C1.1"]},
        {"name": "GND", "kind": "ground", "pins": ["J1.2", "C1.2"]},
    ],
    "simulation": {
        "stimuli": [{"id": "VIN", "source": "voltage", "net": "IN", "reference_net": "GND", "kind": "pulse",
                     "params": {"v1": 0.0, "v2": V_STEP, "td": 0.0, "tr": 1e-9, "tf": 1e-9, "pw": 1.0, "per": 2.0}}],
        "analyses": [{"id": "tran", "kind": "tran", "params": {"step": 1e-6, "stop": 5e-3}}],
        "expectations": [{"id": "v_out_tau", "analysis_id": "tran", "vector": "v(OUT)", "reduce": "at", "at": 1e-3, "nominal": {"value": 3.1606027941, "unit": "V"}, "tol_rel": 0.02}],
    },
}


def _write(tmp_path: Path, data: dict = RC) -> Path:
    path = tmp_path / "design.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _variant(**changes) -> dict:
    data = copy.deepcopy(RC)
    data.update(changes)
    return data


def _run(ir: CircuitIR, tmp_path: Path, plan, answers: dict[str, str], stop_after: Stage | None = Stage.ARCHITECTURE):
    lib = template_library(tmp_path)
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": lib, "design_file": plan, "spice": runner}, answers=answers)
    return Orchestrator(ctx).run(ir, stop_after=stop_after)


def _shown(ir: CircuitIR, tmp_path: Path, plan) -> str:
    state = _run(ir, tmp_path, plan, BASE)
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert ir.components == [] and ir.nets == [] and ir.simulation is None
    return state.open_questions[0].question


def test_the_first_run_shows_the_table_and_applies_nothing(tmp_path: Path):
    plan = load_design_file(_write(tmp_path), template_library(tmp_path))
    table = _shown(_ir(tmp_path), tmp_path, plan)
    assert plan.sha256 in table
    for text in ("R1 Device:R", "SPICE R 1000.0 ohm", "excluded from SPICE: connector", "OUT (signal): R1.2, C1.1", "expectation v_out_tau"):
        assert text in table


def test_the_table_shows_every_field_that_changes_what_is_built(tmp_path: Path):
    def edit(d: dict) -> None:
        d["topology"]["rationale"] = "because"
        d["components"][0]["electrical"] = {"v_max": 50}
        d["components"][0]["serves_requirements"] = ["req.x"]
        d["components"][0]["spice"].update(pin_order=["2", "1"], ignored_pins={"3": "unused"}, model_name="M", model_card=".model M R")
        d["nets"][1]["serves_requirements"] = ["req.y"]
        d["simulation"]["expectations"][0].update(requirement_id="req.z", reference_vector="v(IN)", params={"ref": 1.0})
    plan = load_design_file(_write(tmp_path, _edited(edit)), template_library(tmp_path))
    table = plan.table()
    for text in ("Topology: rc low-pass (analog): because", "electrical v_max=50", "serves req.x", "pin_order ['2', '1']", "ignored_pins {'3': 'unused'}", f"model_card '.model M R' ({model_card_hash('.model M R')}, no model_source",
                 "serves req.y", "against v(IN)", "[ref=1.0]", "verifies req.z"):
        assert text in table


def test_a_confirmation_applies_the_circuit_with_the_files_hash_on_every_part(tmp_path: Path):
    lib = template_library(tmp_path)
    plan = load_design_file(_write(tmp_path), lib)
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    state = _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    assert not state.blocked
    assert [c.ref for c in ir.components] == ["R1", "C1", "J1"] and [n.name for n in ir.nets] == ["IN", "OUT", "GND"]
    for c in ir.components:
        assert c.provenance.kind is ProvenanceKind.USER_REQUIREMENT and plan.sha256 in c.provenance.note
    assert ir.component("R1").electrical["resistance"].value == R_OHM  # the BOM's value is the simulated one
    assert [p.number for p in ir.component("R1").pins] == ["1", "2"]  # read from the library


def test_a_changed_file_is_asked_again(tmp_path: Path):
    lib = template_library(tmp_path)
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, load_design_file(_write(tmp_path), lib))
    changed = copy.deepcopy(RC)
    changed["components"][0]["value"] = "2k"
    state = _run(ir, tmp_path, load_design_file(_write(tmp_path, changed), lib), {CONFIRM_DESIGN_KEY: "yes"})
    assert state.blocked and ir.components == []
    assert any("is not the one you confirmed" in o.message for o in state.outcomes)


def test_a_design_that_is_not_empty_ignores_the_file(tmp_path: Path):
    lib = template_library(tmp_path)
    plan = load_design_file(_write(tmp_path), lib)
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    before = ir.design_dict()
    state = _run(ir, tmp_path, plan, {})
    assert any(NEEDS_EMPTY_IR in o.message for o in state.outcomes)
    assert ir.design_dict() == before


@pytest.mark.parametrize(("change", "message"), [
    ({"provenance": {"kind": "authoritative"}}, "provenance: Extra inputs"),
    ({"format": "other"}, "format:"),
    ({"components": []}, "components:"),
    ({"topology": {"name": "x", "provenance": {}}}, "topology.provenance"),
])
def test_a_file_outside_the_format_is_refused(tmp_path: Path, change: dict, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _variant(**change)), template_library(tmp_path))


def _edited(edit) -> dict:
    data = copy.deepcopy(RC)
    edit(data)
    return data


@pytest.mark.parametrize(("edit", "message"), [
    (lambda d: d["components"][0].update(provenance={"kind": "authoritative"}), "provenance"),
    (lambda d: d["components"][0].update(pins=["1"]), "pins"),
    (lambda d: d["components"][0].update(symbol="Device:NoSuchPart"), "NoSuchPart"),
    (lambda d: d["components"][0].update(footprint="Resistor_SMD"), "not Library:Name"),
    (lambda d: d["components"][0].update(serves_requirements=[]), r"\['R1'\] name no requirement"),
    (lambda d: d["components"][1].update(ref="R1"), r"\['R1'\] appear more than once"),
    (lambda d: d["components"][1].update(ref="CX"), r"\['CX'\] are not annotated"),
    (lambda d: d["nets"][0]["pins"].append("R1.9"), r"R1\.9.*symbol has \['1', '2'\]"),
    (lambda d: d["nets"][0]["pins"].append("U9.1"), "no such part"),
    (lambda d: d["components"][0]["spice"].update(value="not a number"), "must be a number"),
    # a part value is one that is sold (the templates' rule, for a circuit a person or a model writes)
    (lambda d: d["components"][0]["spice"].update(value={"value": 1591.549, "unit": "ohm"}), r"R1 value 1591\.549 ohm is not a value parts are sold in \(IEC 60063 E24 / E96\); nearest: 1580, 1600; use a standard value"),
    (lambda d: d["components"][1]["spice"].update(value={"value": 6.4817e-8, "unit": "F"}), r"C1 value 6\.4817e-08 F is not a value parts are sold in \(IEC 60063 E6 / E12 / E24\); nearest: 6\.2e-08, 6\.8e-08"),
    (lambda d: d["components"][0].update(value_source="two" + chr(10) + "lines"), "R1 value_source must be one line"),
    (lambda d: d["components"][0]["spice"].update(value={"value": 1000, "unit": "V"}), "R1 value is stated in 'V', a R takes ohm"),
    (lambda d: d["components"][0]["spice"].update(model_card=".model M R"), "R1: model_card given without model_name"),
    (lambda d: d["components"][0]["spice"].update(model_name="M", model_card=".include vendor.lib\n.model M R"), r"R1: model_card line '\.include vendor\.lib' is not allowed"),
    (lambda d: d["components"][0]["spice"].update(model_name="M", model_card=".model OTHER R"), "R1: model_card does not define 'M'"),
    (lambda d: d["components"][0]["spice"].update(model_source="https://vendor.example/m.lib"), "R1 has model_source without model_card"),
    (lambda d: d["components"][0]["spice"].update(model_name="M", model_card=".model M R", model_source="a\nb"), "R1 model_source must be one line"),
    (lambda d: d["components"][0]["spice"].update(model_name="M", model_card=".model M R", model_source=f"x {MODEL_CARD_NO_SOURCE}"), "R1 model_source must be one line"),
    (lambda d: d["components"][0]["spice"].update(model_name="M", model_card=".model M R", sha256="sha256:00"), "sha256"),
])
def test_a_part_or_net_the_library_cannot_build_is_refused(tmp_path: Path, edit, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _edited(edit)), template_library(tmp_path))


def test_a_value_outside_the_e_series_needs_the_real_part_named_and_the_table_shows_it(tmp_path: Path):
    """A 5 mohm shunt is a real part outside every E series: the file names it, the confirmation table repeats that as the file's word."""
    data = json.loads(json.dumps(RC))
    data["components"][0]["spice"]["value"] = {"value": 0.005, "unit": "ohm"}
    with pytest.raises(DesignFileError, match=r"R1 value 0\.005 ohm is not a value parts are sold in .*nearest: 0\.00499, 0\.0051"):
        load_design_file(_write(tmp_path, data), template_library(tmp_path / "kicad"))
    data["components"][0]["value_source"] = "Vishay WSL2512R0050FEA (datasheet 30100)"
    plan = load_design_file(_write(tmp_path, data), template_library(tmp_path / "kicad"))
    assert any("R1" in line and "value outside the E series, stated part: Vishay WSL2512R0050FEA (datasheet 30100)" in line for line in plan.parts)
    data["components"][0]["spice"]["value"] = {"value": 0, "unit": "ohm"}  # a zero-ohm link is not a series question
    del data["components"][0]["value_source"]
    load_design_file(_write(tmp_path, data), template_library(tmp_path / "kicad"))


def test_nan_and_broken_json_are_refused(tmp_path: Path):
    lib = template_library(tmp_path)
    path = tmp_path / "d.json"
    for text in ('{"format": NaN}', "{not json"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DesignFileError, match="is not JSON"):
            load_design_file(path, lib)
    with pytest.raises(DesignFileError, match="cannot read"):
        load_design_file(tmp_path / "missing.json", lib)


def test_the_cli_refuses_a_bad_file_with_exit_2_before_the_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    ir_path = tmp_path / "ir.json"
    _ir(tmp_path).save(ir_path)
    before = ir_path.read_bytes()
    assert main(["run", str(ir_path), "--design", str(tmp_path / "missing.json"), "--no-pdf"]) == 2
    assert "--design: cannot read" in capsys.readouterr().err
    assert ir_path.read_bytes() == before


def test_a_file_that_traces_its_parts_has_no_trace_fail_in_review(tmp_path: Path):
    plan = load_design_file(_write(tmp_path), template_library(tmp_path))
    ir = _ir(tmp_path)
    state = _run(ir, tmp_path, plan, {**BASE, "step_response": "3.1606 V"})
    assert state.blocked and [r.id for r in ir.requirements.requirements if r.key == "step_response"] == ["req.step_response"]
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=Stage.INDEPENDENT_REVIEW)
    result = next(r for r in ir.validation.results if r.check_id == "review.requirements_vs_ir")
    assert result.status is ValidationStatus.PASS, result.message


def _applied(tmp_path: Path, data: dict, answers: dict[str, str]) -> tuple[CircuitIR, object]:
    plan = load_design_file(_write(tmp_path, data), template_library(tmp_path))
    ir = _ir(tmp_path)
    _run(ir, tmp_path, plan, {**BASE, **answers})
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    return ir, plan


def _inputs_check(ir: CircuitIR, tmp_path: Path, plan):
    _run(ir, tmp_path, plan, {})
    return next(r for r in ir.validation.results if r.check_id == INPUTS_CHECK)


def test_a_later_run_checks_the_files_requirement_ids_against_the_ir(tmp_path: Path):
    ir, plan = _applied(tmp_path, RC, {"step_response": "3.1606 V"})
    result = _inputs_check(ir, tmp_path, plan)
    assert result.status is ValidationStatus.PASS and result.details["named"] == ["req.step_response"]


def test_a_requirement_id_the_ir_does_not_have_is_named_and_not_verified(tmp_path: Path):
    ir, plan = _applied(tmp_path, RC, {})
    result = _inputs_check(ir, tmp_path, plan)
    assert result.status is ValidationStatus.NOT_VERIFIED and result.details["unknown"] == ["req.step_response"]
    assert "req.step_response" in result.message


def test_an_excluded_part_without_a_reason_still_needs_a_requirement(tmp_path: Path):
    def edit(d: dict) -> None:
        d["components"][2]["spice"]["exclude_reason"] = "  "
    with pytest.raises(DesignFileError, match=r"\['J1'\] name no requirement"):
        load_design_file(_write(tmp_path, _edited(edit)), template_library(tmp_path))


def test_the_reports_name_the_file_not_a_missing_template(tmp_path: Path):
    lib = template_library(tmp_path)
    ir, plan = _applied(tmp_path, RC, {})
    assert design_file_hash(ir) == plan.sha256
    text = theory_report(ir, lib)
    assert f"- 설계 출처: 사용자 제공 설계 (`{plan.sha256}`)" in text and NO_TEMPLATE not in text


@needs_dll
def test_the_files_circuit_passes_spice_at_the_fixtures_value(tmp_path: Path):
    lib = template_library(tmp_path)
    plan = load_design_file(_write(tmp_path), lib)
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=Stage.SPICE)
    result = next(r for r in ir.validation.results if r.check_id == "spice.v_out_tau")
    assert result.status is ValidationStatus.PASS and result.tool
    assert result.details["measured"] == pytest.approx(3.1606027941, rel=0.02)


#: R1 as a subcircuit the file carries: the same 1 k, so the fixture's expectation still holds
CARD = ".subckt RSUB a b\nR1 a b 1k\n.ends RSUB"
SOURCE = "https://vendor.example/rsub.lib (sha256:1234)"


def _with_card(source: str | None) -> dict:
    def edit(d: dict) -> None:
        d["components"][0]["spice"] = {"device": "X", "model_name": "RSUB", "model_card": CARD, **({"model_source": source} if source else {})}
    return _edited(edit)


def _spice_result(tmp_path: Path, data: dict):
    plan = load_design_file(_write(tmp_path, data), template_library(tmp_path))
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=Stage.SPICE)
    return ir, next(r for r in ir.validation.results if r.check_id == "spice.v_out_tau")


def test_a_cards_note_carries_its_hash_and_its_source_or_says_there_is_none(tmp_path: Path):
    lib = template_library(tmp_path)
    for source in (SOURCE, None):
        plan = load_design_file(_write(tmp_path, _with_card(source)), lib)
        card = next(c.payload for c in plan.changes if getattr(c.payload, "ref", None) == "R1").spice.model_card
        # the text is the user's either way (no IR assumption, so the pipeline is not stopped for a confirmation nobody can give)
        assert card.value == CARD and card.provenance.kind is ProvenanceKind.USER_REQUIREMENT
        assert card.provenance.note == f"design file {plan.sha256}; model card {model_card_hash(CARD)} " + (f"from {SOURCE}" if source else MODEL_CARD_NO_SOURCE)
        assert (f"from {SOURCE} as you state it" if source else "no model_source: SPICE results that would pass are NOT_VERIFIED") in plan.table()


@needs_dll
def test_a_sourced_card_passes_spice_with_its_hash_and_source_in_the_result(tmp_path: Path):
    ir, result = _spice_result(tmp_path, _with_card(SOURCE))
    assert result.status is ValidationStatus.PASS, result.message
    (card,) = result.details["model_cards"]
    assert card["of"] == "R1" and card["model"] == "RSUB" and card["sha256"] == model_card_hash(CARD)
    assert card["provenance"] == "user_requirement" and card["note"].endswith(f"from {SOURCE}")
    summary = next(r for r in ir.validation.results if r.check_id == "spice")
    assert summary.details["model_cards"] == [card]


@needs_dll
def test_the_same_card_without_a_source_is_not_verified(tmp_path: Path):
    ir, result = _spice_result(tmp_path, _with_card(None))
    assert result.status is ValidationStatus.NOT_VERIFIED
    assert result.details["measured"] == pytest.approx(3.1606027941, rel=0.02)  # it ran and would have passed
    assert result.details["assumptions"] == [f"R1 model_card {MODEL_CARD_NO_SOURCE}"] and "R1 model_card (no model_source)" in result.message
    assert next(r for r in ir.validation.results if r.check_id == "spice").status is ValidationStatus.NOT_VERIFIED
    assert next(r for r in ir.validation.results if r.check_id == "ir.assumptions").status is ValidationStatus.PASS  # not stopped at IR_BUILD


def test_a_parts_maker_and_part_number_reach_the_bom_columns_and_the_status_stays_its_own(tmp_path: Path):
    """The 9 V piano board's BOM said NOT_VERIFIED under Manufacturer and MPN, its part numbers only in the description text. A part
    may state ``manufacturer`` and ``mpn``: the table shows them as the file's word, the IR keeps them stamped user_requirement with
    the file's hash, and the BOM prints them in their columns - while DatasheetHash stays empty and the verification status, in a
    column of its own, names it (no datasheet grounds them). A cell a spreadsheet would execute is refused at load."""
    from ai_eda.compilers import BOMCompiler, CompileContext

    data = json.loads(json.dumps(RC))
    data["components"][0].update(manufacturer="Yageo", mpn="RC0603FR-071KL")
    ir, plan = _applied(tmp_path, data, {})
    assert any("R1" in line and "manufacturer Yageo, part number RC0603FR-071KL (the file's word, not looked up)" in line for line in plan.parts)
    r1 = ir.component("R1")
    assert (r1.manufacturer.value, r1.mpn.value) == ("Yageo", "RC0603FR-071KL")
    assert r1.mpn.provenance.kind is ProvenanceKind.USER_REQUIREMENT and plan.sha256 in r1.mpn.provenance.note
    bom = BOMCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": template_library(tmp_path)}))
    rows = {line.split(",")[0]: line for line in Path(bom.path).read_text(encoding="utf-8").splitlines()}
    head = rows["Reference"].split(",")
    cells = dict(zip(head, rows["R1"].split(",")))
    assert (cells["Manufacturer"], cells["MPN"], cells["DatasheetHash"]) == ("Yageo", "RC0603FR-071KL", "")
    assert "DatasheetHash (MPN not found in an archived datasheet)" in cells["Unverified"] and "MPN (" not in cells["Unverified"]
    c1 = dict(zip(head, rows["C1"].split(",")))
    assert c1["MPN"] == "" and "MPN (not stated)" in c1["Unverified"]  # a part without one: empty, and said so in its own column
    data["components"][0]["mpn"] = "=HYPERLINK(1)"
    with pytest.raises(DesignFileError, match=r"R1 mpn '=HYPERLINK\(1\)' starts with '='"):
        load_design_file(_write(tmp_path, data), template_library(tmp_path))
