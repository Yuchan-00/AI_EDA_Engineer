"""Device operating-point probes: the op keeps what ngspice reports for a transistor, and ``component.fit`` judges its dissipation on it."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ai_eda.compilers.spice import build_report, device_probes
from ai_eda.design import CONFIRM_DESIGN_KEY
from ai_eda.design.design_file import load_design_file
from ai_eda.ir import SpiceDevice, ValidationStatus
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.runner import SpiceAnalysis
from ai_eda.validation.fit import FIT_CHECK
from ai_eda.workflow.stages import Stage
from tests.test_circuit_templates import _ir, template_library
from tests.test_design_file import _run, _shown, _write

runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

REQ = ["req.output_voltage"]
#: a common-emitter stage: 10 V, R1 1 k collector load, R2 187 k base resistor (E96: a design file takes standard values only), a generic NPN with BF = 100
CE = {
    "format": "ai-eda-design/1",
    "title": "common-emitter stage",
    "topology": {"name": "common-emitter bias", "domains": ["analog"]},
    "components": [
        {"ref": "Q1", "value": "2N3904", "symbol": "Transistor_BJT:2N3904", "footprint": "Package_TO_SOT_THT:TO-92_Inline", "serves_requirements": REQ,
         "electrical": {"power_rating": {"value": 0.05, "unit": "W"}},
         "spice": {"device": "Q", "model_name": "QGEN", "model_card": ".model QGEN NPN (BF=100)", "model_source": "hand-written generic card for this test", "pin_order": ["3", "2", "1"]}},
        {"ref": "R1", "value": "1k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "serves_requirements": REQ, "spice": {"device": "R", "value": {"value": 1000.0, "unit": "ohm"}}},
        {"ref": "R2", "value": "187k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "serves_requirements": REQ, "spice": {"device": "R", "value": {"value": 187000.0, "unit": "ohm"}}},
        {"ref": "J1", "value": "Conn_01x02", "symbol": "Connector_Generic:Conn_01x02", "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", "spice": {"exclude": True, "exclude_reason": "connector"}},
    ],
    "nets": [
        {"name": "VCC", "kind": "power", "pins": ["R1.1", "R2.1", "J1.1"]},
        {"name": "COL", "pins": ["R1.2", "Q1.3"]},
        {"name": "BASE", "pins": ["R2.2", "Q1.2"]},
        {"name": "GND", "kind": "ground", "pins": ["Q1.1", "J1.2"]},
    ],
    "simulation": {
        "stimuli": [{"id": "SUP", "source": "voltage", "net": "VCC", "reference_net": "GND", "kind": "dc", "value": {"value": 10.0, "unit": "V"}}],
        "analyses": [{"id": "op", "kind": "op"}],
        "expectations": [{"id": "v_col", "analysis_id": "op", "vector": "v(COL)", "nominal": {"value": 5.0, "unit": "V"}, "tol_rel": 0.1}],
    },
}


def _fit(tmp_path: Path, q_rating: float | None):
    data = copy.deepcopy(CE)
    data["components"][0]["electrical"] = {"power_rating": {"value": q_rating, "unit": "W"}} if q_rating is not None else {}
    tmp_path.mkdir(exist_ok=True)
    plan = load_design_file(_write(tmp_path, data), template_library(tmp_path))
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=Stage.SPICE)
    return ir, ir.validation.latest(FIT_CHECK)


def test_the_compiler_names_the_probes_of_transistors_only(tmp_path: Path):
    assert device_probes("Q1", SpiceDevice.Q) == {"p": "@q1[p]", "vbe": "@q1[vbe]", "vbc": "@q1[vbc]", "ic": "@q1[ic]", "ib": "@q1[ib]"}
    assert device_probes("M2", SpiceDevice.M)["p"] == "@m2[p]"
    assert device_probes("D1", SpiceDevice.D) == {} and device_probes("R1", SpiceDevice.R) == {}  # ngspice-46 reads a diode's p as inf
    assert device_probes("Q1.A", SpiceDevice.Q) == {}  # a name a probe cannot carry is not probed
    plan = load_design_file(_write(tmp_path, CE), template_library(tmp_path))
    ir = _ir(tmp_path)
    _shown(ir, tmp_path, plan)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    assert build_report(ir)["device_probes"] == {"Q1": device_probes("Q1", SpiceDevice.Q)}


@needs_dll
def test_probes_join_the_node_vectors_and_do_not_outlive_the_run(tmp_path: Path):
    deck = tmp_path / "ce.cir"
    deck.write_text("ce\n.model QGEN NPN (BF=100)\nQ1 COL BASE 0 QGEN\nR1 VCC COL 1k\nR2 VCC BASE 187k\nVSUP VCC 0 DC 10\n.end\n", encoding="utf-8", newline="\n")
    plain = runner.run(deck, SpiceAnalysis.OP, tmp_path / "a")
    probed = runner.run(deck, SpiceAnalysis.OP, tmp_path / "b", probes=("@q1[p]", "@q1[vbc]"))
    again = runner.run(deck, SpiceAnalysis.OP, tmp_path / "c")
    assert plain.succeeded and probed.succeeded and again.succeeded, probed.errors
    assert set(probed.vectors) == set(plain.vectors) | {"@q1[p]", "@q1[vbc]"}
    assert {k: probed.vectors[k] for k in plain.vectors} == plain.vectors and again.vectors == plain.vectors
    assert probed.vectors["@q1[p]"][0] == pytest.approx(25.04e-3, rel=1e-3) and probed.vectors["@q1[vbc]"][0] < 0
    with pytest.raises(ValueError, match="not device probes"):
        runner.run(deck, SpiceAnalysis.OP, tmp_path / "d", probes=("@q1[p]; shell echo x",))


@needs_dll
def test_a_transistors_dissipation_is_judged_against_its_power_rating(tmp_path: Path):
    _, fit = _fit(tmp_path, 0.05)
    q1 = fit.details["components"]["Q1"]
    assert q1["power"]["status"] == "PASS" and q1["power"]["p"] == pytest.approx(25.04e-3, rel=1e-3) and q1["power"]["dissipation"]["tool"] == runner.engine
    # the junction values are shown and judged by nothing: active here (base-collector reverse biased, I_C / I_B = BF)
    op = q1["device_op"]
    assert op["vbc"] < 0 < op["vbe"] and op["ic"] / op["ib"] == pytest.approx(100.0, rel=1e-3)
    assert q1["voltage"]["status"] == "NOT_VERIFIED"  # a pin-to-pin rating is still not judged


@needs_dll
def test_a_transistor_over_its_rating_fails_and_one_without_a_rating_is_not_verified(tmp_path: Path):
    _, hot = _fit(tmp_path / "hot", 0.02)
    assert hot.status is ValidationStatus.FAIL and hot.details["components"]["Q1"]["power"]["reason"].endswith("> 20 mW (power_rating)")
    _, bare = _fit(tmp_path / "bare", None)
    q1 = bare.details["components"]["Q1"]["power"]
    assert q1["status"] == "NOT_VERIFIED" and "no power_rating rating" in q1["reason"] and q1["dissipation"]["value"] == pytest.approx(25.04e-3, rel=1e-3)
