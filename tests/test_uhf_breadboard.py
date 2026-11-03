"""The ``uhf_fm_breadboard`` template (``radio_build = uhf_breadboard``): a discrete UHF FM walkie-talkie for a perfboard.

What runs where:

* always: the template's contract against the family table (needs, serves,
  the 2 | 4 layer policy with 2 by default), the selection by
  ``radio_build`` alone, the two new audio calculators, the closed-world
  refusal naming ``leave_out``, out-of-range inputs (a carrier outside
  430-440 MHz, a 5 V supply), a non-FM modulation and the missing inputs;
* with the packed KiCad 10.0.6 libraries (``needs_libs``): the plan (182
  parts, every one a ``bb_*`` / ``mic`` / ``pot`` row, six fixture networks,
  the frequency plan 36 / 108 / 216 / 432 MHz), byte-identical rebuilds and
  the RF floorplan placing every part in its region;
* with the libraries and ngspice (``needs_ngspice``): every ``spice.rf.*`` row
  of the six fixtures PASS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.agents.base import IRProposal
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design.base import Plan
from ai_eda.design.board import add_board
from ai_eda.design.inputs import read_inputs
from ai_eda.design.rf import family
from ai_eda.design.rf.registry import RF_TEMPLATES
from ai_eda.design.rf.t_breadboard import RADIO_BUILD, REGIONS, TEMPLATE_ID, UHF_FM_BREADBOARD
from ai_eda.ir import CircuitIR, Provenance, ProvenanceKind, ProjectMeta, Traced
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc.radio import butterworth_db_at, noninv_gain_db
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.rf_floorplan import rf_floorplan_placement
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.workflow.orchestrator import Orchestrator

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("Transistor_BJT", "Amplifier_Operational", "Device", "Regulator_Linear")) and \
    _REAL.footprint_file("Capacitor_SMD", "C_Trimmer_Murata_TZC3") is not None and \
    _REAL.footprint_file("Button_Switch_THT", "SW_Slide_SPDT_Straight_CK_OS102011MS2Q") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries not installed (set KICAD10_SYMBOL_DIR)")

BASE = {"radio_build": "uhf_breadboard", "modulation": "FM", "input_voltage": "9 V", "carrier_frequency": "432 MHz"}
NETWORKS = ("tx_tank1", "tx_tank2", "tx_bpf", "pa_match", "lpf", "trsw")
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")


def _ir(tmp_path: Path, answers: dict[str, str], name: str = "bb") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def _plan(ir: CircuitIR, *, confirmed: bool = True) -> Plan:
    inputs, unusable = read_inputs(ir)
    plan = UHF_FM_BREADBOARD.build(ir, inputs, unusable, _REAL, confirmed=confirmed)
    if plan.buildable:
        assert add_board(UHF_FM_BREADBOARD, ir, plan, confirmed=confirmed) is None
    return plan


def _applied(tmp_path: Path, answers: dict[str, str] | None = None) -> CircuitIR:
    ir = _ir(tmp_path, answers or BASE)
    plan = _plan(ir)
    assert plan.buildable, plan.notes
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale)
                                      for c in plan.changes])
    return ir


def _t(v: float, unit: str | None = None) -> Traced:
    return Traced(value=v, unit=unit, provenance=USER)


# --------------------------------------------------------------------------- the contract (no library needed)


def test_template_contract_is_the_family_table() -> None:
    b = family.BUILDS[RADIO_BUILD]
    assert UHF_FM_BREADBOARD.id == TEMPLATE_ID == b.template_id == "uhf_fm_breadboard" and UHF_FM_BREADBOARD in RF_TEMPLATES
    assert UHF_FM_BREADBOARD.needs == b.needs == ("carrier_frequency", "modulation", "input_voltage")
    assert set(UHF_FM_BREADBOARD.serves) == {"radio_build", "carrier_frequency", "modulation", "input_voltage", "audio_bandwidth"}
    assert UHF_FM_BREADBOARD.layer_policy.allowed == (2, 4) and UHF_FM_BREADBOARD.layer_policy.default == 2 and b.stage == 0
    assert "amateur licence" in family.selector_question(CircuitIR(project=ProjectMeta(id="x", name="x"))).question


def test_selected_by_radio_build_alone(tmp_path: Path) -> None:
    ir = _ir(tmp_path, BASE)
    inputs, _ = read_inputs(ir)
    assert [t.id for t in RF_TEMPLATES if t.triggered_by(ir, inputs)] == [TEMPLATE_ID]
    other = _ir(tmp_path, {**BASE, "radio_build": "tx_exciter"}, "o")
    assert not UHF_FM_BREADBOARD.triggered_by(other, read_inputs(other)[0])


def test_the_audio_calculators() -> None:
    g = noninv_gain_db(_t(100e3, "ohm"), _t(1e3, "ohm"), ("r_f", "r_g"))
    assert g.value == pytest.approx(40.0864, abs=1e-4) and g.unit == "dB" and g.provenance.tool == "calc.audio.noninv_gain.db"
    assert butterworth_db_at(_t(2.0), _t(3e3, "Hz"), _t(3e3, "Hz")).value == pytest.approx(-3.0103, abs=1e-4)
    assert butterworth_db_at(_t(2.0), _t(6e3, "Hz"), _t(3e3, "Hz")).value == pytest.approx(-12.3045, abs=1e-4)
    with pytest.raises(ValueError):
        noninv_gain_db(_t(1e3, "ohm"), _t(0.0, "ohm"))


@pytest.mark.parametrize(("answers", "phrase"), [
    ({**BASE, "carrier_frequency": "447.5625 MHz"}, "430000000..440000000"),
    ({**BASE, "input_voltage": "5 V"}, "7.5..12"),
    ({**BASE, "audio_bandwidth": "8 kHz"}, "2000..4000"),
])
def test_out_of_range_inputs_refuse(tmp_path: Path, answers: dict[str, str], phrase: str) -> None:
    plan = _plan(_ir(tmp_path, answers))
    assert not plan.buildable and any(phrase in n for n in plan.notes), plan.notes


def test_missing_inputs_are_asked_and_am_refused(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, {"radio_build": "uhf_breadboard", "modulation": "FM"}))
    assert not plan.buildable and {q.key for q in plan.questions if q.required} == {"carrier_frequency", "input_voltage"}
    ir = _ir(tmp_path, {**BASE, "modulation": "AM"}, "am")
    qs = UHF_FM_BREADBOARD.refusals(ir, *read_inputs(ir))
    assert any(q.key == "modulation" and "varactor-FM" in (q.rationale or "") + q.question for q in qs)


def test_closed_world_names_leave_out(tmp_path: Path) -> None:
    ir = _ir(tmp_path, {**BASE, "tx_power": "0.5 W"})
    qs = UHF_FM_BREADBOARD.refusals(ir, *read_inputs(ir))
    assert [q.key for q in qs] == ["tx_power"] and "--answer leave_out=tx_power" in qs[0].question and qs[0].answer_key == "leave_out"


# --------------------------------------------------------------------------- with the libraries


@needs_libs
def test_the_plan(tmp_path: Path) -> None:
    ir = _ir(tmp_path, BASE)
    plan = _plan(ir, confirmed=False)
    assert plan.buildable, plan.notes
    assert len(plan.parts) == 182
    ir = _applied(tmp_path)
    keys = {c.description for c in ir.components}
    assert keys and all(c.symbol is not None and c.footprint is not None for c in ir.components)
    p = {k: t.value for k, t in ir.parameters.items()}
    assert (p["tx.f_xtal"], p["tx.f1"], p["tx.f2"], p["tx.f3"]) == (36e6, 108e6, 216e6, 432e6)
    assert p["tx.v_bias"] == pytest.approx(2.75) and p["pa.r_load_line"] == pytest.approx(704.1667, rel=1e-6)
    assert ir.rf is not None and [n.id for n in ir.rf.networks] == ["tx_tank1", "tx_tank2", "tx_bpf", "pa_match", "lpf", "trsw"]
    assert {li.id for li in ir.rf.lab_items} >= {"vxo", "tx_spurious", "rx_superregen", "trsw"} and ir.rf.profile_keys == []
    lib_ids = {c.symbol.library + ":" + c.symbol.name for c in ir.components}  # type: ignore[union-attr]
    assert not any("MCU" in x or "Interface" in x for x in lib_ids)  # no programmable part
    assert {"Transistor_BJT:2N3904", "Transistor_BJT:BF199", "Transistor_BJT:BFR92", "Amplifier_Operational:TL071", "Transistor_BJT:BD139"} <= lib_ids


@needs_libs
def test_rebuild_is_byte_identical_and_the_floorplan_places_every_part(tmp_path: Path) -> None:
    a, b = _applied(tmp_path / "a"), _applied(tmp_path / "b")
    assert a.content_hash() == b.content_hash()
    fp = rf_floorplan_placement(a, _REAL)
    assert len(fp.placements) == len(a.components) and (fp.outline.width_mm, fp.outline.height_mm) == (200.0, 100.0)
    blocks = {blk.id: blk for blk in a.rf.blocks}  # type: ignore[union-attr]
    assert set(blocks) == set(REGIONS)


@needs_libs
@needs_ngspice
def test_every_fixture_row_passes(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    results = spice_rf_results(ir, {"spice": runner}, tmp_path)
    rows = {r.check_id: r for r in results}
    for nid in NETWORKS:
        assert rows[f"spice.rf.{nid}"].status is S.PASS, rows[f"spice.rf.{nid}"].message
    assert all(r.status is S.PASS for r in results), [(r.check_id, r.message) for r in results if r.status is not S.PASS]
