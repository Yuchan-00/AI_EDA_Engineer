"""The ``kr447_tx_exciter`` template and its transmit blocks (kr447 design §2.4, wave 2 part P12).

What runs where:

* always: the template's contract against the family table (needs, serves,
  the 4-layer policy), the selection by ``radio_build`` alone, the closed-world
  refusals with the family's sentence naming the serving builds (and "no
  antenna on this board" for the antenna keys), the missing inputs, a non-FM
  modulation, a carrier off the (unverified) raster, out-of-range inputs, a
  stated 2-layer board (refused before anything is asked), a board without its
  companions (the P9 blocks made absent);
* with the packed KiCad 10.0.6 libraries (``needs_libs``, ``KICAD10_SYMBOL_DIR``):
  the confirmation table (every ``kr447.*`` and ``model.*`` row says
  UNVERIFIED), the design numbers (f_T 37.296875 MHz, the PM tank, the load
  line, the low-pass), byte-identical rebuilds (design hash, the compiled
  schematic and board, the table), the two shield cans on the RF floorplan and
  both compilers, a stated tx_power (the profile's exceedance FAILs
  ``rf.regulatory_profile``), the blocks re-based with a net prefix (the
  transceiver's use), the Korean theory / part notes / figures;
* with the libraries and ngspice (``needs_ngspice``): every ``spice.rf.*`` row
  of the ten fixture networks (the cascade pa_lpf among them) PASS inside the kr447 design's §3.3 budget
  (tx_exciter fixtures < 4 s; about 0.8 s on ngspice-42) at pinned values,
  and the whole pipeline on the bench-header board: the design deck (five
  bias rows, three pm_couple rows) PASS, the RF checks, and the stand-in's
  one honest FAIL (``req.input_voltage`` is served by the P9 power block,
  not by the bench headers); with the P9 blocks (the default companions since
  the wave-2 merge) the composed board's deviation chain.

Most tests compose the board with :class:`BenchTxBlock` in place of part P9's
TX power, PTT and audio blocks; :func:`default_companions` composes the P9
blocks themselves (``None`` only when they are absent from the tree).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

import ai_eda.design.templates as templates_mod
from ai_eda.agents.base import AgentContext, IRProposal
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.design.base import Plan
from ai_eda.design.board import add_board
from ai_eda.design.inputs import read_inputs
from ai_eda.design.rf import family, profile
from ai_eda.design.rf.blocks import BlockContext, BlockPrefix
from ai_eda.design.rf.blocks.base import merge_results
from ai_eda.design.rf.blocks.pa import PA_NETWORKS, PaBlock
from ai_eda.design.rf.blocks.tx_chain import (
    BIAS_IDS,
    CHAIN_NETWORKS,
    DRIVER_NETWORKS,
    MOD_NETWORKS,
    PM_COUPLE_IDS,
    PM_OUT_NET,
    TxChainBlock,
    TxDriverBlock,
    TxModBlock,
)
from ai_eda.design.rf.t_tx_exciter import (
    ANTENNA_KEYS,
    BUILD,
    PLANE_REASON,
    REGIONS,
    TEMPLATE_ID,
    BenchTxBlock,
    Companions,
    Kr447TxExciterTemplate,
    bench_companions,
    default_companions,
    tx_net_classes,
)
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, Traced
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc.recompute import recompute_parameters
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.rf_floorplan import rf_floorplan_placement
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.validation.rf import profile_result
from ai_eda.workflow import Stage
from ai_eda.workflow.orchestrator import Orchestrator
from tests.rf_fixture_audit import nets_joining_networks, port_net_outsiders

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_Amplifier", "Oscillator", "Device", "Connector", "Connector_Generic")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-105_38.10x25.40mm") is not None and \
    _REAL.footprint_file("Capacitor_SMD", "C_Trimmer_Murata_TZB4-A") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")
#: the P9 blocks (TX power, PTT, TX audio) are in the tree only after the wave-2 merge
HAS_P9 = all(importlib.util.find_spec(f"ai_eda.design.rf.blocks.{m}") is not None for m in ("power", "ptt", "tx_audio"))

BASE = {"radio_build": "tx_exciter", "modulation": "fm", "input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz"}
TEMPLATE = Kr447TxExciterTemplate(companions=bench_companions())
NETWORKS = (*MOD_NETWORKS, *CHAIN_NETWORKS, *DRIVER_NETWORKS, *PA_NETWORKS)
#: the kr447 design §3.3 budget of the tx_exciter fixtures (s)
FIXTURE_BUDGET_S = 4.0


def _ir(tmp_path: Path, answers: dict[str, str], name: str = "txe") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def _plan(ir: CircuitIR, *, confirmed: bool = True, template: Kr447TxExciterTemplate = TEMPLATE, library: KicadLibrary = _REAL) -> Plan:
    inputs, unusable = read_inputs(ir)
    plan = template.build(ir, inputs, unusable, library, confirmed=confirmed)
    if plan.buildable:
        assert add_board(template, ir, plan, confirmed=confirmed) is None
    return plan


def _applied(tmp_path: Path, answers: dict[str, str] | None = None, name: str = "txe") -> CircuitIR:
    ir = _ir(tmp_path, answers or BASE, name)
    plan = _plan(ir)
    assert plan.buildable, plan.notes
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale)
                                      for c in plan.changes])
    return ir


def _place(ir: CircuitIR) -> None:
    fp = rf_floorplan_placement(ir, _REAL)
    assert ir.pcb is not None
    ir.pcb.outline = fp.outline
    ir.pcb.placements = fp.placements


# --------------------------------------------------------------------------- the contract (no library needed)


def test_template_contract_is_the_family_table() -> None:
    t = Kr447TxExciterTemplate()
    assert t.id == TEMPLATE_ID == "kr447_tx_exciter" == family.BUILDS["tx_exciter"].template_id
    assert t.needs == BUILD.needs and set(t.needs) == {"carrier_frequency", "modulation", "input_voltage"}
    assert t.serves == BUILD.serves and {"radio_build", "tx_power", "frequency_deviation", "audio_bandwidth", "tx_timeout"} <= set(t.serves)
    assert not set(ANTENNA_KEYS) & set(t.serves)  # conducted only: no antenna key is served
    assert t.layer_policy.allowed == (4,) and t.layer_policy.default == 4 and t.layer_policy.reason == PLANE_REASON
    assert t.plane_nets == ("GND", None)
    assert {"tx_mod", "tx_chain", "tx_driver", "pa", "bench_tx", "power", "ptt", "tx_audio"} <= set(REGIONS)
    comp = default_companions(tot=False)
    if HAS_P9:  # after the wave-2 merge: the P9 blocks, supply first and the audio block last
        assert isinstance(comp, Companions) and [b.id for b, _ in comp.supply] == ["power", "ptt"] and [b.id for b, _ in comp.audio] == ["tx_audio"]
    else:
        assert comp is None
    bench = bench_companions()
    assert bench.stand_in and [b.id for b, _ in bench.supply] == ["bench_tx"] and bench.audio == ()
    assert BenchTxBlock.interface_nets == ("TX_5V", "TX_3V3", "PA_5V", "PA_PD", "PM_DRIVE")


def test_selected_by_radio_build_alone(tmp_path: Path) -> None:
    t = Kr447TxExciterTemplate()
    assert t.triggered_by(_ir(tmp_path, {"radio_build": "tx_exciter"}), {})
    assert t.triggered_by(_ir(tmp_path, {"radio_build": "TX-Exciter"}), {})
    for other in ({"radio_build": "transceiver_conducted"}, {"carrier_frequency": "447.5625 MHz", "modulation": "fm"}, {}):
        assert not t.triggered_by(_ir(tmp_path, other), {})


def test_closed_world_refusals_name_the_serving_builds_and_the_missing_antenna(tmp_path: Path) -> None:
    ir = _ir(tmp_path, {**BASE, "erp": "0.5 W", "antenna_gain": "2 dBi", "rx_sensitivity": "-120 dBm", "channel_spacing": "12.5 kHz", "tx_power": "0.5 W"})
    inputs, unusable = read_inputs(ir)
    questions = {q.key: q for q in TEMPLATE.refusals(ir, inputs, unusable)}
    assert set(questions) == {"erp", "antenna_gain", "rx_sensitivity", "channel_spacing"}  # tx_power is served
    assert "erp is not served by radio_build=tx_exciter; erp is served by radio_build=transceiver or transceiver_conducted" in questions["erp"].rationale
    for key in ("erp", "antenna_gain"):
        assert "(no antenna on this board: its output is the conducted U.FL into an attenuator or a dummy load)" in questions[key].rationale
    assert "rx_sensitivity is served by radio_build=rx_frontend, transceiver or transceiver_conducted" in questions["rx_sensitivity"].rationale
    assert "no antenna" not in questions["rx_sensitivity"].rationale
    assert "channel_spacing is served by radio_build=rx_backend, transceiver or transceiver_conducted" in questions["channel_spacing"].rationale
    assert all(not q.required for q in questions.values())
    # a non-FM modulation is refused by the FM rule, not asked
    ir = _ir(tmp_path, {**BASE, "modulation": "am"})
    inputs, unusable = read_inputs(ir)
    (q,) = TEMPLATE.refusals(ir, inputs, unusable)
    assert q.key == "modulation" and not q.required and "FM (F3E) [UNVERIFIED" in q.rationale


def test_the_selection_refuses_through_the_closed_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [TEMPLATE])
    plan = templates_mod.design_from_requirements(_ir(tmp_path, {**BASE, "erp": "0.5 W"}), KicadLibrary(roots=[]))
    assert plan is not None and not plan.buildable and plan.template == TEMPLATE_ID and [q.key for q in plan.questions] == ["erp"]
    assert "erp is served by radio_build=transceiver or transceiver_conducted (no antenna on this board" in plan.notes[0]
    # a stated 2-layer board refuses before anything is asked (the policy's reason)
    plan = templates_mod.design_from_requirements(_ir(tmp_path, {"radio_build": "tx_exciter", "pcb_layers": "2"}), KicadLibrary(roots=[]))
    assert plan is not None and not plan.buildable and PLANE_REASON in plan.notes[0] and [q.key for q in plan.questions] == ["pcb_layers"]


def test_missing_inputs_are_required_questions(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, {"radio_build": "tx_exciter"}), library=KicadLibrary(roots=[]))
    assert not plan.buildable and [q.key for q in plan.questions] == ["carrier_frequency", "input_voltage", "modulation"]
    assert all(q.required for q in plan.questions)
    assert 'carrier_frequency="447.5625 MHz"' in plan.questions[0].question and "modulation=fm" in plan.questions[2].question
    assert "['carrier_frequency', 'input_voltage'] missing" in plan.notes[0]


@pytest.mark.parametrize(("answers", "why"), [
    ({"modulation": "am"}, "this board transmits FM only"),
    ({"carrier_frequency": "447.57 MHz"}, "carrier_frequency 447.57 MHz is not a channel of the KR 447 MHz raster [UNVERIFIED"),
    ({"carrier_frequency": "446 MHz"}, "the channels are 447.5625, 447.5750"),
    ({"input_voltage": "12 V"}, "input_voltage 12 V (req.input_voltage) is outside 6.6..8.4 V"),
    ({"tx_power": "2 W"}, "tx_power 2 W (req.tx_power) is outside 0.01..1 W: the MMZ09332BT1 stage's class [UNVERIFIED"),
    ({"frequency_deviation": "6 kHz"}, "frequency_deviation 6000 Hz (req.frequency_deviation) is outside 500..5000 Hz"),
    ({"tx_timeout": "5 s"}, "tx_timeout 5 s (req.tx_timeout) is outside 10..600 s"),
])
def test_out_of_range_and_non_fm_inputs_refuse(tmp_path: Path, answers: dict[str, str], why: str) -> None:
    plan = _plan(_ir(tmp_path, {**BASE, **answers}), library=KicadLibrary(roots=[]))
    assert not plan.buildable and why in plan.notes[0] and plan.changes == []


def test_without_its_companions_the_board_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_eda.design.rf.t_tx_exciter as t_tx_exciter

    monkeypatch.setattr(t_tx_exciter, "default_companions", lambda *, tot: None)  # the P9 blocks absent from the tree
    plan = _plan(_ir(tmp_path, BASE), template=Kr447TxExciterTemplate(), library=KicadLibrary(roots=[]))
    assert not plan.buildable and "not composed yet" in plan.notes[0] and "input_voltage, frequency_deviation and audio_bandwidth unserved" in plan.notes[0]


def test_the_net_classes_cover_the_blocks_nets() -> None:
    classes = tx_net_classes()
    assert set(classes) == {"RF50", "RF50_H", "RF_OUT", "TX_LUMPED"}
    assert classes["RF50"] == ["TX_RAW", "DRV_IN", "DRV_RFIN", "DRV_OUT", "PA_PAD_IN", "PA_IN", "PA_RFIN"]
    assert classes["RF50_H"] == ["PA_MATCH", "LPF_IN"] and classes["RF_OUT"] == ["TX_OUT"]
    assert PM_OUT_NET in classes["TX_LUMPED"] and "PA_OUT" in classes["TX_LUMPED"] and "TX_B_R5" in classes["TX_LUMPED"]
    member = [n for nets in classes.values() for n in nets]
    assert len(member) == len(set(member))  # one class per net; TX_RAW, which two blocks name, is listed once


# --------------------------------------------------------------------------- the real libraries


@needs_libs
def test_the_table_shows_every_kr447_and_model_row_unverified(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, BASE), confirmed=False)
    assert plan.buildable, plan.notes
    table = plan.table()
    rows = {c.key: c for c in plan.choices}
    kr = [c for c in plan.choices if c.key.startswith("kr447.")]
    assert [c.key for c in kr] == profile.profile_keys() and all("UNVERIFIED" in c.text() for c in kr)
    models = [c for c in plan.choices if c.key.startswith("model.")]
    assert {c.key for c in models} == {"model.bfr92.r_in", "model.bfr92.r_out", "model.buf.r_in", "model.buf.r_out", "model.driver.port_r", "model.l_q.hf",
                                       "model.l_q.uhf", "model.l_q.vhf", "model.npn", "model.pa.r_in", "model.pa.r_off", "model.pa.r_supply", "model.pa.supply",
                                       "model.pa.v_pd", "model.tcxo.r_out",
                                       "model.varactor", "model.varactor.cjo", "model.varactor.m", "model.varactor.vj"}
    assert all("UNVERIFIED" in c.text() for c in models)
    assert rows["pa.p_out"].value == 0.5 and "kr447.max_power [UNVERIFIED" in rows["pa.p_out"].description  # no tx_power stated: the profile's placeholder
    for line in ("kr447.max_deviation = 2500 Hz", "floor.tx_mod.w = 31 mm", "floor.tx_chain.x = 31 mm", "pm.c_tot = 3e-11 F", "tx.tank_qe = 20",
                 "Y801 Oscillator:KT2520K-T / Oscillator:Oscillator_SMD_Kyocera_2520-6Pin_2.5x2.0mm, value 37.297M",
                 "SH802 Device:RFShield_OnePiece / RF_Shielding:Laird_Technologies_BMI-S-103_26.21x26.21mm",
                 "SH801 Device:RFShield_OnePiece / RF_Shielding:Laird_Technologies_BMI-S-105_38.10x25.40mm",
                 "U901 RF_Amplifier:MMZ09332BT1", "fixture pm_mod1: 9 member(s) between src (port), tank (probe), buf (port), bias (control), states bias_lo, bias_nom, bias_hi",
                 "tx.obw99_1k = 8000 Hz [calc.rf.fm.obw99 from kr447.max_deviation, tx.obw.f_1k]",
                 "RF50: nets TX_RAW, DRV_IN, DRV_RFIN, DRV_OUT, PA_PAD_IN, PA_IN, PA_RFIN; Z0 50 ohm +/- 10%",
                 "fixture tx_tank2: 12 member(s) between coll (port), next (port), tx_5v (rail)",
                 "fixture tx_bpf: 19 member(s) between coll (port), pad (port), tx_5v (rail)", "RF50_H: nets PA_MATCH, LPF_IN",
                 "fixture pa_lpf: 11 member(s) between pa_out (port), lpf_out (port), pa_5v (rail)",
                 "pa_lpf.s21_min = -2 dB [calc.rf.db_sum from pa.match.s21_min, lpf.s21_min]",
                 "POWER_PA: nets TX_5V, PA_5V", "stimulus VPMD (dc) on PM_DRIVE", "expectation tx_ic_x3: i(R817) value 0.00240668 +/- 15% A"):
        assert line in table, line
    assert table.count("[UNVERIFIED:") >= len(profile.PROFILE) + 16
    # unconfirmed: every choice is an assumption, never the user's
    change = next(c for c in plan.changes if c.target == "parameters.pm.c_tot")
    assert change.payload.provenance.kind == ProvenanceKind.ASSUMPTION


@needs_libs
def test_the_design_numbers(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    p = {k: float(t.value) for k, t in ir.parameters.items() if isinstance(t.value, (int, float))}
    assert (p["rf.n_mult"], p["tx.f_ref"], p["tx.f1"], p["tx.f2"], p["tx.f3"]) == (12.0, 37296875.0, 111890625.0, 223781250.0, 447562500.0)
    assert p["pm.l"] == pytest.approx(606.98e-9, rel=1e-5) and p["pm.c_fixed"] == pytest.approx(14.8165e-12, rel=1e-5)
    assert p["pm.r_s1"] == p["pm.r_s2"] == pytest.approx(2290.43, abs=0.01) and p["pm.r_div_top"] == pytest.approx(2145.0)
    assert p["pm.k_pm"] == pytest.approx(0.628611, rel=1e-6) and p["tx.k_pm"] == pytest.approx(2 * p["pm.k_pm"])
    assert (p["pm_mod1.phase.lo"], p["pm_mod1.phase.nom"], p["pm_mod1.phase.hi"]) == pytest.approx((-21.4567, 1.4019, 18.2388), abs=1e-4)
    assert p["pm.couple.a_ref"] == pytest.approx(-0.01776, abs=1e-5) and p["pm.f_hp"] == pytest.approx(12.243, abs=1e-3) and p["pm.f_lp"] == pytest.approx(15915.49, abs=0.01)
    assert p["tx.buf1.ic"] == pytest.approx(1.75833e-3, rel=1e-5) and p["tx.x3.ic"] == pytest.approx(2.40668e-3, rel=1e-5)
    # the ported networks: the collector choke (with its Q) on the source, the next stage's divider on the load (kr447 wave-2 review)
    assert [p[f"tx_tank{k}.s21"] for k in (1, 2)] == pytest.approx([-5.4979, -5.5417], abs=1e-4)
    assert (p["tx_tank1.port_r"], p["tx_tank1.port_x"], p["tx.tank1.r_load_eff"]) == pytest.approx((910.455, 273.303, 374.909), abs=1e-3)
    assert p["tx_bpf.s21"] == pytest.approx(-12.3606, abs=1e-4) and p["tx_bpf.rel_m1"] == pytest.approx(-42.3186, abs=1e-3)
    assert p["tx_bpf.rel_p1"] == pytest.approx(-31.3529, abs=1e-3) and p["tx_bpf.bound_p1"] == pytest.approx(-30.3529, abs=1e-3)
    assert p["pa.r_l"] == 20.25 and p["pa.match.l"] == pytest.approx(8.7282e-9, rel=1e-4) and p["pa.match.c"] == pytest.approx(8.6204e-12, rel=1e-4)
    assert [p[f"lpf.{'c' if k % 2 else 'l'}.{k}"] for k in (1, 2, 3, 4)] == pytest.approx([7.8329e-12, 23.588e-9, 13.904e-12, 26.085e-9], rel=1e-4)
    assert p["tx.obw99_1k"] == 8000.0 and p["tx.obw99_3k"] == 12000.0 and p["tx.f_error"] == pytest.approx(1118.90625)
    assert len(ir.components) == 131
    assert ir.rf is not None and [n.id for n in ir.rf.networks] == list(NETWORKS)
    assert "tx_tank3" not in NETWORKS and not any(k.startswith("tx_tank3") for k in p)
    assert [(b.id, b.shield_ref) for b in ir.rf.blocks] == [("bench_tx", None), ("tx_mod", "SH802"), ("tx_chain", "SH801"), ("tx_driver", None), ("pa", None)]
    blocks = {b.id: b for b in ir.rf.blocks}
    assert blocks["tx_mod"].chain[:4] == ["Y801", "C_T801", "D801", "Q801"] and blocks["tx_chain"].chain[:2] == ["C812", "Q803"]
    assert (len(blocks["tx_mod"].refs), len(blocks["tx_chain"].refs)) == (35, 58)  # 34 + 57 parts (and a can each): the design put its 82 under one BMI-S-105
    assert ir.rf.profile_keys == profile.profile_keys() and "model.varactor.cjo" in ir.rf.model_values
    assert [pl.id for pl in ir.rf.frequency_plan] == ["tx_spur_m1", "tx_spur_p1", "tx_residue_half", "tx_residue_3half"]
    assert {li.id for li in ir.rf.lab_items} >= {"tx_frequency", "tx_deviation", "tx_spurious", "pa_power", "pa_harmonics"}
    assert [c.id for c in ir.constraints] == ["c.kr447.conducted_only", "c.kr447.rf_ground_vias", "c.kr447.pa_pd_polarity"]
    serves = {c.ref: set(c.serves_requirements) for c in ir.components}
    assert {"req.radio_build", "req.carrier_frequency"} <= serves["U901"] and "req.radio_build" in serves["J901"]
    assert "req.modulation" in serves["D801"] and "req.carrier_frequency" in serves["Y801"]
    assert ir.simulation is not None and {e.id for e in ir.simulation.expectations} == {*BIAS_IDS, *PM_COUPLE_IDS}
    assert ir.pcb is not None and ir.pcb.stackup is not None and ir.pcb.stackup.layer_count == 4
    assert recompute_parameters(ir).status is S.PASS


@needs_libs
def test_the_chain_fixtures_hold_every_part_on_their_ports_and_no_two_filters_touch(tmp_path: Path) -> None:
    """kr447 wave-2 review findings 1 / 2: a tank's fixture holds the collector choke, the 0 ohm link and the next stage's base divider, and
    no two top-C filters meet on a net (the last tank and the band-pass were joined tap to tap: their two fixtures were not the cascade)."""
    ir = _applied(tmp_path)
    assert ir.rf is not None
    top_c = [n.id for n in ir.rf.networks if "tank" in n.id or "bpf" in n.id]
    assert nets_joining_networks(ir, top_c) == {}
    drv_pad = ir.rf.network("drv_pad")
    assert drv_pad is not None and any(e.id == "s11" and e.quantity == "s11_db" for e in drv_pad.expectations)  # the pad proves its matched input
    assert port_net_outsiders(ir, top_c, allowed=drv_pad.members) == []
    assert top_c == ["tx_tank1", "tx_tank2", "tx_bpf"]
    for nid, choke, link, divider in (("tx_tank1", "L803", "R817", ("R818", "R819")), ("tx_tank2", "L806", "R821", ("R822", "R823")),
                                      ("tx_bpf", "L809", "R825", ())):
        nw = ir.rf.network(nid)
        assert nw is not None and {choke, link, *divider} <= set(nw.members), nid
        assert choke in nw.loss_q and [p.kind for p in nw.ports] == ["port", "port", "rail"] and nw.ports[-1].net == "TX_5V"


@needs_libs
def test_rebuilds_are_byte_identical(tmp_path: Path) -> None:
    outs = []
    for k in (1, 2):
        d = tmp_path / f"b{k}"
        d.mkdir()
        ir = _applied(d)
        _place(ir)
        files = []
        for compiler in (SchematicCompiler(), PCBCompiler()):
            ref = compiler.compile(ir, CompileContext(workdir=d, tools={"kicad_library": _REAL}))
            files.append(Path(ref.path).read_bytes())
        outs.append((ir.content_hash(), files, _plan(_ir(d, BASE)).table()))
    assert outs[0] == outs[1]


@needs_libs
def test_the_floorplan_puts_each_block_under_its_can_and_both_compilers_run(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    _place(ir)
    assert ir.pcb is not None and (ir.pcb.outline.width_mm, ir.pcb.outline.height_mm) == (106.0, 42.0) and len(ir.pcb.placements) == 131
    assert ir.rf is not None
    at = {p.component_ref: (p.x_mm, p.y_mm) for p in ir.pcb.placements}
    for bid in ("tx_mod", "tx_chain"):
        b = ir.rf.block(bid)
        assert b is not None and b.region is not None
        x0, y0, w, h = (float(getattr(b.region, a).value) for a in ("x", "y", "w", "h"))
        assert all(x0 <= at[r][0] <= x0 + w and y0 <= at[r][1] <= y0 + h for r in b.refs), bid
    for compiler in (SchematicCompiler(), PCBCompiler()):
        ref = compiler.compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _REAL}))
        assert Path(ref.path).is_file()


@needs_libs
def test_a_stated_tx_power_designs_the_load_line_and_the_profile_judges_it(tmp_path: Path) -> None:
    ir = _applied(tmp_path, {**BASE, "tx_power": "0.25 W"}, "p025")
    assert float(ir.parameters["pa.p_out"].value) == 0.25 and float(ir.parameters["pa.r_l"].value) == pytest.approx(40.5)  # (5 - 0.5)^2 / (2 x 0.25)
    assert ir.parameters["pa.p_out"].provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert profile_result(ir).status is S.NOT_VERIFIED  # within the placeholder: still never a PASS
    (tmp_path / "over").mkdir()
    over = _applied(tmp_path / "over", {**BASE, "tx_power": "0.8 W"}, "p08")
    assert float(over.parameters["pa.r_l"].value) == pytest.approx(12.65625)
    r = profile_result(over)
    assert r is not None and r.status is S.FAIL and "kr447.max_power" in r.message  # two confirmed values contradict each other


@needs_libs
def test_a_stated_deviation_feeds_the_occupied_bandwidth_rows(tmp_path: Path) -> None:
    ir = _applied(tmp_path, {**BASE, "frequency_deviation": "2 kHz"}, "dev2k")
    dev = ir.parameters["tx.frequency_deviation"]
    assert float(dev.value) == 2000.0 and dev.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    for tag in ("1k", "3k"):
        assert ir.parameters[f"tx.obw99_{tag}"].provenance.derived_from == ["tx.frequency_deviation", f"tx.obw.f_{tag}"]
    assert float(ir.parameters["tx.obw99_1k"].value) < 8000.0  # below the profile maximum's number (2.5 kHz)
    assert recompute_parameters(ir).status is S.PASS


@needs_libs
def test_the_blocks_compose_with_a_net_prefix(tmp_path: Path) -> None:
    """The transceiver's use: internal nets prefixed, the interface nets (TX_PM_OUT, TX_RAW, PA_IN, rails) and GND kept, networks renamed with them."""
    ctx = BlockContext(ir=_ir(tmp_path, BASE), library=_REAL, template_id="t_x", confirmed=True, inputs=read_inputs(_ir(tmp_path, BASE))[0],
                       shared={"power.tx_5v": Traced(value=5.0, unit="V", provenance=_prov()), "power.tx_3v3": Traced(value=3.3, unit="V", provenance=_prov())})
    results = []
    for blk, prefix in ((TxModBlock(), BlockPrefix(800, "TXM_")), (TxChainBlock(), BlockPrefix(800, "TXC_")), (TxDriverBlock(), BlockPrefix(800, "TXD_")),
                        (PaBlock(), BlockPrefix(900, "PA_"))):
        r = blk.build(ctx, prefix)
        ctx.shared.update({k: v for k, v in r.params.items() if k not in ctx.shared})
        results.append(r)
    merged = merge_results(results, "t_x")
    names = {n.name for n in merged.nets}
    assert {PM_OUT_NET, "TX_RAW", "PA_IN", "TX_5V", "TX_3V3", "PA_5V", "PM_DRIVE", "GND"} <= names
    assert "TXM_PM1_T" in names and "PM1_T" not in names and "TXC_TX_B_R5" in names and "PA_LPF_N1" in names
    pm1 = next(n for n in merged.networks if n.id == "pm_mod1")
    assert [p.net for p in pm1.ports] == ["TXM_TCXO_OUT", "TXM_PM1_T", "TXM_BUF1_B", "TXM_PM_BIAS"]
    assert "TXM_PM1_T" in merged.net_classes["TX_LUMPED"] and merged.net_classes["RF50"][0] == "TX_RAW" and "PA_PA_RFIN" in merged.net_classes["RF50"]
    assert sorted(c.ref for c in merged.components if c.ref.startswith("SH")) == ["SH801", "SH802"]


def _prov():
    from ai_eda.design.base import choice_provenance

    return choice_provenance("t_x", "bench rail", True)


@needs_libs
def test_theory_part_notes_and_figures_in_korean(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    sections = TEMPLATE.theory(ir)
    text = "\n".join(s.title + "\n" + s.body for s in sections)
    assert len(sections) == 8 and "전도(conducted) 출력만" in text and "간접 FM" in text and "UNVERIFIED" in text and "BMI-S-103" in text
    for number in ("37.297 MHz", "606.98 nH", "20.25 Ω", "-21.457", "52.75"):
        assert number in text, number
    notes = TEMPLATE.part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components}
    assert "(검증되지 않음" in notes["U901"].substitutes[0] and "안테나 금지" in notes["J901"].why
    assert notes["SH802"].role == "차폐 캔 (BMI-S-103)" and notes["SH801"].role == "차폐 캔 (BMI-S-105)"
    figs = TEMPLATE.theory_figures(ir)
    assert [f.id for f in figs] == ["theory_pm_phase", "theory_tx_tanks", "theory_tx_filters"] and all("<svg" in f.svg for f in figs)


@needs_libs
def test_the_stage_reports_are_deterministic_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_eda.report.stages import build_stage_document

    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [TEMPLATE])
    ir = _applied(tmp_path)
    for stage in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION):
        first, again = build_stage_document(stage, ir, _REAL, None).markdown, build_stage_document(stage, ir, _REAL, None).markdown
        assert first == again and len(first) > 1000, stage
    theory = build_stage_document(Stage.ARCHITECTURE, ir, _REAL, None).markdown
    assert "위상 변조기: 버퍼로 분리된 탱크 두 개" in theory and "![fig](fig:theory_pm_phase)" in theory and "![fig](fig:theory_tx_filters)" in theory
    parts = build_stage_document(Stage.COMPONENT_SELECTION, ir, _REAL, None).markdown
    assert "차폐 캔 (BMI-S-103)" in parts and "전력 증폭기 (MMZ09332BT1)" in parts


# --------------------------------------------------------------------------- ngspice


@needs_libs
@needs_ngspice
def test_every_fixture_row_passes_on_the_default_models_within_the_budget(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    t0 = time.perf_counter()
    results = spice_rf_results(ir, {"spice": runner}, tmp_path)
    elapsed = time.perf_counter() - t0
    got = {r.check_id: r for r in results}
    assert {f"spice.rf.{n}" for n in NETWORKS} <= set(got) and len(results) == 41
    assert all(r.status is S.PASS for r in results), [(r.check_id, r.message) for r in results if r.status is not S.PASS]
    assert all("network verdict under the confirmed model values" in r.message for r in results)
    measured = {cid.removeprefix("spice.rf."): r.details["measured"] for cid, r in got.items() if "measured" in r.details}
    pinned = {  # ngspice-42, 2026-09-29
        "pm_mod1.bias_lo.phase_lo": -21.4597, "pm_mod1.bias_nom.phase_nom": 1.3984, "pm_mod1.bias_hi.phase_hi": 18.2356,
        "pm_mod2.bias_lo.phase_lo": -21.4597, "pm_mod2.bias_hi.phase_hi": 18.2356,
        "tx_tank1.s21": -5.4979, "tx_tank1.rel_m": -47.8356, "tx_tank1.rel_p": -26.2402, "tx_tank2.s21": -5.5417, "tx_tank2.rel_m": -29.0058,
        "tx_tank2.rel_p": -17.6727, "tx_bpf.s21_fc": -12.3606, "tx_bpf.rel_m1": -42.3186, "tx_bpf.rel_p1": -31.3529,
        "tx_bpf.rel_sub2": -168.3153, "tx_bpf.rel_2fc": -100.2590, "drv_pad.s21": -3.0, "pa_pad.s21": -3.0, "pa_match.s21_fc": -0.1460,
        "pa_match.s11_fc": -30.7531, "lpf.s21_fc": -1.0445, "lpf.s11_fc": -18.3852, "lpf.s21_2fc": -52.7787, "lpf.s21_3fc": -80.3007,
    }
    for key, want in pinned.items():
        assert measured[key] == pytest.approx(want, abs=0.01), key
    # the tanks' exact-network nominals and the network agree well inside the 1 dB / 1 deg tolerances
    p = ir.parameters
    assert abs(measured["pm_mod1.bias_nom.phase_nom"] - float(p["pm_mod1.phase.nom"].value)) < 0.01
    # the ported networks' nominals are the networks the fixtures run (choke, link and next divider included)
    for key in ("tx_tank1.s21", "tx_tank1.rel_m", "tx_tank1.rel_p", "tx_tank2.s21", "tx_tank2.rel_m", "tx_tank2.rel_p"):
        assert abs(measured[key] - float(p[key].value)) < 0.01, key
    for row, key in (("tx_bpf.rel_m1", "tx_bpf.rel_m1"), ("tx_bpf.rel_p1", "tx_bpf.rel_p1"), ("tx_bpf.s21_fc", "tx_bpf.s21")):
        assert abs(measured[row] - float(p[key].value)) < 0.01, row
    assert elapsed < FIXTURE_BUDGET_S, f"{elapsed:.2f} s: the kr447 design's §3.3 budget for the tx_exciter fixtures is < {FIXTURE_BUDGET_S} s"


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: Kr447TxExciterTemplate, answers: dict[str, str], *, stop: Stage | None = None,
              skip_placement: bool = False) -> tuple[CircuitIR, object]:
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [template])
    ir = _ir(tmp_path, {})
    tools = {"kicad_library": _REAL, "spice": runner}
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _REAL},
                                      answers={"application": "bench", "jurisdiction": "KR", **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"] and ir.components == []
    confirm = {"confirm_design": "yes", **({"pcb.placement": "skip"} if skip_placement else {})}
    state = Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers=confirm)).run(ir, stop_after=stop)
    assert not state.blocked
    return ir, state


@needs_libs
@needs_ngspice
def test_the_pipeline_on_the_bench_header_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Present, confirm, place: the deck, the fixtures and the RF checks the kr447 design predicts for tx_exciter, and the stand-in's one honest FAIL."""
    ir, state = _pipeline(tmp_path, monkeypatch, TEMPLATE, BASE)
    latest = ir.validation.latest_by_check()
    status = {k: r.status for k, r in latest.items()}
    for check in ("calc.recompute", "spice", "rf.freq_plan", "block.interface.TX_PM_OUT", "block.interface.TX_RAW", "block.interface.PA_IN",
                  "block.interface.TX_5V", "block.interface.TX_3V3", "block.interface.PA_5V", "review.ir_vs_pcb", "review.pcb_vs_bom",
                  *(f"spice.{e}" for e in (*BIAS_IDS, *PM_COUPLE_IDS)), *(f"spice.rf.{n}" for n in NETWORKS)):
        assert status[check] is S.PASS, (check, latest[check].message)
    assert latest["spice"].details["analyses"].keys() == {"op_bias", "pm_ac_low", "pm_ac_ref", "pm_ac_high"}
    for check in ("rf.model_grounding", "rf.regulatory_profile", "rf.lab.tx_deviation", "rf.lab.pa_harmonics", "domain.rf.impedance"):
        assert status[check] is S.NOT_VERIFIED, (check, latest[check].message)
    # no integrator on the bench board: the deviation chain has no V_max / tau_i
    assert status["rf.deviation"] is S.NOT_VERIFIED and "spice.pm_drive_peak: no result recorded" in latest["rf.deviation"].message
    # routing.maze 0.5 reads the PHA-1's SOT-89-3 custom pad and routes most nets, but not all (the PA's 0.5 mm-pitch QFN pads are closer
    # together than the fine rules' 0.2 mm clearance): a half-routed board is never proposed - placement only, the RF copper does not exist
    placement = state.outcome(Stage.PLACEMENT).message
    assert "131 component(s) on a 106.0 x 42.0 mm" in placement and "2 shield can(s)" in placement
    assert "not applied: routing.maze 0.5 connected" in placement and "U901." in placement and "shape 'custom'" not in placement, placement
    assert ir.pcb is not None and ir.pcb.tracks == []
    # placement only: every net is judged (U850.2 is bounded by its anchor and tab boxes) and the nets are a measured open (FAIL)
    conn = latest["pcb.routing.connectivity"]
    assert conn.status is S.FAIL and conn.details["unknown"] == [] and conn.details["custom_pads"] == ["U850.2"]
    assert status["pcb.routing.clearance"] is S.NOT_APPLICABLE  # no IR copper to compare
    # the stand-in's honest FAIL: input_voltage is the P9 power block's, not the bench headers'
    assert status["review.requirements_vs_ir"] is S.FAIL and latest["review.requirements_vs_ir"].details["unserved"] == ["req.input_voltage"]
    assert sorted(k for k, s in status.items() if s is S.FAIL) == ["pcb.routing.connectivity", "repair.loop", "review.requirements_vs_ir"]


@needs_libs
@needs_ngspice
@pytest.mark.skipif(not HAS_P9, reason="the P9 TX power, PTT and audio blocks are not in the tree before the wave-2 merge")
def test_the_p9_composition_closes_the_deviation_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the P9 blocks: every deck row PASS, the integrator designed with tx.k_pm, rf.deviation PASS under the confirmed model values."""
    ir, _state = _pipeline(tmp_path, monkeypatch, Kr447TxExciterTemplate(), {**BASE, "frequency_deviation": "2.5 kHz", "audio_bandwidth": "3 kHz"},
                           stop=Stage.SPICE, skip_placement=True)
    latest = ir.validation.latest_by_check()
    assert latest["spice"].status is S.PASS
    deck = {k: r for k, r in latest.items() if k.startswith("spice.") and not k.startswith("spice.rf.")}
    assert {f"spice.{e}" for e in (*BIAS_IDS, *PM_COUPLE_IDS, "pm_drive_peak", "integrator_1k")} <= set(deck)
    assert all(r.status is S.PASS for r in deck.values()), [(k, r.message) for k, r in deck.items() if r.status is not S.PASS]
    assert all(latest[f"spice.rf.{n}"].status is S.PASS for n in NETWORKS)
    dev = latest["rf.deviation"]
    assert dev.status is S.PASS and "frequency_deviation 2.5 kHz under confirmed model values" in dev.message
    assert "tx.k_pm" in ir.parameters["tx.tau_i"].provenance.derived_from  # the integrator is designed with the modulator's constant
    # PA_5V is emptied at release by the PTT block's active discharge Q205, not by the PA: the PA's draw is switched by its POWER_DOWN pin
    # (model.pa.supply), which the same release edge raises. Without Q205 the row FAILs - the PASS no longer rests on a load the circuit removes
    u901 = ir.component("U901")
    assert u901.spice.model_name == "PASUP" and u901.spice.pin_order == ["11", "13", "10"]  # VCC1, GND, POWER_DOWN (the library's pin numbers)
    assert {"model.pa.r_supply", "model.pa.r_off", "model.pa.v_pd", "model.pa.supply"} <= set(ir.rf.model_values)
    off = latest["spice.pa_supply_off_first"]
    assert off.status is S.PASS and latest["spice.pa_supply_on"].status is S.PASS
    from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
    from ai_eda.ir import ArtifactKind, SpiceBinding
    from ai_eda.tools.spice.stage import run_spice_for

    bare = ir.model_copy(deep=True)
    bare.component("Q205").spice = SpiceBinding(exclude=True, exclude_reason="regression: the active discharge removed", provenance=u901.spice.provenance)
    work = tmp_path / "no_discharge"
    work.mkdir()
    bare.project.workdir = str(work)
    bare.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(bare, CompileContext(workdir=work, tools={"spice": runner}))
    again = {r.check_id: r for r in run_spice_for(bare, {"spice": runner}, work)}
    assert again["spice.pa_supply_off_first"].status is S.FAIL, again["spice.pa_supply_off_first"].message
    assert again["spice.pa_supply_on"].status is S.PASS
