"""The ``kr447_transceiver`` template (both builds) and its trx / antenna blocks (kr447 design §2.5, wave 2 part P13).

What runs where:

* always: the template's contract against the family table (needs, serves,
  the 4-layer policy, both builds select it), the closed-world refusals with
  the family's sentence, a non-FM modulation, the missing inputs, a carrier
  off the (unverified) raster, out-of-range inputs, a stated 2-layer board
  (refused before anything is asked), and the board module's keep-out hook
  (``Plan.keepouts`` into a new ``ir.pcb`` or appended to an existing one,
  a clash refused; a plan without keep-outs proposes the board it always
  did);
* with the packed KiCad 10.0.6 libraries (``needs_libs``,
  ``KICAD10_SYMBOL_DIR``): the confirmation table (every ``kr447.*`` and
  ``model.*`` row says UNVERIFIED), the composition (441 parts, 211 nets,
  23 fixture networks - the cascade ant_end among them -, 15 blocks with ports and regions), the variant
  difference (``ANT1`` and what goes with the antenna against ``J1001``, in
  the design view), byte-identical rebuilds (design hash, table, compiled
  schematic and board), the RF floorplan on the board's 60 x 243 mm outline
  for both builds and with every option (about 0.1 s) and the strict area
  bound that rules out the design's 60 x 145 mm, the keep-outs (and their
  compiled rule areas), the RF checks without ngspice (every block interface
  PASS; a tx_power above the profile placeholder FAILs the profile row), the
  SI classes, the Korean theory / part notes / figures and the four stage
  reports, every optional input served, the full board's router refusal
  (the PA's QFN pad below the 0.2 mm grid; the three custom pads - the
  microphone's and the two PHA-1s' - are read) and a reduced routing case - the
  antenna end alone (trx + antenna blocks) routed on the real footprints for
  both builds, ``ANT_FEED`` through the antenna band, the GND vias inside
  ``trx_bcu``, ``pcb.keepout`` PASS;
* with the libraries and ngspice (``needs_ngspice``): every ``spice.rf.*``
  row of the 23 networks PASS (about 2.2 s on ngspice-42) at pinned trsw /
  lpf values, the antenna match's fixture with a confirmed
  ``antenna_impedance``, and the pipeline through the SPICE stage: every
  design-deck row PASS (``pin_bias`` among them), ``block.interface.*``
  PASS, ``rf.deviation`` PASS under the confirmed model values, no FAIL, and
  the fixture decks plus the union deck inside the kr447 design's §3.3
  budget for the transceiver (< 10 s of engine time, < 60 MB of rawfiles).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import ai_eda.design.rf.t_transceiver as transceiver_module
import ai_eda.design.templates as templates_mod
from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.design.base import TEMPLATE_VERSION, DesignChange, Plan
from ai_eda.design.board import add_board
from ai_eda.design.inputs import read_inputs
from ai_eda.design.rf import family
from ai_eda.design.rf.blocks.trx import ANT_R_RANGE, FEED_NET, TRX_NETWORKS
from ai_eda.design.rf.profile import profile_keys
from ai_eda.design.rf.t_transceiver import BAND_H_MM, KR447_TRANSCEIVER, PLANE_REASON, REGIONS, VARIANTS, Kr447TransceiverTemplate
from ai_eda.design.templates import TEMPLATES, design_from_requirements
from ai_eda.ir import BoardOutline, CircuitIR, Keepout, PCBDesign, ProjectMeta, assumption
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.rf import RFDesign
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.grid import MARGIN_MM, SPACING_MM, footprint_extent
from ai_eda.tools.placement.rf_floorplan import rf_floorplan_placement
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.validation import ValidationContext
from ai_eda.validation.registry import default_registry
from ai_eda.validation.rf import rf_results
from ai_eda.workflow import Stage
from ai_eda.workflow.orchestrator import Orchestrator
from tests.rf_fixture_audit import nets_joining_networks, port_net_outsiders

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_Amplifier", "RF_AM_FM", "RF_Mixer", "Oscillator", "Device", "Connector", "Amplifier_Audio")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-105_38.10x25.40mm") is not None and \
    _REAL.footprint_file("Connector_Wire", "SolderWire-0.5sqmm_1x01_D0.9mm_OD2.1mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")

BASE = {"radio_build": "transceiver", "modulation": "fm", "input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz"}
CONDUCTED = {**BASE, "radio_build": "transceiver_conducted"}
#: every optional key of the family table stated at once (the antenna match, the time-out, the radiated numbers)
EVERYTHING = {**BASE, "antenna_gain": "2.15 dBi", "link_range": "1 km", "antenna_impedance": "36 ohm", "tx_timeout": "180 s", "occupied_bandwidth": "8.5 kHz",
              "channel_spacing": "12.5 kHz", "rx_sensitivity": "-113 dBm", "frequency_deviation": "2.5 kHz", "audio_bandwidth": "3 kHz", "tx_power": "0.5 W",
              "erp": "0.5 W", "field_strength_limit": "0.01 V/m", "frequency_tolerance": "2.5 ppm", "system_impedance": "50 ohm"}
#: the kr447 design §3.3 budget of the transceiver's fixture decks and union deck (s, bytes of rawfiles)
FIXTURE_BUDGET_S = 10.0
RAWFILE_BUDGET_BYTES = 60_000_000
#: the board outline the floorplan gives (mm): 60 mm wide like the design, the length the composed blocks need
OUTLINE_MM = (60.0, 243.0)
#: the block ids of the board in build order
BLOCK_IDS = ["power", "ptt", "lo_chain", "lo_buffer", "rx_frontend", "rx_mixer", "if_backend", "rx_audio", "tx_mod", "tx_chain", "tx_driver", "pa", "trx",
             "antenna", "tx_audio"]


def _ir(tmp_path: Path, answers: dict[str, str], name: str = "trx") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def _plan(ir: CircuitIR, *, confirmed: bool = True, library: KicadLibrary = _REAL) -> Plan:
    inputs, unusable = read_inputs(ir)
    plan = KR447_TRANSCEIVER.build(ir, inputs, unusable, library, confirmed=confirmed)
    if plan.buildable:
        assert add_board(KR447_TRANSCEIVER, ir, plan, confirmed=confirmed) is None
    return plan


def _apply(ir: CircuitIR, plan: Plan) -> CircuitIR:
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale)
                                      for c in plan.changes])
    return ir


def _applied(tmp_path: Path, answers: dict[str, str], name: str = "trx") -> tuple[CircuitIR, Plan]:
    ir = _ir(tmp_path, answers, name)
    plan = _plan(ir)
    assert plan.buildable, plan.notes
    return _apply(ir, plan), plan


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[CircuitIR, Plan]]:
    """Both builds with the base inputs, applied once for the module's read-only tests (a test that changes an IR copies it)."""
    if not HAS_LIBS:
        pytest.skip("KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")
    return {v: _applied(tmp_path_factory.mktemp(v), {**BASE, "radio_build": v}) for v in VARIANTS}


def _run(validator_id: str, ir: CircuitIR, tmp_path: Path) -> dict:
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": _REAL})
    return {r.check_id: r for r in default_registry.get(validator_id).validate(ir, ctx)}


def _engine_numbers(latest: dict) -> tuple[float, int]:
    """Engine seconds and rawfile bytes of the design deck (``spice``) and of every fixture deck (the ``spice.rf.<network>`` summaries)."""
    runs = list(latest["spice"].details["analyses"].values())
    for check_id, r in latest.items():
        if check_id.startswith("spice.rf.") and "decks" in r.details:
            runs += [a for deck in r.details["decks"] for a in deck["analyses"].values()]
    return sum(a["elapsed_s"] for a in runs), sum(Path(a["raw_output_path"]).stat().st_size for a in runs if a["raw_output_path"])


# --------------------------------------------------------------------------- the contract (no library needed)


def test_template_contract_is_the_family_table() -> None:
    t = KR447_TRANSCEIVER
    for build in VARIANTS:
        assert family.BUILDS[build].template_id == t.id == "kr447_transceiver"
        assert t.needs == family.BUILDS[build].needs == ("carrier_frequency", "modulation", "input_voltage") and t.serves == family.BUILDS[build].serves
    assert t.layer_policy.allowed == (4,) and t.layer_policy.default == 4 and t.layer_policy.reason == PLANE_REASON
    assert t.plane_nets == ("GND", None) and t.version == TEMPLATE_VERSION


@pytest.mark.parametrize("build", family.BUILDS)
def test_both_builds_select_it_and_no_other_build_does(tmp_path: Path, build: str) -> None:
    ir = _ir(tmp_path, {"radio_build": build})
    inputs, _ = read_inputs(ir)
    assert KR447_TRANSCEIVER.triggered_by(ir, inputs) is (build in VARIANTS)


def test_the_closed_world_names_the_builds_and_refuses_non_fm(tmp_path: Path) -> None:
    ir = _ir(tmp_path, {**BASE, "output_voltage": "5 V", "modulation": "AM"})
    inputs, unusable = read_inputs(ir)
    questions = {q.key: q for q in KR447_TRANSCEIVER.refusals(ir, inputs, unusable)}
    assert set(questions) == {"output_voltage", "modulation"} and not any(q.required for q in questions.values())
    assert "output_voltage is served by no radio_build of the KR 447 MHz family" in questions["output_voltage"].rationale
    assert "the KR 447 MHz licence-exempt class is FM (F3E) [UNVERIFIED" in questions["modulation"].rationale
    served = _ir(tmp_path, EVERYTHING)  # every key of the family table is served: nothing to refuse
    inputs, unusable = read_inputs(served)
    assert KR447_TRANSCEIVER.refusals(served, inputs, unusable) == []


def test_missing_inputs_are_questions_and_nothing_is_built(tmp_path: Path) -> None:
    empty = KicadLibrary(roots=[tmp_path / "empty"])
    ir = _ir(tmp_path, {"radio_build": "transceiver", "modulation": "fm"})
    plan = _plan(ir, library=empty)
    assert not plan.buildable and {q.key for q in plan.questions} == {"carrier_frequency", "input_voltage"}
    assert all(q.required for q in plan.questions) and "447.5625 MHz" in next(q for q in plan.questions if q.key == "carrier_frequency").question
    plan = _plan(_ir(tmp_path, {"radio_build": "transceiver_conducted", "input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz"}), library=empty)
    assert not plan.buildable and [q.key for q in plan.questions] == ["modulation"] and plan.questions[0].required


@pytest.mark.parametrize(("extra", "why"), [
    ({"carrier_frequency": "447 MHz"}, "a band such as"),
    ({"input_voltage": "6.5 V"}, "input_voltage 6.5 V (req.input_voltage) is outside 6.6..8.4 V"),
    ({"tx_power": "2 W"}, "tx_power 2 W (req.tx_power) is outside 0.01..1 W"),
    ({"antenna_impedance": "5 ohm"}, f"antenna_impedance 5 ohm (req.antenna_impedance) is outside {ANT_R_RANGE[0]:g}..{ANT_R_RANGE[1]:g} ohm"),
    ({"frequency_deviation": "8 kHz"}, "frequency_deviation 8000 Hz (req.frequency_deviation) is outside 500..5000 Hz"),
    ({"tx_timeout": "2 s"}, "tx_timeout 2 s (req.tx_timeout) is outside 10..600 s"),
])
def test_out_of_range_inputs_and_a_carrier_off_the_raster_are_refused(tmp_path: Path, extra: dict[str, str], why: str) -> None:
    plan = _plan(_ir(tmp_path, {**BASE, **extra}), library=KicadLibrary(roots=[tmp_path / "empty"]))
    assert not plan.buildable and not plan.questions and why in plan.notes[-1], plan.notes


def test_a_stated_two_layer_board_is_refused_before_anything_is_asked(tmp_path: Path) -> None:
    ir = _ir(tmp_path, {"radio_build": "transceiver", "pcb_layers": "2"})
    plan = design_from_requirements(ir, KicadLibrary(roots=[tmp_path / "empty"]))
    assert plan is not None and plan.template == "kr447_transceiver" and not plan.buildable
    assert [q.key for q in plan.questions] == ["pcb_layers"] and not plan.questions[0].required and PLANE_REASON in plan.questions[0].rationale


def test_the_board_module_writes_a_plans_keepouts_and_never_replaces_one(tmp_path: Path) -> None:
    """``Plan.keepouts`` go into the new board with the stack, or are appended to an existing ``ir.pcb``; a clash refuses; none changes nothing."""
    divider = next(t for t in TEMPLATES if t.id == "divider")
    ko = Keepout(id="band", layers=["*.Cu"], rect=assumption([0.0, 0.0, 10.0, 2.0], "test", "mm"), forbids=["tracks"], reason="test band")

    def plan(keepouts: list[Keepout]) -> Plan:
        p = Plan(template=divider.id, keepouts=list(keepouts))
        p.changes.append(DesignChange(description="x", target="topology", operation="set", payload=None))
        return p

    ir = _ir(tmp_path, {})
    without, with_ko = plan([]), plan([ko])
    assert add_board(divider, ir, without, confirmed=True) is None and add_board(divider, ir, with_ko, confirmed=True) is None
    board = lambda p: next(c.payload for c in p.changes if c.target == "pcb")  # noqa: E731
    assert board(without).keepouts == [] and [k.id for k in board(with_ko).keepouts] == ["band"]
    assert not any(line.startswith("keep-out") for line in without.board)
    assert "keep-out band: rect x 0 y 0 w 10 h 2 mm on *.Cu, no tracks - test band" in with_ko.board
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10.0, height_mm=10.0))
    appended = plan([ko])
    assert add_board(divider, ir, appended, confirmed=True) is None
    assert [c.payload.id for c in appended.changes if c.target == "pcb.keepouts" and c.operation == "append"] == ["band"]
    _apply(ir, appended)  # the whole board proposal applies through the orchestrator: the keep-out lands beside the existing outline
    assert [k.id for k in ir.pcb.keepouts] == ["band"] and ir.pcb.outline.width_mm == 10.0
    assert "already exist in ir.pcb.keepouts" in (add_board(divider, ir, plan([ko]), confirmed=True) or "")


# --------------------------------------------------------------------------- the plan (the real 10.0.6 libraries)


def test_the_pa_and_trx_block_titles_name_only_what_the_build_holds() -> None:
    """``PaBlock`` / ``TrxBlock`` titles follow their options (they reach ``ir.rf.blocks`` and the topology): the exciter's PA keeps its full
    wording, the transceiver's PA (no low-pass, no connector) and a trx block without the match say neither."""
    from ai_eda.design.rf.blocks.pa import PaBlock
    from ai_eda.design.rf.blocks.trx import TrxBlock

    assert PaBlock().title == "PA, load-line match, harmonic low-pass and the conducted output"
    assert PaBlock(with_output=False).title == "PA, load-line match and harmonic low-pass"
    bare = PaBlock(with_lpf=False, with_output=False, out_net="TX_RF").title
    assert bare.startswith("PA and load-line match into TX_RF") and "conducted" not in bare
    assert TrxBlock().title == "PIN T/R switch and harmonic low-pass" and TrxBlock(with_match=True).title.endswith("and antenna L-match")


@needs_libs
def test_the_confirmation_table_says_unverified_on_every_profile_and_model_row(built) -> None:
    for variant, (_ir_, plan) in built.items():
        table = plan.table()
        rows = [line for line in table.splitlines() if line.lstrip().startswith(("kr447.", "model."))]
        assert rows and all("UNVERIFIED" in line for line in rows), [r for r in rows if "UNVERIFIED" not in r][:3]
        assert {k for k in profile_keys()} <= {line.split()[0] for line in rows}, variant
        assert plan.title == family.BUILDS[variant].title


@needs_libs
def test_the_board_composes_every_block_with_its_ports_regions_and_fixtures(built) -> None:
    ir, plan = built["transceiver"]
    assert len(ir.components) == 441 and len(ir.nets) == 211
    rf = ir.rf
    assert rf is not None and [b.id for b in rf.blocks] == BLOCK_IDS
    assert all(b.region is not None for b in rf.blocks) and {b.id: b.shield_ref for b in rf.blocks if b.shield_ref} == {
        "lo_chain": "SH701", "rx_frontend": "SH601", "tx_mod": "SH802", "tx_chain": "SH801"}
    assert len(rf.networks) == 23 and {"trsw", "lpf", "ant_end"} <= {n.id for n in rf.networks} and "ant_match" not in {n.id for n in rf.networks}
    assert rf.profile_keys == profile_keys() and "model.pin.r_on" in rf.model_values and "model.diode" in rf.model_values
    by = {b.id: b for b in rf.blocks}
    assert {p.name for p in by["trx"].ports} == {"TX_RF", "RX_RF", "TX_5V", FEED_NET} and by["antenna"].ports == []  # no antenna impedance: no port
    assert {p.net for p in by["pa"].ports} == {"PA_IN", "PA_5V", "TX_RF"} and {p.net for p in by["if_backend"].ports} == {"IF1", "RX_5V"}
    nets = {n.name for n in ir.nets}
    assert {"IFB_LAD_IN", "IFB_MIX_IN", "LIM_OUT", "TX_RF", "RX_RF", "TRSW_COM", "LPF_IN", FEED_NET} <= nets and "TX_OUT" not in nets
    refs = {c.ref for c in ir.components}
    assert {"D1001", "D1002", "L1001", "C1001", "C1002", "R1001", "R1002", "L1002", "C1013", "ANT1", "U901", "U501", "Q601", "U650", "U750", "U850"} <= refs
    assert not {"J501", "J601", "J602", "J901", "C913", "L903"} & refs  # the stage boards' U.FLs and the PA's own low-pass are gone
    # kr447 wave-2 review findings 1 / 2 on the composed board: no two top-C filters meet on a net, and the multiplier chains' networks hold
    # every part on their port nets (the TX band-pass's pad port stands for the matched drv_pad, whose own fixture proves its input)
    top_c = ["lo_tank1", "lo_tank2", "lo_bpf", "fe_bpf2", "fe_bpf3", "tx_tank1", "tx_tank2", "tx_bpf"]
    assert nets_joining_networks(ir, top_c) == {}
    chains = ["lo_tank1", "lo_tank2", "lo_bpf", "tx_tank1", "tx_tank2", "tx_bpf"]
    assert port_net_outsiders(ir, chains, allowed=rf.network("drv_pad").members) == []
    gated = next(line for line in rf.frequency_plan if line.id == "tx_ref_on_rx_channel")
    assert gated.kind == "gated" and gated.points_to == ["spice.tx_rail_off_rx", "rf.lab.tr_sequencing"] and gated.f_hz.value == gated.ref_hz.value
    assert {"trsw_power", "tr_sequencing", "antenna_match", "antenna_length", "kc_conformity", "sensitivity"} <= {x.id for x in rf.lab_items}
    # the block titles are design data: this build's PA has no low-pass and no conducted output, its trx block no antenna match
    titles = {b.id: b.title for b in rf.blocks}
    assert "conducted" not in titles["pa"] and "low-pass" not in titles["pa"].split("(")[0] and "match" not in titles["trx"]
    assert all(blk.function == titles[blk.id] for blk in ir.topology.blocks if blk.id in titles)
    pa_lab = {x.id: x for x in rf.lab_items if x.block == "pa"}
    assert "dummy load" not in pa_lab["pa_power"].what and "conducted output" not in pa_lab["pa_harmonics"].what
    p = ir.parameters
    assert p["trx.lq.l"].value == pytest.approx(17.780e-9, rel=1e-3) and p["trx.lq.c"].value == pytest.approx(7.112e-12, rel=1e-3)
    assert p["trx.pin.r"].value == pytest.approx(340.0) and p["trx.ant.length"].value == pytest.approx(0.15909, rel=1e-4)
    assert p["trx.sensitivity"].value == pytest.approx(-112.84, abs=0.05) and p["trx.nf.total"].provenance.tool == "calc.rf.db_sum"
    pin_bias = next(e for e in ir.simulation.expectations if e.id == "pin_bias")
    assert pin_bias.vector == "i(R1002)" and pin_bias.nominal.value == pytest.approx(0.01) and pin_bias.tol_rel.value == pytest.approx(0.2)
    assert {c.id for c in ir.constraints} >= {"c.kr447.kc_before_transmission", "c.kr447.integral_antenna", "c.kr447.tr_sequencing"}


@needs_libs
def test_the_builds_differ_exactly_in_the_antenna_end(built) -> None:
    (ant, ant_plan), (cond, cond_plan) = built["transceiver"], built["transceiver_conducted"]
    assert {c.ref for c in ant.components} ^ {c.ref for c in cond.components} == {"ANT1", "J1001"}
    # every other part is the same in the design view (the hashed content: no clock, no locator)
    a_view, c_view = ant.design_dict(), cond.design_dict()
    a_parts = {c["ref"]: c for c in a_view["components"]}
    c_parts = {c["ref"]: c for c in c_view["components"]}
    assert [r for r in a_parts if r != "ANT1" and a_parts[r] != c_parts[r]] == []
    a_par, c_par = a_view["parameters"], c_view["parameters"]
    assert a_par.keys() - c_par.keys() == {"floor.ant_band.h", "trx.ant.vf", "trx.ant.length"} and c_par.keys() <= a_par.keys()
    assert [k for k in c_par if a_par[k] != c_par[k]] == []
    assert [n.name for n in ant.nets] == [n.name for n in cond.nets]
    diff = {n.name for n in ant.nets for m in cond.nets if m.name == n.name and [p.component_ref for p in n.pins] != [p.component_ref for p in m.pins]}
    assert diff == {FEED_NET, "GND"}  # the feed part, and the U.FL's ground shell
    choices = lambda plan: {c.key: c for c in plan.choices}  # noqa: E731
    a, c = choices(ant_plan), choices(cond_plan)
    assert set(a) - set(c) == {"trx.ant.vf", "floor.ant_band.h"} and set(c) - set(a) == set()
    assert all(a[k] == c[k] for k in set(a) & set(c))
    assert {k for k, _ in ant_plan.computed} - {k for k, _ in cond_plan.computed} == {"trx.ant.length"}
    assert [k.id for k in ant.pcb.keepouts] == ["ant_band", "trx_bcu"] and [k.id for k in cond.pcb.keepouts] == ["trx_bcu"]
    assert [n.id for n in ant.rf.networks] == [n.id for n in cond.rf.networks]
    labs = lambda ir: {x.id for x in ir.rf.lab_items}  # noqa: E731
    assert labs(ant) - labs(cond) == {"antenna_length", "antenna_radiation", "radiated_spurious", "sar"}
    assert labs(cond) - labs(ant) == {"conducted_sample"}
    assert {x.id for x in ant.constraints} ^ {x.id for x in cond.constraints} == {"c.kr447.integral_antenna", "c.kr447.conducted_sample"}
    assert [p.name for p in next(b for b in cond.rf.blocks if b.id == "antenna").ports] == [FEED_NET]  # the U.FL is a system-impedance port


@needs_libs
def test_rebuilds_are_byte_identical(tmp_path: Path) -> None:
    outs = []
    for k in (1, 2):
        d = tmp_path / f"b{k}"
        d.mkdir()
        ir, plan = _applied(d, BASE)
        fp = rf_floorplan_placement(ir, _REAL)
        ir.pcb.outline, ir.pcb.placements = fp.outline, fp.placements
        files = [Path(compiler.compile(ir, CompileContext(workdir=d, tools={"kicad_library": _REAL})).path).read_bytes()
                 for compiler in (SchematicCompiler(), PCBCompiler())]
        outs.append((ir.content_hash(), files, plan.table()))
    assert outs[0] == outs[1]
    assert b"keepout_ant_band" in outs[0][1][1] and b"keepout_trx_bcu" in outs[0][1][1]  # the rule areas of the keep-outs


@needs_libs
@pytest.mark.parametrize("answers", [BASE, CONDUCTED, EVERYTHING, {**CONDUCTED, "tx_timeout": "60 s", "antenna_impedance": "36 ohm"}],
                         ids=["transceiver", "conducted", "everything", "conducted_tot"])
def test_the_floorplan_places_every_build_on_the_boards_outline(tmp_path: Path, answers: dict[str, str]) -> None:
    ir, _plan_ = _applied(tmp_path, answers)
    t0 = time.perf_counter()
    fp = rf_floorplan_placement(ir, _REAL)
    elapsed = time.perf_counter() - t0
    assert (fp.outline.width_mm, fp.outline.height_mm) == OUTLINE_MM and len(fp.placements) == len(ir.components)
    assert elapsed < 5.0, f"{elapsed:.2f} s"  # about 0.1 s on this machine
    assert set(fp.cans.values()) == {"SH601", "SH701", "SH801", "SH802"}
    assert all(fp.block_of[c.ref] for c in ir.components)  # every part in its block's region
    feed = "ANT1" if answers["radio_build"] == "transceiver" else "J1001"
    box = fp.extents[feed]
    assert box.y1 >= 2.0 - 1e-9 and (box.y2 <= BAND_H_MM if feed == "ANT1" else box.y2 <= 8.0)  # at the top edge, inside the band / its region
    ir.pcb.outline, ir.pcb.placements = fp.outline, fp.placements
    ko = _run("pcb.keepout", ir, tmp_path)["pcb.keepout"]
    # nothing violates a keep-out, a region or a can fence, and every pad's copper is bounded - the three custom pads (the microphone's
    # and the two PHA-1s') by the boxes of their anchor and primitives
    assert ko.status is S.PASS and ko.details["unknown"] == [] and [r for r in ko.details["rows"] if r["status"] != "PASS"] == [], ko.message
    assert len(ko.details["regions"]) == len(BLOCK_IDS) + 4  # every region and every can fence was judged


#: where the user-facing parts land on both builds (kr447 wave-2 review finding 15): (left, right, top, bottom) edge distances in mm, as the
#: template's docstring and its Korean floorplan section state them - a deviation from the design's §2.5 (pots, mic and connectors on the
#: bottom edge) left to manual placement; SW201 is on the left edge as §2.5 asks
UI_EDGES_MM = {
    "SW201": (2.0, 49.5, 97.5, 139.5), "RV401": (12.2, 33.4, 185.5, 44.2), "RV402": (2.0, 43.6, 199.8, 29.9), "MK301": (2.0, 53.5, 67.5, 171.0),
    "J401": (44.9, 8.2, 185.5, 52.0), "J101": (2.0, 51.1, 218.5, 19.0),
}


@needs_libs
def test_the_user_facing_parts_land_where_the_template_says(built) -> None:
    """The floorplan does not put the pots, the mic and the connectors on the bottom edge; the template states where they land instead."""
    doc = " ".join((transceiver_module.__doc__ or "").split())
    for variant in VARIANTS:
        ir, _plan_ = built[variant]
        fp = rf_floorplan_placement(ir, _REAL)
        w, h = fp.outline.width_mm, fp.outline.height_mm
        for ref, want in UI_EDGES_MM.items():
            b = fp.extents[ref]
            assert (b.x1, w - b.x2, b.y1, h - b.y2) == pytest.approx(want, abs=0.05), (variant, ref)
    for sentence in ("``RV401`` (volume) is 44.2 mm and ``RV402`` (squelch) 29.9 mm above the bottom edge", "``J101`` (pack) is on the left edge 19 mm above it",
                     "``J401`` (speaker) is 52 mm above it", "``MK301`` is on the left edge 171 mm above it"):
        assert sentence in doc, sentence
    ir, _plan_ = built[VARIANTS[0]]
    section = next(s.body for s in KR447_TRANSCEIVER.theory(ir) if s.title.startswith("기판 배치"))
    assert "RV401(음량)은 아래쪽 가장자리에서 44.2 mm" in section and "J101(팩)은 왼쪽 가장자리에서 아래쪽으로부터 19 mm 위" in section


@needs_libs
def test_the_designs_60_by_145_mm_outline_cannot_hold_the_composed_board(built) -> None:
    """Why the outline is 60 x 243 mm, not the kr447 design's 60 x 145 mm: a strict area bound. The placer keeps every part MARGIN_MM inside
    the edge and SPACING_MM from every other part (a can holds its block's parts inside its fence), so the parts' extents grown by SPACING_MM
    are disjoint inside the outline shrunk by MARGIN_MM and grown by SPACING_MM - their area already exceeds it (the antenna band not even
    counted)."""
    ir, _plan_ = built["transceiver"]
    inside = {r for b in ir.rf.blocks if b.shield_ref for r in b.refs if r != b.shield_ref}
    extents = [footprint_extent(_REAL.load_footprint(c.footprint)) for c in ir.components if c.ref not in inside]
    needed = sum((e.width + SPACING_MM) * (e.height + SPACING_MM) for e in extents)
    usable = (60.0 - 2 * MARGIN_MM + SPACING_MM) * (145.0 - 2 * MARGIN_MM + SPACING_MM)
    assert needed > usable, (needed, usable)
    assert needed <= (OUTLINE_MM[0] - 2 * MARGIN_MM + SPACING_MM) * (OUTLINE_MM[1] - 2 * MARGIN_MM + SPACING_MM)


@needs_libs
def test_every_optional_input_is_served(tmp_path: Path) -> None:
    ir, plan = _applied(tmp_path, EVERYTHING)
    assert len(ir.components) == 453 and "ant_match" in {n.id for n in ir.rf.networks} and {"L1003", "C1006", "U203"} <= {c.ref for c in ir.components}
    served = {rid for c in ir.components for rid in c.serves_requirements} | {rid for n in ir.nets for rid in n.serves_requirements}
    design = {r.id for r in ir.requirements.requirements if r.category in {"electrical", "rf", "mechanical", "thermal", "signal_integrity", "power_integrity"}}
    assert design and design <= served, sorted(design - served)
    obw = next(r.id for r in ir.requirements.requirements if r.key == "occupied_bandwidth")
    assert {c.ref for c in ir.components if obw in c.serves_requirements} >= {"RV301", "C313", "C314", "C316", "C317"}
    p = ir.parameters
    assert p["ant.eirp_dbm"].value == pytest.approx(26.9897 + 2.15 - 1.5, abs=1e-3) and p["ant.fspl"].provenance.tool == "calc.rf.fspl"
    assert p["ant.e_at_range"].unit == "V/m" and p["trx.ant.r"].value == 36.0
    assert "RF_ANT" in {c.name for c in ir.si.net_classes}
    ant = next(b for b in ir.rf.blocks if b.id == "antenna")
    assert [(q.name, q.z0_ohm.value) for q in ant.ports] == [(FEED_NET, 36.0)]


@needs_libs
def test_the_rf_checks_judge_the_interfaces_and_a_tx_power_above_the_profile_fails(built, tmp_path: Path) -> None:
    """IR arithmetic without ngspice: every block port agrees with its partners; a stated tx_power above the kr447.max_power placeholder
    builds (the PA is designed for it) and the profile check FAILs - two confirmed choices contradict each other, never a legal verdict."""
    ir, _plan_ = built["transceiver"]
    got = {r.check_id: r for r in rf_results(ir)}
    interfaces = {k: r for k, r in got.items() if k.startswith("block.interface.")}
    assert {"block.interface.TX_RF", "block.interface.RX_RF", "block.interface.IF1", "block.interface.LO1_MIX", "block.interface.PA_5V"} <= set(interfaces)
    assert all(r.status is S.PASS for r in interfaces.values()), [(k, r.message) for k, r in interfaces.items() if r.status is not S.PASS]
    assert got["rf.regulatory_profile"].status is S.NOT_VERIFIED and got["rf.model_grounding"].status is S.NOT_VERIFIED
    loud, _plan2 = _applied(tmp_path, {**BASE, "tx_power": "0.8 W"})
    assert loud.parameters["pa.p_out"].value == pytest.approx(0.8)
    profile = next(r for r in rf_results(loud) if r.check_id == "rf.regulatory_profile")
    assert profile.status is S.FAIL and "kr447.max_power" in profile.message, profile.message


@needs_libs
def test_the_si_classes_name_existing_nets_once(built) -> None:
    for ir, _plan_ in built.values():
        nets = {n.name for n in ir.nets}
        members = [n for c in ir.si.net_classes for n in c.nets]
        assert set(members) <= nets and len(members) == len(set(members))
        by = {c.name: c for c in ir.si.net_classes}
        assert {"TX_RF", "TRSW_COM", "LPF_IN", "PA_MATCH"} <= set(by["RF50_H"].nets) and {"RX_RF", FEED_NET, "LO1_MIX"} <= set(by["RF50"].nets)
        assert "IFB_LAD_IN" in by["IF_HIZ"].nets and set(by["POWER_PA"].nets) == {"VBAT", "VBAT_F", "V_SYS", "V_TX", "TX_5V", "PA_5V"}


@needs_libs
def test_the_korean_views_cover_the_board(built) -> None:
    ir, _plan_ = built["transceiver"]
    t = KR447_TRANSCEIVER
    sections = t.theory(ir)
    assert sections[0].title.startswith("개요") and any(s.title.startswith("PIN T/R 스위치") for s in sections)
    assert any(s.title.startswith("[4단계 송신 여진기]") for s in sections) and not any("적분기와 주파수 편이" in s.title for s in sections)
    own = sections[:7]
    assert all("기록 없음" not in s.body for s in own) and "UNVERIFIED" in sections[0].body
    assert "R219" not in "".join(s.body for s in sections)  # the stage-1 bench sentence is rewritten for this board
    assert "60 × 243 mm" in next(s.body for s in sections if s.title.startswith("기판 배치"))
    notes = t.part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components} and notes["ANT1"].role.startswith("일체형") and "PIN" in notes["D1001"].role
    figures = t.theory_figures(ir)
    ids = [f.id for f in figures]
    assert ids[0] == "theory_ant_length" and len(ids) == len(set(ids)) and {"theory_tx_filters", "theory_fe_s21", "theory_quad_phase"} <= set(ids)
    cond, _p = built["transceiver_conducted"]
    assert "J1001" in "".join(s.body for s in t.theory(cond)) and "theory_ant_length" not in [f.id for f in t.theory_figures(cond)]


@needs_libs
def test_the_stage_reports_are_deterministic_views(built) -> None:
    """The four Korean stage reports of the placed board: built twice, byte-identical; the template's own sections, part notes and figures in."""
    from ai_eda.report.stages import build_stage_document

    ir = built["transceiver"][0].model_copy(deep=True)
    fp = rf_floorplan_placement(ir, _REAL)
    ir.pcb.outline, ir.pcb.placements = fp.outline, fp.placements
    docs = {}
    for stage in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION, Stage.PCB, Stage.RELEASE):
        first, again = build_stage_document(stage, ir, _REAL, None), build_stage_document(stage, ir, _REAL, None)
        assert first.markdown == again.markdown and len(first.markdown) > 1000, stage
        docs[stage] = first.markdown
    assert "PIN T/R 스위치" in docs[Stage.ARCHITECTURE] and "![fig](fig:theory_ant_length)" in docs[Stage.ARCHITECTURE]
    assert "T/R 스위치 PIN 다이오드" in docs[Stage.COMPONENT_SELECTION] and "일체형 λ/4 도선 안테나" in docs[Stage.COMPONENT_SELECTION]


@needs_libs
def test_the_full_board_is_placed_but_the_router_refuses_a_pad_below_its_grid(built, tmp_path: Path) -> None:
    ir, _plan_ = built["transceiver"]
    res = PCBAgent().run(ir.model_copy(deep=True), AgentContext(workdir=tmp_path, tools={"kicad_library": _REAL}))
    board = res.proposals[0].payload
    notes = " | ".join(res.notes)
    assert len(board.placements) == 441 and board.tracks == [] and "keep-outs ant_band honoured" in notes
    # routing.maze 0.5 reads the custom pads (the microphone's ring, the PHA-1s' SOT-89-3 tabs) and refuses the board before routing at
    # the first pad too small for the 0.2 mm grid: the PA's 0.25 mm-wide QFN pad, whose nearest grid point is outside its inscribed circle
    assert "not routed: pad U901.1 (0.825 x 0.25 mm) is too small for the 0.2 mm routing grid" in notes, notes
    assert "shape 'custom'" not in notes


@needs_libs
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_antenna_end_alone_routes_on_the_real_footprints(built, tmp_path: Path, variant: str) -> None:
    """The reduced routing case: the trx and antenna blocks of the applied board, on a 60 x 20 mm outline (the full board is not routable here)."""
    full, _plan_ = built[variant]
    keep = [b for b in full.rf.blocks if b.id in ("antenna", "trx")]
    refs = {r for b in keep for r in b.refs}
    ir = full.model_copy(deep=True)
    ir.components = [c for c in full.components if c.ref in refs]
    ir.nets = [n.model_copy(update={"pins": [p for p in n.pins if p.component_ref in refs]}) for n in full.nets if any(p.component_ref in refs for p in n.pins)]
    ir.rf, ir.si, ir.simulation = RFDesign(blocks=keep), None, None
    ir.pcb.outline = BoardOutline(width_mm=60.0, height_mm=20.0)
    t0 = time.perf_counter()
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": _REAL}))
    elapsed = time.perf_counter() - t0
    Orchestrator.apply_proposals(ir, res.proposals)
    notes = " | ".join(res.notes)
    assert ir.pcb.tracks and "10 net(s) routed" in notes and "not applied" not in notes, notes
    assert elapsed < 60.0, f"{elapsed:.1f} s"
    results = {**_run("pcb.routing", ir, tmp_path), **_run("pcb.keepout", ir, tmp_path)}
    assert results["pcb.keepout"].status is S.PASS, results["pcb.keepout"].message
    assert results["pcb.routing.connectivity"].status is S.NOT_VERIFIED and "joined only by a copper pour" in results["pcb.routing.connectivity"].message
    in_band = {t.net for t in ir.pcb.tracks if min(t.start[1], t.end[1]) < BAND_H_MM}
    assert in_band == ({FEED_NET} if variant == "transceiver" else in_band) and FEED_NET in {t.net for t in ir.pcb.tracks}
    # trx_bcu bans B.Cu tracks under the switch / low-pass but exempts GND: the shunt parts' ground vias sit inside the region
    tx, ty, tw, th = next(k for k in ir.pcb.keepouts if k.id == "trx_bcu").rect.value
    inside = [v for v in ir.pcb.vias if tx <= v.x_mm <= tx + tw and ty <= v.y_mm <= ty + th]
    assert inside and {v.net for v in inside} == {"GND"}, [(v.net, v.x_mm, v.y_mm) for v in ir.pcb.vias]
    assert not [t for t in ir.pcb.tracks if t.layer == "B.Cu" and t.net != "GND" and ty <= t.start[1] <= ty + th and tx <= t.start[0] <= tx + tw]


# --------------------------------------------------------------------------- ngspice


@needs_libs
@needs_ngspice
def test_every_fixture_passes_within_the_budget(built, tmp_path: Path) -> None:
    ir, _plan_ = built["transceiver"]
    t0 = time.perf_counter()
    results = spice_rf_results(ir, {"spice": runner}, tmp_path)
    elapsed = time.perf_counter() - t0
    got = {r.check_id: r for r in results}
    assert len(results) == 112 and "spice.rf.ant_end" in got and {f"spice.rf.{n}" for n in TRX_NETWORKS if n != "ant_match"} <= set(got)
    assert all(r.status is S.PASS for r in results), [(r.check_id, r.message) for r in results if r.status is not S.PASS]
    measured = {k: got[k].details["measured"] for k in got if k.startswith("spice.rf.trsw.") or k.startswith("spice.rf.lpf.")}
    assert measured["spice.rf.trsw.tx.s21_tx_com"] == pytest.approx(-0.335, abs=0.02)
    assert measured["spice.rf.trsw.tx.s21_tx_rx"] == pytest.approx(-34.49, abs=0.1)
    assert measured["spice.rf.trsw.rx.s21_com_rx"] == pytest.approx(-0.248, abs=0.02)
    assert measured["spice.rf.trsw.rx.s21_com_tx"] == pytest.approx(-27.59, abs=0.1)
    assert measured["spice.rf.lpf.s21_fc"] == pytest.approx(-1.045, abs=0.02) and measured["spice.rf.lpf.s21_2fc"] == pytest.approx(-52.78, abs=0.1)
    assert elapsed < FIXTURE_BUDGET_S, f"{elapsed:.2f} s: the kr447 design's §3.3 budget for the transceiver's fixtures is < {FIXTURE_BUDGET_S} s"


@needs_libs
@needs_ngspice
def test_the_antenna_match_fixture_passes_with_a_confirmed_antenna_impedance(tmp_path: Path) -> None:
    """With antenna_impedance 36 ohm the transceiver build carries the L-match and its fixture (s21 at least -0.5 dB, s11 at most -15 dB)."""
    ir, _plan_ = _applied(tmp_path, {**BASE, "antenna_impedance": "36 ohm"})
    ir.rf.networks = [n for n in ir.rf.networks if n.id in ("ant_match", "lpf")]
    results = {r.check_id: r for r in spice_rf_results(ir, {"spice": runner}, tmp_path)}
    assert {"spice.rf.ant_match", "spice.rf.ant_match.s21_fc", "spice.rf.ant_match.s11_fc"} <= set(results)
    assert all(r.status is S.PASS for r in results.values()), [(k, r.message) for k, r in results.items() if r.status is not S.PASS]
    assert results["spice.rf.ant_match.s11_fc"].details["measured"] < -30.0  # the lossless L-match is exact at f_c; the inductor's Q-40 loss remains
    ports = {p.name: p.z0_ohm.value for p in next(n for n in ir.rf.networks if n.id == "ant_match").ports}
    assert ports == {"ant_port": 50.0, "ant_feed": 36.0}


@needs_libs
@needs_ngspice
def test_the_pipeline_runs_the_deck_and_the_rf_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Present, confirm, simulate (placement skipped: the placement tests cover it): every deck row PASS, the interfaces agree, no FAIL."""
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [KR447_TRANSCEIVER])
    ir = _ir(tmp_path, {})
    answers = {**BASE, "frequency_deviation": "2.5 kHz", "audio_bandwidth": "3 kHz"}
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _REAL},
                                      answers={"application": "bench", "jurisdiction": "KR", **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"] and ir.components == []
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _REAL, "spice": runner},
                                      answers={"confirm_design": "yes", "pcb.placement": "skip"})).run(ir, stop_after=Stage.SPICE)
    assert not state.blocked and len(ir.components) == 441
    latest = ir.validation.latest_by_check()
    status = {k: r.status for k, r in latest.items()}
    deck = {k for k in status if k.startswith("spice.") and not k.startswith("spice.rf.")}
    assert {"spice.pin_bias", "spice.tx_rail_off_rx", "spice.rx_rail_off_tx", "spice.pa_supply_off_first", "spice.ic_lna", "spice.tx_ic_x12",
            "spice.pm_couple_1k", "spice.integrator_1k", "spice.deemph_1k", "spice.main_switch_on"} <= deck
    assert all(status[k] is S.PASS for k in deck), [(k, latest[k].message) for k in deck if status[k] is not S.PASS]
    interfaces = [k for k in status if k.startswith("block.interface.")]
    assert {"block.interface.TX_RF", "block.interface.RX_RF", "block.interface.IF1", "block.interface.LO1_MIX", "block.interface.TX_5V"} <= set(interfaces)
    assert all(status[k] is S.PASS for k in interfaces), [(k, latest[k].message) for k in interfaces if status[k] is not S.PASS]
    assert status["rf.deviation"] is S.PASS and "frequency_deviation 2.5 kHz under confirmed model values" in latest["rf.deviation"].message
    assert status["calc.recompute"] is S.PASS and status["rf.freq_plan"] is S.NOT_VERIFIED
    # the kr447 design's §3.3 budget for the transceiver: the fixture decks and the union deck < 10 s of engine time, < 60 MB of rawfiles
    seconds, raw = _engine_numbers(latest)
    assert seconds < FIXTURE_BUDGET_S and raw < RAWFILE_BUDGET_BYTES, (seconds, raw)
    assert "tx_ref_on_rx_channel" in {row["id"] for row in latest["rf.freq_plan"].details["rows"]}
    for check in ("rf.model_grounding", "rf.regulatory_profile", "rf.lab.trsw_power", "rf.lab.antenna_match", "rf.lab.kc_conformity"):
        assert status[check] is S.NOT_VERIFIED, (check, latest[check].message)
    assert [k for k, s in status.items() if s is S.FAIL] == []


def test_the_template_class_is_the_registered_instance() -> None:
    assert isinstance(KR447_TRANSCEIVER, Kr447TransceiverTemplate) and set(REGIONS) == set(BLOCK_IDS)
