"""The ``kr447_audio_ptt`` template (stage 1 of the KR 447 MHz family) and its four blocks, on the real KiCad 10.0.6 libraries.

The template is not registered in ``RF_TEMPLATES`` yet (the main session
registers the family); every test that selects it patches the lazy registry
(``ai_eda.design.templates.rf_templates``) the way ``tests/test_rf_selection.py``
does. What is checked:

* Without any library (the selection and refusal paths never read one):
  ``radio_build = audio_ptt`` selects it, a confirmed requirement another
  build serves refuses with the family's closed-world sentence, AM refuses,
  out-of-range pack / deviation / bandwidth / time-out / layer counts refuse
  with the reason, a missing pack voltage or modulation is asked.
* ``needs_libs`` (the packed 10.0.6 symbol / footprint libraries,
  ``KICAD10_SYMBOL_DIR``): the confirm_design table shows every ``kr447.*``
  profile row and every ``model.*`` row marked UNVERIFIED; a confirmation
  builds 144 parts (154 with the time-out), every derived value recomputes,
  the parts that serve the stated requirements name them, the RF design
  carries the five floorplan regions, the rails, the lab items and the model
  keys; two independent builds are the same design, place the same and
  compile to byte-identical schematic, board and netlist files; the router
  refuses this board (the microphone footprint's custom pad) - so the
  deliverable is placement only; the Korean theory / figures / part notes
  render with the design's numbers.
* ``needs_ngspice`` too (ngspice-42 here): the design deck's 27 expectations
  all PASS (principle verdicts under the confirmed model values - no IC is
  simulated), within the design's run-time budget (about 1-3 s of engine
  time, about 15 MB of rawfiles: measured 1.6-1.7 s and 14.3-14.4 MB with
  ngspice-42 on Linux; the assertions allow 3x the time for a loaded machine
  and 20 MB),
  and the whole pipeline runs to RELEASE (routing skipped) with exactly one
  FAIL, the honest one - ``pcb.routing.connectivity``: the placed board has no
  copper, so its nets of known pads are open (the microphone's custom-pad net
  is not judged) - and ``rf.deviation`` / ``rf.model_grounding`` /
  ``rf.regulatory_profile`` / ``rf.lab.*`` / ``power.*`` NOT_VERIFIED,
  ``block.interface.*`` PASS, RELEASE FAIL.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import ai_eda.design.templates as templates_mod
from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.agents.keys import PLACEMENT_KEY, ROUTING_KEY
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler, SpiceNetlistCompiler
from ai_eda.design import RF_REGISTRY_MODULE, UNVERIFIED_SUBSTITUTE, Plan
from ai_eda.design.rf.blocks.base import BlockPrefix
from ai_eda.design.rf.blocks.power import PowerBlock
from ai_eda.design.rf.blocks.ptt import PTT_PLAN_S, PttBlock
from ai_eda.design.rf.blocks.rx_audio import RxAudioBlock
from ai_eda.design.rf.blocks.tx_audio import TxAudioBlock
from ai_eda.design.base import quantity
from ai_eda.design.rf.models import MODEL_PREFIX, MODEL_VERDICT
from ai_eda.design.rf.profile import profile_keys
from ai_eda.design.rf.t_audio_ptt import KR447_AUDIO_PTT, REGIONS, TEMPLATE_ID, stage1_blocks
from ai_eda.ir import CircuitIR, ProjectMeta, ValidationStatus as S
from ai_eda.tools.calc import recompute_parameters
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import Orchestrator, Stage

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
LIB = KicadLibrary()
#: the symbol libraries the board's parts come from, and the footprints only the 10.0.6 libraries hold
HAS_LIBS = all(LIB.symbol_file(lib) is not None for lib in (
    "Amplifier_Audio", "Amplifier_Operational", "Comparator", "74xGxx", "4xxx", "Regulator_Linear", "Transistor_FET", "Transistor_BJT", "Sensor_Audio",
)) and LIB.footprint_file("Sensor_Audio", "CUI_CMC-4013-SMT") is not None and LIB.footprint_file("Package_DFN_QFN", "DFN-14-1EP_3x3mm_P0.4mm_EP1.78x2.35mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the audio / PTT parts not installed (set KICAD10_SYMBOL_DIR)")

#: what the scope questions of a KR run need (they stay non-required) and the stage-1 inputs
BASE = {"application": "bench", "jurisdiction": "KR"}
AUDIO = {"input_voltage": "7.4 V", "modulation": "FM", "radio_build": "audio_ptt"}
#: every input the board serves, the time-out included
FULL = {**AUDIO, "frequency_deviation": "2.5 kHz", "audio_bandwidth": "3 kHz", "tx_timeout": "180 s"}
#: the design deck's expectations (all four blocks), each a principle verdict under the confirmed model values
EXPECTATIONS = (
    "main_switch_on", "tx_rail_on", "tx_rail_off_rx", "rx_rail_off_tx",
    "pa_held_off_while_settling", "pa_enabled_after_settle", "pa_off_fast", "pa_supply_on", "pa_supply_off_first", "uvlo_ref", "uvlo_sense_released",
    "limiter_level", "pm_drive_peak", "splatter_h3", "splatter_h5", "splatter_h7", "preemph_300", "preemph_1k", "preemph_bw",
    "splatter_bw", "splatter_2bw", "splatter_raster", "integrator_1k",
    "rx_bpf_300", "deemph_1k", "rx_bpf_bw", "sq_threshold",
)
LAB_ITEMS = ("power_rails", "rail_sequencing", "uvlo", "mic_agc", "splatter_spectrum", "speaker_audio", "squelch")
#: the design's run-time budget for this deck (kr447 design section 3.3): about 1-3 s and about 15 MB; asserted with room for a loaded machine
ENGINE_BUDGET_S = 9.0
RAWFILE_BUDGET_BYTES = 20_000_000


@pytest.fixture()
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy RF registry holds this template (it is registered with the family by the main session)."""
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=RF_REGISTRY_MODULE: [KR447_AUDIO_PTT])


def _ir(answers: dict[str, str], workdir: Path | None = None) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="kr", name="kr447_audio", workdir=str(workdir) if workdir is not None else None))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def _plan(answers: dict[str, str], library: KicadLibrary | None = LIB, *, confirmed: bool = False) -> Plan | None:
    return templates_mod.design_from_requirements(_ir(answers), library, confirmed=confirmed)  # type: ignore[arg-type]


def _confirmed(answers: dict[str, str], workdir: Path | None = None) -> CircuitIR:
    """The IR after a confirmation: the template's changes applied through ``apply_proposals``."""
    ir = _ir(answers, workdir)
    plan = templates_mod.design_from_requirements(ir, LIB, confirmed=True)
    assert plan is not None and plan.buildable, (plan.notes, [q.question for q in plan.questions]) if plan else None
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload) for c in plan.changes])
    return ir


def _placed(ir: CircuitIR, workdir: Path) -> list[str]:
    """Place the board with the PCB agent (routing skipped) and apply its proposal; returns the agent's notes."""
    res = PCBAgent().run(ir, AgentContext(workdir=workdir, tools={"kicad_library": LIB}, answers={ROUTING_KEY: "skip"}))
    Orchestrator.apply_proposals(ir, res.proposals)
    return res.notes


# --------------------------------------------------------------------------- selection and refusals (no library read)


def test_radio_build_selects_the_board_and_a_confirmed_quantity_of_another_board_refuses_it(registered: None):
    assert KR447_AUDIO_PTT.id == TEMPLATE_ID and KR447_AUDIO_PTT.layer_policy.allowed == (2, 4) and KR447_AUDIO_PTT.layer_policy.default == 4
    for key, value, serving in (("carrier_frequency", "447.5625 MHz", "rx_frontend, tx_exciter, transceiver or transceiver_conducted"),
                                ("tx_power", "0.5 W", "tx_exciter, transceiver or transceiver_conducted")):
        plan = _plan({**AUDIO, key: value}, None)
        assert plan is not None and plan.template == TEMPLATE_ID and not plan.buildable and plan.changes == []
        (q,) = plan.questions
        assert q.key == key and not q.required
        assert f"{key} is not served by radio_build=audio_ptt; {key} is served by radio_build={serving}" in q.question, q.question
    # AM: the licence-exempt class is FM [UNVERIFIED]; the board builds only an FM audio chain
    plan = _plan({**AUDIO, "modulation": "AM"}, None)
    assert plan is not None and not plan.buildable and [q.key for q in plan.questions] == ["modulation"]
    assert "FM (F3E) [UNVERIFIED" in plan.questions[0].question and "builds only an FM audio chain" in plan.questions[0].question
    # without radio_build nothing selects it: a confirmed modulation is the family's required question
    plan = _plan({"input_voltage": "7.4 V", "modulation": "FM"}, None)
    assert plan is not None and plan.template == "radio_build" and [(q.key, q.required) for q in plan.questions] == [("radio_build", True)]


@pytest.mark.parametrize(("extra", "expected"), [
    ({"input_voltage": "12 V"}, "input_voltage 12 V (req.input_voltage) is outside 6.6..8.4 V"),
    ({"input_voltage": "5 V"}, "input_voltage 5 V (req.input_voltage) is outside 6.6..8.4 V"),
    ({"frequency_deviation": "10 kHz"}, "frequency_deviation 10000 Hz (req.frequency_deviation) is outside 500..5000 Hz"),
    ({"audio_bandwidth": "1 kHz"}, "audio_bandwidth 1000 Hz (req.audio_bandwidth) is outside 2000..4000 Hz"),
    ({"tx_timeout": "5 s"}, "tx_timeout 5 s (req.tx_timeout) is outside 10..600 s"),
])
def test_out_of_range_inputs_refuse_with_the_reason(registered: None, extra: dict[str, str], expected: str):
    plan = _plan({**AUDIO, **extra}, None)
    assert plan is not None and plan.template == TEMPLATE_ID and not plan.buildable and plan.changes == []
    assert any(expected in n for n in plan.notes), plan.notes


def test_a_missing_pack_voltage_or_modulation_is_asked(registered: None):
    plan = _plan({"modulation": "FM", "radio_build": "audio_ptt"}, None)
    assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("input_voltage", True)]
    assert 'input_voltage="7.4 V"' in plan.questions[0].question
    plan = _plan({"input_voltage": "7.4 V", "radio_build": "audio_ptt"}, None)
    assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("modulation", True)]


# --------------------------------------------------------------------------- the table and the confirmed design


@needs_libs
def test_the_layer_policy_builds_two_or_four_layers_and_refuses_six(registered: None):
    plan = _plan({**AUDIO, "pcb_layers": "6"})
    assert plan is not None and not plan.buildable and any("6 layers is not one of the stackups this version builds (2, 4)" in n for n in plan.notes)
    two = _confirmed({**AUDIO, "pcb_layers": "2"})
    assert two.pcb.stackup.layer_count == 2 and len(two.components) == 144


@needs_libs
def test_the_table_shows_every_kr447_and_model_row_as_unverified(registered: None):
    plan = _plan(AUDIO)
    assert plan is not None and plan.buildable
    table = plan.table()
    lines = table.splitlines()
    kr = {ln.split(" = ", 1)[0].strip(): ln for ln in lines if ln.startswith("  kr447.")}
    assert set(kr) == set(profile_keys()) and len(kr) == len(profile_keys())
    for key, line in kr.items():
        assert "UNVERIFIED" in line and "a placeholder, not a grounded limit" in line, line
    model = [ln for ln in lines if ln.startswith(f"  {MODEL_PREFIX}")]
    assert len(model) >= 14 and all("UNVERIFIED" in ln for ln in model), [ln for ln in model if "UNVERIFIED" not in ln]
    keys = {ln.strip().split(" ", 1)[0].rstrip(":") for ln in model}
    assert {"model.k_pm", "model.opamp", "model.pmos", "model.npn", "model.diode", "model.ldo_rx.r_in", "model.ldo_tx.r_in", "model.fuse.r"} <= keys
    assert "req.input_voltage: input_voltage = 7.4 V" in table
    assert "pcb_layers = 4 - board layer count: this template's default 4 layers" in table and "no RF on this board: 2 or 4 layers" in table
    assert f"every PASS here is a {MODEL_VERDICT}: no IC is simulated, no RF is on this board" in table
    assert "reported, kept: " in table and "RV401" in table  # the volume wiper: a node only the unmodelled LM386 would load
    # the splatter filter is 4th order (decision 4B) and the integrator's tau is a calculator output
    assert "splat.order = 4" in table and "tx.tau_i = " in table and "calc.rf.fm.pm_integrator_tau" in table


@needs_libs
def test_a_confirmation_builds_the_four_blocks_with_traced_values_and_the_rf_design(registered: None):
    ir = _confirmed(FULL)
    refs = [c.ref for c in ir.components]
    assert len(refs) == 154 and len(set(refs)) == 154  # 144 + the 4060 time-out's ten parts
    assert _confirmed(AUDIO).components.__len__() == 144
    by_block = {b.id: b for b in ir.rf.blocks}
    assert list(by_block) == ["power", "ptt", "tx_audio", "rx_audio", "bench"]
    for bid, block in by_block.items():
        hundred = {"power": 1, "ptt": 2, "tx_audio": 3, "rx_audio": 4, "bench": 4}[bid]
        assert all(int("".join(ch for ch in r if ch.isdigit())) // 100 == hundred for r in block.refs), (bid, block.refs)
        r = block.region
        assert (r.x.value, r.y.value, r.w.value, r.h.value) == REGIONS[bid]
    assert {"U203", "R220", "C206", "Q204"} <= set(by_block["ptt"].refs) and by_block["bench"].refs == ["J402", "J403", "J404"]
    assert sorted(li.id for li in ir.rf.lab_items) == sorted((*LAB_ITEMS, "tot"))
    assert sorted(r.rail for r in ir.rf.rails) == ["RX_3V3", "RX_5V", "TX_3V3", "TX_5V"]
    assert set(ir.rf.profile_keys) == set(profile_keys()) and "model.k_pm" in ir.rf.model_values
    # every derived value is a registered calculator's output over parameters the IR holds
    rec = recompute_parameters(ir)
    assert rec.status is S.PASS, rec.message
    # the parts that serve the stated requirements name them (review.requirements_vs_ir traces these)
    serves = {c.ref: set(c.serves_requirements) for c in ir.components}
    assert serves["J101"] == {"req.input_voltage"}
    assert {"J402", "J403", "J404"} == {r for r, s in serves.items() if "req.radio_build" in s}
    assert {"U304", "U401"} == {r for r, s in serves.items() if "req.modulation" in s}
    assert {"R316", "RV301"} == {r for r, s in serves.items() if "req.frequency_deviation" in s}
    assert {"C313", "C314", "C316", "C317", "R405"} == {r for r, s in serves.items() if "req.audio_bandwidth" in s}
    assert {"R220", "C206"} == {r for r, s in serves.items() if "req.tx_timeout" in s}
    # the time-out is the calculator's period at the E24 resistor, at most the stated 180 s
    period = ir.parameters["ptt.tot_period"]
    assert period.provenance.tool == "calc.rf.tot.period" and 0.8 * 180.0 <= period.value <= 180.0
    # the 4-layer stack (the policy's default) and the POWER_PA class sized by IPC-2221 for 0.6 A
    assert ir.pcb.stackup.layer_count == 4
    power = next(c for c in ir.si.net_classes if c.name == "POWER_PA")
    assert set(power.nets) == {"VBAT", "VBAT_F", "V_SYS", "V_TX", "TX_5V", "PA_5V"} and power.min_width_mm.provenance.tool == "calc.ipc2221.width_for_current"
    # the comparators, gate, counter, 3.3 V regulators, microphone amplifier and speaker amplifier are never simulated; the two
    # pack-fed regulators are simulated only as their input load (model.ldo_*.r_in), their output rails as ideal sources
    excluded = {c.ref for c in ir.components if c.spice is not None and c.spice.exclude}
    assert {"U103", "U104", "U201", "U202", "U203", "U204", "U301", "U402", "U403", "SW101", "SW201"} <= excluded
    for ref, key in (("U101", "model.ldo_rx.r_in"), ("U102", "model.ldo_tx.r_in")):
        binding = next(c for c in ir.components if c.ref == ref).spice
        assert not binding.exclude and binding.device.value == "R" and binding.value.value == ir.parameters[key].value
    assert {c.id for c in ir.constraints} >= {"c.kr447.no_transmission", "c.kr447.logic_supply", "c.kr447.pack", "c.kr447.pa_pd_polarity"}


def test_the_block_builders_the_transceiver_composes_are_the_boards_blocks():
    blocks = stage1_blocks(tot=False)
    assert [(type(b), p.ref_base) for b, p in blocks] == [(PowerBlock, 100), (PttBlock, 200), (TxAudioBlock, 300), (RxAudioBlock, 400)]
    assert isinstance(blocks[0][1], BlockPrefix) and stage1_blocks(tot=True)[1][0].tot


@needs_libs
@pytest.mark.parametrize("answers", [AUDIO, FULL], ids=["default", "with_timeout"])
def test_two_builds_are_one_design_and_compile_byte_identical(registered: None, tmp_path: Path, answers: dict[str, str]):
    one, two = _confirmed(answers, tmp_path / "a"), _confirmed(answers, tmp_path / "b")
    assert one.design_dict() == two.design_dict() and one.content_hash() == two.content_hash()
    notes = _placed(one, tmp_path / "a")
    _placed(two, tmp_path / "b")
    assert "placement.rf_floorplan 0.1" in " ".join(notes) and "74.0 x 82.0 mm generated outline" in " ".join(notes)
    assert [(p.component_ref, p.x_mm, p.y_mm, p.rotation_deg) for p in one.pcb.placements] == [(p.component_ref, p.x_mm, p.y_mm, p.rotation_deg) for p in two.pcb.placements]
    assert len(one.pcb.placements) == len(one.components) and one.pcb.tracks == []
    for compiler in (SchematicCompiler(), PCBCompiler(), SpiceNetlistCompiler()):
        a = compiler.compile(one, CompileContext(workdir=tmp_path / "a", tools={"kicad_library": LIB}))
        b = compiler.compile(two, CompileContext(workdir=tmp_path / "b", tools={"kicad_library": LIB}))
        assert Path(a.path).read_bytes() == Path(b.path).read_bytes(), type(compiler).__name__
    board = Path(PCBCompiler().compile(one, CompileContext(workdir=tmp_path / "a", tools={"kicad_library": LIB})).path).read_text(encoding="utf-8")
    assert board.count("(footprint ") == len(one.components)


@needs_libs
def test_the_router_cannot_route_this_board_so_the_deliverable_is_placement_only(registered: None, tmp_path: Path):
    """A known limit, named: the microphone footprint's pad 1 is a custom-shaped pad the maze router does not model, so it refuses
    the whole board (placement only, never half the nets) - the real routing is left to KiCad."""
    ir = _confirmed(AUDIO, tmp_path)
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    notes = " | ".join(res.notes)
    assert "not routed: cannot route: pad MK301.1 of footprint Sensor_Audio:CUI_CMC-4013-SMT has shape 'custom'" in notes
    assert res.proposals and res.proposals[0].payload.tracks == [] and len(res.proposals[0].payload.placements) == 144


@needs_libs
def test_theory_figures_and_part_notes_render_in_korean_with_the_designs_numbers(registered: None):
    ir = _confirmed(FULL)
    sections = KR447_AUDIO_PTT.theory(ir)
    assert [s.title for s in sections][:2] == ["개요: 1단계 오디오 / PTT 시험 기판", "전원과 PTT 레일 전환"] and len(sections) == 8
    text = "\n".join(s.body for s in sections)
    assert "기록 없음" not in text and "검증되지 않은 자리표시값" in text and "KC 적합성평가" in text
    assert "송신 시간 제한(TOT)" in text and "calc.rf.tot.period" in text  # the time-out paragraph appears only with tx_timeout
    assert f"τ = {quantity(ir.parameters['ptt.tau_dly'].value, 's')} (`calc.rc.tau`)" in text
    assert f"τ_i = N · K_pm · V_lim / (2π · Δf_design) = {quantity(ir.parameters['tx.tau_i'].value, 's')}" in text
    assert "`rf.deviation` 은 NOT_VERIFIED" in text
    figures = KR447_AUDIO_PTT.theory_figures(ir)
    assert [f.id for f in figures] == ["theory_tx_audio", "theory_rx_audio", "theory_settle_delay"]
    notes = KR447_AUDIO_PTT.part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components}
    assert "LP38693DT-5.0" in notes["U101"].why and all(UNVERIFIED_SUBSTITUTE in s for n in notes.values() for s in n.substitutes)
    assert "MCP6001-OT" in notes["U303"].why and "Q 0.5412" in notes["U303"].role
    # the theory and parts stage reports render from the IR alone, byte for byte the same twice
    from ai_eda.report.stages import build_stage_document

    for stage in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION):
        first, again = build_stage_document(stage, ir, LIB, None).markdown, build_stage_document(stage, ir, LIB, None).markdown
        assert first == again and len(first) > 1000, stage
    theory = build_stage_document(Stage.ARCHITECTURE, ir, LIB, None).markdown
    assert "송신 대기(settle) 지연과 인터록" in theory and "![fig](fig:theory_settle_delay)" in theory


@needs_libs
def test_the_lm1117_output_capacitor_is_an_electrolytic_in_its_esr_window(registered: None):
    """U102 (LM1117DT-5.0) needs an output capacitor whose ESR lies in a bounded window: C105 is a Device:C_Polarized electrolytic (+ on
    TX_5V), never the 10 uF ceramic the other regulators take, and the choice and the power_rails lab item say why."""
    ir = _confirmed(AUDIO)
    c105 = ir.component("C105")
    assert c105.symbol.library == "Device" and c105.symbol.name == "C_Polarized" and c105.footprint.name == "CP_Elec_5x5.4" and c105.value == "22u"
    tx5 = {(p.component_ref, p.pin_number) for p in ir.net("TX_5V").pins}
    gnd = {(p.component_ref, p.pin_number) for p in ir.net("GND").pins}
    assert ("C105", "1") in tx5 and ("C105", "2") in gnd and ("U102", "2") in tx5
    assert not any(ir.component(ref).symbol.name == "C_Polarized" for ref, _ in tx5 if ref != "C105")  # the ceramics elsewhere stay ceramics
    note = ir.parameters["power.c_tx5v_out"].provenance.note
    assert "LM1117" in note and "ESR" in note and "[UNVERIFIED: TI LM1117 datasheet" in note
    assert "TX_5V" not in ir.parameters["power.c_bulk"].provenance.note.split("output capacitor on")[0].split("capacitor on")[-1]
    lab = next(x for x in ir.rf.lab_items if x.id == "power_rails")
    assert "TX_5V's stability" in lab.what and "oscilloscope" in lab.instruments
    assert "cp_polarity" in ir.parameters and ir.parameters["cp_polarity"].value == "pin 1 = +"


@needs_libs
def test_the_banner_lists_the_deck_rows_that_rest_on_the_profile_placeholders(registered: None):
    """The profile banner never says "no check PASSes with these values": the deck rows whose frequency or target was derived from a
    ``kr447.*`` placeholder are listed from provenance (the raster analysis; the deviation limit when frequency_deviation is not stated)."""
    from ai_eda.ir.provenance import design_data
    from ai_eda.report.rf_report import profile_banner, profile_dependents

    ir = _confirmed(AUDIO)  # frequency_deviation not stated: the TX audio chain is designed for the profile's kr447.max_deviation
    assert design_data(ir.parameters["tx.frequency_deviation"]) == design_data(ir.parameters["kr447.max_deviation"])
    dependents = profile_dependents(ir)
    assert {"spice.splatter_raster", "spice.pm_drive_peak", "spice.integrator_1k"} <= set(dependents)
    assert "spice.rx_bpf_300" not in dependents and "spice.main_switch_on" not in dependents
    banner = "\n".join(profile_banner(ir))
    assert "PASS 가 되는 검사는 없습니다" not in banner and "`rf.regulatory_profile` 은 이 값으로 PASS 가 되지 않습니다" in banner
    assert "적합성 판정이 아니며" in banner and "`spice.pm_drive_peak`" in banner and "`spice.splatter_raster`" in banner
    # a stated deviation is the user's requirement: only the raster row still rests on the profile's number
    stated = profile_dependents(_confirmed(FULL))
    assert "spice.splatter_raster" in stated and "spice.pm_drive_peak" not in stated and "spice.integrator_1k" not in stated


# --------------------------------------------------------------------------- ngspice: the design deck and the whole pipeline


def _deck_numbers(ir: CircuitIR) -> tuple[float, int]:
    """(engine seconds summed over the analyses, rawfile bytes) of the latest SPICE run."""
    summary = ir.validation.latest("spice")
    analyses = summary.details["analyses"]
    return sum(a["elapsed_s"] for a in analyses.values()), sum(Path(a["raw_output_path"]).stat().st_size for a in analyses.values())


#: where the user-facing parts land on both builds (kr447 wave-2 review finding 15): (left, right, top, bottom) edge distances in mm, as the
#: template's docstring and its Korean overview state them - a deviation from the design's §2.1 (connectors on the bottom edge, the PTT
#: switch and the pots on the top edge) left to manual placement
UI_EDGES_MM = {
    "RV401": (12.2, 47.4, 2.0, 66.7), "MK301": (42.5, 27.0, 2.0, 75.5), "RV402": (23.9, 35.7, 16.3, 52.4), "SW201": (42.5, 23.0, 36.5, 39.5),
    "J101": (2.0, 65.1, 42.5, 34.0), "J401": (10.9, 56.2, 16.3, 60.2),
}


@needs_libs
@pytest.mark.parametrize("answers", [AUDIO, FULL], ids=["default", "with_timeout"])
def test_the_user_facing_parts_land_where_the_template_says(registered: None, answers: dict[str, str]):
    """The floorplan does not put the PTT switch and the pots on the top edge nor the connectors on the bottom one; the template says where."""
    import ai_eda.design.rf.t_audio_ptt as module
    from ai_eda.tools.placement.rf_floorplan import rf_floorplan_placement

    ir = _confirmed({**BASE, **answers})
    fp = rf_floorplan_placement(ir, LIB)
    w, h = fp.outline.width_mm, fp.outline.height_mm
    assert (w, h) == (74.0, 82.0)
    for ref, want in UI_EDGES_MM.items():
        b = fp.extents[ref]
        assert (b.x1, w - b.x2, b.y1, h - b.y2) == pytest.approx(want, abs=0.05), ref
    doc = " ".join((module.__doc__ or "").split())
    for sentence in ("``RV402`` (squelch) is 16.3 mm below the top edge", "``SW201`` (PTT) is in the board's interior, 36.5 mm below the top edge",
                     "``J101`` (pack) is on the left edge, 34 mm above the bottom edge", "``J401`` (speaker) is in the interior, 60.2 mm above the bottom edge"):
        assert sentence in doc, sentence
    overview = KR447_AUDIO_PTT.theory(ir)[0].body
    assert "SW201(PTT)은 기판 안쪽" in overview and "J101(팩)은 왼쪽 가장자리(아래쪽에서 34 mm 위)" in overview


@needs_libs
@needs_ngspice
def test_the_pipeline_runs_to_release_with_every_deck_expectation_passing_and_nothing_claimed_beyond(registered: None, tmp_path: Path):
    ir = _ir({}, tmp_path)  # the answers are typed on the first run (they become requirements there)
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": LIB}, answers={**BASE, **AUDIO})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": runner}, answers={CONFIRM_DESIGN_KEY: "yes", ROUTING_KEY: "skip"})).run(ir)
    assert not state.blocked and state.outcomes[-1].stage is Stage.RELEASE
    latest = ir.validation.latest_by_check()
    # routing skipped: the one FAIL is the measured open of the placed board's nets (the microphone's custom-pad net is not judged), as on rx_backend
    assert [c for c, r in latest.items() if r.status is S.FAIL] == ["pcb.routing.connectivity"]
    conn = latest["pcb.routing.connectivity"]
    assert any("MK301.1" in u for u in conn.details["unknown"])
    rows = {r["net"]: r for r in conn.details["nets"]}
    assert rows["VBAT"]["status"] == "FAIL" and all(r["status"] == "NOT_VERIFIED" for r in rows.values() if "MK301.1" in r.get("unknown_pads", []))
    # the design deck: every expectation PASSes on ngspice (principle verdicts; no IC is simulated), within the run-time budget
    assert state.outcome(Stage.SPICE).status is S.PASS and latest["spice"].tool == "ngspice-shared"
    for eid in EXPECTATIONS:
        assert latest[f"spice.{eid}"].status is S.PASS, (eid, latest[f"spice.{eid}"].message)
    assert sorted(k for k in latest if k.startswith("spice.")) == sorted(f"spice.{e}" for e in EXPECTATIONS)
    seconds, raw = _deck_numbers(ir)
    assert seconds < ENGINE_BUDGET_S and raw < RAWFILE_BUDGET_BYTES, (seconds, raw)
    commands = {a["command"] for a in latest["spice"].details["analyses"].values()}
    assert {"tran 5e-5 120m", "ac lin 1 3k 3k", "ac lin 1 12.5k 12.5k", "op"} <= commands
    # the checks that can never PASS here
    assert "model.k_pm" in latest["rf.deviation"].message and latest["rf.deviation"].status is S.NOT_VERIFIED
    for cid in ("rf.model_grounding", "rf.regulatory_profile", *(f"rf.lab.{i}" for i in LAB_ITEMS),
                *(f"power.rail_budget.{r}" for r in ("RX_5V", "TX_5V", "RX_3V3", "TX_3V3")), *(f"power.headroom.U10{i}" for i in range(1, 5))):
        assert latest[cid].status is S.NOT_VERIFIED, (cid, latest[cid].message)
    # IR arithmetic and the design's own consistency
    for rail in ("RX_5V", "TX_5V", "RX_3V3", "TX_3V3"):
        assert latest[f"block.interface.{rail}"].status is S.PASS
    assert latest["calc.recompute"].status is S.PASS and latest["review.requirements_vs_ir"].status is S.PASS
    assert state.outcome(Stage.SCHEMATIC).status is S.PASS and state.outcome(Stage.PCB).status is S.PASS
    assert state.outcome(Stage.ERC).status is S.NOT_VERIFIED and state.outcome(Stage.DRC).status is S.NOT_VERIFIED
    assert "placement.rf_floorplan 0.1: 144 component(s) on a 74.0 x 82.0 mm" in state.outcome(Stage.PLACEMENT).message
    assert state.outcome(Stage.RELEASE).status is S.FAIL  # the placement-only board's open nets: not releasable, and the record says why


@needs_libs
@needs_ngspice
def test_the_time_out_variant_keeps_the_deck_passing_and_the_ptt_timeline_is_the_confirmed_plan(registered: None, tmp_path: Path):
    ir = _ir({}, tmp_path)
    Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": LIB}, answers={**BASE, **FULL})).run(ir, stop_after=Stage.ARCHITECTURE)
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": LIB, "spice": runner},
                                      answers={CONFIRM_DESIGN_KEY: "yes", PLACEMENT_KEY: "skip"})).run(ir, stop_after=Stage.SPICE)
    assert state.outcome(Stage.CALCULATION).status is S.PASS and state.outcome(Stage.SPICE).status is S.PASS
    for eid in EXPECTATIONS:
        assert ir.validation.latest(f"spice.{eid}").status is S.PASS, eid
    excluded = {e["ref"] for e in ir.validation.latest("spice").details["excluded"]}
    assert {"U203", "Q204"} <= excluded  # the 4060 and its clamp are never simulated: the time-out is a lab item
    # the PTT checks read the transient at the confirmed plan's times; the deviation's peak is the integrator input's bound
    held, enabled = ir.validation.latest("spice.pa_held_off_while_settling"), ir.validation.latest("spice.pa_enabled_after_settle")
    assert held.details["at"] == pytest.approx(PTT_PLAN_S["check_on"]) and enabled.details["at"] == pytest.approx(PTT_PLAN_S["check_enable"])
    assert math.isclose(ir.parameters["tx.frequency_deviation"].value, 2500.0)
    seconds, raw = _deck_numbers(ir)
    assert seconds < ENGINE_BUDGET_S and raw < RAWFILE_BUDGET_BYTES, (seconds, raw)
