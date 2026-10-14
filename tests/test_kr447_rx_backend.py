"""The ``kr447_rx_backend`` template and its IF back-end block (kr447 design §2.2, wave 2 part P10).

What runs where:

* always: the template's contract against the family table (needs, serves,
  the 4-layer policy), the selection by ``radio_build`` alone, the closed-world
  refusals with the family's sentence naming the serving builds, the missing
  inputs, a non-FM modulation, a board built with no companions (refused),
  the companions' input ranges (``input_voltage``, ``audio_bandwidth``), a
  stated 2-layer board (refused before anything is asked), the block's
  profile reads;
* with the packed KiCad 10.0.6 libraries (``needs_libs``, ``KICAD10_SYMBOL_DIR``):
  the confirmation table (every ``kr447.*`` and ``model.*`` row says
  UNVERIFIED), the design numbers (7-crystal ladder at f_s 21.388644 MHz,
  centre within a few hertz of IF1), byte-identical rebuilds (design hash, the
  compiled schematic and board), the optional inputs (channel spacing, system
  impedance, frequency tolerance, deviation), the block re-based without its
  input connector (the transceiver's use), ``calc.recompute`` PASS, the RF
  floorplan and both compilers, the Korean theory / part notes / figures,
  and the default board with part P9's RX power and audio blocks (100 parts,
  92 x 64 mm, the companions' design deck, their parts serving
  ``input_voltage`` / ``audio_bandwidth``, the stage-1 part notes);
* with the libraries and ngspice (``needs_ngspice``): every ``spice.rf.*`` row
  of the five fixture networks PASS on the default models inside the kr447
  design's §3.3 budget (rx_backend fixtures < 3 s; measured about 0.9 s on
  ngspice-42), and the whole pipeline on the bench-interface board with
  routing: the RF checks, SI and the one honest FAIL of the stand-in
  (``req.input_voltage`` is served by the RX power block, not by the bench
  headers), and the default board selected through the real registry with
  routing skipped (the companions' five design-deck rows PASS,
  ``review.requirements_vs_ir`` PASS).

Most tests compose the board with :class:`BenchInterfaceBlock` in place of
part P9's RX power and audio blocks (60 parts); :func:`default_companions`
composes the P9 blocks themselves.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import ai_eda.design.templates as templates_mod
from ai_eda.agents.base import AgentContext, IRProposal
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.design.base import CHOICE_NOTE_PREFIX, Plan
from ai_eda.design.board import add_board
from ai_eda.design.inputs import read_inputs
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import family, profile
from ai_eda.design.rf.blocks import BlockContext, BlockPrefix
from ai_eda.design.rf.blocks.if_backend import INTERFACE_NETS, NETWORKS, PROFILE_READS, XTAL_CM, IfBackendBlock
from ai_eda.design.rf.t_rx_backend import (
    BUILD,
    PLANE_REASON,
    REGIONS,
    BenchInterfaceBlock,
    KR447RxBackendTemplate,
    default_companions,
    if_backend_hiz_nets,
)
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, Traced, user_requirement
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc.recompute import recompute_parameters
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.rf_floorplan import rf_floorplan_placement
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.workflow import Stage
from ai_eda.workflow.orchestrator import Orchestrator

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_AM_FM", "Device", "Connector", "Connector_Generic")) and \
    _REAL.footprint_file("Crystal", "Crystal_SMD_HC49-SD") is not None and _REAL.footprint_file("Capacitor_SMD", "C_Trimmer_Murata_TZB4-A") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")

BASE = {"radio_build": "rx_backend", "modulation": "fm", "input_voltage": "7.4 V"}
TEMPLATE = KR447RxBackendTemplate(companions=(BenchInterfaceBlock(),))


def _ir(tmp_path: Path, answers: dict[str, str], name: str = "rxbe") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def _plan(ir: CircuitIR, *, confirmed: bool = True, template: KR447RxBackendTemplate = TEMPLATE, library: KicadLibrary = _REAL) -> Plan:
    inputs, unusable = read_inputs(ir)
    plan = template.build(ir, inputs, unusable, library, confirmed=confirmed)
    if plan.buildable:
        assert add_board(template, ir, plan, confirmed=confirmed) is None
    return plan


def _applied(tmp_path: Path, answers: dict[str, str] | None = None, name: str = "rxbe") -> CircuitIR:
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


def _shared(confirmed: bool = True) -> dict[str, Traced]:
    return {c.key: t for c, t in profile.profile_choices("t_rx", confirmed)}


# --------------------------------------------------------------------------- the contract (no library needed)


def test_template_contract_is_the_family_table() -> None:
    t = KR447RxBackendTemplate()
    assert t.id == "kr447_rx_backend" == family.BUILDS["rx_backend"].template_id
    assert t.needs == ("modulation", "input_voltage") == BUILD.needs
    assert t.serves == BUILD.serves and "radio_build" in t.serves and "carrier_frequency" not in t.serves
    assert t.layer_policy.allowed == (4,) and t.layer_policy.default == 4 and t.layer_policy.reason == PLANE_REASON
    assert t.plane_nets == ("GND", None)
    power, audio = default_companions()  # part P9's blocks: the RX power section (no TX: nothing is sequenced) and the RX audio block
    assert (power.id, power.modes, audio.id) == ("power", ("rx",), "rx_audio") and KR447RxBackendTemplate().companions()[1].id == "rx_audio"
    assert set(REGIONS) >= {"if_backend", "bench_io", "power", "rx_audio"}


def test_selected_by_radio_build_alone(tmp_path: Path) -> None:
    t = KR447RxBackendTemplate()
    assert t.triggered_by(_ir(tmp_path, {"radio_build": "rx_backend"}), {})
    assert t.triggered_by(_ir(tmp_path, {"radio_build": "RX-Backend"}), {})
    for other in ({"radio_build": "rx_frontend"}, {"modulation": "fm", "input_voltage": "7.4 V"}, {}):
        assert not t.triggered_by(_ir(tmp_path, other), {})


def test_closed_world_refusals_name_the_serving_builds(tmp_path: Path) -> None:
    ir = _ir(tmp_path, {**BASE, "carrier_frequency": "447.5625 MHz", "tx_power": "0.5 W", "channel_spacing": "12.5 kHz"})
    inputs, unusable = read_inputs(ir)
    questions = {q.key: q for q in TEMPLATE.refusals(ir, inputs, unusable)}
    assert set(questions) == {"carrier_frequency", "tx_power"}  # channel_spacing is served
    assert "carrier_frequency is served by radio_build=rx_frontend, tx_exciter, transceiver or transceiver_conducted" in questions["carrier_frequency"].rationale
    assert "tx_power is not served by radio_build=rx_backend; tx_power is served by radio_build=tx_exciter, transceiver or transceiver_conducted" in questions["tx_power"].rationale
    assert all(not q.required for q in questions.values())


def test_the_selection_refuses_through_the_closed_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [TEMPLATE])
    plan = templates_mod.design_from_requirements(_ir(tmp_path, {**BASE, "carrier_frequency": "447.5625 MHz"}), KicadLibrary(roots=[]))
    assert plan is not None and not plan.buildable and plan.template == "kr447_rx_backend"
    assert "carrier_frequency is served by radio_build=rx_frontend" in plan.notes[0]
    # a stated 2-layer board refuses before anything is asked (the policy's reason)
    plan = templates_mod.design_from_requirements(_ir(tmp_path, {"radio_build": "rx_backend", "pcb_layers": "2"}), KicadLibrary(roots=[]))
    assert plan is not None and not plan.buildable and PLANE_REASON in plan.notes[0] and [q.key for q in plan.questions] == ["pcb_layers"]


def test_missing_inputs_are_required_questions(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, {"radio_build": "rx_backend"}), library=KicadLibrary(roots=[]))
    assert not plan.buildable and [q.key for q in plan.questions] == ["modulation", "input_voltage"] and all(q.required for q in plan.questions)
    assert "modulation=fm" in plan.questions[0].question and "input_voltage" in plan.notes[0]


def test_a_non_fm_modulation_refuses(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, {**BASE, "modulation": "am"}), library=KicadLibrary(roots=[]))
    assert not plan.buildable and "FM" in plan.notes[0] and "UNVERIFIED" in plan.notes[0]


def test_without_its_companions_the_board_is_refused(tmp_path: Path) -> None:
    plan = _plan(_ir(tmp_path, BASE), template=KR447RxBackendTemplate(companions=()), library=KicadLibrary(roots=[]))
    assert not plan.buildable and "not composed" in plan.notes[0] and "input_voltage and audio_bandwidth unserved" in plan.notes[0]


@pytest.mark.parametrize(("answers", "why"), [
    ({"input_voltage": "5 V"}, "input_voltage 5 V (req.input_voltage) is outside 6.4..8.4 V"),
    ({"input_voltage": "12 V"}, "input_voltage 12 V (req.input_voltage) is outside 6.4..8.4 V"),
    ({"audio_bandwidth": "1 kHz"}, "audio_bandwidth 1000 Hz (req.audio_bandwidth) is outside 2000..4000 Hz"),
])
def test_the_companions_input_ranges_refuse(tmp_path: Path, answers: dict[str, str], why: str) -> None:
    plan = _plan(_ir(tmp_path, {**BASE, **answers}), template=KR447RxBackendTemplate(), library=KicadLibrary(roots=[]))
    assert not plan.buildable and why in plan.notes[0], plan.notes


def test_the_block_reads_the_profile_from_the_composing_template(tmp_path: Path) -> None:
    ctx = BlockContext(ir=_ir(tmp_path, BASE), library=KicadLibrary(roots=[]), template_id="t_rx", confirmed=True, shared={})
    with pytest.raises(TemplateRefusal, match="kr447.channel_raster"):
        IfBackendBlock().build(ctx)
    assert PROFILE_READS == ("kr447.channel_raster", "kr447.max_deviation")


def test_hiz_nets_follow_the_ladder_parameters() -> None:
    t = user_requirement(3.0)
    assert if_backend_hiz_nets({}) == []
    names = if_backend_hiz_nets({"ifb.lad.n": t, "ifb.lad.c_mesh.1": t, "ifb.lad.c_mesh.3": t}, "IFB_")
    assert names[:6] == ["IFB_LAD_IN", "IFB_LAD_N1", "IFB_LAD_N2", "IFB_LAD_OUT", "IFB_LAD_M1", "IFB_LAD_M3"] and "IFB_QUAD" in names


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
    assert {c.key for c in models} == {"model.xtal21.cm", "model.xtal21.rm", "model.xtal21.c0", "model.sa605.port_r", "model.sa605.rf_in_r",
                                       "model.sa605.lim_out_r", "model.l_q.if1", "model.l_q.if2", "model.xtal21"}
    assert all("UNVERIFIED" in c.text() for c in models)
    assert rows["model.xtal21.cm"].value == XTAL_CM.value == 16e-15 and "C0 / 250" in rows["model.xtal21.cm"].description
    for line in ("kr447.channel_raster = 12500 Hz", "ifb.lad.n = 7", "ifb.lad.bw_design = 8750 Hz", "rf.if_bw = 7500 Hz", "floor.if_backend.h = 64 mm",
                 "U501 RF_AM_FM:SA605D / Package_SO:SO-20_12.8x7.5mm_P1.27mm", "Y501 Device:Crystal / Crystal:Crystal_SMD_HC49-SD",
                 "fixture if1_filter: 22 member(s) between IF1 (port), MIX (port)", "IF50: nets IF1; Z0 50 ohm +/- 10%", "IF_HIZ: nets LAD_IN, LAD_N1"):
        assert line in table, line
    # unconfirmed: every choice is an assumption, never the user's
    ir = _ir(tmp_path, BASE)
    change = next(c for c in _plan(ir, confirmed=False).changes if c.target == "parameters.ifb.lad.n")
    assert change.payload.provenance.kind == ProvenanceKind.ASSUMPTION


@needs_libs
def test_the_design_numbers(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    p = {k: float(t.value) for k, t in ir.parameters.items() if isinstance(t.value, (int, float))}
    assert p["ifb.lad.n"] == 7 and p["ifb.lad.bw_design"] == 8750.0
    assert p["ifb.lad.xtal_fs"] == pytest.approx(21388644.05, abs=1.0)  # 2 IF1 - f_c(IF1): the crystals sit about 11.4 kHz below IF1
    assert abs(p["ifb.lad.f0"] - 21.4e6) < 10.0  # the realised centre within a few hertz of IF1
    assert p["ifb.lad.r_end"] == pytest.approx(846.6, rel=1e-3)
    assert p["ifb.lo2.f"] == 20.95e6 and p["ifb.image2"] == 20.5e6
    assert p["ifb.if2.harm_below"] == pytest.approx(21.15e6) and p["ifb.if2.harm_above"] == pytest.approx(21.6e6)
    assert p["ifb.lad.s21.f0"] == pytest.approx(-4.882, abs=0.01) and p["ifb.lad.s21.acs_lo"] - p["ifb.lad.s21.f0"] == pytest.approx(-41.94, abs=0.05)
    assert p["ifb.quad.phase.mid"] == pytest.approx(87.18, abs=0.01) and p["ifb.if2.s21"] == pytest.approx(-3.603, abs=0.01)
    assert sorted(c.ref for c in ir.components if c.ref.startswith("Y5")) == [f"Y50{i}" for i in range(1, 9)]
    assert len(ir.components) == 60 and ir.rf is not None and [n.id for n in ir.rf.networks] == list(NETWORKS)
    assert ir.rf.profile_keys == profile.profile_keys() and "model.xtal21.cm" in ir.rf.model_values
    assert [b.id for b in ir.rf.blocks] == ["if_backend", "bench_io"] and ir.rf.blocks[0].chain[:3] == ["J501", "L501", "C501"]
    assert {pl.id for pl in ir.rf.frequency_plan} == {"ifb_if2_h_below", "ifb_if2_h_above", "ifb_lo2", "ifb_image2"}
    assert all(c.spice is not None and c.spice.exclude for c in ir.components) and ir.simulation is None  # fixture members only; the P9 audio block brings the deck
    u = next(c for c in ir.components if c.ref == "U501")
    assert set(u.serves_requirements) == {"req.radio_build", "req.modulation"}
    assert recompute_parameters(ir).status is S.PASS


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
def test_optional_inputs_and_the_channel_spacing(tmp_path: Path) -> None:
    ir = _applied(tmp_path, {**BASE, "system_impedance": "75 ohm", "frequency_tolerance": "±2.5 ppm", "frequency_deviation": "2 kHz", "channel_spacing": "12.5 kHz"})
    p = ir.parameters
    assert float(p["ifb.z0"].value) == 75.0 and "rf.z0" not in p
    assert p["ifb.match_in.l"].provenance.derived_from == ["rf.if1", "ifb.z0", "ifb.lad.r_end"]
    assert float(p["ifb.lo2.df_max"].value) == pytest.approx(52.375)  # 2.5 ppm of 20.95 MHz
    assert float(p["ifb.quad.f_lo"].value) == 448e3 and float(p["ifb.lad.f_acs_hi"].value) == pytest.approx(float(p["ifb.lad.f0"].value) + 12.5e3)
    assert ir.rf is not None and ir.rf.networks[1].port("IF1").z0_ohm.value == 75.0  # type: ignore[union-attr]
    assert recompute_parameters(ir).status is S.PASS
    plan = _plan(_ir(tmp_path, {**BASE, "channel_spacing": "25 kHz"}))
    assert not plan.buildable and "is not the KR 447 raster kr447.channel_raster 12500 Hz" in plan.notes[0]


@needs_libs
def test_the_block_rebased_without_its_input_connector(tmp_path: Path) -> None:
    """The transceiver's use: the front end drives IF1, internal nets prefixed, interface nets and GND kept."""
    ctx = BlockContext(ir=_ir(tmp_path, BASE), library=_REAL, template_id="t_rx", confirmed=True, shared=_shared())
    res = IfBackendBlock(input_connector=False).build(ctx, BlockPrefix(500, "IFB_"))
    refs = [c.ref for c in res.components]
    assert "J501" not in refs and "U501" in refs and all(r[-3] == "5" for r in refs)
    nets = {n.name for n in res.nets}
    assert set(INTERFACE_NETS) <= nets and "GND" in nets and "IFB_LAD_IN" in nets and "LAD_IN" not in nets
    lad = next(n for n in res.networks if n.id == "if1_ladder")
    assert [p.net for p in lad.ports] == ["IFB_LAD_IN", "IFB_LAD_OUT"] and "Y501" in lad.members
    assert res.chain[0] == "L501" and res.interface_nets == frozenset(INTERFACE_NETS)


@needs_libs
def test_the_floorplan_and_both_compilers(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    _place(ir)
    assert ir.pcb is not None and (ir.pcb.outline.width_mm, ir.pcb.outline.height_mm) == (50.0, 76.0) and len(ir.pcb.placements) == 60
    for compiler in (SchematicCompiler(), PCBCompiler()):
        ref = compiler.compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _REAL}))
        assert Path(ref.path).is_file()


@needs_libs
def test_the_default_board_composes_the_p9_rx_blocks(tmp_path: Path) -> None:
    """The RX power section and the RX audio block of part P9 around the IF back-end: parts, deck, served requirements, floorplan, part notes."""
    ir = _ir(tmp_path, {**BASE, "audio_bandwidth": "3 kHz"})
    plan = _plan(ir, template=KR447RxBackendTemplate())
    assert plan.buildable, plan.notes
    assert any("reported, kept: RV401" in line for line in plan.simulation)  # the volume wiper reaches only the unmodelled LM386
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload, rationale=c.rationale)
                                      for c in plan.changes])
    refs = {c.ref for c in ir.components}
    assert len(refs) == 100 and {"J101", "Q101", "U101", "U103", "U401", "U402", "U403", "RV401", "U501", "J501"} <= refs and not {"J1", "J2", "U102"} & refs
    assert ir.rf is not None and {b.id: len(b.refs) for b in ir.rf.blocks} == {"if_backend": 58, "power": 13, "rx_audio": 29}
    assert ir.simulation is not None and {e.id for e in ir.simulation.expectations} == {"main_switch_on", "rx_bpf_300", "deemph_1k", "rx_bpf_bw", "sq_threshold"}
    served = {rid for c in ir.components for rid in c.serves_requirements}
    assert {"req.input_voltage", "req.audio_bandwidth", "req.modulation", "req.radio_build"} <= served
    nets = {n.name: n for n in ir.nets}
    assert {"RX_5V", "RX_3V3", "V_RX", "DISC_OUT", "RSSI", "MUTE", "PWR_VBAT", "RXA_VOL_IN"} <= set(nets) and "RX_5V" not in {"PWR_RX_5V"}
    assert {p.component_ref for p in nets["DISC_OUT"].pins} >= {"U501", "C403"} and {p.component_ref for p in nets["MUTE"].pins} >= {"U501", "U403"}
    assert "power.rx_5v" in ir.parameters and "model.xtal21.cm" in ir.rf.model_values and recompute_parameters(ir).status is S.PASS
    _place(ir)
    assert ir.pcb is not None and (ir.pcb.outline.width_mm, ir.pcb.outline.height_mm) == (92.0, 64.0) and len(ir.pcb.placements) == 100
    notes = KR447RxBackendTemplate().part_notes(ir)
    assert set(notes) == refs
    assert notes["U101"].role.startswith("U101: RX_5V") and "수신 전용" in notes["Q101"].role and "디커플링" not in notes["C409"].role  # not an SA605 note
    assert notes["U501"].role.startswith("FM IF 시스템")


@needs_libs
def test_theory_part_notes_and_figures_in_korean(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    sections = TEMPLATE.theory(ir)
    text = "\n".join(s.title + "\n" + s.body for s in sections)
    assert len(sections) == 7 and "수정 래더 필터" in text and "직교" in text and "UNVERIFIED" in text
    for number in ("21.389 MHz", "16 fF", "846.56 Ω", "87.179"):
        assert number in text, number
    notes = TEMPLATE.part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components}
    assert "(검증되지 않음" in notes["U501"].substitutes[0] and "직렬공진" in notes["Y501"].why
    figs = TEMPLATE.theory_figures(ir)
    assert [f.id for f in figs] == ["theory_ladder_s21", "theory_quad_phase"] and all("<svg" in f.svg for f in figs)


# --------------------------------------------------------------------------- ngspice


@needs_libs
@needs_ngspice
def test_every_fixture_row_passes_on_the_default_models_within_the_budget(tmp_path: Path) -> None:
    ir = _applied(tmp_path)
    t0 = time.perf_counter()
    results = spice_rf_results(ir, {"spice": runner}, tmp_path)
    elapsed = time.perf_counter() - t0
    got = {r.check_id: r for r in results}
    assert {f"spice.rf.{n}" for n in NETWORKS} <= set(got) and len(results) == 5 + 5 + 8 + 6 + 6 + 4
    assert all(r.status is S.PASS for r in results), [(r.check_id, r.message) for r in results if r.status is not S.PASS]
    assert all("network verdict under the confirmed model values" in r.message for r in results)
    measured = {cid.removeprefix("spice.rf."): r.details["measured"] for cid, r in got.items() if "measured" in r.details}
    pinned = {  # ngspice-42, 2026-09-29
        "if1_filter.s21_f0": -5.565, "if1_filter.pass_lo": -2.710, "if1_filter.pass_hi": -2.779, "if1_filter.acs_lo": -41.96, "if1_filter.acs_hi": -83.46,
        "if1_filter.lo2": -109.30, "if1_filter.image2": -111.40, "if1_ladder.s21_f0": -4.882, "if2_bpf_a.s21_if2": -3.603, "if2_bpf_a.rej_hi": -22.653,
        "quad_tank.phase_if2": 87.179,
    }
    for key, want in pinned.items():
        assert measured[key] == pytest.approx(want, abs=0.01), key
    assert elapsed < 3.0, f"{elapsed:.2f} s: the kr447 design's §3.3 budget for the rx_backend fixtures is < 3 s"


@needs_libs
@needs_ngspice
def test_the_pipeline_on_the_bench_interface_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Present, confirm, route: the RF and SI statuses the kr447 design predicts for rx_backend, and the stand-in's one honest FAIL."""
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [TEMPLATE])
    ir = _ir(tmp_path, {})
    tools = {"kicad_library": _REAL, "spice": runner}
    state = Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers={"application": "bench", "jurisdiction": "KR", **BASE})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"]
    question = state.open_questions[0].question
    assert question.count("[UNVERIFIED:") >= len(profile.PROFILE) + 9 and ir.components == []
    state = Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers={"confirm_design": "yes"})).run(ir, stop_after=None)
    assert not state.blocked
    latest = ir.validation.latest_by_check()
    status = {k: r.status for k, r in latest.items()}
    assert CHOICE_NOTE_PREFIX in ir.parameters["kr447.max_deviation"].provenance.note
    for check in ("calc.recompute", "block.interface.RX_5V", "domain.rf.impedance", "si.impedance.IF50", "si.critical_length", "pcb.keepout",
                  *[f"spice.rf.{n}" for n in NETWORKS]):
        assert status[check] is S.PASS, (check, latest[check].message)
    for check in ("rf.model_grounding", "rf.regulatory_profile", "rf.lab.ifb_xtal_motional", "rf.lab.ifb_sinad", "rf.lab.ifb_image2_rejection", "spice",
                  "pcb.routing.connectivity", "rf.freq_plan"):
        assert status[check] is S.NOT_VERIFIED, (check, latest[check].message)
    # the margin rows PASS; the second-image response row points to its fixture row and to the lab: it never PASSes before a measurement
    plan = {r["id"]: r for r in latest["rf.freq_plan"].details["rows"]}
    assert all(plan[i]["status"] == "PASS" for i in ("ifb_if2_h_below", "ifb_if2_h_above", "ifb_lo2"))
    assert plan["ifb_image2"]["status"] == "NOT_VERIFIED" and latest["spice.rf.if1_filter"].status is S.PASS
    assert "rf_length_fraction is not stated" in latest["si.rf_length"].message and status["si.rf_length"] is S.NOT_VERIFIED
    assert "joined only by a copper pour" in latest["pcb.routing.connectivity"].message  # GND reaches the In1.Cu plane through pad vias
    assert status["review.requirements_vs_ir"] is S.FAIL and latest["review.requirements_vs_ir"].details["unserved"] == ["req.input_voltage"]
    assert ir.pcb is not None and len(ir.pcb.tracks) > 0 and len(ir.pcb.placements) == 60


@needs_libs
@needs_ngspice
def test_the_default_board_through_the_registry(tmp_path: Path) -> None:
    """``RF_TEMPLATES`` as merged selects this template by radio_build alone; with part P9's blocks the deck runs and every requirement is served.

    Routing is skipped here (the routed board: 812 tracks, no FAIL anywhere, a
    74 s confirm run on ngspice-42 - measured once, not asserted), so the one
    FAIL is the skipped board's ``pcb.routing.connectivity``.
    """
    ir = _ir(tmp_path, {})
    tools = {"kicad_library": _REAL, "spice": runner}
    state = Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers={"application": "bench", "jurisdiction": "KR", **BASE})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in state.open_questions] == ["confirm_design"]
    question = state.open_questions[0].question
    rows = [line for line in question.splitlines() if line.lstrip().startswith(("kr447.", "model."))]
    assert len(rows) >= len(profile.PROFILE) + 18 and all("UNVERIFIED" in line for line in rows)
    state = Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers={"confirm_design": "yes", "pcb.routing": "skip"})).run(ir, stop_after=None)
    assert not state.blocked
    latest = ir.validation.latest_by_check()
    status = {k: r.status for k, r in latest.items()}
    assert len(ir.components) == 100 and ir.rf is not None and [b.id for b in ir.rf.blocks] == ["if_backend", "power", "rx_audio"]
    for check in ("spice", "spice.main_switch_on", "spice.rx_bpf_300", "spice.deemph_1k", "spice.rx_bpf_bw", "spice.sq_threshold", "review.requirements_vs_ir",
                  "block.interface.RX_5V", "block.interface.RX_3V3", "calc.recompute", *[f"spice.rf.{n}" for n in NETWORKS]):
        assert status[check] is S.PASS, (check, latest[check].message)
    for check in ("rf.model_grounding", "rf.regulatory_profile", "power.rail_budget.RX_5V", "power.headroom.U101", "rf.lab.power_rails", "rf.lab.squelch",
                  "rf.freq_plan"):
        assert status[check] is S.NOT_VERIFIED, (check, latest[check].message)
    cards = {r["key"]: r for r in latest["rf.model_grounding"].details["rows"] if r.get("card")}
    assert set(cards) >= {"model.xtal21", "model.opamp", "model.pmos"} and cards["model.xtal21"]["bound_by"] == [f"Y50{i}" for i in range(1, 8)]
    from ai_eda.report.rf_report import model_lines

    xtal = next(line for line in "\n".join(model_lines(ir)).splitlines() if line.startswith("| `model.xtal21` |"))
    assert "모델 카드" in xtal and "is a model card" in xtal  # the report prints the recorded card row, not a missing value
    assert [k for k, v in status.items() if v is S.FAIL] == ["pcb.routing.connectivity"]
