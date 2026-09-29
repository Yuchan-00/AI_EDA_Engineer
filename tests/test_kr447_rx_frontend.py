"""The ``kr447_rx_frontend`` template and its blocks (kr447 design §2.3 and §5 part P11: ``rx_frontend``, ``lo_chain``).

What runs where:

* always: the template's family entry (needs / serves / the 4-layer policy),
  its selection by ``radio_build`` alone, the closed-world and plan refusals
  that come before any part is placed (an unserved requirement names the
  builds that serve it, a carrier off the unverified raster, a non-FM
  modulation, 2 layers, no supply companion, an input_voltage outside the
  RX power section's range, a missing carrier), and the blocks' refusals when
  a composition does not share what they read;
* with the packed KiCad 10.0.6 libraries (``needs_libs``,
  ``KICAD10_SYMBOL_DIR``): the confirm_design table (every ``kr447.*`` and
  ``model.*`` row carries UNVERIFIED), what a confirmation applies (105
  parts, the stacked GND pins in one net, ``ir.rf`` with 7 blocks, 7 fixture
  networks, 8 plan rows, 11 lab items; every part on a LO network's port nets
  is a member and no two top-C filters meet on a net), the floorplan placement (both cans
  hold their blocks), byte-identical rebuilds (table, design hash, the
  compiled schematic and PCB), ``calc.recompute`` / ``block.interface.*``
  PASS, the theory / figures / part notes, the optional requirements
  (``frequency_tolerance``, ``rx_sensitivity``, ``system_impedance``), and the
  default board through the real registry (part P9's RX power section: 117
  parts on 71 x 82 mm, ``input_voltage`` served, its deck row);
* with ngspice as well (``needs_dll``): every ``spice.rf.*`` row of the seven
  networks and the five bias expectations PASS on the default models, the
  measured rows equal the calculators' networks, and the fixtures run inside
  the kr447 design §3.3 budget (rx_frontend: < 2 s).

Most tests register the template with the bench-supply stand-in for the RX
power section (part P9) through ``monkeypatch`` (the ``registered``
fixture); the default-board test uses ``RF_TEMPLATES`` as the merge filled
it (:func:`default_companions`: ``PowerBlock(modes=("rx",))``).
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.agents.keys import ROUTING_KEY
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.design import check_inputs_vs_requirements
from ai_eda.design.base import TEMPLATE_VERSION, UNVERIFIED_SUBSTITUTE
from ai_eda.design.inputs import read_inputs
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import registry
from ai_eda.design.rf.blocks.base import BlockContext
from ai_eda.design.rf.blocks.lo_chain import LoBufferBlock, LoChainBlock
from ai_eda.design.rf.blocks.rx_frontend import RxFrontendBlock, RxMixerBlock
from ai_eda.design.rf.family import BUILDS
from ai_eda.design.rf.profile import profile_choices
from ai_eda.design.rf.t_rx_frontend import BenchSupplyBlock, KR447RxFrontendTemplate, default_companions
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, Requirement, RequirementKind, ValidationStatus as S, user_requirement
from ai_eda.tools.calc import radio
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.workflow import Orchestrator, Stage
from tests.rf_fixture_audit import nets_joining_networks, port_net_outsiders

_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_Mixer", "RF_Amplifier", "Oscillator", "Transistor_BJT")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-103_26.21x26.21mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")
runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

BASE = {"application": "bench", "jurisdiction": "EU"}
FRONTEND = {"radio_build": "rx_frontend", "carrier_frequency": "447.5625 MHz", "input_voltage": "7.4 V"}
#: every fixture network of the board and its rows
NETWORK_ROWS = {
    "fe_bpf2": ["s21_fc", "rel_image"],
    "fe_bpf3": ["s21_fc", "rel_image", "rel_lo1"],
    "diplexer": ["s21_if1", "s11_lo1", "s11_sum"],
    "lo_tank1": ["s21", "rel_m", "rel_p"],
    "lo_tank2": ["s21", "rel_m", "rel_p"],
    "lo_bpf": ["s21_lo1", "rel_m1", "rel_p1", "rel_m2", "rel_p2"],
    "lo_pad": ["s21_lo1", "s11_lo1"],
}
BIAS = ["ic_lna", "ic_ifamp", "ic_x3", "ic_x6", "ic_x12"]
#: the kr447 design §3.3 budget of the rx_frontend fixtures + bias ops (s)
FIXTURE_BUDGET_S = 2.0


def _template() -> KR447RxFrontendTemplate:
    return KR447RxFrontendTemplate(companions=(BenchSupplyBlock(),))


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> KR447RxFrontendTemplate:
    """The template with the bench-supply stand-in, registered as the only radio template for the test."""
    tpl = _template()
    monkeypatch.setattr(registry, "RF_TEMPLATES", [tpl])
    return tpl


def _ir(tmp_path: Path, name: str = "rxfe") -> CircuitIR:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))


def _run(ir: CircuitIR, tmp_path: Path, answers: dict[str, str], stop_after: Stage | None = Stage.ARCHITECTURE, *, spice: bool = False):
    tools: dict = {"kicad_library": _REAL}
    if spice:
        tools["spice"] = runner
    return Orchestrator(AgentContext(workdir=tmp_path, tools=tools, answers=answers)).run(ir, stop_after=stop_after)


def _present(ir: CircuitIR, tmp_path: Path, answers: dict[str, str] | None = None) -> str:
    """Run 1: the table is presented and nothing is applied."""
    state = _run(ir, tmp_path, {**BASE, **FRONTEND, **(answers or {})})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY], [(q.key, q.question[:200]) for q in state.open_questions]
    assert ir.components == [] and ir.rf is None
    return state.open_questions[0].question


def _confirm(ir: CircuitIR, tmp_path: Path, stop_after: Stage | None = Stage.ARCHITECTURE, *, spice: bool = False, routing_skip: bool = True):
    answers = {CONFIRM_DESIGN_KEY: "yes", **({ROUTING_KEY: "skip"} if routing_skip else {})}
    return _run(ir, tmp_path, answers, stop_after, spice=spice)


def _refusal(tmp_path: Path, answers: dict[str, str]) -> str:
    """The ARCHITECTURE outcome's message of a run that proposes nothing."""
    ir = _ir(tmp_path)
    state = _run(ir, tmp_path, {**BASE, **answers})
    out = state.outcome(Stage.ARCHITECTURE)
    assert ir.components == [] and ir.rf is None and not any(q.key == CONFIRM_DESIGN_KEY for q in state.open_questions), out.message
    return out.message + " " + " ".join(q.question for q in out.questions)


# --------------------------------------------------------------------------- always


def test_the_template_is_the_family_entry_with_a_four_layer_policy() -> None:
    tpl = _template()
    build = BUILDS["rx_frontend"]
    assert tpl.id == "kr447_rx_frontend" == build.template_id and tpl.title == build.title
    assert tpl.needs == ("carrier_frequency", "input_voltage") and tpl.serves == build.serves
    assert set(build.serves) >= {"carrier_frequency", "modulation", "frequency_tolerance", "system_impedance", "rx_sensitivity", "input_voltage", "radio_build"}
    assert tpl.layer_policy.allowed == (4,) and tpl.layer_policy.default == 4 and "reference plane" in tpl.layer_policy.reason
    (power,) = default_companions()  # part P9's RX power section (receive-only: nothing is sequenced)
    assert (power.id, power.modes) == ("power", ("rx",)) and KR447RxFrontendTemplate().companions()[0].id == "power"


def test_selected_only_by_radio_build(tmp_path: Path) -> None:
    tpl = _template()
    for build, expected in (("rx_frontend", True), ("rx_backend", False), ("RX-Frontend", True)):
        ir = _ir(tmp_path / build)
        ir.requirements.requirements += [
            Requirement(id="req.radio_build", key="radio_build", text="build", kind=RequirementKind.EXPLICIT, value=user_requirement(build), category="rf"),
            Requirement(id="req.carrier_frequency", key="carrier_frequency", text="f", kind=RequirementKind.EXPLICIT, value=user_requirement(447.5625e6, "Hz"), category="rf"),
        ]
        inputs, _ = read_inputs(ir)
        assert tpl.triggered_by(ir, inputs) is expected, build


def test_an_unserved_requirement_refuses_naming_the_builds_that_serve_it(tmp_path: Path, registered) -> None:
    msg = _refusal(tmp_path / "p", {**FRONTEND, "tx_power": "0.5 W"})
    assert "tx_power is served by radio_build=tx_exciter, transceiver or transceiver_conducted" in msg
    msg = _refusal(tmp_path / "d", {**FRONTEND, "frequency_deviation": "2.5 kHz"})
    assert "frequency_deviation is not served by radio_build=rx_frontend" in msg and "audio_ptt, rx_backend, tx_exciter" in msg


def test_a_carrier_off_the_raster_a_non_fm_modulation_and_two_layers_refuse(tmp_path: Path, registered) -> None:
    msg = _refusal(tmp_path / "band", {**FRONTEND, "carrier_frequency": "447 MHz"})
    assert "is not a channel of the KR 447 MHz raster [UNVERIFIED" in msg and "447.5625, 447.5750" in msg and "nothing is rounded to a channel" in msg
    msg = _refusal(tmp_path / "am", {**FRONTEND, "modulation": "AM"})
    assert "modulation 'am'" in msg and "FM telephony [UNVERIFIED" in msg
    msg = _refusal(tmp_path / "l2", {**FRONTEND, "pcb_layers": "2"})
    assert "2 layers is not a count template kr447_rx_frontend builds" in msg and "reference plane" in msg


def test_without_the_power_companion_the_template_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "RF_TEMPLATES", [KR447RxFrontendTemplate(companions=())])
    msg = _refusal(tmp_path, FRONTEND)
    assert "the RX power section (kr447 design §2.1, part P9) is not composed" in msg


def test_an_input_voltage_outside_the_rx_power_range_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "RF_TEMPLATES", [KR447RxFrontendTemplate()])
    for volts in ("5 V", "12 V"):
        msg = _refusal(tmp_path / volts.replace(" ", ""), {**FRONTEND, "input_voltage": volts})
        assert f"input_voltage {volts.split()[0]} V (req.input_voltage) is outside 6.4..8.4 V" in msg, msg


def test_a_missing_carrier_is_a_required_question(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    state = _run(ir, tmp_path, {**BASE, "radio_build": "rx_frontend", "input_voltage": "7.4 V"})
    assert state.blocked and [q.key for q in state.open_questions if q.required] == ["carrier_frequency"]
    q = next(q for q in state.open_questions if q.key == "carrier_frequency")
    assert "447.5625 MHz" in q.question and "is not a channel" in q.question


def _ctx(tmp_path: Path, shared_rails: bool = True, answers: dict | None = None) -> BlockContext:
    ir = _ir(tmp_path)
    ir.requirements.requirements.append(Requirement(id="req.carrier_frequency", key="carrier_frequency", text="f", kind=RequirementKind.EXPLICIT,
                                                    value=user_requirement(447.5625e6, "Hz"), category="rf"))
    inputs, _ = read_inputs(ir)
    shared = {c.key: t for c, t in profile_choices("t", True)}
    if shared_rails:
        for key, v in (("power.rx_5v", 5.0), ("power.rx_3v3", 3.3)):
            shared[key] = user_requirement(v, "V")
    return BlockContext(ir=ir, library=_REAL, template_id="t", confirmed=True, inputs=inputs, shared=shared)


def test_blocks_refuse_what_the_composition_did_not_share(tmp_path: Path) -> None:
    """The rail levels are the power block's parameters; the mixer's budget reads the front end's filter losses (P13 composes in this order)."""
    for block in (RxFrontendBlock(), RxMixerBlock(), LoChainBlock(), LoBufferBlock()):
        with pytest.raises(TemplateRefusal, match="power.rx_5v"):
            block.build(_ctx(tmp_path / block.id, shared_rails=False))
    with pytest.raises(TemplateRefusal, match="fe_bpf2.loss"):
        RxMixerBlock().build(_ctx(tmp_path / "mixer"))


# --------------------------------------------------------------------------- with the libraries


@needs_libs
def test_the_table_shows_every_profile_and_model_row_as_unverified(tmp_path: Path, registered) -> None:
    table = _present(_ir(tmp_path), tmp_path)
    assert table.startswith(f"Template 'kr447_rx_frontend' v{TEMPLATE_VERSION} (KR447 receiver front-end bench board)")
    assert "req.carrier_frequency: carrier_frequency = 447562500 Hz (stated as '447.5625 MHz')" in table
    rows = [line.strip() for line in table.splitlines()]
    profile = [r for r in rows if r.startswith("kr447.")]
    models = [r for r in rows if r.startswith("model.")]
    assert len(profile) == 13 and all("UNVERIFIED" in r for r in profile), profile
    assert {r.split(" ")[0].rstrip(":") for r in models} >= {
        "model.npn", "model.bfr92.r_out", "model.bfr92.r_in", "model.l_q.vhf", "model.l_q.uhf", "model.l_q.if1", "model.lna.port_r", "model.adex10.port_r",
        "model.pha1.port_r", "model.ifamp.port_r", "model.lna.nf", "model.lna.gain", "model.adex10.cl", "model.ifamp.nf", "model.ifamp.gain", "model.ifb.nf"}
    assert all("UNVERIFIED" in r for r in models), [r for r in models if "UNVERIFIED" not in r]
    for line in (
        "rf.lo1 = 426162500 Hz [calc.rf.superhet.lo from rf.f_c, rf.if1, rf.lo1_side]",
        "lo.f_ref = 35513541.6667 Hz [calc.clock.divided from rf.lo1, rf.n_mult]",
        "fe_bpf2.s21 = -3.3670672", "fe_bpf3.rel_image = -33.567324", "lo_tank1.rel_p = -26.3066456", "lo_bpf.rel_p1 = -19.217476", "lo_pad.s21 = -6 dB",
        "fe.sensitivity = -114.14321","Y701 Oscillator:KT2520K-T / Oscillator:Oscillator_SMD_Kyocera_2520-6Pin_2.5x2.0mm, value 35.514M (LO1 reference TCXO: 35.5135417 MHz",
        "U650 RF_Mixer:ADEX-10", "U750 RF_Amplifier:PHA-1", "U750 pin 1 has no name in the library; identified as RF_IN by its library electrical type 'input'",
        "SH601 Device:RFShield_OnePiece / RF_Shielding:Laird_Technologies_BMI-S-103_26.21x26.21mm",
        "fixture lo_pad: 3 member(s) between pad_in (port), lo1_mix (port); rows s21_lo1 (s21_db at 426162500 Hz: -6 +/- 0.2), s11_lo1 (s11_db at 426162500 Hz: at_most -20)",
        "design deck op_bias: ic_lna i(R604) = 0.00545455 +/- 15% under model.npn",
        "RF50: nets RX_RF, LNA_IN, LNA_OUT, MIX_RF, LO1_RAW, LO1_BUF_OUT, LO_PAD_IN, LO1_MIX; Z0 50 ohm +/- 10%",
        "RF_TANK_HI: nets LO_X12_C, LO_B_R1, LO_B_R2, LO_B_R3, LO_B_R4, FE2_R1,",
        "fixture lo_tank1: 11 member(s) between coll (port), next (port), rx_5v (rail)",
        "fixture lo_bpf: 15 member(s) between coll (port), buf_in (port), rx_5v (rail)",
    ):
        assert line in table, line


@needs_libs
def test_a_confirmation_applies_the_board(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    _confirm(ir, tmp_path)
    assert len(ir.components) == 105
    refs = {c.ref for c in ir.components}
    assert {"J1", "J601", "J602", "Q601", "Q650", "U650", "SH601", "Y701", "Q701", "Q702", "Q703", "SH701", "U750"} <= refs
    net = {(p.component_ref, p.pin_number): n.name for n in ir.nets for p in n.pins}
    assert {net[("U650", pin)] for pin in ("1", "4", "5")} == {"GND"} and {net[("Y701", pin)] for pin in ("1", "3")} == {"GND"}
    assert net[("U650", "3")] == "MIX_RF" and net[("U650", "6")] == "LO1_MIX" and net[("U650", "2")] == "MIX_IF"
    assert ir.component("Y701").pin("2").electrical_type.value == "no_connect"
    rf = ir.rf
    assert [b.id for b in rf.blocks] == ["bench_io", "lo_chain", "lo_buffer", "rx_frontend", "rx_mixer", "bench_rf_in", "bench_if_out"]
    assert [n.id for n in rf.networks] == ["lo_tank1", "lo_tank2", "lo_bpf", "lo_pad", "fe_bpf2", "fe_bpf3", "diplexer"]
    # kr447 wave-2 review findings 1 / 2: the LO networks hold every part on their port nets (the collector choke, the 0 ohm link, the next
    # stage's base divider) and no two top-C filters meet on a net (the last tank and the LO band-pass were joined tap to tap)
    top_c = [n.id for n in rf.networks if "tank" in n.id or "bpf" in n.id]
    assert nets_joining_networks(ir, top_c) == {}
    assert port_net_outsiders(ir, ["lo_tank1", "lo_tank2", "lo_bpf"]) == []
    for nid, choke, link, divider in (("lo_tank1", "L701", "R704", ("R705", "R706")), ("lo_tank2", "L704", "R708", ("R709", "R710")),
                                      ("lo_bpf", "L707", "R712", ())):
        nw = rf.network(nid)
        assert {choke, link, *divider} <= set(nw.members) and choke in nw.loss_q and nw.ports[-1].kind == "rail", nid
    assert {n.id: [e.id for e in n.expectations] for n in rf.networks} == NETWORK_ROWS
    assert rf.block("rx_frontend").shield_ref == "SH601" and rf.block("lo_chain").shield_ref == "SH701"
    assert [p.id for p in rf.frequency_plan] == ["lo_spur_p1m", "lo_spur_p2m", "lo_spur_m1p", "lo_spur_p1p", "birdie_fr_harmonic", "birdie_fr_if1", "image", "half_if"]
    assert sorted(x.id for x in rf.lab_items) == sorted(["lo_spur_response", "lo_frequency", "tank_alignment", "procurement_rx_tcxo", "lo_level", "lna",
                                                        "image_rejection", "half_if", "mixer", "sensitivity", "lo_radiation"])
    assert len(rf.profile_keys) == 13 and all(k.startswith("kr447.") for k in rf.profile_keys)
    assert all(k.startswith("model.") for k in rf.model_values) and "model.npn" in rf.model_values and len(rf.model_values) == 16
    # every choice is the user's now, never grounded; every calculator output derived
    for key, t in ir.parameters.items():
        assert t.provenance.kind in (ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.DERIVED), key
    freqs = {k: ir.parameters[k].value for k in ("rf.lo1", "fe.image", "fe.half_if", "lo.f_ref", "lo.f1", "lo.f2", "lo.f3", "lo.lo_spur_p1m", "lo.lo_spur_p2m",
                                                 "lo.lo_spur_m1p", "lo.lo_spur_p1p", "lo.birdie.f")}
    for key, value in (("rf.lo1", 426.1625e6), ("fe.image", 404.7625e6), ("fe.half_if", 436.8625e6), ("lo.f_ref", 35.51354166666e6), ("lo.f1", 106.540625e6),
                       ("lo.f2", 213.08125e6), ("lo.f3", 426.1625e6), ("lo.lo_spur_p1m", 440.2760417e6), ("lo.lo_spur_p2m", 475.7895833e6),
                       ("lo.lo_spur_m1p", 412.0489583e6), ("lo.lo_spur_p1p", 483.0760417e6), ("lo.birdie.f", 461.6760417e6)):
        assert math.isclose(freqs[key], value, rel_tol=1e-9), (key, freqs[key])
    assert ir.pcb is not None and ir.pcb.stackup is not None and ir.pcb.stackup.layer_count == 4
    assert {c.name for c in ir.si.net_classes} >= {"RF50", "IF50", "IF_NODE", "RF_TANK_LO", "RF_TANK_HI"}
    rf_nets = {n.name for n in ir.nets if n.kind.value == "rf"}
    classed = {name for c in ir.si.net_classes if not c.default for name in c.nets}
    assert rf_nets <= classed, sorted(rf_nets - classed)  # every RF net is in a class that says what it is (none left in DEFAULT)


@needs_libs
def test_the_floorplan_places_every_part_with_both_cans_holding_their_blocks(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    state = _confirm(ir, tmp_path, Stage.PLACEMENT)
    assert not state.blocked
    assert len(ir.pcb.placements) == 105 and {p.provenance.tool for p in ir.pcb.placements} == {"placement.rf_floorplan"}
    outline = ir.pcb.outline
    assert (outline.width_mm, outline.height_mm) == (71.0, 64.0)


@needs_libs
def test_the_default_board_composes_the_p9_rx_power_section(tmp_path: Path) -> None:
    """Through the registry as merged: the RX power section of part P9 supplies the board, serves input_voltage and brings its deck row."""
    assert [t.id for t in registry.RF_TEMPLATES].count("kr447_rx_frontend") == 1
    ir = _ir(tmp_path)
    table = _present(ir, tmp_path)
    assert "power.pack_cutoff_v" in table and "model.ldo_rx.r_in" in table
    state = _confirm(ir, tmp_path, Stage.PLACEMENT)
    assert not state.blocked
    refs = {c.ref for c in ir.components}
    assert len(refs) == 117 and {"J101", "Q101", "Q105", "U101", "U103", "SW101"} <= refs and not {"J1", "U102", "Q103"} & refs
    assert ir.rf is not None and [b.id for b in ir.rf.blocks] == ["power", "lo_chain", "lo_buffer", "rx_frontend", "rx_mixer", "bench_rf_in", "bench_if_out"]
    assert "req.input_voltage" in ir.component("J101").serves_requirements
    assert ir.simulation is not None and {e.id for e in ir.simulation.expectations} == {*BIAS, "main_switch_on"}
    assert (ir.pcb.outline.width_mm, ir.pcb.outline.height_mm) == (71.0, 82.0) and len(ir.pcb.placements) == 117
    notes = KR447RxFrontendTemplate().part_notes(ir)
    assert notes["U101"].role.startswith("U101: RX_5V") and "수신 전용" in notes["Q101"].role


@needs_libs
def test_rebuilds_are_byte_identical(tmp_path: Path, registered) -> None:
    """Two independent projects give the same table, the same design hash and the same compiled schematic and PCB bytes."""
    seen = []
    for k in range(2):
        work = tmp_path / f"p{k}"
        ir = _ir(work)
        table = _present(ir, work)
        _confirm(ir, work, Stage.PLACEMENT)
        out = []
        for compiler in (SchematicCompiler(), PCBCompiler()):
            art = compiler.compile(ir, CompileContext(workdir=work / "out", tools={"kicad_library": _REAL}))
            out.append(hashlib.sha256(Path(art.path).read_bytes()).hexdigest())
        seen.append((hashlib.sha256(table.encode()).hexdigest(), ir.content_hash(), *out))
    assert seen[0] == seen[1]


@needs_libs
def test_checks_on_the_confirmed_board(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    _confirm(ir, tmp_path, Stage.CALCULATION)
    latest = ir.validation.latest_by_check()
    assert latest["calc.recompute"].status is S.PASS
    assert check_inputs_vs_requirements(ir).status is S.PASS
    interfaces = {k: v.status for k, v in latest.items() if k.startswith("block.interface.")}
    assert interfaces == {f"block.interface.{n}": S.PASS for n in ("IF1", "LO1_MIX", "LO1_RAW", "MIX_RF", "RX_3V3", "RX_5V", "RX_RF")}
    for check in ("rf.model_grounding", "rf.regulatory_profile", "rf.freq_plan"):
        assert latest[check].status is S.NOT_VERIFIED, (check, latest[check].message)
    rows = {r["id"]: r["status"] for r in latest["rf.freq_plan"].details["rows"]}
    assert rows["birdie_fr_harmonic"] == rows["birdie_fr_if1"] == "PASS"
    assert {rows[k] for k in ("image", "half_if", "lo_spur_p1m", "lo_spur_p2m", "lo_spur_m1p", "lo_spur_p1p")} == {"NOT_VERIFIED"}
    assert all(v.status is S.NOT_VERIFIED for k, v in latest.items() if k.startswith("rf.lab."))


@needs_libs
def test_optional_requirements_reach_the_design(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    table = _present(ir, tmp_path, {"frequency_tolerance": "±1.5 ppm", "rx_sensitivity": "-118 dBm", "system_impedance": "50 ohm", "modulation": "FM"})
    assert "rf.tcxo_stability" not in table and "req.frequency_tolerance: frequency_tolerance = 1.5 ppm" in table
    _confirm(ir, tmp_path)
    assert ir.parameters["rf.frequency_tolerance"].value == 1.5 and math.isclose(ir.parameters["lo.lo1_error"].value, 426.1625e6 * 1.5e-6)
    y = ir.component("Y701")
    ids = {r.key: r.id for r in ir.requirements.requirements}
    assert ids["frequency_tolerance"] in y.serves_requirements and ids["rx_sensitivity"] in ir.component("Q601").serves_requirements
    assert ids["radio_build"] in ir.component("U650").serves_requirements and ids["modulation"] in ir.component("U650").serves_requirements
    lab = next(x for x in ir.rf.lab_items if x.id == "sensitivity")
    assert "rx_sensitivity = -118 dBm" in lab.what and "-114.1 dBm" in lab.what and "never a verdict" in lab.what


@needs_libs
def test_theory_figures_and_part_notes(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    _confirm(ir, tmp_path)
    tpl = registered
    sections = tpl.theory(ir)
    assert [s.title for s in sections][:2] == ["개요: 수신 프런트엔드 시험 보드(3단계)", "주파수 계획"]
    text = "\n".join(s.body for s in sections)
    for needle in ("426.16 MHz", "35.5135417 MHz", "440.28 MHz", "-3.367 dB", "UNVERIFIED", "KC 적합성평가", "−114.1".replace("−", "-")):
        assert needle in text, needle
    assert "기록 없음" not in text
    figures = tpl.theory_figures(ir)
    assert [f.id for f in figures] == ["theory_fe_s21", "theory_lo_s21"] and all(f.svg.startswith("<svg") for f in figures)
    notes = tpl.part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components}
    assert all(UNVERIFIED_SUBSTITUTE in s for n in notes.values() for s in n.substitutes)
    assert "ADI LT5560" in " ".join(notes["U650"].substitutes)


# --------------------------------------------------------------------------- with ngspice


@needs_libs
@needs_dll
def test_every_fixture_row_and_bias_point_passes_on_the_default_models_within_the_budget(tmp_path: Path, registered) -> None:
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    state = _confirm(ir, tmp_path, Stage.SPICE, spice=True)
    assert not state.blocked
    latest = ir.validation.latest_by_check()
    for nid, rows in NETWORK_ROWS.items():
        summary = latest[f"spice.rf.{nid}"]
        assert summary.status is S.PASS, (nid, summary.message)
        for rid in rows:
            r = latest[f"spice.rf.{nid}.{rid}"]
            assert r.status is S.PASS, (nid, rid, r.message)
            assert "network verdict under the confirmed model values" in r.message
    for exp in BIAS:
        assert latest[f"spice.{exp}"].status is S.PASS, latest[f"spice.{exp}"].message
    assert latest["spice"].status is S.PASS
    # the measured networks are the calculators' (within 0.01 dB), and the design's decided Q-40 numbers where the networks are the design's
    measured = {f"{nid}.{rid}": latest[f"spice.rf.{nid}.{rid}"].details["measured"] for nid, rows in NETWORK_ROWS.items() for rid in rows
                if latest[f"spice.rf.{nid}.{rid}"].details.get("measured") is not None}
    for key, value in (("fe_bpf2.s21_fc", -3.37), ("fe_bpf2.rel_image", -13.56), ("fe_bpf3.s21_fc", -9.30), ("fe_bpf3.rel_image", -33.57),
                       ("fe_bpf3.rel_lo1", -15.42), ("lo_bpf.s21_lo1", -8.01), ("lo_bpf.rel_m1", -27.94), ("lo_bpf.rel_p1", -19.22),
                       ("lo_bpf.rel_m2", -56.15), ("lo_bpf.rel_p2", -37.72), ("lo_tank1.s21", -5.50), ("lo_tank2.s21", -5.55),
                       ("lo_tank1.rel_m", -47.79), ("lo_tank1.rel_p", -26.31), ("lo_tank2.rel_m", -29.05), ("lo_tank2.rel_p", -17.65),
                       ("lo_pad.s21_lo1", -6.0)):
        assert abs(measured[key] - value) < 0.01, (key, measured[key])
    assert latest["spice.rf.lo_pad.s11_lo1"].details.get("zero_magnitude") or latest["spice.rf.lo_pad.s11_lo1"].details["measured"] < -100
    assert latest["spice.rf.diplexer.s21_if1"].details["measured"] > -1.0 and latest["spice.rf.diplexer.s11_lo1"].details["measured"] < -10.0
    # the ported networks' nominals are the networks the fixtures run (choke, link and next divider included)
    for nid, rows in (("lo_tank1", ("s21", "rel_m", "rel_p")), ("lo_tank2", ("s21", "rel_m", "rel_p")), ("lo_bpf", ("s21_lo1", "rel_m1", "rel_p1"))):
        for rid in rows:
            key = f"{nid}.{'s21' if rid == 's21_lo1' else rid}"
            assert abs(measured[f"{nid}.{rid}"] - float(ir.parameters[key].value)) < 0.01, (nid, rid)
    # the budget: every fixture deck again, timed alone (kr447 design §3.3: 7 fixtures + bias ops < 2 s)
    start = time.perf_counter()
    results = spice_rf_results(ir, {"spice": runner}, tmp_path / "again")
    elapsed = time.perf_counter() - start
    assert {r.check_id for r in results if r.status is not S.PASS} == set()
    assert elapsed < FIXTURE_BUDGET_S, elapsed
    # the probes are recorded without a verdict: the half-IF and the four LO-spur responses on both front-end filters
    probes = latest["spice.rf.fe_bpf3"].details.get("probes") or {}
    assert set(probes) >= {"half_if", "lo_spur_p1m", "lo_spur_p2m", "lo_spur_m1p", "lo_spur_p1p"}


@needs_libs
def test_the_fixture_nominals_are_the_exact_networks(tmp_path: Path, registered) -> None:
    """The tank nominals come from calc.rf.resonator.top_c on the tank's loaded ports (1 kohm with the collector choke -> 500 ohm with the next
    stage's divider), never a symmetric formula and never the port models alone."""
    ir = _ir(tmp_path)
    _present(ir, tmp_path)
    _confirm(ir, tmp_path)
    p = {k: t.value for k, t in ir.parameters.items()}
    f1, bw1, l1 = p["lo.f1"], p["lo.tank1.bw"], p["lo.tank1.l"]
    assert math.isclose(bw1, 1.4142135623730951 * f1 / 20.0, rel_tol=1e-12)
    r_eff = 1.0 / (1.0 / 500.0 + 1.0 / 4700.0 + 1.0 / 2200.0)  # the x6 stage's port model // its base divider
    assert math.isclose(p["lo.tank1.r_load_eff"], r_eff, rel_tol=1e-12)
    exact_p = radio.top_c_ported_rel_s21_db_value(2, f1, bw1, l1, 1000.0, p["lo.x3.l_choke"], 40.0, 500.0, r_eff, 40.0, f1 + p["lo.f_ref"], f1)
    port_models_only = radio.top_c_rel_s21_db_value(2, f1, bw1, l1, 1000.0, 500.0, 40.0, f1 + p["lo.f_ref"], f1)
    symmetric = -radio.double_tuned_rejection_db(p["lo.tank_qe"], f1, f1 + p["lo.f_ref"])  # the symmetric ceiling at Q_L = Q_e: a paper number only
    assert p["lo_tank1.rel_p"] == exact_p and exact_p - symmetric > 5.0 and abs(exact_p - port_models_only) > 0.1
    # the taps transform: R_p = Q_e w0 L above the collector port with its choke (R') on both tanks
    for k in (1, 2):
        assert 20.0 * 2 * math.pi * p[f"lo.f{k}"] * p[f"lo.tank{k}.l"] > p[f"lo_tank{k}.port_r"]
