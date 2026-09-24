"""CircuitDesignAgent + the deterministic templates (divider / LED / RC low-pass) with the confirm_design flow.

Offline tests run on a synthetic KiCad library written into ``tmp_path``
(``Device:R`` / ``C`` / ``LED``, ``Connector_Generic:Conn_01x02`` / ``03`` and
their footprints; the LED pins are numbered ``1 = K``, ``2 = A`` on purpose,
so a template that wired the LED by pin *number* would get the polarity
wrong). The pipeline runs to RELEASE without kicad-cli (ERC / DRC
NOT_VERIFIED) and, where ngspice is found, against the real engine
(``needs_dll``); nothing here asserts an unmeasured tool behaviour. The
real KiCad ``Device:LED`` pin naming and ERC on the template schematics are
only checked on the Windows PC (``test_vertical_slice`` style, skipped here).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, CircuitDesignAgent
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY, table_hash
from ai_eda.agents.keys import CONTROL_KEYS, PLACEMENT_KEY
from ai_eda.agents.requirement import CONTROL_KEYS as REQUIREMENT_CONTROL_KEYS
from ai_eda.compilers import BOMCompiler, CompileContext, SchematicCompiler, SpiceNetlistCompiler
from ai_eda.design import (
    CHOICE_NOTE_PREFIX,
    INPUTS_CHECK,
    TEMPLATE_VERSION,
    TOOL_ID,
    check_inputs_vs_requirements,
    late_load_changes,
    read_inputs,
    read_value,
)
from ai_eda.ir import (
    ArtifactKind,
    CircuitIR,
    Expectation,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    Requirement,
    RequirementKind,
    ValidationStatus as S,
    llm_generated,
    user_requirement,
)
from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.calc import CALC_VERSION, parse_answer, recompute_parameters
from ai_eda.tools.calc.basic import led_series_resistor
from ai_eda.tools.calc.quantity import QUANTITY_VERSION
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S as SX
from ai_eda.tools.spice import NgspiceShared
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow import Orchestrator, Stage
from tests.test_requirement_agent_llm import USAGE, _req, _service

runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

BASE = {"application": "bench", "jurisdiction": "EU"}
DIVIDER = {"input_voltage": "12 V", "output_voltage": "5 V"}
LED = {"input_voltage": "5 V", "led_forward_voltage": "2.0 V", "led_forward_current": "10 mA"}
RC = {"cutoff_frequency": "1 kHz"}
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")


# --------------------------------------------------------------------------- synthetic library


def _effects():
    return SX("effects", SX("font", SX("size", 1.27, 1.27)))


def _prop(key, value, hide=False):
    return SX("property", Q(key), Q(value), SX("at", 0, 0, 0), SX("hide", True) if hide else None, _effects())


def _pin(number, name, x, angle):
    return SX("pin", "passive", "line", SX("at", x, 0, angle), SX("length", 2.54), SX("name", Q(name), _effects()), SX("number", Q(number), _effects()))


def _symbol(name, ref_prefix, pins, footprint):
    body = SX("symbol", Q(f"{name}_0_1"), SX("rectangle", SX("start", -2.54, 1.016), SX("end", 2.54, -1.016), SX("stroke", SX("width", 0.254), SX("type", "default")), SX("fill", SX("type", "none"))))
    return SX(
        "symbol", Q(name), SX("pin_names", SX("offset", 1.016)), SX("exclude_from_sim", False), SX("in_bom", True), SX("on_board", True),
        _prop("Reference", ref_prefix), _prop("Value", name), _prop("Footprint", footprint, True), _prop("Datasheet", "~", True), _prop("Description", name, True),
        body, SX("symbol", Q(f"{name}_1_1"), *pins), SX("embedded_fonts", False),
    )


def _lib(*symbols):
    return SX("kicad_symbol_lib", SX("version", 20251024), SX("generator", Q("kicad_symbol_editor")), SX("generator_version", Q("10.0")), *symbols)


def _footprint(name, n_pads):
    pads = [SX("pad", Q(str(i + 1)), "smd", "roundrect", SX("at", -0.8 * (n_pads - 1) + 1.6 * i, 0), SX("size", 0.8, 0.9), SX("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste")), SX("roundrect_rratio", 0.25)) for i in range(n_pads)]
    w = 0.8 * n_pads + 0.4
    return SX("footprint", Q(name), SX("version", 20260206), SX("generator", Q("pcbnew")), SX("layer", Q("F.Cu")), SX("descr", Q(name)), SX("attr", "smd"),
              SX("fp_rect", SX("start", -w, -0.6), SX("end", w, 0.6), SX("stroke", SX("width", 0.05), SX("type", "solid")), SX("fill", "no"), SX("layer", Q("F.CrtYd"))),
              *pads, SX("embedded_fonts", False))


def template_library(root: Path, *, led: bool = True, led_pin_names: tuple[str, str] = ("K", "A")) -> KicadLibrary:
    """Device:R / C [/ LED], Connector_Generic:Conn_01x02 / 03 and their footprints. The LED's pin 1 is the cathode ``K``, pin 2 the anode ``A``."""
    two = lambda: [_pin("1", "~", -5.08, 0), _pin("2", "~", 5.08, 180)]  # noqa: E731
    symbols = [_symbol("R", "R", two(), "Resistor_SMD:R_0603_1608Metric"), _symbol("C", "C", two(), "Capacitor_SMD:C_0603_1608Metric")]
    if led:
        symbols.append(_symbol("LED", "D", [_pin("1", led_pin_names[0], -5.08, 0), _pin("2", led_pin_names[1], 5.08, 180)], "LED_SMD:LED_0603_1608Metric"))
    conn = _lib(
        _symbol("Conn_01x02", "J", [_pin("1", "Pin_1", -5.08, 0), _pin("2", "Pin_2", 5.08, 180)], "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
        _symbol("Conn_01x03", "J", [_pin("1", "Pin_1", -7.62, 0), _pin("2", "Pin_2", 0, 90), _pin("3", "Pin_3", 7.62, 180)], "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"),
    )
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    (root / "symbols" / "Device.kicad_sym").write_text(sexpr.dumps(_lib(*symbols)), encoding="utf-8")
    (root / "symbols" / "Connector_Generic.kicad_sym").write_text(sexpr.dumps(conn), encoding="utf-8")
    for lib, name, n in (("Resistor_SMD", "R_0603_1608Metric", 2), ("Capacitor_SMD", "C_0603_1608Metric", 2), ("LED_SMD", "LED_0603_1608Metric", 2),
                         ("Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical", 2), ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical", 3)):
        pretty = root / "footprints" / f"{lib}.pretty"
        pretty.mkdir(parents=True, exist_ok=True)
        sexpr.dump_file(_footprint(name, n), pretty / f"{name}.kicad_mod")
    return KicadLibrary(roots=[root])


# --------------------------------------------------------------------------- helpers


def _ir(tmp_path: Path, name: str = "tpl") -> CircuitIR:
    return CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))


def _ctx(tmp_path: Path, lib: KicadLibrary | None, answers: dict[str, str], *, spice: bool = False) -> AgentContext:
    tools: dict = {"kicad_library": lib} if lib is not None else {}
    if spice:
        tools["spice"] = runner
    return AgentContext(workdir=tmp_path, tools=tools, answers=answers)


def _run(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary | None, answers: dict[str, str], stop_after: Stage | None = Stage.ARCHITECTURE, *, spice: bool = False):
    ctx = _ctx(tmp_path, lib, answers, spice=spice)
    return Orchestrator(ctx).run(ir, stop_after=stop_after), ctx


def _present(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, answers: dict[str, str]):
    """Run 1: the table is presented and the pipeline blocks on confirm_design."""
    state, _ = _run(ir, tmp_path, lib, {**BASE, **answers})
    assert state.blocked and state.current is Stage.ARCHITECTURE
    assert [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert ir.components == [] and ir.nets == [] and ir.parameters == {} and ir.simulation is None and ir.topology is None
    question = state.open_questions[0].question
    assert ir.requirements.presented[CONFIRM_DESIGN_KEY] == table_hash(question) and "presented" not in ir.design_dict()["requirements"]
    return question


def _confirm(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, stop_after: Stage | None = Stage.ARCHITECTURE, *, spice: bool = False):
    """Run 2: ``confirm_design=yes`` alone (the typed answers are already requirements in the IR)."""
    return _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"}, stop_after, spice=spice)


def _validate(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> dict[str, S]:
    return {r.check_id: r.status for r in default_registry.run(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))}


def _netlist(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> str:
    art = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path / "net", tools={"kicad_library": lib}))
    return Path(art.path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- present, then confirm


def test_first_run_presents_the_table_and_applies_nothing(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, lib, DIVIDER)
    assert question.startswith(f"Template 'divider' v{TEMPLATE_VERSION} (unloaded resistive voltage divider)")
    assert "req.input_voltage: input_voltage = 12 V (stated as '12 V')" in question
    assert "req.output_voltage: output_voltage = 5 V (stated as '5 V')" in question
    assert "r2 = 10000 ohm - lower resistor" in question and "tol_rel = 0.01 - relative tolerance" in question
    assert "r1 = 14000 ohm [calc.divider.r1_for_v_out from v_in, v_out_target, r2]" in question
    assert "R1 Device:R / Resistor_SMD:R_0603_1608Metric, value 14k (pins 1, 2 from " in question
    assert "J1 Connector_Generic:Conn_01x03 / Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" in question
    assert "VOUT (signal): J1.2, R1.2, R2.1" in question and "v(VOUT) = 5 V +/- 1% verifies req.output_voltage" in question
    # the agent alone: a pure function of the IR, no validation result, no mutation
    before = ir.content_hash()
    res = CircuitDesignAgent().run(ir, _ctx(tmp_path, lib, {}))
    assert ir.content_hash() == before and res.proposals == [] and res.validation == [] and res.blocked_on_user
    assert res.questions[0].question == question  # deterministic: the same IR yields the same table
    assert [q.key for q in res.questions] == [CONFIRM_DESIGN_KEY, "output_current"]  # the advisory load question is asked with the table
    # before any table was shown, the only proposal is the record of the one shown now (outside the design hash)
    ir.requirements.presented.clear()
    res = CircuitDesignAgent().run(ir, _ctx(tmp_path, lib, {}))
    assert [(p.target, p.operation) for p in res.proposals] == [("requirements.presented", "set")] and res.proposals[0].payload == {CONFIRM_DESIGN_KEY: table_hash(question)}
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.content_hash() == before


def test_confirm_applies_the_divider_with_the_choices_as_user_values(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "divider")
    _present(ir, tmp_path, lib, DIVIDER)
    ids_before = [r.id for r in ir.requirements.requirements]
    state, ctx = _confirm(ir, tmp_path, lib, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert out.status is S.NOT_VERIFIED and out.message.startswith("15 proposal(s) applied, nothing verified; template divider v0.1 confirmed by the user: 15 proposal(s)")
    assert "design choices recorded as the user's values: r2 = 10000 ohm" in out.message
    assert [q.key for q in out.questions] == ["output_current"] and not out.questions[0].required  # advisory, the real key
    assert not state.blocked and state.outcomes[-1].stage is Stage.RELEASE and state.outcomes[-1].status is not S.PASS
    assert [r.id for r in ir.requirements.requirements] == ids_before  # a template never authors a requirement
    assert ir.requirements.get(CONFIRM_DESIGN_KEY) is None and CONFIRM_DESIGN_KEY in CONTROL_KEYS and REQUIREMENT_CONTROL_KEYS is CONTROL_KEYS
    # the design
    assert [c.ref for c in ir.components] == ["R1", "R2", "J1"] and [n.name for n in ir.nets] == ["VIN", "VOUT", "GND"]
    assert ir.topology.name == "resistive divider" and [c.id for c in ir.constraints] == ["c.divider.unloaded"]
    p = ir.parameters
    assert list(p) == ["v_in", "v_out_target", "r2", "r1", "v_out", "tol_rel"]
    for key, rid, value in (("v_in", "req.input_voltage", 12.0), ("v_out_target", "req.output_voltage", 5.0)):
        assert p[key].value == value and p[key].provenance.kind is ProvenanceKind.USER_REQUIREMENT
        assert p[key].provenance.derived_from == [rid] and p[key].provenance.note.startswith(f"parsed from {rid}: ")
    for key in ("r2", "tol_rel"):
        assert p[key].provenance.kind is ProvenanceKind.USER_REQUIREMENT and p[key].provenance.tool is None
        assert p[key].provenance.note.startswith(f"{CHOICE_NOTE_PREFIX}; template divider v{TEMPLATE_VERSION}: {key} = ")
    assert p["r1"].value == 14000.0 and p["r1"].unit == "ohm" and p["r1"].provenance.tool == "calc.divider.r1_for_v_out"
    assert p["r1"].provenance.inputs == {"v_in": "v_in", "v_out": "v_out_target", "r2": "r2"} and p["r1"].provenance.tool_version == CALC_VERSION
    assert p["v_out"].value == 5.0 and p["v_out"].provenance.tool == "calc.divider.v_out"
    r1 = ir.component("R1")
    assert r1.electrical["resistance"].value == r1.spice.value.value == 14000.0 and r1.value == "14k" and r1.symbol.verified and r1.footprint.verified
    assert [(pin.number, pin.provenance.kind, pin.provenance.source.content_hash[:7]) for pin in r1.pins] == [("1", ProvenanceKind.AUTHORITATIVE, "sha256:"), ("2", ProvenanceKind.AUTHORITATIVE, "sha256:")]
    assert r1.pins[0].provenance.source.title == "KiCad symbol Device:R" and r1.pins[0].provenance.source.document_path.endswith("Device.kicad_sym")
    assert ir.component("J1").spice.exclude and ir.net("VOUT").serves_requirements == ["req.output_voltage"]
    exp = ir.simulation.expectations[0]
    assert exp.vector == "v(VOUT)" and exp.nominal.value == 5.0 and exp.nominal.provenance.tool == "calc.divider.v_out" and exp.requirement_id == "req.output_voltage"
    assert exp.tol_rel.value == 0.01 and exp.tol_rel.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    # verification: only computed values are recomputed, no assumption / model value anywhere
    rec = recompute_parameters(ir)
    assert rec.status is S.PASS and rec.message == "5 value(s) recomputed"
    assert set(rec.details["parameters"]) == {"r1", "v_out", "components[R1].electrical[resistance]", "components[R1].spice.value", "simulation.expectations[v_out].nominal"}
    validators = _validate(ir, tmp_path, lib)
    assert validators["ir.assumptions"] is S.PASS and validators["ir.llm_requirements"] is S.PASS and validators["ir.connectivity"] is S.PASS
    assert ir.validation.latest("calc.recompute").status is S.PASS
    # the pipeline placed and compiled it offline
    assert ir.pcb is not None and sorted(x.component_ref for x in ir.pcb.placements) == ["J1", "R1", "R2"]  # the grid placer orders refs
    assert state.outcome(Stage.SCHEMATIC).status is S.PASS and state.outcome(Stage.PCB).status is S.PASS
    assert ir.validation.latest("compile.bom").status is S.PASS and ArtifactKind.BOM in ir.artifacts
    assert state.outcome(Stage.ERC).status is S.NOT_VERIFIED  # kicad-cli decides, and it is not here
    assert _netlist(ir, tmp_path, lib) == "divider\nR1 VIN VOUT 14k\nR2 VOUT 0 10k\nVVIN VIN 0 DC 12\n.end\n"
    # the reviewer reads the typed '5 V' as the requirement value and compares it with the nominal
    report = IndependentReviewer(tools=ctx.tools).review(ir, tmp_path)
    results = {r.check_id: r for r in report.results}
    assert results[ReviewArea.REQUIREMENTS_VS_IR].status is S.PASS and results[ReviewArea.CALCULATIONS_VS_DESIGN].status is S.PASS
    assert results[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.NOT_VERIFIED  # no SPICE engine in this offline run: nothing simulated, nothing claimed
    # hash: stable across save / load (two independent builds: test_two_independent_builds_hash_the_same)
    saved = ir.save(tmp_path / "ir.json")
    assert CircuitIR.load(saved).content_hash() == ir.content_hash()
    # a run on the built design proposes nothing and re-checks the copied inputs against their requirements
    state3, _ = _run(ir, tmp_path, lib, {})
    out3 = state3.outcome(Stage.ARCHITECTURE)
    assert out3.status is S.PASS and out3.message.startswith("design content already present (3 component(s), 3 net(s), a topology, a simulation setup); templates only start an empty design, nothing proposed")
    check = ir.validation.latest(INPUTS_CHECK)
    assert check.status is S.PASS and check.tool == TOOL_ID and check.tool_version == TEMPLATE_VERSION and check.ir_hash is None  # unstamped: an agent result of this run
    assert check.details["quantity_version"] == QUANTITY_VERSION and set(check.details["parameters"]) == {"v_in", "v_out_target"}
    assert len(ir.components) == 3 and out3.questions == []


def test_two_independent_builds_hash_the_same(tmp_path: Path):
    hashes = []
    for name in ("a", "b"):
        lib = template_library(tmp_path / name / "kicad")
        ir = _ir(tmp_path / name, "divider")
        _present(ir, tmp_path / name, lib, DIVIDER)
        _confirm(ir, tmp_path / name, lib, Stage.ARCHITECTURE)
        hashes.append(ir.content_hash())
    assert hashes[0] == hashes[1]


def test_confirmation_counts_only_for_a_table_shown_in_an_earlier_run(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, CONFIRM_DESIGN_KEY: "yes"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in out.questions] == [CONFIRM_DESIGN_KEY, "output_current"] and ir.components == []
    assert f"{CONFIRM_DESIGN_KEY} ignored: answer(s) ['application', 'input_voltage', 'jurisdiction', 'output_voltage'] were given in this run" in out.message
    state, _ = _confirm(ir, tmp_path, lib)  # the inputs are requirements now: the table was shown, the confirmation counts
    assert not state.blocked and len(ir.components) == 3


def test_no_leaves_the_design_empty_and_other_answers_ask_again(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    _present(ir, tmp_path, lib, DIVIDER)
    state, _ = _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "no"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and out.status is S.NOT_VERIFIED and out.questions == [] and ir.components == []
    assert f"{CONFIRM_DESIGN_KEY}='no': template divider not applied, the design stays empty" in out.message
    state, _ = _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "maybe later"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and [q.key for q in out.questions] == [CONFIRM_DESIGN_KEY, "output_current"] and ir.components == []
    assert "answer 'maybe later' to confirm_design not understood" in out.message


# --------------------------------------------------------------------------- refusals: notes and real keys, never a guess


def test_divider_refuses_a_load_with_the_real_key_and_the_pipeline_continues(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, ctx = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, "output_current": "2 A"}, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert out.status is S.NOT_VERIFIED and ir.components == [] and not state.blocked
    assert "template divider not proposed: the request needs 5 V at 2 A from 12 V (req.output_current): a resistive divider cannot supply a load" in out.message
    assert [(q.key, q.required) for q in out.questions] == [("output_current", False)]
    q = out.questions[0].question
    assert q.startswith("The request needs 5 V at 2 A from 12 V (req.output_current): a resistive divider cannot supply a load. This template cannot serve that requirement")
    assert "Change the requirement req.output_current in the IR (or correct the request) so it states 0 A, choose another design" in q
    assert "Answer output_current=0 A" not in q  # req.output_current is the user's typed value: an --answer for that key would be kept out
    assert state.outcomes[-1].stage is Stage.RELEASE and state.outcomes[-1].status is not S.PASS
    assert ir.requirements.get("circuit_for_load") is None  # no system-authored key became a requirement
    report = IndependentReviewer(tools=ctx.tools).review(ir, tmp_path)
    r = {x.check_id: x for x in report.results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.FAIL and set(r.details["unserved"]) == {"req.input_voltage", "req.output_voltage", "req.output_current"}


def test_every_refusal_reason_is_reported_at_once(tmp_path: Path):
    """The fake-LLM demo's confirmed extraction (12 V -> 5 V / 2 A, 90 % efficiency): both reasons, each under its own requirement key."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, "output_current": "2 A", "efficiency": "90 %"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and [q.key for q in out.questions] == ["efficiency", "output_current"]
    assert out.message == (
        "template divider not proposed: req.efficiency (efficiency: '90 %') is not served by the unloaded resistive voltage divider template, "
        "which serves only input_voltage, output_voltage, output_current; the request needs 5 V at 2 A from 12 V (req.output_current): "
        "a resistive divider cannot supply a load"
    )


@pytest.mark.parametrize("extra", [{"efficiency": "90 %"}, {"operating_temperature": "-20..85 °C"}])
def test_closed_world_selection_refuses_unserved_design_requirements(tmp_path: Path, extra: dict[str, str]):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, **extra})
    out = state.outcome(Stage.ARCHITECTURE)
    key = next(iter(extra))
    assert not state.blocked and ir.components == [] and f"template divider not proposed: req.{key} ({key}: {extra[key]!r}) is not served" in out.message
    assert [(q.key, q.required) for q in out.questions] == [(key, False)]
    assert f"req.{key} ({key}: {extra[key]!r}) is not served by the unloaded resistive voltage divider template, which serves only input_voltage, output_voltage, output_current" in out.questions[0].question


def test_divider_serves_a_zero_load_current(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, lib, {**DIVIDER, "output_current": "0 A"})
    assert "req.output_current: output_current = 0 A (stated as '0 A')" in question
    state, ctx = _confirm(ir, tmp_path, lib)
    out = state.outcome(Stage.ARCHITECTURE)
    assert out.questions == [] and "req.output_current = 0 A: VOUT is a high-impedance reference and serves it" in out.message
    assert ir.net("VOUT").serves_requirements == ["req.output_voltage", "req.output_current"]
    r = IndependentReviewer(tools=ctx.tools).check_requirements_vs_ir(ir, tmp_path)
    assert r.status is S.PASS and set(r.details["traced"]) == {"req.input_voltage", "req.output_voltage", "req.output_current"}


def test_invalid_ratio_and_ambiguous_templates_are_notes(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, "input_voltage": "5 V", "output_voltage": "12 V"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and out.questions == [] and "template divider not proposed: output 12 V is not between 0 and the input 5 V" in out.message
    ir = _ir(tmp_path, "amb")
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, **RC})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and "ambiguous: templates ['divider', 'rc_lowpass'] all match the confirmed requirements" in out.message
    assert "refusing to guess" in out.message


def test_agent_refuses_a_non_empty_design_a_parameter_clash_and_a_missing_library(tmp_path: Path, divider_ir: CircuitIR):
    lib = template_library(tmp_path / "kicad")
    ir = divider_ir  # a hand-made design: nothing to propose, and no template input to re-check
    res = CircuitDesignAgent().run(ir, _ctx(tmp_path, lib, {**DIVIDER, CONFIRM_DESIGN_KEY: "yes"}))
    assert res.proposals == [] and res.validation == [] and res.questions == []
    assert res.notes[0].startswith("design content already present (2 component(s), 3 net(s), a topology); templates only start an empty design")
    assert f"{CONFIRM_DESIGN_KEY} ignored: design content already present" in res.notes
    assert check_inputs_vs_requirements(ir) is None
    ir = _ir(tmp_path, "clash")
    _run(ir, tmp_path, lib, {**BASE, **DIVIDER})
    ir.parameters["r1"] = user_requirement(1.0, "ohm")
    res = CircuitDesignAgent().run(ir, _ctx(tmp_path, lib, {}))
    assert res.proposals == [] and res.questions == [] and "template divider not proposed: ir.parameters ['r1'] already exist and the template would overwrite them" in res.notes
    res = CircuitDesignAgent().run(ir, _ctx(tmp_path, None, {}))
    assert res.proposals == [] and res.questions == [] and res.notes == ["no KiCad library in ctx.tools['kicad_library']: a template cannot resolve its symbols and footprints, nothing proposed"]
    ir = _ir(tmp_path, "none")
    state, _ = _run(ir, tmp_path, lib, {**BASE, CONFIRM_DESIGN_KEY: "yes"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and "no template matches the confirmed requirements (templates: divider needs input_voltage + output_voltage; led needs" in out.message
    assert f"{CONFIRM_DESIGN_KEY} ignored: no template to confirm" in out.message


# --------------------------------------------------------------------------- inputs: whole-string parse, aliases, ambiguity, unconfirmed values


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12 V", 12.0), ("12 V DC", 12.0), ("DC 12 V", 12.0), ("12 V (DC)", 12.0), ("직류 12 V", 12.0), ("  1 kHz ", 1000.0), ("10 mA", 0.01), ("230 V AC", 230.0),
        ("12 V max", None), ("min 5 V", None), ("12 V rms", None), ("not more than 12 V", None), ("12V 입력", None), ("3.3~5V", None), ("5V 2A", None), ("±5%", None), ("10k", None), ("5 V typ / 6 V max", None),
    ],
)
def test_parse_answer_consumes_the_whole_text(text: str, expected: float | None):
    q = parse_answer(text)
    assert (None if q is None else q.value) == expected


def test_unusable_values_are_named_with_the_leftover_text_never_guessed(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    cases = {
        "3.3~5V": "a range (3.3..5 V), not one value",
        "5V 2A": "several quantities (5 V, 2 A), not one value",
        "12 V max": "qualifier 'max' beside '12 V' is not read; state one plain value",
        "min 5 V": "qualifier 'min' beside '5 V' is not read",
        "12 V rms": "qualifier 'rms' beside '12 V' is not read",
        "10k": "no quantity with a unit",
        "5 A": "'5 A' is a A quantity, not V",
    }
    for text, why in cases.items():
        ir = _ir(tmp_path)
        state, _ = _run(ir, tmp_path, lib, {**BASE, "input_voltage": "12 V", "output_voltage": text})
        out = state.outcome(Stage.ARCHITECTURE)
        assert not state.blocked and ir.components == [] and f"output_voltage not usable: req.output_voltage: " in out.message and why in out.message, text
    # AC/DC words and aliases are read; the alias lands under the canonical key
    ir = _ir(tmp_path)
    _run(ir, tmp_path, lib, {**BASE, "v_in": "12 V DC", "v_out": "DC 5 V"})
    found, unusable = read_inputs(ir)
    assert unusable == {} and found["input_voltage"].traced.value == 12.0 and found["output_voltage"].traced.value == 5.0
    assert found["input_voltage"].requirement.id == "req.v_in" and found["input_voltage"].traced.provenance.note == "parsed from req.v_in: '12 V DC' -> 12 V"
    # a model's value is not the user's: not read, named
    ir = _ir(tmp_path)
    ir.requirements.requirements.append(Requirement(id="req.input_voltage", key="input_voltage", text="12 V", kind=RequirementKind.EXPLICIT, value=llm_generated(12.0, "m", "V")))
    ir.requirements.requirements.append(Requirement(id="req.output_voltage", key="output_voltage", text="5 V", kind=RequirementKind.EXPLICIT, value=user_requirement(5.0, "V")))
    found, unusable = read_inputs(ir)
    assert found["output_voltage"].traced.value == 5.0 and found["output_voltage"].traced.provenance.note == "parsed from req.output_voltage: 5.0 V -> 5 V"
    assert unusable == {"input_voltage": "req.input_voltage: value is llm_generated, not yet the user's (confirm it, or answer input_voltage directly)"}
    # a confirmed extraction's numeric value with a prefixed unit is scaled, not refused
    ir.requirements.requirements[0] = Requirement(id="req.input_voltage", key="input_voltage", text="12 V", kind=RequirementKind.EXPLICIT, value=user_requirement(12000.0, "mV"))
    found, unusable = read_inputs(ir)
    assert unusable == {} and found["input_voltage"].traced.value == 12.0 and found["input_voltage"].traced.unit == "V"


def test_duplicate_keys_with_different_numbers_are_ambiguous(tmp_path: Path):
    ir = _ir(tmp_path)
    ir.requirements.requirements = [
        Requirement(id="req.input_voltage", key="input_voltage", text="12 V", kind=RequirementKind.EXPLICIT, value=user_requirement("12 V")),
        Requirement(id="req.v_in", key="v_in", text="24 V", kind=RequirementKind.EXPLICIT, value=user_requirement("24 V")),
    ]
    found, unusable = read_inputs(ir)
    assert found == {} and unusable == {"input_voltage": "ambiguous: req.input_voltage says 12 V, req.v_in says 24 V"}
    ir.requirements.requirements[1].value = user_requirement(12.0, "V")  # the same number twice is one value
    found, unusable = read_inputs(ir)
    assert unusable == {} and found["input_voltage"].requirement.id == "req.input_voltage"
    ir.requirements.requirements[1].value = user_requirement("12 V max")  # one of them unreadable: the key is unusable, not picked
    found, unusable = read_inputs(ir)
    assert found == {} and "req.v_in: '12 V max' is not one whole quantity" in unusable["input_voltage"]


# --------------------------------------------------------------------------- inputs_vs_requirements


def test_inputs_vs_requirements_pass_fail_and_not_verified(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    _present(ir, tmp_path, lib, DIVIDER)
    _confirm(ir, tmp_path, lib)
    assert check_inputs_vs_requirements(ir).status is S.PASS
    ir.requirements.get("input_voltage").value = user_requirement("24 V")  # the requirement moved after the design was built
    r = check_inputs_vs_requirements(ir)
    assert r.status is S.FAIL and r.details["repair"] == "human" and r.tool == TOOL_ID
    assert r.message == "1 template input(s) no longer equal their requirement: v_in: the design was built from 12 V, req.input_voltage now reads 24 V"
    state, _ = _run(ir, tmp_path, lib, {})
    assert state.outcome(Stage.ARCHITECTURE).status is S.FAIL and ir.validation.latest(INPUTS_CHECK).details["parameters"]["v_in"]["reread"] == 24.0
    ir.requirements.get("input_voltage").value = user_requirement("12 V max")
    r = check_inputs_vs_requirements(ir)
    assert r.status is S.NOT_VERIFIED and "v_in: req.input_voltage: '12 V max' is not one whole quantity" in r.message
    ir.requirements.requirements = [x for x in ir.requirements.requirements if x.key != "input_voltage"]
    r = check_inputs_vs_requirements(ir)
    assert r.status is S.NOT_VERIFIED and r.message == "1 template input(s) could not be re-read: v_in: req.input_voltage is no longer in the IR"


# --------------------------------------------------------------------------- LED


def test_led_template_wires_by_pin_name_and_models_the_led_as_a_confirmed_choice(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "led")
    question = _present(ir, tmp_path, lib, LED)
    assert "led_model: ideal constant-V_f LED: D1 is excluded from the netlist and replaced by the stimulus VLED = v_f" in question
    assert "req.led_forward_current: led_forward_current = 0.01 A (stated as '10 mA')" in question and "r_led = 300 ohm [calc.led.R from v_in, v_f, i_f]" in question
    state, ctx = _confirm(ir, tmp_path, lib, None)
    assert state.outcome(Stage.ARCHITECTURE).message.startswith("14 proposal(s) applied, nothing verified; template led v0.1 confirmed by the user: 14 proposal(s)")
    assert [c.ref for c in ir.components] == ["R1", "D1", "J1"]
    p = ir.parameters
    assert list(p) == ["v_in", "v_f", "i_f", "r_led", "i_led", "tol_rel"]
    assert p["r_led"].value == 300.0 and p["r_led"].provenance.tool == "calc.led.R" and p["r_led"].provenance.inputs == {"v_supply": "v_in", "v_f": "v_f", "i_f": "i_f"}
    assert p["i_led"].value == pytest.approx(0.01) and p["i_led"].provenance.tool == "calc.led.I" and p["i_led"].provenance.inputs == {"v_supply": "v_in", "v_f": "v_f", "r": "r_led"}
    d1 = ir.component("D1")
    assert [(pin.number, pin.name) for pin in d1.pins] == [("1", "K"), ("2", "A")]
    assert ir.net("LED_A").pins[1].model_dump() == {"component_ref": "D1", "pin_number": "2"} and ir.net("GND").pins[1].model_dump() == {"component_ref": "D1", "pin_number": "1"}
    assert d1.spice.exclude and d1.spice.exclude_reason.startswith("ideal constant-V_f LED: D1 is excluded")
    assert d1.spice.provenance.kind is ProvenanceKind.USER_REQUIREMENT and d1.spice.provenance.note.startswith(f"{CHOICE_NOTE_PREFIX}; template led v{TEMPLATE_VERSION}: led_model:")
    vled = ir.simulation.stimulus("VLED")
    assert vled.net == "LED_A" and vled.value.value == 2.0 and vled.provenance.kind is ProvenanceKind.USER_REQUIREMENT and vled.serves_requirements == ["req.led_forward_voltage"]
    exp = ir.simulation.expectations[0]
    assert exp.vector == "i(VLED)" and exp.nominal.provenance.tool == "calc.led.I" and exp.requirement_id == "req.led_forward_current"
    assert "current through the ideal source VLED that replaces D1" in exp.provenance.note
    assert recompute_parameters(ir).message == "5 value(s) recomputed" and _validate(ir, tmp_path, lib)["ir.assumptions"] is S.PASS
    assert _netlist(ir, tmp_path, lib) == "led\nR1 VCC LED_A 300\nVVIN VCC 0 DC 5\nVVLED LED_A 0 DC 2\n.end\n"
    assert state.outcome(Stage.PCB).status is S.PASS and state.outcome(Stage.SCHEMATIC).status is S.PASS
    assert IndependentReviewer(tools=ctx.tools).check_requirements_vs_ir(ir, tmp_path).status is S.PASS


def test_led_missing_inputs_are_required_questions_and_library_gaps_are_notes(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, "led_forward_voltage": "2.0 V"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and out.status is S.USER_INPUT_REQUIRED and [(q.key, q.required) for q in out.questions] == [("input_voltage", True), ("led_forward_current", True)]
    assert out.questions[1].question == 'The LED indicator template needs led_forward_current in A: answer led_forward_current=<value A> (e.g. led_forward_current="10 mA")'
    # supply below the forward voltage: the calculator refuses, the note says so
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **LED, "input_voltage": "1 V"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and "template led not proposed: supply voltage must exceed LED forward voltage (req.input_voltage = 1 V" in out.message
    # no Device:LED in the library -> the note names it; LED pins not named A / K -> refused, never guessed
    no_led = template_library(tmp_path / "no_led", led=False)
    state, _ = _run(_ir(tmp_path), tmp_path, no_led, {**BASE, **LED})
    assert not state.blocked and "template led not proposed: symbol Device:LED not found in the KiCad libraries" in state.outcome(Stage.ARCHITECTURE).message
    unnamed = template_library(tmp_path / "unnamed", led_pin_names=("~", "~"))
    state, _ = _run(_ir(tmp_path), tmp_path, unnamed, {**BASE, **LED})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and "template led not proposed: Device:LED pins are not named A / K in this library (pins: [('1', '~'), ('2', '~')]); the template will not guess the polarity" in out.message


@needs_dll
def test_led_current_is_measured_through_the_ideal_source_and_the_review_says_so(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "led")
    _present(ir, tmp_path, lib, LED)
    state, ctx = _confirm(ir, tmp_path, lib, Stage.SPICE, spice=True)
    r = ir.validation.latest("spice.i_led")
    assert state.outcome(Stage.SPICE).status is S.PASS and r.status is S.PASS and r.details["measured"] == pytest.approx(0.01, rel=1e-9)
    rep = IndependentReviewer(tools=ctx.tools).check_spice_vs_requirements(ir, tmp_path)
    assert rep.status is S.PASS and rep.message.endswith("; verified with D1 excluded: " + ir.component("D1").spice.exclude_reason)
    assert rep.details["excluded_serving"] == [{"ref": "D1", "reason": ir.component("D1").spice.exclude_reason, "requirements": ["req.led_forward_current"]}]
    assert rep.details["quantity_version"] == QUANTITY_VERSION


# --------------------------------------------------------------------------- RC low-pass


def test_rc_template_ties_the_corner_expectation_to_the_users_requirement(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "rc")
    question = _present(ir, tmp_path, lib, RC)
    assert "c = 1e-07 F - shunt capacitor" in question and "ac_points = 100 - points per decade" in question and "ac_variation = 'dec'" in question
    assert "r = 1591.54943092 ohm [calc.rc.r_for_cutoff from f_c, c]" in question and "expectation h_fc: v(OUT) at 1000 Hz = 0.707106781187 +/- 2% verifies req.cutoff_frequency" in question
    ids_before = [r.id for r in ir.requirements.requirements]
    state, ctx = _confirm(ir, tmp_path, lib, None)
    assert state.outcome(Stage.ARCHITECTURE).message.startswith("18 proposal(s) applied, nothing verified; template rc_lowpass v0.1 confirmed by the user: 18 proposal(s)")
    assert [r.id for r in ir.requirements.requirements] == ids_before and ir.requirements.get("cutoff_gain") is None  # no fabricated requirement
    p = ir.parameters
    assert list(p) == ["f_c", "c", "r", "tau", "h_fc", "tol_rel", "ac_fstart", "ac_fstop", "ac_probe"]
    assert p["r"].value == pytest.approx(1591.5494309189535) and p["r"].provenance.tool == "calc.rc.r_for_cutoff"
    assert p["tau"].provenance.tool == "calc.rc.tau" and p["h_fc"].value == pytest.approx(2 ** -0.5) and p["h_fc"].unit is None
    assert (p["ac_fstart"].value, p["ac_fstop"].value) == (10.0, 100000.0) and p["ac_fstart"].provenance.inputs == {"f_c": "f_c"}
    assert p["c"].provenance.kind is ProvenanceKind.USER_REQUIREMENT and p["c"].value == 1e-7
    ac = ir.simulation.analysis("ac")
    assert ac.params["variation"].value == "dec" and ac.params["points"].value == 100 and ac.params["variation"].provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert ac.params["fstart"].provenance.tool == "calc.rc.ac_fstart" and ac.params["fstop"].value == 100000.0
    exp = ir.simulation.expectations[0]
    assert exp.reduce is Reduce.AT and exp.at.value == 1000.0 and exp.at.unit == "Hz" and exp.at.provenance.derived_from == ["req.cutoff_frequency"]
    assert exp.nominal.provenance.tool == "calc.rc.lowpass_magnitude" and exp.requirement_id == "req.cutoff_frequency" and exp.tol_rel.value == 0.02
    assert [c.id for c in ir.constraints] == ["c.rc_lowpass.cutoff"] and "|H(f_c)| = 1/sqrt(2)" in ir.topology.rationale
    assert recompute_parameters(ir).message == "10 value(s) recomputed"
    assert _netlist(ir, tmp_path, lib) == "rc\nC1 OUT 0 1e-7\nR1 IN OUT 1.5915494309189537k\nVVIN IN 0 DC 1 AC 1\n.end\n"
    assert ir.component("R1").value == "1.5915494309189537k"  # the exact calculator value: E-series snapping is not built
    assert state.outcome(Stage.PCB).status is S.PASS and IndependentReviewer(tools=ctx.tools).check_requirements_vs_ir(ir, tmp_path).status is S.PASS


@needs_dll
def test_rc_corner_is_verified_by_ngspice_and_the_reviewer_compares_the_sweep_point(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "rc")
    _present(ir, tmp_path, lib, RC)
    state, ctx = _confirm(ir, tmp_path, lib, Stage.SPICE, spice=True)
    r = ir.validation.latest("spice.h_fc")
    assert state.outcome(Stage.SPICE).status is S.PASS and r.status is S.PASS
    assert r.details["measured"] == pytest.approx(2 ** -0.5, rel=1e-6) and r.details["bracket"]["method"] == "log-log"
    rep = IndependentReviewer(tools=ctx.tools).check_spice_vs_requirements(ir, tmp_path)
    assert rep.status is S.PASS and rep.details["verified"] == {"h_fc": "req.cutoff_frequency"} and rep.details["not_compared"] == []


def test_reviewer_compares_the_sweep_point_for_a_unitless_at_expectation_and_reads_typed_values():
    reviewer = IndependentReviewer()
    at = user_requirement(1000.0, "Hz")
    exp = Expectation(id="h", analysis_id="ac", vector="v(OUT)", reduce=Reduce.AT, at=at, nominal=user_requirement(2 ** -0.5), tol_rel=user_requirement(0.02), requirement_id="req.x", provenance=USER)
    req = Requirement(id="req.x", key="x", text="1 kHz", kind=RequirementKind.EXPLICIT, value=user_requirement("1 kHz"))
    assert reviewer._nominal_vs_requirement(exp, req) == (None, True)
    req.value = user_requirement(1000.0, "Hz")
    assert reviewer._nominal_vs_requirement(exp, req) == (None, True)
    req.value = user_requirement("2 kHz")
    assert reviewer._nominal_vs_requirement(exp, req) == ("sweep point 1000 Hz is not requirement req.x's 2000 Hz (+/- 40 Hz)", True)
    req.value = user_requirement("1 kHz max")
    assert reviewer._nominal_vs_requirement(exp, req) == ("requirement req.x value '1 kHz max' is not one whole quantity with a unit", False)
    # a unit-ful nominal keeps the nominal comparison (the RC fixture's v(OUT) at t = tau against a voltage requirement)
    exp_v = Expectation(id="v", analysis_id="tran", vector="v(OUT)", reduce=Reduce.AT, at=user_requirement(1e-3, "s"), nominal=user_requirement(3.1606, "V"), tol_rel=user_requirement(0.02), requirement_id="req.y", provenance=USER)
    req_v = Requirement(id="req.y", key="y", text="3.16 V", kind=RequirementKind.EXPLICIT, value=user_requirement(3.16, "V"))
    assert reviewer._nominal_vs_requirement(exp_v, req_v) == (None, True)
    req_v.value = user_requirement("3.16 V")
    assert reviewer._nominal_vs_requirement(exp_v, req_v) == (None, True)
    req_v.value = user_requirement("4 V")
    assert reviewer._nominal_vs_requirement(exp_v, req_v) == ("nominal 3.1606 V is not requirement req.y's 4 V (+/- 0.08 V)", True)
    # typed string values on a plain op expectation: whole-text rule
    exp_op = Expectation(id="o", analysis_id="op", vector="v(VOUT)", nominal=user_requirement(5.0, "V"), tol_rel=user_requirement(0.01), requirement_id="req.z", provenance=USER)
    req_z = Requirement(id="req.z", key="z", text="5 V", kind=RequirementKind.EXPLICIT, value=user_requirement("5 V"))
    assert reviewer._nominal_vs_requirement(exp_op, req_z) == (None, True)
    for text in ("5 V typ", "12 V max", "3.3~5V", "5V 2A"):
        req_z.value = user_requirement(text)
        assert reviewer._nominal_vs_requirement(exp_op, req_z) == (f"requirement req.z value {text!r} is not one whole quantity with a unit", False)
    req_z.value = user_requirement("5 mV")
    assert reviewer._nominal_vs_requirement(exp_op, req_z) == ("nominal 5 V is not requirement req.z's 0.005 V (+/- 5e-05 V)", True)


def test_the_design_package_never_returns_a_model_or_assumption_value(tmp_path: Path):
    """Every Traced the templates emit is the user's, a calculator's, or (pins) the library's - after confirmation nothing needs verification."""
    lib = template_library(tmp_path / "kicad")
    for name, answers in (("divider", DIVIDER), ("led", LED), ("rc", RC)):
        ir = _ir(tmp_path / name, name)
        _present(ir, tmp_path / name, lib, answers)
        _confirm(ir, tmp_path / name, lib)
        kinds = {t.provenance.kind for t in ir.parameters.values()}
        assert kinds <= {ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.DERIVED}, name
        dumped = ir.model_dump(mode="json")
        text = str(dumped["components"]) + str(dumped["nets"]) + str(dumped["simulation"]) + str(dumped["topology"]) + str(dumped["constraints"])
        assert "llm_generated" not in text and "assumption" not in text, name
        assert copy.deepcopy(ir).content_hash() == ir.content_hash()


# --------------------------------------------------------------------------- a confirmation counts only for the table the user saw


DIVIDER_CANNED = {
    "requirements": [
        _req("input_voltage", "Input voltage is 12 V", "12 V input", 12, "V", "12 V"),
        _req("output_voltage", "Output voltage is 5 V", "5 V output", 5, "V", "5 V"),
    ],
    "questions": [], "conflicts": [], "assumptions": [],
    "application": {"summary": "bench reference", "quote": "bench reference"},
    "jurisdictions": [{"code": "EU", "quote": "EU"}],
}


def _llm_run(ir: CircuitIR, svc, tmp_path: Path, lib: KicadLibrary, answers: dict[str, str]):
    return Orchestrator(AgentContext(workdir=tmp_path, llm=svc, tools={"kicad_library": lib}, answers=answers)).run(ir)


def test_confirm_design_beside_confirm_requirements_in_one_run_asks_the_table_and_applies_nothing(tmp_path: Path):
    """The run that confirms the extraction is the first in which the inputs are usable: no table was shown before it."""
    lib = template_library(tmp_path / "kicad")
    svc, client = _service([{"structured": DIVIDER_CANNED, "usage": USAGE}])
    ir = _ir(tmp_path, "llm")
    ir.requirements.raw_input = "12 V input, 5 V output, bench reference, EU"
    state = _llm_run(ir, svc, tmp_path, lib, {})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_KEY] and ir.requirements.presented == {}
    state = _llm_run(ir, svc, tmp_path, lib, {CONFIRM_KEY: "yes", CONFIRM_DESIGN_KEY: "yes"})
    assert ir.requirements.get("input_voltage").value.provenance.kind is ProvenanceKind.USER_REQUIREMENT  # confirmed in this run
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and state.current is Stage.ARCHITECTURE and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert ir.components == [] and ir.nets == [] and ir.parameters == {} and ir.topology is None
    assert f"{CONFIRM_DESIGN_KEY} ignored: answer(s) ['{CONFIRM_KEY}'] were given in this run and can change the inputs a template reads" in out.message
    assert ir.requirements.presented[CONFIRM_DESIGN_KEY] == table_hash(state.open_questions[0].question) and len(client.calls) == 1
    # any non-control answer beside the confirmation is the same case: the table this run shows is not the one on screen
    state = _llm_run(ir, svc, tmp_path, lib, {"application": "bench reference", CONFIRM_DESIGN_KEY: "yes"})
    assert state.blocked and ir.components == [] and "answer(s) ['application'] were given in this run" in state.outcome(Stage.ARCHITECTURE).message
    # a decision on an inferred item is one too; a control key of another agent is not
    state = _llm_run(ir, svc, tmp_path, lib, {ACCEPT_KEY: "nothing", CONFIRM_DESIGN_KEY: "yes"})
    assert state.blocked and ir.components == [] and f"answer(s) ['{ACCEPT_KEY}'] were given in this run" in state.outcome(Stage.ARCHITECTURE).message
    assert len(client.calls) == 1  # the extraction was never re-run: every run above hit the cache
    state = _llm_run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes", PLACEMENT_KEY: "skip"})
    assert not state.blocked and [c.ref for c in ir.components] == ["R1", "R2", "J1"] and ir.pcb is None


def test_confirmation_is_content_based_a_requirement_edited_between_the_runs_re_asks(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    question = _present(ir, tmp_path, lib, DIVIDER)
    saved = ir.save(tmp_path / "ir.json")
    ir = CircuitIR.load(saved)  # the record survives the strict loader
    assert ir.requirements.presented[CONFIRM_DESIGN_KEY] == table_hash(question)
    ir.requirements.get("output_voltage").value = user_requirement("6 V")  # a hand edit, no --answer: the given-now guard cannot see it
    state, _ = _confirm(ir, tmp_path, lib)
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and ir.components == [] and [q.key for q in out.questions] == [CONFIRM_DESIGN_KEY, "output_current"]
    assert f"{CONFIRM_DESIGN_KEY} ignored: the table below (" in out.message and "is not the one you confirmed - the table shown to you before was sha256:" in out.message
    shown = out.questions[0].question
    assert "req.output_voltage: output_voltage = 6 V (stated as '6 V')" in shown and ir.requirements.presented[CONFIRM_DESIGN_KEY] == table_hash(shown) != table_hash(question)
    state, _ = _confirm(ir, tmp_path, lib)  # the 6 V table was shown by the previous run: this confirmation counts
    assert not state.blocked and len(ir.components) == 3 and ir.parameters["v_out_target"].value == 6.0
    # no record at all (an IR whose bookkeeping was dropped) is not a shown table either
    ir = _ir(tmp_path, "norec")
    _present(ir, tmp_path, lib, DIVIDER)
    ir.requirements.presented.clear()
    state, _ = _confirm(ir, tmp_path, lib)
    assert state.blocked and ir.components == [] and "no table was recorded as shown" in state.outcome(Stage.ARCHITECTURE).message
    state, _ = _confirm(ir, tmp_path, lib)
    assert not state.blocked and len(ir.components) == 3


# --------------------------------------------------------------------------- the inputs check is an agent result of its run, never stale within it


def test_inputs_check_is_unstamped_so_a_later_stage_of_the_same_run_cannot_make_it_stale(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    _present(ir, tmp_path, lib, DIVIDER)
    _run(ir, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes", PLACEMENT_KEY: "skip"}, None)
    assert len(ir.components) == 3 and ir.pcb is None
    state, _ = _run(ir, tmp_path, lib, {}, None)  # PLACEMENT now applies a proposal after ARCHITECTURE produced the check
    check = ir.validation.latest(INPUTS_CHECK)
    assert check.status is S.PASS and check.ir_hash is None and ir.pcb is not None
    release = state.outcomes[-1]
    assert release.stage is Stage.RELEASE and "another IR version" not in release.message and INPUTS_CHECK not in release.message


# --------------------------------------------------------------------------- the divider's load question: with the table, and after the build


def test_load_question_is_asked_with_the_table_and_a_zero_answer_before_the_build_is_served(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER})
    assert [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    load = [q for q in state.optional_questions if q.key == "output_current"]
    assert len(load) == 1 and load[0].question.startswith("No load current was stated: does VOUT need to supply one?. Answer output_current=0 A if VOUT drives no load")
    state, _ = _run(ir, tmp_path, lib, {"output_current": "0 A", CONFIRM_DESIGN_KEY: "yes"})  # answered as instructed: a new input, so the table is re-shown
    out = state.outcome(Stage.ARCHITECTURE)
    assert state.blocked and ir.components == [] and "answer(s) ['output_current'] were given in this run" in out.message
    assert "req.output_current: output_current = 0 A (stated as '0 A')" in out.questions[0].question and [q.key for q in out.questions] == [CONFIRM_DESIGN_KEY]
    state, ctx = _confirm(ir, tmp_path, lib, None)
    assert not state.blocked and ir.net("VOUT").serves_requirements == ["req.output_voltage", "req.output_current"]
    r = IndependentReviewer(tools=ctx.tools).check_requirements_vs_ir(ir, tmp_path)
    assert r.status is S.PASS and "req.output_current" in r.details["traced"]


def test_a_zero_load_stated_after_the_build_is_served_by_a_proposal_and_a_load_is_named(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    _present(ir, tmp_path, lib, DIVIDER)
    _confirm(ir, tmp_path, lib)
    assert late_load_changes(ir).changes == [] and ir.net("VOUT").serves_requirements == ["req.output_voltage"]
    state, ctx = _run(ir, tmp_path, lib, {"output_current": "0 A"}, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert out.status is S.PASS and "req.output_current = 0 A stated after the build: VOUT is a high-impedance reference and is proposed to serve it" in out.message
    assert "no component or net serves" not in out.message and ir.net("VOUT").serves_requirements == ["req.output_voltage", "req.output_current"]
    assert ir.net("VOUT").provenance.tool == f"{TOOL_ID}.divider" and ir.validation.latest(INPUTS_CHECK).status is S.PASS
    report = IndependentReviewer(tools=ctx.tools).review(ir, tmp_path)
    assert {x.check_id: x for x in report.results}[ReviewArea.REQUIREMENTS_VS_IR].status is S.PASS
    assert ReviewArea.REQUIREMENTS_VS_IR not in state.outcomes[-1].message
    state, _ = _run(ir, tmp_path, lib, {})  # idempotent: nothing left to serve
    assert "proposed to serve" not in state.outcome(Stage.ARCHITECTURE).message and ir.net("VOUT").serves_requirements == ["req.output_voltage", "req.output_current"]
    # a load stated after the build: named, not served, and the reviewer still FAILs it (honest)
    ir = _ir(tmp_path, "load")
    _present(ir, tmp_path, lib, DIVIDER)
    _confirm(ir, tmp_path, lib)
    state, ctx = _run(ir, tmp_path, lib, {"output_current": "2 A"}, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert "req.output_current = 2 A: a resistive divider cannot supply a load; VOUT does not serve it" in out.message
    assert "no component or net serves: req.output_current" in out.message and ir.net("VOUT").serves_requirements == ["req.output_voltage"]
    r = {x.check_id: x for x in IndependentReviewer(tools=ctx.tools).review(ir, tmp_path).results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.FAIL and r.details["unserved"] == ["req.output_current"]
    # a hand-made design (VOUT not the divider's) is left alone
    ir.net("VOUT").provenance = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="mine")
    ir.requirements.get("output_current").value = user_requirement("0 A")
    assert late_load_changes(ir).changes == [] and late_load_changes(ir).notes == []


def test_a_typed_answer_dropped_for_a_key_that_holds_a_typed_value_is_noted(tmp_path: Path):
    """The refusal names req.output_current and says to change it; answering the key again does nothing, and a note says so."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, "output_current": "2 A"})
    [q] = [q for q in state.optional_questions if q.key == "output_current"]
    assert "Change the requirement req.output_current in the IR" in q.question and "--answer" not in q.question and "Answer output_current=" not in q.question
    state, _ = _run(ir, tmp_path, lib, {"output_current": "0 A"})
    assert ir.requirements.get("output_current").value.value == "2 A" and [r.key for r in ir.requirements.requirements].count("output_current") == 1
    msg = state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    assert "output_current: answer '0 A' not applied - req.output_current already holds your earlier answer '2 A', which is kept; to change it edit that requirement in the IR" in msg
    assert "template divider not proposed: the request needs 5 V at 2 A from 12 V (req.output_current)" in state.outcome(Stage.ARCHITECTURE).message
    # an unreadable stated load names its requirement id too
    ir = _ir(tmp_path, "unreadable")
    state, _ = _run(ir, tmp_path, lib, {**BASE, **DIVIDER, "output_current": "2 A max"})
    [q] = [q for q in state.optional_questions if q.key == "output_current"]
    assert q.question.startswith("Output_current is stated but not readable (req.output_current: '2 A max' is not one whole quantity") and "Change the requirement req.output_current in the IR" in q.question


# --------------------------------------------------------------------------- non-finite numbers: a refusal note, never a traceback


def test_non_finite_typed_values_are_unusable_not_a_crash(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    req = Requirement(id="req.input_voltage", key="input_voltage", text="x", kind=RequirementKind.EXPLICIT, value=user_requirement("1e309 V"))
    assert read_value(req, "V") == (None, "req.input_voltage: '1e309 V' is not a finite number")
    req = Requirement(id="req.cutoff_frequency", key="cutoff_frequency", text="x", kind=RequirementKind.EXPLICIT, value=user_requirement(1e300, "GHz"))
    assert read_value(req, "Hz") == (None, "req.cutoff_frequency: 1e+300 GHz is not a finite number in Hz")
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, "input_voltage": "1e309 V", "output_voltage": "5 V"}, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and "input_voltage not usable: req.input_voltage: '1e309 V' is not a finite number" in out.message
    assert state.outcomes[-1].stage is Stage.RELEASE
    ir = _ir(tmp_path, "rc")
    ir.requirements.requirements.append(Requirement(id="req.cutoff_frequency", key="cutoff_frequency", text="x", kind=RequirementKind.EXPLICIT, value=user_requirement(1e300, "GHz")))
    state, _ = _run(ir, tmp_path, lib, {**BASE})
    assert ir.components == [] and "cutoff_frequency not usable: req.cutoff_frequency: 1e+300 GHz is not a finite number in Hz" in state.outcome(Stage.ARCHITECTURE).message


def test_a_calculator_that_overflows_refuses_the_template_with_a_sentence(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    with pytest.raises(ValueError, match=r"^calc\.led\.R overflows: R = \(V_supply - V_f\) / I_f is not a finite number for these inputs$"):
        led_series_resistor(user_requirement(5.0, "V"), user_requirement(2.0, "V"), user_requirement(1e-320, "A"))
    ir = _ir(tmp_path)
    state, _ = _run(ir, tmp_path, lib, {**BASE, **LED, "led_forward_current": "1e-320 A"}, None)
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and state.outcomes[-1].stage is Stage.RELEASE
    assert "template led not proposed: calc.led.R overflows: R = (V_supply - V_f) / I_f is not a finite number for these inputs (req.input_voltage = 5 V" in out.message
    assert "validation error" not in out.message and "Traced" not in out.message
    # the divider's and the RC's calculators are guarded the same way
    ir = _ir(tmp_path, "div")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "input_voltage": "1e308 V", "output_voltage": "1e-308 V"})
    assert ir.components == [] and "template divider not proposed: calc.divider.r1_for_v_out overflows" in state.outcome(Stage.ARCHITECTURE).message
    ir = _ir(tmp_path, "rc")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "cutoff_frequency": "1e-310 Hz"})
    assert ir.components == [] and "template rc_lowpass not proposed: calc.rc.r_for_cutoff overflows: R = 1 / (2 pi f_c C) is not a finite number" in state.outcome(Stage.ARCHITECTURE).message
    ir = _ir(tmp_path, "rc0")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "cutoff_frequency": "1e-320 Hz"})  # the product underflows to zero: a sentence, not a ZeroDivisionError
    assert ir.components == [] and "template rc_lowpass not proposed: calc.rc.r_for_cutoff underflows: 2 pi f_c C is zero" in state.outcome(Stage.ARCHITECTURE).message


def test_inputs_check_reports_a_non_finite_edited_requirement_as_not_verified(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path)
    _present(ir, tmp_path, lib, DIVIDER)
    _confirm(ir, tmp_path, lib)
    ir.requirements.get("input_voltage").value = user_requirement("1e309 V")
    r = check_inputs_vs_requirements(ir)
    assert r.status is S.NOT_VERIFIED and r.message == "1 template input(s) could not be re-read: v_in: req.input_voltage: '1e309 V' is not a finite number"
    assert r.details["parameters"]["v_in"]["status"] == S.NOT_VERIFIED.value and "reread" not in r.details["parameters"]["v_in"]
    state, _ = _run(ir, tmp_path, lib, {}, None)
    assert state.outcome(Stage.ARCHITECTURE).status is S.NOT_VERIFIED and state.outcomes[-1].stage is Stage.RELEASE
