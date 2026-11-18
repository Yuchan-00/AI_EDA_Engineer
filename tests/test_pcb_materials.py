"""Board base materials (``ai_eda.design.materials``): the sourced material stacks, the IR fields, the selection rule, the templates and the IMS board.

What is proven here:

* Every material stack takes only thicknesses its data sheet offers and passes
  the generic stacks' rule checks where they apply (planes: ``stack_problems``;
  a through via: ``via_drill_problems`` at the router's 0.4 / 0.3 mm drills); an
  off-sheet thickness is named.
* The IR: a stack without a material dumps without the new fields (every FR-4
  stack byte for byte), a material stack round-trips and is hashed, an IMS stack
  has exactly one copper layer on its metal base and no through via, every new
  number has an id ``lookup`` resolves.
* The rule (deterministic, closed): nothing stated is untouched FR-4, a carrier
  at or above 500 MHz proposes RO4350B (447.5625 MHz does not), static flexing
  proposes polyimide, dynamic flexing refuses, a power dissipation asks the
  required question ``pcb_material``, the user's answer wins and a contradiction
  refuses; unreadable values refuse with the reason.
* The templates: the material keys never refuse a template (board keys), a
  stated material builds its stack with its rows, a proposal is a table row the
  user confirms, a question blocks the plan, layer counts are checked against
  the material and the template's policy; the stack serves the requirements.
* The IMS board: the PCB agent keeps ``B.Cu`` out over the outline, the router
  then routes ``F.Cu`` only with no via, ``pcb.keepout`` fails a through-hole
  pad on ``B.Cu``, and the SI measure, rules and report read the one-layer
  stack without a through via.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.circuit import _unserved
from ai_eda.design import TEMPLATES, unserved_requirements
from ai_eda.design.base import LayerPolicy
from ai_eda.design.inputs import LayerCountInput, canonical_key
from ai_eda.design.materials import (
    FLEX_STATIC_BEND_RATIO,
    HF_THRESHOLD_HZ,
    MATERIAL_IDS,
    MATERIAL_STACKS,
    MATERIALS,
    SINGLE_LAYER_KEEPOUT_ID,
    MaterialDecision,
    MaterialDielectric,
    MaterialStack,
    material_layer_count,
    material_stack,
    material_stack_problems,
    material_stackup,
    select_material,
    single_layer_keepout,
)
from ai_eda.design.stackup import board_layers, generic_stackup, via_drill_problems, with_planes
from ai_eda.ir import (
    BoardOutline,
    CircuitIR,
    DielectricKind,
    MetalBase,
    ProjectMeta,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    StackMaterial,
    Stackup,
    StackupCopper,
    StackupDielectric,
    llm_generated,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.report.si_report import stackup_lines
from ai_eda.tools.calc.tline import NO_REFERENCE_PLANE, line_geometry
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.tools.routing.maze import ROUTER_KEEPOUT_VERSION
from ai_eda.tools.si.measure import measure_nets, via_model
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import KEEPOUT_CHECK
from ai_eda.workflow import Orchestrator, Stage
from tests.test_routing import board_ir, fixture_library

_lib = KicadLibrary()
HAS_LIBS = _lib.symbol_file("Device") is not None and _lib.symbol_file("Connector_Generic") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
BASE = {"application": "indicator", "jurisdiction": "EU"}
LED = {"input_voltage": "5 V", "led_forward_voltage": "2 V", "led_forward_current": "10 mA"}


def _req(key: str, value, prov: str = "user") -> Requirement:
    traced = user_requirement(value) if prov == "user" else llm_generated(value, model="m")
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value}", kind=RequirementKind.EXPLICIT, value=traced, category="electrical")


def _ir(*reqs: Requirement) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="m", name="m"))
    ir.requirements.requirements = list(reqs)
    return ir


# --------------------------------------------------------------------------- the material stacks


def test_every_material_stack_takes_offered_thicknesses_and_passes_the_shared_rules():
    assert MATERIAL_IDS == ("fr4", "rogers_ro4350b", "polyimide_flex", "aluminium_ims")
    assert sorted(MATERIAL_STACKS) == [("aluminium_ims", 1), ("polyimide_flex", 2), ("rogers_ro4350b", 2), ("rogers_ro4350b", 4)]
    for ms in MATERIAL_STACKS.values():
        assert material_stack_problems(ms) == [] and material_stack_problems(ms, 0.3) == [], ms.title
        assert ms.material in MATERIALS and all(d.source for d in ms.dielectrics) and ms.copper_source
    assert material_stack("rogers_ro4350b", 2).thickness_mm == pytest.approx(1.59)
    assert material_stack("rogers_ro4350b", 4).thickness_mm == pytest.approx(1.262)
    assert material_stack("polyimide_flex", 2).thickness_mm == pytest.approx(0.12)
    assert material_stack("aluminium_ims", 1).thickness_mm == pytest.approx(1.625)
    four = material_stack("rogers_ro4350b", 4)
    assert [d.product for d in four.dielectrics] == ["Rogers RO4350B", "Rogers RO4450F", "Rogers RO4350B"] and four.planes
    assert [(d.er, d.er_hz, d.loss_tangent) for d in four.dielectrics] == [(3.66, None, 0.0037), (3.52, 10e9, 0.004), (3.66, None, 0.0037)]
    with pytest.raises(ValueError, match="no generic polyimide_flex stack with 4"):
        material_stack("polyimide_flex", 4)
    # a thickness the data sheet does not offer is named; so is a through via too long for the drill
    bad = MaterialStack(material="rogers_ro4350b", title="bad", copper=(("F.Cu", 35.0), ("B.Cu", 70.0)), copper_source="x",
                        dielectrics=(MaterialDielectric(kind=DielectricKind.CORE, thickness_mm=1.0, er=3.66, er_hz=None, er_note="", loss_tangent=0.0037,
                                                        df_hz=10e9, product="Rogers RO4350B", source="x"),))
    problems = material_stack_problems(bad)
    assert any("1 mm is not a thickness its data sheet offers" in p for p in problems) and any("B.Cu 70 um is not a copper thickness" in p for p in problems)
    assert via_drill_problems(material_stack("rogers_ro4350b", 2).generic(), 0.15) != []


def test_every_number_names_its_document():
    srcs = {m.id: m.sources for m in MATERIALS.values()}
    assert "PUB# 92-004" in srcs["rogers_ro4350b"][0] and "Publication #92-150" in srcs["rogers_ro4350b"][1]
    assert "EI-10124 (2/25)" in srcs["polyimide_flex"][0] and "Version 06.2025" in srcs["polyimide_flex"][1]
    assert "DS_21 Rev 01" in srcs["aluminium_ims"][0]
    s = material_stackup(material_stack("aluminium_ims", 1), MaterialDecision(material="aluminium_ims", reason="r"), "t", confirmed=False)
    for key, t in s.traced_items():
        assert t.provenance.kind is ProvenanceKind.ASSUMPTION and "DS_21 Rev 01" in t.provenance.note, key


# --------------------------------------------------------------------------- the IR fields


def test_a_stack_without_a_material_dumps_without_the_new_fields():
    four = generic_stackup(4, "t", confirmed=True)
    dumped = four.model_dump(mode="json")
    assert sorted(dumped) == ["copper", "dielectrics", "provenance", "solder_mask"]
    assert all(sorted(d) == ["er", "er_frequency_hz", "kind", "loss_tangent", "thickness_mm"] for d in dumped["dielectrics"])
    assert "material" not in four.model_dump_json() and four.through_via_span_mm() == pytest.approx(1.6) == four.total_thickness_mm()
    assert with_planes(four, {}) == four


def test_a_material_stack_round_trips_is_hashed_and_resolves_every_id(tmp_path: Path):
    hf = material_stackup(material_stack("rogers_ro4350b", 4), MaterialDecision(material="rogers_ro4350b", reason="r"), "t", confirmed=True,
                          ground_net="GND", power_net="+3V3")
    assert hf.material.id == "rogers_ro4350b" and hf.material.kind == "rigid" and [c.name for c in hf.plane_layers()] == ["In1.Cu", "In2.Cu"]
    assert [d.material for d in hf.dielectrics] == ["Rogers RO4350B", "Rogers RO4450F", "Rogers RO4350B"]
    assert hf.dielectrics[0].er_frequency_hz is None and float(hf.dielectrics[1].er_frequency_hz.value) == 10e9
    items = dict(hf.traced_items())
    assert "pcb.stackup.dielectrics[1].thermal_conductivity_w_mk" in items and all(hf.lookup(k) is t for k, t in items.items())
    ir = CircuitIR(project=ProjectMeta(id="h", name="h"))
    from ai_eda.ir import PCBDesign

    ir.pcb = PCBDesign(layers=board_layers(hf), stackup=generic_stackup(4, "t", confirmed=True, power_net="+3V3"))
    plain = ir.content_hash()
    ir.pcb.stackup = hf
    assert ir.content_hash() != plain and ir.design_dict()["pcb"]["stackup"]["material"]["id"] == "rogers_ro4350b"
    ir.save(tmp_path / "ir.json")
    back = CircuitIR.load(tmp_path / "ir.json")
    assert back.pcb.stackup == hf and back.content_hash() == ir.content_hash()
    # with_planes keeps the material
    assert with_planes(hf, {}).material == hf.material
    # the 4-layer microstrip reads the RO4350B design Dk
    geom, _ = line_geometry(hf, "F.Cu")
    assert float(geom.er.value) == 3.66 and geom.reference_net == "GND" and any("is not recorded" in n for n in geom.notes)


def test_an_ims_stack_has_one_copper_layer_on_its_base_and_no_through_via():
    ims = material_stackup(material_stack("aluminium_ims", 1), MaterialDecision(material="aluminium_ims", reason="r"), "t", confirmed=True)
    assert ims.layer_count == 1 and ims.metal_base is not None and ims.through_via_span_mm() is None
    assert ims.board_thickness_mm() == pytest.approx(0.125) and ims.total_thickness_mm() == pytest.approx(1.625)
    assert [(layer.name, layer.kind) for layer in board_layers(ims)] == [("F.Cu", "signal"), ("B.Cu", "signal")]
    assert float(ims.dielectrics[0].thermal_conductivity_w_mk.value) == 3.0 and ims.lookup("pcb.stackup.metal_base.thickness_mm").value == 1.5
    geom, why = line_geometry(ims, "F.Cu")
    assert geom is None and why.startswith(NO_REFERENCE_PLANE) and "only copper layer" in why
    assert via_model(ims) is None
    base, d = ims.metal_base, ims.dielectrics[0]
    cu = StackupCopper(name="F.Cu", thickness_um=user_requirement(35.0, "um"))
    mat = ims.material
    for kwargs, match in (
        (dict(copper=[cu, StackupCopper(name="B.Cu", thickness_um=user_requirement(35.0, "um"))], dielectrics=[d], metal_base=base, material=mat), "exactly one copper layer"),
        (dict(copper=[cu], dielectrics=[d, d], metal_base=base, material=mat), "one dielectric"),
        (dict(copper=[cu], dielectrics=[d], metal_base=MetalBase(material="Al", thickness_mm=user_requirement(1.5, "um")), material=mat), "unit 'mm'"),
        (dict(copper=[cu], dielectrics=[d], metal_base=None, material=mat), "at least two copper layers"),
        (dict(copper=[cu], dielectrics=[d], metal_base=base, material=StackMaterial(id="x", name="x", kind="rigid", reason="r")), "only an insulated metal substrate"),
        (dict(copper=[cu], dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=user_requirement(0.09, "mm"), er=user_requirement(5.2),
                                                          thermal_conductivity_w_mk=user_requirement(3.0, "W/m/K"))], metal_base=base, material=mat), "unit 'W/mK'"),
    ):
        with pytest.raises(ValidationError, match=match):
            Stackup(**kwargs)


def test_the_si_report_and_rules_read_a_one_layer_stack():
    ims = material_stackup(material_stack("aluminium_ims", 1), MaterialDecision(material="aluminium_ims", reason="chosen"), "t", confirmed=True)
    text = "\n".join(stackup_lines(ims))
    assert "(F.Cu / 금속 베이스)" in text and "aluminium 5052 / 6061" in text and "1.625 mm" in text and "Aismalibar COBRITHERM HTC 3,2W (ims)" in text


# --------------------------------------------------------------------------- the selection rule


def test_nothing_stated_is_untouched_fr4_and_a_carrier_below_500_mhz_is_not_read():
    assert select_material(_ir()).untouched
    assert select_material(_ir(_req("carrier_frequency", "447.5625 MHz"))).untouched
    assert HF_THRESHOLD_HZ == 500e6


def test_a_carrier_at_or_above_500_mhz_proposes_the_high_frequency_laminate():
    for value in ("500 MHz", "2.4 GHz"):
        d = select_material(_ir(_req("carrier_frequency", value)))
        assert d.material == "rogers_ro4350b" and d.proposed and [r.id for r in d.read] == ["req.carrier_frequency"], value
        assert "PUB# 92-004 (c) 2022" in d.reason and "500 MHz" in d.reason and "p. 1" in d.reason
    assert select_material(_ir(_req("carrier_frequency", "2.4 GHz", prov="llm"))).untouched  # an unconfirmed carrier is nobody's requirement


def test_flexing_proposes_polyimide_refuses_dynamic_and_none_keeps_fr4():
    d = select_material(_ir(_req("pcb_flexing", "Static")))
    assert d.material == "polyimide_flex" and d.proposed and "IPC-2223 Use A" in d.reason and FLEX_STATIC_BEND_RATIO == 10.0
    d = select_material(_ir(_req("pcb_flexing", "dynamic")))
    assert d.refusal is not None and "Use B" in d.refusal and "pcb_flexing=static" in d.refusal
    d = select_material(_ir(_req("board_flexing", "none")))
    assert d.material == "fr4" and not d.untouched and not d.proposed and [r.key for r in d.read] == ["board_flexing"]
    d = select_material(_ir(_req("pcb_flexing", "a little")))
    assert d.refusal is not None and "is not exactly one of static, dynamic, none" in d.refusal


def test_a_power_dissipation_asks_instead_of_deciding():
    d = select_material(_ir(_req("power_dissipation", "3 W")))
    q = d.question
    assert q is not None and q.key == "pcb_material" and q.required and q.options == list(MATERIAL_IDS) and d.material == "fr4"
    assert "aluminium_ims" in q.question and "3.00 W/mK" in q.question and "--answer leave_out=power_dissipation" in q.question
    assert "no document read in this version says at which dissipation FR-4 stops being adequate" in q.rationale
    d = select_material(_ir(_req("heat_dissipation", "0 W")))
    assert d.question is None and d.material == "fr4" and not d.untouched
    for value, why in (("-1 W", "below 0 W"), ("3 W max", "not one whole quantity")):
        assert why in select_material(_ir(_req("power_dissipation", value))).refusal
    # two needs that propose different materials ask too
    d = select_material(_ir(_req("pcb_flexing", "static"), _req("carrier_frequency", "2.4 GHz")))
    assert d.question is not None and "propose different materials" in d.question.rationale


def test_the_users_answer_wins_and_a_contradiction_refuses():
    for value, expect in (("ROGERS-RO4350B", "rogers_ro4350b"), (" aluminium_ims ", "aluminium_ims"), ("fr4", "fr4")):
        d = select_material(_ir(_req("pcb_material", value)))
        assert d.material == expect and d.stated.id == "req.pcb_material" and not d.proposed and not d.untouched
    d = select_material(_ir(_req("pcb_material", "fr4"), _req("carrier_frequency", "2.4 GHz")))
    assert d.material == "fr4" and any("it would propose rogers_ro4350b" in n for n in d.notes) and [r.key for r in d.read] == ["pcb_material"]
    d = select_material(_ir(_req("pcb_material", "aluminium_ims"), _req("power_dissipation", "3 W")))
    assert d.material == "aluminium_ims" and d.question is None and [r.key for r in d.read] == ["pcb_material", "power_dissipation"]
    d = select_material(_ir(_req("pcb_material", "fr4"), _req("pcb_flexing", "static")))
    assert d.refusal is not None and "a rigid board" in d.refusal and "pcb_material=polyimide_flex" in d.refusal
    assert select_material(_ir(_req("pcb_material", "polyimide_flex"), _req("pcb_flexing", "static"))).material == "polyimide_flex"
    for reqs, why in (([_req("pcb_material", "rogers")], "is not exactly one of fr4, rogers_ro4350b"),
                      ([_req("pcb_material", "fr4"), _req("board_material", "aluminium_ims")], "ambiguous")):
        d = select_material(_ir(*reqs))
        assert d.refusal is not None and why in d.refusal, d.refusal
    # a model's unconfirmed value is nobody's requirement yet: not stated (the closed world never counted it either)
    assert select_material(_ir(_req("pcb_material", "aluminium_ims", prov="llm"))).untouched


def test_the_layer_count_follows_the_material_and_the_template_policy():
    generic, four_only = LayerPolicy(), LayerPolicy(allowed=(4,), default=4, reason="the RF lines need a reference plane")
    stated4 = LayerCountInput(value=4, requirement=_req("pcb_layers", "4"))
    flex, ims, hf = (MaterialDecision(material=m) for m in ("polyimide_flex", "aluminium_ims", "rogers_ro4350b"))
    assert material_layer_count(flex, LayerCountInput(value=2), generic) == (2, None)
    assert "two copper layers" in material_layer_count(flex, stated4, generic)[1]
    assert "the RF lines need a reference plane" in material_layer_count(flex, LayerCountInput(value=4), four_only)[1]
    assert material_layer_count(ims, LayerCountInput(value=2), generic) == (1, None)
    assert "one copper layer" in material_layer_count(ims, stated4, generic)[1]
    assert material_layer_count(hf, stated4, four_only) == (4, None) and material_layer_count(hf, LayerCountInput(value=2), generic) == (2, None)


def test_the_material_keys_are_board_keys_no_template_refuses():
    assert {canonical_key(k) for k in ("pcb_material", "board_material", "pcb_flexing", "board_flexing", "power_dissipation", "heat_dissipation")} == {
        "pcb_material", "pcb_flexing", "power_dissipation"}
    ir = _ir(_req("pcb_material", "fr4"), _req("pcb_flexing", "none"), _req("power_dissipation", "1 W"))
    for template in TEMPLATES:
        assert unserved_requirements(ir, template) == []


def _item(key: str, value, category: str, *, prov: str = "user", kind: RequirementKind = RequirementKind.EXPLICIT) -> Requirement:
    traced = user_requirement(value) if prov == "user" else llm_generated(value, model="m")
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value}", kind=kind, value=traced, category=category)


#: requirements c270797's closed world (``unserved_requirements``) never counted: a model's unconfirmed value, a category outside the design
#: categories - so no template refused over them, and the material rule reads none of them either
IGNORED_BEFORE = (
    _item("power_dissipation", "0.1 W", "thermal", prov="llm", kind=RequirementKind.IMPLICIT),  # an implicit item a confirmation leaves the model's
    _item("board_material", "FR-4", "other"),
    _item("board_material", "FR-4", "manufacturing"),
    _item("pcb_flexing", "none", "mechanical", prov="llm"),
    _item("pcb_material", "fr4", "electrical", prov="llm"),
    _item("heat_dissipation", "lots", "environmental"),
)


def test_the_material_rule_reads_only_requirements_the_closed_world_counted_before(tmp_path: Path):
    """Review finding: every IR that built at c270797 builds exactly as then - an unconfirmed value or a non-design category is not stated."""
    for r in IGNORED_BEFORE:
        assert select_material(_ir(r)) == MaterialDecision(), r
    from ai_eda.agents.requirement import _answer_requirement
    from ai_eda.design.templates import design_from_requirements
    from tests.test_circuit_templates import template_library

    lib = template_library(tmp_path / "kicad")
    divider = [_answer_requirement(k, v) for k, v in {"input_voltage": "12 V", "output_voltage": "5 V"}.items()]
    plain = design_from_requirements(_ir(*divider), lib)
    assert plain is not None and plain.buildable
    for r in IGNORED_BEFORE:
        plan = design_from_requirements(_ir(*divider, r), lib)
        assert plan is not None and plan.buildable, (r.id, plan.notes)
        assert [design_data(c.payload) for c in plan.changes] == [design_data(c.payload) for c in plain.changes], r
        assert plan.choices == plain.choices and plan.board == plain.board, r


def test_a_models_required_question_under_a_material_key_is_asked_as_at_c270797():
    """Review finding: c270797 asked a model's required question under these keys as optional (no template read them) and never blocked on
    it; the material rule asks its own pcb_material question when a confirmed need requires one."""
    from ai_eda.agents.requirement import _model_question, blocks_pipeline
    from ai_eda.ir import MissingInformation

    for key in ("pcb_material", "board_material", "pcb_flexing", "board_flexing", "power_dissipation", "heat_dissipation"):
        q = MissingInformation(key=key, question=f"What is the {key}?", required=True, source="llm")
        asked = _model_question(q, frozenset())
        assert not asked.required and not blocks_pipeline(q, _ir(), frozenset()), key
        assert asked.rationale == (  # c270797's words, byte for byte (the recorded question is IR content)
            f"asked as required by the model, but no template reads '{key}', so it does not block: an answer becomes a design requirement a "
            f"template must serve (the closed world refuses it otherwise); --answer leave_out={key} closes the question")
    # the layer count was a board key at c270797: its question stays required
    q = MissingInformation(key="pcb_layers", question="How many layers?", required=True, source="llm")
    assert _model_question(q, frozenset()).required and blocks_pipeline(q, _ir(), frozenset())


# --------------------------------------------------------------------------- the templates


def _present(tmp_path: Path, answers: dict[str, str]):
    ir = CircuitIR(project=ProjectMeta(id="m", name="m", workdir=str(tmp_path)))
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={**BASE, **LED, **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    return ir, state


@needs_libs
def test_no_material_key_leaves_the_table_and_the_stack_as_they_were(tmp_path: Path):
    _ir_, state = _present(tmp_path, {})
    table = state.open_questions[0].question
    assert "pcb_material" not in table and "base material" not in table and "stackup: generic 2-layer (the default)" in table


@needs_libs
def test_a_stated_ims_builds_the_one_layer_stack_with_its_consequences(tmp_path: Path):
    ir, state = _present(tmp_path, {"pcb_material": "aluminium_ims"})
    assert [q.key for q in state.open_questions if q.required] == ["confirm_design"]
    table = state.open_questions[0].question
    for text in ("stackup: 1-copper-layer aluminium IMS (COBRITHERM HTC 3,2W) 1.625 mm", "stackup.copper_layers = 1", f"keep-out {SINGLE_LAYER_KEEPOUT_ID}",
                 "base material: Aismalibar COBRITHERM HTC 3,2W (chosen by you: req.pcb_material = aluminium_ims)", "carries no (stackup) section",
                 "no check in this version computes a temperature"):
        assert text in table, text
    assert "pcb_layers = 2" not in table  # the count is the material's, not a choice
    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={"confirm_design": "yes", "pcb.placement": "skip"})).run(
        ir, stop_after=Stage.ARCHITECTURE)
    stack = ir.pcb.stackup
    assert stack.metal_base is not None and stack.material.id == "aluminium_ims" and [layer.name for layer in ir.pcb.layers] == ["F.Cu", "B.Cu"]
    assert stack.provenance.derived_from == ["req.pcb_material"] and all(t.provenance.kind is ProvenanceKind.USER_REQUIREMENT for _, t in stack.traced_items())
    assert _unserved(ir) == []  # the stack serves req.pcb_material
    # a stated layer count does not fit a one-copper-layer stack
    _ir2, state = _present(tmp_path / "b", {"pcb_material": "aluminium_ims", "pcb_layers": "2"})
    assert not any(q.key == "confirm_design" for q in state.open_questions)


@needs_libs
def test_a_proposed_flex_material_is_a_row_the_user_confirms(tmp_path: Path):
    ir, state = _present(tmp_path, {"pcb_flexing": "static"})
    table = state.open_questions[0].question
    assert "pcb_material = 'polyimide_flex' - board base material: DuPont Pyralux AP (AP9121R) - proposed by the material rule" in table
    assert "pcb_layers = 2 - board layer count: 2 layers, the only count of the generic DuPont Pyralux AP (AP9121R) stack" in table
    assert "stackup.min_bend_radius: minimum bend radius 10 x the flex thickness" in table and "at least 1.2 mm" in table
    assert ir.pcb is None  # nothing applied before the confirmation
    Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": _lib}, answers={"confirm_design": "yes", "pcb.placement": "skip"})).run(
        ir, stop_after=Stage.ARCHITECTURE)
    assert ir.pcb.stackup.material.id == "polyimide_flex" and float(ir.pcb.stackup.material.min_bend_ratio.value) == 10.0
    assert ir.pcb.stackup.provenance.derived_from == ["req.pcb_flexing"] and _unserved(ir) == []
    _ir2, state = _present(tmp_path / "b", {"pcb_flexing": "static", "pcb_layers": "4"})
    assert not any(q.key == "confirm_design" for q in state.open_questions)


@needs_libs
def test_a_power_dissipation_blocks_on_the_material_question_until_answered(tmp_path: Path):
    ir, state = _present(tmp_path, {"power_dissipation": "3 W"})
    assert state.blocked and [q.key for q in state.open_questions if q.required] == ["pcb_material"] and ir.components == []
    _ir2, state = _present(tmp_path / "b", {"power_dissipation": "3 W", "pcb_material": "fr4"})
    table = state.open_questions[0].question
    assert "base material: FR-4 (generic) (chosen by you: req.pcb_material = fr4)" in table and "req.power_dissipation = 3 W is answered" in table


@needs_libs
def test_a_hf_material_under_a_four_layer_template_builds_the_rogers_stack_with_planes(tmp_path: Path):
    _ir_, state = _present(tmp_path, {"pcb_material": "rogers_ro4350b", "pcb_layers": "4"})
    table = state.open_questions[0].question
    assert "stackup: 4-layer Rogers RO4350B / RO4450F 1.262 mm (from req.pcb_layers); planes In1.Cu = GND, In2.Cu = VCC" in table
    assert "stackup.planes:" in table and "stackup.plane_edge_clearance = 0.5 mm" in table


# --------------------------------------------------------------------------- the IMS board


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def _ims_board(tmp_path: Path, lib, parts) -> CircuitIR:
    nets = {"A": [(parts[0][0], "1"), (parts[1][0], "1")], "B": [(parts[2][0], "1"), (parts[3][0], "1")]}
    ir = board_ir(tmp_path, lib, parts, nets, (30.0, 20.0))
    stack = material_stackup(material_stack("aluminium_ims", 1), MaterialDecision(material="aluminium_ims", reason="r"), "fixture", confirmed=True)
    ir.pcb.stackup = stack
    ir.pcb.layers = board_layers(stack)
    return ir


def _keepout_result(ir: CircuitIR, tmp_path: Path, lib):
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    return {r.check_id: r for r in default_registry.get(KEEPOUT_CHECK).validate(ir, ctx)}[KEEPOUT_CHECK]


def test_the_pcb_agent_keeps_b_cu_out_and_routes_f_cu_only(tmp_path: Path, lib):
    ir = _ims_board(tmp_path, lib, [("R1", "SMD1", 5.0, 5.0), ("R2", "SMD1", 25.0, 5.0), ("R3", "SMD1", 5.0, 15.0), ("R4", "SMD1", 25.0, 15.0)])
    ko = single_layer_keepout(ir.pcb.stackup, ir.pcb.outline)
    assert ko.layers == ["B.Cu"] and ko.rect.value == [0.0, 0.0, 30.0, 20.0] and ko.forbids == ["tracks", "vias", "pads", "zones"]
    assert ko.provenance.kind is ProvenanceKind.DERIVED and ko.provenance.derived_from == ["pcb.stackup.metal_base", "pcb.outline"]
    assert single_layer_keepout(generic_stackup(2, "t", confirmed=True), ir.pcb.outline) is None
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    Orchestrator.apply_proposals(ir, res.proposals)
    assert [k.id for k in ir.pcb.keepouts] == [SINGLE_LAYER_KEEPOUT_ID] and any(n.startswith(f"keep-out {SINGLE_LAYER_KEEPOUT_ID}") for n in res.notes)
    assert ir.pcb.tracks and {t.layer for t in ir.pcb.tracks} == {"F.Cu"} and ir.pcb.vias == []
    assert {t.provenance.tool_version for t in ir.pcb.tracks} == {ROUTER_KEEPOUT_VERSION}
    assert _keepout_result(ir, tmp_path, lib).status is S.PASS  # SMD pads only: nothing on B.Cu
    measured = measure_nets(ir, library=lib)  # the measure, the paths and the bounds read a stack without a through via
    assert all(m.routed and m.vias == 0 and m.via_length_mm == 0.0 and m.path_problem is None for m in measured.values())
    assert all(s.model is not None and s.model.bound and s.model.e_eff == 5.2 for m in measured.values() for s in m.segments)
    # deterministic, and a second run does not add the keep-out twice
    again = _ims_board(tmp_path / "again", lib, [("R1", "SMD1", 5.0, 5.0), ("R2", "SMD1", 25.0, 5.0), ("R3", "SMD1", 5.0, 15.0), ("R4", "SMD1", 25.0, 15.0)])
    Orchestrator.apply_proposals(again, PCBAgent().run(again, AgentContext(workdir=tmp_path, tools={"kicad_library": lib})).proposals)
    assert [design_data(t) for t in again.pcb.tracks] == [design_data(t) for t in ir.pcb.tracks] and len(again.pcb.keepouts) == 1


def test_a_through_hole_pad_on_b_cu_breaks_the_ims_keepout(tmp_path: Path, lib):
    ir = _ims_board(tmp_path, lib, [("J1", "PAD1", 5.0, 5.0), ("R2", "SMD1", 25.0, 5.0), ("R3", "SMD1", 5.0, 15.0), ("R4", "SMD1", 25.0, 15.0)])
    Orchestrator.apply_proposals(ir, PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib})).proposals)
    assert {t.layer for t in ir.pcb.tracks} == {"F.Cu"} and ir.pcb.vias == []
    bad = _keepout_result(ir, tmp_path, lib)
    assert bad.status is S.FAIL and "keep-out ims_bcu forbids pads; the copper of J1.1 lies in it" in bad.message


def test_the_compiled_ims_board_has_two_copper_layers_and_the_rule_area(tmp_path: Path):
    from ai_eda.compilers import CompileContext
    from ai_eda.compilers.pcb import PCBCompiler
    from ai_eda.ir import PCBDesign, Placement, Provenance
    from ai_eda.tools.kicad import sexpr
    from tests.test_parts_existence import synthetic_library
    from tests.test_pcb_agent import parts_ir

    slib = synthetic_library(tmp_path / "kicad")
    ir = parts_ir(tmp_path, slib, ("R1", "R2"))
    ims = material_stackup(material_stack("aluminium_ims", 1), MaterialDecision(material="aluminium_ims", reason="r"), "t", confirmed=True)
    outline = BoardOutline(width_mm=30.0, height_mm=20.0)
    user = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="by hand")
    ir.pcb = PCBDesign(outline=outline, layers=board_layers(ims), stackup=ims, keepouts=[single_layer_keepout(ims, outline)],
                       placements=[Placement(component_ref="R1", x_mm=10.0, y_mm=10.0, provenance=user), Placement(component_ref="R2", x_mm=20.0, y_mm=10.0, provenance=user)])
    ref = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": slib}))
    tree = sexpr.parse_file(Path(ref.path))
    layers = sexpr.find(tree, "layers")
    copper = [sexpr.args(e)[0] for e in layers[1:] if isinstance(e, list) and sexpr.args(e) and str(sexpr.args(e)[0]).endswith(".Cu")]
    text = Path(ref.path).read_text(encoding="utf-8")
    # KiCad gets two copper layers (an even count), the B.Cu rule area, and no (stackup) section: the material is not in the file
    assert copper == ["F.Cu", "B.Cu"] and "keepout_ims_bcu" in text and "(stackup" not in text
