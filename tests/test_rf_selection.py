"""Radio template selection: the ``radio_build`` / ``tx_timeout`` keys, ``Template.triggered_by``, layer policies, the lazy RF registry.

What is proved here, with test doubles for the radio templates and the
family's selector (the real ones live in ``ai_eda.design.rf``, written by
another part; the tests that need them skip until they exist):

* ``radio_build`` is a categorical *requirement* key (not a control key):
  ``canonical_key`` maps it, ``read_radio_build`` takes exactly one confirmed
  build name and refuses everything else with the reason; ``tx_timeout`` is a
  seconds key with its two aliases. Every existing template refuses both
  (closed world) - none builds a radio.
* The five existing templates keep their selection and their stack byte for
  byte: ``triggered_by`` defaults to ``triggered``, the default layer policy
  is every generic stack with 2 layers, the confirmation row is the old text,
  and a registered radio template changes nothing for them.
* The lazy registry: an absent RF package is "no radio template"; a broken
  one raises; an empty one changes nothing.
* The selection: a registered radio template is selected by its own
  ``triggered_by``; a confirmed ``carrier_frequency`` or ``modulation``
  without ``radio_build`` is the family's required question; an unreadable or
  unbuildable ``radio_build`` is a non-required question with the reason.
* Layer policies: a template that builds only 4 layers defaults to 4, says so
  in its table row, refuses a stated 2 (and an IR's own 2-layer stack) with
  its reason - before it asks for a missing input.
* The extraction prompt names ``tx_timeout``, ``radio_build`` with its six
  values and says a band is not a carrier frequency (``EXTRACTION_VERSION`` 0.4).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

import ai_eda.design.templates as templates_mod
from ai_eda.agents import AgentContext, CircuitDesignAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.keys import CONTROL_KEYS
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design import (
    DEFAULT_LAYER_COUNT,
    DEFAULT_LAYER_POLICY,
    KEY_ALIASES,
    LAYER_COUNT_OPTIONS,
    MODULATION_ALIASES,
    RADIO_BUILD_ALIASES,
    RADIO_BUILD_KEY,
    RADIO_BUILDS,
    RF_FAMILY_MODULE,
    RF_REGISTRY_MODULE,
    TEMPLATES,
    UNIT_OF,
    Choice,
    DesignChange,
    LayerPolicy,
    Plan,
    Template,
    all_templates,
    canonical_key,
    design_from_requirements,
    generic_stackup,
    layer_count_choice_text,
    read_inputs,
    read_radio_build,
    rf_templates,
    template_keys_text,
    unserved_requirements,
)
from ai_eda.design.base import choice_provenance
from ai_eda.design.inputs import present_keys
from ai_eda.design.rf import family
from ai_eda.design.board import layer_policy_refusal, read_board_layers
from ai_eda.design.stackup import board_layers
from ai_eda.ir import CircuitIR, MissingInformation, PCBDesign, ProjectMeta, Requirement, RequirementKind, Traced, llm_generated, user_requirement
from ai_eda.llm.extraction import EXTRACTION_VERSION
from ai_eda.llm.prompts import REQUIREMENT_EXTRACTION_SYSTEM
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.workflow import Orchestrator
from tests.test_circuit_templates import DIVIDER, template_library

#: the reason every test double with a 4-layer-only policy gives (a radio template's own words, quoted by its refusals)
PLANE_REASON = "the RF lines need a reference plane (decision 7A: 4 layers)"
#: the confirmation row every template showed for its default count before layer policies existed
OLD_DEFAULT_ROW = "board layer count: the default 2 layers (answer pcb_layers=4 for a 4-layer board with ground / power planes)"
#: a name for the fake family module (never a real package)
FAKE_FAMILY = "tests_p3_fake_radio_family"


def _ir(*answers: tuple[str, object]) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    for key, answer in answers:
        ir.requirements.requirements.append(_answer_requirement(key, answer))  # type: ignore[arg-type]
    return ir


def _llm(key: str, value: object) -> Requirement:
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value}", kind=RequirementKind.EXPLICIT, value=llm_generated(value, model="m", note="quote"), category="electrical")


class FakeRadioTemplate(Template):
    """Test double of a radio template: selected by ``radio_build`` = its build; needs ``carrier_frequency``; a one-parameter plan."""

    title = "fake radio board"
    triggers = ()
    serves = ("radio_build", "carrier_frequency", "modulation", "input_voltage", "tx_timeout")

    def __init__(self, id: str = "fake_rx_backend", build: str = "rx_backend", policy: LayerPolicy | None = None,
                 needs: tuple[str, ...] = ("carrier_frequency",)) -> None:
        self.id = id
        self.build_name = build
        self.layer_policy = policy or LayerPolicy(allowed=(4,), default=4, reason=PLANE_REASON)
        self.needs = needs

    def triggered_by(self, ir: CircuitIR, inputs) -> bool:
        return read_radio_build(ir)[0] == self.build_name

    def build(self, ir, inputs, unusable, library, *, confirmed: bool) -> Plan:
        plan = Plan(template=self.id, title=self.title)
        missing = [k for k in self.needs if k not in present_keys(ir, inputs)]  # the selection gate's own rule
        if missing:
            plan.questions += [MissingInformation(key=k, question=f"{self.id} needs {k}", rationale="template input") for k in missing]
            plan.notes.append(f"template {self.id} not proposed: input(s) {missing} missing")
            return plan
        plan.choices.append(Choice("fake_x", "a fake free choice", 1.0, None))
        plan.parts.append("(no part: a test double)")
        value = Traced(value=1.0, provenance=choice_provenance(self.id, "fake_x = 1.0: a fake free choice", confirmed))
        plan.changes.append(DesignChange(description="parameter fake_x", target="parameters.fake_x", operation="set", payload=value))
        return plan


def _selector(ir: CircuitIR) -> MissingInformation:
    return MissingInformation(key=RADIO_BUILD_KEY, question="which board of the fake radio family? one of " + ", ".join(RADIO_BUILDS),
                              options=list(RADIO_BUILDS), rationale="fake selector")


@pytest.fixture()
def radio(monkeypatch: pytest.MonkeyPatch):
    """Register radio template doubles and a fake family module; returns the list to fill."""
    registered: list[Template] = []
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=RF_REGISTRY_MODULE: list(registered))
    family = types.ModuleType(FAKE_FAMILY)
    family.selector_question = _selector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FAKE_FAMILY, family)
    monkeypatch.setattr(templates_mod, "RF_FAMILY_MODULE", FAKE_FAMILY)
    return registered


def _applied_hash(plan: Plan) -> str:
    """The design hash of a fresh divider-request IR after the plan's changes are applied."""
    ir = _ir(*DIVIDER.items())
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload) for c in plan.changes])
    return ir.content_hash()


def _plan(ir: CircuitIR) -> Plan | None:
    return design_from_requirements(ir, None)  # type: ignore[arg-type]  # the doubles read no library


# --------------------------------------------------------------------------- the keys


def test_radio_build_is_a_categorical_requirement_key_not_a_control_key():
    assert RADIO_BUILD_KEY == "radio_build" and RADIO_BUILD_ALIASES == ("radio_build",)
    assert RADIO_BUILDS == ("audio_ptt", "rx_backend", "rx_frontend", "tx_exciter", "transceiver", "transceiver_conducted")
    assert canonical_key("radio_build") == "radio_build" and "radio_build" not in KEY_ALIASES and "radio_build" not in UNIT_OF
    assert RADIO_BUILD_KEY not in CONTROL_KEYS
    every = [a for aliases in KEY_ALIASES.values() for a in aliases] + list(MODULATION_ALIASES)
    assert not set(RADIO_BUILD_ALIASES) & set(every)
    # a typed answer is a design requirement: every existing template refuses it (none builds a radio)
    ir = _ir(("radio_build", "rx_backend"))
    req = ir.requirements.requirements[0]
    assert req.category == "electrical" and req.value.provenance.is_authoritative
    for t in TEMPLATES:
        assert [r.id for r in unserved_requirements(ir, t)] == ["req.radio_build"], t.id


def test_read_radio_build_takes_exactly_one_confirmed_build_name():
    assert read_radio_build(_ir()) == (None, None)
    for build in RADIO_BUILDS:
        assert read_radio_build(_ir(("radio_build", build))) == (build, None)
    for text, build in ((" RX_Backend ", "rx_backend"), ("tx-exciter", "tx_exciter"), ("Transceiver-Conducted", "transceiver_conducted")):
        assert read_radio_build(_ir(("radio_build", text))) == (build, None), text
    for text, expect in (
        ("rx_backend board", "is not exactly one radio build name (it contains rx_backend among other words)"),
        ("rx_backend, tx_exciter", "names several radio builds (rx_backend, tx_exciter)"),
        ("IF 백엔드 시험 보드", "names no radio build (state exactly one of audio_ptt, rx_backend"),
        ("transceiver conducted", "(it contains transceiver among other words); state exactly one of audio_ptt"),  # a space is no separator
        ("rx backend", "names no radio build"),
    ):
        build, why = read_radio_build(_ir(("radio_build", text)))
        assert build is None and expect in (why or ""), (text, why)
    build, why = read_radio_build(_ir(("radio_build", 5)))
    assert build is None and "is not a radio build name" in (why or "")
    ir = _ir()
    ir.requirements.requirements.append(_llm("radio_build", "rx_backend"))
    build, why = read_radio_build(ir)
    assert build is None and "not yet the user's (confirm it, or answer radio_build directly)" in (why or "")
    ir = _ir(("radio_build", "rx_backend"))
    ir.requirements.requirements.append(Requirement(id="req.radio_build.2", key="radio_build", text="radio_build: tx_exciter", kind=RequirementKind.EXPLICIT,
                                                    value=user_requirement("tx_exciter"), category="electrical"))
    build, why = read_radio_build(ir)
    assert build is None and (why or "").startswith("ambiguous: req.radio_build says rx_backend, req.radio_build.2 says tx_exciter")
    ir.requirements.requirements[1].value = user_requirement("RX-BACKEND")
    assert read_radio_build(ir) == ("rx_backend", None)


def test_tx_timeout_is_a_seconds_key_that_every_existing_template_refuses():
    assert KEY_ALIASES["tx_timeout"] == ("tx_timeout", "transmit_timeout") and UNIT_OF["tx_timeout"] == "s"
    assert canonical_key("transmit_timeout") == "tx_timeout"
    found, unusable = read_inputs(_ir(("tx_timeout", "180 s")))
    assert unusable == {} and (found["tx_timeout"].traced.value, found["tx_timeout"].traced.unit) == (180.0, "s")
    found, _ = read_inputs(_ir(("transmit_timeout", "180s")))
    assert found["tx_timeout"].traced.value == 180.0 and found["tx_timeout"].requirement.id == "req.transmit_timeout"
    _, unusable = read_inputs(_ir(("tx_timeout", "3 min")))
    assert "no quantity with a unit" in unusable["tx_timeout"]  # minutes are not read: state seconds
    ir = _ir(("tx_timeout", "180 s"))
    for t in TEMPLATES:
        assert [r.id for r in unserved_requirements(ir, t)] == ["req.tx_timeout"], t.id


# --------------------------------------------------------------------------- existing templates unchanged


def test_existing_templates_keep_their_selection_and_their_generic_layer_policy():
    ir = _ir(*DIVIDER.items())
    inputs, _ = read_inputs(ir)
    for t in TEMPLATES:
        assert t.layer_policy is DEFAULT_LAYER_POLICY and t.layer_policy.is_generic and not t.layer_policy.restricts
        assert t.triggered_by(ir, inputs) == t.triggered(inputs)
    assert DEFAULT_LAYER_POLICY == LayerPolicy(allowed=LAYER_COUNT_OPTIONS, default=DEFAULT_LAYER_COUNT)
    assert layer_count_choice_text() == layer_count_choice_text(DEFAULT_LAYER_POLICY) == OLD_DEFAULT_ROW
    assert len(TEMPLATES) == 5 and [t.id for t in all_templates()] == [t.id for t in TEMPLATES]


def test_a_registered_radio_template_changes_nothing_for_the_existing_templates(tmp_path: Path, radio: list[Template]):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(*DIVIDER.items())
    before = design_from_requirements(ir, lib)
    radio.append(FakeRadioTemplate())
    after = design_from_requirements(ir, lib)
    assert before is not None and after is not None and before.buildable and after.buildable
    assert after.template == "divider" and after.table() == before.table() and OLD_DEFAULT_ROW in after.table()
    assert [(c.target, c.operation, c.description) for c in after.changes] == [(c.target, c.operation, c.description) for c in before.changes]
    assert _applied_hash(after) == _applied_hash(before)  # the same design content (wall-clock stamps aside)
    assert "fake_rx_backend needs carrier_frequency" in template_keys_text() and template_keys_text().startswith("divider needs input_voltage + output_voltage; ")


# --------------------------------------------------------------------------- the lazy registry


def test_the_registry_is_optional_but_a_broken_one_is_never_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert RF_REGISTRY_MODULE == "ai_eda.design.rf.registry" and RF_FAMILY_MODULE == "ai_eda.design.rf.family"
    monkeypatch.setitem(sys.modules, RF_REGISTRY_MODULE, None)  # the RF package absent, in any tree
    assert rf_templates() == [] and [t.id for t in all_templates()] == [t.id for t in TEMPLATES]
    assert rf_templates("tests_p3_no_such_package.registry") == []
    empty = types.ModuleType("tests_p3_empty_registry")
    empty.RF_TEMPLATES = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tests_p3_empty_registry", empty)
    assert rf_templates("tests_p3_empty_registry") == []
    holder = types.ModuleType("tests_p3_bad_registry")
    monkeypatch.setitem(sys.modules, "tests_p3_bad_registry", holder)
    with pytest.raises(AttributeError, match="defines no RF_TEMPLATES"):
        rf_templates("tests_p3_bad_registry")
    holder.RF_TEMPLATES = ["not a template"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="holds non-templates"):
        rf_templates("tests_p3_bad_registry")
    # a registry whose own import fails on something else raises: a broken RF package is not "no radio template"
    pkg = tmp_path / "tests_p3_broken_rf"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "registry.py").write_text("import tests_p3_missing_dependency_xyz  # noqa: F401\nRF_TEMPLATES = []\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        with pytest.raises(ModuleNotFoundError, match="tests_p3_missing_dependency_xyz"):
            rf_templates("tests_p3_broken_rf.registry")
    finally:
        for name in ("tests_p3_broken_rf.registry", "tests_p3_broken_rf"):
            sys.modules.pop(name, None)


def test_two_templates_under_one_id_are_refused(radio: list[Template]):
    radio.append(FakeRadioTemplate(id="divider"))
    with pytest.raises(ValueError, match=r"template id\(s\) \['divider'\] registered twice"):
        all_templates()


# --------------------------------------------------------------------------- selection


def test_without_radio_templates_nothing_asks_for_a_radio_build(radio: list[Template], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, FAKE_FAMILY, None)  # the family is never imported without a registered radio template
    assert _plan(_ir(("carrier_frequency", "447.5625 MHz"), ("modulation", "FM"))) is None
    assert _plan(_ir(("radio_build", "rx_backend"))) is None


def test_a_confirmed_radio_quantity_without_radio_build_asks_the_family_question(radio: list[Template]):
    radio.append(FakeRadioTemplate())
    for answers, evidence in (
        ((("carrier_frequency", "447.5625 MHz"),), "carrier_frequency (req.carrier_frequency)"),
        ((("modulation", "FM"),), "modulation fm"),
        ((("carrier_frequency", "447.5625 MHz"), ("modulation", "주파수 변조")), "carrier_frequency (req.carrier_frequency), modulation fm"),
    ):
        plan = _plan(_ir(*answers))
        assert plan is not None and not plan.buildable and plan.template == RADIO_BUILD_KEY
        assert [(q.key, q.required) for q in plan.questions] == [(RADIO_BUILD_KEY, True)] and plan.questions[0].options == list(RADIO_BUILDS)
        assert f"radio requirement(s) confirmed ({evidence}) but no radio_build: asked which board" in plan.notes[0]
    # no radio quantity (or an unconfirmed one): no radio question, the ordinary "no template" path
    assert _plan(_ir(("input_voltage", "7.4 V"))) is None
    ir = _ir()
    ir.requirements.requirements.append(_llm("carrier_frequency", "447.5625 MHz"))
    assert _plan(ir) is None


def test_the_family_question_must_come_from_the_family_under_radio_build(radio: list[Template], monkeypatch: pytest.MonkeyPatch):
    radio.append(FakeRadioTemplate())
    wrong = types.ModuleType("tests_p3_wrong_family")
    wrong.selector_question = lambda ir: MissingInformation(key="board", question="?")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tests_p3_wrong_family", wrong)
    monkeypatch.setattr(templates_mod, "RF_FAMILY_MODULE", "tests_p3_wrong_family")
    with pytest.raises(TypeError, match="must return a MissingInformation under 'radio_build'"):
        _plan(_ir(("modulation", "FM")))


def test_an_unreadable_or_unbuildable_radio_build_is_a_non_required_question(radio: list[Template]):
    radio.append(FakeRadioTemplate())
    plan = _plan(_ir(("radio_build", "IF 백엔드 보드"), ("carrier_frequency", "447.5625 MHz")))
    assert plan is not None and not plan.buildable
    assert [(q.key, q.required) for q in plan.questions] == [(RADIO_BUILD_KEY, False)]  # the requirement must change: never the selector
    assert "names no radio build" in plan.questions[0].question and plan.notes[0].startswith("radio_build not usable: req.radio_build:")
    plan = _plan(_ir(("radio_build", "tx_exciter"), ("carrier_frequency", "447.5625 MHz")))
    assert plan is not None and [(q.key, q.required) for q in plan.questions] == [(RADIO_BUILD_KEY, False)]
    assert "radio_build = tx_exciter: no registered radio template builds it from the confirmed requirements (radio templates: fake_rx_backend)" in plan.notes[0]


def test_a_named_build_selects_its_template_and_the_closed_world_still_applies(radio: list[Template]):
    radio += [FakeRadioTemplate(), FakeRadioTemplate(id="fake_tx_exciter", build="tx_exciter")]
    plan = _plan(_ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz")))
    assert plan is not None and plan.buildable and plan.template == "fake_rx_backend"
    # its missing input is a required question of the template itself
    plan = _plan(_ir(("radio_build", "rx_backend")))
    assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("carrier_frequency", True)]
    # a confirmed requirement it does not serve refuses it (closed world), naming the key
    plan = _plan(_ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz"), ("tx_power", "0.5 W")))
    assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("tx_power", False)]
    # a radio template and an existing one selected together is ambiguous, never a pick
    plan = _plan(_ir(("radio_build", "rx_backend"), *DIVIDER.items()))
    assert plan is not None and plan.title == "ambiguous" and "ambiguous: templates ['divider', 'fake_rx_backend']" in plan.notes[0]


class FamilyRadioTemplate(FakeRadioTemplate):
    """A radio template double whose needs and serves are the family table's (``family.BUILDS``), as a wave-2 template takes them."""

    def __init__(self, build: str = "rx_backend") -> None:
        super().__init__(id=f"fake_{build}", build=build, needs=family.BUILDS[build].needs)
        self.serves = family.BUILDS[build].serves


def test_a_categorical_need_counts_once_read_so_the_closed_world_still_refuses(radio: list[Template]):
    """``modulation`` is a need of five builds and never a numeric input: the gate counts it present once read_modulation reads it, so a
    confirmed requirement the build does not serve (tx_power on rx_backend) refuses it (it was built, and tx_power silently dropped)."""
    radio.append(FamilyRadioTemplate("rx_backend"))
    stated = (("radio_build", "rx_backend"), ("modulation", "FM"), ("input_voltage", "7.4 V"))
    ir = _ir(*stated, ("tx_power", "0.5 W"))
    assert present_keys(ir, read_inputs(ir)[0]) >= {"modulation", "radio_build", "input_voltage", "tx_power"}
    plan = _plan(ir)
    assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("tx_power", False)], plan
    assert "radio_build" in family.BUILDS["rx_backend"].serves and family.serving_builds("radio_build") == RADIO_BUILDS  # never its own refusal
    plan = _plan(_ir(*stated))
    assert plan is not None and plan.buildable and plan.template == "fake_rx_backend"
    # a stated modulation the reader cannot read stays missing: the template's own required question, never a refusal or a build
    for text in ("AM/FM", "no idea"):
        plan = _plan(_ir(("radio_build", "rx_backend"), ("modulation", text), ("input_voltage", "7.4 V"), ("tx_power", "0.5 W")))
        assert plan is not None and not plan.buildable and [(q.key, q.required) for q in plan.questions] == [("modulation", True)], (text, plan)


def test_a_template_needs_only_keys_a_reader_can_supply():
    with pytest.raises(TypeError, match=r"needs \['colour'\], which neither read_inputs nor a categorical reader supplies"):
        type("BadTemplate", (FakeRadioTemplate,), {"needs": ("carrier_frequency", "colour")})
    assert type("GoodTemplate", (FakeRadioTemplate,), {"needs": ("modulation", "radio_build", "carrier_frequency")}).needs[0] == "modulation"


# --------------------------------------------------------------------------- layer policies


def test_layer_policies_are_checked_when_a_template_class_is_defined():
    assert LayerPolicy(allowed=(4,), default=4, reason=PLANE_REASON).restricts
    audio = LayerPolicy(allowed=(2, 4), default=4)
    assert not audio.restricts and not audio.is_generic and audio.allowed_text() == "2 or 4"
    for kwargs, expect in (
        ({"allowed": ()}, "at least one layer count"),
        ({"allowed": (2, 3), "default": 2, "reason": "x"}, r"\[3\] have no generic stack"),
        ({"allowed": (4, 4), "default": 4, "reason": "x"}, "repeat a count"),
        ({"allowed": (4,), "default": 2, "reason": "x"}, "the default 2 layers is not one of the allowed counts"),
        ({"allowed": (4,), "default": 4}, "must say why"),
        ({"allowed": (4,), "default": 4, "reason": "   "}, "must say why"),
        ({"allowed": (True,), "default": True}, "plain integer"),
    ):
        with pytest.raises(ValueError, match=expect):
            LayerPolicy(**kwargs)


def test_a_four_layer_only_template_defaults_to_four_and_says_why(radio: list[Template]):
    radio.append(FakeRadioTemplate())
    ir = _ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz"))
    plan = _plan(ir)
    assert plan is not None and plan.buildable
    table = plan.table()
    assert f"pcb_layers = 4 - board layer count: 4 layers, the only count this template builds: {PLANE_REASON}" in table
    assert OLD_DEFAULT_ROW not in table and "stackup: generic 4-layer (the default); planes In1.Cu = GND, In2.Cu = GND" in table
    pcb = next(c.payload for c in plan.changes if c.target == "pcb")
    assert pcb.stackup.layer_count == 4 and [c.plane_net.value for c in pcb.stackup.plane_layers()] == ["GND", "GND"]
    layers, why = read_board_layers(radio[0], ir)
    assert why is None and layers.value == 4 and layers.is_default
    # a stated 4 is an input, not a choice
    ir = _ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz"), ("pcb_layers", "4"))
    plan = _plan(ir)
    assert plan is not None and plan.buildable and "board layer count" not in plan.table() and "req.pcb_layers: pcb_layers = 4 layers" in plan.table()


def test_a_stated_count_outside_the_policy_refuses_before_a_missing_input_is_asked(radio: list[Template]):
    radio.append(FakeRadioTemplate())
    for key in ("pcb_layers", "layer_count"):
        ir = _ir(("radio_build", "rx_backend"), (key, "2"))  # carrier_frequency missing too
        plan = _plan(ir)
        assert plan is not None and not plan.buildable
        assert [(q.key, q.required) for q in plan.questions] == [(key, False)]  # the requirement's own key; no required question
        why = f"req.{key}: 2 layers is not a count template fake_rx_backend builds (it builds 4 layers): {PLANE_REASON}"
        assert plan.questions[0].rationale == why and plan.notes == [f"template fake_rx_backend not proposed: {why}"]
        assert "Change the requirement to 4 layers (or leave it out: the template's default is 4 layers)" in plan.questions[0].question
        assert read_board_layers(radio[0], ir) == (None, why)
    # an unreadable count is not the policy's business: the existing rule refuses it after the build, as for every template
    ir = _ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz"), ("pcb_layers", "3"))
    assert layer_policy_refusal(radio[0], ir) is None
    plan = _plan(ir)
    assert plan is not None and not plan.buildable and "pcb_layers: req.pcb_layers: 3 layers is not one of the stackups this version builds" in plan.notes[-1]


def test_a_template_default_of_four_that_also_builds_two(radio: list[Template]):
    radio.append(FakeRadioTemplate(id="fake_audio_ptt", build="audio_ptt", policy=LayerPolicy(allowed=(2, 4), default=4), needs=("input_voltage",)))
    plan = _plan(_ir(("radio_build", "audio_ptt"), ("input_voltage", "7.4 V")))
    assert plan is not None and plan.buildable
    assert "pcb_layers = 4 - board layer count: this template's default 4 layers (answer pcb_layers=2 for a 2-layer board without planes)" in plan.table()
    assert next(c.payload for c in plan.changes if c.target == "pcb").stackup.layer_count == 4
    plan = _plan(_ir(("radio_build", "audio_ptt"), ("input_voltage", "7.4 V"), ("pcb_layers", "2")))
    assert plan is not None and plan.buildable and next(c.payload for c in plan.changes if c.target == "pcb").stackup.layer_count == 2


def test_an_irs_own_stack_is_kept_unless_a_restricting_policy_cannot_use_it(radio: list[Template]):
    two = generic_stackup(2, "user", confirmed=True)
    radio += [FakeRadioTemplate(), FakeRadioTemplate(id="fake_generic", build="rx_frontend", policy=DEFAULT_LAYER_POLICY)]
    ir = _ir(("radio_build", "rx_backend"), ("carrier_frequency", "447.5625 MHz"))
    ir.pcb = PCBDesign(layers=board_layers(two), stackup=two)
    plan = _plan(ir)
    assert plan is not None and not plan.buildable
    assert plan.notes[-1] == f"template fake_rx_backend not proposed: pcb_layers: the IR's own 2-layer stack is not a count template fake_rx_backend builds (it builds 4 layers): {PLANE_REASON}"
    ir.requirements.requirements[0] = _answer_requirement("radio_build", "rx_frontend")
    plan = _plan(ir)  # the generic policy keeps any stack the IR already has, as every template did before
    assert plan is not None and plan.buildable and "stackup: the IR's own 2-layer stack is kept (no generic stack proposed)" in plan.table()
    four = generic_stackup(4, "user", confirmed=True)
    ir.requirements.requirements[0] = _answer_requirement("radio_build", "rx_backend")
    ir.pcb = PCBDesign(layers=board_layers(four), stackup=four)
    plan = _plan(ir)
    assert plan is not None and plan.buildable and "stackup: the IR's own 4-layer stack is kept" in plan.table()


# --------------------------------------------------------------------------- through the circuit agent


def test_the_circuit_agent_asks_the_board_then_presents_and_applies_the_radio_template(tmp_path: Path, radio: list[Template]):
    radio.append(FakeRadioTemplate())
    lib = KicadLibrary(roots=[tmp_path / "kicad"])
    ir = _ir(("carrier_frequency", "447.5625 MHz"), ("modulation", "FM"))
    res = CircuitDesignAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert res.blocked_on_user and [q.key for q in res.questions] == [RADIO_BUILD_KEY] and res.proposals == []
    # the user types the board once: a requirement (it persists), then the table is shown and recorded
    ir.requirements.requirements.append(_answer_requirement("radio_build", "rx_backend"))
    res = CircuitDesignAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    assert [q.key for q in res.questions] == ["confirm_design"] and "4 layers, the only count this template builds" in res.questions[0].question
    Orchestrator.apply_proposals(ir, res.proposals)
    res = CircuitDesignAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={"confirm_design": "yes"}))
    assert not res.blocked_on_user and "template fake_rx_backend v0.3 confirmed by the user" in " ".join(res.notes)
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.pcb is not None and ir.pcb.stackup.layer_count == 4 and ir.parameters["fake_x"].value == 1.0 and ir.si is not None


# --------------------------------------------------------------------------- the extraction prompt


def test_the_extraction_prompt_names_tx_timeout_radio_build_and_the_band_rule():
    assert EXTRACTION_VERSION == "0.4"
    assert "tx_timeout (s: the transmit time-out" in REQUIREMENT_EXTRACTION_SYSTEM
    assert "radio_build names which board of the KR 447 MHz walkie-talkie template family to build" in REQUIREMENT_EXTRACTION_SYSTEM
    assert "never map a description of a board to one" in REQUIREMENT_EXTRACTION_SYSTEM
    assert ("exactly one of " + ", ".join(RADIO_BUILDS)) in REQUIREMENT_EXTRACTION_SYSTEM
    assert "A band is not a carrier_frequency" in REQUIREMENT_EXTRACTION_SYSTEM and '"447 MHz 대역"' in REQUIREMENT_EXTRACTION_SYSTEM
    assert "never extract it as carrier_frequency - ask (required) for the channel's carrier_frequency instead" in REQUIREMENT_EXTRACTION_SYSTEM


# --------------------------------------------------------------------------- the real radio family (another part; skips until it exists)


def test_the_real_family_builds_the_same_six_boards_and_asks_under_radio_build():
    family = pytest.importorskip("ai_eda.design.rf.family")
    assert set(family.BUILDS) == set(RADIO_BUILDS)
    q = family.selector_question(_ir(("carrier_frequency", "447.5625 MHz")))
    assert isinstance(q, MissingInformation) and q.key == RADIO_BUILD_KEY and q.required
    for key in ("tx_power", "channel_spacing", "carrier_frequency"):
        assert set(family.serving_builds(key)) <= set(RADIO_BUILDS), key


def test_the_real_registry_holds_templates_selected_only_by_radio_build():
    pytest.importorskip("ai_eda.design.rf.registry")
    rf = rf_templates()
    assert len({t.id for t in all_templates()}) == len(TEMPLATES) + len(rf)
    ir = _ir(("carrier_frequency", "447.5625 MHz"), ("modulation", "FM"), ("input_voltage", "7.4 V"))
    inputs, _ = read_inputs(ir)
    for t in rf:
        assert RADIO_BUILD_KEY in t.serves and not t.triggered_by(ir, inputs), t.id
