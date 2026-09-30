"""The merged KR 447 MHz template family: ``RF_TEMPLATES`` and the selection end to end (kr447 design §2.0, §5 wave 3 step 1).

What runs where:

* always: the registry holds the four stage templates and the transceiver in
  the design's staging order, the family's ``BUILDS`` are the six
  ``RADIO_BUILDS``, each ``radio_build`` value triggers exactly its own
  template (``transceiver`` / ``transceiver_conducted`` both the one
  ``kr447_transceiver``), the selection through ``design_from_requirements``
  names that template, a
  confirmed carrier or modulation without ``radio_build`` is the family's
  required question, and the five base templates select exactly as they did
  with an empty registry;
* with the packed KiCad 10.0.6 libraries (``needs_libs``): every template
  presents a buildable plan from the requirements its project would state
  (the default companions composed; both transceiver builds), with every
  ``kr447.*`` and ``model.*`` row of the table saying UNVERIFIED.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ai_eda.design.templates as templates_mod
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design.inputs import RADIO_BUILD_KEY, RADIO_BUILDS, read_inputs
from ai_eda.design.rf import family, registry
from ai_eda.design.templates import RADIO_SELECTION, TEMPLATES, all_templates, design_from_requirements
from ai_eda.ir import CircuitIR, ProjectMeta
from ai_eda.tools.kicad.library import KicadLibrary

_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_AM_FM", "RF_Mixer", "Amplifier_Audio", "Device")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-103_26.21x26.21mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")

#: every radio_build in the design's staging order (§2.0 table) -> its template id (the two transceiver builds are one template)
STAGES = {"audio_ptt": "kr447_audio_ptt", "rx_backend": "kr447_rx_backend", "rx_frontend": "kr447_rx_frontend", "tx_exciter": "kr447_tx_exciter",
          "transceiver": "kr447_transceiver", "transceiver_conducted": "kr447_transceiver"}
#: the registry's template ids, in order
TEMPLATE_IDS = list(dict.fromkeys(STAGES.values()))
#: what each build's project states (the family's needs), beside radio_build
STATED = {
    "audio_ptt": {"input_voltage": "7.4 V", "modulation": "FM"},
    "rx_backend": {"input_voltage": "7.4 V", "modulation": "FM"},
    "rx_frontend": {"input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz"},
    "tx_exciter": {"input_voltage": "7.4 V", "modulation": "FM", "carrier_frequency": "447.5625 MHz"},
    "transceiver": {"input_voltage": "7.4 V", "modulation": "FM", "carrier_frequency": "447.5625 MHz"},
    "transceiver_conducted": {"input_voltage": "7.4 V", "modulation": "FM", "carrier_frequency": "447.5625 MHz"},
}
#: the base five's sample requirements (tests/test_circuit_templates.py, tests/test_atmega128_template.py)
BASE_FIVE = {
    "divider": {"input_voltage": "12 V", "output_voltage": "5 V"},
    "led": {"input_voltage": "5 V", "led_forward_voltage": "2.0 V", "led_forward_current": "10 mA"},
    "rc_lowpass": {"cutoff_frequency": "1 kHz"},
    "astable": {"input_voltage": "5 V", "oscillation_frequency": "1 kHz"},
    "atmega128_devboard": {"input_voltage": "9 V", "clock_frequency": "16 MHz"},
}


def _ir(tmp_path: Path, answers: dict[str, str], name: str = "sel") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    return ir


def test_the_registry_holds_the_five_templates_in_staging_order() -> None:
    assert [t.id for t in registry.RF_TEMPLATES] == TEMPLATE_IDS
    assert [t.id for t in all_templates()] == [*(t.id for t in TEMPLATES), *TEMPLATE_IDS]
    assert set(family.BUILDS) == set(RADIO_BUILDS) == set(STAGES)  # the wave-3 assertion of the design
    for build, tid in STAGES.items():
        t = next(t for t in registry.RF_TEMPLATES if t.id == tid)
        assert family.BUILDS[build].template_id == tid and RADIO_BUILD_KEY in t.serves and t.serves == family.BUILDS[build].serves


@pytest.mark.parametrize("build", RADIO_BUILDS)
def test_each_radio_build_triggers_exactly_its_own_template(tmp_path: Path, build: str) -> None:
    ir = _ir(tmp_path, {RADIO_BUILD_KEY: build, **STATED.get(build, {"input_voltage": "7.4 V"})})
    inputs, _ = read_inputs(ir)
    triggered = [t.id for t in all_templates() if t.triggered_by(ir, inputs)]
    assert triggered == [STAGES[build]], triggered
    plan = design_from_requirements(ir, KicadLibrary(roots=[tmp_path / "empty"]))
    assert plan is not None and plan.template == STAGES[build]  # selected (it may refuse here: no library on disk)


def test_a_radio_build_no_registered_template_builds_is_a_question_naming_the_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the transceiver absent from the registry (as before part P13), its builds select nothing and the selection says so."""
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [t for t in registry.RF_TEMPLATES if t.id != "kr447_transceiver"])
    plan = design_from_requirements(_ir(tmp_path, {RADIO_BUILD_KEY: "transceiver", **STATED["transceiver"]}), KicadLibrary(roots=[tmp_path / "empty"]))
    assert plan is not None and plan.template == RADIO_SELECTION and not plan.buildable and "no registered radio template builds it" in plan.notes[0]
    assert [q.key for q in plan.questions] == [RADIO_BUILD_KEY] and not plan.questions[0].required


def test_a_radio_request_without_radio_build_is_the_family_question(tmp_path: Path) -> None:
    plan = design_from_requirements(_ir(tmp_path, {"carrier_frequency": "447.5625 MHz", "modulation": "FM"}), KicadLibrary(roots=[tmp_path / "empty"]))
    assert plan is not None and plan.template == RADIO_SELECTION and not plan.buildable
    assert [q.key for q in plan.questions] == [RADIO_BUILD_KEY] and plan.questions[0].required
    assert design_from_requirements(_ir(tmp_path, {"input_voltage": "7.4 V"}), KicadLibrary(roots=[tmp_path / "empty"])) is None


def test_an_unreadable_radio_build_is_a_question_with_the_reason(tmp_path: Path) -> None:
    plan = design_from_requirements(_ir(tmp_path, {RADIO_BUILD_KEY: "walkie-talkie", "input_voltage": "7.4 V"}), KicadLibrary(roots=[tmp_path / "empty"]))
    assert plan is not None and plan.template == RADIO_SELECTION and not plan.buildable and RADIO_BUILD_KEY in plan.notes[0]


@pytest.mark.parametrize("tid", list(BASE_FIVE))
def test_the_base_five_select_as_they_did_with_an_empty_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tid: str) -> None:
    lib = KicadLibrary(roots=[tmp_path / "empty"])
    ir = _ir(tmp_path, BASE_FIVE[tid])
    with_rf = design_from_requirements(ir, lib)
    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [])
    without = design_from_requirements(ir, lib)
    assert with_rf is not None and without is not None and with_rf.template == without.template == tid
    assert (with_rf.buildable, with_rf.notes, [q.key for q in with_rf.questions]) == (without.buildable, without.notes, [q.key for q in without.questions])


@needs_libs
@pytest.mark.parametrize("build", list(STAGES))
def test_every_template_presents_a_buildable_plan_with_the_default_companions(tmp_path: Path, build: str) -> None:
    ir = _ir(tmp_path, {RADIO_BUILD_KEY: build, **STATED[build]})
    plan = design_from_requirements(ir, _REAL)
    assert plan is not None and plan.template == STAGES[build] and plan.buildable, plan.notes
    rows = [line for line in plan.table().splitlines() if line.lstrip().startswith(("kr447.", "model."))]
    assert rows and all("UNVERIFIED" in line for line in rows), [r for r in rows if "UNVERIFIED" not in r][:3]
    parts = {line.split()[0].rstrip(":") for line in plan.parts}
    assert {"J101", "U101"} <= parts or build == "tx_exciter" and {"J101", "U102"} <= parts  # part P9's power block is on every board
