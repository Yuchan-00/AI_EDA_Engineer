"""The five defects the 900 MHz AM walkie-talkie test found outside RF (decision 6A): F1, F2, F3, F4, R1.

* F1 - ``missing_information`` was PASS with an empty message whenever no required question was open, even when
  nothing had read the request (no ``--llm``): no evidence is ``NOT_VERIFIED`` with the reason; PASS only for a
  confirmed extraction of the request text as it is now.
* F2 - ``rc_lowpass`` built a 1.77 mOhm / 100 nF "filter" for a 900 MHz cutoff and every check passed on the ideal
  model: the template now has a validity range (2 Hz..100 kHz) and refuses outside it, naming the range.
* F3 - the astable asked for a missing supply before checking the range of the frequency it had (the answer could
  only lead to a refusal), and the bare key ``frequency`` silently meant the astable's output frequency: ranges are
  checked first, and ``frequency`` is no alias of anything - the user is asked which frequency.
* F4 - the circuit report printed "위반 0" for a clearance check that compared nothing (no limit): ``기록 없음``.
* R1 - a regulatory decision that cites no official sentence counted as grounded (``all([])``): it is not.

Offline: the synthetic template library of ``tests/test_circuit_templates.py``, the scripted LLM of
``tests/test_requirement_agent_llm.py``, the fixture board of ``tests/test_routing.py`` and the packaged candidate
list with the quote lookup patched (no archive, no network).
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.design import INPUTS_CHECK, KEY_ALIASES, TEMPLATES, UNIT_OF
from ai_eda.design.base import AMBIGUOUS_KEYS, served_through_specific_key, specific_keys, unserved_requirements
from ai_eda.design.templates import RcLowpassTemplate
from ai_eda.ir import (
    CircuitIR,
    ManufacturingConstraints,
    MissingInformation,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    RequirementSet,
    Track,
    ValidationStatus as S,
    assumption,
    llm_generated,
    user_requirement,
)
from ai_eda.ir.regulatory import GroundedQuote, Jurisdiction, RegulatoryState
from ai_eda.llm.extraction import CONFIRM_KEY
from ai_eda.regulatory import APPLICABILITY_CHECK, load_candidates, research
from ai_eda.report.stages import NO_RECORD, _routing_results, circuit_report
from ai_eda.review import IndependentReviewer
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import CLEARANCE_CHECK, TOOL_ID as ROUTING_TOOL
from ai_eda.workflow import Orchestrator, Stage
from tests.test_circuit_templates import ASTABLE, BASE, _confirm, _present, _run, template_library
from tests.test_requirement_agent_llm import CANNED, RAW, USAGE, _service
from tests.test_routing import board_ir, fixture_library

#: the module, not the ``research`` function ``ai_eda.regulatory`` re-exports under the same name
research_module = importlib.import_module("ai_eda.regulatory.research")

FREQUENCY_KEYS = {"oscillation_frequency", "cutoff_frequency", "clock_frequency"}


def _ir(tmp_path: Path, name: str = "defects", raw: str = "") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    ir.requirements.raw_input = raw
    return ir


# --------------------------------------------------------------------------- F1 missing_information


def test_f1_missing_information_is_not_verified_when_the_request_was_not_analysed(tmp_path: Path):
    # a request text, no --llm: only the baseline checklist was asked (and answered) - nothing read the request
    ir = _ir(tmp_path, raw="900 MHz AM walkie-talkie, 0.5 W, 2 km range")
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=dict(BASE))).run(ir, stop_after=Stage.MISSING_INFORMATION)
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert not state.blocked and out.status is S.NOT_VERIFIED
    assert out.message == (
        "no required question open, but the request was not analysed for missing information (no --llm: only the baseline checklist was asked; "
        "the request's own numbers and terms are not read)"
    )
    assert ir.requirements.missing == []  # nothing recorded: the checklist is the requirement stage's own question list
    # no request text at all
    ir = _ir(tmp_path, "empty")
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=dict(BASE))).run(ir, stop_after=Stage.MISSING_INFORMATION)
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert out.status is S.NOT_VERIFIED and out.message.startswith("no required question open, but there is no request text")
    # an open required question still stops the stage, as before
    ir = _ir(tmp_path, "open", raw="a converter")
    ir.requirements.missing.append(MissingInformation(key="isolation", question="Must the output be isolated?"))
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=dict(BASE))).run(ir, stop_after=Stage.MISSING_INFORMATION)
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert state.blocked and out.status is S.USER_INPUT_REQUIRED and out.message == "1 required question(s)"


def test_f1_missing_information_passes_after_a_confirmed_extraction_of_this_text(tmp_path: Path):
    canned = copy.deepcopy(CANNED)
    canned["requirements"] = [r for r in canned["requirements"] if r["key"] not in ("output_ripple", "system_note")]
    canned["questions"] = []
    svc, client = _service([{"structured": canned, "usage": USAGE}])
    ir = _ir(tmp_path, raw=RAW)
    state = Orchestrator(AgentContext(workdir=tmp_path, llm=svc)).run(ir)
    assert state.blocked and state.current is Stage.REQUIREMENT_ANALYSIS  # the extraction is shown, not yet confirmed
    state = Orchestrator(AgentContext(workdir=tmp_path, llm=svc, answers={CONFIRM_KEY: "yes"})).run(ir, stop_after=Stage.MISSING_INFORMATION)
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert len(client.calls) == 1 and out.status is S.PASS
    assert out.message == (
        "no required question open; the request was analysed by the model and the extraction confirmed by the user (requirements.extraction PASS)"
    )
    # the request text changes (a correction written into the IR) and the next run has no model: the confirmed extraction
    # is about another text, so nothing checked this one
    ir.requirements.corrections.append("the output must be 3.3 V, not 5 V")
    state = Orchestrator(AgentContext(workdir=tmp_path)).run(ir, stop_after=Stage.MISSING_INFORMATION)
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert not state.blocked and out.status is S.NOT_VERIFIED
    assert out.message.startswith("no required question open, but the confirmed request extraction was made for another request text")


def test_f1_an_extraction_that_could_not_run_is_named(tmp_path: Path):
    svc, client = _service([])  # the script is empty: every call fails, the agent falls back to the checklist
    ir = _ir(tmp_path, raw=RAW)
    state = Orchestrator(AgentContext(workdir=tmp_path, llm=svc, answers=dict(BASE))).run(ir, stop_after=Stage.MISSING_INFORMATION)
    assert ir.validation.latest("requirements.extraction").status is S.NOT_VERIFIED
    out = state.outcome(Stage.MISSING_INFORMATION)
    assert out.status is S.NOT_VERIFIED
    assert out.message == "no required question open, but the request extraction is NOT_VERIFIED for this request text: missing information was not checked"


# --------------------------------------------------------------------------- F2 rc_lowpass validity range


def test_f2_rc_lowpass_refuses_900_mhz_and_accepts_its_bounds(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    assert (RcLowpassTemplate.F_MIN, RcLowpassTemplate.F_MAX) == (2.0, 100_000.0)
    for name, cutoff, shown in (("rf", "900 MHz", "900000000"), ("slow", "1 Hz", "1")):
        ir = _ir(tmp_path, name)
        state, _ = _run(ir, tmp_path, lib, {**BASE, "cutoff_frequency": cutoff})
        out = state.outcome(Stage.ARCHITECTURE)
        assert not state.blocked and ir.components == [] and ir.parameters == {}
        assert f"template rc_lowpass not proposed: cutoff frequency {shown} Hz (req.cutoff_frequency) is outside 2..100000 Hz" in out.message
        assert "an RF filter needs an LC or transmission-line design" in out.message and "not measured" in out.message
        assert CONFIRM_DESIGN_KEY not in ir.requirements.presented  # no table was shown
    for name, cutoff in (("low", "2 Hz"), ("high", "100 kHz")):
        ir = _ir(tmp_path, name)
        question = _present(ir, tmp_path, lib, {"cutoff_frequency": cutoff})
        assert question.startswith("Template 'rc_lowpass'")
        _confirm(ir, tmp_path, lib)
        assert [c.ref for c in ir.components] == ["R1", "C1", "J1"]
    # the range is checked on the value that reached the template, so the ideal-model bounds hold at both ends
    ir = _ir(tmp_path, "edge")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "cutoff_frequency": "100.001 kHz"})
    assert "is outside 2..100000 Hz" in state.outcome(Stage.ARCHITECTURE).message and ir.components == []


# --------------------------------------------------------------------------- F3 range first, and the bare key frequency


def test_f3_astable_refuses_an_out_of_range_frequency_before_asking_for_the_supply(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "astable_rf")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "oscillation_frequency": "900 MHz"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and state.open_questions == [] and ir.components == []
    assert "template astable not proposed: oscillation frequency 900000000 Hz (req.oscillation_frequency) is outside 100..20000 Hz" in out.message
    assert "input_voltage" not in [q.key for q in out.questions]
    # an out-of-range supply beside a missing... frequency cannot happen (the frequency selects the template); an out-of-range
    # supply beside an in-range frequency still refuses with the supply sentence
    ir = _ir(tmp_path, "astable_supply")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "oscillation_frequency": "1 kHz", "input_voltage": "12 V"})
    assert "template astable not proposed: supply 12 V (req.input_voltage) is outside 3..6 V" in state.outcome(Stage.ARCHITECTURE).message
    # an in-range frequency and a missing supply: the required question, as before
    ir = _ir(tmp_path, "astable_ask")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "oscillation_frequency": "1 kHz"})
    assert state.blocked and [q.key for q in state.open_questions] == ["input_voltage"]
    assert state.open_questions[0].question.startswith("The BJT astable multivibrator template needs input_voltage in V")
    # the ATmega128 board the same way: a 900 MHz clock refuses before the supply is asked for; 16 MHz asks
    ir = _ir(tmp_path, "mcu_rf")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "clock_frequency": "900 MHz"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == []
    assert "template atmega128_devboard not proposed: clock frequency 900000000 Hz (req.clock_frequency) is outside 1000000..16000000 Hz" in out.message
    ir = _ir(tmp_path, "mcu_supply")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "clock_frequency": "16 MHz", "input_voltage": "24 V"})
    assert "template atmega128_devboard not proposed: supply 24 V (req.input_voltage) is outside 7..15 V" in state.outcome(Stage.ARCHITECTURE).message
    ir = _ir(tmp_path, "mcu_ask")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "clock_frequency": "16 MHz"})
    assert state.blocked and [q.key for q in state.open_questions] == ["input_voltage"]


def test_f3_the_bare_frequency_key_selects_no_template_and_asks_which(tmp_path: Path):
    assert "frequency" not in {a for aliases in KEY_ALIASES.values() for a in aliases} and AMBIGUOUS_KEYS == {"frequency": "Hz"}
    # a subset, never equality: a later Hz key ending in _frequency (an RF carrier) is listed too
    assert FREQUENCY_KEYS <= set(specific_keys("frequency"))
    assert all(UNIT_OF[k] == "Hz" and k.endswith("_frequency") for k in specific_keys("frequency"))
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "bare")
    state, _ = _run(ir, tmp_path, lib, {**BASE, "frequency": "1 kHz"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and ir.simulation is None and CONFIRM_DESIGN_KEY not in ir.requirements.presented
    assert "no template matches the confirmed requirements" in out.message and "template astable" not in out.message
    [q] = [q for q in state.optional_questions if q.key == "frequency"]
    assert not q.required
    assert q.question.startswith("'frequency' does not say which frequency, so no template reads it: state the quantity you mean instead - one of ")
    assert all(k in q.question for k in FREQUENCY_KEYS) and "--answer oscillation_frequency=\"1 kHz\"" in q.question
    assert q.question.endswith("the requirement req.frequency stays as written")
    assert "req.frequency (frequency: '1 kHz') does not say which frequency: no template reads it (state one of " in out.message
    # the requirement itself is left as written
    req = ir.requirements.get("frequency")
    assert req is not None and req.value.value == "1 kHz" and req.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    # a model's unconfirmed 'frequency' is nobody's requirement yet: no question
    ir = _ir(tmp_path, "bare_llm")
    ir.requirements.requirements.append(Requirement(id="req.frequency", key="frequency", text="x", kind=RequirementKind.EXPLICIT, value=llm_generated("1 kHz", "model")))
    state, _ = _run(ir, tmp_path, lib, dict(BASE))
    assert [q for q in state.optional_questions if q.key == "frequency"] == []


def test_f3_a_bare_frequency_with_the_same_specific_value_is_served(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "same")
    question = _present(ir, tmp_path, lib, {**ASTABLE, "frequency": "1000 Hz"})  # 1 kHz stated twice, spelled differently
    assert question.startswith("Template 'astable'")
    astable = next(t for t in TEMPLATES if t.id == "astable")
    assert served_through_specific_key(ir, astable) == {"req.frequency": "oscillation_frequency"} and unserved_requirements(ir, astable) == []
    state, _ = _confirm(ir, tmp_path, lib)
    assert "req.frequency is served as oscillation_frequency (req.oscillation_frequency): the specific key states the same number" in state.outcome(Stage.ARCHITECTURE).message
    served = {rid for c in ir.components for rid in c.serves_requirements} | {rid for n in ir.nets for rid in n.serves_requirements}
    assert {"req.frequency", "req.oscillation_frequency"} <= served
    # every part that serves the specific requirement serves the bare one too, and nothing else changed
    for c in ir.components:
        assert ("req.oscillation_frequency" in c.serves_requirements) == ("req.frequency" in c.serves_requirements)
    # the reviewer's traceability agrees with the template's decision
    r = IndependentReviewer().check_requirements_vs_ir(ir, tmp_path)
    assert r.status is S.PASS and "req.frequency" in r.details["traced"]
    # the design parameters are read from the specific key only
    assert ir.parameters["f_osc"].provenance.derived_from == ["req.oscillation_frequency"]


def test_f3_a_bare_frequency_with_another_value_refuses(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "other")
    state, _ = _run(ir, tmp_path, lib, {**BASE, **ASTABLE, "frequency": "2 kHz"})
    out = state.outcome(Stage.ARCHITECTURE)
    assert not state.blocked and ir.components == [] and CONFIRM_DESIGN_KEY not in ir.requirements.presented
    assert "template astable not proposed: req.frequency (frequency: '2 kHz') is not served by the BJT astable multivibrator template" in out.message
    [q] = [q for q in state.optional_questions if q.key == "frequency"]
    assert not q.required and "is not served by the BJT astable multivibrator template" in q.question
    astable = next(t for t in TEMPLATES if t.id == "astable")
    assert served_through_specific_key(ir, astable) == {} and [r.id for r in unserved_requirements(ir, astable)] == ["req.frequency"]
    # an unreadable bare frequency is not served either (closed world)
    ir = _ir(tmp_path, "unreadable")
    state, _ = _run(ir, tmp_path, lib, {**BASE, **ASTABLE, "frequency": "about 1 kHz or so"})
    assert ir.components == [] and "req.frequency" in state.outcome(Stage.ARCHITECTURE).message


def test_f3_an_astable_built_from_a_bare_frequency_answer_still_rereads_its_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A project built while ``frequency`` was an alias of ``oscillation_frequency`` keeps its design and still re-reads PASS."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "legacy")
    # build it exactly as the code before this change did: 'frequency' in the oscillation_frequency aliases
    monkeypatch.setitem(KEY_ALIASES, "oscillation_frequency", ("oscillation_frequency", "output_frequency", "frequency", "f_osc", "fosc"))
    _present(ir, tmp_path, lib, {"frequency": "1 kHz", "input_voltage": "5 V"})
    _confirm(ir, tmp_path, lib)
    assert ir.parameters["f_osc"].provenance.derived_from == ["req.frequency"]
    saved = tmp_path / "legacy.json"
    ir.save(saved)
    monkeypatch.undo()
    assert "frequency" not in KEY_ALIASES["oscillation_frequency"]
    ir = CircuitIR.load(saved)
    before = ir.content_hash()
    state, _ = _run(ir, tmp_path, lib, {})
    out = state.outcome(Stage.ARCHITECTURE)
    assert f"{INPUTS_CHECK} PASS" in out.message and "templates only start an empty design" in out.message
    assert ir.validation.latest(INPUTS_CHECK).status is S.PASS
    assert [q for q in state.optional_questions if q.key == "frequency"] == []  # a design is present: nothing is asked
    assert ir.content_hash() == before


# --------------------------------------------------------------------------- F4 circuit report: no count for a comparison that did not run


def _clearance_line(md: str) -> str:
    [line] = [ln for ln in md.splitlines() if ln.startswith("  - 한계 ")]
    return line


def test_f4_the_circuit_report_prints_no_record_for_an_uncompared_clearance_check(tmp_path: Path):
    lib = fixture_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 6.0, 5.5)],
                  {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (12.0, 8.0))
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir.pcb.tracks = [Track(net="N", layer="F.Cu", start=(3.0, 3.0), end=(9.0, 3.0), width_mm=0.4, provenance=prov)]

    def validate() -> None:
        ir.validation.extend(default_registry.get(ROUTING_TOOL).validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})))

    # no clearance limit: the validator writes pairs_compared=0 / violations=[] without comparing anything
    validate()
    c = ir.validation.latest(CLEARANCE_CHECK)
    assert c.status is S.NOT_VERIFIED and c.details["pairs_compared"] == 0 and c.details["violations"] == []
    assert _clearance_line(circuit_report(ir, lib, None)) == f"  - 한계 {NO_RECORD}, 비교한 쌍 {NO_RECORD}, 위반 {NO_RECORD}"
    # a limit: the comparison ran, the counts are the validator's
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(0.25, note="test limit"))
    validate()
    c = ir.validation.latest(CLEARANCE_CHECK)
    assert c.status is S.PASS and c.details["pairs_compared"] == 1
    assert _clearance_line(circuit_report(ir, lib, None)) == "  - 한계 0.25 mm, 비교한 쌍 1, 위반 0"
    # copper the compiler refuses: FAIL before any comparison, so no count either
    ir.pcb.tracks = [Track(net="N", layer="F.Cu", start=(3.0, 3.0), end=(9.0, 3.0), width_mm=-1.0, provenance=prov)]
    validate()
    c = ir.validation.latest(CLEARANCE_CHECK)
    assert c.status is S.FAIL and c.details["malformed"] and c.details["pairs_compared"] == 0
    # (the whole circuit report is not built here: its IPC-2221 section is not part of this check)
    assert _clearance_line("\n".join(_routing_results(ir))) == f"  - 한계 0.25 mm, 비교한 쌍 {NO_RECORD}, 위반 {NO_RECORD}"


# --------------------------------------------------------------------------- R1 empty evidence is not grounded


@pytest.fixture
def every_quote_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every curated quote counts as found in its archived text (no archive, no network): only the citation rule is left to judge."""

    def found(candidate, docs):
        return [GroundedQuote(section=q.section, quote=q.quote, found=True, page=1) for q in candidate.grounding_quotes]

    monkeypatch.setattr(research_module, "_ground_quotes", found)


def _reqs(volts: float = 12.0) -> RequirementSet:
    return RequirementSet(requirements=[Requirement(id="req.input_voltage", key="input_voltage", text="12 V DC input", kind=RequirementKind.EXPLICIT,
                                                    value=user_requirement(volts, "V"))])


def _research(code: str, answers: dict[str, str]):
    return research(RegulatoryState(jurisdictions=[Jurisdiction(code=code, name=code)]), load_candidates(), None, answers, _reqs())


def test_r1_an_exclusion_without_evidence_is_not_grounded(every_quote_found):
    out = _research("EU", {"radio": "yes", "evaluation_kit": "no", "finished_apparatus": "yes"})
    by = {r.candidate_id: r for r in out.candidates}
    lvd, emc = by["reg.EU.LVD.2014-35-EU"], by["reg.EU.EMC.2014-30-EU"]
    # radio equipment is excluded from the LVD / EMC by a rule that names no section: nothing to ground it in
    assert lvd.evaluation.applicability.value == "not_applicable" and lvd.evaluation.evidence == [] and lvd.evidence_grounded is False
    assert emc.evaluation.evidence == [] and emc.evidence_grounded is False
    # decisions that cite a sentence the (patched) archive found stay grounded
    assert by["reg.EU.RED.2014-53-EU"].evidence_grounded and by["reg.EU.RoHS.2011-65-EU"].evidence_grounded
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is S.NOT_VERIFIED
    assert "the decision cites no official sentence for reg.EU.LVD.2014-35-EU, reg.EU.EMC.2014-30-EU" in app.message
    assert "were not grounded for" not in app.message  # every cited quote was found: only the uncited decisions are named
    rows = {c["id"]: c["evidence_grounded"] for c in app.details["candidates"]}
    assert rows == {"reg.EU.LVD.2014-35-EU": False, "reg.EU.EMC.2014-30-EU": False, "reg.EU.RoHS.2011-65-EU": True, "reg.EU.RED.2014-53-EU": True}


def test_r1_a_kr_radio_decision_names_the_uncited_notice(every_quote_found):
    """The KR 고시 entry cites no sentence on either branch: every decided KR run is NOT_VERIFIED naming it (intended, not a regression)."""
    out = _research("KR", {"radio": "yes", "digital_device": "no"})
    by = {r.candidate_id: r for r in out.candidates}
    notice = by["reg.KR.RadioWavesAct.ConformityAssessmentNotice"]
    assert notice.evaluation.applicability.value == "applicable" and notice.evaluation.evidence == [] and not notice.evidence_grounded
    assert by["reg.KR.RadioWavesAct.58-2"].evidence_grounded
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is S.NOT_VERIFIED and "cites no official sentence for reg.KR.RadioWavesAct.ConformityAssessmentNotice" in app.message


def test_r1_decisions_that_all_cite_a_found_sentence_still_pass(every_quote_found):
    out = _research("EU", {"radio": "no", "evaluation_kit": "no", "finished_apparatus": "yes", "highest_rated_voltage": "12 V DC"})
    assert out.undecided == {}
    assert all(r.evidence_grounded for r in out.candidates), [(r.candidate_id, r.evaluation.evidence) for r in out.candidates]
    assert out.result(APPLICABILITY_CHECK).status is S.PASS
