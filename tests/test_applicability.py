"""Deterministic applicability: the table the design pins, plus the three-valued logic and the voltage reading rules."""

from __future__ import annotations

import pytest

from ai_eda.ir import Requirement, RequirementKind, RequirementSet, SourceRef, Traced, ValidationStatus, assumption, authoritative, llm_generated, user_requirement
from ai_eda.ir.regulatory import Applicability
from ai_eda.regulatory.applicability import MAINS_KEY, evaluate, yes_no
from ai_eda.regulatory.candidates import ApplicabilityRule, load_candidates

LVD = ApplicabilityRule.model_validate({
    "kind": "all_of",
    "rules": [
        {"kind": "not", "rule": {"kind": "answer", "key": "radio", "equals": "yes"}},
        {"kind": "voltage_range", "requirement": "input_voltage", "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"},
    ],
})
RED = ApplicabilityRule.model_validate({"kind": "answer", "key": "radio", "equals": "yes", "evidence": "Article 2(1)(1)"})
VOLT = ApplicabilityRule.model_validate({"kind": "voltage_range", "requirement": "input_voltage", "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"})


def _req(value, unit: str | None = "V", text: str = "input voltage", key: str = "input_voltage") -> Requirement:
    traced = user_requirement(value, unit) if not isinstance(value, Traced) else value
    return Requirement(id=f"req.{key}", key=key, text=text, kind=RequirementKind.EXPLICIT, value=traced)


def _reqs(*reqs: Requirement) -> RequirementSet:
    return RequirementSet(requirements=list(reqs))


# --------------------------------------------------------------------------- the design's table


def test_12v_dc_input_is_outside_the_lvd_with_the_article_cited():
    ev = evaluate(LVD, {MAINS_KEY: "no", "radio": "no"}, _reqs(_req(12.0)))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.status is ValidationStatus.NOT_APPLICABLE
    assert ev.inputs_used == {"input_voltage": "12 V DC (requirement req.input_voltage; DC because mains_powered=no)", "radio": "no (answer)"}
    assert "outside the 75-1500 V DC band stated in Article 1" in ev.rationale
    assert ev.evidence == ["Article 1"] and ev.missing == []


def test_12v_dc_is_outside_the_lvd_even_when_radio_is_unanswered():
    # all_of with one NOT_APPLICABLE member is decided whatever the others say (Kleene): no question is asked needlessly
    ev = evaluate(LVD, {}, _reqs(_req(12.0, text="12 V DC input on the header")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.missing == []
    assert ev.inputs_used["input_voltage"] == "12 V DC (requirement req.input_voltage; DC stated with the value)"
    assert ev.evidence == ["Article 1"]


def test_230v_ac_mains_is_inside_the_lvd():
    ev = evaluate(LVD, {"radio": "no", MAINS_KEY: "yes"}, _reqs(_req(230.0)))
    assert ev.applicability is Applicability.APPLICABLE and ev.status is ValidationStatus.NOT_VERIFIED  # applies; compliance not verified
    assert "230 V AC" in ev.inputs_used["input_voltage"] and "inside the 50-1000 V AC band" in ev.rationale


def test_radio_yes_makes_red_applicable_and_excludes_the_lvd_regardless_of_voltage():
    assert evaluate(RED, {"radio": "yes"}).applicability is Applicability.APPLICABLE
    assert evaluate(RED, {"radio": "no"}).applicability is Applicability.NOT_APPLICABLE
    ev = evaluate(LVD, {"radio": "yes"}, _reqs(_req(230.0, text="230 V AC")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.missing == []
    assert ev.evidence == []  # the decisive member (the radio exclusion) cites no quote; Article 1 did not decide


def test_missing_mains_answer_is_user_input_required_naming_the_key():
    ev = evaluate(VOLT, {}, _reqs(_req(230.0)))
    assert ev.applicability is Applicability.UNDECIDED and ev.status is ValidationStatus.USER_INPUT_REQUIRED
    assert ev.missing_keys == [MAINS_KEY] and "does not say AC or DC" in ev.missing[0].reason
    assert ev.inputs_used["input_voltage"].startswith("230 V (requirement req.input_voltage")


def test_missing_input_voltage_is_user_input_required_naming_the_requirement_key():
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs())
    assert ev.status is ValidationStatus.USER_INPUT_REQUIRED and ev.missing_keys == ["input_voltage"]
    ev = evaluate(RED, {})
    assert ev.status is ValidationStatus.USER_INPUT_REQUIRED and ev.missing_keys == ["radio"]


def test_all_of_with_undecided_and_applicable_members_is_undecided_and_names_only_the_missing_keys():
    ev = evaluate(LVD, {MAINS_KEY: "yes"}, _reqs(_req(230.0)))
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == ["radio"]


# --------------------------------------------------------------------------- voltage reading


def test_voltage_from_a_typed_answer_string_with_ac_dc():
    ev = evaluate(VOLT, {"input_voltage": "12 V DC"})
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.inputs_used["input_voltage"] == "12 V DC (answer input_voltage; DC stated with the value)"
    ev = evaluate(VOLT, {"input_voltage": "230VAC"})
    assert ev.applicability is Applicability.APPLICABLE
    ev = evaluate(VOLT, {"input_voltage": "12"})
    assert ev.status is ValidationStatus.USER_INPUT_REQUIRED and "not one unambiguous quantity with a unit" in ev.missing[0].reason
    ev = evaluate(VOLT, {"input_voltage": "2 A", MAINS_KEY: "no"})
    assert ev.missing_keys == ["input_voltage"] and "is a A quantity, not a voltage" in ev.missing[0].reason


def test_voltage_from_a_string_requirement_and_korean_current_words():
    ev = evaluate(VOLT, {}, _reqs(_req(user_requirement("12V DC"), text="input_voltage: 12V DC")))
    assert ev.applicability is Applicability.NOT_APPLICABLE
    ev = evaluate(VOLT, {}, _reqs(_req(220.0, text="220 V 교류 입력")))
    assert ev.applicability is Applicability.APPLICABLE and "AC stated with the value" in ev.inputs_used["input_voltage"]
    ev = evaluate(VOLT, {}, _reqs(_req(48.0, text="48 V 직류")))
    assert ev.applicability is Applicability.NOT_APPLICABLE


def test_voltage_units_are_scaled_and_non_voltages_refused():
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(0.4, "kV")))
    assert ev.applicability is Applicability.APPLICABLE and ev.inputs_used["input_voltage"].startswith("400 V DC")
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(12000.0, "mV")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.inputs_used["input_voltage"].startswith("12 V DC")
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(2.0, "A")))
    assert ev.missing_keys == ["input_voltage"] and "unit 'A' is not a voltage" in ev.missing[0].reason
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(12.0, None)))
    assert ev.missing_keys == ["input_voltage"] and "no unit" in ev.missing[0].reason


def test_voltage_ranges_overlap_the_band():
    ev = evaluate(VOLT, {MAINS_KEY: "yes"}, _reqs(_req([100.0, 240.0])))
    assert ev.applicability is Applicability.APPLICABLE and ev.inputs_used["input_voltage"].startswith("100..240 V AC")
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req([9.0, 36.0])))
    assert ev.applicability is Applicability.NOT_APPLICABLE
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req([60.0, 90.0])))
    assert ev.applicability is Applicability.APPLICABLE  # partly inside 75-1500 V DC
    ev = evaluate(VOLT, {"input_voltage": "3.3-5 V", MAINS_KEY: "no"})
    assert ev.applicability is Applicability.NOT_APPLICABLE


def test_a_mains_answer_that_contradicts_the_stated_kind_is_a_missing_input_never_obeyed_either_way():
    ev = evaluate(VOLT, {MAINS_KEY: "yes"}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.UNDECIDED and ev.status is ValidationStatus.USER_INPUT_REQUIRED
    assert ev.missing_keys == [MAINS_KEY] and "says DC" in ev.missing[0].reason and "mains_powered=yes says AC mains" in ev.missing[0].reason and "contradict" in ev.missing[0].reason
    assert ev.inputs_used["input_voltage"] == "12 V (requirement req.input_voltage; DC stated with the value)"  # no kind decided
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.inputs_used["input_voltage"] == "12 V DC (requirement req.input_voltage; DC stated with the value)"
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(230.0, text="230 V AC")))
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == [MAINS_KEY]  # AC stated, mains says DC: neither wins


def test_ac_dc_is_read_next_to_the_unit_only_never_from_prose_or_the_provenance_note():
    ev = evaluate(VOLT, {}, _reqs(_req(12.0, text="12 V DC from an AC adapter")))  # "AC adapter" is prose, "V DC" is the rating
    assert ev.applicability is Applicability.NOT_APPLICABLE and "DC stated with the value" in ev.inputs_used["input_voltage"]
    ev = evaluate(VOLT, {}, _reqs(_req(12.0, text="12 V from an AC adapter")))
    assert ev.status is ValidationStatus.USER_INPUT_REQUIRED and ev.missing_keys == [MAINS_KEY]
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(12.0, text="12 V from an AC adapter")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and "DC because mains_powered=no" in ev.inputs_used["input_voltage"]
    assert evaluate(VOLT, {"input_voltage": "230VAC"}).applicability is Applicability.APPLICABLE
    assert evaluate(VOLT, {"input_voltage": "48 Vdc"}).applicability is Applicability.NOT_APPLICABLE
    # the provenance note carries the model's own sentence: it is never read
    said = user_requirement(230.0, "V", note="quote: '230V'; parsed: 230 V; model: fake-model; model statement: 'The input is 230 V DC'")
    ev = evaluate(VOLT, {MAINS_KEY: "yes"}, _reqs(_req(said, text="input_voltage: 230V")))
    assert ev.applicability is Applicability.APPLICABLE and "AC because mains_powered=yes" in ev.inputs_used["input_voltage"]


def test_an_untrusted_requirement_is_a_missing_input_not_a_decision():
    # a model extraction the user has not confirmed, a model assumption or a derived figure never decides scope
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(llm_generated(12.0, "m", "V"))))
    assert ev.applicability is Applicability.UNDECIDED and ev.status is ValidationStatus.USER_INPUT_REQUIRED and ev.missing_keys == ["input_voltage"]
    assert "not confirmed" in ev.missing[0].reason and "confirm_requirements" in ev.missing[0].reason and "answer input_voltage directly" in ev.missing[0].reason
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, [_req(assumption(230.0, "guess", "V"))])
    assert ev.applicability is Applicability.UNDECIDED and "model assumption" in ev.missing[0].reason
    # a direct answer under the key settles it; a trusted requirement is read as before
    ev = evaluate(VOLT, {MAINS_KEY: "no", "input_voltage": "12 V DC"}, [_req(assumption(230.0, "guess", "V"))])
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.inputs_used["input_voltage"].startswith("12 V DC (answer input_voltage")
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, {"input_voltage": _req(12.0)})
    assert ev.applicability is Applicability.NOT_APPLICABLE
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_req(authoritative(12.0, SourceRef(title="ds"), "V"))))
    assert ev.applicability is Applicability.NOT_APPLICABLE


def test_a_yes_no_question_needs_a_yes_no_answer():
    ev = evaluate(RED, {"radio": "wifi and bluetooth"})
    assert ev.applicability is Applicability.UNDECIDED and ev.status is ValidationStatus.USER_INPUT_REQUIRED
    assert ev.missing_keys == ["radio"] and "is not yes or no" in ev.missing[0].reason and ev.inputs_used == {"radio": "wifi and bluetooth (answer, not understood)"}
    lvd = evaluate(LVD, {"radio": "wifi and bluetooth", MAINS_KEY: "yes"}, _reqs(_req(230.0)))
    assert lvd.applicability is Applicability.UNDECIDED and lvd.missing_keys == ["radio"]  # not(radio) can not decide either
    assert evaluate(RED, {"radio": "Yes!"}).applicability is Applicability.APPLICABLE and evaluate(RED, {"radio": "아니오"}).applicability is Applicability.NOT_APPLICABLE


def test_the_equipment_is_rated_by_every_voltage_it_names_and_the_users_highest_voltage():
    rule = ApplicabilityRule.model_validate({"kind": "voltage_range", "requirement": "input_voltage", "requirements": ["output_voltage"], "scope_answer": "highest_rated_voltage",
                                             "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"})
    assert rule.answer_keys() == [MAINS_KEY, "highest_rated_voltage"] and rule.requirement_keys() == ["input_voltage", "output_voltage"]
    boost = _reqs(_req(12.0, text="12 V DC input"), _req(400.0, text="400 V DC output", key="output_voltage"))
    ev = evaluate(rule, {}, boost)
    assert ev.applicability is Applicability.APPLICABLE and "output_voltage = 400 V DC" in ev.rationale and "is inside the 75-1500 V DC band" in ev.rationale
    assert ev.inputs_used == {"input_voltage": "12 V DC (requirement req.input_voltage; DC stated with the value)", "output_voltage": "400 V DC (requirement req.output_voltage; DC stated with the value)"}
    # input alone outside the band: not decided until the user states the product's highest voltage
    ev = evaluate(rule, {}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == ["highest_rated_voltage"] and "highest voltage anywhere in the product" in ev.missing[0].reason
    ev = evaluate(rule, {"highest_rated_voltage": "12 V DC"}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.inputs_used["highest_rated_voltage"] == "12 V DC (answer highest_rated_voltage; DC stated with the value)"
    ev = evaluate(rule, {"highest_rated_voltage": "400 V DC"}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.APPLICABLE and "highest_rated_voltage = 400 V DC" in ev.rationale
    ev = evaluate(rule, {"highest_rated_voltage": "high"}, _reqs(_req(12.0, text="12 V DC input")))
    assert ev.applicability is Applicability.UNDECIDED and "can not be read as a voltage" in ev.missing[0].reason
    # an untrusted extra requirement is reported, never read
    ev = evaluate(rule, {"highest_rated_voltage": "12 V DC"}, _reqs(_req(12.0, text="12 V DC input"), _req(llm_generated(400.0, "m", "V"), key="output_voltage")))
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == ["output_voltage"]


# --------------------------------------------------------------------------- logic


def _ans(key: str, equals="yes") -> dict:
    return {"kind": "answer", "key": key, "equals": equals}


@pytest.mark.parametrize(
    "rule, answers, expected",
    [
        ({"kind": "always"}, {}, Applicability.APPLICABLE),
        ({"kind": "never"}, {}, Applicability.NOT_APPLICABLE),
        ({"kind": "not", "rule": {"kind": "always"}}, {}, Applicability.NOT_APPLICABLE),
        ({"kind": "not", "rule": _ans("radio")}, {}, Applicability.UNDECIDED),
        ({"kind": "any_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"radio": "no"}, Applicability.UNDECIDED),
        ({"kind": "any_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"radio": "no", "digital_device": "no"}, Applicability.NOT_APPLICABLE),
        ({"kind": "any_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"digital_device": "yes"}, Applicability.APPLICABLE),
        ({"kind": "all_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"radio": "yes"}, Applicability.UNDECIDED),
        ({"kind": "all_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"radio": "yes", "digital_device": "yes"}, Applicability.APPLICABLE),
        ({"kind": "all_of", "rules": [_ans("radio"), _ans("digital_device")]}, {"digital_device": "no"}, Applicability.NOT_APPLICABLE),
        (_ans("kind", ["consumer", "professional"]), {"kind": "Professional."}, Applicability.APPLICABLE),
        (_ans("kind", ["consumer", "professional"]), {"kind": "military"}, Applicability.NOT_APPLICABLE),
    ],
)
def test_three_valued_logic(rule: dict, answers: dict, expected: Applicability):
    assert evaluate(ApplicabilityRule.model_validate(rule), answers).applicability is expected


def test_any_of_evidence_is_the_deciding_members():
    rule = ApplicabilityRule.model_validate({"kind": "any_of", "rules": [_ans("radio") | {"evidence": "o"}, _ans("digital_device") | {"evidence": "k"}]})
    ev = evaluate(rule, {"radio": "no", "digital_device": "yes"})
    assert ev.applicability is Applicability.APPLICABLE and ev.evidence == ["k"]
    ev = evaluate(rule, {"radio": "no", "digital_device": "no"})
    assert ev.applicability is Applicability.NOT_APPLICABLE and ev.evidence == ["o", "k"]
    ev = evaluate(rule, {"radio": "no"})
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == ["digital_device"] and ev.evidence == ["o", "k"]


@pytest.mark.parametrize("answer, expected", [("yes", "yes"), ("Y", "yes"), ("네", "yes"), ("예.", "yes"), ("true", "yes"), ("no", "no"), ("아니오", "no"), ("N", "no"),
                                              ("false", "no"), ("maybe", None), ("", None), (None, None), ("12 V", None)])
def test_yes_no(answer, expected):
    assert yes_no(answer) == expected


def test_blank_answer_counts_as_missing_and_evaluation_is_deterministic():
    assert evaluate(RED, {"radio": "   "}).missing_keys == ["radio"]
    a = evaluate(LVD, {"radio": "no", MAINS_KEY: "no"}, _reqs(_req(12.0)))
    b = evaluate(LVD, {"radio": "no", MAINS_KEY: "no"}, _reqs(_req(12.0)))
    assert a == b


def test_packaged_rules_decide_the_documented_scenarios():
    cl = load_candidates()
    reqs = _reqs(_req(12.0, text="12 V DC input"))
    answers = {"radio": "no", "finished_apparatus": "yes", "digital_device": "no", MAINS_KEY: "no", "intended_use": "bench tool", "evaluation_kit": "no",
               "highest_rated_voltage": "12 V DC"}
    decisions = {c.id: evaluate(c.applicability_rule, answers, reqs).applicability for c in cl.candidates}
    assert decisions == {
        "reg.EU.LVD.2014-35-EU": Applicability.NOT_APPLICABLE,
        "reg.EU.EMC.2014-30-EU": Applicability.APPLICABLE,
        "reg.EU.RoHS.2011-65-EU": Applicability.APPLICABLE,
        "reg.EU.RED.2014-53-EU": Applicability.NOT_APPLICABLE,
        "reg.KR.ElectricalAppliancesSafetyAct": Applicability.APPLICABLE,
        "reg.KR.ElectricalAppliancesSafetyAct.EnforcementRule": Applicability.APPLICABLE,
        "reg.KR.RadioWavesAct.58-2": Applicability.NOT_APPLICABLE,
        "reg.KR.RadioWavesAct.ConformityAssessmentNotice": Applicability.NOT_APPLICABLE,
        "reg.US.FCC.47CFR15": Applicability.NOT_APPLICABLE,
    }
    mains = {c.id: evaluate(c.applicability_rule, {**answers, MAINS_KEY: "yes", "radio": "yes", "digital_device": "yes"}, _reqs(_req(230.0))).applicability for c in cl.candidates}
    assert mains["reg.EU.LVD.2014-35-EU"] is Applicability.NOT_APPLICABLE  # radio equipment: RED carries the safety objectives
    assert mains["reg.EU.RED.2014-53-EU"] is Applicability.APPLICABLE and mains["reg.EU.EMC.2014-30-EU"] is Applicability.NOT_APPLICABLE
    assert mains["reg.US.FCC.47CFR15"] is Applicability.APPLICABLE and mains["reg.KR.RadioWavesAct.58-2"] is Applicability.APPLICABLE
    assert mains["reg.KR.RadioWavesAct.ConformityAssessmentNotice"] is Applicability.APPLICABLE
    no_radio_mains = evaluate(cl.get("reg.EU.LVD.2014-35-EU").applicability_rule, {**answers, MAINS_KEY: "yes"}, _reqs(_req(230.0)))
    assert no_radio_mains.applicability is Applicability.APPLICABLE and no_radio_mains.evidence == ["Annex II (evaluation kits)", "Article 1"]
    # the input rail alone never says NOT_APPLICABLE: the product's highest voltage is the user's statement; a 12 V-in / 400 V-out design is in scope
    no_highest = evaluate(cl.get("reg.EU.LVD.2014-35-EU").applicability_rule, {k: v for k, v in answers.items() if k != "highest_rated_voltage"}, reqs)
    assert no_highest.applicability is Applicability.UNDECIDED and no_highest.missing_keys == ["highest_rated_voltage"]
    boost = evaluate(cl.get("reg.EU.LVD.2014-35-EU").applicability_rule, {**answers, "highest_rated_voltage": "400 V DC"}, reqs)
    assert boost.applicability is Applicability.APPLICABLE and "highest_rated_voltage = 400 V DC" in boost.rationale
    kit = evaluate(cl.get("reg.EU.LVD.2014-35-EU").applicability_rule, {**answers, "evaluation_kit": "yes"}, reqs)
    assert kit.applicability is Applicability.NOT_APPLICABLE and "Annex II (evaluation kits)" in kit.evidence
    assert evaluate(cl.get("reg.EU.EMC.2014-30-EU").applicability_rule, {**answers, "evaluation_kit": "yes"}, reqs).applicability is Applicability.NOT_APPLICABLE
    undecided = {c.id: evaluate(c.applicability_rule, {}, _reqs()).missing_keys for c in cl.for_jurisdiction("EU")}
    assert undecided == {"reg.EU.LVD.2014-35-EU": ["radio", "evaluation_kit", "input_voltage"], "reg.EU.EMC.2014-30-EU": ["radio", "evaluation_kit", "finished_apparatus"],
                         "reg.EU.RoHS.2011-65-EU": [], "reg.EU.RED.2014-53-EU": ["radio"]}
