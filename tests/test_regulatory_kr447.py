"""The KR 447 MHz licence-free walkie-talkie entries of the packaged regulatory candidate list.

Six KR entries and the ``kr_licence_free_class`` scope question were added on 2026-09-28 offline, from general
knowledge (law.go.kr was blocked): nothing of them was fetched, so every entry is ``fetchable: false`` with its reason,
no URL and no quote, and every number in their prose is UNVERIFIED. What is proven here:

* the packaged list loads with them (15 candidates; the question is asked on KR runs only), and the nine entries of the
  2026-09-23 probe, the earlier scope questions, the placeholders and the ``not_included`` rows are the curated ones
  unchanged (pinned canonical-JSON hashes);
* each new entry guesses neither a URL nor a quote, says why, marks its summary UNVERIFIED, reads only asked questions
  and cites no section (there is nothing to ground a citation in);
* applicability: the demo scope answers make every KR entry applicable; ``radio=no`` decides all six NOT_APPLICABLE
  without the class answer; a radio without it leaves the licence-exempt pair UNDECIDED naming the key; an answer that
  is not yes/no stays undecided;
* research, offline and against the fake official sites: nothing is fetched for them (no request at all, not even to
  the allow-listed rra.go.kr), each is ``unfetchable`` / NOT_VERIFIED with no URL, hash or quote; even when every quote
  of the fetchable entries is found, a decided KR radio run's ``regulatory.applicability`` is NOT_VERIFIED naming them -
  never PASS - and ``regulatory.compliance`` stays NOT_VERIFIED;
* the agents: a KR run asks ``kr_licence_free_class`` as a non-blocking question and remembers the answer as a scope
  answer; typed as ``--answer`` it becomes a ``regulatory`` requirement, never a design category that a template's closed
  world could refuse.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, RegulatoryAgent, RequirementAgent
from ai_eda.agents.requirement import regulatory_scope_keys
from ai_eda.design.base import DESIGN_CATEGORIES
from ai_eda.ir import CircuitIR, Jurisdiction, ProjectMeta, Requirement, RequirementKind, RequirementSet, ValidationStatus, user_requirement
from ai_eda.ir.regulatory import Applicability, GroundedQuote, RegulatoryState
from ai_eda.regulatory import (
    APPLICABILITY_CHECK,
    COMPLIANCE_CHECK,
    DEFAULT_CANDIDATES_PATH,
    RESEARCH_CHECK,
    SOURCES_CHECK,
    evaluate,
    load_candidates,
    research,
)
from ai_eda.regulatory.candidates import UNVERIFIED_NOTE
from ai_eda.workflow import Orchestrator
from tests.fake_sources import FakeSources
from tests.test_regulatory_research import offline_archive, online_archive

research_module = importlib.import_module("ai_eda.regulatory.research")

CLASS_KEY = "kr_licence_free_class"
NOTICE = "reg.KR.MSIT.LicenceExemptRadioStationEquipmentNotice"
RULES = "reg.KR.MSIT.RadioEquipmentRules"
STATIONS = "reg.KR.RadioWavesAct.LicenceExemptStations"
ALLOCATION = "reg.KR.MSIT.FrequencyAllocationTable"
EMF = "reg.KR.MSIT.HumanEMFProtectionStandard"
TEST_METHODS = "reg.KR.RRA.ConformityTestMethods"
#: the new entries in file order; the first two read the class answer, the rest only ``radio``
NEW_IDS = [NOTICE, RULES, STATIONS, ALLOCATION, EMF, TEST_METHODS]
CLASS_IDS = {NOTICE, STATIONS}
#: the four KR entries of the 2026-09-23 probe, in file order
OLD_KR_IDS = ["reg.KR.ElectricalAppliancesSafetyAct", "reg.KR.ElectricalAppliancesSafetyAct.EnforcementRule", "reg.KR.RadioWavesAct.58-2",
              "reg.KR.RadioWavesAct.ConformityAssessmentNotice"]

#: sha256 of each 2026-09-23 entry as canonical JSON (sorted keys, no whitespace) - the additions left them untouched
PROBE_ENTRY_SHA256 = {
    "reg.EU.LVD.2014-35-EU": "96b82db81b32d9c2374222198c135d5f5f2d6b5e45bed3370af87ac5d80765b1",
    "reg.EU.EMC.2014-30-EU": "34c3fcadf3ba4ff739595107c3892d4212c654be03f3854270e4fc892e2b0f1a",
    "reg.EU.RoHS.2011-65-EU": "0f391d415acebd7b734ad90d7f5da9cd20528a737f31909f1bcf078e879f2d18",
    "reg.EU.RED.2014-53-EU": "1fa3919b8cb92e989f9749120fd9cf736cfcf96b63d0a17c35dec23c75607d53",
    "reg.KR.ElectricalAppliancesSafetyAct": "b2894587dce6197ea148fa2d70d1d023faf93739f7da0a71fa998578d3ede14b",
    "reg.KR.ElectricalAppliancesSafetyAct.EnforcementRule": "860fa25ee6d739e70d6f190e0abfeba020f2f22cf25544627fa8a8000055820e",
    "reg.KR.RadioWavesAct.58-2": "1e04ba920ce603c984a9afc24424ad220ee1e56b8df8d36b87bd66ec64470732",
    "reg.KR.RadioWavesAct.ConformityAssessmentNotice": "8139c2aec80507be6aabacd99a2e46e9938d57cc16505fb8cd3c9d9762b58013",
    "reg.US.FCC.47CFR15": "38b637a174514881ad56a5b02b9713bdb9f8f2953fe707d669b75d5d13b09d49",
}
#: the seven scope questions of 2026-09-23, the placeholders and the not_included rows, hashed the same way
PROBE_QUESTIONS_SHA256 = "689becb031277ff23046b9f1966a96107c0427a79abfe5d7840e66b67e3746a4"
PROBE_PLACEHOLDERS_SHA256 = "5b045149d50230856ec9bdedf3d3f13f5b30d90e8de05aafbab2f2bd4dad9940"
PROBE_NOT_INCLUDED_SHA256 = "825ed27af34addacd01a13ff295acef6b1ad8f3ba244313248b064438fd057cc"

#: the scope answers of the kr447 demo runs (a 2S Li-ion pack, 8.4 V DC at most)
DEMO_ANSWERS = {"radio": "yes", CLASS_KEY: "yes", "digital_device": "yes", "mains_powered": "no", "highest_rated_voltage": "8.4 V DC",
                "intended_use": "licence-free 447 MHz FM walkie-talkie (bench boards first)"}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _raw() -> dict:
    return json.loads(DEFAULT_CANDIDATES_PATH.read_text(encoding="utf-8"))


def _reqs(volts: float = 7.4) -> RequirementSet:
    return RequirementSet(requirements=[Requirement(id="req.input_voltage", key="input_voltage", text=f"{volts} V DC battery (2S Li-ion)",
                                                    kind=RequirementKind.EXPLICIT, value=user_requirement(volts, "V"))])


def _kr() -> RegulatoryState:
    return RegulatoryState(jurisdictions=[Jurisdiction(code="KR", name="KR")])


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


@pytest.fixture
def every_quote_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every curated quote counts as found in its archived text (no archive, no network): only the citation rule is left to judge."""

    def found(candidate, docs):
        return [GroundedQuote(section=q.section, quote=q.quote, found=True, page=1) for q in candidate.grounding_quotes]

    monkeypatch.setattr(research_module, "_ground_quotes", found)


# --------------------------------------------------------------------------- the packaged list


def test_packaged_list_carries_the_kr447_entries_and_the_class_question():
    cl = load_candidates()
    assert len(cl.candidates) == 15 and cl.jurisdictions() == ["EU", "KR", "US"]
    assert [c.id for c in cl.for_jurisdiction("KR")] == OLD_KR_IDS + NEW_IDS
    q = next(q for q in cl.scope_questions if q.key == CLASS_KEY)
    assert q.options == ["yes", "no"] and q.jurisdictions == ["KR"] and "(yes/no)" in q.question and "447 MHz" in q.question
    assert "unverified" in q.rationale
    # asked on KR runs only, after the earlier questions; EU / US runs never see it
    assert [x.key for x in cl.questions_for(["KR"])] == ["intended_use", "mains_powered", "highest_rated_voltage", "radio", "digital_device", CLASS_KEY]
    assert CLASS_KEY not in {x.key for x in cl.questions_for(["EU"])} | {x.key for x in cl.questions_for(["US"])}
    # the list still says what it is, and when and how the additions were made
    assert UNVERIFIED_NOTE in cl.provenance and "2026-09-28 offline from general knowledge, with no URL and no quote" in cl.provenance
    assert all(i in cl.provenance for i in NEW_IDS) and CLASS_KEY in cl.provenance and cl.curated_at == "2026-09-28"


def test_the_2026_09_23_probe_entries_questions_and_rows_are_unchanged():
    raw = _raw()
    by_id = {c["id"]: c for c in raw["candidates"]}
    assert {i: _canonical_sha256(by_id[i]) for i in PROBE_ENTRY_SHA256} == PROBE_ENTRY_SHA256
    assert _canonical_sha256(raw["scope_questions"][:7]) == PROBE_QUESTIONS_SHA256 and [q["key"] for q in raw["scope_questions"][7:]] == [CLASS_KEY]
    assert _canonical_sha256(raw["placeholders"]) == PROBE_PLACEHOLDERS_SHA256 and _canonical_sha256(raw["not_included"]) == PROBE_NOT_INCLUDED_SHA256
    # the additions sit together after the existing KR entries; the file order of the probe's entries is kept
    assert [c["id"] for c in raw["candidates"]] == [*list(PROBE_ENTRY_SHA256)[:8], *NEW_IDS, "reg.US.FCC.47CFR15"]


@pytest.mark.parametrize("cid", NEW_IDS)
def test_each_new_entry_is_an_honest_unfetchable_pointer(cid: str):
    cl = load_candidates()
    c = cl.get(cid)
    assert c is not None and c.jurisdiction == "KR"
    # nothing was fetched: no URL, no document, no quote, no marker, a reason that says why - a URL is never guessed
    assert not c.fetchable and c.official_url is None and c.documents() == [] and c.grounding_quotes == [] and c.expected_markers == []
    assert c.unfetchable_reason and ("blocked" in c.unfetchable_reason or "no network" in c.unfetchable_reason)
    assert ("never guessed" in c.unfetchable_reason) or ("not known offline" in c.unfetchable_reason)
    # the prose says it is unverified; the curation note says the entry was added without URL or quotes on purpose
    assert "UNVERIFIED" in c.summary and c.engineering_implication
    assert c.curation_note and "without URL or quotes on purpose" in c.curation_note
    # the rule reads only asked questions, and cites no section: there is no quote to ground a citation in
    assert set(c.applicability_rule.answer_keys()) <= {"radio", CLASS_KEY} and c.applicability_rule.evidence_labels() == []
    assert (CLASS_KEY in c.applicability_rule.answer_keys()) is (cid in CLASS_IDS)
    assert c.applicability_rule.requirement_keys() == []
    # only official domains: law.go.kr for every entry, rra.go.kr beside it for the RRA test methods
    assert c.allowed_domains == (["www.law.go.kr", "www.rra.go.kr"] if cid == TEST_METHODS else ["www.law.go.kr"])
    # an uncertain title or number is flagged in the title and explained in the curation note
    if "to confirm" in c.title:
        assert "confirm" in c.curation_note


def test_the_notes_carry_the_offline_hints_and_the_numbers_stay_unverified():
    cl = load_candidates()
    notice, stations, allocation, emf = cl.get(NOTICE), cl.get(STATIONS), cl.get(ALLOCATION), cl.get(EMF)
    # every KR 447 number in the notice's summary is marked ungrounded, and the kr447.* profile choices are called placeholders
    assert "447.5625-447.8625 MHz" in notice.summary and "None of these numbers is grounded" in notice.summary
    assert "kr447.* profile choices are placeholders" in notice.engineering_implication
    # the decision is only about the class answer: the class's technical conditions are named as not evaluated
    assert notice.not_evaluated and "technical conditions" in notice.not_evaluated and "never passes" in notice.not_evaluated
    assert emf.not_evaluated and "radio answer" in emf.not_evaluated
    # the licence-exempt station provision: both article hints are recorded as hints to confirm, never as citations
    notes = " ".join(r.get("note") or "" for r in stations.related)
    assert "제19조의2" in notes and "제58조의3" in notes and all(r.get("url") is None for r in stations.related)
    assert "not citations" in stations.curation_note
    # the receiver's response frequencies include the LO-spur response at image + f_R (decision 1B)
    assert all(f in allocation.summary for f in ("404.76 MHz", "436.86 MHz", "440.28 MHz", "490.36 MHz"))
    # the KC route is kept: the licence exemption is for the station, never for the equipment's conformity assessment
    assert "does not replace KC conformity assessment" in stations.engineering_implication
    assert any("reg.KR.RadioWavesAct.58-2" in (r.get("note") or "") for r in notice.related)


# --------------------------------------------------------------------------- applicability


def test_the_demo_scope_answers_make_every_kr_entry_applicable():
    cl = load_candidates()
    decided = {c.id: evaluate(c.applicability_rule, DEMO_ANSWERS, _reqs()) for c in cl.for_jurisdiction("KR")}
    assert {cid: e.applicability for cid, e in decided.items()} == {cid: Applicability.APPLICABLE for cid in OLD_KR_IDS + NEW_IDS}
    # applicable means "compliance not verified", never PASS
    assert all(e.status is ValidationStatus.NOT_VERIFIED for e in decided.values())
    assert decided[NOTICE].inputs_used == {"radio": "yes (answer)", CLASS_KEY: "yes (answer)"} and decided[NOTICE].evidence == []


@pytest.mark.parametrize(
    "answers, class_entries, radio_entries, missing",
    [
        # radio = no decides every new entry; the all_of pair needs no class answer for that
        ({"radio": "no"}, Applicability.NOT_APPLICABLE, Applicability.NOT_APPLICABLE, []),
        ({"radio": "no", CLASS_KEY: "yes"}, Applicability.NOT_APPLICABLE, Applicability.NOT_APPLICABLE, []),
        # a radio without the class answer: the licence-exempt pair waits for it, the radio-only entries apply
        ({"radio": "yes"}, Applicability.UNDECIDED, Applicability.APPLICABLE, [CLASS_KEY]),
        ({"radio": "yes", CLASS_KEY: "no"}, Applicability.NOT_APPLICABLE, Applicability.APPLICABLE, []),
        ({"radio": "네", CLASS_KEY: "예"}, Applicability.APPLICABLE, Applicability.APPLICABLE, []),
        # an answer that is not yes/no is not guessed
        ({"radio": "yes", CLASS_KEY: "maybe a PMR446 set"}, Applicability.UNDECIDED, Applicability.APPLICABLE, [CLASS_KEY]),
        # nothing answered: every new entry waits for radio, the licence-exempt pair for the class answer too
        ({}, Applicability.UNDECIDED, Applicability.UNDECIDED, ["radio", CLASS_KEY]),
    ],
)
def test_the_answers_decide_the_new_entries(answers: dict[str, str], class_entries: Applicability, radio_entries: Applicability, missing: list[str]):
    cl = load_candidates()
    for cid in NEW_IDS:
        ev = evaluate(cl.get(cid).applicability_rule, answers, _reqs())
        assert ev.applicability is (class_entries if cid in CLASS_IDS else radio_entries), cid
        if ev.applicability is Applicability.UNDECIDED:
            assert ev.status is ValidationStatus.USER_INPUT_REQUIRED and (ev.missing_keys == missing if cid in CLASS_IDS else ev.missing_keys == ["radio"]), cid


# --------------------------------------------------------------------------- research


def test_offline_research_reports_them_unfetchable_and_never_passes(tmp_path: Path):
    for archive in (None, offline_archive(tmp_path / "sources")):
        out = research(_kr(), load_candidates(), archive, DEMO_ANSWERS, _reqs())
        by = {r.candidate_id: r for r in out.candidates}
        assert list(by) == OLD_KR_IDS + NEW_IDS and out.undecided == {}
        for cid in NEW_IDS:
            r = by[cid]
            assert r.source_status == "unfetchable" and r.verification is ValidationStatus.NOT_VERIFIED and r.documents == [] and r.quotes == []
            assert r.reason.startswith("official text not fetchable by machine: ") and r.evidence_grounded is False
            req = r.requirement
            assert req.applicability is Applicability.APPLICABLE and req.status is ValidationStatus.NOT_VERIFIED and req.basis == "curated"
            p = req.provenance
            assert p.source_url is None and p.content_hash is None and p.source_document is None and p.retrieved_at is None and p.section is None
            assert p.verification_status is ValidationStatus.NOT_VERIFIED and req.grounded_quotes == []
        sources = out.result(SOURCES_CHECK)
        assert sources.status is ValidationStatus.NOT_VERIFIED and all(f"{cid}: official text not fetchable by machine" in sources.message for cid in NEW_IDS)
        app = out.result(APPLICABILITY_CHECK)
        assert app.status is ValidationStatus.NOT_VERIFIED
        uncited = app.message.split("the decision cites no official sentence for ")[1].split(" (the curated rule")[0].split(", ")
        assert uncited == ["reg.KR.RadioWavesAct.ConformityAssessmentNotice", *NEW_IDS]
        assert "not evaluated by the rules: " in app.message and f"{NOTICE}: whether the design meets the class's technical conditions" in app.message
        assert out.result(COMPLIANCE_CHECK).status is ValidationStatus.NOT_VERIFIED and out.result(RESEARCH_CHECK).status is not ValidationStatus.PASS
        assert out.result(COMPLIANCE_CHECK).details["applicable"] == OLD_KR_IDS + NEW_IDS
    assert not (tmp_path / "sources").exists() or not any((tmp_path / "sources").rglob("*.xml"))


def test_online_research_fetches_nothing_for_them(fake: FakeSources, tmp_path: Path):
    """With an approved online session, only the three fetchable KR entries' law.go.kr DRF documents are requested; the new entries -
    even the RRA one whose rra.go.kr domain the session now trusts - send no request, because a URL is never guessed."""
    archive = online_archive(tmp_path / "sources", fake)
    try:
        out = research(_kr(), load_candidates(), archive, DEMO_ANSWERS, _reqs())
        assert archive.policy.trust_reasons["rra.go.kr"].startswith(f"official domain listed for {TEST_METHODS}")
    finally:
        archive.close()
    assert fake.requests_for("rra.go.kr") == [] and fake.requests_for("www.rra.go.kr") == []
    assert {r.host for r in fake.requests} == {"www.law.go.kr"}
    assert sorted({r.query.split("MST=")[1].split("&")[0] for r in fake.requests}) == ["273575", "276245", "276591"]
    by = {r.candidate_id: r for r in out.candidates}
    assert all(by[cid].source_status == "unfetchable" and by[cid].documents == [] for cid in NEW_IDS)
    # the fetchable entries got the fake's 404: not verified, never FAIL, never PASS
    assert all(by[cid].verification is ValidationStatus.NOT_VERIFIED for cid in OLD_KR_IDS)
    assert out.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_even_with_every_probe_quote_found_a_kr_radio_run_is_not_verified(every_quote_found):
    out = research(_kr(), load_candidates(), None, DEMO_ANSWERS, _reqs())
    by = {r.candidate_id: r for r in out.candidates}
    # the probe entries that cite a found sentence are grounded; the notice and the six additions cite nothing
    assert [cid for cid, r in by.items() if r.evidence_grounded] == OLD_KR_IDS[:3]
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is ValidationStatus.NOT_VERIFIED and "were not grounded for" not in app.message
    assert all(cid in app.message.split("cites no official sentence for ")[1] for cid in NEW_IDS)
    assert app.details["undecided"] == {} and all(not c["evidence_grounded"] for c in app.details["candidates"] if c["id"] in NEW_IDS)
    assert out.result(COMPLIANCE_CHECK).status is ValidationStatus.NOT_VERIFIED and out.result(RESEARCH_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_a_radio_without_the_class_answer_names_the_key_to_answer():
    out = research(_kr(), load_candidates(), None, {k: v for k, v in DEMO_ANSWERS.items() if k != CLASS_KEY}, _reqs())
    assert out.undecided == {NOTICE: [CLASS_KEY], STATIONS: [CLASS_KEY]}
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is ValidationStatus.NOT_VERIFIED and app.message.startswith(f"2 of 10 candidate(s) undecided - answer {CLASS_KEY} (--answer key=value)")
    assert out.result(COMPLIANCE_CHECK).details["undecided"] == [NOTICE, STATIONS]


# --------------------------------------------------------------------------- the agents


def _kr_ir(tmp_path: Path) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="kr447", name="kr447", workdir=str(tmp_path)))
    ir.regulatory.jurisdictions = [Jurisdiction(code="KR", name="KR")]
    ir.requirements.requirements.extend(_reqs().requirements)
    return ir


def test_a_kr_run_asks_the_class_question_without_blocking_and_remembers_the_answer(tmp_path: Path):
    ir = _kr_ir(tmp_path)
    first = RegulatoryAgent().run(ir, AgentContext(workdir=tmp_path, answers={k: v for k, v in DEMO_ANSWERS.items() if k != CLASS_KEY}))
    Orchestrator.apply_proposals(ir, first.proposals)
    assert [q.key for q in first.questions] == [CLASS_KEY] and not first.questions[0].required and not first.blocked_on_user
    assert first.questions[0].options == ["yes", "no"]
    by = {r.id: r for r in ir.regulatory.requirements}
    assert by[NOTICE].applicability is Applicability.UNDECIDED and by[NOTICE].missing_inputs == [CLASS_KEY]
    assert by[RULES].applicability is Applicability.APPLICABLE
    # a not-understood answer is asked again, with the answer quoted
    again = RegulatoryAgent().run(ir, AgentContext(workdir=tmp_path, answers={CLASS_KEY: "PMR"}))
    assert [q.key for q in again.questions] == [CLASS_KEY] and "was not understood as yes / no" in again.questions[0].question
    # answered: remembered as a scope answer; the next run without answers asks nothing and keeps the decision
    second = RegulatoryAgent().run(ir, AgentContext(workdir=tmp_path, answers={CLASS_KEY: "yes"}))
    Orchestrator.apply_proposals(ir, second.proposals)
    assert second.questions == [] and ir.regulatory.scope_answers[CLASS_KEY] == "yes"
    third = RegulatoryAgent().run(ir, AgentContext(workdir=tmp_path))
    Orchestrator.apply_proposals(ir, third.proposals)
    assert third.questions == []
    by = {r.id: r for r in ir.regulatory.requirements}
    assert all(by[cid].applicability is Applicability.APPLICABLE and by[cid].status is ValidationStatus.NOT_VERIFIED for cid in NEW_IDS)
    assert all(by[cid].source_status == "unfetchable" for cid in NEW_IDS)


def test_a_typed_class_answer_is_a_regulatory_scope_statement_not_a_design_requirement(tmp_path: Path):
    ctx = AgentContext(workdir=tmp_path, answers={"application": "447 MHz walkie-talkie", "jurisdiction": "KR", CLASS_KEY: "yes", "radio": "yes"})
    assert CLASS_KEY in regulatory_scope_keys(ctx)
    ir = CircuitIR(project=ProjectMeta(id="kr447", name="kr447", workdir=str(tmp_path)))
    result = RequirementAgent().run(ir, ctx)
    Orchestrator.apply_proposals(ir, result.proposals)
    req = ir.requirements.get(CLASS_KEY)
    assert req is not None and req.value.value == "yes" and req.category == "regulatory" and req.category not in DESIGN_CATEGORIES
    assert ir.requirements.get("radio").category == "regulatory"
