"""The curated candidate list: the packaged file is valid and honest, and the loader rejects what it must."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_eda.regulatory.candidates import (
    DEFAULT_CANDIDATES_PATH,
    UNVERIFIED_NOTE,
    ApplicabilityRule,
    CandidateList,
    check_url_host,
    load_candidates,
)
from ai_eda.tools.sources.policy import host_key, host_of

MINIMAL: dict = {
    "schema_version": 1,
    "provenance": f"curated candidate list; every entry is {UNVERIFIED_NOTE}",
    "curated_at": "2026-09-23",
    "scope_questions": [
        {"key": "mains_powered", "question": "AC mains? (yes/no)", "options": ["yes", "no"]},
        {"key": "radio", "question": "radio? (yes/no)", "options": ["yes", "no"]},
    ],
    "candidates": [
        {
            "id": "reg.EU.LVD.test",
            "jurisdiction": "EU",
            "title": "LVD (test)",
            "authority": "test authority",
            "official_url": "https://official.eu.example/lvd",
            "expected_markers": ["LVD TEST TEXT"],
            "allowed_domains": ["official.eu.example"],
            "applicability_rule": {"kind": "voltage_range", "requirement": "input_voltage", "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"},
            "grounding_quotes": [{"section": "Article 1", "quote": "between 50 and 1 000 V for alternating current"}],
        }
    ],
}


def _with(**changes) -> dict:
    d = copy.deepcopy(MINIMAL)
    d.update(changes)
    return d


def _candidate(**changes) -> dict:
    d = copy.deepcopy(MINIMAL)
    d["candidates"][0].update(changes)
    return d


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "candidates.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- the packaged file


def test_packaged_file_loads_and_states_its_provenance():
    cl = load_candidates()
    assert cl.source_path == str(DEFAULT_CANDIDATES_PATH) and cl.sha256 and cl.sha256.startswith("sha256:")
    assert UNVERIFIED_NOTE in cl.provenance and cl.curated_at == "2026-09-23"
    assert cl.jurisdictions() == ["EU", "KR", "US"]
    assert {c.id for c in cl.for_jurisdiction("EU")} == {"reg.EU.LVD.2014-35-EU", "reg.EU.EMC.2014-30-EU", "reg.EU.RoHS.2011-65-EU", "reg.EU.RED.2014-53-EU"}
    assert {c.id for c in cl.for_jurisdiction("KR")} == {"reg.KR.ElectricalAppliancesSafetyAct", "reg.KR.ElectricalAppliancesSafetyAct.EnforcementRule", "reg.KR.RadioWavesAct.58-2",
                                                         "reg.KR.RadioWavesAct.ConformityAssessmentNotice"}
    assert [c.id for c in cl.for_jurisdiction("US")] == ["reg.US.FCC.47CFR15"]
    assert cl.for_jurisdiction("JP") == []
    d = cl.describe()
    assert d["candidates"] == 9 and d["sha256"] == cl.sha256
    # the one entry whose text the probe did not fetch is kept, says why, and guesses neither a URL nor a quote
    unfetchable = [c for c in cl.candidates if not c.fetchable]
    assert [c.id for c in unfetchable] == ["reg.KR.RadioWavesAct.ConformityAssessmentNotice"]
    assert unfetchable[0].official_url is None and unfetchable[0].grounding_quotes == [] and "target=admrul" in unfetchable[0].unfetchable_reason


def test_packaged_file_every_url_host_is_allow_listed_and_https():
    cl = load_candidates()
    for c in cl.candidates:
        allowed = {host_key(d) for d in c.allowed_domains}
        for doc in c.documents():
            assert doc.url.startswith("https://"), doc.url
            assert check_url_host(doc.url, allowed, c.id) in allowed
            if doc.date_discovery is not None:
                assert check_url_host(doc.date_discovery.url, allowed, c.id) in allowed
        for q in c.grounding_quotes:
            assert (q.url or c.official_url) in {d.url for d in c.documents()}
    hosts = cl.allowed_hosts()
    assert set(hosts) == {"eur-lex.europa.eu", "law.go.kr", "ecfr.gov"}
    assert "reg.EU.LVD.2014-35-EU" in hosts["eur-lex.europa.eu"]


def test_packaged_file_rules_only_read_asked_questions_and_cite_existing_quotes():
    cl = load_candidates()
    keys = {q.key for q in cl.scope_questions}
    assert {"intended_use", "mains_powered", "radio"} <= keys
    for c in cl.candidates:
        assert set(c.applicability_rule.answer_keys()) <= keys, c.id
        sections = {q.section for q in c.grounding_quotes}
        assert set(c.applicability_rule.evidence_labels()) <= sections, c.id
        assert c.summary and c.engineering_implication, c.id
        if c.fetchable:
            assert c.official_url and c.grounding_quotes and c.expected_markers, c.id
        else:
            assert c.unfetchable_reason and not c.grounding_quotes and not c.applicability_rule.evidence_labels(), c.id
    lvd = cl.get("reg.EU.LVD.2014-35-EU")
    assert lvd.applicability_rule.kind == "all_of" and lvd.applicability_rule.requirement_keys() == ["input_voltage", "output_voltage"]
    assert lvd.applicability_rule.answer_keys() == ["radio", "evaluation_kit", "mains_powered", "highest_rated_voltage"]
    assert "1 000 V" in lvd.quote("Article 1").quote  # plain space; the official HTML uses U+00A0 and find_quote tolerates it
    assert lvd.not_evaluated and "Annex II" in lvd.not_evaluated and cl.get("reg.EU.RoHS.2011-65-EU").not_evaluated.startswith("the 'subject to paragraph 2'")
    assert [q.key for q in cl.questions_for(["EU"])] == ["intended_use", "mains_powered", "highest_rated_voltage", "evaluation_kit", "radio", "finished_apparatus"]
    assert [q.key for q in cl.questions_for(["US"])] == ["intended_use", "mains_powered", "radio", "digital_device"]
    assert [q.key for q in cl.questions_for(["EU", "KR"])] == ["intended_use", "mains_powered", "highest_rated_voltage", "evaluation_kit", "radio", "finished_apparatus", "digital_device"]


def test_packaged_placeholders_are_declared_and_come_from_env_or_default(monkeypatch):
    cl = load_candidates()
    kr = cl.get("reg.KR.RadioWavesAct.58-2")
    assert "{LAW_GO_KR_OC}" in kr.official_url and "LAW_GO_KR_OC" in cl.placeholders
    assert cl.placeholder_value("LAW_GO_KR_OC", env={}) == ("test", "default from candidates.json (" + cl.placeholders["LAW_GO_KR_OC"].note + ")")
    assert cl.placeholder_value("LAW_GO_KR_OC", env={"AI_EDA_LAW_GO_KR_OC": "myoc"})[0] == "myoc"
    monkeypatch.setenv("AI_EDA_LAW_GO_KR_OC", "fromenv")
    assert cl.placeholder_value("LAW_GO_KR_OC") == ("fromenv", "environment variable AI_EDA_LAW_GO_KR_OC")
    us = cl.get("reg.US.FCC.47CFR15")
    assert "{DATE}" in us.official_url and us.date_discovery is not None
    assert us.date_discovery.match == {"number": 47} and us.date_discovery.field == "up_to_date_as_of"
    assert host_key(host_of(us.date_discovery.url)) == "ecfr.gov"


def test_packaged_not_included_entries_carry_reasons():
    cl = load_candidates()
    assert cl.not_included and all(e.get("reason") for e in cl.not_included)
    assert any("unblock.federalregister.gov" in (e.get("reason") or "") for e in cl.not_included)


# --------------------------------------------------------------------------- validation


def test_minimal_list_validates(tmp_path: Path):
    cl = load_candidates(_write(tmp_path, MINIMAL))
    assert [c.id for c in cl.candidates] == ["reg.EU.LVD.test"] and cl.sha256
    doc = cl.candidates[0].documents()[0]
    assert doc.role == "official" and doc.expected_markers == ["LVD TEST TEXT"]


@pytest.mark.parametrize(
    "data, needle",
    [
        (_candidate(official_url="https://other.example/lvd"), "not in allowed_domains"),
        (_candidate(official_url="https://{HOST}.example/lvd"), "placeholder may not appear in the host"),
        (_candidate(official_url="ftp://official.eu.example/lvd"), "unusable URL"),
        (_candidate(extra_documents=[{"url": "https://mirror.example/lvd", "role": "mirror"}]), "not in allowed_domains"),
        (_candidate(grounding_quotes=[{"section": "Article 1", "quote": "x", "url": "https://official.eu.example/other"}]), "not one of the entry's documents"),
        (_candidate(grounding_quotes=[]), "at least one grounding quote"),
        (_candidate(official_url=None), "needs an official_url"),
        (_candidate(fetchable=False, unfetchable_reason="x", official_url="https://other.example/lvd"), "not in allowed_domains"),
        (_candidate(applicability_rule={"kind": "always", "evidence": "Article 99"}), "not a grounding quote section"),
        (_candidate(applicability_rule={"kind": "answer", "key": "unknown_key", "equals": "yes"}), "no scope question asks it"),
        (_candidate(applicability_rule={"kind": "voltage_range", "requirement": "input_voltage"}), "ac"),
        (_candidate(applicability_rule={"kind": "voltage_range", "requirement": "input_voltage", "dc": [1500, 75]}), "low <= high"),
        (_candidate(applicability_rule={"kind": "answer", "key": "radio"}), "needs 'equals'"),
        (_candidate(applicability_rule={"kind": "all_of", "rules": []}), "at least one rule"),
        (_candidate(applicability_rule={"kind": "always", "key": "radio"}), "not a field of a always rule"),
        (_candidate(fetchable=False), "must say why"),
        (_candidate(id="reg.KR.LVD.test"), "carries jurisdiction"),
        (_candidate(id="lvd"), "must look like reg.<JUR>.<name>"),
        (_candidate(official_url="https://official.eu.example/{DATE}/lvd"), "no date_discovery"),
        (_candidate(official_url="https://official.eu.example/lvd?oc={OC}"), "not declared"),
        (_with(provenance="a list of regulations"), UNVERIFIED_NOTE),
        (_with(curated_at="yesterday"), "ISO date"),
        (_with(schema_version=2), "schema 2"),
        (_with(candidates=MINIMAL["candidates"] * 2), "duplicate candidate ids"),
        (_with(scope_questions=MINIMAL["scope_questions"] * 2), "duplicate scope question keys"),
        (_with(unknown_field=1), "unknown_field"),
    ],
)
def test_invalid_lists_are_rejected_with_the_reason(tmp_path: Path, data: dict, needle: str):
    with pytest.raises(ValueError, match=r"(?s).*" + __import__("re").escape(needle)):
        load_candidates(_write(tmp_path, data))


def test_unreadable_or_non_json_file(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot read"):
        load_candidates(tmp_path / "missing.json")
    p = tmp_path / "bad.json"
    p.write_bytes(b"\xff\xfe not json")
    with pytest.raises(ValueError, match="not a UTF-8 JSON"):
        load_candidates(p)
    p.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be an object"):
        load_candidates(p)


def test_rule_helpers_and_date_placeholder(tmp_path: Path):
    data = _candidate(
        official_url="https://official.eu.example/{DATE}/lvd",
        date_discovery={"url": "https://official.eu.example/index.json", "list_key": "titles", "match": {"number": 47}, "field": "up_to_date_as_of"},
        applicability_rule={"kind": "all_of", "rules": [
            {"kind": "not", "rule": {"kind": "answer", "key": "radio", "equals": "yes"}},
            {"kind": "voltage_range", "requirement": "input_voltage", "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"},
        ]},
    )
    cl = load_candidates(_write(tmp_path, data))
    rule = cl.candidates[0].applicability_rule
    assert rule.answer_keys() == ["radio", "mains_powered"] and rule.requirement_keys() == ["input_voltage"] and rule.evidence_labels() == ["Article 1"]
    doc = cl.candidates[0].documents()[0]
    assert doc.date_discovery is not None and doc.date_discovery.pattern.startswith("^")
    assert ApplicabilityRule(kind="never").answer_keys() == []
    with pytest.raises(ValueError):
        ApplicabilityRule(kind="not")


def test_unfetchable_entry_may_carry_neither_url_nor_quotes(tmp_path: Path):
    data = _candidate(fetchable=False, unfetchable_reason="not probed", official_url=None, grounding_quotes=[], expected_markers=[],
                      applicability_rule={"kind": "answer", "key": "radio", "equals": "yes"})
    cl = load_candidates(_write(tmp_path, data))
    c = cl.candidates[0]
    assert c.official_url is None and c.documents() == [] and c.quotes_for("https://official.eu.example/lvd") == []
    # a URL, when given, is still checked against the allow-list even on an unfetchable entry
    data["candidates"][0]["official_url"] = "https://official.eu.example/lvd"
    assert load_candidates(_write(tmp_path, data)).candidates[0].documents()[0].role == "official"


def test_check_url_host():
    assert check_url_host("https://www.law.go.kr/DRF/lawService.do?OC={LAW_GO_KR_OC}&type=XML", {"law.go.kr"}, "x") == "law.go.kr"
    with pytest.raises(ValueError, match="not in allowed_domains"):
        check_url_host("https://assets.nexperia.com/a.pdf", {"nexperia.com"}, "x")
    with pytest.raises(ValueError, match="placeholder may not appear in the host"):
        check_url_host("https://{H}/a", {"h"}, "x")


def test_candidate_list_is_a_pure_model_too():
    cl = CandidateList.model_validate(MINIMAL)
    assert cl.sha256 is None and cl.source_path is None
    assert cl.questions_for(["EU"]) == cl.scope_questions
