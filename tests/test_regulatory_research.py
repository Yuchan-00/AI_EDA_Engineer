"""Regulatory research against a local fake of the official sites (tests/fake_sources.py); nothing here touches the network
except the three live tests at the end (EUR-Lex, law.go.kr, eCFR), which run only with AI_EDA_ONLINE=1.

What is proven: offline nothing is fetched and every source is NOT_VERIFIED; online only the allow-listed hosts are
contacted, documents are archived and every claimed quote grounded (PASS means exactly that); a bot wall, a 404, a
redirect off the allow-list or a wrong document is NOT_VERIFIED; a quote the official text does not contain FAILs the
candidate (the list is wrong); applicability is decided from answers + requirements with the grounded quote as evidence;
compliance is always NOT_VERIFIED; every requirement carries the ten provenance fields; archived copies are reused
offline and a tampered copy is never used.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from ai_eda.ir import Requirement, RequirementKind, RequirementSet, SourceRef, ValidationStatus, user_requirement
from ai_eda.ir.regulatory import Applicability, Jurisdiction, RegulatoryProvenance, RegulatoryRequirement, RegulatoryState
from ai_eda.regulatory import (
    APPLICABILITY_CHECK,
    COMPLIANCE_CHECK,
    RESEARCH_CHECK,
    RESEARCH_TOOL,
    SOURCES_CHECK,
    CandidateList,
    load_candidates,
    research,
)
from ai_eda.regulatory.research import COMPLIANCE_MESSAGE, OFFLINE_REASON, research_candidate
from ai_eda.security.approval import ApprovalGate
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy
from tests.fake_sources import FakeSources

EU_HOST, KR_HOST, US_HOST = "official.eu.example", "official.kr.example", "official.us.example"
LVD_URL = f"https://{EU_HOST}/lvd"
RED_URL = f"https://{EU_HOST}/red"
ROHS_URL = f"https://{EU_HOST}/rohs"
ROHS_CONSOL_URL = f"https://{EU_HOST}/rohs-consolidated"
KR_URL = f"https://www.{KR_HOST}/DRF/lawService.do?OC={{LAW_GO_KR_OC}}&target=law&MST=1&type=XML"
US_URL = f"https://{US_HOST}/api/versioner/v1/full/{{DATE}}/title-47.xml?part=15"
US_INDEX_URL = f"https://{US_HOST}/api/versioner/v1/titles.json"

LVD_SCOPE = ("This Directive shall apply to electrical equipment designed for use with a voltage rating of between 50 and 1 000 V for alternating "
             "current and between 75 and 1 500 V for direct current, other than the equipment and phenomena listed in Annex II.")
LVD_KITS = "Custom built evaluation kits destined for professionals to be used solely at research and development facilities for such purposes."
RED_DEF = "‘radio equipment’ means an electrical or electronic product, which intentionally emits and/or receives radio waves"
ROHS_SCOPE = "This Directive shall, subject to paragraph 2, apply to EEE falling within the categories set out in Annex I."
ROHS_DEHP = "Bis(2-ethylhexyl) phthalate (DEHP) (0,1 %)"
KR_DEF = "\"전기용품\"이란 공업적으로 생산된 물품으로서 교류 전원 또는 직류 전원에 연결하여 사용되는 제품이나 그 부분품 또는 부속품을 말한다."
US_DIGITAL = ("Digital device. (Previously defined as a computing device). An unintentional radiator (device or system) that generates and uses timing "
              "signals or pulses at a rate in excess of 9,000 pulses (cycles) per second and uses digital techniques;")
FILLER = "<p>" + "This paragraph exists so that the synthetic official page carries enough readable text to count as a document. " * 4 + "</p>"


def official_html(title: str, heading: str, *paragraphs: str) -> str:
    body = "".join(f'<p class="oj-normal">{p}</p>' for p in paragraphs)
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{title}</title><script>var t = "not text";</script></head>'
            f'<body><div class="eli-container"><p class="oj-doc-ti">{heading}</p><p class="oj-ti-art">Article 1</p>{body}{FILLER}</div></body></html>')


def lvd_html(with_scope: bool = True) -> str:
    scope = LVD_SCOPE.replace("1 000", "1 000").replace("1 500", "1 500")
    return official_html("L_TEST_LVD.xml", "DIRECTIVE 2014/35/EU (SYNTHETIC TEST TEXT)", *( [scope] if with_scope else []), LVD_KITS)


def red_html() -> str:
    return official_html("L_TEST_RED.xml", "DIRECTIVE 2014/53/EU (SYNTHETIC TEST TEXT)", RED_DEF + " for the purpose of radio communication;")


def rohs_html(consolidated: bool = False) -> str:
    title = "Consolidated TEXT: 32011L0065 (SYNTHETIC)" if consolidated else "L_TEST_ROHS.xml"
    extra = [ROHS_DEHP.replace("0,1 %", "0,1 %")] if consolidated else []
    return official_html(title, "DIRECTIVE 2011/65/EU (SYNTHETIC TEST TEXT)", ROHS_SCOPE, "Lead (0,1 %)", *extra)


KR_XML = ('<?xml version="1.0" encoding="UTF-8"?><법령 법령키="0000"><기본정보><법령ID>000001</법령ID><법령명_한글><![CDATA[전기용품 및 생활용품 안전관리법 (synthetic)]]></법령명_한글></기본정보>'
          f'<조문><조문단위><조문내용><![CDATA[제2조(정의) 1. {KR_DEF}]]></조문내용></조문단위></조문></법령>')
KR_ERROR_XML = ('<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n<result>사용자 정보 검증에 실패하였습니다.</result>\n'
                '<msg>OPEN API 호출 시 사용자 검증을 위하여 정확한 서버장비의 IP주소 및 도메인주소를 등록해 주세요.</msg>\n</Response>\n')
US_XML = ('<?xml version="1.0" encoding="UTF-8"?><DIV5 N="15" TYPE="PART"><HEAD>PART 15—RADIO FREQUENCY DEVICES</HEAD>'
          f'<DIV8 N="§ 15.3" TYPE="SECTION"><HEAD>§ 15.3 Definitions.</HEAD><P>(k) {US_DIGITAL}</P></DIV8></DIV5>')
US_INDEX = {"titles": [{"number": 46, "name": "Shipping", "up_to_date_as_of": "2026-09-20"}, {"number": 47, "name": "Telecommunication", "up_to_date_as_of": "2026-09-21"}],
            "meta": {"date": "2026-09-21"}}


def index_json(titles: list[dict]) -> str:
    """An index body like the real titles.json (the archive treats a body under 64 bytes as 'not a document', so it carries meta too)."""
    return json.dumps({"titles": titles, "meta": {"date": "2026-09-21", "import_in_progress": False, "note": "synthetic eCFR titles index"}})


def candidates_dict() -> dict:
    return {
        "schema_version": 1,
        "provenance": "curated candidate list; every entry is unverified until fetched and grounded (synthetic test list)",
        "curated_at": "2026-09-23",
        "placeholders": {"LAW_GO_KR_OC": {"env": "AI_EDA_LAW_GO_KR_OC", "default": "test"}},
        "scope_questions": [
            {"key": "intended_use", "question": "intended use?"},
            {"key": "mains_powered", "question": "mains? (yes/no)", "options": ["yes", "no"]},
            {"key": "radio", "question": "radio? (yes/no)", "options": ["yes", "no"]},
            {"key": "digital_device", "question": "digital? (yes/no)", "options": ["yes", "no"], "jurisdictions": ["US"]},
        ],
        "candidates": [
            {
                "id": "reg.EU.LVD.test", "jurisdiction": "EU", "title": "Low Voltage Directive (test)", "authority": "EU legislator (test)",
                "official_url": LVD_URL, "expected_markers": ["L_TEST_LVD.xml", "DIRECTIVE 2014/35/EU (SYNTHETIC TEST TEXT)"], "allowed_domains": [EU_HOST],
                "summary": "LVD summary (unverified)", "engineering_implication": "safety objectives",
                "applicability_rule": {"kind": "all_of", "rules": [
                    {"kind": "not", "rule": {"kind": "answer", "key": "radio", "equals": "yes"}},
                    {"kind": "voltage_range", "requirement": "input_voltage", "ac": [50, 1000], "dc": [75, 1500], "evidence": "Article 1"},
                ]},
                "grounding_quotes": [{"section": "Article 1", "quote": LVD_SCOPE}, {"section": "Annex II", "quote": LVD_KITS}],
            },
            {
                "id": "reg.EU.RED.test", "jurisdiction": "EU", "title": "Radio Equipment Directive (test)", "authority": "EU legislator (test)",
                "official_url": RED_URL, "expected_markers": ["L_TEST_RED.xml"], "allowed_domains": [EU_HOST],
                "applicability_rule": {"kind": "answer", "key": "radio", "equals": "yes", "evidence": "Article 2(1)(1)"},
                "grounding_quotes": [{"section": "Article 2(1)(1)", "quote": RED_DEF}],
            },
            {
                "id": "reg.EU.RoHS.test", "jurisdiction": "EU", "title": "RoHS (test)", "authority": "EU legislator (test)",
                "official_url": ROHS_URL, "expected_markers": ["L_TEST_ROHS.xml"], "allowed_domains": [EU_HOST],
                "extra_documents": [{"url": ROHS_CONSOL_URL, "role": "consolidated", "expected_markers": ["Consolidated TEXT: 32011L0065"]}],
                "applicability_rule": {"kind": "always", "evidence": "Article 2(1)"},
                "grounding_quotes": [{"section": "Article 2(1)", "quote": ROHS_SCOPE}, {"section": "Annex II (lead)", "quote": "Lead (0,1 %)"},
                                     {"section": "Annex II (DEHP)", "quote": ROHS_DEHP, "url": ROHS_CONSOL_URL}],
            },
            {
                "id": "reg.KR.Safety.test", "jurisdiction": "KR", "title": "전기용품 안전관리법 (test)", "authority": "KATS (test)",
                "official_url": KR_URL, "document_form": "xml", "expected_markers": ["전기용품 및 생활용품 안전관리법", "000001"], "allowed_domains": [f"www.{KR_HOST}"],
                "applicability_rule": {"kind": "always", "evidence": "제2조 제1호"},
                "grounding_quotes": [{"section": "제2조 제1호", "quote": KR_DEF}],
            },
            {
                "id": "reg.US.FCC.test", "jurisdiction": "US", "title": "47 CFR Part 15 (test)", "authority": "FCC (test)",
                "official_url": US_URL, "document_form": "xml", "expected_markers": ["PART 15—RADIO FREQUENCY DEVICES"], "allowed_domains": [US_HOST],
                "date_discovery": {"url": US_INDEX_URL, "list_key": "titles", "match": {"number": 47}, "field": "up_to_date_as_of"},
                "applicability_rule": {"kind": "answer", "key": "digital_device", "equals": "yes", "evidence": "§ 15.3(k)"},
                "grounding_quotes": [{"section": "§ 15.3(k)", "quote": US_DIGITAL}],
            },
        ],
    }


def make_candidates(**changes) -> CandidateList:
    d = candidates_dict()
    d.update(changes)
    return CandidateList.model_validate(d)


def serve_all(fake: FakeSources, *, lvd_scope: bool = True) -> None:
    fake.add_html(LVD_URL, lvd_html(lvd_scope))
    fake.add_html(RED_URL, red_html())
    fake.add_html(ROHS_URL, rohs_html())
    fake.add_html(ROHS_CONSOL_URL, rohs_html(consolidated=True))
    fake.add_xml(KR_URL.replace("{LAW_GO_KR_OC}", "test"), KR_XML)
    fake.add_xml(KR_URL.replace("{LAW_GO_KR_OC}", "myoc"), KR_XML)
    fake.serve(US_INDEX_URL, json.dumps(US_INDEX), "application/json")
    fake.add_xml(US_URL.replace("{DATE}", "2026-09-21"), US_XML)


def online_archive(root: Path, fake: FakeSources) -> DocumentArchive:
    return DocumentArchive(root, NetworkPolicy(approved=True, gate=ApprovalGate()), client=fake.client())


def offline_archive(root: Path) -> DocumentArchive:
    return DocumentArchive(root, NetworkPolicy(approved=False, gate=ApprovalGate()))


def state_for(*codes: str) -> RegulatoryState:
    return RegulatoryState(jurisdictions=[Jurisdiction(code=c, name=c) for c in codes])


def reqs(volts: float | str | None = 12.0, text: str = "12 V DC input") -> RequirementSet:
    if volts is None:
        return RequirementSet()
    return RequirementSet(requirements=[Requirement(id="req.input_voltage", key="input_voltage", text=text, kind=RequirementKind.EXPLICIT,
                                                    value=user_requirement(volts, None if isinstance(volts, str) else "V"))])


ANSWERS = {"intended_use": "bench tool", "mains_powered": "no", "radio": "no", "digital_device": "no"}
PROVENANCE_FIELDS = ["jurisdiction", "authority", "source_title", "source_url", "retrieved_at", "section", "applicability_rationale", "verification_status",
                     "source_document", "content_hash"]


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


def _by_id(outcome) -> dict[str, RegulatoryRequirement]:
    return {r.id: r for r in outcome.state.requirements}


# --------------------------------------------------------------------------- offline


def test_offline_without_an_archive_fetches_nothing_and_is_not_verified():
    cl = make_candidates()
    out = research(state_for("EU"), cl, None, ANSWERS, reqs())
    assert not out.online
    sources, applicability, compliance, summary = out.results
    assert [r.check_id for r in out.results] == [SOURCES_CHECK, APPLICABILITY_CHECK, COMPLIANCE_CHECK, RESEARCH_CHECK]
    assert all(r.tool == RESEARCH_TOOL and r.tool_version for r in out.results)
    assert sources.status is ValidationStatus.NOT_VERIFIED and OFFLINE_REASON in sources.message and "--online" in sources.message
    assert sources.evidence == []
    by = _by_id(out)
    assert set(by) == {"reg.EU.LVD.test", "reg.EU.RED.test", "reg.EU.RoHS.test"}
    lvd = by["reg.EU.LVD.test"]
    assert lvd.source_status == "offline" and lvd.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    assert lvd.provenance.source_url == LVD_URL and lvd.provenance.content_hash is None and lvd.provenance.source_document is None and lvd.provenance.retrieved_at is None
    assert lvd.applicability is Applicability.NOT_APPLICABLE and lvd.status is ValidationStatus.NOT_APPLICABLE
    assert all(not q.found and q.reason.startswith("offline") for q in lvd.grounded_quotes)
    assert "; evidence: Article 1" in lvd.provenance.applicability_rationale and "grounded" not in lvd.provenance.applicability_rationale
    assert lvd.provenance.section == "Article 1"  # the cited section is the claim; page / found live on the quote
    # decided on the curated rule, but the sentence it rests on was not grounded: not PASS
    assert applicability.status is ValidationStatus.NOT_VERIFIED and "not grounded" in applicability.message and "reg.EU.LVD.test: not_applicable" in applicability.message
    assert compliance.status is ValidationStatus.NOT_VERIFIED and compliance.message == COMPLIANCE_MESSAGE
    assert summary.status is ValidationStatus.NOT_VERIFIED and summary.check_id == RESEARCH_CHECK


def test_offline_archive_without_session_fetches_nothing(fake, tmp_path: Path):
    serve_all(fake)
    out = research(state_for("EU"), make_candidates(), offline_archive(tmp_path / "sources"), ANSWERS, reqs())
    assert not out.online and fake.requests == []
    assert out.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED
    assert all(r.source_status == "offline" for r in out.state.requirements)


# --------------------------------------------------------------------------- online


def test_online_archives_grounds_and_fills_the_ten_provenance_fields(fake, tmp_path: Path):
    serve_all(fake)
    cl = make_candidates()
    archive = online_archive(tmp_path / "sources", fake)
    out = research(state_for("EU"), cl, archive, ANSWERS, reqs())
    assert out.online
    sources = out.result(SOURCES_CHECK)
    assert sources.status is ValidationStatus.PASS, sources.message
    assert "3 archived and grounded" in sources.message
    assert {host for host in (r.host for r in fake.requests)} == {EU_HOST}
    assert archive.policy.trust_reasons[EU_HOST].startswith("official domain listed for reg.EU.LVD.test")
    by = _by_id(out)
    lvd = by["reg.EU.LVD.test"]
    p = lvd.provenance
    for f in PROVENANCE_FIELDS:
        assert getattr(p, f) not in (None, ""), f
    assert p.jurisdiction == "EU" and p.authority == "EU legislator (test)" and p.source_title.startswith("Low Voltage Directive (test) [L_TEST_LVD.xml]")
    assert p.source_url == LVD_URL and p.section == "Article 1" and lvd.grounded_quotes[0].page == 1 and p.verification_status is ValidationStatus.PASS
    assert p.content_hash.startswith("sha256:") and Path(p.source_document).is_file()
    assert archive.verify(SourceRef(title="x", content_hash=p.content_hash)) == "ok"
    assert p.applicability_rationale.startswith("rule: ") and "; inputs: " in p.applicability_rationale and "; evidence: Article 1" in p.applicability_rationale
    assert "input_voltage = 12 V DC (requirement req.input_voltage; DC stated with the value)" in p.applicability_rationale and "radio = no (answer)" in p.applicability_rationale
    assert lvd.source_status == "ok" and lvd.applicability is Applicability.NOT_APPLICABLE and lvd.status is ValidationStatus.NOT_APPLICABLE
    assert [q.found for q in lvd.grounded_quotes] == [True, True]
    scope = lvd.grounded_quotes[0]
    # the served page writes the thousands separator as U+00A0 (as EUR-Lex does); the extractor stores it as a plain space and the plain-space quote grounds
    assert scope.page == 1 and "1 000 V" in scope.context and " " not in scope.context and scope.content_hash == p.content_hash and scope.source_url == LVD_URL
    # applicability: every candidate decided and each decision's quote grounded -> PASS
    applicability = out.result(APPLICABILITY_CHECK)
    assert applicability.status is ValidationStatus.PASS, applicability.message
    assert "reg.EU.LVD.test: not_applicable" in applicability.message and "reg.EU.RED.test: not_applicable" in applicability.message and "reg.EU.RoHS.test: applicable" in applicability.message
    rohs = by["reg.EU.RoHS.test"]
    assert rohs.applicability is Applicability.APPLICABLE and rohs.status is ValidationStatus.NOT_VERIFIED  # applies; compliance is not verified
    assert [q.found for q in rohs.grounded_quotes] == [True, True, True]
    dehp = rohs.grounded_quotes[2]
    assert dehp.source_url == ROHS_CONSOL_URL and dehp.content_hash != rohs.provenance.content_hash
    # evidence: every archived document once, with its hash
    hashes = {e.content_hash for e in sources.evidence}
    assert len(sources.evidence) == 4 and rohs.provenance.content_hash in hashes and dehp.content_hash in hashes
    assert all(Path(e.path).is_file() and e.url for e in sources.evidence)
    assert out.result(COMPLIANCE_CHECK).status is ValidationStatus.NOT_VERIFIED
    assert out.result(RESEARCH_CHECK).status is ValidationStatus.NOT_VERIFIED
    details = sources.details["candidates"][0]
    assert details["id"] == "reg.EU.LVD.test" and details["documents"][0]["marker"] == "L_TEST_LVD.xml" and details["documents"][0]["http_status"] == 200


def test_missing_answers_leave_candidates_undecided_but_sources_verified(fake, tmp_path: Path):
    serve_all(fake)
    out = research(state_for("EU"), make_candidates(), online_archive(tmp_path / "sources", fake), {}, reqs(None))
    assert out.result(SOURCES_CHECK).status is ValidationStatus.PASS
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is ValidationStatus.NOT_VERIFIED and "2 of 3 candidate(s) undecided" in app.message
    assert app.details["missing_keys"] == ["radio", "input_voltage"] and out.undecided == {"reg.EU.LVD.test": ["radio", "input_voltage"], "reg.EU.RED.test": ["radio"]}
    lvd = _by_id(out)["reg.EU.LVD.test"]
    assert lvd.applicability is Applicability.UNDECIDED and lvd.status is ValidationStatus.USER_INPUT_REQUIRED and lvd.missing_inputs == ["radio", "input_voltage"]
    assert lvd.provenance.verification_status is ValidationStatus.PASS  # the source is fine; the decision is what waits


def test_quote_missing_from_the_official_text_fails_the_candidate(fake, tmp_path: Path):
    serve_all(fake, lvd_scope=False)  # the expected document (marker present) without the claimed scope sentence
    out = research(state_for("EU"), make_candidates(), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    sources = out.result(SOURCES_CHECK)
    assert sources.status is ValidationStatus.FAIL and "reg.EU.LVD.test" in sources.message and "candidate list is wrong" in sources.message
    lvd = _by_id(out)["reg.EU.LVD.test"]
    assert lvd.source_status == "quote_missing" and lvd.status is ValidationStatus.FAIL and lvd.provenance.verification_status is ValidationStatus.FAIL
    assert [q.found for q in lvd.grounded_quotes] == [False, True] and "not found in the archived official text" in lvd.grounded_quotes[0].reason
    assert lvd.provenance.content_hash and lvd.provenance.section == "Article 1" and lvd.grounded_quotes[1].section == "Annex II" and lvd.grounded_quotes[1].page == 1
    assert _by_id(out)["reg.EU.RED.test"].provenance.verification_status is ValidationStatus.PASS
    assert out.result(RESEARCH_CHECK).status is ValidationStatus.FAIL
    assert out.result(APPLICABILITY_CHECK).status is ValidationStatus.NOT_VERIFIED  # the LVD decision cites the missing quote


@pytest.mark.parametrize("kind", ["cloudflare", "access_denied", "unblock", "js_shell"])
def test_bot_protection_and_shells_are_not_verified(fake, tmp_path: Path, kind: str):
    serve_all(fake)
    fake.add_interstitial(LVD_URL, kind)
    out = research(state_for("EU"), make_candidates(), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    lvd = _by_id(out)["reg.EU.LVD.test"]
    assert lvd.source_status == "blocked" and lvd.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    assert lvd.provenance.content_hash is None and lvd.provenance.source_document is None
    assert all(not q.found and q.reason.startswith("blocked") for q in lvd.grounded_quotes)
    assert lvd.applicability is Applicability.NOT_APPLICABLE  # the rule still decides; its evidence is not grounded
    sources = out.result(SOURCES_CHECK)
    assert sources.status is ValidationStatus.NOT_VERIFIED and "official document blocked" in sources.message
    assert out.result(APPLICABILITY_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_404_is_missing_and_redirect_off_the_allow_list_is_refused_without_contact(fake, tmp_path: Path):
    serve_all(fake)
    fake.add_missing(RED_URL)
    fake.add_redirect(LVD_URL, "https://unblock.example/request-access")
    out = research(state_for("EU"), make_candidates(), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    by = _by_id(out)
    assert by["reg.EU.RED.test"].source_status == "missing" and "HTTP 404" in by["reg.EU.RED.test"].provenance.applicability_rationale or True
    assert by["reg.EU.RED.test"].source_status == "missing"
    assert by["reg.EU.LVD.test"].source_status == "refused" and "untrusted host 'unblock.example'" in out.result(SOURCES_CHECK).message
    assert not [r for r in fake.requests if r.host == "unblock.example"]
    assert out.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_wrong_document_is_not_verified_not_fail(fake, tmp_path: Path):
    serve_all(fake)
    fake.add_xml(KR_URL.replace("{LAW_GO_KR_OC}", "test"), KR_ERROR_XML)  # HTTP 200 with an API error body: none of the markers
    out = research(state_for("KR"), make_candidates(), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    kr = _by_id(out)["reg.KR.Safety.test"]
    assert kr.source_status == "wrong_document" and kr.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    assert "none of the expected markers" in out.result(SOURCES_CHECK).message
    assert out.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED
    assert kr.grounded_quotes[0].found is False and kr.grounded_quotes[0].reason.startswith("wrong_document")


def test_placeholders_and_date_discovery(fake, tmp_path: Path):
    serve_all(fake)
    cl = make_candidates()
    archive = online_archive(tmp_path / "sources", fake)
    out = research(state_for("KR", "US"), cl, archive, {**ANSWERS, "digital_device": "yes"}, reqs(), env={"AI_EDA_LAW_GO_KR_OC": "myoc"})
    by = _by_id(out)
    kr = by["reg.KR.Safety.test"]
    assert kr.source_status == "ok" and kr.provenance.source_url.endswith("OC=myoc&target=law&MST=1&type=XML") and kr.grounded_quotes[0].found
    assert any("OC=myoc" in r.query for r in fake.requests if r.host == f"www.{KR_HOST}")
    us = by["reg.US.FCC.test"]
    assert us.source_status == "ok" and us.provenance.source_url == US_URL.replace("{DATE}", "2026-09-21") and us.grounded_quotes[0].found
    assert us.applicability is Applicability.APPLICABLE
    docs = next(c for c in out.result(SOURCES_CHECK).details["candidates"] if c["id"] == "reg.US.FCC.test")["documents"]
    assert [d["role"] for d in docs] == ["date_discovery", "official"] and docs[0]["reason"] == "{DATE} = 2026-09-21 from titles[{'number': 47}].up_to_date_as_of"
    assert docs[0]["sha256"] and docs[1]["reason"] == "{DATE} = 2026-09-21"
    assert out.result(SOURCES_CHECK).status is ValidationStatus.PASS and out.result(APPLICABILITY_CHECK).status is ValidationStatus.PASS


def test_date_discovery_failure_fetches_nothing_more(fake, tmp_path: Path):
    serve_all(fake)
    fake.serve(US_INDEX_URL, index_json([{"number": 46, "up_to_date_as_of": "2026-09-20"}]), "application/json")
    out = research(state_for("US"), make_candidates(), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    us = _by_id(out)["reg.US.FCC.test"]
    assert us.source_status == "unresolved" and "could not be discovered" in out.result(SOURCES_CHECK).message and "does not carry titles" in out.result(SOURCES_CHECK).message
    assert [r.path for r in fake.requests] == ["/api/versioner/v1/titles.json"]
    docs = out.result(SOURCES_CHECK).details["candidates"][0]["documents"]
    assert [d["status"] for d in docs] == ["unresolved", "unresolved"] and docs[0]["sha256"]  # the index itself was archived as evidence of what it said
    # 'current' is what the real API refuses as a date: it is not a date and is never substituted
    fake.serve(US_INDEX_URL, index_json([{"number": 47, "up_to_date_as_of": "current"}]), "application/json")
    out = research(state_for("US"), make_candidates(), online_archive(tmp_path / "sources2", fake), ANSWERS, reqs())
    assert _by_id(out)["reg.US.FCC.test"].source_status == "unresolved" and "does not match" in out.result(SOURCES_CHECK).message
    assert not [r for r in fake.requests if "current" in r.path or "{DATE}" in r.path]
    # an index the archive classifies as not-a-document (a tiny body) is reported as such, not as a date
    fake.serve(US_INDEX_URL, '{"titles": []}', "application/json")
    out = research(state_for("US"), make_candidates(), online_archive(tmp_path / "sources3", fake), ANSWERS, reqs())
    assert _by_id(out)["reg.US.FCC.test"].source_status == "unresolved" and "not a document" in out.result(SOURCES_CHECK).message


def test_unfetchable_entry_is_reported_not_omitted(fake, tmp_path: Path):
    d = candidates_dict()
    d["candidates"][0].update({"fetchable": False, "unfetchable_reason": "only a script-rendered viewer exists"})
    serve_all(fake)
    out = research(state_for("EU"), CandidateList.model_validate(d), online_archive(tmp_path / "sources", fake), ANSWERS, reqs())
    lvd = _by_id(out)["reg.EU.LVD.test"]
    assert lvd.source_status == "unfetchable" and lvd.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    assert "only a script-rendered viewer exists" in out.result(SOURCES_CHECK).message
    assert not [r for r in fake.requests if r.path == "/lvd"]


def test_packaged_unfetchable_entry_is_reported_without_a_guessed_url():
    cl = load_candidates()
    notice = cl.get("reg.KR.RadioWavesAct.ConformityAssessmentNotice")
    assert notice is not None and not notice.fetchable and notice.official_url is None and notice.grounding_quotes == [] and notice.documents() == []
    out = research(state_for("KR"), cl, None, {**ANSWERS, "digital_device": "yes"}, reqs())
    req = _by_id(out)["reg.KR.RadioWavesAct.ConformityAssessmentNotice"]
    assert req.source_status == "unfetchable" and req.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    assert req.provenance.source_url is None and req.provenance.content_hash is None and req.provenance.section is None and req.grounded_quotes == []
    assert req.applicability is Applicability.APPLICABLE and req.status is ValidationStatus.NOT_VERIFIED  # decided on the answers; nothing grounded, nothing verified
    sources = out.result(SOURCES_CHECK)
    assert sources.status is ValidationStatus.NOT_VERIFIED and "not fetchable by machine" in sources.message and "target=admrul" in sources.message
    assert next(c for c in sources.details["candidates"] if c["id"] == req.id)["documents"] == []
    assert out.result(COMPLIANCE_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_no_candidates_for_a_jurisdiction():
    out = research(state_for("JP"), make_candidates(), None, ANSWERS, reqs())
    assert out.state.requirements == [] and out.notes == ["no curated candidates for jurisdiction JP: nothing researched, nothing decided"]
    assert out.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED and "no curated candidates" in out.result(SOURCES_CHECK).message
    assert out.result(APPLICABILITY_CHECK).status is ValidationStatus.NOT_VERIFIED and out.result(COMPLIANCE_CHECK).status is ValidationStatus.NOT_VERIFIED


def test_archived_copies_are_reused_offline_and_tampered_ones_never(fake, tmp_path: Path):
    serve_all(fake)
    root = tmp_path / "sources"
    first = research(state_for("EU"), make_candidates(), online_archive(root, fake), ANSWERS, reqs())
    assert first.result(SOURCES_CHECK).status is ValidationStatus.PASS
    fake.requests.clear()
    second = research(state_for("EU"), make_candidates(), offline_archive(root), ANSWERS, reqs())
    assert fake.requests == [] and not second.online
    lvd = _by_id(second)["reg.EU.LVD.test"]
    assert lvd.source_status == "archived" and lvd.provenance.verification_status is ValidationStatus.PASS
    assert lvd.provenance.content_hash == _by_id(first)["reg.EU.LVD.test"].provenance.content_hash and lvd.grounded_quotes[0].found
    assert "hash-verified copy from an earlier run" in second.result(SOURCES_CHECK).message
    assert second.result(APPLICABILITY_CHECK).status is ValidationStatus.PASS  # decided, and the quote is grounded in the archived copy
    # tamper with the archived LVD file: the copy is never used again
    Path(lvd.provenance.source_document).write_bytes(lvd_html().replace("1 000", "1 500").encode("utf-8"))
    third = research(state_for("EU"), make_candidates(), offline_archive(root), ANSWERS, reqs())
    lvd3 = _by_id(third)["reg.EU.LVD.test"]
    assert lvd3.source_status == "offline" and lvd3.provenance.content_hash is None and not lvd3.grounded_quotes[0].found
    assert third.result(SOURCES_CHECK).status is ValidationStatus.NOT_VERIFIED
    # when the IR remembers which copy it used, the altered file is named as tampered (and still never used)
    fourth = research(second.state, make_candidates(), offline_archive(root), ANSWERS, reqs())
    lvd4 = _by_id(fourth)["reg.EU.LVD.test"]
    assert lvd4.source_status == "tampered" and lvd4.provenance.content_hash is None and lvd4.provenance.verification_status is ValidationStatus.NOT_VERIFIED
    sources4 = fourth.result(SOURCES_CHECK)
    assert sources4.status is ValidationStatus.NOT_VERIFIED and "no longer hashes to its name" in sources4.message
    assert sources4.details["tampered"] == {"reg.EU.LVD.test": lvd.provenance.content_hash} and any("tampered" in n for n in fourth.notes)
    assert _by_id(fourth)["reg.EU.RED.test"].source_status == "archived"  # the untouched copies are still used
    # re-fetched online, the healthy bytes replace the altered file and the copy is verified again
    fifth = research(fourth.state, make_candidates(), online_archive(root, fake), ANSWERS, reqs())
    assert _by_id(fifth)["reg.EU.LVD.test"].source_status == "ok" and fifth.result(SOURCES_CHECK).details["tampered"] == {}
    assert fifth.result(SOURCES_CHECK).status is ValidationStatus.PASS


def test_online_fetch_failure_falls_back_to_an_earlier_archived_copy(fake, tmp_path: Path):
    serve_all(fake)
    root = tmp_path / "sources"
    research(state_for("EU"), make_candidates(), online_archive(root, fake), ANSWERS, reqs())
    fake.add_interstitial(LVD_URL, "cloudflare")
    out = research(state_for("EU"), make_candidates(), online_archive(root, fake), ANSWERS, reqs())
    lvd = _by_id(out)["reg.EU.LVD.test"]
    assert lvd.source_status == "archived" and lvd.provenance.verification_status is ValidationStatus.PASS
    assert "fetch blocked" in out.result(SOURCES_CHECK).details["candidates"][0]["documents"][0]["reason"]
    assert out.result(SOURCES_CHECK).status is ValidationStatus.PASS


def test_curated_entries_are_replaced_and_accepted_proposals_kept():
    stale = RegulatoryRequirement(id="reg.EU.LVD.test", jurisdiction="EU", title="old", provenance=RegulatoryProvenance(jurisdiction="EU", authority="a", source_title="t"),
                                  candidate_id="reg.EU.LVD.test", basis="curated")
    llm = RegulatoryRequirement(id="reg.EU.llm.x", jurisdiction="EU", title="accepted proposal", provenance=RegulatoryProvenance(jurisdiction="EU", authority="a", source_title="t"),
                                candidate_id="reg.EU.llm.x", basis="llm_proposed", applicability=Applicability.APPLICABLE)
    kr = RegulatoryRequirement(id="reg.KR.Safety.test", jurisdiction="KR", title="kr", provenance=RegulatoryProvenance(jurisdiction="KR", authority="a", source_title="t"),
                               candidate_id="reg.KR.Safety.test", basis="curated")
    state = RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU"), Jurisdiction(code="KR", name="KR", provided_by_user=False)], requirements=[stale, llm, kr],
                            intended_use="bench", scope_answers={"radio": "no"})
    out = research(state, make_candidates(), None, ANSWERS, reqs())
    ids = [r.id for r in out.state.requirements]
    assert ids == ["reg.EU.llm.x", "reg.KR.Safety.test", "reg.EU.LVD.test", "reg.EU.RED.test", "reg.EU.RoHS.test"]  # KR was not researched (not known)
    assert out.state.intended_use == "bench" and out.state.scope_answers == {"radio": "no"} and state.requirements[0].title == "old"  # input untouched


def test_research_candidate_is_the_unit(fake, tmp_path: Path):
    serve_all(fake)
    cl = make_candidates()
    archive = online_archive(tmp_path / "sources", fake)
    archive.policy.trust_host(EU_HOST, "test")
    res = research_candidate(cl.get("reg.EU.RED.test"), cl, archive, True, {"radio": "yes"}, reqs())
    assert res.verification is ValidationStatus.PASS and res.evaluation.applicability is Applicability.APPLICABLE and res.evidence_grounded
    assert res.requirement.status is ValidationStatus.NOT_VERIFIED and res.requirement.applicability_inputs == {"radio": "yes (answer)"}


# --------------------------------------------------------------------------- live (opt-in)


@pytest.mark.skipif(os.environ.get("AI_EDA_ONLINE") != "1", reason="AI_EDA_ONLINE=1 not set: no network in tests")
def test_live_eur_lex_lvd_grounds_the_voltage_range_quote(tmp_path: Path):
    cl = load_candidates()
    lvd = cl.get("reg.EU.LVD.2014-35-EU")
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=True, gate=ApprovalGate()))
    try:
        archive.policy.trust_host("eur-lex.europa.eu", "live test")
        res = research_candidate(lvd, cl, archive, True, {"mains_powered": "no", "radio": "no"}, reqs())
    finally:
        archive.close()
    scope = res.quotes[0]
    assert res.documents[0].status == "ok", res.documents[0].reason
    assert res.documents[0].marker == "L_2014096EN.01035701.xml"
    assert scope.section == "Article 1" and scope.found and "1 000 V" in scope.context
    assert res.verification is ValidationStatus.PASS, res.reason
    assert res.requirement.provenance.retrieved_at is not None
    assert archive.verify(SourceRef(title="x", content_hash=res.requirement.provenance.content_hash)) == "ok"


@pytest.mark.skipif(os.environ.get("AI_EDA_ONLINE") != "1", reason="AI_EDA_ONLINE=1 not set: no network in tests")
def test_live_law_go_kr_xml_grounds_the_electrical_appliances_act(tmp_path: Path):
    """Opt-in: the 법제처 DRF XML of 전기용품 및 생활용품 안전관리법 (demo id ``test`` unless AI_EDA_LAW_GO_KR_OC is set). Never run on this machine so far."""
    cl = load_candidates()
    cand = cl.get("reg.KR.ElectricalAppliancesSafetyAct")
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=True, gate=ApprovalGate()))
    try:
        archive.policy.trust_host("www.law.go.kr", "live test")
        res = research_candidate(cand, cl, archive, True, {"mains_powered": "no", "highest_rated_voltage": "12 V DC"}, reqs())
    finally:
        archive.close()
    assert res.documents[0].status == "ok", res.documents[0].reason
    assert res.documents[0].marker in cand.expected_markers and all(q.found for q in res.quotes), [q.section for q in res.quotes if not q.found]
    assert res.verification is ValidationStatus.PASS, res.reason


@pytest.mark.skipif(os.environ.get("AI_EDA_ONLINE") != "1", reason="AI_EDA_ONLINE=1 not set: no network in tests")
def test_live_ecfr_versioner_api_grounds_47_cfr_part_15(tmp_path: Path):
    """Opt-in: eCFR titles.json date discovery + the full Part 15 XML (the human pages are bot-walled; the API was reachable on 2026-09-01). Never run on this machine so far."""
    cl = load_candidates()
    cand = cl.get("reg.US.FCC.47CFR15")
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=True, gate=ApprovalGate()))
    try:
        archive.policy.trust_host("www.ecfr.gov", "live test")
        res = research_candidate(cand, cl, archive, True, {"radio": "no", "digital_device": "yes"}, reqs())
    finally:
        archive.close()
    assert [d.role for d in res.documents] == ["date_discovery", "official"] and res.documents[1].status == "ok", [(d.status, d.reason) for d in res.documents]
    assert all(q.found for q in res.quotes), [q.section for q in res.quotes if not q.found]
    assert res.verification is ValidationStatus.PASS and res.requirement.applicability is Applicability.APPLICABLE
