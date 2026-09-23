"""The document archive against a local fake of the web (tests/fake_sources.py); nothing here touches the network.

What is proven: no socket without the user's online session; only trusted
origins and trusted redirect targets are contacted; https only; PDFs, HTML,
XML and text are archived by content hash with a complete meta.json and
extracted deterministically; quotes are grounded with the requirement
stage's normalisation; interstitials, 404s, timeouts and tampered files are
classified honestly; user files and user URLs enter the same way.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_eda.errors import ApprovalRequiredError
from ai_eda.ir import SourceRef
from ai_eda.security.approval import ApprovalGate, ExternalAction
from ai_eda.tools.sources import (
    ONLINE_SESSION_DETAIL,
    ArchiveError,
    DocumentArchive,
    DocumentMissingError,
    NetworkPolicy,
    TamperedDocumentError,
    host_key,
    normalise_url,
    sha256_of,
)
from ai_eda.tools.sources.archive import MIN_HTML_TEXT_CHARS, text_hash
from ai_eda.tools.sources.extract import (
    HTML_EXTRACTOR,
    HTML_EXTRACTOR_VERSION,
    PDF_EXTRACTOR,
    PDF_EXTRACTOR_VERSION,
    XML_EXTRACTOR,
    extract_pdf,
    sniff_kind,
    strip_html,
)
from tests.fake_sources import INTERSTITIALS, FakeSources
from tests.pdf_fixture import DATASHEET_PAGES, build_pdf, datasheet_pdf

TI_PDF = "https://www.ti.com/lit/ds/symlink/lm2931-n.pdf"
LVD_HTML = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32014L0035"
TRUSTED = {"www.ti.com", "eur-lex.europa.eu", "www.vishay.com", "datasheet.lcsc.com", "www.lcsc.com", "www.ecfr.gov", "www.law.go.kr"}

LVD_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>L_2014096EN.01035701.xml</title>
<style>.oj-normal { font-size: 10pt }</style><script>var tracking = "do not extract me"; window.__cfg = {a: 1};</script></head>
<body><div class="eli-container"><p class="oj-doc-ti">DIRECTIVE 2014/35/EU OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL</p>
<p class="oj-ti-art">Article 1</p><p class="oj-sti-art">Objective and scope</p>
<p class="oj-normal">The aim of this Directive is to ensure that electrical equipment on the market fulfils the requirements providing
for a high level of protection of health and safety of persons, and of domestic animals and property, while guaranteeing the
functioning of the internal market.</p>
<p class="oj-normal">This Directive shall apply to electrical equipment designed for use with a voltage rating of between 50 and
1&nbsp;000 V for alternating current and between 75 and 1&nbsp;500 V for direct current, other than the equipment and phenomena
listed in Annex II.</p><table><tr><td>Lead</td><td>(0,1&nbsp;%)</td></tr></table><noscript>enable your browser</noscript>
</div></body></html>"""

JEONPA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<법령 법령키="0017322025100121065"><기본정보><법령ID>001732</법령ID><법령명_한글><![CDATA[전파법]]></법령명_한글><시행일자>20260102</시행일자></기본정보>
<조문><조문단위 조문키="0058002"><조문번호>58</조문번호><조문내용><![CDATA[제58조의2(적합성평가) ① 방송통신기자재와 전자파장해를 주거나 전자파로부터 영향을 받는 기자재를 제조 또는 판매하거나 수입하려는 자는 해당 기자재에 대하여 적합인증, 적합등록 또는 자기적합확인을 받아야 한다.]]></조문내용>
<항><항번호>②</항번호><항내용><![CDATA[② 전파환경 및 방송통신망 등에 위해를 줄 우려가 있는 기자재는 적합인증을 받아야 한다.]]></항내용></항></조문단위></조문></법령>"""


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


def online_policy(**kw) -> NetworkPolicy:
    kw.setdefault("gate", ApprovalGate())
    kw.setdefault("trusted_hosts", set(TRUSTED))
    return NetworkPolicy(approved=True, **kw)


def make_archive(tmp_path: Path, fake: FakeSources, policy: NetworkPolicy, **kw) -> DocumentArchive:
    return DocumentArchive(tmp_path / "sources", policy, client=fake.client(), **kw)


def log_lines(archive: DocumentArchive) -> list[dict]:
    if not archive.log_path.exists():
        return []
    return [json.loads(line) for line in archive.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- approval


def test_refused_without_online_session_opens_no_socket(fake, tmp_path):
    gate = ApprovalGate()
    policy = NetworkPolicy(approved=False, trusted_hosts={"www.ti.com"}, gate=gate)
    fake.add_pdf(TI_PDF, DATASHEET_PAGES)
    archive = make_archive(tmp_path, fake, policy)
    assert not archive.online
    with pytest.raises(ApprovalRequiredError) as ei:
        archive.fetch(TI_PDF, purpose="datasheet of R1", expect="pdf")
    assert ei.value.action == ExternalAction.NETWORK_FETCH
    assert ei.value.detail == ONLINE_SESSION_DETAIL
    assert fake.requests == [] and fake.client_urls == []
    assert gate.audit == [{"event": "denied", "action": ExternalAction.NETWORK_FETCH, "detail": ONLINE_SESSION_DETAIL}]
    assert archive.attempts == {}
    assert log_lines(archive) == []
    assert sorted(p.name for p in archive.root.iterdir()) == []


def test_from_cli_online_grants_and_consumes_network_fetch_once():
    gate = ApprovalGate()
    policy = NetworkPolicy.from_cli(True, trusted_hosts={"www.ti.com"}, gate=gate)
    assert policy.approved
    events = [(e["event"], e["action"], e["detail"]) for e in gate.audit]
    assert events == [("grant", ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL), ("consume", ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL)]
    assert gate.audit[0]["by"] == "cli --online"
    policy.require_online()  # already approved: no second consumption needed
    assert len(gate.audit) == 2
    offline_gate = ApprovalGate()
    offline = NetworkPolicy.from_cli(False, gate=offline_gate)
    assert not offline.approved and offline_gate.audit == []
    with pytest.raises(ApprovalRequiredError):
        offline.require_online()
    assert offline_gate.audit[-1]["event"] == "denied"


def test_direct_gate_grant_counts_as_the_users_approval():
    gate = ApprovalGate()
    policy = NetworkPolicy(approved=False, gate=gate)
    gate.grant(ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL, approved_by="user")
    policy.require_online()
    assert policy.approved and gate.audit[-1]["event"] == "consume"


# --------------------------------------------------------------------------- trust


def test_refused_for_untrusted_host(fake, tmp_path):
    archive = make_archive(tmp_path, fake, online_policy(trusted_hosts={"www.ti.com"}))
    url = "https://model-invented.example.com/lm7805.pdf"
    fake.add_pdf(url, DATASHEET_PAGES)
    out = archive.fetch(url, purpose="datasheet proposed by a model", expect="pdf")
    assert out.status == "refused" and out.document is None
    assert "model-invented.example.com" in out.reason and "not a trusted origin" in out.reason
    assert fake.requests == [] and fake.client_urls == []
    assert [line["status"] for line in log_lines(archive)] == ["refused"]
    # a second call in the same run does not try again either
    assert archive.fetch(url, purpose="again", expect="pdf") is out


def test_refused_on_redirect_to_untrusted_host(fake, tmp_path):
    ecfr = "https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-15"
    unblock = "https://unblock.federalregister.gov/?url=" + ecfr
    fake.add_redirect(ecfr, unblock, status=302)
    fake.add_html(unblock, INTERSTITIALS["unblock"][2])
    archive = make_archive(tmp_path, fake, online_policy())
    out = archive.fetch(ecfr, purpose="47 CFR Part 15", expect="html")
    assert out.status == "refused"
    assert "unblock.federalregister.gov" in out.reason and "untrusted host" in out.reason
    assert out.redirects == [{"status": 302, "url": ecfr, "location": unblock, "target": unblock}]
    assert out.final_url == ecfr and out.http_status == 302
    assert [r.host for r in fake.requests] == ["www.ecfr.gov"]
    assert fake.requests_for("unblock.federalregister.gov") == []


def test_redirects_same_host_relative_and_cross_host_trusted(fake, tmp_path):
    doc_query = "https://www.vishay.com/doc?85881"
    fake.add_redirect(doc_query, "https://www.vishay.com/docs/85881/", status=301)
    fake.add_redirect("https://www.vishay.com/docs/85881/", "/docs/85881/smf5v0atosmf58a.pdf", status=307)
    data = fake.add_pdf("https://www.vishay.com/docs/85881/smf5v0atosmf58a.pdf", [["SMF5V0A thru SMF58A", "Vishay General Semiconductor"]])
    mirror = "https://datasheet.lcsc.com/szlcsc/C25792.pdf"
    fake.add_redirect(mirror, "https://www.lcsc.com/datasheet/C25792.pdf", status=301)
    fake.add_pdf("https://www.lcsc.com/datasheet/C25792.pdf", [["LCSC C25792 datasheet mirror page one"]])
    archive = make_archive(tmp_path, fake, online_policy())
    out = archive.fetch(doc_query, purpose="D1 datasheet", expect="pdf")
    assert out.ok, out.reason
    assert out.final_url == "https://www.vishay.com/docs/85881/smf5v0atosmf58a.pdf"
    assert [r["status"] for r in out.redirects] == [301, 307]
    assert out.redirects[1]["location"] == "/docs/85881/smf5v0atosmf58a.pdf"
    assert out.redirects[1]["target"] == out.final_url
    assert out.document.meta["redirects"] == out.redirects and out.document.final_url == out.final_url
    assert out.document.sha256 == sha256_of(data)
    out2 = archive.fetch(mirror, purpose="C25792", expect="pdf")
    assert out2.ok and out2.final_url == "https://www.lcsc.com/datasheet/C25792.pdf"
    assert [r.host for r in fake.requests] == ["www.vishay.com", "www.vishay.com", "www.vishay.com", "datasheet.lcsc.com", "www.lcsc.com"]


def test_redirect_loop_is_an_error(fake, tmp_path):
    a, b = "https://www.ti.com/a", "https://www.ti.com/b"
    fake.add_redirect(a, b)
    fake.add_redirect(b, a)
    out = make_archive(tmp_path, fake, online_policy()).fetch(a, purpose="loop")
    assert out.status == "error" and "loop" in out.reason
    assert len(fake.requests) == 2


def test_user_url_is_trusted_exactly_not_its_host(fake, tmp_path):
    user_url = "http://www.jst-mfg.com/product/pdf/eng/ePH.pdf"  # as KiCad footprints carry it: plain http
    https_url = "https://www.jst-mfg.com/product/pdf/eng/ePH.pdf"
    fake.add_pdf(https_url, [["PH connector 2.0 mm pitch", "B2B-PH-K-S  PHR-2  SPH-002T-P0.5S"]])
    other = "https://www.jst-mfg.com/product/pdf/eng/eXA1.pdf"
    fake.add_pdf(other, [["XA connector"]])
    policy = online_policy(trusted_hosts=set(), user_urls={"J1": user_url})
    archive = make_archive(tmp_path, fake, policy)
    out = archive.fetch(user_url, purpose="J1 datasheet", expect="pdf")
    assert out.ok, out.reason
    assert "J1" in out.trusted_by and "http upgraded to https" in out.url_note
    assert out.normalised_url == https_url and out.document.url == user_url and out.document.final_url == https_url
    assert fake.client_urls == [https_url]  # the client never asked for plain http
    assert out.document.find_quote("B2B-PH-K-S")[0].page == 1
    refused = archive.fetch(other, purpose="model-proposed sibling URL on the same host", expect="pdf")
    assert refused.status == "refused" and fake.hits(other) == 0
    # a same-host redirect from a user URL stays on the user's origin
    user2 = "https://www.jst-mfg.com/doc?ePH"
    fake.add_redirect(user2, https_url)
    policy2 = online_policy(trusted_hosts=set(), user_urls={"J1": user2}, gate=ApprovalGate())
    out2 = make_archive(tmp_path / "b", fake, policy2).fetch(user2, purpose="J1", expect="pdf")
    assert out2.ok and out2.final_url == https_url


def test_policy_rules_and_url_normalisation():
    p = NetworkPolicy(approved=False, trusted_hosts={"WWW.TI.com"}, user_urls={"R1": "www.diodes.com/assets/Datasheets/AP2127.pdf", "bad": "~", "typo": "hhttps://www.onsemi.com/x.pdf"})
    assert p.trusted_hosts == {"ti.com"} and host_key("www.TI.com.") == "ti.com"
    assert p.invalid_user_urls == {"bad": "no URL", "typo": "unsupported scheme 'hhttps' (only https is fetched)"}
    assert p.trusted_origin("https://ti.com/lit/x.pdf") == ("ti.com", "trusted host given to the policy")
    origin, why = p.trusted_origin("https://www.diodes.com/assets/Datasheets/AP2127.pdf")
    assert origin == "diodes.com" and "R1" in why
    assert p.trusted_origin("https://www.diodes.com/assets/Datasheets/AP2204.pdf")[0] is None
    assert p.trusted_origin("https://assets.nexperia.com/x.pdf")[0] is None  # subdomains are distinct origins
    assert p.redirect_allowed("ti.com", "https://www.ti.com/lit/ds/symlink/x.pdf?ts=1") is None
    assert p.redirect_allowed("diodes.com", "https://ti.com/x") is None
    assert "untrusted host 'yageogroup.com'" in p.redirect_allowed("yageo.com", "https://www.yageogroup.com/")
    p.trust_host("https://assets.nexperia.com/documents/x.pdf", "KiCad Datasheet field of Diode:PESD5V0L1ULD")
    assert p.trust_reasons["assets.nexperia.com"].startswith("KiCad")
    assert normalise_url(" \"https://assets.nexperia.com/documents/data-sheet/PESD5V0L1ULD.pdf ") == ("https://assets.nexperia.com/documents/data-sheet/PESD5V0L1ULD.pdf", None)
    assert normalise_url("http://WWW.TI.COM/lit/gpn/TLV767#page=3") == ("https://www.ti.com/lit/gpn/TLV767", "http upgraded to https (plain http is never used)")
    assert normalise_url("www.st.com/resource/en/datasheet/x.pdf")[1] == "scheme-less URL taken as https"
    assert normalise_url("www.st.com:8443/x.pdf")[0] == "https://www.st.com:8443/x.pdf" and normalise_url("https://ST.com:443/x")[0] == "https://st.com/x"
    for bad in ("", "~", "ftp://x/y", "file:///C:/x.pdf", "https:///nohost", "mailto:a@b", "https://user:pw@ti.com/x"):
        with pytest.raises(ValueError):
            normalise_url(bad)
    p.trust_host("www.jst-mfg.com:443/product", "footprint descr URL")
    assert "jst-mfg.com" in p.trusted_hosts
    assert p.describe() == {"approved": False, "trusted_hosts": p.trust_reasons, "user_urls": p.user_urls, "invalid_user_urls": p.invalid_user_urls}


# --------------------------------------------------------------------------- documents


def test_pdf_fetched_pages_extracted_quote_found_with_shared_normalisation(fake, tmp_path):
    data = fake.add_pdf(TI_PDF, DATASHEET_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    out = archive.fetch(TI_PDF, purpose="datasheet of U1", expect="pdf")
    assert out.ok and out.status == "ok" and out.reason is None, out.reason
    doc = out.document
    assert doc.kind == "pdf" and doc.page_count == 3 and doc.text_available
    assert doc.sha256 == sha256_of(data) and doc.path.name == doc.sha256[7:] + ".pdf" and doc.path.read_bytes() == data
    assert doc.meta["extractor"] == PDF_EXTRACTOR and doc.meta["extractor_version"] == PDF_EXTRACTOR_VERSION
    assert doc.meta["library_version"] == __import__("pypdf").__version__
    assert "LM2931-N Series Low Dropout Regulators" in doc.pages[0]
    # family number on page 1, orderable code on page 3; whitespace-free matching; token boundaries; case sensitivity
    hits = doc.find_quote("LM2931-N")
    assert [h.page for h in hits] == [1] and hits[0].matched == "LM2931-N" and "[LM2931-N]" in hits[0].context
    assert [h.page for h in doc.find_quote("LM2931AZ-5.0/NOPB")] == [3]
    assert doc.find_quote("LM2931AZ-5.0/NOPB", page=1) == [] and doc.find_quote("LM2931AZ-5.0/NOPB", page=3)[0].offset > 0
    assert doc.find_quote("LM2931AZ-5.0/NOPB", page=99) == [] and doc.find_quote("", page=1) == []
    assert doc.find_quote("Outputvoltage5V") != [] and doc.find_quote("output voltage 5 V") == []  # whitespace free, case exact
    assert doc.find_quote("LM2931A") == [] and doc.find_quote("2931") == []  # inside a token: not a hit
    ci = doc.find_quote("lm2931az-5.0/nopb", ignore_case=True)  # part numbers: ASCII case ignored, offsets exact, original text reported
    assert [h.page for h in ci] == [3] and ci[0].matched == "LM2931AZ-5.0/NOPB" and doc.find_quote("lm2931az-5.0/nopb") == []
    assert doc.pages[2][ci[0].offset:ci[0].offset + len(ci[0].matched)] == ci[0].matched
    assert doc.find_quote("lm2931a", ignore_case=True) == []  # still a token-boundary match
    assert [h.page for h in doc.find_quote("Orderable device")] == [3]
    ref = doc.source_ref(title="LM2931-N datasheet", section=hits[0].section, authority="Texas Instruments")
    assert ref == SourceRef(title="LM2931-N datasheet", url=TI_PDF, authority="Texas Instruments", section="page 1",
                            document_path=str(doc.path), content_hash=doc.sha256, retrieved_at=doc.retrieved_at)
    assert ref.retrieved_at is not None and ref.retrieved_at.tzinfo is not None
    assert SourceRef.from_document(doc, section="page 3", title="ds").section == "page 3"
    assert archive.verify(ref) == "ok"
    assert fake.requests[0].headers["accept"].startswith("application/pdf")


def test_user_agent_names_the_project_and_only_https_is_requested(fake, tmp_path):
    fake.add_pdf(TI_PDF, DATASHEET_PAGES)
    make_archive(tmp_path, fake, online_policy()).fetch("http://www.ti.com/lit/ds/symlink/lm2931-n.pdf", purpose="ds", expect="pdf")
    assert fake.requests[0].user_agent.startswith("ai-eda-engineer/0.0.1 (")
    assert fake.client_urls == [TI_PDF]


def test_html_stripped_scripts_dropped_block_newlines(fake, tmp_path):
    fake.add_html(LVD_HTML, LVD_PAGE)
    archive = make_archive(tmp_path, fake, online_policy())
    out = archive.fetch(LVD_HTML, purpose="EU LVD official text", expect="html")
    assert out.ok, out.reason
    doc = out.document
    text = doc.pages[0]
    assert doc.kind == "html" and doc.page_count == 1 and doc.path.suffix == ".html"
    assert doc.meta["extractor"] == HTML_EXTRACTOR and doc.meta["extractor_version"] == HTML_EXTRACTOR_VERSION
    assert doc.title == "L_2014096EN.01035701.xml" and doc.meta["encoding"] == "utf-8"
    assert "do not extract me" not in text and "font-size" not in text and "enable your browser" not in text
    assert "Article 1\nObjective and scope\nThe aim of this Directive" in text
    assert "Lead\n(0,1 %)" in text  # &nbsp; became a plain space, table cells on their own lines
    hit = doc.find_quote("between 50 and 1 000 V for alternating current and between 75 and 1 500 V for direct current")
    assert hit and hit[0].page == 1
    assert doc.find_quote("Article 1")[0].context.startswith("…")
    assert text.startswith("L_2014096EN.01035701.xml\nDIRECTIVE 2014/35/EU") and "\n\n" not in text
    assert out.text_head.startswith("L_2014096EN.01035701.xml DIRECTIVE 2014/35/EU")


def test_xml_with_korean_element_names_and_cdata(fake, tmp_path):
    url = "https://www.law.go.kr/DRF/lawService.do?OC=test&target=law&type=XML&LM=전파법"
    fake.add_xml(url, JEONPA_XML)
    out = make_archive(tmp_path, fake, online_policy()).fetch(url, purpose="전파법 official XML", expect="html")
    assert out.ok, out.reason
    doc = out.document
    assert doc.kind == "xml" and doc.meta["extractor"] == XML_EXTRACTOR and doc.path.suffix == ".xml"
    assert doc.find_quote("제58조의2(적합성평가)")[0].page == 1
    assert doc.find_quote("적합인증, 적합등록 또는 자기적합확인을 받아야 한다") != []
    assert "<![CDATA[" not in doc.pages[0] and "법령키" not in doc.pages[0] and "전파법" in doc.pages[0]
    assert fake.requests[0].path == "/DRF/lawService.do"


def test_korean_text_round_trip_utf8_html_and_euc_kr_text(fake, tmp_path):
    html_url = "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=276591"
    body = "제1조(목적) 이 법은 전기용품 및 생활용품의 안전관리에 관한 사항을 규정함으로써 국민의 생명·신체 및 재산을 보호하고 소비자의 이익과 안전을 도모함을 목적으로 한다. "
    fake.add_html(html_url, "<html><head><meta charset='utf-8'><title>전기용품 및 생활용품 안전관리법</title></head><body><h1>전기용품 및 생활용품 안전관리법</h1>"
                  + "".join(f"<p>{body}</p>" for _ in range(3)) + "</body></html>")
    text_url = "https://www.law.go.kr/notice.txt"
    fake.add_text(text_url, "안전인증대상전기용품: 제5조 ① 안전인증대상전기용품의 제조업자 또는 수입업자는 안전인증을 받아야 한다.\n" * 3, charset="euc-kr")
    archive = make_archive(tmp_path, fake, online_policy())
    h = archive.fetch(html_url, purpose="KR statute page", expect="html")
    t = archive.fetch(text_url, purpose="KR plain text", expect="any")
    assert h.ok and t.ok, (h.reason, t.reason)
    assert h.document.title == "전기용품 및 생활용품 안전관리법" and h.document.find_quote("국민의 생명·신체 및 재산을 보호")[0].page == 1
    assert t.document.meta["encoding"] == "euc-kr" and t.document.find_quote("안전인증을 받아야 한다") != []
    assert "안전인증대상전기용품" in t.document.pages[0]
    for doc in (h.document, t.document):
        again = DocumentArchive(archive.root, NetworkPolicy(approved=False, gate=ApprovalGate())).load(doc.sha256)
        assert again.pages == doc.pages and again.text_matches_meta and again.meta == doc.meta
        assert json.loads(doc.path.with_name(doc.path.name.split(".")[0] + ".meta.json").read_text(encoding="utf-8"))["title"] == doc.title


# --------------------------------------------------------------------------- honest failures


@pytest.mark.parametrize("kind", sorted(INTERSTITIALS))
def test_interstitial_pages_are_blocked(fake, tmp_path, kind):
    url = f"https://www.ti.com/{kind}/lm7805.pdf"
    fake.add_interstitial(url, kind)
    archive = make_archive(tmp_path, fake, online_policy())
    out = archive.fetch(url, purpose="datasheet", expect="any")
    assert out.status == "blocked", (kind, out.reason)
    assert out.document is None and out.reason and out.http_status == INTERSTITIALS[kind][0]
    assert out.body_sha256 == sha256_of(INTERSTITIALS[kind][2].encode("utf-8"))
    assert sorted(p.name for p in archive.root.iterdir()) == ["fetch_log.jsonl"]  # nothing archived
    assert log_lines(archive)[0]["status"] == "blocked" and log_lines(archive)[0]["text_head"]


def test_html_where_a_pdf_was_expected_is_blocked(fake, tmp_path):
    fake.add_html(TI_PDF, LVD_PAGE)  # a real-looking page, but not the PDF that was asked for
    out = make_archive(tmp_path, fake, online_policy()).fetch(TI_PDF, purpose="datasheet", expect="pdf")
    assert out.status == "blocked" and "expected a PDF but received html" in out.reason and out.document is None


def test_pdf_by_bytes_is_accepted_whatever_the_content_type_says(fake, tmp_path):
    fake.serve("https://www.ti.com/octet.pdf", datasheet_pdf(), "application/octet-stream")
    out = make_archive(tmp_path, fake, online_policy()).fetch("https://www.ti.com/octet.pdf", purpose="ds", expect="pdf")
    assert out.ok and out.document.kind == "pdf"


def test_404_is_missing_even_with_a_full_home_page_body(fake, tmp_path):
    url = "https://www.ti.com/lit/ds/symlink/lm7805.pdf"
    fake.add_missing(url)
    out = make_archive(tmp_path, fake, online_policy()).fetch(url, purpose="datasheet", expect="pdf")
    assert out.status == "missing" and out.http_status == 404 and out.size > 4000 and out.document is None
    assert "HTTP 404" in out.reason and "Sorry, we could not find that page" in out.text_head


def test_timeout_is_an_error_and_is_not_retried(fake, tmp_path):
    url = "https://www.ti.com/slow.pdf"
    fake.add_slow(url, delay=2.5, body=datasheet_pdf(), content_type="application/pdf")
    archive = make_archive(tmp_path, fake, online_policy(), timeout=0.4)
    out = archive.fetch(url, purpose="datasheet", expect="pdf")
    assert out.status == "error" and "timed out after 0.4 s" in out.reason and out.elapsed_s < 2.0
    assert archive.fetch(url, purpose="datasheet", expect="pdf") is out
    assert fake.hits(url) == 1 and len(fake.requests) == 1


def test_small_body_and_binary_content(fake, tmp_path):
    fake.add_text("https://www.ti.com/ok.txt", "OK")
    fake.serve("https://www.ti.com/blob.bin", os.urandom(4096), "application/octet-stream")
    fake.serve("https://www.ti.com/fake.pdf", b"MZ" + os.urandom(4096), "application/octet-stream")
    fake.serve("https://www.ti.com/teapot", b"x" * 500, "text/plain", status=418)
    archive = make_archive(tmp_path, fake, online_policy())
    assert archive.fetch("https://www.ti.com/ok.txt", purpose="p").status == "blocked"
    assert "not a document" in archive.fetch("https://www.ti.com/ok.txt", purpose="p").reason
    blob = archive.fetch("https://www.ti.com/blob.bin", purpose="p")
    assert blob.status == "error" and "unsupported content" in blob.reason
    named = archive.fetch("https://www.ti.com/fake.pdf", purpose="p", expect="pdf")
    assert named.status == "blocked" and "expected a PDF" in named.reason  # .pdf in the name proves nothing
    assert archive.fetch("https://www.ti.com/teapot", purpose="p").status == "error"


def test_unextractable_pdf_is_archived_but_has_no_text(fake, tmp_path):
    url = "https://www.ti.com/encrypted.pdf"
    fake.serve(url, b"%PDF-1.7\n" + os.urandom(2000) + b"\n%%EOF\n", "application/pdf")
    out = make_archive(tmp_path, fake, online_policy()).fetch(url, purpose="ds", expect="pdf")
    assert out.status == "ok" and out.document is not None
    assert not out.document.text_available and out.document.extraction_error and out.reason.startswith("text extraction failed")
    assert out.document.find_quote("anything") == []
    assert extract_pdf(b"%PDF-garbage").error


def test_tampered_file_detected_on_load(fake, tmp_path):
    fake.add_pdf(TI_PDF, DATASHEET_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    doc = archive.fetch(TI_PDF, purpose="ds", expect="pdf").document
    ref = doc.source_ref(title="ds", section="page 1")
    reloaded = DocumentArchive(archive.root, NetworkPolicy(approved=False, gate=ApprovalGate())).load("sha256:" + doc.sha256[7:].upper())
    assert reloaded.pages == doc.pages and reloaded.sha256 == doc.sha256 and reloaded.text_matches_meta
    doc.path.write_bytes(build_pdf([["LM2931-N Series Low Dropout Regulators", "Output voltage 5.5 V"]]))  # an edited datasheet
    with pytest.raises(TamperedDocumentError) as ei:
        archive.load(doc.sha256)
    assert "altered" in str(ei.value)
    assert archive.verify(ref) == "tampered"
    doc.path.unlink()
    assert archive.verify(ref) == "missing"
    with pytest.raises(DocumentMissingError):
        archive.load(doc.sha256)
    assert archive.verify(SourceRef(title="no hash")) == "unarchived"
    assert archive.verify(SourceRef(title="junk", content_hash="sha256:not-a-hash")) == "missing"


def test_add_file_for_a_local_pdf_works_offline(tmp_path):
    p = tmp_path / "LM2931-N (downloaded).pdf"
    data = datasheet_pdf()
    p.write_bytes(data)
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate()))
    doc = archive.add_file(p, title="LM2931-N datasheet", retrieved_at="2026-09-20", authority="Texas Instruments")
    assert doc.sha256 == sha256_of(data) and doc.path == archive.root / (doc.sha256[7:] + ".pdf")
    assert doc.meta["source"] == "user_file" and doc.meta["retrieved_at"] == "2026-09-20" and doc.url is None
    assert doc.meta["original_path"] == str(p.resolve()) and doc.meta["authority"] == "Texas Instruments" and doc.title == "LM2931-N datasheet"
    assert doc.find_quote("LM2931AZ-5.0/NOPB")[0].page == 3
    ref = doc.source_ref(section="page 3")
    assert ref.title == "LM2931-N datasheet" and ref.authority == "Texas Instruments" and ref.url is None and ref.content_hash == doc.sha256
    assert archive.verify(ref) == "ok" and archive.documents() == [doc.sha256]
    when = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)
    assert archive.add_file(p, title="again", retrieved_at=when).meta["retrieved_at"] == when.isoformat()
    assert archive.load(doc.sha256).meta["history"][0]["retrieved_at"] == "2026-09-20"
    (tmp_path / "blob.bin").write_bytes(os.urandom(300))
    with pytest.raises(ArchiveError):
        archive.add_file(tmp_path / "blob.bin", title="x", retrieved_at="2026-09-20")
    with pytest.raises(ArchiveError):
        archive.add_file(p, title="x", retrieved_at="yesterday")
    with pytest.raises(ArchiveError):
        archive.add_file(tmp_path / "nope.pdf", title="x", retrieved_at="2026-09-20")


def test_meta_json_contents_and_deterministic_sha256_naming(fake, tmp_path):
    data = fake.add_pdf(TI_PDF, DATASHEET_PAGES)
    a = make_archive(tmp_path / "run1", fake, online_policy())
    b = make_archive(tmp_path / "run2", fake, online_policy(gate=ApprovalGate()))
    da = a.fetch(TI_PDF, purpose="U1 datasheet", expect="pdf").document
    db = b.fetch(TI_PDF, purpose="U1 datasheet", expect="pdf").document
    assert da.path.name == db.path.name == sha256_of(data)[7:] + ".pdf"
    assert da.sha256 == db.sha256 and da.text_sha256 == db.text_sha256 == text_hash(da.pages)
    meta = json.loads(da.path.with_name(da.sha256[7:] + ".meta.json").read_text(encoding="utf-8"))
    assert meta == da.meta
    required = {"url", "final_url", "redirects", "retrieved_at", "status", "content_type", "size", "sha256", "extractor", "extractor_version", "text_sha256"}
    assert required <= set(meta)
    assert meta["url"] == TI_PDF and meta["final_url"] == TI_PDF and meta["redirects"] == [] and meta["status"] == "ok"
    assert meta["content_type"] == "application/pdf" and meta["size"] == len(data) and meta["sha256"] == da.sha256
    assert meta["kind"] == "pdf" and meta["ext"] == "pdf" and meta["pages"] == 3 and meta["http_status"] == 200
    assert meta["purpose"] == "U1 datasheet" and meta["expect"] == "pdf" and meta["source"] == "network" and meta["schema"] == 1
    assert meta["user_agent"].startswith("ai-eda-engineer/") and meta["extraction_error"] is None and meta["history"] == []
    assert datetime.fromisoformat(meta["retrieved_at"]).tzinfo is not None
    assert a.lookup("http://www.ti.com/lit/ds/symlink/lm2931-n.pdf").sha256 == da.sha256 and a.lookup("https://www.ti.com/other") is None
    lines = log_lines(a)
    assert len(lines) == 1 and lines[0]["status"] == "ok" and lines[0]["sha256"] == da.sha256 and lines[0]["archived_path"] == str(da.path)
    assert fake.hits(TI_PDF) == 2  # one attempt per run: two runs, two fetches
    # same bytes from a second URL in one run: one file, meta rewritten with the earlier fetch kept as history
    alias = "https://www.ti.com/lit/gpn/lm2931-n"
    fake.serve(alias, data, "application/pdf")
    dc = a.fetch(alias, purpose="alias", expect="pdf").document
    assert dc.path == da.path and dc.meta["url"] == alias and dc.meta["history"][0]["url"] == TI_PDF
    assert sorted(p.name for p in a.root.iterdir()) == sorted([da.path.name, da.sha256[7:] + ".meta.json", "fetch_log.jsonl"])


def test_archive_is_a_context_manager_and_owns_only_its_own_client(fake, tmp_path):
    with DocumentArchive(tmp_path / "s", online_policy()) as archive:
        assert archive._client is None and archive._owns_client
        client = archive._http()
        assert client.timeout.read == archive.timeout and archive._http() is client
    assert archive._client is None
    shared = fake.client()
    with DocumentArchive(tmp_path / "t", online_policy(gate=ApprovalGate()), client=shared) as archive2:
        assert not archive2._owns_client
    assert not shared.is_closed


# --------------------------------------------------------------------------- extraction helpers


def test_sniff_kind_and_strip_html_rules():
    assert sniff_kind(b"%PDF-1.4 ...", "text/html") == "pdf"
    assert sniff_kind(b"<!DOCTYPE html><html>", None) == "html" and sniff_kind(b"<html>", "") == "html"
    assert sniff_kind(b"<?xml version='1.0'?><a/>", None) == "xml" and sniff_kind(b"{}", "application/json") == "json"
    assert sniff_kind(b"hello", "text/plain; charset=utf-8") == "text" and sniff_kind(b"\x00\x01", "application/octet-stream") == "binary"
    assert sniff_kind(b"\x00\x01", None, "lm7805.pdf") == "binary" and sniff_kind(b"plain words", None, "notes.txt") == "text"
    assert sniff_kind(b"<r/>", "application/rss+xml") == "xml" and sniff_kind(b"[]", "application/ld+json") == "json"
    text, title = strip_html("<html><head><title> A  title </title><script>x()</script></head><body><div>one</div><p>two&amp;three</p>"
                             "<span>four</span> <b>five</b><br/>six<style>p{}</style><!-- c --><p><![CDATA[seven]]></p></body></html>")
    assert title == "A title"
    assert text == "A title\none\ntwo&three\nfour five\nsix\nseven"


def test_html_shell_below_readable_text_threshold_is_blocked(fake, tmp_path):
    url = "https://www.law.go.kr/법령/전파법"
    fake.add_html(url, "<html><head><title>전파법</title></head><body>" + "<p>짧은 본문</p>" * 3 + "</body></html>")
    out = make_archive(tmp_path, fake, online_policy()).fetch(url, purpose="statute", expect="html")
    assert out.status == "blocked" and "characters of readable text" in out.reason and out.document is None
    assert len(out.text_head) < MIN_HTML_TEXT_CHARS and "짧은 본문" in out.text_head
    assert fake.requests[0].path.startswith("/%EB%B2%95%EB%A0%B9/")  # the Korean path went out percent-encoded
