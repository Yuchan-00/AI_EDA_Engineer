"""Regression tests for the verifier findings on the parts and regulatory tracks (one test per finding, numbered as in the report).

Nothing here touches the network: the web is ``tests/fake_sources.py``, the
KiCad libraries are the synthetic ones of ``tests/test_parts_existence.py``,
the model is ``ScriptedLLMClient``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, ComponentAgent, RegulatoryAgent
from ai_eda.agents.component import CONFIRM_FACTS_KEY, CONFIRM_PARTS_KEY, EXTRACT_FACTS_KEY, FIT_CHECK, FIT_CRITERIA
from ai_eda.compilers import BOMCompiler, CompileContext
from ai_eda.errors import CompileError
from ai_eda.ir import (
    CircuitIR,
    Component,
    Jurisdiction,
    ProjectMeta,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    RequirementSet,
    SourceRef,
    ValidationResult,
    ValidationStatus as S,
    assumption,
    authoritative,
    llm_generated,
    user_requirement,
)
from ai_eda.ir.regulatory import Applicability, GroundedQuote, RegulatoryProvenance, RegulatoryRequirement, RegulatoryState
from ai_eda.parts import CatalogError, CatalogSource, DatasheetFact, ground_facts
from ai_eda.parts.catalog import map_headers, unsafe_cell
from ai_eda.parts.existence import check_component_existence, examine_component
from ai_eda.parts.identity import locate_source, mpn_grounding, verify_source
from ai_eda.regulatory import APPLICABILITY_CHECK, evaluate, load_candidates, research
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.security import ApprovalGate
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.sexpr import Q, S as SX
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy, sha256_of, sniff_kind
from ai_eda.tools.sources.archive import find_block_marker
from ai_eda.tools.sources.extract import extract_html, pdf_header_offset
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow import Orchestrator
from tests.fake_sources import FakeSources
from tests.pdf_fixture import DATASHEET_PAGES, build_pdf, datasheet_pdf
from tests.test_component_agent import ANSWERS, CANDIDATE, USAGE, _ir, _run, _service
from tests.test_parts_existence import VENDOR_HOST, VR1_MPN, VR1_PAGES, VR1_URL, _symbol, make_archive, make_part, online_policy, synthetic_library

LM_MPN = "LM2931AZ-5.0/NOPB"
LVD_ID, EMC_ID, ROHS_ID, RED_ID = "reg.EU.LVD.2014-35-EU", "reg.EU.EMC.2014-30-EU", "reg.EU.RoHS.2011-65-EU", "reg.EU.RED.2014-53-EU"
EU_ANSWERS = {"mains_powered": "no", "radio": "no", "finished_apparatus": "yes", "evaluation_kit": "no", "highest_rated_voltage": "12 V DC", "intended_use": "bench tool"}


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


def offline_archive(root: Path) -> DocumentArchive:
    return DocumentArchive(root, NetworkPolicy(approved=False, gate=ApprovalGate()))


def _req(key: str, value, text: str, kind: RequirementKind = RequirementKind.EXPLICIT) -> Requirement:
    traced = value if not isinstance(value, (int, float)) else user_requirement(float(value), "V")
    return Requirement(id=f"req.{key}", key=key, text=text, kind=kind, value=traced)


def _fact(key: str, value, unit: str | None, page: int, quote: str) -> DatasheetFact:
    return DatasheetFact(key=key, value=value, unit=unit, page=page, quote=quote)


# --------------------------------------------------------------------------- 1 / 13: model facts


def test_01_13_a_models_key_unit_row_and_range_fragment_are_rejected_and_nothing_enters_before_confirmation(tmp_path: Path, fake):
    """The verifier's four probes on the LM2931 fixture: 5 V as i_max, 125 degC out of '-40 to 125 degC', another row's SOIC-8, 0.6 V as v_max."""
    p = tmp_path / "lm2931.pdf"
    p.write_bytes(datasheet_pdf())
    doc = offline_archive(tmp_path / "sources").add_file(p, title="LM2931-N", retrieved_at="2026-09-20")
    g = ground_facts(doc, [
        _fact("i_max", 5, "V", 1, "5 V"),
        _fact("operating_temperature", 125, "degC", 2, "125 degC"),
        _fact("package", "SOIC-8", None, 3, "LM2931AM-5.0/NOPB  SOIC-8"),
        _fact("v_max", 0.6, "V", 1, "0.6 V"),
    ], proposer="model x", mpn=LM_MPN)
    reasons = dict(g.rejected)
    assert reasons["i_max"] == "unit mismatch: key 'i_max' expects A, got 'V' (5 V)"
    assert "part of a larger quantity on the page (-40..125 degC)" in reasons["operating_temperature"]
    assert reasons["package"].startswith("package is not tied to the part: the quote 'LM2931AM-5.0/NOPB  SOIC-8' does not contain the MPN 'LM2931AZ-5.0/NOPB'")
    assert g.accepted_keys == ["v_max"]  # grounding can not tell a dropout voltage from a maximum rating: that is what the user confirms
    # the whole range and the right row are accepted
    ok = ground_facts(doc, [DatasheetFact(key="operating_temperature", value=-40, value_high=125, unit="degC", page=2, quote="-40 to 125 degC"),
                            _fact("package", "TO-92", None, 3, f"Orderable device: {LM_MPN}  TO-92")], proposer="model x", mpn=LM_MPN)
    assert ok.rejected == [] and ok.accepted[0].traced.value == [-40.0, 125.0] and ok.accepted[1].traced.value == "TO-92"
    # through the agent, a model's accepted fact is shown and stays out of the IR until confirm_facts in a later run
    fake.add_pdf(VR1_URL, VR1_PAGES)
    svc, client = _service([{"structured": {"facts": [{"key": "v_max", "value": 200, "unit": "V", "page": 2, "quote": "Maximum operating voltage 200 V"}], "not_found": []},
                             "usage": USAGE}])
    ir = _ir(tmp_path, make_part())
    state, outcome, _ = _run(ir, tmp_path, fake, llm=svc, answers={EXTRACT_FACTS_KEY: "yes"})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_FACTS_KEY]
    assert ir.component("R1").electrical == {} and ir.validation.latest("component.facts.R1.llm").status is S.USER_INPUT_REQUIRED
    state2, _, _ = _run(ir, tmp_path, fake, llm=svc, answers={CONFIRM_FACTS_KEY: "yes"})
    assert not state2.blocked and ir.component("R1").electrical["v_max"].provenance.kind is ProvenanceKind.AUTHORITATIVE
    assert ir.component("R1").electrical["v_max"].provenance.note.endswith(f"; confirmed by user ({CONFIRM_FACTS_KEY})") and len(client.calls) == 1


# --------------------------------------------------------------------------- 2: requirement provenance


def test_02_a_model_assumption_never_decides_the_lvd(tmp_path: Path):
    cl = load_candidates()
    lvd = cl.get(LVD_ID).applicability_rule
    assumed = Requirement(id="req.input_voltage", key="input_voltage", text="input voltage (assumed)", kind=RequirementKind.ASSUMPTION,
                          value=assumption(12.0, unit="V", note="assumed by model: typical wall adapter output, 12 V DC"))
    reqs = RequirementSet(requirements=[assumed])
    ev = evaluate(lvd, {**EU_ANSWERS, "mains_powered": "yes"}, reqs)
    assert ev.applicability is Applicability.UNDECIDED and ev.status is S.USER_INPUT_REQUIRED and "input_voltage" in ev.missing_keys
    assert any("model assumption" in m.reason and "confirm_requirements" in m.reason for m in ev.missing)
    ev2 = evaluate(lvd, EU_ANSWERS, RequirementSet(requirements=[_req("input_voltage", llm_generated(12.0, "m", "V"), "input_voltage: 12V")]))
    assert ev2.applicability is Applicability.UNDECIDED and "not confirmed" in ev2.missing[0].reason
    # the stage and the reviewer see it as undecided, never as a decided NOT_APPLICABLE
    state = RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU")])
    out = research(state, cl, None, EU_ANSWERS, reqs, jurisdictions=["EU"])
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is S.NOT_VERIFIED and "input_voltage" in app.details["missing_keys"] and {r.id: r for r in out.state.requirements}[LVD_ID].applicability is Applicability.UNDECIDED
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ir.regulatory = out.state
    rp = {r.check_id: r for r in IndependentReviewer().review(ir, tmp_path).results}[ReviewArea.REGULATORY_PROVENANCE]
    assert rp.status is S.NOT_VERIFIED and LVD_ID in rp.details["undecided"]


# --------------------------------------------------------------------------- 3: AC/DC words


def test_03_ac_dc_comes_from_the_users_words_next_to_the_unit_never_from_the_note_and_a_contradiction_is_open():
    from ai_eda.regulatory.applicability import _current_kind

    volt = load_candidates().get(LVD_ID).applicability_rule.rules[2]
    assert volt.kind == "voltage_range"
    said = user_requirement(230.0, "V", note="quote: '230V'; parsed: 230 V; model: fake-model; model statement: 'The input is 230 V DC'")
    ev = evaluate(volt, {"mains_powered": "yes", "highest_rated_voltage": "230 V AC"}, RequirementSet(requirements=[_req("input_voltage", said, "input_voltage: 230V")]))
    assert ev.applicability is Applicability.APPLICABLE and ev.inputs_used["input_voltage"] == "230 V AC (requirement req.input_voltage; AC because mains_powered=yes)"
    assert _current_kind("12 V from an AC adapter") is None and _current_kind("12 V DC from an AC adapter") == "dc" and _current_kind("230VAC") == "ac"
    assert _current_kind("The input is 230 V DC") == "dc" and _current_kind("vacuum SDC") is None
    contradiction = evaluate(volt, {"mains_powered": "yes"}, RequirementSet(requirements=[_req("input_voltage", 12.0, "12 V DC input")]))
    assert contradiction.applicability is Applicability.UNDECIDED and contradiction.missing_keys == ["mains_powered"] and "contradict" in contradiction.missing[0].reason


# --------------------------------------------------------------------------- 4: yes/no answers


def test_04_an_unrecognised_answer_to_a_yes_no_question_stays_undecided_and_is_asked_again(tmp_path: Path):
    cl = load_candidates()
    answers = {**EU_ANSWERS, "radio": "wifi and bluetooth", "mains_powered": "yes", "highest_rated_voltage": "230 V AC"}
    reqs = RequirementSet(requirements=[_req("input_voltage", 230.0, "230 V AC input")])
    assert evaluate(cl.get(RED_ID).applicability_rule, answers).applicability is Applicability.UNDECIDED
    assert evaluate(cl.get(LVD_ID).applicability_rule, answers, reqs).applicability is Applicability.UNDECIDED  # not(radio) can not decide either
    out = research(RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU")]), cl, None, answers, reqs, jurisdictions=["EU"])
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is S.NOT_VERIFIED and app.details["missing_keys"] == ["radio"]
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ir.regulatory.jurisdictions = [Jurisdiction(code="EU", name="EU")]
    ir.requirements = reqs
    result = RegulatoryAgent().run(ir, AgentContext(workdir=tmp_path, answers=answers))
    assert [q.key for q in result.questions] == ["radio"] and "was not understood as yes / no" in result.questions[0].question


# --------------------------------------------------------------------------- 5: HTML shells


def test_05_an_html_product_page_never_grounds_an_mpn_and_a_noscript_marker_blocks(fake, tmp_path: Path):
    page_url = f"https://{VENDOR_HOST}/en/product/nc3faah"
    nav = "".join(f"<li>Products Solutions Support Company Careers News Contact {i}</li>" for i in range(6))
    shell = (f"<!DOCTYPE html><html><head><title>{VR1_MPN} - Example Vendor product page</title></head><body>"
             f"<noscript>Please enable JavaScript to view this page. Access denied otherwise.</noscript><nav><ul>{nav}</ul></nav><div id='app'></div>"
             "<footer>Copyright Example Vendor. All rights reserved. Terms of use. Privacy. Imprint. Cookie settings.</footer></body></html>")
    fake.add_html(page_url, shell)
    lib = synthetic_library(tmp_path / "kicad", datasheet_url=page_url)
    archive = make_archive(tmp_path, fake, online_policy())
    res = check_component_existence(make_part(), lib, archive)
    checks = {c["name"]: c for c in res.details["checks"]}
    assert checks["datasheet_archived"]["status"] == "NOT_VERIFIED" and checks["datasheet_archived"]["details"]["fetch_status"] == "blocked"
    assert "noscript text" in checks["datasheet_archived"]["message"] and res.artifact_hash is None
    ex = extract_html(shell.encode("utf-8"))
    assert ex.hidden and "enable JavaScript" in ex.hidden and "enable javascript" not in ex.pages[0].lower()
    assert find_block_marker(ex.title, ex.pages[0], ex.hidden) == "access denied" and find_block_marker(ex.title, ex.pages[0]) is None
    # a real product page (no marker, plenty of text, the MPN in title and body) is archived as evidence of the pointer but is not a datasheet
    product = (f"<!DOCTYPE html><html><head><title>{VR1_MPN} - Example Vendor</title></head><body><h1>{VR1_MPN}</h1><p>Thin film chip resistor, 200 V, 0603. "
               + "Order online, check stock, compare with similar parts, download the datasheet PDF from the documents tab. " * 6 + "</p></body></html>")
    fake.add_html(page_url, product)
    archive2 = make_archive(tmp_path / "b", fake, online_policy())
    report = examine_component(make_part(), lib, archive2)
    check = report.check("datasheet_archived")
    assert check.status is S.NOT_VERIFIED and check.details["kind"] == "html" and "archived html document is not a datasheet" in check.message
    assert report.document is None and report.mpn_hit is None and report.check("mpn_in_datasheet").message == "datasheet not archived: MPN not checked"
    assert archive2.lookup(page_url) is not None  # archived as evidence of what the pointer led to
    ir = _ir(tmp_path / "c", make_part())
    ctx = AgentContext(workdir=tmp_path / "c", tools={"kicad_library": synthetic_library(tmp_path / "c" / "kicad", datasheet_url=page_url),
                                                     "archive": make_archive(tmp_path / "c", fake, online_policy())}, answers=dict(ANSWERS))
    result = ComponentAgent().run(ir, ctx)
    Orchestrator.apply_proposals(ir, result.proposals)
    assert ir.component("R1").mpn.provenance.kind is ProvenanceKind.ASSUMPTION  # never re-tagged on a web page
    # an IR that was tagged from such a page earlier is downgraded by the grounding rule
    doc = archive2.lookup(page_url)
    tagged = make_part()
    tagged.mpn = authoritative(VR1_MPN, doc.source_ref(section="page 1"))
    g = mpn_grounding(tagged, archive2)
    assert not g.grounded and g.label == "authoritative, not a datasheet" and "html" in g.reason


# --------------------------------------------------------------------------- 6: candidate confirmation is row by row


def test_06_only_candidate_rows_shown_in_an_earlier_table_can_be_confirmed(fake, tmp_path: Path):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    u1 = {**CANDIDATE, "ref": "U1", "mpn": "VR1-0603-999V-Z", "kicad_symbol": "Test2:Later", "kicad_footprint": "Test:FP", "datasheet_url": None}
    svc, client = _service([{"structured": {"candidates": [CANDIDATE, u1]}, "usage": USAGE}])
    ir = _ir(tmp_path, make_part(mpn=None), make_part(ref="U1", mpn=None, symbol=None, footprint=None))
    # run 1: the library lacks Test2:Later - R1 accepted and shown, U1 rejected (its MPN never appears in the table)
    state, outcome, _ = _run(ir, tmp_path, fake, llm=svc)
    assert state.blocked
    [q] = state.open_questions
    assert VR1_MPN in q.question and "VR1-0603-999V-Z" not in q.question and "U1: symbol Test2:Later does not exist" in q.question
    assert ir.component("U1").mpn is None
    # the library is fixed between runs
    later = SX("kicad_symbol_lib", SX("version", 20251024), SX("generator", Q("kicad_symbol_editor")), SX("generator_version", Q("10.0")), _symbol("Later", VR1_URL))
    (tmp_path / "kicad" / "symbols" / "Test2.kicad_sym").write_text(sexpr.dumps(later), encoding="utf-8")
    # run 2: the confirmation applies to R1 only; U1 became acceptable but was never shown - it is shown now, marked new, and waits
    state2, outcome2, _ = _run(ir, tmp_path, fake, llm=svc, answers={CONFIRM_PARTS_KEY: "yes"})
    assert state2.blocked and len(client.calls) == 1
    [q2] = state2.open_questions
    assert q2.key == CONFIRM_PARTS_KEY and "U1 (new)" in q2.question and "VR1-0603-999V-Z" in q2.question and "only rows already shown can be confirmed" in q2.question
    r1, u1c = ir.component("R1"), ir.component("U1")
    assert r1.provenance.kind is ProvenanceKind.USER_REQUIREMENT and r1.mpn.provenance.kind is ProvenanceKind.AUTHORITATIVE  # confirmed, then grounded
    assert u1c.provenance.kind is ProvenanceKind.DERIVED and u1c.mpn.provenance.kind is ProvenanceKind.LLM_GENERATED and u1c.symbol.name == "Later"
    cand = ir.validation.latest("component.candidates")
    assert cand.status is S.USER_INPUT_REQUIRED and cand.details["confirmed_refs"] == ["R1"] and cand.details["pending_refs"] == ["U1"] and cand.details["fresh_refs"] == ["U1"]
    assert "awaiting the user's confirmation" in ir.validation.latest("component.existence.U1").message  # not checked, nothing fetched for it
    # run 3: the user has seen U1's row and confirms it
    state3, outcome3, _ = _run(ir, tmp_path, fake, llm=svc, answers={CONFIRM_PARTS_KEY: "yes"})
    assert not state3.blocked and ir.component("U1").provenance.kind is ProvenanceKind.USER_REQUIREMENT and len(client.calls) == 1
    cand3 = ir.validation.latest("component.candidates")
    assert cand3.status is S.PASS and cand3.details["confirmed_refs"] == ["U1"]  # R1 is grounded and no longer a candidate; the earlier reply was reused for U1
    assert any("reusing the candidate reply" in n for n in outcome3.message.split("; "))


# --------------------------------------------------------------------------- 7: PDF header offset


def test_07_a_pdf_with_leading_bytes_before_the_header_is_a_pdf_and_a_non_pdf_says_so(fake, tmp_path: Path):
    body = datasheet_pdf()
    for prefix in (b"\r\n", b"\xef\xbb\xbf", b"  \n"):
        assert sniff_kind(prefix + body, "application/pdf", "x.pdf") == "pdf" and pdf_header_offset(prefix + body) == len(prefix)
    assert sniff_kind(b"<html>%PDF-1.4 in prose</html>", "text/html") == "html" and pdf_header_offset(b"<html>%PDF-") is None
    assert sniff_kind(b"MZ" + bytes(1000), "application/pdf", "x.pdf") == "binary" and sniff_kind(bytes(2000) + b"%PDF-1.4", None, "x.pdf") == "binary"
    url = f"https://{VENDOR_HOST}/ds/vr1.pdf"
    fake.serve(url, b"\r\n" + build_pdf(VR1_PAGES), "application/pdf")
    archive = make_archive(tmp_path, fake, online_policy())
    archive.policy.trust_host(VENDOR_HOST, "test")
    out = archive.fetch(url, purpose="ds", expect="pdf")
    assert out.ok and out.document.kind == "pdf" and out.document.find_quote(VR1_MPN)[0].page == 2 and out.document.sha256 == sha256_of(b"\r\n" + build_pdf(VR1_PAGES))
    fake.serve(f"https://{VENDOR_HOST}/ds/notpdf.pdf", b"MZ" + bytes(2000), "application/octet-stream")
    bad = archive.fetch(f"https://{VENDOR_HOST}/ds/notpdf.pdf", purpose="ds", expect="pdf")
    assert bad.status == "blocked" and "not a PDF by its bytes" in bad.reason and "interstitial" not in bad.reason


# --------------------------------------------------------------------------- 8: extraction version


def test_08_the_extractor_stamp_travels_with_the_claim_and_a_different_re_extraction_is_not_verified(fake, tmp_path: Path):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    ir = _ir(tmp_path, make_part())
    _run(ir, tmp_path, fake)
    r1 = ir.component("R1")
    doc = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate())).load(r1.datasheet.content_hash)
    assert doc.extraction_stamp in r1.mpn.provenance.note and doc.extraction_stamp.startswith("extractor pypdf 1 (lib ")
    facts = ir.validation.latest("component.existence.R1").details["datasheet"]["archived"]
    assert facts["extractor"] == "pypdf" and facts["extractor_version"] == "1"
    g = mpn_grounding(r1, DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate())))
    assert g.grounded and g.extraction == doc.extraction_stamp and g.page == 2
    # the meta records another text hash (the extractor that grounded the claim differs from the current one): not verified until re-grounded
    meta_path = doc.path.with_name(doc.path.name.split(".")[0] + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["text_sha256"] = "sha256:" + "0" * 64
    meta["extractor_version"] = "0"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    g2 = mpn_grounding(r1, DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate())))
    assert not g2.grounded and g2.label == "authoritative, re-extracted" and "differs from the text recorded at archive time" in g2.reason
    v = default_registry.get("ir.component_provenance").validate(ir, ValidationContext(workdir=tmp_path, tools={}))[0]
    assert v.status is S.NOT_VERIFIED and "R1.mpn[authoritative, re-extracted]" in v.details["unverified"]
    cp = {r.check_id: r for r in IndependentReviewer().review(ir, tmp_path).results}[ReviewArea.COMPONENT_PROVENANCE]
    assert cp.status is S.NOT_VERIFIED and "R1.mpn[authoritative, re-extracted]" in cp.details["weak"] and "differs from the text recorded" in cp.message


# --------------------------------------------------------------------------- 9: the reviewer's own evidence


def test_09_the_reviewer_re_locates_the_mpn_and_the_quotes_and_accepts_no_file_outside_an_archive(fake, tmp_path: Path):
    # (a) an MPN tagged from a hand-written text file plus a hand-written PASS existence result
    note = tmp_path / "notes.txt"
    note.write_text(f"Part number: {VR1_MPN}\n" * 4, encoding="utf-8")
    ref = SourceRef(title="my notes", document_path=str(note), content_hash=sha256_of(note.read_bytes()), retrieved_at="2026-09-20T00:00:00+00:00", section="page 1")
    part = make_part()
    part.mpn = authoritative(VR1_MPN, ref)
    ir = _ir(tmp_path, part)
    ir.validation.extend([ValidationResult(check_id="component.existence.R1", status=S.PASS, tool="parts.existence", artifact_hash=ref.content_hash,
                                           message="hand written", details={"mpn": VR1_MPN, "checks": [{"name": "mpn_in_datasheet", "status": "PASS"}]})])
    assert verify_source(ref, None)[0] == "unarchived" and "not an entry of a document archive" in verify_source(ref, None)[1]
    cp = {r.check_id: r for r in IndependentReviewer().review(ir, tmp_path).results}[ReviewArea.COMPONENT_PROVENANCE]
    assert cp.status is S.NOT_VERIFIED and "R1.mpn[authoritative, unarchived]" in cp.details["weak"]
    # (b) an archived PDF that does not contain the MPN, tagged all the same
    fake.add_pdf(VR1_URL, [["VR1 Series", "no part numbers on this page"], ["nothing here either"]])
    archive = make_archive(tmp_path, fake, online_policy())
    archive.policy.trust_host(VENDOR_HOST, "test")
    doc = archive.fetch(VR1_URL, purpose="ds", expect="pdf").document
    part2 = make_part()
    part2.mpn = authoritative(VR1_MPN, doc.source_ref(section="page 2"))
    ir2 = _ir(tmp_path, part2)
    ir2.validation.extend([ValidationResult(check_id="component.existence.R1", status=S.PASS, tool="parts.existence", artifact_hash=doc.sha256,
                                            message="hand written", details={"mpn": VR1_MPN, "checks": [{"name": "mpn_in_datasheet", "status": "PASS"}]})])
    g = mpn_grounding(part2, archive)
    assert not g.grounded and g.label == "authoritative, not in text" and "not found verbatim on page 2" in g.reason
    cp2 = {r.check_id: r for r in IndependentReviewer(tools={"archive": archive}).review(ir2, tmp_path).results}[ReviewArea.COMPONENT_PROVENANCE]
    assert cp2.status is S.NOT_VERIFIED and "R1.mpn[authoritative, not in text]" in cp2.details["weak"] and "not found verbatim on page 2" in cp2.message
    # (c) a regulatory requirement whose quote the stage says it found, but the archived text does not bear
    lvd_url = "https://eur-lex.europa.eu/lvd"
    fake.add_html(lvd_url, "<html><head><title>L_TEST</title></head><body><p>DIRECTIVE 2014/35/EU (SYNTHETIC)</p><p>" + "This page says nothing about voltage ratings. " * 12 + "</p></body></html>")
    archive.policy.trust_host("eur-lex.europa.eu", "test")
    official = archive.fetch(lvd_url, purpose="lvd", expect="html").document
    req = RegulatoryRequirement(
        id=LVD_ID, jurisdiction="EU", title="LVD", candidate_id=LVD_ID, applicability=Applicability.NOT_APPLICABLE, status=S.NOT_APPLICABLE,
        provenance=RegulatoryProvenance(jurisdiction="EU", authority="EU", source_title="LVD", source_url=lvd_url, retrieved_at=official.retrieved_at, section="Article 1 (page 1)",
                                        applicability_rationale="hand written", verification_status=S.PASS, source_document=str(official.path), content_hash=official.sha256),
        grounded_quotes=[GroundedQuote(section="Article 1", quote="between 50 and 1 000 V for alternating current", found=True, page=1, source_url=lvd_url, content_hash=official.sha256)],
    )
    ir3 = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ir3.regulatory = RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU")], requirements=[req])
    rp = {r.check_id: r for r in IndependentReviewer(tools={"archive": archive}).review(ir3, tmp_path).results}[ReviewArea.REGULATORY_PROVENANCE]
    assert rp.status is S.NOT_VERIFIED and rp.details["quotes_not_relocated"] == {LVD_ID: ["Article 1 (page 1)"]} and "not found in the archived text at review time" in rp.message
    assert locate_source(SourceRef(title="x", document_path=str(official.path), content_hash=official.sha256, retrieved_at=official.retrieved_at), None)[0] == "ok"  # an archive entry, found by its meta


# --------------------------------------------------------------------------- 10: catalog cells


def test_10_formula_cells_never_reach_the_bom_and_mfr_alone_is_not_the_mpn(tmp_path: Path):
    csv = tmp_path / "dump.csv"
    csv.write_text(
        "LCSC Part #,MFR.Part #,Manufacturer,Package,Stock,Type,Price (USD)\n"
        f'"=HYPERLINK(""http://x"",""C25792"")",{VR1_MPN},Example Vendor,0603,10,Basic,0.01\n'
        "C2,-2+3|cmd,Example Vendor,0603,10,Basic,0.01\n"
        "C3,GOOD-1,Example Vendor,0603,10,Basic,0.01\n"
        "C4;rm,GOOD-2,Example Vendor,0603,10,Basic,0.01\n",
        encoding="utf-8",
    )
    cat = CatalogSource.load(csv, "2026-09-23", "community dump")
    assert [r.mpn for r in cat.rows] == ["GOOD-1"]
    assert any("line 2: skipped" in n and "starts with '='" in n for n in cat.notes) and any("line 3: skipped" in n and "starts with '-'" in n for n in cat.notes)
    assert any("line 5: skipped" in n and "not a part-number-like token" in n for n in cat.notes)
    assert unsafe_cell("=1+1") and unsafe_cell("@cmd") and unsafe_cell("a\tb") and unsafe_cell("0603") is None and unsafe_cell("") is None
    # the BOM compiler refuses an authoritative cell that would execute, however it got into the IR
    part = make_part()
    part.mpn = authoritative("=HYPERLINK(\"http://x\")", SourceRef(title="edited"))
    ir = _ir(tmp_path, part)
    with pytest.raises(CompileError, match="refusing to write a cell"):
        BOMCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))
    # 'mfr' is the MPN only next to a separate manufacturer column (the cdfer dump); a bare 'MFR' is not guessed
    m, notes = map_headers(["MFR", "Part Number", "Manufacturer", "Package"])
    assert m["mpn"] == 1 and m["manufacturer"] == 2 and any("'MFR' also means mpn" in n for n in notes)
    m2, notes2 = map_headers(["mfr", "package", "stock"])
    assert "mpn" not in m2 and any("ambiguous" in n for n in notes2)
    (tmp_path / "mfr_only.csv").write_text("mfr,package,stock\nAcme,0603,1\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="lacks the required column"):
        CatalogSource.load(tmp_path / "mfr_only.csv", "2026-09-23", "x")


# --------------------------------------------------------------------------- 11: add_file after a fetch


def test_11_a_user_file_with_the_same_bytes_keeps_the_network_provenance(fake, tmp_path: Path):
    data = fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    archive.policy.trust_host(VENDOR_HOST, "test")
    fetched = archive.fetch(VR1_URL, purpose="ds", expect="pdf").document
    local = tmp_path / "vr1 (downloaded).pdf"
    local.write_bytes(data)
    again = archive.add_file(local, title="VR1 datasheet (my copy)", retrieved_at="2026-09-01", authority="Example Vendor")
    assert again.sha256 == fetched.sha256 and again.meta["source"] == "network" and again.url == VR1_URL and again.final_url == VR1_URL
    assert again.meta["retrieved_at"] == fetched.meta["retrieved_at"] and again.meta["http_status"] == 200 and again.meta["authority"] == "Example Vendor"
    assert again.meta["history"][-1]["source"] == "user_file" and again.meta["history"][-1]["original_path"] == str(local.resolve()) and "network provenance kept" in again.meta["history"][-1]["note"]
    reloaded = offline_archive(archive.root).lookup(VR1_URL)
    assert reloaded is not None and reloaded.sha256 == fetched.sha256 and reloaded.source_ref().url == VR1_URL
    # a user file first, then the network: the fetch's provenance takes over and the user file stays in the history
    other = make_archive(tmp_path / "o", fake, online_policy())
    other.policy.trust_host(VENDOR_HOST, "test")
    first = other.add_file(local, title="mine", retrieved_at="2026-09-01")
    assert first.meta["source"] == "user_file"
    second = other.fetch(VR1_URL, purpose="ds", expect="pdf").document
    assert second.meta["source"] == "network" and second.meta["history"][-1]["source"] == "user_file"


# --------------------------------------------------------------------------- 12: the LVD rates the equipment


def test_12_the_input_rail_alone_never_puts_a_design_outside_the_lvd(tmp_path: Path):
    cl = load_candidates()
    lvd = cl.get(LVD_ID).applicability_rule
    input_only = RequirementSet(requirements=[_req("input_voltage", 12.0, "12 V DC input")])
    answers = {k: v for k, v in EU_ANSWERS.items() if k != "highest_rated_voltage"}
    ev = evaluate(lvd, answers, input_only)
    assert ev.applicability is Applicability.UNDECIDED and ev.missing_keys == ["highest_rated_voltage"] and "highest voltage anywhere in the product" in ev.missing[0].reason
    boost = RequirementSet(requirements=[_req("input_voltage", 12.0, "12 V DC input"), _req("output_voltage", 400.0, "400 V DC output for the nixie tubes")])
    ev = evaluate(lvd, answers, boost)
    assert ev.applicability is Applicability.APPLICABLE and "output_voltage = 400 V DC" in ev.rationale and ev.evidence == ["Annex II (evaluation kits)", "Article 1"]
    ev = evaluate(lvd, {**answers, "highest_rated_voltage": "400 V DC"}, input_only)
    assert ev.applicability is Applicability.APPLICABLE
    ev = evaluate(lvd, EU_ANSWERS, input_only)
    assert ev.applicability is Applicability.NOT_APPLICABLE and "input voltage only" not in ev.rationale
    assert "other internal voltages are not read from the IR" in ev.rationale
    out = research(RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU")]), cl, None, answers, input_only, jurisdictions=["EU"])
    app = out.result(APPLICABILITY_CHECK)
    assert app.status is S.NOT_VERIFIED and "answer highest_rated_voltage" in app.message and out.undecided[LVD_ID] == ["highest_rated_voltage"]
    kr = cl.get("reg.KR.ElectricalAppliancesSafetyAct.EnforcementRule").applicability_rule
    assert kr.scope_answer == "highest_rated_voltage" and evaluate(kr, EU_ANSWERS, input_only).applicability is Applicability.APPLICABLE


# --------------------------------------------------------------------------- 14: what the README claims


def test_14_the_readme_states_what_was_not_measured():
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert "저장소 밖" in readme and "law.go.kr" in readme and "eCFR" in readme and "측정되지 않" in readme
    assert "합성" in readme  # the offline tests use synthetic pages built from the list's own quotes


# --------------------------------------------------------------------------- 15: exclusions


def test_15_the_evaluation_kit_exclusion_is_evaluated_and_what_is_not_evaluated_is_said(tmp_path: Path):
    cl = load_candidates()
    kit = {**EU_ANSWERS, "mains_powered": "yes", "evaluation_kit": "yes", "highest_rated_voltage": "230 V AC"}
    reqs = RequirementSet(requirements=[_req("input_voltage", 230.0, "230 V AC input")])
    lvd = evaluate(cl.get(LVD_ID).applicability_rule, kit, reqs)
    assert lvd.applicability is Applicability.NOT_APPLICABLE and lvd.evidence == ["Annex II (evaluation kits)"]
    assert evaluate(cl.get(EMC_ID).applicability_rule, kit, reqs).applicability is Applicability.NOT_APPLICABLE
    product = {**kit, "evaluation_kit": "no"}
    assert evaluate(cl.get(LVD_ID).applicability_rule, product, reqs).applicability is Applicability.APPLICABLE
    assert "no rule reads this free text" in next(q for q in cl.scope_questions if q.key == "intended_use").rationale
    out = research(RegulatoryState(jurisdictions=[Jurisdiction(code="EU", name="EU")]), cl, None, product, reqs, jurisdictions=["EU"])
    by = {r.id: r for r in out.state.requirements}
    assert "not evaluated: the 'subject to paragraph 2'" in by[ROHS_ID].provenance.applicability_rationale and "Article 2(4)" in by[ROHS_ID].provenance.applicability_rationale
    assert "not evaluated: the Annex II exclusions other than custom built evaluation kits" in by[LVD_ID].provenance.applicability_rationale
    app = out.result(APPLICABILITY_CHECK)
    assert "not evaluated by the rules" in app.message and set(app.details["not_evaluated"]) == {LVD_ID, EMC_ID, ROHS_ID}


# --------------------------------------------------------------------------- 16: part fit


def test_16_the_component_stage_claims_nothing_about_fit_and_the_validator_names_what_it_evaluates(fake, tmp_path: Path):
    """Part fit is the ``component.fit`` validator's verdict (electrical stress at the SPICE op, tests/test_component_fit.py), not the agent's.

    The COMPONENT_SELECTION stage therefore carries no ``component.fit`` result and is PASS when its tool-backed existence
    checks all PASS; the validator, evaluated on the same IR without an op, is NOT_VERIFIED and lists the one criterion it
    evaluates and the seven it does not.
    """
    from ai_eda.validation import ValidationContext, default_registry

    fake.add_pdf(VR1_URL, VR1_PAGES)
    ir = _ir(tmp_path, make_part())
    state, outcome, _ = _run(ir, tmp_path, fake)
    assert ir.validation.latest(FIT_CHECK) is None
    assert ir.validation.latest("component.existence.R1").status is S.PASS and outcome.status is S.PASS  # existence proven by the tool
    assert set(FIT_CRITERIA) == {"electrical stress", "safety", "regulatory", "environment", "reliability", "manufacturability", "sourcing", "cost"}
    [fit] = default_registry.get(FIT_CHECK).validate(ir, ValidationContext(workdir=tmp_path))
    assert fit.status is S.NOT_VERIFIED and fit.tool == FIT_CHECK and "needs an operating point from SPICE" in fit.message
    assert fit.details["criteria_evaluated"] == ["electrical stress"] and fit.details["criteria_not_evaluated"] == list(FIT_CRITERIA[1:])


# --------------------------------------------------------------------------- 17: authority and catalog comparison


def test_17_a_models_manufacturer_is_not_the_documents_authority_and_a_catalog_row_of_another_brand_backs_nothing(fake, tmp_path: Path):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    guessed = make_part()
    guessed.manufacturer = llm_generated("Fakeco", "some/model", note="candidate proposed by some/model")
    ir = _ir(tmp_path, guessed)
    _run(ir, tmp_path, fake)
    r1 = ir.component("R1")
    assert r1.datasheet.authority == "example-vendor.com" and r1.mpn.provenance.source.authority == "example-vendor.com"  # the pointer's host, not the model's string
    trusted = make_part()
    trusted.manufacturer = user_requirement("Example Vendor")
    ir2 = _ir(tmp_path / "t", trusted)
    _run(ir2, tmp_path / "t", fake)
    assert ir2.component("R1").datasheet.authority == "Example Vendor"
    csv = tmp_path / "cat.csv"
    csv.write_text(f"mpn,manufacturer,package,supplier_part_number,stock\n{VR1_MPN},Other Brand,0805,C1,5\n", encoding="utf-8")
    cat = CatalogSource.load(csv, "2026-09-23", "JLCPCB export", supplier="JLCPCB")
    ir_part = make_part()
    ir_part.manufacturer = user_requirement("Example Vendor")
    ir_part.package = user_requirement("0603")
    report = examine_component(ir_part, synthetic_library(tmp_path / "kicad"), make_archive(tmp_path / "c", fake, online_policy()), cat)
    check = report.check("catalog")
    assert check.status is S.NOT_VERIFIED and "catalog manufacturer 'Other Brand' differs from IR manufacturer 'Example Vendor'" in check.message
    assert "catalog package '0805' differs from IR package '0603'" in check.message and report.sourcing is None and report.catalog_row is not None
    assert examine_component(make_part(), synthetic_library(tmp_path / "kicad"), make_archive(tmp_path / "d", fake, online_policy()), cat).sourcing is not None  # nothing to compare: the row backs sourcing


# --------------------------------------------------------------------------- 18: the tag is not the grounding


def test_18_the_tag_property_says_what_it_is():
    part = make_part()
    part.mpn = authoritative(VR1_MPN, SourceRef(title="a fixture", content_hash="sha256:" + "0" * 64))
    assert part.mpn_tagged_authoritative and not hasattr(Component, "has_authoritative_identity")
    assert "not** a grounding check" in Component.mpn_tagged_authoritative.__doc__
    assert not mpn_grounding(part).grounded  # the tag alone is nothing
