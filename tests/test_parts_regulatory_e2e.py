"""The parts and regulatory tracks wired into the pipeline, end to end and offline.

The divider fixture (R1 / R2 / J1) now carries datasheet pointers on a
vendor host and the run gives the regulatory scope answers; the web is a
loopback fake (``tests/fake_sources.py``) that serves the vendor's PDFs and
synthetic EUR-Lex pages built from the *packaged* candidate list's own
markers and quotes. What is proven:

* without an online session nothing is fetched (the fake records zero
  requests; the CLI never even builds an HTTP client) and every source stays
  ``NOT_VERIFIED`` - ``component.existence.*`` "not fetched (offline)",
  ``regulatory.sources`` offline, ``regulatory.compliance`` always;
* with one, the datasheets are archived by sha256 and the MPNs found
  verbatim (``component.existence.*`` PASS with evidence, the BOM's
  ``DatasheetHash`` filled), the official texts are archived and every
  claimed quote grounded (``regulatory.sources`` PASS), applicability is
  decided deterministically (LVD not applicable at 12 V DC with the Article 1
  quote as evidence, EMC / RoHS applicable, RED not applicable),
  ``regulatory.compliance`` stays ``NOT_VERIFIED``, both provenance review
  areas PASS, and RELEASE is still not PASS naming ``regulatory.compliance``
  and ``mfg.capability``;
* a datasheet or official text altered after grounding is ``tampered``
  (NOT_VERIFIED, human) for the validator and the reviewer alike;
* a catalog backs sourcing for the parts it lists and leaves the others
  honestly ``not in catalog``;
* the CLI flags build the same session (``--online``, ``--trust-host``,
  ``--catalog`` needs ``--catalog-date``, ``--datasheet-url REF=URL``).
"""

from __future__ import annotations

import csv
import html
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from ai_eda.agents import AgentContext, RequirementAgent
from ai_eda.cli import main as cli_main
from ai_eda.compilers import BOMCompiler, CompileContext
from ai_eda.ir import ArtifactKind, CircuitIR, ProjectMeta, ProvenanceKind, ValidationStatus, authoritative
from ai_eda.ir.regulatory import Applicability
from ai_eda.parts.identity import mpn_grounding, verify_source
from ai_eda.regulatory import APPLICABILITY_CHECK, COMPLIANCE_CHECK, RESEARCH_CHECK, SOURCES_CHECK, CandidateList, load_candidates
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.review.reviewer import REGULATORY_PROVENANCE_FIELDS
from ai_eda.security.approval import ApprovalGate, ExternalAction, default_gate
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.routing import route_naive
from ai_eda.tools.sources import DocumentArchive, sha256_of
from ai_eda.tools.spice import NgspiceShared
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow import Orchestrator, SessionError, Stage, open_session
from tests.fake_sources import FakeSources
from tests.fixtures_kicad import DATASHEETS, HEADER_DS_URL, RESISTOR_DS, RESISTOR_DS_URL, SCOPE_ANSWERS, VENDOR_HOST, divider_with_connector_ir
from tests.test_parts_existence import synthetic_library

S = ValidationStatus
LIB = KicadLibrary()
HAS_LIBS = LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and LIB.symbol_file("Device") is not None
kicad = KicadCli()
ngspice = NgspiceShared()
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
needs_tools = pytest.mark.skipif(not (kicad.available() and HAS_LIBS and ngspice.available()), reason="kicad-cli / KiCad libraries / ngspice.dll not installed")

EUR_LEX = "eur-lex.europa.eu"
LVD_ID, EMC_ID, ROHS_ID, RED_ID = "reg.EU.LVD.2014-35-EU", "reg.EU.EMC.2014-30-EU", "reg.EU.RoHS.2011-65-EU", "reg.EU.RED.2014-53-EU"
#: review areas the real tools prove in a full run (as in tests/test_vertical_slice.py)
PROVEN_AREAS = [
    ReviewArea.IR_VS_SCHEMATIC, ReviewArea.IR_VS_PCB, ReviewArea.SCHEMATIC_VS_PCB, ReviewArea.PCB_VS_BOM, ReviewArea.PCB_VS_CPL,
    ReviewArea.MANUFACTURING_OUTPUTS, ReviewArea.CALCULATIONS_VS_DESIGN, ReviewArea.SPICE_VS_REQUIREMENTS, ReviewArea.ERC, ReviewArea.DRC,
]
FILLER = "This synthetic page stands in for the official site in tests; the markers and quotes above are the packaged candidate list's own. " * 3


# --------------------------------------------------------------------------- the fake web


def official_page(markers: list[str], quotes: list[str]) -> str:
    """A synthetic EUR-Lex-like page: the expected markers as title / headings, every claimed quote as its own paragraph (HTML-escaped verbatim)."""
    esc = lambda s: html.escape(s, quote=False)  # noqa: E731
    heads = "".join(f'<p class="oj-doc-ti">{esc(m)}</p>' for m in markers)
    body = "".join(f'<p class="oj-normal">{esc(q)}</p>' for q in quotes)
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{esc(markers[0])}</title><script>var t = "not text";</script></head>'
            f'<body><div class="eli-container">{heads}{body}<p>{FILLER}</p></div></body></html>')


def serve_official_texts(fake: FakeSources, candidates: CandidateList, codes: tuple[str, ...] = ("EU",)) -> list[str]:
    """Serve every fetchable document of the packaged candidates for ``codes`` with its markers and quotes; returns the URLs."""
    urls: list[str] = []
    for code in codes:
        for c in candidates.for_jurisdiction(code):
            if not c.fetchable:
                continue
            for d in c.documents():
                fake.add_html(d.url, official_page(list(d.expected_markers), [q.quote for q in c.quotes_for(d.url)]))
                urls.append(d.url)
    return urls


def serve_everything(fake: FakeSources) -> None:
    for url, pages in DATASHEETS.items():
        fake.add_pdf(url, pages)
    serve_official_texts(fake, load_candidates())


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


# --------------------------------------------------------------------------- runs


def _routed_ir(tmp_path: Path) -> CircuitIR:
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.tracks = route_naive(ir, LIB)
    return ir


def _session(tmp_path: Path, ir: CircuitIR, fake: FakeSources, *, online: bool, catalog: Path | None = None):
    return open_session(
        workdir=tmp_path, ir=ir, library=LIB, online=online, trust_hosts=[VENDOR_HOST], catalog=catalog,
        catalog_date="2026-09-23" if catalog is not None else None, catalog_authority="JLCPCB export (test)", catalog_supplier="JLCPCB",
        gate=ApprovalGate(), client=fake.client(),
    )


def _context(tmp_path: Path, session) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_cli": kicad, "kicad_library": LIB, "spice": ngspice, **session.tools()}, answers=dict(SCOPE_ANSWERS))


def _run(tmp_path: Path, fake: FakeSources, *, online: bool, ir: CircuitIR | None = None, catalog: Path | None = None, stop_after: Stage | None = None):
    ir = ir if ir is not None else _routed_ir(tmp_path)
    session = _session(tmp_path, ir, fake, online=online, catalog=catalog)
    ctx = _context(tmp_path, session)
    try:
        state = Orchestrator(ctx).run(ir, stop_after=stop_after)
    finally:
        session.close()
    for o in state.outcomes:
        print(f"{o.stage:<24} {o.status:<14} {o.message[:160]}")
    return ir, state, ctx, session


def _bom_rows(ir: CircuitIR, tmp_path: Path, tools: dict[str, Any]) -> dict[str, dict[str, str]]:
    art = ir.artifacts.get(ArtifactKind.BOM) or BOMCompiler().compile(ir, CompileContext(workdir=tmp_path, tools=tools))
    with open(art.path, newline="", encoding="utf-8") as f:
        return {row["Reference"]: row for row in csv.DictReader(f)}


def _blocking(release_message: str) -> list[str]:
    return release_message[release_message.index("(") + 1 : release_message.rindex(")")].split(", ")


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- offline


@needs_tools
def test_offline_run_fetches_nothing_and_is_honestly_not_verified(fake, tmp_path: Path):
    serve_everything(fake)
    ir, state, ctx, session = _run(tmp_path, fake, online=False)
    assert fake.requests == [] and fake.client_urls == []  # no socket use at all
    assert not session.online and not [e for e in session.policy.gate.audit if e["action"] == ExternalAction.NETWORK_FETCH]
    assert not state.blocked and [o.stage for o in state.outcomes] == list(Stage)
    latest = ir.validation.latest_by_check()
    for ref in ("R1", "R2", "J1"):
        ex = latest[f"component.existence.{ref}"]
        assert ex.status is S.NOT_VERIFIED and ex.tool == "parts.existence" and ex.evidence == [] and ex.artifact_hash is None
        checks = {c["name"]: c for c in ex.details["checks"]}
        assert checks["symbol"]["status"] == "PASS" and checks["footprint"]["status"] == "PASS"
        assert checks["datasheet_pointer"]["details"]["origin"] == "ir" and checks["datasheet_pointer"]["details"]["host"] == "example-vendor.com"
        assert checks["datasheet_archived"]["status"] == "NOT_VERIFIED" and "not fetched (offline" in checks["datasheet_archived"]["message"]
        assert checks["mpn_in_datasheet"]["status"] == "NOT_VERIFIED" and checks["catalog"]["status"] == "NOT_APPLICABLE"
        c = ir.component(ref)
        assert c.datasheet.content_hash is None and c.mpn.provenance.kind is ProvenanceKind.AUTHORITATIVE  # tagged, not grounded
    assert latest[SOURCES_CHECK].status is S.NOT_VERIFIED and "not fetched (offline)" in latest[SOURCES_CHECK].message
    assert latest[APPLICABILITY_CHECK].status is S.NOT_VERIFIED and "not grounded" in latest[APPLICABILITY_CHECK].message
    assert latest[COMPLIANCE_CHECK].status is S.NOT_VERIFIED and latest[RESEARCH_CHECK].status is S.NOT_VERIFIED
    by = {r.id: r for r in ir.regulatory.requirements}
    assert by[LVD_ID].applicability is Applicability.NOT_APPLICABLE and by[LVD_ID].source_status == "offline" and by[LVD_ID].provenance.content_hash is None
    assert by[EMC_ID].applicability is Applicability.APPLICABLE and by[ROHS_ID].applicability is Applicability.APPLICABLE and by[RED_ID].applicability is Applicability.NOT_APPLICABLE
    # the IR_BUILD validator downgrades the tagged MPNs; the reviewer agrees; the real-tool areas are proven all the same
    cp = latest["ir.component_provenance"]
    assert cp.status is S.NOT_VERIFIED and set(cp.details["unverified"]) == {"J1.mpn[authoritative, unarchived]", "R1.mpn[authoritative, unarchived]", "R2.mpn[authoritative, unarchived]"}
    for area in PROVEN_AREAS:
        assert latest[area].status is S.PASS, f"{area}: {latest[area].message}"
    review_cp, review_rp = latest[ReviewArea.COMPONENT_PROVENANCE], latest[ReviewArea.REGULATORY_PROVENANCE]
    assert review_cp.status is S.NOT_VERIFIED and "R1.mpn[authoritative, unarchived]" in review_cp.details["weak"] and "R1.existence[NOT_VERIFIED: datasheet_archived, mpn_in_datasheet]" in review_cp.details["weak"]
    assert review_rp.status is S.NOT_VERIFIED and "compliance not assessed" in review_rp.message and set(review_rp.details["incomplete"]) == {LVD_ID, EMC_ID, ROHS_ID, RED_ID}
    assert "retrieved_at" in review_rp.details["incomplete"][LVD_ID] and "verification_status" in review_rp.details["incomplete"][LVD_ID]
    assert not [r for r in latest.values() if r.status is S.FAIL]
    # the BOM prints the tagged MPNs (unchanged rule) but their DatasheetHash is NOT_VERIFIED: the evidence pointer is absent
    rows = _bom_rows(ir, tmp_path, ctx.tools)
    assert rows["R1"]["MPN"] == "RC0603FR-0710kL" and rows["J1"]["MPN"] == "PH1-03-UA"
    assert {row["DatasheetHash"] for row in rows.values()} == {"NOT_VERIFIED"}
    release = state.outcomes[-1]
    assert release.status is S.NOT_VERIFIED
    blocking = set(_blocking(release.message))
    assert {"regulatory.compliance", "regulatory.sources", "mfg.capability", "component.existence.R1", "ir.component_provenance", "review.component_provenance", "review.regulatory_provenance"} <= blocking
    assert not (tmp_path / "sources" / "fetch_log.jsonl").exists() and session.archive.documents() == []


# --------------------------------------------------------------------------- online


@needs_tools
def test_online_run_grounds_parts_and_regulations_against_the_fake(fake, tmp_path: Path):
    serve_everything(fake)
    ir, state, ctx, session = _run(tmp_path, fake, online=True)
    assert session.online
    audit = [e for e in session.policy.gate.audit if e["action"] == ExternalAction.NETWORK_FETCH]
    assert [e["event"] for e in audit] == ["grant", "consume"] and audit[0]["detail"] == "online session"
    assert {r.host for r in fake.requests} == {VENDOR_HOST, EUR_LEX} and all(u.startswith("https://") for u in fake.client_urls)
    assert fake.hits(RESISTOR_DS_URL) == 1 and fake.hits(HEADER_DS_URL) == 1  # one attempt per URL per run: R1 and R2 share the resistor datasheet
    assert not state.blocked
    latest = ir.validation.latest_by_check()

    # parts: archived by sha256, MPN found verbatim on page 2, evidence attached, identity grounded
    for ref in ("R1", "R2", "J1"):
        ex = latest[f"component.existence.{ref}"]
        assert ex.status is S.PASS, ex.message
        [evidence] = ex.evidence
        assert Path(evidence.path).is_file() and sha256_of(Path(evidence.path).read_bytes()) == evidence.content_hash == ex.artifact_hash
        checks = {c["name"]: c for c in ex.details["checks"]}
        assert checks["datasheet_archived"]["details"]["source"] == "fetch" or checks["datasheet_archived"]["details"]["source"] == "earlier_fetch"
        assert checks["mpn_in_datasheet"]["details"]["page"] == 2
        c = ir.component(ref)
        assert c.mpn.provenance.kind is ProvenanceKind.AUTHORITATIVE and c.mpn.provenance.source.content_hash == evidence.content_hash
        assert c.mpn.provenance.source.section == "page 2" and c.mpn.provenance.note.startswith("MPN found verbatim on page 2")
        assert c.datasheet.content_hash == evidence.content_hash and c.datasheet.document_path == evidence.path and c.datasheet.retrieved_at is not None
        assert c.datasheet.url in (RESISTOR_DS_URL, HEADER_DS_URL) and c.datasheet.title == RESISTOR_DS.title if ref != "J1" else True
        assert mpn_grounding(c, ctx.tools["archive"]).grounded and verify_source(c.datasheet, ctx.tools["archive"])[0] == "ok"
    assert ir.component("R1").datasheet.content_hash == ir.component("R2").datasheet.content_hash != ir.component("J1").datasheet.content_hash
    cp = latest["ir.component_provenance"]
    assert cp.status is S.PASS and cp.details["unverified"] == [] and cp.details["grounding"]["R1"]["existence_conflict"] is None
    rows = _bom_rows(ir, tmp_path, ctx.tools)
    assert rows["R1"]["MPN"] == "RC0603FR-0710kL" and rows["R1"]["DatasheetHash"] == ir.component("R1").mpn.provenance.source.content_hash
    assert rows["J1"]["DatasheetHash"] == ir.component("J1").datasheet.content_hash and rows["J1"]["DatasheetHash"] != rows["R1"]["DatasheetHash"]
    assert all(row["DatasheetHash"].startswith("sha256:") for row in rows.values())

    # regulatory: official texts archived and grounded, applicability decided, compliance never
    sources, applicability = latest[SOURCES_CHECK], latest[APPLICABILITY_CHECK]
    assert sources.status is S.PASS, sources.message
    assert applicability.status is S.PASS, applicability.message
    assert "4 archived and grounded" in sources.message and len(sources.evidence) == 5  # LVD, EMC, RoHS + its consolidated text, RED
    by = {r.id: r for r in ir.regulatory.requirements}
    lvd = by[LVD_ID]
    assert lvd.applicability is Applicability.NOT_APPLICABLE and lvd.status is S.NOT_APPLICABLE and lvd.source_status == "ok"
    assert lvd.applicability_inputs["radio"] == "no (answer)" and lvd.applicability_inputs["input_voltage"].startswith("12 V DC (requirement req.v_in")
    scope = lvd.grounded_quotes[0]
    assert scope.section == "Article 1" and scope.found and scope.page == 1 and "1 000 V" in scope.context and scope.content_hash == lvd.provenance.content_hash
    assert "evidence: Article 1 (grounded)" in lvd.provenance.applicability_rationale
    assert by[EMC_ID].applicability is Applicability.APPLICABLE and by[ROHS_ID].applicability is Applicability.APPLICABLE and by[RED_ID].applicability is Applicability.NOT_APPLICABLE
    assert by[EMC_ID].status is S.NOT_VERIFIED and by[ROHS_ID].status is S.NOT_VERIFIED  # they apply; compliance is not verified
    for r in by.values():
        p = r.provenance
        assert all(getattr(p, f) not in (None, "") for f in REGULATORY_PROVENANCE_FIELDS), r.id
        assert p.verification_status is S.PASS and p.source_url.startswith(f"https://{EUR_LEX}/") and Path(p.source_document).is_file()
        assert all(q.found for q in r.grounded_quotes), r.id
    assert latest[COMPLIANCE_CHECK].status is S.NOT_VERIFIED and "engineer" in latest[COMPLIANCE_CHECK].message
    assert latest[RESEARCH_CHECK].status is S.NOT_VERIFIED
    assert latest["regulatory.compliance"].details["applicable"] == [EMC_ID, ROHS_ID]

    # review: both provenance areas PASS on evidence, every proven area still PASS, release still not PASS for the right reasons
    for area in PROVEN_AREAS:
        assert latest[area].status is S.PASS, f"{area}: {latest[area].message}"
    review_cp, review_rp = latest[ReviewArea.COMPONENT_PROVENANCE], latest[ReviewArea.REGULATORY_PROVENANCE]
    assert review_cp.status is S.PASS, review_cp.message
    assert {e.content_hash for e in review_cp.evidence} == {ir.component("R1").datasheet.content_hash, ir.component("J1").datasheet.content_hash}
    assert review_rp.status is S.PASS, review_rp.message
    assert "compliance not assessed" in review_rp.message and review_rp.details["applicable"] == [EMC_ID, ROHS_ID] and len(review_rp.evidence) == 4
    release = state.outcomes[-1]
    assert release.status is S.NOT_VERIFIED
    blocking = set(_blocking(release.message))
    assert {"regulatory.compliance", "regulatory.research", "mfg.capability", "review.manufacturing_capabilities", "component.fit"} <= blocking
    assert not [b for b in blocking if (b.startswith("component.") and b != "component.fit") or b.startswith("review.component") or b.startswith("review.regulatory")
                or b in (SOURCES_CHECK, APPLICABILITY_CHECK)], blocking
    assert "not evaluated by the rules" in applicability.message and "Article 2(4) exclusions" in applicability.message  # PASS says what it did not decide

    # the IR round-trips with everything the tracks recorded
    saved = ir.save(tmp_path / "ir.json")
    back = CircuitIR.load(saved)
    assert back.content_hash() == ir.content_hash() and back.component("R1").mpn.provenance.source.content_hash == ir.component("R1").mpn.provenance.source.content_hash
    assert back.regulatory.requirements[0].grounded_quotes[0].found

    # a later run without an online session reuses the archived copies (re-hashed), fetches nothing and keeps every verdict
    fake.requests.clear()
    ir2, state2, ctx2, session2 = _run(tmp_path, fake, online=False, ir=back)
    assert fake.requests == [] and not session2.online
    latest2 = ir2.validation.latest_by_check()
    for ref in ("R1", "R2", "J1"):
        ex = latest2[f"component.existence.{ref}"]
        assert ex.status is S.PASS and {c["name"]: c for c in ex.details["checks"]}["datasheet_archived"]["details"]["source"] == "ir_reference"
    assert latest2[SOURCES_CHECK].status is S.PASS and "hash-verified copy from an earlier run" in latest2[SOURCES_CHECK].message
    assert latest2[APPLICABILITY_CHECK].status is S.PASS and latest2[COMPLIANCE_CHECK].status is S.NOT_VERIFIED
    assert latest2[ReviewArea.COMPONENT_PROVENANCE].status is S.PASS and latest2[ReviewArea.REGULATORY_PROVENANCE].status is S.PASS
    assert latest2["ir.component_provenance"].status is S.PASS
    # the components are exactly what the online run grounded (no re-tagging, no new retrieval time); the regulatory entries name the same
    # archived texts but honestly say "archived" (a copy from an earlier run) instead of "ok" (fetched in this run)
    assert [c.model_dump(mode="json") for c in ir2.components] == [c.model_dump(mode="json") for c in ir.components]
    assert {r.id: r.provenance.content_hash for r in ir2.regulatory.requirements} == {r.id: r.provenance.content_hash for r in ir.regulatory.requirements}
    assert {r.source_status for r in ir2.regulatory.requirements} == {"archived"} and {r.source_status for r in ir.regulatory.requirements} == {"ok"}

    # tamper with the archived resistor datasheet (shared by R1 and R2): the validator and the reviewer say so, nothing is upgraded back
    ds_path = Path(ir2.component("R1").datasheet.document_path)
    original = ds_path.read_bytes()
    ds_path.write_bytes(original + b"\n% altered after grounding\n")
    v = default_registry.get("ir.component_provenance").validate(ir2, ValidationContext(workdir=tmp_path, tools=ctx2.tools))[0]
    assert v.status is S.NOT_VERIFIED and v.details["tampered"] == ["R1", "R2"] and "tampered" in v.message
    assert set(v.details["unverified"]) == {"R1.mpn[authoritative, tampered]", "R2.mpn[authoritative, tampered]"}
    for tools in (ctx2.tools, {}):  # with the run's archive, and with only the workdir's sources directory (ai-eda review)
        report = IndependentReviewer(tools=tools).review(ir2, tmp_path)
        cp = {r.check_id: r for r in report.results}[ReviewArea.COMPONENT_PROVENANCE]
        assert cp.status is S.NOT_VERIFIED and "tampered" in cp.message and cp.details["tampered"] == ["R1", "R2"] and cp.details["repair"] == "human"
        assert "J1" not in cp.details["reasons"]
    ds_path.write_bytes(original)
    assert {r.check_id: r for r in IndependentReviewer(tools=ctx2.tools).review(ir2, tmp_path).results}[ReviewArea.COMPONENT_PROVENANCE].status is S.PASS
    # tamper with the archived LVD text: the regulatory review names it
    lvd_path = Path({r.id: r for r in ir2.regulatory.requirements}[LVD_ID].provenance.source_document)
    lvd_path.write_bytes(lvd_path.read_bytes().replace(b"1 000 V", b"1 500 V"))
    rp = {r.check_id: r for r in IndependentReviewer(tools=ctx2.tools).review(ir2, tmp_path).results}[ReviewArea.REGULATORY_PROVENANCE]
    assert rp.status is S.NOT_VERIFIED and rp.details["tampered"] == [LVD_ID] and "tampered" in rp.message and "compliance not assessed" in rp.message and rp.details["repair"] == "human"


@needs_libs
def test_catalog_backs_sourcing_for_listed_parts_and_leaves_the_others_not_in_catalog(fake, tmp_path: Path):
    serve_everything(fake)
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "LCSC Part #,MFR.Part #,Manufacturer,Package,Stock,Type,Price (USD)\n"
        "C1,RC0603FR-0710KL,Generic,0603,52000,Basic,0.0009\n"
        "C2,PH1-03-UA,Generic,PinHeader_1x03_P2.54mm_Vertical,120,Extended,0.05\n",
        encoding="utf-8",
    )
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.component("R2").mpn = authoritative("RC0603JR-0710kL", RESISTOR_DS)  # the 5 % variant: in the datasheet, not in the catalog
    ir, state, ctx, session = _run(tmp_path, fake, online=True, ir=ir, catalog=catalog, stop_after=Stage.IR_BUILD)
    assert session.catalog is not None and session.catalog.sha256 == sha256_of(catalog.read_bytes())
    latest = ir.validation.latest_by_check()
    r1, r2, j1 = latest["component.existence.R1"], latest["component.existence.R2"], latest["component.existence.J1"]
    assert r1.status is S.PASS and j1.status is S.PASS and r2.status is S.NOT_VERIFIED
    assert {c["name"]: c["status"] for c in r2.details["checks"]} == {"symbol": "PASS", "footprint": "PASS", "datasheet_pointer": "PASS", "datasheet_archived": "PASS", "mpn_in_datasheet": "PASS", "catalog": "NOT_VERIFIED"}
    assert "not in catalog" in r2.message and r2.details["catalog"]["row"] is None and r2.details["catalog"]["sha256"] == session.catalog.sha256
    assert [e.description for e in r1.evidence] == ["archived datasheet of R1", "JLCPCB catalog export (row 2)"]
    s = ir.component("R1").sourcing
    assert [x.supplier for x in s] == ["JLCPCB"] and s[0].supplier_part_number.value == "C1" and s[0].stock.value == 52000 and s[0].assembly_class.value == "Basic"
    assert s[0].supplier_part_number.provenance.kind is ProvenanceKind.AUTHORITATIVE and s[0].supplier_part_number.provenance.source.content_hash == session.catalog.sha256
    assert ir.component("R2").sourcing == [] and ir.component("J1").sourcing[0].supplier_part_number.value == "C2"
    # identity is grounded for all three (the catalog is sourcing, not identity) ...
    assert latest["ir.component_provenance"].status is S.PASS
    assert all(ir.component(ref).mpn_tagged_authoritative for ref in ("R1", "R2", "J1"))
    rows = _bom_rows(ir, tmp_path, ctx.tools)
    assert rows["R1"]["Supplier"] == "JLCPCB" and rows["R1"]["SupplierPN"] == "C1" and rows["R2"]["Supplier"] == "NOT_VERIFIED" and rows["R2"]["MPN"] == "RC0603JR-0710kL"
    assert rows["R2"]["DatasheetHash"] == rows["R1"]["DatasheetHash"] != "NOT_VERIFIED"
    # ... but the reviewer passes the existence check's worst sub-check through: mixed statuses, R2 named with the reason
    report = {r.check_id: r for r in IndependentReviewer(tools=ctx.tools).review(ir, tmp_path).results}
    cp = report[ReviewArea.COMPONENT_PROVENANCE]
    assert cp.status is S.NOT_VERIFIED and cp.details["weak"] == ["R2.existence[NOT_VERIFIED: catalog]"] and "not in catalog" in cp.message and "R1" not in cp.details["reasons"]


# --------------------------------------------------------------------------- the session and the answers, without the real tools


def test_session_trust_rules_and_usage_errors(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    from tests.test_parts_existence import make_part

    ir = CircuitIR(project=ProjectMeta(id="s", name="s", workdir=str(tmp_path)))
    ir.components = [make_part("R1"), make_part("R2", symbol="NoDs")]
    session = open_session(workdir=tmp_path, ir=ir, library=lib, online=False, trust_hosts=["Mirror.Example.org", " "], datasheet_urls={"R2": "http://mirror.example.org/r2.pdf"},
                           source_urls={"eu_lvd": "https://eur-lex.europa.eu/x"}, gate=ApprovalGate())
    try:
        reasons = session.policy.trust_reasons
        assert reasons["example-vendor.com"].startswith("KiCad Datasheet field of Test:VR1")  # rule (a): the library's own Datasheet host
        assert reasons["eur-lex.europa.eu"].startswith("official domain listed for reg.EU.LVD.2014-35-EU") and "law.go.kr" in reasons and "ecfr.gov" in reasons
        assert reasons["mirror.example.org"] == "host trusted by the user (--trust-host)"
        assert session.policy.user_urls == {"R2": "http://mirror.example.org/r2.pdf", "eu_lvd": "https://eur-lex.europa.eu/x"}
        assert session.policy.trusted_origin("https://mirror.example.org/r2.pdf")[0] == "mirror.example.org"
        assert session.policy.trusted_origin("https://model-invented.example.com/ds.pdf")[0] is None  # a model's host is never trusted
        assert not session.online and not session.archive.online and session.sources_dir == tmp_path / "sources"
        assert session.tools()["archive"] is session.archive and session.tools()["regulatory_candidates"].get("reg.EU.LVD.2014-35-EU") is not None and "catalog" not in session.tools()
        assert "offline" in session.summary() and "trusted hosts: 5" in session.summary()
        assert not [e for e in session.policy.gate.audit if e["action"] == ExternalAction.NETWORK_FETCH]
    finally:
        session.close()
    online = open_session(workdir=tmp_path, ir=ir, library=lib, online=True, gate=ApprovalGate(), approved_by="test")
    try:
        assert online.online and [e["event"] for e in online.policy.gate.audit] == ["grant", "consume"] and online.policy.gate.audit[0]["by"] == "test"
    finally:
        online.close()
    with pytest.raises(SessionError, match="--catalog-date"):
        open_session(workdir=tmp_path, ir=ir, library=lib, online=False, catalog=tmp_path / "nope.csv", gate=ApprovalGate())
    with pytest.raises(SessionError, match="--catalog"):
        open_session(workdir=tmp_path, ir=ir, library=lib, online=False, catalog=tmp_path / "nope.csv", catalog_date="2026-09-23", gate=ApprovalGate())
    with pytest.raises(SessionError, match="different URLs"):
        open_session(workdir=tmp_path, ir=ir, library=lib, online=False, datasheet_urls={"R1": "https://a.example/1"}, source_urls={"R1": "https://a.example/2"}, gate=ApprovalGate())
    with pytest.raises(SessionError, match="regulatory candidates"):
        open_session(workdir=tmp_path, ir=ir, library=lib, online=False, candidates=tmp_path / "missing.json", gate=ApprovalGate())
    from ai_eda.workflow.session import parse_key_urls

    assert parse_key_urls(["R1=https://a.example/x", " U2 = https://b.example/y "], "--datasheet-url") == {"R1": "https://a.example/x", "U2": "https://b.example/y"}
    with pytest.raises(SessionError, match="KEY=URL"):
        parse_key_urls(["R1"], "--datasheet-url")


def test_offline_paths_import_without_httpx():
    """The agents, the archive's offline side (add_file / load / verify) and the CLI import with the network client absent."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['httpx'] = None\n"  # makes `import httpx` raise ImportError
        "import ai_eda.agents, ai_eda.cli, ai_eda.parts.identity, ai_eda.workflow.session\n"
        "from ai_eda.tools.sources import DocumentArchive, NetworkPolicy\n"
        "from ai_eda.security.approval import ApprovalGate\n"
        "import tempfile, pathlib\n"
        "root = pathlib.Path(tempfile.mkdtemp())\n"
        "a = DocumentArchive(root, NetworkPolicy(approved=False, gate=ApprovalGate()))\n"
        "p = root / 'note.txt'; p.write_text('LM2931AZ-5.0/NOPB ' * 8, encoding='utf-8')\n"
        "doc = a.add_file(p, title='note', retrieved_at='2026-09-23')\n"
        "assert a.verify(doc.source_ref()) == 'ok' and doc.find_quote('LM2931AZ-5.0/NOPB')\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_scope_answers_are_regulatory_requirements_and_control_answers_are_none(tmp_path: Path):
    """``--answer mains_powered=no`` is the user's regulatory scope statement, not an electrical requirement a component must serve."""
    ir = CircuitIR(project=ProjectMeta(id="a", name="a", workdir=str(tmp_path)))
    ctx = AgentContext(workdir=tmp_path, answers={**SCOPE_ANSWERS, "confirm_parts": "yes", "confirm_facts": "yes", "propose_regulations": "yes", "extract_datasheet_facts": "yes",
                                                  "accept_regulations": "x", "datasheet_facts_file": "f.json", "ripple": "50 mV"})
    result = RequirementAgent().run(ir, ctx)
    Orchestrator.apply_proposals(ir, result.proposals)
    reqs = {r.key: r for r in ir.requirements.requirements}
    assert set(reqs) == {"application", "jurisdiction", "mains_powered", "radio", "finished_apparatus", "evaluation_kit", "highest_rated_voltage", "ripple"}
    assert {k: r.category for k, r in reqs.items()} == {"application": "application", "jurisdiction": "regulatory", "mains_powered": "regulatory", "radio": "regulatory",
                                                        "finished_apparatus": "regulatory", "evaluation_kit": "regulatory", "highest_rated_voltage": "regulatory", "ripple": "electrical"}
    assert reqs["mains_powered"].value.value == "no" and reqs["mains_powered"].value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert not result.blocked_on_user and {q.key for q in result.questions} == {"operating_temperature", "protection"}  # the optional baseline ones only
    # the reviewer does not demand a component for a scope answer
    report = {r.check_id: r for r in IndependentReviewer().review(ir, tmp_path).results}
    assert report[ReviewArea.REQUIREMENTS_VS_IR].status is S.FAIL and report[ReviewArea.REQUIREMENTS_VS_IR].details["unserved"] == ["req.ripple"]


# --------------------------------------------------------------------------- the CLI


def _saved_ir(tmp_path: Path) -> Path:
    ir = _routed_ir(tmp_path)
    return ir.save(tmp_path / "ir.json")


def _answer_args() -> list[str]:
    return [arg for k, v in SCOPE_ANSWERS.items() for arg in ("--answer", f"{k}={v}")]


@needs_tools
def test_cli_offline_run_builds_no_http_client_and_reports_what_is_unverified(fake, tmp_path: Path, monkeypatch):
    serve_everything(fake)
    ir_path = _saved_ir(tmp_path)

    def no_socket(self):
        raise AssertionError("offline run must not build an HTTP client")

    monkeypatch.setattr(DocumentArchive, "_http", no_socket)
    code, out, err = _cli("run", str(ir_path), *_answer_args())
    assert code == 0, err  # NOT_VERIFIED, nothing wrong, nothing proven
    assert out.startswith("sources: ") and "offline: nothing is fetched" in out.splitlines()[0]
    assert "COMPONENT EXISTENCE" in out and "component.existence.R1 NOT_VERIFIED" in out and "datasheet_archived: NOT_VERIFIED: not fetched (offline" in out
    assert "regulatory_research      NOT_VERIFIED" in out and "BLOCKED" not in out
    # the scope questions were answered; only the requirement stage's optional baseline questions stay open (listed, not blocking)
    optional = out.split("OPTIONAL QUESTIONS", 1)[1]
    assert "[operating_temperature]" in optional and "[protection]" in optional and "mains_powered" not in optional and "radio" not in optional
    assert fake.requests == [] and not (tmp_path / "sources" / "fetch_log.jsonl").exists()
    ir = CircuitIR.load(ir_path)
    assert ir.validation.latest("component.existence.R1").status is S.NOT_VERIFIED and ir.validation.latest(SOURCES_CHECK).status is S.NOT_VERIFIED
    # usage errors exit 2 before anything runs
    before = ir_path.read_bytes()
    code, _, err = _cli("run", str(ir_path), "--catalog", str(tmp_path / "nope.csv"))
    assert code == 2 and "--catalog-date" in err
    code, _, err = _cli("run", str(ir_path), "--datasheet-url", "R1")
    assert code == 2 and "KEY=URL" in err
    code, _, err = _cli("run", str(ir_path), "--regulatory-candidates", str(tmp_path / "missing.json"))
    assert code == 2 and "regulatory candidates" in err
    assert ir_path.read_bytes() == before


@needs_tools
def test_cli_online_run_fetches_through_the_fake_and_review_reverifies(fake, tmp_path: Path, monkeypatch):
    serve_everything(fake)
    ir_path = _saved_ir(tmp_path)
    real_client = httpx.Client

    def loopback_client(**kw):
        kw.setdefault("transport", fake.transport())
        return real_client(**kw)

    monkeypatch.setattr(httpx, "Client", loopback_client)
    audit_before = len(default_gate().audit)
    code, out, err = _cli("run", str(ir_path), "--online", "--trust-host", VENDOR_HOST, *_answer_args())
    assert code == 0, err
    assert "online (fetches allowed" in out.splitlines()[0] and "COMPONENT EXISTENCE" not in out
    grants = [e for e in default_gate().audit[audit_before:] if e["action"] == ExternalAction.NETWORK_FETCH]
    assert [e["event"] for e in grants] == ["grant", "consume"] and grants[0]["by"] == "cli --online" and grants[0]["detail"] == "online session"
    assert {r.host for r in fake.requests} == {VENDOR_HOST, EUR_LEX}
    ir = CircuitIR.load(ir_path)
    latest = ir.validation.latest_by_check()
    assert all(latest[f"component.existence.{ref}"].status is S.PASS for ref in ("R1", "R2", "J1"))
    assert latest[SOURCES_CHECK].status is S.PASS and latest[APPLICABILITY_CHECK].status is S.PASS and latest[COMPLIANCE_CHECK].status is S.NOT_VERIFIED
    assert latest[ReviewArea.COMPONENT_PROVENANCE].status is S.PASS and latest[ReviewArea.REGULATORY_PROVENANCE].status is S.PASS
    assert (tmp_path / "sources" / "fetch_log.jsonl").is_file() and (tmp_path / "sources").glob("*.meta.json")
    # `ai-eda review` re-verifies the archived copies read-only (no client is built) and agrees
    monkeypatch.setattr(DocumentArchive, "_http", lambda self: (_ for _ in ()).throw(AssertionError("review must not fetch")))
    fake.requests.clear()
    code, out, _ = _cli("review", str(ir_path))
    assert code == 0 and fake.requests == []
    lines = {line.split()[0]: line.split()[1] for line in out.splitlines() if line.startswith("review.")}
    assert lines["review.component_provenance"] == "PASS" and lines["review.regulatory_provenance"] == "PASS"
