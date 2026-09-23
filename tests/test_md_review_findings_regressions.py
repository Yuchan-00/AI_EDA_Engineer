"""Regression tests for the 2026-09-23 documentation-and-code review (one section per finding).

The review audited README / ARCHITECTURE / CLAUDE.md against the code and
reviewed the code for defects; the doc findings were fixed in the documents,
the code findings here. Nothing touches the network: the web is
``tests/fake_sources.py``, OpenRouter is ``tests/fake_openrouter.py``, the
model is ``ScriptedLLMClient``.

1.  BOM: the Reference / Footprint / Supplier cells (and the CPL Designator) were not gated against CSV formula
    injection, and identity cells were checked unstripped.
2.  ``confirm_requirements=sure``: a one-word unlisted yes was appended to the request as a correction (hash
    change + a billed re-extraction).
3.  The no-LLM checklist re-asked ``jurisdiction`` although ``ir.regulatory.jurisdictions`` held it.
4.  ``find_codemodel_dir`` did not know Debian/Ubuntu's ``<libdir>/ngspice`` layout.
5.  ``apply_proposals`` stored unvalidated payloads (a dict for ``topology``, a str for ``components``).
6.  A FAIL in IR_BUILD outranked USER_INPUT_REQUIRED: the pipeline ran on and the CLI never showed the question.
7.  ``remove`` proposals matched by model equality, which includes the wall-clock ``created_at``: a rebuilt payload
    silently removed nothing.
8.  ``CircuitIR.load`` dropped unknown keys and accepted any ``schema_version``.
9.  ``Traced[float]`` coerced ``"5"`` / ``True``; tuples came back as lists after a JSON round trip.
10. OpenRouter: ``model`` / ``id`` / ``finish_reason`` / headers / ``error.code`` were not redacted.
11. ``DocumentArchive.lookup`` raised on a corrupt meta and trusted a meta's ``sha256`` over the file name.
12. An accepted model proposal's URL was fetched without re-checking the official-domain allow-list.
13. ``normalise_url`` dropped IPv6 brackets; a port was not part of the origin.
14. ``OPENROUTER_BASE_URL`` accepted plain http to any host.
15. The archive's HTTP-status mapping (401/407/429/503 blocked, 410 missing, tiny body blocked) had no test.
16. RELEASE counted a PASS without a tool, or a PASS for another IR version, as evidence.
17. The design hash included locators (``SourceRef.document_path``, ``LibraryRef.library_path``,
    ``RegulatoryProvenance.source_document``) and regulatory verification outcomes, so the same design in another
    folder or after an online run hashed differently, while a parameter *named* ``created_at`` was stripped by name.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eda.agents import AgentContext, RequirementAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.regulatory import ACCEPT_REGS_KEY
from ai_eda.compilers import BOMCompiler, CompileContext, CPLCompiler
from ai_eda.errors import CompileError, IRSchemaError, ToolUnavailableError
from ai_eda.ir import (
    CircuitDomain,
    SourceRef,
    CircuitIR,
    Jurisdiction,
    Net,
    NetKind,
    PCBDesign,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    SourcingInfo,
    Topology,
    Traced,
    ValidationResult,
    ValidationStatus as S,
    assumption,
    authoritative,
    user_requirement,
)
from ai_eda.ir.regulatory import GroundedQuote, RegulatoryProvenance, RegulatoryRequirement
from ai_eda.llm.client import LLMError, LLMMessage
from ai_eda.llm.extraction import CONFIRM_KEY, is_correction
from ai_eda.llm.openrouter import REDACTED, OpenRouterClient, check_base_url
from ai_eda.security import ApprovalGate
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy
from ai_eda.tools.sources.policy import host_of, normalise_url, port_of
from ai_eda.tools.spice.ngspice_shared import find_codemodel_dir
from ai_eda.workflow import Orchestrator, Stage
from tests.conftest import AUTH, DS, make_component
from tests.fake_openrouter import FakeOpenRouter
from tests.fake_sources import FakeSources
from tests.test_archive import make_archive, online_policy
from tests.test_regulatory_agent import CANNED, GOOD_ID, _ir as _reg_ir, _llm_ctx, _run_agent, _service as _reg_service
from tests.test_regulatory_research import online_archive, serve_all
from tests.test_requirement_agent_llm import CANNED as REQ_CANNED, USAGE, _ir as _req_ir, _run as _req_run, _service as _req_service

ANSWERS = {"application": "test", "jurisdiction": "EU"}


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


def _ir(tmp_path: Path, *components) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    ir.components = list(components)
    return ir


# --------------------------------------------------------------------------- 1: every BOM / CPL cell is gated


@pytest.mark.parametrize(
    "mutate, cell",
    [
        (lambda c: setattr(c, "ref", "=1+1"), "Reference"),
        (lambda c: setattr(c, "sourcing", [SourcingInfo(supplier="=cmd|' /C calc'!A0")]), "Supplier"),
        (lambda c: setattr(c.footprint, "library", "@evil"), "Footprint"),
        (lambda c: setattr(c, "mpn", authoritative(" =1+1", DS)), "MPN"),  # checked stripped, like the catalog reader
    ],
)
def test_1_bom_refuses_every_cell_a_spreadsheet_would_execute(tmp_path: Path, mutate, cell):
    part = make_component("R1", "10k")
    mutate(part)
    with pytest.raises(CompileError, match=f"{cell}.*refusing to write a cell"):
        BOMCompiler().compile(_ir(tmp_path, part), CompileContext(workdir=tmp_path, tools={}))


def test_1_cpl_designator_is_gated_and_plain_cells_still_print(tmp_path: Path):
    part = make_component("R1", "10k")
    part.sourcing = [SourcingInfo(supplier="JLCPCB", supplier_part_number=authoritative("C25804", DS))]
    ir = _ir(tmp_path, part)
    art = BOMCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))
    row = Path(art.path).read_text(encoding="utf-8").splitlines()[1]
    assert row.startswith("R1,10k,resistor,NOT_VERIFIED,MPN-10k,0603,Resistor_SMD:R_0603_1608Metric,JLCPCB,C25804,")
    ir.pcb = PCBDesign(placements=[Placement(component_ref="-R1", x_mm=0.0, y_mm=0.0, provenance=AUTH)])
    with pytest.raises(CompileError, match="Designator.*refusing to write a cell"):
        CPLCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))


# --------------------------------------------------------------------------- 2: a one-word unlisted yes is not a correction


def test_2_a_single_word_without_a_digit_is_not_a_correction():
    assert not is_correction("sure") and not is_correction("fine") and not is_correction("좋습니다") and not is_correction("yup")
    assert is_correction("the input is 24V not 12V") and is_correction("24 V in") and is_correction("입력은 24V")
    assert not is_correction("24V")  # shorter than MIN_CORRECTION_CHARS: asked again, as before


def test_2_an_unlisted_yes_costs_no_call_and_moves_no_hash(tmp_path: Path):
    svc, client = _req_service([{"structured": REQ_CANNED, "usage": USAGE}])
    ir = _req_ir(tmp_path)
    _req_run(ir, svc, tmp_path)
    before = ir.content_hash()
    state = _req_run(ir, svc, tmp_path, answers={CONFIRM_KEY: "sure"})
    assert len(client.calls) == 1 and ir.requirements.corrections == [] and ir.content_hash() == before
    assert state.blocked and CONFIRM_KEY in [q.key for q in state.open_questions]
    assert any("not understood" in n for n in state.outcome(Stage.REQUIREMENT_ANALYSIS).message.split("; "))


# --------------------------------------------------------------------------- 3: a recorded jurisdiction is an answered question


def test_3_no_llm_checklist_does_not_re_ask_a_recorded_jurisdiction(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="j", name="j", workdir=str(tmp_path)))
    ir.regulatory.jurisdictions = [Jurisdiction(code="EU", name="EU", provided_by_user=True)]
    ir.requirements.requirements.append(Requirement(id="req.application", key="application", text="bench", kind=RequirementKind.EXPLICIT, value=user_requirement("bench")))
    result = RequirementAgent().run(ir, AgentContext(workdir=tmp_path))
    keys = [q.key for q in result.questions]
    assert "jurisdiction" not in keys and "application" not in keys and not result.blocked_on_user
    assert not [p for p in result.proposals if p.target == "regulatory.jurisdictions"]


# --------------------------------------------------------------------------- 4: Debian/Ubuntu code model layout


def test_4_find_codemodel_dir_knows_the_debian_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NGSPICE_CODEMODEL_DIR", raising=False)
    libdir = tmp_path / "usr" / "lib" / "x86_64-linux-gnu"
    (libdir / "ngspice").mkdir(parents=True)
    dll = libdir / "libngspice.so.0"
    dll.write_bytes(b"")
    assert find_codemodel_dir(dll) == libdir / "ngspice"
    kicad = tmp_path / "KiCad" / "10.0"
    (kicad / "lib" / "ngspice").mkdir(parents=True)
    (kicad / "bin").mkdir()
    assert find_codemodel_dir(kicad / "bin" / "ngspice.dll") == kicad / "lib" / "ngspice"
    assert find_codemodel_dir(tmp_path / "nowhere" / "libngspice.so.0") is None


# --------------------------------------------------------------------------- 5: proposals are validated against the IR's types


def test_5_apply_proposals_validates_payloads(divider_ir: CircuitIR):
    with pytest.raises(ValueError, match="topology.*not a valid"):
        Orchestrator.apply_proposals(divider_ir, [IRProposal(description="bad topology", target="topology", operation="set", payload={"name": "garbage", "no_provenance": True})])
    assert isinstance(divider_ir.topology, Topology)
    with pytest.raises(ValueError, match="components.*not a valid"):
        Orchestrator.apply_proposals(divider_ir, [IRProposal(description="bad components", target="components", operation="set", payload="not a list")])
    assert isinstance(divider_ir.components, list) and len(divider_ir.components) == 2
    # a JSON-shaped payload that IS valid becomes the model, so a serialised proposal (a future GUI) round-trips
    Orchestrator.apply_proposals(divider_ir, [
        IRProposal(description="topology as data", target="topology", operation="set", payload={"name": "buck", "domains": ["analog"], "provenance": {"kind": "user_requirement"}}),
        IRProposal(description="parameter as data", target="parameters.v_x", operation="set", payload={"value": 1.5, "unit": "V", "provenance": {"kind": "user_requirement"}}),
    ])
    assert isinstance(divider_ir.topology, Topology) and divider_ir.topology.domains == [CircuitDomain.ANALOG]
    assert isinstance(divider_ir.parameters["v_x"], Traced) and divider_ir.parameters["v_x"].value == 1.5
    CircuitIR.model_validate_json(divider_ir.model_dump_json())  # what was applied still loads
    with pytest.raises(ValueError, match="no field"):
        Orchestrator.apply_proposals(divider_ir, [IRProposal(description="typo", target="componentz", operation="set", payload=[])])


# --------------------------------------------------------------------------- 6: a required question always stops the pipeline


def test_6_fail_in_the_same_stage_does_not_hide_a_required_question(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.nets.append(Net(name="X", pins=[PinRef(component_ref="R9", pin_number="1")], provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test")))  # ir.connectivity FAIL
    divider_ir.parameters["guess"] = assumption(1.0, note="unconfirmed", unit="V")  # ir.assumptions USER_INPUT_REQUIRED
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=ANSWERS)).run(divider_ir)
    build = state.outcome(Stage.IR_BUILD)
    assert build.status is S.FAIL  # the aggregate still says FAIL ...
    assert state.blocked and state.current is Stage.IR_BUILD and state.outcome(Stage.CALCULATION) is None  # ... but the user is asked, not run past
    assert [q.key for q in state.open_questions] == ["ir.assumptions"]


# --------------------------------------------------------------------------- 7: remove matches design content and never no-ops silently


def test_7_remove_matches_by_design_content(divider_ir: CircuitIR):
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
    rebuilt = Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="R2", pin_number="2")], provenance=net_p)  # a later created_at
    assert rebuilt != divider_ir.net("GND")  # model equality sees the wall clock ...
    Orchestrator.apply_proposals(divider_ir, [IRProposal(description="drop GND", target="nets", operation="remove", payload=rebuilt)])
    assert divider_ir.net("GND") is None  # ... the proposal does not
    vout = divider_ir.net("VOUT").model_dump(mode="json")  # a serialised payload (what a GUI would send)
    Orchestrator.apply_proposals(divider_ir, [IRProposal(description="drop VOUT", target="nets", operation="remove", payload=vout)])
    assert divider_ir.net("VOUT") is None
    with pytest.raises(ValueError, match="nothing in nets matches"):
        Orchestrator.apply_proposals(divider_ir, [IRProposal(description="drop again", target="nets", operation="remove", payload=rebuilt)])


# --------------------------------------------------------------------------- 8: load refuses what it would silently drop


def test_8_load_refuses_unknown_keys_and_foreign_schema_versions(divider_ir: CircuitIR, tmp_path: Path):
    path = divider_ir.save(tmp_path / "ir.json")
    assert CircuitIR.load(path).content_hash() == divider_ir.content_hash()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["components"][0]["serves_requirement"] = raw["components"][0].pop("serves_requirements")  # a typo in a hand edit
    raw["bogus_top"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(IRSchemaError, match=r"components\[0\]\.serves_requirement.*bogus_top|bogus_top.*components\[0\]\.serves_requirement"):
        CircuitIR.load(path)
    raw = json.loads(divider_ir.model_dump_json())
    raw["schema_version"] = "banana"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(IRSchemaError, match="schema_version 'banana'"):
        CircuitIR.load(path)


# --------------------------------------------------------------------------- 9: Traced values are what they say


def test_9_traced_numbers_are_numbers_and_containers_are_json_shaped(divider_ir: CircuitIR):
    with pytest.raises(ValidationError, match="must be a number"):
        Traced[float](value="5", provenance=AUTH)
    with pytest.raises(ValidationError, match="must be a number"):
        Traced[float](value=True, provenance=AUTH)
    with pytest.raises(ValidationError, match="must be a number"):
        Traced[int](value="5", provenance=AUTH)
    assert Traced[float](value=5, provenance=AUTH).value == 5.0
    pts = user_requirement([(0.0, 0.0), (1e-3, 5.0)])
    assert pts.value == [[0.0, 0.0], [0.001, 5.0]]
    divider_ir.parameters["pts"] = pts
    loaded = CircuitIR.model_validate_json(divider_ir.model_dump_json())
    assert loaded.parameters["pts"] == divider_ir.parameters["pts"] and loaded.design_dict() == divider_ir.design_dict()


# --------------------------------------------------------------------------- 10: every response field is redacted


def test_10_key_echoed_in_response_fields_headers_and_error_code_never_leaves_the_client(caplog: pytest.LogCaptureFixture):
    with FakeOpenRouter() as fake:
        key = fake.api_key
        client = OpenRouterClient(api_key=key, base_url=fake.base_url)
        try:
            caplog.set_level(logging.DEBUG, logger="ai_eda.llm")
            body = {
                "id": f"gen-{key}", "model": f"m-{key}", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": f"stop-{key}", "native_finish_reason": f"n-{key}"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
            }
            fake.script(status=200, body=body, headers={"X-Generation-Id": f"g-{key}", "X-Provider-Name": f"p-{key}"})
            resp = client.complete("m", [LLMMessage(role="user", content="x")])
            for field in ("model_used", "id", "finish_reason", "native_finish_reason", "generation_id", "provider_name"):
                assert key not in str(getattr(resp, field)) and REDACTED in str(getattr(resp, field)), field
            fake.add_error(400, "bad request", code=f"bad-{key}")
            with pytest.raises(LLMError) as ei:
                client.complete("m", [LLMMessage(role="user", content="x")])
            assert key not in str(ei.value) and key not in str(ei.value.code) and REDACTED in str(ei.value.code)
            assert all(key not in r.getMessage() for r in caplog.records)
        finally:
            client.close()


# --------------------------------------------------------------------------- 11: a corrupt or planted meta answers for nothing


def test_11_lookup_ignores_corrupt_and_planted_metas(tmp_path: Path):
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate()))
    root = archive.root
    root.mkdir(parents=True, exist_ok=True)
    (root / ("0" * 64 + ".meta.json")).write_text(json.dumps({"normalised_url": "https://www.ti.com/x", "status": "ok", "sha256": "garbage"}), encoding="utf-8")
    assert archive.lookup("https://www.ti.com/x") is None  # used to raise ValueError('not a sha256')
    body = b"a real archived text\n" * 8
    hexd = hashlib.sha256(body).hexdigest()
    (root / f"{hexd}.txt").write_bytes(body)
    (root / f"{hexd}.meta.json").write_text(json.dumps({"normalised_url": "https://www.ti.com/y", "status": "ok", "sha256": f"sha256:{hexd}", "kind": "text", "ext": "txt"}), encoding="utf-8")
    assert archive.lookup("https://www.ti.com/y").sha256 == f"sha256:{hexd}"
    (root / ("1" * 64 + ".meta.json")).write_text(json.dumps({"normalised_url": "https://www.ti.com/z", "status": "ok", "sha256": f"sha256:{hexd}"}), encoding="utf-8")
    assert archive.lookup("https://www.ti.com/z") is None  # the meta's sha256 names another document: planted, not evidence


# --------------------------------------------------------------------------- 12: an accepted proposal is fetched only from an allow-listed host


def test_12_accepted_proposal_host_is_checked_again_at_fetch_time(fake, tmp_path: Path):
    serve_all(fake)
    svc, client = _reg_service([{"structured": CANNED, "usage": {"prompt_tokens": 300, "completion_tokens": 120, "cost_usd": 0.002}}])
    ir = _reg_ir(tmp_path, "EU")
    _run_agent(ir, _llm_ctx(tmp_path, svc))  # proposals made and shown
    _run_agent(ir, _llm_ctx(tmp_path, svc, **{ACCEPT_REGS_KEY: GOOD_ID}))  # accepted offline
    p = next(p for p in ir.regulatory.proposed_candidates if p.id == GOOD_ID)
    assert p.decision == "accepted" and p.refused is None
    p.official_url = "https://blog.example/eu-rules"  # a hand edit after screening (the stored ``refused`` stays clear)
    archive = online_archive(tmp_path / "sources", fake)
    result = _run_agent(ir, _llm_ctx(tmp_path, svc, archive))
    assert len(client.calls) == 1
    assert not [r for r in fake.requests if r.host == "blog.example"] and "blog.example" not in archive.policy.trusted_hosts
    assert any(GOOD_ID in n and "not in the official-domain allow-list" in n and "not fetched" in n for n in result.notes)
    assert GOOD_ID not in [r.id for r in ir.regulatory.requirements]


# --------------------------------------------------------------------------- 13: IPv6 literals and ports


def test_13_normalise_url_keeps_ipv6_brackets_and_treats_the_port_as_part_of_the_origin():
    assert normalise_url("https://[::1]/x")[0] == "https://[::1]/x" and host_of("https://[::1]/x") == "::1"
    assert normalise_url("https://[2001:db8::1]:8443/x")[0] == "https://[2001:db8::1]:8443/x"
    assert normalise_url("http://ti.com:80/x")[0] == "https://ti.com/x"  # the http default port is not a port
    assert normalise_url("https://ti.com:443/x")[0] == "https://ti.com/x"
    assert normalise_url("https://ti.com:8443/x")[0] == "https://ti.com:8443/x"
    with pytest.raises(ValueError, match="unusable port"):
        normalise_url("https://ti.com:notaport/x")
    assert port_of("https://ti.com:8443/x") == 8443 and port_of("https://ti.com/x") is None
    policy = NetworkPolicy(approved=False, trusted_hosts={"ti.com"}, user_urls={"R1": "https://ti.com:8443/exact.pdf"}, gate=ApprovalGate())
    assert policy.trusted_origin("https://ti.com/x.pdf")[0] == "ti.com"
    origin, why = policy.trusted_origin("https://ti.com:8443/other.pdf")
    assert origin is None and "port 8443" in why
    assert policy.trusted_origin("https://ti.com:8443/exact.pdf")[1].startswith("URL supplied by the user")
    assert policy.redirect_allowed("ti.com", "https://ti.com/y") is None
    assert "port 9999" in policy.redirect_allowed("ti.com", "https://ti.com:9999/")


# --------------------------------------------------------------------------- 14: the key travels over TLS or to loopback only


def test_14_base_url_must_be_https_or_loopback():
    assert check_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"
    assert check_base_url("http://127.0.0.1:1/v1") == "http://127.0.0.1:1/v1" and check_base_url("http://localhost:1") == "http://localhost:1"
    for bad in ("http://example.com/v1", "ftp://openrouter.ai", "openrouter.ai/api/v1", ""):
        with pytest.raises(ToolUnavailableError, match="must be https"):
            check_base_url(bad)
    with pytest.raises(ToolUnavailableError) as ei:
        OpenRouterClient(api_key="sk-secret-key", base_url="http://example.com/v1")
    assert "sk-secret-key" not in str(ei.value)


# --------------------------------------------------------------------------- 15: the archive's HTTP status mapping


@pytest.mark.parametrize("status, expected", [(401, "blocked"), (407, "blocked"), (429, "blocked"), (503, "blocked"), (410, "missing"), (418, "error")])
def test_15_http_statuses_map_to_the_documented_outcomes(fake, tmp_path: Path, status: int, expected: str):
    url = f"https://www.ti.com/s{status}.html"
    fake.serve(url, "<html><body>" + "words " * 200 + "</body></html>", "text/html; charset=utf-8", status=status)
    out = make_archive(tmp_path, fake, online_policy()).fetch(url, purpose="p")
    assert out.status == expected and out.http_status == status and out.document is None


def test_15_a_body_under_64_bytes_is_not_a_document(fake, tmp_path: Path):
    fake.serve("https://www.ti.com/tiny.txt", "x" * 63, "text/plain")
    fake.serve("https://www.ti.com/small.txt", "x" * 64, "text/plain")
    archive = make_archive(tmp_path, fake, online_policy())
    tiny = archive.fetch("https://www.ti.com/tiny.txt", purpose="p")
    assert tiny.status == "blocked" and "63 bytes" in tiny.reason
    assert archive.fetch("https://www.ti.com/small.txt", purpose="p").ok


# --------------------------------------------------------------------------- 16: RELEASE releases on evidence only


def test_16_release_ignores_opinions_and_results_for_another_ir_version(divider_ir: CircuitIR, tmp_path: Path):
    orch = Orchestrator(AgentContext(workdir=tmp_path, answers=ANSWERS))
    ctx = orch.ctx
    divider_ir.validation.extend([ValidationResult(check_id="opinion", status=S.PASS)])
    out = orch._release(divider_ir, ctx)
    assert out.status is S.NOT_VERIFIED and "opinion" in out.message and "not evidence" in out.message
    divider_ir.validation.extend([ValidationResult(check_id="opinion", status=S.PASS, tool="some.tool", ir_hash="sha256:" + "0" * 64)])
    out = orch._release(divider_ir, ctx)
    assert out.status is S.NOT_VERIFIED and "another IR version" in out.message
    divider_ir.validation.extend([ValidationResult(check_id="opinion", status=S.PASS, tool="some.tool", ir_hash=divider_ir.content_hash())])
    assert orch._release(divider_ir, ctx).status is S.PASS


# --------------------------------------------------------------------------- 17: the design hash hashes the design


def test_17_locators_and_verification_outcomes_do_not_move_the_design_hash(divider_ir: CircuitIR, tmp_path: Path):
    r1 = divider_ir.components[0]
    r1.mpn = authoritative("MPN-10k", SourceRef(title="ds", content_hash="sha256:" + "a" * 64, document_path=str(tmp_path / "projA" / "sources" / "x.pdf")))
    h = divider_ir.content_hash()
    r1.mpn.provenance.source.document_path = str(tmp_path / "projB" / "sources" / "x.pdf")
    assert divider_ir.content_hash() == h  # the same archived document (same sha256) under another workdir
    r1.mpn.provenance.source.content_hash = "sha256:" + "b" * 64
    assert divider_ir.content_hash() != h  # another document is another design
    h = divider_ir.content_hash()
    r1.symbol.library_path = "/usr/share/kicad/symbols/Device.kicad_sym"
    assert divider_ir.content_hash() == h
    r1.symbol.library_path = r"C:\Program Files\KiCad\10.0\share\kicad\symbols\Device.kicad_sym"
    assert divider_ir.content_hash() == h  # the KiCad install is a locator, not the design
    req = RegulatoryRequirement(id="reg.EU.LVD", jurisdiction="EU", title="LVD", provenance=RegulatoryProvenance(jurisdiction="EU", authority="EU", source_title="LVD"),
                                grounded_quotes=[GroundedQuote(section="Article 1", quote="1 000 V")])
    divider_ir.regulatory.requirements.append(req)
    h = divider_ir.content_hash()
    req.status, req.source_status, req.provenance.verification_status, req.provenance.source_document = S.FAIL, "ok", S.PASS, "/x/y.html"
    req.grounded_quotes[0].found, req.grounded_quotes[0].page, req.grounded_quotes[0].context = True, 3, "... [1 000 V] ..."
    assert divider_ir.content_hash() == h  # what a run found is state about the design
    req.grounded_quotes[0].quote = "1 500 V"
    assert divider_ir.content_hash() != h  # the claim itself is design provenance
    req.provenance.content_hash = "sha256:" + "c" * 64
    assert divider_ir.content_hash() != h  # which document version was used stays in the hash
    h = divider_ir.content_hash()
    divider_ir.parameters["created_at"] = user_requirement(999.0)
    assert divider_ir.content_hash() != h  # a parameter that happens to be called created_at is design content
    assert "created_at" not in json.dumps(divider_ir.design_dict()["components"])  # Provenance.created_at is still out
    assert "document_path" not in json.dumps(divider_ir.design_dict()) and "library_path" not in json.dumps(divider_ir.design_dict())
