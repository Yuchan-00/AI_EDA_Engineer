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

Round 2 (the LLM and parts/regulatory reviewers):

18. A truncated part number (``LM2596S-5``) was grounded on ``LM2596S-5.0/NOPB``: token boundaries let a match stop
    inside a dot / dash-joined code.
19. ``DC 60 V`` / ``AC220V`` / ``60 V (DC)`` were not read as a stated current kind, so the mains answer decided.
20. A typed ``--answer key=value`` was dropped when an extraction-derived requirement of that key existed.
21. Only the first catalog row of an MPN was consulted: a second row of the right brand was reported as a mismatch.
22. The recorded page was enforced only for the spelling ``page N``; ``Page 3`` / no section searched everywhere.
23. Two ``req.application`` entered when the model returned both an application object and an item keyed application.
24. Cross-kind duplicates with one value merged into whichever the model listed first, dropping the grounded explicit.
25. A quote without the AC/DC suffix (``12V`` in ``12V DC``) was demoted as "part of a larger quantity".
26. A cache entry without ``model`` (or with a list for ``decisions``) crashed the agent instead of re-extracting.
27. The extraction fingerprint hashed only the system prompt, not the prompt as sent nor the quantity parser version.
28. ``DE`` was grounded on ``DE-9`` (a connector): the code token could be part of an identifier.
29. A ``NaN`` number from the model was accepted into an assumption value.
30. ``네 맞습니다`` / ``yes, correct`` / ``ok thanks`` were corrections (a billed re-extraction); a repeated correction
    was silently ignored.

Round 3 (the review/repair/orchestrator/CLI and compiler reviewers):

31. A compiler refusal (FAIL) lived only in the stage table: not in ``ir.validation``, RELEASE or the exit code.
32. A BOM/CPL ``CompileError`` in MANUFACTURING_OUTPUTS crashed the pipeline instead of being a FAIL stage.
33. The reviewer passed through untooled or hash-less ``kicad.erc`` / ``kicad.drc`` / ``mfg.capability`` results
    as tool-backed review PASSes.
34. ``review.requirements_vs_ir`` was a PASS for a design with nothing to trace.
35. ``apply_proposals`` was not atomic: a later invalid proposal left the earlier ones applied.
36. ``ai-eda new`` stored a cwd-relative workdir, so a run from another directory wrote artifacts elsewhere.
37. The repair loop's oscillation fingerprint ignored the repair category: a re-run that turned a stale report into a
    human finding was reported as an oscillation.
38. A finding whose action failed once was never retried, even after the regeneration it depended on.
39. ``RepairOutcome.as_validation_result`` was PASS with a failed action in the log.
40. Two ``--datasheet-url`` for one REF with different URLs were silently last-wins.
41. The CPL wrote ``Rotation`` with ``%g`` (6 significant digits, exponents, ``-0``), so the reviewer failed
    ``pcb_vs_cpl`` as a compiler defect for a rotation with more digits.
42. ``natural_ref_key`` raised ``TypeError`` for a symbol mixing numeric and alphabetic pin numbers.
43. NaN / infinite coordinates were written into the board and CPL (``nan``, ``nanmm``) or crashed with ``ValueError``.
44. Repeated pad numbers in a footprint got duplicate ``(uuid ...)``; repeated pin numbers in a symbol were silently
    left unwired instead of refused.
45. ``SchematicCompiler`` silently fell back to the installed libraries when ``tools['kicad_library']`` was not a
    ``KicadLibrary`` (the PCB compiler refused the same input).

Round 4 (the SPICE / calculator / portability reviewer):

46. A ``model_card`` was a verbatim pass-through: ``.inc`` (ngspice's prefix match of ``.include``), ``.opt``, ``.ic``,
    ``.nodeset``, ``.global``, other analysis cards and bare element lines reached ngspice unreported; ``validate_deck``
    matched forbidden cards by whole word only.
47. ``ngspice_reads`` vouched for a bare ``a`` / ``A`` tail as a trailing letter (2.0) while ngspice-42 reads it as atto.
48. The batch ``NgspiceRunner`` could never PASS in ``run_spice_for``: it reported the hash of its analysis-card copy,
    and Debian's solver banner on stderr counted as an error.
49. A pin in no net was accepted when listed in ``ignored_pins`` (which is for *connected* pins the element ignores).
50. A non-ASCII ``project.id`` failed the SPICE stage with a message about node names.
51. R / C / L values of 0 or negative compiled; ngspice simulates R=0 as ~1 mΩ without a word.
52. A vector with a NaN / infinite sample in the middle was judged (``builtins.max`` skips a NaN that is not first).
53. A non-finite ``tol_abs`` / ``nominal`` was accepted by the compiler and judged (``tol_abs=inf`` passed anything).
54. The rawfile ``Command:`` line is absent on ngspice-42, not empty (docs and parser note).
55. On Linux the system ``libngspice.so.0`` was found only through ``NGSPICE_DLL``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eda.agents import AgentContext, RequirementAgent
from ai_eda.agents.base import IRProposal
from ai_eda.agents.regulatory import ACCEPT_REGS_KEY
from ai_eda.cli import main as cli_main, run_exit_code
from ai_eda.compilers import BOMCompiler, CompileContext, CPLCompiler, PCBCompiler, SchematicCompiler
from ai_eda.compilers.ids import pad_uuid
from ai_eda.compilers.pins import pad_pin_types
from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError, IRSchemaError, ToolUnavailableError
from ai_eda.repair import RepairAction, RepairLoop, RepairStrategy
from ai_eda.repair.loop import RepairOutcome
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.review.reviewer import ReviewReport
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import SExprError
from ai_eda.workflow.session import SessionError, parse_key_urls
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitDomain,
    Expectation,
    Reduce,
    SpiceBinding,
    SpiceDevice,
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
from ai_eda.ir.regulatory import Applicability, GroundedQuote, RegulatoryProvenance, RegulatoryRequirement
from ai_eda.llm.client import LLMError, LLMMessage
from ai_eda.llm.extraction import (
    CONFIRM_KEY,
    REQUIREMENT_EXTRACTION_SYSTEM,
    RequirementExtraction,
    cache_entry_staleness,
    extraction_fingerprint,
    find_quote,
    ground_extraction,
    is_confirmation,
    is_correction,
    is_grounded_explicit,
    jurisdiction_named_in,
)
from ai_eda.llm.openrouter import REDACTED, OpenRouterClient, check_base_url
from ai_eda.parts import CatalogSource
from ai_eda.parts.existence import _catalog_check, find_mpn
from ai_eda.parts.identity import mpn_grounding
from ai_eda.regulatory.applicability import MAINS_KEY, _current_kind, evaluate
from ai_eda.security import ApprovalGate
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy
from ai_eda.tools.sources.policy import host_of, normalise_url, port_of
from ai_eda.tools.spice.ngspice_shared import find_codemodel_dir, find_ngspice_dll, find_system_ngspice, validate_deck
from ai_eda.tools.spice import NgspiceRunner, SpiceAnalysis, SpiceResult
from ai_eda.tools.spice.stage import judge, reduce_expectation, run_spice_for
from ai_eda.tools.calc import ngspice_reads
from ai_eda.compilers.spice import build, build_report
from tests.test_spice_netlist import DIODE_CARD, OPAMP_CARD, USER, component, divider_ir as spice_divider_ir, net, op_setup, resistor
from ai_eda.workflow import Orchestrator, Stage
from tests.conftest import AUTH, DS, make_component
from tests.fake_openrouter import FakeOpenRouter
from tests.fake_sources import FakeSources
from tests.pdf_fixture import build_pdf
from tests.test_applicability import VOLT, _req as _volt_req, _reqs
from tests.test_archive import make_archive, online_policy
from tests.test_parts_existence import make_part
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


# =========================================================================== round 2


def _offline_archive(root: Path) -> DocumentArchive:
    return DocumentArchive(root, NetworkPolicy(approved=False, gate=ApprovalGate()))


def _pdf_doc(tmp_path: Path, pages: list[list[str]], name: str = "ds.pdf"):
    p = tmp_path / name
    p.write_bytes(build_pdf(pages))
    archive = _offline_archive(tmp_path / "sources")
    return archive, archive.add_file(p, title=name, retrieved_at="2026-09-23")


# --------------------------------------------------------------------------- 18: a part number is a whole identifier


def test_18_a_truncated_part_number_is_not_found_in_the_longer_code(tmp_path: Path):
    text = "ORDERING INFORMATION  LM2596S-5.0/NOPB  TO-263  LM2596T-ADJ  TO-220"
    assert find_quote("LM2596S-5", text, identifier=True) is None and find_quote("LM2596S-5", text) is not None  # the request rule is unchanged
    assert find_quote("LM2596S-5.0", text, identifier=True) == (22, 33)  # a slash separates an option suffix
    assert find_quote("LM2596S-5.0/NOPB", text, identifier=True) == (22, 38)
    assert find_quote("2596S-5.0", text, identifier=True) is None  # nor may it start inside the code
    archive, doc = _pdf_doc(tmp_path, [[text]])
    assert find_mpn(doc, "lm2596s-5") == [] and [h.page for h in find_mpn(doc, "LM2596S-5.0")] == [1]
    part = make_part(mpn="LM2596S-5")
    part.mpn = authoritative("LM2596S-5", doc.source_ref(title="ds", section="page 1"))
    g = mpn_grounding(part, archive)
    assert not g.grounded and g.label == "authoritative, not in text"


# --------------------------------------------------------------------------- 19: AC/DC before the number


def test_19_ac_dc_before_the_number_or_in_parentheses_is_a_stated_kind():
    assert _current_kind("DC 60 V") == "dc" and _current_kind("AC220V") == "ac" and _current_kind("60 V (DC)") == "dc" and _current_kind("60VDC") == "dc"
    assert _current_kind("AC adapter, 12 V") is None and _current_kind("12 V") is None  # the bare word in prose is still not a kind
    ev = evaluate(VOLT, {MAINS_KEY: "yes"}, _reqs(_volt_req(60.0, text="DC 60 V input")))
    assert ev.applicability is Applicability.UNDECIDED and [m.key for m in ev.missing] == [MAINS_KEY]  # contradiction, not silently AC
    ev = evaluate(VOLT, {MAINS_KEY: "no"}, _reqs(_volt_req(60.0, text="DC 60 V input")))
    assert ev.applicability is Applicability.NOT_APPLICABLE


# --------------------------------------------------------------------------- 20: a typed answer wins over the extraction


def test_20_a_later_typed_answer_replaces_an_extraction_item_of_the_same_key(tmp_path: Path):
    svc, client = _req_service([{"structured": REQ_CANNED, "usage": USAGE}])
    ir = _req_ir(tmp_path)
    _req_run(ir, svc, tmp_path)
    assert ir.requirements.get("output_voltage").value.provenance.kind is ProvenanceKind.LLM_GENERATED
    state = _req_run(ir, svc, tmp_path, answers={"output_voltage": "24V"})
    r = ir.requirements.get("output_voltage")
    assert r.value.value == "24V" and r.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert [x.key for x in ir.requirements.requirements].count("output_voltage") == 1 and len(client.calls) == 1
    assert "output_voltage: the user's answer takes precedence" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    _req_run(ir, svc, tmp_path, answers={CONFIRM_KEY: "yes"})
    assert ir.requirements.get("output_voltage").value.value == "24V"  # the confirmation keeps the typed answer
    # the no-LLM checklist path replaces too
    RequirementAgent().run(ir, AgentContext(workdir=tmp_path, answers={"efficiency": "95%"}))
    state = Orchestrator(AgentContext(workdir=tmp_path, answers={"efficiency": "95%"})).run(ir, stop_after=Stage.REQUIREMENT_ANALYSIS)
    eff = ir.requirements.get("efficiency")
    assert eff.value.value == "95%" and eff.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert [x.key for x in ir.requirements.requirements].count("efficiency") == 1
    # an answer the user typed earlier is kept (a typed answer is not an extraction)
    Orchestrator(AgentContext(workdir=tmp_path, answers={"efficiency": "80%"})).run(ir, stop_after=Stage.REQUIREMENT_ANALYSIS)
    assert ir.requirements.get("efficiency").value.value == "95%"


# --------------------------------------------------------------------------- 21: every catalog row of an MPN counts


def test_21_the_catalog_row_that_agrees_with_the_ir_backs_the_sourcing(tmp_path: Path):
    csv = tmp_path / "cat.csv"
    csv.write_text("mpn,manufacturer,package,supplier_part_number,stock\nLM2596S-5.0/NOPB,Texas Instruments,TO-263,C1,5\nLM2596S-5.0/NOPB,ON Semi,TO-263,C2,9\n", encoding="utf-8")
    cat = CatalogSource.load(csv, "2026-09-23", "JLCPCB export", supplier="JLCPCB")
    assert cat.duplicates == {"lm2596s-5.0/nopb": 2} and [r.line for r in cat.rows_for("LM2596S-5.0/NOPB")] == [2, 3]
    part = make_part(mpn="LM2596S-5.0/NOPB")
    part.manufacturer = user_requirement("ON Semi")
    row, check = _catalog_check("LM2596S-5.0/NOPB", cat, part)
    assert check.status is S.PASS and row.line == 3 and row.supplier_part_number == "C2"
    part.manufacturer = user_requirement("Nexperia")
    row, check = _catalog_check("LM2596S-5.0/NOPB", cat, part)
    assert check.status is S.NOT_VERIFIED and "2 row(s)" in check.message and "row 2:" in check.message and "row 3:" in check.message


# --------------------------------------------------------------------------- 22: the recorded page, whatever its spelling


def test_22_the_recorded_page_is_enforced_and_a_missing_page_is_not_grounded(tmp_path: Path):
    archive, doc = _pdf_doc(tmp_path, [["LM317 adjustable regulator"], ["Electrical characteristics"], ["Ordering: see page 1"]])
    part = make_part(mpn="LM317")

    def grounding(section):
        part.mpn = authoritative("LM317", doc.source_ref(title="ds", section=section))
        return mpn_grounding(part, archive)

    assert grounding("page 1").grounded and grounding("Page 1").grounded and grounding("p. 1").grounded
    assert grounding("Page 3").label == "authoritative, not in text"
    for section in (None, "", "ordering table"):
        g = grounding(section)
        assert not g.grounded and g.label == "authoritative, no page recorded" and "re-run the existence check" in g.reason


# --------------------------------------------------------------------------- 23: one req.application


def test_23_an_explicit_application_item_and_the_application_object_make_one_requirement(tmp_path: Path):
    canned = json.loads(json.dumps(REQ_CANNED))
    canned["requirements"].append({"key": "application", "text": "A converter", "kind": "explicit", "category": "application", "quote": "변환하는 회로", "value": None, "rationale": None})
    svc, _ = _req_service([{"structured": canned, "usage": USAGE}])
    ir = _req_ir(tmp_path)
    _req_run(ir, svc, tmp_path)
    apps = [r for r in ir.requirements.requirements if r.key == "application"]
    assert len(apps) == 1 and len({r.id for r in ir.requirements.requirements}) == len(ir.requirements.requirements)
    assert apps[0].id == "req.application" and apps[0].text == "application: 변환하는 회로"  # the user's words, not the model's summary


# --------------------------------------------------------------------------- 24: the grounded explicit item survives a same-value duplicate


def test_24_same_value_duplicates_keep_the_grounded_explicit_whatever_the_order():
    raw = "12V 입력을 5V 2A로 변환"
    canned = {
        "requirements": [
            {"key": "output_voltage", "text": "Output 5 V", "kind": "implicit", "category": "electrical", "quote": None,
             "value": {"quote": "5V", "number": 5, "unit": "V", "number_high": None}, "rationale": "a 5 V rail is implied"},
            {"key": "output_voltage", "text": "Output voltage is 5 V", "kind": "explicit", "category": "electrical", "quote": "5V 2A로",
             "value": {"quote": "5V", "number": 5, "unit": "V", "number_high": None}, "rationale": None},
        ],
        "questions": [], "conflicts": [], "assumptions": [], "application": None, "jurisdictions": [],
    }
    grounded = ground_extraction(raw, RequirementExtraction.model_validate(canned), "m")
    survivors = [r for r in grounded.requirements if r.key == "output_voltage"]
    assert len(survivors) == 1 and survivors[0].kind is RequirementKind.EXPLICIT and is_grounded_explicit(survivors[0]) and survivors[0].id == "req.output_voltage"
    assert any("merged into the explicit item" in reason for _, reason in grounded.dropped)


# --------------------------------------------------------------------------- 25: a quote may leave the AC/DC word out


def test_25_a_quote_without_the_ac_dc_suffix_still_grounds():
    for raw, quote in (("12V DC 입력, 5V 출력", "12V"), ("입력 12 V DC", "12 V"), ("입력 230 V AC 50 Hz", "230 V"), ("입력 230 V (AC)", "230 V")):
        canned = {"requirements": [{"key": "input_voltage", "text": "x", "kind": "explicit", "category": "electrical", "quote": quote,
                                    "value": {"quote": quote, "number": float(quote.split()[0].rstrip("V")), "unit": "V", "number_high": None}, "rationale": None}],
                  "questions": [], "conflicts": [], "assumptions": [], "application": None, "jurisdictions": []}
        grounded = ground_extraction(raw, RequirementExtraction.model_validate(canned), "m")
        r = grounded.requirements[0]
        assert is_grounded_explicit(r) and r.value.provenance.kind is ProvenanceKind.LLM_GENERATED, (raw, quote, grounded.dropped, r.value.provenance.note)


# --------------------------------------------------------------------------- 26: a damaged cache entry is a miss, not a crash


def test_26_a_cache_entry_without_its_model_or_with_bad_decisions_is_re_extracted(tmp_path: Path):
    svc, client = _req_service([{"structured": REQ_CANNED, "usage": USAGE}, {"structured": REQ_CANNED, "usage": USAGE}, {"structured": REQ_CANNED, "usage": USAGE}])
    ir = _req_ir(tmp_path)
    _req_run(ir, svc, tmp_path)
    key = next(iter(ir.requirements.extraction_cache))
    assert cache_entry_staleness({**ir.requirements.extraction_cache[key], "model": None}) and cache_entry_staleness({**ir.requirements.extraction_cache[key], "decisions": ["accept"]})
    del ir.requirements.extraction_cache[key]["model"]
    state = _req_run(ir, svc, tmp_path)
    assert len(client.calls) == 2 and "does not record the model" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    ir.requirements.extraction_cache[key]["decisions"] = ["accept"]
    state = _req_run(ir, svc, tmp_path)
    assert len(client.calls) == 3 and "decisions are not an object" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message


# --------------------------------------------------------------------------- 27: the fingerprint covers what the model was asked


def test_27_the_fingerprint_covers_the_prompt_as_sent_and_the_quantity_rules(monkeypatch: pytest.MonkeyPatch):
    fp = extraction_fingerprint()
    assert set(fp) == {"extraction_version", "prompt_hash", "schema_hash", "quantity_version"}
    assert fp["prompt_hash"] != "sha256:" + hashlib.sha256(REQUIREMENT_EXTRACTION_SYSTEM.encode("utf-8")).hexdigest()
    import ai_eda.llm.extraction as ext
    monkeypatch.setattr(ext, "JSON_ONLY_INSTRUCTION", ext.JSON_ONLY_INSTRUCTION + " (changed)")
    assert extraction_fingerprint()["prompt_hash"] != fp["prompt_hash"]
    monkeypatch.setattr(ext, "QUANTITY_VERSION", "9.9")
    assert extraction_fingerprint()["quantity_version"] == "9.9"


# --------------------------------------------------------------------------- 28: a jurisdiction code is a whole token


def test_28_a_code_inside_an_identifier_does_not_name_a_jurisdiction():
    assert not jurisdiction_named_in("DE", "DE-9 커넥터 사용") and not jurisdiction_named_in("US", "USB-C") and not jurisdiction_named_in("IN", "IN-1 pin")
    assert jurisdiction_named_in("DE", "DE에서 판매") and jurisdiction_named_in("EU", "EU에서 판매") and jurisdiction_named_in("DE", "Germany")
    canned = {"requirements": [], "questions": [], "conflicts": [], "assumptions": [], "application": None, "jurisdictions": [{"code": "DE", "quote": "DE-9"}]}
    grounded = ground_extraction("DE-9 커넥터로 연결하는 12V 장치", RequirementExtraction.model_validate(canned), "m")
    assert grounded.jurisdictions == []


# --------------------------------------------------------------------------- 29: no NaN


def test_29_a_nan_number_is_rejected_by_the_schema():
    canned = json.loads(json.dumps(REQ_CANNED))
    canned["requirements"][0]["value"]["number"] = float("nan")
    with pytest.raises(ValidationError):
        RequirementExtraction.model_validate(canned)
    canned["requirements"][0]["value"]["number"] = float("inf")
    with pytest.raises(ValidationError):
        RequirementExtraction.model_validate(canned)


# --------------------------------------------------------------------------- 30: confirmations made of confirmation words


def test_30_confirmation_words_confirm_and_a_repeated_correction_is_explained(tmp_path: Path):
    assert is_confirmation("네 맞습니다") and is_confirmation("yes, correct") and is_confirmation("ok thanks") and is_confirmation("네, 확인합니다.")
    assert not is_confirmation("yes, but change X") and not is_confirmation("sure") and not is_confirmation("thanks")
    assert is_correction("yes, but change the input to 24V") and not is_correction("네 맞습니다")
    svc, client = _req_service([{"structured": REQ_CANNED, "usage": USAGE}, {"structured": REQ_CANNED, "usage": USAGE}])
    ir = _req_ir(tmp_path)
    _req_run(ir, svc, tmp_path)
    _req_run(ir, svc, tmp_path, answers={CONFIRM_KEY: "네 맞습니다"})
    assert len(client.calls) == 1 and ir.requirements.get("input_voltage").value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    _req_run(ir, svc, tmp_path, answers={CONFIRM_KEY: "the output is 3.3V, not 5V"})
    assert len(client.calls) == 2 and ir.requirements.corrections == ["the output is 3.3V, not 5V"]
    state = _req_run(ir, svc, tmp_path, answers={CONFIRM_KEY: "the output is 3.3V, not 5V"})
    assert len(client.calls) == 2 and "already part of the request" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message


# =========================================================================== round 3


def _areas(ir: CircuitIR, tmp_path: Path) -> dict[str, ValidationResult]:
    return {r.check_id: r for r in IndependentReviewer(tools={}).review(ir, tmp_path).results}


# --------------------------------------------------------------------------- 31: a compiler refusal is a recorded verdict


def test_31_a_compile_refusal_is_recorded_and_reaches_release_and_the_exit_code(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[0].symbol.verified = False  # the schematic compiler refuses an unverified symbol
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=ANSWERS)).run(divider_ir)
    assert state.outcome(Stage.SCHEMATIC).status is S.FAIL
    rec = divider_ir.validation.latest("compile.kicad_sch")
    assert rec is not None and rec.status is S.FAIL and rec.is_tool_backed and rec.ir_hash == divider_ir.content_hash() and rec.details["repair"] == "human"
    assert state.outcome(Stage.RELEASE).status is S.FAIL and "compile.kicad_sch" in state.outcome(Stage.RELEASE).message
    assert run_exit_code(state) == 1
    # a compile that succeeds records a tool-backed PASS on the artifact it wrote
    ok = divider_ir.validation.latest("compile.bom")
    assert ok is not None and ok.status is S.PASS and ok.artifact_hash == divider_ir.artifacts[ArtifactKind.BOM].content_hash


# --------------------------------------------------------------------------- 32: a BOM refusal is a FAIL stage, not a crash


def test_32_a_bom_cell_refusal_is_a_fail_stage_not_an_exception(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[0].mpn = authoritative("=CMD()", DS)
    state = Orchestrator(AgentContext(workdir=tmp_path, answers=ANSWERS)).run(divider_ir)
    out = state.outcome(Stage.MANUFACTURING_OUTPUTS)
    assert out.status is S.FAIL and "refusing to write a cell" in out.message and "bom compile refused" in out.message
    assert divider_ir.validation.latest("compile.bom").status is S.FAIL and ArtifactKind.BOM not in divider_ir.artifacts
    assert state.outcome(Stage.RELEASE) is not None  # the pipeline ran to the end


# --------------------------------------------------------------------------- 33: opinions do not become review PASSes


def test_33_untooled_or_hashless_tool_results_are_not_review_evidence(divider_ir: CircuitIR, tmp_path: Path):
    sch = tmp_path / "t.kicad_sch"
    sch.write_text("(kicad_sch)", encoding="utf-8")
    pcb = tmp_path / "t.kicad_pcb"
    pcb.write_text("(kicad_pcb)", encoding="utf-8")
    h = divider_ir.content_hash()
    divider_ir.artifacts[ArtifactKind.SCHEMATIC] = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path=str(sch), content_hash=SourceRef.hash_bytes(sch.read_bytes()), generated_from_ir_hash=h)
    divider_ir.artifacts[ArtifactKind.PCB] = ArtifactRef(kind=ArtifactKind.PCB, path=str(pcb), content_hash=SourceRef.hash_bytes(pcb.read_bytes()), generated_from_ir_hash=h)
    sch_hash, pcb_hash = divider_ir.artifacts[ArtifactKind.SCHEMATIC].content_hash, divider_ir.artifacts[ArtifactKind.PCB].content_hash
    divider_ir.validation.extend([
        ValidationResult(check_id="kicad.erc", status=S.PASS, message="I say it passes", artifact_hash=sch_hash),  # no tool
        ValidationResult(check_id="kicad.drc", status=S.PASS, artifact_hash=pcb_hash, details={"schematic_parity_checked": True, "schematic_hash": sch_hash, "schematic_parity": []}),  # no tool
        ValidationResult(check_id="mfg.capability", status=S.PASS, message="trust me"),  # no tool
    ])
    areas = _areas(divider_ir, tmp_path)
    for area in (ReviewArea.ERC, ReviewArea.DRC, ReviewArea.SCHEMATIC_VS_PCB, ReviewArea.MANUFACTURING_CAPABILITIES):
        assert areas[area].status is S.NOT_VERIFIED and ("not tool-backed" in areas[area].message or "no tool" in areas[area].message), area
    # a tool-backed result without an artifact hash is not evidence about any file either
    divider_ir.validation.add(ValidationResult(check_id="kicad.erc", status=S.PASS, tool="kicad-cli", tool_version="10.0.6"))
    assert _areas(divider_ir, tmp_path)[ReviewArea.ERC].status is S.NOT_VERIFIED
    # the real thing still passes through
    divider_ir.validation.add(ValidationResult(check_id="kicad.erc", status=S.PASS, tool="kicad-cli", tool_version="10.0.6", artifact_hash=sch_hash))
    assert _areas(divider_ir, tmp_path)[ReviewArea.ERC].status is S.PASS


# --------------------------------------------------------------------------- 34: nothing to trace is not a traced design


def test_34_requirements_vs_ir_is_not_a_vacuous_pass(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="e", name="e", workdir=str(tmp_path)))
    ir.requirements.requirements.append(Requirement(id="req.application", key="application", text="x", kind=RequirementKind.EXPLICIT, category="application", value=user_requirement("x")))
    r = _areas(ir, tmp_path)[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.NOT_VERIFIED and "no design-level requirement" in r.message
    ir.requirements.requirements.append(Requirement(id="req.bus", key="bus", text="I2C bus", kind=RequirementKind.EXPLICIT, category="electrical"))
    ir.nets.append(Net(name="SDA", pins=[], provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="t"), serves_requirements=["req.bus"]))
    r = _areas(ir, tmp_path)[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.PASS and r.details["traced"] == ["req.bus"]


# --------------------------------------------------------------------------- 35: proposals apply all or nothing


def test_35_an_invalid_proposal_leaves_the_earlier_ones_unapplied(divider_ir: CircuitIR):
    before, n = divider_ir.content_hash(), len(divider_ir.components)
    with pytest.raises(ValueError, match="not a valid"):
        Orchestrator.apply_proposals(divider_ir, [
            IRProposal(description="ok", target="components", operation="append", payload=make_component("R3", "1k")),
            IRProposal(description="bad", target="components", operation="append", payload="not a component"),
        ])
    assert len(divider_ir.components) == n and divider_ir.content_hash() == before


# --------------------------------------------------------------------------- 36: a new project's workdir is absolute


def test_36_new_records_an_absolute_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert cli_main(["new", "demo", "--dir", "rel/demo", "--request", "x"]) == 0
    ir = CircuitIR.load(tmp_path / "rel" / "demo" / "ir.json")
    assert Path(ir.project.workdir).is_absolute() and Path(ir.project.workdir) == (tmp_path / "rel" / "demo").resolve()


# --------------------------------------------------------------------------- 37 / 38 / 39: the repair loop's bookkeeping


class _Scripted:
    """A reviewer that returns the scripted reports in order (the last one repeats)."""

    def __init__(self, ir_hash: str, *reports: list[ValidationResult]) -> None:
        self.reports = [ReviewReport(ir_hash=ir_hash, results=list(r)) for r in reports]
        self.calls = 0

    def review(self, ir: CircuitIR, workdir: Path) -> ReviewReport:
        self.calls += 1
        return self.reports[min(self.calls - 1, len(self.reports) - 1)]


class _Rerun(RepairStrategy):
    id = "test.rerun"

    def __init__(self, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.attempts = 0

    def can_repair(self, finding):
        return finding.details.get("repair") == "rerun_tool"

    def apply(self, ir, finding, workdir, tools):
        self.attempts += 1
        failed = self.fail_first and self.attempts == 1
        return RepairAction(strategy=self.id, finding_check_id=finding.check_id, description="re-run", ir_hash_before=ir.content_hash(),
                            succeeded=not failed, error="stale" if failed else None)


class _Regen(RepairStrategy):
    id = "test.regen"

    def can_repair(self, finding):
        return finding.details.get("repair") == "regenerate"

    def apply(self, ir, finding, workdir, tools):
        return RepairAction(strategy=self.id, finding_check_id=finding.check_id, description="regenerate", ir_hash_before=ir.content_hash(), succeeded=True)


def test_37_a_rerun_that_reveals_a_human_finding_is_not_an_oscillation(divider_ir: CircuitIR, tmp_path: Path):
    h = divider_ir.content_hash()
    stale = ValidationResult(check_id=ReviewArea.ERC, status=S.FAIL, message="kicad.erc report is stale", details={"repair": "rerun_tool", "tool_check": "kicad.erc"})
    real = ValidationResult(check_id=ReviewArea.ERC, status=S.FAIL, message="kicad.erc: 1 error(s)", details={"repair": "human", "error_types": ["pin_not_connected"]})
    outcome = RepairLoop(tools={}, strategies=[_Rerun()], reviewer=_Scripted(h, [stale], [real])).run(divider_ir, tmp_path)
    assert outcome.iterations == 2 and outcome.stopped_reason == "no repairable failures remain"
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.ERC] and "human" in outcome.unresolved[0].message
    assert outcome.as_validation_result(h, "t").status is S.FAIL


def test_38_a_failed_action_is_retried_after_the_loop_made_progress(divider_ir: CircuitIR, tmp_path: Path):
    h = divider_ir.content_hash()
    rerun = ValidationResult(check_id=ReviewArea.ERC, status=S.FAIL, message="stale report", details={"repair": "rerun_tool", "tool_check": "kicad.erc"})
    regen = ValidationResult(check_id=ReviewArea.IR_VS_SCHEMATIC, status=S.FAIL, message="stale artifact", details={"repair": "regenerate", "artifact": "kicad_sch"})
    strategy = _Rerun(fail_first=True)
    outcome = RepairLoop(tools={}, strategies=[strategy, _Regen()], reviewer=_Scripted(h, [rerun, regen], [rerun], [])).run(divider_ir, tmp_path)
    assert strategy.attempts == 2 and outcome.stopped_reason == "all failures resolved" and outcome.unresolved == []
    assert [a.succeeded for a in outcome.actions] == [False, True, True]
    assert outcome.as_validation_result(h, "t").status is S.NOT_VERIFIED  # an attempt failed on the way: not a clean PASS


def test_39_an_outcome_with_a_failed_action_is_not_pass():
    report = ReviewReport(ir_hash="sha256:h")
    failed = RepairAction(strategy="s", finding_check_id="c", description="d", ir_hash_before="sha256:h", ir_hash_after="sha256:h", succeeded=False, error="boom")
    assert RepairOutcome(actions=[failed], unresolved=[], final_review=report).as_validation_result("sha256:h", "t").status is S.NOT_VERIFIED
    good = failed.model_copy(update={"succeeded": True, "error": None})
    assert RepairOutcome(actions=[good], unresolved=[], final_review=report).as_validation_result("sha256:h", "t").status is S.PASS


# --------------------------------------------------------------------------- 40: one REF, one URL


def test_40_two_different_urls_for_one_key_are_a_usage_error():
    assert parse_key_urls(["R1=https://a.example/x.pdf", "R1=https://a.example/x.pdf"], "--datasheet-url") == {"R1": "https://a.example/x.pdf"}
    with pytest.raises(SessionError, match="names 'R1' twice with different URLs"):
        parse_key_urls(["R1=https://a.example/x.pdf", "R1=https://b.example/y.pdf"], "--datasheet-url")


# --------------------------------------------------------------------------- 41: the CPL rotation is written like the board's


def test_41_cpl_rotation_is_fixed_format(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.pcb = PCBDesign(placements=[
        Placement(component_ref="R1", x_mm=14.0, y_mm=6.0, rotation_deg=123.4567, provenance=AUTH),
        Placement(component_ref="R2", x_mm=1.0, y_mm=2.0, rotation_deg=-0.0, provenance=AUTH),
    ])
    art = CPLCompiler().compile(divider_ir, CompileContext(workdir=tmp_path, tools={}))
    rows = Path(art.path).read_text(encoding="utf-8").splitlines()[1:]
    assert rows[0].split(",")[3] == "123.4567" == sexpr.fmt_num(123.4567) and rows[1].split(",")[3] == "0"
    assert not any("e" in r.split(",")[3] for r in rows)


# --------------------------------------------------------------------------- 42: mixed pin numbers sort


def test_42_mixed_numeric_and_alphabetic_pin_numbers_sort():
    assert sorted(["10", "2", "CD", "A1", "SH", "1"], key=natural_ref_key) == ["1", "2", "10", "A1", "CD", "SH"]
    assert natural_ref_key("R2") < natural_ref_key("R10") and natural_ref_key("J1") < natural_ref_key("R1")


# --------------------------------------------------------------------------- 43: non-finite numbers are refused


def test_43_non_finite_coordinates_are_compile_errors_not_files(divider_ir: CircuitIR, tmp_path: Path):
    with pytest.raises(SExprError, match="not a finite number"):
        sexpr.fmt_num(float("nan"))
    divider_ir.pcb = PCBDesign(placements=[Placement(component_ref="R1", x_mm=float("nan"), y_mm=6.0, provenance=AUTH)])
    with pytest.raises(CompileError, match=r"placement R1\.x_mm is nan"):
        CPLCompiler().compile(divider_ir, CompileContext(workdir=tmp_path, tools={}))
    with pytest.raises(CompileError, match=r"ir\.pcb\.placements\[0\]\.x_mm is nan"):
        PCBCompiler().compile(divider_ir, CompileContext(workdir=tmp_path, tools={}))
    divider_ir.pcb = PCBDesign(placements=[Placement(component_ref="R1", x_mm=1.0, y_mm=6.0, rotation_deg=float("inf"), provenance=AUTH)])
    with pytest.raises(CompileError, match="rotation_deg is inf"):
        PCBCompiler().compile(divider_ir, CompileContext(workdir=tmp_path, tools={}))


# --------------------------------------------------------------------------- 44: repeated pad / pin numbers


def test_44_repeated_pad_numbers_get_distinct_ids_and_repeated_pin_numbers_are_refused(divider_ir: CircuitIR):
    assert pad_uuid("p", "J1", "3") != pad_uuid("p", "J1", "3#2") and pad_uuid("p", "J1", "3") == pad_uuid("p", "J1", "3")  # the first keeps its id
    pin = lambda n: SymbolPin(number=n, name="~", electrical_type="passive", x=0.0, y=0.0, angle=0.0, length=2.54, unit=0, hidden=False)  # noqa: E731
    symbol = SymbolDef(lib_id="Test:S", name="S", node=[], pins=[pin("1"), pin("2"), pin("1")], units=[1], is_power=False, properties={})
    with pytest.raises(CompileError, match="repeats pin number"):
        pad_pin_types(divider_ir.components[0], symbol)


# --------------------------------------------------------------------------- 45: a wrong library object is refused, not replaced


def test_45_schematic_compiler_refuses_a_wrong_library_object(divider_ir: CircuitIR, tmp_path: Path):
    with pytest.raises(CompileError, match="not a KicadLibrary"):
        SchematicCompiler().compile(divider_ir, CompileContext(workdir=tmp_path, tools={"kicad_library": "not a library"}))


# =========================================================================== round 4


import shutil  # noqa: E402


# --------------------------------------------------------------------------- 46: a model card is .model / .subckt text only


@pytest.mark.parametrize("card", [
    ".model dx d(is=1e-14)\nR99 VOUT 0 1",  # an element outside a subcircuit adds a part the IR does not have
    ".model dx d(is=1e-14)\n.inc /tmp/extra.cir",  # ngspice reads .inc as .include
    ".model dx d(is=1e-14)\n.INCLUDE /tmp/extra.cir",
    ".model dx d(is=1e-14)\n.opt temp=100",
    ".model dx d(is=1e-14)\n.ic v(VOUT)=100",
    ".model dx d(is=1e-14)\n.nodeset v(VOUT)=1",
    ".model dx d(is=1e-14)\n.global VOUT",
    ".model dx d(is=1e-14)\n.tf v(VOUT) VVIN",
    ".model dx d(is=1e-14)\n.param x=1",  # .param is the body of a subcircuit, not a top-level card
    ".subckt S a b\nR1 a b 1k",  # unclosed
])
def test_46_model_cards_that_are_not_models_are_refused(card: str):
    ir = spice_divider_ir()
    ir.components.append(component("D1", "1N4148", 2, SpiceBinding(device=SpiceDevice.D, model_name="dx", model_card=authoritative(card, DS), pin_order=["1", "2"], provenance=AUTH)))
    ir.net("VOUT").pins.append(PinRef(component_ref="D1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="D1", pin_number="2"))
    with pytest.raises(CompileError, match="model_card"):
        build(ir)


def test_46_a_subcircuit_with_elements_and_params_inside_is_a_model_card():
    ir = spice_divider_ir()
    card = ".subckt IDEALOPAMP inp inn out\n.param g=100k\nE1 out 0 inp inn {g}\n* a comment\n.ends IDEALOPAMP"
    ir.components.append(component("U1", "opamp", 3, SpiceBinding(device=SpiceDevice.X, model_name="IDEALOPAMP", model_card=user_requirement(card), pin_order=["1", "2", "3"], provenance=USER)))
    ir.net("VIN").pins.append(PinRef(component_ref="U1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="U1", pin_number="2"))
    ir.nets.append(net("OUT", ("U1", "3")))
    assert "XU1 VIN 0 OUT IDEALOPAMP" in build(ir)
    # the runner's own gate matches the way ngspice matches: by prefix
    for line in (".inc x", ".INCLUDE x", ".lib x y", ".opt temp=100", ".ic v(a)=1", ".nodeset v(a)=1", ".global a", ".tf v(a) v1"):
        problems, _ = validate_deck(f"t\n{line}\nR1 a 0 1k\n.end\n")
        assert problems, line
    assert validate_deck("t\n.model dx d(is=1e-14)\nR1 a 0 1k\n.end\n")[0] == []


# --------------------------------------------------------------------------- 47: a bare a/A tail is not modelled


def test_47_ngspice_reads_does_not_vouch_for_a_bare_a_tail():
    assert ngspice_reads("2A") is None and ngspice_reads("1a") is None and ngspice_reads("0.5Amp") is None
    assert ngspice_reads("2") == 2.0 and ngspice_reads("2k") == 2000.0 and ngspice_reads("2kA") == 2000.0  # a unit letter after a scale is ignored, as before


# --------------------------------------------------------------------------- 48: the batch runner is an engine the stage accepts


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="no ngspice binary on PATH")
def test_48_the_batch_runner_reports_the_original_netlist_and_passes_the_stage(tmp_path: Path):
    from ai_eda.compilers import SpiceNetlistCompiler

    ir = spice_divider_ir(tmp_path=tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={}))
    results = {r.check_id: r for r in run_spice_for(ir, {"spice": NgspiceRunner()}, tmp_path)}
    assert results["spice"].status is S.PASS, results["spice"].message
    assert results["spice.vout"].status is S.PASS
    res = NgspiceRunner().run(Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path), SpiceAnalysis.OP, tmp_path / "again")
    assert res.succeeded and res.netlist_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash and res.deck_hash and res.deck_hash != res.netlist_hash
    assert Path(res.deck_path).read_text(encoding="utf-8").endswith(".op\n.end\n")


# --------------------------------------------------------------------------- 49: ignored_pins is for connected pins


def test_49_an_unconnected_pin_cannot_hide_in_ignored_pins():
    ir = spice_divider_ir()
    pot = component("RV1", "10k", 3, SpiceBinding(device=SpiceDevice.R, value=authoritative(10_000.0, DS, "ohm"), pin_order=["1", "2"], ignored_pins={"3": "wiper unused"}, provenance=AUTH))
    ir.components.append(pot)
    ir.net("VIN").pins.append(PinRef(component_ref="RV1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="RV1", pin_number="2"))
    with pytest.raises(CompileError, match="listed in ignored_pins but is in no net"):
        build(ir)
    ir.net("VOUT").pins.append(PinRef(component_ref="RV1", pin_number="3"))  # connected and unused: what ignored_pins is for
    assert build_report(ir)["ignored_pins"] == {"RV1": {"3": "wiper unused"}}


# --------------------------------------------------------------------------- 50: the title is ASCII


def test_50_a_non_ascii_project_id_is_refused_with_the_reason():
    ir = spice_divider_ir()
    ir.project.id = "분압기"
    with pytest.raises(CompileError, match="must be ASCII: it is the SPICE netlist's title line"):
        build(ir)


# --------------------------------------------------------------------------- 51: passive values are positive


@pytest.mark.parametrize("ohms", [0.0, -1000.0])
def test_51_zero_or_negative_passive_values_are_refused(ohms: float):
    ir = spice_divider_ir()
    ir.component("R1").spice.value = authoritative(ohms, DS, "ohm")
    ir.component("R1").electrical["resistance"] = ir.component("R1").spice.value
    with pytest.raises(CompileError, match="value must be positive"):
        build(ir)


# --------------------------------------------------------------------------- 52 / 53: non-finite samples and tolerances


def _tran(vout: list[float]) -> SpiceResult:
    n = len(vout)
    return SpiceResult(engine="t", engine_version="t", netlist_path="x", netlist_hash="sha256:x", analysis=SpiceAnalysis.TRAN, command="tran 1u 5m",
                       vectors={"time": [i * 1e-3 for i in range(n)], "vout": vout}, scale="time", n_points=n, succeeded=True)


def test_52_a_non_finite_sample_anywhere_in_the_vector_is_not_judged():
    exp = Expectation(id="e", analysis_id="tran", vector="v(VOUT)", reduce=Reduce.MAX, nominal=user_requirement(5.0, "V"), tol_abs=user_requirement(0.1, "V"), provenance=USER)
    r = reduce_expectation(_tran([5.0, float("nan"), 2.0]), exp, "vout")
    assert r.measured is None and "non-finite samples" in r.problem
    r = reduce_expectation(_tran([5.0, float("inf"), 2.0]), exp.model_copy(update={"reduce": Reduce.MIN}), "vout")
    assert r.measured is None and "non-finite samples" in r.problem
    assert reduce_expectation(_tran([5.0, 4.0, 2.0]), exp, "vout").measured == 5.0


def test_53_a_non_finite_tolerance_or_nominal_is_no_tolerance():
    inf = Expectation(id="e", analysis_id="op", vector="v(VOUT)", reduce=Reduce.VALUE, nominal=user_requirement(1.0, "V"), tol_abs=user_requirement(float("inf"), "V"), provenance=USER)
    assert judge(1e9, inf)[0] is S.UNRESOLVED
    ir = spice_divider_ir()
    ir.simulation.expectations[0].tol_abs = user_requirement(float("inf"), "V")
    with pytest.raises(CompileError, match="must be a finite number"):
        build(ir)


# --------------------------------------------------------------------------- 55: the system library is found


def test_55_the_system_ngspice_library_is_found_without_the_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ctypes.util

    lib = tmp_path / "libngspice.so.0"
    lib.write_bytes(b"")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libngspice.so.0" if name == "ngspice" else None)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    monkeypatch.delenv("NGSPICE_DLL", raising=False)
    monkeypatch.setattr("ai_eda.tools.kicad.cli.find_kicad_cli", lambda: None)
    if os.name != "nt":
        assert find_system_ngspice() == lib.resolve() and find_ngspice_dll() == lib.resolve()
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    assert find_system_ngspice() is None
