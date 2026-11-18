"""Cross-check (``--llm-crosscheck``): the service runs, the deterministic comparison, the requirement agent and the CLI flags.

Scripted clients only (no HTTP, no CLI). What is pinned: every cross-check
model gets the same prompt and schema independently; a failed one is
reported, never dropped; agreement proceeds as without the cross-check with
both models in the provenance; a differing field is withheld and asked with
every model's proposal; nothing changes without the flag.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ai_eda.agents import AgentContext
from ai_eda.cli import build_parser, parse_llm_options
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, ValidationStatus
from ai_eda.llm.client import LLMMessage, Usage
from ai_eda.llm.crosscheck import AGREEMENT_MARKER, NOT_PROPOSED, compare_extractions, disagreement_question, mark_agreed, withhold
from ai_eda.llm.extraction import CONFIRM_KEY, RequirementExtraction, ground_extraction
from ai_eda.llm.fake import ScriptedLLMClient
from ai_eda.llm.router import DEFAULT_PRIMARY_MODEL, TaskKind, default_router, parse_model_spec
from ai_eda.llm.service import LLMBudget, LLMService
from ai_eda.llm.usage import UsageTracker
from ai_eda.security import ApprovalGate
from ai_eda.workflow import Orchestrator, Stage
from tests.test_requirement_agent_llm import CANNED, RAW, USAGE

OTHER = "other-model"


class Word(BaseModel):
    word: str


def _service(items: list[Any], crosscheck: list[str] | None = None, budget: LLMBudget | None = None) -> tuple[LLMService, ScriptedLLMClient]:
    client = ScriptedLLMClient(items)
    # a token budget beside the USD one: scripted replies without a usage have an unknown cost, which a USD-only budget refuses after
    svc = LLMService(client, default_router(), UsageTracker(), budget or LLMBudget(max_usd=1.0, max_tokens=1_000_000), gate=ApprovalGate(),
                     crosscheck=[parse_model_spec(s) for s in (crosscheck if crosscheck is not None else [OTHER])])
    return svc, client


def _ir(tmp_path: Path) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="conv", name="converter", workdir=str(tmp_path)))
    ir.requirements.raw_input = RAW
    return ir


def _run(ir: CircuitIR, svc: LLMService, tmp_path: Path, answers: dict[str, str] | None = None):
    return Orchestrator(AgentContext(workdir=tmp_path, llm=svc, answers=answers or {})).run(ir, stop_after=Stage.REQUIREMENT_ANALYSIS)


def _grounded(data: dict[str, Any], model: str):
    return ground_extraction(RAW, RequirementExtraction.model_validate(data), model)


def _differing(**changes: Any) -> dict[str, Any]:
    """CANNED with the output voltage changed (a value the request does not state: the 6 V item is demoted to an assumption)."""
    other = copy.deepcopy(CANNED)
    for item in other["requirements"]:
        if item["key"] == "output_voltage":
            item["value"]["number"] = changes.get("number", 6)
    return other


# --------------------------------------------------------------------------- comparison


def test_identical_grounded_extractions_agree_on_every_field():
    result = compare_extractions(_grounded(CANNED, "a"), [_grounded(copy.deepcopy(CANNED), "b")])
    assert result.disagreements == [] and result.models == ["b"] and result.primary_model == "a"
    assert "output_voltage" in result.compared and "application" in result.compared and "jurisdiction" in result.compared
    assert result.agreed == result.compared


def test_a_value_a_missing_item_a_category_and_the_jurisdictions_differ_field_by_field():
    other = _differing()
    other["requirements"] = [r for r in other["requirements"] if r["key"] != "efficiency"]
    for r in other["requirements"]:
        if r["key"] == "output_current":
            r["category"] = "thermal"
    other["jurisdictions"] = []
    result = compare_extractions(_grounded(CANNED, "a"), [_grounded(other, "b")])
    assert result.differing_keys == ["output_voltage", "output_current", "efficiency", "jurisdiction"]
    by_key = {d.key: d for d in result.disagreements}
    volts = by_key["output_voltage"].proposals
    assert [p.model for p in volts] == ["a", "b"] and volts[0].value.startswith("5 V (explicit/given") and volts[1].value.startswith("6 V (assumption/assumed")
    assert by_key["efficiency"].proposals[1].value == NOT_PROPOSED
    assert [p.value for p in by_key["jurisdiction"].proposals] == ["EU", "(none)"]
    assert "input_voltage" in result.agreed and "application" in result.agreed


def test_numbers_agree_within_the_grounding_tolerance_only():
    other = copy.deepcopy(CANNED)
    other["assumptions"] = [{"key": "ripple", "text": "ripple", "category": "electrical", "number": 50.0, "unit": "mV", "number_high": None, "rationale": "r"}]
    base = copy.deepcopy(CANNED)
    base["assumptions"] = [{"key": "ripple", "text": "ripple", "category": "electrical", "number": 50.0 * (1 + 1e-12), "unit": "mV", "number_high": None, "rationale": "r"}]
    assert compare_extractions(_grounded(base, "a"), [_grounded(other, "b")]).disagreements == []
    base["assumptions"][0]["number"] = 50.001  # outside REL_TOL: no widening for the cross-check
    assert compare_extractions(_grounded(base, "a"), [_grounded(other, "b")]).differing_keys == ["ripple"]


def test_a_conflict_not_every_model_declared_disputes_its_keys_and_questions_are_joined():
    base = copy.deepcopy(CANNED)
    base["conflicts"] = [{"keys": ["input_voltage", "output_voltage"], "description": "5 V out of 12 V in is fine, but say so"}]
    other = copy.deepcopy(CANNED)
    other["questions"] = [*other["questions"], {"key": "enclosure", "question": "Which enclosure?", "required": False, "options": [], "rationale": "fit"}]
    result = compare_extractions(_grounded(base, "a"), [_grounded(other, "b")])
    assert result.differing_keys == ["input_voltage", "output_voltage"]
    volts = {d.key: d for d in result.disagreements}["input_voltage"].proposals
    assert volts[0].value.endswith("; conflict input_voltage <-> output_voltage") and volts[1].value.endswith("; no conflict")
    [extra] = result.extra_questions
    assert extra.key == "enclosure" and extra.source == "llm" and "asked by b, not by a (cross-check)" in extra.rationale
    # the same conflict declared by both (other words) is no disagreement; the withheld keys take the conflict with them
    other["conflicts"] = [{"keys": ["output_voltage", "input_voltage"], "description": "different prose"}]
    assert compare_extractions(_grounded(base, "a"), [_grounded(other, "b")]).differing_keys == []
    assert withhold(_grounded(base, "a"), {"input_voltage"}).conflicts == []


def test_withhold_mark_agreed_and_the_question():
    g = _grounded(CANNED, "a")
    w = withhold(g, {"output_voltage", "jurisdiction", "application"})
    assert "output_voltage" not in {r.key for r in w.requirements} and w.jurisdictions == [] and w.application is None
    assert "req.output_voltage" not in w.contexts and "req.application" not in w.contexts
    assert withhold(g, set()) is g
    marked = mark_agreed(list(w.requirements), {"input_voltage"}, ["b"])
    iv = next(r for r in marked if r.key == "input_voltage")
    assert iv.value.provenance.note.endswith(f"; {AGREEMENT_MARKER}b") and iv.value.provenance.note.startswith("quote: ")
    assert next(r for r in marked if r.key == "output_current").value.provenance.note == next(r for r in w.requirements if r.key == "output_current").value.provenance.note
    d = compare_extractions(g, [_grounded(_differing(), "b")]).disagreements[0]
    q = disagreement_question(d, leave_out="leave_out=output_voltage")
    assert q.key == "output_voltage" and q.required and q.source == "system"
    assert "  - a: 5 V" in q.question and "  - b: 6 V" in q.question and "--answer output_voltage=<value>" in q.question and "leave_out=output_voltage" in q.question
    assert "leave_out" not in disagreement_question(d).question


# --------------------------------------------------------------------------- service


def test_cross_structured_sends_the_same_messages_to_each_model_independently():
    svc, client = _service([{"structured": {"word": "a"}}, {"structured": {"word": "b"}, "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost_usd": 0.001}}],
                           crosscheck=[OTHER, "third-model"])
    msgs = [LLMMessage(role="user", content="U")]
    replies = svc.cross_structured(TaskKind.REQUIREMENT_ANALYSIS, msgs, Word)
    assert [c.model for c in client.calls] == [OTHER, "third-model"]
    assert all(c.messages == msgs and c.response_schema == Word.model_json_schema() for c in client.calls)
    assert [(r.spec, r.ok, r.data) for r in replies] == [(OTHER, True, {"word": "a"}), ("third-model", True, {"word": "b"})]
    assert replies[0].cost_usd is None and replies[1].cost_usd == pytest.approx(0.001)  # unknown stays unknown
    assert len(svc.last_attempts) == 2 and len(svc.usage.records) == 2
    # only the asked-for spec runs; the script is used up, so it fails - and is returned as failed, not raised
    only = svc.cross_structured(TaskKind.REQUIREMENT_ANALYSIS, msgs, Word, specs=["third-model"])
    assert [(r.spec, r.ok, r.outcome) for r in only] == [("third-model", False, "script")]


def test_a_failed_cross_check_model_is_returned_with_its_reason_never_raised():
    svc, client = _service([
        "not json", "still not json",  # schema failure on both turns
        {"error": {"message": "upstream down", "status": 503}},
    ], crosscheck=[OTHER, "third-model"])
    replies = svc.cross_structured(TaskKind.REQUIREMENT_ANALYSIS, [LLMMessage(role="user", content="U")], Word)
    assert [(r.spec, r.ok, r.outcome) for r in replies] == [(OTHER, False, "validation"), ("third-model", False, "http")]
    assert "did not match the schema on both turns" in (replies[0].error or "") and "upstream down" in (replies[1].error or "")
    assert len(client.calls) == 3  # one feedback turn, no fallback to another model


def test_the_budget_binds_every_cross_check_request():
    svc, client = _service([{"structured": {"word": "a"}}], budget=LLMBudget(max_tokens=1))
    svc.usage.record(TaskKind.REQUIREMENT_ANALYSIS, "x", Usage(total_tokens=5, cost_usd=0.0))
    [reply] = svc.cross_structured(TaskKind.REQUIREMENT_ANALYSIS, [LLMMessage(role="user", content="U")], Word)
    assert not reply.ok and reply.outcome == "budget" and client.calls == []


def test_a_cross_check_spec_must_name_a_configured_provider():
    from ai_eda.llm.service import check_router_providers

    with pytest.raises(ValueError, match="codex:gpt-6-luna"):
        check_router_providers(default_router(default_provider="claude"), ["claude"], [parse_model_spec("codex:gpt-6-luna")])
    check_router_providers(default_router(default_provider="claude"), ["claude", "codex"], [parse_model_spec("codex:gpt-6-luna")])


# --------------------------------------------------------------------------- requirement agent


def test_agreement_proceeds_as_today_with_both_models_in_the_provenance_and_is_cached(tmp_path: Path):
    other = copy.deepcopy(CANNED)  # the same extraction, plus a question only the cross-check model asks: asked too, never compared
    other["questions"].append({"key": "enclosure", "question": "Which enclosure?", "required": False, "options": [], "rationale": "fit"})
    svc, client = _service([{"structured": CANNED, "usage": USAGE}, {"structured": other}])
    ir = _ir(tmp_path)
    state = _run(ir, svc, tmp_path)
    [enclosure] = [q for q in ir.requirements.missing if q.key == "enclosure"]
    assert enclosure.source == "llm" and f"asked by {OTHER}" in enclosure.rationale and not enclosure.required
    assert [c.model for c in client.calls] == [DEFAULT_PRIMARY_MODEL, OTHER]
    assert client.calls[0].messages == client.calls[1].messages and client.calls[0].response_schema == client.calls[1].response_schema
    assert [q.key for q in state.open_questions] == [CONFIRM_KEY]
    iv = ir.requirements.get("input_voltage")
    assert iv.value.provenance.tool == DEFAULT_PRIMARY_MODEL and iv.value.provenance.note.endswith(f"{AGREEMENT_MARKER}{OTHER}")
    [eu] = ir.regulatory.jurisdictions
    assert eu.code == "EU"
    x = ir.validation.latest("requirements.crosscheck")
    assert x.status == ValidationStatus.PASS and x.details["models"] == [OTHER] and x.details["disagreements"] == []
    entry = next(iter(ir.requirements.extraction_cache.values()))
    assert entry["crosscheck"][OTHER]["ok"] is True
    # the next run replays both from the cache (no call) and the user's confirmation keeps the agreement note
    state = _run(ir, svc, tmp_path, {CONFIRM_KEY: "yes"})
    assert len(client.calls) == 2
    iv = ir.requirements.get("input_voltage")
    assert iv.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT and AGREEMENT_MARKER in iv.value.provenance.note


def test_disagreement_withholds_the_field_and_asks_with_both_proposals(tmp_path: Path):
    svc, client = _service([{"structured": CANNED, "usage": USAGE}, {"structured": _differing()}])
    ir = _ir(tmp_path)
    state = _run(ir, svc, tmp_path)
    assert ir.requirements.get("output_voltage") is None  # neither 5 V nor 6 V entered the design
    assert ir.requirements.get("input_voltage") is not None
    [q] = [q for q in state.open_questions if q.key == "output_voltage"]
    assert q.required and f"  - {DEFAULT_PRIMARY_MODEL}: 5 V" in q.question and f"  - {OTHER}: 6 V" in q.question
    assert "output_voltage" in [m.key for m in ir.requirements.missing]
    x = ir.validation.latest("requirements.crosscheck")
    assert x.status == ValidationStatus.USER_INPUT_REQUIRED and x.details["open"] == ["output_voltage"]
    assert [p["model"] for p in x.details["disagreements"][0]["proposals"]] == [DEFAULT_PRIMARY_MODEL, OTHER]
    table = next(m for m in ir.requirements.missing if m.key == CONFIRM_KEY).question
    assert "output_voltage" not in table
    # a confirmation is refused while the disagreement is open
    _run(ir, svc, tmp_path, {CONFIRM_KEY: "yes"})
    assert not next(iter(ir.requirements.extraction_cache.values())).get("confirmed")
    assert ir.validation.latest("requirements.extraction").status == ValidationStatus.USER_INPUT_REQUIRED
    # the user's own value decides it; then the confirmation goes through (no further model call)
    state = _run(ir, svc, tmp_path, {"output_voltage": "5 V", CONFIRM_KEY: "yes"})
    assert len(client.calls) == 2
    ov = ir.requirements.get("output_voltage")
    assert ov.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT and ov.value.value == "5 V"
    x = ir.validation.latest("requirements.crosscheck")
    assert x.status == ValidationStatus.PASS and x.details["decided"] == ["output_voltage"] and x.details["open"] == []
    assert next(iter(ir.requirements.extraction_cache.values()))["confirmed"] is True


def test_a_failed_cross_check_is_reported_and_retried_and_the_extraction_proceeds(tmp_path: Path):
    svc, client = _service([{"structured": CANNED, "usage": USAGE}, {"error": {"message": "boom", "status": 503}},
                            {"structured": copy.deepcopy(CANNED)}])
    ir = _ir(tmp_path)
    state = _run(ir, svc, tmp_path)
    assert [q.key for q in state.open_questions] == [CONFIRM_KEY] and ir.requirements.get("output_voltage") is not None
    x = ir.validation.latest("requirements.crosscheck")
    assert x.status == ValidationStatus.NOT_VERIFIED and "other-model failed (http)" in x.message and x.details["failed"][0]["spec"] == OTHER
    assert "crosscheck" not in next(iter(ir.requirements.extraction_cache.values()))
    _run(ir, svc, tmp_path)  # asked again (only the cross-check model; the primary is cached)
    assert [c.model for c in client.calls] == [DEFAULT_PRIMARY_MODEL, OTHER, OTHER]
    assert ir.validation.latest("requirements.crosscheck").status == ValidationStatus.PASS


def test_a_task_the_cross_check_does_not_cover_says_so(tmp_path: Path):
    from tests.test_regulatory_agent import GOOD, _llm_ctx, _run_agent
    from tests.test_regulatory_agent import _ir as regulatory_ir

    svc, client = _service([{"structured": {"candidates": [GOOD]}}])
    result = _run_agent(regulatory_ir(tmp_path, "EU"), _llm_ctx(tmp_path, svc))
    assert [c.model for c in client.calls] == [DEFAULT_PRIMARY_MODEL]  # no cross-check call for the regulation proposals
    assert any("covers the requirement extraction only" in n and "regulation proposals" in n for n in result.notes)


def test_without_cross_check_models_nothing_changes(tmp_path: Path):
    svc, client = _service([{"structured": CANNED, "usage": USAGE}], crosscheck=[])
    ir = _ir(tmp_path)
    _run(ir, svc, tmp_path)
    assert len(client.calls) == 1 and ir.validation.latest("requirements.crosscheck") is None
    entry = next(iter(ir.requirements.extraction_cache.values()))
    assert "crosscheck" not in entry and AGREEMENT_MARKER not in (ir.requirements.get("input_voltage").value.provenance.note or "")


# --------------------------------------------------------------------------- CLI


def _args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["run", "ir.json", *argv])


def test_cli_crosscheck_and_provider_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    opts = parse_llm_options(_args("--llm", "claude,codex", "--llm-crosscheck", "codex:gpt-6-luna"))
    assert [c.spec for c in opts.crosscheck] == ["codex:gpt-6-luna"] and opts.per_call_providers == []
    assert "cross-check by --llm-crosscheck: codex:gpt-6-luna" in opts.describe()
    with pytest.raises(ValueError, match="name a provider that is not configured"):
        parse_llm_options(_args("--llm", "claude", "--llm-crosscheck", "codex:gpt-6-luna"))
    with pytest.raises(ValueError, match="--llm-codex-cli needs --llm codex"):
        parse_llm_options(_args("--llm", "claude", "--llm-codex-cli", str(tmp_path)))
    with pytest.raises(ValueError, match="not a file"):
        parse_llm_options(_args("--llm", "codex", "--llm-codex-cli", str(tmp_path / "nope.exe")))
    with pytest.raises(ValueError, match="--llm-openai-base-url needs --llm openai_compat"):
        parse_llm_options(_args("--llm", "claude", "--llm-openai-base-url", "http://127.0.0.1:8080/v1"))
    with pytest.raises(ValueError, match="no default model"):
        parse_llm_options(_args("--llm", "openai_compat", "--llm-openai-base-url", "http://127.0.0.1:8080/v1"))
    # a server on this machine needs no budget; a remote endpoint does
    monkeypatch.delenv("AI_EDA_OPENAI_BASE_URL", raising=False)
    local = parse_llm_options(_args("--llm", "openai_compat", "--llm-model", "openai_compat:llama3", "--llm-openai-base-url", "http://127.0.0.1:8080/v1",
                                    "--llm-openai-key-env", "MY_KEY"))
    assert local.per_call_providers == [] and local.client_kw == {"openai_base_url": "http://127.0.0.1:8080/v1", "openai_key_env": "MY_KEY"}
    with pytest.raises(ValueError, match="needs an explicit budget"):
        parse_llm_options(_args("--llm", "openai_compat", "--llm-model", "openai_compat:gpt-x", "--llm-openai-base-url", "https://api.openai.com/v1"))
    assert parse_llm_options(_args("--llm", "codex")).router.default.spec == "codex:gpt-6-luna"


def test_cli_flags_are_outside_the_gui_run_form():
    from ai_eda.cli import LLM_EXTRA_OPTIONS, RUN_OPTIONS

    assert {o[1] for o in LLM_EXTRA_OPTIONS}.isdisjoint({o.dest for o in RUN_OPTIONS})
    ns = _args()
    assert ns.llm_crosscheck is None and ns.llm_codex_cli is None and ns.llm_openai_base_url is None and ns.llm_openai_key_env is None
