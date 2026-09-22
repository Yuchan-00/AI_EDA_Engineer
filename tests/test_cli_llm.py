"""``ai-eda run --llm fake:<json>`` end to end through the CLI, offline, plus the flag handling.

Without ``--llm`` the CLI is what it was; with the demo script it blocks on the
confirmation question (exit 1), and with ``--answer confirm_requirements=yes``
it proceeds past MISSING_INFORMATION without a second scripted call.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ai_eda.cli import main as cli_main
from ai_eda.ir import CircuitIR, ProvenanceKind, ValidationStatus
from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY, request_hash
from ai_eda.security.approval import ExternalAction, default_gate

DATA = Path(__file__).parent / "data" / "fake_llm_requirements.json"
REQUEST = json.loads(DATA.read_text(encoding="utf-8"))["request"]


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    code, out, _ = _cli("new", "demo", "--dir", str(tmp_path / "demo"), "--request", REQUEST)
    assert code == 0 and "created" in out
    return tmp_path / "demo" / "ir.json"


def test_run_without_llm_is_unchanged(project: Path):
    code, out, _ = _cli("run", str(project))
    assert code == 1
    assert "[application]" in out and "[jurisdiction]" in out and CONFIRM_KEY not in out and "LLM usage" not in out
    ir = CircuitIR.load(project)
    assert ir.requirements.requirements == [] and ir.requirements.extraction_cache == {}


def test_run_with_fake_llm_blocks_on_confirmation_then_proceeds(project: Path):
    audit_before = len(default_gate().audit)
    code, out, err = _cli("run", str(project), "--llm", f"fake:{DATA}", "--llm-budget-usd", "0")
    assert code == 1, err
    assert f"[{CONFIRM_KEY}]" in out and "input_voltage" in out and "Jurisdictions: EU" in out
    assert "[application]" not in out and "[jurisdiction]" not in out  # both were extracted and grounded
    assert "LLM usage: 1 served call(s)" in out and "cost 0.000000 USD" in out and "budget max_usd=0" in out
    assert "requirement_analysis" in out and "USER_INPUT_REQUIRED" in out and "awaiting user confirmation" in out
    grants = [e for e in default_gate().audit[audit_before:] if e["action"] == ExternalAction.PAID_API_CALL]
    assert [e["event"] for e in grants] == ["grant", "consume"] and grants[0]["by"].startswith("cli --llm-budget")
    ir = CircuitIR.load(project)
    assert ir.requirements.get("input_voltage").value.provenance.kind == ProvenanceKind.LLM_GENERATED
    assert ir.requirements.get("input_voltage").value.provenance.tool == "scripted/requirements-demo"
    assert ir.requirements.extraction_cache[request_hash(REQUEST)]["confirmed"] is False
    assert ir.regulatory.jurisdictions[0].provided_by_user is False

    code, out, err = _cli("run", str(project), "--llm", f"fake:{DATA}", "--llm-budget-usd", "0", "--answer", f"{CONFIRM_KEY}=yes")
    assert "LLM usage: 0 served call(s)" in out  # cache hit: the script was not consumed again
    stages = {line.split()[0]: line.split()[1] for line in out.splitlines() if line and not line.startswith(("LLM", "BLOCKED", " "))}
    assert stages["requirement_analysis"] == "PASS" and stages["missing_information"] == "PASS"
    ir = CircuitIR.load(project)
    for key in ("input_voltage", "output_voltage", "output_current", "efficiency", "application"):
        assert ir.requirements.get(key).value.provenance.kind == ProvenanceKind.USER_REQUIREMENT, key
    # the model's implicit item is not the user's: it stays llm_generated and IR_BUILD blocks on it, naming the answer keys
    assert ir.requirements.get("input_reverse_polarity_protection").value.provenance.kind == ProvenanceKind.LLM_GENERATED
    assert ir.regulatory.jurisdictions[0].provided_by_user is True and ir.regulatory.jurisdiction_known
    assert ir.validation.latest("requirements.extraction").status == ValidationStatus.PASS
    assert ir.validation.latest("ir.assumptions").status == ValidationStatus.PASS
    assert ir.validation.latest("ir.llm_requirements").status == ValidationStatus.USER_INPUT_REQUIRED
    assert code == 1 and "BLOCKED" in out and "[ir.llm_requirements]" in out and f"{ACCEPT_KEY}=" in out and "input_reverse_polarity_protection" in out
    assert "release" not in stages

    code, out, err = _cli("run", str(project), "--llm", f"fake:{DATA}", "--llm-budget-usd", "0", "--answer", f"{ACCEPT_KEY}=input_reverse_polarity_protection")
    assert "LLM usage: 0 served call(s)" in out
    assert "BLOCKED" not in out
    ir = CircuitIR.load(project)
    accepted = ir.requirements.get("input_reverse_polarity_protection")
    assert accepted.value.provenance.kind == ProvenanceKind.USER_REQUIREMENT and accepted.value.provenance.note.startswith("accepted by user")
    assert ir.validation.latest("ir.llm_requirements").status == ValidationStatus.PASS
    # the run reaches RELEASE; with no circuit the reviewer finds the confirmed electrical requirements unserved,
    # so the design is not releasable and the exit code says so
    assert "release" in out
    release_line = next(line for line in out.splitlines() if line.startswith("release"))
    assert "PASS" not in release_line.split(maxsplit=2)[1]
    assert code == (1 if "FAIL" in release_line else 0)
    assert "requirements_vs_ir" in release_line or code == 0


def test_llm_flag_without_budget_is_refused_before_anything_runs(project: Path):
    before = CircuitIR.load(project).model_dump_json()
    code, out, err = _cli("run", str(project), "--llm", f"fake:{DATA}")
    assert code == 2 and "explicit budget" in err and out == ""
    assert CircuitIR.load(project).model_dump_json() == before


def test_unknown_llm_spec_and_missing_key(project: Path, monkeypatch):
    code, _, err = _cli("run", str(project), "--llm", "nonsense", "--llm-budget-usd", "1")
    assert code == 2 and "unknown --llm" in err
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code, _, err = _cli("run", str(project), "--llm", "openrouter", "--llm-budget-usd", "1")
    assert code == 2 and "OPENROUTER_API_KEY is not set" in err
    code, _, err = _cli("run", str(project), "--llm", f"fake:{project.parent / 'missing.json'}", "--llm-budget-tokens", "10")
    assert code == 2 and "missing.json" in err


def test_doctor_never_prints_the_key_and_fetches_nothing_offline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-never-printed")
    code, out, _ = _cli("doctor")
    assert code == 0 and "LLM key   : set" in out and "sk-or-v1-never-printed" not in out and "LLM account" not in out
    monkeypatch.delenv("OPENROUTER_API_KEY")
    code, out, _ = _cli("doctor", "--online")
    assert code == 0 and "OPENROUTER_API_KEY not set" in out and "LLM account: not fetched (no key)" in out
