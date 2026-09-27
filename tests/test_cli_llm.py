"""``ai-eda run --llm ...`` end to end through the CLI, offline, plus the flag handling.

Without ``--llm`` the CLI is what it was; with the demo script it blocks on the
confirmation question (exit 1), and with ``--answer confirm_requirements=yes``
it proceeds past MISSING_INFORMATION without a second scripted call. The
``claude`` provider is served by the fake ``claude`` executable of
``tests/fake_claude_cli.py`` (no login, no key, no network): a run on the
subscription route needs no budget, shows its usage and pins the project's
model; ``doctor`` reports the CLI without calling the model.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ai_eda.cli import RUN_OPTIONS, LLMOptions, build_parser, model_pin_refusal, parse_llm_options, run_option_flags
from ai_eda.cli import main as cli_main
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind, ValidationStatus
from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY, request_hash
from ai_eda.llm.providers import describe_providers
from ai_eda.llm.router import TaskKind
from ai_eda.llm.service import SUBSCRIPTION_APPROVED_BY
from ai_eda.security.approval import ExternalAction, default_gate
from tests.fake_claude_cli import DEFAULT_MODEL, DEFAULT_VERSION, FakeClaudeCli, success_structured

DATA = Path(__file__).parent / "data" / "fake_llm_requirements.json"
REQUEST = json.loads(DATA.read_text(encoding="utf-8"))["request"]
#: the demo extraction as the fake CLI's ``structured_output`` (what a model would return for REQUEST)
EXTRACTION = json.loads(DATA.read_text(encoding="utf-8"))["responses"][0]["structured"]
CLAUDE_SPEC = f"claude:{DEFAULT_MODEL}"


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
    # tests/conftest.py lets discovery find only the fake `claude`: no real CLI is probed here, whatever the machine has
    assert "claude cli : NOT FOUND (install Claude Code and run `claude login`, or set AI_EDA_CLAUDE_CLI)" in out
    monkeypatch.delenv("OPENROUTER_API_KEY")
    code, out, _ = _cli("doctor", "--online")
    assert code == 0 and "OPENROUTER_API_KEY not set" in out and "LLM account: not fetched (no key)" in out
    assert "claude cli : NOT FOUND (" in out


# --------------------------------------------------------------------------- the provider flags


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaudeCli:
    """The fake ``claude`` first on PATH, no OpenRouter key, no ``AI_EDA_CLAUDE_CLI``."""
    f = FakeClaudeCli(tmp_path / "bin")
    f.install(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return f


def _parse(*argv: str):
    return build_parser().parse_args(["run", "ir.json", *argv])


def test_every_llm_flag_parses_and_run_option_flags_round_trips():
    values = {
        "llm": "claude,openrouter", "llm_model": "claude:sonnet", "llm_fallback": ["same", "openrouter:x/y:free", "none"],
        "llm_task_model": ["review=claude:opus"], "llm_allow_model_change": True, "llm_claude_cli": "/x/claude",
        "llm_claude_fallback_model": "claude-haiku-4-5", "llm_budget_usd": 0.05, "llm_budget_tokens": 5000,
        "answer": {"confirm_requirements": "yes"}, "online": True, "no_pdf": True,
    }
    argv = run_option_flags(**values)
    assert argv[:4] == ["--answer", "confirm_requirements=yes", "--llm", "claude,openrouter"]
    assert "--llm-allow-model-change" in argv and "--online" in argv and "--no-pdf" in argv
    assert argv.count("--llm-fallback") == 3 and argv[argv.index("--llm-task-model") + 1] == "review=claude:opus"
    ns = _parse(*argv)
    assert ns.llm == "claude,openrouter" and ns.llm_model == "claude:sonnet" and ns.llm_fallback == ["same", "openrouter:x/y:free", "none"]
    assert ns.llm_task_model == ["review=claude:opus"] and ns.llm_allow_model_change is True
    assert ns.llm_claude_cli == "/x/claude" and ns.llm_claude_fallback_model == "claude-haiku-4-5"
    assert ns.llm_budget_usd == 0.05 and ns.llm_budget_tokens == 5000 and ns.answer == ["confirm_requirements=yes"] and ns.online and ns.no_pdf
    # every run option has a dest the parser fills and the defaults render nothing
    ns = _parse()
    assert all(hasattr(ns, o.dest) for o in RUN_OPTIONS) and run_option_flags(**{o.dest: getattr(ns, o.dest) for o in RUN_OPTIONS}) == []
    assert ns.llm is None and ns.llm_fallback is None and ns.llm_allow_model_change is False
    # the flag names exist once: a name that is not a run option is refused, never dropped; a flag takes a bool
    with pytest.raises(TypeError, match="not a run option: llm_modle"):
        run_option_flags(llm_modle="x")
    with pytest.raises(TypeError, match="--online takes a bool"):
        run_option_flags(online="yes")
    assert run_option_flags(llm_fallback=[], llm_model=None, online=False) == []


def test_llm_options_resolve_providers_specs_and_fallbacks_before_any_client():
    assert parse_llm_options(_parse()) is None
    o = parse_llm_options(_parse("--llm", "claude"))
    assert isinstance(o, LLMOptions) and o.providers == ("claude",) and o.script is None and o.primary_spec == CLAUDE_SPEC
    assert o.per_call_providers == [] and o.budget.granted is False and o.router.fallbacks == [] and o.describe() == CLAUDE_SPEC
    o = parse_llm_options(_parse("--llm", "claude,openrouter", "--llm-budget-usd", "1", "--llm-fallback", "same", "--llm-fallback", "none", "--llm-task-model", "review=openrouter:anthropic/claude-opus-4.1"))
    assert o.providers == ("claude", "openrouter") and o.per_call_providers == ["openrouter"]
    assert o.primary_spec == CLAUDE_SPEC and [c.spec for c in o.router.fallbacks] == [f"openrouter:anthropic/{DEFAULT_MODEL}"]
    assert o.router.for_task(TaskKind.REVIEW).spec == "openrouter:anthropic/claude-opus-4.1"
    assert o.describe() == f"{CLAUDE_SPEC}; fallback by --llm-fallback: openrouter:anthropic/{DEFAULT_MODEL}; task models: review=openrouter:anthropic/claude-opus-4.1"
    o = parse_llm_options(_parse("--llm", "openrouter,claude", "--llm-budget-tokens", "10", "--llm-model", "anthropic/claude-sonnet-5", "--llm-fallback", "same"))
    assert o.primary_spec == "openrouter:anthropic/claude-sonnet-5" and [c.spec for c in o.router.fallbacks] == [CLAUDE_SPEC]
    o = parse_llm_options(_parse("--llm", "claude", "--llm-claude-cli", str(DATA), "--llm-claude-fallback-model", "claude-haiku-4-5", "--llm-allow-model-change"))
    assert o.client_kw == {"cli": str(DATA), "fallback_model": "claude-haiku-4-5"} and o.allow_model_change is True
    o = parse_llm_options(_parse("--llm", f"fake:{DATA}", "--llm-budget-usd", "0"))
    assert o.providers == ("script",) and o.script == str(DATA) and o.primary_spec == "script:anthropic/claude-sonnet-5" and o.per_call_providers == ["script"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--llm", "nonsense", "--llm-budget-usd", "1"], "unknown --llm 'nonsense'"),
        (["--llm", "claude,claude"], "names a provider twice"),
        (["--llm", "fake:"], "needs a path"),
        (["--llm", "openrouter,claude"], "the per-call provider openrouter in --llm openrouter,claude needs an explicit budget"),
        (["--llm", "openrouter"], "the per-call provider openrouter"),
        (["--llm", f"fake:{DATA}"], f"the scripted client fake:{DATA} needs an explicit budget"),
        (["--llm", "claude", "--llm-model", "anthropic:claude-sonnet-5"], "direct Messages-API access is not implemented"),
        (["--llm", "claude", "--llm-model", "openrouter:anthropic/claude-sonnet-5"], "name a provider that is not configured (--llm claude)"),
        (["--llm", "claude", "--llm-fallback", "same"], "needs a second configured provider"),
        (["--llm", "claude", "--llm-task-model", "nope=claude:opus"], "unknown task 'nope': use one of " + ", ".join(t.value for t in TaskKind)),
        (["--llm", "claude", "--llm-task-model", "review"], "expects TASK=SPEC"),
        (["--llm", "openrouter", "--llm-budget-usd", "1", "--llm-claude-cli", "/x"], "--llm-claude-cli needs --llm claude"),
        (["--llm", "openrouter", "--llm-budget-usd", "1", "--llm-claude-fallback-model", "x"], "--llm-claude-fallback-model needs --llm claude"),
        (["--llm", "claude", "--llm-claude-cli", "/no/such/claude"], "--llm-claude-cli /no/such/claude: not a file"),
    ],
)
def test_llm_usage_errors_are_exit_2_before_anything_runs(project: Path, argv: list[str], message: str):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_llm_options(_parse(*argv))
    before = CircuitIR.load(project).model_dump_json()
    audit_before = len(default_gate().audit)
    code, out, err = _cli("run", str(project), *argv)
    assert code == 2 and message in err and out == ""
    assert CircuitIR.load(project).model_dump_json() == before and len(default_gate().audit) == audit_before


# --------------------------------------------------------------------------- a run served by the fake Claude Code CLI


def test_run_on_the_subscription_route_needs_no_budget_and_shows_its_usage(project: Path, fake: FakeClaudeCli):
    fake.queue(success_structured(EXTRACTION))
    audit_before = len(default_gate().audit)
    code, out, err = _cli("run", str(project), "--llm", "claude")
    assert code == 1, err
    assert f"  LLM model: {CLAUDE_SPEC} (not pinned yet: the first run that serves a call pins it)" in out
    assert f"[{CONFIRM_KEY}]" in out and "input_voltage" in out and "Jurisdictions: EU" in out
    usage = next(line for line in out.splitlines() if line.startswith("LLM usage:"))
    # 2 + 1133 + 0 input (uncached + cache write + cache read) and 52 output tokens: the measured envelope shape
    assert usage.startswith("LLM usage: 1 served call(s), 1187 tokens, cost 0.000000 USD; budget none (not required: subscription only)")
    assert "; 1 call(s) on the subscription: 1187 tokens, estimated API-equivalent cost 0.004576 USD (not charged)" in usage
    assert f"  LLM model pinned to this project: {CLAUDE_SPEC}" in out
    # the flag was the approval: SUBSCRIPTION_USE granted and consumed with the CLI path, no PAID_API_CALL
    new = default_gate().audit[audit_before:]
    detail = f"claude subscription via {fake.exe} (no per-call cost; usage shown)"
    assert [(e["event"], e["action"], e["detail"]) for e in new if e["action"] == ExternalAction.SUBSCRIPTION_USE] == [("grant", ExternalAction.SUBSCRIPTION_USE, detail), ("consume", ExternalAction.SUBSCRIPTION_USE, detail)]
    assert next(e for e in new if e["event"] == "grant")["by"] == SUBSCRIPTION_APPROVED_BY == "cli --llm claude"
    assert not [e for e in new if e["action"] == ExternalAction.PAID_API_CALL]
    # exactly one structured call through the pinned command, in an empty temp cwd that is gone once the run closed the client
    (call,) = fake.prompt_calls()
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == DEFAULT_MODEL and "--json-schema" in argv and "--tools" in argv and "--strict-mcp-config" in argv
    assert "--no-session-persistence" not in argv and "--bare" not in argv and "--max-budget-usd" not in argv and "--fallback-model" not in argv
    assert json.loads(argv[argv.index("--json-schema") + 1])["type"] == "object"
    assert call["cwd_entries"] == [] and not Path(call["cwd"]).exists() and Path(call["cwd"]) != project.parent
    assert REQUEST in argv[argv.index("-p") + 1]
    # the reply entered the IR the same way a scripted one does: grounded, llm_generated, the served model as the tool
    ir = CircuitIR.load(project)
    req = ir.requirements.get("input_voltage")
    assert req.value.provenance.kind == ProvenanceKind.LLM_GENERATED and req.value.provenance.tool == DEFAULT_MODEL
    entry = ir.requirements.extraction_cache[request_hash(REQUEST)]
    assert entry["model"] == DEFAULT_MODEL and entry["usage"]["cost_source"] == "subscription" and entry["usage"]["estimated_cost_usd"] == 0.004576
    assert entry["cost_usd"] == 0.0 and ir.requirements.llm_model_spec == CLAUDE_SPEC
    assert ir.validation.latest("requirements.extraction").status == ValidationStatus.USER_INPUT_REQUIRED

    # the confirmation run: cache hit, nothing called, the pin stays, a token budget alone is honoured on the subscription route
    code, out, err = _cli("run", str(project), "--llm", "claude", "--llm-budget-tokens", "50000", "--answer", f"{CONFIRM_KEY}=yes")
    assert "LLM usage: 0 served call(s), 0 tokens, cost 0.000000 USD; budget max_tokens=50000" in out and "on the subscription" not in out
    assert f"  LLM model: {CLAUDE_SPEC} (the project pin)" in out and "pinned to this project" not in out
    assert len(fake.prompt_calls()) == 1 and CircuitIR.load(project).requirements.llm_model_spec == CLAUDE_SPEC
    for key in ("input_voltage", "output_voltage", "output_current", "efficiency", "application"):
        assert CircuitIR.load(project).requirements.get(key).value.provenance.kind == ProvenanceKind.USER_REQUIREMENT, key


def test_budget_and_claude_flags_reach_the_cli(tmp_path: Path, fake: FakeClaudeCli):
    _cli("new", "demo2", "--dir", str(tmp_path / "demo2"), "--request", REQUEST)
    fake.queue(success_structured(EXTRACTION))
    code, out, err = _cli(
        "run", str(tmp_path / "demo2" / "ir.json"), "--llm", "claude", "--llm-budget-usd", "0.5",
        "--llm-claude-fallback-model", "claude-haiku-4-5", "--llm-claude-cli", str(fake.exe), "--llm-model", "claude:sonnet",
    )
    assert code == 1, err
    (call,) = fake.prompt_calls()
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.5" and argv[argv.index("--fallback-model") + 1] == "claude-haiku-4-5"
    assert "budget max_usd=0.5" in out and "1 call(s) on the subscription" in out
    assert CircuitIR.load(tmp_path / "demo2" / "ir.json").requirements.llm_model_spec == "claude:sonnet"


def test_openrouter_in_the_mix_without_a_budget_is_refused_naming_openrouter(project: Path, fake: FakeClaudeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    before = CircuitIR.load(project).model_dump_json()
    audit_before = len(default_gate().audit)
    code, out, err = _cli("run", str(project), "--llm", "openrouter,claude")
    assert code == 2 and out == ""
    assert "the per-call provider openrouter in --llm openrouter,claude needs an explicit budget: pass --llm-budget-usd X and/or --llm-budget-tokens N" in err
    assert "only a subscription provider such as claude alone needs none" in err
    assert fake.calls() == [] and CircuitIR.load(project).model_dump_json() == before and len(default_gate().audit) == audit_before
    # with a budget the mix is built: the missing key then names openrouter, still before any call
    code, _, err = _cli("run", str(project), "--llm", "openrouter,claude", "--llm-budget-usd", "1")
    assert code == 2 and "OPENROUTER_API_KEY is not set" in err and fake.prompt_calls() == []
    # claude,openrouter with the key missing names openrouter too; a --llm-claude-cli that is not a file is a usage error
    code, _, err = _cli("run", str(project), "--llm", "claude,openrouter", "--llm-budget-tokens", "10")
    assert code == 2 and "OPENROUTER_API_KEY is not set" in err
    code, _, err = _cli("run", str(project), "--llm", "claude", "--llm-claude-cli", str(project.parent / "no-such-claude"))
    assert code == 2 and err.startswith("--llm: --llm-claude-cli ") and "not a file" in err and fake.prompt_calls() == []
    # no claude binary anywhere: the error names claude and what would make one found
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, _, err = _cli("run", str(project), "--llm", "claude")
    assert code == 2 and "--llm: claude (Claude Code CLI) not found" in err and "AI_EDA_CLAUDE_CLI" in err
    code, _, err = _cli("run", str(project), "--llm", "claude,openrouter", "--llm-budget-usd", "1")
    assert code == 2 and "claude (Claude Code CLI) not found" in err
    assert CircuitIR.load(project).model_dump_json() == before


# --------------------------------------------------------------------------- the project model pin


def test_model_pin_refuses_another_model_and_allow_model_change_repins(project: Path, fake: FakeClaudeCli, tmp_path: Path):
    fake.queue(success_structured(EXTRACTION))
    assert _cli("run", str(project), "--llm", "claude")[0] == 1
    assert CircuitIR.load(project).requirements.llm_model_spec == CLAUDE_SPEC
    before = CircuitIR.load(project).model_dump_json()
    audit_before = len(default_gate().audit)
    # another primary spec: refused in Korean, exit 2, before any client or call - the pin names the model so far
    code, out, err = _cli("run", str(project), "--llm", "claude", "--llm-model", "claude:opus")
    assert code == 2 and out == ""
    assert err.strip() == model_pin_refusal(CLAUDE_SPEC, "claude:opus")
    assert "이 프로젝트는 지금까지 `claude:claude-sonnet-5` 모델로 실행되었습니다. 다른 모델로 계속하려면 --llm-allow-model-change 를 주십시오." in err
    assert len(fake.prompt_calls()) == 1 and CircuitIR.load(project).model_dump_json() == before and len(default_gate().audit) == audit_before
    # the same model on another provider is another spec, the fake included
    code, _, err = _cli("run", str(project), "--llm", f"fake:{DATA}", "--llm-budget-usd", "0")
    assert code == 2 and "`claude:claude-sonnet-5`" in err and "`script:anthropic/claude-sonnet-5`" in err
    # a fallback that differs from the pin is allowed through --llm-fallback (explicit) and shown in the run log line
    code, out, _ = _cli("run", str(project), "--llm", "claude", "--llm-fallback", "claude:opus", "--answer", f"{CONFIRM_KEY}=yes")
    assert code == 1 and f"  LLM model: {CLAUDE_SPEC}; fallback by --llm-fallback: claude:opus (the project pin)" in out
    # --llm-allow-model-change runs, but re-pins only after a served call: the cached extraction serves nothing
    code, out, err = _cli("run", str(project), "--llm", "claude", "--llm-model", "claude:opus", "--llm-allow-model-change")
    assert code == 1, err
    assert "  LLM model: claude:opus (replacing the project pin claude:claude-sonnet-5 by --llm-allow-model-change; re-pinned after a served call)" in out
    assert "LLM usage: 0 served call(s)" in out and "pinned to this project" not in out
    assert CircuitIR.load(project).requirements.llm_model_spec == CLAUDE_SPEC and len(fake.prompt_calls()) == 1
    # a served call under the flag re-pins
    other = tmp_path / "other" / "ir.json"
    _cli("new", "other", "--dir", str(other.parent), "--request", REQUEST)
    ir = CircuitIR.load(other)
    ir.requirements.llm_model_spec = "claude:opus"
    ir.save(other)
    fake.queue(success_structured(EXTRACTION))
    code, out, err = _cli("run", str(other), "--llm", "claude", "--llm-allow-model-change")
    assert code == 1, err
    assert f"  LLM model pinned to this project: {CLAUDE_SPEC}" in out and CircuitIR.load(other).requirements.llm_model_spec == CLAUDE_SPEC
    # without --llm the pin is untouched and nothing is refused
    assert _cli("run", str(other))[0] == 1 and CircuitIR.load(other).requirements.llm_model_spec == CLAUDE_SPEC


def test_model_pin_is_bookkeeping_outside_the_design_view(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ir.requirements.raw_input = REQUEST
    h = ir.content_hash()
    assert ir.requirements.llm_model_spec is None and "llm_model_spec" not in ir.design_dict()["requirements"]
    ir.requirements.llm_model_spec = CLAUDE_SPEC
    assert ir.content_hash() == h and "llm_model_spec" not in ir.design_dict()["requirements"]
    assert ir.model_dump(mode="json")["requirements"]["llm_model_spec"] == CLAUDE_SPEC  # in the file, not in the hash
    path = ir.save(tmp_path / "ir.json")
    loaded = CircuitIR.load(path)
    assert loaded.requirements.llm_model_spec == CLAUDE_SPEC and loaded.content_hash() == h
    ir.requirements.llm_model_spec = "openrouter:anthropic/claude-sonnet-5"
    assert ir.content_hash() == h


# --------------------------------------------------------------------------- doctor and the GUI hook


def test_doctor_reports_the_claude_cli_with_the_fake_and_never_calls_the_model(fake: FakeClaudeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    code, out, _ = _cli("doctor")
    assert code == 0 and "OPENROUTER_API_KEY not set" in out
    assert f"claude cli : {fake.exe} ({DEFAULT_VERSION}, logged in: yes, auth: oauth_token)" in out
    assert [c["argv"] for c in fake.calls()] == [["--version"], ["auth", "status"]] and fake.prompt_calls() == []
    fake.reset()
    fake.set_auth({"loggedIn": False, "authMethod": None, "apiProvider": "firstParty"})
    code, out, _ = _cli("doctor")
    assert code == 0 and f"claude cli : {fake.exe} ({DEFAULT_VERSION}, logged in: no, auth: unknown)" in out and fake.prompt_calls() == []
    # nothing on PATH and no AI_EDA_CLAUDE_CLI: the line says what would make one found
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, out, _ = _cli("doctor")
    assert code == 0 and "claude cli : NOT FOUND (install Claude Code and run `claude login`, or set AI_EDA_CLAUDE_CLI)" in out


def test_describe_providers_reads_the_real_client_against_the_fake(fake: FakeClaudeCli):
    rows = {r.name: r for r in describe_providers()}
    assert rows["openrouter"].available is False and rows["openrouter"].reason == "OPENROUTER_API_KEY not set" and rows["openrouter"].billing == "per_call"
    claude = rows["claude"]
    assert claude.available is True and claude.logged_in is True and claude.billing == "subscription"
    assert claude.reason == f"{fake.exe} ({DEFAULT_VERSION}, logged in: yes, auth: oauth_token)"
    assert [c["argv"] for c in fake.calls()] == [["--version"], ["auth", "status"]]
    assert "sk-" not in json.dumps([r.model_dump() for r in rows.values()])
