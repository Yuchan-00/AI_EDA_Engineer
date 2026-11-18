"""The Codex CLI client against the fake ``codex`` (tests/fake_codex_cli.py): no login, no network, no real CLI.

What is pinned: the measured command line (read-only sandbox, empty temporary
cwd, no user config, no repository), the prompt on stdin as UTF-8, the
schema in a file outside the cwd, the measured event stream parsed into one
response with subscription cost semantics, the measured failure typed, and
the service's SUBSCRIPTION_USE approval for ``--llm codex``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ai_eda.llm.codex_cli as codex_cli
from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMError, LLMMessage
from ai_eda.llm.codex_cli import CodexCliClient, parse_events, parse_usage, render_prompt
from ai_eda.llm.router import DEFAULT_CODEX_MODEL, TaskKind, default_router, parse_model_spec
from ai_eda.llm.service import LLMBudget, LLMService
from ai_eda.security import ApprovalGate
from ai_eda.security.approval import ExternalAction
from tests.fake_codex_cli import CHATGPT_LOGIN, FAKE_MARKER, THREAD_ID, FakeCodexCli, failed_turn, success, success_structured, usage

PINNED = [
    "exec", "--json", "--ignore-user-config", "--ignore-rules", "-c", "model_provider=openai", "-m", "gpt-6-luna",
    "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--disable", "shell_tool", "--disable", "unified_exec",
    "--color", "never", "-C",
]


@pytest.fixture
def fake(tmp_path: Path) -> FakeCodexCli:
    return FakeCodexCli(tmp_path / "bin")


def _client(fake: FakeCodexCli, **kw) -> CodexCliClient:
    return CodexCliClient(codex_cli=str(fake.exe), **kw)


def _user(text: str) -> list[LLMMessage]:
    return [LLMMessage(role="user", content=text)]


# --------------------------------------------------------------------------- pure parts


def test_parse_usage_has_subscription_semantics_and_input_includes_cached():
    u = parse_usage(usage(13046, 10240, 5, 0))
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (13046, 5, 13051)
    assert u.cached_tokens == 10240 and u.cache_write_tokens == 0 and u.reasoning_tokens == 0
    assert u.cost_usd == 0.0 and u.cost_source == "subscription" and u.billing == "subscription" and u.estimated_cost_usd is None
    assert parse_usage(None).total_tokens == 0 and parse_usage({"input_tokens": True}).prompt_tokens == 0


def test_parse_events_keeps_objects_and_reports_other_lines():
    events, unparsed = parse_events('{"type":"turn.started"}\n\nnot json\n[1,2]\n')
    assert events == [{"type": "turn.started"}] and unparsed == ["not json", "[1,2]"]


def test_render_prompt_one_user_message_is_verbatim_else_role_blocks():
    assert render_prompt(_user("전압 12 V")) == "전압 12 V"
    msgs = [LLMMessage(role="system", content="S"), LLMMessage(role="user", content="U"), LLMMessage(role="assistant", content="A"),
            LLMMessage(role="user", content="F")]
    assert render_prompt(msgs) == "[system]\nS\n\n[user]\nU\n\n[assistant]\nA\n\n[user]\nF"
    with pytest.raises(ValueError, match="no caller tools"):
        render_prompt([LLMMessage(role="tool", content="x", tool_call_id="1")])
    with pytest.raises(ValueError, match="user message"):
        render_prompt([LLMMessage(role="system", content="S")])


def test_a_batch_file_is_refused(tmp_path: Path):
    shim = tmp_path / "codex.cmd"
    shim.write_text("@echo off\n", encoding="utf-8")
    with pytest.raises(ToolUnavailableError, match="batch file"):
        codex_cli.refuse_batch_file(str(shim), windows=True)
    assert codex_cli.refuse_batch_file(str(shim), windows=False) == str(shim)


def test_discovery_env_then_app_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(codex_cli.ENV_CLI, raising=False)
    assert codex_cli.find_codex_cli() is None
    # the Codex app's install: the newest bin/<hash>/codex.exe under LOCALAPPDATA (Windows only)
    old, new = tmp_path / "OpenAI" / "Codex" / "bin" / "aaa", tmp_path / "OpenAI" / "Codex" / "bin" / "bbb"
    for d, mtime in ((old, 1_000_000), (new, 2_000_000)):
        d.mkdir(parents=True)
        (d / "codex.exe").write_bytes(b"")
        (d / FAKE_MARKER).write_text("x", encoding="utf-8")
        os.utime(d / "codex.exe", (mtime, mtime))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(codex_cli, "_is_windows", lambda: True)
    assert codex_cli.find_codex_cli() == str(new / "codex.exe")
    monkeypatch.setattr(codex_cli, "_is_windows", lambda: False)
    assert codex_cli.find_codex_cli() is None
    # an explicit absolute path wins
    monkeypatch.setenv(codex_cli.ENV_CLI, str(old / "codex.exe"))
    assert codex_cli.find_codex_cli() == str(old / "codex.exe")


def test_no_cli_found_is_tool_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(codex_cli.ENV_CLI, raising=False)
    monkeypatch.setattr(codex_cli, "_is_windows", lambda: False)
    with pytest.raises(ToolUnavailableError, match="Codex CLI"):
        CodexCliClient()


# --------------------------------------------------------------------------- calls through the fake


def test_plain_call_pins_the_command_and_parses_the_measured_events(fake: FakeCodexCli):
    fake.queue(success("OK"))
    client = _client(fake)
    resp = client.complete("gpt-6-luna", _user("Reply OK. (한국어 확인: 전압 12 V ±5 %)"), temperature=0.3, max_tokens=50)
    [call] = fake.exec_calls()
    argv = call["argv"]
    assert argv[: len(PINNED)] == PINNED and argv[-1] == "-" and "--output-schema" not in argv
    work = argv[len(PINNED)]
    assert Path(call["cwd"]).resolve() == Path(work).resolve() and call["cwd_entries"] == []
    assert not Path(work).exists()  # the temporary directories are removed after the call
    assert fake.stdin_bytes(call) == "Reply OK. (한국어 확인: 전압 12 V ±5 %)".encode("utf-8")
    assert resp.content == "OK" and resp.structured is None and resp.model_used == "gpt-6-luna" and resp.id == THREAD_ID
    assert resp.finish_reason == "stop" and resp.raw_error is None
    assert resp.usage.prompt_tokens == 13046 and resp.usage.cost_usd == 0.0 and resp.usage.billing == "subscription"
    assert resp.raw["temperature_ignored"] is True and resp.raw["max_tokens_ignored"] is True
    assert [e["type"] for e in resp.raw["events"]] == ["thread.started", "turn.started", "item.completed", "turn.completed"]


def test_schema_call_writes_the_schema_file_outside_the_cwd_and_parses_the_object(fake: FakeCodexCli):
    schema = {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"], "additionalProperties": False}
    fake.queue(success_structured({"word": "전압"}, usage_obj=usage(13078, 0, 20)))
    resp = _client(fake).complete("gpt-6-luna", [LLMMessage(role="system", content="S"), *_user("U")], response_schema=schema)
    [call] = fake.exec_calls()
    argv = call["argv"]
    path = argv[argv.index("--output-schema") + 1]
    assert json.loads(call["schema"]) == schema and Path(path).parent != Path(argv[len(PINNED)])
    assert fake.stdin_bytes(call).decode("utf-8") == "[system]\nS\n\n[user]\nU"
    assert resp.structured == {"word": "전압"} and resp.raw_error is None and "max_tokens_ignored" not in resp.raw


def test_a_reply_that_is_not_a_json_object_is_a_defect_not_a_crash(fake: FakeCodexCli):
    fake.queue(success("not json"))
    resp = _client(fake).complete("m", _user("U"), response_schema={"type": "object"})
    assert resp.structured is None and "not valid JSON" in (resp.raw_error or "")


def test_the_measured_failed_turn_is_a_typed_response_error(fake: FakeCodexCli):
    fake.queue(failed_turn())
    with pytest.raises(LLMError) as ei:
        _client(fake).complete("no-such-model-xyz", _user("U"))
    e = ei.value
    assert e.kind == "response" and e.code == codex_cli.TURN_FAILED and e.sent is True and e.status is None
    assert "not supported when using Codex with a ChatGPT account" in e.message and e.message.count("not supported") == 1
    assert e.maybe_billed and not e.retryable


def test_exit_without_events_and_empty_runs_are_typed(fake: FakeCodexCli):
    fake.queue({"stdout": "", "stderr": "boom\n", "exit": 2}, {"stdout": "plain text\n", "exit": 0},
               {"events": [{"type": "thread.started", "thread_id": THREAD_ID}, {"type": "turn.completed", "usage": usage()}], "exit": 0})
    client = _client(fake)
    with pytest.raises(LLMError) as a:
        client.complete("m", _user("U"))
    assert a.value.kind == "http" and a.value.code == "exit 2" and "boom" in a.value.message and not a.value.retryable
    with pytest.raises(LLMError) as b:
        client.complete("m", _user("U"))
    assert b.value.kind == "response" and b.value.code == "unparseable"
    with pytest.raises(LLMError) as c:
        client.complete("m", _user("U"))
    assert c.value.code == codex_cli.NO_REPLY and c.value.usage is not None and c.value.usage.billing == "subscription"


def test_error_events_of_a_completed_run_are_warnings(fake: FakeCodexCli):
    item = success("OK", warnings=["metadata warning"])
    item["events"].insert(2, {"type": "error", "message": "Reconnecting... 1/5"})
    fake.queue(item)
    resp = _client(fake).complete("m", _user("U"))
    assert resp.content == "OK" and resp.raw["warnings"] == ["metadata warning", "Reconnecting... 1/5"]


def test_tools_are_refused_before_anything_runs(fake: FakeCodexCli):
    from ai_eda.llm.client import ToolSpec

    with pytest.raises(ValueError, match="no caller tools"):
        _client(fake).complete("m", _user("U"), tools=[ToolSpec(name="t", description="d", parameters={})])
    assert fake.exec_calls() == []


def test_timeout_kills_the_child_and_is_a_transport_error(fake: FakeCodexCli):
    fake.queue({**success("late"), "sleep": 30})
    with pytest.raises(LLMError) as ei:
        _client(fake, timeout=2).complete("m", _user("U"))
    assert ei.value.kind == "transport" and ei.value.sent is True and "timeout" in ei.value.message


def test_version_and_login_state_read_only_the_measured_line(fake: FakeCodexCli):
    client = _client(fake)
    assert client.version() == "codex-cli 9.9.9 (fake)"
    state = client.login_state()
    assert state.logged_in is True and state.auth_method == "chatgpt" and state.error is None
    fake.set_config(login="Logged in using an API key - sk-proj-***SECRET")
    other = client.login_state()
    assert other.logged_in is None and "unrecognised" in (other.error or "") and "SECRET" not in (other.error or "")
    fake.set_config(login=CHATGPT_LOGIN, login_exit=1)
    assert client.login_state().error == "codex login status exited 1"
    assert [c["argv"] for c in fake.calls()].count(["login", "status"]) == 4  # one at construction, three asked here


def test_a_login_other_than_chatgpt_refuses_the_client(fake: FakeCodexCli):
    """The route is accounted as a subscription only on the measured ChatGPT login: an API-key login (which would bill the key per
    call) or an unreadable state is refused at construction, before any approval is recorded - and the printed line is never echoed."""
    fake.set_config(login="Logged in using an API key - sk-proj-***SECRET")
    with pytest.raises(ToolUnavailableError, match="not logged in with ChatGPT") as ei:
        _client(fake)
    assert "SECRET" not in str(ei.value)
    fake.set_config(login=CHATGPT_LOGIN, login_exit=1)
    with pytest.raises(ToolUnavailableError, match="exited 1"):
        _client(fake)
    assert fake.exec_calls() == []


def test_stream_is_one_shot(fake: FakeCodexCli):
    fake.queue(success("all at once"))
    client = _client(fake)
    assert list(client.stream("m", _user("U"))) == ["all at once"]
    assert client.last_stream_usage is not None and client.last_stream_usage.billing == "subscription"


# --------------------------------------------------------------------------- through the service


def test_service_with_codex_needs_no_budget_and_records_subscription_use(fake: FakeCodexCli):
    from pydantic import BaseModel

    class Word(BaseModel):
        word: str

    gate = ApprovalGate()
    svc = LLMService.from_env(LLMBudget(), router=default_router(default_provider="codex"), providers=("codex",), gate=gate,
                              codex_cli=str(fake.exe))
    assert svc.router.default.spec == f"codex:{DEFAULT_CODEX_MODEL}" and not svc.budget_required
    assert [(e["event"], e["action"], e["by"] if e["event"] == "grant" else None) for e in gate.audit] == [
        ("grant", ExternalAction.SUBSCRIPTION_USE, "cli --llm codex"), ("consume", ExternalAction.SUBSCRIPTION_USE, None),
    ]
    assert gate.audit[0]["detail"] == f"codex subscription via {svc.client.cli} (no per-call cost; usage shown)"
    fake.queue(success_structured({"word": "OK"}))
    word, _ = svc.structured(TaskKind.REQUIREMENT_ANALYSIS, _user("U"), Word)
    assert word.word == "OK"
    [record] = svc.usage.records
    assert record.via == "codex" and record.usage.cost_usd == 0.0 and not record.charge
    assert "1 call(s) on the subscription" in svc.summary()
    # the codex spec's model reaches the CLI as -m
    assert fake.exec_calls()[0]["argv"][7] == DEFAULT_CODEX_MODEL
    assert parse_model_spec("codex:gpt-6-luna").supports_tools is False


def test_api_key_variables_never_reach_the_codex_cli(monkeypatch: pytest.MonkeyPatch):
    """The owner's rule of 2026-11-13 (GPT subscription tokens only, no API charges): no API-key variable reaches the child."""
    from ai_eda.llm.claude_cli import subscription_env
    from ai_eda.llm.codex_cli import CODEX_BILLING_ENV_VARS
    for name in CODEX_BILLING_ENV_VARS:
        monkeypatch.setenv(name, "x")
    env = subscription_env(CODEX_BILLING_ENV_VARS)
    assert not set(CODEX_BILLING_ENV_VARS) & set(env)

