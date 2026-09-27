"""ClaudeCodeClient against the fake ``claude`` on PATH (tests/fake_claude_cli.py). No login, no key, no network.

The real CLI is used by nothing here; the two measurement calls that pinned
the envelope shape are recorded in ``ai_eda/llm/claude_cli.py``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import ai_eda.llm.claude_cli as claude_cli
from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.claude_cli import (
    BILLING,
    COST_SOURCE,
    ENV_CLI,
    PROMPT_TOO_LONG,
    ClaudeCodeClient,
    LoginState,
    argv_limit_problem,
    find_claude_cli,
    finish_reason_of,
    parse_usage,
    refuse_batch_file,
    render_prompt,
    sanitise_arg,
    served_model,
)
from ai_eda.llm.client import LLMError, LLMMessage, LLMResponse, ToolSpec, Usage
from tests.fake_claude_cli import (
    DEFAULT_MODEL,
    DEFAULT_VERSION,
    HELPER_MODEL,
    MARKER_ENV,
    FakeClaudeCli,
    envelope,
    error,
    garbage,
    missing_keys,
    reply,
    slow,
    success,
    success_structured,
)

MODEL = DEFAULT_MODEL
SCHEMA = {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"], "additionalProperties": False}
SCHEMA_TEXT = '{"type":"object","properties":{"word":{"type":"string"}},"required":["word"],"additionalProperties":false}'


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaudeCli:
    f = FakeClaudeCli(tmp_path / "bin")
    f.install(monkeypatch)
    return f


@pytest.fixture
def client(fake: FakeClaudeCli):
    c = ClaudeCodeClient(timeout=10.0)
    yield c
    c.close()


def _msgs(text: str = "hello", system: str | None = "You answer tersely.") -> list[LLMMessage]:
    out: list[LLMMessage] = []
    if system:
        out.append(LLMMessage(role="system", content=system))
    out.append(LLMMessage(role="user", content=text))
    return out


# ------------------------------------------------------------------ discovery


def test_discovery_prefers_constructor_then_env_file_then_path(fake: FakeClaudeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert find_claude_cli() == str(fake.exe)
    other = FakeClaudeCli(tmp_path / "other")
    monkeypatch.setenv(ENV_CLI, str(other.exe))
    assert find_claude_cli() == str(other.exe)  # an existing explicit file beats PATH
    assert ClaudeCodeClient().cli == str(other.exe)
    assert ClaudeCodeClient(cli=str(fake.exe)).cli == str(fake.exe)  # the constructor beats the env var
    monkeypatch.setenv(ENV_CLI, str(tmp_path / "missing" / "claude"))
    assert find_claude_cli() == str(fake.exe)  # a missing explicit file falls through to PATH
    assert ClaudeCodeClient().cli == str(fake.exe)


def test_nothing_found_is_tool_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv(ENV_CLI, raising=False)
    assert find_claude_cli() is None
    with pytest.raises(ToolUnavailableError, match=ENV_CLI):
        ClaudeCodeClient()
    monkeypatch.setenv(ENV_CLI, str(tmp_path / "nope"))
    assert find_claude_cli() is None
    with pytest.raises(ToolUnavailableError):
        ClaudeCodeClient()


def test_billing_and_paid_semantics(client: ClaudeCodeClient):
    assert client.billing == BILLING == "subscription"
    assert client.paid is False and ClaudeCodeClient.paid is False
    assert "ClaudeCodeClient(cli=" in repr(client)


# --------------------------------------------------------------------- argv


def test_plain_call_argv_is_the_pinned_command(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("OK"))
    client.complete(MODEL, _msgs("Reply with the single word OK."))
    (call,) = fake.prompt_calls()
    assert call["argv"] == [
        "-p", "Reply with the single word OK.", "--output-format", "json", "--model", MODEL, "--tools", "",
        "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config", "--system-prompt", "You answer tersely.",
    ]
    assert "--bare" not in call["argv"]
    assert call["stdin"] == ""  # nothing piped; that stdin is /dev/null is pinned by test_every_subprocess_gets_stdin_devnull


def test_schema_call_argv_has_json_schema_and_keeps_session_persistence(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": "OK"}))
    resp = client.complete(MODEL, _msgs('Return {"word": "OK"}.'), response_schema=SCHEMA)
    (call,) = fake.prompt_calls()
    assert call["argv"] == [
        "-p", 'Return {"word": "OK"}.', "--output-format", "json", "--model", MODEL, "--tools", "",
        "--setting-sources", "", "--strict-mcp-config", "--system-prompt", "You answer tersely.", "--json-schema", SCHEMA_TEXT,
    ]
    assert "--no-session-persistence" not in call["argv"] and "--bare" not in call["argv"]
    assert json.loads(call["argv"][call["argv"].index("--json-schema") + 1]) == SCHEMA
    assert resp.structured == {"word": "OK"} and resp.content == '{"word":"OK"}'


def test_feedback_turn_resumes_the_session_with_only_the_feedback_text(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": 1}, session_id="00000000-0000-4000-8000-00000000abcd"), success_structured({"word": "OK"}))
    msgs = _msgs("Return the word.")
    first = client.complete(MODEL, msgs, response_schema=SCHEMA)
    assert first.id == "00000000-0000-4000-8000-00000000abcd"
    first_call = fake.prompt_calls()[0]
    assert os.path.isdir(first_call["cwd"])  # kept for the feedback turn
    feedback = [*msgs, LLMMessage(role="assistant", content=first.content or ""), LLMMessage(role="user", content="That was not a string. Fix it.")]
    second = client.complete(MODEL, feedback, response_schema=SCHEMA)
    assert second.structured == {"word": "OK"}
    _, call = fake.prompt_calls()
    assert call["argv"] == [
        "-p", "That was not a string. Fix it.", "--output-format", "json", "--model", MODEL, "--tools", "",
        "--resume", "00000000-0000-4000-8000-00000000abcd", "--setting-sources", "", "--strict-mcp-config",
        "--system-prompt", "You answer tersely.", "--json-schema", SCHEMA_TEXT,
    ]
    assert "--no-session-persistence" not in call["argv"] and "--bare" not in call["argv"]
    assert call["cwd"] == first_call["cwd"] and call["cwd_entries"] == []
    assert not os.path.exists(call["cwd"])  # removed after the feedback turn


@pytest.mark.parametrize("variant", ["other_model", "different_history", "wrong_echo", "no_schema", "plain_first"])
def test_a_turn_that_is_not_the_feedback_turn_starts_a_fresh_session(fake: FakeClaudeCli, client: ClaudeCodeClient, variant: str):
    schema = None if variant == "plain_first" else SCHEMA
    fake.queue(success_structured({"word": 1}), success("again"))
    msgs = _msgs("Return the word.")
    first = client.complete(MODEL, msgs, response_schema=schema)
    first_cwd = fake.prompt_calls()[0]["cwd"]
    echo = LLMMessage(role="assistant", content=first.content or "")
    follow = [*msgs, echo, LLMMessage(role="user", content="feedback")]
    model, schema2 = MODEL, SCHEMA
    if variant == "other_model":
        model = "another-model"
    elif variant == "different_history":
        follow = [LLMMessage(role="system", content="changed"), *follow[1:]]
    elif variant == "wrong_echo":
        follow[-2] = LLMMessage(role="assistant", content="not what the model said")
    elif variant == "no_schema":
        schema2 = None
    client.complete(model, follow, response_schema=schema2)
    _, call = fake.prompt_calls()
    assert "--resume" not in call["argv"]
    assert call["argv"][1].startswith("[user]\nReturn the word.\n\n[assistant]\n")  # the whole conversation, rendered
    assert call["cwd"] != first_cwd and not os.path.exists(first_cwd)  # the kept cwd was dropped
    if schema2 is None:
        assert "--no-session-persistence" in call["argv"]


def test_cwd_is_an_empty_temp_dir_removed_afterwards_and_the_env_passes_through(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MARKER_ENV, "present")
    fake.queue(success("OK"), success("OK"))
    client.complete(MODEL, _msgs())
    (call,) = fake.prompt_calls()
    assert call["cwd_entries"] == [] and not os.path.exists(call["cwd"])
    assert call["cwd"] != os.getcwd() and Path(call["cwd"]).name.startswith("ai-eda-claude-")
    assert call["env"][MARKER_ENV] is True and call["env"]["CLAUDECODE"] == ("CLAUDECODE" in os.environ)
    monkeypatch.delenv(MARKER_ENV)
    client.complete(MODEL, _msgs())
    assert fake.prompt_calls()[1]["env"][MARKER_ENV] is False


def test_schema_call_cwd_is_kept_until_close(fake: FakeClaudeCli):
    c = ClaudeCodeClient(timeout=10.0)
    fake.queue(success_structured({"word": "OK"}))
    c.complete(MODEL, _msgs(), response_schema=SCHEMA)
    (call,) = fake.prompt_calls()
    assert os.path.isdir(call["cwd"])
    c.close()
    assert not os.path.exists(call["cwd"])
    c.close()  # idempotent


def test_budget_and_fallback_flags_are_appended_only_when_configured(fake: FakeClaudeCli):
    c = ClaudeCodeClient(timeout=10.0, max_budget_usd=0.5, fallback_model="alias-a,alias-b")
    fake.queue(success("OK"))
    c.complete(MODEL, _msgs())
    argv = fake.prompt_calls()[0]["argv"]
    assert argv[-4:] == ["--max-budget-usd", "0.5", "--fallback-model", "alias-a,alias-b"]
    c2 = ClaudeCodeClient(timeout=10.0, max_budget_usd=1)
    fake.queue(success("OK"))
    c2.complete(MODEL, _msgs())
    argv = fake.prompt_calls()[1]["argv"]
    assert argv[-2:] == ["--max-budget-usd", "1"] and "--fallback-model" not in argv


def test_no_system_message_means_no_system_prompt_flag(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("OK"))
    client.complete(MODEL, _msgs("hi", system=None))
    argv = fake.prompt_calls()[0]["argv"]
    assert "--system-prompt" not in argv and argv[:2] == ["-p", "hi"]


def test_several_system_messages_are_joined_and_the_conversation_is_rendered_in_blocks():
    system, prompt = render_prompt([
        LLMMessage(role="system", content="A"), LLMMessage(role="user", content="q1"), LLMMessage(role="system", content="B"),
        LLMMessage(role="assistant", content="a1"), LLMMessage(role="user", content="q2"),
    ])
    assert system == "A\n\nB" and prompt == "[user]\nq1\n\n[assistant]\na1\n\n[user]\nq2"
    assert render_prompt([LLMMessage(role="user", content="only")]) == (None, "only")
    with pytest.raises(ValueError):
        render_prompt([])
    with pytest.raises(ValueError):
        render_prompt([LLMMessage(role="system", content="only a system message")])
    with pytest.raises(ValueError, match="no caller tools"):
        render_prompt([LLMMessage(role="user", content="q"), LLMMessage(role="tool", content="r", tool_call_id="x")])


def test_tools_are_refused_before_any_call(fake: FakeClaudeCli, client: ClaudeCodeClient):
    tool = ToolSpec(name="t", description="d", parameters={"type": "object"})
    with pytest.raises(ValueError, match="no caller tools"):
        client.complete(MODEL, _msgs(), tools=[tool])
    with pytest.raises(ValueError):
        client.complete(MODEL, [], tools=None)
    assert fake.calls() == []
    fake.queue(success("OK"))
    client.complete(MODEL, _msgs(), tools=[])  # an empty list is "no tools"
    assert len(fake.prompt_calls()) == 1


def test_temperature_and_max_tokens_are_ignored_and_recorded(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("OK"), success("OK"))
    resp = client.complete(MODEL, _msgs(), temperature=0.7, max_tokens=123)
    argv = fake.prompt_calls()[0]["argv"]
    assert not any("temperature" in a or "max-tokens" in a or "123" == a for a in argv)
    assert resp.raw["temperature_ignored"] is True and resp.raw["max_tokens_ignored"] is True
    resp2 = client.complete(MODEL, _msgs(), temperature=0.0)
    assert resp2.raw["temperature_ignored"] is True and "max_tokens_ignored" not in resp2.raw


# ----------------------------------------------------------- envelope mapping


def test_success_envelope_maps_to_the_response_with_subscription_cost_semantics(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("OK", input_tokens=2, output_tokens=4, cache_creation=1133, cache_read=7, thinking_tokens=3, cost_usd=0.004576, helper_model=True))
    resp = client.complete(MODEL, _msgs())
    assert isinstance(resp, LLMResponse)
    assert resp.model == MODEL and resp.model_used == MODEL and resp.content == "OK" and resp.structured is None
    assert resp.finish_reason == "stop" and resp.native_finish_reason == "success" and not resp.truncated
    assert resp.id == "00000000-0000-4000-8000-000000000001" and resp.provider_name == "firstParty" and resp.via is None
    assert resp.raw_error is None and resp.tool_calls == []
    u = resp.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (2 + 1133 + 7, 4, 2 + 1133 + 7 + 4)
    assert (u.cached_tokens, u.cache_write_tokens, u.reasoning_tokens) == (7, 1133, 3)
    assert u.cost_usd == 0.0 and u.cost_source == COST_SOURCE == "subscription" and u.billing == "subscription"
    assert u.estimated_cost_usd == pytest.approx(0.004576 + 0.000944)  # the CLI's total, both entries, not a charge
    assert u.is_byok is None
    assert resp.raw["type"] == "result" and resp.raw["modelUsage"][HELPER_MODEL]["outputTokens"] == 9


def test_structured_output_is_delivered_and_its_absence_is_flagged(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": "OK"}), success('{"word": "plain"}'), reply(envelope("x", structured_output=[1, 2])))
    resp = client.complete(MODEL, _msgs(), response_schema=SCHEMA)
    assert resp.structured == {"word": "OK"} and resp.raw_error is None and resp.native_finish_reason == "success"
    client.close()
    resp = client.complete(MODEL, _msgs(), response_schema=SCHEMA)
    assert resp.structured is None and resp.content == '{"word": "plain"}'
    assert resp.raw_error is not None and "no structured_output" in resp.raw_error
    client.close()
    resp = client.complete(MODEL, _msgs(), response_schema=SCHEMA)
    assert resp.structured is None and "structured_output is a JSON list" in (resp.raw_error or "")


def test_served_model_rules():
    env = envelope("OK", model="served-id")
    assert served_model(env, "requested") == "served-id"  # exactly one entry
    env = envelope("OK", model=MODEL, helper_model=True)
    assert served_model(env, MODEL) == MODEL  # keyed exactly as requested, beside the helper entry (measured)
    assert served_model(env, "sonnet") == MODEL  # an alias: the entry whose token counts equal the top-level usage
    env["modelUsage"][HELPER_MODEL]["inputTokens"] = env["usage"]["input_tokens"]
    env["modelUsage"][HELPER_MODEL]["outputTokens"] = env["usage"]["output_tokens"]
    env["modelUsage"][HELPER_MODEL]["cacheCreationInputTokens"] = env["usage"]["cache_creation_input_tokens"]
    assert served_model(env, "sonnet") == "sonnet"  # ambiguous: the requested id, never a guess
    assert served_model({"modelUsage": {}}, "m") == "m" and served_model({}, "m") == "m"


@pytest.mark.parametrize(
    "subtype, stop_reason, expected",
    [("success", "end_turn", "stop"), ("success", "tool_use", "stop"), ("success", "max_tokens", "length"),
     ("error_max_turns", None, "length"), ("error_during_execution", None, "error_during_execution"), (None, "end_turn", None)],
)
def test_finish_reason_by_subtype(subtype, stop_reason, expected):
    env: dict = {"stop_reason": stop_reason}
    if subtype is not None:
        env["subtype"] = subtype
    assert finish_reason_of(env) == expected


def test_truncation_reads_the_envelope(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("partial", subtype="success", stop_reason="max_tokens"))
    resp = client.complete(MODEL, _msgs())
    assert resp.truncated and resp.finish_reason == "length" and resp.native_finish_reason == "success"


def test_envelope_missing_keys_is_parsed_defensively(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(missing_keys("bare"))
    resp = client.complete("some-model", _msgs())
    assert resp.content == "bare" and resp.model_used == "some-model" and resp.id is None and resp.provider_name is None
    assert resp.finish_reason is None and resp.native_finish_reason is None and resp.structured is None
    assert resp.usage == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, cost_source="subscription", billing="subscription")
    assert resp.usage.estimated_cost_usd is None and resp.usage.cached_tokens is None and resp.usage.cache_write_tokens is None


def test_parse_usage_arithmetic_and_types():
    u = parse_usage({"usage": {"input_tokens": 2, "cache_creation_input_tokens": 1133, "cache_read_input_tokens": 0, "output_tokens": 4}, "total_cost_usd": 0.00552})
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens, u.estimated_cost_usd) == (1135, 4, 1139, 0.00552)
    u = parse_usage({"usage": {"input_tokens": "2", "output_tokens": True}, "total_cost_usd": "0.1"})
    assert (u.prompt_tokens, u.completion_tokens, u.estimated_cost_usd) == (0, 0, None)
    assert parse_usage({"usage": "nope"}).cost_usd == 0.0


def test_result_that_is_not_a_string_is_a_defect_not_a_crash(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(reply(envelope({"nested": True})))  # type: ignore[arg-type]
    resp = client.complete(MODEL, _msgs())
    assert resp.content is None and "result is a JSON dict" in (resp.raw_error or "")


# ------------------------------------------------------------------- errors


def test_is_error_envelope_is_a_response_error_carrying_usage(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(error("error_during_execution", "the model could not answer", exit_code=1, input_tokens=5, output_tokens=0, cache_creation=100, cost_usd=0.001))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    e = ei.value
    assert e.kind == "response" and e.code == "error_during_execution" and e.status is None and e.sent is True and e.model == MODEL
    assert e.maybe_billed and not e.retryable and e.retry_after is None
    assert e.usage is not None and e.usage.prompt_tokens == 105 and e.usage.cost_usd == 0.0 and e.usage.estimated_cost_usd == 0.001
    assert "the model could not answer" in str(e) and e.metadata["api_error_status"] is None


def test_is_error_without_subtype_uses_is_error_as_code(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(reply({"type": "result", "is_error": True, "result": "nope"}, exit_code=1))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "is_error" and ei.value.kind == "response" and ei.value.usage is None


def test_integer_api_error_status_becomes_the_typed_status(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(error("rate_limited", "slow down", api_error_status=429), error("bad", "no", api_error_status=True))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status == 429 and ei.value.rate_limited and ei.value.retryable and ei.value.code == "rate_limited"
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.status is None  # a boolean is not a status


def test_non_result_type_is_a_response_error(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(reply({"type": "system", "subtype": "init", "is_error": False}))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.kind == "response" and ei.value.code == "init" and ei.value.sent is True


def test_non_zero_exit_with_a_success_envelope_is_not_a_success(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(reply(envelope("OK"), exit_code=3), reply(envelope("OK", subtype="error_odd"), exit_code=4))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    # the code names why the envelope was refused (the exit code), never the "success" subtype it carried
    assert ei.value.kind == "response" and ei.value.code == "exit 3" and "code=success" not in str(ei.value) and "exit 3" in str(ei.value)
    assert ei.value.metadata["subtype"] == "success" and ei.value.sent is True
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "error_odd"  # a subtype other than success still names the error


def test_non_json_stdout_with_non_zero_exit_is_an_exit_error_with_stderr(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(garbage("Some text\n", exit_code=1, stderr="Error: not logged in\nRun claude login\nthird\nfourth\nfifth\nsixth\n"))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    e = ei.value
    assert e.kind == "http" and e.status is None and e.code == "exit 1" and e.sent is None
    assert not e.maybe_billed and not e.retryable
    assert e.message == "Error: not logged in\nRun claude login\nthird\nfourth\nfifth" and "sixth" not in e.message


def test_non_json_stdout_with_exit_zero_is_an_unparseable_response(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(garbage("plain words\n", exit_code=0, stderr=""), reply("", exit_code=0))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.kind == "response" and ei.value.code == "unparseable" and ei.value.sent is True and "plain words" in ei.value.message
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "unparseable" and "(empty)" in ei.value.message


def test_timeout_is_a_transport_error_that_was_sent(fake: FakeClaudeCli):
    c = ClaudeCodeClient(timeout=0.5)
    fake.queue(slow(5))
    with pytest.raises(LLMError) as ei:
        c.complete(MODEL, _msgs())
    e = ei.value
    assert e.kind == "transport" and e.sent is True and e.maybe_billed and e.retryable and "timeout after 0.5s" in e.message
    (call,) = fake.prompt_calls()
    assert not os.path.exists(call["cwd"])


def test_missing_binary_at_call_time_is_a_transport_error_not_sent(fake: FakeClaudeCli, tmp_path: Path):
    gone = tmp_path / "gone" / "claude"
    gone.parent.mkdir()
    gone.write_bytes(fake.script.read_bytes())
    gone.chmod(0o755)
    c = ClaudeCodeClient(cli=str(gone))
    gone.unlink()
    with pytest.raises(LLMError) as ei:
        c.complete(MODEL, _msgs())
    e = ei.value
    assert e.kind == "transport" and e.sent is False and not e.maybe_billed and "FileNotFoundError" in e.message


# ------------------------------------------------------------------ logging


def test_info_logs_carry_accounting_but_never_the_prompt(fake: FakeClaudeCli, client: ClaudeCodeClient, caplog: pytest.LogCaptureFixture):
    fake.queue(success("the secret answer"), error("error_during_execution", "why it failed"))
    with caplog.at_level(logging.INFO, logger="ai_eda.llm.claude_cli"):
        client.complete(MODEL, _msgs("PROMPT-MARKER", system="SYSTEM-MARKER"))
        with pytest.raises(LLMError):
            client.complete(MODEL, _msgs("PROMPT-MARKER", system="SYSTEM-MARKER"))
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "ai_eda.llm.claude_cli")
    assert "PROMPT-MARKER" not in text and "SYSTEM-MARKER" not in text and "the secret answer" not in text and "why it failed" not in text
    assert f"model={MODEL}" in text and "exit=0" in text and "prompt_tokens=1135" in text and "completion_tokens=4" in text
    assert "estimated_cost=0.004576" in text and "elapsed=" in text and "exit=1 code=error_during_execution" in text


# ------------------------------------------------------------------- stream


def test_stream_is_one_shot_with_the_accounting_fields(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("whole reply at once", cost_usd=0.002))
    assert client.last_stream_usage is None and client.last_stream_response is None
    pieces = list(client.stream(MODEL, _msgs(), temperature=0.5, max_tokens=9))
    assert pieces == ["whole reply at once"]
    assert client.last_stream_usage is not None and client.last_stream_usage.estimated_cost_usd == 0.002 and client.last_stream_usage.cost_usd == 0.0
    assert client.last_stream_response is not None and client.last_stream_response.content == "whole reply at once"
    assert client.last_stream_response.model_used == MODEL and client.last_stream_response.finish_reason == "stop"
    (call,) = fake.prompt_calls()
    assert "--no-session-persistence" in call["argv"] and "--json-schema" not in call["argv"]


def test_stream_error_leaves_no_accounting_view(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("first"), error("error_during_execution", "bad"))
    list(client.stream(MODEL, _msgs()))
    assert client.last_stream_response is not None
    with pytest.raises(LLMError):
        list(client.stream(MODEL, _msgs()))
    assert client.last_stream_usage is None and client.last_stream_response is None


def test_stream_with_empty_content_yields_nothing(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success(""))
    assert list(client.stream(MODEL, _msgs())) == []
    assert client.last_stream_response is not None and client.last_stream_response.content == ""


# ------------------------------------------------------------ version / auth


def test_version_is_probed_once_and_cached(fake: FakeClaudeCli, client: ClaudeCodeClient):
    assert client.version() == DEFAULT_VERSION
    assert client.version() == DEFAULT_VERSION
    assert [c["argv"] for c in fake.calls()] == [["--version"]]
    fake.set_version("", exit_code=1)
    c2 = ClaudeCodeClient()
    assert c2.version() is None and c2.version() is None
    assert [c["argv"] for c in fake.calls()] == [["--version"], ["--version"]]


def test_login_state_reads_only_the_measured_keys(fake: FakeClaudeCli, client: ClaudeCodeClient):
    state = client.login_state()
    assert state == LoginState(logged_in=True, auth_method="oauth_token", api_provider="firstParty", error=None)
    assert fake.calls()[-1]["argv"] == ["auth", "status"]
    fake.set_auth({"loggedIn": False, "authMethod": None, "token": "never-read"})
    assert client.login_state() == LoginState(logged_in=False, auth_method=None, api_provider=None, error=None)
    assert "token" not in LoginState.model_fields
    fake.set_auth({"loggedIn": "yes", "authMethod": 3})
    assert client.login_state() == LoginState()
    fake.set_auth("not json", exit_code=2)
    state = client.login_state()
    assert state.logged_in is None and state.error is not None and "exited 2" in state.error
    fake.set_auth({"loggedIn": True}, exit_code=1)
    state = client.login_state()
    assert state.logged_in is True and state.error == "claude auth status exited 1"


def test_probes_never_call_the_model(fake: FakeClaudeCli, client: ClaudeCodeClient):
    client.version()
    client.login_state()
    assert fake.prompt_calls() == [] and fake.remaining == 0


def test_every_subprocess_gets_stdin_devnull(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The ``-p`` call and both probes run with ``stdin=DEVNULL`` (an open stdin makes the CLI wait 3 s - measured).

    Asserted on the ``subprocess.run`` call itself: the fake's recorded stdin
    cannot tell /dev/null from an idle inherited stdin (pytest's own capture
    already puts /dev/null on fd 0).
    """
    seen: list[tuple[list[str], object]] = []
    real = subprocess.run

    def spy(*args, **kwargs):
        seen.append((list(args[0]), kwargs.get("stdin", "<not given>")))
        return real(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    fake.queue(success("OK"))
    client.complete(MODEL, _msgs())
    client.version()
    client.login_state()
    assert [argv[1] for argv, _ in seen] == ["-p", "--version", "auth"]
    assert all(stdin is subprocess.DEVNULL for _, stdin in seen)


# ------------------------------------------------- discovery: paths, Windows


def test_a_relative_cli_path_is_made_absolute_so_calls_run_from_the_empty_temp_cwd(fake: FakeClaudeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    rel = os.path.relpath(fake.exe, tmp_path)  # bin/claude
    fake.queue(success("one"), success("two"), success("three"), success("four"))
    monkeypatch.setenv(ENV_CLI, rel)
    assert find_claude_cli() == str(fake.exe)
    via_env = ClaudeCodeClient(timeout=10.0)
    assert via_env.cli == str(fake.exe) and via_env.complete(MODEL, _msgs()).content == "one"
    explicit = ClaudeCodeClient(cli=rel, timeout=10.0)
    assert explicit.cli == str(fake.exe) and explicit.complete(MODEL, _msgs()).content == "two"
    dotted = ClaudeCodeClient(cli=os.path.join(".", rel), timeout=10.0)
    assert dotted.cli == str(fake.exe) and dotted.complete(MODEL, _msgs()).content == "three"
    bare = ClaudeCodeClient(cli=fake.exe.name, timeout=10.0)  # a bare name resolves on PATH, not against the temp cwd
    assert Path(bare.cli) == fake.exe and bare.complete(MODEL, _msgs()).content == "four"
    monkeypatch.delenv(ENV_CLI)
    monkeypatch.setenv("PATH", "bin")  # a relative PATH entry
    assert find_claude_cli() == str(fake.exe)
    assert len(fake.prompt_calls()) == 4


def test_a_batch_file_is_refused_on_windows_and_passed_elsewhere():
    for name in ("claude.cmd", "claude.CMD", "claude.bat", "claude.Bat"):
        with pytest.raises(ToolUnavailableError, match=f"batch file .*{name}.*cmd.exe.*{ENV_CLI}.*--llm-claude-cli.*claude.exe"):
            refuse_batch_file(os.path.join("C:", "npm", name), windows=True)
        assert refuse_batch_file(name, windows=False) == name
    assert refuse_batch_file(r"C:\\Users\\u\\.local\\bin\\claude.exe", windows=True).endswith("claude.exe")


@pytest.mark.skipif(os.name == "nt", reason="simulates Windows discovery with POSIX executables")
def test_windows_discovery_prefers_claude_exe_and_refuses_the_npm_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ai_eda.cli import _claude_cli_line

    npm = FakeClaudeCli(tmp_path / "npm")  # a `claude` early on PATH, beside the npm shim
    shim = npm.root / "claude.cmd"
    shim.write_bytes(b'@echo off\r\n"node" "cli.js" %*\r\n')
    shim.chmod(0o755)
    native = FakeClaudeCli(tmp_path / "native")
    exe = native.root / "claude.exe"
    exe.write_bytes(native.script.read_bytes())
    exe.chmod(0o755)
    monkeypatch.setattr(claude_cli, "_is_windows", lambda: True)
    monkeypatch.delenv(ENV_CLI, raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(npm.root), str(native.root)]))
    assert find_claude_cli() == str(exe)  # claude.exe first, even behind another `claude` on PATH
    assert ClaudeCodeClient().cli == str(exe)
    monkeypatch.setenv(ENV_CLI, str(shim))
    with pytest.raises(ToolUnavailableError, match="batch file"):
        ClaudeCodeClient()
    with pytest.raises(ToolUnavailableError, match="batch file"):
        ClaudeCodeClient(cli=str(shim))
    assert _claude_cli_line().startswith(f"claude cli : the Claude Code CLI resolved to the batch file {shim}")
    assert npm.calls() == [] and native.calls() == []  # nothing was run, not even a probe


# ------------------------------------------------ arguments the OS must carry


def test_sanitise_arg_rules():
    assert sanitise_arg("a\x00b\x00") == "a b " and sanitise_arg("x\ud800y\udfff") == "x\ufffdy\ufffd"
    assert sanitise_arg("plain [user]\n\"quoted\" %PATH% & | ^ 한글") == "plain [user]\n\"quoted\" %PATH% & | ^ 한글"


@pytest.mark.parametrize("where", ["user", "system", "both"])
def test_a_nul_or_a_lone_surrogate_in_a_message_is_sanitised_not_a_crash(fake: FakeClaudeCli, client: ClaudeCodeClient, where: str):
    user = "Part\x02number\x00XYZ\x03 \ud800" if where in ("user", "both") else "clean"
    system = "sys\x00tem \udfff" if where in ("system", "both") else "You answer tersely."
    fake.queue(success("OK"), success("OK"))
    resp = client.complete(MODEL, _msgs(user, system=system))
    assert resp.content == "OK" and resp.raw["argv_sanitised"] is True
    argv = fake.prompt_calls()[0]["argv"]
    assert "\x00" not in "".join(argv) and argv[1] == sanitise_arg(user) and argv[argv.index("--system-prompt") + 1] == sanitise_arg(system)
    assert "argv_sanitised" not in client.complete(MODEL, _msgs("clean")).raw


def test_the_feedback_turn_text_is_sanitised_too(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": 1}), success_structured({"word": "OK"}))
    msgs = _msgs("Return the word.")
    first = client.complete(MODEL, msgs, response_schema=SCHEMA)
    feedback = [*msgs, LLMMessage(role="assistant", content=first.content or ""), LLMMessage(role="user", content="bad\x00reply")]
    second = client.complete(MODEL, feedback, response_schema=SCHEMA)
    call = fake.prompt_calls()[1]
    assert "--resume" in call["argv"] and call["argv"][1] == "bad reply" and second.raw["argv_sanitised"] is True


def test_an_argument_the_os_cannot_encode_is_an_llm_error_not_a_value_error(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The safety net behind :func:`sanitise_arg`: ``subprocess`` refusing an argument is a typed error, nothing started."""
    monkeypatch.setattr(claude_cli, "sanitise_arg", lambda text: text)
    fake.queue(success("never"))
    for text in ("x\x00y", "x\ud800y"):
        with pytest.raises(LLMError) as ei:
            client.complete(MODEL, _msgs(text))
        assert ei.value.kind == "transport" and ei.value.sent is False and not ei.value.maybe_billed
        assert "ValueError" in ei.value.message or "UnicodeEncodeError" in ei.value.message
    assert fake.prompt_calls() == [] and fake.remaining == 1


def test_argv_limit_problem_per_platform():
    base = ["C:\\claude.exe", "-p"]
    # Windows: the whole command line subprocess builds, plus its NUL, within 32,767 UTF-16 units
    fits = base + ["a" * (32766 - len(subprocess.list2cmdline(base)) - 1)]
    assert len(subprocess.list2cmdline(fits)) == 32766 and argv_limit_problem(fits, {}, platform="windows") is None
    over = base + ["a" * (32767 - len(subprocess.list2cmdline(base)) - 1)]
    problem = argv_limit_problem(over, {}, platform="windows")
    assert problem is not None and "32767 characters" in problem and "at most 32766" in problem and "CreateProcess" in problem
    assert argv_limit_problem(base + ["한" * 20000], {}, platform="windows") is None  # characters, not bytes, count there
    # Linux: one argument's bytes plus its NUL within 32 pages (measured: 131,071 bytes start, 131,072 do not)
    assert argv_limit_problem(["claude", "-p", "a" * 131071], {}, platform="linux", page_size=4096, arg_max=10**9) is None
    problem = argv_limit_problem(["claude", "-p", "a" * 131072], {}, platform="linux", page_size=4096, arg_max=10**9)
    assert problem is not None and problem.startswith("the prompt would be 131072 bytes") and "MAX_ARG_STRLEN" in problem
    problem = argv_limit_problem(["claude", "-p", "x", "--system-prompt", "한" * 43691], {}, platform="linux", page_size=4096, arg_max=10**9)
    assert problem is not None and problem.startswith("argument 4 would be 131073 bytes")
    # every POSIX system: arguments + environment + pointers within ARG_MAX
    assert argv_limit_problem(["claude", "-p", "a" * 1000], {"K": "v" * 1000}, platform="posix", arg_max=4000) is None
    problem = argv_limit_problem(["claude", "-p", "a" * 1000], {"K": "v" * 3000}, platform="posix", arg_max=4000)
    assert problem is not None and "ARG_MAX" in problem and "at most 4000" in problem


@pytest.mark.skipif(not sys.platform.startswith("linux") and os.name != "nt", reason="the per-argument (Linux) / command-line (Windows) limit")
def test_a_prompt_the_os_would_refuse_is_a_typed_error_before_anything_starts(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success("never"))
    hangul = "동" * 60000  # 180,000 UTF-8 bytes: over one Linux argument; 60,000 characters: over the Windows command line
    before = set(Path(os.environ.get("TMPDIR", "/tmp")).glob("ai-eda-claude-*")) if os.name != "nt" else set()
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs(hangul))
    e = ei.value
    assert e.kind == "transport" and e.code == PROMPT_TOO_LONG and e.sent is False and not e.maybe_billed and e.retryable
    assert "nothing was sent" in e.message and ("MAX_ARG_STRLEN" in e.message or "CreateProcess" in e.message)
    assert fake.calls() == [] and fake.remaining == 1
    if os.name != "nt":
        assert set(Path(os.environ.get("TMPDIR", "/tmp")).glob("ai-eda-claude-*")) == before  # no temp cwd was made


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's per-argument limit (E2BIG)")
def test_the_os_refusing_the_command_line_is_the_same_typed_error(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The safety net behind the pre-flight check: E2BIG from the OS names the limit too."""
    monkeypatch.setattr(claude_cli, "argv_limit_problem", lambda argv, env=None, **kw: None)
    fake.queue(success("never"))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs("동" * 60000))
    assert ei.value.code == PROMPT_TOO_LONG and ei.value.sent is False and "too long" in ei.value.message
    assert fake.calls() == []


@pytest.mark.skipif(os.name == "nt", reason="60,000 characters exceed the Windows command line; that refusal is tested above")
def test_a_capped_ascii_fact_prompt_reaches_the_cli(fake: FakeClaudeCli, client: ClaudeCodeClient):
    from ai_eda.llm.prompts import FACT_PROMPT_MAX_CHARS, datasheet_fact_messages

    msgs = datasheet_fact_messages({"ref": "VR1"}, ["Operating voltage 3.3 V typical. " * 3000], ["v_max"])
    assert len(msgs[-1].content or "") > FACT_PROMPT_MAX_CHARS - 1000
    fake.queue(success("OK"))
    assert client.complete(MODEL, msgs).content == "OK"
    assert fake.prompt_calls()[0]["argv"][1] == msgs[-1].content


# ---------------------------------------------------------- through the service


def _mixed_service(fake_timeout: float, budget):
    """The real CLI client (the fake on PATH) as ``claude`` beside a paid scripted ``openrouter``, claude primary,
    the same model on openrouter as the explicit fallback."""
    from ai_eda.llm.fake import ScriptedLLMClient
    from ai_eda.llm.providers import ProviderClient
    from ai_eda.llm.router import default_router
    from ai_eda.llm.service import LLMService
    from ai_eda.llm.usage import UsageTracker
    from ai_eda.security import ApprovalGate

    paid = ScriptedLLMClient(
        [{"content": "from openrouter", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.002, "cost_source": "provider", "billing": "per_call"}}],
        repeat_last=True,
    )
    paid.paid = True
    pc = ProviderClient({"claude": ClaudeCodeClient(timeout=fake_timeout), "openrouter": paid}, default="claude")
    router = default_router(
        "claude:claude-sonnet-5", default_provider="claude", fallback="openrouter:anthropic/claude-sonnet-5",
        by_task={"review": "openrouter:anthropic/claude-sonnet-5"},
    )
    return LLMService(pc, router, UsageTracker(), budget, gate=ApprovalGate()), paid


def test_a_cli_failure_without_usage_is_a_subscription_record_and_the_paid_fallback_is_served(fake: FakeClaudeCli):
    """A claude timeout / unparseable reply carries no usage; it must not become an unknown-cost *charge* that makes a
    USD-only budget refuse the configured openrouter fallback (and every later per-call request)."""
    from ai_eda.llm.router import TaskKind
    from ai_eda.llm.service import LLMBudget

    svc, paid = _mixed_service(0.5, LLMBudget(max_usd=1.0))
    fake.queue(slow(3.0), garbage("plain words\n", exit_code=0, stderr=""))
    resp = svc.complete(TaskKind.CHAT, _msgs())  # the timeout (transport: retryable) falls back to the paid candidate
    assert resp.content == "from openrouter" and resp.via == "openrouter"
    assert [(a.outcome, a.via) for a in svc.last_attempts] == [("transport", "claude"), ("served", "openrouter")]
    with pytest.raises(LLMError) as ei:  # exit 0 with non-JSON stdout: no status, so no fallback - but no charge either
        svc.complete(TaskKind.CHAT, _msgs())
    assert ei.value.code == "unparseable"
    assert svc.complete(TaskKind.REVIEW, _msgs()).via == "openrouter"  # a later per-call request is still covered
    failed = [r for r in svc.usage.records if r.outcome == "failed"]
    assert [(r.via, r.charge, r.usage.cost_usd, r.usage.billing) for r in failed] == [("claude", False, 0.0, "subscription")] * 2
    assert svc.spent() == (pytest.approx(0.004), 0, 30) and len(paid.calls) == 2
    summary = svc.summary()
    assert "unknown cost" not in summary and "possibly billed" not in summary
    assert summary.startswith("2 served call(s), 2 failed call(s) on the subscription, 30 tokens, cost 0.004000 USD; budget max_usd=1")
    assert "2 call(s) on the subscription: 0 tokens, estimated API-equivalent cost unknown (not charged)" in summary


def test_a_cli_timeout_on_the_subscription_only_route_says_so(fake: FakeClaudeCli):
    from ai_eda.llm.router import TaskKind, default_router
    from ai_eda.llm.service import LLMBudget, LLMService
    from ai_eda.llm.usage import UsageTracker
    from ai_eda.security import ApprovalGate

    svc = LLMService(ClaudeCodeClient(timeout=0.5), default_router(default_provider="claude"), UsageTracker(), LLMBudget(), gate=ApprovalGate(), providers=("claude",))
    fake.queue(slow(3.0))
    with pytest.raises(LLMError):
        svc.complete(TaskKind.CHAT, _msgs())
    [rec] = svc.usage.records
    assert rec.via == "claude" and rec.outcome == "failed" and not rec.charge and svc.spent() == (0.0, 0, 0)
    assert svc.summary() == (
        "0 served call(s), 1 failed call(s) on the subscription, 0 tokens, cost 0.000000 USD; budget none (not required: subscription only); "
        "1 call(s) on the subscription: 0 tokens, estimated API-equivalent cost unknown (not charged)"
    )


@pytest.mark.skipif(not sys.platform.startswith("linux") and os.name != "nt", reason="the per-argument (Linux) / command-line (Windows) limit")
def test_a_prompt_too_long_for_the_cli_falls_back_to_a_configured_per_call_candidate(fake: FakeClaudeCli):
    from ai_eda.llm.router import TaskKind
    from ai_eda.llm.service import LLMBudget

    svc, _ = _mixed_service(10.0, LLMBudget(max_usd=1.0))
    resp = svc.complete(TaskKind.CHAT, _msgs("동" * 60000))
    assert resp.via == "openrouter" and fake.calls() == []
    assert [(a.outcome, a.via, a.usage) for a in svc.last_attempts][0] == ("transport", "claude", None)  # nothing sent, nothing recorded
    assert [r.via for r in svc.usage.records] == ["openrouter"]
