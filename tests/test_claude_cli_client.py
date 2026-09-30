"""ClaudeCodeClient against the fake ``claude`` on PATH (tests/fake_claude_cli.py). No login, no key, no network.

The real CLI is used by nothing here; the measurement calls that pinned the
envelope shape (2026-09-26) and the prompt on stdin (2026-09-28) are recorded
in ``ai_eda/llm/claude_cli.py``.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import subprocess
import sys
import threading
import time
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
    encode_prompt,
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
    reply_bytes,
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
        "-p", "--output-format", "json", "--model", MODEL, "--tools", "",
        "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config", "--system-prompt", "You answer tersely.",
    ]
    assert call["argv"] == client.build_argv(MODEL, system="You answer tersely.")[1:]  # build_argv takes no prompt
    assert "--bare" not in call["argv"]
    # the prompt came on stdin, nothing else did (the bytes / kwargs are pinned by the stdin tests below)
    assert call["prompt_source"] == "stdin" and fake.stdin_bytes(call) == b"Reply with the single word OK."


def test_schema_call_argv_has_json_schema_and_keeps_session_persistence(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": "OK"}))
    resp = client.complete(MODEL, _msgs('Return {"word": "OK"}.'), response_schema=SCHEMA)
    (call,) = fake.prompt_calls()
    assert call["argv"] == [
        "-p", "--output-format", "json", "--model", MODEL, "--tools", "",
        "--setting-sources", "", "--strict-mcp-config", "--system-prompt", "You answer tersely.", "--json-schema", SCHEMA_TEXT,
    ]
    assert call["argv"] == client.build_argv(MODEL, system="You answer tersely.", response_schema=SCHEMA)[1:]
    assert fake.stdin_bytes(call) == b'Return {"word": "OK"}.'
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
    assert fake.stdin_bytes(first_call) == b"Return the word."
    feedback = [*msgs, LLMMessage(role="assistant", content=first.content or ""), LLMMessage(role="user", content="That was not a string. Fix it.")]
    second = client.complete(MODEL, feedback, response_schema=SCHEMA)
    assert second.structured == {"word": "OK"}
    _, call = fake.prompt_calls()
    assert call["argv"] == [
        "-p", "--output-format", "json", "--model", MODEL, "--tools", "",
        "--resume", "00000000-0000-4000-8000-00000000abcd", "--setting-sources", "", "--strict-mcp-config",
        "--system-prompt", "You answer tersely.", "--json-schema", SCHEMA_TEXT,
    ]
    # the resumed session already holds the conversation: stdin carries the feedback text and nothing else
    assert call["prompt_source"] == "stdin" and fake.stdin_bytes(call) == b"That was not a string. Fix it."
    assert not any("Fix it" in a or "Return the word" in a for a in call["argv"])
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
    assert call["prompt_source"] == "stdin" and call["prompt"].startswith("[user]\nReturn the word.\n\n[assistant]\n")  # the whole conversation, rendered
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
    (call,) = fake.prompt_calls()
    assert "--system-prompt" not in call["argv"] and call["argv"][:2] == ["-p", "--output-format"] and fake.stdin_bytes(call) == b"hi"


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


def _writer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == claude_cli.STDIN_WRITER_NAME and t.is_alive()]


def test_the_model_call_writes_bytes_to_stdin_and_the_probes_get_devnull(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The ``-p`` call is a ``Popen`` with three binary pipes whose stdin gets the prompt's UTF-8 bytes and is closed;
    both probes are ``subprocess.run`` with ``stdin=DEVNULL`` (an open, unwritten stdin makes the CLI wait 3 s -
    measured).

    Asserted on the calls themselves: text mode (``text=`` / ``encoding=`` /
    ``errors=`` / ``universal_newlines=``) would let Windows translate LF to
    CRLF and the locale code page re-encode the prompt, ``input=`` would hand
    the write to ``communicate`` (unbounded by the timeout on CPython 3.12's
    Windows), and the fake's recorded stdin cannot tell /dev/null from an
    idle inherited stdin.
    """
    ran: list[tuple[list[str], dict]] = []
    started: list[tuple[list[str], dict, object]] = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def run_spy(*args, **kwargs):
        ran.append((list(args[0]), dict(kwargs)))
        return real_run(*args, **kwargs)

    def popen_spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append((list(args[0]), dict(kwargs), proc.stdin))  # the pipe before the client detaches it
        return proc

    monkeypatch.setattr(subprocess, "run", run_spy)
    monkeypatch.setattr(subprocess, "Popen", popen_spy)
    fake.queue(success("OK"))
    client.complete(MODEL, _msgs("한글 prompt\nline two"))
    client.version()
    client.login_state()
    assert [argv[1] for argv, _ in ran] == ["--version", "auth"]  # the model call does not go through run(input=...)
    assert [argv[1] for argv, _, _ in started] == ["-p", "--version", "auth"]  # run() starts the probes through Popen
    _, model_call, pipe = started[0]
    assert model_call["stdin"] is subprocess.PIPE and model_call["stdout"] is subprocess.PIPE and model_call["stderr"] is subprocess.PIPE
    assert not {"text", "encoding", "errors", "universal_newlines", "input"} & set(model_call)
    assert pipe is not None and pipe.closed  # written and closed (the child saw EOF)
    (call,) = fake.prompt_calls()
    assert fake.stdin_bytes(call) == "한글 prompt\nline two".encode("utf-8")
    assert _writer_threads() == []  # the writer thread ended with the call
    for _, kwargs in ran:
        assert kwargs["stdin"] is subprocess.DEVNULL and "input" not in kwargs


def test_the_prompt_is_not_on_the_command_line_and_arrives_on_stdin_byte_exact(fake: FakeClaudeCli, client: ClaudeCodeClient):
    """UTF-8, every line ending exactly as rendered (LF stays LF, a CR the text carried stays), no BOM - and a prompt
    that starts like a flag is never read as one, because it is no argument at all."""
    prompt = '--tools Bash\n저항 R1 = 10 kΩ ± 1 %, C1 = 100 µF\n"quoted" %PATH% & | ^ <>\nwindows line\r\nlast line\n'
    fake.queue(success("OK"))
    assert client.complete(MODEL, _msgs(prompt)).content == "OK"
    (call,) = fake.prompt_calls()
    data = fake.stdin_bytes(call)
    assert data == prompt.encode("utf-8") and call["stdin_len"] == len(prompt.encode("utf-8"))
    assert not data.startswith(b"\xef\xbb\xbf") and data.count(b"\r\n") == 1 and data.count(b"\n") == 5
    assert call["prompt_source"] == "stdin" and call["prompt"] == prompt
    assert not any("저항" in a or "--tools Bash" in a or "last line" in a for a in call["argv"])
    assert call["argv"].count("--tools") == 1 and call["argv"][call["argv"].index("--tools") + 1] == ""
    assert encode_prompt(prompt) == (prompt.encode("utf-8"), False)


def test_a_child_that_exits_without_reading_its_stdin_is_not_an_error(fake: FakeClaudeCli, client: ClaudeCodeClient):
    """A prompt far larger than a pipe buffer, and a child that answers without reading it: the writer thread swallows
    the broken pipe (as ``communicate`` would), and the envelope decides."""
    fake.queue(reply(envelope("answered early"), ignore_stdin=True))
    resp = client.complete(MODEL, _msgs("x" * 2_000_000))
    assert resp.content == "answered early" and resp.raw_error is None
    (call,) = fake.prompt_calls()
    assert call["stdin_len"] == 0 and call["prompt_source"] is None
    assert _writer_threads() == []


def test_a_timeout_while_the_child_does_not_read_stdin_kills_it_and_never_hangs(fake: FakeClaudeCli):
    """The write of a prompt the child never reads cannot block past the timeout; the child is killed and reaped.

    The error is asserted on every platform (the write runs on the client's
    own thread). The time bound and the reaping are asserted off Windows
    only: there the fake is pip's launcher plus the Python child it starts,
    and whether killing the launcher also ends that child - which may hold
    the pipes until its sleep ends - is not measured here.
    """
    c = ClaudeCodeClient(timeout=0.5)
    fake.queue(reply(envelope("late"), sleep=5, ignore_stdin=True))
    t0 = time.monotonic()
    with pytest.raises(LLMError) as ei:
        c.complete(MODEL, _msgs("y" * 2_000_000))
    assert ei.value.kind == "transport" and ei.value.sent is True and "timeout after 0.5s" in ei.value.message
    (call,) = fake.prompt_calls()
    assert not os.path.exists(call["cwd"])
    if os.name != "nt":
        assert time.monotonic() - t0 < 3.0
        with pytest.raises(ProcessLookupError):
            os.kill(call["pid"], 0)  # killed and reaped, not left running
        assert _writer_threads() == []  # the kill broke the pipe: the writer ended


def _cpython312_windows_communicate(self, input, endtime, orig_timeout):
    """CPython 3.12's Windows ``Popen._communicate`` in its own order (Lib/subprocess.py, 3.12.3): the reader threads,
    then ``self._stdin_write(input)`` *blocking in the calling thread*, and only then the joins bounded by the
    timeout. (A later CPython writes from a thread joined with the timeout.) Patched onto the POSIX ``Popen``, it lets
    Linux run the ordering the Windows PC's 3.12 venv runs."""

    def reader(fh, buffer):
        buffer.append(fh.read())
        fh.close()

    if self.stdout and not hasattr(self, "_stdout_buff"):
        self._stdout_buff = []
        self.stdout_thread = threading.Thread(target=reader, args=(self.stdout, self._stdout_buff), daemon=True)
        self.stdout_thread.start()
    if self.stderr and not hasattr(self, "_stderr_buff"):
        self._stderr_buff = []
        self.stderr_thread = threading.Thread(target=reader, args=(self.stderr, self._stderr_buff), daemon=True)
        self.stderr_thread.start()
    if self.stdin:
        self._stdin_write(input)
    if self.stdout is not None:
        self.stdout_thread.join(self._remaining_time(endtime))
        if self.stdout_thread.is_alive():
            raise subprocess.TimeoutExpired(self.args, orig_timeout)
    if self.stderr is not None:
        self.stderr_thread.join(self._remaining_time(endtime))
        if self.stderr_thread.is_alive():
            raise subprocess.TimeoutExpired(self.args, orig_timeout)
    stdout = stderr = None
    if self.stdout:
        stdout = self._stdout_buff
        self.stdout.close()
    if self.stderr:
        stderr = self._stderr_buff
        self.stderr.close()
    return (stdout[0] if stdout else None, stderr[0] if stderr else None)


@pytest.mark.skipif(os.name == "nt", reason="patches CPython 3.12's Windows communicate onto the POSIX Popen")
def test_the_emulated_cpython312_windows_communicate_does_not_bound_a_stdin_write(monkeypatch: pytest.MonkeyPatch):
    """The emulation is faithful where it matters: ``subprocess.run(input=...)`` against a child that never reads a
    2 MB input blocks until that child exits, far past the timeout - what the client must not rely on."""
    monkeypatch.setattr(subprocess.Popen, "_communicate", _cpython312_windows_communicate)
    t0 = time.monotonic()
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", "import time; time.sleep(1.5)"], input=b"y" * 2_000_000, capture_output=True, timeout=0.3)
    assert time.monotonic() - t0 >= 1.2


@pytest.mark.skipif(os.name == "nt", reason="patches CPython 3.12's Windows communicate onto the POSIX Popen")
def test_the_timeout_bounds_the_stdin_write_under_cpython312s_windows_communicate(fake: FakeClaudeCli, monkeypatch: pytest.MonkeyPatch):
    """With 3.12's Windows ordering, a 0.5 s timeout still holds against a child that sleeps without reading a 2 MB
    prompt: the client writes on its own thread and ``communicate`` never touches stdin."""
    monkeypatch.setattr(subprocess.Popen, "_communicate", _cpython312_windows_communicate)
    c = ClaudeCodeClient(timeout=0.5)
    fake.queue(reply(envelope("late"), sleep=5, ignore_stdin=True))
    t0 = time.monotonic()
    with pytest.raises(LLMError) as ei:
        c.complete(MODEL, _msgs("y" * 2_000_000))
    assert time.monotonic() - t0 < 3.0
    assert ei.value.kind == "transport" and ei.value.sent is True and "timeout after 0.5s" in ei.value.message
    (call,) = fake.prompt_calls()
    with pytest.raises(ProcessLookupError):
        os.kill(call["pid"], 0)
    assert _writer_threads() == [] and not os.path.exists(call["cwd"])
    # and a prompt the child does read still arrives whole under the same ordering
    fake.queue(success("OK"))
    assert ClaudeCodeClient(timeout=10.0).complete(MODEL, _msgs("z" * 300_000)).content == "OK"
    assert fake.stdin_bytes(fake.prompt_calls()[-1]) == b"z" * 300_000


def test_an_interrupt_during_the_call_kills_and_reaps_the_child(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """A ``KeyboardInterrupt`` reaches the calling thread (it never sits in a blocking write) and leaves no child."""
    real = subprocess.Popen.communicate
    seen: list[float | None] = []

    def interrupted(self, input=None, timeout=None):
        seen.append(timeout)
        if len(seen) == 1:
            deadline = time.monotonic() + 5.0
            while not fake.prompt_calls() and time.monotonic() < deadline:  # the fake has read its prompt
                time.sleep(0.02)
            raise KeyboardInterrupt
        return real(self, input, timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    fake.queue(reply(envelope("never"), sleep=5))
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        client.complete(MODEL, _msgs("read me"))
    assert time.monotonic() - t0 < 4.0 and seen[0] == client.timeout and len(seen) == 2  # interrupted, then reaped
    (call,) = fake.prompt_calls()
    assert fake.stdin_bytes(call) == b"read me" and not os.path.exists(call["cwd"])
    with pytest.raises(ProcessLookupError):
        os.kill(call["pid"], 0)
    assert _writer_threads() == []


class _Pipe:
    """A stand-in for a child's stdin: ``write`` / ``close`` raise what they are given."""

    def __init__(self, write_error: BaseException | None = None, close_error: BaseException | None = None) -> None:
        self.write_error, self.close_error = write_error, close_error
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.write_error is not None:
            raise self.write_error
        self.data += data
        return len(data)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_feed_stdin_ignores_only_a_broken_pipe_and_always_closes():
    feed = claude_cli._feed_stdin
    for broken in (BrokenPipeError(errno.EPIPE, "gone"), OSError(errno.EINVAL, "Windows: the reader is gone")):
        failures: list[OSError] = []
        pipe = _Pipe(write_error=broken, close_error=broken)
        feed(pipe, b"x", failures)
        assert pipe.closed and failures == []
    failures = []
    pipe = _Pipe()
    feed(pipe, b"abc", failures)
    assert (pipe.data, pipe.closed, failures) == (b"abc", True, [])
    other = OSError(errno.EIO, "disk on fire")
    failures = []
    pipe = _Pipe(write_error=other)
    feed(pipe, b"abc", failures)
    assert pipe.closed and failures == [other]
    failures = []
    feed(_Pipe(close_error=other), b"", failures)
    assert failures == [other]


def test_a_failed_stdin_write_is_named_only_when_the_envelope_does_not_decide(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """A write error other than a broken pipe never becomes a start failure (the child ran): an envelope still
    decides the call; without one, the error names the failed write beside the child's own output."""
    real = claude_cli._feed_stdin

    def failing(pipe, data, failures):
        real(pipe, b"", failures)  # nothing written, stdin closed
        failures.append(OSError(errno.EIO, "disk on fire"))

    monkeypatch.setattr(claude_cli, "_feed_stdin", failing)
    fake.queue(success("fine"), garbage("nope\n", exit_code=2, stderr="Error: empty prompt\n"), garbage("nope\n", exit_code=0, stderr=""))
    assert client.complete(MODEL, _msgs()).content == "fine"
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "exit 2" and ei.value.sent is None
    assert ei.value.message == "Error: empty prompt; writing the prompt to stdin failed (OSError: [Errno 5] disk on fire)"
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "unparseable" and ei.value.message.endswith("; writing the prompt to stdin failed (OSError: [Errno 5] disk on fire)")


def test_non_utf8_bytes_on_stdout_and_stderr_are_replaced_never_raised(fake: FakeClaudeCli, client: ClaudeCodeClient):
    good = json.dumps(envelope("caf__X__"), ensure_ascii=False).encode("utf-8").replace(b"__X__", b"\xe9\xff")
    fake.queue(
        reply_bytes(good, stderr=b"note \xff\xfe\n"),
        reply_bytes(b"\xff\xfe not json\n", stderr=b"Error: \xc3\x28 bad\n", exit_code=1),
        reply_bytes(b"plain \x80 words\n", exit_code=0),
    )
    resp = client.complete(MODEL, _msgs())
    assert resp.content == "caf\ufffd\ufffd"
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.kind == "http" and ei.value.code == "exit 1" and ei.value.message == "Error: \ufffd( bad"
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs())
    assert ei.value.code == "unparseable" and "plain \ufffd words" in ei.value.message


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


def test_encode_prompt_rules():
    assert encode_prompt("a\x00b\x00") == (b"a b ", True)
    assert encode_prompt("x\ud800y\udfff") == ("x\ufffdy\ufffd".encode("utf-8"), True)
    assert encode_prompt("\ud83d\ude00") == ("\ufffd\ufffd".encode("utf-8"), True)  # two lone surrogates, not a pair, in a str
    text = "plain [user]\n\"quoted\" %PATH% & | ^ 한글 Ω ± µ \x02\r\n"
    assert encode_prompt(text) == (text.encode("utf-8"), False)  # other control characters and CRs pass unchanged
    assert encode_prompt("") == (b"", False)


@pytest.mark.parametrize("where", ["user", "system", "both"])
def test_a_nul_or_a_lone_surrogate_in_a_message_is_sanitised_not_a_crash(fake: FakeClaudeCli, client: ClaudeCodeClient, where: str):
    """The prompt (stdin) and the arguments (argv) are sanitised by the same rule but recorded apart:
    ``prompt_sanitised`` for the prompt, ``argv_sanitised`` only for an argument."""
    user = "Part\x02number\x00XYZ\x03 \ud800" if where in ("user", "both") else "clean"
    system = "sys\x00tem \udfff" if where in ("system", "both") else "You answer tersely."
    fake.queue(success("OK"), success("OK"))
    resp = client.complete(MODEL, _msgs(user, system=system))
    assert resp.content == "OK"
    assert resp.raw.get("prompt_sanitised") is (True if where in ("user", "both") else None)
    assert resp.raw.get("argv_sanitised") is (True if where in ("system", "both") else None)
    call = fake.prompt_calls()[0]
    argv = call["argv"]
    assert "\x00" not in "".join(argv) and argv[argv.index("--system-prompt") + 1] == sanitise_arg(system)
    data = fake.stdin_bytes(call)
    assert b"\x00" not in data and data == sanitise_arg(user).encode("utf-8")
    if where in ("user", "both"):
        assert data == "Part\x02number XYZ\x03 \ufffd".encode("utf-8")  # NUL -> space, lone surrogate -> U+FFFD, \x02 / \x03 kept
    clean = client.complete(MODEL, _msgs("clean")).raw
    assert "argv_sanitised" not in clean and "prompt_sanitised" not in clean


def test_the_feedback_turn_text_is_sanitised_too(fake: FakeClaudeCli, client: ClaudeCodeClient):
    fake.queue(success_structured({"word": 1}), success_structured({"word": "OK"}))
    msgs = _msgs("Return the word.")
    first = client.complete(MODEL, msgs, response_schema=SCHEMA)
    feedback = [*msgs, LLMMessage(role="assistant", content=first.content or ""), LLMMessage(role="user", content="bad\x00reply")]
    second = client.complete(MODEL, feedback, response_schema=SCHEMA)
    call = fake.prompt_calls()[1]
    assert "--resume" in call["argv"] and fake.stdin_bytes(call) == b"bad reply" and "bad reply" not in call["argv"]
    assert second.raw["prompt_sanitised"] is True and "argv_sanitised" not in second.raw


def test_an_argument_the_os_cannot_encode_is_an_llm_error_not_a_value_error(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The safety net behind :func:`sanitise_arg`: ``subprocess`` refusing an argument (here the system prompt; the
    prompt itself is no argument) is a typed error, nothing started."""
    monkeypatch.setattr(claude_cli, "sanitise_arg", lambda text: text)
    fake.queue(success("never"))
    for text in ("x\x00y", "x\ud800y"):
        with pytest.raises(LLMError) as ei:
            client.complete(MODEL, _msgs("hi", system=text))
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
    assert "the longest argument being argument 2 with" in problem  # "-p" takes no value: argv[2] is no prompt any more
    assert argv_limit_problem(base + ["한" * 20000], {}, platform="windows") is None  # characters, not bytes, count there
    problem = argv_limit_problem(["C:\\claude.exe", "-p", "--model", "m", "--system-prompt", "S" * 40000], {}, platform="windows")
    assert problem is not None and "the longest argument being the --system-prompt value (argument 5) with 40000 characters" in problem
    # Linux: one argument's bytes plus its NUL within 32 pages (measured: 131,071 bytes start, 131,072 do not)
    assert argv_limit_problem(["claude", "-p", "a" * 131071], {}, platform="linux", page_size=4096, arg_max=10**9) is None
    problem = argv_limit_problem(["claude", "-p", "a" * 131072], {}, platform="linux", page_size=4096, arg_max=10**9)
    assert problem is not None and problem.startswith("argument 2 would be 131072 bytes") and "MAX_ARG_STRLEN" in problem
    assert "prompt" not in problem  # argv[2] is not the prompt: the prompt goes on stdin
    problem = argv_limit_problem(["claude", "-p", "--system-prompt", "한" * 43691], {}, platform="linux", page_size=4096, arg_max=10**9)
    assert problem is not None and problem.startswith("the --system-prompt value (argument 3) would be 131073 bytes")
    # flags and values are walked in order: a value spelled like a flag is still that flag's value
    schema = "{" + "x" * 131072 + "}"
    problem = argv_limit_problem(["claude", "-p", "--system-prompt", "--json-schema", "--json-schema", schema], {}, platform="linux", page_size=4096, arg_max=10**9)
    assert problem is not None and problem.startswith("the --json-schema value (argument 5) would be 131074 bytes")
    # every POSIX system: arguments + environment + pointers within ARG_MAX
    assert argv_limit_problem(["claude", "-p", "a" * 1000], {"K": "v" * 1000}, platform="posix", arg_max=4000) is None
    problem = argv_limit_problem(["claude", "-p", "a" * 1000], {"K": "v" * 3000}, platform="posix", arg_max=4000)
    assert problem is not None and "ARG_MAX" in problem and "at most 4000" in problem


def _nothing_may_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make creating the temporary cwd or starting a process fail the test (a refusal must come before both)."""

    def boom(*args, **kwargs):
        raise AssertionError(f"nothing may start: {args[:1]}")

    monkeypatch.setattr(claude_cli.tempfile, "mkdtemp", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)


def _assert_refused_before_starting(e: LLMError) -> None:
    assert e.kind == "transport" and e.code == PROMPT_TOO_LONG and e.sent is False and not e.maybe_billed and e.retryable
    assert "nothing was sent" in e.message and "the --system-prompt value" in e.message
    # the prompt is on stdin: the message must not claim it is on the command line
    assert "prompt on its command line" not in e.message and "takes the prompt" not in e.message
    assert "the prompt would go on stdin" in e.message


def test_a_system_prompt_too_long_for_the_windows_command_line_is_refused_before_anything_starts(
    fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch,
):
    """Windows' 32,767-character command line still binds the arguments; the check runs on the platform it names."""
    monkeypatch.setattr(claude_cli, "_platform_kind", lambda: "windows")
    _nothing_may_start(monkeypatch)
    fake.queue(success("never"))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs("a short prompt", system="S" * 40000))
    _assert_refused_before_starting(ei.value)
    assert "CreateProcess" in ei.value.message and "with 40000 characters" in ei.value.message
    assert fake.calls() == [] and fake.remaining == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's per-argument limit (MAX_ARG_STRLEN)")
def test_a_system_prompt_over_one_linux_argument_is_refused_before_anything_starts(
    fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch,
):
    _nothing_may_start(monkeypatch)
    fake.queue(success("never"))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs("a short prompt", system="동" * 60000))  # 180,000 UTF-8 bytes in one argument
    _assert_refused_before_starting(ei.value)
    assert "would be 180000 bytes" in ei.value.message and "MAX_ARG_STRLEN" in ei.value.message
    assert fake.calls() == [] and fake.remaining == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's per-argument limit (E2BIG)")
def test_the_os_refusing_the_command_line_is_the_same_typed_error(fake: FakeClaudeCli, client: ClaudeCodeClient, monkeypatch: pytest.MonkeyPatch):
    """The safety net behind the pre-flight check: E2BIG from the OS names the limit too."""
    monkeypatch.setattr(claude_cli, "argv_limit_problem", lambda argv, env=None, **kw: None)
    fake.queue(success("never"))
    with pytest.raises(LLMError) as ei:
        client.complete(MODEL, _msgs("hi", system="동" * 60000))
    assert ei.value.code == PROMPT_TOO_LONG and ei.value.sent is False and "too long" in ei.value.message
    assert fake.calls() == []


def test_a_60000_character_cjk_fact_prompt_reaches_the_cli_on_stdin(fake: FakeClaudeCli, client: ClaudeCodeClient):
    """A capped datasheet-facts prompt of dense multi-byte text: over one Linux argument (131,072 bytes) and over the
    Windows command line (32,767 characters), so the old command-line prompt refused it on both - on stdin it
    arrives whole, and the command line that is left fits both platforms."""
    from ai_eda.llm.prompts import FACT_PROMPT_MAX_CHARS, datasheet_fact_messages

    page = "정격전압삼점삼볼트전형값동작온도영하사십도에서팔십오도까지저항십킬로옴 10 kΩ ± 1 %, 누설전류 1 µA.\n"
    msgs = datasheet_fact_messages({"ref": "U1", "mpn": "부품-123"}, [page * 2000], ["v_max", "operating_temperature"])
    content = msgs[-1].content or ""
    assert len(content) > FACT_PROMPT_MAX_CHARS - 1000 and len(content) > 32767 and len(content.encode("utf-8")) > 131072
    fake.queue(success("OK"))
    assert client.complete(MODEL, msgs).content == "OK"
    (call,) = fake.prompt_calls()
    assert call["prompt_source"] == "stdin" and fake.stdin_bytes(call) == content.encode("utf-8")
    assert not any("정격" in a for a in call["argv"])
    argv = [client.cli, *call["argv"]]
    assert argv_limit_problem(argv, platform="windows") is None
    assert argv_limit_problem(argv, {}, platform="linux", page_size=4096, arg_max=10**9) is None


def test_a_capped_ascii_fact_prompt_reaches_the_cli(fake: FakeClaudeCli, client: ClaudeCodeClient):
    from ai_eda.llm.prompts import FACT_PROMPT_MAX_CHARS, datasheet_fact_messages

    msgs = datasheet_fact_messages({"ref": "VR1"}, ["Operating voltage 3.3 V typical. " * 3000], ["v_max"])
    assert len(msgs[-1].content or "") > FACT_PROMPT_MAX_CHARS - 1000
    fake.queue(success("OK"))
    assert client.complete(MODEL, msgs).content == "OK"
    assert fake.stdin_bytes(fake.prompt_calls()[0]) == (msgs[-1].content or "").encode("utf-8")


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
    """A command line the OS would refuse (here a 60,000-character system prompt: 180,000 bytes in one Linux argument,
    over the Windows command line) is a transport error that sent nothing, so the explicit fallback serves."""
    from ai_eda.llm.router import TaskKind
    from ai_eda.llm.service import LLMBudget

    svc, _ = _mixed_service(10.0, LLMBudget(max_usd=1.0))
    resp = svc.complete(TaskKind.CHAT, _msgs("hi", system="동" * 60000))
    assert resp.via == "openrouter" and fake.calls() == []
    assert [(a.outcome, a.via, a.usage) for a in svc.last_attempts][0] == ("transport", "claude", None)  # nothing sent, nothing recorded
    assert [r.via for r in svc.usage.records] == ["openrouter"]
