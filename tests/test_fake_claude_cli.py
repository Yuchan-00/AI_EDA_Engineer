"""The fake ``claude`` executable (tests/fake_claude_cli.py) behaves as its docstring says. No login, no network."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from ai_eda.llm.claude_cli import find_claude_cli
from tests.fake_claude_cli import (
    DEFAULT_AUTH,
    DEFAULT_VERSION,
    FAKE_MARKER,
    HELPER_MODEL,
    MARKER_ENV,
    FakeClaudeCli,
    envelope,
    error,
    garbage,
    is_fake_claude_cli,
    missing_keys,
    reply,
    reply_bytes,
    slow,
    success,
    success_structured,
    windows_exe_bytes,
)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaudeCli:
    f = FakeClaudeCli(tmp_path / "bin")
    f.install(monkeypatch)
    return f


def _run(fake: FakeClaudeCli, *args: str, cwd: Path | None = None, stdin: str | None = None, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(fake.exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        cwd=str(cwd) if cwd else None, input=stdin, stdin=None if stdin is not None else subprocess.DEVNULL,
    )


def _run_bytes(
    fake: FakeClaudeCli, *args: str, stdin: bytes, timeout: float = 10.0, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Binary mode both ways, like the client's model call; ``env`` replaces the inherited environment when given."""
    return subprocess.run([str(fake.exe), *args], input=stdin, capture_output=True, timeout=timeout, env=env)


def test_install_puts_an_executable_claude_on_path(fake: FakeClaudeCli):
    assert fake.script.is_file() and (fake.root / FAKE_MARKER).is_file()
    assert os.access(fake.script, os.X_OK)
    assert fake.script.read_text(encoding="utf-8").startswith(f"#!{sys.executable}\n")
    # exe is spelled exactly as discovery returns it (on Windows PATHEXT would spell a bare `claude` lookup claude.EXE)
    assert fake.found == str(fake.exe) and find_claude_cli() == str(fake.exe) and is_fake_claude_cli(fake.exe)
    assert Path(shutil.which("claude") or "").parent == fake.root
    assert "AI_EDA_CLAUDE_CLI" not in os.environ
    assert not list(fake.root.glob("*.cmd")) and not list(fake.root.glob("*.bat"))  # a batch file is refused by the client
    if os.name == "nt":
        assert fake.exe.name == "claude.exe" and fake.exe.read_bytes()[:2] == b"MZ"
    else:
        assert fake.exe == fake.script and not fake.windows_exe.exists()


def test_windows_exe_is_the_console_script_layout():
    """The Windows fake is built the way pip builds a console script: launcher, shebang, zip with ``__main__.py``."""
    data = windows_exe_bytes(b"MZ-launcher-stub", r"C:\Program Files\Python\python.exe", "print('fake')\n")
    assert data.startswith(b"MZ-launcher-stub" + b'#!"C:\\Program Files\\Python\\python.exe"\n\r\n')
    with zipfile.ZipFile(io.BytesIO(data)) as zf:  # a zip reader finds the archive behind the prefix
        assert zf.namelist() == ["__main__.py"] and zf.read("__main__.py") == b"print('fake')\n"
    assert windows_exe_bytes(b"L", r"C:\py\python.exe", "x").startswith(b"L#!C:\\py\\python.exe\n\r\n")  # no space: no quotes


def test_version_and_auth_status_come_from_the_config_not_the_queue(fake: FakeClaudeCli):
    proc = _run(fake, "--version")
    assert proc.returncode == 0 and proc.stdout.strip() == DEFAULT_VERSION
    proc = _run(fake, "auth", "status")
    assert proc.returncode == 0 and json.loads(proc.stdout) == DEFAULT_AUTH
    fake.set_version("1.2.3 (Claude Code)", exit_code=0)
    fake.set_auth({"loggedIn": False, "authMethod": None}, exit_code=0)
    assert _run(fake, "--version").stdout.strip() == "1.2.3 (Claude Code)"
    assert json.loads(_run(fake, "auth", "status").stdout) == {"loggedIn": False, "authMethod": None}
    fake.set_auth("not json", exit_code=3)
    proc = _run(fake, "auth", "status")
    assert proc.returncode == 3 and proc.stdout == "not json"
    assert fake.remaining == 0
    assert [c["argv"] for c in fake.calls()] == [["--version"], ["auth", "status"], ["--version"], ["auth", "status"], ["auth", "status"]]
    assert fake.prompt_calls() == []


def test_prompt_calls_are_answered_from_the_queue_in_order(fake: FakeClaudeCli):
    fake.queue(success("first"), success_structured({"word": "OK"}), error("error_during_execution", "bad", exit_code=1))
    assert fake.remaining == 3
    a = _run(fake, "-p", "one", "--output-format", "json")
    b = _run(fake, "-p", "two", "--output-format", "json")
    c = _run(fake, "-p", "three", "--output-format", "json")
    assert fake.remaining == 0
    ea, eb, ec = json.loads(a.stdout), json.loads(b.stdout), json.loads(c.stdout)
    assert (a.returncode, ea["type"], ea["subtype"], ea["is_error"], ea["result"]) == (0, "result", "success", False, "first")
    assert (b.returncode, eb["result"], eb["structured_output"], eb["stop_reason"], eb["num_turns"]) == (0, '{"word":"OK"}', {"word": "OK"}, "tool_use", 2)
    assert "structured_output" not in ea
    assert (c.returncode, ec["is_error"], ec["subtype"], ec["result"]) == (1, True, "error_during_execution", "bad")
    assert [x["argv"][1] for x in fake.prompt_calls()] == ["one", "two", "three"]


def test_envelope_has_every_measured_key_in_the_measured_order():
    env = envelope("OK")
    assert list(env) == [
        "duration_api_ms", "stop_reason", "session_id", "total_cost_usd", "usage", "modelUsage", "permission_denials",
        "terminal_reason", "fast_mode_state", "fast_mode_disabled_reason", "subagent_stats", "is_error", "num_turns",
        "subtype", "api_error_status", "result", "ttft_ms", "type", "duration_ms", "uuid", "ttft_stream_ms",
        "time_to_request_ms", "first_content_frame_ms", "time_to_request_from_spawn_ms", "warm_spare_claimed",
        "time_origin_ms", "queued_turn_count", "result_index",
    ]
    assert list(env["usage"]) == [
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens", "output_tokens_details",
        "server_tool_use", "service_tier", "cache_creation", "inference_geo", "iterations", "speed",
    ]
    (model, entry), = env["modelUsage"].items()
    assert list(entry) == [
        "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "webSearchRequests", "costUSD",
        "contextWindow", "maxOutputTokens", "thinkingTokens", "canonicalModel", "provider", "costBasis",
    ]
    assert entry["canonicalModel"] == model and env["total_cost_usd"] == entry["costUSD"]
    with_helper = envelope("OK", helper_model=True)
    assert list(with_helper["modelUsage"]) == [HELPER_MODEL, model]
    assert with_helper["total_cost_usd"] == pytest.approx(entry["costUSD"] + 0.000944)
    assert envelope("x", extra_key=1, subtype="odd")["extra_key"] == 1 and envelope("x", subtype="odd")["subtype"] == "odd"


def test_missing_keys_garbage_and_sleep_items(fake: FakeClaudeCli):
    fake.queue(missing_keys("bare"), garbage("nope\n", exit_code=2, stderr="Error: boom\n"), slow(1.5))
    a = _run(fake, "-p", "x")
    assert a.returncode == 0 and json.loads(a.stdout) == {"type": "result", "result": "bare", "is_error": False}
    b = _run(fake, "-p", "y")
    assert (b.returncode, b.stdout, b.stderr) == (2, "nope\n", "Error: boom\n")
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run(fake, "-p", "z", timeout=0.5)
    if os.name != "nt":
        # the timeout fired before the sleep ended. Not on Windows: there subprocess.run collects the pipes after
        # killing the launcher, and the Python child it started may hold them until its own sleep ends
        assert time.monotonic() - t0 < 1.5
    assert fake.remaining == 0


def test_empty_queue_answers_an_error_envelope_with_exit_1(fake: FakeClaudeCli):
    proc = _run(fake, "-p", "anything")
    env = json.loads(proc.stdout)
    assert proc.returncode == 1 and env["is_error"] is True and env["subtype"] == "fake_queue_empty"


def test_every_invocation_is_recorded_with_cwd_entries_env_presence_and_stdin(fake: FakeClaudeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "note.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv(MARKER_ENV, "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake.queue(success("a"), success("b"))
    _run(fake, "-p", "with stdin", "--model", "m", cwd=cwd, stdin="piped text")
    monkeypatch.delenv(MARKER_ENV)
    _run(fake, "-p", "no stdin", cwd=cwd)
    first, second = fake.prompt_calls()
    assert first["argv"] == ["-p", "with stdin", "--model", "m"] and first["cwd"] == str(cwd) and first["cwd_entries"] == ["note.txt"]
    assert first["stdin"] == "piped text" and second["stdin"] == ""
    # a positional prompt still works (the real CLI accepts one); stdin is then only recorded
    assert (first["prompt"], first["prompt_source"], second["prompt"], second["prompt_source"]) == ("with stdin", "argv", "no stdin", "argv")
    assert isinstance(first["pid"], int) and first["pid"] != os.getpid()
    assert first["env"] == {"CLAUDECODE": "CLAUDECODE" in os.environ, "ANTHROPIC_API_KEY": False, MARKER_ENV: True}
    assert second["env"][MARKER_ENV] is False


def test_reset_clears_the_log_and_the_queue(fake: FakeClaudeCli):
    fake.queue(success("a"))
    _run(fake, "--version")
    assert fake.calls() and fake.remaining == 1
    fake.reset()
    assert fake.calls() == [] and fake.remaining == 0


@pytest.mark.skipif(os.name == "nt", reason="the stand-in is a POSIX shell script")
def test_discovery_in_tests_finds_only_the_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """tests/conftest.py hides every ``claude`` but this double from discovery, so ``doctor`` / ``describe_providers``
    in a test never probe the machine's real CLI (``--version``, ``auth status``)."""
    import ai_eda.llm.claude_cli as claude_cli
    from ai_eda.cli import _claude_cli_line

    machine = tmp_path / "machine"
    machine.mkdir()
    stand_in = machine / "claude"
    ran = tmp_path / "ran"
    stand_in.write_text(f"#!/bin/sh\necho \"$@\" >> '{ran}'\n", encoding="utf-8")
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", str(machine))
    monkeypatch.delenv("AI_EDA_CLAUDE_CLI", raising=False)
    assert shutil.which("claude") == str(stand_in) and find_claude_cli() == str(stand_in)  # discovery itself would find it
    assert claude_cli.find_claude_cli() is None  # what the product calls in a test does not
    monkeypatch.setenv("AI_EDA_CLAUDE_CLI", str(stand_in))
    assert claude_cli.find_claude_cli() is None
    assert _claude_cli_line().startswith("claude cli : NOT FOUND (")
    assert not ran.exists()
    fake = FakeClaudeCli(tmp_path / "bin")
    fake.install(monkeypatch)
    assert claude_cli.find_claude_cli() == str(fake.exe)  # the double is found


def test_a_p_call_without_a_positional_prompt_reads_the_prompt_from_stdin_as_exact_bytes(fake: FakeClaudeCli):
    """Like the real CLI (2026-09-28): ``-p`` with no prompt argument takes the prompt from stdin; the log keeps the
    exact bytes (base64, length, sha256) beside the text decoded with replacement."""
    data = "첫 줄 Ω ± µ\r\n둘째 줄\n".encode("utf-8") + b"\xff tail\x00\n"
    fake.queue(success("a"), success("b"))
    proc = _run_bytes(fake, "-p", "--output-format", "json", "--model", "m", "--tools", "", "--system-prompt", "sys", stdin=data)
    assert proc.returncode == 0 and json.loads(proc.stdout.decode("utf-8"))["result"] == "a"
    big = ("동" * 100_000).encode("utf-8")  # 300,000 bytes: several pipe buffers, read until EOF
    _run_bytes(fake, "-p", "--tools", "", stdin=big)
    first, second = fake.prompt_calls()
    assert FakeClaudeCli.stdin_bytes(first) == data and base64.b64decode(first["stdin_b64"]) == data
    assert first["stdin_len"] == len(data) and first["stdin_sha256"] == hashlib.sha256(data).hexdigest()
    assert first["prompt_source"] == "stdin" and first["prompt"] == data.decode("utf-8", "replace") == first["stdin"]
    assert first["argv"] == ["-p", "--output-format", "json", "--model", "m", "--tools", "", "--system-prompt", "sys"]  # no value read as a prompt
    assert FakeClaudeCli.stdin_bytes(second) == big and second["prompt_source"] == "stdin"


def test_probes_record_stdin_but_no_prompt(fake: FakeClaudeCli):
    _run(fake, "--version")
    _run(fake, "auth", "status")
    assert [(c["prompt"], c["prompt_source"], c["stdin_len"]) for c in fake.calls()] == [(None, None, 0), (None, None, 0)]


def test_reply_bytes_writes_exact_bytes_and_ignore_stdin_exits_without_reading(fake: FakeClaudeCli):
    fake.queue(reply_bytes(b"\xff\xfe out \xe9", stderr=b"err \x80\n", exit_code=4), reply(envelope("early"), ignore_stdin=True))
    proc = _run_bytes(fake, "-p", stdin=b"prompt")
    assert (proc.returncode, proc.stdout, proc.stderr) == (4, b"\xff\xfe out \xe9", b"err \x80\n")
    proc = _run_bytes(fake, "-p", stdin=b"x" * 2_000_000)  # the writer sees a broken pipe, which subprocess ignores
    assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == "early"
    first, second = fake.prompt_calls()
    assert first["stdin_len"] == 6 and (second["stdin_len"], second["prompt_source"]) == (0, None)


@pytest.mark.parametrize("encoding", [None, "cp949", "latin-1"])
def test_text_output_is_written_as_utf8_whatever_the_locale(fake: FakeClaudeCli, encoding: str | None):
    """The fake's stdio encoding is forced to cp949 (the Windows PC's code page) and latin-1 through
    ``PYTHONIOENCODING`` (UTF-8 mode off): a fake that printed through its text streams would then write cp949 /
    fail to encode, so the UTF-8 bytes asserted here prove the output does not depend on the locale. ``None`` keeps
    the inherited environment (this suite's own UTF-8 mode)."""
    env = None if encoding is None else {**os.environ, "PYTHONIOENCODING": encoding, "PYTHONUTF8": "0"}
    fake.queue(success("저항 10 kΩ ± 1 %"), reply("plain", stderr="경고 µ\n", exit_code=0))
    proc = _run_bytes(fake, "-p", stdin=b"q", env=env)
    assert json.loads(proc.stdout.decode("utf-8"))["result"] == "저항 10 kΩ ± 1 %"
    proc = _run_bytes(fake, "-p", stdin=b"q", env=env)
    assert proc.stdout == b"plain" and proc.stderr == "경고 µ\n".encode("utf-8")
    fake.set_version("9.9.9 (클로드 코드, fake Ω)")  # the probes' answers too
    fake.set_auth({"loggedIn": True, "authMethod": "메서드"})
    proc = _run_bytes(fake, "--version", stdin=b"", env=env)
    assert proc.returncode == 0 and proc.stdout == "9.9.9 (클로드 코드, fake Ω)\n".encode("utf-8")
    proc = _run_bytes(fake, "auth", "status", stdin=b"", env=env)
    assert proc.returncode == 0 and json.loads(proc.stdout.decode("utf-8"))["authMethod"] == "메서드"
