"""A fake ``claude`` executable for tests: no login, no network, no key.

:class:`FakeClaudeCli` writes an executable Python script named ``claude``
(a ``#!`` line pointing at the running interpreter, ``chmod +x``) into a
directory to prepend to PATH, plus the marker file :data:`FAKE_MARKER` by
which ``tests/conftest.py`` lets discovery find it (and nothing else). On
Windows the executable is ``claude.exe``: the console-script launcher pip
ships (``pip._vendor.distlib``'s ``t64.exe`` & co., the same stub pip puts in
front of every console script in ``Scripts``) followed by a shebang naming the running
interpreter and a zip whose ``__main__.py`` is the script - never a ``.cmd``
shim, which the client refuses because ``cmd.exe`` would re-parse the
arguments. Without that launcher the fake cannot be built on Windows and the
test is skipped; the Windows path is written but not exercised here (Linux).

Like the real CLI (measured 2026-09-28), a ``-p`` call without a positional
prompt reads its prompt from stdin as UTF-8 *bytes*, until EOF (or
:data:`STDIN_IDLE` seconds of silence - the real CLI gave up after 3 s; the
fake waits longer so a slow machine never cuts a prompt); a positional
prompt still works (the real CLI accepts one), and then stdin is only
recorded. Every invocation is appended to a JSON-lines log - ``argv``,
``cwd``, the cwd's entries, the exact stdin bytes (``stdin_b64``,
``stdin_len``, ``stdin_sha256``, and ``stdin`` decoded as UTF-8 with
replacement), ``prompt`` / ``prompt_source`` (``"argv"`` / ``"stdin"``;
``None`` for the probes), the fake's ``pid``, whether ``CLAUDECODE`` / ``ANTHROPIC_API_KEY`` /
:data:`MARKER_ENV` were in the environment - and answered:

* ``--version`` and ``auth status`` from the config file (:meth:`set_version`,
  :meth:`set_auth`), never from the queue;
* every other call (a ``-p`` call) from a queue of scripted replies, in
  order - each reply is ``{"stdout": <str | JSON object>, "stderr": str,
  "exit": int, "sleep": float}`` (``stdout_b64`` / ``stderr_b64`` write raw
  bytes instead, :func:`reply_bytes`; ``ignore_stdin`` exits without reading
  stdin); the helpers below build the envelopes the real CLI prints
  (measured shape, see ``ai_eda/llm/claude_cli.py``). Every text the fake
  prints (the probes' answers too) is written as UTF-8 bytes whatever its
  stdio encoding (``PYTHONIOENCODING``, the locale, cp949 on the Windows
  PC). An empty queue answers
  an ``is_error`` envelope with subtype ``fake_queue_empty`` and exit 1, so a
  test that under-scripts fails loudly.

The double only proves what the *client* does with the CLI's contract; it
proves nothing about the real CLI (the measurement calls did that).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import shutil
import stat
import struct
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ai_eda.llm.claude_cli import CLI_NAME, WINDOWS_CLI_NAME, absolute_cli_path

#: the file whose presence next to an executable marks it as this fake (tests/conftest.py lets discovery return only such)
FAKE_MARKER = ".fake-claude-cli"
#: an environment variable the tests set to prove passthrough (recorded by presence, never by value)
MARKER_ENV = "AI_EDA_FAKE_CLAUDE_MARKER"
#: env keys whose presence the fake records
RECORDED_ENV_KEYS: tuple[str, ...] = ("CLAUDECODE", "ANTHROPIC_API_KEY", MARKER_ENV)
DEFAULT_VERSION = "9.9.9 (Claude Code, fake)"
DEFAULT_AUTH: dict[str, Any] = {
    "loggedIn": True,
    "authMethod": "oauth_token",
    "apiProvider": "firstParty",
    "analyticsDisabled": False,
    "projectsDirectory": "<fake>",
    "configDirectory": "<fake>",
}
#: the served-model id the fake reports by default (the measured key shape: exactly as requested)
DEFAULT_MODEL = "claude-sonnet-5"
#: the auxiliary entry the real CLI added to ``modelUsage`` in both 2026-09-26 measurement calls (its own request, not the
#: reply; the 2026-09-28 stdin call again had an auxiliary entry, its id not recorded here)
HELPER_MODEL = "claude-haiku-4-5-20251001"
#: flags the fake knows take a value (the argument after one is never read as the positional prompt)
FAKE_VALUE_FLAGS: tuple[str, ...] = (
    "--output-format", "--input-format", "--model", "--tools", "--resume", "-r", "--session-id", "--setting-sources",
    "--system-prompt", "--append-system-prompt", "--json-schema", "--max-budget-usd", "--fallback-model",
    "--permission-mode", "--settings",
)
#: seconds of stdin silence after which a stdin prompt read ends without EOF (the real CLI measured 3 s)
STDIN_IDLE = 10.0

_SCRIPT = r'''#!{python}
# fake `claude` written by tests/fake_claude_cli.py - standard library only
import base64, hashlib, json, os, sys, time
LOG = {log!r}
QUEUE = {queue!r}
CONFIG = {config!r}
ENV_KEYS = {env_keys!r}
VALUE_FLAGS = {value_flags!r}
STDIN_IDLE = {stdin_idle!r}


def _positional(argv):
    # the first positional argument (a prompt given on the command line), flags' values skipped; None when there is none
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in VALUE_FLAGS:
            i += 2
        elif a == "--":
            return argv[i + 1] if i + 1 < len(argv) else None
        elif a.startswith("-") and a != "-":
            i += 1
        else:
            return a
    return None


def _stdin_bytes(wait):
    # wait=True (the prompt comes on stdin): every byte until EOF, or STDIN_IDLE seconds of silence;
    # wait=False: only what is already there (recorded, not used as the prompt)
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return b""
        if os.name != "posix":
            return sys.stdin.buffer.read()
        import select
        fd = sys.stdin.fileno()
        chunks = []
        while True:
            ready, _, _ = select.select([fd], [], [], STDIN_IDLE if wait else 0.2)
            if not ready:
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except (OSError, ValueError):
        return b""


def _config():
    try:
        with open(CONFIG, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {{}}


def _pop_reply():
    try:
        with open(QUEUE, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        lines = []
    if not lines:
        return None
    with open(QUEUE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
    return json.loads(lines[0])


def _record(argv, raw, prompt, source):
    record = {{
        "argv": argv,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "cwd_entries": sorted(os.listdir(os.getcwd())),
        "stdin": raw.decode("utf-8", "replace"),
        "stdin_b64": base64.b64encode(raw).decode("ascii"),
        "stdin_len": len(raw),
        "stdin_sha256": hashlib.sha256(raw).hexdigest(),
        "prompt": prompt,
        "prompt_source": source,
        "env": {{k: (k in os.environ) for k in ENV_KEYS}},
    }}
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _out(stream, text, b64):
    data = base64.b64decode(b64) if b64 is not None else text.encode("utf-8")
    if data:
        stream.buffer.write(data)
        stream.flush()


argv = sys.argv[1:]
cfg = _config()
if "--version" in argv or argv[:2] == ["auth", "status"]:
    _record(argv, _stdin_bytes(False), None, None)
    if "--version" in argv:
        _out(sys.stdout, str(cfg.get("version", "")) + "\n", None)
        sys.exit(int(cfg.get("version_exit", 0)))
    auth = cfg.get("auth")
    _out(sys.stdout, auth if isinstance(auth, str) else json.dumps(auth, indent=2) + "\n", None)
    sys.exit(int(cfg.get("auth_exit", 0)))

reply = _pop_reply()
if reply is None:
    reply = {{
        "stdout": {{"type": "result", "subtype": "fake_queue_empty", "is_error": True, "result": "fake claude: no scripted reply left"}},
        "exit": 1,
    }}
positional = _positional(argv)
if positional is not None:
    raw = _stdin_bytes(False)
    _record(argv, raw, positional, "argv")
elif reply.get("ignore_stdin"):
    _record(argv, b"", None, None)
else:
    raw = _stdin_bytes(True)
    _record(argv, raw, raw.decode("utf-8", "replace"), "stdin")
if reply.get("sleep"):
    time.sleep(float(reply["sleep"]))
err = reply.get("stderr", "")
_out(sys.stderr, str(err), reply.get("stderr_b64"))
out = reply.get("stdout", "")
if not isinstance(out, str):
    out = json.dumps(out, ensure_ascii=False)
_out(sys.stdout, out, reply.get("stdout_b64"))
sys.exit(int(reply.get("exit", 0)))
'''



def is_fake_claude_cli(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` is an executable written by :class:`FakeClaudeCli` (its directory holds :data:`FAKE_MARKER`)."""
    return (Path(path).parent / FAKE_MARKER).is_file()


def windows_launcher() -> bytes | None:
    """pip's console-script launcher for this interpreter (``t64.exe`` / ``t32.exe`` / ``t64-arm.exe``), or ``None``."""
    name = f"t{'64' if struct.calcsize('P') == 8 else '32'}{'-arm' if sysconfig.get_platform() == 'win-arm64' else ''}.exe"
    for package in ("pip._vendor.distlib", "distlib"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        for folder in (spec.submodule_search_locations or []) if spec is not None else []:
            candidate = Path(folder) / name
            if candidate.is_file():
                return candidate.read_bytes()
    return None


def windows_exe_bytes(launcher: bytes, python: str, script: str) -> bytes:
    """A console-script ``.exe`` the way pip / distlib build one: launcher + shebang (``\\n`` then ``\\r\\n``) + a zip with ``__main__.py``.

    The launcher reads the shebang in front of the appended zip, starts that
    interpreter with its own path and the arguments, and Python runs the zip's
    ``__main__.py``.
    """
    exe = f'"{python}"' if " " in python and not python.startswith('"') else python
    shebang = b"#!" + exe.encode("utf-8") + b"\n\r\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("__main__.py", script.encode("utf-8"))
    return launcher + shebang + buf.getvalue()


def envelope(
    result: str = "OK",
    *,
    model: str = DEFAULT_MODEL,
    structured: dict[str, Any] | None = None,
    input_tokens: int = 2,
    output_tokens: int = 4,
    cache_creation: int = 1133,
    cache_read: int = 0,
    thinking_tokens: int = 0,
    cost_usd: float = 0.004576,
    helper_model: bool = False,
    session_id: str = "00000000-0000-4000-8000-000000000001",
    subtype: str = "success",
    is_error: bool = False,
    stop_reason: str | None = "end_turn",
    num_turns: int = 1,
    api_error_status: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A result envelope in the measured shape (every measured key present; ``overrides`` replace / add keys).

    ``helper_model=True`` adds the auxiliary ``modelUsage`` entry the real CLI
    printed beside the served model, and ``total_cost_usd`` then sums both,
    as measured.
    """
    served_entry = {
        "inputTokens": input_tokens, "outputTokens": output_tokens, "cacheReadInputTokens": cache_read,
        "cacheCreationInputTokens": cache_creation, "webSearchRequests": 0, "costUSD": cost_usd,
        "contextWindow": 1000000, "maxOutputTokens": 64000, "thinkingTokens": thinking_tokens,
        "canonicalModel": model, "provider": "firstParty", "costBasis": "list",
    }
    model_usage: dict[str, Any] = {}
    total = cost_usd
    if helper_model:
        model_usage[HELPER_MODEL] = {
            "inputTokens": 899, "outputTokens": 9, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
            "webSearchRequests": 0, "costUSD": 0.000944, "contextWindow": 200000, "maxOutputTokens": 32000,
            "thinkingTokens": 0, "canonicalModel": "claude-haiku-4-5", "provider": "firstParty", "costBasis": "list",
        }
        total = round(total + 0.000944, 9)
    model_usage[model] = served_entry
    env: dict[str, Any] = {
        "duration_api_ms": 1622,
        "stop_reason": stop_reason,
        "session_id": session_id,
        "total_cost_usd": total,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output_tokens,
            "output_tokens_details": {"thinking_tokens": thinking_tokens},
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
            "service_tier": "standard",
            "cache_creation": {"ephemeral_1h_input_tokens": cache_creation, "ephemeral_5m_input_tokens": 0},
            "inference_geo": "not_available",
            "iterations": [{
                "input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": cache_creation}, "type": "message",
            }],
            "speed": "standard",
        },
        "modelUsage": model_usage,
        "permission_denials": [],
        "terminal_reason": "completed",
        "fast_mode_state": "off",
        "fast_mode_disabled_reason": "sdk_opt_in_required",
        "subagent_stats": {"spawned": 0},
        "is_error": is_error,
        "num_turns": num_turns,
        "subtype": subtype,
        "api_error_status": api_error_status,
        "result": result,
    }
    if structured is not None:
        env["structured_output"] = structured
    env.update({
        "ttft_ms": 1237, "type": "result", "duration_ms": 1958, "uuid": "00000000-0000-4000-8000-0000000000aa",
        "ttft_stream_ms": 1172, "time_to_request_ms": 102, "first_content_frame_ms": 1172,
        "time_to_request_from_spawn_ms": 0, "warm_spare_claimed": False, "time_origin_ms": 0, "queued_turn_count": 0, "result_index": 0,
    })
    env.update(overrides)
    return env


def reply(
    stdout: str | dict[str, Any], *, exit_code: int = 0, stderr: str = "", sleep: float = 0.0, ignore_stdin: bool = False,
) -> dict[str, Any]:
    """One queue item. ``ignore_stdin``: the fake exits without reading stdin (a child that never reads its prompt)."""
    item: dict[str, Any] = {"stdout": stdout, "exit": exit_code}
    if stderr:
        item["stderr"] = stderr
    if sleep:
        item["sleep"] = sleep
    if ignore_stdin:
        item["ignore_stdin"] = True
    return item


def reply_bytes(stdout: bytes, *, stderr: bytes = b"", exit_code: int = 0) -> dict[str, Any]:
    """A queue item that writes these exact bytes on stdout / stderr (for output that is not valid UTF-8)."""
    return {"stdout_b64": base64.b64encode(stdout).decode("ascii"), "stderr_b64": base64.b64encode(stderr).decode("ascii"), "exit": exit_code}


def success(result: str = "OK", **kw: Any) -> dict[str, Any]:
    """A success reply (plain text); ``kw`` goes to :func:`envelope`."""
    return reply(envelope(result, **kw))


def success_structured(structured: dict[str, Any], *, result: str | None = None, **kw: Any) -> dict[str, Any]:
    """A success reply with ``structured_output`` (``result`` is its JSON text unless given; measured: ``stop_reason: tool_use``, 2 turns)."""
    kw.setdefault("stop_reason", "tool_use")
    kw.setdefault("num_turns", 2)
    kw.setdefault("output_tokens", 52)
    text = json.dumps(structured, separators=(",", ":")) if result is None else result
    return reply(envelope(text, structured=structured, **kw))


def error(subtype: str = "error_during_execution", message: str = "something failed", *, exit_code: int = 1, **kw: Any) -> dict[str, Any]:
    """An ``is_error: true`` envelope with ``subtype`` (unmeasured on the real CLI; the fake keeps the measured key set)."""
    kw.setdefault("stop_reason", None)
    return reply(envelope(message, subtype=subtype, is_error=True, **kw), exit_code=exit_code)


def missing_keys(result: str = "OK") -> dict[str, Any]:
    """A result envelope with only ``type`` / ``result`` (no usage, no modelUsage, no session_id, no subtype)."""
    return reply({"type": "result", "result": result, "is_error": False})


def garbage(stdout: str = "not json at all\n", *, exit_code: int = 1, stderr: str = "Error: boom\nsecond line\n") -> dict[str, Any]:
    """Non-JSON stdout with a non-zero exit."""
    return reply(stdout, exit_code=exit_code, stderr=stderr)


def slow(seconds: float, **kw: Any) -> dict[str, Any]:
    """A success reply delivered after ``seconds`` (for the timeout case)."""
    return reply(envelope("late", **kw), sleep=seconds)


class FakeClaudeCli:
    """The double. ``root`` receives ``claude`` (and ``claude.exe`` on Windows), :data:`FAKE_MARKER`, ``calls.jsonl``,
    ``queue.jsonl`` and ``config.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.script = self.root / CLI_NAME
        self.windows_exe = self.root / WINDOWS_CLI_NAME
        self.log_path = self.root / "calls.jsonl"
        self.queue_path = self.root / "queue.jsonl"
        self.config_path = self.root / "config.json"
        #: what discovery returned once :meth:`install` put the fake on PATH (``None`` before)
        self.found: str | None = None
        text = _SCRIPT.format(
            python=sys.executable, log=str(self.log_path), queue=str(self.queue_path), config=str(self.config_path),
            env_keys=RECORDED_ENV_KEYS, value_flags=FAKE_VALUE_FLAGS, stdin_idle=STDIN_IDLE,
        )
        self.script.write_text(text, encoding="utf-8", newline="\n")
        self.script.chmod(self.script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if os.name == "nt":
            launcher = windows_launcher()
            if launcher is None:
                pytest.skip("no console-script launcher (pip's distlib t64.exe) to build the fake claude.exe on Windows")
            self.windows_exe.write_bytes(windows_exe_bytes(launcher, sys.executable, text))
        (self.root / FAKE_MARKER).write_text("written by tests/fake_claude_cli.py\n", encoding="utf-8")
        self.reset()
        self.set_version(DEFAULT_VERSION)
        self.set_auth(DEFAULT_AUTH)

    # ------------------------------------------------------------ install

    @property
    def bin_dir(self) -> str:
        """The directory to prepend to PATH."""
        return str(self.root)

    @property
    def exe(self) -> Path:
        """The executable a client resolves: what discovery returned after :meth:`install` (the exact spelling
        ``find_claude_cli`` gives), else the script on POSIX and ``claude.exe`` on Windows."""
        if self.found is not None:
            return Path(self.found)
        return self.windows_exe if os.name == "nt" else self.script

    def install(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Prepend the fake to PATH, clear ``AI_EDA_CLAUDE_CLI`` and record what discovery now finds; returns the new PATH."""
        path = self.bin_dir + os.pathsep + os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", path)
        monkeypatch.delenv("AI_EDA_CLAUDE_CLI", raising=False)
        # the same lookups ai_eda.llm.claude_cli.find_claude_cli makes, so exe is spelled exactly as it returns it
        found = (shutil.which(WINDOWS_CLI_NAME) if os.name == "nt" else None) or shutil.which(CLI_NAME)
        assert found is not None and Path(found).parent == self.root, f"the fake is not what PATH resolves: {found}"
        self.found = absolute_cli_path(found)
        return path

    # ------------------------------------------------------------ scripting

    def reset(self) -> None:
        self.log_path.write_text("", encoding="utf-8")
        self.queue_path.write_text("", encoding="utf-8")

    def queue(self, *items: dict[str, Any]) -> None:
        """Append scripted replies (see :func:`reply` and the helpers) for the next ``-p`` calls, in order."""
        with self.queue_path.open("a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    @property
    def remaining(self) -> int:
        return len([ln for ln in self.queue_path.read_text(encoding="utf-8").splitlines() if ln.strip()])

    def _config(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_config(self, **updates: Any) -> None:
        cfg = self._config()
        cfg.update(updates)
        self.config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    def set_version(self, text: str, exit_code: int = 0) -> None:
        self._write_config(version=text, version_exit=exit_code)

    def set_auth(self, answer: dict[str, Any] | str, exit_code: int = 0) -> None:
        """What ``auth status`` prints: a JSON object, or a raw string (to script non-JSON output)."""
        self._write_config(auth=answer, auth_exit=exit_code)

    # ------------------------------------------------------------ evidence

    def calls(self) -> list[dict[str, Any]]:
        """Every recorded invocation, oldest first (``argv``, ``pid``, ``cwd``, ``cwd_entries``, ``stdin`` + ``stdin_b64`` /
        ``stdin_len`` / ``stdin_sha256``, ``prompt``, ``prompt_source``, ``env``)."""
        text = self.log_path.read_text(encoding="utf-8")
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    @staticmethod
    def stdin_bytes(call: dict[str, Any]) -> bytes:
        """The exact bytes the fake read on stdin in ``call`` (decoded from its ``stdin_b64``)."""
        return base64.b64decode(call["stdin_b64"])

    def prompt_calls(self) -> list[dict[str, Any]]:
        """The recorded ``-p`` invocations only (no ``--version`` / ``auth status``)."""
        return [c for c in self.calls() if "-p" in c["argv"]]


__all__ = [
    "FAKE_MARKER",
    "DEFAULT_AUTH",
    "DEFAULT_MODEL",
    "DEFAULT_VERSION",
    "HELPER_MODEL",
    "MARKER_ENV",
    "RECORDED_ENV_KEYS",
    "STDIN_IDLE",
    "FAKE_VALUE_FLAGS",
    "FakeClaudeCli",
    "envelope",
    "error",
    "garbage",
    "is_fake_claude_cli",
    "missing_keys",
    "reply",
    "reply_bytes",
    "slow",
    "success",
    "success_structured",
    "windows_exe_bytes",
    "windows_launcher",
]
