"""A fake ``codex`` executable for tests: no login, no network, no key.

:class:`FakeCodexCli` writes an executable Python script named ``codex``
into a directory, plus the marker file :data:`FAKE_MARKER` by which
``tests/conftest.py`` lets discovery find it (and nothing else). On Windows
the executable is ``codex.exe``, built like the fake ``claude.exe``
(:func:`tests.fake_claude_cli.windows_exe_bytes`: pip's console-script
launcher + a shebang + a zip); without that launcher the test is skipped.

Like the real CLI (measured, see ``ai_eda/llm/codex_cli.py``), an ``exec``
call with ``-`` reads its prompt from stdin until EOF. Every invocation is
appended to a JSON-lines log - ``argv``, ``cwd``, the cwd's entries, the
exact stdin bytes (``stdin_b64``), and the ``--output-schema`` file's text
(``schema``) - and answered:

* ``--version`` and ``login status`` from the config file (``login status``
  prints on stderr, as measured);
* every ``exec`` call from a queue of scripted replies, in order - each
  ``{"events": [<event objects>], "stdout": <raw text>, "stderr": str,
  "exit": int, "sleep": float}``; the helpers below build the measured
  event streams. An empty queue answers a failed turn with exit 1, so a test
  that under-scripts fails loudly.

The double only proves what the *client* does with the CLI's contract; it
proves nothing about the real CLI (the measurement calls did that).
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.fake_claude_cli import windows_exe_bytes, windows_launcher

#: the file whose presence next to an executable marks it as this fake (tests/conftest.py lets discovery return only such)
FAKE_MARKER = ".fake-codex-cli"
DEFAULT_VERSION = "codex-cli 9.9.9 (fake)"
#: the measured ``codex login status`` line of a ChatGPT login
CHATGPT_LOGIN = "Logged in using ChatGPT"
THREAD_ID = "01a1d05d-f9af-7031-937c-ba4eb500b369"

_SCRIPT = r'''#!{python}
# fake `codex` written by tests/fake_codex_cli.py - standard library only
import base64, json, os, sys, time
LOG = {log!r}
QUEUE = {queue!r}
CONFIG = {config!r}


def _config():
    try:
        with open(CONFIG, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {{}}


def _pop():
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


def _record(argv, raw, schema):
    record = {{
        "argv": argv, "cwd": os.getcwd(), "cwd_entries": sorted(os.listdir(os.getcwd())),
        "stdin_b64": base64.b64encode(raw).decode("ascii"), "schema": schema,
    }}
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _out(stream, text):
    data = text.encode("utf-8")
    if data:
        stream.buffer.write(data)
        stream.flush()


argv = sys.argv[1:]
cfg = _config()
if argv == ["--version"]:
    _record(argv, b"", None)
    _out(sys.stdout, str(cfg.get("version", "")) + "\n")
    sys.exit(int(cfg.get("version_exit", 0)))
if argv[:2] == ["login", "status"]:
    _record(argv, b"", None)
    _out(sys.stderr, str(cfg.get("login", "")) + "\n")
    sys.exit(int(cfg.get("login_exit", 0)))
reply = _pop()
if reply is None:
    reply = {{"events": [{{"type": "error", "message": "fake codex: no scripted reply left"}},
                         {{"type": "turn.failed", "error": {{"message": "fake codex: no scripted reply left"}}}}], "exit": 1}}
raw = b"" if reply.get("ignore_stdin") else sys.stdin.buffer.read()
schema = None
if "--output-schema" in argv:
    with open(argv[argv.index("--output-schema") + 1], "r", encoding="utf-8") as fh:
        schema = fh.read()
_record(argv, raw, schema)
if reply.get("sleep"):
    time.sleep(float(reply["sleep"]))
_out(sys.stderr, str(reply.get("stderr", "")))
if "stdout" in reply:
    _out(sys.stdout, str(reply["stdout"]))
for event in reply.get("events", []):
    _out(sys.stdout, json.dumps(event, ensure_ascii=False) + "\n")
sys.exit(int(reply.get("exit", 0)))
'''


def is_fake_codex_cli(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` is an executable written by :class:`FakeCodexCli` (its directory holds :data:`FAKE_MARKER`)."""
    return (Path(path).parent / FAKE_MARKER).is_file()


def usage(input_tokens: int = 13046, cached: int = 10240, output_tokens: int = 5, reasoning: int = 0) -> dict[str, int]:
    """A ``turn.completed`` usage object in the measured shape."""
    return {"input_tokens": input_tokens, "cached_input_tokens": cached, "cache_write_input_tokens": 0,
            "output_tokens": output_tokens, "reasoning_output_tokens": reasoning}


def success(text: str = "OK", *, usage_obj: dict[str, int] | None = None, warnings: list[str] | None = None, exit_code: int = 0) -> dict[str, Any]:
    """The measured successful event stream: thread.started, turn.started, item.completed (agent_message), turn.completed."""
    events: list[dict[str, Any]] = [{"type": "thread.started", "thread_id": THREAD_ID}]
    events += [{"type": "item.completed", "item": {"id": f"item_w{i}", "type": "error", "message": w}} for i, w in enumerate(warnings or [])]
    events += [
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": usage_obj or usage()},
    ]
    return {"events": events, "exit": exit_code}


def success_structured(obj: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """A success whose agent message is the JSON object text (measured for ``--output-schema``)."""
    return success(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), **kw)


def failed_turn(message: str = "The 'no-such-model-xyz' model is not supported when using Codex with a ChatGPT account.") -> dict[str, Any]:
    """The measured failure: a warning item, an ``error`` event and ``turn.failed`` with the same message; exit 1, no usage."""
    return {"events": [
        {"type": "thread.started", "thread_id": THREAD_ID},
        {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": "Model metadata for `x` not found. Defaulting to fallback metadata"}},
        {"type": "turn.started"},
        {"type": "error", "message": message},
        {"type": "turn.failed", "error": {"message": message}},
    ], "exit": 1}


class FakeCodexCli:
    """The double. ``root`` receives ``codex`` (and ``codex.exe`` on Windows), :data:`FAKE_MARKER`, ``calls.jsonl``,
    ``queue.jsonl`` and ``config.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.script = self.root / "codex"
        self.windows_exe = self.root / "codex.exe"
        self.log_path = self.root / "calls.jsonl"
        self.queue_path = self.root / "queue.jsonl"
        self.config_path = self.root / "config.json"
        text = _SCRIPT.format(python=sys.executable, log=str(self.log_path), queue=str(self.queue_path), config=str(self.config_path))
        self.script.write_text(text, encoding="utf-8", newline="\n")
        self.script.chmod(self.script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if os.name == "nt":
            launcher = windows_launcher()
            if launcher is None:
                pytest.skip("no console-script launcher (pip's distlib t64.exe) to build the fake codex.exe on Windows")
            self.windows_exe.write_bytes(windows_exe_bytes(launcher, sys.executable, text))
        (self.root / FAKE_MARKER).write_text("written by tests/fake_codex_cli.py\n", encoding="utf-8")
        self.log_path.write_text("", encoding="utf-8")
        self.queue_path.write_text("", encoding="utf-8")
        self.set_config(version=DEFAULT_VERSION, login=CHATGPT_LOGIN)

    @property
    def exe(self) -> Path:
        return self.windows_exe if os.name == "nt" else self.script

    def set_config(self, **updates: Any) -> None:
        try:
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        cfg.update(updates)
        self.config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    def queue(self, *items: dict[str, Any]) -> None:
        with self.queue_path.open("a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(ln) for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def exec_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls() if c["argv"][:1] == ["exec"]]

    @staticmethod
    def stdin_bytes(call: dict[str, Any]) -> bytes:
        return base64.b64decode(call["stdin_b64"])


__all__ = [
    "CHATGPT_LOGIN",
    "DEFAULT_VERSION",
    "FAKE_MARKER",
    "THREAD_ID",
    "FakeCodexCli",
    "failed_turn",
    "is_fake_codex_cli",
    "success",
    "success_structured",
    "usage",
]
