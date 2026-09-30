"""Claude through the locally installed Claude Code CLI (``claude -p``) - a subprocess, no HTTP, no key.

The CLI uses the login the user already has (a Claude subscription, or an
API key it was set up with); this module never sees a key or a token. It
speaks only the CLI's ``--print`` / ``--output-format json`` contract and
maps one result envelope to one :class:`~ai_eda.llm.client.LLMResponse`.

Invariants enforced here:

* **Only measured envelope keys are parsed.** Every key this module reads
  was observed in the measurement calls below; a key that was seen
  only as ``null`` or not at all is read defensively (missing → ``None``)
  and marked "unmeasured" in this docstring. Nothing about the reply is
  trusted: it is a proposal like every other model output.
* **The subscription login must be reachable**, so ``--bare`` is never
  passed (it reads no OAuth / keychain). CLAUDE.md auto-discovery, hooks,
  plugins and MCP servers are kept out by ``--tools ""``,
  ``--setting-sources ""``, ``--strict-mcp-config`` and by running the CLI
  with ``cwd`` = a fresh empty temporary directory (removed afterwards;
  never the project or the repository).
* **Cost semantics:** a call on the subscription is a *known* zero charge,
  so ``Usage.cost_usd = 0.0`` with ``cost_source = "subscription"`` and
  ``billing = "subscription"``; the envelope's ``total_cost_usd`` (the CLI's
  API-equivalent estimate, not a charge) travels as
  ``Usage.estimated_cost_usd`` so the run can *show* it. It is never
  budgeted. When the CLI is authenticated with an API key instead
  (``login_state().auth_method`` says which), the same number would be a
  real charge on that key - the client cannot tell per call, so ``doctor``
  shows the auth method.
* **No retry / fallback by error class on this route.** The CLI's error text
  is never parsed for retry decisions (the project's rule). An error envelope
  becomes ``LLMError(kind="response", code=<subtype>)`` with
  ``status = api_error_status`` only when the CLI printed an integer there
  (unmeasured: the key was observed as ``null``), so the service's
  status-based rule applies to a typed status and to nothing else; with no
  status such an error is neither retried nor fallen back from. The CLI's own
  ``--fallback-model`` is the user's opt-in (constructor ``fallback_model``),
  off by default.
* **The prompt travels on stdin; only the flags stay on the command line,
  and that command line is still checked before anything runs.** A model
  call runs ``claude -p`` with no prompt argument and writes the rendered
  prompt (on the schema-feedback turn: the feedback text only) to the
  child's stdin, which is closed after writing (measured 2026-09-28, below).
  It is written as UTF-8 *bytes* to a binary pipe (never ``text=True``), so
  neither Windows' text-mode newline translation (LF → CRLF) nor the locale
  code page (cp949 on the PC this was built on) can alter it - a design
  reason, not a Windows measurement; stdout / stderr come back as bytes and
  are decoded here as UTF-8 with ``errors="replace"``. The prompt is made
  encodable first (:func:`encode_prompt`: a lone surrogate - UTF-8 cannot
  carry it - becomes U+FFFD and a NUL byte a space, the rule of
  ``clean_control_chars``; a NUL on the CLI's stdin is unmeasured); the
  response then records ``raw["prompt_sanitised"] = True``. What stays on
  the command line - the model, the system prompt (``--system-prompt``),
  the JSON schema (``--json-schema``) and the other pinned flags - is made
  passable the same way (:func:`sanitise_arg`; ``raw["argv_sanitised"] =
  True`` when an *argument* changed), and a command line the OS would
  refuse is an ``LLMError(kind="transport", code="prompt_too_long",
  sent=False)`` naming the limit (:func:`argv_limit_problem`: on Windows
  CreateProcess takes at most 32,767 characters for the whole command line
  - the message names the longest argument; on Linux one argument may not
  exceed 32 pages, 131,072 bytes with 4 KiB pages - the message names that
  argument, a pinned flag's value by its flag; on every POSIX system the
  arguments plus the environment may not exceed ``ARG_MAX`` - that message
  gives only their total) instead of an opaque ``OSError`` from the OS. The
  code keeps its name, but since the prompt left the command line only a
  very long system prompt or schema (or, on POSIX, an environment that
  nearly fills ``ARG_MAX``) can reach a limit: a ``FACT_PROMPT_MAX_CHARS``
  datasheet-facts prompt is no longer refused on any platform (stdin on
  Windows is unmeasured). The prompt is written by a daemon thread of this
  client while ``communicate`` - handed no stdin - reads stdout / stderr
  (no pipe deadlock), so the call's ``timeout`` bounds the write on every
  platform and interpreter: ``subprocess.run(input=...)`` would not, because
  CPython 3.12's Windows ``communicate`` writes ``input`` in the calling
  thread before its first timeout check, and a child that stalls before
  reading a prompt larger than the pipe buffer would block the call with no
  bound. On expiry (or any other exception, a ``KeyboardInterrupt``
  included) the child is killed and reaped. A child that exits without
  reading its stdin is no error of its own (the writer ignores the broken
  pipe, as ``communicate`` does; the exit code and the envelope decide).
  These three properties are checked against the fake ``claude`` on Linux
  only - the timeout also under an emulation of 3.12's Windows
  ``communicate`` ordering -, not measured with the real CLI.
* **A batch file is never run.** On Windows ``subprocess`` launches a
  ``.cmd`` / ``.bat`` (the npm ``claude.cmd`` shim) through ``cmd.exe``,
  which re-parses the arguments on its command line (the system prompt and
  the JSON schema: quotes, ``%VAR%``, ``&``, ``|``, newlines) - so discovery
  prefers ``claude.exe`` on PATH and a resolved batch file is refused with
  ``ToolUnavailableError`` naming the remedy (point ``AI_EDA_CLAUDE_CLI`` /
  ``--llm-claude-cli`` at the native ``claude.exe``). The path is made
  absolute at construction, because a model call runs with ``cwd`` = the
  empty temporary directory.

Measured on Claude Code 2.1.283, 2026-09-26 (``claude --version`` →
``2.1.283 (Claude Code)``, exit 0; ``claude auth status`` → a JSON object with
the keys ``loggedIn``, ``authMethod``, ``apiProvider``, ``analyticsDisabled``,
``projectsDirectory``, ``configDirectory``, exit 0, values ``loggedIn: true``,
``authMethod: "oauth_token"``, ``apiProvider: "firstParty"``). Two ``-p`` calls
were run from an empty temporary cwd with ``--output-format json --tools ""
--no-session-persistence --setting-sources "" --strict-mcp-config
--system-prompt "You answer tersely." --model claude-sonnet-5``, the second
also with ``--json-schema
'{"type":"object","properties":{"word":{"type":"string"}},"required":["word"],"additionalProperties":false}'``.
Every flag was accepted (nothing was rejected, so nothing was dropped from the
pinned command). Both exited 0 and left the cwd empty. The plain call's stderr
held one line, ``Warning: no stdin data received in 3s, proceeding without it.
If piping from a slow command, redirect stdin explicitly: < /dev/null to skip,
or wait longer.`` (stdin was inherited and open; the CLI waited 3 s) - hence
``stdin=subprocess.DEVNULL`` for the ``--version`` / ``auth status`` probes
below (a model call now writes its prompt to stdin and closes it, so it
never waits: 2026-09-28); the schema call's stderr was empty. Both
printed exactly one JSON object on stdout with these keys, verbatim and in
this order: ``duration_api_ms``, ``stop_reason``, ``session_id``,
``total_cost_usd``, ``usage``, ``modelUsage``, ``permission_denials``,
``terminal_reason``, ``fast_mode_state``, ``fast_mode_disabled_reason``,
``subagent_stats``, ``is_error``, ``num_turns``, ``subtype``,
``api_error_status``, ``result``, [``structured_output`` - schema call only],
``ttft_ms``, ``type``, ``duration_ms``, ``uuid``, ``ttft_stream_ms``,
``time_to_request_ms``, ``first_content_frame_ms``,
``time_to_request_from_spawn_ms``, ``warm_spare_claimed``, ``time_origin_ms``,
``queued_turn_count``, ``result_index``. Values: ``type: "result"``,
``subtype: "success"``, ``is_error: false``, ``api_error_status: null``,
``terminal_reason: "completed"``, ``permission_denials: []``; plain call
``result: "OK"``, ``stop_reason: "end_turn"``, ``num_turns: 1``; schema call
``result: "{\\"word\\":\\"OK\\"}"`` (the JSON text), ``structured_output:
{"word": "OK"}`` (an object), ``stop_reason: "tool_use"``, ``num_turns: 2``.
``usage`` keys: ``input_tokens``, ``cache_creation_input_tokens``,
``cache_read_input_tokens``, ``output_tokens``, ``output_tokens_details``
(``{thinking_tokens}``), ``server_tool_use`` (``{web_search_requests,
web_fetch_requests}``), ``service_tier``, ``cache_creation``
(``{ephemeral_1h_input_tokens, ephemeral_5m_input_tokens}``), ``inference_geo``,
``iterations`` (a list of per-request objects), ``speed``; the plain call read
``input_tokens: 2, cache_creation_input_tokens: 1133, cache_read_input_tokens:
0, output_tokens: 4`` - ``input_tokens`` counts only the uncached part of the
prompt, so this module's ``prompt_tokens`` is the sum of the three input
counts. ``modelUsage`` maps a model id to ``{inputTokens, outputTokens,
cacheReadInputTokens, cacheCreationInputTokens, webSearchRequests, costUSD,
contextWindow, maxOutputTokens, thinkingTokens, canonicalModel, provider,
costBasis}`` and held TWO entries in both calls: ``claude-sonnet-5`` (the
served model, keyed exactly as requested, ``canonicalModel: "claude-sonnet-5"``,
``provider: "firstParty"``, token counts equal to the top-level ``usage``) and
``claude-haiku-4-5-20251001`` (an auxiliary request the CLI made itself, about
900 input / 10 output tokens, ``cacheCreationInputTokens: 0``). ``total_cost_usd``
(0.00552 / 0.007887) was the sum of both entries' ``costUSD``. ``session_id``
and ``uuid`` were UUID strings; run nested inside a Claude Code session (the
environment carried ``CLAUDECODE=1`` and ``CLAUDE_CODE_SESSION_ID``) the
``session_id`` equalled the inherited session id rather than a fresh one -
which variable causes that is not measured, and a feedback turn's
``--resume`` from inside such a nested session is therefore unmeasured.

Measured on Claude Code 2.1.283, 2026-09-28 (Linux), the prompt on stdin: the
2026-09-26 schema call's flags exactly (the same ``--model`` as the
2026-09-26 calls; a ``--json-schema`` with the string properties ``first`` /
``middle`` / ``last`` and the integer ``last_row``) with NO prompt argument
after ``-p``; the prompt was written to stdin as UTF-8 and stdin was closed
after writing (``subprocess.run(input=...)``), cwd = a fresh empty temporary
directory. The prompt: 45,134 characters = 58,721 UTF-8 bytes = 45,134 UTF-16
units (longer than Windows' 32,767-character command line), 647 numbered rows
mixing ASCII with Korean text and the non-ASCII symbols Ω, ±, µ, with the
markers ``CODE_FIRST`` / ``CODE_MIDDLE`` / ``CODE_LAST`` at its start, middle
and end. Result: exit 0, stderr empty (no "no stdin data received in 3s"
warning), the cwd left empty, 4.89 s; one JSON envelope with exactly the same
29 keys in the same order as the 2026-09-26 schema call; ``type: "result"``,
``subtype: "success"``, ``is_error: false``, ``api_error_status: null``,
``stop_reason: "tool_use"``, ``num_turns: 2``, ``terminal_reason:
"completed"``; ``structured_output`` held all three markers and ``last_row:
647`` exactly (the whole stdin reached the model, nothing was cut).
``usage``: ``input_tokens: 2, cache_creation_input_tokens: 37959,
cache_read_input_tokens: 0, output_tokens: 136``. ``modelUsage`` again held
two entries: the requested model (``inputTokens`` 2,
``cacheCreationInputTokens`` 37959, ``outputTokens`` 136, ``costUSD``
0.1532) and the CLI's auxiliary model entry, which this time read 37,188
input tokens (``outputTokens`` 22, ``costUSD`` 0.037298) - the CLI's
auxiliary request processed the whole prompt too (with the tiny 2026-09-26
prompts it read about 900); ``total_cost_usd`` 0.190498 was the sum of both
entries.

Unmeasured (parsed or handled defensively): an ``is_error: true`` envelope,
an integer ``api_error_status``, any ``subtype`` other than ``success``, a
``stop_reason`` of ``max_tokens``, ``--resume`` (and so the feedback text on
stdin of a ``--resume`` turn), a non-zero exit code, the CLI on Windows
(native ``claude.exe`` or otherwise, including whether it reads UTF-8
stdin), a NUL byte or a lone surrogate on stdin (both are replaced before
writing), whether the auxiliary request always re-reads the whole prompt,
and a real CLI that stalls, exits without reading stdin or is killed on the
timeout (whether killing it also ends any process it started is unknown:
the reap then waits at most :data:`REAP_TIMEOUT` per step).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator

from pydantic import BaseModel

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolSpec, Usage

#: environment variable naming the CLI binary explicitly (used when it names an existing file)
ENV_CLI = "AI_EDA_CLAUDE_CLI"
#: the executable looked up on PATH (on Windows ``claude.exe`` is looked up first; ``claude`` then resolves through PATHEXT)
CLI_NAME = "claude"
#: the native Windows executable, preferred over whatever PATHEXT resolves ``claude`` to (the npm shim is ``claude.cmd``)
WINDOWS_CLI_NAME = "claude.exe"
#: suffixes Windows runs through ``cmd.exe`` whatever ``shell=`` says (it would re-parse the arguments): never executed
BATCH_SUFFIXES: tuple[str, ...] = (".cmd", ".bat")
#: how a subscription-billed call is accounted: a known zero charge, the estimate shown beside it
COST_SOURCE = "subscription"
BILLING = "subscription"
#: longest stderr / envelope excerpt kept in an :class:`LLMError` message
ERROR_TEXT_LIMIT = 2000
#: seconds allowed for ``--version`` / ``auth status``
PROBE_TIMEOUT = 30.0
#: seconds allowed, once the child is gone (or killed on the timeout), to collect its pipes and to let the stdin writer
#: thread finish; only a process the child started that still holds a pipe can use them up (the child is reaped anyway)
REAP_TIMEOUT = 5.0
#: the name of the thread that writes a model call's prompt to the child's stdin
STDIN_WRITER_NAME = "claude-cli-stdin"
#: CreateProcess's limit for the whole command line, in UTF-16 code units including the terminating NUL
WINDOWS_COMMAND_LINE_LIMIT = 32767
#: Linux's MAX_ARG_STRLEN in pages: one argument (its bytes and its NUL) may not exceed 32 pages
LINUX_ARG_PAGES = 32
#: the page size assumed when ``sysconf`` cannot say
DEFAULT_PAGE_SIZE = 4096
#: the ``LLMError.code`` of a command line the OS would refuse (nothing was sent). The name predates the prompt's
#: move to stdin and is kept: only the arguments (a system prompt, a JSON schema, the flags) can reach a limit now -
#: with the environment, on POSIX, for ``ARG_MAX``
PROMPT_TOO_LONG = "prompt_too_long"
#: the pinned flags that take a value: the argument after one is that flag's value (names it in a limit message)
VALUE_FLAGS: tuple[str, ...] = (
    "--output-format", "--model", "--tools", "--resume", "--setting-sources", "--system-prompt", "--json-schema",
    "--max-budget-usd", "--fallback-model",
)
#: Windows' ERROR_FILENAME_EXCED_RANGE, what CreateProcess raises for an overlong command line
_WINERROR_TOO_LONG = 206

log = logging.getLogger("ai_eda.llm.claude_cli")


def _is_windows() -> bool:
    return os.name == "nt"


def absolute_cli_path(path: str) -> str:
    """``path`` as an absolute path: kept when absolute, a bare name resolved on PATH, anything else against the cwd.

    A model call runs with ``cwd`` = an empty temporary directory, where a
    relative executable path would no longer resolve; a bare name that PATH
    does not know is returned unchanged (the call then fails as a missing
    binary).
    """
    if os.path.isabs(path):
        return path
    if not os.path.dirname(path):
        found = shutil.which(path)
        return os.path.abspath(found) if found else path
    return os.path.abspath(path)


def refuse_batch_file(path: str, *, windows: bool | None = None) -> str:
    """``path`` unchanged, or :class:`~ai_eda.errors.ToolUnavailableError` when Windows would run it through ``cmd.exe``.

    ``subprocess`` on Windows launches a ``.cmd`` / ``.bat`` through the
    command interpreter regardless of ``shell=False``, and ``cmd.exe``
    re-parses the arguments (quotes, ``%VAR%``, ``&``, ``|``, newlines)
    without any escaping - the system prompt and the JSON schema on the
    command line would be mangled or executed (the prompt itself goes on
    stdin, but what a batch file does with it is not this module's contract
    either). ``windows`` defaults to the running platform (tests pass it
    explicitly).
    """
    if (_is_windows() if windows is None else windows) and Path(path).suffix.lower() in BATCH_SUFFIXES:
        raise ToolUnavailableError(
            f"the Claude Code CLI resolved to the batch file {path}: Windows runs a .cmd/.bat through cmd.exe, which "
            "re-parses the arguments on the command line (the system prompt, the JSON schema: quotes, %VAR%, &, |, newlines) - refused. Set "
            f"{ENV_CLI} or --llm-claude-cli to the native {WINDOWS_CLI_NAME} (the native installer's, or the one in "
            "the npm package's bin directory)"
        )
    return path


def find_claude_cli() -> str | None:
    """The CLI binary as an absolute path: ``$AI_EDA_CLAUDE_CLI`` when it names an existing file, else ``claude`` on PATH, else ``None``.

    Like ``AI_EDA_BROWSER`` for the report printer, an explicit path that does
    not exist falls through to the PATH lookup rather than aborting. On
    Windows ``claude.exe`` is looked up before ``claude`` (which PATHEXT may
    resolve to the npm ``claude.cmd`` shim - :func:`refuse_batch_file` refuses
    that at construction, and says so in ``doctor``).
    """
    explicit = os.environ.get(ENV_CLI, "").strip()
    if explicit and Path(explicit).is_file():
        return absolute_cli_path(explicit)
    found = (shutil.which(WINDOWS_CLI_NAME) if _is_windows() else None) or shutil.which(CLI_NAME)
    return absolute_cli_path(found) if found else None


#: a lone UTF-16 surrogate (a ``str`` can hold one; no encoding can carry it to a child process)
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _passable(text: str) -> str:
    """NUL → space (the rule of ``clean_control_chars``), every lone surrogate → U+FFFD."""
    if "\x00" in text:
        text = text.replace("\x00", " ")
    return _SURROGATE_RE.sub("\ufffd", text)


def sanitise_arg(text: str) -> str:
    """``text`` as an argument the OS can pass: a NUL byte becomes a space (the rule of ``clean_control_chars``), a lone surrogate U+FFFD.

    ``execve`` / ``CreateProcess`` cannot carry a NUL, and a lone surrogate
    has no UTF-8 encoding; both would otherwise escape as ``ValueError``
    before any child exists.
    """
    return _passable(text)


def encode_prompt(prompt: str) -> tuple[bytes, bool]:
    """``(stdin bytes, sanitised)``: ``prompt`` as the UTF-8 bytes written to the CLI's stdin, and whether it had to change.

    A lone surrogate becomes U+FFFD (UTF-8 cannot carry it, so the encode
    would raise before any child exists) and a NUL byte a space (the rule of
    ``clean_control_chars``; how the CLI reads a NUL on stdin is
    unmeasured). Nothing else is touched: no BOM is added and line endings
    stay exactly as rendered (the bytes are written in binary mode, so
    Windows' text-mode LF → CRLF translation never applies).
    """
    clean = _passable(prompt)
    return clean.encode("utf-8"), clean != prompt


def _broken_pipe(e: OSError) -> bool:
    """Whether ``e`` only says the reader of a pipe is gone: ``BrokenPipeError`` (EPIPE) on POSIX, EINVAL on Windows
    (a child that exited or closed its stdin) - the two cases ``Popen.communicate`` ignores on a stdin write."""
    return isinstance(e, BrokenPipeError) or e.errno == errno.EINVAL


def _feed_stdin(pipe: IO[bytes], data: bytes, failures: list[OSError]) -> None:
    """Write ``data`` to ``pipe`` (a child's stdin) and close it - run on its own thread, never the caller's.

    A broken pipe (:func:`_broken_pipe`: the child exited or closed its stdin
    before reading everything) is no error of its own, as in
    ``Popen.communicate``: the exit code and the envelope decide. Any other
    ``OSError`` of the write or the close is appended to ``failures`` for the
    caller to report; the pipe is closed whatever happened, so the child
    always sees EOF.
    """
    try:
        if data:
            pipe.write(data)
    except OSError as e:
        if not _broken_pipe(e):
            failures.append(e)
    finally:
        try:
            pipe.close()
        except OSError as e:
            if not _broken_pipe(e):
                failures.append(e)


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """After ``proc.kill()``: collect the rest of its pipes and wait for it, each within :data:`REAP_TIMEOUT`.

    The killed child itself exits at once; only a process it started that
    still holds stdout / stderr can keep the pipes open, and then the child
    is waited for alone (the pipes' readers end when that process does).
    """
    try:
        proc.communicate(timeout=REAP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            proc.wait(timeout=REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            log.info("claude-cli: the killed child %s did not exit within %gs", proc.pid, REAP_TIMEOUT)


def _platform_kind() -> str:
    if _is_windows():
        return "windows"
    return "linux" if sys.platform.startswith("linux") else "posix"


def _sysconf(name: str) -> int | None:
    try:
        value = os.sysconf(name)
    except (AttributeError, ValueError, OSError):
        return None
    return value if isinstance(value, int) and value > 0 else None


def _argument_name(argv: list[str], i: int) -> str:
    """``argv[i]`` named for a limit message: ``the --system-prompt value (argument 12)`` when a :data:`VALUE_FLAGS` flag
    takes it (flags and their values walked in order, so a value that looks like a flag is still a value), else ``argument i``."""
    j = 1
    while j < len(argv):
        if argv[j] in VALUE_FLAGS and j + 1 < len(argv):
            if j + 1 == i:
                return f"the {argv[j]} value (argument {i})"
            j += 2
        else:
            j += 1
    return f"argument {i}"


def argv_limit_problem(
    argv: list[str],
    env: dict[str, str] | None = None,
    *,
    platform: str | None = None,
    page_size: int | None = None,
    arg_max: int | None = None,
) -> str | None:
    """Why the OS would refuse to start ``argv`` (with ``env``), or ``None`` when it fits.

    ``windows``: the command line ``subprocess`` builds
    (``subprocess.list2cmdline``) plus its NUL may not exceed
    :data:`WINDOWS_COMMAND_LINE_LIMIT` UTF-16 code units; the message names
    the longest argument. ``linux``: one argument's UTF-8 bytes plus its NUL
    may not exceed :data:`LINUX_ARG_PAGES` pages (MAX_ARG_STRLEN; measured
    here: 131,071 bytes start, 131,072 do not); the message names that
    argument (a pinned flag's value by its flag). Every POSIX system
    (``linux`` and ``posix``): the arguments, the environment and their
    pointers may not exceed ``ARG_MAX``. ``platform``, ``page_size`` and
    ``arg_max`` default to the running system's (tests pass them). The
    prompt is not an argument (it goes on stdin), so none of these limits
    applies to it.
    """
    kind = platform or _platform_kind()
    if kind == "windows":
        line = subprocess.list2cmdline(argv)
        units = len(line.encode("utf-16-le", "surrogatepass")) // 2 + 1
        if units > WINDOWS_COMMAND_LINE_LIMIT:
            widths = [len(a.encode("utf-16-le", "surrogatepass")) // 2 for a in argv]
            longest = max(range(1, len(argv)), key=lambda k: widths[k], default=None)
            which = "" if longest is None else f", the longest argument being {_argument_name(argv, longest)} with {widths[longest]} characters"
            return f"the command line would be {units - 1} characters{which}; Windows (CreateProcess) takes at most {WINDOWS_COMMAND_LINE_LIMIT - 1}"
        return None
    sizes = [len(a.encode("utf-8", "surrogatepass")) + 1 for a in argv]
    if kind == "linux":
        limit = LINUX_ARG_PAGES * (page_size or _sysconf("SC_PAGE_SIZE") or DEFAULT_PAGE_SIZE)
        for i, size in enumerate(sizes):
            if size > limit:
                return f"{_argument_name(argv, i)} would be {size - 1} bytes; Linux takes at most {limit - 1} bytes in one argument (MAX_ARG_STRLEN)"
    total_max = arg_max or _sysconf("SC_ARG_MAX")
    if total_max:
        environ = os.environ if env is None else env
        env_bytes = sum(len(f"{k}={v}".encode("utf-8", "surrogateescape")) + 1 for k, v in environ.items())
        total = sum(sizes) + env_bytes + 8 * (len(argv) + len(environ) + 2)
        if total > total_max:
            return f"the arguments and the environment would be {total} bytes; this system takes at most {total_max} (ARG_MAX)"
    return None


class LoginState(BaseModel):
    """What ``claude auth status`` reported - never a token, only the measured keys.

    ``logged_in`` / ``auth_method`` / ``api_provider`` are ``None`` when the
    CLI did not print them (or printed something that is not that type);
    ``error`` says why nothing could be read (the command failed or printed no
    JSON object).
    """

    logged_in: bool | None = None
    auth_method: str | None = None
    api_provider: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Command:
    """One model call as it is started: ``argv`` (no prompt in it) and ``stdin`` (the prompt's UTF-8 bytes).

    ``argv_sanitised``: :func:`sanitise_arg` changed an argument (the model,
    the system prompt, the schema ...); ``prompt_sanitised``:
    :func:`encode_prompt` changed the prompt. Each becomes its own ``raw``
    flag on the response.
    """

    argv: list[str]
    stdin: bytes
    argv_sanitised: bool
    prompt_sanitised: bool


@dataclass
class _Resumable:
    """A reply whose session was left on disk so one feedback turn can ``--resume`` it (see :meth:`ClaudeCodeClient.complete`)."""

    session_id: str
    model: str
    messages: list[LLMMessage]
    content: str
    cwd: str


def _amount(value: float) -> str:
    """A plain decimal for ``--max-budget-usd`` (no exponent notation)."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _opt_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _opt_float(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _excerpt(text: str, limit: int = ERROR_TEXT_LIMIT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def _first_lines(text: str, n: int = 5) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines[:n])


def parse_usage(envelope: dict[str, Any]) -> Usage:
    """The envelope's ``usage`` / ``total_cost_usd`` as a :class:`~ai_eda.llm.client.Usage` with subscription cost semantics.

    ``prompt_tokens`` is ``input_tokens + cache_creation_input_tokens +
    cache_read_input_tokens`` (``input_tokens`` alone counts only the uncached
    part; measured 2 + 1133 + 0 for one prompt), ``completion_tokens`` is
    ``output_tokens``, the cache counts keep their own fields and
    ``reasoning_tokens`` is ``output_tokens_details.thinking_tokens``.
    ``cost_usd`` is ``0.0`` (a known zero charge, not an unknown) and
    ``estimated_cost_usd`` is ``total_cost_usd`` when the envelope printed a
    number, else ``None``.
    """
    u = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
    details = u.get("output_tokens_details") if isinstance(u.get("output_tokens_details"), dict) else {}
    uncached = _int(u.get("input_tokens"))
    cache_write = _opt_int(u.get("cache_creation_input_tokens"))
    cache_read = _opt_int(u.get("cache_read_input_tokens"))
    prompt = uncached + (cache_write or 0) + (cache_read or 0)
    completion = _int(u.get("output_tokens"))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=0.0,
        cost_source=COST_SOURCE,
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=_opt_int(details.get("thinking_tokens")),
        estimated_cost_usd=_opt_float(envelope.get("total_cost_usd")),
        billing=BILLING,
    )


def served_model(envelope: dict[str, Any], requested: str) -> str:
    """The native id of the model that answered, from ``modelUsage``.

    The only entry when there is one; else the entry keyed by the requested
    id (measured: the served model is keyed exactly as ``--model`` named it,
    next to an auxiliary entry the CLI made itself); else the single entry
    whose four token counts equal the top-level ``usage`` (measured to be the
    served model's); else the requested id.
    """
    mu = envelope.get("modelUsage")
    if not isinstance(mu, dict) or not mu:
        return requested
    keys = [k for k in mu if isinstance(k, str)]
    if len(keys) == 1:
        return keys[0]
    if requested in mu:
        return requested
    u = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
    top = (_int(u.get("input_tokens")), _int(u.get("output_tokens")), _int(u.get("cache_read_input_tokens")), _int(u.get("cache_creation_input_tokens")))
    matches = [
        k for k in keys
        if isinstance(mu[k], dict)
        and (_int(mu[k].get("inputTokens")), _int(mu[k].get("outputTokens")), _int(mu[k].get("cacheReadInputTokens")), _int(mu[k].get("cacheCreationInputTokens"))) == top
    ]
    return matches[0] if len(matches) == 1 else requested


def finish_reason_of(envelope: dict[str, Any]) -> str | None:
    """``subtype: success`` → ``"stop"`` (``"length"`` when ``stop_reason`` is ``max_tokens`` - unmeasured spelling),
    ``error_max_turns`` → ``"length"``, any other subtype → its own name, no subtype → ``None``."""
    subtype = envelope.get("subtype")
    if not isinstance(subtype, str):
        return None
    if subtype == "success":
        return "length" if envelope.get("stop_reason") == "max_tokens" else "stop"
    if subtype == "error_max_turns":
        return "length"
    return subtype


def render_prompt(messages: list[LLMMessage]) -> tuple[str | None, str]:
    """``(system_text, prompt)`` for the CLI: system messages joined by blank lines; the conversation as one string.

    With exactly one non-system message its content is the prompt; with
    several they are rendered as ``[user]`` / ``[assistant]`` blocks in order
    (the CLI takes one prompt string, not a message list). A ``tool`` message
    or an assistant message carrying tool calls is refused: this route carries
    no caller tools.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    system = [m.content or "" for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    for m in rest:
        if m.role not in ("user", "assistant") or m.tool_calls:
            raise ValueError("the Claude Code CLI route carries no caller tools: tool messages / tool calls are not supported")
    if not rest:
        raise ValueError("messages must contain a user message")
    if len(rest) == 1:
        prompt = rest[0].content or ""
    else:
        prompt = "\n\n".join(f"[{m.role}]\n{m.content or ''}" for m in rest)
    return ("\n\n".join(system) if system else None), prompt


class ClaudeCodeClient(LLMClient):
    """:class:`~ai_eda.llm.client.LLMClient` over ``claude -p``. See the module docstring for the measured contract.

    ``cli`` names the binary (else :func:`find_claude_cli`; absent →
    :class:`~ai_eda.errors.ToolUnavailableError`); it is kept as an absolute
    path (:func:`absolute_cli_path`) and a Windows batch file is refused
    (:func:`refuse_batch_file`). ``timeout`` bounds one call, the write of
    the prompt to stdin included (the child is killed and reaped on expiry).
    ``max_budget_usd`` is passed as the CLI's own
    ``--max-budget-usd`` (a second safety net when the user gave a USD budget;
    the service's check comes first). ``fallback_model`` is passed as
    ``--fallback-model`` - the user's explicit opt-in to the CLI's own
    cross-model fallback, off by default.

    ``temperature`` and ``max_tokens`` are not settable on the CLI: both are
    ignored, and the response records ``raw["temperature_ignored"] = True``
    (always) and ``raw["max_tokens_ignored"] = True`` (when one was given);
    ``truncated`` therefore reads the envelope's subtype / stop reason. So
    nothing caps a call in flight on this route: the service still lowers
    ``max_tokens`` to the remaining token budget, but here a token budget is
    a pre-condition only and bounds the spend at the budget plus one
    *uncapped* call; ``--max-budget-usd`` is the only in-call cap and it is
    USD-denominated (the CLI's API-equivalent estimate), passed only with a
    USD budget.
    """

    #: the service's 0-USD rule is about per-call charges: a subscription call makes none, so it is not "paid"
    paid = False
    billing = BILLING

    def __init__(
        self,
        cli: str | None = None,
        *,
        timeout: float = 600.0,
        max_budget_usd: float | None = None,
        fallback_model: str | None = None,
    ) -> None:
        found = absolute_cli_path(cli) if cli else find_claude_cli()
        if not found:
            raise ToolUnavailableError(f"claude (Claude Code CLI) not found: install Claude Code and run `claude login`, or set {ENV_CLI}")
        self.cli = refuse_batch_file(found)
        self.timeout = float(timeout)
        self.max_budget_usd = float(max_budget_usd) if max_budget_usd is not None else None
        self.fallback_model = fallback_model
        self._version: str | None = None
        self._version_probed = False
        self._resumable: _Resumable | None = None
        #: usage of the most recent :meth:`stream` (set when it completed; ``None`` while running / after an error)
        self.last_stream_usage: Usage | None = None
        #: full accounting view of the most recent :meth:`stream`
        self.last_stream_response: LLMResponse | None = None

    def __repr__(self) -> str:
        return f"ClaudeCodeClient(cli={self.cli!r}, timeout={self.timeout:g})"

    __str__ = __repr__

    # ------------------------------------------------------------- probes

    def _probe(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        """Run a non-model subcommand (``--version``, ``auth status``); ``None`` when it could not run at all."""
        try:
            return subprocess.run(
                [self.cli, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL, env=dict(os.environ),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    def version(self) -> str | None:
        """``claude --version`` (e.g. ``2.1.283 (Claude Code)``), probed once and cached; ``None`` when it failed."""
        if not self._version_probed:
            self._version_probed = True
            proc = self._probe(["--version"])
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                self._version = proc.stdout.strip()
        return self._version

    def login_state(self) -> LoginState:
        """``claude auth status`` → :class:`LoginState` (``loggedIn`` / ``authMethod`` / ``apiProvider``; never a token). Never calls the model."""
        proc = self._probe(["auth", "status"])
        if proc is None:
            return LoginState(error="claude auth status could not run")
        try:
            data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            return LoginState(error=f"claude auth status exited {proc.returncode} without a JSON object: {_excerpt(_first_lines(proc.stderr or proc.stdout), 300)}")
        logged = data.get("loggedIn")
        method = data.get("authMethod")
        provider = data.get("apiProvider")
        return LoginState(
            logged_in=logged if isinstance(logged, bool) else None,
            auth_method=method if isinstance(method, str) else None,
            api_provider=provider if isinstance(provider, str) else None,
            error=None if proc.returncode == 0 else f"claude auth status exited {proc.returncode}",
        )

    # ---------------------------------------------------------- lifecycle

    def _drop_resumable(self) -> None:
        held, self._resumable = self._resumable, None
        if held is not None:
            shutil.rmtree(held.cwd, ignore_errors=True)

    def close(self) -> None:
        """Remove the temporary cwd kept for a possible feedback turn. Session files the CLI wrote under its own
        directory (a call whose session persistence was left on) are the CLI's and are not touched."""
        self._drop_resumable()

    # ------------------------------------------------------- request build

    def _feedback_turn(self, model: str, messages: list[LLMMessage], response_schema: dict[str, Any] | None) -> _Resumable | None:
        """The kept session when ``messages`` is the service's schema-feedback turn on the previous structured reply."""
        held = self._resumable
        if held is None or response_schema is None or model != held.model:
            return None
        if len(messages) != len(held.messages) + 2 or messages[:-2] != held.messages:
            return None
        echo, feedback = messages[-2], messages[-1]
        if echo.role != "assistant" or (echo.content or "") != held.content or feedback.role != "user":
            return None
        return held

    def build_argv(
        self,
        model: str,
        *,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        resume: str | None = None,
    ) -> list[str]:
        """The pinned command line - without the prompt, which goes on stdin (:func:`encode_prompt`).

        ``-p`` takes no argument. ``--no-session-persistence`` is left out
        when the reply may need a feedback turn (a structured call) or when
        resuming one; ``--resume`` and ``--no-session-persistence`` exclude
        each other. Every argument after the executable is passed through
        :func:`sanitise_arg`.
        """
        return self._command(model, "", system, response_schema=response_schema, resume=resume).argv

    def _command(
        self,
        model: str,
        prompt: str,
        system: str | None,
        *,
        response_schema: dict[str, Any] | None = None,
        resume: str | None = None,
    ) -> _Command:
        """:meth:`build_argv`'s command plus ``prompt`` as the stdin bytes, and which of the two had to be sanitised."""
        stdin, prompt_sanitised = encode_prompt(prompt)
        argv = ["-p", "--output-format", "json", "--model", model, "--tools", ""]
        if resume is not None:
            argv += ["--resume", resume]
        elif response_schema is None:
            argv.append("--no-session-persistence")
        argv += ["--setting-sources", "", "--strict-mcp-config"]
        if system is not None:
            argv += ["--system-prompt", system]
        if response_schema is not None:
            argv += ["--json-schema", json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", _amount(self.max_budget_usd)]
        if self.fallback_model:
            argv += ["--fallback-model", self.fallback_model]
        clean = [sanitise_arg(a) for a in argv]
        return _Command(argv=[self.cli, *clean], stdin=stdin, argv_sanitised=clean != argv, prompt_sanitised=prompt_sanitised)

    # ------------------------------------------------------- envelope parse

    def _error_from_envelope(self, model: str, envelope: dict[str, Any], returncode: int) -> LLMError | None:
        """An ``LLMError`` for an error envelope (``is_error``, a non-``result`` type, a non-zero exit) - or ``None``."""
        etype = envelope.get("type")
        subtype = envelope.get("subtype")
        is_error = envelope.get("is_error") is True
        if not is_error and etype == "result" and returncode == 0:
            return None
        usage = parse_usage(envelope) if isinstance(envelope.get("usage"), dict) else None
        named = subtype if isinstance(subtype, str) and subtype else None
        # the code names why the envelope is refused: its error subtype, else what made it an error
        if is_error:
            code: str = named or "is_error"
        elif etype != "result":
            code = named or f"type:{etype}"
        else:
            code = named if named is not None and named != "success" else f"exit {returncode}"
        status = envelope.get("api_error_status")
        result = envelope.get("result")
        message = result if isinstance(result, str) and result.strip() else json.dumps(envelope, ensure_ascii=False)
        if not is_error and etype == "result":
            message = f"exit {returncode} with a result envelope: {message}"
        return LLMError(
            _excerpt(message),
            kind="response",
            status=status if isinstance(status, int) and not isinstance(status, bool) else None,
            code=code,
            metadata={"api_error_status": status, "type": etype, "subtype": subtype, "num_turns": envelope.get("num_turns")},
            model=model,
            usage=usage,
            sent=True,
        )

    def _parse_response(
        self, model: str, envelope: dict[str, Any], *, want_structured: bool, temperature_ignored: bool, max_tokens_ignored: bool,
        argv_sanitised: bool = False, prompt_sanitised: bool = False,
    ) -> LLMResponse:
        problems: list[str] = []
        result = envelope.get("result")
        content: str | None
        if isinstance(result, str):
            content = result
        else:
            content = None
            if result is not None:
                problems.append(f"result is a JSON {type(result).__name__}, not a string")
        structured: dict[str, Any] | None = None
        if "structured_output" in envelope:
            so = envelope["structured_output"]
            if isinstance(so, dict):
                structured = so
            elif so is not None:
                problems.append(f"structured_output is a JSON {type(so).__name__}, not an object")
        if want_structured and structured is None:
            problems.append("the CLI printed no structured_output for the requested schema (content is parsed as JSON instead)")
        subtype = envelope.get("subtype")
        used = served_model(envelope, model)
        entry = envelope["modelUsage"].get(used) if isinstance(envelope.get("modelUsage"), dict) else None
        provider = entry.get("provider") if isinstance(entry, dict) else None
        session_id = envelope.get("session_id")
        raw = dict(envelope)
        raw["temperature_ignored"] = temperature_ignored
        if max_tokens_ignored:
            raw["max_tokens_ignored"] = True
        if argv_sanitised:
            raw["argv_sanitised"] = True
        if prompt_sanitised:
            raw["prompt_sanitised"] = True
        return LLMResponse(
            model=model,
            content=content,
            structured=structured,
            usage=parse_usage(envelope),
            raw=raw,
            model_used=used,
            finish_reason=finish_reason_of(envelope),
            native_finish_reason=subtype if isinstance(subtype, str) else None,
            id=session_id if isinstance(session_id, str) else None,
            provider_name=provider if isinstance(provider, str) else None,
            raw_error="; ".join(problems) if problems else None,
        )

    # ------------------------------------------------------------------ API

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One ``claude -p`` call → one response.

        ``tools`` → ``ValueError`` (this route carries no caller tools).
        ``temperature`` / ``max_tokens`` are ignored (see the class docstring).
        A structured call (``response_schema``) leaves session persistence on
        and keeps its empty cwd, so that the service's one schema-feedback turn
        (the same messages + the assistant's reply + one user message, same
        model, schema requested again) runs as ``--resume <session_id>`` with
        the feedback text only, in the same cwd; that cwd is removed after the
        feedback turn, at the next unrelated call, or by :meth:`close`. Every
        other call runs in a fresh empty cwd removed right after it.

        The prompt (on the feedback turn: the feedback text only) is written
        to the CLI's stdin as UTF-8 bytes (:func:`encode_prompt`; a changed
        prompt is recorded as ``raw["prompt_sanitised"] = True``, a changed
        argument as ``raw["argv_sanitised"] = True``). A command line the OS
        would refuse (:func:`argv_limit_problem` - the system prompt, the
        schema and the flags; the prompt is not on it) is an
        ``LLMError(kind="transport", code="prompt_too_long", sent=False)``
        raised before any directory or process exists.
        """
        if tools:
            raise ValueError("the Claude Code CLI route carries no caller tools (tools must be None or empty)")
        system, prompt = render_prompt(messages)
        want_structured = response_schema is not None
        held = self._feedback_turn(model, messages, response_schema)
        if held is not None:
            cmd = self._command(model, messages[-1].content or "", system, response_schema=response_schema, resume=held.session_id)
        else:
            cmd = self._command(model, prompt, system, response_schema=response_schema)
        problem = argv_limit_problem(cmd.argv)
        if problem is not None:
            log.info("claude-cli complete model=%s refused before starting: %s", model, problem)
            raise LLMError(
                f"{problem}; nothing was sent (the prompt would go on stdin, but the system prompt, the JSON schema and the "
                "flags stay on the Claude Code CLI's command line)",
                kind="transport", code=PROMPT_TOO_LONG, model=model, sent=False,
            )
        if held is not None:
            cwd = held.cwd
            self._resumable = None
        else:
            self._drop_resumable()
            cwd = tempfile.mkdtemp(prefix="ai-eda-claude-")
        keep_cwd = False
        t0 = time.monotonic()
        try:
            envelope, returncode = self._run(model, cmd.argv, cmd.stdin, cwd)
            resp = self._parse_response(
                model, envelope, want_structured=want_structured, temperature_ignored=True, max_tokens_ignored=max_tokens is not None,
                argv_sanitised=cmd.argv_sanitised, prompt_sanitised=cmd.prompt_sanitised,
            )
            if want_structured and held is None and resp.id:
                self._resumable = _Resumable(session_id=resp.id, model=model, messages=list(messages), content=resp.content or "", cwd=cwd)
                keep_cwd = True
        finally:
            if not keep_cwd:
                shutil.rmtree(cwd, ignore_errors=True)
        log.info(
            "claude-cli complete model=%s used=%s subtype=%s exit=%d prompt_tokens=%d completion_tokens=%d estimated_cost=%s elapsed=%.2fs",
            model, resp.model_used, resp.native_finish_reason, returncode, resp.usage.prompt_tokens, resp.usage.completion_tokens,
            "unknown" if resp.usage.estimated_cost_usd is None else f"{resp.usage.estimated_cost_usd:.6f}", time.monotonic() - t0,
        )
        return resp

    def _exchange(self, proc: subprocess.Popen[bytes], stdin: bytes) -> tuple[int, bytes, bytes, OSError | None]:
        """Feed ``stdin`` to the started ``proc`` and collect it: ``(exit code, stdout, stderr, write error)``.

        The prompt is written by a daemon thread of this client
        (:func:`_feed_stdin`), never by ``communicate``: ``proc.stdin`` is
        detached first, so ``communicate`` only reads stdout / stderr, and
        its ``timeout`` bounds the whole exchange on every platform and
        interpreter - CPython 3.12's Windows ``communicate`` writes ``input=``
        in the calling thread before its first timeout check, so a child that
        never read a prompt larger than the pipe buffer would block
        ``subprocess.run(input=...)`` with no bound. Writing and reading run
        together (no pipe deadlock). On the timeout (``TimeoutExpired``
        propagates) and on any other exception - a ``KeyboardInterrupt``
        reaches this thread, which never sits in a blocking write - the child
        is killed and reaped (:func:`_reap`); the writer then sees the broken
        pipe and ends, and is joined within :data:`REAP_TIMEOUT` (a daemon: a
        process the child started that still holds stdin cannot pin the call).
        """
        pipe = proc.stdin
        assert pipe is not None, "the child is started with stdin=PIPE"
        proc.stdin = None  # communicate() must never write (3.12 on Windows: unbounded) nor close it
        failures: list[OSError] = []
        writer = threading.Thread(target=_feed_stdin, args=(pipe, stdin, failures), name=STDIN_WRITER_NAME, daemon=True)
        try:
            writer.start()
            out, err = proc.communicate(timeout=self.timeout)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
                _reap(proc)
            if writer.ident is None:
                _feed_stdin(pipe, b"", failures)  # the writer never started: close stdin here so nothing waits on it
            raise
        finally:
            if writer.ident is not None:
                writer.join(REAP_TIMEOUT)
                if writer.is_alive():
                    log.info("claude-cli: stdin still held open by a process the child started; the writer is left to end with it")
        return proc.returncode, out or b"", err or b"", failures[0] if failures else None

    def _run(self, model: str, argv: list[str], stdin: bytes, cwd: str) -> tuple[dict[str, Any], int]:
        """Run the CLI with ``stdin`` written to its stdin; return ``(envelope, exit code)`` or raise the typed error.

        Bytes in, bytes out (no text mode): the child is started with three
        pipes, :meth:`_exchange` writes ``stdin`` on its own thread while the
        output is read, closes stdin after the last byte, ignores a broken
        pipe from a child that exits without reading it, and on the timeout
        kills and reaps the child - the timeout bounds the write too, whatever
        the platform. stdout / stderr are decoded here as UTF-8 with
        ``errors="replace"`` (never an exception). A write that failed other
        than by a broken pipe is named in the error only when the exit code
        and the envelope do not already decide the call. See the module
        docstring for the error mapping.
        """
        t0 = time.monotonic()
        try:
            child = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=dict(os.environ),
            )
        except OSError as e:
            # the child could not start (missing binary, E2BIG, winerror 206); a broken stdin pipe never lands here
            log.info("claude-cli complete model=%s could not start: %s", model, e.__class__.__name__)
            too_long = e.errno == errno.E2BIG or getattr(e, "winerror", None) == _WINERROR_TOO_LONG
            raise LLMError(
                (f"the OS refused the command line as too long ({e.__class__.__name__}: {e}); nothing was sent" if too_long else f"{e.__class__.__name__}: {e}"),
                kind="transport", code=PROMPT_TOO_LONG if too_long else None, model=model, sent=False,
            ) from None
        except ValueError as e:
            # argv the OS cannot encode (a NUL, a lone surrogate) that sanitise_arg did not catch: nothing started
            log.info("claude-cli complete model=%s could not start: %s", model, e.__class__.__name__)
            raise LLMError(f"{e.__class__.__name__}: {e}", kind="transport", model=model, sent=False) from None
        try:
            returncode, out_bytes, err_bytes, write_error = self._exchange(child, stdin)
        except subprocess.TimeoutExpired:
            log.info("claude-cli complete model=%s timeout after %gs", model, self.timeout)
            raise LLMError(f"timeout after {self.timeout:g}s", kind="transport", model=model, sent=True) from None
        proc = subprocess.CompletedProcess(argv, returncode, out_bytes.decode("utf-8", "replace"), err_bytes.decode("utf-8", "replace"))
        write_note = "" if write_error is None else f"; writing the prompt to stdin failed ({write_error.__class__.__name__}: {write_error})"
        if write_error is not None:
            log.info("claude-cli complete model=%s stdin write failed: %s", model, write_error.__class__.__name__)
        elapsed = time.monotonic() - t0
        envelope: dict[str, Any] | None = None
        try:
            data = json.loads(proc.stdout) if proc.stdout.strip() else None
        except ValueError:
            data = None
        if isinstance(data, dict):
            envelope = data
        if envelope is None:
            if proc.returncode != 0:
                log.info("claude-cli complete model=%s exit=%d (no envelope) elapsed=%.2fs", model, proc.returncode, elapsed)
                raise LLMError(
                    _excerpt(_first_lines(proc.stderr) or _first_lines(proc.stdout) or f"exit {proc.returncode} with empty output") + write_note,
                    kind="http", status=None, code=f"exit {proc.returncode}", model=model, sent=None,
                )
            log.info("claude-cli complete model=%s exit=0 (unparseable stdout) elapsed=%.2fs", model, elapsed)
            raise LLMError(
                "stdout is not a JSON envelope: " + _excerpt(_first_lines(proc.stdout) or "(empty)", 500) + write_note,
                kind="response", status=None, code="unparseable", model=model, sent=True,
            )
        err = self._error_from_envelope(model, envelope, proc.returncode)
        if err is not None:
            log.info("claude-cli complete model=%s exit=%d code=%s status=%s elapsed=%.2fs", model, proc.returncode, err.code, err.status, elapsed)
            raise err
        return envelope, proc.returncode

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """One-shot: runs :meth:`complete` and yields the whole content once (the CLI's ``json`` output is not streamed).

        ``last_stream_usage`` / ``last_stream_response`` are set from the
        response before the piece is yielded; an error leaves both ``None``.
        """
        self.last_stream_usage = None
        self.last_stream_response = None
        resp = self.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        self.last_stream_usage = resp.usage
        self.last_stream_response = resp
        if resp.content:
            yield resp.content


__all__ = [
    "BATCH_SUFFIXES",
    "BILLING",
    "COST_SOURCE",
    "ENV_CLI",
    "PROMPT_TOO_LONG",
    "VALUE_FLAGS",
    "WINDOWS_CLI_NAME",
    "WINDOWS_COMMAND_LINE_LIMIT",
    "ClaudeCodeClient",
    "LoginState",
    "absolute_cli_path",
    "argv_limit_problem",
    "encode_prompt",
    "find_claude_cli",
    "finish_reason_of",
    "parse_usage",
    "refuse_batch_file",
    "render_prompt",
    "sanitise_arg",
    "served_model",
]
