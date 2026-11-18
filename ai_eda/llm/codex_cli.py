"""A model through the locally installed Codex CLI (``codex exec``) - a subprocess, no HTTP, no key.

The CLI uses the login the user already has (the ChatGPT account; ``codex
login status`` says which); this module never sees a key or a token. It
speaks only the ``codex exec --json`` contract measured below and maps one
run (a JSON-lines event stream) to one :class:`~ai_eda.llm.client.LLMResponse`.

Invariants enforced here:

* **Only measured event keys are parsed.** ``thread.started.thread_id``,
  ``item.completed.item`` (``type`` ``agent_message`` with ``text``; ``type``
  ``error`` with ``message``, a non-fatal warning), ``turn.completed.usage``
  (``input_tokens``, ``cached_input_tokens``, ``cache_write_input_tokens``,
  ``output_tokens``, ``reasoning_output_tokens``), ``error.message`` and
  ``turn.failed.error.message``. Any other event or key is kept in
  ``raw["events"]`` and never interpreted. Nothing about the reply is
  trusted: it is a proposal like every other model output.
* **No repository, no tools, no user config.** Every call runs with ``cwd`` =
  ``-C`` = a fresh empty temporary directory (removed afterwards; never the
  project), ``-s read-only``, ``--skip-git-repo-check``, ``--ephemeral`` (no
  session files), ``--ignore-user-config`` (``~/.codex/config.toml`` - its
  model provider, MCP servers and plugins - is not loaded; the login in
  ``CODEX_HOME`` still is), ``--ignore-rules``, ``-c model_provider=openai``
  (the ChatGPT login: the route the user approved when the configured local
  proxy is down) and ``--disable shell_tool --disable unified_exec``. The
  JSON schema of a structured call is written to a file in a sibling
  temporary directory (``--output-schema <file>``), never into the cwd.
* **The prompt travels on stdin as UTF-8 bytes** (``-`` as the prompt
  argument; :func:`~ai_eda.llm.claude_cli.encode_prompt`), written by a
  daemon thread so the call's ``timeout`` bounds the write
  (:func:`~ai_eda.llm.claude_cli.exchange`, shared with the Claude Code
  client). The CLI takes one prompt, so system messages and the
  conversation are rendered into it as ``[system]`` / ``[user]`` /
  ``[assistant]`` blocks (:func:`render_prompt`); a schema-feedback turn is
  the whole conversation again (``--ephemeral``: nothing to resume). The
  command line carries flags, the model and two temporary paths only, so no
  OS length limit applies to the prompt.
* **Cost semantics** are the subscription's, as for the Claude Code CLI: a
  call is a *known* zero charge (``cost_usd = 0.0``, ``cost_source`` /
  ``billing`` ``"subscription"``); the CLI prints no cost and no estimate, so
  ``estimated_cost_usd`` stays ``None``. That accounting holds only on a
  ChatGPT login (an API-key login would bill the key per call), so the
  constructor reads ``codex login status`` (no model call) and refuses the
  client with :class:`~ai_eda.errors.ToolUnavailableError` unless it prints
  the measured ChatGPT line (:meth:`CodexCliClient.login_state`); the login
  is not re-read per call.
* **Errors are typed, never parsed for retry.** A failed turn
  (``turn.failed``, a non-zero exit after events, or ``error`` events with no
  ``turn.completed``) is ``LLMError(kind="response", code="turn_failed",
  sent=True)`` with the CLI's own messages (``error`` events in a run that
  still completed with exit 0 - unmeasured - are kept as warnings); a non-zero
  exit without any event is ``kind="http"`` with ``code="exit <n>"`` and no
  status (neither retried nor fallen back from); a run that ended without a
  reply is ``kind="response"``, ``code="no_reply"``. ``temperature`` and
  ``max_tokens`` are not settable on this CLI: they are ignored and the
  response records ``raw["temperature_ignored"] = True`` (always) and
  ``raw["max_tokens_ignored"] = True`` (when one was given).
* **A batch file is never run** (an npm ``codex.cmd`` shim would be run
  through ``cmd.exe``, which re-parses the arguments): discovery prefers
  ``codex.exe`` and a resolved ``.cmd`` / ``.bat`` is refused.

Measured on codex-cli 0.167.0-alpha.3.2 (Windows 11, the Codex app's
``%LOCALAPPDATA%\\OpenAI\\Codex\\bin\\<hash>\\codex.exe``), 2026-11-13.
``codex --version`` -> ``codex-cli 0.167.0-alpha.3.2`` on stdout, exit 0.
``codex login status`` -> stdout empty, stderr ``Logged in using ChatGPT``,
exit 0. Three ``exec`` calls, each with cwd = ``-C`` = a fresh empty
temporary directory and the prompt written to stdin as UTF-8 then closed:
``codex exec --json -o <io>/last_message.txt --ignore-user-config
--ignore-rules -c model_provider=openai -m gpt-6-luna -s read-only
--skip-git-repo-check --ephemeral --disable shell_tool --disable
unified_exec --color never -C <work> [--output-schema <io>/schema.json] -``.
Every flag was accepted.

1. Plain prompt (ASCII plus Korean text and ``±``): exit 0, 5.2 s, stderr
   empty, the cwd left empty, stdout exactly four JSON lines -
   ``{"type":"thread.started","thread_id":"<uuid>"}``,
   ``{"type":"turn.started"}``, ``{"type":"item.completed","item":{"id":
   "item_0","type":"agent_message","text":"OK"}}``,
   ``{"type":"turn.completed","usage":{"input_tokens":13046,
   "cached_input_tokens":10240,"cache_write_input_tokens":0,
   "output_tokens":5,"reasoning_output_tokens":0}}``. The ``-o`` file held
   the same two bytes ``OK``. No event names the served model and none
   carries a cost.
2. ``--output-schema`` with ``{"type":"object","properties":{"word":
   {"type":"string"},"number":{"type":"integer"}},"required":["word",
   "number"],"additionalProperties":false}`` and a prompt asking for the
   Korean word 전압 and the number 647: exit 0, 3.8 s, stderr empty, the
   same four event types; the ``agent_message`` text was the JSON object
   text ``{"word":"전압","number":647}`` (UTF-8 bytes ``eca084 ec9595`` -
   the Korean stdin reached the model and came back intact), equal to the
   ``-o`` file; usage ``input_tokens`` 13078, ``cached_input_tokens`` 0,
   ``output_tokens`` 20. ``input_tokens`` includes the cached part (13046
   with 10240 cached and 13078 with none for prompts of the same size), so
   ``prompt_tokens`` here is ``input_tokens`` and ``cached_tokens`` is
   ``cached_input_tokens``.
3. ``-m no-such-model-xyz``: exit 1, 2.0 s, stderr empty, no ``-o`` file,
   stdout ``thread.started``; ``item.completed`` with ``{"type":"error",
   "message":"Model metadata for `no-such-model-xyz` not found. Defaulting
   to fallback metadata; ..."}`` (a warning item); ``turn.started``;
   ``{"type":"error","message":"The 'no-such-model-xyz' model is not
   supported when using Codex with a ChatGPT account."}``;
   ``{"type":"turn.failed","error":{"message":<the same text>}}``. No usage.

The client reads the reply from the event stream and does not pass ``-o``
(its file held exactly the last ``agent_message`` text in both successful
calls). Unmeasured (handled defensively): whether ``--disable shell_tool /
unified_exec`` removes every tool (no measured prompt asked for one; a tool
event would only be kept in ``raw``), several ``agent_message`` items in one
run (the last one is the reply), a ``reasoning_output_tokens`` value other
than 0 (taken as part of ``output_tokens``, as the OpenAI usage object
counts it), an API-key login (and whether an ``OPENAI_API_KEY`` in the
environment, which the child inherits, would be used instead of the ChatGPT
login - the measurement environment had none), a stalled or killed CLI, and
the CLI on Linux / macOS.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.claude_cli import BATCH_SUFFIXES, LoginState, absolute_cli_path, encode_prompt, exchange, sanitise_arg, subscription_env
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolSpec, Usage

#: environment variable naming the CLI binary explicitly (used when it names an existing file by an absolute path)
ENV_CLI = "AI_EDA_CODEX_CLI"
#: the executable looked up on PATH (on Windows ``codex.exe`` first: an npm install also puts a ``codex.cmd`` shim there)
CLI_NAME = "codex"
WINDOWS_CLI_NAME = "codex.exe"
#: where the Codex desktop app installs the CLI on Windows: ``%LOCALAPPDATA%\OpenAI\Codex\bin\<hash>\codex.exe`` (newest wins)
WINDOWS_APP_BIN = ("OpenAI", "Codex", "bin")
#: the model provider every call names (``-c model_provider=openai``): the ChatGPT login
MODEL_PROVIDER = "openai"
#: how a call is accounted: a known zero charge on the user's subscription (no estimate: the CLI prints none)
COST_SOURCE = "subscription"
BILLING = "subscription"
#: environment variables that would move a Codex call from the ChatGPT login to per-call API billing: left out of every child
#: environment (the owner's rule of 2026-11-13: GPT subscription tokens only, no API charges)
CODEX_BILLING_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")
#: the measured ``codex login status`` line of a ChatGPT login (stderr, exit 0)
CHATGPT_LOGIN_LINE = "Logged in using ChatGPT"
#: longest error excerpt kept in an :class:`LLMError` message
ERROR_TEXT_LIMIT = 2000
#: seconds allowed for ``--version`` / ``login status``
PROBE_TIMEOUT = 30.0
#: the ``LLMError.code`` of a failed turn (``turn.failed`` / ``error`` events) and of a run that ended without a reply
TURN_FAILED = "turn_failed"
NO_REPLY = "no_reply"

log = logging.getLogger("ai_eda.llm.codex_cli")


def _is_windows() -> bool:
    return os.name == "nt"


def _app_install() -> str | None:
    """The newest ``codex.exe`` the Codex desktop app installed under ``%LOCALAPPDATA%`` (Windows only), else ``None``."""
    base = os.environ.get("LOCALAPPDATA", "").strip()
    if not _is_windows() or not base or not os.path.isabs(base):
        return None
    found = [p for p in Path(base, *WINDOWS_APP_BIN).glob(f"*/{WINDOWS_CLI_NAME}") if p.is_file()]
    if not found:
        return None
    return str(max(found, key=lambda p: p.stat().st_mtime))


def find_codex_cli() -> str | None:
    """The CLI binary as an absolute path: ``$AI_EDA_CODEX_CLI`` (an absolute path to an existing file), else ``codex`` on PATH
    (``codex.exe`` first on Windows; PATH's absolute entries only), else the Codex app's newest ``codex.exe``, else ``None``."""
    from ai_eda.tools.kicad.install import which_absolute

    explicit = os.environ.get(ENV_CLI, "").strip()
    if explicit and Path(explicit).is_absolute() and Path(explicit).is_file():
        return explicit
    found = (which_absolute(WINDOWS_CLI_NAME) if _is_windows() else None) or which_absolute(CLI_NAME)
    if found:
        return absolute_cli_path(found)
    return _app_install()


def refuse_batch_file(path: str, *, windows: bool | None = None) -> str:
    """``path`` unchanged, or :class:`~ai_eda.errors.ToolUnavailableError` when Windows would run it through ``cmd.exe``."""
    if (_is_windows() if windows is None else windows) and Path(path).suffix.lower() in BATCH_SUFFIXES:
        raise ToolUnavailableError(
            f"the Codex CLI resolved to the batch file {path}: Windows runs a .cmd/.bat through cmd.exe, which re-parses the arguments - "
            f"refused. Set {ENV_CLI} or --llm-codex-cli to the native {WINDOWS_CLI_NAME}"
        )
    return path


def render_prompt(messages: list[LLMMessage]) -> str:
    """The one prompt the CLI reads: a lone user message as it is; otherwise ``[system]`` / ``[user]`` / ``[assistant]`` blocks in order.

    A ``tool`` message or an assistant message carrying tool calls is refused
    (this route carries no caller tools), as is a conversation without a user
    message.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    for m in messages:
        if m.role not in ("system", "user", "assistant") or m.tool_calls:
            raise ValueError("the Codex CLI route carries no caller tools: tool messages / tool calls are not supported")
    if not any(m.role == "user" for m in messages):
        raise ValueError("messages must contain a user message")
    if len(messages) == 1:
        return messages[0].content or ""
    return "\n\n".join(f"[{m.role}]\n{m.content or ''}" for m in messages)


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _opt_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _excerpt(text: str, limit: int = ERROR_TEXT_LIMIT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def parse_usage(usage: Any) -> Usage:
    """``turn.completed.usage`` as a :class:`~ai_eda.llm.client.Usage` with subscription cost semantics (see the module docstring)."""
    u = usage if isinstance(usage, dict) else {}
    prompt = _int(u.get("input_tokens"))
    completion = _int(u.get("output_tokens"))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=0.0,
        cost_source=COST_SOURCE,
        cached_tokens=_opt_int(u.get("cached_input_tokens")),
        cache_write_tokens=_opt_int(u.get("cache_write_input_tokens")),
        reasoning_tokens=_opt_int(u.get("reasoning_output_tokens")),
        billing=BILLING,
    )


def parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    """``(events, unparsed lines)``: every stdout line that is a JSON object, and the non-empty lines that are not."""
    events: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            unparsed.append(line)
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            unparsed.append(line)
    return events, unparsed


def _item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item")
    return item if event.get("type") == "item.completed" and isinstance(item, dict) else {}


def _failure_messages(events: list[dict[str, Any]]) -> list[str]:
    """The texts of the ``error`` and ``turn.failed`` events, in order and without repeats."""
    out: list[str] = []
    for e in events:
        text: Any = None
        if e.get("type") == "error":
            text = e.get("message")
        elif e.get("type") == "turn.failed" and isinstance(e.get("error"), dict):
            text = e["error"].get("message")
        if isinstance(text, str) and text.strip() and text.strip() not in out:
            out.append(text.strip())
    return out


class CodexCliClient(LLMClient):
    """:class:`~ai_eda.llm.client.LLMClient` over ``codex exec``. See the module docstring for the measured contract.

    ``codex_cli`` names the binary (else :func:`find_codex_cli`; absent ->
    :class:`~ai_eda.errors.ToolUnavailableError`); it is kept as an absolute
    path and a Windows batch file is refused. ``timeout`` bounds one call, the
    write of the prompt included (the child is killed and reaped on expiry).
    The option is not called ``cli`` so that ``--llm-claude-cli`` (routed by
    constructor signature) never reaches this client.
    """

    #: the service's 0-USD rule is about per-call charges: a subscription call makes none
    paid = False
    billing = BILLING

    def __init__(self, codex_cli: str | None = None, *, timeout: float = 600.0) -> None:
        found = absolute_cli_path(codex_cli) if codex_cli else find_codex_cli()
        if not found:
            raise ToolUnavailableError(f"codex (Codex CLI) not found: install Codex and run `codex login`, or set {ENV_CLI}")
        self.cli = refuse_batch_file(found)
        self.timeout = float(timeout)
        self._version: str | None = None
        self._version_probed = False
        # the route is billed as a subscription only on the measured ChatGPT login; any other state (an API-key login, which
        # would bill the key per call, or an unreadable one) is refused here, before the service records SUBSCRIPTION_USE
        state = self.login_state()
        if state.logged_in is not True:
            raise ToolUnavailableError(
                f"the Codex CLI {self.cli} is not logged in with ChatGPT ({state.error or 'not logged in'}): the codex route is used only on a "
                "ChatGPT login, the subscription it is accounted as (an API-key login would bill each call); run `codex login`"
            )
        #: usage of the most recent :meth:`stream` (set when it completed; ``None`` while running / after an error)
        self.last_stream_usage: Usage | None = None
        #: full accounting view of the most recent :meth:`stream`
        self.last_stream_response: LLMResponse | None = None

    def __repr__(self) -> str:
        return f"CodexCliClient(cli={self.cli!r}, timeout={self.timeout:g})"

    __str__ = __repr__

    # ------------------------------------------------------------- probes

    def _probe(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                [self.cli, *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL, env=subscription_env(CODEX_BILLING_ENV_VARS),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    def version(self) -> str | None:
        """``codex --version`` (e.g. ``codex-cli 0.167.0-alpha.3.2``), probed once and cached; ``None`` when it failed."""
        if not self._version_probed:
            self._version_probed = True
            proc = self._probe(["--version"])
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                self._version = proc.stdout.strip()
        return self._version

    def login_state(self) -> LoginState:
        """``codex login status`` -> :class:`~ai_eda.llm.claude_cli.LoginState`. Only the measured ChatGPT line is read; any other
        output is reported as unrecognised and never echoed (it could name a key). Never calls the model."""
        proc = self._probe(["login", "status"])
        if proc is None:
            return LoginState(error="codex login status could not run")
        if proc.returncode != 0:
            return LoginState(error=f"codex login status exited {proc.returncode}")
        lines = [ln.strip() for ln in f"{proc.stdout}\n{proc.stderr}".splitlines() if ln.strip()]
        if CHATGPT_LOGIN_LINE in lines:
            return LoginState(logged_in=True, auth_method="chatgpt")
        return LoginState(error="codex login status printed an unrecognised line (not shown)")

    def close(self) -> None:
        """Nothing is kept between calls (every call's temporary directories are removed when it ends)."""

    # ------------------------------------------------------- request build

    def build_argv(self, model: str, work: str, schema_file: str | None = None) -> list[str]:
        """The pinned command line (see the module docstring): the prompt is not on it (``-`` = stdin)."""
        argv = [
            "exec", "--json", "--ignore-user-config", "--ignore-rules", "-c", f"model_provider={MODEL_PROVIDER}", "-m", model,
            "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--disable", "shell_tool", "--disable", "unified_exec",
            "--color", "never", "-C", work,
        ]
        if schema_file is not None:
            argv += ["--output-schema", schema_file]
        argv.append("-")
        return [self.cli, *(sanitise_arg(a) for a in argv)]

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
        """One ``codex exec`` run -> one response (``tools`` -> ``ValueError``; ``temperature`` / ``max_tokens`` ignored)."""
        if tools:
            raise ValueError("the Codex CLI route carries no caller tools (tools must be None or empty)")
        stdin, prompt_sanitised = encode_prompt(render_prompt(messages))
        root = tempfile.mkdtemp(prefix="ai-eda-codex-")
        t0 = time.monotonic()
        try:
            work = os.path.join(root, "work")
            io_dir = os.path.join(root, "io")
            os.mkdir(work)
            os.mkdir(io_dir)
            schema_file: str | None = None
            if response_schema is not None:
                schema_file = os.path.join(io_dir, "schema.json")
                Path(schema_file).write_text(json.dumps(response_schema, ensure_ascii=False), encoding="utf-8")
            argv = self.build_argv(model, work, schema_file)
            returncode, out, err = self._run(model, argv, stdin, work)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        resp = self._parse(model, returncode, out, err, want_structured=response_schema is not None)
        raw = dict(resp.raw or {})
        raw["temperature_ignored"] = True
        if max_tokens is not None:
            raw["max_tokens_ignored"] = True
        if prompt_sanitised:
            raw["prompt_sanitised"] = True
        resp = resp.model_copy(update={"raw": raw})
        log.info(
            "codex-cli complete model=%s exit=%d prompt_tokens=%d completion_tokens=%d elapsed=%.2fs",
            model, returncode, resp.usage.prompt_tokens, resp.usage.completion_tokens, time.monotonic() - t0,
        )
        return resp

    def _run(self, model: str, argv: list[str], stdin: bytes, cwd: str) -> tuple[int, str, str]:
        """Start the CLI with ``stdin`` on its stdin; ``(exit code, stdout, stderr)`` decoded as UTF-8 (replace), or the typed error."""
        try:
            child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=subscription_env(CODEX_BILLING_ENV_VARS))
        except (OSError, ValueError) as e:
            log.info("codex-cli complete model=%s could not start: %s", model, e.__class__.__name__)
            raise LLMError(f"{e.__class__.__name__}: {e}", kind="transport", model=model, sent=False) from None
        try:
            returncode, out, err, write_error = exchange(child, stdin, self.timeout)
        except subprocess.TimeoutExpired:
            log.info("codex-cli complete model=%s timeout after %gs", model, self.timeout)
            raise LLMError(f"timeout after {self.timeout:g}s", kind="transport", model=model, sent=True) from None
        text_err = err.decode("utf-8", "replace")
        if write_error is not None:
            text_err += f"\nwriting the prompt to stdin failed ({write_error.__class__.__name__}: {write_error})"
        return returncode, out.decode("utf-8", "replace"), text_err

    def _parse(self, model: str, returncode: int, stdout: str, stderr: str, *, want_structured: bool) -> LLMResponse:
        events, unparsed = parse_events(stdout)
        failures = _failure_messages(events)
        completed = [e for e in events if e.get("type") == "turn.completed"]
        replies = [it.get("text") for it in map(_item, events) if it.get("type") == "agent_message" and isinstance(it.get("text"), str)]
        warnings = [it.get("message") for it in map(_item, events) if it.get("type") == "error" and isinstance(it.get("message"), str)]
        turn_failed = any(e.get("type") == "turn.failed" for e in events)
        if not turn_failed and returncode == 0 and completed:
            warnings += failures  # ``error`` events of a run that still completed (unmeasured: e.g. a reconnect): kept, not fatal
            failures = []
        if turn_failed or failures or (returncode != 0 and events):
            message = "; ".join(failures) or f"exit {returncode} with {len(events)} event(s) and no failure message"
            raise LLMError(_excerpt(message), kind="response", code=TURN_FAILED, model=model, sent=True,
                           metadata={"exit": returncode, "warnings": warnings})
        if not events:
            detail = _excerpt("\n".join([*unparsed, stderr.strip()]).strip() or "(no output)", 500)
            if returncode != 0:
                raise LLMError(detail, kind="http", code=f"exit {returncode}", model=model, sent=None)
            raise LLMError(f"stdout holds no JSON event: {detail}", kind="response", code="unparseable", model=model, sent=True)
        if not completed or not replies:
            raise LLMError(f"the run ended without {'turn.completed' if not completed else 'an agent_message'}", kind="response",
                           code=NO_REPLY, model=model, sent=True, usage=parse_usage(completed[-1].get("usage")) if completed else None)
        content = replies[-1]
        problems: list[str] = []
        structured: dict[str, Any] | None = None
        if want_structured:
            try:
                obj = json.loads(content)
            except ValueError as e:
                problems.append(f"structured content is not valid JSON: {e}")
            else:
                if isinstance(obj, dict):
                    structured = obj
                else:
                    problems.append(f"structured content is a JSON {type(obj).__name__}, not an object")
        if unparsed:
            problems.append(f"{len(unparsed)} stdout line(s) were not JSON events")
        thread = next((e.get("thread_id") for e in events if e.get("type") == "thread.started" and isinstance(e.get("thread_id"), str)), None)
        return LLMResponse(
            model=model,
            content=content,
            structured=structured,
            usage=parse_usage(completed[-1].get("usage")),
            raw={"events": events, "exit": returncode, "warnings": warnings},
            # no event names the served model (measured): the requested id is all there is
            model_used=model,
            finish_reason="stop",
            native_finish_reason="turn.completed",
            id=thread,
            raw_error="; ".join(problems) if problems else None,
        )

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """One-shot: runs :meth:`complete` and yields the whole content once (``last_stream_*`` set before the yield)."""
        self.last_stream_usage = None
        self.last_stream_response = None
        resp = self.complete(model, messages, temperature=temperature, max_tokens=max_tokens)
        self.last_stream_usage = resp.usage
        self.last_stream_response = resp
        if resp.content:
            yield resp.content


__all__ = [
    "BILLING",
    "CHATGPT_LOGIN_LINE",
    "COST_SOURCE",
    "ENV_CLI",
    "MODEL_PROVIDER",
    "NO_REPLY",
    "TURN_FAILED",
    "WINDOWS_CLI_NAME",
    "CodexCliClient",
    "find_codex_cli",
    "parse_events",
    "parse_usage",
    "refuse_batch_file",
    "render_prompt",
]
