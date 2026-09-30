"""Command line entry point.

    ai-eda doctor [--online]  check external tools (kicad-cli, KiCad's ngspice.dll, whether the OpenRouter key
                             is set - never its value - and the Claude Code CLI: its path, version and login
                             state, never a token, and the model is never called; --online additionally
                             fetches the key's limit/usage, which sends the key to OPENROUTER_BASE_URL and is
                             recorded as a SECRET_ACCESS approval granted by the flag)
    ai-eda new NAME [--dir DIR] [--request TEXT]
                             create an empty project IR at DIR/ir.json (default projects/NAME); refused
                             (exit 2, nothing written) when that ir.json already exists - new never
                             replaces a project; an existing folder without ir.json is fine
    ai-eda run IR.json       run the pipeline until it blocks or finishes (exit 1 when blocked or the
                             final stage is FAIL; NOT_VERIFIED is exit 0 - nothing wrong, nothing proven;
                             exit 2 for a usage error such as --answer without '=')
        --answer KEY=VALUE               answer an open question (confirm_requirements=yes,
                                         accept_implicit=k1,k2 / reject_implicit=k3 for model-inferred items;
                                         leave_out=k1,k2 leaves those requirements out of the design (recorded
                                         in ir.requirements.left_out as your decision; closes an open question
                                         under the key; a typed answer to the key in a later run brings it
                                         back; a confirm_design=yes in the same run is ignored);
                                         regulatory scope: mains_powered=no radio=no finished_apparatus=yes
                                         evaluation_kit=no highest_rated_voltage="12 V DC" ...;
                                         confirm_parts=yes|no for the candidate-parts table; datasheet_facts_file=<json>;
                                         extract_datasheet_facts=yes / propose_regulations=yes ask the model (billed);
                                         confirm_facts=yes|no for the model's datasheet-facts table;
                                         accept_regulations=id1,id2 / reject_regulations=id3)
        --llm openrouter | claude | openrouter,claude | claude,openrouter | fake:<json>
                                         let the requirement agent extract from the free-text request. Every
                                         named provider is built before anything runs (a missing
                                         OPENROUTER_API_KEY or claude binary names the provider in the error)
                                         and the FIRST one is the default provider of an unprefixed model spec.
                                         openrouter is billed per call and needs a budget flag; claude is the
                                         locally installed Claude Code CLI (`claude -p`) on the user's own
                                         login - a subscription, not billed per call, so no budget is required
                                         (the flag itself is the SUBSCRIPTION_USE approval, recorded in the
                                         gate's audit log) but the run SHOWS the tokens and the CLI's
                                         API-equivalent cost estimate in its "LLM usage:" line. Caveat: when
                                         that CLI is authenticated with an API key instead of a subscription
                                         login (doctor's "auth:" says which; oauth_token is the subscription),
                                         the same estimate is a real charge on that key - the client cannot
                                         tell per call. fake:<json> replays a script offline (provider name
                                         script; billed per call by contract, so it needs a budget flag -
                                         0 USD allows it, it is free)
        --llm-model SPEC                 primary model as provider:model (openrouter:anthropic/claude-sonnet-5,
                                         claude:claude-sonnet-5, the CLI aliases claude:sonnet / claude:opus;
                                         unprefixed = the default provider; default: Sonnet on the default
                                         provider). anthropic: is reserved (no direct Messages-API client)
        --llm-fallback SPEC|same|none    a fallback candidate tried after a retryable failure (repeatable;
                                         default none - one project uses one model; a fallback exists only when
                                         you ask for one); `same` = the primary model on the other configured
                                         provider (openrouter:anthropic/<id> <-> claude:<id>, only for an id
                                         claude-<family>-<major>; others must be named explicitly)
        --llm-task-model TASK=SPEC       the model of one task (repeatable; TASK is a TaskKind value such as
                                         requirement_analysis, review; an unknown TASK is exit 2 listing them)
        --llm-allow-model-change         run with a primary spec other than the project's pin. The first run
                                         that serves at least one call pins its primary spec in
                                         ir.requirements.llm_model_spec (bookkeeping, outside the design hash);
                                         a later run with another primary spec is refused (exit 2, before any
                                         call) unless this flag is given, which re-pins after a served call
        --llm-claude-cli PATH            the claude binary (default $AI_EDA_CLAUDE_CLI when it exists, then
                                         `claude` on PATH); needs --llm claude
        --llm-claude-fallback-model LIST passthrough to the CLI's own --fallback-model (its cross-model
                                         fallback; off by default); needs --llm claude
        --llm-budget-usd X / --llm-budget-tokens N   the budget you grant (= the PAID_API_CALL approval);
                                         required whenever a per-call provider (openrouter, fake) is
                                         configured - without one no call is made (0 USD allows only the free
                                         fake client); with claude alone it is optional: the USD limit binds
                                         charges only (a subscription call adds none) and the token limit
                                         binds every call. --llm-budget-usd is also passed to the CLI as its own
                                         --max-budget-usd, a second safety net
        --online                         open an online session: the one NETWORK_FETCH approval of this run;
                                         datasheets and official regulatory texts are then fetched (https only)
                                         from the trusted hosts - KiCad library Datasheet hosts of the parts,
                                         the official domains of the regulatory candidate list, --trust-host -
                                         and archived under --sources-dir by sha256. Without it nothing is
                                         fetched and every source stays NOT_VERIFIED (offline)
        --trust-host HOST                trust every URL on HOST (repeatable)
        --datasheet-url REF=URL          the datasheet of component REF is exactly this URL (repeatable)
        --source-url ID=URL              an official text is exactly this URL (repeatable)
        --sources-dir DIR                the document archive (default <workdir>/sources)
        --catalog CSV --catalog-date ISO your distributor catalog export and the date you exported it
                                         (--catalog-authority / --catalog-supplier label it); backs sourcing
                                         values, never an identity
        --regulatory-candidates PATH     a candidate list other than the packaged one
        --fab-capability FILE            your JSON file naming the fab's capability page (url, or a saved
                                         file + retrieved_at) and the limits it states (key, value, unit,
                                         page, quote); the URL is trusted exactly, every limit is grounded
                                         verbatim on the archived page and recorded in ir.pcb.manufacturing
        As the run passes ARCHITECTURE (once the design exists), COMPONENT_SELECTION, PCB and RELEASE it
        writes the Korean stage reports <workdir>/reports/01..04_*.md + .html (inline SVG figures) and,
        when a headless Chromium / Chrome / Edge is found (--browser PATH names one, --no-pdf skips it),
        .pdf (views like report.html: no status computed, not artifacts, not hashed; a report that cannot
        be built is printed on stderr and never aborts the run); after RELEASE all four are re-written
        with the full run record. Each write prints "  report written: reports/<name> (+ .html, .pdf)"
        or "(+ .html; pdf not produced: <reason>)".
    ai-eda review IR.json    run only the independent reviewer (exit 1 on any FAIL)
    ai-eda relocate IR.json [--dry-run] [--from OLD]
                             re-record project.workdir as this ir.json's folder after the project folder
                             was copied or moved: artifact / evidence / archived-document paths under the
                             old folder are rebased onto the new one (paths outside it are left as they
                             are), the SPICE results reference is dropped (results.json names its rawfiles
                             by absolute path under the old folder; the next run re-simulates),
                             pipeline.json is not touched and the design hash does not change; the old
                             folder is the recorded absolute workdir, else the one the artifacts show (a
                             relative workdir's match, the root-level artifacts' folder), else --from OLD;
                             it refuses (exit 2, nothing written) when an artifact would stay outside this
                             folder; --dry-run prints what would change and writes nothing (exit 0; 2 for
                             an IR error, a busy project or an old folder it cannot tell)
    ai-eda report IR.json [-o FILE]
                             write one self-contained HTML file (default <workdir>/report.html) showing
                             what ir.json and <workdir>/pipeline.json record: every status is copied, none
                             is computed, nothing is saved (exit 0 written, 2 usage/IR error; never 1 -
                             the report is not a verdict)
    ai-eda serve IR.json [--port N]
                             serve that report on 127.0.0.1 only (default port 8765; 0 = a free port,
                             printed), GET / only, re-rendered on every request so a later `run` shows on
                             refresh; there is no --host flag on purpose
    ai-eda stage-reports IR.json [--dir DIR] [--no-pdf] [--browser PATH]
                             re-write the four Korean stage reports (theory, parts, circuit, final) as .md,
                             .html and .pdf from the saved ir.json and <workdir>/pipeline.json without
                             re-running anything (default folder <workdir>/reports; a missing run record
                             makes the reports say so; --no-pdf writes no PDF, --browser names the headless
                             browser to print with, else it is discovered - none found means .md + .html
                             only and the reason is printed); exit 0 written, 2 usage/IR error - never 1
    ai-eda gui [--root DIR] [--port N] [--open]
                             the local web GUI (Korean) for the projects under DIR (default `projects`
                             under the current directory, the folder `new` uses): create a project, run /
                             review / re-write its reports from forms, answer its questions and preview
                             every result (schematic, board, waveforms, validation, BOM, the stage reports,
                             report.html, the PDFs) with downloads and a zip. 127.0.0.1 only (default port
                             8766; 0 = a free port, printed), no --host on purpose; --open opens the page
                             in the default browser. A run is an `ai-eda` subprocess whose flags come from
                             the form, so keys, budgets, --llm claude and --online are approved exactly as
                             on the command line; the GUI never deletes or renames anything
    ai-eda doctor also prints "browser (pdf): <path or NOT FOUND>" - the Chromium / Chrome / Edge the
                             stage reports are printed with ($AI_EDA_BROWSER, PATH, a Playwright install,
                             the Windows / macOS install paths)

Where a project writes: the folder that holds its ir.json, always
(:func:`ai_eda.workdir.project_workdir`). ``run`` / ``review`` / ``report`` /
``serve`` / ``stage-reports`` refuse (exit 2; nothing is run, read from or
created in the other folder) an ir.json whose recorded absolute
``project.workdir`` names another folder, or whose registered artifacts
lie by absolute path outside its folder - a copied or moved project -
naming both paths and ``ai-eda relocate``. ``run`` / ``review`` /
``stage-reports`` / ``relocate`` hold the per-project lock
``<workdir>/.ai-eda.lock`` while they work (non-blocking: a second one
refuses with "another ai-eda run is using this project", exit 2; ``report``
/ ``serve`` / ``gui`` / ``new`` / ``doctor`` and ``relocate --dry-run`` take
none) and read ir.json again once they hold it, so a command that saved
meanwhile is never overwritten with a stale copy. ``run`` records the
resolved ir.json path in ``pipeline.json`` (``ir_file``), so a report can
tell a record inherited by a copy from its own.

Without ``--llm`` the pipeline is exactly what it was before the LLM stage.
The ``run`` options live in one table, :data:`RUN_OPTIONS`, that builds the
parser and :func:`run_option_flags` (the flag builder a GUI reuses, so the
flag names exist once). Without ``--online`` no code of ours opens a socket, and the headless
browser that ``run`` / ``stage-reports`` print the stage reports with runs
with name resolution and the proxy switched off
(``ai_eda.report.pdf.PRINT_FLAGS``), so it cannot reach any host either
(measured: no ``connect()`` at all; ``--no-pdf`` skips the browser
altogether). ``serve`` and ``gui`` are the commands
that open a socket without a flag: a loopback *listening* socket on
127.0.0.1 that answers only requests whose Host header names this machine
(``gui`` also refuses a POST from another origin) - no outbound connection
is ever made, so no ``ExternalAction`` is involved; the runs ``gui`` starts
are ``ai-eda`` subprocesses under their own flags.
The budget flags are the user's approval of paid calls, ``--llm claude`` the
approval of using the subscription login and ``--online`` the approval of
network fetches: all are recorded in the approval gate's audit log. The IR
is saved (and the LLM usage printed) whatever happens after the
pipeline starts, so a paid extraction or an archived fetch is never lost to
a later crash; ``run`` then records its stage outcomes in
``<workdir>/pipeline.json`` (a run log like ``ir.validation``: not an
artifact, not hashed) for ``report`` / ``serve``.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from ai_eda import __version__
from ai_eda.errors import IRSchemaError
from ai_eda.workdir import project_workdir  # re-export: the one "where does this project write" rule (ai_eda.workdir)

FAKE_PREFIX = "fake:"
#: the ``--llm`` provider names that are real routes (the fake is ``fake:<json>``, provider name ``script``)
LLM_PROVIDER_CHOICES = ("openrouter", "claude")
#: ``--llm-fallback`` words that are not specs
FALLBACK_NONE = "none"
FALLBACK_SAME = "same"
#: what ``--llm-fallback same`` / ``--llm-task-model`` resolve against when the fake is configured
SCRIPT_PROVIDER = "script"


class RunOption(NamedTuple):
    """One ``run`` option: the flag, the parsed attribute, its kind (``str`` / ``float`` / ``int`` / ``append`` / ``flag``), metavar and help."""

    flag: str
    dest: str
    kind: str
    metavar: str | None
    help: str


#: every option of ``ai-eda run`` in one place: the parser is built from it and :func:`run_option_flags` renders it
RUN_OPTIONS: tuple[RunOption, ...] = (
    RunOption(
        "--answer", "answer", "append", "KEY=VALUE",
        "answer an open question; confirm_requirements=yes, accept_implicit=k1,k2, reject_implicit=k3 steer the LLM extraction; "
        "leave_out=k1,k2 leaves those requirements out of the design (recorded as your decision; it closes an open question under the key, "
        "a typed answer to the key in a later run brings it back, and a confirm_design=yes in the same run is ignored); "
        "mains_powered=yes|no, radio=yes|no, finished_apparatus=yes|no, evaluation_kit=yes|no, digital_device=yes|no, kr_licence_free_class=yes|no, "
        "highest_rated_voltage='12 V DC', intended_use=... are the regulatory scope answers; "
        "confirm_parts=yes|no decides the candidate-parts table; datasheet_facts_file=<json> grounds your datasheet facts "
        "(layouts: ai_eda.parts.datasheet_facts.load_facts_file); extract_datasheet_facts=yes and propose_regulations=yes ask the model (billed); "
        "confirm_facts=yes|no decides the model's datasheet-facts table; accept_regulations=id1,id2 / reject_regulations=id3 decide shown model proposals",
    ),
    RunOption(
        "--llm", "llm", "str", "openrouter|claude|openrouter,claude|claude,openrouter|fake:<json>",
        "extract requirements from the request with a model. Comma-separated providers are all built and the first is the default "
        "provider of an unprefixed model spec: openrouter needs OPENROUTER_API_KEY and a budget flag (billed per call); claude is the "
        "Claude Code CLI on your own login (a subscription: no per-call charge, so no budget is needed - the flag is the SUBSCRIPTION_USE "
        "approval - but the tokens and the CLI's API-equivalent cost estimate are shown); fake:<json> replays a script offline "
        "(provider name script; needs a budget flag, 0 USD allows it)",
    ),
    RunOption(
        "--llm-model", "llm_model", "str", "SPEC",
        "primary model as provider:model (openrouter:anthropic/claude-sonnet-5, claude:claude-sonnet-5, claude:sonnet; unprefixed = the "
        "default provider; default Sonnet on the default provider). The first run that serves a call pins this spec in "
        "ir.requirements.llm_model_spec; a later run with another spec is refused unless --llm-allow-model-change is given",
    ),
    RunOption(
        "--llm-fallback", "llm_fallback", "append", "SPEC|same|none",
        "a fallback candidate for retryable failures (repeatable; default none: one project, one model). 'same' is the primary model on "
        "the other configured provider (openrouter:anthropic/<id> <-> claude:<id>, only for an id claude-<family>-<major>); 'none' adds nothing",
    ),
    RunOption(
        "--llm-task-model", "llm_task_model", "append", "TASK=SPEC",
        "the model of one task (repeatable; TASK is a TaskKind value: requirement_analysis, component_proposal, circuit_design, "
        "regulatory_research, result_interpretation, review, repair_planning, chat)",
    ),
    RunOption(
        "--llm-allow-model-change", "llm_allow_model_change", "flag", None,
        "run with a primary model spec other than the project's pin (ir.requirements.llm_model_spec) and re-pin it after a served call",
    ),
    RunOption("--llm-claude-cli", "llm_claude_cli", "str", "PATH", "the claude binary (default $AI_EDA_CLAUDE_CLI when it exists, else claude on PATH); needs --llm claude"),
    RunOption("--llm-claude-fallback-model", "llm_claude_fallback_model", "str", "LIST", "passthrough to the Claude Code CLI's own --fallback-model (off by default); needs --llm claude"),
    RunOption(
        "--llm-budget-usd", "llm_budget_usd", "float", "X",
        "approve up to X USD of provider-reported cost for this run (required with a per-call provider: openrouter, fake; counts charges only - "
        "a subscription call adds none; also passed to the Claude Code CLI as its own --max-budget-usd)",
    ),
    RunOption("--llm-budget-tokens", "llm_budget_tokens", "int", "N", "approve up to N prompt+completion tokens for this run (binds every call, subscription calls included)"),
    RunOption("--online", "online", "flag", None, "approve network fetches for this run (NETWORK_FETCH 'online session'): datasheets and official regulatory texts are fetched from trusted hosts only and archived by sha256; without it nothing is fetched"),
    RunOption("--trust-host", "trust_host", "append", "HOST", "trust every URL on HOST for fetching (repeatable; KiCad library datasheet hosts and the candidate list's official domains are trusted already)"),
    RunOption("--datasheet-url", "datasheet_url", "append", "REF=URL", "the datasheet of component REF is exactly this URL (repeatable; trusted as that URL only)"),
    RunOption("--source-url", "source_url", "append", "ID=URL", "an official text is exactly this URL (repeatable; trusted as that URL only)"),
    RunOption("--sources-dir", "sources_dir", "str", "DIR", "the document archive directory (default <workdir>/sources)"),
    RunOption("--catalog", "catalog", "str", "CSV", "your distributor catalog export (columns mpn, manufacturer, package + optional supplier_part_number, stock, unit_price, currency, assembly_class; JLCPCB/LCSC header spellings are mapped)"),
    RunOption("--catalog-date", "catalog_date", "str", "ISO8601", "the date you exported the catalog (required with --catalog)"),
    RunOption("--catalog-authority", "catalog_authority", "str", "TEXT", "who produced the catalog data, e.g. 'JLCPCB export'"),
    RunOption("--catalog-supplier", "catalog_supplier", "str", "NAME", "the supplier label for sourcing entries, e.g. JLCPCB"),
    RunOption("--regulatory-candidates", "regulatory_candidates", "str", "PATH", "a regulatory candidate list other than the packaged ai_eda/regulatory/candidates.json"),
    RunOption("--fab-capability", "fab_capability", "str", "FILE", "JSON file naming the fab's capability page (source.url, or source.file + retrieved_at for a saved page) and its limits (key, value, unit, page, quote); the URL is trusted exactly, the limits are grounded verbatim on the archived page"),
    RunOption("--no-pdf", "no_pdf", "flag", None, "write the stage reports as .md + .html only (no headless browser print)"),
    RunOption("--browser", "browser", "str", "PATH", "the Chromium / Chrome / Edge binary that prints the stage reports to PDF (default: discovered - $AI_EDA_BROWSER, PATH, Playwright, install paths)"),
)


def _add_run_option(parser: argparse.ArgumentParser, option: RunOption) -> None:
    if option.kind == "flag":
        parser.add_argument(option.flag, dest=option.dest, action="store_true", help=option.help)
    elif option.kind == "append":
        parser.add_argument(option.flag, dest=option.dest, action="append", metavar=option.metavar, help=option.help)
    else:
        typ = {"str": str, "float": float, "int": int}[option.kind]
        parser.add_argument(option.flag, dest=option.dest, type=typ, metavar=option.metavar, help=option.help)


def run_option_flags(**values: Any) -> list[str]:
    """The ``ai-eda run`` flags for ``values`` keyed by option name (:data:`RUN_OPTIONS` dests: ``llm``, ``llm_model``, ``answer``, ...).

    The GUI's run panel builds its command line with this, so the flag names
    exist in one place. ``None`` / ``False`` / an empty list render nothing;
    an ``append`` option takes a list (or a dict for ``KEY=VALUE`` options,
    rendered as ``key=value`` pairs); a ``flag`` option takes a bool. A name
    that is not a run option is a ``TypeError``, never silently dropped.
    Round trip: ``build_parser().parse_args(["run", ir, *run_option_flags(**v)])``
    yields ``v`` again.
    """
    known = {o.dest: o for o in RUN_OPTIONS}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise TypeError(f"not a run option: {', '.join(unknown)} (known: {', '.join(known)})")
    argv: list[str] = []
    for option in RUN_OPTIONS:
        value = values.get(option.dest)
        if value is None or value is False:
            continue
        if option.kind == "flag":
            if value is not True:
                raise TypeError(f"{option.flag} takes a bool, not {value!r}")
            argv.append(option.flag)
        elif option.kind == "append":
            items = [f"{k}={v}" for k, v in value.items()] if isinstance(value, dict) else list(value)
            for item in items:
                argv += [option.flag, str(item)]
        else:
            argv += [option.flag, str(value)]
    return argv


def cmd_doctor(args: argparse.Namespace) -> int:
    from ai_eda.errors import ToolUnavailableError
    from ai_eda.llm.openrouter import OpenRouterClient
    from ai_eda.tools.kicad import KicadCli, KicadLibrary
    from ai_eda.tools.spice import NgspiceRunner, NgspiceShared

    kicad = KicadCli()
    print(f"ai-eda {__version__}")
    print(f"kicad-cli : {kicad.binary or 'NOT FOUND'}" + (f"  (v{kicad.version()})" if kicad.available() else ""))
    lib = KicadLibrary()
    print(f"kicad libs: {[str(r) for r in lib.roots] or 'NOT FOUND'}")
    shared = NgspiceShared()
    if shared.available():
        try:
            reset = "" if shared.engine_info()["reset_supported"] else ", no ngSpice_Reset: a ControlledExit is not recoverable"
            info = f"  ({shared.version()}, build {shared.build()}, code models {'loaded' if shared.codemodels_loaded else 'NOT loaded'}{reset})"
        except ToolUnavailableError as e:
            info = f"  (UNUSABLE: {e})"
        print(f"ngspice dll: {shared.dll_path}{info}")
    else:
        print("ngspice dll: NOT FOUND (set NGSPICE_DLL or install KiCad, whose bin/ngspice.dll is used)")
    batch = NgspiceRunner()
    print(f"ngspice exe: {batch.binary or 'not on PATH (optional; the shared library is the engine used)'}")
    print(_browser_line())
    key_set = OpenRouterClient.key_present()
    print(f"LLM key   : {'set' if key_set else 'OPENROUTER_API_KEY not set'}")
    print(_claude_cli_line())
    if key_set and getattr(args, "online", False):
        from ai_eda.llm.client import LLMError
        from ai_eda.llm.openrouter import default_base_url
        from ai_eda.security.approval import ExternalAction, default_gate, require_approval

        # --online is the user's approval to send the key over the network once; record it as such
        detail = f"doctor --online key lookup at {default_base_url()}"
        default_gate().grant(ExternalAction.SECRET_ACCESS, detail, approved_by="cli --online")
        require_approval(ExternalAction.SECRET_ACCESS, detail)
        client = OpenRouterClient(app_title="AI EDA ENGINEER", timeout=20.0)
        try:
            info = client.key_info("auth/key")
        except LLMError as e:
            print(f"LLM account: unavailable ({e.kind}, status={e.status})")
        else:
            limit = info.get("limit")
            remaining = info.get("limit_remaining")
            print(
                f"LLM account: label={info.get('label')!r} limit={'none' if limit is None else limit} "
                f"remaining={'n/a' if remaining is None else remaining} usage={info.get('usage')} "
                f"free_tier={info.get('is_free_tier')}"
            )
        finally:
            client.close()
    elif getattr(args, "online", False):
        print("LLM account: not fetched (no key)")
    return 0


def _claude_cli_line() -> str:
    """``claude cli : <path> (<version>, logged in: yes|no|unknown, auth: <method>)`` or ``NOT FOUND`` with what would make one found.

    Runs ``claude --version`` and ``claude auth status`` only (through
    :func:`~ai_eda.llm.providers.describe_providers`); the model is never
    called and no token is printed.
    """
    from ai_eda.llm.claude_cli import ENV_CLI, find_claude_cli
    from ai_eda.llm.providers import describe_providers

    if find_claude_cli() is None:
        return f"claude cli : NOT FOUND (install Claude Code and run `claude login`, or set {ENV_CLI})"
    row = next(r for r in describe_providers() if r.name == "claude")
    return f"claude cli : {row.reason}"


def _browser_line() -> str:
    """``browser (pdf): <path>  (<version>)`` or ``NOT FOUND`` with what would make one found (the stage reports are then .md + .html only)."""
    from ai_eda.report.pdf import BROWSER_ENV, browser_version, find_browser

    browser = find_browser()
    if browser is None:
        return f"browser (pdf): NOT FOUND (chromium / chrome / edge on PATH, a Playwright install, or {BROWSER_ENV}; stage reports are written as .md + .html only)"
    version = browser_version(browser)
    return f"browser (pdf): {browser}" + (f"  ({version})" if version else "")


def _browser_option(args: argparse.Namespace) -> tuple[Path | None, bool] | None:
    """``(browser, pdf)`` from ``--browser`` / ``--no-pdf``; ``None`` after printing why when ``--browser`` names no file (a usage error, exit 2)."""
    chosen = getattr(args, "browser", None)
    if chosen and not Path(chosen).is_file():
        print(f"--browser {chosen}: not a file", file=sys.stderr)
        return None
    return (Path(chosen) if chosen else None), not getattr(args, "no_pdf", False)


class ProjectExistsError(FileExistsError):
    """``<workdir>/ir.json`` already exists (``new`` never replaces a project); a :class:`FileExistsError`, so existing handlers still catch it."""


def new_project(name: str, request: str | None = None, workdir: str | Path | None = None) -> Path:
    """Write the empty project IR ``ai-eda new`` creates and return the ir.json path (``<workdir>/ir.json``, absolute).

    The one code path of ``new`` and the GUI's "new project" form.
    ``workdir`` defaults to ``projects/<name>`` under the current directory.
    :class:`ProjectExistsError` (a :class:`FileExistsError`; nothing
    written) when ``<workdir>/ir.json`` already exists as a file or a link:
    ``new`` never replaces a project. An existing folder without ir.json is
    accepted; a ``workdir`` that is a file, or lies below one, raises the
    ``FileExistsError`` / ``NotADirectoryError`` of creating the folder.
    """
    from ai_eda.ir import CircuitIR, ProjectMeta

    # recorded absolute: a relative workdir would be resolved against whatever directory a later `run` / `review`
    # is started from, and the artifacts, the archive and the parts cache would land away from ir.json
    resolved = Path(workdir or f"projects/{name}").resolve()
    target = resolved / "ir.json"
    if os.path.lexists(target):
        raise ProjectExistsError(errno.EEXIST, "an ir.json already exists", str(target))
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(resolved)))
    ir.requirements.raw_input = request or ""
    return ir.save(target)


def cmd_new(args: argparse.Namespace) -> int:
    try:
        path = new_project(args.name, args.request, args.dir)
    except ProjectExistsError as e:
        print(
            f"refusing to create {e.filename}: an ir.json already exists there (ai-eda new never replaces a project; choose another --dir or name)",
            file=sys.stderr,
        )
        return 2
    except (FileExistsError, NotADirectoryError) as e:
        # creating the folder failed: the path, or a component of it, is a file - no ir.json is involved
        where = args.dir if args.dir else f"projects/{args.name}"
        print(f"refusing to create a project in {where}: it is not a folder ({e.filename or where} is a file, or lies below one); choose another --dir",
              file=sys.stderr)
        return 2
    print(f"created {path}")
    return 0


def _load(path: str):
    from ai_eda.ir import CircuitIR

    return CircuitIR.load(path)


def _lock_project(workdir: Path, command: str):
    """The held :class:`~ai_eda.workdir.ProjectLock` of ``workdir``, or ``None`` after printing why (busy, a linked lock file: exit 2).

    A folder where no lock can be taken proceeds unlocked; the lock's note says so on stderr.
    """
    from ai_eda.workdir import ProjectLock, ProjectLockError

    lock = ProjectLock(workdir, command=command)
    try:
        lock.acquire()
    except ProjectLockError as e:
        print(str(e), file=sys.stderr)
        return None
    if lock.note is not None:
        print(f"note: {lock.note}", file=sys.stderr)
    return lock


def _reload_locked(ir_path: str):
    """ir.json read again once the lock is held, with its workdir judged again; ``None`` after printing why (exit 2).

    The IR loaded before the lock only decided where the lock goes: another
    command may have saved ir.json between that load and the lock, and a
    run on the older copy would overwrite its work when it saves.
    """
    ir = _load(ir_path)
    try:
        project_workdir(ir, ir_path)
    except IRSchemaError as e:
        print(str(e), file=sys.stderr)
        return None
    return ir


@dataclass
class LLMOptions:
    """What the ``--llm*`` flags asked for, resolved before any client exists (see :func:`parse_llm_options`)."""

    #: the ``--llm`` text as given
    label: str
    #: provider names in configuration order (the first is the default provider); ``("script",)`` for the fake
    providers: tuple[str, ...]
    #: the fake's script path (``fake:<json>``), else ``None``
    script: str | None
    router: Any
    budget: Any
    allow_model_change: bool
    #: constructor options routed to the clients by signature (``cli`` / ``fallback_model`` for the CLI client)
    client_kw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_spec(self) -> str:
        """The primary ``provider:model`` spec - what the project pin records and compares."""
        return self.router.default.spec

    @property
    def per_call_providers(self) -> list[str]:
        """The configured providers billed per call (they need the budget flags)."""
        from ai_eda.llm.providers import PROVIDER_BILLING
        from ai_eda.llm.usage import SUBSCRIPTION

        return [p for p in self.providers if PROVIDER_BILLING.get(p, "per_call") != SUBSCRIPTION]

    def describe(self) -> str:
        """One line for the run log: the primary spec, the explicit fallbacks and the task overrides."""
        text = self.primary_spec
        fallbacks = [c.spec for c in self.router.fallbacks]
        if fallbacks:
            text += f"; fallback by --llm-fallback: {', '.join(fallbacks)}"
        overrides = [f"{task}={cfg.spec}" for task, cfg in self.router.by_task.items() if cfg.spec != self.primary_spec]
        if overrides:
            text += f"; task models: {', '.join(overrides)}"
        return text


def _llm_providers(spec: str) -> tuple[tuple[str, ...], str | None]:
    """``(providers, script path)`` for the ``--llm`` value; ``ValueError`` for anything but the documented forms."""
    if spec.startswith(FAKE_PREFIX):
        path = spec[len(FAKE_PREFIX):]
        if not path:
            raise ValueError("--llm fake:<json file> needs a path")
        return (SCRIPT_PROVIDER,), path
    names = tuple(p.strip().lower() for p in spec.split(","))
    bad = [n for n in names if n not in LLM_PROVIDER_CHOICES]
    if bad or not names:
        raise ValueError(
            f"unknown --llm {spec!r}: use 'openrouter', 'claude', 'openrouter,claude', 'claude,openrouter' or 'fake:<json file>'"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"--llm {spec!r} names a provider twice")
    return names, None


def _task_models(items: list[str] | None) -> dict[str, str]:
    """``TASK=SPEC`` pairs from ``--llm-task-model``; ``ValueError`` names the offending item (an unknown TASK is refused by the router)."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--llm-task-model expects TASK=SPEC, got {item!r}")
        task, spec = item.split("=", 1)
        if not task.strip() or not spec.strip():
            raise ValueError(f"--llm-task-model expects TASK=SPEC with both parts, got {item!r}")
        out[task.strip()] = spec.strip()
    return out


def parse_llm_options(args: argparse.Namespace) -> LLMOptions | None:
    """The :class:`LLMOptions` of the ``--llm*`` flags, or ``None`` without ``--llm``. Builds no client, sends nothing.

    ``ValueError`` (a usage error, exit 2) for an unknown ``--llm`` value, a
    provider named twice, a spec naming an unconfigured or reserved provider,
    ``--llm-fallback same`` without a second provider, an unknown task in
    ``--llm-task-model``, a ``--llm-claude-*`` flag without ``--llm claude``,
    and a per-call provider (openrouter, the fake) without a budget flag -
    the message names the provider that needs it. A subscription provider
    alone needs no budget: the ``--llm claude`` flag is its approval.
    """
    spec = getattr(args, "llm", None)
    if not spec:
        return None
    from ai_eda.llm.router import default_model, default_router, parse_model_spec, same_model_fallback
    from ai_eda.llm.service import LLMBudget, check_router_providers

    providers, script = _llm_providers(spec)
    default_provider = providers[0]
    budget = LLMBudget(max_usd=getattr(args, "llm_budget_usd", None), max_tokens=getattr(args, "llm_budget_tokens", None))
    model = getattr(args, "llm_model", None)
    primary = parse_model_spec(model or default_model(default_provider), default_provider)
    fallbacks: list[str] = []
    for item in getattr(args, "llm_fallback", None) or []:
        word = item.strip()
        if word.lower() == FALLBACK_NONE:
            continue
        fallbacks.append(same_model_fallback(primary, providers).spec if word.lower() == FALLBACK_SAME else word)
    router = default_router(model, fallback=fallbacks, default_provider=default_provider, by_task=_task_models(getattr(args, "llm_task_model", None)))
    check_router_providers(router, providers)
    client_kw: dict[str, Any] = {}
    for flag, dest, key in (("--llm-claude-cli", "llm_claude_cli", "cli"), ("--llm-claude-fallback-model", "llm_claude_fallback_model", "fallback_model")):
        value = getattr(args, dest, None)
        if value:
            if "claude" not in providers:
                raise ValueError(f"{flag} needs --llm claude (configured: {spec})")
            if key == "cli" and not Path(value).is_file():
                raise ValueError(f"{flag} {value}: not a file")
            client_kw[key] = value
    options = LLMOptions(
        label=spec, providers=providers, script=script, router=router, budget=budget,
        allow_model_change=bool(getattr(args, "llm_allow_model_change", False)), client_kw=client_kw,
    )
    per_call = options.per_call_providers
    if per_call and not budget.granted:
        who = f"the scripted client {spec}" if script else f"the per-call provider {'+'.join(per_call)} in --llm {spec}"
        raise ValueError(
            f"{who} needs an explicit budget: pass --llm-budget-usd X and/or --llm-budget-tokens N (that is the approval"
            + ("; 0 USD allows the free fake client)" if script else "; only a subscription provider such as claude alone needs none)")
        )
    return options


def build_llm_service(options: LLMOptions | argparse.Namespace | None):
    """The :class:`~ai_eda.llm.service.LLMService` for the ``--llm*`` flags (a :class:`LLMOptions` or the parsed args), or ``None``.

    Every configured provider's client is built here: ``ToolUnavailableError``
    when ``openrouter`` has no key or ``claude`` no binary (the message names
    the provider), ``OSError`` for a missing fake script. Construction records
    the approvals (``PAID_API_CALL`` for the budget, ``SUBSCRIPTION_USE`` for a
    subscription member) in the default gate; no model is called.
    """
    if isinstance(options, argparse.Namespace):
        options = parse_llm_options(options)
    if options is None:
        return None
    from ai_eda.llm.service import LLMService

    approved_by = "cli --llm-budget-usd/--llm-budget-tokens"
    if options.script is not None:
        return LLMService.from_script(options.script, options.budget, router=options.router, approved_by=approved_by)
    return LLMService.from_env(options.budget, router=options.router, providers=options.providers, approved_by=approved_by, **options.client_kw)


def model_pin_refusal(pinned: str, configured: str) -> str:
    """The Korean refusal of a run whose primary spec differs from the project's pin (exit 2, before any call)."""
    return (
        f"이 프로젝트는 지금까지 `{pinned}` 모델로 실행되었습니다. 다른 모델로 계속하려면 --llm-allow-model-change 를 주십시오. "
        f"(이번 실행의 모델: `{configured}`)"
    )


def _model_line(ir, options: LLMOptions) -> str:
    """``LLM model: <spec> (...)``: the run's models against the project pin (printed with leading spaces: not a stage line)."""
    pinned = ir.requirements.llm_model_spec
    if pinned is None:
        state = "not pinned yet: the first run that serves a call pins it"
    elif pinned == options.primary_spec:
        state = "the project pin"
    else:
        state = f"replacing the project pin {pinned} by --llm-allow-model-change; re-pinned after a served call"
    return f"  LLM model: {options.describe()} ({state})"


def _pin_model(ir, llm, options: LLMOptions) -> str | None:
    """Set / re-pin ``ir.requirements.llm_model_spec`` after a run that served at least one call; returns the new pin or ``None``."""
    served = any(r.outcome == "served" for r in llm.usage.records)
    if not served:
        return None
    if ir.requirements.llm_model_spec is None or (options.allow_model_change and ir.requirements.llm_model_spec != options.primary_spec):
        ir.requirements.llm_model_spec = options.primary_spec
        return options.primary_spec
    return None


def _close_llm(llm) -> None:
    """Close the client(s) behind a service: the HTTP pool, the CLI client's kept temp directory."""
    close = getattr(getattr(llm, "client", None), "close", None)
    if callable(close):
        try:
            close()
        except Exception as e:  # noqa: BLE001 - reported, never masks the run's own outcome
            print(f"could not close the LLM client: {e}", file=sys.stderr)


def parse_answers(items: list[str] | None) -> dict[str, str]:
    """``KEY=VALUE`` pairs from ``--answer``; ``ValueError`` names the offending item when '=' is missing or the key is empty."""
    answers: dict[str, str] = {}
    for kv in items or []:
        if "=" not in kv:
            raise ValueError(f"--answer expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        if not k.strip():
            raise ValueError(f"--answer expects KEY=VALUE with a non-empty key, got {kv!r}")
        answers[k.strip()] = v
    return answers


def build_source_session(args: argparse.Namespace, ir, workdir: Path, library):
    """The run's :class:`~ai_eda.workflow.session.SourceSession` from the ``--online`` / ``--trust-host`` / ``--datasheet-url`` /
    ``--source-url`` / ``--sources-dir`` / ``--catalog*`` / ``--regulatory-candidates`` / ``--fab-capability`` flags (``SessionError`` for a usage error)."""
    from ai_eda.security.approval import default_gate
    from ai_eda.workflow.session import open_session, parse_key_urls

    return open_session(
        workdir=workdir, ir=ir, library=library, online=bool(getattr(args, "online", False)),
        trust_hosts=getattr(args, "trust_host", None) or [],
        datasheet_urls=parse_key_urls(getattr(args, "datasheet_url", None), "--datasheet-url"),
        source_urls=parse_key_urls(getattr(args, "source_url", None), "--source-url"),
        sources_dir=getattr(args, "sources_dir", None),
        catalog=getattr(args, "catalog", None), catalog_date=getattr(args, "catalog_date", None),
        catalog_authority=getattr(args, "catalog_authority", None), catalog_supplier=getattr(args, "catalog_supplier", None),
        candidates=getattr(args, "regulatory_candidates", None), fab_capability=getattr(args, "fab_capability", None), gate=default_gate(),
    )


def cmd_run(args: argparse.Namespace) -> int:
    try:
        answers = parse_answers(args.answer)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        llm_options = parse_llm_options(args)
    except ValueError as e:
        print(f"--llm: {e}", file=sys.stderr)
        return 2
    browser_option = _browser_option(args)
    if browser_option is None:
        return 2
    browser, pdf = browser_option
    ir = _load(args.ir)
    try:
        workdir = project_workdir(ir, args.ir)
    except IRSchemaError as e:  # a copied / moved project is refused here: nothing is run or created anywhere
        print(str(e), file=sys.stderr)
        return 2
    # the lock before anything is created in the workdir (the source session makes <workdir>/sources)
    lock = _lock_project(workdir, "run")
    if lock is None:
        return 2
    try:
        locked_ir = _reload_locked(args.ir)
        if locked_ir is None:
            return 2
        return _run_locked(args, locked_ir, workdir, answers, llm_options, browser, pdf)
    finally:
        lock.release()  # after the IR, pipeline.json and the printed summary


def _run_locked(args: argparse.Namespace, ir, workdir: Path, answers: dict[str, str], llm_options, browser: Path | None, pdf: bool) -> int:
    """The part of ``run`` that works on the project, with its lock held and ir.json read under it."""
    from ai_eda.agents import AgentContext
    from ai_eda.errors import ApprovalRequiredError, ToolUnavailableError
    from ai_eda.tools.kicad import KicadCli, KicadLibrary
    from ai_eda.tools.spice import NgspiceShared
    from ai_eda.workflow import Orchestrator, PipelineState, SessionError

    if llm_options is not None:
        # the project pin: one project, one model - refused before any client exists or any call is made
        pinned = ir.requirements.llm_model_spec
        if pinned is not None and pinned != llm_options.primary_spec and not llm_options.allow_model_change:
            print(model_pin_refusal(pinned, llm_options.primary_spec), file=sys.stderr)
            return 2
    try:
        llm = build_llm_service(llm_options)
    except ApprovalRequiredError as e:
        print(f"--llm: {e}: pass --llm-budget-usd X and/or --llm-budget-tokens N (that is the approval).", file=sys.stderr)
        return 2
    except (ToolUnavailableError, ValueError, OSError) as e:
        print(f"--llm: {e}", file=sys.stderr)
        return 2
    library = KicadLibrary()
    try:
        session = build_source_session(args, ir, workdir, library)
    except SessionError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(session.summary())
    for note in session.notes:
        print(f"  note: {note}")
    if llm is not None and llm_options is not None:
        print(_model_line(ir, llm_options))
    ctx = AgentContext(
        workdir=workdir,
        tools={"kicad_cli": KicadCli(), "kicad_library": library, "spice": NgspiceShared(), **session.tools()},
        answers=answers,
        llm=llm,
    )
    if llm is not None:
        ctx.usage = llm.usage
    state = PipelineState()
    results_before = len(ir.validation.results)  # the run's starting index in ir.validation.results, recorded in pipeline.json
    aborted: str | None = None
    reports_written: list[Path] = []
    try:
        Orchestrator(ctx).run(ir, state=state, after_stage=_stage_report_writer(ir, library, workdir, reports_written, browser=browser, pdf=pdf))
    except BaseException as e:
        aborted = type(e).__name__  # only the type: an error text can embed a URL or a header, and it is written to disk
        raise
    finally:
        session.close()
        # whatever happened after the pipeline started (a defect in a later stage, Ctrl-C), what the
        # requirement stage applied - a paid extraction included - is on disk, and the spend is reported
        for o in state.outcomes:
            print(f"{o.stage:<24} {o.status:<20} {o.message}")
        if llm is not None:
            print(f"\nLLM usage: {llm.summary()}")
            if llm_options is not None:
                pin = _pin_model(ir, llm, llm_options)
                if pin is not None:
                    print(f"  LLM model pinned to this project: {pin} (ir.requirements.llm_model_spec; --llm-allow-model-change changes it)")
            _close_llm(llm)
        saved = False
        try:
            ir.save(args.ir)
            saved = True
        except Exception as e:  # noqa: BLE001 - reported, never masks the original exception
            print(f"could not save {args.ir}: {e}", file=sys.stderr)
        _record_pipeline(state, ir, args.ir, workdir, results_before=results_before, aborted=aborted, ir_saved=saved)
        if reports_written:
            print(f"  stage reports: {', '.join(_report_label(p, workdir) for p in dict.fromkeys(reports_written))} (ai-eda stage-reports {args.ir} re-writes them)")
        if aborted is not None:
            print(f"pipeline aborted by an unexpected error; IR saved to {args.ir} with what had been applied", file=sys.stderr)
            # the questions the IR now holds were meant for the user: show them, so a confirmation given
            # next time refers to a table that was actually seen
            _print_questions(ir.requirements.blocking_questions, header="\nOPEN QUESTIONS in the saved IR (pass with --answer key=value):")
    _print_existence(ir)
    if state.blocked:
        _print_questions(state.open_questions, header="\nBLOCKED - answer these to continue (pass with --answer key=value):")
    _print_questions(state.optional_questions, header="\nOPTIONAL QUESTIONS (not blocking; answer with --answer key=value to decide more):")
    return run_exit_code(state)


def _record_pipeline(state, ir, ir_path: str, workdir: Path, *, results_before: int, aborted: str | None, ir_saved: bool) -> None:
    """Write ``<workdir>/pipeline.json`` for ``ai-eda report``; a failure is reported on stderr and never masks the run's own outcome."""
    from ai_eda.report.pipeline_log import PIPELINE_FILE, save_pipeline_record, sha256_of_file

    try:
        ir_file_sha256 = sha256_of_file(ir_path) if ir_saved else None
        path = save_pipeline_record(state, ir, ir_path, workdir, results_before=results_before, ir_file_sha256=ir_file_sha256, aborted=aborted)
    except Exception as e:  # noqa: BLE001 - reported, never masks the original exception
        print(f"could not save {Path(workdir) / PIPELINE_FILE}: {e}", file=sys.stderr)
        return
    # the leading spaces keep the line out of the stage table (one line per stage, first word = stage)
    print(f"  stage outcomes recorded in {path} (ai-eda report {ir_path} renders them)")


def _report_label(path: Path, workdir: Path) -> str:
    """``reports/<name>``: a stage report named relative to the workdir (the printed line never carries the absolute path)."""
    from ai_eda.report.stages import REPORTS_DIR

    return f"{REPORTS_DIR}/{path.name}"


def _stage_report_writer(ir, library, workdir: Path, written: list[Path], *, browser: Path | None = None, pdf: bool = True):
    """The ``after_stage`` callback of ``run``: writes the stage's report (.md, .html and, with a browser, .pdf) from the live state and prints where.

    Only the stages in ``STAGE_REPORTS`` write one, ARCHITECTURE only once the
    design exists (``ir.components``); after RELEASE all four are re-written
    with the full record (the per-stage writes are progress views). A builder
    that raises is reported on stderr and the pipeline continues: a report is
    a view of the run, never a reason to abort it. ``browser`` / ``pdf`` are
    the ``--browser`` / ``--no-pdf`` flags.
    """
    from ai_eda.report.stages import REPORTS_DIR, STAGE_REPORTS, write_stage_report
    from ai_eda.workflow import Stage

    def after_stage(stage, state) -> None:
        if stage not in STAGE_REPORTS:
            return
        for report_stage in (list(STAGE_REPORTS) if stage is Stage.RELEASE else [stage]):
            if report_stage is Stage.ARCHITECTURE and not ir.components:
                continue
            try:
                result = write_stage_report(report_stage, ir, library, state, workdir, browser=browser, pdf=pdf)
            except Exception as e:  # noqa: BLE001 - a report must never abort a run; the reason is printed, the run goes on
                print(f"  could not write {REPORTS_DIR}/{STAGE_REPORTS[report_stage]}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if result is not None:
                written.append(result.markdown)
                print(f"  report written: {_report_label(result.markdown, workdir)} {result.summary()}")  # leading spaces: not a stage line

    return after_stage


def _print_questions(questions, *, header: str) -> None:
    if not questions:
        return
    print(header)
    for q in questions:
        label = " (model question)" if getattr(q, "source", "system") == "llm" else ""
        print(f"  [{q.key}]{label} {q.question}")


def _print_existence(ir) -> None:
    """Why an identity is not verified: the sub-checks of every ``component.existence.<ref>`` result that is not PASS."""
    from ai_eda.ir import ValidationStatus
    from ai_eda.parts.existence import CHECK_PREFIX

    latest = ir.validation.latest_by_check()
    rows = [(k, r) for k, r in sorted(latest.items()) if k.startswith(CHECK_PREFIX) and r.status is not ValidationStatus.PASS]
    if not rows:
        return
    print("\nCOMPONENT EXISTENCE - why an identity is not verified (sub-checks that did not pass):")
    for check_id, r in rows:
        print(f"  {check_id} {r.status}")
        for c in r.details.get("checks", []):
            if c.get("status") not in (str(ValidationStatus.PASS), str(ValidationStatus.NOT_APPLICABLE)):
                print(f"    - {c.get('name')}: {c.get('status')}: {c.get('message')}")


def run_exit_code(state) -> int:
    """``ai-eda run``'s exit status: 0 only when the pipeline neither blocked nor ended in FAIL.

    A blocked pipeline (open questions) and a final stage - normally RELEASE
    - whose status is FAIL both exit 1, so a script can tell a failing design
    from one that merely lacks evidence (NOT_VERIFIED exits 0: nothing is
    wrong, nothing is proven).
    """
    from ai_eda.ir import ValidationStatus

    if state.blocked:
        return 1
    if state.outcomes and state.outcomes[-1].status is ValidationStatus.FAIL:
        return 1
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from ai_eda.parts.identity import open_archive
    from ai_eda.review import IndependentReviewer

    ir = _load(args.ir)
    try:
        workdir = project_workdir(ir, args.ir)
    except IRSchemaError as e:
        print(str(e), file=sys.stderr)
        return 2
    lock = _lock_project(workdir, "review")
    if lock is None:
        return 2
    try:
        ir = _reload_locked(args.ir)
        if ir is None:
            return 2
        archive = open_archive({}, workdir)  # earlier runs' archived copies, read-only: hashes are re-verified, nothing is fetched
        tools = {"archive": archive} if archive is not None else {}
        report = IndependentReviewer(tools=tools).review(ir, workdir)
        for r in report.results:
            print(f"{r.check_id:<36} {r.status:<20} {r.message}")
        print(report.summary())
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        return 0 if not report.failures else 1
    finally:
        lock.release()


def _report_inputs(args: argparse.Namespace):
    """``(ir, workdir, ir_sha)`` for ``report`` / ``serve`` / ``stage-reports`` - the IR, its workdir and the hash of the bytes the
    IR was parsed from (one read) - or ``None`` after printing why (a usage / IR error, a copied or moved project: exit 2)."""
    from ai_eda.report.data import load_ir_file

    try:
        ir, ir_sha = load_ir_file(Path(args.ir))
    except (OSError, ValueError, IRSchemaError) as e:  # missing / unreadable file, not JSON or not an IR, another schema
        print(f"{args.ir}: {e}" if isinstance(e, (OSError, ValueError)) else str(e), file=sys.stderr)
        return None
    try:
        return ir, project_workdir(ir, args.ir), ir_sha
    except IRSchemaError as e:
        print(str(e), file=sys.stderr)
        return None


def cmd_report(args: argparse.Namespace) -> int:
    """Write one self-contained HTML file; exit 0 written, 2 for a usage / IR error - never 1, the report is not a verdict."""
    from ai_eda.report import build_report_data, render_html
    from ai_eda.report.pipeline_log import PIPELINE_FILE

    loaded = _report_inputs(args)
    if loaded is None:
        return 2
    ir, workdir, ir_sha = loaded
    out = Path(args.output) if args.output else workdir / "report.html"
    # the IR is the only original design data and pipeline.json its run log: neither is ever overwritten with HTML
    protected = {Path(args.ir).resolve(), (workdir / PIPELINE_FILE).resolve()}
    if out.resolve() in protected:
        print(f"refusing to write the report over {out}: pass another -o path", file=sys.stderr)
        return 2
    try:
        html = render_html(build_report_data(ir, Path(args.ir), workdir, ir_sha=ir_sha))
        out.write_text(html, encoding="utf-8", newline="\n")
    except OSError as e:
        print(f"could not write {out}: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


def cmd_stage_reports(args: argparse.Namespace) -> int:
    """Write the four Korean stage reports from the saved IR and pipeline.json; exit 0 written, 2 for a usage / IR error - never 1 (a report is not a verdict)."""
    loaded = _report_inputs(args)
    if loaded is None:
        return 2
    lock = _lock_project(loaded[1], "stage-reports")
    if lock is None:
        return 2
    try:
        return _stage_reports_locked(args)
    finally:
        lock.release()


def _stage_reports_locked(args: argparse.Namespace) -> int:
    """The part of ``stage-reports`` that reads and writes, with the project lock held (ir.json read again under it)."""
    from ai_eda.report import load_pipeline_record, write_all_stage_reports
    from ai_eda.report.data import describes_other_ir
    from ai_eda.report.pipeline_log import PIPELINE_FILE, PipelineRecordError
    from ai_eda.tools.kicad import KicadLibrary

    loaded = _report_inputs(args)
    if loaded is None:
        return 2
    ir, workdir, _ir_sha = loaded
    try:
        found = load_pipeline_record(workdir)
    except PipelineRecordError as e:
        print(f"{e} - the reports are written without a run record", file=sys.stderr)
        found = None
    record = found.record if found is not None else None
    if record is not None and describes_other_ir(record, Path(args.ir)):
        # a copied / relocated project's inherited record: its stages are not this ir.json's (the reports print no path)
        print(f"{PIPELINE_FILE} describes another ir.json ({record.ir_file}) - the reports are written without a run record", file=sys.stderr)
        record = None
    if record is None:
        print("no run record: the reports say so where they need one (run `ai-eda run` first to record stage outcomes)", file=sys.stderr)
    browser_option = _browser_option(args)
    if browser_option is None:
        return 2
    browser, pdf = browser_option
    reports_dir = Path(args.dir) if args.dir else None
    try:
        results = write_all_stage_reports(ir, KicadLibrary(), record, workdir, reports_dir=reports_dir, browser=browser, pdf=pdf)
    except OSError as e:
        print(f"could not write the stage reports: {e}", file=sys.stderr)
        return 2
    for r in results:
        print(f"wrote {r.markdown} {r.summary()}")
    return 0


def cmd_relocate(args: argparse.Namespace) -> int:
    """Re-record ``project.workdir`` as the ir.json's folder (:func:`ai_eda.workdir.relocate_project`); exit 0 done / nothing to do, 2 IR error or busy."""
    from ai_eda.workdir import RelocateError, relocate_project

    ir_path = Path(args.ir)
    try:
        _load(args.ir)  # an unreadable IR is refused before the lock file is created
    except (OSError, ValueError, IRSchemaError) as e:
        print(f"{args.ir}: {e}" if isinstance(e, (OSError, ValueError)) else str(e), file=sys.stderr)
        return 2
    lock = None
    if not args.dry_run:  # a dry run writes nothing, the lock file included
        lock = _lock_project(ir_path.resolve().parent, "relocate")
        if lock is None:
            return 2
    try:
        report = relocate_project(ir_path, dry_run=args.dry_run, from_folder=args.from_folder)  # reads ir.json itself, under the lock
    except (OSError, ValueError, IRSchemaError, RuntimeError, RelocateError) as e:
        print(f"{args.ir}: {e}" if isinstance(e, (OSError, ValueError)) else str(e), file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.release()
    print(relocate_summary(report))
    for path in report.missing_on_disk:
        print(f"  not on disk at the new place: {_below(path, report.new_workdir)}")
    return 0


def _below(path: str, folder: Path) -> str:
    """``path`` relative to ``folder`` (``/``-joined) when it lies below it, else as written."""
    try:
        return Path(path).relative_to(folder).as_posix()
    except ValueError:
        return path


def relocate_summary(report) -> str:
    """The one line ``ai-eda relocate`` prints for a :class:`~ai_eda.workdir.RelocateReport`."""
    from ai_eda.workdir import foreign_absolute

    new, old = report.new_workdir, report.old_workdir
    if not report.changed:
        if not old:
            return f"nothing to relocate: project.workdir is not recorded, so this ir.json's folder {new} is the workdir"
        if not (Path(old).is_absolute() or foreign_absolute(old)):
            return f"nothing to relocate: the relative project.workdir {old!r} names this ir.json's folder {new}"
        return f"nothing to relocate: project.workdir already is {new}"
    dry = report.dry_run
    breakdown = ", ".join(f"{k} {n}" for k, n in report.rebased.items() if n) or "none"
    parts = [
        f"{'would relocate' if dry else 'relocated'} {report.ir_path}: project.workdir {old} -> {new}",
        *([f"the artifacts were compiled into {report.rebased_from}"] if report.rebased_from and report.rebased_from != old else []),
        f"{'would rebase' if dry else 'rebased'} {report.rebased_total} path(s) ({breakdown})",
        f"{report.untouched} path(s) outside the old folder left as they are",
        *report.notes,
    ]
    if report.dropped:
        parts.append(
            f"SPICE results reference {'would be dropped' if dry else 'dropped'} (results.json names its rawfiles by absolute path "
            "under the old folder; the next `ai-eda run` re-simulates)"
        )
    parts.append(f"design hash unchanged ({report.design_hash[:23]})")
    if dry:
        parts.append("nothing was written (--dry-run)")
    return "; ".join(parts)


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the report on 127.0.0.1 (a loopback listening socket, nothing outbound); exit 0 on Ctrl-C, 2 for a usage / IR / bind error."""
    from ai_eda.report import serve

    if _report_inputs(args) is None:
        return 2
    return serve(Path(args.ir), args.port)


def cmd_gui(args: argparse.Namespace) -> int:
    """Serve the local web GUI on 127.0.0.1 (a loopback listening socket, nothing outbound); exit 0 on Ctrl-C, 2 for a bind error (a port outside 0-65535 is a usage error, also 2)."""
    from ai_eda.gui.server import serve_gui

    return serve_gui(Path(args.root), args.port, open_browser=args.open)


def port_number(text: str) -> int:
    """An ``--port`` value: a TCP port 0-65535 (0 = a free port); anything else is a usage error (exit 2), never a traceback from ``bind``."""
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535 (0 picks a free port): {port}")
    return port


def build_parser() -> argparse.ArgumentParser:
    """The ``ai-eda`` argument parser (``run``'s options come from :data:`RUN_OPTIONS`)."""
    p = argparse.ArgumentParser(prog="ai-eda", description="AI EDA ENGINEER")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check external tools")
    d.add_argument("--online", action="store_true", help="with a key set, fetch the OpenRouter key's limit/usage (one GET; the key is never printed)")
    d.set_defaults(fn=cmd_doctor)

    n = sub.add_parser("new", help="create an empty project")
    n.add_argument("name")
    n.add_argument("--dir")
    n.add_argument("--request", help="natural-language design request")
    n.set_defaults(fn=cmd_new)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("ir")
    for option in RUN_OPTIONS:
        _add_run_option(r, option)
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("review", help="independent review only")
    v.add_argument("ir")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_review)

    rl = sub.add_parser(
        "relocate",
        help="re-record project.workdir as this ir.json's folder after a copy or move; rebases artifact / evidence paths; no design change",
    )
    rl.add_argument("ir")
    rl.add_argument("--dry-run", action="store_true", help="print what would change and write nothing (no lock file either)")
    rl.add_argument("--from", dest="from_folder", metavar="OLD", help="the absolute folder the project was copied or moved from, when the IR "
                    "cannot tell it (no recorded workdir and no root-level artifact, or several candidates)")
    rl.set_defaults(fn=cmd_relocate)

    rp = sub.add_parser("report", help="write a self-contained HTML report of the IR, its validation log and the last run (read-only; no verdict)")
    rp.add_argument("ir")
    rp.add_argument("-o", "--output", metavar="FILE", help="where to write the HTML (default <workdir>/report.html; never the ir.json or pipeline.json)")
    rp.set_defaults(fn=cmd_report)

    sv = sub.add_parser("serve", help="serve the report on 127.0.0.1 (read-only, re-rendered on every request; no --host on purpose)")
    sv.add_argument("ir")
    sv.add_argument("--port", type=port_number, default=8765, help="TCP port on 127.0.0.1 (default 8765; 0 picks a free port and prints it)")
    sv.set_defaults(fn=cmd_serve)

    sr = sub.add_parser("stage-reports", help="re-write the four Korean stage reports (theory, parts, circuit, final) from ir.json and pipeline.json (read-only; no verdict)")
    sr.add_argument("ir")
    sr.add_argument("--dir", metavar="DIR", help="folder for the four reports (default <workdir>/reports)")
    sr.add_argument("--no-pdf", action="store_true", help="write .md + .html only (no headless browser print)")
    sr.add_argument("--browser", metavar="PATH", help="the Chromium / Chrome / Edge binary to print the PDFs with (default: discovered - $AI_EDA_BROWSER, PATH, Playwright, install paths)")
    sr.set_defaults(fn=cmd_stage_reports)

    g = sub.add_parser("gui", help="the local web GUI for the projects under --root (127.0.0.1 only; runs are ai-eda subprocesses; no --host on purpose)")
    g.add_argument("--root", metavar="DIR", default="projects", help="the projects root: one folder per project holding ir.json (default: projects under the current directory, as for new)")
    g.add_argument("--port", type=port_number, default=8766, help="TCP port on 127.0.0.1 (default 8766; 0 picks a free port and prints it)")
    g.add_argument("--open", action="store_true", help="open the page in the default web browser")
    g.set_defaults(fn=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
