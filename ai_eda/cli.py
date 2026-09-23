"""Command line entry point.

    ai-eda doctor [--online]  check external tools (kicad-cli, KiCad's ngspice.dll, whether the LLM key is
                             set - never its value; --online additionally fetches the key's limit/usage,
                             which sends the key to OPENROUTER_BASE_URL and is recorded as a SECRET_ACCESS
                             approval granted by the flag)
    ai-eda new NAME          create an empty project IR
    ai-eda run IR.json       run the pipeline until it blocks or finishes (exit 1 when blocked or the
                             final stage is FAIL; NOT_VERIFIED is exit 0 - nothing wrong, nothing proven;
                             exit 2 for a usage error such as --answer without '=')
        --answer KEY=VALUE               answer an open question (confirm_requirements=yes,
                                         accept_implicit=k1,k2 / reject_implicit=k3 for model-inferred items;
                                         regulatory scope: mains_powered=no radio=no finished_apparatus=yes
                                         evaluation_kit=no highest_rated_voltage="12 V DC" ...;
                                         confirm_parts=yes|no for the candidate-parts table; datasheet_facts_file=<json>;
                                         extract_datasheet_facts=yes / propose_regulations=yes ask the model (billed);
                                         confirm_facts=yes|no for the model's datasheet-facts table;
                                         accept_regulations=id1,id2 / reject_regulations=id3)
        --llm openrouter | fake:<json>   let the requirement agent extract from the free-text request
        --llm-model ID                   primary model (default anthropic/claude-sonnet-5)
        --llm-budget-usd X / --llm-budget-tokens N   the budget you grant; without one no call is made
                                         (0 USD allows only the free fake client)
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
    ai-eda review IR.json    run only the independent reviewer (exit 1 on any FAIL)

Without ``--llm`` the pipeline is exactly what it was before the LLM stage;
without ``--online`` it opens no socket. The budget flags are the user's
approval of paid calls and ``--online`` the approval of network fetches:
both are recorded in the approval gate's audit log. The IR is saved (and
the LLM usage printed) whatever happens after the pipeline starts, so a
paid extraction or an archived fetch is never lost to a later crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_eda import __version__

FAKE_PREFIX = "fake:"


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
            info = f"  ({shared.version()}, build {shared.build()}, code models {'loaded' if shared.codemodels_loaded else 'NOT loaded'})"
        except ToolUnavailableError as e:
            info = f"  (UNUSABLE: {e})"
        print(f"ngspice dll: {shared.dll_path}{info}")
    else:
        print("ngspice dll: NOT FOUND (set NGSPICE_DLL or install KiCad, whose bin/ngspice.dll is used)")
    batch = NgspiceRunner()
    print(f"ngspice exe: {batch.binary or 'not on PATH (optional; the shared library is the engine used)'}")
    key_set = OpenRouterClient.key_present()
    print(f"LLM key   : {'set' if key_set else 'OPENROUTER_API_KEY not set'}")
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


def cmd_new(args: argparse.Namespace) -> int:
    from ai_eda.ir import CircuitIR, ProjectMeta

    workdir = Path(args.dir or f"projects/{args.name}")
    ir = CircuitIR(project=ProjectMeta(id=args.name, name=args.name, workdir=str(workdir)))
    ir.requirements.raw_input = args.request or ""
    path = ir.save(workdir / "ir.json")
    print(f"created {path}")
    return 0


def _load(path: str):
    from ai_eda.ir import CircuitIR

    return CircuitIR.load(path)


def build_llm_service(args: argparse.Namespace):
    """The :class:`~ai_eda.llm.service.LLMService` for ``--llm``, or ``None`` when the flag is absent.

    Raises ``ApprovalRequiredError`` when no budget flag was given (the flags
    are the approval), ``ToolUnavailableError`` when ``--llm openrouter`` has
    no key, ``ValueError`` for an unknown ``--llm`` value.
    """
    spec = getattr(args, "llm", None)
    if not spec:
        return None
    from ai_eda.llm.router import default_router
    from ai_eda.llm.service import LLMBudget, LLMService

    budget = LLMBudget(max_usd=getattr(args, "llm_budget_usd", None), max_tokens=getattr(args, "llm_budget_tokens", None))
    router = default_router(getattr(args, "llm_model", None))
    approved_by = "cli --llm-budget-usd/--llm-budget-tokens"
    if spec == "openrouter":
        return LLMService.from_env(budget, router=router, approved_by=approved_by)
    if spec.startswith(FAKE_PREFIX):
        path = spec[len(FAKE_PREFIX):]
        if not path:
            raise ValueError("--llm fake:<json file> needs a path")
        return LLMService.from_script(path, budget, router=router, approved_by=approved_by)
    raise ValueError(f"unknown --llm {spec!r}: use 'openrouter' or 'fake:<json file>'")


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
    ``--source-url`` / ``--sources-dir`` / ``--catalog*`` / ``--regulatory-candidates`` flags (``SessionError`` for a usage error)."""
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
        candidates=getattr(args, "regulatory_candidates", None), gate=default_gate(),
    )


def cmd_run(args: argparse.Namespace) -> int:
    from ai_eda.agents import AgentContext
    from ai_eda.errors import ApprovalRequiredError, ToolUnavailableError
    from ai_eda.tools.kicad import KicadCli, KicadLibrary
    from ai_eda.tools.spice import NgspiceShared
    from ai_eda.workflow import Orchestrator, SessionError

    try:
        answers = parse_answers(args.answer)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        llm = build_llm_service(args)
    except ApprovalRequiredError:
        print("LLM use needs an explicit budget: pass --llm-budget-usd X and/or --llm-budget-tokens N (that is the approval).", file=sys.stderr)
        return 2
    except (ToolUnavailableError, ValueError, OSError) as e:
        print(f"--llm: {e}", file=sys.stderr)
        return 2
    ir = _load(args.ir)
    workdir = Path(ir.project.workdir or Path(args.ir).parent)
    library = KicadLibrary()
    try:
        session = build_source_session(args, ir, workdir, library)
    except SessionError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(session.summary())
    for note in session.notes:
        print(f"  note: {note}")
    ctx = AgentContext(
        workdir=workdir,
        tools={"kicad_cli": KicadCli(), "kicad_library": library, "spice": NgspiceShared(), **session.tools()},
        answers=answers,
        llm=llm,
    )
    if llm is not None:
        ctx.usage = llm.usage
    state = None
    try:
        state = Orchestrator(ctx).run(ir)
    finally:
        session.close()
        # whatever happened after the pipeline started (a defect in a later stage, Ctrl-C), what the
        # requirement stage applied - a paid extraction included - is on disk, and the spend is reported
        if state is not None:
            for o in state.outcomes:
                print(f"{o.stage:<24} {o.status:<20} {o.message}")
        if llm is not None:
            print(f"\nLLM usage: {llm.summary()}")
        try:
            ir.save(args.ir)
        except Exception as e:  # noqa: BLE001 - reported, never masks the original exception
            print(f"could not save {args.ir}: {e}", file=sys.stderr)
        if state is None:
            print(f"pipeline aborted by an unexpected error; IR saved to {args.ir} with what had been applied", file=sys.stderr)
            # the questions the IR now holds were meant for the user: show them, so a confirmation given
            # next time refers to a table that was actually seen
            _print_questions(ir.requirements.blocking_questions, header="\nOPEN QUESTIONS in the saved IR (pass with --answer key=value):")
    _print_existence(ir)
    if state.blocked:
        _print_questions(state.open_questions, header="\nBLOCKED - answer these to continue (pass with --answer key=value):")
    _print_questions(state.optional_questions, header="\nOPTIONAL QUESTIONS (not blocking; answer with --answer key=value to decide more):")
    return run_exit_code(state)


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
    workdir = Path(ir.project.workdir or Path(args.ir).parent)
    archive = open_archive({}, workdir)  # earlier runs' archived copies, read-only: hashes are re-verified, nothing is fetched
    tools = {"archive": archive} if archive is not None else {}
    report = IndependentReviewer(tools=tools).review(ir, workdir)
    for r in report.results:
        print(f"{r.check_id:<36} {r.status:<20} {r.message}")
    print(report.summary())
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    return 0 if not report.failures else 1


def main(argv: list[str] | None = None) -> int:
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
    r.add_argument(
        "--answer", action="append", metavar="KEY=VALUE",
        help=(
            "answer an open question; confirm_requirements=yes, accept_implicit=k1,k2, reject_implicit=k3 steer the LLM extraction; "
            "mains_powered=yes|no, radio=yes|no, finished_apparatus=yes|no, evaluation_kit=yes|no, digital_device=yes|no, "
            "highest_rated_voltage='12 V DC', intended_use=... are the regulatory scope answers; "
            "confirm_parts=yes|no decides the candidate-parts table; datasheet_facts_file=<json> grounds your datasheet facts "
            "(layouts: ai_eda.parts.datasheet_facts.load_facts_file); extract_datasheet_facts=yes and propose_regulations=yes ask the model (billed); "
            "confirm_facts=yes|no decides the model's datasheet-facts table; accept_regulations=id1,id2 / reject_regulations=id3 decide shown model proposals"
        ),
    )
    r.add_argument("--llm", metavar="openrouter|fake:<json>", help="extract requirements from the request with a model (openrouter needs OPENROUTER_API_KEY; fake:<json> replays a script offline)")
    r.add_argument("--llm-model", metavar="ID", help="primary model id (default anthropic/claude-sonnet-5; fallback anthropic/claude-haiku-4.5)")
    r.add_argument("--llm-budget-usd", type=float, metavar="X", help="approve up to X USD of provider-reported cost for this run")
    r.add_argument("--llm-budget-tokens", type=int, metavar="N", help="approve up to N prompt+completion tokens for this run")
    r.add_argument("--online", action="store_true", help="approve network fetches for this run (NETWORK_FETCH 'online session'): datasheets and official regulatory texts are fetched from trusted hosts only and archived by sha256; without it nothing is fetched")
    r.add_argument("--trust-host", action="append", metavar="HOST", help="trust every URL on HOST for fetching (repeatable; KiCad library datasheet hosts and the candidate list's official domains are trusted already)")
    r.add_argument("--datasheet-url", action="append", metavar="REF=URL", help="the datasheet of component REF is exactly this URL (repeatable; trusted as that URL only)")
    r.add_argument("--source-url", action="append", metavar="ID=URL", help="an official text is exactly this URL (repeatable; trusted as that URL only)")
    r.add_argument("--sources-dir", metavar="DIR", help="the document archive directory (default <workdir>/sources)")
    r.add_argument("--catalog", metavar="CSV", help="your distributor catalog export (columns mpn, manufacturer, package + optional supplier_part_number, stock, unit_price, currency, assembly_class; JLCPCB/LCSC header spellings are mapped)")
    r.add_argument("--catalog-date", metavar="ISO8601", help="the date you exported the catalog (required with --catalog)")
    r.add_argument("--catalog-authority", metavar="TEXT", help="who produced the catalog data, e.g. 'JLCPCB export'")
    r.add_argument("--catalog-supplier", metavar="NAME", help="the supplier label for sourcing entries, e.g. JLCPCB")
    r.add_argument("--regulatory-candidates", metavar="PATH", help="a regulatory candidate list other than the packaged ai_eda/regulatory/candidates.json")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("review", help="independent review only")
    v.add_argument("ir")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_review)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
