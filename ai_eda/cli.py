"""Command line entry point.

    ai-eda doctor            check external tools (kicad-cli, KiCad's ngspice.dll, LLM key)
    ai-eda new NAME          create an empty project IR
    ai-eda run IR.json       run the pipeline until it blocks or finishes (exit 1 when blocked or the
                             final stage is FAIL; NOT_VERIFIED is exit 0 - nothing wrong, nothing proven)
    ai-eda review IR.json    run only the independent reviewer (exit 1 on any FAIL)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_eda import __version__


def cmd_doctor(_: argparse.Namespace) -> int:
    from ai_eda.errors import ToolUnavailableError
    from ai_eda.tools.kicad import KicadCli, KicadLibrary
    from ai_eda.tools.spice import NgspiceRunner, NgspiceShared
    import os

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
    print(f"LLM key   : {'set' if os.environ.get('OPENROUTER_API_KEY') else 'OPENROUTER_API_KEY not set'}")
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


def cmd_run(args: argparse.Namespace) -> int:
    from ai_eda.agents import AgentContext
    from ai_eda.tools.kicad import KicadCli, KicadLibrary
    from ai_eda.tools.spice import NgspiceShared
    from ai_eda.workflow import Orchestrator

    ir = _load(args.ir)
    workdir = Path(ir.project.workdir or Path(args.ir).parent)
    ctx = AgentContext(
        workdir=workdir,
        tools={"kicad_cli": KicadCli(), "kicad_library": KicadLibrary(), "spice": NgspiceShared()},
        answers=dict(kv.split("=", 1) for kv in (args.answer or [])),
    )
    state = Orchestrator(ctx).run(ir)
    for o in state.outcomes:
        print(f"{o.stage:<24} {o.status:<20} {o.message}")
    if state.blocked:
        print("\nBLOCKED - answer these to continue (pass with --answer key=value):")
        for q in state.open_questions:
            print(f"  [{q.key}] {q.question}")
    ir.save(args.ir)
    return run_exit_code(state)


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
    from ai_eda.review import IndependentReviewer

    ir = _load(args.ir)
    report = IndependentReviewer().review(ir, Path(ir.project.workdir or Path(args.ir).parent))
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

    sub.add_parser("doctor", help="check external tools").set_defaults(fn=cmd_doctor)

    n = sub.add_parser("new", help="create an empty project")
    n.add_argument("name")
    n.add_argument("--dir")
    n.add_argument("--request", help="natural-language design request")
    n.set_defaults(fn=cmd_new)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("ir")
    r.add_argument("--answer", action="append", metavar="KEY=VALUE")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("review", help="independent review only")
    v.add_argument("ir")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_review)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
