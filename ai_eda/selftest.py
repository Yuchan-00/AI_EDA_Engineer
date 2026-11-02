"""``ai-eda selftest``: prove an installation works by designing the divider end to end.

Invariant: the self-test judges nothing about the design. It answers one
question: did each external tool the installation is supposed to provide
actually run? Every stage line copies the status and message the run
recorded in its ``pipeline.json``; the self-test computes exactly two lines
itself - "Korean stage reports written" (``PASS`` when the four
``reports/*.md`` are on disk, else ``NOT_VERIFIED``) and "KiCad finds its
libraries" (below). It runs the documented command-line sequence (``ai-eda
new``, ``ai-eda run`` with the answers that select the ``divider`` template,
``ai-eda run --answer confirm_design=yes``) as subprocesses of this
interpreter in a folder of its own, so it exercises exactly what a user
types.

A required stage counts as "ran" only when its recorded status is a verdict
(``PASS`` / ``FAIL``) whose message names no missing tool
(:data:`MISSING_MARKERS`: "not available", "not found", ...) in its first
clause (up to the first ``; ``: the DRC stage appends the schematic-parity
reason there, which can quote kicad-cli's own stderr); any other
status - ``NOT_VERIFIED`` for whatever reason, or ``NOT RUN`` when the
pipeline stopped before the stage - fails the self-test. ARCHITECTURE, which
proposes and never judges, counts when its message says the template's
proposals were applied. ERC / DRC verdicts are copied as recorded: whether
KiCad's checks pass on this template is a fact about the design, not about
the installation, so a ``FAIL`` there still means the tool ran. The "KiCad
finds its libraries" line is ``PASS`` / ``FAIL`` from the violation types of
the latest ERC / DRC results in ``ir.json`` (``NOT RUN`` without one): any of
:data:`CONFIGURATION_VIOLATIONS`, which describe KiCad's own configuration
(its global library tables - missing until KiCad's first start or ``ai-eda
setup-kicad`` writes them - or a library they name that is not installed),
never the design, fails it.

Exit status: 0 when every required check ran (``--skip-kicad`` drops the
KiCad stages, ``--skip-ngspice`` the SPICE stage), 1 otherwise, 2 for a usage
or setup error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

#: the request and answers that select the ``divider`` template (README, 사용)
REQUEST = "12V 입력을 5V로 변환하는 회로"
FIRST_ANSWERS: tuple[str, ...] = (
    "application=bench supply",
    "jurisdiction=EU",
    "input_voltage=12 V",
    "output_voltage=5 V",
)
CONFIRM_ANSWER = "confirm_design=yes"
#: words in the first clause of a stage message that say a tool was missing rather than judged
MISSING_MARKERS: tuple[str, ...] = ("not available", "not found", "no ngspice", "unavailable")
#: ERC / DRC violation types that say KiCad's library tables do not name the libraries (measured in the install CI on
#: fresh KiCad 10 installs, 2026-10-14: ERC lib_symbol_issues + footprint_link_issues, DRC lib_footprint_issues per part)
CONFIGURATION_VIOLATIONS: frozenset[str] = frozenset({"lib_symbol_issues", "footprint_link_issues", "lib_footprint_issues"})


@dataclass
class Check:
    """One line of the self-test: what was expected to run and what the run record says."""

    name: str
    stage: str
    required: bool
    status: str = "NOT RUN"
    message: str = ""
    #: a stage that proposes rather than judges (ARCHITECTURE: a proposal is never evidence, so it stays NOT_VERIFIED)
    #: counts as ran when its message says the proposals were applied
    proposes: bool = False

    @property
    def ran(self) -> bool:
        if self.stage == "kicad-config":
            return self.status == "PASS"
        if self.proposes:
            return self.status != "FAIL" and "proposal(s) applied" in self.message
        if self.status not in ("PASS", "FAIL"):
            return False
        # the first clause only: the DRC message appends "; schematic parity not evaluated (<reason>)", and the reason
        # quotes whatever kicad-cli wrote on stderr ("Font '...' not found") after the tool ran
        first = self.message.split("; ", 1)[0].lower()
        return not any(m in first for m in MISSING_MARKERS)


def _outcomes(workdir: Path) -> dict[str, dict]:
    path = workdir / "pipeline.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    outcomes = (data.get("state") or {}).get("outcomes") or []
    return {o.get("stage"): o for o in outcomes if isinstance(o, dict)}


def _record(workdir: Path) -> dict:
    try:
        return json.loads((workdir / "pipeline.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _violations(workdir: Path) -> dict[str, tuple[dict, dict[str, int]]]:
    """``{"ERC" / "DRC": (latest kicad.erc / kicad.drc result in ir.json, {violation type: count})}`` (copied, not judged)."""
    try:
        ir = json.loads((workdir / "ir.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    latest: dict[str, dict] = {}
    for r in (ir.get("validation") or {}).get("results") or []:
        if isinstance(r, dict) and r.get("check_id") in ("kicad.erc", "kicad.drc"):
            latest["ERC" if r["check_id"] == "kicad.erc" else "DRC"] = r
    out: dict[str, tuple[dict, dict[str, int]]] = {}
    for label, r in sorted(latest.items()):
        details = r.get("details") or {}
        counts: dict[str, int] = {}
        for key in ("errors", "warnings", "other_severity", "unconnected_items", "schematic_parity"):
            for v in details.get(key) or []:
                kind = str(v.get("type", key)) if isinstance(v, dict) else key
                counts[kind] = counts.get(kind, 0) + 1
        out[label] = (r, counts)
    return out


def _violation_lines(workdir: Path) -> list[str]:
    """``ERC FAIL: lib_symbol_mismatch x2, ...`` for the latest kicad.erc / kicad.drc result in ir.json (copied, not judged)."""
    out: list[str] = []
    for label, (r, counts) in _violations(workdir).items():
        if counts:
            out.append(f"{label} {r.get('status')}: " + ", ".join(f"{k} x{n}" for k, n in sorted(counts.items())))
        else:
            out.append(f"{label} {r.get('status')}: {r.get('message', '')}")
    return out


def _configuration_check(workdir: Path, required: bool) -> Check:
    """``KiCad finds its libraries``: PASS when ERC / DRC ran without a :data:`CONFIGURATION_VIOLATIONS` type, else FAIL naming them."""
    check = Check("KiCad finds its libraries", "kicad-config", required)
    found = _violations(workdir)
    if not found:
        check.message = "no kicad-cli ERC / DRC result"
        return check
    bad = {k: n for _, counts in found.values() for k, n in counts.items() if k in CONFIGURATION_VIOLATIONS}
    if bad:
        check.status = "FAIL"
        check.message = (
            "KiCad did not resolve its libraries (its global library tables, or the libraries they name: `ai-eda doctor` shows the tables, "
            "`ai-eda setup-kicad` writes missing ones): " + ", ".join(f"{k} x{n}" for k, n in sorted(bad.items()))
        )
    else:
        check.status = "PASS"
        check.message = "no library-table violation in " + " / ".join(found)
    return check


def _cli(args: list[str], log: Path) -> int:
    """Run ``python -m ai_eda.cli <args>`` with this interpreter, appending its output to ``log``."""
    cmd = [sys.executable, "-m", "ai_eda.cli", *args]
    # the log is UTF-8 on every system (Windows pipes default to the legacy code page) and unbuffered, so a run that
    # hangs or is killed still shows how far it got
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ ai-eda {' '.join(args)}    [started {time.strftime('%H:%M:%S')}]\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)
        fh.write(f"[exit {proc.returncode} at {time.strftime('%H:%M:%S')}]\n")
    return proc.returncode


def run_selftest(workdir: Path | None, *, skip_kicad: bool = False, skip_ngspice: bool = False, keep: bool = False) -> int:
    """Design the divider in ``workdir`` (a new temporary folder when ``None``), print the checks, return the exit status."""
    temp = workdir is None
    root = Path(tempfile.mkdtemp(prefix="ai-eda-selftest-")) if temp else Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    project = root / "selftest"
    if (project / "ir.json").exists():
        print(f"selftest: {project / 'ir.json'} already exists; choose an empty --dir", file=sys.stderr)
        return 2
    log = root / "selftest.log"
    print(f"selftest: designing the 12 V -> 5 V divider in {project} (log: {log})")
    rc_new = _cli(["new", "selftest", "--dir", str(project), "--request", REQUEST], log)
    if rc_new != 0:
        print(f"selftest: 'ai-eda new' exited {rc_new}; see {log}", file=sys.stderr)
        return 1
    ir = str(project / "ir.json")
    first = ["run", ir, "--no-pdf"] + [x for a in FIRST_ANSWERS for x in ("--answer", a)]
    rc_first = _cli(first, log)
    rc_confirm = _cli(["run", ir, "--answer", CONFIRM_ANSWER], log)
    found = _outcomes(project)
    checks = [
        Check("circuit template applied", "architecture", True, proposes=True),
        Check("calculators re-derived", "calculation", True),
        Check("ngspice simulation", "spice", not skip_ngspice),
        Check("schematic compiled", "schematic", True),
        Check("board placed / routed / compiled", "pcb", True),
        Check("KiCad ERC (kicad-cli)", "erc", not skip_kicad),
        Check("KiCad DRC (kicad-cli)", "drc", not skip_kicad),
        Check("BOM / CPL / gerbers", "manufacturing_outputs", False),
    ]
    for c in checks:
        o = found.get(c.stage)
        if o is not None:
            c.status = str(o.get("status", "?"))
            c.message = str(o.get("message", ""))
    reports = sorted(p.name for p in (project / "reports").glob("*.md")) if (project / "reports").is_dir() else []
    report_check = Check("Korean stage reports written", "reports", True)
    report_check.status = "PASS" if len(reports) == 4 else "NOT_VERIFIED"
    report_check.message = ", ".join(reports) if reports else "no reports/*.md"
    checks.append(report_check)
    checks.append(_configuration_check(project, not skip_kicad))
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "ok  " if c.ran else ("--  " if not c.required else "FAIL")
        print(f"  [{mark}] {c.name:<{width}}  {c.status:<13} {c.message[:110]}")
    failed = [c for c in checks if c.required and not c.ran]
    print(f"selftest: first run exited {rc_first} (questions expected), confirm run exited {rc_confirm}")
    for line in _violation_lines(project):
        print(f"selftest: {line}")
    record = _record(project)
    if record.get("aborted"):
        print(f"selftest: the pipeline stopped in stage {record.get('aborted_stage')!r} (a defect, not a verdict)")
    if failed:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        print("selftest: last lines of the log:")
        for line in tail:
            print(f"    {line}")
    if failed:
        print("selftest: NOT working - " + ", ".join(c.name for c in failed) + f" (see {log}; ai-eda doctor names what is missing)")
    else:
        print("selftest: every required tool ran - the installation works")
    if temp and not keep and not failed:
        shutil.rmtree(root, ignore_errors=True)
    elif temp:
        print(f"selftest: kept {root}")
    return 1 if failed else 0


__all__ = ["CONFIGURATION_VIOLATIONS", "CONFIRM_ANSWER", "FIRST_ANSWERS", "REQUEST", "run_selftest"]
