"""Thin wrapper around ``kicad-cli``.

Only *this* module produces ERC/DRC results. The results carry the KiCad
version, the report path and the hash of the file that was actually checked.

Facts about kicad-cli 10.0.6 that this module relies on (all verified by
running the binary, see ``docs/ARCHITECTURE.md``):

* Exit codes: 0 = check ran, 5 = ``--exit-code-violations`` hit, 3 = input
  not loadable (no report written), 1 = bad flag/value.
* ERC violations live under ``sheets[].violations[]``; DRC has three flat
  lists ``violations``, ``unconnected_items`` and ``schematic_parity``.
  Entry fields are ``type``, ``severity`` (``error``/``warning``), optional
  ``excluded: true`` (+ ``comment``) - everything else is localized text.
* ``--schematic-parity`` needs ``<stem>.kicad_sch`` next to the board. With
  no sibling schematic kicad-cli exits 0, prints a (localized) message to
  stderr and writes ``schematic_parity: []`` - i.e. parity silently "passes".
  stderr is empty on every successful run, so a non-empty stderr with parity
  requested is treated as "parity did not run".
* The default gerber export uses Protel extensions; with ``--no-protel-ext``
  every plot is ``*.gbr`` and the ``<stem>-job.gbrjob`` manifest lists exactly
  the files of that run, which is how :meth:`KicadCli.export_gerbers` tells
  fresh output from stale files in the same directory.
* Every export embeds a timestamp, so exports are never byte-identical
  between runs; only our own compiled files are.
* Zones: our boards store zones unfilled. ``pcb drc --refill-zones`` and
  ``pcb export gerbers --check-zones`` fill them in memory (stderr stays
  empty, also with ``--schematic-parity``), so DRC judges and the plots
  contain the pour; without the flags DRC sees no zone copper and the gerbers
  omit it entirely. Both are always passed and recorded.

Status policy (spec 12): a report with *any* violation - error or warning,
excluded or not - is FAIL. KiCad's severity is advisory; hiding warnings
behind PASS would hide unknowns, and ``lib_symbol_mismatch`` /
``lib_footprint_mismatch`` warnings are compiler defects. The counts stay in
the message and the lists in ``details``.

Project file (``<stem>.kicad_pro`` beside the schematic / board): KiCad keeps
the design rules there, so :meth:`KicadCli.run_erc` / :meth:`KicadCli.run_drc`
record whether one sat beside the checked file (``details["project_present"]``,
``project_path``, the hash of what was there at run time ``project_disk_hash``
and whether kicad-cli rewrote it ``project_rewritten``), and
:func:`run_erc_for` / :func:`run_drc_for` add ``project_hash`` +
``design_rules`` **only** when that file *is* the IR's fresh
``KICAD_PROJECT`` artifact (same path, generated from the current IR,
unchanged on disk before and after the run) - otherwise ``project_reason``
and no rules, so a stale or hand-edited project file can never pass as the
compiler's. Whether kicad-cli 10 actually *applies* the sibling project's
``board.design_settings.rules`` is NOT measured yet:
:data:`PROJECT_RULES_MEASURED_VERSIONS` lists the kicad-cli versions for
which ``tests/test_kicad_cli.py`` has demonstrated it (a 5 mm
``min_track_width`` produces ``track_width`` violations on the vertical-slice
board, 0.127 mm none, ERC output unchanged, project file not rewritten) and
is empty until that run is recorded; the capability check reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from ai_eda.errors import ToolExecutionError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Evidence, ValidationResult, ValidationStatus

#: Non-copper plots a fab needs (KiCad 10 canonical names). Courtyard / fab /
#: user layers are deliberately left out of the manufacturing set.
NON_COPPER_GERBER_LAYERS: tuple[str, ...] = (
    "F.Paste",
    "B.Paste",
    "F.Silkscreen",
    "B.Silkscreen",
    "F.Mask",
    "B.Mask",
    "Edge.Cuts",
)
#: The 2-layer fab set; boards with inner layers use :func:`gerber_layers`.
DEFAULT_GERBER_LAYERS: tuple[str, ...] = ("F.Cu", "B.Cu", *NON_COPPER_GERBER_LAYERS)

WARNING_POLICY = "any violation (error or warning, excluded or not) is FAIL; counts in message, lists in details"

#: kicad-cli versions measured to apply ``board.design_settings.rules`` of the sibling ``.kicad_pro`` in ``pcb drc``
#: (``tests/test_kicad_cli.py`` canary). Empty until the measurement is recorded: an unmeasured version can not
#: prove the fab minimums (``ai_eda.tools.manufacturing.capability``).
PROJECT_RULES_MEASURED_VERSIONS: frozenset[str] = frozenset()


def gerber_layers(copper: Iterable[str]) -> tuple[str, ...]:
    """The plot list for a board with the given copper layers (``F.Cu, In1.Cu .., B.Cu``) plus the fab set."""
    return (*copper, *NON_COPPER_GERBER_LAYERS)


def _windows_candidates() -> list[Path]:
    return [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "KiCad", Path("C:/Program Files/KiCad")]


def version_key(path: Path) -> tuple[int, ...]:
    """Numeric sort key for a KiCad version directory: ``10.0`` > ``9.0`` (lexicographic sorting gets this wrong)."""
    parts = path.name.split(".")
    if parts and all(p.isdigit() for p in parts):
        return tuple(int(p) for p in parts)
    return (-1,)  # non-version directories sort last


def version_dirs(base: Path) -> list[Path]:
    """Version directories under ``base`` (``.../Programs/KiCad``), highest version first."""
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir()), key=version_key, reverse=True)


def find_kicad_cli() -> str | None:
    found = shutil.which("kicad-cli")
    if found:
        return found
    for base in _windows_candidates():
        for ver in version_dirs(base):
            exe = ver / "bin" / "kicad-cli.exe"
            if exe.exists():
                return str(exe)
    return None


def install_root(binary: str | None) -> Path | None:
    """``<root>`` of a KiCad installation whose ``<root>/bin/kicad-cli(.exe)`` is ``binary``, if it has ``share/kicad``."""
    if not binary:
        return None
    exe = Path(binary)
    if not exe.is_absolute():
        resolved = shutil.which(binary)
        if not resolved:
            return None
        exe = Path(resolved)
    root = exe.resolve().parent.parent
    return root if (root / "share" / "kicad").is_dir() else None


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sibling_schematic(pcb: Path) -> Path:
    """The schematic ``kicad-cli pcb drc --schematic-parity`` will look for: ``<stem>.kicad_sch`` beside the board."""
    return pcb.with_suffix(".kicad_sch")


def sibling_project(path: Path) -> Path:
    """The project file KiCad looks for beside a schematic or board: ``<stem>.kicad_pro``."""
    return path.with_suffix(".kicad_pro")


def project_rules(path: Path) -> dict[str, float] | None:
    """``board.design_settings.rules`` of a ``.kicad_pro`` as floats (non-numeric entries left out); ``None`` when absent or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rules = data.get("board", {}).get("design_settings", {}).get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, dict):
        return None
    return {str(k): float(v) for k, v in rules.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _project_snapshot(checked: Path) -> dict:
    """What sits beside ``checked`` before kicad-cli runs: presence, path and hash of ``<stem>.kicad_pro``."""
    pro = sibling_project(checked)
    present = pro.is_file()
    return {"project_present": present, "project_path": str(pro), "project_disk_hash": _file_hash(pro) if present else None}


def _project_after(snapshot: dict) -> dict:
    """The snapshot plus whether kicad-cli rewrote (or created) the project file during the run."""
    pro = Path(snapshot["project_path"])
    now = _file_hash(pro) if pro.is_file() else None
    return {**snapshot, "project_rewritten": now != snapshot["project_disk_hash"]}


class KicadCli:
    tool_id = "kicad-cli"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or find_kicad_cli()
        self._version: str | None = None

    def available(self) -> bool:
        return self.binary is not None

    def _run(self, args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
        if not self.available():
            raise ToolUnavailableError("kicad-cli not found")
        # kicad-cli writes UTF-8 regardless of the console code page (cp949 on Korean Windows)
        proc = subprocess.run(
            [self.binary, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
        if proc.returncode not in (0, 5):  # 5 = ERC/DRC found violations (still a valid run)
            raise ToolExecutionError(f"kicad-cli {' '.join(args)} exited {proc.returncode}: {proc.stderr[-2000:]}")
        return proc

    def version(self) -> str:
        if self._version is None:
            self._version = self._run(["version"]).stdout.strip()
        return self._version

    # --- ERC / DRC -----------------------------------------------------------

    def run_erc(self, schematic: Path, report_path: Path) -> ValidationResult:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        project = _project_snapshot(schematic)
        self._run(["sch", "erc", "--format", "json", "--severity-all", "-o", str(report_path), str(schematic)])
        res = self._report_to_result("kicad.erc", schematic, report_path, "sheets")
        res.details.update(_project_after(project))
        return res

    def run_drc(self, pcb: Path, report_path: Path, schematic_parity: bool = False, refill_zones: bool = True) -> ValidationResult:
        """DRC on ``pcb``; with ``schematic_parity`` the sibling ``<stem>.kicad_sch`` must exist.

        ``refill_zones`` (default) passes ``--refill-zones`` so zones written
        unfilled by the compiler are filled in memory before the check - the
        state the fab files are plotted in; ``details["zones_refilled"]``
        records it. The result's ``details["schematic_parity_checked"]`` is
        True only when kicad-cli really evaluated parity (flag passed, sibling
        present, no stderr complaint); ``details["schematic_parity"]`` lists
        the parity findings and ``details["schematic_hash"]`` the schematic
        they were computed against.
        """
        report_path.parent.mkdir(parents=True, exist_ok=True)
        sch = sibling_schematic(pcb)
        if schematic_parity and not sch.is_file():
            raise ToolExecutionError(f"--schematic-parity requires {sch} next to {pcb}; kicad-cli would silently skip parity")
        args = ["pcb", "drc", "--format", "json", "--severity-all", "-o", str(report_path)]
        if refill_zones:
            args.append("--refill-zones")
        if schematic_parity:
            args.append("--schematic-parity")
        args.append(str(pcb))
        project = _project_snapshot(pcb)
        proc = self._run(args)
        if schematic_parity and proc.stderr.strip():
            # kicad-cli exits 0 and writes schematic_parity: [] when the schematic could not be netlisted;
            # the only sign is a localized stderr message, so do not match on its text.
            raise ToolExecutionError("schematic parity did not run: " + proc.stderr.strip()[-500:])
        res = self._report_to_result("kicad.drc", pcb, report_path, None)
        res.details.update(_project_after(project))
        res.details["zones_refilled"] = refill_zones
        res.details["schematic_parity_checked"] = schematic_parity
        if schematic_parity:
            res.details["schematic_path"] = str(sch)
            res.details["schematic_hash"] = _file_hash(sch)
        return res

    def _report_to_result(self, check_id: str, checked_file: Path, report_path: Path, sheets_key: str | None) -> ValidationResult:
        if not report_path.exists():
            raise ToolExecutionError(f"kicad-cli produced no report at {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        included = list(report.get("included_severities") or [])
        if "error" not in included:
            raise ToolExecutionError(f"{check_id} report at {report_path} does not include error severity (included: {included})")
        violations = self._collect_violations(report, sheets_key, report_path)
        errors = [v for v in violations if v.get("severity") == "error"]
        warnings = [v for v in violations if v.get("severity") == "warning"]
        others = [v for v in violations if v.get("severity") not in ("error", "warning")]
        # A human exclusion is an override that must stay visible: excluded violations still count.
        excluded = [v for v in violations if v.get("excluded")]
        # WARNING_POLICY: a warning is a finding, not a pass; unknown severities are not a pass either.
        status = ValidationStatus.FAIL if violations else ValidationStatus.PASS
        message = f"{len(errors)} error(s), {len(warnings)} warning(s)"
        if others:
            message += f", {len(others)} of other severity"
        if excluded:
            message += f" ({len(excluded)} marked excluded in the design, still counted)"
        details: dict = {
            "errors": errors,
            "warnings": warnings,
            "other_severity": others,
            "excluded": excluded,
            "included_severities": included,
            "ignored_checks": report.get("ignored_checks", []),
            "warning_policy": WARNING_POLICY,
        }
        if sheets_key is None:
            details["unconnected_items"] = list(report.get("unconnected_items", []))
            details["schematic_parity"] = list(report.get("schematic_parity", []))
        return ValidationResult(
            check_id=check_id,
            status=status,
            message=message,
            tool=self.tool_id,
            tool_version=report.get("kicad_version") or self.version(),
            artifact_hash=_file_hash(checked_file),
            evidence=[Evidence(description=f"{check_id} JSON report", path=str(report_path), content_hash=_file_hash(report_path))],
            details=details,
        )

    @staticmethod
    def _collect_violations(report: dict, sheets_key: str | None, report_path: Path | None = None) -> list[dict]:
        """Every violation of the report; a report whose container keys are not the expected ones is an error.

        An ERC report must have ``sheets[].violations``; a DRC report
        ``violations`` and ``unconnected_items`` (``schematic_parity`` only
        when parity was requested). Falling back to "no violations found"
        when the layout of the report changes would turn every future
        kicad-cli into a silent PASS.
        """
        out: list[dict] = []
        where = f" at {report_path}" if report_path else ""
        if sheets_key is not None:
            sheets = report.get(sheets_key)
            if not isinstance(sheets, list):
                raise ToolExecutionError(f"ERC report{where} has no {sheets_key!r} list (keys: {sorted(report)}); cannot read violations")
            for i, sheet in enumerate(sheets):
                if not isinstance(sheet, dict) or not isinstance(sheet.get("violations"), list):
                    raise ToolExecutionError(f"ERC report{where}: {sheets_key}[{i}] has no 'violations' list")
                out.extend(sheet["violations"])
            return out
        for key in ("violations", "unconnected_items"):
            if not isinstance(report.get(key), list):
                raise ToolExecutionError(f"DRC report{where} has no {key!r} list (keys: {sorted(report)}); cannot read violations")
            out.extend(report[key])
        parity = report.get("schematic_parity", [])
        if not isinstance(parity, list):
            raise ToolExecutionError(f"DRC report{where}: 'schematic_parity' is not a list")
        out.extend(parity)
        return out

    # --- exports -------------------------------------------------------------

    def export_gerbers(
        self, pcb: Path, out_dir: Path, layers: tuple[str, ...] | list[str] = DEFAULT_GERBER_LAYERS, check_zones: bool = True
    ) -> list[Path]:
        """Plot ``layers`` of ``pcb`` as ``*.gbr`` into ``out_dir``.

        ``check_zones`` (default) passes ``--check-zones`` so unfilled zones
        are filled before plotting - otherwise the copper plots silently omit
        every pour. Returns the ``<stem>-job.gbrjob`` manifest plus exactly
        the files it lists (so stale files from earlier runs in the same
        directory are never returned). Raises ToolExecutionError if the
        manifest or a listed file is missing.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        args = ["pcb", "export", "gerbers", "--no-protel-ext"]
        if check_zones:
            args.append("--check-zones")
        self._run([*args, "-l", ",".join(layers), "-o", str(out_dir) + os.sep, str(pcb)])
        job = out_dir / f"{pcb.stem}-job.gbrjob"
        if not job.is_file():
            raise ToolExecutionError(f"gerber export produced no job file at {job}")
        try:
            manifest = json.loads(job.read_text(encoding="utf-8"))
            listed = [str(entry["Path"]) for entry in manifest["FilesAttributes"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise ToolExecutionError(f"gerber job file {job} is not a valid manifest: {exc!r}") from exc
        files = [out_dir / name for name in listed]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise ToolExecutionError(f"gerber job file lists files that were not written: {missing}")
        return sorted(files, key=lambda p: p.name) + [job]

    def export_drill(self, pcb: Path, out_dir: Path) -> list[Path]:
        """Merged Excellon drill file ``<stem>.drl`` (written even when the board has no holes)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self._run(["pcb", "export", "drill", "--format", "excellon", "-o", str(out_dir) + os.sep, str(pcb)])
        drl = out_dir / f"{pcb.stem}.drl"
        if not drl.is_file():
            raise ToolExecutionError(f"drill export produced no {drl.name} in {out_dir}")
        return [drl]

    def export_netlist(self, schematic: Path, out_path: Path, fmt: str = "kicadsexpr") -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(["sch", "export", "netlist", "--format", fmt, "-o", str(out_path), str(schematic)])
        return out_path

    def export_spice_netlist(self, schematic: Path, out_path: Path) -> Path:
        return self.export_netlist(schematic, out_path, fmt="spice")


# --- pipeline helpers ---------------------------------------------------------------
# The orchestrator's ERC/DRC stages and the repair loop's RerunTool must run the
# checks identically; both call these so the parity decision lives in one place.


def fresh_artifact(ir: CircuitIR, kind: ArtifactKind) -> ArtifactRef:
    """``ir.artifacts[kind]`` only if it was generated from the current IR and is unchanged on disk.

    A tool run on a stale or hand-edited artifact would be stamped with the
    current IR hash and look like evidence about the design; refusing
    (ToolExecutionError) makes the repair loop regenerate first.
    """
    art = ir.artifacts.get(kind)
    if art is None:
        raise ToolExecutionError(f"no {kind} artifact to check")
    if art.is_stale(ir.content_hash()):
        raise ToolExecutionError(
            f"{kind} artifact was generated from IR {art.generated_from_ir_hash} but the IR is now {ir.content_hash()}; regenerate it first"
        )
    if not art.matches_disk():
        raise ToolExecutionError(f"{kind} artifact {art.path} on disk does not match its recorded hash; regenerate it first")
    return art


def project_details(ir: CircuitIR, res: ValidationResult) -> None:
    """Add ``project_hash`` + ``design_rules`` to an ERC / DRC result only when the project file it ran beside *is* the fresh ``KICAD_PROJECT`` artifact.

    The artifact must name the sibling path, be generated from the current
    IR, match its recorded hash on disk before the run (``project_disk_hash``)
    and still after it (kicad-cli must not have rewritten it); otherwise
    ``project_reason`` says why and no rules are recorded - a tool that ran
    beside some other project file proves nothing about the compiler's rules.
    """
    art = ir.artifacts.get(ArtifactKind.KICAD_PROJECT)
    sibling = Path(str(res.details.get("project_path") or ""))
    if art is None:
        res.details["project_reason"] = "no kicad_pro artifact registered for the IR"
        return
    if not res.details.get("project_present"):
        res.details["project_reason"] = f"no {sibling.name} beside the checked file"
        return
    if Path(art.path).resolve() != sibling.resolve():
        res.details["project_reason"] = f"kicad_pro artifact {art.path} is not {sibling.name} beside the checked file"
        return
    if art.is_stale(ir.content_hash()):
        res.details["project_reason"] = f"kicad_pro artifact was generated from IR {art.generated_from_ir_hash} but the IR is now {ir.content_hash()}"
        return
    if res.details.get("project_disk_hash") != art.content_hash:
        res.details["project_reason"] = "the project file beside the checked file does not match the kicad_pro artifact's recorded hash (edited after it was compiled)"
        return
    if res.details.get("project_rewritten") or not art.matches_disk():
        res.details["project_reason"] = "kicad-cli rewrote the project file during the run"
        return
    rules = project_rules(sibling)
    if rules is None:
        res.details["project_reason"] = "the project file carries no board.design_settings.rules"
        return
    res.details["project_hash"] = art.content_hash
    res.details["design_rules"] = rules


def run_erc_for(ir: CircuitIR, kicad: KicadCli, workdir: Path) -> ValidationResult:
    """ERC on the fresh ``ir.artifacts[SCHEMATIC]`` (see :func:`fresh_artifact`); the result is stamped with the IR hash."""
    art = fresh_artifact(ir, ArtifactKind.SCHEMATIC)
    res = kicad.run_erc(Path(art.path), workdir / "reports" / "erc.json")
    project_details(ir, res)
    res.ir_hash = ir.content_hash()
    return res


def parity_schematic(ir: CircuitIR) -> tuple[Path | None, str]:
    """The schematic parity can be evaluated against, or ``(None, reason)``.

    kicad-cli only ever reads ``<stem>.kicad_sch`` beside the board, so the
    schematic artifact must *be* that file; otherwise parity would silently
    compare the board with nothing.
    """
    pcb = ir.artifacts.get(ArtifactKind.PCB)
    sch = ir.artifacts.get(ArtifactKind.SCHEMATIC)
    if pcb is None:
        return None, "no PCB artifact"
    if sch is None:
        return None, "no schematic artifact"
    expected = sibling_schematic(Path(pcb.path))
    if Path(sch.path).resolve() != expected.resolve():
        return None, f"schematic artifact {sch.path} is not {expected.name} beside the board"
    if not expected.is_file():
        return None, f"{expected} is missing on disk"
    return expected, ""


def run_drc_for(ir: CircuitIR, kicad: KicadCli, workdir: Path) -> ValidationResult:
    """DRC on the fresh ``ir.artifacts[PCB]`` with schematic parity whenever the schematic artifact is the board's sibling."""
    art = fresh_artifact(ir, ArtifactKind.PCB)
    sch, reason = parity_schematic(ir)
    res = kicad.run_drc(Path(art.path), workdir / "reports" / "drc.json", schematic_parity=sch is not None)
    if sch is None:
        res.details["schematic_parity_reason"] = reason
    project_details(ir, res)
    res.ir_hash = ir.content_hash()
    return res
