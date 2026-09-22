"""Run an IR's simulation setup on a real SPICE engine and judge its expectations.

This is the SPICE counterpart of :func:`ai_eda.tools.kicad.cli.run_erc_for`:
the orchestrator's SPICE stage (:class:`~ai_eda.agents.simulation.SimulationAgent`)
and the repair loop's ``RerunTool`` both call :func:`run_spice_for`, so a
re-run is exactly the original run.

Invariants:

* **Only a fresh netlist is simulated.** The ``SPICE_NETLIST`` artifact must
  be generated from the current IR and unchanged on disk
  (:func:`~ai_eda.tools.kicad.cli.fresh_artifact`); otherwise
  ``ToolExecutionError`` and the loop has to regenerate first. Every result
  is stamped with that file's hash (``artifact_hash``).
* **The analysis command is a pure function of the IR**
  (:func:`ai_eda.compilers.spice.analysis_command` via ``build_report``); the
  netlist carries no analysis cards.
* **Every number that reaches a verdict is recorded**, and so is the engine
  that produced it. All :class:`~ai_eda.tools.spice.SpiceResult` objects
  (vectors included) are written to ``<workdir>/spice/results.json`` and
  registered as the ``SPICE_RESULT`` artifact, together with
  ``SpiceRunner.engine_info()`` (build, code-model state, init-time settings
  and their hash, self-test) and the *conditions* of the run: component
  values at nominal, one temperature (``.temp`` or ngspice's 27 degC
  default), no tolerance corners. The rawfiles ngspice wrote are attached as
  :class:`~ai_eda.ir.Evidence` (path + sha256) to every result, together with
  ``results.json``. Rawfiles carry a date line, so they are evidence, not
  deterministic artifacts.
* **An expectation is judged, never interpreted.** ``measured`` comes from
  the reduction the IR asked for (``value`` / ``at`` / ``final`` / ``max`` /
  ``min``) and the verdict is ``|measured - nominal| <= max(tol_abs,
  tol_rel * |nominal|)`` (the limit actually used is in
  ``details["tolerance"]``). ``at`` between two samples is an interpolation
  (:meth:`~ai_eda.tools.spice.SpiceResult.interpolate`): the two bracketing
  samples are recorded in ``details["bracket"]`` and the verdict is PASS only
  when *both* neighbours are within tolerance too, FAIL only when both are
  outside on the same side, and otherwise UNRESOLVED ("the sweep grid is too
  coarse for this tolerance") - an interpolation error is never reported as
  a design deviation. No tolerance at all is UNRESOLVED; so is ``tol_rel``
  alone on a nominal of 0 (a relative tolerance on zero is no tolerance). A
  vector the plot does not contain is FAIL ("vector not produced"); an
  analysis that did not succeed makes the ``spice`` summary FAIL with
  ngspice's own lines and leaves its expectations NOT_VERIFIED - unless the
  *environment* prevented the run (``SpiceResult.unverifiable``: no place
  for the rawfile, a dead engine, XSPICE code models that did not load while
  the deck uses an ``A`` element), which is NOT_VERIFIED with the cause, not
  a verdict about the design.
* **An assumption is not evidence.** When the netlist rests on any
  ``assumption``-provenance value or decision (``report["assumptions"]``),
  every expectation that would PASS is NOT_VERIFIED instead, naming the
  assumptions; a FAIL stays FAIL.
* **The summary is the worst expectation.** ``spice`` = worst of the
  ``spice.<id>`` results (NOT_VERIFIED when there is none: analyses that ran
  without an expectation verify nothing), FAIL when any analysis failed.
  Its details list the analyses run, the engine info, the netlist hash, the
  conditions, the provenance kinds of the values that went into the netlist
  and what the compiler excluded or ignored (from
  :func:`ai_eda.compilers.spice.build_report`).
* **Retired expectations do not haunt the release.** Every ``spice.<id>``
  recorded earlier for an expectation that is no longer in the setup gets a
  superseding NOT_APPLICABLE result (:func:`retire_expectation_results`), so
  ``ir.validation.overall()`` reflects the current setup.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eda.compilers.spice import build_report, spice_vector_name
from ai_eda.errors import CompileError, ToolExecutionError, ToolUnavailableError
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitIR,
    Evidence,
    Expectation,
    Reduce,
    ValidationResult,
    ValidationStatus,
    worst_status,
)
from ai_eda.tools.kicad.cli import fresh_artifact
from ai_eda.tools.spice.runner import Interpolation, SpiceResult, SpiceRunner

CHECK_ID = "spice"
#: subdirectory of the workdir that holds ``results.json`` and one rawfile directory per analysis id
RESULTS_DIR = "spice"
RESULTS_FILE = "results.json"
#: ``results.json`` layout version (2: ``engine_info`` and ``conditions`` blocks)
RESULTS_FORMAT = "2"
#: ngspice's default analysis temperature when the deck has no ``.temp`` card (measured: "Doing analysis at TEMP = 27.000000")
NGSPICE_DEFAULT_TEMP_C = 27.0


def results_path(workdir: Path | str) -> Path:
    return Path(workdir) / RESULTS_DIR / RESULTS_FILE


def read_results(path: Path | str) -> dict[str, Any]:
    """The ``results.json`` a run wrote (``ValueError`` when it is not this module's layout)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != RESULTS_FORMAT or not isinstance(data.get("analyses"), dict):
        raise ValueError(f"{path} is not a spice results.json (format {RESULTS_FORMAT})")
    return data


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _num(traced) -> float | None:
    return None if traced is None else float(traced.value)


def _kind(traced) -> str | None:
    return None if traced is None else traced.provenance.kind.value


@dataclass
class Reduction:
    """What :func:`reduce_expectation` picked from a result: the number, or why there is none, and the samples it sits between."""

    measured: float | None
    problem: str | None
    interpolation: Interpolation | None = None


def reduce_expectation(res: SpiceResult, exp: Expectation, vector: str) -> Reduction:
    """The number ``exp.reduce`` picks from ``vector`` of ``res`` (with its bracket for ``at``), or why there is none."""
    interpolation: Interpolation | None = None
    try:
        if exp.reduce == Reduce.VALUE:
            measured = res.final(vector)
        elif exp.reduce == Reduce.AT:
            if exp.at is None:
                return Reduction(None, "reduce=at without 'at'")
            interpolation = res.interpolate(vector, float(exp.at.value))
            measured = interpolation.value
        elif exp.reduce == Reduce.FINAL:
            measured = res.final(vector)
        elif exp.reduce == Reduce.MAX:
            measured = res.max(vector)
        elif exp.reduce == Reduce.MIN:
            measured = res.min(vector)
        else:  # pragma: no cover - the enum is closed
            return Reduction(None, f"unknown reduce {exp.reduce!r}")
    except KeyError:
        return Reduction(None, f"vector not produced: {vector!r} is not in the {res.analysis.value} plot (vectors: {sorted(res.vectors)})")
    except ValueError as e:
        return Reduction(None, str(e))
    if not math.isfinite(measured):
        return Reduction(None, f"{vector} {exp.reduce.value} is not finite ({measured!r})")
    return Reduction(float(measured), None, interpolation)


def reduce_result(res: SpiceResult, exp: Expectation, vector: str) -> tuple[float | None, str | None]:
    """``(measured, problem)`` of :func:`reduce_expectation`."""
    r = reduce_expectation(res, exp, vector)
    return r.measured, r.problem


def judge(measured: float, exp: Expectation, interpolation: Interpolation | None = None) -> tuple[ValidationStatus, float | None, float]:
    """``(status, tolerance limit, deviation)`` for ``measured`` against ``exp.nominal``.

    UNRESOLVED without a usable tolerance (none given, or only ``tol_rel``
    on a nominal of 0). With an ``interpolation`` between two samples, the
    whole bracket ``[min(y0, y1), max(y0, y1)]`` is judged: PASS when all of
    it is within the limit, FAIL when all of it is outside on one side,
    UNRESOLVED in between (the grid cannot resolve the tolerance).
    """
    nominal = float(exp.nominal.value)
    deviation = abs(measured - nominal)
    limits = []
    if exp.tol_abs is not None:
        limits.append(abs(float(exp.tol_abs.value)))
    if exp.tol_rel is not None and nominal != 0.0:
        limits.append(abs(float(exp.tol_rel.value)) * abs(nominal))
    if not limits:
        return ValidationStatus.UNRESOLVED, None, deviation
    limit = max(limits)
    if interpolation is None or interpolation.exact:
        return (ValidationStatus.PASS if deviation <= limit else ValidationStatus.FAIL), limit, deviation
    lo, hi = interpolation.low, interpolation.high
    if hi - nominal <= limit and nominal - lo <= limit:
        return ValidationStatus.PASS, limit, deviation
    if lo - nominal > limit or nominal - hi > limit:
        return ValidationStatus.FAIL, limit, deviation
    return ValidationStatus.UNRESOLVED, limit, deviation


def retire_expectation_results(ir: CircuitIR, keep: set[str], status: ValidationStatus, why: str, **stamp: Any) -> list[ValidationResult]:
    """A superseding result for every recorded ``spice.<id>`` whose expectation id is not in ``keep``.

    ``ValidationState`` never forgets a check id, so an expectation that was
    removed or renamed would otherwise keep its last verdict in
    ``overall()`` forever. Returns the results (the caller appends them).
    """
    out: list[ValidationResult] = []
    prefix = f"{CHECK_ID}."
    for check_id, last in ir.validation.latest_by_check().items():
        if not check_id.startswith(prefix) or check_id[len(prefix):] in keep:
            continue
        if last.status is status or last.status is ValidationStatus.NOT_APPLICABLE:
            continue  # already superseded
        out.append(ValidationResult(check_id=check_id, status=status, message=why, details={"superseded": last.status.value}, **stamp))
    return out


def _uses_code_models(netlist_text: str) -> bool:
    """True when the deck has an XSPICE ``A`` element (which needs the code models to be loaded)."""
    for ln in netlist_text.split("\n")[1:]:
        s = ln.strip()
        if s and s[0] in "aA" and not s.startswith("."):
            return True
    return False


def run_spice_for(ir: CircuitIR, tools: dict[str, Any], workdir: Path | str) -> list[ValidationResult]:
    """Run every analysis of ``ir.simulation`` on the fresh ``SPICE_NETLIST`` artifact and judge every expectation.

    Returns ``[summary "spice", *"spice.<expectation id>", *retired]``,
    registers the ``SPICE_RESULT`` artifact (``<workdir>/spice/results.json``)
    and never changes the design. ``ToolUnavailableError`` when
    ``tools["spice"]`` is missing, unavailable or fails to initialise;
    ``ToolExecutionError`` when the netlist artifact is missing, stale or
    changed on disk.
    """
    setup = ir.simulation
    if setup is None:
        summary = ValidationResult(check_id=CHECK_ID, status=ValidationStatus.NOT_VERIFIED, message="no simulation setup in IR")
        return [summary, *retire_expectation_results(ir, set(), ValidationStatus.NOT_APPLICABLE, "no simulation setup in the IR any more")]
    runner = tools.get("spice")
    if not isinstance(runner, SpiceRunner) or not runner.available():
        raise ToolUnavailableError("no SPICE engine available (tools['spice'] is missing or unavailable)")
    art = fresh_artifact(ir, ArtifactKind.SPICE_NETLIST)
    netlist = Path(art.path)
    ir_hash = ir.content_hash()
    report = build_report(ir)  # the artifact is fresh, so this is the report of exactly that netlist
    out_dir = Path(workdir) / RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, SpiceResult] = {}
    for spec in setup.analyses:
        results[spec.id] = runner.run(netlist, spec.kind, out_dir / spec.id, command=report["analyses"][spec.id])
    engine_info = dict(runner.engine_info())
    engine_version = next((r.engine_version for r in results.values()), None) or str(engine_info.get("version") or runner.version())
    engine_info.setdefault("engine", runner.engine)
    engine_info["version"] = engine_version
    temperature = report["temperature_c"]
    conditions = {
        "values": "nominal (SpiceBinding.value of every part; no tolerance corners, no Monte Carlo)",
        "corners": "not simulated",
        "temperature_c": NGSPICE_DEFAULT_TEMP_C if temperature is None else float(temperature),
        "temperature_source": "ngspice default (no .temp card)" if temperature is None else "SimulationSetup.temperature_c (.temp card)",
        "component_tolerances_recorded": sorted(c.ref for c in ir.components if "tolerance" in c.electrical),
    }

    payload: dict[str, Any] = {
        "format": RESULTS_FORMAT,
        "engine": runner.engine,
        "engine_version": engine_version,
        "engine_info": engine_info,
        "conditions": conditions,
        "netlist_path": str(netlist),
        "netlist_hash": art.content_hash,
        "ir_hash": ir_hash,
        "netlist_report": {
            k: report[k]
            for k in (
                "elements", "excluded", "ignored_pins", "nets_touching_only_excluded", "ground_net", "models", "value_sources",
                "accepted_provenance_kinds", "assumptions", "inexact_numbers", "unmodelled_numbers", "stimuli", "temperature_c",
            )
        },
        "analyses": {
            spec.id: {"kind": spec.kind.value, "command": report["analyses"][spec.id], "result": results[spec.id].model_dump(mode="json")}
            for spec in setup.analyses
        },
    }
    path = results_path(workdir)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    results_hash = _sha256(path)
    ir.artifacts[ArtifactKind.SPICE_RESULT] = ArtifactRef(
        kind=ArtifactKind.SPICE_RESULT,
        path=str(path),
        content_hash=results_hash,
        generated_from_ir_hash=ir_hash,
        generator=runner.engine,
        generator_version=engine_version,
    )
    results_evidence = Evidence(description="spice results.json (every SpiceResult of this run)", path=str(path), content_hash=results_hash)
    netlist_evidence = Evidence(description="SPICE netlist that was simulated", path=str(netlist), content_hash=art.content_hash)
    report_path = netlist.with_name(netlist.name + ".report.json")
    report_evidence = Evidence(description="netlist compile report", path=str(report_path), content_hash=_sha256(report_path)) if report_path.is_file() else None

    def raw_evidence(res: SpiceResult, analysis_id: str) -> Evidence | None:
        if not res.raw_output_path:
            return None
        return Evidence(description=f"ngspice rawfile of analysis {analysis_id} ({res.command})", path=res.raw_output_path, content_hash=res.raw_output_hash)

    failed_analyses: dict[str, list[str]] = {aid: list(r.errors) for aid, r in results.items() if not r.succeeded}
    # each result may only claim the hash the runner itself computed for the file it loaded
    for aid, r in results.items():
        if r.netlist_hash != art.content_hash:
            failed_analyses.setdefault(aid, []).append(f"runner loaded a file with hash {r.netlist_hash}, the artifact records {art.content_hash}")
    # an environment that prevented the run is not a verdict about the design
    unverifiable: dict[str, str] = {aid: r.unverifiable for aid, r in results.items() if aid in failed_analyses and r.unverifiable}
    if failed_analyses and engine_info.get("codemodels_loaded") is False and _uses_code_models(netlist.read_text(encoding="utf-8", errors="replace")):
        cause = "the XSPICE code models did not load in this engine (" + "; ".join(engine_info.get("codemodel_errors") or ["no details"]) + ") and the deck uses an A element"
        for aid in failed_analyses:
            unverifiable.setdefault(aid, cause)
    assumptions: list[str] = list(report["assumptions"])
    engine_stamp = {k: engine_info.get(k) for k in ("build", "codemodels_loaded", "settings_hash", "dll_path")}

    expectation_results: list[ValidationResult] = []
    for exp in setup.expectations:
        res = results.get(exp.analysis_id)
        details: dict[str, Any] = {
            "analysis_id": exp.analysis_id,
            "vector": exp.vector,
            "reduce": exp.reduce.value,
            "at": _num(exp.at),
            "nominal": _num(exp.nominal),
            "unit": exp.nominal.unit,
            "tol_abs": _num(exp.tol_abs),
            "tol_rel": _num(exp.tol_rel),
            "requirement_id": exp.requirement_id,
            "provenance_kinds_used": sorted({k for k in (_kind(exp.nominal), _kind(exp.tol_abs), _kind(exp.tol_rel), _kind(exp.at)) if k}),
            "conditions": conditions,
            "engine": engine_stamp,
            "assumptions": assumptions,
        }
        evidence = [e for e in (results_evidence, netlist_evidence) if e]
        if res is None:
            status, message = ValidationStatus.FAIL, f"expectation {exp.id} names analysis {exp.analysis_id!r}, which is not in the simulation setup"
            details["repair"] = "human"
        else:
            details["command"] = res.command
            details["plot_name"] = res.plot_name
            ev = raw_evidence(res, exp.analysis_id)
            if ev is not None:
                evidence.insert(0, ev)
            if exp.analysis_id in unverifiable:
                status, message = ValidationStatus.NOT_VERIFIED, f"analysis {exp.analysis_id} could not be run here: {unverifiable[exp.analysis_id]}"
                details["unverifiable"] = unverifiable[exp.analysis_id]
            elif exp.analysis_id in failed_analyses:
                status, message = ValidationStatus.NOT_VERIFIED, f"analysis {exp.analysis_id} did not succeed: " + "; ".join(failed_analyses[exp.analysis_id][:3])
            else:
                try:
                    vector = spice_vector_name(exp.vector, ir)
                except CompileError as e:
                    vector, reduction = "", Reduction(None, str(e))
                else:
                    details["spice_vector"] = vector
                    reduction = reduce_expectation(res, exp, vector)
                if reduction.problem is not None:
                    status, message = ValidationStatus.FAIL, reduction.problem
                    details["repair"] = "human"
                else:
                    measured = reduction.measured
                    interp = reduction.interpolation
                    details["measured"] = measured
                    if interp is not None:
                        details["bracket"] = {"x0": interp.x0, "y0": interp.y0, "x1": interp.x1, "y1": interp.y1, "method": interp.method, "exact": interp.exact}
                    status, limit, deviation = judge(measured, exp, interp)  # type: ignore[arg-type]
                    details["tolerance"] = limit
                    details["deviation"] = deviation
                    label = f"{exp.vector} {exp.reduce.value}" + (f" {res.scale}={details['at']:g}" if exp.reduce == Reduce.AT else "")
                    unit = f" {exp.nominal.unit}" if exp.nominal.unit else ""
                    if status is ValidationStatus.UNRESOLVED and limit is None:
                        if exp.tol_rel is not None and float(exp.nominal.value) == 0.0:
                            message = f"no usable tolerance: nominal is 0 and only tol_rel is given (a relative tolerance on zero is no tolerance); {label} = {measured:.6g}{unit}"
                        else:
                            message = f"no tolerance: {label} = {measured:.6g}{unit} vs nominal {details['nominal']:.6g}{unit}, nothing to judge against"
                    elif status is ValidationStatus.UNRESOLVED:
                        message = (
                            f"{label} = {measured:.6g}{unit} interpolated ({interp.method}) between ({interp.x0:g}, {interp.y0:.6g}) and ({interp.x1:g}, {interp.y1:.6g}), "  # type: ignore[union-attr]
                            f"which spread more than the tolerance +/- {limit:.3g}{unit} around nominal {details['nominal']:.6g}{unit}: "
                            "the sweep grid is too coarse to judge this; put 'at' on a sweep point or refine the sweep"
                        )
                    else:
                        message = (
                            f"{label} = {measured:.6g}{unit}, nominal {details['nominal']:.6g}{unit} "
                            f"+/- {limit:.3g}{unit} (deviation {deviation:.3g}{unit})"
                        )
                        if status is ValidationStatus.FAIL:
                            details["repair"] = "human"
                        elif assumptions:
                            status = ValidationStatus.NOT_VERIFIED
                            message += f"; not evidence: the netlist rests on {len(assumptions)} assumption(s) {assumptions} that nobody confirmed"
        expectation_results.append(
            ValidationResult(
                check_id=f"{CHECK_ID}.{exp.id}",
                status=status,
                message=message,
                tool=runner.engine,
                tool_version=engine_version,
                artifact_hash=art.content_hash,
                ir_hash=ir_hash,
                evidence=evidence,
                details=details,
            )
        )

    summary_details: dict[str, Any] = {
        "engine": runner.engine,
        "engine_version": engine_version,
        "engine_info": engine_info,
        "conditions": conditions,
        "netlist_path": str(netlist),
        "netlist_hash": art.content_hash,
        "results_path": str(path),
        "results_hash": results_hash,
        "analyses": {
            aid: {
                "kind": r.analysis.value, "command": r.command, "succeeded": r.succeeded, "plot_name": r.plot_name, "n_points": r.n_points,
                "scale": r.scale, "elapsed_s": r.elapsed_s, "timed_out": r.timed_out, "raw_output_path": r.raw_output_path, "raw_output_hash": r.raw_output_hash,
                "errors": list(r.errors), "unverifiable": r.unverifiable,
            }
            for aid, r in results.items()
        },
        "expectations": {r.check_id.removeprefix(f"{CHECK_ID}."): r.status.value for r in expectation_results},
        "netlist_provenance_kinds": report["accepted_provenance_kinds"],
        "assumptions": assumptions,
        "excluded": report["excluded"],
        "ignored_pins": report["ignored_pins"],
        "value_sources": report["value_sources"],
        "nets_touching_only_excluded": report["nets_touching_only_excluded"],
        "inexact_numbers": report["inexact_numbers"],
        "unmodelled_numbers": report["unmodelled_numbers"],
    }
    summary_evidence = [e for e in (results_evidence, netlist_evidence, report_evidence) if e]
    summary_evidence += [ev for aid, r in results.items() for ev in [raw_evidence(r, aid)] if ev is not None]
    ran = ", ".join(f"{aid} ({r.command}, {r.n_points} pt)" for aid, r in results.items() if r.succeeded)
    if failed_analyses and set(failed_analyses) <= set(unverifiable):
        summary_details["unverifiable"] = unverifiable
        summary_details["errors"] = failed_analyses
        message = "; ".join(f"analysis {aid} could not be run here: {why}" for aid, why in unverifiable.items())
        status = ValidationStatus.NOT_VERIFIED
    elif failed_analyses:
        summary_details["repair"] = "human"
        summary_details["errors"] = failed_analyses
        message = "; ".join(f"analysis {aid} failed: " + " | ".join(errs[:4]) for aid, errs in failed_analyses.items())
        status = ValidationStatus.FAIL
    else:
        status = worst_status(r.status for r in expectation_results)
        counts: dict[str, int] = {}
        for r in expectation_results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        judged = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "none"
        message = f"{len(results)} analysis(es) run [{ran or 'none'}], {len(expectation_results)} expectation(s): {judged}"
        if status is ValidationStatus.FAIL:
            summary_details["repair"] = "human"
            failing = [r.check_id for r in expectation_results if r.status is ValidationStatus.FAIL]
            message += f"; failed: {failing}"
        elif assumptions and expectation_results:
            message += f"; the netlist rests on assumption(s) {assumptions}: not evidence until confirmed"
    summary = ValidationResult(
        check_id=CHECK_ID,
        status=status,
        message=message,
        tool=runner.engine,
        tool_version=engine_version,
        artifact_hash=art.content_hash,
        ir_hash=ir_hash,
        evidence=summary_evidence,
        details=summary_details,
    )
    retired = retire_expectation_results(
        ir, {e.id for e in setup.expectations}, ValidationStatus.NOT_APPLICABLE,
        "expectation is no longer in the simulation setup (superseded by this run)", tool=runner.engine, tool_version=engine_version, ir_hash=ir_hash,
    )
    return [summary, *expectation_results, *retired]


__all__ = [
    "CHECK_ID",
    "NGSPICE_DEFAULT_TEMP_C",
    "RESULTS_DIR",
    "RESULTS_FILE",
    "RESULTS_FORMAT",
    "Reduction",
    "judge",
    "read_results",
    "reduce_expectation",
    "reduce_result",
    "results_path",
    "retire_expectation_results",
    "run_spice_for",
]
