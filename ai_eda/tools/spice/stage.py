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
  ``min`` / ``frequency`` / ``db_at`` / ``rms`` / ``db_rms`` /
  ``harmonic_dbc`` / ``am_depth``) and the verdict is ``|measured - nominal| <= max(tol_abs,
  tol_rel * |nominal|)`` (the limit actually used is in
  ``details["tolerance"]``). ``at`` between two samples is an interpolation
  (:meth:`~ai_eda.tools.spice.SpiceResult.interpolate`): the two bracketing
  samples are recorded in ``details["bracket"]`` and the verdict is PASS only
  when *both* neighbours are within tolerance too, FAIL only when both are
  outside on the same side, and otherwise UNRESOLVED ("the sweep grid is too
  coarse for this tolerance") - an interpolation error is never reported as
  a design deviation. ``frequency`` is the mean rising-edge frequency of the
  saved transient window (:func:`ai_eda.tools.spice.measure.rising_edge_frequency`):
  the edge count, the first and last edge time, the thresholds, vmin /
  vmax and the flatness floor go into ``details["frequency"]`` so the verdict
  is auditable, and a waveform without three countable edges, or whose swing
  is within the engine's own resolution (``reltol`` * level + ``vntol`` for a
  voltage, + ``abstol`` for a current: numerical ripple is not an
  oscillation), is FAIL ("no oscillation detected"), never PASS. The level
  and window reductions (``db_at`` / ``rms`` / ``db_rms`` / ``harmonic_dbc``
  / ``am_depth``, :mod:`ai_eda.tools.spice.measure`) put their audit data -
  the window, the largest sample step and its limit, the amplitudes, the
  reference level, the bias bounds - into ``details[<reduce value>]`` (the
  same rule as ``details["frequency"]``), and the expectation's ``params`` /
  ``reference_vector`` into ``details["params"]`` / ``details["reference_vector"]``;
  ``db_at`` is judged on the dB bracket of the two samples its magnitudes were
  read between, like ``at``; ``rms`` / ``db_rms`` / ``harmonic_dbc`` /
  ``am_depth`` are judged on the bracket their documented bias bounds leave
  (the grid's read-low bound up to ``f_max``, the harmonic's grid attenuation
  and leakage bound, the AM crest and envelope bounds; ``details["bias_bracket"]``)
  the same way - PASS only when all of it is inside the tolerance, FAIL only
  when all of it is outside on one side, UNRESOLVED otherwise, naming the
  bounds and the remedy: a measurement bias is never reported as a design
  deviation, nor a biased number as a PASS. A number they cannot give (a
  grid too coarse, a zero reference, no fundamental, a waveform that does not
  repeat at ``f0``, a window outside the saved samples) is
  FAIL for a human with the reason; a harmonic level at or below the
  engine's resolution is measured but never PASS (NOT_VERIFIED, "not
  resolved", ``details["unresolved"]``). No tolerance
  at all is UNRESOLVED; so is ``tol_rel`` alone on a nominal of 0 (a relative
  tolerance on zero is no tolerance). A vector the plot does not contain is
  FAIL ("vector not produced"; a plot without its own scale vector is
  "scale vector not produced", naming the scale, never the expectation's
  vector); an
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

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
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
from ai_eda.tools.spice.measure import ABSTOL, VNTOL, am_depth, harmonic_level, rising_edge_frequency, window_rms
from ai_eda.tools.spice.runner import Interpolation, SpiceAnalysis, SpiceResult, SpiceRunner

CHECK_ID = "spice"
#: the SI transients' check ids (``spice.si.<net>``): not expectations, never retired here
SI_CHECK_PREFIX = "spice.si"
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
    #: what a ``frequency`` / level / window reduction measured besides the number (edge count and times, thresholds,
    #: vmin / vmax; the window, the grid, the amplitudes, the bias bounds); the stage copies it into
    #: ``details[<reduce value>]`` (``details["frequency"]``, ``details["harmonic_dbc"]`` ...)
    extra: dict[str, Any] = field(default_factory=dict)
    #: why the number, though measured, is not resolved well enough to be evidence (a harmonic at the engine's
    #: resolution floor): the stage never turns such a reduction into a PASS - NOT_VERIFIED with this reason instead
    unresolved: str | None = None
    #: ``(low, high)``: where the true value lies given the measurement's documented one-sided biases (the grid's
    #: attenuation, a crest between samples, the envelope smear, leakage); ``None`` for an unbounded edge. The
    #: stage judges this bracket like the ``at`` bracket (:func:`judge`), never the bare number
    bias: tuple[float | None, float | None] | None = None
    #: the bounds behind ``bias`` and the remedy, for the UNRESOLVED message
    bias_note: str | None = None


def _frequency_details(edges: list[float], **scalars: float | None) -> dict[str, Any]:
    return {
        "edges": len(edges),
        "first_edge_s": edges[0] if edges else None,
        "last_edge_s": edges[-1] if edges else None,
        **scalars,
    }


#: reductions that read the result's scale (``time`` / ``frequency``)
_SCALE_REDUCES = frozenset({Reduce.AT, Reduce.FREQUENCY, Reduce.DB_AT, Reduce.RMS, Reduce.DB_RMS, Reduce.HARMONIC_DBC, Reduce.AM_DEPTH})
#: the level / window reductions (:func:`_level`)
_LEVEL_REDUCES = frozenset({Reduce.DB_AT, Reduce.RMS, Reduce.DB_RMS, Reduce.HARMONIC_DBC, Reduce.AM_DEPTH})


def _param(exp: Expectation, key: str) -> float:
    return float(exp.params[key].value)


def _abs_floor(vector_expr: str) -> float:
    """The absolute part of the engine's resolution for a vector expression: ``abstol`` for ``i(...)``, ``vntol`` otherwise."""
    return ABSTOL if vector_expr.strip().lower().startswith("i") else VNTOL


def _audit(obj: Any, *drop: str) -> dict[str, Any]:
    return {k: v for k, v in dataclasses.asdict(obj).items() if k not in ("problem", *drop)}


def _level(res: SpiceResult, exp: Expectation, vector: str, reference: str | None) -> Reduction:
    """``db_at`` / ``rms`` / ``db_rms`` / ``harmonic_dbc`` / ``am_depth`` of ``vector`` (module docstring of :mod:`ai_eda.tools.spice.measure`)."""
    kind = exp.reduce
    want = SpiceAnalysis.AC if kind == Reduce.DB_AT else SpiceAnalysis.TRAN
    if res.analysis != want:
        return Reduction(None, f"reduce={kind.value} needs a{'n' if want == SpiceAnalysis.AC else ''} {want.value} result, this is {res.analysis.value}")
    if reference is not None:
        try:
            ref_samples = res.vector(reference)
        except KeyError:
            return Reduction(None, f"reference vector not produced: {reference!r} is not in the {res.analysis.value} plot (vectors: {sorted(res.vectors)})")
        if any(not math.isfinite(x) for x in ref_samples):
            return Reduction(None, f"{reference} contains non-finite samples (the simulation did not produce a usable reference)")
    ref_level = _param(exp, "ref") if "ref" in exp.params else None
    reference_text = exp.reference_vector if reference is not None else (None if ref_level is None else f"ref {ref_level:.6g}")
    if kind == Reduce.DB_AT:
        at = float(exp.at.value)  # type: ignore[union-attr]
        iv = res.interpolate(vector, at)
        if reference is not None:
            ir_ = res.interpolate(reference, at)
            refs = (ir_.y0, ir_.y1, ir_.value)
        else:
            refs = (float(ref_level), float(ref_level), float(ref_level))  # type: ignore[arg-type]
        mags = (iv.y0, iv.y1, iv.value)
        extra: dict[str, Any] = {"at": at, "magnitude": iv.value, "reference": reference_text, "reference_magnitude": refs[2],
                                 "bracket_magnitudes": {"x0": iv.x0, "vector": [iv.y0, iv.y1], "reference": [refs[0], refs[1]]}}
        if any(not math.isfinite(m) or m <= 0.0 for m in (*mags, *refs)):
            return Reduction(None, f"a magnitude of 0 (or not finite) has no level in dB: |{exp.vector}| = {iv.value:.6g}, reference {refs[2]:.6g} at {res.scale}={at:g}", extra=extra)
        db = [20.0 * math.log10(m / r) for m, r in zip(mags, refs)]
        interp = Interpolation(value=db[2], exact=iv.exact, x0=iv.x0, y0=db[0], x1=iv.x1, y1=db[1], method=iv.method)
        return Reduction(db[2], None, interp, extra)
    times, values = res.scale_values(), res.vector(vector)
    t0, t1 = _param(exp, "t_start"), _param(exp, "t_stop")
    window = {"t_start": t0, "t_stop": t1}
    if kind in (Reduce.RMS, Reduce.DB_RMS):
        if "f_max" not in exp.params:  # the compiler refuses it; a hand-built expectation says so here, never "vector not produced"
            return Reduction(None, f"reduce={kind.value} needs params['f_max'] (the highest frequency the RMS must include): no grid guard without it")
        f_max = _param(exp, "f_max")
        rms = window_rms(times, values, t0, t1, f_max)
        extra = {**window, "rms": rms.value, "samples": rms.samples, "max_step": rms.max_step, "f_max": f_max,
                 "step_limit": rms.step_limit, "bias_bound_rel": rms.bias_bound_rel}
        if rms.problem is not None or rms.value is None:
            return Reduction(None, rms.problem or "no RMS measured", extra=extra)
        b = float(rms.bias_bound_rel or 0.0)
        note = (f"a component up to f_max = {f_max:g} Hz reads up to {100.0 * b:.3g} % low on the largest step {rms.max_step:.3g} s; "
                "lower the tran step or widen the tolerance")
        if kind == Reduce.RMS:
            return Reduction(rms.value, None, extra=extra, bias=(rms.value, rms.value / (1.0 - b)), bias_note=note)
        if reference is not None:
            ref = window_rms(times, ref_samples, t0, t1, f_max)
            if ref.problem is not None or ref.value is None:
                return Reduction(None, f"reference {exp.reference_vector}: {ref.problem or 'no RMS measured'}", extra=extra)
            ref_rms = ref.value
        else:
            ref_rms = float(ref_level)  # type: ignore[arg-type]
        extra.update(reference=reference_text, reference_rms=ref_rms)
        if ref_rms <= 0.0:
            return Reduction(None, "reference RMS is zero: no level in dB", extra=extra)
        if rms.value <= 0.0:
            return Reduction(None, f"the RMS of {exp.vector} over the window is zero: no level in dB", extra=extra)
        db = 20.0 * math.log10(rms.value / ref_rms)
        shift = -20.0 * math.log10(1.0 - b)
        # the vector's RMS reads low by up to b; a reference vector on the same grid too (a params["ref"] level does not)
        return Reduction(db, None, extra=extra, bias=(db - shift if reference is not None else db, db + shift), bias_note=note)
    if kind == Reduce.HARMONIC_DBC:
        k = int(round(_param(exp, "k")))
        h = harmonic_level(times, values, _param(exp, "f0"), k, t0, t1, abs_floor=_abs_floor(exp.vector))
        extra = {**window, "f0": _param(exp, "f0"), "k": k, **_audit(h, "dbc")}
        if h.problem is not None or h.dbc is None:
            return Reduction(None, h.problem or "no harmonic level measured", extra=extra)
        unresolved = None
        if h.ak_below_floor:
            unresolved = (f"harmonic {k} ({h.ak:.3g}) is at or below the engine's resolution {h.floor:.3g} (reltol * peak + the absolute tolerance): "
                          "its level is not resolved, so it is no evidence")
        note = (f"the grid attenuates harmonic {k} by up to {h.grid_attenuation_k_db:.3g} dB and the fundamental by up to {h.grid_attenuation_1_db:.3g} dB "
                f"on the largest step {h.max_step:.3g} s, and the periodicity residual ({100.0 * (h.periodicity_residual_rel or 0.0):.3g} % of the "
                f"fundamental) allows a leakage of {h.leakage_bound:.3g} into each bin; lower the tran step, or window whole periods of the steady state")
        return Reduction(h.dbc, None, extra=extra, unresolved=unresolved, bias=(h.dbc_low, h.dbc_high), bias_note=note)
    # AM_DEPTH
    d = am_depth(times, values, _param(exp, "f_carrier"), _param(exp, "f_mod"), t0, t1, abs_floor=_abs_floor(exp.vector))
    extra = {**window, "f_carrier": _param(exp, "f_carrier"), "f_mod": _param(exp, "f_mod"), "depth_percent": d.depth, **_audit(d, "depth")}
    if d.problem is not None or d.depth is None:
        return Reduction(None, d.problem or "no modulation depth measured", extra=extra)
    note = (f"each sampled crest reads up to {100.0 * (d.crest_bias_bound_rel or 0.0):.3g} % low on the largest step {d.max_step:.3g} s and the "
            f"envelope smear at f_carrier / f_mod = {_param(exp, 'f_carrier') / _param(exp, 'f_mod'):.6g} moves A_max / A_min by up to "
            f"{100.0 * d.envelope_bias_bound_rel:.3g} % of the envelope's swing; use a faster carrier or a lower tran step, or widen the tolerance")
    return Reduction(d.depth, None, extra=extra, bias=(d.depth_low, d.depth_high), bias_note=note)


def reduce_expectation(res: SpiceResult, exp: Expectation, vector: str, reference: str | None = None) -> Reduction:
    """The number ``exp.reduce`` picks from ``vector`` of ``res`` (with its bracket for ``at`` / ``db_at``), or why there is none.

    ``reference`` is ngspice's name of ``exp.reference_vector``
    (:func:`~ai_eda.compilers.spice.spice_vector_name`, which needs the IR -
    the stage maps it next to ``vector``); ``None`` when the expectation
    compares with ``params["ref"]`` or has no reference.
    """
    interpolation: Interpolation | None = None
    extra: dict[str, Any] = {}
    unresolved: str | None = None
    bias: tuple[float | None, float | None] | None = None
    bias_note: str | None = None
    if exp.reduce in _SCALE_REDUCES and res.scale is not None:
        # these reductions read the scale first: a plot without its scale vector is reported as such, never as the
        # expectation's vector being absent (that KeyError would name a vector that is present)
        try:
            res.scale_values()
        except KeyError:
            return Reduction(None, f"scale vector not produced: {res.scale!r} is not in the {res.analysis.value} plot (vectors: {sorted(res.vectors)})")
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
        elif exp.reduce == Reduce.FREQUENCY:
            if res.analysis != SpiceAnalysis.TRAN:
                return Reduction(None, f"reduce=frequency needs a tran result, this is {res.analysis.value}")
            # the flatness floor's absolute part is the engine's resolution for the vector's kind: a current vector
            # (``i(...)``) is measured in amperes
            abs_floor = ABSTOL if exp.vector.strip().lower().startswith("i") else VNTOL
            edge = rising_edge_frequency(res.scale_values(), res.vector(vector), abs_floor=abs_floor)
            extra = _frequency_details(edge.edges, low=edge.low, mid=edge.mid, high=edge.high, vmin=edge.vmin, vmax=edge.vmax, floor=edge.floor)
            if edge.problem is not None or edge.frequency is None:
                return Reduction(None, edge.problem or "no frequency measured", extra=extra)
            measured = edge.frequency
        elif exp.reduce in _LEVEL_REDUCES:
            # the window measurements check their own samples (finite, a time axis that never steps back); the
            # generic checks below still apply to the vector as a whole
            level = _level(res, exp, vector, reference)
            if level.problem is not None or level.measured is None:
                return level
            measured, interpolation, extra, unresolved = level.measured, level.interpolation, level.extra, level.unresolved
            bias, bias_note = level.bias, level.bias_note
        else:  # pragma: no cover - the enum is closed
            return Reduction(None, f"unknown reduce {exp.reduce!r}")
    except KeyError:
        return Reduction(None, f"vector not produced: {vector!r} is not in the {res.analysis.value} plot (vectors: {sorted(res.vectors)})")
    except ValueError as e:
        return Reduction(None, str(e), extra=extra)
    # every sample must be a number: builtins.max/min skip a NaN that is not first, and a waveform with a NaN or
    # an infinity in it is not a usable result wherever the bad sample sits
    try:
        samples = res.vector(vector)
    except KeyError:
        samples = []
    if any(not math.isfinite(x) for x in samples):
        return Reduction(None, f"{vector} contains non-finite samples (the simulation did not produce a usable waveform)")
    if exp.reduce in (Reduce.AT, Reduce.DB_AT) and any(not math.isfinite(x) for x in res.scale_values()):
        return Reduction(None, "the scale vector contains non-finite samples")
    if not math.isfinite(measured):
        return Reduction(None, f"{vector} {exp.reduce.value} is not finite ({measured!r})")
    return Reduction(float(measured), None, interpolation, extra, unresolved, bias, bias_note)


def _label_suffix(exp: Expectation, res: SpiceResult) -> str:
    """What a result message names besides ``<vector> <reduce>``: the sweep point, the reference, the window, f0 / k, the carrier."""
    if exp.reduce in (Reduce.AT, Reduce.DB_AT) and exp.at is not None:
        text = f" {res.scale}={float(exp.at.value):g}"
    else:
        text = ""
    if exp.reduce in (Reduce.DB_AT, Reduce.DB_RMS):
        text += f" re {exp.reference_vector}" if exp.reference_vector is not None else (f" re {_param(exp, 'ref'):g}" if "ref" in exp.params else "")
    if exp.reduce == Reduce.HARMONIC_DBC:
        text += f" k={int(round(_param(exp, 'k')))} of f0={_param(exp, 'f0'):g} Hz"
    if exp.reduce == Reduce.AM_DEPTH:
        text += f" (f_carrier={_param(exp, 'f_carrier'):g} Hz, f_mod={_param(exp, 'f_mod'):g} Hz)"
    if "t_start" in exp.params and "t_stop" in exp.params:
        text += f" over [{_param(exp, 't_start'):g}, {_param(exp, 't_stop'):g}] s"
    return text


def reduce_result(res: SpiceResult, exp: Expectation, vector: str, reference: str | None = None) -> tuple[float | None, str | None]:
    """``(measured, problem)`` of :func:`reduce_expectation`."""
    r = reduce_expectation(res, exp, vector, reference)
    return r.measured, r.problem


def judge(measured: float, exp: Expectation, interpolation: Interpolation | None = None,
          bias: tuple[float | None, float | None] | None = None) -> tuple[ValidationStatus, float | None, float]:
    """``(status, tolerance limit, deviation)`` for ``measured`` against ``exp.nominal``.

    UNRESOLVED without a usable tolerance (none given, or only ``tol_rel``
    on a nominal of 0). With an ``interpolation`` between two samples, the
    whole bracket ``[min(y0, y1), max(y0, y1)]`` is judged: PASS when all of
    it is within the limit, FAIL when all of it is outside on one side,
    UNRESOLVED in between (the grid cannot resolve the tolerance). A ``bias``
    bracket ``(low, high)`` (a window measurement's documented bias bounds;
    ``None`` is an unbounded edge) is judged the same way: a measurement
    error is never reported as a design deviation, nor a biased number as a PASS.
    """
    nominal = float(exp.nominal.value)
    deviation = abs(measured - nominal)
    limits = []
    if exp.tol_abs is not None:
        limits.append(abs(float(exp.tol_abs.value)))
    if exp.tol_rel is not None and nominal != 0.0:
        limits.append(abs(float(exp.tol_rel.value)) * abs(nominal))
    if not limits or any(not math.isfinite(x) for x in limits) or not math.isfinite(nominal):
        # a tolerance that is not a number is no tolerance (inf would pass anything, nan nothing)
        return ValidationStatus.UNRESOLVED, None, deviation
    limit = max(limits)
    if bias is not None:
        lo = min(measured, bias[0]) if bias[0] is not None else -math.inf
        hi = max(measured, bias[1]) if bias[1] is not None else math.inf
    elif interpolation is None or interpolation.exact:
        return (ValidationStatus.PASS if deviation <= limit else ValidationStatus.FAIL), limit, deviation
    else:
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
    ``spice.si.<net>`` (:data:`SI_CHECK_PREFIX`, the SI transients) is not an
    expectation and is left to :func:`ai_eda.tools.spice.si_check.spice_si_results`.
    """
    out: list[ValidationResult] = []
    prefix = f"{CHECK_ID}."
    for check_id, last in ir.validation.latest_by_check().items():
        if not check_id.startswith(prefix) or check_id[len(prefix):] in keep:
            continue
        if check_id == SI_CHECK_PREFIX or check_id.startswith(SI_CHECK_PREFIX + "."):
            continue  # the SI transients (ai_eda.tools.spice.si_check) retire their own results
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
    # each result may only claim the hash the runner itself computed for the file it loaded (the batch runner
    # loads a copy with the analysis card and reports the original's hash plus ``deck_hash`` of the copy)
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
            "provenance_kinds_used": sorted({k for k in (_kind(exp.nominal), _kind(exp.tol_abs), _kind(exp.tol_rel), _kind(exp.at), *(_kind(t) for t in exp.params.values())) if k}),
            "conditions": conditions,
            "engine": engine_stamp,
            "assumptions": assumptions,
        }
        if exp.params:
            details["params"] = {k: _num(t) for k, t in sorted(exp.params.items())}
        if exp.reference_vector is not None:
            details["reference_vector"] = exp.reference_vector
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
                    reference = None if exp.reference_vector is None else spice_vector_name(exp.reference_vector, ir)
                except CompileError as e:
                    vector, reduction = "", Reduction(None, str(e))
                else:
                    details["spice_vector"] = vector
                    if reference is not None:
                        details["spice_reference_vector"] = reference
                    reduction = reduce_expectation(res, exp, vector, reference)
                if reduction.extra:
                    # the audit trail of a frequency / level / window reduction, under the reduction's own name
                    details[exp.reduce.value] = dict(reduction.extra)
                if reduction.problem is not None:
                    status, message = ValidationStatus.FAIL, reduction.problem
                    details["repair"] = "human"
                else:
                    measured = reduction.measured
                    interp = reduction.interpolation
                    details["measured"] = measured
                    if interp is not None:
                        details["bracket"] = {"x0": interp.x0, "y0": interp.y0, "x1": interp.x1, "y1": interp.y1, "method": interp.method, "exact": interp.exact}
                    if reduction.bias is not None:
                        details["bias_bracket"] = {"low": reduction.bias[0], "high": reduction.bias[1], "why": reduction.bias_note}
                    status, limit, deviation = judge(measured, exp, interp, reduction.bias)  # type: ignore[arg-type]
                    details["tolerance"] = limit
                    details["deviation"] = deviation
                    label = f"{exp.vector} {exp.reduce.value}" + _label_suffix(exp, res)
                    unit = f" {exp.nominal.unit}" if exp.nominal.unit else ""
                    if status is ValidationStatus.UNRESOLVED and limit is None:
                        if exp.tol_rel is not None and float(exp.nominal.value) == 0.0:
                            message = f"no usable tolerance: nominal is 0 and only tol_rel is given (a relative tolerance on zero is no tolerance); {label} = {measured:.6g}{unit}"
                        else:
                            message = f"no tolerance: {label} = {measured:.6g}{unit} vs nominal {details['nominal']:.6g}{unit}, nothing to judge against"
                    elif status is ValidationStatus.UNRESOLVED and reduction.bias is not None:
                        lo_b, hi_b = reduction.bias
                        span = f"[{'-inf' if lo_b is None else f'{lo_b:.6g}'}, {'+inf' if hi_b is None else f'{hi_b:.6g}'}]{unit}"
                        message = (
                            f"{label} = {measured:.6g}{unit}, but the true value lies anywhere in {span} given the measurement's bias bounds, "
                            f"which straddles the tolerance +/- {limit:.3g}{unit} around nominal {details['nominal']:.6g}{unit}: not judged - {reduction.bias_note}"
                        )
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
                        elif reduction.unresolved is not None and status is ValidationStatus.PASS:
                            status = ValidationStatus.NOT_VERIFIED
                            details["unresolved"] = reduction.unresolved
                            message += f"; not evidence: {reduction.unresolved}"
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
