"""AM / SFFM stimuli and the level / window reductions through the compiler, the deck check and ngspice.

Without ngspice: the compiler writes ``AM(VA VO MF FC TD)`` / ``SFFM(VO VA FC MDI FS)``
only with every parameter, refuses a zero ``mf`` / ``fc`` (AM) or ``fc`` /
``fs`` (SFFM) - ngspice-42 silently substitutes a default of its own there
(measured: ``AM(1 2 0 20k 0)`` peaks at 3.0 V, not 2.0 V; ``SFFM(0 1 1k 5 0)``
is not the 1 kHz sine) -, and refuses every malformed level / window
expectation with the sentence that says why; the runner's
:func:`validate_deck` accepts the AM / SFFM lines, and no analysis,
``.save``, ``.meas`` or ``.four`` card ever reaches the deck.

With ngspice (real decks compiled from IRs, run through the SPICE stage;
measured on ngspice-42, Ubuntu libngspice0, 2026-09-28; the whole module runs in
about 2 s):

* ``AM(1 2 1k 100k 0)``: ``am_depth`` over the second modulation period
  49.969 % (1 / VO = 50 %; the envelope-smear bound at ratio 100 allows
  0.06 % of VA on each amplitude), judged PASS at 50 +/- 0.2 %;
* ``AM(1 2 1k 20k 0)`` (ratio 20, the smallest the compiler admits) reads
  49.24 %: the documented envelope bias (up to 1.54 % of VA on each
  amplitude at ratio 20); the verdict is judged on the bias bracket
  [49.22, 50.03] %, so 50 +/- 0.2 % is UNRESOLVED there (never a FAIL of a
  correct wave, never a PASS of a 50.70 % one), which is why an accuracy
  claim needs a faster carrier;
* ``SFFM(0 1 1k 0 10k)`` (MDI 0): ``frequency`` = 1000.0 Hz = FC;
* ``PULSE(0 1 0 1n 1n 0.5m 1m)``: ``harmonic_dbc`` k = 3 = -9.5424 dBc
  (20 log10(1/3) = -9.5424) within 0.001 dB; k = 2 far below (about -76 dBc,
  under the engine's resolution floor, so never PASS: NOT_VERIFIED
  "not resolved");
* ``SINE(0 1 1k)``: ``rms`` over two periods 0.707104 V (1 / sqrt(2) = 0.707107: the 1 us grid reads it 2e-6 low);
* an RC low-pass at its corner (1 kHz): ``db_at`` -3.0103 dB against
  ``ref`` 1 V and against ``reference_vector`` v(IN);
* a 10 k / 10 k divider: ``db_rms`` of v(OUT) against v(IN) -6.0206 dB.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import build, build_report
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    ArtifactKind,
    CircuitIR,
    Component,
    Expectation,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    llm_generated,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.simulation import REDUCE_PARAMS, STIMULUS_PARAMS
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis
from ai_eda.tools.spice.ngspice_shared import validate_deck
from ai_eda.tools.spice.stage import run_spice_for

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")
DERIVED = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")


def u(value, unit=None):
    return user_requirement(value, unit)


def _part(ref: str, dev: SpiceDevice, value: float, unit: str) -> Component:
    pins = [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=USER) for i in (1, 2)]
    return Component(ref=ref, value=ref, pins=pins, provenance=DERIVED, spice=SpiceBinding(device=dev, value=u(value, unit), provenance=USER))


def _net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=DERIVED)


def tran(step: float, stop: float, start: float | None = None) -> AnalysisSpec:
    params = {"step": u(step, "s"), "stop": u(stop, "s")}
    if start is not None:
        params["start"] = u(start, "s")
    return AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params=params, provenance=USER)


def source_ir(tmp_path: Path | None, kind: StimulusKind, params: dict[str, float], analysis: AnalysisSpec, *expectations: Expectation) -> CircuitIR:
    """A source ``VS`` of ``kind`` driving a 1 k load between OUT and GND."""
    ir = CircuitIR(project=ProjectMeta(id="rfsrc", name="rfsrc", workdir=None if tmp_path is None else str(tmp_path)))
    ir.components = [_part("R1", SpiceDevice.R, 1000.0, "ohm")]
    ir.nets = [_net("OUT", ("R1", "1")), _net("GND", ("R1", "2"), kind=NetKind.GROUND)]
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VS", source="voltage", net="OUT", reference_net="GND", kind=kind, params={k: u(v) for k, v in params.items()}, provenance=USER)],
        analyses=[analysis], expectations=list(expectations),
    )
    return ir


def exp(id: str, reduce: Reduce, nominal: float, unit: str | None, *, analysis: str = "tran", vector: str = "v(OUT)", tol_abs: float | None = None,
        tol_rel: float | None = None, reference: str | None = None, at: float | None = None, **params) -> Expectation:
    return Expectation(
        id=id, analysis_id=analysis, vector=vector, reduce=reduce, nominal=u(nominal, unit), tol_abs=None if tol_abs is None else u(tol_abs, unit),
        tol_rel=None if tol_rel is None else u(tol_rel), reference_vector=reference, at=None if at is None else u(at, "Hz"),
        params={k: u(v) for k, v in params.items()}, provenance=USER,
    )


AM_100 = {"va": 1.0, "vo": 2.0, "mf": 1e3, "fc": 100e3, "td": 0.0}
AM_20 = {**AM_100, "fc": 20e3}


def depth(id: str, fc: float, nominal: float = 50.0, tol: float = 0.2) -> Expectation:
    return exp(id, Reduce.AM_DEPTH, nominal, "percent", tol_abs=tol, f_carrier=fc, f_mod=1e3, t_start=1e-3, t_stop=2e-3)


def _run(ir: CircuitIR, tmp_path: Path) -> dict:
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"spice": runner}))
    t0 = time.perf_counter()
    results = run_spice_for(ir, {"spice": runner}, tmp_path)
    for r in results:
        print(f"  {r.check_id:<22} {r.status:<14} {r.message}")
    print(f"  ({time.perf_counter() - t0:.2f} s)")
    return {r.check_id: r for r in results}


# --------------------------------------------------------------------------- the compiler, without ngspice


def test_am_and_sffm_lines_are_written_in_ngspice_order_and_pass_the_deck_check():
    assert STIMULUS_PARAMS[StimulusKind.AM] == ("va", "vo", "mf", "fc", "td") and STIMULUS_PARAMS[StimulusKind.SFFM] == ("vo", "va", "fc", "mdi", "fs")
    am = build(source_ir(None, StimulusKind.AM, AM_100, tran(1e-7, 3e-3)))
    assert am == "rfsrc\nR1 OUT 0 1k\nVVS OUT 0 AM(1 2 1k 100k 0)\n.end\n"
    sffm = build(source_ir(None, StimulusKind.SFFM, {"vo": 0.0, "va": 1.0, "fc": 1e3, "mdi": 0.0, "fs": 10e3}, tran(1e-6, 5e-3)))
    assert "VVS OUT 0 SFFM(0 1 1k 0 10k)\n" in sffm
    for text in (am, sffm, "t\nV1 A 0 AM(0.5 2 100k 900meg 0)\nV2 B 0 SFFM(0 1 1k 5 10k) AC 1\nR1 A B 1k\nR2 B 0 1k\n.end\n"):
        assert validate_deck(text)[0] == []
        assert not any(tok in text.lower() for tok in (".save", ".meas", ".four", ".tran", ".control"))


def test_modulated_sources_refuse_what_ngspice_would_silently_replace():
    def refused(kind: StimulusKind, params: dict[str, float], match: str) -> None:
        with pytest.raises(CompileError, match=match):
            build(source_ir(None, kind, params, tran(1e-6, 5e-3)))

    refused(StimulusKind.AM, {**AM_100, "mf": 0.0}, r"stimulus VS: am param mf must be > 0, got 0.0 \(ngspice-42 silently replaces a zero one")
    refused(StimulusKind.AM, {**AM_100, "fc": -1.0}, "am param fc must be > 0")
    refused(StimulusKind.AM, {**AM_100, "td": -1e-3}, r"am param td \(a delay\) must be >= 0")
    refused(StimulusKind.SFFM, {"vo": 0.0, "va": 1.0, "fc": 1e3, "mdi": 5.0, "fs": 0.0}, "sffm param fs must be > 0")
    refused(StimulusKind.SFFM, {"vo": 0.0, "va": 1.0, "fc": 0.0, "mdi": 5.0, "fs": 10e3}, "sffm param fc must be > 0")
    refused(StimulusKind.AM, {k: v for k, v in AM_100.items() if k != "td"}, r"am needs params \['va', 'vo', 'mf', 'fc', 'td'\], missing \['td'\]")
    refused(StimulusKind.SFFM, {"vo": 0.0, "va": 1.0, "fc": 1e3, "mdi": 5.0, "fs": 10e3, "phasec": 0.0}, r"unexpected params \['phasec'\]")
    # VO = 0 (DSB-SC) and MDI = 0 are honoured by ngspice (measured): they compile
    assert "AM(1 0 1k 100k 0)" in build(source_ir(None, StimulusKind.AM, {**AM_100, "vo": 0.0}, tran(1e-7, 3e-3)))
    ir = source_ir(None, StimulusKind.AM, AM_100, tran(1e-7, 3e-3))
    ir.simulation.stimuli[0].params["fc"] = llm_generated(100e3, "Hz")
    with pytest.raises(CompileError, match="llm_generated"):
        build(ir)


def _divider(tmp_path: Path | None = None, *expectations: Expectation, analyses: list[AnalysisSpec] | None = None) -> CircuitIR:
    """IN -R1 10k- OUT -R2 10k- GND, a 1 V / 1 kHz sine with AC 1 on IN."""
    ir = CircuitIR(project=ProjectMeta(id="div", name="div", workdir=None if tmp_path is None else str(tmp_path)))
    ir.components = [_part("R1", SpiceDevice.R, 10e3, "ohm"), _part("R2", SpiceDevice.R, 10e3, "ohm")]
    ir.nets = [_net("IN", ("R1", "1")), _net("OUT", ("R1", "2"), ("R2", "1")), _net("GND", ("R2", "2"), kind=NetKind.GROUND)]
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="IN", reference_net="GND", kind=StimulusKind.SINE,
                          params={"vo": u(0.0), "va": u(1.0), "freq": u(1e3), "ac": u(1.0)}, provenance=USER)],
        analyses=analyses or [tran(1e-6, 3e-3), AnalysisSpec(id="ac", kind=SpiceAnalysis.AC, params={
            "variation": u("lin"), "points": u(3), "fstart": u(500.0), "fstop": u(1500.0)}, provenance=USER)],
        expectations=list(expectations),
    )
    return ir


def test_level_and_window_expectations_refuse_what_they_cannot_mean():
    def refused(e: Expectation, match: str) -> None:
        with pytest.raises(CompileError, match=match):
            build(_divider(None, e))

    ok = exp("x", Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, reference="v(IN)", t_start=1e-3, t_stop=3e-3, f_max=1e3)
    build(_divider(None, ok))
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, ref=1.0), "reduce=db_at needs 'at'")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", tol_abs=0.1, at=1e3, ref=1.0), r"reduce=db_at reads a magnitude at one frequency, which only an ac analysis produces \(tran is tran\)")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", vector="vp(OUT)", tol_abs=0.1, at=1e3, ref=1.0), "reduce=db_at needs a magnitude vector")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3), r"needs exactly one of reference_vector or params\['ref'\]")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3, ref=1.0, reference="v(IN)"), r"needs exactly one of reference_vector or params\['ref'\]")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3, ref=0.0), r"param ref must be > 0")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3, reference="vp(IN)"), "names a part of a complex vector; a reference level is a magnitude")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3, reference="v(NOPE)"), "unknown net 'NOPE'")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, at=1e3, reference="i(VIN)"), "not the same kind of quantity")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_rel=0.1, at=1e3, ref=1.0), "a level in dB needs tol_abs")
    refused(exp("x", Reduce.DB_AT, -3.0, "dB", analysis="ac", tol_abs=0.1, tol_rel=0.1, at=1e3, ref=1.0), "a relative tolerance on a logarithm is no tolerance")
    refused(exp("x", Reduce.DB_AT, -3.0, "V", analysis="ac", tol_abs=0.1, at=1e3, ref=1.0), r"nominal unit 'V' is not the unit of reduce=db_at \(dB\)")
    refused(exp("x", Reduce.RMS, 0.35, "V", analysis="ac", tol_abs=0.01, t_start=0.0, t_stop=1e-3, f_max=1e3), "reduce=rms reads a time window, which only a tran analysis produces")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3), r"reduce=rms needs params \['t_start', 't_stop', 'f_max'\], missing \['t_stop', 'f_max'\]")
    # f_max (the highest frequency the RMS must include) is required: without it no grid guard is possible
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3), r"missing \['f_max'\]")
    refused(exp("x", Reduce.DB_RMS, -6.0, "dB", tol_abs=0.1, reference="v(IN)", t_start=1e-3, t_stop=3e-3), r"missing \['f_max'\]")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f_max=0.0), r"param f_max must be > 0")
    bad_fmax = exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f_max=1e3)
    bad_fmax.params["f_max"] = u(1.0, "s")
    refused(bad_fmax, "param f_max carries unit 's', reduce=rms takes f_max in Hz")
    # one period of f0 cannot show that the waveform repeats at f0
    refused(exp("x", Reduce.HARMONIC_DBC, -40.0, "dBc", tol_abs=1.0, f0=1e3, k=3, t_start=1e-3, t_stop=2e-3), "at least 2 whole periods of f0")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=4e-3, f_max=1e3), r"must lie inside the tran window \[0, 0.003\] s")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=2e-3, t_stop=1e-3, f_max=1e3), "with 0 <= t_start < t_stop")
    refused(exp("x", Reduce.RMS, 0.35, "A", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f_max=1e3), r"nominal unit 'A' is not the unit of reduce=rms \(V\)")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, reference="v(IN)", t_start=1e-3, t_stop=2e-3, f_max=1e3), "reduce=rms takes no reference_vector")
    refused(exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f0=1e3), r"unexpected params \['f0'\] for reduce=rms")
    refused(exp("x", Reduce.MAX, 0.5, "V", tol_abs=0.01, t_start=1e-3), r"reduce=max takes no params, got \['t_start'\]")
    refused(exp("x", Reduce.MAX, 0.5, "V", tol_abs=0.01, reference="v(IN)"), "reduce=max takes no reference_vector")
    refused(exp("x", Reduce.HARMONIC_DBC, -40.0, "dBc", tol_abs=1.0, f0=1e3, k=1, t_start=1e-3, t_stop=3e-3), r"k must be an integer >= 2 \(the harmonic's number")
    refused(exp("x", Reduce.HARMONIC_DBC, -40.0, "dBc", tol_abs=1.0, f0=1e3, k=2.5, t_start=1e-3, t_stop=3e-3), "k must be an integer >= 2")
    refused(exp("x", Reduce.HARMONIC_DBC, -40.0, "dBc", tol_abs=1.0, f0=1e3, k=3, t_start=1e-3, t_stop=2.5e-3), r"the window must hold whole periods of f0 = 1000 Hz: \(t_stop - t_start\) f0 = 1.5")
    refused(exp("x", Reduce.HARMONIC_DBC, -40.0, "dB", tol_abs=1.0, f0=1e3, k=3, t_start=1e-3, t_stop=3e-3), r"is not the unit of reduce=harmonic_dbc \(dBc\)")
    refused(exp("x", Reduce.AM_DEPTH, 50.0, "percent", tol_abs=1.0, f_carrier=10e3, f_mod=1e3, t_start=1e-3, t_stop=3e-3), "carrier too slow for the envelope: f_carrier / f_mod = 10 < 20")
    refused(exp("x", Reduce.AM_DEPTH, 50.0, "percent", tol_abs=1.0, f_carrier=100e3, f_mod=1e3, t_start=1e-3, t_stop=1.5e-3), "window shorter than one modulation period")
    refused(exp("x", Reduce.AM_DEPTH, 50.0, "percent", tol_abs=1.0, f_carrier=1e3, f_mod=100e3, t_start=1e-3, t_stop=3e-3), "needs f_carrier > f_mod > 0")
    refused(exp("x", Reduce.AM_DEPTH, 0.5, None, tol_abs=0.01, f_carrier=100e3, f_mod=1e3, t_start=1e-3, t_stop=3e-3).model_copy(
        update={"nominal": u(0.5, "ratio")}), r"is not the unit of reduce=am_depth \(percent\)")
    # with a tran start, ngspice saves its first point up to one step after it (measured: tran 1u 5m 1m -> 1.00028 ms)
    started = _divider(None, exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=3e-3, f_max=1e3), analyses=[tran(1e-6, 3e-3, 1e-3)])
    with pytest.raises(CompileError, match=r"t_start 0.001 s must be at least one tran step after the analysis' start \(0.001 s \+ 1e-06 s\)"):
        build(started)
    started.simulation.expectations[0].params["t_start"] = u(1e-3 + 1e-6)
    build(started)
    bad_unit = exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f_max=1e3)
    bad_unit.params["t_start"] = u(1e-3, "ms")
    refused(bad_unit, "param t_start carries unit 'ms', reduce=rms takes t_start in s")
    llm = exp("x", Reduce.RMS, 0.35, "V", tol_abs=0.01, t_start=1e-3, t_stop=2e-3, f_max=1e3)
    llm.params["t_stop"] = llm_generated(2e-3, "s")
    refused(llm, "param t_stop has llm_generated provenance")
    # the accepted params never reach the netlist: the text is the same with and without the level expectations
    assert build(_divider(None, ok)) == build(_divider(None))
    assert build_report(_divider(None, ok))["vectors"] == {"x": "out"}
    assert set(REDUCE_PARAMS) == set(Reduce)


def test_the_reviewer_traces_the_level_and_depth_reductions_to_their_requirements():
    """Regression: '%' and 'percent' (one unit to the compiler) were 'not comparable'; a unitless db_at nominal -3 was compared as -3 Hz."""
    from ai_eda.ir import Requirement, RequirementKind
    from ai_eda.review.reviewer import IndependentReviewer

    nv = IndependentReviewer._nominal_vs_requirement

    def req(value, unit=None):
        return Requirement(id="req.r", key="r", text="r", kind=RequirementKind.EXPLICIT, value=u(value, unit))

    for nominal_unit, requirement in (("%", req("50 %")), ("percent", req(50.0, "%")), ("percent", req("50 percent")), (None, req("50 %"))):
        e = exp("d", Reduce.AM_DEPTH, 50.0, nominal_unit, tol_abs=0.5, f_carrier=100e3, f_mod=1e3, t_start=1e-3, t_stop=2e-3)
        assert nv(e, requirement) == (None, True), (nominal_unit, requirement.value)
    e = exp("d", Reduce.AM_DEPTH, 50.0, "%", tol_abs=0.5, f_carrier=100e3, f_mod=1e3, t_start=1e-3, t_stop=2e-3)
    assert nv(e, req("60 %")) == ("nominal 50 percent is not requirement req.r's 60 percent (+/- 0.5 percent)", True)
    # db_at at the cutoff the requirement names: the sweep point is compared, with or without 'dB' on the nominal
    for unit in ("dB", None):
        cut = exp("c", Reduce.DB_AT, -3.0103, unit, analysis="ac", tol_abs=0.01, at=1e3, ref=1.0)
        assert nv(cut, req("1 kHz")) == (None, True), unit
        problem, comparable = nv(cut, req("2 kHz"))
        assert comparable and problem.startswith("sweep point 1000 Hz is not requirement req.r's 2000 Hz"), unit  # formerly 'nominal -3 Hz ...'
    # a db_at traced to a dB requirement keeps the level comparison; a dBc requirement is another reference: not comparable
    level = exp("c", Reduce.DB_AT, -40.0, "dB", analysis="ac", tol_abs=0.5, at=3e3, ref=1.0)
    assert nv(level, req("-40 dB")) == (None, True) and nv(level, req("-45 dB"))[1] is True
    assert nv(level, req("-40 dBc")) == ("nominal unit 'dB' is not the requirement's 'dBc'", False)
    unitless = exp("c", Reduce.DB_AT, -40.0, None, analysis="ac", tol_abs=0.5, at=3e3, ref=1.0)
    assert nv(unitless, req("-40 dBc")) == ("nominal unit 'dB' is not the requirement's 'dBc'", False)
    h = exp("h", Reduce.HARMONIC_DBC, -40.0, "dBc", tol_abs=1.0, f0=1e3, k=3, t_start=1e-3, t_stop=3e-3)
    assert nv(h, req("–40 dBc")) == (None, True) and nv(h, req(-40.0, "DBC")) == (None, True)


# --------------------------------------------------------------------------- real ngspice


@needs_ngspice
def test_am_depth_of_ngspice_s_am_source(tmp_path: Path):
    """``AM(1 2 1k 100k 0)`` has depth 1 / VO = 50 %: read 49.967 % over one modulation period; ratio 20 shows the documented bias."""
    ir = source_ir(tmp_path, StimulusKind.AM, AM_100, tran(1e-7, 3e-3), depth("depth", 100e3))
    res = _run(ir, tmp_path)
    d = res["spice.depth"]
    assert d.status is S.PASS, d.message
    assert d.details["measured"] == pytest.approx(49.969, abs=0.01) and d.details["unit"] == "percent"
    audit = d.details["am_depth"]
    assert audit["cycles"] == 100 and audit["a_max"] == pytest.approx(3.0, abs=2e-3) and audit["a_min"] == pytest.approx(1.0, abs=2e-3)
    assert audit["max_step"] <= audit["step_limit"] == pytest.approx(1 / (16 * 100e3)) and audit["envelope_bias_bound_rel"] == pytest.approx(6.1675e-4, rel=1e-3)
    assert d.details["params"] == {"f_carrier": 100e3, "f_mod": 1e3, "t_start": 1e-3, "t_stop": 2e-3}
    assert "(f_carrier=100000 Hz, f_mod=1000 Hz) over [0.001, 0.002] s" in d.message
    slow_dir = tmp_path / "ratio20"
    slow_dir.mkdir()
    slow = source_ir(slow_dir, StimulusKind.AM, AM_20, tran(5e-7, 3e-3), depth("depth", 20e3))
    r20 = _run(slow, slow_dir)["spice.depth"]
    # the envelope bias at ratio 20 (up to 1.54 % of VA on each amplitude) reads the depth about 0.8 points low; the verdict is
    # judged on the bias bracket, which reaches 50 %: UNRESOLVED, never a FAIL of a correct 50 % wave (formerly FAIL, repair=human)
    assert r20.status is S.UNRESOLVED and 48.9 < r20.details["measured"] < 49.6, r20.message
    assert "repair" not in r20.details and "given the measurement's bias bounds" in r20.message and "use a faster carrier" in r20.message
    bracket = r20.details["bias_bracket"]
    assert bracket["low"] <= r20.details["measured"] and bracket["high"] >= 50.0
    assert r20.details["am_depth"]["envelope_bias_bound_rel"] == pytest.approx(0.015356, rel=1e-3)
    # ratio 100 is judged on its (narrow) bracket too
    assert d.details["bias_bracket"]["low"] > 49.8 and d.details["bias_bracket"]["high"] < 50.2


@needs_ngspice
def test_a_biased_am_depth_never_passes_a_depth_outside_the_band(tmp_path: Path):
    """Regression: AM(1 1.9724 1k 20k 0) has the true depth 1 / 1.9724 = 50.70 %, outside 50 +/- 0.2; it read 49.93 % and PASSed."""
    ir = source_ir(tmp_path, StimulusKind.AM, {**AM_20, "vo": 1.9724}, tran(5e-7, 3e-3), depth("depth", 20e3))
    r = _run(ir, tmp_path)["spice.depth"]
    assert 49.5 < r.details["measured"] < 50.2 and r.status is not S.PASS, r.message
    assert r.status is S.UNRESOLVED and r.details["bias_bracket"]["high"] >= 100 / 1.9724


def _two_tone(tmp_path: Path, a3: float, step: float, *expectations: Expectation) -> CircuitIR:
    """SINE(0 1 1k) in series with SINE(0 a3 3k) across a 1 k load: the third harmonic is exactly 20 log10(a3) dBc."""
    ir = CircuitIR(project=ProjectMeta(id="tones", name="tones", workdir=str(tmp_path)))
    ir.components = [_part("R1", SpiceDevice.R, 1000.0, "ohm"), _part("R2", SpiceDevice.R, 1e6, "ohm")]
    ir.nets = [_net("OUT", ("R1", "1")), _net("MID", ("R2", "1")), _net("GND", ("R1", "2"), ("R2", "2"), kind=NetKind.GROUND)]
    ir.simulation = SimulationSetup(
        stimuli=[
            Stimulus(id="V1", source="voltage", net="MID", reference_net="GND", kind=StimulusKind.SINE, params={"vo": u(0.0), "va": u(1.0), "freq": u(1e3)}, provenance=USER),
            Stimulus(id="V3", source="voltage", net="OUT", reference_net="MID", kind=StimulusKind.SINE, params={"vo": u(0.0), "va": u(a3), "freq": u(3e3)}, provenance=USER),
        ],
        analyses=[tran(step, 5e-3)], expectations=list(expectations),
    )
    return ir


@needs_ngspice
def test_a_grid_attenuated_harmonic_never_passes_a_level_outside_the_band(tmp_path: Path):
    """Regression: at a 16 us step (inside the 16.67 us guard) a true -19.95 dBc third harmonic read -20.0086 and PASSed -20 +/- 0.01."""
    h = exp("h", Reduce.HARMONIC_DBC, -20.0, "dBc", tol_abs=0.01, f0=1e3, k=3, t_start=1e-3, t_stop=5e-3)
    for true_dbc in (-20.0, -19.95):
        work = tmp_path / f"t{abs(true_dbc)}"
        work.mkdir()
        r = _run(_two_tone(work, 10 ** (true_dbc / 20), 16e-6, h), work)["spice.h"]
        assert r.status is S.UNRESOLVED, r.message  # formerly FAIL for the true -20.00 dBc and PASS for -19.95
        b = r.details["bias_bracket"]
        assert b["low"] <= r.details["measured"] <= b["high"] and b["low"] <= true_dbc <= b["high"]
        assert r.details["harmonic_dbc"]["grid_attenuation_k_db"] > 0.05


@needs_ngspice
def test_a_detuned_fundamental_is_named_not_measured(tmp_path: Path):
    """Regression: a square wave at 1025.17 Hz (the astable's measured frequency for 1 kHz) read -10.66 dBc at f0 = 1 kHz, a plain FAIL."""
    h3 = exp("h3", Reduce.HARMONIC_DBC, 20 * math.log10(1 / 3), "dBc", tol_abs=0.5, f0=1e3, k=3, t_start=1e-3, t_stop=5e-3)
    per = 1 / 1025.17
    pulse = {"v1": 0.0, "v2": 1.0, "td": 0.0, "tr": 1e-9, "tf": 1e-9, "pw": per / 2, "per": per}
    r = _run(source_ir(tmp_path, StimulusKind.PULSE, pulse, tran(1e-6, 5e-3), h3), tmp_path)["spice.h3"]
    assert r.status is S.FAIL and "measured" not in r.details and r.details["repair"] == "human"
    assert "does not repeat at f0 = 1000 Hz" in r.message and "its rising edges give 1025.1" in r.message and "reduce=frequency" in r.message
    assert r.details["harmonic_dbc"]["f_measured"] == pytest.approx(1025.17, rel=1e-4)


@needs_ngspice
def test_an_rms_on_a_grid_too_coarse_for_f_max_gives_no_number(tmp_path: Path):
    """Regression: SINE(0 1 1k) on `tran 1m 50m` has a 1 ms grid; its RMS read 0.976 V (true 0.7071 V) and PASSed 0.97 +/- 0.05."""
    rms = exp("rms", Reduce.RMS, 0.97, "V", tol_abs=0.05, t_start=10e-3, t_stop=50e-3, f_max=1e3)
    r = _run(source_ir(tmp_path, StimulusKind.SINE, {"vo": 0.0, "va": 1.0, "freq": 1e3}, tran(1e-3, 50e-3), rms), tmp_path)["spice.rms"]
    assert r.status is S.FAIL and r.message.startswith("time grid too coarse for an RMS up to f_max = 1000 Hz") and r.details["repair"] == "human"
    assert r.details["rms"]["step_limit"] == pytest.approx(5e-5) and r.details["rms"]["max_step"] > 5e-5


@needs_ngspice
def test_sffm_with_mdi_0_is_its_carrier(tmp_path: Path):
    f = exp("f", Reduce.FREQUENCY, 1000.0, "Hz", tol_rel=1e-3)
    ir = source_ir(tmp_path, StimulusKind.SFFM, {"vo": 0.0, "va": 1.0, "fc": 1e3, "mdi": 0.0, "fs": 10e3}, tran(1e-6, 5e-3), f)
    r = _run(ir, tmp_path)["spice.f"]
    assert r.status is S.PASS and r.details["measured"] == pytest.approx(1000.0, rel=1e-4), r.message


@needs_ngspice
def test_harmonics_of_a_square_wave(tmp_path: Path):
    """The third harmonic of a 50 % square wave is 20 log10(1/3) = -9.5424 dBc; the second vanishes below the engine's floor and is never PASS."""
    h3 = exp("h3", Reduce.HARMONIC_DBC, 20 * math.log10(1 / 3), "dBc", tol_abs=0.01, f0=1e3, k=3, t_start=1e-3, t_stop=5e-3)
    h2 = exp("h2", Reduce.HARMONIC_DBC, -100.0, "dBc", tol_abs=60.0, f0=1e3, k=2, t_start=1e-3, t_stop=5e-3)
    pulse = {"v1": 0.0, "v2": 1.0, "td": 0.0, "tr": 1e-9, "tf": 1e-9, "pw": 0.5e-3, "per": 1e-3}
    res = _run(source_ir(tmp_path, StimulusKind.PULSE, pulse, tran(1e-6, 5e-3), h3, h2), tmp_path)
    r3, r2 = res["spice.h3"], res["spice.h2"]
    assert r3.status is S.PASS and r3.details["measured"] == pytest.approx(-9.5424, abs=1e-3), r3.message
    audit = r3.details["harmonic_dbc"]
    assert audit["a1"] == pytest.approx(2 / math.pi, rel=1e-3) and audit["periods"] == pytest.approx(4.0) and audit["max_step"] <= audit["step_limit"]
    assert r2.details["measured"] < -60.0 and r2.details["harmonic_dbc"]["ak_below_floor"]
    assert r2.status is S.NOT_VERIFIED and "not evidence: harmonic 2" in r2.message and "is at or below the engine's resolution" in r2.details["unresolved"]
    assert res["spice"].status is S.NOT_VERIFIED


@needs_ngspice
def test_rms_of_a_sine_and_levels_in_db(tmp_path: Path):
    rms = exp("rms", Reduce.RMS, 1 / math.sqrt(2), "V", vector="v(IN)", tol_abs=1e-4, t_start=1e-3, t_stop=3e-3, f_max=1e3)
    div = exp("div", Reduce.DB_RMS, 20 * math.log10(0.5), "dB", tol_abs=0.01, reference="v(IN)", t_start=1e-3, t_stop=3e-3, f_max=1e3)
    res = _run(_divider(tmp_path, rms, div), tmp_path)
    assert res["spice.rms"].status is S.PASS and res["spice.rms"].details["measured"] == pytest.approx(0.707104, abs=5e-6), res["spice.rms"].message
    audit = res["spice.rms"].details["rms"]
    assert audit["f_max"] == 1e3 and audit["max_step"] <= audit["step_limit"] == pytest.approx(5e-5) and 0.0 < audit["bias_bound_rel"] < 1e-4
    d = res["spice.div"]
    assert d.status is S.PASS and d.details["measured"] == pytest.approx(-6.0206, abs=1e-4), d.message
    assert d.details["spice_reference_vector"] == "in" and d.details["db_rms"]["reference_rms"] == pytest.approx(0.707104, abs=5e-6)


@needs_ngspice
def test_rc_lowpass_is_minus_3_db_at_its_corner(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="rc", name="rc", workdir=str(tmp_path)))
    c = 1.0 / (2 * math.pi * 1000.0 * 1000.0)  # f_c = 1 / (2 pi R C) = 1 kHz with R = 1 k
    ir.components = [_part("R1", SpiceDevice.R, 1000.0, "ohm"), _part("C1", SpiceDevice.C, c, "F")]
    ir.nets = [_net("IN", ("R1", "1")), _net("OUT", ("R1", "2"), ("C1", "1")), _net("GND", ("C1", "2"), kind=NetKind.GROUND)]
    at_fc = {"analysis": "ac", "tol_abs": 0.01, "at": 1000.0}
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="IN", reference_net="GND", kind=StimulusKind.DC, value=u(0.0, "V"), params={"ac": u(1.0)}, provenance=USER)],
        analyses=[AnalysisSpec(id="ac", kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(3), "fstart": u(500.0), "fstop": u(1500.0)}, provenance=USER)],
        expectations=[exp("ref1", Reduce.DB_AT, -3.0103, "dB", ref=1.0, **at_fc), exp("refvec", Reduce.DB_AT, -3.0103, "dB", reference="v(IN)", **at_fc)],
    )
    res = _run(ir, tmp_path)
    for key in ("spice.ref1", "spice.refvec"):
        r = res[key]
        assert r.status is S.PASS and r.details["measured"] == pytest.approx(10 * math.log10(0.5), abs=1e-6), r.message
        assert r.details["bracket"]["exact"] and r.details["db_at"]["at"] == 1000.0
    assert res["spice.refvec"].details["db_at"]["reference_magnitude"] == pytest.approx(1.0) and "re v(IN)" in res["spice.refvec"].message
