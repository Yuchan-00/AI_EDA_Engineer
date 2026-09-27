"""``spice.si.<net>``: an ngspice transient of every electrically long net over a plane, as a lossless transmission line.

Invariant: this is tool evidence about the *routed* copper, never an opinion.
For each net the critical-length rule finds electrically long
(:func:`ai_eda.tools.si.promote.critical_rows`) and whose copper lies over a
reference plane (so Z0 and t_d are defined), a small deck is compiled by the
SPICE netlist compiler (the same whitelist as the design's netlist) and run
on the engine in ``tools["spice"]``:

    source --R_drive-- A ==T(Z0, t_d)== B --C_load-- 0

* the source is a pulse from 0 to 1 V and back with the class's edge
  ``t_r`` for both edges (the network is linear: overshoot as a fraction of
  the swing does not depend on the swing, so 1 V stands for any logic
  level);
* ``Z0`` is ``calc.tline.microstrip.z0`` at the width that carries most of
  the net's line, ``t_d`` the line's routed delay (tracks + via barrels,
  :attr:`~ai_eda.tools.si.measure.NetMeasure.line`: a 2-pad net's copper,
  or its extracted pad-to-pad path) - its neck-downs lumped, a documented
  simplification;
* ``R_drive`` / ``C_load`` / ``t_r`` / the band are the class's driver model
  (:mod:`ai_eda.tools.si.driver`: a grounded datasheet fact of a driver that
  drives the net, or the confirmed choice).

The deck is one driver, one line and one receiver, so it judges only a net
that *is* that circuit; anything else is NOT_VERIFIED naming what the deck
does not represent, never a verdict about a circuit that is not in the IR:

* a net with more than two pads (a tree: its branches reflect, and the
  other pads load it);
* a pad of a part the deck does not model: a pin that is neither an IC pin
  that can drive or receive (input / output / bidirectional / tri-state /
  open collector / emitter) nor a connector's (a part of a ``Connector*``
  symbol library: ``C_load`` stands for what is plugged in) - a resistor, a
  capacitor, an inductor, a switch, a crystal ... (a 100 nF capacitor on the
  net makes a 1 ns edge impossible);
* a class ``driver`` on the net whose pins there cannot drive it (a
  microcontroller's ``~RESET`` *input*): the edge the deck launches does not
  exist.

The judgement, at the far end ``B``: the overshoot above the high level and
the undershoot below the low level are each at most ``ringing_tol_rel`` of
the swing, and the waveform has settled inside that band over the last 10 %
of each level. The deck and every number are in the result (``details``),
the deck and the rawfile are :class:`~ai_eda.ir.Evidence` with their hashes;
the rawfile carries ngspice's date line, so it is evidence, not a
deterministic artifact. A net without a plane is NOT_VERIFIED "impedance is
undefined without a reference plane"; a missing driver value is NOT_VERIFIED
naming it; no engine is NOT_VERIFIED. Every ``spice.si.<net>`` recorded
earlier for a net that is no longer long gets a superseding NOT_APPLICABLE.
Each net's deck and rawfile live under ``spice_si/`` at a stem unique to the
exact net name (:func:`deck_stem`: the name made safe plus 8 hex digits of
its SHA-256), so ``D+`` and ``D-`` - or ``clk`` and ``CLK`` on a
case-insensitive file system - never overwrite each other's evidence.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from ai_eda.compilers.spice import analysis_command, build
from ai_eda.errors import CompileError, ToolExecutionError, ToolUnavailableError
from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Component,
    Evidence,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    ValidationResult,
    ValidationStatus,
)
from ai_eda.tools.calc.tline import NO_REFERENCE_PLANE
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.si.driver import DRIVING_TYPES, driver_value, pins_on_net
from ai_eda.tools.si.measure import SI_VERSION, measure_nets
from ai_eda.tools.si.promote import critical_rows
from ai_eda.tools.spice.runner import SpiceAnalysis, SpiceResult, SpiceRunner

#: check id prefix (``spice.si.<net>``); the SPICE stage's own retirement of stale ``spice.<id>`` results leaves it alone
CHECK_PREFIX = "spice.si"
#: where the decks and rawfiles go, under the workdir
SI_DIR = "spice_si"
#: the hold of each level: HOLD_EDGES x (t_r + 2 t_d) + HOLD_TAUS x (R_drive + Z0) C_load, so a damped ringing settles inside it
HOLD_EDGES = 20.0
HOLD_TAUS = 10.0
#: time steps per the shorter of t_r and t_d
STEPS_PER_EDGE = 50.0
#: the final fraction of each level over which the waveform must stay inside the band (settling)
SETTLE_FRACTION = 0.1
#: pin types of an IC pin the deck models as the driver or the receiver
MODELLED_PIN_TYPES = frozenset({*DRIVING_TYPES, PinElectricalType.INPUT})
#: a part of a symbol library with this prefix is a connector: its pin is the off-board end, ``C_load`` stands for what is plugged in
CONNECTOR_LIBRARY_PREFIX = "Connector"

_PROV = Provenance(kind=ProvenanceKind.DERIVED, tool="spice.si", tool_version=SI_VERSION, note="spice.si deck element (see ai_eda.tools.spice.si_check)")


def _traced(value: float, unit: str | None, note: str) -> Traced:
    return Traced(value=value, unit=unit, provenance=_PROV.model_copy(update={"note": note}))


def _pins(n: int) -> list[Pin]:
    return [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=_PROV) for i in range(1, n + 1)]


def deck_ir(project_id: str, *, z0: float, td: float, t_r: float, r_drive: float, c_load: float) -> tuple[CircuitIR, float, float, float]:
    """The deck's IR (module docstring) and ``(hold, step, stop)`` of its transient."""
    hold = HOLD_EDGES * (t_r + 2.0 * td) + HOLD_TAUS * (r_drive + z0) * c_load
    step = min(t_r, td) / STEPS_PER_EDGE
    stop = 2.0 * (t_r + hold)
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id))

    def part(ref: str, n: int, binding: SpiceBinding) -> Component:
        return Component(ref=ref, value=ref, pins=_pins(n), provenance=_PROV, spice=binding)

    ir.components = [
        part("RS", 2, SpiceBinding(device=SpiceDevice.R, value=_traced(r_drive, "ohm", "driver source resistance"), provenance=_PROV)),
        part("T1", 4, SpiceBinding(device=SpiceDevice.T, pin_order=["1", "2", "3", "4"], params={
            "z0": _traced(z0, "ohm", "calc.tline.microstrip.z0 of the net's main width"), "td": _traced(td, "s", "the net's routed delay")}, provenance=_PROV)),
        part("CL", 2, SpiceBinding(device=SpiceDevice.C, value=_traced(c_load, "F", "far-end load"), provenance=_PROV)),
    ]

    def net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
        return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=_PROV)

    ir.nets = [net("S", ("RS", "1")), net("A", ("RS", "2"), ("T1", "1")), net("B", ("T1", "3"), ("CL", "1")),
               net("GND", ("T1", "2"), ("T1", "4"), ("CL", "2"), kind=NetKind.GROUND)]
    pulse = {"v1": 0.0, "v2": 1.0, "td": 0.0, "tr": t_r, "tf": t_r, "pw": hold, "per": 2.0 * (t_r + hold)}
    units = {"v1": "V", "v2": "V", "td": "s", "tr": "s", "tf": "s", "pw": "s", "per": "s"}
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VDRV", source="voltage", net="S", reference_net="GND", kind=StimulusKind.PULSE,
                          params={k: _traced(v, units[k], f"pulse {k}") for k, v in pulse.items()}, provenance=_PROV)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": _traced(step, "s", "tran step"), "stop": _traced(stop, "s", "tran stop")}, provenance=_PROV)],
    )
    return ir, hold, step, stop


def judge_waveform(time: list[float], vb: list[float], *, t_r: float, hold: float, tol: float) -> tuple[ValidationStatus, dict[str, Any], str]:
    """Overshoot / undershoot / settling of the far-end waveform against the band ``tol`` (module docstring)."""
    if not time or len(time) != len(vb) or any(not math.isfinite(x) for x in (*time, *vb)):
        return ValidationStatus.FAIL, {}, "the far-end waveform is empty or not finite"
    fall = t_r + hold  # the falling edge starts here
    high = [v for t, v in zip(time, vb) if t <= fall]
    low = [v for t, v in zip(time, vb) if t > fall]
    overshoot = max(0.0, max(high) - 1.0)
    undershoot = max(0.0, -min(low)) if low else 0.0
    settle_high = [v for t, v in zip(time, vb) if fall - SETTLE_FRACTION * hold <= t <= fall]
    settle_low = [v for t, v in zip(time, vb) if time[-1] - SETTLE_FRACTION * hold <= t]
    dev_high = max((abs(v - 1.0) for v in settle_high), default=math.inf)
    dev_low = max((abs(v) for v in settle_low), default=math.inf)
    measured = {"overshoot_rel": overshoot, "undershoot_rel": undershoot, "settle_dev_high_rel": dev_high, "settle_dev_low_rel": dev_low,
                "max_v": max(vb), "min_v": min(vb), "final_v": vb[-1], "band_rel": tol}
    bad = []
    if overshoot > tol:
        bad.append(f"overshoot {overshoot:.1%} > {tol:.0%}")
    if undershoot > tol:
        bad.append(f"undershoot {undershoot:.1%} > {tol:.0%}")
    if dev_high > tol or dev_low > tol:
        bad.append(f"not settled inside +/- {tol:.0%} over the last {SETTLE_FRACTION:.0%} of a level (high {dev_high:.1%}, low {dev_low:.1%})")
    text = f"overshoot {overshoot:.1%}, undershoot {undershoot:.1%}, settled to {max(dev_high, dev_low):.2%} (band {tol:.0%} of the swing)"
    return (ValidationStatus.FAIL if bad else ValidationStatus.PASS), measured, ("; ".join(bad) + f" ({text})" if bad else text)


def _safe(net: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", net) or "net"


def deck_stem(net: str) -> str:
    """The file stem of ``net``'s deck and run directory: the name made safe plus 8 hex digits of its SHA-256 (unique per exact name, case included)."""
    return f"{_safe(net)}-{hashlib.sha256(net.encode('utf-8')).hexdigest()[:8]}"


def topology_problem(ir: CircuitIR, net: str, driver: str | None) -> str | None:
    """Why the one-line deck does not represent ``net`` (module docstring), or ``None`` when it does."""
    n = ir.net(net)
    if n is None:
        return f"{net} is not an IR net"
    pads = sorted({(p.component_ref, p.pin_number) for p in n.pins})
    if len(pads) != 2:
        labels = ", ".join(f"{r}.{p}" for r, p in pads)
        return f"{net} joins {len(pads)} pads ({labels}): the deck is one driver, one line and one receiver, not a tree with its branches and loads"
    unmodelled: list[str] = []
    for ref, number in pads:
        comp = ir.component(ref)
        pin = comp.pin(number) if comp is not None else None
        if comp is None or pin is None:
            unmodelled.append(f"{ref}.{number} (no pin record)")
            continue
        if comp.symbol is not None and comp.symbol.library.startswith(CONNECTOR_LIBRARY_PREFIX):
            continue
        if pin.electrical_type in MODELLED_PIN_TYPES:
            continue
        what = f"{comp.symbol.library}:{comp.symbol.name}" if comp.symbol is not None else "no symbol"
        unmodelled.append(f"{ref}.{number} {comp.value} ({what}, {pin.electrical_type.value} pin)")
    if unmodelled:
        return f"{net} carries a part the deck does not model: {'; '.join(unmodelled)}"
    if driver is not None:
        mine = pins_on_net(ir, driver, net)
        if mine and not any(p.electrical_type in DRIVING_TYPES for p in mine):
            kinds = ", ".join(f"pin {p.number} {p.name} is {p.electrical_type.value}" for p in mine)
            return f"the class driver {driver} does not drive {net} ({kinds}): the edge the deck would launch does not exist"
    return None


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def spice_si_results(ir: CircuitIR, tools: dict[str, Any], workdir: Path | str) -> list[ValidationResult]:
    """Every ``spice.si.<net>`` result (module docstring), plus superseding NOT_APPLICABLE ones for nets no longer long; ``[]`` without ``ir.si``.

    ``tools["kicad_library"]`` (when present) gives the pad boxes the
    pad-to-pad paths are extracted with.
    """
    si = ir.si
    if si is None:
        return []
    library = tools.get("kicad_library")
    measures = measure_nets(ir, library=library if isinstance(library, KicadLibrary) else None)
    rows = [r for r in critical_rows(ir, measures) if r.status == "long"]
    runner = tools.get("spice")
    ir_hash = ir.content_hash()
    out: list[ValidationResult] = []
    for r in rows:
        check = f"{CHECK_PREFIX}.{r.net}"
        cls = si.class_of(r.net)
        m = measures[r.net]
        base: dict[str, Any] = {"net": r.net, "class": None if cls is None else cls.name, "length_mm": round(m.length_mm, 6), "delay_s": m.delay_s,
                                "l_crit_mm": r.l_crit_mm, "kind": "spice (lossless T line)"}

        def nv(message: str, **extra: Any) -> ValidationResult:
            return ValidationResult(check_id=check, status=ValidationStatus.NOT_VERIFIED, message=message, ir_hash=ir_hash, details={**base, **extra})

        line = m.line
        assert line is not None  # a long row has a line
        base.update(line_mm=round(line.length_mm, 6), line=line.describe())
        if line.bound or line.delay_s is None:
            out.append(nv(f"{r.net} is electrically long but {NO_REFERENCE_PLANE} - use pcb_layers=4 or add a plane: Z0 and t_d are not defined, nothing to simulate"))
            continue
        main = next((s for s in m.segments if line.main is not None and (s.layer, s.width_mm) == line.main), None)
        if main is None or main.model is None or main.model.z0_ohm is None:
            out.append(nv(f"{r.net}: no segment with a defined Z0"))
            continue
        z0 = float(main.model.z0_ohm)
        td = float(line.delay_s)
        base.update(z0_ohm=z0, main_width_mm=main.width_mm, main_layer=main.layer)
        assert cls is not None
        topo = topology_problem(ir, r.net, cls.driver)
        if topo is not None:
            out.append(nv(f"{topo}; not simulated (the lossless-line deck would judge a circuit that is not in the IR)"))
            continue
        values: dict[str, float] = {}
        missing: list[str] = []
        sources: dict[str, str] = {}
        for field in ("t_rise_s", "r_drive_ohm", "c_load_f"):
            v, why = driver_value(ir, cls, field, r.net)
            if v is None:
                missing.append(why)
            else:
                values[field], sources[field] = v.value, v.source
        tol_t = cls.ringing_tol_rel
        if tol_t is None:
            missing.append(f"si.net_classes[{cls.name}].ringing_tol_rel")
        base.update(sources=sources)
        if missing:
            out.append(nv(f"{r.net}: missing " + ", ".join(missing)))
            continue
        if not isinstance(runner, SpiceRunner) or not runner.available():
            out.append(nv(f"{r.net}: no SPICE engine available"))
            continue
        tol = float(tol_t.value)  # type: ignore[union-attr]
        dir_ = Path(workdir) / SI_DIR
        dir_.mkdir(parents=True, exist_ok=True)
        stem = deck_stem(r.net)
        deck, hold, step, stop = deck_ir(f"si_{stem}", z0=z0, td=td, t_r=values["t_rise_s"], r_drive=values["r_drive_ohm"], c_load=values["c_load_f"])
        try:
            text = build(deck)
            command = analysis_command(deck.simulation.analyses[0], deck.simulation)  # type: ignore[union-attr]
        except CompileError as e:
            out.append(ValidationResult(check_id=check, status=ValidationStatus.FAIL, message=f"{r.net}: the SI deck does not compile: {e}", ir_hash=ir_hash,
                                        details={**base, "repair": "human"}))
            continue
        path = dir_ / f"{stem}.cir"
        path.write_text(text, encoding="utf-8", newline="\n")
        deck_hash = _sha(path)
        try:
            res: SpiceResult = runner.run(path, SpiceAnalysis.TRAN, dir_ / stem, command=command)
        except (ToolUnavailableError, ToolExecutionError) as e:
            out.append(nv(f"{r.net}: the SPICE engine could not run the deck: {e}"))
            continue
        engine_version = res.engine_version or runner.version()
        evidence = [Evidence(description=f"spice.si deck of {r.net} (compiled from the IR copper and the class's driver model)", path=str(path), content_hash=deck_hash)]
        if res.raw_output_path:
            evidence.insert(0, Evidence(description=f"ngspice rawfile of {r.net} ({res.command})", path=res.raw_output_path, content_hash=res.raw_output_hash))
        details = {**base, "deck": text, "command": res.command, "t_rise_s": values["t_rise_s"], "r_drive_ohm": values["r_drive_ohm"], "c_load_f": values["c_load_f"],
                   "td_s": td, "hold_s": hold, "step_s": step, "stop_s": stop, "band_rel": tol,
                   "model": f"one lossless line: Z0 of the width with the most copper along it, t_d of {line.describe()} (neck-downs lumped)"}
        stamp = dict(check_id=check, tool=runner.engine, tool_version=engine_version, artifact_hash=deck_hash, ir_hash=ir_hash, evidence=evidence)
        if not res.succeeded:
            status = ValidationStatus.NOT_VERIFIED if res.unverifiable else ValidationStatus.FAIL
            out.append(ValidationResult(status=status, message=f"{r.net}: the transient did not succeed: " + "; ".join(res.errors[:3]),
                                        details={**details, "errors": list(res.errors), **({} if res.unverifiable else {"repair": "human"})}, **stamp))
            continue
        try:
            time, vb = res.scale_values(), res.vector("b")
        except (KeyError, ValueError) as e:
            out.append(ValidationResult(status=ValidationStatus.FAIL, message=f"{r.net}: vector not produced: {e}", details={**details, "repair": "human"}, **stamp))
            continue
        status, measured, text_msg = judge_waveform(time, vb, t_r=values["t_rise_s"], hold=hold, tol=tol)
        details["measured"] = measured
        if status is ValidationStatus.FAIL:
            details["repair"] = "human"
        out.append(ValidationResult(status=status, message=(
            f"{r.net}: {line.describe()} {line.length_mm:.2f} mm as Z0 {z0:.2f} ohm / t_d {td * 1e12:.1f} ps driven through {values['r_drive_ohm']:g} ohm with t_r {values['t_rise_s']:g} s "
            f"into {values['c_load_f']:g} F: {text_msg}"), details=details, **stamp))
    keep = {res.check_id for res in out}
    for check_id, last in ir.validation.latest_by_check().items():
        if check_id.startswith(CHECK_PREFIX + ".") and check_id not in keep and last.status is not ValidationStatus.NOT_APPLICABLE:
            out.append(ValidationResult(check_id=check_id, status=ValidationStatus.NOT_APPLICABLE, message="the net is no longer electrically long by the critical-length rule (superseded)",
                                        ir_hash=ir_hash, details={"superseded": last.status.value}))
    return out


__all__ = ["CHECK_PREFIX", "SI_DIR", "deck_ir", "deck_stem", "judge_waveform", "spice_si_results", "topology_problem"]
