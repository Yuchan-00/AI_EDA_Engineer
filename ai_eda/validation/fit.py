"""``component.fit``: electrical stress of every component at the simulated operating point, judged against its datasheet ratings.

Of the eight selection criteria of the spec (:data:`FIT_CRITERIA`) exactly one
is evaluated here - *electrical stress* - and only as a nominal operating-point
check; the result says so in every message (:data:`EVALUATED` /
:data:`NOT_EVALUATED`). Invariants:

* **The operating point is read, never computed.** The op comes from the
  latest SPICE run through :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`
  (current netlist, unchanged ``results.json``, a succeeded op that the
  environment did not prevent); without one the whole result is NOT_VERIFIED
  with the broken link named. A node voltage is ngspice's last op sample of
  the net's vector; the ground net is 0 V.
* **Only what the op measures is judged.** The *voltage* criterion compares
  ``|v(n1) - v(n2)|`` with ``v_max`` for two-terminal ``R`` and ``C``
  bindings only (:data:`STRESS_DEVICES`, two pins in ``pin_order``): an
  inductor's op voltage is 0 by definition, a diode's ``v_max`` is a reverse
  rating, a source has no rating, and a multi-terminal part's ``v_max`` is a
  pin-to-pin rating (V_DS, V_CE ...) that node-to-ground voltages do not
  bound - all of those are NOT_VERIFIED with the voltages recorded, never a
  proxy PASS or FAIL. The *power* criterion computes ``P = V^2 / R`` with the
  registered calculator ``calc.power.P_VR`` for a **plain** ``R`` binding
  only (a value, no model, no params such as ``m`` / ``tc1`` - with those
  ngspice simulates a resistance that is not ``electrical["resistance"]``)
  from the authoritative resistance the compiler already reconciled with
  the netlist; ngspice records no element currents in the op, so every other
  device is NOT_VERIFIED ("dissipation not computed").
* **A rating is a datasheet fact.** ``v_max`` (V) and ``power_rating`` (W)
  are read from ``Component.electrical`` and must carry ``authoritative`` or
  ``user_requirement`` provenance and the expected unit; an assumption or a
  model's number is NOT_VERIFIED naming the kind. No derating is applied:
  the comparison is at 100 % of the rating and ``details.derating`` says so.
* **PASS only on evidence.** A criterion within its limit is PASS only when
  the netlist rests on no assumption and the run's ``spice`` summary is not
  FAIL (an analysis of that run failed, or an expectation did: the op is
  then not evidence about a correct design); otherwise NOT_VERIFIED with
  the numbers kept. Exceeding a limit is FAIL with ``repair: human`` - a
  tool never changes a part. The result is one ``ValidationResult`` for the
  whole design (``details.components[ref][criterion]``), its status the
  worst over every component and criterion, stamped with the netlist hash
  (``artifact_hash``) and the run's rawfile / ``results.json`` as evidence.
  The derived dissipation lives only in the details and is never written
  into the IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_eda.ir import CircuitIR, Component, Evidence, SpiceDevice, Traced, ValidationResult, ValidationStatus, derived, worst_status
from ai_eda.ir.simulation import TWO_TERMINAL_DEVICES
from ai_eda.tools.calc.basic import CALC_VERSION, power_from_voltage_resistance
from ai_eda.tools.calc.recompute import unit_key
from ai_eda.tools.spice.evidence import FreshSpiceRun, fresh_spice_run, pin_net
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry

FIT_CHECK = "component.fit"
TOOL = FIT_CHECK
#: bumped when the judgement rules change
FIT_VERSION = "0.1"
#: the selection criteria of the spec, in the spec's order
FIT_CRITERIA: tuple[str, ...] = ("electrical stress", "safety", "regulatory", "environment", "reliability", "manufacturability", "sourcing", "cost")
EVALUATED: tuple[str, ...] = ("electrical stress",)
NOT_EVALUATED: tuple[str, ...] = tuple(c for c in FIT_CRITERIA if c not in EVALUATED)
#: devices whose ``v_max`` is the voltage across the two nodes the op measures
STRESS_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.R, SpiceDevice.C})
DERATING = "no derating applied: compared at 100 % of the rating"
GROUND_FACTS = "ground it with datasheet_facts_file / confirm_facts"
NO_ELEMENT_CURRENTS = "ngspice records no element currents in the op"
#: why a two-terminal device other than R / C is not judged on ``v_max``
_NOT_JUDGED: dict[SpiceDevice, str] = {
    SpiceDevice.L: "op voltage across an inductor is 0 by definition; transient peaks are not judged",
    SpiceDevice.D: "a diode's v_max is a reverse rating; the op bias is forward/unsigned and is not judged against it",
    SpiceDevice.V: "a voltage source has no voltage rating to judge",
    SpiceDevice.I: "a current source has no voltage rating to judge",
}
MULTI_TERMINAL = "v_max of a multi-terminal part is a pin-to-pin rating (V_DS, V_CE ...); node-to-ground voltages are recorded, not judged"

S = ValidationStatus


def conditions_note(temperature_c: float) -> str:
    return f"nominal values, operating point only, one temperature ({temperature_c:g} degC)"


def fmt_v(v: float) -> str:
    return f"{v:.6g} V"


def fmt_w(p: float) -> str:
    return f"{p * 1e3:.6g} mW" if abs(p) < 1.0 else f"{p:.6g} W"


def fmt_c(t: float) -> str:
    return f"{t:.6g} degC"


@dataclass
class ComponentStress:
    """What the op says about one component: its pin voltages, the voltage across it, its dissipation - or why not."""

    ref: str
    device: str | None
    #: why the part is not in the netlist at all (no binding / excluded); everything else is then unavailable
    excluded: str | None = None
    #: pin number -> op node voltage (V; the ground net is 0.0) or the reason it is unavailable
    pins: dict[str, float | str] = field(default_factory=dict)
    #: pin number -> net name (``None`` when the pin is in no net)
    nets: dict[str, str | None] = field(default_factory=dict)
    two_terminal: bool = False
    v_across: float | None = None
    v_reason: str | None = None
    #: an ``R`` bound with a value and neither model nor params: ngspice simulated exactly ``electrical["resistance"]``
    plain_r: bool = False
    dissipation: Traced[float] | None = None
    p_reason: str | None = None

    def voltages(self) -> dict[str, Any]:
        return {"pins": dict(self.pins), "nets": dict(self.nets)}


def traced_json(t: Traced) -> dict[str, Any]:
    """A derived value as it is shown in the details: value, unit, the calculator and the ids that filled its roles."""
    p = t.provenance
    return {"value": t.value, "unit": t.unit, "tool": p.tool, "tool_version": p.tool_version, "inputs": dict(p.inputs), "note": p.note}


def rating(c: Component, key: str, unit: str) -> tuple[Traced | None, str | None]:
    """``(traced, None)`` when ``electrical[key]`` is a usable rating (authoritative / user, a number, unit ``unit``), else ``(None, why)``."""
    t = c.electrical.get(key)
    if t is None:
        return None, f"no {key} rating ({GROUND_FACTS})"
    if not t.provenance.is_authoritative:
        return None, f"{key} is {t.provenance.kind.value} (needs confirmation)"
    if isinstance(t.value, bool) or not isinstance(t.value, (int, float)):
        return None, f"{key} is not a number ({t.value!r})"
    if unit_key(t.unit) != unit_key(unit):
        return None, f"{key} carries unit {t.unit!r}, expected {unit}"
    return t, None


def component_stress(ir: CircuitIR, run: FreshSpiceRun, c: Component) -> ComponentStress:
    """Pin voltages, voltage across and dissipation of ``c`` at the run's op (module docstring for what is computed when)."""
    b = c.spice
    if b is None:
        return ComponentStress(ref=c.ref, device=None, excluded="no SPICE binding: no operating point")
    device = b.device.value if b.device is not None else None
    if b.exclude:
        return ComponentStress(ref=c.ref, device=device, excluded=f"excluded from SPICE ({b.exclude_reason or 'no reason recorded'}): no operating point")
    st = ComponentStress(ref=c.ref, device=device)
    picked = run.op()
    aid = picked[0] if not isinstance(picked, str) else None
    for pin in b.pin_order:
        net = pin_net(ir, c.ref, pin)
        st.nets[pin] = net.name if net is not None else None
        if net is None:
            st.pins[pin] = f"pin {c.ref}.{pin} is in no net"
            continue
        v = run.node_voltage(ir, net.name)
        st.pins[pin] = v if isinstance(v, float) else f"pin {c.ref}.{pin} net {net.name}: {v}"
    st.two_terminal = b.device in TWO_TERMINAL_DEVICES and len(b.pin_order) == 2
    if not st.two_terminal:
        st.v_reason = MULTI_TERMINAL
    else:
        v1, v2 = (st.pins[p] for p in b.pin_order)
        if isinstance(v1, float) and isinstance(v2, float):
            st.v_across = abs(v1 - v2)
        else:
            st.v_reason = v1 if isinstance(v1, str) else str(v2)
    st.plain_r = b.device is SpiceDevice.R and b.value is not None and b.model_name is None and b.model_card is None and not b.params
    if b.device is not SpiceDevice.R:
        st.p_reason = f"dissipation of a {device} device is not computed: {NO_ELEMENT_CURRENTS}"
        return st
    if not st.plain_r:
        what = []
        if b.params:
            what.append(f"SPICE params {sorted(b.params)}")
        if b.model_name is not None or b.model_card is not None:
            what.append(f"model {b.model_name!r}")
        if b.value is None:
            what.append("no value")
        st.p_reason = f"dissipation not computed: {c.ref} carries {' and '.join(what)}; ngspice simulates a resistance that is not electrical[resistance]"
        return st
    if "resistance" not in c.electrical:
        st.p_reason = f"electrical[resistance] is missing ({GROUND_FACTS}); the SPICE value alone is a binding, not the shipped part"
        return st
    r, why = rating(c, "resistance", "ohm")
    if r is None:
        st.p_reason = why
        return st
    if st.v_across is None:
        st.p_reason = st.v_reason
        return st
    n1, n2 = (st.nets[p] for p in b.pin_order)
    v = derived(st.v_across, tool=run.engine, tool_version=run.engine_version, unit="V", note=f"op |v({n1}) - v({n2})| of analysis {aid}")
    st.dissipation = power_from_voltage_resistance(v, r, (f"{c.ref}.v_across", f"{c.ref}.resistance"))
    return st


def gate(criterion: dict[str, Any], run: FreshSpiceRun) -> dict[str, Any]:
    """A would-be PASS is NOT_VERIFIED when the netlist rests on assumptions or the run's ``spice`` summary is FAIL; a FAIL stays FAIL."""
    if criterion["status"] != S.PASS.value:
        return criterion
    if run.assumptions:
        return {**criterion, "status": S.NOT_VERIFIED.value, "reason": f"within limit ({criterion['reason']}) but the netlist rests on assumption(s) {run.assumptions} that nobody confirmed"}
    if run.spice.status is S.FAIL:
        return {**criterion, "status": S.NOT_VERIFIED.value, "reason": f"within limit ({criterion['reason']}) but computed on an op of a run whose spice summary is FAIL"}
    return criterion


def judge_voltage(c: Component, st: ComponentStress, analysis: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"analysis": analysis, **st.voltages()}
    if st.excluded:
        return {"status": S.NOT_VERIFIED.value, "reason": st.excluded}
    if not st.two_terminal:
        return {"status": S.NOT_VERIFIED.value, "reason": MULTI_TERMINAL, **facts}
    device = c.spice.device if c.spice is not None else None
    if device not in STRESS_DEVICES:
        return {"status": S.NOT_VERIFIED.value, "reason": _NOT_JUDGED.get(device, f"a {device} device is not judged on v_max"), "v_across": st.v_across, **facts}
    v_max, why = rating(c, "v_max", "V")
    if v_max is None:
        return {"status": S.NOT_VERIFIED.value, "reason": str(why), "v_across": st.v_across, **facts}
    if st.v_across is None:
        return {"status": S.NOT_VERIFIED.value, "reason": str(st.v_reason), "v_max": v_max.value, **facts}
    limit = float(v_max.value)
    facts.update({"v": st.v_across, "v_max": limit, "factor": 1.0, "limit": limit, "derating": DERATING})
    if st.v_across > limit:
        return {"status": S.FAIL.value, "reason": f"{fmt_v(st.v_across)} > {fmt_v(limit)} (v_max)", "repair": "human", **facts}
    return {"status": S.PASS.value, "reason": f"{fmt_v(st.v_across)} <= {fmt_v(limit)} (v_max)", **facts}


def judge_power(c: Component, st: ComponentStress, analysis: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"analysis": analysis, "derating": DERATING}
    if st.excluded:
        return {"status": S.NOT_VERIFIED.value, "reason": st.excluded}
    if st.dissipation is None:
        return {"status": S.NOT_VERIFIED.value, "reason": str(st.p_reason), **facts}
    facts["dissipation"] = traced_json(st.dissipation)
    p = float(st.dissipation.value)
    p_max, why = rating(c, "power_rating", "W")
    if p_max is None:
        return {"status": S.NOT_VERIFIED.value, "reason": str(why), **facts}
    limit = float(p_max.value)
    facts.update({"p": p, "power_rating": limit, "factor": 1.0, "limit": limit})
    if p > limit:
        return {"status": S.FAIL.value, "reason": f"{fmt_w(p)} > {fmt_w(limit)} (power_rating)", "repair": "human", **facts}
    return {"status": S.PASS.value, "reason": f"{fmt_w(p)} <= {fmt_w(limit)} (power_rating)", **facts}


def summarise(components: dict[str, dict[str, Any]], criteria: tuple[str, ...]) -> str:
    """``R1 PASS, R2 FAIL (power: 3.6 mW > 1 mW (power_rating)), J1 NOT_VERIFIED (excluded ...)`` - every non-PASS reason named."""
    parts: list[str] = []
    for ref, entry in components.items():
        why = "; ".join(f"{k}: {entry[k]['reason']}" for k in criteria if k in entry and entry[k]["status"] != S.PASS.value)
        parts.append(f"{ref} {entry['status']}" + (f" ({why})" if why else ""))
    return ", ".join(parts)


class ComponentFitValidator(Validator):
    id = FIT_CHECK
    domains = frozenset()
    description = "Electrical stress of every component at the simulated operating point vs its datasheet ratings (v_max, power_rating)"
    consumes = frozenset({SPICE_CHECK_ID})

    def _result(self, ir: CircuitIR, status: ValidationStatus, message: str, run: FreshSpiceRun | None = None, analysis: str | None = None, **details: Any) -> ValidationResult:
        base: dict[str, Any] = {"fit_version": FIT_VERSION, "criteria_evaluated": list(EVALUATED), "criteria_not_evaluated": list(NOT_EVALUATED), "calc_version": CALC_VERSION}
        evidence: list[Evidence] = []
        if run is not None:
            base.update({"engine": run.engine, "engine_version": run.engine_version, "engine_stamp": run.engine_stamp, "conditions": run.conditions,
                         "spice_status": run.spice.status.value, "assumptions": run.assumptions, "derating": DERATING})
            if analysis is not None:
                base["analysis"] = analysis
                evidence = run.evidence(analysis)
        return ValidationResult(
            check_id=self.id, status=status, message=message, tool=TOOL, tool_version=FIT_VERSION,
            artifact_hash=run.netlist.content_hash if run is not None else None, ir_hash=ir.content_hash(), evidence=evidence, details={**base, **details},
        )

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        if not ir.components:
            return [self._result(ir, S.NOT_APPLICABLE, "no components")]
        run = fresh_spice_run(ir, needs="component fit needs an operating point from SPICE")
        if isinstance(run, str):
            return [self._result(ir, S.NOT_VERIFIED, run)]
        picked = run.op()
        if isinstance(picked, str):
            return [self._result(ir, S.NOT_VERIFIED, picked, run, analyses=run.analysis_ids)]
        aid, _ = picked
        components: dict[str, dict[str, Any]] = {}
        for c in ir.components:
            st = component_stress(ir, run, c)
            voltage = gate(judge_voltage(c, st, aid), run)
            power = gate(judge_power(c, st, aid), run)
            status = worst_status([S(voltage["status"]), S(power["status"])])
            components[c.ref] = {"status": status.value, "device": st.device, "voltage": voltage, "power": power}
        overall = worst_status(S(e["status"]) for e in components.values())
        message = (f"{len(components)} component(s) at the operating point ({aid}): {summarise(components, ('voltage', 'power'))}; "
                   f"{conditions_note(run.temperature_c)}; criteria evaluated: {', '.join(EVALUATED)}; not evaluated: {', '.join(NOT_EVALUATED)}")
        if run.spice.status is S.FAIL:
            message = f"stress computed on an op of a run whose spice summary is FAIL; {message}"
        details: dict[str, Any] = {"components": components}
        if overall is S.FAIL:
            details["repair"] = "human"
        return [self._result(ir, overall, message, run, aid, **details)]


default_registry.register(ComponentFitValidator())

__all__ = [
    "DERATING",
    "EVALUATED",
    "FIT_CHECK",
    "FIT_CRITERIA",
    "FIT_VERSION",
    "MULTI_TERMINAL",
    "NOT_EVALUATED",
    "STRESS_DEVICES",
    "TOOL",
    "ComponentFitValidator",
    "ComponentStress",
    "component_stress",
    "conditions_note",
    "gate",
    "judge_power",
    "judge_voltage",
    "rating",
    "summarise",
    "traced_json",
]
