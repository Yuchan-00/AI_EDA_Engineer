"""Domain-specific validators.

:class:`AnalogBiasValidator` and :class:`PowerThermalValidator` read real
SPICE evidence through :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`;
the others still return NOT_VERIFIED until an analysis backend exists.
Keeping the stubs here means the *selection* mechanism is exercised end to
end and the GUI can show "this check exists but has not run".
"""

from __future__ import annotations

from typing import Any

from ai_eda.ir import CircuitDomain, CircuitIR, Evidence, NetKind, ValidationResult, ValidationStatus, worst_status
from ai_eda.tools.calc.basic import CALC_VERSION, junction_temperature
from ai_eda.tools.calc.recompute import unit_key
from ai_eda.tools.spice.evidence import FreshSpiceRun, fresh_spice_run
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.fit import component_stress, fmt_c, gate, rating, summarise, traced_json
from ai_eda.validation.registry import default_registry

THERMAL_TOOL = "domain.power.thermal"
#: bumped when the judgement rules change
THERMAL_VERSION = "0.1"
#: the ``Component.electrical`` keys the thermal check reads and the unit family each must carry
THERMAL_KEYS: dict[str, str] = {"theta_ja": "K/W", "t_j_max": "degC"}
NO_THERMAL_KEYS = "no component carries theta_ja / t_j_max (datasheet facts: theta_ja in K/W, t_j_max in degC)"
NO_THERMAL_KEYS_OUTSIDE_POWER = NO_THERMAL_KEYS + "; the design is not in the POWER domain, so nothing is claimed"

S = ValidationStatus


class PowerThermalValidator(Validator):
    """Junction temperature of every simulated part from its op dissipation: Tj = Ta + P * theta_ja vs t_j_max.

    Per component: the dissipation is the one ``component.fit`` computes
    (:func:`~ai_eda.validation.fit.component_stress`: a plain ``R`` binding
    at the fresh op; anything else is "not computed", ngspice records no
    element currents), ``theta_ja`` (K/W; ``degC/W`` spellings are the same
    unit) and ``t_j_max`` (degC) are authoritative / user datasheet facts in
    ``Component.electrical``, and the ambient ``Ta`` is
    ``ir.simulation.temperature_c`` (authoritative / user) which must be the
    temperature the op actually ran at (``results.json`` conditions) - the
    same op at another ambient is another op. ``Tj`` comes from the
    registered calculator ``calc.thermal.T_j`` and lives only in the
    details. FAIL (``repair: human``) when ``Tj > t_j_max``; PASS only with
    every input authoritative, no netlist assumption and a ``spice``
    summary that is not FAIL; every other case is NOT_VERIFIED naming the
    missing key, kind, unit or ambient. One result for the design, worst
    over its components. The check applies to every design (a part that
    dissipates power is not confined to the POWER domain): a design where
    no component carries a thermal key at all is NOT_VERIFIED naming the
    keys to ground when the topology declares the POWER domain, and
    NOT_APPLICABLE otherwise (nothing claimed, nothing to judge); as soon
    as one component carries a key it is judged whatever the domain.
    """

    id = THERMAL_TOOL
    domains = frozenset()  # every design; the POWER domain only decides how "no thermal data" is reported
    description = "Junction temperature from the op dissipation (Tj = Ta + P * theta_ja) vs the datasheet t_j_max"
    consumes = frozenset({SPICE_CHECK_ID})

    def _result(self, ir: CircuitIR, status: ValidationStatus, message: str, run: FreshSpiceRun | None = None, analysis: str | None = None, **details: Any) -> ValidationResult:
        base: dict[str, Any] = {"thermal_version": THERMAL_VERSION, "calc_version": CALC_VERSION, "keys": dict(THERMAL_KEYS)}
        evidence: list[Evidence] = []
        if run is not None:
            base.update({"engine": run.engine, "engine_version": run.engine_version, "engine_stamp": run.engine_stamp, "conditions": run.conditions,
                         "spice_status": run.spice.status.value, "assumptions": run.assumptions})
            if analysis is not None:
                base["analysis"] = analysis
                evidence = run.evidence(analysis)
        return ValidationResult(
            check_id=self.id, status=status, message=message, tool=THERMAL_TOOL, tool_version=THERMAL_VERSION,
            artifact_hash=run.netlist.content_hash if run is not None else None, ir_hash=ir.content_hash(), evidence=evidence, details={**base, **details},
        )

    @staticmethod
    def _ambient(ir: CircuitIR, run: FreshSpiceRun) -> tuple[Any, str | None]:
        """``(Traced Ta, None)`` when the stated ambient is authoritative and is the op's temperature, else ``(None, why)``."""
        t = ir.simulation.temperature_c if ir.simulation is not None else None
        source = run.conditions.get("temperature_source")
        if t is None:
            return None, f"no ambient: simulation.temperature_c is not set (the op ran at {fmt_c(run.temperature_c)}, {source})"
        if not t.provenance.is_authoritative:
            return None, f"simulation.temperature_c is {t.provenance.kind.value} (needs confirmation)"
        if t.unit is not None and unit_key(t.unit) != unit_key("degC"):
            return None, f"simulation.temperature_c carries unit {t.unit!r}, expected degC"
        if float(t.value) != run.temperature_c:
            return None, f"the op ran at {fmt_c(run.temperature_c)}, not the stated ambient {fmt_c(float(t.value))}"
        return t, None

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        if not ir.components:
            return [self._result(ir, S.NOT_APPLICABLE, "no components")]
        if not any(k in c.electrical for c in ir.components for k in THERMAL_KEYS):
            power = ir.topology is not None and CircuitDomain.POWER in ir.topology.domains
            return [self._result(ir, S.NOT_VERIFIED if power else S.NOT_APPLICABLE, NO_THERMAL_KEYS if power else NO_THERMAL_KEYS_OUTSIDE_POWER)]
        run = fresh_spice_run(ir, needs="thermal analysis needs an operating point from SPICE")
        if isinstance(run, str):
            return [self._result(ir, S.NOT_VERIFIED, run)]
        picked = run.op()
        if isinstance(picked, str):
            return [self._result(ir, S.NOT_VERIFIED, picked, run, analyses=run.analysis_ids)]
        aid, _ = picked
        t_a, ambient_reason = self._ambient(ir, run)
        components: dict[str, dict[str, Any]] = {}
        for c in ir.components:
            entry: dict[str, Any] = {"analysis": aid}
            missing = [k for k in THERMAL_KEYS if k not in c.electrical]
            st = component_stress(ir, run, c)
            if st.excluded:
                thermal = {"status": S.NOT_VERIFIED.value, "reason": st.excluded}
            elif missing:
                thermal = {"status": S.NOT_VERIFIED.value, "reason": f"no {' / '.join(missing)} (datasheet facts: theta_ja in K/W, t_j_max in degC)"}
            else:
                theta, why_theta = rating(c, "theta_ja", "K/W")
                t_j_max, why_tj = rating(c, "t_j_max", "degC")
                if theta is None or t_j_max is None:
                    thermal = {"status": S.NOT_VERIFIED.value, "reason": "; ".join(w for w in (why_theta, why_tj) if w)}
                elif st.dissipation is None:
                    thermal = {"status": S.NOT_VERIFIED.value, "reason": str(st.p_reason), "theta_ja": theta.value, "t_j_max": t_j_max.value}
                elif t_a is None:
                    thermal = {"status": S.NOT_VERIFIED.value, "reason": str(ambient_reason), "p": st.dissipation.value, "theta_ja": theta.value, "t_j_max": t_j_max.value}
                else:
                    t_j = junction_temperature(t_a, st.dissipation, theta, ("simulation.temperature_c", f"{c.ref}.p", f"{c.ref}.theta_ja"))
                    facts = {"p": st.dissipation.value, "dissipation": traced_json(st.dissipation), "theta_ja": theta.value, "t_a": float(t_a.value),
                             "t_j": t_j.value, "t_j_max": float(t_j_max.value), "calc": traced_json(t_j)}
                    if t_j.value > float(t_j_max.value):
                        thermal = {"status": S.FAIL.value, "reason": f"Tj {fmt_c(t_j.value)} > t_j_max {fmt_c(float(t_j_max.value))}", "repair": "human", **facts}
                    else:
                        thermal = {"status": S.PASS.value, "reason": f"Tj {fmt_c(t_j.value)} <= t_j_max {fmt_c(float(t_j_max.value))}", **facts}
            thermal = gate(thermal, run)
            components[c.ref] = {"status": thermal["status"], "device": st.device, "thermal": {**entry, **thermal}}
        overall = worst_status(S(e["status"]) for e in components.values())
        ambient = f"ambient {fmt_c(float(t_a.value))} (simulation.temperature_c, the op's temperature)" if t_a is not None else str(ambient_reason)
        message = (f"{len(components)} component(s) at the operating point ({aid}): {summarise(components, ('thermal',))}; {ambient}; "
                   "Tj = Ta + P * theta_ja (calc.thermal.T_j) with P from the op; nominal values, operating point only, no transient or tolerance corners")
        if run.spice.status is S.FAIL:
            message = f"computed on an op of a run whose spice summary is FAIL; {message}"
        details: dict[str, Any] = {"components": components, "t_a": float(t_a.value) if t_a is not None else None}
        if overall is S.FAIL:
            details["repair"] = "human"
        return [self._result(ir, overall, message, run, aid, **details)]


class RFImpedanceValidator(Validator):
    id = "domain.rf.impedance"
    domains = frozenset({CircuitDomain.RF})
    description = "Trace impedance / matching / S-parameter checks"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("RF analysis backend not implemented")]


class SignalIntegrityValidator(Validator):
    id = "domain.high_speed.si"
    domains = frozenset({CircuitDomain.HIGH_SPEED})
    description = "Length matching, impedance control, crosstalk"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("signal integrity backend not implemented")]


class PowerIntegrityValidator(Validator):
    id = "domain.power.pi"
    domains = frozenset({CircuitDomain.POWER, CircuitDomain.HIGH_SPEED, CircuitDomain.DIGITAL})
    description = "Decoupling, PDN impedance, IR drop"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("power integrity backend not implemented")]


class AnalogBiasValidator(Validator):
    """The operating point of the current design is real, complete and *judged* by a PASSing expectation.

    PASS only when the latest ``spice`` result (a) ran on the current
    ``SPICE_NETLIST`` artifact - the result's ``artifact_hash`` is the
    artifact's ``content_hash``, the artifact is not stale with respect to
    the IR and unchanged on disk - (b) that run included a successful ``op``
    analysis (read from ``results.json`` through the ``SPICE_RESULT``
    artifact, which must also match its recorded hash and name the same
    netlist hash; all of (a) and (b) through
    :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`) whose vectors cover
    every non-ground IR net with a finite voltage, (c) the ``spice`` summary
    itself is PASS and (d) at least one ``spice.<expectation>`` of that run
    judged a vector of that ``op`` analysis and PASSed. A converged op with
    finite voltages is not a correct bias: a design whose op expectation
    just failed, or that rests on assumed values, converges too - that is
    NOT_VERIFIED ("op converged; bias correctness not judged"), with the
    voltages kept in the details. The validator never computes a bias itself.
    """

    id = "domain.analog.bias"
    domains = frozenset({CircuitDomain.ANALOG, CircuitDomain.MIXED_SIGNAL})
    description = "Operating point of every net from a SPICE run of the current netlist, judged by a PASSing op expectation"
    consumes = frozenset({SPICE_CHECK_ID})

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        run = fresh_spice_run(ir, needs="analog bias needs an operating point from SPICE")
        if isinstance(run, str):
            return [self.not_verified(run)]
        picked = run.op()
        if isinstance(picked, str):
            return [self.not_verified(picked, analyses=run.analysis_ids)]
        aid, res = picked
        netlist, spice = run.netlist, run.spice
        voltages: dict[str, float] = {}
        missing: list[str] = []
        not_finite: list[str] = []
        for net in ir.nets:
            if net.kind == NetKind.GROUND:
                continue
            v = run.node_voltage(ir, net.name)
            if isinstance(v, float):
                voltages[net.name] = v
            elif "no op vector" in v:
                missing.append(net.name)
            else:
                samples = (res.get("vectors") or {}).get(net.name.lower())
                not_finite.append(f"{net.name}={float(samples[-1])!r}" if samples else f"{net.name}: {v}")
        if missing or not_finite:
            return [
                self.not_verified(
                    f"operating point does not cover every net: missing {missing} (a net that touches only excluded parts has no node), non-finite {not_finite}",
                    analysis=aid, missing=missing, not_finite=not_finite, voltages=voltages,
                )
            ]
        evidence = run.evidence(aid)
        details = {"analysis": aid, "command": res.get("command"), "voltages": voltages, "spice_status": spice.status.value}
        # finite is not correct: only a PASSing expectation on this op analysis, in a PASSing run, judges the bias
        judged = [
            r for r in ir.validation.latest_by_check().values()
            if r.check_id.startswith(f"{SPICE_CHECK_ID}.") and r.is_tool_backed and r.artifact_hash == netlist.content_hash and r.details.get("analysis_id") == aid
        ]
        passing = [r.check_id for r in judged if r.status is ValidationStatus.PASS]
        details["op_expectations"] = {r.check_id: r.status.value for r in judged}
        if spice.status is not ValidationStatus.PASS or not passing:
            why = f"the spice summary is {spice.status.value}" if spice.status is not ValidationStatus.PASS else f"no expectation on analysis {aid} PASSed ({details['op_expectations'] or 'none judged'})"
            return [
                ValidationResult(
                    check_id=self.id,
                    status=ValidationStatus.NOT_VERIFIED,
                    message=f"operating point ({aid}) converged with a finite voltage on all {len(voltages)} non-ground net(s), but bias correctness is not judged: {why}",
                    tool=run.engine,
                    tool_version=run.engine_version,
                    artifact_hash=netlist.content_hash,
                    ir_hash=ir.content_hash(),
                    evidence=evidence,
                    details=details,
                )
            ]
        return [
            ValidationResult(
                check_id=self.id,
                status=ValidationStatus.PASS,
                message=f"operating point ({aid}) gives a finite voltage on all {len(voltages)} non-ground net(s), judged by {passing}",
                tool=run.engine,
                tool_version=run.engine_version,
                artifact_hash=netlist.content_hash,
                ir_hash=ir.content_hash(),
                evidence=evidence,
                details=details,
            )
        ]


for _v in (
    PowerThermalValidator(),
    RFImpedanceValidator(),
    SignalIntegrityValidator(),
    PowerIntegrityValidator(),
    AnalogBiasValidator(),
):
    default_registry.register(_v)
