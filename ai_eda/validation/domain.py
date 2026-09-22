"""Domain-specific validators.

:class:`AnalogBiasValidator` reads real SPICE evidence; the others still
return NOT_VERIFIED until an analysis backend exists. Keeping the stubs here
means the *selection* mechanism is exercised end to end and the GUI can show
"this check exists but has not run".
"""

from __future__ import annotations

import math
from pathlib import Path

from ai_eda.ir import ArtifactKind, CircuitDomain, CircuitIR, Evidence, NetKind, ValidationResult, ValidationStatus
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID, read_results
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry


class PowerThermalValidator(Validator):
    id = "domain.power.thermal"
    domains = frozenset({CircuitDomain.POWER})
    description = "Power dissipation and junction temperature vs datasheet limits"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("thermal analysis backend not implemented")]


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
    netlist hash) whose vectors cover every non-ground IR net with a finite
    voltage, (c) the ``spice`` summary itself is PASS and (d) at least one
    ``spice.<expectation>`` of that run judged a vector of that ``op``
    analysis and PASSed. A converged op with finite voltages is not a
    correct bias: a design whose op expectation just failed, or that rests
    on assumed values, converges too - that is NOT_VERIFIED ("op converged;
    bias correctness not judged"), with the voltages kept in the details.
    The validator never computes a bias itself.
    """

    id = "domain.analog.bias"
    domains = frozenset({CircuitDomain.ANALOG, CircuitDomain.MIXED_SIGNAL})
    description = "Operating point of every net from a SPICE run of the current netlist, judged by a PASSing op expectation"
    consumes = frozenset({SPICE_CHECK_ID})

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        spice = ir.validation.latest(SPICE_CHECK_ID)
        if spice is None or not spice.is_tool_backed or spice.artifact_hash is None:
            return [self.not_verified("analog bias needs an operating point from SPICE; no tool-backed spice result attached")]
        netlist = ir.artifacts.get(ArtifactKind.SPICE_NETLIST)
        if netlist is None:
            return [self.not_verified("no SPICE netlist artifact")]
        if netlist.is_stale(ir.content_hash()):
            return [self.not_verified("the SPICE netlist was generated from a different IR version; regenerate and re-simulate")]
        if not netlist.matches_disk():
            return [self.not_verified("the SPICE netlist on disk does not match its recorded hash")]
        if spice.artifact_hash != netlist.content_hash:
            return [self.not_verified("the latest spice result ran on a different netlist than the current artifact")]
        results = ir.artifacts.get(ArtifactKind.SPICE_RESULT)
        if results is None or not Path(results.path).is_file():
            return [self.not_verified("no SPICE results artifact (results.json)")]
        if not results.matches_disk():
            return [self.not_verified("results.json on disk does not match its recorded hash")]
        try:
            data = read_results(results.path)
        except ValueError as e:
            return [self.not_verified(f"unreadable results.json: {e}")]
        if data.get("netlist_hash") != netlist.content_hash:
            return [self.not_verified("results.json was produced for a different netlist than the current artifact")]
        ops = [(aid, a) for aid, a in data["analyses"].items() if a.get("kind") == "op" and a.get("result", {}).get("succeeded")]
        if not ops:
            return [self.not_verified("no successful operating-point (op) analysis in the simulation results", analyses=sorted(data["analyses"]))]
        aid, analysis = ops[0]
        res = analysis["result"]
        vectors: dict = res.get("vectors", {})
        voltages: dict[str, float] = {}
        missing: list[str] = []
        not_finite: list[str] = []
        for net in ir.nets:
            if net.kind == NetKind.GROUND:
                continue
            samples = vectors.get(net.name.lower())
            if not samples:
                missing.append(net.name)
                continue
            v = float(samples[-1])
            if not math.isfinite(v):
                not_finite.append(f"{net.name}={v!r}")
            else:
                voltages[net.name] = v
        if missing or not_finite:
            return [
                self.not_verified(
                    f"operating point does not cover every net: missing {missing} (a net that touches only excluded parts has no node), non-finite {not_finite}",
                    analysis=aid, missing=missing, not_finite=not_finite, voltages=voltages,
                )
            ]
        evidence = [Evidence(description="spice results.json", path=results.path, content_hash=results.content_hash)]
        if res.get("raw_output_path"):
            evidence.insert(0, Evidence(description=f"ngspice rawfile of analysis {aid} ({res.get('command')})", path=res["raw_output_path"], content_hash=res.get("raw_output_hash")))
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
                    tool=str(data.get("engine")),
                    tool_version=str(data.get("engine_version")),
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
                tool=str(data.get("engine")),
                tool_version=str(data.get("engine_version")),
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
