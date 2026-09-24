"""Recompute every derived traced value with the calculator that claims to have produced it.

Invariant: a ``derived`` value is only evidence when the tool it names is one
of the registered deterministic calculators and running that calculator
again on the inputs its provenance names gives the stored value back. The
CALCULATION stage records the outcome as the ``calc.recompute``
:class:`~ai_eda.ir.ValidationResult`:

* every derived value recomputed and equal (relative 1e-9) -> PASS;
* a recomputed value that differs -> FAIL with both values (``repair: human``:
  either the stored number, its inputs or its recorded provenance is wrong,
  and a tool must not pick);
* a derived value whose tool is not registered, whose provenance records no
  roles (``Provenance.inputs``), whose roles are not the calculator's, whose
  inputs are missing or carry the wrong unit for their role, or whose
  calculator refuses (domain error) -> NOT_VERIFIED for that value, and
  NOT_VERIFIED overall unless something else FAILed;
* no derived values -> NOT_VERIFIED (nothing was recomputed).

The call is rebuilt **by role** from ``Provenance.inputs`` (the calculators
write it, :mod:`ai_eda.tools.calc.basic`), never from the positional order of
``derived_from``: a provenance that names the ids in the wrong roles is
reported as exactly that, and a value whose provenance records no roles is
not verifiable (its positional recompute is still shown in the details for
the human).

What is walked: ``ir.parameters`` and every other ``Traced`` a design number
can hide in - expectation ``nominal`` / ``at`` / ``tol_abs`` / ``tol_rel``,
stimulus ``value`` / ``params``, analysis ``params``, ``temperature_c``,
each component's SPICE ``value`` / ``params`` and ``electrical`` entries -
reported under their path (``simulation.expectations[v_out].nominal``).
After a JSON round trip these are independent copies of the parameters they
were built from, so a recomputed parameter says nothing about them.

Inputs are looked up in ``ir.parameters`` first and, like the reviewer's
``calculations_vs_design`` check, in ``ir.requirements`` (a requirement's
traced ``value``) as a fallback.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator

from ai_eda.ir import CircuitIR, ProvenanceKind, Traced, ValidationResult, ValidationStatus
from ai_eda.tools.calc.basic import (
    CALC_VERSION,
    ROLE_UNITS,
    ROLES,
    current_from_voltage_resistance,
    divider_r1_for_v_out,
    junction_temperature,
    led_current,
    led_series_resistor,
    parallel_resistance,
    power_from_voltage_current,
    power_from_voltage_resistance,
    rc_ac_fstart,
    rc_ac_fstop,
    rc_lowpass_magnitude,
    rc_lowpass_phase_deg,
    rc_r_for_cutoff,
    rc_step_response,
    rc_time_constant,
    voltage_divider_output,
    voltage_divider_ratio,
)

CHECK_ID = "calc.recompute"
TOOL_ID = "calc"
#: relative tolerance for "the same number" (floating point round-off between two evaluations)
REL_TOL = 1e-9

Calculator = Callable[..., Traced]

#: tool id (as written into ``Provenance.tool`` by the calculator) -> (callable, ordered input roles)
CALCULATORS: dict[str, tuple[Calculator, tuple[str, ...]]] = {
    "calc.ohms_law.I": (current_from_voltage_resistance, ROLES["calc.ohms_law.I"]),
    "calc.power.P": (power_from_voltage_current, ROLES["calc.power.P"]),
    "calc.power.P_VR": (power_from_voltage_resistance, ROLES["calc.power.P_VR"]),
    "calc.thermal.T_j": (junction_temperature, ROLES["calc.thermal.T_j"]),
    "calc.divider.ratio": (voltage_divider_ratio, ROLES["calc.divider.ratio"]),
    "calc.divider.v_out": (voltage_divider_output, ROLES["calc.divider.v_out"]),
    "calc.rc.tau": (rc_time_constant, ROLES["calc.rc.tau"]),
    "calc.rc.step_response": (rc_step_response, ROLES["calc.rc.step_response"]),
    "calc.rc.lowpass_magnitude": (rc_lowpass_magnitude, ROLES["calc.rc.lowpass_magnitude"]),
    "calc.rc.lowpass_phase_deg": (rc_lowpass_phase_deg, ROLES["calc.rc.lowpass_phase_deg"]),
    "calc.parallel.R": (parallel_resistance, ROLES["calc.parallel.R"]),
    "calc.led.R": (led_series_resistor, ROLES["calc.led.R"]),
    "calc.divider.r1_for_v_out": (divider_r1_for_v_out, ROLES["calc.divider.r1_for_v_out"]),
    "calc.led.I": (led_current, ROLES["calc.led.I"]),
    "calc.rc.r_for_cutoff": (rc_r_for_cutoff, ROLES["calc.rc.r_for_cutoff"]),
    "calc.rc.ac_fstart": (rc_ac_fstart, ROLES["calc.rc.ac_fstart"]),
    "calc.rc.ac_fstop": (rc_ac_fstop, ROLES["calc.rc.ac_fstop"]),
}

#: lower-cased unit spelling -> the key a role's expected unit lower-cases to. A thermal resistance is written
#: degC/W (or with the degree sign) in most datasheets and K/W in the calculator roles: the same unit.
_UNIT_ALIASES = {
    "ω": "ohm", "ohms": "ohm", "ohm": "ohm", "r": "ohm", "volt": "v", "volts": "v", "sec": "s", "secs": "s", "farad": "f", "hz": "hz",
    "k/w": "k/w", "degc/w": "k/w", "°c/w": "k/w", "℃/w": "k/w", "deg c/w": "k/w",
    "degc": "degc", "°c": "degc", "℃": "degc", "deg c": "degc",
}


def _unit_key(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = " ".join(unit.strip().lower().split())
    return _UNIT_ALIASES.get(u, u)


def _input(ir: CircuitIR, key: str) -> Traced | None:
    if key in ir.parameters:
        return ir.parameters[key]
    req = ir.requirements.get(key)
    return req.value if req is not None else None


def _same(a: float, b: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def derived_values(ir: CircuitIR) -> Iterator[tuple[str, Traced]]:
    """Every ``Traced`` with ``derived`` provenance the design holds, as ``(path, traced)``.

    ``ir.parameters`` entries keep their key as the path; everything else is
    addressed by where it lives (``components[R1].spice.value``,
    ``simulation.expectations[v_out].nominal`` ...).
    """
    for key, t in ir.parameters.items():
        if t.provenance.kind == ProvenanceKind.DERIVED:
            yield key, t
    for c in ir.components:
        for k, t in c.electrical.items():
            if t.provenance.kind == ProvenanceKind.DERIVED:
                yield f"components[{c.ref}].electrical[{k}]", t
        if c.spice is not None:
            if c.spice.value is not None and c.spice.value.provenance.kind == ProvenanceKind.DERIVED:
                yield f"components[{c.ref}].spice.value", c.spice.value
            for k, t in c.spice.params.items():
                if t.provenance.kind == ProvenanceKind.DERIVED:
                    yield f"components[{c.ref}].spice.params[{k}]", t
    setup = ir.simulation
    if setup is None:
        return
    for s in setup.stimuli:
        if s.value is not None and s.value.provenance.kind == ProvenanceKind.DERIVED:
            yield f"simulation.stimuli[{s.id}].value", s.value
        for k, t in s.params.items():
            if t.provenance.kind == ProvenanceKind.DERIVED:
                yield f"simulation.stimuli[{s.id}].params[{k}]", t
    for a in setup.analyses:
        for k, t in a.params.items():
            if t.provenance.kind == ProvenanceKind.DERIVED:
                yield f"simulation.analyses[{a.id}].params[{k}]", t
    for e in setup.expectations:
        for label, t in (("nominal", e.nominal), ("at", e.at), ("tol_abs", e.tol_abs), ("tol_rel", e.tol_rel)):
            if t is not None and t.provenance.kind == ProvenanceKind.DERIVED:
                yield f"simulation.expectations[{e.id}].{label}", t
    if setup.temperature_c is not None and setup.temperature_c.provenance.kind == ProvenanceKind.DERIVED:
        yield "simulation.temperature_c", setup.temperature_c


def recompute_parameters(ir: CircuitIR) -> ValidationResult:
    """Recompute every ``derived`` value of the IR (see the module docstring for the verdict rules)."""
    per_parameter: dict[str, dict] = {}
    mismatches: list[str] = []
    unverified: list[str] = []
    recomputed = 0
    for key, traced in derived_values(ir):
        prov = traced.provenance
        entry: dict = {"tool": prov.tool, "tool_version": prov.tool_version, "inputs": list(prov.derived_from), "roles": dict(prov.inputs), "stored": traced.value}
        per_parameter[key] = entry

        def refuse(reason: str) -> None:
            entry["status"] = ValidationStatus.NOT_VERIFIED
            entry["reason"] = reason
            unverified.append(f"{key}: {reason}")

        calc = CALCULATORS.get(prov.tool or "")
        if calc is None:
            refuse(f"tool {prov.tool!r} is not a registered calculator")
            continue
        fn, roles = calc
        if not prov.inputs:
            reason = f"provenance records no input roles (only the positional list {list(prov.derived_from)}); {prov.tool} takes {len(roles)} inputs {list(roles)}"
            if len(prov.derived_from) == len(roles):
                positional = [_input(ir, k) for k in prov.derived_from]
                if all(v is not None for v in positional):
                    try:
                        entry["positional_recompute"] = fn(*positional, tuple(prov.derived_from)).value
                    except (ValueError, ZeroDivisionError, TypeError) as e:
                        entry["positional_recompute"] = f"refused: {e}"
            refuse(reason)
            continue
        if list(prov.inputs) != list(roles):
            refuse(f"recorded roles {list(prov.inputs)} are not {prov.tool}'s roles {list(roles)}")
            continue
        if list(prov.inputs.values()) != list(prov.derived_from):
            refuse(f"inputs {prov.inputs} and derived_from {list(prov.derived_from)} name different ids")
            continue
        inputs = {role: _input(ir, k) for role, k in prov.inputs.items()}
        missing = [k for role, k in prov.inputs.items() if inputs[role] is None]
        if missing:
            refuse(f"input(s) {missing} not found in ir.parameters / ir.requirements")
            continue
        bad_units = [
            f"{role}={prov.inputs[role]!r} carries unit {inputs[role].unit!r}, {prov.tool} expects {expected!r}"
            for role, expected in zip(roles, ROLE_UNITS.get(prov.tool or "", ()))
            if expected is not None and inputs[role].unit is not None and _unit_key(inputs[role].unit) != _unit_key(expected)
        ]
        if bad_units:
            refuse("input unit does not fit its role: " + "; ".join(bad_units))
            continue
        entry["input_values"] = {k: inputs[role].value for role, k in prov.inputs.items()}
        try:
            result = fn(*(inputs[role] for role in roles), tuple(prov.derived_from))
        except (ValueError, ZeroDivisionError, TypeError) as e:
            refuse(f"{prov.tool} refused the inputs: {e}")
            continue
        entry["recomputed"] = result.value
        entry["calc_version"] = result.provenance.tool_version
        recomputed += 1
        if _same(result.value, traced.value):
            entry["status"] = ValidationStatus.PASS
        else:
            entry["status"] = ValidationStatus.FAIL
            mismatches.append(f"{key}: stored {traced.value!r}, {prov.tool} recomputes {result.value!r} from {entry['input_values']}")
    details: dict = {"parameters": per_parameter, "mismatches": mismatches, "unverified": unverified, "rel_tol": REL_TOL}
    if mismatches:
        details["repair"] = "human"
        return ValidationResult(
            check_id=CHECK_ID, status=ValidationStatus.FAIL, message=f"{len(mismatches)} derived value(s) do not match their calculator: " + "; ".join(mismatches),
            tool=TOOL_ID, tool_version=CALC_VERSION, details=details,
        )
    if not per_parameter:
        return ValidationResult(check_id=CHECK_ID, status=ValidationStatus.NOT_VERIFIED, message="no derived parameters to recompute", tool=TOOL_ID, tool_version=CALC_VERSION, details=details)
    if unverified:
        return ValidationResult(
            check_id=CHECK_ID, status=ValidationStatus.NOT_VERIFIED,
            message=f"{recomputed} value(s) recomputed, {len(unverified)} could not be: " + "; ".join(unverified),
            tool=TOOL_ID, tool_version=CALC_VERSION, details=details,
        )
    return ValidationResult(check_id=CHECK_ID, status=ValidationStatus.PASS, message=f"{recomputed} value(s) recomputed", tool=TOOL_ID, tool_version=CALC_VERSION, details=details)


__all__ = ["CALCULATORS", "CHECK_ID", "REL_TOL", "TOOL_ID", "derived_values", "recompute_parameters", "unit_key"]

#: public name of :func:`_unit_key` for the validators that compare a rating's unit with a role's (``K/W`` == ``degC/W``)
unit_key = _unit_key
