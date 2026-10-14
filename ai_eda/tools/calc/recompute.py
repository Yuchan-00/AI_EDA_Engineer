"""Recompute every derived traced value with the calculator that claims to have produced it.

Invariant: a ``derived`` value is only evidence when the tool it names is one
of the registered deterministic calculators and running that calculator
again on the inputs its provenance names gives the stored value back. The
CALCULATION stage records the outcome as the ``calc.recompute``
:class:`~ai_eda.ir.ValidationResult`:

* every derived value recomputed and equal (relative 1e-9 of the larger magnitude, no absolute floor: a pF or nH
  value is compared as closely as a kilohm; exact zeros are equal) -> PASS;
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

The walk also covers the traced numbers of ``ir.si`` (their ids,
``si.net_classes[POWER].min_width_mm``, are the paths). Inputs are looked up
in ``ir.parameters`` first, then - for an id under :data:`SI_PREFIX` - in
``ir.si``, for an id under :data:`STACKUP_PREFIX` - in the board's stackup (a transmission-line
calculator records ``pcb.stackup.dielectrics[0].thickness_mm`` and the like),
and, like the reviewer's ``calculations_vs_design`` check, in
``ir.requirements`` (a requirement's traced ``value``) as a fallback.

The walk also covers ``ir.rf`` (``rf.networks[lpf].expectations[s21_fc].nominal``,
``RFDesign.traced_items``); its derived values take their inputs from
``ir.parameters`` (dotted keys such as ``rf.n_mult`` included), the
requirements or the stackup, never from an ``ir.rf`` item.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator

from ai_eda.ir import CircuitIR, ProvenanceKind, Traced, ValidationResult, ValidationStatus
from ai_eda.tools.calc.basic import (
    CALC_VERSION,
    ROLE_UNITS,
    ROLES,
    astable_c_for_frequency,
    astable_frequency,
    astable_tran_start,
    astable_tran_step,
    astable_tran_stop,
    astable_v_be_reverse,
    clock_divided,
    crystal_load_capacitance,
    current_from_voltage_resistance,
    divider_r1_for_v_out,
    ipc2221_width_for_current,
    junction_temperature,
    lc_cutoff,
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
    rc_tran_step,
    rc_tran_stop,
    regulator_dissipation,
    voltage_divider_output,
    voltage_divider_ratio,
)
from ai_eda.tools.calc.radio import RADIO_CALCULATORS
from ai_eda.tools.calc.rf import RF_CALCULATORS
from ai_eda.tools.calc.tline import (
    critical_length,
    edge_coupled_e_eff_even,
    edge_coupled_e_eff_odd,
    edge_coupled_z_diff,
    edge_coupled_z_even,
    edge_coupled_z_odd,
    line_delay,
    microstrip_e_eff,
    microstrip_z0,
    spacing_for_zdiff,
    stripline_z0,
    tpd,
    width_for_z0_microstrip,
    width_for_z0_stripline,
    width_for_zdiff,
)

CHECK_ID = "calc.recompute"
#: ids under this prefix name a traced number of the design's SI constraints (``si.net_classes[POWER].min_width_mm``), resolved
#: by :meth:`ai_eda.ir.SIConstraints.lookup`; the derived ones among them are recomputed like the parameters
SI_PREFIX = "si"
#: ids under this prefix name a number of the board's stackup (``pcb.stackup.dielectrics[0].er``), resolved by
#: :meth:`ai_eda.ir.Stackup.lookup`: a transmission-line value records the stack's numbers it was computed from
STACKUP_PREFIX = "pcb.stackup"
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
    "calc.astable.c_for_frequency": (astable_c_for_frequency, ROLES["calc.astable.c_for_frequency"]),
    "calc.astable.f": (astable_frequency, ROLES["calc.astable.f"]),
    "calc.astable.tran_step": (astable_tran_step, ROLES["calc.astable.tran_step"]),
    "calc.astable.tran_stop": (astable_tran_stop, ROLES["calc.astable.tran_stop"]),
    "calc.astable.tran_start": (astable_tran_start, ROLES["calc.astable.tran_start"]),
    "calc.astable.v_be_reverse": (astable_v_be_reverse, ROLES["calc.astable.v_be_reverse"]),
    "calc.rc.tran_step": (rc_tran_step, ROLES["calc.rc.tran_step"]),
    "calc.rc.tran_stop": (rc_tran_stop, ROLES["calc.rc.tran_stop"]),
    "calc.regulator.p_dissipation": (regulator_dissipation, ROLES["calc.regulator.p_dissipation"]),
    "calc.crystal.load_capacitance": (crystal_load_capacitance, ROLES["calc.crystal.load_capacitance"]),
    "calc.lc.cutoff": (lc_cutoff, ROLES["calc.lc.cutoff"]),
    "calc.tline.microstrip.z0": (microstrip_z0, ROLES["calc.tline.microstrip.z0"]),
    "calc.tline.microstrip.e_eff": (microstrip_e_eff, ROLES["calc.tline.microstrip.e_eff"]),
    "calc.tline.stripline.z0": (stripline_z0, ROLES["calc.tline.stripline.z0"]),
    "calc.tline.edge_coupled_microstrip.z_even": (edge_coupled_z_even, ROLES["calc.tline.edge_coupled_microstrip.z_even"]),
    "calc.tline.edge_coupled_microstrip.z_odd": (edge_coupled_z_odd, ROLES["calc.tline.edge_coupled_microstrip.z_odd"]),
    "calc.tline.edge_coupled_microstrip.z_diff": (edge_coupled_z_diff, ROLES["calc.tline.edge_coupled_microstrip.z_diff"]),
    "calc.tline.edge_coupled_microstrip.e_eff_even": (edge_coupled_e_eff_even, ROLES["calc.tline.edge_coupled_microstrip.e_eff_even"]),
    "calc.tline.edge_coupled_microstrip.e_eff_odd": (edge_coupled_e_eff_odd, ROLES["calc.tline.edge_coupled_microstrip.e_eff_odd"]),
    "calc.tline.width_for_z0.microstrip": (width_for_z0_microstrip, ROLES["calc.tline.width_for_z0.microstrip"]),
    "calc.tline.width_for_z0.stripline": (width_for_z0_stripline, ROLES["calc.tline.width_for_z0.stripline"]),
    "calc.tline.edge_coupled_microstrip.s_for_zdiff": (spacing_for_zdiff, ROLES["calc.tline.edge_coupled_microstrip.s_for_zdiff"]),
    "calc.tline.edge_coupled_microstrip.w_for_zdiff": (width_for_zdiff, ROLES["calc.tline.edge_coupled_microstrip.w_for_zdiff"]),
    "calc.tline.tpd": (tpd, ROLES["calc.tline.tpd"]),
    "calc.tline.delay": (line_delay, ROLES["calc.tline.delay"]),
    "calc.tline.critical_length": (critical_length, ROLES["calc.tline.critical_length"]),
    "calc.ipc2221.width_for_current": (ipc2221_width_for_current, ROLES["calc.ipc2221.width_for_current"]),
    "calc.clock.divided": (clock_divided, ROLES["calc.clock.divided"]),
    # RF (ai_eda.tools.calc.rf): every calc.rf.* calculator and calc.crystal.c_for_load, with the roles basic.ROLES names
    **{tool: (fn, ROLES[tool]) for tool, fn in RF_CALCULATORS.items()},
    # RADIO (ai_eda.tools.calc.radio): frequency plan, FM / PM, top-C and crystal-ladder networks, pads, audio, power budget
    **{tool: (fn, ROLES[tool]) for tool, fn in RADIO_CALCULATORS.items()},
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
    if key.startswith(STACKUP_PREFIX) and ir.pcb is not None and ir.pcb.stackup is not None:
        return ir.pcb.stackup.lookup(key, STACKUP_PREFIX)
    if key.startswith(SI_PREFIX + ".") and ir.si is not None:
        return ir.si.lookup(key, SI_PREFIX)
    req = ir.requirements.get(key)
    return req.value if req is not None else None


def _same(a: float, b: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    # purely relative: an absolute floor (the former max(1.0, ...)) made every pF / nH value "equal" to any other,
    # so a tampered matching or filter capacitor recomputed PASS; exact zeros are still equal (0 <= 0)
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def derived_values(ir: CircuitIR) -> Iterator[tuple[str, Traced]]:
    """Every ``Traced`` with ``derived`` provenance the design holds, as ``(path, traced)``.

    ``ir.parameters`` entries keep their key as the path; everything else is
    addressed by where it lives (``components[R1].spice.value``,
    ``simulation.expectations[v_out].nominal`` ...). ``ir.si`` and ``ir.rf``
    are walked through their own ``traced_items`` ids
    (``si.net_classes[POWER].min_width_mm``,
    ``rf.networks[lpf].expectations[s21_fc].nominal``): a fixture nominal a
    calculator produced is re-derived like a parameter (its inputs live in
    ``ir.parameters``, the requirements or the stackup, never in ``ir.rf``).
    """
    for key, t in ir.parameters.items():
        if t.provenance.kind == ProvenanceKind.DERIVED:
            yield key, t
    if ir.si is not None:
        for key, t in ir.si.traced_items(SI_PREFIX):
            if t.provenance.kind == ProvenanceKind.DERIVED:
                yield key, t
    if ir.rf is not None:
        for key, t in ir.rf.traced_items():
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
        for k, t in e.params.items():
            if t.provenance.kind == ProvenanceKind.DERIVED:
                yield f"simulation.expectations[{e.id}].params[{k}]", t
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
                    except (ValueError, ArithmeticError, TypeError) as e:
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
            refuse(f"input(s) {missing} not found in ir.parameters / ir.si / the stackup / ir.requirements")
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
        except (ValueError, ArithmeticError, TypeError) as e:  # ArithmeticError: ZeroDivisionError, and an OverflowError of a float ** / exp
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


__all__ = ["CALCULATORS", "CHECK_ID", "REL_TOL", "SI_PREFIX", "STACKUP_PREFIX", "TOOL_ID", "derived_values", "recompute_parameters", "resolve_input", "unit_key"]

#: public name of :func:`_unit_key` for the validators that compare a rating's unit with a role's (``K/W`` == ``degC/W``)
unit_key = _unit_key

#: public name of :func:`_input`: the traced value an id names in ``ir.parameters``, ``ir.si``, the stackup or ``ir.requirements`` (``None``
#: when none) - the reviewer checks a derived value's inputs exist with the same resolution the recompute uses
resolve_input = _input
