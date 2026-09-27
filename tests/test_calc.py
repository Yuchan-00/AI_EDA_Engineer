import math

import pytest

from ai_eda.ir import ProvenanceKind, Requirement, RequirementKind, SourceRef, ValidationStatus, authoritative, derived, user_requirement
from ai_eda.tools.calc import (
    CALCULATORS,
    CALC_VERSION,
    ROLE_UNITS,
    ROLES,
    crystal_load_capacitance,
    current_from_voltage_resistance,
    lc_cutoff,
    led_series_resistor,
    parallel_resistance,
    power_from_voltage_current,
    rc_lowpass_magnitude,
    rc_lowpass_phase_deg,
    rc_step_response,
    rc_time_constant,
    rc_tran_step,
    rc_tran_stop,
    recompute_parameters,
    regulator_dissipation,
    voltage_divider_output,
)

DS = SourceRef(title="ds")


def test_divider_output_is_derived_with_inputs():
    v = voltage_divider_output(user_requirement(12.0, "V"), authoritative(10e3, DS), authoritative(10e3, DS))
    assert v.value == pytest.approx(6.0)
    assert v.unit == "V"
    assert v.provenance.kind == ProvenanceKind.DERIVED
    assert v.provenance.tool == "calc.divider.v_out"
    assert v.provenance.derived_from == ["v_in", "r1", "r2"]
    assert v.provenance.inputs == {"v_in": "v_in", "r1": "r1", "r2": "r2"}  # role -> id, written by the calculator
    v = voltage_divider_output(user_requirement(12.0, "V"), authoritative(10e3, DS), authoritative(10e3, DS), ("supply", "top", "bottom"))
    assert v.provenance.inputs == {"v_in": "supply", "r1": "top", "r2": "bottom"} and v.provenance.derived_from == ["supply", "top", "bottom"]
    with pytest.raises(ValueError, match="takes 2 input ids"):
        voltage_divider_output(user_requirement(12.0, "V"), authoritative(10e3, DS), authoritative(10e3, DS), ("v_in", "r1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="takes 2 input ids"):
        rc_time_constant(authoritative(1e3, DS), authoritative(1e-6, DS), ("r", "c", "extra"))  # type: ignore[arg-type]


def test_ohm_and_power():
    i = current_from_voltage_resistance(user_requirement(5.0), authoritative(250.0, DS))
    assert i.value == pytest.approx(0.02)
    p = power_from_voltage_current(user_requirement(5.0), i)
    assert p.value == pytest.approx(0.1)
    assert p.unit == "W"


def test_rc():
    assert rc_time_constant(authoritative(1e3, DS), authoritative(1e-6, DS)).value == pytest.approx(1e-3)
    tau = rc_time_constant(authoritative(1e3, DS), authoritative(1e-6, DS))
    v = rc_step_response(user_requirement(5.0, "V"), tau, tau)
    assert v.value == pytest.approx(5.0 * (1 - math.exp(-1))) and v.provenance.tool == "calc.rc.step_response" and v.unit == "V"
    assert rc_step_response(user_requirement(5.0), user_requirement(0.0), tau).value == 0.0
    with pytest.raises(ValueError):
        rc_step_response(user_requirement(5.0), user_requirement(1.0), user_requirement(0.0))
    r = parallel_resistance(authoritative(10e3, DS), authoritative(10e3, DS))
    assert r.value == pytest.approx(5e3) and r.provenance.tool == "calc.parallel.R"


def test_every_calculator_is_registered_with_its_own_tool_id():
    """The registry must name the tool id each calculator writes, else the CALCULATION stage could never recompute it."""
    sample = {
        "calc.ohms_law.I": (user_requirement(12.0), authoritative(1e3, DS)),
        "calc.power.P": (user_requirement(12.0), user_requirement(0.5)),
        "calc.power.P_VR": (user_requirement(6.0), authoritative(1e4, DS)),
        "calc.thermal.T_j": (user_requirement(85.0), user_requirement(3.6e-3), authoritative(100.0, DS)),
        "calc.divider.ratio": (authoritative(1e3, DS), authoritative(1e3, DS)),
        "calc.divider.v_out": (user_requirement(12.0), authoritative(1e3, DS), authoritative(1e3, DS)),
        "calc.rc.tau": (authoritative(1e3, DS), authoritative(1e-6, DS)),
        "calc.rc.step_response": (user_requirement(5.0), user_requirement(1e-3), user_requirement(1e-3)),
        "calc.rc.lowpass_magnitude": (user_requirement(150.0), user_requirement(1e-3)),
        "calc.rc.lowpass_phase_deg": (user_requirement(150.0), user_requirement(1e-3)),
        "calc.parallel.R": (authoritative(1e3, DS), authoritative(1e3, DS)),
        "calc.led.R": (user_requirement(5.0), authoritative(2.0, DS), authoritative(0.01, DS)),
        "calc.divider.r1_for_v_out": (user_requirement(12.0), user_requirement(5.0), user_requirement(1e4)),
        "calc.led.I": (user_requirement(5.0), authoritative(2.0, DS), user_requirement(300.0)),
        "calc.rc.r_for_cutoff": (user_requirement(1e3), user_requirement(1e-7)),
        "calc.rc.ac_fstart": (user_requirement(1e3),),
        "calc.rc.ac_fstop": (user_requirement(1e3),),
        "calc.astable.c_for_frequency": (user_requirement(1e3), user_requirement(1e4), user_requirement(5.0), user_requirement(0.7)),
        "calc.astable.f": (user_requirement(1e4), user_requirement(72e-9), user_requirement(5.0), user_requirement(0.7)),
        "calc.astable.tran_step": (user_requirement(1e3),),
        "calc.astable.tran_stop": (user_requirement(1e3),),
        "calc.astable.tran_start": (user_requirement(1e3),),
        "calc.astable.v_be_reverse": (user_requirement(5.0), user_requirement(0.7)),
        "calc.rc.tran_step": (user_requirement(1e-3),),
        "calc.rc.tran_stop": (user_requirement(1e-3),),
        "calc.regulator.p_dissipation": (user_requirement(9.0), user_requirement(5.0), user_requirement(0.05)),
        "calc.crystal.load_capacitance": (user_requirement(22e-12), user_requirement(22e-12), user_requirement(4e-12)),
        "calc.lc.cutoff": (user_requirement(10e-6), user_requirement(100e-9)),
        # transmission lines (CALC_VERSION 0.8): mm, um copper, er
        "calc.tline.microstrip.z0": (user_requirement(0.35), user_requirement(0.2), user_requirement(35.0), user_requirement(4.5)),
        "calc.tline.microstrip.e_eff": (user_requirement(0.35), user_requirement(0.2), user_requirement(35.0), user_requirement(4.5)),
        "calc.tline.stripline.z0": (user_requirement(0.3), user_requirement(1.0), user_requirement(17.5), user_requirement(4.5)),
        "calc.tline.edge_coupled_microstrip.z_even": (user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.edge_coupled_microstrip.z_odd": (user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.edge_coupled_microstrip.z_diff": (user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.edge_coupled_microstrip.e_eff_even": (user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.edge_coupled_microstrip.e_eff_odd": (user_requirement(0.15), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.width_for_z0.microstrip": (user_requirement(50.0), user_requirement(0.2), user_requirement(35.0), user_requirement(4.5)),
        "calc.tline.width_for_z0.stripline": (user_requirement(50.0), user_requirement(1.0), user_requirement(17.5), user_requirement(4.5)),
        "calc.tline.edge_coupled_microstrip.s_for_zdiff": (user_requirement(100.0), user_requirement(0.3), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.edge_coupled_microstrip.w_for_zdiff": (user_requirement(100.0), user_requirement(0.2), user_requirement(0.2), user_requirement(0.0), user_requirement(4.3)),
        "calc.tline.tpd": (user_requirement(3.2),),
        "calc.tline.delay": (user_requirement(100.0), user_requirement(6e-9)),
        "calc.tline.critical_length": (user_requirement(1e-9), user_requirement(6e-9), user_requirement(0.5)),
        "calc.ipc2221.width_for_current": (user_requirement(1.0), user_requirement(10.0), user_requirement(35.0)),
        "calc.clock.divided": (user_requirement(16e6), user_requirement(4.0)),
    }
    assert set(CALCULATORS) == set(sample) == set(ROLES) == set(ROLE_UNITS)
    for tool, (fn, keys) in CALCULATORS.items():
        out = fn(*sample[tool], keys)
        assert out.provenance.tool == tool and out.provenance.derived_from == list(keys) and out.provenance.tool_version == CALC_VERSION
        assert out.provenance.inputs == dict(zip(keys, keys)) and len(ROLE_UNITS[tool]) == len(keys)


def test_rc_lowpass_magnitude_and_phase():
    tau = user_requirement(1e-3, "s")
    fc = 1 / (2 * math.pi * 1e-3)
    assert rc_lowpass_magnitude(user_requirement(fc, "Hz"), tau).value == pytest.approx(1 / math.sqrt(2))
    assert rc_lowpass_phase_deg(user_requirement(fc, "Hz"), tau).value == pytest.approx(-45.0)
    assert rc_lowpass_magnitude(user_requirement(0.0, "Hz"), tau).value == 1.0 and rc_lowpass_phase_deg(user_requirement(0.0, "Hz"), tau).value == 0.0
    with pytest.raises(ValueError):
        rc_lowpass_magnitude(user_requirement(1.0, "Hz"), user_requirement(0.0, "s"))


def test_recompute_parameters_verdicts(divider_ir):
    ir = divider_ir
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.PASS and res.message == "1 value(s) recomputed" and res.tool == "calc"
    assert res.details["parameters"]["v_out"] == {**res.details["parameters"]["v_out"], "stored": 6.0, "recomputed": 6.0, "status": "PASS", "inputs": ["v_in", "r1", "r2"]}
    # a stored number the calculator does not reproduce is FAIL for a human (the tool never picks a side)
    ir.parameters["v_out"] = derived(6.5, tool="calc.divider.v_out", inputs={"v_in": "v_in", "r1": "r1", "r2": "r2"}, unit="V")
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.FAIL and res.details["repair"] == "human"
    assert res.details["mismatches"] == ["v_out: stored 6.5, calc.divider.v_out recomputes 6.0 from {'v_in': 12.0, 'r1': 10000.0, 'r2': 10000.0}"]
    # unknown tool / missing input / wrong roles / no roles at all -> NOT_VERIFIED for that parameter, never PASS
    ir.parameters["v_out"] = derived(6.0, tool="calc.mystery", inputs={"v_in": "v_in"}, unit="V")
    ir.parameters["i_out"] = derived(1.0, tool="calc.ohms_law.I", inputs={"v": "v_out", "r": "r_load"}, unit="A")
    ir.parameters["p"] = derived(1.0, tool="calc.power.P", inputs={"v": "v_in"}, unit="W")
    ir.parameters["q"] = derived(6.0, tool="calc.divider.v_out", derived_from=["v_in", "r1", "r2"], unit="V")  # positional only
    res = recompute_parameters(ir)
    assert res.status is ValidationStatus.NOT_VERIFIED and res.details["mismatches"] == []
    assert [u.split(":")[0] for u in res.details["unverified"]] == ["v_out", "i_out", "p", "q"]
    assert "not a registered calculator" in res.details["parameters"]["v_out"]["reason"]
    assert "['r_load'] not found" in res.details["parameters"]["i_out"]["reason"]
    assert "recorded roles ['v'] are not calc.power.P's roles ['v', 'i']" in res.details["parameters"]["p"]["reason"]
    assert "records no input roles" in res.details["parameters"]["q"]["reason"] and res.details["parameters"]["q"]["positional_recompute"] == 6.0
    # a requirement's traced value is an acceptable input, like the reviewer's calculations_vs_design accepts it
    ir.parameters = {"v_out": derived(6.0, tool="calc.divider.v_out", inputs={"v_in": "v_in", "r1": "r1", "r2": "r2"}, unit="V"), "r1": authoritative(1e4, DS), "r2": authoritative(1e4, DS)}
    ir.requirements.requirements.append(Requirement(id="req.v_in", key="v_in", text="12 V", kind=RequirementKind.EXPLICIT, value=user_requirement(12.0, "V")))
    assert recompute_parameters(ir).status is ValidationStatus.PASS
    ir.parameters = {}
    assert recompute_parameters(ir).status is ValidationStatus.NOT_VERIFIED
    with pytest.raises(ValueError, match="name different ids"):
        derived(1.0, tool="calc.power.P", derived_from=["v", "i"], inputs={"v": "i", "i": "v"})


def test_led_resistor_guards():
    r = led_series_resistor(user_requirement(5.0), authoritative(2.0, DS), authoritative(0.01, DS))
    assert r.value == pytest.approx(300.0)
    with pytest.raises(ValueError):
        led_series_resistor(user_requirement(1.0), authoritative(2.0, DS), authoritative(0.01, DS))


def test_regulator_crystal_lc_and_rc_window_calculators():
    """The four calculators the ATmega128 board adds (added in CALC_VERSION 0.7): values, units, roles and the inputs each refuses."""
    p = regulator_dissipation(user_requirement(9.0, "V"), user_requirement(5.0, "V"), user_requirement(0.05, "A"), ("v_in", "v_out_reg", "i_load_budget"))
    assert p.value == pytest.approx(0.2) and p.unit == "W" and p.provenance.inputs == {"v_in": "v_in", "v_out": "v_out_reg", "i_load": "i_load_budget"}
    assert regulator_dissipation(user_requirement(5.0), user_requirement(5.0), user_requirement(0.05)).value == 0.0  # no headroom, no loss (dropout is not this calculator's business)
    with pytest.raises(ValueError, match="below the output voltage"):
        regulator_dissipation(user_requirement(4.0), user_requirement(5.0), user_requirement(0.05))
    with pytest.raises(ValueError, match="load current must not be negative"):
        regulator_dissipation(user_requirement(9.0), user_requirement(5.0), user_requirement(-0.01))
    c_l = crystal_load_capacitance(user_requirement(22e-12, "F"), user_requirement(22e-12, "F"), user_requirement(4e-12, "F"), ("c_xtal", "c_xtal", "c_stray"))
    assert c_l.value == pytest.approx(15e-12) and c_l.unit == "F" and c_l.provenance.derived_from == ["c_xtal", "c_xtal", "c_stray"]
    assert crystal_load_capacitance(user_requirement(10e-12), user_requirement(30e-12), user_requirement(0.0)).value == pytest.approx(7.5e-12)
    with pytest.raises(ValueError, match="load capacitors must be positive"):
        crystal_load_capacitance(user_requirement(0.0), user_requirement(22e-12), user_requirement(4e-12))
    with pytest.raises(ValueError, match="stray capacitance must not be negative"):
        crystal_load_capacitance(user_requirement(22e-12), user_requirement(22e-12), user_requirement(-1e-12))
    f0 = lc_cutoff(user_requirement(10e-6, "H"), user_requirement(100e-9, "F"))
    assert f0.value == pytest.approx(1.0 / (2 * math.pi * math.sqrt(1e-12))) and f0.unit == "Hz" and f0.provenance.tool == "calc.lc.cutoff"
    with pytest.raises(ValueError, match="inductance must be positive"):
        lc_cutoff(user_requirement(0.0), user_requirement(1e-7))
    with pytest.raises(ValueError, match="capacitance must be positive"):
        lc_cutoff(user_requirement(1e-5), user_requirement(-1e-7))
    with pytest.raises(ValueError, match="underflows"):
        lc_cutoff(user_requirement(1e-200), user_requirement(1e-200))
    tau = rc_time_constant(user_requirement(1e4, "ohm"), user_requirement(1e-7, "F"))
    step, stop = rc_tran_step(tau), rc_tran_stop(tau)
    assert (step.value, stop.value, step.unit, stop.unit) == (pytest.approx(1e-5), pytest.approx(5e-3), "s", "s") and step.provenance.inputs == {"tau": "tau"}
    for fn in (rc_tran_step, rc_tran_stop):
        with pytest.raises(ValueError, match="time constant must be positive"):
            fn(user_requirement(0.0, "s"))
    assert CALC_VERSION == "0.8" and ROLE_UNITS["calc.lc.cutoff"] == ("H", "F") and ROLE_UNITS["calc.regulator.p_dissipation"] == ("V", "V", "A")
