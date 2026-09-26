"""Deterministic calculators.

Invariant: every calculator records *which id filled which role* in the
provenance it writes (``Provenance.inputs``: role -> id, in the calculator's
own parameter order, plus the same ids as ``derived_from``), so
:func:`ai_eda.tools.calc.recompute.recompute_parameters` can rebuild the call
by role instead of trusting the positional order of a list. A caller that
passes the wrong number of ids gets a ``ValueError`` here, not a provenance
that cannot be checked later; so does a result that is not a finite number
(``'... overflows'``), since a ``Traced`` never holds one. The roles and the unit each role expects are
published in :data:`ROLES` / :data:`ROLE_UNITS` for the recompute registry.
"""

from __future__ import annotations

import math

from ai_eda.ir.provenance import Traced, derived

CALC_VERSION = "0.6"

#: tool id -> input roles, in the calculator's parameter order
ROLES: dict[str, tuple[str, ...]] = {
    "calc.ohms_law.I": ("v", "r"),
    "calc.power.P": ("v", "i"),
    "calc.power.P_VR": ("v", "r"),
    "calc.thermal.T_j": ("t_a", "p", "theta_ja"),
    "calc.divider.ratio": ("r1", "r2"),
    "calc.divider.v_out": ("v_in", "r1", "r2"),
    "calc.rc.tau": ("r", "c"),
    "calc.rc.step_response": ("v_step", "t", "tau"),
    "calc.rc.lowpass_magnitude": ("f", "tau"),
    "calc.rc.lowpass_phase_deg": ("f", "tau"),
    "calc.parallel.R": ("r_a", "r_b"),
    "calc.led.R": ("v_supply", "v_f", "i_f"),
    "calc.divider.r1_for_v_out": ("v_in", "v_out", "r2"),
    "calc.led.I": ("v_supply", "v_f", "r"),
    "calc.rc.r_for_cutoff": ("f_c", "c"),
    "calc.rc.ac_fstart": ("f_c",),
    "calc.rc.ac_fstop": ("f_c",),
    "calc.astable.c_for_frequency": ("f_osc", "r_b", "v_cc", "v_be"),
    "calc.astable.f": ("r_b", "c", "v_cc", "v_be"),
    "calc.astable.tran_step": ("f_osc",),
    "calc.astable.tran_stop": ("f_osc",),
    "calc.astable.tran_start": ("f_osc",),
    "calc.astable.v_be_reverse": ("v_cc", "v_be"),
}
#: tool id -> the unit each role's input is expected to carry (``None``: any); an input whose unit is set
#: and differs is refused by the recompute (a swapped voltage / resistance would otherwise be computed)
ROLE_UNITS: dict[str, tuple[str | None, ...]] = {
    "calc.ohms_law.I": ("V", "ohm"),
    "calc.power.P": ("V", "A"),
    "calc.power.P_VR": ("V", "ohm"),
    "calc.thermal.T_j": ("degC", "W", "K/W"),
    "calc.divider.ratio": ("ohm", "ohm"),
    "calc.divider.v_out": ("V", "ohm", "ohm"),
    "calc.rc.tau": ("ohm", "F"),
    "calc.rc.step_response": ("V", "s", "s"),
    "calc.rc.lowpass_magnitude": ("Hz", "s"),
    "calc.rc.lowpass_phase_deg": ("Hz", "s"),
    "calc.parallel.R": ("ohm", "ohm"),
    "calc.led.R": ("V", "V", "A"),
    "calc.divider.r1_for_v_out": ("V", "V", "ohm"),
    "calc.led.I": ("V", "V", "ohm"),
    "calc.rc.r_for_cutoff": ("Hz", "F"),
    "calc.rc.ac_fstart": ("Hz",),
    "calc.rc.ac_fstop": ("Hz",),
    "calc.astable.c_for_frequency": ("Hz", "ohm", "V", "V"),
    "calc.astable.f": ("ohm", "F", "V", "V"),
    "calc.astable.tran_step": ("Hz",),
    "calc.astable.tran_stop": ("Hz",),
    "calc.astable.tran_start": ("Hz",),
    "calc.astable.v_be_reverse": ("V", "V"),
}


def _derived(value: float, tool: str, ids: tuple[str, ...], unit: str | None, note: str) -> Traced[float]:
    roles = ROLES[tool]
    if len(ids) != len(roles):
        raise ValueError(f"{tool} takes {len(roles)} input ids {list(roles)}, got {len(ids)}: {list(ids)}")
    if not math.isfinite(value):
        # a Traced holds no inf / nan (the IR cannot carry one): say so in one sentence a note can quote
        raise ValueError(f"{tool} overflows: {note} is not a finite number for these inputs")
    return derived(value, tool=tool, inputs=dict(zip(roles, ids)), unit=unit, tool_version=CALC_VERSION, note=note)


def current_from_voltage_resistance(v: Traced[float], r: Traced[float], ids: tuple[str, str] = ("v", "r")) -> Traced[float]:
    if r.value == 0:
        raise ZeroDivisionError("resistance is zero")
    return _derived(v.value / r.value, "calc.ohms_law.I", ids, "A", "I = V / R")


def power_from_voltage_current(v: Traced[float], i: Traced[float], ids: tuple[str, str] = ("v", "i")) -> Traced[float]:
    return _derived(v.value * i.value, "calc.power.P", ids, "W", "P = V * I")


def power_from_voltage_resistance(v: Traced[float], r: Traced[float], ids: tuple[str, str] = ("v", "r")) -> Traced[float]:
    """Dissipation of a resistance ``r`` with the voltage ``v`` across it: P = V^2 / R."""
    if r.value == 0:
        raise ZeroDivisionError("resistance is zero")
    return _derived(v.value * v.value / r.value, "calc.power.P_VR", ids, "W", "P = V^2 / R")


def junction_temperature(t_a: Traced[float], p: Traced[float], theta_ja: Traced[float], ids: tuple[str, str, str] = ("t_a", "p", "theta_ja")) -> Traced[float]:
    """Steady-state junction temperature at ambient ``t_a`` (degC) with dissipation ``p`` (W) through ``theta_ja`` (K/W): Tj = Ta + P * theta_ja."""
    if p.value < 0:
        raise ValueError("dissipation must not be negative")
    if theta_ja.value < 0:
        raise ValueError("thermal resistance must not be negative")
    return _derived(t_a.value + p.value * theta_ja.value, "calc.thermal.T_j", ids, "degC", "Tj = Ta + P * theta_ja")


def voltage_divider_ratio(r1: Traced[float], r2: Traced[float], ids: tuple[str, str] = ("r1", "r2")) -> Traced[float]:
    total = r1.value + r2.value
    if total == 0:
        raise ZeroDivisionError("r1 + r2 is zero")
    return _derived(r2.value / total, "calc.divider.ratio", ids, None, "ratio = R2 / (R1 + R2)")


def voltage_divider_output(v_in: Traced[float], r1: Traced[float], r2: Traced[float], ids: tuple[str, str, str] = ("v_in", "r1", "r2")) -> Traced[float]:
    ratio = voltage_divider_ratio(r1, r2, ids[1:])
    return _derived(v_in.value * ratio.value, "calc.divider.v_out", ids, "V", "V_out = V_in * R2 / (R1 + R2)")


def rc_time_constant(r: Traced[float], c: Traced[float], ids: tuple[str, str] = ("r", "c")) -> Traced[float]:
    return _derived(r.value * c.value, "calc.rc.tau", ids, "s", "tau = R * C")


def rc_step_response(v_step: Traced[float], t: Traced[float], tau: Traced[float], ids: tuple[str, str, str] = ("v_step", "t", "tau")) -> Traced[float]:
    """Capacitor voltage of a series RC driven by an ideal step of ``v_step`` at ``t`` after the step (initially discharged)."""
    if tau.value <= 0:
        raise ValueError("time constant must be positive")
    if t.value < 0:
        raise ValueError("time must not be negative")
    return _derived(v_step.value * (1.0 - math.exp(-t.value / tau.value)), "calc.rc.step_response", ids, "V", "v(t) = V_step * (1 - exp(-t / tau))")


def rc_lowpass_magnitude(f: Traced[float], tau: Traced[float], ids: tuple[str, str] = ("f", "tau")) -> Traced[float]:
    """|H(f)| of a first-order RC low-pass with time constant ``tau``: 1 / sqrt(1 + (2 pi f tau)^2)."""
    if tau.value <= 0:
        raise ValueError("time constant must be positive")
    if f.value < 0:
        raise ValueError("frequency must not be negative")
    w = 2.0 * math.pi * f.value * tau.value
    return _derived(1.0 / math.sqrt(1.0 + w * w), "calc.rc.lowpass_magnitude", ids, None, "|H| = 1 / sqrt(1 + (2 pi f tau)^2)")


def rc_lowpass_phase_deg(f: Traced[float], tau: Traced[float], ids: tuple[str, str] = ("f", "tau")) -> Traced[float]:
    """Phase of a first-order RC low-pass in degrees: -atan(2 pi f tau)."""
    if tau.value <= 0:
        raise ValueError("time constant must be positive")
    if f.value < 0:
        raise ValueError("frequency must not be negative")
    return _derived(-math.degrees(math.atan(2.0 * math.pi * f.value * tau.value)), "calc.rc.lowpass_phase_deg", ids, "deg", "phase = -atan(2 pi f tau)")


def parallel_resistance(r_a: Traced[float], r_b: Traced[float], ids: tuple[str, str] = ("r_a", "r_b")) -> Traced[float]:
    total = r_a.value + r_b.value
    if total == 0:
        raise ZeroDivisionError("r_a + r_b is zero")
    return _derived(r_a.value * r_b.value / total, "calc.parallel.R", ids, "ohm", "R = Ra * Rb / (Ra + Rb)")


def led_series_resistor(v_supply: Traced[float], v_forward: Traced[float], i_forward: Traced[float], ids: tuple[str, str, str] = ("v_supply", "v_f", "i_f")) -> Traced[float]:
    if i_forward.value <= 0:
        raise ValueError("forward current must be positive")
    if v_supply.value <= v_forward.value:
        raise ValueError("supply voltage must exceed LED forward voltage")
    return _derived((v_supply.value - v_forward.value) / i_forward.value, "calc.led.R", ids, "ohm", "R = (V_supply - V_f) / I_f")


def divider_r1_for_v_out(v_in: Traced[float], v_out: Traced[float], r2: Traced[float], ids: tuple[str, str, str] = ("v_in", "v_out", "r2")) -> Traced[float]:
    """The upper resistor that makes an unloaded divider with lower resistor ``r2`` output ``v_out`` from ``v_in``: R1 = R2 (V_in - V_out) / V_out."""
    if r2.value <= 0:
        raise ValueError("r2 must be positive")
    if v_out.value <= 0 or v_out.value >= v_in.value:
        raise ValueError("a divider needs 0 < v_out < v_in")
    return _derived(r2.value * (v_in.value - v_out.value) / v_out.value, "calc.divider.r1_for_v_out", ids, "ohm", "R1 = R2 * (V_in - V_out) / V_out")


def led_current(v_supply: Traced[float], v_forward: Traced[float], r: Traced[float], ids: tuple[str, str, str] = ("v_supply", "v_f", "r")) -> Traced[float]:
    """Current through a series resistor ``r`` feeding an ideal constant-drop LED: I = (V_supply - V_f) / R."""
    if r.value <= 0:
        raise ValueError("resistance must be positive")
    if v_supply.value <= v_forward.value:
        raise ValueError("supply voltage must exceed LED forward voltage")
    return _derived((v_supply.value - v_forward.value) / r.value, "calc.led.I", ids, "A", "I = (V_supply - V_f) / R (ideal constant-V_f LED)")


def rc_r_for_cutoff(f_c: Traced[float], c: Traced[float], ids: tuple[str, str] = ("f_c", "c")) -> Traced[float]:
    """The resistor that gives a first-order RC low-pass with capacitor ``c`` the cutoff (|H| = 1/sqrt 2) ``f_c``: R = 1 / (2 pi f_c C)."""
    if f_c.value <= 0:
        raise ValueError("cutoff frequency must be positive")
    if c.value <= 0:
        raise ValueError("capacitance must be positive")
    denominator = 2.0 * math.pi * f_c.value * c.value
    if denominator == 0:
        raise ValueError("calc.rc.r_for_cutoff underflows: 2 pi f_c C is zero in float arithmetic for these inputs")
    return _derived(1.0 / denominator, "calc.rc.r_for_cutoff", ids, "ohm", "R = 1 / (2 pi f_c C)")


def rc_ac_fstart(f_c: Traced[float], ids: tuple[str] = ("f_c",)) -> Traced[float]:
    """Start of an ac sweep around the corner ``f_c``: two decades below it."""
    if f_c.value <= 0:
        raise ValueError("cutoff frequency must be positive")
    return _derived(f_c.value / 100.0, "calc.rc.ac_fstart", ids, "Hz", "f_start = f_c / 100 (two decades below the corner)")


def rc_ac_fstop(f_c: Traced[float], ids: tuple[str] = ("f_c",)) -> Traced[float]:
    """End of an ac sweep around the corner ``f_c``: two decades above it."""
    if f_c.value <= 0:
        raise ValueError("cutoff frequency must be positive")
    return _derived(f_c.value * 100.0, "calc.rc.ac_fstop", ids, "Hz", "f_stop = f_c * 100 (two decades above the corner)")


# --------------------------------------------------------------------------- BJT astable multivibrator
#
# The period of a symmetric collector-coupled astable (two equal base resistors R_b and timing capacitors C, supply
# V_cc) is 2 * R_b * C * ln((2 V_cc - V_BE) / (V_cc - V_BE)): each capacitor charges through R_b from about -(V_cc - V_BE)
# towards V_cc until the base reaches V_BE. The textbook 1.386 R C is the V_BE = 0 limit of the same expression.

#: periods of the oscillation the transient saves (the window ends at tran_stop = TRAN_STOP_PERIODS / f)
TRAN_STOP_PERIODS = 20.0
#: periods skipped before the window is saved (tran_start = TRAN_START_PERIODS / f: the start-up settles first)
TRAN_START_PERIODS = 10.0
#: time steps per period (tran_step = 1 / (TRAN_STEPS_PER_PERIOD * f))
TRAN_STEPS_PER_PERIOD = 200.0


def _astable_log_term(v_cc: Traced[float], v_be: Traced[float]) -> float:
    """ln((2 V_cc - V_BE) / (V_cc - V_BE)); refuses the inputs for which the expression has no meaning."""
    if v_be.value < 0:
        raise ValueError("base-emitter voltage must not be negative")
    if v_cc.value <= v_be.value:
        raise ValueError("supply voltage must exceed the base-emitter voltage (the period's log argument must be > 1)")
    return math.log((2.0 * v_cc.value - v_be.value) / (v_cc.value - v_be.value))


def astable_c_for_frequency(f_osc: Traced[float], r_b: Traced[float], v_cc: Traced[float], v_be: Traced[float], ids: tuple[str, str, str, str] = ("f_osc", "r_b", "v_cc", "v_be")) -> Traced[float]:
    """The timing capacitor that makes a symmetric BJT astable with base resistors ``r_b`` oscillate at ``f_osc``: C = 1 / (2 f R_b ln((2 V_cc - V_BE) / (V_cc - V_BE)))."""
    if f_osc.value <= 0:
        raise ValueError("oscillation frequency must be positive")
    if r_b.value <= 0:
        raise ValueError("base resistance must be positive")
    denominator = 2.0 * f_osc.value * r_b.value * _astable_log_term(v_cc, v_be)
    if denominator == 0:
        raise ValueError("calc.astable.c_for_frequency underflows: 2 f R_b ln(...) is zero in float arithmetic for these inputs")
    return _derived(1.0 / denominator, "calc.astable.c_for_frequency", ids, "F", "C = 1 / (2 f R_b ln((2 V_cc - V_BE) / (V_cc - V_BE)))")


def astable_frequency(r_b: Traced[float], c: Traced[float], v_cc: Traced[float], v_be: Traced[float], ids: tuple[str, str, str, str] = ("r_b", "c", "v_cc", "v_be")) -> Traced[float]:
    """The oscillation frequency of a symmetric BJT astable: f = 1 / (2 R_b C ln((2 V_cc - V_BE) / (V_cc - V_BE)))."""
    if r_b.value <= 0:
        raise ValueError("base resistance must be positive")
    if c.value <= 0:
        raise ValueError("capacitance must be positive")
    denominator = 2.0 * r_b.value * c.value * _astable_log_term(v_cc, v_be)
    if denominator == 0:
        raise ValueError("calc.astable.f underflows: 2 R_b C ln(...) is zero in float arithmetic for these inputs")
    return _derived(1.0 / denominator, "calc.astable.f", ids, "Hz", "f = 1 / (2 R_b C ln((2 V_cc - V_BE) / (V_cc - V_BE)))")


def astable_tran_step(f_osc: Traced[float], ids: tuple[str] = ("f_osc",)) -> Traced[float]:
    """Transient time step for an oscillation at ``f_osc``: 1 / (200 f), 200 points per period."""
    if f_osc.value <= 0:
        raise ValueError("oscillation frequency must be positive")
    return _derived(1.0 / (TRAN_STEPS_PER_PERIOD * f_osc.value), "calc.astable.tran_step", ids, "s", "tran_step = 1 / (200 f) (200 points per period)")


def astable_tran_stop(f_osc: Traced[float], ids: tuple[str] = ("f_osc",)) -> Traced[float]:
    """End of the transient for an oscillation at ``f_osc``: 20 / f (20 periods)."""
    if f_osc.value <= 0:
        raise ValueError("oscillation frequency must be positive")
    return _derived(TRAN_STOP_PERIODS / f_osc.value, "calc.astable.tran_stop", ids, "s", "tran_stop = 20 / f (20 periods)")


def astable_tran_start(f_osc: Traced[float], ids: tuple[str] = ("f_osc",)) -> Traced[float]:
    """Start of the saved transient window for an oscillation at ``f_osc``: 10 / f (the first 10 periods settle the start-up)."""
    if f_osc.value <= 0:
        raise ValueError("oscillation frequency must be positive")
    return _derived(TRAN_START_PERIODS / f_osc.value, "calc.astable.tran_start", ids, "s", "tran_start = 10 / f (the first 10 periods are start-up)")


def astable_v_be_reverse(v_cc: Traced[float], v_be: Traced[float], ids: tuple[str, str] = ("v_cc", "v_be")) -> Traced[float]:
    """The reverse voltage each base-emitter junction of the astable sees once per period: V_cc - V_BE (the other transistor's base swings that far negative when its capacitor's collector side switches)."""
    if v_be.value < 0:
        raise ValueError("base-emitter voltage must not be negative")
    if v_cc.value <= v_be.value:
        raise ValueError("supply voltage must exceed the base-emitter voltage")
    return _derived(v_cc.value - v_be.value, "calc.astable.v_be_reverse", ids, "V", "V_BE_reverse = V_cc - V_BE")
