from __future__ import annotations

from ai_eda.ir.provenance import Traced, derived

CALC_VERSION = "0.1"


def _key(t: Traced, fallback: str) -> str:
    """Traced values do not carry their own key; callers pass ids via ``derived_from``."""
    return fallback


def current_from_voltage_resistance(v: Traced[float], r: Traced[float], ids: tuple[str, str] = ("v", "r")) -> Traced[float]:
    if r.value == 0:
        raise ZeroDivisionError("resistance is zero")
    return derived(v.value / r.value, tool="calc.ohms_law.I", derived_from=list(ids), unit="A", tool_version=CALC_VERSION, note="I = V / R")


def power_from_voltage_current(v: Traced[float], i: Traced[float], ids: tuple[str, str] = ("v", "i")) -> Traced[float]:
    return derived(v.value * i.value, tool="calc.power.P", derived_from=list(ids), unit="W", tool_version=CALC_VERSION, note="P = V * I")


def voltage_divider_ratio(r1: Traced[float], r2: Traced[float], ids: tuple[str, str] = ("r1", "r2")) -> Traced[float]:
    total = r1.value + r2.value
    if total == 0:
        raise ZeroDivisionError("r1 + r2 is zero")
    return derived(r2.value / total, tool="calc.divider.ratio", derived_from=list(ids), unit=None, tool_version=CALC_VERSION, note="ratio = R2 / (R1 + R2)")


def voltage_divider_output(v_in: Traced[float], r1: Traced[float], r2: Traced[float], ids: tuple[str, str, str] = ("v_in", "r1", "r2")) -> Traced[float]:
    ratio = voltage_divider_ratio(r1, r2, ids[1:])
    return derived(v_in.value * ratio.value, tool="calc.divider.v_out", derived_from=list(ids), unit="V", tool_version=CALC_VERSION, note="V_out = V_in * R2 / (R1 + R2)")


def rc_time_constant(r: Traced[float], c: Traced[float], ids: tuple[str, str] = ("r", "c")) -> Traced[float]:
    return derived(r.value * c.value, tool="calc.rc.tau", derived_from=list(ids), unit="s", tool_version=CALC_VERSION, note="tau = R * C")


def led_series_resistor(v_supply: Traced[float], v_forward: Traced[float], i_forward: Traced[float], ids: tuple[str, str, str] = ("v_supply", "v_f", "i_f")) -> Traced[float]:
    if i_forward.value <= 0:
        raise ValueError("forward current must be positive")
    if v_supply.value <= v_forward.value:
        raise ValueError("supply voltage must exceed LED forward voltage")
    return derived((v_supply.value - v_forward.value) / i_forward.value, tool="calc.led.R", derived_from=list(ids), unit="ohm", tool_version=CALC_VERSION, note="R = (V_supply - V_f) / I_f")
