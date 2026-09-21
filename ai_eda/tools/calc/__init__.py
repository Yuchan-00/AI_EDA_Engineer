"""Deterministic circuit calculators.

Every function takes ``Traced`` inputs and returns ``Traced`` outputs whose
provenance lists the inputs it was derived from, so the chain
requirement -> calculation -> component value is auditable.
"""

from ai_eda.tools.calc.basic import (
    CALC_VERSION,
    current_from_voltage_resistance,
    led_series_resistor,
    power_from_voltage_current,
    rc_time_constant,
    voltage_divider_output,
    voltage_divider_ratio,
)

__all__ = [
    "CALC_VERSION",
    "current_from_voltage_resistance",
    "led_series_resistor",
    "power_from_voltage_current",
    "rc_time_constant",
    "voltage_divider_output",
    "voltage_divider_ratio",
]
