"""Deterministic circuit calculators.

Every function takes ``Traced`` inputs and returns ``Traced`` outputs whose
provenance lists the inputs it was derived from, so the chain
requirement -> calculation -> component value is auditable.
:data:`CALCULATORS` maps each calculator's tool id (the ``Provenance.tool``
it writes) to the callable and the order of its inputs, so
:func:`recompute_parameters` can re-run any derived parameter from the IR
(the CALCULATION stage, ``calc.recompute``).

:mod:`ai_eda.tools.calc.si` parses and formats SPICE numbers with ngspice's
scale-factor semantics (``m`` = milli, ``meg`` = mega).
:mod:`ai_eda.tools.calc.quantity` reads engineering quantities from
requirement text (``M`` = mega, units required, never guessed).
"""

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
from ai_eda.tools.calc.quantity import (
    PREFIX_EXPONENTS,
    PREFIXABLE_UNITS,
    QUANTITY_VERSION,
    UNITS,
    Quantity,
    QuantityRange,
    find_quantities,
    format_quantity,
    parse_answer,
    parse_quantity,
    parse_unit,
)
from ai_eda.tools.calc.recompute import CALCULATORS, derived_values, recompute_parameters
from ai_eda.tools.calc.si import (
    MIL_FACTOR,
    NGSPICE_EXACT_MANTISSA,
    NGSPICE_EXPONENT_RANGE,
    SI_VERSION,
    SPICE_SCALE_EXPONENTS,
    format_spice_number,
    ngspice_reads,
    parse_spice_number,
    spice_value,
)

__all__ = [
    "CALCULATORS",
    "CALC_VERSION",
    "MIL_FACTOR",
    "NGSPICE_EXACT_MANTISSA",
    "NGSPICE_EXPONENT_RANGE",
    "PREFIXABLE_UNITS",
    "PREFIX_EXPONENTS",
    "QUANTITY_VERSION",
    "Quantity",
    "QuantityRange",
    "ROLES",
    "ROLE_UNITS",
    "SI_VERSION",
    "SPICE_SCALE_EXPONENTS",
    "UNITS",
    "current_from_voltage_resistance",
    "derived_values",
    "divider_r1_for_v_out",
    "find_quantities",
    "format_quantity",
    "format_spice_number",
    "junction_temperature",
    "led_current",
    "led_series_resistor",
    "ngspice_reads",
    "parallel_resistance",
    "parse_answer",
    "parse_quantity",
    "parse_spice_number",
    "parse_unit",
    "power_from_voltage_current",
    "power_from_voltage_resistance",
    "rc_ac_fstart",
    "rc_ac_fstop",
    "rc_lowpass_magnitude",
    "rc_lowpass_phase_deg",
    "rc_r_for_cutoff",
    "rc_step_response",
    "rc_time_constant",
    "recompute_parameters",
    "spice_value",
    "voltage_divider_output",
    "voltage_divider_ratio",
]
