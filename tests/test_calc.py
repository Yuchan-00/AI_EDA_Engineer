import pytest

from ai_eda.ir import ProvenanceKind, SourceRef, authoritative, user_requirement
from ai_eda.tools.calc import (
    current_from_voltage_resistance,
    led_series_resistor,
    power_from_voltage_current,
    rc_time_constant,
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


def test_ohm_and_power():
    i = current_from_voltage_resistance(user_requirement(5.0), authoritative(250.0, DS))
    assert i.value == pytest.approx(0.02)
    p = power_from_voltage_current(user_requirement(5.0), i)
    assert p.value == pytest.approx(0.1)
    assert p.unit == "W"


def test_rc():
    assert rc_time_constant(authoritative(1e3, DS), authoritative(1e-6, DS)).value == pytest.approx(1e-3)


def test_led_resistor_guards():
    r = led_series_resistor(user_requirement(5.0), authoritative(2.0, DS), authoritative(0.01, DS))
    assert r.value == pytest.approx(300.0)
    with pytest.raises(ValueError):
        led_series_resistor(user_requirement(1.0), authoritative(2.0, DS), authoritative(0.01, DS))
