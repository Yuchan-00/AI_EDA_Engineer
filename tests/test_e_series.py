"""IEC 60063 preferred values: the tables, the three picking modes, the registered calculators and their recompute."""

from __future__ import annotations

import math

import pytest

from ai_eda.ir import ProvenanceKind, user_requirement
from ai_eda.ir.validation import ValidationStatus
from ai_eda.tools.calc import CALC_VERSION, CALCULATORS, ROLE_UNITS, ROLES, format_part_value, recompute_parameters
from ai_eda.tools.calc.e_series import (
    E_SERIES_CALCULATORS,
    MODES,
    SERIES,
    SERIES_TOLERANCE,
    at_least,
    at_most,
    is_standard,
    nearest,
    series_of,
    standard_value,
    tool_id,
)


def test_the_tables_are_the_iec_60063_series():
    assert [len(SERIES[s]) for s in ("e6", "e12", "e24", "e96")] == [6, 12, 24, 96] and set(SERIES) == set(SERIES_TOLERANCE)
    # E96 is the rounded geometric series; E24 and below are the historical values (3.3, 4.7 ... are not the rounded roots)
    assert list(SERIES["e96"]) == [round(100 * 10 ** (i / 96)) for i in range(96)]
    assert set(SERIES["e6"]) < set(SERIES["e12"]) < set(SERIES["e24"])
    for values in SERIES.values():
        assert list(values) == sorted(set(values))


def test_nearest_is_by_ratio_and_exact_at_every_decade():
    assert nearest(1591.549430918953, "e24") == 1600.0 and nearest(1591.549430918953, "e96") == 1580.0
    assert nearest(6.48172677616823e-08, "e12") == 6.8e-08 and nearest(1.1254e-05, "e12") == 1.2e-05
    assert nearest(9.5, "e12") == 10.0  # 8.2 is 1.3 away and 10 only 0.5; by ratio 10 wins as well, and 9.5 in E24 goes to 9.1
    assert nearest(9.5, "e24") == 9.1
    assert nearest(math.sqrt(10.0 * 11.0), "e24") == 10.0  # a tie by ratio goes to the lower value
    # the value is the double its decimal spelling parses to, so the part text and the number agree
    for value, text in ((4.7e-08, "47n"), (1.5e3, "1.5k"), (2.2e-11, "22p"), (9.76e5, "976k"), (1.0e-6, "1u"), (5.1e-3, "5.1m")):
        series = "e96" if text == "976k" else "e24"
        assert nearest(value * 1.001, series) == value and format_part_value(value) == text


def test_one_sided_modes_and_round_off():
    assert at_most(860.0, "e24") == 820.0 and at_least(860.0, "e24") == 910.0
    assert at_most(250.0, "e96") == 249.0 and at_least(250.0, "e96") == 255.0
    # a solved number that is a series value up to float round-off stays that value in every mode
    noisy = 75000.0 * (1.0 + 3e-13)
    assert at_most(noisy, "e24") == at_least(noisy, "e24") == nearest(noisy, "e24") == 75000.0
    assert at_most(75000.0 * (1 - 3e-13), "e24") == 75000.0
    for bad in (0.0, -1.0, math.inf, math.nan, True, "1k"):
        with pytest.raises(ValueError, match="positive finite number"):
            nearest(bad, "e24")
    with pytest.raises(ValueError, match="unknown E series"):
        nearest(1.0, "e48")


def test_is_standard_and_series_of():
    assert is_standard(14000.0, "e96") and not is_standard(14000.0, "e24")
    assert is_standard(300.0, "e24") and not is_standard(300.0, "e96")
    assert not is_standard(1591.549430918953, "e96") and not is_standard(0.0, "e24") and not is_standard(math.nan, "e24")
    assert series_of(1e-7) == "e6" and series_of(1.2e3) == "e12" and series_of(1.6e3) == "e24" and series_of(1.58e3) == "e96"
    assert series_of(1591.5) is None and series_of(1.58e3, ("e6", "e12", "e24")) is None


def test_the_calculators_are_registered_and_keep_the_unit():
    assert set(E_SERIES_CALCULATORS) == {tool_id(s, m) for s in SERIES for m in MODES} and len(E_SERIES_CALCULATORS) == 12
    for tool, fn in E_SERIES_CALCULATORS.items():
        assert CALCULATORS[tool] == (fn, ("x",)) and ROLES[tool] == ("x",) and ROLE_UNITS[tool] == (None,)
    out = E_SERIES_CALCULATORS["calc.e_series.e24.nearest"](user_requirement(1591.549430918953, "ohm"), ("r_ideal",))
    assert (out.value, out.unit) == (1600.0, "ohm") and out.provenance.kind is ProvenanceKind.DERIVED
    assert out.provenance.tool == "calc.e_series.e24.nearest" and out.provenance.inputs == {"x": "r_ideal"} and out.provenance.tool_version == CALC_VERSION
    assert E_SERIES_CALCULATORS["calc.e_series.e12.nearest"](user_requirement(6.48e-8, "F"), ("c_ideal",)).unit == "F"
    with pytest.raises(ValueError, match="no E series calculator"):
        tool_id("e24", "closest")


def test_standard_value_walks_the_ladder():
    ideal = user_requirement(1591.549430918953, "ohm")
    assert standard_value(ideal, "r_ideal", ("e24", "e96")).value == 1600.0  # no judge: the first series
    def within(limit):
        return lambda t: abs(t.value / ideal.value - 1.0) <= limit

    loose = standard_value(ideal, "r_ideal", ("e24", "e96"), accept=within(0.006))
    assert loose.value == 1600.0 and loose.provenance.tool == "calc.e_series.e24.nearest"  # 1.6k is 0.53 % off: E24 is enough at 0.6 %
    missed = standard_value(ideal, "r_ideal", ("e24", "e96"), accept=within(0.004))
    assert missed.value == 1580.0 and missed.provenance.tool == "calc.e_series.e96.nearest"  # 1.58k is 0.73 % off: no series fits, the finest is returned
    odd = user_requirement(1400.5, "ohm")
    finer = standard_value(odd, "r_ideal", ("e24", "e96"), accept=lambda t: abs(t.value / odd.value - 1.0) <= 0.01)
    assert finer.value == 1400.0 and finer.provenance.tool == "calc.e_series.e96.nearest"  # E24 offers 1.3k / 1.5k (7 % off), E96 1.40k
    assert standard_value(ideal, "r_ideal", ("e24",), mode="at_least").value == 1600.0
    assert standard_value(ideal, "r_ideal", ("e24",), mode="at_most").value == 1500.0
    with pytest.raises(ValueError, match="at least one series"):
        standard_value(ideal, "r_ideal", ())


def test_recompute_proves_a_placed_value_is_the_series_value_of_its_ideal(divider_ir):
    ir = divider_ir
    ir.parameters["r_ideal"] = user_requirement(1591.549430918953, "ohm")
    ir.parameters["r"] = standard_value(ir.parameters["r_ideal"], "r_ideal", ("e24",))
    ok = recompute_parameters(ir)
    assert ok.status is ValidationStatus.PASS and ok.details["parameters"]["r"]["recomputed"] == 1600.0
    ir.parameters["r"] = ir.parameters["r"].model_copy(update={"value": 1591.5})  # the display rounding of the solved number is not a part
    bad = recompute_parameters(ir)
    assert bad.status is ValidationStatus.FAIL and "calc.e_series.e24.nearest recomputes 1600.0" in bad.message
