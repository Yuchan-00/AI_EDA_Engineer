"""SPICE number parsing / formatting (``ai_eda.tools.calc.si``).

The table pins ngspice's scale-factor semantics: ``m``/``M`` is milli,
``meg``/``MEG``/``Meg`` is mega (KiCad's ``1M`` = mega convention is *not*
followed), ``f`` is femto, trailing unit letters are only ignored on request,
and ``format_spice_number`` round-trips through ``parse_spice_number``
exactly for every finite float.
"""

from __future__ import annotations

import math
import random

import pytest

from ai_eda.ir import ProvenanceKind
from ai_eda.tools.calc import (
    MIL_FACTOR,
    NGSPICE_EXACT_MANTISSA,
    SI_VERSION,
    SPICE_SCALE_EXPONENTS,
    format_spice_number,
    ngspice_reads,
    parse_spice_number,
    spice_value,
)

# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10k", 1e4),
        ("10K", 1e4),
        ("4.7u", 4.7e-6),
        ("4.7U", 4.7e-6),
        ("100n", 1e-7),
        ("2.2p", 2.2e-12),
        ("1f", 1e-15),  # femto, not farad
        ("1t", 1e12),
        ("1g", 1e9),
        ("1G", 1e9),
        ("1meg", 1e6),
        ("1MEG", 1e6),
        ("1Meg", 1e6),
        ("2.5meg", 2.5e6),
        ("1m", 1e-3),
        ("1M", 1e-3),  # ngspice: M is milli
        ("1e4", 1e4),
        ("1E4", 1e4),
        ("1e-3", 1e-3),
        ("1.5e3k", 1.5e6),  # exponent and scale factor combine
        ("12", 12.0),
        ("12.", 12.0),
        (".5", 0.5),
        ("0.5", 0.5),
        ("-3.3", -3.3),
        ("+5", 5.0),
        ("-10k", -1e4),
        ("0", 0.0),
    ],
)
def test_parse_table(text: str, expected: float):
    assert parse_spice_number(text) == expected


def test_milli_versus_mega_is_the_ngspice_convention():
    assert parse_spice_number("1m") == parse_spice_number("1M") == 1e-3
    assert parse_spice_number("1meg") == parse_spice_number("1MEG") == parse_spice_number("1Meg") == 1e6
    assert parse_spice_number("1M") != parse_spice_number("1Meg")
    assert SPICE_SCALE_EXPONENTS["m"] == -3 and SPICE_SCALE_EXPONENTS["meg"] == 6


def test_mil_is_a_length_factor():
    assert parse_spice_number("1mil") == pytest.approx(MIL_FACTOR)
    assert parse_spice_number("10MIL") == pytest.approx(10 * 25.4e-6)


def test_scale_factor_is_applied_as_a_decimal_exponent():
    # "4.7u" must be the float nearest to 4.7e-6, not 4.7 * 1e-6 (which differs in the last bit)
    assert parse_spice_number("4.7u") == float("4.7e-6")
    assert parse_spice_number("0.1m") == float("0.1e-3")


@pytest.mark.parametrize("text", ["10kOhm", "4.7uF", "10R", "1megV", "12V", "1kohm"])
def test_trailing_letters_are_rejected_unless_asked(text: str):
    with pytest.raises(ValueError, match="trailing letters"):
        parse_spice_number(text)
    # asked for explicitly: ngspice's behaviour (letters after the scale factor are ignored)
    assert math.isfinite(parse_spice_number(text, ignore_trailing_letters=True))


def test_trailing_letters_table():
    assert parse_spice_number("10kOhm", ignore_trailing_letters=True) == 1e4
    assert parse_spice_number("4.7uF", ignore_trailing_letters=True) == 4.7e-6
    assert parse_spice_number("10R", ignore_trailing_letters=True) == 10.0
    assert parse_spice_number("1mA", ignore_trailing_letters=True) == 1e-3
    assert parse_spice_number("1megV", ignore_trailing_letters=True) == 1e6
    assert parse_spice_number("1F", ignore_trailing_letters=True) == 1e-15  # still femto, the F is the scale factor


@pytest.mark.parametrize(
    "text",
    ["", " ", "4k7", "1k5", "10 k", " 10k", "10k ", "1e+", "abc", "k", "inf", "nan", "1e400", "10k 1%", "1,5", "--1", "0x10", "1_000"],
)
def test_rejected_forms(text: str):
    with pytest.raises(ValueError):
        parse_spice_number(text)
    with pytest.raises(ValueError):
        parse_spice_number(text, ignore_trailing_letters=True)


def test_dangling_exponent_marker_is_a_trailing_letter():
    # ngspice reads "1e" as 1 (the e is not an exponent without digits, and letters after the number are ignored)
    with pytest.raises(ValueError, match="trailing letters"):
        parse_spice_number("1e")
    assert parse_spice_number("1e", ignore_trailing_letters=True) == 1.0


def test_non_string_is_rejected():
    with pytest.raises(ValueError):
        parse_spice_number(10)  # type: ignore[arg-type]


def test_spice_value_is_a_derived_traced_number():
    t = spice_value("4.7u", "F", derived_from=["C1.value"])
    assert t.value == 4.7e-6 and t.unit == "F"
    assert t.provenance.kind == ProvenanceKind.DERIVED
    assert t.provenance.tool == "calc.si.parse" and t.provenance.tool_version == SI_VERSION
    assert t.provenance.derived_from == ["C1.value"]
    assert "'4.7u'" in (t.provenance.note or "")
    assert not t.provenance.needs_verification
    with pytest.raises(ValueError):
        spice_value("10kOhm")
    assert spice_value("10kOhm", "ohm", ignore_trailing_letters=True).value == 1e4


# --------------------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "0"),
        (0, "0"),
        (12.0, "12"),
        (12, "12"),
        (10_000.0, "10k"),
        (1e-5, "1e-5"),  # ngspice reads "10u" as 9.999999999999999e-06, "1e-5" exactly
        (4.7e-6, "4.7u"),
        (1e6, "1meg"),
        (2.5e6, "2.5meg"),
        (1e-3, "1m"),
        (0.5, "500m"),
        (2.2e-12, "2.2p"),
        (1e-15, "1f"),
        (1e12, "1t"),
        (1e9, "1g"),
        (-3300.0, "-3.3k"),
        (123456.789, "123.456789k"),
        (999.0, "999"),
        (1000.0, "1k"),
        (0.1, "100m"),
        (1 / 3, "333.3333333333333m"),
        (1e15, "1e15"),  # outside the suffix range: plain exponent
        (1e20, "1e20"),
        (1.5e-16, "1500e-19"),  # "1.5e-16" is misread (15 * pow(10, -17)); the third spelling is exact
        (5e-324, "5e-324"),
        (2.2e-11, "22.0p"),  # "22p" is misread, one more mantissa digit is not
        (3.3, "3.300000"),  # 33 * pow(10, -1) is 3.3000000000000003
        (1e-4, "1e-4"),
        (1e-7, "1e-7"),
    ],
)
def test_format_table(value: float, expected: str):
    assert format_spice_number(value) == expected


def test_format_never_writes_M_for_mega():
    assert "M" not in format_spice_number(1e6) and format_spice_number(1e6).endswith("meg")
    assert format_spice_number(1e-3).endswith("m")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_format_rejects_non_finite(bad: float):
    with pytest.raises(ValueError):
        format_spice_number(bad)


@pytest.mark.parametrize("bad", [True, False, "10k", None, [1.0]])
def test_format_rejects_non_numbers(bad):
    with pytest.raises(TypeError):
        format_spice_number(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- round trip


_EDGE_CASES = [
    0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1 / 3, 2 / 3, 12.0, 6.0, 10_000.0, 4700.0, 1e-5, 4.7e-6, 100e-9, 22e-12, 1e6, 1.5e-3,
    999.9999999999999, 1000.0000000000001, 1e15, 1e-15, 9.999999999999999e14, 1e-16, 1.7976931348623157e308,
    2.2250738585072014e-308, 5e-324, 123456789.123456789, 3.3, 2.54e-3, 25.4e-6, 1e21, 1e-21,
]


@pytest.mark.parametrize("value", _EDGE_CASES + [-v for v in _EDGE_CASES if v != 0.0])
def test_round_trip_edge_cases(value: float):
    text = format_spice_number(value)
    assert parse_spice_number(text) == value
    assert " " not in text and "\n" not in text


def test_round_trip_random_floats():
    rng = random.Random(20260921)
    for _ in range(5000):
        exponent = rng.uniform(-30, 30)
        value = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 10.0) * 10.0**exponent
        assert parse_spice_number(format_spice_number(value)) == value, value
    for _ in range(2000):
        # short decimal values as they occur in datasheets (4.7, 0.047, 2200, ...)
        mantissa = rng.randint(1, 99999) / 10 ** rng.randint(0, 5)
        value = mantissa * 10.0 ** rng.randint(-14, 14)
        assert parse_spice_number(format_spice_number(value)) == value, value


def test_round_trip_survives_ignore_trailing_letters():
    for value in _EDGE_CASES:
        assert parse_spice_number(format_spice_number(value), ignore_trailing_letters=True) == value


# --------------------------------------------------------------------------- ngspice's own arithmetic


@pytest.mark.parametrize(
    "text, expected",
    [
        # measured on KiCad 10.0.6's ngspice.dll (ngspice-46) through ngSpice_Circ / ngGet_Vec_Info, 2026-09-21
        ("10u", 9.999999999999999e-06),
        ("100n", 1.0000000000000001e-07),
        ("22p", 2.1999999999999998e-11),
        ("3.3", 3.3000000000000003),
        ("1mil", 2.5399999999999997e-05),
        ("1M", 0.001),
        ("1m", 0.001),
        ("1meg", 1000000.0),
        ("1F", 1e-15),
        ("1e", 1.0),
        ("10kOhm", 10000.0),
        ("4.7u", 4.7e-06),
        ("10k", 10000.0),
        ("12", 12.0),
        ("5m", 0.005),
        ("-500m", -0.5),
        ("1e-5", 1e-05),
        ("22.0p", 2.2e-11),
        ("3.300000", 3.3),
        ("6.000000000000001", 6.000000000000002),
    ],
)
def test_ngspice_reads_matches_the_measured_dll(text: str, expected: float):
    assert ngspice_reads(text) == expected


def test_ngspice_reads_declines_outside_the_modelled_region():
    assert ngspice_reads(str(NGSPICE_EXACT_MANTISSA - 1)) == float(NGSPICE_EXACT_MANTISSA - 1)
    assert ngspice_reads(str(NGSPICE_EXACT_MANTISSA)) is None
    assert ngspice_reads("9007199254740991") is None  # 2**53 - 1: the DLL reads this odd value as 2**53
    assert ngspice_reads("1.0000000000000001k") is None  # 17 significant digits
    assert ngspice_reads("1e50") is None and ngspice_reads("1e-45") is None
    with pytest.raises(ValueError):
        ngspice_reads("abc")


def test_format_prefers_the_engineering_form_when_ngspice_reads_it_exactly():
    for value in (10_000.0, 4.7e-6, 1e6, 0.5, 12.0, 1e-9, 5e-3, 1e12, -3300.0, 1 / 3):
        text = format_spice_number(value)
        assert text[-1].isalpha() or "e" not in text, text  # a suffix form, not an exponent form
        assert ngspice_reads(text) == value


def _significant_digits(value: float) -> int:
    mantissa = repr(abs(value)).split("e")[0].replace(".", "")
    return len(mantissa.strip("0"))


def test_format_is_read_back_exactly_by_ngspice_for_short_decimals():
    """Every value with up to 6 significant digits (datasheet values, calculator inputs) has an exact spelling."""
    rng = random.Random(20260921)
    values = [float(f"{rng.randint(1, 999999)}e{rng.randint(-24, 12)}") for _ in range(3000)]
    values += [float(f"{rng.randint(1, 999)}{rng.choice(['', '.5', '.7', '.2', '.05'])}e{rng.randint(-15, 9)}") for _ in range(3000)]
    values += [-v for v in values[:500]]
    misses = [v for v in values if ngspice_reads(format_spice_number(v)) != v]
    assert misses == [], misses[:10]


def test_format_exactness_rate_for_full_precision_doubles():
    """Values with 16-17 significant digits have few equivalent spellings below the modelled bound: about half
    still get an exact one, the rest fall back to the engineering form (one ULP off in ngspice, reported by the
    netlist compiler) - never at the cost of our own round trip."""
    rng = random.Random(7)
    values = [rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 10.0) * 10.0 ** rng.randint(-20, 20) for _ in range(3000)]
    exact = 0
    for v in values:
        text = format_spice_number(v)
        assert parse_spice_number(text) == v
        read = ngspice_reads(text)
        if read == v:
            exact += 1
        elif read is None:  # only a long mantissa can leave the modelled region
            assert _significant_digits(v) >= 16, (v, text)
    assert exact >= 0.45 * len(values), exact


def test_formatted_numbers_are_parseable_by_a_naive_reader_too():
    """Sanity check against an independent evaluation (mantissa * 10**exp) to ~1e-12: no suffix confusion."""
    table = {"": 0, "t": 12, "g": 9, "meg": 6, "k": 3, "m": -3, "u": -6, "n": -9, "p": -12, "f": -15}
    for value in (10_000.0, 4.7e-6, 1e6, 1e-3, 0.5, 2.2e-12, 1e12, -3300.0):
        text = format_spice_number(value)
        for suffix in sorted(table, key=len, reverse=True):
            if text.endswith(suffix):
                mantissa = float(text[: len(text) - len(suffix)] or "0")
                assert math.isclose(mantissa * 10 ** table[suffix], value, rel_tol=1e-12)
                break
