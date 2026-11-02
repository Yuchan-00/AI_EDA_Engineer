"""The level units, ppm, the field strength and the typeset minus in the quantity parser (``QUANTITY_VERSION`` 0.4).

A level is typed as written (``-110 dBm`` is -110 with the unit ``dBm``),
never turned into watts by the parser; dB takes no prefix; the V/m branch
must end its token, so the strings that read as a volt before the field
strength existed (``0.5 V/ms``, ``2 V/mm``) read exactly the same now (their
HEAD reading was measured before the change and is pinned here).
"""

from __future__ import annotations

import pytest

from ai_eda.tools.calc.quantity import (
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


def _read(text: str) -> list[tuple]:
    out = []
    for _, q in find_quantities(text):
        if isinstance(q, QuantityRange):
            out.append((q.low, q.high, q.unit))
        else:
            out.append((q.value, q.unit, q.plus_minus))
    return out


@pytest.mark.parametrize(("text", "value", "unit"), [
    ("-110 dBm", -110.0, "dBm"),
    ("−110 dBm", -110.0, "dBm"),  # U+2212 minus
    ("27 dBm", 27.0, "dBm"),
    ("27dBm", 27.0, "dBm"),
    ("30 DBM", 30.0, "dBm"),  # the dB family is case-insensitive
    ("-3 dBW", -3.0, "dBW"),
    ("3 dB", 3.0, "dB"),
    ("-40 dBc", -40.0, "dBc"),
    ("2.15 dBi", 2.15, "dBi"),
    ("94 dBuV/m", 94.0, "dBuV/m"),
    ("94 dBµV/m", 94.0, "dBuV/m"),  # micro sign
    ("94 dBμV/m", 94.0, "dBuV/m"),  # Greek mu
    ("2.5 ppm", 2.5, "ppm"),
    ("50 mV/m", 0.05, "V/m"),
    ("10 V/m", 10.0, "V/m"),
    ("5 uV/m", 5e-6, "V/m"),
])
def test_a_level_ppm_or_field_strength_is_typed_as_written(text: str, value: float, unit: str) -> None:
    q = parse_answer(text)
    assert isinstance(q, Quantity) and q.unit == unit and q.value == pytest.approx(value, rel=1e-15) and not q.plus_minus
    assert parse_quantity(text) == q


def test_level_ranges_tolerances_particles_and_token_ends() -> None:
    assert _read("-110..-100 dBm") == [(-110.0, -100.0, "dBm")]
    assert parse_answer("-110..-100 dBm") is None  # a range is not one value
    assert _read("±2.5 ppm") == [(2.5, "ppm", True)] and parse_answer("±2.5 ppm") is None  # a tolerance, not a value
    assert _read("출력 20dBm으로") == [(20.0, "dBm", False)]  # a Korean particle may follow
    for text in ("10 dBmW", "3 dBx", "5 kdBm", "2 dBd", "3 dBu", "10 dBV"):
        assert find_quantities(text) == [], text  # the unit must end its token; dB takes no prefix; unsupported levels
    assert parse_answer("10 dBm max") is None and parse_answer("min -110 dBm") is None  # a stated limit is not a value


def test_a_density_or_slope_is_not_a_level() -> None:
    for text in ("-174 dBm/Hz", "-90 dBc/Hz", "20 dB/dec", "50 ppm/°C", "5 ppm/K"):
        assert find_quantities(text) == [], text
    assert _read("3 V/m/s") == [(3.0, "V", False)]  # as before the field-strength unit existed


def test_what_read_as_a_volt_before_still_reads_as_a_volt() -> None:
    """Measured with the HEAD parser (0a2c4c7) before the V/m branch existed; the branch changes none of them."""
    head = {
        "0.5 V/ms": [(0.5, "V", False)],
        "2 V/mm": [(2.0, "V", False)],
        "slew 0.5 V/us": [(0.5, "V", False)],
        "3 V/μs": [(3.0, "V", False)],
        "1 V/m2": [(1.0, "V", False)],
        "5 mV/mA": [(0.005, "V", False)],
        "12V/5V": [(12.0, "V", False), (5.0, "V", False)],
        "5 pm": [(5e-12, "m", False)],  # pico-metre, not ppm
    }
    for text, expected in head.items():
        assert _read(text) == expected, text


def test_the_unit_tables_and_parse_unit_know_the_new_units() -> None:
    assert QUANTITY_VERSION == "0.4"
    assert parse_unit("dBm") == ("dBm", 0) and parse_unit("DBM") == ("dBm", 0) and parse_unit("dB") == ("dB", 0)
    assert parse_unit("dBW") == ("dBW", 0) and parse_unit("dBc") == ("dBc", 0) and parse_unit("dBi") == ("dBi", 0)
    assert parse_unit("dBuV/m") == ("dBuV/m", 0) and parse_unit("ppm") == ("ppm", 0)
    assert parse_unit("mV/m") == ("V/m", -3) and parse_unit("V/m") == ("V/m", 0)
    assert parse_unit("kdBm") is None and parse_unit("mppm") is None
    assert parse_unit("K/W") == ("K/W", 0) and parse_unit("W") == ("W", 0)  # "dBW" ends with W, K/W is not confused with it
    for unit in ("dB", "dBm", "dBW", "dBc", "dBi", "dBuV/m", "ppm", "V/m"):
        assert unit in UNITS, unit
    assert "V/m" in PREFIXABLE_UNITS and not {"dB", "dBm", "ppm"} & PREFIXABLE_UNITS
    assert format_quantity(parse_answer("-110 dBm")) == "-110 dBm"  # type: ignore[arg-type]


def test_the_existing_units_are_untouched_by_the_level_branch() -> None:
    assert _read("R_thJA 62 °C/W max, T_J 150 °C") == [(62.0, "K/W", False), (150.0, "degC", False)]
    assert _read("12V 입력을 5V 2A로") == [(12.0, "V", False), (5.0, "V", False), (2.0, "A", False)]
    assert _read("효율 90% 이상") == [(90.0, "percent", False)]
    assert parse_answer("12 V DC") == Quantity(value=12.0, unit="V", original="12 V DC")


# --------------------------------------------------------------------------- the typeset minus (0.4)


@pytest.mark.parametrize(("text", "expected"), [
    ("sensitivity –120 dBm", [(-120.0, "dBm", False)]),  # en dash U+2013
    ("sensitivity \u2010120 dBm", [(-120.0, "dBm", False)]),  # hyphen U+2010
    ("sensitivity \u2012120 dBm", [(-120.0, "dBm", False)]),  # figure dash U+2012
    ("수신 감도 －120 dBm", [(-120.0, "dBm", False)]),  # full-width hyphen-minus U+FF0D
    ("harmonics –50 dBc", [(-50.0, "dBc", False)]),
    ("temperature –20 °C", [(-20.0, "degC", False)]),  # the older sign loss of the volt / degree readings too
    ("–120 ~ –100 dBm", [(-120.0, -100.0, "dBm")]),  # formerly a single +100 dBm
    ("–40 to 85 °C", [(-40.0, 85.0, "degC")]),
    ("–40 ~ 85 °C", [(-40.0, 85.0, "degC")]),
    ("10–20 dBm", [(10.0, 20.0, "dBm")]),  # an en dash between two numbers is still the range separator
    ("3.3 V – 5 V", [(3.3, 5.0, "V")]),
    ("3.3 V — 5 V", [(3.3, 5.0, "V")]),
    # an unattached dash before a number is a minus, a separator or punctuation: the number is not read at all
    ("sensitivity – 120 dBm", []),
    ("1 kHz–3 dB", [(1000.0, "Hz", False)]),
    ("spurious — 50 dBc", []),
    ("12–5V", []),
    # the ASCII hyphen keeps its documented rule (after a word or number it separates)
    ("12-5V", [(5.0, "V", False)]),
    ("-120 dBm", [(-120.0, "dBm", False)]),
])
def test_a_typeset_minus_is_the_sign_and_never_dropped(text: str, expected: list[tuple]) -> None:
    assert _read(text) == expected, text


def test_parse_quantity_and_parse_answer_read_a_typeset_minus_or_nothing() -> None:
    q = parse_quantity("수신 감도 －120 dBm")
    assert isinstance(q, Quantity) and q.value == -120.0 and q.unit == "dBm"  # formerly +120 dBm
    assert parse_quantity("sensitivity – 120 dBm") is None  # the digits stand outside every quantity read
    a = parse_answer("–117 dBm")
    assert a is not None and a.value == -117.0 and a.unit == "dBm"  # formerly None only because the dash was left over
    assert parse_answer("－117 dBm") is not None and parse_answer("－117 dBm").value == -117.0  # type: ignore[union-attr]
    assert parse_answer("— 117 dBm") is None
