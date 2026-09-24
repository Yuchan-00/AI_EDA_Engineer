"""Engineering-quantity parser (``ai_eda.tools.calc.quantity``).

The table pins every ambiguity rule in the module docstring: ``m`` is milli
only before a unit and the metre alone, ``M`` is mega (not SPICE's milli),
``u``/``µ``/``μ`` are micro, a missing unit is never guessed, Korean
particles may follow a unit, and ranges follow the low-side rules.
"""

from __future__ import annotations

import pytest

from ai_eda.tools.calc import (
    PREFIXABLE_UNITS,
    PREFIX_EXPONENTS,
    QUANTITY_VERSION,
    UNITS,
    Quantity,
    QuantityRange,
    find_quantities,
    format_quantity,
    parse_quantity,
    parse_unit,
)

# --------------------------------------------------------------------------- single quantities


@pytest.mark.parametrize(
    "text, value, unit",
    [
        # required forms
        ("12V", 12.0, "V"),
        ("12 V", 12.0, "V"),
        ("12 volts", 12.0, "V"),
        ("500mA", 0.5, "A"),
        ("2 A", 2.0, "A"),
        ("3.3V", 3.3, "V"),
        ("1.5 kW", 1500.0, "W"),
        ("85°C", 85.0, "degC"),
        ("-20 °C", -20.0, "degC"),
        ("90%", 90.0, "percent"),
        ("10k ohm", 1e4, "ohm"),
        ("10 kΩ", 1e4, "ohm"),
        ("4.7 uF", 4.7e-6, "F"),
        ("100 MHz", 1e8, "Hz"),
        # prefixes
        ("1 THz", 1e12, "Hz"),
        ("2 GHz", 2e9, "Hz"),
        ("1 MΩ", 1e6, "ohm"),
        ("10K ohm", 1e4, "ohm"),  # K is kilo in engineering text
        ("5 cm", 0.05, "m"),
        ("5 mm", 0.005, "m"),
        ("5 µs", 5e-6, "s"),  # micro sign U+00B5
        ("5 μs", 5e-6, "s"),  # Greek mu U+03BC
        ("5 us", 5e-6, "s"),
        ("100 nF", 1e-7, "F"),
        ("22 pF", 2.2e-11, "F"),
        ("1 fF", 1e-15, "F"),
        ("1 mΩ", 1e-3, "ohm"),
        ("4.7 mH", 4.7e-3, "H"),
        ("5 kg", 5000.0, "g"),
        # ambiguity rules
        ("5 mA", 0.005, "A"),  # m before a unit: milli
        ("5 m", 5.0, "m"),  # m alone: metres
        ("5m", 5.0, "m"),
        ("1F", 1.0, "F"),  # farad, not femto
        ("5 g", 5.0, "g"),  # gram, not giga
        ("5 s", 5.0, "s"),
        ("5 H", 5.0, "H"),
        ("12 v", 12.0, "V"),  # lower-case v accepted for the volt
        # unit spellings
        ("10kOhm", 1e4, "ohm"),
        ("10 kohm", 1e4, "ohm"),
        ("10 k Ω", 1e4, "ohm"),
        ("2 amps", 2.0, "A"),
        ("2 amperes", 2.0, "A"),
        ("5 watts", 5.0, "W"),
        ("5 hertz", 5.0, "Hz"),
        ("5 hz", 5.0, "Hz"),
        ("5 sec", 5.0, "s"),
        ("5 seconds", 5.0, "s"),
        ("5 Ω", 5.0, "ohm"),  # U+2126 ohm sign
        ("90 percent", 90.0, "percent"),
        ("90 %", 90.0, "percent"),
        ("85 degC", 85.0, "degC"),
        ("85 deg C", 85.0, "degC"),
        ("85℃", 85.0, "degC"),
        # thermal resistance: K/W and the degC/W spellings datasheets use (matched before the plain degC forms)
        ("62 K/W", 62.0, "K/W"),
        ("62 °C/W", 62.0, "K/W"),
        ("62°C/W", 62.0, "K/W"),
        ("62 ℃/W", 62.0, "K/W"),
        ("62 degC/W", 62.0, "K/W"),
        ("62 deg C/W", 62.0, "K/W"),
        ("62 °C / W", 62.0, "K/W"),
        # AC/DC suffix, numbers
        ("12VDC", 12.0, "V"),
        ("230 V AC", 230.0, "V"),
        ("1,000 V", 1000.0, "V"),
        ("2,000,000 Hz", 2e6, "Hz"),
        (".5V", 0.5, "V"),
        ("+5V", 5.0, "V"),
        ("1e3 V", 1000.0, "V"),
        ("−12 V", -12.0, "V"),  # U+2212 minus
        ("  12 V  ", 12.0, "V"),
        # Korean: SI symbols followed by particles / words, Korean unit words
        ("12V 입력", 12.0, "V"),
        ("5V로", 5.0, "V"),
        ("2A로", 2.0, "A"),
        ("효율 90% 이상", 90.0, "percent"),
        ("12볼트", 12.0, "V"),
        ("12볼트를", 12.0, "V"),
        ("2암페어", 2.0, "A"),
        ("10옴", 10.0, "ohm"),
        ("90퍼센트", 90.0, "percent"),
        ("약12V", 12.0, "V"),
    ],
)
def test_parse_single(text: str, value: float, unit: str) -> None:
    q = parse_quantity(text)
    assert isinstance(q, Quantity), q
    assert q.unit == unit
    assert q.value == value  # correctly rounded: the prefix is applied as a decimal exponent
    assert not q.plus_minus


@pytest.mark.parametrize(
    "text, value, unit",
    [
        ("±5%", 5.0, "percent"),
        ("± 5 %", 5.0, "percent"),
        ("+/-5%", 5.0, "percent"),
        ("±0.1 V", 0.1, "V"),
    ],
)
def test_plus_minus(text: str, value: float, unit: str) -> None:
    q = parse_quantity(text)
    assert isinstance(q, Quantity)
    assert q.plus_minus and q.value == value and q.unit == unit


@pytest.mark.parametrize(
    "text",
    [
        "12",  # bare number: no unit is guessed
        "10k",  # prefix without a unit
        "5 M",
        "5 G",
        "1f",  # femto without a unit (not the farad)
        "5 k",
        "5 c",
        "85C",  # bare C is not degC (could be coulomb)
        "85°F",  # not supported
        "5 h",  # hour not supported
        "5 a",  # lower-case a is not the ampere
        "5 S",  # siemens not supported
        "12 Vin",  # unit must end its token
        "5 Vout",
        "5 mAh",  # mAh / Ah / Wh are not modelled
        "5 Ah",
        "5 Wh",
        "1,5 V",  # decimal comma is not read
        "5V 2A",  # two quantities are not one
        "12-5V",  # low above high is not a range
        "LM7805 at 12V",  # another number in the phrase
        "",
        "   ",
        "volts",
        "V",
        "abc",
        "5 V to 12 V",  # 'to' with a unit on the low side: two quantities
        "3.3V-5A",  # mismatched range units
    ],
)
def test_parse_none(text: str) -> None:
    assert parse_quantity(text) is None


def test_prefix_and_unit_tables_are_explicit() -> None:
    assert PREFIX_EXPONENTS["M"] == 6, "M is mega in engineering text (SPICE's milli is in calc.si)"
    assert PREFIX_EXPONENTS["m"] == -3
    assert PREFIX_EXPONENTS["K"] == PREFIX_EXPONENTS["k"] == 3
    assert PREFIX_EXPONENTS["u"] == PREFIX_EXPONENTS["µ"] == PREFIX_EXPONENTS["μ"] == -6
    assert {"V", "A", "W", "ohm", "F", "H", "Hz", "s", "m", "g", "degC", "K/W", "percent"} <= set(UNITS)
    assert set(UNITS["K/W"]) == {"K/W", "°C/W", "℃/W", "degC/W", "deg C/W"} and "K/W" not in PREFIXABLE_UNITS
    assert "볼트" in UNITS["V"] and "옴" in UNITS["ohm"]


def test_correctly_rounded_prefix() -> None:
    q = parse_quantity("4.7 uF")
    assert isinstance(q, Quantity)
    assert q.value == float("4.7e-6") == float("47e-7")  # one rounding, not 4.7 * 1e-6
    q = parse_quantity("10 uF")
    assert isinstance(q, Quantity)
    assert q.value == 1e-5  # not 9.999999999999999e-06


def test_original_is_the_matched_text() -> None:
    q = parse_quantity("  500mA ")
    assert isinstance(q, Quantity) and q.original == "500mA"
    q = parse_quantity("12V 입력")
    assert isinstance(q, Quantity) and q.original == "12V"


def test_non_string_rejected() -> None:
    with pytest.raises(TypeError):
        parse_quantity(12)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        find_quantities(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- ranges


@pytest.mark.parametrize(
    "text, low, high, unit",
    [
        ("3.3-5V", 3.3, 5.0, "V"),
        ("3.3~5 V", 3.3, 5.0, "V"),
        ("-20..85 °C", -20.0, 85.0, "degC"),
        ("-20...85°C", -20.0, 85.0, "degC"),
        ("-20 … 85°C", -20.0, 85.0, "degC"),
        ("3.3 V – 5 V", 3.3, 5.0, "V"),
        ("3.3 V — 5 V", 3.3, 5.0, "V"),
        ("3.3 to 5 V", 3.3, 5.0, "V"),
        ("-40~+85℃", -40.0, 85.0, "degC"),
        ("1-10 kΩ", 1000.0, 10000.0, "ohm"),  # high side's prefix applies to a bare low side
        ("100-500 mA", 0.1, 0.5, "A"),
        ("500mA-2A", 0.5, 2.0, "A"),  # a fully united low side is parsed on its own
        ("3.3V-5V", 3.3, 5.0, "V"),
        ("0~100%", 0.0, 100.0, "percent"),
        ("-20~85°C에서 동작", -20.0, 85.0, "degC"),
        ("5-5 V", 5.0, 5.0, "V"),
        ("3.3～5V", 3.3, 5.0, "V"),  # full-width tilde
    ],
)
def test_parse_range(text: str, low: float, high: float, unit: str) -> None:
    r = parse_quantity(text)
    assert isinstance(r, QuantityRange), r
    assert (r.low, r.high, r.unit) == (low, high, unit)


# --------------------------------------------------------------------------- find_quantities


def _found(text: str) -> list[tuple[str, object]]:
    return [(text[s:e], q) for (s, e), q in find_quantities(text)]


def test_find_korean_request() -> None:
    text = "12V 입력을 5V 2A로 변환하는 회로, 효율 90% 이상, EU에서 판매"
    hits = find_quantities(text)
    assert [(text[s:e], q.value, q.unit) for (s, e), q in hits] == [  # type: ignore[union-attr]
        ("12V", 12.0, "V"),
        ("5V", 5.0, "V"),
        ("2A", 2.0, "A"),
        ("90%", 90.0, "percent"),
    ]
    for (s, e), q in hits:
        assert text[s:e] == q.original


def test_find_skips_identifiers_and_versions() -> None:
    assert _found("LM7805, CR2032, v1.2, R12, C3") == []
    assert [t for t, _ in _found("LM7805 at 12V and CR2032")] == ["12V"]


def test_find_prefers_ranges_and_falls_back_to_singles() -> None:
    assert [type(q) for _, q in _found("-20~85°C에서 동작, ±5% tolerance")] == [QuantityRange, Quantity]
    assert [t for t, _ in _found("12-5V")] == ["5V"]  # rejected range -> the single quantity
    assert [t for t, _ in _found("5 V to 12 V boost")] == ["5 V", "12 V"]


def test_find_signs_and_separators() -> None:
    assert [(t, q.value) for t, q in _found("3.3V/-5V")] == [("3.3V", 3.3), ("-5V", -5.0)]  # type: ignore[union-attr]
    assert [(t, q.value) for t, q in _found("12V,-5V")] == [("12V", 12.0), ("-5V", -5.0)]  # type: ignore[union-attr]
    assert [t for t, _ in _found("1,000 V and 2,000,000 Hz")] == ["1,000 V", "2,000,000 Hz"]
    assert _found("1,5 V") == []
    assert [t for t, _ in _found("R1-10k ohm")] == ["10k ohm"]
    assert [t for t, _ in _found("end.12V")] == ["12V"]


def test_find_metre_and_volt_in_prose() -> None:
    hits = _found("a 5 m long 12 V cable")
    assert [(t, q.unit) for t, q in hits] == [("5 m", "m"), ("12 V", "V")]  # type: ignore[union-attr]


def test_find_korean_unit_words() -> None:
    assert [(t, q.value, q.unit) for t, q in _found("12볼트를 5볼트로")] == [("12볼트", 12.0, "V"), ("5볼트", 5.0, "V")]  # type: ignore[union-attr]
    assert _found("temp 85C") == []


def test_find_spans_do_not_overlap_and_are_ordered() -> None:
    hits = find_quantities("12 V, 500 mA, 10 kΩ, 4.7 uF, 100 MHz, 85 °C, 90 %")
    ends = [e for (_, e), _ in hits]
    starts = [s for (s, _), _ in hits]
    assert len(hits) == 7 and all(starts[i + 1] >= ends[i] for i in range(len(hits) - 1))


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("kΩ", ("ohm", 3)),
        ("m", ("m", 0)),
        ("mA", ("A", -3)),
        ("MHz", ("Hz", 6)),
        ("uF", ("F", -6)),
        ("%", ("percent", 0)),
        ("°C", ("degC", 0)),
        ("°C/W", ("K/W", 0)),
        ("K/W", ("K/W", 0)),
        ("K", None),
        ("volts", ("V", 0)),
        ("k", None),
        ("C", None),
        ("", None),
        ("Vin", None),
    ],
)
def test_parse_unit(text: str, expected: tuple[str, int] | None) -> None:
    assert parse_unit(text) == expected


def test_format_quantity() -> None:
    assert format_quantity(parse_quantity("500mA")) == "0.5 A"  # type: ignore[arg-type]
    assert format_quantity(parse_quantity("±5%")) == "±5 percent"  # type: ignore[arg-type]
    assert format_quantity(parse_quantity("-20..85 °C")) == "-20..85 degC"  # type: ignore[arg-type]
    assert format_quantity(parse_quantity("100 MHz")) == "100000000 Hz"  # type: ignore[arg-type]


def test_models_are_frozen() -> None:
    q = parse_quantity("12V")
    assert isinstance(q, Quantity)
    with pytest.raises(Exception):
        q.value = 5.0  # type: ignore[misc]


def test_a_thermal_resistance_is_never_read_as_a_temperature() -> None:
    """Regression: ``62 °C/W`` used to parse as 62 degC (``/`` may follow a unit, so ``°C`` matched and ``/W`` was dropped);
    a t_ datasheet key could then have been grounded from a thermal-resistance quote. The K/W forms now win."""
    assert [(q.value, q.unit) for _, q in find_quantities("R_thJA 62 °C/W max, T_J 150 °C")] == [(62.0, "K/W"), (150.0, "degC")]
    assert parse_quantity("62 °C/W") == Quantity(value=62.0, unit="K/W", original="62 °C/W")
    assert format_quantity(parse_quantity("62 °C/W")) == "62 K/W"  # type: ignore[arg-type]
    assert parse_quantity("5 K/Wh") is None and parse_quantity("62 k/W") is None  # the kelvin is capital; no prefix, no other unit
    assert QUANTITY_VERSION == "0.2"  # the unit table changed: cached extractions re-ground (the fingerprint carries this)
