"""Deterministic engineering-quantity parser for requirement text.

Invariant: a number found in a requirement becomes a value only when its
unit is written next to it, and the same text always yields the same value.
Nothing here guesses a missing unit and no LLM is involved, so the grounding
step (:mod:`ai_eda.llm.extraction`) can check a model's claim ("12 V") against
the user's own words with the same parser every time.

This is *not* :mod:`ai_eda.tools.calc.si`. That module reads SPICE numbers
with ngspice's rules (``M`` = milli, ``meg`` = mega, no units). This one reads
what people write in requirements, datasheets and chat.

Prefixes (case-sensitive, applied as a decimal exponent so the result is
correctly rounded: ``4.7 uF`` is ``float("47e-7")``)::

    T = 1e12   G = 1e9   M = 1e6   k / K = 1e3   c = 1e-2   m = 1e-3
    u / µ (U+00B5) / μ (U+03BC) = 1e-6   n = 1e-9   p = 1e-12   f = 1e-15

Units, canonical symbol <- accepted spellings (:data:`UNITS`): ``V`` (V, v,
volt(s), 볼트), ``A`` (A, amp(s), ampere(s), 암페어), ``W`` (W, watt(s), 와트),
``ohm`` (Ω U+03A9, Ω U+2126, ohm(s), 옴), ``F`` (F, farad(s), 패럿), ``H`` (H,
henry, henries, 헨리), ``Hz`` (Hz, hz, hertz, 헤르츠), ``s`` (s, sec,
second(s)), ``m`` (m, meter(s), metre(s), 미터), ``g`` (g, gram(s), 그램),
``degC`` (°C, ℃, degC, deg C - no prefix), ``K/W`` (K/W, °C/W, ℃/W, degC/W,
deg C/W - a thermal resistance; no prefix, and matched before the degC forms
so ``62 °C/W`` is 62 K/W, never 62 degC), ``percent`` (%, percent, 퍼센트 -
no prefix, value kept as written: ``90%`` is 90, not 0.9).

Ambiguity rules (each one is pinned by ``tests/test_quantity.py``):

* ``m`` after a number is *milli* only when a unit follows (``5 mA``,
  ``5 mm``); alone it is the metre (``5 m``, ``5m``).
* ``M`` is *mega* (``100 MHz``, ``1 MΩ``) - engineering text, unlike SPICE.
* ``K`` is kilo like ``k`` (``10K ohm``); the kelvin is not a unit here.
* ``g`` alone is the gram, ``G`` is giga; ``F`` is the farad, ``f`` is femto;
  ``H`` is the henry (``h``, the hour, is not supported); ``s`` is the second
  (``S``, the siemens, is not supported); lower-case ``v`` is accepted for the
  volt because it has no other engineering meaning - the other single-letter
  symbols are case-sensitive.
* A prefix with no unit (``10k``, ``5 M``, ``1f``) and a bare number are
  *not* quantities: the unit is missing and is not guessed -> ``None``.
* ``°C`` / ``℃`` / ``degC`` / ``deg C`` are degrees Celsius; a bare ``C`` is
  not (it could be the coulomb) and ``°F`` is not supported -> ``None``.
* A unit must end its token: ``5 mAh``, ``12 Vin`` and ``5 Vout`` are not
  quantities (``mAh``/``Ah``/``Wh`` are not modelled). Korean particles may
  follow directly (``5V로``, ``2A로``, ``12볼트를``, ``90%이상``), and ``DC`` /
  ``AC`` may follow a volt or ampere unit (``12VDC``, ``230 V AC``) and are
  dropped from the unit.
* A prefix may be separated from its unit by whitespace (``10k ohm``,
  ``10 k Ω``), and the unit from the number (``12 V``).
* ``±`` or ``+/-`` before a number marks a tolerance: :attr:`Quantity.plus_minus`
  is ``True`` and the value is the magnitude (``±5%`` -> 5 percent).
* Ranges (:class:`QuantityRange`): ``3.3-5V``, ``3.3~5 V``, ``-20..85 °C``,
  ``-20 … 85°C``, ``3.3 V – 5 V``, ``3.3 to 5 V``. The low side either has no
  unit of its own - then the high side's prefix *and* unit apply to it
  (``1-10 kΩ`` is 1 kΩ..10 kΩ, ``100-500 mA`` is 0.1..0.5 A) - or a full unit
  of the same kind (``500mA-2A``). Mismatched units, a low side above the
  high side (``12-5V`` is a step-down, not a range) and the word ``to`` with a
  unit on the low side (``5 V to 12 V`` is two quantities, e.g. a converter)
  are not ranges; :func:`find_quantities` then reports the single quantities
  it can see.
* Thousands separators are read (``1,000 V``); a decimal comma is not
  (``1,5 V`` yields nothing, because ``5`` right after ``1,`` is not read
  as a new number).
* :func:`find_quantities` starts a number only where one can start: not
  inside a word or identifier (``LM7805``, ``CR2032``, ``v1.2`` yield
  nothing), but after Korean text (``약12V``), punctuation and whitespace.
* :func:`parse_quantity` answers only for an unambiguous phrase: exactly one
  quantity and no other digit anywhere else (``12V 입력`` -> 12 V;
  ``5V 2A``, ``12-5V`` and ``LM7805 at 12V`` -> ``None``).
* :func:`parse_answer` is stricter still, for a value that becomes a design
  input or a review target: the whole stripped text must be that one
  quantity (not a range, not a ``±`` tolerance), with nothing beside it but
  the AC / DC words (``12 V DC``, ``DC 12 V``, ``12 V (DC)``, ``직류 12 V``).
  A qualifier is not read away: ``12 V max``, ``min 5 V``, ``12 V rms``,
  ``12V 입력`` and ``not more than 12 V`` -> ``None`` - a stated limit is not
  a nominal value.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

QUANTITY_VERSION = "0.2"

#: SI prefix letter -> decimal exponent (engineering text: ``M`` is mega)
PREFIX_EXPONENTS: dict[str, int] = {
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "K": 3,
    "c": -2,
    "m": -3,
    "u": -6,
    "µ": -6,  # micro sign
    "μ": -6,  # Greek small mu
    "n": -9,
    "p": -12,
    "f": -15,
}

#: single-letter / symbol spellings that take a prefix; matched case-sensitively
_SYMBOLS: dict[str, str] = {
    "Hz": "Hz",
    "V": "V",
    "v": "V",
    "A": "A",
    "W": "W",
    "F": "F",
    "H": "H",
    "s": "s",
    "m": "m",
    "g": "g",
    "Ω": "ohm",  # Greek capital omega
    "Ω": "ohm",  # ohm sign
}
#: unit words that take a prefix; matched case-insensitively
_WORDS: dict[str, str] = {
    "volt": "V", "volts": "V",
    "amp": "A", "amps": "A", "ampere": "A", "amperes": "A",
    "watt": "W", "watts": "W",
    "ohm": "ohm", "ohms": "ohm",
    "farad": "F", "farads": "F",
    "henry": "H", "henries": "H",
    "hertz": "Hz", "hz": "Hz",
    "second": "s", "seconds": "s", "sec": "s",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "gram": "g", "grams": "g",
}
#: Korean unit words (no prefix)
_KOREAN: dict[str, str] = {
    "볼트": "V",
    "암페어": "A",
    "와트": "W",
    "옴": "ohm",
    "패럿": "F",
    "헨리": "H",
    "헤르츠": "Hz",
    "미터": "m",
    "그램": "g",
    "퍼센트": "percent",
}
#: units that never take a prefix; the regex below spells their forms (the thermal resistance first: its
#: spellings start with a degC spelling, and the longer match must win)
_PLAIN: dict[str, str] = {"%": "percent", "percent": "percent", "K/W": "K/W", "degc": "degC", "℃": "degC"}

#: canonical unit -> every accepted spelling (documentation and tests)
UNITS: dict[str, tuple[str, ...]] = {}
for _table in (_SYMBOLS, _WORDS, _KOREAN, _PLAIN):
    for _spelling, _canon in _table.items():
        UNITS.setdefault(_canon, ())
        UNITS[_canon] = (*UNITS[_canon], _spelling)
UNITS["degC"] = (*UNITS["degC"], "°C", "deg C")
UNITS["K/W"] = (*UNITS["K/W"], "°C/W", "℃/W", "degC/W", "deg C/W")
#: units that may carry a prefix
PREFIXABLE_UNITS: frozenset[str] = frozenset({"V", "A", "W", "ohm", "F", "H", "Hz", "s", "m", "g"})


class Quantity(BaseModel):
    """One value with its canonical unit, prefix applied (``500mA`` -> 0.5 A)."""

    model_config = ConfigDict(frozen=True)

    value: float
    unit: str
    original: str
    #: written with ``±`` / ``+/-``: a tolerance, ``value`` is the magnitude
    plus_minus: bool = False


class QuantityRange(BaseModel):
    """``low..high`` in one canonical unit (``-20..85 °C``), ``low <= high``."""

    model_config = ConfigDict(frozen=True)

    low: float
    high: float
    unit: str
    original: str


def _alternation(spellings: dict[str, str]) -> str:
    return "|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True))


_SIGN = r"[+\-−]?"
_UNSIGNED = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?|\.\d+(?:[eE][+\-]?\d+)?"
#: what may not follow a unit (the unit must end its token; Hangul may follow)
_END = r"(?![A-Za-z0-9_µμΩΩ°℃])"
_PREFIX_CLASS = "[" + "".join(PREFIX_EXPONENTS) + "]"
_VA = "|".join(re.escape(s) for s in ("V", "v", "A"))
_SYM = _alternation({s: c for s, c in _SYMBOLS.items() if s not in ("V", "v", "A")})
_WORD = _alternation(_WORDS)
_KWORD = _alternation(_KOREAN)
#: the K/W forms stand before the degC forms: an alternation takes the first branch that matches, and
#: ``°C`` alone would match inside ``°C/W`` (``/`` may follow a unit) and read a thermal resistance as a temperature
_DEGC_RE = r"°\s?[Cc]|℃|(?i:deg\s?c)"
_PLAIN_RE = rf"%|(?i:percent)|K/W|(?:{_DEGC_RE})\s?/\s?W|{_DEGC_RE}"


def _num(tag: str) -> str:
    return rf"(?P<num{tag}>{_SIGN}(?:{_UNSIGNED}))"


def _unit(tag: str) -> str:
    return (
        rf"(?:(?P<pfx{tag}>{_PREFIX_CLASS})?\s*"
        rf"(?:(?P<va{tag}>{_VA})(?:\s*(?i:dc|ac))?|(?P<sym{tag}>{_SYM})|(?P<word{tag}>(?i:{_WORD})))"
        rf"|(?P<kword{tag}>{_KWORD})|(?P<plain{tag}>{_PLAIN_RE})){_END}"
    )


_SEP = r"(?P<sep>~|～|〜|\.{2,3}|…|–|—|−|-|(?i:to))"
_SINGLE_RE = re.compile(rf"(?P<pm>±|\+/-)?\s*{_num('')}\s*{_unit('')}")
_RANGE_RE = re.compile(rf"{_num('lo')}\s*(?:{_unit('lo')})?\s*{_SEP}\s*{_num('hi')}\s*{_unit('hi')}")
_UNIT_ONLY_RE = re.compile(_unit(""))
_START_RE = re.compile(r"±|\+/-|[+\-−]?(?:\d|\.\d)")
_DECIMAL_RE = re.compile(r"([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?")


def _value(numtext: str, exp10: int) -> float:
    """``numtext`` times ``10**exp10`` as the nearest double (exact decimal, single rounding)."""
    s = numtext.replace(",", "").replace("−", "-")
    m = _DECIMAL_RE.fullmatch(s)
    if m is None:  # pragma: no cover - the outer regex only admits these spellings
        raise ValueError(f"not a number: {numtext!r}")
    sign, int_part, frac, exp = m.group(1), m.group(2) or "", m.group(3) or "", int(m.group(4) or 0)
    digits = (int_part + frac).lstrip("0") or "0"
    return float(f"{sign}{digits}e{exp - len(frac) + exp10}")


def _unit_of(m: re.Match[str], tag: str) -> tuple[str, int]:
    """``(canonical unit, prefix exponent)`` for the unit groups tagged ``tag`` in a match."""
    prefix = m.group(f"pfx{tag}")
    exp10 = PREFIX_EXPONENTS[prefix] if prefix else 0
    if m.group(f"va{tag}") is not None:
        return _SYMBOLS[m.group(f"va{tag}")], exp10
    if m.group(f"sym{tag}") is not None:
        return _SYMBOLS[m.group(f"sym{tag}")], exp10
    if m.group(f"word{tag}") is not None:
        return _WORDS[m.group(f"word{tag}").lower()], exp10
    if m.group(f"kword{tag}") is not None:
        return _KOREAN[m.group(f"kword{tag}")], 0
    plain = m.group(f"plain{tag}")
    if plain == "%" or plain.lower() == "percent":
        return "percent", 0
    if plain.endswith("W"):
        return "K/W", 0
    return "degC", 0


def _has_unit(m: re.Match[str], tag: str) -> bool:
    return any(m.group(f"{g}{tag}") is not None for g in ("va", "sym", "word", "kword", "plain"))


def _build_single(m: re.Match[str]) -> Quantity:
    unit, exp10 = _unit_of(m, "")
    return Quantity(value=_value(m.group("num"), exp10), unit=unit, original=m.group(0), plus_minus=m.group("pm") is not None)


def _build_range(m: re.Match[str]) -> QuantityRange | None:
    """The range a ``_RANGE_RE`` match spells, or ``None`` when the rules say it is not one."""
    hi_unit, hi_exp = _unit_of(m, "hi")
    if _has_unit(m, "lo"):
        if m.group("sep").lower() == "to":
            return None  # "5 V to 12 V" is two quantities
        lo_unit, lo_exp = _unit_of(m, "lo")
        if lo_unit != hi_unit:
            return None
    else:
        lo_exp = hi_exp
    low, high = _value(m.group("numlo"), lo_exp), _value(m.group("numhi"), hi_exp)
    if low > high:
        return None
    return QuantityRange(low=low, high=high, unit=hi_unit, original=m.group(0))


def parse_quantity(text: str) -> Quantity | QuantityRange | None:
    """The one quantity (or one range) ``text`` states, or ``None``.

    ``text`` is usually a token (``"500mA"``, ``"-20..85 °C"``) but may be a
    short phrase whose only number is that quantity (``"12V 입력"``,
    ``"효율 90% 이상"``). ``None`` when there is no quantity (a bare number, a
    prefix without a unit, an unknown unit), when there are two or more
    (``"5V 2A"``), when another digit appears outside the quantity
    (``"12-5V"``, ``"LM7805 at 12V"`` - the phrase is not unambiguous) or
    when the only candidate is a range the rules reject (module docstring).
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    hits = find_quantities(text)
    if len(hits) != 1:
        return None
    (start, end), found = hits[0]
    rest = text[:start] + text[end:]
    if any(ch.isdigit() for ch in rest):
        return None
    return found


#: what may stand beside the one quantity of an answer: the AC / DC words (``DC 12 V``, ``12 V (DC)``, ``직류 12 V``),
#: the same spellings :mod:`ai_eda.regulatory.applicability` reads next to a voltage; anything else is a qualifier
_ACDC_WORDS_RE = re.compile(r"^\s*(?:\(?\s*(?:AC|DC|ac|dc|Ac|Dc|교류|직류)\s*\)?\s*)*$")


def parse_answer(text: str) -> Quantity | None:
    """The one quantity ``text`` states *as a whole*, or ``None``.

    Unlike :func:`parse_quantity` the quantity must be the entire stripped
    text - only the AC / DC words may stand beside it - and it must be a
    single value (no range, no ``±`` tolerance). ``'12 V'``, ``'12 V DC'``,
    ``'DC 12 V'``, ``'12 V (DC)'`` and ``'10 mA'`` parse; ``'12 V max'``,
    ``'min 5 V'``, ``'12 V rms'``, ``'12V 입력'``, ``'3.3~5V'``, ``'5V 2A'``,
    ``'±5%'`` and ``'10k'`` do not. Used wherever a typed answer becomes a
    number the design rests on (:mod:`ai_eda.design.inputs`) or a value a
    verdict is compared with (the reviewer), so the two cannot drift.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    hits = find_quantities(text)
    if len(hits) != 1:
        return None
    (start, end), found = hits[0]
    if not isinstance(found, Quantity) or found.plus_minus:
        return None
    if _ACDC_WORDS_RE.fullmatch(text[:start] + " " + text[end:]) is None:
        return None
    return found


def parse_unit(text: str) -> tuple[str, int] | None:
    """``(canonical unit, prefix exponent)`` for a unit spelling on its own (``"kΩ"`` -> ``("ohm", 3)``,
    ``"m"`` -> ``("m", 0)``, ``"mA"`` -> ``("A", -3)``); ``None`` for a prefix alone or an unknown unit."""
    m = _UNIT_ONLY_RE.fullmatch(text.strip())
    return None if m is None else _unit_of(m, "")


def _start_ok(text: str, i: int) -> bool:
    """A number may start at ``i`` unless it continues a word, identifier or decimal (``LM7805``, ``v1.2``, ``1,5``)."""
    if i == 0:
        return True
    prev = text[i - 1]
    if prev.isascii() and (prev.isalnum() or prev == "_"):
        return False
    if prev in ".," and i >= 2 and text[i - 2].isdigit():
        return False
    return True


def find_quantities(text: str) -> list[tuple[tuple[int, int], Quantity | QuantityRange]]:
    """Every quantity / range in a sentence with its ``(start, end)`` span, left to right, non-overlapping.

    Ranges are preferred over the single quantities they contain; a rejected
    range falls back to the single quantities (``12-5V`` -> ``5 V``).
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    out: list[tuple[tuple[int, int], Quantity | QuantityRange]] = []
    i = 0
    while i < len(text):
        m = _START_RE.search(text, i)
        if m is None:
            break
        start = m.start()
        if not _start_ok(text, start):
            i = start + 1
            continue
        found: Quantity | QuantityRange | None = None
        end = start
        rm = _RANGE_RE.match(text, start)
        if rm is not None:
            found = _build_range(rm)
            end = rm.end()
        if found is None:
            sm = _SINGLE_RE.match(text, start)
            if sm is not None:
                found = _build_single(sm)
                end = sm.end()
        if found is None:
            i = start + 1
            continue
        out.append(((start, end), found))
        i = end
    return out


def format_quantity(q: Quantity | QuantityRange) -> str:
    """Short text for notes: ``0.5 A``, ``±5 percent``, ``-20..85 degC``."""
    if isinstance(q, QuantityRange):
        return f"{q.low:.12g}..{q.high:.12g} {q.unit}"
    return f"{'±' if q.plus_minus else ''}{q.value:.12g} {q.unit}"


__all__ = [
    "PREFIXABLE_UNITS",
    "PREFIX_EXPONENTS",
    "QUANTITY_VERSION",
    "UNITS",
    "Quantity",
    "QuantityRange",
    "find_quantities",
    "format_quantity",
    "parse_answer",
    "parse_quantity",
    "parse_unit",
]
