"""Readable part values: the ``Component.value`` text a person reads on the schematic, the BOM and the board.

Invariant: a part value is *display text*, never simulation input. The
netlist compiler writes every number from the component's SPICE binding with
:func:`ai_eda.tools.calc.si.format_spice_number` (the spelling ngspice reads
back as exactly that double, ``1e-7`` / ``22.0p`` / ``1.5915494309189537k``);
``Component.value`` is the same number in KiCad's value style -
engineering notation with an SI prefix and no unit letter, at most
:data:`PART_VALUE_DIGITS` significant digits, trailing zeros dropped::

    1e-7 -> 100n     1e-5 -> 10u     2.2e-11 -> 22p     1500 -> 1.5k
    10000 -> 10k     1e6 -> 1M       0.5 -> 500m        6.48172677616823e-08 -> 64.817n

The prefixes are the engineering ones KiCad writes (``M`` is mega, ``m``
milli, ``u`` micro); ngspice's convention (``M`` = milli, ``meg`` = mega)
applies only to the netlist, which never reads this text. A magnitude
outside ``f`` .. ``T`` is written with a plain exponent (``1e-18``).

A part value and the design value therefore differ by up to the display
rounding, and a check that compares them never compares strings:
:func:`parse_part_value` reads the text with the requirement quantity parser
(:func:`ai_eda.tools.calc.quantity.parse_answer`, the IR unit appended - ``M``
is mega there too) and :func:`part_value_agrees` accepts it when it lies
within :func:`display_tolerance` of the design value: half a unit in the
last of :data:`PART_VALUE_DIGITS` significant digits (:data:`AGREEMENT_RULE`
is the sentence a check prints to say so).
"""

from __future__ import annotations

import math
from decimal import Decimal

from ai_eda.tools.calc.quantity import parse_answer

PART_VALUE_VERSION = "0.1"
#: at most this many significant digits in a part value (KiCad's usual value field; 1.5915k, 64.817n)
PART_VALUE_DIGITS = 5
#: what a check that compares a part value with a design value says it did
AGREEMENT_RULE = (
    f"the part value is parsed with the requirement quantity parser (M = mega) and compared with the design value "
    f"within the {PART_VALUE_DIGITS}-significant-digit display rounding"
)

#: engineering exponent -> KiCad value prefix (``u`` for micro, ``M`` for mega)
_PREFIX_FOR_EXPONENT: dict[int, str] = {-15: "f", -12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
#: IR unit -> the symbol appended before parsing (the quantity parser needs a unit next to the number)
_UNIT_SYMBOL: dict[str, str] = {"ohm": "Ω"}


def _number(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot write {value!r} as a part value")
    return value


def format_part_value(value: float) -> str:
    """``value`` as KiCad writes a part value: SI prefix, no unit, at most :data:`PART_VALUE_DIGITS` significant digits, no trailing zeros.

    The number is rounded once to :data:`PART_VALUE_DIGITS` significant
    digits (Python's correctly rounded ``e`` formatting of the exact double)
    and the prefix is chosen from the *rounded* number, so ``999.996`` is
    ``1k``, not ``1000``. ``0`` is ``"0"``; negative numbers keep their sign;
    a magnitude below 1e-15 or from 1e15 up gets a plain exponent
    (``1.2345e-18``). Booleans, non-finite numbers and a number whose
    rounding passes the largest double (``1.7977e308`` would read back as
    ``inf``) are refused.
    """
    value = _number(value)
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    rounded = f"{abs(value):.{PART_VALUE_DIGITS - 1}e}"
    if not math.isfinite(float(rounded)):
        raise ValueError(f"cannot write {value!r} as a part value: {rounded} is past the largest double and would read back as infinity")
    mantissa, _, exp_text = rounded.partition("e")
    digits, exponent = mantissa.replace(".", ""), int(exp_text)  # |rounded| = digits[0].digits[1:] x 10**exponent
    e3 = (exponent // 3) * 3
    prefix = _PREFIX_FOR_EXPONENT.get(e3)
    if prefix is None:
        frac = digits[1:].rstrip("0")
        return f"{sign}{digits[0]}{'.' + frac if frac else ''}e{exponent}"
    point = 1 + exponent - e3  # 1..3 digits before the decimal point, never more than PART_VALUE_DIGITS
    whole, frac = digits[:point], digits[point:].rstrip("0")
    return f"{sign}{whole}{'.' + frac if frac else ''}{prefix}"


def parse_part_value(text: str, unit: str | None) -> float | None:
    """The number the part value ``text`` spells in the IR unit ``unit`` (``ohm``, ``F``, ``H``, ...), or ``None``.

    Read by :func:`~ai_eda.tools.calc.quantity.parse_answer` with the unit's
    symbol appended (``1.5k`` + ``Ω``, ``100n`` + ``F``), so the prefix
    rules are the requirement parser's: ``M`` is mega, ``m`` milli, ``u`` /
    ``µ`` micro, ``K`` kilo. ``None`` when the text is not one plain number
    with an optional prefix in that unit - a part name (``LM7805``), a value
    that carries its own unit (``16MHz``) or words (``10k 1%``) - when the
    number overflows a double (``1e400``: the parser reads it as ``inf``,
    which is no value), or when there is no unit to read it in: nothing is
    compared then.
    """
    if not unit or not isinstance(text, str) or not text or text != text.strip():
        return None
    q = parse_answer(text + _UNIT_SYMBOL.get(unit, unit))
    if q is None or q.unit != unit or not math.isfinite(q.value):
        return None
    return q.value


def display_tolerance(value: float) -> float:
    """Half a unit in the last of :data:`PART_VALUE_DIGITS` significant digits of ``value``: the most the display rounding moves it.

    The digit position comes from the exact decimal exponent of the double
    (``Decimal(value).adjusted()``, no ``log10`` rounding), so ``999.996``
    (shown as ``1k``) has the tolerance of the hundreds decade, 0.005.
    """
    value = _number(value)
    if value == 0:
        return 0.0
    return 0.5 * 10.0 ** (Decimal(abs(value)).adjusted() - (PART_VALUE_DIGITS - 1))


def part_value_agrees(text: str, value: float, unit: str | None) -> bool | None:
    """Whether the part value ``text`` spells the design ``value`` (``unit``) within the display rounding; ``None`` when ``text`` is not a number in ``unit``.

    ``|parse_part_value(text) - value| <= display_tolerance(value)``, with a
    few ULPs of slack for the parse itself (the parser is correctly rounded,
    the tolerance is a power of ten computed in binary). ``None`` means
    nothing was compared, never agreement.
    """
    value = _number(value)
    parsed = parse_part_value(text, unit)
    if parsed is None:
        return None
    slack = 4 * math.ulp(max(abs(parsed), abs(value)))
    return abs(parsed - value) <= display_tolerance(value) * (1 + 1e-12) + slack


__all__ = [
    "AGREEMENT_RULE",
    "PART_VALUE_DIGITS",
    "PART_VALUE_VERSION",
    "display_tolerance",
    "format_part_value",
    "parse_part_value",
    "part_value_agrees",
]
