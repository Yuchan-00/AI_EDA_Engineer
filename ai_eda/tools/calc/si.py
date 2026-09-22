"""SPICE number parsing and formatting (SI scale suffixes with **ngspice semantics**).

Invariant: a number the netlist compiler writes is read back by ngspice as
the IR value, and a value string this module reads is interpreted exactly the
way ngspice would interpret it - never the "everyday" or KiCad convention.

ngspice scale factors (``INPevaluate``), matched case-insensitively::

    T = 1e12   G = 1e9   MEG = 1e6   K = 1e3
    M = 1e-3   (MILLI - not mega!)   MIL = 25.4e-6 (a thousandth of an inch)
    U = 1e-6   N = 1e-9   P = 1e-12   F = 1e-15 (FEMTO - "1F" is not one farad)

So ``"1M"``, ``"1m"`` and ``"1 mA"`` are one *milli*, while ``"1meg"``,
``"1MEG"`` and ``"1Meg"`` are one *mega*. KiCad's schematic value normaliser
uses the opposite convention for ``M`` (it rewrites ``1M`` to ``1Meg`` on
export); this module deliberately does **not** and callers must not feed it
un-normalised KiCad value fields expecting SI. Letters after the scale
factor (``10kOhm``, ``4.7uF``) are ignored by ngspice; :func:`parse_spice_number`
rejects them unless ``ignore_trailing_letters=True`` is passed, because in
our own data a trailing unit is a sign of a value that was never parsed.
Forms ngspice does not read as one number (``4k7``, ``10 k`` with a space,
``1e+``) are rejected in both modes; a dangling exponent marker (``1e``) is
just a trailing letter to ngspice (the value is 1), so it is rejected only in
the strict mode. ``a`` (atto) is not a scale factor here: ngspice
versions differ on it, so a value that needs it is written with an exponent.

:func:`parse_spice_number` is *correctly rounded*: the scale factor is
applied as a decimal exponent (``4.7u`` becomes ``float("4.7e-6")``), so the
result is the double nearest to the decimal value, independent of
floating-point multiplication order. ngspice itself is not: measured on the
ngspice-46 ``ngspice.dll`` bundled with KiCad 10.0.6, ``INPevaluate``
accumulates the digits into a double ``M`` and returns ``M * pow(10, E)``,
which is one ULP off for ``10u`` (9.999999999999999e-06), ``100n``, ``22p``,
``3.3`` (3.3000000000000003) and about a quarter of random doubles.
:func:`ngspice_reads` reproduces that arithmetic (validated bit-for-bit in
the lab on 13,975 spellings plus 9,000 random 16-digit mantissas below
:data:`NGSPICE_EXACT_MANTISSA` with exponents within +-40 - that run is not
in the repository; what is, is the measurement harness
``tests/ngspice_parser_harness.py`` and the DLL-gated test that re-measures
a fresh 500-spelling sample of the same families on the installed DLL, plus
20 pinned values. At and above that bound - odd mantissas within 47 of
``2**53``, and everything with 17 or more significant digits - the DLL
follows neither a plain nor a fused multiply-add model, so it answers
``None`` instead of guessing).

:func:`format_spice_number` therefore writes engineering notation with these
suffixes (``10000.0`` -> ``"10k"``, ``4.7e-6`` -> ``"4.7u"``, ``1e6`` ->
``"1meg"``) *when ngspice reads that spelling back exactly*, and otherwise
tries the equivalent spellings with fewer or more digits (``1e-5`` ->
``"1e-5"`` because ``10u`` is misread, ``2.2e-11`` -> ``"22.0p"``, ``3.3`` ->
``"3.300000"``) and returns the first one ngspice reads exactly, falling back
to the engineering form (nearest, one ULP off in ngspice) when no spelling
below ``2**53`` digits is exact. Every candidate is an exact decimal spelling
of the value, so ``parse_spice_number(format_spice_number(x)) == x`` holds
for every finite float ``x``; ``ngspice_reads(format_spice_number(x)) == x``
holds whenever an exact spelling exists (measured: every value with up to 6
significant digits, about half of random full-precision doubles - the
netlist compiler reports the rest as ``inexact_numbers``).
"""

from __future__ import annotations

import math
import re

from ai_eda.ir.provenance import Traced, derived

SI_VERSION = "0.2"

#: scale suffix (lower case) -> decimal exponent; ``mil`` is handled separately
SPICE_SCALE_EXPONENTS: dict[str, int] = {
    "t": 12,
    "g": 9,
    "meg": 6,
    "k": 3,
    "m": -3,
    "u": -6,
    "n": -9,
    "p": -12,
    "f": -15,
}

#: one mil in metres: ngspice multiplies by 25.4e-6
MIL_FACTOR = 25.4e-6

#: ngspice accumulates a number's digits into a double; below this bound the accumulation was measured
#: exact and its arithmetic is reproduced by :func:`ngspice_reads`. The DLL misreads odd mantissas
#: within 47 of 2**53 (measured on offsets 1..1999) and everything from 2**53 up follows no simple
#: model, so the bound keeps a margin of 1024 below 2**53.
NGSPICE_EXACT_MANTISSA = 2**53 - 1024
#: decimal exponents for which ngspice's ``pow(10, e)`` was verified to equal Python's ``10.0 ** e``
NGSPICE_EXPONENT_RANGE = (-40, 40)
#: how many extra trailing zeros :func:`format_spice_number` is willing to try
_MAX_EXTRA_ZEROS = 8

#: decimal exponent -> suffix used by :func:`format_spice_number`
_SUFFIX_FOR_EXPONENT: dict[int, str] = {
    12: "t",
    9: "g",
    6: "meg",
    3: "k",
    0: "",
    -3: "m",
    -6: "u",
    -9: "n",
    -12: "p",
    -15: "f",
}

# ``meg`` and ``mil`` are tried before the single-letter ``m`` (ngspice checks the two
# following characters before deciding the scale is milli).
_NUMBER = re.compile(
    r"""
    ^
    (?P<sign>[+-]?)
    (?P<mant>\d+\.?\d*|\.\d+)
    (?:[eE](?P<exp>[+-]?\d+))?
    (?P<scale>meg|mil|[tgkmunpf])?
    (?P<trail>[a-zA-Z]*)
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

_REPR = re.compile(r"^(?P<int>\d+)(?:\.(?P<frac>\d+))?(?:e(?P<exp>[+-]?\d+))?$")


def parse_spice_number(text: str, *, ignore_trailing_letters: bool = False) -> float:
    """Parse ``text`` as ngspice would (see the module docstring for the suffix table), correctly rounded.

    Raises ``ValueError`` for anything ngspice would not read as one number
    (empty, whitespace inside, ``4k7``, missing exponent digits, ``inf``) and,
    unless ``ignore_trailing_letters`` is set, for letters after the scale
    factor (``"10kOhm"``). No whitespace is tolerated at either end: callers
    that read tokens have already split on whitespace.
    """
    m = _match(text)
    trail = m.group("trail")
    if trail and not ignore_trailing_letters:
        raise ValueError(f"trailing letters {trail!r} after the scale factor in {text!r} (ngspice would ignore them; pass ignore_trailing_letters=True to do the same)")
    exponent = int(m.group("exp") or 0)
    scale = (m.group("scale") or "").lower()
    mant = m.group("mant")
    if scale == "mil":
        return _finite(float(f"{m.group('sign')}{mant}e{exponent}") * MIL_FACTOR, text)
    exponent += SPICE_SCALE_EXPONENTS.get(scale, 0)
    return _finite(float(f"{m.group('sign')}{mant}e{exponent}"), text)


def _match(text: str) -> re.Match[str]:
    if not isinstance(text, str):
        raise ValueError(f"SPICE number must be a string, got {type(text).__name__}")
    m = _NUMBER.match(text)
    if m is None:
        raise ValueError(f"not a SPICE number: {text!r}")
    return m


def _finite(value: float, text: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"SPICE number {text!r} is out of range")
    return value


def ngspice_reads(text: str) -> float | None:
    """The double ngspice-46 (KiCad 10's ``ngspice.dll``) produces for ``text``, or ``None`` outside the modelled region.

    ``INPevaluate`` is not a correctly rounded parser: the digits are
    accumulated into a double ``M`` (exact below ``2**53``), the fraction
    length, exponent and scale factor are summed into ``E``, and the result is
    ``M * pow(10, E)`` (``mil``: ``M * 25.4 * pow(10, E - 6)``) - so ``10u`` is
    9.999999999999999e-06 and ``3.3`` is 3.3000000000000003. Reproduced
    bit-for-bit against the DLL for ``M < NGSPICE_EXACT_MANTISSA`` and ``E``
    within :data:`NGSPICE_EXPONENT_RANGE`; ``None`` beyond that (odd
    mantissas just below ``2**53`` and 17-digit mantissas came back from the
    DLL matching neither a plain nor a fused multiply-add model, so they are
    not guessed). Trailing letters are ignored as ngspice ignores them; a
    non-number raises ``ValueError``.
    """
    m = _match(text)
    int_part, _, frac = m.group("mant").partition(".")
    mantissa = int((int_part + frac) or "0")
    if mantissa >= NGSPICE_EXACT_MANTISSA:
        return None
    exponent = int(m.group("exp") or 0) - len(frac)
    scale = (m.group("scale") or "").lower()
    value = float(mantissa)
    if scale == "mil":
        value *= 25.4
        exponent -= 6
    else:
        exponent += SPICE_SCALE_EXPONENTS.get(scale, 0)
    if not NGSPICE_EXPONENT_RANGE[0] <= exponent <= NGSPICE_EXPONENT_RANGE[1]:
        return None
    value *= 10.0**exponent
    return -value if m.group("sign") == "-" else value


def spice_value(text: str, unit: str | None = None, derived_from: list[str] | None = None, *, ignore_trailing_letters: bool = False) -> Traced[float]:
    """:func:`parse_spice_number` wrapped as a ``derived`` traced value.

    The provenance records the tool (``calc.si.parse``), its version and the
    original text, so a value that entered the IR as ``"4.7u"`` can be audited
    back to that string.
    """
    value = parse_spice_number(text, ignore_trailing_letters=ignore_trailing_letters)
    return derived(
        value,
        tool="calc.si.parse",
        derived_from=list(derived_from or []),
        unit=unit,
        tool_version=SI_VERSION,
        note=f"parsed {text!r} with ngspice scale-factor semantics (m = milli, meg = mega)",
    )


def _decimal_digits(value: float) -> tuple[str, int]:
    """``value == int(digits) * 10**exponent`` with ``digits`` free of leading and trailing zeros.

    Built from ``repr`` (the shortest string that round-trips), so the decimal
    value is exactly the one ``float()`` maps back to ``value``.
    """
    m = _REPR.match(repr(value))
    if m is None:  # pragma: no cover - repr of a finite positive float always matches
        raise ValueError(f"unexpected float repr {repr(value)!r}")
    frac = m.group("frac") or ""
    digits = (m.group("int") + frac).lstrip("0")
    exponent = int(m.group("exp") or 0) - len(frac)
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    return stripped, exponent


def _shift(digits: str, places: int) -> str:
    """``int(digits) * 10**places`` written as a plain decimal (``places`` may be negative)."""
    if places >= 0:
        return digits + "0" * places
    point = len(digits) + places
    if point > 0:
        return digits[:point] + "." + digits[point:]
    return "0." + "0" * (-point) + digits


def _engineering(sign: str, digits: str, exponent: int) -> tuple[str, int | None]:
    """The engineering spelling of ``int(digits) * 10**exponent`` and how many zeros it appended to ``digits``.

    Magnitudes outside the suffix range (below 1e-15, 1e15 and above) get a
    plain exponent (``1e20``); ``None`` then says the spelling is not a
    suffix form and cannot be padded.
    """
    magnitude = len(digits) - 1 + exponent  # floor(log10(|value|))
    if -15 <= magnitude < 15:
        suffix_exponent = (magnitude // 3) * 3
        places = exponent - suffix_exponent
        return f"{sign}{_shift(digits, places)}{_SUFFIX_FOR_EXPONENT[suffix_exponent]}", max(0, places)
    mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{mantissa}e{magnitude}", None


def _pad(engineering: str, zeros: int) -> str:
    """``10u`` -> ``10.0u``: the same value with ``zeros`` more mantissa digits (ngspice reads a different M, E pair)."""
    head = engineering.rstrip("abcdefghijklmnopqrstuvwxyz")
    suffix = engineering[len(head):]
    if "." not in head:
        head += "."
    return f"{head}{'0' * zeros}{suffix}"


def _spellings(sign: str, digits: str, exponent: int) -> list[str]:
    """Exact decimal spellings of ``int(digits) * 10**exponent``: the engineering form first, then the
    same digits with ``k`` trailing zeros (as the padded engineering form when that is possible, else
    as ``<digits>e<exp>``), fewest digits first, all with a mantissa below ``2**53``."""
    engineering, appended = _engineering(sign, digits, exponent)
    out = [engineering]
    for k in range(0, _MAX_EXTRA_ZEROS + 1):
        if int(digits) * 10**k >= NGSPICE_EXACT_MANTISSA:
            break
        if k == appended:
            continue  # that is the engineering form itself
        if appended is not None and k > appended:
            out.append(_pad(engineering, k - appended))
        else:
            e = exponent - k
            out.append(f"{sign}{digits}{'0' * k}" + (f"e{e}" if e else ""))
    return out


def format_spice_number(value: float) -> str:
    """The spelling of ``value`` the netlist should carry: engineering notation with ngspice suffixes,
    adjusted so ngspice reads it back exactly whenever any spelling does.

    ``0 -> "0"``, ``12.0 -> "12"``, ``10000.0 -> "10k"``, ``4.7e-6 -> "4.7u"``,
    ``1e6 -> "1meg"``, ``0.5 -> "500m"``; misread engineering forms are
    replaced by an equivalent one (``1e-5 -> "1e-5"`` because ngspice reads
    ``10u`` as 9.999999999999999e-06, ``2.2e-11 -> "22.0p"``, ``3.3 ->
    "3.300000"``). Magnitudes outside the suffix range use a plain exponent
    (``1e20``). Every result parses back to exactly ``value`` with
    :func:`parse_spice_number`; :func:`ngspice_reads` says whether ngspice
    does too (it does not only when no spelling below ``2**53`` mantissa digits
    is exact, e.g. for some 16-17 significant digit values; the engineering
    form is then returned and ngspice is one ULP off). Non-finite values and
    booleans are rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot write {value!r} as a SPICE number")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    digits, exponent = _decimal_digits(abs(value))
    candidates = _spellings(sign, digits, exponent)
    for text in candidates:
        if ngspice_reads(text) == value:
            return text
    return candidates[0]


__all__ = [
    "MIL_FACTOR",
    "NGSPICE_EXACT_MANTISSA",
    "NGSPICE_EXPONENT_RANGE",
    "SI_VERSION",
    "SPICE_SCALE_EXPONENTS",
    "format_spice_number",
    "ngspice_reads",
    "parse_spice_number",
    "spice_value",
]
