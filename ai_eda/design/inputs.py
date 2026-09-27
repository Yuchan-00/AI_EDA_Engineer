"""Template inputs: the confirmed requirement values a circuit template may read.

Invariant: a template reads only what the user said (``user_requirement``)
or an authoritative source states; a model's extraction or assumption is
never a design input. A typed answer becomes a number only through
:func:`ai_eda.tools.calc.quantity.parse_answer` (the whole text is one
quantity in the key's unit; ``'12 V max'``, ranges and ``'5V 2A'`` are
unusable, with the reason; so is a number that overflows a float, ``'1e309 V'``
or ``1e300 GHz`` - the IR holds no infinity), and two confirmed requirements
under one canonical key that state different numbers are *ambiguous*, never a
pick.
The copied value keeps the requirement's provenance kind, records the
requirement id in ``derived_from`` and starts its note with
:data:`PARSED_NOTE_PREFIX`, so a later run can prove the parameter is still
the requirement (:func:`~ai_eda.design.checks.check_inputs_vs_requirements`).

The board's layer count (``pcb_layers``, alias ``layer_count``) is an
optional input of every template, read by :func:`read_layer_count`: a plain
integer, one of :data:`LAYER_COUNT_OPTIONS`; without one the template uses
:data:`DEFAULT_LAYER_COUNT` and must show that choice in its confirmation
table (:mod:`ai_eda.design.stackup` builds the stack for either count).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import CircuitIR, Provenance, Requirement, Traced
from ai_eda.tools.calc.quantity import Quantity, QuantityRange, find_quantities, format_quantity, parse_answer, parse_unit

#: canonical template input key -> the requirement keys that mean it (typed answers, confirmed extractions)
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "input_voltage": ("input_voltage", "supply_voltage", "v_in", "vin", "dc_input", "vin_dc"),
    "output_voltage": ("output_voltage", "v_out", "vout"),
    "output_current": ("output_current", "load_current", "i_out", "iout"),
    "cutoff_frequency": ("cutoff_frequency", "corner_frequency", "f_c", "fc"),
    "led_forward_voltage": ("led_forward_voltage", "forward_voltage", "v_f", "vf"),
    "led_forward_current": ("led_forward_current", "forward_current", "led_current", "i_f", "if"),
    "oscillation_frequency": ("oscillation_frequency", "output_frequency", "frequency", "f_osc", "fosc"),
    "clock_frequency": ("clock_frequency", "crystal_frequency", "mcu_clock", "f_clk", "fclk"),
}
#: canonical key -> the unit its value must carry
UNIT_OF: dict[str, str] = {
    "input_voltage": "V",
    "output_voltage": "V",
    "output_current": "A",
    "cutoff_frequency": "Hz",
    "led_forward_voltage": "V",
    "led_forward_current": "A",
    "oscillation_frequency": "Hz",
    "clock_frequency": "Hz",
}
#: how the note of a value copied from a requirement starts (followed by the requirement id)
PARSED_NOTE_PREFIX = "parsed from "

#: the board's copper layer count: an optional input every template reads through :func:`read_layer_count` (the
#: stack itself is :mod:`ai_eda.design.stackup`); a count, not a quantity, so it is not in :data:`KEY_ALIASES`
LAYER_COUNT_KEY = "pcb_layers"
#: requirement keys that mean the layer count
LAYER_COUNT_ALIASES: tuple[str, ...] = ("pcb_layers", "layer_count")
#: the layer counts the generic stackups exist for
LAYER_COUNT_OPTIONS: tuple[int, ...] = (2, 4)
#: the count a template uses when no requirement states one (and says so in its confirmation table)
DEFAULT_LAYER_COUNT = 2
#: the ``unit`` of a copied layer count (``read_value(req, LAYER_UNIT)`` reads a plain integer)
LAYER_UNIT = "layers"
#: board-level keys (canonical key -> the requirement keys that mean it); every template serves them through the stackup
BOARD_KEY_ALIASES: dict[str, tuple[str, ...]] = {LAYER_COUNT_KEY: LAYER_COUNT_ALIASES}


@dataclass(frozen=True)
class DesignInput:
    """One confirmed requirement value, read as a number in the canonical unit."""

    key: str
    requirement: Requirement
    traced: Traced


def canonical_key(key: str) -> str | None:
    """The canonical template key a requirement key means (a quantity key or a board key such as ``pcb_layers``), or ``None``."""
    for canon, aliases in (*KEY_ALIASES.items(), *BOARD_KEY_ALIASES.items()):
        if key in aliases:
            return canon
    return None


def _why_not_answer(text: str) -> str:
    """Why :func:`parse_answer` refused ``text`` (for a note a human reads)."""
    hits = find_quantities(text)
    if not hits:
        return "no quantity with a unit"
    if len(hits) > 1:
        return f"several quantities ({', '.join(format_quantity(q) for _, q in hits)}), not one value"
    (start, end), q = hits[0]
    if isinstance(q, QuantityRange):
        return f"a range ({format_quantity(q)}), not one value"
    if isinstance(q, Quantity) and q.plus_minus:
        return f"a tolerance ({format_quantity(q)}), not a value"
    leftover = (text[:start] + " " + text[end:]).strip()
    return f"qualifier {leftover!r} beside {q.original!r} is not read; state one plain value"


def read_value(req: Requirement, unit: str) -> tuple[Traced | None, str | None]:
    """``(traced, None)`` with ``req``'s value as a number in ``unit``, or ``(None, why)``.

    ``unit`` :data:`LAYER_UNIT` reads a layer count: a plain positive integer
    (``4`` or ``"4"``; ``"4 layers"`` is not read), with no unit.
    """
    value = req.value
    if value is None:
        return None, f"{req.id}: has no value"
    prov = value.provenance
    if not prov.is_authoritative:
        return None, f"{req.id}: value is {prov.kind.value}, not yet the user's (confirm it, or answer {req.key} directly)"
    raw = value.value
    if unit == LAYER_UNIT:
        return _read_count(req, raw, value.unit, prov)
    if isinstance(raw, str):
        q = parse_answer(raw)
        if q is None:
            return None, f"{req.id}: {raw!r} is not one whole quantity: {_why_not_answer(raw)}"
        if q.unit != unit:
            return None, f"{req.id}: {raw!r} is a {q.unit} quantity, not {unit}"
        number = q.value
        if not math.isfinite(number):
            return None, f"{req.id}: {raw!r} is not a finite number"
        note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} -> {number:.12g} {unit}"
    elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, f"{req.id}: value {raw!r} is not one number"
    else:
        if value.unit is None:
            return None, f"{req.id}: value {raw!r} carries no unit"
        parsed = parse_unit(value.unit)
        if parsed is None or parsed[0] != unit:
            return None, f"{req.id}: unit {value.unit!r} is not {unit}"
        number = float(raw) * 10.0 ** parsed[1]
        if not math.isfinite(number):
            return None, f"{req.id}: {raw!r} {value.unit} is not a finite number in {unit}"
        note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} {value.unit} -> {number:.12g} {unit}"
    traced = Traced(value=number, unit=unit, provenance=Provenance(kind=prov.kind, source=prov.source, derived_from=[req.id], note=note))
    return traced, None


def _read_count(req: Requirement, raw: object, unit: str | None, prov: Provenance) -> tuple[Traced | None, str | None]:
    """A layer count as the user stated it: an int, or text that is one plain integer; nothing else is read."""
    if unit is not None and unit != LAYER_UNIT:
        return None, f"{req.id}: a layer count has no unit, got {unit!r}"
    if isinstance(raw, bool):
        return None, f"{req.id}: value {raw!r} is not a layer count"
    if isinstance(raw, int):
        count = raw
    elif isinstance(raw, str) and raw.strip().isascii() and raw.strip().isdigit():
        count = int(raw.strip())
    else:
        return None, f"{req.id}: {raw!r} is not one plain layer count (answer {LAYER_COUNT_KEY}=2 or {LAYER_COUNT_KEY}=4)"
    if count <= 0:
        return None, f"{req.id}: {raw!r} is not a positive layer count"
    note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} -> {count} {LAYER_UNIT}"
    return Traced(value=count, unit=LAYER_UNIT, provenance=Provenance(kind=prov.kind, source=prov.source, derived_from=[req.id], note=note)), None


@dataclass(frozen=True)
class LayerCountInput:
    """The board's copper layer count and where it came from.

    ``requirement`` / ``traced`` are the confirmed requirement and its copy
    (``derived_from`` the requirement id, :data:`PARSED_NOTE_PREFIX` note, unit
    :data:`LAYER_UNIT`); both are ``None`` when no requirement states a count
    and the template uses :data:`DEFAULT_LAYER_COUNT` - a choice it must show.
    """

    value: int
    requirement: Requirement | None = None
    traced: Traced | None = None

    @property
    def is_default(self) -> bool:
        return self.requirement is None


def read_layer_count(ir: CircuitIR) -> tuple[LayerCountInput | None, str | None]:
    """``(input, None)`` with the board's layer count, or ``(None, why)`` when a stated count is unusable.

    No requirement under :data:`LAYER_COUNT_ALIASES` -> the default
    (:data:`DEFAULT_LAYER_COUNT`, ``is_default``). A stated count must be
    confirmed (the user's or authoritative), one plain integer, one of
    :data:`LAYER_COUNT_OPTIONS`, and every requirement that states one must
    state the same (two different counts are ambiguous, never a pick).
    """
    candidates = [r for r in ir.requirements.requirements if r.key in LAYER_COUNT_ALIASES]
    if not candidates:
        return LayerCountInput(value=DEFAULT_LAYER_COUNT), None
    readings: list[tuple[Requirement, Traced]] = []
    reasons: list[str] = []
    for r in candidates:
        traced, why = read_value(r, LAYER_UNIT)
        if traced is None:
            reasons.append(why or f"{r.id}: unreadable")
        else:
            readings.append((r, traced))
    if reasons:
        return None, "; ".join(reasons)
    counts = {int(t.value) for _, t in readings}
    if len(counts) > 1:
        return None, "ambiguous: " + ", ".join(f"{r.id} says {int(t.value)}" for r, t in readings)
    count = counts.pop()
    if count not in LAYER_COUNT_OPTIONS:
        return None, f"{readings[0][0].id}: {count} layers is not one of the stackups this version builds ({', '.join(map(str, LAYER_COUNT_OPTIONS))})"
    req, traced = readings[0]
    return LayerCountInput(value=count, requirement=req, traced=traced), None


def read_inputs(ir: CircuitIR) -> tuple[dict[str, DesignInput], dict[str, str]]:
    """Confirmed numeric requirement values by canonical key, and the keys that are present but unusable (with why).

    A key with two or more confirmed requirements (an alias beside the
    canonical key, a typed answer beside a confirmed extraction) is usable
    only when every one of them reads to the same number; different numbers
    are ``ambiguous`` and an unreadable one makes the key unusable.
    """
    found: dict[str, DesignInput] = {}
    unusable: dict[str, str] = {}
    for canon, aliases in KEY_ALIASES.items():
        candidates = [r for r in ir.requirements.requirements if r.key in aliases]
        if not candidates:
            continue
        unit = UNIT_OF[canon]
        readings: list[tuple[Requirement, Traced]] = []
        reasons: list[str] = []
        for r in candidates:
            traced, why = read_value(r, unit)
            if traced is None:
                reasons.append(why or f"{r.id}: unreadable")
            else:
                readings.append((r, traced))
        if reasons:
            unusable[canon] = "; ".join(reasons)
            continue
        first_req, first = readings[0]
        differing = [(r, t) for r, t in readings[1:] if t.value != first.value]
        if differing:
            stated = ", ".join(f"{r.id} says {t.value:.12g} {unit}" for r, t in readings)
            unusable[canon] = f"ambiguous: {stated}"
            continue
        found[canon] = DesignInput(canon, first_req, first)
    return found, unusable


def is_template_input(t: Traced) -> bool:
    """Whether a traced value is a requirement copied by :func:`read_value` (one requirement id, the parsed-from note)."""
    p = t.provenance
    return p.is_authoritative and len(p.derived_from) == 1 and p.derived_from[0].startswith("req.") and (p.note or "").startswith(PARSED_NOTE_PREFIX)


__all__ = [
    "BOARD_KEY_ALIASES",
    "DEFAULT_LAYER_COUNT",
    "KEY_ALIASES",
    "LAYER_COUNT_ALIASES",
    "LAYER_COUNT_KEY",
    "LAYER_COUNT_OPTIONS",
    "LAYER_UNIT",
    "PARSED_NOTE_PREFIX",
    "UNIT_OF",
    "DesignInput",
    "LayerCountInput",
    "canonical_key",
    "is_template_input",
    "read_inputs",
    "read_layer_count",
    "read_value",
]
