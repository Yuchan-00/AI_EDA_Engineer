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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import CircuitIR, Provenance, Requirement, Traced
from ai_eda.tools.calc.quantity import Quantity, QuantityRange, find_quantities, format_quantity, parse_answer, parse_unit

#: canonical template input key -> the requirement keys that mean it (typed answers, confirmed extractions)
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "input_voltage": ("input_voltage", "supply_voltage", "v_in", "vin"),
    "output_voltage": ("output_voltage", "v_out", "vout"),
    "output_current": ("output_current", "load_current", "i_out", "iout"),
    "cutoff_frequency": ("cutoff_frequency", "corner_frequency", "f_c", "fc"),
    "led_forward_voltage": ("led_forward_voltage", "forward_voltage", "v_f", "vf"),
    "led_forward_current": ("led_forward_current", "forward_current", "led_current", "i_f", "if"),
}
#: canonical key -> the unit its value must carry
UNIT_OF: dict[str, str] = {
    "input_voltage": "V",
    "output_voltage": "V",
    "output_current": "A",
    "cutoff_frequency": "Hz",
    "led_forward_voltage": "V",
    "led_forward_current": "A",
}
#: how the note of a value copied from a requirement starts (followed by the requirement id)
PARSED_NOTE_PREFIX = "parsed from "


@dataclass(frozen=True)
class DesignInput:
    """One confirmed requirement value, read as a number in the canonical unit."""

    key: str
    requirement: Requirement
    traced: Traced


def canonical_key(key: str) -> str | None:
    """The canonical template key a requirement key means, or ``None``."""
    for canon, aliases in KEY_ALIASES.items():
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
    """``(traced, None)`` with ``req``'s value as a number in ``unit``, or ``(None, why)``."""
    value = req.value
    if value is None:
        return None, f"{req.id}: has no value"
    prov = value.provenance
    if not prov.is_authoritative:
        return None, f"{req.id}: value is {prov.kind.value}, not yet the user's (confirm it, or answer {req.key} directly)"
    raw = value.value
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


__all__ = ["KEY_ALIASES", "PARSED_NOTE_PREFIX", "UNIT_OF", "DesignInput", "canonical_key", "is_template_input", "read_inputs", "read_value"]
