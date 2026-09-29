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

RF keys (:data:`RF_KEY_ALIASES`, units in :data:`RF_UNIT_OF`) are read like
every other key: ``carrier_frequency`` (Hz), ``tx_power`` (W - the conducted
power at the antenna port; an ERP / EIRP limit goes under ``erp`` / ``eirp``,
never under ``tx_power``; ``calc.rf.erp_to_eirp`` / ``calc.rf.eirp_to_erp`` bridge
the two references with the half-wave dipole's 2.15 dBi), ``frequency_deviation``
(the peak deviation; ``±2.5 kHz`` reads as 2.5 kHz), ``audio_bandwidth``,
``channel_bandwidth``, ``occupied_bandwidth`` and ``channel_spacing`` (Hz;
three different quantities: the channel allocation, the regulated 99 %
occupied bandwidth and the raster), ``modulation_depth`` (percent, kept as
written: ``80 %`` is 80), ``modulation_index`` (:data:`RATIO_UNIT`: one plain
number; beta for FM / PM, m for AM), ``rx_sensitivity`` (dBm),
``system_impedance`` (the reference impedance the RF ports and lines are
designed to) and ``antenna_impedance`` (the antenna's stated feed-point
resistance) in ohm - different quantities, which is why a match exists -,
``antenna_gain`` (dBi), ``link_range`` (m), ``frequency_tolerance`` (ppm),
``field_strength_limit`` (V/m) and ``tx_timeout`` (s: the transmit time-out
after which a transmission is cut off; ``3 min`` is not read - state seconds).
``frequency_tolerance`` and ``frequency_deviation``
are the keys (:data:`SYMMETRIC_TOLERANCE_KEYS`) where a ``±`` answer is a value -
``±2.5 ppm`` reads as 2.5 ppm, ``±2.5 kHz`` as a 2.5 kHz peak deviation: a bare
``±x`` there has no nominal beside it and means its magnitude; under every other
key it is a tolerance, not a value. A level is
typed by the quantity parser (``27 dBm`` is 27 dBm) and becomes a number in
another unit only here, in :func:`read_value`, through a named calculator
(dBm / dBW -> W by ``calc.rf.dbm_to_w`` / ``calc.rf.dbw_to_w``, dBuV/m -> V/m by
``calc.rf.dbuvm_to_vm``) that its note names; every other unit mismatch is
refused. ``modulation`` (aliases :data:`MODULATION_ALIASES`) is categorical and
read by :func:`read_modulation`, never as a quantity. ``battery_voltage`` is the
one battery alias of ``input_voltage`` (``vbat`` names an MCU's backup-supply
pin, not the board's supply).

``radio_build`` (:data:`RADIO_BUILD_KEY`) is the other categorical key: which
board of the radio template family is built, exactly one of
:data:`RADIO_BUILDS`, read by :func:`read_radio_build`. It is a requirement,
not a control key - which board is built is the product - so it is typed once
(``--answer radio_build=rx_backend``), persists as ``user_requirement`` and is
served (or refused, closed world) like every design requirement. The whole
value must be one build name (case-insensitive, ``-`` read as ``_``): a
description of a board is never mapped to one. A categorical value is never
copied into ``ir.parameters`` with the :data:`PARSED_NOTE_PREFIX` note, which
:func:`~ai_eda.design.checks.check_inputs_vs_requirements` re-reads as a
quantity.

No alias is a bare physical word: ``frequency`` (which frequency?), ``power``,
``bandwidth``, ``impedance``, ``sensitivity``, ``range``, ``gain``,
``deviation``, ``depth``, ``index``, ``voltage`` and ``current`` name no key,
and no alias merges two different quantities (``center_frequency`` is not the
carrier, ``if_bandwidth`` is not the channel bandwidth): a requirement under
such a key stays unread instead of being guessed into one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ai_eda.ir import CircuitIR, Provenance, Requirement, Traced
from ai_eda.tools.calc import rf as rf_calc
from ai_eda.tools.calc.quantity import Quantity, QuantityRange, find_quantities, format_quantity, parse_answer, parse_unit

#: canonical template input key -> the requirement keys that mean it (typed answers, confirmed extractions)
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "input_voltage": ("input_voltage", "supply_voltage", "v_in", "vin", "dc_input", "vin_dc"),
    "output_voltage": ("output_voltage", "v_out", "vout"),
    "output_current": ("output_current", "load_current", "i_out", "iout"),
    "cutoff_frequency": ("cutoff_frequency", "corner_frequency", "f_c", "fc"),
    "led_forward_voltage": ("led_forward_voltage", "forward_voltage", "v_f", "vf"),
    "led_forward_current": ("led_forward_current", "forward_current", "led_current", "i_f", "if"),
    "oscillation_frequency": ("oscillation_frequency", "output_frequency", "f_osc", "fosc"),
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

# --------------------------------------------------------------------------- RF keys
# Added to KEY_ALIASES / UNIT_OF by mutation after their literals (below), so the literals above stay as they are.

#: the unit of a copied plain ratio (``read_value(req, RATIO_UNIT)`` reads one plain finite number, ``5`` or ``"5"``); the
#: copy carries this unit, never ``None``, so :func:`~ai_eda.design.checks.check_inputs_vs_requirements` can re-read it
RATIO_UNIT = "ratio"
#: canonical RF key -> the requirement keys that mean it (never a bare physical word, never an alias that merges two quantities)
RF_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "carrier_frequency": ("carrier_frequency", "rf_frequency", "f_carrier", "tx_frequency"),
    # the conducted power at the antenna port; a radiated limit goes under erp / eirp
    "tx_power": ("tx_power", "transmit_power", "rf_output_power", "output_power_rf", "p_tx"),
    "erp": ("erp", "effective_radiated_power"),
    "eirp": ("eirp", "equivalent_isotropic_radiated_power"),
    "frequency_deviation": ("frequency_deviation", "fm_deviation", "peak_deviation"),
    "modulation_depth": ("modulation_depth", "am_modulation_depth", "am_depth"),
    "modulation_index": ("modulation_index", "fm_modulation_index"),
    "audio_bandwidth": ("audio_bandwidth", "af_bandwidth", "baseband_bandwidth", "max_audio_frequency"),
    "channel_bandwidth": ("channel_bandwidth",),
    "occupied_bandwidth": ("occupied_bandwidth", "obw"),
    "channel_spacing": ("channel_spacing", "channel_step", "channel_raster"),
    "rx_sensitivity": ("rx_sensitivity", "receiver_sensitivity"),
    "system_impedance": ("system_impedance", "rf_port_impedance", "rf_impedance"),
    "antenna_impedance": ("antenna_impedance", "antenna_feed_impedance"),
    "antenna_gain": ("antenna_gain", "g_antenna"),
    "link_range": ("link_range", "communication_range", "radio_range"),
    "frequency_tolerance": ("frequency_tolerance", "frequency_stability"),
    "field_strength_limit": ("field_strength_limit", "e_field_limit"),
    # the transmit time-out: a transmission is cut off after it (a radio template builds its time-out timer only when stated)
    "tx_timeout": ("tx_timeout", "transmit_timeout"),
}
#: canonical RF key -> the unit its value must carry (W keys also read dBm / dBW, V/m keys dBuV/m, through calc.rf)
RF_UNIT_OF: dict[str, str] = {
    "carrier_frequency": "Hz", "tx_power": "W", "erp": "W", "eirp": "W", "frequency_deviation": "Hz",
    "modulation_depth": "percent", "modulation_index": RATIO_UNIT, "audio_bandwidth": "Hz", "channel_bandwidth": "Hz",
    "occupied_bandwidth": "Hz", "channel_spacing": "Hz", "rx_sensitivity": "dBm", "system_impedance": "ohm",
    "antenna_impedance": "ohm", "antenna_gain": "dBi", "link_range": "m", "frequency_tolerance": "ppm",
    "field_strength_limit": "V/m", "tx_timeout": "s",
}
#: the battery aliases of ``input_voltage`` (not ``vbat`` / ``v_bat``: datasheets use those for an MCU's backup-supply pin)
BATTERY_ALIASES: tuple[str, ...] = ("battery_voltage",)
#: canonical keys whose value is naturally written with ``±`` and means its magnitude: a frequency tolerance
#: (``±2.5 ppm`` -> 2.5 ppm) and a peak FM deviation (``±2.5 kHz`` -> 2.5 kHz: the deviation swings the carrier
#: both ways, and a bare ``±Δf`` names no nominal); under every other key a ``±`` answer is a tolerance, not a value
SYMMETRIC_TOLERANCE_KEYS: frozenset[str] = frozenset({"frequency_tolerance", "frequency_deviation"})
#: the categorical modulation key (read by :func:`read_modulation`, never as a quantity) and the requirement keys that mean it
MODULATION_KEY = "modulation"
MODULATION_ALIASES: tuple[str, ...] = ("modulation", "modulation_type", "emission_mode")
#: the modulation names :func:`read_modulation` recognises (ASCII tokens, case-insensitive, whole words)
MODULATIONS: tuple[str, ...] = ("am", "fm", "pm", "ssb", "dsb", "fsk", "gfsk", "ask", "ook", "psk", "lora")
#: Korean phrases for a modulation (whitespace inside optional)
MODULATION_PHRASES: dict[str, str] = {"진폭 변조": "am", "주파수 변조": "fm"}
#: the categorical radio-build key (read by :func:`read_radio_build`, never as a quantity): which board of the radio
#: template family is built - a requirement (it names the product), never a control key - and the keys that mean it
RADIO_BUILD_KEY = "radio_build"
RADIO_BUILD_ALIASES: tuple[str, ...] = ("radio_build",)
#: the boards of the radio template family, in build order (``transceiver_conducted``: the transceiver with a coaxial
#: connector in place of the integral antenna, for conducted bench measurements)
RADIO_BUILDS: tuple[str, ...] = ("audio_ptt", "rx_backend", "rx_frontend", "tx_exciter", "transceiver", "transceiver_conducted")
#: (level unit, target unit) -> (calculator, its tool id): the only unit conversions :func:`read_value` makes
LEVEL_CONVERSIONS = {
    ("dBm", "W"): (rf_calc.dbm_to_w, "calc.rf.dbm_to_w"),
    ("dBW", "W"): (rf_calc.dbw_to_w, "calc.rf.dbw_to_w"),
    ("dBuV/m", "V/m"): (rf_calc.dbuvm_to_vm, "calc.rf.dbuvm_to_vm"),
}

KEY_ALIASES.update(RF_KEY_ALIASES)
KEY_ALIASES["input_voltage"] = (*KEY_ALIASES["input_voltage"], *BATTERY_ALIASES)
UNIT_OF.update(RF_UNIT_OF)


@dataclass(frozen=True)
class DesignInput:
    """One confirmed requirement value, read as a number in the canonical unit."""

    key: str
    requirement: Requirement
    traced: Traced


def canonical_key(key: str) -> str | None:
    """The canonical template key a requirement key means (a quantity key, a board key such as ``pcb_layers`` or a
    categorical key: ``modulation``, ``radio_build``), or ``None``."""
    for canon, aliases in (*KEY_ALIASES.items(), *BOARD_KEY_ALIASES.items(), (MODULATION_KEY, MODULATION_ALIASES),
                           (RADIO_BUILD_KEY, RADIO_BUILD_ALIASES)):
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
    ``unit`` :data:`RATIO_UNIT` reads one plain finite number (``5`` or ``"5"``).
    A level answer for a W / V/m key (``27 dBm``, ``-3 dBW``, ``94 dBuV/m`` - as
    text or as a number whose unit is the level) is converted by the named
    calculator of :data:`LEVEL_CONVERSIONS`, which the note names; the copy
    still records only the requirement id, so a later run re-reads it the same
    way. A ``±`` answer is a value only under :data:`SYMMETRIC_TOLERANCE_KEYS`.
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
    if unit == RATIO_UNIT:
        return _read_ratio(req, raw, value.unit, prov)
    symmetric = canonical_key(req.key) in SYMMETRIC_TOLERANCE_KEYS
    if isinstance(raw, str):
        q = parse_answer(raw)
        if q is None and symmetric:
            q = _symmetric_answer(raw)
        if q is None:
            return None, f"{req.id}: {raw!r} is not one whole quantity: {_why_not_answer(raw)}"
        if q.unit != unit:
            if unit == "dBm" and q.unit == "V":
                return None, _SENSITIVITY_IN_VOLTS.format(id=req.id, raw=repr(raw))
            if (q.unit, unit) not in LEVEL_CONVERSIONS:
                return None, f"{req.id}: {raw!r} is a {q.unit} quantity, not {unit}"
            number, why = _convert_level(req, q.value, q.unit, unit, prov)
            if number is None:
                return None, f"{req.id}: {raw!r}: {why}"
            tool = LEVEL_CONVERSIONS[(q.unit, unit)][1]
            note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} -> {q.value:.12g} {q.unit} -> {number:.12g} {unit} ({tool})"
        else:
            number = q.value
            if not math.isfinite(number):
                return None, f"{req.id}: {raw!r} is not a finite number"
            note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} -> {number:.12g} {unit}"
    elif symmetric and _symmetric_pair(raw) is not None:
        # a confirmed extraction of "±2.5 ppm" stores the symmetric range [-2.5, 2.5]
        magnitude = _symmetric_pair(raw)
        assert magnitude is not None
        if value.unit is None:
            return None, f"{req.id}: value {raw!r} carries no unit"
        parsed = parse_unit(value.unit)
        if parsed is None or parsed[0] != unit:
            return None, f"{req.id}: unit {value.unit!r} is not {unit}"
        number = magnitude * 10.0 ** parsed[1]
        if not math.isfinite(number):
            return None, f"{req.id}: {raw!r} {value.unit} is not a finite number in {unit}"
        note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} {value.unit} -> {number:.12g} {unit} (a ± value, read as its magnitude)"
    elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, f"{req.id}: value {raw!r} is not one number"
    else:
        if value.unit is None:
            return None, f"{req.id}: value {raw!r} carries no unit"
        parsed = parse_unit(value.unit)
        if parsed is not None and parsed[1] == 0 and (parsed[0], unit) in LEVEL_CONVERSIONS:
            number, why = _convert_level(req, float(raw), parsed[0], unit, prov)
            if number is None:
                return None, f"{req.id}: {raw!r} {value.unit}: {why}"
            tool = LEVEL_CONVERSIONS[(parsed[0], unit)][1]
            note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} {value.unit} -> {number:.12g} {unit} ({tool})"
        else:
            if unit == "dBm" and parsed is not None and parsed[0] == "V":
                return None, _SENSITIVITY_IN_VOLTS.format(id=req.id, raw=f"{raw!r} {value.unit}")
            if parsed is None or parsed[0] != unit:
                return None, f"{req.id}: unit {value.unit!r} is not {unit}"
            number = float(raw) * 10.0 ** parsed[1]
            if not math.isfinite(number):
                return None, f"{req.id}: {raw!r} {value.unit} is not a finite number in {unit}"
            note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} {value.unit} -> {number:.12g} {unit}"
    traced = Traced(value=number, unit=unit, provenance=Provenance(kind=prov.kind, source=prov.source, derived_from=[req.id], note=note))
    return traced, None


#: why a receiver sensitivity stated in volts is not read (the dBm figure needs a convention the text does not give)
_SENSITIVITY_IN_VOLTS = (
    "{id}: {raw} is a voltage; a receiver sensitivity in volts needs the source impedance and the EMF / PD convention - state it in dBm"
)


def _convert_level(req: Requirement, level: float, level_unit: str, unit: str, prov: Provenance) -> tuple[float | None, str | None]:
    """``level`` (in ``level_unit``) as a number in ``unit`` through the calculator :data:`LEVEL_CONVERSIONS` names."""
    fn, tool = LEVEL_CONVERSIONS[(level_unit, unit)]
    if not math.isfinite(level):
        return None, f"{level!r} {level_unit} is not a finite number"
    try:
        out = fn(Traced(value=float(level), unit=level_unit, provenance=Provenance(kind=prov.kind, source=prov.source)), (req.id,))
    except ValueError as e:
        return None, f"{tool} refused it: {e}"
    return float(out.value), None


def _symmetric_answer(text: str) -> Quantity | None:
    """A ``±`` answer (``±2.5 ppm``, ``±2.5 kHz``) as its magnitude - for :data:`SYMMETRIC_TOLERANCE_KEYS` only; nothing else may
    stand beside it."""
    hits = find_quantities(text)
    if len(hits) != 1:
        return None
    (start, end), q = hits[0]
    if not isinstance(q, Quantity) or not q.plus_minus or (text[:start] + text[end:]).strip():
        return None
    return Quantity(value=abs(q.value), unit=q.unit, original=q.original)


def _symmetric_pair(raw: object) -> float | None:
    """The magnitude ``a`` of a stored symmetric range ``[-a, a]`` (a > 0), else ``None``."""
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    lo, hi = raw
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (lo, hi)):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= 0 or lo != -hi:
        return None
    return float(hi)


_PLAIN_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _read_ratio(req: Requirement, raw: object, unit: str | None, prov: Provenance) -> tuple[Traced | None, str | None]:
    """A plain ratio (a modulation index) as the user stated it: a number, or text that is one plain number."""
    if unit not in (None, "", RATIO_UNIT):
        return None, f"{req.id}: a ratio has no unit, got {unit!r}"
    if isinstance(raw, bool):
        return None, f"{req.id}: {raw!r} is not one plain number"
    if isinstance(raw, (int, float)):
        number = float(raw)
    elif isinstance(raw, str) and _PLAIN_NUMBER_RE.fullmatch(raw.strip()):
        number = float(raw.strip())
    else:
        return None, f"{req.id}: {raw!r} is not one plain number"
    if not math.isfinite(number):
        return None, f"{req.id}: {raw!r} is not a finite number"
    note = f"{PARSED_NOTE_PREFIX}{req.id}: {raw!r} -> {number:.12g} {RATIO_UNIT}"
    return Traced(value=number, unit=RATIO_UNIT, provenance=Provenance(kind=prov.kind, source=prov.source, derived_from=[req.id], note=note)), None


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


_MODULATION_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(MODULATIONS, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.IGNORECASE)
_MODULATION_PHRASE_RES = {re.compile(r"\s*".join(map(re.escape, phrase.replace(" ", "")))): name for phrase, name in MODULATION_PHRASES.items()}


def _modulations_named(text: str) -> list[str]:
    """The modulation names a text states, sorted (ASCII tokens as whole words, Korean phrases)."""
    found = {m.group(1).lower() for m in _MODULATION_TOKEN_RE.finditer(text)}
    found |= {name for rx, name in _MODULATION_PHRASE_RES.items() if rx.search(text)}
    return sorted(found)


def read_modulation(ir: CircuitIR) -> tuple[str | None, str | None]:
    """``(name, None)`` with the confirmed modulation (``"fm"``, one of :data:`MODULATIONS`), or ``(None, why)``.

    ``(None, None)`` when no requirement under :data:`MODULATION_ALIASES`
    states one (a template asks for it). A stated modulation must be
    confirmed (the user's or authoritative) and name exactly one modulation
    (``"FM"``, ``"narrowband FM"``, ``"주파수 변조"``); none, several
    (``"AM/FM"``) or requirements that name different ones are refused with
    the reason, never picked.
    """
    candidates = [r for r in ir.requirements.requirements if r.key in MODULATION_ALIASES]
    if not candidates:
        return None, None
    names: dict[str, str] = {}
    reasons: list[str] = []
    for r in candidates:
        value = r.value
        if value is None:
            reasons.append(f"{r.id}: has no value")
            continue
        if not value.provenance.is_authoritative:
            reasons.append(f"{r.id}: value is {value.provenance.kind.value}, not yet the user's (confirm it, or answer {r.key} directly)")
            continue
        raw = value.value
        if not isinstance(raw, str):
            reasons.append(f"{r.id}: {raw!r} is not a modulation name (one of {', '.join(MODULATIONS)})")
            continue
        named = _modulations_named(raw)
        if not named:
            reasons.append(f"{r.id}: {raw!r} names no modulation (one of {', '.join(MODULATIONS)})")
        elif len(named) > 1:
            reasons.append(f"{r.id}: {raw!r} names several modulations ({', '.join(named)}); state one")
        else:
            names[r.id] = named[0]
    if reasons:
        return None, "; ".join(reasons)
    if len(set(names.values())) > 1:
        return None, "ambiguous: " + ", ".join(f"{rid} says {name}" for rid, name in names.items())
    return next(iter(names.values())), None


#: a build name inside a longer text (whole words; ``-`` or ``_`` between its parts) - only to name what a refused text holds
_RADIO_BUILD_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(sorted((b.replace("_", "[_-]") for b in RADIO_BUILDS), key=len, reverse=True)) + r")(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)


def read_radio_build(ir: CircuitIR) -> tuple[str | None, str | None]:
    """``(build, None)`` with the confirmed radio build (one of :data:`RADIO_BUILDS`), or ``(None, why)``.

    ``(None, None)`` when no requirement under :data:`RADIO_BUILD_ALIASES`
    states one (the radio selector asks for it). A stated build must be
    confirmed (the user's or authoritative) and its whole value one build name
    - case-insensitive, surrounding whitespace ignored, ``-`` read as ``_``
    (``"RX-Backend"`` is ``rx_backend``). A text that holds a build name among
    other words (``"rx_backend board"``), names several (``"rx_backend,
    tx_exciter"``), names none (``"IF 백엔드 보드"``: a description is never
    mapped to a build) or requirements that name different builds are refused
    with the reason, never picked.
    """
    candidates = [r for r in ir.requirements.requirements if r.key in RADIO_BUILD_ALIASES]
    if not candidates:
        return None, None
    choices = ", ".join(RADIO_BUILDS)
    builds: dict[str, str] = {}
    reasons: list[str] = []
    for r in candidates:
        value = r.value
        if value is None:
            reasons.append(f"{r.id}: has no value")
            continue
        if not value.provenance.is_authoritative:
            reasons.append(f"{r.id}: value is {value.provenance.kind.value}, not yet the user's (confirm it, or answer {r.key} directly)")
            continue
        raw = value.value
        if not isinstance(raw, str):
            reasons.append(f"{r.id}: {raw!r} is not a radio build name (one of {choices})")
            continue
        whole = raw.strip().lower().replace("-", "_")
        if whole in RADIO_BUILDS:
            builds[r.id] = whole
            continue
        named = sorted({m.group(1).lower().replace("-", "_") for m in _RADIO_BUILD_TOKEN_RE.finditer(raw)})
        if len(named) > 1:
            reasons.append(f"{r.id}: {raw!r} names several radio builds ({', '.join(named)}); state exactly one")
        elif named:
            reasons.append(f"{r.id}: {raw!r} is not exactly one radio build name (it contains {named[0]} among other words); state exactly one of {choices} and nothing else")
        else:
            reasons.append(f"{r.id}: {raw!r} names no radio build (state exactly one of {choices})")
    if reasons:
        return None, "; ".join(reasons)
    if len(set(builds.values())) > 1:
        return None, "ambiguous: " + ", ".join(f"{rid} says {build}" for rid, build in builds.items())
    return next(iter(builds.values())), None


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


def present_keys(ir: CircuitIR, inputs: dict[str, DesignInput]) -> set[str]:
    """The canonical keys a template may count as present: the numeric ``inputs`` plus each categorical key its reader reads.

    ``modulation`` counts when :func:`read_modulation` returns a name,
    ``radio_build`` when :func:`read_radio_build` returns a build; a stated
    but unreadable value (not confirmed, ambiguous, naming several) stays
    missing. The one rule for a template's ``needs``: the selection gate of
    :func:`ai_eda.design.templates.design_from_requirements` and a template's
    own ``build`` ask for the same missing keys.
    """
    present = set(inputs)
    if read_modulation(ir)[0] is not None:
        present.add(MODULATION_KEY)
    if read_radio_build(ir)[0] is not None:
        present.add(RADIO_BUILD_KEY)
    return present


#: the categorical keys a template's ``needs`` may name beside the numeric ones (read by :func:`present_keys`)
CATEGORICAL_KEYS: tuple[str, ...] = (MODULATION_KEY, RADIO_BUILD_KEY)


def is_template_input(t: Traced) -> bool:
    """Whether a traced value is a requirement copied by :func:`read_value` (one requirement id, the parsed-from note)."""
    p = t.provenance
    return p.is_authoritative and len(p.derived_from) == 1 and p.derived_from[0].startswith("req.") and (p.note or "").startswith(PARSED_NOTE_PREFIX)


__all__ = [
    "BATTERY_ALIASES",
    "BOARD_KEY_ALIASES",
    "CATEGORICAL_KEYS",
    "DEFAULT_LAYER_COUNT",
    "KEY_ALIASES",
    "LAYER_COUNT_ALIASES",
    "LAYER_COUNT_KEY",
    "LAYER_COUNT_OPTIONS",
    "LAYER_UNIT",
    "LEVEL_CONVERSIONS",
    "MODULATIONS",
    "MODULATION_ALIASES",
    "MODULATION_KEY",
    "MODULATION_PHRASES",
    "PARSED_NOTE_PREFIX",
    "RADIO_BUILDS",
    "RADIO_BUILD_ALIASES",
    "RADIO_BUILD_KEY",
    "RATIO_UNIT",
    "RF_KEY_ALIASES",
    "RF_UNIT_OF",
    "SYMMETRIC_TOLERANCE_KEYS",
    "UNIT_OF",
    "DesignInput",
    "LayerCountInput",
    "canonical_key",
    "is_template_input",
    "present_keys",
    "read_inputs",
    "read_layer_count",
    "read_modulation",
    "read_radio_build",
    "read_value",
]
