"""Keyboard design content of the IR (``ir.keyboard``): the key matrix, the I2C buses, the matrix scenarios.

Invariant: this is a *declaration* the checks verify, never a verdict and
never a substitute for the netlist. ``keyboard.matrix`` reads the declared
matrix only to know what to look for and judges it against ``ir.nets`` and
the library pin names of the parts (a declared key whose switch, diode, row
or column the nets do not show is a FAIL, not a fact); ``keyboard.keycaps``
judges the keycap outlines at the placed key positions; the SPICE checks
``spice.i2c.<bus>`` / ``spice.matrix.<scenario>`` build their own decks from
the IR's parts and the model values named here. Positions are never stored
here: a key is where its placement puts it (``pcb.fixed`` / ``pcb.placements``).

Every number a check needs that is not a part of the netlist (a pull-up
inside the scanner, an input threshold, a contact resistance, a capacitance
of a cable or a pin, the bus clock, the rise-time limit) is a parameter key
(``ir.parameters``) named here, never a literal: a template's confirmed
choice, the user's value, a grounded fact or a calculator's output, with its
provenance. ``model_values`` lists the ``kb.model.*`` keys no datasheet or
measurement grounds, so a verdict that uses them says it is a verdict under
those confirmed model values.

The models refuse what they cannot mean (a ``ValueError`` naming it):

* :class:`KeyboardKey` - a key switch ``ref`` in series with exactly one
  diode ``diode_ref`` at a matrix position (``row``, ``col`` >= 0), its keycap
  size in key units in the key's own frame (``w_u`` / ``h_u`` > 0) and an
  optional legend.
* :class:`KeyMatrix` - the row and column net names (index = row / column
  number; no name twice, rows and columns disjoint), the current direction
  (``col2row``: column -> switch -> diode anode -> cathode -> row), the keys
  (refs unique across switches and diodes, positions unique and in range)
  and ``scanner_ref``, the part whose pins drive the rows and read the columns.
* :class:`I2CBus` - a bus by its SDA / SCL nets, the supply rail, the
  pull-up resistors, the parameter keys of its clock, its rise-time limit and
  every capacitance on it (pins, copper, cable), and the threshold fractions
  of VDD between which the rise time is measured (I2C-bus specification
  UM10204: t_r is the time from 0.3 VDD to 0.7 VDD).
* :class:`MatrixScenario` - one set of pressed keys with one row driven
  active and the columns that must then read pressed / unpressed, plus the
  parameter keys of the scanner's input pull-up, its input thresholds, the
  closed contact's resistance and the supply.
* :class:`KeyboardDesign` - the build it belongs to, the matrix, the buses,
  the scenarios and ``model_values``.

Identifiers (bus and scenario ids) are plain identifiers: they become
check-id segments and deck stems.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ai_eda.ir.provenance import Provenance, ProvenanceKind

#: plain identifiers: bus / scenario ids become check-id segments (``spice.i2c.<bus>``) and deck stems
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
#: the prefix of every keyboard modelling number no datasheet or measurement grounds
MODEL_PREFIX = "kb.model."
#: the current directions of a diode matrix
DIRECTIONS: tuple[str, ...] = ("col2row", "row2col")

Direction = Literal["col2row", "row2col"]


def _check_id(value: str, what: str) -> None:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise ValueError(f"{what} {value!r} must be a plain identifier (a letter, then letters, digits or _): it becomes a check-id segment")


def _check_name(value: str, what: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ch.isspace() for ch in value):
        raise ValueError(f"{what} {value!r} must be a non-empty name without spaces")


def _unique(values: list[str], what: str) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"{what} {v!r} is used twice")
        seen.add(v)


def _unrecorded() -> Provenance:
    return Provenance(kind=ProvenanceKind.ASSUMPTION, note="origin not recorded")


class KeyboardKey(BaseModel):
    """One key: its switch, the diode in series with it, its matrix position and its keycap size (module docstring)."""

    ref: str
    diode_ref: str
    row: int
    col: int
    #: keycap width / height in key units, in the key's own frame (before the placement's rotation)
    w_u: float = 1.0
    h_u: float = 1.0
    #: the legend drawn in the layout figure ("Q", "Esc"; empty = blank)
    label: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> KeyboardKey:
        _check_name(self.ref, "key ref")
        _check_name(self.diode_ref, f"key {self.ref}: diode ref")
        if self.ref == self.diode_ref:
            raise ValueError(f"key {self.ref}: the switch and its diode must be two parts")
        if self.row < 0 or self.col < 0:
            raise ValueError(f"key {self.ref}: row / col must be >= 0, got ({self.row}, {self.col})")
        for label, v in (("w_u", self.w_u), ("h_u", self.h_u)):
            if not (isinstance(v, (int, float)) and v > 0 and v == v and v != float("inf")):
                raise ValueError(f"key {self.ref}: {label} must be a finite number > 0, got {v!r}")
        return self


class KeyMatrix(BaseModel):
    """The diode key matrix: row / column nets, direction, keys and the scanning part (module docstring)."""

    rows: list[str]
    cols: list[str]
    direction: Direction = "col2row"
    keys: list[KeyboardKey]
    #: the part whose pins drive the rows and read the columns (an MCU or an I/O expander)
    scanner_ref: str

    @model_validator(mode="after")
    def _consistent(self) -> KeyMatrix:
        if not self.rows or not self.cols:
            raise ValueError("a key matrix needs at least one row and one column net")
        for n in self.rows + self.cols:
            _check_name(n, "matrix net")
        _unique(self.rows, "row net")
        _unique(self.cols, "column net")
        both = set(self.rows) & set(self.cols)
        if both:
            raise ValueError(f"nets {sorted(both)} are both a row and a column")
        if not self.keys:
            raise ValueError("a key matrix needs at least one key")
        _check_name(self.scanner_ref, "scanner ref")
        _unique([r for k in self.keys for r in (k.ref, k.diode_ref)], "key / diode ref")
        if self.scanner_ref in {r for k in self.keys for r in (k.ref, k.diode_ref)}:
            raise ValueError(f"scanner {self.scanner_ref!r} is also a key or a diode")
        seen: dict[tuple[int, int], str] = {}
        for k in self.keys:
            if k.row >= len(self.rows) or k.col >= len(self.cols):
                raise ValueError(f"key {k.ref}: position ({k.row}, {k.col}) is outside the {len(self.rows)} x {len(self.cols)} matrix")
            pos = (k.row, k.col)
            if pos in seen:
                raise ValueError(f"keys {seen[pos]} and {k.ref} share matrix position {pos}")
            seen[pos] = k.ref
        return self

    def key(self, ref: str) -> KeyboardKey | None:
        return next((k for k in self.keys if k.ref == ref), None)

    def at(self, row: int, col: int) -> KeyboardKey | None:
        return next((k for k in self.keys if k.row == row and k.col == col), None)


class I2CBus(BaseModel):
    """An I2C bus: its nets, rail, pull-ups and the parameter keys of its clock, rise-time limit and capacitances (module docstring)."""

    id: str
    sda: str
    scl: str
    #: the supply net the pull-ups go to (``+5V``)
    rail: str
    #: the pull-up resistor refs (one or more per line; their nets show which line each pulls)
    pullup_refs: list[str]
    #: ``ir.parameters`` key of the SCL clock (Hz)
    f_scl_key: str
    #: ``ir.parameters`` key of the rise-time limit (s) the bus must meet at that clock
    t_rise_max_key: str
    #: ``ir.parameters`` keys (F) whose sum is the capacitance of each line to ground: pins, copper, cable
    c_keys: list[str]
    #: ``ir.parameters`` key of the rail voltage (V)
    v_rail_key: str
    #: t_r is measured between these fractions of the rail (UM10204: 0.3 VDD to 0.7 VDD)
    v_lo_frac: float = 0.3
    v_hi_frac: float = 0.7
    provenance: Provenance = Field(default_factory=_unrecorded)

    @model_validator(mode="after")
    def _consistent(self) -> I2CBus:
        _check_id(self.id, "I2C bus id")
        what = f"I2C bus {self.id}"
        for label, n in (("sda", self.sda), ("scl", self.scl), ("rail", self.rail)):
            _check_name(n, f"{what}: {label}")
        if len({self.sda, self.scl, self.rail}) != 3:
            raise ValueError(f"{what}: sda, scl and rail must be three different nets")
        if not self.pullup_refs:
            raise ValueError(f"{what}: names no pull-up resistor")
        _unique(self.pullup_refs, f"{what}: pull-up ref")
        for key in [self.f_scl_key, self.t_rise_max_key, self.v_rail_key, *self.c_keys]:
            _check_name(key, f"{what}: parameter key")
        if not self.c_keys:
            raise ValueError(f"{what}: names no capacitance (a bus has at least its pins)")
        _unique(self.c_keys, f"{what}: capacitance key")
        if not (0.0 < self.v_lo_frac < self.v_hi_frac < 1.0):
            raise ValueError(f"{what}: thresholds must satisfy 0 < v_lo_frac < v_hi_frac < 1, got {self.v_lo_frac} / {self.v_hi_frac}")
        return self


class MatrixScenario(BaseModel):
    """Pressed keys, the one row driven active and the columns that must read pressed / unpressed (module docstring)."""

    id: str
    #: switch refs closed in this scenario (every other key switch is open)
    pressed: list[str]
    #: the row driven active (low for ``col2row``); every other row is high-impedance
    driven_row: int
    #: columns that must read pressed (a closed key in the driven row) and unpressed (no closed key there - a ghost would read pressed)
    expect_pressed: list[int]
    expect_released: list[int]
    #: ``ir.parameters`` keys: the scanner's input pull-up (ohm), its input-low maximum and input-high minimum (V),
    #: a closed contact's resistance (ohm) and the supply (V)
    r_pullup_key: str
    v_il_max_key: str
    v_ih_min_key: str
    r_contact_key: str
    v_supply_key: str
    #: why this scenario: the rectangle it closes, the ghost it would show without the diodes
    purpose: str = ""
    provenance: Provenance = Field(default_factory=_unrecorded)

    @model_validator(mode="after")
    def _consistent(self) -> MatrixScenario:
        _check_id(self.id, "matrix scenario id")
        what = f"matrix scenario {self.id}"
        if not self.pressed:
            raise ValueError(f"{what}: presses no key")
        _unique(self.pressed, f"{what}: pressed key")
        if self.driven_row < 0:
            raise ValueError(f"{what}: driven_row must be >= 0")
        if not (self.expect_pressed or self.expect_released):
            raise ValueError(f"{what}: expects nothing")
        both = set(self.expect_pressed) & set(self.expect_released)
        if both:
            raise ValueError(f"{what}: columns {sorted(both)} are expected both pressed and released")
        if any(c < 0 for c in self.expect_pressed + self.expect_released):
            raise ValueError(f"{what}: column indices must be >= 0")
        for key in (self.r_pullup_key, self.v_il_max_key, self.v_ih_min_key, self.r_contact_key, self.v_supply_key):
            _check_name(key, f"{what}: parameter key")
        return self


class KeyboardDesign(BaseModel):
    """``ir.keyboard``: the build, the key matrix, the I2C buses, the matrix scenarios and the model values (module docstring)."""

    #: the build this board is (``main_half`` / ``secondary_half``)
    build: str
    matrix: KeyMatrix
    i2c: list[I2CBus] = Field(default_factory=list)
    scenarios: list[MatrixScenario] = Field(default_factory=list)
    #: the ``kb.model.*`` parameter keys that no datasheet or measurement grounds
    model_values: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=_unrecorded)

    @model_validator(mode="after")
    def _consistent(self) -> KeyboardDesign:
        _check_name(self.build, "keyboard build")
        _unique([b.id for b in self.i2c], "I2C bus id")
        _unique([s.id for s in self.scenarios], "matrix scenario id")
        refs = {k.ref for k in self.matrix.keys}
        for s in self.scenarios:
            unknown = [r for r in s.pressed if r not in refs]
            if unknown:
                raise ValueError(f"matrix scenario {s.id}: pressed {unknown} are not keys of the matrix")
            if s.driven_row >= len(self.matrix.rows):
                raise ValueError(f"matrix scenario {s.id}: driven_row {s.driven_row} is outside the {len(self.matrix.rows)} rows")
            out = [c for c in s.expect_pressed + s.expect_released if c >= len(self.matrix.cols)]
            if out:
                raise ValueError(f"matrix scenario {s.id}: columns {out} are outside the {len(self.matrix.cols)} columns")
        for key in self.model_values:
            if not isinstance(key, str) or not key.startswith(MODEL_PREFIX) or len(key) <= len(MODEL_PREFIX):
                raise ValueError(f"model value key {key!r} must start with {MODEL_PREFIX!r} (a modelling number no datasheet or measurement grounds)")
        _unique(self.model_values, "model value key")
        return self

    def bus(self, id: str) -> I2CBus | None:
        return next((b for b in self.i2c if b.id == id), None)

    def scenario(self, id: str) -> MatrixScenario | None:
        return next((s for s in self.scenarios if s.id == id), None)


__all__ = [
    "DIRECTIONS",
    "I2CBus",
    "ID_RE",
    "KeyMatrix",
    "KeyboardDesign",
    "KeyboardKey",
    "MODEL_PREFIX",
    "MatrixScenario",
]
