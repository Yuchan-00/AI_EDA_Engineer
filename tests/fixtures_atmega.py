"""A synthetic KiCad library with every part the ``atmega128_devboard`` template uses (and the ones the other templates use).

Offline tests cannot read the real KiCad libraries, so this module writes a
small, faithful stand-in into a temporary directory:

* ``MCU_Microchip_ATmega:ATmega128-16A`` - 64 pins with the real ATmega128
  names, numbers and electrical types (``1 ~{PEN}`` input, ``2..9 PE0..PE7``,
  ``10..17 PB0..PB7``, ``18 PG3``, ``19 PG4``, ``20 ~{RESET}`` input, ``21 VCC``
  / ``22 GND`` / ``64 AVCC`` power_in, ``23 XTAL2`` output, ``24 XTAL1`` input,
  ``25..32 PD0..PD7``, ``33 PG0``, ``34 PG1``, ``35..42 PC0..PC7``, ``43 PG2``,
  ``44..51 PA7..PA0``, ``52 VCC`` / ``53 GND`` / ``62 AREF`` / ``63 GND``
  passive, the ports bidirectional) and the real pin positions. In the real
  library pins 52 / 53 / 63 are *hidden* and stacked on 21 / 22; here they
  get their own positions unless ``stacked_power_pins=True`` reproduces the
  real stacking (the schematic compiler accepts a stack whose pins the IR
  puts in one net and draws one stub per point; the default keeps one label
  per pin). ``rename`` renames pins (by name) to
  test the template's refusal;
* its footprint ``Package_QFP:TQFP-64_14x14mm_P0.8mm``: 64 SMD pads, four
  sides of 16 at 0.8 mm pitch, pad 1.5 x 0.45 mm (0.45 x 1.5 mm on the top /
  bottom rows), pad centres at +/-7.6625 mm like the real one;
* ``Regulator_Linear:L7805`` (``1 IN`` / ``2 GND`` / ``3 OUT``),
  ``Connector:Barrel_Jack_Switch`` (pins 1..3 with empty names, as in KiCad),
  ``Device:R`` / ``C`` / ``C_Polarized`` / ``L`` / ``Crystal`` / ``LED`` /
  ``D`` (LED and diode ``1 = K``, ``2 = A``), ``Switch:SW_Push``, the
  ``Connector_Generic`` headers ``Conn_01x02`` / ``01x03`` / ``01x04`` /
  ``01x05`` / ``01x08`` / ``02x03_Odd_Even`` and every footprint the
  templates name, with pad geometry copied from the KiCad 10.0.6 footprints
  (TO-220, barrel jack, DO-41, CP radial, HC-49, 6 mm switch, 0603 / 0805,
  2.54 mm headers).

Nothing here is a measured KiCad fact beyond what the template's docstring
records; it is a fixture.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S as SX

#: (number, name, electrical type, x, y, angle) of the ATmega128-16A pins as the KiCad 10.0.6 symbol has them
ATMEGA128_PINS: tuple[tuple[str, str, str, float, float, float], ...] = (
    ("1", "~{PEN}", "input", -15.24, 22.86, 0),
    *[(str(2 + k), f"PE{k}", "bidirectional", -15.24, -25.4 - 2.54 * k, 0) for k in range(8)],
    *[(str(10 + k), f"PB{k}", "bidirectional", 15.24, 20.32 - 2.54 * k, 180) for k in range(8)],
    ("18", "PG3", "bidirectional", -15.24, 10.16, 0),
    ("19", "PG4", "bidirectional", -15.24, 7.62, 0),
    ("20", "~{RESET}", "input", -15.24, 43.18, 0),
    ("21", "VCC", "power_in", 0.0, 50.8, 270),
    ("22", "GND", "power_in", 0.0, -50.8, 90),
    ("23", "XTAL2", "output", -15.24, 33.02, 0),
    ("24", "XTAL1", "input", -15.24, 38.1, 0),
    *[(str(25 + k), f"PD{k}", "bidirectional", 15.24, -25.4 - 2.54 * k, 180) for k in range(8)],
    ("33", "PG0", "bidirectional", -15.24, 17.78, 0),
    ("34", "PG1", "bidirectional", -15.24, 15.24, 0),
    *[(str(35 + k), f"PC{k}", "bidirectional", 15.24, -2.54 - 2.54 * k, 180) for k in range(8)],
    ("43", "PG2", "bidirectional", -15.24, 12.7, 0),
    *[(str(44 + k), f"PA{7 - k}", "bidirectional", 15.24, 25.4 + 2.54 * k, 180) for k in range(8)],
    ("52", "VCC", "passive", 0.0, 50.8, 270),
    ("53", "GND", "passive", 0.0, -50.8, 90),
    *[(str(54 + k), f"PF{7 - k}", "bidirectional", -15.24, -20.32 + 2.54 * k, 0) for k in range(8)],
    ("62", "AREF", "passive", -15.24, 27.94, 0),
    ("63", "GND", "passive", 0.0, -50.8, 90),
    ("64", "AVCC", "power_in", 2.54, 50.8, 270),
)
#: the pins the real symbol stacks (hidden) on 21 / 22, and where the fixture puts them when not stacked
STACKED_PINS: dict[str, tuple[float, float]] = {"52": (-2.54, 50.8), "53": (-2.54, -50.8), "63": (2.54, -50.8)}


def _effects() -> list:
    return SX("effects", SX("font", SX("size", 1.27, 1.27)))


def _prop(key: str, value: str, hide: bool = False, at: tuple[float, float] = (0.0, 0.0), justify: tuple[str, ...] = ()) -> list:
    effects = SX("effects", SX("font", SX("size", 1.27, 1.27)), SX("justify", *justify) if justify else None)
    return SX("property", Q(key), Q(value), SX("at", at[0], at[1], 0), SX("hide", True) if hide else None, effects)


#: Reference / Value field positions (library frame, Y up) and justification: right of the origin, above / below it, left-justified
DEFAULT_FIELDS: tuple[tuple[tuple[float, float], tuple[str, ...]], tuple[tuple[float, float], tuple[str, ...]]] = (((2.54, 1.27), ("left",)), ((2.54, -1.27), ("left",)))
#: the ATmega128-16A's own field positions in KiCad 10.0.6: Reference above the body's top-left corner, Value below the body
MCU_FIELDS: tuple[tuple[tuple[float, float], tuple[str, ...]], tuple[tuple[float, float], tuple[str, ...]]] = (((-12.7, 49.53), ("left", "bottom")), ((2.54, -49.53), ("left", "top")))


def _pin(number: str, name: str, x: float, y: float, angle: float, etype: str = "passive", hidden: bool = False) -> list:
    return SX(
        "pin", etype, "line", SX("at", x, y, angle), SX("length", 2.54), SX("hide", "yes") if hidden else None,
        SX("name", Q(name), _effects()), SX("number", Q(number), _effects()),
    )


def _symbol(name: str, ref_prefix: str, pins: list, *, body: tuple[float, float] = (2.54, 1.016), description: str | None = None, footprint: str = "", fields=DEFAULT_FIELDS) -> list:
    w, h = body
    graphics = SX("symbol", Q(f"{name}_0_1"), SX("rectangle", SX("start", -w, h), SX("end", w, -h), SX("stroke", SX("width", 0.254), SX("type", "default")), SX("fill", SX("type", "none"))))
    (ref_at, ref_justify), (value_at, value_justify) = fields
    return SX(
        "symbol", Q(name), SX("pin_names", SX("offset", 1.016)), SX("exclude_from_sim", False), SX("in_bom", True), SX("on_board", True),
        _prop("Reference", ref_prefix, at=ref_at, justify=ref_justify), _prop("Value", name, at=value_at, justify=value_justify),
        _prop("Footprint", footprint, True), _prop("Datasheet", "~", True), _prop("Description", description or name, True),
        graphics, SX("symbol", Q(f"{name}_1_1"), *pins), SX("embedded_fonts", False),
    )


def _lib(*symbols: list) -> list:
    return SX("kicad_symbol_lib", SX("version", 20251024), SX("generator", Q("kicad_symbol_editor")), SX("generator_version", Q("10.0")), *symbols)


def _two(n1: str = "", n2: str = "") -> list:
    """A two-pin symbol's pins (horizontal, 1 on the left); the names are the library's (empty strings for R / C / C_Polarized, as KiCad 10.0.6 writes them)."""
    return [_pin("1", n1, -5.08, 0, 0), _pin("2", n2, 5.08, 0, 180)]


def _row(n: int) -> list:
    """A 1xN connector's pins ``Pin_1`` .. down the left side."""
    return [_pin(str(k + 1), f"Pin_{k + 1}", -5.08, 2.54 * (n - 1) / 2 - 2.54 * k, 0) for k in range(n)]


def _odd_even(rows: int) -> list:
    """A 2xN odd/even connector: odd pins on the left, even on the right."""
    out = []
    for r in range(rows):
        y = 2.54 * (rows - 1) / 2 - 2.54 * r
        out += [_pin(str(2 * r + 1), f"Pin_{2 * r + 1}", -5.08, y, 0), _pin(str(2 * r + 2), f"Pin_{2 * r + 2}", 5.08, y, 180)]
    return out


def _mcu(stacked_power_pins: bool, rename: dict[str, str], extra_pins: tuple = (), pin_angles: dict[str, float] | None = None) -> list:
    pins = []
    for number, name, etype, x, y, angle in (*ATMEGA128_PINS, *extra_pins):
        hidden = number in STACKED_PINS and stacked_power_pins
        if number in STACKED_PINS and not stacked_power_pins:
            x, y = STACKED_PINS[number]
        pins.append(_pin(number, rename.get(name, name), x, y, (pin_angles or {}).get(number, angle), etype, hidden))
    return _symbol(
        "ATmega128-16A", "U", pins, body=(12.7, 48.26), footprint="Package_QFP:TQFP-64_14x14mm_P0.8mm",
        description="16MHz, 128kB Flash, 4kB SRAM, 4kB EEPROM, JTAG, TQFP-64", fields=MCU_FIELDS,
    )


# --------------------------------------------------------------------------- footprints


def _smd(number: str, x: float, y: float, w: float, h: float) -> list:
    return SX("pad", Q(number), "smd", "roundrect", SX("at", x, y), SX("size", w, h), SX("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste")), SX("roundrect_rratio", 0.25))


def _tht(number: str, x: float, y: float, w: float, h: float, drill: float, shape: str = "circle") -> list:
    return SX("pad", Q(number), "thru_hole", shape, SX("at", x, y), SX("size", w, h), SX("drill", drill), SX("layers", Q("*.Cu"), Q("*.Mask")))


def _footprint(name: str, pads: list, courtyard: tuple[float, float, float, float], tht: bool) -> list:
    x1, y1, x2, y2 = courtyard
    return SX(
        "footprint", Q(name), SX("version", 20260206), SX("generator", Q("pcbnew")), SX("layer", Q("F.Cu")), SX("descr", Q(name)), SX("attr", "through_hole" if tht else "smd"),
        SX("fp_rect", SX("start", x1, y1), SX("end", x2, y2), SX("stroke", SX("width", 0.05), SX("type", "solid")), SX("fill", "no"), SX("layer", Q("F.CrtYd"))),
        *pads, SX("embedded_fonts", False),
    )


def qfp64_pads(pitch: float = 0.8, centre: float = 7.6625, long: float = 1.5, short: float = 0.45) -> list:
    """64 pads, 16 per side, counter-clockwise from the top of the left side (pin 1), as KiCad's TQFP-64."""
    pads = []
    half = pitch * 7.5
    for k in range(16):
        pads.append(_smd(str(1 + k), -centre, -half + pitch * k, long, short))
        pads.append(_smd(str(17 + k), -half + pitch * k, centre, short, long))
        pads.append(_smd(str(33 + k), centre, half - pitch * k, long, short))
        pads.append(_smd(str(49 + k), half - pitch * k, -centre, short, long))
    return sorted(pads, key=lambda p: int(p[1]))


def _two_smd(pitch: float, w: float, h: float) -> list:
    return [_smd("1", -pitch / 2, 0, w, h), _smd("2", pitch / 2, 0, w, h)]


def _header_pads(n: int, cols: int = 1) -> list:
    pads = []
    for k in range(n):
        row, col = (k // cols, k % cols)
        pads.append(_tht(str(k + 1), 2.54 * col, 2.54 * row, 1.7, 1.7, 1.0, "rect" if k == 0 else "circle"))
    return pads


def _header(n: int, cols: int = 1) -> tuple[list, tuple[float, float, float, float], bool]:
    rows = n // cols
    return _header_pads(n, cols), (-1.77, -1.77, 1.77 + 2.54 * (cols - 1), 1.77 + 2.54 * (rows - 1)), True


#: library -> footprint name -> (pads, courtyard, tht)
FOOTPRINTS: dict[str, dict[str, tuple[list, tuple[float, float, float, float], bool]]] = {
    "Package_QFP": {"TQFP-64_14x14mm_P0.8mm": (qfp64_pads(), (-8.65, -8.65, 8.65, 8.65), False)},
    "Package_TO_SOT_THT": {
        "TO-220-3_Vertical": ([_tht("1", 0, 0, 1.905, 2.0, 1.1, "rect"), _tht("2", 2.54, 0, 1.905, 2.0, 1.1, "oval"), _tht("3", 5.08, 0, 1.905, 2.0, 1.1, "oval")], (-2.71, -3.4, 7.79, 1.5), True),
        "TO-92_Inline": ([_tht(str(k + 1), 1.27 * k, 0, 1.05, 1.5, 0.75, "oval") for k in range(3)], (-1.53, -2.73, 4.07, 2.01), True),
    },
    "Connector_BarrelJack": {
        "BarrelJack_Horizontal": ([_tht("1", 0, 0, 3.5, 3.5, 1.0, "rect"), _tht("2", -6.0, 0, 3.0, 3.5, 1.0), _tht("3", -3.0, 4.7, 3.5, 3.5, 3.0)], (-14.0, -4.75, 2.0, 6.75), True),
    },
    "Diode_THT": {"D_DO-41_SOD81_P10.16mm_Horizontal": ([_tht("1", 0, 0, 2.2, 2.2, 1.1, "rect"), _tht("2", 10.16, 0, 2.2, 2.2, 1.1)], (-1.35, -1.6, 11.51, 1.6), True)},
    "Capacitor_THT": {
        "CP_Radial_D5.0mm_P2.50mm": ([_tht("1", 0, 0, 1.6, 1.6, 0.8, "rect"), _tht("2", 2.5, 0, 1.6, 1.6, 0.8)], (-1.5, -2.75, 4.0, 2.75), True),
        "C_Disc_D5.0mm_W2.5mm_P5.00mm": ([_tht("1", 0, 0, 1.6, 1.6, 0.8), _tht("2", 5.0, 0, 1.6, 1.6, 0.8)], (-1.05, -1.5, 6.05, 1.5), True),
    },
    "Resistor_THT": {"R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal": ([_tht("1", 0, 0, 1.6, 1.6, 0.8), _tht("2", 7.62, 0, 1.6, 1.6, 0.8)], (-1.05, -1.5, 8.67, 1.5), True)},
    "Crystal": {"Crystal_HC49-4H_Vertical": ([_tht("1", 0, 0, 1.5, 1.5, 0.8), _tht("2", 4.88, 0, 1.5, 1.5, 0.8)], (-3.59, -2.83, 8.47, 2.83), True)},
    "Button_Switch_THT": {"SW_PUSH_6mm": ([_tht("1", 0, 0, 2.0, 2.0, 1.1), _tht("1", 6.5, 0, 2.0, 2.0, 1.1), _tht("2", 0, 4.5, 2.0, 2.0, 1.1), _tht("2", 6.5, 4.5, 2.0, 2.0, 1.1)], (-1.5, -1.5, 8.0, 6.0), True)},
    "Capacitor_SMD": {"C_0603_1608Metric": (_two_smd(1.55, 0.9, 0.95), (-1.48, -0.73, 1.48, 0.73), False)},
    "Resistor_SMD": {
        "R_0603_1608Metric": (_two_smd(1.65, 0.8, 0.95), (-1.48, -0.73, 1.48, 0.73), False),
        "R_0805_2012Metric": (_two_smd(1.825, 1.025, 1.4), (-1.68, -0.95, 1.68, 0.95), False),
    },
    "Inductor_SMD": {"L_0805_2012Metric": (_two_smd(2.125, 0.875, 1.2), (-1.75, -0.85, 1.75, 0.85), False)},
    "LED_SMD": {
        "LED_0603_1608Metric": (_two_smd(1.575, 0.8, 0.95), (-1.48, -0.73, 1.48, 0.73), False),
        "LED_0805_2012Metric": (_two_smd(1.875, 0.975, 1.4), (-1.68, -0.95, 1.68, 0.95), False),
    },
    "Connector_PinHeader_2.54mm": {
        "PinHeader_1x02_P2.54mm_Vertical": _header(2),
        "PinHeader_1x03_P2.54mm_Vertical": _header(3),
        "PinHeader_1x04_P2.54mm_Vertical": _header(4),
        "PinHeader_1x05_P2.54mm_Vertical": _header(5),
        "PinHeader_1x08_P2.54mm_Vertical": _header(8),
        "PinHeader_2x03_P2.54mm_Vertical": _header(6, 2),
    },
}


def atmega_library(
    root: Path,
    *,
    stacked_power_pins: bool = False,
    rename: dict[str, str] | None = None,
    regulator_pin_names: tuple[str, str, str] = ("IN", "GND", "OUT"),
    omit_symbols: tuple[str, ...] = (),
    extra_pins: tuple[tuple[str, str, str, float, float, float], ...] = (),
    pin_angles: dict[str, float] | None = None,
) -> KicadLibrary:
    """Write the synthetic library under ``root`` (``symbols/`` + ``footprints/``) and return a :class:`KicadLibrary` on it.

    ``rename`` maps an ATmega128 pin name to the name the fixture writes
    (``{"PA0": "PA0_AD0"}``: a library whose pin the template cannot find);
    ``extra_pins`` adds MCU pins ``(number, name, type, x, y, angle)`` after
    the 64 (a symbol with a pin the template does not wire, or a third VCC);
    ``pin_angles`` turns MCU pins by number (a stacked pin pointing another
    way); ``regulator_pin_names`` renames the L7805's pins 1..3;
    ``omit_symbols`` leaves ``"Lib:Name"`` entries out (a library without a
    part).
    """
    rename = dict(rename or {})
    in_, gnd, out = regulator_pin_names
    symbols: dict[str, list[list]] = {
        "MCU_Microchip_ATmega": [_mcu(stacked_power_pins, rename, extra_pins, pin_angles)],
        "Regulator_Linear": [_symbol("L7805", "U", [_pin("1", in_, -7.62, 0, 0, "power_in"), _pin("2", gnd, 0, -7.62, 90, "power_in"), _pin("3", out, 7.62, 0, 180, "power_out")], body=(5.08, 3.81))],
        "Connector": [
            _symbol("Barrel_Jack", "J", [_pin("1", "", 5.08, 2.54, 180), _pin("2", "", 5.08, -2.54, 180)], body=(2.54, 3.81)),
            _symbol("Barrel_Jack_Switch", "J", [_pin("1", "", 5.08, 2.54, 180), _pin("2", "", 5.08, -2.54, 180), _pin("3", "", 5.08, 0, 180)], body=(2.54, 3.81)),
        ],
        "Device": [
            _symbol("R", "R", _two()), _symbol("C", "C", _two()), _symbol("C_Polarized", "C", _two()), _symbol("L", "L", _two("1", "2")),
            _symbol("Crystal", "Y", _two("1", "2")), _symbol("LED", "D", _two("K", "A")), _symbol("D", "D", _two("K", "A")),
        ],
        "Switch": [_symbol("SW_Push", "SW", _two("1", "2"))],
        "Transistor_BJT": [_symbol("2N3904", "Q", [_pin("1", "E", 2.54, -5.08, 90), _pin("2", "B", -5.08, 0, 0, "input"), _pin("3", "C", 2.54, 5.08, 270)])],
        "Connector_Generic": [
            _symbol("Conn_01x02", "J", _row(2)), _symbol("Conn_01x03", "J", _row(3)), _symbol("Conn_01x04", "J", _row(4), body=(2.54, 5.08)),
            _symbol("Conn_01x05", "J", _row(5), body=(2.54, 6.35)), _symbol("Conn_01x08", "J", _row(8), body=(2.54, 10.16)),
            _symbol("Conn_02x03_Odd_Even", "J", _odd_even(3), body=(2.54, 3.81)),
        ],
    }
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    for lib, entries in symbols.items():
        kept = [s for s in entries if f"{lib}:{s[1]}" not in omit_symbols]
        (root / "symbols" / f"{lib}.kicad_sym").write_text(sexpr.dumps(_lib(*kept)), encoding="utf-8")
    for lib, entries in FOOTPRINTS.items():
        pretty = root / "footprints" / f"{lib}.pretty"
        pretty.mkdir(parents=True, exist_ok=True)
        for name, (pads, courtyard, tht) in entries.items():
            sexpr.dump_file(_footprint(name, pads, courtyard, tht), pretty / f"{name}.kicad_mod")
    return KicadLibrary(roots=[root])


__all__ = ["ATMEGA128_PINS", "FOOTPRINTS", "STACKED_PINS", "atmega_library", "qfp64_pads"]
