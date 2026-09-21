"""KiCad library loader tests.

The synthetic-library tests run everywhere; the ones marked ``needs_libs`` read
the real KiCad 10 libraries and are skipped when they are not installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ai_eda.errors import CompileError
from ai_eda.ir import LibraryRef, PinElectricalType
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import (
    KICAD_PIN_TYPES,
    BBox,
    KicadLibrary,
    LibraryFormatError,
    LibraryLookupError,
    Pad,
    SymbolPin,
    ir_pin_type_to_kicad,
    kicad_pin_type_to_ir,
)
from ai_eda.tools.kicad.sexpr import Q, S

_probe = KicadLibrary()
HAS_LIBS = _probe.symbol_file("Device") is not None and _probe.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")


@pytest.fixture(scope="module")
def lib() -> KicadLibrary:
    return KicadLibrary()


def ref(library: str, name: str) -> LibraryRef:
    return LibraryRef(library=library, name=name)


# --------------------------------------------------------------------------- pin type mapping


def test_pin_type_mapping_is_identity_over_the_12_kicad_types():
    assert len(KICAD_PIN_TYPES) == 12
    for t in PinElectricalType:
        assert kicad_pin_type_to_ir(t.value) is t
        assert ir_pin_type_to_kicad(t) == t.value
        assert ir_pin_type_to_kicad(t.value) == t.value
    with pytest.raises(LibraryFormatError):
        kicad_pin_type_to_ir("bogus_type")
    assert issubclass(LibraryFormatError, CompileError)
    assert issubclass(LibraryLookupError, LookupError)


# --------------------------------------------------------------------------- real libraries


R_PINS = [
    SymbolPin(number="1", name="", electrical_type="passive", x=0.0, y=3.81, angle=270.0, length=1.27, unit=1, hidden=False),
    SymbolPin(number="2", name="", electrical_type="passive", x=0.0, y=-3.81, angle=90.0, length=1.27, unit=1, hidden=False),
]
CONN_PINS = [
    SymbolPin(number="1", name="Pin_1", electrical_type="passive", x=-5.08, y=2.54, angle=0.0, length=3.81, unit=1, hidden=False),
    SymbolPin(number="2", name="Pin_2", electrical_type="passive", x=-5.08, y=0.0, angle=0.0, length=3.81, unit=1, hidden=False),
    SymbolPin(number="3", name="Pin_3", electrical_type="passive", x=-5.08, y=-2.54, angle=0.0, length=3.81, unit=1, hidden=False),
]
R0603_PADS = [
    Pad("1", "smd", "roundrect", -0.825, 0.0, 0.0, 0.8, 0.95, None, ["F.Cu", "F.Mask", "F.Paste"], 0.25),
    Pad("2", "smd", "roundrect", 0.825, 0.0, 0.0, 0.8, 0.95, None, ["F.Cu", "F.Mask", "F.Paste"], 0.25),
]
PINHEADER_PADS = [
    Pad("1", "thru_hole", "rect", 0.0, 0.0, 0.0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"], None),
    Pad("2", "thru_hole", "circle", 0.0, 2.54, 0.0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"], None),
    Pad("3", "thru_hole", "circle", 0.0, 5.08, 0.0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"], None),
]


@needs_libs
def test_device_r_symbol(lib: KicadLibrary):
    sym = lib.load_symbol(ref("Device", "R"))
    assert sym.lib_id == "Device:R" and sym.name == "R"
    assert sym.pins == R_PINS
    assert lib.symbol_pins(ref("Device", "R")) == R_PINS
    assert sym.units == [1] and not sym.is_power and sym.extends_from is None
    assert sym.properties["Reference"] == "R" and sym.properties["Footprint"] == ""
    assert sym.properties["ki_fp_filters"] == "R_*"
    assert [str(u[1]) for u in sexpr.find_all(sym.node, "symbol")] == ["R_0_1", "R_1_1"]
    assert sym.pin("2").ir_type is PinElectricalType.PASSIVE and sym.pin("9") is None
    assert sym.library_path.endswith("Device.kicad_sym")


@needs_libs
def test_conn_01x03_symbol(lib: KicadLibrary):
    sym = lib.load_symbol(ref("Connector_Generic", "Conn_01x03"))
    assert sym.pins == CONN_PINS
    assert sym.units == [1]
    assert [str(u[1]) for u in sexpr.find_all(sym.node, "symbol")] == ["Conn_01x03_1_1"]
    assert sexpr.get(sexpr.find(sym.node, "pin_names"), "offset") == "1.016"


@needs_libs
def test_lib_symbols_entry_is_renamed_copy_of_library_block(lib: KicadLibrary):
    sym = lib.load_symbol(ref("Device", "R"))
    entry = sym.lib_symbols_entry()
    assert entry[1] == "Device:R" and isinstance(entry[1], sexpr.QStr)
    assert entry is not sym.node and sym.node[1] == "R"
    # identical to the block in the fully parsed library apart from the name
    full = lib.symbol_library("Device")
    block = next(s for s in sexpr.find_all(full, "symbol") if s[1] == "R")
    entry[1] = Q("R")
    assert sexpr.strict_equal(entry, block)
    assert sexpr.strict_equal(sym.node, block)  # fast text extraction == full parse
    assert sexpr.dumps(sym.node) in sexpr.dumps(full).replace("\n\t", "\n")


@needs_libs
def test_power_symbol(lib: KicadLibrary):
    gnd = lib.load_symbol(ref("power", "GND"))
    assert gnd.is_power and gnd.properties["Value"] == "GND" and gnd.properties["Reference"] == "#PWR"
    assert len(gnd.pins) == 1 and gnd.pins[0].electrical_type == "power_in" and gnd.pins[0].length == 0.0


@needs_libs
def test_extends_symbol_resolves_to_self_contained_node(lib: KicadLibrary):
    child = lib.load_symbol(ref("Device", "Q_Photo_NPN_CE"))
    parent = lib.load_symbol(ref("Device", "Q_Photo_NPN"))
    assert child.extends_from == "Q_Photo_NPN" and parent.extends_from is None
    assert sexpr.find(child.node, "extends") is None
    assert child.pins == parent.pins and len(child.pins) >= 2
    assert [str(u[1]) for u in sexpr.find_all(child.node, "symbol")] == ["Q_Photo_NPN_CE_0_1", "Q_Photo_NPN_CE_1_1"]
    assert child.properties["Value"] == "Q_Photo_NPN_CE"
    assert child.properties["Description"] == "NPN phototransistor, collector/emitter"
    assert child.properties["Reference"] == "Q"
    # inherited, non-overridden parts come from the parent
    assert sexpr.strict_equal(sexpr.find(child.node, "pin_names"), sexpr.find(parent.node, "pin_names"))
    # the flattened node serialises and re-parses as a valid symbol
    again = sexpr.parse(sexpr.dumps(child.lib_symbols_entry()))
    assert again[1] == "Device:Q_Photo_NPN_CE" and sexpr.find(again, "extends") is None


@needs_libs
def test_multi_level_extends_chain(lib: KicadLibrary):
    if lib.symbol_file("Amplifier_Current") is None:
        pytest.skip("Amplifier_Current.kicad_sym not installed")
    leaf = lib.load_symbol(ref("Amplifier_Current", "INA281A2"))
    root = lib.load_symbol(ref("Amplifier_Current", "AD8211"))
    assert leaf.extends_from == "INA281A1"
    assert sexpr.find(leaf.node, "extends") is None
    assert leaf.pins == root.pins and len(leaf.pins) == 5
    assert leaf.properties["Value"] == "INA281A2"
    assert {str(u[1]) for u in sexpr.find_all(leaf.node, "symbol")} == {"INA281A2_0_1", "INA281A2_1_1"}


@needs_libs
def test_r0603_footprint(lib: KicadLibrary):
    fp = lib.load_footprint(ref("Resistor_SMD", "R_0603_1608Metric"))
    assert fp.lib_id == "Resistor_SMD:R_0603_1608Metric"
    assert fp.pads == R0603_PADS
    assert lib.footprint_pads(ref("Resistor_SMD", "R_0603_1608Metric")) == R0603_PADS
    assert fp.attr == "smd" and fp.attributes == ["smd"]
    assert fp.courtyard == BBox(-1.48, -0.73, 1.48, 0.73)
    assert fp.tags == "resistor" and fp.descr.startswith("Resistor SMD 0603")
    assert fp.properties["Reference"] == "REF**" and fp.properties["Value"] == "R_0603_1608Metric"
    assert sexpr.head(fp.node) == "footprint" and fp.node[1] == "R_0603_1608Metric"
    assert fp.pad("2").x == 0.825 and fp.pad("3") is None


@needs_libs
def test_pinheader_footprint(lib: KicadLibrary):
    fp = lib.load_footprint(ref("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical"))
    assert fp.pads == PINHEADER_PADS
    assert fp.attr == "through_hole"
    assert fp.courtyard == BBox(-1.77, -1.77, 1.77, 6.85)
    assert all(p.drill == 1.0 for p in fp.pads)


@needs_libs
def test_resolve_uses_real_parse(lib: KicadLibrary):
    r = lib.resolve_symbol(ref("Device", "R"))
    assert r.verified and r.library_path and Path(r.library_path).name == "Device.kicad_sym"
    assert not lib.resolve_symbol(ref("Device", "DefinitelyNotASymbol")).verified
    assert not lib.resolve_symbol(ref("NoSuchLib", "R")).verified
    f = lib.resolve_footprint(ref("Resistor_SMD", "R_0603_1608Metric"))
    assert f.verified and f.library_path.endswith("R_0603_1608Metric.kicad_mod")
    assert not lib.resolve_footprint(ref("Resistor_SMD", "R_9999")).verified
    with pytest.raises(LibraryLookupError):
        lib.load_symbol(ref("Device", "DefinitelyNotASymbol"))
    with pytest.raises(LibraryLookupError):
        lib.load_footprint(ref("Nope", "X"))


@needs_libs
def test_symbol_and_footprint_caches(lib: KicadLibrary):
    a = lib.load_symbol(ref("Device", "R"))
    assert lib.load_symbol(ref("Device", "R")) is a
    fp = lib.load_footprint(ref("Resistor_SMD", "R_0603_1608Metric"))
    assert lib.load_footprint(ref("Resistor_SMD", "R_0603_1608Metric")) is fp
    assert lib.symbol_library("Device") is lib.symbol_library("Device")


@needs_libs
def test_full_device_library_parse_time(lib: KicadLibrary):
    path = lib.symbol_file("Device")
    t0 = time.perf_counter()
    tree = sexpr.parse_file(path)
    dt = time.perf_counter() - t0
    print(f"\nparsed {path.name} ({path.stat().st_size / 1e6:.1f} MB) in {dt:.2f} s")
    assert dt < 15
    names = [str(s[1]) for s in sexpr.find_all(tree, "symbol")]
    assert "R" in names and len(names) > 100
    assert "R" in lib.symbol_names("Device")


# --------------------------------------------------------------------------- synthetic library (always runs)


def _effects() -> list:
    return S("effects", S("font", S("size", 1.27, 1.27)))


def _prop(key: str, value: str, hide: bool = False) -> list:
    return S("property", Q(key), Q(value), S("at", 0, 0, 0), S("hide", True) if hide else None, _effects())


def _pin(etype: str, number: str, name: str, x: float, y: float, angle: int) -> list:
    return S(
        "pin", etype, "line", S("at", x, y, angle), S("length", 2.54),
        S("name", Q(name), _effects()), S("number", Q(number), _effects()),
    )


def _base_symbol(name: str) -> list:
    return S(
        "symbol", Q(name),
        S("pin_names", S("offset", 1.016)),
        S("exclude_from_sim", False), S("in_bom", True), S("on_board", True),
        _prop("Reference", "U"), _prop("Value", name), _prop("Footprint", "", True),
        _prop("Datasheet", "", True), _prop("Description", "base part", True),
        S("symbol", Q(f"{name}_0_1"), S("rectangle", S("start", -2.54, 2.54), S("end", 2.54, -2.54),
                                        S("stroke", S("width", 0.254), S("type", "default")), S("fill", S("type", "none")))),
        S("symbol", Q(f"{name}_1_1"), _pin("input", "1", "IN", -5.08, 0, 0), _pin("output", "2", "OUT", 5.08, 0, 180)),
        S("symbol", Q(f"{name}_1_2"), _pin("input", "1", "IN", -5.08, 0, 0), _pin("output", "2", "OUT", 5.08, 0, 180)),
        S("symbol", Q(f"{name}_2_1"), _pin("power_in", "3", "VCC", 0, 5.08, 270), S("pin", "no_connect", "line",
                                        S("at", 0, -5.08, 90), S("length", 0), S("hide", True),
                                        S("name", Q("NC"), _effects()), S("number", Q("4"), _effects()))),
        S("embedded_fonts", False),
    )


def _derived(name: str, parent: str, **props: str) -> list:
    return S("symbol", Q(name), S("extends", Q(parent)), *[_prop(k, v) for k, v in props.items()], S("embedded_fonts", False))


def _synthetic_lib() -> list:
    return S(
        "kicad_symbol_lib", S("version", 20251024), S("generator", Q("kicad_symbol_editor")), S("generator_version", Q("10.0")),
        _base_symbol("Base"),
        _derived("Mid", "Base", Value="Mid", Description="mid part"),
        _derived("Leaf", "Mid", Value="Leaf", ki_keywords="leaf kw"),
        _derived("Loop1", "Loop2"), _derived("Loop2", "Loop1"),
        _derived("Orphan", "Missing"),
        S("symbol", Q("BadPin"), S("in_bom", True), _prop("Reference", "X"), _prop("Value", "BadPin"),
          S("symbol", Q("BadPin_1_1"), _pin("bogus_type", "1", "A", 0, 0, 0)), S("embedded_fonts", False)),
        S("symbol", Q("BadUnit"), S("in_bom", True), _prop("Reference", "X"), _prop("Value", "BadUnit"),
          S("symbol", Q("BadUnit_x"), _pin("passive", "1", "A", 0, 0, 0)), S("embedded_fonts", False)),
    )


def _synthetic_footprint() -> list:
    return S(
        "footprint", Q("FP"), S("version", 20260206), S("generator", Q("pcbnew")), S("layer", Q("F.Cu")),
        S("descr", Q("test fp")), S("tags", Q("t")), S("attr", "through_hole", "exclude_from_pos_files"),
        S("fp_poly", S("pts", S("xy", -1, -1), S("xy", 3, -1), S("xy", 3, 2), S("xy", -1, 2)),
          S("stroke", S("width", 0.05), S("type", "solid")), S("fill", "no"), S("layer", Q("F.CrtYd"))),
        S("fp_line", S("start", -9, -9), S("end", 9, 9), S("stroke", S("width", 0.1), S("type", "solid")), S("layer", Q("F.Fab"))),
        S("pad", Q("1"), "thru_hole", "oval", S("at", 0, 0, 90), S("size", 1.7, 2.2), S("drill", "oval", 1, 1.5),
          S("layers", Q("*.Cu"), Q("*.Mask")), S("remove_unused_layers", False)),
        S("pad", Q(""), "np_thru_hole", "circle", S("at", 2, 1), S("size", 3, 3), S("drill", 3), S("layers", Q("*.Cu"), Q("*.Mask"))),
        S("pad", Q("2"), "smd", "rect", S("at", 1, 1), S("size", 1, 1), S("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste"))),
        S("embedded_fonts", False),
    )


@pytest.fixture(params=["pretty", "compact"])
def synthetic(tmp_path: Path, request: pytest.FixtureRequest) -> KicadLibrary:
    """The same library written in KiCad layout (fast text path) and as one line (full-parse fallback)."""
    text = sexpr.dumps(_synthetic_lib())
    if request.param == "compact":
        text = " ".join(text.split())
    (tmp_path / "symbols").mkdir()
    (tmp_path / "symbols" / "Test.kicad_sym").write_text(text, encoding="utf-8")
    pretty = tmp_path / "footprints" / "Test.pretty"
    pretty.mkdir(parents=True)
    sexpr.dump_file(_synthetic_footprint(), pretty / "FP.kicad_mod")
    sexpr.dump_file(_synthetic_footprint(), pretty / "WrongName.kicad_mod")
    return KicadLibrary(roots=[tmp_path])


def test_synthetic_base_symbol(synthetic: KicadLibrary):
    base = synthetic.load_symbol(ref("Test", "Base"))
    assert base.units == [1, 2]
    assert [p.number for p in base.pins] == ["1", "2", "3", "4"]  # De Morgan body style 2 pins are not repeated
    assert base.pin("1").electrical_type == "input" and base.pin("1").ir_type is PinElectricalType.INPUT
    assert base.pin("3").unit == 2 and base.pin("4").hidden and not base.pin("3").hidden
    assert base.pin("4").length == 0.0 and base.pin("4").electrical_type == "no_connect"
    assert base.properties == {"Reference": "U", "Value": "Base", "Footprint": "", "Datasheet": "", "Description": "base part"}


def test_synthetic_extends_chain_is_flattened(synthetic: KicadLibrary):
    leaf = synthetic.load_symbol(ref("Test", "Leaf"))
    base = synthetic.load_symbol(ref("Test", "Base"))
    assert leaf.extends_from == "Mid"
    assert sexpr.find(leaf.node, "extends") is None
    assert leaf.pins == base.pins and leaf.units == [1, 2]
    assert [str(u[1]) for u in sexpr.find_all(leaf.node, "symbol")] == ["Leaf_0_1", "Leaf_1_1", "Leaf_1_2", "Leaf_2_1"]
    # override chain: Value from Leaf, Description from Mid, Reference from Base, new key appended after the last property
    assert list(leaf.properties.items()) == [
        ("Reference", "U"), ("Value", "Leaf"), ("Footprint", ""), ("Datasheet", ""),
        ("Description", "mid part"), ("ki_keywords", "leaf kw"),
    ]
    heads = [sexpr.head(c) for c in leaf.node if isinstance(c, list)]
    assert heads[: heads.index("symbol")][-6:] == ["property"] * 6  # properties stay contiguous, before the units
    assert str(sexpr.find_all(leaf.node, "property")[-1][1]) == "ki_keywords"  # new key appended last
    # the library file itself was not mutated by flattening
    raw = next(s for s in sexpr.find_all(synthetic.symbol_library("Test"), "symbol") if s[1] == "Base")
    assert sexpr.get(raw, "extends") is None and raw[1] == "Base"
    assert sexpr.strict_equal(sexpr.parse(sexpr.dumps(leaf.lib_symbols_entry())), leaf.lib_symbols_entry())


def test_synthetic_errors(synthetic: KicadLibrary):
    with pytest.raises(LibraryFormatError, match="circular"):
        synthetic.load_symbol(ref("Test", "Loop1"))
    with pytest.raises(LibraryFormatError, match="missing symbol"):
        synthetic.load_symbol(ref("Test", "Orphan"))
    with pytest.raises(LibraryFormatError, match="bogus_type"):
        synthetic.load_symbol(ref("Test", "BadPin"))
    with pytest.raises(LibraryFormatError, match="NAME_u_b"):
        synthetic.load_symbol(ref("Test", "BadUnit"))
    with pytest.raises(LibraryLookupError):
        synthetic.load_symbol(ref("Test", "Nope"))
    with pytest.raises(LibraryLookupError):
        synthetic.load_symbol(ref("Other", "Base"))
    assert not synthetic.resolve_symbol(ref("Test", "Nope")).verified
    assert synthetic.resolve_symbol(ref("Test", "Leaf")).verified
    with pytest.raises(LibraryFormatError, match="named 'FP'"):
        synthetic.load_footprint(ref("Test", "WrongName"))


def test_synthetic_footprint(synthetic: KicadLibrary):
    fp = synthetic.load_footprint(ref("Test", "FP"))
    assert fp.attr == "through_hole" and fp.attributes == ["through_hole", "exclude_from_pos_files"]
    assert fp.courtyard == BBox(-1, -1, 3, 2)  # the F.Fab line is ignored
    assert fp.pads == [
        Pad("1", "thru_hole", "oval", 0.0, 0.0, 90.0, 1.7, 2.2, 1.0, ["*.Cu", "*.Mask"], None),
        Pad("", "np_thru_hole", "circle", 2.0, 1.0, 0.0, 3.0, 3.0, 3.0, ["*.Cu", "*.Mask"], None),
        Pad("2", "smd", "rect", 1.0, 1.0, 0.0, 1.0, 1.0, None, ["F.Cu", "F.Mask", "F.Paste"], None),
    ]
    assert fp.descr == "test fp" and fp.tags == "t" and fp.properties == {}
    assert synthetic.resolve_footprint(ref("Test", "FP")).verified


def test_malformed_library_file_raises_format_error(tmp_path: Path):
    (tmp_path / "symbols").mkdir()
    (tmp_path / "symbols" / "Broken.kicad_sym").write_text('(kicad_symbol_lib (symbol "A" (extends "B")', encoding="utf-8")
    lib = KicadLibrary(roots=[tmp_path])
    with pytest.raises(LibraryFormatError):
        lib.load_symbol(ref("Broken", "A"))
    (tmp_path / "symbols" / "NotALib.kicad_sym").write_text('(kicad_pcb (symbol "A"))', encoding="utf-8")
    with pytest.raises(LibraryFormatError):
        lib.load_symbol(ref("NotALib", "A"))
