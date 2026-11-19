"""Footprint pads vs symbol pins: a mounting pad ``MP`` without an ``MP`` pin is mechanical, not a refusal.

The KiCad library numbers a part's mounting pins ``MP``: ``Device:R_Potentiometer_MountingPin`` has a pin ``MP``
("MountPin"), its plain sibling ``Device:R_Potentiometer`` has none, and the potentiometer footprints have two plated
``MP`` pads. One rule (:func:`ai_eda.compilers.pins.match_pads`) pairs pads with pins for the PCB compiler, the design
file (before anything is confirmed or simulated) and the RF parts table: ``MP`` pads of a symbol without the pin and
unnumbered copper pads are written without a net and named in the compile notes; any other numbered plated pad that is
no pin is still refused.

Pure tests use a synthetic library on disk; the last test compiles ``Device:R_Potentiometer`` on the real
``Potentiometer_Bourns_PTV09A-1_Single_Vertical`` footprint, routes it and runs kicad-cli DRC with schematic parity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler
from ai_eda.compilers.pins import MOUNTING_PAD, match_pads
from ai_eda.design.design_file import DesignFileError, load_design_file
from ai_eda.design.library_parts import TemplateRefusal, library_component
from ai_eda.design.rf.parts import check_compilable
from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, Net, PCBDesign, PinRef, Placement, ProjectMeta, Provenance, ProvenanceKind
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad.library import Pad, SymbolDef, SymbolPin
from ai_eda.tools.kicad.sexpr import Q, S
from ai_eda.tools.routing import route_board
from tests.test_circuit_templates import _footprint, _lib, _pin, _symbol, template_library
from tests.test_design_file import RC

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")
REAL = KicadLibrary()
CLI = KicadCli()
PTV09 = ("Potentiometer_THT", "Potentiometer_Bourns_PTV09A-1_Single_Vertical")
needs_kicad = pytest.mark.skipif(
    not CLI.available() or REAL.symbol_file("Device") is None or REAL.footprint_file(*PTV09) is None, reason="kicad-cli / KiCad libraries not installed"
)


def _pad(number: str, pad_type: str = "thru_hole", layers: tuple[str, ...] = ("*.Cu", "*.Mask")) -> Pad:
    return Pad(number, pad_type, "circle", 0.0, 0.0, 0.0, 1.8, 1.8, 1.0 if pad_type != "smd" else None, list(layers))


POT_PADS = [_pad("1"), _pad("2"), _pad("3"), _pad(MOUNTING_PAD), _pad(MOUNTING_PAD)]


# --------------------------------------------------------------------------- the rule


def test_mounting_pads_of_a_symbol_without_the_pin_are_mechanical():
    m = match_pads(["1", "2", "3"], POT_PADS)
    assert m.without_pad == m.extra == frozenset() and m.mounting == 2 and m.unnumbered == 0
    assert m.note("RV9") == "RV9: mounting pad(s) MP left without a net (mechanical)"


def test_mounting_pads_of_a_symbol_with_the_pin_are_its_pads():
    m = match_pads(["1", "2", "3", "MP"], POT_PADS)
    assert m.without_pad == m.extra == frozenset() and m.mounting == 0 and m.note("RV1") is None


def test_any_other_numbered_plated_pad_without_a_pin_is_extra():
    m = match_pads(["1", "2", "3"], [*POT_PADS, _pad("4"), _pad("5", "smd", ("F.Cu", "F.Mask")), _pad("6", "np_thru_hole", ("*.Cu", "*.Mask"))])
    assert m.extra == {"4", "5"} and m.without_pad == frozenset()  # a numbered NPTH hole is never copper, so never a pin's
    assert match_pads(["1", "2", "3", "4"], POT_PADS).without_pad == {"4"}


def test_unnumbered_copper_pads_are_counted_and_paste_apertures_and_holes_are_not():
    lugs = [_pad(""), _pad(""), _pad("", "smd", ("F.Cu", "F.Mask")), _pad("", "smd", ("F.Paste",)), _pad("", "np_thru_hole", ("*.Cu", "*.Mask"))]
    m = match_pads(["1", "2", "3"], [_pad("1"), _pad("2"), _pad("3"), *lugs])
    assert m.extra == frozenset() and m.mounting == 0 and m.unnumbered == 3
    assert m.note("SW9") == "SW9: 3 unnumbered pad(s) left without a net (mechanical)"
    both = match_pads(["1", "2", "3"], [*POT_PADS, *lugs])
    assert both.note("X1") == "X1: mounting pad(s) MP and 3 unnumbered pad(s) left without a net (mechanical)"


def test_the_rf_parts_check_applies_the_same_rule():
    pins = [SymbolPin(n, "", "passive", 0.0, 0.0, 0.0, 2.54, 1, False) for n in ("1", "2", "3")]
    pot = SymbolDef("Device:R_Potentiometer", "R_Potentiometer", ["symbol", "R_Potentiometer"], pins, [1], False, {})
    check_compilable("RV1", pot, POT_PADS, "Potentiometer_THT:X")  # MP: mechanical
    with pytest.raises(TemplateRefusal, match=r"pads \['4'\] that are not pins of 'Device:R_Potentiometer'"):
        check_compilable("RV1", pot, [*POT_PADS, _pad("4")], "Potentiometer_THT:X")


def test_the_rf_parts_check_refuses_a_symbol_pin_without_a_pad():
    pins = [SymbolPin(n, "", "passive", 0.0, 0.0, 0.0, 2.54, 1, False) for n in ("1", "2", "3", "4")]
    four = SymbolDef("Device:X4", "X4", ["symbol", "X4"], pins, [1], False, {})
    with pytest.raises(TemplateRefusal, match=r"symbol pins \['4'\] of 'Device:X4' have no pad in footprint Potentiometer_THT:X"):
        check_compilable("U1", four, POT_PADS, "Potentiometer_THT:X")  # the MP pads stand in for no pin


# --------------------------------------------------------------------------- a synthetic library on disk


def _pot_footprint(name: str, extra: list) -> list:
    node = _footprint(name, 3, tht=True)
    node[-1:-1] = extra  # before (embedded_fonts)
    return node


def _tht_pad(number: str, x: float, y: float, size: float = 2.0) -> list:
    return S("pad", Q(number), "thru_hole", "circle", S("at", x, y), S("size", size, size), S("drill", size / 2), S("layers", Q("*.Cu"), Q("*.Mask")))


def _pot_library(root: Path) -> KicadLibrary:
    """:func:`template_library` plus ``Pot:R_Potentiometer`` (pins 1-3), ``Pot:R_Potentiometer_MountingPin`` (1-3 and MP) and
    three ``Pot_THT`` footprints: two MP pads, two unnumbered lugs (and a paste-only aperture), and a fourth numbered pad."""
    template_library(root)
    three = lambda: [_pin("1", "1", -5.08, 0), _pin("2", "2", 0, 270, y=5.08), _pin("3", "3", 5.08, 180)]  # noqa: E731
    pots = _lib(_symbol("R_Potentiometer", "RV", three(), "Pot_THT:Pot_MP"),
                _symbol("R_Potentiometer_MountingPin", "RV", [*three(), _pin("MP", "MountPin", 0, 90, y=-5.08)], "Pot_THT:Pot_MP"))
    (root / "symbols" / "Pot.kicad_sym").write_text(sexpr.dumps(pots), encoding="utf-8")
    pretty = root / "footprints" / "Pot_THT.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    paste = S("pad", Q(""), "smd", "rect", S("at", 2.54, 3.0), S("size", 0.5, 0.5), S("layers", Q("F.Paste")))
    for name, extra in (("Pot_MP", [_tht_pad("MP", -3.0, 6.0, 3.0), _tht_pad("MP", 8.0, 6.0, 3.0)]),
                        ("Pot_Lugs", [_tht_pad("", -3.0, 6.0), _tht_pad("", 8.0, 6.0), paste]),
                        ("Pot_4", [_tht_pad("4", 2.54, 6.0)])):
        sexpr.dump_file(_pot_footprint(name, extra), pretty / f"{name}.kicad_mod")
    return KicadLibrary(roots=[root])


def _pot(lib: KicadLibrary, ref: str, symbol: str, footprint: str):
    return library_component(lib, ref, "10k", "pot", ("Pot", symbol), ("Pot_THT", footprint), USER, ["req.x"])


def _board(tmp_path: Path, lib: KicadLibrary, parts: list[tuple[str, str, str]], nets: dict[str, list[tuple[str, str]]]) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="pots", name="pots", workdir=str(tmp_path)))
    ir.components = [_pot(lib, *p) for p in parts]
    ir.nets = [Net(name=name, pins=[PinRef(component_ref=r, pin_number=n) for r, n in pins], provenance=USER) for name, pins in nets.items()]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0 * len(parts), height_mm=30.0),
                       placements=[Placement(component_ref=r, x_mm=10.0 + 30.0 * i, y_mm=10.0, side=BoardSide.TOP, provenance=USER) for i, (r, _, _) in enumerate(parts)])
    return ir


def _pads(tree: list, ref: str) -> list[list]:
    fp = next(f for f in sexpr.find_all(tree, "footprint")
              if any(p[1] == "Reference" and str(p[2]) == ref for p in sexpr.find_all(f, "property")))
    return sexpr.find_all(fp, "pad")


def _net(pad: list) -> str | None:
    net = sexpr.find(pad, "net")
    return None if net is None else str(net[1])


def test_the_pcb_compiler_writes_mechanical_pads_without_a_net_and_names_them(tmp_path: Path):
    lib = _pot_library(tmp_path / "lib")
    ir = _board(tmp_path, lib, [("RV1", "R_Potentiometer", "Pot_MP"), ("RV2", "R_Potentiometer_MountingPin", "Pot_MP"), ("RV3", "R_Potentiometer", "Pot_Lugs")],
                {"A": [("RV1", "1"), ("RV2", "1"), ("RV3", "1")], "W": [("RV1", "2"), ("RV2", "2"), ("RV3", "2")],
                 "GND": [("RV1", "3"), ("RV2", "3"), ("RV2", "MP"), ("RV3", "3")]})
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    assert art.notes == ["RV1: mounting pad(s) MP left without a net (mechanical)", "RV3: 2 unnumbered pad(s) left without a net (mechanical)"]
    tree = sexpr.parse_file(Path(art.path))
    rv1, rv2, rv3 = (_pads(tree, r) for r in ("RV1", "RV2", "RV3"))
    assert [(str(p[1]), _net(p)) for p in rv1] == [("1", "A"), ("2", "W"), ("3", "GND"), ("MP", None), ("MP", None)]
    assert all(sexpr.find(p, "pintype") is None for p in rv1 if str(p[1]) == MOUNTING_PAD)  # no pin, no pin type
    # the symbol with the pin keeps today's behaviour: both MP pads carry the pin's net and type
    assert [(str(p[1]), _net(p), str(sexpr.find(p, "pintype")[1])) for p in rv2 if str(p[1]) == MOUNTING_PAD] == [("MP", "GND", "passive")] * 2
    assert [(str(p[1]), _net(p)) for p in rv3] == [("1", "A"), ("2", "W"), ("3", "GND"), ("", None), ("", None), ("", None)]


def test_the_pcb_compiler_still_refuses_any_other_pad_without_a_pin(tmp_path: Path):
    lib = _pot_library(tmp_path / "lib")
    ir = _board(tmp_path, lib, [("RV1", "R_Potentiometer", "Pot_4")], {"A": [("RV1", "1"), ("RV1", "3")]})
    with pytest.raises(CompileError, match=r"'RV1': footprint Pot_THT:Pot_4 has pads \['4'\] that are not IR pins \(pins: \['1', '2', '3'\]\)"):
        PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))


def _design(**parts) -> dict:
    data = copy.deepcopy(RC)
    for ref, (symbol, footprint) in parts.items():
        data["components"].append({"ref": ref, "value": "10k", "symbol": symbol, "footprint": footprint,
                                   "spice": {"exclude": True, "exclude_reason": "panel control, not simulated"}})
        data["nets"][0]["pins"].append(f"{ref}.1")
    return data


def _load(tmp_path: Path, data: dict):
    path = tmp_path / "design.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_design_file(path, _pot_library(tmp_path / "lib"))


def test_a_design_file_whose_footprint_pads_are_not_the_symbols_pins_is_refused_before_confirmation(tmp_path: Path):
    with pytest.raises(DesignFileError, match=r"--design: RV1: footprint Pot_THT:Pot_4 has pads \['4'\] that are not pins of 'Pot:R_Potentiometer' "
                                              r"\(the PCB compiler refuses it\)"):
        _load(tmp_path, _design(RV1=("Pot:R_Potentiometer", "Pot_THT:Pot_4")))
    with pytest.raises(DesignFileError, match=r"--design: J2: symbol pins \['3'\] of 'Connector_Generic:Conn_01x03' have no pad in footprint "
                                              r"Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"):
        _load(tmp_path, _design(J2=("Connector_Generic:Conn_01x03", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")))


def test_a_design_file_with_mounting_and_unnumbered_pads_is_built(tmp_path: Path):
    plan = _load(tmp_path, _design(RV1=("Pot:R_Potentiometer", "Pot_THT:Pot_MP"), RV2=("Pot:R_Potentiometer", "Pot_THT:Pot_Lugs"),
                                   RV3=("Pot:R_Potentiometer_MountingPin", "Pot_THT:Pot_MP")))
    assert [c.payload.ref for c in plan.changes if c.target == "components"] == ["R1", "C1", "J1", "RV1", "RV2", "RV3"]


# --------------------------------------------------------------------------- real KiCad


@needs_kicad
def test_a_potentiometer_on_the_ptv09_footprint_passes_drc_and_parity_with_its_mounting_pads_unconnected(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="pot_mp", name="pot_mp", workdir=str(tmp_path)))
    ir.components = [
        library_component(REAL, "RV1", "10k", "volume", ("Device", "R_Potentiometer"), PTV09, USER, ["req.volume"]),
        library_component(REAL, "J1", "Conn_01x03", "panel", ("Connector_Generic", "Conn_01x03"),
                          ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical"), USER, ["req.volume"]),
    ]
    ir.nets = [Net(name=name, pins=[PinRef(component_ref="J1", pin_number=pin), PinRef(component_ref="RV1", pin_number=pin)], provenance=USER)
               for name, pin in (("CW", "1"), ("WIPER", "2"), ("CCW", "3"))]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0, height_mm=24.0), placements=[
        Placement(component_ref="J1", x_mm=6.0, y_mm=8.0, side=BoardSide.TOP, provenance=USER),
        Placement(component_ref="RV1", x_mm=14.0, y_mm=15.0, side=BoardSide.TOP, provenance=USER),
    ])
    routing = route_board(ir, REAL)
    assert routing.unrouted == {} and set(routing.stats["net_order"]) == {"CW", "WIPER", "CCW"}  # the MP pads are no net to route
    assert {t.net for t in routing.tracks} == {"CW", "WIPER", "CCW"}
    ir.pcb.tracks, ir.pcb.vias = routing.tracks, routing.vias
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": REAL})
    SchematicCompiler().compile(ir, ctx)
    art = PCBCompiler().compile(ir, ctx)
    assert art.notes == ["RV1: mounting pad(s) MP left without a net (mechanical)"]
    assert [(str(p[1]), _net(p)) for p in _pads(sexpr.parse_file(Path(art.path)), "RV1")] == [
        ("1", "CW"), ("2", "WIPER"), ("3", "CCW"), ("MP", None), ("MP", None)]
    drc = CLI.run_drc(Path(art.path), tmp_path / "drc.json", schematic_parity=True)
    assert drc.details["schematic_parity_checked"], drc.details.get("schematic_parity_reason")
    assert drc.details["errors"] == [], drc.details["errors"]
    assert drc.details["unconnected_items"] == [] and drc.details["schematic_parity"] == [], drc.details
