"""Multi-unit symbols (a dual op-amp): the pins of each unit, the unit UUIDs, one schematic instance per unit on every layout
path, the SPICE binding of a multi-unit part, and real KiCad evidence (ERC, netlist, DRC schematic parity) for the
MCP6002-xSN board of :func:`tests.textbook_cases.dual_amplifier`."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler
from ai_eda.compilers.ids import pin_uuid, stable_uuid, symbol_uuid, unit_uuid
from ai_eda.compilers.pins import load_verified_symbol, unit_pins
from ai_eda.compilers.schematic import _unit_view
from ai_eda.compilers.schematic_layout import symbol_extent
from ai_eda.compilers.spice import build as spice_build
from ai_eda.compilers.textbook import pipeline
from ai_eda.compilers.textbook.glyphs import cell, field_boxes, measure
from ai_eda.compilers.textbook.model import Kind, LayoutFailure, Orient, Place, load_model, unit_letter
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.parts import check_compilable
from ai_eda.errors import CompileError
from ai_eda.ir import Component, LibraryRef, Provenance, ProvenanceKind, SpiceBinding, SpiceDevice, user_requirement
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad.library import SymbolDef, SymbolPin
from tests.test_schematic_compiler import netlist_nets
from tests.textbook_cases import dual_amplifier
from tests.textbook_corpus import grid_board

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Amplifier_Operational") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Amplifier_Operational") is None, reason="KiCad CLI/libraries not installed")
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")
OPAMP = "Amplifier_Operational:MCP6002-xSN"


def _pin(number: str, unit: int) -> SymbolPin:
    return SymbolPin(number, "", "passive", 0.0, 0.0, 0.0, 2.54, unit, False)


def _symbols(node) -> list:
    return [s for s in sexpr.find_all(node, "symbol") if str(sexpr.get(s, "lib_id")) == OPAMP]


def _part(lib: str, name: str, ref: str = "X1") -> Component:
    return Component(ref=ref, value=name, symbol=LIB.resolve_symbol(LibraryRef(library=lib, name=name)), provenance=USER)


# --------------------------------------------------------------------------- pure


def test_unit_pins_are_a_units_own_and_common_pins_in_library_order():
    dual = SymbolDef("L:D", "D", ["symbol", "D"], [_pin("1", 1), _pin("5", 2), _pin("2", 1), _pin("4", 3), _pin("8", 3)], [1, 2, 3], False, {})
    assert [p.number for p in unit_pins(dual, 1)] == ["1", "2"]
    assert [p.number for p in unit_pins(dual, 2)] == ["5"]
    assert [p.number for p in unit_pins(dual, 3)] == ["4", "8"]
    common = SymbolDef("L:C", "C", ["symbol", "C"], [_pin("1", 0), _pin("2", 0)], [1], False, {})
    assert [p.number for p in unit_pins(common, 1)] == ["1", "2"]  # a single-unit symbol: every pin, unit 0 included


def test_unit_uuids_are_stable_and_distinct_from_the_symbol_uuid():
    assert unit_uuid("p", "U1", 2) == stable_uuid("p", "symbol", "U1", "unit", 2) == unit_uuid("p", "U1", 2)
    ids = {symbol_uuid("p", "U1"), unit_uuid("p", "U1", 2), unit_uuid("p", "U1", 3), unit_uuid("q", "U1", 2), unit_uuid("p", "U2", 2)}
    assert len(ids) == 5


def test_unit_letters():
    assert [unit_letter(u) for u in (1, 2, 3, 5, 26, 27)] == ["A", "B", "C", "E", "Z", "AA"]


# --------------------------------------------------------------------------- the library and the compilers


@needs_libs
def test_multi_unit_symbols_are_loaded_only_when_asked_and_common_pins_are_refused():
    dual = _part("Amplifier_Operational", "MCP6002-xSN")
    with pytest.raises(CompileError, match=r"has units \[1, 2, 3\]; multi-unit symbols are not supported yet"):
        load_verified_symbol(dual, LIB)
    assert load_verified_symbol(dual, LIB, multi_unit=True).units == [1, 2, 3]
    assert load_verified_symbol(_part("Amplifier_Operational", "LM358"), LIB, multi_unit=True).units == [1, 2, 3]
    split = _part("Device", "D_Dual_CommonAnode_AKK_Split", "D1")  # units [1, 2], the common anode pin 1 in unit 0
    with pytest.raises(CompileError, match=re.escape("D1: Device:D_Dual_CommonAnode_AKK_Split has common (unit 0) pins ['1']; "
                                                     "multi-unit symbols with common pins are not supported")):
        load_verified_symbol(split, LIB, multi_unit=True)
    for flag in (False, True):  # a single-unit symbol whose pins are all unit 0 is unaffected
        assert load_verified_symbol(_part("Switch", "SW_Push", "SW1"), LIB, multi_unit=flag).units == [1]


@needs_libs
def test_every_unit_is_one_instance_with_its_uuid_and_only_its_pins(tmp_path):
    ir = dual_amplifier(tmp_path, LIB)
    pid = ir.project.id
    node = SchematicCompiler("legacy").build(ir, LIB)
    units = _symbols(node)
    assert [int(sexpr.get(s, "unit")) for s in units] == [1, 2, 3]
    assert [str(sexpr.get(s, "uuid")) for s in units] == [symbol_uuid(pid, "U1"), unit_uuid(pid, "U1", 2), unit_uuid(pid, "U1", 3)]
    for s, numbers in zip(units, (["1", "2", "3"], ["5", "6", "7"], ["4", "8"])):
        pins = sexpr.find_all(s, "pin")
        assert [str(p[1]) for p in pins] == numbers
        assert [str(sexpr.get(p, "uuid")) for p in pins] == [pin_uuid(pid, "U1", n) for n in numbers]
        path = sexpr.find(sexpr.find(sexpr.find(s, "instances"), "project"), "path")
        assert str(sexpr.get(path, "reference")) == "U1" and sexpr.get(path, "unit") == sexpr.get(s, "unit")
    lib_symbols = sexpr.find(node, "lib_symbols")
    assert [str(e[1]) for e in sexpr.find_all(lib_symbols, "symbol")].count(OPAMP) == 1  # embedded once
    r1 = next(s for s in sexpr.find_all(node, "symbol") if any(str(p[2]) == "R1" for p in sexpr.find_all(s, "property") if str(p[1]) == "Reference"))
    assert int(sexpr.get(r1, "unit")) == 1 and str(sexpr.get(r1, "uuid")) == symbol_uuid(pid, "R1")


@needs_libs
def test_the_legacy_table_has_one_cell_per_unit_measured_on_its_own_unit(tmp_path):
    ir = dual_amplifier(tmp_path, LIB)
    compiler = SchematicCompiler("legacy")
    drawn, flags, _ = compiler._power_flags(ir, LIB)
    components = [*compiler._ordered(ir, compiler._checked_components(ir)), *flags]
    pin_nets = compiler._pin_net_names(drawn)
    placed = compiler._place(components, LIB, pin_nets)
    assert list(placed) == ["J1", "R1", "R2", "U1:1", "U1:2", "U1:3", "#FLG01", "#FLG02"]
    assert [placed[k].unit for k in ("U1:1", "U1:2", "U1:3")] == [1, 2, 3] and placed["R1"].unit == 1
    assert len({(ps.x, ps.y) for ps in placed.values()}) == len(placed)
    symbol = placed["U1:3"].symbol
    power, whole = symbol_extent(_unit_view(symbol, 3), {}), symbol_extent(symbol, {})
    assert power.xmax - power.xmin < whole.xmax - whole.xmin  # the power unit's cell holds its two pins, not the amplifier triangle
    assert _unit_view(placed["R1"].symbol, 1) is placed["R1"].symbol
    for key in ("U1:1", "U1:2", "U1:3"):  # each cell is the extent of its own unit: its graphics, its pins, its pins' labels
        ps = placed[key]
        labels = {p.number: len(pin_nets[("U1", p.number)]) for p in unit_pins(symbol, ps.unit) if ("U1", p.number) in pin_nets}
        assert ps.extent == symbol_extent(_unit_view(symbol, ps.unit), labels).shifted(ps.x, ps.y), key


@needs_libs
def test_the_legacy_table_wires_the_nets_of_every_unit(tmp_path):
    """The table router lets a pin's first wire leave through its own unit's cell (keyed ``ref:unit``), so every net of the
    dual board is drawn as wires, not left to labels."""
    _, notes = SchematicCompiler("legacy").build_with_notes(dual_amplifier(tmp_path, LIB), LIB)
    assert notes[0] == "wiring: 6 of 6 net(s) drawn as wires (labels kept on every pin)", notes


@needs_libs
def test_the_textbook_policy_compiles_a_multi_unit_design_as_version_1_0(tmp_path):
    """Under the textbook policy a design with a multi-unit symbol is drawn by the textbook 1.0 layout: the artifact says
    1.0 and every unit is written."""
    ir = dual_amplifier(tmp_path / "ir", LIB)
    artifact = SchematicCompiler("textbook").compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": LIB}))
    assert artifact.generator_version == "1.0"
    assert artifact.notes[0].startswith("layout: textbook 1.0;")
    # changed with the default switch: 0.9 is deleted, so no note explains a dispatch past it any more
    assert not any("0.9" in note for note in artifact.notes)
    node = sexpr.parse(Path(artifact.path).read_text(encoding="utf-8"))
    assert [int(sexpr.get(s, "unit")) for s in _symbols(node)] == [1, 2, 3]


@needs_libs
def test_the_parts_table_check_refuses_only_multi_unit_symbols_with_common_pins():
    def load(lib: str, name: str, fp_lib: str, fp_name: str):
        symbol = LIB.load_symbol(LIB.resolve_symbol(LibraryRef(library=lib, name=name)))
        return symbol, LIB.load_footprint(LIB.resolve_footprint(LibraryRef(library=fp_lib, name=fp_name))).pads, f"{fp_lib}:{fp_name}"

    check_compilable("U1", *load("Amplifier_Operational", "MCP6002-xSN", "Package_SO", "SOIC-8_3.9x4.9mm_P1.27mm"))  # no common pins
    with pytest.raises(TemplateRefusal, match=re.escape("has units [1, 2] and common (unit 0) pins; multi-unit symbols with common (unit 0) pins")):
        check_compilable("D1", *load("Device", "D_Dual_CommonAnode_AKK_Split", "Package_TO_SOT_SMD", "SOT-23"))


@needs_libs
def test_the_textbook_model_has_one_glyph_per_unit(tmp_path):
    model = load_model(SchematicCompiler(), dual_amplifier(tmp_path, LIB), LIB)
    units = {k: g for k, g in model.glyphs.items() if g.ref == "U1"}
    assert list(units) == ["U1:1", "U1:2", "U1:3"]
    assert [g.kind for g in units.values()] == [Kind.AMP, Kind.AMP, Kind.PWRUNIT]
    assert [g.shown_ref for g in units.values()] == ["U1A", "U1B", "U1C"] and model.glyphs["R1"].shown_ref == "R1"
    assert [tuple(p.number for p in units[k].amp_io()) for k in ("U1:1", "U1:2")] == [("1", "3", "2"), ("7", "5", "6")]
    assert units["U1:3"].net == {"4": "GND", "8": "VCC"} and model.glyph_of("U1", "6").key == "U1:2"


# --------------------------------------------------------------------------- SPICE (outside the layout)


@needs_libs
def test_spice_binds_a_multi_unit_part_as_one_element_over_every_units_pins(tmp_path):
    """spice.py sees the part, not its units: one binding per ref, whose element takes the pins of every unit (a dual
    subcircuit). A binding over one unit's pins is refused with the existing unused-pin message."""
    ir = dual_amplifier(tmp_path, LIB)
    card = ".subckt DUALOPAMP out1 inn1 inp1 vneg inp2 inn2 out2 vpos\nE1 out1 0 inp1 inn1 100k\nE2 out2 0 inp2 inn2 100k\n.ends DUALOPAMP"
    for c in ir.components:
        if c.ref in ("R1", "R2"):
            c.spice = SpiceBinding(device=SpiceDevice.R, value=user_requirement(1e4 if c.ref == "R1" else 1e5, "ohm"), provenance=USER)
        elif c.ref == "J1":
            c.spice = SpiceBinding(exclude=True, exclude_reason="connector: no SPICE meaning", provenance=USER)
        else:
            c.spice = SpiceBinding(device=SpiceDevice.X, model_name="DUALOPAMP", model_card=user_requirement(card),
                                   pin_order=[str(n) for n in range(1, 9)], provenance=USER)
    assert "XU1 OUT SUM 0 0 0 SPARE SPARE VCC DUALOPAMP\n" in spice_build(ir)
    ir.component("U1").spice = SpiceBinding(device=SpiceDevice.X, model_name="ONE", pin_order=["3", "2", "1"], provenance=USER,
                                            model_card=user_requirement(".subckt ONE inp inn out\nE1 out 0 inp inn 100k\n.ends ONE"))
    with pytest.raises(CompileError, match=r"U1\.5 is in a net but the SPICE binding does not use it"):
        spice_build(ir)


# --------------------------------------------------------------------------- real KiCad evidence


def _forced_table(monkeypatch):
    def fail(ctx, fallbacks):
        raise LayoutFailure("forced")
    monkeypatch.setattr(pipeline, "_textbook", fail)
    return SchematicCompiler().build_with_notes


PATHS = {
    "textbook": lambda mp: SchematicCompiler().build_with_notes,
    "table": _forced_table,
    "legacy": lambda mp: SchematicCompiler("legacy").build_with_notes,
    "textbook policy": lambda mp: SchematicCompiler("textbook").build_with_notes,
}
FIRST = {
    "textbook": "generated by compiler.kicad_sch 1.0 textbook from IR project dual_amplifier",
    "table": "generated by compiler.kicad_sch 1.0 textbook (table fallback) from IR project dual_amplifier",
    "legacy": "generated by compiler.kicad_sch 0.8 from IR project dual_amplifier",
    "textbook policy": "generated by compiler.kicad_sch 1.0 textbook from IR project dual_amplifier",
}


def _board(ir, node, workdir: Path) -> Path:
    """Compile the board of ``ir`` and write ``node`` as its sibling schematic; the board path."""
    pcb = Path(PCBCompiler().compile(ir, CompileContext(workdir=workdir, tools={"kicad_library": LIB})).path)
    pcb.with_suffix(".kicad_sch").write_text(sexpr.dumps(node), encoding="utf-8", newline="\n")
    return pcb


def _tstamps(net: Path, ref: str) -> list[str]:
    comp = next(c for c in sexpr.find_all(sexpr.find(sexpr.parse_file(net), "components"), "comp") if str(sexpr.get(c, "ref")) == ref)
    return [str(t) for t in sexpr.find(comp, "tstamps")[1:]]


@needs_kicad
@pytest.mark.parametrize("path", list(PATHS))
def test_the_dual_board_passes_erc_netlist_and_parity_on_every_path(tmp_path, monkeypatch, path):
    ir = grid_board(dual_amplifier(tmp_path / "ir", LIB))
    node, notes = PATHS[path](monkeypatch)(ir, LIB)
    comment = next(c for c in sexpr.find_all(sexpr.find(node, "title_block"), "comment") if str(c[1]) == "1")
    assert str(comment[2]) == FIRST[path]
    if path == "table":
        assert "whole-sheet table" in notes[0]
    if path == "textbook policy":  # changed with the default switch: 0.9 is deleted, the policy reaches 1.0 directly
        assert notes[0].startswith("layout: textbook 1.0;") and not any("0.9" in note for note in notes)
    assert [int(sexpr.get(s, "unit")) for s in _symbols(node)] == [1, 2, 3]  # every unit placed
    pcb = _board(ir, node, tmp_path / "out")
    sch = pcb.with_suffix(".kicad_sch")
    erc = CLI.run_erc(sch, tmp_path / "erc.json")
    assert erc.details["errors"] == erc.details["warnings"] == [], erc.details
    net = CLI.export_netlist(sch, tmp_path / "dual.net", fmt="kicadsexpr")
    exported = netlist_nets(net)
    assert sorted({pin for pins in exported.values() for ref, pin in pins if ref == "U1"}, key=int) == [str(n) for n in range(1, 9)]
    assert len(_tstamps(net, "U1")) == 3  # one component, three unit instances
    want = {n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets}
    partition = lambda nets: {frozenset(p) for name, p in nets.items() if p and not name.startswith("unconnected-")}  # noqa: E731
    assert partition(exported) == partition(want) and set(want) <= set(exported)
    drc = CLI.run_drc(pcb, tmp_path / "drc.json", schematic_parity=True)
    assert drc.details["schematic_parity_checked"], drc.details.get("schematic_parity_reason")
    assert drc.details["schematic_parity"] == [], drc.details["schematic_parity"]


@needs_kicad
def test_parity_with_the_footprint_path_on_a_unit_the_netlist_does_not_list_first(tmp_path):
    """The measured answer behind :mod:`ai_eda.compilers.ids`: the lowest unit keeps ``symbol_uuid`` (the footprint path),
    the others get ``unit_uuid``. The netlist lists a ref's unit UUIDs in neither unit nor UUID order; parity holds with the
    path on a unit that is not listed first - and with a path no unit carries, so parity does not check the path."""
    ir = grid_board(dual_amplifier(tmp_path / "ir", LIB, project_id="dual_d"))
    pid = ir.project.id
    unit_of = {symbol_uuid(pid, "U1"): 1, unit_uuid(pid, "U1", 2): 2, unit_uuid(pid, "U1", 3): 3}
    pcb = _board(ir, SchematicCompiler("legacy").build(ir, LIB), tmp_path / "out")
    order = [unit_of[t] for t in _tstamps(CLI.export_netlist(pcb.with_suffix(".kicad_sch"), tmp_path / "d.net"), "U1")]
    assert order[0] != 1  # premise: the unit the footprint path points at is not the first-listed tstamp
    assert order != sorted(order) and order != [unit_of[u] for u in sorted(unit_of, reverse=True)]
    board = pcb.read_text(encoding="utf-8")
    assert f'(path "/{symbol_uuid(pid, "U1")}")' in board
    drc = CLI.run_drc(pcb, tmp_path / "drc.json", schematic_parity=True)
    assert drc.details["schematic_parity_checked"] and drc.details["schematic_parity"] == [], drc.details
    pcb.write_text(board.replace(f'(path "/{symbol_uuid(pid, "U1")}")', f'(path "/{stable_uuid(pid, "no such unit")}")'), encoding="utf-8", newline="\n")
    drc = CLI.run_drc(pcb, tmp_path / "drc2.json", schematic_parity=True)
    assert drc.details["schematic_parity_checked"] and drc.details["schematic_parity"] == [], drc.details


@needs_kicad
def test_the_reference_is_measured_with_the_unit_letter_kicad_draws(tmp_path):
    ir = dual_amplifier(tmp_path, LIB)
    node, _ = SchematicCompiler().build_with_notes(ir, LIB)
    sch = tmp_path / "dual_amplifier.kicad_sch"
    sch.write_text(sexpr.dumps(node), encoding="utf-8", newline="\n")
    subprocess.run([str(CLI.binary), "sch", "export", "svg", "-o", str(tmp_path / "svg"), str(sch)], capture_output=True, check=True)
    svg = (tmp_path / "svg" / "dual_amplifier.svg").read_text(encoding="utf-8")
    model = load_model(SchematicCompiler(), ir, LIB)
    for s in _symbols(node):
        g = model.glyphs[f"U1:{sexpr.get(s, 'unit')}"]
        props = {str(p[1]): p for p in sexpr.find_all(s, "property")}
        assert str(props["Reference"][2]) == "U1"  # the file keeps the ref; KiCad adds the letter
        at = sexpr.find(s, "at")
        place = Place(cell(sexpr.to_float(at[1])), cell(sexpr.to_float(at[2])), Orient(int(sexpr.to_float(at[3]))))
        anchors = tuple((cell(sexpr.to_float(sexpr.find(props[k], "at")[1])), cell(sexpr.to_float(sexpr.find(props[k], "at")[2]))) for k in ("Reference", "Value"))
        m = measure(g, place, anchors=anchors)
        plain = field_boxes(m.field_nodes, (place.x, place.y), place.orient)[0]
        strokes = svg.split(f"<desc>{g.shown_ref}</desc>", 1)[1].split("</g>", 1)[0]  # kicad-cli draws U1A / U1B / U1C as strokes
        xs, ys = zip(*((float(x), float(y)) for x, y in re.findall(r"(\d+\.\d+) (\d+\.\d+)", strokes)))
        box = m.fields[0]
        assert box[0] - 0.1 <= min(xs) and max(xs) <= box[2] + 0.1 and box[1] - 0.1 <= min(ys) and max(ys) <= box[3] + 0.1, (g.key, box, xs, ys)
        assert max(xs) - min(xs) > plain[2] - plain[0] + 0.5  # measuring the bare ref would miss the letter
