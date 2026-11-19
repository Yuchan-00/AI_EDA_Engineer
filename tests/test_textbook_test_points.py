"""Test points on the textbook schematic: each pad drawn on a label or power symbol of its net, in a block of its own, and nothing else changes.

Before: a ``Connector:TestPoint`` on a supply net or a driver net was a pin of that net for every motif - the supply chain
saw "a part off the line", the amplifier's follower loop saw a stray part - and the stage recipes failed; its label became a
root of its net in the router, stranding the net's other groups, whose name labels then overlapped, and the whole sheet fell
back to the block grid (the 9 V piano board with 13 test points, 2026-11-19).
"""

from __future__ import annotations

import pytest

from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.ir import Component, LibraryRef, Pin, PinElectricalType, PinRef, Provenance, ProvenanceKind
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from tests.test_schematic_compiler import netlist_nets
from tests.textbook_cases import op_amp_piano

LIB = KicadLibrary()
CLI = KicadCli()
HAS = LIB.symbol_file("Connector") is not None and LIB.symbol_file("Device") is not None
needs_libs = pytest.mark.skipif(not HAS, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not (HAS and CLI.available()), reason="KiCad CLI / libraries not installed")
PROV = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test point fixture")
#: the pads: the supplies, ground, and the driver's nodes inside U3B's follower loop
PADS = (("TP1", "VBAT"), ("TP2", "VDD"), ("TP3", "GND"), ("TP4", "DRV"), ("TP5", "SPK"), ("TP6", "VREF"))


def _with_pads(ir):
    libref = LIB.resolve_symbol(LibraryRef(library="Connector", name="TestPoint"))
    for ref, net in PADS:
        ir.components.append(Component(ref=ref, value=net, symbol=libref, provenance=PROV,
                                       pins=[Pin(number="1", name="1", electrical_type=PinElectricalType.PASSIVE, provenance=PROV)]))
        next(n for n in ir.nets if n.name == net).pins.append(PinRef(component_ref=ref, pin_number="1"))
    # as a design file may: the pads named in the blocks of the stages they measure
    blocks = {b.id: b for b in ir.topology.blocks}
    blocks["power"].component_refs += ["TP1", "TP2", "TP3"]
    blocks["driver"].component_refs += ["TP4", "TP5"]
    blocks["reference"].component_refs += ["TP6"]
    return ir


def _layout(notes: list[str]) -> str:
    return next(n for n in notes if n.startswith("layout:"))


@needs_libs
def test_test_points_get_a_block_of_their_own_and_every_stage_keeps_its_recipe(tmp_path):
    plain = SchematicCompiler().compile(op_amp_piano(tmp_path / "a", LIB), CompileContext(workdir=tmp_path / "a", tools={"kicad_library": LIB}))
    ir = _with_pads(op_amp_piano(tmp_path / "b", LIB))
    art = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "b", tools={"kicad_library": LIB}))
    head = _layout(art.notes)
    assert "fallbacks: none" in head, head
    assert "test_points=test_points" in head, head
    recipes = lambda h: h.split("blocks ", 1)[1].split(";", 1)[0].split(", ")  # noqa: E731
    assert [r for r in recipes(head) if not r.startswith("test_points=")] == recipes(_layout(plain.notes))
    assert any("block test_points: test point(s) TP1, TP2, TP3, TP4, TP5, TP6" in n for n in art.notes), art.notes


@needs_kicad
def test_each_test_point_is_on_its_net_and_erc_is_clean(tmp_path):
    ir = _with_pads(op_amp_piano(tmp_path, LIB))
    art = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    from pathlib import Path

    path = Path(art.path)
    got = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    for ref, net in PADS:
        assert (ref, "1") in {tuple(p) for p in got[net]}, (ref, net, got.get(net))
    want = {n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets}
    assert {n: {tuple(p) for p in pins} for n, pins in got.items() if n in want} == want
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert erc.details["errors"] == [], erc.details["errors"]


def _sheet(path):
    from ai_eda.tools.kicad import sexpr

    tree = sexpr.parse_file(path)
    wires = {tuple(tuple(round(float(v), 2) for v in pt[1:3]) for pt in sexpr.find(w, "pts")[1:]) for w in sexpr.find_all(tree, "wire")}
    labels = {(str(lab[1]), tuple(round(float(v), 2) for v in sexpr.find(lab, "at")[1:3])) for lab in sexpr.find_all(tree, "global_label")}
    symbols = []
    for s in sexpr.find_all(tree, "symbol"):
        if sexpr.find(s, "lib_id") is None:
            continue
        props = {str(p[1]): str(p[2]) for p in sexpr.find_all(s, "property")}
        symbols.append((str(sexpr.find(s, "lib_id")[1]), props.get("Reference"), props.get("Value"),
                        tuple(round(float(v), 2) for v in sexpr.find(s, "at")[1:3])))
    return wires, labels, symbols


@needs_libs
def test_pads_leave_every_wire_and_label_of_the_drawing_as_it_was(tmp_path):
    """Before: each pad's stub was also a hint of the test points' drawing, so the router took it for a group of the net to
    join - it wired the stub across the sheet or stranded the net's other groups by labels, and the nets competing with it
    changed too (the piano board: a 1.27 mm step at R38's VREF node, SUM wired round the summer instead of named)."""
    plain = SchematicCompiler().compile(op_amp_piano(tmp_path / "a", LIB), CompileContext(workdir=tmp_path / "a", tools={"kicad_library": LIB}))
    ir = _with_pads(op_amp_piano(tmp_path / "b", LIB))
    art = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path / "b", tools={"kicad_library": LIB}))
    pw, pl, ps = _sheet(plain.path)
    tw, tl, ts = _sheet(art.path)
    bottom = max(p[1] for w in pw for p in w)  # the pads hang in a row under the drawing
    assert {w for w in tw if all(p[1] <= bottom for p in w)} == pw
    assert {lab for lab in tl if lab[1][1] <= bottom} == pl
    pads = {at for _lib, ref, _v, at in ts if ref.startswith("TP")}
    assert all(at[1] > bottom for at in pads), pads
    # the parts' symbols stand where they stood (power symbols may be numbered on)
    assert {(lib, ref, at) for lib, ref, _v, at in ts if not ref.startswith(("TP", "#"))} == {(lib, ref, at) for lib, ref, _v, at in ps if not ref.startswith("#")}


@needs_libs
def test_a_pad_on_a_rail_carries_the_rails_power_symbol_and_no_power_flag(tmp_path):
    """Before: every pad had a global label of its net, also a ground or supply pad, where the rest of the sheet draws the
    rail by its power symbol; and a PWR_FLAG could go beside the lowest ground symbol, which a pad's would be."""
    ir = _with_pads(op_amp_piano(tmp_path, LIB))
    art = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    wires, labels, symbols = _sheet(art.path)
    row = min(at[1] for _lib, ref, _v, at in symbols if ref.startswith("TP")) - 5.08
    in_row = [(lib, value) for lib, ref, value, at in symbols if ref.startswith("#") and at[1] >= row]
    assert sorted(in_row) == sorted([("power:VCC", "VBAT"), ("power:VCC", "VDD"), ("power:GND", "GND")]), in_row
    assert not {net for net, at in labels if at[1] >= row} & {"VBAT", "VDD", "GND"}
    assert {net for net, at in labels if at[1] >= row} == {"DRV", "SPK", "VREF"}
