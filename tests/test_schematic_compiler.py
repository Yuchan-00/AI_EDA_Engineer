"""Schematic compiler tests.

The geometry / refusal tests run everywhere. The ones marked ``needs_libs``
read the real KiCad 10 libraries; the ones marked ``needs_kicad`` additionally
run the real ``kicad-cli`` (ERC + netlist export) - they are the proof that the
generated file is a KiCad schematic with the IR's connectivity, not a
plausible-looking text file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.ids import sheet_uuid, symbol_uuid
from ai_eda.compilers.schematic_layout import (
    grid_positions,
    label_orientation,
    natural_ref_key,
    pin_body_direction,
    pin_position,
    stub_end,
    transform_offset,
)
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, LibraryRef, Net, Pin, PinElectricalType, PinRef, Provenance, ProvenanceKind
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad.library import SymbolPin
from tests.fixtures_kicad import EXPECTED_NETS, PROJECT_ID, divider_with_connector_ir, ir_net_map

_probe = KicadLibrary()
HAS_LIBS = _probe.symbol_file("Device") is not None and _probe.symbol_file("Connector_Generic") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
_kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (_kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")

#: ERC warning types that can only come from a compiler defect (not from the design or the environment)
COMPILER_DEFECT_WARNINGS = {
    "lib_symbol_mismatch",
    "lib_symbol_issues",
    "unconnected_wire_endpoint",
    "endpoint_off_grid",
    "isolated_pin_label",
    "label_dangling",
    "global_label_dangling",
}


@pytest.fixture(scope="module")
def lib() -> KicadLibrary:
    return KicadLibrary()


@pytest.fixture
def ir(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    return divider_with_connector_ir(tmp_path, lib)


def compile_to(ir: CircuitIR, workdir: Path, lib: KicadLibrary):
    ref = SchematicCompiler().compile(ir, CompileContext(workdir=workdir, tools={"kicad_library": lib}))
    return ref, Path(ref.path)


def netlist_nets(path: Path) -> dict[str, set[tuple[str, str]]]:
    """Parse a ``kicadsexpr`` netlist into ``{net name: {(ref, pin)}}``."""
    export = sexpr.parse_file(path)
    nets = sexpr.find(export, "nets")
    assert nets is not None, "netlist has no (nets) block"
    out: dict[str, set[tuple[str, str]]] = {}
    for net in sexpr.find_all(nets, "net"):
        name = str(sexpr.get(net, "name"))
        out[name] = {(str(sexpr.get(n, "ref")), str(sexpr.get(n, "pin"))) for n in sexpr.find_all(net, "node")}
    return out


# --------------------------------------------------------------------------- pure geometry (no libraries needed)

R_PIN1 = SymbolPin("1", "", "passive", 0.0, 3.81, 270.0, 1.27, 1, False)
R_PIN2 = SymbolPin("2", "", "passive", 0.0, -3.81, 90.0, 1.27, 1, False)
J_PIN1 = SymbolPin("1", "Pin_1", "passive", -5.08, 2.54, 0.0, 3.81, 1, False)
J_PIN2 = SymbolPin("2", "Pin_2", "passive", -5.08, 0.0, 0.0, 3.81, 1, False)


def test_pin_formula_matches_validated_schematic():
    # values from the hand-validated div_v1.kicad_sch (ERC 0/0, netlist checked)
    assert pin_position(76.2, 63.5, 0, None, R_PIN1) == (76.2, 59.69)
    assert pin_position(76.2, 63.5, 0, None, R_PIN2) == (76.2, 67.31)
    assert pin_position(50.8, 76.2, 0, None, J_PIN1) == (45.72, 73.66)
    assert pin_position(50.8, 76.2, 0, None, J_PIN2) == (45.72, 76.2)
    assert stub_end((76.2, 59.69), pin_body_direction(0, None, 270.0)) == (76.2, 57.15)
    assert stub_end((45.72, 73.66), pin_body_direction(0, None, 0.0)) == (43.18, 73.66)
    assert label_orientation(pin_body_direction(0, None, 270.0)) == (90, "left")  # text goes up
    assert label_orientation(pin_body_direction(0, None, 90.0)) == (270, "right")  # text goes down
    assert label_orientation(pin_body_direction(0, None, 0.0)) == (180, "right")  # text goes left
    assert label_orientation(pin_body_direction(0, None, 180.0)) == (0, "left")


def test_pin_formula_rotations_and_mirrors_match_kicad_demos():
    # demos/simulation/rectifier: R1 (at 120.65 93.98 90) pin 1 -> (116.84, 93.98)
    assert pin_position(120.65, 93.98, 90, None, R_PIN1) == (116.84, 93.98)
    # demos/royalblue54L_feather: J1 Conn_01x02 (at 156.21 111.76 180): pin1 -> (161.29,111.76), pin2 (-5.08,-2.54) -> (161.29,109.22)
    assert pin_position(156.21, 111.76, 180, None, J_PIN2) == (161.29, 111.76)
    assert pin_position(156.21, 111.76, 180, None, SymbolPin("2", "", "passive", -5.08, -2.54, 0.0, 3.81, 1, False)) == (161.29, 109.22)
    # experimentally confirmed variants (netlist pin -> net): rot270 pin1 at X+3.81, rot180 pin1 at Y+3.81, mirror x flips y
    assert transform_offset(0, 3.81, 270) == (3.81, 0.0)
    assert transform_offset(0, 3.81, 180) == (0.0, 3.81)
    assert transform_offset(0, 3.81, 0, "x") == (0.0, 3.81)
    assert transform_offset(0, 3.81, 90, "y") == (3.81, 0.0)
    assert transform_offset(0, 3.81, 90, "x") == (-3.81, 0.0)
    assert transform_offset(-5.08, 2.54, 0, "y") == (5.08, -2.54)
    with pytest.raises(ValueError):
        transform_offset(0, 0, 45)


def test_grid_positions_are_deterministic_and_on_grid():
    pos = grid_positions(["R10", "R2", "J1", "R1"])
    assert list(pos) == ["J1", "R1", "R2", "R10"]
    assert pos == grid_positions(["R1", "J1", "R10", "R2"])
    for x, y in pos.values():
        assert abs(x / 2.54 - round(x / 2.54)) < 1e-9 and abs(y / 2.54 - round(y / 2.54)) < 1e-9
    assert natural_ref_key("R2") < natural_ref_key("R10") < natural_ref_key("U1")
    five = grid_positions([f"R{i}" for i in range(1, 6)], columns=4)
    assert five["R5"][1] == pytest.approx(five["R1"][1] + 25.4) and five["R5"][0] == five["R1"][0]


# --------------------------------------------------------------------------- compile against the real libraries


@needs_libs
def test_compile_writes_file_with_ir_hash(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    ref, path = compile_to(ir, tmp_path / "out", lib)
    assert path.exists() and path.name == f"{PROJECT_ID}.kicad_sch"
    assert ref.generated_from_ir_hash == ir.content_hash()
    assert ref.generator == SchematicCompiler.id and ref.matches_disk()
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text and text.endswith(")\n")
    assert f"generated by {SchematicCompiler.id} {SchematicCompiler.version}" in text  # provenance comment in title_block


@needs_libs
def test_compile_is_byte_deterministic(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    _, a = compile_to(ir, tmp_path / "a", lib)
    _, b = compile_to(divider_with_connector_ir(tmp_path, KicadLibrary()), tmp_path / "b", KicadLibrary())
    assert a.read_bytes() == b.read_bytes()


@needs_libs
def test_structure_uuids_and_embedded_library_symbols(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    _, path = compile_to(ir, tmp_path, lib)
    sch = sexpr.parse_file(path)
    assert sexpr.head(sch) == "kicad_sch" and sexpr.get(sch, "version") == "20260101"
    assert sexpr.get(sch, "uuid") == sheet_uuid(PROJECT_ID)
    # embedded lib_symbols are verbatim library copies renamed Lib:Name
    lib_symbols = sexpr.find(sch, "lib_symbols")
    entries = {str(s[1]): s for s in sexpr.find_all(lib_symbols, "symbol")}
    assert set(entries) == {"Device:R", "Connector_Generic:Conn_01x03"}
    for lib_id, entry in entries.items():
        library, name = lib_id.split(":")
        block = next(s for s in sexpr.find_all(lib.symbol_library(library), "symbol") if s[1] == name)
        entry = sexpr.deep_copy(entry)
        entry[1] = sexpr.Q(name)
        assert sexpr.strict_equal(entry, block), lib_id
    # symbol instances: stable uuids, instances path = /<root uuid>, footprint property, pins listed
    symbols = {str(sexpr.get(s, "property", 2)): s for s in sexpr.find_all(sch, "symbol")}
    assert set(symbols) == {"R1", "R2", "J1"}
    for ref, sym in symbols.items():
        assert sexpr.get(sym, "uuid") == symbol_uuid(PROJECT_ID, ref)
        inst = sexpr.find(sexpr.find(sexpr.find(sym, "instances"), "project"), "path")
        assert inst[1] == "/" + sheet_uuid(PROJECT_ID) and sexpr.get(inst, "reference") == ref
        props = {str(p[1]): str(p[2]) for p in sexpr.find_all(sym, "property")}
        assert props["Footprint"] == ("Resistor_SMD:R_0603_1608Metric" if ref.startswith("R") else "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical")
        assert all(sexpr.find(p, "effects") is not None for p in sexpr.find_all(sym, "property"))
    assert [str(p[1]) for p in sexpr.find_all(symbols["J1"], "pin")] == ["1", "2", "3"]
    # one wire + one global label per (net, pin); labels named after the nets, anchored at wire ends
    labels = sexpr.find_all(sch, "global_label")
    wires = sexpr.find_all(sch, "wire")
    assert len(labels) == len(wires) == sum(len(v) for v in EXPECTED_NETS.values())
    assert sorted(str(l[1]) for l in labels) == sorted(n for n, pins in EXPECTED_NETS.items() for _ in pins)
    wire_ends = {tuple(str(a) for a in xy[1:3]) for w in wires for xy in sexpr.find_all(sexpr.find(w, "pts"), "xy")}
    for label in labels:
        at = sexpr.find(label, "at")
        assert (str(at[1]), str(at[2])) in wire_ends
    assert not sexpr.find_all(sch, "no_connect") and not sexpr.find_all(sch, "junction")


@needs_libs
def test_no_connect_pin_emits_marker(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    j1 = ir.component("J1")
    j1.pins[2] = j1.pins[2].model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT})
    ir.net("GND").pins = [p for p in ir.net("GND").pins if p.component_ref != "J1"]
    _, path = compile_to(ir, tmp_path, lib)
    sch = sexpr.parse_file(path)
    ncs = sexpr.find_all(sch, "no_connect")
    assert len(ncs) == 1
    at = sexpr.find(ncs[0], "at")
    j1_pos = sexpr.find(next(s for s in sexpr.find_all(sch, "symbol") if sexpr.get(s, "property", 2) == "J1"), "at")
    assert (float(at[1]), float(at[2])) == pytest.approx((float(j1_pos[1]) - 5.08, float(j1_pos[2]) + 2.54))  # pin 3 end


# --------------------------------------------------------------------------- refusals


def _fake_net(name: str, *pins: tuple[str, str]) -> Net:
    return Net(name=name, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test"))


@needs_libs
def test_unverified_symbol_is_refused(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    r1 = ir.component("R1")
    r1.symbol = lib.resolve_symbol(LibraryRef(library="Device", name="DefinitelyNotASymbol"))
    assert not r1.symbol.verified
    with pytest.raises(CompileError, match="R1.*Device:DefinitelyNotASymbol.*not verified"):
        compile_to(ir, tmp_path, lib)
    assert not list(tmp_path.glob("*.kicad_sch"))
    # a ref that claims verified but is not on disk is refused too (the disk is the truth)
    r1.symbol = LibraryRef(library="Device", name="DefinitelyNotASymbol", verified=True)
    with pytest.raises(CompileError, match="R1.*not found"):
        compile_to(ir, tmp_path, lib)
    r1.symbol = None
    with pytest.raises(CompileError, match="R1: no KiCad symbol"):
        compile_to(ir, tmp_path, lib)


@needs_libs
def test_ir_pins_must_match_library_pins(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    r1 = ir.component("R1")
    r1.pins.append(Pin(number="3", name="ghost", provenance=r1.pins[0].provenance))
    with pytest.raises(CompileError, match=r"R1: IR pins \['1', '2', '3'\] do not match library symbol 'Device:R' pins \['1', '2'\]"):
        compile_to(ir, tmp_path, lib)
    r1.pins = r1.pins[:1]
    with pytest.raises(CompileError, match=r"missing in IR: \['2'\]"):
        compile_to(ir, tmp_path, lib)


@needs_libs
def test_pin_in_no_net_is_refused(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    gnd = ir.net("GND")
    gnd.pins = [p for p in gnd.pins if p.component_ref != "R2"]
    with pytest.raises(CompileError, match="pin R2.2 .*not in any net"):
        compile_to(ir, tmp_path, lib)


@needs_libs
def test_bad_net_references_are_refused(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    ir.nets.append(_fake_net("X", ("R9", "1")))
    with pytest.raises(CompileError, match="unknown component 'R9'"):
        compile_to(ir, tmp_path, lib)
    ir.nets[-1] = _fake_net("X", ("R1", "7"))
    with pytest.raises(CompileError, match="R1 has no pin '7'"):
        compile_to(ir, tmp_path, lib)
    ir.nets[-1] = _fake_net("X", ("R1", "1"))
    with pytest.raises(CompileError, match="R1.1 is in both nets"):
        compile_to(ir, tmp_path, lib)
    ir.nets[-1] = _fake_net("VIN")
    with pytest.raises(CompileError, match="duplicate net names"):
        compile_to(ir, tmp_path, lib)


@needs_libs
def test_unannotated_or_duplicate_references_are_refused(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    ir.component("R2").ref = "R1"
    with pytest.raises(CompileError, match="duplicate component reference 'R1'"):
        compile_to(ir, tmp_path, lib)
    ir.components[1].ref = "R?"
    with pytest.raises(CompileError, match="not annotated"):
        compile_to(ir, tmp_path, lib)


@needs_libs
def test_footprint_claimed_verified_but_missing_is_refused(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    r1 = ir.component("R1")
    r1.footprint = LibraryRef(library="Resistor_SMD", name="R_0603_DoesNotExist", verified=True)
    with pytest.raises(CompileError, match="R1: footprint .* not in the installed"):
        compile_to(ir, tmp_path, lib)
    r1.footprint = None  # no footprint at all is allowed for a schematic (property left empty, logged)
    _, path = compile_to(ir, tmp_path, lib)
    sch = sexpr.parse_file(path)
    r1_sym = next(s for s in sexpr.find_all(sch, "symbol") if sexpr.get(s, "property", 2) == "R1")
    assert {str(p[1]): str(p[2]) for p in sexpr.find_all(r1_sym, "property")}["Footprint"] == ""


# --------------------------------------------------------------------------- the real tool


@needs_kicad
def test_real_erc_passes_with_zero_errors(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    _, path = compile_to(ir, tmp_path, lib)
    result = _kicad.run_erc(path, tmp_path / "erc.json")
    warnings = result.details["warnings"]
    for w in warnings:
        print("ERC warning:", w.get("type"), w.get("description"))
    assert result.tool == "kicad-cli" and result.tool_version.startswith("10.")
    assert result.details["errors"] == [], result.details["errors"]
    assert result.status.value == "PASS"
    assert not [w for w in warnings if w.get("type") in COMPILER_DEFECT_WARNINGS], warnings
    assert result.artifact_hash == "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()


@needs_kicad
def test_real_netlist_reproduces_ir_connectivity(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    _, path = compile_to(ir, tmp_path, lib)
    net_path = _kicad.export_netlist(path, tmp_path / "out.net", fmt="kicadsexpr")
    nets = netlist_nets(net_path)
    assert nets == ir_net_map(ir) == EXPECTED_NETS
    export = sexpr.parse_file(net_path)
    comps = {str(sexpr.get(c, "ref")): c for c in sexpr.find_all(sexpr.find(export, "components"), "comp")}
    assert set(comps) == {"R1", "R2", "J1"}
    assert sexpr.get(comps["R1"], "footprint") == "Resistor_SMD:R_0603_1608Metric"
    assert sexpr.get(comps["J1"], "footprint") == "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"
    for ref, comp in comps.items():
        assert sexpr.get(comp, "tstamps") == symbol_uuid(PROJECT_ID, ref)


@needs_kicad
def test_real_erc_on_no_connect_variant(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    """A no_connect pin is accepted by ERC and shows up as an unconnected-* net in the netlist."""
    j1 = ir.component("J1")
    j1.pins[2] = j1.pins[2].model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT})
    ir.net("GND").pins = [p for p in ir.net("GND").pins if p.component_ref != "J1"]
    _, path = compile_to(ir, tmp_path, lib)
    result = _kicad.run_erc(path, tmp_path / "erc.json")
    assert result.details["errors"] == [], result.details["errors"]
    nets = netlist_nets(_kicad.export_netlist(path, tmp_path / "nc.net", fmt="kicadsexpr"))
    assert nets["VIN"] == EXPECTED_NETS["VIN"] and nets["VOUT"] == EXPECTED_NETS["VOUT"]
    assert nets["GND"] == {("R2", "2")}
    assert any(name.startswith("unconnected-(J1-") and pins == {("J1", "3")} for name, pins in nets.items())
