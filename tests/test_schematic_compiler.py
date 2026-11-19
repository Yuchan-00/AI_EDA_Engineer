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


def compile_to(ir: CircuitIR, workdir: Path, lib: KicadLibrary, layout: str | None = None):
    ref = SchematicCompiler(layout).compile(ir, CompileContext(workdir=workdir, tools={"kicad_library": lib}))
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


# The legacy 0.8 table (a global label on every pin, gutter wires, the wiring note, flags) is tested with an explicit
# layout="legacy" since textbook 1.0 became the default; the refusals below run on the default.


@needs_libs
def test_compile_writes_file_with_ir_hash(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    ref, path = compile_to(ir, tmp_path / "out", lib, "legacy")
    assert path.exists() and path.name == f"{PROJECT_ID}.kicad_sch"
    assert ref.generated_from_ir_hash == ir.content_hash()
    assert ref.generator == SchematicCompiler.id and ref.matches_disk()
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text and text.endswith(")\n")
    assert f"generated by {SchematicCompiler.id} {SchematicCompiler.version}" in text  # provenance comment in title_block


@needs_libs
def test_compile_is_byte_deterministic(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    for layout in ("legacy", None):
        _, a = compile_to(ir, tmp_path / f"a{layout}", lib, layout)
        _, b = compile_to(divider_with_connector_ir(tmp_path, KicadLibrary()), tmp_path / f"b{layout}", KicadLibrary(), layout)
        assert a.read_bytes() == b.read_bytes()


@needs_libs
def test_structure_uuids_and_embedded_library_symbols(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    _, path = compile_to(ir, tmp_path, lib, "legacy")
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
    # one global label per (net, pin) anchored at a wire end (the pin's stub); the routed wires come on top (B11)
    labels = sexpr.find_all(sch, "global_label")
    wires = sexpr.find_all(sch, "wire")
    pin_count = sum(len(v) for v in EXPECTED_NETS.values())
    assert len(labels) == pin_count and len(wires) > pin_count
    assert sorted(str(l[1]) for l in labels) == sorted(n for n, pins in EXPECTED_NETS.items() for _ in pins)
    wire_ends = {tuple(str(a) for a in xy[1:3]) for w in wires for xy in sexpr.find_all(sexpr.find(w, "pts"), "xy")}
    for label in labels:
        at = sexpr.find(label, "at")
        assert (str(at[1]), str(at[2])) in wire_ends
    assert not sexpr.find_all(sch, "no_connect")


@needs_libs
def test_every_small_net_is_drawn_as_wires_with_a_junction_where_three_pins_meet(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    """B11: VIN (2 pins), VOUT (3 pins) and GND (2 pins) each get a trunk + branches; VOUT's three branches meet at one junction.
    The note says so; the labels stay on every pin, so the netlist (tested below with kicad-cli) is unchanged."""
    node, notes = SchematicCompiler("legacy").build_with_notes(ir, lib)
    wires = sexpr.find_all(node, "wire")
    junctions = sexpr.find_all(node, "junction")
    pin_count = sum(len(v) for v in EXPECTED_NETS.values())
    assert len(wires) >= pin_count + 3 * 2  # every net adds at least a trunk and two branches beyond its stubs
    assert len(junctions) == 1
    assert notes == ["wiring: 3 of 3 net(s) drawn as wires (labels kept on every pin)"]
    # every wire is orthogonal and on the 1.27 mm grid
    for w in wires:
        (ax, ay), (bx, by) = [tuple(float(v) for v in xy[1:3]) for xy in sexpr.find_all(sexpr.find(w, "pts"), "xy")]
        assert ax == bx or ay == by
        for v in (ax, ay, bx, by):
            assert abs(v / 1.27 - round(v / 1.27)) < 1e-6, (ax, ay, bx, by)
    # the compile stage carries the note
    ref, _ = compile_to(ir, tmp_path, lib, "legacy")
    assert ref.notes == notes


@needs_libs
def test_no_connect_pin_emits_marker(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    j1 = ir.component("J1")
    j1.pins[2] = j1.pins[2].model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT})
    ir.net("GND").pins = [p for p in ir.net("GND").pins if p.component_ref != "J1"]
    _, path = compile_to(ir, tmp_path, lib, "legacy")
    sch = sexpr.parse_file(path)
    ncs = sexpr.find_all(sch, "no_connect")
    assert len(ncs) == 1
    at = sexpr.find(ncs[0], "at")
    j1_pos = sexpr.find(next(s for s in sexpr.find_all(sch, "symbol") if sexpr.get(s, "property", 2) == "J1"), "at")
    assert (float(at[1]), float(at[2])) == pytest.approx((float(j1_pos[1]) - 5.08, float(j1_pos[2]) + 2.54))  # pin 3 end


@needs_libs
def test_a_net_with_only_power_input_pins_gets_exactly_one_pwr_flag_and_a_driven_net_none(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary):
    """B12: two L7805 in series and an NE555 on the divider. VIN (U1 IN, the NE555's inputs) and GND (both regulators' GND)
    carry power-input pins and otherwise only input / passive pins, so each gets one ``power:PWR_FLAG`` with its own stub +
    label. MID (U1 OUT -> U2 IN, U3 VCC) and VOUT (U2 OUT) have a power output and get none. X ties the NE555's ``output``
    pin to its own GND pin: KiCad reports a flag there against the output (``pin_to_pin``), so X gets none and a note, and
    ERC keeps its one honest error. The netlist is the IR's (no ``#FLG``)."""
    prov = ir.component("R1").provenance
    parts = {"U1": ("Regulator_Linear", "L7805"), "U2": ("Regulator_Linear", "L7805"), "U3": ("Timer", "NE555P")}
    for ref, (library, name) in parts.items():
        symbol = lib.resolve_symbol(LibraryRef(library=library, name=name))
        if not symbol.verified:
            pytest.skip(f"{library}:{name} not installed")
        pins = [Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in lib.load_symbol(symbol).pins]
        ir.components.append(ir.component("R1").model_copy(update={"ref": ref, "value": name, "symbol": symbol, "footprint": None, "pins": pins, "spice": None}))
    more = {"VIN": [("U1", "1"), ("U3", "2"), ("U3", "4"), ("U3", "6"), ("U3", "7")], "GND": [("U1", "2"), ("U2", "2")], "VOUT": [("U2", "3"), ("U3", "5")]}
    for name, pins in more.items():
        ir.net(name).pins += [PinRef(component_ref=r, pin_number=p) for r, p in pins]
    ir.nets += [_fake_net("MID", ("U1", "3"), ("U2", "1"), ("U3", "8")), _fake_net("X", ("U3", "1"), ("U3", "3"))]
    node, notes = SchematicCompiler("legacy").build_with_notes(ir, lib)
    assert sexpr.dumps(node) == sexpr.dumps(SchematicCompiler("legacy").build(ir, KicadLibrary()))  # flags included: the same bytes again
    flags = [s for s in sexpr.find_all(node, "symbol") if sexpr.get(s, "lib_id") == "power:PWR_FLAG"]
    assert [str(sexpr.get(s, "property", 2)) for s in flags] == ["#FLG01", "#FLG02"]  # net-name order: GND, VIN
    assert all(sexpr.get(sexpr.find(s, "property"), "hide") is True for s in flags)  # the library hides a flag's reference, so the instance does
    labels = [str(g[1]) for g in sexpr.find_all(node, "global_label")]
    members = {n.name: len(n.pins) for n in ir.nets}
    assert {n: labels.count(n) - members[n] for n in members} == {"VIN": 1, "GND": 1, "VOUT": 0, "MID": 0, "X": 0}
    assert notes[-2] == "power flags: PWR_FLAG on 2 net(s) with a power-input pin and no power-output pin (GND, VIN)"
    assert notes[-1].startswith("power flags: none on X (")
    # GND's four IR pins are still drawn as wires: the flag's pin, a fifth, is not the router's (5 > MAX_WIRED_PINS would be "labels only")
    assert notes[0].startswith("wiring: ") and " of 5 net(s)" in notes[0] and not any(n.startswith("not wired") and "GND" in n for n in notes)
    if not _kicad.available():
        return
    _, path = compile_to(ir, tmp_path, lib, "legacy")
    result = _kicad.run_erc(path, tmp_path / "erc.json")
    errors = result.details["errors"]  # X only (U3's GND pin behind its own output): not driven, and no pin_to_pin from a flag
    assert [e["type"] for e in errors] == ["power_pin_not_driven"] and "U3" in errors[0]["items"][0]["description"], errors
    assert result.details["warnings"] == [], result.details["warnings"]
    assert netlist_nets(_kicad.export_netlist(path, tmp_path / "out.net", fmt="kicadsexpr")) == ir_net_map(ir)


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
    _, path = compile_to(ir, tmp_path, lib, "legacy")
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
    _, path = compile_to(ir, tmp_path, lib, "legacy")
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
    _, path = compile_to(ir, tmp_path, lib, "legacy")
    result = _kicad.run_erc(path, tmp_path / "erc.json")
    assert result.details["errors"] == [], result.details["errors"]
    nets = netlist_nets(_kicad.export_netlist(path, tmp_path / "nc.net", fmt="kicadsexpr"))
    assert nets["VIN"] == EXPECTED_NETS["VIN"] and nets["VOUT"] == EXPECTED_NETS["VOUT"]
    assert nets["GND"] == {("R2", "2")}
    assert any(name.startswith("unconnected-(J1-") and pins == {("J1", "3")} for name, pins in nets.items())


def _instance_fields(path: Path) -> dict[str, dict[str, str]]:
    """Reference -> the properties of its (first) placed symbol."""
    out: dict[str, dict[str, str]] = {}
    for item in sexpr.parse_file(path):
        if sexpr.head(item) == "symbol":
            props = {str(p[1]): str(p[2]) for p in sexpr.find_all(item, "property") if len(p) > 2}
            out.setdefault(props["Reference"], props)
    return out


@needs_kicad
@pytest.mark.parametrize("layout", ["textbook", "legacy"])
def test_identity_fields_carry_what_the_ir_states_and_a_library_link_never_contradicts_a_stated_part(tmp_path: Path, lib: KicadLibrary, layout: str):
    """Manufacturer / MPN are written when the IR states them (typed by the user or found in a datasheet), never for a
    model's proposal. A Datasheet link that is only the library symbol's own property describes the symbol, not a part
    number the IR states, so it is left empty for such a part; a part without a stated MPN keeps it, a URL of the IR's
    own stays, and an MPN found in a datasheet brings that document's URL."""
    from ai_eda.compilers.schematic import LIBRARY_AUTHORITY
    from ai_eda.ir import SourceRef, authoritative, llm_generated, user_requirement
    from ai_eda.parts.pointers import datasheet_pointer
    from tests.textbook_corpus import corpus_ir

    ir = corpus_ir("bjt_lowside_switch", tmp_path / "ir", lib)
    q1, d1, r1, j1, j2 = (ir.component(r) for r in ("Q1", "D1", "R1", "J1", "J2"))
    q1.datasheet, d1.datasheet = datasheet_pointer(q1, lib), datasheet_pointer(d1, lib)  # what the pipeline records: the library's link
    assert q1.datasheet.url and q1.datasheet.authority.startswith(LIBRARY_AUTHORITY) and d1.datasheet.url
    q1.manufacturer, q1.mpn = user_requirement("Example Maker"), user_requirement("EX3904-T")
    r1.mpn = llm_generated("EX-R-1K", model="any")
    j1.mpn = user_requirement("EX-HDR-2")
    j1.datasheet = datasheet_pointer(j1, lib, user_urls={"J1": "https://example.com/ex-hdr-2.pdf"})
    j2.mpn = authoritative("EX-HDR-3", SourceRef(title="EX-HDR-3 datasheet", url="https://example.com/ex-hdr-3.pdf", content_hash="sha256:" + "0" * 64))
    _, path = compile_to(ir, tmp_path / layout, lib, layout)
    fields = _instance_fields(path)
    assert (fields["Q1"]["Manufacturer"], fields["Q1"]["MPN"], fields["Q1"]["Datasheet"]) == ("Example Maker", "EX3904-T", "")
    assert fields["D1"]["Datasheet"] == d1.datasheet.url and "MPN" not in fields["D1"] and "Manufacturer" not in fields["D1"]
    assert "MPN" not in fields["R1"]  # a model's proposal is not a statement
    assert (fields["J1"]["MPN"], fields["J1"]["Datasheet"]) == ("EX-HDR-2", "https://example.com/ex-hdr-2.pdf")
    assert (fields["J2"]["MPN"], fields["J2"]["Datasheet"]) == ("EX-HDR-3", "https://example.com/ex-hdr-3.pdf")
    assert "Manufacturer" not in fields["J2"]
    result = _kicad.run_erc(path, path.with_suffix(".erc.json"))
    assert result.details["errors"] == [] and result.details["warnings"] == []
