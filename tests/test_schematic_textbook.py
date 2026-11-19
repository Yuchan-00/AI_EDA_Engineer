"""Layout policy (textbook 1.0 is the default, legacy 0.8 is explicit), the textbook gates on the circuit templates,
structural goldens, and real KiCad connectivity/ERC evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.agents.keys import CONTROL_KEYS, SCHEMATIC_LAYOUT_KEY
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.schematic_layout import GRID_MM
from ai_eda.compilers.textbook import VERSION
from ai_eda.design.design_file import load_design_file
from ai_eda.errors import CompileError
from ai_eda.gui import schematic_render as render
from ai_eda.ir import ArtifactKind, CircuitIR, ProjectMeta, Topology
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.layout_metrics import schematic_gates, schematic_metrics
from ai_eda.workflow import Orchestrator, Stage
from tests.test_design_file import RC
from tests.test_legacy_bytes_pinned import _digests, _recorded
from tests.test_schematic_compiler import netlist_nets
from tests.test_textbook_placer import WIRE_09, structure
from tests.textbook_cases import CASES, amplifier, circuit
from tests.textbook_corpus import corpus_ir

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")
STRUCTURE = Path(__file__).parent / "data" / "textbook10_structure.json"


def compile_layout(ir, workdir, policy=None):
    return SchematicCompiler(policy).compile(ir, CompileContext(workdir=workdir, tools={"kicad_library": LIB}))


def partition(nets):
    return {frozenset(pins) for pins in nets.values() if pins}


def geometry(root):
    return {
        "symbols": {next(str(p[2]) for p in sexpr.find_all(s, "property") if p[1] == "Reference"): {
            "at": [float(v) for v in sexpr.find(s, "at")[1:]], "mirror": sexpr.get(s, "mirror")}
            for s in sexpr.find_all(root, "symbol")},
        "wires": sorted([[list(p) for p in render._pts(w, "wire")] for w in sexpr.find_all(root, "wire")]),
        "junctions": sorted([list(render._xy(sexpr.find(j, "at"), "junction")) for j in sexpr.find_all(root, "junction")]),
    }


@needs_kicad
@pytest.mark.parametrize("name", CASES)
def test_templates_have_equal_connectivity_zero_erc_and_clean_metrics(tmp_path, name):
    ir = circuit(name, tmp_path, LIB)
    before = ir.model_dump_json()
    expected = {n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets}
    exports = []
    for policy in ("legacy", None):  # explicit legacy, and the default: textbook 1.0
        artifact = compile_layout(ir, tmp_path / str(policy), policy)
        path = Path(artifact.path)
        result = CLI.run_erc(path, path.with_suffix(".erc.json"))
        assert result.details["errors"] == result.details["warnings"] == [], result.details
        assert result.status.value == "PASS"
        exported = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
        assert partition(exported) == partition(expected)
        exports.append(partition(exported))
        metrics = schematic_metrics(path)
        if policy is None:
            assert artifact.generator_version == VERSION == "1.0"
            # changed with the default switch: 1.0 names each signal net by one global label (DRC parity) where 0.9 drew
            # none, so "zero labels" becomes exactly one label on every signal net (power and ground are named by their
            # symbols), no net joined by labels, and no local label
            assert metrics["overlaps"] == metrics["local_label_count"] == 0
            gates = schematic_gates(path, ir)
            signal = {n.name: 1 for n in ir.nets if n.kind.value == "signal"}
            assert signal and gates["labels_per_net"] == signal and gates["label_joined_nets"] == 0, gates["labels_per_net"]
            # the design gates of the templates
            assert gates["crossings"] == (1 if name == "astable" else 0) and gates["paper"] == "A4", gates
            assert gates["near_symbol_hits"] == gates["parallel_hits"] == 0, gates
            assert gates["wire_rmst_ratio"] <= 1.5 and gates["wire_mm"] <= WIRE_09.get(name, 688), gates
            assert gates["supply_above_ground"]["ok"] == gates["supply_above_ground"]["of"], gates
            assert path.read_bytes() == Path(compile_layout(ir, tmp_path / "repeat", policy).path).read_bytes()
    assert exports[0] == exports[1]
    assert ir.model_dump_json() == before


@needs_libs
@pytest.mark.parametrize("name", ["astable", "divider", "rc_lowpass", "bjt_lowside_switch"])
def test_structural_goldens_and_upright_grid_fields(tmp_path, name):
    # changed with the default switch: the 0.9 coordinate goldens are deleted with 0.9; the 1.0 drawing is pinned by its
    # structure (recipe, rows, columns, slots, mirror, field sides), never by raw coordinates
    ir = circuit(name, tmp_path, LIB)
    assert structure(ir) == json.loads(STRUCTURE.read_text(encoding="utf-8"))[name]
    root = sexpr.parse_file(compile_layout(ir, tmp_path).path)
    for s in sexpr.find_all(root, "symbol"):
        at = sexpr.find(s, "at")
        for value in at[1:3]:
            assert float(value) / GRID_MM == pytest.approx(round(float(value) / GRID_MM))
        inst = render._Instance(float(at[1]), float(at[2]), int(at[3]), sexpr.get(s, "mirror"))
        for prop in sexpr.find_all(s, "property"):
            if str(prop[1]) not in {"Reference", "Value"}:
                continue
            out = render._Out()
            render._draw_fields(out, ["symbol", prop], inst)
            assert not any('rotate(90' in item or 'rotate(-90' in item for item in out.body)
    for wire in sexpr.find_all(root, "wire"):
        for p in render._pts(wire, "wire"):
            assert all(v / GRID_MM == pytest.approx(round(v / GRID_MM)) for v in p)


@needs_libs
def test_policy_precedence_validation_and_control_key(tmp_path):
    ir = circuit("divider", tmp_path, LIB)
    compiler = SchematicCompiler("legacy", answer_layout="textbook")
    assert compiler.layout_policy(ir) == "legacy"
    ir.topology.schematic_layout = "textbook"
    assert compiler.layout_policy(ir) == "textbook"
    ir.topology.schematic_layout = "legacy"
    assert SchematicCompiler("textbook").layout_policy(ir) == "legacy"
    ir.topology.schematic_layout = None
    assert SchematicCompiler(answer_layout="textbook").layout_policy(ir) == "textbook"
    with pytest.raises(CompileError, match="unknown schematic layout"):
        SchematicCompiler("typo").build(ir, LIB)
    assert SCHEMATIC_LAYOUT_KEY in CONTROL_KEYS
    ctx = AgentContext(workdir=tmp_path, answers={SCHEMATIC_LAYOUT_KEY: "textbook"})
    orchestrator = Orchestrator(ctx)
    assert ctx.tools["compilers"][ArtifactKind.SCHEMATIC].layout_policy(ir) == "textbook"
    orchestrator.run(ir, stop_after=Stage.REQUIREMENT_ANALYSIS)
    assert not any(r.key == SCHEMATIC_LAYOUT_KEY for r in ir.requirements.requirements)


@needs_libs
def test_design_file_policy_is_round_tripped_hashed_and_shown(tmp_path):
    data = copy.deepcopy(RC)
    path = tmp_path / "design.json"
    data["topology"]["schematic_layout"] = "textbook"
    path.write_text(json.dumps(data), encoding="utf-8")
    plan = load_design_file(path, LIB)
    topology = next(c.payload for c in plan.changes if c.target == "topology")
    assert topology.schematic_layout == "textbook"
    assert "schematic layout: textbook" in plan.table()
    assert Topology.model_validate_json(topology.model_dump_json()).schematic_layout == "textbook"
    file_ir = CircuitIR(project=ProjectMeta(id="file_rc", name="file RC"),
                        topology=topology,
                        components=[c.payload for c in plan.changes if c.target == "components"],
                        nets=[c.payload for c in plan.changes if c.target == "nets"])
    artifact = compile_layout(file_ir, tmp_path / "file")
    assert artifact.generator_version == VERSION
    assert schematic_metrics(artifact.path)["overlaps"] == 0
    ir = circuit("divider", tmp_path, LIB)
    old_hash = ir.content_hash()
    assert "schematic_layout" not in ir.design_dict()["topology"]
    ir.topology.schematic_layout = "textbook"
    assert ir.content_hash() != old_hash
    ir.topology.schematic_layout = None
    assert ir.content_hash() == old_hash


@needs_libs
def test_the_default_is_textbook_and_explicit_legacy_equals_the_recorded_08_bytes(tmp_path):
    ir = circuit("divider", tmp_path, LIB)
    assert SchematicCompiler().layout_policy(ir) == "textbook"
    default = compile_layout(ir, tmp_path / "default")
    assert default.generator_version == "1.0"
    assert "generated by compiler.kicad_sch 1.0 textbook from IR project" in Path(default.path).read_text(encoding="utf-8")
    recorded = _recorded()["divider"]
    explicit = {"argument": compile_layout(ir, tmp_path / "argument", "legacy"),
                "answer": SchematicCompiler(answer_layout="legacy").compile(ir, CompileContext(workdir=tmp_path / "answer", tools={"kicad_library": LIB}))}
    ir.topology.schematic_layout = "legacy"
    explicit["ir field"] = compile_layout(ir, tmp_path / "field", "textbook")  # the IR field wins over the argument
    for how, artifact in explicit.items():
        assert artifact.generator_version == "0.8", how
        digest, library = _digests(Path(artifact.path))
        if library != recorded["lib_symbols_sha256"]:
            pytest.skip("the installed KiCad library symbols differ from the ones the hash was recorded with")
        assert digest == recorded["sha256"], how


@needs_libs
@pytest.mark.parametrize("name", ["analog_piano", "stm32f407"])
def test_the_35_part_piano_and_the_100_pin_stm32_draw_textbook_with_no_whole_sheet_fallback(tmp_path, name):
    # changed with the default switch: 0.9 drew at most 20 parts and one IC and fell back to the legacy table above that;
    # 1.0 has no size limit
    ir = corpus_ir(name, tmp_path / "ir", LIB)
    artifact = compile_layout(ir, tmp_path / "out")
    assert artifact.generator_version == "1.0"
    assert artifact.notes[0].startswith("layout: textbook 1.0;") and not any("whole-sheet table" in note for note in artifact.notes)
    assert "(table fallback)" not in Path(artifact.path).read_text(encoding="utf-8")


#: The corpus gates of the layout design that the default drawing does not meet yet, each listed under "Limits" in
#: docs/SCHEMATIC_LAYOUT.md (by the design's name there). A gate that starts to hold, or a new one that breaks, fails below.
OPEN_CORPUS_GATES = {
    ("atmega128", "supply above ground"), ("stm32f407", "supply above ground"),
    ("divider", "blocks left to right"), ("led", "blocks left to right"), ("i2c_pullup", "blocks left to right"),
    ("atmega128", "blocks left to right"), ("stm32f407", "blocks left to right"),
    ("bjt_lowside_switch", "median distance"),
    ("atmega128", "labels per net"), ("stm32f407", "labels per net"), ("atmega128", "pins reached by a label"),
}
DOC_NAMES = {"divider": "divider", "led": "LED", "i2c_pullup": "I2C", "bjt_lowside_switch": "BJT switch",
             "atmega128": "ATmega128", "stm32f407": "STM32"}


@needs_libs
def test_the_open_corpus_gates_are_exactly_the_documented_limits(tmp_path):
    from tests.textbook_corpus import CORPUS

    open_gates = set()
    for name in CORPUS:
        ir = corpus_ir(name, tmp_path / "ir" / name, LIB)
        gates = schematic_gates(Path(compile_layout(ir, tmp_path / "out" / name).path), ir)
        held = {
            "supply above ground": gates["supply_above_ground"]["ok"] == gates["supply_above_ground"]["of"],
            "blocks left to right": gates["blocks_left_to_right"] is None or gates["blocks_left_to_right"]["ok"] == gates["blocks_left_to_right"]["of"],
        }
        if name not in ("amplifier", "atmega128", "stm32f407"):
            held["median distance"] = gates["median_connected_mm"] <= 40
        if name in ("atmega128", "stm32f407"):
            held["labels per net"] = max(gates["labels_per_net"].values()) <= 2
        if name == "atmega128":
            held["pins reached by a label"] = gates["pins_reached_by_label_pct"] <= 40
        open_gates |= {(name, gate) for gate, ok in held.items() if not ok}
    assert open_gates == OPEN_CORPUS_GATES
    doc = (Path(__file__).resolve().parent.parent / "docs" / "SCHEMATIC_LAYOUT.md").read_text(encoding="utf-8")
    items: list[str] = []  # the bullets under "Limits:", each joined over its wrapped lines
    for line in doc[doc.index("Limits:"):].splitlines()[1:]:
        if line.lstrip().startswith("- "):
            items.append(line.strip())
        elif items and line.startswith(" "):
            items[-1] += " " + line.strip()
    for name, gate in sorted(OPEN_CORPUS_GATES):
        item = next((i for i in items if i.startswith("- " + gate)), "")
        assert DOC_NAMES[name] in item, (name, gate, item)


@needs_kicad
def test_amplifier_feedback_above_core_and_connectivity(tmp_path):
    ir = amplifier(tmp_path, LIB)
    path = Path(compile_layout(ir, tmp_path, "textbook").path)
    layout = geometry(sexpr.parse_file(path))["symbols"]
    assert layout["R2"]["at"][1] < layout["U1"]["at"][1]
    assert layout["R2"]["at"][2] in (90, 270)
    # changed with the default switch: named terminals replace 0.9's TestPoint ports, so J1 (the IN terminal) leads
    assert layout["J1"]["at"][0] < layout["R1"]["at"][0] < layout["U1"]["at"][0]
    assert schematic_metrics(path)["overlaps"] == 0
    exported = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    expected = {n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets}
    # KiCad exports intentional no-connect pins as separate singleton nets.
    assert partition({n: p for n, p in exported.items() if not n.startswith("unconnected-")}) == partition(expected)
    result = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert result.details["errors"] == result.details["warnings"] == [], result.details


@needs_libs
def test_topology_names_and_input_list_order_do_not_select_a_template(tmp_path):
    ir = circuit("astable", tmp_path, LIB)
    first = compile_layout(ir, tmp_path / "a", "textbook")
    ir.topology.name = "unrelated design-file title"
    ir.components.reverse()
    ir.nets.reverse()
    for n in ir.nets:
        n.pins.reverse()
    second = compile_layout(ir, tmp_path / "b", "textbook")
    assert Path(first.path).read_bytes() == Path(second.path).read_bytes()


@needs_libs
@pytest.mark.parametrize("fault", ["pin", "net", "symbol"])
def test_textbook_preserves_library_and_connectivity_refusals(tmp_path, fault):
    ir = circuit("divider", tmp_path, LIB)
    if fault == "pin":
        ir.component("R1").pins.pop()
    elif fault == "net":
        ir.net("GND").pins.append(ir.net("VIN").pins[0])
    else:
        ir.component("R1").symbol.verified = False
    with pytest.raises(CompileError):
        compile_layout(ir, tmp_path, "textbook")


@needs_kicad
@pytest.mark.parametrize("name", ["astable", "bjt_lowside_switch"])
def test_design_file_style_emitter_parts_stay_in_the_columns(tmp_path, name):
    from ai_eda.ir import Component, LibraryRef, Net, Pin, PinRef

    ir = circuit(name, tmp_path, LIB)
    symbol_ref = LIB.resolve_symbol(LibraryRef(library="Diode", name="1N4001"))
    diode = LIB.load_symbol(symbol_ref)
    ground = ir.net("GND")
    for q in [c for c in ir.components if c.ref.startswith("Q")]:
        emitter = next(p.number for p in q.pins if p.name == "E")
        ref = "D" + str(10 + int(q.ref[1:]))
        a = next(p.number for p in diode.pins if p.name == "A")
        k = next(p.number for p in diode.pins if p.name == "K")
        ir.components.append(Component(ref=ref, value="1N4001", symbol=symbol_ref, provenance=q.provenance,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=q.provenance) for p in diode.pins]))
        ground.pins = [p for p in ground.pins if (p.component_ref, p.pin_number) != (q.ref, emitter)]
        ground.pins.append(PinRef(component_ref=ref, pin_number=k))
        ir.nets.append(Net(name=q.ref + "_E", provenance=q.provenance,
                           pins=[PinRef(component_ref=q.ref, pin_number=emitter), PinRef(component_ref=ref, pin_number=a)]))
    path = Path(compile_layout(ir, tmp_path, "textbook").path)
    metrics = schematic_metrics(path)
    assert metrics["overlaps"] == 0
    assert metrics["wire_crossings"] == (1 if name == "astable" else 0)
    exported = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    assert partition(exported) == partition({n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets})
    result = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert result.details["errors"] == result.details["warnings"] == []
