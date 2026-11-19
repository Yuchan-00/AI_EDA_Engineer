"""Textbook 1.0 general placer: blocks and their DAG, rail derivation and rail lines, the columns recipe (potentials,
orientation, path cover, rows, columns, order, coordinates), arrangement (pendant blocks, band wrap), the label box KiCad
draws, structural goldens and the placer's gates on the corpus."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.schematic import GENERATOR, GENERATOR_VERSION, SCH_FORMAT_VERSION, _global_label
from ai_eda.compilers.textbook import arrange, blocks, build, pipeline, rails
from ai_eda.compilers.textbook.columns import COLUMNS, Layout
from ai_eda.compilers.textbook.glyphs import label_box, mm, pin_cell, pin_side
from ai_eda.compilers.textbook.labels import _on_wire, stub_label
from ai_eda.compilers.textbook.model import BlockInfo, Label, Orient, load_model
from ai_eda.compilers.textbook.recipes import Ctx, recipe_order
from ai_eda.ir import CircuitIR, Component, LibraryRef, Net, Pin, PinRef, ProjectMeta, Provenance, ProvenanceKind, Topology
from ai_eda.ir.topology import Block
from ai_eda.ir import PinElectricalType
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad.sexpr import Q, S
from ai_eda.tools.layout_metrics import schematic_gates
from tests.test_textbook_fuzz import random_ir
from tests.textbook_corpus import CORPUS, SMALL, corpus_ir

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")
GOLDEN = Path(__file__).resolve().parent / "data" / "textbook10_structure.json"
STRUCTURED = (*SMALL, "analog_piano", "atmega128", "stm32f407")
#: the 0.9 textbook wire lengths of the templates (mm), the most the placer may draw
WIRE_09 = {"divider": 227, "led": 189, "rc_lowpass": 241, "lc_lowpass": 262, "bjt_lowside_switch": 328, "i2c_pullup": 318}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("placer")
    return {name: corpus_ir(name, base / name, LIB) for name in CORPUS}


def prepared(ir: CircuitIR):
    """The model with its blocks and rail order, and a context, as the pipeline prepares them."""
    model = load_model(SchematicCompiler(), ir, LIB)
    model = replace(model, blocks=blocks.partition(model), rails=rails.rail_order(model))
    return model, Ctx(SchematicCompiler(), LIB, model, model.notes)


def layouts(ir: CircuitIR) -> dict[str, list[Layout]]:
    """Block id -> the columns layouts of its groups, as the pipeline claims them."""
    model, ctx = prepared(ir)
    out: dict[str, list[Layout]] = {}
    for block in model.blocks:
        free = set(block.members)
        out[block.id] = []
        while free and (claimed := COLUMNS.claim(model, block, frozenset(free))):
            out[block.id].append(Layout(ctx, block, claimed))
            free -= claimed
    return out


def every(ir: CircuitIR) -> list[Layout]:
    return [lay for ls in layouts(ir).values() for lay in ls]


def circuit(project: str, parts: list[tuple[str, str, str, str]], nets: list[tuple[str, str, tuple[str, ...]]]) -> CircuitIR:
    """A small IR from (ref, value, library, symbol) and (net, kind, pins "REF.N")."""
    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="placer test")
    ir = CircuitIR(project=ProjectMeta(id=project, name=project), topology=Topology(name=project, provenance=prov))
    for ref, value, lib, name in parts:
        libref = LIB.resolve_symbol(LibraryRef(library=lib, name=name))
        symbol = LIB.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))
    for name, kind, pins in nets:
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split(".")[0], pin_number=p.split(".")[1]) for p in pins]))
    return ir


def pin_y(lay_drawing, model, key: str, number: str) -> int:
    place = lay_drawing.place[key]
    return pin_cell(model.glyphs[key], number, (place.x, place.y), place.orient)[1]


# --------------------------------------------------------------------------- blocks and rails


@needs_libs
def test_rail_order_follows_the_supply_derivation(corpus):
    model, _ = prepared(corpus["analog_piano"])
    assert rails.rail_order(model) == ("VBATIN", "BAT", "OSC")
    tree = rails.derivation(model)
    assert tree["VBATIN"] == (None, None, 0)
    assert tree["BAT"] == ("VBATIN", "SW9", 0)  # a switch in series keeps the level
    assert tree["OSC"] == ("BAT", "Q5", 1)  # a follower adds one


@needs_libs
def test_block_layers_come_from_the_block_dag_and_a_cycle_is_dropped_with_a_note(corpus):
    model, _ = prepared(corpus["analog_piano"])
    assert [(b.id, b.layer) for b in model.blocks] == [("power", 0), ("keyboard", 1), ("oscillator", 2), ("driver", 3)]
    ir = corpus["analog_piano"].model_copy(deep=True)
    driver = next(b for b in ir.topology.blocks if b.id == "driver")
    power = next(b for b in ir.topology.blocks if b.id == "power")
    driver.output_nets.append("BAT")  # driver -> power, against power -> driver (BAT leaves power, enters the driver)
    power.input_nets.append("BAT")
    model, _ = prepared(ir)
    assert any(n == "block order: edge driver -> power dropped (it would close a cycle)" for n in model.notes), model.notes
    assert [b.id for b in model.blocks][0] == "power"


@needs_libs
def test_rails_are_one_line_per_band_with_the_symbol_at_the_right_end_and_a_ground_symbol_per_pin(corpus):
    model, ctx = prepared(corpus["analog_piano"])
    drawn = {b.id: pipeline.draw_block(ctx, b, recipe_order(), []) for b in model.blocks}
    sheet = arrange.arrange(ctx, [d for b in model.blocks for d in drawn[b.id]], 0)
    sheet = rails.draw_rails(ctx, sheet)
    bat = [s for s in sheet.rails if s.net == "BAT"]
    assert len(bat) == 1  # one supply line across the top, from the switch to the driver
    symbol = next(g for g in sheet.glyphs if g.kind == "VCC" and g.net == "BAT")
    assert symbol.value == "BAT" and symbol.at[0] >= bat[0].x1 - 1 and symbol.at[1] < bat[0].y
    assert len([s for s in sheet.rails if s.net == "OSC"]) == 2  # the oscillator's band and the keyboard's own band
    grounds = [g for g in sheet.generated() if g.kind == "GND"]
    assert len(grounds) == len(model.nets["GND"].pins) == 6
    assert {f.net for f in sheet.glyphs if f.kind == "FLAG"} == {"BAT", "GND", "OSC", "VBATIN"}  # each symbol's net is driven


@needs_libs
def test_a_pendant_block_hangs_below_the_block_it_feeds_instead_of_wrapping_the_spine(corpus):
    model, ctx = prepared(corpus["analog_piano"])
    drawings = [d for b in model.blocks for d in pipeline.draw_block(ctx, b, recipe_order(), [])]
    sheet = arrange.arrange(ctx, drawings, 0)
    at = {d.block: (off, d.box) for d, off in zip(sheet.drawings, sheet.offset)}
    (kx, ky), kbox = at["keyboard"]
    (ox, oy), obox = at["oscillator"]
    assert ky + kbox[1] > oy + obox[3]  # below the oscillator
    assert [len(b) for b in sheet.bands] == [3, 1]  # power, oscillator, driver on the spine; the keyboard in its band
    # the note travels with its sheet variant (it reaches the build only when this variant is kept)
    assert "block keyboard: hung below oscillator, whose Q2B it feeds" in sheet.notes
    assert not any("hung below" in n for n in ctx.notes)  # not in the build's notes before this variant is kept
    wrapped = arrange.arrange(replace(ctx, notes=[]), drawings, 1)
    assert wrapped is None  # no satellite of the design's own rule: no second variant


@needs_libs
def test_band_wrap_cuts_at_the_fewest_signal_nets_ties_to_the_later_boundary_that_fits(corpus):
    model, ctx = prepared(corpus["analog_piano"])
    drawings = [d for b in model.blocks for d in pipeline.draw_block(ctx, b, recipe_order(), [])]
    index = {d.block: i for i, d in enumerate(drawings)}
    width = {d.block: d.box[2] - d.box[0] for d in drawings}
    spine = [index[b] for b in ("power", "keyboard", "oscillator", "driver")]
    assert arrange.cut(model, drawings, spine) == [spine[:1], spine[1:]]  # power | rest: no signal net cut (OSC, BAT are rails)
    rest = spine[1:]  # keyboard | oscillator cuts Q2B, oscillator | driver cuts Q1C: a tie, the later boundary that fits wins
    two = width["keyboard"] + arrange.BLOCK_GAP + width["oscillator"]
    assert arrange.cut(model, drawings, rest, two) == [rest[:2], rest[2:]]
    assert arrange.cut(model, drawings, rest, two - 1)[0] == rest[:1]  # only the first boundary fits


# --------------------------------------------------------------------------- the columns recipe


@needs_libs
def test_potentials_stand_a_bjt_collector_over_base_over_emitter_and_flip_a_pnp_high_side_switch():
    ir = circuit("flips", [("Q1", "2N3904", "Transistor_BJT", "2N3904"), ("R1", "1k", "Device", "R"), ("R2", "10k", "Device", "R"),
                           ("Q2", "2N3906", "Transistor_BJT", "2N3906"), ("R3", "1k", "Device", "R"), ("R4", "10k", "Device", "R")],
                 [("VCC", "power", ("R1.1", "Q2.1")), ("GND", "ground", ("Q1.1", "R3.2")),
                  ("LOAD", "signal", ("R1.2", "Q1.3")), ("BASE", "signal", ("R2.2", "Q1.2")), ("IN", "signal", ("R2.1", "R4.1")),
                  ("OUT", "signal", ("Q2.3", "R3.1")), ("PB", "signal", ("Q2.2", "R4.2"))])
    model, ctx = prepared(ir)
    lays = every(ir)
    for lay in lays:
        if "Q1" in lay.keys:
            assert lay.pot["LOAD"] > lay.pot["GND"] and lay.orient["Q1"].rot == 0 and lay.orient["Q1"].mirror != "x"
            d = lay.draw(False)
            assert pin_y(d, model, "Q1", "3") < pin_y(d, model, "Q1", "2") < pin_y(d, model, "Q1", "1")  # C over B over E
        if "Q2" in lay.keys:
            assert lay.orient["Q2"] != Orient()  # the PNP's emitter (on the supply) is at the bottom of the library symbol
            d = lay.draw(False)
            assert pin_y(d, model, "Q2", "1") < pin_y(d, model, "Q2", "3")  # emitter above collector: supply above ground


@needs_libs
def test_the_piano_path_cover_and_which_parts_stand_or_lie(corpus):
    lays = every(corpus["analog_piano"])
    columns = {tuple(c) for lay in lays for c in lay.cover.columns}
    assert columns == {("J1",), ("RV1",), ("R10", "Q1"), ("R11", "Q2"), ("Q3", "R13", "R14"), ("R15", "D1", "Q4")}
    assert {k: v for lay in lays for k, v in lay.cover.bundles.items()} == {"D1": ["J2"]}  # the speaker beside its diode
    assert sorted((k for lay in lays for k in lay.cover.hanging), key=lambda k: (len(k), k)) == ["R9", *(f"SW{i}" for i in range(1, 9))]
    lying = {k for lay in lays for k in lay.hslots}
    assert lying == {"C1", "C2", "R12", *(f"R{i}" for i in range(1, 9))}
    vertical = {k for lay in lays for k, v in lay.vertical.items() if v}
    assert {"D1", "R13", "R14", "R15", "R9", "R10", "R11", "SW1"} <= vertical and not {"C1", "C2"} & vertical
    driver = next(lay for lay in lays if "J2" in lay.keys)
    d = driver.draw(False)
    model = driver.model
    j2 = [pin_cell(model.glyphs["J2"], p, (d.place["J2"].x, d.place["J2"].y), d.place["J2"].orient) for p in ("1", "2")]
    assert j2[0][0] == j2[1][0] and j2[0][1] < j2[1][1]  # the connector stands, pin 1 on top


@needs_libs
def test_the_row_solver_never_closes_a_cycle_and_a_refused_merge_is_a_jog_with_a_note(corpus):
    ir = circuit("cycle", [("Q1", "2N3904", "Transistor_BJT", "2N3904"), ("C1", "100n", "Device", "C"), ("R1", "1k", "Device", "R"),
                           ("R2", "100", "Device", "R"), ("R3", "10k", "Device", "R")],
                 [("VCC", "power", ("R1.1",)), ("A", "signal", ("R1.2", "Q1.3", "C1.1")), ("B", "signal", ("Q1.1", "C1.2", "R2.1")),
                  ("BASE", "signal", ("Q1.2", "R3.1")), ("GND", "ground", ("R2.2", "R3.2"))])  # A over Q1 over B: C1 cannot lie flat
    lays = every(ir)
    lay = next(lay for lay in lays if "C1" in lay.keys)
    assert lay.jog == {"C1"} and any(n.startswith("columns: C1 would close a cycle") for n in lay.notes)
    for lay in lays + every(corpus["analog_piano"]) + every(corpus["astable"]) + [x for s in range(10) for x in every(random_ir(s))]:
        succ = lay._classes()
        indeg = {c: 0 for c in {lay.uf.find(i) for i in lay.row}}
        for a in succ:
            for b in succ[a]:
                indeg[b] += 1
        ready, seen = [c for c, n in indeg.items() if not n], 0
        while ready:
            c = ready.pop()
            seen += 1
            for b in succ.get(c, ()):
                indeg[b] -= 1
                if not indeg[b]:
                    ready.append(b)
        assert seen == len(indeg)  # a topological order covers every class: no cycle
        for u, v in lay.edges:
            assert lay.row[u] < lay.row[v]


@needs_libs
def test_a_feedback_part_gets_the_row_above_its_owner(corpus):
    ir = corpus["amplifier"]
    (lay,) = every(ir)
    assert lay.bridge == {"R2": "U1"} and lay.row["c:R2"] == lay.row["c:U1"] - 1
    d = lay.draw(False)
    assert d.place["R2"].y < d.place["U1"].y
    r1 = pin_y(d, lay.model, "R1", "2")
    assert r1 == pin_y(d, lay.model, "U1", "2")  # the input resistor at the height of the '-' pin it is tied to


@needs_libs
def test_the_greedy_switch_never_raises_the_sketch_cost_and_precedence_always_holds(corpus):
    lays = [lay for name in CORPUS for lay in every(corpus[name])] + [lay for s in range(30) for lay in every(random_ir(s))]
    assert len(lays) > 50
    for lay in lays:
        assert lay.sketch <= lay.before and lay.legal(lay.order)
        assert sorted(lay.order) == sorted(lay.members)


# --------------------------------------------------------------------------- the label box


def test_the_label_box_covers_kicads_outline_along_the_label():
    box = label_box(Label("T1", (0, 0), 0, "left", "name"))
    assert box == pytest.approx((-0.0762, -1.27, 3.5112 + 0.0762, 1.27), abs=1e-4)
    assert label_box(Label("T1", (10, 10), 180, "right", "name")) == pytest.approx((mm(10) - 3.5874, mm(10) - 1.27, mm(10) + 0.0762, mm(10) + 1.27), abs=1e-4)
    up = label_box(Label("OSC", (0, 0), 90, "left", "name"))
    assert up[3] == pytest.approx(0.0762) and up[1] == pytest.approx(-(4.0132 + 1.1309 + 0.0762), abs=1e-4)


@needs_kicad
def test_the_label_box_holds_every_label_outline_kicad_draws(tmp_path):
    names = ["T5", "OSC", "Q2B", "HSE_OUT", "VCAP_1", "+5V", "iiiiii", "MMMMMM", "SPI1_MOSI", "a b", "PA10", "/X.y"]
    labels = [Label(n, (20 + j * 32, 20 + i * 12), a, j_, "name") for i, n in enumerate(names) for j, (a, j_) in enumerate(((0, "left"), (180, "right"), (90, "left"), (270, "right")))]
    nodes = [_global_label(lab.net, (mm(lab.at[0]), mm(lab.at[1])), lab.angle, lab.justify, f"00000000-0000-0000-0000-{i:012d}") for i, lab in enumerate(labels)]
    root = S("kicad_sch", S("version", SCH_FORMAT_VERSION), S("generator", Q(GENERATOR)), S("generator_version", Q(GENERATOR_VERSION)),
             S("uuid", Q("11111111-1111-1111-1111-111111111111")), S("paper", Q("A2")), S("lib_symbols"), *nodes,
             S("sheet_instances", S("path", Q("/"), S("page", Q("1")))), S("embedded_fonts", False))
    path = tmp_path / "labels.kicad_sch"
    path.write_text(sexpr.dumps(root), encoding="utf-8", newline="\n")
    CLI._run(["sch", "export", "svg", "--output", str(tmp_path), "--exclude-drawing-sheet", str(path)])
    svg = (tmp_path / "labels.svg").read_text(encoding="utf-8")
    outlines = [[tuple(map(float, p.split(","))) for p in m.group(1).split()] for m in re.finditer(r'd="M ([^"]*?)Z"', svg)]
    assert len(outlines) == len(labels)
    for lab in labels:
        x, y = mm(lab.at[0]), mm(lab.at[1])
        (outline,) = [o for o in outlines if (x, y) in {(round(px, 4), round(py, 4)) for px, py in o} or
                      min(px for px, _ in o) - 1e-3 <= x <= max(px for px, _ in o) + 1e-3 and min(py for _, py in o) - 1e-3 <= y <= max(py for _, py in o) + 1e-3]
        x0, y0, x1, y1 = label_box(lab)
        pad = 0.0762  # half the outline's stroke
        horizontal = lab.angle in (0, 180)
        along = [px for px, _ in outline] if horizontal else [py for _, py in outline]
        lo, hi = (x0, x1) if horizontal else (y0, y1)
        assert lo <= min(along) - pad + 1e-3 and max(along) + pad - 1e-3 <= hi, lab  # the whole length, stroke included
        assert hi - lo - (max(along) - min(along) + 2 * pad) < 0.01, lab  # and not longer
        across = [py for _, py in outline] if horizontal else [px for px, _ in outline]
        lo, hi = (y0, y1) if horizontal else (x0, x1)
        assert max(across) - min(across) == pytest.approx(2.5408, abs=1e-3) and hi - lo == pytest.approx(2.54)  # one cell across


# --------------------------------------------------------------------------- structure and gates


def structure(ir: CircuitIR) -> dict:
    """Per block, per columns group: its rows, columns, slot order, mirrored halves, and of the variant the pipeline keeps
    (its index) the orientations and field sides; feedback bridges and jogs."""
    model, ctx = prepared(ir)
    out: dict[str, list] = {}
    for block, lays in layouts(ir).items():
        out[block] = []
        for lay in lays:
            variants = lay.drawings()
            d = min(variants, key=lambda v: pipeline.rank(ctx, v))
            out[block].append({
                "variant": variants.index(d),
                "mirror": {k: lay.half[k] for k in sorted(lay.half)},
                "rows": {k: lay.row["c:" + k] for k in lay.cells},
                "columns": [lay.members[s] for s in lay.order if s.startswith("col:")],
                "slots": [s for s in lay.order],
                "orient": {k: [d.place[k].orient.rot, d.place[k].orient.mirror] for k in sorted(d.place)},
                "fields": {k: d.place[k].field_side for k in sorted(d.place)},
                "bridges": lay.bridge,
                "jogs": sorted(lay.jog),
            })
    return out


@needs_libs
def test_structural_goldens(corpus):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden) == sorted(STRUCTURED)
    for name in STRUCTURED:
        assert structure(corpus[name]) == golden[name], name


@needs_libs
def test_the_placer_gates_on_the_corpus(corpus, tmp_path):
    for name in CORPUS:
        ir = corpus[name]
        start = time.perf_counter()
        result = build(SchematicCompiler(), ir, LIB)
        elapsed = time.perf_counter() - start
        assert result.score is not None and result.fallbacks == () and result.score[:2] == (0, 0), (name, result.notes[0])
        path = tmp_path / f"{name}.kicad_sch"
        path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
        g = schematic_gates(path, ir)
        assert (g["near_symbol_hits"], g["parallel_hits"]) == (0, 0), name
        assert g["supply_above_ground"]["ok"] == g["supply_above_ground"]["of"] or name in ("atmega128", "stm32f407"), (name, g)
        if name in WIRE_09:
            assert g["crossings"] == 0 and g["paper"] == "A4" and g["wire_mm"] <= WIRE_09[name], (name, g)
        if name == "astable":
            assert g["crossings"] <= 3 and g["paper"] == "A4", g
        if name == "analog_piano":
            assert g["paper"] in ("A4", "A3") and g["median_connected_mm"] <= 40 and g["wire_rmst_ratio"] <= 1.5 and g["wire_mm"] < 1500, g
            assert g["blocks_left_to_right"] == {"ok": 4, "of": 4} and elapsed < 5, (g, elapsed)
        if name in ("atmega128", "stm32f407"):
            assert elapsed < 10, (name, elapsed)
        if name == "amplifier":
            places = {str(sexpr.find(s, "property")[2]): float(sexpr.find(s, "at")[2]) for s in sexpr.find_all(result.node, "symbol") if sexpr.find(s, "lib_id")}
            assert places["R2"] < places["U1"]  # feedback above the amplifier


# --------------------------------------------------------------------------- blocks: units, repeated ids, order


def with_blocks(ir: CircuitIR, blocks_: list[tuple[str, list[str]]]) -> CircuitIR:
    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="placer test")
    for bid, refs in blocks_:
        ir.topology.blocks.append(Block(id=bid, function=bid, component_refs=refs, provenance=prov))
    return ir


@needs_libs
def test_a_moved_amplifier_unit_carries_the_signal_flow_between_its_blocks():
    ir = circuit("units", [("U1", "MCP6002", "Amplifier_Operational", "MCP6002-xSN"), ("R1", "10k", "Device", "R"),
                           ("R2", "100k", "Device", "R"), ("C1", "100n", "Device", "C"), ("R3", "10k", "Device", "R"),
                           ("R4", "10k", "Device", "R"), ("R5", "10k", "Device", "R"), ("J1", "IN", "Connector_Generic", "Conn_01x03")],
                 [("VCC", "power", ("U1.8", "J1.1", "R3.1")), ("GND", "ground", ("U1.4", "U1.3", "J1.3")), ("IN", "signal", ("J1.2", "R1.1")),
                  ("AM", "signal", ("R1.2", "U1.2", "R2.1")), ("AOUT", "signal", ("U1.1", "R2.2", "C1.1")),
                  ("BP", "signal", ("C1.2", "U1.5", "R3.2", "R4.1")), ("BM", "signal", ("U1.6", "R5.1")), ("BOUT", "signal", ("U1.7", "R5.2", "R4.2"))])
    ir = with_blocks(ir, [("front", ["J1", "U1", "R1", "R2", "C1"]), ("back", ["R3", "R4", "R5"])])
    model, _ = prepared(ir)
    members = {b.id: b.members for b in model.blocks}
    assert "U1:2" in members["back"] and {"U1:1", "U1:3"} <= set(members["front"])
    assert "block order: BP runs from block front to block back after unit U1B moved" in model.notes
    assert [(b.id, b.layer) for b in model.blocks] == [("front", 0), ("back", 1)]  # the moved unit keeps front -> back


@needs_libs
def test_a_block_id_declared_twice_is_one_block_with_a_note():
    ir = circuit("twice", [("R1", "1k", "Device", "R"), ("R2", "1k", "Device", "R"), ("R3", "1k", "Device", "R"),
                           ("R4", "1k", "Device", "R"), ("J1", "IO", "Connector_Generic", "Conn_01x03")],
                 [("VCC", "power", ("J1.1", "R1.1", "R3.1")), ("GND", "ground", ("J1.3", "R2.2", "R4.2")),
                  ("A", "signal", ("R1.2", "R2.1", "J1.2")), ("B", "signal", ("R3.2", "R4.1"))])
    ir = with_blocks(ir, [("a", ["R1", "R2"]), ("a", ["R3", "R4"]), ("io", ["J1"])])
    model, _ = prepared(ir)
    assert [(b.id, b.members) for b in model.blocks] == [("a", ("R1", "R2", "R3", "R4")), ("io", ("J1",))]
    assert "block a: declared more than once; its parts are drawn in one block" in model.notes


def test_layers_take_the_longest_path_over_a_shallow_and_a_deep_predecessor():
    bs = [BlockInfo(i, ()) for i in "abcd"]
    assert blocks.layers(bs, {"a": {"b", "d"}, "b": {"c"}, "c": {"d"}, "d": set()}) == {"a": 0, "b": 1, "c": 2, "d": 3}


@needs_libs
def test_rails_derive_through_a_diode_a_regulator_and_a_filter(corpus):
    model, _ = prepared(corpus["atmega128"])
    assert rails.derivation(model) == {"VIN_RAW": (None, None, 0), "VIN": ("VIN_RAW", "D1", 0), "+5V": ("VIN", "U2", 0), "AVCC": ("+5V", "L1", 0)}


@needs_libs
def test_band_wrap_chooses_its_cuts_over_the_whole_spine(corpus):
    model, ctx = prepared(corpus["analog_piano"])
    drawings = [d for b in model.blocks for d in pipeline.draw_block(ctx, b, recipe_order(), [])]
    index = {d.block: i for i, d in enumerate(drawings)}
    for block, w in (("power", 30), ("keyboard", 30), ("oscillator", 60), ("driver", 30)):
        drawings[index[block]] = replace(drawings[index[block]], box=(0, 0, w, 10))
    spine = [index[b] for b in ("power", "keyboard", "oscillator", "driver")]
    # band by band, the free cut after power would leave keyboard | oscillator | driver: three bands; two are enough
    assert arrange.cut(model, drawings, spine, 100) == [spine[:2], spine[2:]]


# --------------------------------------------------------------------------- mirror, taps, bundles, buses, labels


def through_x(d, model, key: str) -> int:
    g = model.glyphs[key]
    if g.through() is not None and len(g.pins) == 3:
        return pin_cell(g, g.through()[0].number, (d.place[key].x, d.place[key].y), d.place[key].orient)[0]
    return d.place[key].x


@needs_libs
def test_the_astable_is_two_mirrored_halves_loads_outboard_with_the_cross_coupling_between_them(corpus, tmp_path):
    (lay,) = layouts(corpus["astable"])["astable"]
    assert lay.half == {"R1": "L", "Q1": "L", "C1": "L", "R4": "L", "R3": "R", "C2": "R", "R2": "R", "Q2": "R"}
    assert [lay.members[s] for s in lay.order] == [["R1", "Q1"], ["C1"], ["R4"], ["R3"], ["C2"], ["R2", "Q2"]]
    d = lay.draw(False)
    model = lay.model
    axis = through_x(d, model, "R1") + through_x(d, model, "R2")
    for a, b in (("Q1", "Q2"), ("R4", "R3"), ("C1", "C2")):
        assert through_x(d, model, a) + through_x(d, model, b) == axis and d.place[a].y == d.place[b].y, (a, b)
    assert pin_side(model.glyphs["Q1"], "2", d.place["Q1"].orient) == "R" and pin_side(model.glyphs["Q2"], "2", d.place["Q2"].orient) == "L"
    assert d.place["R1"].field_side == "L" and d.place["R2"].field_side == "R"  # fields face outward
    assert len(d.hints["Q1_B"]) == 4 and len(d.hints["Q2_B"]) == 4  # the X: each base wire turns once around the other
    result = build(SchematicCompiler(), corpus["astable"], LIB)
    path = tmp_path / "astable.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    g = schematic_gates(path, corpus["astable"])
    assert g["crossings"] == 1 and g["wire_rmst_ratio"] <= 1.5, g  # one crossing: the X


@needs_libs
def test_the_piano_oscillator_is_mirrored_with_its_bias_on_the_axis(corpus):
    (lay,) = layouts(corpus["analog_piano"])["oscillator"]
    assert lay.half == {"R11": "L", "Q2": "L", "C2": "L", "R10": "R", "Q1": "R", "C1": "R"}
    d = lay.draw(False)
    model = lay.model
    axis = through_x(d, model, "R11") + through_x(d, model, "R10")
    assert through_x(d, model, "Q2") + through_x(d, model, "Q1") == axis and d.place["C2"].x + d.place["C1"].x == axis
    assert 2 * d.place["R9"].x == axis and d.place["C1"].y == d.place["C2"].y


@needs_libs
def test_two_like_circuits_that_share_only_a_rail_are_not_mirrored_and_cross_coupled_ones_are():
    parts = [("Q1", "2N3904", "Transistor_BJT", "2N3904"), ("Q2", "2N3904", "Transistor_BJT", "2N3904"), ("R1", "1k", "Device", "R"),
             ("R2", "1k", "Device", "R"), ("R3", "10k", "Device", "R"), ("R4", "10k", "Device", "R"), ("J1", "IN", "Connector_Generic", "Conn_01x04")]
    nets = [("VCC", "power", ("R1.1", "R2.1", "R3.1", "R4.1", "J1.1")), ("GND", "ground", ("Q1.1", "Q2.1", "J1.4")),
            ("C1", "signal", ("R1.2", "Q1.3", "J1.2")), ("C2", "signal", ("R2.2", "Q2.3", "J1.3")),
            ("B1", "signal", ("R3.2", "Q1.2")), ("B2", "signal", ("R4.2", "Q2.2"))]
    twins = circuit("twins", parts, nets)
    assert all(lay.half == {} for lay in every(twins))
    coupled = circuit("coupled", parts + [("C3", "10n", "Device", "C"), ("C4", "10n", "Device", "C")],
                      [*nets[:2], ("C1", "signal", ("R1.2", "Q1.3", "J1.2", "C3.1")), ("C2", "signal", ("R2.2", "Q2.3", "J1.3", "C4.1")),
                       ("B1", "signal", ("R3.2", "Q1.2", "C4.2")), ("B2", "signal", ("R4.2", "Q2.2", "C3.2"))])
    assert any(lay.half.get("Q1") and lay.half.get("Q2") for lay in every(coupled))


@needs_libs
def test_a_transistor_fed_from_a_divider_tap_takes_the_tap_height_and_a_bundled_connector_is_centred(corpus):
    (lay,) = layouts(corpus["analog_piano"])["driver"]
    d = lay.draw(False)
    model = lay.model
    base = pin_y(d, model, "Q4", "2")
    assert pin_y(d, model, "R13", "1") < base < pin_y(d, model, "R14", "1")  # Q4B leaves the R13 / R14 tap straight
    assert pin_y(d, model, "J2", "1") < d.place["D1"].y < pin_y(d, model, "J2", "2")  # the speaker centred on its diode
    assert "SPKP" in d.hints and "SPKN" in d.hints


@needs_libs
def test_parts_into_a_summing_bus_stand_in_columns_over_it():
    parts = [("U1", "TL081", "Amplifier_Operational", "TL081"), ("R9", "100k", "Device", "R"), ("J1", "OSC", "Connector_Generic", "Conn_01x06"),
             ("J2", "PWR", "Connector_Generic", "Conn_01x02")]
    nets = [("VCC", "power", ("J2.1", "U1.7")), ("GND", "ground", ("J2.2", "U1.4", "U1.3", "J1.6")),
            ("SUM", "signal", ("U1.2", "R9.1", *(f"R{i}.2" for i in range(1, 5)))), ("OUT", "signal", ("U1.6", "R9.2", "J1.5"))]
    for i in range(1, 5):
        parts += [(f"SW{i}", "KEY", "Switch", "SW_Push"), (f"R{i}", "15k", "Device", "R")]
        nets += [(f"OSC{i}", "signal", (f"J1.{i}", f"SW{i}.1")), (f"KEY{i}", "signal", (f"SW{i}.2", f"R{i}.1"))]
    ir = circuit("bus", parts, nets)
    u1 = next(c for c in ir.components if c.ref == "U1")
    u1.pins = [p.model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT}) if p.number in ("1", "5", "8") else p for p in u1.pins]
    (lay,) = [lay for lay in every(ir) if "R1" in lay.keys]
    cols = [ms for s, ms in lay.members.items() if s.startswith("col:")]
    for i in range(1, 5):
        assert [f"SW{i}", f"R{i}"] in cols, cols
        assert lay.row[f"c:R{i}"] < lay.row["n:SUM"]


def test_a_name_label_never_lies_along_a_wire_of_its_own_net():
    stub = stub_label("N", (10, 10), "R", "name")  # a pin stub leaving (10, 10) to the right, the text further right
    assert _on_wire(stub, [((10, 10), (20, 10))])  # the pin's wire runs on to the right, through the label
    assert not _on_wire(stub, [((10, 10), (10, 20))])  # it leaves downward: the label is clear of it
    assert not _on_wire(stub, [((10, 10), (12, 10))])  # its own stub


@needs_libs
def test_no_wire_doubles_another_on_the_corpus(corpus):
    for name in ("amplifier", "rc_lowpass", "analog_piano", "astable"):
        result = build(SchematicCompiler(), corpus[name], LIB)
        wires = []
        for w in sexpr.find_all(result.node, "wire"):
            a, b = [tuple(round(float(v) / 1.27) for v in xy[1:3]) for xy in sexpr.find_all(sexpr.find(w, "pts"), "xy")]
            wires.append((a, b))
        for i, (a, b) in enumerate(wires):
            for c, e in wires[i + 1:]:
                if a[0] == b[0] == c[0] == e[0]:
                    assert min(max(a[1], b[1]), max(c[1], e[1])) <= max(min(a[1], b[1]), min(c[1], e[1])), (name, (a, b), (c, e))
                if a[1] == b[1] == c[1] == e[1]:
                    assert min(max(a[0], b[0]), max(c[0], e[0])) <= max(min(a[0], b[0]), min(c[0], e[0])), (name, (a, b), (c, e))
