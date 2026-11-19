"""Textbook 1.0 layout API: pin geometry, measurement, kinds, the always-succeeding path, the fallback ladder, determinism,
and real KiCad evidence (ERC, netlist, DRC schematic parity) for every corpus design."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler
from ai_eda.compilers.schematic_layout import pin_position
from ai_eda.compilers.textbook import VERSION, LayoutFailure, build
from ai_eda.compilers.textbook import columns, pipeline, route
from ai_eda.compilers.textbook.emit import emit, guard, junctions
from ai_eda.compilers.textbook.glyphs import TRANSFORMS, body_box, generated, glyph_boxes, label_box, mm, orient_two, overlaps, pin_cell, pin_side
from ai_eda.compilers.textbook.labels import _gap, name_nets
from ai_eda.compilers.textbook.model import BlockInfo, Drawing, Hit, Kind, Label, Orient, Place, Role, Sheet, Wiring, load_model
from ai_eda.compilers.textbook.recipes import GRID, RECIPE_ORDER, Ctx
from ai_eda.errors import CompileError
from ai_eda.ir import Net, PinRef
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from tests.test_schematic_compiler import netlist_nets
from tests.textbook_corpus import CORPUS, corpus_ir, grid_board

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")
ROOT = Path(__file__).resolve().parent.parent
#: the 0.9 layout's body boxes (8 transforms at cell (40, 30)) and two-terminal orientations of every corpus glyph,
#: recorded from its ``Drawing`` before the 0.9 code was deleted
ORACLE09 = json.loads((Path(__file__).parent / "data" / "orient09_golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Every corpus design: its IR and its layout model."""
    base = tmp_path_factory.mktemp("corpus")
    irs = {name: corpus_ir(name, base / name, LIB) for name in CORPUS}
    return {name: (ir, load_model(SchematicCompiler(), ir, LIB)) for name, ir in irs.items()}


def hook(ir) -> tuple[str, list[str]]:
    node, notes = SchematicCompiler().build_with_notes(ir, LIB)
    return sexpr.dumps(node), notes


# --------------------------------------------------------------------------- geometry against the existing helpers


@needs_libs
def test_pin_cell_is_pin_position_for_every_corpus_pin_under_all_eight_transforms(corpus):
    assert len(TRANSFORMS) == len(set(TRANSFORMS)) == 8
    checked = 0
    for _, model in corpus.values():
        for g in model.glyphs.values():
            for o in TRANSFORMS:
                for p in g.pins:
                    want = pin_position(mm(7), mm(11), o.rot, o.mirror, g.lib_pin(p.number))
                    got = pin_cell(g, p, (7, 11), o)
                    assert (mm(got[0]), mm(got[1])) == want, (g.key, p.number, o)
                    checked += 1
    assert checked > 3000


@needs_libs
def test_off_grid_library_pin_is_a_compile_error(corpus):
    g = corpus["divider"][1].glyphs["R1"]
    moved = replace(g.symbol, pins=[replace(p, x=p.x + 0.5) for p in g.symbol.pins])
    with pytest.raises(CompileError, match="off the 1.27 mm grid"):
        pin_cell(replace(g, symbol=moved), g.pins[0], (0, 0), Orient())


@needs_libs
def test_orient_two_and_measure_equal_the_09_rules(corpus):
    assert [[o.rot, o.mirror] for o in TRANSFORMS] == ORACLE09["transforms"]
    pairs = boxes = 0
    for name, (ir, model) in corpus.items():
        old = ORACLE09["corpus"][name]
        assert sorted(old) == sorted(model.glyphs), name
        for g in model.glyphs.values():
            for o, box in zip(TRANSFORMS, old[g.key]["boxes"]):
                assert list(body_box(g, Place(40, 30, o))) == box, (name, g.key, o)
                boxes += 1
            if g.kind is not Kind.TWO and len(g.pins) != 2:
                continue
            for net in g.nets():
                for vertical in (True, False):
                    rot = old[g.key]["pairs"][f"{net}|{'v' if vertical else 'h'}"]
                    expected = None if rot is None else Orient(rot)
                    try:
                        got = orient_two(g, net, vertical)
                    except LayoutFailure:
                        got = None
                    assert got == expected, (name, g.key, net, vertical)
                    pairs += 1
    assert pairs > 100 and boxes > 1000


@needs_libs
def test_kinds_roles_and_pin_roles_come_from_the_library(corpus):
    piano = corpus["analog_piano"][1].glyphs
    q = piano["Q1"]
    assert q.kind is Kind.THRU3 and [p.name for p in q.through()] == ["C", "E"] and q.control().name == "B"
    assert pin_side(q, q.through()[0], Orient()) == "T" and pin_side(q, q.control(), Orient()) == "L"
    assert piano["RV1"].kind is Kind.THRU3 and piano["RV1"].control().number == "2"
    assert piano["SW9"].kind is Kind.OTHER
    assert piano["J1"].kind is Kind.CONN and piano["J2"].kind is Kind.CONN
    assert piano["SW1"].kind is Kind.TWO and piano["R1"].kind is Kind.TWO
    assert piano["C1"].dc_blocking and not piano["R1"].dc_blocking and not piano["D1"].dc_blocking
    _, amp = corpus["amplifier"]
    u1 = amp.glyphs["U1"]
    assert u1.kind is Kind.AMP and amp.glyphs["J1"].kind is Kind.CONN
    out, upper, lower = u1.amp_io()
    assert (out.number, upper.name, lower.name) == ("6", "+", "-")
    assert {p.name for p in u1.power_pins()} == {"V+", "V-"}
    assert amp.nets["VCC"].role is Role.SUPPLY and amp.nets["GND"].role is Role.GROUND and amp.nets["SUM"].role is Role.SIGNAL
    assert corpus["atmega128"][1].glyphs["U1"].kind is Kind.IC and corpus["stm32f407"][1].glyphs["U1"].kind is Kind.IC


@needs_libs
def test_a_power_net_on_bottom_power_inputs_only_is_negative(tmp_path):
    ir = corpus_ir("amplifier", tmp_path, LIB)
    gnd = ir.net("GND")
    gnd.pins = [p for p in gnd.pins if (p.component_ref, p.pin_number) not in {("U1", "4"), ("J1", "4")}]
    ir.nets.append(Net(name="VEE", kind="power", provenance=gnd.provenance,
                       pins=[PinRef(component_ref="U1", pin_number="4"), PinRef(component_ref="J1", pin_number="4")]))
    model = load_model(SchematicCompiler(), ir, LIB)
    assert model.nets["VEE"].role is Role.NEGATIVE and model.nets["VCC"].role is Role.SUPPLY


def test_overlaps_returns_index_pairs_of_positive_area_only():
    boxes = [(0, 0, 2, 2), (1, 1, 3, 3), (2, 0, 4, 1), (10, 10, 11, 11), (0.5, 0.5, 0.6, 0.6)]
    assert overlaps(boxes) == [(0, 1), (0, 4)]  # (1, 2) only touch
    assert overlaps([]) == []


def test_recipe_order_ends_with_the_block_grid():
    assert [r.name for r in RECIPE_ORDER] == ["test_points", "hub", "opamp_stage", "cross_coupled", "array", "supply_chain", "columns", "grid"]
    assert RECIPE_ORDER[-1] is GRID


# --------------------------------------------------------------------------- the always-succeeding path


#: (design, block) -> the recipes that draw it, for the blocks a motif recipe claims
MOTIF_BLOCKS = {
    ("led", "inferred.J1"): {"supply_chain"},
    ("lc_lowpass", "lowpass"): {"array"},
    ("astable", "astable"): {"cross_coupled"},
    ("amplifier", "inferred.J1"): {"opamp_stage"},
    ("analog_piano", "power"): {"supply_chain"},
    ("analog_piano", "keyboard"): {"array"},
    ("analog_piano", "oscillator"): {"cross_coupled"},
}


@needs_libs
def test_every_corpus_design_draws_with_notes_naming_each_block_recipe(corpus):
    for name, (ir, model) in corpus.items():
        start = time.perf_counter()
        result = build(SchematicCompiler(), ir, LIB)
        elapsed = time.perf_counter() - start
        assert result.score is not None and result.fallbacks == (), (name, result.notes[0])
        head = result.notes[0]
        assert head.startswith(f"layout: textbook {VERSION}; blocks ") and "fallbacks: none" in head
        # changed deliberately with the general placer: the columns recipe draws every block no motif claims; the block
        # grid draws only an IC of 20 pins or more and the glyphs on its signal nets (no recipe claims them yet)
        # changed deliberately with the motif recipes: a block that holds a motif names its recipe (MOTIF_BLOCKS); the
        # power blocks of the MCU boards are supply chains drawn by the columns engine's rail rows
        big = [g for g in model.glyphs.values() if g.kind is Kind.IC and len(g.pins) >= columns.HUB_PINS]
        nets = {n for g in big for n in g.net.values() if model.nets[n].role is Role.SIGNAL}
        for block in pipeline.blocks_mod.partition(replace(model, notes=[])):
            left = [k for k in block.members if model.glyphs[k] in big or set(model.glyphs[k].net.values()) & nets]
            drawn = set(head.split(f" {block.id}=", 1)[1].split(",")[0].split(";")[0].split("+"))
            want = ({"columns"} if len(left) < len(block.members) else set()) | ({"grid"} if left else set())
            if (name, block.id) in MOTIF_BLOCKS:
                want = MOTIF_BLOCKS[(name, block.id)]
            elif big and len(left) < len(block.members):  # the part off the IC: one or both of the two rail-row drawings
                rest = drawn - {"grid"}
                assert rest and rest <= {"columns", "supply_chain"}, (name, block.id, drawn)
                want = rest | ({"grid"} if left else set())
            assert drawn == want, (name, block.id, drawn)
        assert result.score[0] == 0  # no hard guard finding
        comment = next(c for c in sexpr.find_all(sexpr.find(result.node, "title_block"), "comment") if str(c[1]) == "1")
        assert str(comment[2]) == f"generated by compiler.kicad_sch 1.0 textbook from IR project {ir.project.id}"
        assert elapsed < 10, (name, elapsed)


@needs_libs
@pytest.mark.parametrize("force", ["router", "grid", "guard"])
def test_a_forced_layout_failure_ends_in_the_whole_sheet_table_with_a_note(tmp_path, monkeypatch, force):
    ir = corpus_ir("divider", tmp_path, LIB)

    def fail(*args, **kwargs):
        raise LayoutFailure(f"forced {force} failure")

    if force == "router":
        monkeypatch.setattr(route, "route", fail)
    elif force == "grid":  # every recipe that could draw the divider fails (the columns recipe draws it since the placer)
        monkeypatch.setattr(GRID, "draw", fail)
        monkeypatch.setattr(columns.COLUMNS, "draw", fail)
    else:
        real = pipeline.guard
        monkeypatch.setattr(pipeline, "guard", lambda *a, **k: [*real(*a, **k), Hit("forced finding", ("R1",))])
    text, notes = hook(ir)
    assert "generated by compiler.kicad_sch 1.0 textbook (table fallback) from IR project divider" in text
    assert notes[0].startswith("layout: textbook 1.0; whole-sheet table fallback: ") and "whole-sheet table" in notes[0].split("fallbacks: ")[1]
    assert "forced" in notes[0]
    legacy = sexpr.dumps(SchematicCompiler("legacy").build(ir, LIB))
    assert text.replace("1.0 textbook (table fallback) from", "0.8 from") == legacy  # the legacy table, nothing else changed


@needs_libs
def test_a_failing_recipe_hands_its_block_to_the_next_and_a_guilty_block_is_redrawn_as_grid(tmp_path, monkeypatch):
    ir = corpus_ir("divider", tmp_path, LIB)
    monkeypatch.setattr(columns.COLUMNS, "claim", lambda model, block, free: free)

    def unavailable(ctx, block, members):  # the columns recipe draws the divider since the placer: forced to fail here
        raise LayoutFailure("columns: not available", block.id)

    monkeypatch.setattr(columns.COLUMNS, "draw", unavailable)
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and any("columns failed (columns: not available), next recipe" in f for f in result.fallbacks)
    assert "columns failed" in result.notes[0]
    # a block columns drew that the guard faults is redrawn as grid, once, and the sheet passes
    monkeypatch.setattr(columns.COLUMNS, "draw", lambda ctx, block, members: [replace(d, recipe="columns") for d in GRID.draw(ctx, block, members)])
    real = pipeline.guard

    def guard(ctx, sheet, *a, **k):
        hits = real(ctx, sheet, *a, **k)
        flagged = [k for d in sheet.drawings if d.recipe == "columns" for k in d.place]
        return hits + [Hit("forced finding", tuple(flagged))] if flagged else hits

    monkeypatch.setattr(pipeline, "guard", guard)
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and result.score[0] == 0
    assert any(f.startswith("block grid for ") for f in result.fallbacks)
    assert "=grid" in result.notes[0] and "=columns" not in result.notes[0]
    # a LayoutFailure at sheet level that names a block redraws that block as grid, once
    monkeypatch.setattr(pipeline, "guard", real)
    real_route = route.route

    def failing(ctx, sheet, obstacles):
        if any(d.block == "divider" and d.recipe == "columns" for d in sheet.drawings):
            raise LayoutFailure("forced router failure", "divider")
        return real_route(ctx, sheet, obstacles)

    monkeypatch.setattr(route, "route", failing)
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and "block grid for divider (forced router failure)" in result.fallbacks
    assert "divider=grid" in result.notes[0] and "=columns" in result.notes[0]


@needs_libs
def test_only_the_blocks_a_hard_finding_names_are_redrawn_as_grid(tmp_path, monkeypatch):
    ir = corpus_ir("divider", tmp_path, LIB)
    monkeypatch.setattr(columns.COLUMNS, "claim", lambda model, block, free: free)
    monkeypatch.setattr(columns.COLUMNS, "draw", lambda ctx, block, members: [replace(d, recipe="columns") for d in GRID.draw(ctx, block, members)])
    real = pipeline.guard

    def guard_r1(ctx, sheet, *a, **k):
        hits = real(ctx, sheet, *a, **k)
        faulted = any(d.recipe == "columns" and "R1" in d.place for d in sheet.drawings)
        return hits + [Hit("forced finding", ("R1",))] if faulted else hits

    monkeypatch.setattr(pipeline, "guard", guard_r1)
    result = build(SchematicCompiler(), ir, LIB)
    blocks = pipeline.blocks_mod.partition(load_model(SchematicCompiler(), ir, LIB))
    assert len(blocks) > 1 and result.score is not None
    for block in blocks:
        assert f"{block.id}={'grid' if 'R1' in block.members else 'columns'}" in result.notes[0], result.notes[0]


@needs_libs
@pytest.mark.parametrize("fault", ["pin", "net", "symbol"])
def test_ir_and_library_faults_stay_compile_errors(tmp_path, fault):
    ir = corpus_ir("divider", tmp_path, LIB)
    if fault == "pin":
        ir.component("R1").pins.pop()
    elif fault == "net":
        ir.net("GND").pins.append(ir.net("VIN").pins[0])
    else:
        ir.component("R1").symbol.verified = False
    with pytest.raises(CompileError):
        hook(ir)


# --------------------------------------------------------------------------- guards, junctions, names and scores on hand-made sheets
# The divider: R1 VIN (pin 1, top) - VOUT (pin 2); R2 VOUT - GND; J1 VIN / VOUT / GND. Unturned, a resistor's pins sit 3 cells
# above and below its origin.

DIVIDER = {"R1": Place(40, 40), "R2": Place(60, 40), "J1": Place(100, 40)}


def _ctx(model) -> Ctx:
    return Ctx(SchematicCompiler(), LIB, model, [])


def _sheet(place: dict[str, Place]) -> Sheet:
    return Sheet((Drawing("t", "test", place),), ((0, 0),))


def _hard(model, place, segments=None, labels=()) -> list[str]:
    return [str(h) for h in guard(_ctx(model), _sheet(place), Wiring(segments=dict(segments or {})), list(labels)) if h.hard]


@needs_libs
def test_the_guard_finds_every_connection_kicad_would_make_that_the_ir_does_not(corpus):
    model = corpus["divider"][1]
    r1 = model.glyphs["R1"]
    assert pin_cell(r1, "1", (40, 40), Orient()) == (40, 37) and pin_cell(r1, "2", (40, 40), Orient()) == (40, 43)
    vout = [((40, 43), (40, 46)), ((40, 46), (52, 46)), ((52, 46), (52, 34)), ((52, 34), (60, 34)), ((60, 34), (60, 37))]
    assert _hard(model, DIVIDER, {"VOUT": vout}) == []
    # a wire through a foreign pin, straight or diagonal
    for gnd in ([((36, 43), (44, 43))], [((37, 40), (43, 46))]):
        assert any("touches pin R1.2 (VOUT) at (40, 43)" in h for h in _hard(model, DIVIDER, {"GND": gnd})), gnd
    # a wire on another net's wire end, straight or diagonal; a plain crossing is no connection
    assert any("touches a wire end of GND at (25, 50)" in h for h in _hard(model, DIVIDER, {"VIN": [((20, 50), (30, 50))], "GND": [((25, 50), (25, 55))]}))
    assert any("touches a wire end of GND at (25, 55)" in h for h in _hard(model, DIVIDER, {"VIN": [((20, 50), (30, 60))], "GND": [((25, 55), (25, 65))]}))
    assert _hard(model, DIVIDER, {"VIN": [((20, 50), (30, 50))], "GND": [((25, 45), (25, 55))]}) == []
    # labels: off every wire and pin of their net, or on another net's wire
    assert _hard(model, DIVIDER, labels=[Label("VIN", (20, 60), 0, "left", "name")]) == ["label VIN at (20, 60) is on no wire or pin of its net"]
    hits = _hard(model, DIVIDER, {"GND": [((25, 45), (25, 55))]}, [Label("VIN", (25, 50), 0, "left", "name")])
    assert "label VIN at (25, 50) touches GND" in hits
    # overlapping bodies
    assert any(h.startswith("overlap: body:R1 and body:R2") for h in _hard(model, {**DIVIDER, "R2": Place(41, 41)}))


@needs_libs
def test_the_guard_refuses_pins_of_different_nets_on_one_point(corpus, tmp_path):
    model = corpus["divider"][1]
    r2 = model.glyphs["R2"]
    turned = {**DIVIDER, "R2": Place(40, 46, Orient(180))}  # R2.2 (GND) on R1.2 (VOUT); the bodies only touch
    assert pin_cell(r2, "2", (40, 46), Orient(180)) == (40, 43)
    hits = [h for h in guard(_ctx(model), _sheet(turned), Wiring(), []) if h.hard]
    assert hits == ["pins R1.2 (VOUT), R2.2 (GND) meet at (40, 43)"] and hits[0].owners == ("R1", "R2")
    assert _hard(model, {**DIVIDER, "R2": Place(40, 46)}) == []  # R2.1 (VOUT) on R1.2 (VOUT): one net, a direct connection
    if not CLI.available():
        return
    # the premise: KiCad joins the two pins into one net
    path = tmp_path / "pins.kicad_sch"
    path.write_text(sexpr.dumps(emit(_ctx(model), _sheet(turned), Wiring(), [], ("A4", 297.0, 210.0), "pins")), encoding="utf-8", newline="\n")
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    assert {("R1", "2"), ("R2", "2")} in [set(pins) for pins in nets.values()]


def test_junction_dots_at_tees_and_at_pins_with_two_wires():
    pin = {(0, 0): [("R1", "A", "R1.1")]}
    assert junctions({"A": [((0, 0), (10, 0)), ((5, 0), (5, 5))]}, {}) == {"A": [(5, 0)]}  # a T
    assert junctions({"A": [((0, 0), (5, 0)), ((5, 0), (10, 0))]}, {}) == {}  # a straight joint
    assert junctions({"A": [((0, 0), (10, 0)), ((5, -5), (5, 5))]}, {}) == {}  # a crossing
    assert junctions({"A": [((0, 0), (5, 0)), ((0, 0), (0, 5))]}, pin) == {"A": [(0, 0)]}  # two wires at a pin
    assert junctions({"A": [((0, 0), (5, 0))]}, pin) == {}


@needs_libs
def test_spacing_findings_are_scored_not_refused(corpus):
    model = corpus["divider"][1]

    def soft(segments):
        return [str(h) for h in guard(_ctx(model), _sheet(DIVIDER), Wiring(segments=segments), []) if not h.hard]

    assert body_box(model.glyphs["R1"], DIVIDER["R1"])[2] == pytest.approx(51.816)
    assert soft({"GND": [((41, 38), (41, 42))]}) == ["spacing: wire of GND (41, 38)-(41, 42) within 1.27 mm of R1"]  # 0.25 mm off R1
    assert soft({"GND": [((42, 38), (42, 42))]}) == []  # 1.52 mm off
    assert soft({"VIN": [((20, 60), (30, 60))], "GND": [((22, 61), (34, 61))]}) == ["spacing: VIN and GND run parallel 1.27 mm apart (h 60)"]
    assert soft({"VIN": [((20, 60), (30, 60))], "GND": [((22, 62), (34, 62))]}) == []


@needs_libs
def test_name_labels_one_per_wire_component_on_its_lowest_horizontal_wire_clear_of_boxes(corpus):
    model = corpus["divider"][1]
    place = {**DIVIDER, "R2": Place(60, 52)}  # R2.1 (VOUT) at (60, 49)
    sheet = _sheet(place)
    wiring = Wiring(segments={"VOUT": [((40, 43), (40, 49)), ((40, 49), (60, 49))]})
    blocker = (mm(41), mm(50), mm(48), mm(53))  # under the left end of the VOUT wire
    labels = name_nets(_ctx(model), sheet, wiring, [*glyph_boxes(model, sheet), ("body:blocker", blocker)])
    assert sorted(lab.net for lab in labels) == ["GND", "GND", "VIN", "VIN", "VOUT", "VOUT"]  # J1 stands apart: two components each
    wire = [lab for lab in labels if lab.net == "VOUT" and lab.at[1] == 49 and 40 < lab.at[0] < 60]
    assert len(wire) == 1 and wire[0].angle == 270 and wire[0].stub is None  # hanging below the horizontal wire
    assert _gap(label_box(wire[0]), blocker) >= 0.5
    assert [h for h in guard(_ctx(model), sheet, wiring, labels) if h.hard] == []


def test_score_counts_findings_crossings_extra_labels_bends_wire_and_area():
    segments = {"A": [((0, 0), (4, 0)), ((4, 0), (4, 3))], "B": [((10, 0), (10, 5))]}
    labels = [Label("A", (2, 0), 270, "right", "name"), Label("A", (4, 2), 0, "left", "name"), Label("B", (10, 2), 0, "left", "name")]
    hits = [Hit("hard"), Hit("soft", hard=False), Hit("soft too", hard=False)]
    assert pipeline.score(hits, Wiring(crossings=2), labels, segments, (0.0, 0.0, 12.7, 25.4)) == (1, 2, 2, 1, 1, 12, 200)


@needs_libs
def test_a_recipe_variant_with_internal_faults_ranks_last(corpus):
    model = corpus["divider"][1]
    ctx = _ctx(model)
    block = BlockInfo("divider", ("R1", "R2"))
    good = GRID.draw(ctx, block, frozenset(block.members))[0]
    bad = replace(good, place={k: Place(0, 0) for k in good.place})  # both resistors on one spot

    class Two:
        name = "two"

        def claim(self, model, block, free):
            return free

        def draw(self, ctx, block, members):
            return [bad, good]

    assert pipeline.rank(ctx, bad) > pipeline.rank(ctx, good)
    assert pipeline.draw_block(ctx, block, (Two(),), [])[0].place == good.place


@needs_libs
def test_generated_power_symbols_name_their_net():
    def glyph(kind, net):
        g = generated(LIB, kind, net, (0, 0), Orient(), "#PWR01")
        return g.lib, g.value, g.names_net

    assert glyph("VCC", "OSC") == ("power:VCC", "OSC", True)
    assert glyph("GND", "GND") == ("power:GND", "GND", True)
    assert glyph("NEG", "VEE") == ("power:VEE", "VEE", True)  # the library's own symbol for that rail
    assert glyph("NEG", "VNEG_STAGE") == ("power:VCC", "VNEG_STAGE", True)
    assert glyph("FLAG", "VCC") == ("power:PWR_FLAG", "PWR_FLAG", False)


# --------------------------------------------------------------------------- determinism


@needs_libs
@pytest.mark.parametrize("name", ["astable", "amplifier", "analog_piano"])
def test_identical_bytes_twice_and_with_the_ir_lists_reversed(tmp_path, name):
    ir = corpus_ir(name, tmp_path, LIB)
    first, _ = hook(ir)
    assert hook(ir)[0] == first
    ir.components.reverse()
    ir.nets.reverse()
    for n in ir.nets:
        n.pins.reverse()
    for c in ir.components:
        c.pins.reverse()
    assert hook(ir)[0] == first


@needs_libs
def test_identical_bytes_under_any_hash_seed(tmp_path):
    script = (
        "import hashlib, os, sys; sys.path.insert(0, os.getcwd())\n"
        "from pathlib import Path\n"
        "from ai_eda.compilers import SchematicCompiler\n"
        "from ai_eda.tools.kicad import KicadLibrary, sexpr\n"
        "from tests.textbook_corpus import corpus_ir\n"
        "lib = KicadLibrary()\n"
        "for name in ('analog_piano', 'amplifier'):\n"
        "    ir = corpus_ir(name, Path(sys.argv[1]) / name, lib)\n"
        "    print(name, hashlib.sha256(sexpr.dumps(SchematicCompiler().build(ir, lib)).encode()).hexdigest())\n"
    )
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.run([sys.executable, "-c", script, str(tmp_path / seed)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] and outputs[0].count("\n") == 2


# --------------------------------------------------------------------------- real KiCad evidence


def partition(nets):
    return {frozenset(pins) for name, pins in nets.items() if pins and not name.startswith("unconnected-")}


@needs_kicad
@pytest.mark.parametrize("name", CORPUS)
def test_every_corpus_design_passes_erc_and_keeps_the_ir_netlist(tmp_path, name):
    ir = corpus_ir(name, tmp_path / "ir", LIB)
    text, _ = hook(ir)
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(text, encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert erc.details["errors"] == erc.details["warnings"] == [], erc.details
    exported = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    assert partition(exported) == partition({n.name: {(p.component_ref, p.pin_number) for p in n.pins} for n in ir.nets})
    assert {n.name for n in ir.nets if len(n.pins) > 1} <= set(exported)  # every IR net keeps its name


def _parity_ir(name: str, tmp_path: Path):
    if name == "pcb_compiler_board":
        from tests.fixtures_kicad import ROUTABLE_PLACEMENTS, divider_with_connector_ir
        from ai_eda.tools.routing import route_naive

        ir = divider_with_connector_ir(tmp_path / "ir", LIB)
        ir.pcb.placements = list(ROUTABLE_PLACEMENTS)
        ir.pcb.tracks = route_naive(ir, LIB)
        return ir
    return grid_board(corpus_ir(name, tmp_path / "ir", LIB))


@needs_kicad
@pytest.mark.parametrize("name", ["pcb_compiler_board", "analog_piano", "atmega128"])
def test_drc_schematic_parity_holds_with_the_textbook_schematic(tmp_path, name):
    ir = _parity_ir(name, tmp_path)
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": LIB})
    pcb = Path(PCBCompiler().compile(ir, ctx).path)
    text, _ = hook(ir)
    pcb.with_suffix(".kicad_sch").write_text(text, encoding="utf-8", newline="\n")
    res = CLI.run_drc(pcb, tmp_path / "drc.json", schematic_parity=True)
    assert res.details["schematic_parity_checked"], res.details.get("schematic_parity_reason")
    assert res.details["schematic_parity"] == [], res.details["schematic_parity"]
