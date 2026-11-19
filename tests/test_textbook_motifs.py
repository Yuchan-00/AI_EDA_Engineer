"""Textbook 1.0 motif recipes: the op-amp stage, the cross-coupled pair, the arrays (summing bus, chain ladder) and the
supply chain - their detectors (hits and near misses), their structure on the corpus and the op-amp fixtures, and the
gates of the design for the op-amp stages, the astable and the piano."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.textbook import blocks, build, motifs, pipeline, rails
from ai_eda.compilers.textbook.glyphs import body_box, generated_pin, mm, pin_cell
from ai_eda.compilers.textbook.model import LayoutFailure, Orient, load_model
from ai_eda.compilers.textbook.recipes import Ctx, recipe_order
from ai_eda.ir import CircuitIR, Component, LibraryRef, Net, Pin, PinRef, ProjectMeta, Provenance, ProvenanceKind, Topology
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.layout_metrics import schematic_gates
from tests.test_schematic_compiler import netlist_nets
from tests.textbook_cases import AMPLIFIER_VARIANTS, amplifier, amplifier_variant, dual_amplifier
from tests.textbook_corpus import corpus_ir

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")


def prepared(ir: CircuitIR):
    model = load_model(SchematicCompiler(), ir, LIB)
    model = replace(model, blocks=blocks.partition(model), rails=rails.rail_order(model))
    return model, Ctx(SchematicCompiler(), LIB, model, model.notes)


def block(model, name: str):
    return next(b for b in model.blocks if b.id == name)


def sheet_of(ir: CircuitIR):
    """The kept sheet variant of ``ir``, as the pipeline builds it."""
    model, ctx = prepared(ir)
    drawn = {b.id: pipeline.draw_block(ctx, b, recipe_order(), []) for b in model.blocks}
    return model, drawn, pipeline.best_sheet(ctx, model.blocks, drawn)


def circuit(project: str, parts, nets, blocks_: tuple = ()) -> CircuitIR:
    """A small IR from (ref, value, library, symbol), (net, kind, pins "REF.N") and (block id, refs, inputs, outputs)."""
    from ai_eda.ir.topology import Block

    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="motif test")
    ir = CircuitIR(project=ProjectMeta(id=project, name=project), topology=Topology(name=project, provenance=prov))
    for bid, refs, ins, outs in blocks_:
        ir.topology.blocks.append(Block(id=bid, function=bid, component_refs=list(refs), input_nets=list(ins), output_nets=list(outs), provenance=prov))
    for ref, value, lib, name in parts:
        libref = LIB.resolve_symbol(LibraryRef(library=lib, name=name))
        symbol = LIB.load_symbol(libref)
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov,
                                       pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins]))
    for name, kind, pins in nets:
        ir.nets.append(Net(name=name, kind=kind, provenance=prov, pins=[PinRef(component_ref=p.split(".")[0], pin_number=p.split(".")[1]) for p in pins]))
    return ir


def summing_board() -> CircuitIR:
    """Three keys from a header, each through a resistor, into the '-' input of an MCP6002 summer with its feedback
    resistor ('+' on ground, unit B a follower tied as a spare)."""
    parts = [(f"SW{i}", "SW_Push", "Switch", "SW_Push") for i in (1, 2, 3)] + [(f"R{i}", "15k", "Device", "R") for i in (1, 2, 3)]
    parts += [("R4", "10k", "Device", "R"), ("U1", "MCP6002", "Amplifier_Operational", "MCP6002-xSN"),
              ("J1", "IO", "Connector_Generic", "Conn_01x06")]
    nets = [(f"IN{i}", "signal", (f"J1.{i}", f"SW{i}.1")) for i in (1, 2, 3)]
    nets += [(f"K{i}", "signal", (f"SW{i}.2", f"R{i}.1")) for i in (1, 2, 3)]
    nets += [("SUM", "signal", ("R1.2", "R2.2", "R3.2", "R4.1", "U1.2")), ("OUT", "signal", ("R4.2", "U1.1", "J1.4")),
             ("VCC", "power", ("J1.5", "U1.8")), ("GND", "ground", ("J1.6", "U1.3", "U1.4", "U1.5")), ("SPARE", "signal", ("U1.6", "U1.7"))]
    blocks_ = (("io", ["J1"], [], ["IN1", "IN2", "IN3"]),
               ("mixer", ["SW1", "SW2", "SW3", "R1", "R2", "R3", "R4", "U1"], ["IN1", "IN2", "IN3"], ["OUT"]))
    return circuit("summing", parts, nets, blocks_)


# --------------------------------------------------------------------------- detectors


@pytest.fixture(scope="module")
def piano(tmp_path_factory):
    return corpus_ir("analog_piano", tmp_path_factory.mktemp("piano"), LIB)


@needs_libs
def test_cross_coupled_finds_the_piano_oscillator_with_an_empty_bias_slot(piano):
    model, _ = prepared(piano)
    found = motifs.cross_coupled(model, set(block(model, "oscillator").members))
    assert set(found["q"]) == {"Q1", "Q2"}
    net = {k: set(model.glyphs[k].net.values()) for k in found["coupling"]}
    assert net == {"C1": {"Q1C", "Q2B"}, "C2": {"Q2C", "Q1B"}}
    by_q = dict(zip(found["q"], zip(found["loads"], found["biases"])))
    assert by_q["Q1"] == (["R10"], ["R9"]) and by_q["Q2"] == (["R11"], [])  # Q2B's bias slot is empty: the keyboard feeds it


@needs_libs
def test_array_finds_the_piano_keyboard_ladder_in_chain_order(piano):
    model, _ = prepared(piano)
    lad = motifs.ladder(model, set(block(model, "keyboard").members))
    assert lad["common"] == "OSC"
    assert lad["cells"] == [f"SW{i}" for i in range(1, 9)]
    assert lad["taps"] == [f"T{i}" for i in range(1, 9)]
    assert lad["links"] == [f"R{i}" for i in range(1, 8)]
    assert lad["end"] == ["R8"]


@needs_libs
def test_detectors_return_none_on_near_misses(piano, tmp_path):
    model, _ = prepared(piano)
    osc = set(block(model, "oscillator").members)
    assert motifs.cross_coupled(model, osc - {"C2"}) is None  # a missing coupling
    assert motifs.ladder(model, {"SW1", "SW2", "R1"}) is None  # only 2 cells
    ir = corpus_ir("astable", tmp_path, LIB)
    model, _ = prepared(ir)
    keys = set(block(model, "astable").members)
    assert motifs.cross_coupled(model, keys) is not None
    for ref, lib, name in (("C2", "Device", "C_Polarized"), ("Q2", "Transistor_BJT", "2N3906")):  # mixed library symbols
        changed = ir.model_copy(deep=True)
        c = next(c for c in changed.components if c.ref == ref)
        c.symbol = LIB.resolve_symbol(LibraryRef(library=lib, name=name))
        model, _ = prepared(changed)
        assert motifs.cross_coupled(model, keys) is None, ref


@needs_libs
def test_a_summing_bus_takes_its_branches_in_order_and_the_summer_at_its_end():
    model, _ = prepared(summing_board())
    bus = motifs.summing_bus(model, set(model.glyphs))
    assert bus.net == "SUM" and bus.branches == [["SW1", "R1"], ["SW2", "R2"], ["SW3", "R3"]]
    assert bus.summer.unit == "U1:1" and bus.summer.fb == ["R4"] and bus.summer.inputs == []
    assert motifs.summing_bus(model, set(model.glyphs) - {"R3"}) is None  # two branches are not a bus


@needs_libs
def test_op_amp_stage_analysis_on_the_fixtures(tmp_path):
    model, _ = prepared(amplifier(tmp_path, LIB))
    (s,) = motifs.stages_of(model, set(model.glyphs))
    assert (s.unit, s.F, s.P, s.O, s.fb, s.inputs, s.pmode) == ("U1", "SUM", "GND", "OUT", ["R2"], ["R1"], "ground")
    model, _ = prepared(amplifier_variant(tmp_path / "b", LIB, "bias_r"))
    (s,) = motifs.stages_of(model, set(model.glyphs))
    assert (s.pmode, s.chain) == ("bias", ["R3"])
    model, _ = prepared(amplifier_variant(tmp_path / "n", LIB, "noninv"))
    (s,) = motifs.stages_of(model, set(model.glyphs))
    assert (s.F, s.P, s.fb, s.shunts, s.pmode) == ("FB", "IN", ["R1"], ["R2"], "signal")
    model, _ = prepared(dual_amplifier(tmp_path / "d", LIB))
    a, b = motifs.stages_of(model, set(model.glyphs))
    assert (a.unit, a.fb, b.unit, b.fb, b.F == b.O) == ("U1:1", ["R2"], "U1:2", [], True)  # unit B: a follower


# --------------------------------------------------------------------------- structure


def _pin(model, sheet, key, number):
    pl = sheet.places()[key]
    return pin_cell(model.glyphs[key], number, (pl.x, pl.y), pl.orient)


def _pin_named(model, sheet, key, name):
    return _pin(model, sheet, key, next(p.number for p in model.glyphs[key].pins if p.name.strip() == name))


@needs_libs
def test_piano_blocks_take_their_motifs_and_the_oscillator_is_mirrored_with_the_x(piano):
    model, drawn, best = sheet_of(piano)
    assert {b: [d.recipe for d in ds] for b, ds in drawn.items()} == {
        "power": ["supply_chain"], "keyboard": ["array"], "oscillator": ["cross_coupled"], "driver": ["columns"]}
    sheet = best.sheet
    places = sheet.places()
    q1, q2 = places["Q1"], places["Q2"]
    assert q1.y == q2.y and q1.orient != q2.orient  # one height, mirror images
    assert places["C1"].y == places["C2"].y  # the couplings on one row
    axis = (q1.x + q2.x) / 2
    for left, right in (("R11", "R10"), ("C2", "C1")):
        assert places[left].x < axis < places[right].x and abs(places[left].x + places[right].x - 2 * axis) <= 1
    diag = [s for net in ("Q1B", "Q2B") for s in sheet.hints()[net] if s[0][0] != s[1][0] and s[0][1] != s[1][1]]
    assert len(diag) == 2 and all(abs(a[0] - b[0]) == abs(a[1] - b[1]) for a, b in diag)  # two 45-degree diagonals
    xs = [places[f"SW{i}"].x for i in range(1, 9)]
    assert xs == sorted(xs)  # SW1..SW8 left to right
    assert len({places[f"R{i}"].y for i in range(1, 8)}) == 1  # R1..R7 on one row


@needs_libs
def test_the_astable_is_the_textbook_pair_with_one_crossing(tmp_path):
    ir = corpus_ir("astable", tmp_path, LIB)
    result = build(SchematicCompiler(), ir, LIB)
    assert "astable=cross_coupled" in result.notes[0] and any("verified swap" in n for n in result.notes)
    path = tmp_path / "astable.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    g = schematic_gates(path, ir)
    assert g["crossings"] == 1 and g["paper"] == "A4" and g["wire_mm"] <= 688
    assert g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0


def _stage_gates(ir, tmp_path):
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and result.fallbacks == (), result.notes[0]
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    return result, path, schematic_gates(path, ir)


@needs_libs
@pytest.mark.parametrize("variant", ("base", *AMPLIFIER_VARIANTS))
def test_tl081_stages_have_feedback_above_or_below_by_the_rule_and_meet_their_gates(variant, tmp_path):
    ir = amplifier(tmp_path, LIB) if variant == "base" else amplifier_variant(tmp_path, LIB, variant)
    result, path, g = _stage_gates(ir, tmp_path)
    assert "opamp_stage" in result.notes[0]
    assert g["crossings"] == 0 and g["paper"] == "A4" and g["wire_mm"] < 300, g
    assert g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0
    model, _, best = sheet_of(ir)
    sheet = best.sheet
    places = sheet.places()
    assert places["U1"].orient == Orient()  # a unit with power pins stays at rotation 0
    fb = "R1" if variant == "noninv" else "R2"
    u1 = body_box(model.glyphs["U1"], places["U1"])
    r = body_box(model.glyphs[fb], places[fb])
    if variant in ("noninv", "bias_r"):  # the input wired straight into '+' (or biased on its row) above: feedback below
        assert r[1] > u1[3]
        if variant == "noninv":
            assert _pin(model, sheet, "J1", "1")[1] == _pin_named(model, sheet, "U1", "+")[1]
    else:
        assert r[3] < u1[1]  # the feedback above the unit
        minus = _pin_named(model, sheet, "U1", "-")
        assert _pin(model, sheet, "R1", next(p for p, n in model.glyphs["R1"].net.items() if n == "SUM"))[1] == minus[1]  # R1 on the '-' row
    centres = {k: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for k in ("R1", "R2", "U1") for b in [body_box(model.glyphs[k], places[k])]}
    assert max(math.dist(centres[a], centres[b]) for a in centres for b in centres) <= 30
    if variant in ("base", "fb_cap", "dual_supply"):  # '+' on a ground symbol hanging 2.54 mm or more clear of the '-' wire
        plus = _pin_named(model, sheet, "U1", "+")
        gnd = next(gen for gen in sheet.generated() if gen.kind == "GND" and generated_pin(gen)[1] == plus[1])
        bottom = mm(generated_pin(gnd)[1] + 2)  # the symbol hangs 2 cells below its pin
        assert mm(_pin_named(model, sheet, "U1", "-")[1]) - bottom >= 2.54 - 1e-9


@needs_libs
def test_the_dual_board_draws_both_units_with_the_inverting_input_on_top_and_its_power_unit_beside(tmp_path):
    ir = dual_amplifier(tmp_path, LIB)
    result, path, g = _stage_gates(ir, tmp_path)
    assert g["crossings"] == 0 and g["paper"] == "A4"
    model, _, best = sheet_of(ir)
    sheet = best.sheet
    places = sheet.places()
    assert {"U1:1", "U1:2", "U1:3"} <= set(places)  # every unit placed
    for unit in ("U1:1", "U1:2"):
        assert places[unit].orient == Orient(0, "x")  # mirrored: '-' on top
        assert _pin_named(model, sheet, unit, "-")[1] < _pin_named(model, sheet, unit, "+")[1]
    assert body_box(model.glyphs["R2"], places["R2"])[3] < body_box(model.glyphs["U1:1"], places["U1:1"])[1]  # feedback above
    # the power unit right of its last working stage, the spare U1B (a follower nothing listens to) in a strip past it
    assert places["U1:1"].x < places["U1:3"].x < places["U1:2"].x


@needs_libs
def test_a_summing_bus_is_drawn_as_columns_over_the_bus_into_the_summer(tmp_path):
    ir = summing_board()
    model, drawn, best = sheet_of(ir)
    assert [d.recipe for ds in drawn.values() for d in ds].count("array") == 1
    places = best.sheet.places()
    assert places["SW1"].x < places["SW2"].x < places["SW3"].x < places["U1:1"].x
    bus_y = _pin_named(model, best.sheet, "U1:1", "-")[1]
    assert all(_pin(model, best.sheet, f"R{i}", "2")[1] < bus_y for i in (1, 2, 3))
    assert any(a[1] == b[1] == bus_y for a, b in best.wiring.segments["SUM"])  # the bus line on the '-' row


@needs_libs
def test_a_supply_chain_is_one_line_with_the_bridges_in_it_and_shunts_hanging():
    parts = [("J1", "BAT", "Connector_Generic", "Conn_01x02"), ("SW1", "SW_SPDT", "Switch", "SW_SPDT"),
             ("C1", "10u", "Device", "C"), ("U1", "LP2950-50", "Regulator_Linear", "LP2950-5.0_TO92"),
             ("C2", "1u", "Device", "C"), ("C3", "2u2", "Device", "C"), ("R1", "1", "Device", "R"), ("R2", "22", "Device", "R"),
             ("C4", "100n", "Device", "C")]
    nets = [("VIN", "power", ("J1.1", "SW1.2", "SW1.3")), ("VBAT", "power", ("SW1.1", "C1.1", "U1.3")),
            ("V5", "power", ("U1.1", "C2.1", "C3.1", "R2.1")), ("ESR", "signal", ("C3.2", "R1.1")),
            ("VA", "power", ("R2.2", "C4.1")), ("GND", "ground", ("J1.2", "C1.2", "U1.2", "C2.2", "R1.2", "C4.2"))]
    ir = circuit("chain", parts, nets, (("power", [p[0] for p in parts], [], ["VA"]),))
    model, _ = prepared(ir)
    plan = motifs.chain_plan(model, set(model.glyphs))
    assert plan["rails"] == ["VIN", "VBAT", "V5", "VA"] and plan["bridges"] == ["SW1", "U1", "R2"] and plan["source"] == "J1"
    assert plan["shunts"] == {"VBAT": [["C1"]], "V5": [["C2"], ["C3", "R1"]], "VA": [["C4"]]}
    model, drawn, best = sheet_of(ir)
    (d,) = [d for ds in drawn.values() for d in ds]
    assert d.recipe == "supply_chain"
    places = best.sheet.places()
    assert places["J1"].x < places["SW1"].x < places["U1"].x < places["R2"].x  # left to right in derivation order
    assert _pin(model, best.sheet, "U1", "3")[1] == _pin(model, best.sheet, "U1", "1")[1] == _pin(model, best.sheet, "R2", "1")[1]
    for k in ("C1", "C2", "C3", "C4"):
        assert places[k].y > _pin(model, best.sheet, "U1", "1")[1]  # shunts hang below the line
    assert motifs.chain_plan(model, set(model.glyphs) - {"R1"}) is None  # a part the plan cannot place: the columns rows


@needs_libs
def test_a_series_ladder_runs_its_path_on_one_row_with_the_shunts_off_its_nodes(tmp_path):
    ir = corpus_ir("lc_lowpass", tmp_path, LIB)
    model, _ = prepared(ir)
    plan = motifs.series_ladder(model, block(model, "lowpass"), set(block(model, "lowpass").members))
    assert [k for k, _, _ in plan["path"]] == ["R1", "L1"] and plan["shunts"] == {"OUT": [["C1"], ["R2"]]}
    model, drawn, best = sheet_of(ir)
    assert [d.recipe for d in drawn["lowpass"]] == ["array"]
    y = {_pin(model, best.sheet, k, p)[1] for k in ("R1", "L1") for p in ("1", "2")}
    assert len(y) == 1  # the series path on one row
    row = _pin(model, best.sheet, "R1", "1")[1]
    assert all(min(_pin(model, best.sheet, k, p)[1] for p in ("1", "2")) > row for k in ("C1", "R2"))  # shunts hang below
    # a volume pot starting the path, its tail capacitor below it; a shunt up to a net that leaves the group
    parts = [("RV1", "10k", "Device", "R_Potentiometer"), ("C1", "1u", "Device", "C"), ("R1", "10k", "Device", "R"),
             ("R2", "10k", "Device", "R"), ("C2", "22n", "Device", "C"), ("C3", "10n", "Device", "C"), ("J1", "IO", "Connector_Generic", "Conn_01x04")]
    nets = [("MIX", "signal", ("J1.1", "RV1.3")), ("LOW", "signal", ("RV1.1", "C1.1")), ("WIP", "signal", ("RV1.2", "R1.1")),
            ("A", "signal", ("R1.2", "R2.1", "C2.1")), ("B", "signal", ("R2.2", "C3.1", "J1.2")), ("FB", "signal", ("C2.2", "J1.3")),
            ("GND", "ground", ("C1.2", "C3.2", "J1.4"))]
    ir = circuit("ladder", parts, nets, (("io", ["J1"], ["B", "FB"], ["MIX"]), ("filter", ["RV1", "C1", "R1", "R2", "C2", "C3"], ["MIX", "FB"], ["B"])))
    model, _ = prepared(ir)
    plan = motifs.series_ladder(model, block(model, "filter"), set(block(model, "filter").members))
    assert plan["pot"] == "RV1" and plan["tail"] == ["C1"] and [k for k, _, _ in plan["path"]] == ["RV1", "R1", "R2"]
    assert plan["shunts"] == {"A": [["C2"]], "B": [["C3"]]}
    model, drawn, best = sheet_of(ir)
    assert [d.recipe for d in drawn["filter"]] == ["array"] and not [h for h in best.hits if h.hard]
    places = best.sheet.places()
    assert places["C2"].y < places["R1"].y < places["C3"].y  # up to the net that leaves, down to ground


# --------------------------------------------------------------------------- fallbacks


@needs_libs
def test_a_failing_motif_hands_its_block_to_the_columns_recipe_with_a_note(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise LayoutFailure("forced")

    monkeypatch.setattr(motifs, "draw_stage", fail)
    result = build(SchematicCompiler(), amplifier(tmp_path, LIB), LIB)
    assert result.score is not None
    assert any("opamp_stage failed" in f for f in result.fallbacks) and "=columns" in result.notes[0]
    monkeypatch.undo()
    monkeypatch.setattr(motifs, "draw_cross", fail)
    result = build(SchematicCompiler(), corpus_ir("astable", tmp_path / "a", LIB), LIB)
    assert "astable=cross_coupled" in result.notes[0] and any("the columns mirror instead" in n for n in result.notes)


# --------------------------------------------------------------------------- kicad


@needs_kicad
@pytest.mark.parametrize("name", ("dual", "summing", *AMPLIFIER_VARIANTS))
def test_motif_boards_pass_erc_and_keep_the_netlist(name, tmp_path):
    ir = dual_amplifier(tmp_path, LIB) if name == "dual" else summing_board() if name == "summing" else amplifier_variant(tmp_path, LIB, name)
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and result.fallbacks == (), result.notes[0]
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"], erc.details
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    assert got == {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}


@needs_libs
def test_piano_gates(piano, tmp_path):
    start = time.perf_counter()
    result = build(SchematicCompiler(), piano, LIB)
    assert time.perf_counter() - start < 5
    path = tmp_path / "piano.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    g = schematic_gates(path, piano)
    assert result.fallbacks == () and g["crossings"] <= 2 and g["paper"] in ("A4", "A3")
    assert g["wire_mm"] < 1500 and g["wire_rmst_ratio"] <= 1.5 and g["median_connected_mm"] <= 40
    assert g["blocks_left_to_right"] == {"ok": 4, "of": 4}
    names = [n for n in result.notes[0].split("; ") if "global label" in n]
    assert names and int(names[0].split(" net(s)")[0]) <= 18
    assert g["label_joined_nets"] == 0  # real wires: no label pair


# --------------------------------------------------------------------------- odd inputs, wiring quality and notes


def _no_connects(ir: CircuitIR) -> CircuitIR:
    """Every pin no net joins marked no-connect (the op-amps' null and spare pins)."""
    from ai_eda.ir import PinElectricalType

    joined = {(p.component_ref, p.pin_number) for n in ir.nets for p in n.pins}
    for c in ir.components:
        for p in c.pins:
            if (c.ref, p.number) not in joined:
                p.electrical_type = PinElectricalType.NO_CONNECT
    return ir


TL = ("U1", "TL081", "Amplifier_Operational", "TL081")


def _tl081(inp: str, inm: str, out: str, vp: str = "VCC", vm: str = "GND") -> list[tuple[str, str]]:
    return [(inm, "U1.2"), (inp, "U1.3"), (out, "U1.6"), (vp, "U1.7"), (vm, "U1.4")]


def _nets(pairs, kinds=None) -> list:
    kinds = {"VCC": "power", "VEE": "power", "GND": "ground", **(kinds or {})}
    out: dict[str, list[str]] = {}
    for net, pin in pairs:
        out.setdefault(net, []).append(pin)
    return [(n, kinds.get(n, "signal"), tuple(p)) for n, p in out.items()]


def summer(third: str = "R") -> CircuitIR:
    """A TL081 inverting summer: R1, R2 and R3 (or a capacitor) from a header into '-', R4 the feedback, '+' on ground."""
    parts = [("R1", "10k", "Device", "R"), ("R2", "10k", "Device", "R"), ("R3", "10k", "Device", third), ("R4", "100k", "Device", "R"),
             TL, ("J1", "IO", "Connector_Generic", "Conn_01x06")]
    pairs = _tl081("GND", "SUM", "OUT") + [("A", "J1.1"), ("A", "R1.1"), ("B", "J1.2"), ("B", "R2.1"), ("C", "J1.3"), ("C", "R3.1"),
                                          ("SUM", "R1.2"), ("SUM", "R2.2"), ("SUM", "R3.2"), ("SUM", "R4.1"), ("OUT", "R4.2"), ("OUT", "J1.4"),
                                          ("VCC", "J1.5"), ("GND", "J1.6")]
    return _no_connects(circuit(f"summer_{third}", parts, _nets(pairs)))


def key_ladder(n: int = 8) -> CircuitIR:
    """n keys from one signal net COM to their own taps, the taps chained by resistors, a header on COM and both chain ends."""
    parts = [("J1", "IO", "Connector_Generic", "Conn_01x03")] + [(f"SW{i}", "key", "Switch", "SW_Push") for i in range(1, n + 1)]
    parts += [(f"R{i}", "1k", "Device", "R") for i in range(1, n)]
    nets = [("COM", "signal", tuple(["J1.1"] + [f"SW{i}.1" for i in range(1, n + 1)]))]
    for i in range(1, n + 1):
        pins = [f"SW{i}.2"] + ([f"R{i - 1}.2"] if i > 1 else []) + ([f"R{i}.1"] if i < n else []) + (["J1.2"] if i == 1 else []) + (["J1.3"] if i == n else [])
        nets.append((f"T{i}", "signal", tuple(pins)))
    return circuit(f"ladder{n}", parts, nets, (("keys", [p[0] for p in parts], [], ["COM"]),))


def volume(ends: str) -> CircuitIR:
    """A potentiometer starting a series ladder, used as a rheostat (one end open) or with both ends on one net."""
    pins = ("J1.1", "RV1.1", "RV1.3") if ends == "shorted" else ("J1.1", "RV1.1")
    ir = circuit(f"volume_{ends}", [("J1", "IO", "Connector_Generic", "Conn_01x03"), ("RV1", "10k", "Device", "R_Potentiometer"),
                                    ("R1", "10k", "Device", "R"), ("C1", "10n", "Device", "C")],
                 [("IN", "signal", pins), ("W", "signal", ("RV1.2", "R1.1")), ("OUT", "signal", ("R1.2", "C1.1", "J1.2")),
                  ("GND", "ground", ("C1.2", "J1.3"))],
                 (("io", ["J1"], ["OUT"], ["IN"]), ("vol", ["RV1", "R1", "C1"], ["IN"], ["OUT"])))
    return _no_connects(ir)


def divider_biased() -> CircuitIR:
    """A single-supply TL081 inverting stage whose '+' sits on a two-resistor divider between the rails."""
    parts = [("J1", "IO", "Connector_Generic", "Conn_01x04"), TL, ("R1", "10k", "Device", "R"), ("R2", "100k", "Device", "R"),
             ("R3", "10k", "Device", "R"), ("R4", "10k", "Device", "R")]
    pairs = _tl081("P", "M", "O") + [("IN", "R1.1"), ("M", "R1.2"), ("M", "R2.1"), ("O", "R2.2"), ("VCC", "R3.1"), ("P", "R3.2"),
                                     ("P", "R4.1"), ("GND", "R4.2"), ("IN", "J1.1"), ("O", "J1.2"), ("VCC", "J1.3"), ("GND", "J1.4")]
    return _no_connects(circuit("divider_biased", parts, _nets(pairs)))


def split_supply() -> CircuitIR:
    """A TL081 inverting stage on VCC / VEE, decoupled, from a header whose input pin sits right under its output pin."""
    parts = [("J1", "IO", "Connector_Generic", "Conn_01x05"), TL, ("R1", "10k", "Device", "R"), ("R2", "10k", "Device", "R"),
             ("C1", "100n", "Device", "C"), ("C2", "100n", "Device", "C")]
    pairs = _tl081("GND", "M", "O", "VCC", "VEE") + [("IN", "R1.1"), ("M", "R1.2"), ("M", "R2.1"), ("O", "R2.2"), ("VCC", "J1.1"),
                                                     ("GND", "J1.2"), ("VEE", "J1.3"), ("O", "J1.4"), ("IN", "J1.5"), ("VCC", "C1.1"),
                                                     ("GND", "C1.2"), ("GND", "C2.1"), ("VEE", "C2.2")]
    return _no_connects(circuit("split_supply", parts, _nets(pairs)))


def _recipes(result) -> dict[str, str]:
    head = result.notes[0].split("; blocks ", 1)[1].split("; fallbacks", 1)[0]
    return dict(item.split("=", 1) for item in head.split(", "))


def _sum_pin(model, sheet, key: str):
    return _pin(model, sheet, key, next(p for p, n in model.glyphs[key].net.items() if n == "SUM"))


@needs_libs
@pytest.mark.parametrize("ends", ("open", "shorted"))
def test_a_potentiometer_that_is_no_volume_control_is_not_a_series_ladder(ends):
    ir = volume(ends)
    model, _ = prepared(ir)
    assert motifs.series_ladder(model, block(model, "vol"), set(block(model, "vol").members)) is None
    result = build(SchematicCompiler(), ir, LIB)  # never raises: the columns recipe draws it
    assert result.fallbacks == () and _recipes(result)["vol"] == "columns"


@needs_libs
def test_an_inverting_summer_keeps_its_feedback_resistor_off_the_bus():
    ir = summer("R")
    model, _ = prepared(ir)
    bus = motifs.summing_bus(model, set(model.glyphs))
    assert bus.branches == [["R1"], ["R2"], ["R3"]] and bus.summer.fb == ["R4"]
    model, drawn, best = sheet_of(ir)
    assert [d.recipe for ds in drawn.values() for d in ds] == ["array"]  # the header is drawn with the bus, on its input side
    places = best.sheet.places()
    assert body_box(model.glyphs["R4"], places["R4"])[3] < body_box(model.glyphs["U1"], places["U1"])[1]  # feedback above
    assert places["J1"].x < places["R1"].x < places["R2"].x < places["R3"].x < places["U1"].x
    assert best.score[2] == 0  # no crossing


@needs_libs
def test_a_summer_of_unlike_inputs_is_an_op_amp_stage_with_its_inputs_stacked():
    ir = summer("C")
    result = build(SchematicCompiler(), ir, LIB)
    assert result.fallbacks == () and set(_recipes(result).values()) == {"opamp_stage"}
    model, _, best = sheet_of(ir)
    sheet = best.sheet
    minus = _pin_named(model, sheet, "U1", "-")
    ends = [_sum_pin(model, sheet, k) for k in ("R1", "R2", "R3")]
    rows = [p[1] for p in ends]
    assert rows[0] == minus[1] and rows == sorted(rows) and len(set(rows)) == 3  # the first on the '-' row, the others under it
    assert len({p[0] for p in ends}) == 1  # all meeting one riser


@needs_libs
def test_a_key_ladder_on_a_signal_net_is_a_ladder_not_a_summing_bus():
    ir = key_ladder(8)
    model, _ = prepared(ir)
    assert motifs.summing_bus(model, set(model.glyphs)) is None
    assert motifs.ladder(model, set(model.glyphs))["cells"] == [f"SW{i}" for i in range(1, 9)]
    result = build(SchematicCompiler(), ir, LIB)
    assert result.fallbacks == () and _recipes(result) == {"keys": "array"}


@needs_libs
def test_a_divider_biased_input_is_drawn_by_the_stage_on_its_own_side_supply_end_left():
    ir = divider_biased()
    model, _ = prepared(ir)
    (s,) = motifs.stages_of(model, set(model.glyphs))
    assert (s.pmode, s.chain, s.chain_nets) == ("chain", ["R3", "R4"], ["VCC", "P", "GND"])
    result = build(SchematicCompiler(), ir, LIB)
    assert result.fallbacks == () and set(_recipes(result).values()) == {"opamp_stage"}
    model, _, best = sheet_of(ir)
    sheet = best.sheet
    row = {_pin(model, sheet, k, p)[1] for k in ("R3", "R4") for p in ("1", "2")}
    # '+' is on top of a TL081: the divider on one row over the unit, the feedback under it, no wire crossing another
    assert len(row) == 1 and row.pop() < _pin_named(model, sheet, "U1", "+")[1]
    assert body_box(model.glyphs["R2"], sheet.places()["R2"])[1] > body_box(model.glyphs["U1"], sheet.places()["U1"])[3]
    assert best.score[2] == 0
    left = min(_pin(model, sheet, "R3", p)[0] for p in ("1", "2"))
    assert any(gen.kind == "VCC" and generated_pin(gen)[0] <= left for gen in sheet.generated())  # the supply at the left end


@needs_libs
def test_a_header_input_right_under_its_output_keeps_the_stage():
    result = build(SchematicCompiler(), split_supply(), LIB)
    assert result.fallbacks == () and "inferred.J1=opamp_stage" in result.notes[0]


def _wire_faults(segments: dict) -> list:
    """U-notches (a 1-cell piece whose both ends turn the same way into 1-cell pieces) and same-net parallel runs 1 cell
    apart over more than 1 cell, net by net."""
    out = []
    for net, segs in segments.items():
        short = [s for s in segs if abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]) == 1]
        for s in short:
            a, b = s
            turns = []
            for p in (a, b):
                for o in short:
                    q = o[1] if o[0] == p and o is not s else o[0] if o[1] == p and o is not s else None
                    if q is not None and (q[0] - p[0]) * (b[0] - a[0]) + (q[1] - p[1]) * (b[1] - a[1]) == 0:
                        turns.append((q[0] - p[0], q[1] - p[1]))
            if len(turns) == 2 and turns[0] == turns[1]:
                out.append((net, "notch", s))
        for i, s in enumerate(segs):
            for o in segs[i + 1:]:
                for ax in (0, 1):
                    if s[0][1 - ax] == s[1][1 - ax] and o[0][1 - ax] == o[1][1 - ax] and abs(s[0][1 - ax] - o[0][1 - ax]) == 1:
                        lo = max(min(s[0][ax], s[1][ax]), min(o[0][ax], o[1][ax]))
                        hi = min(max(s[0][ax], s[1][ax]), max(o[0][ax], o[1][ax]))
                        if hi - lo > 1:
                            out.append((net, "parallel", s, o))
    return out


@needs_libs
def test_blocks_meet_at_the_trunk_row_with_no_notch_or_doubled_run(piano, tmp_path):
    from tests.textbook_cases import oscillator_bank

    for ir in (piano, oscillator_bank(tmp_path, LIB)):
        _, _, best = sheet_of(ir)
        assert _wire_faults(best.wiring.segments) == [], ir.project.id


@needs_libs
def test_an_oscillator_bank_draws_every_voice_as_the_same_cell(tmp_path):
    from tests.textbook_cases import oscillator_bank

    ir = oscillator_bank(tmp_path, LIB)
    result = build(SchematicCompiler(), ir, LIB)
    assert result.fallbacks == () and _recipes(result)["voices"] == "opamp_stage" and _recipes(result)["driver"] == "opamp_stage"
    model, drawn, best = sheet_of(ir)
    sheet = best.sheet
    places = sheet.places()
    units = [f"U1:{i}" for i in (1, 2, 3)]
    assert len({places[u].y for u in units}) == 1 and [places[u].x for u in units] == sorted(places[u].x for u in units)
    cell = None
    for i, u in enumerate(units, 1):  # one cell, repeated: every part at the same offset from its unit
        offsets = tuple((places[f"{k}{i}"].x - places[u].x, places[f"{k}{i}"].y - places[u].y, places[f"{k}{i}"].orient)
                        for k in ("C1", "R1", "RV", "R2", "R3"))
        assert cell in (None, offsets), u
        cell = offsets
        assert places[f"RV{i}"].y > places[u].y > places[f"R1{i}"].y  # the chain under the unit, the feedback above
    ends = {_pin(model, sheet, f"R3{i}", next(p for p, n in model.glyphs[f"R3{i}"].net.items() if n == "VREF")) for i in (1, 2, 3)}
    rows = {p[1] for p in ends}
    assert len(rows) == 1
    y = rows.pop()
    assert all(any(a[1] == b[1] == y and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) for a, b in best.wiring.segments["VREF"]) for p in ends)
    (voices,) = drawn["voices"]
    assert {lab.net for lab in voices.labels if lab.kind == "pair"} >= {"OSC1", "OSC2", "OSC3"}  # the bank's outputs named
    assert places["U1:3"].x < places["U1:5"].x < places["U1:4"].x  # the power unit, then the spare
    assert places["Q1"].y < places["U2:1"].y < places["Q2"].y  # the followers: the upper on the supply, the lower on ground
    assert places["U2:1"].x < places["R8"].x < places["C20"].x < places["J2"].x  # the load in line to its terminal
    assert any("followers Q1/Q2 inside the loop, bias R7 between the nodes, feedback wire over the top" in n for n in result.notes)
    assert any("spare unit(s) U1D in a strip at the end" in n for n in result.notes)


@needs_libs
@pytest.mark.parametrize("variant, fb", (("bias_r", "R2"), ("noninv", "R1")))
def test_a_stage_whose_rule_puts_the_feedback_below_is_scored_against_feedback_above(variant, fb, tmp_path):
    result = build(SchematicCompiler(), amplifier_variant(tmp_path, LIB, variant), LIB)
    stage = [n for n in result.notes if "opamp_stage: U1 " in n]
    assert stage and stage[0].endswith(f"+ on top, feedback {fb} below"), stage
    assert any("feedback below kept over the other variant" in n for n in result.notes)
    base = build(SchematicCompiler(), amplifier(tmp_path / "base", LIB), LIB)
    assert any(n.endswith("+ on top, feedback R2 above") for n in base.notes)
    assert not any("kept over the other variant" in n for n in base.notes)  # nothing to choose: one drawing


@needs_libs
def test_a_wire_through_a_text_it_does_not_own_fails_the_drawing(tmp_path):
    from ai_eda.compilers.textbook.glyphs import drawing_boxes

    model, ctx = prepared(amplifier(tmp_path, LIB))
    pen = motifs.Pen(ctx, model.blocks[0], "test")
    pen.put_two("R1", "IN", (0, 0), False, "T")
    field = next(b for tag, b in pen.boxes() if tag == "field:R1")
    y = round((field[1] + field[3]) / 2 / mm(1))
    pen.wire("OUT", (-20, y), (20, y))
    assert "runs through field:R1" in motifs._wire_through_text(pen._raw(), drawing_boxes(model, pen._raw()))
    with pytest.raises(LayoutFailure):
        pen.drawing()


@needs_libs
def test_a_source_connector_ties_its_pins_of_one_rail_to_one_symbol():
    parts = [("J1", "DC", "Connector_Generic", "Conn_01x03"), ("D1", "1N4007", "Device", "D"), ("C1", "10u", "Device", "C")]
    nets = [("VIN", "power", ("J1.1", "D1.2")), ("V5", "power", ("D1.1", "C1.1")), ("GND", "ground", ("J1.2", "J1.3", "C1.2"))]
    model, drawn, best = sheet_of(circuit("source", parts, nets, (("power", ["J1", "D1", "C1"], [], []),)))
    (d,) = drawn["power"]
    assert d.recipe == "supply_chain"
    assert sum(1 for gen in d.glyphs if gen.net == "GND") == 2  # one for J1's two ground pins, one under C1


@needs_libs
def test_a_supply_block_that_is_not_one_line_says_so(piano):
    result = build(SchematicCompiler(), piano, LIB)
    assert any(n.startswith("block power: supply_chain: not one line") for n in result.notes)


@needs_libs
def test_the_regulator_stands_in_the_line_with_its_ground_symbol_below_it():
    parts = [("J1", "BAT", "Connector_Generic", "Conn_01x02"), ("U1", "LP2950-50", "Regulator_Linear", "LP2950-5.0_TO92"),
             ("C1", "1u", "Device", "C")]
    nets = [("VIN", "power", ("J1.1", "U1.3")), ("V5", "power", ("U1.1", "C1.1")), ("GND", "ground", ("J1.2", "U1.2", "C1.2"))]
    model, drawn, best = sheet_of(circuit("regulator", parts, nets, (("power", ["J1", "U1", "C1"], [], []),)))
    gnd = _pin(model, best.sheet, "U1", "2")
    assert any(gen.kind == "GND" and generated_pin(gen)[0] == gnd[0] and generated_pin(gen)[1] > gnd[1] for gen in best.sheet.generated())


@needs_libs
def test_the_astable_halves_mirror_at_one_pitch_of_ten_cells_or_more(tmp_path):
    ir = corpus_ir("astable", tmp_path, LIB)
    model, drawn, best = sheet_of(ir)
    (d,) = drawn["astable"]
    found = motifs.cross_coupled(model, set(d.place))
    places = best.sheet.places()
    a, b = found["q"]
    columns = sorted(places[k].x for k in [*found["loads"][0], *found["loads"][1], *found["biases"][0], *found["biases"][1]])
    steps = {q - p for p, q in zip(columns, columns[1:])}
    assert len(columns) == 4 and len(steps) == 1 and steps.pop() >= 10  # four columns at one pitch p >= 10 cells
    assert places[a].y == places[b].y and places[a].orient != places[b].orient
    axis = (places[a].x + places[b].x) / 2
    for x, y in (found["coupling"], *zip(*found["loads"]), *zip(*found["biases"])):
        assert abs(places[x].x + places[y].x - 2 * axis) <= 1 and places[x].y == places[y].y


@needs_libs
def test_unequal_halves_are_mirrored_by_symbol_but_never_called_a_verified_swap(tmp_path):
    ir = corpus_ir("astable", tmp_path, LIB)
    model, _ = prepared(ir)
    found = motifs.cross_coupled(model, set(block(model, "astable").members))
    next(c for c in ir.components if c.ref == found["coupling"][1]).value = "22u"
    result = build(SchematicCompiler(), ir, LIB)
    assert any("mirrored by symbol (values differ)" in n for n in result.notes)
    assert not any("verified swap" in n for n in result.notes)


@needs_libs
def test_every_array_and_supply_rung_falls_to_the_columns_recipe_with_a_note(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise LayoutFailure("forced")

    monkeypatch.setattr(motifs, "draw_ladder", fail)
    result = build(SchematicCompiler(), corpus_ir("lc_lowpass", tmp_path, LIB), LIB)
    assert any("array failed (forced)" in f for f in result.fallbacks) and _recipes(result)["lowpass"] == "columns"
    monkeypatch.undo()
    monkeypatch.setattr(motifs, "draw_bus", fail)
    result = build(SchematicCompiler(), summing_board(), LIB)
    assert any("array failed (forced)" in f for f in result.fallbacks) and "array" not in _recipes(result)["mixer"]
    monkeypatch.undo()
    monkeypatch.setattr(motifs, "draw_chain_line", fail)
    parts = [("J1", "BAT", "Connector_Generic", "Conn_01x02"), ("D1", "1N4007", "Device", "D"), ("C1", "10u", "Device", "C")]
    nets = [("VIN", "power", ("J1.1", "D1.2")), ("V5", "power", ("D1.1", "C1.1")), ("GND", "ground", ("J1.2", "C1.2"))]
    result = build(SchematicCompiler(), circuit("line", parts, nets, (("power", ["J1", "D1", "C1"], [], []),)), LIB)
    assert any("supply_chain line not drawn (forced); columns rail rows instead" in n for n in result.notes)
    assert result.fallbacks == () and _recipes(result)["power"] == "supply_chain"


@needs_kicad
@pytest.mark.parametrize("name", ("bank", "summer_R", "summer_C", "divider_biased", "split_supply"))
def test_more_motif_boards_pass_erc_and_keep_the_netlist(name, tmp_path):
    from tests.textbook_cases import oscillator_bank

    ir = {"bank": lambda: oscillator_bank(tmp_path, LIB), "summer_R": lambda: summer("R"), "summer_C": lambda: summer("C"),
          "divider_biased": divider_biased, "split_supply": split_supply}[name]()
    result = build(SchematicCompiler(), ir, LIB)
    assert result.score is not None and result.fallbacks == (), result.notes[0]
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"], erc.details
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    assert got == {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}
    g = schematic_gates(path, ir)
    assert g["crossings"] == 0 and g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0
