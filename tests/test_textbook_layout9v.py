"""Textbook 1.0 arrangement and routing on a board the size of the 9 V op-amp piano: bands cut at the width of the paper
the drawing needs, repeated blocks column-aligned like an array, a supply line through a series pass with its gate drive
under it, a potentiometer with a mounting pin as a volume control, a summer's power unit out of its output's way, field
text kept clear of routed wires, combinations of two resistors drawn as one element, and an inverting stage drawn with the
followers inside its loop."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.textbook import arrange, blocks, build, motifs, pipeline, rails, route
from ai_eda.compilers.textbook.glyphs import body_box, glyph_boxes, pin_cell, to_cells
from ai_eda.compilers.textbook.model import Kind, load_model, mounting
from ai_eda.compilers.textbook.recipes import Ctx, recipe_order
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.layout_metrics import schematic_gates
from tests.test_schematic_compiler import netlist_nets
from tests.textbook_cases import op_amp_piano

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")


def sheet_of(ir):
    """The model, the block drawings and the kept sheet variant (routed and scored) of ``ir``, as the pipeline builds them."""
    model = load_model(SchematicCompiler(), ir, LIB)
    model = replace(model, blocks=blocks.partition(model), rails=rails.rail_order(model))
    ctx = Ctx(SchematicCompiler(), LIB, model, model.notes)
    drawn = {b.id: pipeline.draw_block(ctx, b, recipe_order(), []) for b in model.blocks}
    return model, ctx, pipeline.best_sheet(ctx, model.blocks, drawn)


@pytest.fixture(scope="module")
def piano(tmp_path_factory):
    ir = op_amp_piano(tmp_path_factory.mktemp("op_amp_piano"), LIB)
    model, ctx, best = sheet_of(ir)
    return ir, model, ctx, best


def pin(model, places, key: str, number: str):
    pl = places[key]
    return pin_cell(model.glyphs[key], number, (pl.x, pl.y), pl.orient)


@needs_libs
def test_the_op_amp_piano_draws_every_block_by_its_recipe_with_no_fallback(piano, tmp_path):
    ir = op_amp_piano(tmp_path, LIB)
    t = time.time()
    r = build(SchematicCompiler(), ir, LIB)
    assert time.time() - t < 10
    assert r.fallbacks == ()
    head = r.notes[0]
    for want in ("power=supply_chain", "reference=columns", "voices_low=opamp_stage", "voices_high=opamp_stage", "keyboard=array",
                 "mixer_tone=array", "driver=opamp_stage", "paper A2"):
        assert want in head, head
    assert r.score[:3] == (0, 0, 0)  # no guard finding, no spacing finding, no crossing


@needs_libs
def test_bands_are_cut_at_the_width_of_the_paper_the_drawing_needs(piano):
    """A3 bands would be taller than an A3 sheet, so the bands are cut at the A2 width: four bands, and the keyboard, the
    mixer and the driver share the last one, the summer's output running on to the volume control (never back)."""
    ir, model, ctx, best = piano
    sheet = best.sheet
    assert any("wrapped into 4 bands" in n and "517.8 mm" in n for n in best.notes)
    band_of = {sheet.drawings[i].block: b for b, band in enumerate(sheet.bands) for i in band}
    assert band_of["keyboard"] == band_of["mixer_tone"] == band_of["driver"] == 3
    assert band_of["power"] == 0 and band_of["voices_low"] == 1 and band_of["voices_high"] == 2
    places = sheet.places()
    out = pin(model, places, "U3:1", "1")
    vol = pin(model, places, "RV9", "3")
    assert out[0] < vol[0] and abs(out[1] - vol[1]) <= 12
    mix = best.wiring.segments["MIX"]
    feedback = min(pin(model, places, "R4", n)[0] for n in ("1", "2"))
    assert max(max(a[0], b[0]) for a, b in mix) <= vol[0] and min(min(a[0], b[0]) for a, b in mix) >= feedback


@needs_libs
def test_a_repeated_block_stands_in_columns_under_the_one_it_repeats(piano):
    """voices_high repeats voices_low (same recipe, same symbols): every unit, feedback resistor and chain part stands at
    the x of its counterpart, like an array, though the reference block sits left of voices_low only."""
    ir, model, ctx, best = piano
    places = best.sheet.places()
    assert any("voices_high: column-aligned under block voices_low" in n for n in best.notes)
    for k in range(1, 5):
        assert places[f"U1:{k}"].x == places[f"U2:{k}"].x
        assert places[f"R1{k}"].x == places[f"R1{k + 4}"].x
        assert places[f"RV{k}"].x == places[f"RV{k + 4}"].x
    assert places["U1:5"].x == places["U2:5"].x
    assert places["U2:1"].y > places["U1:1"].y


@needs_libs
def test_the_cut_starts_a_band_with_a_repeat_and_alignment_never_passes_the_usable_width(piano):
    """With a lead for a repeat (the cells it moves right), the cut puts that repeat at the start of a band and counts the
    lead in the band's width; a band the move would make wider than the usable width is not moved."""
    ir, model, ctx, best = piano
    sheet = best.sheet
    drawings = list(sheet.drawings)
    spine = [i for band in sheet.bands for i in band]
    high = next(i for i in spine if drawings[i].block == "voices_high")
    keys = arrange.cut(model, drawings, spine, 407, {high: 40})
    assert any(band[0] == high for band in keys)
    assert all(arrange._width(drawings, band) + (40 if band[0] == high else 0) <= 407 for band in keys if len(band) > 1)
    offsets = {i: sheet.offset[i] for i in range(len(drawings))}
    shifted = {i: (dx - 30, dy) if i == high else (dx, dy) for i, (dx, dy) in offsets.items()}
    before = dict(shifted)
    arrange._align_repeats(model, drawings, [list(b) for b in sheet.bands], shifted, [], usable=1)
    assert shifted == before


@needs_libs
def test_the_supply_line_runs_through_a_series_pass_with_its_gate_drive_under_it(piano):
    """J1 -> Q3 -> VS -> Q4 -> VBAT -> U4 -> VDD -> R1 -> VDDO on one line; the MOSFETs' gates face down onto one gate
    line; R50 and C23 hang from VS between the MOSFETs, then R52 -> SW9 -> R51 down to the gate line, C22 from VBAT to it."""
    ir, model, ctx, best = piano
    places = best.sheet.places()
    plan = motifs.chain_plan(model, {k for b in model.blocks if b.id == "power" for k in b.members})
    assert plan["rails"] == ["BATP", "VBAT", "VDD", "VDDO"] and plan["bridges"] == [("Q3", "Q4"), "U4", "R1"]
    assert plan["control"] == ["C22", "C23", "R50", "R51", "R52", "SW9"]
    line = pin(model, places, "J1", "1")[1]
    for key, number in (("Q3", "2"), ("Q3", "3"), ("Q4", "3"), ("Q4", "2"), ("U4", "3"), ("U4", "1")):
        assert pin(model, places, key, number)[1] == line, (key, number)
    g3, g4 = pin(model, places, "Q3", "1"), pin(model, places, "Q4", "1")
    assert g3[1] > line and g4[1] > line
    gate = [s for s in best.wiring.segments["VG"] if s[0][1] == s[1][1]]
    gy = max(s[0][1] for s in gate)
    assert min(min(a[0], b[0]) for a, b in gate if a[1] == gy) <= g3[0] and max(max(a[0], b[0]) for a, b in gate if a[1] == gy) >= g4[0]
    for key in ("R50", "C23", "R52", "SW9", "R51"):
        assert g3[0] < places[key].x < g4[0], key
        assert line < places[key].y < gy, key
    assert places["C22"].x > g4[0] and places["C22"].x < places["U4"].x
    assert pin(model, places, "Q3", "3")[0] < places["R50"].x < pin(model, places, "Q4", "3")[0]
    assert any("through Q3+Q4" in n for n in model.notes)


@needs_libs
def test_a_series_pass_needs_its_control_network_inside_the_block():
    """series_bridges finds two three-terminal parts from one rail over a signal net to another; a control network that
    reaches a net outside the block keeps the plan from being one line."""
    ir = op_amp_piano(None, LIB)
    model = load_model(SchematicCompiler(), ir, LIB)
    power = set(next(b for b in ir.topology.blocks if b.id == "power").component_refs)
    found = motifs.series_bridges(model, power)
    assert [(a, b, ks, ns) for a, b, ks, ns in found] == [("BATP", "VBAT", ("Q3", "Q4"), ("VS",))]
    assert motifs.chain_plan(model, power - {"SW9"}) is None or "SW9" not in motifs.chain_plan(model, power - {"SW9"})["control"]
    assert motifs.series_bridges(model, power - {"Q4"}) == []


@needs_libs
def test_a_potentiometer_with_a_mounting_pin_is_a_three_terminal_volume_control(piano):
    ir, model, ctx, best = piano
    g = model.glyphs["RV9"]
    assert g.kind is Kind.THRU3
    assert [p.number for p in g.through()] == ["3", "1"] or [p.number for p in g.through()] == ["1", "3"]
    assert g.control().number == "2"
    assert [p.number for p in g.pins if mounting(p)] == ["MP"]
    ladder = next(d for d in best.sheet.drawings if d.block == "mixer_tone")
    assert ladder.recipe == "array"
    places = best.sheet.places()
    mp = pin(model, places, "RV9", "MP")
    gnd = [gen for gen in best.sheet.generated() if gen.net == "GND"]
    assert any(abs(gen.at[0] - mp[0]) <= 2 and 0 <= gen.at[1] - mp[1] <= 4 for gen in gnd)
    w = pin(model, places, "RV9", "2")
    assert w[1] == pin(model, places, "R5", "1")[1]  # the wiper on the filter's row
    assert places["R6"].x > places["R5"].x


@needs_libs
def test_the_summers_power_unit_stands_under_it_when_its_output_runs_on(piano):
    ir, model, ctx, best = piano
    places = best.sheet.places()
    a = to_cells(body_box(model.glyphs["U3:1"], places["U3:1"]))
    c = to_cells(body_box(model.glyphs["U3:3"], places["U3:3"]))
    assert a[0] <= (c[0] + c[2]) // 2 <= a[2]
    assert c[1] > a[3]


def near_text(model, best) -> list:
    """Grid points of routed wires (pins and recipe wires aside) within TEXT_CLEAR mm of a field text."""
    sheet = best.sheet
    hinted = {p for segs in sheet.hints().values() for s in segs for p in route.raster(s)}
    pins = {pin(model, sheet.places(), k, p.number) for k in sheet.places() for p in model.glyphs[k].pins}
    texts = [route._grown(box, route.TEXT_CLEAR) for tag, box in glyph_boxes(model, sheet) if tag.startswith(("field:", "gfield:"))]
    near = []
    for net, segs in best.wiring.segments.items():
        for s in segs:
            for x, y in route.raster(s):
                if (x, y) in pins or (x, y) in hinted:
                    continue
                if any(bx0 < x * 1.27 < bx1 and by0 < y * 1.27 < by1 for bx0, by0, bx1, by1 in texts):
                    near.append((net, (x, y)))
    return near


@needs_libs
def test_routed_wires_keep_clear_of_field_text(piano, tmp_path):
    """No grid point of a routed wire (pins aside) lies within TEXT_CLEAR mm of a field text (a part's Reference / Value, a
    power symbol's name): no wire skims along a text or turns at it. Recipe wires are checked by the recipes. On the
    astable the shortest way for the output header's wire passes the supply symbol's name, so the rule moves it."""
    from tests.textbook_corpus import corpus_ir

    ir, model, ctx, best = piano
    assert not near_text(model, best)
    model, ctx, best = sheet_of(corpus_ir("astable", tmp_path, LIB))
    assert not near_text(model, best)


@needs_libs
def test_the_op_amp_piano_is_deterministic(tmp_path):
    a = sexpr.dumps(build(SchematicCompiler(), op_amp_piano(tmp_path, LIB), LIB).node)
    ir = op_amp_piano(tmp_path, LIB)
    ir.nets.reverse()
    ir.components.reverse()
    b = sexpr.dumps(build(SchematicCompiler(), ir, LIB).node)
    assert a == b


@needs_kicad
def test_the_op_amp_piano_passes_erc_and_its_netlist_equals_the_ir(tmp_path):
    ir = op_amp_piano(tmp_path, LIB)
    r = build(SchematicCompiler(), ir, LIB)
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(r.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"], (erc.details["errors"] + erc.details["warnings"])[:3]
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    want = {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}
    assert got == want
    g = schematic_gates(path, ir)
    assert g["crossings"] == 0 and g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0
    assert g["supply_above_ground"]["ok"] == g["supply_above_ground"]["of"]
    assert g["paper"] == "A2"


def combined(ir):
    """The piano with two combinations: R11 as 56k + 3k3 in series (FB1 between them), R21 as 420k || 420k."""
    from ai_eda.ir import Net, PinRef

    comps = {c.ref: c for c in ir.components}
    comps["R11"].value = "56k"
    comps["R21"].value = "420k"
    ir.components.append(comps["R11"].model_copy(update={"ref": "R19", "value": "3k3"}))
    ir.components.append(comps["R21"].model_copy(update={"ref": "R29", "value": "420k"}))
    for net in ir.nets:
        for p in list(net.pins):
            if p.component_ref == "R11" and p.pin_number == "1":  # R11.1 was on OSC1: R19 takes it, R11.1 goes to FB1
                net.pins.remove(p)
                net.pins.append(PinRef(component_ref="R19", pin_number="2"))
            if p.component_ref == "R21":
                net.pins.append(PinRef(component_ref="R29", pin_number=p.pin_number))
    ir.nets.append(Net(name="FB1", kind="signal", provenance=ir.nets[0].provenance,
                       pins=[PinRef(component_ref="R11", pin_number="1"), PinRef(component_ref="R19", pin_number="1")]))
    block = next(b for b in ir.topology.blocks if b.id == "voices_low")
    block.component_refs += ["R19", "R29"]
    return ir


@needs_libs
def test_a_combination_is_drawn_as_one_element_in_series_in_line_in_parallel_side_by_side(tmp_path):
    from ai_eda.compilers.textbook import compound

    ir = combined(op_amp_piano(tmp_path, LIB))
    model = load_model(SchematicCompiler(), ir, LIB)
    assert [(k, a, b, m) for k, a, b, m in compound.pairs(model)] == [("series", "R11", "R19", "FB1"), ("parallel", "R21", "R29", None)]
    contracted, compounds = compound.contract(model)
    assert "R19" not in contracted.glyphs and "R29" not in contracted.glyphs and "FB1" not in contracted.nets
    assert contracted.glyphs["R11"].kind is Kind.TWO and set(contracted.glyphs["R11"].net.values()) == {"INM1", "OSC1"}
    r = build(SchematicCompiler(), ir, LIB)
    assert r.fallbacks == () and "voices_low=opamp_stage" in r.notes[0], r.notes[0]
    assert any("combinations drawn as one element: R11+R19, R21||R29" in n for n in r.notes)
    at = {}
    for item in r.node:
        if isinstance(item, list) and item and item[0] == "symbol":
            ref = next(str(p[2]) for p in item if isinstance(p, list) and p and p[0] == "property" and str(p[1]) == "Reference")
            pos = next(p for p in item if isinstance(p, list) and p and p[0] == "at")
            at[ref] = (float(pos[1]), float(pos[2]), float(pos[3]))
    # in series: in line on the feedback row, pin to pin (7.62 mm apart); in parallel: side by side, one above the other
    assert at["R11"][1] == at["R19"][1] and abs(at["R11"][0] - at["R19"][0]) == pytest.approx(7.62)
    assert at["R21"][0] == at["R29"][0] and at["R29"][1] > at["R21"][1]
    labels = [item for item in r.node if isinstance(item, list) and item and item[0] == "global_label" and str(item[1]) == "FB1"]
    assert len(labels) == 1


def symbols_at(node) -> dict[tuple[str, int], tuple[float, float]]:
    """(Reference, unit) -> the placed symbol's origin (mm) in an emitted sheet."""
    at = {}
    for item in node:
        if isinstance(item, list) and item and item[0] == "symbol":
            ref = next(str(p[2]) for p in item if isinstance(p, list) and p and p[0] == "property" and str(p[1]) == "Reference")
            unit = next((int(p[1]) for p in item if isinstance(p, list) and p and p[0] == "unit"), 1)
            pos = next(p for p in item if isinstance(p, list) and p and p[0] == "at")
            at[(ref, unit)] = (float(pos[1]), float(pos[2]))
    return at


@needs_libs
def test_a_combination_keeps_the_repeated_block_in_columns_and_the_summer_output_running_on(tmp_path):
    """A combination in the first voice bank is still a repeat of the second (its outline counts as one resistor): the banks
    stand in columns, and the keyboard, the summer and the volume control share the last band, MIX running on."""
    r = build(SchematicCompiler(), combined(op_amp_piano(tmp_path, LIB)), LIB)
    assert any("voices_high: column-aligned under block voices_low" in n for n in r.notes)
    at = symbols_at(r.node)
    for unit in range(1, 6):
        assert at[("U1", unit)][0] == at[("U2", unit)][0], unit
    assert at[("U3", 1)][0] < at[("RV9", 1)][0] and abs(at[("U3", 1)][1] - at[("RV9", 1)][1]) < 25
    assert at[("SW1", 1)][1] > at[("U2", 1)][1] and abs(at[("SW1", 1)][1] - at[("U3", 1)][1]) < 40


@needs_libs
def test_a_series_pair_has_its_fields_over_its_own_bodies_and_the_middle_name_between_them(tmp_path):
    """R11 + R19 lie on the feedback row with their fields above: each part's Reference and Value stand over its own body
    (centred on it, as a lone resistor's do), and the name of the net between them stands at the point where they meet,
    between the two parts' texts."""
    r = build(SchematicCompiler(), combined(op_amp_piano(tmp_path, LIB)), LIB)
    fields, at = {}, symbols_at(r.node)
    for item in r.node:
        if isinstance(item, list) and item and item[0] == "symbol":
            props = {str(p[1]): p for p in item if isinstance(p, list) and p and p[0] == "property"}
            ref = str(props["Reference"][2])
            fields[ref] = [tuple(float(v) for v in next(q for q in props[k] if isinstance(q, list) and q[0] == "at")[1:3])
                           for k in ("Reference", "Value")]
    row = at[("R11", 1)][1]
    for ref in ("R11", "R19", "R12"):
        assert all(f[0] == at[(ref, 1)][0] for f in fields[ref]), (ref, fields[ref], at[(ref, 1)])
        assert all(f[1] < row for f in fields[ref]), (ref, fields[ref], row)
    assert fields["R11"][0][1] == fields["R19"][0][1]
    (label,) = [item for item in r.node if isinstance(item, list) and item and item[0] == "global_label" and str(item[1]) == "FB1"]
    lx, ly = (float(v) for v in next(q for q in label if isinstance(q, list) and q[0] == "at")[1:3])
    assert lx == (at[("R11", 1)][0] + at[("R19", 1)][0]) / 2 and fields["R11"][1][1] < ly <= row


@needs_libs
def test_a_combination_is_not_found_on_near_misses(tmp_path):
    """No pair when the two resistors sit in different blocks, are different library symbols, or the net between them
    reaches a third part; the parallel pair is found alone each time."""
    from ai_eda.compilers.textbook import compound
    from ai_eda.ir import LibraryRef, PinRef

    def found(ir):
        return [(k, a, b) for k, a, b, _ in compound.pairs(load_model(SchematicCompiler(), ir, LIB))]

    ir = combined(op_amp_piano(tmp_path, LIB))
    low, high = (next(b for b in ir.topology.blocks if b.id == i) for i in ("voices_low", "voices_high"))
    low.component_refs.remove("R19")
    high.component_refs.append("R19")
    assert found(ir) == [("parallel", "R21", "R29")]
    ir = combined(op_amp_piano(tmp_path, LIB))
    r19 = next(c for c in ir.components if c.ref == "R19")
    r19.symbol = LIB.resolve_symbol(LibraryRef(library="Device", name="R_Small"))
    assert found(ir) == [("parallel", "R21", "R29")]
    ir = combined(op_amp_piano(tmp_path, LIB))
    fb1 = next(n for n in ir.nets if n.name == "FB1")
    gnd = next(n for n in ir.nets if n.name == "GND")
    c11 = next(p for p in gnd.pins if p.component_ref == "C11")
    gnd.pins.remove(c11)
    fb1.pins.append(c11)
    assert found(ir) == [("parallel", "R21", "R29")]


@needs_libs
def test_a_combination_the_guard_refuses_is_drawn_as_separate_parts(tmp_path, monkeypatch):
    """A guard finding on the sheet with the combinations put back: the sheet is drawn again with every part on its own,
    noted as a fallback, not the whole-sheet table."""
    from ai_eda.compilers.textbook.model import Hit

    real = pipeline.guard

    def forced(ctx, *args, **kwargs):
        if "R19" in ctx.model.glyphs and ctx.compounds:
            return [Hit("forced: R19 over a wire", ("R19",))]
        return real(ctx, *args, **kwargs)

    monkeypatch.setattr(pipeline, "guard", forced)
    r = build(SchematicCompiler(), combined(op_amp_piano(tmp_path, LIB)), LIB)
    assert r.fallbacks == ("combinations drawn as separate parts (combination drawn as two parts: forced: R19 over a wire)",)
    assert "whole-sheet table" not in r.notes[0] and "voices_low=opamp_stage" in r.notes[0], r.notes[0]
    assert not any("combinations drawn as one element" in n for n in r.notes)
    assert ("R19", 1) in symbols_at(r.node)


@needs_kicad
def test_a_combination_passes_erc_and_keeps_every_net_name(tmp_path):
    ir = combined(op_amp_piano(tmp_path, LIB))
    r = build(SchematicCompiler(), ir, LIB)
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(r.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"]
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    want = {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}
    assert got == want
    assert {n.strip("/") for n in nets if not n.startswith("unconnected-")} == {n.name for n in ir.nets if n.pins}
    g = schematic_gates(path, ir)
    assert g["crossings"] == 0 and g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0


def inverting_driver(ir):
    """The piano with the driver of its third revision: TL082 unit B as an inverting stage around the BC337 / BC327
    followers, R5 into its '-' (BN), R6 || C8 || C9 back from the speaker node SPK, '+' on VREF."""
    from ai_eda.ir import Net, PinRef

    nets = {n.name: n for n in ir.nets}
    minus = next(p for p in nets["SPK"].pins if p.component_ref == "U3")
    plus = next(p for p in nets["SKB"].pins if p.component_ref == "U3")
    nets["SPK"].pins.remove(minus)
    nets["SPK"].pins += [PinRef(component_ref="R6", pin_number="2"), PinRef(component_ref="C9", pin_number="2")]
    nets["GND"].pins = [p for p in nets["GND"].pins if p.component_ref != "C9"]
    nets["VREF"].pins.append(plus)
    ir.nets = [n for n in ir.nets if n.name not in ("SKA", "SKB")]
    ir.nets.append(Net(name="BN", kind="signal", provenance=ir.nets[0].provenance,
                       pins=[PinRef(component_ref="R5", pin_number="2"), PinRef(component_ref="R6", pin_number="1"),
                             PinRef(component_ref="C8", pin_number="1"), PinRef(component_ref="C9", pin_number="1"), minus]))
    return ir


@needs_libs
def test_an_inverting_stage_around_followers_is_drawn_with_them_its_feedback_over_the_top(tmp_path):
    """The unit whose loop closes through the followers goes to their block with the parts from their node back to its
    '-', though another block holds more parts on its nets; the stage is drawn as one: the followers on the output riser,
    the feedback parts on rows over the top between the '-' riser and the speaker node, the load in line."""
    ir = inverting_driver(op_amp_piano(tmp_path, LIB))
    model = load_model(SchematicCompiler(), ir, LIB)
    assert blocks.follower_loop(model, "U3:2") == (["Q1", "Q2"], ["C8", "C9", "R6"])
    assert blocks.follower_loop(model, "U3:1") is None
    r = build(SchematicCompiler(), ir, LIB)
    assert r.fallbacks == () and "driver=opamp_stage" in r.notes[0], r.notes[0]
    assert any("unit U3B drawn in block driver (its loop runs through Q1/Q2)" in n for n in r.notes)
    assert any("followers Q1/Q2 inside the loop, bias R7 between the nodes, feedback C8+C9+R6 over the top from SPK" in n for n in r.notes)
    at = symbols_at(r.node)
    u, q1, q2 = at[("U3", 2)], at[("Q1", 1)], at[("Q2", 1)]
    for ref in ("R6", "C8", "C9"):
        assert at[(ref, 1)][1] < q1[1] < u[1] < q2[1], ref
        assert u[0] < at[(ref, 1)][0] < at[("J2", 1)][0], ref
    assert u[0] < q1[0] == q2[0] < at[("R8", 1)][0] < at[("C10", 1)][0] < at[("J2", 1)][0]


@needs_kicad
def test_the_inverting_driver_passes_erc_with_no_crossing(tmp_path):
    ir = inverting_driver(op_amp_piano(tmp_path, LIB))
    r = build(SchematicCompiler(), ir, LIB)
    path = tmp_path / f"{ir.project.id}.kicad_sch"
    path.write_text(sexpr.dumps(r.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"], (erc.details["errors"] + erc.details["warnings"])[:3]
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    assert got == {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}
    g = schematic_gates(path, ir)
    assert g["crossings"] == 0 and g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0
