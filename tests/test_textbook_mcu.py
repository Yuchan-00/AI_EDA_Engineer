"""Textbook 1.0 on the MCU boards: the header fan-out plans of the IC hub (LIS against the IC's pin rows, both flips,
deterministic ties) and the boards' gates as drawn today."""

from __future__ import annotations

import time

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.textbook import build, hub
from ai_eda.compilers.textbook.model import load_model
from ai_eda.tools.kicad import KicadLibrary, sexpr
from ai_eda.tools.layout_metrics import schematic_gates
from tests.textbook_corpus import corpus_ir

LIB = KicadLibrary()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")


@pytest.fixture(scope="module")
def boards(tmp_path_factory):
    base = tmp_path_factory.mktemp("mcu")
    return {name: corpus_ir(name, base / name, LIB) for name in ("atmega128", "stm32f407")}


def test_lis_returns_the_indices_of_a_longest_increasing_run_first_found():
    assert hub.lis([]) == []
    assert hub.lis([3, 1, 2, 5, 4]) == [1, 2, 4]  # 1, 2, 4 (the 4 replaced the 5 at the end)
    assert hub.lis([5, 4, 3]) == [2]
    assert hub.lis([1, 1, 2]) == [1, 2]  # strictly increasing


@needs_libs
@pytest.mark.parametrize("name, wired, reached", (("atmega128", 58, 59), ("stm32f407", 55, 115)))
def test_header_plans_wire_the_lis_pins_and_label_the_rest(boards, name, wired, reached):
    model = load_model(SchematicCompiler(), boards[name], LIB)
    (ic,) = hub.hubs(model)
    plans = hub.header_plans(model, ic)
    assert sum(len(p.wired) for p in plans) == wired
    assert sum(p.reached for p in plans) == reached
    assert plans == hub.header_plans(model, ic)  # deterministic
    for p in plans:
        seq = [ic.pin(pin).dy if p.side in ("L", "R") else ic.pin(pin).dx for _, pin in p.wired]
        assert seq == sorted(seq) or seq == sorted(seq, reverse=True)  # a monotone matching: 0 crossings in the channel
        assert {ic.pin(pin).side for _, pin in p.wired} <= {p.side}


@needs_libs
def test_the_atmega_headers_face_the_sides_their_pins_reach(boards):
    model = load_model(SchematicCompiler(), boards["atmega128"], LIB)
    (ic,) = hub.hubs(model)
    sides = {p.header: p.side for p in hub.header_plans(model, ic)}
    assert sides == {"J10": "L", "J2": "L", "J3": "L", "J4": "R", "J5": "R", "J6": "R", "J7": "R", "J8": "L", "J9": "L"}


@needs_libs
@pytest.mark.parametrize("name", ("atmega128", "stm32f407"))
def test_mcu_boards_build_on_one_sheet_without_the_table_rung(boards, name, tmp_path):
    ir = boards[name]
    start = time.perf_counter()
    result = build(SchematicCompiler(), ir, LIB)
    assert time.perf_counter() - start < 10
    assert result.score is not None and "whole-sheet table" not in result.fallbacks, result.notes[0]
    path = tmp_path / f"{name}.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    g = schematic_gates(path, ir)
    assert g["paper"] in ("A4", "A3", "A2") and g["crossings"] <= 9
    assert g["near_symbol_hits"] == 0 and g["parallel_hits"] == 0


@needs_libs
def test_header_plans_flip_a_header_only_for_a_longer_run_and_line_it_up_by_the_median(boards):
    model = load_model(SchematicCompiler(), boards["atmega128"], LIB)
    (ic,) = hub.hubs(model)
    plans = {p.header: p for p in hub.header_plans(model, ic)}
    assert [h for h, p in plans.items() if p.flipped] == ["J2"]  # its pins run against the IC's rows
    assert {h: p.offset for h, p in plans.items()} == {"J2": 20, "J3": 18, "J4": -28, "J5": -10, "J6": 8, "J7": 26, "J8": 26,
                                                        "J9": 8, "J10": -10}
    model = load_model(SchematicCompiler(), boards["stm32f407"], LIB)
    (ic,) = hub.hubs(model)
    plans = {p.header: p for p in hub.header_plans(model, ic)}
    assert [h for h, p in plans.items() if p.flipped] == ["J10"]
    assert len(plans["J2"].wired) == 1 and not plans["J2"].flipped  # one pin: both ways tie, the header stays as drawn
    assert (plans["J4"].offset, plans["J10"].offset) == (-24, -10)
