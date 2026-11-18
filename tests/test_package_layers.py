"""The package start (``design.package_layers`` 0.1, the owner's rule of 2026-11-13): start from the layer count the board's IC packages
and parts commonly use, then add layers when nets stay unrouted or remove them while every net routes, never below a manufacturer's floor.

What is proved:

* the table: every row opened, its text as the research gives it, the rows whose quote came through an HTML extractor marked, the
  import check refuses an unopened or off-ladder row;
* the footprint facts: P from the name's ``P<x>mm`` token or the smallest pad spacing, BODY, N, the BGA test, the ball depth D
  (consecutive rings, a separate centre island not counted; not read when the pads are not on the P grid, KiCad's BGA-152 among them);
* the matching: the MPN when the IR has one, else the value, else the symbol name, case-insensitive with spaces removed, the field
  named in the reason; the generic BGA rows by pitch and depth (the 1.0 mm / 1.27 mm boundary, the 0.5 mm rows, the cap above 12);
  a part row beats a generic row for the start, the floor is the largest of all; footprint-bound part rows, the package row, the
  conditional rows (P_D read, Ethernet and the reference design unreadable, each a note), the catalog-only rows never match; no part
  row on a passive footprint, TI's ``CC12`` row not on another maker's part (Yageo MLCCs);
* the board: the largest start and floor, the P6 minimums as floors, the start never below the floor, rounding to the ladder, a
  floor above the ladder refused, ``applies`` only for a row with a start above 2 or a floor, and no note when it does not apply;
* the PCB agent's search, with canned routing (``tests/test_layer_escalation.py``'s pattern) and with real routing on the fixture
  library: start high then remove (down to the board's own count, today's copper byte for byte), start high then add, the floor and
  a failing lower count stop the removal, the board's own count checked against the fab's options, a stated count untouched (the
  floor's source and quote named), a material's own stacks and a start capped at the ladder, no count from the start buildable (the
  search starts below it, or the P6 start runs as before), nothing routed whole (all or nothing, ``pcb.routing=partial``), and a
  board no row touches routed exactly as before, its notes included.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_eda.agents.pcb as pcb_module
from ai_eda.agents import PCBAgent
from ai_eda.design import package_layers as pl
from ai_eda.design.materials import MATERIAL_STACKS, MaterialDecision, material_stackup
from ai_eda.design.package_layers import (
    ROWS,
    BoardConditions,
    LayerRow,
    ball_depth,
    board_conditions,
    board_layers,
    board_row_notes,
    footprint_facts,
    grid_misfit,
    identities,
    ladder_count,
    name_tokens,
    pad_spacing,
    part_layers,
    plan_for_ir,
)
from ai_eda.design.stackup import ESCALATION_TOOL
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, LibraryRef, ManufacturingConstraints, NetKind, ProvenanceKind, assumption
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad.library import KicadLibrary
from tests.test_layer_escalation import (
    ONE,
    ROUTED,
    _bga,
    _canned,
    _ctx,
    _escalation,
    _gnd_board,
    _material_req,
    _plain,
    _sourced,
    _stated,
    _template_board,
)
from tests.test_routing import board_ir, fixture_library

ROW = {r.id: r for r in ROWS}
AN4488 = ROW["st_an4488_stm32f4"]
AN2519 = ROW["mchp_an2519_avr"]
#: the 2-layer record of a canned own-count routing that routes every net
OWN_ROUTED = "2 copper layers (signals F.Cu, B.Cu; planes none): routing.maze 0.2 routed every net after 3 iteration(s) (1.5 s wall clock)"
FOUR = "4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu GND):"
SIX = "6 copper layers (signals F.Cu, In2.Cu, In3.Cu, B.Cu; planes In1.Cu GND, In4.Cu GND):"
LEFT = "routing.maze 0.9 left 1 of 2 net(s) unrouted (1 ripped up in negotiation) after 3 iteration(s) (1.5 s wall clock)"
#: a 0.5 mm-pitch 6 x 6 BGA: the JLCPCB 0.5 mm row (D = 3), start 6, no floor
BGA05 = "BGA-36_3x3mm_Layout6x6_P0.5mm"


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


@pytest.fixture(autouse=True)
def steady_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pcb_module, "_clock", itertools.count(0.0, 1.5).__next__)


def _pad(number: str, x: float, y: float, pad_type: str = "smd", layers: tuple[str, ...] = ("F.Cu", "F.Mask")):
    return SimpleNamespace(number=number, pad_type=pad_type, layers=list(layers), x=x, y=y)


def _grid(n: int, m: int, pitch: float, *, skip=lambda i, j: False):
    return [_pad(f"{j * n + i + 1}", i * pitch, j * pitch) for j in range(m) for i in range(n) if not skip(i, j)]


def _bga_facts(n: int, pitch: float, name: str | None = None, library: str = "Package_BGA", **kw):
    name = name or f"BGA-{n * n}_10x10mm_Layout{n}x{n}_P{pitch:g}mm"
    return footprint_facts(f"{library}:{name}", name, _grid(n, n, pitch, **kw))


def _mpn(ir: CircuitIR, ref: str, mpn: str) -> CircuitIR:
    ir.component(ref).mpn = assumption(mpn, note="fixture MPN")
    return ir


# --------------------------------------------------------------------------- the table


def test_every_row_is_opened_and_quoted_as_the_research_gives_it():
    assert len(ROWS) == 30 and all(r.opened for r in ROWS) and len({r.id for r in ROWS}) == 30
    assert (AN4488.start, AN4488.floor, AN4488.quote) == (4, 4, "The minimum configuration that can be used is 4 or 6 layers stack-up.")
    assert AN4488.source.startswith("STMicroelectronics AN4488 'Getting started with STM32F4xxxx MCU hardware development', Rev 7 (03-Oct-2018)")
    # the quotes that came through an HTML extractor are marked, and a cited quote says so
    unchecked = {r.id for r in ROWS if not r.quote_checked}
    assert unchecked == {"espressif_esp32", "ti_spruiy5a_am64x_am243x", "ti_sprad13a_am62x", "mchp_an5672_polarfire_fc484", "ti_snvsbd7b_lmr36510",
                         "ti_snvsar0_lm53602_lm53603", "ti_snvsbi9a_lmr33610", "ti_snvaa93_lmr38020_lmr38025", "ti_cc1101_cc1190em869rd"}
    assert ROW["ti_snvsbd7b_lmr36510"].cite(quote=True).endswith('"Use a four-layer board with the copper thickness for the four layers" '
                                                                 "(quoted through an HTML extractor, not re-checked against the PDF)")
    assert AN4488.cite(quote=True).endswith('"The minimum configuration that can be used is 4 or 6 layers stack-up."')
    # the families named only by series or package code never match in this version
    assert {r.id for r in ROWS if r.unresolved} == {"espressif_esp32", "renesas_r01an7402_ra_rx_bga", "mchp_an5672_polarfire_fc484"}


def test_the_import_check_refuses_an_unopened_or_off_ladder_row():
    base = dict(kind="part", start=4, floor=4, opened=True, quote_checked=True, match="m", source="s", quote="q", conditions="c", prefixes=("X",))
    with pytest.raises(ValueError, match="was not opened"):
        pl._check_rows([LayerRow(id="a", **{**base, "opened": False})])
    with pytest.raises(ValueError, match="must be counts of the ladder"):
        pl._check_rows([LayerRow(id="a", **{**base, "start": 5})])
    with pytest.raises(ValueError, match="must be counts of the ladder"):
        pl._check_rows([LayerRow(id="a", **{**base, "start": 4, "floor": 6})])
    with pytest.raises(ValueError, match="duplicate row ids"):
        pl._check_rows([LayerRow(id="a", **base), LayerRow(id="a", **base)])


# --------------------------------------------------------------------------- footprint facts


def test_the_name_tokens_give_pitch_body_and_the_bga_test():
    assert name_tokens("LFBGA-144_10x10mm_Layout12x12_P0.8mm") == (0.8, (10.0, 10.0), True)
    assert name_tokens("BGA-256_17.0x17.0mm_Layout16x16_P1.0mm_Ball0.5mm_Pad0.4mm_NSMD") == (1.0, (17.0, 17.0), True)
    assert name_tokens("LQFP-100_14x14mm_P0.5mm") == (0.5, (14.0, 14.0), False)
    assert name_tokens("Texas_DSBGA-8_1.5x2mm_P0.5mm")[2] is False  # DSBGA is not one of the tokens
    assert name_tokens("BGA10") == (None, None, False)  # a token, not a substring


def test_the_pitch_falls_back_to_the_smallest_pad_spacing_and_n_counts_numbers():
    pads = [*_grid(4, 4, 1.27), _pad("1", 0.0, 0.0), _pad("", 9.0, 9.0), _pad("17", 9.0, 9.0, pad_type="np_thru_hole", layers=("*.Cu",)),
            _pad("18", 20.0, 20.0, layers=("F.Paste",))]
    facts = footprint_facts("Package_BGA:Custom_Array", "Custom_Array", pads)
    assert (facts.bga, facts.pitch, facts.pitch_from, facts.body, facts.pads, facts.depth) == (True, 1.27, "pads", None, 16, 2)
    assert pad_spacing([(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]) == 1.0  # along one row or column only: the diagonal does not count
    assert pad_spacing([(0.0, 0.0)]) is None
    # outside Package_BGA and without a BGA token: no ball depth
    qfn = footprint_facts("Package_DFN_QFN:QFN-16_3x3mm_P0.5mm", "QFN-16_3x3mm_P0.5mm", _grid(4, 4, 0.5))
    assert (qfn.bga, qfn.depth, qfn.library) == (False, None, "Package_DFN_QFN")


def test_the_ball_depth_counts_consecutive_rings_from_the_edge():
    full = [(i * 0.8, j * 0.8) for i in range(10) for j in range(10)]
    assert ball_depth(full, 0.8) == 5  # ceil(10 / 2)
    assert ball_depth([(i, j) for i in range(16) for j in range(12)], 1.0) == 6  # the shorter side: ceil(12 / 2)
    # a 15 x 15 array with rings 0-2 and a separate centre island (rings 5-7): the empty ring 3 stops the count
    island = [(i, j) for i in range(15) for j in range(15) if min(i, j, 14 - i, 14 - j) not in (3, 4)]
    assert ball_depth(island, 1.0) == 3
    assert ball_depth([(i, j) for i in range(8) for j in range(8) if min(i, j, 7 - i, 7 - j) == 0], 1.0) == 1  # a perimeter ring
    assert ball_depth([], 1.0) == 0


def test_a_bga_whose_pads_are_not_on_its_pitch_grid_has_no_ball_depth():
    """Review finding: on the name's P grid a footprint whose balls sit 2 P apart has every odd ring empty, and D came out 1 silently
    (KiCad's BGA-152_14x18mm_Layout13x17_P0.5mm: pads 1.0 mm apart)."""
    name = "BGA-152_14x18mm_Layout13x17_P0.5mm"
    coarse = footprint_facts(f"Package_BGA:{name}", name, _grid(13, 17, 1.0))
    assert (coarse.pitch, coarse.depth, coarse.depth_why) == (0.5, None, "the smallest pad spacing is 1 mm, not the pitch of 0.5 mm")
    assert ball_depth([(i * 1.0, j * 1.0) for j in range(17) for i in range(13)], 0.5) == 1  # what the change counted
    p = part_layers("U1", [], None, coarse)
    assert p.matches == [] and p.notes == [f"U1: footprint Package_BGA:{name}: the ball depth D is not read - the smallest pad spacing is 1 mm, "
                                           "not the pitch of 0.5 mm; the BGA rows that need it do not match"]
    # a staggered array: every other row half a pitch over
    stagger = [_pad(f"{j * 10 + i + 1}", i * 0.8 + (0.4 if j % 2 else 0.0), j * 0.8) for j in range(10) for i in range(10)]
    s = footprint_facts("Package_BGA:Stagger-100_P0.8mm", "Stagger-100_P0.8mm", stagger)
    assert (s.depth, s.depth_why) == (None, "50 of 100 pad centres lie more than 0.25 pitch off the 0.8 mm grid")
    assert grid_misfit([(i * 0.8, j * 0.8) for i in range(4) for j in range(4)], 0.8) is None
    # KiCad's 0.01 mm coordinate rounding is no misfit: a 1.27 mm array stepping 1.26 to 1.28 mm keeps its depth
    rounded = [_pad(f"{j * 8 + i + 1}", round((i - 3.5) * 1.27, 2), round((j - 3.5) * 1.27, 2)) for j in range(8) for i in range(8)]
    r = footprint_facts("Package_BGA:BGA-64_P1.27mm", "BGA-64_P1.27mm", rounded)
    assert pad_spacing([(p.x, p.y) for p in rounded]) == 1.26 and (r.depth, r.depth_why) == (4, None)


KICAD = KicadLibrary()


@pytest.mark.skipif(KICAD.footprint_file("Package_BGA", "BGA-152_14x18mm_Layout13x17_P0.5mm") is None,
                    reason="KiCad's Package_BGA library not installed")
def test_kicads_own_bga_152_and_bga_352_footprints():
    """The installed KiCad library: BGA-152's pads are 1.0 mm apart under a P0.5mm name (no depth, said); BGA-352's 0.01 mm-rounded
    1.27 mm array keeps D = 4 and National AN-1126's 1.27 mm row."""
    def facts(name: str):
        fp = KICAD.load_footprint(LibraryRef(library="Package_BGA", name=name))
        return footprint_facts(fp.lib_id, fp.name, fp.pads)

    bga152 = facts("BGA-152_14x18mm_Layout13x17_P0.5mm")
    assert (bga152.pitch, bga152.depth, bga152.depth_why) == (0.5, None, "the smallest pad spacing is 1 mm, not the pitch of 0.5 mm")
    assert part_layers("U1", [], None, bga152).matches == []
    bga352 = facts("BGA-352_35.0x35.0mm_Layout26x26_P1.27mm")
    assert (bga352.depth, bga352.depth_why) == (4, None)
    assert [m.row.id for m in part_layers("U1", [], None, bga352).matches] == ["ns_an1126_bga_p127"]


# --------------------------------------------------------------------------- matching a part


def test_part_rows_match_the_mpn_else_the_value_else_the_symbol():
    assert identities("STM32F407VET6", "10k", "STM32F407VETx") == [("MPN", "STM32F407VET6")]
    assert identities(None, "STM32F407VET6", "STM32F407VETx") == [("value", "STM32F407VET6"), ("symbol", "STM32F407VETx")]
    p = part_layers("U1", identities(None, "STM32F407VET6", "STM32F407VETx"), None, None)
    assert (p.start, p.floor, p.applies) == (4, 4, True) and [m.row.id for m in p.matches] == ["st_an4488_stm32f4"]
    assert p.matches[0].why == "value 'STM32F407VET6' starts with 'STM32F4'"
    # the symbol name when the value says nothing; case and spaces do not matter
    p = part_layers("U1", identities(None, "MCU", "stm32 f407vetx"), None, None)
    assert p.matches[0].why == "symbol 'stm32 f407vetx' starts with 'STM32F4'"
    # an MPN that names another part wins over a value that would match
    assert part_layers("U1", identities("VR1-0603-200V-A", "STM32F407VET6", None), None, None).matches == []
    # MSP430 is not MSP432E4's prefix; CC12 matches the CC1200
    assert [m.row.id for m in part_layers("U1", [("MPN", "MSP432E401YTPDTR")], None, None).matches] == ["ti_slaa770_msp432e4"]
    assert [m.row.id for m in part_layers("U1", [("MPN", "CC1200RHBR")], None, None).matches] == ["ti_swra407_cc112x_cc117x_cc12xx"]
    # a part no row matches: start 2, floor 0, nothing applies
    nothing = part_layers("R1", [("value", "10k")], None, footprint_facts("Package_QFP:LQFP-48_7x7mm_P0.5mm", "LQFP-48_7x7mm_P0.5mm", _grid(4, 4, 0.5)))
    assert (nothing.start, nothing.floor, nothing.applies, nothing.notes) == (2, 0, False, [])


@pytest.mark.parametrize(("n", "pitch", "row", "start"), [
    (8, 0.8, "nxp_an10778_bga_d4", 2),     # D = 4: the mounting layer and one more escape it
    (12, 0.8, "nxp_an10778_bga_d6", 4),    # LFBGA-144: D = 6
    (16, 1.0, "nxp_an10778_bga_d8", 6),    # BGA-256 at 1.0 mm: NXP, not AN-1126 (the table's conflict rule)
    (20, 0.65, "nxp_an10778_bga_d10", 8),
    (24, 0.8, "nxp_an10778_bga_d12", 10),
    (26, 0.8, "nxp_an10778_bga_d13", 12),
    (6, 1.27, "ns_an1126_bga_p127", 4),    # 1.27 mm, D = 3
    (6, 0.5, "jlcpcb_bga_p05", 6),         # 0.5 mm, D = 3
])
def test_the_generic_bga_rows_by_pitch_and_depth(n: int, pitch: float, row: str, start: int):
    p = part_layers("U1", [("value", "X")], None, _bga_facts(n, pitch))
    assert [m.row.id for m in p.matches] == [row] and (p.start, p.floor) == (start, 0)
    assert p.matches[0].why.startswith(f"footprint Package_BGA:BGA-{n * n}_10x10mm_Layout{n}x{n}_P{pitch:g}mm, BGA, P {pitch:g} mm, body 10 x 10 mm")


def test_bga_rows_outside_their_ranges_and_the_cap_above_the_ladder():
    assert part_layers("U1", [], None, _bga_facts(4, 0.5)).matches == []      # 0.5 mm, D = 2: the outer rows escape without a fan-out via
    assert part_layers("U1", [], None, _bga_facts(8, 0.4)).matches == []      # 0.4 mm: no row (a gap of the research)
    assert part_layers("U1", [], None, _bga_facts(8, 1.27)).start == 4       # D = 4 at 1.27 mm: AN-1126
    p = part_layers("U1", [], None, _bga_facts(30, 1.0))                     # D = 15: 13 layers by the rule, capped at 12
    assert p.start == 12 and p.notes == ["U1: D = 15 gives 13 layers by NXP AN10778 sec. 5.2, which exceeds the ladder; the start is capped at 12"]


def test_a_part_row_beats_a_generic_row_and_the_floor_is_the_largest():
    lpc = part_layers("U1", [("MPN", "LPC55S69JEV98")], None, _bga_facts(10, 0.5))
    assert {m.row.id for m in lpc.matches} == {"jlcpcb_bga_p05", "nxp_an13033_lpc55"}
    assert (lpc.start, lpc.floor) == (4, 0) and [m.row.id for m in lpc.start_rows] == ["nxp_an13033_lpc55"]  # 4, not the generic 6
    # the Ethernet floor is skipped with its reason: no requirement key names interfaces
    [note] = lpc.notes
    assert note.startswith("U1 (MPN 'LPC55S69JEV98' starts with 'LPC55'): nxp_an13033_lpc55_ethernet (start 4, floor 4, NXP AN13033 Rev. 2.2")
    assert note.endswith(f"skipped - {pl.UNREADABLE[pl.ETHERNET]}")
    # an STM32F4 in a 0.5 mm BGA: its part row (start 4, floor 4) wins over the generic 0.5 mm start of 6
    stm = part_layers("U1", [("MPN", "STM32F429BIT6")], None, _bga_facts(10, 0.5))
    assert (stm.start, stm.floor) == (4, 4)


def test_footprint_bound_part_rows_and_the_package_row():
    am64 = [("MPN", "AM6442BSDGHAALV")]
    ok = footprint_facts("Package_BGA:FCBGA-441_17.2x17.2mm_Layout21x21_P0.8mm", "FCBGA-441_17.2x17.2mm_Layout21x21_P0.8mm", _grid(21, 21, 0.8))
    p = part_layers("U1", am64, None, ok)
    assert (p.start, p.floor) == (6, 6) and "ti_spruiy5a_am64x_am243x" in {m.row.id for m in p.matches}
    # the same part in another package: said, not applied
    p = part_layers("U1", am64, None, _bga_facts(10, 0.8))
    assert "ti_spruiy5a_am64x_am243x" not in {m.row.id for m in p.matches}
    assert any(n.startswith("U1 (MPN 'AM6442BSDGHAALV' starts with 'AM64'): ti_spruiy5a_am64x_am243x not applied - footprint ") and n.endswith(
        "body is not 17.2 x 17.2 mm") for n in p.notes)
    # TFBGA296 (N, P, BODY exact): a package row with a floor of 6
    corners = {(a, b) for a in (0, 1, 16, 17) for b in (0, 1, 16, 17) if (a in (0, 17) or b in (0, 17)) and not (a in (1, 16) and b in (1, 16))}
    pads = _grid(18, 18, 0.8, skip=lambda i, j: (7 <= i <= 10 and 7 <= j <= 10) or (i, j) in corners)
    assert len(pads) == 296
    tf = part_layers("U1", [], None, footprint_facts("Package_BGA:TFBGA-296_15x15mm_P0.8mm", "TFBGA-296_15x15mm_P0.8mm", pads))
    assert "nxp_an10778_tfbga296" in {m.row.id for m in tf.matches} and (tf.start, tf.floor) == (6, 6)
    assert [m.row.id for m in tf.floor_rows] == ["nxp_an10778_tfbga296"]
    # MIMXRT1062 needs a BGA footprint
    assert part_layers("U1", [("MPN", "MIMXRT1062DVL6A")], None, _bga_facts(14, 0.8)).floor == 4
    assert part_layers("U1", [("MPN", "MIMXRT1062DVL6A")], None, None).floor == 0


def test_the_conditional_rows_read_power_dissipation_and_say_what_they_cannot_read():
    qfn = footprint_facts("Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm", "QFN-32-1EP_5x5mm_P0.5mm", _grid(4, 4, 0.5))
    hot = BoardConditions(power_dissipation_w=7.0, power_dissipation_req="req.power_dissipation")
    p = part_layers("U1", [("MPN", "CY8C4245LQI-483")], "Infineon Technologies", qfn, hot)
    assert (p.start, p.floor) == (4, 4)
    assert p.matches[0].why == ("manufacturer 'Infineon Technologies' names Infineon; footprint Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm, P 0.5 mm, "
                                "body 5 x 5 mm, N 16; req.power_dissipation states P_D = 7 W (> 5 W)")
    cool = BoardConditions(power_dissipation_w=5.0, power_dissipation_req="req.power_dissipation")
    assert part_layers("U1", [], "Cypress Semiconductor", qfn, cool).matches == []  # 5 W is not more than 5 W
    unstated = part_layers("U1", [], "Infineon", qfn, BoardConditions())
    assert unstated.matches == [] and unstated.notes[0].endswith("skipped - the requirements state no power_dissipation (the system's total P_D is "
                                                                 "never estimated)")
    assert part_layers("U1", [], "Texas Instruments", qfn, hot).matches == []
    # the CC1101 + CC1190 board row: both parts present, the reference design cannot be read
    matches, notes = board_row_notes([("U1", [("MPN", "CC1101RGPR")]), ("U2", [("value", "CC1190RGVR")])])
    assert matches == [] and notes == [
        f"board (U1 MPN 'CC1101RGPR' starts with 'CC1101'; U2 value 'CC1190RGVR' starts with 'CC1190'): ti_cc1101_cc1190em869rd (start 4, floor 4, "
        f"{ROW['ti_cc1101_cc1190em869rd'].source}) skipped - {pl.UNREADABLE[pl.CC1190_REFERENCE]}"]
    assert board_row_notes([("U1", [("MPN", "CC1101RGPR")])]) == ([], [])


def _two_pad(lib_id: str):
    return footprint_facts(lib_id, lib_id.split(":")[1], [_pad("1", -1.0, 0.0), _pad("2", 1.0, 0.0)])


def test_a_yageo_mlcc_is_not_a_ti_cc12xx_radio():
    """Review finding: 'CC12' also starts Yageo MLCC part numbers (Digi-Key lists CC1206KRX7R9BB104 as a YAGEO 0.1 uF 50 V X7R 1206
    capacitor), so a plain capacitor started the board at 4 layers citing a TI RF design note."""
    cap = _two_pad("Capacitor_SMD:C_1206_3216Metric")
    for mpn in ("CC1206KRX7R9BB104", "CC1210KKX7R7BB105"):
        assert part_layers("C1", identities(mpn, "100n", "C"), "YAGEO", cap).matches == []
        assert part_layers("C1", identities(mpn, "100n", "C"), None, cap).matches == []      # a passive footprint: no part row
        assert part_layers("C1", identities(mpn, "100n", "C"), "YAGEO", None).matches == []  # another maker's part number
    # TI's own parts still match, the manufacturer field unset or naming TI as a word
    qfn = footprint_facts("Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm", "QFN-32-1EP_5x5mm_P0.5mm", _grid(4, 4, 0.5))
    for maker in (None, "", "Texas Instruments", "TI", "Texas Instruments Inc."):
        assert [m.row.id for m in part_layers("U1", [("MPN", "CC1200RHBR")], maker, qfn).matches] == ["ti_swra407_cc112x_cc117x_cc12xx"]
    assert part_layers("U1", [("MPN", "CC1200RHBR")], "Lattice Semiconductor", qfn).matches == []  # 'TI' is not a word of 'Lattice'
    # no part row matches a resistor / inductor footprint either, whatever its value says
    assert part_layers("R1", [("value", "STM32F407VET6")], None, _two_pad("Resistor_SMD:R_0603_1608Metric")).matches == []
    assert part_layers("L1", [("value", "ATmega328P")], None, _two_pad("Inductor_SMD:L_0805_2012Metric")).matches == []
    # the board: nothing applies, nothing is said
    b = board_layers([part_layers("C1", identities("CC1206KRX7R9BB104", "100n", "C"), "YAGEO", cap)])
    assert (b.start, b.applies, b.notes, b.start_why) == (2, False, [], [])


def test_a_plan_that_does_not_apply_says_nothing():
    """Review finding: the parts' and the board rows' notes were carried, and written by the PCB agent, on boards nothing applies to."""
    qfn = footprint_facts("Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm", "QFN-32-1EP_5x5mm_P0.5mm", _grid(4, 4, 0.5))
    unstated = part_layers("U1", [], "Infineon", qfn)  # an Infineon QFN, no power_dissipation stated: its row is skipped, with a note
    assert len(unstated.notes) == 1 and not unstated.applies
    board_note = "board (U2 ...): ti_cc1101_cc1190em869rd skipped"
    b = board_layers([unstated], notes=[board_note])
    assert (b.applies, b.notes, b.parts[0].notes) == (False, [], unstated.notes)
    # with a part that applies, every note is said
    b = board_layers([unstated, part_layers("U2", [("MPN", "STM32F407VET6")], None, None)], notes=[board_note])
    assert b.applies and b.notes == [board_note, *unstated.notes]
    # a row named by the manufacturer alone says nothing of the maker's other packages
    other = footprint_facts("Package_TO_SOT_SMD:Fixture-3", "Fixture-3", [_pad("1", 0.0, 0.0), _pad("2", 1.9, 0.0), _pad("3", 0.95, 2.0)])
    assert part_layers("Q1", [], "Infineon Technologies", other).notes == []
    assert part_layers("Q1", [], "Infineon Technologies", None).notes == []


def test_board_conditions_read_the_confirmed_power_dissipation(tmp_path: Path, lib: KicadLibrary):
    ir = _plain(tmp_path, lib)
    assert board_conditions(ir) == BoardConditions()
    ir.requirements.requirements.append(_material_req("power_dissipation", "7 W"))
    c = board_conditions(ir)
    assert (c.power_dissipation_w, c.power_dissipation_req) == (7.0, "req.power_dissipation")


# --------------------------------------------------------------------------- the board


def test_the_board_takes_the_largest_start_and_floor_and_rounds_to_the_ladder():
    avr = part_layers("U1", [("MPN", "ATmega328P-AU")], None, None)
    stm = part_layers("U2", [("MPN", "STM32F407VET6")], None, None)
    big = part_layers("U3", [], None, _bga_facts(16, 1.0))  # start 6, no floor
    b = board_layers([avr, stm, big])
    assert (b.start, b.floor, b.applies, b.refusal) == (6, 4, True, None)
    assert b.start_why == [big.start_rows[0].line()] and b.floor_why == [stm.floor_rows[0].line(quote=True)]
    assert b.derived == ["package_layers:U1=mchp_an2519_avr", "package_layers:U2=st_an4488_stm32f4", "package_layers:U3=nxp_an10778_bga_d8"]
    # the P6 minimums are floors: a sourced 5 rounds up to 6 and raises the start with it
    b = board_layers([avr], [(5, "R1 needs at least 5 copper layers (Vendor X1 datasheet rev. B)", "components[R1].min_pcb_layers")])
    assert (b.start, b.floor) == (6, 6) and b.start_why == b.floor_why == ["R1 needs at least 5 copper layers (Vendor X1 datasheet rev. B)"]
    assert "the floor of 5 copper layers rounds up to 6, the next count of the ladder" in b.notes
    assert b.derived[-1] == "components[R1].min_pcb_layers"
    # a floor above the ladder refuses the plan
    assert board_layers([avr], [(14, "R1 needs 14", "d")]).refusal.startswith("the floor of 14 copper layers is above the ladder's last count 12")
    # the P6 minimums alone do not make the package search apply (the P6 start stays as it was)
    assert board_layers([part_layers("R1", [("value", "10k")], None, None)], [(6, "R1 needs 6", "d")]).applies is False
    # and such a plan adds no note (review finding: a P6 minimum of 5 got a "rounds up to 6" line, changing the board's notes)
    assert board_layers([part_layers("R1", [("value", "10k")], None, None)], [(5, "R1 needs 5", "d")]).notes == []
    assert [ladder_count(n) for n in (0, 1, 2, 3, 4, 5, 11, 12, 13)] == [2, 2, 2, 4, 4, 6, 12, 12, None]


def test_the_plan_reads_the_irs_parts_and_their_footprints(tmp_path: Path, lib: KicadLibrary):
    ir = _bga_ir(tmp_path, lib)
    b = plan_for_ir(ir, lib)
    assert (b.start, b.floor, b.applies) == (6, 0, True) and b.derived == ["package_layers:R2=jlcpcb_bga_p05"]
    assert b.start_why[0].startswith(f"R2 (footprint Test:{BGA05}, BGA, P 0.5 mm, body 3 x 3 mm, N 36, D 3): jlcpcb_bga_p05 start 6, floor none - "
                                     "JLCPCB blog 'BGA PCB Design Complete Guide")
    # a board no row touches
    assert plan_for_ir(_plain(tmp_path, lib), lib).applies is False


# --------------------------------------------------------------------------- the PCB agent's search (canned routing)


def _bga_ir(tmp_path: Path, lib: KicadLibrary, mpn: str | None = None) -> CircuitIR:
    """``_plain`` with R2 (on net A) on a 0.5 mm 6 x 6 BGA: the JLCPCB row's start of 6, no floor."""
    _bga(lib, BGA05, 6, 0.5, 0.3)
    ir = _plain(tmp_path, lib)
    r2 = ir.component("R2")
    r2.footprint = lib.resolve_footprint(LibraryRef(library="Test", name=BGA05))
    assert r2.footprint.verified
    if mpn is not None:
        _mpn(ir, "R2", mpn)
    return ir


def _payload(res) -> str:
    return json.dumps([design_data(p.payload) for p in res.proposals], sort_keys=True)


def test_start_high_then_remove_down_to_the_boards_own_count_keeps_todays_copper(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: True, 4: True})
    today = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))
    assert calls == [(2, None)] and _escalation(today.notes) == []
    calls.clear()
    res = PCBAgent().run(_mpn(_plain(tmp_path, lib), "R1", "ATmega328P-AU"), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4"), (2, None)]
    assert _payload(res) == _payload(today) and res.proposals[0].rationale == today.proposals[0].rationale
    assert res.proposals[0].payload.stackup is None
    assert _escalation(res.notes) == [
        f"layer escalation: package start 4 copper layers (design.package_layers 0.1) - R1 (MPN 'ATmega328P-AU' starts with 'ATmega'): "
        f"mchp_an2519_avr start 4, floor none - {AN2519.source}",
        "layer escalation: package floor: none - no matched row states a recommended, required or minimum count, so layers may be removed down to "
        "the board's own count",
        f"layer escalation: {FOUR} {ROUTED}",
        f"layer escalation: fewer layers: {OWN_ROUTED}",
        "layer escalation: 2 layers kept - every net routes on the board's own 2 copper layers (the router as before, no stack): the fewest from "
        "the package start 4 at which every net routes (the project owner's rule)",
    ]
    # the notes besides the escalation lines are today's
    assert [n for n in res.notes if not n.startswith("layer escalation")] == today.notes


def test_start_high_then_add_never_goes_below_the_floor(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The STM32F407 example's case: ST AN4488 starts it at 4 with a floor of 4; 4 leaves nets unrouted, 6 routes and is kept."""
    calls = _canned(monkeypatch, {4: False, 6: True})
    ir = _plain(tmp_path, lib)
    ir.component("R1").mpn = None
    ir.component("R1").value = "STM32F407VET6"
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4"), (6, "6-layer 1.6 mm FR-4")]  # 2 is never routed
    why = f"R1 (value 'STM32F407VET6' starts with 'STM32F4'): st_an4488_stm32f4 start 4, floor 4 - {AN4488.source}"
    lines = _escalation(res.notes)
    assert lines == [
        f"layer escalation: package start 4 copper layers (design.package_layers 0.1) - {why}",
        f"layer escalation: package floor 4 copper layers - {why}: \"{AN4488.quote}\"",
        f"layer escalation: {FOUR} {LEFT}",
        f"layer escalation: {SIX} {ROUTED}",
        "layer escalation: 6 layers kept - the first count of 2, 4, 6, 8, 10, 12 from the package start 4 up at which every net routes (the "
        "project owner's rule)",
    ]
    stack = res.proposals[0].payload.stackup
    prov = stack.provenance
    assert stack.layer_count == 6 and (prov.kind, prov.tool, prov.tool_version) == (ProvenanceKind.DERIVED, "design.package_layers", "0.1")
    assert prov.derived_from == ["pcb.placements", "layer_escalation:2,4,6,8,10,12", "package_layers:R1=st_an4488_stm32f4"]
    assert prov.note.startswith("layer count 6: the package start left nets unrouted; the first count above it at which every net routes - the "
                                f"package rule ({pl.PACKAGE_RULE}) with the project owner's layer-escalation rule (")
    assert f"; package start 4: {why}; package floor 4: {why}: \"{AN4488.quote}\" - 4 copper layers (" in prov.note
    assert res.proposals[0].rationale.endswith("DRC decides validity; layer count 6 by the package-start rule and the project owner's "
                                               "layer-escalation rule (see ir.pcb.stackup provenance); " + pcb_module.SILK_RATIONALE)


def test_the_floor_stops_the_removal(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: True, 4: True})
    res = PCBAgent().run(_mpn(_plain(tmp_path, lib), "R1", "STM32F407VET6"), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")]  # every net would route on 2, but the floor is 4
    assert _escalation(res.notes)[-1] == ("layer escalation: 4 layers kept - the fewest copper layers from the package start 4 down to the package "
                                          "floor 4 at which every net routes (the package floor of 4 copper layers allows no fewer; the project "
                                          "owner's rule)")
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4 and stack.provenance.note.startswith(
        "layer count 4: the fewest copper layers from the package start at which every net routes (the package floor of 4 copper layers allows no fewer)")
    # a P6 minimum is a floor too: an AVR (start 4, no floor) with a part's sourced minimum of 6 starts and stays at 6
    calls = _canned(monkeypatch, {2: True, 4: True, 6: True})
    ir = _mpn(_plain(tmp_path, lib), "R1", "ATmega328P-AU")
    ir.component("R2").min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4")]
    assert _escalation(res.notes)[1] == "layer escalation: package floor 6 copper layers - R2 needs at least 6 copper layers (Vendor X1 datasheet rev. B, p. 12 'PCB layout')"
    assert res.proposals[0].payload.stackup.provenance.derived_from[-2:] == ["package_layers:R1=mchp_an2519_avr", "components[R2].min_pcb_layers"]
    # a count between the kept one and the floor that the fab does not offer is passed over, and the note says no lower count was routed
    calls = _canned(monkeypatch, {6: True, 4: True})
    ir = _bga_ir(tmp_path, lib)
    _mpn(ir, "R1", "STM32F407VET6")  # floor 4; the 0.5 mm BGA on R2 starts the board at 6
    ir.pcb.manufacturing = ManufacturingConstraints(layer_count_options=assumption([2, 6], note="a fab that builds 2 and 6 layers"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4")]
    assert _escalation(res.notes)[-2:] == [
        "layer escalation: fewer layers: 4 copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [2, 6]",
        "layer escalation: 6 layers kept - the fewest copper layers from the package start 6 down to the package floor 4 at which every net "
        "routes (no count below 6 down to the package floor of 4 could be routed; the project owner's rule)"]


def test_a_lower_count_that_leaves_nets_unrouted_stops_the_removal(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {6: True, 4: False, 2: True})
    res = PCBAgent().run(_bga_ir(tmp_path, lib), _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4"), (4, "4-layer 1.6 mm FR-4")]  # 2 is never tried below a count that failed
    assert _escalation(res.notes)[-2:] == [
        f"layer escalation: fewer layers: {FOUR} {LEFT}",
        "layer escalation: 6 layers kept - the fewest copper layers from the package start 6 at which every net routes (4 copper layers left 1 "
        "net(s) unrouted; the project owner's rule)"]
    assert res.proposals[0].payload.stackup.layer_count == 6
    # 4 routes, 2 does not: the 4-layer stack is kept, its record lists every count tried, up and down
    calls = _canned(monkeypatch, {6: True, 4: True, 2: False})
    res = PCBAgent().run(_bga_ir(tmp_path, lib), _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4"), (4, "4-layer 1.6 mm FR-4"), (2, None)]
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4 and [(z.net, z.layer) for z in res.proposals[0].payload.zones] == [("GND", "In1.Cu"), ("GND", "In2.Cu")]
    assert (f"{SIX} routing.maze 0.9 routed every net after 3 iteration(s); fewer layers: {FOUR} routing.maze 0.9 routed every net after 3 "
            "iteration(s); fewer layers: 2 copper layers "
            "(signals F.Cu, B.Cu; planes none): routing.maze 0.2 left 1 of 2 net(s) unrouted") in stack.provenance.note
    # a part row overrides the generic 0.5 mm start: an LPC55 on that BGA starts at 4 (and its Ethernet floor is skipped, with the reason)
    calls = _canned(monkeypatch, {4: True, 2: True})
    res = PCBAgent().run(_bga_ir(tmp_path, lib, mpn="LPC55S69JEV98"), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4"), (2, None)]
    assert any("nxp_an13033_lpc55_ethernet" in n and n.endswith(pl.UNREADABLE[pl.ETHERNET]) for n in _escalation(res.notes))


def test_a_stated_count_is_untouched_and_the_floors_source_is_named(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: True})
    ir = _stated(_mpn(_plain(tmp_path, lib), "R1", "STM32F407VET6"), 2)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)] and res.proposals[0].payload.stackup is None
    why = f"R1 (MPN 'STM32F407VET6' starts with 'STM32F4'): st_an4488_stm32f4 start 4, floor 4 - {AN4488.source}"
    lines = _escalation(res.notes)
    assert lines[2].startswith("layer escalation: the package start of 4 copper layers is not applied - req.layers states 2 copper layer(s), the "
                               "user's choice, which is never escalated")
    assert lines[3] == f"layer escalation: req.layers states 2 copper layer(s), below the package floor of 4 - {why}: \"{AN4488.quote}\""
    # the copper is the stated board's, exactly as without the part
    calls.clear()
    plain = PCBAgent().run(_stated(_plain(tmp_path, lib), 2), _ctx(tmp_path, lib))
    assert calls == [(2, None)] and _payload(plain) == _payload(res)


def test_a_material_board_searches_only_its_own_stacks(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    ro2, ro4 = MATERIAL_STACKS[("rogers_ro4350b", 2)], MATERIAL_STACKS[("rogers_ro4350b", 4)]

    def board(mpn: str) -> CircuitIR:
        stack = material_stackup(ro2, MaterialDecision(material="rogers_ro4350b", reason="carrier 2.4 GHz"), "divider", confirmed=True,
                                 ground_net="GND", power_net="+5V")
        return _mpn(_template_board(tmp_path, lib, "divider", stack), "R1", mpn)

    # an AVR (no floor): its own 4-layer stack, then its own 2-layer board as it is
    calls = _canned(monkeypatch, {2: True, 4: True})
    ir = board("ATmega328P-AU")
    own = ir.pcb.stackup
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, ro4.title), (2, None)] and res.proposals[0].payload.stackup == own
    # an STM32F4 (floor 4): the RO4350B 4-layer stack is kept with its material record
    calls = _canned(monkeypatch, {4: True})
    res = PCBAgent().run(board("STM32F407VET6"), _ctx(tmp_path, lib))
    stack = res.proposals[0].payload.stackup
    assert calls == [(4, ro4.title)] and stack.material.id == "rogers_ro4350b" and stack.layer_count == 4
    assert stack.provenance.derived_from[-2:] == ["package_layers:R1=st_an4488_stm32f4", "material:rogers_ro4350b"]
    # a floor no stack of the material reaches: not applied, the board is routed at its own count as before - the P6 start says so too
    calls = _canned(monkeypatch, {2: True})
    ir = board("ATmega328P-AU")
    ir.component("R2").min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    assert "layer escalation: the package start of 6 copper layers is not applied - no count of the ladder (4) reaches the package floor of 6" in res.notes
    assert ("layer escalation: the parts need at least 6 copper layers (R2 needs at least 6 copper layers (Vendor X1 datasheet rev. B, p. 12 "
            "'PCB layout')); not applied - no count of the ladder (4) has that many") in res.notes


def test_a_start_above_the_ladder_is_capped_and_a_reachable_floor_holds(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Review finding: a material board whose stacks stop below the package start (RO4350B: 2 and 4; the 0.5 mm BGA starts at 6) is
    searched from its last stack, so a floor of 4 it can build holds (the search was skipped and the board routed at 2, below it)."""
    ro4 = MATERIAL_STACKS[("rogers_ro4350b", 4)]

    def board() -> CircuitIR:
        _bga(lib, BGA05, 6, 0.5, 0.3)
        stack = material_stackup(MATERIAL_STACKS[("rogers_ro4350b", 2)], MaterialDecision(material="rogers_ro4350b", reason="carrier 2.4 GHz"),
                                 "divider", confirmed=True, ground_net="GND", power_net="+5V")
        ir = _template_board(tmp_path, lib, "divider", stack)
        ir.component("R2").footprint = lib.resolve_footprint(LibraryRef(library="Test", name=BGA05))
        return ir

    calls = _canned(monkeypatch, {2: True, 4: True})
    res = PCBAgent().run(_mpn(board(), "R1", "STM32F407VET6"), _ctx(tmp_path, lib))  # AN4488: floor 4
    assert calls == [(4, ro4.title)]
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4 and stack.material.id == "rogers_ro4350b" and stack.provenance.tool == "design.package_layers"
    assert "layer escalation: the package start of 6 copper layers is above the ladder's last count (2, 4); the search starts at 4" in res.notes
    assert "; package start 6 (the search started at 4, the ladder's last count): R2 (footprint Test:" in stack.provenance.note
    # a part's sourced P6 minimum of 4 is the same floor (the base commit 22d7895 keeps 4 layers on this board)
    calls = _canned(monkeypatch, {2: True, 4: True})
    ir = board()
    ir.component("R1").min_pcb_layers = _sourced(4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, ro4.title)] and res.proposals[0].payload.stackup.layer_count == 4
    # no floor: from the capped start down to the board's own stack, kept as it is
    calls = _canned(monkeypatch, {2: True, 4: True})
    ir = board()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, ro4.title), (2, None)] and res.proposals[0].payload.stackup == ir.pcb.stackup


def test_when_no_count_from_the_start_can_be_built_the_boards_own_count_is_routed(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: True})
    ir = _mpn(_plain(tmp_path, lib), "R1", "STM32F407VET6")
    ir.pcb.manufacturing = ManufacturingConstraints(layer_count_options=assumption([2], note="a fab that builds 2 layers only"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)] and res.proposals[0].payload.stackup is None
    lines = _escalation(res.notes)
    assert lines[2:7] == [f"layer escalation: {n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [2]" for n in (4, 6, 8, 10, 12)]
    assert lines[7] == ("layer escalation: no count from the package start 4 up could be built; the board is routed as without the package search - "
                        "below the package floor of 4")


def _fab(*counts: int) -> ManufacturingConstraints:
    return ManufacturingConstraints(layer_count_options=assumption(list(counts), note=f"a fab that builds {counts} layers"))


def test_when_no_count_from_the_start_can_be_built_the_search_starts_below_it(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Review finding: a fab that builds 2 and 4 layers, the 0.5 mm BGA on R2 (start 6) and an STM32F4 on R1 (floor 4). 6 to 12 are
    skipped, so the search starts at 4, the highest count below the start at or above the floor, and keeps it (the change fell back to
    the board's own 2 layers, below the floor; the STM32F4 alone kept 4)."""

    def board(stm32: bool = True) -> CircuitIR:
        ir = _bga_ir(tmp_path, lib)
        if stm32:
            _mpn(ir, "R1", "STM32F407VET6")
        ir.pcb.manufacturing = _fab(2, 4)
        return ir

    calls = _canned(monkeypatch, {2: True, 4: True})
    res = PCBAgent().run(board(), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")]
    assert _escalation(res.notes)[2:] == [
        *(f"layer escalation: {n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [2, 4]" for n in (6, 8, 10, 12)),
        "layer escalation: no count from the package start 6 up could be built; the search starts at the highest count below it (down to the "
        "package floor of 4) that can be",
        f"layer escalation: fewer layers: {FOUR} {ROUTED}",
        "layer escalation: 4 layers kept - the fewest copper layers from the package start 6 down to the package floor 4 at which every net "
        "routes (the package floor of 4 copper layers allows no fewer; the project owner's rule)",
    ]
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4 and stack.provenance.tool == "design.package_layers"
    assert stack.provenance.derived_from[-2:] == ["package_layers:R1=st_an4488_stm32f4", "package_layers:R2=jlcpcb_bga_p05"]
    # the board's own count congesting changes nothing: the search keeps 4 (the change's fallback climb stamped it without the rows)
    calls = _canned(monkeypatch, {2: False, 4: True})
    res = PCBAgent().run(board(), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup.provenance.tool == "design.package_layers"
    # a part's sourced P6 minimum of 4 as the floor (the base commit 22d7895 keeps 4 layers on this board)
    calls = _canned(monkeypatch, {2: True, 4: True})
    ir = board(stm32=False)
    ir.component("R1").min_pcb_layers = _sourced(4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup.layer_count == 4
    # the count below the start leaves nets unrouted and nothing above it can be built: it stands, all or nothing
    calls = _canned(monkeypatch, {4: False})
    res = PCBAgent().run(board(), _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup is None
    assert _escalation(res.notes)[-1] == ("layer escalation: no count above 4 could be built and 4 copper layers leave nets unrouted; the 4-layer "
                                          "result stands")
    res = PCBAgent().run(board(), _ctx(tmp_path, lib, **{pcb_module.ROUTING_KEY: "partial"}))
    assert res.proposals[0].payload.stackup.provenance.note.startswith(
        "layer count 4: no count from the package start 6 up could be built; 4, the highest count below it that could be, leaves nets unrouted")
    # refused rather than skipped (the router refuses every count from the start up, no fab options): the same search from 4
    refused = CompileError("cannot route: fixture refusal")
    calls = _canned(monkeypatch, {**dict.fromkeys((6, 8, 10, 12), refused), 4: True})
    ir = board()
    ir.pcb.manufacturing = ManufacturingConstraints()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [6, 8, 10, 12, 4] and res.proposals[0].payload.stackup.layer_count == 4
    assert _escalation(res.notes)[2:7] == [
        *(f"layer escalation: {n} copper layers: not routed - cannot route: fixture refusal" for n in (6, 8, 10, 12)),
        "layer escalation: no count from the package start 6 up could be built; the search starts at the highest count below it (down to the "
        "package floor of 4) that can be"]


def test_when_no_count_at_or_above_the_floor_can_be_built_the_p6_start_runs_as_before(tmp_path: Path, lib: KicadLibrary,
                                                                                      monkeypatch: pytest.MonkeyPatch):
    """A package floor the fab cannot build (TI AM64x on its 17.2 mm BGA: floor 6; the fab builds 2 and 4) and a part's sourced minimum
    of 4: the search keeps nothing and the board is routed as without it - the P6 start at 4, as the base commit does."""
    name = "BGA-441_17.2x17.2mm_Layout21x21_P0.8mm"
    _bga(lib, name, 21, 0.8, 0.4)
    calls = _canned(monkeypatch, {4: True})
    ir = _plain(tmp_path, lib)
    ir.component("R2").footprint = lib.resolve_footprint(LibraryRef(library="Test", name=name))
    _mpn(ir, "R2", "AM6442BSDGHAALV")
    ir.component("R1").min_pcb_layers = _sourced(4)
    ir.pcb.manufacturing = _fab(2, 4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")]
    assert res.proposals[0].payload.stackup.layer_count == 4 and res.proposals[0].payload.stackup.provenance.tool == ESCALATION_TOOL
    lines = _escalation(res.notes)
    assert lines[1].startswith("layer escalation: package floor 6 copper layers - R2 (MPN 'AM6442BSDGHAALV' starts with 'AM64'; footprint Test:")
    assert ("layer escalation: no count from the package start 6 up could be built; the board is routed as without the package search - below the "
            "package floor of 6") in lines
    assert lines[-1].startswith("layer escalation: 4 layers kept - the first count of 2, 4, 6, 8, 10, 12 at which every net routes (the project "
                                "owner's rule, started at the parts' minimum of 4")


def test_the_boards_own_count_is_checked_against_the_fab_options_on_the_way_down(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Review finding: removal routed and kept the board's own 2 layers on a fab that builds 4 and 6, throwing away a 4-layer result that
    routes every net (the manufacturing capability check then fails the 2-layer board)."""
    calls = _canned(monkeypatch, {2: True, 4: True})
    ir = _mpn(_plain(tmp_path, lib), "R1", "ATmega328P-AU")
    ir.pcb.manufacturing = _fab(4, 6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(4, "4-layer 1.6 mm FR-4")]  # 2 is not offered: never routed
    assert _escalation(res.notes)[-2:] == [
        "layer escalation: fewer layers: 2 copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [4, 6]",
        "layer escalation: 4 layers kept - the fewest copper layers from the package start 4 at which every net routes (no count below 4 could be "
        "routed; the project owner's rule)"]
    assert res.proposals[0].payload.stackup.layer_count == 4


def test_when_no_count_routes_every_net_the_first_routed_stands_all_or_nothing(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, dict.fromkeys((4, 6, 8, 10, 12), False))
    ir = _mpn(_plain(tmp_path, lib), "R1", "STM32F407VET6")
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [4, 6, 8, 10, 12]
    board = res.proposals[0].payload
    assert board.stackup is None and board.tracks == [] and board.zones == []  # no stack without its copper
    assert _escalation(res.notes)[-1] == "layer escalation: no count up to 12 routes every net; the 4-layer result stands"
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{pcb_module.ROUTING_KEY: "partial"}))
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4 and stack.provenance.tool == "design.package_layers"
    assert stack.provenance.note.startswith("layer count 4: the first count routed from the package start; no count up to 12 routes every net, so "
                                            "this 4-layer result stands")


def test_a_board_no_row_touches_is_routed_as_before(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """A row with start 2 and no floor (a shallow 0.8 mm BGA) changes nothing; the P6 start of a sourced minimum stays the P6 climb."""
    _bga(lib, "BGA-64_7x7mm_Layout8x8_P0.8mm", 8, 0.8, 0.4)
    ir = _plain(tmp_path, lib)
    ir.component("R2").footprint = lib.resolve_footprint(LibraryRef(library="Test", name="BGA-64_7x7mm_Layout8x8_P0.8mm"))
    plan = plan_for_ir(ir, lib)
    assert plan.applies is False and [(m.ref, m.row.id) for p in plan.parts for m in p.matches] == [("R2", "nxp_an10778_bga_d4")]
    calls = _canned(monkeypatch, {2: True})
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)] and _escalation(res.notes) == []
    calls = _canned(monkeypatch, {6: True})
    ir.component("R1").min_pcb_layers = _sourced(5)  # P6 starts a minimum of 5 at 6
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup.provenance.tool == ESCALATION_TOOL
    assert not any("package" in n or "rounds up" in n for n in _escalation(res.notes))


def test_parts_no_row_applies_to_leave_the_boards_notes_and_copper_as_they_were(tmp_path: Path, lib: KicadLibrary,
                                                                                monkeypatch: pytest.MonkeyPatch):
    """Review findings: a Yageo CC1206 MLCC matched TI's CC12xx row (start 4); an Infineon part outside a QFN, and a CC1101 + CC1190 pair
    whose board row cannot be read, added 'layer escalation:' notes to a board nothing applies to."""
    calls = _canned(monkeypatch, {2: True})
    today = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))

    def yageo(ir: CircuitIR) -> None:
        _mpn(ir, "C1", "CC1206KRX7R9BB104")
        ir.component("C1").manufacturer = assumption("YAGEO", note="fixture manufacturer")

    def infineon(ir: CircuitIR) -> None:
        ir.component("R1").manufacturer = assumption("Infineon Technologies", note="fixture manufacturer")

    def radio_pair(ir: CircuitIR) -> None:
        _mpn(ir, "R1", "CC1101RGPR")
        _mpn(ir, "R2", "CC1190RGVR")

    for change in (yageo, infineon, radio_pair):
        calls.clear()
        ir = _plain(tmp_path, lib)
        change(ir)
        assert plan_for_ir(ir, lib).applies is False
        res = PCBAgent().run(ir, _ctx(tmp_path, lib))
        assert calls == [(2, None)] and res.notes == today.notes and _payload(res) == _payload(today), change.__name__
    # on a board the plan applies to, the skipped board row is said
    calls = _canned(monkeypatch, {4: True})
    ir = _plain(tmp_path, lib)
    radio_pair(ir)
    _mpn(ir, "C1", "STM32F407VET6")
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert any(n.startswith("layer escalation: board (R1 MPN 'CC1101RGPR' starts with 'CC1101'; R2 MPN 'CC1190RGVR' starts with 'CC1190'): "
                            "ti_cc1101_cc1190em869rd") for n in res.notes)


# --------------------------------------------------------------------------- real routing


def _two_nets(tmp: Path, lib: KicadLibrary) -> CircuitIR:
    """``tests/test_layer_escalation.py``'s board that routes at 2 layers: nets A and GND (kind ground) side by side."""
    ir = board_ir(tmp, lib, [("R1", "SMD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0), ("C1", "SMD1", 3.0, 7.0), ("C2", "SMD1", 9.0, 7.0)],
                  {"A": [("R1", "1"), ("R2", "1")], "GND": [("C1", "1"), ("C2", "1")]}, (12.0, 10.0))
    ir.nets[1].kind = NetKind.GROUND
    return ir


def test_real_routing_starts_at_four_and_ends_on_todays_two_layer_copper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The two-net board routes at 2 layers; an AVR on it starts the search at 4 (it routes there), removes to 2 and ends on exactly
    today's 2-layer copper and payload."""
    seen: list[object] = []
    real = pcb_module.route_board

    def spy(ir, library, params=None, **kw):
        seen.append(kw.get("stack").label if kw.get("stack") is not None else None)
        return real(ir, library, params, **kw)

    monkeypatch.setattr(pcb_module, "route_board", spy)
    lib_a = fixture_library(tmp_path / "a" / "kicad")
    today = PCBAgent().run(_two_nets(tmp_path / "a", lib_a), _ctx(tmp_path / "a", lib_a))
    assert seen == [None] and _escalation(today.notes) == []
    seen.clear()
    lib_b = fixture_library(tmp_path / "b" / "kicad")
    res = PCBAgent().run(_mpn(_two_nets(tmp_path / "b", lib_b), "R1", "ATmega328P-AU"), _ctx(tmp_path / "b", lib_b))
    assert seen == ["4-layer 1.6 mm FR-4", None]
    assert _payload(res) == _payload(today) and res.proposals[0].payload.stackup is None and res.proposals[0].payload.tracks
    assert res.proposals[0].description == today.proposals[0].description and res.proposals[0].rationale == today.proposals[0].rationale
    lines = _escalation(res.notes)
    assert lines[2].startswith("layer escalation: 4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu GND): routing.maze 0.9 routed every net")
    assert lines[3].startswith("layer escalation: fewer layers: 2 copper layers (signals F.Cu, B.Cu; planes none): routing.maze 0.2 routed every net")
    assert lines[4].startswith("layer escalation: 2 layers kept - every net routes on the board's own 2 copper layers")
    assert [n for n in res.notes if not n.startswith("layer escalation")] == today.notes


def test_real_routing_keeps_four_layers_at_an_stm32f4s_floor(tmp_path: Path, lib: KicadLibrary):
    """At one negotiation iteration the GND board leaves A unrouted on 2 layers; an STM32F4 on it starts at 4 (floor 4), where A routes."""
    res = PCBAgent(routing=ONE).run(_mpn(_gnd_board(tmp_path, lib), "P1", "STM32F407VET6"), _ctx(tmp_path, lib))
    board = res.proposals[0].payload
    assert board.stackup.layer_count == 4 and board.stackup.provenance.tool == "design.package_layers"
    assert [(v.net, v.kind) for v in board.vias] == [("GND", "micro")] * 2 and {t.net for t in board.tracks} == {"A", "GND"}
    lines = _escalation(res.notes)
    assert len(lines) == 4 and lines[2].startswith("layer escalation: 4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu +3V3): "
                                                   "routing.maze 0.9 routed every net after 1 iteration(s)")
    assert lines[3].endswith("(the package floor of 4 copper layers allows no fewer; the project owner's rule)")
    # two runs give the same bytes
    again = PCBAgent(routing=ONE).run(_mpn(_gnd_board(tmp_path, lib), "P1", "STM32F407VET6"), _ctx(tmp_path, lib))
    assert _payload(again) == _payload(res)

