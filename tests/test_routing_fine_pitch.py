"""Fine-pitch pads in the router: escape stubs (``routing.maze`` 0.6) and the fan-out room (0.7) (:mod:`ai_eda.tools.routing.maze`, module docstring).

Two pad situations used to end a net before it was searched: a pad whose nearest grid point lies outside its inscribed circle (0.5
refused the whole board) and a pad whose grid cell lies inside its neighbours' keep-out on every layer (the fine-pitch case: at a
0.4 / 0.5 mm pitch every cell of a pad's own centreline is within ``clearance + width/2 + grid/2`` of the pads beside it). What is
checked here, on a synthetic library (the routing fixtures of ``tests/test_routing.py`` plus the footprints below):

* a row of 0.7 x 0.25 mm pads at a 0.5 mm pitch (``ROW5``: one side of a QFN, its exposed pad net-less) fans out: every pad gets a stub
  no wider than the pad, from its centre out of its edge to a grid cell on one line in front of the row, the cells as far apart as the
  stubs' claims need; every net routes, every track and via is stamped 0.6 with the escape knobs and the escaped pads in its
  provenance, each stub segment names its pad, width and points (read back by ``escape_entry``), and ``pcb.routing.*`` PASS at the
  router's clearance - also with the row 0.125 mm off the grid (pads 2 and 4 off-grid, 1 / 3 / 5 fenced);
* a lone 0.2 mm pad off the 0.25 mm grid gets a straight stub to the nearest cell; a pad of a net with fewer than two pads gets none,
  but its board is 0.6's (0.5 refused it);
* refusals are the net's, named: an off-grid pad in a closed cage (no cell has a way out of the footprint's escape area) leaves its net
  unrouted with the old sentence and why no escape exists, while the other net routes; a fenced pad no escape reaches keeps 0.2's
  copper and stamp, its reason extended; a coupled pair's off-grid pad still refuses the board (0.3); a refusal names what a reserved
  cell is - a terminal cell, a cell on an escape's way out;
* doomed nets: a signal net with a caged pad whose other pad took a lane of a narrowed fan is doomed and the escape pass runs again
  without it - the pads its escape had fenced escape and their nets route, its escape is listed as withdrawn;
* plane nets: a fenced pad of the row on the GND plane gets a stub to its own via inside the plane zone, the other pads fan around it;
  a pad whose 0.4 axis walk finds no via site (posts on all four axes) gets a diagonal stub to a via; a plane net one of whose pads no
  escape reaches is unrouted and leaves no trace (the static maps equal those of the same board without plane nets); with two plane
  nets the second one's via walk no longer meets the doomed first one's via - the board is 0.4's copper and stamp;
* the fan-out room (routing.maze 0.7): a neighbour's ground pad whose 0.4 via walk lands in the row's escape area leaves a pad of the
  row without an escape under 0.6; the static phase runs again with the row's fan-out room (its pads' box grown by ``fanout_margin``,
  2.2 mm at the fine rules) kept clear of other footprints' plane vias and the escapes' ways out running to the room's edge - every pad
  escapes, every net routes, stamped 0.7 with the room named (``fanout_room:...``); a board where the room frees no more pads keeps
  0.6's copper, stats and refusal byte for byte;
* a net rule's fence beside escapes (``ROW3V``: one side of the transceiver's QFN-12, off the grid as the floorplan put it): another net's
  bent escape stub counts by its copper, never by its box, so a rule net's escaped terminal (the transceivers' PA_PD) stays usable and a
  fenced one's refusal names what fences it; an escape whose copper would put another net's usable rule terminal inside that net's
  fence is refused (the result does not depend on the escape order); at the board's width the fence is the owner map's keep-out;
* a board with more than ``LARGE_BOARD_NETS`` nets to route negotiates up to ``LARGE_BOARD_ITERATIONS`` iterations, recorded in the
  params entry with its net count; every other board keeps 40 and the entry character for character;
* a fab ``min_track_width_mm`` above ``escape_min_width_mm`` is the narrowest stub and recorded as a raise, only on a board with escapes;
  a stub keeps the pad's narrow side when that fits (a 0.254 mm pad, not a multiple of the width step);
* ``si.impedance`` names an escape segment of a controlled net like a neck-down, only the segment the provenance records;
* a board that needs none of it is 0.2's: no escape stats, the params entry character for character;
* with the packed KiCad 10.0.6 libraries (``KICAD10_SYMBOL_DIR``; skipped without them): the real ``QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm
  _ThermalVias`` (the transceiver's PA, off the grid as the floorplan puts it) and ``DFN-14-1EP_3x3mm_P0.4mm_EP1.78x2.35mm`` (the
  MAX9814) with every pin wired to its own 0603 resistor: every pad escapes, every net routes at the fine rules, clearance clean.

Nothing here claims DRC: the router's clearances are its parameters, and the checks measure the IR geometry it promised.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ai_eda.errors import CompileError
from ai_eda.ir import (
    BoardOutline,
    BoardSide,
    CircuitIR,
    Component,
    LibraryRef,
    ManufacturingConstraints,
    Net,
    NetKind,
    PCBDesign,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    ProjectMeta,
    Track,
    assumption,
)
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing import maze
from ai_eda.tools.routing.maze import (
    ESCAPE_ENTRY_PREFIX,
    FINE_RULES,
    LARGE_BOARD_ITERATIONS,
    LARGE_BOARD_NETS,
    ROUTER_ESCAPE_VERSION,
    ROUTER_FANOUT_VERSION,
    ROUTER_KEEPOUT_VERSION,
    ROUTER_VERSION,
    NetRule,
    RoutingParams,
    _Board,
    effective_params,
    escape_entry,
    fanout_margin,
    route_board,
    routed_net_count,
)
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import _Board as LayoutBoard
from ai_eda.validation.layout import clearance_rows
from ai_eda.validation.si import _neckdown
from tests.test_routing import DEFAULT_ENTRY, NET_P, board_ir, fixture_library
from tests.test_si_checks import stacked

#: one side of a QFN: five 0.7 x 0.25 mm pads at a 0.5 mm pitch along y, and a net-less exposed pad east of them (so the row leaves west)
ROW5 = (
    '(pad "6" smd rect (at 1.3 0) (size 1.2 2.4) (layers "F.Cu" "F.Mask" "F.Paste"))\n  '
    + "\n  ".join(f'(pad "{i + 1}" smd rect (at 0 {y}) (size 0.7 0.25) (layers "F.Cu" "F.Mask" "F.Paste"))' for i, y in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)))
)
#: a 0.2 mm pad in a closed cage of net-less bars 0.7 mm from its centre (no via fits inside, no cell there reaches out)
TCAGE = (
    '(pad "1" smd rect (at 0 0) (size 0.2 0.2) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
    '  (pad "" smd rect (at -0.9 0) (size 0.4 2.2) (layers "F.Cu" "F.Mask"))\n  (pad "" smd rect (at 0.9 0) (size 0.4 2.2) (layers "F.Cu" "F.Mask"))\n'
    '  (pad "" smd rect (at 0 -0.9) (size 2.2 0.4) (layers "F.Cu" "F.Mask"))\n  (pad "" smd rect (at 0 0.9) (size 2.2 0.4) (layers "F.Cu" "F.Mask"))'
)
#: a 0.7 x 0.25 mm pad between two net-less neighbours 0.4 mm away, net-less walls 0.95 mm in front of both of its ends
COMB = (
    '(pad "1" smd rect (at 0 0) (size 0.7 0.25) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
    '  (pad "" smd rect (at 0 -0.4) (size 0.7 0.25) (layers "F.Cu" "F.Mask"))\n  (pad "" smd rect (at 0 0.4) (size 0.7 0.25) (layers "F.Cu" "F.Mask"))\n'
    '  (pad "" smd rect (at -0.95 0) (size 0.3 2.4) (layers "F.Cu" "F.Mask"))\n  (pad "" smd rect (at 0.95 0) (size 0.3 2.4) (layers "F.Cu" "F.Mask"))'
)
#: a 1 mm SMD pad with net-less posts on all four grid axes 1.6 mm away: the 0.4 axis walk finds no via site, a diagonal one exists
POST = (
    '(pad "1" smd rect (at 0 0) (size 1.0 1.0) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
    + "\n".join(f'  (pad "" smd rect (at {x} {y}) (size 0.5 0.5) (layers "F.Cu" "F.Mask"))' for x, y in ((1.6, 0), (-1.6, 0), (0, 1.6), (0, -1.6)))
)
#: one side of the transceiver's QFN-12 (U901.10 / .11 / .12): three 0.25 x 0.825 mm pads at a 0.5 mm pitch along x over a net-less
#: exposed pad (the row leaves north)
ROW3V = (
    "\n  ".join(f'(pad "{i + 1}" smd rect (at {x} 0) (size 0.25 0.825) (layers "F.Cu" "F.Mask" "F.Paste"))' for i, x in enumerate((0.5, 0.0, -0.5)))
    + '\n  (pad "4" smd rect (at 0 1.3) (size 1.6 1.2) (layers "F.Cu" "F.Mask" "F.Paste"))'
)
FINE = RoutingParams(**FINE_RULES)
PLANE = [(0.5, 0.5), (19.5, 0.5), (19.5, 17.5), (0.5, 17.5)]


def fine_library(root: Path) -> KicadLibrary:
    lib = fixture_library(root)
    pretty = root / "footprints" / "Test.pretty"
    for name, pads in (("ROW5", ROW5), ("TCAGE", TCAGE), ("COMB", COMB), ("POST", POST), ("ROW3V", ROW3V)):
        (pretty / f"{name}.kicad_mod").write_text(f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n  {pads})\n', encoding="utf-8")
    return lib


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fine_library(tmp_path / "kicad")


def row_board(tmp_path: Path, lib: KicadLibrary, x: float = 12.0, y: float = 8.0) -> CircuitIR:
    """``ROW5`` at (x, y), each pad on its own net ``S<i>`` to a through-hole pad on a column at x = 3 (20 x 18 mm)."""
    parts = [("U1", "ROW5", x, y)] + [(f"R{i}", "PAD1", 3.0, 2.0 + 3.0 * i) for i in range(1, 6)]
    return board_ir(tmp_path, lib, parts, {f"S{i}": [("U1", str(i)), (f"R{i}", "1")] for i in range(1, 6)}, (20.0, 18.0))


def _routing_checks(ir: CircuitIR, tracks, vias, lib: KicadLibrary, tmp_path: Path, limit: float) -> dict[str, str]:
    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(tracks), list(vias)
    x.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(limit, note="the router's own clearance as the limit"))
    results = default_registry.get("pcb.routing").validate(x, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    return {r.check_id: r.status.value for r in results}


def _stub_segments(r, pad: str) -> list[Track]:
    return [t for t in r.tracks if any(e.startswith(f"{ESCAPE_ENTRY_PREFIX}{pad}:") for e in t.provenance.derived_from)]


# --------------------------------------------------------------------------- a fine-pitch row fans out


@pytest.mark.parametrize("x", [12.0, 12.125])
def test_a_fine_pitch_row_fans_out_and_every_net_routes_legally(tmp_path: Path, lib: KicadLibrary, x: float):
    ir = row_board(tmp_path, lib, x)
    board = _Board(ir, lib, FINE)
    before = {t.label: t for ts in board.terminals.values() for t in ts}
    r = route_board(ir, lib, FINE)
    assert r.unrouted == {} and r.version == ROUTER_ESCAPE_VERSION == "0.6" and r.stats["routed_nets"] == 5
    rows = {e["pad"]: e for e in r.stats["escapes"]}
    assert sorted(rows) == [f"U1.{i}" for i in range(1, 6)] and r.stats["escape_refused"] == {}
    whys = {pad: row["why"] for pad, row in rows.items()}
    if x == 12.0:
        assert set(whys.values()) == {"fenced"}  # every pad's own cell is within 0.425 mm of a neighbour's box (0.375 mm away)
    else:
        assert whys["U1.2"] == whys["U1.4"] == "off-grid" and {whys["U1.1"], whys["U1.3"], whys["U1.5"]} == {"fenced"}
    cells = sorted(tuple(row["cell"]) for row in rows.values())
    assert len({c[0] for c in cells}) == 1  # a fan: the cells on one line in front of the row ...
    assert [round(b[1] - a[1], 6) for a, b in zip(cells, cells[1:])] == [0.6] * 4  # ... as far apart as the stubs' claims need
    for pad, row in rows.items():
        geom = before[pad].pad
        assert row["kind"] == "stub" and 0.1 <= row["width_mm"] <= 0.25  # never wider than the pad
        assert row["points"][0] == [geom.cx, geom.cy] and row["cell"][0] < geom.cx - geom.hw  # from the pad centre, out of its west edge
        segs = _stub_segments(r, pad)
        assert segs and all(t.width_mm == row["width_mm"] and t.layer == "F.Cu" for t in segs)
        label, width, pts = escape_entry(segs[0])
        assert (label, width, [list(p) for p in pts]) == (pad, row["width_mm"], row["points"])
        assert "escape stub of pad " + pad in segs[0].provenance.note and "routing.maze 0.6" in segs[0].provenance.note
    for item in [*r.tracks, *r.vias]:
        assert item.provenance.tool_version == "0.6"
        params = next(e for e in item.provenance.derived_from if e.startswith("params:"))
        assert params.endswith(",escape_reach=1.5,escape_min_width=0.1,escape_width_step=0.01")
        assert f"escapes:{','.join(e['pad'] for e in r.stats['escapes'])};model=centre+edge+cell" in item.provenance.derived_from
    checks = _routing_checks(ir, r.tracks, r.vias, lib, tmp_path, FINE.clearance_mm)
    assert checks == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}, checks
    again = route_board(ir, lib, FINE)
    assert [design_data(t) for t in again.tracks] == [design_data(t) for t in r.tracks] and again.stats == r.stats  # deterministic


def test_the_old_refusal_of_an_off_grid_pad_is_an_escape_stub_now(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("T1", "TINY", 9.12, 3.0)], {"N": [("R1", "1"), ("T1", "1")]}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.version == "0.6"
    (row,) = r.stats["escapes"]
    assert (row["pad"], row["kind"], row["why"], row["width_mm"], row["points"], row["cell"]) == ("T1.1", "stub", "off-grid", 0.2, [[9.12, 3.0], [9.25, 3.0]], [9.25, 3.0])
    (stub,) = _stub_segments(r, "T1.1")
    assert (stub.start, stub.end, stub.width_mm) == ((9.25, 3.0), (9.12, 3.0), 0.2)
    assert _routing_checks(ir, r.tracks, r.vias, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
    # a pad of a net with fewer than two pads gets no escape; the board is still 0.6's, since 0.5 refused it
    single = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 6.0, 3.0), ("T1", "TINY", 9.12, 3.0)],
                      {"N": [("R1", "1"), ("R2", "1")], "L": [("T1", "1")]}, (12.0, 6.0))
    lone = route_board(single, lib)
    assert lone.unrouted == {} and lone.version == "0.6" and "escapes" not in lone.stats and lone.stats["skipped_nets"] == ["L"]


# --------------------------------------------------------------------------- refusals are the net's


def test_a_pad_no_escape_reaches_leaves_its_net_unrouted_and_names_it(tmp_path: Path, lib: KicadLibrary):
    parts = [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 2.0, 8.0), ("R3", "PAD1", 12.0, 8.0)]
    nets = {"N": [("R1", "1"), ("T1", "1")], "M": [("R2", "1"), ("R3", "1")]}
    caged = board_ir(tmp_path, lib, [*parts, ("T1", "TCAGE", 9.12, 4.0)], nets, (14.0, 10.0))
    r = route_board(caged, lib)
    assert list(r.unrouted) == ["N"] and r.version == "0.6" and [t.net for t in r.tracks] == ["M"]
    why = r.unrouted["N"]
    assert why.startswith("pad T1.1 (0.2 x 0.2 mm) is too small for the 0.25 mm routing grid: the nearest grid point is 0.1200 mm from its centre")
    assert "; no escape stub within 1.5 mm in front of its edge: " in why and "no way out of the footprint's escape area" in why
    assert r.stats["escape_refused"] == {"T1.1": why} and "escapes:;refused=T1.1;model=centre+edge+cell" in r.tracks[0].provenance.derived_from
    # a fenced pad no escape reaches: its net was unrouted by 0.2 too, so the board keeps 0.2's copper and stamp, the reason extended
    combed = board_ir(tmp_path, lib, [*parts, ("T1", "COMB", 9.0, 4.0)], nets, (14.0, 10.0))
    r = route_board(combed, lib)
    assert list(r.unrouted) == ["N"] and r.version == ROUTER_VERSION == "0.2" and r.stats["escapes"] == []
    assert r.unrouted["N"].startswith("T1.1 terminal cell (9, 4) is inside a keep-out on every copper layer of the pad (T1.(unnumbered) within")
    assert "; no escape stub within 1.5 mm in front of its edge: the nearest candidate cell" in r.unrouted["N"]
    assert r.stats["escape_refused"] == {"T1.1": r.unrouted["N"]}
    assert all(e.startswith(("net:", "placement:", "params:")) for t in r.tracks for e in t.provenance.derived_from)


def test_a_coupled_pairs_off_grid_pad_still_refuses_the_board(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("T1", "TINY", 12.12, 3.0), ("R2", "PAD1", 3.0, 6.0), ("R3", "PAD1", 12.0, 6.0)],
                  {"A": [("R1", "1"), ("T1", "1")], "B": [("R2", "1"), ("R3", "1")]}, (16.0, 10.0))
    rules = {"A": NetRule(width_mm=0.3, pair_partner="B", pair_spacing_mm=0.3), "B": NetRule(width_mm=0.3, pair_partner="A", pair_spacing_mm=0.3)}
    with pytest.raises(CompileError, match=r"pad T1.1 \(0.2 x 0.2 mm\) is too small for the 0.25 mm routing grid"):
        route_board(ir, lib, rules=rules)


def narrowed_row(tmp_path: Path, lib: KicadLibrary, d: float, doomed: bool = False) -> CircuitIR:
    """``row_board`` with net-less bars 2d apart across the front of the row (the fan has room for fewer lanes); ``doomed``: net S2 also
    holds an off-grid pad in a closed cage (``TCAGE`` at x = 16.1, 0.1 mm off the 0.2 mm grid), which no escape reaches."""
    parts = [("U1", "ROW5", 12.0, 8.0), ("H1", "HBAR", 10.0, 8.0 - d), ("H2", "HBAR", 10.0, 8.0 + d)]
    parts += [(f"R{i}", "PAD1", 3.0, 2.0 + 3.0 * i) for i in range(1, 6)]
    nets = {f"S{i}": [("U1", str(i)), (f"R{i}", "1")] for i in range(1, 6)}
    if doomed:
        parts.append(("T1", "TCAGE", 16.1, 15.0))
        nets["S2"].append(("T1", "1"))
    return board_ir(tmp_path, lib, parts, nets, (20.0, 18.0))


def test_a_refusal_names_what_the_reserved_cell_is(tmp_path: Path, lib: KicadLibrary):
    ir = narrowed_row(tmp_path, lib, 1.1)
    board = _Board(ir, lib, effective_params(ir, FINE)[0])
    why = board.escape_refused["U1.3"]
    # (10.8, 8) is on the way out of U1.2's escape, reserved for S2 - not one of S2's terminal cells
    assert why.endswith("the nearest candidate cell (10.8, 8): its clearance would take a cell on the way out reserved for an escape of net S2")
    assert "terminal cell of net" not in why.split("; ", 1)[1]
    s2 = {board.pos(t.cell) for t in board.terminals["S2"]}
    assert (10.8, 8.0) not in s2 and any(e["pad"] == "U1.2" and e["net"] == "S2" for e in board.escapes)
    # a terminal cell is still called one
    ir = narrowed_row(tmp_path, lib, 1.7)
    board = _Board(ir, lib, effective_params(ir, FINE)[0])
    assert board.escape_refused["U1.4"].endswith("the nearest candidate cell (11.2, 8.4): its clearance would take the terminal cell of net S5")


def test_a_doomed_signal_nets_escape_is_withdrawn_and_the_pads_it_fenced_escape(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """S2's caged pad T1.1 has no escape, so S2 can never route; a single pass still gave its U1.2 a lane of the narrowed fan, and that
    escape fenced U1.3 (S3) and U1.4 (S4). The second pass leaves S2 out: every other pad escapes and every other net routes."""
    ir = narrowed_row(tmp_path, lib, 1.7, doomed=True)
    p = effective_params(ir, FINE)[0]
    board = _Board(ir, lib, p)
    assert board.static_passes == 2 and list(board.doomed) == ["S2"] and board.doomed["S2"].kind == "signal"
    assert sorted(e["pad"] for e in board.escapes) == ["U1.1", "U1.3", "U1.4", "U1.5"] and set(board.escape_refused) == {"T1.1", "U1.2"}
    assert board.escape_refused["T1.1"].startswith("pad T1.1 (0.2 x 0.2 mm) is too small for the 0.2 mm routing grid")
    assert board.escape_refused["U1.2"].endswith("; its escape is withdrawn: net S2 stays unrouted, since no escape reaches T1.1")
    assert not any(t.escape is not None for t in board.terminals["S2"])
    r = route_board(ir, lib, FINE)
    assert list(r.unrouted) == ["S2"] and r.unrouted["S2"] == board.escape_refused["T1.1"] and r.stats["routed_nets"] == 4
    assert all("escapes:U1.1,U1.3,U1.4,U1.5;refused=T1.1,U1.2;model=centre+edge+cell" in t.provenance.derived_from for t in r.tracks)
    assert not [t for t in r.tracks if t.net == "S2"]
    # a single pass (the doom switched off) keeps S2's escape: U1.3 is refused beside it, and S3 / S4 are lost with S2
    monkeypatch.setattr(maze._Board, "_next_doomed", lambda self: None)
    single = _Board(ir, lib, p)
    assert single.static_passes == 1 and set(single.escape_refused) == {"T1.1", "U1.3", "U1.4"}
    assert single.escape_refused["U1.3"].endswith("its clearance would take the terminal cell of net S2")
    assert sorted(route_board(ir, lib, FINE).unrouted) == ["S2", "S3", "S4"]


# --------------------------------------------------------------------------- plane nets


def test_a_fenced_plane_pad_in_the_row_gets_a_stub_to_its_own_via_and_the_rest_fans_around_it(tmp_path: Path, lib: KicadLibrary):
    parts = [("U1", "ROW5", 12.0, 8.0)] + [(f"R{i}", "PAD1", 3.0, 2.0 + 3.0 * i) for i in (1, 2, 4, 5)] + [("R3", "PAD1", 16.0, 14.0)]
    nets = {f"S{i}": [("U1", str(i)), (f"R{i}", "1")] for i in (1, 2, 4, 5)}
    nets["GND"] = [("U1", "3"), ("R3", "1")]
    ir = stacked(board_ir(tmp_path, lib, parts, nets, (20.0, 18.0)), 4)
    ir.net("GND").kind = NetKind.GROUND
    r = route_board(ir, lib, FINE, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]})
    assert r.unrouted == {} and r.version == "0.6"
    rows = {e["pad"]: e for e in r.stats["escapes"]}
    assert sorted(rows) == ["U1.1", "U1.2", "U1.3", "U1.4", "U1.5"] and rows["U1.3"]["kind"] == "via" and rows["U1.3"]["why"] == "fenced"
    plane = r.stats["plane_nets"]["GND"]
    assert plane["pads"] == ["U1.3"] and plane["through_hole"] == ["R3.1"] and plane["vias"] == 1
    (via,) = [v for v in r.vias if v.net == "GND"]
    assert [via.x_mm, via.y_mm] == rows["U1.3"]["cell"] and rows["U1.3"]["points"][-1] == rows["U1.3"]["cell"]
    assert all(t.width_mm <= 0.25 for t in _stub_segments(r, "U1.3"))
    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(r.tracks), list(r.vias)
    _, bad = clearance_rows(LayoutBoard(x, lib), FINE.clearance_mm)
    assert bad == []


def test_a_plane_pad_whose_axis_walk_finds_no_via_site_gets_a_diagonal_stub_to_one(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("C1", "POST", 6.0, 6.0), ("C2", "SMD1", 14.0, 6.0), ("R5", "PAD1", 4.0, 14.0), ("R6", "PAD1", 16.0, 14.0)],
                  {"GND": [("C1", "1"), ("C2", "1")], "S": [("R5", "1"), ("R6", "1")]}, (20.0, 18.0))
    ir.net("GND").kind = NetKind.GROUND
    ir = stacked(ir, 4)
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]})
    assert r.unrouted == {} and r.version == "0.6" and r.stats["plane_nets"]["GND"]["pads"] == ["C1.1", "C2.1"]
    (row,) = r.stats["escapes"]
    assert (row["pad"], row["kind"], row["why"]) == ("C1.1", "via", "no via site")
    (cx, cy), (vx, vy) = row["points"][0], row["cell"]
    assert (cx, cy) == (6.0, 6.0) and vx != cx and vy != cy  # off every axis of the pad: the walk could not find it
    assert any((v.x_mm, v.y_mm) == (vx, vy) for v in r.vias if v.net == "GND")


def test_a_plane_net_with_a_pad_no_escape_reaches_is_unrouted_and_leaves_no_trace_on_the_static_maps(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("C1", "SMD1", 6.0, 6.0), ("T1", "TCAGE", 14.12, 6.0), ("R5", "PAD1", 4.0, 14.0), ("R6", "PAD1", 16.0, 14.0)],
                  {"GND": [("C1", "1"), ("T1", "1")], "S": [("R5", "1"), ("R6", "1")]}, (20.0, 18.0))
    ir.net("GND").kind = NetKind.GROUND
    ir = stacked(ir, 4)
    p, _ = effective_params(ir, RoutingParams())
    with_plane = _Board(ir, lib, p, None, True, plane={"GND": p.track_width_mm}, plane_areas={"GND": [PLANE, PLANE]})
    without = _Board(ir, lib, p, None, True)
    assert list(with_plane.plane_problems) == ["GND"] and with_plane.escape_ran  # T1.1 is off the grid: 0.5 refused this board
    assert "the via would come" in with_plane.plane_problems["GND"]  # no via fits inside the cage
    # C1.1's via and stub were claimed while T1.1 waited; GND is doomed and the second pass leaves it out: the maps are those of the
    # board without plane nets
    assert with_plane.static_passes == 2 and list(with_plane.doomed) == ["GND"] and with_plane.escapes == []
    assert with_plane.owner == without.owner and with_plane.via_pad_ok == without.via_pad_ok
    assert [g for g in with_plane.pads if g.net == with_plane.net_index["GND"] and g.kind != "pad"] == []
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]})
    assert list(r.unrouted) == ["GND"] and r.unrouted["GND"].startswith("plane net: pad T1.1 (0.2 x 0.2 mm) is too small")
    assert not [t for t in r.tracks if t.net == "GND"] and not [v for v in r.vias if v.net == "GND"] and [t for t in r.tracks if t.net == "S"]


def test_a_doomed_plane_nets_via_is_gone_before_the_next_plane_net_walks(tmp_path: Path, lib: KicadLibrary):
    """GND waits for an escape of K1.1 (fenced between its neighbours, walls at both ends: none exists) while its C1.1 via is claimed;
    VCC walks after it (name order) from V1.1, caged on three sides with a net-less B.Cu part in front, and its only via site is beside
    that GND via. The pass that dooms GND is thrown away and VCC walks again without it: VCC gets the site, only GND is unrouted, and the
    board is 0.4's (0.4 undid GND before VCC walked) - its copper, its stamp, no escape knob - with GND's reason extended."""
    parts = [
        ("C1", "SMD1", 11.5, 9.0), ("K1", "COMB", 16.0, 15.0), ("V1", "SMD1", 6.0, 9.0), ("B1", "SMD1", 7.5, 9.0),
        ("W1", "VBAR", 4.25, 9.0), ("W2", "VBAR", 12.75, 9.0), ("H1", "HBAR", 6.0, 7.25), ("H2", "HBAR", 10.0, 7.25), ("H3", "HBAR", 14.0, 7.25),
        ("H4", "HBAR", 6.0, 10.75), ("H5", "HBAR", 10.0, 10.75), ("H6", "HBAR", 14.0, 10.75), ("R1", "PAD1", 2.0, 2.0), ("R2", "PAD1", 18.0, 2.0),
    ]
    nets = {"GND": [("C1", "1"), ("K1", "1")], "VCC": [("V1", "1")], "S": [("R1", "1"), ("R2", "1")]}
    ir = board_ir(tmp_path, lib, parts, nets, (20.0, 18.0), sides={"B1": BoardSide.BOTTOM})
    ir.net("GND").kind = NetKind.GROUND
    ir.net("VCC").kind = NetKind.POWER
    ir = stacked(ir, 4)
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE)], "VCC": [("In2.Cu", PLANE)]})
    assert list(r.unrouted) == ["GND"] and r.version == ROUTER_KEEPOUT_VERSION == "0.4"
    assert [(v.net, v.x_mm, v.y_mm) for v in r.vias] == [("VCC", 9.5, 9.0)] and r.stats["plane_nets"]["VCC"]["pads"] == ["V1.1"]
    assert r.unrouted["GND"].startswith("plane net: pad K1.1: its terminal cell is inside a keep-out on F.Cu")
    assert "; no escape stub within 1.5 mm in front of its edge: " in r.unrouted["GND"]
    assert r.stats["escapes"] == [] and list(r.stats["escape_refused"]) == ["K1.1"]
    for item in [*r.tracks, *r.vias]:
        assert item.provenance.tool_version == "0.4" and not any(e.startswith("escapes:") for e in item.provenance.derived_from)
        assert "escape_reach" not in next(e for e in item.provenance.derived_from if e.startswith("params:"))
    p, _ = effective_params(ir, RoutingParams())
    both = _Board(ir, lib, p, None, True, plane={"GND": p.track_width_mm, "VCC": p.track_width_mm}, plane_areas={"GND": [PLANE], "VCC": [PLANE]})
    vcc_only = _Board(ir, lib, p, None, True, plane={"VCC": p.track_width_mm}, plane_areas={"VCC": [PLANE]})
    assert both.static_passes == 2 and list(both.doomed) == ["GND"] and not both.escape_ran
    assert both.plane_links["VCC"] == vcc_only.plane_links["VCC"] and both.owner == vcc_only.owner and both.via_pad_ok == vcc_only.via_pad_ok


# --------------------------------------------------------------------------- the fan-out room (routing.maze 0.7)


def neighbour_via_board(tmp_path: Path, lib: KicadLibrary, gx: float, gy: float) -> CircuitIR:
    """``row_board`` on a 4-layer GND-plane board plus a GND pad ``C1`` (1 mm SMD) west of the row at (gx, gy): 0.4's via walk puts its
    via east of it first - into the row's escape area when the pad is close enough."""
    parts = [("U1", "ROW5", 12.0, 8.0)] + [(f"R{i}", "PAD1", 3.0, 2.0 + 3.0 * i) for i in range(1, 6)] + [("C1", "SMD1", gx, gy), ("C2", "SMD1", 16.0, 15.0)]
    nets = {f"S{i}": [("U1", str(i)), (f"R{i}", "1")] for i in range(1, 6)}
    nets["GND"] = [("C1", "1"), ("C2", "1")]
    ir = stacked(board_ir(tmp_path, lib, parts, nets, (20.0, 18.0)), 4)
    ir.net("GND").kind = NetKind.GROUND
    return ir


GND_PLANES = {"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]}


def test_a_neighbours_plane_via_is_kept_out_of_the_fan_out_room_when_0_6_leaves_a_pad(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """0.6 walks C1.1's via to (10, 8), inside the row's escape area, and U1.4 then finds no escape; 0.7 runs the static phase again with
    U1's fan-out room (its pads' box - the net-less exposed pad included - grown by fanout_margin, 2.2 mm at the fine rules) kept clear
    of other footprints' plane vias: C1.1's walk meets the room in every direction, so it waits for an escape and gets a stub to a via
    outside the room; every pad of the row escapes and every net routes, stamped 0.7 with the room named in every item's provenance."""
    ir = neighbour_via_board(tmp_path, lib, 9.2, 8.0)
    p = effective_params(ir, FINE)[0]
    kw = {"plane": {"GND": p.track_width_mm}, "plane_areas": {"GND": [PLANE, PLANE]}}
    assert fanout_margin(FINE) == 2.2 and fanout_margin(RoutingParams()) == 2.4  # 1.5 + grid + the larger of the keep-out and the via's reach
    # 0.6 alone (the room switched off, so the second run changes nothing and is not kept)
    monkeypatch.setattr(maze._Board, "_in_fanout", lambda self, ref, box: None)
    old = _Board(ir, lib, p, None, True, **kw)
    assert not old.rules07 and list(old.escape_refused) == ["U1.4"] and old.fanout_rooms == {}
    assert old.escape_refused["U1.4"].endswith("its clearance would take a cell on the way out reserved for an escape of net S3")
    assert [(t.label, old.pos(k)) for t, _l, k in old.plane_links["GND"]][0] == ("C1.1", (10.0, 8.0))
    monkeypatch.undo()
    board = _Board(ir, lib, p, None, True, **kw)
    assert board.rules07 and board.escape_refused == {} and board.fanout_kept == ["C1.1"] and board.static_passes == 1
    room = board.fanout_rooms["U1"]
    assert list(board.fanout_rooms) == ["U1"] and room == pytest.approx((11.65 - 2.2, 6.8 - 2.2, 13.9 + 2.2, 9.2 + 2.2))  # ROW5's pads, EP included
    (via_cell,) = [k for t, _l, k in board.plane_links["GND"] if t.label == "C1.1"]
    vx, vy = board.pos(via_cell)
    r_claim = FINE.via_diameter_mm / 2.0 + board.pad_radius  # the via's claim on the owner map
    assert (vx, vy) == (8.6, 8.8) and vx + r_claim <= room[0] + 1e-9  # west of the room, its claim outside it
    (row,) = [e for e in board.escapes if e["pad"] == "C1.1"]
    assert (row["kind"], row["why"], row["cell"]) == ("via", "no via site", [8.6, 8.8])
    r = route_board(ir, lib, FINE, inner_layers=True, plane_nets=GND_PLANES)
    assert r.unrouted == {} and r.version == ROUTER_FANOUT_VERSION == "0.7"
    assert sorted(e["pad"] for e in r.stats["escapes"]) == ["C1.1", *(f"U1.{i}" for i in range(1, 6))]
    assert r.stats["fanout_room"] == {"rooms": {"U1": [9.45, 4.6, 16.1, 11.4]}, "kept_out": ["C1.1"]}
    for item in [*r.tracks, *r.vias]:
        assert item.provenance.tool_version == "0.7" and "fanout_room:U1;margin=2.2;kept_out=C1.1;model=pads_box+margin" in item.provenance.derived_from
        assert next(e for e in item.provenance.derived_from if e.startswith("params:")).endswith(",escape_reach=1.5,escape_min_width=0.1,escape_width_step=0.01")
    assert "routing.maze 0.7: " in r.tracks[0].provenance.note and "fan-out rooms of U1 (their pads' box grown by 2.2 mm" in r.tracks[0].provenance.note
    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(r.tracks), list(r.vias)
    _, bad = clearance_rows(LayoutBoard(x, lib), FINE.clearance_mm)
    assert bad == []
    again = route_board(ir, lib, FINE, inner_layers=True, plane_nets=GND_PLANES)
    assert [design_data(t) for t in again.tracks] == [design_data(t) for t in r.tracks] and again.stats == r.stats  # deterministic


def test_a_board_the_fan_out_room_does_not_help_keeps_0_6_byte_for_byte(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """With C1 at (9, 8) U1.3 has no escape under 0.6, and the second run with the room leaves it without one too: the 0.6 result is
    restored - its copper, stats, stamp and refusal, exactly as a router without 0.7 gives them."""
    ir = neighbour_via_board(tmp_path, lib, 9.0, 8.0)
    r = route_board(ir, lib, FINE, inner_layers=True, plane_nets=GND_PLANES)
    assert list(r.unrouted) == ["S3"] and r.version == "0.6" and "fanout_room" not in r.stats
    assert not any(e.startswith("fanout_room:") for t in r.tracks for e in t.provenance.derived_from)
    monkeypatch.setattr(maze._Board, "_in_fanout", lambda self, ref, box: None)
    monkeypatch.setattr(maze._Board, "_room_of", lambda self, ref, margin: self._area_of(ref))
    plain = route_board(ir, lib, FINE, inner_layers=True, plane_nets=GND_PLANES)
    assert [design_data(t) for t in plain.tracks] == [design_data(t) for t in r.tracks] and [design_data(v) for v in plain.vias] == [design_data(v) for v in r.vias]
    assert plain.stats == r.stats and plain.unrouted == r.unrouted


def test_a_large_board_negotiates_longer_and_says_so_every_other_board_keeps_40(tmp_path: Path, lib: KicadLibrary):
    small = row_board(tmp_path, lib)
    assert routed_net_count(small) == 5 and RoutingParams.for_board(small, lib).max_iterations == 40 and RoutingParams.for_board(small, lib).board_nets is None
    parts = [(f"R{i}", "PAD1", 1.0 + (i % 20) * 1.0, 1.0 + (i // 20) * 3.0) for i in range(2 * (LARGE_BOARD_NETS + 1))]
    nets = {f"N{i}": [(f"R{2 * i}", "1"), (f"R{2 * i + 1}", "1")] for i in range(LARGE_BOARD_NETS + 1)}
    big = board_ir(tmp_path, lib, parts, nets, (22.0, 34.0))
    params = RoutingParams.for_board(big, lib)
    assert routed_net_count(big) == LARGE_BOARD_NETS + 1
    assert params.max_iterations == LARGE_BOARD_ITERATIONS and params.board_nets == LARGE_BOARD_NETS + 1 and params.rules is None
    assert params.derived_from_entry() == DEFAULT_ENTRY.replace("max_iterations=40", f"max_iterations={LARGE_BOARD_ITERATIONS}") + f",large_board_nets={LARGE_BOARD_NETS + 1}"
    exactly = board_ir(tmp_path, lib, parts[: 2 * LARGE_BOARD_NETS], {k: v for k, v in list(nets.items())[:LARGE_BOARD_NETS]}, (22.0, 34.0))
    assert RoutingParams.for_board(exactly, lib) == RoutingParams()  # exactly LARGE_BOARD_NETS nets: not a large board
    with pytest.raises(CompileError, match="board_nets must be an integer >= 1"):
        RoutingParams(board_nets=0).check()


# --------------------------------------------------------------------------- a net rule's fence beside escapes


def row3_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """``ROW3V`` where the floorplan put U901 on the transceiver (x 7.87, 0.07 mm off the 0.2 mm grid; y 20.1125): pad 1 (net A) escapes
    first (the fan's straightest pad) by a stub to (8.4, 19.2), pad 2 (net B) after it by a bent stub (7.87, 19.7) -> (7.8, 19.2)
    whose box comes 0.405 mm from A's cell and whose copper 0.4692 mm; each pad on its own net to a through-hole pad at y = 3."""
    parts = [("U1", "ROW3V", 7.87, 20.1125), ("R1", "PAD1", 12.0, 3.0), ("R2", "PAD1", 8.0, 3.0), ("R3", "PAD1", 4.0, 3.0)]
    nets = {"A": [("U1", "1"), ("R1", "1")], "B": [("U1", "2"), ("R2", "1")], "C": [("U1", "3"), ("R3", "1")]}
    return board_ir(tmp_path, lib, parts, nets, (20.0, 25.0))


def test_a_rule_nets_escaped_terminal_beside_a_later_bent_escape_stays_usable(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The transceivers' PA_PD in the SI re-route: a net rule's fence measured another net's bent escape stub by its box, whose corner
    the stub does not fill, and fenced the rule net's own escape cell that the escape pass (exact) had accepted - the net was unrouted
    and the refusal blamed "a foreign pad or the board edge ... the pad pitch". The fence now measures an escape by its copper."""
    ir = row3_board(tmp_path, lib)
    rules = {"A": NetRule(width_mm=0.35, neckdown_width_mm=0.25)}
    got = route_board(ir, lib, FINE, rules=rules)
    esc = {row["pad"]: row for row in got.stats["escapes"]}
    assert [row["pad"] for row in got.stats["escapes"]][:2] == ["U1.1", "U1.2"]  # A's escape first, B's after it
    assert esc["U1.1"]["cell"] == [8.4, 19.2] and esc["U1.2"]["points"] == [[7.87, 20.1125], [7.87, 19.7], [7.8, 19.2]]
    assert got.unrouted == {} and {t.width_mm for t in got.tracks if t.net == "A"} == {0.25, 0.35}
    assert got.stats["rules"]["A"]["neckdown_pads"] == ["U1.1"]
    # the geometry: B's bent segment's box is inside the neck-down fence radius (0.2 + 0.25/2 + 0.2/2), its copper is not
    board = _Board(ir, lib, FINE, net_rules={0: (0.35, 0.2, 0.25, 1.0)})
    idx_a = board.net_index["A"]
    assert idx_a == 0
    cell = next(t.cell for t in board.terminals["A"] if t.label == "U1.1")
    x, y = board.xy(cell)
    bent = next(g for g in board.pads if g.kind == "escape" and g.ref == "U1" and g.number == "2:escape" and g.shape[1] == (7.87, 19.7))
    limit = 0.2 + 0.25 / 2 + 0.2 / 2
    box, copper = maze._pt_box(x, y, bent.box), board._pt_gap(x, y, bent)
    assert box == pytest.approx(0.405) and copper == pytest.approx(0.4692, abs=1e-4) and box < limit < copper
    assert not board.rule_blocked(cell, 0, idx_a, 0.25, 0.2, board._clear_of)
    # the box measure of before fences the cell, and the refusal names the entry that fences it (not "a foreign pad or the board edge")
    monkeypatch.setattr(maze, "_EXACT_FENCE_KINDS", frozenset())
    old = route_board(ir, lib, FINE, rules=rules)
    assert list(old.unrouted) == ["A"]
    assert old.unrouted["A"] == ("U1.1 terminal cell (8.4, 19.2) is inside its net rule's keep-out on every copper layer of the pad (the escape "
                                 "stub of pad U1.2 is within clearance 0.2 + width 0.25/2 of it)")


def test_a_later_escape_never_fences_a_rule_nets_earlier_terminal(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Order independence: without a neck-down, A's escape cell needs 0.2 + 0.35/2 + 0.2/2 = 0.475 mm from B's copper; B's bent stub
    (0.4692 mm) keeps only the board width's clearance of the reserved cell. An escape whose copper would put another net's usable rule
    terminal inside that net's fence is refused, so the escape pass chooses again and every net routes."""
    ir = row3_board(tmp_path, lib)
    rules = {"A": NetRule(width_mm=0.35)}
    got = route_board(ir, lib, FINE, rules=rules)
    assert got.unrouted == {} and {t.width_mm for t in got.tracks if t.net == "A"} == {0.25, 0.35}
    board = _Board(ir, lib, FINE, net_rules={0: (0.35, 0.2, None, None)})
    for t in board.terminals["A"]:  # every terminal of the rule net keeps its fence clear on a layer it may start on
        assert any(not board.rule_blocked(t.cell, layer, 0, 0.35, 0.2, board._clear_of) for layer in board.usable_layers(t, 0)), t.label
    assert all(net == 0 for _c, net, _pad in board._rule_guard) and {pad.number for _c, _n, pad in board._rule_guard} == {"1"}
    # the refusal a guarded cell gives: B's stub of before, beside A's cell (8.4, 19.2)
    k = next(k for k in range(board.n) if board.pos(k) == (8.4, 19.2))
    board._rule_guard.append((k, 0, next(t.pad for t in board.terminals["A"] if t.label == "U1.1")))
    hit = board._guard_hit([(7.87, 20.1125), (7.87, 19.7), (7.8, 19.2)], 0.125, 0, None, board.net_index["B"])
    assert hit == "its copper would put the terminal cell (8.4, 19.2) of pad U1.1 (net A) inside that net rule's keep-out on F.Cu"
    # without the guard B's escape of before is taken and A is unrouted by its own fence (the escape order decided it)
    monkeypatch.setattr(maze._Board, "_guard_hit", lambda self, *a: None)
    old = route_board(ir, lib, FINE, rules=rules)
    assert list(old.unrouted) == ["A"] and "(the escape stub of pad U1.2 is within clearance 0.2 + width 0.35/2 of it)" in old.unrouted["A"]


def test_at_the_board_width_a_rule_fence_never_fences_a_cell_the_owner_map_leaves_free(tmp_path: Path, lib: KicadLibrary):
    """A net rule's fence at the board's width and clearance is the owner map's keep-out: pads (and 0.4 plane vias) by their box, as
    the owner map claims them, escapes by their exact copper, as their claim reaches (row3 and the ROW5 fan, off the grid)."""
    for ir in (row3_board(tmp_path / "a", lib), row_board(tmp_path / "b", lib, x=12.125)):
        board = _Board(ir, lib, FINE)
        assert any(g.kind == "escape" for g in board.pads)
        edge = FINE.edge_clearance_mm + FINE.track_width_mm / 2.0
        for name, idx in board.net_index.items():
            fence = board.rule_fence(idx, FINE.track_width_mm, FINE.clearance_mm, None, None, FINE.clearance_mm, board._clear_of)
            for layer in (0, 1):
                bad = [board.pos(k) for k in range(board.n) if fence[layer * board.n + k] and board.owner[layer][k] in (None, idx)
                       and board._edge_distance(k) >= edge]
                assert not bad, (name, layer, bad[:5])


# --------------------------------------------------------------------------- SI, parameters, no change without a need


def test_si_impedance_names_an_escape_segment_like_a_neck_down(tmp_path: Path, lib: KicadLibrary):
    r = route_board(row_board(tmp_path, lib), lib, FINE)
    stub = _stub_segments(r, "U1.1")[0]
    neck, words = _neckdown(stub, None)
    assert neck and words.startswith("escape stub of pad U1.1 recorded by the router (routing.maze 0.6")
    other = next(t for t in r.tracks if t.net == "S1" and not any(e.startswith(ESCAPE_ENTRY_PREFIX) for e in t.provenance.derived_from))
    assert escape_entry(other) is None and not _neckdown(other.model_copy(update={"width_mm": 0.1}), None)[0]
    wider = stub.model_copy(update={"width_mm": stub.width_mm + 0.05})  # the width the entry records is the stub's
    moved = stub.model_copy(update={"start": (stub.start[0] - 1.0, stub.start[1])})  # not one of the recorded segments
    assert not _neckdown(wider, None)[0] and not _neckdown(moved, None)[0]


def test_a_board_that_needs_no_escape_is_routing_maze_0_2_character_for_character(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.version == ROUTER_VERSION and "escapes" not in r.stats and "escape_refused" not in r.stats
    assert all(DEFAULT_ENTRY in t.provenance.derived_from for t in r.tracks)
    assert RoutingParams().derived_from_entry() == DEFAULT_ENTRY
    assert RoutingParams().derived_from_entry(escapes=True) == DEFAULT_ENTRY + ",escape_reach=1.5,escape_min_width=0.1,escape_width_step=0.01"
    for bad in ({"escape_reach_mm": 0.0}, {"escape_min_width_mm": -0.1}, {"escape_width_step_mm": math.inf}):
        with pytest.raises(CompileError, match="routing parameter escape_"):
            RoutingParams(**bad).check()


def test_a_fab_minimum_track_width_raises_the_narrowest_stub_and_is_recorded_only_with_escapes(tmp_path: Path, lib: KicadLibrary):
    fab = ManufacturingConstraints(min_track_width_mm=assumption(0.127, note="fab page not read"))
    ir = row_board(tmp_path, lib)
    ir.pcb.manufacturing = fab
    r = route_board(ir, lib, FINE)
    assert r.unrouted == {} and r.version == "0.6" and r.params.escape_min_width_mm == 0.127
    assert r.stats["raised"] == {"escape_min_width_mm": [0.1, 0.127]}  # the fine rules' 0.25 mm track is above the limit
    assert all(e["width_mm"] >= 0.127 for e in r.stats["escapes"])
    for item in [*r.tracks, *r.vias]:
        assert next(e for e in item.provenance.derived_from if e.startswith("params:")).endswith(",escape_min_width=0.127,escape_width_step=0.01")
    # a board without an escape records no escape knob, raised or not
    plain = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))
    plain.pcb.manufacturing = fab
    r = route_board(plain, lib)
    assert r.version == ROUTER_VERSION and r.stats["raised"] == {} and r.params == RoutingParams()
    assert all(DEFAULT_ENTRY in t.provenance.derived_from for t in r.tracks)


def test_a_stub_keeps_the_pads_narrow_side_when_it_fits(tmp_path: Path):
    """The width is the pad's narrow side (or the net's width) when that keeps every clearance - 0.254 mm here, not a multiple of the
    0.01 mm width step -, else the widest multiple of the step that does (the fanned row: every stub a multiple or the full 0.25 mm)."""
    lib = fine_library(tmp_path / "kicad")
    (tmp_path / "kicad" / "footprints" / "Test.pretty" / "NARROW.kicad_mod").write_text(
        '(footprint "NARROW" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n'
        '  (pad "1" smd rect (at 0 0) (size 0.254 1.2) (layers "F.Cu" "F.Mask" "F.Paste")))\n', encoding="utf-8")
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("T1", "NARROW", 9.125, 3.125)], {"N": [("R1", "1"), ("T1", "1")]}, (12.0, 6.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and [(e["pad"], e["kind"], e["why"], e["width_mm"]) for e in r.stats["escapes"]] == [("T1.1", "stub", "off-grid", 0.254)]
    step = RoutingParams().escape_width_step_mm
    for d in (1.5, 1.7):  # the narrowed fans squeeze a stub below the pad's 0.25 mm (to 0.22 mm)
        board = _Board(narrowed_row(tmp_path, lib, d), lib, FINE)
        widths = [e["width_mm"] for e in board.escapes if e["kind"] == "stub"]
        assert min(widths) < 0.25 and all(w == 0.25 or abs(w / step - round(w / step)) < 1e-6 for w in widths), widths


# --------------------------------------------------------------------------- the real KiCad 10.0.6 footprints

REAL = KicadLibrary()
QFN12 = ("Package_DFN_QFN", "QFN-12-1EP_3x3mm_P0.5mm_EP1.6x1.6mm_ThermalVias")
DFN14 = ("Package_DFN_QFN", "DFN-14-1EP_3x3mm_P0.4mm_EP1.78x2.35mm")
HAS_REAL = all(REAL.footprint_file(*fp) is not None for fp in (QFN12, DFN14, ("Resistor_SMD", "R_0603_1608Metric")))
needs_real = pytest.mark.skipif(not HAS_REAL, reason="KiCad 10 footprint libraries with the QFN-12 / DFN-14 not installed (set KICAD10_SYMBOL_DIR)")


def _part(ref: str, lib_id: tuple[str, str], pins: list[str]) -> Component:
    return Component(
        ref=ref, value="x", pins=[Pin(number=n, name=n, electrical_type=PinElectricalType.PASSIVE, provenance=NET_P) for n in pins],
        symbol=LibraryRef(library="Device", name="R"), footprint=REAL.resolve_footprint(LibraryRef(library=lib_id[0], name=lib_id[1])), provenance=NET_P,
    )


def _real_board(tmp_path: Path, lib_id: tuple[str, str], pins: int, x: float, y: float) -> CircuitIR:
    """The part at (x, y) on a 24 x 24 mm board, pin i wired to its own 0603 resistor on a circle of 8.5 mm around it."""
    ir = CircuitIR(project=ProjectMeta(id="fine", name="fine", workdir=str(tmp_path)))
    ir.components.append(_part("U1", lib_id, [str(i) for i in range(1, pins + 1)]))
    placements = [Placement(component_ref="U1", x_mm=x, y_mm=y, provenance=NET_P)]
    for i in range(1, pins + 1):
        ir.components.append(_part(f"R{i}", ("Resistor_SMD", "R_0603_1608Metric"), ["1", "2"]))
        angle = (i - 0.5) / pins * 2.0 * math.pi
        placements.append(Placement(component_ref=f"R{i}", x_mm=round(x + 8.5 * math.cos(angle), 1), y_mm=round(y + 8.5 * math.sin(angle), 1), provenance=NET_P))
    ir.nets = [Net(name=f"P{i}", pins=[PinRef(component_ref="U1", pin_number=str(i)), PinRef(component_ref=f"R{i}", pin_number="1")], provenance=NET_P)
               for i in range(1, pins + 1)]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=24.0, height_mm=24.0), placements=placements)
    return ir


@needs_real
@pytest.mark.parametrize(("lib_id", "pins", "x", "y", "offgrid"), [
    (QFN12, 12, 11.92, 12.0, True),  # the PA as the transceiver's floorplan puts it: pads 0.13 mm off the grid (0.5 refused the board)
    (DFN14, 14, 12.0, 12.0, False),  # the MAX9814 on the grid: every pad fenced by its neighbours 0.15 mm away
])
def test_the_real_fine_pitch_packages_escape_and_route(tmp_path: Path, lib_id, pins: int, x: float, y: float, offgrid: bool):
    ir = _real_board(tmp_path, lib_id, pins, x, y)
    params = RoutingParams.for_board(ir, REAL)
    assert params.rules == "fine" and params.grid_mm == 0.2
    r = route_board(ir, REAL, params)
    assert r.unrouted == {} and r.version == "0.6" and r.stats["escape_refused"] == {}
    rows = r.stats["escapes"]
    assert sorted(row["pad"] for row in rows) == sorted(f"U1.{i}" for i in range(1, pins + 1))
    assert ("off-grid" in {row["why"] for row in rows}) is offgrid and all(row["width_mm"] <= 0.25 for row in rows)
    x2 = copy.deepcopy(ir)
    x2.pcb.tracks, x2.pcb.vias = list(r.tracks), list(r.vias)
    _, bad = clearance_rows(LayoutBoard(x2, REAL), params.clearance_mm)
    assert bad == [], bad[:3]
    checks = _routing_checks(ir, r.tracks, r.vias, REAL, tmp_path, params.clearance_mm)
    assert checks == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}, checks
