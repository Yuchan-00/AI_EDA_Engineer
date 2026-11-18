"""Baseline digests: today's copper on every board kind the multilayer routing phases touch, pinned byte for byte.

The multilayer plan extends ``routing.maze`` to N signal layers behind an opt-in ``stack``; every board routed without one must keep
its copper and provenance exactly. :data:`GOLDEN_0_2` / :data:`GOLDEN_0_6` (``tests/test_routing_rules.py``) pin the 2-layer
fixture boards; :data:`BASELINE` pins the rest, each the SHA-256 of :func:`tests.test_routing_rules.canonical` (the design view of
the tracks and vias with their provenance, the unrouted reasons, the stats and the parameters) as the router at git c270797
produced it. The boards, all on synthetic fixture libraries (no KiCad):

* 4-layer boards routed with ``inner_layers=True``: GND planes with plane vias (``tests/test_rf_floorplan.py``, routing.maze 0.4),
  the gap board without and with a per-net width / clearance rule (``tests/test_routing_rules.py``, 0.2 / 0.3), a custom pad's
  plane via (``tests/test_routing_custom_pads.py``, 0.5);
* rule nets on F.Cu / B.Cu: a coupled pair, a pair beside a meandered match group, a length budget that counts a via (0.3);
* escapes at the fine rules: the 0.5 mm-pitch row fanned out on and off the grid, a rule net's neck-down beside another net's bent
  escape stub (0.6).

Each board is routed twice, on fresh IRs with fresh library instances, and both runs must give the same bytes. The boards whose
tests patch router hooks or need the real KiCad 10 libraries (the fine-pitch plane rows, the 0.7 fan-out room, the 0.8 LQFP-48
dogbones) assert their :data:`BASELINE` entry at the end of their own test in ``tests/test_routing_fine_pitch.py``, so the same
patches and skip markers apply; :data:`FINE_PITCH_PINNED` names them.

The digests were captured once on unmodified c270797 and are never regenerated: a later phase that changes one has changed a
board's copper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from ai_eda.ir import BoardSide, NetKind
from ai_eda.tools.routing.maze import NetRule, Routing, route_board
from tests.test_rf_floorplan import PLANE, _plane_board, rf_library
from tests.test_routing import board_ir
from tests.test_routing_custom_pads import _library as custom_pad_library
from tests.test_routing_rules import PAIR_RULES, _gap_board, _pair_board, canonical, digest, rules_library
from tests.test_si_checks import stacked

# --------------------------------------------------------------------------- boards


def _plane_gnd(tmp: Path) -> Routing:
    """``tests/test_rf_floorplan.py``: three SMD GND pads (a via each), a through-hole GND pad and a signal net, GND on In1 / In2."""
    lib = rf_library(tmp / "kicad")
    return route_board(_plane_board(tmp, lib), lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", PLANE), ("In2.Cu", PLANE)]})


def _gap(rules: dict[str, NetRule] | None) -> Callable[[Path], Routing]:
    """``tests/test_routing_rules.py``: the 4-layer gap board, PWR at the board's width or at 0.8 mm / 0.4 mm clearance."""
    def build(tmp: Path) -> Routing:
        ir, lib = _gap_board(tmp)
        return route_board(ir, lib, rules=rules, inner_layers=True)
    return build


def _custom_pad_plane(tmp: Path) -> Routing:
    """``tests/test_routing_custom_pads.py`` (the plane net's custom pad): the SOT-89 tab pad on GND gets its via beyond its copper."""
    lib = custom_pad_library(tmp / "kicad")
    ir = board_ir(tmp, lib, [("U1", "SOT89", 8.0, 6.0), ("C1", "SMD1", 3.0, 3.0), ("R4", "PAD1", 12.0, 10.0)],
                  {"GND": [("U1", "2"), ("C1", "1"), ("R4", "1")]}, (16.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    stacked(ir, 4)
    plane = [(0.5, 0.5), (15.5, 0.5), (15.5, 11.5), (0.5, 11.5)]
    return route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", plane), ("In2.Cu", plane)]})


def _pair(tmp: Path) -> Routing:
    """``tests/test_routing_rules.py``: the coupled pair DP / DN (w 0.3, s 0.25), mitred corner and skew compensation."""
    ir, lib = _pair_board(tmp)
    return route_board(ir, lib, rules=PAIR_RULES)


def _pair_and_match_group(tmp: Path) -> Routing:
    """``tests/test_routing_rules.py`` (determinism): the pair beside a two-net match group that meanders to a 0.5 mm skew."""
    ir, lib = _pair_board(tmp, [("R1", "PAD1", 2.0, 15.0), ("R2", "PAD1", 12.0, 15.0), ("R3", "PAD1", 2.0, 10.0), ("R4", "PAD1", 12.0, 12.5)],
                          {"D0": [("R1", "1"), ("R2", "1")], "D1": [("R3", "1"), ("R4", "1")]})
    rules = dict(PAIR_RULES)
    rules.update({n: NetRule(match_group="BUS", max_skew_mm=0.5, width_mm=0.3) for n in ("D0", "D1")})
    return route_board(ir, lib, rules=rules)


def _budget_via(tmp: Path) -> Routing:
    """``tests/test_routing_rules.py``: top SMD to bottom SMD, one via counted at 1.6 mm against an 8 mm length budget."""
    lib = rules_library(tmp / "kicad")
    ir = board_ir(tmp, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM})
    return route_board(ir, lib, rules={"N": NetRule(max_length_mm=8.0, via_length_mm=1.6)})


# tests/test_routing_fine_pitch.py imports BASELINE from this module, so its helpers are imported inside the builders


def _escape_row(x: float) -> Callable[[Path], Routing]:
    """``tests/test_routing_fine_pitch.py``: ``ROW5`` (0.5 mm pitch) at x, each pad on its own net, fanned out at the fine rules."""
    def build(tmp: Path) -> Routing:
        from tests.test_routing_fine_pitch import FINE, fine_library, row_board

        lib = fine_library(tmp / "kicad")
        return route_board(row_board(tmp, lib, x), lib, FINE)
    return build


def _escape_row3_neckdown(tmp: Path) -> Routing:
    """``tests/test_routing_fine_pitch.py``: ``ROW3V`` off the grid, net A's 0.35 mm rule necked down to 0.25 mm beside B's bent stub."""
    from tests.test_routing_fine_pitch import FINE, fine_library, row3_board

    lib = fine_library(tmp / "kicad")
    return route_board(row3_board(tmp, lib), lib, FINE, rules={"A": NetRule(width_mm=0.35, neckdown_width_mm=0.25)})


#: name -> builder(tmp) -> the routing of a fresh IR on a fresh library
BOARDS: dict[str, Callable[[Path], Routing]] = {
    "plane_gnd_4layer": _plane_gnd,
    "gap_4layer": _gap(None),
    "gap_4layer_pwr_rule": _gap({"PWR": NetRule(net_class="POWER", width_mm=0.8, clearance_mm=0.4)}),
    "custom_pad_plane_4layer": _custom_pad_plane,
    "pair": _pair,
    "pair_and_match_group": _pair_and_match_group,
    "length_budget_via": _budget_via,
    "escape_row_fine": _escape_row(12.0),
    "escape_row_fine_offgrid": _escape_row(12.125),
    "escape_row3_neckdown": _escape_row3_neckdown,
}

#: the entries asserted at the end of their own test in tests/test_routing_fine_pitch.py
FINE_PITCH_PINNED = frozenset({
    "fine_fenced_plane_row", "fine_plane_diagonal_stub", "fine_doomed_plane_net", "fanout_room_0_7", "fanout_room_kept_0_6",
    "qfp48_escape_0_6", "qfp48_dogbone_0_8",
})

#: SHA-256 of :func:`tests.test_routing_rules.canonical` for each board, produced by the router at git c270797 (never regenerated)
BASELINE: dict[str, str] = {
    "custom_pad_plane_4layer": "3814e5358b90b99cf5186d547d9a1891976200cb76d55ae14e1f1fa9c91c7c35",
    "escape_row3_neckdown": "63b59671eaed1a478e8736837174611074c0edcb2732978a043ba302cca35b66",
    "escape_row_fine": "0a202fd217238637dc5d8f90332e6fc63cdf69582ac118783cb37bb40d734e55",
    "escape_row_fine_offgrid": "98c97914c644e9f206a77bad8f5f59312b1ee1f071f9a8c181ddc4579052b972",
    "fanout_room_0_7": "0b2600bfcebc5ceb35b1fe630cc0592dd26a35e7cc058b6cf2955853c6708711",
    "fanout_room_kept_0_6": "851762c21695274c96b082f8d577611673fb4c764f3fd7e55a7676ecaa674090",
    "fine_doomed_plane_net": "21ea5217d8ed9062de8d9914313797cd689d6c5ea5518096c8a6f872b157b361",
    "fine_fenced_plane_row": "63d2d1152ac3b7a3bedc74b45cc916f1053061d74ef4bd6f3e22ad88f161a69a",
    "fine_plane_diagonal_stub": "b90987e47e88eabab8c1a9e24e99ebead9e216ce24db3e159886befeb41205fa",
    "gap_4layer": "3735a04356601dcd60a234a12950deaf44043a21f384615c73c49331b52b47df",
    "gap_4layer_pwr_rule": "799b9a45a6e108b823ac2b674212d48aab1bf860c45ee6840facfb2bfb2f73b5",
    "length_budget_via": "f094a75dd1690dc5276709859a6f8a2b6cb54b228a9ca4e962a5608c59c51865",
    "pair": "1e1a10ab2c14be1624e0db1d9485a4296c97ad7015274c5d00479c6d64b196a5",
    "pair_and_match_group": "69188e400085aab9e24630e046bda7dc939d6db590efe0103f6e72cada078049",
    "plane_gnd_4layer": "d39b632f6e4a2b9f462aabe707467b683997c70e0b318095da2026cc8be41335",
    "qfp48_dogbone_0_8": "4c8d6710714b0d40ef67b687fb87bf18e5cf889bf30e95e60843a73b1a14b477",
    "qfp48_escape_0_6": "68d7edc6dafbde072e283a777c2465ff5028c4f01018c0a99b713601f2257de8",
}


# --------------------------------------------------------------------------- tests


def test_every_baseline_entry_is_asserted_somewhere():
    assert set(BASELINE) == set(BOARDS) | FINE_PITCH_PINNED and not set(BOARDS) & FINE_PITCH_PINNED


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_todays_copper_is_pinned_byte_for_byte(tmp_path: Path, name: str):
    a, b = BOARDS[name](tmp_path / "a"), BOARDS[name](tmp_path / "b")
    assert canonical(a) == canonical(b), name  # deterministic: a fresh IR on a fresh library gives the same bytes
    assert digest(a) == BASELINE[name], name
