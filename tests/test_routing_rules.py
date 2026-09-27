"""routing.maze 0.3: per-net rules (``NetRule``) - widths / halos in the negotiation, neck-down, length budgets, meanders, coupled pairs.

Every board is synthetic (the fixture library of ``tests/test_routing.py``
plus the footprints of :func:`rules_library`), so nothing here needs KiCad.
Two promises are checked:

* **No rule, no change.** A board routed without a non-empty rule is
  routing.maze 0.2's, byte for byte: :data:`GOLDEN_0_2` holds the SHA-256 of
  the canonical serialisation (:func:`canonical`: the design view of the
  copper with its provenance, the unrouted reasons, the stats and the
  parameters) that the committed 0.2 router produced on each board of
  :data:`FIXTURES`; the 0.3 router must reproduce every one, with
  ``rules=None``, ``{}`` and rules that constrain nothing.
* **What a rule does is legal and recorded.** The independent IR-geometry
  validator (``pcb.routing.*``, :mod:`ai_eda.validation.layout`) and its
  ``clearance_rows`` judge the copper - not the router's own model - and
  every number the router used is in the provenance and the stats.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, CircuitIR, Layer, ManufacturingConstraints, PCBDesign, assumption
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.coupling import uncoupled_lengths
from ai_eda.tools.routing.maze import (
    FINE_RULES,
    ROUTER_RULES_VERSION,
    ROUTER_VERSION,
    NetRule,
    Routing,
    RoutingParams,
    route_board,
)
from ai_eda.validation.layout import _Board as LayoutBoard
from ai_eda.validation.layout import clearance_rows
from tests.test_routing import (
    SWAP_NETS,
    SWAP_PARTS,
    _astable,
    _placed_mcu,
    board_ir,
    fixture_library,
)

# --------------------------------------------------------------------------- fixtures

#: extra footprints: a 0.8 mm-pitch row of five 0.45 x 1.5 mm SMD pads (a QFP side), a net-less THT bar, two-pad THT
#: pairs (vertical: pin 1 north; horizontal: pin 1 east)
_EXTRA = {
    "ROW5": "\n  ".join(f'(pad "{k + 1}" smd rect (at {(k - 2) * 0.8:g} 0) (size 0.45 1.5) (layers "F.Cu" "F.Mask" "F.Paste"))' for k in range(5)),
    "TBAR": '(pad "" thru_hole rect (at 0 0) (size 0.5 5.0) (drill 0.3) (layers "*.Cu" "*.Mask"))',
    "PAIRV": '(pad "1" thru_hole circle (at 0 -1.5) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n  (pad "2" thru_hole circle (at 0 1.5) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
    "PAIRH": '(pad "1" thru_hole circle (at 1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n  (pad "2" thru_hole circle (at -1.5 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))',
    "BIG5": '(pad "1" smd rect (at 0 0) (size 5 5) (layers "F.Cu" "F.Mask" "F.Paste"))',
}


def rules_library(root: Path) -> KicadLibrary:
    """:func:`tests.test_routing.fixture_library` plus :data:`_EXTRA`."""
    lib = fixture_library(root)
    pretty = root / "footprints" / "Test.pretty"
    for name, pads in _EXTRA.items():
        (pretty / f"{name}.kicad_mod").write_text(
            f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr through_hole)\n'
            f'  (fp_rect (start -1 -1) (end 1 1) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
            f"  {pads})\n",
            encoding="utf-8",
        )
    return lib


def _swap(tmp: Path, capped: bool):
    lib = fixture_library(tmp / "kicad")
    return board_ir(tmp, lib, SWAP_PARTS, SWAP_NETS, (12.0, 9.0)), lib, RoutingParams(max_iterations=1) if capped else None


def _recover(tmp: Path):
    lib = fixture_library(tmp / "kicad")
    ir = board_ir(tmp, lib, [("Z1", "BPLANE", 6.0, 5.0), ("R1", "SMD1", 2.0, 5.0), ("R2", "SMD1", 10.0, 5.0), ("R3", "SMD1", 6.0, 3.0), ("R4", "SMD1", 6.0, 7.0)],
                  {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0))
    return ir, lib, RoutingParams(max_iterations=1)


def _recover_halo(tmp: Path):
    lib = fixture_library(tmp / "kicad")
    parts = [("R1", "TINY", 5.0, 5.0), ("R2", "SMD1", 3.5, 3.25), ("R3", "SMD1", 6.0, 3.25), ("R4", "TINY", 5.5, 7.0), ("Z1", "BPLANE", 5.0, 5.0)]
    ir = board_ir(tmp, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (10.0, 10.0))
    return ir, lib, RoutingParams(grid_mm=0.25, track_width_mm=0.9, clearance_mm=0.2, max_iterations=1, via_diameter_mm=1.0, via_drill_mm=0.4)


def _simple(parts, nets, size, sides=None, params=None, fab=None):
    def build(tmp: Path):
        lib = fixture_library(tmp / "kicad")
        ir = board_ir(tmp, lib, parts, nets, size, sides=sides)
        if fab is not None:
            ir.pcb.manufacturing = fab
        return ir, lib, params
    return build


def _astable_fixture(tmp: Path):
    from tests.test_circuit_templates import template_library

    lib = template_library(tmp / "kicad")
    return _astable(tmp, lib), lib, None


def _divider_fixture(tmp: Path):
    from tests.fixtures_kicad import divider_with_connector_ir
    from tests.test_circuit_templates import template_library

    lib = template_library(tmp / "kicad")
    ir = divider_with_connector_ir(tmp, lib)
    return ir, lib, RoutingParams.for_board(ir, lib)


def _mcu_fixture(tmp: Path):
    ir, lib = _placed_mcu(tmp)
    return ir, lib, RoutingParams.for_board(ir, lib)


def _mcu_pa_fixture(tmp: Path):
    ir, lib = _placed_mcu(tmp)
    keep = {"U1", "J1"}
    ir.components = [c for c in ir.components if c.ref in keep]
    ir.nets = [n for n in ir.nets if n.name.startswith("PA")]
    ir.pcb.placements = [p for p in ir.pcb.placements if p.component_ref in keep]
    return ir, lib, RoutingParams.for_board(ir, lib)


_FAB = ManufacturingConstraints(
    fab="JLCPCB", min_track_width_mm=assumption(0.5, note="fab page not read"), min_via_drill_mm=assumption(0.5, note="idem"),
    min_via_diameter_mm=assumption(1.0, note="idem"), min_hole_to_edge_mm=assumption(0.6, note="idem"),
)

#: the boards of ``tests/test_routing.py`` (and the divider / astable / 64-pin MCU template boards): name -> builder(tmp) -> (ir, lib, params)
FIXTURES = {
    "straight": _simple([("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0)),
    "offgrid": _simple([("R1", "PAD1", 3.1, 3.0), ("R2", "PAD1", 9.0, 3.05)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0)),
    "via": _simple([("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM}),
    "foreign_pad": _simple([("R1", "PAD1", 3.0, 4.0), ("R2", "PAD1", 9.0, 4.0), ("R3", "PAD1", 6.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (12.0, 8.0)),
    "wall": _simple([("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0), ("W1", "WALL", 6.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0)),
    "corner": _simple([("R1", "PAD1", 1.0, 1.0), ("R2", "PAD1", 7.0, 5.0)], {"N": [("R1", "1"), ("R2", "1")]}, (8.0, 6.0)),
    "two_nets": _simple(
        [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0), ("W1", "WALL", 6.0, 4.0), ("R3", "PAD1", 3.0, 7.0), ("R4", "PAD1", 9.0, 7.0)],
        {"N": [("R1", "1"), ("R2", "1")], "K": [("R3", "1"), ("R4", "1")]}, (12.0, 10.0),
    ),
    "fab": _simple([("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM}, fab=_FAB),
    "keepout": _simple(
        [("R1", "SMD054", 4.875, 3.0), ("R2", "PAD1", 1.0, 3.0), ("R3", "SMD1", 5.895, 3.0), ("R4", "PAD1", 9.0, 3.0)],
        {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")]}, (10.0, 6.0),
    ),
    "cage": _simple(
        [("R1", "PAD1", 3.0, 3.0), ("C1", "CAGE", 9.0, 3.0), ("R2", "PAD1", 3.0, 7.0), ("R3", "PAD1", 9.0, 7.0), ("R4", "PAD1", 6.0, 9.0)],
        {"N": [("R1", "1"), ("C1", "1")], "K": [("R2", "1"), ("R3", "1")], "S": [("R4", "1")]}, (12.0, 10.0),
    ),
    "compile": _simple(
        [("R1", "SMD2", 2.5, 4.0), ("R2", "SMD2", 10.5, 4.0), ("W1", "WALL2", 6.5, 8.0), ("R3", "PAD2", 2.5, 12.0), ("R4", "PAD2", 10.5, 12.0)],
        {"N": [("R1", "2"), ("R2", "1")], "K": [("R3", "2"), ("R4", "1")]}, (14.0, 16.0),
    ),
    "plus": _simple(
        [("R1", "PAD1", 2.0, 6.0), ("R2", "PAD1", 6.0, 2.0), ("R3", "PAD1", 6.0, 6.0), ("R4", "PAD1", 6.0, 10.0), ("R5", "PAD1", 10.0, 6.0), ("R6", "PAD1", 2.0, 2.0), ("R7", "PAD1", 10.0, 10.0)],
        {"S": [(r, "1") for r in ("R1", "R2", "R3", "R4", "R5")], "T": [("R6", "1"), ("R7", "1")]}, (12.0, 12.0),
    ),
    "dup": _simple([("R1", "PAD1", 2.0, 4.0), ("S1", "DUP1", 6.1, 4.1)], {"N": [("R1", "1"), ("S1", "1")]}, (10.0, 8.0)),
    "swap": lambda tmp: _swap(tmp, False),
    "swap_capped": lambda tmp: _swap(tmp, True),
    "recover": _recover,
    "recover_halo": _recover_halo,
    "astable": _astable_fixture,
    "divider": _divider_fixture,
    "mcu_pa": _mcu_pa_fixture,
    "mcu": _mcu_fixture,
}

#: the fields routing.maze 0.2's RoutingParams had (the 0.3 knobs are compared through derived_from_entry, which leaves them out without rules)
PARAMS_0_2 = (
    "grid_mm", "track_width_mm", "clearance_mm", "via_diameter_mm", "via_drill_mm", "edge_clearance_mm", "via_cost", "bend_cost", "base_cost",
    "history_cost", "present_cost", "present_growth", "max_iterations", "window_mm", "rules", "pad_pitch_mm", "pitch_footprint",
)


def canonical(r) -> bytes:
    """The bytes a routing result is compared by: copper (design view, provenance included), unrouted reasons, stats, parameters."""
    copper = json.dumps(design_data(PCBDesign(tracks=r.tracks, vias=r.vias)), sort_keys=True)
    params = [(name, getattr(r.params, name)) for name in PARAMS_0_2]
    return "\n".join([copper, repr(r.unrouted), repr(r.stats), repr(params), r.params.derived_from_entry()]).encode()


def digest(r) -> str:
    return hashlib.sha256(canonical(r)).hexdigest()


#: SHA-256 of :func:`canonical` for each board of :data:`FIXTURES`, produced by the committed routing.maze 0.2 (git dbd384d)
GOLDEN_0_2: dict[str, str] = {
    "astable": "749ff73ac3a9ebd5e41b6c2929a9fe3d913ba42a70de1f5ef69b5310ddde4fa3",
    "cage": "de931b9efb6c85b8703fe22262a230360c5902b4e034205122392d62a6d71658",
    "compile": "d9dbe8a0cbef5bd055fd5f2f8a64bccb77719feddc8dae2c4571dcea01df1126",
    "corner": "e24880897a1418cfaccfd92759dc51220ad664c84513c0b594dff36ab8019548",
    "divider": "85cc59da241fcfe8174fdece987a3a08cb993f594dc3badfaac5641b47aae4f4",
    "dup": "3205d813d0beda6266bb19779f6a231658133f5c21a980304097ea88d307b7f5",
    "fab": "e61106d938cbd7dd0f7c017717ff5b81f189fea7decb23bcdc6205fff1a262e9",
    "foreign_pad": "80fbda11d7273d4e3a193bb08b4eae1c60fdcc338a21d656e778afffbf67fc1a",
    "keepout": "105c29c76891794adaf3b33ea4caa80e66d88f4d73bde4871c96ff25b23b13c6",
    "mcu": "5594bdf8ee045a90d17c556ff187b60710e3a70a649f0e922c87194f7395bf82",
    "mcu_pa": "9407258c51715408e89cc6a583b318f7425254572a53b7a8847e740c32a9fe63",
    "offgrid": "f6a93d4ffa6b77b7af49520912e87c5a557716cd05629fff1cf184ae4e2bf348",
    "plus": "ab637b81287768ab77569ded55f22cd7a875f89e6b551b41b4d83599c6c8871e",
    "recover": "3918b2c5e2a56130806e30b1530a21ff630985654f1be02d6dee776353110e4b",
    "recover_halo": "dfe96b70d3279dfe5d8ba3c21900b861d09bac932559e9f114e747fb40265f86",
    "straight": "bbee7aeeaf4c3751b2c6f0085b6a2ca6d244342041188797f898b9bbc6e69409",
    "swap": "a57f3108609a9782c9d3bb930ffa7e06318af3518e0201ce68d5f94fa727f110",
    "swap_capped": "ec8de87c247c575d936ff50914a711c96130f259a455589ff760cc2f315598be",
    "two_nets": "839f1c661f08a943d24dabc838b8f007eaeb09aa99cb026e41124b715f8bf5c1",
    "via": "0873f8b3854d5297c54607fc29c0f90bd2fe31594812b4007e6936b7c8e17dbe",
    "wall": "848698a9caad87c41022a8710cc8fd0cd59b44e175601b12c99a76a546f30f7a",
}


# --------------------------------------------------------------------------- helpers


def _checks(ir: CircuitIR, r: Routing, lib: KicadLibrary, tmp_path: Path, clearance: float) -> dict:
    """The ``pcb.routing`` validator on ``ir`` carrying ``r``'s copper at the given clearance limit (IR geometry, not DRC)."""
    from ai_eda.validation import ValidationContext, default_registry

    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(r.tracks), list(r.vias)
    x.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(clearance, note="the router's own clearance as the limit"))
    return {c.check_id: c.status.value for c in default_registry.get("pcb.routing").validate(x, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))}


def _violations_of(ir: CircuitIR, r: Routing, lib: KicadLibrary, net: str, limit: float) -> list[dict]:
    """``clearance_rows`` at ``limit`` restricted to the copper and pads of ``net`` (the validator's exact distances)."""
    x = copy.deepcopy(ir)
    x.pcb.tracks, x.pcb.vias = list(r.tracks), list(r.vias)
    board = LayoutBoard(x, lib)
    pads = {f"{p.component_ref}.{p.pin_number}" for n in ir.nets if n.name == net for p in n.pins}
    _, rows = clearance_rows(board, limit)
    return [row for row in rows if f":{net}]" in row["a"] or f":{net}]" in row["b"] or row["a"] in pads or row["b"] in pads]


def _length(tracks) -> float:
    return sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in tracks)


# --------------------------------------------------------------------------- no rule, no change


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_without_rules_every_fixture_is_routing_maze_0_2_byte_for_byte(tmp_path: Path, name: str):
    ir, lib, params = FIXTURES[name](tmp_path)
    before = ir.content_hash()
    plain = route_board(ir, lib, params)
    assert ir.content_hash() == before and plain.version == ROUTER_VERSION == "0.2" and plain.rules == {}
    assert digest(plain) == GOLDEN_0_2[name], name
    # an empty mapping and rules that constrain nothing (a class name only) are no rules either
    nothing = {net.name: NetRule(net_class="DEFAULT") for net in ir.nets}
    for rules in ({}, nothing, {ir.nets[0].name: NetRule()}):
        again = route_board(copy.deepcopy(ir), lib, params, rules=rules)
        assert canonical(again) == canonical(plain) and again.version == ROUTER_VERSION, (name, rules)


def test_inner_layers_are_refused_as_in_0_2_unless_the_caller_opts_in(tmp_path: Path):
    """A 4-layer board (``In1.Cu`` / ``In2.Cu`` planes) is refused with 0.2's message by default; with ``inner_layers=True`` it is
    routed on F.Cu / B.Cu only, its copper that of the same board with two layers, byte for byte, and the stats name the inner
    layers. A copper layer that is not ``In<k>.Cu`` is refused either way."""
    lib = rules_library(tmp_path / "kicad")

    def two_layer():
        return board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 6.0))

    four = two_layer()
    four.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="power"), Layer(name="In2.Cu", kind="power"), Layer(name="B.Cu", kind="signal")]
    with pytest.raises(CompileError, match=r"^cannot route: this router knows only \['F.Cu', 'B.Cu'\], ir.pcb.layers also has \['In1.Cu', 'In2.Cu'\]$"):
        route_board(four, lib)
    routed, twin = route_board(four, lib, inner_layers=True), route_board(two_layer(), lib)
    assert json.dumps(design_data(PCBDesign(tracks=routed.tracks, vias=routed.vias))) == json.dumps(design_data(PCBDesign(tracks=twin.tracks, vias=twin.vias)))
    assert routed.stats.pop("inner_layers") == ["In1.Cu", "In2.Cu"] and routed.stats == twin.stats and "inner_layers" not in twin.stats
    assert {t.layer for t in routed.tracks} <= {"F.Cu", "B.Cu"} and routed.version == twin.version == ROUTER_VERSION
    four.pcb.layers[1] = Layer(name="Mid.Cu", kind="signal")
    with pytest.raises(CompileError, match=r"knows only \['F.Cu', 'B.Cu'\] \(and inner In<k>.Cu layers, which it never routes\), ir.pcb.layers also has \['Mid.Cu'\]"):
        route_board(four, lib, inner_layers=True)


# --------------------------------------------------------------------------- rules are checked first


def test_rules_that_make_no_sense_are_refused_before_anything_is_routed(tmp_path: Path):
    lib = rules_library(tmp_path / "kicad")
    parts = [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 9.0, 3.0), ("R3", "PAD1", 3.0, 6.0), ("R4", "PAD1", 9.0, 6.0), ("R5", "PAD1", 3.0, 9.0), ("R6", "PAD1", 6.0, 9.0), ("R7", "PAD1", 9.0, 9.0)]
    ir = board_ir(tmp_path, lib, parts, {"A": [("R1", "1"), ("R2", "1")], "B": [("R3", "1"), ("R4", "1")], "C": [("R5", "1"), ("R6", "1"), ("R7", "1")]}, (12.0, 11.0))
    pair = dict(width_mm=0.3, pair_spacing_mm=0.25)
    for rules, match in (
        ({"Z": NetRule(width_mm=0.3)}, "net rule for 'Z': the IR has no such net"),
        ({"A": NetRule(width_mm=0.0)}, r"width_mm must be a finite number > 0"),
        ({"A": NetRule(max_length_mm=math.inf)}, r"max_length_mm must be a finite number > 0"),
        ({"A": NetRule(clearance_mm=-0.1)}, r"clearance_mm must be a finite number >= 0"),
        ({"A": NetRule(width_mm=True)}, r"width_mm must be a finite number > 0"),
        ({"A": NetRule(match_group="")}, r"match_group must be a non-empty name"),
        ({"A": NetRule(neckdown_width_mm=0.2)}, "neckdown_width_mm needs width_mm"),
        ({"A": NetRule(width_mm=0.3, neckdown_width_mm=0.3)}, "neckdown_width_mm 0.3 must be below width_mm 0.3"),
        ({"A": NetRule(width_mm=0.5, neckdown_radius_mm=1.0)}, "neckdown_radius_mm without neckdown_width_mm"),
        ({"A": NetRule(match_group="G")}, "match_group and max_skew_mm go together"),
        ({"A": NetRule(match_group="G", max_skew_mm=0.5), "B": NetRule(match_group="G", max_skew_mm=0.2)}, r"match group 'G': its nets give different max_skew_mm \[0.2, 0.5\]"),
        ({"A": NetRule(pair_spacing_mm=0.2)}, r"\['pair_spacing_mm'\] without pair_partner"),
        ({"A": NetRule(pair_partner="B", **pair)}, "pair_partner 'B' must be another net whose rule names 'A' back"),
        ({"A": NetRule(pair_partner="A", **pair)}, "pair_partner 'A' must be another net"),
        ({"A": NetRule(pair_partner="B", width_mm=0.3), "B": NetRule(pair_partner="A", width_mm=0.3)}, "a coupled pair needs width_mm and pair_spacing_mm"),
        ({"A": NetRule(pair_partner="B", **pair), "B": NetRule(pair_partner="A", width_mm=0.35, pair_spacing_mm=0.25)}, r"width_mm 0.3 differs from its partner 'B' \(0.35\)"),
        ({"A": NetRule(pair_partner="B", width_mm=0.3, pair_spacing_mm=0.1), "B": NetRule(pair_partner="A", width_mm=0.3, pair_spacing_mm=0.1)},
         "pair_spacing_mm 0.1 is below the board clearance 0.25 mm"),
        ({"A": NetRule(pair_partner="B", match_group="G", max_skew_mm=0.1, **pair), "B": NetRule(pair_partner="A", match_group="G", max_skew_mm=0.1, **pair)},
         "a coupled pair in a match group is not implemented"),
        ({"A": NetRule(pair_partner="C", **pair), "C": NetRule(pair_partner="A", **pair)}, r"coupled pair A/C: net 'C' has 3 pad\(s\)"),
    ):
        with pytest.raises(CompileError, match=match):
            route_board(ir, lib, rules=rules)
    with pytest.raises(CompileError, match="net rules must be a mapping"):
        route_board(ir, lib, rules=[NetRule(width_mm=0.3)])
    with pytest.raises(CompileError, match="the value a NetRule"):
        route_board(ir, lib, rules={"A": {"width_mm": 0.3}})
    # the new knobs are parameters like the others
    for bad in (RoutingParams(neckdown_radius_mm=0.0), RoutingParams(meander_pitch_mm=-1.0), RoutingParams(pair_bump_amplitude_mm=math.nan)):
        with pytest.raises(CompileError, match="routing parameter"):
            route_board(ir, lib, bad)


def test_rule_widths_below_the_fab_minimum_are_raised_and_recorded(tmp_path: Path):
    lib = rules_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 12.0, 3.0)], {"A": [("R1", "1"), ("R2", "1")]}, (15.0, 6.0))
    ir.pcb.manufacturing = ManufacturingConstraints(min_track_width_mm=assumption(0.5, note="fab page not read"))
    r = route_board(ir, lib, rules={"A": NetRule(width_mm=0.3, clearance_mm=0.1, neckdown_width_mm=0.25)})
    assert r.unrouted == {} and {t.width_mm for t in r.tracks} == {0.5}
    entry = r.stats["rules"]["A"]
    # the width to the fab minimum, the clearance to the board's, and the neck-down dropped: the raise left nothing to narrow
    assert entry["raised"] == {"width_mm": [0.3, 0.5], "clearance_mm": [0.1, 0.25], "neckdown_width_mm": [0.25, None]}
    assert r.rules["A"] == NetRule(width_mm=0.5, clearance_mm=0.25, via_length_mm=0.0)
    assert entry["rule"] == r.tracks[0].provenance.derived_from[-1] == r.rules["A"].derived_from_entry()
    assert entry["rule"].startswith("rule:class=None,width=0.5,clearance=0.25,neckdown_width=None,neckdown_radius=None,max_length=None,via_length=0.0,")


# --------------------------------------------------------------------------- per-net width and clearance


def _gap_board(tmp_path: Path):
    """A 4-layer board (GND / +5V planes inside) with two net-less THT bars at x = 10 leaving a 1.6 mm gap at y 5.5 .. 7.1."""
    lib = rules_library(tmp_path / "kicad")
    parts = [("R1", "PAD1", 3.0, 6.25), ("R2", "PAD1", 17.0, 6.25), ("R3", "PAD1", 3.0, 3.0), ("R4", "PAD1", 17.0, 3.0), ("T1", "TBAR", 10.0, 3.0), ("T2", "TBAR", 10.0, 9.6)]
    ir = board_ir(tmp_path, lib, parts, {"PWR": [("R1", "1"), ("R2", "1")], "SIG": [("R3", "1"), ("R4", "1")]}, (20.0, 16.0))
    ir.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="power"), Layer(name="In2.Cu", kind="power"), Layer(name="B.Cu", kind="signal")]
    return ir, lib


def test_per_net_width_and_clearance_are_negotiated_on_a_4_layer_board(tmp_path: Path):
    """PWR at 0.8 mm / 0.4 mm clearance does not fit the 1.6 mm gap between the bars (its keep-out 0.4 + 0.4 + 0.125 from each bar
    leaves no grid row); SIG at the board's 0.4 / 0.25 does. Without the rule PWR takes the gap straight (14 mm); with it, PWR goes
    round the lower bar and SIG takes the gap. The inner plane layers are listed and never routed; the independent validator
    passes at the board clearance, and nothing comes nearer than PWR's own 0.4 mm to PWR's copper or pads."""
    ir, lib = _gap_board(tmp_path)
    plain = route_board(ir, lib, inner_layers=True)
    assert plain.version == "0.2" and plain.stats["net_length_mm"]["PWR"] == 14.0 and plain.stats["inner_layers"] == ["In1.Cu", "In2.Cu"]
    rules = {"PWR": NetRule(net_class="POWER", width_mm=0.8, clearance_mm=0.4)}
    r = route_board(ir, lib, rules=rules, inner_layers=True)
    assert r.unrouted == {} and r.version == ROUTER_RULES_VERSION == "0.3" and r.stats["inner_layers"] == ["In1.Cu", "In2.Cu"]
    assert {t.layer for t in r.tracks} <= {"F.Cu", "B.Cu"}
    assert {t.width_mm for t in r.tracks if t.net == "PWR"} == {0.8} and {t.width_mm for t in r.tracks if t.net == "SIG"} == {0.4}
    # PWR crosses x = 10 only below the lower bar (edge 12.1) by at least its clearance + half width; SIG goes through the gap
    crossings = {net: [t.start[1] for t in r.tracks if t.net == net and min(t.start[0], t.end[0]) < 10.0 < max(t.start[0], t.end[0])] for net in ("PWR", "SIG")}
    assert crossings["PWR"] and all(y >= 12.1 + 0.4 + 0.4 for y in crossings["PWR"])
    assert crossings["SIG"] and all(5.5 < y < 7.1 for y in crossings["SIG"])
    assert r.stats["net_length_mm"]["PWR"] == 28.0 > 14.0 + 2 * (12.1 + 0.8 - 6.25)
    assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
    assert _violations_of(ir, r, lib, "PWR", 0.4) == []
    # every number in the provenance: the 0.3 knobs after the 0.2 ones, the effective rule last (for the rule net only)
    entry = RoutingParams().derived_from_entry(rules=True)
    assert entry == RoutingParams().derived_from_entry() + ",neckdown_radius=1.0,meander_amplitude=1.0,meander_pitch=1.0,pair_bump_amplitude=0.25,pair_bump_length=0.5"
    for t in r.tracks:
        prov = t.provenance
        assert prov.tool == "routing.maze" and prov.tool_version == "0.3" and prov.inputs == {}
        if t.net == "PWR":
            assert prov.derived_from == ["net:PWR", "placement:R1", "placement:R2", entry, r.rules["PWR"].derived_from_entry()]
            assert "net rule POWER: width 0.8 mm, clearance 0.4 mm" in prov.note
        else:
            assert prov.derived_from == ["net:SIG", "placement:R3", "placement:R4", entry] and "net rule" not in prov.note
    assert r.stats["rules"]["PWR"]["length_mm"] == 28.0 and r.stats["rules"]["PWR"]["routed"] is True


# --------------------------------------------------------------------------- neck-down


def test_neck_down_lets_a_wide_net_leave_a_0_8_mm_pitch_pad(tmp_path: Path):
    """U1 is a row of 0.45 x 1.5 mm pads at 0.8 mm pitch (a QFP side), routed at the fine rules. CLK asks for 0.6 mm: its keep-out
    0.2 + 0.3 + 0.1 = 0.6 mm reaches past the 0.575 mm from U1.3's centre to its neighbours' boxes, so the pad cannot take the width -
    unrouted, with that reason. With a 0.25 mm neck-down CLK leaves U1.3 at 0.25 mm and widens to 0.6 mm once it is more than
    ``neckdown_radius_mm`` from the pad; the neck-down pad and length are recorded, and the validator passes at the board clearance."""
    lib = rules_library(tmp_path / "kicad")
    parts = [("U1", "ROW5", 5.0, 5.0), ("R1", "SMD1", 5.0, 12.0), ("R2", "SMD1", 1.0, 12.0)]
    ir = board_ir(tmp_path, lib, parts, {"CLK": [("U1", "3"), ("R1", "1")], "N2": [("U1", "2"), ("R2", "1")], "N4": [("U1", "4")]}, (10.0, 14.0))
    p = RoutingParams(**FINE_RULES)
    wide = route_board(ir, lib, p, rules={"CLK": NetRule(width_mm=0.6)})
    assert list(wide.unrouted) == ["CLK"] and "inside its net rule's keep-out" in wide.unrouted["CLK"] and "the pad pitch cannot take that width" in wide.unrouted["CLK"]
    assert {t.net for t in wide.tracks} == {"N2"}
    r = route_board(ir, lib, p, rules={"CLK": NetRule(net_class="CLOCK", width_mm=0.6, neckdown_width_mm=0.25)})
    assert r.unrouted == {} and r.stats["legal"], r.stats
    clk = [t for t in r.tracks if t.net == "CLK"]
    assert {t.width_mm for t in clk} == {0.25, 0.6}
    pad = (5.0 - 0.225, 5.0 - 0.75, 5.0 + 0.225, 5.0 + 0.75)  # U1.3's box
    for t in clk:
        near = min(math.hypot(max(pad[0] - x, 0.0, x - pad[2]), max(pad[1] - y, 0.0, y - pad[3])) for x, y in (t.start, t.end))
        if t.width_mm == 0.25:
            assert near <= 1.0 + p.grid_mm + 1e-9, t  # the neck-down stays within the radius (the step that leaves the zone included)
        else:
            assert near > 1.0 - 1e-9, t
    entry = r.stats["rules"]["CLK"]
    assert entry["neckdown_pads"] == ["U1.3"] and entry["neckdown_length_mm"] == pytest.approx(_length([t for t in clk if t.width_mm == 0.25]))
    assert 0.0 < entry["neckdown_length_mm"] <= 1.0 + 0.75 + p.grid_mm + 1e-9
    assert "neck-down to 0.25 mm within 1 mm (+ the one 0.2 mm grid step that leaves that zone) of U1.3" in clk[0].provenance.note
    assert _checks(ir, r, lib, tmp_path, p.clearance_mm) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}


# --------------------------------------------------------------------------- length budgets


def test_a_length_budget_fails_a_net_honestly_and_counts_its_vias(tmp_path: Path):
    """A THT bar across the board (both layers) makes the shortest legal route 18.5 mm: a 15 mm budget leaves the net unrouted with
    that length named, 20 mm routes it. Top SMD to bottom SMD needs one via: at 1.6 mm per via the route is 6 + 1.6 = 7.6 mm, so
    7.5 mm fails and 8 mm passes."""
    lib = rules_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 4.0), ("R2", "PAD1", 14.0, 4.0), ("T1", "TBAR", 8.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (16.0, 8.0))
    short = route_board(ir, lib, rules={"N": NetRule(max_length_mm=15.0)})
    assert short.tracks == [] and short.unrouted == {
        "N": "no route within max_length_mm 15: the shortest legal branch to R2.1 is 18.500 mm with 15.000 mm of the budget left (stubs 0.000 mm, 0.000 mm already routed)"
    }
    assert short.stats["rules"]["N"]["routed"] is False and short.stats["rules"]["N"]["length_mm"] is None
    ok = route_board(ir, lib, rules={"N": NetRule(max_length_mm=20.0)})
    assert ok.unrouted == {} and _length(ok.tracks) == 18.5 == ok.stats["rules"]["N"]["length_mm"]
    assert "length budget 20 mm (vias 0 mm each)" in ok.tracks[0].provenance.note
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 9.0, 4.0)], {"N": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0), sides={"R2": BoardSide.BOTTOM})
    over = route_board(ir, lib, rules={"N": NetRule(max_length_mm=7.5, via_length_mm=1.6)})
    assert over.tracks == [] and over.vias == [] and "the shortest legal branch to R2.1 is 7.600 mm" in over.unrouted["N"]
    fits = route_board(ir, lib, rules={"N": NetRule(max_length_mm=8.0, via_length_mm=1.6)})
    assert fits.unrouted == {} and len(fits.vias) == 1 and _length(fits.tracks) == 6.0 and fits.stats["rules"]["N"]["length_mm"] == 7.6


def test_a_budgeted_multi_pad_net_is_grown_again_from_its_other_pads(tmp_path: Path):
    """The Steiner growth starts at the pad nearest the centroid (R2 here) and makes a 20 mm tree; grown from R1 the same three pads
    need 16 mm. An 18 mm budget is met by that second tree; 15 mm by none, and the reason is the first tree's."""
    lib = rules_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 2.0), ("R2", "PAD1", 10.0, 6.0), ("R3", "PAD1", 14.0, 2.0)], {"N": [("R1", "1"), ("R2", "1"), ("R3", "1")]}, (16.0, 14.0))
    assert route_board(ir, lib).stats["net_length_mm"]["N"] == 20.0
    r = route_board(ir, lib, rules={"N": NetRule(max_length_mm=18.0)})
    assert r.unrouted == {} and r.stats["net_length_mm"]["N"] == 16.0 == r.stats["rules"]["N"]["length_mm"]
    assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
    r = route_board(ir, lib, rules={"N": NetRule(max_length_mm=15.0)})
    assert r.tracks == [] and r.unrouted["N"].startswith("no route within max_length_mm 15: the shortest legal branch to ")


def test_the_length_budget_is_a_hard_bound_inside_the_negotiation(tmp_path: Path):
    """On the crossing board the only mutually legal pair of routes has B going round R1 (15 mm). With B's budget at 11 mm that
    arrangement is never taken: B keeps a route within 11 mm in every iteration, the negotiation cannot legalise A against it, and A
    is left without copper and with the reason - the budget is not traded away for connectivity, and what is emitted is legal."""
    lib = rules_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, SWAP_PARTS, SWAP_NETS, (12.0, 9.0))
    assert route_board(ir, lib).stats["net_length_mm"] == {"A": 12.0, "B": 15.0}
    r = route_board(ir, lib, rules={"B": NetRule(max_length_mm=11.0)})
    assert r.stats["net_length_mm"]["B"] <= 11.0 and list(r.unrouted) == ["A"] and r.stats["dropped"] == ["A"] and r.stats["legal"] is False
    assert r.unrouted["A"].startswith("no legal route after 40 negotiation iteration(s): its copper still broke the clearance of B")
    assert _checks(ir, r, lib, tmp_path, 0.25)["pcb.routing.clearance"] == "PASS"
    # the budget on A instead: A cannot be shorter than 12 mm at all, so A fails with its budget reason in iteration 1
    r = route_board(ir, lib, rules={"A": NetRule(max_length_mm=11.0)})
    assert r.unrouted["A"].startswith("no route within max_length_mm 11: the shortest legal branch to R2.1 is 12.000 mm") and r.stats["iterations"] == 1


# --------------------------------------------------------------------------- length matching


def _bus_board(tmp_path: Path, extra: list | None = None):
    lib = rules_library(tmp_path / "kicad")
    parts = [("R1", "PAD1", 2.0, 3.0), ("R2", "PAD1", 22.0, 3.0), ("R3", "PAD1", 2.0, 8.0), ("R4", "PAD1", 22.0, 11.0), *(extra or [])]
    return board_ir(tmp_path, lib, parts, {"D0": [("R1", "1"), ("R2", "1")], "D1": [("R3", "1"), ("R4", "1")]}, (24.0, 14.0)), lib


def test_meanders_bring_a_match_group_within_its_skew_budget(tmp_path: Path):
    """D0 runs straight (20 mm), D1 has an L (23 mm). With a 0.5 mm skew budget D0 gets the smallest whole number of grid steps of
    serpentine that reaches the window [22.5, 23]: five steps up and five down in two bumps on its one straight run, 2.5 mm; with a
    0.1 mm budget six steps, 3 mm, matching exactly. The meander is legal (independent validator) and recorded."""
    ir, lib = _bus_board(tmp_path)
    for skew, added, final in ((0.5, 2.5, 22.5), (0.1, 3.0, 23.0)):
        rules = {n: NetRule(net_class="BUS", match_group="BUS", max_skew_mm=skew) for n in ("D0", "D1")}
        r = route_board(ir, lib, rules=rules)
        g = r.stats["match_groups"]["BUS"]
        assert r.unrouted == {} and g["matched"] is True and g["reasons"] == {}, g
        assert g["before_mm"] == {"D0": 20.0, "D1": 23.0} and g["target_mm"] == 23.0 and g["after_mm"] == {"D0": final, "D1": 23.0}
        assert g["skew_mm"] == pytest.approx(23.0 - final) and g["skew_mm"] <= skew
        m = g["meanders"]["D0"]
        assert list(g["meanders"]) == ["D0"] and m["added_mm"] == added and m["bumps"] == 2 and m["leg_pitch_mm"] == 1.0 and m["layer"] == "F.Cu"
        assert sum(m["amplitude_mm"]) * 2 == added and max(m["amplitude_mm"]) <= 1.0 and m["run"] == [(2.0, 3.0), (22.0, 3.0)]
        assert r.stats["net_length_mm"]["D0"] == final and _length([t for t in r.tracks if t.net == "D0"]) == pytest.approx(final)
        assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
        assert all(f"meander +{added:g} mm in 2 bump(s)" in t.provenance.note for t in r.tracks if t.net == "D0")
        assert all("match group BUS (skew" in t.provenance.note for t in r.tracks)
    # a skew budget the lengths already meet: no meander
    r = route_board(ir, lib, rules={n: NetRule(match_group="BUS", max_skew_mm=3.5) for n in ("D0", "D1")})
    assert r.stats["match_groups"]["BUS"]["meanders"] == {} and r.stats["match_groups"]["BUS"]["matched"] and r.stats["net_length_mm"]["D0"] == 20.0


def test_a_match_group_without_room_for_meanders_is_reported_and_its_nets_still_applied(tmp_path: Path):
    """Net-less F.Cu bars 1 mm above and below D0's run leave no cell for a bump: the group is not matched, the reason says why,
    and both nets keep their copper (the skew check judges them)."""
    bars = [(f"H{k}", "HBAR", x, y) for k, (x, y) in enumerate((x, y) for x in (5.0, 9.0, 13.0, 17.0) for y in (2.0, 4.0))]
    ir, lib = _bus_board(tmp_path, bars)
    r = route_board(ir, lib, rules={n: NetRule(match_group="BUS", max_skew_mm=0.5) for n in ("D0", "D1")})
    g = r.stats["match_groups"]["BUS"]
    assert r.unrouted == {} and g["matched"] is False and g["meanders"] == {} and g["skew_mm"] == 3.0
    assert g["reasons"] == {"D0": "no straight run holds 2 bump(s) of up to 1 mm at a 1 mm leg pitch (20 grid steps needed) clear of the other copper"}
    assert r.stats["net_length_mm"] == {"D0": 20.0, "D1": 23.0}
    assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}


# --------------------------------------------------------------------------- coupled differential pairs


def _pair_board(tmp_path: Path, extra: list | None = None, extra_nets: dict | None = None):
    """J1 (pads north / south of (3, 6)) to J2 (pads east / west of (20, 14)): the pair leaves J1 eastward and enters J2 from the
    north - one right turn, which keeps DP (J1's north pad) on the outside and brings it to J2's east pad."""
    lib = rules_library(tmp_path / "kicad")
    parts = [("J1", "PAIRV", 3.0, 6.0), ("J2", "PAIRH", 20.0, 14.0), *(extra or [])]
    nets = {"DP": [("J1", "1"), ("J2", "1")], "DN": [("J1", "2"), ("J2", "2")], **(extra_nets or {})}
    return board_ir(tmp_path, lib, parts, nets, (26.0, 18.0)), lib


PAIR = dict(net_class="USB", width_mm=0.3, pair_spacing_mm=0.25)
PAIR_RULES = {"DP": NetRule(pair_partner="DN", **PAIR), "DN": NetRule(pair_partner="DP", **PAIR)}


def test_a_coupled_pair_keeps_its_spacing_mitres_its_corner_and_records_its_breakout(tmp_path: Path):
    """w = 0.3, s = 0.25: the centreline (2w + s = 0.85 mm wide) launches at (4.25, 6) eastward and lands at (20, 12.75) from the
    north; the tracks run 0.275 mm either side of it - 0.55 mm centre to centre, 0.25 mm apart, on every coupled segment. At the
    corner DP (outside) is chamfered between (20, 5.725) and (20.275, 6), DN (inside) turns at the offset lines' intersection
    (19.725, 6.275); that corner makes DP (2 + sqrt 2) * 0.275 = 0.9389 mm longer, which two 0.2347 mm bumps outward on DN's long
    straight give back. The breakouts are straight, 1.75 mm each, and recorded; the validator passes at the pair's clearance."""
    ir, lib = _pair_board(tmp_path)
    r = route_board(ir, lib, rules=PAIR_RULES)
    assert r.unrouted == {} and r.vias == [] and r.version == "0.3", r.unrouted
    st = r.stats["pairs"]["DN/DP"]
    assert st["reason"] is None and st["layer"] == "F.Cu" and st["centreline_width_mm"] == 0.85 and st["offset_mm"] == 0.275
    assert [(s["pads"], s["at"], s["direction"]) for s in st["launch"]] == [(["J1.2", "J1.1"], (4.25, 6.0), "east"), (["J2.2", "J2.1"], (20.0, 12.75), "north")]
    assert all(s["breakout_mm"] == {"DN": 1.750179, "DP": 1.750179} for s in st["launch"]) and st["breakouts_mm"] == {"DN": 3.500357, "DP": 3.500357}
    # the uncoupled length is si.diff's: the breakouts plus the corner and the bumps (ai_eda.tools.routing.coupling), one definition
    _, unc_p, unc_n, _ = uncoupled_lengths([t for t in r.tracks if t.net == "DP"], [t for t in r.tracks if t.net == "DN"])
    assert st["uncoupled_mm"] == {"DP": pytest.approx(unc_p, abs=1e-6), "DN": pytest.approx(unc_n, abs=1e-6)} and min(unc_p, unc_n) > 3.500357
    assert st["coupled_mm"] == {"DN": 21.95, "DP": 22.888909} and st["skew_before_mm"] == pytest.approx(0.275 * (2 + math.sqrt(2)), abs=1e-6)
    comp = st["compensation"]
    assert comp["net"] == "DN" and comp["bumps"] == 2 and comp["amplitude_mm"] == pytest.approx(0.938909 / 4, abs=1e-6) and comp["at"] == [(4.25, 6.275), (19.725, 6.275)]
    assert st["skew_mm"] <= 1e-5 and st["length_mm"]["DP"] == pytest.approx(st["length_mm"]["DN"], abs=1e-5)
    dp = [(t.start, t.end) for t in r.tracks if t.net == "DP"]
    assert dp == [((3.0, 4.5), (4.25, 5.725)), ((4.25, 5.725), (20.0, 5.725)), ((20.0, 5.725), (20.275, 6.0)), ((20.275, 6.0), (20.275, 12.75)), ((20.275, 12.75), (21.5, 14.0))]
    dn = [(t.start, t.end) for t in r.tracks if t.net == "DN"]
    assert dn[0] == ((3.0, 7.5), (4.25, 6.275)) and dn[-2:] == [((19.725, 6.275), (19.725, 12.75)), ((19.725, 12.75), (18.5, 14.0))]
    # constant spacing: DN's coupled segments that are not bumps lie exactly 0.55 mm from DP's parallel segment
    flat = [s for s in dn[1:-1] if s[0][1] == s[1][1] == 6.275]
    assert _length_pairs(flat) == pytest.approx(19.725 - 4.25 - 2 * 0.5) and all(abs(6.275 - 5.725) - 0.55 < 1e-9 for _ in flat)
    assert 20.275 - 19.725 == pytest.approx(0.55)
    bumps = [s for s in dn[1:-1] if s not in flat and s[0][0] != 19.725]
    assert bumps and all(max(a[1], b[1]) <= 6.275 + comp["amplitude_mm"] + 1e-9 and min(a[1], b[1]) >= 6.275 for a, b in bumps)  # outward: away from DP
    assert {t.width_mm for t in r.tracks} == {0.3}
    assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
    for t in r.tracks:
        prov = t.provenance
        assert prov.tool_version == "0.3" and prov.derived_from[:3] == [f"net:{t.net}", "placement:J1", "placement:J2"]
        assert prov.derived_from[-1] == r.rules[t.net].derived_from_entry() and f"pair_partner={'DN' if t.net == 'DP' else 'DP'}" in prov.derived_from[-1]
        assert "coupled differential pair DN/DP: centreline of width 2w + s = 0.85 mm" in prov.note and "mitred corners" in prov.note
    # without the pair rules the two nets are ordinary 0.2 nets: nothing couples them
    plain = route_board(ir, lib)
    assert plain.version == "0.2" and "pairs" not in plain.stats and plain.unrouted == {}


def _length_pairs(segments) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments)


def test_a_pair_shares_the_board_with_other_nets_and_keeps_its_budgets(tmp_path: Path):
    """Two SMD nets cross the pair's straight way: the negotiation keeps them and the pair apart (the pair's envelope and breakouts
    are its halo), the exact audit passes, the validator passes; a breakout budget below 1.75 mm leaves no launch - both nets
    unrouted with the reason, no copper; a length budget below the pair's length unroutes both as well."""
    extra = [("R1", "SMD1", 12.0, 2.0), ("R2", "SMD1", 12.0, 10.0), ("R3", "SMD1", 8.0, 2.0), ("R4", "SMD1", 16.0, 16.0)]
    ir, lib = _pair_board(tmp_path, extra, {"X": [("R1", "1"), ("R2", "1")], "Y": [("R3", "1"), ("R4", "1")]})
    r = route_board(ir, lib, rules=PAIR_RULES)
    assert r.unrouted == {} and r.stats["pairs"]["DN/DP"]["reason"] is None and r.stats["pairs"]["DN/DP"]["skew_mm"] <= 1e-5
    assert _checks(ir, r, lib, tmp_path, 0.25) == {"pcb.routing.connectivity": "PASS", "pcb.routing.clearance": "PASS"}
    assert [row for row in _violations_of(ir, r, lib, "DP", 0.25)] == []
    tight = {n: NetRule(pair_partner=("DN" if n == "DP" else "DP"), pair_uncoupled_max_mm=1.5, **PAIR) for n in ("DP", "DN")}
    r = route_board(ir, lib, rules=tight)
    assert set(r.unrouted) == {"DN", "DP"} and r.unrouted["DN"] == r.unrouted["DP"] and {t.net for t in r.tracks} == {"X", "Y"}
    assert r.unrouted["DP"] == "no launch for the coupled pair DN/DP at J1.2/J1.1 within 10 mm: the breakout of DN would be 1.750 mm, over pair_uncoupled_max_mm 1.5"
    budget = {n: NetRule(pair_partner=("DN" if n == "DP" else "DP"), max_length_mm=20.0, **PAIR) for n in ("DP", "DN")}
    r = route_board(ir, lib, rules=budget)
    assert set(r.unrouted) == {"DN", "DP"} and "over max_length_mm 20" in r.unrouted["DP"] and {t.net for t in r.tracks} == {"X", "Y"}


def test_the_uncoupled_budget_is_one_number_per_net_for_the_router_and_si_diff(tmp_path: Path):
    """pair_uncoupled_max_mm bounds each net's uncoupled copper - both breakouts, the mitred corner and the bumps - the number
    si.diff judges (ai_eda.tools.routing.coupling): 3.0 mm fits either 1.75 mm breakout but not the two of a net (3.5 mm), 4.0 mm
    fits the breakouts but not the corner's copper as well, 6.0 mm fits all of it - and then si.diff's uncoupled rows pass."""
    from ai_eda.tools.si.measure import measure_nets
    from ai_eda.validation.si import diff_results
    from tests.test_si_checks import cls, default_classes, si_of, stacked, u
    from ai_eda.ir import DiffPair

    ir, lib = _pair_board(tmp_path)

    def rules(budget: float) -> dict[str, NetRule]:
        return {n: NetRule(pair_partner=("DN" if n == "DP" else "DP"), pair_uncoupled_max_mm=budget, **PAIR) for n in ("DP", "DN")}

    r = route_board(ir, lib, rules=rules(3.0))
    assert set(r.unrouted) == {"DN", "DP"} and "no two launches keep each net's breakouts within pair_uncoupled_max_mm 3" in r.unrouted["DP"] and r.tracks == []
    r = route_board(ir, lib, rules=rules(4.0))
    assert set(r.unrouted) == {"DN", "DP"} and "of uncoupled copper (breakouts, mitred corners, compensation), over pair_uncoupled_max_mm 4" in r.unrouted["DP"]
    r = route_board(ir, lib, rules=rules(6.0))
    assert r.unrouted == {}, r.unrouted
    st = r.stats["pairs"]["DN/DP"]
    assert 4.0 < max(st["uncoupled_mm"].values()) <= 6.0
    stacked(ir, 4)
    ir.pcb.tracks = list(r.tracks)
    ir.si = si_of(*default_classes(), cls("USB", nets=["DP", "DN"], pairs=[DiffPair(p="DP", n="DN")], target_zdiff_ohm=u(90.0, "ohm"), zdiff_tol_rel=u(0.5),
                                            pair_uncoupled_max_mm=u(6.0, "mm")))
    (diff,) = diff_results(ir, ir.si, measure_nets(ir))
    rows = {row["net"]: row for row in diff.details["rows"] if "uncoupled_mm" in row}
    assert {n: rows[n]["uncoupled_mm"] for n in ("DP", "DN")} == {n: pytest.approx(st["uncoupled_mm"][n], abs=1e-6) for n in ("DP", "DN")}
    assert all(rows[n]["status"] == "PASS" for n in ("DP", "DN"))


# --------------------------------------------------------------------------- determinism


def test_rules_route_deterministically(tmp_path: Path):
    """Neck-down, meanders and a coupled pair on fresh copies with fresh library instances: the same bytes every time."""
    runs = []
    for k in range(2):
        tmp = tmp_path / f"run{k}"
        ir, lib = _pair_board(tmp, [("R1", "PAD1", 2.0, 15.0), ("R2", "PAD1", 12.0, 15.0), ("R3", "PAD1", 2.0, 10.0), ("R4", "PAD1", 12.0, 12.5)],
                              {"D0": [("R1", "1"), ("R2", "1")], "D1": [("R3", "1"), ("R4", "1")]})
        rules = dict(PAIR_RULES)
        rules.update({n: NetRule(match_group="BUS", max_skew_mm=0.5, width_mm=0.3) for n in ("D0", "D1")})
        r = route_board(ir, lib, rules=rules)
        assert r.unrouted == {} and r.stats["match_groups"]["BUS"]["matched"], (r.unrouted, r.stats["match_groups"])
        runs.append(canonical(r) + repr(r.stats).encode())
    assert runs[0] == runs[1]


# --------------------------------------------------------------------------- review regressions (routing.maze 0.3)


#: a 15 x 11 mm board of SMD pads whose 4-pad net N3 routes at 10.25 mm without rules; with 2 % + 0.3 mm length budgets on the
#: multi-pad nets, N3 conflicts in iteration 1 and its reroute in iteration 2, grown at that iteration's costs, finds no tree
#: within 10.76 mm (found by a randomised sweep, seed 5 / trial 25)
FLIP_PARTS = [("R1", "SMD1", 10.0, 9.0), ("R2", "SMD1", 1.0, 2.0), ("R3", "SMD1", 13.0, 9.0), ("R4", "SMD1", 5.0, 2.0), ("R5", "SMD1", 6.0, 10.0),
              ("R6", "SMD1", 6.0, 3.0), ("R7", "SMD1", 14.0, 1.0), ("R8", "SMD1", 4.0, 8.0), ("R9", "SMD1", 1.0, 5.0), ("R10", "SMD1", 9.0, 4.0),
              ("R11", "SMD1", 6.0, 6.0), ("R12", "SMD1", 4.0, 6.0), ("R13", "SMD1", 8.0, 1.0), ("R14", "SMD1", 12.0, 2.0), ("R15", "SMD1", 13.0, 4.0),
              ("R16", "SMD1", 12.0, 5.0), ("R17", "SMD1", 2.0, 4.0), ("R18", "SMD1", 14.0, 6.0), ("R19", "SMD1", 5.0, 7.0)]
FLIP_NETS = {"N0": [("R1", "1"), ("R2", "1"), ("R3", "1"), ("R4", "1")], "N1": [("R5", "1"), ("R6", "1"), ("R7", "1")], "N2": [("R8", "1"), ("R9", "1")],
             "N3": [("R10", "1"), ("R11", "1"), ("R12", "1"), ("R13", "1")], "N4": [("R14", "1"), ("R15", "1")], "N5": [("R16", "1"), ("R17", "1")],
             "N6": [("R18", "1"), ("R19", "1")]}


def test_a_budgeted_net_keeps_its_route_when_a_later_reroute_misses_the_budget(tmp_path: Path):
    """The plain board meets every budget (N3 at 10.25 mm <= 10.76 mm); a reroute that fails at one iteration's costs keeps the
    earlier route (within its budget) instead of losing the net for good - N3 ends routed within 10.76 mm on a legal board."""
    lib = rules_library(tmp_path / "kicad")
    ir = board_ir(tmp_path, lib, FLIP_PARTS, FLIP_NETS, (15.0, 11.0))
    p = RoutingParams(max_iterations=12)
    plain = route_board(ir, lib, p)
    budgets = {"N0": 24.78, "N1": 19.17, "N3": 10.76}
    assert all(plain.stats["net_length_mm"][n] <= b for n, b in budgets.items())
    r = route_board(ir, lib, p, rules={n: NetRule(max_length_mm=b) for n, b in budgets.items()})
    assert r.stats["legal"] and r.stats["dropped"] == [] and "N3" not in r.unrouted, r.unrouted
    assert set(r.unrouted) == set(plain.unrouted)  # only the net the plain router cannot reach either (N6)
    assert all(r.stats["net_length_mm"][n] <= b + 1e-6 for n, b in budgets.items())


def test_a_meander_is_never_drawn_on_the_nets_own_pad(tmp_path: Path):
    """D0 leaves a 5 x 5 mm pad (R1) for a 4 mm run; D1 is 6 mm. The only straight run long enough starts inside R1's pad: a bump
    there would lie on the pad (shorted: no electrical length), so the group is honestly not matched - no copper on R1's pad."""
    lib = rules_library(tmp_path / "kicad")
    parts = [("R1", "BIG5", 4.0, 5.0), ("R2", "SMD1", 8.0, 5.0), ("R3", "SMD1", 4.0, 11.0), ("R4", "SMD1", 8.0, 13.0)]
    ir = board_ir(tmp_path, lib, parts, {"D0": [("R1", "1"), ("R2", "1")], "D1": [("R3", "1"), ("R4", "1")]}, (12.0, 16.0))
    r = route_board(ir, lib, rules={n: NetRule(match_group="G", max_skew_mm=0.5) for n in ("D0", "D1")})
    g = r.stats["match_groups"]["G"]
    assert g["matched"] is False and g["meanders"] == {} and "no straight run holds" in g["reasons"]["D0"]
    pad = (1.5, 2.5, 6.5, 7.5)
    on_pad = [t for t in r.tracks if t.net == "D0" and all(pad[0] <= x <= pad[2] and pad[1] <= y <= pad[3] for x, y in (t.start, t.end))]
    assert on_pad == []
