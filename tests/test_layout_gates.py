"""The schematic layout gates (``layout_metrics.schematic_gates``) on hand-made sheets with known answers."""

from __future__ import annotations

import pytest

from ai_eda.ir import Block, CircuitIR, Net, PinRef, ProjectMeta, Provenance, ProvenanceKind, Topology
from ai_eda.tools.layout_metrics import METRICS_VERSION, rectilinear_mst_length, schematic_gates

#: a two-pin test symbol: body x -1 .. 1, y -2.54 .. 2.54; pin 1 at (0, -3.81) above, pin 2 at (0, 3.81) below (schematic frame)
LIB = '''(lib_symbols (symbol "T:R" (pin_numbers (hide yes)) (pin_names (hide yes))
  (symbol "R_0_1" (rectangle (start -1 2.54) (end 1 -2.54)))
  (symbol "R_1_1" (pin passive line (at 0 3.81 270) (length 1.27) (name "~") (number "1"))
                  (pin passive line (at 0 -3.81 90) (length 1.27) (name "~") (number "2")))))'''


def symbol(ref: str, x: float, y: float) -> str:
    return f'(symbol (lib_id "T:R") (at {x} {y} 0) (unit 1) (property "Reference" "{ref}" (at {x} {y} 0) (effects (font (size 1.27 1.27)) hide)))'


def wire(*points: tuple[float, float]) -> str:
    return "(wire (pts " + " ".join(f"(xy {x} {y})" for x, y in points) + "))"


def label(name: str, x: float, y: float, angle: int = 90) -> str:
    return f'(global_label "{name}" (shape passive) (at {x} {y} {angle}) (effects (font (size 1.27 1.27))))'


def sheet(tmp_path, *items: str):
    path = tmp_path / "gates.kicad_sch"
    path.write_text('(kicad_sch (paper "A4") ' + LIB + " " + " ".join(items) + ")", encoding="utf-8")
    return path


def test_metrics_version_is_two():
    assert METRICS_VERSION == "2"


def test_a_wire_one_millimetre_from_a_foreign_body_is_one_near_hit(tmp_path):
    # R1's body ends at x = 51.8; the wire runs alongside at x = 52.8, on no pin of R1
    gates = schematic_gates(sheet(tmp_path, symbol("R1", 50.8, 50.8), wire((52.8, 45.72), (52.8, 55.88))))
    assert gates["near_symbol_hits"] == 1
    gates = schematic_gates(sheet(tmp_path, symbol("R1", 50.8, 50.8), wire((53.34, 45.72), (53.34, 55.88))))
    assert gates["near_symbol_hits"] == 0  # 1.54 mm clear
    # the symbol's own net does not count: a wire on R1's pin 1 passing close to its body
    gates = schematic_gates(sheet(tmp_path, symbol("R1", 50.8, 50.8), wire((50.8, 46.99), (52.8, 46.99), (52.8, 55.88))))
    assert gates["near_symbol_hits"] == 0


def test_two_nets_parallel_at_one_grid_step_are_one_parallel_hit(tmp_path):
    gates = schematic_gates(sheet(tmp_path, wire((10, 20), (30, 20)), wire((15, 21.27), (40, 21.27))))
    assert gates["parallel_hits"] == 1
    gates = schematic_gates(sheet(tmp_path, wire((10, 20), (30, 20)), wire((15, 22.54), (40, 22.54))))
    assert gates["parallel_hits"] == 0  # 2.54 mm is the minimum, not a hit
    gates = schematic_gates(sheet(tmp_path, wire((10, 20), (30, 20)), wire((31, 21.27), (40, 21.27))))
    assert gates["parallel_hits"] == 0  # side by side without overlapping runs
    gates = schematic_gates(sheet(tmp_path, wire((10, 20), (30, 20)), wire((15, 21.27), (40, 21.27)), wire((30, 20), (30, 21.27))))
    assert gates["parallel_hits"] == 0  # one net


def test_rmst_ratio_median_distance_and_label_reach(tmp_path):
    # net A: R1.2 - R2.2 wired around (25.4 mm of wire over a 20.32 mm rectilinear MST); net B: R2.1 - R3.1 by two labels
    # (no wire, MST 30.48 mm); connected parts R1-R2 20.32 mm, R2-R3 30.48 mm apart
    items = [symbol("R1", 50.8, 50.8), symbol("R2", 71.12, 50.8), symbol("R3", 101.6, 50.8),
             wire((50.8, 54.61), (50.8, 57.15), (71.12, 57.15), (71.12, 54.61)),
             label("B", 71.12, 46.99), label("B", 101.6, 46.99)]
    gates = schematic_gates(sheet(tmp_path, *items))
    assert gates["wire_mm"] == pytest.approx(25.4)
    assert gates["wire_rmst_ratio"] == pytest.approx(25.4 / (20.32 + 30.48))
    assert gates["median_connected_mm"] == pytest.approx((20.32 + 30.48) / 2)
    assert gates["labels_per_net"] == {"B": 2}
    assert gates["label_joined_nets"] == 1
    assert gates["pins_reached_by_label_pct"] == 50.0  # R2.1 and R3.1 of the four pins on two-pin nets
    assert gates["crossings"] == 0 and gates["paper"] == "A4"
    assert gates["supply_above_ground"] == {"ok": 0, "of": 0} and gates["blocks_left_to_right"] is None
    x0, y0, x1, y1 = gates["content_box"]
    assert x0 <= 49.8 and x1 >= 102.6 and y1 >= 57.15


def _ir(nets: dict[str, tuple[str, list[tuple[str, str]]]], blocks: list[tuple[str, list[str], list[str]]] = ()) -> CircuitIR:
    """An IR of nets ``{name: (kind, [(ref, pin)])}`` and blocks ``(id, input nets, output nets)`` over R1 and R2."""
    prov = Provenance(kind=ProvenanceKind.ASSUMPTION)
    topology = Topology(name="t", provenance=prov, blocks=[Block(id=b, function=b, component_refs=["R1", "R2"], input_nets=i, output_nets=o, provenance=prov)
                                                           for b, i, o in blocks]) if blocks else None
    return CircuitIR(project=ProjectMeta(id="gates", name="gates"), topology=topology,
                     nets=[Net(name=n, kind=k, provenance=prov, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins]) for n, (k, pins) in nets.items()])


def test_supply_above_ground_and_blocks_left_to_right(tmp_path):
    # R1 at x 50.8 and R2 at x 76.2: pin 1 on top, pin 2 below
    path = sheet(tmp_path, symbol("R1", 50.8, 50.8), symbol("R2", 76.2, 50.8))
    powered = _ir({"VCC": ("power", [("R1", "1"), ("R2", "2")]), "GND": ("ground", [("R1", "2"), ("R2", "1")])})
    gates = schematic_gates(path, powered)
    assert gates["supply_above_ground"] == {"ok": 1, "of": 2}  # R2 stands on its head
    assert gates["blocks_left_to_right"] is None
    flow = _ir({"IN": ("signal", [("R1", "1")]), "MID": ("signal", [("R1", "2"), ("R2", "1")]), "OUT": ("signal", [("R2", "2")])},
               [("forward", ["IN"], ["OUT"]), ("backward", ["OUT"], ["IN"])])
    assert schematic_gates(path, flow)["blocks_left_to_right"] == {"ok": 1, "of": 2}


def test_rectilinear_mst():
    assert rectilinear_mst_length([(0, 0), (3, 4), (3, 4), (3, 0)]) == 7
    assert rectilinear_mst_length([(1, 1)]) == rectilinear_mst_length([]) == 0
