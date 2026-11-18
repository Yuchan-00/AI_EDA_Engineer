"""Metric definitions on synthetic geometry, plus three measured frozen baselines."""

import gzip
import hashlib
import json
import math

import pytest

from ai_eda.ir import BoardOutline, Layer, Net, PCBDesign, PinRef, Placement, Track, Via, Zone
from ai_eda.tools.layout_metrics import mst_length, pcb_metrics, schematic_metrics
from tests.layout_baseline_data import DATA, load_geometry
from tests.test_pcb_agent import parts_ir
from tests.test_parts_existence import synthetic_library


def schematic(tmp_path, extra=""):
    path = tmp_path / "small.kicad_sch"
    path.write_text('''(kicad_sch
      (lib_symbols (symbol "Test:S" (symbol "S_1_1" (rectangle (start -2 -2) (end 2 2)))))
      (symbol (lib_id "Test:S") (at 30 30 0) (unit 1)
        (property "Value" "hidden" (at 30 30 0) (effects (font (size 1 1)) hide)))
      (symbol (lib_id "Test:S") (at 31 30 90) (unit 1))
      (text "A" (at 30 30 0) (effects (font (size 1 1))))
      (text "A" (at 30 30 0) (effects (font (size 1 1))))
      (global_label "GLOBAL" (at 80 80 0) (effects (font (size 1 1))))
      (label "LOCAL" (at 90 90 90) (effects (font (size 1 1))))
      (wire (pts (xy 0 0) (xy 10 0) (xy 10 10)))
      (wire (pts (xy 4 -2) (xy 4 2)))
    ''' + extra + ")", encoding="utf-8")
    return path


def test_schematic_overlap_counts_wire_segments_crossings_bends_and_labels(tmp_path):
    metrics = schematic_metrics(schematic(tmp_path))
    assert metrics == {"symbol_symbol_overlaps": 1, "symbol_text_overlaps": 4, "text_text_overlaps": 1, "overlaps": 6,
                       "global_label_count": 1, "local_label_count": 1, "wire_segment_count": 3, "wire_length_mm": 24.0,
                       "wire_crossings": 1, "bends": 1}
    assert schematic_metrics(schematic(tmp_path)) == metrics


def test_junctions_are_not_crossings_and_t_joints_are_not_bends(tmp_path):
    metrics = schematic_metrics(schematic(tmp_path, '(junction (at 4 0)) (wire (pts (xy 10 0) (xy 12 0)))'))
    assert metrics["wire_crossings"] == metrics["bends"] == 0
    assert metrics["wire_length_mm"] == 26


def test_missing_symbol_refuses_instead_of_measuring_a_placeholder(tmp_path):
    path = tmp_path / "bad.kicad_sch"
    path.write_text('(kicad_sch (symbol (lib_id "Missing:X") (at 0 0)))')
    with pytest.raises(ValueError, match="metrics.symbol"):
        schematic_metrics(path)


def test_mst_is_euclidean_and_order_independent():
    points = [(0, 0), (3, 0), (3, 4), (3, 4)]
    assert mst_length(points) == mst_length(list(reversed(points))) == 7
    assert mst_length([]) == mst_length([(1, 2)]) == 0
    assert mst_length([(0, 0), (3, 4)]) == 5
    with pytest.raises(ValueError, match="non-finite"):
        mst_length([(math.inf, 0)])


def test_pcb_geometry_plane_partition_and_missing_placement(tmp_path):
    lib = synthetic_library(tmp_path / "library")
    ir = parts_ir(tmp_path, lib, ("R1", "R2"), nets=False)
    prov = ir.components[0].provenance
    ir.nets = [Net(name=name, pins=[PinRef(component_ref=r, pin_number=pin) for r in ("R1", "R2")], provenance=prov)
               for name, pin in (("SIGNAL", "1"), ("GND", "2"))]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=20, height_mm=10),
                       placements=[Placement(component_ref="R1", x_mm=2, y_mm=2), Placement(component_ref="R2", x_mm=5, y_mm=6)],
                       layers=[Layer(name="F.Cu", kind="signal"), Layer(name="B.Cu", kind="signal")],
                       tracks=[Track(net="SIGNAL", layer="F.Cu", start=(0, 0), end=(3, 4), width_mm=0.25)],
                       vias=[Via(net="SIGNAL", x_mm=3, y_mm=4, diameter_mm=0.6, drill_mm=0.3)],
                       zones=[Zone(net="GND", layer="B.Cu", polygon=[(0, 0), (20, 0), (20, 10), (0, 10)])])
    metrics = pcb_metrics(ir, lib)
    assert metrics["board_area_mm2"] == 200
    assert metrics["footprint_bbox_area_mm2"] == 5.76
    assert metrics["airwire_mst_mm"] == 10
    assert metrics["signal_airwire_mst_mm"] == metrics["plane_airwire_mst_mm"] == 5
    assert metrics["airwire_mst_per_net_mm"] == {"GND": 5, "SIGNAL": 5}
    assert metrics["routed_track_length_mm"] == 5 and metrics["via_count"] == 1 and metrics["layer_count"] == 2
    ir.pcb.placements.reverse()
    ir.nets.reverse()
    assert pcb_metrics(ir, lib) == metrics
    ir.pcb.placements.pop()
    with pytest.raises(ValueError, match="metrics.placement"):
        pcb_metrics(ir, lib)


@pytest.mark.parametrize("name", ["stm32_display", "atmega128_devboard", "divider"])
def test_frozen_baselines_are_remeasured_offline(tmp_path, name):
    expected = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))["boards"][name]
    raw = gzip.decompress((DATA / f"{name}.kicad_sch.gz").read_bytes())
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == expected["schematic_sha256"]
    geometry = gzip.decompress((DATA / f"{name}.json.gz").read_bytes())
    assert "sha256:" + hashlib.sha256(geometry).hexdigest() == expected["geometry_sha256"]
    path = tmp_path / "baseline.kicad_sch"
    path.write_bytes(raw)
    assert schematic_metrics(path) == expected["schematic"]
    ir, library = load_geometry(name)
    assert pcb_metrics(ir, library) == expected["pcb"]
