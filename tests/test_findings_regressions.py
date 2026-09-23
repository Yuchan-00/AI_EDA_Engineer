"""Regression tests for the verifier findings on the vertical slice.

Each test names the defect it pins down. Tests that need the KiCad
libraries or ``kicad-cli`` are skipped when those are absent and otherwise
run against the real binary; everything else runs everywhere.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from ai_eda.agents import AgentContext
from ai_eda.agents.repair import RepairAgent
from ai_eda.compilers import BOMCompiler, CPLCompiler, CompileContext, GerberExporter, PCBCompiler, SchematicCompiler
from ai_eda.compilers.pins import pad_pin_types
from ai_eda.compilers.schematic_layout import Extent, layout_pitch, layout_positions, symbol_extent
from ai_eda.errors import CompileError, ToolExecutionError
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    BoardSide,
    CircuitIR,
    Component,
    Layer,
    LibraryRef,
    Net,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Track,
    ValidationResult,
    ValidationStatus,
    Zone,
    authoritative,
    hash_file_set,
)
from ai_eda.repair import RepairAction, RepairLoop, RepairStrategy
from ai_eda.repair.loop import MUTATION_STOP
from ai_eda.repair.strategies import RerunTool
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad import cli as kicad_cli
from ai_eda.tools.kicad import library as kicad_library
from ai_eda.tools.kicad.board import read_board_footprints
from ai_eda.tools.manufacturing.outputs import check_gerber_set, check_output_artifact, required_functions
from ai_eda.tools.routing import route_naive
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow import Orchestrator, Stage
from tests.conftest import AUTH, DS, make_component
from tests.fixtures_kicad import HAND_PLACED, HEADER_DS, ROUTABLE_PLACEMENTS, divider_with_connector_ir, ir_net_map
from tests.test_schematic_compiler import COMPILER_DEFECT_WARNINGS, netlist_nets

S = ValidationStatus
LIB = KicadLibrary()
HAS_LIBS = LIB.symbol_file("Device") is not None and LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")


def _review(ir: CircuitIR, workdir: Path, area: ReviewArea) -> ValidationResult:
    return next(r for r in IndependentReviewer().review(ir, workdir).results if r.check_id == area)


def _compile_board(ir: CircuitIR, workdir: Path) -> ArtifactRef:
    ctx = CompileContext(workdir=workdir, tools={"kicad_library": LIB})
    ir.artifacts[ArtifactKind.SCHEMATIC] = SchematicCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.CPL] = CPLCompiler().compile(ir, ctx)
    return ir.artifacts[ArtifactKind.PCB]


def _routed(tmp_path: Path) -> CircuitIR:
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.tracks = route_naive(ir, LIB)
    return ir


# --------------------------------------------------------------------------- findings 1 / 10: BOM / CPL vs the board


@needs_libs
def test_pcb_vs_cpl_fails_when_a_placement_moves_after_the_cpl_was_written(tmp_path: Path):
    ir = _routed(tmp_path)
    _compile_board(ir, tmp_path)
    assert _review(ir, tmp_path, ReviewArea.PCB_VS_CPL).status is S.PASS
    ir.pcb.placement("R1").x_mm = 16.0  # design change: the CPL now says 14.0 mm
    r = _review(ir, tmp_path, ReviewArea.PCB_VS_CPL)
    assert r.status is S.FAIL and r.details["repair"] == "regenerate" and r.details["artifact"] is ArtifactKind.CPL
    assert "older IR" in r.message


@needs_libs
def test_pcb_vs_cpl_and_bom_fail_on_hand_edited_csv(tmp_path: Path):
    ir = _routed(tmp_path)
    _compile_board(ir, tmp_path)
    cpl = Path(ir.artifacts[ArtifactKind.CPL].path)
    cpl.write_text(cpl.read_text(encoding="utf-8").replace("14.0000mm", "99.0000mm"), encoding="utf-8")
    r = _review(ir, tmp_path, ReviewArea.PCB_VS_CPL)
    assert r.status is S.FAIL and r.details == {"artifact": ArtifactKind.CPL, "repair": "regenerate"} and "recorded hash" in r.message
    bom = Path(ir.artifacts[ArtifactKind.BOM].path)
    bom.write_text(bom.read_text(encoding="utf-8").replace("RC0603FR-0710kL", "WRONG-MPN"), encoding="utf-8")
    r = _review(ir, tmp_path, ReviewArea.PCB_VS_BOM)
    assert r.status is S.FAIL and r.details == {"artifact": ArtifactKind.BOM, "repair": "regenerate"} and "recorded hash" in r.message
    assert r.evidence[0].path == str(bom) and r.evidence[0].content_hash  # evidence names the csv that was read


@needs_libs
def test_pcb_vs_bom_and_cpl_compare_rows_with_the_compiled_board(tmp_path: Path):
    ir = _routed(tmp_path)
    pcb = _compile_board(ir, tmp_path)
    for area, what in ((ReviewArea.PCB_VS_BOM, "reference, value, footprint"), (ReviewArea.PCB_VS_CPL, "position, rotation, side")):
        r = _review(ir, tmp_path, area)
        assert r.status is S.PASS and what in r.message, r.message
        assert [e.path for e in r.evidence] == [ir.artifacts[ArtifactKind.BOM if area is ReviewArea.PCB_VS_BOM else ArtifactKind.CPL].path, pcb.path]
        assert all(e.content_hash for e in r.evidence)
    fps = {fp.ref: fp for fp in read_board_footprints(Path(pcb.path))}
    assert fps["R1"].rotation == -90.0 and fps["R1"].x == 14.0 and fps["R1"].side == "Top" and fps["R1"].lib_id == "Resistor_SMD:R_0603_1608Metric"
    # a board that disagrees with a fresh CPL/BOM is a FAIL for a human, not a regenerate: rewrite the board file
    # and re-register it so both artifacts look fresh
    text = Path(pcb.path).read_text(encoding="utf-8")
    Path(pcb.path).write_text(text.replace("(at 14 6 -90)", "(at 15 6 -90)", 1).replace('(property "Value" "10k"', '(property "Value" "47k"', 1), encoding="utf-8")
    pcb.content_hash = pcb.disk_hash()
    r = _review(ir, tmp_path, ReviewArea.PCB_VS_CPL)
    assert r.status is S.FAIL and r.details["repair"] == "human" and any("position CPL (14.0, 6.0) vs board (15.0, 6.0)" in m for m in r.details["mismatches"])
    r = _review(ir, tmp_path, ReviewArea.PCB_VS_BOM)
    assert r.status is S.FAIL and r.details["repair"] == "human" and any("value BOM '10k' vs board '47k'" in m for m in r.details["mismatches"])


def test_pcb_vs_bom_without_a_board_is_not_verified_and_stale_board_asks_for_regeneration(divider_ir: CircuitIR, tmp_path: Path):
    ctx = CompileContext(workdir=tmp_path)
    divider_ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(divider_ir, ctx)
    r = _review(divider_ir, tmp_path, ReviewArea.PCB_VS_BOM)
    assert r.status is S.NOT_VERIFIED and "no PCB artifact" in r.message
    # a component that exists in the IR but was never placed: the BOM lists it, no board proves it -> not PASS
    divider_ir.components.append(make_component("R9", "1k"))
    divider_ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(divider_ir, ctx)
    assert _review(divider_ir, tmp_path, ReviewArea.PCB_VS_BOM).status is S.NOT_VERIFIED
    # a board artifact from another IR version is not a comparison base either
    stale = ArtifactRef(kind=ArtifactKind.PCB, path=str(tmp_path / "x.kicad_pcb"), content_hash="sha256:x", generated_from_ir_hash="sha256:older")
    (tmp_path / "x.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    divider_ir.artifacts[ArtifactKind.PCB] = stale
    r = _review(divider_ir, tmp_path, ReviewArea.PCB_VS_BOM)
    assert r.status is S.FAIL and r.details == {"artifact": ArtifactKind.PCB, "repair": "regenerate"} and "cannot compare with the board" in r.message


# --------------------------------------------------------------------------- finding 2: symbol overlap on the schematic grid


def _header_2x20(ref: str) -> Component:
    sym = LIB.load_symbol(LibraryRef(library="Connector_Generic", name="Conn_02x20_Odd_Even"))
    auth = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=HEADER_DS)
    return Component(
        ref=ref,
        value="Conn_02x20",
        description="header",
        mpn=authoritative("X", HEADER_DS),
        package=authoritative("2x20", HEADER_DS),
        pins=[Pin(number=p.number, name=p.name, electrical_type=PinElectricalType(p.electrical_type), provenance=auth) for p in sym.pins],
        symbol=LIB.resolve_symbol(LibraryRef(library="Connector_Generic", name="Conn_02x20_Odd_Even")),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test"),
    )


def _tall_headers_ir(tmp_path: Path, count: int = 5) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="tall", name="tall", workdir=str(tmp_path)))
    refs = [f"J{i}" for i in range(1, count + 1)]
    ir.components = [_header_2x20(r) for r in refs]
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
    ir.nets = [Net(name=f"N{k}", pins=[PinRef(component_ref=r, pin_number=str(k)) for r in refs], provenance=net_p) for k in range(1, 41)]
    return ir


def test_layout_pitch_grows_with_the_symbols_and_never_below_the_default():
    small = Extent(-3.81, -13.97, 3.81, 15.24)
    assert layout_pitch([small, small]) == (25.4, 35.56)
    tall = Extent(-15.24, -26.67, 17.78, 29.21)
    px, py = layout_pitch([tall] * 5)
    assert px >= tall.width + 5.08 and py >= tall.height + 5.08 and px % 2.54 == pytest.approx(0) and py % 2.54 == pytest.approx(0)
    # asymmetric neighbours: the pitch covers the worst pairing, not just each box's own width
    left, right = Extent(9, 0, 10, 1), Extent(-10, 0, -9, 1)
    px, _ = layout_pitch([left, right])
    assert px >= 10 - (-10) + 5.08
    pos = layout_positions({"A": left, "B": right}, columns=2)
    assert not left.shifted(*pos["A"]).overlaps(right.shifted(*pos["B"]))


@needs_libs
def test_symbol_extent_covers_body_pins_stubs_and_labels():
    r = LIB.load_symbol(LibraryRef(library="Device", name="R"))
    bare = symbol_extent(r, {})
    labelled = symbol_extent(r, {"1": 3, "2": 4})
    assert bare.height < labelled.height and labelled.ymin < -3.81 - 2.54 and labelled.ymax > 3.81 + 2.54
    c = LIB.load_symbol(LibraryRef(library="Connector_Generic", name="Conn_02x20_Odd_Even"))
    e = symbol_extent(c, {str(k): 3 for k in range(1, 41)})
    assert e.ymin <= -25.4 and e.ymax >= 22.86 and e.height > 25.4  # taller than the old fixed pitch


@needs_libs
def test_tall_symbols_are_spaced_so_no_connection_points_coincide(tmp_path: Path):
    ir = _tall_headers_ir(tmp_path)
    node = SchematicCompiler().build(ir, LIB)
    points: list[tuple[str, str]] = []
    for w in sexpr.find_all(node, "wire"):
        points += [(str(xy[1]), str(xy[2])) for xy in sexpr.find_all(sexpr.find(w, "pts"), "xy")]
    assert len(points) == len(set(points)) == 2 * 40 * 5
    symbols = {str(sexpr.get(s, "property", 2)): sexpr.find(s, "at") for s in sexpr.find_all(node, "symbol")}
    assert float(symbols["J5"][2]) - float(symbols["J1"][2]) > 25.4  # second row is further down than the old pitch


@needs_kicad
def test_real_netlist_of_tall_symbols_reproduces_every_ir_net(tmp_path: Path):
    ir = _tall_headers_ir(tmp_path)
    ref = SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    erc = kicad.run_erc(Path(ref.path), tmp_path / "erc.json")
    assert erc.details["errors"] == [] and not [w for w in erc.details["warnings"] if w.get("type") in COMPILER_DEFECT_WARNINGS | {"multiple_net_names"}]
    nets = netlist_nets(kicad.export_netlist(Path(ref.path), tmp_path / "tall.net"))
    assert nets == ir_net_map(ir) and len(nets) == 40 and all(len(pins) == 5 for pins in nets.values())


@needs_libs
def test_stacked_pins_are_refused_instead_of_silently_merged(tmp_path: Path):
    lib_ref = LibraryRef(library="MCU_ST_STM32F1", name="STM32F103C8Tx")
    sym = LIB.load_symbol(lib_ref)
    auth = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)
    mcu = Component(
        ref="U1",
        value="STM32",
        pins=[Pin(number=p.number, name=p.name, electrical_type=PinElectricalType(p.electrical_type), provenance=auth) for p in sym.pins],
        symbol=LIB.resolve_symbol(lib_ref),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test"),
    )
    ir = CircuitIR(project=ProjectMeta(id="mcu", name="mcu", workdir=str(tmp_path)), components=[mcu])
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
    ir.nets = [Net(name=f"N{p.number}", pins=[PinRef(component_ref="U1", pin_number=p.number)], provenance=net_p) for p in sym.pins]
    with pytest.raises(CompileError, match="coincides with"):
        SchematicCompiler().build(ir, LIB)


# --------------------------------------------------------------------------- finding 3: pin electrical types from the library


@needs_libs
def test_ir_pin_type_that_differs_from_the_library_is_refused_by_both_compilers(tmp_path: Path):
    ir = _routed(tmp_path)
    r1 = ir.component("R1")
    r1.pins[0] = r1.pins[0].model_copy(update={"electrical_type": PinElectricalType.INPUT})
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": LIB})
    with pytest.raises(CompileError, match="R1: IR pin electrical types differ .*pin 1: IR input vs library passive"):
        SchematicCompiler().compile(ir, ctx)
    with pytest.raises(CompileError, match="pin 1: IR input vs library passive"):
        PCBCompiler().compile(ir, ctx)
    assert not list(tmp_path.glob("*.kicad_*"))


@needs_libs
def test_board_pintype_comes_from_the_library_and_no_connect_is_written_like_kicad(tmp_path: Path):
    ir = _routed(tmp_path)
    j1 = ir.component("J1")
    j1.pins[2] = j1.pins[2].model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT})
    ir.net("GND").pins = [p for p in ir.net("GND").pins if p.component_ref != "J1"]
    ir.pcb.tracks = route_naive(ir, LIB)
    assert pad_pin_types(j1, LIB.load_symbol(j1.symbol)) == {"1": "passive", "2": "passive", "3": "passive+no_connect"}
    ref = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    tree = sexpr.parse_file(Path(ref.path))
    j1_fp = next(fp for fp in sexpr.find_all(tree, "footprint") if any(p[1] == "Reference" and p[2] == "J1" for p in sexpr.find_all(fp, "property")))
    assert {str(p[1]): sexpr.get(p, "pintype") for p in sexpr.find_all(j1_fp, "pad")} == {"1": "passive", "2": "passive", "3": "passive+no_connect"}
    ir.component("R1").symbol = None
    with pytest.raises(CompileError, match="R1: no KiCad symbol"):
        PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "nosym", tools={"kicad_library": LIB}))


# --------------------------------------------------------------------------- finding 4: bottom-side pad flip rules


def _with_footprint(ir: CircuitIR, ref: str, library: str, name: str, side: BoardSide, x: float, y: float) -> None:
    fp = LIB.load_footprint(LibraryRef(library=library, name=name))
    comp = ir.component(ref)
    numbers = sorted({p.number for p in fp.pads if p.number and p.pad_type != "np_thru_hole"})
    assert numbers == ["1", "2"], numbers  # two-pad parts stand in for the resistor
    comp.footprint = LIB.resolve_footprint(LibraryRef(library=library, name=name))
    ir.pcb.placement(ref).x_mm, ir.pcb.placement(ref).y_mm, ir.pcb.placement(ref).side = x, y, side
    ir.pcb.tracks = []


@needs_libs
def test_bottom_side_pad_flip_swaps_chamfer_corners_and_mirrors_rect_delta(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    _with_footprint(ir, "R1", "Inductor_SMD", "L_Bourns_SDR0604", BoardSide.BOTTOM, 20.0, 8.0)
    ir.pcb.placement("R2").y_mm = 16.0
    ref = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": LIB}))
    text = Path(ref.path).read_text(encoding="utf-8")
    assert "(chamfer bottom_left bottom_right)" in text and "(chamfer top_left top_right)" not in text
    lib_text = Path(LIB.footprint_file("Inductor_SMD", "L_Bourns_SDR0604")).read_text(encoding="utf-8")
    assert "(chamfer top_left top_right)" in lib_text  # the library really says top_*
    if kicad.available():
        res = kicad.run_drc(Path(ref.path), tmp_path / "drc.json")
        assert not [v for v in res.details["warnings"] + res.details["errors"] if v.get("type") == "lib_footprint_mismatch"], res.message
    # rect_delta (trapezoid pads): delta y is mirrored on the bottom
    node = sexpr.parse('(pad "1" smd trapezoid (at 0 1) (size 1 2) (rect_delta 0 0.3) (layers "F.Cu"))')
    pl = Placement(component_ref="X", x_mm=0, y_mm=0, side=BoardSide.BOTTOM)
    out = PCBCompiler._pad("p", make_component("X", "1"), pl, node, {}, {"1": "passive"})
    assert [sexpr.to_float(a) for a in sexpr.args(sexpr.find(out, "rect_delta"))] == [0.0, -0.3]
    assert [sexpr.to_float(a) for a in sexpr.args(sexpr.find(out, "at"))] == [0.0, -1.0]
    assert sexpr.args(sexpr.find(out, "layers")) == ["B.Cu"]


def test_pad_child_without_a_flip_rule_is_refused_on_the_bottom_side():
    node = sexpr.parse('(pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (zone_layer_connections "F.Cu"))')
    comp = make_component("X", "1")
    top = Placement(component_ref="X", x_mm=0, y_mm=0, side=BoardSide.TOP)
    assert sexpr.find(PCBCompiler._pad("p", comp, top, node, {}, {}), "zone_layer_connections") is not None  # top: copied
    bottom = Placement(component_ref="X", x_mm=0, y_mm=0, side=BoardSide.BOTTOM)
    with pytest.raises(CompileError, match=r"no flip rule for its \(zone_layer_connections \.\.\.\) child"):
        PCBCompiler._pad("p", comp, bottom, node, {}, {})
    weird = sexpr.parse('(pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (chamfer top_left sideways))')
    with pytest.raises(CompileError, match="unknown chamfer corner"):
        PCBCompiler._pad("p", comp, bottom, weird, {}, {})


# --------------------------------------------------------------------------- finding 5: KiCad version directories


def test_kicad_version_directories_sort_numerically_and_cli_install_comes_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "Programs" / "KiCad"
    for ver in ("9.0", "8.0", "10.0", "junk"):
        (base / ver / "bin").mkdir(parents=True)
        (base / ver / "bin" / "kicad-cli.exe").write_bytes(b"")
        (base / ver / "share" / "kicad").mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("KICAD10_SYMBOL_DIR", raising=False)
    monkeypatch.delenv("KICAD_SYMBOL_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert [p.name for p in kicad_cli.version_dirs(base)] == ["10.0", "9.0", "8.0", "junk"]
    assert kicad_cli.find_kicad_cli() == str(base / "10.0" / "bin" / "kicad-cli.exe")
    assert [str(r) for r in kicad_library._default_library_roots()] == [str(base / v / "share" / "kicad") for v in ("10.0", "9.0", "8.0", "junk")]
    # kicad-cli 9 on PATH: its libraries are the ones its ERC/DRC compares against, so they come first
    monkeypatch.setattr(shutil, "which", lambda name: str(base / "9.0" / "bin" / "kicad-cli.exe"))
    assert kicad_library._default_library_roots()[0] == base / "9.0" / "share" / "kicad"
    assert kicad_cli.install_root(str(base / "9.0" / "bin" / "kicad-cli.exe")) == base / "9.0"
    assert kicad_cli.install_root(str(tmp_path / "elsewhere" / "kicad-cli.exe")) is None


# --------------------------------------------------------------------------- findings 6 / 11: proposals are not evidence


def test_agent_stage_with_proposals_only_is_not_verified(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    ctx = AgentContext(workdir=tmp_path, answers={"application": "bench supply", "jurisdiction": "EU"})
    state = Orchestrator(ctx).run(ir, stop_after=Stage.REQUIREMENT_ANALYSIS)
    o = state.outcome(Stage.REQUIREMENT_ANALYSIS)
    assert o.status is S.NOT_VERIFIED and o.message.startswith("3 proposal(s) applied, nothing verified")
    assert ir.requirements.get("application") is not None  # the proposals were still applied


# --------------------------------------------------------------------------- finding 7: mfg.* results name the bytes they read


def _gerber_set(out: Path, ir_hash: str, functions: dict[str, str] | None = None) -> ArtifactRef:
    out.mkdir(exist_ok=True)
    functions = functions or {"F_Cu": "Copper,L1,Top", "B_Cu": "Copper,L2,Bot", "F_Mask": "Soldermask,Top", "B_Mask": "Soldermask,Bot", "Edge_Cuts": "Profile,NP"}
    files = []
    for name, fn in functions.items():
        f = out / f"b-{name}.gbr"
        f.write_text(f"%TF.FileFunction,{fn}*%\nG04 body*\nM02*\n", encoding="utf-8")
        files.append(f)
    job = out / "b-job.gbrjob"
    job.write_text(json.dumps({"FilesAttributes": [{"Path": f.name} for f in files]}), encoding="utf-8")
    files.append(job)
    return ArtifactRef(kind=ArtifactKind.GERBER, path=str(job), files=[str(f) for f in files], content_hash=hash_file_set(files), generated_from_ir_hash=ir_hash)


def test_output_check_is_stamped_with_the_disk_hash_not_the_recorded_one(divider_ir: CircuitIR, tmp_path: Path):
    art = _gerber_set(tmp_path / "g", divider_ir.content_hash())
    assert check_output_artifact(art).artifact_hash == art.content_hash  # unmodified: identical
    Path(art.files[0]).write_text("%TF.FileFunction,Copper,L1,Top*%\nG04 TAMPERED*\nM02*\n", encoding="utf-8")
    res = check_output_artifact(art)
    assert res.status is S.PASS and res.artifact_hash == art.disk_hash() != art.content_hash
    assert res.details["disk_differs_from_recorded"] is True and res.details["recorded_artifact_hash"] == art.content_hash
    # the reviewer therefore never accepts this result as a check of the recorded artifact
    divider_ir.artifacts[ArtifactKind.GERBER] = art
    divider_ir.validation.add(res)
    Path(art.files[0]).unlink()
    res = check_output_artifact(art)
    assert res.status is S.FAIL and res.artifact_hash is None and "missing" in res.message


def test_rerun_tool_refuses_stale_or_edited_artifacts(divider_ir: CircuitIR, tmp_path: Path):
    art = _gerber_set(tmp_path / "g", "sha256:older-ir")
    divider_ir.artifacts[ArtifactKind.GERBER] = art
    finding = ValidationResult(check_id=ReviewArea.MANUFACTURING_OUTPUTS, status=S.FAIL, details={"repair": "rerun_tool", "tool_check": "mfg.gerber"})
    action = RerunTool().apply(divider_ir, finding, tmp_path, {})
    assert not action.succeeded and "regenerate it first" in action.error and divider_ir.validation.latest("mfg.gerber") is None
    art.generated_from_ir_hash = divider_ir.content_hash()
    Path(art.files[0]).write_text("edited", encoding="utf-8")
    action = RerunTool().apply(divider_ir, finding, tmp_path, {})
    assert not action.succeeded and "does not match its recorded hash" in action.error
    art.content_hash = art.disk_hash()
    assert RerunTool().apply(divider_ir, finding, tmp_path, {}).succeeded


# --------------------------------------------------------------------------- finding 8: ERC report without the sheets container


def test_erc_report_without_sheets_is_an_error_not_a_pass(tmp_path: Path):
    sch = tmp_path / "x.kicad_sch"
    sch.write_text("(kicad_sch)", encoding="utf-8")
    rep = tmp_path / "erc.json"
    base = {"$schema": "x", "source": "x", "kicad_version": "10.0.6", "included_severities": ["error", "warning"], "ignored_checks": []}
    rep.write_text(json.dumps({**base, "schematics": [{"violations": [{"type": "pin_not_connected", "severity": "error"}]}]}), encoding="utf-8")
    k = KicadCli(binary="dummy")
    with pytest.raises(ToolExecutionError, match="has no 'sheets' list"):
        k._report_to_result("kicad.erc", sch, rep, "sheets")
    rep.write_text(json.dumps({**base, "sheets": [{"uuid_path": "/"}]}), encoding="utf-8")
    with pytest.raises(ToolExecutionError, match=r"sheets\[0\] has no 'violations' list"):
        k._report_to_result("kicad.erc", sch, rep, "sheets")
    rep.write_text(json.dumps({**base, "sheets": [{"violations": []}]}), encoding="utf-8")
    assert k._report_to_result("kicad.erc", sch, rep, "sheets").status is S.PASS
    # DRC: the flat lists must exist too
    rep.write_text(json.dumps({**base, "violations": []}), encoding="utf-8")
    with pytest.raises(ToolExecutionError, match="has no 'unconnected_items' list"):
        k._report_to_result("kicad.drc", sch, rep, None)


# --------------------------------------------------------------------------- finding 12: warnings are findings


def test_erc_and_drc_warnings_are_fail_not_pass(tmp_path: Path):
    sch = tmp_path / "x.kicad_sch"
    sch.write_text("(kicad_sch)", encoding="utf-8")
    rep = tmp_path / "erc.json"
    base = {"kicad_version": "10.0.6", "included_severities": ["error", "warning", "exclusions"], "ignored_checks": []}
    rep.write_text(json.dumps({**base, "sheets": [{"violations": [{"type": "lib_symbol_mismatch", "severity": "warning"}]}]}), encoding="utf-8")
    k = KicadCli(binary="dummy")
    res = k._report_to_result("kicad.erc", sch, rep, "sheets")
    assert res.status is S.FAIL and res.message == "0 error(s), 1 warning(s)" and res.details["warning_policy"]
    rep.write_text(json.dumps({**base, "violations": [], "unconnected_items": [], "schematic_parity": [{"type": "net_conflict", "severity": "warning", "excluded": True}]}), encoding="utf-8")
    res = k._report_to_result("kicad.drc", sch, rep, None)
    assert res.status is S.FAIL and "1 marked excluded" in res.message
    rep.write_text(json.dumps({**base, "violations": [], "unconnected_items": []}), encoding="utf-8")
    assert k._report_to_result("kicad.drc", sch, rep, None).status is S.PASS
    # the reviewer passes a FAIL through as a human matter and names the types
    ir = CircuitIR(project=ProjectMeta(id="p", name="p"))
    art = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path=str(sch), content_hash="sha256:s", generated_from_ir_hash=ir.content_hash())
    ir.artifacts[ArtifactKind.SCHEMATIC] = art
    ir.validation.add(ValidationResult(check_id="kicad.erc", status=S.FAIL, tool="kicad-cli", artifact_hash="sha256:s", details={"errors": [], "warnings": [{"type": "isolated_pin_label"}]}))
    r = _review(ir, tmp_path, ReviewArea.ERC)
    assert r.status is S.FAIL and r.details == {"repair": "human", "error_types": [], "warning_types": ["isolated_pin_label"]}


# --------------------------------------------------------------------------- finding 13: repair attempts are recorded, mutation is refused


def test_repair_strategy_that_mutates_the_ir_is_a_failed_action_and_stops_the_loop(divider_ir: CircuitIR, tmp_path: Path):
    tools = {"compilers": {ArtifactKind.BOM: BOMCompiler(), ArtifactKind.CPL: CPLCompiler()}}
    divider_ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(divider_ir, CompileContext(workdir=tmp_path))
    divider_ir.components.append(make_component("R3", "1k"))  # BOM stale

    class Mutating(RepairStrategy):
        id = "mutating"

        def can_repair(self, finding):
            return finding.details.get("repair") == "regenerate"

        def apply(self, ir, finding, workdir, tools):
            ir.components.append(make_component("RX", "4k7"))  # a design change
            ir.artifacts[ArtifactKind.BOM] = tools["compilers"][ArtifactKind.BOM].compile(ir, CompileContext(workdir=workdir))
            return RepairAction(strategy=self.id, finding_check_id=finding.check_id, description="mutate", ir_hash_before="lie", ir_hash_after="lie", succeeded=True)

    outcome = RepairLoop(tools=tools, strategies=[Mutating()], max_iterations=5).run(divider_ir, tmp_path)
    assert outcome.stopped_reason == MUTATION_STOP and outcome.iterations == 1 and outcome.mutated_ir
    assert len(outcome.actions) == 1 and not outcome.actions[0].succeeded and "changed the design" in outcome.actions[0].error
    assert outcome.actions[0].ir_hash_before != outcome.actions[0].ir_hash_after != "lie"
    assert [u.check_id for u in outcome.unresolved] == [ReviewArea.PCB_VS_BOM] and "changed the design" in outcome.unresolved[0].message
    log = outcome.as_validation_result(divider_ir.content_hash(), RepairLoop.version)
    assert log.status is S.FAIL and log.details["actions"][0]["strategy"] == "mutating" and log.details["stopped_reason"] == MUTATION_STOP


def test_repair_agent_persists_every_attempt_in_the_ir(divider_ir: CircuitIR, tmp_path: Path):
    tools = {"compilers": {ArtifactKind.BOM: BOMCompiler(), ArtifactKind.CPL: CPLCompiler()}}
    divider_ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(divider_ir, CompileContext(workdir=tmp_path))
    divider_ir.components.append(make_component("R3", "1k"))
    result = RepairAgent().run(divider_ir, AgentContext(workdir=tmp_path, tools=tools))
    divider_ir.validation.extend(result.validation)
    log = divider_ir.validation.latest("repair.loop")
    assert log is not None and log.tool == "repair.loop" and log.status is S.PASS and log.ir_hash == divider_ir.content_hash()
    actions = log.details["actions"]
    assert len(actions) == 1 and actions[0]["strategy"] == "repair.regenerate_artifact" and actions[0]["succeeded"] is True
    assert actions[0]["ir_hash_before"] == actions[0]["ir_hash_after"] == divider_ir.content_hash() and actions[0]["at"]
    saved = CircuitIR.load(divider_ir.save(tmp_path / "ir.json"))
    assert saved.validation.latest("repair.loop").details["actions"][0]["description"] == "regenerate bom from IR"
    assert saved.content_hash() == divider_ir.content_hash()  # the log lives outside the design hash
    # nothing to repair -> NOT_APPLICABLE, never PASS on no evidence
    result = RepairAgent().run(divider_ir, AgentContext(workdir=tmp_path, tools=tools))
    assert next(r for r in result.validation if r.check_id == "repair.loop").status is S.NOT_APPLICABLE


# --------------------------------------------------------------------------- findings 9 / 14 / 15: layer count and zones reach the checks


def test_gerber_completeness_follows_the_layer_count():
    assert required_functions(4) == {"Copper,L1", "Copper,L2", "Copper,L3", "Copper,L4", "Soldermask,Top", "Soldermask,Bot", "Profile"}
    with pytest.raises(ValueError):
        required_functions(3)


def test_gerber_check_uses_the_ir_layer_count_and_zone_layers(divider_ir: CircuitIR, tmp_path: Path):
    art = _gerber_set(tmp_path / "g", divider_ir.content_hash())
    files = [Path(f) for f in art.files]
    assert check_gerber_set(files).status is S.PASS
    res = check_gerber_set(files, layer_count=4)
    assert res.status is S.FAIL and "missing layers: ['Copper,L3', 'Copper,L4']" in res.message
    res = check_gerber_set(files, zone_layers=["B.Cu"])
    assert res.status is S.FAIL and "zone on B.Cu" in res.message and "no filled region" in res.message
    bcu = next(f for f in files if f.name.endswith("B_Cu.gbr"))
    bcu.write_text("%TF.FileFunction,Copper,L2,Bot*%\nG36*\nX0Y0D02*\nG37*\nM02*\n", encoding="utf-8")
    res = check_gerber_set(files, zone_layers=["B.Cu"])
    assert res.status is S.PASS and res.details["zone_layers_with_copper"] == ["B.Cu"]
    # through the artifact check the board description comes from the IR
    from ai_eda.ir import BoardOutline, PCBDesign

    art = _gerber_set(tmp_path / "g", divider_ir.content_hash())
    divider_ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10, height_mm=10), layers=[Layer(name=n, kind="signal") for n in ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")])
    res = check_output_artifact(art, divider_ir)
    assert res.status is S.FAIL and res.details["layer_count"] == 4 and res.details["board_from_ir"] is True
    assert check_output_artifact(art).details["board_from_ir"] is False


@needs_kicad
def test_zone_is_refilled_for_drc_and_plotted_in_the_gerbers(tmp_path: Path):
    """A GND pour that is the *only* GND connection: DRC must see it and the copper plot must contain it."""
    ir = _routed(tmp_path)
    ir.pcb.tracks = [t for t in ir.pcb.tracks if t.net != "GND"]
    ir.pcb.zones = [Zone(net="GND", layer="F.Cu", polygon=[(0.5, 0.5), (29.5, 0.5), (29.5, 19.5), (0.5, 19.5)], provenance=HAND_PLACED)]
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": LIB, "kicad_cli": kicad})
    SchematicCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    pcb = Path(ir.artifacts[ArtifactKind.PCB].path)
    assert "filled_polygon" not in pcb.read_text(encoding="utf-8")
    unfilled = kicad.run_drc(pcb, tmp_path / "drc_unfilled.json", refill_zones=False)
    assert unfilled.status is S.FAIL and {v["type"] for v in unfilled.details["errors"]} == {"unconnected_items"}
    res = kicad.run_drc(pcb, tmp_path / "drc.json", schematic_parity=True)
    assert res.status is S.PASS and res.details["errors"] == [] and res.details["warnings"] == [] and res.details["zones_refilled"] is True
    assert res.details["schematic_parity_checked"] is True and res.details["schematic_parity"] == []
    art = GerberExporter().compile(ir, ctx)
    fcu = next(Path(f) for f in art.files if f.endswith("F_Cu.gbr"))
    assert "G36*" in fcu.read_text(encoding="utf-8")
    check = check_output_artifact(art, ir)
    assert check.status is S.PASS and check.details["zone_layers_with_copper"] == ["F.Cu"], check.message
    plain = kicad.export_gerbers(pcb, tmp_path / "plain", check_zones=False)
    assert "G36*" not in next(f for f in plain if f.name.endswith("F_Cu.gbr")).read_text(encoding="utf-8")


@needs_kicad
def test_four_layer_board_exports_inner_copper_and_passes_the_check(tmp_path: Path):
    ir = _routed(tmp_path)
    ir.pcb.layers = [Layer(name="F.Cu", kind="signal"), Layer(name="In1.Cu", kind="power"), Layer(name="In2.Cu", kind="power"), Layer(name="B.Cu", kind="signal")]
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": LIB, "kicad_cli": kicad})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    art = GerberExporter().compile(ir, ctx)
    names = sorted(Path(f).name for f in art.files)
    assert "divider_conn-In1_Cu.gbr" in names and "divider_conn-In2_Cu.gbr" in names and len(names) == 12
    res = check_output_artifact(art, ir)
    assert res.status is S.PASS, res.message
    assert {"Copper,L1", "Copper,L2", "Copper,L3", "Copper,L4"} <= set(res.details["functions"]) and res.details["layer_count"] == 4
    # without the IR the check can only assume 2 layers (L1 + L2 exist, so it cannot tell): that is why the IR is passed
    assert check_output_artifact(art).details["board_from_ir"] is False


# --------------------------------------------------------------------------- finding 16: layout provenance


def test_layout_items_default_to_an_unrecorded_assumption_and_the_reviewer_says_so(tmp_path: Path):
    from ai_eda.ir import BoardOutline, PCBDesign, unrecorded_origin

    t = Track(net="GND", layer="F.Cu", start=(0, 0), end=(1, 1), width_mm=0.25)
    assert t.provenance.kind is ProvenanceKind.ASSUMPTION and t.provenance.needs_verification and t.provenance.note == unrecorded_origin().note
    ir = CircuitIR(project=ProjectMeta(id="p", name="p"))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10, height_mm=10), placements=[Placement(component_ref="R1", x_mm=1, y_mm=1)], tracks=[t])
    pcb = tmp_path / "p.kicad_pcb"
    pcb.write_text("(kicad_pcb)", encoding="utf-8")
    ir.artifacts[ArtifactKind.PCB] = ArtifactRef(kind=ArtifactKind.PCB, path=str(pcb), content_hash=ArtifactRef(kind=ArtifactKind.PCB, path=str(pcb)).disk_hash(), generated_from_ir_hash=ir.content_hash())
    r = _review(ir, tmp_path, ReviewArea.IR_VS_PCB)
    assert r.status is S.NOT_VERIFIED and r.details["unverified_layout"] == ["placement[R1]", "track[0:GND]"] and r.details["repair"] == "human"
    ir.pcb.placements[0].provenance = HAND_PLACED
    ir.pcb.tracks[0].provenance = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.naive", tool_version="0.1")
    ir.artifacts[ArtifactKind.PCB].generated_from_ir_hash = ir.content_hash()
    assert _review(ir, tmp_path, ReviewArea.IR_VS_PCB).status is S.PASS
    # a stale board is still FAIL first
    ir.pcb.placements[0].x_mm = 2
    assert _review(ir, tmp_path, ReviewArea.IR_VS_PCB).status is S.FAIL


def test_net_can_serve_a_requirement():
    from ai_eda.ir import Requirement, RequirementKind

    ir = CircuitIR(project=ProjectMeta(id="p", name="p"))
    ir.requirements.requirements.append(Requirement(id="req.bus", key="bus", text="I2C bus", kind=RequirementKind.EXPLICIT, category="electrical"))
    ir.nets.append(Net(name="SDA", pins=[], provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="t"), serves_requirements=["req.bus"]))
    assert _review(ir, Path("."), ReviewArea.REQUIREMENTS_VS_IR).status is S.PASS


# --------------------------------------------------------------------------- finding 17: absent identity is NOT_VERIFIED, not blank


def test_missing_identity_is_not_verified_in_bom_reviewer_and_validator(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="p", name="p", workdir=str(tmp_path)))
    r1 = make_component("R1", "10k")
    r1.mpn, r1.package, r1.manufacturer = None, None, None
    ir.components = [r1]
    art = BOMCompiler().compile(ir, CompileContext(workdir=tmp_path))
    row = next(iter(csv.DictReader(open(art.path, newline="", encoding="utf-8"))))
    assert (row["Manufacturer"], row["MPN"], row["Package"], row["Supplier"], row["SupplierPN"]) == ("NOT_VERIFIED",) * 5
    r = _review(ir, tmp_path, ReviewArea.COMPONENT_PROVENANCE)
    assert r.status is S.NOT_VERIFIED and "R1.mpn[missing]" in r.details["weak"] and r.details["repair"] == "human"
    v = default_registry.get("ir.component_provenance").validate(ir, ValidationContext(workdir=tmp_path))[0]
    assert v.status is S.NOT_VERIFIED and v.details["unverified"] == ["R1.mpn[missing]"]
    # an MPN merely *tagged* authoritative (the fixture's SourceRef names no archived copy) is not grounded either:
    # identity becomes authoritative only through an archived, hash-verified datasheet (tests/test_parts_regulatory_e2e.py)
    ir.components = [make_component("R1", "10k")]
    r = _review(ir, tmp_path, ReviewArea.COMPONENT_PROVENANCE)
    assert r.status is S.NOT_VERIFIED and "R1.mpn[authoritative, unarchived]" in r.details["weak"] and "tagged authoritative but not grounded" in r.message
    v = default_registry.get("ir.component_provenance").validate(ir, ValidationContext(workdir=tmp_path))[0]
    assert v.status is S.NOT_VERIFIED and v.details["unverified"] == ["R1.mpn[authoritative, unarchived]"] and "not grounded" in v.message
