"""A footprint that is no part to buy or place - KiCad's ``(attr exclude_from_pos_files exclude_from_bom)``, set by KiCad's own
library on its test points, mounting holes and fiducials - leaves the BOM and the CPL, and the schematic says the same.

Before: ``compiler.bom`` wrote a row for every component and ``compiler.cpl`` one for every placement, so a test point pad
(``Connector:TestPoint`` on ``TestPoint:TestPoint_THTPad_D2.0mm_Drill1.0mm``) told an assembler to buy and place a part that is
only copper; the schematic wrote ``(in_bom yes)`` while the board copied the footprint's ``exclude_from_bom``, and KiCad 10's
DRC schematic parity reported the mismatch (``footprint_symbol_mismatch``, measured with kicad-cli 10.0.6 on 2026-11-19). The
silk placer also labelled a test point's pad with its net name at 0.8 mm next to the design file's own label of that net.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ai_eda.compilers import BOMCompiler, CompileContext, CPLCompiler, PCBCompiler, SchematicCompiler
from ai_eda.ir import (
    ArtifactKind,
    BoardSide,
    Component,
    LibraryRef,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    Provenance,
    ProvenanceKind,
    SilkKind,
    SilkText,
    ValidationStatus as S,
)
from ai_eda.ir.simulation import SpiceBinding
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from ai_eda.tools.kicad.library import fabrication_exclusions
from ai_eda.tools.silkscreen.place import place_silkscreen
from tests.fixtures_kicad import HAND_PLACED, divider_with_connector_ir

LIB = KicadLibrary()
TP_SYMBOL = LibraryRef(library="Connector", name="TestPoint")
TP_FOOTPRINT = LibraryRef(library="TestPoint", name="TestPoint_THTPad_D2.0mm_Drill1.0mm")
HAS_LIBS = LIB.symbol_file("Connector") is not None and LIB.footprint_file(TP_FOOTPRINT.library, TP_FOOTPRINT.name) is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")
FIXTURE = Provenance(kind=ProvenanceKind.DERIVED, tool="test", note="test point fixture")


def _with_test_point(tmp_path: Path):
    """The divider fixture (J1, R1, R2 on a 30 x 20 mm board) with a test point TP1 on VOUT right of the resistors."""
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.components.append(Component(
        ref="TP1", value="VOUT", description="test point on VOUT (a plated pad, nothing to buy)",
        pins=[Pin(number="1", name="1", electrical_type=PinElectricalType.PASSIVE, provenance=FIXTURE)],
        symbol=LIB.resolve_symbol(TP_SYMBOL), footprint=LIB.resolve_footprint(TP_FOOTPRINT), provenance=FIXTURE,
        spice=SpiceBinding(exclude=True, exclude_reason="a test point pad", provenance=FIXTURE),
    ))
    next(n for n in ir.nets if n.name == "VOUT").pins.append(PinRef(component_ref="TP1", pin_number="1"))
    ir.pcb.placements.append(Placement(component_ref="TP1", x_mm=22.0, y_mm=8.0, rotation_deg=0.0, side=BoardSide.TOP, provenance=HAND_PLACED))
    return ir


def _compile(ir, workdir: Path) -> None:
    ctx = CompileContext(workdir=workdir, tools={"kicad_library": LIB})
    ir.artifacts[ArtifactKind.SCHEMATIC] = SchematicCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.BOM] = BOMCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.CPL] = CPLCompiler().compile(ir, ctx)


def _rows(path: str, key: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row[key] for row in csv.DictReader(f)]


@needs_libs
def test_the_library_footprint_says_what_a_test_point_leaves_out():
    assert fabrication_exclusions(LIB, TP_FOOTPRINT) == {"exclude_from_bom", "exclude_from_pos_files"}
    assert fabrication_exclusions(LIB, LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric")) == frozenset()
    # a footprint that cannot be read is never assumed to be excluded: it stays in the BOM and the CPL
    assert fabrication_exclusions(LIB, LibraryRef(library="TestPoint", name="No_Such_Footprint")) == frozenset()
    assert fabrication_exclusions(LIB, None) == frozenset()


@needs_libs
def test_bom_and_cpl_leave_out_a_test_point_and_say_so(tmp_path: Path):
    ir = _with_test_point(tmp_path)
    _compile(ir, tmp_path)
    bom, cpl = ir.artifacts[ArtifactKind.BOM], ir.artifacts[ArtifactKind.CPL]
    assert _rows(bom.path, "Reference") == ["J1", "R1", "R2"]
    assert _rows(cpl.path, "Designator") == ["J1", "R1", "R2"]
    assert any(n.startswith("TP1: not in the BOM") and "exclude_from_bom" in n for n in bom.notes), bom.notes
    assert any(n.startswith("TP1: not in the CPL") and "exclude_from_pos_files" in n for n in cpl.notes), cpl.notes


@needs_libs
def test_the_schematic_marks_the_test_point_as_its_footprint_does_and_nothing_else(tmp_path: Path):
    ir = _with_test_point(tmp_path)
    _compile(ir, tmp_path)
    tree = sexpr.parse_file(Path(ir.artifacts[ArtifactKind.SCHEMATIC].path))
    flags = {}
    for sym in sexpr.find_all(tree, "symbol"):
        ref = next((str(p[2]) for p in sexpr.find_all(sym, "property") if len(p) > 2 and p[1] == "Reference"), None)
        if ref is None or sexpr.find(sym, "lib_id") is None:
            continue
        flags[ref] = (str(sexpr.get(sym, "in_bom")), sexpr.get(sym, "in_pos_files"))
    assert flags["TP1"] == ("no", "no")
    # every other part is written as before: in the BOM, no in_pos_files token (KiCad's default: yes)
    assert {ref: f for ref, f in flags.items() if ref != "TP1" and not ref.startswith("#")} == {r: ("yes", None) for r in ("J1", "R1", "R2")}
    board = Path(ir.artifacts[ArtifactKind.PCB].path).read_text(encoding="utf-8")
    assert "(attr exclude_from_pos_files exclude_from_bom)" in board


@needs_libs
def test_the_reviewer_finds_bom_and_cpl_in_agreement_with_a_board_that_carries_a_test_point(tmp_path: Path):
    ir = _with_test_point(tmp_path)
    _compile(ir, tmp_path)
    results = {r.check_id: r for r in IndependentReviewer().review(ir, tmp_path).results}
    for area in (ReviewArea.PCB_VS_BOM, ReviewArea.PCB_VS_CPL):
        assert results[area].status is S.PASS, (area, results[area].message, results[area].details)
    # the reviewer still catches a test point that went into the BOM after all (a hand-edited BOM)
    bom = Path(ir.artifacts[ArtifactKind.BOM].path)
    rows = bom.read_text(encoding="utf-8").splitlines()
    bom.write_text("\n".join([*rows, rows[-1].replace("R2", "TP1", 1)]) + "\n", encoding="utf-8")
    r = next(r for r in IndependentReviewer().review(ir, tmp_path).results if r.check_id == ReviewArea.PCB_VS_BOM)
    assert r.status is S.FAIL and r.details.get("only_in_bom") == ["TP1"], (r.message, r.details)


@needs_kicad
def test_kicad_schematic_parity_finds_no_attribute_mismatch_on_a_test_point(tmp_path: Path):
    ir = _with_test_point(tmp_path)
    _compile(ir, tmp_path)
    res = kicad.run_drc(Path(ir.artifacts[ArtifactKind.PCB].path), tmp_path / "drc.json", schematic_parity=True)
    assert res.details["schematic_parity_checked"] is True
    assert res.details["schematic_parity"] == [], res.details["schematic_parity"]


@needs_libs
def test_a_design_text_that_names_the_test_points_net_is_not_repeated_by_a_pin_label(tmp_path: Path):
    ir = _with_test_point(tmp_path)
    plain = place_silkscreen(ir, LIB)
    assert [(t.text, t.component_ref) for t in plain.texts if t.kind is SilkKind.PIN_LABEL and t.component_ref == "TP1"] == [("VOUT", "TP1")]
    label = SilkText(text="VOUT", x_mm=22.0, y_mm=5.5, size_mm=1.0, kind=SilkKind.USER, component_ref="TP1", provenance=FIXTURE)
    kept = place_silkscreen(ir, LIB, keep=[label], kept_by="design file")
    assert [t for t in kept.texts if t.kind is SilkKind.PIN_LABEL and t.component_ref == "TP1"] == []
    assert any("TP1.1 VOUT" in n and "design file" in n for n in kept.notes), kept.notes
    # the connector's own labels are untouched: J1's pins still get theirs
    assert {t.text for t in kept.texts if t.kind is SilkKind.PIN_LABEL and t.component_ref == "J1"} == \
        {t.text for t in plain.texts if t.kind is SilkKind.PIN_LABEL and t.component_ref == "J1"}


@needs_libs
def test_a_test_point_has_no_datasheet_mpn_or_catalog_row_to_check(tmp_path: Path):
    """Before: ``parts.existence`` asked a test point's pad for a datasheet pointer and an MPN like a part to buy, so every pad
    stayed ``component.existence`` NOT_VERIFIED ("IR has no datasheet pointer ...") and named in the release message, a check
    that no online session could ever pass. The pad's symbol and footprint are still checked; with an MPN it is a part again."""
    from ai_eda.ir import Traced
    from ai_eda.parts import examine_component

    tp = next(c for c in _with_test_point(tmp_path).components if c.ref == "TP1")
    report = examine_component(tp, LIB, None, None, fetch=False)
    checks = {c.name: c.status for c in report.checks}
    assert checks == {"symbol": S.PASS, "footprint": S.PASS, "datasheet_pointer": S.NOT_APPLICABLE, "datasheet_archived": S.NOT_APPLICABLE,
                      "mpn_in_datasheet": S.NOT_APPLICABLE, "catalog": S.NOT_APPLICABLE}, checks
    result = report.result(None)
    assert result.status is S.PASS and "exclude_from_bom" in report.checks[2].message, result.message
    bought = tp.model_copy(update={"mpn": Traced(value="5011", provenance=FIXTURE)})
    assert {c.name: c.status for c in examine_component(bought, LIB, None, None, fetch=False).checks}["datasheet_pointer"] is S.NOT_VERIFIED
    resistor = next(c for c in _with_test_point(tmp_path / "r").components if c.ref == "R1")
    assert {c.name: c.status for c in examine_component(resistor, LIB, None, None, fetch=False).checks}["datasheet_pointer"] is not S.NOT_APPLICABLE
