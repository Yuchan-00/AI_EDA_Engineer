"""The CPL's ``Mid X`` / ``Mid Y`` are the part's centre - the centre of its pads' box - and the reviewer measures the same centre on
the compiled board.

The 9 V piano board's CPL wrote every footprint's origin under ``Mid X`` / ``Mid Y``: pin 1 for KiCad's through-hole footprints,
3.8 to 7.6 mm from the part's middle (a DIP-14, a TO-220, a radial capacitor). An assembler's centroid file means the centre.

The 9 V piano board's rev 3 CPL wrote that centre in KiCad's page frame (Y down) while its Gerbers put the copper at Y 0 to -141.5
(Gerber's Y axis up): an assembler that lines the CPL up with the copper would mirror every part. compiler.cpl 0.3 writes the Gerbers'
frame - Y up, a part on the board has a negative Mid Y -, the reviewer measures in the frame the CPL's version wrote and names it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import re

from ai_eda.compilers import CompileContext, CPLCompiler, PCBCompiler
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.ir import ArtifactKind, BoardSide, ValidationStatus as S
from ai_eda.review import IndependentReviewer
from ai_eda.review.areas import ReviewArea
from ai_eda.tools.kicad.library import KicadLibrary
from tests.test_silkscreen import silk_ir, silk_library


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return silk_library(tmp_path / "kicad")


def test_mid_x_y_is_the_centre_of_the_pads_on_every_side_and_turn_and_the_reviewer_agrees(tmp_path: Path, lib: KicadLibrary):
    # 1x3 headers (pads 2.54 mm apart down from pin 1 at the origin): as placed, turned 90 degrees, on the bottom side; a chip resistor
    parts = [("J1", 10.0, 5.0), ("J2", 20.0, 5.0, 90.0), ("J3", 30.0, 12.0, 0.0, BoardSide.BOTTOM), ("R1", 5.0, 15.0)]
    ir = silk_ir(tmp_path, lib, parts, {}, (40.0, 20.0))
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.CPL] = cpl = CPLCompiler().compile(ir, ctx)
    rows = {r["Designator"]: (r["Mid X"], r["Mid Y"]) for r in csv.DictReader(Path(cpl.path).open(encoding="utf-8"))}
    # the Gerbers' frame: Y up from the page origin (the board's top-left corner here), so every Mid Y on the board is negative
    assert rows == {"J1": ("10.0000mm", "-7.5400mm"), "J2": ("22.5400mm", "-5.0000mm"), "J3": ("30.0000mm", "-9.4600mm"), "R1": ("5.0000mm", "-15.0000mm")}
    assert cpl.notes == [] and cpl.generator_version == "0.4"
    review = next(r for r in IndependentReviewer().review(ir, tmp_path).results if r.check_id == ReviewArea.PCB_VS_CPL)
    assert review.status is S.PASS, review.details
    assert "the Gerbers' frame (mm from KiCad's page origin, Y up)" in review.message and review.details["frame"].startswith("the Gerbers'")


def test_a_cpl_written_by_the_anchor_version_is_reviewed_by_its_anchors_and_sent_back_to_be_recompiled(tmp_path: Path, lib: KicadLibrary):
    """A project compiled before compiler.cpl 0.2 holds a CPL whose Mid X / Mid Y are the footprint anchors (the IR records the
    artifact's generator version 0.1). The reviewer compares it with the anchors on the board - not with the pads' centre, which
    would call every through-hole row a disagreement about the board - and fails it as written to the anchor convention, to be
    recompiled; a 0.1 CPL whose anchors do not match the board still names the rows, by anchor."""
    import hashlib

    ir = silk_ir(tmp_path, lib, [("J1", 10.0, 5.0), ("R1", 5.0, 15.0)], {}, (40.0, 20.0))
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.CPL] = CPLCompiler().compile(ir, ctx)

    def old_cpl(j1: str) -> None:
        data = ("Designator,Mid X,Mid Y,Rotation,Layer\n" f"J1,{j1},5.0000mm,0,Top\n" "R1,5.0000mm,15.0000mm,0,Top\n").encode("utf-8")
        path = tmp_path / "cpl.csv"
        path.write_bytes(data)
        ir.artifacts[ArtifactKind.CPL] = ir.artifacts[ArtifactKind.CPL].model_copy(
            update={"generator_version": "0.1", "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()})

    def review():
        return next(r for r in IndependentReviewer().review(ir, tmp_path).results if r.check_id == ReviewArea.PCB_VS_CPL)

    old_cpl("10.0000mm")
    r = review()
    assert r.status is S.FAIL and r.details.get("repair") == "regenerate" and "mismatches" not in r.details, r.details
    assert "compiler.cpl 0.1" in r.message and "footprint anchors" in r.message, r.message
    old_cpl("11.0000mm")
    r = review()
    assert r.status is S.FAIL and r.details["mismatches"] == [
        "J1: position CPL (11.0, 5.0) vs board (10.0, 5.0) - the footprint anchor, in KiCad's page frame (mm from the page origin, Y down)"], r.details


def _old_cpl(ir, tmp_path: Path, version: str, rows: str) -> None:
    import hashlib

    data = ("Designator,Mid X,Mid Y,Rotation,Layer\n" + rows).encode("utf-8")
    (tmp_path / "cpl.csv").write_bytes(data)
    ir.artifacts[ArtifactKind.CPL] = ir.artifacts[ArtifactKind.CPL].model_copy(
        update={"generator_version": version, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()})


def _review(ir, tmp_path: Path):
    return next(r for r in IndependentReviewer().review(ir, tmp_path).results if r.check_id == ReviewArea.PCB_VS_CPL)


def test_a_cpl_in_the_page_frame_is_reviewed_in_that_frame_and_sent_back_and_the_gerber_frame_is_not_read_as_it(tmp_path: Path, lib: KicadLibrary):
    """compiler.cpl 0.2 wrote the centres in KiCad's page frame (Y down): the reviewer reads such a CPL in that frame - its rows match
    the board there - says so, and fails it to be recompiled (its Mid Y has the opposite sign of the Gerbers'). The same positive
    rows stamped 0.3 are read in the Gerbers' frame and disagree with the board, naming the frame; 0.3's own rows pass."""
    ir = silk_ir(tmp_path, lib, [("J1", 10.0, 5.0), ("R1", 5.0, 15.0)], {}, (40.0, 20.0))
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.CPL] = CPLCompiler().compile(ir, ctx)
    assert _review(ir, tmp_path).status is S.PASS
    page = "J1,10.0000mm,7.5400mm,0,Top\nR1,5.0000mm,15.0000mm,0,Top\n"
    _old_cpl(ir, tmp_path, "0.2", page)
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details.get("repair") == "regenerate" and "mismatches" not in r.details, r.details
    assert "compiler.cpl 0.2" in r.message and "in KiCad's page frame (mm from the page origin, Y down)" in r.message, r.message
    assert "opposite sign of the Gerbers' Y" in r.message
    _old_cpl(ir, tmp_path, "0.3", page)
    r = _review(ir, tmp_path)
    assert r.status is S.FAIL and r.details["repair"] == "human" and "Gerbers' frame" in r.message, r.message
    assert r.details["mismatches"][0] == (
        "J1: position CPL (10.0, 7.54) vs board (10.0, -7.54) - the centre of its pads' box, in the Gerbers' frame (mm from KiCad's page "
        "origin, Y up)")


@pytest.mark.skipif(not KicadCli().available(), reason="kicad-cli not installed")
def test_every_cpl_centre_is_the_centre_of_its_pads_in_the_gerbers_kicad_cli_plots(tmp_path: Path, lib: KicadLibrary):
    """Measured on the plot itself: kicad-cli's F_Cu Gerber flashes every pad of the board, and the box of the flashes within 4 mm of a
    CPL point is centred on it, in the Gerber's own coordinates (``%FSLAX46Y46*%`` in mm: integers of 1 nm). The page-frame CPL of
    compiler.cpl 0.2 found no flash near its points (every Mid Y had the wrong sign)."""
    parts = [("J1", 10.0, 5.0), ("J2", 25.0, 5.0, 90.0), ("R1", 5.0, 15.0)]
    ir = silk_ir(tmp_path, lib, parts, {}, (40.0, 20.0))
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    cpl = CPLCompiler().compile(ir, ctx)
    gerbers = KicadCli().export_gerbers(Path(ir.artifacts[ArtifactKind.PCB].path), tmp_path / "gerber", ["F.Cu"], check_zones=False)
    text = next(g for g in gerbers if g.name.endswith("F_Cu.gbr")).read_text(encoding="utf-8")
    assert "%FSLAX46Y46*%" in text and "%MOMM*%" in text
    flashes = [(int(x) * 1e-6, int(y) * 1e-6) for x, y in re.findall(r"^X(-?\d+)Y(-?\d+)D03\*$", text, flags=re.MULTILINE)]
    assert len(flashes) == 3 + 3 + 2
    for row in csv.DictReader(Path(cpl.path).open(encoding="utf-8")):
        x, y = float(row["Mid X"].rstrip("m")), float(row["Mid Y"].rstrip("m"))
        near = [f for f in flashes if abs(f[0] - x) <= 4.0 and abs(f[1] - y) <= 4.0]
        assert near, (row, flashes)
        xs, ys = [f[0] for f in near], [f[1] for f in near]
        assert ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0) == pytest.approx((x, y), abs=1e-4), row
