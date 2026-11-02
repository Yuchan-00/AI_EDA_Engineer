"""The built-in 3D preview (:mod:`ai_eda.tools.model3d`): STEP envelopes, model references, the scene, the GLB and the iso SVG.

Everything runs on hand-written data: STEP fixtures written here (a
millimetre file with multi-line records, circles and pcurve points; an inch
file; refusals), a synthetic footprint library with ``F.Fab`` / silk /
courtyard graphics and ``(model ...)`` references, and a synthetic 3D
directory. One test reads a real KiCad STEP file, only when
``AI_EDA_TEST_STEP_TO92`` names ``Package_TO_SOT_THT.3dshapes/TO-92_Inline.step``
(skipped otherwise). The GLB is parsed here (header, chunks, accessors,
windings) rather than trusted.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import struct
from dataclasses import astuple
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ai_eda.errors import ToolExecutionError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, Component, LibraryRef, PCBDesign, Placement, ProjectMeta, Provenance, ProvenanceKind, Track, Traced, Via
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import GLB_EXPORT_FLAGS, PNG_SIGNATURE, STEP_EXPORT_FLAGS, KicadCli
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S
from ai_eda.tools.model3d import (
    BODY_CAPTION,
    Box3,
    ModelRef,
    SceneError,
    build_scene,
    find_3dmodel_dir,
    footprint_models,
    iso_svg,
    model_transform,
    read_step_envelope,
    resolve_model_path,
    scene_caption,
    step_bbox,
    transformed_box,
    write_glb,
)
from ai_eda.tools.model3d.iso import VIEWS
from ai_eda.tools.model3d.scene import COPPER_MM, DEFAULT_BOARD_THICKNESS_MM, MASK_MM, PAD_MM, SILK_MM, Scene, Solid, _clean, solid_faces, triangulate

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="model3d test")
SVG_NS = "{http://www.w3.org/2000/svg}"

# --------------------------------------------------------------------------- STEP fixtures


def _step(data: str, units: str = "#90 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );") -> str:
    return (
        "ISO-10303-21;\nHEADER;\n/* a comment; with a semicolon and a 'quote */\n"
        "FILE_DESCRIPTION(('hand-written; test'),'2;1');\nFILE_NAME('x.step','',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
        + data
        + "\n#80 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)\nGLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#93)) GLOBAL_UNIT_ASSIGNED_CONTEXT\n((#90,#91,#92)) REPRESENTATION_CONTEXT('Context #1',\n  '3D Context with UNIT and UNCERTAINTY') );\n"
        + units
        + "\n#91 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );\n#92 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );\n"
        "#93 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#90,'distance_accuracy_value','it''s; fine');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )


#: points x -1..2, y -0.5..0.5, z 0..1.5 (one record split over three lines), a pcurve point (2-D, ignored), a circle about +z
#: centred (4, 0, 1) with r 0.5 (x 3.5..4.5, y -0.5..0.5, z unchanged) and one about +x centred (0, 0, 2) with r 0.25
#: (y -0.25..0.25, z 1.75..2.25, x unchanged)
MM_DATA = """#1 = CARTESIAN_POINT('',(-1.,-0.5,0.E+000));
#2 = CARTESIAN_POINT('corner; with ''quotes''',(2.,
  0.5,
  1.5));
#3 = CARTESIAN_POINT('',(100.,100.));
#4 = CIRCLE('',#5,0.5);
#5 = AXIS2_PLACEMENT_3D('',#6,#7,#8);
#6 = CARTESIAN_POINT('',(4.,0.,1.));
#7 = DIRECTION('',(0.,0.,1.));
#8 = DIRECTION('',(1.,0.,0.));
#9 = CIRCLE('',#10,0.25);
#10 = AXIS2_PLACEMENT_3D('',#11,#12,$);
#11 = CARTESIAN_POINT('',(0.,0.,2.));
#12 = DIRECTION('',(1.,0.,0.));
#13 = CIRCLE('',#14,7.);
#14 = AXIS2_PLACEMENT_2D('',#3,#15);
#15 = DIRECTION('',(1.,0.));"""

INCH_UNITS = """#90 = ( CONVERSION_BASED_UNIT('INCH',#94) LENGTH_UNIT() NAMED_UNIT(#95) );
#94 = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#96);
#95 = DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);
#96 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );"""


class _approx_box:
    """``box == _approx_box(expected)``: every coordinate within 1e-9 mm."""

    def __init__(self, box: Box3) -> None:
        self.box = box

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Box3) and all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(astuple(other), astuple(self.box)))

    def __repr__(self) -> str:
        return f"approx({self.box})"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1", newline="\n")
    return path


def test_step_bbox_reads_multiline_points_circles_and_millimetres(tmp_path: Path):
    env = read_step_envelope(_write(tmp_path / "mm.step", _step(MM_DATA)))
    assert env.reason == "" and env.unit_mm == 1.0 and env.points == 4 and env.circles == 2  # the 2-D point and the 2-D circle are skipped
    assert env.box == _approx_box(Box3(-1.0, -0.5, 0.0, 4.5, 0.5, 2.25))
    assert step_bbox(tmp_path / "mm.step") == env.box


def test_step_bbox_converts_inch_and_metre_files(tmp_path: Path):
    inch = read_step_envelope(_write(tmp_path / "in.step", _step(MM_DATA, INCH_UNITS)))
    assert inch.unit_mm == pytest.approx(25.4)
    assert inch.box == _approx_box(Box3(-25.4, -12.7, 0.0, 4.5 * 25.4, 12.7, 2.25 * 25.4))
    metre = read_step_envelope(_write(tmp_path / "m.step", _step(MM_DATA, "#90 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );")))
    assert metre.unit_mm == 1000.0 and metre.box.z2 == pytest.approx(2250.0)
    via_metres = "#90 = ( CONVERSION_BASED_UNIT('INCH',#94) LENGTH_UNIT() NAMED_UNIT(#95) );\n#94 = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.0254),#96);\n#95 = DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n#96 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );"
    assert read_step_envelope(_write(tmp_path / "in2.step", _step(MM_DATA, via_metres))).unit_mm == pytest.approx(25.4)
    packed = tmp_path / "mm.stpz"
    packed.write_bytes(gzip.compress(_step(MM_DATA).encode("latin-1"), mtime=0))
    assert step_bbox(packed) == _approx_box(Box3(-1.0, -0.5, 0.0, 4.5, 0.5, 2.25))


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("not a step file", "ISO-10303-21"),
        (_step(MM_DATA, "#90 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.EXA.,.METRE.) );"), "알 수 없는 길이 단위"),
        (_step(MM_DATA, "#90 = ( NAMED_UNIT(*) SI_UNIT(.MILLI.,.GRAM.) MASS_UNIT() );"), "길이 단위"),
        (_step(MM_DATA + "\n#70 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#71)) REPRESENTATION_CONTEXT('','') );\n#71 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );"), "다름"),
        (_step("#1 = CARTESIAN_POINT('',(1.,2.));"), "CARTESIAN_POINT 가 없음"),
        (_step(MM_DATA + "\n#40 = CARTESIAN_POINT('',(1.,NaN,2.));"), "CARTESIAN_POINT"),
        (_step(MM_DATA + "\n#50 = MAPPED_ITEM('',#51,#52);"), "MAPPED_ITEM"),
        (
            _step(MM_DATA + "\n#60 = ITEM_DEFINED_TRANSFORMATION('','',#5,#61);\n#61 = AXIS2_PLACEMENT_3D('',#62,$,$);\n#62 = CARTESIAN_POINT('',(0.,0.,5.));"),
            "ITEM_DEFINED_TRANSFORMATION",
        ),
    ],
)
def test_step_bbox_refuses_with_a_reason(tmp_path: Path, text: str, words: str):
    env = read_step_envelope(_write(tmp_path / "bad.step", text))
    assert env.box is None and words in env.reason


def test_step_bbox_accepts_identity_assembly_placements_and_reports_a_missing_file(tmp_path: Path):
    identity = _step(MM_DATA + "\n#60 = ITEM_DEFINED_TRANSFORMATION('','',#61,#63);\n#61 = AXIS2_PLACEMENT_3D('',#62,$,$);\n#62 = CARTESIAN_POINT('',(0.,0.,0.));\n#63 = AXIS2_PLACEMENT_3D('',#62,#64,#65);\n#64 = DIRECTION('',(0.,0.,1.));\n#65 = DIRECTION('',(1.,0.,0.));")
    assert read_step_envelope(_write(tmp_path / "asm.step", identity)).box is not None
    missing = read_step_envelope(tmp_path / "nope.step")
    assert missing.box is None and "읽을 수 없음" in missing.reason


def test_step_bbox_on_the_real_to92_model():
    """Measured on the KiCad 10.0.6 3D library (``TO-92_Inline.step``): CARTESIAN_POINTs x -0.75..3.29, y -1.33..0.19, z -2.5..7.3.

    The circle of the round back (r 2.415 about +z at x = 1.27) widens x / y to the whole circle; z is the points'.
    """
    path = os.environ.get("AI_EDA_TEST_STEP_TO92")
    if not path or not Path(path).is_file():
        pytest.skip("set AI_EDA_TEST_STEP_TO92 to Package_TO_SOT_THT.3dshapes/TO-92_Inline.step of the KiCad 3D library")
    env = read_step_envelope(path)
    b = env.box
    assert b is not None and env.unit_mm == 1.0, env.reason
    assert (b.z1, b.z2) == pytest.approx((-2.5, 7.3), abs=0.01)
    assert b.x1 <= -0.745 and b.x2 >= 3.285 and b.y1 <= -1.33 and b.y2 >= 0.19  # contains the points' box
    assert (b.x1, b.x2, b.y1, b.y2) == pytest.approx((1.27 - 2.415, 1.27 + 2.415, -2.415, 2.415), abs=0.01)


# --------------------------------------------------------------------------- model references and transforms


def test_model_transform_is_scale_then_negated_rotations_then_offset():
    assert model_transform(ModelRef("m.step", rotate=(0.0, 0.0, 90.0)), (1.0, 0.0, 0.0)) == pytest.approx((0.0, -1.0, 0.0))
    assert model_transform(ModelRef("m.step", rotate=(90.0, 0.0, 0.0)), (0.0, 0.0, 1.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert model_transform(ModelRef("m.step", rotate=(0.0, 90.0, 0.0)), (0.0, 0.0, 1.0)) == pytest.approx((-1.0, 0.0, 0.0))
    ref = ModelRef("m.step", offset=(1.0, 2.0, 3.0), scale=(2.0, 1.0, 1.0), rotate=(0.0, 0.0, 90.0))
    assert model_transform(ref, (1.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 3.0))  # (2, 0, 0) -> (0, -2, 0) -> + offset
    box = Box3(-1.0, -1.0, 0.0, 1.0, 1.0, 2.0)
    assert transformed_box(ModelRef("m.step", rotate=(90.0, 0.0, 0.0)), box) == _approx_box(Box3(-1.0, 0.0, -1.0, 1.0, 2.0, 1.0))
    assert transformed_box(ModelRef("m.step", scale=(1.0, 1.0, 3.0), offset=(0.0, 0.0, 0.5)), box).z2 == pytest.approx(6.5)


def test_resolve_model_path_stays_inside_the_3d_library(tmp_path: Path):
    d = tmp_path / "3d"
    model = _write(d / "Pkg.3dshapes" / "A.step", _step(MM_DATA))
    assert resolve_model_path("${KICAD10_3DMODEL_DIR}/Pkg.3dshapes/A.step", d) == (model.resolve(), "")
    assert resolve_model_path("$(KICAD10_3DMODEL_DIR)/Pkg.3dshapes/A.step", d)[0] == model.resolve()
    assert resolve_model_path(str(model), None) == (model, "")
    for path, words in (
        ("${KICAD9_3DMODEL_DIR}/Pkg.3dshapes/A.step", "KICAD9_3DMODEL_DIR"),
        ("${KICAD10_3DMODEL_DIR}/Pkg.3dshapes/B.step", "STEP 파일이 없음"),
        ("${KICAD10_3DMODEL_DIR}/../outside.step", "밖"),
        ("Pkg.3dshapes/A.step", "상대 경로"),
        (str(tmp_path / "missing.step"), "없음"),
    ):
        found, why = resolve_model_path(path, d)
        assert found is None and words in why, (path, why)
    assert resolve_model_path("${KICAD10_3DMODEL_DIR}/Pkg.3dshapes/A.step", None)[1].startswith("KiCad 3D 모델 라이브러리를 찾지 못함")


def test_find_3dmodel_dir_prefers_existing_env_dirs_then_the_library_share_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    share = tmp_path / "share"
    (share / "3dmodels").mkdir(parents=True)
    other = tmp_path / "other3d"
    other.mkdir()
    lib = KicadLibrary(roots=[tmp_path / "nothing", share])
    monkeypatch.delenv("KICAD10_3DMODEL_DIR", raising=False)
    monkeypatch.delenv("KICAD_3DMODEL_DIR", raising=False)
    assert find_3dmodel_dir(lib) == share / "3dmodels"
    monkeypatch.setenv("KICAD10_3DMODEL_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("KICAD_3DMODEL_DIR", str(other))
    assert find_3dmodel_dir(lib) == other  # a KICAD10 variable naming no directory is not used
    monkeypatch.setenv("KICAD10_3DMODEL_DIR", str(tmp_path))
    assert find_3dmodel_dir(lib) == tmp_path
    monkeypatch.delenv("KICAD10_3DMODEL_DIR")
    monkeypatch.delenv("KICAD_3DMODEL_DIR")
    assert find_3dmodel_dir(KicadLibrary(roots=[tmp_path / "nothing"])) is None


# --------------------------------------------------------------------------- synthetic library / board

#: F.Fab box of BOX (footprint frame, y down): deliberately off-centre so rotations and the bottom side are told apart
FAB = (-1.0, -0.5, 3.0, 1.5)
BOX_STEP_Z = 1.5
MODEL = "${KICAD10_3DMODEL_DIR}/Test.3dshapes/BOX.step"


def _stroke(w: float = 0.1):
    return S("stroke", S("width", w), S("type", "solid"))


def _model(path: str, offset=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0), hide: bool = False):
    return S("model", Q(path), S("hide", "yes") if hide else None, S("offset", S("xyz", *offset)), S("scale", S("xyz", *scale)), S("rotate", S("xyz", *rotate)))


def _footprint(name: str, *items) -> list:
    return S("footprint", Q(name), S("version", 20260206), S("generator", Q("pcbnew")), S("layer", Q("F.Cu")), S("attr", "smd"), *items, S("embedded_fonts", False))


def _smd(number: str, x: float, y: float, shape: str = "rect") -> list:
    return S("pad", Q(number), "smd", shape, S("at", x, y), S("size", 0.6, 0.8), S("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste")), S("roundrect_rratio", 0.25) if shape == "roundrect" else None)


def _library(root: Path, box_model=None) -> KicadLibrary:
    x1, y1, x2, y2 = FAB
    box = _footprint(
        "BOX",
        S("fp_rect", S("start", x1, y1), S("end", x2, y2), _stroke(), S("fill", "no"), S("layer", Q("F.Fab"))),
        S("fp_text", "user", Q("${REFERENCE}"), S("at", 20, 20, 0), S("layer", Q("F.Fab")), S("effects", S("font", S("size", 1, 1)))),  # texts are not part of the box
        S("fp_line", S("start", -1, -1), S("end", 3, -1), _stroke(0.12), S("layer", Q("F.SilkS"))),
        S("fp_rect", S("start", -1.5, -1.2), S("end", 3.5, 2.0), _stroke(0.05), S("fill", "no"), S("layer", Q("F.CrtYd"))),
        _smd("1", 0.0, 0.0), _smd("2", 2.0, 0.0, "roundrect"),
        box_model if box_model is not None else _model(MODEL),
    )
    tht = _footprint(
        "THT",
        S("fp_rect", S("start", -1.5, -1.5), S("end", 4.0, 3.5), _stroke(0.05), S("fill", "no"), S("layer", Q("F.CrtYd"))),
        S("fp_poly", S("pts", S("xy", 0, -1.3), S("xy", 0.5, -1.8), S("xy", -0.5, -1.8)), _stroke(0.1), S("fill", "yes"), S("layer", Q("F.SilkS"))),
        S("pad", Q("1"), "thru_hole", "circle", S("at", 0, 0), S("size", 1.6, 1.6), S("drill", 0.8), S("layers", Q("*.Cu"), Q("*.Mask"))),
        S("pad", Q("2"), "thru_hole", "oval", S("at", 2.54, 0), S("size", 1.6, 2.4), S("drill", 0.8), S("layers", Q("*.Cu"), Q("*.Mask"))),
        S("pad", Q(""), "np_thru_hole", "circle", S("at", 1.27, 2.5), S("size", 1.0, 1.0), S("drill", 1.0), S("layers", Q("*.Cu"), Q("*.Mask"))),
        _model("${KICAD10_3DMODEL_DIR}/Test.3dshapes/MISSING.step"),
    )
    bare = _footprint("BARE", _smd("1", 0.0, 0.0))
    pretty = root / "footprints" / "Test_Lib.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    for name, node in (("BOX", box), ("THT", tht), ("BARE", bare)):
        sexpr.dump_file(node, pretty / f"{name}.kicad_mod")
    return KicadLibrary(roots=[root])


def _box_step(root: Path, z: float = BOX_STEP_Z) -> Path:
    pts = "\n".join(f"#{i + 1} = CARTESIAN_POINT('',({x},{y},{zz}));" for i, (x, y, zz) in enumerate([(-1.0, -1.5, -0.4), (3.0, 0.5, z)]))
    return _write(root / "Test.3dshapes" / "BOX.step", _step(pts))


def _ir(parts: list[tuple[str, str, Placement]], *, tracks=(), vias=()) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="m3d", name="m3d"))
    ir.components = [Component(ref=ref, value=ref, footprint=LibraryRef(library="Test_Lib", name=fp), provenance=USER) for ref, fp, _ in parts]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=30.0, height_mm=20.0), placements=[p for _, _, p in parts], tracks=list(tracks), vias=list(vias))
    return ir


def _place(ref: str, rot: float = 0.0, side: BoardSide = BoardSide.TOP, x: float = 10.0, y: float = 10.0) -> Placement:
    return Placement(component_ref=ref, x_mm=x, y_mm=y, rotation_deg=rot, side=side, provenance=USER)


@pytest.fixture()
def world(tmp_path: Path) -> tuple[KicadLibrary, Path]:
    lib = _library(tmp_path / "kicad")
    models = tmp_path / "3d"
    _box_step(models)
    return lib, models


def _xy_box(solid: Solid) -> tuple[float, float, float, float]:
    xs = [p[0] for p in solid.polygon]
    ys = [p[1] for p in solid.polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _only(scene: Scene, kind: str, label: str | None = None) -> list[Solid]:
    return [s for s in scene.solids if s.kind == kind and (label is None or s.label == label)]


@pytest.mark.parametrize(
    ("rot", "side", "expected"),
    [
        (0.0, BoardSide.TOP, (9.0, 9.5, 13.0, 11.5)),
        (90.0, BoardSide.TOP, (9.5, 7.0, 11.5, 11.0)),  # +90 is counter-clockwise on screen: the long side points up (-y)
        (180.0, BoardSide.TOP, (7.0, 8.5, 11.0, 10.5)),
        (270.0, BoardSide.TOP, (8.5, 9.0, 10.5, 13.0)),
        (0.0, BoardSide.BOTTOM, (9.0, 8.5, 13.0, 10.5)),  # mirrored about the footprint's x axis
        (90.0, BoardSide.BOTTOM, (8.5, 7.0, 10.5, 11.0)),
    ],
)
def test_body_box_is_the_fab_outline_placed_like_the_footprint_with_the_step_height(world, rot: float, side: BoardSide, expected):
    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1", rot, side))])
    scene = build_scene(ir, lib, model_dir=models)
    (body,) = _only(scene, "body")
    assert _xy_box(body) == pytest.approx(expected)
    T = DEFAULT_BOARD_THICKNESS_MM
    assert (body.z0, body.z1) == pytest.approx((T, T + BOX_STEP_Z) if side == BoardSide.TOP else (-BOX_STEP_Z, 0.0))
    (info,) = scene.bodies
    assert info.height_mm == pytest.approx(BOX_STEP_Z) and info.outline_source == "F.Fab" and info.models == (MODEL,) and info.reason == ""
    assert info.envelopes[0][1] == _approx_box(Box3(-1.0, -1.5, -0.4, 3.0, 0.5, BOX_STEP_Z))
    # the pads and the silk follow the same side
    pads = _only(scene, "pad")
    assert len(pads) == 2 and {p.faces for p in pads} == {"top" if side == BoardSide.TOP else "bottom"}
    assert {s.faces for s in _only(scene, "silk")} == {"top" if side == BoardSide.TOP else "bottom"}


@pytest.mark.parametrize(
    ("model", "height"),
    [
        (_model(MODEL, scale=(1, 1, 2)), 3.0),
        (_model(MODEL, offset=(0, 0, 1)), 2.5),
        (_model(MODEL, rotate=(90, 0, 0)), 1.5),  # x rotation: the envelope's y (-1.5..0.5) becomes z, negated: max 1.5
        (_model(MODEL, rotate=(0, 0, 90)), BOX_STEP_Z),  # a z rotation does not change the height
    ],
)
def test_the_model_transform_sets_the_body_height(tmp_path: Path, model, height: float):
    lib = _library(tmp_path / "kicad", box_model=model)
    _box_step(tmp_path / "3d")
    scene = build_scene(_ir([("U1", "BOX", _place("U1"))]), lib, model_dir=tmp_path / "3d")
    assert scene.bodies[0].height_mm == pytest.approx(height)


def test_no_step_file_gives_a_flat_outline_and_says_why(world):
    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1")), ("J1", "THT", _place("J1", x=20.0)), ("X1", "BARE", _place("X1", x=25.0, y=15.0))])
    scene = build_scene(ir, lib, model_dir=models)
    by_ref = {b.ref: b for b in scene.bodies}
    assert by_ref["U1"].height_mm == pytest.approx(BOX_STEP_Z)
    assert by_ref["J1"].height_mm is None and by_ref["J1"].outline_source == "courtyard" and "STEP 파일이 없음" in by_ref["J1"].reason
    assert by_ref["X1"].height_mm is None and by_ref["X1"].outline_source == "" and "코트야드" in by_ref["X1"].reason
    outline = _only(scene, "outline", "J1")
    assert len(outline) == 4 and all(s.faces == "top" and s.material == "outline" for s in outline)
    assert _only(scene, "body", "J1") == [] and _only(scene, "outline", "X1") == []
    assert [b.ref for b in scene.bodies_with_step] == ["U1"] and [b.ref for b in scene.bodies_without_step] == ["J1", "X1"]
    assert any("J1" in n and "평면 외곽선" in n for n in scene.notes)
    # no 3D library at all: every body is an outline and the notes say so
    bare = build_scene(ir, lib, model_dir=None)
    assert not bare.model_dir_found and bare.bodies_with_step == []
    assert any("찾지 못함" in n and "U1" in n and "J1" in n for n in bare.notes)
    # a hidden model is not a body
    hidden = build_scene(_ir([("U1", "BOX", _place("U1"))]), _library(Path(str(models) + "_h"), box_model=_model(MODEL, hide=True)), model_dir=models)
    assert hidden.bodies[0].height_mm is None and "숨김" in hidden.bodies[0].reason


def test_pads_drills_tracks_vias_and_mask(world):
    lib, models = world
    tracks = [
        Track(net="A", layer="F.Cu", start=(1.0, 1.0), end=(5.0, 1.0), width_mm=0.25, provenance=USER),
        Track(net="B", layer="B.Cu", start=(1.0, 3.0), end=(1.0, 6.0), width_mm=0.3, provenance=USER),
        Track(net="C", layer="In1.Cu", start=(2.0, 3.0), end=(2.0, 6.0), width_mm=0.3, provenance=USER),
    ]
    vias = [Via(net="A", x_mm=5.0, y_mm=1.0, drill_mm=0.3, diameter_mm=0.6, provenance=USER)]
    ir = _ir([("J1", "THT", _place("J1", x=20.0))], tracks=tracks, vias=vias)
    scene = build_scene(ir, lib, model_dir=models)
    T = DEFAULT_BOARD_THICKNESS_MM
    (slab,) = _only(scene, "slab")
    assert _xy_box(slab) == (0.0, 0.0, 30.0, 20.0) and (slab.z0, slab.z1, slab.faces) == (0.0, T, "all")
    masks = _only(scene, "mask")
    assert sorted((m.faces, m.z0, m.z1) for m in masks) == sorted([("top", T, pytest.approx(T + MASK_MM)), ("bottom", -MASK_MM, 0.0)])
    top_track, bottom_track = _only(scene, "track")
    assert (top_track.label, top_track.faces, top_track.z1) == ("A", "top", pytest.approx(T + COPPER_MM))
    assert (bottom_track.label, bottom_track.faces, bottom_track.z0) == ("B", "bottom", pytest.approx(-COPPER_MM))
    assert _xy_box(top_track) == pytest.approx((0.875, 0.875, 5.125, 1.125))  # round caps: half the width past each end
    assert any("안쪽 층 트랙 1개" in n for n in scene.notes)
    rings = _only(scene, "via")
    assert sorted(r.faces for r in rings) == ["bottom", "top"] and {r.material for r in rings} == {"copper"}  # tented: under the mask
    drills = _only(scene, "drill")
    assert len(drills) == 4  # two plated pads, the unplated hole, the via
    assert all(d.faces == "all" and d.z0 < 0 and d.z1 > T and len(d.polygon) == 8 for d in drills)
    pads = _only(scene, "pad")
    assert len(pads) == 4 and {p.label for p in pads} == {"J1.1", "J1.2"}  # the unplated hole has no copper
    assert {p.faces for p in pads} == {"top", "bottom"} and all(p.material == "pad" for p in pads)
    top_pad = next(p for p in pads if p.faces == "top")
    assert top_pad.z1 == pytest.approx(T + PAD_MM) and top_pad.z1 > T + MASK_MM  # an opened pad lies above the mask
    silk = _only(scene, "silk", "J1")
    assert silk and all(s.z1 == pytest.approx(T + SILK_MM) for s in silk)


def _inside_convex(poly: tuple, pt: tuple[float, float], tol: float = 1e-6) -> bool:
    n = len(poly)
    crosses = [(poly[(i + 1) % n][0] - poly[i][0]) * (pt[1] - poly[i][1]) - (poly[(i + 1) % n][1] - poly[i][1]) * (pt[0] - poly[i][0]) for i in range(n)]
    return all(c >= -tol for c in crosses) or all(c <= tol for c in crosses)


def test_custom_and_trapezoid_pads_are_drawn_around_their_real_copper(tmp_path: Path):
    """A custom pad's copper is its anchor plus its primitives, a trapezoid's reaches |rect_delta| / 2 past its size, an offset pad's
    copper is not centred on its hole: the drawn pad holds every primitive point and every trapezoid corner, on both sides."""
    root = tmp_path / "kicad"
    _library(root)
    odd = _footprint(
        "ODD",
        S("fp_rect", S("start", -3.5, -1.5), S("end", 6.5, 2.5), _stroke(0.05), S("fill", "no"), S("layer", Q("F.CrtYd"))),
        # a 0.2 mm anchor with a polygon, a circle and an arc around it (the arc bulges to x = -0.8 at its mid point)
        S("pad", Q("1"), "smd", "custom", S("at", -2, 0), S("size", 0.2, 0.2), S("layers", Q("F.Cu"), Q("F.Mask")),
          S("options", S("clearance", "outline"), S("anchor", "rect")),
          S("primitives",
            S("gr_poly", S("pts", S("xy", -0.3, -0.6), S("xy", 0.9, -0.4), S("xy", 0.5, 0.7)), S("width", 0.1), S("fill", "yes")),
            S("gr_circle", S("center", 0, 1.0), S("end", 0.3, 1.0), S("width", 0)),
            S("gr_arc", S("start", -0.5, 0), S("mid", -0.8, 0.3), S("end", -0.5, 0.6), S("width", 0.1)))),
        # KiCad 10's AMS_LGA-10-1EP pad 11: size 1 x 0.3, rect_delta 0 0.3 - one edge 1.3 mm wide, the other 0.7 mm
        S("pad", Q("2"), "smd", "trapezoid", S("at", 2, 0), S("size", 1, 0.3), S("rect_delta", 0, 0.3), S("layers", Q("F.Cu"), Q("F.Mask"))),
        S("pad", Q("3"), "thru_hole", "rect", S("at", 5, 0), S("size", 1, 1.8), S("drill", 0.75, S("offset", 0, 0.4)), S("layers", Q("*.Cu"), Q("*.Mask"))),
    )
    sexpr.dump_file(odd, root / "footprints" / "Test_Lib.pretty" / "ODD.kicad_mod")
    lib = KicadLibrary(roots=[root])
    for side, sign in ((BoardSide.TOP, 1.0), (BoardSide.BOTTOM, -1.0)):
        scene = build_scene(_ir([("U1", "ODD", _place("U1", side=side))]), lib, model_dir=None)
        pads = {p.label: p for p in _only(scene, "pad")}
        trap, offset = pads["U1.2"], pads["U1.3"]
        # a custom pad is drawn as the boxes every user of it reads (geometry.custom_pad_parts): the anchor's and each primitive's
        customs = [p for p in _only(scene, "pad") if p.label == "U1.1"]
        assert len(customs) == 4  # the anchor, the polygon, the circle, the arc
        # every primitive point (the polygon's vertices grown by half its 0.1 mm width, the circle's extremes, the arc's) is inside one
        prim = [(-0.35, -0.65), (0.95, -0.45), (0.55, 0.75), (-0.3, 1.0), (0.3, 1.0), (0.0, 1.3), (0.0, 0.7), (-0.85, 0.3), (-0.55, 0.65)]
        assert all(any(_inside_convex(c.polygon, (10.0 - 2.0 + x, 10.0 + sign * y)) for c in customs) for x, y in prim), [c.polygon for c in customs]
        box = (8.0 - 0.85, 10.0 - 0.65, 8.0 + 0.95, 10.0 + 1.3) if sign > 0 else (8.0 - 0.85, 10.0 - 1.3, 8.0 + 0.95, 10.0 + 0.65)
        boxes = [_xy_box(c) for c in customs]
        assert (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)) == pytest.approx(box)
        corners = {(round(12.0 + x, 6), round(10.0 + sign * y, 6)) for x, y in ((-0.65, 0.15), (-0.35, -0.15), (0.35, -0.15), (0.65, 0.15))}
        assert set(trap.polygon) == corners
        assert _xy_box(offset) == pytest.approx((14.5, 10.0 + sign * 0.4 - 0.9, 15.5, 10.0 + sign * 0.4 + 0.9))
        assert any(_xy_box(d) == pytest.approx((15.0 - 0.375, 10.0 - 0.375, 15.0 + 0.375, 10.0 + 0.375), abs=0.01) for d in _only(scene, "drill"))
        assert "사용자 정의 모양 패드 1개는 앵커와 각 구리 도형을 감싸는 사각형(외접 사각형)들로 그림" in scene.notes
        assert not any("사다리꼴" in n or "(size) 사각형" in n for n in scene.notes)


def test_board_thickness_comes_from_grounded_manufacturing_data_only(world):
    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1"))])
    assert any("가정" in n and "1.6 mm" in n for n in build_scene(ir, lib, model_dir=models).notes)
    ir.pcb.manufacturing.board_thickness_mm = Traced[float](value=1.0, unit="mm", provenance=Provenance(kind=ProvenanceKind.ASSUMPTION, note="guess"))
    assume = build_scene(ir, lib, model_dir=models)
    assert assume.thickness_mm == DEFAULT_BOARD_THICKNESS_MM and not assume.thickness_grounded
    ir.pcb.manufacturing.board_thickness_mm = Traced[float](value=1.0, unit="mm", provenance=USER)
    grounded = build_scene(ir, lib, model_dir=models)
    assert grounded.thickness_mm == 1.0 and grounded.thickness_grounded and not any("가정" in n for n in grounded.notes)
    (body,) = _only(grounded, "body")
    assert (body.z0, body.z1) == pytest.approx((1.0, 1.0 + BOX_STEP_Z))


def test_the_default_thickness_is_the_pcb_compilers():
    from ai_eda.compilers.pcb import DEFAULT_BOARD_THICKNESS_MM as COMPILER_DEFAULT

    assert DEFAULT_BOARD_THICKNESS_MM == COMPILER_DEFAULT


def test_scene_refuses_what_the_compiler_refuses_and_changes_nothing(world):
    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1"))])
    before = (ir.model_dump_json(), ir.content_hash())
    build_scene(ir, lib, model_dir=models)
    assert (ir.model_dump_json(), ir.content_hash()) == before and ir.artifacts == {}
    no_outline = ir.model_copy(deep=True)
    no_outline.pcb.outline = None
    no_place = ir.model_copy(deep=True)
    no_place.pcb.placements = []
    unknown = ir.model_copy(deep=True)
    unknown.components[0].footprint = LibraryRef(library="Test_Lib", name="NOPE")
    for bad, words in ((no_outline, "outline"), (no_place, "placement"), (unknown, "not found")):
        with pytest.raises(SceneError, match=words):
            build_scene(bad, lib, model_dir=models)
    with pytest.raises(ValueError):  # a SceneError is also a ValueError (and a CompileError)
        build_scene(no_outline, lib, model_dir=models)


def test_ir_silk_texts_are_omitted_in_3d_and_said_so(world):
    from ai_eda.ir import SilkText

    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1"))])
    plain = build_scene(ir, lib, model_dir=models)
    # a text on the fab layer is not silk: it is not counted
    ir.pcb.silkscreen = [SilkText(text="m3d", x_mm=5.0, y_mm=5.0, provenance=USER),
                         SilkText(text="U1", x_mm=5.0, y_mm=5.0, layer="F.Fab", kind="reference", component_ref="U1", provenance=USER)]
    scene = build_scene(ir, lib, model_dir=models)
    assert scene.solids == plain.solids and any("IR 실크 문자 1개" in n for n in scene.notes)


# --------------------------------------------------------------------------- faces / triangulation


def test_faces_wind_counter_clockwise_seen_from_outside():
    solid = Solid("body", "body", ((0.0, 0.0), (2.0, 0.0), (2.0, -1.0), (0.0, -1.0)), 0.0, 1.0, "all", "X")  # ccw in y-up
    faces = solid_faces(solid)
    assert [f.which for f in faces] == ["top", "bottom", "side", "side", "side", "side"]
    centre = (1.0, 0.5, 0.5)
    for f in faces:
        a, b, c = f.points[:3]
        cross = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]), (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        assert sum(x * y for x, y in zip(cross, f.normal)) > 0
        mid = [sum(p[i] for p in f.points) / len(f.points) for i in range(3)]
        assert sum((mid[i] - centre[i]) * f.normal[i] for i in range(3)) > 0  # outward
    assert [f.which for f in solid_faces(Solid("track", "copper", solid.polygon, 1.6, 1.635, "top", "n"))] == ["top"]


def test_triangulate_handles_a_concave_polygon():
    # an L shape, counter-clockwise seen from the top (board y down)
    poly = ((0.0, 0.0), (0.0, -2.0), (2.0, -2.0), (2.0, -1.0), (1.0, -1.0), (1.0, 0.0))
    poly = tuple(reversed(poly))
    tris = triangulate(poly)
    assert len(tris) == 4
    area = 0.0
    for a, b, c in tris:
        (x1, y1), (x2, y2), (x3, y3) = [(poly[i][0], -poly[i][1]) for i in (a, b, c)]
        signed = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2
        assert signed > 0
        area += signed
    assert area == pytest.approx(3.0)


# --------------------------------------------------------------------------- GLB


def parse_glb(blob: bytes) -> tuple[dict, bytes]:
    magic, version, length = struct.unpack_from("<III", blob, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(blob))
    json_len, json_type = struct.unpack_from("<II", blob, 12)
    assert json_type == 0x4E4F534A and json_len % 4 == 0
    doc = json.loads(blob[20 : 20 + json_len].decode("ascii"))
    off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", blob, off)
    assert bin_type == 0x004E4942 and bin_len % 4 == 0 and off + 8 + bin_len == length
    assert doc["buffers"][0]["byteLength"] <= bin_len and "uri" not in doc["buffers"][0]
    return doc, blob[off + 8 : off + 8 + bin_len]


def _accessor(doc: dict, binary: bytes, index: int) -> list:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    fmt, width = {5126: ("f", 4), 5123: ("H", 2), 5125: ("I", 4)}[acc["componentType"]]
    n = {"VEC3": 3, "SCALAR": 1}[acc["type"]]
    assert view["byteOffset"] % 4 == 0 and view["byteOffset"] + view["byteLength"] <= len(binary)
    assert view["byteLength"] == acc["count"] * n * width
    values = struct.unpack_from(f"<{acc['count'] * n}{fmt}", binary, view["byteOffset"])
    return [values[i : i + n] for i in range(0, len(values), n)] if n > 1 else list(values)


def test_write_glb_is_a_valid_deterministic_gltf_binary(world, tmp_path: Path):
    lib, models = world
    tracks = [Track(net="A", layer="F.Cu", start=(1.0, 1.0), end=(5.0, 1.0), width_mm=0.25, provenance=USER)]
    ir = _ir([("U1", "BOX", _place("U1", 90.0)), ("J1", "THT", _place("J1", x=20.0))], tracks=tracks)
    scene = build_scene(ir, lib, model_dir=models)
    blob = write_glb(scene)
    assert blob == write_glb(build_scene(ir, KicadLibrary(roots=lib.roots), model_dir=models))  # fresh caches, same bytes
    doc, binary = parse_glb(blob)
    assert doc["asset"] == {"version": "2.0", "generator": "ai_eda.tools.model3d glb 0.1"}
    assert str(tmp_path) not in blob.decode("latin-1")
    names = [n["name"] for n in doc["nodes"]]
    assert names == ["board", "mask", "drill", "copper", "pad", "silk", "body", "outline"]
    assert {n["name"]: n["extras"]["group"] for n in doc["nodes"]} == {
        "board": "board", "mask": "board", "drill": "board", "copper": "copper", "pad": "copper", "silk": "silk", "body": "parts", "outline": "parts"
    }
    mats = {m["name"]: m for m in doc["materials"]}
    assert mats["mask"]["alphaMode"] == "BLEND" and "alphaMode" not in mats["body"]
    extras = doc["scenes"][0]["extras"]
    assert extras["project"] == "m3d" and extras["board_thickness_assumed"] is True and BODY_CAPTION in extras["notes"]
    for node in doc["nodes"]:
        prim = doc["meshes"][node["mesh"]]["primitives"][0]
        assert prim["mode"] == 4 and doc["materials"][prim["material"]]["name"] == node["name"]
        pos = _accessor(doc, binary, prim["attributes"]["POSITION"])
        nor = _accessor(doc, binary, prim["attributes"]["NORMAL"])
        idx = _accessor(doc, binary, prim["indices"])
        acc = doc["accessors"][prim["attributes"]["POSITION"]]
        assert acc["min"] == [min(p[i] for p in pos) for i in range(3)] and acc["max"] == [max(p[i] for p in pos) for i in range(3)]
        assert len(idx) % 3 == 0 and max(idx) < len(pos) and len(nor) == len(pos)
        for k in range(0, len(idx), 3):
            a, b, c = (pos[i] for i in idx[k : k + 3])
            n = nor[idx[k]]
            assert nor[idx[k + 1]] == n == nor[idx[k + 2]]  # flat faces
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            length = math.sqrt(sum(x * x for x in cross))
            assert length > 0 and sum(x * y for x, y in zip(cross, n)) / length > 0.99, (node["name"], k)
    # the frame: glTF X = board x, Y = up, Z = board y, metres - the rotated body of U1 at hand values
    body = next(n for n in doc["nodes"] if n["name"] == "body")
    acc = doc["accessors"][doc["meshes"][body["mesh"]]["primitives"][0]["attributes"]["POSITION"]]
    T = DEFAULT_BOARD_THICKNESS_MM
    assert acc["min"] == pytest.approx([0.0095, T / 1000, 0.007], abs=1e-7) and acc["max"] == pytest.approx([0.0115, (T + BOX_STEP_Z) / 1000, 0.011], abs=1e-7)


def test_glb_writes_nothing_machine_dependent(world):
    lib, models = world
    scene = build_scene(_ir([("U1", "BOX", _place("U1")), ("J1", "THT", _place("J1", x=20.0))]), lib, model_dir=models)
    doc, _ = parse_glb(write_glb(scene))
    text = json.dumps(doc, ensure_ascii=False)
    assert str(models) not in text and str(lib.roots[0]) not in text and not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}", text)
    assert any("J1" in n for n in doc["scenes"][0]["extras"]["notes"])  # the missing model is named by component, not by path


# --------------------------------------------------------------------------- iso SVG


def test_iso_svg_is_valid_deterministic_and_captioned(world, tmp_path: Path):
    lib, models = world
    ir = _ir([("U1", "BOX", _place("U1")), ("J1", "THT", _place("J1", x=20.0))])
    scene = build_scene(ir, lib, model_dir=models)
    for view in VIEWS:
        svg = iso_svg(scene, view)
        assert svg == iso_svg(build_scene(ir, KicadLibrary(roots=lib.roots), model_dir=models), view)
        root = ET.fromstring(svg)
        assert root.tag == f"{SVG_NS}svg" and root.find(f"{SVG_NS}title").text == f"m3d: 3D 미리보기 ({view})"
        polygons = list(root.iter(f"{SVG_NS}polygon"))
        assert polygons and str(tmp_path) not in svg
        classes = [p.get("class").split()[0] for p in polygons]
        if view in ("iso", "top"):
            assert classes[0] == "k-slab" and "k-body" in classes
            first_body = classes.index("k-body")
            assert all(c == "k-body" for c in classes[first_body:])  # nothing flat is painted over a body
            assert {p.get("data-ref") for p in polygons if p.get("class").startswith("k-body")} == {"U1"}
    text = " ".join(t.text for t in ET.fromstring(iso_svg(scene)).iter(f"{SVG_NS}text"))
    assert "부품 = F.Fab 외곽" in text and "평면 외곽선 1개" in text
    assert "caption" not in iso_svg(scene, caption=False)
    assert BODY_CAPTION in scene_caption(scene) and "판정하지 않습니다" in scene_caption(scene)
    with pytest.raises(ValueError, match="unknown view"):
        iso_svg(scene, "side")


def test_iso_svg_paints_far_bodies_before_near_ones():
    def prism(kind: str, material: str, x1: float, y1: float, x2: float, y2: float, z0: float, z1: float, label: str) -> Solid:
        return Solid(kind, material, _clean([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]), z0, z1, "all", label)

    slab = prism("slab", "board", 0.0, 0.0, 10.0, 10.0, 0.0, 1.6, "board")
    far = prism("body", "body", 0.0, 0.0, 4.0, 2.0, 1.6, 6.0, "FAR")  # small board y: away from an eye in front (+y)
    near = prism("body", "body", 0.0, 6.0, 4.0, 8.0, 1.6, 2.0, "NEAR")
    for order in ([far, near], [near, far]):
        scene = Scene("s", (0.0, 0.0, 10.0, 10.0), 1.6, False, solids=[slab, *order])
        polygons = list(ET.fromstring(iso_svg(scene, caption=False)).iter(f"{SVG_NS}polygon"))
        refs = [p.get("data-ref") for p in polygons if p.get("data-ref")]
        assert refs.index("FAR") < refs.index("NEAR") and refs[-1] == "NEAR"
        assert polygons[0].get("class") == "k-slab g-board"


# --------------------------------------------------------------------------- kicad-cli 3D route (command construction; the real binary: tests/test_kicad_3d_export.py)


def _fake_run(content: bytes | None):
    calls: list[tuple[list[str], tuple[int, ...]]] = []

    def run(args, timeout=600, ok_codes=(0, 5)):
        calls.append((list(args), ok_codes))
        if content is not None:
            Path(args[args.index("--output") + 1]).write_bytes(content)
        return subprocess.CompletedProcess(args, 0, "", "")

    return calls, run


def test_kicad_3d_exports_pass_the_documented_flags_and_check_the_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kicad = KicadCli(binary="kicad-cli-not-run")
    pcb = tmp_path / "b.kicad_pcb"
    calls, run = _fake_run(b"ISO-10303-21;\nHEADER;")
    monkeypatch.setattr(kicad, "_run", run)
    out = tmp_path / "out" / "b.step"
    assert kicad.export_step(pcb, out) == out
    assert calls[-1] == (["pcb", "export", "step", *STEP_EXPORT_FLAGS, "--output", str(out), str(pcb)], (0,))
    calls, run = _fake_run(b"glTF\x02\x00\x00\x00")
    monkeypatch.setattr(kicad, "_run", run)
    assert kicad.export_glb(pcb, tmp_path / "b.glb") == tmp_path / "b.glb"
    assert calls[-1][0][:3 + len(GLB_EXPORT_FLAGS)] == ["pcb", "export", "glb", *GLB_EXPORT_FLAGS]
    calls, run = _fake_run(PNG_SIGNATURE + b"rest")
    monkeypatch.setattr(kicad, "_run", run)
    kicad.render(pcb, tmp_path / "top.png", "bottom", width=800, height=600, zoom=1.5, perspective=True)
    assert calls[-1][0] == ["pcb", "render", "--side", "bottom", "--width", "800", "--height", "600", "--quality", "basic", "--background", "opaque",
                            "--zoom", "1.5", "--perspective", "--output", str(tmp_path / "top.png"), str(pcb)]
    for bad in ({"side": "under"}, {"quality": "ultra"}, {"background": "red"}, {"width": 0}, {"zoom": -1.0}):
        n = len(calls)
        with pytest.raises(ValueError):
            kicad.render(pcb, tmp_path / "x.png", **({"side": "top"} | bad))
        assert len(calls) == n  # refused before running anything


def test_kicad_3d_exports_refuse_stale_or_wrong_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kicad = KicadCli(binary="kicad-cli-not-run")
    pcb = tmp_path / "b.kicad_pcb"
    out = tmp_path / "b.step"
    out.write_bytes(b"ISO-10303-21; an old export")
    _, run = _fake_run(None)  # the tool writes nothing
    monkeypatch.setattr(kicad, "_run", run)
    with pytest.raises(ToolExecutionError, match="wrote no file"):
        kicad.export_step(pcb, out)
    assert not out.exists()  # the stale file was removed, never passed off as fresh
    _, run = _fake_run(b"<html>not a model</html>")
    monkeypatch.setattr(kicad, "_run", run)
    with pytest.raises(ToolExecutionError, match="does not start with"):
        kicad.export_glb(pcb, tmp_path / "b.glb")


@pytest.mark.skipif(os.name != "posix", reason="the fake kicad-cli is a #! script")
def test_kicad_3d_export_refuses_exit_code_5_and_reads_help_flags(tmp_path: Path):
    script = tmp_path / "kicad-cli"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nargs = sys.argv[1:]\n"
        "if '--help' in args:\n    print('Usage: step [--help] [--force] [--subst-models] [--include-tracks] [-o OUT]')\n    sys.exit(0)\n"
        "open(args[args.index('--output') + 1], 'wb').write(b'ISO-10303-21;')\nsys.exit(5)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    kicad = KicadCli(binary=str(script))
    assert {"--help", "--force", "--subst-models", "--include-tracks"} <= kicad.accepted_flags(["pcb", "export", "step"])
    with pytest.raises(ToolExecutionError, match="exited 5"):
        kicad.export_step(tmp_path / "b.kicad_pcb", tmp_path / "b.step")


def test_footprint_model_references_are_read_as_written():
    from ai_eda.tools.kicad.library import FootprintDef

    def fp(*models) -> FootprintDef:
        return FootprintDef(lib_id="T:X", name="X", node=_footprint("X", *models), pads=[], attr="smd", courtyard=None)

    legacy = S("model", Q("${KICAD10_3DMODEL_DIR}/A.3dshapes/X.wrl"), S("at", S("xyz", 1, 0, 0.5)), S("scale", S("xyz", 1, 1, 1)), S("rotate", S("xyz", 0, 0, 0)))
    bare_hide = S("model", Q("B.step"), "hide", S("offset", S("xyz", 0, 0, 0)))
    (a, b, c) = footprint_models(fp(legacy, bare_hide, _model(MODEL, offset=(0.5, -1, 2), rotate=(0, 0, 45), hide=True)))
    assert a.offset == pytest.approx((25.4, 0.0, 12.7)) and not a.is_step and not a.hidden  # legacy (at ...) is in inches
    assert b.hidden and b.is_step and b.scale == (1.0, 1.0, 1.0)
    assert (c.path, c.offset, c.rotate, c.hidden) == (MODEL, (0.5, -1.0, 2.0), (0.0, 0.0, 45.0), True)
    with pytest.raises(ValueError, match="T:X"):
        footprint_models(fp(S("model", Q("C.step"), S("offset", S("xyz", 0, "nan", 0)))))


def test_a_fab_drawing_without_area_falls_back_to_the_courtyard(tmp_path: Path):
    root = tmp_path / "kicad"
    _library(root)  # writes the Test_Lib footprints; LINE is added beside them
    line = _footprint(
        "LINE",
        S("fp_line", S("start", -1, 0), S("end", 1, 0), _stroke(), S("layer", Q("F.Fab"))),
        S("fp_rect", S("start", -1.5, -1), S("end", 1.5, 1), _stroke(0.05), S("fill", "no"), S("layer", Q("F.CrtYd"))),
        _smd("1", 0.0, 0.0),
        _model(MODEL),
    )
    sexpr.dump_file(line, root / "footprints" / "Test_Lib.pretty" / "LINE.kicad_mod")
    _box_step(tmp_path / "3d")
    scene = build_scene(_ir([("U9", "LINE", _place("U9"))]), KicadLibrary(roots=[root]), model_dir=tmp_path / "3d")
    (body,) = _only(scene, "body")
    assert scene.bodies[0].outline_source == "courtyard" and _xy_box(body) == pytest.approx((8.5, 9.0, 11.5, 11.0))
