"""PCB compiler, board geometry and the naive placeholder router.

Pure tests run everywhere; tests that need the installed KiCad libraries or
``kicad-cli`` are skipped when they are absent and otherwise run against the
real binary (DRC, gerber and drill export).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_eda.compilers import ids
from ai_eda.compilers.base import CompileContext
from ai_eda.compilers.pcb import (
    FILE_VERSION,
    NON_COPPER_LAYERS,
    PCBCompiler,
    copper_layer_index,
    design_rules,
    net_numbers,
)
from ai_eda.errors import CompileError
from ai_eda.ir import (
    ArtifactKind,
    BoardOutline,
    BoardSide,
    CircuitIR,
    Layer,
    LibraryRef,
    ManufacturingConstraints,
    Net,
    PCBDesign,
    PinRef,
    Placement,
    Provenance,
    ProvenanceKind,
    SourceRef,
    Track,
    ValidationStatus,
    Via,
    Zone,
    assumption,
    authoritative,
)
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.geometry import (
    courtyard_bbox,
    footprint_angle,
    footprint_bbox,
    mirrored_layer,
    normalize_angle,
    pad_angle,
    pad_center,
    pads_bbox,
    rotate,
    text_angle,
    to_board,
)
from ai_eda.tools.kicad.library import BBox, KicadLibrary, Pad
from ai_eda.tools.manufacturing.outputs import check_drill_files, check_gerber_set
from ai_eda.tools.routing import NET_CLASS_TRACK_WIDTH_MM, route_naive, track_width_for
from tests.fixtures_kicad import PROJECT_ID, ROUTABLE_PLACEMENTS, SHORTING_PLACEMENTS, divider_with_connector_ir

LIB = KicadLibrary()
HAS_LIBS = LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and LIB.symbol_file("Device") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")

R0603 = LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric")
PINHEADER = LibraryRef(library="Connector_PinHeader_2.54mm", name="PinHeader_1x03_P2.54mm_Vertical")

# ROUTABLE_PLACEMENTS / SHORTING_PLACEMENTS live in tests.fixtures_kicad: the shared fixture is
# placed so the naive router (straight pad-centre chains, F.Cu only) connects it DRC-clean, and
# the first, shorting layout is kept as the negative case below.


# --------------------------------------------------------------------------- helpers / fixtures


def _place(ir: CircuitIR, placements: list[Placement]) -> CircuitIR:
    ir.pcb.placements = list(placements)
    ir.pcb.tracks = route_naive(ir, LIB)
    return ir


@pytest.fixture
def board_ir(tmp_path: Path) -> CircuitIR:
    """The shared divider-with-connector IR, placed for the naive router and routed."""
    if not HAS_LIBS:
        pytest.skip("KiCad libraries not installed")
    return _place(divider_with_connector_ir(tmp_path, LIB), ROUTABLE_PLACEMENTS)


@pytest.fixture
def ctx(tmp_path: Path) -> CompileContext:
    return CompileContext(workdir=tmp_path / "out", tools={"kicad_library": LIB})


def _compile(ir: CircuitIR, ctx: CompileContext):
    ref = PCBCompiler().compile(ir, ctx)
    return ref, Path(ref.path)


def _footprints(tree: list) -> dict[str, list]:
    out = {}
    for fp in sexpr.find_all(tree, "footprint"):
        ref = next(str(p[2]) for p in sexpr.find_all(fp, "property") if p[1] == "Reference")
        out[ref] = fp
    return out


def _pads(fp: list) -> dict[str, list]:
    return {str(p[1]): p for p in sexpr.find_all(fp, "pad")}


def _drc(pcb: Path, report: Path):
    res = kicad.run_drc(pcb, report, schematic_parity=False)
    print(f"\nDRC {pcb.name}: {res.message}")
    for w in res.details["warnings"]:
        print("  warning:", w.get("type"), "-", w.get("description"))
    for e in res.details["errors"]:
        print("  ERROR:", e.get("type"), "-", e.get("description"), [i.get("description") for i in e.get("items", [])])
    return res


# --------------------------------------------------------------------------- geometry (pure)

PAD1 = Pad("1", "smd", "roundrect", -0.825, 0.0, 0.0, 0.8, 0.95, None, ["F.Cu", "F.Mask", "F.Paste"], 0.25)
PAD2 = Pad("2", "smd", "roundrect", 0.825, 0.0, 0.0, 0.8, 0.95, None, ["F.Cu", "F.Mask", "F.Paste"], 0.25)
HDR3 = Pad("3", "thru_hole", "circle", 0.0, 5.08, 0.0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"], None)


def _pl(x: float, y: float, rot: float, side: BoardSide = BoardSide.TOP) -> Placement:
    return Placement(component_ref="X", x_mm=x, y_mm=y, rotation_deg=rot, side=side)


@pytest.mark.parametrize(
    "rot,expected1,expected2",
    [
        (0, (117.175, 105.0), (118.825, 105.0)),
        (90, (118.0, 105.825), (118.0, 104.175)),  # +90 = CCW on screen: the left pad moves below the centre
        (180, (118.825, 105.0), (117.175, 105.0)),
        (270, (118.0, 104.175), (118.0, 105.825)),
        (-90, (118.0, 104.175), (118.0, 105.825)),
    ],
)
def test_pad_center_rotation_top(rot, expected1, expected2):
    p = _pl(118, 105, rot)
    assert pad_center(p, PAD1) == expected1
    assert pad_center(p, PAD2) == expected2
    assert pad_angle(p, PAD1) == normalize_angle(rot)


def test_pad_center_rotation_matches_hand_values_for_header():
    # PinHeader pad 3 (0, 5.08): at (5, 10) rot 90 the pad row points to +x
    assert pad_center(_pl(5, 10, 90), HDR3) == (10.08, 10.0)
    assert pad_center(_pl(5, 10, 0), HDR3) == (5.0, 15.08)
    assert rotate(0.0, 5.08, 90) == (5.08, 0.0)
    assert rotate(-0.825, 0.0, 45) == (round(-0.825 * 0.5**0.5, 6), round(0.825 * 0.5**0.5, 6))


def test_pad_center_bottom_side():
    # J1 on the bottom rotated 180 lands its pads exactly where the top-side rot-0 header has them
    bottom180 = _pl(105, 105, 180, BoardSide.BOTTOM)
    assert pad_center(bottom180, Pad("2", "thru_hole", "circle", 0, 2.54, 0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"])) == (105.0, 107.54)
    assert pad_center(bottom180, HDR3) == (105.0, 110.08)
    # bottom rot 90 at (108, 112): stored rel (0, -2.54) -> (105.46, 112) (the value DRC confirmed with 0 unconnected)
    bottom90 = _pl(108, 112, 90, BoardSide.BOTTOM)
    assert pad_center(bottom90, Pad("2", "thru_hole", "circle", 0, 2.54, 0, 1.7, 1.7, 1.0, ["*.Cu", "*.Mask"])) == (105.46, 112.0)
    assert pad_center(bottom90, HDR3) == (102.92, 112.0)
    # bottom rot 0: y simply mirrors about the footprint origin
    assert pad_center(_pl(5, 6, 0, BoardSide.BOTTOM), HDR3) == (5.0, 0.92)
    assert to_board(_pl(5, 6, 0, BoardSide.BOTTOM), 1.0, 2.0) == (6.0, 4.0)
    # pad angle: PAD::Flip negates the library angle, then the footprint rotation is added
    rotated_pad = Pad("1", "smd", "rect", 0, 0, 30.0, 1, 2, None, ["F.Cu"], None)
    assert pad_angle(_pl(0, 0, 90, BoardSide.BOTTOM), rotated_pad) == 60.0
    assert pad_angle(_pl(0, 0, 90), rotated_pad) == 120.0


def test_text_angle_and_layer_mirroring():
    assert text_angle(_pl(0, 0, 90), 0) == 90.0
    assert text_angle(_pl(0, 0, 0), 90) == 90.0
    assert text_angle(_pl(0, 0, 0, BoardSide.BOTTOM), 0) == 180.0  # lib (0 -1.8 0) -> (0 1.8 180) on B.SilkS
    assert text_angle(_pl(0, 0, 180, BoardSide.BOTTOM), 0) == 0.0
    assert text_angle(_pl(0, 0, 90, BoardSide.BOTTOM), 90) == 180.0
    for layer, flipped in [("F.Cu", "B.Cu"), ("B.SilkS", "F.SilkS"), ("*.Cu", "*.Cu"), ("Edge.Cuts", "Edge.Cuts"), ("F.CrtYd", "B.CrtYd")]:
        assert mirrored_layer(layer, BoardSide.BOTTOM) == flipped
        assert mirrored_layer(layer, BoardSide.TOP) == layer
    assert normalize_angle(-90) == 270.0 and normalize_angle(360) == 0.0 and normalize_angle(450) == 90.0
    # footprint (at ..) orientation is (-180, 180] (KiCad writes -90, never 270); 180 stays 180
    assert [footprint_angle(a) for a in (0, 90, 180, 270, -90, -180, 360, 450)] == [0.0, 90.0, 180.0, -90.0, -90.0, 180.0, 0.0, 90.0]


@needs_libs
def test_courtyard_and_bboxes_in_board_frame():
    fp = LIB.load_footprint(R0603)
    assert courtyard_bbox(_pl(14, 6, 270), fp) == BBox(13.27, 4.52, 14.73, 7.48)
    assert courtyard_bbox(_pl(14, 6, 0), fp) == BBox(12.52, 5.27, 15.48, 6.73)
    assert pads_bbox(_pl(14, 6, 0), fp) == BBox(12.775, 5.525, 15.225, 6.475)
    assert pads_bbox(_pl(14, 6, 90), fp) == BBox(13.525, 4.775, 14.475, 7.225)
    assert footprint_bbox(_pl(14, 6, 0), fp) == BBox(12.52, 5.27, 15.48, 6.73)
    hdr = LIB.load_footprint(PINHEADER)
    assert courtyard_bbox(_pl(5, 6, 0), hdr) == BBox(3.23, 4.23, 6.77, 12.85)
    assert courtyard_bbox(_pl(5, 6, 0, BoardSide.BOTTOM), hdr) == BBox(3.23, -0.85, 6.77, 7.77)


# --------------------------------------------------------------------------- naive router (pure + libs)


def test_track_width_table():
    assert NET_CLASS_TRACK_WIDTH_MM["Default"] == 0.25
    assert track_width_for("Default") == 0.25 and track_width_for("NoSuchClass") == 0.25


@needs_libs
def test_naive_router_chains_sorted_pad_centres(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.placements = ROUTABLE_PLACEMENTS
    tracks = route_naive(ir, LIB)
    by_net = {}
    for t in tracks:
        by_net.setdefault(t.net, []).append(t)
    assert set(by_net) == {"VIN", "VOUT", "GND"}
    assert [t.net for t in tracks] == ["GND", "VIN", "VOUT", "VOUT"]  # nets in name order, chains in (x, y) order
    assert by_net["VIN"][0].start == (5.0, 6.0) and by_net["VIN"][0].end == (14.0, 5.175)
    assert by_net["VOUT"][0].start == (5.0, 8.54) and by_net["VOUT"][0].end == (14.0, 6.825)
    assert by_net["VOUT"][1].start == (14.0, 6.825) and by_net["VOUT"][1].end == (14.0, 10.255)
    assert by_net["GND"][0].start == (5.0, 11.08) and by_net["GND"][0].end == (14.0, 11.905)
    assert all(t.layer == "F.Cu" and t.width_mm == 0.25 for t in tracks)
    geometry = [(t.net, t.layer, t.start, t.end, t.width_mm) for t in tracks]
    assert [(t.net, t.layer, t.start, t.end, t.width_mm) for t in route_naive(ir, LIB)] == geometry  # deterministic
    # every generated segment is traceable to this router, its net and the placements it joined
    for t in tracks:
        p = t.provenance
        assert p.kind is ProvenanceKind.DERIVED and p.tool == "routing.naive" and p.tool_version and not p.needs_verification
        assert f"net:{t.net}" in p.derived_from and any(d.startswith("placement:") for d in p.derived_from)


@needs_libs
def test_naive_router_refuses_to_guess(tmp_path: Path):
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.placements = [p for p in ROUTABLE_PLACEMENTS if p.component_ref != "R2"]
    with pytest.raises(CompileError, match="R2.*no placement"):
        route_naive(ir, LIB)
    ir.pcb.placements = ROUTABLE_PLACEMENTS
    ir.nets[0].pins.append(PinRef(component_ref="R1", pin_number="7"))
    with pytest.raises(CompileError, match="no pad '7'"):
        route_naive(ir, LIB)
    ir.pcb = None
    with pytest.raises(CompileError, match="ir.pcb is None"):
        route_naive(ir, LIB)


# --------------------------------------------------------------------------- compiler (libs, no kicad-cli)


@needs_libs
def test_compile_writes_board_with_ir_hash(board_ir: CircuitIR, ctx: CompileContext):
    ref, path = _compile(board_ir, ctx)
    assert path == ctx.workdir / f"{PROJECT_ID}.kicad_pcb" and path.exists()
    assert ref.kind is ArtifactKind.PCB and ref.generator == "compiler.kicad_pcb"
    assert ref.generated_from_ir_hash == board_ir.content_hash()
    assert ref.content_hash == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert ref.matches_disk()
    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b")\n")
    tree = sexpr.parse(data.decode("utf-8"))
    assert sexpr.head(tree) == "kicad_pcb"
    assert sexpr.get(tree, "version") == str(FILE_VERSION)
    assert sexpr.get(tree, "generator") == "pcbnew" and sexpr.get(tree, "generator_version") == "10.0"
    assert sexpr.find(tree, "net") is None  # 20260206: nets are referenced by name only
    assert sexpr.dumps(tree) == data.decode("utf-8")  # the writer's layout is KiCad's own


@needs_libs
def test_compile_is_deterministic(board_ir: CircuitIR, tmp_path: Path):
    ctx_a = CompileContext(workdir=tmp_path / "a", tools={"kicad_library": LIB})
    ctx_b = CompileContext(workdir=tmp_path / "b", tools={"kicad_library": KicadLibrary()})  # fresh caches
    ref_a, path_a = _compile(board_ir, ctx_a)
    ref_b, path_b = _compile(board_ir.model_copy(deep=True), ctx_b)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert ref_a.content_hash == ref_b.content_hash
    # every uuid is a uuid5 from ids: recompiling with a different project id changes all of them
    ir2 = board_ir.model_copy(deep=True)
    ir2.project.id = "other"
    _, path_c = _compile(ir2, CompileContext(workdir=tmp_path / "c", tools={"kicad_library": LIB}))
    uuids_a = {str(u[1]) for u in _all_nodes(sexpr.parse_file(path_a), "uuid")}
    uuids_c = {str(u[1]) for u in _all_nodes(sexpr.parse_file(path_c), "uuid")}
    assert uuids_a and uuids_a.isdisjoint(uuids_c)


def _all_nodes(node: list, name: str) -> list[list]:
    found = []
    for child in node:
        if isinstance(child, list):
            if sexpr.head(child) == name:
                found.append(child)
            found.extend(_all_nodes(child, name))
    return found


@needs_libs
def test_embedded_footprints_match_library_and_ir(board_ir: CircuitIR, ctx: CompileContext):
    _, path = _compile(board_ir, ctx)
    tree = sexpr.parse_file(path)
    fps = _footprints(tree)
    assert list(fps) == sorted(["R1", "R2", "J1"], key=lambda r: ids.footprint_uuid(PROJECT_ID, r))  # KiCad's order
    r1, j1 = fps["R1"], fps["J1"]
    assert r1[1] == "Resistor_SMD:R_0603_1608Metric" and j1[1] == "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"
    assert sexpr.get(r1, "layer") == "F.Cu" and sexpr.args(sexpr.find(r1, "at")) == ["14", "6", "-90"]  # 270 -> -90 like KiCad
    setup = sexpr.find(tree, "setup")
    assert sexpr.get(setup, "pad_to_mask_clearance") == "0" and sexpr.find(setup, "pcbplotparams") is not None
    assert sexpr.args(sexpr.find(j1, "at")) == ["5", "6"]  # rotation 0 is omitted
    assert sexpr.get(r1, "uuid") == ids.footprint_uuid(PROJECT_ID, "R1")
    assert sexpr.get(r1, "path") == "/" + ids.symbol_uuid(PROJECT_ID, "R1")
    assert sexpr.get(r1, "sheetname") == "/" and sexpr.get(r1, "sheetfile") == f"{PROJECT_ID}.kicad_sch"
    assert sexpr.args(sexpr.find(r1, "attr")) == ["smd"] and sexpr.args(sexpr.find(j1, "attr")) == ["through_hole"]
    assert sexpr.find(r1, "version") is None and sexpr.find(r1, "generator") is None
    props = {str(p[1]): p for p in sexpr.find_all(r1, "property")}
    assert [str(p[1]) for p in sexpr.find_all(r1, "property")] == ["Reference", "Value", "Datasheet", "Description", "KiLib_Generator"]
    assert props["Value"][2] == "10k" and props["Description"][2] == board_ir.component("R1").description
    assert sexpr.args(sexpr.find(props["Reference"], "at")) == ["0", "-1.43", "270"]  # library position, rotated text
    assert sexpr.get(props["Reference"], "uuid") and sexpr.get(props["Datasheet"], "hide") == "yes"
    # pads: geometry verbatim from the library, nets from the IR, deterministic uuids
    lib_fp = LIB.load_footprint(R0603)
    lib_pads = {str(p[1]): p for p in sexpr.find_all(lib_fp.node, "pad")}
    pads = _pads(r1)
    for number, pad in pads.items():
        lib_pad = lib_pads[number]
        assert sexpr.strict_equal(sexpr.find(pad, "size"), sexpr.find(lib_pad, "size"))
        assert sexpr.strict_equal(sexpr.find(pad, "layers"), sexpr.find(lib_pad, "layers"))
        assert sexpr.args(sexpr.find(pad, "at")) == [*sexpr.args(sexpr.find(lib_pad, "at")), "270"]
        assert sexpr.get(pad, "uuid") == ids.pad_uuid(PROJECT_ID, "R1", number)
        assert sexpr.get(pad, "pintype") == "passive"
        heads = [sexpr.head(c) for c in pad if isinstance(c, list)]
        assert heads.index("net") > heads.index("roundrect_rratio") and heads[-1] == "uuid"
    assert sexpr.get(pads["1"], "net") == "VIN" and sexpr.get(pads["2"], "net") == "VOUT"
    assert {n: sexpr.get(p, "net") for n, p in _pads(j1).items()} == {"1": "VIN", "2": "VOUT", "3": "GND"}
    assert {n: sexpr.get(p, "net") for n, p in _pads(fps["R2"]).items()} == {"1": "VOUT", "2": "GND"}
    # graphics verbatim (plus a uuid each)
    lib_lines = [sexpr.args(sexpr.find(g, "start")) for g in sexpr.find_all(lib_fp.node, "fp_line")]
    assert [sexpr.args(sexpr.find(g, "start")) for g in sexpr.find_all(r1, "fp_line")] == lib_lines
    assert all(sexpr.get(g, "uuid") for g in sexpr.find_all(r1, "fp_line") + sexpr.find_all(r1, "fp_rect"))
    assert sexpr.find(r1, "model") is not None and sexpr.find(r1, "embedded_fonts") is not None
    # board-level: outline, layers, tracks
    rect = sexpr.find(tree, "gr_rect")
    assert sexpr.args(sexpr.find(rect, "start")) == ["0", "0"] and sexpr.args(sexpr.find(rect, "end")) == ["30", "20"]
    assert sexpr.get(rect, "layer") == "Edge.Cuts"
    layers = sexpr.find(tree, "layers")
    assert [[str(a) for a in l] for l in layers[1:3]] == [["0", "F.Cu", "signal"], ["2", "B.Cu", "signal"]]
    assert [str(l[1]) for l in layers[3:]] == [name for _, name, _ in NON_COPPER_LAYERS]
    segments = sexpr.find_all(tree, "segment")
    assert len(segments) == len(board_ir.pcb.tracks) == 4
    assert {sexpr.get(s, "net") for s in segments} == {"VIN", "VOUT", "GND"}
    assert sexpr.get(segments[0], "uuid") == ids.net_item_uuid(PROJECT_ID, "track", 0)


@needs_libs
def test_bottom_side_footprint_is_flipped_like_kicad(board_ir: CircuitIR, ctx: CompileContext):
    board_ir.pcb.placements = [
        Placement(component_ref="J1", x_mm=5.0, y_mm=6.0, rotation_deg=180.0, side=BoardSide.BOTTOM),
        *ROUTABLE_PLACEMENTS[1:],
    ]
    board_ir.pcb.tracks = route_naive(board_ir, LIB)
    assert {t.start for t in board_ir.pcb.tracks} == {(5.0, 6.0), (5.0, 8.54), (5.0, 11.08), (14.0, 6.825)}  # same pad positions
    _, path = _compile(board_ir, ctx)
    j1 = _footprints(sexpr.parse_file(path))["J1"]
    assert sexpr.get(j1, "layer") == "B.Cu" and sexpr.args(sexpr.find(j1, "at")) == ["5", "6", "180"]
    pads = _pads(j1)
    assert sexpr.args(sexpr.find(pads["2"], "at")) == ["0", "-2.54", "180"]  # y negated, angle = rotation
    assert sexpr.args(sexpr.find(pads["3"], "layers")) == ["*.Cu", "*.Mask"]
    ref = next(p for p in sexpr.find_all(j1, "property") if p[1] == "Reference")
    assert sexpr.args(sexpr.find(ref, "at")) == ["0", "2.38", "0"] and sexpr.get(ref, "layer") == "B.SilkS"
    assert sexpr.args(sexpr.find(sexpr.find(ref, "effects"), "justify")) == ["mirror"]
    crtyd = next(g for g in sexpr.find_all(j1, "fp_rect") if sexpr.get(g, "layer") == "B.CrtYd")
    assert sexpr.args(sexpr.find(crtyd, "start")) == ["-1.77", "1.77"] and sexpr.args(sexpr.find(crtyd, "end")) == ["1.77", "-6.85"]
    fab_text = sexpr.find(j1, "fp_text")
    assert sexpr.get(fab_text, "layer") == "B.Fab" and sexpr.args(sexpr.find(fab_text, "at")) == ["0", "-2.54", "270"]
    silk = [g for g in sexpr.find_all(j1, "fp_line") if sexpr.get(g, "layer") == "B.SilkS"]
    assert len(silk) == 6 and all(sexpr.get(g, "layer") != "F.SilkS" for g in sexpr.find_all(j1, "fp_line"))


@needs_libs
def test_manufacturing_values_map_to_rules_only_when_authoritative(board_ir: CircuitIR, ctx: CompileContext):
    ds = SourceRef(title="JLCPCB capabilities", authority="JLCPCB")
    board_ir.pcb.manufacturing = ManufacturingConstraints(
        fab="JLCPCB",
        min_track_width_mm=authoritative(0.127, ds, "mm"),
        min_clearance_mm=assumption(0.127, "typical", "mm"),
        min_via_drill_mm=authoritative(0.3, ds, "mm"),
        min_via_diameter_mm=authoritative(0.5, ds, "mm"),
        min_hole_to_edge_mm=authoritative(0.5, ds, "mm"),
        board_thickness_mm=authoritative(1.2, ds, "mm"),
        copper_weight_oz=authoritative(1.0, ds, "oz"),
    )
    assert design_rules(board_ir) == {"min_track_width": 0.127, "min_through_hole_diameter": 0.3, "min_via_diameter": 0.5}
    _, path = _compile(board_ir, ctx)
    tree = sexpr.parse_file(path)
    assert sexpr.get(sexpr.find(tree, "general"), "thickness") == "1.2"
    board_ir.pcb.manufacturing = ManufacturingConstraints(board_thickness_mm=assumption(1.0, "guess", "mm"))
    assert design_rules(board_ir) == {}
    _, path = _compile(board_ir, ctx)
    assert sexpr.get(sexpr.find(sexpr.parse_file(path), "general"), "thickness") == "1.6"


@needs_libs
def test_vias_zones_and_net_numbers(board_ir: CircuitIR, ctx: CompileContext):
    assert net_numbers(board_ir) == {"": 0, "GND": 1, "VIN": 2, "VOUT": 3}
    # a GND via joined to J1.3 on both layers, plus an (unfilled) GND pour on B.Cu
    board_ir.pcb.vias = [Via(net="GND", x_mm=10.0, y_mm=15.0, drill_mm=0.3, diameter_mm=0.6)]
    board_ir.pcb.tracks += [
        Track(net="GND", layer="F.Cu", start=(5.0, 11.08), end=(10.0, 15.0), width_mm=0.25),
        Track(net="GND", layer="B.Cu", start=(10.0, 15.0), end=(5.0, 11.08), width_mm=0.25),
    ]
    board_ir.pcb.zones = [Zone(net="GND", layer="B.Cu", polygon=[(0.5, 0.5), (29.5, 0.5), (29.5, 19.5), (0.5, 19.5)])]
    _, path = _compile(board_ir, ctx)
    tree = sexpr.parse_file(path)
    via = sexpr.find(tree, "via")
    assert sexpr.args(sexpr.find(via, "at")) == ["10", "15"] and sexpr.args(sexpr.find(via, "layers")) == ["F.Cu", "B.Cu"]
    assert sexpr.get(via, "net") == "GND" and sexpr.get(via, "uuid") == ids.net_item_uuid(PROJECT_ID, "via", 0)
    assert {sexpr.get(s, "layer") for s in sexpr.find_all(tree, "segment")} == {"F.Cu", "B.Cu"}
    zone = sexpr.find(tree, "zone")
    assert sexpr.get(zone, "net") == "GND" and sexpr.get(zone, "layer") == "B.Cu"
    assert len(sexpr.find_all(sexpr.find(sexpr.find(zone, "polygon"), "pts"), "xy")) == 4
    assert sexpr.args(sexpr.find(zone, "connect_pads")) == ["yes"]
    assert copper_layer_index("In1.Cu") == 4 and copper_layer_index("In2.Cu") == 6
    with pytest.raises(CompileError):
        copper_layer_index("Edge.Cuts")
    if kicad.available():
        res = _drc(path, ctx.workdir / "drc_via_zone.json")
        assert res.details["errors"] == [] and res.status is ValidationStatus.PASS
        assert res.details["warnings"] == []


@needs_libs
def test_compile_errors_instead_of_guessing(board_ir: CircuitIR, ctx: CompileContext):
    ir = board_ir.model_copy(deep=True)
    ir.pcb = None
    with pytest.raises(CompileError, match="ir.pcb is None"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.component("R2").footprint = LibraryRef(library="Resistor_SMD", name="R_9999_NotAFootprint", verified=False)
    with pytest.raises(CompileError, match="R_9999_NotAFootprint.*'R2'.*not found"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)  # an IR that *claims* verification is re-checked on disk
    ir.component("R2").footprint = LibraryRef(library="Resistor_SMD", name="R_9999_NotAFootprint", verified=True, library_path="x")
    with pytest.raises(CompileError, match="not found"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.component("R2").footprint = None
    with pytest.raises(CompileError, match="'R2' has no footprint"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)  # net pin without a pad: add IR pin "3" to R1 and connect it
    r1 = ir.component("R1")
    r1.pins.append(r1.pins[0].model_copy(update={"number": "3"}))
    ir.nets[0].pins.append(PinRef(component_ref="R1", pin_number="3"))
    with pytest.raises(CompileError, match=r"IR pins \['3'\] have no pad"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)  # net references a pin the IR component does not define
    ir.nets[0].pins.append(PinRef(component_ref="R1", pin_number="3"))
    with pytest.raises(CompileError, match="R1.3, which is not an IR pin"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)  # footprint has a pad the IR does not know: header on a 2-pin part
    ir.component("R2").footprint = PINHEADER
    with pytest.raises(CompileError, match=r"pads \['3'\] that are not IR pins"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.pcb.placements = ROUTABLE_PLACEMENTS[:2]  # J1, R1 placed; R2 not
    with pytest.raises(CompileError, match="'R2' has no placement"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.pcb.outline = None
    with pytest.raises(CompileError, match="outline"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.pcb.tracks[0].net = "NOPE"
    with pytest.raises(CompileError, match="unknown net 'NOPE'"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.pcb.tracks[0].layer = "F.SilkS"
    with pytest.raises(CompileError, match="not a copper layer"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.pcb.layers = [Layer(name="F.Cu", kind="signal")]
    with pytest.raises(CompileError, match="F.Cu and B.Cu"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.nets.append(Net(name="VIN", pins=[], provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test")))
    with pytest.raises(CompileError, match="duplicate net names"):
        PCBCompiler().compile(ir, ctx)

    ir = board_ir.model_copy(deep=True)
    ir.project.id = "bad name/with slash"
    with pytest.raises(CompileError, match="file stem"):
        PCBCompiler().compile(ir, ctx)


# --------------------------------------------------------------------------- real kicad-cli


@needs_kicad
def test_real_drc_passes_with_zero_errors(board_ir: CircuitIR, ctx: CompileContext):
    _, path = _compile(board_ir, ctx)
    res = _drc(path, ctx.workdir / "drc.json")
    assert res.tool == "kicad-cli" and res.tool_version.startswith("10.")
    assert res.artifact_hash == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert res.details["errors"] == []
    assert res.status is ValidationStatus.PASS
    # a correct board of this size produces no warnings either (lib_footprint_mismatch would mean the
    # embedded copy differs from the library; silk/courtyard warnings would mean a placement problem)
    assert res.details["warnings"] == []
    report = (ctx.workdir / "drc.json").read_text(encoding="utf-8")
    assert '"unconnected_items": []' in report.replace(" ", "").replace("\n", "").replace('"unconnected_items":[]', '"unconnected_items": []')


@needs_kicad
def test_real_drc_passes_with_header_on_bottom(board_ir: CircuitIR, ctx: CompileContext):
    board_ir.pcb.placements = [
        Placement(component_ref="J1", x_mm=5.0, y_mm=6.0, rotation_deg=180.0, side=BoardSide.BOTTOM),
        *ROUTABLE_PLACEMENTS[1:],
    ]
    board_ir.pcb.tracks = route_naive(board_ir, LIB)
    _, path = _compile(board_ir, ctx)
    res = _drc(path, ctx.workdir / "drc_bottom.json")
    assert res.details["errors"] == [] and res.status is ValidationStatus.PASS
    assert res.details["warnings"] == []


@needs_kicad
def test_naive_router_is_not_a_router_drc_is_the_judge(tmp_path: Path):
    """J1 rotated towards the resistors: straight chains short J1's pads. DRC, not the router, says so."""
    ir = _place(divider_with_connector_ir(tmp_path, LIB), SHORTING_PLACEMENTS)
    ctx = CompileContext(workdir=tmp_path / "fixture_layout", tools={"kicad_library": LIB})
    _, path = _compile(ir, ctx)  # compiles fine: the compiler does not judge the layout
    res = _drc(path, ctx.workdir / "drc.json")
    assert res.status is ValidationStatus.FAIL
    assert {e["type"] for e in res.details["errors"]} & {"clearance", "shorting_items", "tracks_crossing"}


@needs_kicad
def test_end_to_end_erc_and_drc_with_schematic_parity(board_ir: CircuitIR, ctx: CompileContext):
    """IR -> .kicad_sch (schematic compiler) + .kicad_pcb (this compiler) -> real ERC and DRC --schematic-parity."""
    from ai_eda.compilers.schematic import SchematicCompiler

    board_ir.component("R2").description = ""  # both compilers must fall back to the library symbol's Description
    try:
        sch = SchematicCompiler().compile(board_ir, ctx)
    except NotImplementedError:  # pragma: no cover - schematic compiler not landed yet
        pytest.skip("SchematicCompiler not implemented")
    _, pcb_path = _compile(board_ir, ctx)
    assert Path(sch.path).parent == pcb_path.parent and Path(sch.path).stem == pcb_path.stem  # parity needs the same basename
    erc = kicad.run_erc(Path(sch.path), ctx.workdir / "erc.json")
    print(f"\nERC: {erc.message}")
    assert erc.status is ValidationStatus.PASS and erc.details["errors"] == []
    res = kicad.run_drc(pcb_path, ctx.workdir / "drc_parity.json", schematic_parity=True)
    print(f"DRC+parity: {res.message}")
    for w in res.details["warnings"]:
        print("  warning:", w.get("type"), "-", w.get("description"), [i.get("description") for i in w.get("items", [])])
    assert res.details["errors"] == [] and res.status is ValidationStatus.PASS
    assert res.details["warnings"] == []  # any parity finding (net_conflict, *_mismatch, missing_footprint) is a warning
    report = (ctx.workdir / "drc_parity.json").read_text(encoding="utf-8")
    assert '"schematic_parity"' in report
    r2 = _footprints(sexpr.parse_file(pcb_path))["R2"]
    assert next(str(p[2]) for p in sexpr.find_all(r2, "property") if p[1] == "Description") == "Resistor"


@needs_kicad
def test_gerber_and_drill_export(board_ir: CircuitIR, ctx: CompileContext, tmp_path: Path):
    _, path = _compile(board_ir, ctx)
    gerber_dir = tmp_path / "gerbers"
    (gerber_dir / f"{PROJECT_ID}-F_Fab.gbr").parent.mkdir(parents=True)
    (gerber_dir / f"{PROJECT_ID}-F_Fab.gbr").write_text("stale file from an earlier run\n")  # must not be returned
    listed = kicad.export_gerbers(path, gerber_dir)
    stem = PROJECT_ID
    # --no-protel-ext + the 9-layer fab set: every plot is *.gbr and the job manifest lists exactly them
    assert [p.name for p in listed] == sorted(
        f"{stem}-{layer}.gbr" for layer in ("F_Cu", "B_Cu", "F_Paste", "B_Paste", "F_Silkscreen", "B_Silkscreen", "F_Mask", "B_Mask", "Edge_Cuts")
    ) + [f"{stem}-job.gbrjob"]
    functions = {}
    for f in listed:
        if f.suffix == ".gbrjob":
            assert f.read_text(encoding="utf-8").lstrip().startswith("{")
            continue
        header = f.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]
        fn = next(line for line in header if line.startswith("%TF.FileFunction,"))
        functions[f.name] = fn[len("%TF.FileFunction,"):].rstrip("*%")
    assert functions[f"{stem}-F_Cu.gbr"] == "Copper,L1,Top"
    assert functions[f"{stem}-B_Cu.gbr"] == "Copper,L2,Bot"
    assert functions[f"{stem}-F_Mask.gbr"] == "Soldermask,Top"
    assert functions[f"{stem}-B_Mask.gbr"] == "Soldermask,Bot"
    assert functions[f"{stem}-Edge_Cuts.gbr"] == "Profile,NP"
    assert functions[f"{stem}-F_Paste.gbr"] == "Paste,Top" and functions[f"{stem}-F_Silkscreen.gbr"] == "Legend,Top"
    drill_dir = tmp_path / "drill"
    drills = kicad.export_drill(path, drill_dir)
    assert [d.name for d in drills] == [f"{stem}.drl"]
    assert drills[0].read_text(encoding="utf-8", errors="ignore").startswith("M48")
    assert check_drill_files(drills).status is ValidationStatus.PASS
    res = check_gerber_set(listed)
    assert res.status is ValidationStatus.PASS, res.message
    assert set(res.details["functions"]) >= {"Copper,L1", "Copper,L2", "Soldermask,Top", "Soldermask,Bot", "Profile,NP"}


@needs_kicad
def test_check_gerber_set_accepts_protel_named_export(board_ir: CircuitIR, ctx: CompileContext, tmp_path: Path):
    """The checker classifies by %TF.FileFunction, so kicad-cli's default Protel-named export passes too."""
    _, path = _compile(board_ir, ctx)
    out = tmp_path / "protel"
    out.mkdir()
    kicad._run(["pcb", "export", "gerbers", "-o", str(out) + "\\", str(path)])
    produced = sorted(p for p in out.iterdir() if p.is_file())
    assert {p.suffix for p in produced} >= {".gtl", ".gbl", ".gts", ".gbs", ".gm1", ".gbrjob"}
    res = check_gerber_set(produced)
    assert res.status is ValidationStatus.PASS, res.message
    # a manifest that names a file which is not in the set is a completeness failure
    subset = [p for p in produced if p.suffix != ".gtl"]
    res = check_gerber_set(subset)
    assert res.status is ValidationStatus.FAIL and "manifest lists files not in the set" in res.message and "Copper,L1" in res.message
