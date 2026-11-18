"""Arc tracks in the KiCad board compiler: a track with ``mid`` is written as KiCad's own ``(arc (start) (mid) (end) ...)``.

What is proved

* without KiCad, on the synthetic library of ``tests/test_routing.py``: the exact text of the arc node, in IR order between its
  neighbours and under the uuid key a segment has; the compiler's version unchanged; a board without an arc compiles to the bytes
  it had before arc tracks existed (a pinned digest) and holds no ``(arc``, and turning its chamfer into the arc changes that one
  node and nothing else; three points that are no arc - a track that never met the IR's check, or one that is flat at the file's
  1 nm resolution - and a radius above ``ARC_MAX_RADIUS_MM`` are CompileErrors that name the track; an allowed net's arc is cut
  out of a keep-out's rule area by its own box, not by the box of its two ends;
* with kicad-cli (skipped without it or without the KiCad libraries), on the divider of ``tests/fixtures_kicad.py`` drawn by hand
  with two rounded corners (:func:`~ai_eda.ir.arc.fillet`), one turning each way: DRC with schematic parity finds nothing, and
  without the two arcs both nets are open - the arcs carry the connection; ``pcb upgrade`` writes both arcs back point for point;
  DRC judges an arc on its own copper, not on its chord, and applies the project file's minimum track width to it; the copper
  Gerber draws each arc as one circular interpolation through its mid; and an arc above the radius limit is plotted as a
  straight line, which is why the compiler refuses it.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import ai_eda.compilers.pcb as pcb_compiler
from ai_eda.compilers import CompileContext, GerberExporter, PCBCompiler, ProjectFileCompiler, SchematicCompiler, ids
from ai_eda.compilers.pcb import ARC_MAX_RADIUS_MM, EXEMPTION_MARGIN_MM
from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Provenance, ProvenanceKind, Track, ValidationResult, authoritative
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.arc import arc_geometry, fillet, point_arc_distance
from ai_eda.ir.pcb import track_bbox, track_points
from ai_eda.tools import keepout as kg
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli, run_drc_for
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.outputs import check_output_artifact
from tests.conftest import DS
from tests.fixtures_kicad import PROJECT_ID, divider_with_connector_ir
from tests.test_rf_floorplan import _area_of, _compiled_rule_areas, keepout, with_keepouts
from tests.test_routing import board_ir, fixture_library

HAND = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand for the test")
W = 0.4
REAL = KicadLibrary()
HAS_LIBS = REAL.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and REAL.symbol_file("Device") is not None
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _track(net: str, a: tuple[float, float], b: tuple[float, float], mid: tuple[float, float] | None = None, *, layer: str = "F.Cu", width: float = W) -> Track:
    """A track from ``a`` to ``b``: straight, or with ``mid`` the arc through it."""
    return Track(net=net, layer=layer, start=a, mid=mid, end=b, width_mm=width, provenance=HAND)


def _compile(ir: CircuitIR, out: Path, lib: KicadLibrary) -> tuple[ArtifactRef, str]:
    ref = PCBCompiler().compile(ir, CompileContext(workdir=out, tools={"kicad_library": lib}))
    return ref, Path(ref.path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the node (no KiCad)


def _l_board(tmp_path: Path, lib: KicadLibrary, corner: str) -> CircuitIR:
    """Two two-pad parts and net N drawn as an L on F.Cu, from R1.2 (5.5, 3) over the corner (10.5, 3) to R2.1 (10.5, 9); the corner
    is cut 2 mm back along both legs and closed by ``corner``: ``"arc"`` - the tangent arc, a quarter circle of radius 2 mm about
    (8.5, 5) - or ``"chamfer"``, the straight track between the same two points. Keep-out FEED forbids tracks around the corner and
    allows N, so the three tracks are cut out of its rule area (the quarter arc and its chamfer have the same box)."""
    ir = board_ir(tmp_path, lib, [("R1", "SMD2", 4.0, 3.0), ("R2", "SMD2", 12.0, 9.0)], {"N": [("R1", "2"), ("R2", "1")]}, (16.0, 12.0))
    a, mid, b, radius = fillet((10.5, 3.0), (5.5, 3.0), (10.5, 9.0), 2.0)
    assert (a, mid, b) == ((8.5, 3.0), (9.914214, 3.585786), (10.5, 5.0)) and radius == pytest.approx(2.0, abs=1e-12)
    ir.pcb.tracks = [_track("N", (5.5, 3.0), a), _track("N", a, b, mid if corner == "arc" else None), _track("N", b, (10.5, 9.0))]
    return with_keepouts(ir, [keepout("FEED", [7.0, 1.0, 6.0, 6.0], forbids=["tracks"], nets=["N"])])


#: the corner of :func:`_l_board` as the board file must hold it: the arc, and the chamfer in its place
ARC_NODE = (
    "\t(arc\n"
    "\t\t(start 8.5 3)\n"
    "\t\t(mid 9.914214 3.585786)\n"
    "\t\t(end 10.5 5)\n"
    "\t\t(width 0.4)\n"
    '\t\t(layer "F.Cu")\n'
    '\t\t(net "N")\n'
    '\t\t(uuid "0819a6bd-34e2-528f-a4a9-213d06b5bc65")\n'
    "\t)\n"
)
CHAMFER_NODE = (
    "\t(segment\n"
    "\t\t(start 8.5 3)\n"
    "\t\t(end 10.5 5)\n"
    "\t\t(width 0.4)\n"
    '\t\t(layer "F.Cu")\n'
    '\t\t(net "N")\n'
    '\t\t(uuid "0819a6bd-34e2-528f-a4a9-213d06b5bc65")\n'
    "\t)\n"
)
#: sha256 of the board of ``_l_board(.., "chamfer")`` as compiler.kicad_pcb 0.2 wrote it before a track could be an arc
PLAIN_BOARD_SHA256 = "16bd67e5f569894d53d832dfc619f4f4bde42e4fe5fb48c54fd459efac9d3a6e"


def test_a_track_with_mid_is_written_as_kicads_arc_between_its_neighbours(tmp_path: Path, lib: KicadLibrary):
    """The arc node, byte for byte: ``start`` / ``mid`` / ``end`` in the six decimals of every other coordinate, then width, layer,
    net and the uuid of its place in ``ir.pcb.tracks`` - the form KiCad writes itself."""
    ir = _l_board(tmp_path, lib, "arc")
    ref, text = _compile(ir, tmp_path / "a", lib)
    assert ARC_NODE in text and text.count("(arc") == 1
    assert ref.generator == "compiler.kicad_pcb" and ref.generator_version == "0.2"  # PCBCompiler.version does not change with an arc
    tracks = [n for n in sexpr.parse(text)[1:] if sexpr.head(n) in ("segment", "arc")]
    assert [sexpr.head(n) for n in tracks] == ["segment", "arc", "segment"]  # IR order: the arc between its neighbours
    assert [sexpr.get(n, "uuid") for n in tracks] == [ids.net_item_uuid("route", "track", i) for i in range(3)]  # one key space
    assert [sexpr.head(child) for child in tracks[1][1:]] == ["start", "mid", "end", "width", "layer", "net", "uuid"]
    assert sexpr.args(sexpr.find(tracks[0], "end")) == sexpr.args(sexpr.find(tracks[1], "start")) == ["8.5", "3"]
    assert sexpr.args(sexpr.find(tracks[1], "end")) == sexpr.args(sexpr.find(tracks[2], "start")) == ["10.5", "5"]
    _, again = _compile(ir.model_copy(deep=True), tmp_path / "b", lib)
    assert again == text  # deterministic


def test_a_board_without_an_arc_compiles_to_the_bytes_it_had_before_arcs(tmp_path: Path, lib: KicadLibrary):
    """The digest was taken with the compiler as it was before a track could be an arc. And the arc changes its own node only:
    every other byte of the board (the neighbours, every uuid, the rule areas cut around the three tracks) stays."""
    ref, text = _compile(_l_board(tmp_path, lib, "chamfer"), tmp_path / "plain", lib)
    assert "(arc" not in text and text.count("(segment") == 3
    assert ref.content_hash == "sha256:" + PLAIN_BOARD_SHA256
    _, rounded = _compile(_l_board(tmp_path, lib, "arc"), tmp_path / "arc", lib)
    assert text.count(CHAMFER_NODE) == 1 and rounded == text.replace(CHAMFER_NODE, ARC_NODE)


def test_three_points_that_never_met_the_irs_check_are_a_compile_error_naming_the_track(tmp_path: Path, lib: KicadLibrary):
    """The IR refuses a degenerate arc where a Track is built; a copy with an update, a model built without validation and an
    assignment never meet that check, so the compiler checks again."""
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib})
    ir = _l_board(tmp_path, lib, "chamfer")
    with pytest.raises(ValueError, match="is no arc: its mid point lies on the line through its start and end"):
        _track("N", (8.5, 3.0), (10.5, 5.0), (9.5, 4.0))
    ir.pcb.tracks[1] = ir.pcb.tracks[1].model_copy(update={"mid": (9.5, 4.0)})  # the middle of the chamfer: on the chord
    with pytest.raises(CompileError) as e:
        PCBCompiler().compile(ir, ctx)
    assert str(e.value) == ("track #1 (N): start (8.5, 3.0) / mid (9.5, 4.0) / end (10.5, 5.0) is no arc: its mid point lies on the line through its start and end "
                            "(within 1e-06 mm): the three points define no circle")
    ir.pcb.tracks[1] = Track.model_construct(net="N", layer="F.Cu", start=(8.5, 3.0), mid=(8.5, 3.0), end=(10.5, 5.0), width_mm=W, provenance=HAND)
    with pytest.raises(CompileError, match=r"track #1 \(N\): start \(8\.5, 3\.0\) / mid \(8\.5, 3\.0\) / end \(10\.5, 5\.0\) is no arc: its mid point coincides with its start"):
        PCBCompiler().compile(ir, ctx)
    ir.pcb.tracks[1] = _track("N", (8.5, 3.0), (10.5, 5.0), (9.914214, 3.585786))
    ir.pcb.tracks[1].end = (8.5, 3.0)
    with pytest.raises(CompileError, match=r"track #1 \(N\): .* is no arc: its start and end coincide \(a full circle has no start / mid / end form\)"):
        PCBCompiler().compile(ir, ctx)
    ir.pcb.tracks[1].end, ir.pcb.tracks[1].mid = (10.5, 5.0), (math.nan, 4.0)
    with pytest.raises(CompileError, match=r"ir\.pcb\.tracks\[1\]\.mid\[0\] is nan: not a finite number"):
        PCBCompiler().compile(ir, ctx)  # compile's finite check walks mid like every other number of the board
    with pytest.raises(CompileError, match=r"track #1 \(N\): .* is no arc: its mid point \(nan, 4\.0\) is not finite"):
        PCBCompiler().build(ir, ctx)  # and the tree builder alone refuses it as well
    ir.pcb.tracks[1].mid = (9.914214, 3.585786)
    assert ARC_NODE in sexpr.dumps(PCBCompiler().build(ir, ctx))  # the same track with its three points right is written


def test_an_arc_the_file_cannot_hold_or_kicad_would_plot_straight_is_a_compile_error(tmp_path: Path, lib: KicadLibrary):
    """Two refusals of an arc the IR accepts. Off the 1 nm grid: the writer rounds each point on its own, and three points whose mid
    lies 1.05 nm off the chord come out 0.71 nm from one line - no arc in the file. And KiCad's limit: kicad-cli 10.0.6 plots a
    track arc of radius 1073.751 mm as the straight line between its ends (the canary at the end of this file), so a radius above
    ``ARC_MAX_RADIUS_MM`` is refused."""
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib})
    ir = _l_board(tmp_path, lib, "chamfer")
    ir.pcb.tracks[1] = _track("N", (8.0000004, 3.0), (10.0000004, 5.0), (8.99999951, 4.000000595))
    with pytest.raises(CompileError) as e:
        PCBCompiler().compile(ir, ctx)
    assert str(e.value) == ("track #1 (N): start (8.0, 3.0) / mid (9.0, 4.000001) / end (10.0, 5.0), the arc's points at the board file's 1 nm resolution, is no arc: "
                            "its mid point lies on the line through its start and end (within 1e-06 mm): the three points define no circle")
    assert ARC_MAX_RADIUS_MM == 1073.7
    ir.pcb.tracks[1] = _track("N", (2.0, 8.0), (14.0, 8.0), (8.0, 8.016))  # a 12 mm chord bulging 16 um: radius 1125.008 mm
    assert arc_geometry((2.0, 8.0), (8.0, 8.016), (14.0, 8.0)).r == pytest.approx(1125.008, abs=1e-6)
    with pytest.raises(CompileError) as e:
        PCBCompiler().compile(ir, ctx)
    assert str(e.value) == ("track #1 (N): an arc of radius 1125.008 mm, above the limit of 1073.7 mm - kicad-cli 10.0.6 plots a track arc of radius 1073.751 mm "
                            "or more as the straight line between its ends (measured), which would contradict the IR")
    ir.pcb.tracks[1] = _track("N", (2.0, 8.0), (14.0, 8.0), (8.0, 8.017))  # bulging 17 um: radius 1058.832 mm, written
    assert arc_geometry((2.0, 8.0), (8.0, 8.017), (14.0, 8.0)).r == pytest.approx(1058.832, abs=1e-3)
    _, text = _compile(ir, tmp_path / "out", lib)
    assert "\t(arc\n\t\t(start 2 8)\n\t\t(mid 8 8.017)\n\t\t(end 14 8)\n" in text


def test_an_allowed_nets_arc_is_cut_out_of_a_rule_area_by_its_own_box_not_by_its_chords(tmp_path: Path, lib: KicadLibrary):
    """A half circle of N from (6, 5) over (8, 3) to (10, 5) inside keep-out FEED, which forbids tracks and allows N. The box of its
    two ends is flat (y = 5); the arc reaches y = 3, and the rule area is cut back around all of it."""
    ir = board_ir(tmp_path, lib, [("R1", "SMD2", 4.0, 10.0), ("R2", "SMD2", 12.0, 10.0)], {"N": [("R1", "2"), ("R2", "1")]}, (16.0, 12.0))
    arc = _track("N", (6.0, 5.0), (10.0, 5.0), (8.0, 3.0))
    ir.pcb.tracks = [arc]
    with_keepouts(ir, [keepout("FEED", [4.0, 1.0, 8.0, 6.0], forbids=["tracks"], nets=["N"])])
    zones, _ = _compiled_rule_areas(ir, tmp_path / "out", lib)
    assert track_bbox(arc) == (6.0, 3.0, 10.0, 5.0)
    r = W / 2 + EXEMPTION_MARGIN_MM
    cut = (6.0 - r, 3.0 - r, 10.0 + r, 5.0 + r)
    assert zones and all(kg.box_area_overlap(cut, _area_of(z)) == 0.0 for z in zones)
    assert sum(abs(kg.polygon_area(_area_of(z))) for z in zones) == pytest.approx(8.0 * 6.0 - (4.0 + 2 * r) * (2.0 + 2 * r))  # FEED less the cut
    # every point of the arc's centreline keeps half the width and the margin from what is left of the rule area
    chain = track_points(arc)
    assert len(chain) > 20 and min(kg.point_area_distance(p, _area_of(z)) for p in chain for z in zones) >= r - 1e-9
    # the box of the two ends, grown the same way, ends 1.7 mm below the top of the arc: cut by it, the rule area would cover the arc
    ends_box = [(6.0 - r, 5.0 - r), (10.0 + r, 5.0 - r), (10.0 + r, 5.0 + r), (6.0 - r, 5.0 + r)]
    assert kg.point_area_distance((8.0, 3.0), ends_box) == pytest.approx(2.0 - r)


# --------------------------------------------------------------------------- kicad-cli: DRC, load and save, Gerber

#: the two rounded corners of :func:`_divider`: net, the pad its track leaves, the corner, the end of the second leg, the pad it
#: reaches. GND turns left at its corner (down, then right), VIN right (up, then right): one arc each way
ROUNDED = (("GND", (5.0, 11.08), (5.0, 14.0), (14.0, 14.0), (14.0, 11.905)), ("VIN", (5.0, 6.0), (5.0, 3.0), (14.0, 3.0), (14.0, 5.175)))


def _divider(tmp_path: Path, corners: str = "arc") -> CircuitIR:
    """The divider of ``tests/fixtures_kicad.py`` (header J1 down the left edge, R1 over R2 at x = 14) with its nets drawn by hand
    on F.Cu, 0.4 mm wide: GND from J1.3 down to y = 14, right, and up into R2.2; VIN from J1.1 up to y = 3, right, and down into
    R1.1; VOUT straight from J1.2 to the track that joins R1.2 and R2.1. The first corner of GND and of VIN is cut 1 mm back along
    both legs and closed by ``corners``: ``"arc"`` - the tangent arc (:func:`~ai_eda.ir.arc.fillet`, radius 1 mm) -, ``"chamfer"`` -
    the straight track between the two cut points - or ``"open"``, nothing. The cut points lie more than 1 mm from every pad's
    copper, so only the corner track joins them (a corner beside its pad proves nothing: where the cut track's end still overlaps
    the pad's copper, the net stays whole without the corner track - measured with the corner 0.92 mm below J1.3)."""
    ir = divider_with_connector_ir(tmp_path, REAL)
    tracks = []
    for net, pad1, corner, turn, pad2 in ROUNDED:
        a, mid, b, radius = fillet(corner, pad1, turn, 1.0)
        assert radius == pytest.approx(1.0, abs=1e-12)
        closing = {"arc": [_track(net, a, b, mid)], "chamfer": [_track(net, a, b)], "open": []}[corners]
        tracks += [_track(net, pad1, a), *closing, _track(net, b, turn), _track(net, turn, pad2)]
    ir.pcb.tracks = [*tracks, _track("VOUT", (5.0, 8.54), (14.0, 8.54)), _track("VOUT", (14.0, 6.825), (14.0, 10.255))]
    return ir


def _uuid(index: int) -> str:
    return ids.net_item_uuid(PROJECT_ID, "track", index)


def _drc(ir: CircuitIR, out: Path) -> ValidationResult:
    """Schematic, board and project file compiled into ``out``, and DRC run as the pipeline runs it: with schematic parity."""
    ctx = CompileContext(workdir=out, tools={"kicad_library": REAL})
    ir.artifacts[ArtifactKind.SCHEMATIC] = SchematicCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    ir.artifacts[ArtifactKind.KICAD_PROJECT] = ProjectFileCompiler().compile(ir, ctx)
    return run_drc_for(ir, kicad, out)


@needs_kicad
def test_kicad_passes_a_board_with_arcs_and_the_arcs_carry_their_nets(tmp_path: Path):
    """Canary (kicad-cli; measured on 10.0.6): the divider with two rounded corners passes DRC with schematic parity - no violation,
    nothing unconnected, no parity finding. That the arcs are what joins each net is shown by the same board without them: one
    missing connection per net. ``pcb upgrade`` (load and save) writes both arcs back with the same points, width, layer, net and
    uuid, and no track more or less."""
    ir = _divider(tmp_path / "arcs")
    arcs = [i for i, t in enumerate(ir.pcb.tracks) if t.is_arc]
    assert [ir.pcb.tracks[i].net for i in arcs] == ["GND", "VIN"]
    res = _drc(ir, tmp_path / "arcs")
    assert res.status is S.PASS and res.details["errors"] == [] and res.details["warnings"] == [], res.message
    assert res.details["unconnected_items"] == [] and res.details["schematic_parity"] == [] and res.details["schematic_parity_checked"] is True
    pcb = Path(ir.artifacts[ArtifactKind.PCB].path)
    saved = tmp_path / "saved" / pcb.name
    saved.parent.mkdir()
    shutil.copy(pcb, saved)
    kicad._run(["pcb", "upgrade", "--force", str(saved)], ok_codes=(0,))
    before, after = sexpr.parse_file(pcb), sexpr.parse_file(saved)
    written = {str(sexpr.get(n, "uuid")): n for n in sexpr.find_all(before, "arc")}
    assert sorted(written) == sorted(_uuid(i) for i in arcs)
    assert {str(sexpr.get(n, "uuid")): n for n in sexpr.find_all(after, "arc")} == written  # KiCad re-sorts items on save: matched by uuid
    assert len(sexpr.find_all(after, "segment")) == len(sexpr.find_all(before, "segment")) == len(ir.pcb.tracks) - 2
    # the control: the same copper without the two arcs
    cut = _divider(tmp_path / "open", "open")
    res = _drc(cut, tmp_path / "open")
    net_of = {_uuid(i): t.net for i, t in enumerate(cut.pcb.tracks)}
    missing = res.details["unconnected_items"]
    assert res.status is S.FAIL and len(missing) == 2, res.message
    assert sorted(({net_of[item["uuid"]] for item in m["items"]} for m in missing), key=sorted) == [{"GND"}, {"VIN"}]  # the two ends an arc joined
    assert {v["type"] for v in res.details["errors"]} == {"unconnected_items"} and {v["type"] for v in res.details["warnings"]} == {"track_dangling"}


@needs_kicad
def test_kicad_judges_an_arc_on_its_own_copper_not_on_its_chord(tmp_path: Path):
    """Canary (kicad-cli; measured on 10.0.6). A free L of GND with a 3 mm fillet and a 0.2 mm stub of VIN that ends 0.45 mm
    outside the middle of the arc: 0.15 mm between the two coppers, under KiCad's 0.2 mm - DRC reports one ``clearance``
    violation, between exactly these two tracks. With the chamfer in the arc's place (the chord, 0.88 mm further from the stub)
    it reports none: what KiCad judges is the arc."""
    a, mid, b, _ = fillet((20.0, 5.0), (20.0, 12.0), (27.0, 5.0), 3.0)
    tip = (round(mid[0] - 0.45 * math.sqrt(0.5), 6), round(mid[1] - 0.45 * math.sqrt(0.5), 6))
    assert point_arc_distance(tip, a, mid, b) == pytest.approx(0.45, abs=1e-6)
    found = {}
    for corner in ("arc", "chamfer"):
        out = tmp_path / corner
        ir = _divider(out)
        n = len(ir.pcb.tracks)
        ir.pcb.tracks += [_track("GND", (20.0, 12.0), a), _track("GND", a, b, mid if corner == "arc" else None), _track("GND", b, (27.0, 5.0)),
                          _track("VIN", tip, (19.0, 4.0), width=0.2)]
        ref, _ = _compile(ir, out, REAL)
        res = kicad.run_drc(Path(ref.path), out / "drc.json")
        found[corner] = [sorted(item["uuid"] for item in v["items"]) for v in res.details["errors"] + res.details["warnings"] if v["type"] == "clearance"]
    assert found == {"arc": [sorted((_uuid(n + 1), _uuid(n + 3)))], "chamfer": []}


@needs_kicad
def test_kicad_applies_the_projects_minimum_track_width_to_an_arc(tmp_path: Path):
    """Canary (kicad-cli; measured on 10.0.6). With a minimum track width of 5 mm in the project file beside the board, DRC reports
    one ``track_width`` violation per track of the divider - each of the two arcs like each segment. The width rule a fab limit
    becomes is judged on an arc as well."""
    ir = _divider(tmp_path)
    ir.pcb.manufacturing.min_track_width_mm = authoritative(5.0, DS, "mm")
    res = _drc(ir, tmp_path)
    assert res.details["design_rules"] == {"min_track_width": 5.0}, res.details.get("project_reason")
    narrow = [[item["uuid"] for item in v["items"]] for v in res.details["errors"] + res.details["warnings"] if v["type"] == "track_width"]
    assert sorted(narrow) == sorted([_uuid(i)] for i in range(len(ir.pcb.tracks))) and sum(t.is_arc for t in ir.pcb.tracks) == 2


@dataclass(frozen=True)
class _Draw:
    """One draw (``D01``) of a Gerber plot, from ``start`` to ``end`` in the file's own integers (1 nm, Y up). ``sense`` is 0 for a
    straight stroke, -1 for a clockwise arc (``G02``) and +1 for a counter-clockwise one (``G03``) about ``centre``; ``aperture``
    is the definition of the aperture it is drawn with (``C,0.400000``: a circle of 0.4 mm)."""

    start: tuple[int, int]
    end: tuple[int, int]
    sense: int
    centre: tuple[int, int] | None
    aperture: str


_SENSE = {"G01*": 0, "G02*": -1, "G03*": 1}
_COORDINATES = re.compile(r"X(-?\d+)Y(-?\d+)(?:I(-?\d+)J(-?\d+))?D0([123])\*")


def _draws(text: str) -> list[_Draw]:
    """The draws of a Gerber plot as kicad-cli 10.0.6 writes one (measured): ``%FSLAX46Y46*%`` in ``%MOMM*%`` - absolute integers of
    1 nm - with X and Y on every coordinate line; ``G01*`` / ``G02*`` / ``G03*`` set the interpolation, ``G75*`` the multi-quadrant
    mode a circular draw needs, ``D<nn>*`` the aperture; ``D02`` moves, ``D03`` flashes, ``D01`` draws - a circular draw with its
    centre as the offset ``I`` / ``J`` from its start. A coordinate line of any other form fails the test: measure again."""
    assert "%FSLAX46Y46*%" in text and "%MOMM*%" in text
    apertures = dict(re.findall(r"^%ADD(\d+)([^*]+)\*%$", text, flags=re.MULTILINE))
    out: list[_Draw] = []
    sense, multi_quadrant, aperture, at = 0, False, "", None
    for line in text.splitlines():
        if line in _SENSE:
            sense = _SENSE[line]
        elif line == "G75*":
            multi_quadrant = True
        elif re.fullmatch(r"D[1-9]\d+\*", line):
            aperture = apertures[line[1:-1]]
        elif line.endswith(("D01*", "D02*", "D03*")):
            m = _COORDINATES.fullmatch(line)
            assert m is not None, f"a coordinate line of a form that was not measured: {line!r}"
            to = (int(m[1]), int(m[2]))
            if m[5] == "1":
                assert at is not None and (sense == 0) == (m[3] is None) and (sense == 0 or multi_quadrant), line
                out.append(_Draw(at, to, sense, None if sense == 0 else (at[0] + int(m[3]), at[1] + int(m[4])), aperture))
            at = to
    return out


def _nm(p: tuple[float, float]) -> tuple[int, int]:
    """A board point (mm, Y down) in the plot's integers (1 nm, Y up)."""
    return round(p[0] * 1e6), -round(p[1] * 1e6)


def _mm(p: tuple[float, float]) -> tuple[float, float]:
    """A point of the plot in the board's frame."""
    return p[0] * 1e-6, -p[1] * 1e-6


def _middle(d: _Draw) -> tuple[float, float]:
    """The point half-way along a circular draw, in the plot's frame (Y up: counter-clockwise runs toward larger angles)."""
    (sx, sy), (ex, ey), (cx, cy) = d.start, d.end, d.centre
    a0, a1 = math.atan2(sy - cy, sx - cx), math.atan2(ey - cy, ex - cx)
    sweep = (a1 - a0) % math.tau if d.sense > 0 else -((a0 - a1) % math.tau)
    r, a = math.hypot(sx - cx, sy - cy), a0 + sweep / 2.0
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _copper_plot(ir: CircuitIR, out: Path, layer: str) -> str:
    """The Gerber plot of one copper layer of the compiled board, exported the way the pipeline exports its fab files; the set
    passes the pipeline's own ``mfg.gerber`` check."""
    ctx = CompileContext(workdir=out, tools={"kicad_library": REAL, "kicad_cli": kicad})
    ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
    art = GerberExporter().compile(ir, ctx)
    check = check_output_artifact(art, ir)
    assert check.status is S.PASS, check.message
    (plot,) = [Path(f) for f in art.files if Path(f).name.endswith(f"-{layer.replace('.', '_')}.gbr")]
    return plot.read_text(encoding="utf-8")


@needs_kicad
def test_the_copper_gerber_draws_each_arc_as_one_circular_interpolation_through_its_mid(tmp_path: Path):
    """Canary (kicad-cli; measured on 10.0.6). What KiCad plots for a track arc is a circular interpolation, not a chain of short
    strokes and not a region: a move to the arc's start, ``G75*``, ``G03*`` (counter-clockwise as the board is seen; ``G02*`` for
    the arc that turns the other way), one ``D01`` to the arc's end with the centre as ``I`` / ``J``, and ``G01*`` again - drawn
    with the round aperture of the track's width. So the exported copper holds each arc's own end points, and the drawn arc
    passes through its ``mid`` (the bound here is 0.01 mm; measured, the two agree to the nanometre). Every straight track is
    one stroke, and there is no other draw."""
    ir = _divider(tmp_path)
    text = _copper_plot(ir, tmp_path, "F.Cu")
    assert "X5000000Y-13000000D02*\nG75*\nG03*\nX6000000Y-14000000I1000000J0D01*\nG01*\n" in text  # GND: (5, 13) to (6, 14) about (6, 13)
    assert "X5000000Y-4000000D02*\nG75*\nG02*\nX6000000Y-3000000I1000000J0D01*\nG01*\n" in text  # VIN: (5, 4) to (6, 3) about (6, 4)
    assert "G36*" not in text  # no region: an arc cannot stand in for the filled pour mfg.gerber looks for on a zone's layer
    draws = _draws(text)
    arcs = [t for t in ir.pcb.tracks if t.is_arc]
    straight = [t for t in ir.pcb.tracks if not t.is_arc]
    assert len(arcs) == 2 and len(draws) == len(ir.pcb.tracks)
    assert sorted(sorted((d.start, d.end)) for d in draws if d.sense == 0) == sorted(sorted((_nm(t.start), _nm(t.end))) for t in straight)
    for t in arcs:
        (d,) = [d for d in draws if d.sense != 0 and (d.start, d.end) == (_nm(t.start), _nm(t.end))]  # its own end points, start to end
        g = arc_geometry(t.start, t.mid, t.end)
        assert d.sense == (-1 if g.sweep > 0 else 1)  # the IR's positive sweep is clockwise as the board is seen: G02
        assert d.aperture == f"C,{t.width_mm:.6f}"
        assert math.dist(d.start, d.centre) == pytest.approx(g.r * 1e6, abs=2.0) and math.dist(d.end, d.centre) == pytest.approx(g.r * 1e6, abs=2.0)
        middle = _mm(_middle(d))
        assert point_arc_distance(t.mid, _mm(d.start), middle, _mm(d.end)) <= 0.01  # the drawn arc passes through the IR's mid
        assert point_arc_distance(t.mid, _mm(d.start), middle, _mm(d.end)) < 2e-6 and point_arc_distance(middle, t.start, t.mid, t.end) < 2e-6
    assert {d.sense for d in draws} == {0, -1, 1}  # one arc each way


@needs_kicad
def test_kicad_plots_an_arc_above_the_radius_limit_as_a_straight_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Canary (kicad-cli; measured on 10.0.6) - the reason for ``ARC_MAX_RADIUS_MM``. Three arcs over a 26 mm chord on B.Cu, written
    with the limit lifted: radius 1073.696 mm (under the limit) and 1073.737 mm are plotted as circular interpolations, radius
    1073.751 mm as one straight stroke between its ends - its 0.079 mm of bulge is not in the copper. Between the last two lies
    2^30 nm = 1073.741824 mm, where the arc's diameter no longer fits KiCad's signed 32-bit nanometre coordinate. With the limit
    in place the compiler writes the first arc and refuses the other two.

    The other end of the scale: a half circle 12 nm across is plotted as its chord as well. It lies 6 nm from that chord - inside
    the 1 um every reader of an arc allows (``ARC_SAGITTA_MM``) - and the compiler writes it."""
    bulges = {15.0: 0.078703, 16.0: 0.0787, 17.0: 0.078699}  # the mid's distance from the chord: the smaller, the wider the arc
    arcs = [_track("GND", (2.0, y), (28.0, y), (15.0, round(y + s, 6)), layer="B.Cu") for y, s in bulges.items()]
    radii = [arc_geometry(t.start, t.mid, t.end).r for t in arcs]
    assert radii == pytest.approx([1073.696, 1073.737, 1073.751], abs=1e-3)
    assert radii[0] < ARC_MAX_RADIUS_MM < radii[1] < 2**30 * 1e-6 < radii[2]
    tiny = _track("GND", (10.0, 18.5), (10.000012, 18.5), (10.000006, 18.500006), layer="B.Cu")
    assert arc_geometry(tiny.start, tiny.mid, tiny.end).r == pytest.approx(6e-6, abs=1e-9)
    ir = _divider(tmp_path)
    ir.pcb.tracks += [tiny, arcs[0]]
    assert _compile(ir, tmp_path / "limit", REAL)[1].count("(arc") == 4  # the two corners, the tiny arc and the widest arc under the limit
    for wider in arcs[1:]:
        ir.pcb.tracks[-1] = wider
        with pytest.raises(CompileError, match=r"track #11 \(GND\): an arc of radius 1073\.7\d+ mm, above the limit of 1073\.7 mm - kicad-cli 10\.0\.6 plots a track arc"):
            _compile(ir, tmp_path / "limit", REAL)
    monkeypatch.setattr(pcb_compiler, "ARC_MAX_RADIUS_MM", math.inf)
    ir.pcb.tracks[-1:] = arcs
    by_ends = {(d.start, d.end): d for d in _draws(_copper_plot(ir, tmp_path / "lifted", "B.Cu"))}
    drawn = [by_ends[_nm(t.start), _nm(t.end)] for t in arcs]
    assert [d.sense for d in drawn] == [1, 1, 0]  # two arcs (counter-clockwise as seen: they bulge down), then the straight stroke
    for t, d in zip(arcs[:2], drawn):
        assert point_arc_distance(t.mid, _mm(d.start), _mm(_middle(d)), _mm(d.end)) < 2e-6
    assert by_ends[_nm(tiny.start), _nm(tiny.end)].sense == 0 and len(by_ends) == 4  # the tiny arc: one stroke too; nothing else is drawn
