"""Custom pads (KiCad ``custom`` shape: an anchor plus copper primitives) in the router and in everything that reads pad copper.

The one extent is :func:`ai_eda.tools.kicad.geometry.custom_pad_parts`: the box of the pad's anchor and one box around each copper
primitive (grown by half its stroke), read from the library file (:class:`~ai_eda.tools.kicad.library.PadPrimitive`). What is checked:

* the library reader keeps the anchor and every primitive kind KiCad 10 writes (``gr_poly`` with its ``arc`` pieces, ``gr_line``,
  ``gr_rect``, ``gr_circle``, ``gr_arc``, ``gr_curve``), skips the editor annotations (``gr_bbox`` / ``gr_vector``) and lists what it
  does not read; the parts bound every primitive on both sides and at a quarter turn, with ``exact`` / ``inscribed_r`` only where the
  box is known to be copper; a pad with an unread primitive is refused;
* ``routing.maze`` 0.5: a SOT-89-style pad (the real ``Package_TO_SOT_SMD:SOT-89-3`` pad 2: a rect anchor and a tab polygon) routes,
  every other net keeps the router's clearance from the tab on the exact geometry, the copper is stamped 0.5 with the pads named, and the
  ``pcb.routing.*`` checks PASS; a ring pad (the ``Sensor_Audio:CUI_CMC-4013-SMT`` microphone's pad 1) around a centre pad leaves the
  centre pad's net unrouted with the fenced-terminal reason while the ring's net routes on its anchor; a tiny anchor lands the track on
  an exact primitive; a plane net's pad via clears the pad's whole copper; a board without a custom pad stays 0.2;
* every user sees the same boxes: ``pads_bbox``, the silkscreen's pad copper, ``pcb.routing`` / ``pcb.keepout``, the SI path pads, the
  3D scene and the router's own obstacles; connectivity joins copper to a custom pad only through a part known to be copper and says
  NOT_VERIFIED (``bounded_pads``) - never FAIL - where copper meets it only inside a bounding box and disc (copper in a ring box's
  corner, clear of the ring's disc, is a measured open); clearance likewise (a foreign track across a ring box's corner but clear of
  the ring's bounding disc by the limit is measured clear, within the limit of that disc NOT_VERIFIED, across the anchor's disc a
  FAIL); ``pcb.keepout`` FAILs on the anchor, and a keep-out meeting only a ring's box is NOT_VERIFIED; the silkscreen measures
  circles as discs (the ring's a bound: a silk line in its hole and a designed text within its margin of the disc are NOT_VERIFIED, the
  footprint's silk circle outside it PASSes, a library silk circle just outside it too - listed below the text margin);
* with the packed KiCad 10.0.6 libraries (``KICAD10_SYMBOL_DIR``; skipped without them): the real SOT-89-3 and CUI_CMC-4013-SMT parts,
  every custom pad of the installed footprint libraries read with no unread primitive, and a real SOT-89-3 board routed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_eda.errors import CompileError
from ai_eda.ir import BoardSide, LibraryRef, NetKind, Placement, SilkText
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.kicad.geometry import PadPart, custom_pad_parts, pads_bbox
from ai_eda.tools.kicad.library import KicadLibrary, PadPrimitive
from ai_eda.tools.model3d.scene import build_scene
from ai_eda.tools.routing import maze
from ai_eda.tools.routing.maze import ROUTER_CUSTOM_PAD_VERSION, ROUTER_VERSION, RoutingParams, route_board
from ai_eda.tools.si.paths import net_pads
from ai_eda.tools.silkscreen import place_silkscreen, text_extent
from ai_eda.tools.silkscreen.geometry import pad_copper
from ai_eda.validation.layout import CLEARANCE_CHECK, CONNECTIVITY_CHECK, KEEPOUT_CHECK, SILK_CLEARANCE_CHECK, _Board, _point_box_distance, _seg_box_distance
from tests.test_layout_validation import _checks, _limit, _track
from tests.test_rf_floorplan import _run, keepout, with_keepouts
from tests.test_routing import board_ir, fixture_library
from tests.test_si_checks import stacked

#: the real ``Package_TO_SOT_SMD:SOT-89-3`` pads (KiCad 10.0.6): pad 2 is a rect anchor with the tab polygon east of it
SOT89 = (
    '(pad "1" smd rect (at -1.95 -1.5) (size 1.3 0.9) (layers "F.Cu" "F.Mask" "F.Paste"))\n'
    '  (pad "2" smd custom (at -1.8625 0) (size 1.475 0.9) (layers "F.Cu" "F.Mask" "F.Paste") (options (clearance outline) (anchor rect))\n'
    '    (primitives (gr_poly (pts (xy 3.8625 0.8665) (xy 0.7375 0.8665) (xy 0.7375 -0.8665) (xy 3.8625 -0.8665)) (width 0) (fill yes))))\n'
    '  (pad "3" smd roundrect (at -1.95 1.5) (size 1.3 0.9) (layers "F.Cu" "F.Mask" "F.Paste") (roundrect_rratio 0.25))'
)
#: the real ``Sensor_Audio:CUI_CMC-4013-SMT`` pads: pad 1 a ring (a 0.75 mm circle anchor on it, the circle primitive of width 0.75
#: around the footprint origin) enclosing pad 2
RING = (
    '(pad "1" smd custom (at 0 -1.375) (size 0.75 0.75) (layers "F.Cu" "F.Mask" "F.Paste") (options (clearance outline) (anchor circle))\n'
    '    (primitives (gr_circle (center 0 1.375) (end 1.375 1.375) (width 0.75) (fill no))))\n'
    '  (pad "2" smd circle (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask" "F.Paste"))'
)
#: the ring pads with a silk circle outside the ring (r 2.1, like the real footprint's) and a silk line inside its hole (0.2 mm from the ring's
#: copper, 0.3 mm from pad 2): the line meets the ring's disc, which only bounds its copper
RING_SILK = RING + (
    '\n  (fp_circle (center 0 0) (end 2.1 0) (stroke (width 0.12) (type solid)) (fill no) (layer "F.SilkS"))'
    '\n  (fp_line (start -0.2 -0.8) (end 0.2 -0.8) (stroke (width 0.05) (type solid)) (layer "F.SilkS"))'
)
#: the ring pads with a library silk circle just outside the ring's disc (r 1.85, stroke 0.05: its chords' inner edge about 0.065 mm
#: from the disc of r 1.75 that bounds the ring's copper)
RING_NEAR = RING + '\n  (fp_circle (center 0 0) (end 1.85 0) (stroke (width 0.05) (type solid)) (fill no) (layer "F.SilkS"))'
#: a 0.1 mm anchor whose copper is the filled 1 x 1 mm rectangle beside it (its centre 1 mm east of the anchor)
TINY_ANCHOR = (
    '(pad "1" smd custom (at 0 0) (size 0.1 0.1) (layers "F.Cu" "F.Mask" "F.Paste") (options (anchor rect))\n'
    '    (primitives (gr_rect (start 0.5 -0.5) (end 1.5 0.5) (width 0) (fill yes))))'
)
#: every primitive kind, with the two editor annotations (not copper) and a head the reader does not know in a second pad
EVERY = (
    '(pad "1" smd custom (at 1 2 90) (size 0.4 0.2) (layers "F.Cu" "F.Mask") (options (clearance outline) (anchor rect))\n'
    '    (primitives\n'
    '      (gr_poly (pts (xy 0 -1) (xy 2 -1) (arc (start 2 -1) (mid 3 0) (end 2 1)) (xy 0 1)) (width 0.2) (fill yes))\n'
    '      (gr_line (start -1 0) (end -3 0) (width 0.4))\n'
    '      (gr_rect (start -1 1) (end 1 2) (width 0) (fill yes))\n'
    '      (gr_circle (center 0 -3) (end 0.5 -3) (width 0) (fill yes))\n'
    '      (gr_arc (start -1 -1) (mid -1.5 -1.5) (end -2 -1) (width 0.1))\n'
    '      (gr_curve (pts (xy 3 3) (xy 4 3) (xy 4 4) (xy 3 5)) (width 0.2))\n'
    '      (gr_bbox (start -10 -10) (end 10 10))\n'
    '      (gr_vector (start 0 0) (end 9 9))))\n'
    '  (pad "2" smd custom (at 5 5) (size 0.5 0.5) (layers "F.Cu") (primitives (gr_blob (start 0 0) (end 1 1))))'
)


def _library(root: Path) -> KicadLibrary:
    """The router tests' fixture library plus the custom-pad footprints above (``Test:SOT89`` / ``RING`` / ``RINGSILK`` / ``RINGNEAR`` /
    ``TINYANCHOR`` / ``EVERY``)."""
    lib = fixture_library(root)
    pretty = root / "footprints" / "Test.pretty"
    for name, pads in (("SOT89", SOT89), ("RING", RING), ("RINGSILK", RING_SILK), ("RINGNEAR", RING_NEAR), ("TINYANCHOR", TINY_ANCHOR), ("EVERY", EVERY)):
        (pretty / f"{name}.kicad_mod").write_text(
            f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n'
            f'  (fp_rect (start -3 -3) (end 3 3) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
            f"  {pads})\n",
            encoding="utf-8",
        )
    return KicadLibrary(roots=[root])


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return _library(tmp_path / "kicad")


def _pad(lib: KicadLibrary, name: str, number: str):
    return next(p for p in lib.load_footprint(LibraryRef(library="Test", name=name)).pads if p.number == number)


def _box(part: PadPart) -> tuple[float, float, float, float]:
    return (part.box.x1, part.box.y1, part.box.x2, part.box.y2)


def _close(got: list, want: list, tol: float = 1e-9) -> bool:
    """Two lists of equal-length number tuples agree within ``tol``."""
    return len(got) == len(want) and all(len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b)) for a, b in zip(got, want))


def _at(x: float = 0.0, y: float = 0.0, rot: float = 0.0, side: BoardSide = BoardSide.TOP) -> Placement:
    return Placement(component_ref="U1", x_mm=x, y_mm=y, rotation_deg=rot, side=side)


# --------------------------------------------------------------------------- the library reader and the parts


def test_the_library_keeps_the_anchor_and_every_primitive_kind_and_lists_what_it_cannot_read(lib: KicadLibrary):
    pad = _pad(lib, "EVERY", "1")
    assert pad.shape == "custom" and pad.anchor == "rect" and pad.unread_primitives == ()
    assert [p.kind for p in pad.primitives] == ["poly", "line", "rect", "circle", "arc", "curve"]  # gr_bbox / gr_vector are no copper
    assert pad.primitives[0] == PadPrimitive("poly", ((0.0, -1.0), (2.0, -1.0), (0.0, 1.0)), 0.2, True, (((2.0, -1.0), (3.0, 0.0), (2.0, 1.0)),))
    assert pad.primitives[1] == PadPrimitive("line", ((-1.0, 0.0), (-3.0, 0.0)), 0.4, False)
    assert pad.primitives[3] == PadPrimitive("circle", ((0.0, -3.0), (0.5, -3.0)), 0.0, True)
    assert pad.primitives[5].points == ((3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 5.0))
    unread = _pad(lib, "EVERY", "2")
    assert unread.anchor == "rect" and unread.primitives == () and unread.unread_primitives == ("gr_blob",)
    ring = _pad(lib, "RING", "1")
    assert ring.anchor == "circle" and ring.primitives == (PadPrimitive("circle", ((0.0, 1.375), (1.375, 1.375)), 0.75, False),)
    # every other shape carries nothing of this
    plain = _pad(lib, "RING", "2")
    assert plain.anchor is None and plain.primitives == () and plain.unread_primitives == ()
    with pytest.raises(CompileError, match=r"custom pad '2' has primitive\(s\) gr_blob the library reader does not read: its copper is unknown"):
        custom_pad_parts(_at(), unread)
    with pytest.raises(CompileError, match="has shape 'circle', not 'custom'"):
        custom_pad_parts(_at(), plain)


def test_the_parts_bound_every_primitive_on_both_sides_and_at_a_quarter_turn(lib: KicadLibrary):
    """EVERY's pad 1 sits at (1, 2) turned 90 degrees in the footprint: a pad-frame point (u, v) lands at (1 + v, 2 - u) on the top side
    of a footprint at the origin; the bottom side mirrors the footprint's y."""
    pad = _pad(lib, "EVERY", "1")
    top = custom_pad_parts(_at(), pad)
    assert [p.what for p in top] == ["anchor rect", "gr_poly[0]", "gr_line[1]", "gr_rect[2]", "gr_circle[3]", "gr_arc[4]", "gr_curve[5]"]
    want = [
        (0.9, 1.8, 1.1, 2.2),  # the 0.4 x 0.2 anchor turned: 0.2 wide in x, 0.4 in y
        (-0.1, -1.1, 2.1, 2.1),  # the polygon (its arc bulges to u = 3 -> y = -1) grown by 0.1
        (0.8, 2.8, 1.2, 5.2),  # the line u -1..-3 -> y 3..5, grown by 0.2
        (2.0, 1.0, 3.0, 3.0),  # the filled rectangle: exact
        (-2.5, 1.5, -1.5, 2.5),  # the filled disc of radius 0.5 around v = -3 -> x = -2
        (-0.55, 2.95, 0.05, 4.05),  # the half circle around (-1.5, -1) -> (0, 3.5): x -0.5..0, y 3..4, grown by 0.05
        (3.9, -2.1, 6.1, -0.9),  # the curve's four control points grown by 0.1
    ]
    for part, box in zip(top, want):
        assert _box(part) == pytest.approx(box, abs=1e-6), part
    assert [p.exact for p in top] == [True, False, False, True, False, False, False]
    assert [p.inscribed_r for p in top] == pytest.approx([0.1, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    assert [p.anchor for p in top] == [True] + [False] * 6
    # the bottom side: every box mirrored in y (the footprint's y), the same kinds and flags
    bottom = custom_pad_parts(_at(side=BoardSide.BOTTOM), pad)
    for t, b in zip(top, bottom):
        assert _box(b) == pytest.approx((t.box.x1, -t.box.y2, t.box.x2, -t.box.y1), abs=1e-6)
        assert (b.exact, b.inscribed_r, b.what) == (t.exact, t.inscribed_r, t.what)
    # a footprint turned 90 degrees: every box turned with it ((x, y) -> (y, -x))
    turned = custom_pad_parts(_at(rot=90.0), pad)
    for t, r in zip(top, turned):
        assert _box(r) == pytest.approx((t.box.y1, -t.box.x2, t.box.y2, -t.box.x1), abs=1e-6)
    # pads_bbox reads the parts: the whole copper, not the 0.4 x 0.2 anchor (pad 2, unread, counts with its size box as before)
    fp = lib.load_footprint(LibraryRef(library="Test", name="EVERY"))
    b = pads_bbox(_at(), fp)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((-2.5, -2.1, 6.1, 5.25), abs=1e-6)


# --------------------------------------------------------------------------- routing.maze 0.5


def _sot89_board(tmp_path: Path, lib: KicadLibrary):
    """U1 (SOT89) at (8, 6): pad 1 at (6.05, 4.5), the anchor at (6.1375, 6), the tab x 6.875..10 / y 5.1335..6.8665, pad 3 at (6.05, 7.5).
    D's two SMD pads sit above and below the tab on F.Cu: its straight line crosses the tab."""
    return board_ir(
        tmp_path, lib,
        [("U1", "SOT89", 8.0, 6.0), ("R1", "PAD1", 2.0, 2.0), ("R2", "PAD1", 2.0, 6.0), ("R3", "PAD1", 2.0, 10.0), ("R4", "SMD1", 9.0, 1.5),
         ("R5", "SMD1", 9.0, 10.5)],
        {"A": [("U1", "1"), ("R1", "1")], "B": [("U1", "2"), ("R2", "1")], "C": [("U1", "3"), ("R3", "1")], "D": [("R4", "1"), ("R5", "1")]},
        (16.0, 12.0),
    )


TAB = (6.875, 5.1335, 10.0, 6.8665)
ANCHOR = (5.4, 5.55, 6.875, 6.45)


def test_a_sot89_pad_routes_and_every_other_net_keeps_clear_of_its_tab(tmp_path: Path, lib: KicadLibrary):
    ir = _sot89_board(tmp_path, lib)
    r = route_board(ir, lib)
    assert r.unrouted == {} and r.stats["routed_nets"] == 4 and r.version == ROUTER_CUSTOM_PAD_VERSION == "0.5"
    entry = "custom_pads:U1.2;model=anchor+primitive_boxes"
    for item in [*r.tracks, *r.vias]:
        assert item.provenance.tool_version == "0.5" and entry in item.provenance.derived_from, item
        assert "routing.maze 0.5: custom pad(s) U1.2 are obstacles as the box of their anchor" in item.provenance.note
    assert r.stats["custom_pads"] == [{"pad": "U1.2", "parts": [
        {"part": "anchor rect", "box": list(ANCHOR), "exact": True, "inscribed_r": 0.45},
        {"part": "gr_poly[0]", "box": list(TAB), "exact": True, "inscribed_r": 0.8665},
    ]}]
    # B's copper ends on the anchor's centre (the stub), never on the tab
    assert (6.1375, 6.0) in {pt for t in r.tracks if t.net == "B" for pt in (t.start, t.end)}
    # every other net's copper keeps the router's clearance from the tab and the anchor, on the exact geometry
    p = r.params
    for t in r.tracks:
        if t.net != "B" and t.layer == "F.Cu":
            for box in (TAB, ANCHOR):
                assert _seg_box_distance(t.start, t.end, box) - t.width_mm / 2.0 >= p.clearance_mm - 1e-9, (t, box)
    for v in r.vias:
        if v.net != "B":
            for box in (TAB, ANCHOR):
                assert _point_box_distance((v.x_mm, v.y_mm), box) - v.diameter_mm / 2.0 >= p.clearance_mm - 1e-9, (v, box)
    # the IR geometry checks: every net connected, every pair clear of the 0.25 mm limit, the custom pad named
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    _limit(ir)
    checks = _checks(ir, tmp_path, lib)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS, checks[CONNECTIVITY_CHECK].message
    assert checks[CLEARANCE_CHECK].status is S.PASS, checks[CLEARANCE_CHECK].message
    assert checks[CONNECTIVITY_CHECK].details["custom_pads"] == ["U1.2"] and checks[CLEARANCE_CHECK].details["custom_pads"] == ["U1.2"]
    # a foreign track across the tab (but clear of the anchor) is a short there: the tab is copper to the check too
    ir.pcb.tracks.append(_track("D", (9.0, 4.0), (9.0, 8.0)))
    clearance = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and any(v["b"] == "U1.2" and v["distance_mm"] == 0.0 for v in clearance.details["violations"])


def test_a_ring_pad_leaves_its_enclosed_centre_pad_unrouted_and_routes_on_its_anchor(tmp_path: Path, lib: KicadLibrary):
    """MK1 (RING) at (8, 6): the anchor (a 0.75 mm circle) at (8, 4.625) on the ring, the ring's box 6.25..9.75 around pad 2 at (8, 6).
    Pad 2's terminal is inside the ring's keep-out (on the real copper too: a closed ring leaves no way out on F.Cu)."""
    ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0), ("R2", "PAD1", 13.0, 9.0)],
                  {"M": [("MK1", "1"), ("R1", "1")], "P": [("MK1", "2"), ("R2", "1")]}, (16.0, 12.0))
    r = route_board(ir, lib)
    assert list(r.unrouted) == ["P"] and r.unrouted["P"].startswith("MK1.2 terminal cell (8, 6) is inside a keep-out on every copper layer")
    assert r.stats["routed_nets"] == 1 and {t.net for t in r.tracks} == {"M"} and r.version == "0.5"
    assert (8.0, 4.625) in {pt for t in r.tracks for pt in (t.start, t.end)}  # M lands on the anchor
    assert r.stats["custom_pads"][0]["parts"][1] == {"part": "gr_circle[0]", "box": [6.25, 4.25, 9.75, 7.75], "exact": False, "inscribed_r": 0.0}
    # P has no copper at all: its open is measured (FAIL); M's pads are one copper set
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    conn = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    rows = {row["net"]: row for row in conn.details["nets"]}
    assert conn.status is S.FAIL and rows["M"]["status"] == "PASS" and rows["P"]["status"] == "FAIL" and rows["P"]["unconnected"] == ["R2.1"]


def test_copper_meeting_a_custom_pad_only_inside_a_bounding_box_is_not_verified_never_fail(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    # a track from R1.1 that ends on the ring's lower-left, inside its box, away from the anchor: the ring's copper is not known there
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 7.5)), _track("M", (3.0, 7.5), (7.0, 7.5))]
    conn = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    row = conn.details["nets"][0]
    assert conn.status is S.NOT_VERIFIED and row["status"] == "NOT_VERIFIED" and row["bounded_pads"] == ["MK1.1"], conn.message
    assert "meets a custom pad only inside the box that bounds its copper" in conn.message
    # a track that never reaches the ring's box is a measured open
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 7.5))]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.FAIL
    # so is one that ends in the ring box's corner (6.5, 7.6) but clear of the disc that bounds the ring's copper (2.19 mm from its
    # centre less the 0.2 mm half width, against r 1.75): it cannot meet the ring's copper
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 7.6)), _track("M", (3.0, 7.6), (6.5, 7.6))]
    conn = _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    assert conn.status is S.FAIL and "bounded_pads" not in conn.details["nets"][0], conn.message
    # a track onto the anchor joins the pad
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 4.625)), _track("M", (3.0, 4.625), (8.0, 4.625))]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS
    # an exact primitive joins too: the SOT89 tab
    ir = _sot89_board(tmp_path, lib)
    ir.nets = [n for n in ir.nets if n.name == "B"]
    ir.pcb.tracks = [_track("B", (2.0, 6.0), (2.0, 11.5)), _track("B", (2.0, 11.5), (9.0, 11.5)), _track("B", (9.0, 11.5), (9.0, 6.0))]
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS


def test_clearance_to_a_bounding_box_only_is_not_verified_and_to_known_copper_fails(tmp_path: Path, lib: KicadLibrary):
    """A foreign track that starts inside the ring box's corner, 2.26 mm from the ring's centre, is 0.313 mm (less its 0.2 mm half width)
    from the disc of r 1.75 that bounds the ring's copper - over the 0.25 mm limit: measured clear, however close to the box. Moved to
    2.05 mm from the centre it is 0.10 mm from that disc - closer than the limit only to a shape that bounds the ring: NOT_VERIFIED
    naming it, never FAIL. Across the anchor's disc it is a measured short."""
    ir = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0), ("R3", "PAD1", 12.0, 10.0)],
                  {"M": [("MK1", "1"), ("R1", "1")], "Q": [("R3", "1")]}, (16.0, 12.0))
    _limit(ir)
    ir.pcb.tracks = [_track("M", (3.0, 3.0), (3.0, 4.625)), _track("M", (3.0, 4.625), (8.0, 4.625)), _track("Q", (9.6, 7.6), (12.0, 10.0))]
    clearance = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert clearance.status is S.PASS and "custom_pad_bounds" not in clearance.details, clearance.message
    ir.pcb.tracks[2] = _track("Q", (9.45, 7.45), (12.0, 10.0))
    clearance = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert clearance.status is S.NOT_VERIFIED and clearance.details["violations"] == [], clearance.message
    assert [(r["a"], r["b"], r["status"]) for r in clearance.details["custom_pad_bounds"]] == [("track[2:Q]", "MK1.1", "NOT_VERIFIED")]
    assert clearance.details["custom_pad_bounds"][0]["distance_mm"] == pytest.approx(math.hypot(1.45, 1.45) - 0.2 - 1.75, abs=1e-6)
    assert "only to the box / disc that bounds a custom pad's copper" in clearance.message and "from the disc that bounds" in clearance.message
    ir.pcb.tracks = [ir.pcb.tracks[2], _track("Q", (8.0, 3.0), (8.0, 4.4))]  # into the anchor's disc: its copper for sure
    clearance = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and [(r["a"], r["b"]) for r in clearance.details["violations"]] == [("track[1:Q]", "MK1.1")]
    # inside the anchor box's corner but beside its disc (the anchor's copper exactly): measured to the disc, not a short on the box
    ir.pcb.tracks = [ir.pcb.tracks[0], _track("Q", (8.36, 4.26), (12.0, 1.0), w=0.1)]
    clearance = _checks(ir, tmp_path, lib)[CLEARANCE_CHECK]
    row = next(r for r in clearance.details["violations"] if r["a"] == "track[1:Q]")
    assert clearance.status is S.FAIL and row["b"] == "MK1.1" and "overlaps" not in row["message"], clearance.message
    assert row["distance_mm"] == pytest.approx(math.hypot(0.36, 0.365) - 0.05 - 0.375, abs=1e-6)


def test_a_tiny_anchor_lands_the_track_on_an_exact_primitive(tmp_path: Path, lib: KicadLibrary):
    """U1's 0.1 mm anchor at (3.1, 4.1) is 0.14 mm from the nearest 0.25 mm grid point, outside its 0.05 mm inscribed circle: the track
    lands on the filled rectangle's centre (4.1, 4.1) instead, whose inscribed circle (0.5 mm) holds a grid point."""
    ir = board_ir(tmp_path, lib, [("U1", "TINYANCHOR", 3.1, 4.1), ("R1", "PAD1", 10.0, 4.0)], {"N": [("U1", "1"), ("R1", "1")]}, (14.0, 8.0))
    r = route_board(ir, lib)
    assert r.unrouted == {} and (4.1, 4.1) in {pt for t in r.tracks for pt in (t.start, t.end)}
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    assert _checks(ir, tmp_path, lib)[CONNECTIVITY_CHECK].status is S.PASS


def test_a_plane_nets_custom_pad_gets_its_via_beyond_its_whole_copper(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("U1", "SOT89", 8.0, 6.0), ("C1", "SMD1", 3.0, 3.0), ("R4", "PAD1", 12.0, 10.0)],
                  {"GND": [("U1", "2"), ("C1", "1"), ("R4", "1")]}, (16.0, 12.0))
    ir.nets[0].kind = NetKind.GROUND
    stacked(ir, 4)
    plane = [(0.5, 0.5), (15.5, 0.5), (15.5, 11.5), (0.5, 11.5)]
    r = route_board(ir, lib, inner_layers=True, plane_nets={"GND": [("In1.Cu", plane), ("In2.Cu", plane)]})
    assert r.unrouted == {} and r.version == "0.5"
    row = r.stats["plane_nets"]["GND"]
    assert row["pads"] == ["C1.1", "U1.2"] and row["through_hole"] == ["R4.1"]
    # the pad's via: its disc overlaps neither the anchor nor the tab (no via in pad), and the stub starts on the anchor's centre
    for v in r.vias:
        for box in (TAB, ANCHOR):
            assert _point_box_distance((v.x_mm, v.y_mm), box) >= v.diameter_mm / 2.0 - 1e-9
    assert (6.1375, 6.0) in {pt for t in r.tracks for pt in (t.start, t.end)}
    # the walk's reach counts from the pad's whole copper in each direction (east over the tab: 10.0 - 6.1375 mm)
    board = maze._Board(ir, lib, r.params, inner_layers=True)
    term = next(t for t in board.terminals["GND"] if t.label == "U1.2")
    assert term.pad.reach == pytest.approx((10.0 - 6.1375, 6.8665 - 6.0, 6.1375 - 5.4, 6.0 - 5.1335))


def test_a_board_without_a_custom_pad_is_still_router_0_2(tmp_path: Path, lib: KicadLibrary):
    ir = board_ir(tmp_path, lib, [("R1", "PAD1", 2.0, 2.0), ("R2", "PAD1", 10.0, 6.0)], {"A": [("R1", "1"), ("R2", "1")]}, (12.0, 8.0))
    r = route_board(ir, lib)
    assert r.version == ROUTER_VERSION == "0.2" and "custom_pads" not in r.stats and "keepouts" not in r.stats
    assert all(t.provenance.tool_version == "0.2" and not any(e.startswith("custom_pads:") for e in t.provenance.derived_from) for t in r.tracks)


# --------------------------------------------------------------------------- one extent for every user


def test_every_user_of_pad_copper_sees_the_same_boxes(tmp_path: Path, lib: KicadLibrary):
    ir = _sot89_board(tmp_path, lib)
    fp = lib.load_footprint(LibraryRef(library="Test", name="SOT89"))
    placement = ir.pcb.placement("U1")
    pad = fp.pad("2")
    boxes = [_box(p) for p in custom_pad_parts(placement, pad)]
    assert boxes == [ANCHOR, TAB]
    # the footprint's pad extent (placement, silkscreen body) reaches the tab's east edge
    assert pads_bbox(placement, fp).x2 == 10.0
    # the silkscreen keeps its texts off the same boxes
    assert [p.shape.bbox() for p in pad_copper("U1", fp, placement) if p.label == "U1.2"] == boxes
    # pcb.routing / pcb.keepout read one item per box
    items = [p for p in _Board(ir, lib).pads if p.label == "U1.2"]
    assert _close([(p.cx - p.hw, p.cy - p.hh, p.cx + p.hw, p.cy + p.hh) for p in items], boxes)
    assert [(p.exact, p.joins, p.part) for p in items] == [(True, True, "anchor rect"), (True, True, "gr_poly[0]")]
    # the router's own obstacles
    rboard = maze._Board(ir, lib, RoutingParams())
    assert _close([g.box for g in rboard.pads if g.ref == "U1" and g.number == "2"], boxes)
    # the SI path extraction contracts copper inside the parts known to be copper (here both)
    assert _close([(b.cx - b.hw, b.cy - b.hh, b.cx + b.hw, b.cy + b.hh) for b in net_pads(ir, lib).pads["B"]["U1.2"]], boxes)
    # the 3D scene draws the same rectangles
    scene = build_scene(ir, lib, model_dir=None)
    drawn = [s.polygon for s in scene.solids if s.kind == "pad" and s.label == "U1.2"]
    assert _close(sorted((min(x for x, _ in poly), min(y for _, y in poly), max(x for x, _ in poly), max(y for _, y in poly)) for poly in drawn), sorted(boxes), 1e-6)
    # a ring's box is no copper to the SI paths (nothing inside it is known), only its anchor is
    ring = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    assert [(b.cx, b.cy) for b in net_pads(ring, lib).pads["M"]["MK1.1"]] == [(8.0, 4.625)]
    # the silkscreen measures circles as discs: the anchor exactly, the ring's disc a bound (it covers the hole)
    silk = [p for p in pad_copper("MK1", lib.load_footprint(LibraryRef(library="Test", name="RING")), ring.pcb.placement("MK1")) if p.label == "MK1.1"]
    assert [(p.shape.points, p.shape.r, p.bound) for p in silk] == [(((8.0, 4.625),), 0.375, False), (((8.0, 6.0),), 1.75, True)]


def test_silk_meeting_a_bounding_disc_is_not_verified_and_silk_outside_it_passes(tmp_path: Path, lib: KicadLibrary):
    """RINGSILK's own silk circle (r 2.1) keeps clear of the ring's disc (r 1.75): judged, PASS; its silk line inside the ring's hole
    meets the disc that only bounds the ring's copper: NOT_VERIFIED naming it, never FAIL. The placer keeps its reference off the disc."""
    ir = board_ir(tmp_path, lib, [("MK1", "RINGSILK", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    got = _run("pcb.silk", ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert got.status is S.NOT_VERIFIED, got.message
    assert [b.split(" is ")[0] for b in got.details["custom_pad_bounds"]] == ["MK1:fp_line[0]"], got.details["custom_pad_bounds"]
    assert "the shape that bounds the copper of custom pad MK1.1" in got.message
    plain = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    plain.pcb.silkscreen = place_silkscreen(plain, lib).texts
    assert _run("pcb.silk", plain, tmp_path, lib)[SILK_CLEARANCE_CHECK].status is S.PASS


def test_library_silk_just_outside_a_bounding_disc_is_judged_and_a_designed_text_within_its_margin_is_not(tmp_path: Path, lib: KicadLibrary):
    """RINGNEAR's library silk circle is about 0.065 mm outside the disc (r 1.75) that bounds the ring's copper: a bound is an outer limit,
    so the circle touches no copper (the only rule for a library graphic) - judged, PASS, listed below the text margin as at least that
    far, never under ``custom_pad_bounds``. A designed text 0.1 mm from the disc is within the placer's 0.15 mm margin of a shape that
    only bounds the copper: NOT_VERIFIED naming it, never FAIL."""
    ir = board_ir(tmp_path, lib, [("MK1", "RINGNEAR", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    got = _run("pcb.silk", ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert got.status is S.PASS and "custom_pad_bounds" not in got.details, got.message
    near = [b for b in got.details["below_margin"] if b["b"] == "pad MK1.1"]
    bound = [b for b in near if "at least" in b["message"]]
    # every chord of the circle is at least 0.065 mm from the ring's disc; the chords beside the anchor (a disc known to be copper) are
    # also measured to it, plainly
    assert {b["a"] for b in bound} == {b["a"] for b in near} and len(bound) == 43, near
    assert all(b["a"].startswith("MK1:fp_circle[0].") and b["distance_mm"] == pytest.approx(0.06513, abs=2e-6)
               and "measured to the shape that bounds its copper" in b["message"] for b in bound)
    assert any(b not in bound for b in near) and all(b["distance_mm"] > 0.06 and "at least" not in b["message"] for b in near if b not in bound)
    # a designed text whose estimated box ends 0.1 mm east of the disc (x 9.85; the disc ends at 9.75)
    x = 9.85 + text_extent("X", 0.5, 0.1)[0] / 2.0
    ir.pcb.silkscreen = [*ir.pcb.silkscreen, SilkText(text="X", x_mm=x, y_mm=6.0, size_mm=0.5, thickness_mm=0.1)]
    got = _run("pcb.silk", ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert got.status is S.NOT_VERIFIED and got.details["violations"] == [], got.message
    bounds = got.details["custom_pad_bounds"]
    assert len(bounds) == 1 and " is 0.1 mm from the shape that bounds the copper of custom pad MK1.1" in bounds[0], bounds


def test_a_keep_out_fails_on_known_copper_and_is_not_verified_on_a_bounding_box(tmp_path: Path, lib: KicadLibrary):
    base = board_ir(tmp_path, lib, [("MK1", "RING", 8.0, 6.0), ("R1", "PAD1", 3.0, 3.0)], {"M": [("MK1", "1"), ("R1", "1")]}, (16.0, 12.0))
    # over the ring box's lower-right corner only (x 9.5..10, y 7.5..8): inside the box, outside the ring (1.95 mm from its centre)
    corner = with_keepouts(base.model_copy(deep=True), [keepout("K1", [9.5, 7.5, 0.5, 0.5], forbids=["pads"], layers=["F.Cu"])])
    got = _run("pcb.keepout", corner, tmp_path, lib)[KEEPOUT_CHECK]
    assert got.status is S.NOT_VERIFIED and "the bounding box of MK1.1 reaches it but its copper shape is not judged here" in got.message
    # over the anchor: its copper
    anchor = with_keepouts(base.model_copy(deep=True), [keepout("K2", [7.9, 4.5, 0.2, 0.2], forbids=["pads"], layers=["F.Cu"])])
    got = _run("pcb.keepout", anchor, tmp_path, lib)[KEEPOUT_CHECK]
    assert got.status is S.FAIL and "the copper of MK1.1 lies in it" in got.message


# --------------------------------------------------------------------------- the real KiCad 10.0.6 footprints

REAL = KicadLibrary()
HAS_REAL = REAL.footprint_file("Package_TO_SOT_SMD", "SOT-89-3") is not None and REAL.footprint_file("Sensor_Audio", "CUI_CMC-4013-SMT") is not None
needs_real = pytest.mark.skipif(not HAS_REAL, reason="KiCad 10 footprint libraries with SOT-89-3 / CUI_CMC-4013-SMT not installed (set KICAD10_SYMBOL_DIR)")


@needs_real
def test_the_real_sot89_and_microphone_pads_are_read_as_the_fixtures_say():
    sot = REAL.load_footprint(LibraryRef(library="Package_TO_SOT_SMD", name="SOT-89-3")).pad("2")
    assert [(_box(p), p.exact) for p in custom_pad_parts(_at(8.0, 6.0), sot)] == [(ANCHOR, True), (TAB, True)]
    mic = REAL.load_footprint(LibraryRef(library="Sensor_Audio", name="CUI_CMC-4013-SMT")).pad("1")
    parts = custom_pad_parts(_at(8.0, 6.0), mic)
    assert [(_box(p), p.exact, p.inscribed_r) for p in parts] == [((7.625, 4.25, 8.375, 5.0), False, 0.375), ((6.25, 4.25, 9.75, 7.75), False, 0.0)]


@needs_real
def test_every_custom_pad_of_the_installed_footprint_libraries_is_read_whole():
    """KiCad 10.0.6 has 2026 custom pads in 503 footprints (gr_poly, gr_circle, gr_rect, gr_arc, gr_line; rect or circle anchors): the
    reader reads every primitive of every one of them, and every part is a finite box that holds its pad's anchor centre's row or column."""
    roots = [r / "footprints" for r in REAL.roots if (r / "footprints").is_dir()]
    pads = 0
    for root in roots[:1]:
        for path in sorted(root.glob("*.pretty/*.kicad_mod")):
            text = path.read_text(encoding="utf-8")
            if " custom" not in text:
                continue
            fp = REAL.load_footprint(LibraryRef(library=path.parent.stem, name=path.stem))
            for pad in fp.pads:
                if pad.shape != "custom":
                    continue
                pads += 1
                assert pad.unread_primitives == (), (fp.lib_id, pad.number, pad.unread_primitives)
                parts = custom_pad_parts(_at(), pad)
                assert parts[0].anchor and all(math.isfinite(v) for p in parts for v in _box(p)) and len(parts) == 1 + len(pad.primitives)
    assert pads >= 2000, pads


@needs_real
def test_a_real_sot89_board_routes(tmp_path: Path):
    from ai_eda.ir import BoardOutline, CircuitIR, Component, Net, PCBDesign, Pin, PinElectricalType, PinRef, ProjectMeta
    from tests.test_routing import NET_P

    ir = CircuitIR(project=ProjectMeta(id="sot", name="sot", workdir=str(tmp_path)))
    for ref, lib_name, fp_name, pins in (("U1", "Package_TO_SOT_SMD", "SOT-89-3", ("1", "2", "3")), ("R1", "Resistor_SMD", "R_0603_1608Metric", ("1", "2")),
                                         ("R2", "Resistor_SMD", "R_0603_1608Metric", ("1", "2")), ("R3", "Resistor_SMD", "R_0603_1608Metric", ("1", "2"))):
        ir.components.append(Component(
            ref=ref, value="x", pins=[Pin(number=n, name=n, electrical_type=PinElectricalType.PASSIVE, provenance=NET_P) for n in pins],
            symbol=LibraryRef(library="Device", name="R"), footprint=REAL.resolve_footprint(LibraryRef(library=lib_name, name=fp_name)), provenance=NET_P,
        ))
    ir.nets = [Net(name=n, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=NET_P)
               for n, pins in (("IN", [("U1", "1"), ("R1", "1")]), ("GND", [("U1", "2"), ("R2", "1"), ("R3", "2")]), ("OUT", [("U1", "3"), ("R3", "1")]),
                               ("X", [("R1", "2"), ("R2", "2")]))]
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=20.0, height_mm=14.0), placements=[
        Placement(component_ref="U1", x_mm=10.0, y_mm=7.0, provenance=NET_P), Placement(component_ref="R1", x_mm=3.0, y_mm=3.0, provenance=NET_P),
        Placement(component_ref="R2", x_mm=16.0, y_mm=3.0, provenance=NET_P), Placement(component_ref="R3", x_mm=16.0, y_mm=11.0, provenance=NET_P)])
    r = route_board(ir, REAL, RoutingParams.for_board(ir, REAL))
    assert r.unrouted == {} and r.version == "0.5" and r.stats["custom_pads"][0]["pad"] == "U1.2"
    ir.pcb.tracks, ir.pcb.vias = list(r.tracks), list(r.vias)
    _limit(ir, r.params.clearance_mm)
    checks = _checks(ir, tmp_path, REAL)
    assert checks[CONNECTIVITY_CHECK].status is S.PASS and checks[CLEARANCE_CHECK].status is S.PASS, [c.message for c in checks.values()]
