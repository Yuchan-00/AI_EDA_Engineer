"""The core-and-ring placer (``ai_eda.tools.placement.core_ring``) on a synthetic 64-pad microcontroller board.

Everything runs on a synthetic KiCad library written into ``tmp_path`` by
:func:`mcu_library`: ``Test_MCU:QFP64`` (64 pads of 0.45 x 1.5 mm at 0.8 mm
pitch on four sides, courtyard 17.4 mm square), ``R0603`` (two 0.8 x 0.95 mm
pads 1.65 mm apart), ``XTAL`` (two 1.5 mm THT pads 4.88 mm apart, an
HC-49-like courtyard), ``SW4`` (a 6 mm push button: pads ``1 1 2 2``),
``HDR1x08`` / ``HDR1x04`` / ``HDR1x02`` / ``HDR2x03`` (THT pin headers, 2.54 mm; the
2x03 is used by one test only),
``HOLE`` (one unnumbered NPTH) and ``JACK`` (the pads and courtyard of KiCad
10.0.6's ``BarrelJack_Horizontal``: the body reaches 14 mm past pad 1 on one
side and 2 mm on the other; used by one test only). :func:`mcu_ir` wires a small development
board around ``U1``: decoupling and crystal capacitors, a reset pull-up and
button, two 8-pin port headers, a UART-like 1x04 header, a power-in pair with
a series diode, a power LED and a mounting hole. None of this is a design
claim; the tests check the placer's promises (and :mod:`tests.test_routing` /
:mod:`tests.test_pcb_agent` reuse the fixture for the routing rules).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import (
    BoardOutline,
    CircuitIR,
    Component,
    LibraryRef,
    Net,
    NetKind,
    PCBDesign,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
)
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center, pads_bbox
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement import core_ring
from ai_eda.tools.placement.core_ring import (
    BODY_OVERHANG_MM,
    CORE_MIN_PADS,
    MARGIN_MM,
    OUTLINE_QUANTUM_MM,
    PLACER_ID,
    PLACER_VERSION,
    SPACING_MM,
    body_axis,
    core_ring_placement,
    find_core,
    pad_count,
)
from ai_eda.tools.placement.grid import _disjoint, _resolve

FIX = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
LIB = "Test_MCU"
#: QFP64 pad rows sit 7.7 mm from the centre; pins 1-16 west (top to bottom), 17-32 south, 33-48 east (bottom to top), 49-64 north
QFP_ROW = 7.7
QFP_PITCH = 0.8


def qfp_pad(n: int) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` of synthetic QFP64 pad ``n`` in the footprint frame (y down)."""
    k = (n - 1) % 16
    off = -6.0 + QFP_PITCH * k
    if n <= 16:
        return -QFP_ROW, off, 1.5, 0.45
    if n <= 32:
        return off, QFP_ROW, 0.45, 1.5
    if n <= 48:
        return QFP_ROW, -off, 1.5, 0.45
    return -off, -QFP_ROW, 0.45, 1.5


def _fp(name: str, pads: list[str], courtyard: tuple[float, float, float, float], attr: str = "smd") -> str:
    x1, y1, x2, y2 = courtyard
    return (
        f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr {attr})\n'
        f'  (fp_rect (start {x1} {y1}) (end {x2} {y2}) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n'
        + "".join(f"  {p}\n" for p in pads)
        + ")\n"
    )


def _smd(number: str, x: float, y: float, w: float, h: float) -> str:
    return f'(pad "{number}" smd rect (at {x:g} {y:g}) (size {w:g} {h:g}) (layers "F.Cu" "F.Mask" "F.Paste"))'


def _tht(number: str, x: float, y: float, size: float, shape: str = "circle") -> str:
    return f'(pad "{number}" thru_hole {shape} (at {x:g} {y:g}) (size {size:g} {size:g}) (drill {size / 2:g}) (layers "*.Cu" "*.Mask"))'


def _header(n: int) -> str:
    pads = [_tht(str(i + 1), 0.0, 2.54 * i, 1.7, "rect" if i == 0 else "circle") for i in range(n)]
    return _fp(f"HDR1x{n:02d}", pads, (-1.77, -1.77, 1.77, 2.54 * (n - 1) + 1.77), "through_hole")


def mcu_library(root: Path) -> KicadLibrary:
    """The synthetic ``Test_MCU`` footprint library of the module docstring."""
    pretty = root / "footprints" / f"{LIB}.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    files = {
        "QFP64": _fp("QFP64", [_smd(str(n), *qfp_pad(n)) for n in range(1, 65)], (-8.7, -8.7, 8.7, 8.7)),
        "R0603": _fp("R0603", [_smd("1", -0.825, 0, 0.8, 0.95), _smd("2", 0.825, 0, 0.8, 0.95)], (-1.48, -0.73, 1.48, 0.73)),
        "XTAL": _fp("XTAL", [_tht("1", 0, 0, 1.5), _tht("2", 4.88, 0, 1.5)], (-3.59, -2.83, 8.47, 2.83), "through_hole"),
        "SW4": _fp("SW4", [_tht("1", 0, 0, 2.0), _tht("1", 6.5, 0, 2.0), _tht("2", 0, 4.5, 2.0), _tht("2", 6.5, 4.5, 2.0)], (-1.5, -1.5, 8.0, 6.0), "through_hole"),
        "HDR1x08": _header(8),
        "HDR1x04": _header(4),
        "HDR1x02": _header(2),
        "HOLE": _fp("HOLE", ['(pad "" np_thru_hole circle (at 0 0) (size 3.2 3.2) (drill 3.2) (layers "*.Cu" "*.Mask"))'], (-1.85, -1.85, 1.85, 1.85), "through_hole"),
        "HDR2x03": _fp("HDR2x03", [_tht(str(2 * r + c + 1), 2.54 * c, 2.54 * r, 1.7, "rect" if r == c == 0 else "circle") for r in range(3) for c in range(2)],
                       (-1.77, -1.77, 4.31, 6.85), "through_hole"),
        "JACK": _fp("JACK", [_tht("1", 0, 0, 3.5, "rect"), _tht("2", -6.0, 0, 3.5), _tht("3", -3.0, 4.7, 3.5)], (-14.0, -4.75, 2.0, 6.75), "through_hole"),
    }
    for name, text in files.items():
        (pretty / f"{name}.kicad_mod").write_text(text, encoding="utf-8")
    return KicadLibrary(roots=[root])


#: port A: MCU pins 51..44 to J1 pins 1..8; port B: MCU pins 10..17 to J2 pins 1..8 (the ATmega128 numbering)
PORTS = {"J1": [51, 50, 49, 48, 47, 46, 45, 44], "J2": list(range(10, 18))}


def mcu_parts_and_nets(*, decoupling: int = 2) -> tuple[dict[str, str], dict[str, tuple[NetKind, list[tuple[str, str]]]]]:
    """``{ref: footprint}`` and ``{net: (kind, [(ref, pin)])}`` of the fixture board (``decoupling`` VCC capacitors)."""
    parts = {"U1": "QFP64", "L1": "R0603", "R1": "R0603", "R2": "R0603", "D1": "R0603", "D2": "R0603", "Y1": "XTAL", "SW1": "SW4",
             "J1": "HDR1x08", "J2": "HDR1x08", "J3": "HDR1x04", "J4": "HDR1x02", "H1": "HOLE"}
    caps = [f"C{i}" for i in range(1, decoupling + 1)]
    c_avcc, c_aref, c_rst, c_x1, c_x2, c_vin = (f"C{decoupling + i}" for i in range(1, 7))
    for c in (*caps, c_avcc, c_aref, c_rst, c_x1, c_x2, c_vin):
        parts[c] = "R0603"
    nets: dict[str, tuple[NetKind, list[tuple[str, str]]]] = {
        "VCC": (NetKind.POWER, [("U1", "21"), ("U1", "52"), *((c, "1") for c in caps), ("L1", "1"), ("R1", "1"), ("J3", "1"), ("R2", "1")]),
        "GND": (NetKind.GROUND, [("U1", "22"), ("U1", "53"), ("U1", "63"), *((c, "2") for c in caps), (c_avcc, "2"), (c_aref, "2"), (c_rst, "2"),
                                 (c_x1, "2"), (c_x2, "2"), ("SW1", "2"), ("J3", "2"), ("J4", "2"), (c_vin, "2"), ("D2", "2")]),
        "AVCC": (NetKind.POWER, [("L1", "2"), (c_avcc, "1"), ("U1", "64")]),
        "AREF": (NetKind.SIGNAL, [(c_aref, "1"), ("U1", "62")]),
        "RESET": (NetKind.SIGNAL, [("R1", "2"), (c_rst, "1"), ("SW1", "1"), ("U1", "20")]),
        "XTAL1": (NetKind.SIGNAL, [("Y1", "1"), (c_x1, "1"), ("U1", "24")]),
        "XTAL2": (NetKind.SIGNAL, [("Y1", "2"), (c_x2, "1"), ("U1", "23")]),
        "RXD": (NetKind.SIGNAL, [("U1", "2"), ("J3", "3")]),
        "TXD": (NetKind.SIGNAL, [("U1", "3"), ("J3", "4")]),
        "VIN_RAW": (NetKind.POWER, [("J4", "1"), ("D1", "1")]),
        "VIN": (NetKind.POWER, [("D1", "2"), (c_vin, "1")]),
        "LED": (NetKind.SIGNAL, [("R2", "2"), ("D2", "1")]),
    }
    for header, pins in PORTS.items():
        port = "A" if header == "J1" else "B"
        for i, pin in enumerate(pins):
            nets[f"P{port}{i}"] = (NetKind.SIGNAL, [("U1", str(pin)), (header, str(i + 1))])
    return parts, nets


def mcu_ir(tmp_path: Path, lib: KicadLibrary, *, decoupling: int = 2, outline: BoardOutline | None = None) -> CircuitIR:
    """The fixture board as an IR: verified ``Test_MCU`` footprints, no symbols, the nets of :func:`mcu_parts_and_nets`, no placement."""
    parts, nets = mcu_parts_and_nets(decoupling=decoupling)
    ir = CircuitIR(project=ProjectMeta(id="mcu", name="mcu", workdir=str(tmp_path)))
    for ref in sorted(parts, key=natural_ref_key):
        fp = lib.resolve_footprint(LibraryRef(library=LIB, name=parts[ref]))
        assert fp.verified, parts[ref]
        ir.components.append(Component(ref=ref, value=parts[ref], footprint=fp, provenance=FIX))
    for name, (kind, pins) in nets.items():
        ir.nets.append(Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=FIX))
    if outline is not None:
        ir.pcb = PCBDesign(outline=outline)
    return ir


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return mcu_library(tmp_path / "kicad")


def _centre(box) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


# --------------------------------------------------------------------------- the tool


def test_the_core_is_the_part_with_the_most_pads_centred_on_the_board(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    parts = _resolve(ir, lib)
    assert find_core(parts)[0] == "U1" and find_core(parts)[2] == 64 >= CORE_MIN_PADS
    assert pad_count(lib.load_footprint(ir.component("SW1").footprint)) == 2  # two pads per number: one logical pad each
    assert pad_count(lib.load_footprint(ir.component("H1").footprint)) == 0
    ring = core_ring_placement(ir, lib)
    o = ring.outline
    u1 = next(p for p in ring.placements if p.component_ref == "U1")
    assert ring.core == "U1" and ring.core_pads == 64 and ring.rings["U1"] == "core"
    assert (u1.x_mm, u1.y_mm, u1.rotation_deg, u1.side) == (o.width_mm / 2, o.height_mm / 2, 0.0, "top")  # the exact centre
    assert (o.origin_x_mm, o.origin_y_mm) == (0.0, 0.0)
    # the half sizes are whole multiples of the quantum, so the core centre is on the 0.2 and the 0.25 mm routing grids
    for half in (o.width_mm / 2, o.height_mm / 2):
        assert half == round(half / OUTLINE_QUANTUM_MM) * OUTLINE_QUANTUM_MM and round(half / 0.2, 9).is_integer() and round(half / 0.25, 9).is_integer()
    assert ring.extents["U1"] == footprint_bbox(u1, lib.load_footprint(ir.component("U1").footprint))
    # ties go to the natural ref order: two 64-pad parts, U2 < U10
    two = mcu_ir(tmp_path, lib)
    for c in two.components:
        if c.ref == "U1":
            c.ref = "U10"
    two.components.append(Component(ref="U2", value="QFP64", footprint=ir.component("U1").footprint, provenance=FIX))
    for net in two.nets:
        for pin in net.pins:
            if pin.component_ref == "U1":
                pin.component_ref = "U10"
    assert find_core(_resolve(two, lib))[0] == "U2" and core_ring_placement(two, lib).core == "U2"


def test_rings_are_disjoint_inside_the_outline_and_hold_the_right_parts(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    ring = core_ring_placement(ir, lib)
    by_ring = {k: sorted((r for r, v in ring.rings.items() if v == k), key=natural_ref_key) for k in ("inner", "outer")}
    # inner: every connected pin's net reaches a core pad, <= 4 pads, not a connector / switch
    assert by_ring["inner"] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "L1", "R1", "Y1"]
    # outer: connectors, the switch, the parts on nets without a core pad (VIN_RAW / VIN / LED), the net-less hole
    assert by_ring["outer"] == ["C8", "D1", "D2", "H1", "J1", "J2", "J3", "J4", "R2", "SW1"]
    assert [p.component_ref for p in ring.placements] == sorted(ring.rings, key=natural_ref_key)  # natural order, every part once
    boxes = ring.extents
    refs = list(boxes)
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            assert _disjoint(boxes[a], boxes[b]), (a, b, boxes[a], boxes[b])
    o = ring.outline
    for ref, box in boxes.items():  # at least the margin inside the outline
        assert o.origin_x_mm + MARGIN_MM <= box.x1 + 1e-9 and box.x2 <= o.origin_x_mm + o.width_mm - MARGIN_MM + 1e-9, ref
        assert o.origin_y_mm + MARGIN_MM <= box.y1 + 1e-9 and box.y2 <= o.origin_y_mm + o.height_mm - MARGIN_MM + 1e-9, ref
    # the bands: inner parts start SPACING_MM outside the core extent and stay inside the inner band; outer parts outside the outer band's inner edge
    core = boxes["U1"]
    inner_edge, outer_edge = ring.bands["inner"], ring.bands["outer"]
    for got, want in zip((inner_edge.x1, inner_edge.y1, inner_edge.x2, inner_edge.y2), (core.x1 - SPACING_MM, core.y1 - SPACING_MM, core.x2 + SPACING_MM, core.y2 + SPACING_MM)):
        assert math.isclose(got, want, abs_tol=1e-6), (inner_edge, core)
    depth = max(min(boxes[r].width, boxes[r].height) for r in by_ring["inner"])
    for r in by_ring["inner"]:
        b = boxes[r]
        assert not (b.x1 > inner_edge.x1 + 1e-9 and b.x2 < inner_edge.x2 - 1e-9 and b.y1 > inner_edge.y1 + 1e-9 and b.y2 < inner_edge.y2 - 1e-9), r  # outside the core zone
        assert inner_edge.x1 - depth - 1e-9 <= b.x1 and b.x2 <= inner_edge.x2 + depth + 1e-9 and inner_edge.y1 - depth - 1e-9 <= b.y1 and b.y2 <= inner_edge.y2 + depth + 1e-9, r
        # each inner part touches the inner edge of its band (tangential, flush with the core side)
        assert any(math.isclose(v, e, abs_tol=1e-6) for v, e in ((b.x1, inner_edge.x2), (b.x2, inner_edge.x1), (b.y1, inner_edge.y2), (b.y2, inner_edge.y1))), r
    assert outer_edge.x1 <= inner_edge.x1 - depth - SPACING_MM + 1e-9 and outer_edge.x2 >= inner_edge.x2 + depth + SPACING_MM - 1e-9
    for r in by_ring["outer"]:
        b = boxes[r]
        assert b.x2 <= outer_edge.x1 + 1e-9 or b.x1 >= outer_edge.x2 - 1e-9 or b.y2 <= outer_edge.y1 + 1e-9 or b.y1 >= outer_edge.y2 - 1e-9, r
        # ... and at the board edge: its outer side is exactly MARGIN_MM inside the outline on one side
        assert any(math.isclose(v, e, abs_tol=1e-6) for v, e in ((b.x1, MARGIN_MM), (b.y1, MARGIN_MM), (b.x2, o.width_mm - MARGIN_MM), (b.y2, o.height_mm - MARGIN_MM))), r


def test_parts_lie_tangential_and_connectors_face_out(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    ring = core_ring_placement(ir, lib)
    for ref, (run, _start) in ring.walk.items():
        b = ring.extents[ref]
        along, across = (b.height, b.width) if run in (0, 2) else (b.width, b.height)
        assert along >= across - 1e-9, (ref, run, b)  # the longer extent runs along the side
        assert ring.placements[[p.component_ref for p in ring.placements].index(ref)].rotation_deg in (0.0, 90.0, 180.0, 270.0)
    # a header's pin row is parallel to its side: every pad at the same distance from the edge
    for ref in ("J1", "J2", "J3", "J4"):
        pl = next(p for p in ring.placements if p.component_ref == ref)
        fp = lib.load_footprint(ir.component(ref).footprint)
        run = ring.walk[ref][0]
        across = {round(pad_center(pl, pad)[0 if run in (0, 2) else 1], 6) for pad in fp.pads}
        assert len(across) == 1, (ref, across)
    # a 2-row header (ISP-like, wired to U1 pins 4..6 and the rails) lies with its rows along the edge and the pin-1 row outward
    ir.components.append(Component(ref="J5", value="HDR2x03", footprint=lib.resolve_footprint(LibraryRef(library=LIB, name="HDR2x03")), provenance=FIX))
    for pin, net in (("1", "ISP1"), ("3", "ISP3"), ("5", "ISP5")):
        ir.nets.append(Net(name=net, pins=[PinRef(component_ref="J5", pin_number=pin), PinRef(component_ref="U1", pin_number=str(int(pin) // 2 + 4))], provenance=FIX))
    for net in ir.nets:
        if net.name in ("VCC", "GND"):
            net.pins.append(PinRef(component_ref="J5", pin_number="2" if net.name == "VCC" else "4"))
    ring = core_ring_placement(ir, lib)
    pl = next(p for p in ring.placements if p.component_ref == "J5")
    fp = lib.load_footprint(ir.component("J5").footprint)
    run = ring.walk["J5"][0]
    out = {pad.number: round((pad_center(pl, pad)[0], -pad_center(pl, pad)[1], -pad_center(pl, pad)[0], pad_center(pl, pad)[1])[run], 6) for pad in fp.pads}
    assert out["1"] == out["3"] == out["5"] > out["2"] == out["4"] == out["6"], (run, out)


def test_pull_angles_follow_the_core_pads_and_rails_count_only_without_a_signal(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    ring = core_ring_placement(ir, lib)
    fp = lib.load_footprint(ir.component("U1").footprint)

    def angle_of(*pins: int) -> float:
        sx = sy = 0.0
        for n in pins:
            x, y = fp.pad(str(n)).x, fp.pad(str(n)).y
            sx += math.cos(math.atan2(-y, x))
            sy += math.sin(math.atan2(-y, x))
        return math.degrees(math.atan2(sy, sx)) % 360.0

    assert ring.pulls["C6"] == "signal" and math.isclose(ring.angles["C6"], angle_of(24), abs_tol=1e-6)  # XTAL1, not the three GND pads
    assert ring.pulls["R1"] == "signal" and math.isclose(ring.angles["R1"], angle_of(20), abs_tol=1e-6)  # RESET, not the VCC pads
    assert ring.pulls["Y1"] == "signal" and math.isclose(ring.angles["Y1"], angle_of(24, 23), abs_tol=1e-6)
    assert ring.pulls["J1"] == "signal" and math.isclose(ring.angles["J1"], angle_of(*PORTS["J1"]), abs_tol=1e-6)
    assert ring.pulls["C1"] == "rail" and math.isclose(ring.angles["C1"], angle_of(21, 52, 22, 53, 63), abs_tol=1e-6)  # decoupling: VCC + GND pads
    assert ring.pulls["L1"] == "rail" and math.isclose(ring.angles["L1"], angle_of(21, 52, 64), abs_tol=1e-6)
    # parts joined by nets that reach no core pad are a group and share its angle: D1 (VIN_RAW / VIN, no core pad at all) with J4 and C8, whose
    # only core pads are the three GND pads; the LED D2 and its resistor R2 (joined by LED) with the GND and VCC pads they reach
    for ref in ("C8", "D1", "J4"):
        assert ring.pulls[ref] == "group:C8+D1+J4" and math.isclose(ring.angles[ref], angle_of(22, 53, 63), abs_tol=1e-6), ref
    for ref in ("D2", "R2"):
        assert ring.pulls[ref] == "group:D2+R2" and math.isclose(ring.angles[ref], angle_of(22, 53, 63, 21, 52), abs_tol=1e-6), ref
    # a group member with a signal path of its own keeps it: the port-A header takes the direction of its pins, which are its own
    assert ring.pulls["J2"] == "signal"
    # nothing at all: no angle, packed after the others
    assert ring.pulls["H1"] == "none" and ring.angles["H1"] is None
    outer = sorted((r for r, v in ring.rings.items() if v == "outer"), key=lambda r: ring.walk[r][1])
    assert outer[-1] == "H1"
    # the provenance records the ring, the angle and where it came from, the core and the parameters
    prov = {p.component_ref: p.provenance for p in ring.placements}
    assert prov["C6"].derived_from == [
        "footprint:Test_MCU:R0603", "ring:inner", f"pull_angle_deg:{round(ring.angles['C6'], 3)}", "pull:signal", "core:U1", f"spacing_mm:{SPACING_MM}", f"margin_mm:{MARGIN_MM}",
    ]
    assert prov["D1"].derived_from[1:4] == ["ring:outer", f"pull_angle_deg:{round(ring.angles['C8'], 3)}", "pull:group:C8+D1+J4"]
    assert prov["H1"].derived_from[2:4] == ["pull_angle_deg:none", "pull:none"]
    assert prov["U1"].derived_from[1:4] == ["ring:core", "pull_angle_deg:none", "pull:core"]
    for p in prov.values():
        assert p.kind is ProvenanceKind.DERIVED and p.tool == PLACER_ID == "placement.core_ring" and p.tool_version == PLACER_VERSION == "0.2"
        assert p.inputs == {} and not p.needs_verification and "DRC" in p.note


def test_each_ring_is_walked_in_pull_angle_order(tmp_path: Path, lib: KicadLibrary):
    """Along each ring's counter-clockwise walk, the pull angles never step back more than once (one wrap through 360 degrees)."""
    ring = core_ring_placement(mcu_ir(tmp_path, lib, decoupling=4), lib)
    for which in ("inner", "outer"):
        refs = sorted((r for r, v in ring.rings.items() if v == which and ring.angles[r] is not None), key=lambda r: ring.walk[r][1])
        angles = [ring.angles[r] for r in refs]
        descents = sum(1 for a, b in zip(angles, angles[1:] + angles[:1]) if b < a - 1e-9)
        assert len(refs) >= 5 and descents <= 1, (which, list(zip(refs, angles)))
        # the walk is east -> north -> west -> south: the run index never decreases along it
        runs = [ring.walk[r][0] for r in refs]
        assert runs == sorted(runs), (which, list(zip(refs, runs)))
        # and a part's placed box really is on its run's side of the core
        core = ring.extents["U1"]
        for r in refs:
            b = ring.extents[r]
            side = {0: b.x1 >= core.x2, 1: b.y2 <= core.y1, 2: b.x2 <= core.x1, 3: b.y1 >= core.y2}[ring.walk[r][0]]
            assert side, (r, ring.walk[r], b)
    # the four decoupling capacitors share one pull angle: they sit next to each other, in natural order
    caps = sorted(["C1", "C2", "C3", "C4"], key=lambda r: ring.walk[r][1])
    assert caps == ["C1", "C2", "C3", "C4"] and len({ring.angles[c] for c in caps}) == 1


def test_equal_angles_keep_a_group_together_in_chain_order_and_its_shared_pads_face_each_other(tmp_path: Path, lib: KicadLibrary):
    """Parts with one pull angle are ordered by where it came from, then along their group's chain: the power-in group walks J4 -> D1 -> C8
    (VIN_RAW, then VIN, from its connector), a rail-only part with exactly the LED group's angle never lands between the LED and its
    resistor, and the resistor and the LED turn their shared LED pads toward each other (the orientation cost counts the net's other part)."""
    ir = mcu_ir(tmp_path, lib)
    # J5: a power-out pair on VCC / GND only - rail-only, and its rails are exactly the pads the D2 + R2 group falls back to
    ir.components.append(Component(ref="J5", value="HDR1x02", footprint=lib.resolve_footprint(LibraryRef(library=LIB, name="HDR1x02")), provenance=FIX))
    for net in ir.nets:
        if net.name in ("VCC", "GND"):
            net.pins.append(PinRef(component_ref="J5", pin_number="1" if net.name == "VCC" else "2"))
    ring = core_ring_placement(ir, lib)
    assert ring.pulls["J5"] == "rail" and ring.pulls["D2"] == ring.pulls["R2"] == "group:D2+R2"
    assert ring.angles["J5"] == ring.angles["D2"] == ring.angles["R2"]  # a three-way tie that natural ref order alone would break D2, J5, R2
    order = sorted((r for r, v in ring.rings.items() if v == "outer"), key=lambda r: ring.walk[r][1])
    assert abs(order.index("D2") - order.index("R2")) == 1, order
    assert len({ring.angles[r] for r in ("J4", "D1", "C8")}) == 1
    first = order.index("J4")
    assert order[first:first + 3] == ["J4", "D1", "C8"], order  # the chain from the connector, not C8 < D1 < J4
    pl = {p.component_ref: p for p in ring.placements}

    def at(ref: str, pin: str) -> tuple[float, float]:
        return pad_center(pl[ref], lib.load_footprint(ir.component(ref).footprint).pad(pin))

    assert math.dist(at("R2", "2"), at("D2", "1")) < math.dist(at("R2", "1"), at("D2", "1"))  # R2's LED pad faces the LED
    assert math.dist(at("D2", "1"), at("R2", "2")) < math.dist(at("D2", "2"), at("R2", "2"))  # and the LED's faces R2
    assert math.dist(at("D1", "2"), at("C8", "1")) < math.dist(at("D1", "1"), at("C8", "1"))  # the diode's VIN end toward the capacitor


def test_a_connector_with_a_body_is_turned_so_the_body_ends_at_the_board_edge(tmp_path: Path, lib: KicadLibrary):
    """A barrel jack (JACK: its courtyard reaches 14 mm past pad 1 on one side and 2 mm on the other) is turned radially: the side its body
    overhangs faces out of the board and its extent ends exactly MARGIN_MM inside the outline, so no part stands in front of it; headers and
    the push button have no such body and stay tangential. The outer band gets as deep as the jack is long, so every outer part still sits
    at the edge."""
    jack = lib.load_footprint(LibraryRef(library=LIB, name="JACK"))
    assert body_axis(jack) == "x"
    for name in ("HDR1x08", "HDR1x02", "HDR2x03", "SW4", "XTAL", "R0603", "QFP64", "HOLE"):
        assert body_axis(lib.load_footprint(LibraryRef(library=LIB, name=name))) is None, name
    ir = mcu_ir(tmp_path, lib)
    ir.component("J4").footprint = lib.resolve_footprint(LibraryRef(library=LIB, name="JACK"))
    ring = core_ring_placement(ir, lib)
    o = ring.outline
    pl = next(p for p in ring.placements if p.component_ref == "J4")
    run = ring.walk["J4"][0]
    ext, pads = ring.extents["J4"], pads_bbox(pl, jack)
    outward = (ext.x2 - pads.x2, pads.y1 - ext.y1, pads.x1 - ext.x1, ext.y2 - pads.y2)[run]
    inward = (pads.x1 - ext.x1, ext.y2 - pads.y2, ext.x2 - pads.x2, pads.y1 - ext.y1)[run]
    assert math.isclose(outward, 6.25, abs_tol=1e-6) and outward >= inward + BODY_OVERHANG_MM, (run, pl.rotation_deg, ext, pads)
    edge = (ext.x2, ext.y1, ext.x1, ext.y2)[run]
    assert math.isclose(edge, (o.width_mm - MARGIN_MM, MARGIN_MM, MARGIN_MM, o.height_mm - MARGIN_MM)[run], abs_tol=1e-6), (run, ext, o)
    along, across = (ext.height, ext.width) if run in (0, 2) else (ext.width, ext.height)
    assert across == pytest.approx(16.0) and along == pytest.approx(11.5)  # radial: the jack's 16 mm run across the side
    for ref, b in ring.extents.items():  # nothing between the body's end and the edge
        if ref != "J4":
            overlap_along = (b.y1 < ext.y2 and ext.y1 < b.y2) if run in (0, 2) else (b.x1 < ext.x2 and ext.x1 < b.x2)
            beyond = (b.x1 >= ext.x2, b.y2 <= ext.y1, b.x2 <= ext.x1, b.y1 >= ext.y2)[run]
            assert not (overlap_along and beyond), ref
    for r in (r for r, v in ring.rings.items() if v == "outer"):
        b = ring.extents[r]
        assert any(math.isclose(v, e, abs_tol=1e-6) for v, e in ((b.x1, MARGIN_MM), (b.y1, MARGIN_MM), (b.x2, o.width_mm - MARGIN_MM), (b.y2, o.height_mm - MARGIN_MM))), r
    assert [design_data(p) for p in core_ring_placement(copy.deepcopy(ir), lib).placements] == [design_data(p) for p in ring.placements]


def test_same_ir_same_placement(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    a = core_ring_placement(ir, lib)
    b = core_ring_placement(copy.deepcopy(ir), mcu_library(tmp_path / "kicad2"))
    assert [design_data(p) for p in a.placements] == [design_data(p) for p in b.placements]
    assert a.outline == b.outline and a.extents == b.extents and a.walk == b.walk and a.angles == b.angles and a.rings == b.rings
    # the IR order of components, of nets and of the pins inside each net does not matter either
    for decoupling in (2, 4, 8):
        base = mcu_ir(tmp_path, lib, decoupling=decoupling)
        a = core_ring_placement(base, lib)
        shuffled = copy.deepcopy(base)
        shuffled.components.reverse()
        shuffled.nets.reverse()
        for net in shuffled.nets:
            net.pins.reverse()
        c = core_ring_placement(shuffled, lib)
        assert [design_data(p) for p in c.placements] == [design_data(p) for p in a.placements], decoupling
        assert (c.angles, c.pulls, c.walk, c.outline) == (a.angles, a.pulls, a.walk, a.outline), decoupling
    assert ir.pcb is None  # pure: nothing written into the IR


def test_a_user_outline_is_kept_the_core_centred_and_a_too_small_one_refused(tmp_path: Path, lib: KicadLibrary):
    big = BoardOutline(width_mm=90.0, height_mm=80.0, origin_x_mm=100.0, origin_y_mm=50.0)
    ring = core_ring_placement(mcu_ir(tmp_path, lib), lib, outline=big)
    u1 = next(p for p in ring.placements if p.component_ref == "U1")
    assert ring.outline == big and (u1.x_mm, u1.y_mm) == (145.0, 90.0)
    for ref, b in ring.extents.items():
        assert 100.0 <= b.x1 and b.x2 <= 190.0 and 50.0 <= b.y1 and b.y2 <= 130.0, ref
    # the outer ring sits at the user outline's edge (margin inside it): every outer part touches one side's line
    for r in (r for r, v in ring.rings.items() if v == "outer"):
        b = ring.extents[r]
        edges = ((b.x1, 100.0 + MARGIN_MM), (b.y1, 50.0 + MARGIN_MM), (b.x2, 190.0 - MARGIN_MM), (b.y2, 130.0 - MARGIN_MM))
        assert any(math.isclose(v, e, abs_tol=1e-6) for v, e in edges), (r, b)
    with pytest.raises(CompileError, match=r"the 45.0 x 45.0 mm outline at \(0.0, 0.0\) leaves no room for the outer ring .* not resized"):
        core_ring_placement(mcu_ir(tmp_path, lib), lib, outline=BoardOutline(width_mm=45.0, height_mm=45.0))
    # room for the band but not for its parts (ten more 8-pin headers): refused, never resized; without the outline the band grows instead
    crowded = mcu_ir(tmp_path, lib)
    hdr = crowded.component("J1").footprint
    crowded.components += [Component(ref=f"J{i}", value="HDR1x08", footprint=hdr, provenance=FIX) for i in range(5, 15)]
    with pytest.raises(CompileError, match=r"the outer ring cannot hold its 20 part\(s\) inside the 60.0 x 60.0 mm outline: they need .*not resized"):
        core_ring_placement(crowded, lib, outline=BoardOutline(width_mm=60.0, height_mm=60.0))
    grown = core_ring_placement(crowded, lib)
    assert grown.outline.width_mm > 60.0 and len(grown.placements) == 31 and all(grown.rings[f"J{i}"] == "outer" for i in range(5, 15))


def test_an_overflowing_inner_ring_is_refused_never_overlapped(tmp_path: Path, lib: KicadLibrary):
    """The inner band's size is set by the core: 30 decoupling capacitors (2.96 mm + 1 mm each) do not fit around a 17.4 mm core."""
    assert len(core_ring_placement(mcu_ir(tmp_path, lib, decoupling=8), lib).placements) == 27
    with pytest.raises(CompileError, match=r"the inner ring cannot hold its 38 part\(s\) \(C1, .*\): they need .* mm along a band of .* mm .*refusing rather than overlapping parts"):
        core_ring_placement(mcu_ir(tmp_path, lib, decoupling=30), lib)


def test_refusals_instead_of_guesses(tmp_path: Path, lib: KicadLibrary):
    ir = mcu_ir(tmp_path, lib)
    ir.component("C1").footprint = None
    with pytest.raises(CompileError, match="'C1' has no footprint; the placer never picks one"):
        core_ring_placement(ir, lib)
    ir = mcu_ir(tmp_path, lib)
    ir.component("C1").footprint = LibraryRef(library=LIB, name="Missing")
    with pytest.raises(CompileError, match="Test_MCU:Missing of 'C1' was not found"):
        core_ring_placement(ir, lib)
    with pytest.raises(CompileError, match="no components"):
        core_ring_placement(CircuitIR(project=ProjectMeta(id="e", name="e", workdir=str(tmp_path))), lib)
    for bad in ({"spacing": 0.0}, {"margin": -1.0}):
        with pytest.raises(CompileError, match="spacing must be > 0 and margin >= 0"):
            core_ring_placement(mcu_ir(tmp_path, lib), lib, **bad)


def test_the_guard_refuses_what_the_arithmetic_got_wrong(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """Behind the band arithmetic, the re-measured extents are checked: a band that puts parts on top of each other is refused."""
    original = core_ring._Band.box

    def collapsed(self, run, start, length, depth):
        return original(self, run, self.starts[run], length, depth)  # every part of a run at its start

    monkeypatch.setattr(core_ring._Band, "box", collapsed)
    with pytest.raises(CompileError, match="touch or overlap"):
        core_ring_placement(mcu_ir(tmp_path, lib), lib)


def test_a_core_alone_gets_its_extent_plus_the_margin_rounded_up(tmp_path: Path, lib: KicadLibrary):
    """A board with the core only: no band, no walk, and the outline is the core extent plus the margin, rounded up to the quantum."""
    solo = mcu_ir(tmp_path, lib)
    solo.components = [c for c in solo.components if c.ref == "U1"]
    solo.nets = []
    ring = core_ring_placement(solo, lib)
    assert ring.bands == {} and ring.walk == {} and ring.outline == BoardOutline(width_mm=22.0, height_mm=22.0)  # 17.4 + 2 * 2.0 = 21.4 -> 22
