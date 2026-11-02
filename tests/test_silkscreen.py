"""Silkscreen as a designed layer: the IR model, the placer (``silkscreen.place``), the PCBAgent step, the ``pcb.silk.*`` checks
(IR geometry, never DRC) and the PCB compiler writing it.

Offline tests run on a synthetic KiCad library written here (``Test_Silk``: an SMD resistor with KiCad-style silk lines and a
``Reference`` property, a part whose silk mark sticks out above its courtyard, 1x3 and 2x2 THT headers, footprints whose own silk
comes 0.09 mm from / touches a pad, one with a ``custom`` pad, one whose custom pad has a primitive the library reader does not read,
a header with a ``custom`` mounting pad, a pad with a drill offset,
footprints whose own texts meet their own silk / pad) plus ``Device:R`` and ``Connector_Generic`` symbols. Two tests
need the installed KiCad libraries (they skip without them: a real header gets its pin labels and every check passes) and one
needs ``kicad-cli`` too - the silk DRC canary (``silk_over_copper`` / ``silk_overlap`` = 0 on a silk-placed board), NOT measured
on any machine yet; it skips here.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.keys import CONTROL_KEYS, ROUTING_KEY, SILKSCREEN_KEY
from ai_eda.compilers import CompileContext, PCBCompiler, ids
from ai_eda.errors import CompileError
from ai_eda.ir import (
    BoardOutline,
    BoardSide,
    CircuitIR,
    Component,
    LibraryRef,
    ManufacturingConstraints,
    Net,
    PCBDesign,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SilkKind,
    SilkText,
    SourceRef,
    ValidationStatus as S,
    assumption,
    authoritative,
)
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.geometry import rotate
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.capability import NOT_COMPARED as CAPABILITY_NOT_COMPARED
from ai_eda.tools.manufacturing.capability_file import CAPABILITY_KEYS, MM_KEYS, ground_capability, load_capability_file
from ai_eda.tools.silkscreen import PLACER_ID, PLACER_VERSION, SilkParams, place_silkscreen, shape_distance, text_box, text_extent
from ai_eda.tools.silkscreen.geometry import footprint_silk, ir_text_box, pad_copper, side_of_layer
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import (
    SILK_CLEARANCE_CHECK,
    SILK_OVERLAP_CHECK,
    SILK_SIZE_CHECK,
    SILK_TO_EDGE_MM,
    SILK_TO_PAD_MM,
    SILK_TOOL_ID,
    SILK_TOOL_VERSION,
    SilkscreenValidator,
)
from ai_eda.workflow import Orchestrator, Stage
from tests.test_circuit_templates import _lib as symbol_lib
from tests.test_circuit_templates import _pin, _symbol

DATA = Path(__file__).parent / "data"
#: an IR saved with the model as it stood before SilkText existed, and the content hash that model computed for it
PRE_SILK_IR = DATA / "ir_before_silkscreen.json"
PRE_SILK_HASH = "sha256:93d701d3cbfc42e871c5eca2e3145d472b91cd1d4c0c82e26cf51698963eb536"
ANSWERS = {"application": "test", "jurisdiction": "EU"}
P = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand")
SRC = SourceRef(title="Example Fab PCB capabilities", content_hash="sha256:" + "0" * 64)


# --------------------------------------------------------------------------- the synthetic library


def _fp(name: str, body: str, attr: str = "smd") -> str:
    return f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (descr "{name}") (attr {attr})\n{body}  (embedded_fonts no))\n'


def _line(x1: float, y1: float, x2: float, y2: float, layer: str = "F.SilkS", w: float = 0.12) -> str:
    return f'  (fp_line (start {x1} {y1}) (end {x2} {y2}) (stroke (width {w}) (type solid)) (layer "{layer}"))\n'


def _rect(x1: float, y1: float, x2: float, y2: float, layer: str, w: float) -> str:
    return f'  (fp_rect (start {x1} {y1}) (end {x2} {y2}) (stroke (width {w}) (type solid)) (fill no) (layer "{layer}"))\n'


REFERENCE = '  (property "Reference" "REF**" (at 0 {y} 0) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))\n'
VALUE = '  (property "Value" "V" (at 0 {y} 0) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))\n'
SMD_PADS = (
    '  (pad "1" smd roundrect (at -0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask" "F.Paste") (roundrect_rratio 0.25))\n'
    '  (pad "2" smd roundrect (at 0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask" "F.Paste") (roundrect_rratio 0.25))\n'
)


def _tht(n: int, cols: int = 1) -> str:
    out = ""
    for i in range(n):
        col, row = i % cols, i // cols
        shape = "rect" if i == 0 else "circle"
        out += f'  (pad "{i + 1}" thru_hole {shape} (at {2.54 * col} {2.54 * row}) (size 1.7 1.7) (drill 1) (layers "*.Cu" "*.Mask"))\n'
    return out


FOOTPRINTS = {
    # KiCad-style chip resistor: silk lines between the pads (0.24 mm from their rounded copper), Reference above, ${REFERENCE} on F.Fab
    "R": _fp("R", REFERENCE.format(y=-1.5) + VALUE.format(y=1.5) + _line(-0.2, -0.55, 0.2, -0.55) + _line(-0.2, 0.55, 0.2, 0.55)
             + _rect(-1.2, -0.7, 1.2, 0.7, "F.CrtYd", 0.05)
             + '  (fp_text user "${REFERENCE}" (at 0 0 0) (layer "F.Fab") (effects (font (size 0.4 0.4) (thickness 0.06))))\n' + SMD_PADS),
    # a polarity mark on the silk *outside* the courtyard, where the "above" candidate would go
    "MARK": _fp("MARK", REFERENCE.format(y=-1.5) + _line(-1.5, -1.0, 1.5, -1.0) + _rect(-1.2, -0.7, 1.2, 0.7, "F.CrtYd", 0.05) + SMD_PADS),
    "CONN3": _fp("CONN3", REFERENCE.format(y=-2.33) + _rect(-1.33, -1.33, 1.33, 6.41, "F.SilkS", 0.12) + _rect(-1.8, -1.8, 1.8, 6.88, "F.CrtYd", 0.05) + _tht(3), "through_hole"),
    "CONN2x2": _fp("CONN2x2", REFERENCE.format(y=-2.33) + _rect(-1.33, -1.33, 3.87, 3.87, "F.SilkS", 0.12) + _rect(-1.8, -1.8, 4.34, 4.34, "F.CrtYd", 0.05) + _tht(4, cols=2), "through_hole"),
    # the footprint's own silk line 0.09 mm from pad 1 (0.65 - 0.5 - 0.06), like KiCad 10's BarrelJack_Horizontal
    "NEAR": _fp("NEAR", REFERENCE.format(y=-1.8) + _line(-1.2, 0.65, -0.8, 0.65) + _rect(-1.8, -1.0, 1.8, 1.0, "F.CrtYd", 0.05)
                + '  (pad "1" smd rect (at -1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n  (pad "2" smd rect (at 1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
    # the footprint's own silk line across pad 1: silk over copper
    "TOUCH": _fp("TOUCH", REFERENCE.format(y=-1.8) + _line(-1.2, 0.4, -0.8, 0.4) + _rect(-1.8, -1.0, 1.8, 1.0, "F.CrtYd", 0.05)
                 + '  (pad "1" smd rect (at -1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n  (pad "2" smd rect (at 1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
    # a custom pad: its copper is the boxes of its anchor and its polygon (geometry.custom_pad_parts)
    "CUSTOM": _fp("CUSTOM", REFERENCE.format(y=-1.8) + _rect(-1.8, -1.0, 1.8, 1.0, "F.CrtYd", 0.05)
                  + '  (pad "1" smd custom (at -1 0) (size 0.5 0.5) (layers "F.Cu" "F.Mask") (primitives (gr_poly (pts (xy -0.6 -0.6) (xy 0.6 -0.6) (xy 0.6 0.6)) (width 0))))\n'
                  + '  (pad "2" smd rect (at 1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
    # a custom pad with a primitive the library reader does not read: its copper is unknown, the courtyard stands in for it
    "CUSTOMX": _fp("CUSTOMX", REFERENCE.format(y=-1.8) + _rect(-1.8, -1.0, 1.8, 1.0, "F.CrtYd", 0.05)
                   + '  (pad "1" smd custom (at -1 0) (size 0.5 0.5) (layers "F.Cu" "F.Mask") (primitives (gr_blob (pts (xy -0.6 -0.6) (xy 0.6 -0.6) (xy 0.6 0.6)) (width 0))))\n'
                   + '  (pad "2" smd rect (at 1 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
    # a 1x3 header with a custom mounting pad: the boxes of that pad's anchor and polygon are its copper
    "CONNCUSTOM": _fp("CONNCUSTOM", REFERENCE.format(y=-2.33) + _rect(-1.33, -1.33, 1.33, 6.41, "F.SilkS", 0.12) + _rect(-1.8, -1.8, 1.8, 6.88, "F.CrtYd", 0.05) + _tht(3)
                      + '  (pad "MP" smd custom (at 0 6.2) (size 0.3 0.3) (layers "F.Cu" "F.Mask") (primitives (gr_poly (pts (xy -0.4 -0.2) (xy 0.4 -0.2) (xy 0.4 0.2)) (width 0))))\n',
                      "through_hole"),
    # pad 1's copper sits 0.4 mm below its hole (drill offset, KiCad's PAD::ShapePos): the silk line 0.2 mm above that copper would cross
    # the copper if it were centred on the hole
    "OFFSET": _fp("OFFSET", REFERENCE.format(y=-2.5) + _line(-0.4, -0.76, 0.4, -0.76) + _rect(-1.0, -1.2, 3.6, 1.6, "F.CrtYd", 0.05)
                  + '  (pad "1" thru_hole rect (at 0 0) (size 1 1.8) (drill 0.75 (offset 0 0.4)) (layers "*.Cu" "*.Mask"))\n'
                  + '  (pad "2" thru_hole circle (at 2.54 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))\n', "through_hole"),
    # the footprint's own pin-1 text "1" meets its own silk line (by the estimated box) and its "+" sits 0.1 mm from pad 1's copper,
    # like KiCad 10's LED_WS2812B-Mini / AMASS_XT60PW-M: the footprint's own design
    "OWNTEXT": _fp("OWNTEXT", REFERENCE.format(y=-2.2) + _line(-1.3, -0.4, -1.3, 0.4) + _rect(-2.5, -1.6, 3.0, 1.5, "F.CrtYd", 0.05)
                   + '  (fp_text user "1" (at -1.75 0 0) (layer "F.SilkS") (effects (font (size 0.8 0.8) (thickness 0.12))))\n'
                   + '  (fp_text user "+" (at 0 -0.95 0) (layer "F.SilkS") (effects (font (size 0.5 0.5) (thickness 0.1))))\n'
                   + '  (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n  (pad "2" smd rect (at 2 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
    # the footprint's own "A" whose estimated box meets pad 1's copper: not a proof either way (KiCad's silk_over_copper decides)
    "OWNPAD": _fp("OWNPAD", REFERENCE.format(y=-2.2) + _rect(-1.5, -1.6, 3.0, 1.5, "F.CrtYd", 0.05)
                  + '  (fp_text user "A" (at 0 -0.7 0) (layer "F.SilkS") (effects (font (size 0.5 0.5) (thickness 0.1))))\n'
                  + '  (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n  (pad "2" smd rect (at 2 0) (size 1 1) (layers "F.Cu" "F.Mask"))\n'),
}


def silk_library(root: Path) -> KicadLibrary:
    """``Device:R``, ``Connector_Generic:Conn_01x03`` / ``Conn_02x02_Odd_Even`` and the ``Test_Silk`` footprints above."""
    two = [_pin("1", "~", -5.08, 0), _pin("2", "~", 5.08, 180)]
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    (root / "symbols" / "Device.kicad_sym").write_text(sexpr.dumps(symbol_lib(_symbol("R", "R", two, "Test_Silk:R"))), encoding="utf-8")
    conn = symbol_lib(
        _symbol("Conn_01x03", "J", [_pin(str(i), f"Pin_{i}", -5.08, 0, y=2.54 * (1 - i)) for i in (1, 2, 3)], "Test_Silk:CONN3"),
        _symbol("Conn_02x02_Odd_Even", "J", [_pin(str(i), f"Pin_{i}", -5.08 if i % 2 else 5.08, 0 if i % 2 else 180, y=-2.54 * ((i - 1) // 2)) for i in (1, 2, 3, 4)], "Test_Silk:CONN2x2"),
    )
    (root / "symbols" / "Connector_Generic.kicad_sym").write_text(sexpr.dumps(conn), encoding="utf-8")
    pretty = root / "footprints" / "Test_Silk.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    for name, text in FOOTPRINTS.items():
        (pretty / f"{name}.kicad_mod").write_text(text, encoding="utf-8")
    return KicadLibrary(roots=[root])


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return silk_library(tmp_path / "kicad")


#: ref -> (footprint, symbol library, symbol, pin numbers)
KINDS = {
    "R": ("R", "Device", "R", ("1", "2")),
    "M": ("MARK", "Device", "R", ("1", "2")),
    "J": ("CONN3", "Connector_Generic", "Conn_01x03", ("1", "2", "3")),
    "H": ("CONN2x2", "Connector_Generic", "Conn_02x02_Odd_Even", ("1", "2", "3", "4")),
    "N": ("NEAR", "Device", "R", ("1", "2")),
    "T": ("TOUCH", "Device", "R", ("1", "2")),
    "C": ("CUSTOM", "Device", "R", ("1", "2")),
    "X": ("CUSTOMX", "Device", "R", ("1", "2")),
    "P": ("CONNCUSTOM", "Connector_Generic", "Conn_01x03", ("1", "2", "3")),
    "D": ("OFFSET", "Device", "R", ("1", "2")),
    "O": ("OWNTEXT", "Device", "R", ("1", "2")),
    "Q": ("OWNPAD", "Device", "R", ("1", "2")),
}


def _part(lib: KicadLibrary, ref: str) -> Component:
    fp, sym_lib, sym, pins = KINDS[ref[0]]
    c = Component(
        ref=ref, value="10k" if ref[0] != "J" else "Conn", description="silk fixture",
        pins=[Pin(number=n, name="~", electrical_type=PinElectricalType.PASSIVE, provenance=P) for n in pins],
        symbol=LibraryRef(library=sym_lib, name=sym), footprint=LibraryRef(library="Test_Silk", name=fp), provenance=P,
    )
    c.symbol = lib.resolve_symbol(c.symbol)
    c.footprint = lib.resolve_footprint(c.footprint)
    assert c.symbol.verified and c.footprint.verified, ref
    return c


def silk_ir(tmp_path: Path, lib: KicadLibrary, parts: list[tuple], nets: dict[str, list[tuple[str, str]]], size: tuple[float, float], name: str = "silk") -> CircuitIR:
    """``parts`` = ``(ref, x, y[, rotation[, side]])``; the footprint / symbol follow the ref's letter (:data:`KINDS`)."""
    ir = CircuitIR(project=ProjectMeta(id="silk", name=name, workdir=str(tmp_path)))
    placements = []
    for ref, x, y, *rest in parts:
        ir.components.append(_part(lib, ref))
        rot = rest[0] if rest else 0.0
        side = rest[1] if len(rest) > 1 else BoardSide.TOP
        placements.append(Placement(component_ref=ref, x_mm=x, y_mm=y, rotation_deg=rot, side=side, provenance=P))
    for net, pins in nets.items():
        ir.nets.append(Net(name=net, pins=[PinRef(component_ref=r, pin_number=n) for r, n in pins], provenance=P))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=size[0], height_mm=size[1]), placements=placements)
    return ir


#: R1 plain, R2 turned 90 degrees, M1 with its mark above, J1 a 1x3 header right of centre (labels towards the board centre: left)
BASIC_PARTS = [("R1", 10.0, 10.0), ("R2", 20.0, 10.0, 90.0), ("M1", 10.0, 15.0), ("J1", 32.0, 5.0)]
BASIC_NETS = {"VIN": [("J1", "1"), ("R1", "1")], "Net-(R1-Pad2)": [("R1", "2"), ("R2", "1")], "GND": [("R2", "2"), ("J1", "3"), ("M1", "2")], "SIG": [("M1", "1")]}


def basic_ir(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    return silk_ir(tmp_path, lib, BASIC_PARTS, BASIC_NETS, (40.0, 20.0))


def _texts(res, kind: SilkKind) -> dict[str, SilkText]:
    return {(t.component_ref if kind == SilkKind.REFERENCE else t.text): t for t in res.texts if t.kind == kind}


def _checks(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary | None) -> dict[str, object]:
    tools = {"kicad_library": lib} if lib is not None else {}
    results = default_registry.get(SILK_TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path, tools=tools))
    assert [r.check_id for r in results] == [SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK]
    for r in results:
        assert r.tool == SILK_TOOL_ID == "pcb.silk" and r.tool_version == SILK_TOOL_VERSION == "0.1" and r.details.get("kind", "ir_geometry") == "ir_geometry"
        assert "not DRC" in r.message and "drc" not in r.check_id  # never readable as KiCad's verdict
    return {r.check_id: r for r in results}


# --------------------------------------------------------------------------- the IR model


def test_an_ir_saved_before_the_silkscreen_model_keeps_its_hash_and_round_trips(tmp_path: Path):
    ir = CircuitIR.load(PRE_SILK_IR)
    raw = json.loads(PRE_SILK_IR.read_text(encoding="utf-8"))
    assert "silkscreen" not in raw["pcb"] and "min_silk_text_height_mm" not in raw["pcb"]["manufacturing"]  # really written by the old model
    assert ir.content_hash() == PRE_SILK_HASH  # the hash the old model computed for it
    design = ir.design_dict()
    assert "silkscreen" not in design["pcb"] and not set(design["pcb"]["manufacturing"]) & {"min_silk_text_height_mm", "min_silk_line_width_mm"}
    # saved again by the new model: the fields are in the file, the hash is the same, and it loads back the same
    ir.save(tmp_path / "ir.json")
    again = json.loads((tmp_path / "ir.json").read_text(encoding="utf-8"))
    assert again["pcb"]["silkscreen"] == [] and again["pcb"]["manufacturing"]["min_silk_text_height_mm"] is None
    assert CircuitIR.load(tmp_path / "ir.json").content_hash() == PRE_SILK_HASH
    # content is hashed as soon as there is some
    ir.pcb.silkscreen = [SilkText(text="HELLO", x_mm=5.0, y_mm=2.0, provenance=USER)]
    assert ir.content_hash() != PRE_SILK_HASH and ir.design_dict()["pcb"]["silkscreen"][0]["text"] == "HELLO"
    ir.save(tmp_path / "ir2.json")
    back = CircuitIR.load(tmp_path / "ir2.json")
    assert back.pcb.silkscreen == ir.pcb.silkscreen and back.content_hash() == ir.content_hash()
    ir.pcb.silkscreen = []
    assert ir.content_hash() == PRE_SILK_HASH
    ir.pcb.manufacturing.min_silk_text_height_mm = assumption(1.0, note="a limit", unit="mm")
    assert ir.content_hash() != PRE_SILK_HASH


def test_silk_text_defaults_traceability_and_the_control_key():
    t = SilkText(text="R1", x_mm=1.0, y_mm=2.0)
    assert (t.layer, t.size_mm, t.thickness_mm, t.justify, t.kind, t.rotation_deg) == ("F.SilkS", 1.0, 0.15, "center", SilkKind.USER, 0.0)
    assert t.provenance.needs_verification  # nobody said where it came from: an assumption
    board = PCBDesign(silkscreen=[t])
    assert board.layout_items()[-1] == ("silk[0:user:R1]", t.provenance)
    assert SILKSCREEN_KEY == "pcb.silkscreen" and SILKSCREEN_KEY in CONTROL_KEYS
    # the silk limits are capability keys like the others (lengths, grounded through --fab-capability); mfg.capability does not compare them
    assert {"min_silk_text_height_mm", "min_silk_line_width_mm"} <= CAPABILITY_KEYS and {"min_silk_text_height_mm", "min_silk_line_width_mm"} <= MM_KEYS
    assert {"min_silk_text_height_mm", "min_silk_line_width_mm"} <= set(CAPABILITY_NOT_COMPARED)


# --------------------------------------------------------------------------- geometry


def test_text_box_estimate_justification_rotation_and_mirror():
    w, h = text_extent("R10", 1.0, 0.15)
    assert (round(w, 6), round(h, 6)) == (2.85, 1.35)  # 3 x 1.0 x 0.9 + 0.15, 1.0 x 1.2 + 0.15
    box = text_box("R10", 10.0, 5.0, 0.0, 1.0, 0.15, "center").bbox()
    assert [round(v, 6) for v in box] == [8.575, 4.325, 11.425, 5.675]
    assert [round(v, 6) for v in text_box("R10", 10.0, 5.0, 0.0, 1.0, 0.15, "left").bbox()] == [10.0, 4.325, 12.85, 5.675]
    assert [round(v, 6) for v in text_box("R10", 10.0, 5.0, 0.0, 1.0, 0.15, "right").bbox()] == [7.15, 4.325, 10.0, 5.675]
    # mirrored (bottom side): a left-justified text runs to the left
    assert [round(v, 6) for v in text_box("R10", 10.0, 5.0, 0.0, 1.0, 0.15, "left", mirrored=True).bbox()] == [7.15, 4.325, 10.0, 5.675]
    # 90 degrees counter-clockwise on screen: the text reads upwards, a left-justified one runs from the anchor to -y
    assert [round(v, 6) for v in text_box("R10", 10.0, 5.0, 90.0, 1.0, 0.15, "left").bbox()] == [9.325, 2.15, 10.675, 5.0]


def test_pad_copper_is_exact_for_rounded_pads_and_unknown_for_custom(tmp_path: Path, lib: KicadLibrary):
    fp = lib.load_footprint(LibraryRef(library="Test_Silk", name="R"))
    pads = pad_copper("R1", fp, Placement(component_ref="R1", x_mm=0.0, y_mm=0.0))
    # roundrect 0.8 x 0.9, rratio 0.25: a 0.4 x 0.5 core grown by r = 0.2
    assert pads[0].shape.r == 0.2 and sorted(pads[0].shape.points) == [(-1.0, -0.25), (-1.0, 0.25), (-0.6, -0.25), (-0.6, 0.25)] and pads[0].sides == {"F"}
    # the chip resistor's own silk line keeps 0.24 mm from the rounded copper (a bounding box would say 0.16)
    silk = footprint_silk("R1", fp, Placement(component_ref="R1", x_mm=0.0, y_mm=0.0))
    line = next(i for i in silk.items if i.kind == "graphic")
    assert shape_distance(line.shape, pads[1].shape) == 0.24
    assert silk.reference.text == "R1" and (silk.reference.x, silk.reference.y, silk.reference.layer) == (0.0, -1.5, "F.SilkS")
    # a custom pad: one rectangle per box of its anchor and its primitives (the extent the router reads), all under its label
    custom = pad_copper("C1", lib.load_footprint(LibraryRef(library="Test_Silk", name="CUSTOM")), Placement(component_ref="C1", x_mm=0.0, y_mm=0.0))
    assert [p.label for p in custom] == ["C1.1", "C1.1", "C1.2"] and all(p.pad_shape == "custom" for p in custom[:2])
    assert [p.shape.bbox() for p in custom[:2]] == [(-1.25, -0.25, -0.75, 0.25), (-1.6, -0.6, -0.4, 0.6)] and all(p.shape.r == 0.0 for p in custom[:2])
    # one with a primitive the reader does not read: unknown copper
    unread = pad_copper("X1", lib.load_footprint(LibraryRef(library="Test_Silk", name="CUSTOMX")), Placement(component_ref="X1", x_mm=0.0, y_mm=0.0))
    assert unread[0].shape is None and unread[0].pad_shape == "custom" and unread[1].shape is not None
    # a through-hole pad is on both silk sides; on the bottom side the silk and the default reference mirror to B.SilkS
    conn = lib.load_footprint(LibraryRef(library="Test_Silk", name="CONN3"))
    assert all(p.sides == {"F", "B"} for p in pad_copper("J1", conn, Placement(component_ref="J1", x_mm=0.0, y_mm=0.0)))
    bottom = footprint_silk("J1", conn, Placement(component_ref="J1", x_mm=5.0, y_mm=5.0, side=BoardSide.BOTTOM))
    assert {i.layer for i in bottom.items} == {"B.SilkS"} and bottom.reference.layer == "B.SilkS" and bottom.reference.y == 5.0 + 2.33


def test_pad_copper_sits_at_the_drill_offset_and_the_hole_stays_at_the_pad_position(tmp_path: Path, lib: KicadLibrary):
    """``(drill 0.75 (offset 0 0.4))``: KiCad draws the copper at ``(at)`` + the offset rotated by the pad angle (``PAD::ShapePos``)."""
    from ai_eda.tools.kicad.geometry import pad_center, pad_copper_center, pads_bbox

    fp = lib.load_footprint(LibraryRef(library="Test_Silk", name="OFFSET"))
    pad1 = fp.pads[0]
    assert (pad1.offset_x, pad1.offset_y) == (0.0, 0.4) and (fp.pads[1].offset_x, fp.pads[1].offset_y) == (0.0, 0.0)
    at = Placement(component_ref="D1", x_mm=10.0, y_mm=10.0)
    copper = pad_copper("D1", fp, at)[0]
    assert pad_center(at, pad1) == (10.0, 10.0) and pad_copper_center(at, pad1) == (10.0, 10.4)
    assert [round(v, 6) for v in copper.shape.bbox()] == [9.5, 9.5, 10.5, 11.3]  # 1 x 1.8 centred 0.4 mm below the hole
    assert pads_bbox(at, fp).y2 == 11.3
    # the offset turns with the footprint and mirrors with it on the bottom side
    assert pad_copper_center(Placement(component_ref="D1", x_mm=10.0, y_mm=10.0, rotation_deg=90.0), pad1) == (10.4, 10.0)
    assert pad_copper_center(Placement(component_ref="D1", x_mm=10.0, y_mm=10.0, side=BoardSide.BOTTOM), pad1) == (10.0, 9.6)
    # the footprint's own silk line is 0.2 mm above the real copper (it would cross copper centred on the hole): PASS, nothing listed
    line = next(i for i in footprint_silk("D1", fp, at).items if i.kind == "graphic")
    assert shape_distance(line.shape, copper.shape) == 0.2
    ir = silk_ir(tmp_path, lib, [("D1", 10.0, 10.0)], {}, (20.0, 20.0), name="off")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.PASS and clearance.details["below_margin"] == [], clearance.message
    # a user text on the offset copper (below the hole, where copper centred on the hole would not reach) is caught
    ir.pcb.silkscreen = [*ir.pcb.silkscreen, SilkText(text="X", x_mm=10.0, y_mm=11.2, size_mm=0.5, thickness_mm=0.1, provenance=USER)]
    assert "overlaps pad D1.1 copper" in _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK].message


# --------------------------------------------------------------------------- the placer


def test_placer_puts_references_beside_their_courtyards_clear_of_every_keep_out(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    before = ir.content_hash()
    res = place_silkscreen(ir, lib)
    assert ir.content_hash() == before  # pure: the IR is never modified
    refs = _texts(res, SilkKind.REFERENCE)
    assert set(refs) == {"R1", "R2", "M1", "J1"} and res.references_on_silk == ["J1", "M1", "R1", "R2"] and res.references_on_fab == []
    # R1: the first candidate, above its courtyard (10 - 0.7 - gap 0.1 - 1.35 / 2), 1.0 mm, KiCad's 0.15 mm stroke
    r1 = refs["R1"]
    assert (r1.x_mm, r1.y_mm, r1.rotation_deg, r1.layer, r1.size_mm, r1.thickness_mm, r1.justify) == (10.0, 8.525, 0.0, "F.SilkS", 1.0, 0.15, "center")
    # M1: its own silk mark sits above the courtyard, so the reference never covers it: the second candidate (below)
    m1 = refs["M1"]
    assert (m1.x_mm, m1.y_mm) == (10.0, 16.475) and "candidate:1:below@0" in m1.provenance.derived_from
    # R2 turned 90 degrees: its courtyard is 1.4 wide and 2.4 tall and flush with its pads, so the box beside it is widened to the
    # pads' keep-out (pad edge 8.8 - (0.15 - 0.1)); above that
    assert (refs["R2"].x_mm, refs["R2"].y_mm) == (20.0, round(10.0 - 1.2 - 0.05 - 0.1 - 0.675, 6))
    # every text keeps the margins from every pad, the outline and the footprint silk (re-measured here)
    placements = {p.component_ref: p for p in ir.pcb.placements}
    pads = [pad for c in ir.components for pad in pad_copper(c.ref, lib.load_footprint(c.footprint), placements[c.ref])]
    graphics = [i for c in ir.components for i in footprint_silk(c.ref, lib.load_footprint(c.footprint), placements[c.ref]).items]
    for t in res.texts:
        box = ir_text_box(t)
        assert all(shape_distance(box, pad.shape) >= 0.15 for pad in pads), t
        assert all(shape_distance(box, g.shape) >= 0.1 for g in graphics if side_of_layer(g.layer) == side_of_layer(t.layer)), t
        x1, y1, x2, y2 = box.bbox()
        assert x1 >= 0.3 and y1 >= 0.3 and x2 <= 39.7 and y2 <= 19.7, t
    # provenance: derived / silkscreen.place with the parameters and the keep-out rules
    for t in res.texts:
        prov = t.provenance
        assert prov.kind is ProvenanceKind.DERIVED and prov.tool == PLACER_ID == "silkscreen.place" and prov.tool_version == PLACER_VERSION == "0.1"
        assert prov.derived_from[-1] == SilkParams().tag() and "tracks are not keep-outs" in prov.note and "vias are tented" in prov.note
        assert "not KiCad's font metrics" in prov.note


def test_placer_is_deterministic(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    a = place_silkscreen(ir, lib)
    b = place_silkscreen(copy.deepcopy(ir), silk_library(tmp_path / "kicad2"))
    assert [design_data(t) for t in a.texts] == [design_data(t) for t in b.texts] and a.description() == b.description() and a.notes == b.notes


def test_edge_keep_out_and_the_fab_fallback(tmp_path: Path, lib: KicadLibrary):
    # a part 1.2 mm under the top edge: "above" would cross the 0.3 mm edge rule, the reference goes below
    ir = silk_ir(tmp_path, lib, [("R1", 5.0, 1.2)], {}, (10.0, 6.0))
    r1 = _texts(place_silkscreen(ir, lib), SilkKind.REFERENCE)["R1"]
    assert (r1.x_mm, r1.y_mm, r1.layer) == (5.0, round(1.2 + 0.7 + 0.1 + 0.675, 6), "F.SilkS")
    # a board barely larger than the courtyard: no candidate at 1.0 or 0.8 mm is free, the reference goes to F.Fab at the courtyard centre
    ir = silk_ir(tmp_path, lib, [("R1", 1.7, 1.2)], {}, (3.4, 2.4), name="tiny")
    res = place_silkscreen(ir, lib)
    r1 = _texts(res, SilkKind.REFERENCE)["R1"]
    assert (r1.layer, r1.x_mm, r1.y_mm, r1.size_mm) == ("F.Fab", 1.7, 1.2, 1.0) and res.references_on_fab == ["R1"] and res.references_on_silk == []
    assert "candidates_tried:20" in r1.provenance.derived_from and "on the fab layer" in r1.provenance.note
    assert "0 reference(s) on the silkscreen, 1 on the fab layer (R1: no free silk position at 1/0.8 mm)" in res.description()
    assert "silkscreen: reference(s) on the fab layer (no free silk position): R1" in res.notes
    assert res.title is None and any(n.startswith("silkscreen title not placed: the 5.62 x 2.02 mm title box is larger than the board") for n in res.notes)
    # the 0.8 mm size is tried before giving up: 1.3 mm between the courtyard and the edge rule fits 0.8 mm text (1.11 + gap) but not 1.0 (1.35)
    ir = silk_ir(tmp_path, lib, [("R1", 2.6, 2.3)], {}, (5.2, 3.5), name="x")
    r1 = _texts(place_silkscreen(ir, lib), SilkKind.REFERENCE)["R1"]
    assert r1.layer == "F.SilkS" and r1.size_mm == 0.8 and "size_mm:0.8" in r1.provenance.derived_from


def test_a_footprint_with_a_custom_pad_gets_its_reference_and_pin_labels_on_an_empty_board(tmp_path: Path, lib: KicadLibrary):
    """A custom pad's copper is the boxes of its anchor and primitives: the reference goes ``text_gap`` outside the courtyard like any
    part's. A pad whose copper is not read (an unread primitive) has the courtyard stand in for it (kept ``silk_to_pad`` clear): the box
    the candidates go around is widened by ``silk_to_pad - text_gap`` too, so the first candidate is free - not the fab layer for every
    such part."""
    ir = silk_ir(tmp_path, lib, [("C1", 50.0, 50.0)], {}, (100.0, 100.0), name="cu")
    res = place_silkscreen(ir, lib)
    c1 = _texts(res, SilkKind.REFERENCE)["C1"]
    assert res.references_on_silk == ["C1"] and res.references_on_fab == [] and c1.layer == "F.SilkS"
    assert "candidate:0:above@0" in c1.provenance.derived_from and c1.size_mm == 1.0
    # courtyard top 49.0; the box bottom text_gap 0.1 mm above it (the polygon's box, top 49.4, is 0.5 mm away): 49.0 - 0.1 - 1.35 / 2
    assert (c1.x_mm, c1.y_mm) == (50.0, 48.225) and round(49.0 - ir_text_box(c1).bbox()[3], 6) == 0.1
    ir = silk_ir(tmp_path, lib, [("X1", 50.0, 50.0)], {}, (100.0, 100.0), name="cx")
    res = place_silkscreen(ir, lib)
    x1 = _texts(res, SilkKind.REFERENCE)["X1"]
    assert res.references_on_silk == ["X1"] and "candidate:0:above@0" in x1.provenance.derived_from
    # courtyard top 49.0; the box bottom 0.15 mm above it: 49.0 - 0.15 - 1.35 / 2
    assert (x1.x_mm, x1.y_mm) == (50.0, 48.175) and round(49.0 - ir_text_box(x1).bbox()[3], 6) == 0.15
    # a header with a custom mounting pad gets every pin label
    ir = silk_ir(tmp_path, lib, [("P1", 50.0, 50.0)], {"A": [("P1", "1")], "B": [("P1", "2")], "C": [("P1", "3")]}, (100.0, 100.0), name="pc")
    res = place_silkscreen(ir, lib)
    assert res.references_on_silk == ["P1"] and res.labels_placed == ["P1.1 A", "P1.2 B", "P1.3 C"] and res.labels_skipped == []
    # the mounting pad's copper is its boxes, not the courtyard: the labels go text_gap outside the (unwidened) courtyard
    assert {t.x_mm for t in _texts(res, SilkKind.PIN_LABEL).values()} == {round(50.0 + 1.8 + 0.1, 6)}


def test_a_reference_prefers_a_spot_outside_every_other_footprints_courtyard(tmp_path: Path, lib: KicadLibrary):
    """A designator inside a neighbour's courtyard reads as the neighbour's: while a free candidate outside every other courtyard
    exists it wins; only when none does, the first free one is taken and the note names the courtyard."""
    # R1's first candidate (above) clears J1's pads and silk but lies inside J1's courtyard (bottom at 4.1 + 6.88): below wins
    ir = silk_ir(tmp_path, lib, [("J1", 10.0, 4.1), ("R1", 10.0, 13.0)], {}, (20.0, 20.0), name="court")
    res = place_silkscreen(ir, lib)
    r1 = _texts(res, SilkKind.REFERENCE)["R1"]
    assert "candidate:1:below@0" in r1.provenance.derived_from and (r1.x_mm, r1.y_mm) == (10.0, 14.475) and res.references_in_courtyard == []
    assert ir_text_box(r1).bbox()[1] > 4.1 + 6.88 and not any("courtyard" in n for n in res.notes)
    # a board so narrow that every other candidate touches the edge rule: the first free one, in J1's courtyard, and the note says so
    ir = silk_ir(tmp_path, lib, [("J1", 2.1, 4.25), ("R1", 2.1, 13.0)], {}, (4.2, 15.0), name="narrow")
    res = place_silkscreen(ir, lib)
    r1 = _texts(res, SilkKind.REFERENCE)["R1"]
    assert r1.layer == "F.SilkS" and "candidate:0:above@0" in r1.provenance.derived_from and ir_text_box(r1).bbox()[1] < 4.25 + 6.88
    assert res.references_in_courtyard == ["R1: within 0.1 mm of the courtyard of J1"]
    assert "no candidate outside every other footprint's courtyard" in r1.provenance.note
    assert "silkscreen: reference(s) at another footprint's courtyard (no candidate outside every other courtyard): R1: within 0.1 mm of the courtyard of J1" in res.notes


def test_connector_pin_labels_go_beside_their_pads_outside_the_courtyard(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    res = place_silkscreen(ir, lib)
    labels = _texts(res, SilkKind.PIN_LABEL)
    # J1.2 has no net and J1's other nets are named; R1.2's Net-(...) is not a connector's; J1 is right of the board centre: labels on its left
    assert set(labels) == {"VIN", "GND"} and res.labels_placed == ["J1.1 VIN", "J1.3 GND"] and res.labels_skipped == []
    vin, gnd = labels["VIN"], labels["GND"]
    assert (vin.x_mm, vin.y_mm, vin.rotation_deg, vin.justify, vin.size_mm, vin.layer, vin.component_ref) == (30.1, 5.0, 0.0, "right", 0.8, "F.SilkS", "J1")
    assert (gnd.x_mm, gnd.y_mm) == (30.1, 10.08) and "pad:J1.3" in gnd.provenance.derived_from and "net:GND" in gnd.provenance.derived_from
    # J1's own reference avoided the label side: above, not left
    assert _texts(res, SilkKind.REFERENCE)["J1"].y_mm < 3.2
    # a double-row header: each column's labels on its own outer side; a horizontal row gets vertical labels
    ir = silk_ir(tmp_path, lib, [("H1", 10.0, 8.0)], {"A": [("H1", "1")], "B": [("H1", "2")], "C": [("H1", "3")], "Net-(H1-Pad4)": [("H1", "4")]}, (30.0, 20.0), name="h")
    labels = _texts(place_silkscreen(ir, lib), SilkKind.PIN_LABEL)
    assert set(labels) == {"A", "B", "C"}
    assert (labels["A"].x_mm, labels["A"].justify) == (round(10.0 - 1.8 - 0.1, 6), "right") and (labels["B"].x_mm, labels["B"].justify) == (round(10.0 + 4.34 + 0.1, 6), "left")
    assert labels["A"].y_mm == 8.0 and labels["C"].y_mm == 10.54
    ir = silk_ir(tmp_path, lib, [("J1", 10.0, 10.0, 90.0)], {"A": [("J1", "1")], "B": [("J1", "2")], "C": [("J1", "3")]}, (30.0, 20.0), name="v")
    labels = _texts(place_silkscreen(ir, lib), SilkKind.PIN_LABEL)
    assert {t.rotation_deg for t in labels.values()} == {90.0} and {t.y_mm for t in labels.values()} == {round(10.0 + 1.8 + 0.1, 6)}  # below: toward the board centre
    assert {t.justify for t in labels.values()} == {"right"} and sorted(t.x_mm for t in labels.values()) == [10.0, 12.54, 15.08]


def test_a_colliding_pin_label_is_skipped_and_named_never_overlapped(tmp_path: Path, lib: KicadLibrary):
    # J1 at the left edge: labels on its left would leave the board; on its right R1's pads lie in the way of pad 2's long label
    ir = silk_ir(tmp_path, lib, [("J1", 2.5, 5.0), ("R1", 5.6, 8.3)], {"A": [("J1", "1")], "LONG_NET_NAME": [("J1", "2")], "C": [("J1", "3")]}, (20.0, 14.0), name="c")
    res = place_silkscreen(ir, lib)
    placed = _texts(res, SilkKind.PIN_LABEL)
    assert set(placed) == {"A", "C"} and [label for label, _ in res.labels_skipped] == ["J1.2 LONG_NET_NAME"]
    assert "within 0.15 mm of pad R1.1" in res.labels_skipped[0][1]
    assert any(n.startswith("silkscreen: pin label(s) skipped (never overlapped): J1.2 LONG_NET_NAME: within") for n in res.notes)
    # a net name the estimate cannot size is skipped, not guessed
    ir = silk_ir(tmp_path, lib, [("J1", 10.0, 5.0)], {"전원": [("J1", "1")]}, (20.0, 14.0), name="k")
    res = place_silkscreen(ir, lib)
    assert _texts(res, SilkKind.PIN_LABEL) == {} and res.labels_skipped == [("J1.1 전원", "net name is not printable ASCII (the text-box estimate assumes Latin glyphs)")]


def test_title_in_the_corner_with_the_most_room_ascii_only(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    res = place_silkscreen(ir, lib)
    [title] = [t for t in res.texts if t.kind == SilkKind.TITLE]
    assert title.text == "silk" and title.size_mm == 1.5 and title.thickness_mm == 0.225 and title.layer == "F.SilkS"
    corner = next(d.split(":", 1)[1] for d in title.provenance.derived_from if d.startswith("corner:"))
    assert res.title_corner == f"at the {corner} corner" and corner in ("top-left", "top-right", "bottom-left", "bottom-right")
    # the corner's anchor: the outline inset by the 0.3 mm edge rule, justified away from the corner's edge
    assert title.justify == ("left" if corner.endswith("left") else "right") and title.x_mm == (0.3 if corner.endswith("left") else 39.7)
    # the bottom-right corner has the most room: its box could grow x4.015 before touching J1's silk outline (bottom edge at 11.41 + 0.06
    # stroke, + 0.1 gap: (19.7 - 11.57) / 2.025); top-left stops at R1's reference, top-right at J1's, bottom-left at M1's
    assert corner == "bottom-right" and (title.x_mm, title.y_mm) == (39.7, 18.687) and "free_scale:4.015" in title.provenance.derived_from
    # no date, no hash in the title; non-ASCII characters are dropped with a note (KiCad's stroke font, Latin estimate)
    ir.project.name = "보드 v1"
    res = place_silkscreen(ir, lib)
    assert [t.text for t in res.texts if t.kind == SilkKind.TITLE] == ["v1"] and any("characters outside printable ASCII dropped" in n for n in res.notes)
    ir.project.name = "보드"
    res = place_silkscreen(ir, lib)
    assert res.title is None and "silkscreen title not placed: the project name has no printable ASCII character" in res.notes


def test_title_slides_along_the_edge_when_every_corner_is_taken(tmp_path: Path, lib: KicadLibrary):
    corners = [("R1", 1.7, 1.2), ("R2", 28.3, 1.2), ("R3", 1.7, 12.8), ("R4", 28.3, 12.8)]
    ir = silk_ir(tmp_path, lib, corners, {}, (30.0, 14.0), name="slide")
    res = place_silkscreen(ir, lib)
    [title] = [t for t in res.texts if t.kind == SilkKind.TITLE]
    assert res.title_corner.startswith("on the top edge, ") and any(n.startswith("silkscreen title: no corner was free; placed on the top edge") for n in res.notes)
    slide = float(next(d.split(":", 1)[1] for d in title.provenance.derived_from if d.startswith("slide_mm:")))
    assert slide > 0 and title.x_mm == round(0.3 + slide, 6) and title.justify == "left"


def test_bottom_side_silk_is_mirrored_on_b_silks(tmp_path: Path, lib: KicadLibrary):
    ir = silk_ir(tmp_path, lib, [("J1", 14.0, 5.0, 0.0, BoardSide.BOTTOM)], {"A": [("J1", "1")]}, (20.0, 14.0), name="b")
    res = place_silkscreen(ir, lib)
    ref = _texts(res, SilkKind.REFERENCE)["J1"]
    label = _texts(res, SilkKind.PIN_LABEL)["A"]
    assert ref.layer == "B.SilkS" and label.layer == "B.SilkS"
    # mirrored: the label left of the header runs to the left with justify "left" (a mirrored left-justified text runs leftwards)
    assert label.x_mm == round(14.0 - 1.8 - 0.1, 6) and label.justify == "left" and ir_text_box(label).bbox()[2] == label.x_mm


# --------------------------------------------------------------------------- the agent


def _agent(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, **answers: str):
    before = ir.content_hash()
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={**ANSWERS, **answers}))
    assert ir.content_hash() == before and res.questions == [] and not res.blocked_on_user
    return res


def test_agent_adds_silk_to_the_one_proposal_only_when_there_is_none(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip"})
    # an already placed board, routing skipped: the only proposal is the existing board plus its silkscreen
    [proposal] = res.proposals
    assert proposal.target == "pcb" and proposal.operation == "set" and proposal.payload.placements == ir.pcb.placements and proposal.payload.tracks == []
    assert design_data(proposal.payload.model_copy(update={"silkscreen": []})) == design_data(ir.pcb)
    assert [design_data(t) for t in proposal.payload.silkscreen] == [design_data(t) for t in place_silkscreen(ir, lib).texts]
    assert res.notes[-1].startswith("silkscreen.place 0.1: 4 reference(s) on the silkscreen, 0 on the fab layer, 2 connector pin label(s) placed, 0 skipped, title 'silk'")
    assert proposal.description == res.notes[-1] and "KiCad DRC decides silk_over_copper / silk_overlap" in proposal.rationale
    Orchestrator.apply_proposals(ir, res.proposals)
    assert len(ir.pcb.silkscreen) == 7
    # existing silkscreen is never replaced - not even a single hand-placed text
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip"})
    assert res.proposals == [] and res.notes[-1] == "silkscreen not placed: ir.pcb already has 7 silkscreen text(s); the agent never replaces silkscreen"
    hand = basic_ir(tmp_path, lib)
    hand.pcb.silkscreen = [SilkText(text="MY BOARD", x_mm=20.0, y_mm=18.0, kind=SilkKind.USER, provenance=USER)]
    res = _agent(hand, tmp_path, lib, **{ROUTING_KEY: "skip"})
    assert res.proposals == [] and res.notes[-1] == "silkscreen not placed: ir.pcb already has 1 silkscreen text(s); the agent never replaces silkscreen"


def test_silkscreen_skip_is_a_control_key(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip", SILKSCREEN_KEY: " Skip "})
    assert res.proposals == [] and res.notes[-1] == "silkscreen skipped by answer"
    res = _agent(ir, tmp_path, lib, **{ROUTING_KEY: "skip", SILKSCREEN_KEY: "yes"})
    assert res.notes[-2].startswith("silkscreen.place 0.1: ") or any(n.startswith("silkscreen.place 0.1: ") for n in res.notes)
    assert "pcb.silkscreen='yes' not understood (the only answer is 'skip'); placing the silkscreen as usual" in res.notes and res.proposals[0].payload.silkscreen
    # a board the placer cannot read: no silk, the reason noted, never a guess
    broken = basic_ir(tmp_path, lib)
    broken.pcb.outline = None
    res = _agent(broken, tmp_path, lib, **{ROUTING_KEY: "skip"})
    assert res.proposals == [] and res.notes[-1] == "silkscreen not placed: ir.pcb.outline is None: the edge keep-out cannot be measured"


def test_through_the_pipeline_the_silk_checks_judge_what_the_board_carries(tmp_path: Path):
    """The grid-placed synthetic board (``tests/test_pcb_agent.py``): its footprints have no ``Reference`` property, so without designed
    silk the compiler leaves each reference at the footprint origin - over the pads - and ``pcb.silk.clearance`` FAILs (honestly); with the
    agent's silk it PASSes. ``pcb.silkscreen`` never becomes a requirement or a question."""
    from tests.test_pcb_agent import parts_ir
    from tests.test_parts_existence import synthetic_library

    for answer in ("skip", None):
        work = tmp_path / str(answer)
        slib = synthetic_library(work / "kicad")
        ir = parts_ir(work, slib)
        extra = {SILKSCREEN_KEY: answer} if answer else {}
        state = Orchestrator(AgentContext(workdir=work, tools={"kicad_library": slib}, answers={**ANSWERS, **extra})).run(ir, stop_after=Stage.IR_BUILD)
        assert ir.requirements.get(SILKSCREEN_KEY) is None and all(q.key != SILKSCREEN_KEY for q in state.optional_questions + state.open_questions)
        placement = state.outcome(Stage.PLACEMENT)
        assert placement.status is S.NOT_VERIFIED
        clearance = ir.validation.latest(SILK_CLEARANCE_CHECK)
        assert clearance.ir_hash == ir.content_hash() and ir.validation.latest(SILK_SIZE_CHECK).status is S.NOT_VERIFIED
        if answer == "skip":
            assert "silkscreen skipped by answer" in placement.message and ir.pcb.silkscreen == []
            assert clearance.status is S.FAIL and "R1:Reference (library position) overlaps pad R1.1 copper" in clearance.message
            assert state.outcome(Stage.IR_BUILD).status is S.FAIL
        else:
            assert "silkscreen.place 0.1: 5 reference(s) on the silkscreen" in placement.message and len(ir.pcb.silkscreen) >= 5
            assert clearance.status is S.PASS and ir.validation.latest(SILK_OVERLAP_CHECK).status is S.PASS
            assert state.outcome(Stage.IR_BUILD).status is not S.FAIL


# --------------------------------------------------------------------------- the checks


def test_silk_validator_is_registered_and_not_applicable_without_a_board(tmp_path: Path, lib: KicadLibrary):
    v = default_registry.get(SILK_TOOL_ID)
    assert isinstance(v, SilkscreenValidator) and v.domains == frozenset() and v.consumes == frozenset()
    ir = CircuitIR(project=ProjectMeta(id="x", name="x", workdir=str(tmp_path)))
    assert all(c.status is S.NOT_APPLICABLE and "no ir.pcb" in c.message for c in _checks(ir, tmp_path, lib).values())
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10.0, height_mm=10.0))
    assert all(c.status is S.NOT_APPLICABLE for c in _checks(ir, tmp_path, lib).values())
    # no library: the footprints' silk and pads are unknown, nothing is claimed
    ir = basic_ir(tmp_path, lib)
    for c in _checks(ir, tmp_path, None).values():
        assert c.status is S.NOT_VERIFIED and "no KiCad library" in c.message


def test_placed_silk_passes_clearance_and_overlap_and_size_needs_a_limit(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    checks = _checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].status is S.PASS and f">= {SILK_TO_PAD_MM:g} mm from pad copper and >= {SILK_TO_EDGE_MM:g} mm inside the outline" in checks[SILK_CLEARANCE_CHECK].message
    assert "the silkscreen placer's margins, not a fab rule" in checks[SILK_CLEARANCE_CHECK].message
    assert checks[SILK_OVERLAP_CHECK].status is S.PASS and checks[SILK_CLEARANCE_CHECK].details["designed_texts"] == 7
    size = checks[SILK_SIZE_CHECK]
    assert size.status is S.NOT_VERIFIED and "no min_silk_text_height_mm in ir.pcb.manufacturing" in size.message and "no min_silk_line_width_mm" in size.message
    # one limit grounded, the other missing: still NOT_VERIFIED, naming the missing key
    ir.pcb.manufacturing = ManufacturingConstraints(min_silk_text_height_mm=authoritative(0.8, SRC, unit="mm"))
    size = _checks(ir, tmp_path, lib)[SILK_SIZE_CHECK]
    assert size.status is S.NOT_VERIFIED and "no min_silk_line_width_mm" in size.message and "min_silk_text_height_mm" not in size.message.split(";")[0]
    # both grounded and met: PASS, naming the limits and their provenance
    ir.pcb.manufacturing = ManufacturingConstraints(min_silk_text_height_mm=authoritative(0.8, SRC, unit="mm"), min_silk_line_width_mm=authoritative(0.1, SRC, unit="mm"))
    size = _checks(ir, tmp_path, lib)[SILK_SIZE_CHECK]
    assert size.status is S.PASS and "min_silk_text_height_mm 0.8 mm (authoritative), min_silk_line_width_mm 0.1 mm (authoritative)" in size.message
    # a fab that wants 1 mm text and 0.15 mm lines: the 0.8 mm pin labels and the library's 0.12 mm silk lines FAIL, each named
    ir.pcb.manufacturing = ManufacturingConstraints(min_silk_text_height_mm=authoritative(1.0, SRC, unit="mm"), min_silk_line_width_mm=authoritative(0.15, SRC, unit="mm"))
    size = _checks(ir, tmp_path, lib)[SILK_SIZE_CHECK]
    assert size.status is S.FAIL
    items = {(r["item"], r["key"]) for r in size.details["violations"]}
    assert ("silk[4:pin_label:VIN]", "min_silk_text_height_mm") in items and ("R1:fp_line[0]", "min_silk_line_width_mm") in items


def test_a_crafted_overlap_and_a_crafted_pad_violation_fail(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    placed = place_silkscreen(ir, lib).texts
    r1 = next(t for t in placed if t.component_ref == "R1" and t.kind == SilkKind.REFERENCE)
    # a user text on top of R1's reference: overlap FAIL naming both
    ir.pcb.silkscreen = [*placed, SilkText(text="X", x_mm=r1.x_mm, y_mm=r1.y_mm, provenance=USER)]
    checks = _checks(ir, tmp_path, lib)
    assert checks[SILK_OVERLAP_CHECK].status is S.FAIL and "silk[7:user:X] touches or overlaps" not in checks[SILK_OVERLAP_CHECK].message
    assert f"silk[{placed.index(r1)}:reference:R1] touches or overlaps silk[7:user:X]" in checks[SILK_OVERLAP_CHECK].message
    # a user text 0.1 mm beside pad R1.1 (its core edge at x = 9.0, grown by 0.2): clearance FAIL with the distance, overlap on the copper too
    ir.pcb.silkscreen = [*placed, SilkText(text="Y", x_mm=10.0 - 1.2 - 0.1 - text_extent("Y", 1.0, 0.15)[0] / 2, y_mm=10.0, provenance=USER)]
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and "silk[7:user:Y] is 0.1 mm from pad R1.1 copper (< 0.15 mm)" in clearance.message
    ir.pcb.silkscreen = [*placed, SilkText(text="Z", x_mm=9.2, y_mm=10.0, provenance=USER)]
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and "silk[7:user:Z] overlaps pad R1.1 copper" in clearance.message
    # a text over the edge rule
    ir.pcb.silkscreen = [*placed, SilkText(text="EDGE", x_mm=20.0, y_mm=0.5, provenance=USER)]
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and "silk[7:user:EDGE] is not inside the outline inset by 0.3 mm" in clearance.message


def test_library_silk_is_judged_by_overlap_only_and_references_default_to_the_library(tmp_path: Path, lib: KicadLibrary):
    # a footprint whose own silk comes 0.09 mm from its pad: listed, not failed; one whose silk crosses its pad: FAIL
    ir = silk_ir(tmp_path, lib, [("N1", 5.0, 5.0)], {}, (12.0, 10.0), name="n")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.PASS and [r["message"] for r in clearance.details["below_margin"]] == [
        "library silk N1:fp_line[0] is 0.09 mm from pad N1.1 (the footprint's own design; below the 0.15 mm text margin)"
    ]
    ir = silk_ir(tmp_path, lib, [("T1", 5.0, 5.0)], {}, (12.0, 10.0), name="t")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.FAIL and "T1:fp_line[0] overlaps pad T1.1 copper" in clearance.message
    # no designed reference: the library's own position is what the compiled board carries, and it is judged
    ir = basic_ir(tmp_path, lib)
    checks = _checks(ir, tmp_path, lib)
    assert set(checks[SILK_CLEARANCE_CHECK].details["library_default_references"]) == {"R1", "R2", "M1", "J1"}
    # a pad whose copper is not read: the clearance cannot be proven (NOT_VERIFIED naming it); overlap is unaffected
    ir = silk_ir(tmp_path, lib, [("X1", 5.0, 5.0)], {}, (12.0, 10.0), name="cx")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts  # the courtyard stands in for the custom pad's unread copper
    checks = _checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].status is S.NOT_VERIFIED and "pad X1.1 of Test_Silk:CUSTOMX has shape 'custom'" in checks[SILK_CLEARANCE_CHECK].message
    assert checks[SILK_OVERLAP_CHECK].status is S.PASS
    # a custom pad whose primitives are read is bounded by its boxes: the clearance is judged (PASS)
    ir = silk_ir(tmp_path, lib, [("C1", 5.0, 5.0)], {}, (12.0, 10.0), name="cu")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    assert _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK].status is S.PASS
    # the edge: a library silk graphic that leaves the outline FAILs; one inside it but within 0.3 mm of it is only listed
    # (M1's mark runs from x - 1.5 to x + 1.5 with a 0.12 mm stroke; the designed reference keeps the library default out of it)
    for x, status, message in (
        (1.4, S.FAIL, "M1:fp_line[0] leaves the outline"),
        (1.7, S.PASS, "library silk M1:fp_line[0] is within 0.3 mm of the outline (the footprint's own design)"),
    ):
        ir = silk_ir(tmp_path, lib, [("M1", x, 5.0)], {}, (12.0, 10.0), name="e")
        ir.pcb.silkscreen = [SilkText(text="M1", x_mm=6.0, y_mm=8.0, kind=SilkKind.REFERENCE, component_ref="M1", provenance=USER)]
        clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
        assert clearance.status is status, clearance.message
        if status is S.FAIL:
            assert message in clearance.message and clearance.details["below_margin"] == []
        else:
            assert [r["message"] for r in clearance.details["below_margin"]] == [message] and clearance.details["violations"] == []


def test_a_footprints_own_library_texts_are_its_design_not_a_failure(tmp_path: Path, lib: KicadLibrary):
    """A stock footprint's own ``fp_text`` (a pin-1 "1", a "+") judged by the conservative text-box estimate against its own silk and
    pads: listed, never a FAIL the design cannot fix (KiCad 10's LED_WS2812B-Mini, AMASS_XT60PW-M); an IR text is still held to it."""
    ir = silk_ir(tmp_path, lib, [("O1", 10.0, 10.0)], {}, (20.0, 20.0), name="own")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    checks = _checks(ir, tmp_path, lib)
    overlap, clearance = checks[SILK_OVERLAP_CHECK], checks[SILK_CLEARANCE_CHECK]
    assert overlap.status is S.PASS and [(r["a"], r["b"]) for r in overlap.details["library_own_overlaps"]] == [("O1:text[1]", "O1:fp_line[0]")]
    assert "1 library text(s) meeting their own footprint's silk, listed" in overlap.message
    # the "+" 0.1 mm from pad 1's copper: below the placer's 0.15 mm text margin, the footprint's own design - listed, not failed
    assert clearance.status is S.PASS and [r["message"] for r in clearance.details["below_margin"]] == [
        "library silk O1:text[+] is 0.1 mm from pad O1.1 (the footprint's own design; below the 0.15 mm text margin)"
    ]
    # an IR text on the same spot is the design's: FAIL, naming it
    one = next(i for i in footprint_silk("O1", lib.load_footprint(ir.components[0].footprint), ir.pcb.placements[0]).items if i.label == "O1:text[1]")
    x1, y1, x2, y2 = one.shape.bbox()
    ir.pcb.silkscreen = [*ir.pcb.silkscreen, SilkText(text="X", x_mm=round((x1 + x2) / 2, 3), y_mm=round((y1 + y2) / 2, 3), size_mm=0.5, thickness_mm=0.1, provenance=USER)]
    overlap = _checks(ir, tmp_path, lib)[SILK_OVERLAP_CHECK]
    assert overlap.status is S.FAIL and "touches or overlaps O1:text[1]" in overlap.message
    # the library default Reference is where the design left it (no designed reference): against its own footprint's silk it still FAILs
    plain = silk_ir(tmp_path, lib, [("M1", 10.0, 10.0)], {}, (20.0, 20.0), name="own")
    overlap = _checks(plain, tmp_path, lib)[SILK_OVERLAP_CHECK]
    assert overlap.status is S.FAIL and "M1:Reference (library position) touches or overlaps M1:fp_line[0]" in overlap.message
    assert overlap.details["library_own_overlaps"] == []
    # a library text whose estimated box meets its own pad: not a proof either way - NOT_VERIFIED naming it, KiCad's DRC decides
    ir = silk_ir(tmp_path, lib, [("Q1", 10.0, 10.0)], {}, (20.0, 20.0), name="pad")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    clearance = _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK]
    assert clearance.status is S.NOT_VERIFIED and clearance.details["violations"] == []
    assert clearance.details["library_text_estimates"] == [
        "the estimated box of library text Q1:text[A] meets pad Q1.1 copper (the estimate is conservative; KiCad's silk_over_copper decides)"
    ] and "Q1:text[A] meets pad Q1.1 copper" in clearance.message


def test_fab_references_are_not_silk_and_refused_texts_fail_all_three(tmp_path: Path, lib: KicadLibrary):
    ir = silk_ir(tmp_path, lib, [("R1", 1.7, 1.2)], {}, (3.4, 2.4), name="tiny")
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    checks = _checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].details["on_fab"] == ["R1"] and checks[SILK_CLEARANCE_CHECK].details["library_default_references"] == []
    ir = basic_ir(tmp_path, lib)
    placed = place_silkscreen(ir, lib).texts
    for bad, why in (
        (SilkText(text="X", x_mm=1.0, y_mm=1.0, layer="F.Cu", provenance=USER), "is on layer 'F.Cu', which is not a silkscreen or fab layer"),
        (SilkText(text="R9", x_mm=1.0, y_mm=1.0, kind=SilkKind.REFERENCE, component_ref="R9", provenance=USER), "names component 'R9', which is not in the IR"),
        (SilkText(text="X", x_mm=1.0, y_mm=1.0, layer="F.Fab", provenance=USER), "only a reference goes there"),
        (SilkText(text="X", x_mm=float("nan"), y_mm=1.0, provenance=USER), "non-finite number"),
        (SilkText(text="X", x_mm=1.0, y_mm=1.0, size_mm=0.0, provenance=USER), "non-positive size"),
        (SilkText(text="A\nB", x_mm=1.0, y_mm=1.0, provenance=USER), "control character"),
    ):
        ir.pcb.silkscreen = [*placed, bad]
        for c in _checks(ir, tmp_path, lib).values():
            assert c.status is S.FAIL and why in c.message and "(the compiler refuses it)" in c.message, (why, c.message)
        with pytest.raises(CompileError, match=r"ir\.pcb\.silkscreen\[7\]|not a finite number"):
            PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    # a reference that is not its component's, one on the other side (each replacing R1's), a second one
    r1 = next(t for t in placed if t.component_ref == "R1")
    for bad, why, extra in (
        (r1.model_copy(update={"text": "R2"}), "is not the reference of its component 'R1'", False),
        (r1.model_copy(update={"layer": "B.SilkS"}), "the other side than its footprint", False),
        (r1.model_copy(), "second reference text for 'R1'", True),
    ):
        ir.pcb.silkscreen = [*placed, bad] if extra else [bad if t is r1 else t for t in placed]
        assert why in _checks(ir, tmp_path, lib)[SILK_CLEARANCE_CHECK].message
        with pytest.raises(CompileError, match=why.replace("'", ".")):
            PCBCompiler().build(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))


def test_silk_limits_ground_through_the_capability_file_and_feed_the_size_check(tmp_path: Path, lib: KicadLibrary):
    from tests.test_fab_capability import FILE_SOURCE, _page, _write_file

    _archive, doc = _page(tmp_path)
    grounded = ground_capability(doc, load_capability_file(_write_file(tmp_path, source=FILE_SOURCE)))
    for key, value in (("min_silk_text_height_mm", 0.8), ("min_silk_line_width_mm", 0.1)):
        traced = grounded.accepted[key]
        assert traced.value == value and traced.unit == "mm" and traced.provenance.kind is ProvenanceKind.AUTHORITATIVE and traced.provenance.source.content_hash == doc.sha256
    ir = basic_ir(tmp_path, lib)
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    ir.pcb.manufacturing = grounded.constraints()
    assert _checks(ir, tmp_path, lib)[SILK_SIZE_CHECK].status is S.PASS


# --------------------------------------------------------------------------- the compiler


def _board(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> tuple[bytes, list]:
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib}))
    data = Path(art.path).read_bytes()
    return data, sexpr.parse(data.decode("utf-8"))


def _footprint(tree: list, ref: str) -> list:
    return next(fp for fp in sexpr.find_all(tree, "footprint") if any(str(p[1]) == "Reference" and str(p[2]) == ref for p in sexpr.find_all(fp, "property")))


def _reference(fp: list) -> list:
    return next(p for p in sexpr.find_all(fp, "property") if str(p[1]) == "Reference")


def _from_stored(placement: Placement, sx: float, sy: float) -> tuple[float, float]:
    """Where KiCad draws a footprint text stored at ``(sx, sy)``: relative to the footprint and rotated with it (no further mirroring)."""
    rx, ry = rotate(sx, sy, placement.rotation_deg)
    return round(placement.x_mm + rx, 6), round(placement.y_mm + ry, 6)


def test_compiler_writes_the_reference_position_from_the_ir_and_board_texts_deterministically(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    first, tree = _board(ir, tmp_path / "a", lib)
    again, _ = _board(copy.deepcopy(ir), tmp_path / "b", silk_library(tmp_path / "kicad2"))
    assert first == again  # byte-identical
    placements = {p.component_ref: p for p in ir.pcb.placements}
    for t in (t for t in ir.pcb.silkscreen if t.kind == SilkKind.REFERENCE):
        prop = _reference(_footprint(tree, t.component_ref))
        at = sexpr.find(prop, "at")
        # KiCad places the stored (relative, un-rotated) position back at the IR's board position
        assert _from_stored(placements[t.component_ref], float(at[1]), float(at[2])) == (t.x_mm, t.y_mm)
        assert float(at[3]) == t.rotation_deg and sexpr.get(prop, "layer") == t.layer and sexpr.find(prop, "hide") is None
        font = sexpr.find(sexpr.find(prop, "effects"), "font")
        assert [float(v) for v in sexpr.args(sexpr.find(font, "size"))] == [t.size_mm, t.size_mm] and float(sexpr.get(font, "thickness")) == t.thickness_mm
        assert sexpr.get(prop, "uuid") == ids.net_item_uuid("silk", "fp_property", t.component_ref, "Reference")
    # R2 is turned 90 degrees: its stored offset is the IR offset rotated back
    r2 = next(t for t in ir.pcb.silkscreen if t.component_ref == "R2")
    at = sexpr.find(_reference(_footprint(tree, "R2")), "at")
    assert (float(at[1]), float(at[2])) == (round(10.0 - r2.y_mm, 6), 0.0)
    # the title and pin labels are gr_text items in IR order, with deterministic uuids and the IR justification
    texts = sexpr.find_all(tree, "gr_text")
    board_texts = [(i, t) for i, t in enumerate(ir.pcb.silkscreen) if t.kind != SilkKind.REFERENCE]
    assert [str(n[1]) for n in texts] == [t.text for _, t in board_texts]
    for node, (i, t) in zip(texts, board_texts):
        assert sexpr.get(node, "layer") == "F.SilkS" and sexpr.get(node, "uuid") == ids.net_item_uuid("silk", "silk", i)
        at = sexpr.find(node, "at")
        assert (float(at[1]), float(at[2]), float(at[3])) == (t.x_mm, t.y_mm, t.rotation_deg)
        justify = sexpr.find(sexpr.find(node, "effects"), "justify")
        assert (sexpr.args(justify) if justify is not None else ["center"]) == [t.justify]
    vin = next(n for n in texts if str(n[1]) == "VIN")
    assert "(gr_text \"VIN\"" in first.decode() and sexpr.args(sexpr.find(sexpr.find(vin, "effects"), "justify")) == ["right"]


def test_compiler_without_silk_keeps_the_library_position_and_writes_fab_and_mirrored_references(tmp_path: Path, lib: KicadLibrary):
    ir = basic_ir(tmp_path, lib)
    _, tree = _board(ir, tmp_path / "plain", lib)
    assert sexpr.find_all(tree, "gr_text") == []
    prop = _reference(_footprint(tree, "R1"))
    assert [float(v) for v in sexpr.args(sexpr.find(prop, "at"))] == [0.0, -1.5, 0.0] and sexpr.get(prop, "layer") == "F.SilkS"
    # a reference that went to the fab layer is written there
    tiny = silk_ir(tmp_path, lib, [("R1", 1.7, 1.2)], {}, (3.4, 2.4), name="tiny")
    tiny.pcb.silkscreen = place_silkscreen(tiny, lib).texts
    _, tree = _board(tiny, tmp_path / "tiny", lib)
    prop = _reference(_footprint(tree, "R1"))
    assert sexpr.get(prop, "layer") == "F.Fab" and [float(v) for v in sexpr.args(sexpr.find(prop, "at"))][:2] == [0.0, 0.0]
    # the bottom side: B.SilkS, mirrored, stored relative to the footprint
    bottom = silk_ir(tmp_path, lib, [("J1", 10.0, 5.0, 0.0, BoardSide.BOTTOM)], {"A": [("J1", "1")]}, (20.0, 14.0), name="b")
    bottom.pcb.silkscreen = place_silkscreen(bottom, lib).texts
    data, tree = _board(bottom, tmp_path / "bottom", lib)
    prop = _reference(_footprint(tree, "J1"))
    ref = next(t for t in bottom.pcb.silkscreen if t.kind == SilkKind.REFERENCE)
    at = sexpr.find(prop, "at")
    assert sexpr.get(prop, "layer") == "B.SilkS" and (float(at[1]), float(at[2])) == (round(ref.x_mm - 10.0, 6), round(ref.y_mm - 5.0, 6))
    assert sexpr.args(sexpr.find(sexpr.find(prop, "effects"), "justify")) == ["mirror"]
    label = next(n for n in sexpr.find_all(tree, "gr_text") if str(n[1]) == "A")
    label_ir = next(t for t in bottom.pcb.silkscreen if t.text == "A")
    assert sexpr.get(label, "layer") == "B.SilkS" and sexpr.args(sexpr.find(sexpr.find(label, "effects"), "justify")) == [label_ir.justify, "mirror"]


def test_the_silk_items_the_compiler_writes_are_the_ones_the_checks_measured(tmp_path: Path, lib: KicadLibrary):
    """Re-read the compiled board: every silk reference / gr_text box computed from the file equals the box of its IR text."""
    ir = basic_ir(tmp_path, lib)
    ir.pcb.silkscreen = place_silkscreen(ir, lib).texts
    _, tree = _board(ir, tmp_path, lib)
    placements = {p.component_ref: p for p in ir.pcb.placements}
    for t in ir.pcb.silkscreen:
        if t.kind == SilkKind.REFERENCE:
            node = _reference(_footprint(tree, t.component_ref))
            at = sexpr.find(node, "at")
            x, y = _from_stored(placements[t.component_ref], float(at[1]), float(at[2]))
        else:
            node = next(n for n in sexpr.find_all(tree, "gr_text") if str(n[1]) == t.text)
            at = sexpr.find(node, "at")
            x, y = float(at[1]), float(at[2])
        assert math.isclose(x, t.x_mm, abs_tol=1e-6) and math.isclose(y, t.y_mm, abs_tol=1e-6), t


# --------------------------------------------------------------------------- real libraries / kicad-cli (skip without them)

_real = KicadLibrary()
HAS_LIBS = _real.footprint_file("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical") is not None and _real.symbol_file("Connector_Generic") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed (set KICAD10_SYMBOL_DIR)")
_kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (_kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")


def _real_divider(tmp_path: Path) -> CircuitIR:
    from tests.fixtures_kicad import divider_with_connector_ir

    ir = divider_with_connector_ir(tmp_path, _real)
    ir.pcb.silkscreen = place_silkscreen(ir, _real).texts
    return ir


@needs_libs
def test_real_library_header_gets_its_pin_labels_and_every_check_passes(tmp_path: Path):
    ir = _real_divider(tmp_path)
    labels = {t.text: t for t in ir.pcb.silkscreen if t.kind == SilkKind.PIN_LABEL}
    assert set(labels) == {n.name for n in ir.nets if any(p.component_ref == "J1" for p in n.pins)}
    checks = _checks(ir, tmp_path, _real)
    assert checks[SILK_CLEARANCE_CHECK].status is S.PASS and checks[SILK_OVERLAP_CHECK].status is S.PASS, [c.message for c in checks.values()]


@needs_kicad
def test_silk_drc_canary_no_silk_over_copper_or_silk_overlap(tmp_path: Path):
    """NOT measured on any machine yet (kicad-cli 10.0.6): the silk-placed board's DRC reports no ``silk_over_copper`` /
    ``silk_overlap`` and accepts the compiled ``gr_text`` / Reference forms. Run it on the Windows PC and record the result."""
    ir = _real_divider(tmp_path)
    art = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": _real}))
    res = _kicad.run_drc(Path(art.path), tmp_path / "drc.json")
    found = {str(v.get("type")) for k in ("errors", "warnings") for v in res.details.get(k, []) if isinstance(v, dict)}
    assert not found & {"silk_over_copper", "silk_overlap", "silk_edge_clearance", "text_height", "text_thickness"}, found
