"""The ``.kicad_sch`` SVG preview (:mod:`ai_eda.gui.schematic_render`).

The strongest correctness check available without KiCad: the divider fixture
of ``tests/fixtures_kicad.py`` is compiled by the real
:class:`~ai_eda.compilers.SchematicCompiler` with every symbol at each
rotation 0 / 90 / 180 / 270 and each mirror (none / x / y) the compiler
supports, written to disk, and the preview drawn from that file must put
every pin's connection point exactly where the compiler's own
``_PlacedPin.position`` says (the compiler's placement structures are
imported, nothing is re-derived here), with the pin running toward the body
along the compiler's ``body_dir``. It runs on the synthetic library of
``tests/test_circuit_templates.py`` everywhere and, where the KiCad
libraries are installed, on the real ``Device:R`` / ``Conn_01x03`` as well.

The real demo schematic is scratch data, not a fixture: set
``AI_EDA_DEMO_KICAD_SCH`` to a compiled ``.kicad_sch`` to render it (skipped
otherwise).
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.compilers.schematic import _PlacedPin, _PlacedSymbol
from ai_eda.compilers.schematic_layout import natural_ref_key, snap
from ai_eda.gui.schematic_render import (
    BEZIER_SEGMENTS,
    CONTENT_MARGIN_MM,
    PAPER_SIZES_MM,
    arc_path,
    render_kicad_sch,
    render_kicad_sch_file,
)
from ai_eda.ir import CircuitIR
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q
from ai_eda.tools.kicad.sexpr import S as SX
from tests.fixtures_kicad import divider_with_connector_ir
from tests.test_circuit_templates import template_library

SVG = "{http://www.w3.org/2000/svg}"
ROTATIONS = (0, 90, 180, 270)
MIRRORS = (None, "x", "y")
DEMO_ENV = "AI_EDA_DEMO_KICAD_SCH"

_probe = KicadLibrary()
HAS_LIBS = _probe.symbol_file("Device") is not None and _probe.symbol_file("Connector_Generic") is not None


# --------------------------------------------------------------------------- helpers


class _TransformedCompiler(SchematicCompiler):
    """The real compiler with every placed symbol given one rotation / mirror; keeps the placement it wrote."""

    def __init__(self, rotation: int, mirror: str | None) -> None:
        self.rotation = rotation
        self.mirror = mirror
        self.placed: dict[str, _PlacedSymbol] = {}

    def _place(self, components, library, pin_nets):  # type: ignore[override]
        placed = super()._place(components, library, pin_nets)
        for ps in placed.values():
            ps.rotation = self.rotation
            ps.mirror = self.mirror
        self.placed = placed
        return placed


def _library(kind: str, tmp_path: Path) -> KicadLibrary:
    if kind == "installed":
        if not HAS_LIBS:
            pytest.skip("KiCad libraries not installed")
        return KicadLibrary()
    return template_library(tmp_path / "kicad")


def _compile(tmp_path: Path, lib: KicadLibrary, rotation: int = 0, mirror: str | None = None) -> tuple[CircuitIR, Path, dict[tuple[str, str], _PlacedPin]]:
    """Compile the divider fixture to ``<tmp>/proj/divider_conn.kicad_sch``; returns the IR, the file and the compiler's placed pins."""
    ir = divider_with_connector_ir(tmp_path / "proj", library=lib)
    compiler = _TransformedCompiler(rotation, mirror)
    ref = compiler.compile(ir, CompileContext(workdir=tmp_path / "proj", tools={"kicad_library": lib}))
    return ir, tmp_path / "proj" / Path(ref.path).name, SchematicCompiler._pin_map(compiler.placed)


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG}svg"
    return root


def _classed(root: ET.Element, tag: str, cls: str) -> list[ET.Element]:
    return [el for el in root.iter(f"{SVG}{tag}") if cls in (el.get("class") or "").split()]


def _symbols(root: ET.Element) -> list[ET.Element]:
    return [g for g in root.iter(f"{SVG}g") if g.get("class") == "symbol"]


def _rendered_pins(root: ET.Element) -> dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]:
    """``(ref, pin number) -> (connection point, inner end)`` read back from the SVG."""
    out: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]] = {}
    for g in _symbols(root):
        for line in _classed(g, "line", "pin"):
            key = (g.get("data-ref") or "", line.get("data-number") or "")
            assert key not in out, f"pin {key} drawn twice"
            out[key] = ((float(line.get("x1")), float(line.get("y1"))), (float(line.get("x2")), float(line.get("y2"))))
    return out


def _effects() -> list:
    return SX("effects", SX("font", SX("size", 1.27, 1.27)))


def _pin(number: str, name: str, x: float, y: float, angle: int, *extra: list) -> list:
    return SX("pin", "passive", "line", SX("at", x, y, angle), SX("length", 2.54), *extra, SX("name", Q(name), _effects()), SX("number", Q(number), _effects()))


def _lib_symbol(lib_id: str, *units: list, pin_names: list | None = None, pin_numbers: list | None = None) -> list:
    return SX("symbol", Q(lib_id), pin_numbers, pin_names, SX("in_bom", True), SX("on_board", True), *units)


def _unit(name: str, *items: list) -> list:
    return SX("symbol", Q(name), *items)


def _instance(lib_id: str, ref: str, x: float, y: float, rot: int = 0, mirror: str | None = None, value: str = "V", unit: int = 1) -> list:
    return SX(
        "symbol", SX("lib_id", Q(lib_id)), SX("at", x, y, rot), SX("mirror", mirror) if mirror else None, SX("unit", unit),
        SX("property", Q("Reference"), Q(ref), SX("at", x + 2.54, y - 1.27, 0), SX("effects", SX("font", SX("size", 1.27, 1.27)), SX("justify", "left"))),
        SX("property", Q("Value"), Q(value), SX("at", x + 2.54, y + 1.27, 0), SX("effects", SX("font", SX("size", 1.27, 1.27)), SX("justify", "left"))),
        SX("property", Q("Footprint"), Q("Hidden:Footprint"), SX("at", x, y, 0), SX("hide", True), _effects()),
    )


def _sheet(lib_symbols: list[list], *items: list, paper: list | None = None) -> list:
    return SX("kicad_sch", SX("version", 20260101), SX("generator", Q("eeschema")), paper if paper is not None else SX("paper", Q("A4")), SX("lib_symbols", *lib_symbols), *items)


def _stroke() -> list:
    return SX("stroke", SX("width", 0), SX("type", "default"))


def _fill(kind: str = "none") -> list:
    return SX("fill", SX("type", kind))


def _arc_contains(d: str, point: tuple[float, float]) -> bool:
    """Whether the SVG arc ``M x1 y1 A r r 0 fa fs x2 y2`` passes through ``point`` (SVG 1.1 F.6.5, independent of the renderer)."""
    m = re.fullmatch(r"M (\S+) (\S+) A (\S+) (\S+) 0 ([01]) ([01]) (\S+) (\S+)", d)
    assert m, d
    x1, y1, r, _, fa, fs, x2, y2 = (float(v) for v in m.groups())
    mx, my = (x1 - x2) / 2, (y1 - y2) / 2
    q = math.hypot(mx, my)
    r = max(r, q)
    k = math.sqrt(max(0.0, r * r - q * q)) / q
    sign = -1 if fa == fs else 1
    cx = sign * k * my + (x1 + x2) / 2
    cy = -sign * k * mx + (y1 + y2) / 2
    t1 = math.atan2(y1 - cy, x1 - cx)
    dt = (math.atan2(y2 - cy, x2 - cx) - t1) % (2 * math.pi)
    if not fs:
        dt -= 2 * math.pi
    tp = (math.atan2(point[1] - cy, point[0] - cx) - t1) % (2 * math.pi)
    on_circle = abs(math.hypot(point[0] - cx, point[1] - cy) - r) < 1e-3
    within = tp <= dt + 1e-9 if fs else (tp - 2 * math.pi) >= dt - 1e-9 or tp < 1e-9
    return on_circle and within


# --------------------------------------------------------------------------- the compiler's placement


@pytest.mark.parametrize("kind", ["synthetic", "installed"])
@pytest.mark.parametrize("mirror", MIRRORS)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_rendered_pin_ends_are_the_compilers_placed_pin_positions(tmp_path: Path, kind: str, rotation: int, mirror: str | None):
    lib = _library(kind, tmp_path)
    ir, sch, pins = _compile(tmp_path, lib, rotation, mirror)
    node = sexpr.parse_file(sch)
    assert all(sexpr.get(s, "at", 3) == str(rotation) for s in sexpr.find_all(node, "symbol"))  # the file states the transform
    assert all(sexpr.get(s, "mirror") == mirror for s in sexpr.find_all(node, "symbol"))
    root = _parse(render_kicad_sch_file(sch))

    rendered = _rendered_pins(root)
    assert set(rendered) == set(pins)
    for key, pp in pins.items():
        conn, inner = rendered[key]
        assert conn == pp.position, f"{key}: drawn at {conn}, the compiler placed it at {pp.position}"
        expected_inner = (snap(pp.position[0] + pp.pin.length * pp.body_dir[0]), snap(pp.position[1] + pp.pin.length * pp.body_dir[1]))
        assert inner == expected_inner, f"{key}: the pin runs to {inner}, not toward the body ({pp.body_dir})"

    # one symbol group per component, one wire stub and one global label per net pin, each stub starting on its drawn pin end
    net_pins = sum(len(n.pins) for n in ir.nets)
    assert [g.get("data-ref") for g in _symbols(root)] == sorted((c.ref for c in ir.components), key=natural_ref_key)
    wires = _classed(root, "line", "wire")
    assert len(wires) == net_pins
    assert len([g for g in root.iter(f"{SVG}g") if g.get("class") == "global-label"]) == net_pins
    ends = {conn for conn, _ in rendered.values()}
    for w in wires:
        start = (float(w.get("x1")), float(w.get("y1")))
        assert start in ends, f"wire stub starting at {start} does not start on a drawn pin end"


@pytest.mark.parametrize("mirror", MIRRORS)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_the_body_turns_with_the_symbol(tmp_path: Path, rotation: int, mirror: str | None):
    """The synthetic resistor body is 5.08 x 2.032 mm in the library: lying at 0 / 180, standing at 90 / 270, a mirror never changes its size."""
    lib = template_library(tmp_path / "kicad")
    _, sch, _ = _compile(tmp_path, lib, rotation, mirror)
    root = _parse(render_kicad_sch_file(sch))
    r1 = next(g for g in _symbols(root) if g.get("data-ref") == "R1")
    (rect,) = _classed(r1, "rect", "body")
    size = (float(rect.get("width")), float(rect.get("height")))
    assert size == ((5.08, 2.032) if rotation in (0, 180) else (2.032, 5.08))
    r1_at = next(sexpr.find(s, "at") for s in sexpr.find_all(sexpr.parse_file(sch), "symbol") if sexpr.get(s, "lib_id") == "Device:R")
    x, y = float(r1_at[1]), float(r1_at[2])
    assert (float(rect.get("x")) + size[0] / 2, float(rect.get("y")) + size[1] / 2) == pytest.approx((x, y))  # centred on its origin


@pytest.mark.parametrize(
    ("rotation", "mirror", "rotate", "anchor"),
    [(0, None, None, "start"), (90, None, "rotate(-90", "start"), (180, None, None, "end"), (270, None, "rotate(-90", "end"), (0, "y", None, "end"), (0, "x", None, "start")],
)
def test_fields_of_a_turned_symbol_stay_readable_and_are_rejustified(tmp_path: Path, rotation: int, mirror: str | None, rotate: str | None, anchor: str):
    """KiCad draws a field of a symbol turned by 90 / 270 vertically and flips its justification so its box lies where the transform puts it."""
    lib = template_library(tmp_path / "kicad")
    _, sch, _ = _compile(tmp_path, lib, rotation, mirror)
    root = _parse(render_kicad_sch_file(sch))
    r1 = next(g for g in _symbols(root) if g.get("data-ref") == "R1")
    (ref,) = [t for t in _classed(r1, "text", "field") if t.get("data-field") == "Reference"]
    assert ref.text == "R1"
    transform = ref.get("transform")
    if rotate is None:
        assert transform is None
    else:
        assert transform is not None and transform.startswith(rotate)
    assert ref.get("text-anchor") == anchor
    assert [t.get("data-field") for t in _classed(r1, "text", "field")] == ["Reference", "Value"]  # hidden Footprint / Datasheet / Description skipped


def test_rendering_is_deterministic(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    _, sch_a, _ = _compile(tmp_path / "a", lib, 90, "y")
    _, sch_b, _ = _compile(tmp_path / "b", lib, 90, "y")
    assert sch_a.read_bytes() == sch_b.read_bytes()
    first = render_kicad_sch_file(sch_a)
    assert first == render_kicad_sch_file(sch_b) == render_kicad_sch(sexpr.parse_file(sch_a))
    assert "<script" not in first and "href" not in first and str(tmp_path) not in first
    root = _parse(first)
    assert root.get("viewBox") == "0 0 297 210" and root.get("width") == "297mm"  # the compiler writes A4


# --------------------------------------------------------------------------- symbol graphics


def test_three_point_arcs_become_svg_arcs_through_their_mid_point():
    lib = _lib_symbol(
        "T:Arcs",
        _unit(
            "Arcs_1_1",
            SX("arc", SX("start", -2.54, 0), SX("mid", 0, 2.54), SX("end", 2.54, 0), _stroke(), _fill()),  # upper half, library frame (Y up)
            SX("arc", SX("start", 5.08, 0), SX("mid", 0, 0), SX("end", 2.54, 2.54), _stroke(), _fill()),  # 270 degrees around (2.54, 0)
        ),
    )
    for mirror, top in ((None, True), ("x", False)):
        root = _parse(render_kicad_sch(_sheet([lib], _instance("T:Arcs", "A1", 100, 100, 0, mirror))))
        half, large = [p.get("d") for p in _classed(root, "path", "arc")]
        assert half.startswith("M 97.46 100 A 2.54 2.54 0 0 ")
        assert _arc_contains(half, (100, 97.46 if top else 102.54))  # the sheet is Y down: the library's upper half is drawn above
        assert not _arc_contains(half, (100, 102.54 if top else 97.46))
        assert large.split()[7] == "1"  # M x y A r r 0 <large-arc> <sweep> x y
        assert _arc_contains(large, (100, 100))  # the mid point (library (0, 0))
        assert _arc_contains(large, (102.54, 102.54 if top else 97.46))  # a quarter the other way round: on the long arc
        assert not _arc_contains(large, (102.54 + 1.796, 100 - (1.796 if top else -1.796)))  # the short gap between start and end
    d, centre, radius = arc_path((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
    assert centre is None and radius == 0.0 and d == "M 0 0 L 1 1 L 2 2"  # collinear: no circle, straight segments


def test_a_bezier_is_flattened_into_sixteen_segments_through_the_curve():
    lib = _lib_symbol("T:Bez", _unit("Bez_1_1", SX("bezier", SX("pts", SX("xy", 0, 0), SX("xy", 0, 2.54), SX("xy", 2.54, 2.54), SX("xy", 2.54, 0)), _stroke(), _fill())))
    root = _parse(render_kicad_sch(_sheet([lib], _instance("T:Bez", "B1", 50, 50, 90))))
    (poly,) = _classed(root, "polyline", "bezier")
    pts = [tuple(float(v) for v in p.split(",")) for p in poly.get("points").split()]
    assert len(pts) == BEZIER_SEGMENTS + 1
    assert pts[0] == (50.0, 50.0)
    assert pts[-1] == (50.0, 47.46)  # library (2.54, 0) at rotation 90: straight up on the sheet
    assert pts[8] == pytest.approx((50 - 1.905, 50 - 1.27))  # B(0.5) = (1.27, 1.905) in the library, turned by 90


def test_rectangle_fill_types_circle_and_polyline():
    lib = _lib_symbol(
        "T:Shapes",
        _unit(
            "Shapes_0_1",
            SX("rectangle", SX("start", -2.54, 2.54), SX("end", 2.54, -2.54), SX("stroke", SX("width", 0.254), SX("type", "default")), _fill("background")),
            SX("polyline", SX("pts", SX("xy", -1, 0), SX("xy", 1, 1), SX("xy", 1, -1), SX("xy", -1, 0)), _stroke(), _fill("outline")),
            SX("circle", SX("center", 0, 1.27), SX("radius", 0.5), _stroke(), _fill("none")),
        ),
        _unit("Shapes_2_1", SX("rectangle", SX("start", 10, 10), SX("end", 12, 12), _stroke(), _fill())),  # another unit: not drawn for unit 1
        _unit("Shapes_1_2", SX("rectangle", SX("start", 20, 20), SX("end", 22, 22), _stroke(), _fill())),  # De Morgan body style: not drawn
    )
    root = _parse(render_kicad_sch(_sheet([lib], _instance("T:Shapes", "U1", 30, 40))))
    (rect,) = _classed(root, "rect", "body")
    assert (rect.get("x"), rect.get("y"), rect.get("width"), rect.get("height")) == ("27.46", "37.46", "5.08", "5.08")
    assert rect.get("fill") == "#fff8dc" and rect.get("stroke") == "#8b0000" and rect.get("stroke-width") == "0.254"
    (poly,) = _classed(root, "polyline", "polyline")
    assert poly.get("fill") == "#8b0000" and poly.get("stroke-width") == "0.15"  # outline fill = the outline colour; width 0 = default line
    (circle,) = _classed(root, "circle", "body")
    assert (circle.get("cx"), circle.get("cy"), circle.get("r"), circle.get("fill")) == ("30", "38.73", "0.5", "none")


# --------------------------------------------------------------------------- pins


def _pin_texts(root: ET.Element, cls: str) -> list[str]:
    return [t.text or "" for t in _classed(root, "text", cls)]


def test_pin_names_and_numbers_follow_the_symbol_and_the_pin():
    pins = lambda: [_pin("1", "IN", -5.08, 0, 0), _pin("2", "OUT", 5.08, 0, 180), _pin("3", "~", 0, -5.08, 90)]  # noqa: E731
    shown = _lib_symbol("T:Shown", _unit("Shown_1_1", *pins()), pin_names=SX("pin_names", SX("offset", 1.016)))
    names_hidden = _lib_symbol("T:NoNames", _unit("NoNames_1_1", *pins()), pin_names=SX("pin_names", SX("offset", 1.016), SX("hide", True)))
    old_style = _lib_symbol("T:Old", _unit("Old_1_1", *pins()), pin_names=["pin_names", "hide"], pin_numbers=["pin_numbers", "hide"])
    hidden_pin = _lib_symbol("T:HidPin", _unit("HidPin_1_1", _pin("1", "IN", -5.08, 0, 0), _pin("2", "PWR", 5.08, 0, 180, SX("hide", True))))

    root = _parse(render_kicad_sch(_sheet([shown], _instance("T:Shown", "U1", 50, 50))))
    assert _pin_texts(root, "pin-name") == ["IN", "OUT"]  # "~" is KiCad's empty name
    assert _pin_texts(root, "pin-number") == ["1", "2", "3"]
    name_in = next(t for t in _classed(root, "text", "pin-name") if t.text == "IN")
    assert float(name_in.get("x")) == pytest.approx(50 - 5.08 + 2.54 + 1.016) and name_in.get("text-anchor") == "start"  # inside the body, at the offset
    num3 = next(t for t in _classed(root, "text", "pin-number") if t.text == "3")
    assert (num3.get("transform") or "").startswith("rotate(-90")  # a vertical pin's number reads bottom-to-top

    root = _parse(render_kicad_sch(_sheet([names_hidden], _instance("T:NoNames", "U1", 50, 50))))
    assert _pin_texts(root, "pin-name") == [] and _pin_texts(root, "pin-number") == ["1", "2", "3"]

    root = _parse(render_kicad_sch(_sheet([old_style], _instance("T:Old", "U1", 50, 50))))
    assert _pin_texts(root, "pin-name") == [] and _pin_texts(root, "pin-number") == []
    assert len(_classed(root, "line", "pin")) == 3  # the pins themselves are still drawn

    root = _parse(render_kicad_sch(_sheet([hidden_pin], _instance("T:HidPin", "U1", 50, 50))))
    lines = {line.get("data-number"): line for line in _classed(root, "line", "pin")}
    assert lines["2"].get("data-hidden") == "yes" and lines["1"].get("data-hidden") is None
    assert _pin_texts(root, "pin-name") == ["IN"] and _pin_texts(root, "pin-number") == ["1"]


def test_names_outside_when_the_offset_is_zero():
    """``(pin_names (offset 0))``: the name above the pin line, the number below (KiCad's rule, e.g. the 2N3904)."""
    lib = _lib_symbol("T:Out", _unit("Out_1_1", _pin("1", "B", -5.08, 0, 0)), pin_names=SX("pin_names", SX("offset", 0)))
    root = _parse(render_kicad_sch(_sheet([lib], _instance("T:Out", "Q1", 50, 50))))
    (name,) = _classed(root, "text", "pin-name")
    (number,) = _classed(root, "text", "pin-number")
    assert float(name.get("y")) < 50 < float(number.get("y"))
    assert name.get("text-anchor") == number.get("text-anchor") == "middle"
    assert float(name.get("x")) == float(number.get("x")) == pytest.approx(50 - 5.08 + 1.27)


# --------------------------------------------------------------------------- sheet items, paper, escaping, refusals


def test_sheet_items_wire_label_junction_no_connect():
    lib = _lib_symbol("T:R", _unit("R_1_1", _pin("1", "~", -5.08, 0, 0)))
    tree = _sheet(
        [lib],
        SX("wire", SX("pts", SX("xy", 10, 20), SX("xy", 30, 20)), _stroke()),
        SX("junction", SX("at", 30, 20), SX("diameter", 0)),
        SX("no_connect", SX("at", 40, 40)),
        SX("global_label", Q("VIN"), SX("shape", "input"), SX("at", 30, 20, 180), SX("effects", SX("font", SX("size", 1.27, 1.27)), SX("justify", "right"))),
        SX("global_label", Q("OUT"), SX("shape", "passive"), SX("at", 10, 20, 90), SX("effects", SX("font", SX("size", 1.27, 1.27)), SX("justify", "left"))),
        SX("label", Q("local"), SX("at", 20, 20, 0), _effects()),
        _instance("T:R", "R1", 60, 60),
    )
    root = _parse(render_kicad_sch(tree))
    (wire,) = _classed(root, "line", "wire")
    assert (wire.get("x1"), wire.get("y1"), wire.get("x2"), wire.get("y2"), wire.get("stroke")) == ("10", "20", "30", "20", "#006400")
    (dot,) = _classed(root, "circle", "junction")
    assert (dot.get("cx"), dot.get("cy"), dot.get("fill")) == ("30", "20", "#006400")
    (nc,) = _classed(root, "path", "no-connect")
    assert nc.get("d") == "M 39.365 39.365 L 40.635 40.635 M 39.365 40.635 L 40.635 39.365"
    vin, out = [g for g in root.iter(f"{SVG}g") if g.get("class") == "global-label"]
    vin_pts = [tuple(float(v) for v in p.split(",")) for p in vin.find(f"{SVG}polygon").get("points").split()]
    assert vin_pts[0] == (30.0, 20.0) and all(x <= 30 for x, _ in vin_pts)  # an input tip on the anchor, the flag pointing along 180 (-x)
    assert vin.find(f"{SVG}text").get("text-anchor") == "end" and vin.find(f"{SVG}text").text == "VIN"
    out_pts = [tuple(float(v) for v in p.split(",")) for p in out.find(f"{SVG}polygon").get("points").split()]
    assert all(y <= 20 for _, y in out_pts) and min(y for _, y in out_pts) < 20 - 3  # angle 90: the flag runs up the sheet
    assert (out.find(f"{SVG}text").get("transform") or "").startswith("rotate(-90")
    (local,) = [g for g in root.iter(f"{SVG}g") if g.get("class") == "label"]
    assert float(local.find(f"{SVG}text").get("y")) < 20  # sits above the wire


def test_paper_sizes_and_unknown_paper_uses_the_content_box():
    wire = SX("wire", SX("pts", SX("xy", 10, 20), SX("xy", 30, 20)), _stroke())
    a3 = _parse(render_kicad_sch(_sheet([], wire, paper=SX("paper", Q("A3")))))
    assert a3.get("viewBox") == "0 0 420 297" and a3.get("width") == "420mm" and a3.get("height") == "297mm"
    assert a3.get("data-content-box") == f"{10 - CONTENT_MARGIN_MM:g} {20 - CONTENT_MARGIN_MM:g} {20 + 2 * CONTENT_MARGIN_MM:g} {2 * CONTENT_MARGIN_MM:g}"
    assert _parse(render_kicad_sch(_sheet([], wire, paper=SX("paper", Q("A4"), "portrait")))).get("viewBox") == "0 0 210 297"
    assert _parse(render_kicad_sch(_sheet([], wire, paper=SX("paper", Q("User"), 100, 50)))).get("viewBox") == "0 0 100 50"
    assert _parse(render_kicad_sch(_sheet([], wire, paper=SX("paper", Q("USLetter"))))).get("viewBox") == "0 0 279.4 215.9"
    assert set(PAPER_SIZES_MM) >= {"A5", "A4", "A3", "A2", "A1", "A0", "A", "B", "C", "D", "E", "USLetter", "USLegal", "USLedger"}
    m = CONTENT_MARGIN_MM
    root = _parse(render_kicad_sch(_sheet([], wire, paper=SX("paper", Q("Weird")))))
    assert root.get("viewBox") == f"{10 - m:g} {20 - m:g} {20 + 2 * m:g} {2 * m:g}" == root.get("data-content-box")
    assert root.get("width") == f"{20 + 2 * m:g}mm"


def test_every_text_is_escaped():
    nasty = '<b>&"x"</b> \'q\' \x01'
    lib = _lib_symbol('T:"Q"<x>', _unit("Q_1_1", _pin("1", "<n&m>", -5.08, 0, 0), _pin("2", "~{RST}", 5.08, 0, 180)))
    tree = _sheet(
        [lib],
        _instance('T:"Q"<x>', "R<1>", 50, 50, value=nasty),
        SX("global_label", Q("</text><script>alert(1)</script>"), SX("shape", "passive"), SX("at", 10, 10, 0), _effects()),
        SX("title_block", SX("title", Q("a & <b>"))),
    )
    svg = render_kicad_sch(tree)
    assert "<script" not in svg and "<b>" not in svg and "\x01" not in svg
    root = _parse(svg)  # well-formed XML
    (sym,) = _symbols(root)
    assert sym.get("data-ref") == "R<1>" and sym.get("data-lib-id") == 'T:"Q"<x>'
    fields = {t.get("data-field"): t.text for t in _classed(root, "text", "field")}
    assert fields == {"Reference": "R<1>", "Value": nasty.replace("\x01", "")}
    assert _pin_texts(root, "pin-name")[0] == "<n&m>"
    overlined = next(t for t in _classed(root, "text", "pin-name") if t.text is None or t.text == "")
    assert overlined.find(f"{SVG}tspan").get("text-decoration") == "overline" and overlined.find(f"{SVG}tspan").text == "RST"
    (label,) = [g for g in root.iter(f"{SVG}g") if g.get("class") == "global-label"]
    assert label.get("data-net") == "</text><script>alert(1)</script>" == label.find(f"{SVG}text").text
    assert root.find(f"{SVG}title").text == "KiCad schematic preview: a & <b>"


def test_a_symbol_missing_from_lib_symbols_is_a_dashed_placeholder():
    root = _parse(render_kicad_sch(_sheet([], _instance("Nope:Gone", "U9", 50, 50))))
    (sym,) = _symbols(root)
    assert sym.get("data-missing") == "yes" and sym.get("data-ref") == "U9"
    assert _classed(sym, "rect", "body")[0].get("stroke-dasharray")


def test_malformed_trees_are_refused():
    with pytest.raises(ValueError, match="not a KiCad schematic"):
        render_kicad_sch(SX("kicad_pcb", SX("version", 1)))
    lib = _lib_symbol("T:R", _unit("R_1_1", _pin("1", "~", -5.08, 0, 0)))
    with pytest.raises(ValueError, match="rotation"):
        render_kicad_sch(_sheet([lib], _instance("T:R", "R1", 50, 50, rot=45)))
    with pytest.raises(ValueError):
        render_kicad_sch(_sheet([lib], SX("wire", SX("pts", SX("xy", "nan", 0), SX("xy", 1, 0)))))
    with pytest.raises(ValueError):
        render_kicad_sch(_sheet([lib], SX("symbol", SX("lib_id", Q("T:R")), SX("unit", 1))))  # no (at x y)


# --------------------------------------------------------------------------- the real demo (scratch data)


def test_the_real_demo_schematic_renders_every_symbol():
    path = os.environ.get(DEMO_ENV)
    if not path or not Path(path).is_file():
        pytest.skip(f"set {DEMO_ENV} to a compiled .kicad_sch (the demo is scratch data, not a fixture)")
    node = sexpr.parse_file(path)
    instances = sexpr.find_all(node, "symbol")
    svg = render_kicad_sch_file(path)
    assert svg == render_kicad_sch_file(path)
    root = _parse(svg)
    groups = _symbols(root)
    assert len(groups) == len(instances) > 0
    assert not [g.get("data-ref") for g in groups if g.get("data-missing")]
    lib = {str(s[1]): s for s in sexpr.find_all(sexpr.find(node, "lib_symbols"), "symbol")}
    for g, inst in zip(groups, instances):
        lib_sym = lib[str(sexpr.get(inst, "lib_id"))]
        pin_count = sum(len(sexpr.find_all(u, "pin")) for u in sexpr.find_all(lib_sym, "symbol"))
        assert len(_classed(g, "line", "pin")) == pin_count, g.get("data-ref")
        assert _classed(g, "rect", "body") + _classed(g, "polyline", "body") + _classed(g, "circle", "body") + _classed(g, "path", "body"), g.get("data-ref")
        assert {t.get("data-field") for t in _classed(g, "text", "field")} == {"Reference", "Value"}
    ends = {conn for conn, _ in _rendered_pins(root).values()}
    wires = _classed(root, "line", "wire")
    assert len(wires) == len(sexpr.find_all(node, "wire"))
    for w in wires:  # every compiler stub starts on a drawn pin end
        assert (float(w.get("x1")), float(w.get("y1"))) in ends
    assert len([g for g in root.iter(f"{SVG}g") if g.get("class") == "global-label"]) == len(sexpr.find_all(node, "global_label"))
