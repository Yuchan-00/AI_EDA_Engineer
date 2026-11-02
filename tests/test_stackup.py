"""The board stackup: the IR model, older IRs keeping their hash, the generic stacks and plane zones, the ``pcb_layers`` input
and the fab-capability ``stackup`` block grounded verbatim on the archived vendor page.

What is proven here, without KiCad or the network:

* :class:`~ai_eda.ir.Stackup` refuses what it cannot mean (layer names and
  order, dielectric count, units, ranges), gives every number an id that
  ``lookup`` resolves, and the via span / board thickness from the stack.
* ``PCBDesign.stackup`` stays out of the design view while ``None``: an IR
  saved by the model before the field existed (``tests/data/ir_before_stackup.json``,
  saved at HEAD dbd384d, and the older ``ir_before_silkscreen.json``) keeps
  the hash that model computed; a stack is hashed as soon as it is set.
* The generic 2- / 4-layer stacks are template choices (assumption until
  confirmed, then the user's), 1.6 mm, planes only on the 4-layer inner
  layers; ``plane_zones`` insets the outline and the PCB compiler writes the
  In1.Cu / In2.Cu layers and the plane zones; the whole thing is
  deterministic.
* ``pcb_layers`` (alias ``layer_count``) reads a plain 2 or 4, defaults to 2,
  refuses anything else with the reason, never refuses a template (a board
  key) and is re-checked by ``design.inputs_vs_requirements``.
* A capability file's ``stackup`` block grounds number by number (lengths
  exactly in um / mm, the Dk frequency in Hz, Dk / Df as one bare number),
  is recorded only when every number grounds, carries no plane, and is
  re-verified by ``relocate_stackup`` (an edited number is ``value_mismatch``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eda.compilers import CompileContext
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.design import TEMPLATES, check_inputs_vs_requirements, unserved_requirements
from ai_eda.design.inputs import DEFAULT_LAYER_COUNT, LAYER_UNIT, canonical_key, read_layer_count
from ai_eda.design.stackup import GENERIC_NOTE, PLANE_EDGE_CLEARANCE_MM, board_layers, generic_stackup, plane_zones, stackup_choices, with_planes
from ai_eda.ir import (
    BoardOutline,
    CircuitIR,
    CopperRole,
    DielectricKind,
    PCBDesign,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    SourceRef,
    Stackup,
    StackupCopper,
    StackupDielectric,
    ValidationStatus as S,
    assumption,
    llm_generated,
    user_requirement,
)
from ai_eda.tools.calc.tline import line_geometry, microstrip
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.manufacturing import (
    CapabilityFileError,
    capability_source_result,
    ground_capability,
    load_capability_file,
    relocate_stackup,
)
from ai_eda.tools.manufacturing.capability_file import hz_from_token, um_from_token
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy
from ai_eda.security import ApprovalGate
from tests.test_parts_existence import synthetic_library
from tests.test_pcb_agent import parts_ir

DATA = Path(__file__).parent / "data"
#: an IR saved by the model at HEAD dbd384d (before PCBDesign.stackup existed) and the content hash that model computed for it
PRE_STACKUP_IR = DATA / "ir_before_stackup.json"
PRE_STACKUP_HASH = "sha256:6e617ab6b64947a86898e99795b8142a9112bef5b28f2362e3ecf75eef017487"
PRE_SILK_IR = DATA / "ir_before_silkscreen.json"
PRE_SILK_HASH = "sha256:93d701d3cbfc42e871c5eca2e3145d472b91cd1d4c0c82e26cf51698963eb536"
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="stated by hand")


def _um(v: float):
    return user_requirement(v, "um")


def _mm(v: float):
    return user_requirement(v, "mm")


def two_layer(**overrides) -> Stackup:
    data = dict(copper=[StackupCopper(name="F.Cu", thickness_um=_um(35.0)), StackupCopper(name="B.Cu", thickness_um=_um(35.0))],
                dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=_mm(1.53), er=user_requirement(4.5))], provenance=USER)
    data.update(overrides)
    return Stackup(**data)


# --------------------------------------------------------------------------- the IR model


def test_stackup_model_refuses_what_it_cannot_mean():
    s = two_layer()
    assert s.layer_count == 2 and s.signal_layers() == ["F.Cu", "B.Cu"] and s.plane_layers() == []
    assert s.board_thickness_mm() == pytest.approx(1.6) and s.span_mm("B.Cu", "F.Cu") == pytest.approx(1.6) and s.span_mm("F.Cu", "F.Cu") == pytest.approx(0.035)
    c = s.copper
    d = s.dielectrics[0]
    for kwargs, match in (
        (dict(copper=[c[0]], dielectrics=[]), "at least two copper layers"),
        (dict(copper=[c[1], c[0]]), "top to bottom"),
        (dict(copper=[c[0], StackupCopper(name="In1.Cu", thickness_um=_um(35.0)), c[1]]), "need 2 dielectric"),
        (dict(copper=[c[0], StackupCopper(name="In2.Cu", thickness_um=_um(35.0)), c[1]], dielectrics=[d, d]), "top to bottom"),
        (dict(copper=[StackupCopper(name="F.Cu", thickness_um=_mm(0.035)), c[1]]), "must carry unit 'um'"),
        (dict(copper=[StackupCopper(name="F.Cu", thickness_um=_um(0.0)), c[1]]), "must be > 0"),
        (dict(dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=user_requirement(1.53), er=user_requirement(4.5))]), "must carry unit 'mm'"),
        (dict(dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=_mm(1.53), er=user_requirement(0.9))]), "er must be >= 1"),
        (dict(dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=_mm(1.53), er=user_requirement(4.5, "F/m"))]), "no unit"),
        (dict(dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=_mm(1.53), er=user_requirement(4.5), loss_tangent=user_requirement(1.2))]), "loss_tangent must be < 1"),
        (dict(dielectrics=[StackupDielectric(kind=DielectricKind.CORE, thickness_mm=_mm(1.53), er=user_requirement(4.5), er_frequency_hz=user_requirement(1e6, "MHz"))]), "unit 'Hz'"),
        (dict(copper=[StackupCopper(name="F.Cu", thickness_um=_um(35.0), plane_net=user_requirement(" ")), c[1]]), "plane_net must name a net"),
    ):
        with pytest.raises(ValidationError, match=match):
            two_layer(**kwargs)
    with pytest.raises(ValueError, match="not in the stackup"):
        s.index("In1.Cu")


def test_every_stack_number_has_an_id_the_lookup_resolves():
    s = generic_stackup(4, "fixture", confirmed=True)
    items = dict(s.traced_items())
    assert list(items)[:3] == ["pcb.stackup.copper[F.Cu].thickness_um", "pcb.stackup.copper[In1.Cu].thickness_um", "pcb.stackup.copper[In1.Cu].plane_net"]
    assert "pcb.stackup.dielectrics[1].er_frequency_hz" in items and "pcb.stackup.dielectrics[2].loss_tangent" not in items
    assert all(s.lookup(k) is t for k, t in items.items())
    assert s.lookup("pcb.stackup.dielectrics[9].er") is None and s.lookup("parameters.x") is None
    assert s.copper[1].role is CopperRole.PLANE and s.copper[0].role is CopperRole.SIGNAL
    assert s.span_mm("F.Cu", "In1.Cu") == pytest.approx(0.035 + 0.2 + 0.0175)


def test_an_ir_saved_before_the_stackup_model_keeps_its_hash_and_round_trips(tmp_path: Path):
    raw = json.loads(PRE_STACKUP_IR.read_text(encoding="utf-8"))
    assert "stackup" not in raw["pcb"] and raw["pcb"]["silkscreen"] and raw["pcb"]["zones"]  # really written by the older model, with content
    ir = CircuitIR.load(PRE_STACKUP_IR)
    assert ir.content_hash() == PRE_STACKUP_HASH and "stackup" not in ir.design_dict()["pcb"]
    assert CircuitIR.load(PRE_SILK_IR).content_hash() == PRE_SILK_HASH  # the older fixture too
    ir.save(tmp_path / "ir.json")
    again = json.loads((tmp_path / "ir.json").read_text(encoding="utf-8"))
    assert again["pcb"]["stackup"] is None and CircuitIR.load(tmp_path / "ir.json").content_hash() == PRE_STACKUP_HASH
    # a stack is design content as soon as it is there
    ir.pcb.stackup = generic_stackup(2, "fixture", confirmed=True)
    changed = ir.content_hash()
    assert changed != PRE_STACKUP_HASH and ir.design_dict()["pcb"]["stackup"]["copper"][0]["name"] == "F.Cu"
    ir.save(tmp_path / "ir2.json")
    back = CircuitIR.load(tmp_path / "ir2.json")
    assert back.pcb.stackup == ir.pcb.stackup and back.content_hash() == changed
    # the stack's wall clock is not design content: the same stack built again hashes the same
    ir.pcb.stackup = generic_stackup(2, "fixture", confirmed=True)
    assert ir.content_hash() == changed
    ir.pcb.stackup = None
    assert ir.content_hash() == PRE_STACKUP_HASH


# --------------------------------------------------------------------------- the generic stacks and the plane zones


def test_generic_stacks_are_choices_of_1_6_mm_with_planes_only_on_four_layers():
    two, four = generic_stackup(2, "divider", confirmed=False), generic_stackup(4, "divider", confirmed=True, ground_net="GND", power_net="+3V3")
    assert two.board_thickness_mm() == pytest.approx(1.6) and four.board_thickness_mm() == pytest.approx(1.6)
    assert two.plane_layers() == [] and [(c.name, c.plane_net.value) for c in four.plane_layers()] == [("In1.Cu", "GND"), ("In2.Cu", "+3V3")]
    assert [d.kind for d in four.dielectrics] == [DielectricKind.PREPREG, DielectricKind.CORE, DielectricKind.PREPREG]
    # unconfirmed: every number is an assumption; confirmed: the user's, with the template and its version in the note
    assert all(t.provenance.kind is ProvenanceKind.ASSUMPTION for _, t in two.traced_items()) and two.provenance.kind is ProvenanceKind.ASSUMPTION
    assert all(t.provenance.kind is ProvenanceKind.USER_REQUIREMENT for _, t in four.traced_items())
    note = four.dielectrics[0].er.provenance.note
    assert "design choice confirmed by user; template divider v" in note and GENERIC_NOTE in note and "er 4.5 @ 1 MHz" in note
    assert four.copper[1].plane_net.provenance.note.endswith("the planes are zones KiCad fills)")
    with pytest.raises(ValueError, match="no generic stack for 6 layers"):
        generic_stackup(6, "divider", confirmed=False)
    # deterministic: the same arguments give the same stack (design view)
    again = generic_stackup(4, "divider", confirmed=True, ground_net="GND", power_net="+3V3")
    assert again.model_dump(mode="json", context={"view": "design"}) == four.model_dump(mode="json", context={"view": "design"})
    # the outer layers of the 4-layer stack are microstrips over the planes; 2-layer has no reference
    assert line_geometry(four, "F.Cu")[0].reference_net == "GND" and line_geometry(two, "F.Cu")[0] is None
    assert microstrip(0.35, float(four.dielectrics[0].thickness_mm.value), float(four.copper[0].thickness_um.value), float(four.dielectrics[0].er.value)).z0_ohm == pytest.approx(49.71, abs=0.01)


def test_plane_zones_inset_the_outline_and_board_layers_name_the_planes():
    four = generic_stackup(4, "t", confirmed=True)
    outline = BoardOutline(width_mm=88.0, height_mm=60.0, origin_x_mm=10.0, origin_y_mm=5.0)
    zones = plane_zones(four, outline)
    assert [(z.net, z.layer) for z in zones] == [("GND", "In1.Cu"), ("+5V", "In2.Cu")]
    assert zones[0].polygon == [(10.5, 5.5), (97.5, 5.5), (97.5, 64.5), (10.5, 64.5)] and PLANE_EDGE_CLEARANCE_MM == 0.5
    p = zones[1].provenance
    assert p.kind is ProvenanceKind.DERIVED and p.tool == "design.stackup" and p.derived_from == ["pcb.stackup.copper[In2.Cu].plane_net", "pcb.outline"]
    traced = plane_zones(four, outline, user_requirement(1.0, "mm", note="keep 1 mm"))
    assert traced[0].polygon[0] == (11.0, 6.0) and "(user_requirement: keep 1 mm)" in traced[0].provenance.note
    assert plane_zones(generic_stackup(2, "t", confirmed=True), outline) == []
    for bad in (-0.1, float("nan"), 30.0):
        with pytest.raises(ValueError):
            plane_zones(four, outline, bad)
    assert [(layer.name, layer.kind) for layer in board_layers(four)] == [("F.Cu", "signal"), ("In1.Cu", "power"), ("In2.Cu", "power"), ("B.Cu", "signal")]


def test_with_planes_assigns_nets_to_a_grounded_stack():
    bare = Stackup(
        copper=[StackupCopper(name=n, thickness_um=_um(t)) for n, t in (("F.Cu", 35.0), ("In1.Cu", 17.5), ("In2.Cu", 17.5), ("B.Cu", 35.0))],
        dielectrics=[StackupDielectric(kind=k, thickness_mm=_mm(t), er=user_requirement(4.4)) for k, t in ((DielectricKind.PREPREG, 0.21), (DielectricKind.CORE, 1.065), (DielectricKind.PREPREG, 0.21))],
        provenance=USER,
    )
    assert line_geometry(bare, "F.Cu")[0] is None  # a fab stack has no plane until the design says so
    planed = with_planes(bare, {"In1.Cu": user_requirement("GND"), "In2.Cu": user_requirement("+5V")})
    assert [c.role for c in planed.copper] == [CopperRole.SIGNAL, CopperRole.PLANE, CopperRole.PLANE, CopperRole.SIGNAL]
    assert planed.dielectrics == bare.dielectrics and line_geometry(planed, "B.Cu")[0].reference_net == "+5V"
    with pytest.raises(ValueError, match="not in the stackup"):
        with_planes(bare, {"In3.Cu": user_requirement("GND")})


def test_the_pcb_compiler_writes_the_inner_layers_and_the_plane_zones(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ir = parts_ir(tmp_path, lib, ("R1", "R2"))
    four = generic_stackup(4, "t", confirmed=True, ground_net="N0", power_net="IN")
    outline = BoardOutline(width_mm=30.0, height_mm=20.0)
    ir.pcb = PCBDesign(outline=outline, layers=board_layers(four), stackup=four, zones=plane_zones(four, outline),
                       placements=[Placement(component_ref="R1", x_mm=10.0, y_mm=10.0, provenance=USER), Placement(component_ref="R2", x_mm=20.0, y_mm=10.0, provenance=USER)])
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib})
    ref = PCBCompiler().compile(ir, ctx)
    tree = sexpr.parse_file(Path(ref.path))
    layers = sexpr.find(tree, "layers")
    copper = [(sexpr.args(entry)[0], sexpr.args(entry)[1]) for entry in layers[1:] if isinstance(entry, list) and len(sexpr.args(entry)) >= 2 and str(sexpr.args(entry)[0]).endswith(".Cu")]
    assert [name for name, _ in copper] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    zones = sexpr.find_all(tree, "zone")
    assert [(sexpr.get(z, "net"), sexpr.get(z, "layer")) for z in zones] == [("N0", "In1.Cu"), ("IN", "In2.Cu")]
    first = Path(ref.path).read_bytes()
    ref2 = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out2", tools={"kicad_library": lib}))
    assert Path(ref2.path).read_bytes() == first  # deterministic


def test_stackup_choices_show_the_default_count_and_the_planes():
    from ai_eda.design.inputs import LayerCountInput

    rows = [c.text() for c in stackup_choices(LayerCountInput(value=DEFAULT_LAYER_COUNT))]
    assert rows[0].startswith("pcb_layers = 2 - board layer count: the default 2 layers") and "a generic value, not a fab's" in rows[1]
    stated = Requirement(id="req.pcb_layers", key="pcb_layers", text="pcb_layers: 4", kind=RequirementKind.EXPLICIT, value=user_requirement("4"), category="electrical")
    rows = [c.text() for c in stackup_choices(LayerCountInput(value=4, requirement=stated), power_net="+3V3")]
    assert not any(r.startswith("pcb_layers") for r in rows)  # stated by the user: an input, not a choice
    assert any("In2.Cu = +3V3 plane" in r for r in rows) and any(r.startswith("stackup.plane_edge_clearance = 0.5 mm") for r in rows)


# --------------------------------------------------------------------------- the pcb_layers input


def _req(key: str, value, prov: str = "user") -> Requirement:
    traced = user_requirement(value) if prov == "user" else llm_generated(value, model="m")
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value}", kind=RequirementKind.EXPLICIT, value=traced, category="electrical")


def test_pcb_layers_reads_a_plain_count_defaults_to_two_and_refuses_the_rest():
    ir = CircuitIR(project=ProjectMeta(id="l", name="l"))
    got, why = read_layer_count(ir)
    assert why is None and got.value == DEFAULT_LAYER_COUNT == 2 and got.is_default and got.traced is None
    for value, expect in (("4", 4), (4, 4), (" 2 ", 2)):
        ir.requirements.requirements = [_req("pcb_layers", value)]
        got, why = read_layer_count(ir)
        assert why is None and got.value == expect and not got.is_default
        assert got.traced.unit == LAYER_UNIT and got.traced.provenance.derived_from == ["req.pcb_layers"] and got.traced.provenance.note.startswith("parsed from req.pcb_layers")
    ir.requirements.requirements = [_req("layer_count", "4")]
    assert read_layer_count(ir)[0].value == 4 and canonical_key("layer_count") == "pcb_layers"
    for reqs, expect in (
        ([_req("pcb_layers", "3")], "not one of the stackups"),
        ([_req("pcb_layers", "4 layers")], "not one plain layer count"),
        ([_req("pcb_layers", True)], "not a layer count"),
        ([_req("pcb_layers", "0")], "not a positive layer count"),
        ([_req("pcb_layers", "4", prov="llm")], "not yet the user's"),
        ([_req("pcb_layers", "2"), _req("layer_count", "4")], "ambiguous"),
    ):
        ir.requirements.requirements = reqs
        got, why = read_layer_count(ir)
        assert got is None and expect in why, why


def test_pcb_layers_never_refuses_a_template_and_is_rechecked_like_any_input():
    ir = CircuitIR(project=ProjectMeta(id="l", name="l"))
    ir.requirements.requirements = [_req("pcb_layers", "4"), _req("layer_count", "4")]
    for template in TEMPLATES:
        assert unserved_requirements(ir, template) == []  # a board key, served by every template through the stack
    got, _ = read_layer_count(ir)
    ir.parameters["pcb_layers"] = got.traced
    res = check_inputs_vs_requirements(ir)
    assert res is not None and res.status is S.PASS
    ir.requirements.requirements[0] = _req("pcb_layers", "2")
    res = check_inputs_vs_requirements(ir)
    assert res.status is S.FAIL and "pcb_layers" in res.message


# --------------------------------------------------------------------------- the capability file's stackup block


STACKUP_HTML = """<!DOCTYPE html><html><head><title>Example Fab 4-layer stackup</title></head><body>
<h1>4-layer impedance stackup JLC-like 7628</h1>
<table>
<tr><td>Outer layer copper</td><td>35um</td></tr>
<tr><td>Inner layer copper</td><td>0.0175 mm</td></tr>
<tr><td>Prepreg thickness</td><td>0.2104 mm</td></tr>
<tr><td>Prepreg 7628 sheet</td><td>0.2104 mm</td></tr>
<tr><td>Prepreg Dk</td><td>4.4</td></tr>
<tr><td>Core thickness</td><td>1.065 mm</td></tr>
<tr><td>Core Dk</td><td>4.6</td></tr>
<tr><td>Dk measured at</td><td>1 GHz</td></tr>
<tr><td>Loss tangent</td><td>0.02</td></tr>
<tr><td>Solder mask thickness</td><td>10 um</td></tr>
<tr><td>Solder mask Dk</td><td>3.8</td></tr>
<tr><td>FR-4 Dk 4.5 @ 1 MHz</td><td>typical 4.45</td></tr>
</table></body></html>
"""


def _q(value, unit, quote):
    return {"value": value, "unit": unit, "page": 1, "quote": quote}


def _stack_block() -> dict:
    prepreg = {"kind": "prepreg", "thickness": _q(0.2104, "mm", "Prepreg thickness 0.2104 mm"), "er": _q(4.4, None, "Prepreg Dk 4.4"),
               "er_frequency": _q(1, "GHz", "1 GHz"), "loss_tangent": _q(0.02, None, "Loss tangent 0.02")}
    return {
        "copper": [{"layer": "F.Cu", "thickness": _q(35, "um", "Outer layer copper 35um")}, {"layer": "In1.Cu", "thickness": _q(0.0175, "mm", "Inner layer copper 0.0175 mm")},
                   {"layer": "In2.Cu", "thickness": _q(0.0175, "mm", "Inner layer copper 0.0175 mm")}, {"layer": "B.Cu", "thickness": _q(35, "um", "Outer layer copper 35um")}],
        "dielectrics": [prepreg, {"kind": "core", "thickness": _q(1.065, "mm", "Core thickness 1.065 mm"), "er": _q(4.6, None, "Core Dk 4.6")}, copy.deepcopy(prepreg)],
        "solder_mask": {"thickness": _q(10, "um", "Solder mask thickness 10 um"), "er": _q(3.8, None, "Solder mask Dk 3.8")},
    }


def _file(tmp_path: Path, stackup: dict, limits: list | None = None) -> Path:
    data = {"fab": "Example Fab", "source": {"file": "stack.html", "retrieved_at": "2026-09-27", "title": "Example Fab 4-layer stackup", "authority": "Example Fab"},
            "limits": limits or [], "stackup": stackup}
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _doc(tmp_path: Path):
    page = tmp_path / "stack.html"
    page.write_text(STACKUP_HTML, encoding="utf-8")
    archive = DocumentArchive(tmp_path / "sources", NetworkPolicy(approved=False, gate=ApprovalGate()))
    return archive, archive.add_file(page, title="Example Fab 4-layer stackup", retrieved_at="2026-09-27", authority="Example Fab")


def test_a_stackup_block_grounds_number_by_number_and_is_reverified(tmp_path: Path):
    archive, doc = _doc(tmp_path)
    f = load_capability_file(_file(tmp_path, _stack_block()))  # a stackup without limits is a valid file
    assert f.limits == [] and f.describe()["stackup"] is True
    g = ground_capability(doc, f)
    assert g.rejected == [] and g.stackup_numbers == g.stackup_grounded == 16
    s = g.stackup
    assert s is not None and s.copper_names() == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"] and s.plane_layers() == []  # the page never assigns a net
    assert [c.thickness_um.value for c in s.copper] == [35.0, 17.5, 17.5, 35.0] and [c.thickness_um.unit for c in s.copper] == ["um"] * 4  # exact, from the page's tokens
    d0 = s.dielectrics[0]
    assert (d0.thickness_mm.value, d0.er.value, d0.er_frequency_hz.value, d0.loss_tangent.value) == (0.2104, 4.4, 1e9, 0.02)
    assert (d0.er.unit, d0.er_frequency_hz.unit, d0.loss_tangent.unit) == (None, "Hz", None)
    assert s.dielectrics[1].er_frequency_hz is None and s.solder_mask.thickness_um.value == 10.0
    assert all(t.provenance.kind is ProvenanceKind.AUTHORITATIVE and t.provenance.source.section == "page 1" for _, t in s.traced_items())
    assert s.provenance.kind is ProvenanceKind.AUTHORITATIVE and "no plane layer" in s.provenance.note
    assert {r["key"] for r in g.rows} >= {"pcb.stackup.dielectrics[0].er", "pcb.stackup.solder_mask.er"}
    res = capability_source_result(f, doc, g)
    # grounded, but no agent writes a fab stack into ir.pcb.stackup yet: reported as not recorded, and it never makes the PASS
    assert res.status is S.NOT_VERIFIED and "stackup: 16 of 16 number(s) grounded, a complete stack, not recorded in the IR" in res.message
    assert res.details["stackup"] == {"numbers": 16, "grounded": 16, "complete": True, "recorded": False}
    # re-verification re-reads every number on the archived page
    checks = relocate_stackup(s, archive)
    assert len(checks) == 16 and {c.status for c in checks} == {"ok"}
    edited = s.model_copy(deep=True)
    edited.dielectrics[0].er = edited.dielectrics[0].er.model_copy(update={"value": 4.2})
    edited.copper[0].thickness_um = edited.copper[0].thickness_um.model_copy(update={"value": 18.0})
    bad = {c.key: c for c in relocate_stackup(edited, archive) if c.status != "ok"}
    assert set(bad) == {"pcb.stackup.dielectrics[0].er", "pcb.stackup.copper[F.Cu].thickness_um"} and {c.status for c in bad.values()} == {"value_mismatch"}
    assert "re-reads as 4.4" in bad["pcb.stackup.dielectrics[0].er"].reason and "re-reads as 35 um" in bad["pcb.stackup.copper[F.Cu].thickness_um"].reason
    # a Dk quote may name the material and the frequency: FR-4 and 1 MHz are not the bare number
    block = _stack_block()
    block["dielectrics"][1]["er"] = _q(4.5, None, "FR-4 Dk 4.5 @ 1 MHz")
    g2 = ground_capability(doc, load_capability_file(_file(tmp_path, block)))
    assert g2.rejected == [] and g2.stackup.dielectrics[1].er.value == 4.5
    assert relocate_stackup(None, archive) == [] and relocate_stackup(generic_stackup(4, "t", confirmed=True), archive) == []  # choices are not page claims
    # with planes assigned by the design, the grounded stack gives the outer layers a microstrip geometry
    planed = with_planes(s, {"In1.Cu": user_requirement("GND"), "In2.Cu": user_requirement("+5V")})
    geo, why = line_geometry(planed, "F.Cu")
    assert why is None and (geo.h.value, geo.er.value) == (0.2104, 4.4) and any("solder mask" in n for n in geo.notes)


def test_a_stackup_number_that_does_not_ground_leaves_no_stack(tmp_path: Path):
    _archive, doc = _doc(tmp_path)
    cases = [
        (lambda b: b["dielectrics"][0]["er"].update(value=4.5, quote="FR-4 Dk 4.5 @ 1 MHz typical 4.45"), "pcb.stackup.dielectrics[0].er", "exactly one bare number"),
        (lambda b: b["dielectrics"][0]["thickness"].__setitem__("quote", "Prepreg 7628 sheet 0.2104 mm"), "pcb.stackup.dielectrics[0].thickness_mm", "other numbers besides"),
        (lambda b: b["dielectrics"][0]["er"].__setitem__("unit", "F/m"), "pcb.stackup.dielectrics[0].er", "has no unit"),
        (lambda b: b["dielectrics"][1]["er"].update(value=4.5, quote="Core Dk 4.6"), "pcb.stackup.dielectrics[1].er", "number mismatch"),
        (lambda b: b["dielectrics"][1]["er"].update(value=4, quote="Core Dk 4"), "pcb.stackup.dielectrics[1].er", "longer number"),
        (lambda b: b["copper"][0]["thickness"].update(value=1, unit="oz"), "pcb.stackup.copper[F.Cu].thickness_um", "not a length unit"),
        (lambda b: b["copper"][0]["thickness"].update(value=36), "pcb.stackup.copper[F.Cu].thickness_um", "number mismatch"),
        (lambda b: b["dielectrics"][0]["er_frequency"].update(value=1, unit="MHz"), "pcb.stackup.dielectrics[0].er_frequency_hz", "mismatch"),
        (lambda b: b["dielectrics"][1]["thickness"].update(quote="Core thickness 1.066 mm"), "pcb.stackup.dielectrics[1].thickness_mm", "not found verbatim"),
        (lambda b: b["dielectrics"][1]["thickness"].update(page=2), "pcb.stackup.dielectrics[1].thickness_mm", "page 2 does not exist"),
    ]
    for mutate, key, expect in cases:
        block = _stack_block()
        mutate(block)
        f = load_capability_file(_file(tmp_path, block))
        g = ground_capability(doc, f)
        reasons = dict(g.rejected)
        assert key in reasons and expect in reasons[key], (key, reasons)
        assert g.stackup is None and g.stackup_grounded == 15 and "a partial stack is no stack" in reasons["stackup"]
        res = capability_source_result(f, doc, g)
        assert res.status is S.NOT_VERIFIED and "stackup: 15 of 16 number(s) grounded, no stack (a partial stack is no stack), not recorded in the IR" in res.message


def test_the_stackup_block_schema_is_strict(tmp_path: Path):
    def bad(mutate, expect: str):
        block = _stack_block()
        mutate(block)
        with pytest.raises(CapabilityFileError) as e:
            load_capability_file(_file(tmp_path, block))
        assert expect in str(e.value), str(e.value)

    bad(lambda b: b["copper"].reverse(), "top to bottom")
    bad(lambda b: b["dielectrics"].pop(), "need 3 dielectric")
    bad(lambda b: b["dielectrics"][0].__setitem__("dk", 4.4), "dk")  # unknown key: refused, never mapped
    bad(lambda b: b["dielectrics"][0].__setitem__("kind", "glass"), "kind")
    bad(lambda b: b["dielectrics"][0]["er"].__setitem__("value", "4.4"), "value")  # text is not a number
    bad(lambda b: b["copper"][0].__setitem__("plane_net", "GND"), "plane_net")  # the page never assigns a plane
    data = json.loads(_file(tmp_path, _stack_block()).read_text(encoding="utf-8"))
    data.pop("stackup")
    (tmp_path / "empty.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CapabilityFileError, match="at least one limit"):
        load_capability_file(tmp_path / "empty.json")
    (tmp_path / "inf.json").write_text(_file(tmp_path, _stack_block()).read_text(encoding="utf-8").replace("0.2104", "1e400", 1), encoding="utf-8")
    with pytest.raises(CapabilityFileError):
        load_capability_file(tmp_path / "inf.json")


def test_exact_token_conversions():
    assert um_from_token("35um") == 35.0 and um_from_token("0.0175 mm") == 17.5 and um_from_token("35 µm") == 35.0
    assert hz_from_token("1 GHz") == 1e9 and hz_from_token("1.2GHz") == 1.2e9 and hz_from_token("100 MHz") == 1e8
    for fn, bad in ((um_from_token, "1 oz"), (hz_from_token, "1 GB"), (hz_from_token, "35um")):
        with pytest.raises(ValueError):
            fn(bad)


def test_the_fab_capability_agent_reports_a_grounded_stack_as_not_recorded_and_leaves_the_designs_stack(tmp_path: Path):
    """No agent writes a fab stack into ir.pcb.stackup yet: the agent proposes the limits only, the IR keeps the template's
    generic stack, and mfg.capability_source says so instead of claiming the stack was recorded (and is no PASS on it)."""
    from ai_eda.agents import AgentContext, FabCapabilityAgent
    from ai_eda.ir import BoardOutline, CircuitIR, ManufacturingConstraints, PCBDesign, ProjectMeta
    from ai_eda.workflow import Orchestrator

    archive, _doc_ = _doc(tmp_path)
    f = load_capability_file(_file(tmp_path, _stack_block()))
    ir = CircuitIR(project=ProjectMeta(id="cap", name="cap", workdir=str(tmp_path)))
    gen = generic_stackup(4, "t", confirmed=True)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=20.0, height_mm=10.0), manufacturing=ManufacturingConstraints(), layers=board_layers(gen), stackup=gen)
    before = ir.pcb.stackup.model_dump()
    res = FabCapabilityAgent().run(ir, AgentContext(workdir=tmp_path, tools={"archive": archive, "fab_capability_file": f}))
    assert [p.target for p in res.proposals] == ["pcb.manufacturing"]
    (src,) = [v for v in res.validation if v.check_id == "mfg.capability_source"]
    assert src.status is S.NOT_VERIFIED and "not recorded in the IR" in src.message and src.details["stackup"]["recorded"] is False
    Orchestrator.apply_proposals(ir, res.proposals)
    assert ir.pcb.stackup.model_dump() == before
