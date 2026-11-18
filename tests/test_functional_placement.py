"""Functional placement constraints on library-backed synthetic geometry."""

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.keys import CONTROL_KEYS, PLACEMENT_POLICY_KEY
from ai_eda.errors import CompileError
from ai_eda.ir import (BoardOutline, CircuitIR, Component, Keepout, LibraryRef, Net, NetKind, PCBDesign,
                       Pin, PinAssociation, PinRef, ProjectMeta, Topology, Traced)
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center, pads_bbox
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.functional import (PLACER_ID, PLACER_VERSION, connection_distance,
                                               functional_placement, grow)
from ai_eda.tools.placement.grid import _disjoint
from ai_eda.tools.placement.outline import rectangles
from ai_eda.tools.routing.maze import RoutingParams, fanout_margin
from tests.test_core_ring import FIX, LIB, mcu_library
from tests.test_parts_existence import synthetic_library
from tests.test_pcb_agent import parts_ir


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "lib"
    mcu_library(root)
    pretty = root / "footprints" / "MountingHole.pretty"
    pretty.mkdir()
    (pretty / "MountingHole_3.2mm_M3.kicad_mod").write_text(
        '(footprint "MountingHole_3.2mm_M3" (layer "F.Cu") (attr exclude_from_pos_files exclude_from_bom) '
        '(pad "" np_thru_hole circle (at 0 0) (size 3.2 3.2) (drill 3.2) (layers "*.Cu" "*.Mask")))')
    symbols = root / "symbols"
    symbols.mkdir()
    (symbols / "Mechanical.kicad_sym").write_text(
        '(kicad_symbol_lib (version 20250114) (generator "kicad_symbol_editor") '
        '(symbol "MountingHole" (property "Reference" "H" (at 0 0 0)) '
        '(property "Value" "MountingHole" (at 0 0 0)) '
        '(symbol "MountingHole_0_1" (circle (center 0 0) (radius 1) (stroke (width 0) (type default)) (fill (type none))))))')
    return KicadLibrary(roots=[root])


def board(lib, *, clock=False, large_clock=False):
    ir = CircuitIR(project=ProjectMeta(id="functional", name="functional"))
    names = {"U1": "QFP64", "J1": "HDR1x04", "C1": "R0603"}
    if clock:
        names["Y1"] = "XTAL" if large_clock else "R0603"
    for ref, name in names.items():
        fp = lib.resolve_footprint(LibraryRef(library=LIB, name=name))
        pins = [Pin(number=n, name=n, provenance=FIX) for n in sorted({p.number for p in lib.load_footprint(fp).pads if p.number})]
        ir.components.append(Component(ref=ref, value=name, footprint=fp, pins=pins, provenance=FIX))
    for k in range(4):
        ir.nets.append(Net(name=f"IO{k}", pins=[PinRef(component_ref="U1", pin_number=str(k + 8)), PinRef(component_ref="J1", pin_number=str(k + 1))], provenance=FIX))
    ir.nets.append(Net(name="VCC", kind=NetKind.POWER, pins=[PinRef(component_ref="U1", pin_number="35"), PinRef(component_ref="C1", pin_number="1")], provenance=FIX))
    associations = [PinAssociation(terminal=PinRef(component_ref="C1", pin_number="1"), ic_pin=PinRef(component_ref="U1", pin_number="35"), role="decoupling", provenance=FIX)]
    if clock:
        for i in (1, 2):
            ir.nets.append(Net(name=f"OSC{i}", pins=[PinRef(component_ref="U1", pin_number=str(54 + i)), PinRef(component_ref="Y1", pin_number=str(i))], provenance=FIX))
            associations.append(PinAssociation(terminal=PinRef(component_ref="Y1", pin_number=str(i)), ic_pin=PinRef(component_ref="U1", pin_number=str(54 + i)), role="crystal", provenance=FIX))
    ir.topology = Topology(name="core", pin_associations=associations, provenance=FIX)
    return ir


def test_pin_side_pin_order_and_vertical_access(library):
    ir = board(library)
    plan = functional_placement(ir, library)
    ps = {p.component_ref: p for p in plan.placements}
    fp = {c.ref: library.load_footprint(c.footprint) for c in ir.components}
    j = footprint_bbox(ps["J1"], fp["J1"])
    assert j.x1 == pytest.approx(plan.outline.origin_x_mm + 2)
    assert ps["J1"].x_mm < ps["U1"].x_mm
    assert pad_center(ps["J1"], fp["J1"].pad("1"))[1] < pad_center(ps["J1"], fp["J1"].pad("4"))[1]
    assert any(k.id == "access_J1" and k.forbids == ["footprints"] for k in plan.keepouts)
    assert all(p.provenance.tool == PLACER_ID and p.provenance.tool_version == PLACER_VERSION for p in plan.placements)
    assert all("UNVERIFIED" in c.description for c in plan.components)
    assert len(plan.components) == 4
    assert all(library.load_footprint(c.footprint).pads[0].pad_type == "np_thru_hole" for c in plan.components)


def test_decoupler_and_crystal_assigned_pads(library):
    ir = board(library, clock=True)
    plan = functional_placement(ir, library)
    ps = {p.component_ref: p for p in plan.placements}
    fps = {c.ref: library.load_footprint(c.footprint) for c in ir.components}
    for a in ir.topology.pin_associations:
        d = connection_distance(ps[a.terminal.component_ref], fps[a.terminal.component_ref], a.terminal.pin_number,
                                ps[a.ic_pin.component_ref], fps[a.ic_pin.component_ref], a.ic_pin.pin_number)
        assert d <= (5 if a.role == "crystal" else 3)
    assert any("routed lengths" in n for n in plan.notes)


def test_fixed_outline_kept_and_determinism(library):
    ir = board(library, clock=True)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=70, height_mm=65, origin_x_mm=11.25, origin_y_mm=-7.5))
    before = ir.content_hash()
    a = functional_placement(ir, library)
    ir.components.reverse()
    ir.nets.reverse()
    ir.topology.pin_associations.reverse()
    b = functional_placement(ir, library)
    assert a.outline.model_dump() == ir.pcb.outline.model_dump()
    assert [p.model_dump() for p in a.placements] == [p.model_dump() for p in b.placements]
    assert a.keepouts == b.keepouts
    ir.components.reverse(); ir.nets.reverse(); ir.topology.pin_associations.reverse()
    assert ir.content_hash() == before


def test_infeasible_outline_and_pin_constraints_reported(library):
    ir = board(library)
    with pytest.raises(CompileError, match="fixed outline"):
        functional_placement(ir, library, outline=BoardOutline(width_mm=12, height_mm=12))
    ir = board(library, clock=True)
    # Two distant target pins cannot both reach a small crystal's terminals within 5 mm.
    a = ir.topology.pin_associations[-1]
    old = a.ic_pin
    a.ic_pin = PinRef(component_ref="U1", pin_number="20")
    n = next(n for n in ir.nets if old in n.pins)
    n.pins = [a.ic_pin if p == old else p for p in n.pins]
    plan = functional_placement(ir, library)
    assert any("constraint not met: Y1" in n and "> 5 mm" in n for n in plan.notes)
    p = next(p for p in plan.placements if p.component_ref == "Y1")
    assert any("constraint not met: Y1" in n for n in p.provenance.derived_from)


def test_router_fanout_is_a_hard_obstacle(library):
    ir = board(library)
    # Raise the reservation above 3 mm to force a named constraint conflict.
    fp = library.load_footprint(ir.components[0].footprint)
    from dataclasses import replace
    fp.pads = [replace(p, x=p.x * 0.6, y=p.y * 0.6, size_w=p.size_w * 0.6, size_h=p.size_h * 0.6) for p in fp.pads]
    original = library.load_footprint
    library.load_footprint = lambda ref: fp if ref.name == "QFP64" else original(ref)
    params = RoutingParams(escape_reach_mm=4)
    plan = functional_placement(ir, library, routing=params)
    ps = {p.component_ref: p for p in plan.placements}
    capacitor = original(ir.components[2].footprint)
    assert _disjoint(footprint_bbox(ps["C1"], capacitor), grow(pads_bbox(ps["U1"], fp), fanout_margin(params) - 1e-6))
    assert any("constraint not met: C1" in n and "FANOUT" in n for n in plan.notes)


def test_fallback_search_checks_tangential_positions(library):
    from ai_eda.tools.placement.functional import NORMAL, _support, at, centre, pin_side
    ir = board(library)
    fps = {c.ref: library.load_footprint(c.footprint) for c in ir.components}
    a = ir.topology.pin_associations[0]
    target = at("U1", fps["U1"], 0, 0)
    px, py = pad_center(target, fps["U1"].pad(a.ic_pin.pin_number))
    nx, ny = NORMAL[pin_side(fps["U1"], a.ic_pin.pin_number)]
    def room(p, placed):
        x, y = centre(footprint_bbox(p, fps["C1"]))
        normal = (x - px) * nx + (y - py) * ny
        tangent = -(x - px) * ny + (y - py) * nx
        return (normal >= 10 and abs(tangent) < 0.01) or (normal >= 1 and tangent >= 8)
    p = _support("C1", fps, [a], {"U1":target}, room)
    x, y = centre(footprint_bbox(p, fps["C1"]))
    assert -(x - px) * ny + (y - py) * nx >= 8
    assert connection_distance(p, fps["C1"], "1", target, fps["U1"], a.ic_pin.pin_number) < 9


def test_keepout_footprint_ban_is_honoured(library):
    ir = board(library)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=60, height_mm=60), keepouts=[Keepout(id="blocked", layers=["F.Cu"],
        rect=Traced(value=[0, 0, 60, 60], unit="mm", provenance=FIX), forbids=["footprints"], reason="reserved", provenance=FIX)])
    with pytest.raises(CompileError, match="constraints infeasible"):
        functional_placement(ir, library)


def test_outline_area_order_and_seed():
    candidates = list(rectangles(20, 10, max_side=12))
    keys = [(o.width_mm * o.height_mm, max(o.width_mm, o.height_mm) / min(o.width_mm, o.height_mm), o.width_mm, o.height_mm) for o in candidates]
    assert keys == sorted(keys)
    assert all(k[0] >= 60 for k in keys)
    assert list(rectangles(0, 0, fixed=BoardOutline(width_mm=12.5, height_mm=13.75)))[0].width_mm == 12.5


def test_policy_is_control_key_and_absent_legacy_is_byte_identical(tmp_path, monkeypatch):
    import ai_eda.agents.pcb as pcb
    lib = synthetic_library(tmp_path / "legacy")
    ir = parts_ir(tmp_path, lib)
    def fail(*a, **kw):
        raise AssertionError("legacy dispatch reached functional placement")
    monkeypatch.setattr(pcb, "functional_placement", fail)
    answers = {"pcb.routing": "skip", "pcb.silkscreen": "skip"}
    a = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers=answers))
    b = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={**answers, PLACEMENT_POLICY_KEY: "legacy"}))
    from ai_eda.ir.provenance import design_data
    assert design_data(a.proposals[0].payload) == design_data(b.proposals[0].payload)
    from ai_eda.compilers import CompileContext, PCBCompiler
    from ai_eda.tools.kicad import sexpr
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    assert sexpr.dumps(PCBCompiler().build(ir.model_copy(update={"pcb": a.proposals[0].payload}), ctx)).encode() == sexpr.dumps(PCBCompiler().build(ir.model_copy(update={"pcb": b.proposals[0].payload}), ctx)).encode()
    assert PLACEMENT_POLICY_KEY in CONTROL_KEYS


def test_agent_reports_constraint_failure_and_does_not_mutate(library, tmp_path):
    ir = board(library)
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=10, height_mm=10))
    before = ir.content_hash()
    result = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": library}, answers={PLACEMENT_POLICY_KEY: "functional", "pcb.routing": "skip"}))
    assert not result.proposals
    assert any("constraints infeasible" in n for n in result.notes)
    assert ir.content_hash() == before


def test_agent_proposes_holes_and_outline_together(library, tmp_path):
    from ai_eda.workflow import Orchestrator
    ir = board(library)
    before = ir.content_hash()
    result = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": library}, answers={PLACEMENT_POLICY_KEY: "functional", "pcb.routing": "skip", "pcb.silkscreen": "skip"}))
    assert ir.content_hash() == before
    assert len(result.proposals) == 5
    Orchestrator.apply_proposals(ir, result.proposals)
    assert len(ir.components) == len(ir.pcb.placements) == 7
    assert ir.pcb.keepouts


def test_known_library_mating_direction_faces_out(library):
    ir = board(library)
    original = library.load_footprint
    fp = original(ir.components[1].footprint)
    fp.name = "RightAngleHeader"
    fp.properties["mating_direction"] = "west"
    library.load_footprint = lambda ref: fp if ref.name == "HDR1x04" else original(ref)
    plan = functional_placement(ir, library)
    p = next(p for p in plan.placements if p.component_ref == "J1")
    assert p.rotation_deg == 0
    assert any("access=west" in entry for entry in p.provenance.derived_from)


def test_vertical_header_can_turn_without_inventing_a_cable_direction(library):
    from ai_eda.tools.kicad.library import BBox
    ir = board(library)
    original = library.load_footprint
    fp = original(ir.components[1].footprint)
    fp.courtyard = BBox(-1, -1, 1, 49)
    library.load_footprint = lambda ref: fp if ref.name == "HDR1x04" else original(ref)
    plan = functional_placement(ir, library, outline=BoardOutline(width_mm=160, height_mm=70))
    p = next(p for p in plan.placements if p.component_ref == "J1")
    assert p.rotation_deg in (90, 270)
    assert footprint_bbox(p, fp).x1 == pytest.approx(2)
    assert any(k.id == "access_J1" for k in plan.keepouts)


def test_absent_policy_keeps_core_ring_bytes(library, tmp_path, monkeypatch):
    import ai_eda.agents.pcb as pcb
    import json
    from ai_eda.ir.provenance import design_data
    ir = board(library)
    def fail(*args, **kwargs):
        raise AssertionError("legacy core dispatch reached functional placement")
    monkeypatch.setattr(pcb, "functional_placement", fail)
    answers = {"pcb.routing": "skip", "pcb.silkscreen": "skip"}
    outputs = []
    for policy in ({}, {PLACEMENT_POLICY_KEY: "legacy"}):
        result = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": library}, answers={**answers, **policy}))
        payload = next(p.payload for p in result.proposals if p.target == "pcb")
        assert all(p.provenance.tool == "placement.core_ring" for p in payload.placements)
        outputs.append(json.dumps(design_data(payload), sort_keys=True).encode())
    assert outputs[0] == outputs[1]


def test_small_board_omits_automatic_holes_and_their_reservations(tmp_path):
    from ai_eda.tools.placement.functional import HOLE_RULE
    lib = synthetic_library(tmp_path / "small")
    ir = parts_ir(tmp_path, lib)
    plan = functional_placement(ir, lib)
    assert plan.outline.width_mm * plan.outline.height_mm < 400
    assert not plan.components
    assert not any(k.id.startswith("functional_H") for k in plan.keepouts)
    assert f"{HOLE_RULE}; omitted" in plan.notes


@pytest.mark.parametrize("area, expected", [(399, 0), (400, 4), (401, 4)])
def test_mounting_hole_threshold_uses_outline_without_reservations(tmp_path, monkeypatch, area, expected):
    import ai_eda.tools.placement.functional as functional
    lib = synthetic_library(tmp_path / "threshold")
    ir = parts_ir(tmp_path, lib)
    def selected(*args):
        raise CompileError("automatic mounting selected")
    monkeypatch.setattr(functional, "mounting_components", selected)
    if expected:
        with pytest.raises(CompileError, match="automatic mounting selected"):
            functional_placement(ir, lib, outline=BoardOutline(width_mm=20, height_mm=area / 20))
        return
    plan = functional_placement(ir, lib, outline=BoardOutline(width_mm=20, height_mm=area / 20))
    assert len(plan.components) == expected


def test_mounting_holes_compile_as_board_only(library, tmp_path):
    from ai_eda.compilers import CompileContext, PCBCompiler, SchematicCompiler
    from ai_eda.tools.kicad import sexpr
    from ai_eda.workflow import Orchestrator
    ir = board(library)
    result = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": library},
        answers={PLACEMENT_POLICY_KEY: "functional", "pcb.routing": "skip", "pcb.silkscreen": "skip"}))
    Orchestrator.apply_proposals(ir, result.proposals)
    assert {c.ref for c in SchematicCompiler._checked_components(ir)} == {"U1", "C1", "J1"}
    ir.components = [c for c in ir.components if c.board_only]
    ir.pcb.placements = [p for p in ir.pcb.placements if p.component_ref.startswith("H")]
    ir.pcb.keepouts = [k for k in ir.pcb.keepouts if k.id.startswith("functional_H")]
    ir.nets = []
    node = PCBCompiler().build(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": library}))
    holes = [fp for fp in sexpr.find_all(node, "footprint") if str(fp[1]).startswith("MountingHole:")]
    assert len(holes) == 4
    assert all("board_only" in sexpr.find(fp, "attr") for fp in holes)
    assert all("exclude_from_bom" in sexpr.find(fp, "attr") and "exclude_from_pos_files" in sexpr.find(fp, "attr") for fp in holes)
    assert "board_only" not in board(library).components[0].model_dump()
