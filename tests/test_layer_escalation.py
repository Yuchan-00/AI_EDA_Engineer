"""The PCB agent's layer escalation (multilayer plan, phase P6): the owner's ladder 2 -> 4 -> 6 -> 8 -> 10 -> 12 when nets stay unrouted.

What is proved, on the synthetic fixture library of ``tests/test_routing.py``:

* real routing: a 2-layer board whose GND net crosses net A's only corridor
  (a B.Cu track keep-out leaves A no way under it) loses A at one
  negotiation iteration; at 4 layers GND is the In1.Cu plane, joined by micro
  vias, and A routes. The payload carries the 4-layer stack, its layers and
  plane zones, the copper is stamped routing.maze 0.9, the stack's provenance
  quotes the owner's rule and every count's record, the notes come in order,
  and two runs give the same bytes;
* the ladder's logic, with ``route_board`` replaced by canned results keyed by
  the copper layer count (the pattern of ``tests/test_si_pcb_agent.py``): the
  first count that routes every net is kept, an odd count is refused and never
  routed, ``layer_count_options`` and a material's own stacks bound the ladder,
  a stated count, a board's own stack and a static fence never escalate, a
  board without exactly one ground net says why, a part's sourced minimum and
  an area array's escape depth start the ladder higher;
* the design file's ``min_pcb_layers`` needs its source.

Every test is fast: one negotiation iteration on boards a few millimetres wide, or no routing at all.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

import ai_eda.agents.pcb as pcb_module
from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.pcb import ESCALATION_RULE, LADDER_SIGNALS, _area_array, _parts_minimum
from ai_eda.design.design_file import DesignFileError, load_design_file
from ai_eda.design.materials import MATERIAL_STACKS, MaterialDecision, material_stackup
from ai_eda.design.stackup import (
    ESCALATION_TOOL,
    ESCALATION_VERSION,
    STACKUP_TOOL,
    board_layers,
    escalation_description,
    generic_stackup,
)
from ai_eda.errors import CompileError
from ai_eda.ir import (
    CircuitIR,
    ManufacturingConstraints,
    Net,
    NetKind,
    PinRef,
    Provenance,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    SourceRef,
    Traced,
    assumption,
    user_requirement,
)
from ai_eda.ir.provenance import design_data
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.maze import Routing, RoutingParams
from ai_eda.validation.layout import CLEARANCE_CHECK, CONNECTIVITY_CHECK
from ai_eda.workflow import Orchestrator
from tests.test_design_file import RC, _write
from tests.test_circuit_templates import template_library
from tests.test_pcb_agent import _routing_checks
from tests.test_rf_floorplan import keepout
from tests.test_routing import NET_P, board_ir, fixture_library
from tests.test_si_checks import stacked

ONE = RoutingParams(max_iterations=1)
#: the owner's ladder as the kept note names it
LADDER = "2, 4, 6, 8, 10, 12"


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


@pytest.fixture(autouse=True)
def steady_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each routing pass reads the clock twice: a counter in 1.5 s steps makes every pass take 1.5 s, so the notes are reproducible."""
    monkeypatch.setattr(pcb_module, "_clock", itertools.count(0.0, 1.5).__next__)


def _ctx(tmp_path: Path, lib: KicadLibrary, **answers: str) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers=answers)


def _gnd_board(tmp_path: Path, lib: KicadLibrary) -> CircuitIR:
    """GND joins two F.Cu SMD pads at the left and right edges straight across A's only corridor (R1 at the top to R2 at the bottom); a
    B.Cu track keep-out leaves A no way under GND; +3V3 joins two through-hole pads out of the way. At one negotiation iteration A is
    ripped up and finds no route against GND's copper; at 4 layers GND is the In1.Cu plane (micro vias, no track) and A runs straight."""
    parts = [("C1", "SMD1", 1.0, 5.0), ("C2", "SMD1", 19.0, 5.0), ("R1", "SMD1", 10.0, 1.5), ("R2", "SMD1", 10.0, 8.5),
             ("P1", "PAD1", 3.0, 8.5), ("P2", "PAD1", 6.0, 8.5)]
    nets = {"GND": [("C1", "1"), ("C2", "1")], "A": [("R1", "1"), ("R2", "1")], "+3V3": [("P1", "1"), ("P2", "1")]}
    ir = board_ir(tmp_path, lib, parts, nets, (20.0, 10.0))
    ir.nets[0].kind = NetKind.GROUND
    ir.nets[2].kind = NetKind.POWER
    ir.pcb.keepouts = [keepout("KB", [0.0, 0.0, 20.0, 10.0], forbids=["tracks"], layers=["B.Cu"])]
    return ir


def _board_notes(notes: list[str]) -> list[str]:
    return [n for n in notes if not n.startswith("silkscreen")]


# --------------------------------------------------------------------------- real routing


def test_a_congested_two_layer_board_is_kept_at_four_layers_with_micro_plane_vias(tmp_path: Path, lib: KicadLibrary):
    ir = _gnd_board(tmp_path, lib)
    before = ir.content_hash()
    res = PCBAgent(routing=ONE).run(ir, _ctx(tmp_path, lib))
    assert ir.content_hash() == before and res.questions == []  # the agent proposes, never mutates, never asks
    [proposal] = res.proposals
    board = proposal.payload
    # the 4-layer escalation stack, its layers and its plane zones
    assert [(layer.name, layer.kind) for layer in board.layers] == [("F.Cu", "signal"), ("In1.Cu", "power"), ("In2.Cu", "power"), ("B.Cu", "signal")]
    assert [(c.name, c.plane_net.value if c.plane_net else None) for c in board.stackup.copper] == [
        ("F.Cu", None), ("In1.Cu", "GND"), ("In2.Cu", "+3V3"), ("B.Cu", None)]
    assert [(z.net, z.layer, z.polygon[0], z.provenance.tool) for z in board.zones] == [
        ("GND", "In1.Cu", (0.5, 0.5), STACKUP_TOOL), ("+3V3", "In2.Cu", (0.5, 0.5), STACKUP_TOOL)]
    # GND joins its plane by micro vias F.Cu-In1.Cu and has no track; A runs on F.Cu; everything is routing.maze 0.9
    assert [(v.net, v.kind, v.layers) for v in board.vias] == [("GND", "micro", ("F.Cu", "In1.Cu"))] * 2
    # GND's copper is its two pad-to-via stubs; +3V3's through-hole pads reach In2.Cu by their barrels
    assert {t.net for t in board.tracks} == {"A", "GND"} and len([t for t in board.tracks if t.net == "GND"]) == 2
    assert {t.layer for t in board.tracks} == {"F.Cu"}
    assert all(i.provenance.tool_version == "0.9" and i.provenance.derived_from[-1].startswith("stack:4-layer 1.6 mm FR-4;") for i in [*board.tracks, *board.vias])
    # the stack's provenance: the escalation tool, the owner's rule, every count's record, the stack
    prov = board.stackup.provenance
    two = "2 copper layers (signals F.Cu, B.Cu; planes none): routing.maze 0.4 left 1 of 3 net(s) unrouted (1 ripped up in negotiation) after 1 iteration(s)"
    four = "4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu +3V3): routing.maze 0.9 routed every net after 1 iteration(s)"
    assert (prov.kind, prov.tool, prov.tool_version) == (ProvenanceKind.DERIVED, ESCALATION_TOOL, ESCALATION_VERSION)
    assert prov.derived_from == ["pcb.placements", "layer_escalation:2,4,6,8,10,12"]
    assert prov.note == (f"layer count 4: chosen by the project owner's layer-escalation rule ({ESCALATION_RULE}) - {two}; {four}; "
                         f"{escalation_description(4, 'GND', '+3V3')}")
    assert all(c.thickness_um.provenance.kind is ProvenanceKind.ASSUMPTION for c in board.stackup.copper)  # numbers: the generic stack's choices
    # the notes: the routing description, one line per count with its time, the kept count, the planes - then the silkscreen
    notes = _board_notes(res.notes)
    assert notes[0] == "not placed: ir.pcb already has 6 placement(s); the agent never replaces a layout"
    assert notes[1].startswith("routing.maze 0.9: 3 net(s) routed on F.Cu/B.Cu, 3 track(s), 2 via(s), ") and notes[1].endswith(
        "; via types: through 0, micro 2; corners: 0 of 0 corner(s) of 90 degrees or sharper rounded into tangent arcs and 0 mitred into "
        "45-degree chamfers where no arc fits (routing.corners 0.2, the project owner's corner-style rule - a geometry convention, not an "
        "electrical requirement), 0 left as routed, 0 skipped; sidesteps: 0 of 0 sidestep(s) of the router (a short track between two "
        "right-angle corners) straightened before the corners were rounded, 0 left as routed")
    assert notes[2:] == [
        f"layer escalation: {two} (1.5 s wall clock)",
        f"layer escalation: {four} (1.5 s wall clock)",
        f"layer escalation: 4 layers kept - the first count of {LADDER} at which every net routes (the project owner's rule)",
        "plane zones: GND on In1.Cu, +3V3 on In2.Cu (outline inset 0.5 mm; KiCad fills them)",
    ]
    assert proposal.description.startswith(notes[1])
    assert proposal.rationale.startswith("every net maze-routed on F.Cu/B.Cu from the existing placements") and proposal.rationale.endswith(
        "DRC decides validity; layer count 4 by the project owner's layer-escalation rule (see ir.pcb.stackup provenance); " + pcb_module.SILK_RATIONALE)
    # the applied board's IR geometry: every net connected (GND through its plane fill), every clearance kept
    Orchestrator.apply_proposals(ir, res.proposals)
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(0.25, note="the router's own clearance as the limit"))
    checks = _routing_checks(ir, tmp_path, lib)
    assert checks[CLEARANCE_CHECK].status.value == "PASS", checks[CLEARANCE_CHECK].message
    rows = {row["net"]: row["status"] for row in checks[CONNECTIVITY_CHECK].details["nets"]}
    assert rows["A"] == "PASS" and rows["GND"] == "NOT_VERIFIED"


def test_two_escalated_runs_give_the_same_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def run(tmp: Path):
        monkeypatch.setattr(pcb_module, "_clock", itertools.count(0.0, 1.5).__next__)
        lib = fixture_library(tmp / "kicad")
        res = PCBAgent(routing=ONE).run(_gnd_board(tmp, lib), _ctx(tmp, lib))
        return json.dumps([design_data(p.payload) for p in res.proposals], sort_keys=True), res.notes

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert a == b and "layer escalation: 4 layers kept" in " | ".join(a[1])


def test_a_board_that_routes_is_routed_once_without_a_stack(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """With the default iteration cap the GND board routes at 2 layers (A takes the negotiated detour): no stack, no escalation line."""
    calls: list[object] = []
    real = pcb_module.route_board

    def spy(ir, library, params=None, **kw):
        calls.append(kw.get("stack"))
        return real(ir, library, params, **kw)

    monkeypatch.setattr(pcb_module, "route_board", spy)
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0), ("C1", "SMD1", 3.0, 7.0), ("C2", "SMD1", 9.0, 7.0)],
                  {"A": [("R1", "1"), ("R2", "1")], "GND": [("C1", "1"), ("C2", "1")]}, (12.0, 10.0))
    ir.nets[1].kind = NetKind.GROUND
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [None] and not any("layer escalation" in n for n in res.notes)
    assert res.proposals[0].payload.stackup is None and [layer.name for layer in res.proposals[0].payload.layers] == ["F.Cu", "B.Cu"]


# --------------------------------------------------------------------------- the ladder's logic (canned routing)


def _copper(ir: CircuitIR) -> int:
    return ir.pcb.stackup.layer_count if ir.pcb.stackup is not None else len(ir.pcb.layers)


def _canned(monkeypatch: pytest.MonkeyPatch, outcomes: dict[int, object]) -> list[tuple[int, str | None]]:
    """Replace the agent's ``route_board``: the copper layer count of the board it is given picks the outcome - ``True`` every net
    routed, ``False`` A left unrouted after being ripped up in negotiation, ``"fence"`` A left unrouted by a static fence, an
    exception raised. Returns the calls, ``(count, stack label or None)``."""
    calls: list[tuple[int, str | None]] = []

    def fake(ir, library, params=None, **kw):
        n, stack = _copper(ir), kw.get("stack")
        calls.append((n, stack.label if stack is not None else None))
        outcome = outcomes[n]
        if isinstance(outcome, Exception):
            raise outcome
        ok = outcome is True
        stats = {"routed_nets": 2 if ok else 1, "unrouted_nets": 0 if ok else 1, "skipped_nets": [], "net_length_mm": {}, "total_length_mm": 0.0,
                 "track_count": 0, "via_count": 0, "grid": (1, 1, 1), "raised": {}, "iterations": 3, "legal": ok, "history": [],
                 "dropped": ["A"] if outcome is False else [], "recovered": [], "net_order": ["A", "GND"]}
        if stack is not None:
            stats["stack"] = {"label": stack.label, "signal_layers": list(stack.signal), "plane_layers": dict(stack.planes), "stitch": {},
                              "via_types": [{"kind": v.kind, "top": v.top, "bottom": v.bottom, "count": 0} for v in stack.vias]}
        return Routing(params=params or RoutingParams(), unrouted={} if ok else {"A": "fixture: no legal route"}, stats=stats,
                       version="0.9" if stack is not None else "0.2")

    monkeypatch.setattr(pcb_module, "route_board", fake)
    return calls


def _plain(tmp_path: Path, lib: KicadLibrary, *, ground: int = 1) -> CircuitIR:
    """Nets A, GND and AGND on a bare 2-layer board (no stack); the first ``ground`` of GND / AGND are of kind ground."""
    parts = [("R1", "SMD1", 3.0, 3.0), ("R2", "SMD1", 9.0, 3.0), ("C1", "PAD1", 3.0, 7.0), ("C2", "PAD1", 9.0, 7.0), ("C3", "PAD1", 6.0, 10.0)]
    nets = {"A": [("R1", "1"), ("R2", "1")], "GND": [("C1", "1"), ("C2", "1")], "AGND": [("C3", "1"), ("C2", "1")]}
    ir = board_ir(tmp_path, lib, parts, nets, (12.0, 12.0))
    for net in ir.nets[1:1 + ground]:
        net.kind = NetKind.GROUND
    return ir


def _stated(ir: CircuitIR, layers: int) -> CircuitIR:
    """``ir`` with the user's confirmed ``pcb_layers`` requirement."""
    ir.requirements.requirements.append(Requirement(id="req.layers", key="pcb_layers", text=f"pcb_layers: {layers}", kind=RequirementKind.EXPLICIT,
                                                    value=user_requirement(layers), category="electrical"))
    return ir


def _escalation(notes: list[str]) -> list[str]:
    return [n for n in notes if n.startswith("layer escalation")]


ROUTED = "routing.maze 0.9 routed every net after 3 iteration(s) (1.5 s wall clock)"
LEFT = "left 1 of 2 net(s) unrouted (1 ripped up in negotiation) after 3 iteration(s) (1.5 s wall clock)"


def test_eight_layers_are_kept_when_four_and_six_fail(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: False, 4: False, 6: False, 8: True})
    res = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, "4-layer 1.6 mm FR-4"), (6, "6-layer 1.6 mm FR-4"), (8, "8-layer 1.6 mm FR-4")]
    assert _escalation(res.notes) == [
        f"layer escalation: 2 copper layers (signals F.Cu, B.Cu; planes none): routing.maze 0.2 {LEFT}",
        f"layer escalation: 4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu GND): routing.maze 0.9 {LEFT}",
        f"layer escalation: 6 copper layers (signals F.Cu, In2.Cu, In3.Cu, B.Cu; planes In1.Cu GND, In4.Cu GND): routing.maze 0.9 {LEFT}",
        f"layer escalation: 8 copper layers (signals F.Cu, In2.Cu, In3.Cu, In5.Cu, B.Cu; planes In1.Cu GND, In4.Cu GND, In6.Cu GND): {ROUTED}",
        f"layer escalation: 8 layers kept - the first count of {LADDER} at which every net routes (the project owner's rule)",
    ]
    board = res.proposals[0].payload
    assert [layer.name for layer in board.layers] == ["F.Cu", *(f"In{i}.Cu" for i in range(1, 7)), "B.Cu"]
    assert [(z.net, z.layer) for z in board.zones] == [("GND", "In1.Cu"), ("GND", "In4.Cu"), ("GND", "In6.Cu")]
    assert board.stackup.provenance.note.startswith("layer count 8: chosen by the project owner's layer-escalation rule")
    assert res.notes[1].startswith("routing.maze 0.9: 2 net(s) routed on F.Cu/In2.Cu/In3.Cu/In5.Cu/B.Cu, ")
    assert res.notes[1].endswith("; via types: through 0, micro 0, blind 0, buried 0")


def test_when_no_count_routes_every_net_the_two_layer_result_stands(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, dict.fromkeys((2, 4, 6, 8, 10, 12), False))
    ir = _plain(tmp_path, lib)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [2, 4, 6, 8, 10, 12]
    [proposal] = res.proposals  # the board was placed already: the existing board and its silkscreen, no copper, no stack
    assert proposal.payload.tracks == [] and proposal.payload.stackup is None and [layer.name for layer in proposal.payload.layers] == ["F.Cu", "B.Cu"]
    notes = _board_notes(res.notes)
    # today's notes first, then one line per count and the closing line
    assert notes[1].startswith("not applied: routing.maze 0.2 connected 1 of 2 net(s)") and notes[2] == "not routed: A: fixture: no legal route"
    assert len(notes[3:]) == 7 and all(n.startswith("layer escalation: ") for n in notes[3:])
    assert notes[-1] == "layer escalation: no count up to 12 routes every net; the 2-layer result stands"
    # with the partial answer the 2-layer routing applies, the escalation lines after the per-net reasons
    calls.clear()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{pcb_module.ROUTING_KEY: "partial"}))
    notes = _board_notes(res.notes)
    assert notes[2].startswith("partial: ") and notes[3] == "not routed: A: fixture: no legal route" and notes[-1].endswith("the 2-layer result stands")
    assert res.proposals[0].payload.stackup is None


def test_an_odd_count_is_refused_and_never_routed(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    """The owner once typed 5: were it on the ladder, it is refused with KiCad's reason and the ladder goes on."""
    monkeypatch.setattr(pcb_module, "LAYER_ESCALATION", (2, 4, 5, 8))
    calls = _canned(monkeypatch, {2: False, 4: False, 8: True})
    res = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [2, 4, 8]
    assert ("layer escalation: 5 copper layers: refused - KiCad builds only even copper layer counts (kicad-cli 10.0.6 refused a 5-layer board: "
            "'5 is not a valid layer number'; compilers/pcb.py refuses odd counts)") in res.notes
    assert "layer escalation: 8 layers kept - the first count of 2, 4, 5, 8 at which every net routes (the project owner's rule)" in res.notes


def test_layer_count_options_stop_the_ladder_at_four(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: False, 4: False})
    ir = _plain(tmp_path, lib)
    ir.pcb.manufacturing = ManufacturingConstraints(layer_count_options=assumption([2, 4], note="a fab that builds 2 and 4 layers"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [2, 4]
    lines = _escalation(res.notes)
    assert lines[2:6] == [f"layer escalation: {n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [2, 4]" for n in (6, 8, 10, 12)]
    assert lines[-1] == "layer escalation: no count up to 12 routes every net; the 2-layer result stands"


def test_a_rung_the_router_refuses_is_recorded_and_the_ladder_goes_on(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: False, 4: CompileError("cannot route: fixture refusal"), 6: True})
    res = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [2, 4, 6]
    assert "layer escalation: 4 copper layers: not routed - cannot route: fixture refusal" in res.notes
    assert res.proposals[0].payload.stackup.layer_count == 6
    assert "; 4 copper layers: not routed - cannot route: fixture refusal; " in res.proposals[0].payload.stackup.provenance.note


def test_a_stated_count_a_boards_own_stack_and_a_static_fence_never_escalate(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: False, 4: False})
    # a stated count: the user's choice stands; the note names the next count and how to reach it (no template builds this board's stack)
    res = PCBAgent().run(_stated(_plain(tmp_path, lib), 2), _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    assert _escalation(res.notes) == [
        "layer escalation: not attempted - req.layers states 2 copper layer(s), the user's choice, which is never escalated; the next count of the "
        "ladder is 4 layers, which this version reaches only by escalation (pcb_layers takes 2 or 4) - leave the count out "
        "(--answer leave_out=pcb_layers) and the PCB stage climbs the ladder"]
    # the IR's own stack (no template chose it): silent
    calls.clear()
    own = _plain(tmp_path, lib)
    stacked(own, 4)
    res = PCBAgent().run(own, _ctx(tmp_path, lib))
    assert calls == [(4, None)] and _escalation(res.notes) == []
    # a static fence (nothing ripped up in negotiation): silent
    calls.clear()
    fenced = _canned(monkeypatch, {2: "fence"})
    res = PCBAgent().run(_plain(tmp_path, lib), _ctx(tmp_path, lib))
    assert fenced == [(2, None)] and _escalation(res.notes) == []


@pytest.mark.parametrize(("ground", "names"), [(0, "none"), (2, "GND, AGND")])
def test_without_exactly_one_ground_net_the_note_says_why(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch, ground: int, names: str):
    calls = _canned(monkeypatch, {2: False})
    res = PCBAgent().run(_plain(tmp_path, lib, ground=ground), _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    assert _escalation(res.notes) == [f"layer escalation: not attempted - {ground} net(s) of kind ground ({names}); the plane layers need exactly one"]


# --------------------------------------------------------------------------- template boards and materials


def _template_board(tmp_path: Path, lib: KicadLibrary, template_id: str, stack) -> CircuitIR:
    """``_plain`` as template ``template_id`` built it: a part with the template's structural provenance, the template's stack."""
    ir = _plain(tmp_path, lib)
    ir.components[0].provenance = Provenance(kind=ProvenanceKind.DERIVED, tool=f"design.template.{template_id}", tool_version="0.3")
    ir.pcb.stackup = stack
    ir.pcb.layers = board_layers(stack)
    return ir


def test_a_template_board_on_its_default_count_climbs_the_ladder(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: False, 4: True})
    ir = _template_board(tmp_path, lib, "divider", generic_stackup(2, "divider", confirmed=True, ground_net="GND", power_net="+5V"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, "4-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup.layer_count == 4
    # stated by the user, the same board is not escalated; the next count is one the template builds, so the answer is named
    calls.clear()
    _stated(ir, 2)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)] and _escalation(res.notes) == [
        "layer escalation: not attempted - req.layers states 2 copper layer(s), the user's choice, which is never escalated; the next count of the "
        "ladder is 4 layers: answer --answer pcb_layers=4 to ask for it"]


def test_a_template_that_restricts_its_counts_is_not_escalated_past_them(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    from ai_eda.design.templates import all_templates

    template = next(t for t in all_templates() if t.layer_policy.allowed == (4,))
    calls = _canned(monkeypatch, {4: False})
    ir = _template_board(tmp_path, lib, template.id, generic_stackup(4, template.id, confirmed=True, ground_net="GND", power_net="GND"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [4]
    assert _escalation(res.notes) == [f"layer escalation: not attempted - template {template.id} builds only 4 layers: {template.layer_policy.reason.strip()}"]


def test_a_material_board_climbs_only_its_own_stacks(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    ro2, ro4 = MATERIAL_STACKS[("rogers_ro4350b", 2)], MATERIAL_STACKS[("rogers_ro4350b", 4)]
    decision = MaterialDecision(material="rogers_ro4350b", reason="carrier 2.4 GHz")
    calls = _canned(monkeypatch, {2: False, 4: False})
    ir = _template_board(tmp_path, lib, "divider", material_stackup(ro2, decision, "divider", confirmed=True, ground_net="GND", power_net="+5V"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, ro4.title)]  # RO4350B's own 4-layer stack, never an FR-4 one
    assert _escalation(res.notes)[-1] == ("layer escalation: no count up to 4 routes every net; the 2-layer result stands; past 4 the Rogers RO4350B "
                                          "board has no stack of its material and no FR-4 stack is swapped in")
    # at 4 layers it routes: the RO4350B stack is kept with its material record, through vias only (a micro via over 0.51 mm is too deep)
    calls = _canned(monkeypatch, {2: False, 4: True})
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    stack = res.proposals[0].payload.stackup
    assert stack.material.id == "rogers_ro4350b" and stack.material.reason == "carrier 2.4 GHz" and stack.layer_count == 4
    assert stack.provenance.derived_from == ["pcb.placements", "layer_escalation:2,4,6,8,10,12", "material:rogers_ro4350b"]
    assert "via structures through F.Cu-B.Cu - " not in stack.provenance.note and stack.provenance.note.endswith("via structures through F.Cu-B.Cu")
    assert stack.dielectrics[1].material == "Rogers RO4450F" and "chosen by the project owner's layer-escalation rule" in stack.dielectrics[1].er.provenance.note


def _material_req(key: str, value: str) -> Requirement:
    """The user's typed answer (``--answer key=value``): an explicit electrical requirement, as the requirement agent records it."""
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value}", kind=RequirementKind.EXPLICIT, value=user_requirement(value), category="electrical")


def _serving(stack, *ids: str):
    """``stack`` as ``add_board`` leaves it: its provenance names the requirements it was built from (``Stackup.served_requirements``)."""
    return stack.model_copy(update={"provenance": stack.provenance.model_copy(update={"derived_from": list(ids)})})


def test_an_escalated_stack_still_serves_the_requirements_the_templates_stack_served(tmp_path: Path, lib: KicadLibrary,
                                                                                       monkeypatch: pytest.MonkeyPatch):
    """Review finding: the climb replaced the stack's provenance and dropped req.pcb_material, so the reviewer turned a PASS into a FAIL."""
    from ai_eda.agents.circuit import _unserved

    ro2 = MATERIAL_STACKS[("rogers_ro4350b", 2)]
    decision = MaterialDecision(material="rogers_ro4350b", reason="chosen by you: req.pcb_material = rogers_ro4350b")
    stack = _serving(material_stackup(ro2, decision, "divider", confirmed=True, ground_net="GND", power_net="+5V"), "req.pcb_material")
    ir = _template_board(tmp_path, lib, "divider", stack)
    ir.requirements.requirements.append(_material_req("pcb_material", "rogers_ro4350b"))
    assert _unserved(ir) == []
    _canned(monkeypatch, {2: False, 4: True})
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    four = res.proposals[0].payload.stackup
    assert four.layer_count == 4 and four.served_requirements() == ["req.pcb_material"]
    assert four.provenance.derived_from == ["req.pcb_material", "pcb.placements", "layer_escalation:2,4,6,8,10,12", "material:rogers_ro4350b"]
    Orchestrator.apply_proposals(ir, res.proposals)
    assert _unserved(ir) == []
    # the FR-4 ladder too: a generic stack built from req.pcb_flexing = none keeps serving it at 4 layers
    ir = _template_board(tmp_path, lib, "divider", _serving(generic_stackup(2, "divider", confirmed=True, ground_net="GND", power_net="+5V"),
                                                             "req.pcb_flexing"))
    ir.requirements.requirements.append(_material_req("pcb_flexing", "none"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert res.proposals[0].payload.stackup.provenance.derived_from[:2] == ["req.pcb_flexing", "pcb.placements"]


def test_a_board_without_a_stack_never_climbs_fr4_stacks_against_its_stated_material(tmp_path: Path, lib: KicadLibrary,
                                                                                      monkeypatch: pytest.MonkeyPatch):
    """Review finding: a design-file / bare-IR board read no material, so a stated RO4350B board was routed and kept on the FR-4 stack."""
    calls = _canned(monkeypatch, {2: False, 4: True})
    for req, why in (
        (_material_req("pcb_material", "rogers_ro4350b"),
         "req.pcb_material: the board's base material is Rogers RO4350B (chosen by you: req.pcb_material = rogers_ro4350b)"),
        (_material_req("pcb_flexing", "static"),
         "req.pcb_flexing: the board's base material is DuPont Pyralux AP (AP9121R) (proposed by the material rule: req.pcb_flexing: "),
        (_material_req("power_dissipation", "3 W"),
         "req.power_dissipation: the material rule asks pcb_material first (pcb_material: no document read in this version says "),
        (_material_req("pcb_material", "rogers"), "the material requirements are unusable (pcb_material: req.pcb_material: 'rogers' is not exactly one of "),
    ):
        calls.clear()
        ir = _plain(tmp_path, lib)
        ir.requirements.requirements.append(req)
        res = PCBAgent().run(ir, _ctx(tmp_path, lib))
        assert calls == [(2, None)], req.key
        [line] = _escalation(res.notes)
        assert line.startswith(f"layer escalation: not attempted - {why}"), line
        assert line.endswith("; this board has no stack and no stage builds a material stack for a board no template made, so no FR-4 stack is "
                             "swapped in"), line
        assert res.proposals[0].payload.stackup is None
    # the user's FR-4 is the escalation stacks' material: the ladder climbs
    calls.clear()
    ir = _plain(tmp_path, lib)
    ir.requirements.requirements.append(_material_req("pcb_material", "fr4"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, "4-layer 1.6 mm FR-4")] and res.proposals[0].payload.stackup.layer_count == 4


def test_a_parts_minimum_no_offered_count_meets_still_lets_the_board_climb_from_its_own_count(tmp_path: Path, lib: KicadLibrary,
                                                                                             monkeypatch: pytest.MonkeyPatch):
    """Review finding: no count at or above the parts' minimum was offered, the board was routed at its own count and never climbed to the
    count the fab offers (the rungs below the minimum were skipped and the congestion trigger never ran)."""
    calls = _canned(monkeypatch, {2: False, 4: True})
    ir = _plain(tmp_path, lib)
    ir.components[0].min_pcb_layers = _sourced(6)
    ir.pcb.manufacturing = ManufacturingConstraints(layer_count_options=assumption([2, 4], note="a fab that builds 2 and 4 layers"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, "4-layer 1.6 mm FR-4")]
    stack = res.proposals[0].payload.stackup
    assert stack.layer_count == 4
    below = "skipped - below the parts' minimum of 6 copper layers"
    lines = _escalation(res.notes)
    assert lines[:2] == [f"layer escalation: 2 copper layers: {below}", f"layer escalation: 4 copper layers: {below}"]
    assert lines[2:6] == [f"layer escalation: {n} copper layers: skipped - not in ir.pcb.manufacturing.layer_count_options [2, 4]" for n in (6, 8, 10, 12)]
    assert lines[6] == "layer escalation: no count from 6 up was routed; the board is routed at its own 2 copper layers"
    assert lines[7].startswith("layer escalation: 2 copper layers (signals F.Cu, B.Cu; planes none): routing.maze 0.2 left 1 of 2")
    assert lines[8] == f"layer escalation: 4 copper layers (signals F.Cu, B.Cu; planes In1.Cu GND, In2.Cu GND): {ROUTED}"
    assert lines[-1] == (f"layer escalation: 4 layers kept - the first count of {LADDER} at which every net routes (the project owner's rule; "
                         "below the parts' minimum of 6, at or above which no count was routed)")
    # the kept count is below the parts' minimum, and its record says so (never "started at the parts' minimum")
    why = "R1 needs at least 6 copper layers (Vendor X1 datasheet rev. B, p. 12 'PCB layout')"
    assert "started at" not in stack.provenance.note and f"; below the parts' minimum of 6 copper layers ({why}): no count at or above it was routed - " in (
        stack.provenance.note)
    assert stack.provenance.derived_from == ["pcb.placements", "layer_escalation:2,4,6,8,10,12"]
    # an RO4350B board (its own stacks: 4) with the same minimum climbs to its 4-layer stack
    calls.clear()
    ro2, ro4 = MATERIAL_STACKS[("rogers_ro4350b", 2)], MATERIAL_STACKS[("rogers_ro4350b", 4)]
    ir = _template_board(tmp_path, lib, "divider", material_stackup(ro2, MaterialDecision(material="rogers_ro4350b", reason="carrier 2.4 GHz"), "divider",
                                                                     confirmed=True, ground_net="GND", power_net="+5V"))
    ir.components[0].min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None), (4, ro4.title)] and res.proposals[0].payload.stackup.layer_count == 4


def test_a_parts_minimum_rung_that_routes_no_whole_board_leaves_no_stack_without_copper(tmp_path: Path, lib: KicadLibrary,
                                                                                      monkeypatch: pytest.MonkeyPatch):
    """Review finding: with no rung routing every net and no partial answer, the 6-layer stack went into the IR without copper and the
    advised re-run with pcb.routing=partial could never route it. Now the board's own placement is proposed and the re-run climbs again."""
    calls = _canned(monkeypatch, dict.fromkeys((6, 8, 10, 12), False))
    ir = _plain(tmp_path, lib)
    ir.components[0].min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [6, 8, 10, 12]
    board = res.proposals[0].payload
    assert board.stackup is None and [layer.name for layer in board.layers] == ["F.Cu", "B.Cu"] and board.zones == [] and board.tracks == []
    notes = _board_notes(res.notes)
    assert notes[1].startswith("not applied: routing.maze 0.9 connected 1 of 2 net(s)") and notes[2] == "not routed: A: fixture: no legal route"
    assert notes[-1] == "layer escalation: no count up to 12 routes every net; the 6-layer result stands"
    assert not any(n.startswith("plane zones") for n in notes)  # no plane is proposed
    Orchestrator.apply_proposals(ir, res.proposals)
    # the advised answer: the same climb, and its first routed count applies with the nets it routed
    calls.clear()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{pcb_module.ROUTING_KEY: "partial"}))
    assert [n for n, _ in calls] == [6, 8, 10, 12]
    board = res.proposals[0].payload
    assert board.stackup.layer_count == 6 and [(z.net, z.layer) for z in board.zones] == [("GND", "In1.Cu"), ("GND", "In4.Cu")]
    assert "no count up to 12 routes every net, so this 6-layer result stands" in board.stackup.provenance.note
    assert "plane zones: GND on In1.Cu, GND on In4.Cu (outline inset 0.5 mm; KiCad fills them)" in res.notes


def test_a_parts_minimum_on_a_board_without_an_outline_is_a_reason_not_a_crash(tmp_path: Path, lib: KicadLibrary):
    """Review finding: the climb drew plane zones on outline None and raised AttributeError out of the PCB stage."""
    ir = _plain(tmp_path, lib)
    ir.pcb.outline = None
    ir.components[0].min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    no_outline = "cannot route: ir.pcb.outline is None; the grid needs a board outline"
    assert res.proposals == [] or res.proposals[0].payload.tracks == []
    assert [f"layer escalation: {n} copper layers: not routed - {no_outline}" for n in (6, 8, 10, 12)] == [
        n for n in res.notes if n.startswith("layer escalation: ") and "not routed - " in n]
    assert f"not routed: {no_outline}" in res.notes
    # a plane clearance parameter that is no number is a count's reason too (TypeError), never a crash
    ir = _plain(tmp_path, lib)
    ir.components[0].min_pcb_layers = _sourced(6)
    ir.parameters["plane_edge_clearance"] = Traced(value=None, unit="mm", provenance=Provenance(kind=ProvenanceKind.ASSUMPTION, note="unset"))
    res = PCBAgent(routing=ONE).run(ir, _ctx(tmp_path, lib))
    assert any(n.startswith("layer escalation: 6 copper layers: not routed - ") for n in res.notes)


def test_a_flex_board_has_no_stack_to_climb_to(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    flex = MATERIAL_STACKS[("polyimide_flex", 2)]
    calls = _canned(monkeypatch, {2: False})
    ir = _template_board(tmp_path, lib, "divider", material_stackup(flex, MaterialDecision(material="polyimide_flex", reason="bent once"), "divider",
                                                                     confirmed=True))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    assert _escalation(res.notes) == ["layer escalation: not attempted - the DuPont Pyralux AP (AP9121R) board has no stack of its material above 2 "
                                      "copper layer(s) (materials.py: [2]); no FR-4 stack is swapped in"]


# --------------------------------------------------------------------------- the parts' minimum


def _sourced(n: int, title: str = "Vendor X1 datasheet rev. B", section: str | None = "p. 12 'PCB layout'") -> Traced[int]:
    return Traced(value=n, unit="layers", provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, source=SourceRef(title=title, section=section)))


def test_a_parts_sourced_minimum_starts_the_ladder_at_six(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {6: True})
    ir = _plain(tmp_path, lib)
    ir.components[0].min_pcb_layers = _sourced(6)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(6, "6-layer 1.6 mm FR-4")]  # 2 and 4 are never routed
    why = "R1 needs at least 6 copper layers (Vendor X1 datasheet rev. B, p. 12 'PCB layout')"
    assert _escalation(res.notes) == [
        "layer escalation: 2 copper layers: skipped - below the parts' minimum of 6 copper layers",
        "layer escalation: 4 copper layers: skipped - below the parts' minimum of 6 copper layers",
        f"layer escalation: 6 copper layers (signals F.Cu, In2.Cu, In3.Cu, B.Cu; planes In1.Cu GND, In4.Cu GND): {ROUTED}",
        f"layer escalation: 6 layers kept - the first count of {LADDER} at which every net routes (the project owner's rule, started at the parts' minimum of 6)",
    ]
    prov = res.proposals[0].payload.stackup.provenance
    assert prov.derived_from[-1] == "components[R1].min_pcb_layers" and f"; started at the parts' minimum of 6 copper layers: {why} - " in prov.note
    # a minimum of 5 starts at 6 too (the first count at or above it); none routes every net: the first routed count stands - all or
    # nothing, so the proposal is the board's own placement (no stack without copper), and pcb.routing=partial applies its nets
    calls = _canned(monkeypatch, dict.fromkeys((6, 8, 10, 12), False))
    ir.components[0].min_pcb_layers = _sourced(5)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [6, 8, 10, 12]
    board = res.proposals[0].payload
    assert board.stackup is None and board.tracks == []
    assert _escalation(res.notes)[-1] == "layer escalation: no count up to 12 routes every net; the 6-layer result stands"
    res = PCBAgent().run(ir, _ctx(tmp_path, lib, **{pcb_module.ROUTING_KEY: "partial"}))
    board = res.proposals[0].payload
    assert board.stackup.layer_count == 6 and "no count up to 12 routes every net, so this 6-layer result stands" in board.stackup.provenance.note


def test_a_minimum_without_its_source_is_not_counted_and_a_stated_count_is_not_overridden(tmp_path: Path, lib: KicadLibrary,
                                                                                            monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {2: True})
    ir = _plain(tmp_path, lib)
    ir.components[0].min_pcb_layers = Traced(value=6, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="trust me"))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    assert _escalation(res.notes) == ["layer escalation: R1 min_pcb_layers 6 names no source document - not counted (a part fact is read from its "
                                      "manufacturer's datasheet, the document and page cited)"]
    # sourced, but the user stated 2 layers: the user's count stands and the note says why the minimum is not applied
    calls.clear()
    ir.components[0].min_pcb_layers = _sourced(6, section=None)
    _stated(ir, 2)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert calls == [(2, None)]
    [line] = _escalation(res.notes)
    assert line.startswith("layer escalation: the parts need at least 6 copper layers (R1 needs at least 6 copper layers (Vendor X1 datasheet rev. B)); "
                           "not applied - req.layers states 2 copper layer(s), the user's choice")


def _bga(lib: KicadLibrary, name: str, n: int, pitch: float, land: float) -> None:
    """An ``n`` x ``n`` area array of round SMD lands, numbered row by row from 1, written into the fixture library."""
    pads = "\n  ".join(f'(pad "{r * n + c + 1}" smd circle (at {c * pitch:g} {r * pitch:g}) (size {land:g} {land:g}) (layers "F.Cu" "F.Mask" "F.Paste"))'
                       for r in range(n) for c in range(n))
    root = Path(lib.roots[0])
    (root / "footprints" / "Test.pretty" / f"{name}.kicad_mod").write_text(
        f'(footprint "{name}" (version 20260206) (generator "pcbnew") (layer "F.Cu") (attr smd)\n'
        f'  (fp_rect (start -1 -1) (end {n * pitch:g} {n * pitch:g}) (stroke (width 0.05) (type solid)) (fill no) (layer "F.CrtYd"))\n  {pads})\n',
        encoding="utf-8")


def _bga_board(tmp_path: Path, lib: KicadLibrary, name: str, n: int, pitch: float, land: float) -> CircuitIR:
    _bga(lib, name, n, pitch, land)
    ir = board_ir(tmp_path, lib, [("U1", name, 10.0, 10.0), ("R1", "PAD1", 2.0, 2.0)], {}, (60.0, 60.0))
    # every land on a net to R1 (an unconnected land needs no escape): the pins are the land numbers
    ir.nets = [Net(name=f"N{k}", pins=[PinRef(component_ref="U1", pin_number=str(k)), PinRef(component_ref="R1", pin_number="1")], provenance=NET_P)
               for k in range(1, n * n + 1)]
    return ir


def test_the_escape_depth_of_an_area_array_sets_the_minimum(tmp_path: Path, lib: KicadLibrary):
    """A 10 x 10 array at 2.0 mm with 0.5 mm lands is 5 rows deep: 3 signal layers (NXP AN10778 sec. 5.2), the channel holds one trace
    at the default rules (g = 2.0 - 0.8 = 1.2 mm >= 2 x 0.25 + 0.4 = 0.9 mm, TI SPRU811A): 6 copper layers, the first count with 4."""
    ir = _bga_board(tmp_path, lib, "BGA10", 10, 2.0, 0.5)
    fp = lib.load_footprint(ir.component("U1").footprint)
    assert _area_array(fp, {str(k) for k in range(1, 101)}) == (5, 2.0, 0.5)
    assert _area_array(fp, {str(k) for k in range(1, 11)}) == (1, 2.0, 0.5)  # only the outer row on a net: 1 row deep
    assert LADDER_SIGNALS == {2: 2, 4: 2, 6: 4, 8: 5, 10: 6, 12: 8}
    m = _parts_minimum(ir, lib, RoutingParams())
    assert m.layers == 6 and m.derived == ["escape_depth:U1=5"] and m.notes == []
    assert m.reasons[0].startswith("U1 (Test:BGA10): pads 5 rows deep in a 2 mm area array need 3 signal layers (NXP AN10778 Rev. 2")
    assert "one trace fits: g = 2 - 0.8 = 1.2 mm >= 2s + w = 0.9 mm, TI SPRU811A (May 2005) p. 21, 24-27" in m.reasons[0]
    assert m.reasons[0].endswith("not re-opened here) - 6 copper layers is the first count of the ladder with 4")
    # a perimeter-only or shallow array adds nothing (4 rows: the two outer layers escape them)
    shallow = _bga_board(tmp_path / "s", lib, "BGA8", 8, 2.0, 0.5)
    assert _parts_minimum(shallow, lib, RoutingParams()) == pcb_module._Minimum()
    # 17 x 17 is 9 rows deep: 5 signal layers, which the 8-layer stack is the first to have
    deep = _bga_board(tmp_path / "d", lib, "BGA17", 17, 2.0, 0.5)
    m = _parts_minimum(deep, lib, RoutingParams())
    assert m.layers == 8 and m.derived == ["escape_depth:U1=9"] and m.reasons[0].endswith("- 8 copper layers is the first count of the ladder with 5")


def test_an_area_array_whose_channel_holds_no_trace_gets_no_estimate(tmp_path: Path, lib: KicadLibrary):
    ir = _bga_board(tmp_path, lib, "BGA10F", 10, 0.8, 0.4)
    m = _parts_minimum(ir, lib, RoutingParams())
    assert m.layers == 0 and m.reasons == []
    [note] = m.notes
    assert note.startswith("U1 (Test:BGA10F): pads 5 rows deep in a 0.8 mm area array - no escape-depth estimate: one trace between lands needs "
                           "g >= 2s + w = 0.9 mm and g = pitch - d = 0 mm (d = the larger of the 0.4 mm land and the 0.8 mm via pad)")


def test_the_escape_depth_starts_the_agents_ladder(tmp_path: Path, lib: KicadLibrary, monkeypatch: pytest.MonkeyPatch):
    calls = _canned(monkeypatch, {6: True})
    ir = _bga_board(tmp_path, lib, "BGA10", 10, 2.0, 0.5)
    ir.nets.append(Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="R1", pin_number="2")], provenance=NET_P))
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [n for n, _ in calls] == [6]
    assert res.proposals[0].payload.stackup.provenance.derived_from[-1] == "escape_depth:U1=5"


# --------------------------------------------------------------------------- the design file


def _with_minimum(**fields) -> dict:
    data = json.loads(json.dumps(RC))
    data["components"][0].update(fields)
    return data


@pytest.mark.parametrize(("fields", "message"), [
    ({"min_pcb_layers": 6}, "R1 min_pcb_layers 6 needs min_pcb_layers_source: one line naming the manufacturer document"),
    ({"min_pcb_layers": 6, "min_pcb_layers_source": "  "}, "R1 min_pcb_layers 6 needs min_pcb_layers_source"),
    ({"min_pcb_layers": 6, "min_pcb_layers_source": "a\nb"}, "R1 min_pcb_layers_source must be one line"),
    ({"min_pcb_layers_source": "Vendor datasheet p. 3"}, "R1 has min_pcb_layers_source without min_pcb_layers"),
    ({"min_pcb_layers": 0, "min_pcb_layers_source": "Vendor datasheet p. 3"}, "R1 min_pcb_layers must be a whole copper layer count >= 1"),
])
def test_a_design_file_minimum_needs_its_source(tmp_path: Path, fields: dict, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _with_minimum(**fields)), template_library(tmp_path))


def test_a_design_file_minimum_reaches_the_ir_with_its_source(tmp_path: Path):
    source = "Vendor X1 datasheet rev. B, p. 12 'PCB layout'"
    plan = load_design_file(_write(tmp_path, _with_minimum(min_pcb_layers=6, min_pcb_layers_source=source)), template_library(tmp_path))
    assert any(line.startswith("R1 ") and f"; needs at least 6 copper layers per {source} (the file's word, not opened)" in line for line in plan.parts)
    [r1] = [c.payload for c in plan.changes if c.target == "components" and c.payload.ref == "R1"]
    t = r1.min_pcb_layers
    assert (t.value, t.unit, t.provenance.kind, t.provenance.source.title) == (6, "layers", ProvenanceKind.USER_REQUIREMENT, source)
    assert plan.sha256 in t.provenance.note and "as the file states it (not opened)" in t.provenance.note
    # without the field the part and the table are what they were
    plain = load_design_file(_write(tmp_path, RC), template_library(tmp_path))
    assert all("copper layers" not in line for line in plain.parts)
    assert all("min_pcb_layers" not in json.dumps(c.payload.model_dump(mode="json")) for c in plain.changes if c.target == "components")
