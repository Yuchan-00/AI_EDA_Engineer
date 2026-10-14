"""The PCB agent's signal-integrity flow: route with the class rules -> measure -> promote -> re-route once -> planes -> silkscreen.

Synthetic boards (the fixture library of ``tests/test_routing.py``), no KiCad,
no ngspice. What is proved:

* a board whose classes constrain nothing - every template board whose nets
  stay short - gets routing.maze 0.2's copper byte for byte (the same design
  view as a board with no SI data at all);
* an electrically long net over a plane is promoted to the controlled class
  (a second ``si`` proposal with ``derived`` provenance) and the board is
  re-routed once with the class's ``calc.tline.width_for_z0`` width; the
  independent ``si.impedance`` check then PASSes the promoted net;
* without a plane the promotion is recorded, nothing is re-routed (the rule
  gives no width) and the checks say why;
* a re-route that loses a net keeps the first pass's copper;
* the 4-layer planes are zones of the agent's own (not "existing copper" for
  a later run), and the whole flow is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ai_eda.agents.pcb as pcb_module
from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.design.board import PLANE_CLEARANCE_KEY
from ai_eda.design.stackup import STACKUP_TOOL
from ai_eda.ir import CircuitIR, ValidationStatus as S, user_requirement
from ai_eda.ir.provenance import design_data
from ai_eda.tools.routing.maze import ROUTER_RULES_VERSION, ROUTER_VERSION, route_board
from ai_eda.validation.si import si_results
from ai_eda.workflow import Orchestrator
from tests.test_routing import fixture_library
from tests.test_si_checks import default_classes, long_board, si_of


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def _ctx(tmp_path: Path, lib) -> AgentContext:
    return AgentContext(workdir=tmp_path, tools={"kicad_library": lib})


def _board(tmp_path: Path, lib, layers: int | None, *, si: bool = True, length: float = 90.0) -> CircuitIR:
    ir = long_board(tmp_path, lib, layers, length=length)
    if si:
        ir.si = si_of(*default_classes())
    if layers == 4:
        ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    return ir


def _apply(ir: CircuitIR, res) -> CircuitIR:
    Orchestrator.apply_proposals(ir, res.proposals)
    return ir


def _copper(ir: CircuitIR) -> list:
    return [design_data(t) for t in ir.pcb.tracks] + [design_data(v) for v in ir.pcb.vias]


def test_a_board_whose_classes_constrain_nothing_routes_exactly_as_without_classes(tmp_path: Path, lib):
    with_si = _board(tmp_path / "a", lib, 2, length=30.0)
    plain = _board(tmp_path / "b", lib, None, si=False, length=30.0)
    ra = PCBAgent().run(with_si, _ctx(tmp_path, lib))
    rb = PCBAgent().run(plain, _ctx(tmp_path, lib))
    assert [p.target for p in ra.proposals] == ["pcb"] and [p.target for p in rb.proposals] == ["pcb"]  # nothing promoted
    a, b = _apply(with_si, ra), _apply(plain, rb)
    assert _copper(a) == _copper(b) and all(t.provenance.tool_version == ROUTER_VERSION for t in a.pcb.tracks)
    direct = route_board(plain.model_copy(update={"pcb": plain.pcb.model_copy(update={"tracks": [], "vias": [], "silkscreen": []})}), lib)
    assert [design_data(t) for t in direct.tracks] == [design_data(t) for t in a.pcb.tracks]
    assert si_results(a)[0].status is S.PASS  # si.critical_length: every net short


def test_a_long_net_over_a_plane_is_promoted_and_the_board_rerouted_once_at_the_controlled_width(tmp_path: Path, lib):
    ir = _board(tmp_path, lib, 4)
    before = ir.content_hash()
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert ir.content_hash() == before  # the agent proposes, never mutates
    assert [p.target for p in res.proposals] == ["pcb", "si"]
    notes = " | ".join(res.notes)
    assert "si promotion: LONG: critical-length rule" in notes and "si re-route: routed once more with the promoted rules (LONG 0.35 mm)" in notes
    assert "plane zones: GND on In1.Cu, GND on In2.Cu" in notes
    board = _apply(ir, res)
    assert board.si.class_of("LONG").name == "Z50" and board.si.promotion_of("LONG").provenance.tool == "si.promote"
    long_tracks = [t for t in board.pcb.tracks if t.net == "LONG"]
    assert long_tracks and {t.width_mm for t in long_tracks} == {0.35} and all(t.provenance.tool_version == ROUTER_RULES_VERSION for t in long_tracks)
    assert {t.width_mm for t in board.pcb.tracks if t.net == "SHORT"} == {0.4}
    assert any("rule:class=Z50,width=0.35" in e for e in long_tracks[0].provenance.derived_from)
    planes = [z for z in board.pcb.zones if z.provenance.tool == STACKUP_TOOL]
    assert [(z.net, z.layer) for z in planes] == [("GND", "In1.Cu"), ("GND", "In2.Cu")] and planes[0].polygon[0] == (0.5, 0.5)
    got = {r.check_id: r for r in si_results(board)}
    assert got["si.impedance.Z50"].status is S.PASS and "within 50 ohm +/- 10%" in got["si.impedance.Z50"].message
    assert got["si.critical_length"].status is S.PASS and got["si.critical_length"].details["long_over_plane"] == ["LONG"]
    # deterministic: the same board gives the same proposals
    again = PCBAgent().run(_board(tmp_path / "again", lib, 4), _ctx(tmp_path, lib))
    assert [design_data(p.payload) for p in again.proposals] == [design_data(p.payload) for p in res.proposals]


def test_without_a_plane_the_promotion_is_recorded_and_nothing_is_rerouted(tmp_path: Path, lib):
    ir = _board(tmp_path, lib, 2)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert [p.target for p in res.proposals] == ["pcb", "si"]
    assert any("si re-route: not needed" in n and "impedance is undefined without a reference plane" in n for n in res.notes)
    board = _apply(ir, res)
    plain = _apply(_board(tmp_path / "plain", lib, None, si=False), PCBAgent().run(_board(tmp_path / "plain", lib, None, si=False), _ctx(tmp_path, lib)))
    assert _copper(board) == _copper(plain)  # the 0.2 copper, byte for byte
    assert not board.pcb.zones
    got = {r.check_id: r for r in si_results(board)}
    assert got["si.impedance.Z50"].status is S.NOT_VERIFIED and "use pcb_layers=4 or add a plane" in got["si.impedance.Z50"].message
    assert got["si.critical_length"].status is S.NOT_VERIFIED and got["si.critical_length"].details["long_without_plane"] == ["LONG"]


def test_a_reroute_that_loses_a_net_keeps_the_first_pass(tmp_path: Path, lib, monkeypatch):
    calls: list[int] = []
    real = pcb_module.route_board

    def flaky(ir, library, params=None, **kw):
        calls.append(1)
        got = real(ir, library, params, **kw)
        if len(calls) == 2:  # the re-route "loses" LONG
            got.tracks = [t for t in got.tracks if t.net != "LONG"]
            got.unrouted = {"LONG": "fixture: no legal route at 0.35 mm"}
        return got

    monkeypatch.setattr(pcb_module, "route_board", flaky)
    ir = _board(tmp_path, lib, 4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert len(calls) == 2 and any("si re-route left 1 net(s) unrouted (LONG: fixture" in n and "first pass's copper is kept" in n for n in res.notes)
    board = _apply(ir, res)
    assert {t.width_mm for t in board.pcb.tracks if t.net == "LONG"} == {0.4} and board.si.class_of("LONG").name == "Z50"
    imp = {r.check_id: r for r in si_results(board)}["si.impedance.Z50"]
    # the promoted net is judged at the width it has: 0.4 mm over the 0.2 mm prepreg is still inside 50 ohm +/- 10 %
    assert imp.status is S.PASS and {row["width_mm"] for row in imp.details["segments"]} == {0.4}


def test_the_agents_own_planes_are_not_existing_copper_for_a_later_run(tmp_path: Path, lib):
    ir = _board(tmp_path, lib, 4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    board = _apply(ir, res)
    board.pcb.tracks, board.pcb.vias, board.pcb.silkscreen = [], [], []
    board.si = si_of(*default_classes())
    again = PCBAgent().run(board, _ctx(tmp_path, lib))
    assert not any(n.startswith("not routed: ir.pcb already has copper") for n in again.notes)
    after = _apply(board, again)
    assert after.pcb.tracks and len([z for z in after.pcb.zones if z.provenance.tool == STACKUP_TOOL]) == 2  # not added twice


def _pair(tmp_path: Path, layers: int):
    from ai_eda.ir import DiffPair
    from tests.test_routing_rules import _pair_board
    from tests.test_si_checks import cls, stacked, u

    ir, lib = _pair_board(tmp_path)
    stacked(ir, layers)
    ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    ir.si = si_of(*default_classes(), cls("USB", nets=["DP", "DN"], pairs=[DiffPair(p="DP", n="DN")], target_zdiff_ohm=u(90.0, "ohm"), zdiff_tol_rel=u(0.1)))
    return ir, lib


def test_a_declared_pair_is_routed_coupled_over_a_plane_and_si_diff_judges_it(tmp_path: Path):
    ir, lib = _pair(tmp_path, 4)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert any(n.startswith("si rules: USB (2 net(s)): pair w") for n in res.notes), res.notes
    board = _apply(ir, res)
    tracks = [t for t in board.pcb.tracks if t.net in ("DP", "DN")]
    assert tracks and all("coupled differential pair DN/DP" in t.provenance.note for t in tracks)
    diff = {r.check_id: r for r in si_results(board)}["si.diff.DP/DN"]
    assert diff.status is S.PASS and diff.details["coupled_mm"] > 10.0 and all(
        abs(row["z_diff_ohm"] - 90.0) <= 9.0 for row in diff.details["rows"] if "z_diff_ohm" in row)


def test_without_a_plane_a_declared_pair_is_two_nets_and_si_diff_says_why(tmp_path: Path):
    ir, lib = _pair(tmp_path, 2)
    res = PCBAgent().run(ir, _ctx(tmp_path, lib))
    assert any("pairs DP/DN routed as two nets" in n and "impedance is undefined without a reference plane" in n for n in res.notes), res.notes
    board = _apply(ir, res)
    assert not any("coupled differential pair" in (t.provenance.note or "") for t in board.pcb.tracks)
    diff = {r.check_id: r for r in si_results(board)}["si.diff.DP/DN"]
    assert diff.status is S.NOT_VERIFIED and "use pcb_layers=4 or add a plane" in diff.message


def test_a_kept_first_pass_narrower_than_the_controlled_width_is_judged_not_excused_as_a_neck_down(tmp_path: Path, lib, monkeypatch):
    """The first pass at a 0.25 mm board width (the fine rules' width) kept after a re-route that loses LONG: LONG's 0.25 mm track is
    not a neck-down the router recorded, so si.impedance judges it - 59.2 ohm is outside 50 ohm +/- 10 % and FAILs."""
    from ai_eda.tools.routing.maze import RoutingParams

    real = pcb_module.route_board
    calls: list[int] = []

    def flaky(ir, library, params=None, **kw):
        calls.append(1)
        got = real(ir, library, params, **kw)
        if len(calls) == 2:
            got.tracks = [t for t in got.tracks if t.net != "LONG"]
            got.unrouted = {"LONG": "fixture: no legal route at 0.35 mm"}
        return got

    monkeypatch.setattr(pcb_module, "route_board", flaky)
    ir = _board(tmp_path, lib, 4)
    res = PCBAgent(routing=RoutingParams(track_width_mm=0.25)).run(ir, _ctx(tmp_path, lib))
    assert len(calls) == 2 and any("first pass's copper is kept" in n for n in res.notes)
    board = _apply(ir, res)
    assert {t.width_mm for t in board.pcb.tracks if t.net == "LONG"} == {0.25} and board.si.class_of("LONG").name == "Z50"
    imp = {r.check_id: r for r in si_results(board, lib)}["si.impedance.Z50"]
    assert imp.status is S.FAIL and "Z0 59.20 ohm outside 50 ohm +/- 10%" in imp.message and imp.details["neckdowns"] == []
