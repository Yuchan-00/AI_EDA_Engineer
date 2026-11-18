"""Optional design-file SI: confirmation, provenance and unchanged routing unless promotion is requested."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, IRProposal, PCBAgent
from ai_eda.agents.circuit import table_hash
from ai_eda.design.base import CHOICE_NOTE_PREFIX, CONFIRM_DESIGN_KEY
from ai_eda.design.board import BoardContext, si_choices
from ai_eda.design.design_file import DesignFileError, load_design_file
from ai_eda.ir import CircuitIR, ProjectMeta, ProvenanceKind
from ai_eda.ir.provenance import design_data
from ai_eda.tools.si.rules import net_rules
from ai_eda.validation.si import si_results
from ai_eda.workflow import Orchestrator
from tests.test_circuit_templates import _ir, template_library
from tests.test_design_file import _run, _shown, _variant, _write
from tests.test_routing import fixture_library
from tests.test_si_checks import long_board


def _apply(plan, ir=None):
    if ir is None:
        ir = CircuitIR(project=ProjectMeta(id="design-si", name="design-si"))
    Orchestrator.apply_proposals(ir, [IRProposal(**c.model_dump()) for c in plan.changes])
    return ir


def _plan(tmp_path, block):
    return load_design_file(_write(tmp_path, _variant(si=block)), template_library(tmp_path))


def test_absent_block_preserves_the_plan_confirmation_hash_and_ir(tmp_path: Path):
    # Fingerprints from the loader before the optional SI block; only the library locator is normalized in the table.
    plan = load_design_file(_write(tmp_path), template_library(tmp_path))
    table = plan.table().replace(str(tmp_path), "<root>").replace("\\", "/")
    assert table_hash(table) == "sha256:2c3ecd7fb4b7cafcc7ab732f4d30c8211122ea2d8e561e830bca7829d0f8af0c"
    changes = json.dumps([design_data(c) for c in plan.changes], sort_keys=True)
    assert hashlib.sha256(changes.encode()).hexdigest() == "083fd3f254626e6e6ab8b41b2dd0671178acbd57d7b4e258d184c28f82de1fc6"
    ir = _apply(plan)
    assert ir.si is None and not plan.choices and not plan.board
    assert ir.content_hash() == "sha256:f89b4d69a0d2e81f4a508e99a764038c71009009ad99215ecd0b8d2692a5dbb8"


def test_present_block_shows_shared_choices_and_waits_for_confirmation(tmp_path: Path):
    plan = _plan(tmp_path, {})
    ir = _ir(tmp_path)
    table = _shown(ir, tmp_path, plan)
    assert ir.si is None
    assert "SI design choices" in table and "not routed as controlled impedance" in table
    ctx = BoardContext(template_id="design_file", confirmed=True, params={}, stackup=None, driver={})
    si_choices(ctx, promote=False)
    assert plan.choices == ctx.choices
    assert {c.key for c in plan.choices} == {"si.t_rise", "si.r_drive", "si.c_load", "si.ringing_tol", "si.critical_fraction"}
    assert all(c.text() in table for c in ctx.choices)
    _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    assert [c.name for c in ir.si.net_classes] == ["DEFAULT"]
    default = ir.si.net_classes[0]
    assert default.promote_to is None and default.target_z0_ohm is None
    for key, traced in ctx.driver.items():
        actual = getattr(default, key)
        assert design_data(actual) == design_data(traced)
        assert actual.provenance.note.startswith(CHOICE_NOTE_PREFIX)
    assert default.driver is None and default.xtalk_budget_rel is None


def test_adding_the_block_invalidates_an_earlier_confirmation(tmp_path: Path):
    ir = _ir(tmp_path)
    plain = load_design_file(_write(tmp_path), template_library(tmp_path))
    _shown(ir, tmp_path, plain)
    plan = _plan(tmp_path, {})
    state = _run(ir, tmp_path, plan, {CONFIRM_DESIGN_KEY: "yes"})
    assert state.blocked and not ir.components and ir.si is None
    assert any("is not the one you confirmed" in o.message for o in state.outcomes)


@pytest.mark.parametrize("structured", [False, True])
def test_overrides_convert_units_and_keep_the_files_provenance(tmp_path: Path, structured: bool):
    values = {"t_rise": (2, "ns"), "r_drive": (0.033, "kohm"), "c_load": (10, "pF")}
    block = {key: {"value": v, "unit": u} if structured else f"{v} {u}" for key, (v, u) in values.items()}
    block.update(ringing_tol=0.2, critical_fraction=0.25)
    plan = _plan(tmp_path, block)
    si = _apply(plan).si
    default = si.net_classes[0]
    for actual, value, unit in [(default.t_rise_s, 2e-9, "s"), (default.r_drive_ohm, 33, "ohm"),
                                (default.c_load_f, 1e-11, "F"), (default.ringing_tol_rel, 0.2, None),
                                (si.critical_fraction, 0.25, None)]:
        assert actual.value == value and actual.unit == unit
        assert actual.provenance.kind is ProvenanceKind.USER_REQUIREMENT
        assert actual.provenance.note == f"design file {plan.sha256}"
    assert all("your value" in c.text() for c in plan.choices)


@pytest.mark.parametrize("promote", [False, True])
def test_xtalk_budget_is_a_user_value_on_each_class(tmp_path: Path, promote: bool):
    plan = _plan(tmp_path, {"xtalk_budget_rel": 0.08, "promote": promote})
    for cls in _apply(plan).si.net_classes:
        budget = cls.xtalk_budget_rel
        assert budget.value == 0.08 and budget.unit is None
        assert budget.provenance.kind is ProvenanceKind.USER_REQUIREMENT
        assert budget.provenance.note == f"design file {plan.sha256}"
    assert "si.xtalk_budget_rel = 0.08" in plan.table() and "precedence over the NM / 3" in plan.table()


@pytest.mark.parametrize("block", [
    {"t_rise": 2}, {"t_rise": "2 V"}, {"t_rise": "1 ns to 2 ns"}, {"t_rise": "2 ns max"},
    {"t_rise": "0 ns"}, {"t_rise": "-1 ns"}, {"t_rise": "1e999 s"},
    {"r_drive": {"value": 33, "unit": "V"}}, {"r_drive": {"value": 33}},
    {"c_load": {"value": True, "unit": "pF"}}, {"c_load": {"value": 10**400, "unit": "F"}},
    {"ringing_tol": 0}, {"ringing_tol": 1.1}, {"critical_fraction": -0.5},
    {"critical_fraction": 10**400}, {"critical_fraction": True},
    {"xtalk_budget_rel": "8%"}, {"xtalk_budget_rel": {"value": 0.08}}, {"xtalk_budget_rel": [0.08]},
    {"xtalk_budget_rel": float("nan")}, {"xtalk_budget_rel": float("inf")},
    {"promote": "true"}, {"promote": 1}, {"unknown": 1},
    {"provenance": {"kind": "authoritative"}},
])
def test_invalid_overrides_are_refused_before_the_pipeline(tmp_path: Path, block):
    with pytest.raises(DesignFileError):
        _plan(tmp_path, block)


@pytest.mark.parametrize("promote", [False, True])
def test_only_explicit_promotion_changes_a_long_nets_routing(tmp_path: Path, promote: bool):
    plan = _plan(tmp_path, {"promote": promote})
    lib = fixture_library(tmp_path / "routing-library")
    board = long_board(tmp_path, lib, 4)
    si_change = next(c for c in plan.changes if c.target == "si")
    Orchestrator.apply_proposals(board, [IRProposal(**si_change.model_dump())])
    assert net_rules(board).rules == {}
    plain = board.model_copy(deep=True)
    plain.si = None
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": lib})
    result = PCBAgent().run(board, ctx)
    Orchestrator.apply_proposals(board, result.proposals)
    if promote:
        assert [c.name for c in board.si.net_classes] == ["DEFAULT", "Z50"]
        assert board.si.class_of("LONG").name == "Z50"
        assert board.si.promotion_of("LONG").provenance.tool == "si.promote"
        assert net_rules(board).rules["LONG"].width_mm == 0.35
        assert {t.width_mm for t in board.pcb.tracks if t.net == "LONG"} == {0.35}
        assert any("routed once more" in note for note in result.notes)
    else:
        assert [p.target for p in result.proposals] == ["pcb"]
        assert all(board.si.promotion_of(n.name) is None for n in board.nets)
        assert net_rules(board).rules == {}
        Orchestrator.apply_proposals(plain, PCBAgent().run(plain, ctx).proposals)
        assert design_data(board.pcb) == design_data(plain.pcb)
        critical = next(r for r in si_results(board, lib) if r.check_id == "si.critical_length")
        assert "LONG" in critical.details["long_over_plane"]
