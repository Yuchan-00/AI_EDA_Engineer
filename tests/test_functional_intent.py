"""Functional intent is explicit design content; inference is a separate pure view."""

import copy
import hashlib
import json
import random
import runpy
import sys
from pathlib import Path

import pytest

from ai_eda.agents import IRProposal
from ai_eda.agents.circuit import table_hash
from ai_eda.design.design_file import DesignFileError, load_design_file
from ai_eda.ir import CircuitIR, Net, NetKind, PinRef, ProvenanceKind, Topology
from ai_eda.ir.provenance import design_data
from ai_eda.tools.functional_intent import infer_blocks, resolve_intent
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.workflow import Orchestrator
from tests.test_circuit_templates import _ir, template_library
from tests.test_design_file import RC, _write
from tests.test_parts_existence import make_part

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "designs"


def apply(plan, tmp_path):
    ir = _ir(tmp_path)
    Orchestrator.apply_proposals(ir, [IRProposal(**c.model_dump()) for c in plan.changes])
    return ir


def declared():
    data = copy.deepcopy(RC)
    data["topology"]["blocks"] = [{"id": "filter", "function": "low-pass filter", "domain": "analog",
                                    "component_refs": ["R1", "C1"], "input_nets": ["IN"], "output_nets": ["OUT"]}]
    data["topology"]["pin_associations"] = [{"terminal": "C1.1", "ic_pin": "R1.2", "role": "output shunt",
                                               "rationale": "the output terminal served by the shunt"}]
    return data


def test_absent_keys_preserve_pre_contract_plan_confirmation_and_ir(tmp_path):
    old = json.loads((DATA / "functional_intent_legacy.json").read_text(encoding="utf-8"))
    plan = load_design_file(_write(tmp_path), template_library(tmp_path))
    changes = [design_data(c) for c in plan.changes]
    assert hashlib.sha256(json.dumps(changes, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == old["changes_sha256"]
    # The legacy table includes library locators; restore them for this installation.
    assert table_hash(old["table"]) == old["table_hash"]
    table = old["table"]
    for c in plan.changes:
        if c.target == "components":
            table = table.replace("{library:" + c.payload.symbol.library + "}", c.payload.symbol.library_path)
    assert plan.table() == table
    assert table_hash(plan.table()) == table_hash(table)
    ir = apply(plan, tmp_path)
    assert ir.design_dict() == old["ir"]
    assert ir.content_hash() == old["ir_hash"]
    assert "pin_associations" not in ir.topology.model_dump()
    assert "inferred" not in infer_blocks(ir.components, ir.nets)[0].model_copy(update={"inferred": False}).model_dump()


def test_declared_intent_round_trips_and_is_shown_in_confirmation(tmp_path):
    plan = load_design_file(_write(tmp_path, declared()), template_library(tmp_path))
    ir = apply(plan, tmp_path)
    loaded = CircuitIR.loads(ir.model_dump_json())
    assert loaded.topology == ir.topology
    assert loaded.content_hash() == ir.content_hash()
    assert ir.topology.blocks[0].component_refs == ["R1", "C1"]
    assert not ir.topology.blocks[0].inferred
    a = ir.topology.pin_associations[0]
    assert str(a.terminal) == "C1.1" and str(a.ic_pin) == "R1.2"
    assert a.provenance.kind == ProvenanceKind.USER_REQUIREMENT
    assert plan.sha256 in a.provenance.note
    assert "block filter: low-pass filter" in plan.table()
    assert "C1.1 -> R1.2: output shunt" in plan.table()
    before = ir.content_hash()
    ir.topology.pin_associations[0].role = "changed intent"
    assert ir.content_hash() != before


@pytest.mark.parametrize(("edit", "reason"), [
    (lambda t: t["blocks"][0]["component_refs"].append("U404"), "intent.component_ref.*U404"),
    (lambda t: t["blocks"][0]["input_nets"].append("MISSING"), "intent.net_ref.*MISSING"),
    (lambda t: t["blocks"].append(t["blocks"][0]), "intent.block_id"),
    (lambda t: t["pin_associations"][0].update(terminal="C1.999"), "intent.pin_ref.*C1.999"),
    (lambda t: t["pin_associations"][0].update(ic_pin="U404.1"), "intent.pin_ref.*U404"),
    (lambda t: t["pin_associations"][0].update(ic_pin="R1.1"), "intent.association_net"),
    (lambda t: t["pin_associations"][0].update(ic_pin="C1.2"), "intent.self_association"),
    (lambda t: t["pin_associations"][0].update(terminal="C1"), "intent.pin_ref"),
    (lambda t: t["pin_associations"].append(t["pin_associations"][0]), "intent.duplicate_association"),
    (lambda t: t["pin_associations"][0].update(provenance={"kind": "authoritative"}), "provenance.*Extra inputs"),
])
def test_invalid_intent_is_refused_with_named_reason(tmp_path, edit, reason):
    data = declared()
    edit(data["topology"])
    with pytest.raises(DesignFileError, match=reason):
        load_design_file(_write(tmp_path, data), template_library(tmp_path))


def test_inference_is_pure_deterministic_and_rails_do_not_join_groups(tmp_path):
    ir = _ir(tmp_path)
    ir.components = [make_part(ref) for ref in ("U1", "U2", "R10", "R2", "C1", "C2")]
    prov = ir.components[0].provenance
    for c in ir.components[-2:]:
        c.symbol.library, c.symbol.name = "Device", "C"
    def net(name, pins, kind=NetKind.SIGNAL):
        return Net(name=name, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], kind=kind, provenance=prov)
    ir.nets = [net("VCC", [(c.ref, "1") for c in ir.components], NetKind.POWER),
               net("GND", [(r, "2") for r in ("C1", "C2")], NetKind.GROUND),
               net("S1", [("U1", "2"), ("R2", "2")]), net("S2", [("U2", "2"), ("R10", "2")])]
    before = ir.model_dump_json()
    expected = resolve_intent(ir)
    assert ir.model_dump_json() == before
    assert [b.id for b in expected.blocks] == sorted(b.id for b in expected.blocks)
    assert {tuple(b.component_refs) for b in expected.blocks} == {("C1",), ("C2",), ("R2", "U1"), ("R10", "U2")}
    assert all(b.inferred and b.provenance.kind == ProvenanceKind.DERIVED and "inferred" in b.provenance.note for b in expected.blocks)
    assert expected.pin_associations == []
    assert len(expected.unassociated) == 2 and "C1: unassociated" in expected.unassociated[0]
    rng = random.Random(17)
    for _ in range(12):
        rng.shuffle(ir.components)
        rng.shuffle(ir.nets)
        for n in ir.nets:
            rng.shuffle(n.pins)
        assert resolve_intent(ir) == expected  # includes provenance timestamps


def test_declared_blocks_take_precedence_and_view_does_not_alias_ir(tmp_path):
    ir = apply(load_design_file(_write(tmp_path, declared()), template_library(tmp_path)), tmp_path)
    view = resolve_intent(ir)
    assert [b.id for b in view.blocks] == ["filter"]
    view.blocks[0].component_refs.clear()
    view.pin_associations[0].role = "other"
    assert ir.topology.blocks[0].component_refs == ["R1", "C1"]
    assert ir.topology.pin_associations[0].role == "output shunt"


def test_stm32_example_loads_and_generator_uses_existing_wired_pins(tmp_path, monkeypatch):
    library = KicadLibrary()
    if library.symbol_file("MCU_ST_STM32F4") is None:
        pytest.skip("STM32 KiCad library unavailable")
    output = tmp_path / "design.json"
    monkeypatch.setattr(sys, "argv", ["make_stm32f407_display_devboard.py", str(output)])
    runpy.run_path(str(EXAMPLES / "make_stm32f407_display_devboard.py"), run_name="__main__")
    assert output.read_bytes() == (EXAMPLES / "stm32f407_display_devboard.json").read_bytes()
    ir = apply(load_design_file(output, library), tmp_path)
    assert len(ir.topology.blocks) == 14
    assert {ref for b in ir.topology.blocks for ref in b.component_refs} == {c.ref for c in ir.components}
    vdd = {p.number for p in ir.component("U1").pins if p.name == "VDD"}
    assigned = [a for a in ir.topology.pin_associations if a.role == "VDD decoupling"]
    assert {a.ic_pin.pin_number for a in assigned} == vdd
    assert len(assigned) == len(vdd)
    assert resolve_intent(ir).unassociated == []
