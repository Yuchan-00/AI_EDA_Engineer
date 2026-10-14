"""Every closed-world refusal is leavable by an answer, and a model question never forces the user into one.

The defect (2026-09-30, an end-to-end demo with ``--llm claude`` on the
``kr447_transceiver`` template):

* P1 - the model extracted descriptive explicit items under keys no template
  reads, in design categories (``antenna_switch_present`` electrical,
  ``antenna_integration`` mechanical, ``battery_voltage_max``), and an
  assumption (``battery_cell_count``) the user accepted. The closed world
  refused the design and told the user to "leave this requirement out" - but
  no answer did that (``--answer leave_out=...`` even became a requirement
  ``req.leave_out`` that refused the template once more); only editing
  ir.json or a correction that re-ran the model got out.
* P2 - a *required* model question under a key no template reads
  (``frequency_deviation_or_occupied_bandwidth``, ``regulatory_certification``)
  could only be closed by answering it, and the answer was recorded as an
  ``electrical`` requirement the closed world refused: a dead end.

The fix, pinned here:

* ``--answer leave_out=<key>,<key>`` - a control key and a requirement
  decision - moves the named design requirements into
  ``ir.requirements.left_out`` (the user's recorded decision, design content
  once non-empty, so older IRs keep their hashes); no template reads them,
  the closed world does not refuse on them, the reviewer does not demand them,
  and the confirmation tables, ``ai-eda report`` and the stage reports list
  them. Every closed-world refusal names that exact answer. It closes an open
  question under the key (with or without a requirement), is idempotent,
  loses to nothing but a later typed answer to the same key (which brings it
  back), and a ``confirm_design=yes`` beside it is ignored like beside
  ``accept_implicit``.
* A model question marked required under a key nothing reads is asked as
  optional (a model question is still only a question); under a key a
  template reads it stays required. A typed answer to any question is still
  recorded as a design requirement (no category guessed from a key's
  spelling) - and its refusal is leavable.

Offline on the divider with the synthetic library of
``tests/test_circuit_templates.py`` and the scripted LLM; the real-library twin
on ``kr447_transceiver`` needs the packed KiCad 10.0.6 libraries
(``KICAD10_SYMBOL_DIR``) and skips without them.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ai_eda.agents import AgentContext, RequirementAgent
from ai_eda.agents.keys import CONFIRM_DESIGN_KEY, CONTROL_KEYS, LEAVE_OUT_KEY, REQUIREMENT_DECISION_KEYS
from ai_eda.agents.leave_out import design_references, split_keys
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design import leave_out_answer, left_out_lines, template_reads_key
from ai_eda.design.inputs import read_inputs
from ai_eda.design.rf import family
from ai_eda.design.rf.t_transceiver import KR447_TRANSCEIVER
from ai_eda.ir import CircuitIR, LeftOutRequirement, ProjectMeta, ProvenanceKind, RequirementKind, ValidationStatus as S
from ai_eda.llm.extraction import ACCEPT_KEY, CONFIRM_KEY
from ai_eda.report.data import build_report_data, load_ir_file
from ai_eda.report.html import render_html
from ai_eda.report.stages import final_report, theory_report
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.workflow import Orchestrator, Stage
from tests.test_circuit_templates import template_library
from tests.test_requirement_agent_llm import USAGE, _req, _service

DATA = Path(__file__).parent / "data"

#: a divider request with a descriptive item no template reads, a model question marked required under a key nothing reads, a
#: required model question under a template key, and an assumption under a key no template reads (the demo's shapes, offline)
RAW = "12V 입력에서 5V 출력 분압기, 전원 스위치 포함, 한국에서 사용"
CANNED: dict[str, Any] = {
    "requirements": [
        _req("input_voltage", "Input 12 V", "12V 입력", 12, "V", "12V"),
        _req("output_voltage", "Output 5 V", "5V 출력", 5, "V", "5V"),
        {"key": "power_switch_present", "text": "A power switch is included", "kind": "explicit", "category": "electrical",
         "quote": "전원 스위치 포함", "value": None, "rationale": None},
    ],
    "questions": [
        {"key": "divider_tolerance_class", "question": "Which resistor tolerance class must the divider use?", "required": True, "options": [],
         "rationale": "the model thinks so"},
        {"key": "output_current", "question": "Does VOUT supply a load current?", "required": True, "options": [], "rationale": "load"},
    ],
    "conflicts": [],
    "assumptions": [{"key": "cell_count", "text": "the supply is two cells", "category": "electrical", "number": 2, "unit": "cell",
                     "number_high": None, "rationale": "a 12 V pack is usually several cells"}],
    "application": {"summary": "bench divider", "quote": "12V 입력에서 5V 출력 분압기"},
    "jurisdictions": [{"code": "KR", "quote": "한국에서 사용"}],
}
SCOPE = {"mains_powered": "no", "radio": "no"}


def _ir(tmp_path: Path, raw: str = RAW, name: str = "leave") -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))
    ir.requirements.raw_input = raw
    return ir


def _run(ir: CircuitIR, svc, tmp_path: Path, lib, answers: dict[str, str], stop_after: Stage | None = Stage.ARCHITECTURE):
    ctx = AgentContext(workdir=tmp_path, llm=svc, tools={"kicad_library": lib} if lib is not None else {}, answers=answers)
    return Orchestrator(ctx).run(ir, stop_after=stop_after)


def _refused_to_confirmed(tmp_path: Path):
    """Runs 1-2 of the P1 flow: the extraction is shown, then confirmed with the assumption accepted - the closed world refuses."""
    lib = template_library(tmp_path / "kicad")
    svc, client = _service([{"structured": CANNED, "usage": USAGE}])
    ir = _ir(tmp_path)
    state = _run(ir, svc, tmp_path, lib, {})
    assert state.blocked and [q.key for q in state.open_questions] == ["output_current", CONFIRM_KEY]
    state = _run(ir, svc, tmp_path, lib, {CONFIRM_KEY: "yes", ACCEPT_KEY: "cell_count", "output_current": "0 A", **SCOPE})
    assert len(client.calls) == 1
    return ir, svc, lib, state


# --------------------------------------------------------------------------- the keys


def test_leave_out_is_a_control_key_and_a_requirement_decision() -> None:
    assert LEAVE_OUT_KEY == "leave_out" and LEAVE_OUT_KEY in CONTROL_KEYS and LEAVE_OUT_KEY in REQUIREMENT_DECISION_KEYS
    assert leave_out_answer(["a", "b", "a"]) == "--answer leave_out=a,b" and split_keys(" a, b;a ,, ") == ["a", "b"]
    # nothing is guessed from a key's spelling: only a template key or alias is read
    assert template_reads_key("battery_voltage") and template_reads_key("pcb_layers") and template_reads_key("radio_build")
    assert not any(template_reads_key(k) for k in ("battery_voltage_max", "antenna_switch_present", "frequency", "isolation"))


def test_on_head_the_answer_itself_became_a_requirement_now_it_never_does(tmp_path: Path) -> None:
    """HEAD recorded ``--answer leave_out=x`` as ``req.leave_out`` (electrical), which the closed world refused in turn."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "output_voltage": "5 V",
                                    "power_switch_present": "yes", LEAVE_OUT_KEY: "nothing_here"})
    assert ir.requirements.get(LEAVE_OUT_KEY) is None and ir.requirements.left_out == []


# --------------------------------------------------------------------------- P1: the closed world is leavable


def test_p1_every_closed_world_refusal_names_the_exact_answer_and_the_answer_leaves_it_out(tmp_path: Path) -> None:
    ir, svc, lib, state = _refused_to_confirmed(tmp_path)
    arch = state.outcome(Stage.ARCHITECTURE)
    assert arch.status is S.NOT_VERIFIED and "template divider not proposed" in arch.message
    refused = {q.key: q for q in arch.questions}
    assert set(refused) == {"power_switch_present", "cell_count"} and not any(q.required for q in refused.values())
    for key, q in refused.items():
        assert f"--answer leave_out={key} " in q.question  # the exact answer, one key
        assert "every requirement refused here at once: --answer leave_out=" in q.question and "a later typed answer to" in q.question
    accepted = ir.requirements.get("cell_count")
    assert accepted.kind is RequirementKind.ASSUMPTION and accepted.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    before = ir.content_hash()

    # run 3: the answer the refusal named - the requirements move into left_out, the template presents its table
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count"})
    assert ir.requirements.get("power_switch_present") is None and ir.requirements.get("cell_count") is None
    left = {x.key: x for x in ir.requirements.left_out}
    assert set(left) == {"power_switch_present", "cell_count"}
    assert left["power_switch_present"].requirement.id == "req.power_switch_present" and left["cell_count"].requirement.id == "req.cell_count"
    assert all(x.note == f"left out of the design by the user (--answer leave_out={x.key})" for x in left.values())
    note = state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    assert "power_switch_present: req.power_switch_present (power_switch_present: 전원 스위치 포함) left out of the design by your decision" in note
    assert ir.content_hash() != before and "left_out" in ir.design_dict()["requirements"]  # the user's decision is design content
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    table = state.open_questions[0].question
    assert "Requirements you left out of this design (leave_out;" in table
    assert "  req.power_switch_present: power_switch_present: '전원 스위치 포함' [electrical]" in table and "  req.cell_count: cell_count: 2.0 [electrical]" in table
    # a template never reads a left-out requirement
    inputs, _ = read_inputs(ir)
    assert set(inputs) == {"input_voltage", "output_voltage", "output_current"}

    # run 4: confirm alone - the design is built; the reviewer does not demand the left-out requirements but names them
    state = _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=None)
    assert [c.ref for c in ir.components] == ["R1", "R2", "J1"] and {x.key for x in ir.requirements.left_out} == {"power_switch_present", "cell_count"}
    report = IndependentReviewer(tools={"kicad_library": lib}).review(ir, tmp_path)
    r = {x.check_id: x for x in report.results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.PASS and r.details["left_out"] == ["req.power_switch_present", "req.cell_count"]
    assert r.message.endswith("left out of the design by the user (not demanded): req.power_switch_present, req.cell_count")


def test_a_confirm_design_beside_a_leave_out_is_ignored_like_beside_accept_implicit(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count", CONFIRM_DESIGN_KEY: "yes"})
    arch = state.outcome(Stage.ARCHITECTURE)
    assert ir.components == [] and "confirm_design ignored: answer(s) ['leave_out'] were given in this run" in arch.message
    assert {x.key for x in ir.requirements.left_out} == {"power_switch_present", "cell_count"}  # the leave-out itself is applied
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    # the table shown now lists the leave-outs: confirming it alone builds
    _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    assert [c.ref for c in ir.components] == ["R1", "R2", "J1"]


def test_leave_out_is_idempotent(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present"})
    once = ir.content_hash()
    left = [x.model_dump() for x in ir.requirements.left_out]
    ctx = AgentContext(workdir=tmp_path, llm=svc, answers={LEAVE_OUT_KEY: "power_switch_present"})
    result = RequirementAgent().run(ir, ctx)
    assert not any(p.target in ("requirements.left_out", "requirements.requirements") for p in result.proposals)
    assert "power_switch_present: already left out of the design; nothing changed" in result.notes
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present"})
    assert ir.content_hash() == once and [x.model_dump() for x in ir.requirements.left_out] == left


def test_a_later_typed_answer_brings_the_key_back_and_in_the_same_run_the_leave_out_wins(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    # the same run: the leave-out wins over the typed answer to the same key (noted), nothing is recorded under the key
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count", "power_switch_present": "yes"})
    assert ir.requirements.get("power_switch_present") is None and "power_switch_present" in ir.requirements.left_out_keys
    msg = state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    assert "power_switch_present: answer 'yes' not recorded - you also left power_switch_present out in this run (leave_out wins)" in msg
    # a later run: the typed answer is recorded as usual and the leave-out is withdrawn - the closed world refuses it again
    state = _run(ir, svc, tmp_path, lib, {"power_switch_present": "yes"})
    r = ir.requirements.get("power_switch_present")
    assert r is not None and r.value.value == "yes" and r.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT and r.category == "electrical"
    assert ir.requirements.left_out_keys == {"cell_count"}
    assert "power_switch_present: your typed answer brings power_switch_present back into the design" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    [q] = [q for q in state.outcome(Stage.ARCHITECTURE).questions if q.key == "power_switch_present"]
    assert "--answer leave_out=power_switch_present" in q.question


def test_a_leave_out_never_takes_the_application_the_scope_or_a_control_key_nor_a_requirement_the_design_references(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count"})
    _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    assert ir.components
    before = ir.content_hash()
    ctx = AgentContext(workdir=tmp_path, llm=svc, answers={LEAVE_OUT_KEY: "application,jurisdiction,mains_powered,confirm_design,input_voltage,no_such_key"})
    result = RequirementAgent().run(ir, ctx)
    assert not any(p.target in ("requirements.left_out", "requirements.requirements") for p in result.proposals)
    notes = "\n".join(result.notes)
    for key in ("application", "jurisdiction", "mains_powered", "confirm_design"):
        assert f"{key}: not left out - it is not a design requirement key" in notes
    assert "input_voltage: req.input_voltage is referenced by the design (" in notes and "leaving it out would untrace the design; not left out" in notes
    assert "no_such_key: no requirement and no open question under no_such_key; nothing left out" in notes
    assert design_references(ir, "req.input_voltage") and design_references(ir, "req.power_switch_present") == []
    Orchestrator.apply_proposals(ir, result.proposals)
    assert ir.content_hash() == before


def test_a_requirement_typed_after_the_build_is_leavable_and_the_review_stops_demanding_it(tmp_path: Path) -> None:
    """Not only a template's refusal: the reviewer's 'not traced' FAIL of a requirement added after the build is leavable too."""
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count"})
    _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    _run(ir, svc, tmp_path, lib, {"operating_temperature": "-20..60 °C"})
    reviewer = IndependentReviewer(tools={"kicad_library": lib})
    r = {x.check_id: x for x in reviewer.review(ir, tmp_path).results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.FAIL and r.details["unserved"] == ["req.operating_temperature"]
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "operating_temperature"})
    r = {x.check_id: x for x in reviewer.review(ir, tmp_path).results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is S.PASS and "req.operating_temperature" in r.details["left_out"]


# --------------------------------------------------------------------------- P2: a model question never forces a refusal


def test_p2_a_required_model_question_under_a_key_nothing_reads_is_optional_under_a_template_key_it_stays_required(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    svc, _client = _service([{"structured": CANNED, "usage": USAGE}])
    ir = _ir(tmp_path)
    state = _run(ir, svc, tmp_path, lib, {})
    assert state.blocked and [q.key for q in state.open_questions] == ["output_current", CONFIRM_KEY]  # a template key: still required
    [q] = [q for q in state.optional_questions if q.key == "divider_tolerance_class"]
    assert q.source == "llm" and not q.required and "leave_out=divider_tolerance_class closes the question" in q.rationale
    assert "divider_tolerance_class: the model asked this as required, but no template reads divider_tolerance_class: asked as optional" in (
        state.outcome(Stage.REQUIREMENT_ANALYSIS).message)
    recorded = {m.key: m for m in ir.requirements.missing}
    assert not recorded["divider_tolerance_class"].required and recorded["output_current"].required
    # the extraction's confirmation table warns about the item no template reads, with the answer that leaves it out
    confirm = recorded[CONFIRM_KEY].question
    assert "No template reads these keys" in confirm and "--answer leave_out=power_switch_present,cell_count)" in confirm


def test_p2_the_leave_out_closes_a_question_with_no_requirement_and_a_model_item_cannot_undo_it(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    svc, client = _service([{"structured": CANNED, "usage": USAGE}])
    ir = _ir(tmp_path)
    _run(ir, svc, tmp_path, lib, {})
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "divider_tolerance_class,power_switch_present"})
    left = {x.key: x for x in ir.requirements.left_out}
    assert left["divider_tolerance_class"].requirement is None and left["divider_tolerance_class"].question.source == "llm"
    assert left["power_switch_present"].requirement.value.provenance.kind is ProvenanceKind.LLM_GENERATED  # left out before confirmation
    # the design table's line says whose value it is: a model's, never confirmed (a user-confirmed one carries no such mark)
    assert any(line.startswith("req.power_switch_present: ") and line.endswith("[electrical] (llm_generated: never confirmed by you)")
               for line in left_out_lines(ir))
    assert "divider_tolerance_class" not in {q.key for q in [*state.open_questions, *state.optional_questions]}
    assert "divider_tolerance_class" not in {m.key for m in ir.requirements.missing}
    confirm = next(m for m in ir.requirements.missing if m.key == CONFIRM_KEY).question
    assert "Left out of the design by your decision (leave_out;" in confirm and "power_switch_present: req.power_switch_present" in confirm
    # the extraction runs again on every unconfirmed run and after a correction: the left-out item is not proposed again
    corrected = copy.deepcopy(CANNED)
    svc2, client2 = _service([{"structured": corrected, "usage": USAGE}])
    _run(ir, svc2, tmp_path, lib, {CONFIRM_KEY: "the switch is a toggle switch on the input"})
    assert len(client2.calls) == 1 and ir.requirements.get("power_switch_present") is None
    _run(ir, svc2, tmp_path, lib, {CONFIRM_KEY: "yes", ACCEPT_KEY: "cell_count", "output_current": "0 A", **SCOPE})
    assert ir.requirements.get("power_switch_present") is None and ir.requirements.get("input_voltage").value.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    state = _run(ir, svc2, tmp_path, lib, {LEAVE_OUT_KEY: "cell_count"})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY] and len(client.calls) == 1


def test_p2_an_answered_model_question_is_a_requirement_whose_refusal_is_leavable(tmp_path: Path) -> None:
    """The typed answer is recorded as before (``electrical``, no category guessed from the key); its refusal names the way out."""
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    state = _run(ir, svc, tmp_path, lib, {"divider_tolerance_class": "1 %"})
    r = ir.requirements.get("divider_tolerance_class")
    assert r.category == "electrical" and r.value.value == "1 %" and r.value.provenance.kind is ProvenanceKind.USER_REQUIREMENT and r.value.provenance.tool is None
    [q] = [q for q in state.outcome(Stage.ARCHITECTURE).questions if q.key == "divider_tolerance_class"]
    assert "--answer leave_out=divider_tolerance_class " in q.question
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "divider_tolerance_class,power_switch_present,cell_count"})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert ir.requirements.left_out_keys == {"divider_tolerance_class", "power_switch_present", "cell_count"}


def test_the_checklist_path_without_a_model_leaves_out_and_closes_a_baseline_question(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    base = {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "output_voltage": "5 V"}
    state = _run(ir, None, tmp_path, lib, {**base, "efficiency": "90 %"})
    [q] = [q for q in state.outcome(Stage.ARCHITECTURE).questions if q.key == "efficiency"]
    assert "--answer leave_out=efficiency " in q.question and "every requirement refused here at once" not in q.question  # one key refused
    state = _run(ir, None, tmp_path, lib, {LEAVE_OUT_KEY: "efficiency,protection"})
    left = {x.key: x for x in ir.requirements.left_out}
    assert left["efficiency"].requirement.id == "req.efficiency" and left["protection"].requirement is None and left["protection"].question.source == "system"
    assert "protection" not in {q.key for q in state.optional_questions}  # a baseline question closed
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]


# --------------------------------------------------------------------------- the record: hash, views


def test_an_empty_left_out_keeps_every_older_hash_and_a_non_empty_one_is_design_content(tmp_path: Path) -> None:
    for name in ("ir_before_rf.json", "ir_before_kr447.json"):
        ir = CircuitIR.load(DATA / name)
        assert ir.requirements.left_out == [] and "left_out" not in ir.design_dict()["requirements"]
    ir = _ir(tmp_path)
    ir.requirements.requirements.append(_answer_requirement("efficiency", "90 %"))
    empty = ir.content_hash()
    moved = ir.requirements.requirements.pop()
    ir.requirements.left_out.append(LeftOutRequirement(key="efficiency", requirement=moved, note="left out"))
    assert ir.content_hash() != empty and ir.design_dict()["requirements"]["left_out"][0]["requirement"]["id"] == "req.efficiency"
    path = ir.save(tmp_path / "ir.json")
    again = CircuitIR.load(path)
    assert again.content_hash() == ir.content_hash() and again.requirements.left_out_keys == {"efficiency"}


def test_report_html_stage_reports_and_gui_data_list_the_left_out_requirements(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count"})
    _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    path = ir.save(tmp_path / "ir.json")
    loaded, sha = load_ir_file(path)
    data = build_report_data(loaded, path, tmp_path, ir_sha=sha)
    rows = {x.key: x for x in data.requirements.left_out}
    assert rows["power_switch_present"].id == "req.power_switch_present" and rows["cell_count"].category == "electrical"
    assert json.loads(data.model_dump_json())["requirements"]["left_out"][0]["key"] == "power_switch_present"  # what the GUI copies
    html = render_html(data)
    assert "left out of the design (the user&#x27;s decision: --answer leave_out" in html  # esc() escapes quotes
    assert "req.power_switch_present" in html
    final = final_report(loaded, None)
    assert "### 설계에서 뺀 요구사항 (사용자 결정)" in final and "`req.power_switch_present`" in final and "`req.cell_count`" in final
    assert "- 설계에서 뺀 요구사항 (사용자 결정, `--answer leave_out`): `req.power_switch_present`, `req.cell_count`" in theory_report(loaded, lib)
    # the GUI renders the rows the report data carries (copied, never computed) and names the answer beside the optional questions
    from ai_eda.gui.page import APP_JS

    assert "d.report.requirements.left_out" in APP_JS and "code('leave_out=<키>')" in APP_JS
    # a design without leave-outs reports exactly as before: no section, no header line
    plain = CircuitIR.model_validate(loaded.model_dump())
    plain.requirements.left_out = []
    assert "설계에서 뺀 요구사항" not in final_report(plain, None) and "설계에서 뺀 요구사항" not in theory_report(plain, lib)


# --------------------------------------------------------------------------- the radio family's closed world


def test_the_family_names_the_answer_and_no_fm_reason_for_a_key_that_is_no_radio_key(tmp_path: Path) -> None:
    assert family.unserved_message("antenna_switch_present", "transceiver") == (
        "antenna_switch_present is not served by radio_build=transceiver; antenna_switch_present is served by no radio_build of the KR 447 MHz family")
    assert "FM telephony" in family.unserved_message("modulation_depth")
    ir = CircuitIR(project=ProjectMeta(id="trx", name="trx", workdir=str(tmp_path)))
    for key, value in {"radio_build": "transceiver", "modulation": "fm", "input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz",
                       "antenna_switch_present": "안테나 스위치", "battery_voltage_max": "8.4 V"}.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    inputs, unusable = read_inputs(ir)
    questions = {q.key: q for q in KR447_TRANSCEIVER.refusals(ir, inputs, unusable)}
    assert set(questions) == {"antenna_switch_present", "battery_voltage_max"}
    for key, q in questions.items():
        assert f"Leave it out of the transceiver board with --answer leave_out={key} " in q.question
        assert "--answer leave_out=antenna_switch_present,battery_voltage_max" in q.question and "FM telephony" not in q.question
    ir.requirements.requirements.append(_answer_requirement("tx_power", "0.5 W"))
    ir.requirements.requirements[0] = _answer_requirement("radio_build", "rx_backend")
    from ai_eda.design.rf.t_rx_backend import KR447RxBackendTemplate

    inputs, unusable = read_inputs(ir)
    [q] = [q for q in KR447RxBackendTemplate().refusals(ir, inputs, unusable) if q.key == "tx_power"]
    assert "--answer leave_out=tx_power " in q.question and "or start the project of radio_build=tx_exciter, transceiver or transceiver_conducted" in q.question


# --------------------------------------------------------------------------- real-library twin: the demo on kr447_transceiver

from tests.test_kr447_transceiver import needs_libs  # noqa: E402  (the same gate as the transceiver's own library tests)

#: the demo's request (project kr447_walkie_talkie) and the shapes the model returned for it
RAW_KR = ("한국에서 면허 없이 쓸 수 있는 447 MHz 대역 생활무전기(FM)를 설계해 주세요. 송신부, 수신부, 안테나 스위치, 전원이 모두 들어간 한 장짜리 "
          "송수신기 보드이고, 채널은 447.5625 MHz, 전원은 7.4 V 2셀 리튬이온 배터리(만충 8.4 V), 안테나는 보드에 붙는 일체형입니다.")
CANNED_KR: dict[str, Any] = {
    "requirements": [
        {"key": "modulation", "text": "The radio uses FM", "kind": "explicit", "category": "electrical", "quote": "생활무전기(FM)", "value": None, "rationale": None},
        _req("carrier_frequency", "The channel is 447.5625 MHz", "채널은 447.5625 MHz", 447.5625, "MHz"),
        {"key": "antenna_switch_present", "text": "An antenna switch is included", "kind": "explicit", "category": "electrical", "quote": "안테나 스위치",
         "value": None, "rationale": None},
        _req("battery_voltage_max", "The battery's full charge is 8.4 V", "만충 8.4 V", 8.4, "V"),
        {"key": "antenna_integration", "text": "The antenna is integral", "kind": "explicit", "category": "mechanical",
         "quote": "안테나는 보드에 붙는 일체형입니다", "value": None, "rationale": None},
    ],
    "questions": [
        {"key": "frequency_deviation_or_occupied_bandwidth", "question": "What peak deviation or occupied bandwidth is required?", "required": True,
         "options": [], "rationale": "the voice channel"},
    ],
    "conflicts": [],
    "assumptions": [
        {"key": "battery_voltage", "text": "the pack is 7.4 V", "category": "electrical", "number": 7.4, "unit": "V", "number_high": None,
         "rationale": "the quote holds other numbers besides 7.4 V"},
        {"key": "battery_cell_count", "text": "two cells", "category": "electrical", "number": 2, "unit": "cell", "number_high": None, "rationale": "2셀"},
    ],
    "application": {"summary": "licence-free 447 MHz FM walkie-talkie", "quote": "447 MHz 대역 생활무전기(FM)를 설계해 주세요"},
    "jurisdictions": [{"code": "KR", "quote": "한국에서 면허 없이"}],
}
KR_SCOPE = {"mains_powered": "no", "radio": "yes", "digital_device": "yes", "kr_licence_free_class": "yes", "highest_rated_voltage": "8.4 V DC"}
LEFT_KR = ("antenna_switch_present", "battery_voltage_max", "antenna_integration", "battery_cell_count", "frequency_deviation_or_occupied_bandwidth")


@needs_libs
def test_the_demo_on_the_real_transceiver_is_leavable_and_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1 + P2 on ``kr447_transceiver`` with the real libraries: refused with the exact answer, left out, presented, confirmed, built."""
    import ai_eda.design.templates as templates_mod
    from ai_eda.tools.kicad.library import KicadLibrary

    monkeypatch.setattr(templates_mod, "rf_templates", lambda module=templates_mod.RF_REGISTRY_MODULE: [KR447_TRANSCEIVER])
    lib = KicadLibrary()
    svc, client = _service([{"structured": CANNED_KR, "usage": USAGE}])
    ir = _ir(tmp_path, raw=RAW_KR, name="walkie")
    state = _run(ir, svc, tmp_path, lib, {})
    # P2: the model's required question under a key nothing reads does not block; only the extraction's confirmation does
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_KEY]
    assert not next(q for q in state.optional_questions if q.key == "frequency_deviation_or_occupied_bandwidth").required
    # the demo's run 2: confirm, accept both assumptions, name the build, answer the model's question - the closed world refuses
    state = _run(ir, svc, tmp_path, lib, {CONFIRM_KEY: "yes", ACCEPT_KEY: "battery_voltage,battery_cell_count", "radio_build": "transceiver",
                                          "frequency_deviation_or_occupied_bandwidth": "2.5 kHz peak", **KR_SCOPE})
    arch = state.outcome(Stage.ARCHITECTURE)
    refused = {q.key: q for q in arch.questions}
    assert set(refused) == set(LEFT_KR) and ir.components == []
    for key, q in refused.items():
        assert f"Leave it out of the transceiver board with --answer leave_out={key} " in q.question
        assert f"--answer leave_out={','.join(q2 for q2 in refused)}" in q.question
        assert q.answer_key == LEAVE_OUT_KEY  # the report and the GUI offer the leave-out, not a value under the refused key
    assert "FM telephony" not in refused["antenna_switch_present"].rationale
    # the answer the refusal named
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: ",".join(LEFT_KR)})
    assert ir.requirements.left_out_keys == set(LEFT_KR) and state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    table = state.open_questions[0].question
    assert table.startswith("Template 'kr447_transceiver'") and "Requirements you left out of this design (leave_out;" in table
    for key in LEFT_KR:
        assert f"  req.{key}: {key}" in table
    state = _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    assert len(ir.components) == 441 and len(client.calls) == 1 and ir.requirements.left_out_keys == set(LEFT_KR)


# --------------------------------------------------------------------------- the CLI and questions an older run recorded


def _cli(*argv: str) -> tuple[int, str, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from ai_eda.cli import main as cli_main

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_the_cli_names_the_answer_and_takes_it(tmp_path: Path) -> None:
    from ai_eda.cli import RUN_OPTIONS

    assert "leave_out=k1,k2 leaves those requirements out of the design" in next(o.help for o in RUN_OPTIONS if o.flag == "--answer")
    code, _out, _ = _cli("new", "lo", "--dir", str(tmp_path / "lo"))
    assert code == 0
    project = str(tmp_path / "lo" / "ir.json")
    answers = ["application=bench", "jurisdiction=KR", "input_voltage=12 V", "output_voltage=5 V", "efficiency=90 %", "mains_powered=no", "radio=no"]
    code, out, _ = _cli("run", project, *[a for kv in answers for a in ("--answer", kv)])
    assert "[efficiency] req.efficiency (efficiency: '90 %') is not served by the unloaded resistive voltage divider template" in out
    assert "Leave it out of this design with --answer leave_out=efficiency " in out
    _cli("run", project, "--answer", "leave_out=efficiency")
    ir = CircuitIR.load(project)
    assert ir.requirements.get("efficiency") is None and ir.requirements.left_out_keys == {"efficiency"}
    assert ir.requirements.get(LEAVE_OUT_KEY) is None


def test_a_required_model_question_an_older_run_recorded_is_brought_up_to_date_so_every_reader_agrees(tmp_path: Path) -> None:
    """IRs saved before the rule hold the model's required question as it was; a run without a model brings the recorded list up to date.

    Only the extraction path rewrote ``requirements.missing`` before, so on a
    run without a model the gate (which applies the rule), the reviewer
    (``blocking_questions``) and ``ai-eda report`` disagreed: the pipeline
    passed the gate and then blocked at INDEPENDENT_REVIEW naming no
    question, and the report offered ``--answer <key>=...`` for a question a
    leave-out had closed - which brings the key back.
    """
    from ai_eda.ir import MissingInformation

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="12V 입력에서 5V 출력 분압기")
    ir.requirements.missing = [
        MissingInformation(key="regulatory_certification", question="Which certification?", required=True, source="llm"),
        MissingInformation(key="isolation_rating", question="Which isolation rating?", required=True, source="system"),
    ]
    answers = {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "output_voltage": "5 V", **SCOPE}
    state = _run(ir, None, tmp_path, lib, answers)
    gate = state.outcome(Stage.MISSING_INFORMATION)
    assert [q.key for q in gate.questions] == ["isolation_rating"]  # the model's question under a key nothing reads does not block
    recorded = {m.key: m for m in ir.requirements.missing}
    assert not recorded["regulatory_certification"].required and recorded["regulatory_certification"].source == "llm"
    assert "--answer leave_out=regulatory_certification closes the question" in recorded["regulatory_certification"].rationale
    assert "regulatory_certification: the model asked this as required, but no template reads regulatory_certification: recorded as optional" in (
        state.outcome(Stage.REQUIREMENT_ANALYSIS).message)
    # the reviewer's gate reads the same list the pipeline gate does
    assert [q.key for q in ir.requirements.blocking_questions] == ["isolation_rating"]
    state = _run(ir, None, tmp_path, lib, {LEAVE_OUT_KEY: "isolation_rating"})
    left = {x.key: x for x in ir.requirements.left_out}
    assert left["isolation_rating"].requirement is None and left["isolation_rating"].question.question == "Which isolation rating?"
    assert "isolation_rating" not in {m.key for m in ir.requirements.missing}  # closed: no longer recorded as open
    assert ir.requirements.blocking_questions == [] and ir.requirements.can_proceed
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    # the report offers no answer that would bring a left-out key back
    path = ir.save(tmp_path / "ir.json")
    loaded, sha = load_ir_file(path)
    rows = {q.key: q for q in build_report_data(loaded, path, tmp_path, ir_sha=sha).questions}
    assert "isolation_rating" not in rows and not rows["regulatory_certification"].required
    _run(ir, None, tmp_path, lib, {LEAVE_OUT_KEY: "regulatory_certification"})
    assert {m.key for m in ir.requirements.missing} == set() and ir.requirements.left_out_keys == {"isolation_rating", "regulatory_certification"}


def test_a_stale_model_question_no_longer_blocks_the_review_of_a_full_run(tmp_path: Path) -> None:
    """Before: the gate passed, the design was built, and INDEPENDENT_REVIEW blocked with USER_INPUT_REQUIRED naming no question."""
    from ai_eda.ir import MissingInformation

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="12V 입력에서 5V 출력 분압기")
    ir.requirements.missing = [MissingInformation(key="regulatory_certification", question="Which certification?", required=True, source="llm")]
    _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "output_voltage": "5 V", **SCOPE})
    state = _run(ir, None, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"}, stop_after=None)
    assert ir.components and not state.blocked
    review = state.outcome(Stage.INDEPENDENT_REVIEW)
    assert review.status is not S.USER_INPUT_REQUIRED and review.questions == []
    assert ir.validation.latest(ReviewArea.REQUIREMENTS_VS_IR).status is not S.USER_INPUT_REQUIRED


def test_a_run_without_a_model_closes_the_recorded_questions_it_answers_and_nothing_else(tmp_path: Path) -> None:
    """A typed answer closes the recorded question under its key; a control key's question stays; an IR with nothing stale is not rewritten."""
    from ai_eda.ir import MissingInformation

    ir = _ir(tmp_path, raw="")
    ir.requirements.missing = [
        MissingInformation(key="isolation_rating", question="Which isolation rating?", required=True),
        MissingInformation(key="operating_temperature", question="What range?", required=False),
        MissingInformation(key=CONFIRM_KEY, question="Confirm the table?", required=True),
    ]
    before = ir.content_hash()
    result = RequirementAgent().run(ir, AgentContext(workdir=tmp_path, answers={}))
    assert not any(p.target == "requirements.missing" for p in result.proposals)  # nothing stale: the recorded list is kept byte for byte
    Orchestrator.apply_proposals(ir, result.proposals)
    assert ir.content_hash() == before
    result = RequirementAgent().run(ir, AgentContext(workdir=tmp_path, answers={"isolation_rating": "1 kV", CONFIRM_KEY: "yes"}))
    Orchestrator.apply_proposals(ir, result.proposals)
    # the answered question is closed; confirm_requirements is a control answer that confirms nothing without the extraction: still open
    assert [m.key for m in ir.requirements.missing] == ["operating_temperature", CONFIRM_KEY]
    assert ir.requirements.get("isolation_rating").value.value == "1 kV"




# --------------------------------------------------------------------------- the review round: labels, answers named everywhere


def test_a_model_question_closed_by_a_leave_out_is_labelled_in_every_view(tmp_path: Path) -> None:
    """A model-authored question is labelled wherever it is shown (``MissingInformation``): the left-out views too."""
    lib = template_library(tmp_path / "kicad")
    svc, _client = _service([{"structured": CANNED, "usage": USAGE}])
    ir = _ir(tmp_path)
    _run(ir, svc, tmp_path, lib, {})
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "divider_tolerance_class"})
    [x] = ir.requirements.left_out
    assert x.requirement is None and x.question.source == "llm"
    path = ir.save(tmp_path / "ir.json")
    loaded, sha = load_ir_file(path)
    data = build_report_data(loaded, path, tmp_path, ir_sha=sha)
    [row] = data.requirements.left_out
    assert row.question_source == "llm"
    assert ('<span>Which resistor tolerance class must the divider use?</span><span class="warn"> (model output)</span>' in render_html(data))
    assert "(요구사항 없음; 닫힌 질문 (모델 질문): Which resistor tolerance class must the divider use?)" in final_report(loaded, None)
    assert "the model's open question 'Which resistor tolerance class" in "\n".join(left_out_lines(loaded))


def test_a_question_answered_before_the_leave_out_is_not_recorded_as_the_one_it_closed(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {"divider_tolerance_class": "1 %"})
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "divider_tolerance_class"})
    x = next(x for x in ir.requirements.left_out if x.key == "divider_tolerance_class")
    assert x.requirement.id == "req.divider_tolerance_class" and x.question is None  # answered a run earlier: not the question the leave-out closed
    path = ir.save(tmp_path / "ir.json")
    loaded, sha = load_ir_file(path)
    [row] = [r for r in build_report_data(loaded, path, tmp_path, ir_sha=sha).requirements.left_out if r.key == "divider_tolerance_class"]
    assert row.question == "" and row.question_source == ""


def test_the_model_note_names_a_demoted_question_only_while_it_is_asked(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    svc, _client = _service([{"structured": CANNED, "usage": USAGE}])
    ir = _ir(tmp_path)
    note = "divider_tolerance_class: the model asked this as required"
    state = _run(ir, svc, tmp_path, lib, {})
    assert note in state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    state = _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "divider_tolerance_class"})
    assert note not in state.outcome(Stage.REQUIREMENT_ANALYSIS).message  # left out: not asked, and the note would name an answer already given
    state = _run(ir, svc, tmp_path, lib, {})
    assert note not in state.outcome(Stage.REQUIREMENT_ANALYSIS).message
    # answered (brought back by the typed answer): not asked either
    state = _run(ir, svc, tmp_path, lib, {"divider_tolerance_class": "1 %"})
    assert ir.requirements.get("divider_tolerance_class") is not None and note not in state.outcome(Stage.REQUIREMENT_ANALYSIS).message


def test_a_refusal_is_answered_by_leave_out_in_the_report_and_the_gui_other_questions_keep_their_value(tmp_path: Path) -> None:
    from ai_eda.design.board import layer_policy_refusal
    from ai_eda.design.templates import DividerTemplate
    from ai_eda.ir import MissingInformation
    from ai_eda.report.data import answer_command

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    state = _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "output_voltage": "5 V",
                                           "efficiency": "90 %", "mains_powered": "no", "radio": "no"})
    [q] = [q for q in state.outcome(Stage.ARCHITECTURE).questions if q.key == "efficiency"]
    assert q.answer_key == LEAVE_OUT_KEY and answer_command(Path("p/ir.json"), q) == "ai-eda run p/ir.json --answer leave_out=efficiency"
    plain = MissingInformation(key="operating_temperature", question="range?", required=False)
    assert answer_command(Path("p/ir.json"), plain) == "ai-eda run p/ir.json --answer operating_temperature=<value>"
    # the divider's load refusal (a stated load) and the kr447 family's refusals are marked too; the load *question* is not
    divider = DividerTemplate()
    assert divider._load_question("why", "req.output_current").answer_key == LEAVE_OUT_KEY and divider._load_question("why").answer_key is None
    trx = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    for key, value in {"radio_build": "transceiver", "modulation": "fm", "antenna_switch_present": "yes", "pcb_layers": "2"}.items():
        trx.requirements.requirements.append(_answer_requirement(key, value))
    inputs, unusable = read_inputs(trx)
    [fam] = KR447_TRANSCEIVER.refusals(trx, inputs, unusable)
    assert fam.key == "antenna_switch_present" and fam.answer_key == LEAVE_OUT_KEY
    layer = layer_policy_refusal(KR447_TRANSCEIVER, trx)
    assert layer is not None and layer.key == "pcb_layers" and layer.answer_key == LEAVE_OUT_KEY
    # the field is design content only when set: an unmarked question hashes as before
    assert "answer_key" not in plain.model_dump(mode="json", context={"view": "design"}) and "answer_key" in fam.model_dump(mode="json", context={"view": "design"})
    # the GUI offers the leave-out itself for a marked question and sends every ticked key (and a leave_out= line) as ONE answer
    from ai_eda.gui.page import APP_JS

    assert "q.answer_key === leaveKey" in APP_JS and "'data-leave-out': q.key" in APP_JS
    assert "answers[leaveKey] = [...new Set(leave)].join(',')" in APP_JS and "if (leaveKey && key === leaveKey) leave.push(" in APP_JS


def test_the_not_traced_fail_names_the_requirement_and_the_answer_in_the_run_that_shows_it(tmp_path: Path) -> None:
    ir, svc, lib, _state = _refused_to_confirmed(tmp_path)
    _run(ir, svc, tmp_path, lib, {LEAVE_OUT_KEY: "power_switch_present,cell_count"})
    _run(ir, svc, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    state = _run(ir, svc, tmp_path, lib, {"operating_temperature": "-20..60 °C"}, stop_after=None)
    circuit = state.outcome(Stage.ARCHITECTURE).message
    assert ("confirmed design requirement(s) no component or net serves: req.operating_temperature (the reviewer reports them as not traced); "
            "to leave them out of the design: --answer leave_out=operating_temperature") in circuit
    r = ir.validation.latest(ReviewArea.REQUIREMENTS_VS_IR)
    assert r.status is S.FAIL and r.details["leave_out_keys"] == ["operating_temperature"]
    assert r.message.startswith("requirements not traced to any component: req.operating_temperature; to leave them out of the design: "
                                "--answer leave_out=operating_temperature")
    # the run's own output names it (the review stage's message, not only its counts)
    assert f"review.requirements_vs_ir FAIL: {r.message}" in state.outcome(Stage.INDEPENDENT_REVIEW).message


def test_a_requirement_under_another_alias_with_the_same_number_is_traced_and_listed(tmp_path: Path) -> None:
    """``req.battery_voltage`` beside ``req.input_voltage``: the closed world counts it as served, so the build traces it and the table lists it."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    state = _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "battery_voltage": "12 V",
                                           "output_voltage": "5 V", **SCOPE})
    table = state.open_questions[0].question
    assert ("  req.battery_voltage: input_voltage = 12 V (stated as '12 V'; the same number as req.input_voltage, served by the parts serving it)"
            in table)
    _run(ir, None, tmp_path, lib, {CONFIRM_DESIGN_KEY: "yes"})
    serving = [c.ref for c in ir.components if "req.input_voltage" in c.serves_requirements]
    assert serving and all("req.battery_voltage" in c.serves_requirements for c in ir.components if c.ref in serving)
    r = {x.check_id: x for x in IndependentReviewer(tools={"kicad_library": lib}).review(ir, tmp_path).results}[ReviewArea.REQUIREMENTS_VS_IR]
    assert r.status is not S.FAIL and "req.battery_voltage" not in r.details.get("unserved", [])


def test_two_templates_matching_name_one_leave_out_per_circuit_and_it_builds(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    state = _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "5 V", "output_voltage": "3 V",
                                           "led_forward_voltage": "2 V", "led_forward_current": "10 mA", **SCOPE})
    msg = state.outcome(Stage.ARCHITECTURE).message
    assert "refusing to guess - state the requirements of one circuit" in msg
    assert "to build divider: --answer leave_out=led_forward_voltage,led_forward_current" in msg and "to build led: --answer leave_out=output_voltage" in msg
    state = _run(ir, None, tmp_path, lib, {LEAVE_OUT_KEY: "output_voltage"})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert state.open_questions[0].question.startswith("Template 'led'")


def test_an_alias_disagreement_names_the_leave_out_of_each_side_and_it_builds(tmp_path: Path) -> None:
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, raw="")
    state = _run(ir, None, tmp_path, lib, {"application": "bench", "jurisdiction": "KR", "input_voltage": "12 V", "battery_voltage": "9 V",
                                           "output_voltage": "5 V", **SCOPE})
    msg = state.outcome(Stage.ARCHITECTURE).message
    assert ("input_voltage not usable: ambiguous: req.input_voltage says 12 V, req.battery_voltage says 9 V; leave one of them out: "
            "--answer leave_out=input_voltage (keeps req.battery_voltage) or --answer leave_out=battery_voltage (keeps req.input_voltage)") in msg
    state = _run(ir, None, tmp_path, lib, {"input_voltage": "9 V"})  # a typed value is kept: re-typing never clears it
    assert "not applied" in state.outcome(Stage.REQUIREMENT_ANALYSIS).message and state.outcome(Stage.ARCHITECTURE).message.count("not usable") >= 1
    state = _run(ir, None, tmp_path, lib, {LEAVE_OUT_KEY: "battery_voltage"})
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    assert "req.input_voltage: input_voltage = 12 V" in state.open_questions[0].question


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_the_gui_sends_every_ticked_leave_out_and_the_extra_line_as_one_answer(tmp_path: Path) -> None:
    """``collectAnswers`` of app.js (run in node on stub elements): several refusal fields and a ``leave_out=`` line merge into one answer."""
    import re
    import subprocess

    from ai_eda.gui.page import APP_JS

    function = re.search(r"function collectAnswers\(\) \{.*?\n\}", APP_JS, re.S).group(0)
    script = tmp_path / "collect.js"
    script.write_text(
        "const state = {data: {answers_rule: {leave_out_key: 'leave_out'}}};\n"
        "const fields = {'[data-answer]': [{value: ' 12 V ', dataset: {answer: 'input_voltage'}}],\n"
        "  '[data-leave-out]': [{checked: true, dataset: {leaveOut: 'antenna_switch_present'}}, {checked: false, dataset: {leaveOut: 'x'}},\n"
        "                       {checked: true, dataset: {leaveOut: 'battery_voltage_max'}}]};\n"
        "const document = {querySelectorAll: (sel) => fields[sel] || []};\n"
        "const $ = () => ({value: 'leave_out=battery_cell_count, antenna_switch_present\\nmains_powered=no'});\n"
        f"{function}\nconsole.log(JSON.stringify(collectAnswers()));\n", encoding="utf-8")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"input_voltage": "12 V", "mains_powered": "no",
                                         "leave_out": "antenna_switch_present,battery_voltage_max,battery_cell_count"}
