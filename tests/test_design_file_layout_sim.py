"""A design file's simulation-only elements and nodes, and its board layout (``simulation.elements`` / ``simulation.nodes``, ``board``).

A circuit no built-in template covers - a push button that switches an RC -
is fully described by a design file:

* the button stays an excluded part; a voltage-controlled switch across its
  nets (a simulation-only ``S`` element) is pressed by a PWL stimulus on a
  private node, and two window means of one transient judge the pressed and
  the released state;
* the ``board`` section fixes the button and the connector, packs C1 and R1
  into a region in that order and puts a text beside the button: the
  ``placement.fixed`` placer keeps them, the silkscreen placer places the
  references around the text, which survives the silkscreen stage.

Every new field is checked at load, each refusal naming the field; a file
without them loads, plans and hashes as before (the fingerprints of
``tests/test_design_file_si.py`` pin that, unchanged).

The library is ``tests/test_circuit_templates.py``'s synthetic one plus a
synthetic two-pin push button (``Switch:SW_Push`` on a two-pad through-hole
fixture footprint): pin numbers and pads are read from those files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.agents.keys import ROUTING_KEY
from ai_eda.compilers.spice import build, model_card_hash
from ai_eda.design.base import CONFIRM_DESIGN_KEY
from ai_eda.design.board import design_file_silk_text, proposed_silk_text, template_silk_text
from ai_eda.design.design_file import DesignFileError, load_design_file
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, PCBDesign, ProvenanceKind, SilkKind, SpiceDevice, ValidationStatus
from ai_eda.ir.provenance import design_data
from ai_eda.ir.simulation import MODEL_CARD_NO_SOURCE
from ai_eda.report.stages import _silkscreen_section
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.placement.fixed import FIXED, FREE_AREA, fixed_placement
from ai_eda.tools.silkscreen.place import PLACER_KEEP_VERSION, stroke_for
from ai_eda.tools.spice import NgspiceShared
from ai_eda.validation.layout import SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK
from ai_eda.workflow import Orchestrator
from ai_eda.workflow.stages import Stage
from tests.test_circuit_templates import BASE, _footprint, _ir, _lib, _pin, _symbol, template_library
from tests.test_silkscreen import _checks as silk_checks

runner = NgspiceShared()
needs_dll = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

#: the switch of tests/test_spice_switch.py (a fixture, not a part's datasheet values): 0.1 ohm closed, 1e8 ohm open, closes above
#: VT + VH = 0.6 V and opens below VT - VH = 0.4 V of its control
SW_CARD = ".model SWK SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)"
SW_SOURCE = "the fixture switch of tests/test_spice_switch.py"
#: the press schedule on the private node PRESS: 0 V (open) until 1 ms, 1 V (closed) from 1.001 ms to 3 ms, 0 V again from 3.001 ms
PRESS = [[0.0, 0.0], [1e-3, 0.0], [1.001e-3, 1.0], [3e-3, 1.0], [3.001e-3, 0.0], [6e-3, 0.0]]
#: v(OUT) pressed: 5 V across R1 + RON in series with R2 = 10 k (C1 charged; tau = (R1 + RON) || R2 x 10 nF = 9.1 us, the window starts 55 tau
#: after the press); released: C1 discharges through R2 (tau 100 us, the window starts 15 tau after the release) to the 5 V x R2 / ROFF leak
V_PRESSED = 5.0 * 10_000.0 / (1_000.0 + 10_000.0 + 0.1)


def _part(ref: str, symbol: str, footprint: str, spice: dict, value: str = "x") -> dict:
    serves = {} if spice.get("exclude") else {"serves_requirements": ["req.key"]}
    return {"ref": ref, "value": value, "symbol": symbol, "footprint": footprint, **serves, "spice": spice}


#: J1 (power in) -> SW1 (the push button) -> R1 1 k -> OUT; C1 10 nF and R2 10 k from OUT to ground
KEY_RC: dict = {
    "format": "ai-eda-design/1",
    "title": "Key-switched RC",
    "topology": {"name": "key-switched rc", "domains": ["analog"]},
    "components": [
        _part("J1", "Connector_Generic:Conn_01x02", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", {"exclude": True, "exclude_reason": "connector"},
              "Conn_01x02"),
        _part("SW1", "Switch:SW_Push", "Button_Switch_THT:SW_Fixture_2pin",
              {"exclude": True, "exclude_reason": "push button: the simulation-only switch K1 across its nets presses it"}, "SW_Push"),
        _part("R1", "Device:R", "Resistor_SMD:R_0603_1608Metric", {"device": "R", "value": {"value": 1000, "unit": "ohm"}}, "1k"),
        _part("R2", "Device:R", "Resistor_SMD:R_0603_1608Metric", {"device": "R", "value": {"value": 10000, "unit": "ohm"}}, "10k"),
        _part("C1", "Device:C", "Capacitor_SMD:C_0603_1608Metric", {"device": "C", "value": {"value": 1e-8, "unit": "F"}}, "10n"),
    ],
    "nets": [
        {"name": "VIN", "pins": ["J1.1", "SW1.1"]},
        {"name": "IN", "pins": ["SW1.2", "R1.1"]},
        {"name": "OUT", "pins": ["R1.2", "C1.1", "R2.1"]},
        {"name": "GND", "kind": "ground", "pins": ["J1.2", "C1.2", "R2.2"]},
    ],
    "simulation": {
        "nodes": ["PRESS"],
        "elements": [{"id": "K1", "device": "S", "net": "VIN", "reference_net": "IN", "control_net": "PRESS", "control_reference_net": "GND",
                      "model_name": "SWK", "model_card": SW_CARD, "model_source": SW_SOURCE, "reason": "SW1, pressed by the schedule on PRESS"}],
        "stimuli": [{"id": "VIN", "source": "voltage", "net": "VIN", "reference_net": "GND", "kind": "dc", "value": {"value": 5, "unit": "V"}},
                    {"id": "PRESS", "source": "voltage", "net": "PRESS", "reference_net": "GND", "kind": "pwl", "params": {"points": PRESS}}],
        "analyses": [{"id": "tran", "kind": "tran", "params": {"step": 1e-6, "stop": 6e-3}}],
        "expectations": [
            {"id": "v_pressed", "analysis_id": "tran", "vector": "v(OUT)", "reduce": "mean", "params": {"t_start": 1.5e-3, "t_stop": 2.9e-3},
             "nominal": {"value": V_PRESSED, "unit": "V"}, "tol_rel": 0.01},
            {"id": "v_released", "analysis_id": "tran", "vector": "v(OUT)", "reduce": "mean", "params": {"t_start": 4.5e-3, "t_stop": 6e-3},
             "nominal": {"value": 0, "unit": "V"}, "tol_abs": 0.01},
        ],
    },
    "board": {
        "outline": {"width_mm": 30, "height_mm": 20},
        "fixed": [{"ref": "SW1", "x_mm": 8, "y_mm": 15}, {"ref": "J1", "x_mm": 24, "y_mm": 15, "rotation_deg": 90}],
        "pack_regions": [{"id": "rc", "rect_mm": [2, 2, 16, 8], "refs": ["C1", "R1"]}],
        "silk_texts": [{"text": "KEY", "x_mm": 9.27, "y_mm": 17.8, "ref": "SW1"}],
    },
}


def key_library(root: Path) -> KicadLibrary:
    """The template fixture library plus ``Switch:SW_Push`` (pins 1, 2) on ``Button_Switch_THT:SW_Fixture_2pin`` (two through-hole pads)."""
    template_library(root)
    button = _symbol("SW_Push", "SW", [_pin("1", "1", -5.08, 0), _pin("2", "2", 5.08, 180)], "Button_Switch_THT:SW_Fixture_2pin")
    (root / "symbols" / "Switch.kicad_sym").write_text(sexpr.dumps(_lib(button)), encoding="utf-8")
    pretty = root / "footprints" / "Button_Switch_THT.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    sexpr.dump_file(_footprint("SW_Fixture_2pin", 2, True), pretty / "SW_Fixture_2pin.kicad_mod")
    return KicadLibrary(roots=[root])


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return key_library(tmp_path / "kicad")


def _write(tmp_path: Path, data: dict = KEY_RC) -> Path:
    path = tmp_path / "design.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _edited(edit) -> dict:
    data = copy.deepcopy(KEY_RC)
    edit(data)
    return data


def _run(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary, plan, answers: dict[str, str], stop_after: Stage = Stage.ARCHITECTURE):
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": lib, "design_file": plan, "spice": runner}, answers=answers)
    return Orchestrator(ctx).run(ir, stop_after=stop_after)


def _applied(tmp_path: Path, lib: KicadLibrary, ir: CircuitIR | None = None, data: dict = KEY_RC):
    plan = load_design_file(_write(tmp_path, data), lib)
    ir = ir if ir is not None else _ir(tmp_path)
    state = _run(ir, tmp_path, lib, plan, BASE)
    assert state.blocked and [q.key for q in state.open_questions] == [CONFIRM_DESIGN_KEY]
    table = state.open_questions[0].question
    _run(ir, tmp_path, lib, plan, {CONFIRM_DESIGN_KEY: "yes"})
    return ir, plan, table


# --------------------------------------------------------------------------- loading, the table, applying


def test_the_file_loads_and_the_table_shows_the_switch_its_node_and_the_board(tmp_path: Path, lib: KicadLibrary):
    plan = load_design_file(_write(tmp_path), lib)
    table = plan.table()
    sha = model_card_hash(SW_CARD)
    for text in (
        "simulation-only node(s): PRESS (no net of the board)",
        f"simulation-only element K1: S between VIN and IN, closed by v(PRESS) - v(GND) above VT + VH: model SWK model_card {SW_CARD!r} "
        f"({sha}, from {SW_SOURCE} as you state it, not fetched) - SW1, pressed by the schedule on PRESS",
        "stimulus PRESS: voltage pwl on PRESS against GND",
        "expectation v_pressed: mean of v(OUT) in tran [t_start=0.0015, t_stop=0.0029]",
        "Board layout (the file's board section, kept as written:",
        "  board outline: 30 x 20 mm at (0, 0) mm",
        "  fixed part SW1: at (8, 15) mm, rotation 0 deg, top side",
        "  fixed part J1: at (24, 15) mm, rotation 90 deg, top side",
        "  pack region rc: rect x 2 y 2 w 16 h 8 mm, top side, packed in this order: C1, R1",
        "  silk text 'KEY' (user): at (9.27, 17.8) mm on F.SilkS, 1 mm, beside SW1",
    ):
        assert text in table, text
    # the board is one change setting a board that holds only the layout: no stack, no SI classes come with it
    [board] = [c for c in plan.changes if c.target == "pcb"]
    assert board.operation == "set" and board.payload.stackup is None and board.payload.keepouts == [] and not any(c.target == "si" for c in plan.changes)
    assert board.rationale == f"design file {plan.sha256}; board layout: outline, 2 fixed part(s), 1 pack region(s), 1 silk text(s)"


def test_a_confirmation_applies_the_switch_and_the_board_as_the_files_values(tmp_path: Path, lib: KicadLibrary):
    ir, plan, _table = _applied(tmp_path, lib)
    note = f"design file {plan.sha256}"
    sim = ir.simulation
    assert sim.nodes == ["PRESS"]
    [k1] = sim.elements
    assert (k1.id, k1.device, k1.net, k1.reference_net, k1.control_net, k1.control_reference_net, k1.model_name) == (
        "K1", SpiceDevice.S, "VIN", "IN", "PRESS", "GND", "SWK")
    assert k1.provenance.kind is ProvenanceKind.USER_REQUIREMENT and k1.provenance.note == note and k1.reason == "SW1, pressed by the schedule on PRESS"
    assert k1.model_card.value == SW_CARD and k1.model_card.provenance.kind is ProvenanceKind.USER_REQUIREMENT
    assert k1.model_card.provenance.note == f"{note}; model card {model_card_hash(SW_CARD)} from {SW_SOURCE}"
    # the netlist holds the switch across the button's nets, its control pair and its card; the button itself stays out
    netlist = build(ir)
    assert "SK1 VIN IN PRESS 0 SWK" in netlist and SW_CARD in netlist and "VPRESS PRESS 0 PWL(" in netlist
    assert not any(line.startswith(("SW1", "XSW1", "J1")) for line in netlist.splitlines())
    # the board: the layout and nothing else (no stack, no classes - the PCB agent decides those as for a file without the section)
    pcb = ir.pcb
    assert pcb.outline.width_mm == 30 and pcb.outline.height_mm == 20 and pcb.stackup is None and ir.si is None and pcb.placements == []
    assert [(p.component_ref, p.x_mm, p.y_mm, p.rotation_deg, p.side.value) for p in pcb.fixed] == [("SW1", 8, 15, 0, "top"), ("J1", 24, 15, 90, "top")]
    [region] = pcb.pack_regions
    assert (region.id, region.rect.value, region.rect.unit, region.refs) == ("rc", [2, 2, 16, 8], "mm", ["C1", "R1"])
    [text] = pcb.silkscreen
    assert (text.text, text.kind, text.layer, text.component_ref, text.size_mm, text.thickness_mm) == ("KEY", SilkKind.USER, "F.SilkS", "SW1", 1.0, 0.15)
    for traced in (*pcb.fixed, region, region.rect, text):
        assert traced.provenance.kind is ProvenanceKind.USER_REQUIREMENT and traced.provenance.note == note
    # the text carries the design file's mark (the PCB agent keeps it), never a template's
    assert design_file_silk_text(text) and proposed_silk_text(text) and not template_silk_text(text)


def test_the_placer_keeps_the_files_positions_and_its_text_survives_the_silkscreen_stage(tmp_path: Path, lib: KicadLibrary):
    ir, plan, _table = _applied(tmp_path, lib)
    placed = fixed_placement(ir, lib)
    by = {p.component_ref: p for p in placed.placements}
    assert placed.region_of["SW1"] == placed.region_of["J1"] == FIXED and placed.region_of["C1"] == placed.region_of["R1"] == "rc"
    assert placed.region_of["R2"] == FREE_AREA
    for given in ir.pcb.fixed:  # where the file says, with the file's provenance
        assert design_data(by[given.component_ref]) == design_data(given)
    # the region's parts in its list order (C1 left of R1, one row), inside its rectangle
    assert by["C1"].y_mm == by["R1"].y_mm and 2 < by["C1"].x_mm < by["R1"].x_mm < 18 and 2 < by["C1"].y_mm < 10
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={**BASE, ROUTING_KEY: "skip"}))
    [proposal] = res.proposals
    silk = proposal.payload.silkscreen
    assert [design_data(t) for t in silk[:1]] == [design_data(t) for t in ir.pcb.silkscreen]  # kept first, exactly as the file put it
    assert sorted(t.component_ref for t in silk if t.kind == SilkKind.REFERENCE) == ["C1", "J1", "R1", "R2", "SW1"]
    assert all(t.provenance.tool_version == PLACER_KEEP_VERSION and "kept_texts:1" in t.provenance.derived_from
               and t.provenance.note.endswith("; the design file's 1 kept text(s) are keep-outs too") for t in silk[1:])
    assert any(n.startswith("silkscreen.place 0.2: 1 design file text(s) kept where the design file put them, 5 reference(s)") for n in res.notes)
    Orchestrator.apply_proposals(ir, res.proposals)
    checks = silk_checks(ir, tmp_path, lib)
    assert checks[SILK_CLEARANCE_CHECK].status is ValidationStatus.PASS and checks[SILK_OVERLAP_CHECK].status is ValidationStatus.PASS
    assert ("- 설계 파일이 보드와 함께 제안한 문자 1개 ('KEY'): 설계 파일이 둔 자리 그대로입니다. 실크 배치기(`silkscreen.place` 0.2)는 이 문자들을 "
            "옮기지 않고 금지 영역으로 삼아 나머지 문자를 그 둘레에 놓았습니다.") in _silkscreen_section(ir)
    assert not any("템플릿이 보드와 함께" in line for line in _silkscreen_section(ir))


@needs_dll
def test_the_switch_pressed_and_released_passes_spice_in_one_transient(tmp_path: Path, lib: KicadLibrary):
    ir, plan, _table = _applied(tmp_path, lib)
    _run(ir, tmp_path, lib, plan, {ROUTING_KEY: "skip"}, stop_after=Stage.SPICE)
    results = {r.check_id: r for r in ir.validation.results}
    pressed, released = results["spice.v_pressed"], results["spice.v_released"]
    assert pressed.status is ValidationStatus.PASS and released.status is ValidationStatus.PASS, (pressed.message, released.message)
    assert pressed.details["measured"] == pytest.approx(V_PRESSED, rel=1e-3)
    assert abs(released.details["measured"]) < 1e-3  # 5 V x R2 / ROFF = 0.5 mV through the open switch
    [card] = [c for c in results["spice"].details["model_cards"] if c["of"] == "simulation element K1"]
    assert card["model"] == "SWK" and card["sha256"] == model_card_hash(SW_CARD) and card["note"].endswith(f"from {SW_SOURCE}")


def test_an_element_card_without_a_source_is_simulated_but_noted(tmp_path: Path, lib: KicadLibrary):
    data = _edited(lambda d: d["simulation"]["elements"][0].pop("model_source"))
    plan = load_design_file(_write(tmp_path, data), lib)
    assert "no model_source: SPICE results that would pass are NOT_VERIFIED" in plan.table()
    setup = next(c.payload for c in plan.changes if c.target == "simulation")
    assert setup.elements[0].model_card.provenance.note.endswith(MODEL_CARD_NO_SOURCE)


def test_a_model_may_be_defined_by_another_elements_card(tmp_path: Path, lib: KicadLibrary):
    """Two keys share one switch model: the second element names the model the first one's card defines."""
    def edit(d: dict) -> None:
        d["simulation"]["nodes"].append("PRESS2")
        d["simulation"]["elements"].append({"id": "K2", "device": "S", "net": "VIN", "reference_net": "IN", "control_net": "PRESS2",
                                            "control_reference_net": "GND", "model_name": "SWK", "reason": "a second key on the same nets"})
    plan = load_design_file(_write(tmp_path, _edited(edit)), lib)
    setup = next(c.payload for c in plan.changes if c.target == "simulation")
    assert [e.id for e in setup.elements] == ["K1", "K2"] and setup.elements[1].model_card is None


# --------------------------------------------------------------------------- left out of everything while absent


def test_empty_new_fields_change_neither_the_table_nor_the_changes(tmp_path: Path, lib: KicadLibrary):
    def plain(d: dict) -> None:
        d["simulation"].pop("elements")
        d["simulation"].pop("nodes")
        d.pop("board")
        d["simulation"]["stimuli"] = d["simulation"]["stimuli"][:1]
        d["simulation"]["expectations"] = []
    base = _edited(plain)
    empty = copy.deepcopy(base)
    empty["simulation"].update(elements=[], nodes=[])
    empty["board"] = {}
    views = []
    for data in (base, empty):
        plan = load_design_file(_write(tmp_path, data), lib)
        changes = json.dumps([design_data(c) for c in plan.changes], sort_keys=True)
        views.append((plan.table().replace(plan.sha256, "<sha>"), changes.replace(plan.sha256, "<sha>"), plan.layout, plan.fixed, plan.silk_texts))
    assert views[0] == views[1]
    assert "simulation-only node" not in views[0][0] and "simulation-only element" not in views[0][0] and "Board layout" not in views[0][0]
    assert '"pcb"' not in views[0][1]


# --------------------------------------------------------------------------- a board the IR already has


def test_the_layout_is_appended_to_a_board_the_ir_already_has_and_never_replaces_one(tmp_path: Path, lib: KicadLibrary):
    ir = _ir(tmp_path)
    ir.pcb = PCBDesign()  # a board without placements, as the user may have started it
    ir, plan, _table = _applied(tmp_path, lib, ir)
    assert ir.components and ir.pcb.outline.width_mm == 30 and [p.component_ref for p in ir.pcb.fixed] == ["SW1", "J1"]
    assert [r.id for r in ir.pcb.pack_regions] == ["rc"] and [t.text for t in ir.pcb.silkscreen] == ["KEY"]
    targets = [c.target for c in plan.for_board(PCBDesign()).changes if c.target.startswith("pcb")]
    assert targets == ["pcb.outline", "pcb.fixed", "pcb.fixed", "pcb.pack_regions", "pcb.silkscreen"]
    assert plan.for_board(None) is plan
    # an outline already set: nothing is proposed, and the note says why
    other = tmp_path / "other"
    other.mkdir()
    taken = _ir(other)
    taken.pcb = PCBDesign(outline=BoardOutline(width_mm=50.0, height_mm=20.0))
    plan = load_design_file(_write(other), lib)
    state = _run(taken, other, lib, plan, BASE)
    assert not taken.components and taken.pcb.outline.width_mm == 50.0
    assert any("its board section clashes with the IR's board - ir.pcb already has an outline (50 x 20 mm); the design file never replaces it" in o.message
               for o in state.outcomes)


# --------------------------------------------------------------------------- refusals, each naming the field


def _element(**change):
    def edit(d: dict) -> None:
        d["simulation"]["elements"][0].update(change)
        for key in [k for k, v in change.items() if v is None]:
            d["simulation"]["elements"][0].pop(key)
    return edit


def _load(edit):
    def edit_load(d: dict) -> None:
        d["simulation"]["elements"].append({"id": "RL", "device": "R", "net": "OUT", "reference_net": "GND", "value": {"value": 8.2, "unit": "ohm"},
                                            "reason": "a load the board does not carry"})
        edit(d["simulation"]["elements"][1])
    return edit_load


@pytest.mark.parametrize(("edit", "message"), [
    (lambda d: d["simulation"].update(nodes=["P RESS"]), r"simulation\.nodes\.0: 'P RESS' is not a plain node name"),
    (lambda d: d["simulation"].update(nodes=["out"]), r"simulation\.nodes\.0: 'out' is a net of the file or the ground node"),
    (lambda d: d["simulation"].update(nodes=["PRESS", "press"]), r"simulation\.nodes\.1: 'press' is listed twice"),
    (_element(id="K-1"), r"simulation\.elements\.0\.id: 'K-1' must be a plain identifier"),
    (_element(id="r1"), r"simulation\.elements\.0\.id: 'r1' is a component ref of the file"),
    (_element(device="Q"), r"simulation\.elements\.0\.device: element 'K1' is a Q; a simulation-only element is R, C, L or D, or a voltage-controlled switch S"),
    (_load(lambda e: e.update(id="1")), r"simulation\.elements\.1\.id: element '1' would be the netlist element R1, which R1 already is"),
    (_element(net="NOPE"), r"simulation\.elements\.0\.net: element 'K1' names 'NOPE', which is no net of the file and no simulation node"),
    (_element(reference_net="VIN"), r"simulation\.elements\.0\.reference_net: element 'K1': net and reference_net are both 'VIN'"),
    (_element(control_net=None), r"simulation\.elements\.0\.control_net: element 'K1' is a voltage-controlled switch: it needs both control nodes"),
    (_element(control_reference_net="NOPE"), r"simulation\.elements\.0\.control_reference_net: element 'K1' names 'NOPE'"),
    (_element(control_reference_net="PRESS"), r"simulation\.elements\.0\.control_reference_net: .*both 'PRESS' \(the control voltage would be identically 0\)"),
    (_element(params={"ic": 1}), r"simulation\.elements\.0\.params: element 'K1' is a voltage-controlled switch: it takes its numbers on its SW model card"),
    (_load(lambda e: e.update(control_net="PRESS")), r"simulation\.elements\.1\.control_net: element 'RL' is a R; only a voltage-controlled switch \(S\) has"),
    (_load(lambda e: e.pop("value")), r"simulation\.elements\.1\.value: element 'RL' is a R: it needs a positive number in ohm, got none"),
    (_load(lambda e: e.update(value=-8)), r"simulation\.elements\.1\.value: element 'RL' is a R: it needs a positive number in ohm, got -8"),
    (_load(lambda e: e.update(value={"value": 8, "unit": "V"})), r"simulation\.elements\.1\.value: element 'RL' is stated in 'V', a R takes ohm"),
    (_load(lambda e: e.update(model_name="X")), r"simulation\.elements\.1\.model_name: element 'RL' is a R: it takes a value, not a model"),
    (_element(value=1), r"simulation\.elements\.0\.value: element 'K1' is a S: it takes a model"),
    (_element(model_name=None), r"simulation\.elements\.0\.model_name: element 'K1' is a S: it needs a single-token model_name"),
    (_element(model_card=".include vendor.lib\n" + SW_CARD), r"simulation\.elements\.0\.model_card: simulation element K1: model_card line '\.include vendor\.lib' is not allowed"),
    (_element(model_card=".model OTHER SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)"), r"simulation\.elements\.0\.model_card: simulation element K1: model_card does not define 'SWK'"),
    (_element(model_card=".model SWK SW(RON=0.1 ROFF=1e8 VT=0.5)"), r"simulation\.elements\.0\.model_card: element 'K1': SW model 'SWK' does not give \['vh'\]"),
    (_element(model_card=".model SWK D(IS=1n)"), r"simulation\.elements\.0\.model_card: element 'K1': model 'SWK' is a D model; a voltage-controlled switch needs an SW model"),
    (_element(model_source="a\nb"), r"simulation\.elements\.0\.model_source: simulation element K1: model_source must be one line"),
    (_element(model_card=None), r"simulation\.elements\.0\.model_source: element 'K1' has model_source without model_card"),
    (_element(model_card=None, model_source=None), r"simulation\.elements\.0\.model_name: no model_card in the file defines 'SWK'"),
    (_element(reason=" "), r"simulation\.elements\.0\.reason: element 'K1' needs a reason"),
    (_element(provenance={"kind": "authoritative"}), r"simulation\.elements\.0\.provenance: Extra inputs"),
    (_element(device="Z"), r"simulation\.elements\.0\.device: Input should be"),
])
def test_a_simulation_element_or_node_the_compiler_would_refuse_is_refused_by_its_field(tmp_path: Path, lib: KicadLibrary, edit, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _edited(edit)), lib)


def _board(edit):
    def edit_board(d: dict) -> None:
        edit(d["board"])
    return edit_board


@pytest.mark.parametrize(("edit", "message"), [
    (_board(lambda b: b.pop("outline")), r"board\.outline: the fixed parts, pack regions and silk texts are positions on the board's outline"),
    (_board(lambda b: [b.pop(k) for k in ("outline", "fixed", "pack_regions")]), r"board\.outline: the fixed parts, pack regions and silk texts are"),
    (_board(lambda b: b["silk_texts"][0].update(y_mm=20.5)), r"board\.silk_texts\.0: silk text 'KEY' at \(9\.27, 20\.5\) mm lies outside the outline \(0, 0\)-\(30, 20\) mm"),
    (_board(lambda b: b["outline"].update(width_mm=0)), r"board\.outline: width_mm and height_mm must be > 0, got 0 x 20"),
    (_board(lambda b: b["fixed"][0].update(ref="SW9")), r"board\.fixed\.0\.ref: 'SW9' is no component of the file"),
    (_board(lambda b: b["fixed"][1].update(ref="SW1")), r"board\.fixed\.1\.ref: SW1 is fixed twice"),
    (_board(lambda b: b["pack_regions"][0]["refs"].append("SW1")), r"board\.pack_regions\.0\.refs\.2: SW1 is already fixed"),
    (_board(lambda b: b["pack_regions"].append({"id": "more", "rect_mm": [20, 2, 8, 8], "refs": ["C1"]})),
     r"board\.pack_regions\.1\.refs\.0: C1 is already in pack region rc"),
    (_board(lambda b: b["pack_regions"].append({"id": "rc", "rect_mm": [20, 2, 8, 8], "refs": ["R2"]})), r"board\.pack_regions\.1\.id: pack region id 'rc' is used twice"),
    (_board(lambda b: b["pack_regions"][0]["refs"].append("C9")), r"board\.pack_regions\.0\.refs\.2: 'C9' is no component of the file"),
    (_board(lambda b: b["pack_regions"][0].update(id="1rc")), r"board\.pack_regions\.0: Value error, pack region id '1rc' must be a plain identifier"),
    (_board(lambda b: b["pack_regions"][0].update(rect_mm=[2, 2, 16])), r"board\.pack_regions\.0: Value error, pack region rc: rect must be \[x, y, w, h\]"),
    (_board(lambda b: b["pack_regions"][0].update(rect_mm=[20, 2, 16, 8])),
     r"board\.pack_regions\.0\.rect_mm: pack region rc \(20, 2\)-\(36, 10\) mm leaves the outline \(0, 0\)-\(30, 20\) mm"),
    (_board(lambda b: b["silk_texts"][0].update(text="건반")), r"board\.silk_texts\.0\.text: '건반' must be one line of printable ASCII"),
    (_board(lambda b: b["silk_texts"][0].update(text="A\nB")), r"board\.silk_texts\.0\.text: 'A\\nB' must be one line of printable ASCII"),
    (_board(lambda b: b["silk_texts"][0].update(ref="SW9")), r"board\.silk_texts\.0\.ref: 'SW9' is no component of the file"),
    (_board(lambda b: b["silk_texts"][0].update(size_mm=0)), r"board\.silk_texts\.0: silk text 'KEY' has a non-positive size 0 mm"),
    (_board(lambda b: b["silk_texts"][0].update(kind="reference")), r"board\.silk_texts\.0\.kind: Input should be 'user' or 'title'"),
    (_board(lambda b: b["fixed"][0].update(x_mm="8")), r"board\.fixed\.0\.x_mm: Input should be a valid number"),
    (_board(lambda b: b["fixed"][0].update(side="left")), r"board\.fixed\.0\.side: Input should be 'top' or 'bottom'"),
    (_board(lambda b: b["outline"].update(width_mm=True)), r"board\.outline\.width_mm: Input should be a valid number"),
    (_board(lambda b: b.update(keepouts=[])), r"board\.keepouts: Extra inputs"),
    (_board(lambda b: b["fixed"][0].update(provenance={"kind": "authoritative"})), r"board\.fixed\.0\.provenance: Extra inputs"),
])
def test_a_board_section_the_layout_cannot_take_is_refused_by_its_field(tmp_path: Path, lib: KicadLibrary, edit, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _edited(edit)), lib)


def test_an_infinite_board_number_is_refused(tmp_path: Path, lib: KicadLibrary):
    path = _write(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace('"x_mm": 9.27', '"x_mm": 1e400'), encoding="utf-8")
    with pytest.raises(DesignFileError, match=r"board\.silk_texts\.0\.x_mm: Input should be a finite number"):
        load_design_file(path, lib)


# --------------------------------------------------------------------------- the compiler's names, card registry and params, at load


def _second_key(**fields):
    """A second switch K2 on its own press node, naming the model SWK (``fields`` override or add)."""
    def edit(d: dict) -> None:
        d["simulation"]["nodes"].append("PRESS2")
        d["simulation"]["elements"].append({"id": "K2", "device": "S", "net": "VIN", "reference_net": "IN", "control_net": "PRESS2",
                                            "control_reference_net": "GND", "model_name": "SWK", "reason": "a second key on the same nets", **fields})
    return edit


def _card_on_the_excluded_button(**binding):
    """The switch card moves onto SW1, which is excluded from SPICE (``binding``: what else its binding keeps): the netlist never carries it."""
    def edit(d: dict) -> None:
        d["components"][1]["spice"].update(model_name="SWK", model_card=SW_CARD, **binding)
        d["simulation"]["elements"][0].pop("model_card")
        d["simulation"]["elements"][0].pop("model_source")
    return edit


@pytest.mark.parametrize(("edit", "message"), [
    # an id the netlist line would split (the compiler's pattern ends in $, which also matches before a final newline)
    (_element(id="K1\n"), r"simulation\.elements\.0\.id: 'K1\\n' must be a plain identifier"),
    (lambda d: d["simulation"].update(nodes=["0"]), r"simulation\.nodes\.0: '0' is a net of the file or the ground node"),
    (_second_key(id="K1"), r"simulation\.elements\.1\.id: element 'K1' would be the netlist element SK1, which simulation element K1 already is"),
    (_element(reason="pressed\nby PRESS"), r"simulation\.elements\.0\.reason: element 'K1' needs a reason: one line"),
    # the card registry: only a card the netlist carries defines a model, and one model name has one card
    (_card_on_the_excluded_button(), r"simulation\.elements\.0\.model_name: no model_card in the file defines 'SWK' \(an element's, or a part's that is "
                                     r"not excluded from SPICE"),
    (_card_on_the_excluded_button(device="D"), r"simulation\.elements\.0\.model_name: no model_card in the file defines 'SWK'"),
    (_second_key(model_card=".model SWK SW(RON=0.2 ROFF=1e8 VT=0.5 VH=0.1)"),
     r"simulation\.elements\.1\.model_card: simulation element K2: model 'SWK' has another, different model_card in the file"),
    (_second_key(model_name="SWK2", model_card=".model SWK2 SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)\n" + SW_CARD),
     r"simulation\.elements\.1\.model_card: simulation element K2: model_card defines 'swk', which the card for 'swk' also defines"),
    (_load(lambda e: e.update(id="DCLAMP", device="D", value=None, model_name="DSUB", model_card=".subckt DSUB a b\nR1 a b 1k\n.ends DSUB")),
     r"simulation\.elements\.1\.model_card: element 'DCLAMP' is a D: it needs a \.model, but 'DSUB' is a \.subckt"),
    # params the compiler cannot write
    (_load(lambda e: e.update(params={"bad key": 1})), r"simulation\.elements\.1\.params: simulation element RL: parameter name 'bad key' is not a plain identifier"),
    (_load(lambda e: e.update(params={"tc1": "a b"})), r"simulation\.elements\.1\.params: simulation element RL param tc1: 'a b' is not a single SPICE token"),
    (_load(lambda e: e.update(params={"tc1": [1, 2]})), r"simulation\.elements\.1\.params: simulation element RL param tc1 must be a number, got list"),
    (_load(lambda e: e.update(params={"tc1": 10**400})), r"simulation\.elements\.1\.params: simulation element RL: a parameter is not a number ngspice can read"),
])
def test_what_the_compiler_would_refuse_later_is_refused_at_load_by_its_field(tmp_path: Path, lib: KicadLibrary, edit, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _edited(edit)), lib)


def test_a_second_switch_may_repeat_the_same_card_and_a_diode_takes_its_own(tmp_path: Path, lib: KicadLibrary):
    """The compiler takes one card per model name, the same text twice included; a D element on its own .model card is netlisted."""
    def edit(d: dict) -> None:
        _second_key(model_card=SW_CARD, model_source=SW_SOURCE)(d)
        d["simulation"]["elements"].append({"id": "CLAMP", "device": "D", "net": "OUT", "reference_net": "GND", "model_name": "DX",
                                            "model_card": ".model DX D(IS=1e-14)", "params": {"area": 2}, "reason": "a clamp the board does not carry"})
    ir, _plan, _table = _applied(tmp_path, lib, data=_edited(edit))
    netlist = build(ir)
    assert "SK1 VIN IN PRESS 0 SWK" in netlist and "SK2 VIN IN PRESS2 0 SWK" in netlist and "DCLAMP OUT 0 DX area=2" in netlist
    assert netlist.count(SW_CARD) == 1


@pytest.mark.parametrize(("edit", "message"), [
    (_board(lambda b: b["pack_regions"][0].update(id="rc\n")), r"board\.pack_regions\.0\.id: pack region id 'rc\\n' must be a plain identifier"),
    (_board(lambda b: b["outline"].update(origin_x_mm=5)), r"board\.pack_regions\.0\.rect_mm: pack region rc \(2, 2\)-\(18, 10\) mm leaves the outline \(5, 0\)"),
    (_board(lambda b: b["outline"].update(origin_y_mm=5)), r"board\.pack_regions\.0\.rect_mm: pack region rc \(2, 2\)-\(18, 10\) mm leaves the outline \(0, 5\)"),
    (_board(lambda b: b["pack_regions"][0].update(rect_mm=[2, 14, 16, 8])), r"board\.pack_regions\.0\.rect_mm: pack region rc \(2, 14\)-\(18, 22\) mm leaves"),
    (_board(lambda b: b["silk_texts"][0].update(x_mm=-0.5)), r"board\.silk_texts\.0: silk text 'KEY' at \(-0\.5, 17\.8\) mm lies outside the outline"),
])
def test_a_board_item_off_the_outline_or_with_a_broken_id_is_refused_by_its_field(tmp_path: Path, lib: KicadLibrary, edit, message: str):
    with pytest.raises(DesignFileError, match=message):
        load_design_file(_write(tmp_path, _edited(edit)), lib)


def test_sides_kinds_sizes_and_the_origin_are_kept_as_the_file_states_them(tmp_path: Path, lib: KicadLibrary):
    def edit(d: dict) -> None:
        d["board"]["outline"].update(origin_x_mm=1, origin_y_mm=2, width_mm=29, height_mm=18)
        d["board"]["fixed"][0]["side"] = "bottom"
        d["board"]["pack_regions"][0]["side"] = "bottom"
        d["board"]["silk_texts"][0].update(side="bottom", kind="title", size_mm=2)
    plan = load_design_file(_write(tmp_path, _edited(edit)), lib)
    assert (plan.outline.origin_x_mm, plan.outline.origin_y_mm, plan.outline.width_mm, plan.outline.height_mm) == (1, 2, 29, 18)
    assert plan.fixed[0].side is BoardSide.BOTTOM and plan.pack_regions[0].side is BoardSide.BOTTOM
    [text] = plan.silk_texts
    assert (text.layer, text.kind, text.size_mm, text.thickness_mm) == ("B.SilkS", SilkKind.TITLE, 2.0, stroke_for(2.0)) and stroke_for(2.0) == 0.3


def test_only_the_files_own_mark_keeps_a_text(tmp_path: Path, lib: KicadLibrary):
    """A user / title text with user_requirement provenance, no tool and exactly the note ``design file sha256:<hex>``; nothing else passes."""
    plan = load_design_file(_write(tmp_path), lib)
    [text] = plan.silk_texts
    prov = text.provenance
    assert design_file_silk_text(text) and proposed_silk_text(text)
    for changed in (
        {"provenance": prov.model_copy(update={"kind": ProvenanceKind.ASSUMPTION})},
        {"provenance": prov.model_copy(update={"tool": "silkscreen.place"})},
        {"provenance": prov.model_copy(update={"note": prov.note + "; drawn by hand"})},
        {"provenance": prov.model_copy(update={"note": prov.note[:-1]})},
        {"kind": SilkKind.REFERENCE},
    ):
        other = text.model_copy(update=changed)
        assert not design_file_silk_text(other) and not proposed_silk_text(other), changed


def test_a_files_title_text_is_kept_and_no_second_title_is_placed(tmp_path: Path, lib: KicadLibrary):
    data = _edited(lambda d: d["board"]["silk_texts"][0].update(kind="title"))
    ir, _plan, _table = _applied(tmp_path, lib, data=data)
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers={**BASE, ROUTING_KEY: "skip"}))
    [proposal] = res.proposals
    assert [t.text for t in proposal.payload.silkscreen if t.kind == SilkKind.TITLE] == ["KEY"]
    assert "silkscreen title: the design file's title 'KEY' is kept where the design file put it; no second title placed" in res.notes
    assert any(n.startswith("silkscreen.place 0.2: 1 design file text(s) kept") and "title 'KEY' kept where the design file put it" in n for n in res.notes)
