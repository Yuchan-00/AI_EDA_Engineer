"""The templates' board decisions: the stack (``pcb_layers`` 2 | 4) and the need-driven SI classes in the confirmation table and the IR.

Synthetic libraries (``tests/test_circuit_templates.py``, ``tests/fixtures_atmega.py``),
no KiCad, no ngspice. What is proved:

* every template shows the stack and its classes in the ``confirm_design``
  table and, once confirmed, carries a 2-layer generic stack (the default,
  shown as a choice) or the 4-layer one ``pcb_layers=4`` asks for (with the
  plane edge clearance), a ``DEFAULT`` class with the conservative driver
  edge and the controlled class ``Z50`` without nets - every SI number a
  confirmed choice (``user_requirement``) or a calculator output that
  ``calc.recompute`` re-derives;
* the ATmega128 board declares only what its circuit needs: the crystal
  loop's length, the supply rails' IPC-2221 minimum width, the ISP SPI
  timing path (``f_SCK = f_clk / 4``, NOT_VERIFIED offline naming the
  datasheet keys nobody grounded);
* a stated layer count is served by the stack built from it (the reviewer
  finds nothing unserved), and the template version names the change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.design import TEMPLATE_VERSION
from ai_eda.design.board import CONTROLLED_CLASS, DEFAULT_CLASS, PLANE_CLEARANCE_KEY, T_RISE_S
from ai_eda.ir import ProvenanceKind, ValidationStatus as S
from ai_eda.review import IndependentReviewer
from ai_eda.tools.calc import recompute_parameters
from ai_eda.tools.si import measure_nets
from ai_eda.validation.si import si_results, timing_results
from ai_eda.workflow import Stage
from tests.fixtures_atmega import atmega_library
from tests.test_circuit_templates import ASTABLE, DIVIDER, LED, RC, _confirm, _ir, _present, template_library

TEMPLATE_INPUTS = {"divider": DIVIDER, "led": LED, "rc_lowpass": RC, "astable": ASTABLE}


def test_the_template_version_names_the_board_decisions():
    assert TEMPLATE_VERSION == "0.3"


@pytest.mark.parametrize("name", sorted(TEMPLATE_INPUTS))
def test_every_template_shows_and_applies_the_default_stack_and_classes(tmp_path: Path, name: str):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, name)
    table = _present(ir, tmp_path, lib, TEMPLATE_INPUTS[name])
    assert "pcb_layers = 2 - board layer count: the default 2 layers" in table and "a generic value, not a fab's" in table
    assert "Board stack and signal integrity" in table and f"{DEFAULT_CLASS} (default: every net no class lists)" in table
    assert f"si.t_rise = {T_RISE_S:.12g} s - driver edge for the critical-length rule" in table and "conservative" in table
    assert f"{CONTROLLED_CLASS}: nets -; Z0 50 ohm +/- 10%" in table
    _confirm(ir, tmp_path, lib)
    stack, si = ir.pcb.stackup, ir.si
    assert stack is not None and stack.layer_count == 2 and not stack.plane_layers() and [layer.name for layer in ir.pcb.layers] == ["F.Cu", "B.Cu"]
    assert ir.pcb.placements == [] and ir.pcb.outline is None  # the board decisions only: PLACEMENT places
    assert [c.name for c in si.net_classes] == [DEFAULT_CLASS, CONTROLLED_CLASS] and si.default_class().promote_to == CONTROLLED_CLASS
    assert si.net_class(CONTROLLED_CLASS).members() == [] and si.timing_paths == []
    for key, t in si.traced_items():
        assert t.provenance.kind is ProvenanceKind.USER_REQUIREMENT and "design choice confirmed by user" in (t.provenance.note or ""), key
    assert PLANE_CLEARANCE_KEY not in ir.parameters


def test_pcb_layers_4_builds_the_planes_and_the_count_is_served_by_the_stack(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "osc")
    table = _present(ir, tmp_path, lib, {**ASTABLE, "pcb_layers": "4"})
    assert "req.pcb_layers: pcb_layers = 4 layers" in table and "In1.Cu = GND plane, In2.Cu = VCC plane" in table and "pcb_layers = 2" not in table
    state, _ = _confirm(ir, tmp_path, lib, Stage.IR_BUILD)
    stack = ir.pcb.stackup
    assert stack.layer_count == 4 and [c.plane_net.value for c in stack.plane_layers()] == ["GND", "VCC"] and stack.served_requirements() == ["req.pcb_layers"]
    assert ir.parameters["pcb_layers"].value == 4 and ir.parameters[PLANE_CLEARANCE_KEY].value == 0.5
    # PLACEMENT routed it on the outer layers and drew the two planes; the promotion found nothing long on this small board
    assert [(z.net, z.layer) for z in ir.pcb.zones] == [("GND", "In1.Cu"), ("VCC", "In2.Cu")] and ir.pcb.tracks
    assert "plane zones: GND on In1.Cu, VCC on In2.Cu" in state.outcome(Stage.PLACEMENT).message
    assert ir.validation.latest("si.critical_length").status is S.PASS
    traced = IndependentReviewer(tools={"kicad_library": lib}).check_requirements_vs_ir(ir, tmp_path)
    assert "req.pcb_layers" not in traced.details.get("unserved", []) and traced.status is not S.FAIL


def _atmega(tmp_path: Path, answers: dict[str, str] | None = None):
    lib = atmega_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "atm")
    table = _present(ir, tmp_path, lib, answers or {"input_voltage": "9 V", "clock_frequency": "16 MHz"})
    _confirm(ir, tmp_path, lib)
    return ir, lib, table


def test_the_atmega_board_declares_only_what_its_circuit_needs(tmp_path: Path):
    ir, lib, table = _atmega(tmp_path)
    si = ir.si
    assert [c.name for c in si.net_classes] == [DEFAULT_CLASS, CONTROLLED_CLASS, "XTAL", "POWER", "ISP_SPI", "RC_RESET"]
    xtal, power, isp, reset = si.net_class("XTAL"), si.net_class("POWER"), si.net_class("ISP_SPI"), si.net_class("RC_RESET")
    assert xtal.nets == ["XTAL1", "XTAL2"] and xtal.max_length_mm.value == 25.0 and xtal.t_rise_s is None  # a loop length, no driven edge
    assert power.nets == ["VIN_RAW", "VIN", "+5V", "GND", "AVCC"] and power.min_width_mm.provenance.tool == "calc.ipc2221.width_for_current"
    assert power.min_width_mm.provenance.inputs == {"i": "i_load_budget", "dt": "power_temp_rise", "t": "pcb.stackup.copper[F.Cu].thickness_um"}
    assert power.min_width_mm.value == pytest.approx(ir.parameters["w_power_min"].value) and power.min_width_mm.value < 0.25  # 50 mA needs no wider track
    # SCK / MOSI are driven by the off-board programmer during ISP (and by U1 as port pins): no driver's facts; MISO (PE1) is U1's
    assert isp.nets == ["PB1", "PE0"] and isp.promote_to == CONTROLLED_CLASS and isp.driver is None and si.default_class().driver == "U1"
    assert si.class_of("PE1").name == DEFAULT_CLASS
    # ~RESET is the RC reset (R2 / C8, SW1, the programmer): no fast driven edge - neither the critical-length rule nor spice.si touches it
    assert reset.nets == ["RESET"] and reset.t_rise_s is None and reset.promote_to is None
    assert "the highest rate" not in table and "faster than the ATmega128 accepts" in table and "f_clk / 4 gives exactly 2" in table
    (path,) = si.timing_paths
    assert (path.name, path.clock_net, path.data_nets) == ("ISP", "PB1", ["PE0"]) and path.f_clk_hz.value == 4e6 and path.capture_fraction.value == 0.5
    assert path.f_clk_hz.provenance.tool == "calc.clock.divided" and path.f_clk_hz.provenance.inputs == {"f": "f_clk", "n": "isp_sck_divider"}
    assert path.terms_from == {"t_co_max_s": "J2.t_co", "t_co_min_s": "J2.t_co_min", "t_su_min_s": "U1.t_su", "t_h_min_s": "U1.t_h"}
    for line in ("XTAL: nets XTAL1, XTAL2; max length 25 mm", "POWER: nets VIN_RAW, VIN, +5V, GND, AVCC; min width", "timing path ISP: clock PB1 -> data PE0",
                 "si.xtal_max_length = 25 mm", "isp_sck_divider = 4", "f_sck = 4000000 Hz [calc.clock.divided from f_clk, isp_sck_divider]"):
        assert line in table, line
    res = recompute_parameters(ir)
    assert res.status is S.PASS and {"w_power_min", "f_sck", "si.net_classes[POWER].min_width_mm", "si.timing_paths[ISP].f_clk_hz"} <= set(res.details["parameters"])


def test_the_isp_timing_path_is_not_verified_offline_naming_the_datasheet_keys(tmp_path: Path):
    ir, lib, _ = _atmega(tmp_path)
    assert ir.pcb.tracks == []  # confirmed only (PLACEMENT not run): no copper, so the flight times are missing too
    (t,) = timing_results(ir, ir.si, measure_nets(ir))
    assert t.status is S.NOT_VERIFIED
    for missing in ("J2.t_co (datasheet fact key t_co of J2 not grounded: J2 is a connector and the part that launches the edge is off-board, behind it",
                    "--datasheet-url J2=<its document>", "J2.t_co_min", "U1.t_su (datasheet fact key t_su of U1 not grounded)", "U1.t_h",
                    "routed copper of PB1", "routed copper of PE0"):
        assert missing in t.message, missing


@pytest.mark.parametrize("name", ["astable", "divider"])
def test_the_template_boards_route_byte_identically_with_and_without_their_si_classes(tmp_path: Path, name: str):
    """Nothing declared, nothing promoted: the PLACEMENT copper of the astable and the divider is routing.maze 0.2's, byte for byte."""
    from ai_eda.agents import AgentContext, PCBAgent
    from ai_eda.ir.provenance import design_data
    from ai_eda.tools.routing.maze import ROUTER_VERSION
    from ai_eda.workflow import Orchestrator

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, name)
    _present(ir, tmp_path, lib, TEMPLATE_INPUTS[name])
    _confirm(ir, tmp_path, lib)
    bare = ir.model_copy(deep=True)
    bare.si, bare.pcb = None, None
    ctx = AgentContext(workdir=tmp_path, tools={"kicad_library": lib})
    with_si, without = PCBAgent().run(ir, ctx), PCBAgent().run(bare, ctx)
    assert [p.target for p in with_si.proposals] == ["pcb"]  # nothing promoted
    a, b = with_si.proposals[0].payload, without.proposals[0].payload
    assert [design_data(t) for t in a.tracks] == [design_data(t) for t in b.tracks] and [design_data(v) for v in a.vias] == [design_data(v) for v in b.vias]
    assert a.tracks and all(t.provenance.tool_version == ROUTER_VERSION for t in a.tracks)
    assert [design_data(p) for p in a.placements] == [design_data(p) for p in b.placements] and a.stackup is not None and b.stackup is None
    Orchestrator.apply_proposals(ir, with_si.proposals)
    assert {r.check_id: r.status for r in si_results(ir)}["si.critical_length"] is S.PASS
