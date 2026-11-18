"""The simulation-only voltage-controlled switch (``SpiceDevice.S``) and the windowed ``Reduce.FREQUENCY``.

A design has one netlist and every analysis runs on it, so a keyboard's
notes - different circuit states - are verified inside one transient: each
key stays an open (excluded) part, a switch across it is pressed by a PWL
schedule on a private node, and each note's frequency is measured in its
own time window. Here:

* the IR: an ``S`` element carries its two control nodes; every element
  without them dumps and hashes exactly as before they existed (pinned from
  a dump printed by the code before the fields were added);
* the compiler: the element line, the ``SW`` card's rules and every refusal;
* the windowed frequency on synthetic samples: two windows of a waveform
  whose frequency changes halfway give the two frequencies, the unwindowed
  measure is unchanged, a window without enough crossings or outside the
  saved samples is refused by name;
* on ngspice (skipped where the engine is absent): the switch presents its
  RON / ROFF and switches at VT +/- VH, and a two-transistor astable whose
  timing chain is tapped by switches pressed one after another measures in
  each window, within 1 cent, the frequency of a deck with that key wired in.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import analysis_command, build, build_report, netlist_elements, spice_vector_name, switch_model_problem
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    ArtifactKind,
    CircuitIR,
    Component,
    Expectation,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    SimElement,
    SimulationSetup,
    SourceRef,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    ValidationResult,
    ValidationStatus,
    authoritative,
    user_requirement,
)
from ai_eda.ir.simulation import CONTROLLED_DEVICES, NODE_COUNTS, OPTIONAL_WINDOW_REDUCES, REDUCE_PARAMS, SIM_ELEMENT_DEVICES, SWITCH_MODEL_PARAMS
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult, rising_edge_frequency, validate_deck, window_frequency
from ai_eda.tools.spice.stage import read_results, reduce_expectation, run_spice_for

DS = SourceRef(title="Generic resistor datasheet", authority="Vendor", content_hash="sha256:abc")
AUTH = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)
DERIVED = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")
DIODE_CARD = ".model D1N4148 D(Is=2.52n Rs=0.568 N=1.752 Cjo=4p M=0.4 tt=20n)"
#: the switch the measurements below were made with: 0.1 ohm closed, 1e8 ohm open, closes above 0.6 V, opens below 0.4 V
SW_CARD = ".model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)"
#: 1 cent as a relative frequency tolerance: 2^(1/1200) - 1
ONE_CENT = 2.0 ** (1.0 / 1200.0) - 1.0

runner = NgspiceShared()
needs_engine = pytest.mark.skipif(not runner.available(), reason="ngspice.dll (KiCad's bundled ngspice shared library) not found")


def _norm(text: str) -> str:
    """A dump with every wall-clock ``created_at`` replaced (the only field that differs between two runs)."""
    return re.sub(r'"created_at":"[^"]*"', '"created_at":"T"', text)


def _part(ref: str, n: int, spice: SpiceBinding, prov: Provenance = DERIVED) -> Component:
    pins = [Pin(number=str(i), name=f"~{i}", electrical_type=PinElectricalType.PASSIVE, provenance=AUTH) for i in range(1, n + 1)]
    return Component(ref=ref, value=ref, pins=pins, provenance=prov, spice=spice)


def _net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=DERIVED)


# --------------------------------------------------------------------------- the IR


def legacy_ir() -> CircuitIR:
    """A setup with every kind of thing that existed before the switch: R / C / D elements, a private node, PWL / PULSE stimuli, an unwindowed frequency, a window mean."""
    ir = CircuitIR(project=ProjectMeta(id="legacy", name="legacy"))
    ir.components = [_part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=authoritative(10_000.0, DS, "ohm"), provenance=AUTH)),
                     _part("R2", 2, SpiceBinding(device=SpiceDevice.R, value=authoritative(10_000.0, DS, "ohm"), provenance=AUTH))]
    ir.nets = [_net("VIN", ("R1", "1"), kind=NetKind.POWER), _net("VOUT", ("R1", "2"), ("R2", "1")), _net("GND", ("R2", "2"), kind=NetKind.GROUND)]
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VIN", source="voltage", net="VIN", reference_net="GND", kind=StimulusKind.PULSE,
                          params={k: user_requirement(v) for k, v in dict(v1=0.0, v2=5.0, td=0.0, tr=1e-6, tf=1e-6, pw=0.5e-3, per=1e-3).items()}, provenance=USER),
                 Stimulus(id="DRV", source="voltage", net="DRV", reference_net="GND", kind=StimulusKind.PWL,
                          params={"points": user_requirement([[0.0, 0.0], [1e-3, 1.0]])}, provenance=USER)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(1e-6, "s"), "stop": user_requirement(10e-3, "s")}, provenance=USER)],
        expectations=[
            Expectation(id="f_in", analysis_id="tran", vector="v(VIN)", reduce=Reduce.FREQUENCY, nominal=user_requirement(1000.0, "Hz"), tol_rel=user_requirement(0.01), provenance=USER),
            Expectation(id="v_mean", analysis_id="tran", vector="v(VOUT)", reduce=Reduce.MEAN, nominal=user_requirement(1.25, "V"), tol_abs=user_requirement(0.05, "V"),
                        params={"t_start": user_requirement(2e-3, "s"), "t_stop": user_requirement(8e-3, "s")}, provenance=USER),
        ],
        elements=[SimElement(id="CBUS", device=SpiceDevice.C, net="VOUT", reference_net="GND", value=user_requirement(100e-12, "F"), reason="bus wiring capacitance", provenance=USER),
                  SimElement(id="DOD", device=SpiceDevice.D, net="VOUT", reference_net="DRV", model_name="D1N4148", model_card=authoritative(DIODE_CARD, DS), reason="open-drain driver", provenance=USER)],
        nodes=["DRV"],
    )
    return ir


#: printed from :func:`legacy_ir` by the code before ``SimElement.control_net`` / ``control_reference_net`` and the frequency window existed
LEGACY_CONTENT_HASH = "sha256:4888e67ffe639f4f1609b4b9ca3bc13784d14920f1bad0287a2e2b3e9343b7de"
LEGACY_SETUP_DUMP_SHA256 = "45884ec7997ffe801652ce5cfd68351877c04cfe480de429a6242773ce8d7fb0"
LEGACY_ELEMENT_KEYS = ["id", "device", "net", "reference_net", "value", "model_name", "model_card", "params", "reason", "provenance", "serves_requirements"]
LEGACY_NETLIST = ("legacy\n" + DIODE_CARD + "\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\nCBUS VOUT 0 100p\nDOD VOUT DRV D1N4148\n"
                  "VDRV DRV 0 PWL(0 0 1m 1)\nVVIN VIN 0 PULSE(0 5 0 1u 1u 500u 1m)\n.end\n")


def test_a_setup_without_a_switch_dumps_hashes_and_compiles_exactly_as_before():
    ir = legacy_ir()
    assert ir.content_hash() == LEGACY_CONTENT_HASH
    assert hashlib.sha256(_norm(ir.simulation.model_dump_json()).encode()).hexdigest() == LEGACY_SETUP_DUMP_SHA256
    assert [list(json.loads(e.model_dump_json())) for e in ir.simulation.elements] == [LEGACY_ELEMENT_KEYS, LEGACY_ELEMENT_KEYS]
    assert all("control_net" not in e and "control_reference_net" not in e for e in ir.design_dict()["simulation"]["elements"])
    assert build(ir) == LEGACY_NETLIST
    # a saved IR without the fields loads, hashes and compiles the same
    loaded = CircuitIR.model_validate_json(ir.model_dump_json())
    assert loaded.content_hash() == LEGACY_CONTENT_HASH and build(loaded) == LEGACY_NETLIST


def _switch(**kw) -> SimElement:
    args = dict(id="K1", device=SpiceDevice.S, net="A", reference_net="GND", control_net="CTL", control_reference_net="GND",
                model_name="SWM", model_card=user_requirement(SW_CARD, note="test switch model"), reason="push button pressed by the schedule on CTL", provenance=USER)
    args.update(kw)
    return SimElement(**args)


def test_a_switch_element_carries_its_control_nodes_in_every_dump_once_set():
    assert SpiceDevice.S in SIM_ELEMENT_DEVICES and SpiceDevice.S in CONTROLLED_DEVICES and NODE_COUNTS[SpiceDevice.S] == (4,)
    assert SWITCH_MODEL_PARAMS == ("ron", "roff", "vt", "vh")
    el = _switch()
    dumped = json.loads(el.model_dump_json())
    assert list(dumped)[:6] == ["id", "device", "net", "reference_net", "control_net", "control_reference_net"]
    assert dumped["control_net"] == "CTL" and dumped["control_reference_net"] == "GND" and dumped["device"] == "S"
    assert SimElement.model_validate_json(el.model_dump_json()) == el
    # an S element without its control nodes is a valid IR object (the compiler refuses it, below) and dumps without them
    bare = _switch(control_net=None, control_reference_net=None)
    assert "control_net" not in json.loads(bare.model_dump_json()) and "control_reference_net" not in bare.model_dump()
    ir = switch_ir()
    h = ir.content_hash()
    assert ir.design_dict()["simulation"]["elements"][0]["control_net"] == "CTL"
    ir.simulation.elements[0].control_net = "VDD"
    assert ir.content_hash() != h


# --------------------------------------------------------------------------- the compiler


def switch_ir(*, ramp: bool = True) -> CircuitIR:
    """R1 = 1 k from VDD (1 V) to A; the switch K1 from A to ground, controlled by the private node CTL (a PWL ramp 0 -> 1 -> 0 V)."""
    ir = CircuitIR(project=ProjectMeta(id="switch", name="switch"))
    ir.components = [_part("R1", 2, SpiceBinding(device=SpiceDevice.R, value=authoritative(1000.0, DS, "ohm"), provenance=AUTH))]
    ir.nets = [_net("VDD", ("R1", "1"), kind=NetKind.POWER), _net("A", ("R1", "2")), _net("GND", kind=NetKind.GROUND)]
    points = [[0.0, 0.0], [1e-3, 0.0], [2e-3, 1.0], [3e-3, 1.0], [4e-3, 0.0]] if ramp else [[0.0, 0.0], [1e-3, 0.0], [1.001e-3, 1.0], [2e-3, 1.0], [2.001e-3, 0.0]]
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VDD", source="voltage", net="VDD", reference_net="GND", kind=StimulusKind.DC, value=user_requirement(1.0, "V"), provenance=USER),
                 Stimulus(id="CTL", source="voltage", net="CTL", reference_net="GND", kind=StimulusKind.PWL, params={"points": user_requirement(points)}, provenance=USER)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN,
                               params={"step": user_requirement(1e-6 if ramp else 9.56e-6, "s"), "stop": user_requirement(5e-3 if ramp else 3e-3, "s")}, provenance=USER)],
        elements=[_switch()],
        nodes=["CTL"],
    )
    return ir


SWITCH_NETLIST = "switch\n" + SW_CARD + "\nR1 VDD A 1k\nSK1 A 0 CTL 0 SWM\nVCTL CTL 0 PWL(0 0 1m 0 2m 1 3m 1 4m 0)\nVVDD VDD 0 DC 1\n.end\n"


def test_the_switch_line_names_its_switched_pair_then_its_control_pair_then_its_sw_model():
    ir = switch_ir()
    text = build(ir)
    assert text == SWITCH_NETLIST
    assert netlist_elements(text) == [("R1", ["VDD", "A"], "1k"), ("SK1", ["A", "0", "CTL", "0"], "SWM"), ("VCTL", ["CTL", "0"], "PWL(0 0 1m 0 2m 1 3m 1 4m 0)"),
                                      ("VVDD", ["VDD", "0"], "DC 1")]
    report = build_report(ir)
    assert report["sim_elements"] == ["SK1"] and report["sim_nodes"] == ["CTL"] and report["models"] == ["swm"]
    assert report["model_cards"] == [{"of": "simulation element K1", "model": "SWM", "sha256": "sha256:" + hashlib.sha256(SW_CARD.encode()).hexdigest(),
                                      "provenance": "user_requirement", "note": "test switch model", "source": None}]
    assert validate_deck(text)[0] == []
    # the card may be spelled with spaces, commas, lower case and a continuation line; the element line is the same
    ir.simulation.elements[0].model_card = user_requirement(".model SWM sw ( ron = 0.1, roff=1e8\n+ vt=0.5 vh=0.1 )", note="test switch model")
    assert build(ir).split("\n")[-6:] == SWITCH_NETLIST.split("\n")[-6:]
    # a control node may be an IR net: the net then has a node even when only excluded parts touch it
    ir = switch_ir()
    ir.components.append(_part("J1", 1, SpiceBinding(exclude=True, exclude_reason="connector", provenance=USER)))
    ir.nets.append(_net("EXT", ("J1", "1")))
    ir.simulation.elements[0].control_net = "EXT"
    assert "SK1 A 0 EXT 0 SWM\n" in build(ir) and build_report(ir)["nets_touching_only_excluded"] == []


@pytest.mark.parametrize(("change", "message"), [
    (dict(control_net=None), "a voltage-controlled switch needs both control nodes (control_net, control_reference_net), missing ['control_net']"),
    (dict(control_reference_net=None), "missing ['control_reference_net']"),
    (dict(control_net=None, control_reference_net=None), "missing ['control_net', 'control_reference_net']"),
    (dict(control_net="NOPE"), "simulation element K1: unknown net or simulation node 'NOPE' for control_net"),
    (dict(control_reference_net="nope2"), "simulation element K1: unknown net or simulation node 'nope2' for control_reference_net"),
    (dict(net="NOPE"), "simulation element K1: unknown net or simulation node 'NOPE'"),
    (dict(control_net="GND"), "control_net and control_reference_net are both 'GND' (the control voltage would be identically 0)"),
    (dict(reference_net="A"), "net and reference_net are both 'A'"),
    (dict(value=user_requirement(1.0, "ohm")), "a S element takes a model, not a value"),
    (dict(model_name=None, model_card=None), "a S element needs a single-token model_name"),
    (dict(params={"m": user_requirement(2.0)}), "a voltage-controlled switch takes its numbers on its SW model card, no params (got ['m'])"),
    (dict(model_card=None), "model 'SWM' is not defined by any model_card in the IR"),
    (dict(model_card=Traced(value=SW_CARD, provenance=DERIVED)), "a model card must have authoritative or user_requirement provenance"),
    (dict(model_card=Traced(value=SW_CARD, provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED))), "llm_generated provenance"),
    (dict(provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED)), "llm_generated provenance"),
])
def test_a_switch_is_refused_by_name_when_it_is_not_fully_stated(change: dict, message: str):
    ir = switch_ir()
    ir.simulation.elements = [_switch(**change)]
    with pytest.raises(CompileError, match=re.escape(message)):
        build(ir)


@pytest.mark.parametrize(("card", "message"), [
    (".model SWM SW", "SW model 'SWM' does not give ['ron', 'roff', 'vt', 'vh']: ngspice would use defaults of its own"),
    (".model SWM SW(RON=0.1 ROFF=1e8 VT=0.5)", "does not give ['vh']"),
    (".model SWM SW(RON=1e8 ROFF=0.1 VT=0.5 VH=0.1)", "needs 0 < RON < ROFF, got RON=1e+08 ROFF=0.1"),
    (".model SWM SW(RON=0 ROFF=1e8 VT=0.5 VH=0.1)", "needs 0 < RON < ROFF"),
    (".model SWM SW(RON=1 ROFF=1 VT=0.5 VH=0.1)", "needs 0 < RON < ROFF, got RON=1 ROFF=1"),
    (".model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=-0.1)", "needs VH >= 0 (the hysteresis around VT), got VH=-0.1"),
    (".model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1 IT=1m)", "parameter 'it' is not one of ['ron', 'roff', 'vt', 'vh']"),
    (".model SWM SW(RON=0.1 RON=0.2 ROFF=1e8 VT=0.5 VH=0.1)", "gives ron twice"),
    (".model SWM SW(RON=0.1ohm ROFF=1e8 VT=0.5 VH=0.1)", "SW model 'SWM' parameter ron: trailing letters"),
    (".model SWM SW(RON ROFF=1e8 VT=0.5 VH=0.1)", "SW model 'SWM': 'RON' is not a name=value parameter"),
    (".model SWM CSW(RON=0.1 ROFF=1e8 IT=1m IH=0.1m)", "model 'SWM' is a CSW model; a voltage-controlled switch needs an SW model"),
    (".model SWM D(Is=1n)", "model 'SWM' is a D model"),
    (".subckt SWM a b c d\nR1 a b 1k\n.ends", "a S element needs a .model, but 'SWM' is a .subckt"),
])
def test_the_sw_card_must_state_every_parameter_and_a_real_switch(card: str, message: str):
    ir = switch_ir()
    ir.simulation.elements = [_switch(model_card=user_requirement(card, note="test switch model"))]
    with pytest.raises(CompileError, match=re.escape(message)):
        build(ir)
    assert switch_model_problem(SW_CARD, "SWM") is None and switch_model_problem(SW_CARD, "swm") is None


#: a subcircuit whose own SW model is complete, then the top-level ``.model SWM SW`` a top-level switch uses: bare
SUBCKT_THEN_BARE = ".subckt SWX a b\n.model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)\nR1 a b 1k\n.ends\n.model SWM SW"


def test_the_sw_card_is_judged_as_ngspice_reads_it():
    """The top-level ``.model`` the switch uses is the one judged: continuations over comments and blank lines, a subcircuit's model local to it, the first of two."""
    def problem(card: str) -> str | None:
        return switch_model_problem(card, "SWM")

    # a "+" line continues the last line that is neither blank nor a comment, indented or not (ngspice-46, the engine test below)
    assert problem(".model SWM SW(RON=0.1 ROFF=1e8\n* the button's contact\n+ VT=0.5 VH=0.1)") is None
    assert problem(".model SWM SW(RON=0.1 ROFF=1e8\n\n+ VT=0.5 VH=0.1)") is None
    assert problem(".model SWM SW(RON=0.1 ROFF=1e8\n  + VT=0.5 VH=0.1)") is None
    assert problem(".model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0)") is None  # no hysteresis is a stated VH
    # the subcircuit's complete model is not the switch's: the bare top-level one is, and it is refused by name
    assert problem(SUBCKT_THEN_BARE) == ("SW model 'SWM' does not give ['ron', 'roff', 'vt', 'vh']: ngspice would use defaults of its own "
                                         "(a bare SW model is a 1 ohm switch closed at 0 V control - measured)")
    ir = switch_ir()
    ir.simulation.elements = [_switch(model_card=user_requirement(SUBCKT_THEN_BARE, note="test switch model"))]
    with pytest.raises(CompileError, match=re.escape("simulation element K1: SW model 'SWM' does not give")):
        build(ir)
    only_local = ".subckt SWX a b\n.model SWM SW(RON=0.1 ROFF=1e8 VT=0.5 VH=0.1)\nR1 a b 1k\n.ends"
    assert problem(only_local) == "no top-level .model line defines 'SWM' (a .model inside a .subckt is local to it; ngspice would not find it)"
    ir.simulation.elements = [_switch(model_card=user_requirement(only_local, note="test switch model"))]
    with pytest.raises(CompileError, match=re.escape("simulation element K1: no top-level .model line defines 'SWM'")):
        build(ir)
    # of two top-level definitions the first is simulated
    assert problem(SW_CARD + "\n.model SWM SW") is None
    assert problem(".model SWM SW\n" + SW_CARD).startswith("SW model 'SWM' does not give")


def test_control_nodes_are_refused_on_every_other_device_and_a_part_is_never_a_switch():
    ir = switch_ir()
    ir.simulation.elements = [_switch(), SimElement(id="CBUS", device=SpiceDevice.C, net="A", reference_net="GND", control_net="CTL", value=user_requirement(1e-12, "F"),
                                                     reason="x", provenance=USER)]
    with pytest.raises(CompileError, match=re.escape("simulation element CBUS: ['control_net'] belong to a voltage-controlled switch (S); a C element has no control nodes")):
        build(ir)
    ir.simulation.elements[1] = SimElement(id="DX", device=SpiceDevice.D, net="A", reference_net="GND", control_reference_net="GND", model_name="D1N4148",
                                           model_card=authoritative(DIODE_CARD, DS), reason="x", provenance=USER)
    with pytest.raises(CompileError, match=re.escape("['control_reference_net'] belong to a voltage-controlled switch (S); a D element has no control nodes")):
        build(ir)
    # a transistor is still a board part, never a simulation-only element (the old refusal, now naming the switch too)
    ir.simulation.elements[1] = SimElement(id="QX", device=SpiceDevice.Q, net="A", reference_net="GND", model_name="Q", reason="x", provenance=USER)
    with pytest.raises(CompileError, match=re.escape("a simulation-only element is R, C, L or D, or a voltage-controlled switch S")):
        build(ir)
    # a part bound as S is refused: the switch is the simulation's, the part is the board's
    ir = switch_ir()
    ir.components.append(_part("SW1", 4, SpiceBinding(device=SpiceDevice.S, model_name="SWM", model_card=user_requirement(SW_CARD, note="x"), pin_order=["1", "2", "3", "4"],
                                                     provenance=USER)))
    ir.nets += [_net("N1", ("SW1", "1")), _net("N2", ("SW1", "2")), _net("N3", ("SW1", "3")), _net("N4", ("SW1", "4"))]
    with pytest.raises(CompileError, match=re.escape("SW1: a voltage-controlled switch (S) is a simulation-only element (SimulationSetup.elements")):
        build(ir)


def test_the_deck_validator_and_the_element_parser_know_the_switch():
    problems, _ = validate_deck("t\nR1 a 0 1k\nS1 a 0 c 0\n.end\n")
    assert problems == ["line 3: 'S1 a 0 c 0' needs 4 nodes and a model (ngspice-46 refuses to load the deck: 'Unable to find definition of model')"]
    assert validate_deck("t\nR1 a 0 1k\nS1 a 0 c 0 SWM\n.model SWM SW(RON=1 ROFF=1meg VT=0.5 VH=0.1)\n.end\n")[0] == []
    assert netlist_elements("t\nS1 a 0 c 0 SWM\n.end\n") == [("S1", ["a", "0", "c", "0"], "SWM")]
    with pytest.raises(ValueError, match="malformed element line"):
        netlist_elements("t\nS1 a 0 c 0\n.end\n")


# --------------------------------------------------------------------------- the windowed frequency (no engine)


def _times(n: int, dt: float = 1e-5) -> list[float]:
    return [i * dt for i in range(n)]


def _two_tone(n: int = 3000, switch_at: int = 1500) -> list[float]:
    """A 0 / 5 V square sampled every 10 us: 1 kHz (100 samples a period) until sample 1500 (15 ms), then 2.5 kHz (40 samples) - a key released and the next one pressed."""
    return [(5.0 if i % 100 < 50 else 0.0) if i < switch_at else (5.0 if (i - switch_at) % 40 < 20 else 0.0) for i in range(n)]


def _tran(times: list[float], out: list[float]) -> SpiceResult:
    return SpiceResult(engine="x", engine_version="x", netlist_path="n", netlist_hash="h", analysis=SpiceAnalysis.TRAN, command="tran 1e-5 30m",
                       vectors={"time": times, "out": out}, vector_types={"time": "time", "out": "voltage"}, scale="time", n_points=len(times), succeeded=True)


def _f(window: tuple[float, float] | None = None, nominal: float = 1000.0, **kw) -> Expectation:
    params = {} if window is None else {"t_start": user_requirement(window[0], "s"), "t_stop": user_requirement(window[1], "s")}
    return Expectation(id="f_key", analysis_id="tran", vector="v(OUT)", reduce=Reduce.FREQUENCY, nominal=user_requirement(nominal, "Hz"), tol_rel=user_requirement(0.01),
                       params=params, provenance=USER, **kw)


def test_two_windows_of_one_transient_give_the_two_frequencies():
    assert REDUCE_PARAMS[Reduce.FREQUENCY] == ((), ("t_start", "t_stop")) and Reduce.FREQUENCY in OPTIONAL_WINDOW_REDUCES
    t = _times(3000)
    res = _tran(t, _two_tone())
    first = reduce_expectation(res, _f((1e-3, 14e-3)), "out")
    second = reduce_expectation(res, _f((16e-3, 29e-3), nominal=2500.0), "out")
    assert first.problem is None and first.measured == pytest.approx(1000.0, rel=1e-9)
    assert second.problem is None and second.measured == pytest.approx(2500.0, rel=1e-9)
    # the window is recorded beside the edges, first; the crossings are only the window's own
    assert list(first.extra)[:3] == ["t_start", "t_stop", "edges"] and first.extra["t_start"] == 1e-3 and first.extra["t_stop"] == 14e-3
    assert first.extra["edges"] == 13 and 1e-3 <= first.extra["first_edge_s"] < first.extra["last_edge_s"] <= 14e-3
    assert second.extra["edges"] == 33 and 16e-3 <= second.extra["first_edge_s"]
    # the pure function gives the same numbers
    direct = window_frequency(t, res.vectors["out"], 16e-3, 29e-3)
    assert direct.frequency == second.measured and direct.edges[0] == second.extra["first_edge_s"]


def test_without_a_window_the_frequency_is_measured_as_before():
    t = _times(3000)
    v = _two_tone()
    res = _tran(t, v)
    whole = reduce_expectation(res, _f(), "out")
    old = rising_edge_frequency(t, v)
    assert whole.measured == old.frequency and 1000.0 < whole.measured < 2500.0  # the two notes blend into one number: why the window exists
    assert whole.extra == {"edges": len(old.edges), "first_edge_s": old.edges[0], "last_edge_s": old.edges[-1], "low": old.low, "mid": old.mid, "high": old.high,
                           "vmin": old.vmin, "vmax": old.vmax, "floor": old.floor}
    assert list(whole.extra) == ["edges", "first_edge_s", "last_edge_s", "low", "mid", "high", "vmin", "vmax", "floor"]
    # the unwindowed problem texts are the old ones, word for word
    flat = reduce_expectation(_tran(t, [5.0] * 3000), _f(), "out")
    assert flat.problem == "no oscillation detected: the waveform is flat at 5 .. 5 (swing 0 is within the engine's resolution 0.005)"
    step = reduce_expectation(_tran(t, [0.0] * 1500 + [5.0] * 1500), _f(), "out")
    assert step.problem == "no oscillation detected: 1 rising edge(s) through the mid level 2.5 in the saved window, 3 are needed for a frequency (swing 0 .. 5)"


def test_a_window_without_enough_crossings_or_outside_the_samples_is_refused_by_name():
    t = _times(3000)
    res = _tran(t, _two_tone())
    short = reduce_expectation(res, _f((1.2e-3, 3.6e-3)), "out")  # 2.4 periods of 1 kHz: two rising edges (2 ms, 3 ms)
    assert short.measured is None and short.problem == (
        "no oscillation detected: 2 rising edge(s) through the mid level 2.5 in the window [0.0012, 0.0036] s, 3 are needed for a frequency (swing 0 .. 5)"
    )
    assert short.extra["t_start"] == 1.2e-3 and short.extra["edges"] == 2
    flat = window_frequency(t, [0.0] * 1000 + [5.0] * 2000, 15e-3, 25e-3)
    assert flat.frequency is None and flat.problem.startswith("no oscillation detected: the waveform is flat at 5 .. 5 in the window [0.015, 0.025] s")
    outside = reduce_expectation(res, _f((20e-3, 40e-3)), "out")
    assert outside.measured is None and outside.problem == "the window [0.02, 0.04] s is not inside the saved samples [0, 0.02999] s"
    reversed_ = window_frequency(t, res.vectors["out"], 5e-3, 2e-3)
    assert reversed_.frequency is None and "is not an interval" in reversed_.problem
    # a hand-built expectation with half a window is said to be one (the compiler refuses it before)
    half = _f()
    half.params = {"t_start": user_requirement(1e-3, "s")}
    assert reduce_expectation(res, half, "out").problem == "reduce=frequency takes its window as both params t_start and t_stop or neither, got only ['t_start']"


def _windowed_ir(window: dict[str, float], **analysis) -> CircuitIR:
    ir = legacy_ir()
    params = {"step": 1e-6, "stop": 10e-3, **analysis}
    ir.simulation.analyses[0].params = {k: user_requirement(v, "s") for k, v in params.items()}
    ir.simulation.expectations = ir.simulation.expectations[:1]  # the frequency alone (the mean's window would meet the start rule too)
    ir.simulation.expectations[0].params = {k: user_requirement(v, "s") for k, v in window.items()}
    return ir


def test_the_compiler_takes_a_frequency_window_by_the_windowed_reductions_rule():
    assert build(_windowed_ir({"t_start": 2e-3, "t_stop": 8e-3})) == LEGACY_NETLIST  # the window is no deck content
    assert build(_windowed_ir({"t_start": 2.01e-3, "t_stop": 8e-3}, start=2e-3)) == LEGACY_NETLIST
    for window, analysis, message in (
        ({"t_start": 2e-3}, {}, "expectation f_in: reduce=frequency takes its window as both t_start and t_stop or neither, got only ['t_start']"),
        ({"t_stop": 8e-3}, {}, "got only ['t_stop']"),
        ({"t_start": 2e-3, "t_stop": 12e-3}, {}, "expectation f_in: the window [0.002, 0.012] s must lie inside the tran window [0, 0.01] s of tran"),
        ({"t_start": 8e-3, "t_stop": 2e-3}, {}, "with 0 <= t_start < t_stop"),
        ({"t_start": 1e-3, "t_stop": 8e-3}, {"start": 2e-3}, "must lie inside the tran window [0.002, 0.01] s"),
        ({"t_start": 2e-3, "t_stop": 8e-3}, {"start": 2e-3}, "t_start 0.002 s must be at least one tran step after the analysis' start"),
        ({"t_start": 2e-3, "t_stop": 8e-3, "f_max": 1e3}, {}, "unexpected params ['f_max'] for reduce=frequency (it takes ['t_start', 't_stop'])"),
    ):
        with pytest.raises(CompileError, match=re.escape(message)):
            build(_windowed_ir(window, **analysis))
    ir = _windowed_ir({"t_start": 2e-3, "t_stop": 8e-3})
    ir.simulation.expectations[0].params["t_start"] = user_requirement(2e-3, "ms")
    with pytest.raises(CompileError, match=re.escape("param t_start carries unit 'ms', reduce=frequency takes t_start in s")):
        build(ir)


def test_the_reports_say_the_window_of_a_windowed_frequency_and_the_control_nodes_of_a_switch():
    from ai_eda.report.stages import theory_report

    plain = theory_report(legacy_ir())
    assert "구간이 주어진" not in plain and " 구간)" not in plain
    ir = _windowed_ir({"t_start": 2e-3, "t_stop": 8e-3})
    ir.simulation.elements.append(_switch(net="VOUT", control_net="DRV"))
    ir.simulation.elements[-1].model_card = user_requirement(SW_CARD, note="test switch model")
    text = theory_report(ir)
    assert "상승 에지 주파수 (2 ms ~ 8 ms 구간)" in text
    assert "구간 [t_start, t_stop] 이 주어진 주파수 기대값은" in text
    assert "- `K1` (S) VOUT → GND: 모델 `SWM`; 제어 DRV → GND — push button pressed by the schedule on CTL" in text


def test_the_final_report_names_the_window_its_frequency_was_measured_in():
    from ai_eda.report.stages import final_report

    def recorded(ir: CircuitIR) -> str:
        exp = ir.simulation.expectations[0]
        r = reduce_expectation(_tran(_times(3000), _two_tone()), exp, "out")
        ir.validation.add(ValidationResult(check_id=f"spice.{exp.id}", status=ValidationStatus.PASS, message="recorded",
                                           details={"measured": r.measured, "frequency": dict(r.extra)}))
        return final_report(ir, None)

    text = recorded(_windowed_ir({"t_start": 1e-3, "t_stop": 14e-3}))
    assert "| `f_in` | `v(VIN)` / 상승 에지 주파수 (1 ms ~ 14 ms 구간) | 1 kHz | 1 kHz |" in text
    assert "- `f_in` 주파수 측정 (1 ms ~ 14 ms 구간): 상승 에지 13개, 첫 에지 1.995 ms, 마지막 에지 13.995 ms" in text
    plain = legacy_ir()
    plain.simulation.expectations = plain.simulation.expectations[:1]
    text = recorded(plain)
    assert "- `f_in` 주파수 측정: 상승 에지 " in text and " 구간)" not in text


# --------------------------------------------------------------------------- on the engine


@needs_engine
def test_the_switch_presents_ron_and_roff_and_switches_at_vt_plus_and_minus_vh(tmp_path: Path):
    """The measured behaviour the SimElement docstring records, re-run: 0.1 ohm / 1e8 ohm, abrupt, closing at 0.6 V and opening at 0.4 V."""
    ir = switch_ir()
    path = tmp_path / "switch.cir"
    path.write_text(build(ir), encoding="utf-8", newline="\n")
    res = runner.run(path, SpiceAnalysis.TRAN, tmp_path / "ramp", command=analysis_command(ir.simulation.analyses[0], ir.simulation))
    assert res.succeeded, res.errors
    t, a, ctl, i = res.vectors["time"], res.vectors["a"], res.vectors["ctl"], res.vectors["vvdd#branch"]
    r = [a[k] / -i[k] for k in range(len(t))]
    closed = [k for k in range(len(t)) if r[k] < 1e3]
    assert closed and all(r[k] == pytest.approx(0.1, rel=1e-6) for k in closed)
    assert all(r[k] == pytest.approx(1e8, rel=1e-6) for k in range(len(t)) if k not in closed)  # nothing in between
    first, last = closed[0], closed[-1]
    assert closed == list(range(first, last + 1))  # one closed stretch
    assert ctl[first] >= 0.6 > ctl[first - 1] and 1.6e-3 <= t[first] < 1.601e-3  # closes at VT + VH on the way up
    assert ctl[last] > 0.4 >= ctl[last + 1] and 3.6e-3 <= t[last + 1] < 3.601e-3  # opens at VT - VH on the way down
    # a 1 us step under a 9.56 us tran step: the engine cuts its step at the corner and switches within one point of the 0.6 V crossing
    ir = switch_ir(ramp=False)
    path.write_text(build(ir), encoding="utf-8", newline="\n")
    res = runner.run(path, SpiceAnalysis.TRAN, tmp_path / "step", command=analysis_command(ir.simulation.analyses[0], ir.simulation))
    assert res.succeeded, res.errors
    t, a = res.vectors["time"], res.vectors["a"]
    on = next(k for k in range(len(t)) if a[k] < 0.5)
    off = next(k for k in range(on, len(t)) if a[k] > 0.5)
    assert 1.0006e-3 <= t[on] < 1.001e-3 and t[on - 1] < 1.0006e-3
    assert 2.0006e-3 <= t[off] < 2.001e-3 and t[off - 1] < 2.0006e-3


@needs_engine
def test_ngspice_reads_the_sw_card_the_way_the_compiler_judges_it(tmp_path: Path):
    """The engine facts :func:`switch_model_problem` rests on, measured on the deck of :func:`switch_ir` with only the card replaced."""
    def resistance(name: str, card: str) -> tuple[float, float]:
        """(the switch at 0 V control, at 1 V control) in ohm: v / i across it."""
        path = tmp_path / f"{name}.cir"
        path.write_text(SWITCH_NETLIST.replace(SW_CARD, card), encoding="utf-8", newline="\n")
        res = runner.run(path, SpiceAnalysis.TRAN, tmp_path / name, command="tran 1u 5m")
        assert res.succeeded, res.errors
        t, a, i = res.vectors["time"], res.vectors["a"], res.vectors["vvdd#branch"]
        at = lambda x: min(range(len(t)), key=lambda k: abs(t[k] - x))  # noqa: E731
        return a[0] / -i[0], a[at(2.5e-3)] / -i[at(2.5e-3)]

    assert resistance("plain", SW_CARD) == pytest.approx((1e8, 0.1), rel=1e-6)
    # continuations over a comment, a blank line and with an indent: the card's own RON / ROFF (and VT: open at 0 V)
    assert resistance("comment", ".model SWM SW(RON=0.1 ROFF=1e8\n* the contact\n+ VT=0.5 VH=0.1)") == pytest.approx((1e8, 0.1), rel=1e-6)
    assert resistance("blank", ".model SWM SW(RON=0.1 ROFF=1e8\n\n+ VT=0.5 VH=0.1)") == pytest.approx((1e8, 0.1), rel=1e-6)
    assert resistance("indent", ".model SWM SW(RON=0.1 ROFF=1e8\n  + VT=0.5 VH=0.1)") == pytest.approx((1e8, 0.1), rel=1e-6)
    # the subcircuit's model is not the switch's: the bare top-level one is, a 1 ohm switch already closed at 0 V
    assert resistance("subckt", SUBCKT_THEN_BARE) == pytest.approx((1.0, 1.0), rel=1e-6)
    # of two top-level definitions the first
    assert resistance("first", SW_CARD + "\n.model SWM SW") == pytest.approx((1e8, 0.1), rel=1e-6)


#: a two-transistor collector-coupled astable on a 3.0 V rail (1 k collectors, 100 nF, a 6.265 k fixed half) whose second base
#: resistor is a chain tapped by three keys: the design formula's base resistances for C4 / G4 / C5 (41628 / 25700 / 17681 ohm)
ASTABLE_RAIL = 3.0
CHAIN = {3: 17681.0, 2: 8019.0, 1: 15928.0}  # R3 base - T3, R2 T3 - T2, R1 T2 - T1
KEY_HZ = {1: 261.6256, 2: 391.9954, 3: 523.2511}  # the notes the chain was computed for (equal temperament)
#: the tran step, period / 2552 of G4: measured on this circuit, one deck's periods scatter by about +/- 0.3 us at 1 us (+/- 1 us
#: with single periods 4 us short at 2 us, where the mean also reads 0.5 cent high), so two decks agree to a small part of a cent
ASTABLE_STEP = 1e-6
SETTLE, MEASURE = 3, 10  # periods after a press before the window, periods in the window


def key_astable(*, pressed: int | None = None) -> CircuitIR:
    """The astable with keys SW1 .. SW3 open; ``pressed`` wires that key in (a 0 V source), else switches K1 .. K3 press them one after another."""
    npn = lambda ref: _part(ref, 3, SpiceBinding(device=SpiceDevice.Q, model_name="QNPN", pin_order=["3", "2", "1"], provenance=USER,  # noqa: E731
                                                 model_card=user_requirement(".model QNPN NPN (TR=200n)", note="generic NPN, ngspice's defaults with TR 200 ns")), USER)
    res_ = lambda ref, ohm: _part(ref, 2, SpiceBinding(device=SpiceDevice.R, value=user_requirement(ohm, "ohm"), provenance=USER), USER)  # noqa: E731
    cap = lambda ref, **params: _part(ref, 2, SpiceBinding(device=SpiceDevice.C, value=user_requirement(100e-9, "F"), params=params, provenance=USER), USER)  # noqa: E731

    def key(k: int) -> Component:
        if k == pressed:
            return _part(f"SW{k}", 2, SpiceBinding(device=SpiceDevice.V, value=user_requirement(0.0, "V"), provenance=USER), USER)
        return _part(f"SW{k}", 2, SpiceBinding(exclude=True, exclude_reason="push button: open unless pressed", provenance=USER), USER)

    ir = CircuitIR(project=ProjectMeta(id=f"keys_{pressed or 'switched'}", name="keyed astable"))
    ir.components = [npn("Q1"), npn("Q2"), res_("RC1", 1000.0), res_("RC2", 1000.0), res_("RB1", 6265.0), cap("C1"),
                     cap("C2", ic=user_requirement(-1.0, "V", note="breaks the symmetry so the oscillation starts")),
                     *(res_(f"R{k}", ohm) for k, ohm in CHAIN.items()), *(key(k) for k in (1, 2, 3))]
    ir.nets = [
        _net("OSC", ("RC1", "1"), ("RC2", "1"), ("RB1", "1"), ("SW1", "2"), ("SW2", "2"), ("SW3", "2"), kind=NetKind.POWER),
        _net("Q1C", ("RC1", "2"), ("Q1", "3"), ("C1", "1")),
        _net("Q2B", ("C1", "2"), ("Q2", "2"), ("R3", "1")),
        _net("T3", ("R3", "2"), ("R2", "1"), ("SW3", "1")),
        _net("T2", ("R2", "2"), ("R1", "1"), ("SW2", "1")),
        _net("T1", ("R1", "2"), ("SW1", "1")),
        _net("Q2C", ("RC2", "2"), ("Q2", "3"), ("C2", "1")),
        _net("Q1B", ("C2", "2"), ("RB1", "2"), ("Q1", "2")),
        _net("GND", ("Q1", "1"), ("Q2", "1"), kind=NetKind.GROUND),
    ]
    rail = Stimulus(id="OSC", source="voltage", net="OSC", reference_net="GND", kind=StimulusKind.DC, value=user_requirement(ASTABLE_RAIL, "V"), provenance=USER)
    if pressed is not None:
        period = 1.0 / KEY_HZ[pressed]
        stop = (SETTLE + MEASURE + 2) * period
        ir.simulation = SimulationSetup(
            stimuli=[rail],
            analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(ASTABLE_STEP, "s"), "stop": user_requirement(stop, "s"),
                                                                              "start": user_requirement(SETTLE * period, "s"), "uic": user_requirement(True)}, provenance=USER)],
            expectations=[_note(pressed, ((SETTLE + 0.5) * period, (SETTLE + MEASURE + 1.5) * period), KEY_HZ[pressed])],
        )
        return ir
    t, stimuli, windows = 2e-3, [rail], {}
    for k in (1, 2, 3):  # silence, then each key for SETTLE + MEASURE + 1 periods, then silence
        period = 1.0 / KEY_HZ[k]
        on, off = t, t + (SETTLE + MEASURE + 1) * period
        points = [[0.0, 0.0], [on, 0.0], [on + 1e-6, 1.0], [off, 1.0], [off + 1e-6, 0.0]]
        stimuli.append(Stimulus(id=f"P{k}", source="voltage", net=f"P{k}", reference_net="GND", kind=StimulusKind.PWL, params={"points": user_requirement(points)}, provenance=USER))
        windows[k] = (on + SETTLE * period, on + (SETTLE + MEASURE + 0.5) * period)
        t = off
    ir.simulation = SimulationSetup(
        stimuli=stimuli,
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(ASTABLE_STEP, "s"), "stop": user_requirement(t + 4e-3, "s"),
                                                                          "uic": user_requirement(True)}, provenance=USER)],
        expectations=[_note(k, windows[k], KEY_HZ[k]) for k in (1, 2, 3)],
        elements=[_switch(id=f"K{k}", net=f"T{k}", reference_net="OSC", control_net=f"P{k}", control_reference_net="GND", reason=f"key {k} pressed by the schedule on P{k}")
                  for k in (1, 2, 3)],
        nodes=["P1", "P2", "P3"],
    )
    return ir


def _note(k: int, window: tuple[float, float], nominal: float) -> Expectation:
    return Expectation(id=f"f_key{k}", analysis_id="tran", vector="v(Q1C)", reduce=Reduce.FREQUENCY, nominal=user_requirement(nominal, "Hz"),
                       tol_rel=user_requirement(ONE_CENT, note="1 cent"), params={"t_start": user_requirement(window[0], "s"), "t_stop": user_requirement(window[1], "s")},
                       provenance=USER)


@needs_engine
def test_a_switch_pressed_note_is_the_note_of_the_deck_with_that_key_wired_in(tmp_path: Path):
    """Three keys pressed one after another in one transient vs one deck per key with that key wired in: within 1 cent each.

    The wired decks are measured first; their frequencies become the switched
    expectations' nominals with a 1-cent tolerance, so the SPICE stage itself
    judges each window. The switches' ROFF (1e8 ohm) across the open keys
    between the base and the pressed tap is the one modelled difference
    (computed +0.30 cents for key 1, +0.04 for key 2, none for key 3; RON's
    0.1 ohm is 2e-6 of the chain).
    """
    wired: dict[int, float] = {}
    for k in (1, 2, 3):
        ref = key_astable(pressed=k)
        path = tmp_path / f"wired{k}.cir"
        path.write_text(build(ref), encoding="utf-8", newline="\n")
        assert f"VSW{k} T{k} OSC 0\n" in path.read_text(encoding="utf-8") and "SK" not in path.read_text(encoding="utf-8")
        res = runner.run(path, SpiceAnalysis.TRAN, tmp_path / f"wired{k}", command=analysis_command(ref.simulation.analyses[0], ref.simulation))
        assert res.succeeded, res.errors
        exp = ref.simulation.expectations[0]
        r = reduce_expectation(res, exp, spice_vector_name(exp.vector, ref))
        assert r.problem is None and r.extra["edges"] >= MEASURE, r.problem
        wired[k] = r.measured
        assert abs(1200.0 * math.log2(wired[k] / KEY_HZ[k])) < 10.0  # the chain plays its notes (the formula's own error is a few cents)

    ir = key_astable()
    for exp in ir.simulation.expectations:
        exp.nominal = user_requirement(wired[int(exp.id[-1])], "Hz", note="the deck with that key wired in")
    ir.project.workdir = str(tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path))
    text = Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    assert "SK1 T1 OSC P1 0 SWM\n" in text and "SK3 T3 OSC P3 0 SWM\n" in text and "VSW" not in text
    results = {r.check_id: r for r in run_spice_for(ir, {"spice": runner}, tmp_path)}
    for check_id, r in sorted(results.items()):
        print(f"  {check_id:<14} {r.status:<6} {r.message}")
    assert results["spice"].status is ValidationStatus.PASS, results["spice"].message
    for k in (1, 2, 3):
        r = results[f"spice.f_key{k}"]
        cents = 1200.0 * math.log2(r.details["measured"] / wired[k])
        print(f"  key {k}: switched {r.details['measured']:.4f} Hz, wired {wired[k]:.4f} Hz, {cents:+.3f} cents")
        assert r.status is ValidationStatus.PASS and abs(cents) < 1.0
        freq = r.details["frequency"]
        assert freq["t_start"] == r.details["params"]["t_start"] and freq["t_stop"] == r.details["params"]["t_stop"] and freq["edges"] >= MEASURE
        assert freq["t_start"] <= freq["first_edge_s"] < freq["last_edge_s"] <= freq["t_stop"]
        assert f" over [{freq['t_start']:g}, {freq['t_stop']:g}] s = " in r.message
    # silence before the first press and from 2 ms after the last release: no oscillation in either window
    data = read_results(ir.artifacts[ArtifactKind.SPICE_RESULT].path)["analyses"]["tran"]["result"]["vectors"]
    t, v = data["time"], data["q1c"]
    for window in ((t[0], 2e-3), (t[-1] - 2e-3, t[-1])):  # (a uic run saves its first point after 0: 20 ns here)
        quiet = window_frequency(t, v, *window)
        assert quiet.frequency is None and quiet.problem.startswith("no oscillation detected") and "in the window [" in quiet.problem
        assert max(x for x, y in zip(v, t) if window[0] <= y <= window[1]) < 0.2  # Q1 stays on: its collector low, no current in a driver
