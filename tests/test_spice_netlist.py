"""IR -> SPICE netlist compiler (``ai_eda.compilers.spice``) and the simulation IR.

No ngspice is needed: the netlist is a pure function of the IR, so the tests
pin the exact text of a divider netlist, its determinism, every refusal the
compiler makes instead of guessing, the analysis command text, and - when
kicad-cli and the KiCad libraries are installed - that kicad-cli's own SPICE
export of the compiled schematic describes the same passives (ref, node set,
value) after the documented normalisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SchematicCompiler, SpiceNetlistCompiler
from ai_eda.compilers.spice import (
    analysis_command,
    build,
    build_report,
    element_name,
    netlist_elements,
    spice_vector_name,
    stimulus_name,
)
from ai_eda.errors import CompileError, NothingToCompileError
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
    SimulationSetup,
    SourceRef,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    assumption,
    authoritative,
    llm_generated,
    user_requirement,
)
from ai_eda.tools.calc import ngspice_reads, parse_spice_number
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

DS = SourceRef(title="Generic resistor datasheet", authority="Vendor", content_hash="sha256:abc")
AUTH = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)
DERIVED = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")

DIODE_CARD = ".model D1N4148 D(Is=2.52n Rs=0.568 N=1.752 Cjo=4p M=0.4 tt=20n)"
OPAMP_CARD = ".subckt IDEALOPAMP inp inn out\nE1 out 0 inp inn 100k\n.ends IDEALOPAMP"

#: the golden netlist for :func:`divider_ir` - exact lines, LF, ``.end`` last
GOLDEN = "divider\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\nVVIN VIN 0 DC 12\n.end\n"


# --------------------------------------------------------------------------- inline fixtures


def _pin(number: str, kind: PinElectricalType = PinElectricalType.PASSIVE) -> Pin:
    return Pin(number=number, name=f"~{number}", electrical_type=kind, provenance=AUTH)


def component(ref: str, value: str, n_pins: int, spice: SpiceBinding | None) -> Component:
    return Component(ref=ref, value=value, pins=[_pin(str(i)) for i in range(1, n_pins + 1)], provenance=DERIVED, spice=spice)


def resistor(ref: str, ohms: float = 10_000.0, **binding) -> Component:
    return component(ref, "10k", 2, SpiceBinding(device=SpiceDevice.R, value=authoritative(ohms, DS, "ohm"), provenance=AUTH, **binding))


def connector(ref: str, n_pins: int = 3) -> Component:
    return component(ref, f"Conn_01x0{n_pins}", n_pins, SpiceBinding(exclude=True, exclude_reason="connector: no SPICE meaning", provenance=USER))


def net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=DERIVED)


def vin_stimulus(**kw) -> Stimulus:
    args = dict(id="VIN", source="voltage", net="VIN", reference_net="GND", kind=StimulusKind.DC, value=user_requirement(12.0, "V"), provenance=USER)
    args.update(kw)
    return Stimulus(**args)


def op_setup() -> SimulationSetup:
    return SimulationSetup(
        stimuli=[vin_stimulus()],
        analyses=[AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=USER)],
        expectations=[
            Expectation(id="vout", analysis_id="op", vector="v(VOUT)", reduce=Reduce.VALUE, nominal=user_requirement(6.0, "V"), tol_rel=user_requirement(0.01), requirement_id="req.v_out", provenance=USER)
        ],
    )


def divider_ir(order: tuple[str, ...] = ("R1", "R2", "J1"), tmp_path: Path | None = None) -> CircuitIR:
    """R1 / R2 = 10k divider with an excluded 3-pin header, 12 V DC stimulus on VIN, GND ground, op analysis."""
    parts = {"R1": resistor("R1"), "R2": resistor("R2"), "J1": connector("J1")}
    ir = CircuitIR(project=ProjectMeta(id="divider", name="divider", workdir=None if tmp_path is None else str(tmp_path)))
    ir.components = [parts[r] for r in order]
    ir.nets = [
        net("VIN", ("J1", "1"), ("R1", "1"), kind=NetKind.POWER),
        net("VOUT", ("J1", "2"), ("R1", "2"), ("R2", "1")),
        net("GND", ("J1", "3"), ("R2", "2"), kind=NetKind.GROUND),
    ]
    ir.parameters["v_in"] = user_requirement(12.0, "V")
    ir.simulation = op_setup()
    return ir


def analysis(kind: SpiceAnalysis, **params) -> AnalysisSpec:
    return AnalysisSpec(id=f"a_{kind.value}", kind=kind, params={k: user_requirement(v) for k, v in params.items()}, provenance=USER)


# --------------------------------------------------------------------------- golden text & determinism


def test_golden_divider_netlist():
    text = build(divider_ir())
    assert text == GOLDEN
    assert text.splitlines() == ["divider", "R1 VIN VOUT 10k", "R2 VOUT 0 10k", "VVIN VIN 0 DC 12", ".end"]
    assert "\r" not in text and text.endswith(".end\n")
    assert ".control" not in text and ".op" not in text and ".include" not in text


def test_netlist_is_deterministic_regardless_of_ir_order():
    a = build(divider_ir(("R1", "R2", "J1")))
    b = build(divider_ir(("J1", "R2", "R1")))
    assert a == b == GOLDEN
    ir = divider_ir()
    ir.nets.reverse()
    assert build(ir) == GOLDEN


def test_netlist_elements_parses_our_layout():
    assert netlist_elements(GOLDEN) == [
        ("R1", ["VIN", "VOUT"], "10k"),
        ("R2", ["VOUT", "0"], "10k"),
        ("VVIN", ["VIN", "0"], "DC 12"),
    ]
    text = "t\n.model D1N4148 D(Is=1n)\n.subckt A 1 2\nR1 1 2 1k\n.ends\nD1 a b D1N4148\nXU1 a b 0 A m=2\nQ1 c b e NPNX\n+ area=2\n.end\n"
    assert netlist_elements(text) == [
        ("D1", ["a", "b"], "D1N4148"),
        ("XU1", ["a", "b", "0"], "A m=2"),
        ("Q1", ["c", "b", "e"], "NPNX area=2"),
    ]
    with pytest.raises(ValueError):
        netlist_elements("t\nJ1 __J1\n.end\n")
    with pytest.raises(ValueError):
        netlist_elements("t\nR1 a\n.end\n")


def test_compile_writes_cir_and_report(tmp_path: Path):
    ir = divider_ir(tmp_path=tmp_path)
    ref = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path / "out"))
    path = Path(ref.path)
    assert path == tmp_path / "out" / "divider.cir"
    assert path.read_bytes() == GOLDEN.encode("utf-8")  # LF bytes, no BOM
    assert ref.kind == ArtifactKind.SPICE_NETLIST
    assert ref.generated_from_ir_hash == ir.content_hash() and not ref.is_stale(ir.content_hash())
    assert ref.generator == SpiceNetlistCompiler.id == "compiler.spice" and ref.generator_version == SpiceNetlistCompiler.version
    assert ref.matches_disk()
    report = json.loads(path.with_name("divider.cir.report.json").read_text(encoding="utf-8"))
    assert report["generator"] == SpiceNetlistCompiler.id
    assert report["excluded"] == [{"ref": "J1", "reason": "connector: no SPICE meaning"}]
    assert report["analyses"] == {"op": "op"} and report["vectors"] == {"vout": "vout"}
    # a second compile of an equal IR is byte-identical (no timestamps, no hashes in the text)
    again = SpiceNetlistCompiler().compile(divider_ir(("J1", "R1", "R2"), tmp_path=tmp_path), CompileContext(workdir=tmp_path / "again"))
    assert Path(again.path).read_bytes() == path.read_bytes() and again.content_hash == ref.content_hash


def test_build_report_contents():
    ir = divider_ir()
    ir.components.append(connector("J2", 1))
    ir.nets.append(net("AUX", ("J2", "1")))
    ir.nets.append(net("FLOATING"))
    report = build_report(ir)
    assert report["title"] == "divider"
    assert report["elements"] == ["R1", "R2"] and report["stimuli"] == ["VIN"]
    assert [e["ref"] for e in report["excluded"]] == ["J1", "J2"]
    assert report["nets_touching_only_excluded"] == ["AUX"]
    assert report["nets_without_pins"] == ["FLOATING"]
    assert report["ground_net"] == "GND"
    assert report["accepted_provenance_kinds"] == ["authoritative", "user_requirement"]
    assert report["assumptions"] == [] and report["models"] == []
    assert report["inexact_numbers"] == [] and report["unmodelled_numbers"] == []
    assert report["component_count"] == 4
    # a value ngspice cannot read back exactly (no spelling with < 2**53 - 1024 digits is read exactly) is reported
    ir.component("R2").spice.value = authoritative(6.000000000000001, DS, "ohm")
    report = build_report(ir)
    assert report["inexact_numbers"] == [{"what": "R2 value", "text": "6.000000000000001", "value": 6.000000000000001, "ngspice_reads": 6.000000000000002}]
    assert "R2 VOUT 0 6.000000000000001" in build(ir)
    ir.component("R2").spice.value = authoritative(1000.0000000000001, DS, "ohm")  # 17 digits: outside the modelled region
    assert build_report(ir)["unmodelled_numbers"] == [{"what": "R2 value", "text": "1.0000000000000001k", "value": 1000.0000000000001}]
    ir.component("R2").spice.value = authoritative(10_000.0, DS, "ohm")
    # an assumption is accepted but listed, so the reviewer can see what the numbers rest on
    ir.component("R1").spice.value = assumption(10_000.0, "nominal, not measured", "ohm")
    report = build_report(ir)
    assert "assumption" in report["accepted_provenance_kinds"] and report["assumptions"] == ["R1 value"]
    assert build(ir) == GOLDEN  # the text does not change with provenance


def test_ir_roundtrip_and_hash_cover_the_simulation():
    ir = divider_ir()
    h = ir.content_hash()
    loaded = CircuitIR.model_validate_json(ir.model_dump_json())
    assert loaded.content_hash() == h and build(loaded) == GOLDEN
    assert loaded.component("J1").spice.exclude and loaded.simulation.stimulus("VIN").kind == StimulusKind.DC
    ir.simulation.stimuli[0].value = user_requirement(9.0, "V")
    assert ir.content_hash() != h
    ir2 = divider_ir()
    ir2.component("R1").spice.pin_order = ["2", "1"]
    assert ir2.content_hash() != h and build(ir2).splitlines()[1] == "R1 VOUT VIN 10k"


# --------------------------------------------------------------------------- element naming, models, stimuli


def test_element_names_start_with_the_device_letter():
    assert element_name("R1", SpiceDevice.R) == "R1"
    assert element_name("U1", SpiceDevice.X) == "XU1"
    assert element_name("D3", SpiceDevice.R) == "RD3"
    assert element_name("rv1", SpiceDevice.R) == "rv1"
    assert stimulus_name(vin_stimulus()) == "VVIN"
    assert stimulus_name(vin_stimulus(source="current", value=user_requirement(1e-3, "A"))) == "IVIN"


def test_model_cards_are_verbatim_deduplicated_and_sorted():
    ir = divider_ir()
    diode = lambda ref: component(ref, "1N4148", 2, SpiceBinding(device=SpiceDevice.D, model_name="D1N4148", model_card=authoritative(DIODE_CARD, DS), pin_order=["2", "1"], provenance=AUTH))  # noqa: E731
    ir.components += [diode("D2"), diode("D1")]
    ir.nets.append(net("K", ("D1", "1"), ("D2", "1")))
    ir.net("VOUT").pins += [PinRef(component_ref="D1", pin_number="2"), PinRef(component_ref="D2", pin_number="2")]
    opamp = component("U1", "opamp", 3, SpiceBinding(device=SpiceDevice.X, model_name="IDEALOPAMP", model_card=user_requirement("\n" + OPAMP_CARD + "\n\n"), pin_order=["1", "2", "3"], provenance=USER))
    ir.components.append(opamp)
    ir.net("VOUT").pins.append(PinRef(component_ref="U1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="U1", pin_number="2"))
    ir.nets.append(net("OUT", ("U1", "3")))
    text = build(ir)
    assert text == (
        "divider\n"
        + DIODE_CARD + "\n"
        + OPAMP_CARD + "\n"
        + "D1 VOUT K D1N4148\nD2 VOUT K D1N4148\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\nXU1 VOUT 0 OUT IDEALOPAMP\nVVIN VIN 0 DC 12\n.end\n"
    )
    assert build_report(ir)["models"] == ["d1n4148", "idealopamp"]
    assert netlist_elements(text)[4] == ("XU1", ["VOUT", "0", "OUT"], "IDEALOPAMP")
    # a card that does not define the referenced model, or a model with no card at all, is refused
    ir.component("D1").spice.model_card = authoritative(DIODE_CARD.replace("D1N4148", "OTHER"), DS)
    with pytest.raises(CompileError, match="D1: model_card does not define 'D1N4148'"):
        build(ir)
    ir.component("D1").spice.model_card = None
    ir.component("D2").spice.model_card = None
    with pytest.raises(CompileError, match="D1: model 'D1N4148' is not defined by any model_card"):
        build(ir)
    # the netlist must stay self-contained: no .include smuggled in through a card
    ir.component("D1").spice.model_card = authoritative(".include diode.lib\n" + DIODE_CARD, DS)
    with pytest.raises(CompileError, match=r"D1: model_card line '.include diode.lib' is not allowed"):
        build(ir)


def test_subckt_port_count_and_kind_must_match():
    ir = divider_ir()
    u1 = component("U1", "opamp", 2, SpiceBinding(device=SpiceDevice.X, model_name="IDEALOPAMP", model_card=user_requirement(OPAMP_CARD), pin_order=["1", "2"], provenance=USER))
    ir.components.append(u1)
    ir.net("VOUT").pins.append(PinRef(component_ref="U1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="U1", pin_number="2"))
    with pytest.raises(CompileError, match=r"U1: .subckt IDEALOPAMP has 3 ports .* but pin_order has 2 pins"):
        build(ir)
    u1.spice = SpiceBinding(device=SpiceDevice.D, model_name="IDEALOPAMP", model_card=user_requirement(OPAMP_CARD), provenance=USER)
    with pytest.raises(CompileError, match="U1: a D element needs a .model, but 'IDEALOPAMP' is a .subckt"):
        build(ir)


def test_params_and_stimulus_kinds_render():
    ir = divider_ir()
    ir.component("R1").spice.params = {"tc1": user_requirement(1e-4), "m": user_requirement(2)}
    ir.simulation.stimuli = [
        vin_stimulus(kind=StimulusKind.PULSE, value=None, params={k: user_requirement(v) for k, v in dict(v1=0, v2=5, td=0, tr=1e-9, tf=1e-9, pw=1, per=2).items()}),
        vin_stimulus(id="AC1", net="VOUT", source="current", kind=StimulusKind.SINE, value=user_requirement(0.0), params={"vo": user_requirement(0), "va": user_requirement(1), "freq": user_requirement(1e3), "td": user_requirement(0), "ac": user_requirement(1)}),
        vin_stimulus(id="PW", net="VOUT", kind=StimulusKind.PWL, value=None, params={"points": user_requirement([[0, 0], [1e-3, 5], [2e-3, 0]])}),
    ]
    ir.simulation.expectations = []
    lines = build(ir).splitlines()
    assert lines[1] == "R1 VIN VOUT 10k m=2 tc1=1e-4"  # not "100u": ngspice would read that one ULP low
    assert lines[3:7] == [
        "IAC1 VOUT 0 DC 0 SINE(0 1 1k 0) AC 1",
        "VPW VOUT 0 PWL(0 0 1m 5 2m 0)",
        "VVIN VIN 0 PULSE(0 5 0 1n 1n 1 2)",
        ".end",
    ]
    ir.simulation.stimuli[0].params.pop("per")
    with pytest.raises(CompileError, match=r"stimulus VIN: pulse needs params .* missing \['per'\]"):
        build(ir)


def test_temperature_card_and_setup_lookups():
    ir = divider_ir()
    ir.simulation.temperature_c = user_requirement(85.0, "degC")
    assert build(ir).splitlines()[1] == ".temp 85"
    assert ir.simulation.analysis("op").kind == SpiceAnalysis.OP and ir.simulation.analysis("nope") is None
    assert ir.simulation.stimulus("VIN").net == "VIN" and ir.simulation.stimulus("nope") is None


# --------------------------------------------------------------------------- refusals (CompileError, never a guess)


def test_component_without_binding_is_refused():
    ir = divider_ir()
    ir.component("R1").spice = None
    with pytest.raises(CompileError, match="no SPICE binding for R1"):
        build(ir)
    for c in ir.components:
        c.spice = None
    with pytest.raises(NothingToCompileError, match="no SPICE binding for J1"):  # a CompileError subclass: the mapping stage has not run
        build(ir)
    with pytest.raises(NothingToCompileError, match="no components"):
        build(CircuitIR(project=ProjectMeta(id="empty", name="empty")))


def test_llm_generated_values_and_cards_are_refused():
    ir = divider_ir()
    ir.component("R1").spice.value = llm_generated(10_000.0, model="x", unit="ohm")
    with pytest.raises(CompileError, match="R1 value has llm_generated provenance"):
        build(ir)
    ir = divider_ir()
    d1 = component("D1", "1N4148", 2, SpiceBinding(device=SpiceDevice.D, model_name="D1N4148", model_card=llm_generated(DIODE_CARD, model="x"), provenance=AUTH))
    ir.components.append(d1)
    ir.net("VOUT").pins.append(PinRef(component_ref="D1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="D1", pin_number="2"))
    with pytest.raises(CompileError, match="D1 model_card has llm_generated provenance"):
        build(ir)
    d1.spice.model_card = assumption(DIODE_CARD, "looks right")
    with pytest.raises(CompileError, match="D1 model_card: a model card must have authoritative or user_requirement provenance, got assumption"):
        build(ir)
    d1.spice.model_card = authoritative(DIODE_CARD, DS)
    assert "D1 VOUT 0 D1N4148" in build(ir)
    ir.simulation.stimuli[0].value = llm_generated(12.0, model="x")
    with pytest.raises(CompileError, match="stimulus VIN value has llm_generated provenance"):
        build(ir)
    ir.simulation.stimuli[0].value = user_requirement(12.0)
    ir.simulation.expectations[0].nominal = llm_generated(6.0, model="x")
    with pytest.raises(CompileError, match="expectation vout nominal has llm_generated provenance"):
        build(ir)
    with pytest.raises(CompileError, match="analysis a_tran param stop has llm_generated provenance"):
        analysis_command(AnalysisSpec(id="a_tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(1e-6), "stop": llm_generated(1e-3, model="x")}, provenance=USER))


def test_ground_net_rules():
    ir = divider_ir()
    ir.net("GND").kind = NetKind.SIGNAL
    with pytest.raises(CompileError, match=r"exactly one NetKind.GROUND net is required .* found 0"):
        build(ir)
    ir.net("GND").kind = NetKind.GROUND
    ir.net("VIN").kind = NetKind.GROUND
    with pytest.raises(CompileError, match=r"found 2: \['VIN', 'GND'\]"):
        build(ir)
    ir = divider_ir()
    ir.net("GND").name = "AGND"  # any name works for the ground net: it becomes node 0
    ir.simulation.stimuli[0].reference_net = "AGND"
    assert build(ir) == GOLDEN
    ir.nets.append(net("gnd"))  # ... but a non-ground net ngspice would treat as ground is refused
    with pytest.raises(CompileError, match="net 'gnd' is not the GROUND net but ngspice treats 'gnd' as ground"):
        build(ir)


def test_nets_differing_only_by_case_are_refused():
    ir = divider_ir()
    ir.nets.append(net("vout"))
    with pytest.raises(CompileError, match="nets 'VOUT' and 'vout' differ only by case"):
        build(ir)
    ir = divider_ir()
    ir.nets.append(net("bad net"))
    with pytest.raises(CompileError, match="net 'bad net' is not usable as a SPICE node name"):
        build(ir)


def test_pin_order_must_match_the_component_pins():
    ir = divider_ir()
    ir.component("R1").spice.pin_order = ["1", "3"]
    with pytest.raises(CompileError, match=r"R1: pin_order \['1', '3'\] names pins \['3'\] that R1 does not have"):
        build(ir)
    ir.component("R1").spice.pin_order = ["1"]
    with pytest.raises(CompileError, match="R1: a R element takes 2 nodes, pin_order has 1"):
        build(ir)
    ir.component("R1").spice.pin_order = ["1", "1"]
    with pytest.raises(CompileError, match=r"R1: pin_order \['1', '1'\] repeats a pin"):
        build(ir)
    ir.component("R1").spice.pin_order = []
    with pytest.raises(CompileError, match="R1: pin_order is empty"):
        build(ir)


def test_pins_in_no_net_are_refused():
    ir = divider_ir()
    ir.net("GND").pins = [p for p in ir.net("GND").pins if p.component_ref != "R2"]
    with pytest.raises(CompileError, match="R2.2 is used by the SPICE element but is in no net"):
        build(ir)
    # an unused pin of an included part must be in a net too, unless the IR says no_connect
    ir = divider_ir()
    q = component("Q1", "opamp", 4, SpiceBinding(device=SpiceDevice.X, model_name="IDEALOPAMP", model_card=user_requirement(OPAMP_CARD), pin_order=["1", "2", "3"], provenance=USER))
    ir.components.append(q)
    ir.net("VIN").pins.append(PinRef(component_ref="Q1", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="Q1", pin_number="2"))
    ir.nets.append(net("OUT", ("Q1", "3")))
    with pytest.raises(CompileError, match=r"Q1.4 \(passive\) is in no net; connect it or mark it no_connect"):
        build(ir)
    q.pins[3] = _pin("4", PinElectricalType.NO_CONNECT)
    assert "XQ1 VIN 0 OUT IDEALOPAMP" in build(ir)
    # excluded parts are not checked
    ir.net("VIN").pins = [p for p in ir.net("VIN").pins if p.component_ref != "J1"]
    assert "XQ1 VIN 0 OUT IDEALOPAMP" in build(ir)


def test_stimulus_on_unknown_net_is_refused():
    ir = divider_ir()
    ir.simulation.stimuli[0].net = "VCC"
    with pytest.raises(CompileError, match="stimulus VIN: unknown net 'VCC'"):
        build(ir)
    ir.simulation.stimuli[0].net = "GND"
    with pytest.raises(CompileError, match="stimulus VIN: net and reference_net are both 'GND'"):
        build(ir)
    ir.simulation.stimuli[0].net = "VIN"
    ir.simulation.stimuli[0].id = "bad id"
    with pytest.raises(CompileError, match="stimulus bad id: id must be a plain identifier"):
        build(ir)
    ir = divider_ir()
    ir.simulation.stimuli.append(vin_stimulus(id="R1"))  # would be element VR1 - fine; but a clash is refused:
    ir.simulation.stimuli.append(vin_stimulus(id="r1"))
    with pytest.raises(CompileError, match="element name 'Vr1' collides with stimulus R1"):
        build(ir)


def test_expectation_vector_rules():
    ir = divider_ir()
    assert spice_vector_name("v(VOUT)", ir) == "vout"
    assert spice_vector_name("V( vout )", ir) == "vout"
    assert spice_vector_name("i(VIN)", ir) == "vvin#branch"
    for bad, msg in [
        ("v(VX)", "unknown net 'VX'"),
        ("i(VZ)", "'VZ' is neither a stimulus id nor a component ref"),
        ("v(GND)", "'GND' is the ground net"),
        ("i(R1)", "R1 is not in the netlist as a V device"),
        ("i(J1)", "J1 is not in the netlist as a V device"),
        ("vout", "not of the form"),
        ("v(VOUT)-v(VIN)", "not of the form"),
        ("v(VOUT,VIN)", "unknown net 'VOUT,VIN'"),  # no expression grammar: a comma is just an impossible net name
    ]:
        with pytest.raises(CompileError, match=msg):
            spice_vector_name(bad, ir)
    ir.simulation.expectations[0].vector = "v(VX)"
    with pytest.raises(CompileError, match="unknown net 'VX'"):
        build(ir)
    ir.simulation.expectations[0].vector = "v(VOUT)"
    ir.simulation.expectations[0].analysis_id = "tran"
    with pytest.raises(CompileError, match="expectation vout: unknown analysis 'tran'"):
        build(ir)
    ir.simulation.expectations[0].analysis_id = "op"
    ir.simulation.expectations[0].reduce = Reduce.FINAL
    with pytest.raises(CompileError, match="expectation vout: an op result is a single point; use reduce=value"):
        build(ir)


def test_reduce_frequency_is_only_for_a_tran_analysis():
    """Rising edges are counted over time: op has one point, dc / ac sweep a source or a frequency, not time."""
    ir = divider_ir()
    ir.simulation.stimuli[0].params["ac"] = user_requirement(1)
    ir.simulation.analyses += [
        AnalysisSpec(id="dc", kind=SpiceAnalysis.DC, params={"source": user_requirement("VIN"), "start": user_requirement(0), "stop": user_requirement(12), "step": user_requirement(1)}, provenance=USER),
        AnalysisSpec(id="ac", kind=SpiceAnalysis.AC, params={"variation": user_requirement("dec"), "points": user_requirement(10), "fstart": user_requirement(1.0), "fstop": user_requirement(1e6)}, provenance=USER),
        AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(1e-6), "stop": user_requirement(5e-3), "uic": user_requirement(True)}, provenance=USER),
    ]
    exp = ir.simulation.expectations[0]
    exp.reduce = Reduce.FREQUENCY
    exp.nominal = user_requirement(1000.0, "Hz")
    with pytest.raises(CompileError, match="expectation vout: an op result is a single point; use reduce=value"):
        build(ir)
    for analysis_id in ("dc", "ac"):
        exp.analysis_id = analysis_id
        with pytest.raises(CompileError, match=f"expectation vout: reduce=frequency counts rising edges over time, which only a tran analysis produces \\({analysis_id} is {analysis_id}\\)"):
            build(ir)
    exp.analysis_id = "tran"
    report = build_report(ir)
    assert report["analyses"]["tran"] == "tran 1u 5m uic"
    assert build(ir) == GOLDEN.replace("DC 12", "DC 12 AC 1")  # no analysis, no ``uic`` and no expectation ever reaches the netlist text


def test_dc_analysis_must_reference_a_stimulus():
    setup = op_setup()
    setup.analyses.append(AnalysisSpec(id="sweep", kind=SpiceAnalysis.DC, params={"source": user_requirement("VX"), "start": user_requirement(0), "stop": user_requirement(12), "step": user_requirement(1)}, provenance=USER))
    ir = divider_ir()
    ir.simulation = setup
    with pytest.raises(CompileError, match="analysis sweep: dc sweeps unknown stimulus 'VX'"):
        build(ir)
    with pytest.raises(CompileError, match="dc sweeps unknown stimulus 'VX'"):
        analysis_command(setup.analyses[1], setup)
    setup.analyses[1].params["source"] = user_requirement("VIN")
    assert analysis_command(setup.analyses[1], setup) == "dc VVIN 0 12 1"
    assert build_report(ir)["analyses"] == {"op": "op", "sweep": "dc VVIN 0 12 1"}


# --------------------------------------------------------------------------- analysis commands


def test_analysis_commands():
    setup = op_setup()
    setup.stimuli[0].params["ac"] = user_requirement(1)
    assert analysis_command(analysis(SpiceAnalysis.OP), setup) == "op"
    assert analysis_command(analysis(SpiceAnalysis.DC, source="VIN", start=0, stop=12, step=1), setup) == "dc VVIN 0 12 1"
    assert analysis_command(analysis(SpiceAnalysis.DC, source="VIN", start=12, stop=0, step=-0.5), setup) == "dc VVIN 12 0 -500m"
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-6, stop=5e-3)) == "tran 1u 5m"
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-5, stop=5e-3, start=1e-3)) == "tran 1e-5 5m 1m"  # "10u" would be misread
    # ``uic`` is a bool: True appends ngspice's keyword after start (or after stop without one), False emits nothing
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=5e-6, stop=30e-3, start=10e-3, uic=True)) == "tran 5.00u 30m 10m uic"
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-6, stop=5e-3, uic=True)) == "tran 1u 5m uic"
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-6, stop=5e-3, uic=False)) == "tran 1u 5m"
    assert analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-6, stop=5e-3, start=1e-3, uic=False)) == "tran 1u 5m 1m"
    assert analysis_command(analysis(SpiceAnalysis.AC, variation="dec", points=10, fstart=1, fstop=1e6), setup) == "ac dec 10 1 1meg"
    assert analysis_command(analysis(SpiceAnalysis.AC, variation="LIN", points=100, fstart=1e3, fstop=2e3)) == "ac lin 100 1k 2k"
    # every number parses back to the IR value (m = milli, meg = mega) - by our parser and by ngspice's arithmetic
    for command, expected in (("tran 1e-5 5m 1m", [1e-5, 5e-3, 1e-3]), ("tran 1u 5m", [1e-6, 5e-3]), ("ac dec 10 1 1meg", [10, 1, 1e6])):
        numbers = command.split()[2 if command.startswith("ac") else 1:]
        assert [parse_spice_number(t) for t in numbers] == expected
        assert [ngspice_reads(t) for t in numbers] == expected


def test_analysis_command_refusals():
    setup = op_setup()
    with pytest.raises(CompileError, match=r"analysis a_tran: tran needs params .* missing \['stop'\]"):
        analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-5))
    with pytest.raises(CompileError, match=r"analysis a_op: unexpected params \['stop'\] for op"):
        analysis_command(analysis(SpiceAnalysis.OP, stop=1))
    with pytest.raises(CompileError, match="tran needs 0 < step <= stop"):
        analysis_command(analysis(SpiceAnalysis.TRAN, step=1, stop=1e-3))
    with pytest.raises(CompileError, match="tran start must satisfy"):
        analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-5, stop=1e-3, start=2e-3))
    for bad in (1, 0, 1.0, "uic", "true", None):  # ngspice would take any token; the IR takes only a bool
        with pytest.raises(CompileError, match=f"analysis a_tran param uic must be a bool .*got {type(bad).__name__}"):
            analysis_command(analysis(SpiceAnalysis.TRAN, step=1e-5, stop=1e-3, uic=bad))
    with pytest.raises(CompileError, match=r"analysis a_dc: unexpected params \['uic'\] for dc"):
        analysis_command(analysis(SpiceAnalysis.DC, source="VIN", start=0, stop=12, step=1, uic=True), setup)
    with pytest.raises(CompileError, match="llm_generated"):
        analysis_command(AnalysisSpec(id="a_tran", kind=SpiceAnalysis.TRAN, params={"step": user_requirement(1e-6), "stop": user_requirement(1e-3), "uic": llm_generated(True, model="x")}, provenance=USER))
    with pytest.raises(CompileError, match="dc sweep 0.0 -> 12.0 with step -1.0 never terminates"):
        analysis_command(analysis(SpiceAnalysis.DC, source="VIN", start=0, stop=12, step=-1), setup)
    with pytest.raises(CompileError, match=r"ac variation must be one of \['dec', 'oct', 'lin'\], got 'log'"):
        analysis_command(analysis(SpiceAnalysis.AC, variation="log", points=10, fstart=1, fstop=1e6))
    with pytest.raises(CompileError, match="ac points must be a positive integer, got 0"):
        analysis_command(analysis(SpiceAnalysis.AC, variation="dec", points=0, fstart=1, fstop=1e6))
    with pytest.raises(CompileError, match="ac needs 0 < fstart <= fstop"):
        analysis_command(analysis(SpiceAnalysis.AC, variation="dec", points=10, fstart=0, fstop=1e6))
    with pytest.raises(CompileError, match="an ac analysis needs a stimulus with an 'ac' magnitude param; none has one"):
        analysis_command(analysis(SpiceAnalysis.AC, variation="dec", points=10, fstart=1, fstop=1e6), setup)


# --------------------------------------------------------------------------- cross-check against kicad-cli's own SPICE export

_probe = KicadLibrary()
HAS_LIBS = _probe.symbol_file("Device") is not None and _probe.symbol_file("Connector_Generic") is not None
_kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not (_kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")

Passive = tuple[str, frozenset[str], float]
_PASSIVE_KEYS = {"R": "resistance", "C": "capacitance", "L": "inductance"}


def kicad_node(name: str) -> str:
    """KiCad SPICE node -> comparable node: strip one sheet-path ``/``, fold case, ``gnd`` -> ``0`` (ngspice's alias)."""
    if name.startswith("/"):
        name = name[1:]
    name = name.lower()
    return "0" if name in ("gnd", "0") else name


def kicad_passives(text: str) -> tuple[set[Passive], list[str]]:
    """``(R/C/L tuples, problems)`` from a ``kicad-cli sch export netlist --format spice`` file.

    Only element lines matter; ``.title`` / ``.include`` / ``.control`` blocks
    and comments are skipped. A ``__<ref>`` placeholder (KiCad's output for a
    symbol it has no model for; ngspice refuses the whole circuit on it), a
    non-numeric value, a relative ``.include`` (unresolved library) or a
    malformed passive line is a problem, never silently dropped.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("+") and lines:
            lines[-1] += " " + raw[1:]
        else:
            lines.append(raw)
    tuples: set[Passive] = set()
    problems: list[str] = []
    in_control = False
    for line in lines:
        s = line.strip()
        if not s or s[0] == "*":
            continue
        first = s.split()[0].lower()
        if first == ".control":
            in_control = True
            continue
        if first == ".endc":
            in_control = False
            continue
        if in_control:
            continue
        if first == ".include":
            inc = s[len(".include"):].strip().strip('"')
            if not Path(inc).is_absolute():
                problems.append(f"unresolved model library: {s!r}")
            continue
        if s[0] == ".":
            continue
        toks = s.split()
        if any(t.startswith("__") for t in toks[1:]):
            problems.append(f"placeholder (no SPICE model) element line: {s!r}")
            continue
        if toks[0][0].upper() not in _PASSIVE_KEYS:
            continue  # not an R/C/L: not part of this cross-check
        if len(toks) < 4 or any("=" not in t for t in toks[4:]):
            problems.append(f"malformed passive element line: {s!r}")
            continue
        try:
            value = parse_spice_number(toks[3], ignore_trailing_letters=True)
        except ValueError:
            problems.append(f"non-numeric value {toks[3]!r} in {s!r}")
            continue
        tuples.add((toks[0].upper(), frozenset(kicad_node(n) for n in toks[1:3]), value))
    return tuples, problems


def our_passives(text: str) -> set[Passive]:
    """The same tuples from our own netlist (nodes are already ``0`` for ground; ngspice folds case)."""
    return {
        (name.upper(), frozenset(n.lower() for n in nodes), parse_spice_number(rest.split()[0]))
        for name, nodes, rest in netlist_elements(text)
        if name[0].upper() in _PASSIVE_KEYS
    }


def same_passives(a: set[Passive], b: set[Passive]) -> bool:
    if len(a) != len(b):
        return False
    return all(any(r == ref and n == nodes and math.isclose(v, value, rel_tol=1e-9) for r, n, v in b) for ref, nodes, value in a)


def test_cross_check_normalisation_on_recorded_kicad_output():
    """The exact text kicad-cli 10.0.6 wrote for the divider fixture (CRLF, GND kept, J1 placeholder)."""
    recorded = ".title KiCad schematic\r\nJ1 __J1\r\nR1 VIN VOUT 10k\r\nR2 VOUT GND 10k\r\n.end"
    got, problems = kicad_passives(recorded)
    assert got == {("R1", frozenset({"vin", "vout"}), 10_000.0), ("R2", frozenset({"0", "vout"}), 10_000.0)}
    assert problems == ["placeholder (no SPICE model) element line: 'J1 __J1'"]
    assert same_passives(got, our_passives(GOLDEN))
    # documented KiCad quirks the normalisation must catch or reject
    assert kicad_passives(".title x\r\nR1 /VIN Net-_R1-Pad2_ 1Meg\r\n.end")[0] == {("R1", frozenset({"vin", "net-_r1-pad2_"}), 1e6)}
    assert kicad_passives(".title x\nR1 VIN VOUT 4k7\n.end")[1] == ["non-numeric value '4k7' in 'R1 VIN VOUT 4k7'"]
    assert kicad_passives(".title x\nR1 VIN VOUT 10 kOhm\n.end")[1] == ["malformed passive element line: 'R1 VIN VOUT 10 kOhm'"]
    assert kicad_passives(".title x\nR1 __R1\n.end")[1] == ["placeholder (no SPICE model) element line: 'R1 __R1'"]
    assert kicad_passives('.title x\n.include "missing.mod"\n.end')[1] == ['unresolved model library: \'.include "missing.mod"\'']
    assert kicad_passives(".title x\n.control\nR1 a b 1k\n.endc\nR1 VIN GND 10kOhm\n.end") == ({("R1", frozenset({"vin", "0"}), 1e4)}, [])
    assert not same_passives(our_passives(GOLDEN), kicad_passives(".title x\nR1 VIN VOUT 1M\nR2 VOUT GND 10k\n.end")[0])  # 1M is milli to ngspice


def bind_fixture(ir: CircuitIR) -> CircuitIR:
    """Bind the shared KiCad fixture: R -> R with its authoritative resistance, J -> excluded."""
    for c in ir.components:
        if c.ref.startswith("R"):
            value = c.electrical.get("resistance") or authoritative(10_000.0, DS, "ohm")
            c.spice = SpiceBinding(device=SpiceDevice.R, value=value, provenance=AUTH)
        else:
            c.spice = SpiceBinding(exclude=True, exclude_reason="connector: no SPICE meaning", provenance=USER)
    return ir


@needs_kicad
def test_kicad_cli_spice_export_matches_our_passives(tmp_path: Path):
    from tests.fixtures_kicad import PROJECT_ID, divider_with_connector_ir

    lib = KicadLibrary()
    ir = bind_fixture(divider_with_connector_ir(tmp_path, lib))
    ours = build(ir)
    assert ours == f"{PROJECT_ID}\nR1 VIN VOUT 10k\nR2 VOUT 0 10k\nVVIN VIN 0 DC 12\n.end\n"
    sch = Path(SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib})).path)
    sch_text = sch.read_text(encoding="utf-8")
    # the schematic compiler marks the excluded header (and any part without a binding) exclude_from_sim;
    # the two embedded library symbols (Device:R, Conn_01x03) carry their own `exclude_from_sim no`
    assert sch_text.count("(exclude_from_sim yes)") == 1 and sch_text.count("(exclude_from_sim no)") == 2 + 2
    exported = _kicad.export_netlist(sch, tmp_path / "kicad.cir", fmt="spice").read_text(encoding="utf-8")
    got, problems = kicad_passives(exported)
    assert same_passives(got, our_passives(ours)), (got, ours)
    assert got == {("R1", frozenset({"vin", "vout"}), 10_000.0), ("R2", frozenset({"0", "vout"}), 10_000.0)}
    # exclude_from_sim removes KiCad's `J1 __J1` placeholder (which made ngspice refuse the whole file)
    assert problems == [], problems
    assert "__J1" not in exported and exported.startswith(".title KiCad schematic")


@needs_kicad
def test_schematic_marks_unbound_parts_exclude_from_sim(tmp_path: Path):
    """A part with no SpiceBinding at all is excluded from KiCad's simulation too - the IR has no model for it."""
    from tests.fixtures_kicad import divider_with_connector_ir

    lib = KicadLibrary()
    ir = divider_with_connector_ir(tmp_path, lib)
    ir.component("R2").spice = None
    sch_text = Path(SchematicCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"kicad_library": lib})).path).read_text(encoding="utf-8")
    assert sch_text.count("(exclude_from_sim yes)") == 2 and sch_text.count("(exclude_from_sim no)") == 2 + 1  # 2 library symbols + R1


def test_net_names_the_runner_would_reject_are_refused_by_the_compiler():
    """The compiler applies the runner's node rule (NODE_RE) and validate_deck, so what compiles always runs."""
    for name in ("N$1", "n{1}", "n;1", "n`1", "넷"):
        ir = divider_ir()
        ir.nets.append(net(name))
        with pytest.raises(CompileError, match="not usable as a SPICE node name"):
            build(ir)
    ir = divider_ir()
    ir.simulation.analyses.append(AnalysisSpec(id="bad id", kind=SpiceAnalysis.OP, provenance=USER))
    with pytest.raises(CompileError, match="analysis id 'bad id' must be a plain identifier"):
        build(ir)
    ir = divider_ir()
    ir.simulation.expectations[0].id = "v out"
    with pytest.raises(CompileError, match="expectation id 'v out' must be a plain identifier"):
        build(ir)
