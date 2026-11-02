"""The KR 447 MHz RF design library (``ai_eda/design/rf``, part P8 of the kr447 design §5).

What runs where:

* always: the model values and cards (marked UNVERIFIED choices; every card
  through the SPICE compiler, which accepts the op-amp macro's E / R / C
  lines, the crystal and potentiometer subcircuits and the P-FET wrapper with
  its nested ``.model``), the inductor-Q bands, the regulatory profile
  (placeholders never grounded; the 25-channel raster and its refusal), the
  family table (the kr447 design §2.0 serve / need table, the closed-world
  message, the required selector question), the registry's four stage templates, the block
  API (prefixing, structure checks, composition, ``exclude_floating``) and
  the name-based pin lookup on synthetic libraries (a missing or ambiguous
  name, a changed stack, a lost description fact, a pad without a pin all
  refuse);
* with ngspice (``needs_ngspice``): each card simulated - the op-amp's
  open-loop gain (1e5 at DC, -3 dB at 10 Hz, unity at 1 MHz), the crystal's
  series resonance at 21.4 MHz, the varactor's 10.18 pF at 2 V reverse, the
  P-FET switch, the potentiometer's divider - and a deck whose floating
  capacitors ``exclude_floating`` removed;
* with the packed KiCad 10.0.6 libraries (``needs_libs``,
  ``KICAD10_SYMBOL_DIR``): every parts-table row instantiates (the
  microphone's two through-hole pads carry its symbol's pin numbers, its
  polarity UNVERIFIED), the three refused rows refuse with the compilers'
  reasons, and one of every row compiles through the real schematic and
  PCB compilers;
* with the kr447 wave-1 IR types (``ai_eda.ir.rf``, part P1) and requirement
  keys (``RADIO_BUILDS``, part P3): the contract field names the block API
  renames through, and the build names - skipped until those parts are merged.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import BaseModel

from ai_eda.compilers.base import CompileContext
from ai_eda.compilers.spice import analysis_command, build
from ai_eda.design import inputs as design_inputs
from ai_eda.design.base import CHOICE_NOTE_PREFIX, Choice
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf import family, models, profile
from ai_eda.design.rf.blocks import (
    Block,
    BlockBuilder,
    BlockContext,
    BlockPrefix,
    BlockResult,
    apply_prefix,
    exclude_floating,
    merge_results,
    rename_vector,
)
from ai_eda.design.rf.parts import PARTS, REFUSED_PARTS, PlacedPart, check_refused, instantiate, symbol_stacks
from ai_eda.design.rf.registry import RF_TEMPLATES
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    BoardOutline,
    BoardSide,
    CircuitIR,
    Component,
    Constraint,
    ConstraintKind,
    Expectation,
    Net,
    NetKind,
    PCBDesign,
    Pin,
    PinElectricalType,
    PinRef,
    Placement,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    Requirement,
    RequirementKind,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    ValidationStatus,
    user_requirement,
)
from ai_eda.tools.calc.recompute import recompute_parameters
from ai_eda.tools.calc.si import format_spice_number
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S as SX
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test")
STRUCT = Provenance(kind=ProvenanceKind.DERIVED, tool="design.template.test")
runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_AM_FM", "RF_Mixer", "Oscillator", "Amplifier_Operational")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-105_38.10x25.40mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")


# --------------------------------------------------------------------------- helpers


def _pins(*spec: tuple[str, str]) -> list[Pin]:
    return [Pin(number=n, name=name, provenance=USER) for n, name in spec]


def _placed(key: str, ref: str, functions: dict[str, tuple[str, ...]], names: dict[str, str] | None = None) -> PlacedPart:
    """A PlacedPart of row ``key`` without a library: pins numbered as ``functions`` says (names from ``names``, else empty)."""
    numbers = sorted({n for ns in functions.values() for n in ns}, key=lambda s: (len(s), s))
    comp = Component(ref=ref, value=ref, pins=_pins(*[(n, (names or {}).get(n, "")) for n in numbers]), provenance=STRUCT)
    return PlacedPart(part=PARTS[key], component=comp, functions=functions)


def _net(name: str, *pins: tuple[str, str], kind: NetKind = NetKind.SIGNAL) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=STRUCT)


def _dc(id_: str, net: str, v: float, ac: float | None = None) -> Stimulus:
    params = {} if ac is None else {"ac": user_requirement(ac, "V")}
    return Stimulus(id=id_, source="voltage", net=net, reference_net="GND", kind=StimulusKind.DC, value=user_requirement(v, "V"), params=params, provenance=STRUCT)


def _resistor(ref: str, ohm: float) -> Component:
    return Component(ref=ref, value=ref, pins=_pins(("1", ""), ("2", "")), provenance=STRUCT,
                     spice=SpiceBinding(device=SpiceDevice.R, value=user_requirement(ohm, "ohm"), provenance=STRUCT))


def _capacitor(ref: str, farad: float) -> Component:
    return Component(ref=ref, value=ref, pins=_pins(("1", ""), ("2", "")), provenance=STRUCT,
                     spice=SpiceBinding(device=SpiceDevice.C, value=user_requirement(farad, "F"), provenance=STRUCT))


def _ir(tmp_path: Path, name: str) -> CircuitIR:
    return CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(tmp_path)))


def _ac(fstart: float, fstop: float, points: int, variation: str = "lin") -> AnalysisSpec:
    return AnalysisSpec(id="ac", kind=SpiceAnalysis.AC, params={"variation": Traced(value=variation, provenance=STRUCT), "points": user_requirement(points),
                                                                "fstart": user_requirement(fstart, "Hz"), "fstop": user_requirement(fstop, "Hz")}, provenance=STRUCT)


def _run(ir: CircuitIR, tmp_path: Path, analysis: SpiceAnalysis = SpiceAnalysis.AC):
    deck = tmp_path / f"{ir.project.id}.cir"
    deck.write_text(build(ir), encoding="utf-8", newline="\n")
    command = analysis_command(ir.simulation.analyses[0], ir.simulation) if ir.simulation and ir.simulation.analyses else None
    res = runner.run(deck, analysis, tmp_path / ir.project.id, command)
    assert res.succeeded, res.errors
    return res


def _choice(key: str) -> Traced:
    return models.model_choice("test", key, True)[1]


# --------------------------------------------------------------------------- model values and cards


def test_model_values_are_unverified_choices_under_the_model_prefix() -> None:
    assert models.MODEL_VALUES, "the model-value table is empty"
    for key, mv in models.MODEL_VALUES.items():
        assert key.startswith(models.MODEL_PREFIX)
        assert models.MODEL_VALUE_PHRASE in mv.description and "[UNVERIFIED: " in mv.description, key
        ch, pending = models.model_choice("t_rf", key, False)
        assert isinstance(ch, Choice) and "UNVERIFIED" in ch.text() and ch.key == key
        assert pending.provenance.kind is ProvenanceKind.ASSUMPTION  # not a design value before the table is confirmed
        _, confirmed = models.model_choice("t_rf", key, True)
        assert confirmed.provenance.kind is ProvenanceKind.USER_REQUIREMENT and confirmed.provenance.note.startswith(CHOICE_NOTE_PREFIX)
        assert confirmed.value == pending.value == mv.value and confirmed.unit == mv.unit
    # the design's named values (kr447 design §2.0 table and the update pass; model.xtal21.cm 16 fF since the wave-2 merge: 6 fF admits no 7.5 kHz ladder)
    expect = {"model.xtal21.cm": 16e-15, "model.xtal21.rm": 25.0, "model.xtal21.c0": 4e-12, "model.sa605.port_r": 1500.0, "model.bfr92.r_out": 1000.0,
              "model.bfr92.r_in": 500.0, "model.pin.r_on": 1.0, "model.pin.c_off": 0.3e-12, "model.buf.r_in": 10e3, "model.buf.r_out": 50.0,
              "model.opamp.a0": 1e5, "model.opamp.gbw": 1e6, "model.varactor.cjo": 20e-12, "model.varactor.vj": 0.7, "model.varactor.m": 0.5}
    assert {k: models.MODEL_VALUES[k].value for k in expect} == expect


def test_a_model_value_outside_the_prefix_or_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="must start with 'model.'"):
        models.ModelValue("pm.q_l", 10.0, None, "loaded Q", "nothing")
    with pytest.raises(ValueError, match="finite"):
        models.ModelValue("model.x", math.inf, None, "x", "nothing")
    own = models.ModelValue("model.mic_level", 0.01, "V", "microphone level at MICOUT", "Maxim MAX9814 datasheet")
    ch, traced = models.model_choice("t_rf", own, True)  # a block's own model value goes through the same marking
    assert "[UNVERIFIED: Maxim MAX9814 datasheet]" in ch.text() and traced.value == 0.01


@pytest.mark.parametrize(("f_hz", "key", "q"), [
    (450e3, "model.l_q.if2", 50.0), (21.4e6, "model.l_q.if1", 30.0), (20.95e6, "model.l_q.if1", 30.0),
    (35.5135417e6, "model.l_q.hf", 40.0), (37.296875e6, "model.l_q.hf", 40.0),
    (106.540625e6, "model.l_q.vhf", 40.0), (111.890625e6, "model.l_q.vhf", 40.0), (213.08125e6, "model.l_q.vhf", 40.0), (223.78125e6, "model.l_q.vhf", 40.0),
    (426.1625e6, "model.l_q.uhf", 40.0), (447.5625e6, "model.l_q.uhf", 40.0), (895.125e6, "model.l_q.uhf", 40.0),
])
def test_every_tank_frequency_has_an_inductor_q(f_hz: float, key: str, q: float) -> None:
    assert models.inductor_q_key(f_hz) == key and models.MODEL_VALUES[key].value == q


def test_a_frequency_outside_every_q_band_is_refused() -> None:
    for f in (50e3, 1.2e9):
        with pytest.raises(ValueError, match="would be a guess"):
            models.inductor_q_key(f)


def test_fixed_cards_are_the_design_texts() -> None:
    assert models.npn_card().text == ".model QNPN NPN (TR=200n)"  # the astable template's card (model.npn)
    assert models.diode_card().text == ".model DG D"
    pmos = models.pmos_card()
    assert pmos.text.splitlines() == [".subckt PMOSG3 d g s", "M1 d g s s PMOSG", ".model PMOSG PMOS (VTO=-1 KP=0.5)", ".ends"]
    for card in (models.npn_card(), models.diode_card(), pmos):
        assert "[UNVERIFIED: " in card.description and card.key.startswith(models.MODEL_PREFIX)


def test_varactor_and_opamp_cards_are_spelled_from_their_model_values() -> None:
    var = models.varactor_card(_choice("model.varactor.cjo"), _choice("model.varactor.vj"), _choice("model.varactor.m"))
    assert var.text == f".model DVAR D (CJO={format_spice_number(20e-12)} VJ={format_spice_number(0.7)} M={format_spice_number(0.5)})"
    assert var.spelled_from == ("model.varactor.cjo", "model.varactor.vj", "model.varactor.m")
    with pytest.raises(ValueError, match=r"lie in \(0, 1\)"):
        models.varactor_card(_choice("model.varactor.cjo"), _choice("model.varactor.vj"), user_requirement(1.0))
    opa = models.opamp_card(_choice("model.opamp.a0"), _choice("model.opamp.gbw"))
    c_g = 1e5 / (2 * math.pi * 1e3 * 1e6)  # the pole at GBW / A0 = 10 Hz behind R_g = 1 kohm
    assert opa.text.splitlines() == [".subckt OPA1P inp inn out", "Eg g 0 inp inn 100k", "Rg g p 1k", f"Cg p 0 {format_spice_number(c_g)}", "Eo out 0 p 0 1", ".ends"]
    assert c_g == pytest.approx(15.9155e-6, rel=1e-5)  # the decided splatter decks' macro


def test_crystal_lm_is_a_calculator_output_that_recompute_rederives(tmp_path: Path) -> None:
    ir = _ir(tmp_path, "xtal")
    ir.parameters["rf.if1"] = Traced(value=21.4e6, unit="Hz", provenance=USER)
    ir.parameters["model.xtal21.cm"] = _choice("model.xtal21.cm")
    lm = models.crystal_lm(ir.parameters["rf.if1"], ir.parameters["model.xtal21.cm"])
    assert lm.provenance.tool == "calc.rf.lc.l_for_resonance" and lm.provenance.derived_from == ["rf.if1", "model.xtal21.cm"]
    assert lm.value == pytest.approx(1.0 / ((2 * math.pi * 21.4e6) ** 2 * 16e-15)) and lm.value == pytest.approx(3.457e-3, rel=1e-3)
    ir.parameters["xtal21.lm"] = lm
    assert recompute_parameters(ir).status is ValidationStatus.PASS
    card = models.crystal_card(lm, _choice("model.xtal21.cm"), _choice("model.xtal21.rm"), _choice("model.xtal21.c0"))
    assert card.text.splitlines()[0] == ".subckt XTAL21 1 2" and f"Lm 1 a {format_spice_number(lm.value)}" in card.text
    with pytest.raises(ValueError, match="C_0 must be positive"):
        models.crystal_card(lm, _choice("model.xtal21.cm"), _choice("model.xtal21.rm"), user_requirement(0.0, "F"))


def test_potentiometer_card_refuses_an_end_position() -> None:
    card = models.potentiometer_card(10e3, 0.5)
    assert card.name == "POT_10000_500" and card.text.splitlines()[1:3] == ["Ra a w 5k", "Rb w b 5k"]
    for pos in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            models.potentiometer_card(10e3, pos)


def _card_board(tmp_path: Path, confirmed: bool = True) -> CircuitIR:
    """One op-amp, crystal, P-FET, varactor, NPN, diode and potentiometer bound through ``card_binding`` (no library needed)."""
    t = "t_rf"
    cards = {
        "opamp": models.opamp_card(_choice("model.opamp.a0"), _choice("model.opamp.gbw")),
        "xtal": models.crystal_card(models.crystal_lm(user_requirement(21.4e6, "Hz"), _choice("model.xtal21.cm")), _choice("model.xtal21.cm"),
                                    _choice("model.xtal21.rm"), _choice("model.xtal21.c0")),
        "pmos": models.pmos_card(),
        "var": models.varactor_card(_choice("model.varactor.cjo"), _choice("model.varactor.vj"), _choice("model.varactor.m")),
        "npn": models.npn_card(),
        "diode": models.diode_card(),
        "pot": models.potentiometer_card(10e3, 0.5),
    }
    text = {k: models.card_choice(t, c, confirmed)[1] for k, c in cards.items()}
    u1 = _placed("opamp", "U1", {"OUT": ("1",), "V-": ("2",), "+": ("3",), "-": ("4",), "V+": ("5",)}, {"2": "V-", "3": "+", "4": "-", "5": "V+"})
    y1 = _placed("crystal", "Y1", {"1": ("1",), "2": ("2",)}, {"1": "1", "2": "2"})
    q1 = _placed("pfet", "Q1", {"G": ("1",), "S": ("2",), "D": ("3",)}, {"1": "G", "2": "S", "3": "D"})
    d1 = _placed("varactor", "D1", {"K": ("1",), "A": ("2",)}, {"1": "K", "2": "A"})
    q2 = _placed("npn_small", "Q2", {"B": ("1",), "E": ("2",), "C": ("3",)}, {"1": "B", "2": "E", "3": "C"})
    d2 = _placed("diode_sw", "D2", {"K": ("1",), "A": ("2",)}, {"1": "K", "2": "A"})
    rv1 = _placed("pot", "RV1", {"END1": ("1",), "WIPER": ("2",), "END3": ("3",), "MOUNT": ("MP",)}, {"1": "1", "2": "2", "3": "3", "MP": "MountPin"})
    bindings = {
        "U1": models.card_binding(u1, cards["opamp"], text["opamp"], STRUCT, ignored={"V+": models.OPAMP_SUPPLY_IGNORED, "V-": models.OPAMP_SUPPLY_IGNORED}),
        "Y1": models.card_binding(y1, cards["xtal"], text["xtal"], STRUCT),
        "Q1": models.card_binding(q1, cards["pmos"], text["pmos"], STRUCT),
        "D1": models.card_binding(d1, cards["var"], text["var"], STRUCT),
        "Q2": models.card_binding(q2, cards["npn"], text["npn"], STRUCT),
        "D2": models.card_binding(d2, cards["diode"], text["diode"], STRUCT),
        "RV1": models.card_binding(rv1, cards["pot"], text["pot"], STRUCT, ignored={"MOUNT": models.POT_MOUNT_IGNORED}),
    }
    ir = _ir(tmp_path, "cards")
    ir.components = [p.component.model_copy(update={"spice": bindings[p.ref]}) for p in (u1, y1, q1, d1, q2, d2, rv1)] + [_resistor("RL", 1e3)]
    ir.nets = [
        _net("GND", ("U1", "2"), ("U1", "4"), ("Y1", "2"), ("D1", "2"), ("Q2", "2"), ("D2", "1"), ("RV1", "3"), ("RV1", "MP"), ("RL", "2"), kind=NetKind.GROUND),
        _net("VP", ("U1", "5"), ("Q1", "2"), ("RV1", "1")), _net("IN", ("U1", "3"), ("Y1", "1"), ("D1", "1"), ("Q2", "1"), ("D2", "2")),
        _net("OUT", ("U1", "1"), ("Q2", "3")), _net("GATE", ("Q1", "1")), _net("DRAIN", ("Q1", "3"), ("RL", "1")), _net("WIPER", ("RV1", "2")),
    ]
    ir.simulation = SimulationSetup(stimuli=[_dc("VIN", "IN", 0.0), _dc("VP", "VP", 3.3), _dc("VG", "GATE", 0.0)])
    return ir


def test_every_card_is_accepted_by_the_spice_compiler(tmp_path: Path) -> None:
    text = build(_card_board(tmp_path))
    for line in (".subckt OPA1P inp inn out", "Eg g 0 inp inn 100k", ".subckt XTAL21 1 2", ".subckt PMOSG3 d g s", "M1 d g s s PMOSG",
                 ".model PMOSG PMOS (VTO=-1 KP=0.5)", ".model QNPN NPN (TR=200n)", ".model DG D", ".subckt POT_10000_500 a w b",
                 "XU1 IN 0 OUT OPA1P", "XY1 IN 0 XTAL21", "XQ1 DRAIN GATE VP PMOSG3", "D1 0 IN DVAR", "Q2 OUT IN 0 QNPN", "D2 IN 0 DG", "XRV1 VP WIPER 0 POT_10000_500"):
        assert line in text.splitlines(), line


def test_an_unconfirmed_card_never_reaches_the_netlist(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="a model card must have authoritative or user_requirement provenance"):
        build(_card_board(tmp_path, confirmed=False))


def test_card_binding_refuses_another_cards_text() -> None:
    u1 = _placed("opamp", "U1", {"OUT": ("1",), "V-": ("2",), "+": ("3",), "-": ("4",), "V+": ("5",)})
    wrong = models.card_choice("t", models.npn_card(), True)[1]
    with pytest.raises(TemplateRefusal, match="does not match model.opamp"):
        models.card_binding(u1, models.opamp_card(_choice("model.opamp.a0"), _choice("model.opamp.gbw")), wrong, STRUCT)


def test_model_keys_lists_parameters_and_cards() -> None:
    assert models.model_keys(["model.pin.r_on", "rf.if1", "model.l_q.uhf"], [models.npn_card()]) == ["model.l_q.uhf", "model.npn", "model.pin.r_on"]


# --------------------------------------------------------------------------- cards on ngspice


def _single(tmp_path: Path, name: str, comps: list[Component], nets: list[Net], stimuli: list[Stimulus], analysis: AnalysisSpec | None) -> CircuitIR:
    ir = _ir(tmp_path, name)
    ir.components, ir.nets = comps, nets
    ir.simulation = SimulationSetup(stimuli=stimuli, analyses=[analysis] if analysis else [])
    return ir


@needs_ngspice
def test_opamp_macro_has_the_confirmed_gain_and_gbw(tmp_path: Path) -> None:
    card = models.opamp_card(_choice("model.opamp.a0"), _choice("model.opamp.gbw"))
    u1 = _placed("opamp", "U1", {"OUT": ("1",), "V-": ("2",), "+": ("3",), "-": ("4",), "V+": ("5",)})
    b = models.card_binding(u1, card, models.card_choice("t", card, True)[1], STRUCT, ignored={"V+": "supply", "V-": "supply"})
    ir = _single(tmp_path, "opa", [u1.component.model_copy(update={"spice": b}), _resistor("RL", 10e3)],
                 [_net("IN", ("U1", "3")), _net("GND", ("U1", "4"), ("U1", "2"), ("RL", "2"), kind=NetKind.GROUND), _net("OUT", ("U1", "1"), ("RL", "1")),
                  _net("VP", ("U1", "5"))], [_dc("VIN", "IN", 0.0, ac=1.0), _dc("VP", "VP", 3.3)], _ac(1.0, 1e7, 10, "dec"))
    res = _run(ir, tmp_path)
    assert res.value_at("out", 1.0) == pytest.approx(1e5 / math.sqrt(1.01), rel=1e-4)  # A0 / sqrt(1 + (f / f_p)^2), f_p = 10 Hz
    assert res.value_at("out", 10.0) == pytest.approx(1e5 / math.sqrt(2.0), rel=1e-4)  # -3 dB at the pole GBW / A0
    assert res.value_at("out", 1e6) == pytest.approx(1.0, rel=1e-3)  # unity at GBW


@needs_ngspice
def test_crystal_card_resonates_in_series_at_21_4_mhz(tmp_path: Path) -> None:
    lm = models.crystal_lm(user_requirement(21.4e6, "Hz"), _choice("model.xtal21.cm"))
    card = models.crystal_card(lm, _choice("model.xtal21.cm"), _choice("model.xtal21.rm"), _choice("model.xtal21.c0"))
    y1 = _placed("crystal", "Y1", {"1": ("1",), "2": ("2",)}, {"1": "1", "2": "2"})
    b = models.card_binding(y1, card, models.card_choice("t", card, True)[1], STRUCT)
    ir = _single(tmp_path, "xtal", [_resistor("RS", 50.0), y1.component.model_copy(update={"spice": b}), _resistor("RL", 50.0)],
                 [_net("SRC", ("RS", "1")), _net("A", ("RS", "2"), ("Y1", "1")), _net("OUT", ("Y1", "2"), ("RL", "1")),
                  _net("GND", ("RL", "2"), kind=NetKind.GROUND)], [_dc("VS", "SRC", 0.0, ac=1.0)], _ac(21.396e6, 21.404e6, 2001))
    res = _run(ir, tmp_path)

    def expected(f: float) -> float:  # the Butterworth-Van Dyke network between two 50 ohm terminations, by complex arithmetic
        w = 2 * math.pi * f
        zs = 25.0 + 1j * (w * lm.value - 1 / (w * 16e-15))
        zc = 1 / (1j * w * 4e-12)
        return abs(50.0 / (100.0 + zs * zc / (zs + zc)))

    for f in (21.396e6, 21.399e6, 21.4e6, 21.401e6, 21.404e6):
        assert res.value_at("out", f) == pytest.approx(expected(f), rel=1e-6), f
    assert expected(21.4e6) == pytest.approx(0.4, rel=1e-3)  # R_m = 25 ohm at the series resonance f_s = 1 / (2 pi sqrt(L_m C_m))
    f, v = res.vectors["frequency"], res.vectors["out"]
    k = max(range(len(v)), key=lambda i: v[i])
    assert 21.4e6 - 74.0 <= f[k] <= 21.4e6 - 64.0  # C_0 bridging the terminations moves the peak 69.6 Hz below f_s at C_m 16 fF (4 Hz grid)


@needs_ngspice
def test_varactor_card_has_the_junction_law_capacitance(tmp_path: Path) -> None:
    card = models.varactor_card(_choice("model.varactor.cjo"), _choice("model.varactor.vj"), _choice("model.varactor.m"))
    d1 = _placed("varactor", "D1", {"K": ("1",), "A": ("2",)}, {"1": "K", "2": "A"})
    b = models.card_binding(d1, card, models.card_choice("t", card, True)[1], STRUCT)
    r = 10e3
    ir = _single(tmp_path, "var", [_resistor("R1", r), d1.component.model_copy(update={"spice": b})],
                 [_net("SRC", ("R1", "1")), _net("T", ("R1", "2"), ("D1", "1")), _net("GND", ("D1", "2"), kind=NetKind.GROUND)],
                 [_dc("VS", "SRC", 2.0, ac=1.0)], _ac(1.4e6, 1.6e6, 3))
    res = _run(ir, tmp_path)
    f0 = 1.5e6
    phase = res.value_at("t.phase_deg", f0)
    c_measured = math.tan(math.radians(-phase)) / (2 * math.pi * f0 * r)
    assert c_measured == pytest.approx(20e-12 / math.sqrt(1 + 2.0 / 0.7), rel=5e-3)  # 10.18 pF at 2 V reverse (the PM tank's D801)


@needs_ngspice
def test_pmos_wrapper_switches_and_the_pot_divides(tmp_path: Path) -> None:
    card = models.pmos_card()
    q1 = _placed("pfet", "Q1", {"G": ("1",), "S": ("2",), "D": ("3",)}, {"1": "G", "2": "S", "3": "D"})
    b = models.card_binding(q1, card, models.card_choice("t", card, True)[1], STRUCT)
    pot = models.potentiometer_card(10e3, 0.5)
    rv1 = _placed("pot", "RV1", {"END1": ("1",), "WIPER": ("2",), "END3": ("3",), "MOUNT": ("MP",)})
    bp = models.card_binding(rv1, pot, models.card_choice("t", pot, True)[1], STRUCT, ignored={"MOUNT": models.POT_MOUNT_IGNORED})
    for gate, lo, hi in ((0.0, 8.3, 8.4), (8.4, -1e-3, 1e-3)):
        ir = _single(tmp_path / f"g{gate}", "pfet", [q1.component.model_copy(update={"spice": b}), _resistor("RL", 1e3), rv1.component.model_copy(update={"spice": bp})],
                     [_net("SRC", ("Q1", "2"), ("RV1", "1")), _net("GATE", ("Q1", "1")), _net("DRAIN", ("Q1", "3"), ("RL", "1")), _net("W", ("RV1", "2")),
                      _net("GND", ("RL", "2"), ("RV1", "3"), ("RV1", "MP"), kind=NetKind.GROUND)],
                     [_dc("VS", "SRC", 8.4), _dc("VG", "GATE", gate)], None)
        (tmp_path / f"g{gate}").mkdir()
        res = _run(ir, tmp_path / f"g{gate}", SpiceAnalysis.OP)
        assert lo <= res.vectors["drain"][0] <= hi
        assert res.vectors["w"][0] == pytest.approx(4.2, abs=1e-9)


# --------------------------------------------------------------------------- regulatory profile


def test_profile_values_are_unverified_placeholders_never_grounded() -> None:
    for confirmed in (False, True):
        rows = profile.profile_choices("t_rf", confirmed)
        assert [c.key for c, _ in rows] == profile.profile_keys()
        for ch, traced in rows:
            assert ch.key.startswith(profile.PROFILE_PREFIX)
            assert "UNVERIFIED" in ch.text() and profile.UNVERIFIED_TAIL in ch.text(), ch.key
            assert traced.provenance.kind is (ProvenanceKind.USER_REQUIREMENT if confirmed else ProvenanceKind.ASSUMPTION)
            assert traced.provenance.kind is not ProvenanceKind.AUTHORITATIVE and traced.provenance.source is None
    values = {c.key: t.value for c, t in profile.profile_choices("t_rf", True)}
    assert values["kr447.max_power"] == 0.5 and values["kr447.max_deviation"] == 2.5e3 and values["kr447.max_obw"] == 8.5e3
    assert values["kr447.freq_tolerance"] == 2.5 and values["kr447.spurious_max"] == -36.0 and values["kr447.rx_spurious_max"] == -57.0
    assert values["kr447.emission"] == "F3E" and values["kr447.tx_timeout"] == "not known"


def test_profile_keys_end_with_the_suffixes_the_regulatory_check_matches() -> None:
    suffixes = {k.split(".", 1)[1] for k in profile.profile_keys()}
    assert {"max_power", "max_deviation", "max_obw", "freq_tolerance", "band_low", "band_high", "channel_raster"} <= suffixes


def test_the_raster_has_25_channels_and_refuses_a_band_as_a_carrier() -> None:
    lo, hi, step = (profile.PROFILE_BY_KEY[k].value for k in ("kr447.band_low", "kr447.band_high", "kr447.channel_raster"))
    plan = profile.channel_plan(lo, hi, step)
    assert len(plan) == 25 and plan[0] == 447.5625e6 and plan[-1] == pytest.approx(447.8625e6)
    from ai_eda.tools.calc.quantity import parse_answer

    for text, index in (("447.5625 MHz", 0), ("447.575 MHz", 1), ("447.8625 MHz", 24)):
        assert profile.channel_index(parse_answer(text).value, lo, hi, step) == index
        assert profile.raster_refusal(parse_answer(text).value, lo, hi, step) is None
    why = profile.raster_refusal(447e6, lo, hi, step)
    assert why is not None and "447 MHz 대역" in why and "447.5625" in why and "447.8625" in why and "25 channels" in why and "UNVERIFIED" in why
    assert profile.channel_index(447.57e6, lo, hi, step) is None and profile.channel_index(447.875e6, lo, hi, step) is None
    with pytest.raises(ValueError, match="not a whole number"):
        profile.channel_plan(447.5625e6, 447.87e6, 12.5e3)


# --------------------------------------------------------------------------- family


#: the kr447 design §2.0 table: build -> (needs, served incl. optional reads)
DESIGN_TABLE = {
    "audio_ptt": ({"input_voltage", "modulation"}, {"modulation", "frequency_deviation", "audio_bandwidth", "input_voltage", "tx_timeout"}),
    "rx_backend": ({"modulation", "input_voltage"}, {"modulation", "frequency_deviation", "audio_bandwidth", "channel_spacing", "frequency_tolerance",
                                                     "system_impedance", "input_voltage"}),
    "rx_frontend": ({"carrier_frequency", "input_voltage"}, {"carrier_frequency", "modulation", "frequency_tolerance", "system_impedance", "rx_sensitivity",
                                                             "input_voltage"}),
    "tx_exciter": ({"carrier_frequency", "modulation", "input_voltage"}, {"carrier_frequency", "modulation", "frequency_deviation", "audio_bandwidth",
                                                                           "occupied_bandwidth", "frequency_tolerance", "tx_power", "system_impedance",
                                                                           "input_voltage", "tx_timeout"}),
}
_ALL = {"carrier_frequency", "modulation", "frequency_deviation", "audio_bandwidth", "channel_spacing", "occupied_bandwidth", "frequency_tolerance", "tx_power",
        "erp", "eirp", "system_impedance", "antenna_impedance", "antenna_gain", "link_range", "field_strength_limit", "rx_sensitivity", "input_voltage", "tx_timeout"}
DESIGN_TABLE["transceiver"] = DESIGN_TABLE["transceiver_conducted"] = ({"carrier_frequency", "modulation", "input_voltage"}, _ALL)
for _needs, _serves in DESIGN_TABLE.values():
    _serves.add("radio_build")  # every build serves the requirement that selected it (§2.0: every kr447 template serves radio_build)


def test_the_builds_follow_the_design_table() -> None:
    assert list(family.BUILDS) == ["audio_ptt", "rx_backend", "rx_frontend", "tx_exciter", "transceiver", "transceiver_conducted"]
    for build, (needs, serves) in DESIGN_TABLE.items():
        b = family.BUILDS[build]
        assert set(b.needs) == needs and set(b.serves) == serves, build
    assert family.BUILDS["audio_ptt"].layers == ((2, 4), 4)
    assert all(b.layers == ((4,), 4) for k, b in family.BUILDS.items() if k != "audio_ptt")  # decision 7A: the RF lines need a plane
    assert family.BUILDS["transceiver_conducted"].template_id == family.BUILDS["transceiver"].template_id == "kr447_transceiver"
    assert [b.stage for b in family.BUILDS.values()] == [1, 2, 3, 4, 5, 5]


def test_closed_world_messages_name_the_serving_builds() -> None:
    assert family.serving_builds("tx_power") == ("tx_exciter", "transceiver", "transceiver_conducted")
    assert family.unserved_message("tx_power", "rx_backend") == (
        "tx_power is not served by radio_build=rx_backend; tx_power is served by radio_build=tx_exciter, transceiver or transceiver_conducted")
    assert family.unserved_message("channel_spacing") == "channel_spacing is served by radio_build=rx_backend, transceiver or transceiver_conducted"
    assert "served by no radio_build" in family.unserved_message("modulation_depth")


def test_the_selector_question_is_required_and_recommends_rx_backend(tmp_path: Path) -> None:
    ir = _ir(tmp_path, "sel")
    ir.requirements.requirements.append(Requirement(id="req.f", key="carrier_frequency", category="rf", text="447.5625 MHz", kind=RequirementKind.EXPLICIT,
                                                    value=Traced(value="447.5625 MHz", provenance=USER)))
    q = family.selector_question(ir)
    assert q.key == family.SELECTOR_KEY == "radio_build" and q.required and q.options == list(family.BUILDS)
    assert "rx_backend (stage 2, recommended first" in q.question and "UNVERIFIED" in q.question and "KC conformity" in q.question
    assert q.rationale.startswith("carrier_frequency is confirmed but radio_build is not")
    assert family.selector_question(_ir(tmp_path, "empty")).rationale.startswith("radio_build is not stated")


def test_the_build_names_equal_the_requirement_key_constants() -> None:
    assert tuple(family.BUILDS) == design_inputs.RADIO_BUILDS and family.SELECTOR_KEY == design_inputs.RADIO_BUILD_KEY == "radio_build"


def test_the_registry_holds_the_five_templates_in_staging_order() -> None:
    """Filled at the wave-2 merge in the staging order, the transceiver last (tests/test_kr447_registry.py checks the selection end to end)."""
    assert isinstance(RF_TEMPLATES, list) and [t.id for t in RF_TEMPLATES] == [
        "kr447_audio_ptt", "kr447_rx_backend", "kr447_rx_frontend", "kr447_tx_exciter", "kr447_transceiver"]


# --------------------------------------------------------------------------- block API


def test_prefix_rebases_references_and_prefixes_internal_nets() -> None:
    p = BlockPrefix(300, "TXA_")
    assert (p.ref("R6"), p.ref("C_T1"), p.ref("SH1"), p.ref("U99")) == ("R306", "C_T301", "SH301", "U399")
    for bad in ("R100", "ANT", "R0", "1R"):
        with pytest.raises(TemplateRefusal, match="cannot be re-based"):
            p.ref(bad)
    assert p.net("LIM_OUT") == "TXA_LIM_OUT" and p.net("PM_DRIVE", {"PM_DRIVE"}) == "PM_DRIVE" and p.net("GND") == "GND"
    assert BlockPrefix().ref("R6") == "R6" and BlockPrefix().net("X") == "X"
    for kw in ({"ref_base": 150}, {"ref_base": -100}, {"net_prefix": "1X"}, {"net_prefix": "A B"}):
        with pytest.raises(ValueError):
            BlockPrefix(**kw)


def _small_result() -> BlockResult:
    r1, c1 = _resistor("R1", 1e3), _capacitor("C1", 1e-9)
    placed = PlacedPart(part=PARTS["res_0603"], component=r1, functions={"1": ("1",), "2": ("2",)})
    return BlockResult(
        block_id="demo", title="demo block", components=[r1, c1],
        nets=[_net("PM_DRIVE", ("R1", "1")), _net("MID", ("R1", "2"), ("C1", "1")), _net("GND", ("C1", "2"), kind=NetKind.GROUND)],
        chain=["R1", "C1"], shield_ref="C1", placed={"R1": placed}, interface_nets=frozenset({"PM_DRIVE"}),
        stimuli=[_dc("VDRV", "PM_DRIVE", 0.0, ac=1.0)],
        expectations=[Expectation(id="mid_1k", analysis_id="ac", vector="v(MID)", reduce=Reduce.DB_AT, at=user_requirement(1e3, "Hz"),
                                  nominal=user_requirement(0.0, "dB"), tol_abs=user_requirement(0.5, "dB"), reference_vector="v(PM_DRIVE)", provenance=STRUCT),
                      Expectation(id="i_r1", analysis_id="op", vector="i(R1)", nominal=user_requirement(0.0, "A"), tol_abs=user_requirement(1e-9, "A"), provenance=STRUCT),
                      Expectation(id="i_src", analysis_id="op", vector="i(VDRV)", nominal=user_requirement(0.0, "A"), tol_abs=user_requirement(1e-9, "A"), provenance=STRUCT)],
        net_classes={"AUDIO": ["MID", "PM_DRIVE"]}, open_pins={}, params={"demo.r": user_requirement(1e3, "ohm")},
    )


def test_apply_prefix_renames_every_part_and_internal_net_and_keeps_ids() -> None:
    before = _small_result()
    snapshot = [c.ref for c in before.components]
    out = apply_prefix(before, BlockPrefix(300, "TXA_"))
    assert [c.ref for c in out.components] == ["R301", "C301"] and [c.ref for c in before.components] == snapshot  # the input is untouched
    assert {n.name: [(p.component_ref, p.pin_number) for p in n.pins] for n in out.nets} == {
        "PM_DRIVE": [("R301", "1")], "TXA_MID": [("R301", "2"), ("C301", "1")], "GND": [("C301", "2")]}
    assert out.chain == ["R301", "C301"] and out.shield_ref == "C301" and list(out.placed) == ["R301"] and out.placed["R301"].component.ref == "R301"
    assert [(e.id, e.vector, e.reference_vector) for e in out.expectations] == [
        ("mid_1k", "v(TXA_MID)", "v(PM_DRIVE)"), ("i_r1", "i(R301)", None), ("i_src", "i(VDRV)", None)]  # ids are check ids: unchanged
    assert out.stimuli[0].net == "PM_DRIVE" and out.net_classes == {"AUDIO": ["TXA_MID", "PM_DRIVE"]} and out.params == before.params
    assert apply_prefix(before, BlockPrefix()) is before
    assert rename_vector("vp( MID )", lambda n: "X_" + n, {}) == "vp( X_MID )" and rename_vector("not a vector", str, {}) == "not a vector"


class _Port(BaseModel):
    name: str
    net: str
    reference_net: str = "GND"


class _State(BaseModel):
    id: str
    port_dc_v: dict[str, Traced] = {}
    bindings: dict[str, SpiceBinding] = {}


class _Network(BaseModel):
    id: str
    block: str
    members: list[str]
    bindings: dict[str, SpiceBinding] = {}
    loss_q: dict[str, Traced] = {}
    ports: list[_Port] = []
    states: list[_State] = []


class _Rail(BaseModel):
    rail: str
    regulator_ref: str


def test_apply_prefix_renames_fixture_objects_by_their_contract_field_names() -> None:
    r = _small_result()
    binding = SpiceBinding(device=SpiceDevice.C, value=user_requirement(1e-12, "F"), provenance=STRUCT)
    r.ports = [_Port(name="DRV", net="PM_DRIVE"), _Port(name="TAP", net="MID")]
    r.networks = [_Network(id="tank", block="demo", members=["R1", "C1"], bindings={"C1": binding}, loss_q={"R1": user_requirement(40.0)},
                           ports=[_Port(name="IN", net="PM_DRIVE"), _Port(name="OUT", net="MID")],
                           states=[_State(id="bias_lo", port_dc_v={"IN": user_requirement(1.44, "V")}, bindings={"C1": binding})])]
    r.rails = [_Rail(rail="MID", regulator_ref="R1")]
    assert r.check() == []
    out = apply_prefix(r, BlockPrefix(800, "TX_"))
    nw = out.networks[0]
    assert nw.id == "tank" and nw.members == ["R801", "C801"] and list(nw.bindings) == ["C801"] and list(nw.loss_q) == ["R801"]
    assert [(p.name, p.net) for p in nw.ports] == [("IN", "PM_DRIVE"), ("OUT", "TX_MID")] and list(nw.states[0].bindings) == ["C801"]
    assert nw.states[0].port_dc_v == r.networks[0].states[0].port_dc_v  # port names are the fixture's, never renamed
    assert [(p.name, p.net) for p in out.ports] == [("DRV", "PM_DRIVE"), ("TAP", "TX_MID")]
    assert (out.rails[0].rail, out.rails[0].regulator_ref) == ("TX_MID", "R801")


def test_the_real_fixture_types_carry_the_contract_field_names() -> None:
    rf = pytest.importorskip("ai_eda.ir.rf", reason="ai_eda.ir.rf (kr447 part P1) is not merged yet")
    need = {"RFPort": {"name", "net", "reference_net"}, "RFState": {"id", "port_dc_v", "bindings"},
            "RFNetwork": {"id", "block", "members", "bindings", "loss_q", "ports", "states"}, "RailBudget": {"rail", "regulator_ref"}}
    for cls, fields in need.items():
        assert fields <= set(getattr(rf, cls).model_fields), cls
    from ai_eda.ir.simulation import AnalysisSpec
    from ai_eda.tools.spice.runner import SpiceAnalysis

    binding = SpiceBinding(device=SpiceDevice.C, value=user_requirement(1e-12, "F"), provenance=STRUCT)
    sweep = AnalysisSpec(id="ac1", kind=SpiceAnalysis.AC, params={"variation": user_requirement("dec"), "points": user_requirement(20),
                                                                   "fstart": user_requirement(1e3, "Hz"), "fstop": user_requirement(1e6, "Hz")}, provenance=STRUCT)
    nw = rf.RFNetwork(id="tank", block="demo", members=["R1", "C1"], bindings={"C1": binding}, loss_q={"R1": user_requirement(40.0)},
                      q_ref_hz=user_requirement(1e5, "Hz"),
                      ports=[rf.RFPort(name="IN", net="PM_DRIVE", kind="port", z0_ohm=user_requirement(50.0, "ohm")),
                             rf.RFPort(name="OUT", net="MID", kind="probe"), rf.RFPort(name="VB", net="VB", kind="control")],
                      states=[rf.RFState(id="bias_lo", port_dc_v={"VB": user_requirement(1.44, "V")}, bindings={"C1": binding})], sweep=[sweep],
                      expectations=[rf.RFExpectation(id="ph", state="bias_lo", quantity="phase21_deg", drive="IN", to="OUT",
                                                     at=user_requirement(1e5, "Hz"), nominal=user_requirement(-20.0, "deg"), tol_abs=user_requirement(1.0, "deg"))])
    r = _small_result()
    r.networks = [nw]
    r.rails = [rf.RailBudget(rail="MID", regulator_ref="R1", v_out=user_requirement(5.0, "V"), i_min=user_requirement(0.0, "A"), i_max=user_requirement(0.1, "A"))]
    out = apply_prefix(r, BlockPrefix(100, "P_"))
    got = out.networks[0]
    assert got.members == ["R101", "C101"] and [p.net for p in got.ports] == ["PM_DRIVE", "P_MID", "P_VB"] and list(got.loss_q) == ["R101"]
    assert list(got.bindings) == ["C101"] and list(got.states[0].bindings) == ["C101"] and got.expectations[0].id == "ph"
    assert (out.rails[0].rail, out.rails[0].regulator_ref) == ("P_MID", "R101")
    # the renamed objects are still valid RF IR: a design built from their dump validates
    again = rf.RFDesign.model_validate({"blocks": [{"id": "demo"}], "networks": [got.model_dump()], "rails": [out.rails[0].model_dump()]})
    assert again.network("tank").port("OUT").net == "P_MID"


def test_block_check_names_every_structural_problem() -> None:
    r = _small_result()
    assert r.check() == []
    r.placed["R1"] = PlacedPart(part=PARTS["res_0603"], component=r.components[0], functions={"1": ("1",), "2": ("2",)}, stacks=(("1", "2"),))
    r.nets.append(_net("ELSEWHERE", ("C1", "1"), ("Q9", "1")))
    r.components.append(_capacitor("C2", 1e-9))
    r.stimuli.append(_dc("VX", "NOWHERE", 1.0))
    r.expectations.append(Expectation(id="bad", analysis_id="op", vector="v(NOWHERE)", nominal=user_requirement(1.0, "V"), tol_abs=user_requirement(0.1, "V"), provenance=STRUCT))
    r.chain.append("U9")
    r.open_pins[("R1", "1")] = "claimed open"
    text = " | ".join(r.check())
    for phrase in ("stacked in the library", "unknown part 'Q9'", "C1.1 is in nets 'MID' and 'ELSEWHERE'", "C2.1 (unnamed) is in no net",
                   "stimulus VX names net 'NOWHERE'", "reads 'v(NOWHERE)'", "chain / shield names unknown part 'U9'",
                   "R1.1 is marked open (claimed open) but sits in net 'PM_DRIVE'"):
        assert phrase in text, phrase


class _Demo(Block):
    id = "demo"
    title = "demo block"
    interface_nets = ("PM_DRIVE",)

    def __init__(self, broken: bool = False) -> None:
        self.broken = broken

    def build_local(self, ctx: BlockContext) -> BlockResult:
        r = _small_result()
        if self.broken:
            r.components.append(_capacitor("C9", 1e-9))
        return r


def test_block_build_checks_then_prefixes(tmp_path: Path) -> None:
    ctx = BlockContext(ir=_ir(tmp_path, "b"), library=KicadLibrary(roots=[]), template_id="t_rf", confirmed=False)
    out = _Demo().build(ctx, BlockPrefix(400, "RXA_"))
    assert [c.ref for c in out.components] == ["R401", "C401"] and "PM_DRIVE" in out.interface_nets
    with pytest.raises(TemplateRefusal, match=r"block demo: C9\.1 \(unnamed\) is in no net"):
        _Demo(broken=True).build(ctx)


def test_merge_joins_interface_nets_and_refuses_what_would_collide() -> None:
    a = apply_prefix(_small_result(), BlockPrefix(100, "A_"))
    b = apply_prefix(_small_result(), BlockPrefix(200, "B_"))
    b.expectations = [e.model_copy(update={"id": e.id + "_b"}) for e in b.expectations]
    merged = merge_results([a, b])
    by_name = {n.name: [(p.component_ref, p.pin_number) for p in n.pins] for n in merged.nets}
    assert by_name["PM_DRIVE"] == [("R101", "1"), ("R201", "1")] and by_name["GND"] == [("C101", "2"), ("C201", "2")]
    assert {"A_MID", "B_MID"} <= set(by_name) and len(merged.stimuli) == 1  # the identical VDRV stimulus is one source
    with pytest.raises(TemplateRefusal, match="reference 'R101' is used by blocks"):
        merge_results([a, a])
    same_names = apply_prefix(_small_result(), BlockPrefix(200))  # internal MID not prefixed: it would silently join A's MID
    a_plain = apply_prefix(_small_result(), BlockPrefix(100))
    with pytest.raises(TemplateRefusal, match="net 'MID' of block 'demo' is internal"):
        merge_results([a_plain, same_names])
    b2 = apply_prefix(_small_result(), BlockPrefix(200, "B_"))
    with pytest.raises(TemplateRefusal, match="expectation id 'mid_1k' is used by two blocks"):
        merge_results([a, b2])
    b.params = {"demo.r": user_requirement(2e3, "ohm")}
    with pytest.raises(TemplateRefusal, match="parameter 'demo.r' has different values"):
        merge_results([a, b])


def test_a_second_different_card_under_one_key_is_refused_never_hidden(tmp_path: Path) -> None:
    """Two potentiometers of different resistance (the volume and the squelch pot) under the default key: the second card's text would
    become the user's on confirm_design=yes without its row in the table - refused, in one block and across blocks; distinct keys show both."""
    ctx = BlockContext(ir=_ir(tmp_path, "cards"), library=KicadLibrary(roots=[]), template_id="t_rf", confirmed=False)
    vol, sql = models.potentiometer_card(10e3, 0.5), models.potentiometer_card(50e3, 0.5)
    assert vol.key == sql.key == "model.pot" and vol.text != sql.text
    b = BlockBuilder(ctx, "rx_audio")
    b.card(vol)
    assert b.card(vol).value == vol.text and [c.key for c in b.result.choices] == ["model.pot"]  # the same card twice: one row
    with pytest.raises(TemplateRefusal, match=r"the table row 'model.pot' already shows another card .*POT_50000_500 needs its own key"):
        b.card(sql)
    # with a key per card, both texts reach the table (and model_keys)
    b = BlockBuilder(ctx, "rx_audio")
    b.card(models.potentiometer_card(10e3, 0.5, key="model.pot.vol"))
    b.card(models.potentiometer_card(50e3, 0.5, key="model.pot.sql"))
    rows = {c.key: c.description for c in b.result.choices}
    assert set(rows) == {"model.pot.vol", "model.pot.sql"} and "Ra a w 25k" in rows["model.pot.sql"] and b.result.model_keys == ["model.pot.vol", "model.pot.sql"]
    # across blocks: the merge keeps an identical row once and refuses a different one under the same key
    one, two = BlockBuilder(ctx, "tx_audio"), BlockBuilder(ctx, "rx_audio")
    one.card(vol)
    two.card(vol)
    assert [c.key for c in merge_results([one.result, two.result]).choices] == ["model.pot"]
    three = BlockBuilder(ctx, "rx_audio")
    three.card(sql)
    with pytest.raises(TemplateRefusal, match=r"table row 'model.pot' differs between blocks"):
        merge_results([one.result, three.result])


def test_constraint_targets_follow_the_renaming_and_are_checked() -> None:
    def constraint(cid: str, target: str) -> Constraint:
        return Constraint(id=cid, kind=ConstraintKind.ELECTRICAL, description=f"{target} limit (free text, not renamed)", target=target, provenance=STRUCT)

    r = _small_result()
    r.constraints = [constraint("c_part", "R1"), constraint("c_net", "MID"), constraint("c_if", "PM_DRIVE"), constraint("c_gnd", "GND"),
                     constraint("c_all", "*"), constraint("c_block", "demo")]
    assert r.check() == []
    out = apply_prefix(r, BlockPrefix(100, "IF_"))
    assert [c.target for c in out.constraints] == ["R101", "IF_MID", "PM_DRIVE", "GND", "*", "demo"]
    assert out.constraints[0].description == "R1 limit (free text, not renamed)" and [c.target for c in r.constraints][0] == "R1"
    r.constraints.append(constraint("c_bad", "NOPE"))
    r.constraints.append(constraint("c_part", "C1"))
    text = " | ".join(r.check())
    assert "constraint c_bad targets 'NOPE', which is neither '*', the block 'demo', a part nor a net of it" in text
    assert "constraint id 'c_part' is used twice" in text
    # two instances of one block would repeat the constraint ids: refused like expectation ids
    a, b = apply_prefix(_small_result(), BlockPrefix(100, "A_")), apply_prefix(_small_result(), BlockPrefix(200, "B_"))
    b.expectations = [e.model_copy(update={"id": e.id + "_b"}) for e in b.expectations]
    a.constraints, b.constraints = [constraint("c_all", "*")], [constraint("c_all", "*")]
    with pytest.raises(TemplateRefusal, match="constraint id 'c_all' is used by two blocks"):
        merge_results([a, b])


def _floating_board() -> tuple[list[Component], list[Net], list[Stimulus]]:
    """R1 / C1 driven by VIN (kept); U1 excluded; C2 on U1's node alone; R5 -> C5 hanging off U1 (two rounds); Q1's base touched by nothing else."""
    u1 = Component(ref="U1", value="U1", pins=_pins(("1", ""), ("2", "")), provenance=STRUCT,
                   spice=SpiceBinding(exclude=True, exclude_reason="IC, no model", provenance=STRUCT))
    q1 = Component(ref="Q1", value="Q1", pins=_pins(("1", "B"), ("2", "E"), ("3", "C")), provenance=STRUCT,
                   spice=SpiceBinding(device=SpiceDevice.Q, model_name="QNPN", model_card=Traced(value=".model QNPN NPN (TR=200n)", provenance=USER),
                                      pin_order=["3", "1", "2"], provenance=STRUCT))
    comps = [_resistor("R1", 1e3), _capacitor("C1", 1e-9), u1, _capacitor("C2", 1e-9), _resistor("R5", 1e3), _capacitor("C5", 1e-9), q1, _resistor("R6", 1e3)]
    nets = [_net("VIN", ("R1", "1"), ("R6", "1")), _net("A", ("R1", "2"), ("C1", "1")), _net("X", ("U1", "1"), ("C2", "1")), _net("P", ("U1", "2"), ("R5", "1")),
            _net("Q", ("R5", "2"), ("C5", "1")), _net("BASE", ("Q1", "1")), _net("COL", ("Q1", "3"), ("R6", "2")),
            _net("GND", ("C1", "2"), ("C2", "2"), ("C5", "2"), ("Q1", "2"), kind=NetKind.GROUND)]
    return comps, nets, [_dc("VIN", "VIN", 5.0)]


def test_exclude_floating_removes_dangling_two_terminal_parts_until_nothing_changes() -> None:
    comps, nets, stimuli = _floating_board()
    out, report = exclude_floating(comps, nets, stimuli, "t_rf")
    assert [(f.ref, f.net) for f in report.excluded] == [("C2", "X"), ("C5", "Q"), ("R5", "P")]
    assert "reaches only parts outside the netlist (U1)" in report.excluded[0].reason and "singular matrix" in report.excluded[0].reason
    assert [(f.ref, f.net) for f in report.unresolved] == [("Q1", "BASE")]  # never excluded silently
    kept = {c.ref: c.spice for c in out}
    assert kept["C2"].exclude and kept["C5"].exclude and kept["R5"].exclude and not kept["C1"].exclude and not kept["Q1"].exclude
    assert kept["C2"].exclude_reason == report.excluded[0].reason and comps[3].spice.exclude is False  # the input list is not changed


def test_exclude_floating_finds_the_atmega_templates_hand_made_list(tmp_path: Path) -> None:
    from tests.test_atmega128_template import _built

    ir, *_ = _built(tmp_path)
    hand_made = {"C1", "C7", "C9", "C10"}  # the atmega128_devboard template's own floating-node exclusions
    assert all(ir.component(r).spice.exclude for r in hand_made)
    reincluded = [c.model_copy(update={"spice": SpiceBinding(device=SpiceDevice.C, value=c.electrical["capacitance"], provenance=STRUCT)}) if c.ref in hand_made else c
                  for c in ir.components]
    out, report = exclude_floating(reincluded, ir.nets, ir.simulation.stimuli, "t_rf")
    # the same four, plus the ~PEN pull-up R3, whose node PEN reaches only U1: the template keeps it (harmless - it carries no current),
    # the generalised rule removes every dead branch
    assert {f.ref for f in report.excluded} == hand_made | {"R3"} and report.unresolved == []
    assert {f.net for f in report.excluded} == {"VIN", "AREF", "XTAL1", "XTAL2", "PEN"}


@needs_ngspice
def test_a_deck_without_its_floating_capacitors_solves(tmp_path: Path) -> None:
    comps, nets, stimuli = _floating_board()
    comps = [c for c in comps if c.ref != "Q1"] + [_resistor("RB", 1e6)]
    nets = [n if n.name != "BASE" else _net("BASE", ("RB", "1")) for n in nets if n.name != "COL"] + [_net("COL", ("R6", "2"), ("RB", "2"))]
    nets = [n if n.name != "GND" else _net("GND", ("C1", "2"), ("C2", "2"), ("C5", "2"), kind=NetKind.GROUND) for n in nets]
    out, report = exclude_floating(comps, nets, stimuli, "t_rf")
    assert {f.ref for f in report.excluded} >= {"C2", "C5", "R5"}
    ir = _single(tmp_path, "floating", out, nets, stimuli, None)
    res = _run(ir, tmp_path, SpiceAnalysis.OP)
    assert res.vectors["a"][0] == pytest.approx(5.0)


# --------------------------------------------------------------------------- name-based lookup on synthetic libraries


def _effects():
    return SX("effects", SX("font", SX("size", 1.27, 1.27)))


def _prop(key: str, value: str):
    return SX("property", Q(key), Q(value), SX("at", 0, 0, 0), _effects())


def _spin(number: str, name: str, etype: str, x: float, y: float):
    return SX("pin", etype, "line", SX("at", x, y, 0), SX("length", 2.54), SX("name", Q(name), _effects()), SX("number", Q(number), _effects()))


def _write_lib(root: Path, lib: str, name: str, pins: list[tuple[str, str, str, float, float]], description: str, fp: tuple[str, str], pads: list[str]) -> KicadLibrary:
    body = SX("symbol", Q(f"{name}_1_1"), *[_spin(*p) for p in pins])
    sym = SX("symbol", Q(name), SX("exclude_from_sim", False), SX("in_bom", True), SX("on_board", True), _prop("Reference", "U"), _prop("Value", name),
             _prop("Footprint", f"{fp[0]}:{fp[1]}"), _prop("Datasheet", "~"), _prop("Description", description), body, SX("embedded_fonts", False))
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    (root / "symbols" / f"{lib}.kicad_sym").write_text(sexpr.dumps(SX("kicad_symbol_lib", SX("version", 20251024), SX("generator", Q("kicad_symbol_editor")), sym)), encoding="utf-8")
    pretty = root / "footprints" / f"{fp[0]}.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    pad_nodes = [SX("pad", Q(n), "smd", "rect", SX("at", 1.0 * i, 0), SX("size", 0.6, 0.6), SX("layers", Q("F.Cu"), Q("F.Mask"))) for i, n in enumerate(pads)]
    sexpr.dump_file(SX("footprint", Q(fp[1]), SX("version", 20260206), SX("layer", Q("F.Cu")), SX("attr", "smd"), *pad_nodes, SX("embedded_fonts", False)), pretty / f"{fp[1]}.kicad_mod")
    return KicadLibrary(roots=[root])


_SOT23_5 = ("Package_TO_SOT_SMD", "SOT-23-5")
_LMV331 = [("1", "+", "input", -7.62, 2.54), ("2", "V-", "power_in", -2.54, -7.62), ("3", "-", "input", -7.62, -2.54), ("4", "", "open_collector", 7.62, 0),
           ("5", "V+", "power_in", -2.54, 7.62)]


def _inst(lib: KicadLibrary, key: str) -> PlacedPart:
    return instantiate(lib, key, "U1", "v", "d", STRUCT)


def test_pins_are_found_by_name_and_an_unnamed_output_by_its_unique_type(tmp_path: Path) -> None:
    placed = _inst(_write_lib(tmp_path, "Comparator", "LMV331", _LMV331, "comparator", _SOT23_5, ["1", "2", "3", "4", "5"]), "comparator")
    assert placed.functions == {"+": ("1",), "V-": ("2",), "-": ("3",), "OUT": ("4",), "V+": ("5",)} and placed.pin("OUT") == "4"
    assert placed.notes == ("U1 pin 4 has no name in the library; identified as OUT by its library electrical type 'open_collector' (the only unnamed pin of that type)",)
    assert placed.at("V+") == [("U1", "5")]


def test_a_missing_pin_name_refuses_the_part(tmp_path: Path) -> None:
    pins = [p if p[0] != "5" else ("5", "VCC", "power_in", -2.54, 7.62) for p in _LMV331]
    lib = _write_lib(tmp_path, "Comparator", "LMV331", pins, "comparator", _SOT23_5, ["1", "2", "3", "4", "5"])
    with pytest.raises(TemplateRefusal, match=r"the library has no pin named 'V\+', the template expects 1; .* will not guess the pinout"):
        _inst(lib, "comparator")


def test_two_unnamed_outputs_refuse_the_type_rule(tmp_path: Path) -> None:
    pins = [("1", "", "output", 7.62, 0), ("2", "V-", "power_in", -2.54, -7.62), ("3", "+", "input", -7.62, 2.54), ("4", "", "output", -7.62, -2.54),
            ("5", "V+", "power_in", -2.54, 7.62)]
    lib = _write_lib(tmp_path, "Amplifier_Operational", "MCP6001-OT", pins, "1MHz, Low-Power Op Amp, SOT-23-5", _SOT23_5, ["1", "2", "3", "4", "5"])
    with pytest.raises(TemplateRefusal, match="2 unnamed pin"):
        _inst(lib, "opamp")


def test_a_lost_description_fact_refuses_the_part(tmp_path: Path) -> None:
    lib = _write_lib(tmp_path, "Transistor_FET", "AO3401A", [("1", "G", "input", -5.08, 0), ("2", "S", "passive", 2.54, -5.08), ("3", "D", "passive", 2.54, 5.08)],
                     "P-Channel MOSFET, SOT-23", ("Package_TO_SOT_SMD", "SOT-23"), ["1", "2", "3"])
    with pytest.raises(TemplateRefusal, match="no longer says '-4.0A Id'"):
        _inst(lib, "pfet")


def test_a_changed_stack_refuses_the_part(tmp_path: Path) -> None:
    unstacked = [("1", "GND", "power_in", -5.08, -10.16), ("2", "IF", "output", 10.16, 0), ("3", "RF", "input", -10.16, 0), ("4", "GND", "passive", 0, -10.16),
                 ("5", "GND", "passive", 5.08, -10.16), ("6", "LO", "input", 0, 10.16)]
    lib = _write_lib(tmp_path, "RF_Mixer", "ADEX-10", unstacked, "Mixer, +7 dBm LO, 10 to 1000 MHz, CD542", ("RF_Mini-Circuits", "Mini-Circuits_CD542_LandPatternPL-052"),
                     ["1", "2", "3", "4", "5", "6"])
    with pytest.raises(TemplateRefusal, match=r"the library stacks the pins of \[\], the parts table records \['GND'\]"):
        _inst(lib, "mixer")
    stacked = [p if p[1] != "GND" else (p[0], "GND", p[2], -5.08, -10.16) for p in unstacked]
    lib = _write_lib(tmp_path / "ok", "RF_Mixer", "ADEX-10", stacked, "Mixer, +7 dBm LO, 10 to 1000 MHz, CD542", ("RF_Mini-Circuits", "Mini-Circuits_CD542_LandPatternPL-052"),
                     ["1", "2", "3", "4", "5", "6"])
    placed = _inst(lib, "mixer")
    assert placed.stacks == (("1", "4", "5"),) and placed.at("GND") == [("U1", "1"), ("U1", "4"), ("U1", "5")]


def test_a_pad_without_a_pin_refuses_like_the_pcb_compiler(tmp_path: Path) -> None:
    lib = _write_lib(tmp_path, "Comparator", "LMV331", _LMV331, "comparator", _SOT23_5, ["1", "2", "3", "4", "5", "6"])
    with pytest.raises(TemplateRefusal, match=r"has pads \['6'\] that are not pins"):
        _inst(lib, "comparator")


def test_an_unknown_part_key_is_refused() -> None:
    with pytest.raises(TemplateRefusal, match="no part 'mcp6002'"):
        instantiate(KicadLibrary(roots=[]), "mcp6002", "U1", "v", "d", STRUCT)
    assert not any(p.symbol == ("Amplifier_Operational", "MCP6002-xSN") for p in PARTS.values())  # the dual op-amp is refused, never a row


def test_builder_builds_a_block_from_the_parts_table(tmp_path: Path) -> None:
    lib = _write_lib(tmp_path, "Comparator", "LMV331", _LMV331, "comparator", _SOT23_5, ["1", "2", "3", "4", "5"])
    ctx = BlockContext(ir=_ir(tmp_path, "blk"), library=lib, template_id="t_rf", confirmed=True)
    b = BlockBuilder(ctx, "sq", "squelch comparator", interface_nets=("RSSI", "MUTE", "RX_3V3"))
    u = b.part("comparator", "U3", "LMV331", "squelch comparator")
    b.net("RSSI", NetKind.ANALOG, u.at("+"))
    b.net("SQ_REF", NetKind.ANALOG, u.at("-"))
    b.net("MUTE", NetKind.SIGNAL, u.at("OUT"))
    b.net("RX_3V3", NetKind.POWER, u.at("V+"))
    q = b.model("model.l_q.if2")
    b.choice("sq.hyst", 0.05, "V", "squelch hysteresis")
    with pytest.raises(TemplateRefusal, match=r"U3\.2 \(V-\) is in no net"):
        b.done()
    b.leave_open("U3", "V-", "test: left open on purpose")
    result = b.done()
    assert result.open_pins == {("U3", "2"): "test: left open on purpose"} and result.placed["U3"].component.pin("2").electrical_type is PinElectricalType.NO_CONNECT
    assert result.model_keys == ["model.l_q.if2"] and result.params["model.l_q.if2"] is q and [c.key for c in result.choices] == ["model.l_q.if2", "sq.hyst"]
    with pytest.raises(TemplateRefusal, match="written twice"):
        b.choice("sq.hyst", 0.1, "V", "again")
    out = apply_prefix(result, BlockPrefix(400, "RXA_"))
    assert [n.name for n in out.nets] == ["RSSI", "RXA_SQ_REF", "MUTE", "RX_3V3"] and out.open_pins == {("U403", "2"): "test: left open on purpose"}


# --------------------------------------------------------------------------- the real KiCad 10.0.6 libraries


@needs_libs
def test_every_part_instantiates_from_the_real_libraries() -> None:
    notes: dict[str, tuple[str, ...]] = {}
    for key in PARTS:
        placed = instantiate(_REAL, key, "X1", "v", "d", STRUCT)
        notes[key] = placed.notes
        stacks = symbol_stacks(_REAL.load_symbol(placed.component.symbol))
        assert stacks == placed.stacks
    assert instantiate(_REAL, "mic_amp", "U1", "v", "d", STRUCT).stacks == (("4", "7", "11", "15"),)
    assert instantiate(_REAL, "mixer", "U1", "v", "d", STRUCT).stacks == (("1", "4", "5"),)
    assert instantiate(_REAL, "tcxo", "Y1", "v", "d", STRUCT).stacks == (("1", "3"),)
    assert any("identified as OUT by its library electrical type 'output'" in n for n in notes["opamp"])
    assert any("'open_collector'" in n for n in notes["comparator"]) and any("A AND B = B AND A" in n for n in notes["and_gate"])
    assert instantiate(_REAL, "opamp", "U1", "v", "d", STRUCT).functions["OUT"] == ("1",)
    assert instantiate(_REAL, "ldo_rx5v", "U1", "v", "d", STRUCT).functions == {"OUT": ("1",), "GND": ("2",), "IN": ("3",)}  # the library's order (critic2)


@needs_libs
def test_the_microphone_is_the_through_hole_capsule_whose_pads_follow_the_symbol_pins_and_whose_polarity_is_unverified() -> None:
    """Decision 1A: the microphone is ``Sensor_Audio:POM-2244P-C3310-2-R`` - two plain through-hole pads (no custom ring pad around the
    signal pad, so the + terminal needs no via in a pad). The symbol's pin 1 is named ``-`` and pin 2 ``+``; the footprint's pads carry
    the same numbers, so ``-`` lands on pad 1 and ``+`` on pad 2 (invariant 2: both read from the library files). Which terminal PUI Audio
    makes the case is not checked against its datasheet, and the row says so."""
    row = PARTS["mic"]
    assert row.footprint_id == "Sensor_Audio:POM-2244P-C3310-2-R" and row.lib_id == "Device:Microphone_Condenser"
    assert any("case / negative" in u and u.endswith("[UNVERIFIED: PUI Audio POM-2244P-C3310-2-R datasheet]") for u in row.unverified)
    placed = instantiate(_REAL, "mic", "MK1", "v", "d", STRUCT)
    assert placed.functions == {"-": ("1",), "+": ("2",)}
    fp = _REAL.load_footprint(placed.component.footprint)
    assert fp.attr == "through_hole"
    pads = {p.number: (p.pad_type, p.shape, p.x, p.y, p.drill) for p in fp.pads}
    assert pads == {"1": ("thru_hole", "roundrect", 0.0, 0.0, 0.65), "2": ("thru_hole", "circle", 1.9, 0.0, 0.65)}
    assert {p.number for p in _REAL.load_symbol(placed.component.symbol).pins} == set(pads)


@needs_libs
def test_refused_rows_refuse_with_the_compilers_reasons() -> None:
    reasons = {s[1]: check_refused(_REAL, s) for s in REFUSED_PARTS}
    assert "multi-unit symbols are not supported" in reasons["MCP6002-xSN"]
    assert "pads ['MP'] that are not pins" in reasons["R_Potentiometer"]
    assert "pads ['17'] that are not pins" in reasons["SKY13380-350LF"]


@needs_libs
def test_one_of_every_part_compiles_through_the_schematic_and_pcb_compilers(tmp_path: Path) -> None:
    from ai_eda.compilers.pcb import PCBCompiler
    from ai_eda.compilers.schematic import SchematicCompiler

    ir = _ir(tmp_path, "rfparts")
    placed = [instantiate(_REAL, key, f"X{i + 1}", "v", key, STRUCT) for i, key in enumerate(PARTS)]
    ir.components = [p.component for p in placed]
    nets = []
    for p in placed:
        for k, (function, numbers) in enumerate(p.functions.items()):
            if all(p.component.pin(n).electrical_type is PinElectricalType.NO_CONNECT for n in numbers):
                continue
            nets.append(_net(f"N_{p.ref}_{k}", *p.at(function)))  # every pin of a function in one net: stacked pins together
    ir.nets = nets
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=560.0, height_mm=240.0),
                       placements=[Placement(component_ref=p.ref, x_mm=10.0 + 50.0 * (i % 11), y_mm=20.0 + 40.0 * (i // 11), side=BoardSide.TOP, provenance=STRUCT)
                                   for i, p in enumerate(placed)])
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": _REAL})
    for compiler in (SchematicCompiler(), PCBCompiler()):
        ref = compiler.compile(ir, ctx)
        assert Path(ref.path).is_file()
