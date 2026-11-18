"""``spice.xtalk.<victim>``: coupled-line (CPL) decks of the victims ``si.crosstalk`` hands off, judged against a budget.

The boards are synthetic (the fixture library of ``tests/test_routing.py``,
the 4-layer generic stack, the 50-ohm width): ``VIC`` (R1.1 an output ->
R2.1 an input) runs on F.Cu from x 10 to 60 mm; ``AGG`` (R3.1 -> R4.1)
comes up on B.Cu, runs beside it on F.Cu from x 45 to 20 mm at an edge gap of
1.5 h and goes down again through two vias - every neighbour computed, both
nets two-pad lines, so each hands off to a deck of the other. Without
ngspice: the plan places the span on both paths at the hand-computed
distances and the compiler writes the golden deck; a stub runner's set
waveforms give PASS (sum within the budget), FAIL (one exact deck alone
over it, ``repair: human``, the aggressor named), NOT_VERIFIED (only the sum
over it, only a lumped span over it, only one orientation of an exact span
over it - named, no engine, a refused deck, a missing driver value or budget,
the cap; a victim over the cap alone is named and the next one still runs);
an aggressor that is a tree is not handed off; stale results are superseded
and the SPICE stage never retires them; the budget is the user's stated
value first, else NM / 3 from grounded levels over the driver's VCC (the
SN74LVC1G04 rows, SCEA005's LVTTL, an ATmega128 at 3 V), else NOT_VERIFIED
naming the part and key;
the report and ``domain.high_speed.si`` 0.2 show the verdicts. With ngspice
(KiCad 10.0.6's ``ngspice.dll``, ngspice-46; one process per run): a deck
whose CPL ports sit at the pads confirms itself at its step and at 2x it
(measured 2026-11-13) and the victim PASSes its budget, with the deck and
both rawfiles as evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ai_eda.compilers.spice import build
from ai_eda.ir import (
    CircuitIR,
    NetKind,
    PinElectricalType,
    PinRef,
    Provenance,
    ProvenanceKind,
    SourceRef,
    ValidationResult,
    Via,
    assumption,
    authoritative,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.si import NetClass
from ai_eda.report.stages import ReportFigures
from ai_eda.report.si_report import crosstalk_lines, spice_lines
from ai_eda.tools.calc.tline import solve_microstrip_width
from ai_eda.tools.si.measure import measure_nets
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice import xtalk_check as X
from ai_eda.tools.spice.si_check import deck_stem
from ai_eda.tools.spice.stage import retire_expectation_results
from ai_eda.tools.spice.xtalk_check import DeckRun, Span, XtPiece, nm3_budget, spice_xtalk_results, victim_budget, xtalk_deck
from ai_eda.validation import ValidationContext
from ai_eda.validation.si import crosstalk_scan
from tests.test_routing import board_ir, fixture_library
from tests.test_si_checks import default_classes, si_of, stacked, track

needs_ngspice = pytest.mark.skipif(not NgspiceShared().available(), reason="ngspice shared library not found")
#: the 50-ohm width on the 4-layer generic F.Cu (h 0.2 mm, t 35 um, er 4.5) and an edge gap of 1.5 h
W = round(solve_microstrip_width(50.0, 0.2, 35.0, 4.5), 4)
GAP = 0.3
YA = 10.0 + W + GAP
YB = 10.0 - W - GAP
#: the SN74LVC1G04 (TI SCES214AF sec. 5.3 / 5.5) rows as crosstalk.json's budget entry quotes them, VCC = 3 V - fixture facts
DS = SourceRef(title="SN74LVC1G04 datasheet SCES214AF, sections 5.3 and 5.5 (as quoted by the crosstalk research step)", authority="Texas Instruments")
#: the 3.3-V LVTTL levels of TI SCEA005 Figure 1 (VCC 3.3 V) and the ATmega128 at VCC = 3 V (2467X DC characteristics), both as
#: crosstalk.json's budget entries quote them - fixture facts
SCEA005 = SourceRef(title="TI SCEA005 Figure 1, 3.3-V VCC family switching levels (as quoted by the crosstalk research step)", authority="Texas Instruments")
ATMEGA = SourceRef(title="ATmega128 datasheet 2467X-AVR-06/11, DC Characteristics at VCC = 3 V (as quoted by the crosstalk research step)",
                   authority="Microchip")


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def _via(x: float, y: float, net: str) -> Via:
    return Via(net=net, x_mm=x, y_mm=y, diameter_mm=0.6, drill_mm=0.3)


def pair_ir(tmp_path: Path, lib, *, budget: float | None = 0.05, below: bool = False, **class_kw) -> CircuitIR:
    """VIC and AGG (module docstring); ``below``: a third net AG2 beside VIC on the other side, x 25 to 40 mm."""
    parts = [("R1", "PAD1", 10.0, 10.0), ("R2", "PAD1", 60.0, 10.0), ("R3", "PAD1", 45.0, 20.0), ("R4", "PAD1", 20.0, 20.0)]
    nets = {"VIC": [("R1", "1"), ("R2", "1")], "AGG": [("R3", "1"), ("R4", "1")]}
    if below:
        parts += [("R5", "PAD1", 25.0, 0.0), ("R6", "PAD1", 40.0, 0.0)]
        nets["AG2"] = [("R5", "1"), ("R6", "1")]
    ir = board_ir(tmp_path, lib, parts, nets, (70.0, 30.0))
    stacked(ir, 4)
    for ref in ("R1", "R3", "R5"):
        if ir.component(ref) is not None:
            ir.component(ref).pins[0].electrical_type = PinElectricalType.OUTPUT
    for ref in ("R2", "R4", "R6"):
        if ir.component(ref) is not None:
            ir.component(ref).pins[0].electrical_type = PinElectricalType.INPUT
    kw = dict(class_kw)
    if budget is not None:
        kw["xtalk_budget_rel"] = user_requirement(budget)
    ir.si = si_of(*default_classes(**kw))
    ir.pcb.tracks = [track("VIC", (10.0, 10.0), (60.0, 10.0), width=W),
                     track("AGG", (45.0, 20.0), (45.0, YA), width=W, layer="B.Cu"), track("AGG", (45.0, YA), (20.0, YA), width=W),
                     track("AGG", (20.0, YA), (20.0, 20.0), width=W, layer="B.Cu")]
    ir.pcb.vias = [_via(45.0, YA, "AGG"), _via(20.0, YA, "AGG")]
    if below:
        ir.pcb.tracks += [track("AG2", (25.0, 0.0), (25.0, YB), width=W, layer="B.Cu"), track("AG2", (25.0, YB), (40.0, YB), width=W),
                          track("AG2", (40.0, YB), (40.0, 0.0), width=W, layer="B.Cu")]
        ir.pcb.vias += [_via(25.0, YB, "AG2"), _via(40.0, YB, "AG2")]
    return ir


def stem_of(victim: str, aggressor: str) -> str:
    return f"{deck_stem(victim)}-{hashlib.sha256(aggressor.encode('utf-8')).hexdigest()[:8]}-0"


class Stub(SpiceRunner):
    """Set waveforms per deck: ``peaks[(victim, aggressor)]`` at the victim's receiver (``v1``, or ``v0`` when the victim is driven from its
    other end), the aggressor's far end (``a1`` / ``a0``) settled at 1 V; ``refuse`` victims get a refused run."""

    engine = "stub"

    def __init__(self, peaks: dict[tuple[str, str], float], refuse: tuple[str, ...] = ()) -> None:
        self.peaks, self.refuse, self.calls = peaks, refuse, []

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "stub-1"

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None) -> SpiceResult:
        self.calls.append((Path(netlist_path).name, command))
        base = dict(engine=self.engine, engine_version="stub-1", netlist_path=str(netlist_path), netlist_hash="sha256:0", analysis=analysis, command=command or "")
        for (v, a), peak in self.peaks.items():
            if Path(netlist_path).stem.startswith(stem_of(v, a)):
                if v in self.refuse:
                    return SpiceResult(**base, errors=["line 3: refused by the deck check"], succeeded=False)
                time = [0.0, 1e-9, 2e-9, 3e-8]
                victim, far = [0.0, peak, 0.0, 0.0], [0.0, 0.5, 1.0, 1.0]
                return SpiceResult(**base, vectors={"time": time, "v0": victim, "v1": victim, "a0": far, "a1": far}, scale="time", n_points=4, succeeded=True)
        raise AssertionError(f"no waveform set for {netlist_path}")


def run(ir: CircuitIR, lib, tmp_path: Path, runner: SpiceRunner | None) -> dict[str, ValidationResult]:
    tools = {"kicad_library": lib, **({"spice": runner} if runner is not None else {})}
    return {r.check_id: r for r in spice_xtalk_results(ir, tools, tmp_path / "w")}


# --------------------------------------------------------------------------- the budget


def test_the_budget_field_is_only_a_fraction_the_user_stated_and_none_keeps_the_hash(tmp_path: Path, lib):
    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")
    for value, expect in ((assumption(0.05, "a template's 5 %"), "only as a budget the user stated"), (user_requirement(1.5), r"must be in \(0, 1\]"),
                          (user_requirement(0.0), "> 0"), (user_requirement(0.05, "%"), "no unit")):
        with pytest.raises(ValueError, match=expect):
            NetClass(name="A", xtalk_budget_rel=value, provenance=prov)
    ir = pair_ir(tmp_path, lib, budget=None)
    h = ir.content_hash()
    ir.si.net_classes[0].xtalk_budget_rel = None
    assert ir.content_hash() == h  # a None budget is not design content
    with_budget = pair_ir(tmp_path / "b", lib, budget=0.05)
    assert with_budget.content_hash() != pair_ir(tmp_path / "c", lib, budget=None).content_hash()
    assert ("si.net_classes[DEFAULT].xtalk_budget_rel", with_budget.si.net_classes[0].xtalk_budget_rel) in with_budget.si.traced_items()


def _levels(ir: CircuitIR, ref: str, source: SourceRef = DS, **facts: tuple[float, str]) -> None:
    comp = ir.component(ref)
    for key, (value, unit) in facts.items():
        comp.electrical[key] = authoritative(value, source, unit, note=f"fixture: {source.title}")


def test_nm3_from_the_lvc1g04_rows(tmp_path: Path, lib):
    """VCC = 3 V. At 16 mA: NMH = 2.4 - 2.0 = NML = 0.8 - 0.4 = 0.4 V; at 100 uA: NMH = 2.9 - 2.0 = 0.9 V, NML = 0.8 - 0.1 = 0.7 V. The
    swing is VCC (an upper bound of the aggressor's step), not VOH - VOL (2.0 / 2.8 V, which would overstate the budget)."""
    ir = pair_ir(tmp_path, lib, budget=None)
    _levels(ir, "R2", v_ih=(2.0, "V"), v_il=(0.8, "V"))
    _levels(ir, "R1", v_oh=(2.4, "V"), v_ol=(0.4, "V"), v_cc=(3.0, "V"), i_oh=(-0.016, "A"), i_ol=(0.016, "A"))
    value, det, why = nm3_budget(ir, "R1", "R2")
    assert why == "" and det["nmh_v"] == pytest.approx(0.4) and det["nml_v"] == pytest.approx(0.4) and det["swing_v"] == 3.0
    assert value == pytest.approx(0.4 / 3.0 / 3.0) and det["i_oh"] == -0.016 and det["row"].startswith("v_oh at i_oh")
    _levels(ir, "R1", v_oh=(2.9, "V"), v_ol=(0.1, "V"), i_oh=(-1e-4, "A"), i_ol=(1e-4, "A"))
    value, det, why = nm3_budget(ir, "R1", "R2")
    assert det["nmh_v"] == pytest.approx(0.9) and det["nml_v"] == pytest.approx(0.7) and value == pytest.approx(0.7 / 3.0 / 3.0)
    # the check reads it: VIC's budget is NM / 3 of R1 -> R2
    plan = crosstalk_scan(ir, ir.si, measure_nets(ir, library=lib)).plans["VIC"]
    b = victim_budget(ir, ir.si.class_of("VIC"), "VIC", NetKind.SIGNAL, X._directions(ir, plan.victim))
    assert b.value == pytest.approx(0.7 / 3.0 / 3.0) and b.source.startswith("NM / 3 of the grounded datasheet levels (R1 -> R2)")
    assert "swing = VCC_drv" in b.details["rule"]


def test_the_swing_is_the_drivers_supply_so_nm3_gives_the_plans_figures(tmp_path: Path, lib):
    """crosstalk.json P4: 3.3-V LVTTL (SCEA005 Fig. 1: VOH 2.4, VIH 2.0, VIL 0.8, VOL 0.4 V) gives 0.4 V / 3 of 3.3 V = 4.0 %, tighter than
    Bogatin's 5 % (over VOH - VOL = 2.0 V it would be 6.7 %, looser); an ATmega128 at 3 V (VOH 2.2 V at -10 mA, VOL 0.5 V at 10 mA, VIH 1.8 V,
    VIL 0.6 V) gives 0.1 V / 3 of 3 V = 1.1 %, so a deck peak of 1.5 % FAILs - over VOH - VOL = 1.7 V (1.96 %) it would PASS while a 3 V
    aggressor step puts 45 mV on the victim, above NM / 3 = 33 mV."""
    ir = pair_ir(tmp_path, lib, budget=None)
    _levels(ir, "R1", SCEA005, v_oh=(2.4, "V"), v_ol=(0.4, "V"), v_cc=(3.3, "V"))
    _levels(ir, "R2", SCEA005, v_ih=(2.0, "V"), v_il=(0.8, "V"))
    value, det, why = nm3_budget(ir, "R1", "R2")
    assert why == "" and value == pytest.approx(0.4 / 3.0 / 3.3) and round(value, 3) == 0.040 and det["swing_v"] == 3.3
    del ir.component("R1").electrical["v_cc"]
    assert nm3_budget(ir, "R1", "R2")[2] == "missing the grounded datasheet fact(s) R1.v_cc"
    _levels(ir, "R1", SCEA005, v_cc=(2.0, "V"))  # below its own VOH: not the supply its levels were read at
    assert "R1's v_cc 2 V is not a supply at or above its v_oh 2.4 V" in nm3_budget(ir, "R1", "R2")[2]
    _levels(ir, "R1", ATMEGA, v_oh=(2.2, "V"), v_ol=(0.5, "V"), v_cc=(3.0, "V"), i_oh=(-0.010, "A"), i_ol=(0.010, "A"))
    _levels(ir, "R2", ATMEGA, v_ih=(1.8, "V"), v_il=(0.6, "V"))
    assert nm3_budget(ir, "R1", "R2")[0] == pytest.approx(0.1 / 3.0 / 3.0)
    r = run(ir, lib, tmp_path, Stub({("VIC", "AGG"): 0.015, ("AGG", "VIC"): 0.015}))["spice.xtalk.VIC"]
    assert r.status is S.FAIL and "1.500% of the aggressor step > the budget 1.111%" in r.message and "NM / 3 of the grounded" in r.message
    assert r.details["budget"]["directions"][0]["swing_v"] == 3.0 and "supply v_cc" in r.details["swing"]


def test_a_missing_level_a_connector_a_per_pin_threshold_and_no_margin_are_named(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib, budget=None)
    _levels(ir, "R1", v_oh=(2.4, "V"))
    _levels(ir, "R2", v_ih=(2.0, "V"), v_il=(3.0, "V/m"))  # a wrong unit is not a grounded level
    assert nm3_budget(ir, "R1", "R2")[2] == "missing the grounded datasheet fact(s) R1.v_ol, R1.v_cc, R2.v_il"
    r = run(ir, lib, tmp_path, Stub({}))["spice.xtalk.VIC"]
    assert r.status is S.NOT_VERIFIED and "no crosstalk budget" in r.message and "si.net_classes[DEFAULT].xtalk_budget_rel" in r.message
    assert "R1.v_ol" in r.message and "R1.v_cc" in r.message and "R2.v_il" in r.message and not r.tool  # nothing ran
    ir.component("R2").electrical["v_il"] = authoritative(0.8, DS, "V")
    ir.component("R1").electrical["v_ol"] = authoritative(0.4, DS, "V")
    ir.component("R1").electrical["v_cc"] = authoritative(3.0, DS, "V")
    ir.component("R2").electrical["v_ih2"] = authoritative(2.55, DS, "V")
    assert "more than one input threshold (v_ih2)" in nm3_budget(ir, "R1", "R2")[2]
    del ir.component("R2").electrical["v_ih2"]
    ir.component("R2").electrical["v_ih"] = authoritative(2.5, DS, "V")  # VOH 2.4 < VIH 2.5
    assert "no noise margin (NMH = -0.1 V" in nm3_budget(ir, "R1", "R2")[2]
    ir.component("R2").symbol = ir.component("R2").symbol.model_copy(update={"library": "Connector_Generic"})
    assert nm3_budget(ir, "R1", "R2")[2] == "R2 is a connector: the levels are the off-board part's behind it"


def test_a_stated_budget_takes_precedence_and_is_the_only_one_for_an_analog_victim(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib, budget=0.02)
    _levels(ir, "R1", v_oh=(2.4, "V"), v_ol=(0.4, "V"))
    _levels(ir, "R2", v_ih=(2.0, "V"), v_il=(0.8, "V"))
    cls = ir.si.class_of("VIC")
    b = victim_budget(ir, cls, "VIC", NetKind.SIGNAL, [("R1", "R2")])
    assert b.value == 0.02 and b.source == "si.net_classes[DEFAULT].xtalk_budget_rel (stated by the user)"
    assert victim_budget(ir, cls, "VIC", NetKind.ANALOG, [("R1", "R2")]).value == 0.02
    none = pair_ir(tmp_path / "n", lib, budget=None)
    b = victim_budget(none, none.si.class_of("VIC"), "VIC", NetKind.ANALOG, [("R1", "R2")])
    assert b.value is None and b.missing == f"si.net_classes[DEFAULT].xtalk_budget_rel: {X.NO_ANALOG_BUDGET}"
    none.nets[0].kind = NetKind.ANALOG
    r = run(none, lib, tmp_path / "w", Stub({}))["spice.xtalk.VIC"]
    assert r.status is S.NOT_VERIFIED and X.NO_ANALOG_BUDGET in r.message


# --------------------------------------------------------------------------- the plan and the deck


GOLDEN = """xt_VIC-c520993f
CLA A1 0 5p
CLV V1 0 5p
P1 VI AI 0 VO AO 0 P1_cpl
.model P1_cpl CPL length=25m R=1 0 1 L=305.3607688001450n 38.16044505988813n 305.3607688001450n G=0 0 0 C=123.94627348895425p -6.917475738082569p 123.94627348895425p
RSA S A0 25
RSV V0 0 25
TA0 A0 0 AO 0 td=67.47366537877631p z0=49.99662307382530
TA1 AI 0 A1 0 td=67.47366537877632p z0=49.99662307382530
TV0 V0 0 VI 0 td=60.03263402580592p z0=49.99662307382530
TV1 VO 0 V1 0 td=90.0489510387089p z0=49.99662307382530
VVDRV S 0 PULSE(0 1 0 1n 1n 37.09358938396155n 76.1871787679231n)
.end
"""


def test_the_span_sits_on_both_paths_at_the_hand_computed_distances_and_the_deck_is_golden(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib)
    measures = measure_nets(ir, library=lib)
    scan = crosstalk_scan(ir, ir.si, measures)
    rows = {r["net"]: r for r in scan.result.details["rows"]}
    assert rows["VIC"]["status"] == S.NOT_APPLICABLE.value and rows["VIC"]["judged_by"] == "spice.xtalk.VIC"
    assert rows["VIC"]["reason"].startswith("judged by spice.xtalk.VIC (1 coupled-line deck(s)), not by this rule")
    assert scan.result.tool_version == "0.2" and scan.result.status is S.NOT_APPLICABLE and set(scan.plans) == {"VIC", "AGG"}
    plan = scan.plans["VIC"]
    (span,) = plan.spans
    barrel = ir.pcb.stackup.through_via_span_mm()
    stub = 20.0 - YA  # AGG's B.Cu leg from its pad down to the via
    # VIC runs from R1.1 at x 10: the span x 20..45 is 10..35 mm along it; AGG's path R3.1 -> R4.1 meets x 45 after its leg and a barrel
    assert span.v_pos == pytest.approx((10.0, 35.0), abs=1e-9) and span.length_mm == pytest.approx(25.0, abs=1e-9)
    assert span.a_pos == pytest.approx((stub + barrel + 25.0, stub + barrel), abs=1e-9)  # beside x 20, beside x 45: the other way round
    assert span.exact and span.gap_mm == pytest.approx(GAP, abs=1e-9) and plan.victim.drive_ends == (0,) and plan.aggressors["AGG"].drive_ends == (0,)
    assert [e.layer for e in plan.aggressors["AGG"].path.edges] == ["B.Cu", None, "F.Cu", None, "B.Cu"]
    d = xtalk_deck("xt_" + deck_stem("VIC"), victim=plan.victim, aggressor=plan.aggressors["AGG"], span=span, v_drive=0, a_drive=0, t_r=1e-9,
                   r_v=25.0, c_v=5e-12, r_a=25.0, c_a=5e-12)
    assert build(d.ir) == GOLDEN and build(d.ir) == build(xtalk_deck("xt_" + deck_stem("VIC"), victim=plan.victim, aggressor=plan.aggressors["AGG"],
                                                                        span=span, v_drive=0, a_drive=0, t_r=1e-9, r_v=25.0, c_v=5e-12, r_a=25.0,
                                                                        c_a=5e-12).ir)
    lines = d.info["lines"]
    assert lines["TV0"]["length_mm"] == pytest.approx(10.0) and lines["TV1"]["length_mm"] == pytest.approx(15.0)
    assert lines["TA0"]["length_mm"] == pytest.approx(stub + barrel) == lines["TA1"]["length_mm"]
    assert (d.victim_node, d.aggressor_far) == ("v1", "a1") and d.step == pytest.approx(lines["TV0"]["td_s"] / 50.0)
    assert d.info["points"] < X.MAX_POINTS


# --------------------------------------------------------------------------- the verdicts (stub runner)


def test_pass_when_the_sum_is_within_the_budget_and_fail_on_one_exact_deck_alone(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib, below=True)
    stub = Stub({("VIC", "AGG"): 0.02, ("VIC", "AG2"): 0.02, ("AGG", "VIC"): 0.02, ("AGG", "AG2"): 0.001, ("AG2", "VIC"): 0.02, ("AG2", "AGG"): 0.001})
    res = run(ir, lib, tmp_path, stub)
    vic = res["spice.xtalk.VIC"]
    assert vic.status is S.PASS and vic.tool == "stub" and vic.tool_version == "stub-1" and vic.ir_hash == ir.content_hash()
    assert vic.details["measured"]["total_rel"] == pytest.approx(0.04) and vic.details["deck_version"] == "0.1"
    assert vic.artifact_hash and any(e.description.startswith("spice.xtalk deck") for e in vic.evidence)
    assert len(vic.details["decks"]) == 2 and all(len(d["runs"]) == 2 for d in vic.details["decks"])  # each deck at its step and at 2x it
    assert vic.details["swing"].startswith("a deck's 1 V aggressor step stands for the victim's swing") and vic.details["model_omits"] == X.XT_MODEL_OMITS
    assert "an aggressor on a higher supply puts proportionally more noise on the victim" in vic.details["swing"]
    fine, coarse = vic.details["decks"][0]["runs"]
    assert float(coarse["command"].split()[1].rstrip("p")) == pytest.approx(2.0 * float(fine["command"].split()[1].rstrip("p")))
    stub.peaks[("VIC", "AGG")] = 0.06
    vic = run(ir, lib, tmp_path / "f", stub)["spice.xtalk.VIC"]
    assert vic.status is S.FAIL and vic.details["repair"] == "human"
    assert "the deck of AGG on F.Cu, span 0" in vic.message and "gap 0.3 mm" in vic.message and "6.000% of the aggressor step > the budget 5.000%" in vic.message


def test_not_verified_when_only_the_sum_exceeds_or_only_a_lumped_span_does(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib, below=True)
    stub = Stub({("VIC", "AGG"): 0.03, ("VIC", "AG2"): 0.03, ("AGG", "VIC"): 0.01, ("AGG", "AG2"): 0.001, ("AG2", "VIC"): 0.01, ("AG2", "AGG"): 0.001})
    vic = run(ir, lib, tmp_path, stub)["spice.xtalk.VIC"]
    assert vic.status is S.NOT_VERIFIED and "only the sum of the spans' peaks, 6.000%, exceeds the budget 5.000%" in vic.message
    assert "aggressors switching together not simulated" in vic.message
    piece = XtPiece("AGG", "F.Cu", W, 0.25, 10.0, ((0, 0), (10, 0)), ((0, 1), (10, 1)), 54.0, 45.0, 3.6, 3.1)
    lumped = Span("AGG", 0, "F.Cu", W, (0.3, 0.25), (0.0, 20.0), (0.0, 20.0), piece)
    exact = Span("AGG", 0, "F.Cu", W, (0.25,), (0.0, 20.0), (0.0, 20.0), piece)

    def deck(span: Span, peak: float) -> DeckRun:
        return DeckRun(span=span, v_drive=0, a_drive=0, path=tmp_path / "x.cir", deck_hash="", text="", info={}, peak=peak)

    status, message, _ = X._judge("V", 0.05, [deck(lumped, 0.06)])
    assert status is S.NOT_VERIFIED and "only a span lumped at its smallest gap exceeds the budget" in message and "lumped at the smallest of gaps 0.25, 0.3" in message
    assert X._judge("V", 0.05, [deck(exact, 0.06)])[0] is S.FAIL
    # an orientation that cannot be told never makes a FAIL: the most favourable one must exceed too; the reason names the orientation over
    # the budget (not "aggressors switching together": one aggressor, one span)
    ends = {"victim_driver": "R1.1", "aggressor_driver": "R3.1"}
    two = [DeckRun(span=exact, v_drive=0, a_drive=0, path=tmp_path / "x.cir", deck_hash="", text="", info=ends, peak=0.06),
           DeckRun(span=exact, v_drive=1, a_drive=0, path=tmp_path / "y.cir", deck_hash="", text="", info={**ends, "victim_driver": "R2.1"}, peak=0.01)]
    status, message, _ = X._judge("V", 0.05, two)
    assert status is S.NOT_VERIFIED and message.startswith("V: the deck of AGG on F.Cu, span 0")
    assert ("exceeds the budget 5.000% only with victim driven from R1.1, aggressor from R3.1: 6.000% (1.000% in its most favourable orientation): "
            "which end drives cannot be told, so no orientation makes a verdict") in message
    assert "switching together" not in message and "no span alone" not in message and "R2.1" not in message


def test_every_reason_not_to_judge_is_not_verified_naming_it(tmp_path: Path, lib, monkeypatch):
    ir = pair_ir(tmp_path, lib)
    peaks = {("VIC", "AGG"): 0.01, ("AGG", "VIC"): 0.01}
    none = run(ir, lib, tmp_path / "a", None)["spice.xtalk.VIC"]
    assert none.status is S.NOT_VERIFIED and none.message == "VIC: no SPICE engine available (1 deck(s) built, none run)"
    refused = run(ir, lib, tmp_path / "b", Stub(peaks, refuse=("VIC",)))
    assert refused["spice.xtalk.VIC"].status is S.NOT_VERIFIED and "1 of 1 deck(s) gave no usable peak" in refused["spice.xtalk.VIC"].message
    assert "refused by the deck check" in refused["spice.xtalk.VIC"].message and refused["spice.xtalk.AGG"].status is S.PASS
    no_r = pair_ir(tmp_path / "c", lib, r_drive_ohm=None)
    r = run(no_r, lib, tmp_path / "c", Stub(peaks))["spice.xtalk.VIC"]
    assert r.status is S.NOT_VERIFIED and "missing si.net_classes[DEFAULT].r_drive_ohm" in r.message
    monkeypatch.setattr(X, "MAX_XTALK_DECKS", 1)
    capped = run(ir, lib, tmp_path / "d", Stub(peaks))
    statuses = sorted(r.status.value for r in capped.values())
    assert statuses == ["NOT_VERIFIED", "PASS"]
    (cut,) = [r for r in capped.values() if r.status is S.NOT_VERIFIED]
    assert "not simulated: below the 1 worst by the matched-line estimate (MAX_XTALK_DECKS = 1" in cut.message
    monkeypatch.setattr(X, "MAX_XTALK_DECKS", 8)
    # an aggressor that is a tree is not handed off: si.crosstalk names it, spice.xtalk builds nothing for that victim
    tree = pair_ir(tmp_path / "e", lib)
    tree.net("AGG").pins.append(PinRef(component_ref="R9", pin_number="1"))  # a third pad
    scan = crosstalk_scan(tree, tree.si, measure_nets(tree, library=lib))
    row = {x["net"]: x for x in scan.result.details["rows"]}["VIC"]
    assert row["status"] == S.NOT_VERIFIED.value and "no coupled-line deck: aggressor AGG joins 3 pads" in row["reason"] and "VIC" not in scan.plans
    assert "spice.xtalk.VIC" not in run(tree, lib, tmp_path / "e", Stub(peaks))


def test_a_victim_over_the_cap_alone_is_named_and_does_not_stop_the_victims_after_it(tmp_path: Path, lib, monkeypatch):
    """VIC's two pads bidirectional: its driver cannot be told, so as a victim it needs 2 x (1 + 1) = 4 decks and as an aggressor it runs both
    ways (AGG and AG2: 2 + 1 = 3 decks each). VIC ranks first by the matched-line estimate; with a cap of 3 decks it alone is over the cap and
    says so, AGG still runs, AG2 is below the 1 worst."""
    ir = pair_ir(tmp_path, lib, below=True)
    for ref in ("R1", "R2"):
        ir.component(ref).pins[0].electrical_type = PinElectricalType.BIDIRECTIONAL
    peaks = {(v, a): 0.01 for v in ("VIC", "AGG", "AG2") for a in ("VIC", "AGG", "AG2") if v != a}
    monkeypatch.setattr(X, "MAX_XTALK_DECKS", 3)
    stub = Stub(peaks)
    res = run(ir, lib, tmp_path, stub)
    vic, agg, ag2 = res["spice.xtalk.VIC"], res["spice.xtalk.AGG"], res["spice.xtalk.AG2"]
    assert vic.status is S.NOT_VERIFIED and vic.message == "VIC: not simulated: it alone needs 4 decks, more than MAX_XTALK_DECKS = 3 per run"
    assert vic.details["decks_needed"] == 4 and not vic.tool
    assert agg.status is S.PASS and len(agg.details["decks"]) == 3 and len(stub.calls) == 6  # its 3 decks, each at two steps
    assert ag2.status is S.NOT_VERIFIED and ag2.message.startswith("AG2: not simulated: below the 1 worst by the matched-line estimate")


def test_stale_results_are_superseded_and_the_spice_stage_leaves_them_alone(tmp_path: Path, lib):
    ir = pair_ir(tmp_path, lib)
    ir.validation.add(ValidationResult(check_id="spice.xtalk.OLD", status=S.FAIL, message="an earlier victim"))
    ir.validation.add(ValidationResult(check_id="spice.gone", status=S.PASS, message="an expectation no longer in the setup"))
    assert [r.check_id for r in retire_expectation_results(ir, set(), S.NOT_APPLICABLE, "gone")] == ["spice.gone"]
    res = run(ir, lib, tmp_path, Stub({("VIC", "AGG"): 0.01, ("AGG", "VIC"): 0.01}))
    assert res["spice.xtalk.OLD"].status is S.NOT_APPLICABLE and res["spice.xtalk.OLD"].details["superseded"] == "FAIL"
    assert list(res) == ["spice.xtalk.VIC", "spice.xtalk.AGG", "spice.xtalk.OLD"]  # IR net order, then the superseded ones


def test_the_report_lists_the_decks_and_the_domain_check_takes_fresh_verdicts(tmp_path: Path, lib):
    from ai_eda.validation.domain import HS_VERSION, SignalIntegrityValidator
    from ai_eda.validation.si import crosstalk_result

    ir = pair_ir(tmp_path, lib)
    ir.validation.add(crosstalk_result(ir, ir.si, measure_nets(ir, library=lib)))
    v = SignalIntegrityValidator()
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    (before,) = v.validate(ir, ctx)
    assert v.consumes == frozenset({"spice"}) and HS_VERSION == "0.2"
    assert before.status is S.NOT_VERIFIED and "spice.xtalk.VIC (no result for the current IR)" in before.message
    stub = Stub({("VIC", "AGG"): 0.06, ("AGG", "VIC"): 0.01})
    for r in spice_xtalk_results(ir, {"kicad_library": lib, "spice": stub}, tmp_path / "w"):
        ir.validation.add(r)
    (after,) = v.validate(ir, ctx)
    rows = {x["net"]: x for x in after.details["rows"]}
    assert after.status is S.FAIL and after.details["repair"] == "human" and rows["VIC"]["status"] == "FAIL" and rows["AGG"]["status"] == "PASS"
    assert rows["VIC"]["judged_result"]["check"] == "spice.xtalk.VIC" and "coupled-line verdicts: spice.xtalk.VIC FAIL, spice.xtalk.AGG PASS" in after.message
    text = "\n".join(crosstalk_lines(ir))
    assert "#### 결합선 시뮬레이션 (`spice.xtalk.<피해 넷>`, ngspice CPL)" in text and "`spice.xtalk.VIC`: **FAIL**" in text and "`spice.xtalk.AGG`: **PASS**" in text
    assert "6.000" in text  # the deck's recorded peak
    assert "spice.xtalk" not in "\n".join(spice_lines(ir, ReportFigures()))  # never a reflection result
    ir.parameters["fixture_change"] = user_requirement(1.0, "V")  # another IR (the copper as it was): the verdicts are no longer fresh
    (stale,) = v.validate(ir, ctx)
    assert stale.status is S.NOT_VERIFIED and "spice.xtalk.VIC (no result for the current IR)" in stale.message


# --------------------------------------------------------------------------- the engine (one process per run)


@needs_ngspice
def test_a_deck_with_its_ports_at_the_pads_confirms_itself_at_its_step_on_ngspice_46(tmp_path: Path, lib):
    """Both nets exactly the span (no T lines; AGG driven from the end beside VIC's receiver): each step runs in a process of its own; at the
    rule's step (2.96 ps) and at 2x it the deck settles within SETTLE_TOL and the peaks agree (measured 2026-11-13: 2.033 % and 2.022 %),
    so VIC PASSes its 5 % budget. The same deck at a 1.2 ps step reads 2.214 % and settles at 0.9869 V of 1 V (module docstring): the step
    matters. If this fails, the engine changed: measure the module docstring's cases again before a verdict rests on a CPL deck."""
    parts = [("R1", "TINY", 20.0, 10.0), ("R2", "TINY", 45.0, 10.0), ("R3", "TINY", 45.0, YA), ("R4", "TINY", 20.0, YA)]
    ir = board_ir(tmp_path, lib, parts, {"VIC": [("R1", "1"), ("R2", "1")], "AGG": [("R3", "1"), ("R4", "1")]}, (70.0, 30.0))
    stacked(ir, 4)
    for ref, t in (("R1", PinElectricalType.OUTPUT), ("R2", PinElectricalType.INPUT), ("R3", PinElectricalType.OUTPUT), ("R4", PinElectricalType.INPUT)):
        ir.component(ref).pins[0].electrical_type = t
    ir.si = si_of(*default_classes(xtalk_budget_rel=user_requirement(0.05)))
    ir.pcb.tracks = [track("VIC", (20.0, 10.0), (45.0, 10.0), width=W), track("AGG", (45.0, YA), (20.0, YA), width=W)]
    runner = NgspiceShared()
    r = run(ir, lib, tmp_path, runner)["spice.xtalk.VIC"]
    assert r.tool == runner.engine and r.tool_version == runner.version() and r.tool_version.startswith("ngspice-")
    (deck,) = r.details["decks"]
    assert deck["lines"] == {} and not any(line.startswith("T") for line in deck["deck_text"].splitlines())
    assert r.status is S.PASS and deck["problem"] is None, r.message
    fine, coarse = deck["runs"]
    for x in (fine, coarse):
        assert x["succeeded"] and abs(x["final_aggressor_far_v"] - 1.0) <= X.SETTLE_TOL and abs(x["final_victim_v"]) <= X.SETTLE_TOL
    assert fine["peak_rel"] == pytest.approx(0.02033, rel=0.02) and coarse["peak_rel"] == pytest.approx(0.02022, rel=0.02)
    assert deck["peak_rel"] == max(fine["peak_rel"], coarse["peak_rel"])
    assert sum(1 for e in r.evidence if "rawfile" in e.description and e.content_hash) == 2 and r.artifact_hash == deck_hash_of(r)


def deck_hash_of(r: ValidationResult) -> str:
    return next(e.content_hash for e in r.evidence if e.description.startswith("spice.xtalk deck"))
