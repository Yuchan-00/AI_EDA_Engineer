"""Signal integrity: the ``ir.si`` model, the class -> router rule mapping, the measurements, the promotion and the ``si.*`` checks.

Every board is synthetic (the fixture library of ``tests/test_routing.py``),
so nothing here needs KiCad or ngspice. What is proved:

* the model refuses what it cannot mean, every traced number has an id the
  recompute resolves, and an IR without SI data keeps its hash (``ir.si`` is
  out of the design view while ``None``);
* the mapping gives a controlled class the ``calc.tline.width_for_z0`` width
  over the plane (rounded up to 0.01 mm), converts delay / skew budgets with
  the largest t_pd, keeps a minimum width only when it is wider than the
  board's, solves a declared pair's gap for its Z_diff, and gives nothing
  where the stackup has no plane (with the reason);
* the critical-length rule measures the routed copper (microstrip t_pd over
  a plane, the sqrt(er)/c0 upper bound without one), promotes only the long
  signal nets of a class that names a controlled class, with ``derived``
  provenance naming the rule and the numbers;
* every check has its PASS / FAIL / NOT_VERIFIED paths - no stackup, no
  plane, a delay only bounded, missing datasheet timing terms.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ai_eda.design.stackup import board_layers, generic_stackup
from ai_eda.ir import (
    CircuitIR,
    DiffPair,
    NetClass,
    NetKind,
    PinElectricalType,
    PinRef,
    Promotion,
    Provenance,
    ProvenanceKind,
    SIConstraints,
    TimingPath,
    Track,
    Via,
    authoritative,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.tools.calc import CALCULATORS, clock_divided, ipc2221_width_for_current, recompute_parameters
from ai_eda.tools.calc.tline import C0, edge_coupled_microstrip, microstrip, solve_microstrip_width
from ai_eda.tools.routing.maze import RoutingParams
from ai_eda.tools.si import PROMOTE_TOOL, critical_rows, line_model, measure_nets, net_rules, promote
from ai_eda.tools.si.rules import round_up
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.si import SHORT_TEXT, si_results
from tests.conftest import DS
from tests.test_routing import board_ir, fixture_library

USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")


def u(value, unit=None):
    return user_requirement(value, unit)


def cls(name: str, **kw) -> NetClass:
    return NetClass(name=name, provenance=USER, **kw)


def driver(**kw) -> dict:
    return {"t_rise_s": u(1e-9, "s"), "r_drive_ohm": u(25.0, "ohm"), "c_load_f": u(5e-12, "F"), "ringing_tol_rel": u(0.15), **kw}


def default_classes(**default_kw) -> list[NetClass]:
    return [cls("DEFAULT", default=True, promote_to="Z50", **{**driver(), **default_kw}),
            cls("Z50", target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1), **driver())]


def si_of(*classes: NetClass, paths: tuple[TimingPath, ...] = (), fraction: float | None = 0.5) -> SIConstraints:
    return SIConstraints(net_classes=list(classes), timing_paths=list(paths), critical_fraction=None if fraction is None else u(fraction), provenance=USER)


def stacked(ir: CircuitIR, layers: int | None) -> CircuitIR:
    if layers is not None:
        stack = generic_stackup(layers, "fixture", confirmed=True, ground_net="GND", power_net="GND")
        ir.pcb.stackup = stack
        ir.pcb.layers = board_layers(stack)
    return ir


def track(net: str, a, b, width: float = 0.4, layer: str = "F.Cu") -> Track:
    return Track(net=net, layer=layer, start=a, end=b, width_mm=width)


def long_board(tmp: Path, lib, layers: int | None = 4, length: float = 90.0) -> CircuitIR:
    """``LONG`` between two THT pads ``length`` (>= 30) mm apart, ``SHORT`` 10 mm, and an 8 mm ground net; 4-layer stack by default.

    ``LONG`` joins a driver pin (R1.1, an output) to a receiver pin (R2.1, an input): the one-line ``spice.si`` deck models it.
    """
    x1 = 5.0 + length
    parts = [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", x1, 5.0), ("R3", "PAD1", 5.0, 15.0), ("R4", "PAD1", 15.0, 15.0), ("R5", "PAD1", 20.0, 15.0), ("R6", "PAD1", 28.0, 15.0)]
    nets = {"LONG": [("R1", "1"), ("R2", "1")], "SHORT": [("R3", "1"), ("R4", "1")], "GND": [("R5", "1"), ("R6", "1")]}
    ir = board_ir(tmp, lib, parts, nets, (x1 + 5.0, 20.0))
    ir.nets[2].kind = NetKind.GROUND
    ir.component("R1").pins[0].electrical_type = PinElectricalType.OUTPUT
    ir.component("R2").pins[0].electrical_type = PinElectricalType.INPUT
    return stacked(ir, layers)


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


# --------------------------------------------------------------------------- the model


def test_the_model_refuses_what_it_cannot_mean():
    def bad(expect: str, **kw):
        with pytest.raises(ValueError, match=expect):
            SIConstraints(provenance=USER, **kw)

    bad("unique", net_classes=[cls("A"), cls("A")])
    bad("at most one default", net_classes=[cls("A", default=True), cls("B", default=True)])
    bad("in two classes", net_classes=[cls("A", nets=["X"]), cls("B", nets=["X"])])
    bad("not a controlled-impedance class", net_classes=[cls("A", promote_to="B"), cls("B")])
    bad("does not promote", net_classes=[cls("A"), cls("B", target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1), promoted=[
        Promotion(net="X", from_class="A", length_mm=1.0, delay_s=1e-12, l_crit_mm=0.5, t_rise_s=1e-9, provenance=USER)])])
    bad("disagree on max_skew_s", net_classes=[cls("A", match_group="G", max_skew_s=u(1e-12, "s")), cls("B", match_group="G", max_skew_s=u(2e-12, "s"))])
    bad(r"critical_fraction must be in \(0, 1\]", critical_fraction=u(1.5))
    for field, value, expect in (("target_z0_ohm", u(50.0, "Ohm"), "unit 'ohm'"), ("t_rise_s", u(1.0, "ns"), "unit 's'"), ("max_length_mm", u(-1.0, "mm"), "> 0"),
                                 ("z0_tol_rel", u(0.1, "%"), "no unit"), ("c_load_f", u("5p", "F"), "must be a number")):
        with pytest.raises(ValueError, match=expect):
            cls("A", **{field: value})
    promo = Promotion(net="X", from_class="A", length_mm=1.0, delay_s=1e-12, l_crit_mm=0.5, t_rise_s=1e-9, provenance=USER)
    z50 = dict(target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1))
    # a declared net may be promoted by its own class, and stays declared there (its budgets still apply)
    ok = SIConstraints(net_classes=[cls("A", nets=["X"], promote_to="Z"), cls("Z", promoted=[promo], **z50)], provenance=USER)
    assert ok.class_of("X").name == "Z" and ok.declared_class_of("X").name == "A"
    bad("declared in B", net_classes=[cls("A", promote_to="Z"), cls("B", nets=["X"]), cls("Z", promoted=[promo], **z50)])
    bad("promoted twice", net_classes=[cls("A", promote_to="Z"), cls("Z", promoted=[promo], **z50), cls("Y", promoted=[promo], **z50)])
    with pytest.raises(ValueError, match="needs z0_tol_rel"):
        cls("A", target_z0_ohm=u(50.0, "ohm"))
    with pytest.raises(ValueError, match="need a target_zdiff_ohm"):
        cls("A", nets=["P", "N"], pairs=[DiffPair(p="P", n="N")])
    with pytest.raises(ValueError, match="not in the class's nets"):
        cls("A", nets=["P"], target_zdiff_ohm=u(100.0, "ohm"), zdiff_tol_rel=u(0.1), pairs=[DiffPair(p="P", n="N")])
    with pytest.raises(ValueError, match="only a controlled-impedance class"):
        cls("A", promoted=[Promotion(net="X", from_class="B", length_mm=1.0, delay_s=1e-12, l_crit_mm=0.5, t_rise_s=1e-9, provenance=USER)])
    with pytest.raises(ValueError, match="not '<ref>.<fact key>'"):
        TimingPath(name="P", clock_net="C", data_nets=["D"], terms_from={"t_su_min_s": "U1"}, provenance=USER)
    with pytest.raises(ValueError, match="not one of"):
        TimingPath(name="P", clock_net="C", data_nets=["D"], terms_from={"t_rise": "U1.t_rise"}, provenance=USER)
    with pytest.raises(ValueError, match="other than the clock"):
        TimingPath(name="P", clock_net="C", data_nets=["C"], provenance=USER)
    with pytest.raises(ValueError, match="finite number >= 0"):
        Promotion(net="X", from_class="A", length_mm=math.inf, delay_s=1e-12, l_crit_mm=0.5, t_rise_s=1e-9, provenance=USER)


def test_class_resolution_ids_and_lookup():
    si = si_of(*default_classes(), cls("XTAL", nets=["X1"], max_length_mm=u(25.0, "mm")))
    assert si.class_of("X1").name == "XTAL" and si.class_of("ANY").name == "DEFAULT" and si.declared_class_of("ANY").name == "DEFAULT"
    ids = dict(si.traced_items())
    assert "si.critical_fraction" in ids and "si.net_classes[Z50].target_z0_ohm" in ids and "si.net_classes[XTAL].max_length_mm" in ids
    assert si.lookup("si.net_classes[XTAL].max_length_mm").value == 25.0 and si.lookup("si.nope") is None and si.lookup("pcb.x") is None
    promoted = si.model_copy(update={"net_classes": [si.net_classes[0], si.net_classes[1].model_copy(update={"promoted": [
        Promotion(net="ANY", from_class="DEFAULT", length_mm=90.0, delay_s=5e-10, l_crit_mm=80.0, t_rise_s=1e-9, provenance=USER)]}), si.net_classes[2]]})
    promoted = SIConstraints.model_validate(promoted.model_dump())
    assert promoted.class_of("ANY").name == "Z50" and promoted.declared_class_of("ANY").name == "DEFAULT" and promoted.promotion_of("ANY").l_crit_mm == 80.0


def test_an_ir_without_si_data_keeps_its_hash_and_si_is_hashed_once_set(tmp_path: Path):
    data = Path(__file__).parent / "data"
    for name, expected in (("ir_before_stackup.json", "sha256:6e617ab6b64947a86898e99795b8142a9112bef5b28f2362e3ecf75eef017487"),):
        ir = CircuitIR.load(data / name)
        assert ir.si is None and "si" not in ir.design_dict() and ir.content_hash() == expected
        saved = ir.save(tmp_path / name)
        assert CircuitIR.load(saved).content_hash() == expected
        ir.si = si_of(*default_classes())
        assert "si" in ir.design_dict() and ir.content_hash() != expected
        again = CircuitIR.load(ir.save(tmp_path / f"si-{name}"))
        assert again.si == ir.si and again.content_hash() == ir.content_hash()


# --------------------------------------------------------------------------- calculators


def test_ipc2221_width_and_the_divided_clock_are_registered_calculators():
    w = ipc2221_width_for_current(u(1.0, "A"), u(10.0, "degC"), u(35.0, "um"), ("i", "dt", "t"))
    # back through the IPC-2221 external-layer formula: I = 0.048 dT^0.44 A^0.725 with A = w t in mil^2
    area = (w.value / 0.0254) * (35.0 / 1000.0 / 0.0254)
    assert 0.048 * 10.0 ** 0.44 * area ** 0.725 == pytest.approx(1.0, rel=1e-12)
    assert w.value == pytest.approx(0.30039, abs=1e-5) and w.unit == "mm" and w.provenance.tool == "calc.ipc2221.width_for_current"
    for args, expect in (((0.0, 10.0, 35.0), "positive"), ((40.0, 10.0, 35.0), "35 A"), ((1.0, 5.0, 35.0), "10..100"), ((1.0, 10.0, 0.0), "thickness")):
        with pytest.raises(ValueError, match=expect):
            ipc2221_width_for_current(u(args[0], "A"), u(args[1], "degC"), u(args[2], "um"))
    f = clock_divided(u(16e6, "Hz"), u(4.0), ("f_clk", "div"))
    assert f.value == 4e6 and f.unit == "Hz" and f.provenance.inputs == {"f": "f_clk", "n": "div"}
    with pytest.raises(ValueError, match="at least 1"):
        clock_divided(u(16e6, "Hz"), u(0.5))
    assert {"calc.ipc2221.width_for_current", "calc.clock.divided"} <= set(CALCULATORS)


def test_recompute_re_derives_the_derived_si_numbers_from_their_ids(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib)
    ir.parameters["i_load"] = u(0.5, "A")
    ir.parameters["dt"] = u(10.0, "degC")
    w = ipc2221_width_for_current(ir.parameters["i_load"], ir.parameters["dt"], ir.pcb.stackup.copper[0].thickness_um, ("i_load", "dt", "pcb.stackup.copper[F.Cu].thickness_um"))
    ir.si = si_of(*default_classes(), cls("POWER", nets=["GND"], min_width_mm=w, power_current_a=ir.parameters["i_load"], power_temp_rise_c=ir.parameters["dt"]))
    res = recompute_parameters(ir)
    assert res.status is S.PASS and "si.net_classes[POWER].min_width_mm" in res.details["parameters"]
    ir.pcb.stackup.copper[0].thickness_um.value = 70.0  # the stack changed under the stored width
    res = recompute_parameters(ir)
    assert res.status is S.FAIL and "si.net_classes[POWER].min_width_mm" in res.message


# --------------------------------------------------------------------------- measurement


def test_measurement_uses_the_microstrip_over_a_plane_and_the_upper_bound_without_one(tmp_path: Path, lib):
    four = long_board(tmp_path / "4", lib, 4)
    four.pcb.tracks = [track("LONG", (5.0, 5.0), (50.0, 5.0)), track("LONG", (50.0, 5.0), (95.0, 5.0), layer="B.Cu")]
    four.pcb.vias = [Via(net="LONG", x_mm=50.0, y_mm=5.0, drill_mm=0.4, diameter_mm=0.8)]
    m = measure_nets(four)["LONG"]
    ms = microstrip(0.4, 0.2, 35.0, 4.5)
    span = four.pcb.stackup.span_mm("F.Cu", "B.Cu")
    assert m.track_length_mm == pytest.approx(90.0) and m.vias == 1 and m.via_length_mm == pytest.approx(span) and m.length_mm == pytest.approx(90.0 + span)
    t_pd = math.sqrt(ms.e_eff) / C0
    assert m.delay_s == pytest.approx(0.09 * t_pd + span / 1000.0 * math.sqrt(4.5) / C0, rel=1e-12) and not m.bound
    assert all(s.model.z0_ohm == pytest.approx(ms.z0_ohm) for s in m.segments)
    two = long_board(tmp_path / "2", lib, 2)
    two.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    m2 = measure_nets(two)["LONG"]
    assert m2.bound and m2.delay_s == pytest.approx(0.09 * math.sqrt(4.5) / C0) and m2.segments[0].model.z0_ohm is None
    assert "impedance is undefined without a reference plane" in m2.segments[0].model.reason
    none = long_board(tmp_path / "0", lib, None)
    none.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    m0 = measure_nets(none)["LONG"]
    assert m0.delay_s is None and m0.problems and "no stackup" in m0.problems[0]
    model, why = line_model(None, "F.Cu", 0.4)
    assert model is None and why == "no stackup"


# --------------------------------------------------------------------------- class -> rule mapping


def test_a_controlled_class_gets_the_width_for_z0_over_the_plane_and_nothing_without_one(tmp_path: Path, lib):
    four = long_board(tmp_path / "4", lib, 4)
    four.si = si_of(*default_classes(), cls("FAST", nets=["LONG"], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    fine = RoutingParams(track_width_mm=0.25)
    rules = net_rules(four, fine)
    exact = solve_microstrip_width(50.0, 0.2, 35.0, 4.5)
    assert exact == pytest.approx(0.346357, abs=1e-6) and round_up(exact) == 0.35
    r = rules.rules["LONG"]
    assert r.net_class == "FAST" and r.width_mm == 0.35 and r.neckdown_width_mm == 0.25 and r.max_length_mm is None
    assert set(rules.rules) == {"LONG"}  # DEFAULT / Z50 constrain nothing, so no other net gets a rule
    assert "calc.tline.width_for_z0.microstrip(50 ohm" in " ".join(rules.classes["FAST"].derivation)
    assert rules.classes["FAST"].exact["width_for_z0_mm"] == pytest.approx(exact)
    # the board width already above the controlled width: no neck-down
    assert net_rules(four, RoutingParams()).rules["LONG"].neckdown_width_mm is None
    two = long_board(tmp_path / "2", lib, 2)
    two.si = four.si
    rules2 = net_rules(two, fine)
    assert rules2.rules == {} and any("impedance is undefined without a reference plane" in n for n in rules2.notes)
    bare = long_board(tmp_path / "0", lib, None)
    bare.si = four.si
    assert net_rules(bare, fine).rules == {} and any("no stackup" in n for n in net_rules(bare, fine).notes)
    # an IR without ir.si: no rule at all
    four.si = None
    assert net_rules(four, fine).rules == {}


def test_budgets_minimum_widths_and_pairs_map_onto_the_router_rule(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(
        *default_classes(),
        cls("BUDGET", nets=["LONG"], max_length_mm=u(120.0, "mm"), max_delay_s=u(500e-12, "s")),
        cls("WIDE", nets=["SHORT"], min_width_mm=u(0.73, "mm")),
        cls("THIN", nets=["GND"], min_width_mm=u(0.1, "mm")),
    )
    rules = net_rules(ir, RoutingParams())
    t_pd = math.sqrt(microstrip(0.4, 0.2, 35.0, 4.5).e_eff) / C0
    assert rules.rules["LONG"].max_length_mm == pytest.approx(500e-12 / t_pd * 1000.0) and rules.rules["LONG"].max_length_mm < 120.0
    assert rules.rules["LONG"].via_length_mm == pytest.approx(round(ir.pcb.stackup.span_mm("F.Cu", "B.Cu"), 6))
    assert rules.rules["SHORT"].width_mm == 0.73 and "GND" not in rules.rules  # a minimum below the board width adds no rule
    assert "is not above the 0.4 mm" in " ".join(rules.classes["THIN"].derivation)
    # without a plane the delay budget is converted with the upper bound sqrt(er)/c0: a shorter length budget
    two = long_board(tmp_path / "2", lib, 2)
    two.si = ir.si
    assert net_rules(two, RoutingParams()).rules["LONG"].max_length_mm == pytest.approx(500e-12 / (math.sqrt(4.5) / C0) * 1000.0)
    # a declared pair: w from the Z0 target, s solved for Z_diff, both on the 0.01 mm step
    pair = long_board(tmp_path / "p", lib, 4)
    pair.si = si_of(*default_classes(), cls("DIFF", nets=["LONG", "SHORT"], pairs=[DiffPair(p="LONG", n="SHORT")], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1),
                                          target_zdiff_ohm=u(90.0, "ohm"), zdiff_tol_rel=u(0.1), pair_uncoupled_max_mm=u(5.0, "mm"), pair_max_skew_s=u(5e-12, "s")))
    pr = net_rules(pair, RoutingParams(track_width_mm=0.25, clearance_mm=0.2))
    a, b = pr.rules["LONG"], pr.rules["SHORT"]
    assert a.pair_partner == "SHORT" and b.pair_partner == "LONG" and a.width_mm == b.width_mm == 0.35 and a.pair_spacing_mm == b.pair_spacing_mm
    assert edge_coupled_microstrip(a.width_mm, a.pair_spacing_mm, 0.2, 35.0, 4.5).z_diff == pytest.approx(90.0, rel=0.02)
    assert a.pair_uncoupled_max_mm == 5.0 and a.pair_max_skew_mm > 0 and a.neckdown_width_mm is None and a.match_group is None
    assert net_rules(pair, RoutingParams(track_width_mm=0.25, clearance_mm=0.2)).signature() == pr.signature()  # deterministic


# --------------------------------------------------------------------------- the critical-length rule and the promotion


def test_the_critical_length_rule_promotes_only_long_signal_nets_with_derived_provenance(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0)), track("GND", (20.0, 15.0), (28.0, 15.0))]
    rows = {r.net: r for r in critical_rows(ir)}
    t_pd = math.sqrt(microstrip(0.4, 0.2, 35.0, 4.5).e_eff) / C0
    l_crit = 0.5 * 1e-9 / t_pd * 1000.0
    assert rows["LONG"].status == "long" and rows["LONG"].l_crit_mm == pytest.approx(l_crit) and 80.0 < l_crit < 90.0
    assert rows["SHORT"].status == "short" and rows["GND"].status == "not_applicable" and "ground" in rows["GND"].reason
    new, promotions, notes = promote(ir)
    assert [p.net for p in promotions] == ["LONG"] and new.class_of("LONG").name == "Z50" and new.promotion_of("LONG").from_class == "DEFAULT"
    p = promotions[0]
    assert p.provenance.kind is ProvenanceKind.DERIVED and p.provenance.tool == PROMOTE_TOOL and "critical-length rule" in p.provenance.note
    assert p.length_mm == pytest.approx(90.0) and p.l_crit_mm == pytest.approx(l_crit, abs=1e-5) and "si.critical_fraction" in p.provenance.derived_from
    assert design_data(promote(ir)[0]) == design_data(new)  # deterministic (up to the clock default of Provenance.created_at)
    promoted_ir = ir.model_copy(update={"si": new})
    assert promote(promoted_ir)[0] is None  # a promoted net is controlled already
    # a class that names no controlled class: noted, not promoted
    ir.si = si_of(cls("DEFAULT", default=True, **driver()), cls("Z50", target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    new, promotions, notes = promote(ir)
    assert new is None and promotions == [] and "names no controlled class" in notes[0]
    # no driver edge, no fraction: the rule cannot judge
    ir.si = si_of(cls("DEFAULT", default=True, promote_to="Z50"), cls("Z50", target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    assert {r.net: r.status for r in critical_rows(ir)}["LONG"] == "not_applicable"
    ir.si = si_of(*default_classes(), fraction=None)
    assert {r.net: r.status for r in critical_rows(ir)}["LONG"] == "unknown"


def test_a_grounded_datasheet_edge_replaces_the_class_choice(tmp_path: Path, lib):
    """The driver's grounded t_rise replaces the class's choice on a net it drives (an output pin there), never on a net it only
    listens to (an input pin: a microcontroller's ~RESET) or is not on."""
    from tests.test_parts_existence import make_part

    ir = long_board(tmp_path, lib, 4)
    drv = make_part("U9")
    drv.electrical["t_rise"] = authoritative(3e-9, DS, "s")
    drv.pins[0].electrical_type = PinElectricalType.OUTPUT
    ir.components.append(drv)
    ir.net("LONG").pins.append(PinRef(component_ref="U9", pin_number="1"))
    ir.si = si_of(*default_classes(driver="U9"))
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    row = {r.net: r for r in critical_rows(ir)}["LONG"]
    assert row.t_rise_s == 3e-9 and row.t_rise_source.startswith("U9.t_rise (grounded datasheet fact)") and row.status == "short"
    drv.pins[0].electrical_type = PinElectricalType.INPUT  # U9 only listens to LONG: its edge is not LONG's
    row = {r.net: r for r in critical_rows(ir)}["LONG"]
    assert row.t_rise_s == 1e-9 and "U9 has no pin that drives LONG, so its t_rise does not apply" in row.t_rise_source
    ir.net("LONG").pins.pop()  # not on the net at all
    assert {r.net: r for r in critical_rows(ir)}["LONG"].t_rise_s == 1e-9
    drv.pins[0].electrical_type = PinElectricalType.OUTPUT
    ir.net("LONG").pins.append(PinRef(component_ref="U9", pin_number="1"))
    drv.electrical["t_rise"] = user_requirement(3e-9, "ns")  # a wrong unit is not read: the class's choice applies
    assert {r.net: r for r in critical_rows(ir)}["LONG"].t_rise_s == 1e-9


# --------------------------------------------------------------------------- the checks


def _results(ir: CircuitIR) -> dict[str, object]:
    return {r.check_id: r for r in si_results(ir)}


def test_the_validator_applies_only_to_a_design_with_si_constraints(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    assert not any(r.check_id.startswith("si.") for r in default_registry.run(ir, ctx))
    ir.si = si_of(*default_classes())
    got = [r for r in default_registry.run(ir, ctx) if r.check_id.startswith("si.")]
    assert got and all(r.tool == "si" and r.tool_version and r.details["kind"] == "ir_geometry+calculators" and "not DRC" in r.message for r in got)


#: the provenance the router gives a track of a net whose rule necks down to 0.25 mm within 1 mm of its pads (0.25 mm grid)
NECK_PROV = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.3", derived_from=[
    "net:LONG", "params:grid=0.25,width=0.4,clearance=0.25", "rule:class=FAST,width=0.35,clearance=0.25,neckdown_width=0.25,neckdown_radius=1.0"])


def test_impedance_checks_pass_fail_and_say_why_they_cannot_judge(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes(), cls("FAST", nets=["LONG"], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    neck = track("LONG", (93.0, 5.0), (95.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV})
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (93.0, 5.0), width=0.35), neck]
    res = _results(ir)
    imp = res["si.impedance.FAST"]
    assert imp.status is S.PASS and "neck-downs not judged: LONG F.Cu 2.000 mm at 0.25 mm" in imp.message

    def reasons(r) -> str:
        return " | ".join(row["reason"] for row in r.details["segments"])

    assert "neck-down recorded by the router (within 1 mm + one 0.25 mm grid step of a pad; its position not checked: no KiCad library)" in reasons(imp)
    with_pads = {r.check_id: r for r in si_results(ir, lib)}["si.impedance.FAST"]  # the library gives R2's pad box: the neck-down is beside it
    assert with_pads.status is S.PASS and "neck-down recorded by the router within 1 mm + one 0.25 mm grid step of the net's pads" in reasons(with_pads)
    far = neck.model_copy(update={"start": (5.0, 5.0), "end": (50.0, 5.0)})
    ir.pcb.tracks = [far, track("LONG", (50.0, 5.0), (95.0, 5.0), width=0.35)]  # a "neck-down" 45 mm long, far from every pad: judged at its width
    imp = {r.check_id: r for r in si_results(ir, lib)}["si.impedance.FAST"]
    assert imp.status is S.FAIL and "the track leaves 1 mm + one 0.25 mm grid step of the net's pads" in imp.message and imp.details["neckdowns"] == []
    # a narrower track the router never necked down (the first pass's board width kept after a refused re-route) is judged: 59.2 ohm FAILs
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0), width=0.25)]
    imp = _results(ir)["si.impedance.FAST"]
    assert imp.status is S.FAIL and "Z0 59.20 ohm outside 50 ohm +/- 10%" in imp.message and "not a neck-down the router recorded" in imp.message
    assert imp.details["neckdowns"] == []
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (93.0, 5.0), width=0.35), neck]
    assert res["si.impedance.DEFAULT"].status is S.NOT_APPLICABLE and res["si.impedance.Z50"].status is S.NOT_APPLICABLE
    ir.pcb.tracks[0] = track("LONG", (5.0, 5.0), (93.0, 5.0), width=0.8)  # wider than the target width: Z0 far below 45 ohm
    imp = _results(ir)["si.impedance.FAST"]
    assert imp.status is S.FAIL and "outside 50 ohm +/- 10%" in imp.message and imp.details["repair"] == "human"
    two = long_board(tmp_path / "2", lib, 2)
    two.si, two.pcb.tracks = ir.si, [track("LONG", (5.0, 5.0), (95.0, 5.0), width=0.35)]
    imp = _results(two)["si.impedance.FAST"]
    assert imp.status is S.NOT_VERIFIED and "impedance is undefined without a reference plane - use pcb_layers=4 or add a plane" in imp.message
    bare = long_board(tmp_path / "0", lib, None)
    bare.si, bare.pcb.tracks = ir.si, two.pcb.tracks
    assert _results(bare)["si.impedance.FAST"].status is S.NOT_VERIFIED and "no stackup" in _results(bare)["si.impedance.FAST"].message


def test_length_delay_width_and_skew_checks(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes(), cls("B", nets=["LONG", "SHORT"], max_length_mm=u(95.0, "mm"), max_delay_s=u(600e-12, "s"), min_width_mm=u(0.4, "mm"),
                                            match_group="G", max_skew_s=u(10e-12, "s")))
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    res = _results(ir)
    assert res["si.length.B"].status is S.PASS and res["si.delay.B"].status is S.PASS and res["si.width.B"].status is S.PASS
    skew = res["si.skew.G"]
    assert skew.status is S.FAIL and "delay spread" in skew.message
    ir.si.net_classes[2].max_length_mm = u(50.0, "mm")
    ir.pcb.tracks[1] = track("SHORT", (5.0, 15.0), (15.0, 15.0), width=0.3)
    res = _results(ir)
    assert res["si.length.B"].status is S.FAIL and "90.000 mm > 50 mm" in res["si.length.B"].message
    assert res["si.width.B"].status is S.FAIL and "SHORT F.Cu 0.3 mm" in res["si.width.B"].message
    # without a plane the delay is an upper bound: a budget it breaks is not a FAIL
    two = long_board(tmp_path / "2", lib, 2)
    two.si = si_of(*default_classes(), cls("B", nets=["LONG"], max_delay_s=u(400e-12, "s")))
    two.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    d = _results(two)["si.delay.B"]
    assert d.status is S.NOT_VERIFIED and "upper bound" in d.message
    two.si.net_classes[2].max_delay_s = u(1e-9, "s")
    assert _results(two)["si.delay.B"].status is S.PASS
    # a via without a stackup: its barrel is unknown
    bare = long_board(tmp_path / "0", lib, None)
    bare.si = si_of(*default_classes(), cls("B", nets=["LONG"], max_length_mm=u(95.0, "mm")))
    bare.pcb.tracks, bare.pcb.vias = [track("LONG", (5.0, 5.0), (95.0, 5.0))], [Via(net="LONG", x_mm=50.0, y_mm=5.0, drill_mm=0.4, diameter_mm=0.8)]
    assert _results(bare)["si.length.B"].status is S.NOT_VERIFIED


def test_the_critical_length_check_hands_long_nets_to_spice_and_names_what_it_cannot_judge(tmp_path: Path, lib):
    four = long_board(tmp_path, lib, 4)
    four.si = si_of(*default_classes())
    four.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    c = _results(four)["si.critical_length"]
    assert c.status is S.PASS and "judged by spice.si: LONG" in c.message and SHORT_TEXT in c.message and c.details["long_over_plane"] == ["LONG"]
    rows = {row["net"]: row for row in c.details["nets"]}
    # the rule found LONG long: its row is not a PASS - spice.si.LONG gives the verdict
    assert rows["LONG"]["status"] == "NOT_APPLICABLE" and "judged by spice.si.LONG, not by this rule" in rows["LONG"]["reason"] and rows["SHORT"]["status"] == "PASS"
    only_long = four.model_copy(deep=True)
    only_long.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0))]
    only_long.nets = [n for n in only_long.nets if n.name != "SHORT"]  # every judged net is long over a plane
    c1 = _results(only_long)["si.critical_length"]
    assert c1.status is S.NOT_APPLICABLE and c1.message.startswith("no net is electrically short: 1 electrically long net(s) over a plane, judged by spice.si: LONG")
    two = long_board(tmp_path / "2", lib, 2)
    two.si, two.pcb.tracks = four.si, four.pcb.tracks
    c = _results(two)["si.critical_length"]
    assert c.status is S.NOT_VERIFIED and "use pcb_layers=4 or add a plane: LONG" in c.message and c.details["long_without_plane"] == ["LONG"]
    two.pcb.tracks = [four.pcb.tracks[1]]  # LONG unrouted
    c = _results(two)["si.critical_length"]
    assert c.status is S.NOT_VERIFIED and "no routed copper" in c.message


def _timing_ir(tmp_path: Path, lib, **path_kw) -> CircuitIR:
    ir = long_board(tmp_path, lib, 4)
    path = TimingPath(name="SPI", clock_net="SHORT", data_nets=["LONG"], f_clk_hz=u(4e6, "Hz"), capture_fraction=u(0.5),
                      terms_from={"t_co_max_s": "R1.t_co", "t_co_min_s": "R1.t_co_min", "t_su_min_s": "R2.t_su", "t_h_min_s": "R2.t_h"}, provenance=USER, **path_kw)
    ir.si = si_of(*default_classes(), paths=(path,))
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    return ir


def test_timing_margins_are_not_verified_naming_every_missing_datasheet_term_and_judged_when_grounded(tmp_path: Path, lib):
    ir = _timing_ir(tmp_path, lib)
    t = _results(ir)["si.timing.SPI"]
    assert t.status is S.NOT_VERIFIED
    for missing in ("R1.t_co (datasheet fact key t_co of R1 not grounded)", "R1.t_co_min", "R2.t_su", "R2.t_h"):
        assert missing in t.message
    r1, r2 = ir.component("R1"), ir.component("R2")
    r1.electrical.update(t_co=authoritative(10e-9, DS, "s"), t_co_min=authoritative(2e-9, DS, "s"))
    r2.electrical.update(t_su=authoritative(20e-9, DS, "s"), t_h=authoritative(10e-9, DS, "s"))
    t = _results(ir)["si.timing.SPI"]
    assert t.status is S.PASS and t.details["period_s"] == pytest.approx(250e-9) and t.details["terms"]["t_su_min_s"].startswith("R2.t_su (grounded")
    setup = t.details["data"][0]["setup_margin_ns"]
    assert setup[0] == pytest.approx(125.0 - 10.0 - 20.0 - measure_nets(ir)["LONG"].delay_s * 1e9, abs=1e-6)
    r2.electrical["t_su"] = authoritative(200e-9, DS, "s")  # longer than half the period: even the best case fails
    t = _results(ir)["si.timing.SPI"]
    assert t.status is S.FAIL and t.details["repair"] == "human"
    r2.electrical["t_su"] = user_requirement(20e-9, "s")  # the user's own number is authoritative too: it counts
    assert _results(ir)["si.timing.SPI"].status is S.PASS
    # a model's unconfirmed value is never read: the term is missing again
    r2.electrical["t_su"] = user_requirement(20e-9, "s").model_copy(update={"provenance": Provenance(kind=ProvenanceKind.LLM_GENERATED, note="model")})
    t = _results(ir)["si.timing.SPI"]
    assert t.status is S.NOT_VERIFIED and "R2.t_su (datasheet fact key t_su of R2 not grounded)" in t.message


def test_a_declared_pair_is_judged_on_its_coupled_section(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    w, s = 0.35, 0.2
    z = edge_coupled_microstrip(w, s, 0.2, 35.0, 4.5).z_diff
    ir.si = si_of(*default_classes(), cls("DIFF", nets=["LONG", "SHORT"], pairs=[DiffPair(p="LONG", n="SHORT")], target_zdiff_ohm=u(round(z), "ohm"), zdiff_tol_rel=u(0.1),
                                            pair_uncoupled_max_mm=u(3.0, "mm"), pair_max_skew_s=u(5e-12, "s")))
    pitch = w + s
    ir.pcb.tracks = [track("LONG", (10.0, 5.0), (60.0, 5.0), width=w), track("LONG", (8.0, 5.0), (10.0, 5.0), width=w),
                     track("SHORT", (10.0, 5.0 + pitch), (60.0, 5.0 + pitch), width=w), track("SHORT", (8.0, 5.0 + pitch), (10.0, 5.0 + pitch), width=w)]
    d = _results(ir)["si.diff.LONG/SHORT"]
    assert d.status is S.PASS and d.details["coupled_mm"] == pytest.approx(52.0)
    ir.pcb.tracks.append(track("SHORT", (60.0, 5.0 + pitch), (60.0, 15.0), width=w))  # 9.45 mm uncoupled, and as much skew
    d = _results(ir)["si.diff.LONG/SHORT"]
    assert d.status is S.FAIL and "SHORT uncoupled 9.450 mm > 3 mm" in d.message and "intra-pair skew" in d.message
    two = long_board(tmp_path / "2", lib, 2)
    two.si, two.pcb.tracks = ir.si, ir.pcb.tracks
    assert _results(two)["si.diff.LONG/SHORT"].status is S.NOT_VERIFIED


def test_checks_are_deterministic_and_never_mutate_the_ir(tmp_path: Path, lib):
    ir = _timing_ir(tmp_path, lib)
    before = ir.content_hash()
    snap = copy.deepcopy(ir)
    a = [r.model_dump(exclude={"timestamp"}) for r in si_results(ir)]
    b = [r.model_dump(exclude={"timestamp"}) for r in si_results(snap)]
    assert a == b and ir.content_hash() == before


# --------------------------------------------------------------------------- the line: the longest pad-to-pad path


def _tree(tmp_path: Path, lib) -> CircuitIR:
    """LONG joins R1 (5, 5), R2 (75, 5) and R7 (60, 17): a 70 mm trunk, and from its middle (40, 5) a branch through a via to B.Cu
    that reaches R7 - 102 mm of track in all, but no pad-to-pad line longer than the 70 mm trunk."""
    ir = long_board(tmp_path, lib, 4, length=70.0)
    tap = ir.component("R2").model_copy(deep=True, update={"ref": "R7"})
    ir.components.append(tap)
    ir.pcb.placements.append(ir.pcb.placement("R2").model_copy(update={"component_ref": "R7", "x_mm": 60.0, "y_mm": 17.0}))
    ir.net("LONG").pins.append(PinRef(component_ref="R7", pin_number="1"))
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (75.0, 5.0)), track("LONG", (40.0, 5.0), (40.0, 10.0)),
                     track("LONG", (40.0, 10.0), (40.0, 17.0), layer="B.Cu"), track("LONG", (40.0, 17.0), (60.0, 17.0), layer="B.Cu"),
                     track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    ir.pcb.vias = [Via(net="LONG", x_mm=40.0, y_mm=10.0, drill_mm=0.4, diameter_mm=0.8)]
    return ir


def test_the_critical_length_rule_measures_the_longest_pad_to_pad_path_not_the_summed_branches(tmp_path: Path, lib):
    ir = _tree(tmp_path, lib)
    m = measure_nets(ir, library=lib)["LONG"]
    barrel = ir.pcb.stackup.span_mm("F.Cu", "B.Cu")
    assert m.length_mm == pytest.approx(102.0 + barrel)  # the whole copper: tracks plus the via barrel
    assert m.path is not None and m.path.ends == ("R1.1", "R2.1") and m.path.length_mm == pytest.approx(70.0) and m.path.vias == 0
    row = {r.net: r for r in critical_rows(ir, library=lib)}["LONG"]
    assert row.status == "short" and row.length_mm == pytest.approx(70.0) and row.total_mm == pytest.approx(102.0 + barrel) and row.l_crit_mm > 70.0
    assert row.ends == ("R1.1", "R2.1") and row.measure == "the longest pad-to-pad path R1.1-R2.1"
    assert promote(ir, library=lib)[0] is None  # a line that exists decides: nothing is promoted
    # without the library the pads are unknown: the 102 mm total is only an upper bound of every path - possibly long, never promoted
    row = {r.net: r for r in critical_rows(ir)}["LONG"]
    assert row.status == "possibly_long" and "no pad-to-pad path was extracted (no KiCad library" in row.reason
    new, promotions, notes = promote(ir)
    assert new is None and promotions == [] and any(n.startswith("LONG not promoted: the whole copper of its 3 pads") for n in notes)
    c = {r.check_id: r for r in si_results(ir)}["si.critical_length"]
    assert c.status is S.NOT_VERIFIED and c.details["possibly_long"] == ["LONG"] and "possibly long" in c.message
    assert {r.check_id: r for r in si_results(ir, lib)}["si.critical_length"].status is S.PASS
    # the trunk split in two at (60, 5) is still one line
    ir.pcb.tracks[0] = track("LONG", (5.0, 5.0), (60.0, 5.0))
    ir.pcb.tracks.append(track("LONG", (60.0, 5.0), (75.0, 5.0), layer="F.Cu"))
    m = measure_nets(ir, library=lib)["LONG"]
    assert m.path.length_mm == pytest.approx(70.0)  # a split track is one line
    ir.pcb.tracks = ir.pcb.tracks[:-1]  # the 60..75 piece gone: R2 is not reached
    assert measure_nets(ir, library=lib)["LONG"].path_problem == "the copper does not join pad(s) R2.1 to the others"


def test_a_copper_loop_leaves_the_path_unextracted(tmp_path: Path, lib):
    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("LONG", (20.0, 5.0), (20.0, 8.0)), track("LONG", (20.0, 8.0), (30.0, 8.0)),
                     track("LONG", (30.0, 8.0), (30.0, 5.0))]
    m = measure_nets(ir, library=lib)["LONG"]
    assert m.path is None and "forms a loop" in m.path_problem
    # a 2-pad net's line is still its whole copper (the rule's documented measure there)
    assert m.line is not None and m.line.how == "two-pad copper" and m.line.length_mm == pytest.approx(m.length_mm)


def test_the_high_speed_domain_stub_now_speaks_only_of_crosstalk(tmp_path: Path, lib):
    from ai_eda.validation.domain import SignalIntegrityValidator

    v = SignalIntegrityValidator()
    (r,) = v.validate(long_board(tmp_path, lib, 4), ValidationContext(workdir=tmp_path, tools={}))
    assert r.status is S.NOT_VERIFIED and r.message.startswith("crosstalk analysis not implemented") and "si.*" in r.message
    assert v.description.startswith("Crosstalk")
