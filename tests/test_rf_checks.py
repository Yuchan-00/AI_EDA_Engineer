"""The RF checks: ``si.rf_length`` (the carrier form of the critical-length rule) and a real ``domain.rf.impedance``, plus the design-view rules of the new IR fields.

Every board is synthetic (the fixture library of ``tests/test_routing.py``), so
nothing here needs KiCad or ngspice. What is proved:

* the RF markers (``NetKind.RF``, ``NetClass.rf_frequency_hz``) and the
  frequency (the class's, else the confirmed ``carrier_frequency``
  requirement; ambiguous or unconfirmed is no frequency) are read from the IR
  only;
* ``si.rf_length`` on the F12 case: a 20 mm line at 900 MHz on 2 layers
  (er 4.5, the no-plane bound: lambda_g / 10 = 15.70 mm) is NOT_VERIFIED
  "possibly long", 10 mm is PASS, 25 mm over the 4-layer plane is
  NOT_VERIFIED "needs impedance control" without a target Z0 and a
  NOT_APPLICABLE row handed to si.impedance / domain.rf.impedance with one;
  no stated fraction, no frequency or no copper is NOT_VERIFIED naming it;
  the check exists only on a design with RF nets;
* ``domain.rf.impedance`` is selected only for a design that carries RF,
  PASSes only through a PASSing ``si.impedance`` row of a 50 ohm class over a
  plane, FAILs on a wrong width, and is NOT_VERIFIED - never NOT_APPLICABLE -
  when the RF class's only copper is a neck-down or its nets are unrouted,
  without a plane, without ``ir.si``, a target or a placed board;
* the pinned IR saved by the code before these fields existed
  (``tests/data/ir_before_rf.json``: a simulation with ``at`` / ``tol_abs``
  expectations and an ``si`` block with a target-Z0 class and a
  ``critical_fraction``) keeps its hash, and each new field changes it as
  soon as it holds content; ``calc.recompute`` walks ``Expectation.params``.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ai_eda.ir import (
    CircuitDomain,
    CircuitIR,
    NetKind,
    Reduce,
    Requirement,
    RequirementKind,
    SIConstraints,
    Topology,
    llm_generated,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.tools.calc import derived_values, rc_time_constant, recompute_parameters
from ai_eda.tools.calc.tline import C0, guided_wavelength_mm, propagation_delay, rf_critical_length_mm
from ai_eda.tools.si.rf import RF_LENGTH_CHECK, design_carrier_frequency, has_rf, rf_frequency_for, rf_nets
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.domain import NO_RF, RF_NOT_JUDGED, RFImpedanceValidator
from ai_eda.validation.si import si_results
from tests.test_routing import board_ir, fixture_library
from tests.test_si_checks import NECK_PROV, USER, cls, si_of, stacked, track, u

DATA = Path(__file__).parent / "data"
#: an IR saved by the HEAD 0a2c4c7 code (before Expectation.params / reference_vector, NetClass.rf_frequency_hz and
#: SIConstraints.rf_length_fraction existed) and the content hash that code computed for it. Made with
#: ``git archive 0a2c4c7 | tar -x -C <scratchpad>/head_0a2c4c7`` and that snapshot's code on the path
#: (``PYTHONPATH=<scratchpad>/head_0a2c4c7 python -P make_ir_before_rf.py tests/data/ir_before_rf.json``): an RC
#: low-pass with an ac analysis, an ``at`` + ``tol_abs`` and a ``max`` + ``tol_rel`` expectation, a DEFAULT class
#: promoting to Z50 (target 50 ohm) and ``critical_fraction`` 0.5
PRE_RF_IR = DATA / "ir_before_rf.json"
PRE_RF_HASH = "sha256:ee7ead20d3d741c07260e9dbec72c70f97f71e6f910b6b0bfcf812fc892fc3ae"
PRE_STACKUP_HASH = "sha256:6e617ab6b64947a86898e99795b8142a9112bef5b28f2362e3ecf75eef017487"
PRE_SILK_HASH = "sha256:93d701d3cbfc42e871c5eca2e3145d472b91cd1d4c0c82e26cf51698963eb536"
F_900 = 900e6


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def rf_board(tmp: Path, lib, layers: int | None, length: float, *, kind: NetKind = NetKind.RF, width: float = 0.4, routed: bool = True) -> CircuitIR:
    """``ANT`` between two pads ``length`` mm apart (kind ``rf`` unless ``kind`` says), a ground pair, the generic stack of ``layers``."""
    parts = [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", 5.0 + length, 5.0), ("R3", "PAD1", 5.0, 15.0), ("R4", "PAD1", 15.0, 15.0)]
    ir = board_ir(tmp, lib, parts, {"ANT": [("R1", "1"), ("R2", "1")], "GND": [("R3", "1"), ("R4", "1")]}, (max(length, 10.0) + 10.0, 20.0))
    ir.nets[0].kind, ir.nets[1].kind = kind, NetKind.GROUND
    stacked(ir, layers)
    if routed:
        ir.pcb.tracks = [track("ANT", (5.0, 5.0), (5.0 + length, 5.0), width=width), track("GND", (5.0, 15.0), (15.0, 15.0))]
    return ir


def rf_si(*classes, fraction: float | None = 0.1) -> SIConstraints:
    si = si_of(*classes)
    return si.model_copy(update={"rf_length_fraction": None if fraction is None else u(fraction)})


def rf_class(name: str = "RF", *, target: bool = False, f: float | None = F_900, **kw):
    z = {"target_z0_ohm": u(50.0, "ohm"), "z0_tol_rel": u(0.1)} if target else {}
    return cls(name, nets=["ANT"], rf_frequency_hz=None if f is None else u(f, "Hz"), **z, **kw)


def carrier(value, rid: str = "req.carrier", key: str = "carrier_frequency", confirmed: bool = True) -> Requirement:
    v = user_requirement(value, "Hz") if confirmed else llm_generated(value, "Hz")
    return Requirement(id=rid, kind=RequirementKind.EXPLICIT, key=key, text=f"{key} {value}", value=v)


def by_id(results) -> dict:
    return {r.check_id: r for r in results}


# --------------------------------------------------------------------------- the helpers


def test_guided_wavelength_and_the_rf_critical_length():
    t_pd = propagation_delay(4.5)  # the no-plane bound sqrt(er) / c0 on the generic FR-4
    assert guided_wavelength_mm(F_900, t_pd) == pytest.approx(1000.0 * C0 / (F_900 * math.sqrt(4.5)), rel=1e-12)
    assert rf_critical_length_mm(F_900, t_pd, 0.1) == pytest.approx(15.7027, abs=1e-4)
    for args, match in (((0.0, t_pd, 0.1), "f must be a positive"), ((F_900, 0.0, 0.1), "t_pd must be a positive"), ((F_900, t_pd, 0.0), r"fraction must be in \(0, 1\]"),
                        ((F_900, t_pd, 1.5), r"fraction must be in \(0, 1\]")):
        with pytest.raises(ValueError, match=match):
            rf_critical_length_mm(*args)


def test_the_si_model_takes_the_rf_fields_with_their_units_and_ranges():
    si = rf_si(rf_class())
    ids = dict(si.traced_items())
    assert ids["si.rf_length_fraction"].value == 0.1 and ids["si.net_classes[RF].rf_frequency_hz"].value == F_900
    assert si.lookup("si.net_classes[RF].rf_frequency_hz").unit == "Hz"
    with pytest.raises(ValueError, match="rf_frequency_hz must carry unit 'Hz'"):
        cls("X", rf_frequency_hz=u(900.0, "MHz"))
    with pytest.raises(ValueError, match="rf_frequency_hz must be > 0"):
        cls("X", rf_frequency_hz=u(0.0, "Hz"))
    with pytest.raises(ValueError, match=r"si.rf_length_fraction must be in \(0, 1\]"):
        SIConstraints(rf_length_fraction=u(2.0), provenance=USER)
    with pytest.raises(ValueError, match="no unit"):
        SIConstraints(rf_length_fraction=u(0.1, "%"), provenance=USER)


def test_rf_nets_and_their_frequency_come_from_the_ir_only(tmp_path: Path, lib):
    plain = rf_board(tmp_path, lib, 4, 20.0, kind=NetKind.SIGNAL)
    assert rf_nets(plain) == [] and not has_rf(plain)
    plain.topology = Topology(name="t", domains=[CircuitDomain.RF], provenance=USER)
    assert rf_nets(plain) == [] and has_rf(plain)
    marked = rf_board(tmp_path / "m", lib, 4, 20.0, kind=NetKind.SIGNAL)
    marked.si = rf_si(rf_class())
    assert rf_nets(marked) == ["ANT"] and rf_frequency_for(marked, "ANT") == (F_900, "si.net_classes[RF].rf_frequency_hz (user_requirement)")
    kind_only = rf_board(tmp_path / "k", lib, 4, 20.0)
    assert rf_nets(kind_only) == ["ANT"]
    f, why = rf_frequency_for(kind_only, "ANT")
    assert f is None and why.startswith("no frequency: neither ANT is in no net class that states rf_frequency_hz nor is a carrier_frequency requirement confirmed")
    kind_only.requirements.requirements.append(carrier(447.0125e6))
    assert rf_frequency_for(kind_only, "ANT") == (447.0125e6, "the confirmed carrier_frequency (req.carrier)")
    assert design_carrier_frequency(kind_only)[0].value == 447.0125e6
    kind_only.requirements.requirements.append(carrier(446.0e6, rid="req.carrier2"))
    t, why = design_carrier_frequency(kind_only)
    assert t is None and why == "carrier_frequency is ambiguous: req.carrier says 447012500 Hz, req.carrier2 says 446000000 Hz"
    unconfirmed = rf_board(tmp_path / "u", lib, 4, 20.0)
    unconfirmed.requirements.requirements.append(carrier(447.0125e6, confirmed=False))
    t, why = design_carrier_frequency(unconfirmed)
    assert t is None and why.startswith("carrier_frequency is not usable: req.carrier: value is llm_generated")
    typed = rf_board(tmp_path / "t", lib, 4, 20.0)
    typed.requirements.requirements.append(Requirement(id="req.c", kind=RequirementKind.EXPLICIT, key="carrier_frequency", text="x",
                                                       value=user_requirement("447.0125 MHz")))
    assert design_carrier_frequency(typed)[0].value == pytest.approx(447.0125e6)
    # the class's own frequency comes first
    both = rf_board(tmp_path / "b", lib, 4, 20.0)
    both.si = rf_si(rf_class(f=433.92e6))
    both.requirements.requirements.append(carrier(447.0125e6))
    assert rf_frequency_for(both, "ANT")[0] == 433.92e6


# --------------------------------------------------------------------------- si.rf_length


def test_si_rf_length_on_the_f12_case(tmp_path: Path, lib):
    """900 MHz: 20 mm on 2 layers is possibly long (bound), 10 mm short, 25 mm over the plane needs impedance control without a target."""
    two20 = rf_board(tmp_path / "a", lib, 2, 20.0)
    two20.si = rf_si(rf_class())
    r = by_id(si_results(two20, lib))[RF_LENGTH_CHECK]
    row = r.details["nets"][0]
    assert r.status is S.NOT_VERIFIED and row["l_crit_mm"] == pytest.approx(15.7027, abs=1e-3) and row["bound"]
    assert "possibly long at 900 MHz (0.1 lambda_g rule" in r.message and "impedance is undefined without a reference plane" in r.message
    assert r.details["fraction"] == 0.1 and "not DRC" in r.message and r.tool == "si"
    two10 = rf_board(tmp_path / "b", lib, 2, 10.0)
    two10.si = rf_si(rf_class())
    r = by_id(si_results(two10, lib))[RF_LENGTH_CHECK]
    assert r.status is S.PASS and "electrically short at 900 MHz (0.1 lambda_g rule" in r.details["nets"][0]["reason"] and "no-plane upper bound" in r.details["nets"][0]["reason"]
    four25 = rf_board(tmp_path / "c", lib, 4, 25.0)
    four25.si = rf_si(rf_class())
    r = by_id(si_results(four25, lib))[RF_LENGTH_CHECK]
    row = r.details["nets"][0]
    assert r.status is S.NOT_VERIFIED and not row["bound"] and row["l_crit_mm"] < 25.0
    assert "needs impedance control: class RF carries RF at 900 MHz but states no target_z0_ohm" in r.message
    four25.si = rf_si(rf_class(target=True))
    r = by_id(si_results(four25, lib))[RF_LENGTH_CHECK]
    assert r.status is S.NOT_APPLICABLE and "electrically long at 900 MHz: judged by si.impedance.RF / domain.rf.impedance" in r.message
    # the same line and t_pd as the rise-time rule (a class with a driver edge): one delay per net
    four25.si = rf_si(rf_class(target=True, t_rise_s=u(1e-9, "s")))
    r = by_id(si_results(four25, lib))[RF_LENGTH_CHECK]
    crit = by_id(si_results(four25, lib))["si.critical_length"]
    ant = next(n for n in crit.details["nets"] if n["net"] == "ANT")
    assert ant["t_pd_ps_per_mm"] == r.details["nets"][0]["t_pd_ps_per_mm"] and ant["length_mm"] == r.details["nets"][0]["length_mm"]


def test_si_rf_length_names_what_it_cannot_judge(tmp_path: Path, lib):
    ir = rf_board(tmp_path / "a", lib, 4, 20.0)
    ir.si = rf_si(rf_class(), fraction=None)
    r = by_id(si_results(ir, lib))[RF_LENGTH_CHECK]
    assert r.status is S.NOT_VERIFIED and "rf_length_fraction is not stated" in r.message
    nof = rf_board(tmp_path / "b", lib, 4, 20.0)
    nof.si = rf_si(rf_class(f=None))  # the class lists ANT (kind rf) but states no frequency, and no carrier is confirmed
    r = by_id(si_results(nof, lib))[RF_LENGTH_CHECK]
    assert r.status is S.NOT_VERIFIED and "no frequency: neither class RF states rf_frequency_hz nor is a carrier_frequency requirement confirmed" in r.message
    nof.requirements.requirements.append(carrier(F_900))
    assert by_id(si_results(nof, lib))[RF_LENGTH_CHECK].details["nets"][0]["f_source"] == "the confirmed carrier_frequency (req.carrier)"
    bare = rf_board(tmp_path / "c", lib, 4, 20.0, routed=False)
    bare.si = rf_si(rf_class())
    r = by_id(si_results(bare, lib))[RF_LENGTH_CHECK]
    assert r.status is S.NOT_VERIFIED and "ANT: no routed copper" in r.message
    # no RF net: no si.rf_length at all (older designs keep their results)
    plain = rf_board(tmp_path / "d", lib, 4, 20.0, kind=NetKind.SIGNAL)
    plain.si = rf_si(cls("DEFAULT", default=True))
    assert RF_LENGTH_CHECK not in by_id(si_results(plain, lib))


# --------------------------------------------------------------------------- domain.rf.impedance


def _rf_impedance(ir: CircuitIR, tmp_path: Path, lib) -> object:
    return RFImpedanceValidator().validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))[0]


def test_domain_rf_impedance_is_selected_only_for_a_design_that_carries_rf(tmp_path: Path, lib):
    plain = rf_board(tmp_path, lib, 4, 20.0, kind=NetKind.SIGNAL)
    assert "domain.rf.impedance" not in [v.id for v in default_registry.select(plain)]
    r = _rf_impedance(plain, tmp_path, lib)
    assert r.status is S.NOT_APPLICABLE and r.message.startswith(NO_RF)
    plain.topology = Topology(name="t", domains=[CircuitDomain.RF], provenance=USER)
    assert "domain.rf.impedance" in [v.id for v in default_registry.select(plain)]
    r = _rf_impedance(plain, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "which nets carry RF is not stated" in r.message
    marked = rf_board(tmp_path / "m", lib, 4, 20.0)  # a net of kind rf, no topology at all
    assert "domain.rf.impedance" in [v.id for v in default_registry.select(marked)]


def test_domain_rf_impedance_passes_only_through_a_passing_si_impedance_row(tmp_path: Path, lib):
    ir = rf_board(tmp_path / "a", lib, 4, 25.0, width=0.35)
    ir.si = rf_si(rf_class(target=True))
    r = _rf_impedance(ir, tmp_path, lib)
    assert r.status is S.PASS, r.message
    assert r.tool == "domain.rf" and r.tool_version == "0.1" and r.details["kind"] == "ir_geometry+calculators" and r.details["not_judged"] == RF_NOT_JUDGED
    assert r.details["rf_nets"] == ["ANT"] and r.details["classes"]["RF"]["check"] == "si.impedance.RF" and r.details["classes"]["RF"]["copied_status"] == "PASS"
    assert "not judged: matching networks, S-parameters, the antenna" in r.message and r.message.endswith("not DRC; the fab's measured impedance is the only real one)")
    assert r.message.count("not DRC") == 1
    wide = rf_board(tmp_path / "b", lib, 4, 25.0, width=0.8)
    wide.si = ir.si
    r = _rf_impedance(wide, tmp_path, lib)
    assert r.status is S.FAIL and "outside 50 ohm +/- 10%" in r.message and r.details["repair"] == "human"


def test_domain_rf_impedance_is_never_not_applicable_on_a_design_with_rf(tmp_path: Path, lib):
    si = rf_si(rf_class(target=True))
    neck = rf_board(tmp_path / "a", lib, 4, 25.0)
    neck.si = si
    neck.pcb.tracks = [track("ANT", (5.0, 5.0), (30.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV}), neck.pcb.tracks[1]]
    r = _rf_impedance(neck, tmp_path, None)  # without the library the recorded neck-down is not placed, so all ANT copper is a neck-down
    assert r.details["classes"]["RF"]["copied_status"] == "NOT_APPLICABLE"
    assert r.status is S.NOT_VERIFIED and "no judged segment of the RF nets: ANT F.Cu 0.25 mm: neck-down recorded by the router" in r.message
    unrouted = rf_board(tmp_path / "b", lib, 4, 25.0, routed=False)
    unrouted.si = si
    r = _rf_impedance(unrouted, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "no routed copper" in r.message
    two = rf_board(tmp_path / "c", lib, 2, 25.0, width=0.35)
    two.si = si
    r = _rf_impedance(two, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "impedance is undefined without a reference plane - use pcb_layers=4 or add a plane" in r.message
    no_target = rf_board(tmp_path / "d", lib, 4, 25.0, width=0.35)
    no_target.si = rf_si(rf_class())
    r = _rf_impedance(no_target, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "class RF states no target_z0_ohm (its RF nets: ANT)" in r.message
    no_si = rf_board(tmp_path / "e", lib, 4, 25.0)
    r = _rf_impedance(no_si, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and r.message.startswith("no ir.si: no net class states a target Z0 for the RF nets ANT")
    unplaced = rf_board(tmp_path / "f", lib, 4, 25.0)
    unplaced.si, unplaced.pcb = si, None
    r = _rf_impedance(unplaced, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and r.message.startswith("no placed board")
    classless = rf_board(tmp_path / "g", lib, 4, 25.0)
    classless.si = rf_si(cls("OTHER", nets=["GND"]))
    r = _rf_impedance(classless, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "ANT: in no net class (no default class), so no target Z0 is stated" in r.message


def _mixed_board(tmp: Path, lib) -> CircuitIR:
    """ANT (kind rf) and CLK (a signal net) in ONE class Z50 (target 50 ohm +/- 10 %) on a 4-layer board."""
    parts = [("R1", "PAD1", 5.0, 5.0), ("R2", "PAD1", 30.0, 5.0), ("R3", "PAD1", 5.0, 15.0), ("R4", "PAD1", 15.0, 15.0),
             ("R5", "PAD1", 5.0, 10.0), ("R6", "PAD1", 30.0, 10.0)]
    nets = {"ANT": [("R1", "1"), ("R2", "1")], "GND": [("R3", "1"), ("R4", "1")], "CLK": [("R5", "1"), ("R6", "1")]}
    ir = board_ir(tmp, lib, parts, nets, (40.0, 20.0))
    ir.nets[0].kind, ir.nets[1].kind = NetKind.RF, NetKind.GROUND
    stacked(ir, 4)
    ir.si = si_of(cls("Z50", nets=["ANT", "CLK"], target_z0_ohm=u(50.0, "ohm"), z0_tol_rel=u(0.1)))
    return ir


def test_a_non_rf_member_never_decides_the_rf_impedance_verdict(tmp_path: Path, lib):
    """Regression: a class holding ANT (rf) and CLK copied CLK's PASS although no ANT segment was judged."""
    gnd = track("GND", (5.0, 15.0), (15.0, 15.0))
    # (a) ANT's only copper is a neck-down (no library: the recorded neck-down is not placed), CLK passes
    a = _mixed_board(tmp_path / "a", lib)
    a.pcb.tracks = [track("ANT", (5.0, 5.0), (30.0, 5.0), width=0.25).model_copy(update={"provenance": NECK_PROV}),
                    track("CLK", (5.0, 10.0), (30.0, 10.0), width=0.35), gnd]
    r = _rf_impedance(a, tmp_path, None)
    z50 = r.details["classes"]["Z50"]
    assert z50["copied_status"] == "PASS" and r.status is S.NOT_VERIFIED, r.message  # formerly PASS
    assert "no judged segment of the RF nets: ANT F.Cu 0.25 mm" in r.message and "CLK" not in r.message.split("; not judged:")[0]
    assert [row["net"] for row in z50["segments"]] == ["ANT"] and [row["net"] for row in z50["other_members"]] == ["CLK"]
    # (b) ANT has one pad and no copper, CLK passes
    b = _mixed_board(tmp_path / "b", lib)
    b.nets[0].pins = b.nets[0].pins[:1]
    b.pcb.tracks = [track("CLK", (5.0, 10.0), (30.0, 10.0), width=0.35), gnd]
    r = _rf_impedance(b, tmp_path, lib)
    assert r.status is S.NOT_VERIFIED and "ANT: fewer than two pads" in r.message, r.message  # formerly PASS
    # (c) ANT passes, CLK fails: the RF check PASSes on ANT, si.impedance.Z50 still FAILs on CLK
    c = _mixed_board(tmp_path / "c", lib)
    c.pcb.tracks = [track("ANT", (5.0, 5.0), (30.0, 5.0), width=0.35), track("CLK", (5.0, 10.0), (30.0, 10.0), width=0.8), gnd]
    r = _rf_impedance(c, tmp_path, lib)
    assert r.status is S.PASS and "segment(s) of the RF net(s) ANT within 50 ohm" in r.message and "repair" not in r.details, r.message
    assert r.details["classes"]["Z50"]["copied_status"] == "FAIL" and r.details["classes"]["Z50"]["other_members"][0]["status"] == "FAIL"
    si = by_id(si_results(c, lib))
    assert si["si.impedance.Z50"].status is S.FAIL and "CLK" in si["si.impedance.Z50"].message
    # (d) ANT fails, CLK passes: FAIL, naming ANT
    d = _mixed_board(tmp_path / "d", lib)
    d.pcb.tracks = [track("ANT", (5.0, 5.0), (30.0, 5.0), width=0.8), track("CLK", (5.0, 10.0), (30.0, 10.0), width=0.35), gnd]
    r = _rf_impedance(d, tmp_path, lib)
    assert r.status is S.FAIL and r.message.startswith("class Z50: FAIL - ANT F.Cu 0.8 mm: Z0") and r.details["repair"] == "human"


# --------------------------------------------------------------------------- the design view


def test_the_pinned_ir_keeps_its_hash_and_every_new_field_is_hashed_once_set(tmp_path: Path):
    ir = CircuitIR.load(PRE_RF_IR)
    assert ir.content_hash() == PRE_RF_HASH
    assert CircuitIR.load(ir.save(tmp_path / "again.json")).content_hash() == PRE_RF_HASH
    for name, expected in (("ir_before_stackup.json", PRE_STACKUP_HASH), ("ir_before_silkscreen.json", PRE_SILK_HASH)):
        assert CircuitIR.load(DATA / name).content_hash() == expected
    view = ir.design_dict()
    assert all("params" not in e and "reference_vector" not in e for e in view["simulation"]["expectations"])
    assert "rf_length_fraction" not in view["si"] and all("rf_frequency_hz" not in c for c in view["si"]["net_classes"])
    saved = design_data(ir.simulation.expectations[0])
    assert "params" not in saved and "reference_vector" not in saved
    assert "params" in ir.simulation.expectations[0].model_dump() and "rf_frequency_hz" in ir.si.net_classes[0].model_dump()  # the file keeps them

    def changed(mutate) -> str:
        other = copy.deepcopy(ir)
        mutate(other)
        return other.content_hash()

    hashes = {
        "rf_frequency_hz": changed(lambda x: setattr(x.si.net_classes[1], "rf_frequency_hz", u(F_900, "Hz"))),
        "rf_length_fraction": changed(lambda x: setattr(x.si, "rf_length_fraction", u(0.1))),
        "params": changed(lambda x: x.simulation.expectations[0].params.update(ref=u(1.0, "V"))),
        "reference_vector": changed(lambda x: setattr(x.simulation.expectations[0], "reference_vector", "v(IN)")),
    }
    assert PRE_RF_HASH not in hashes.values() and len(set(hashes.values())) == 4


def test_recompute_walks_the_expectation_params(tmp_path: Path):
    ir = CircuitIR.load(PRE_RF_IR)
    ir.parameters["r1"] = user_requirement(1591.5494309189535, "ohm")
    ir.parameters["c1"] = user_requirement(1e-7, "F")
    tau = rc_time_constant(ir.parameters["r1"], ir.parameters["c1"], ("r1", "c1"))
    exp = ir.simulation.expectations[1]
    exp.reduce = Reduce.MAX
    exp.params = {"t_stop": tau}
    assert ("simulation.expectations[passband].params[t_stop]", tau) in list(derived_values(ir))
    assert recompute_parameters(ir).status is S.PASS
    exp.params = {"t_stop": tau.model_copy(update={"value": tau.value * 2})}
    bad = recompute_parameters(ir)
    assert bad.status is S.FAIL and any("params[t_stop]" in m for m in bad.details["mismatches"])
