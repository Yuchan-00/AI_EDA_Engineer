"""The RF IR (``ir.rf``), the one-sided expectation ``bound`` and the PCB keep-outs.

Nothing here needs KiCad; one test needs ngspice (``skipif``). What is proved:

* the pinned IR saved by the code before these fields existed
  (``tests/data/ir_before_kr447.json``: a simulation with ``at`` / ``db_at``
  / ``tol_abs`` / ``tol_rel`` expectations, an ``si`` block and a placed,
  routed ``pcb`` with a zone) keeps its hash, and ``CircuitIR.rf``,
  ``Expectation.bound`` and ``PCBDesign.keepouts`` each change it as soon as
  they hold content (the older pinned fixtures keep theirs too);
* the RF models refuse what they cannot mean: a port without its z0, a
  probe with a load, an expectation with both or neither of ``tol_abs`` /
  ``bound``, any ``tol_rel``, a bound that claims a requirement, an S21 to a
  probe, a state or member the network lacks, a non-ac sweep, a plan row
  without what its kind needs, a ref in two blocks ...; ``traced_items`` /
  ``lookup`` name every traced number, and ``calc.recompute`` re-derives a
  fixture nominal under ``rf.networks[...]``;
* ``Keepout``: exactly one of rect / polygon in mm, copper layers only, an
  exception only for an item it forbids; unique ids; a keep-out is a layout
  item with a provenance;
* ``judge`` with a bound: PASS on the passing side (equal passes), FAIL on
  the failing side, the whole interpolation / bias bracket judged
  (UNRESOLVED when it straddles), ``(status, None, measured - nominal)``;
  an ``RFExpectation`` is judged by the same function;
* the SPICE compiler refuses a bound with a tolerance or a requirement and
  lets a one-sided dB level through without ``tol_abs``; the stage records
  ``details["bound"]`` and a signed deviation, and never retires a
  ``spice.rf.*`` result (the fixture runner's), while a design expectation
  called ``rf`` is still retired.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from ai_eda.compilers import CompileContext, SpiceNetlistCompiler
from ai_eda.compilers.spice import build
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    ArtifactKind,
    CircuitIR,
    Keepout,
    LabItem,
    PCBDesign,
    PlanLine,
    Provenance,
    ProvenanceKind,
    RailBudget,
    RFBlock,
    RFDesign,
    RFExpectation,
    RFNetwork,
    RFPort,
    RFProbe,
    RFRegion,
    RFState,
    SpiceBinding,
    SpiceDevice,
    ValidationResult,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.provenance import design_data
from ai_eda.tools.calc import derived_values, rc_lowpass_phase_deg, rc_time_constant, recompute_parameters
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice.runner import Interpolation
from ai_eda.tools.spice.stage import RF_CHECK_PREFIX, judge, retire_expectation_results, run_spice_for

DATA = Path(__file__).parent / "data"
#: an IR saved by the HEAD 42b0e53 code (before CircuitIR.rf, Expectation.bound and PCBDesign.keepouts existed) and the
#: content hash that code computed for it. Made with ``git archive 42b0e53 | tar -x -C <scratchpad>/kr447-p1/head_42b0e53``
#: and that snapshot's code on the path
#: (``PYTHONPATH=<scratchpad>/kr447-p1/head_42b0e53 python -P make_ir_before_kr447.py tests/data/ir_before_kr447.json``):
#: an RC low-pass with an ac analysis, an ``at`` + ``tol_abs``, two ``db_at`` + ``tol_abs`` (one against ``reference_vector``,
#: one against ``params["ref"]``) and a ``max`` + ``tol_rel`` expectation, a DEFAULT + RF50 ``si`` block with
#: ``rf_length_fraction``, and a placed board with a track, a via and a zone
PRE_KR447_IR = DATA / "ir_before_kr447.json"
PRE_KR447_HASH = "sha256:30386ed479cb5186d344ad374d321c0405ad2248bfad69f3458a288e80d2364d"
OLDER_PINNED = {
    "ir_before_rf.json": "sha256:ee7ead20d3d741c07260e9dbec72c70f97f71e6f910b6b0bfcf812fc892fc3ae",
    "ir_before_stackup.json": "sha256:6e617ab6b64947a86898e99795b8142a9112bef5b28f2362e3ecf75eef017487",
    "ir_before_silkscreen.json": "sha256:93d701d3cbfc42e871c5eca2e3145d472b91cd1d4c0c82e26cf51698963eb536",
}
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fixture")
F_C = 447.5625e6


def u(value, unit=None):
    return user_requirement(value, unit)


def port(name: str, net: str, *, kind: str = "port", z0: float | None = 50.0, **kw) -> RFPort:
    return RFPort(name=name, net=net, kind=kind, z0_ohm=None if z0 is None or kind != "port" else u(z0, "ohm"), **kw)


def ac(id: str = "ac1", fstart: float = 100e6, fstop: float = 2e9) -> AnalysisSpec:
    return AnalysisSpec(id=id, kind=SpiceAnalysis.AC, params={"variation": u("dec"), "points": u(200), "fstart": u(fstart, "Hz"), "fstop": u(fstop, "Hz")}, provenance=USER)


def rfexp(id: str = "s21_fc", quantity: str = "s21_db", drive: str = "P1", to: str = "P2", *, at: float = F_C, nominal: float = -1.5,
          tol: float | None = None, bound: str | None = "at_least", ref_at: float | None = None, **kw) -> RFExpectation:
    unit = "deg" if quantity == "phase21_deg" else "dB"
    return RFExpectation(id=id, quantity=quantity, drive=drive, to=to, at=u(at, "Hz"), ref_at=None if ref_at is None else u(ref_at, "Hz"),
                         nominal=u(nominal, unit), tol_abs=None if tol is None else u(tol, unit), bound=bound, **kw)


def lpf(**over) -> RFNetwork:
    """A 3-element LPF fixture: C1 / L1 / C2 between P1 (IN) and P2 (OUT), L1 at Q 40, a probe on MID."""
    fields = dict(
        id="lpf", members=["C1", "L1", "C2"], loss_q={"L1": u(40.0)}, q_ref_hz=u(F_C, "Hz"),
        ports=[port("P1", "IN"), port("P2", "OUT"), port("MID", "MID", kind="probe", z0=None)], sweep=[ac()],
        expectations=[
            rfexp(),
            rfexp("s11_fc", "s11_db", "P1", "P1", nominal=-15.0, bound="at_most"),
            rfexp("rej_2f", "rel_s21_db", at=2 * F_C, ref_at=F_C, nominal=-45.0, bound="at_most"),
            rfexp("mid_phase", "phase21_deg", "P1", "MID", nominal=-90.0, tol=5.0, bound=None),
        ],
        probes=[RFProbe(id="p3f", quantity="rel_s21_db", drive="P1", to="P2", at=u(3 * F_C, "Hz"), ref_at=u(F_C, "Hz"))],
    )
    fields.update(over)
    return RFNetwork(**fields)


def keepout(**over) -> Keepout:
    fields = dict(id="ant_band", layers=["*.Cu"], rect=u([0.0, 0.0, 60.0, 6.0], "mm"), forbids=["tracks", "footprints"],
                  allowed_refs=["ANT1"], allowed_nets=["ANT_FEED"], reason="antenna band: no parts, no tracks", provenance=USER)
    fields.update(over)
    return Keepout(**fields)


# --------------------------------------------------------------------------- the design view


def test_the_pinned_ir_keeps_its_hash_and_every_new_field_is_hashed_once_set(tmp_path: Path):
    ir = CircuitIR.load(PRE_KR447_IR)
    assert ir.content_hash() == PRE_KR447_HASH
    assert CircuitIR.load(ir.save(tmp_path / "again.json")).content_hash() == PRE_KR447_HASH
    for name, expected in OLDER_PINNED.items():
        assert CircuitIR.load(DATA / name).content_hash() == expected, name
    view = ir.design_dict()
    assert "rf" not in view and "keepouts" not in view["pcb"]
    assert all("bound" not in e for e in view["simulation"]["expectations"])
    assert "bound" not in design_data(ir.simulation.expectations[1])
    dumped = ir.model_dump()
    assert dumped["rf"] is None and dumped["pcb"]["keepouts"] == [] and dumped["simulation"]["expectations"][1]["bound"] is None  # the file keeps them

    def changed(mutate) -> str:
        other = copy.deepcopy(ir)
        mutate(other)
        return other.content_hash()

    hashes = {
        "rf": changed(lambda x: setattr(x, "rf", RFDesign(networks=[lpf()]))),
        "bound": changed(lambda x: setattr(x.simulation.expectations[1], "bound", "at_most")),
        "keepouts": changed(lambda x: x.pcb.keepouts.append(keepout())),
    }
    assert PRE_KR447_HASH not in hashes.values() and len(set(hashes.values())) == 3


def test_an_ir_with_rf_content_round_trips_through_its_file(tmp_path: Path):
    ir = CircuitIR.load(PRE_KR447_IR)
    ir.rf = RFDesign(
        blocks=[RFBlock(id="trx", title="T/R switch and LPF", refs=["C1", "L1", "C2", "SH1"], chain=["C1", "L1", "C2"], shield_ref="SH1",
                        region=RFRegion(x=u(0.0, "mm"), y=u(6.0, "mm"), w=u(30.0, "mm"), h=u(26.0, "mm")),
                        ports=[port("ANT", "ANT_PORT"), port("V5", "TX_5V", kind="rail", z0=None, voltage_v=u(5.0, "V"))])],
        networks=[lpf(block="trx")],
        frequency_plan=[PlanLine(id="image", kind="response", f_hz=u(404.7625e6, "Hz"), points_to=["spice.rf.fe_bpf3", "rf.lab.image"])],
        lab_items=[LabItem(id="obw", block="trx", what="occupied bandwidth with the prescribed test modulation", instruments=["spectrum analyser"],
                           reason="the test modulation is not grounded")],
        rails=[RailBudget(rail="TX_5V", regulator_ref="U102", v_out=u(5.0, "V"), i_min=u(0.287, "A"), i_max=u(0.441, "A"), dropout_v=u(1.2, "V"))],
        model_values=["model.l_q"], profile_keys=["kr447.max_power"],
    )
    ir.pcb.keepouts = [keepout()]
    ir.simulation.expectations[2].bound, ir.simulation.expectations[2].tol_abs = "at_least", None
    saved = CircuitIR.load(ir.save(tmp_path / "ir.json"))
    assert saved.content_hash() == ir.content_hash() and saved.model_dump(mode="json") == ir.model_dump(mode="json")
    assert saved.rf.network("lpf").port("MID").kind == "probe" and saved.rf.block_of("L1").id == "trx" and saved.pcb.keepout("ant_band").allowed_refs == ["ANT1"]


# --------------------------------------------------------------------------- the RF models


def test_rf_ports_carry_what_their_kind_means():
    assert port("P1", "IN").z0_ohm.value == 50.0 and port("P1", "IN").reference_net == "GND"
    for kw, match in (
        (dict(name="P1", net="IN", kind="port"), "a 'port' needs z0_ohm"),
        (dict(name="Q", net="Q", kind="probe", z0_ohm=u(50.0, "ohm")), "a probe has no load"),
        (dict(name="V", net="V_TX", kind="rail", z0_ohm=u(50.0, "ohm")), "only a 'port' has a reference impedance"),
        (dict(name="P1", net="IN", kind="port", z0_ohm=u(50.0, "ohm"), voltage_v=u(5.0, "V")), "voltage_v belongs to a 'rail' or 'control'"),
        (dict(name="P1", net="IN", kind="port", z0_ohm=u(50.0, "Ω")), "z0_ohm must carry unit 'ohm'"),
        (dict(name="P1", net="IN", kind="port", z0_ohm=u(0.0, "ohm")), "z0_ohm must be > 0"),
        (dict(name="P1", net="IN", kind="port", z0_ohm=u(50.0, "ohm"), frequency_hz=u(447.0, "MHz")), "frequency_hz must carry unit 'Hz'"),
        (dict(name="P1", net="GND", kind="port", z0_ohm=u(50.0, "ohm")), "net and reference_net are both 'GND'"),
        (dict(name="P.1", net="IN", kind="port", z0_ohm=u(50.0, "ohm")), "must be a plain identifier"),
        (dict(name="P1", net="IN", kind="socket", z0_ohm=u(50.0, "ohm")), "kind"),
    ):
        with pytest.raises(ValueError, match=match):
            RFPort(**kw)
    assert RFPort(name="PTT", net="PTT_N", kind="control", voltage_v=u(3.3, "V"), direction="in").direction == "in"


def test_rf_expectations_carry_exactly_one_of_a_tolerance_and_a_bound():
    assert rfexp().bound == "at_least" and rfexp(tol=1.0, bound=None).tol_abs.value == 1.0 and rfexp().unit == "dB"
    for kw, match in (
        (dict(tol=1.0, bound="at_least"), "exactly one of tol_abs .* or bound .*, got both"),
        (dict(tol=None, bound=None), "got neither"),
        (dict(tol_rel=u(0.1)), "tol_rel is not allowed"),
        (dict(requirement_id="req.tx_power"), "a one-sided bound cannot claim a requirement yet"),
        (dict(quantity="rel_s21_db"), "rel_s21_db needs ref_at"),
        (dict(ref_at=F_C), "ref_at belongs to rel_s21_db, not s21_db"),
        (dict(quantity="s11_db", nominal=-15.0), "s11_db is the reflection at the drive port: 'to' must be 'P1'"),
        (dict(to="P1"), "s21_db needs two ports"),
        (dict(at=0.0), "at must be > 0"),
        (dict(bound="at_best"), "bound"),
        (dict(id="s21.fc"), "must be a plain identifier"),
    ):
        with pytest.raises(ValueError, match=match):
            rfexp(**kw)
    with pytest.raises(ValueError, match="nominal carries unit 'deg', s21_db is in dB"):
        RFExpectation(id="x", quantity="s21_db", drive="P1", to="P2", at=u(F_C, "Hz"), nominal=u(-1.0, "deg"), bound="at_least")
    with pytest.raises(ValueError, match="tol_abs carries unit 'dB', phase21_deg is in deg"):
        RFExpectation(id="x", quantity="phase21_deg", drive="P1", to="P2", at=u(F_C, "Hz"), nominal=u(90.0, "deg"), tol_abs=u(1.0, "dB"))
    with pytest.raises(ValueError, match="tol_abs must be > 0"):
        rfexp(tol=0.0, bound=None)
    with pytest.raises(ValueError, match="at must carry unit 'Hz'"):
        RFExpectation(id="x", quantity="s21_db", drive="P1", to="P2", at=u(447.0, "MHz"), nominal=u(-1.0, "dB"), bound="at_least")
    # a requirement can be claimed with a two-sided tolerance
    assert rfexp(tol=0.5, bound=None, requirement_id="req.x").requirement_id == "req.x"


def test_rf_networks_refuse_what_their_fixture_could_not_mean():
    net = lpf()
    assert [p.name for p in net.ports] == ["P1", "P2", "MID"] and net.port("MID").kind == "probe" and net.state("tx") is None
    bias = [RFState(id="bias_lo", port_dc_v={"P1": u(1.44, "V")}), RFState(id="bias_hi", port_dc_v={"P1": u(2.56, "V")})]
    stated = lpf(states=bias, expectations=[rfexp(state="bias_lo", tol=1.0, bound=None)], probes=[])
    assert stated.state("bias_hi").port_dc_v["P1"].value == 2.56
    for over, match in (
        (dict(members=[]), "needs at least one member"),
        (dict(members=["C1", "C1"]), "member 'C1' is used twice"),
        (dict(bindings={"R9": SpiceBinding(device=SpiceDevice.R, value=u(50.0, "ohm"), provenance=USER)}), r"bindings names \['R9'\]"),
        (dict(loss_q={"L9": u(40.0)}), r"loss_q names \['L9'\]"),
        (dict(q_ref_hz=None), "loss_q needs q_ref_hz"),
        (dict(loss_q={}), "q_ref_hz without loss_q"),
        (dict(loss_q={"L1": u(40.0, "Q")}), "loss_q\\[L1\\] must carry no unit"),
        (dict(ports=[port("P1", "IN"), port("P2", "IN")]), "port net 'IN' is used twice"),
        (dict(ports=[port("P1", "IN"), port("P1", "OUT")]), "port name 'P1' is used twice"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("V", "TX_5V", kind="rail", z0=None)], expectations=[rfexp(drive="V", to="P2")], probes=[]), "drive 'V' is a rail"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("V", "VB", kind="control", z0=None)], expectations=[rfexp(quantity="phase21_deg", to="V", nominal=0.0, tol=1.0, bound=None)], probes=[]), "port 'V' is a control"),
        (dict(ports=[port("Q", "IN", kind="probe", z0=None)], expectations=[], probes=[]), "needs at least one 'port'"),
        (dict(sweep=[]), "needs an ac sweep"),
        (dict(sweep=[AnalysisSpec(id="t", kind=SpiceAnalysis.TRAN, params={"step": u(1e-9, "s"), "stop": u(1e-6, "s")}, provenance=USER)]), "ac only"),
        (dict(states=bias), "state must be one of"),
        (dict(expectations=[rfexp(state="tx")]), "names state 'tx', but the network has no states"),
        (dict(expectations=[rfexp(to="P9")]), "port 'P9' is not a port of the network"),
        (dict(expectations=[rfexp(drive="MID")]), "drive 'MID' is a probe"),
        (dict(expectations=[rfexp(to="MID")]), "s21_db to the probe 'MID' is no S-parameter"),
        (dict(expectations=[rfexp(), rfexp()]), "expectation / probe id 's21_fc' is used twice"),
        (dict(probes=[RFProbe(id="s21_fc", quantity="phase21_deg", drive="P1", to="MID", at=u(F_C, "Hz"))]), "id 's21_fc' is used twice"),
        (dict(states=[RFState(id="tx", port_dc_v={"P9": u(1.0, "V")})], expectations=[], probes=[]), r"port_dc_v on \['P9'\]"),
        (dict(states=[RFState(id="tx", bindings={"D9": SpiceBinding(exclude=True, exclude_reason="x", provenance=USER)})], expectations=[], probes=[]), r"binds \['D9'\]"),
        # what the fixture runner could not simulate: a DC level on a probe, a control / level-less rail without a level in every state
        (dict(states=[RFState(id="tx", port_dc_v={"MID": u(1.0, "V")})], expectations=[], probes=[]), r"port_dc_v on the probe\(s\) \['MID'\]: a probe has no element"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("MID", "MID", kind="probe", z0=None), port("VB", "VB", kind="control", z0=None)], probes=[]),
         "control port VB needs a DC level, but the network has no states"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("MID", "MID", kind="probe", z0=None), port("VB", "VB", kind="control", z0=None, voltage_v=u(2.0, "V"))],
              probes=[]),
         "control port VB needs a DC level, but the network has no states"),  # a control's voltage_v is the interface's, never the fixture's level
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("MID", "MID", kind="probe", z0=None), port("V", "TX_5V", kind="rail", z0=None)], probes=[]),
         "rail port V needs a DC level, but the network has no states .* or the rail a voltage_v"),
        (dict(ports=[port("P1", "IN"), port("P2", "OUT"), port("VB", "VB", kind="control", z0=None)], probes=[],
              states=[RFState(id="on", port_dc_v={"VB": u(2.0, "V")}), RFState(id="off")], expectations=[rfexp(state="on")]),
         r"control port VB needs a DC level in every state .*state\(s\) \['off'\] give none"),
        (dict(sweep=[AnalysisSpec(id="rf_at_0", kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(1), "fstart": u(F_C, "Hz"),
                                                                               "fstop": u(F_C, "Hz")}, provenance=USER)]),
         "starts with 'rf_at_', the prefix of the fixture runner's own point analyses"),
    ):
        with pytest.raises(ValueError, match=match):
            lpf(**over)
    with pytest.raises(ValueError, match="port_dc_v\\[P1\\] must carry unit 'V'"):
        RFState(id="s", port_dc_v={"P1": u(1.0, "mV")})
    # a rail / control port is a DC level of the fixture (the PM tanks' bias node), never driven or read
    biased = lpf(ports=[port("P1", "IN"), port("P2", "OUT"), port("VB", "VB", kind="control", z0=None)],
                 states=[RFState(id="bias_lo", port_dc_v={"VB": u(1.44, "V")})],
                 expectations=[rfexp(state="bias_lo", quantity="phase21_deg", nominal=-21.46, tol=1.0, bound=None)], probes=[])
    assert biased.port("VB").kind == "control" and biased.state("bias_lo").port_dc_v["VB"].value == 1.44
    # the port order is free (each row names its drive): a bias port or a probe may come first; a rail with voltage_v needs no state
    first = lpf(ports=[port("VB", "VB", kind="control", z0=None), port("MID", "MID", kind="probe", z0=None), port("P1", "IN"), port("P2", "OUT")],
                states=[RFState(id="bias_lo", port_dc_v={"VB": u(1.44, "V")})], expectations=[rfexp(state="bias_lo")], probes=[])
    assert [p.kind for p in first.ports] == ["control", "probe", "port", "port"]
    assert lpf(ports=[port("V", "TX_5V", kind="rail", z0=None, voltage_v=u(5.0, "V")), port("P1", "IN"), port("P2", "OUT"), port("MID", "MID", kind="probe", z0=None)],
               probes=[]).port("V").kind == "rail"
    # a relative level or a phase to a probe is meaningful
    assert lpf(expectations=[rfexp("r", "rel_s21_db", "P1", "MID", ref_at=1e6, nominal=-3.0, bound="at_most")]).expectations[0].to == "MID"


def test_plan_lines_lab_items_rails_and_blocks():
    assert PlanLine(id="birdie", kind="margin", f_hz=u(461.676e6, "Hz"), ref_hz=u(F_C, "Hz"), min_margin_hz=u(1e6, "Hz")).min_margin_hz.value == 1e6
    assert PlanLine(id="mode", kind="coincidence", f_hz=u(F_C, "Hz"), ref_hz=u(F_C, "Hz")).kind == "coincidence"
    for kw, match in (
        (dict(kind="margin", f_hz=u(1e6, "Hz"), ref_hz=u(2e6, "Hz")), "a margin row needs ref_hz and min_margin_hz"),
        (dict(kind="response", f_hz=u(1e6, "Hz"), min_margin_hz=u(1.0, "Hz"), points_to=["x"]), "min_margin_hz belongs to a margin row"),
        (dict(kind="coincidence", f_hz=u(1e6, "Hz")), "a coincidence row needs ref_hz"),
        (dict(kind="gated", f_hz=u(1e6, "Hz")), "points_to must name the checks"),
        (dict(kind="response", f_hz=u(0.0, "Hz"), points_to=["x"]), "f_hz must be > 0"),
        (dict(kind="plan", f_hz=u(1e6, "Hz")), "kind"),
    ):
        with pytest.raises(ValueError, match=match):
            PlanLine(id="row", **kw)
    with pytest.raises(ValueError, match="'what' and 'reason' must say something"):
        LabItem(id="obw", what=" ", reason="x")
    rail = dict(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.086, "A"), i_max=u(0.161, "A"))
    assert RailBudget(**rail, i_rating=u(0.5, "A"), path_r_ohm=u(0.17, "ohm")).i_rating.value == 0.5
    for over, match in (
        (dict(i_max=u(0.05, "A")), "i_max 0.05 A is below i_min"),
        (dict(rail="RX 5V"), "must be a net name"),
        (dict(v_out=u(5.0, "mV")), "v_out must carry unit 'V'"),
        (dict(i_rating=u(0.0, "A")), "i_rating must be > 0"),
        (dict(dropout_v=u(-0.1, "V")), "dropout_v must be >= 0"),
    ):
        with pytest.raises(ValueError, match=match):
            RailBudget(**{**rail, **over})
    region = RFRegion(x=u(32.0, "mm"), y=u(6.0, "mm"), w=u(26.0, "mm"), h=u(26.0, "mm"))
    assert region.box() == (32.0, 6.0, 58.0, 32.0)
    with pytest.raises(ValueError, match="region w must be > 0"):
        RFRegion(x=u(0.0, "mm"), y=u(0.0, "mm"), w=u(0.0, "mm"), h=u(1.0, "mm"))
    with pytest.raises(ValueError, match="region x must carry unit 'mm'"):
        RFRegion(x=u(0.0, "cm"), y=u(0.0, "mm"), w=u(1.0, "mm"), h=u(1.0, "mm"))
    for kw, match in (
        (dict(refs=["L1"], chain=["L2"]), r"chain names \['L2'\]"),
        (dict(refs=["L1"], shield_ref="SH9"), "shield_ref 'SH9' is not a ref of the block"),
        (dict(refs=["L1", "L1"]), "ref 'L1' is used twice"),
        (dict(ports=[port("A", "X"), port("A", "Y")]), "port name 'A' is used twice"),
    ):
        with pytest.raises(ValueError, match=match):
            RFBlock(id="blk", **kw)


def test_the_rf_design_ties_its_parts_together_and_names_every_traced_number():
    blocks = [RFBlock(id="lna", refs=["L1", "C1"], region=RFRegion(x=u(32.0, "mm"), y=u(6.0, "mm"), w=u(26.0, "mm"), h=u(26.0, "mm")),
                      ports=[port("RF_IN", "RF_IN")]),
              RFBlock(id="lo", refs=["Y701"])]
    rails = [RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.086, "A"), i_max=u(0.161, "A"))]
    plan = [PlanLine(id="birdie", kind="margin", f_hz=u(461.676e6, "Hz"), ref_hz=u(F_C, "Hz"), min_margin_hz=u(1e6, "Hz"))]
    binding = SpiceBinding(device=SpiceDevice.L, value=u(23.588e-9, "H"), params={"ic": u(0.0, "A")}, provenance=USER)
    states = [RFState(id="tx", port_dc_v={"P1": u(5.0, "V")}, bindings={"C1": SpiceBinding(device=SpiceDevice.C, value=u(7.8e-12, "F"), provenance=USER)})]
    net = lpf(block="lna", bindings={"L1": binding}, states=states, expectations=[rfexp(state="tx")],
              probes=[RFProbe(id="p3f", state="tx", quantity="rel_s21_db", drive="P1", to="P2", at=u(3 * F_C, "Hz"), ref_at=u(F_C, "Hz"))])
    rf = RFDesign(blocks=blocks, networks=[net], frequency_plan=plan, rails=rails, model_values=["model.l_q"], profile_keys=["kr447.max_power"])
    ids = [k for k, _ in rf.traced_items()]
    assert ids == [
        "rf.blocks[lna].region.x", "rf.blocks[lna].region.y", "rf.blocks[lna].region.w", "rf.blocks[lna].region.h",
        "rf.blocks[lna].ports[RF_IN].z0_ohm",
        "rf.networks[lpf].ports[P1].z0_ohm", "rf.networks[lpf].ports[P2].z0_ohm",
        "rf.networks[lpf].bindings[L1].value", "rf.networks[lpf].bindings[L1].params[ic]",
        "rf.networks[lpf].loss_q[L1]", "rf.networks[lpf].q_ref_hz",
        "rf.networks[lpf].states[tx].port_dc_v[P1]", "rf.networks[lpf].states[tx].bindings[C1].value",
        "rf.networks[lpf].sweep[ac1].params[variation]", "rf.networks[lpf].sweep[ac1].params[points]",
        "rf.networks[lpf].sweep[ac1].params[fstart]", "rf.networks[lpf].sweep[ac1].params[fstop]",
        "rf.networks[lpf].expectations[s21_fc].at", "rf.networks[lpf].expectations[s21_fc].nominal",
        "rf.networks[lpf].probes[p3f].at", "rf.networks[lpf].probes[p3f].ref_at",
        "rf.frequency_plan[birdie].f_hz", "rf.frequency_plan[birdie].ref_hz", "rf.frequency_plan[birdie].min_margin_hz",
        "rf.rails[RX_5V].v_out", "rf.rails[RX_5V].i_min", "rf.rails[RX_5V].i_max",
    ]
    assert rf.lookup("rf.networks[lpf].loss_q[L1]").value == 40.0 and rf.lookup("rf.networks[lpf].nope") is None and rf.lookup("si.x") is None
    assert [k for k, _ in rf.traced_items("x")][0] == "x.blocks[lna].region.x"
    assert rf.block_of("Y701").id == "lo" and rf.block_of("R1") is None and rf.network("lpf") is net and rf.block("nope") is None
    for kw, match in (
        (dict(blocks=[RFBlock(id="a", refs=["L1"]), RFBlock(id="b", refs=["L1"])]), "ref 'L1' is in two blocks"),
        (dict(blocks=[RFBlock(id="a"), RFBlock(id="a")]), "block id 'a' is used twice"),
        (dict(networks=[lpf(), lpf()]), "network id 'lpf' is used twice"),
        (dict(networks=[lpf(block="pa")]), "network lpf: block 'pa' is not a block of the design"),
        (dict(lab_items=[LabItem(id="obw", block="pa", what="x", reason="y")]), "lab item obw: block 'pa'"),
        (dict(lab_items=[LabItem(id="obw", what="x", reason="y"), LabItem(id="obw", what="x", reason="y")]), "lab item id 'obw' is used twice"),
        (dict(rails=[*rails, RailBudget(rail="RX_5V", regulator_ref="U9", v_out=u(5.0, "V"), i_min=u(0.0, "A"), i_max=u(0.1, "A"))]), "rail 'RX_5V' is used twice"),
        (dict(model_values=["l_q"]), "must start with 'model.'"),
        (dict(model_values=["model.l_q", "model.l_q"]), "model value key 'model.l_q' is used twice"),
        (dict(profile_keys=["kr447.max_power", "kr447.max_power"]), "profile key 'kr447.max_power' is used twice"),
        (dict(frequency_plan=[*plan, *plan]), "plan line id 'birdie' is used twice"),
    ):
        with pytest.raises(ValueError, match=match):
            RFDesign(**kw)


def test_recompute_rederives_a_fixture_nominal_under_rf():
    ir = CircuitIR.load(PRE_KR447_IR)
    ir.parameters["f_q"] = u(1000.0, "Hz")
    ir.parameters["tau"] = rc_time_constant(ir.parameters["r1"], ir.parameters["c1"], ("r1", "c1"))
    phase = rc_lowpass_phase_deg(ir.parameters["f_q"], ir.parameters["tau"], ("f_q", "tau"))
    assert phase.value == pytest.approx(-45.0)
    exp = RFExpectation(id="corner_phase", quantity="phase21_deg", drive="P1", to="OUT", at=u(1000.0, "Hz"), nominal=phase, tol_abs=u(1.0, "deg"))
    ir.rf = RFDesign(networks=[RFNetwork(id="rc", members=["R1", "C1"], ports=[port("P1", "IN"), port("OUT", "OUT", kind="probe", z0=None)],
                                         sweep=[ac(fstart=10.0, fstop=1e5)], expectations=[exp])])
    path = "rf.networks[rc].expectations[corner_phase].nominal"
    assert (path, phase) in list(derived_values(ir))
    good = recompute_parameters(ir)
    assert good.status is S.PASS and good.details["parameters"][path]["status"] is S.PASS
    ir.rf.networks[0].expectations[0].nominal = phase.model_copy(update={"value": -40.0})
    bad = recompute_parameters(ir)
    assert bad.status is S.FAIL and any(m.startswith(path) for m in bad.details["mismatches"])


# --------------------------------------------------------------------------- keep-outs


def test_keepouts_state_one_area_on_copper_layers_and_only_exceptions_to_what_they_forbid():
    k = keepout()
    assert k.outline() == [(0.0, 0.0), (60.0, 0.0), (60.0, 6.0), (0.0, 6.0)] and k.bbox() == (0.0, 0.0, 60.0, 6.0)
    assert k.covers_layer("In2.Cu") and k.covers_layer("B.Cu") and not k.covers_layer("F.SilkS")
    tri = keepout(layers=["B.Cu"], rect=None, polygon=u([[10.0, 10.0], [20.0, 10.0], [15.0, 18.0]], "mm"), forbids=["tracks"], allowed_refs=[])
    assert tri.bbox() == (10.0, 10.0, 20.0, 18.0) and tri.covers_layer("B.Cu") and not tri.covers_layer("F.Cu")
    for over, match in (
        (dict(rect=None), "exactly one of rect"),
        (dict(polygon=u([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "mm")), "exactly one of rect"),
        (dict(rect=u([0.0, 0.0, 60.0, 6.0])), "unit 'mm'"),
        (dict(rect=u([0.0, 0.0, 60.0], "mm")), r"rect must be \[x, y, w, h\]"),
        (dict(rect=u([0.0, 0.0, 60.0, 0.0], "mm")), "rect width and height must be > 0"),
        (dict(rect=None, polygon=u([[0.0, 0.0], [1.0, 1.0]], "mm")), "at least three"),
        (dict(rect=None, polygon=u([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], "mm")), "encloses no area"),
        (dict(layers=[]), "at least one copper layer"),
        (dict(layers=["*.Cu", "F.Cu"]), "stands alone"),
        (dict(layers=["F.SilkS"]), "is not a copper layer"),
        (dict(layers=["F.Cu", "F.Cu"]), "listed twice"),
        (dict(forbids=[]), "forbids nothing"),
        (dict(forbids=["tracks", "tracks"]), "listed twice in forbids"),
        (dict(forbids=["silk"]), "forbids"),
        (dict(forbids=["tracks"]), "allowed_refs are exceptions to a footprints / pads ban"),
        (dict(forbids=["footprints"]), "allowed_nets are exceptions to a pads / tracks / vias / zones ban"),
        (dict(allowed_nets=["ANT_FEED", "ANT_FEED"]), "listed twice in allowed_nets"),
        (dict(reason=" "), "reason must say why"),
        (dict(id="ant band"), "plain identifier"),
    ):
        with pytest.raises(ValueError, match=match):
            keepout(**over)
    with pytest.raises(ValueError, match="keep-out ids must be unique"):
        PCBDesign(keepouts=[keepout(), keepout()])
    unrecorded = Keepout(id="ant_band", layers=["*.Cu"], rect=u([0.0, 0.0, 60.0, 6.0], "mm"), forbids=["tracks"], reason="antenna band")
    board = PCBDesign(keepouts=[unrecorded])
    label, prov = board.layout_items()[-1]
    assert label == "keepout[0:ant_band]" and prov.needs_verification  # a keep-out nobody said the origin of is an assumption


# --------------------------------------------------------------------------- judging a bound


def _interp(y0: float, y1: float) -> Interpolation:
    return Interpolation(value=(y0 + y1) / 2.0, exact=False, x0=1.0, y0=y0, x1=2.0, y1=y1, method="linear")


def test_judge_takes_a_one_sided_bound_on_its_passing_side_and_the_whole_bracket():
    least = rfexp(nominal=-1.5, bound="at_least")
    most = rfexp(nominal=-30.0, bound="at_most")
    assert judge(-1.0, least) == (S.PASS, None, pytest.approx(0.5))
    assert judge(-1.5, least) == (S.PASS, None, 0.0)  # equal passes
    assert judge(-2.0, least) == (S.FAIL, None, pytest.approx(-0.5))
    assert judge(-31.0, most) == (S.PASS, None, pytest.approx(-1.0))
    assert judge(-30.0, most)[0] is S.PASS and judge(-29.0, most)[0] is S.FAIL
    # an interpolated value: the two samples it sits between decide
    assert judge(-1.2, least, _interp(-1.0, -1.4))[0] is S.PASS
    assert judge(-1.5, least, _interp(-1.0, -2.0))[0] is S.UNRESOLVED
    assert judge(-2.0, least, _interp(-1.8, -2.2))[0] is S.FAIL
    assert judge(-1.0, least, Interpolation(value=-1.0, exact=True, x0=1.0, y0=-1.0, x1=1.0, y1=-1.0, method="exact"))[0] is S.PASS
    # a bias bracket (None = unbounded edge)
    assert judge(-31.0, most, bias=(None, -30.5))[0] is S.PASS
    assert judge(-31.0, most, bias=(-31.5, None))[0] is S.UNRESOLVED
    assert judge(-29.0, most, bias=(-29.5, None))[0] is S.FAIL
    assert judge(-31.0, most, bias=(-31.5, -29.5))[0] is S.UNRESOLVED
    # a bound beside a tolerance is two claims: nothing judged (the compiler and RFExpectation refuse it)
    both = rfexp(nominal=-1.5, tol=None, bound="at_least").model_copy(update={"tol_abs": u(0.1, "dB")})
    assert judge(-1.0, both) == (S.UNRESOLVED, None, pytest.approx(0.5))
    # a two-sided RF expectation is judged by the tolerance, as a design-deck expectation is
    assert judge(-9.5, rfexp(nominal=-10.0, tol=1.0, bound=None)) == (S.PASS, 1.0, pytest.approx(0.5))
    assert judge(-8.5, rfexp(nominal=-10.0, tol=1.0, bound=None))[0] is S.FAIL
    assert RF_CHECK_PREFIX == "spice.rf"


# --------------------------------------------------------------------------- the compiler and the stage


def _bounded(ir: CircuitIR) -> CircuitIR:
    """The pinned RC IR with its two dB rows turned into one-sided bounds, and two more bounded rows at 3 kHz (between sweep points)."""
    exps = ir.simulation.expectations
    exps[1].bound, exps[1].tol_abs, exps[1].nominal = "at_most", None, u(-2.0, "dB")  # corner_db: -3.0103 dB at 1 kHz
    exps[2].bound, exps[2].tol_abs, exps[2].nominal = "at_least", None, u(-19.0, "dB")  # stop_db: -20.04 dB at 10 kHz
    for id, nominal in (("mid_straddle", -10.0), ("mid_below", -2.0)):
        exps.append(exps[2].model_copy(update={"id": id, "at": u(3000.0, "Hz"), "nominal": u(nominal, "dB"), "bound": "at_most"}, deep=True))
    return ir


def test_the_compiler_refuses_a_bound_with_a_tolerance_or_a_requirement_and_waives_the_db_rule_for_it():
    ir = _bounded(CircuitIR.load(PRE_KR447_IR))
    assert build(ir).startswith("rc_before_kr447\n")  # one-sided dB levels compile without tol_abs
    exps = ir.simulation.expectations
    for mutate, match in (
        (lambda e: setattr(e, "tol_abs", u(0.1, "dB")), r"a one-sided bound \(at_most\) takes no tolerance"),
        (lambda e: setattr(e, "tol_rel", u(0.1)), "takes no tolerance"),
        (lambda e: setattr(e, "requirement_id", "req.f_c"), "a one-sided bound cannot claim a requirement yet"),
        (lambda e: setattr(e, "bound", None), "a level in dB needs tol_abs .* or a one-sided bound"),
    ):
        other = copy.deepcopy(ir)
        mutate(other.simulation.expectations[1])
        with pytest.raises(CompileError, match=match):
            build(other)
    # a bound on a plain reduction (at, on a voltage) compiles too
    exps[0].bound, exps[0].tol_abs, exps[0].requirement_id = "at_least", None, None
    build(ir)


class _FakeAC(SpiceRunner):
    """An ac 'engine' that returns the exact RC low-pass magnitudes (corner 1 kHz) on five decade points - no ngspice needed."""

    engine = "fake-ac"
    F = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "fake-1"

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None) -> SpiceResult:
        out = [1.0 / math.sqrt(1.0 + (f / 1000.0) ** 2) for f in self.F]
        return SpiceResult(engine=self.engine, engine_version="fake-1", netlist_path=str(netlist_path), netlist_hash=self.netlist_hash(netlist_path),
                           analysis=analysis, command=command or "ac", plot_name="ac1", scale="frequency", n_points=len(self.F),
                           vectors={"frequency": list(self.F), "out": out, "in": [1.0] * len(self.F)}, succeeded=True)


def _run(ir: CircuitIR, tmp_path: Path, runner: SpiceRunner) -> dict:
    ir.project.workdir = str(tmp_path)
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = SpiceNetlistCompiler().compile(ir, CompileContext(workdir=tmp_path, tools={"spice": runner}))
    return {r.check_id: r for r in run_spice_for(ir, {"spice": runner}, tmp_path)}


def test_the_stage_records_a_bound_and_a_signed_deviation(tmp_path: Path):
    got = _run(_bounded(CircuitIR.load(PRE_KR447_IR)), tmp_path, _FakeAC())
    corner, stop, straddle, below = (got[f"spice.{i}"] for i in ("corner_db", "stop_db", "mid_straddle", "mid_below"))
    assert corner.status is S.PASS and corner.details["bound"] == "at_most" and corner.details["tolerance"] is None
    assert corner.details["deviation"] == pytest.approx(-3.0103 + 2.0, abs=1e-4)
    assert "the one-sided bound at most -2 dB (measured - nominal -1.01 dB)" in corner.message, corner.message
    assert stop.status is S.FAIL and stop.details["repair"] == "human" and "at least -19 dB" in stop.message and stop.details["deviation"] < 0
    # 3 kHz lies between the 1 kHz (-3.01 dB) and 10 kHz (-20.04 dB) samples: the whole bracket is judged
    assert straddle.status is S.UNRESOLVED and "which lie on both sides of the one-sided bound at most -10 dB" in straddle.message
    assert "repair" not in straddle.details and straddle.details["bracket"]["exact"] is False
    assert below.status is S.PASS, below.message
    assert got["spice.corner"].status is S.PASS and "bound" not in got["spice.corner"].details  # a tolerance row is unchanged
    assert got["spice"].status is S.FAIL and got["spice"].details["expectations"]["mid_straddle"] == "UNRESOLVED"


@pytest.mark.skipif(not NgspiceShared().available(), reason="ngspice shared library not found")
def test_a_bound_on_a_real_ngspice_deck(tmp_path: Path):
    got = _run(_bounded(CircuitIR.load(PRE_KR447_IR)), tmp_path, NgspiceShared())
    assert got["spice.corner_db"].status is S.PASS and got["spice.corner_db"].details["measured"] == pytest.approx(-3.0103, abs=0.01)
    assert got["spice.stop_db"].status is S.FAIL and got["spice.stop_db"].details["measured"] == pytest.approx(-20.04, abs=0.05)
    # dec 20 puts 3 kHz between 2818 Hz (-9.515 dB) and 3162 Hz (-10.414 dB): the bracket straddles -10 dB (measured
    # -10.0026 dB would pass alone), and lies wholly below -2 dB
    straddle = got["spice.mid_straddle"]
    assert straddle.status is S.UNRESOLVED and straddle.details["measured"] == pytest.approx(-10.0026, abs=1e-3), straddle.message
    assert got["spice.mid_below"].status is S.PASS


def test_retire_leaves_the_rf_fixture_results_to_their_runner():
    ir = CircuitIR.load(PRE_KR447_IR)
    for check_id in ("spice.rf.lpf", "spice.rf.lpf.s21_fc", "spice.rf.trsw.tx.s21", "spice.rf", "spice.rf_gain", "spice.si.ANT", "spice.old", "spice.corner"):
        ir.validation.add(ValidationResult(check_id=check_id, status=S.PASS, message="earlier run"))
    retired = retire_expectation_results(ir, {"corner"}, S.NOT_APPLICABLE, "gone")
    assert sorted(r.check_id for r in retired) == ["spice.old", "spice.rf", "spice.rf_gain"]
