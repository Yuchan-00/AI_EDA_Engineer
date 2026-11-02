"""``spice.si.<net>``: the lossless-line transient of every electrically long net over a plane.

Without ngspice: the deck is compiled by the SPICE netlist compiler (a
``T`` line with ``z0`` / ``td``), the waveform judgement is exact on
synthetic waveforms, and every reason not to simulate is NOT_VERIFIED
naming it (no plane, a missing driver value, no engine); the SPICE stage's
own retirement of stale ``spice.<id>`` results leaves ``spice.si.*`` alone.
With ngspice (skipped otherwise): the deck reproduces the case measured on
ngspice-42 (1 ns edge through 40 ohm into 50 ohm / 1 ns with 5 pF: 11.5 %
overshoot) within 0.5 %, and a long routed net is judged PASS / FAIL against
its class's band with the rawfile as evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.compilers.spice import build
from ai_eda.design.board import PLANE_CLEARANCE_KEY
from ai_eda.ir import Component, LibraryRef, Net, Pin, PinElectricalType, PinRef, Provenance, ProvenanceKind, ValidationResult, ValidationStatus as S, user_requirement
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis
from ai_eda.tools.calc.tline import microstrip
from ai_eda.tools.si import measure_nets
from ai_eda.tools.spice.si_check import deck_ir, deck_stem, judge_waveform, spice_si_results
from ai_eda.tools.spice.stage import retire_expectation_results
from tests.test_routing import fixture_library
from tests.test_si_checks import default_classes, long_board, si_of, track

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
#: the measured case of tests/test_spice_tline.py (5 V swing there): max v(far end) / swing
MEASURED_PEAK_REL = 5.5757 / 5.0


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def _routed(tmp_path: Path, lib, layers: int, **driver_kw):
    ir = long_board(tmp_path, lib, layers)
    ir.si = si_of(*default_classes(**driver_kw))
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    return ir


def test_the_deck_is_the_compilers_lossless_line():
    ir, hold, step, stop = deck_ir("si_x", z0=50.0, td=1e-9, t_r=1e-9, r_drive=40.0, c_load=5e-12)
    text = build(ir)
    assert any(line.startswith("T1 A 0 B 0 td=") and "z0=50" in line for line in text.splitlines())
    assert hold == pytest.approx(20 * 3e-9 + 10 * 90 * 5e-12) and step == pytest.approx(1e-9 / 50) and stop == pytest.approx(2 * (1e-9 + hold))
    assert build(deck_ir("si_x", z0=50.0, td=1e-9, t_r=1e-9, r_drive=40.0, c_load=5e-12)[0]) == text  # deterministic


def test_the_waveform_judgement():
    t_r, hold = 1e-9, 20e-9
    time = [i * 0.1e-9 for i in range(int((2 * (t_r + hold)) / 0.1e-9) + 1)]

    def wave(peak: float, dip: float) -> list[float]:
        out = []
        for t in time:
            if t < t_r:
                out.append(t / t_r)
            elif t < t_r + 2e-9:
                out.append(1.0 + peak)
            elif t <= t_r + hold:
                out.append(1.0)
            elif t < t_r + hold + 2e-9:
                out.append(-dip)
            else:
                out.append(0.0)
        return out

    status, measured, text = judge_waveform(time, wave(0.10, 0.05), t_r=t_r, hold=hold, tol=0.15)
    assert status is S.PASS and measured["overshoot_rel"] == pytest.approx(0.10) and measured["undershoot_rel"] == pytest.approx(0.05) and "overshoot 10.0%" in text
    status, measured, text = judge_waveform(time, wave(0.20, 0.05), t_r=t_r, hold=hold, tol=0.15)
    assert status is S.FAIL and "overshoot 20.0% > 15%" in text
    status, _, text = judge_waveform(time, wave(0.0, 0.3), t_r=t_r, hold=hold, tol=0.15)
    assert status is S.FAIL and "undershoot 30.0% > 15%" in text
    ringing = [v + (0.3 if t > t_r + hold - 1e-9 and t <= t_r + hold else 0.0) for t, v in zip(time, wave(0.0, 0.0))]
    status, _, text = judge_waveform(time, ringing, t_r=t_r, hold=hold, tol=0.15)
    assert status is S.FAIL and "not settled" in text
    assert judge_waveform([], [], t_r=t_r, hold=hold, tol=0.15)[0] is S.FAIL


def test_every_reason_not_to_simulate_is_not_verified_naming_it(tmp_path: Path, lib):
    two = _routed(tmp_path / "2", lib, 2)
    res = spice_si_results(two, {"spice": runner}, tmp_path / "w2")
    assert [r.check_id for r in res] == ["spice.si.LONG"] and res[0].status is S.NOT_VERIFIED
    assert "impedance is undefined without a reference plane - use pcb_layers=4 or add a plane" in res[0].message
    four = _routed(tmp_path / "4", lib, 4)
    res = spice_si_results(four, {}, tmp_path / "w4")
    assert res[0].status is S.NOT_VERIFIED and "no SPICE engine available" in res[0].message and res[0].details["z0_ohm"] > 40
    no_drive = _routed(tmp_path / "n", lib, 4, r_drive_ohm=None)
    res = spice_si_results(no_drive, {"spice": runner}, tmp_path / "wn")
    assert res[0].status is S.NOT_VERIFIED and "missing si.net_classes[DEFAULT].r_drive_ohm" in res[0].message
    four.si = None
    assert spice_si_results(four, {"spice": runner}, tmp_path / "w0") == []


def test_stale_results_are_superseded_by_their_own_check_and_never_by_the_spice_stage(tmp_path: Path, lib):
    ir = _routed(tmp_path, lib, 2)
    ir.validation.add(ValidationResult(check_id="spice.si.OLD", status=S.FAIL, message="an earlier long net"))
    ir.validation.add(ValidationResult(check_id="spice.gone", status=S.PASS, message="an expectation no longer in the setup"))
    retired = retire_expectation_results(ir, set(), S.NOT_APPLICABLE, "gone")
    assert [r.check_id for r in retired] == ["spice.gone"]
    res = {r.check_id: r for r in spice_si_results(ir, {}, tmp_path / "w")}
    assert res["spice.si.OLD"].status is S.NOT_APPLICABLE and res["spice.si.OLD"].details["superseded"] == "FAIL"


@needs_ngspice
def test_the_deck_reproduces_the_measured_overshoot_of_a_lossless_line(tmp_path: Path):
    ir, hold, step, stop = deck_ir("si_measured", z0=50.0, td=1e-9, t_r=1e-9, r_drive=40.0, c_load=5e-12)
    path = tmp_path / "deck.cir"
    path.write_text(build(ir), encoding="utf-8", newline="\n")
    from ai_eda.compilers.spice import analysis_command

    res = runner.run(path, SpiceAnalysis.TRAN, tmp_path / "run", analysis_command(ir.simulation.analyses[0], ir.simulation))
    assert res.succeeded, res.errors
    peak = max(res.vector("b"))
    assert peak == pytest.approx(MEASURED_PEAK_REL, rel=0.005)
    status, measured, _ = judge_waveform(res.scale_values(), res.vector("b"), t_r=1e-9, hold=hold, tol=0.15)
    assert status is S.PASS and measured["overshoot_rel"] == pytest.approx(MEASURED_PEAK_REL - 1.0, abs=0.006)
    assert judge_waveform(res.scale_values(), res.vector("b"), t_r=1e-9, hold=hold, tol=0.10)[0] is S.FAIL


@needs_ngspice
def test_a_long_routed_net_over_a_plane_is_judged_by_ngspice_with_rawfile_evidence(tmp_path: Path, lib):
    ir = _routed(tmp_path, lib, 4)
    res = spice_si_results(ir, {"spice": runner}, tmp_path / "w")
    r = {x.check_id: x for x in res}["spice.si.LONG"]
    assert r.status in (S.PASS, S.FAIL) and r.tool == runner.engine and r.tool_version and r.artifact_hash and r.ir_hash == ir.content_hash()
    assert any("rawfile" in e.description and e.content_hash for e in r.evidence) and any(e.description.startswith("spice.si deck") for e in r.evidence)
    over = r.details["measured"]["overshoot_rel"]
    # the deck carries the calculator's Z0 of the routed 0.4 mm track over the 0.2 mm prepreg and the net's measured delay
    z0 = microstrip(0.4, 0.2, 35.0, 4.5).z0_ohm
    assert r.details["z0_ohm"] == pytest.approx(z0, rel=1e-12) and r.details["td_s"] == pytest.approx(measure_nets(ir)["LONG"].delay_s, rel=1e-12)
    (tline,) = [line for line in r.details["deck"].splitlines() if line.startswith("T1 ")]
    assert float(tline.split("z0=")[1].split()[0]) == pytest.approx(z0, rel=1e-12)
    # the band decides: just above the measured overshoot passes, just below fails
    wide = _routed(tmp_path / "wide", lib, 4, ringing_tol_rel=user_requirement(over + 0.02))
    narrow = _routed(tmp_path / "narrow", lib, 4, ringing_tol_rel=user_requirement(max(over - 0.02, 0.001)))
    assert {x.check_id: x for x in spice_si_results(wide, {"spice": runner}, tmp_path / "ww")}["spice.si.LONG"].status is S.PASS
    assert {x.check_id: x for x in spice_si_results(narrow, {"spice": runner}, tmp_path / "wn")}["spice.si.LONG"].status is S.FAIL


def test_the_simulation_agent_adds_the_si_transients_to_the_spice_stage(tmp_path: Path, lib):
    from ai_eda.agents import AgentContext, SimulationAgent

    ir = _routed(tmp_path, lib, 2)
    res = SimulationAgent().run(ir, AgentContext(workdir=tmp_path, tools={}))
    got = {r.check_id: r for r in res.validation}
    assert got["spice"].status is S.NOT_VERIFIED and got["spice.si.LONG"].status is S.NOT_VERIFIED  # no setup of its own; the SI transient still answers
    assert any(n.startswith("spice.si: 1 electrically long net(s) as lossless lines: 1 NOT_VERIFIED") for n in res.notes)
    ir.si = None
    assert not any(r.check_id.startswith("spice.si") for r in SimulationAgent().run(ir, AgentContext(workdir=tmp_path, tools={})).validation)


def test_a_net_the_one_line_deck_does_not_represent_is_not_verified_naming_why(tmp_path: Path, lib):
    """The deck is one driver, one line, one receiver: a tree, a part it does not model (a 100 nF capacitor makes a 1 ns edge
    impossible) or a class driver whose pin on the net is an input is NOT_VERIFIED - never a verdict about a circuit not in the IR."""
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="fixture")
    ir = _routed(tmp_path, lib, 4)
    # a third pad on a 5 mm branch at the middle: the pad-to-pad path R1.1-R2.1 (90 mm) is long, and the net is a tree
    tap = ir.component("R2").model_copy(deep=True, update={"ref": "R7"})
    ir.components.append(tap)
    ir.pcb.placements.append(ir.pcb.placement("R2").model_copy(update={"component_ref": "R7", "x_mm": 50.0, "y_mm": 10.0}))
    ir.net("LONG").pins.append(PinRef(component_ref="R7", pin_number="1"))
    ir.pcb.tracks.append(ir.pcb.tracks[0].model_copy(update={"start": (50.0, 5.0), "end": (50.0, 10.0)}))
    r = {x.check_id: x for x in spice_si_results(ir, {"spice": runner, "kicad_library": lib}, tmp_path / "w")}["spice.si.LONG"]
    assert r.status is S.NOT_VERIFIED and "LONG joins 3 pads (R1.1, R2.1, R7.1)" in r.message and "not a tree" in r.message and "deck" not in r.details
    assert r.details["line"] == "the longest pad-to-pad path R1.1-R2.1" and r.details["line_mm"] == pytest.approx(90.0) and r.details["length_mm"] == pytest.approx(95.0)
    # without the library the 95 mm of copper is only an upper bound of every path: possibly long, never simulated
    assert not any(x.check_id == "spice.si.LONG" for x in spice_si_results(ir, {"spice": runner}, tmp_path / "w0"))
    cap = Component(ref="C8", value="100n", symbol=LibraryRef(library="Device", name="C"), provenance=prov,
                    pins=[Pin(number=n, name="~", electrical_type=PinElectricalType.PASSIVE, provenance=prov) for n in ("1", "2")])
    ir.components.append(cap)
    ir.pcb.tracks.pop()
    ir.net("LONG").pins = [PinRef(component_ref="R1", pin_number="1"), PinRef(component_ref="C8", pin_number="1")]
    r = {x.check_id: x for x in spice_si_results(ir, {"spice": runner}, tmp_path / "w")}["spice.si.LONG"]
    assert r.status is S.NOT_VERIFIED and "LONG carries a part the deck does not model: C8.1 100n (Device:C, passive pin)" in r.message
    # the class names R2 as the driver, but R2's pin on LONG is an input: the edge the deck would launch does not exist
    board = _routed(tmp_path / "d", lib, 4)
    board.si.net_classes[0] = board.si.net_classes[0].model_copy(update={"driver": "R2"})
    r = {x.check_id: x for x in spice_si_results(board, {"spice": runner}, tmp_path / "wd")}["spice.si.LONG"]
    assert r.status is S.NOT_VERIFIED and "the class driver R2 does not drive LONG (pin 1 ~ is input)" in r.message


def test_every_net_has_its_own_deck_stem():
    """D+ and D- (or clk and CLK on a case-insensitive file system) never share a deck or a rawfile."""
    stems = [deck_stem(n) for n in ("D+", "D-", "clk", "CLK", "D_")]
    assert len({x.lower() for x in stems}) == len(stems) and all(x.split("-")[0] in ("D_", "clk", "CLK") for x in stems)
    assert deck_stem("D+") == deck_stem("D+")


@needs_ngspice
def test_two_nets_whose_names_differ_only_in_punctuation_keep_their_own_evidence(tmp_path: Path, lib):
    import hashlib

    ir = _routed(tmp_path, lib, 4)
    ir.net("LONG").name = "D+"
    ir.pcb.tracks = [t.model_copy(update={"net": "D+"}) if t.net == "LONG" else t for t in ir.pcb.tracks]
    # a second long driver -> receiver net, D-, under the first one
    for ref, x, kind in (("R7", 5.0, PinElectricalType.OUTPUT), ("R8", 95.0, PinElectricalType.INPUT)):
        c = ir.component("R1").model_copy(deep=True, update={"ref": ref})
        c.pins[0].electrical_type = kind
        ir.components.append(c)
        ir.pcb.placements.append(ir.pcb.placement("R1").model_copy(update={"component_ref": ref, "x_mm": x, "y_mm": 10.0}))
    ir.nets.append(Net(name="D-", pins=[PinRef(component_ref="R7", pin_number="1"), PinRef(component_ref="R8", pin_number="1")], provenance=ir.net("D+").provenance))
    ir.pcb.tracks.append(ir.pcb.tracks[0].model_copy(update={"net": "D-", "start": (5.0, 10.0), "end": (95.0, 10.0)}))
    res = {r.check_id: r for r in spice_si_results(ir, {"spice": runner}, tmp_path / "w")}
    assert {"spice.si.D+", "spice.si.D-"} <= set(res)
    for check in ("spice.si.D+", "spice.si.D-"):
        r = res[check]
        assert r.status in (S.PASS, S.FAIL) and len(r.evidence) == 2
        for e in r.evidence:
            assert "sha256:" + hashlib.sha256(Path(e.path).read_bytes()).hexdigest() == e.content_hash, (check, e.path)
    assert len({e.path for r in (res["spice.si.D+"], res["spice.si.D-"]) for e in r.evidence}) == 4
