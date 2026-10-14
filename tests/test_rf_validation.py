"""The RF design checks (:mod:`ai_eda.validation.rf`): frequency plan, the unverified regulatory profile, model grounding,
the deviation chain, lab items, block interfaces and the rail budgets.

Nothing here needs KiCad or ngspice: the SPICE verdicts the checks read are
recorded results built the way the SPICE stage and the fixture runner stamp
them (tool, design hash, ``details["measured"]``, evidence files with their
hashes). What is proved:

* the validator is registered, consumes ``spice`` (so it runs again after the
  SPICE stage), gives nothing on a design without ``ir.rf`` and retires every
  result of its own the IR no longer produces;
* ``rf.freq_plan``: margin / coincidence rows are arithmetic on confirmed
  frequencies (the N 12 plan's 13 f_R and 3 f_R - 4 LO2 margins PASS), an
  unconfirmed frequency is no verdict, a response / gated row copies what it
  points to only when that is a tool-backed result about this design, a lab
  item is NOT_VERIFIED;
* ``rf.regulatory_profile`` never PASSes, FAILs a confirmed exceedance
  (27 dBm is 0.501 W > 0.5 W) and a carrier outside the band or off the raster
  (447.000 MHz: a band is not a carrier), names the channel of a carrier on
  the raster;
* ``rf.model_grounding`` names every ungrounded ``model.*`` value;
  ``rf.lab.<id>`` is NOT_VERIFIED;
* ``rf.deviation`` multiplies N, the two tanks' chord slopes (the
  ``pm_n12`` numbers -21.43 / +1.39 / +18.21 deg at 1.44 / 2.00 / 2.56 V give
  0.6177 rad/V each and the linearity figure -0.151), a, V_max and tau_i, and
  is PASS / FAIL only when every factor PASSed on this design with its
  evidence on disk; the model constant ``model.k_pm`` is never a verdict;
* ``block.interface.<net>`` compares impedance, frequency, rail voltage and
  direction;
* ``power.rail_budget`` / ``power.headroom`` through ``calc.power.rail_budget``
  / ``calc.regulator.headroom``: the LM1117 at the 6.4 V cut-off is +0.125 V
  (NOT_VERIFIED, ungrounded), at 6.0 V -0.275 V (FAIL); a regulator fed from
  another rail is judged at that rail's output; PASS only on grounded values.

The tests that need the RF IR types (``ai_eda.ir.rf``) or the power
calculators (``ai_eda.tools.calc.radio``) skip until those parts are merged.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_eda.ir import (
    CircuitIR,
    Component,
    Evidence,
    Net,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Requirement,
    RequirementKind,
    SourceRef,
    Traced,
    ValidationResult,
    assumption,
    authoritative,
    derived,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.tools.calc.recompute import CALCULATORS
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation import rf as rfv

try:  # part P1 (the RF IR) - merged in the same wave
    from ai_eda.ir.rf import LabItem, PlanLine, RailBudget, RFBlock, RFDesign, RFExpectation, RFNetwork, RFPort, RFState
    from ai_eda.ir.simulation import AnalysisSpec
    from ai_eda.tools.spice.runner import SpiceAnalysis
except ImportError:  # pragma: no cover - before the merge
    RFDesign = None  # type: ignore[assignment,misc]

needs_rf_ir = pytest.mark.skipif(RFDesign is None, reason="needs ai_eda.ir.rf (the RF IR types) - runs after the wave-1 merge")
needs_power_calcs = pytest.mark.skipif(
    RFDesign is None or rfv.RAIL_BUDGET_CALC not in CALCULATORS or rfv.HEADROOM_CALC not in CALCULATORS,
    reason="needs ai_eda.ir.rf and calc.power.rail_budget / calc.regulator.headroom - runs after the wave-1 merge",
)

F_C = 447.5625e6
IF1 = 21.4e6
LO1 = F_C - IF1
F_R = LO1 / 12
LO2 = IF1 - 450e3
F_T = F_C / 12
USER = Provenance(kind=ProvenanceKind.USER_REQUIREMENT)
UNVERIFIED = "[UNVERIFIED: 「신고하지 아니하고 개설할 수 있는 무선국용 무선기기」; general knowledge, law.go.kr not reachable]"
DATASHEET = SourceRef(title="a grounded datasheet page", url="https://example.invalid/ds.pdf", content_hash="sha256:" + "0" * 64)


def u(value, unit=None, note=None):
    return user_requirement(value, unit, note)


def fact(value, unit=None):
    return authoritative(value, DATASHEET, unit)


def new_ir(**params: Traced) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="t", name="t"))
    ir.parameters.update(params)
    return ir


def require(ir: CircuitIR, key: str, value, unit: str | None = None, *, confirmed: bool = True) -> None:
    v = user_requirement(value, unit) if confirmed else assumption(value, "the model assumed it", unit)
    ir.requirements.requirements.append(Requirement(id=f"req.{key}", key=key, text=f"{key} {value}", kind=RequirementKind.EXPLICIT, value=v))


def by_id(results: list[ValidationResult]) -> dict[str, ValidationResult]:
    return {r.check_id: r for r in results}


def rows_of(result: ValidationResult, key: str = "id") -> dict[str, dict]:
    return {r[key]: r for r in result.details["rows"]}


def record(ir: CircuitIR, check_id: str, status: S = S.PASS, *, measured: float | None = None, unit: str | None = None, tmp: Path | None = None,
           ir_hash: str | None = "current", tool: str | None = "ngspice", state: str | None = None) -> ValidationResult:
    """A recorded result as the SPICE stage / the fixture runner stamps it, with an evidence file (and its hash) when ``tmp`` is given."""
    evidence: list[Evidence] = []
    if tmp is not None:
        path = tmp / (check_id.replace(".", "_") + ".raw")
        path.write_bytes(f"rawfile of {check_id}\n".encode())
        evidence.append(Evidence(description="rawfile", path=str(path), content_hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()))
    details: dict = {}
    if measured is not None:
        details["measured"] = measured
    if unit is not None:
        details["unit"] = unit
    if state is not None:
        details["state"] = state
    r = ValidationResult(check_id=check_id, status=status, message=f"{check_id} {status}", tool=tool, tool_version="ngspice-42" if tool else None,
                         ir_hash=ir.content_hash() if ir_hash == "current" else ir_hash, evidence=evidence, details=details)
    ir.validation.add(r)
    return r


def run(ir: CircuitIR) -> dict[str, ValidationResult]:
    return by_id(rfv.rf_results(ir))


# --------------------------------------------------------------------------- no RF IR needed


def test_the_rf_validator_is_registered_consumes_spice_and_is_silent_without_rf_content():
    v = default_registry.get(rfv.RF_TOOL)
    assert isinstance(v, rfv.RFChecksValidator) and v.consumes == frozenset({"spice"})
    ir = new_ir()
    assert not v.applies_to(ir) and v.validate(ir, ValidationContext(workdir=Path("."))) == []
    assert rfv.rf_results(ir) == [] and all(r.check_id.split(".")[0] != "rf" for r in default_registry.run(ir, ValidationContext(workdir=Path("."))))


def test_an_old_rf_result_is_retired_when_the_design_no_longer_produces_it():
    """A lab item / rail / interface removed, or ``ir.rf`` gone: the old verdict gets a superseding NOT_APPLICABLE, once."""
    ir = new_ir()
    record(ir, "rf.lab.deviation", S.NOT_VERIFIED, tool=rfv.RF_TOOL)
    record(ir, "power.headroom.U102", S.FAIL, tool=rfv.RF_TOOL)
    record(ir, "spice.rf.lpf", S.FAIL)  # the fixture runner retires its own results
    v = default_registry.get(rfv.RF_TOOL)
    assert v.applies_to(ir)
    out = v.validate(ir, ValidationContext(workdir=Path(".")))
    assert {r.check_id: (r.status, r.details["superseded"]) for r in out} == {"rf.lab.deviation": (S.NOT_APPLICABLE, "NOT_VERIFIED"),
                                                                               "power.headroom.U102": (S.NOT_APPLICABLE, "FAIL")}
    assert all(r.tool == rfv.RF_TOOL and "superseded" in r.message for r in out)
    ir.validation.extend(out)
    assert not v.applies_to(ir) and rfv.rf_results(ir) == []


def test_the_raster_arithmetic_names_the_channel():
    assert rfv.raster_channel(447.5625e6, 447.5625e6, 12.5e3) == (0, True)
    assert rfv.raster_channel(447.8625e6, 447.5625e6, 12.5e3) == (24, True)
    assert rfv.raster_channel(447.7e6, 447.5625e6, 12.5e3) == (11, True)
    n, on = rfv.raster_channel(447.57e6, 447.5625e6, 12.5e3)
    assert (n, on) == (1, False)


def test_value_in_applies_si_prefixes_and_refuses_other_units():
    assert rfv.value_in(u(2.5, "kHz"), "Hz", "x") == (2500.0, None)
    assert rfv.value_in(u(500.0, "mW"), "W", "x") == (0.5, None)
    assert rfv.value_in(u(12), None, "N") == (12.0, None)
    assert rfv.value_in(None, "Hz", "x") == (None, "x: not stated")
    v, why = rfv.value_in(u(2.5, "ppm"), "Hz", "x")
    assert v is None and "carries unit 'ppm', not Hz" in why
    v, why = rfv.value_in(u(12, "Hz"), None, "N")
    assert v is None and "dimensionless" in why


def test_basis_follows_a_derived_value_to_its_sources():
    """Confirmed = the user's or authoritative all the way down; grounded = authoritative with a source; a confirmed choice is not grounded."""
    ir = new_ir(**{"a": u(2.0, "V"), "b": fact(3.0, "V"), "c": assumption(1.0, "guess", "V")})
    ir.parameters["ab"] = derived(5.0, "calc.x", unit="V", inputs={"x": "a", "y": "b"})
    ir.parameters["bc"] = derived(4.0, "calc.x", unit="V", inputs={"x": "b", "y": "c"})
    ir.parameters["bz"] = derived(4.0, "calc.x", unit="V", inputs={"x": "b", "y": "nowhere"})
    ab = rfv.basis(ir, "ab", ir.parameters["ab"])
    assert ab.confirmed and not ab.grounded and ab.ungrounded == ["a (user_requirement)"]
    assert rfv.basis(ir, "b", ir.parameters["b"]).grounded
    bc = rfv.basis(ir, "bc", ir.parameters["bc"])
    assert not bc.confirmed and bc.weak == ["c (assumption)"]
    assert rfv.basis(ir, "bz", ir.parameters["bz"]).weak == ["nowhere (an input of bz that nothing in the IR names)"]
    no_source = Traced(value=1.0, unit="V", provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE))
    assert not rfv.basis(ir, "n", no_source).grounded and rfv.basis(ir, "n", no_source).confirmed


def test_a_calculator_this_build_lacks_is_a_reason_not_a_crash():
    why = rfv.call_calculator("calc.power.nonexistent", {}, {})
    assert why == "calc.power.nonexistent is not a registered calculator in this build"
    why = rfv.call_calculator("calc.ohms_law.I", {"v": u(1.0, "V")}, {"v": "v"})
    assert isinstance(why, str) and "are not known here" in why


def test_the_gui_validation_tab_offers_an_rf_group():
    from ai_eda.gui.page import APP_HTML, APP_JS

    assert '<option value="rf">RF (rf.* / spice.rf.* / block.interface.* / power.*)</option>' in APP_HTML
    assert '<option value="si">' in APP_HTML and '<option value="all">' in APP_HTML
    body = APP_JS.split("function drawValidation()", 1)[1].split("\n}\n", 1)[0]
    for prefix in ("'rf.'", "'spice.rf.'", "'block.interface.'", "'power.rail_budget.'", "'power.headroom.'", "'domain.rf.impedance'", "'si.rf_length'"):
        assert prefix in body, prefix


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_the_gui_group_predicate_sorts_check_ids(tmp_path: Path):
    """The ``inGroup`` lines of ``drawValidation``, run in node: 'all' keeps every row, 'si' and 'rf' keep their own."""
    from ai_eda.gui.page import APP_JS

    body = APP_JS.split("function drawValidation()", 1)[1].split("\n}\n", 1)[0]
    lines = body.split("\n")
    start = next(i for i, ln in enumerate(lines) if "const rfPrefixes" in ln)
    end = next(i for i, ln in enumerate(lines) if ln.strip().startswith("const rows ="))
    predicate = "\n".join(lines[start:end])
    ids = ["rf.freq_plan", "rf.lab.deviation", "spice.rf.lpf.s21_fc", "block.interface.IF1", "power.rail_budget.RX_5V", "power.headroom.U102",
           "domain.rf.impedance", "si.rf_length", "si.impedance.RF50", "spice.si.PG3", "spice.pm_drive_peak", "kicad.erc", "domain.power.thermal"]
    script = tmp_path / "group.js"
    script.write_text("const out = {};\nfor (const group of ['all', 'si', 'rf']) {\n" + predicate +
                      f"\n  out[group] = {ids!r}.filter((id) => inGroup({{check_id: id}}));\n}}\nconsole.log(JSON.stringify(out));\n", encoding="utf-8")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.returncode == 0, result.stderr
    import json

    got = json.loads(result.stdout)
    assert got["all"] == ids
    assert got["si"] == ["si.rf_length", "si.impedance.RF50", "spice.si.PG3"]
    assert got["rf"] == ids[:8]


# --------------------------------------------------------------------------- frequency plan


def plan(*lines, labs=()) -> RFDesign:
    return RFDesign(frequency_plan=list(lines), lab_items=[LabItem(id=x, what=f"measure {x}", reason="lab only") for x in labs])


def margin(id, f, ref, m, prov=u):
    return PlanLine(id=id, kind="margin", f_hz=prov(f, "Hz"), ref_hz=u(ref, "Hz"), min_margin_hz=u(m, "Hz"))


@needs_rf_ir
def test_margin_and_coincidence_rows_are_arithmetic_on_confirmed_frequencies():
    """The N 12 birdie margins (13 f_R 14.11 MHz from f_c; 3 f_R - 4 LO2 1.341 MHz from IF1) PASS against 1 MHz; closer FAILs; the same
    number twice is a coincidence; an assumed frequency is no verdict."""
    rf = plan(
        margin("fr13", 13 * F_R, F_C, 1e6),
        margin("fr3_lo2", 3 * F_R - 4 * LO2, IF1, 1e6),
        margin("too_close", LO1 + LO2, F_C, 1e6),
        PlanLine(id="h_if1", kind="coincidence", f_hz=u(47 * 455e3, "Hz"), ref_hz=u(IF1, "Hz")),
        PlanLine(id="mode", kind="coincidence", f_hz=u(12 * F_T, "Hz"), ref_hz=u(F_C, "Hz"), note="the TX reference's 12th harmonic in RX"),
        margin("guessed", 13 * F_R, F_C, 1e6, prov=lambda v, unit: assumption(v, "not confirmed", unit)),
    )
    ir = new_ir()
    ir.rf = rf
    r = run(ir)[rfv.FREQ_PLAN_CHECK]
    rows = rows_of(r)
    assert rows["fr13"]["status"] == "PASS" and math.isclose(rows["fr13"]["distance_hz"], 14.113541666e6, rel_tol=1e-9)
    assert "461.6760417 MHz" in rows["fr13"]["reason"] and ">= the margin 1 MHz" in rows["fr13"]["reason"]
    assert rows["fr3_lo2"]["status"] == "PASS" and math.isclose(rows["fr3_lo2"]["f_hz"], 22.740625e6, rel_tol=1e-12)
    assert math.isclose(rows["fr3_lo2"]["distance_hz"], 1.340625e6, rel_tol=1e-9)
    assert rows["too_close"]["status"] == "FAIL" and "450 kHz < the margin 1 MHz" in rows["too_close"]["reason"] and rows["too_close"]["repair"] == "human"
    assert rows["h_if1"]["status"] == "PASS" and "no coincidence" in rows["h_if1"]["reason"]
    assert rows["mode"]["status"] == "FAIL" and "coincide" in rows["mode"]["reason"] and "12th harmonic" in rows["mode"]["reason"]
    assert rows["guessed"]["status"] == "NOT_VERIFIED" and "unconfirmed guessed.f_hz (assumption)" in rows["guessed"]["reason"]
    assert r.status is S.FAIL and r.tool == rfv.RF_TOOL and r.tool_version == rfv.RF_VERSION and r.details["repair"] == "human"
    assert "not a measurement" in r.message


@needs_rf_ir
def test_response_and_gated_rows_take_their_verdict_from_what_they_point_to(tmp_path: Path):
    image_fr = PlanLine(id="image_fr", kind="response", f_hz=u(LO1 - IF1 + F_R, "Hz"), ref_hz=u(F_C, "Hz"),
                        points_to=["spice.rf.fe_bpf3", "rf.lab.lo_spur_response"])
    gated = PlanLine(id="mode", kind="gated", f_hz=u(12 * F_T, "Hz"), ref_hz=u(F_C, "Hz"), points_to=["spice.tx_rail_off_rx"])
    ir = new_ir()
    ir.rf = plan(image_fr, gated, labs=["lo_spur_response"])
    rows = rows_of(run(ir)[rfv.FREQ_PLAN_CHECK])
    assert math.isclose(rows["image_fr"]["f_hz"], 440.2760417e6, rel_tol=1e-9) and "-7.286458333 MHz from 447.5625 MHz" in rows["image_fr"]["reason"]
    assert rows["image_fr"]["pointed"] == {"spice.rf.fe_bpf3": "NOT_VERIFIED", "rf.lab.lo_spur_response": "NOT_VERIFIED"}
    assert "spice.rf.fe_bpf3: no result recorded" in rows["image_fr"]["reason"] and "rf.lab.lo_spur_response: no lab evidence" in rows["image_fr"]["reason"]
    assert rows["mode"]["status"] == "NOT_VERIFIED"
    record(ir, "spice.rf.fe_bpf3", S.PASS)
    record(ir, "spice.tx_rail_off_rx", S.PASS)
    r = run(ir)[rfv.FREQ_PLAN_CHECK]
    rows = rows_of(r)
    assert rows["image_fr"]["status"] == "NOT_VERIFIED" and rows["image_fr"]["pointed"]["spice.rf.fe_bpf3"] == "PASS"
    assert rows["mode"]["status"] == "PASS" and r.status is S.NOT_VERIFIED
    for status, kw, expect in ((S.FAIL, {}, "FAIL"), (S.PASS, {"ir_hash": "sha256:other"}, "NOT_VERIFIED"), (S.PASS, {"tool": None}, "NOT_VERIFIED"),
                               (S.NOT_APPLICABLE, {}, "NOT_VERIFIED"), (S.UNRESOLVED, {}, "UNRESOLVED")):
        record(ir, "spice.tx_rail_off_rx", status, **kw)
        assert rows_of(run(ir)[rfv.FREQ_PLAN_CHECK])["mode"]["status"] == expect, (status, kw)
    ir.rf.frequency_plan.append(PlanLine(id="typo", kind="gated", f_hz=u(F_C, "Hz"), points_to=["rf.lab.nowhere"]))
    assert "rf.lab.nowhere: names no lab item of the design" in rows_of(run(ir)[rfv.FREQ_PLAN_CHECK])["typo"]["reason"]


# --------------------------------------------------------------------------- regulatory profile


PROFILE = {
    "kr447.band_low": (447.5625e6, "Hz"), "kr447.band_high": (447.8625e6, "Hz"), "kr447.channel_raster": (12.5e3, "Hz"),
    "kr447.max_power": (0.5, "W"), "kr447.max_deviation": (2.5e3, "Hz"), "kr447.max_obw": (8.5e3, "Hz"), "kr447.freq_tolerance": (2.5, "ppm"),
    "kr447.emission": ("F3E", None), "kr447.conformity": ("KC 적합인증 before any transmission", None),
}


def profile_ir(*, confirmed: bool = True) -> CircuitIR:
    ir = new_ir()
    for key, (value, unit) in PROFILE.items():
        ir.parameters[key] = user_requirement(value, unit, f"{key}: choice {UNVERIFIED}") if confirmed else assumption(value, f"default {UNVERIFIED}", unit)
    ir.rf = RFDesign(profile_keys=list(PROFILE))
    return ir


@needs_rf_ir
def test_the_regulatory_profile_never_passes_and_fails_a_confirmed_exceedance():
    ir = profile_ir()
    for key, value, unit in (("tx_power", 0.5, "W"), ("frequency_deviation", 2.5e3, "Hz"), ("occupied_bandwidth", 8.0e3, "Hz"),
                             ("frequency_tolerance", 2.5, "ppm"), ("carrier_frequency", F_C, "Hz"), ("erp", 0.4, "W")):
        require(ir, key, value, unit)
    r = run(ir)[rfv.PROFILE_CHECK]
    rows = rows_of(r, "key")
    assert r.status is S.NOT_VERIFIED and rfv.UNVERIFIED_PROFILE in r.message
    assert all(row["status"] == "NOT_VERIFIED" for row in rows.values()), rows
    assert "within an UNVERIFIED placeholder (never a PASS)" in rows["kr447.max_power"]["reason"] and UNVERIFIED in rows["kr447.max_power"]["source"]
    assert "channel 1 of 25" in next(row["reason"] for row in r.details["rows"] if row.get("check") == "raster")
    assert "not compared (no requirement rule for it" in rows["kr447.emission"]["reason"] and "not compared" in rows["erp"]["reason"]
    # 27 dBm is 0.501 W: over the 0.5 W choice; 3 kHz deviation, a 9 kHz OBW and 3 ppm likewise
    ir = profile_ir()
    for key, value, unit in (("tx_power", "27 dBm", None), ("frequency_deviation", 3e3, "Hz"), ("occupied_bandwidth", 9e3, "Hz"), ("frequency_tolerance", 3.0, "ppm")):
        require(ir, key, value, unit)
    r = run(ir)[rfv.PROFILE_CHECK]
    rows = rows_of(r, "key")
    assert r.status is S.FAIL and r.details["repair"] == "human"
    assert rows["kr447.max_power"]["status"] == "FAIL" and math.isclose(rows["kr447.max_power"]["value"], 10 ** (27 / 10) / 1000, rel_tol=1e-12)
    assert "exceeds it - two confirmed choices contradict each other (not a legal verdict)" in rows["kr447.max_power"]["reason"]
    assert all(rows[k]["status"] == "FAIL" for k in ("kr447.max_deviation", "kr447.max_obw", "kr447.freq_tolerance"))
    band = next(row for row in r.details["rows"] if row.get("check") is None and row["key"].startswith("kr447.band_low"))
    assert band["status"] == "NOT_APPLICABLE" and "no confirmed carrier_frequency" in band["reason"]


@needs_rf_ir
def test_the_design_librarys_own_profile_placeholders_are_what_the_profile_check_reads():
    """The ``kr447.*`` parameters ``ai_eda.design.rf.profile`` makes (the templates' confirmation rows) feed ``rf.regulatory_profile``."""
    from ai_eda.design.rf import profile

    def ir_of(confirmed: bool) -> CircuitIR:
        ir = new_ir(**{c.key: t for c, t in profile.profile_choices("kr447_tx_exciter", confirmed)})
        ir.rf = RFDesign(profile_keys=profile.profile_keys())
        return ir

    ir = ir_of(True)
    for key, value, unit in (("tx_power", 0.5, "W"), ("frequency_deviation", 2.5e3, "Hz"), ("carrier_frequency", F_C, "Hz")):
        require(ir, key, value, unit)
    r = run(ir)[rfv.PROFILE_CHECK]
    rows = rows_of(r, "key")
    assert r.status is S.NOT_VERIFIED and rfv.UNVERIFIED_PROFILE in r.message
    assert {"kr447.max_power", "kr447.max_deviation", "kr447.max_obw", "kr447.freq_tolerance"} <= set(rows)
    assert all(row["status"] != "PASS" for row in r.details["rows"]), r.details["rows"]
    assert "channel 1 of 25" in next(row["reason"] for row in r.details["rows"] if row.get("check") == "raster")
    ir = ir_of(True)
    require(ir, "tx_power", "27 dBm")
    r = run(ir)[rfv.PROFILE_CHECK]
    assert r.status is S.FAIL and rows_of(r, "key")["kr447.max_power"]["status"] == "FAIL"
    ir = ir_of(False)  # an unconfirmed placeholder is an assumption: no exceedance is claimed on it
    require(ir, "tx_power", "27 dBm")
    r = run(ir)[rfv.PROFILE_CHECK]
    assert r.status is not S.PASS and rows_of(r, "key")["kr447.max_power"]["status"] != "PASS"


@needs_rf_ir
def test_a_band_is_not_a_carrier_and_the_carrier_must_sit_on_the_raster():
    def raster_row(value, confirmed=True):
        ir = profile_ir(confirmed=confirmed)
        require(ir, "carrier_frequency", value, "Hz")
        r = run(ir)[rfv.PROFILE_CHECK]
        return r, {row["check"]: row for row in r.details["rows"] if "check" in row}

    r, rows = raster_row(447.8625e6)
    assert rows["raster"]["status"] == "NOT_VERIFIED" and rows["raster"]["channel"] == 25 and rows["raster"]["channels"] == 25 and r.status is S.NOT_VERIFIED
    r, rows = raster_row(447.0e6)
    assert r.status is S.FAIL and rows["band"]["status"] == "FAIL" and "raster" not in rows
    assert "447 MHz is outside the profile band 447.5625 MHz - 447.8625 MHz (a band is not a carrier frequency: state the channel)" in rows["band"]["reason"]
    r, rows = raster_row(447.57e6)
    assert rows["band"]["status"] == "NOT_VERIFIED" and rows["raster"]["status"] == "FAIL"
    assert "the nearest channels are 447.5625 MHz and 447.575 MHz" in rows["raster"]["reason"]
    r, rows = raster_row(447.0e6, confirmed=False)
    assert r.status is S.NOT_VERIFIED and "rests on unconfirmed" in rows["band"]["reason"]


@needs_rf_ir
def test_an_unusable_or_unconfirmed_requirement_is_named_not_compared():
    ir = profile_ir()
    require(ir, "tx_power", 0.6, "W", confirmed=False)
    rows = rows_of(run(ir)[rfv.PROFILE_CHECK], "key")
    assert rows["kr447.max_power"]["status"] == "NOT_VERIFIED" and "tx_power is not usable" in rows["kr447.max_power"]["reason"]
    ir.rf.profile_keys.append("kr447.max_erp_typo")
    rows = rows_of(run(ir)[rfv.PROFILE_CHECK], "key")
    assert "has no value in ir.parameters" in rows["kr447.max_erp_typo"]["reason"]


# --------------------------------------------------------------------------- model grounding and lab items


@needs_rf_ir
def test_model_grounding_names_every_ungrounded_model_value():
    ir = new_ir(**{"model.l_q": u(40.0, None, "inductor Q [UNVERIFIED]"), "model.varactor": u(".model DVAR D (CJO=20p VJ=0.7 M=0.5)"),
                   "model.opamp.gbw": fact(1e6, "Hz"), "model.pin.r_on": assumption(1.0, "not confirmed", "ohm")})
    ir.rf = RFDesign(model_values=["model.l_q", "model.varactor", "model.opamp.gbw", "model.xtal21.cm"])
    r = run(ir)[rfv.MODEL_CHECK]
    rows = rows_of(r, "key")
    assert r.status is S.NOT_VERIFIED and rfv.UNDER_MODEL_VALUES in r.message
    assert rows["model.l_q"]["reason"] == "a confirmed model choice, not grounded"
    assert rows["model.opamp.gbw"]["status"] == "PASS" and "a grounded datasheet page" in rows["model.opamp.gbw"]["reason"]
    assert "has no value in ir.parameters" in rows["model.xtal21.cm"]["reason"]
    assert rows["model.pin.r_on"]["listed"] is False and "not listed in ir.rf.model_values" in rows["model.pin.r_on"]["reason"]
    assert "4 of 5 model value(s) not grounded" in r.message
    ir = new_ir(**{"model.opamp.gbw": fact(1e6, "Hz")})
    ir.rf = RFDesign(model_values=["model.opamp.gbw"])
    assert run(ir)[rfv.MODEL_CHECK].status is S.PASS
    ir.rf = RFDesign()
    assert rfv.MODEL_CHECK in run(ir), "an unlisted model.* parameter is still a model value"
    ir.parameters.clear()
    assert rfv.MODEL_CHECK not in run(ir)


@needs_rf_ir
def test_model_grounding_names_a_listed_card_as_a_card():
    """A model card (SPICE text in the confirm_design table, no number in ir.parameters) is named as a card by the parts bound to it, never grounded."""
    from ai_eda.ir import SpiceBinding, SpiceDevice

    ir = new_ir(**{"model.opamp.gbw": u(1e6, "Hz", "op-amp GBW [UNVERIFIED]")})
    card = Traced(value=".subckt OPA1P inp inn out\nEo out 0 inp inn 100k\n.ends",
                  provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="confirmed: model.opamp: generic op-amp macro [UNVERIFIED]"))
    ir.components = [Component(ref="U1", value="MCP6001", provenance=USER,
                               spice=SpiceBinding(device=SpiceDevice.X, model_name="OPA1P", model_card=card, pin_order=["3", "4", "1"], provenance=USER))]
    ir.rf = RFDesign(model_values=["model.opamp", "model.opamp.gbw", "model.xtal21"])
    rows = rows_of(run(ir)[rfv.MODEL_CHECK], "key")
    assert rows["model.opamp"]["status"] == "NOT_VERIFIED" and rows["model.opamp"]["card"] is True and rows["model.opamp"]["bound_by"] == ["U1"]
    assert "is a model card" in rows["model.opamp"]["reason"] and "has no value" not in rows["model.opamp"]["reason"]
    assert "has no value in ir.parameters" in rows["model.xtal21"]["reason"] and "card" not in rows["model.xtal21"]  # listed, bound by nothing


@needs_rf_ir
def test_lab_items_are_never_verified_here():
    ir = new_ir()
    ir.rf = RFDesign(lab_items=[LabItem(id="deviation", block="tx_chain", what="peak deviation and modulation response", instruments=["modulation analyser"],
                                        reason="the real deviation needs the built exciter"),
                                LabItem(id="uvlo", what="the low-pack inhibit's real trip point", reason="U204 has no model")],
                     blocks=[RFBlock(id="tx_chain")])
    out = run(ir)
    assert {k for k in out if k.startswith("rf.lab.")} == {"rf.lab.deviation", "rf.lab.uvlo"}
    r = out["rf.lab.deviation"]
    assert r.status is S.NOT_VERIFIED and r.message == ("no lab evidence: peak deviation and modulation response [modulation analyser] - "
                                                        "the real deviation needs the built exciter")
    assert r.details["block"] == "tx_chain" and "no lab-evidence importer" in r.details["evidence_path"]


# --------------------------------------------------------------------------- deviation chain

#: one tank of kr447/decided/pm_n12.cir under the model values: phase at f_T per bias state
PM_PHASES = {"bias_lo": (1.44, -21.43), "bias_nom": (2.00, 1.39), "bias_hi": (2.56, 18.21)}


def pm_network(id: str) -> RFNetwork:
    ports = [RFPort(name="TCXO", net=f"{id}_IN", kind="port", z0_ohm=u(50.0, "ohm")), RFPort(name="BIAS", net=f"{id}_BIAS", kind="port", z0_ohm=u(10.0, "ohm")),
             RFPort(name="TANK", net=f"{id}_TANK", kind="probe")]
    states = [RFState(id=s, port_dc_v={"BIAS": u(v, "V")}) for s, (v, _) in PM_PHASES.items()]
    exps = [RFExpectation(id=f"phi_{s.split('_')[1]}", state=s, quantity="phase21_deg", drive="TCXO", to="TANK", at=u(F_T, "Hz"), nominal=u(phi, "deg"),
                          tol_abs=u(1.0, "deg")) for s, (_, phi) in PM_PHASES.items()]
    sweep = [AnalysisSpec(id="ac1", kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(201), "fstart": u(36e6, "Hz"), "fstop": u(38e6, "Hz")}, provenance=USER)]
    return RFNetwork(id=id, members=[f"L_{id}", f"C_{id}", f"D_{id}"], ports=ports, states=states, sweep=sweep, expectations=exps)


TAU_I = 0.960e-3
COUPLE_DB = {"spice.pm_couple_300": -0.4, "spice.pm_couple_1k": -0.2, "spice.pm_couple_3k": -0.6}
V_MAX = 0.9


def deviation_ir(tmp: Path, *, v_max: float = V_MAX, deviation: float | None = 2.5e3, skip: tuple[str, ...] = ()) -> CircuitIR:
    ir = new_ir(**{"rf.n_mult": u(12), "tx.tau_i": u(TAU_I, "s")})
    ir.rf = RFDesign(networks=[pm_network("pm_mod1"), pm_network("pm_mod2")])
    if deviation is not None:
        require(ir, "frequency_deviation", deviation, "Hz")
    for net in ("pm_mod1", "pm_mod2"):
        for s, (_, phi) in PM_PHASES.items():
            record(ir, f"spice.rf.{net}.{s}.phi_{s.split('_')[1]}", measured=phi, unit="deg", tmp=tmp, state=s)
    for check, db in COUPLE_DB.items():
        if check not in skip:
            record(ir, check, measured=db, unit="dB", tmp=tmp)
    if "spice.pm_drive_peak" not in skip:
        record(ir, "spice.pm_drive_peak", measured=v_max, unit="V", tmp=tmp)
    if "spice.integrator_1k" not in skip:
        record(ir, "spice.integrator_1k", measured=-15.6, unit="dB", tmp=tmp)
    return ir


def expected_delta(v_max: float = V_MAX) -> tuple[float, float]:
    slope = math.radians(18.21 - (-21.43)) / (2.56 - 1.44)
    a = 10 ** (max(COUPLE_DB.values()) / 20)
    return slope, 12 * 2 * slope * a * v_max / (2 * math.pi * TAU_I)


@needs_rf_ir
def test_the_deviation_chain_passes_only_when_every_factor_passed_on_this_design(tmp_path: Path):
    ir = deviation_ir(tmp_path)
    r = run(ir)[rfv.DEVIATION_CHECK]
    slope, delta = expected_delta()
    assert math.isclose(slope, 0.61772, rel_tol=1e-4)
    assert r.status is S.PASS, r.message
    assert math.isclose(r.details["delta_f_hz"], delta, rel_tol=1e-12) and r.details["requirement_hz"] == 2.5e3
    assert rfv.UNDER_MODEL_VALUES in r.message and "the real deviation is a lab item" in r.message
    factors = {f["factor"]: f for f in r.details["factors"]}
    assert factors["N"]["value"] == 12 and math.isclose(factors["K_pm"]["value"], 2 * slope, rel_tol=1e-12)
    assert math.isclose(factors["a"]["value"], 10 ** (-0.2 / 20), rel_tol=1e-12) and "spice.pm_couple_1k" in factors["a"]["source"]
    assert factors["V_max"]["value"] == V_MAX and factors["tau_i"]["source"] == "tx.tau_i, checked by spice.integrator_1k"
    tank = r.details["tanks"][0]
    assert tank["port"] == "BIAS" and tank["v_lo"] == 1.44 and tank["v_hi"] == 2.56 and math.isclose(tank["linearity"], -0.1514, abs_tol=1e-4)
    assert r.evidence and all(e.path and e.content_hash for e in r.evidence) and len(r.evidence) == 2 * 2 + 3 + 1 + 1
    assert r.tool == rfv.RF_TOOL


@needs_rf_ir
def test_verified_factors_that_multiply_past_the_requirement_fail(tmp_path: Path):
    ir = deviation_ir(tmp_path, v_max=1.2)
    r = run(ir)[rfv.DEVIATION_CHECK]
    _, delta = expected_delta(1.2)
    assert delta > 2.5e3 and r.status is S.FAIL and r.details["repair"] == "human" and "the verified factors over-deviate" in r.message
    ir = deviation_ir(tmp_path, deviation=None)
    r = run(ir)[rfv.DEVIATION_CHECK]
    assert r.status is S.NOT_VERIFIED and "no confirmed frequency_deviation requirement" in r.message


@pytest.mark.parametrize("spoil, why", [
    ("stale", "spice.pm_drive_peak: judged on another design version"),
    ("failed", "spice.integrator_1k is FAIL"),
    ("tampered", "does not match its recorded hash"),
    ("missing", "spice.pm_couple_3k: no result recorded"),
    ("no_evidence", "spice.pm_drive_peak: no evidence file with a recorded hash"),
    ("guessed_n", "rf.n_mult rests on unconfirmed rf.n_mult (assumption)"),
    ("one_state", "network pm_mod2 has no state bias_hi"),
    ("flat", "the tanks' phase does not move with the bias"),
])
@needs_rf_ir
def test_a_factor_that_is_not_a_pass_about_this_design_is_no_verdict(tmp_path: Path, spoil: str, why: str):
    ir = deviation_ir(tmp_path, skip=("spice.pm_couple_3k",) if spoil == "missing" else ())
    if spoil == "stale":
        record(ir, "spice.pm_drive_peak", measured=V_MAX, unit="V", tmp=tmp_path, ir_hash="sha256:older")
    elif spoil == "failed":
        record(ir, "spice.integrator_1k", S.FAIL, measured=-19.0, unit="dB", tmp=tmp_path)
    elif spoil == "tampered":
        Path(ir.validation.latest("spice.rf.pm_mod1.bias_hi.phi_hi").evidence[0].path).write_bytes(b"edited")
    elif spoil == "no_evidence":
        record(ir, "spice.pm_drive_peak", measured=V_MAX, unit="V")
    elif spoil == "guessed_n":
        ir.parameters["rf.n_mult"] = assumption(12, "not confirmed")
    elif spoil == "flat":
        for net in ("pm_mod1", "pm_mod2"):
            for state, exp in (("bias_lo", "phi_lo"), ("bias_hi", "phi_hi")):
                record(ir, f"spice.rf.{net}.{state}.{exp}", measured=1.39, unit="deg", tmp=tmp_path, state=state)
    elif spoil == "one_state":
        net = ir.rf.networks[1]
        net.expectations = [e for e in net.expectations if e.state != "bias_hi"]
        net.states = [s for s in net.states if s.id != "bias_hi"]
        for check in [r.check_id for r in ir.validation.results]:
            ir.validation.latest(check).ir_hash = ir.content_hash()
    r = run(ir)[rfv.DEVIATION_CHECK]
    assert r.status is S.NOT_VERIFIED and why in r.message, r.message
    assert "the real deviation is a lab item" in r.message


def _two_node_network(id: str, rows: list[tuple[str, str, str]]) -> RFNetwork:
    """``pm_network`` with a second probe (OUT) and the phase rows ``(expectation id, state, port read)`` at f_T."""
    base = pm_network(id)
    ports = [*base.ports, RFPort(name="OUT", net=f"{id}_OUT", kind="probe")]
    exps = [RFExpectation(id=eid, state=state, quantity="phase21_deg", drive="TCXO", to=to, at=u(F_T, "Hz"), nominal=u(PM_PHASES[state][1], "deg"),
                          tol_abs=u(1.0, "deg")) for eid, state, to in rows]
    return base.model_copy(update={"ports": ports, "expectations": exps})


@needs_rf_ir
def test_a_chord_slope_is_never_taken_between_two_different_nodes(tmp_path: Path):
    """bias_lo read at TANK and bias_hi read at OUT (same drive, same f_T): a phase difference between two nodes, no tank slope - K_pm is
    NOT_VERIFIED naming both rows (it was PASS at 'K_pm 0.784775 rad/V'); a network that reads both nodes in each state has 2 pairs, not 4."""
    ir = deviation_ir(tmp_path)
    ir.rf.networks[0] = _two_node_network("pm_mod1", [("phi_lo", "bias_lo", "TANK"), ("phi_hi", "bias_hi", "OUT")])
    record(ir, "spice.rf.pm_mod1.bias_lo.phi_lo", measured=-21.43, unit="deg", tmp=tmp_path, state="bias_lo")
    record(ir, "spice.rf.pm_mod1.bias_hi.phi_hi", measured=18.21 - 90.0, unit="deg", tmp=tmp_path, state="bias_hi")
    for r in ir.validation.results:
        r.ir_hash = ir.content_hash()
    r = run(ir)[rfv.DEVIATION_CHECK]
    assert r.status is S.NOT_VERIFIED, r.message
    assert "no bias_lo / bias_hi phase21_deg expectation pair(s) reading the same drive -> to" in r.message
    assert "phi_lo (TCXO->TANK" in r.message and "phi_hi (TCXO->OUT" in r.message and "delta_f_hz" not in r.details
    both = _two_node_network("pm_mod1", [("lo_t", "bias_lo", "TANK"), ("lo_o", "bias_lo", "OUT"), ("hi_t", "bias_hi", "TANK"), ("hi_o", "bias_hi", "OUT")])
    pairs, _ = rfv._phase_pairs(both)
    assert sorted((a.id, b.id) for a, b in pairs) == [("lo_o", "hi_o"), ("lo_t", "hi_t")]
    # a recorded result that read other ports than its expectation names is not that expectation's number
    ir = deviation_ir(tmp_path)
    ir.validation.latest("spice.rf.pm_mod1.bias_hi.phi_hi").details.update(drive="TCXO", to="OUT")
    r = run(ir)[rfv.DEVIATION_CHECK]
    assert r.status is S.NOT_VERIFIED and "records TCXO->OUT, not the expectation's TCXO->TANK" in r.message, r.message


@needs_rf_ir
def test_the_audio_board_deviation_rests_on_the_model_constant_and_is_never_a_verdict(tmp_path: Path):
    ir = new_ir(**{"rf.n_mult": u(12), "tx.tau_i": u(TAU_I, "s"), "model.k_pm": u(1.257, "rad/V")})
    ir.rf = RFDesign(model_values=["model.k_pm"])
    require(ir, "frequency_deviation", 2.5e3, "Hz")
    record(ir, "spice.pm_drive_peak", measured=V_MAX, unit="V", tmp=tmp_path)
    record(ir, "spice.integrator_1k", measured=-15.6, unit="dB", tmp=tmp_path)
    r = run(ir)[rfv.DEVIATION_CHECK]
    assert r.status is S.NOT_VERIFIED and "K_pm is the model constant model.k_pm, not a verified network slope" in r.message
    assert "spice.pm_couple_1k: no result recorded" in r.message and "delta_f_hz" not in r.details
    assert rfv.DEVIATION_CHECK not in run(new_ir_with_rf(RFDesign())), "no phase modulator, no deviation chain"


def new_ir_with_rf(rf) -> CircuitIR:
    ir = new_ir()
    ir.rf = rf
    return ir


# --------------------------------------------------------------------------- block interfaces


def block(id: str, *ports: RFPort) -> RFBlock:
    return RFBlock(id=id, refs=[f"U_{id}"], ports=list(ports))


def iface_ir(*blocks: RFBlock, nets=("IF1", "RX_5V", "LO1_MIX")) -> CircuitIR:
    ir = new_ir()
    ir.nets = [Net(name=n, provenance=USER) for n in nets]
    ir.rf = RFDesign(blocks=list(blocks))
    return ir


@needs_rf_ir
def test_block_interfaces_compare_impedance_frequency_rail_voltage_and_direction():
    fe = block("rx_frontend", RFPort(name="IF1_OUT", net="IF1", kind="port", z0_ohm=u(50.0, "ohm"), frequency_hz=u(IF1, "Hz"), direction="out"),
               RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=u(5.0, "V"), direction="in"),
               RFPort(name="LO", net="LO1_MIX", kind="port", z0_ohm=u(50.0, "ohm"), direction="in"))
    be = block("if_backend", RFPort(name="IF1_IN", net="IF1", kind="port", z0_ohm=u(50.0, "ohm"), frequency_hz=u(21.4e6, "Hz"), direction="in"),
               RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=u(5.0, "V"), direction="in"),
               RFPort(name="ALONE", net="LO1_MIX", kind="port", z0_ohm=u(50.0, "ohm"), direction="in"))
    power = block("power", RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=u(5.0, "V"), direction="out"))
    out = run(iface_ir(fe, be, power))
    assert out["block.interface.IF1"].status is S.PASS and "IR arithmetic" in out["block.interface.IF1"].message
    assert out["block.interface.RX_5V"].status is S.PASS and "driven by power.RX_5V" in out["block.interface.RX_5V"].message
    lo = out["block.interface.LO1_MIX"]
    assert lo.status is S.NOT_VERIFIED and "no block port drives LO1_MIX" in lo.message
    bad_be = block("if_backend", RFPort(name="IF1_IN", net="IF1", kind="port", z0_ohm=u(75.0, "ohm"), direction="out"),
                   RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=u(3.3, "V"), direction="out"))
    out = run(iface_ir(fe, bad_be, power))
    r = out["block.interface.IF1"]
    assert r.status is S.FAIL and "z0 disagrees: rx_frontend.IF1_OUT 50 ohm, if_backend.IF1_IN 75 ohm" in r.message
    assert "2 outputs drive IF1" in r.message and "frequency stated by rx_frontend.IF1_OUT but not by if_backend.IF1_IN" in r.message
    assert out["block.interface.RX_5V"].status is S.FAIL and "voltage disagrees" in out["block.interface.RX_5V"].message
    out = run(iface_ir(fe, be, power, nets=("RX_5V", "LO1_MIX")))
    assert out["block.interface.IF1"].status is S.FAIL and "net IF1 is not a net of the design" in out["block.interface.IF1"].message
    assert not any(k.startswith("block.interface.") for k in run(iface_ir(power)))


# --------------------------------------------------------------------------- rails


def power_ir(*rails: RailBudget, pack: float | None = 6.4, wiring: dict[str, list[str]] | None = None) -> CircuitIR:
    ir = new_ir()
    if pack is not None:
        ir.parameters["power.pack_cutoff_v"] = u(pack, "V", "the pack is used only above it (confirmed choice)")
    refs = sorted({r.regulator_ref for r in rails})
    ir.components = [Component(ref=ref, value="LDO", provenance=USER) for ref in refs]
    wiring = wiring or {}
    nets: dict[str, list[PinRef]] = {}
    for r in rails:
        nets.setdefault(r.rail, []).append(PinRef(component_ref=r.regulator_ref, pin_number="1"))
    for ref, names in wiring.items():
        for i, n in enumerate(names):
            nets.setdefault(n, []).append(PinRef(component_ref=ref, pin_number=str(10 + i)))
    ir.nets = [Net(name=n, pins=pins, provenance=USER) for n, pins in nets.items()]
    ir.rf = RFDesign(rails=list(rails))
    return ir


def lm1117(v=u, **over) -> RailBudget:
    kw = dict(rail="TX_5V", regulator_ref="U102", v_out=u(5.0, "V"), i_min=v(0.287, "A"), i_max=v(0.441, "A"), i_rating=v(0.8, "A"),
              dropout_v=v(1.2, "V"), path_r_ohm=v(0.17, "ohm"))
    kw.update(over)
    return RailBudget(**kw)


@needs_power_calcs
def test_the_lm1117_headroom_is_positive_at_the_cutoff_but_ungrounded_and_negative_at_6_v():
    out = run(power_ir(lm1117()))
    h = out["power.headroom.U102"]
    assert h.status is S.NOT_VERIFIED and math.isclose(h.details["headroom_v"], 6.4 - 0.441 * 0.17 - 1.2 - 5.0, rel_tol=1e-12)
    assert "+0.125 V" in h.message and "ungrounded" in h.message and h.details["v_in_from"] == "power.pack_cutoff_v"
    assert h.details["calculator"] == "calc.regulator.headroom"
    out = run(power_ir(lm1117(), pack=6.0))
    h = out["power.headroom.U102"]
    assert h.status is S.FAIL and math.isclose(h.details["headroom_v"], -0.27497, rel_tol=1e-9) and h.details["repair"] == "human"
    assert "out of regulation at the minimum input" in h.message
    out = run(power_ir(lm1117(v=fact)))
    assert out["power.headroom.U102"].status is S.PASS and out["power.rail_budget.TX_5V"].status is S.PASS
    out = run(power_ir(lm1117(), pack=None))
    assert out["power.headroom.U102"].status is S.NOT_VERIFIED and "no power.pack_cutoff_v" in out["power.headroom.U102"].message


@needs_power_calcs
def test_an_unstated_dropout_only_bounds_the_headroom():
    out = run(power_ir(lm1117(dropout_v=None)))
    h = out["power.headroom.U102"]
    assert h.status is S.NOT_VERIFIED and "the dropout not stated" in h.message and h.details["unstated"] == ["the dropout"]
    out = run(power_ir(lm1117(dropout_v=None, path_r_ohm=None), pack=4.9))
    h = out["power.headroom.U102"]
    assert h.status is S.FAIL and "(with the dropout and the path resistance at 0, an upper bound)" in h.message


@needs_power_calcs
def test_a_regulator_fed_from_another_rail_is_judged_at_that_rails_output():
    rx5 = RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.086, "A"), i_max=u(0.161, "A"), i_rating=u(0.5, "A"), dropout_v=u(0.45, "V"))
    rx33 = RailBudget(rail="RX_3V3", regulator_ref="U103", v_out=u(3.3, "V"), i_min=u(0.01, "A"), i_max=u(0.03, "A"), i_rating=u(0.25, "A"), dropout_v=u(0.25, "V"),
                      path_r_ohm=u(0.0, "ohm"))
    out = run(power_ir(rx5, rx33, wiring={"U103": ["RX_5V"]}))
    h = out["power.headroom.U103"]
    assert h.details["v_in_from"] == "rf.rails[RX_5V].v_out" and math.isclose(h.details["headroom_v"], 5.0 - 0.25 - 3.3, rel_tol=1e-12)
    assert "the output of U101 (RX_5V)" in h.message
    out = run(power_ir(rx5, rx33, lm1117(), wiring={"U103": ["RX_5V", "TX_5V"]}))
    assert out["power.headroom.U103"].status is S.NOT_VERIFIED and "touch several budget rails (RX_5V, TX_5V)" in out["power.headroom.U103"].message


@needs_power_calcs
def test_a_rail_budget_fails_a_confirmed_overload_and_passes_only_grounded_currents():
    lp2985 = RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.091, "A"), i_max=u(0.171, "A"), i_rating=u(0.150, "A"))
    out = run(power_ir(lp2985))
    b = out["power.rail_budget.RX_5V"]
    assert b.status is S.FAIL and math.isclose(b.details["margin_a"], -0.021, rel_tol=1e-9) and "the load exceeds the rating" in b.message
    assert b.details["calculator"] == "calc.power.rail_budget"
    guessed = RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.091, "A"), i_max=assumption(0.171, "guess", "A"), i_rating=u(0.150, "A"))
    assert run(power_ir(guessed))["power.rail_budget.RX_5V"].status is S.NOT_VERIFIED
    lp38693 = RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.086, "A"), i_max=u(0.161, "A"), i_rating=u(0.5, "A"))
    b = run(power_ir(lp38693))["power.rail_budget.RX_5V"]
    assert b.status is S.NOT_VERIFIED and "ungrounded" in b.message and math.isclose(b.details["margin_a"], 0.339, rel_tol=1e-9)
    no_rating = RailBudget(rail="RX_5V", regulator_ref="U101", v_out=u(5.0, "V"), i_min=u(0.086, "A"), i_max=u(0.161, "A"))
    assert "no current rating stated" in run(power_ir(no_rating))["power.rail_budget.RX_5V"].message


# --------------------------------------------------------------------------- wiring through the registry


@needs_rf_ir
def test_the_fixture_prefix_is_the_fixture_runners_and_the_registry_runs_the_checks(tmp_path: Path):
    from ai_eda.tools.spice.stage import RF_CHECK_PREFIX

    assert rfv.FIXTURE_PREFIX == RF_CHECK_PREFIX
    ir = new_ir()
    ir.rf = RFDesign(lab_items=[LabItem(id="obw", what="occupied bandwidth", reason="lab only")],
                     frequency_plan=[margin("fr13", 13 * F_R, F_C, 1e6)])
    results = default_registry.run(ir, ValidationContext(workdir=tmp_path))
    ids = {r.check_id for r in results}
    assert {"rf.lab.obw", "rf.freq_plan"} <= ids
    ir.validation.extend(results)
    ir.rf.lab_items = []
    later = by_id(default_registry.run(ir, ValidationContext(workdir=tmp_path)))
    assert later["rf.lab.obw"].status is S.NOT_APPLICABLE and later["rf.freq_plan"].status is S.PASS
