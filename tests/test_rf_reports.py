"""The RF sections of the Korean stage reports (kr447 design §5, wave 2 part P14): the banner, the RF theory, the fixture verdicts, the S-parameter charts.

A synthetic RF design (no KiCad): three fixture networks of the fixture
runner's own test boards - the Q_e 20 / Q_u 40 double-tuned top-C tank of
``tests/test_rf_fixture.py``, a 6 dB pi pad of E-series-like resistors and a
phase network with two bias states read at a high-impedance probe - with a
frequency plan, lab items, a rail budget, blocks, a keep-out, the kr447
profile and model values as confirmed choices. The fixtures run on an ac
solver that writes a genuine ngspice ASCII rawfile (:class:`RawFakeAC`), and
on ngspice itself where it is installed. What is proved:

* a design without ``ir.rf`` gets no RF text and no banner;
* every one of the four reports of a design with a regulatory profile starts
  with the UNVERIFIED banner (every ``kr447.*`` key named, the KC sentence,
  the recorded ``rf.regulatory_profile`` status copied);
* the theory section carries the plan (with the recorded row statuses), the
  profile table (each document as ``[UNVERIFIED]``), the model-value list
  (the ``rf.model_grounding`` rows) and every fixture row's nominal, rule and
  origin;
* every status is copied: a status changed in the record changes the report;
* the circuit report copies each fixture row's recorded measurement and
  status and draws the S-parameter charts from the rawfiles - the curve at a
  row's own frequency is the recorded measurement - and draws nothing from a
  summary about another design, a rawfile changed or removed since, or a
  summary without a tool;
* the final report lists the RF results, the lab items and the rail budget;
* a one-sided ``bound`` row is a one-sided limit in the tolerance chart and in
  the theory / final tables;
* the reports are byte-identical for the same IR and files and name files by
  basename only.
"""

from __future__ import annotations

import cmath
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import pytest

from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design.rf.models import MODEL_VALUES, model_choice
from ai_eda.design.rf.profile import profile_choices, profile_keys
from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Expectation,
    Keepout,
    PCBDesign,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    Reduce,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Traced,
    ValidationResult,
    user_requirement,
)
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.rf import LabItem, PlanLine, RailBudget, RFBlock, RFDesign, RFExpectation, RFNetwork, RFPort, RFProbe, RFRegion, RFState
from ai_eda.report import rf_report
from ai_eda.report.figures import BOUND_RULE, NO_TOLERANCE, tolerance_figure, tolerance_rows
from ai_eda.report.rf_figures import fixture_freshness, network_curves, recorded_rawfile, s_parameter_figures
from ai_eda.report.stages import ReportFigures, build_stage_document, build_stage_report, final_report, stage_figures, theory_report, write_all_stage_reports
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.validation.rf import rf_results
from ai_eda.workflow.stages import Stage
from tests.test_rf_fixture import F_T, F_TANK, FakeAC, Board, tank_board, tank_s21

engine = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not engine.available(), reason="ngspice shared library not found")

PROV = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="test fixture")
F_C = 447.5625e6
F_PAD = 100e6
#: the pad's resistors: a 6 dB 50-ohm pi pad rounded to E96-like values (150 / 37.4 ohm), so S11 is finite
R_SHUNT, R_SERIES = 150.0, 37.4
#: the phase network: 1 kohm into L 1 uH // C (24 / 27 pF per state), read at a probe
R_PM, L_PM = 1000.0, 1e-6
C_PM = {"lo": 24e-12, "hi": 27e-12}
F_PM = 31.6e6
TEMPLATE = "kr447_test"


def u(value: Any, unit: str | None = None) -> Traced:
    return user_requirement(value, unit)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- an ac solver that writes a real rawfile


def write_ascii_rawfile(path: Path, vectors: dict[str, list[float]]) -> None:
    """An ngspice ASCII rawfile of an ac plot (``Flags: complex``) holding the frequency scale and every node voltage of ``vectors``."""
    nodes = sorted(k for k in vectors if "." not in k and k != "frequency")
    freq = vectors["frequency"]
    lines = ["Title: rf report test", "Date: Thu Jan  1 00:00:00  2026", "Plotname: AC Analysis", "Flags: complex",
             f"No. Variables: {len(nodes) + 1}", f"No. Points: {len(freq)}", "Variables:", "\t0\tfrequency\tfrequency grid=3"]
    lines += [f"\t{i}\tv({n})\tvoltage" for i, n in enumerate(nodes, 1)]
    lines.append("Values:")
    for k, f in enumerate(freq):
        lines.append(f" {k}\t{f:.15e},{0.0:.15e}")
        lines += [f"\t{vectors[n + '.real'][k]:.15e},{vectors[n + '.imag'][k]:.15e}" for n in nodes]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class RawFakeAC(FakeAC):
    """The fixture tests' complex nodal solver, writing a genuine ngspice ASCII rawfile (the report reads the file, never the solver)."""

    engine = "fake-raw-ac"

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path, command: str | None = None):  # type: ignore[override]
        res = super().run(netlist_path, analysis, workdir, command)
        if not res.succeeded or not res.raw_output_path:
            return res
        raw = Path(res.raw_output_path)
        write_ascii_rawfile(raw, res.vectors)
        return res.model_copy(update={"raw_output_hash": _sha(raw)})


# --------------------------------------------------------------------------- the synthetic RF design


def pad_s(f: float = F_PAD) -> tuple[complex, complex]:
    """(S21, S11) of the pi pad between 50-ohm ports (resistive: frequency independent)."""
    z_b = R_SHUNT * 50.0 / (R_SHUNT + 50.0)
    z_in = R_SHUNT * (R_SERIES + z_b) / (R_SHUNT + R_SERIES + z_b)
    v_a = z_in / (50.0 + z_in)
    v_b = v_a * z_b / (R_SERIES + z_b)
    return complex(2.0 * v_b), complex(2.0 * v_a - 1.0)


def pm_ratio(state: str, f: float = F_PM) -> complex:
    """2 V_T / V_s of the phase network (50 ohm source + 1 kohm into L // C): what the runner reads at the probe."""
    w = 2.0 * math.pi * f
    z_t = 1.0 / (1.0 / (1j * w * L_PM) + 1j * w * C_PM[state])
    return 2.0 * z_t / (50.0 + R_PM + z_t)


def lin(aid: str, f0: float, f1: float, points: int) -> AnalysisSpec:
    return AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": u("lin"), "points": u(points), "fstart": u(f0, "Hz"), "fstop": u(f1, "Hz")}, provenance=PROV)


def row(eid: str, quantity: str, at: float, nominal: float, *, tol: float | None = 0.05, bound: str | None = None, **kw: Any) -> RFExpectation:
    unit = "deg" if quantity == "phase21_deg" else "dB"
    return RFExpectation(id=eid, quantity=quantity, at=u(at, "Hz"), nominal=u(nominal, unit), tol_abs=None if bound else u(tol, unit), bound=bound, **kw)


def db(s: complex) -> float:
    return 20.0 * math.log10(abs(s))


def networks() -> list[RFNetwork]:
    s0 = db(tank_s21(F_TANK))
    tank = RFNetwork(
        id="tank", block="tx_chain", members=["C1", "L1", "C2", "C3", "L2", "C4", "C5"],
        ports=[RFPort(name="IN", net="A", kind="port", z0_ohm=u(50.0, "ohm")), RFPort(name="OUT", net="B", kind="port", z0_ohm=u(50.0, "ohm"))],
        loss_q={"L1": u(40.0), "L2": u(40.0)}, q_ref_hz=u(F_TANK, "Hz"), sweep=[lin("wide", F_TANK - F_T, F_TANK + F_T, 3)],
        expectations=[
            row("s21_w", "s21_db", F_TANK, s0, drive="IN", to="OUT"),
            row("rel_m", "rel_s21_db", F_TANK - F_T, db(tank_s21(F_TANK - F_T)) - s0, ref_at=u(F_TANK, "Hz"), drive="IN", to="OUT"),
            row("rel_p", "rel_s21_db", F_TANK + F_T, db(tank_s21(F_TANK + F_T)) - s0, ref_at=u(F_TANK, "Hz"), drive="IN", to="OUT"),
            row("rej_p", "rel_s21_db", F_TANK + F_T, -10.0, bound="at_most", ref_at=u(F_TANK, "Hz"), drive="IN", to="OUT"),
        ])
    s21, s11 = pad_s()
    pad = RFNetwork(
        id="pad", block="pad_block", members=["R10", "R11", "R12"],
        ports=[RFPort(name="PI", net="P_IN", kind="port", z0_ohm=u(50.0, "ohm")), RFPort(name="PO", net="P_OUT", kind="port", z0_ohm=u(50.0, "ohm"))],
        sweep=[lin("band", 50e6, 150e6, 11)],
        expectations=[
            row("s21", "s21_db", F_PAD, db(s21), tol=0.01, drive="PI", to="PO"),
            row("s11", "s11_db", F_PAD, -20.0, bound="at_most", drive="PI", to="PI"),
            row("s21_tight", "s21_db", F_PAD, -5.0, tol=0.1, drive="PI", to="PO"),  # a nominal nobody computed: the honest FAIL of the fixture
        ])
    pm = RFNetwork(
        id="pmx", block="pm", members=["R20", "L20", "C20"],
        ports=[RFPort(name="PI", net="PM_IN", kind="port", z0_ohm=u(50.0, "ohm")), RFPort(name="PT", net="PM_T", kind="probe")],
        states=[RFState(id=s, bindings={"C20": SpiceBinding(device=SpiceDevice.C, value=u(c, "F"), provenance=PROV)}) for s, c in C_PM.items()],
        sweep=[lin("sw", 20e6, 45e6, 51)],
        expectations=[row(f"ph_{s}", "phase21_deg", F_PM, math.degrees(cmath.phase(pm_ratio(s))), tol=0.5, state=s, drive="PI", to="PT") for s in C_PM],
        probes=[RFProbe(id="lvl", state="lo", quantity="rel_s21_db", drive="PI", to="PT", at=u(33e6, "Hz"), ref_at=u(F_PM, "Hz"))],
    )
    return [tank, pad, pm]


def rf_ir(tmp_path: Path, *, name: str = "rfrep") -> CircuitIR:
    """The synthetic RF design (module docstring) with its requirements, profile and model values as confirmed choices; nothing run yet."""
    b = Board(name)
    tank_board(b)
    b.r("R10", R_SHUNT, "P_IN", "GND").r("R11", R_SERIES, "P_IN", "P_OUT").r("R12", R_SHUNT, "P_OUT", "GND")
    b.r("R20", R_PM, "PM_IN", "PM_T").l("L20", L_PM, "PM_T", "GND").c("C20", 25e-12, "PM_T", "GND")
    ir = b.ir()
    ir.project = ProjectMeta(id=name, name=name, workdir=str(tmp_path))
    for key, value in {"carrier_frequency": "447.5625 MHz", "tx_power": "0.5 W", "modulation": "fm"}.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    for choice, traced in profile_choices(TEMPLATE, True):
        ir.parameters[choice.key] = traced
    models = ["model.xtal21.cm", "model.sa605.port_r", "model.l_q.uhf"]
    for key in models:
        choice, traced = model_choice(TEMPLATE, key, True)
        ir.parameters[choice.key] = traced
    image = Traced(value=404.7625e6, unit="Hz", provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="calc.rf.superhet.image", tool_version="0.10",
                                                                       derived_from=["carrier_frequency"], note="LO - IF1"))
    ir.rf = RFDesign(
        blocks=[
            RFBlock(id="tx_chain", title="TX 체배 탱크", refs=["C1", "L1", "C2", "C3", "L2", "C4", "C5"], chain=["C1", "L1", "C3", "L2", "C5"],
                    region=RFRegion(x=u(1.0, "mm"), y=u(61.0, "mm"), w=u(40.0, "mm"), h=u(28.0, "mm")),
                    ports=[RFPort(name="TANK_IN", net="P_OUT", kind="port", z0_ohm=u(50.0, "ohm"), frequency_hz=u(F_TANK, "Hz"), direction="in")]),
            RFBlock(id="pad_block", title="감쇠기", refs=["R10", "R11", "R12"],
                    ports=[RFPort(name="PAD_OUT", net="P_OUT", kind="port", z0_ohm=u(50.0, "ohm"), frequency_hz=u(F_TANK, "Hz"), direction="out")]),
            RFBlock(id="pm", title="위상 변조 탱크", refs=["R20", "L20", "C20"]),
        ],
        networks=networks(),
        frequency_plan=[
            PlanLine(id="birdie21", kind="margin", f_hz=u(439.95e6, "Hz"), ref_hz=u(F_C, "Hz"), min_margin_hz=u(1e6, "Hz"), note="21 x LO2"),
            PlanLine(id="image", kind="response", f_hz=image, ref_hz=u(F_C, "Hz"), points_to=["spice.rf.tank.rel_m", "rf.lab.image_rejection"]),
            PlanLine(id="if_vs_lo2", kind="coincidence", f_hz=u(21.4e6, "Hz"), ref_hz=u(20.95e6, "Hz")),
        ],
        lab_items=[
            LabItem(id="image_rejection", block="tx_chain", what="image response rejection at 404.76 MHz", instruments=["signal generator", "SINAD meter"],
                    reason="needs the real front end and mixer"),
            LabItem(id="kc_conformity", what="KC conformity assessment", instruments=["designated test lab"], reason="legality is not decided by this system"),
        ],
        rails=[RailBudget(rail="TX_5V", regulator_ref="U99", v_out=u(5.0, "V"), i_min=u(0.287, "A"), i_max=u(0.441, "A"), i_rating=u(0.8, "A"),
                          dropout_v=u(1.2, "V"), path_r_ohm=u(0.17, "ohm"))],
        model_values=models,
        profile_keys=profile_keys(),
    )
    ir.pcb = PCBDesign(keepouts=[Keepout(id="ant_band", layers=["*.Cu"], rect=Traced(value=[0.0, 0.0, 60.0, 6.0], unit="mm", provenance=PROV),
                                         forbids=["tracks", "footprints"], allowed_refs=["ANT1"], reason="antenna band", provenance=PROV)])
    return ir


def record(ir: CircuitIR, results: list[ValidationResult]) -> None:
    h = ir.content_hash()
    for r in results:
        ir.validation.add(r if r.ir_hash is not None else r.model_copy(update={"ir_hash": h}))


def simulated(tmp_path: Path, runner: Any = None, *, name: str = "rfrep") -> CircuitIR:
    """The synthetic design with its fixtures run (``runner``; the rawfile-writing solver by default) and the RF checks recorded."""
    ir = rf_ir(tmp_path, name=name)
    record(ir, spice_rf_results(ir, {"spice": runner or RawFakeAC()}, tmp_path))
    record(ir, rf_results(ir))
    return ir


def section(md: str, heading: str, level: str = "## ") -> str:
    """The text of one ``## `` section of a report (up to the next heading of that level)."""
    start = md.index(heading)
    nxt = md.find("\n" + level, start + len(heading))
    return md[start:] if nxt < 0 else md[start:nxt]


def subsection(md: str, heading: str) -> str:
    return section(md, heading, "### ")


# --------------------------------------------------------------------------- no RF, no RF text


def test_a_design_without_rf_gets_no_rf_text_and_no_banner(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="plain", name="plain", workdir=str(tmp_path)))
    assert rf_report.profile_banner(ir) == [] and rf_report.rf_theory_section(ir) == [] and rf_report.rf_final_section(ir) == []
    figures = ReportFigures()
    rf_report.rf_figures_of(ir, figures)
    assert rf_report.rf_circuit_section(ir, figures) == [] and not figures.figures and not figures.missing
    for stage in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION, Stage.PCB, Stage.RELEASE):
        md = build_stage_report(stage, ir, None, None)
        assert md is not None and "UNVERIFIED" not in md and "무선(RF)" not in md


# --------------------------------------------------------------------------- the banner


def test_every_report_starts_with_the_unverified_profile_banner(tmp_path: Path):
    ir = simulated(tmp_path)
    assert ir.validation.latest("rf.regulatory_profile").status is S.NOT_VERIFIED
    for stage in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION, Stage.PCB, Stage.RELEASE):
        md = build_stage_report(stage, ir, None, None)
        assert md is not None
        head = md[: md.index("\n## ")]  # before the first section: the banner belongs to the header
        assert rf_report.BANNER_TITLE in head, stage
        assert all(f"`{k}`" in head for k in profile_keys()), stage
        assert "KC 적합성평가" in head and "grounding 된 사실이 아닙니다" in head and "「무선설비규칙」" in head
        assert "`rf.regulatory_profile` **NOT_VERIFIED** (`rf.checks` v0.1)" in head
        assert "(`kr447.conformity` = KC 적합인증 before any transmission" in head
    # the banner says what holds: the profile check never PASSes on the placeholders, a row derived from them may (listed from provenance)
    banner = "\n".join(rf_report.profile_banner(ir))
    assert "이 값으로 PASS 가 되는 검사는 없습니다" not in banner and "`rf.regulatory_profile` 은 이 값으로 PASS 가 되지 않습니다" in banner
    assert "이 설계에는 출처가 자리표시값에 닿는 시뮬레이션 행이 없습니다" in banner and rf_report.profile_dependents(ir) == []
    # a value the user has not confirmed yet is named as such
    t = ir.parameters["kr447.max_power"]
    ir.parameters["kr447.max_power"] = t.model_copy(update={"provenance": t.provenance.model_copy(update={"kind": ProvenanceKind.ASSUMPTION})})
    assert any("아직 사용자가 확인하지 않은(가정) 값: `kr447.max_power`" in line for line in rf_report.profile_banner(ir))


# --------------------------------------------------------------------------- the theory report


def test_the_theory_section_carries_plan_profile_models_and_fixtures(tmp_path: Path):
    ir = simulated(tmp_path)
    md = theory_report(ir)
    rf = section(md, "## 무선(RF) 설계")
    # the plan: every row, its numbers, the recorded row status (copied), a calculator's origin
    plan = ir.validation.latest("rf.freq_plan")
    recorded = {r["id"]: r["status"] for r in plan.details["rows"]}
    assert recorded == {"birdie21": "PASS", "image": "NOT_VERIFIED", "if_vs_lo2": "PASS"}
    assert "| `birdie21` | 여유 (margin) | 439.95 MHz | 447.5625 MHz | −7.6125 MHz | 1 MHz |" in rf
    assert "| `image` | 응답 (response) | 404.7625 MHz | 447.5625 MHz | −42.8 MHz | - | 계산기 출력 (calc.rf.superhet.image v0.10) |" in rf
    assert "`spice.rf.tank.rel_m`, `rf.lab.image_rejection` | NOT_VERIFIED |" in rf
    assert f"- `rf.freq_plan`: **{plan.status}** (`rf.checks` v0.1) [현재 IR]" in rf
    # the response / gated rule as rf.freq_plan applies it (kr447 wave-2 review finding 9): the worst of what the row points to - a lab item is
    # always NOT_VERIFIED, a check id follows its latest current-design result - so a row pointing only to fixture rows follows their verdict
    assert "'판정을 가진 곳' 에 적힌 항목들의 가장 나쁜 상태를 따릅니다" in rf and "고정구 행만 가리키는 행은 그 회로망 판정" in rf
    assert "그래서 수신기의 영상·스퓨리어스 응답 같은 행은 실험실 측정 전에는 PASS 가 되지 않습니다" not in rf
    # the profile: each value with its document as UNVERIFIED and the requirement it is compared with
    assert "| `kr447.max_power` | 0.5 W | 사용자 확인 선택값 |" in rf
    assert "「신고하지 아니하고 개설할 수 있는 무선국용 무선기기」; general knowledge, law.go.kr not reachable | `tx_power` 이하 | NOT_VERIFIED |" in rf
    assert "| `kr447.band_low` | 447.5625 MHz |" in rf and "`carrier_frequency` 가 대역 안 | NOT_VERIFIED (band), NOT_VERIFIED (raster) |" in rf
    assert "| `kr447.emission` | F3E |" in rf and "비교 없음 (조건·기록용)" in rf
    # the model values: every listed key with what would ground it and the recorded row
    mg = ir.validation.latest("rf.model_grounding")
    assert mg.status is S.NOT_VERIFIED
    for key in ("model.xtal21.cm", "model.sa605.port_r", "model.l_q.uhf"):
        line = next(x for x in rf.splitlines() if x.startswith(f"| `{key}` |"))
        assert MODEL_VALUES[key].source in line and "NOT_VERIFIED — a confirmed model choice, not grounded" in line and "| 예 |" in line
    # the fixtures: the formulas, each row's nominal, rule and origin
    assert "S21 = 2·V_읽기 / V_s · √(R_구동 / R_읽기)" in rf and "R_s = 2π·f_Q·L / Q" in rf
    assert "#### `tank` (블록 `tx_chain`)" in rf and "`L1` Q = 40" in rf
    assert "| `rej_p` | - | 상대 S21 (dB) | `IN` → `OUT` | 261.078125 MHz | 223.78125 MHz | -10 dB | ≤ 공칭 (한쪽 한계) | 사용자 요구사항 |" in rf
    assert "| `s21` | - | S21 (dB) | `PI` → `PO` | 100 MHz |" in rf and "± 0.01 dB" in rf
    assert "| `ph_lo` | `lo` | S21 위상 (°) | `PI` → `PT` |" in rf and "상태: `lo`, 바인딩 교체 `C20`; `hi`, 바인딩 교체 `C20`" in rf
    assert "기록만 하는 프로브(판정 없음): `lvl` 상대 S21 (dB) `PI` → `PT` @ 33 MHz (기준 31.6 MHz)" in rf


def test_statuses_are_copied_never_computed(tmp_path: Path):
    ir = simulated(tmp_path)
    before = theory_report(ir)
    plan = ir.validation.latest("rf.freq_plan")
    rows = [dict(r, status="UNRESOLVED") if r["id"] == "birdie21" else r for r in plan.details["rows"]]
    ir.validation.add(plan.model_copy(update={"status": S.UNRESOLVED, "details": {**plan.details, "rows": rows}}))
    changed = theory_report(ir)
    assert changed != before and "| `birdie21` |" in changed
    assert next(x for x in changed.splitlines() if x.startswith("| `birdie21` |")).endswith("| UNRESOLVED | 21 x LO2 |")
    assert "- `rf.freq_plan`: **UNRESOLVED**" in changed
    # a fixture row's recorded status and measurement are printed as recorded
    r = ir.validation.latest("spice.rf.pad.s21")
    ir.validation.add(r.model_copy(update={"status": S.FAIL, "details": {**r.details, "measured": -1.25}}))
    figures = stage_figures(Stage.PCB, ir, None)
    md = "\n".join(rf_report.fixture_result_lines(ir, figures))
    assert next(x for x in md.splitlines() if x.startswith("| `s21` |")).endswith("| -1.2500 dB | 0.0000 dB | FAIL |")


# --------------------------------------------------------------------------- the circuit report: fixture verdicts and charts


def test_the_circuit_section_copies_every_fixture_row(tmp_path: Path):
    ir = simulated(tmp_path)
    assert ir.validation.latest("spice.rf.tank").status is S.PASS
    assert ir.validation.latest("spice.rf.pmx").status is S.PASS
    assert ir.validation.latest("spice.rf.pad").status is S.FAIL  # the row nobody computed
    doc = build_stage_document(Stage.PCB, ir, None, None)
    rf = section(doc.markdown, "## 무선(RF) 블록과 고정구 검증")
    for n in ("tank", "pad", "pmx"):
        summary = ir.validation.latest(f"spice.rf.{n}")
        assert f"- `spice.rf.{n}`: **{summary.status}** (`fake-raw-ac` fake-1) [현재 IR]" in rf
    for check in ("spice.rf.tank.s21_w", "spice.rf.tank.rel_p", "spice.rf.pmx.lo.ph_lo", "spice.rf.pad.s11"):
        r = ir.validation.latest(check)
        eid = check.rsplit(".", 1)[-1]
        line = next(x for x in rf.splitlines() if x.startswith(f"| `{eid}` |"))
        assert line.endswith(f"| {r.status} |") and f"{r.details['measured']:.4f}" in line, (check, line)
    fail = ir.validation.latest("spice.rf.pad.s21_tight")
    assert fail.status is S.FAIL and f"  - `spice.rf.pad.s21_tight`: FAIL — " in rf
    assert "기록된 프로브 (판정 없음):" in rf and "`lo.lvl`:" in rf
    assert "- 덱 (기록 그대로):" in rf and "구동 `PI`): 추가 소자" in rf and "상태 `lo`" in rf
    # blocks, interface, keep-out
    assert "| `tx_chain` | TX 체배 탱크 | 7 | C1 → L1 → C3 → L2 → C5 | - | x 1–41, y 61–89 mm |" in rf
    assert f"- `block.interface.P_OUT`: **{ir.validation.latest('block.interface.P_OUT').status}**" in rf
    assert "| `ant_band` | *.Cu | x 0–60, y 0–6 mm | tracks, footprints | `ANT1` | - | 유지 (`zones` 를 금지하지 않음) | antenna band |" in rf
    assert "keep-out 은 라우터의 장애물이고 평면 존을 자르며" not in rf and "평면 존은 `zones` 를 금지하는 keep-out 만 자릅니다" in rf
    assert rf.count("- `pcb.keepout`: 기록 없음") == 1
    # a keep-out that forbids zones clips the planes (except its allowed nets'), and says so per keep-out
    ko = ir.pcb.keepouts[0]
    ir.pcb.keepouts = [ko.model_copy(update={"forbids": ["tracks", "zones"], "allowed_nets": ["GND"]})]
    blocks = "\n".join(rf_report.block_lines(ir))
    assert "| 자름 (예외 넷 `GND` 의 평면은 유지) |" in blocks
    # regions and cans without any keep-out: pcb.keepout judges the regions, so the section copies it
    ir.pcb.keepouts = []
    blocks = "\n".join(rf_report.block_lines(ir))
    assert "keep-out" in blocks and blocks.count("- `pcb.keepout`: 기록 없음") == 1 and "배치 영역 안에" in blocks
    ir.rf.blocks = [b.model_copy(update={"region": None, "shield_ref": None}) for b in ir.rf.blocks]
    assert "`pcb.keepout`" not in "\n".join(rf_report.block_lines(ir))  # nothing the check would judge
    # evidence by its path under the fixture folder, never an absolute path
    assert str(tmp_path) not in doc.markdown and str(tmp_path) not in doc.html
    assert "증거 파일 (덱 + rawfile, 해시와 함께 기록, " in rf and ".raw`" in rf


def test_the_s_parameter_charts_are_drawn_from_the_fresh_rawfiles(tmp_path: Path):
    ir = simulated(tmp_path)
    figures = stage_figures(Stage.PCB, ir, None)
    ids = [f for f in figures.figures if f.startswith("rf_s21")]
    assert ids == ["rf_s21_tank_wide_level", "rf_s21_pad_band_level", "rf_s21_pmx_sw_level", "rf_s21_pmx_sw_phase"], ids
    for slot in ("rf_s21:tank", "rf_s21:pad", "rf_s21:pmx"):
        assert slot not in figures.missing, figures.missing
    tank = figures.figures["rf_s21_tank_wide_level"]
    assert 'data-name="S21 IN→OUT"' in tank.svg and "rel_p PASS" in tank.svg and "rej_p PASS" in tank.svg
    assert "#008300" in tank.svg and ".raw" in tank.caption and "판정은 기록된 결과의 것" in tank.caption
    pad = figures.figures["rf_s21_pad_band_level"]
    assert 'data-name="S21 PI→PO"' in pad.svg and 'data-name="S11 PI"' in pad.svg and "s21_tight FAIL" in pad.svg and "#e34948" in pad.svg
    pm = figures.figures["rf_s21_pmx_sw_phase"]
    assert 'data-name="∠(2V/V_s) PI→PT (프로브) [lo]"' in pm.svg and 'data-name="∠(2V/V_s) PI→PT (프로브) [hi]"' in pm.svg
    assert "[lo] ph_lo PASS" in pm.svg and "[hi] ph_hi PASS" in pm.svg
    # the html embeds every chart; the markdown keeps a placeholder and a caption per chart
    doc = build_stage_document(Stage.PCB, ir, None, None)
    for fig_id in ids:
        assert f"![fig](fig:{fig_id})" in doc.markdown and figures.figures[fig_id].svg.strip()[:40] in doc.html


def test_the_curve_at_a_rows_own_frequency_is_the_recorded_measurement(tmp_path: Path):
    ir = simulated(tmp_path)
    net = ir.rf.network("tank")
    (sweep,) = network_curves(ir, net, ir.validation.latest("spice.rf.tank"))
    (curve,) = sweep.curves
    rec = ir.validation.latest("spice.rf.tank.s21_w").details
    k = min(range(len(curve.xs)), key=lambda i: abs(curve.xs[i] - F_TANK))
    assert curve.xs[k] == pytest.approx(F_TANK, rel=1e-12) and curve.ys[k] == pytest.approx(rec["measured"], abs=1e-9)
    rel = ir.validation.latest("spice.rf.tank.rel_p").details
    assert curve.ys[-1] - curve.ys[k] == pytest.approx(rel["measured"], abs=1e-9)
    pm = ir.rf.network("pmx")
    (sw,) = network_curves(ir, pm, ir.validation.latest("spice.rf.pmx"))
    lo = next(c for c in sw.curves if c.state == "lo")
    assert lo.kind == "phase" and lo.probe and len(lo.xs) == 51
    j = min(range(len(lo.xs)), key=lambda i: abs(lo.xs[i] - 31.5e6))
    assert lo.ys[j] == pytest.approx(math.degrees(cmath.phase(pm_ratio("lo", lo.xs[j]))), abs=1e-6)


def test_nothing_is_drawn_from_evidence_about_another_design_or_changed_files(tmp_path: Path):
    ir = simulated(tmp_path)
    summary = ir.validation.latest("spice.rf.tank")
    assert fixture_freshness(ir, summary) is None
    # a changed design: every summary is about the previous IR
    changed = ir.model_copy(deep=True)
    changed.parameters["kr447.max_power"] = u(0.4, "W")
    figs, reasons = s_parameter_figures(changed, changed.rf.network("tank"))
    assert figs == [] and reasons == ["spice.rf.tank 은 이전 IR 버전의 결과입니다"]
    fig = stage_figures(Stage.PCB, changed, None)
    assert "rf_s21:tank" in fig.missing and "이전 IR 버전" in fig.missing["rf_s21:tank"]
    # a rawfile changed or removed since the run
    raws = [e.path for e in summary.evidence if e.path.endswith(".raw") and "/wide/" in e.path.replace("\\", "/")]
    assert len(raws) == 1
    raw = Path(raws[0])
    # every analysis writes a rawfile of one name into its own folder: the evidence line lists each by its path under the fixture folder
    stem = Path(next(e.path for e in summary.evidence if e.path.endswith(".cir"))).stem
    circuit = "\n".join(rf_report.fixture_result_lines(ir, stage_figures(Stage.PCB, ir, None)))
    line = next(x for x in circuit.splitlines() if x.startswith("  - 증거 파일") and f"`{stem}.cir`" in x)
    assert f"`{stem}/wide/{stem}.ac.raw`" in line and f"`{stem}/rf_at_0/{stem}.ac.raw`" in line and f"{len(summary.evidence)}개)" in line
    assert line.count(".ac.raw`") == len(summary.evidence) - 1 and str(tmp_path) not in line
    raw.write_text(raw.read_text(encoding="ascii").replace("Title: rf report test", "Title: edited"), encoding="ascii")
    assert recorded_rawfile(summary, str(raw)) == f"rawfile {raw.name} 이 기록된 해시와 다릅니다"
    figs, reasons = s_parameter_figures(ir, ir.rf.network("tank"))
    assert figs == [] and any("기록된 해시와 다릅니다" in r for r in reasons)
    assert any(r.startswith("덱 tank, 해석 wide: rawfile") for r in reasons), reasons  # the reason names the analysis whose file changed
    raw.unlink()
    assert recorded_rawfile(summary, str(raw)) == f"rawfile {raw.name} 이 디스크에 없습니다"
    assert recorded_rawfile(summary, str(tmp_path / "other.raw")) == f"rawfile other.raw 이 {summary.check_id} 의 증거 목록에 없습니다"
    # the other networks' files are untouched: their charts are still drawn
    assert s_parameter_figures(ir, ir.rf.network("pad"))[0]
    # no engine: compiled decks, nothing simulated, no chart
    ir2 = rf_ir(tmp_path / "b", name="rfrep2")
    record(ir2, spice_rf_results(ir2, {}, tmp_path / "b"))
    figs, reasons = s_parameter_figures(ir2, ir2.rf.network("tank"))
    assert figs == [] and "도구 기록이 없어" in reasons[0]
    md = "\n".join(rf_report.fixture_result_lines(ir2, stage_figures(Stage.PCB, ir2, None)))
    assert "S-파라미터 그림이 없습니다 (spice.rf.tank 은 도구 기록이 없어" in md


def test_an_unreadable_rawfile_is_a_reason_named_by_basename(tmp_path: Path):
    """A rawfile whose recorded hash matches but whose body is not a rawfile: the chart is left out and the parser's reason is printed
    without the absolute path it names."""
    ir = simulated(tmp_path)
    summary = ir.validation.latest("spice.rf.pad")
    (raw,) = [Path(e.path) for e in summary.evidence if e.path.endswith(".raw") and "/band/" in e.path.replace("\\", "/")]
    raw.write_text("Title: broken\nNo. Variables: 2\n", encoding="ascii")
    evidence = [e.model_copy(update={"content_hash": _sha(raw)}) if e.path == str(raw) else e for e in summary.evidence]
    ir.validation.add(summary.model_copy(update={"evidence": evidence}))
    figures = stage_figures(Stage.PCB, ir, None)
    assert not [f for f in figures.figures if f.startswith("rf_s21_pad")]
    text = figures.missing["rf_s21:pad"]
    assert text.startswith("S-파라미터 그림이 없습니다 (") and f"rawfile {raw.name} 을 읽을 수 없습니다" in text and str(tmp_path) not in text
    assert "rf_s21_tank_wide_level" in figures.figures  # the other networks are drawn


# --------------------------------------------------------------------------- the final report


def test_the_final_section_lists_rf_results_lab_items_and_rails(tmp_path: Path):
    ir = simulated(tmp_path)
    md = final_report(ir, None)
    rf = section(md, "## 무선(RF) 검증 요약과 실험실 항목")
    for check in ("spice.rf.tank", "spice.rf.pad", "spice.rf.pmx", "rf.freq_plan", "rf.regulatory_profile", "rf.model_grounding", "block.interface.P_OUT",
                  "power.rail_budget.TX_5V", "power.headroom.U99"):
        r = ir.validation.latest(check)
        assert r is not None, check
        assert f"| `{check}` | {r.status} |" in rf, check
    assert "| `rf.lab.image_rejection` |" not in subsection(rf, "### RF 검사의 최신 결과")  # the lab items have their own table
    lab = ir.validation.latest("rf.lab.image_rejection")
    assert lab.status is S.NOT_VERIFIED
    assert "| `rf.lab.image_rejection` | `tx_chain` | image response rejection at 404.76 MHz | signal generator, SINAD meter | needs the real front end and mixer | NOT_VERIFIED |" in rf
    assert "| `rf.lab.kc_conformity` | - | KC conformity assessment |" in rf
    budget = ir.validation.latest("power.rail_budget.TX_5V")
    head = ir.validation.latest("power.headroom.U99")
    assert f"| `TX_5V` | `U99` | 5 V | 287 mA – 441 mA | 800 mA | 1.2 V | 0.17 Ω | 사용자 요구사항 | {budget.status} | {head.status} |" in rf
    assert "### 정직한 한계" in rf and "KC 적합성평가 전에는 어떤 송신도 하지 않습니다" in rf


# --------------------------------------------------------------------------- one-sided bounds in the tolerance chart and the tables


def _bound_ir(tmp_path: Path) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="bound", name="bound", workdir=str(tmp_path)))
    an = AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=PROV)
    ir.simulation = SimulationSetup(analyses=[an], expectations=[
        Expectation(id="rail_on", analysis_id="op", vector="v(V_TX)", reduce=Reduce.VALUE, nominal=u(5.4, "V"), bound="at_least", provenance=PROV),
        Expectation(id="rail_off", analysis_id="op", vector="v(V_RX)", reduce=Reduce.VALUE, nominal=u(0.2, "V"), bound="at_most", provenance=PROV),
        Expectation(id="vref", analysis_id="op", vector="v(REF)", reduce=Reduce.VALUE, nominal=u(1.0, "V"), tol_abs=u(0.02, "V"), provenance=PROV),
    ], provenance=PROV)
    h = ir.content_hash()
    for eid, measured, status, bound, tol in (("rail_on", 7.3, S.PASS, "at_least", None), ("rail_off", 0.35, S.FAIL, "at_most", None), ("vref", 1.01, S.PASS, None, 0.02)):
        details = {"measured": measured, "nominal": {"rail_on": 5.4, "rail_off": 0.2, "vref": 1.0}[eid], "tolerance": tol, "deviation": 0.0}
        if bound:
            details["bound"] = bound
        ir.validation.add(ValidationResult(check_id=f"spice.{eid}", status=status, message="recorded", tool="ngspice", tool_version="ngspice-42", ir_hash=h, details=details))
    return ir


def test_a_bound_row_is_a_one_sided_limit_in_the_chart_and_the_tables(tmp_path: Path):
    ir = _bound_ir(tmp_path)
    rows = {r.label: r for r in tolerance_rows(ir)}
    assert rows["rail_on"].bound == "at_least" and rows["rail_off"].bound == "at_most" and rows["vref"].bound is None
    assert rows["rail_on"].tolerance is None and rows["rail_on"].tolerance_recorded
    fig = tolerance_figure(list(rows.values()))
    assert fig.svg.count('class="band bound"') == 2 and 'data-bound="at_least"' in fig.svg and 'data-bound="at_most"' in fig.svg
    assert 'data-side="0.75"' in fig.svg and 'data-side="0.75"' in fig.svg  # 7.3 > 5.4 and 0.35 > 0.2: both above the nominal
    assert "측정 7.3 V / ≥ 5.4 V (한쪽 한계)" in fig.svg and "측정 350 mV / ≤ 200 mV (한쪽 한계)" in fig.svg
    assert NO_TOLERANCE not in fig.svg and "허용치 (±1) / 한쪽 한계의 통과 쪽" in fig.svg and "한쪽 한계 행(≥ / ≤ 공칭)" in fig.caption
    # a chart of tolerance rows only is the chart it was
    plain = tolerance_figure([rows["vref"]])
    assert "bound" not in plain.svg and "허용치 (±1)</text>" in plain.svg and "한쪽 한계" not in plain.caption
    # the tables
    md = theory_report(ir)
    assert "| `rail_on` | op | `v(V_TX)` | 값 (동작점) | 5.4 V | ≥ 공칭 (한쪽 한계) | - |" in md and BOUND_RULE in md
    fin = final_report(ir, None)
    assert "| `rail_on` |" in fin and "| ≥ 5.4 V (한쪽 한계) | PASS |" in fin and "| ≤ 0.2 V (한쪽 한계) | FAIL |" in fin


# --------------------------------------------------------------------------- determinism and files


def test_the_reports_are_byte_identical_and_name_files_by_basename(tmp_path: Path):
    ir = simulated(tmp_path)
    first = [build_stage_document(s, ir, None, None) for s in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION, Stage.PCB, Stage.RELEASE)]
    again = [build_stage_document(s, ir, None, None) for s in (Stage.ARCHITECTURE, Stage.COMPONENT_SELECTION, Stage.PCB, Stage.RELEASE)]
    assert [(d.markdown, d.html) for d in first] == [(d.markdown, d.html) for d in again]
    assert all(str(tmp_path) not in d.markdown and str(tmp_path) not in d.html for d in first)
    written = write_all_stage_reports(ir, None, None, tmp_path, pdf=False)
    assert len(written) == 4
    for w in written:
        assert rf_report.BANNER_TITLE in w.markdown.read_text(encoding="utf-8")
    assert ir.artifacts == {}  # a view registers nothing


# --------------------------------------------------------------------------- on ngspice


@needs_ngspice
def test_ngspice_fixture_rawfiles_become_the_s_parameter_charts(tmp_path: Path):
    """The same fixtures on ngspice-42: every row the design computed PASSes, the one nobody computed FAILs, inside the §3.3 budget
    (60 s per deck; each deck here runs in well under a second), and the charts read ngspice's own binary rawfiles."""
    t0 = time.perf_counter()
    ir = simulated(tmp_path, engine)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0
    rows = {k: r for k, r in ir.validation.latest_by_check().items() if k.startswith("spice.rf.") and k.count(".") >= 3}
    assert {k: r.status for k, r in rows.items()} == {
        "spice.rf.tank.s21_w": S.PASS, "spice.rf.tank.rel_m": S.PASS, "spice.rf.tank.rel_p": S.PASS, "spice.rf.tank.rej_p": S.PASS,
        "spice.rf.pad.s21": S.PASS, "spice.rf.pad.s11": S.PASS, "spice.rf.pad.s21_tight": S.FAIL,
        "spice.rf.pmx.lo.ph_lo": S.PASS, "spice.rf.pmx.hi.ph_hi": S.PASS,
    }, {k: r.message for k, r in rows.items() if r.status is not S.PASS}
    assert ir.validation.latest("spice.rf.tank").tool == engine.engine
    figures = stage_figures(Stage.PCB, ir, None)
    assert [f for f in figures.figures if f.startswith("rf_s21")] == ["rf_s21_tank_wide_level", "rf_s21_pad_band_level", "rf_s21_pmx_sw_level", "rf_s21_pmx_sw_phase"]
    (sweep,) = network_curves(ir, ir.rf.network("tank"), ir.validation.latest("spice.rf.tank"))
    (curve,) = sweep.curves
    k = min(range(len(curve.xs)), key=lambda i: abs(curve.xs[i] - F_TANK))
    assert curve.ys[k] == pytest.approx(ir.validation.latest("spice.rf.tank.s21_w").details["measured"], abs=1e-6)
    md = build_stage_document(Stage.PCB, ir, None, None).markdown
    assert f"- `spice.rf.tank`: **PASS** (`{engine.engine}`" in md
