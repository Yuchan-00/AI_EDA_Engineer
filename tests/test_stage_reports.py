"""The four Korean Markdown stage reports (theory, parts, circuit, final), their writers and the CLI integration.

A report is a view: it computes no status, registers no artifact, changes
no hash, and holds no wall-clock and no absolute path, so two builds from
the same IR and record are byte-identical. The designs are built through
the real ``CircuitDesignAgent`` on the synthetic library of
``tests/test_circuit_templates.py`` (present, then confirm) and run to
COMPONENT_SELECTION (reports 1 / 2) or to RELEASE (reports 3 / 4, the CLI);
only the theory-vs-simulation rows with measured values need ngspice
(``needs_dll``). The CLI runs (``_cli_project``) disable kicad-cli
discovery, so their expected exit code (NOT_VERIFIED exits 0) and the
content of ``<workdir>/reports/`` never depend on an ERC / DRC verdict on
the synthetic-library boards, which no machine has measured (CLAUDE.md);
``test_cli_runs_ignore_a_kicad_cli_on_path`` pins that with a stand-in
binary that would FAIL the run.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import stat
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.agents.keys import ROUTING_KEY
from ai_eda.cli import main as cli_main
from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.design import NO_RECORD, TEMPLATES, UNVERIFIED_SUBSTITUTE, PartNote, Template, TheorySection, quantity
from ai_eda.design.templates import astable_drawing, nearest_e12
from ai_eda.ir import ArtifactKind, CircuitIR, Provenance, ProvenanceKind, Traced, ValidationResult, ValidationStatus as S
from ai_eda.report import (
    PIPELINE_FILE,
    REPORTS_DIR,
    STAGE_REPORTS,
    build_stage_report,
    circuit_report,
    final_report,
    load_pipeline_record,
    parts_report,
    render_report_file,
    template_of,
    theory_report,
    write_all_stage_reports,
    write_stage_report,
)
from ai_eda.report.stages import (
    NO_MEASUREMENT,
    NO_ROUTING,
    NO_RUN_RECORD,
    NO_TEMPLATE,
    NO_TEMPLATE_INFO,
    SUBSTITUTES_HEADING,
    copper_resistance_ohm,
    ipc2221_current_a,
    net_routing_stats,
    routing_params_of,
    strip_paths,
    template_for,
)
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import STAGE_ORDER, Orchestrator, PipelineState, Stage
from tests.fixtures_kicad import divider_with_connector_ir
from tests.test_circuit_templates import ASTABLE, BASE, DIVIDER, LED, RC, _confirm, _ir, _present, _run, template_library

needs_dll = pytest.mark.skipif(not NgspiceShared().available(), reason="ngspice shared library not found")

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T")


def _build(tmp_path: Path, name: str, answers: dict[str, str], stop_after: Stage = Stage.COMPONENT_SELECTION) -> tuple[CircuitIR, KicadLibrary]:
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path / name, name)
    _present(ir, tmp_path / name, lib, answers)
    state, _ = _confirm(ir, tmp_path / name, lib, stop_after)
    assert not state.blocked and state.outcome(stop_after) is not None
    return ir, lib


def _release(tmp_path: Path, name: str, answers: dict[str, str], *, spice: bool = False, extra: dict[str, str] | None = None) -> tuple[CircuitIR, KicadLibrary, PipelineState]:
    """Present, confirm and run the whole pipeline (to RELEASE) offline; ``extra`` answers (``pcb.routing=skip``) go with the confirmation."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path / name, name)
    _present(ir, tmp_path / name, lib, answers)
    state, _ = _run(ir, tmp_path / name, lib, {CONFIRM_DESIGN_KEY: "yes", **(extra or {})}, None, spice=spice)
    assert not state.blocked and state.outcomes[-1].stage is Stage.RELEASE
    return ir, lib, state


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _table_rows(text: str, heading: str) -> list[list[str]]:
    """The body rows of the first Markdown table under ``heading`` (cells stripped)."""
    section = text.split(heading, 1)[1]
    lines: list[str] = []
    for ln in section.splitlines():
        if ln.startswith("|"):
            lines.append(ln)
        elif lines:
            break  # the table ended
    assert len(lines) >= 2, heading
    return [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines[2:]]


@pytest.fixture
def astable(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary]:
    return _build(tmp_path, "astable", ASTABLE)


@pytest.fixture
def divider(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary]:
    return _build(tmp_path, "divider", DIVIDER)


def _assert_clean(text: str, tmp_path: Path) -> None:
    """No ISO timestamp and no absolute path anywhere in a report."""
    assert not ISO_RE.search(text)
    assert "/tmp" not in text and str(tmp_path) not in text and "C:\\" not in text


# --------------------------------------------------------------------------- report 1: theory


def test_astable_theory_report_states_the_formulas_with_the_design_numbers(astable, tmp_path: Path):
    ir, lib = astable
    text = theory_report(ir, lib)
    assert text.startswith("# 이론 보고서: astable\n") and "템플릿 `astable` v0.1" in text
    # the period expression and the substituted numbers (C, the half period, the reverse V_BE, the ln term)
    assert "ln((2·V_cc − V_BE)/(V_cc − V_BE))" in text and "(식 1)" in text
    assert "64.817 nF" in text and "500 µs" in text and "4.3 V" in text and "0.771399" in text
    assert "I_C(sat) ≈ V_cc/R_c = 5 mA" in text and "P(R_c) = V_cc²/R_c = 25 mW" in text
    assert "τ_c = R_c·C = 64.817 µs" in text and "2.2·τ_c ≈ 142.6 µs" in text
    # the model card comes from the IR's SPICE binding, the tran window from the calculator parameters
    assert "`.model QNPN NPN (TR=200n)`" in text and "step  = 1/(200·f) = 5 µs" in text and "stop  = 20/f      = 20 ms" in text
    # the parameter table: key, value, origin, calculator id or the confirmed-choice label, the formula note
    assert "| `c` | 64.817 nF | 계산기 출력 | `calc.astable.c_for_frequency` v0.6 ← f_osc, r_b, v_in, v_be | C = 1 / (2 f R_b ln((2 V_cc - V_BE) / (V_cc - V_BE))) |" in text
    assert "| `r_b` | 10 kΩ | 사용자 확인 선택값 | template astable v0.1 |" in text
    assert "| `v_in` | 5 V | 사용자 요구사항 | req.input_voltage |" in text
    # constraints, simulation setup, the judging rule and the frequency measurement rule (Reduce.FREQUENCY is used)
    assert "`c.astable.reverse_vbe`" in text and "`c.astable.nonpolar_caps`" in text
    assert "|측정값 − 공칭값| ≤ max(tol_abs, tol_rel · |공칭값|)" in text
    assert "### 주파수 측정식 (`Reduce.FREQUENCY`)" in text and "f_measured = (N − 1) / (t_N − t_1)" in text
    assert "| `f_osc` | tran | `v(OUT)` | 상승 에지 주파수 | 1 kHz | - | 10 % | req.oscillation_frequency |" in text
    assert "| `out_low` | tran | `v(OUT)` | 최솟값 | 0 V | 0.25 V | - | (요구사항 없음) |" in text
    _assert_clean(text, tmp_path)


def test_divider_theory_report(divider, tmp_path: Path):
    ir, lib = divider
    text = theory_report(ir, lib)
    assert "V_out = V_in·R2/(R1+R2)" in text and "R1 = R2·(V_in − V_out)/V_out" in text
    assert "= 10 kΩ·(12 V − 5 V)/5 V = 14 kΩ" in text and "R_th = R1‖R2" in text and "5.8333 kΩ" in text
    assert "|측정값 − 5 V| ≤ 1 % × 5 V = 50 mV" in text
    assert "주파수 측정식" not in text  # no Reduce.FREQUENCY expectation in a divider
    assert "| `r1` | 14 kΩ | 계산기 출력 | `calc.divider.r1_for_v_out` v0.6 ← v_in, v_out_target, r2 | R1 = R2 * (V_in - V_out) / V_out |" in text
    _assert_clean(text, tmp_path)


def test_theory_report_is_deterministic_and_leaves_the_ir_alone(astable):
    ir, lib = astable
    before, n_results, n_artifacts = ir.content_hash(), len(ir.validation.results), len(ir.artifacts)
    first, second = theory_report(ir, lib), theory_report(ir, lib)
    assert first == second and first.endswith("\n") and "\r" not in first
    assert ir.content_hash() == before and len(ir.validation.results) == n_results and len(ir.artifacts) == n_artifacts
    assert parts_report(ir, lib) == parts_report(ir, lib)
    assert ir.content_hash() == before and len(ir.validation.results) == n_results and ArtifactKind.BOM not in ir.artifacts


def test_a_missing_parameter_prints_no_record_instead_of_a_guess(astable):
    ir, lib = astable
    stripped = ir.model_copy(deep=True)
    del stripped.parameters["c"]
    text = theory_report(stripped, lib)
    assert NO_RECORD in text and "64.817 nF" not in text
    assert "T_half = R_b·C·ln항 | 기록 없음 |" in text
    notes = template_for(stripped).part_notes(stripped)
    assert f"정전용량 {NO_RECORD} ± 공차" in notes["C1"].criteria[1]


# --------------------------------------------------------------------------- report 2: parts


def test_astable_parts_report_lists_every_part_with_notes_library_facts_and_existence_checks(astable, tmp_path: Path):
    ir, lib = astable
    text = parts_report(ir, lib)
    assert text.startswith("# 부품 선정 보고서: astable\n")
    notes = template_for(ir).part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components} == {"Q1", "Q2", "R1", "R2", "R3", "R4", "C1", "C2", "J1"}
    for c in ir.components:
        assert f"## {c.ref} — {c.description}" in text
        assert f"| `{c.ref}` | {c.value} | {c.description} | `{c.symbol.library}:{c.symbol.name}` | `{c.footprint.library}:{c.footprint.name}` |" in text
        note = notes[c.ref]
        assert f"- 역할: {note.role}" in text and f"- 선정 이유: {note.why}" in text
        for criterion in note.criteria:
            assert f"- {criterion}" in text
        for sub in note.substitutes:
            assert sub.endswith(UNVERIFIED_SUBSTITUTE) and f"- {sub}" in text
        assert note.criteria and note.substitutes
    assert "| `Q1` | 2N3904 | NPN switching transistor | `Transistor_BJT:2N3904` | `Package_TO_SOT_THT:TO-92_Inline` | 기록 없음 | 1=E 2=B 3=C |" in text
    # every line under the substitutes heading carries the marker, and the heading says the names are unverified
    sections = text.split(f"### {SUBSTITUTES_HEADING}\n\n")[1:]
    assert len(sections) == 9
    for sec in sections:
        lines = [ln for ln in sec.split("\n\n", 1)[0].splitlines() if ln.strip()]
        assert lines and all(ln.startswith("- ") and "(검증되지 않음" in ln for ln in lines)
    assert "PN2222A" in text and "BC547 / BC548" in text and "`Transistor_BJT:BC547`" in text and "2N4401" in text
    # the astable's computed criteria
    assert "- V_CEO ≥ 2·V_cc = 10 V (여유 2배)" in text and "- I_C(max) ≥ 10 × V_cc/R_c = 50 mA" in text
    assert "- V_EBO ≥ V_cc − V_BE = 4.3 V" in text and "TO-92 핀 순서 E-B-C (KiCad 심볼 `Q_NPN_EBC`" in text
    assert "- 정격 전력 ≥ 2 × 계산 소비전력 = 2 × 25 mW = 50 mW" in text and "정전용량 64.817 nF ± 공차: Δf/f ≈ −ΔC/C 이므로 +10 % 이면 f ≈ 909.09 Hz" in text
    assert "- 정격 전압 ≥ 2·V_cc = 10 V" in text and "무극성 (세라믹 / 필름" in text
    # library facts, labelled as the library's own text (the synthetic library's Description is the symbol name, Datasheet is '~')
    assert "- Description (KiCad 라이브러리 기재): 2N3904" in text and "- Description (KiCad 라이브러리 기재): R" in text
    assert "- Datasheet (KiCad 라이브러리 기재): (비어 있음)" in text and "라이브러리 파일 `Transistor_BJT.kicad_sym`" in text
    assert "- 라이브러리 핀: 1=E 2=B 3=C" in text
    # IR facts: nothing grounded in this offline build, the SPICE binding is shown
    assert "- 제조사: 기록 없음" in text and "- MPN: 기록 없음" in text and "- SPICE: Q, 모델 `QNPN`, 노드 순서 3, 2, 1" in text
    assert "- 전기 특성: resistance = 1 kΩ [사용자 요구사항]" in text and "- SPICE: 넷리스트에서 제외 (connector, no electrical model)" in text
    # the existence result with its sub-checks, library files named by basename only
    assert "### 존재 확인 (`component.existence.Q1`)" in text and "- 최신 결과: **NOT_VERIFIED** (`parts.existence` v0.2)" in text
    assert "  - symbol: PASS: symbol Transistor_BJT:2N3904 found in Transistor_BJT.kicad_sym" in text
    assert "  - footprint: PASS: footprint Package_TO_SOT_THT:TO-92_Inline found in TO-92_Inline.kicad_mod" in text
    assert "  - datasheet_pointer: NOT_VERIFIED: no datasheet pointer" in text and "  - catalog: NOT_APPLICABLE: no catalog loaded" in text
    assert text.count("### 존재 확인") == 9 and "PASS" not in text.split("### 존재 확인")[0].split("## Q1")[1]  # statuses appear only in the existence rows
    assert "- req.oscillation_frequency (oscillation_frequency: 1 kHz)" in text and "- req.input_voltage (input_voltage: 5 V)" in text
    _assert_clean(text, tmp_path)


def test_divider_parts_report(divider, tmp_path: Path):
    ir, lib = divider
    text = parts_report(ir, lib)
    for ref in ("R1", "R2", "J1"):
        assert f"## {ref} — " in text and f"### 존재 확인 (`component.existence.{ref}`)" in text
    assert "- 역할: R1: 상단 분압 저항 (VIN–VOUT), 출력 비율을 정함" in text
    assert "- 저항값 14 kΩ (계산값 그대로: E 계열 반올림은 하지 않았음)" in text
    assert "- 정격 전력 ≥ 2 × 계산 소비전력 = 2 × 3.5 mW = 7 mW" in text
    assert "- 전기 특성: resistance = 14 kΩ [계산기 출력 (calc.divider.r1_for_v_out v0.6)]" in text
    assert "- req.output_voltage (output_voltage: 5 V)" in text
    _assert_clean(text, tmp_path)


def test_parts_report_without_a_library_or_with_a_missing_symbol_says_so(astable, tmp_path: Path):
    ir, _lib = astable
    text = parts_report(ir, None)
    assert "KiCad 라이브러리를 열 수 없어 `Transistor_BJT:2N3904` 의 기재를 읽지 못했습니다." in text
    assert "Description (KiCad 라이브러리 기재)" not in text and "- 역할: Q1:" in text  # the template notes do not need the library
    empty = KicadLibrary(roots=[tmp_path / "nowhere"])
    text = parts_report(ir, empty)
    assert "심볼 `Device:R` 이 라이브러리에 없습니다: 라이브러리 기재 없음." in text and "Description (KiCad 라이브러리 기재)" not in text
    _assert_clean(text, tmp_path)


# --------------------------------------------------------------------------- non-template designs and the helpers


def test_non_template_design_gets_honest_fallbacks(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = divider_with_connector_ir(tmp_path / "fx", lib)
    assert template_of(ir) is None and template_for(ir) is None
    theory, parts = theory_report(ir, lib), parts_report(ir, lib)
    assert f"- 설계 출처: {NO_TEMPLATE}" in theory and f"{NO_TEMPLATE}: 이 설계는 템플릿이 만들지 않았으므로" in theory
    assert "| `v_out` | 6 V | 계산기 출력 | `calc.divider.v_out` v0.6 ← v_in, r1, r2 | V_out = V_in * R2 / (R1 + R2) |" in theory
    assert "제약 조건 기록 없음." in theory and "- `dc_vin`: dc (source = VIN, start = 0 V, stop = 12 V, step = 1 V)" in theory
    assert parts.count(f"{NO_TEMPLATE_INFO}: 역할·선정 이유·대체 기준은 템플릿이 만든 설계에만 있습니다.") == 3
    assert "- Description (KiCad 라이브러리 기재): R" in parts  # library facts are still listed
    assert "- MPN: RC0603FR-0710kL [공식 자료 (데이터시트 / 라이브러리)]" in parts and "https://www.example-vendor.com/ds/rc0603.pdf" in parts
    assert parts.count("기록 없음: COMPONENT_SELECTION 단계가 이 부품을 아직 확인하지 않았습니다.") == 3  # no existence result yet
    assert "- req.v_out (6 V output (half the input) within 1 %)" in parts
    assert SUBSTITUTES_HEADING not in parts.split("## R1")[1]
    circuit, final = circuit_report(ir, lib, None), final_report(ir, None)
    assert "| `R1` | `Resistor_SMD:R_0603_1608Metric` |" in circuit and NO_ROUTING in circuit and "### 배치 규칙과 그 한계" not in circuit  # hand placements, no copper
    assert "- 배치 도구 기록 없음: 위치의 출처는" in circuit and f"{NO_RUN_RECORD}: 단계 결과는" in circuit
    assert "| `v_out` |" in final and NO_MEASUREMENT in final and f"{NO_RUN_RECORD}: RELEASE 판정은" in final and "## 검증 매트릭스" in final
    for text in (theory, parts, circuit, final):
        _assert_clean(text, tmp_path)


def test_template_of_reads_the_structural_provenance(astable, divider):
    assert template_of(astable[0]) == ("astable", "0.1") and template_for(astable[0]) is TEMPLATES[3]
    assert template_of(divider[0]) == ("divider", "0.1") and template_for(divider[0]) is TEMPLATES[0]
    ir = astable[0].model_copy(deep=True)
    ir.topology = None  # a component's provenance still names the template
    assert template_of(ir) == ("astable", "0.1")


def test_base_template_hooks_default_to_no_theory_and_no_notes(astable):
    class Bare(Template):
        id, title, triggers, needs, serves = "bare", "bare", (), (), ()

        def build(self, ir, inputs, unusable, library, *, confirmed):  # pragma: no cover - never built
            raise NotImplementedError

    ir = astable[0]
    sections = Bare().theory(ir)
    assert [s.title for s in sections] == ["이론 설명 없음"] and isinstance(sections[0], TheorySection) and "bare" in sections[0].body
    assert Bare().part_notes(ir) == {}
    for t in TEMPLATES:  # every registered template overrides both hooks
        assert type(t).theory is not Template.theory and type(t).part_notes is not Template.part_notes
    note = PartNote(role="r", why="w")
    assert note.criteria == [] and note.substitutes == []


def test_strip_paths_removes_absolute_paths_but_not_formulas_or_urls():
    assert strip_paths("symbol Device:R found in /tmp/pytest-0/kicad/symbols/Device.kicad_sym") == "symbol Device:R found in Device.kicad_sym"
    assert strip_paths("C:\\Users\\x\\lib\\Device.kicad_sym read") == "Device.kicad_sym read"
    assert strip_paths("(footprint in /a/b/R.kicad_mod)") == "(footprint in R.kicad_mod)"
    formulas = "I_B = (V_cc − V_BE)/R_b, f = 1/(2·f), R1·R2/(R1+R2), 1 / ( 2 · R_b ) and https://www.onsemi.com/pub/Collateral/2N3903-D.PDF"
    assert strip_paths(formulas) == formulas


def test_quantity_formats_with_si_prefixes_and_unit_symbols():
    assert quantity(6.48172677616823e-08, "F") == "64.817 nF" and quantity(0.0005, "s") == "500 µs" and quantity(1000.0, "Hz") == "1 kHz"
    assert quantity(10_000.0, "ohm") == "10 kΩ" and quantity(0.7, "V") == "0.7 V" and quantity(0.00043, "A") == "430 µA"
    assert quantity(0.1) == "0.1" and quantity(0.0, "V") == "0 V" and quantity(None, "V") == NO_RECORD


def test_stage_report_table_and_the_builder_dispatch(astable, tmp_path: Path):
    assert REPORTS_DIR == "reports"
    assert STAGE_REPORTS == {Stage.ARCHITECTURE: "01_이론_보고서.md", Stage.COMPONENT_SELECTION: "02_부품선정_보고서.md", Stage.PCB: "03_회로_보고서.md", Stage.RELEASE: "04_최종_보고서.md"}
    ir, lib = astable
    assert build_stage_report(Stage.ARCHITECTURE, ir, lib, None) == theory_report(ir, lib)
    assert build_stage_report(Stage.COMPONENT_SELECTION, ir, lib, None) == parts_report(ir, lib)
    assert build_stage_report(Stage.PCB, ir, lib, None) == circuit_report(ir, lib, None)
    assert build_stage_report(Stage.RELEASE, ir, lib, None) == final_report(ir, None)
    for stage in STAGE_ORDER:
        if stage not in STAGE_REPORTS:
            assert build_stage_report(stage, ir, lib, None) is None and write_stage_report(stage, ir, lib, None, tmp_path) is None
    assert not (tmp_path / REPORTS_DIR).exists()


# --------------------------------------------------------------------------- report 3: circuit


@pytest.fixture
def astable_release(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary, PipelineState]:
    return _release(tmp_path, "astable", ASTABLE)


def test_circuit_report_lists_nets_placements_routing_and_the_reference_values(astable_release, tmp_path: Path):
    ir, lib, state = astable_release
    assert ir.pcb is not None and ir.pcb.tracks, state.outcome(Stage.PLACEMENT).message  # the synthetic astable routes completely
    text = circuit_report(ir, lib, state)
    assert text.startswith("# 회로 보고서: astable\n") and "## 회로도" in text and "## 배치" in text and "## 배선" in text
    # every net with its pins and role, every placement with its position
    for n in ir.nets:
        pins = ", ".join(f"{p.component_ref}.{p.pin_number}" for p in n.pins)
        assert f"| `{n.name}` | {n.kind.value} | {pins} | {n.provenance.note} |" in text
    for p in ir.pcb.placements:
        c = ir.component(p.component_ref)
        assert f"| `{p.component_ref}` | `{c.footprint.library}:{c.footprint.name}` | {p.x_mm:g} | {p.y_mm:g} | 0 | top | 계산기 출력 (placement.grid v0.1) |" in text
    assert "- 배치 파라미터 (provenance `derived_from`): `columns` = 4, `margin_mm` = 2.0, `spacing_mm` = 1.0" in text
    assert "### 배치 규칙과 그 한계" in text and "참조 순서로 놓이므로" in text and "kicad-cli DRC 만이 판정" in text
    # the routing parameters come from the first track's provenance, the per-net lengths sum to the total
    params = routing_params_of(ir)
    assert params == {"grid": "0.25", "width": "0.4", "clearance": "0.25", "via": "0.8/0.4", "edge": "0.3", "via_cost": "12.0", "bend_cost": "0.6"}
    assert "- 배선 파라미터 (첫 트랙의 provenance `params:` 항목): `grid` = 0.25, `width` = 0.4, `clearance` = 0.25, `via` = 0.8/0.4, `edge` = 0.3, `via_cost` = 12.0, `bend_cost` = 0.6" in text
    rows = _table_rows(text, "### 넷별 배선 통계")
    assert [r[0] for r in rows] == [f"`{n.name}`" for n in ir.nets] + ["**합계**"]
    per_net = sum(float(r[2]) for r in rows[:-1])
    assert per_net == pytest.approx(float(rows[-1][2]), abs=0.01) and int(rows[-1][1]) == len(ir.pcb.tracks) and int(rows[-1][4]) == len(ir.pcb.vias)
    stats = net_routing_stats(ir)
    assert sum(s["segments"] for s in stats) == len(ir.pcb.tracks) and float(rows[-1][2]) == pytest.approx(sum(s["length_mm"] for s in stats), abs=0.001)
    # the keep-out rules with the parameter values substituted
    assert "r = c + w/2 + g/2 = 0.25 + 0.2 + 0.125 = 0.575 mm" in text and "r = w + c + g/2 = 0.775 mm" in text and "e + w/2 = 0.500 mm" in text
    # IPC-2221 for 0.4 mm / 1 oz at 10 degC, checked against an inline computation of the formula
    area_mil2 = (0.4 / 0.0254) * (35.0 / 1000.0 / 0.0254)
    capacity = 0.048 * 10.0 ** 0.44 * area_mil2 ** 0.725
    assert ipc2221_current_a(0.4) == pytest.approx(capacity) and 1.2 < capacity < 1.3
    assert f"A = 0.4 mm × 35 µm = {area_mil2:.3f} mil²" in text and f"= {quantity(capacity, 'A', 4)}" in text
    assert "설계의 최대 정상 전류 I_C(sat) ≈ V_cc/R_c = 5 mA 이므로 여유는" in text
    longest = max(stats, key=lambda s: s["length_mm"])
    r_cu = 1.68e-8 * (longest["length_mm"] / 1000) / (0.4e-3 * 35e-6)
    assert copper_resistance_ohm(longest["length_mm"], 0.4) == pytest.approx(r_cu)
    assert f"- 가장 긴 넷 `{longest['net']}` ({longest['length_mm']:.3f} mm, 폭 0.4 mm) 의 구리 저항:" in text and quantity(r_cu, "ohm", 4) in text
    assert "공급 5 V 는 이 구간에 들고, 라우터 간격 0.25 mm ≥ 0.1 mm 입니다." in text
    # the IR-geometry verdicts and the stage messages, copied
    assert "- `pcb.routing.connectivity`: **PASS** (`pcb.routing` v0.1) — " in text and "(IR geometry check, not DRC)" in text
    assert "- `pcb.routing.clearance`: **NOT_VERIFIED** (`pcb.routing` v0.1) — no clearance limit in ir.pcb.manufacturing" in text
    assert "- `placement`: **NOT_VERIFIED** — " in text and "routing.maze 0.1:" in text
    assert "- `pcb`: **PASS** — astable.kicad_pcb" in text and "- `drc`: **NOT_VERIFIED** — kicad-cli not available" in text
    assert "PASS" not in text.split("## 배선")[0]  # statuses appear only where results are quoted
    _assert_clean(text, tmp_path)
    before = ir.content_hash()
    assert circuit_report(ir, lib, state) == text and ir.content_hash() == before


def test_circuit_report_of_an_unrouted_board_says_so(tmp_path: Path):
    ir, lib, state = _release(tmp_path, "skip", DIVIDER, extra={ROUTING_KEY: "skip"})
    assert ir.pcb is not None and ir.pcb.placements and ir.pcb.tracks == [] and ir.pcb.vias == []
    text = circuit_report(ir, lib, state)
    assert f"{NO_ROUTING}: IR 에 트랙·비아가 없습니다." in text and "### 넷별 배선 통계" not in text and "IPC-2221" not in text
    assert "| `R1` | `Resistor_SMD:R_0603_1608Metric` |" in text  # the placement is still reported
    assert "- `placement`: **NOT_VERIFIED** — " in text and "routing skipped by answer" in text
    assert state.outcome(Stage.RELEASE).status is S.FAIL and "- `pcb.routing.connectivity`: **FAIL**" in text
    assert ir.validation.latest("pcb.routing.connectivity").status is S.FAIL
    _assert_clean(text, tmp_path)


def test_circuit_report_without_a_record_or_a_board(astable, tmp_path: Path):
    ir, lib = astable  # run to COMPONENT_SELECTION: no board yet
    text = circuit_report(ir, lib, None)
    assert f"배치 {NO_RECORD}" in text and NO_ROUTING in text and f"{NO_RUN_RECORD}: 단계 결과는" in text
    assert "- `pcb.routing.connectivity`: 기록 없음" in text  # no board yet: the IR geometry checks have not run
    _assert_clean(text, tmp_path)


# --------------------------------------------------------------------------- report 4: final


def _matrix(text: str) -> dict[str, str]:
    """``check id -> status heading`` from the verification matrix section."""
    section = text.split("## 검증 매트릭스", 1)[1].split("## 산출물", 1)[0]
    out: dict[str, str] = {}
    status = None
    for line in section.splitlines():
        if line.startswith("### "):
            status = line[4:].split(" — ", 1)[0]
        elif line.startswith("| `") and status is not None:
            out[line.split("`")[1]] = status
    return out


def test_final_report_without_ngspice_has_no_measurement_and_copies_every_status(astable_release, tmp_path: Path):
    ir, _lib, state = astable_release
    text = final_report(ir, state)
    assert text.startswith("# 최종 보고서: astable\n")
    # requirements, every stage of the record, the recorded RELEASE outcome
    assert "| `req.input_voltage` | input_voltage | input_voltage: 5 V | explicit | given | 5 V | electrical | 사용자 요구사항 |" in text
    for stage in STAGE_ORDER:
        o = state.outcome(stage)
        assert f"| `{stage}` | {o.status} |" in text, stage
    release = state.outcome(Stage.RELEASE)
    assert f"- 기록된 상태: **{release.status}**" in text and f"- 기록된 메시지: {release.message}" in text
    # one row per expectation; the SPICE stage did not run here, so no measured value and no invented deviation
    rows = _table_rows(text, "## 이론값 대 시뮬레이션")
    assert [r[0] for r in rows] == [f"`{e.id}`" for e in ir.simulation.expectations] == ["`f_osc`", "`out_high`", "`out_low`"]
    verdict = ir.validation.latest("spice.f_osc")  # no engine here: the stage records only the summary, no per-expectation result
    assert rows[0][2:] == ["1 kHz", NO_MEASUREMENT, "-", "-", "100 Hz", str(verdict.status) if verdict is not None else NO_RECORD]
    assert rows[1][2:7] == ["5 V", NO_MEASUREMENT, "-", "-", "0.25 V"] and rows[2][2:7] == ["0 V", NO_MEASUREMENT, "-", "-", "0.25 V"]
    assert "PASS" not in "".join(r[7] for r in rows)
    # the matrix groups every latest result under its own status: a check that is not PASS is never listed as PASS
    latest = ir.validation.latest_by_check()
    matrix = _matrix(text)
    assert set(matrix) == set(latest)
    for check, status in matrix.items():
        assert status == str(latest[check].status), check
    assert any(s != "PASS" for s in matrix.values()) and "### PASS — 통과 (도구 증거 있음)" in text
    # artifacts by basename with freshness, remaining work with the reasons, a conclusion that claims nothing beyond the statuses
    assert "| `kicad_pcb` | astable.kicad_pcb | - | `compiler.kicad_pcb` v0.2 | sha256:" in text and "| 현재 IR 에서 생성 |" in text
    assert "## 남은 일" in text and "- `review.drc`: kicad.drc has not been run" in text
    assert "이 보고서는 저장된 상태 이상을 주장하지 않습니다." in text and f"기록된 RELEASE 상태는 {release.status} 입니다." in text
    _assert_clean(text, tmp_path)
    before = ir.content_hash()
    assert final_report(ir, state) == text and ir.content_hash() == before


def test_final_report_without_a_record(astable):
    ir, _lib = astable
    text = final_report(ir, None)
    assert f"{NO_RUN_RECORD}: `ai-eda run` 이 `pipeline.json` 에 남긴 단계 결과가 없습니다." in text
    assert f"{NO_RUN_RECORD}: RELEASE 판정은" in text and f"RELEASE 판정은 {NO_RUN_RECORD} 없습니다." in text
    assert "| `f_osc` |" in text and NO_MEASUREMENT in text


@needs_dll
def test_final_report_compares_theory_with_the_measured_values(tmp_path: Path):
    ir, _lib, state = _release(tmp_path, "osc", ASTABLE, spice=True)
    assert state.outcome(Stage.SPICE).status is S.PASS
    text = final_report(ir, state)
    rows = {r[0]: r for r in _table_rows(text, "## 이론값 대 시뮬레이션")}
    f_osc = ir.validation.latest("spice.f_osc")
    measured = f_osc.details["measured"]
    assert rows["`f_osc`"][2:] == ["1 kHz", quantity(measured, "Hz"), quantity(measured - 1000.0, "Hz"), f"{(measured - 1000.0) / 1000.0 * 100:+.3g} %", "100 Hz", "PASS"]
    high = ir.validation.latest("spice.out_high").details["measured"]
    assert rows["`out_high`"][3] == quantity(high, "V") and rows["`out_high`"][7] == "PASS"
    low = ir.validation.latest("spice.out_low").details["measured"]
    assert rows["`out_low`"][3] == quantity(low, "V") and rows["`out_low`"][5] == "-"  # a relative deviation from a 0 V nominal is meaningless
    freq = f_osc.details["frequency"]
    assert f"- `f_osc` 주파수 측정: 상승 에지 {freq['edges']}개, 첫 에지 {quantity(freq['first_edge_s'], 's')}" in text
    assert "- `spice` 최신 결과: **PASS** (`ngspice-shared` ngspice-42)" in text or "- `spice` 최신 결과: **PASS** (`ngspice-shared` ngspice-" in text
    assert NO_MEASUREMENT not in text
    _assert_clean(text, tmp_path)


# --------------------------------------------------------------------------- writers, the CLI and report.html


def test_write_stage_reports_are_views_not_artifacts(astable_release, tmp_path: Path):
    ir, lib, state = astable_release
    workdir = tmp_path / "out"
    before, n_results, artifacts = ir.content_hash(), len(ir.validation.results), dict(ir.artifacts)
    paths = [write_stage_report(stage, ir, lib, state, workdir) for stage in STAGE_REPORTS]
    assert [p.name for p in paths] == list(STAGE_REPORTS.values()) and all(p.parent == workdir / REPORTS_DIR for p in paths)
    for stage, p in zip(STAGE_REPORTS, paths):
        raw = p.read_bytes()
        assert b"\r" not in raw and raw.decode("utf-8") == build_stage_report(stage, ir, lib, state)
        _assert_clean(raw.decode("utf-8"), tmp_path)
    assert ir.content_hash() == before and len(ir.validation.results) == n_results and ir.artifacts == artifacts
    assert not any(Path(a.path).parent.name == REPORTS_DIR for a in ir.artifacts.values())
    assert REPORTS_DIR not in json.dumps(ir.design_dict()) and "보고서" not in json.dumps(ir.design_dict(), ensure_ascii=False)
    again = write_all_stage_reports(ir, lib, state, workdir)
    assert again == paths and [p.read_bytes() for p in paths] == [build_stage_report(s, ir, lib, state).encode("utf-8") for s in STAGE_REPORTS]
    elsewhere = write_all_stage_reports(ir, lib, state, workdir, reports_dir=tmp_path / "elsewhere")
    assert [p.parent for p in elsewhere] == [tmp_path / "elsewhere"] * 4 and [p.read_bytes() for p in elsewhere] == [p.read_bytes() for p in paths]


def _cli_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "demo") -> Path:
    """``ai-eda new`` on the synthetic template library (found through ``KICAD10_SYMBOL_DIR``); returns the ir.json path.

    kicad-cli discovery is disabled for the whole test: the runs below expect
    the no-kicad-cli outcome (ERC / DRC NOT_VERIFIED, exit 0, only the four
    reports in ``reports/``), and an ERC / DRC verdict on the synthetic
    library's schematics and boards is unmeasured on every machine.
    """
    lib = template_library(tmp_path / "kicad")
    monkeypatch.setenv("KICAD10_SYMBOL_DIR", str(lib.roots[0] / "symbols"))
    monkeypatch.setattr("ai_eda.tools.kicad.cli.find_kicad_cli", lambda: None)
    code, _out, err = _cli("new", name, "--dir", str(tmp_path / name), "--request", "5 V, 1 kHz astable")
    assert code == 0, err
    return tmp_path / name / "ir.json"


def _answers(values: dict[str, str]) -> list[str]:
    return [arg for k, v in values.items() for arg in ("--answer", f"{k}={v}")]


def test_cli_run_writes_the_four_reports_and_stage_reports_rewrites_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir_path = _cli_project(tmp_path, monkeypatch)
    reports = tmp_path / "demo" / REPORTS_DIR
    # run 1: the table is presented and the pipeline blocks at ARCHITECTURE - the design is empty, so no report yet
    code, out, err = _cli("run", str(ir_path), *_answers({**BASE, **ASTABLE}))
    assert code == 1 and "report written:" not in out and not reports.exists(), err
    # run 2: confirmed, the pipeline runs to RELEASE and every report is written as its stage completes
    code, out, err = _cli("run", str(ir_path), *_answers({CONFIRM_DESIGN_KEY: "yes"}))
    assert code == 0, err
    lines = [ln for ln in out.splitlines() if "report written:" in ln]
    assert lines == [f"  report written: {REPORTS_DIR}/{name}" for name in STAGE_REPORTS.values()]
    assert all(ln.startswith(" ") for ln in lines)  # never mistaken for a stage line (first word = stage)
    summary = next(ln for ln in out.splitlines() if "stage reports:" in ln)
    assert summary.startswith("  stage reports: ") and all(f"{REPORTS_DIR}/{name}" in summary for name in STAGE_REPORTS.values())
    assert sorted(p.name for p in reports.iterdir()) == sorted(STAGE_REPORTS.values())
    run_time = {name: (reports / name).read_bytes() for name in STAGE_REPORTS.values()}
    for raw in run_time.values():
        _assert_clean(raw.decode("utf-8"), tmp_path)
    # the reports are not artifacts and the IR hash the run recorded is the hash of the saved IR
    ir = CircuitIR.load(ir_path)
    record = load_pipeline_record(tmp_path / "demo").record
    assert record.ir_hash == ir.content_hash() and record.state.outcomes[-1].stage is Stage.RELEASE
    assert not any(REPORTS_DIR in a.path for a in ir.artifacts.values()) and REPORTS_DIR not in json.dumps(ir.design_dict())
    # report.html links the files that exist, relative to the workdir
    html = render_report_file(ir_path).read_text(encoding="utf-8")
    assert "<h2>Stage reports</h2>" in html
    for name in STAGE_REPORTS.values():
        assert f'<a href="{REPORTS_DIR}/{name}">{name}</a>' in html
    # stage-reports re-writes all four from ir.json + pipeline.json: the final report is byte-identical to the run-time file
    # (same IR, same outcomes), the earlier ones were written from earlier states and only need to exist
    ir_bytes, record_bytes = ir_path.read_bytes(), (tmp_path / "demo" / PIPELINE_FILE).read_bytes()
    for name in STAGE_REPORTS.values():
        (reports / name).unlink()
    code, out, err = _cli("stage-reports", str(ir_path))
    assert code == 0 and err == "", err
    assert out.splitlines() == [f"wrote {reports / name}" for name in STAGE_REPORTS.values()]
    assert (reports / STAGE_REPORTS[Stage.RELEASE]).read_bytes() == run_time[STAGE_REPORTS[Stage.RELEASE]]
    for name in STAGE_REPORTS.values():
        assert (reports / name).is_file()
    assert ir_path.read_bytes() == ir_bytes and (tmp_path / "demo" / PIPELINE_FILE).read_bytes() == record_bytes  # nothing is saved
    # --dir puts the files elsewhere; a missing ir.json is a usage error (exit 2, never 1)
    code, out, _ = _cli("stage-reports", str(ir_path), "--dir", str(tmp_path / "other"))
    assert code == 0 and sorted(p.name for p in (tmp_path / "other").iterdir()) == sorted(STAGE_REPORTS.values())
    assert (tmp_path / "other" / STAGE_REPORTS[Stage.RELEASE]).read_bytes() == run_time[STAGE_REPORTS[Stage.RELEASE]]
    code, out, err = _cli("stage-reports", str(tmp_path / "missing.json"))
    assert code == 2 and "missing.json" in err and out == ""
    # without a run record the reports say so on stderr and in the text
    (tmp_path / "demo" / PIPELINE_FILE).unlink()
    code, out, err = _cli("stage-reports", str(ir_path))
    assert code == 0 and "no run record" in err and NO_RUN_RECORD in (reports / STAGE_REPORTS[Stage.RELEASE]).read_text(encoding="utf-8")


def test_cli_stage_reports_without_pcb_names_the_second_run_of_a_blocked_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A design that only reached the presented table: stage-reports still writes all four with the honest fallbacks."""
    ir_path = _cli_project(tmp_path, monkeypatch)
    _cli("run", str(ir_path), *_answers({**BASE, **DIVIDER}))
    code, out, err = _cli("stage-reports", str(ir_path))
    assert code == 0, err
    text = (tmp_path / "demo" / REPORTS_DIR / STAGE_REPORTS[Stage.PCB]).read_text(encoding="utf-8")
    assert NO_TEMPLATE in text and NO_ROUTING in text and "- `pcb`: 도달하지 않음" in text
    final = (tmp_path / "demo" / REPORTS_DIR / STAGE_REPORTS[Stage.RELEASE]).read_text(encoding="utf-8")
    assert "실행이 필수 질문에서 멈췄습니다" in final and "| `release` | - | 도달하지 않음 | - |" in final
    _assert_clean(text, tmp_path)
    _assert_clean(final, tmp_path)


def test_a_raising_builder_never_breaks_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir_path = _cli_project(tmp_path, monkeypatch)
    _cli("run", str(ir_path), *_answers({**BASE, **ASTABLE}))

    def boom(ir, record):
        raise RuntimeError("builder defect")

    monkeypatch.setattr("ai_eda.report.stages.final_report", boom)
    code, out, err = _cli("run", str(ir_path), *_answers({CONFIRM_DESIGN_KEY: "yes"}))
    assert code == 0, err  # the run's own outcome (NOT_VERIFIED without kicad-cli) is untouched
    assert f"  could not write {REPORTS_DIR}/{STAGE_REPORTS[Stage.RELEASE]}: RuntimeError: builder defect" in err
    reports = tmp_path / "demo" / REPORTS_DIR
    assert sorted(p.name for p in reports.iterdir()) == sorted(name for stage, name in STAGE_REPORTS.items() if stage is not Stage.RELEASE)
    assert [ln for ln in out.splitlines() if "report written:" in ln] == [f"  report written: {REPORTS_DIR}/{name}" for stage, name in STAGE_REPORTS.items() if stage is not Stage.RELEASE]
    record = load_pipeline_record(tmp_path / "demo").record
    assert record.aborted is None and record.state.outcomes[-1].stage is Stage.RELEASE and not record.state.blocked
    assert "release" in {ln.split()[0] for ln in out.splitlines() if ln and not ln.startswith(" ")}


def test_orchestrator_after_stage_sees_every_outcome_as_it_is_appended(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = divider_with_connector_ir(tmp_path, lib)
    seen: list[tuple[Stage, int]] = []
    from tests.fixtures_kicad import SCOPE_ANSWERS
    from ai_eda.agents import AgentContext

    state = Orchestrator(AgentContext(workdir=tmp_path, tools={"kicad_library": lib}, answers=SCOPE_ANSWERS)).run(
        ir, stop_after=Stage.IR_BUILD, after_stage=lambda stage, st: seen.append((stage, len(st.outcomes))))
    assert seen == [(o.stage, i + 1) for i, o in enumerate(state.outcomes)] and seen[-1][0] is Stage.IR_BUILD


# --------------------------------------------------------------------------- regressions from the stage-reports review


def test_astable_theory_e12_sentence_follows_the_design_capacitor(astable):
    """The 'nearest E12 value' sentence is computed from the IR's C, never hard-coded for the 1 kHz demo."""
    ir, lib = astable
    assert nearest_e12(64.817e-9) == pytest.approx(68e-9) and nearest_e12(6.4817e-9) == pytest.approx(6.8e-9)
    assert nearest_e12(9.5) == pytest.approx(10.0) and nearest_e12(0.95) == pytest.approx(1.0) and nearest_e12(None) is None and nearest_e12(0.0) is None
    ln_term = math.log((2 * 5.0 - 0.7) / (5.0 - 0.7))
    text = theory_report(ir, lib)
    assert f"가장 가까운 E12 값 68 nF 를 쓰면 (식 1)로 f = {quantity(1 / (2 * 10_000.0 * 68e-9 * ln_term), 'Hz')} 가 됩니다." in text
    fast = ir.model_copy(deep=True)
    c_10k = 1 / (2 * 10_000.0 * 10_000.0 * ln_term)  # the calculator's C for 10 kHz with the same R_b / V_cc / V_BE
    for key, value in (("f_osc", 10_000.0), ("f_osc_design", 10_000.0), ("c", c_10k)):
        fast.parameters[key] = fast.parameters[key].model_copy(update={"value": value})
    text = theory_report(fast, lib)
    assert f"| **{quantity(c_10k, 'F')}** |" in text and "68 nF" not in text
    assert f"가장 가까운 E12 값 6.8 nF 를 쓰면 (식 1)로 f = {quantity(1 / (2 * 10_000.0 * 6.8e-9 * ln_term), 'Hz')} 가 됩니다." in text
    assert "9.532 kHz" in text


def _drawing(body: str) -> list[str]:
    return body.split("```\n", 1)[1].split("\n```", 1)[0].splitlines()


def _columns(line: str, marks: str) -> list[int]:
    return [i for i, ch in enumerate(line) if ch in marks]


def test_astable_drawing_wires_r3_to_q1_b_and_r4_to_q2_b_in_aligned_columns(astable):
    """The ASCII schematic follows the built nets (R3 on Q1_B with C2, R4 on Q2_B with C1) and every row shares the bus columns."""
    ir, _lib = astable
    nets = {n.name: {f"{p.component_ref}" for p in n.pins} for n in ir.nets}
    assert {"R3", "C2", "Q1"} <= nets["Q1_B"] and {"R4", "C1", "Q2"} <= nets["Q2_B"]
    for lines in (_drawing(template_for(ir).theory(ir)[0].body), astable_drawing("1.2345 kΩ", "10.567 kΩ").splitlines()):
        vcc, labels, bars1, coupling, bars2, names, emitters = lines
        cols = _columns(vcc, "+")
        assert len(cols) == 4 and vcc.startswith("VCC ---+")
        assert [labels.find(lab) for lab in ("R1", "R4", "R3", "R2")] == cols  # each resistor starts on its bus column
        assert _columns(bars1, "|") == cols and _columns(bars2, "|") == cols
        assert _columns(coupling, "+") == cols
        c0, c1, c2, c3 = cols
        assert coupling[c0 - 4:c0] == "Q1_C" and "C1" in coupling[c0:c1] and coupling[c1 + 1:c1 + 5] == "Q2_B"  # R1 column -> Q1_C, C1 -> R4 column = Q2_B
        assert coupling[c2 - 4:c2] == "Q1_B" and "C2" in coupling[c2:c3] and coupling[c3 + 1:c3 + 4] == "OUT"  # R3 column = Q1_B, C2 -> R2 column = OUT
        assert [names.find(n) + 3 for n in ("Q1.C", "Q2.B", "Q1.B", "Q2.C")] == cols
        assert emitters.strip().startswith("Q1.E = GND") and emitters.strip().endswith("Q2.E = GND")
    assert "R1 1.2345 kΩ" in astable_drawing("1.2345 kΩ", "10.567 kΩ")


def test_remaining_work_strips_paths_from_stored_messages(astable, tmp_path: Path):
    """A NOT_VERIFIED message that embeds an absolute path (existence.py's 'tampered: the archived datasheet <path>') is cut to the basename."""
    ir, _lib = astable
    ir = ir.model_copy(deep=True)
    path = str(tmp_path / "proj" / "sources" / "abc123.pdf")
    ir.validation.add(ValidationResult(check_id="component.existence.Q1", status=S.NOT_VERIFIED, tool="parts.existence", tool_version="0.2",
                                       message=f"datasheet_archived: tampered: the archived datasheet {path} no longer hashes to abc123"))
    text = final_report(ir, None)
    remaining = text.split("## 남은 일", 1)[1]
    assert "- `component.existence.Q1`: datasheet_archived: tampered: the archived datasheet abc123.pdf no longer hashes to abc123" in remaining
    assert path not in text and str(tmp_path) not in text
    _assert_clean(text, tmp_path)


def test_unrouted_board_routing_section_makes_no_drc_claim(tmp_path: Path):
    """Without copper the routing section says so and defers to the recorded `drc` outcome; it never asserts a DRC verdict."""
    ir, lib, state = _release(tmp_path, "skip", DIVIDER, extra={ROUTING_KEY: "skip"})
    assert ir.validation.latest("kicad.drc") is None and state.outcome(Stage.DRC).status is S.NOT_VERIFIED
    text = circuit_report(ir, lib, state)
    routing = text.split("## 배선", 1)[1].split("### IR 기하 검사 결과", 1)[0]
    assert f"{NO_ROUTING}: IR 에 트랙·비아가 없습니다. 보드는 배치만 된 상태입니다(구리 없음)." in routing
    assert "실패합니다" not in routing and "unconnected_items" not in routing and "판정하지 않으며" in routing and "`drc` 항목" in routing
    assert "- `drc`: **NOT_VERIFIED** — kicad-cli not available" in text
    # a design without a board is not 'placed only'
    no_board = ir.model_copy(deep=True)
    no_board.pcb = None
    text = circuit_report(no_board, lib, None)
    assert f"{NO_ROUTING}: IR 에 보드가 없어 트랙·비아도 없습니다." in text and "배치만 된 상태" not in text and "unconnected_items" not in text
    _assert_clean(text, tmp_path)


def test_placement_rule_names_the_prefix_then_number_order(astable_release):
    """The placement paragraph describes natural_ref_key's order (C1 < J1 < Q1 < R1 < R10), which the positions table above it shows."""
    ir, lib, state = astable_release
    text = circuit_report(ir, lib, state)
    rows = _table_rows(text, "## 배치")
    refs = [r[0].strip("`") for r in rows]
    assert refs == sorted(refs, key=natural_ref_key) == ["C1", "C2", "J1", "Q1", "Q2", "R1", "R2", "R3", "R4"]
    assert "참조 지정자의 자연 순서(접두 문자 순, 그다음 번호 순: C1, C2, …, J1, …, Q1, …, R1, R2, …, R10)입니다." in text
    assert "(R1, R2, …, C1, …)" not in text


def test_final_report_tolerance_of_a_zero_nominal_with_only_tol_rel_is_none(astable):
    """The 허용치 column follows judge(): a relative tolerance on a nominal of 0 is no tolerance, so the cell is '-', never '0 V'."""
    ir, _lib = astable
    ir = ir.model_copy(deep=True)
    out_low = next(e for e in ir.simulation.expectations if e.id == "out_low")
    assert float(out_low.nominal.value) == 0.0
    out_low.tol_abs = None
    out_low.tol_rel = Traced(value=0.1, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="hand-built: relative tolerance on zero"))
    text = final_report(ir, None)
    rows = {r[0]: r for r in _table_rows(text, "## 이론값 대 시뮬레이션")}
    assert rows["`out_low`"][2] == "0 V" and rows["`out_low`"][6] == "-" and rows["`out_low`"][7] == NO_RECORD
    assert "0 V |" not in "| ".join(rows["`out_low`"][6:])
    assert "공칭값이 0 이면 tol_rel 은 허용치가 아니므로 tol_abs 만 셉니다" in text
    theory = theory_report(ir, None)
    assert "공칭값이 0 이면 tol_rel 은 허용치가 아니므로 tol_abs 가 있어야 판정합니다(없으면 UNRESOLVED)" in theory


def test_resistor_criteria_label_follows_the_parameter_provenance(astable, divider):
    """A confirmed template choice (R_c, R_b, the divider's R2) is labelled as such; only a calculator output is 'unrounded'."""
    ir, lib = astable
    assert ir.parameters["r_c"].provenance.kind is ProvenanceKind.USER_REQUIREMENT and ir.parameters["c"].provenance.kind is ProvenanceKind.DERIVED
    text = parts_report(ir, lib)
    assert "- 저항값 1 kΩ (템플릿 선택값, 사용자 확인)" in text and "- 저항값 10 kΩ (템플릿 선택값, 사용자 확인)" in text
    assert "계산값 그대로: E 계열 반올림은 하지 않았음" not in text  # no astable resistor is a calculator output
    ir, lib = divider
    text = parts_report(ir, lib)
    assert "- 저항값 14 kΩ (계산값 그대로: E 계열 반올림은 하지 않았음)" in text and "- 저항값 10 kΩ (템플릿 선택값, 사용자 확인)" in text
    assert "저항값 10 kΩ (계산값 그대로" not in text


def test_led_theory_and_parts_reports(tmp_path: Path):
    """The led template's hooks render with the design's numbers (5 V / 2 V / 10 mA -> 300 Ω, 30 mW) and are deterministic."""
    ir, lib = _build(tmp_path, "led", LED)
    assert template_of(ir) == ("led", "0.1")
    text = theory_report(ir, lib)
    assert "R = (V_in − V_f)/I_f" in text and "= (5 V − 2 V)/10 mA = 300 Ω   (`calc.led.R`)" in text and "= 3 V/300 Ω = 10 mA   (`calc.led.I`)" in text
    assert "| 저항 소비전력 | P(R) = (V_in − V_f)·I = (V_in − V_f)²/R | 30 mW |" in text and "| LED 소비전력 | P(LED) = V_f·I | 20 mW |" in text
    assert "V_f 가 0.1 V 커지면 전류는 약 3.33 % 줄어듭니다" in text and "|측정값 − 10 mA| ≤ 1 % × 10 mA = 100 µA" in text
    assert "| `r_led` | 300 Ω | 계산기 출력 | `calc.led.R` v0.6 ← v_in, v_f, i_f | R = (V_supply - V_f) / I_f |" in text
    parts = parts_report(ir, lib)
    notes = template_for(ir).part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components} == {"R1", "D1", "J1"}
    assert "- 저항값 300 Ω (계산값 그대로: E 계열 반올림은 하지 않았음)" in parts and "- 정격 전력 ≥ 2 × 계산 소비전력 = 2 × 30 mW = 60 mW" in parts
    assert "- 데이터시트의 V_f(@ I_f = 10 mA) 가 요구값 2 V 과 같을 것 (다르면 R1 을 다시 계산)" in parts and "- 최대 순방향 전류 I_f(max) ≥ 10 mA 에 여유" in parts
    for note in notes.values():
        assert note.criteria and note.substitutes and all(sub.endswith(UNVERIFIED_SUBSTITUTE) for sub in note.substitutes)
    assert theory_report(ir, lib) == text and parts_report(ir, lib) == parts
    for t in (text, parts):
        _assert_clean(t, tmp_path)


def test_rc_lowpass_theory_and_parts_reports(tmp_path: Path):
    """The rc_lowpass hooks render with the design's numbers (1 kHz, 100 nF -> 1.5915 kΩ, τ = 159.15 µs) and state judge()'s bracket rule."""
    ir, lib = _build(tmp_path, "rc_lowpass", RC)
    assert template_of(ir) == ("rc_lowpass", "0.1")
    text = theory_report(ir, lib)
    assert "R = 1/(2π·f_c·C) = 1/(2π·1 kHz·100 nF) = 1.5915 kΩ" in text and "τ = R·C = 159.15 µs   (`calc.rc.tau`)" in text and "검산: f_c = 1/(2π·R·C) = 1000 Hz" in text
    assert "| f_c = 1 kHz | 1/√2 = 0.70711 (−3.01 dB) | −45° |" in text and "| 10·f_c = 10 kHz | 0.0995 (−20.04 dB) | −84.3° |" in text
    assert "스위프는 `dec` 100 점/decade, f_start = f_c/100 = 10 Hz 부터 f_stop = 100·f_c = 100 kHz 까지" in text
    assert "|측정값 − 0.70711| ≤ 2 % × 0.70711 = 0.014142" in text
    # judge()'s rule for an interpolated bracket: both inside PASS, both outside on the same side FAIL, otherwise UNRESOLVED
    assert "둘 다 허용치 안이면 PASS, 둘 다 같은 쪽으로 허용치 밖이면 FAIL, 그 밖(한쪽만 밖이거나 양쪽으로 걸침)은 격자가 허용치를 분해하지 못하므로 UNRESOLVED" in text
    assert "두 이웃이 허용치 밖이면 UNRESOLVED" not in text
    parts = parts_report(ir, lib)
    notes = template_for(ir).part_notes(ir)
    assert set(notes) == {c.ref for c in ir.components} == {"R1", "C1", "J1"}
    assert "- 저항값 1.5915 kΩ (계산값 그대로: E 계열 반올림은 하지 않았음)" in parts and "- 정격 전력 ≥ 2 × 계산 소비전력 = 2 × 628.32 µW = 1.2566 mW" in parts
    assert "- 정전용량 100 nF, 공차 5 % 이하 (Δf_c/f_c ≈ −ΔC/C: 커패시터 오차가 그대로 차단 주파수 오차가 됨)" in parts
    for note in notes.values():
        assert note.criteria and note.substitutes and all(sub.endswith(UNVERIFIED_SUBSTITUTE) for sub in note.substitutes)
    assert theory_report(ir, lib) == text and parts_report(ir, lib) == parts
    for t in (text, parts):
        _assert_clean(t, tmp_path)


#: a stand-in kicad-cli with the real CLI surface and report shapes whose ERC always finds one warning (a FAIL under WARNING_POLICY)
FAKE_KICAD_CLI = """#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv[1:]
def opt(flag):
    return args[args.index(flag) + 1]
if args[:1] == ["version"]:
    print("10.0.6-fake"); sys.exit(0)
common = {"$schema": "x", "source": args[-1], "date": "", "kicad_version": "10.0.6-fake", "included_severities": ["error", "warning"], "ignored_checks": []}
if args[:2] == ["sch", "erc"]:
    warning = {"type": "lib_symbol_mismatch", "description": "fake warning", "severity": "warning", "excluded": False, "items": []}
    Path(opt("-o")).write_text(json.dumps({**common, "sheets": [{"uuid_path": "/", "path": "/", "violations": [warning]}]}), encoding="utf-8")
    sys.exit(5)
if args[:2] == ["pcb", "drc"]:
    Path(opt("-o")).write_text(json.dumps({**common, "violations": [], "unconnected_items": [], "schematic_parity": []}), encoding="utf-8")
    sys.exit(0)
if args[:3] == ["pcb", "export", "gerbers"]:
    out = Path(opt("-o")); out.mkdir(parents=True, exist_ok=True)
    stem, listed = Path(args[-1]).stem, []
    for layer in opt("-l").split(","):
        name = f"{stem}-{layer.replace('.', '_')}.gbr"
        (out / name).write_text("%TF.FileFunction,Other," + layer + "*%\\n%FSLAX46Y46*%\\n%MOMM*%\\nM02*\\n", encoding="utf-8"); listed.append({"Path": name})
    (out / f"{stem}-job.gbrjob").write_text(json.dumps({"FilesAttributes": listed}), encoding="utf-8"); sys.exit(0)
if args[:3] == ["pcb", "export", "drill"]:
    out = Path(opt("-o")); out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args[-1]).stem}.drl").write_text("M48\\nM30\\n", encoding="utf-8"); sys.exit(0)
print("unsupported: " + " ".join(args), file=sys.stderr); sys.exit(1)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="a script stand-in is not executable through PATH on Windows")
def test_cli_runs_ignore_a_kicad_cli_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The CLI-run tests measure the no-kicad-cli outcome: a kicad-cli on PATH whose ERC would FAIL the run (exit 1, erc.json in reports/) is not picked up."""
    fake = tmp_path / "bin" / "kicad-cli"
    fake.parent.mkdir()
    fake.write_text(FAKE_KICAD_CLI, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ.get("PATH", ""))
    assert Path(shutil.which("kicad-cli")) == fake  # the stand-in would be discovered by an unpinned run
    ir_path = _cli_project(tmp_path, monkeypatch)
    _cli("run", str(ir_path), *_answers({**BASE, **ASTABLE}))
    code, out, err = _cli("run", str(ir_path), *_answers({CONFIRM_DESIGN_KEY: "yes"}))
    assert code == 0, err
    stages = {ln.split()[0]: ln for ln in out.splitlines() if ln and not ln.startswith(" ")}
    assert "kicad-cli not available" in stages["erc"] and "kicad-cli not available" in stages["drc"] and "NOT_VERIFIED" in stages["release"]
    assert sorted(p.name for p in (tmp_path / "demo" / REPORTS_DIR).iterdir()) == sorted(STAGE_REPORTS.values())
