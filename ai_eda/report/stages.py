"""The four Korean Markdown stage reports: theory, parts, circuit and final.

Invariant: every report is a *view*, like ``report.html`` (which stays the
English view). A builder computes no :class:`~ai_eda.ir.ValidationStatus`
(every status it prints is a stored ``ValidationResult.status`` or a recorded
``StageOutcome.status``, copied verbatim), registers no artifact, is never
hashed into the IR, never saves the IR, and prints **no wall-clock and no
absolute path**: library files are named by their lib id, files by their
basename, and a path that reaches a builder inside a stored message is cut
down to its basename (:func:`strip_paths`). So the same IR and the same run
record give byte-identical text. Numbers come only from the IR
(``ir.parameters`` with their calculator notes - the formulas -,
``ir.validation``), the run record, the KiCad library on disk and the
templates' own theory text (:meth:`~ai_eda.design.base.Template.theory`,
:meth:`~ai_eda.design.base.Template.part_notes`); a value the IR does not
hold is printed as :data:`~ai_eda.design.base.NO_RECORD`, never guessed. No
LLM is involved anywhere; the reports are Korean prose because the reader
reads Korean. ``<workdir>/reports/*.md`` are not artifacts.

The stage that produces each report's content is the key of
:data:`STAGE_REPORTS`; the builders take only what they need, so a report
can be rebuilt from a saved IR without re-running anything. The run record
a builder reads (:data:`RunRecord`) is either the saved ``pipeline.json``
(:class:`~ai_eda.report.pipeline_log.PipelineRecord`, ``ai-eda
stage-reports``) or the live :class:`~ai_eda.workflow.orchestrator.PipelineState`
of the run that is writing the report (``ai-eda run`` through
``Orchestrator.run(after_stage=...)``); only the outcomes' stage, status,
message and question count are printed, never ``at``, so the final report
written at RELEASE equals the one ``stage-reports`` re-writes afterwards.
The reference formulas of the circuit report (IPC-2221 current capacity and
table 6-1 B2 spacing, copper resistance) are display values computed from
the IR's own geometry, named as such, never a verdict and never written to
the IR.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from ai_eda.design import TEMPLATES
from ai_eda.design.base import CHOICE_NOTE_PREFIX, NO_RECORD, TOOL_ID, PartNote, Template, TheorySection, parameter_value, quantity
from ai_eda.ir import CircuitIR, Component, Provenance, ProvenanceKind, Reduce, Traced, ValidationResult, ValidationStatus
from ai_eda.parts.existence import CHECK_PREFIX as EXISTENCE_PREFIX
from ai_eda.report.pipeline_log import PipelineRecord
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.grid import PLACER_ID
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK
from ai_eda.validation.layout import CLEARANCE_CHECK, CONNECTIVITY_CHECK
from ai_eda.workflow.orchestrator import PipelineState, StageOutcome
from ai_eda.workflow.stages import STAGE_ORDER, Stage

#: the report each stage's outcome completes (written right after that stage by ``ai-eda run``)
STAGE_REPORTS: dict[Stage, str] = {
    Stage.ARCHITECTURE: "01_이론_보고서.md",
    Stage.COMPONENT_SELECTION: "02_부품선정_보고서.md",
    Stage.PCB: "03_회로_보고서.md",
    Stage.RELEASE: "04_최종_보고서.md",
}
#: where the reports live under the project workdir
REPORTS_DIR = "reports"
#: header line for a design no template built
NO_TEMPLATE = "템플릿 설계가 아님"
#: what the parts report prints for a part its template says nothing about
NO_TEMPLATE_INFO = "템플릿 정보 없음"
#: the heading under which substitute names are printed: they are not verified by the pipeline
SUBSTITUTES_HEADING = "대체 후보 (파이프라인이 검증하지 않은 이름)"
#: what a report says where it needs a run record and has none
NO_RUN_RECORD = "실행 기록 없음"
#: the KiCad symbol properties the parts report shows, labelled as the library's own text
LIBRARY_PROPERTIES: tuple[str, ...] = ("Description", "Datasheet", "ki_keywords")
#: an absolute POSIX or Windows path inside a stored message (two or more segments, so a formula's ``a/(b+c)`` is not one;
#: not the ``//`` of a URL): replaced by its basename
_PATH_RE = re.compile(r"(?<![:/\w])(?:[A-Za-z]:\\|/)(?:[^\s'\"\]\),;|/\\]+[/\\])+[^\s'\"\]\),;|/\\]+")

PROVENANCE_LABELS: dict[ProvenanceKind, str] = {
    ProvenanceKind.USER_REQUIREMENT: "사용자 요구사항",
    ProvenanceKind.AUTHORITATIVE: "공식 자료 (데이터시트 / 라이브러리)",
    ProvenanceKind.DERIVED: "계산기 출력",
    ProvenanceKind.ASSUMPTION: "가정 (사용자 확인 필요)",
    ProvenanceKind.LLM_GENERATED: "모델 제안 (검증되지 않음)",
}
CHOICE_LABEL = "사용자 확인 선택값"

REDUCE_LABELS: dict[Reduce, str] = {
    Reduce.VALUE: "값 (동작점)",
    Reduce.AT: "스위프 축의 한 점에서의 값",
    Reduce.FINAL: "마지막 값",
    Reduce.MAX: "최댓값",
    Reduce.MIN: "최솟값",
    Reduce.FREQUENCY: "상승 에지 주파수",
}


# --- small text helpers ----------------------------------------------------------


def strip_paths(text: str) -> str:
    """A stored message with every absolute path replaced by its basename (a report names files, never where they are).

    Only tool messages and provenance notes pass through here; the templates'
    own theory text is printed as written (it holds formulas, never a path).
    """
    return _PATH_RE.sub(lambda m: m.group(0).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1], text)


def _cell(text: object) -> str:
    """One Markdown table cell: pipes escaped, newlines folded, paths stripped."""
    return strip_paths(str(text)).replace("|", "\\|").replace("\r", "").replace("\n", " ")


def _traced_text(t: Traced | None) -> str:
    if t is None:
        return NO_RECORD
    if isinstance(t.value, bool) or not isinstance(t.value, (int, float)):
        return f"{t.value}" + (f" {t.unit}" if t.unit else "")
    return quantity(float(t.value), t.unit)


def _version(version: str | None) -> str:
    """`` v0.2`` for a numeric version, `` ngspice-42`` for one that already names itself, nothing for ``None``."""
    if not version:
        return ""
    return f" v{version}" if version[0].isdigit() else f" {version}"


def _tool_text(tool: str | None, version: str | None) -> str:
    """````tool` vX`` for a result's tool, ``(도구 없음)`` without one."""
    return f"`{tool}`{_version(version)}" if tool else "(도구 없음)"


def _provenance_text(p: Provenance) -> str:
    """``kind (tool vX)`` for a fact's origin, in words a reader can weigh."""
    label = PROVENANCE_LABELS.get(p.kind, str(p.kind))
    if p.tool:
        label += f" ({p.tool}{_version(p.tool_version)})"
    return label


def _lib_id(component: Component, which: str) -> str:
    ref = getattr(component, which)
    return f"{ref.library}:{ref.name}" if ref is not None else NO_RECORD


def _pins_text(component: Component) -> str:
    """``1=E 2=B 3=C``; an unnamed pin (``~``) shows its number only."""
    return " ".join(f"{p.number}={p.name}" if p.name not in ("", "~") else p.number for p in component.pins) or NO_RECORD


def _table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _requirement_text(ir: CircuitIR, rid: str) -> str:
    for r in ir.requirements.requirements:
        if r.id == rid:
            return f"{rid} ({r.text})"
    return f"{rid} (요구사항 본문 {NO_RECORD})"


# --- which template built the design --------------------------------------------------


def template_of(ir: CircuitIR) -> tuple[str, str] | None:
    """``(template id, version)`` from the structural provenance tool ``design.template.<id>`` on the topology or a component; ``None`` without one."""
    prefix = f"{TOOL_ID}."
    candidates: list[Provenance] = []
    if ir.topology is not None:
        candidates.append(ir.topology.provenance)
    candidates.extend(c.provenance for c in ir.components)
    for p in candidates:
        if p.tool and p.tool.startswith(prefix):
            return p.tool[len(prefix):], p.tool_version or NO_RECORD
    return None


def template_for(ir: CircuitIR) -> Template | None:
    """The registered template whose id built ``ir``, or ``None``."""
    found = template_of(ir)
    if found is None:
        return None
    for t in TEMPLATES:
        if t.id == found[0]:
            return t
    return None


def _header(title: str, ir: CircuitIR) -> list[str]:
    found = template_of(ir)
    template_line = f"템플릿 `{found[0]}` v{found[1]}" if found is not None else NO_TEMPLATE
    return [
        f"# {title}: {ir.project.name}",
        "",
        f"- 프로젝트 id: `{ir.project.id}`",
        f"- 설명: {ir.project.description or NO_RECORD}",
        f"- 설계 출처: {template_line}",
        "- 이 보고서는 IR(설계 데이터)과 그 검증 기록을 읽어 만든 뷰입니다. 판정을 새로 계산하지 않고, 산출물로 등록되지 않으며, IR 에 아무것도 쓰지 않습니다. "
        "IR 에 없는 값은 '" + NO_RECORD + "' 으로 표시합니다.",
        "",
    ]


# --- 1. theory report ----------------------------------------------------------------


def _parameter_rows(ir: CircuitIR) -> list[list[object]]:
    rows: list[list[object]] = []
    for key, t in ir.parameters.items():
        p = t.provenance
        note = p.note or ""
        if p.kind is ProvenanceKind.USER_REQUIREMENT and note.startswith(CHOICE_NOTE_PREFIX):
            parts = note.split(": ", 2)
            origin, basis = CHOICE_LABEL, parts[0].split("; ", 1)[-1]
            formula = parts[2] if len(parts) == 3 else note
        elif p.kind is ProvenanceKind.DERIVED and p.tool:
            origin = PROVENANCE_LABELS[ProvenanceKind.DERIVED]
            basis = _tool_text(p.tool, p.tool_version) + (f" ← {', '.join(p.derived_from)}" if p.derived_from else "")
            formula = note
        else:
            origin = PROVENANCE_LABELS.get(p.kind, str(p.kind))
            basis = ", ".join(p.derived_from) if p.derived_from else (p.tool or "-")
            formula = note
        rows.append([f"`{key}`", _traced_text(t), origin, basis, formula or "-"])
    return rows


def _simulation_section(ir: CircuitIR) -> list[str]:
    sim = ir.simulation
    out = ["## 시뮬레이션 설정", ""]
    if sim is None:
        out += ["시뮬레이션 설정 없음.", ""]
        return out
    out.append("### 자극(stimuli)")
    out.append("")
    for s in sim.stimuli:
        params = ", ".join(f"{k} = {_traced_text(v)}" for k, v in s.params.items())
        out.append(f"- `{s.id}` ({s.source}, {s.kind.value}) {s.net} → {s.reference_net}: {_traced_text(s.value)}" + (f"; {params}" if params else "")
                   + (f" — {s.provenance.note}" if s.provenance.note else ""))
    out += ["", "### 해석(analyses)", ""]
    for a in sim.analyses:
        params = ", ".join(f"{k} = {_traced_text(v)}" for k, v in a.params.items())
        out.append(f"- `{a.id}`: {a.kind.value}" + (f" ({params})" if params else "") + (f" — {a.provenance.note}" if a.provenance.note else ""))
    out += ["", "### 기대값(expectations)", ""]
    rows: list[list[object]] = []
    for e in sim.expectations:
        rows.append([
            f"`{e.id}`", e.analysis_id, f"`{e.vector}`", REDUCE_LABELS.get(e.reduce, str(e.reduce)) + (f" at {_traced_text(e.at)}" if e.at is not None else ""),
            _traced_text(e.nominal), _traced_text(e.tol_abs) if e.tol_abs is not None else "-",
            (f"{float(e.tol_rel.value) * 100:.4g} %" if e.tol_rel is not None else "-"), e.requirement_id or "(요구사항 없음)",
        ])
    out.append(_table(["id", "해석", "벡터", "축약", "이론 공칭값", "허용치(절대)", "허용치(상대)", "검증하는 요구사항"], rows))
    out += ["", "### 판정식", "", "    |측정값 − 공칭값| ≤ max(tol_abs, tol_rel · |공칭값|)", "",
            "허용치가 하나만 있으면 그 하나가 한계이고, 공칭값이 0 이면 tol_rel 은 허용치가 아니므로 tol_abs 가 있어야 판정합니다(없으면 UNRESOLVED). "
            "기대값은 부품 공칭값과 한 온도에서만 판정하며(공차 코너 없음), 결과가 없으면 PASS 가 아니라 NOT_VERIFIED 입니다.", ""]
    if any(e.reduce is Reduce.FREQUENCY for e in sim.expectations):
        out += [
            "### 주파수 측정식 (`Reduce.FREQUENCY`)", "",
            "파형 v(t)의 최소·최대에서 문턱을 정합니다.", "",
            "    swing = v_max − v_min",
            "    v_low = v_min + 0.25·swing,  v_mid = v_min + 0.5·swing,  v_high = v_min + 0.75·swing", "",
            "히스테리시스: v ≤ v_low 를 지난 뒤(무장) 처음으로 v ≥ v_high 가 되는 순간을 상승 에지 1개로 셉니다(문턱 근처의 채터링을 두 번 세지 않기 위해). "
            "에지 시각은 v_mid 를 지나는 두 표본 사이를 선형 보간합니다.", "",
            "    t_cross = t_i + (v_mid − v_i) · (t_{i+1} − t_i) / (v_{i+1} − v_i)", "",
            "N개(N ≥ 3)의 에지 시각 t_1 … t_N 에서", "",
            "    f_measured = (N − 1) / (t_N − t_1)", "",
            "에지가 3개 미만이거나 swing 이 엔진 자체의 수렴 허용오차(reltol·max|v| + vntol) 이하이면 '발진 없음' 으로 FAIL 이며 절대 PASS 가 되지 않습니다.", "",
        ]
    return out


def theory_report(ir: CircuitIR, library: KicadLibrary | None = None) -> str:
    """Report 1: the circuit theory of the template that built ``ir`` with the IR's numbers, its inputs, constraints and simulation setup.

    ``library`` is accepted for symmetry with the other builders and not read:
    theory needs no library fact.
    """
    out = _header("이론 보고서", ir)
    out += [
        "## 설계 입력과 결정 구조", "",
        "이 시스템은 \"LLM 이 설계의 진실을 정하지 않는다\" 는 원칙으로 동작합니다. 회로에 들어간 숫자는 세 종류뿐입니다: "
        "사용자 요구사항, 템플릿의 설계 선택값(사용자가 표를 보고 확인한 값), 등록된 계산기의 출력(계산기 id 와 입력을 기록; CALCULATION 단계가 같은 계산기로 다시 계산해 대조). "
        "아래 표의 '식·설명' 열이 계산기의 식이거나 선택값의 이유입니다.", "",
    ]
    if ir.parameters:
        out.append(_table(["키", "값", "출처", "계산기 / 근거", "식·설명"], _parameter_rows(ir)))
    else:
        out.append(f"설계 파라미터 {NO_RECORD}.")
    out.append("")
    template = template_for(ir)
    sections: list[TheorySection]
    if template is not None:
        sections = template.theory(ir)
    else:
        sections = [TheorySection("회로 이론", f"{NO_TEMPLATE}: 이 설계는 템플릿이 만들지 않았으므로 템플릿의 이론 설명이 없습니다. 파라미터 표의 식과 아래 시뮬레이션 설정이 IR 이 담고 있는 전부입니다.")]
    for sec in sections:
        out += [f"## {sec.title}", "", sec.body, ""]
    out += ["## 제약 조건", ""]
    if ir.constraints:
        rows = [[f"`{c.id}`", c.kind.value, c.target, c.description, ", ".join(f"{k} = {_traced_text(v)}" for k, v in c.parameters.items()) or "-"] for c in ir.constraints]
        out.append(_table(["id", "종류", "대상", "내용", "파라미터"], rows))
    else:
        out.append(f"제약 조건 {NO_RECORD}.")
    out.append("")
    out += _simulation_section(ir)
    return "\n".join(out).rstrip("\n") + "\n"


# --- 2. parts report -----------------------------------------------------------------


def _library_facts(component: Component, library: KicadLibrary | None) -> list[str]:
    """The symbol's ``Description`` / ``Datasheet`` / ``ki_keywords`` as the library file states them, labelled as such; or why there are none."""
    out = ["### KiCad 라이브러리 기재", ""]
    if component.symbol is None:
        out += [f"심볼 참조 {NO_RECORD}: 라이브러리 기재 없음.", ""]
        return out
    lib_id = _lib_id(component, "symbol")
    if library is None:
        out += [f"KiCad 라이브러리를 열 수 없어 `{lib_id}` 의 기재를 읽지 못했습니다.", ""]
        return out
    try:
        sym = library.load_symbol(component.symbol)
    except LibraryLookupError:
        out += [f"심볼 `{lib_id}` 이 라이브러리에 없습니다: 라이브러리 기재 없음.", ""]
        return out
    except LibraryFormatError as e:
        out += [f"심볼 `{lib_id}` 을 읽을 수 없습니다 ({strip_paths(str(e))}): 라이브러리 기재 없음.", ""]
        return out
    out.append(f"- 심볼 `{lib_id}`" + (f" (`{sym.extends_from}` 에서 파생)" if sym.extends_from else "") + f", 라이브러리 파일 `{component.symbol.library}.kicad_sym`")
    for prop in LIBRARY_PROPERTIES:
        value = sym.properties.get(prop)
        shown = "(속성 없음)" if value is None else ("(비어 있음)" if value.strip() in ("", "~") else _cell(value))
        out.append(f"- {prop} (KiCad 라이브러리 기재): {shown}")
    out.append(f"- 라이브러리 핀: {' '.join(f'{p.number}={p.name}' for p in sym.pins) or NO_RECORD}")
    if component.footprint is not None:
        out.append(f"- 풋프린트 `{_lib_id(component, 'footprint')}`" + (" (라이브러리에서 확인됨)" if component.footprint.verified else " (라이브러리에서 확인되지 않음)"))
    out.append("")
    return out


def _ir_facts(component: Component) -> list[str]:
    out = ["### IR 기재", ""]
    for label, t in (("제조사", component.manufacturer), ("MPN", component.mpn), ("패키지", component.package)):
        out.append(f"- {label}: " + (f"{t.value} [{_provenance_text(t.provenance)}]" if t is not None else NO_RECORD))
    ds = component.datasheet
    if ds is None:
        out.append(f"- 데이터시트: {NO_RECORD}")
    else:
        parts = [ds.title]
        if ds.url:
            parts.append(ds.url)
        if ds.authority:
            parts.append(f"출처 {ds.authority}")
        if ds.content_hash:
            parts.append(f"보관본 해시 {ds.content_hash}")
        out.append("- 데이터시트: " + ", ".join(_cell(x) for x in parts))
    if component.electrical:
        out.append("- 전기 특성: " + "; ".join(f"{k} = {_traced_text(v)} [{_provenance_text(v.provenance)}]" for k, v in component.electrical.items()))
    else:
        out.append(f"- 전기 특성: {NO_RECORD}")
    if component.sourcing:
        for s in component.sourcing:
            fields = [f"공급사 {s.supplier}"]
            if s.supplier_part_number is not None:
                fields.append(f"품번 {s.supplier_part_number.value}")
            if s.stock is not None:
                fields.append(f"재고 {s.stock.value}")
            if s.unit_price is not None:
                fields.append(f"단가 {s.unit_price.value} {s.currency or ''}".rstrip())
            if s.assembly_class is not None:
                fields.append(f"조립 등급 {s.assembly_class.value}")
            out.append("- 소싱: " + ", ".join(_cell(x) for x in fields))
    else:
        out.append(f"- 소싱: {NO_RECORD}")
    if component.spice is not None:
        b = component.spice
        if b.exclude:
            out.append(f"- SPICE: 넷리스트에서 제외 ({_cell(b.exclude_reason) or '이유 없음'})")
        else:
            out.append(f"- SPICE: {b.device.value if b.device else '?'}" + (f" 값 {_traced_text(b.value)}" if b.value is not None else "") + (f", 모델 `{b.model_name}`" if b.model_name else "")
                       + f", 노드 순서 {', '.join(b.pin_order)}")
    out.append("")
    return out


def _existence_lines(ir: CircuitIR, ref: str) -> list[str]:
    out = [f"### 존재 확인 (`{EXISTENCE_PREFIX}{ref}`)", ""]
    r: ValidationResult | None = ir.validation.latest(f"{EXISTENCE_PREFIX}{ref}")
    if r is None:
        out += [f"{NO_RECORD}: COMPONENT_SELECTION 단계가 이 부품을 아직 확인하지 않았습니다.", ""]
        return out
    out.append(f"- 최신 결과: **{r.status}** ({_tool_text(r.tool, r.tool_version)})")
    checks = r.details.get("checks") if isinstance(r.details, dict) else None
    if isinstance(checks, list) and checks:
        out.append("- 세부 확인:")
        for c in checks:
            if isinstance(c, dict):
                out.append(f"  - {c.get('name', '?')}: {c.get('status', '?')}: {_cell(c.get('message', ''))}")
    else:
        out.append(f"- 세부 확인: {NO_RECORD}; 메시지: {_cell(r.message) or '-'}")
    out.append("")
    return out


def _part_note_lines(note: PartNote | None) -> list[str]:
    if note is None:
        return [f"{NO_TEMPLATE_INFO}: 역할·선정 이유·대체 기준은 템플릿이 만든 설계에만 있습니다.", ""]
    out = [f"- 역할: {note.role}", f"- 선정 이유: {note.why}", "", "### 대체 부품이 만족해야 하는 조건", ""]
    out += [f"- {c}" for c in note.criteria] or ["- (조건 없음)"]
    out += ["", f"### {SUBSTITUTES_HEADING}", ""]
    out += [f"- {s}" for s in note.substitutes] or ["- (후보 없음)"]
    out.append("")
    return out


def parts_report(ir: CircuitIR, library: KicadLibrary | None = None) -> str:
    """Report 2: every part - the BOM-like table, then role / why / criteria / unverified substitutes, library facts, IR facts, existence check, requirements served."""
    out = _header("부품 선정 보고서", ir)
    out += ["## 부품 목록", ""]
    if not ir.components:
        out += [f"부품 {NO_RECORD}: 설계가 비어 있습니다.", ""]
        return "\n".join(out).rstrip("\n") + "\n"
    rows = [[f"`{c.ref}`", c.value, c.description or "-", f"`{_lib_id(c, 'symbol')}`", f"`{_lib_id(c, 'footprint')}`", c.package.value if c.package is not None else NO_RECORD, _pins_text(c)] for c in ir.components]
    out.append(_table(["ref", "값", "설명", "심볼", "풋프린트", "패키지", "핀"], rows))
    out += ["", f"'{SUBSTITUTES_HEADING}' 아래의 부품 이름은 제안일 뿐이며 파이프라인은 그 부품의 핀 배열·정격을 확인하지 않았습니다. "
            "부품의 심볼·풋프린트·핀은 KiCad 라이브러리 파일에서 읽은 것이고, 존재 확인의 세부 항목이 무엇이 확인되었는지를 말합니다.", ""]
    template = template_for(ir)
    notes = template.part_notes(ir) if template is not None else {}
    for c in ir.components:
        out += [f"## {c.ref} — {c.description or c.value}", "", f"- 값: {c.value}", f"- 출처: {_provenance_text(c.provenance)}" + (f" — {strip_paths(c.provenance.note)}" if c.provenance.note else ""), ""]
        out += _part_note_lines(notes.get(c.ref))
        out += _library_facts(c, library)
        out += _ir_facts(c)
        out += _existence_lines(ir, c.ref)
        out += ["### 담당 요구사항", ""]
        out += [f"- {_requirement_text(ir, rid)}" for rid in c.serves_requirements] or ["- (담당 요구사항 없음)"]
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# --- 3. circuit report ---------------------------------------------------------------

#: what the circuit report says for a board without IR copper
NO_ROUTING = "배선 없음"
#: what the final report prints for an expectation without a measured value
NO_MEASUREMENT = "측정 없음"
#: IPC-2221 external-layer current capacity  I = k · ΔT^0.44 · A^0.725  (I in A, ΔT in °C, A in mil²)
IPC2221_K_OUTER = 0.048
IPC2221_DT_EXP = 0.44
IPC2221_AREA_EXP = 0.725
#: the temperature rise the report evaluates the capacity at
IPC2221_DELTA_T_C = 10.0
#: 1 oz copper
COPPER_THICKNESS_UM = 35.0
COPPER_RESISTIVITY_OHM_M = 1.68e-8
MIL_MM = 0.0254
#: IPC-2221 table 6-1, B2 (external, uncoated, sea level): minimum conductor spacing for 0–15 V
IPC2221_B2_SPACING_MM = 0.1
IPC2221_B2_MAX_V = 15.0
#: the order the verification matrix groups the latest results in
MATRIX_ORDER: tuple[ValidationStatus, ...] = (
    ValidationStatus.PASS, ValidationStatus.FAIL, ValidationStatus.NOT_VERIFIED, ValidationStatus.NOT_APPLICABLE,
    ValidationStatus.UNRESOLVED, ValidationStatus.USER_INPUT_REQUIRED,
)
STATUS_WORDS: dict[ValidationStatus, str] = {
    ValidationStatus.PASS: "통과 (도구 증거 있음)",
    ValidationStatus.FAIL: "실패",
    ValidationStatus.NOT_VERIFIED: "검증되지 않음 (증거 없음)",
    ValidationStatus.NOT_APPLICABLE: "해당 없음",
    ValidationStatus.UNRESOLVED: "미해결",
    ValidationStatus.USER_INPUT_REQUIRED: "사용자 입력 필요",
}
#: the stage messages the circuit report quotes from the run record
CIRCUIT_STAGES: tuple[Stage, ...] = (Stage.PLACEMENT, Stage.SCHEMATIC, Stage.PCB, Stage.DRC)

#: a run record as the builders accept it: the saved ``pipeline.json`` or the live state of the run that is writing the report
RunRecord = PipelineRecord | PipelineState | None


def ipc2221_current_a(width_mm: float, delta_t_c: float = IPC2221_DELTA_T_C, thickness_um: float = COPPER_THICKNESS_UM) -> float:
    """IPC-2221 external-layer current capacity in A for a ``width_mm`` wide, ``thickness_um`` thick track at a rise of ``delta_t_c`` (a display value)."""
    area_mil2 = (width_mm / MIL_MM) * (thickness_um / 1000.0 / MIL_MM)
    return IPC2221_K_OUTER * delta_t_c ** IPC2221_DT_EXP * area_mil2 ** IPC2221_AREA_EXP


def copper_resistance_ohm(length_mm: float, width_mm: float, thickness_um: float = COPPER_THICKNESS_UM) -> float:
    """DC resistance R = ρ·L/(w·t) of a copper track (a display value)."""
    return COPPER_RESISTIVITY_OHM_M * (length_mm / 1000.0) / ((width_mm / 1000.0) * (thickness_um * 1e-6))


def routing_params_of(ir: CircuitIR) -> dict[str, str] | None:
    """The router's parameters as the first track's (or via's) provenance records them (``params:grid=0.25,width=0.4,...``); ``None`` without copper."""
    if ir.pcb is None:
        return None
    for item in [*ir.pcb.tracks, *ir.pcb.vias]:
        for entry in item.provenance.derived_from:
            if entry.startswith("params:"):
                out: dict[str, str] = {}
                for kv in entry[len("params:"):].split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        out[k.strip()] = v.strip()
                return out
    return None


def net_routing_stats(ir: CircuitIR) -> list[dict[str, Any]]:
    """Per net (IR order, then nets only the copper names): segment count, length in mm, layers, via count, track widths - computed from ``ir.pcb.tracks`` / ``vias``."""
    if ir.pcb is None:
        return []
    order = [n.name for n in ir.nets]
    for item in [*ir.pcb.tracks, *ir.pcb.vias]:
        if item.net not in order:
            order.append(item.net)
    stats = {name: {"net": name, "segments": 0, "length_mm": 0.0, "layers": set(), "vias": 0, "widths": set()} for name in order}
    for t in ir.pcb.tracks:
        s = stats[t.net]
        s["segments"] += 1
        s["length_mm"] += math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
        s["layers"].add(t.layer)
        s["widths"].add(t.width_mm)
    for v in ir.pcb.vias:
        stats[v.net]["vias"] += 1
    out = []
    for name in order:
        s = stats[name]
        out.append({**s, "layers": sorted(s["layers"]), "widths": sorted(s["widths"])})
    return out


def _outcomes(record: RunRecord) -> list[StageOutcome] | None:
    """The recorded stage outcomes of a run (a saved record or the live state), ``None`` without one."""
    if record is None:
        return None
    state = record.state if isinstance(record, PipelineRecord) else record
    return list(state.outcomes)


def _outcome(record: RunRecord, stage: Stage) -> StageOutcome | None:
    for o in _outcomes(record) or []:
        if o.stage == stage:
            return o
    return None


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _mm(value: float | None, digits: int = 3) -> str:
    return NO_RECORD if value is None else f"{value:.{digits}f} mm"


def _largest_dc_current(ir: CircuitIR) -> tuple[str, float] | None:
    """``(formula, amperes)`` of the largest steady current the template's theory states, from ``ir.parameters``; ``None`` when no template states one."""
    found = template_of(ir)
    if found is None:
        return None
    p = {k: parameter_value(ir, k) for k in ("v_in", "r_c", "r1", "r2", "i_led")}
    if found[0] == "astable" and p["v_in"] is not None and p["r_c"]:
        return "I_C(sat) ≈ V_cc/R_c", p["v_in"] / p["r_c"]
    if found[0] == "divider" and p["v_in"] is not None and p["r1"] is not None and p["r2"] is not None and (p["r1"] + p["r2"]):
        return "I = V_in/(R1+R2)", p["v_in"] / (p["r1"] + p["r2"])
    if found[0] == "led" and p["i_led"] is not None:
        return "I_LED = (V_in − V_f)/R", p["i_led"]
    return None


def _schematic_section(ir: CircuitIR) -> list[str]:
    out = ["## 회로도", ""]
    t = ir.topology
    if t is None:
        out += [f"토폴로지 {NO_RECORD}.", ""]
    else:
        out += [f"- 토폴로지: {t.name} ({', '.join(d.value for d in t.domains) or '도메인 없음'}) [{_provenance_text(t.provenance)}]",
                f"- 설계 근거: {strip_paths(t.rationale) or NO_RECORD}"]
        for b in t.blocks:
            out.append(f"- 블록 `{b.id}`: {b.function} ({b.domain.value}); 부품 {', '.join(b.component_refs) or '-'}; 입력 넷 {', '.join(b.input_nets) or '-'}; 출력 넷 {', '.join(b.output_nets) or '-'}")
        out.append("")
    out += ["### 넷과 그 역할", "",
            "회로도의 연결은 아래 넷 목록이 전부입니다(회로도 파일은 이 IR 에서 컴파일한 파생물). '역할' 열은 넷을 만든 템플릿·도구가 기록한 provenance 메모입니다.", ""]
    if not ir.nets:
        out += [f"넷 {NO_RECORD}.", ""]
        return out
    rows = []
    for n in ir.nets:
        pins = ", ".join(f"{p.component_ref}.{p.pin_number}" for p in n.pins) or "-"
        rows.append([f"`{n.name}`", n.kind.value, pins, n.provenance.note or NO_RECORD, ", ".join(n.serves_requirements) or "-"])
    out += [_table(["넷", "종류", "핀", "역할", "담당 요구사항"], rows), ""]
    return out


def _placement_section(ir: CircuitIR) -> list[str]:
    out = ["## 배치", ""]
    pcb = ir.pcb
    if pcb is None or not pcb.placements:
        out += [f"배치 {NO_RECORD}: IR 에 부품 위치가 없습니다.", ""]
        return out
    first = pcb.placements[0].provenance
    tools = sorted({(p.provenance.tool or "", p.provenance.tool_version or "") for p in pcb.placements})
    params = sorted({e for p in pcb.placements for e in p.provenance.derived_from if not e.startswith("footprint:")})
    if first.tool:
        out.append("- 배치 도구: " + ", ".join(_tool_text(t, v) for t, v in tools if t) + (f" — {first.note}" if first.note else ""))
    else:
        out.append(f"- 배치 도구 {NO_RECORD}: 위치의 출처는 {_provenance_text(first)}" + (f" ({first.note})" if first.note else ""))
    if params:
        out.append("- 배치 파라미터 (provenance `derived_from`): " + ", ".join(f"`{e.split(':', 1)[0]}` = {e.split(':', 1)[1]}" if ":" in e else f"`{e}`" for e in params))
    o = pcb.outline
    if o is not None:
        out.append(f"- 보드 외곽: {o.width_mm:g} × {o.height_mm:g} mm, 원점 ({o.origin_x_mm:g}, {o.origin_y_mm:g}) mm, 층 {', '.join(l.name for l in pcb.layers)}")
    else:
        out.append(f"- 보드 외곽 {NO_RECORD}")
    out.append("")
    rows = []
    for p in pcb.placements:
        c = ir.component(p.component_ref)
        rows.append([f"`{p.component_ref}`", f"`{_lib_id(c, 'footprint')}`" if c is not None else NO_RECORD, f"{p.x_mm:g}", f"{p.y_mm:g}", f"{p.rotation_deg:g}", p.side.value,
                     _provenance_text(p.provenance)])
    out += [_table(["ref", "풋프린트", "x (mm)", "y (mm)", "회전 (°)", "면", "출처"], rows), ""]
    if any(t == PLACER_ID for t, _v in tools):
        out += [
            "### 배치 규칙과 그 한계", "",
            f"`{PLACER_ID}` 는 각 부품의 extent(라이브러리 풋프린트의 courtyard ∪ 패드 경계상자)를 읽어 행 우선 격자에 놓습니다: "
            "피치 = 가장 큰 extent + `spacing_mm`, 외곽 = 격자 경계상자 + `margin_mm`, 부품 순서는 참조 지정자의 자연 순서(접두 문자 순, 그다음 번호 순: C1, C2, …, J1, …, Q1, …, R1, R2, …, R10)입니다. "
            "배치 뒤 모든 extent 쌍이 서로 떨어져 있고 외곽 안에 있음을 재측정해 확인하며, 아니면 거부합니다.", "",
            "정직한 한계: 이 배치기는 신호 흐름·결합 길이·열·EMI 를 전혀 고려하지 않는 placeholder 입니다. 부품이 참조 순서로 놓이므로 서로 이어지는 부품이 멀리 떨어질 수 있고, "
            "그것이 배선이 길어지는 원인입니다. 배치의 유효성(courtyard 겹침, 외곽 위반)은 kicad-cli DRC 만이 판정합니다.", "",
        ]
    return out


def _routing_section(ir: CircuitIR) -> list[str]:
    out = ["## 배선", ""]
    pcb = ir.pcb
    if pcb is None or (not pcb.tracks and not pcb.vias):
        # no verdict here: whether a board without copper passes DRC is kicad-cli's to say, and its
        # recorded result (or its absence) is copied into the stage-record section below
        if pcb is None:
            out += [f"{NO_ROUTING}: IR 에 보드가 없어 트랙·비아도 없습니다.", ""]
        elif not pcb.placements:
            out += [f"{NO_ROUTING}: IR 에 트랙·비아가 없습니다. 보드에는 부품 위치도 구리도 없습니다.", ""]
        else:
            out += [f"{NO_ROUTING}: IR 에 트랙·비아가 없습니다. 보드는 배치만 된 상태입니다(구리 없음). "
                    "이 보드가 DRC 를 통과하는지는 이 보고서가 판정하지 않으며, kicad-cli 가 실제로 낸 결과만 아래 단계 기록의 `drc` 항목에 옮깁니다.", ""]
        out += _routing_results(ir)
        return out
    first = pcb.tracks[0].provenance if pcb.tracks else pcb.vias[0].provenance
    params = routing_params_of(ir) or {}
    g, w, c, e = (_float(params.get(k)) for k in ("grid", "width", "clearance", "edge"))
    via = params.get("via", "")
    d_v, drill = (_float(x) for x in via.split("/", 1)) if "/" in via else (None, None)
    if first.tool:
        out.append(f"- 라우터: {_tool_text(first.tool, first.tool_version)}" + (f" — {first.note}" if first.note else ""))
    else:
        out.append(f"- 라우터 {NO_RECORD}: 구리의 출처는 {_provenance_text(first)}" + (f" ({first.note})" if first.note else ""))
    if params:
        out.append("- 배선 파라미터 (첫 트랙의 provenance `params:` 항목): " + ", ".join(f"`{k}` = {v}" for k, v in params.items()))
    else:
        out.append(f"- 배선 파라미터 {NO_RECORD} (구리의 provenance 에 `params:` 항목이 없음)")
    out.append("")
    stats = net_routing_stats(ir)
    rows = []
    for s in stats:
        rows.append([f"`{s['net']}`", s["segments"], f"{s['length_mm']:.3f}", ", ".join(s["layers"]) or "-", s["vias"], ", ".join(f"{x:g}" for x in s["widths"]) or "-"])
    total_len = sum(s["length_mm"] for s in stats)
    rows.append(["**합계**", len(pcb.tracks), f"{total_len:.3f}", ", ".join(sorted({l for s in stats for l in s['layers']})) or "-", len(pcb.vias), "-"])
    out += ["### 넷별 배선 통계 (IR 의 트랙·비아에서 계산)", "", _table(["넷", "세그먼트", "길이 (mm)", "층", "비아", "폭 (mm)"], rows), ""]
    unrouted = [s["net"] for s in stats if s["segments"] == 0 and s["vias"] == 0]
    if unrouted:
        out += [f"구리가 없는 넷: {', '.join(f'`{n}`' for n in unrouted)} (핀이 하나뿐인 넷은 이을 것이 없고, 그 외는 라우터가 잇지 못한 넷입니다 — `pcb.routing.connectivity` 가 말합니다).", ""]
    if None not in (g, w, c, e, d_v):
        out += [
            "### 라우터의 keep-out 규칙 (파라미터 값을 넣은 것)", "",
            "격자 미로 라우터는 F.Cu / B.Cu 두 층의 정사각 격자 위에서 넷을 (패드 수, 이름) 순으로 하나씩 A* 탐색으로 잇습니다 "
            "(한 칸 1, 방향 전환 `bend_cost`, 층 전환(비아) `via_cost`; 휴리스틱은 맨해튼 거리). 다음 반경 안의 셀은 다른 넷이 쓰지 못합니다 (중심선 기준 거리):", "",
            f"    다른 넷 패드 상자로부터        r = c + w/2 + g/2 = {c:g} + {w / 2:g} + {g / 2:g} = {c + w / 2 + g / 2:.3f} mm",
            f"    이미 놓인 다른 넷 트랙으로부터  r = w + c + g/2 = {w + c + g / 2:.3f} mm   (두 중심선 거리 ≥ w/2 + c + w/2)",
            f"    비아 주변 (양 층)              r = d_v/2 + c + w/2 = {d_v / 2 + c + w / 2:.3f} mm",
            f"    보드 가장자리                  e + w/2 = {e + w / 2:.3f} mm   (비아는 e + d_v/2 = {e + d_v / 2:.3f} mm)", "",
            "g/2 항은 격자 셀 사이를 지나는 구간까지 보수적으로 덮는 여유입니다. 비아는 어떤 패드 안에도 놓지 않고, 한 넷이라도 잇지 못하면 그 넷의 구리는 통째로 버립니다(반쯤 배선된 넷 없음). "
            "이 규칙은 라우터 자체의 파라미터이지 설계 규칙이 아니며, 보드의 유효성은 kicad-cli DRC 가 판정합니다.", "",
        ]
    out += ["### 구리 이론값 (참고 공식; IR 에 기록되지 않는 표시값)", ""]
    widths = sorted({t.width_mm for t in pcb.tracks})
    current = _largest_dc_current(ir)
    for width in widths:
        cap = ipc2221_current_a(width)
        area = (width / MIL_MM) * (COPPER_THICKNESS_UM / 1000.0 / MIL_MM)
        out += [
            f"- IPC-2221 외층 전류 용량 (폭 {width:g} mm, 구리 {COPPER_THICKNESS_UM:g} µm = 1 oz, ΔT = {IPC2221_DELTA_T_C:g} °C):", "",
            f"      I = k · ΔT^{IPC2221_DT_EXP} · A^{IPC2221_AREA_EXP},  k = {IPC2221_K_OUTER} (외층),  A = {width:g} mm × {COPPER_THICKNESS_UM:g} µm = {area:.3f} mil²",
            f"      I = {IPC2221_K_OUTER} × {IPC2221_DELTA_T_C:g}^{IPC2221_DT_EXP} × {area:.3f}^{IPC2221_AREA_EXP} = {quantity(cap, 'A', 4)}", "",
        ]
        if current is not None:
            out.append(f"  설계의 최대 정상 전류 {current[0]} = {quantity(current[1], 'A', 4)} 이므로 여유는 {cap / current[1]:.3g} 배입니다." if current[1] > 0
                       else f"  설계의 최대 정상 전류 {current[0]} = {quantity(current[1], 'A', 4)}.")
        else:
            out.append("  이 설계의 템플릿은 정상 전류를 명시하지 않으므로 용량만 적습니다.")
        out.append("")
    longest = max(stats, key=lambda s: s["length_mm"]) if stats else None
    if longest is not None and longest["length_mm"] > 0 and longest["widths"]:
        w_l = min(longest["widths"])
        r = copper_resistance_ohm(longest["length_mm"], w_l)
        out += [
            f"- 가장 긴 넷 `{longest['net']}` ({longest['length_mm']:.3f} mm, 폭 {w_l:g} mm) 의 구리 저항:", "",
            f"      R = ρ·L/(w·t),  ρ_Cu = {COPPER_RESISTIVITY_OHM_M:g} Ω·m,  t = {COPPER_THICKNESS_UM:g} µm",
            f"      R = {COPPER_RESISTIVITY_OHM_M:g} × {longest['length_mm'] / 1000:.6g} / ({w_l / 1000:.6g} × {COPPER_THICKNESS_UM * 1e-6:g}) = {quantity(r, 'ohm', 4)}", "",
        ]
    v_in = parameter_value(ir, "v_in")
    out.append(f"- IPC-2221 표 6-1 B2 (외층, 코팅 없음, 0–{IPC2221_B2_MAX_V:g} V) 최소 도체 간격 {IPC2221_B2_SPACING_MM:g} mm:")
    if c is None:
        out.append(f"  사용한 간격 {NO_RECORD}: 비교할 수 없습니다.")
    elif v_in is None:
        out.append(f"  라우터 간격 {c:g} mm; 공급 전압(`v_in`) {NO_RECORD} 이라 전압 구간을 정할 수 없어 비교하지 않습니다.")
    elif v_in <= IPC2221_B2_MAX_V:
        rel = "≥" if c >= IPC2221_B2_SPACING_MM else "<"
        out.append(f"  공급 {quantity(v_in, 'V')} 는 이 구간에 들고, 라우터 간격 {c:g} mm {rel} {IPC2221_B2_SPACING_MM:g} mm 입니다.")
    else:
        out.append(f"  공급 {quantity(v_in, 'V')} 는 이 구간 밖이라 다른 행이 적용되며 여기서 비교하지 않습니다.")
    out += ["", "이 값들은 참고 공식의 결과일 뿐 IR 에 기록되지 않고 판정도 아닙니다. 간격·폭의 유효성은 fab 한계값(`pcb.manufacturing`)에 대한 DRC 가 정합니다.", ""]
    out += _routing_results(ir)
    return out


def _routing_results(ir: CircuitIR) -> list[str]:
    """The latest ``pcb.routing.connectivity`` / ``pcb.routing.clearance`` results with their per-net rows, copied."""
    out = ["### IR 기하 검사 결과 (`pcb.routing.*`; DRC 아님)", ""]
    for check in (CONNECTIVITY_CHECK, CLEARANCE_CHECK):
        r = ir.validation.latest(check)
        if r is None:
            out.append(f"- `{check}`: {NO_RECORD}")
            continue
        out.append(f"- `{check}`: **{r.status}** ({_tool_text(r.tool, r.tool_version)}) — {_cell(r.message)}")
        nets = r.details.get("nets") if isinstance(r.details, dict) else None
        if isinstance(nets, list) and nets:
            for row in nets:
                if isinstance(row, dict):
                    out.append(f"  - `{row.get('net', '?')}`: {row.get('status', '?')}: {_cell(row.get('message', ''))}")
        if check == CLEARANCE_CHECK and isinstance(r.details, dict):
            lim = r.details.get("limit_mm")
            out.append(f"  - 한계 {NO_RECORD if lim is None else f'{lim:g} mm'}, 비교한 쌍 {r.details.get('pairs_compared', NO_RECORD)}, 위반 {len(r.details.get('violations') or [])}")
    out.append("")
    return out


def _stage_record_section(record: RunRecord, stages: tuple[Stage, ...], title: str) -> list[str]:
    out = [f"## {title}", ""]
    outcomes = _outcomes(record)
    if outcomes is None:
        out += [f"{NO_RUN_RECORD}: 단계 결과는 `ai-eda run` 의 실행 기록(`pipeline.json`)에서 옵니다.", ""]
        return out
    for stage in stages:
        o = _outcome(record, stage)
        if o is None:
            out.append(f"- `{stage}`: 도달하지 않음")
        else:
            out.append(f"- `{stage}`: **{o.status}** — {_cell(o.message) or '(메시지 없음)'}")
    out.append("")
    return out


def circuit_report(ir: CircuitIR, library: KicadLibrary | None, record: RunRecord) -> str:
    """Report 3: the schematic (nets and roles), the placement (rule, positions, limits) and the routing (parameters, per-net statistics, reference copper values, ``pcb.routing.*`` verdicts).

    ``library`` is accepted for symmetry with the other builders and not
    read: every geometry number here is in the IR already.
    """
    out = _header("회로 보고서", ir)
    out += [
        "회로도와 보드는 이 IR 에서 컴파일한 파생물입니다. 아래는 IR 이 담고 있는 연결·위치·구리와, 그것을 만든 도구가 기록한 규칙, "
        "그리고 참고 공식으로 계산한 표시값입니다. ERC / DRC 판정은 kicad-cli 만이 내리며 그 결과는 단계 기록에 그대로 옮깁니다.", "",
    ]
    out += _schematic_section(ir)
    out += _placement_section(ir)
    out += _routing_section(ir)
    out += _stage_record_section(record, CIRCUIT_STAGES, "단계 기록 (PLACEMENT / SCHEMATIC / PCB / DRC)")
    return "\n".join(out).rstrip("\n") + "\n"


# --- 4. final report -----------------------------------------------------------------


def _freshness(r: ValidationResult, design_hash: str) -> str:
    if r.ir_hash is None:
        return "IR 해시 없음"
    return "현재 IR" if r.ir_hash == design_hash else f"이전 IR 버전 ({r.ir_hash[:16]})"


def _requirements_section(ir: CircuitIR) -> list[str]:
    out = ["## 프로젝트 개요", ""]
    raw = ir.requirements.raw_input.strip()
    out += ["요청문:", "", "    " + (strip_paths(raw).replace("\n", "\n    ") if raw else "(비어 있음)"), ""]
    if ir.requirements.corrections:
        out += ["사용자 정정:", ""] + [f"- {_cell(c)}" for c in ir.requirements.corrections] + [""]
    rows = []
    for r in ir.requirements.requirements:
        rows.append([f"`{r.id}`", r.key, r.text, r.kind.value, r.status.value, _traced_text(r.value), r.category, _provenance_text(r.value.provenance) if r.value is not None else "-"])
    out += [_table(["id", "키", "본문", "종류", "상태", "값", "범주", "값의 출처"], rows) if rows else f"요구사항 {NO_RECORD}.", ""]
    return out


def _all_stages_section(record: RunRecord) -> list[str]:
    out = ["## 단계 결과", ""]
    outcomes = _outcomes(record)
    if outcomes is None:
        out += [f"{NO_RUN_RECORD}: `ai-eda run` 이 `pipeline.json` 에 남긴 단계 결과가 없습니다.", ""]
        return out
    state = record.state if isinstance(record, PipelineRecord) else record
    notes = []
    if isinstance(record, PipelineRecord) and record.aborted:
        notes.append(f"실행이 `{record.aborted_stage}` 단계에서 {record.aborted} 로 중단되었습니다.")
    if state.blocked:
        notes.append("실행이 필수 질문에서 멈췄습니다 (`--answer` 로 답한 뒤 다시 실행).")
    if notes:
        out += [f"- {n}" for n in notes] + [""]
    rows = []
    for stage in STAGE_ORDER:
        o = _outcome(record, stage)
        if o is None:
            rows.append([f"`{stage}`", "-", "도달하지 않음", "-"])
        else:
            rows.append([f"`{stage}`", str(o.status), o.message or "-", f"{len(o.questions)}개" if o.questions else "-"])
    out += [_table(["단계", "상태", "메시지", "질문"], rows), ""]
    return out


def _theory_vs_simulation(ir: CircuitIR) -> list[str]:
    out = ["## 이론값 대 시뮬레이션", ""]
    sim = ir.simulation
    if sim is None or not sim.expectations:
        out += [f"기대값 {NO_RECORD}: 비교할 시뮬레이션 설정이 없습니다.", ""]
        return out
    summary = ir.validation.latest(SPICE_CHECK)
    out.append(f"- `{SPICE_CHECK}` 최신 결과: " + (f"**{summary.status}** ({_tool_text(summary.tool, summary.tool_version)}) — {_cell(summary.message)}" if summary is not None else NO_RECORD))
    out += ["- 이론 공칭값은 IR 의 기대값(템플릿의 식으로 계산기가 낸 값 또는 사용자 값), 측정값은 ngspice 결과(`spice.<id>` 의 `details.measured`)입니다. "
            "허용치 = max(tol_abs, tol_rel·|공칭값|); 공칭값이 0 이면 tol_rel 은 허용치가 아니므로 tol_abs 만 셉니다(SPICE 단계의 판정 규칙과 같음). 판정은 저장된 결과의 상태 그대로입니다.", ""]
    rows = []
    extra: list[str] = []
    for e in sim.expectations:
        r = ir.validation.latest(f"{SPICE_CHECK}.{e.id}")
        unit = e.nominal.unit
        nominal = float(e.nominal.value) if isinstance(e.nominal.value, (int, float)) and not isinstance(e.nominal.value, bool) else None
        details = r.details if r is not None and isinstance(r.details, dict) else {}
        measured = details.get("measured")
        measured = float(measured) if isinstance(measured, (int, float)) and not isinstance(measured, bool) else None
        tol_abs = float(e.tol_abs.value) if e.tol_abs is not None else None
        tol_rel = float(e.tol_rel.value) if e.tol_rel is not None else None
        limit = None
        if nominal is not None:
            # the SPICE stage's rule (ai_eda.tools.spice.stage.judge): a relative tolerance on a nominal of 0 is no tolerance
            relative = tol_rel * abs(nominal) if tol_rel is not None and nominal != 0.0 else None
            candidates = [x for x in (tol_abs, relative) if x is not None]
            limit = max(candidates) if candidates else None
        if measured is not None and nominal is not None:
            dev = measured - nominal
            dev_text = quantity(dev, unit)
            pct = f"{dev / abs(nominal) * 100:+.3g} %" if nominal != 0 else "-"
        else:
            dev_text, pct = "-", "-"
        rows.append([
            f"`{e.id}`", f"`{e.vector}` / {REDUCE_LABELS.get(e.reduce, str(e.reduce))}" + (f" at {_traced_text(e.at)}" if e.at is not None else ""),
            _traced_text(e.nominal), quantity(measured, unit) if measured is not None else NO_MEASUREMENT, dev_text, pct,
            quantity(limit, unit) if limit is not None else "-", r.status if r is not None else NO_RECORD,
        ])
        if r is not None and r.status is not ValidationStatus.PASS:
            extra.append(f"- `{e.id}`: {r.status} — {_cell(r.message) or '(메시지 없음)'}")
        freq = details.get("frequency")
        if isinstance(freq, dict):
            edges = freq.get("edges")
            t1, tn = freq.get("first_edge_s"), freq.get("last_edge_s")
            extra.append(
                f"- `{e.id}` 주파수 측정: 상승 에지 {edges}개, 첫 에지 {quantity(t1, 's') if isinstance(t1, (int, float)) else NO_RECORD}, 마지막 에지 {quantity(tn, 's') if isinstance(tn, (int, float)) else NO_RECORD}"
                + (f", f = ({edges} − 1)/({tn:.6g} − {t1:.6g}) s" if isinstance(edges, int) and isinstance(t1, (int, float)) and isinstance(tn, (int, float)) else "")
                + f"; 파형 최소 {quantity(freq.get('vmin'), 'V') if isinstance(freq.get('vmin'), (int, float)) else NO_RECORD}, 최대 {quantity(freq.get('vmax'), 'V') if isinstance(freq.get('vmax'), (int, float)) else NO_RECORD}"
            )
    out += [_table(["기대값", "벡터 / 축약", "이론 공칭값", "시뮬레이션 측정값", "편차", "편차 (%)", "허용치", "판정"], rows), ""]
    if extra:
        out += extra + [""]
    return out


def _matrix_section(ir: CircuitIR) -> list[str]:
    out = ["## 검증 매트릭스 (검사별 최신 결과)", ""]
    latest = ir.validation.latest_by_check()
    if not latest:
        out += [f"검증 결과 {NO_RECORD}.", ""]
        return out
    design_hash = ir.content_hash()
    out += ["'대상 IR' 은 결과에 찍힌 IR 해시가 지금의 설계 해시와 같은지의 사실이며, 상태는 저장된 값 그대로입니다. 도구 없는 PASS 는 의견이지 증거가 아닙니다.", ""]
    for status in MATRIX_ORDER:
        ids = sorted(k for k, r in latest.items() if r.status is status)
        if not ids:
            continue
        rows = [[f"`{k}`", _tool_text(latest[k].tool, latest[k].tool_version), _freshness(latest[k], design_hash), latest[k].message or "-"] for k in ids]
        out += [f"### {status} — {STATUS_WORDS.get(status, str(status))} ({len(ids)}개)", "", _table(["검사", "도구", "대상 IR", "메시지"], rows), ""]
    return out


def _artifacts_section(ir: CircuitIR) -> list[str]:
    out = ["## 산출물", ""]
    if not ir.artifacts:
        out += [f"등록된 산출물 {NO_RECORD}.", ""]
        return out
    design_hash = ir.content_hash()
    rows = []
    for kind, a in ir.artifacts.items():
        fresh = "현재 IR 에서 생성" if a.generated_from_ir_hash == design_hash else ("생성 IR 해시 없음" if a.generated_from_ir_hash is None else f"오래됨 (IR {a.generated_from_ir_hash[:16]})")
        rows.append([f"`{kind.value}`", Path(a.path).name, f"{len(a.files)}개" if a.files else "-", _tool_text(a.generator, a.generator_version) if a.generator else "-",
                     a.content_hash or NO_RECORD, fresh, "; ".join(a.notes) or "-"])
    out += ["산출물은 IR 에서 컴파일한 파생물이며 파일 이름만 적습니다(위치는 적지 않음). '신선도' 는 산출물의 `generated_from_ir_hash` 와 지금의 설계 해시를 비교한 사실입니다.", "",
            _table(["종류", "파일", "구성 파일", "생성기", "내용 해시", "신선도", "비고"], rows), ""]
    return out


def _release_section(record: RunRecord) -> list[str]:
    out = ["## RELEASE 상태", ""]
    o = _outcome(record, Stage.RELEASE)
    if _outcomes(record) is None:
        out += [f"{NO_RUN_RECORD}: RELEASE 판정은 `ai-eda run` 의 기록된 결과만 옮깁니다.", ""]
    elif o is None:
        out += ["기록된 실행이 RELEASE 단계에 도달하지 않았습니다.", ""]
    else:
        out += [f"- 기록된 상태: **{o.status}**", f"- 기록된 메시지: {_cell(o.message) or '(메시지 없음)'}", ""]
    return out


def _remaining_section(ir: CircuitIR) -> list[str]:
    out = ["## 남은 일", ""]
    latest = ir.validation.latest_by_check()
    groups = (
        (ValidationStatus.FAIL, "실패한 검사 (설계 변경은 사람이 결정)"),
        (ValidationStatus.USER_INPUT_REQUIRED, "사용자 답이 필요한 검사"),
        (ValidationStatus.UNRESOLVED, "미해결 검사"),
        (ValidationStatus.NOT_VERIFIED, "증거가 없어 검증되지 않은 검사 (이유)"),
    )
    any_row = False
    for status, title in groups:
        ids = sorted(k for k, r in latest.items() if r.status is status)
        if not ids:
            continue
        any_row = True
        out += [f"### {title}", ""] + [f"- `{k}`: {_cell(latest[k].message) or '(이유 없음)'}" for k in ids] + [""]
    if not any_row:
        out += ["최신 결과 중 FAIL / NOT_VERIFIED / UNRESOLVED / USER_INPUT_REQUIRED 인 검사가 없습니다.", ""]
    return out


def _conclusion_section(ir: CircuitIR, record: RunRecord) -> list[str]:
    latest = ir.validation.latest_by_check()
    counts = {s: sum(1 for r in latest.values() if r.status is s) for s in MATRIX_ORDER}
    tools = sorted({r.tool for r in latest.values() if r.status is ValidationStatus.PASS and r.tool})
    opinions = sorted(k for k, r in latest.items() if r.status is ValidationStatus.PASS and not r.tool)
    release = _outcome(record, Stage.RELEASE)
    parts = [f"검사 {len(latest)}개의 최신 결과는 " + ", ".join(f"{s} {n}개" for s, n in counts.items() if n) + " 입니다."]
    if tools:
        parts.append("PASS 로 기록된 검사는 " + ", ".join(f"`{t}`" for t in tools) + " 가 판정했습니다.")
    if opinions:
        parts.append("도구 없이 PASS 로 기록된 검사 (" + ", ".join(f"`{k}`" for k in opinions) + ") 는 의견이며 증거로 세지 않습니다.")
    if counts[ValidationStatus.NOT_VERIFIED]:
        parts.append(f"NOT_VERIFIED {counts[ValidationStatus.NOT_VERIFIED]}개는 증거가 없는 것이지 문제가 없다는 뜻이 아닙니다; 이유는 '남은 일' 에 있습니다.")
    if counts[ValidationStatus.FAIL]:
        parts.append(f"FAIL {counts[ValidationStatus.FAIL]}개는 사람이 설계를 바꿔야 합니다.")
    if release is None:
        parts.append(f"RELEASE 판정은 {NO_RUN_RECORD if _outcomes(record) is None else '기록된 실행이 그 단계에 이르지 않아'} 없습니다.")
    else:
        parts.append(f"기록된 RELEASE 상태는 {release.status} 입니다.")
    parts.append("이 보고서는 저장된 상태 이상을 주장하지 않습니다.")
    return ["## 결론", "", " ".join(parts), ""]


def final_report(ir: CircuitIR, record: RunRecord) -> str:
    """Report 4: requirements, every stage outcome, theory vs simulation per expectation, the verification matrix, artifacts, the recorded RELEASE outcome, remaining work, conclusion."""
    out = _header("최종 보고서", ir)
    out += _requirements_section(ir)
    out += _all_stages_section(record)
    out += _theory_vs_simulation(ir)
    out += _matrix_section(ir)
    out += _artifacts_section(ir)
    out += _release_section(record)
    out += _remaining_section(ir)
    out += _conclusion_section(ir, record)
    return "\n".join(out).rstrip("\n") + "\n"


# --- writers -------------------------------------------------------------------------


def build_stage_report(stage: Stage, ir: CircuitIR, library: KicadLibrary | None, record: RunRecord) -> str | None:
    """The Markdown of the report ``stage`` completes (:data:`STAGE_REPORTS`), ``None`` for a stage without one."""
    if stage is Stage.ARCHITECTURE:
        return theory_report(ir, library)
    if stage is Stage.COMPONENT_SELECTION:
        return parts_report(ir, library)
    if stage is Stage.PCB:
        return circuit_report(ir, library, record)
    if stage is Stage.RELEASE:
        return final_report(ir, record)
    return None


def write_stage_report(stage: Stage, ir: CircuitIR, library: KicadLibrary | None, record: RunRecord, workdir: Path, *, reports_dir: Path | None = None) -> Path | None:
    """Write the report of ``stage`` to ``<workdir>/reports/<name>`` (LF, UTF-8; ``reports_dir`` overrides the folder) and return its path; ``None`` for a stage without a report.

    Nothing else is touched: the IR is not saved, no artifact is registered.
    """
    text = build_stage_report(stage, ir, library, record)
    if text is None:
        return None
    folder = Path(reports_dir) if reports_dir is not None else Path(workdir) / REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / STAGE_REPORTS[stage]
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_all_stage_reports(ir: CircuitIR, library: KicadLibrary | None, record: RunRecord, workdir: Path, *, reports_dir: Path | None = None) -> list[Path]:
    """Write the four reports from a saved IR and its run record (``ai-eda stage-reports``); returns the paths in stage order."""
    out: list[Path] = []
    for stage in STAGE_REPORTS:
        path = write_stage_report(stage, ir, library, record, workdir, reports_dir=reports_dir)
        if path is not None:
            out.append(path)
    return out


__all__ = [
    "CIRCUIT_STAGES",
    "COPPER_RESISTIVITY_OHM_M",
    "COPPER_THICKNESS_UM",
    "IPC2221_B2_MAX_V",
    "IPC2221_B2_SPACING_MM",
    "IPC2221_DELTA_T_C",
    "IPC2221_K_OUTER",
    "LIBRARY_PROPERTIES",
    "MATRIX_ORDER",
    "NO_MEASUREMENT",
    "NO_ROUTING",
    "NO_RUN_RECORD",
    "NO_TEMPLATE",
    "NO_TEMPLATE_INFO",
    "REPORTS_DIR",
    "STAGE_REPORTS",
    "SUBSTITUTES_HEADING",
    "RunRecord",
    "build_stage_report",
    "circuit_report",
    "copper_resistance_ohm",
    "final_report",
    "ipc2221_current_a",
    "net_routing_stats",
    "parts_report",
    "routing_params_of",
    "strip_paths",
    "template_for",
    "template_of",
    "theory_report",
    "write_all_stage_reports",
    "write_stage_report",
]
