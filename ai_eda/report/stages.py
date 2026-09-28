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

Each report is delivered three ways from the one Markdown text: ``<name>.md``,
``<name>.html`` (the Markdown rendered by :mod:`ai_eda.report.pdf` with the
report's figures inline as SVG - :mod:`ai_eda.report.figures`: the
template's theory curves and the measured waveform in the theory report,
the placement, the routed board (both with the silkscreen drawn on them),
the per-net copper length and the isometric 3D preview in the circuit
report, the theory-vs-simulation tolerance chart and the waveform in the
final report; the parts report has none and says so) and, when a
headless Chromium / Chrome / Edge is found or given, ``<name>.pdf`` (the HTML
printed; no browser means no PDF and a reason, never a failure). The
Markdown holds a placeholder line ``![fig](fig:<id>)`` followed by the
caption in italics where a figure goes, so it stays readable on its own; a
figure whose data is missing is one Korean sentence saying what is missing
(:class:`ReportFigures`). The waveform is read from the fresh ``SPICE_RESULT``
artifact only through :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`
(the rule every SPICE-reading validator follows), the board from the KiCad
library the pipeline resolved, the 3D preview from the scene the ``MODEL_3D``
artifact is compiled from (:func:`~ai_eda.tools.model3d.scene.build_scene`:
the same library plus the STEP files of the 3D model library it finds). The
circuit report's silkscreen section copies the IR's silk texts, the placer's
recorded parameters and the latest ``pcb.silk.*`` results; it judges
nothing. A design with ``ir.si`` also gets the circuit report's
"임피던스·타이밍" section and the theory report's transmission-line section
(:mod:`ai_eda.report.si_report`) with four more circuit figures
(:mod:`ai_eda.report.si_figures`: the board coloured by net class, Z0 against
width over the plane, the ``si.critical_length`` delay bars, the step
response of each fresh ``spice.si.<net>`` result read from its rawfile) -
the same rule: the ``si.*`` / ``spice.si.*`` statuses copied, a display
value named as one. ``.md`` and ``.html`` are deterministic; the
PDF bytes carry the browser's own creation date and are a derived document
like a rawfile. None of the three is an artifact or hashed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eda.design import TEMPLATES
from ai_eda.design.base import CHOICE_NOTE_PREFIX, NO_RECORD, TOOL_ID, PartNote, Template, TheorySection, parameter_value, quantity
from ai_eda.design.templates import display_spelled
from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Component, Provenance, ProvenanceKind, Reduce, SilkKind, Traced, ValidationResult, ValidationStatus
from ai_eda.parts.existence import CHECK_PREFIX as EXISTENCE_PREFIX
from ai_eda.report.figures import Figure, bar_figure, board_figure, expectation_limit, model3d_figure_from_scene, plot_vector, tolerance_figure, tolerance_rows, waveform_figures
from ai_eda.report.pdf import NO_BROWSER_REASON, find_browser, html_to_pdf, markdown_to_html
from ai_eda.report.pipeline_log import PipelineRecord
from ai_eda.report.si_figures import delay_bar_figure, delay_rows, fresh_si_rawfile, step_response_figure, z0_width_figure
from ai_eda.tools.calc.part_value import PART_VALUE_DIGITS
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.core_ring import BODY_OVERHANG_MM, CORE_MIN_PADS, EDGE_REF_PREFIXES, INNER_MAX_PADS
from ai_eda.tools.placement.core_ring import PLACER_ID as RING_PLACER_ID
from ai_eda.tools.placement.grid import PLACER_ID
from ai_eda.tools.model3d.scene import BODY_CAPTION, Scene, build_scene
from ai_eda.tools.silkscreen.geometry import TEXT_HEIGHT_FACTOR, TEXT_WIDTH_FACTOR
from ai_eda.tools.silkscreen.place import CONNECTOR_LIBRARY_PREFIX, REFERENCE_CANDIDATES, TITLE_SLIDE_STEP_MM
from ai_eda.tools.silkscreen.place import PLACER_ID as SILK_PLACER_ID
from ai_eda.tools.spice.evidence import fresh_spice_run
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK
from ai_eda.validation.layout import CLEARANCE_CHECK, CONNECTIVITY_CHECK, SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK
from ai_eda.workflow.orchestrator import PipelineState, StageOutcome
from ai_eda.workflow.stages import STAGE_ORDER, Stage

#: the report each stage's outcome completes (written right after that stage by ``ai-eda run``)
STAGE_REPORTS: dict[Stage, str] = {
    Stage.ARCHITECTURE: "01_이론_보고서.md",
    Stage.COMPONENT_SELECTION: "02_부품선정_보고서.md",
    Stage.PCB: "03_회로_보고서.md",
    Stage.RELEASE: "04_최종_보고서.md",
}
#: the title of each report (its ``# `` heading and the HTML ``<title>``, followed by ``: <project name>``)
REPORT_TITLES: dict[Stage, str] = {
    Stage.ARCHITECTURE: "이론 보고서",
    Stage.COMPONENT_SELECTION: "부품 선정 보고서",
    Stage.PCB: "회로 보고서",
    Stage.RELEASE: "최종 보고서",
}
#: where the reports live under the project workdir
REPORTS_DIR = "reports"
#: the file suffixes one report is written in (the PDF only when a browser prints it)
REPORT_SUFFIXES: tuple[str, ...] = (".md", ".html", ".pdf")
#: what the parts report says instead of a figure
NO_CHART_IN_PARTS = "이 보고서에는 그림이 없습니다: 부품 선정은 표와 라이브러리 기재로 설명하며 그릴 수치 곡선이 없습니다."
#: the reason recorded when the caller asked for no PDF
PDF_NOT_REQUESTED = "not requested (--no-pdf)"
#: header line for a design no template built
NO_TEMPLATE = "템플릿 설계가 아님"
#: what the parts report prints for a part its template says nothing about
NO_TEMPLATE_INFO = "템플릿 정보 없음"
#: the heading under which substitute names are printed: they are not verified by the pipeline
SUBSTITUTES_HEADING = "대체 후보 (파이프라인이 검증하지 않은 이름)"
#: what a report says where it needs a run record and has none
NO_RUN_RECORD = "실행 기록 없음"
#: under the parts table of a template design whose every numeric part carries the display spelling (``display_spelled``; a v0.1
#: design keeps the netlist spelling and gets no note): the value column is display text, the netlist carries the exact design number
VALUE_COLUMN_NOTE = (
    f"템플릿이 만든 부품의 '값' 은 KiCad 방식의 표기입니다: 단위 없이 SI 접두어(`p` `n` `u` `m` `k` `M` = 메가 `G`), 유효숫자 최대 {PART_VALUE_DIGITS}자리, "
    "끝자리 0 생략 (예: `100n`, `1.5915k`). 시뮬레이션 넷리스트와 계산기는 이 표기가 아니라 IR 의 정확한 설계값(SPICE 바인딩)을 쓰므로, "
    f"두 값은 유효숫자 {PART_VALUE_DIGITS}자리 표기의 반올림만큼 다를 수 있습니다 (E 계열 반올림은 어느 쪽에도 하지 않음)."
)
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
    Reduce.DB_AT: "한 주파수에서의 크기 (dB)",
    Reduce.RMS: "구간 RMS",
    Reduce.DB_RMS: "구간 RMS 비 (dB)",
    Reduce.HARMONIC_DBC: "k차 고조파 레벨 (dBc)",
    Reduce.AM_DEPTH: "AM 변조도",
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


# --- figures ---------------------------------------------------------------------------

#: the figure slots of the reports, in the order the text reaches them
SLOT_THEORY = "theory"
SLOT_WAVEFORM = "waveform"
SLOT_PLACEMENT = "placement"
SLOT_BOARD = "board"
SLOT_COPPER = "copper_bars"
SLOT_TOLERANCE = "tolerance"
SLOT_ISO3D = "iso3d"
#: the missing-data sentences (one per slot; the reason, when there is one, is appended in parentheses)
NO_WAVEFORM = "시뮬레이션 결과가 아직 없어 파형 그림이 없습니다"
NO_PLACEMENT_FIGURE = "배치가 없어 배치도가 없습니다"
NO_BOARD_FIGURE = "배선이 없어 보드 그림이 없습니다"
NO_COPPER_FIGURE = "배선이 없어 넷별 동박 길이 그래프가 없습니다"
NO_TOLERANCE_FIGURE = "기대값이 없어 이론값 대 시뮬레이션 그림이 없습니다"
NO_LIBRARY_FIGURE = "KiCad 라이브러리를 열 수 없어 보드 그림이 없습니다"
NO_ISO3D_FIGURE = "3D 미리보기 그림이 없습니다"


@dataclass
class ReportFigures:
    """The figures of one report, by slot: what was drawn (:class:`~ai_eda.report.figures.Figure` by id) and, per slot without a figure, the Korean sentence saying what is missing.

    A builder prints a slot as the placeholder line ``![fig](fig:<id>)`` plus
    the italic caption for every figure of the slot, then the slot's missing
    sentence when there is one; a slot the report never filled prints nothing.
    """

    figures: dict[str, Figure] = field(default_factory=dict)
    slots: dict[str, list[str]] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    #: the 3D scene the ``iso3d`` figure was drawn from (the circuit report's body table reads it; built once per report)
    scene: Scene | None = None

    def add(self, slot: str, *figures: Figure) -> None:
        for fig in figures:
            if fig.id in self.figures:
                raise ValueError(f"figure id {fig.id!r} used twice in one report")
            self.figures[fig.id] = fig
            self.slots.setdefault(slot, []).append(fig.id)

    def miss(self, slot: str, sentence: str, reason: str | None = None) -> None:
        """Record why ``slot`` has no figure: ``sentence`` (a full Korean sentence without its final period) plus the ``reason`` in parentheses."""
        text = f"{sentence} ({strip_paths(reason)})" if reason else sentence
        self.missing[slot] = text.rstrip(".") + "."

    def lines(self, slot: str) -> list[str]:
        """The Markdown of ``slot``: placeholder + caption per figure, then the missing sentence; a slot nobody filled prints nothing."""
        out: list[str] = []
        for fig_id in self.slots.get(slot, []):
            fig = self.figures[fig_id]
            out += [f"![fig](fig:{fig.id})", "", f"*{_caption_line(fig)}*", ""]
        if slot in self.missing:
            out += [self.missing[slot], ""]
        return out


def _caption_line(fig: Figure) -> str:
    """The caption as one italic Markdown line: ``title — caption``, no ``*``, no line break, no path (the HTML uses the figure's own ``<figcaption>``)."""
    text = f"{fig.title} — {fig.caption}" if fig.caption else fig.title
    text = strip_paths(text).replace("*", "∗").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def theory_figures_of(ir: CircuitIR, figures: ReportFigures) -> None:
    """The template's theory curves into the ``theory`` slot; a non-template design or a template whose numbers the IR lacks gets a sentence."""
    template = template_for(ir)
    if template is None:
        figures.miss(SLOT_THEORY, f"{NO_TEMPLATE}: 템플릿의 이론 곡선이 없습니다")
        return
    try:
        drawn = template.theory_figures(ir)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        figures.miss(SLOT_THEORY, f"템플릿 `{template.id}` 의 이론 그림을 그릴 수 없습니다", str(e))
        return
    if not drawn:
        figures.miss(SLOT_THEORY, f"템플릿 `{template.id}` 의 이론 그림을 그릴 수 없습니다: 필요한 파라미터가 IR 에 없습니다 ({NO_RECORD})")
        return
    figures.add(SLOT_THEORY, *drawn)


def waveform_figures_of(ir: CircuitIR, figures: ReportFigures) -> None:
    """The measured waveforms of the fresh SPICE run into the ``waveform`` slot: per non-op analysis the expectations' vectors plus the template's :meth:`~ai_eda.design.base.Template.theory_figure_vectors`.

    The run is located only through :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`
    (the latest tool-backed ``spice`` result on the netlist of the current
    IR, results.json unchanged on disk); anything else is the reason in the
    missing sentence. Voltages and branch currents become separate figures
    (``waveform`` / ``waveform_i``; a second analysis gets ``waveform_<id>``;
    more than four vectors of one kind are split over several charts). A
    template hint that the recorded run has no vector for is left out and
    named in a sentence after the figure; the expectations' own vectors are
    never optional (one missing is the reason the slot has no figure).
    """
    sim = ir.simulation
    if sim is None or not sim.expectations:
        figures.miss(SLOT_WAVEFORM, f"{NO_WAVEFORM}: 시뮬레이션 설정(기대값)이 없습니다")
        return
    run = fresh_spice_run(ir, needs="waveform figure")
    if isinstance(run, str):
        figures.miss(SLOT_WAVEFORM, NO_WAVEFORM, run)
        return
    template = template_for(ir)
    hints = list(template.theory_figure_vectors()) if template is not None else []
    groups: dict[str, list[str]] = {}
    for e in sim.expectations:
        vectors = groups.setdefault(e.analysis_id, [])
        if e.vector not in vectors:
            vectors.append(e.vector)
    reasons: list[str] = []
    for aid, vectors in groups.items():
        info = run.data.get("analyses", {}).get(aid) or {}
        result = info.get("result") if isinstance(info.get("result"), dict) else {}
        if info.get("kind") == "op":
            reasons.append(f"해석 `{aid}` 은 동작점(op) 해석이라 시간·주파수 축 파형이 없습니다")
            continue
        if not result.get("succeeded") or result.get("unverifiable"):
            reasons.append(f"해석 `{aid}` 이 완료되지 않아 파형이 없습니다")
            continue
        plot = result.get("vectors") if isinstance(result.get("vectors"), dict) else {}
        extra = [h for h in hints if h not in vectors]
        present = [h for h in extra if plot_vector(plot, h) is not None]
        absent = [h for h in extra if plot_vector(plot, h) is None]
        fig_id = SLOT_WAVEFORM if not figures.slots.get(SLOT_WAVEFORM) else f"{SLOT_WAVEFORM}_{aid}"
        try:
            figures.add(SLOT_WAVEFORM, *waveform_figures(run.data, aid, vectors + present, title=f"시뮬레이션 파형 — 해석 {aid}", fig_id=fig_id))
        except (ValueError, TypeError, KeyError) as e:
            reasons.append(f"해석 `{aid}` 의 파형을 그릴 수 없습니다 ({strip_paths(str(e))})")
            continue
        if absent:
            reasons.append(f"해석 `{aid}` 의 기록에 템플릿이 함께 그리려던 벡터 {', '.join(f'`{h}`' for h in absent)} 이(가) 없어 기대값 벡터{'와 나머지 벡터' if present else ''}만 그렸습니다")
    if reasons:
        figures.miss(SLOT_WAVEFORM, "; ".join(reasons) if figures.slots.get(SLOT_WAVEFORM) else f"파형 그림이 없습니다: {'; '.join(reasons)}")


def board_figures_of(ir: CircuitIR, library: KicadLibrary | None, figures: ReportFigures) -> None:
    """The placement figure (no copper), the routed board and the per-net copper length bars into their slots; each missing input is a sentence."""
    pcb = ir.pcb
    if pcb is None or not pcb.placements:
        figures.miss(SLOT_PLACEMENT, f"{NO_PLACEMENT_FIGURE}: IR 에 부품 위치가 없습니다")
        figures.miss(SLOT_BOARD, f"{NO_BOARD_FIGURE}: IR 에 보드가 없습니다")
        figures.miss(SLOT_COPPER, f"{NO_COPPER_FIGURE}: IR 에 보드가 없습니다")
        return
    routed = bool(pcb.tracks or pcb.vias)
    if library is None:
        figures.miss(SLOT_PLACEMENT, NO_LIBRARY_FIGURE)
        figures.miss(SLOT_BOARD, NO_LIBRARY_FIGURE)
    else:
        try:
            figures.add(SLOT_PLACEMENT, board_figure(ir, library, copper=False))
            if routed:
                figures.add(SLOT_BOARD, board_figure(ir, library, copper=True))
        except (ValueError, LibraryLookupError, LibraryFormatError) as e:
            # every slot left without a figure gets the sentence: the placement figure failing (a footprint not in the library)
            # would fail the routed board figure the same way, and a routed board's slot is never filled by the branch below
            for slot in ((SLOT_PLACEMENT, SLOT_BOARD) if routed else (SLOT_PLACEMENT,)):
                if slot not in figures.slots:
                    figures.miss(slot, "보드 그림을 그릴 수 없습니다", str(e))
    if not routed:
        figures.miss(SLOT_BOARD, f"{NO_BOARD_FIGURE}: IR 에 트랙·비아가 없습니다")
        figures.miss(SLOT_COPPER, f"{NO_COPPER_FIGURE}: IR 에 트랙·비아가 없습니다")
        return
    stats = net_routing_stats(ir)
    figures.add(SLOT_COPPER, bar_figure([s["net"] for s in stats], [s["length_mm"] for s in stats], title="넷별 동박 길이", y_label="길이 (mm)", unit="mm", fig_id=SLOT_COPPER))


def model3d_figure_of(ir: CircuitIR, library: KicadLibrary | None, figures: ReportFigures) -> None:
    """The built-in 3D preview (isometric) into the ``iso3d`` slot and its scene into ``figures.scene``; each missing input is a sentence.

    The scene is the one the ``MODEL_3D`` artifact is compiled from
    (:func:`~ai_eda.tools.model3d.scene.build_scene` of the IR, the KiCad
    library and the 3D model library it finds); a scene that cannot be built
    is the reason in the sentence.
    """
    pcb = ir.pcb
    if pcb is None or not pcb.placements:
        figures.miss(SLOT_ISO3D, f"{NO_ISO3D_FIGURE}: IR 에 부품 위치가 없습니다")
        return
    if library is None:
        figures.miss(SLOT_ISO3D, f"{NO_ISO3D_FIGURE}: KiCad 라이브러리를 열 수 없습니다")
        return
    try:
        scene = build_scene(ir, library)
    except (ValueError, CompileError, LibraryLookupError) as e:  # SceneError is both a ValueError and a CompileError
        figures.miss(SLOT_ISO3D, "3D 미리보기를 그릴 수 없습니다", str(e))
        return
    figures.scene = scene
    figures.add(SLOT_ISO3D, model3d_figure_from_scene(scene, ir, fig_id=SLOT_ISO3D))


def tolerance_figure_of(ir: CircuitIR, figures: ReportFigures) -> None:
    """The theory-vs-simulation tolerance chart into the ``tolerance`` slot (rows from :func:`~ai_eda.report.figures.tolerance_rows`, nothing recomputed)."""
    rows = tolerance_rows(ir)
    if not rows:
        figures.miss(SLOT_TOLERANCE, f"{NO_TOLERANCE_FIGURE}: 비교할 시뮬레이션 설정이 없습니다")
        return
    figures.add(SLOT_TOLERANCE, tolerance_figure(rows, fig_id=SLOT_TOLERANCE))


def si_figures_of(ir: CircuitIR, library: KicadLibrary | None, figures: ReportFigures) -> None:
    """The signal-integrity figures of the circuit report (a design with ``ir.si`` only): the board coloured by net class, Z0 against width,
    the per-net delay bars of the ``si.critical_length`` record and the SPICE step response of every fresh ``spice.si.<net>`` result
    (:mod:`ai_eda.report.si_figures`); each missing input is a sentence."""
    from ai_eda.report.si_report import SLOT_SI_BOARD, SLOT_SI_DELAY, SLOT_SI_STEP, SLOT_SI_Z0, _safe_id

    if ir.si is None:
        return
    pcb = ir.pcb
    routed = pcb is not None and bool(pcb.tracks or pcb.vias) and bool(pcb.placements)
    if not routed:
        figures.miss(SLOT_SI_BOARD, "배선이 없어 넷 클래스 색 보드 그림이 없습니다")
    elif library is None:
        figures.miss(SLOT_SI_BOARD, NO_LIBRARY_FIGURE)
    else:
        try:
            figures.add(SLOT_SI_BOARD, board_figure(ir, library, copper=True, colour_by="class", fig_id=SLOT_SI_BOARD))
        except (ValueError, LibraryLookupError, LibraryFormatError) as e:
            figures.miss(SLOT_SI_BOARD, "넷 클래스 색 보드 그림을 그릴 수 없습니다", str(e))
    z0 = z0_width_figure(ir, fig_id=SLOT_SI_Z0)
    if isinstance(z0, str):
        lead = "기준 평면이 없어 임피던스가 정의되지 않으므로 Z0–폭 곡선이 없습니다" if "reference plane" in z0 else "Z0–폭 곡선이 없습니다"
        figures.miss(SLOT_SI_Z0, lead, z0)
    else:
        figures.add(SLOT_SI_Z0, z0)
    crit = ir.validation.latest("si.critical_length")
    rows = delay_rows(crit) if crit is not None else []
    if rows:
        figures.add(SLOT_SI_DELAY, delay_bar_figure(rows, title=f"{ir.project.id}: 넷별 배선 지연과 임계 길이 문턱", fig_id=SLOT_SI_DELAY))
    else:
        figures.miss(SLOT_SI_DELAY, "넷별 지연 그림이 없습니다", "si.critical_length 기록에 배선 지연이 있는 넷이 없습니다" if crit is not None else "si.critical_length 기록 없음")
    results = [r for k, r in sorted(ir.validation.latest_by_check().items()) if k.startswith("spice.si.") and r.status is not ValidationStatus.NOT_APPLICABLE]
    reasons: list[str] = []
    not_verified: list[str] = []
    used: set[str] = set()
    for r in results:
        if r.status is ValidationStatus.NOT_VERIFIED:
            not_verified.append(r.check_id.removeprefix("spice.si."))
            continue
        raw = fresh_si_rawfile(ir, r)
        if isinstance(raw, str):
            reasons.append(raw)
            continue
        fig_id = f"{SLOT_SI_STEP}_{_safe_id(r.check_id.removeprefix('spice.si.'))}"
        while fig_id in used:
            fig_id += "_"
        used.add(fig_id)
        try:
            figures.add(SLOT_SI_STEP, step_response_figure(r, raw, fig_id=fig_id))
        except (ValueError, OSError) as e:
            reasons.append(f"{r.check_id}: {e}")
    if not_verified:
        reasons.append(f"spice.si 결과 {len(not_verified)}개({', '.join(not_verified)})는 NOT_VERIFIED 라 파형이 없습니다 (이유는 각 결과의 메시지)")
    if not figures.slots.get(SLOT_SI_STEP):
        figures.miss(SLOT_SI_STEP, "SPICE 계단 응답 그림이 없습니다", "; ".join(reasons) or "spice.si 결과 없음: 기준 평면 위의 전기적으로 긴 넷이 없습니다")
    elif reasons:
        figures.miss(SLOT_SI_STEP, "그리지 않은 결과: " + "; ".join(reasons))


def stage_figures(stage: Stage, ir: CircuitIR, library: KicadLibrary | None) -> ReportFigures:
    """Every figure the report of ``stage`` shows (theory: template curves + waveform; circuit: placement, board, copper bars, the SI figures,
    3D preview; final: tolerance + waveform; parts: none)."""
    figures = ReportFigures()
    if stage is Stage.ARCHITECTURE:
        theory_figures_of(ir, figures)
        waveform_figures_of(ir, figures)
    elif stage is Stage.PCB:
        board_figures_of(ir, library, figures)
        si_figures_of(ir, library, figures)
        model3d_figure_of(ir, library, figures)
    elif stage is Stage.RELEASE:
        tolerance_figure_of(ir, figures)
        waveform_figures_of(ir, figures)
    return figures


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


def _simulation_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    sim = ir.simulation
    out = ["## 시뮬레이션 설정", ""]
    if sim is None:
        out += ["시뮬레이션 설정 없음.", ""]
        out += ["### 측정 파형", ""] + figures.lines(SLOT_WAVEFORM)
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
    level_kinds = [k for k in (Reduce.DB_AT, Reduce.RMS, Reduce.DB_RMS, Reduce.HARMONIC_DBC, Reduce.AM_DEPTH) if any(e.reduce is k for e in sim.expectations)]
    if level_kinds:
        out += ["### 레벨·구간 측정식", "",
                "ngspice 가 쓴 표본을 잇는 구간별 선형 보간 파형에서 이 프로젝트가 직접 계산합니다(덱에는 .meas / .four 가 없습니다). "
                "구간 [t_start, t_stop] 은 저장된 표본 안에 있어야 하고, 결과마다 가장 큰 표본 간격이 기록됩니다.", ""]
        formulas = {
            Reduce.DB_AT: "    L = 20·log10(|v(at)| / R)   (R = |기준 벡터(at)| 또는 params.ref; 두 크기는 같은 두 표본 사이에서 읽음)",
            Reduce.RMS: "    RMS = sqrt( (1/T) · Σ h·(v0² + v0·v1 + v1²)/3 )   (T = t_stop − t_start)",
            Reduce.DB_RMS: "    L = 20·log10(RMS(v) / RMS(기준))   (기준 RMS 가 0 이면 값 없음)",
            Reduce.HARMONIC_DBC: "    A_n = (2/T)·|∫ v(t)·e^(−j2πn·f0·t) dt|,  L_k = 20·log10(A_k / A_1) dBc   (정수 주기, 최대 간격 ≤ 1/(20·k·f0))",
            Reduce.AM_DEPTH: "    m = 100·(A_max − A_min)/(A_max + A_min) %   (반송파 주기마다 A = (max − min)/2, f_carrier ≥ 20·f_mod, 최대 간격 ≤ 1/(16·f_carrier))",
        }
        for k in level_kinds:
            out += [f"`{k.value}` ({REDUCE_LABELS[k]}):", "", formulas[k], ""]
        out += ["dB 기대값은 tol_abs 로만 판정합니다(로그 값의 상대 허용치는 허용치가 아닙니다). 값을 낼 수 없는 경우(간격이 너무 큼, 기준 0, 기본파 없음)는 이유와 함께 FAIL 이고, "
                "엔진 분해능 이하의 고조파 레벨은 PASS 가 되지 않습니다(NOT_VERIFIED).", ""]
    out += ["### 측정 파형", "",
            "아래 파형은 현재 IR 의 넷리스트로 실행된 최신 ngspice 결과(`results.json`)에서 기대값이 읽는 벡터(템플릿이 지정한 넷 포함)를 그대로 그린 것입니다. 판정은 최종 보고서의 '이론값 대 시뮬레이션' 표에 있습니다.", ""]
    out += figures.lines(SLOT_WAVEFORM)
    return out


def theory_report(ir: CircuitIR, library: KicadLibrary | None = None, *, figures: ReportFigures | None = None) -> str:
    """Report 1: the circuit theory of the template that built ``ir`` with the IR's numbers, its inputs, constraints and simulation setup.

    ``library`` is accepted for symmetry with the other builders and not read:
    theory needs no library fact. ``figures`` are the report's figures
    (:func:`stage_figures`; built here when not given): the template's
    theory curves and the measured waveform, each printed as a placeholder
    line and its caption, or as one sentence saying what is missing.
    """
    if figures is None:
        figures = stage_figures(Stage.ARCHITECTURE, ir, library)
    out = _header(REPORT_TITLES[Stage.ARCHITECTURE], ir)
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
    out += ["## 이론 그림", "", "템플릿의 이론 식을 이 설계의 파라미터 값으로 그린 곡선입니다(IR 의 `parameters` 만 읽음). 시뮬레이션 결과가 아니며 판정도 아닙니다.", ""]
    out += figures.lines(SLOT_THEORY)
    if ir.si is not None:
        from ai_eda.report.si_report import si_theory_section

        out += si_theory_section(ir)
    out += ["## 제약 조건", ""]
    if ir.constraints:
        rows = [[f"`{c.id}`", c.kind.value, c.target, c.description, ", ".join(f"{k} = {_traced_text(v)}" for k, v in c.parameters.items()) or "-"] for c in ir.constraints]
        out.append(_table(["id", "종류", "대상", "내용", "파라미터"], rows))
    else:
        out.append(f"제약 조건 {NO_RECORD}.")
    out.append("")
    out += _simulation_section(ir, figures)
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
    out = _header(REPORT_TITLES[Stage.COMPONENT_SELECTION], ir)
    out += ["## 부품 목록", ""]
    if not ir.components:
        out += [f"부품 {NO_RECORD}: 설계가 비어 있습니다.", "", NO_CHART_IN_PARTS, ""]
        return "\n".join(out).rstrip("\n") + "\n"
    rows = [[f"`{c.ref}`", c.value, c.description or "-", f"`{_lib_id(c, 'symbol')}`", f"`{_lib_id(c, 'footprint')}`", c.package.value if c.package is not None else NO_RECORD, _pins_text(c)] for c in ir.components]
    out.append(_table(["ref", "값", "설명", "심볼", "풋프린트", "패키지", "핀"], rows))
    template = template_for(ir)
    spelled = [x for x in (display_spelled(c) for c in ir.components) if x is not None]
    if template is not None and spelled and all(spelled):
        # only where every numeric part carries the display spelling: a v0.1 design keeps the netlist spelling (never rebuilt)
        out += ["", VALUE_COLUMN_NOTE]
    out += ["", f"'{SUBSTITUTES_HEADING}' 아래의 부품 이름은 제안일 뿐이며 파이프라인은 그 부품의 핀 배열·정격을 확인하지 않았습니다. "
            "부품의 심볼·풋프린트·핀은 KiCad 라이브러리 파일에서 읽은 것이고, 존재 확인의 세부 항목이 무엇이 확인되었는지를 말합니다.", "", NO_CHART_IN_PARTS, ""]
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
#: who decides what happens to a board with unrouted nets (the circuit report, routing.maze 0.2): the user's opt-in answer, never an agent
PARTIAL_ROUTING_RULE = (
    "그런 보드는 기본적으로 배치만 남기며(all-or-nothing), 사용자가 `--answer pcb.routing=partial` 로 답했을 때만 "
    "완성된 넷을 넷 단위로 적용합니다."
)
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
    p = {k: parameter_value(ir, k) for k in ("v_in", "r_c", "r1", "r2", "i_led", "i_load_budget")}
    if found[0] == "atmega128_devboard" and p["i_load_budget"] is not None:
        return "I_load (+5V 레일의 설계 부하 예산, 측정값 아님)", p["i_load_budget"]
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


#: ``derived_from`` entries a placer writes per part (shown per row, not in the parameter line)
_PER_PART_ENTRIES = ("footprint:", "ring:", "pull_angle_deg:", "pull:", "core:")
_RING_NAMES = {"inner": "안쪽", "outer": "바깥", "core": "코어"}


def _placement_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    out = ["## 배치", ""]
    pcb = ir.pcb
    if pcb is None or not pcb.placements:
        out += [f"배치 {NO_RECORD}: IR 에 부품 위치가 없습니다.", ""]
        out += figures.lines(SLOT_PLACEMENT)
        return out
    first = pcb.placements[0].provenance
    tools = sorted({(p.provenance.tool or "", p.provenance.tool_version or "") for p in pcb.placements})
    params = sorted({e for p in pcb.placements for e in p.provenance.derived_from if not e.startswith(_PER_PART_ENTRIES)})
    ring = any(t == RING_PLACER_ID for t, _v in tools)
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
    if ring:
        ring_rows = []
        for p in pcb.placements:
            entry = dict(e.split(":", 1) for e in p.provenance.derived_from if ":" in e)
            ring_rows.append([f"`{p.component_ref}`", _RING_NAMES.get(entry.get("ring", ""), entry.get("ring") or NO_RECORD),
                              entry.get("pull_angle_deg") or NO_RECORD, f"`{entry['pull']}`" if entry.get("pull") else NO_RECORD])
        out += ["부품별 링과 당김 각 (각 배치의 provenance `derived_from` 그대로):", "", _table(["ref", "링", "당김 각 (°)", "당김 출처"], ring_rows), ""]
    out += figures.lines(SLOT_PLACEMENT)
    if ring:
        core = next((e.split(":", 1)[1] for p in pcb.placements for e in p.provenance.derived_from if e.startswith("core:")), NO_RECORD)
        counts = {name: sum(1 for p in pcb.placements if f"ring:{name}" in p.provenance.derived_from) for name in ("inner", "outer")}
        out += [
            "### 배치 규칙과 그 한계", "",
            f"`{RING_PLACER_ID}` 는 패드가 {CORE_MIN_PADS}개 이상인 부품이 있을 때 PCB 에이전트가 쓰는 배치기입니다. 패드(서로 다른 패드 번호)가 가장 많은 부품이 코어이며 "
            f"(이 보드: `{core}`) 보드 중앙에 회전 0 으로 놓입니다. 다른 부품마다 '당김 각'을 구합니다: 그 부품이 넷으로 이어지는 코어 패드들의 방향을 코어 중심에서 본 원형 평균 "
            "(0° = 동쪽, 화면에서 반시계 방향). 전원·접지 넷(종류 POWER / GROUND)은 신호 넷으로 코어에 닿지 않는 부품에만 셈에 넣고(크리스탈 커패시터는 GND 패드가 아니라 XTAL 핀을 따름), "
            "코어 패드에 닿지 않는 넷으로 서로 이어진 부품들은 한 그룹으로 그룹 전체의 각을 따릅니다(표의 '당김 출처': `signal` / `rail` / `group:<참조들>` / `none`). "
            "각이 같은 부품은 당김 출처로 먼저 나뉘어 그룹 사이에 다른 부품이 끼지 않고, 그룹 안에서는 코어에 닿지 않는 넷을 따라 그룹의 커넥터(없으면 첫 부품)부터 "
            "너비 우선으로 이어지는 순서(예: DC 잭 → 다이오드 → 입력 커패시터 → 레귤레이터)로 놓입니다.", "",
            f"안쪽 링(이 보드 {counts['inner']}개): 패드가 {INNER_MAX_PADS}개 이하이고, 연결된 모든 핀의 넷이 코어 패드에도 닿으며, 참조 접두가 "
            f"{' / '.join(EDGE_REF_PREFIXES)} (커넥터·스위치)가 아닌 부품 — 디커플링, 크리스탈과 부하 커패시터, 리셋 R/C, 풀업. 코어 extent 밖 `spacing_mm` 에서 시작하는 띠에 놓입니다. "
            f"바깥 링(이 보드 {counts['outer']}개): 나머지 전부(커넥터, 레귤레이터 쪽 부품, LED 와 저항, 스위치)로, 보드 가장자리에서 `margin_mm` 안쪽 띠의 바깥 가장자리에 붙습니다. "
            "각 띠는 남동쪽 모서리에서 시작해 반시계 방향으로 도는 네 변이며, 부품은 당김 각이 가리키는 위치 순서로 놓이고 변을 따라 눕습니다(긴 쪽이 변과 나란히; 커넥터·스위치는 핀 1 쪽이 바깥; "
            "두 방향 중 패드가 넷의 상대편 - 코어 패드, 없으면 같은 넷의 다른 부품 - 에 더 가까운 쪽). "
            f"예외: extent 가 한쪽으로 패드보다 {BODY_OVERHANG_MM:g} mm 이상 더 튀어나온 커넥터·스위치(DC 잭의 몸체처럼)는 그 쪽이 보드 가장자리를 향하도록 변에 수직으로 돌려 놓아, "
            "몸체 끝이 외곽에서 `margin_mm` 안쪽에 오고 그 앞에는 부품이 없습니다(어느 끝이 플러그 입구인지는 풋프린트가 말하지 않으므로 읽지 않습니다). "
            "외곽은 바깥 띠 + `margin_mm` 을 1 mm 단위로 올린 크기이고 코어 중심이 정수 mm 에 옵니다. 배치 뒤 모든 extent 쌍이 서로 떨어져 있고 외곽 안에 있음을 재측정해 확인하며, 아니면 거부합니다.", "",
            "정직한 한계: 이 배치기는 연결의 방향만 봅니다. 디커플링 커패시터와 그 전원 핀 사이의 실제 거리, 신호 무결성, 열, EMI 는 고려하지 않고, "
            "커넥터의 결합 방향은 위의 몸체 규칙으로만 다룹니다. "
            "배치의 유효성(courtyard 겹침, 외곽 위반)은 kicad-cli DRC 만이 판정합니다.", "",
        ]
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


def _routing_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
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
                    "라우터가 이 보드에서 잇지 못한 넷과 그 이유(한 넷이라도 남으면 구리를 통째로 제안하지 않음)는 아래 단계 기록의 `placement` 메시지에 그대로 있습니다. "
                    "이 보드가 DRC 를 통과하는지는 이 보고서가 판정하지 않으며, kicad-cli 가 실제로 낸 결과만 아래 단계 기록의 `drc` 항목에 옮깁니다.", ""]
        out += figures.lines(SLOT_BOARD) + figures.lines(SLOT_COPPER)
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
    out += figures.lines(SLOT_BOARD) + figures.lines(SLOT_COPPER)
    if None not in (g, w, c, e, d_v) and "max_iterations" in params:
        # routing.maze 0.2 records its negotiation numbers in the params entry: the negotiated model, with the values the copper was routed at
        out += [
            "### 라우터의 keep-out 규칙 (파라미터 값을 넣은 것)", "",
            "격자 미로 라우터는 F.Cu / B.Cu 두 층의 정사각 격자 위에서 넷마다 A* 탐색(한 칸 `base_cost`, 방향 전환 `bend_cost`, 층 전환(비아) `via_cost`; "
            "휴리스틱은 맨해튼 거리)으로 잇고, 패드가 여럿인 넷은 패드 중심의 무게중심에 가장 가까운 패드에서 시작해 가장 가까운 패드를 하나씩 더하는 Steiner 트리로 키웁니다. "
            "넷끼리의 겹침은 금지가 아니라 비용으로 협상합니다(rip-up and reroute): 다른 넷의 간격 영역 안의 셀은 반복마다 "
            f"`present_cost` = {params.get('present_cost')} 에서 `present_growth` = {params.get('present_growth')} 배씩 비싸지고, 겹쳤던 셀에는 `history_cost` = "
            f"{params.get('history_cost')} 가 쌓이며, 겹친 넷만 다시 배선해 겹침이 없어지거나 `max_iterations` = {params.get('max_iterations')} 에 이르면 멈춥니다. "
            "내보내는 구리는 서로 다음 거리를 모두 지킵니다 (중심선 기준 거리):", "",
            f"    다른 넷 패드 상자로부터        r = c + w/2 + g/2 = {c:g} + {w / 2:g} + {g / 2:g} = {c + w / 2 + g / 2:.3f} mm",
            f"    다른 넷 트랙으로부터           r = w + c + g/2 = {w + c + g / 2:.3f} mm   (두 중심선 거리 ≥ w/2 + c + w/2)",
            f"    비아와 다른 넷 트랙 (양 층)    r = d_v/2 + c + w/2 = {d_v / 2 + c + w / 2:.3f} mm",
            f"    비아와 다른 넷 비아            r = d_v + c = {d_v + c:.3f} mm",
            f"    보드 가장자리                  e + w/2 = {e + w / 2:.3f} mm   (비아는 e + d_v/2 = {e + d_v / 2:.3f} mm)", "",
            "g/2 항은 격자 셀 사이를 지나는 구간까지 보수적으로 덮는 여유입니다. 비아는 어떤 패드 안에도 놓지 않고, 합법적인 배선을 찾지 못한 넷은 구리 없이 남깁니다(반쯤 배선된 넷 없음). "
            f"{PARTIAL_ROUTING_RULE} "
            "이 규칙은 라우터 자체의 파라미터이지 설계 규칙이 아니며, 보드의 유효성은 kicad-cli DRC 가 판정합니다.", "",
        ]
    elif None not in (g, w, c, e, d_v):
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
        if not width > 0:
            # IPC-2221's area**0.725 of a non-positive width is complex / zero: no display value, say so instead
            out += [f"- 폭 {width:g} mm 트랙: 폭이 양수가 아니라 IPC-2221 전류 용량을 계산하지 않습니다.", ""]
            continue
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
    if longest is not None and longest["length_mm"] > 0 and longest["widths"] and min(longest["widths"]) > 0:
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
    """The latest ``pcb.routing.connectivity`` / ``pcb.routing.clearance`` results with their per-net rows, copied.

    The clearance pair / violation counts are printed only when the check
    compared copper against a limit (``limit_mm`` a number, no malformed
    copper); its no-limit, malformed and outline branches write
    ``pairs_compared=0, violations=[]`` without comparing anything, so those
    print :data:`NO_RECORD` - never "0 violations" for a comparison that did
    not run.
    """
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
            compared = isinstance(lim, (int, float)) and not isinstance(lim, bool) and not r.details.get("malformed")
            if compared:
                counts = f"비교한 쌍 {r.details.get('pairs_compared', NO_RECORD)}, 위반 {len(r.details.get('violations') or [])}"
            else:
                counts = f"비교한 쌍 {NO_RECORD}, 위반 {NO_RECORD}"
            out.append(f"  - 한계 {NO_RECORD if lim is None else f'{lim:g} mm'}, {counts}")
    out.append("")
    return out


#: the silk text kinds in the words of the circuit report
SILK_KIND_WORDS: dict[str, str] = {SilkKind.REFERENCE: "참조", SilkKind.TITLE: "제목", SilkKind.PIN_LABEL: "핀 라벨", SilkKind.USER: "사용자"}
#: the silkscreen placer's candidate names (``REFERENCE_CANDIDATES``) in the words of the circuit report
SILK_CANDIDATE_WORDS: dict[str, str] = {
    "above": "위", "below": "아래", "left": "왼쪽", "right": "오른쪽",
    "top-left": "왼쪽 위 모서리", "top-right": "오른쪽 위 모서리", "bottom-left": "왼쪽 아래 모서리", "bottom-right": "오른쪽 아래 모서리",
}
#: rows of a silk check's details the circuit report prints at most (the counts are complete)
_SILK_ROWS = 20
#: the KiCad 3D exports the circuit report lists (MANUFACTURING_OUTPUTS registers them when kicad-cli runs)
KICAD_3D_KINDS: tuple[ArtifactKind, ...] = (ArtifactKind.KICAD_STEP, ArtifactKind.KICAD_GLB, ArtifactKind.KICAD_RENDER)


def silk_params_of(ir: CircuitIR) -> dict[str, str] | None:
    """The silkscreen placer's parameters as the first silk text's provenance records them (``params:silk_to_pad=0.15,...``); ``None`` without one."""
    if ir.pcb is None:
        return None
    for t in ir.pcb.silkscreen:
        for entry in t.provenance.derived_from:
            if entry.startswith("params:"):
                out: dict[str, str] = {}
                for kv in entry[len("params:"):].split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        out[k.strip()] = v.strip()
                return out
    return None


def _artifact_freshness(a: ArtifactRef, design_hash: str) -> str:
    if a.generated_from_ir_hash is None:
        return "생성 IR 해시 없음"
    return "현재 IR 에서 생성" if a.generated_from_ir_hash == design_hash else f"오래됨 (IR {a.generated_from_ir_hash[:16]})"


def _silk_rules(params: dict[str, str]) -> list[str]:
    """The placer's rules with the recorded parameter values (``silkscreen.place`` 0.1); the fixed parts are the placer's own constants."""
    get = params.get
    sizes = get("reference_sizes", NO_RECORD).replace("/", " mm → ")
    candidates = ", ".join(f"{SILK_CANDIDATE_WORDS.get(name, name)} {rot:g}°" for name, rot in REFERENCE_CANDIDATES)
    return [
        "### 실크 배치 규칙 (파라미터 값을 넣은 것)", "",
        f"`{SILK_PLACER_ID}` 는 배치·배선이 끝난 보드에 참조 지정자, 커넥터 핀 라벨, 보드 제목을 놓습니다. IR 에 실크가 비어 있을 때만 놓고, 이미 있는 실크는 바꾸지 않습니다. "
        "면(F / B)마다 다음을 금지 영역(keep-out)으로 둡니다:", "",
        f"    패드 구리 (라이브러리 형상: 사각 / 둥근 사각 / 원 / 캡슐) + silk_to_pad = {get('silk_to_pad', NO_RECORD)} mm",
        f"    보드 외곽을 silk_to_edge = {get('silk_to_edge', NO_RECORD)} mm 안쪽으로 줄인 사각형의 바깥",
        f"    풋프린트 자신의 실크 선·문자 (1번 핀·극성 표시) + gap = {get('gap', NO_RECORD)} mm",
        f"    먼저 놓은 문자 + gap = {get('gap', NO_RECORD)} mm", "",
        "트랙은 금지 영역이 아닙니다(솔더 마스크 아래 구리 위의 실크는 정상). 비아도 아닙니다: 컴파일된 보드가 모든 비아를 마스크로 덮습니다(텐팅). "
        f"참조 지정자는 자연 순서(C1, C2, …, R10)로, 코트야드 둘레의 후보 {len(REFERENCE_CANDIDATES)}곳({candidates})을 문자 크기 {sizes} mm 순으로 시험해 "
        f"금지 영역에 닿지 않고 다른 풋프린트의 코트야드에서 gap = {get('gap', NO_RECORD)} mm 이상 떨어진 첫 곳에 놓습니다(굵기는 KiCad 기본 0.15 mm; "
        "이웃의 코트야드 안에 있는 참조 지정자는 그 이웃의 것으로 읽히기 때문 — 모든 후보가 이웃의 코트야드에 걸리면 금지 영역만 피한 첫 곳에 놓고 그 코트야드를 메모에 적습니다). "
        "빈 곳이 없으면 조립도 층(F.Fab / B.Fab)으로 옮기고 이름을 적습니다. "
        f"핀 라벨을 받을 커넥터(`{CONNECTOR_LIBRARY_PREFIX}*` 심볼 라이브러리)의 참조는 그 라벨 쪽을 마지막에 시험합니다. "
        f"커넥터 핀 라벨: 패드마다 넷 이름(이름 없는 넷, `Net-(…)` 제외)을 {get('pin_label', NO_RECORD)} mm 문자로 코트야드 바깥, 패드 옆에 놓으며, "
        "충돌하는 라벨은 겹치지 않고 건너뛴 뒤 이름을 적습니다. "
        f"제목: 프로젝트 이름(ASCII 만)을 {get('title', NO_RECORD)} mm 로 여유가 가장 큰 보드 모서리에 놓고, 모서리가 모두 막히면 위·아래 변을 따라 "
        f"{TITLE_SLIDE_STEP_MM:g} mm 씩 밀어 모서리에 가장 가까운 빈 자리에 놓습니다. 날짜·해시는 넣지 않습니다(결정성).", "",
        f"문자 상자는 추정입니다: 폭 = 글자 수 × 크기 × {TEXT_WIDTH_FACTOR:g} + 굵기, 높이 = 크기 × {TEXT_HEIGHT_FACTOR:g} + 굵기 (KiCad 스트로크 글꼴의 실제 치수가 아님). "
        "이 여유는 배치기의 것이지 fab 규칙이 아니며, 컴파일된 보드의 실크 판정(`silk_over_copper` / `silk_overlap` / `text_height`)은 kicad-cli DRC 의 몫입니다 "
        "(KiCad 10.0.6 에서 이 보드들에 대한 실크 DRC 는 아직 측정되지 않았습니다).", "",
    ]


def _silk_results(ir: CircuitIR) -> list[str]:
    """The latest ``pcb.silk.*`` results with their violation rows, copied (IR geometry, not DRC)."""
    out = ["### IR 기하 검사 결과 (`pcb.silk.*`; DRC 아님)", ""]
    for check in (SILK_CLEARANCE_CHECK, SILK_OVERLAP_CHECK, SILK_SIZE_CHECK):
        r = ir.validation.latest(check)
        if r is None:
            out.append(f"- `{check}`: {NO_RECORD}")
            continue
        out.append(f"- `{check}`: **{r.status}** ({_tool_text(r.tool, r.tool_version)}) — {_cell(r.message)}")
        details = r.details if isinstance(r.details, dict) else {}
        rows = details.get("violations")
        if isinstance(rows, list):
            for row in rows[:_SILK_ROWS]:
                if isinstance(row, dict):
                    out.append(f"  - {row.get('status', '?')}: {_cell(row.get('message', ''))}")
            count = details.get("violation_count", len(rows))
            if isinstance(count, int) and count > _SILK_ROWS:
                out.append(f"  - … 위반 {count}개 중 {_SILK_ROWS}개만 적음 (나머지는 검사 결과의 details)")
        below = details.get("below_margin")
        if check == SILK_CLEARANCE_CHECK and isinstance(below, list) and below:
            out.append(f"  - 문자 여유보다 가까운 라이브러리 실크 {len(below)}곳 (풋프린트 자체의 설계; 패드와 닿지 않으면 실패가 아님):")
            out += [f"    - {_cell(row.get('message', ''))}" for row in below[:_SILK_ROWS] if isinstance(row, dict)]
        own = details.get("library_own_overlaps")
        if check == SILK_OVERLAP_CHECK and isinstance(own, list) and own:
            out.append(f"  - 같은 풋프린트의 라이브러리 문자와 실크가 (추정 문자 상자로) 닿는 곳 {len(own)}곳 (풋프린트 자체의 설계; 실패가 아님):")
            out += [f"    - {_cell(row.get('message', ''))}" for row in own[:_SILK_ROWS] if isinstance(row, dict)]
    out.append("")
    return out


def _silkscreen_section(ir: CircuitIR) -> list[str]:
    """The designed silkscreen (counts, table, the placer's rules with its recorded parameters) and the ``pcb.silk.*`` verdicts, copied."""
    out = ["## 실크스크린", ""]
    pcb = ir.pcb
    if pcb is None or not pcb.placements:
        out += [f"실크스크린 {NO_RECORD}: IR 에 배치된 보드가 없습니다.", ""]
        return out + _silk_results(ir)
    texts = list(pcb.silkscreen)
    if not texts:
        out += [
            "IR 에 설계된 실크 문자가 없습니다(`--answer pcb.silkscreen=skip` 으로 건너뛰었거나 실크를 놓기 전의 보드). 컴파일된 보드의 참조 지정자는 "
            "풋프린트 라이브러리의 기본 위치에 있고, 풋프린트 자신의 실크 선(외곽, 1번 핀·극성 표시)은 라이브러리 그대로입니다. 배치도·보드 그림의 검정 선이 그 실크 선이며, "
            "그림의 참조·값 라벨은 읽기용 배치일 뿐 실크가 아닙니다.", "",
        ]
        return out + _silk_results(ir)
    tools = sorted({(t.provenance.tool or "", t.provenance.tool_version or "") for t in texts})
    if any(t for t, _v in tools):
        out.append("- 실크 도구: " + ", ".join(_tool_text(t, v) for t, v in tools if t))
    else:
        out.append(f"- 실크 도구 {NO_RECORD}: 문자의 출처는 {_provenance_text(texts[0].provenance)}")
    params = silk_params_of(ir)
    if params:
        out.append("- 실크 파라미터 (첫 문자의 provenance `params:` 항목): " + ", ".join(f"`{k}` = {v}" for k, v in params.items()))
    refs = [t for t in texts if t.kind == SilkKind.REFERENCE]
    on_fab = [t.component_ref or t.text for t in refs if not t.layer.endswith(".SilkS")]
    labels = [t for t in texts if t.kind == SilkKind.PIN_LABEL]
    titles = [t for t in texts if t.kind == SilkKind.TITLE]
    users = [t for t in texts if t.kind == SilkKind.USER]
    fab_text = f"조립도 층 {len(on_fab)}개" + (f" ({', '.join(on_fab)})" if on_fab else "")
    title_text = ", ".join(f"'{t.text}'" for t in titles) or "없음"
    out.append(f"- 실크 문자 {len(texts)}개: 참조 {len(refs)}개 (실크 {len(refs) - len(on_fab)}개, {fab_text}), 커넥터 핀 라벨 {len(labels)}개, 제목 {title_text}, 사용자 문자 {len(users)}개")
    out.append("- 건너뛴 핀 라벨과 그 이유, 제목의 위치 설명은 아래 단계 기록의 `placement` 메시지에 그대로 있습니다. 위의 배치도·보드 그림에 검정으로 그려진 것이 이 실크입니다.")
    out.append("")
    origins = [_provenance_text(t.provenance) for t in texts]
    shared = len(set(origins)) == 1  # one origin for every text: said once, not in every row
    if shared:
        out += [f"모든 실크 문자의 출처: {origins[0]}.", ""]
    rows = []
    for t, origin in zip(texts, origins):
        rows.append([SILK_KIND_WORDS.get(t.kind, str(t.kind)), f"`{t.text}`", t.component_ref or "-", f"{t.x_mm:g}", f"{t.y_mm:g}", f"{t.rotation_deg:g}", t.layer,
                     f"{t.size_mm:g} / {t.thickness_mm:g}", t.justify, *([] if shared else [origin])])
    header = ["종류", "문자", "부품", "x (mm)", "y (mm)", "회전 (°)", "층", "크기 / 굵기 (mm)", "정렬", *([] if shared else ["출처"])]
    out += [_table(header, rows), ""]
    if params and any(t == SILK_PLACER_ID for t, _v in tools):
        out += _silk_rules(params)
    return out + _silk_results(ir)


def _model3d_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    """The built-in 3D preview (figure, the body rule, one row per part box from the scene), the registered preview GLB and KiCad's own 3D exports."""
    out = ["## 3D 미리보기", ""]
    out += figures.lines(SLOT_ISO3D)
    design_hash = ir.content_hash()
    preview = ir.artifacts.get(ArtifactKind.MODEL_3D)
    if preview is not None:
        out.append(f"- 내장 미리보기 파일: `{Path(preview.path).name}` (산출물 `{ArtifactKind.MODEL_3D.value}`, {_artifact_freshness(preview, design_hash)}) — glTF 2.0 바이너리. "
                   "같은 IR 과 같은 라이브러리에서 바이트 단위로 같은 파일이며, 그림이지 검사가 아니므로 판정을 기록하지 않습니다.")
    else:
        out.append("- 내장 미리보기 파일: 등록되지 않음 (PCB 단계가 보드를 컴파일한 뒤 만듭니다; 이유는 아래 단계 기록의 `pcb` 메시지).")
    out += [
        f"- 부품 상자 규칙: {BODY_CAPTION}. 상자의 x·y 는 풋프린트 F.Fab 그래픽의 경계상자(없으면 코트야드)이고, 높이는 풋프린트의 `(model …)` 이 가리키는 STEP 파일의 "
        "좌표점과 원에서 읽은 외곽 상자(모델의 offset / scale / rotate 적용)의 가장 높은 점입니다. 이 외곽은 근사이지 측정이 아니며, STEP 파일을 찾지 못한 부품은 "
        "평면 외곽선만 그립니다. 실크 문자는 3D 에서 생략하고(실크 선은 라이브러리의 것), 구리·마스크·실크의 두께는 그림용 값입니다.",
    ]
    scene = figures.scene
    if scene is not None:
        out.append("")
        rows = []
        for b in scene.bodies:
            height = f"{b.height_mm:.2f}" if b.height_mm is not None else "없음 (평면 외곽선)"
            models = ", ".join(f"`{Path(m.replace(chr(92), '/')).name}`" for m in b.models) or "-"
            rows.append([f"`{b.ref}`", f"`{b.lib_id}`", "앞" if b.side == "top" else "뒤", height, b.outline_source or "-", models, b.reason or "-"])
        out += [_table(["부품", "풋프린트", "면", "높이 (mm)", "x·y 출처", "STEP 모델", "비고"], rows)]
    out.append("")
    exports = [(kind, ir.artifacts.get(kind)) for kind in KICAD_3D_KINDS]
    if any(a is not None for _k, a in exports):
        out.append("KiCad 3D 모델(실제 부품 모양), MANUFACTURING_OUTPUTS 단계가 kicad-cli 로 내보낸 파일 (플래그는 KiCad 10.0.6 에서 아직 측정되지 않음; 파일에 KiCad 의 시각이 들어가 해시는 재현성이 아니라 어느 파일인지를 말함):")
        out.append("")
        for kind, a in exports:
            if a is None:
                out.append(f"- `{kind.value}`: 등록되지 않음")
            else:
                names = ", ".join(f"`{Path(f).name}`" for f in (a.files or [a.path]))
                out.append(f"- `{kind.value}`: {names} ({_artifact_freshness(a, design_hash)})")
    else:
        out.append("KiCad 3D 모델(실제 부품 모양): 등록된 파일 없음. MANUFACTURING_OUTPUTS 단계가 kicad-cli 가 있을 때 `kicad-cli pcb export step` / `pcb export glb` / "
                   "`pcb render` 로 만들며, 이 명령의 플래그는 KiCad 10.0.6 에서 아직 측정되지 않았습니다.")
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


def circuit_report(ir: CircuitIR, library: KicadLibrary | None, record: RunRecord, *, figures: ReportFigures | None = None) -> str:
    """Report 3: the schematic (nets and roles), the placement (rule, positions, limits), the routing (parameters, per-net statistics, reference copper values,
    ``pcb.routing.*`` verdicts), the silkscreen (texts, the placer's rules with its recorded parameters, ``pcb.silk.*`` verdicts) and the 3D preview
    (the isometric figure with its caption, one row per part box, the registered preview GLB and KiCad's own 3D exports).

    The text reads no library fact (every geometry number is in the IR)
    except the 3D section's part boxes, which are the preview scene's (the
    footprints and the STEP files on disk, built once with the figure);
    ``library`` is what the board figures read pad shapes from
    (:func:`stage_figures`, built here when ``figures`` is not given) - the
    library the pipeline resolved, exactly as the PCB compiler reads it.
    """
    if figures is None:
        figures = stage_figures(Stage.PCB, ir, library)
    out = _header(REPORT_TITLES[Stage.PCB], ir)
    out += [
        "회로도와 보드는 이 IR 에서 컴파일한 파생물입니다. 아래는 IR 이 담고 있는 연결·위치·구리와, 그것을 만든 도구가 기록한 규칙, "
        "그리고 참고 공식으로 계산한 표시값입니다. ERC / DRC 판정은 kicad-cli 만이 내리며 그 결과는 단계 기록에 그대로 옮깁니다.", "",
    ]
    out += _schematic_section(ir)
    out += _placement_section(ir, figures)
    out += _routing_section(ir, figures)
    if ir.si is not None:
        from ai_eda.report.si_report import si_circuit_section

        out += si_circuit_section(ir, figures)
    out += _silkscreen_section(ir)
    out += _model3d_section(ir, figures)
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


def _theory_vs_simulation(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    out = ["## 이론값 대 시뮬레이션", ""]
    sim = ir.simulation
    if sim is None or not sim.expectations:
        out += [f"기대값 {NO_RECORD}: 비교할 시뮬레이션 설정이 없습니다.", ""]
        out += figures.lines(SLOT_TOLERANCE) + figures.lines(SLOT_WAVEFORM)
        return out
    summary = ir.validation.latest(SPICE_CHECK)
    out.append(f"- `{SPICE_CHECK}` 최신 결과: " + (f"**{summary.status}** ({_tool_text(summary.tool, summary.tool_version)}) — {_cell(summary.message)}" if summary is not None else NO_RECORD))
    out += ["- 이론 공칭값은 IR 의 기대값(템플릿의 식으로 계산기가 낸 값 또는 사용자 값), 측정값은 ngspice 결과(`spice.<id>` 의 `details.measured`)입니다. "
            "허용치는 SPICE 단계가 결과에 기록한 값(`details.tolerance`)이고, 기록이 없는 기대값만 같은 규칙 max(tol_abs, tol_rel·|공칭값|) 로 IR 에서 계산합니다; "
            "공칭값이 0 이면 tol_rel 은 허용치가 아니므로 tol_abs 만 셉니다(SPICE 단계의 판정 규칙과 같음). 판정은 저장된 결과의 상태 그대로입니다.", ""]
    rows = []
    extra: list[str] = []
    for e in sim.expectations:
        r = ir.validation.latest(f"{SPICE_CHECK}.{e.id}")
        unit = e.nominal.unit
        nominal = float(e.nominal.value) if isinstance(e.nominal.value, (int, float)) and not isinstance(e.nominal.value, bool) else None
        details = r.details if r is not None and isinstance(r.details, dict) else {}
        measured = details.get("measured")
        measured = float(measured) if isinstance(measured, (int, float)) and not isinstance(measured, bool) else None
        # the recorded limit first (the one the verdict was judged against); the stage's rule on the IR only without a record
        limit, _recorded = expectation_limit(e, details if r is not None else None)
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
    out += figures.lines(SLOT_TOLERANCE)
    out += ["### 측정 파형", ""] + figures.lines(SLOT_WAVEFORM)
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


def final_report(ir: CircuitIR, record: RunRecord, *, figures: ReportFigures | None = None) -> str:
    """Report 4: requirements, every stage outcome, theory vs simulation per expectation (table, tolerance chart, waveform), the verification matrix, artifacts, the recorded RELEASE outcome, remaining work, conclusion."""
    if figures is None:
        figures = stage_figures(Stage.RELEASE, ir, None)
    out = _header(REPORT_TITLES[Stage.RELEASE], ir)
    out += _requirements_section(ir)
    out += _all_stages_section(record)
    out += _theory_vs_simulation(ir, figures)
    out += _matrix_section(ir)
    out += _artifacts_section(ir)
    out += _release_section(record)
    out += _remaining_section(ir)
    out += _conclusion_section(ir, record)
    return "\n".join(out).rstrip("\n") + "\n"


# --- writers -------------------------------------------------------------------------


@dataclass(frozen=True)
class StageDocument:
    """One report built once: its title, the Markdown text, the same text rendered as a self-contained HTML page, and the figures the placeholders name."""

    stage: Stage
    title: str
    markdown: str
    html: str
    figures: ReportFigures


def build_stage_document(stage: Stage, ir: CircuitIR, library: KicadLibrary | None, record: RunRecord) -> StageDocument | None:
    """The report ``stage`` completes (:data:`STAGE_REPORTS`) as Markdown + HTML with its figures built exactly once; ``None`` for a stage without a report."""
    if stage not in STAGE_REPORTS:
        return None
    figures = stage_figures(stage, ir, library)
    if stage is Stage.ARCHITECTURE:
        md = theory_report(ir, library, figures=figures)
    elif stage is Stage.COMPONENT_SELECTION:
        md = parts_report(ir, library)
    elif stage is Stage.PCB:
        md = circuit_report(ir, library, record, figures=figures)
    else:
        md = final_report(ir, record, figures=figures)
    title = f"{REPORT_TITLES[stage]}: {ir.project.name}"
    return StageDocument(stage, title, md, markdown_to_html(md, title=title, figures=figures.figures), figures)


def build_stage_report(stage: Stage, ir: CircuitIR, library: KicadLibrary | None, record: RunRecord) -> str | None:
    """The Markdown of the report ``stage`` completes (:data:`STAGE_REPORTS`), ``None`` for a stage without one."""
    doc = build_stage_document(stage, ir, library, record)
    return doc.markdown if doc is not None else None


@dataclass(frozen=True)
class StageReportResult:
    """What :func:`write_stage_report` wrote: the ``.md`` and ``.html`` paths, the ``.pdf`` path when a browser printed it, else why not (``pdf_reason``)."""

    stage: Stage
    markdown: Path
    html: Path
    pdf: Path | None
    pdf_reason: str | None
    #: the ids of the figures embedded in the HTML (the Markdown's placeholders)
    figure_ids: tuple[str, ...]

    @property
    def paths(self) -> list[Path]:
        """The files this write produced, ``.md`` first."""
        return [self.markdown, self.html] + ([self.pdf] if self.pdf is not None else [])

    def summary(self) -> str:
        """``(+ .html, .pdf)`` or ``(+ .html; pdf not produced: <reason>)`` for a printed line."""
        if self.pdf is not None:
            return "(+ .html, .pdf)"
        return f"(+ .html; pdf not produced: {self.pdf_reason})"


def _remove_stale_pdf(path: Path) -> None:
    """A ``.pdf`` from an earlier write is removed when this write produces none: a PDF beside the ``.md`` always belongs to it."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_stage_report(
    stage: Stage, ir: CircuitIR, library: KicadLibrary | None, record: RunRecord, workdir: Path, *,
    reports_dir: Path | None = None, browser: Path | str | None = None, pdf: bool = True,
) -> StageReportResult | None:
    """Write the report of ``stage`` as ``<workdir>/reports/<name>.md`` and ``.html`` (LF, UTF-8; ``reports_dir`` overrides the folder) and, when ``pdf`` and a browser is given or found, ``.pdf``.

    ``None`` for a stage without a report. No browser (or ``pdf=False``)
    means no PDF and a reason in the result, never an exception; a stale
    ``.pdf`` from an earlier write is then removed so that a PDF on disk
    always matches the ``.md`` / ``.html`` beside it. Nothing else is
    touched: the IR is not saved, no artifact is registered.
    """
    doc = build_stage_document(stage, ir, library, record)
    if doc is None:
        return None
    folder = Path(reports_dir) if reports_dir is not None else Path(workdir) / REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / STAGE_REPORTS[stage]
    html_path, pdf_path = md_path.with_suffix(".html"), md_path.with_suffix(".pdf")
    md_path.write_text(doc.markdown, encoding="utf-8", newline="\n")
    html_path.write_text(doc.html, encoding="utf-8", newline="\n")
    written: Path | None = None
    reason: str | None = None
    if not pdf:
        reason = PDF_NOT_REQUESTED
        _remove_stale_pdf(pdf_path)
    else:
        exe = Path(browser) if browser is not None else find_browser()
        if exe is None:
            reason = NO_BROWSER_REASON
            _remove_stale_pdf(pdf_path)
        else:
            result = html_to_pdf(html_path, pdf_path, exe)  # removes a stale PDF itself; ok only when this run wrote one
            written, reason = (pdf_path, None) if result.ok else (None, result.reason)
    return StageReportResult(stage, md_path, html_path, written, reason, tuple(doc.figures.figures))


def write_all_stage_reports(
    ir: CircuitIR, library: KicadLibrary | None, record: RunRecord, workdir: Path, *,
    reports_dir: Path | None = None, browser: Path | str | None = None, pdf: bool = True,
) -> list[StageReportResult]:
    """Write the four reports from a saved IR and its run record (``ai-eda stage-reports``); returns the results in stage order."""
    out: list[StageReportResult] = []
    for stage in STAGE_REPORTS:
        result = write_stage_report(stage, ir, library, record, workdir, reports_dir=reports_dir, browser=browser, pdf=pdf)
        if result is not None:
            out.append(result)
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
    "NO_BOARD_FIGURE",
    "NO_CHART_IN_PARTS",
    "NO_COPPER_FIGURE",
    "NO_LIBRARY_FIGURE",
    "NO_MEASUREMENT",
    "NO_PLACEMENT_FIGURE",
    "NO_ROUTING",
    "NO_RUN_RECORD",
    "NO_TEMPLATE",
    "NO_TEMPLATE_INFO",
    "NO_TOLERANCE_FIGURE",
    "NO_WAVEFORM",
    "PARTIAL_ROUTING_RULE",
    "PDF_NOT_REQUESTED",
    "REPORTS_DIR",
    "REPORT_SUFFIXES",
    "REPORT_TITLES",
    "SLOT_BOARD",
    "SLOT_COPPER",
    "SLOT_PLACEMENT",
    "SLOT_THEORY",
    "SLOT_TOLERANCE",
    "SLOT_WAVEFORM",
    "STAGE_REPORTS",
    "SUBSTITUTES_HEADING",
    "VALUE_COLUMN_NOTE",
    "ReportFigures",
    "RunRecord",
    "StageDocument",
    "StageReportResult",
    "board_figures_of",
    "build_stage_document",
    "build_stage_report",
    "circuit_report",
    "copper_resistance_ohm",
    "final_report",
    "ipc2221_current_a",
    "net_routing_stats",
    "parts_report",
    "routing_params_of",
    "stage_figures",
    "strip_paths",
    "template_for",
    "template_of",
    "theory_figures_of",
    "theory_report",
    "tolerance_figure_of",
    "waveform_figures_of",
    "write_all_stage_reports",
    "write_stage_report",
]
