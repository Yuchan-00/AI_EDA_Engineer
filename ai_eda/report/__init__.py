"""Read-only HTML report of a project and the loopback server that serves it.

Invariant: the report is a *view*. It computes no status (every one shown is
copied from ``ir.validation`` or from the recorded ``pipeline.json``),
registers no artifact, never saves the IR, writes nothing but the requested
HTML file, and opens no socket except the loopback listener of ``serve``.
The Korean stage reports (:mod:`ai_eda.report.stages`) are views under the
same rule, written under ``<workdir>/reports/`` as Markdown, as
self-contained HTML with deterministic inline SVG figures
(:mod:`ai_eda.report.figures`, rendered by :mod:`ai_eda.report.pdf`) and,
when a headless Chromium / Chrome / Edge is found on the machine
(:func:`~ai_eda.report.pdf.find_browser`), as PDF - a derived document
whose bytes carry the browser's own creation date; none of the three is an
artifact or hashed.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.report.data import ReportData, build_report_data, load_ir_file
from ai_eda.report.figures import Figure
from ai_eda.report.html import esc, render_html
from ai_eda.report.pdf import BROWSER_ENV, PdfResult, browser_version, find_browser, html_to_pdf, markdown_to_html
from ai_eda.report.pipeline_log import (
    PIPELINE_FILE,
    LoadedPipelineRecord,
    PipelineRecord,
    PipelineRecordError,
    load_pipeline_record,
    save_pipeline_record,
)
from ai_eda.report.server import LOOPBACK, ReportServer, serve
from ai_eda.report.stages import (
    REPORTS_DIR,
    REPORT_SUFFIXES,
    STAGE_REPORTS,
    ReportFigures,
    StageDocument,
    StageReportResult,
    build_stage_document,
    build_stage_report,
    circuit_report,
    final_report,
    parts_report,
    stage_figures,
    template_of,
    theory_report,
    write_all_stage_reports,
    write_stage_report,
)


def render_report_file(ir_path: Path, output: Path | None = None) -> Path:
    """Write the report of ``ir_path`` to ``output`` (default ``<workdir>/report.html``) and return the path written."""
    from ai_eda.cli import project_workdir

    ir_path = Path(ir_path)
    ir, ir_sha = load_ir_file(ir_path)
    workdir = project_workdir(ir, ir_path)
    out = Path(output) if output is not None else workdir / "report.html"
    out.write_text(render_html(build_report_data(ir, ir_path, workdir, ir_sha=ir_sha)), encoding="utf-8", newline="\n")
    return out


__all__ = [
    "BROWSER_ENV",
    "LOOPBACK",
    "PIPELINE_FILE",
    "REPORTS_DIR",
    "REPORT_SUFFIXES",
    "STAGE_REPORTS",
    "Figure",
    "LoadedPipelineRecord",
    "PdfResult",
    "PipelineRecord",
    "PipelineRecordError",
    "ReportData",
    "ReportFigures",
    "ReportServer",
    "StageDocument",
    "StageReportResult",
    "browser_version",
    "build_report_data",
    "build_stage_document",
    "build_stage_report",
    "circuit_report",
    "esc",
    "final_report",
    "find_browser",
    "html_to_pdf",
    "load_ir_file",
    "load_pipeline_record",
    "markdown_to_html",
    "parts_report",
    "render_html",
    "render_report_file",
    "save_pipeline_record",
    "serve",
    "stage_figures",
    "template_of",
    "theory_report",
    "write_all_stage_reports",
    "write_stage_report",
]
