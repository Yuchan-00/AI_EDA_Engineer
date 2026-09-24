"""Read-only HTML report of a project and the loopback server that serves it.

Invariant: the report is a *view*. It computes no status (every one shown is
copied from ``ir.validation`` or from the recorded ``pipeline.json``),
registers no artifact, never saves the IR, writes nothing but the requested
HTML file, and opens no socket except the loopback listener of ``serve``.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.report.data import ReportData, build_report_data, load_ir_file
from ai_eda.report.html import esc, render_html
from ai_eda.report.pipeline_log import (
    PIPELINE_FILE,
    LoadedPipelineRecord,
    PipelineRecord,
    PipelineRecordError,
    load_pipeline_record,
    save_pipeline_record,
)
from ai_eda.report.server import LOOPBACK, ReportServer, serve


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
    "LOOPBACK",
    "PIPELINE_FILE",
    "LoadedPipelineRecord",
    "PipelineRecord",
    "PipelineRecordError",
    "ReportData",
    "ReportServer",
    "build_report_data",
    "esc",
    "load_ir_file",
    "load_pipeline_record",
    "render_html",
    "render_report_file",
    "save_pipeline_record",
    "serve",
]
