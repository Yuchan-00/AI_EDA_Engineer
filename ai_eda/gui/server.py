"""The GUI's loopback HTTP server: the page, the project / run JSON API, the previews and the downloads.

Invariant: the listener binds :data:`~ai_eda.report.server.LOOPBACK`
(127.0.0.1) and nothing else - there is no host option, by design, exactly
like ``ai-eda serve``. Every request must carry a ``Host`` header that names
this machine (:func:`~ai_eda.report.server.host_allowed`; anything else is
421), and every POST must also come from the page itself: an ``Origin``
header that is absent or this server's own origin
(``http://127.0.0.1:<port>`` / ``http://localhost:<port>`` /
``http://[::1]:<port>``; anything else is 403) and a ``Content-Type`` of
``application/json`` (415 otherwise), so a page on another origin can
neither start a run nor create a project. Every response carries
``Cache-Control: no-store``, ``X-Content-Type-Options: nosniff`` and
``X-Frame-Options: SAMEORIGIN``; the app page has the CSP
:data:`APP_CSP`, the documents the page shows in a sandboxed iframe
(``report.html``, the stage-report HTML) :data:`DOCUMENT_CSP`, every other
response :data:`DATA_CSP` - except the stage-report PDFs, which are served
inline with no CSP so the browser's own PDF viewer can render them (they
are files the pipeline wrote; ``nosniff``, ``no-store`` and
``SAMEORIGIN`` still apply).

The CSP choice: the page's script and stylesheet are *files*
(``/static/app.js``, ``/static/app.css``, the strings of
:mod:`ai_eda.gui.page`) and :data:`APP_CSP` is ``default-src 'self'`` with
no ``'unsafe-inline'`` and no ``'unsafe-eval'`` for either scripts or
styles. The page is written to need nothing more: no inline ``<script>``,
no inline event handler (every listener is added by ``app.js``), no
``style`` attribute or ``<style>`` element (layer toggles and states are
classes of ``app.css``), and the preview SVGs it inlines carry presentation
attributes only. ``img-src 'self' data:`` and ``frame-src 'self'`` are the
two additions: the sandboxed iframe loads a same-origin document. The 3D
tab needs nothing more: its WebGL viewer is code in ``app.js`` that fetches
the same-origin ``/preview/<name>/board.glb`` (``model/gltf-binary``;
``connect-src`` falls back to ``'self'``) and compiles its two shaders
through WebGL (no ``eval``); KiCad's render PNGs are same-origin images.

What it serves is a view: every status in the JSON is copied from
``ir.validation`` / ``pipeline.json`` through
:func:`~ai_eda.report.data.build_report_data` and :mod:`ai_eda.gui.projects`,
every preview comes from :mod:`ai_eda.gui.preview`, and nothing here
computes a status, registers an artifact, hashes or edits ``ir.json``. The
only writes are a new project folder (``POST /api/projects``, the code path
of ``ai-eda new``) and the run logs of the ``ai-eda`` subprocesses
:class:`~ai_eda.gui.runs.RunManager` starts (``POST .../run`` /
``.../review`` / ``.../stage-reports``) - their approvals are the CLI's own,
since the form becomes the CLI's flags. There is no delete or rename
endpoint (deleting would be a ``SYSTEM_DELETE``; not in the GUI).

Files are served only from a project's workdir through
:func:`~ai_eda.gui.preview.confined_file` (no ``..``, no symbolic link, no
directory listing); an unknown path is 404, a preview whose input does not
exist yet is 404 with the Korean sentence the page shows, a render failure
is 503 - every 4xx / 5xx body is one plain-text line with ``&`` / ``<`` /
``>`` escaped, never a traceback. The process opens no outbound connection:
no :class:`~ai_eda.security.ExternalAction` is involved (an ``--online`` run
is the subprocess's, under the user's checkbox, recorded by its own gate).
"""

from __future__ import annotations

import html
import json
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from ai_eda.agents.keys import CONFIRM_DESIGN_KEY, CONTROL_KEYS, REQUIREMENT_DECISION_KEYS
from ai_eda.gui.labels import label_map
from ai_eda.gui.preview import (
    BOM_FILE,
    CPL_FILE,
    PreviewMissing,
    artifact_download,
    artifact_file,
    artifact_rows,
    artifact_state,
    board_available,
    board_svg,
    bom_table,
    confined_file,
    cpl_table,
    kicad_3d_files,
    kicad_render_file,
    missing_sentence,
    model3d_glb,
    model3d_scene,
    model3d_summary,
    model3d_svg,
    project_schematic_svg,
    project_zip,
    report_files,
    simulation_summary,
    stage_report_file,
    waveform_svg,
    NO_BOM,
    NO_CPL,
    NO_KICAD_RENDER,
    NO_REPORT_FILE,
)
from ai_eda.gui.projects import ProjectError, ProjectInfo, ProjectNotFoundError, ProjectsRoot
from ai_eda.gui.runs import RUN_KINDS, RunConflictError, RunManager, RunNotFoundError, RunRequestError, RunStartError, allowed_options
from ai_eda.ir import ArtifactKind, CircuitIR
from ai_eda.llm.router import TaskKind, default_model, parse_model_spec
from ai_eda.report.server import ALLOWED_HOSTS, LOOPBACK, host_allowed
from ai_eda.tools.kicad.library import KicadLibrary

#: the default port of ``ai-eda gui`` (``serve`` uses 8765)
DEFAULT_PORT = 8766
#: the largest POST body accepted (a form is a few hundred bytes)
MAX_BODY_BYTES = 1 << 20
#: how long a ``describe_providers`` answer is reused (it runs ``claude --version`` / ``auth status``)
PROVIDERS_TTL_S = 30.0
#: the app page: its own script / style only, data: images, same-origin frames
APP_CSP = "default-src 'self'; img-src 'self' data:; frame-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'"
#: a document shown in the page's sandboxed iframe (report.html, a stage report's HTML): inline style and data: images only
DOCUMENT_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'self'"
#: every other response (JSON, SVG, text, downloads) but the inline stage-report PDFs, which carry no CSP (the browser's viewer renders them)
DATA_CSP = "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'self'"
TEXT = "text/plain; charset=utf-8"
JSON_TYPE = "application/json; charset=utf-8"
SVG_TYPE = "image/svg+xml; charset=utf-8"
HTML_TYPE = "text/html; charset=utf-8"
#: the built-in 3D preview the page's WebGL viewer loads (glTF 2.0 binary, the registered media type)
GLB_TYPE = "model/gltf-binary"
PNG_TYPE = "image/png"
#: the longest error line sent (the rest is cut; the log or the page has the detail)
MAX_ERROR_LINE = 600


class HttpError(Exception):
    """An answer other than 2xx: ``code`` and the one line of its body."""

    def __init__(self, code: int, line: str) -> None:
        super().__init__(line)
        self.code = code
        self.line = line


@dataclass
class Response:
    code: int
    body: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)
    csp: str | None = DATA_CSP


def error_line(text: object) -> str:
    """One line of plain text: whitespace collapsed, ``&`` / ``<`` / ``>`` escaped, at most :data:`MAX_ERROR_LINE` characters."""
    line = " ".join(str(text).split()) or "error"
    if len(line) > MAX_ERROR_LINE:
        line = line[: MAX_ERROR_LINE - 1] + "…"
    return html.escape(line, quote=False)


def text_response(code: int, line: str) -> Response:
    return Response(code, (error_line(line) + "\n").encode("utf-8"), TEXT)


def json_response(value: Any, code: int = 200) -> Response:
    return Response(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), JSON_TYPE)


def content_disposition(kind: str, filename: str) -> str:
    """``attachment`` / ``inline`` with an ASCII fallback name and the RFC 5987 UTF-8 name (header values are Latin-1)."""
    fallback = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in filename) or "download"
    return f"{kind}; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON number")


# --------------------------------------------------------------------------- the application


class _Providers:
    """``ai_eda.llm.providers.describe_providers()`` rows, reused for ``ttl`` seconds (looked up at call time, so a test can replace it)."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] | None = None
        self._at = 0.0

    def rows(self) -> list[dict[str, Any]]:
        from ai_eda.llm import providers

        with self._lock:
            now = time.monotonic()
            if self._rows is None or self.ttl <= 0 or now - self._at > self.ttl:
                self._rows = [r.model_dump(mode="json") for r in providers.describe_providers()]
                self._at = now
            return [dict(r) for r in self._rows]


@dataclass
class _Loaded:
    """One project as read for one request: its info, the IR parsed from ``ir_bytes`` and the sha of those bytes."""

    info: ProjectInfo
    workdir: Path
    ir: CircuitIR
    ir_bytes: bytes
    ir_sha: str


class GuiApp:
    """The routes, independent of the socket: :meth:`get` / :meth:`post` return a :class:`Response` or raise :class:`HttpError`."""

    def __init__(
        self,
        root: str | Path,
        *,
        runs: RunManager | None = None,
        library: KicadLibrary | None = None,
        providers_ttl: float = PROVIDERS_TTL_S,
    ) -> None:
        self.projects = ProjectsRoot(root)
        self.runs = runs if runs is not None else RunManager()
        self._library = library
        self._library_lock = threading.Lock()
        self._providers = _Providers(providers_ttl)
        #: the port the page is served on (the Origin check); set by :class:`GuiServer`
        self.port = 0

    # ------------------------------------------------------------------ shared

    def own_origins(self) -> frozenset[str]:
        """The page's own origins: ``http://<loopback name>:<port>`` (plus the port-less form on port 80)."""
        names = sorted(ALLOWED_HOSTS)
        origins = {f"http://{n}:{self.port}" for n in names}
        if self.port == 80:
            origins |= {f"http://{n}" for n in names}
        return frozenset(origins)

    def library(self) -> KicadLibrary:
        with self._library_lock:
            if self._library is None:
                self._library = KicadLibrary()
            return self._library

    def _project(self, name: str) -> ProjectInfo:
        try:
            return self.projects.get(name)
        except ProjectNotFoundError as e:
            raise HttpError(404, str(e)) from e

    def _load(self, name: str) -> _Loaded:
        """The project with its IR read once (bytes, parse, sha); 503 for a project listed with an error."""
        from ai_eda.report.pipeline_log import sha256_of_bytes

        info = self._project(name)
        if info.error is not None or info.workdir is None:
            raise HttpError(503, f"project unavailable: {info.error or 'no workdir'}")
        try:
            raw = info.ir_path.read_bytes()
            ir = CircuitIR.loads(raw, source=str(info.ir_path))
        except Exception as e:  # noqa: BLE001 - the file changed since it was listed: one line, never a traceback
            raise HttpError(503, f"project unavailable: ir.json을 읽을 수 없습니다: {e}") from e
        return _Loaded(info, info.workdir, ir, raw, sha256_of_bytes(raw))

    # ------------------------------------------------------------------ GET

    def get(self, segments: list[str]) -> Response:
        from ai_eda.gui.page import APP_CSS, APP_HTML, APP_JS

        match segments:
            case []:
                return Response(200, APP_HTML.encode("utf-8"), HTML_TYPE, csp=APP_CSP)
            case ["static", "app.css"]:
                return Response(200, APP_CSS.encode("utf-8"), "text/css; charset=utf-8")
            case ["static", "app.js"]:
                return Response(200, APP_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            case ["api", "projects"]:
                return json_response({"root": str(self.projects.root), "projects": [p.model_dump(mode="json") for p in self.projects.list()]})
            case ["api", "projects", name]:
                return json_response(self.project_json(name))
            case ["api", "projects", name, "runs"]:
                info = self._runnable(name)
                active = self.runs.active_id(info.workdir)
                return json_response({"runs": [r.model_dump(mode="json") for r in self.runs.list_runs(info.workdir)], "active": active})
            case ["api", "projects", name, "runs", run_id]:
                info = self._runnable(name)
                try:
                    return json_response(self.runs.status(info.workdir, run_id).model_dump(mode="json"))
                except RunNotFoundError as e:
                    raise HttpError(404, str(e)) from e
            case ["preview", name, "schematic.svg"]:
                p = self._load(name)
                return self._svg(lambda: project_schematic_svg(p.ir, p.workdir), "schematic")
            case ["preview", name, "board.svg"]:
                p = self._load(name)
                return self._svg(lambda: self._board(p.ir), "board")
            case ["preview", name, "board.glb"]:
                p = self._load(name)
                return self._build(lambda: Response(200, model3d_glb(self._scene(p.ir)), GLB_TYPE), "3D preview")
            case ["preview", name, "board3d", file] if file.endswith(".svg"):
                p = self._load(name)
                return self._svg(lambda: model3d_svg(self._scene(p.ir), file[: -len(".svg")]), "3D preview")
            case ["preview", name, "model3d.json"]:
                p = self._load(name)
                return self._build(lambda: json_response(model3d_summary(self._scene(p.ir))), "3D preview")
            case ["preview", name, "kicad3d", file]:
                p = self._load(name)
                path = kicad_render_file(p.ir, p.workdir, file)
                if path is None:
                    raise HttpError(404, NO_KICAD_RENDER)
                return Response(200, path.read_bytes(), PNG_TYPE)
            case ["preview", name, "waveform", file] if file.endswith(".svg"):
                p = self._load(name)
                return self._svg(lambda: waveform_svg(p.workdir, file[: -len(".svg")]), "waveform")
            case ["preview", name, "report.html"]:
                return self.report_page(name)
            case ["preview", name, "reports", file]:
                return self.stage_report(name, file)
            case ["preview", name, "bom.json"]:
                p = self._load(name)
                notes = list(p.ir.artifacts[ArtifactKind.BOM].notes) if ArtifactKind.BOM in p.ir.artifacts else []
                return self._table(p, ArtifactKind.BOM, lambda: bom_table(p.workdir), NO_BOM, extra={"notes": notes})
            case ["preview", name, "cpl.json"]:
                p = self._load(name)
                return self._table(p, ArtifactKind.CPL, lambda: cpl_table(p.workdir), NO_CPL)
            case ["preview", name, "artifacts.json"]:
                p = self._load(name)

                def artifacts() -> Response:
                    rows = artifact_rows(p.ir, p.workdir)
                    return json_response({"design_hash": p.ir.content_hash(), "artifacts": rows, "labels": label_map(rows)})

                return self._build(artifacts, "artifacts")
            case ["preview", name, "ir.json"]:
                p = self._load(name)
                return Response(200, p.ir_bytes, JSON_TYPE)
            case ["files", zip_name] if zip_name.endswith(".zip") and len(zip_name) > len(".zip"):
                return self.zip(zip_name[: -len(".zip")])
            case ["files", name, *rest] if rest:
                return self.download(name, rest)
        raise HttpError(404, "not found")

    def _runnable(self, name: str) -> ProjectInfo:
        info = self._project(name)
        if info.error is not None or info.workdir is None:
            raise HttpError(503, f"project unavailable: {info.error or 'no workdir'}")
        return info

    def _build(self, build: Callable[[], Response], what: str) -> Response:
        try:
            return build()
        except HttpError:
            raise
        except PreviewMissing as e:
            raise HttpError(404, str(e)) from e
        except Exception as e:  # noqa: BLE001 - one escaped line, never a traceback
            raise HttpError(503, f"{what} unavailable: {e}") from e

    def _svg(self, draw: Callable[[], str], what: str) -> Response:
        return self._build(lambda: Response(200, draw().encode("utf-8"), SVG_TYPE), what)

    def _table(
        self, p: _Loaded, kind: ArtifactKind, read: Callable[[], dict[str, Any] | None], missing: str, extra: Mapping[str, Any] | None = None,
    ) -> Response:
        """A CSV preview with its artifact's facts (``state``: the :func:`~ai_eda.gui.preview.artifact_state` row, ``None`` when not registered)."""

        def build() -> Response:
            table = read()
            if table is None:
                raise PreviewMissing(missing_sentence(p.ir, p.workdir, kind, missing))
            state = artifact_state(artifact_rows(p.ir, p.workdir), kind)
            return json_response({**table, **(extra or {}), "state": state, "labels": label_map(state)})

        return self._build(build, "table")

    def _board(self, ir: CircuitIR) -> str:
        library = self.library()
        with self._library_lock:  # the library's parse caches are not shared across threads while one fills them
            return board_svg(ir, library)

    def _scene(self, ir: CircuitIR):
        """The 3D preview scene of ``ir`` (:func:`~ai_eda.gui.preview.model3d_scene`), built under the library lock like the board figure."""
        library = self.library()
        with self._library_lock:
            return model3d_scene(ir, library)

    def project_json(self, name: str) -> dict[str, Any]:
        """Everything the page shows about one project (see the module docstring of :mod:`ai_eda.gui`); statuses copied, none computed."""
        from ai_eda.report import pdf as report_pdf
        from ai_eda.report.data import build_report_data

        p = self._load(name)
        info, ir, workdir = p.info, p.ir, p.workdir

        def build() -> dict[str, Any]:
            report = build_report_data(ir, info.ir_path, workdir, ir_sha=p.ir_sha).model_dump(mode="json")
            last = info.last_run
            files = report_files(workdir)
            simulation = simulation_summary(ir, workdir)
            active = self.runs.active_id(workdir) if info.workdir_exists else None
            rows = artifact_rows(ir, workdir)
            providers = self._providers.rows()
            return {
                "info": info.model_dump(mode="json"),
                "report": report,
                # the Korean wording of every hash / file / run label the report and the artifact rows carry (ai_eda.gui.labels)
                "labels": label_map(report, rows),
                "questions": {
                    "required": [q.model_dump(mode="json") for q in last.open_questions] if last is not None else [],
                    "optional": [q.model_dump(mode="json") for q in last.optional_questions] if last is not None else [],
                },
                # the circuit agent's rule, from its own key sets: confirm_design=yes is ignored when any answer outside the control
                # keys, or a requirement decision, goes with it in the same run (ai_eda.agents.circuit)
                "answers_rule": {
                    "confirm_key": CONFIRM_DESIGN_KEY,
                    "control_keys": sorted(CONTROL_KEYS),
                    "decision_keys": sorted(REQUIREMENT_DECISION_KEYS),
                },
                "llm": {
                    "providers": providers,
                    "pinned_model": ir.requirements.llm_model_spec,
                    # the spec a run uses when the model field is empty and this provider comes first (the CLI's own resolution)
                    "default_specs": {r["name"]: parse_model_spec(default_model(r["name"]), r["name"]).spec for r in providers},
                    # the TASK names ``--llm-task-model TASK=SPEC`` takes (the router's own enum)
                    "tasks": [str(t) for t in TaskKind],
                },
                "browser_found": report_pdf.find_browser() is not None,
                "previews": {
                    "schematic": artifact_file(ir, workdir, ArtifactKind.SCHEMATIC) is not None,
                    "schematic_file": artifact_download(ir, workdir, ArtifactKind.SCHEMATIC),
                    # each previewed file's own facts, copied from its artifact row: generated from the current design hash? on disk?
                    "schematic_state": artifact_state(rows, ArtifactKind.SCHEMATIC),
                    "board": board_available(ir),
                    "board_file": artifact_download(ir, workdir, ArtifactKind.PCB),
                    "board_state": artifact_state(rows, ArtifactKind.PCB),
                    # the 3D tab: drawn from the current IR like the board (same input); the compiled preview GLB's own facts beside it
                    "model3d": board_available(ir),
                    "model3d_file": artifact_download(ir, workdir, ArtifactKind.MODEL_3D),
                    "model3d_state": artifact_state(rows, ArtifactKind.MODEL_3D),
                    # KiCad's own STEP / GLB / render exports (real part shapes; only where kicad-cli ran), with their facts
                    "kicad_3d": kicad_3d_files(ir, workdir, rows),
                    "bom_state": artifact_state(rows, ArtifactKind.BOM),
                    "cpl_state": artifact_state(rows, ArtifactKind.CPL),
                    "waveforms": [w["id"] for w in simulation["waveforms"]],
                    "reports": files["stage_reports"],
                    "report_html": True,
                    "report_html_file": files["report_html"],
                    "bom": confined_file(workdir, (BOM_FILE,)) is not None,
                    "cpl": confined_file(workdir, (CPL_FILE,)) is not None,
                },
                "simulation": simulation,
                "run_options": {kind: list(allowed_options(kind)) for kind in RUN_KINDS},
                "active_run": active,
                "runs": [r.model_dump(mode="json") for r in self.runs.list_runs(workdir)] if info.workdir_exists else [],
            }

        try:
            return build()
        except HttpError:
            raise
        except Exception as e:  # noqa: BLE001
            raise HttpError(503, f"project unavailable: {e}") from e

    def report_page(self, name: str) -> Response:
        """The English report of the project rendered now (``ai-eda serve``'s page), for the page's sandboxed iframe."""
        from ai_eda.report.data import build_report_data
        from ai_eda.report.html import render_html

        p = self._load(name)

        def build() -> Response:
            page = render_html(build_report_data(p.ir, p.info.ir_path, p.workdir, ir_sha=p.ir_sha))
            return Response(200, page.encode("utf-8"), HTML_TYPE, csp=DOCUMENT_CSP)

        return self._build(build, "report")

    def stage_report(self, name: str, file: str) -> Response:
        """One stage-report file: ``.html`` for the iframe (its own CSP), ``.pdf`` inline for the browser's viewer, ``.md`` as text."""
        info = self._runnable(name)
        path = stage_report_file(info.workdir, file)
        if path is None:
            raise HttpError(404, NO_REPORT_FILE)
        suffix = path.suffix.lower()
        data = path.read_bytes()
        if suffix == ".html":
            return Response(200, data, HTML_TYPE, csp=DOCUMENT_CSP)
        if suffix == ".pdf":
            return Response(200, data, "application/pdf", headers={"Content-Disposition": content_disposition("inline", path.name)}, csp=None)
        return Response(200, data, "text/plain; charset=utf-8", headers={"Content-Disposition": content_disposition("inline", path.name)})

    def download(self, name: str, parts: list[str]) -> Response:
        """``<workdir>/<parts...>`` as an attachment, confined to the workdir (:func:`~ai_eda.gui.preview.confined_file`)."""
        info = self._runnable(name)
        path = confined_file(info.workdir, parts)
        if path is None:
            raise HttpError(404, "not found")
        return Response(200, path.read_bytes(), "application/octet-stream", headers={"Content-Disposition": content_disposition("attachment", path.name)})

    def zip(self, name: str) -> Response:
        """The project as a zip built in memory (:func:`~ai_eda.gui.preview.project_zip`)."""
        p = self._load(name)
        return self._build(
            lambda: Response(200, project_zip(p.ir, p.ir_bytes, p.workdir, p.info.name), "application/zip",
                             headers={"Content-Disposition": content_disposition("attachment", f"{p.info.name}.zip")}),
            "zip",
        )

    # ------------------------------------------------------------------ POST

    def post(self, segments: list[str], body: Mapping[str, Any]) -> Response:
        match segments:
            case ["api", "projects"]:
                return self.create(body)
            case ["api", "projects", name, ("run" | "review" | "stage-reports") as kind]:
                return self.start(name, kind, body)
        raise HttpError(404, "not found")

    def create(self, body: Mapping[str, Any]) -> Response:
        unknown = sorted(str(k) for k in body if k not in ("name", "request"))
        if unknown:
            raise HttpError(400, f"알 수 없는 필드: {', '.join(unknown)} (name, request만 받습니다)")
        try:
            info = self.projects.create(body.get("name"), body.get("request", ""))  # type: ignore[arg-type]
        except ProjectError as e:
            raise HttpError(400, str(e)) from e
        return json_response({"name": info.name, "ir_path": str(info.ir_path), "workdir": str(info.workdir) if info.workdir else None}, 201)

    def start(self, name: str, kind: str, body: Mapping[str, Any]) -> Response:
        unknown = sorted(str(k) for k in body if k not in ("answers", "options"))
        if unknown:
            raise HttpError(400, f"알 수 없는 필드: {', '.join(unknown)} (answers, options만 받습니다)")
        info = self._project(name)
        try:
            handle = self.runs.start_project(kind, info, body.get("answers"), body.get("options"))
        except RunRequestError as e:
            raise HttpError(400, str(e)) from e
        except RunConflictError as e:
            raise HttpError(409, str(e)) from e
        except RunStartError as e:
            raise HttpError(503, str(e)) from e
        return json_response({"run_id": handle.id, "kind": handle.kind, "started": handle.started}, 202)


# --------------------------------------------------------------------------- the socket


def split_path(target: str) -> list[str] | None:
    """The percent-decoded segments of a request target's path (``/`` -> ``[]``); ``None`` for an empty segment, bad UTF-8, or a decoded ``/`` / NUL."""
    path = urlsplit(target).path
    if path == "/":
        return []
    if not path.startswith("/"):
        return None
    out: list[str] = []
    for raw in path[1:].split("/"):
        if not raw:
            return None
        try:
            seg = unquote(raw, encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
        if "/" in seg or "\x00" in seg:
            return None
        out.append(seg)
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "ai-eda-gui"
    sys_version = ""  # no Python version in the Server header
    protocol_version = "HTTP/1.1"

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - BaseHTTPRequestHandler's signature
        return  # quiet: the page and the run logs are the output

    @property
    def app(self) -> GuiApp:
        return self.server.app  # type: ignore[attr-defined]

    def _reply(self, response: Response) -> None:
        self.send_response(response.code)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        if response.csp:
            self.send_header("Content-Security-Policy", response.csp)
        for key, value in response.headers.items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if response.body and self.command != "HEAD":
            self.wfile.write(response.body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """The stdlib's own errors (a bad request line, an unsupported method) as one plain-text line too, never its HTML page."""
        self.close_connection = True
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = "error"
        self._reply(text_response(code, message or phrase))

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if not host_allowed(self.headers.get("Host")):
            self._reply(text_response(421, "misdirected request: this server answers only for 127.0.0.1, localhost or [::1]"))
            return
        segments = split_path(self.path)
        try:
            if segments is None:
                raise HttpError(404, "not found")
            response = self.app.get(segments)
        except HttpError as e:
            response = text_response(e.code, e.line)
        except Exception as e:  # noqa: BLE001 - one escaped line, never a traceback
            response = text_response(503, f"unavailable: {e}")
        self._reply(response)

    def do_POST(self) -> None:  # noqa: N802
        self.close_connection = True  # a refused POST leaves its body unread: never reuse the connection
        if not host_allowed(self.headers.get("Host")):
            self._reply(text_response(421, "misdirected request: this server answers only for 127.0.0.1, localhost or [::1]"))
            return
        try:
            response = self._post()
        except HttpError as e:
            response = text_response(e.code, e.line)
        except Exception as e:  # noqa: BLE001
            response = text_response(503, f"unavailable: {e}")
        self._reply(response)

    def _post(self) -> Response:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.app.own_origins():
            raise HttpError(403, "forbidden: a request from another origin cannot create a project or start a run")
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HttpError(415, "unsupported media type: POST bodies are application/json only")
        segments = split_path(self.path)
        if segments is None:
            raise HttpError(404, "not found")
        if self.headers.get("Transfer-Encoding"):
            raise HttpError(411, "length required: send the body with Content-Length")
        length_header = self.headers.get("Content-Length")
        if length_header is None or not length_header.strip().isdigit():
            raise HttpError(411, "length required: send the body with Content-Length")
        length = int(length_header)
        if length > MAX_BODY_BYTES:
            raise HttpError(413, f"request body too large (limit {MAX_BODY_BYTES} bytes)")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise HttpError(400, "the body ended before Content-Length")
        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError) as e:
            raise HttpError(400, f"the body is not JSON: {e}") from e
        if not isinstance(body, dict):
            raise HttpError(400, "the body must be a JSON object")
        return self.app.post(segments, body)


class _LoopbackServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` whose handler threads never keep the process alive; no ``SO_REUSEADDR`` on Windows, where it would let a second listener take the port."""

    daemon_threads = True
    allow_reuse_address = os.name != "nt"


class GuiServer:
    """``ThreadingHTTPServer`` on 127.0.0.1 serving :class:`GuiApp` for the projects under ``root`` (``port`` 0 = ephemeral).

    ``runs`` (a :class:`~ai_eda.gui.runs.RunManager`), ``library`` (the
    KiCad library the board preview reads; default discovered on first use)
    and ``providers_ttl`` exist for tests.
    """

    def __init__(
        self,
        root: str | Path,
        port: int = 0,
        *,
        runs: RunManager | None = None,
        library: KicadLibrary | None = None,
        providers_ttl: float = PROVIDERS_TTL_S,
    ) -> None:
        self.app = GuiApp(root, runs=runs, library=library, providers_ttl=providers_ttl)
        self._server = _LoopbackServer((LOOPBACK, port), _Handler)
        self._server.app = self.app  # type: ignore[attr-defined]
        self.app.port = self.port

    @property
    def root(self) -> Path:
        return self.app.projects.root

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}/"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        """Stop a :meth:`serve_forever` running in *another* thread, then close the socket."""
        self._server.shutdown()
        self._server.server_close()

    def close(self) -> None:
        """Close the listening socket (after :meth:`serve_forever` returned in this thread, or before it ever ran)."""
        self._server.server_close()


def serve_gui(root: str | Path, port: int = DEFAULT_PORT, open_browser: bool = False) -> int:
    """Run the GUI until Ctrl-C (exit 0); a bind failure (a taken port, a port outside 0-65535) prints the error and returns 2. ``open_browser`` opens the page (a local action)."""
    try:
        server = GuiServer(root, port)
    except (OSError, OverflowError) as e:  # bind() raises OverflowError for a port outside 0-65535
        print(f"could not listen on {LOOPBACK}:{port}: {e}", file=sys.stderr)
        return 2
    print(
        f"serving the GUI for the projects under {server.root} at {server.url} "
        "(127.0.0.1 only; runs are `ai-eda` subprocesses with the CLI's own approvals; Ctrl-C to stop)",
        flush=True,
    )
    try:
        if open_browser:
            webbrowser.open(server.url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()  # not shutdown(): that waits for a serve_forever loop in another thread
    return 0


__all__ = [
    "APP_CSP",
    "DATA_CSP",
    "DEFAULT_PORT",
    "DOCUMENT_CSP",
    "GuiApp",
    "GuiServer",
    "HttpError",
    "MAX_BODY_BYTES",
    "Response",
    "serve_gui",
    "split_path",
]
