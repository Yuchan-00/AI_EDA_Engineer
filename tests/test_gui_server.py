"""The GUI's loopback server (``ai_eda/gui/server.py``) and its preview builders (``ai_eda/gui/preview.py``).

The project the previews are checked on is compiled from the synthetic
fixtures: the divider template of ``tests/test_circuit_templates.py`` run to
RELEASE offline on its synthetic KiCad library (schematic, routed board, BOM,
CPL), then ir.json, pipeline.json, the stage reports and report.html written
the way ``ai-eda run`` writes them, plus a fixture ``spice/results.json``
(tran / dc / ac / op, a failed and an unverifiable analysis), a ``sources/``
archive and a ``gui/`` run log that the zip must leave out. Every request
goes through a real socket on 127.0.0.1; runs are real ``ai-eda``
subprocesses (or a sleeping fake passed through ``RunManager.python_args``).
The ``claude`` provider is the fake of ``tests/fake_claude_cli.py`` - the
real binary is never run (``tests/conftest.py``).
"""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

import pytest

import ai_eda.gui.server as server_module
from ai_eda.agents.circuit import CONFIRM_DESIGN_KEY
from ai_eda.cli import main as cli_main
from ai_eda.cli import model_pin_refusal
from ai_eda.gui.projects import ProjectsRoot
from ai_eda.gui.preview import (
    NO_BOARD,
    NO_BOM,
    NO_CPL,
    NO_RESULTS,
    NO_SCHEMATIC,
    NO_WAVEFORM,
    RESULTS_CURRENT,
    RESULTS_NOT_CURRENT,
    ZIP_DATE_TIME,
    bom_table,
    confined_file,
    read_csv_table,
    waveform_plan,
    workdir_parts,
)
from ai_eda.gui.runs import EXIT_PREFIX, RunManager
from ai_eda.gui.server import APP_CSP, DATA_CSP, DOCUMENT_CSP, GuiServer, content_disposition, serve_gui, split_path
from ai_eda.ir import CircuitIR
from ai_eda.report import render_report_file, save_pipeline_record, write_all_stage_reports
from ai_eda.report.data import build_report_data, load_ir_file
from ai_eda.report.pipeline_log import sha256_of_file
from ai_eda.report.stages import STAGE_REPORTS
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import STAGE_ORDER, Stage
from tests.fake_claude_cli import DEFAULT_MODEL, FakeClaudeCli, success_structured
from ai_eda.design.templates import TEMPLATES
from tests.test_circuit_templates import ASTABLE, DIVIDER, _ir, _present, _run, template_library

SVG_NS = "{http://www.w3.org/2000/svg}"
DATA = Path(__file__).parent / "data" / "fake_llm_requirements.json"
LLM_REQUEST = json.loads(DATA.read_text(encoding="utf-8"))["request"]
EXTRACTION = json.loads(DATA.read_text(encoding="utf-8"))["responses"][0]["structured"]
CLAUDE_SPEC = f"claude:{DEFAULT_MODEL}"
#: a real CLI run of a small project finishes in about a second; generous for a slow CI machine
RUN_TIMEOUT = 120.0
SLEEPER = [sys.executable, "-c", "import time; time.sleep(2)"]
THEORY_MD = STAGE_REPORTS[Stage.ARCHITECTURE]
THEORY_HTML = THEORY_MD.removesuffix(".md") + ".html"
THEORY_PDF = THEORY_MD.removesuffix(".md") + ".pdf"
SECRET = "sk-or-v1-GUI-TEST-SECRET-0123456789"

needs_dll = pytest.mark.skipif(not NgspiceShared().available(), reason="ngspice shared library not found")


# --------------------------------------------------------------------------- the fixture project


def _series(n: int, f) -> list[float]:
    return [float(f(i)) for i in range(n)]


def fixture_results() -> dict:
    """A ``spice/results.json`` in the SPICE stage's layout (format 2): op, a 5-voltage + 1-current tran, a dc sweep, an ac with the complex parts,
    a failed tran, a tran the environment could not run and an analysis whose id a URL cannot carry."""
    n = 40
    t = _series(n, lambda i: i * 1e-5)

    def tran(vectors: dict[str, list[float]], succeeded: bool = True, unverifiable: str | None = None) -> dict:
        types = {"time": "time", **{k: ("current" if "#branch" in k else "voltage") for k in vectors}}
        return {"kind": "tran", "command": "tran 10u 390u", "result": {
            "succeeded": succeeded, "unverifiable": unverifiable, "scale": "time", "n_points": n, "command": "tran 10u 390u",
            "vectors": {"time": t, **vectors}, "vector_types": types}}

    wave = {
        "vin": _series(n, lambda i: 12.0), "vout": _series(n, lambda i: 6.0 * (1 - 0.9 ** i)), "n1": _series(n, lambda i: i / 10),
        "n2": _series(n, lambda i: 1.0), "n3": _series(n, lambda i: 2.0), "vvin#branch": _series(n, lambda i: -6e-4),
    }
    freq = _series(n, lambda i: 10 ** (i / 8))
    return {
        "format": "2", "engine": "ngspice-shared", "engine_version": "ngspice-test", "conditions": {"temperature_c": 27.0},
        "netlist_hash": "sha256:" + "0" * 64,
        "analyses": {
            "op": {"kind": "op", "command": "op", "result": {"succeeded": True, "unverifiable": None, "scale": None, "n_points": 1,
                                                             "vectors": {"vin": [12.0], "vout": [6.0], "vvin#branch": [-6e-4]}}},
            "tran": tran(wave),
            "dc": {"kind": "dc", "command": "dc vvin 0 12 1", "result": {
                "succeeded": True, "unverifiable": None, "scale": "v-sweep", "n_points": 13,
                "vectors": {"v-sweep": _series(13, float), "vout": _series(13, lambda i: i / 2)}, "vector_types": {"v-sweep": "voltage", "vout": "voltage"}}},
            "ac": {"kind": "ac", "command": "ac dec 8 1 1e5", "result": {
                "succeeded": True, "unverifiable": None, "scale": "frequency", "n_points": n,
                "vectors": {"frequency": freq, "vout": _series(n, lambda i: 1 / (1 + i)), "vout.phase_deg": _series(n, lambda i: -i),
                            "vout.real": _series(n, lambda i: 0.5), "vout.imag": _series(n, lambda i: -0.5)}}},
            "tran_failed": tran({"vout": _series(n, lambda i: 0.0)}, succeeded=False),
            "tran_env": tran({"vout": _series(n, lambda i: 0.0)}, unverifiable="the engine was not available"),
            "bad id!": tran({"vout": _series(n, lambda i: 0.0)}),
        },
    }


#: the charts fixture_results() plans, in order
FIXTURE_WAVEFORMS = ["tran", "tran_2", "tran_i", "dc", "ac"]


def build_project(root: Path, name: str = "divider", answers: dict[str, str] | None = None, *, spice: bool = False) -> tuple[CircuitIR, KicadLibrary, Path]:
    """Compile a template design to RELEASE under ``<root>/<name>`` and write what ``ai-eda run`` writes beside it."""
    lib = template_library(root / "kicad")
    workdir = root / name
    ir = _ir(workdir, name)
    _present(ir, workdir, lib, answers or DIVIDER)
    state, _ = _run(ir, workdir, lib, {CONFIRM_DESIGN_KEY: "yes"}, None, spice=spice)
    assert not state.blocked and state.outcomes[-1].stage is Stage.RELEASE
    ir_path = ir.save(workdir / "ir.json")
    save_pipeline_record(state, ir, ir_path, workdir, results_before=0, ir_file_sha256=sha256_of_file(ir_path))
    write_all_stage_reports(ir, lib, state, workdir, pdf=False)
    render_report_file(ir_path)
    if not spice:
        (workdir / "reports" / THEORY_PDF).write_bytes(b"%PDF-1.4\n% a stand-in for the printed report\n")
        (workdir / "spice" / "tran").mkdir(parents=True)
        (workdir / "spice" / "results.json").write_text(json.dumps(fixture_results()), encoding="utf-8")
        (workdir / "spice" / "tran" / "divider.tran.raw").write_bytes(b"Title: fixture rawfile\n")
    (workdir / "sources").mkdir()
    (workdir / "sources" / "datasheet.pdf").write_bytes(b"%PDF archived document")
    (workdir / "gui" / "runs").mkdir(parents=True)
    (workdir / "gui" / "runs" / "001-run.log").write_text("$ ai-eda run\n# exit code 0\n", encoding="utf-8")
    return ir, lib, ir_path


class Running:
    """A :class:`GuiServer` serving in a thread."""

    def __init__(self, root: Path, **kw) -> None:
        kw.setdefault("providers_ttl", 0)
        self.server = GuiServer(root, 0, **kw)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.port

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory):
    """One compiled project served read-only by one server for the tests that change nothing."""
    root = tmp_path_factory.mktemp("gui_root")
    ir, lib, ir_path = build_project(root)
    running = Running(root, library=lib)
    yield running, ir, lib, ir_path
    running.stop()


@pytest.fixture
def fresh(tmp_path: Path):
    """A compiled project and a server of its own, for tests that change files."""
    ir, lib, ir_path = build_project(tmp_path / "root")
    running = Running(tmp_path / "root", library=lib)
    yield running, ir, lib, ir_path
    running.stop()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaudeCli:
    """The fake ``claude`` first on PATH (the run's child inherits PATH), no OpenRouter key."""
    f = FakeClaudeCli(tmp_path / "bin")
    f.install(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return f


# --------------------------------------------------------------------------- HTTP helpers


def request(port: int, method: str, path: str, body: object = None, headers: dict[str, str] | None = None, *, raw: bytes | None = None) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        hdrs = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def raw_post(port: int, path: str, length_header: str) -> tuple[str, bytes]:
    """A POST written by hand (headers only, no body sent): ``(status line, body)``."""
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        head = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n{length_header}\r\n"
        sock.sendall(head.encode("ascii"))
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    head_bytes, _sep, body = data.partition(b"\r\n\r\n")
    return head_bytes.decode("latin-1").splitlines()[0], body


def get(port: int, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    return request(port, "GET", path, headers=headers)


def get_json(port: int, path: str) -> dict:
    status, headers, body = get(port, path)
    assert status == 200, (path, body)
    assert headers["content-type"] == "application/json; charset=utf-8"
    return json.loads(body)


def post(port: int, path: str, body: object, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    return request(port, "POST", path, body, headers)


def one_line(body: bytes) -> str:
    """An error body: exactly one line of text, no markup."""
    text = body.decode("utf-8")
    assert text.endswith("\n") and text.count("\n") == 1 and "<" not in text and "Traceback" not in text, text
    return text.strip()


def wait_run(port: int, name: str, run_id: str) -> dict:
    deadline = time.monotonic() + RUN_TIMEOUT
    while True:
        status = get_json(port, f"/api/projects/{name}/runs/{run_id}")
        if not status["running"] and status["exit_code"] is not None:
            return status
        assert time.monotonic() < deadline, f"{run_id} still running: {status['log_tail'][-500:]}"
        time.sleep(0.1)


def start_run(port: int, name: str, kind: str = "run", headers: dict[str, str] | None = None, **body: object) -> str:
    status, _h, data = post(port, f"/api/projects/{name}/{kind}", body, headers)
    assert status == 202, data
    return json.loads(data)["run_id"]


def parse_svg(body: bytes) -> ET.Element:
    root = ET.fromstring(body.decode("utf-8"))
    assert root.tag == f"{SVG_NS}svg"
    return root


def classed(root: ET.Element, tag: str, cls: str) -> list[ET.Element]:
    return [e for e in root.iter(f"{SVG_NS}{tag}") if cls in (e.get("class") or "").split()]


# --------------------------------------------------------------------------- the page and the headers


def test_the_app_page_is_self_contained_and_carries_its_headers(shared):
    running, *_ = shared
    status, headers, body = get(running.port, "/")
    assert status == 200 and headers["content-type"] == "text/html; charset=utf-8"
    assert headers["content-security-policy"] == APP_CSP
    for directive in ("default-src 'self'", "img-src 'self' data:", "frame-src 'self'"):
        assert directive in APP_CSP
    assert headers["cache-control"] == "no-store" and headers["x-content-type-options"] == "nosniff" and headers["x-frame-options"] == "SAMEORIGIN"
    assert headers["server"] == "ai-eda-gui" and "Python" not in headers["server"]
    page = body.decode("utf-8")
    assert '<html lang="ko">' in page
    assert not re.search(r"<script(?![^>]*\bsrc=)", page), "no inline script: the CSP forbids it"
    assert "/static/app.css" in page and "/static/app.js" in page
    for path, ctype in (("/static/app.css", "text/css; charset=utf-8"), ("/static/app.js", "text/javascript; charset=utf-8")):
        status, headers, asset = get(running.port, path)
        assert status == 200 and headers["content-type"] == ctype and headers["content-security-policy"] == DATA_CSP
        assert b"http://" not in asset and b"https://" not in asset and b"@import" not in asset
    assert b"http://" not in body and b"https://" not in body


class _PageScan(HTMLParser):
    """Every start tag of a document with its attributes, and the text inside ``<script>`` / ``<style>`` elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.inline: list[tuple[str, str]] = []
        self._open: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, {k: v or "" for k, v in attrs}))
        if tag in ("script", "style"):
            self._open = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._open:
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._open and data.strip():
            self.inline.append((self._open, data))

    def all(self, tag: str) -> list[dict[str, str]]:
        return [attrs for t, attrs in self.tags if t == tag]


PAGE_TABS = [("overview", "개요"), ("schematic", "회로도"), ("board", "기판"), ("simulation", "시뮬레이션"),
             ("validation", "검증"), ("parts", "부품"), ("reports", "보고서"), ("files", "파일")]


def test_the_page_is_the_korean_app_and_needs_nothing_its_csp_forbids(shared):
    """The page and APP_CSP agree: script and style are same-origin files, and the page carries no inline script, handler or style.

    The CSP choice (the server docstring): ``default-src 'self'`` without
    ``'unsafe-inline'`` / ``'unsafe-eval'``, so every listener is added by
    app.js, every state is a class of app.css, and text reaches the document
    through text nodes. The form's fields are the ``RUN_OPTIONS`` dests
    one to one, every control has a label, and the script only asks for
    elements the page has.
    """
    from ai_eda.cli import RUN_OPTIONS
    from ai_eda.gui.page import APP_CSS, APP_HTML, APP_JS
    from ai_eda.llm.router import TaskKind

    running, *_ = shared
    status, headers, body = get(running.port, "/")
    assert status == 200 and body.decode("utf-8") == APP_HTML
    for forbidden in ("'unsafe-inline'", "'unsafe-eval'", "http:", "https:", "*"):
        assert forbidden not in APP_CSP, forbidden
    scan = _PageScan()
    scan.feed(APP_HTML)
    assert [a["src"] for a in scan.all("script")] == ["/static/app.js"] and scan.inline == [], "one same-origin script, nothing inline"
    assert scan.all("style") == []
    assert sorted((a["rel"], a["href"]) for a in scan.all("link")) == [("icon", "data:,"), ("stylesheet", "/static/app.css")]
    for tag, attrs in scan.tags:
        assert "style" not in attrs, (tag, attrs)
        assert not [k for k in attrs if k.startswith("on")], (tag, attrs)
        for key in ("src", "href", "action", "formaction"):
            if key in attrs:
                value = attrs[key]
                assert value.startswith(("/", "#", "data:")) and not value.startswith("//"), (tag, key, value)
    # the script builds its DOM from text nodes and classes: nothing the CSP (or an injected value) could turn into markup or style
    for pattern in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "setAttribute('on",
                    "setAttribute('style'", ".style.", "cssText", "http://", "https://", "import("):
        assert pattern not in APP_JS, pattern
    assert "@import" not in APP_CSS and "url(" not in APP_CSS
    # the eight tabs, in order, each controlling its panel
    tabs = [a for a in scan.all("button") if a.get("role") == "tab"]
    assert [(a["data-tab"], a["aria-controls"]) for a in tabs] == [(key, f"panel-{key}") for key, _ in PAGE_TABS]
    for key, label in PAGE_TABS:
        assert f'id="tab-{key}"' in APP_HTML and f">{label}</button>" in APP_HTML and f'aria-labelledby="tab-{key}"' in APP_HTML
    # every control has a label (a bound <label for>, or an aria-label / aria-labelledby)
    ids = {a["id"] for _, a in scan.tags if "id" in a}
    bound = {a["for"] for a in scan.all("label")}
    for tag in ("input", "select", "textarea"):
        for attrs in scan.all(tag):
            assert attrs.get("id") in bound or "aria-label" in attrs or "aria-labelledby" in attrs, (tag, attrs)
    for attrs in scan.all("label"):
        assert attrs["for"] in ids, attrs
    # the run form's fields are the CLI's run options, one to one (answers go through the question fields and "추가 답변")
    fields = [a["data-opt"] for _, a in scan.tags if "data-opt" in a]
    assert len(fields) == len(set(fields))
    assert set(fields) == {o.dest for o in RUN_OPTIONS} - {"answer"}
    # the script asks only for elements the page has (or that it creates itself)
    created = set(re.findall(r"\bid: '([A-Za-z0-9_-]+)'", APP_JS))
    wanted = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", APP_JS))
    assert wanted and wanted <= ids | created, wanted - ids - created
    # the new-project form: the name rule and the four template requests, each naming its template's inputs (the answers without --llm)
    assert 'pattern="[A-Za-z0-9][A-Za-z0-9_\\-]{0,63}"' in APP_HTML
    examples = page_examples()
    assert len(examples) == 4 and "5 V 입력, 1 kHz 구형파 발진기" in [e["value"] for e in examples]
    needs = {t.id: set(t.needs) for t in TEMPLATES}
    assert sorted(e["data-template"] for e in examples) == sorted(needs)
    for e in examples:
        assert {pair.split("=", 1)[0] for pair in e["data-inputs"].split("|")} == needs[e["data-template"]], e
    # without --llm the request is not read: the page says where the template inputs go, never that they are asked
    assert "질문으로 값을 받습니다" not in APP_HTML and "LLM 없이 실행하면 요청문은 읽지 않습니다" in APP_HTML
    assert '<option value="">예시 고르기</option>' in APP_HTML, "a placeholder short enough for the 240 px sidebar"
    # the palette and the Korean-capable font stack of the design
    for token in ("#fcfcfb", "#0b0b0b", "#52514e", "#e6e4dc", "#2a78d6", "#008300", "#e34948", "#8a8983",
                  '"Noto Sans CJK KR"', '"Malgun Gothic"', '"Apple SD Gothic Neo"'):
        assert token in APP_CSS, token
    # the task names the per-task model rows offer are the router's own
    project = get_json(running.port, "/api/projects/divider")
    assert project["llm"]["tasks"] == [str(t) for t in TaskKind]


def page_examples() -> list[dict[str, str]]:
    """The attributes of the new-project form's example options (the ones naming a template)."""
    from ai_eda.gui.page import APP_HTML

    scan = _PageScan()
    scan.feed(APP_HTML)
    return [a for a in scan.all("option") if "data-template" in a]


def test_each_example_request_names_the_answers_that_select_its_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without --llm the request text is not read: the example's ``data-inputs`` given as answers (after the two baseline
    questions) are what makes its template present the design table - the path the page tells the user to take."""
    from tests.test_circuit_templates import BASE

    lib = template_library(tmp_path / "kicad")
    monkeypatch.setenv("KICAD10_SYMBOL_DIR", str(lib.roots[0] / "symbols"))
    monkeypatch.setattr("ai_eda.tools.kicad.cli.find_kicad_cli", lambda: None)
    for example in page_examples():
        name = example["data-template"]
        ir_path = tmp_path / name / "ir.json"
        assert _cli("new", name, "--dir", str(ir_path.parent), "--request", example["value"])[0] == 0
        inputs = dict(pair.split("=", 1) for pair in example["data-inputs"].split("|"))
        answers = [arg for k, v in {**BASE, **inputs}.items() for arg in ("--answer", f"{k}={v}")]
        code, out, err = _cli("run", str(ir_path), "--no-pdf", *answers)
        assert code == 1, (name, err[-2000:])
        last = ProjectsRoot(tmp_path).get(name).last_run  # what the page's question form shows, as the GUI reads it
        assert last is not None and [q.key for q in last.open_questions] == [CONFIRM_DESIGN_KEY], (name, out[-2000:])
        assert f"template {name} v" in last.open_questions[0].rationale, (name, last.open_questions[0].rationale)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH (only a syntax check of app.js)")
def test_the_page_script_parses(tmp_path: Path):
    from ai_eda.gui.page import APP_JS

    script = tmp_path / "app.js"
    script.write_text(APP_JS, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.returncode == 0, result.stderr


def _dump_dom(browser: Path, url: str, profile: Path) -> tuple[str, str]:
    """The DOM a headless browser holds after the page settled, and its log (console lines included); no host but 127.0.0.1 resolves."""
    result = subprocess.run(
        [str(browser), "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--disable-extensions",
         "--disable-background-networking", "--no-proxy-server", "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
         f"--user-data-dir={profile}", "--enable-logging=stderr", "--v=0", "--virtual-time-budget=15000", "--dump-dom", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout, result.stderr


@pytest.mark.browser
def test_the_page_runs_in_a_real_browser_under_its_csp(shared, tmp_path: Path):
    """The real page in a headless Chromium: the CSP refuses nothing, no script error, and the hash opens a project and a tab.

    The overview copies the recorded run (19 stage rows, the RELEASE line, the
    status words), the files tab lists the artifacts and the zip, and the
    reports tab shows the stage report in a sandboxed iframe.
    """
    from ai_eda.report.pdf import find_browser

    browser = find_browser()
    if browser is None:
        pytest.skip("no headless Chromium / Chrome / Edge found")
    running, ir, *_ = shared
    base = f"http://127.0.0.1:{running.port}/"
    dom, log = _dump_dom(browser, base + "#project=divider", tmp_path / "p1")
    for text in ("Refused to", "Content Security Policy", "Uncaught"):
        assert text not in log, [line for line in log.splitlines() if text in line][:3]
    assert 'aria-current="page"' in dom and ">divider</h2>" in dom
    stages = dom.split('id="ov-stages"', 1)[1].split("</table>", 1)[0]
    assert stages.count("<tr") == 1 + len(STAGE_ORDER), "the header row and one row per stage"
    assert 'class="status status-pass">PASS<' in dom and "RELEASE" in dom
    assert 'data-opt="llm"' in dom and "사용 안 함" in dom
    dom, log = _dump_dom(browser, base + "#project=divider&tab=files", tmp_path / "p2")
    assert "Refused to" not in log and "Uncaught" not in log
    files = dom.split('id="files-view"', 1)[1]
    assert 'href="/files/divider.zip"' in files and "divider.kicad_sch" in files and "예" in files
    dom, log = _dump_dom(browser, base + "#project=divider&tab=reports&report=architecture", tmp_path / "p3")
    assert "Refused to" not in log and "Uncaught" not in log
    frame = re.search(r"<iframe[^>]*>", dom)
    assert frame is not None and 'sandbox=""' in frame.group(0) and "/preview/divider/reports/" in frame.group(0)


@pytest.mark.browser
def test_every_tab_and_the_design_confirmation_form_render_in_a_real_browser(shared, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A tab whose renderer throws shows the page's own "그릴 수 없음" notice (renderTab catches it, so no "Uncaught" is logged):
    no tab of the compiled fixture does, and a project blocked on the design table shows its folded optional questions with
    the warning, not an error."""
    from ai_eda.report.pdf import find_browser
    from tests.test_circuit_templates import BASE

    browser = find_browser()
    if browser is None:
        pytest.skip("no headless Chromium / Chrome / Edge found")
    running, *_ = shared
    base = f"http://127.0.0.1:{running.port}/"
    for i, (tab, _label) in enumerate(PAGE_TABS):
        dom, log = _dump_dom(browser, base + f"#project=divider&tab={tab}", tmp_path / f"t{i}")
        assert "그릴 수 없음" not in dom and "Uncaught" not in log and "Refused to" not in log, tab
        assert f'id="panel-{tab}" class="panel" role="tabpanel" aria-labelledby="tab-{tab}">' in dom, tab
    # a project blocked on confirm_design (the divider example's inputs given as answers, as the page tells the user to)
    lib = template_library(tmp_path / "kicad")
    monkeypatch.setenv("KICAD10_SYMBOL_DIR", str(lib.roots[0] / "symbols"))
    monkeypatch.setattr("ai_eda.tools.kicad.cli.find_kicad_cli", lambda: None)
    root = tmp_path / "root"
    assert _cli("new", "confirm", "--dir", str(root / "confirm"), "--request", "12 V 입력을 5 V로 나누는 무부하 저항 분배기")[0] == 0
    answers = [arg for k, v in {**BASE, **DIVIDER}.items() for arg in ("--answer", f"{k}={v}")]
    assert _cli("run", str(root / "confirm" / "ir.json"), "--no-pdf", *answers)[0] == 1
    # and a project run to RELEASE with no design: bom.csv / cpl.csv hold their header only
    assert _cli("new", "nodesign", "--dir", str(root / "nodesign"), "--request", "12 V 입력을 5 V로 나누는 무부하 저항 분배기")[0] == 0
    assert _cli("run", str(root / "nodesign" / "ir.json"), "--no-pdf", *[a for k, v in BASE.items() for a in ("--answer", f"{k}={v}")])[0] == 0
    confirm = Running(root)
    try:
        project = get_json(confirm.port, "/api/projects/confirm")
        assert [q["key"] for q in project["questions"]["required"]] == [CONFIRM_DESIGN_KEY] and project["questions"]["optional"]
        dom, log = _dump_dom(browser, f"http://127.0.0.1:{confirm.port}/#project=confirm", tmp_path / "confirm_profile")
        assert get_json(confirm.port, "/preview/nodesign/bom.json")["rows"] == [] and get_json(confirm.port, "/preview/nodesign/cpl.json")["rows"] == []
        parts, parts_log = _dump_dom(browser, f"http://127.0.0.1:{confirm.port}/#project=nodesign&tab=parts", tmp_path / "parts_profile")
    finally:
        confirm.stop()
    assert "그릴 수 없음" not in dom and "Uncaught" not in log, [line for line in log.splitlines() if "Uncaught" in line][:3]
    folded = dom.split('id="optional-questions"', 1)
    assert len(folded) == 2 and "같은 실행에 보내지 마십시오" in folded[1] and 'data-answer="operating_temperature"' in folded[1]
    assert "이 확인은 다른 답 없이 혼자 보내십시오" in dom and 'data-answer="confirm_design"' in dom
    # a header-only table is a sentence saying why, per file (a BOM row per component, a CPL row per placement), not a bare header row
    assert "그릴 수 없음" not in parts and "Uncaught" not in parts_log
    view = parts.split('id="parts-view"', 1)[1]
    assert "bom.csv에 행이 없습니다: BOM은 IR의 부품마다 한 줄이라" in view and "cpl.csv에 행이 없습니다: CPL은 기판에 배치된 부품마다 한 줄이라" in view
    assert "<th" not in view.split("부품 존재 확인", 1)[0], "no header-only BOM / CPL table"


def test_host_guard_answers_only_for_this_machine(shared):
    running, *_ = shared
    port = running.port
    for host in ("evil.example", f"evil.example:{port}", "127.0.0.1.evil.example", ""):
        status, headers, body = get(port, "/api/projects", headers={"Host": host})
        assert status == 421 and one_line(body).startswith("misdirected request"), host
        status, _h, body = post(port, "/api/projects", {"name": "x"}, headers={"Host": host})
        assert status == 421, host
    for host in (f"localhost:{port}", "127.0.0.1", f"[::1]:{port}"):
        assert get(port, "/api/projects", headers={"Host": host})[0] == 200, host


def test_unknown_routes_are_a_one_line_404_and_other_methods_a_one_line_error(shared):
    running, *_ = shared
    port = running.port
    for path in ("/index.html", "/api", "/api/projects/", "/api/projects/divider/nope", "/preview/divider", "/preview/divider/nope.svg",
                 "/preview/nope/schematic.svg", "/files", "/files/divider", "/files/.zip", "/static/other.js", "/api/projects/..%2Fx",
                 "/api/projects/%ff", "/preview/divider/waveform/tran.png"):
        status, headers, body = get(port, path)
        assert status == 404 and headers["content-type"] == "text/plain; charset=utf-8", path
        one_line(body)
    for method in ("PUT", "DELETE", "PATCH"):
        status, _h, body = request(port, method, "/api/projects/divider")
        assert status == 501 and one_line(body), method
    status, _h, _b = post(port, "/api/projects/divider", {})
    assert status == 404, "a GET-only path does not take a POST"


# --------------------------------------------------------------------------- projects and the project JSON


def test_project_list_and_project_json_copy_what_the_files_record(shared):
    running, ir, _lib, ir_path = shared
    before = ir_path.read_bytes()
    pipeline_before = (ir_path.parent / "pipeline.json").read_bytes()
    listing = get_json(running.port, "/api/projects")
    assert [p["name"] for p in listing["projects"]] == ["divider"], "the library folder has no ir.json: not a project"
    assert listing["root"] == str(ir_path.parent.parent.resolve())
    project = get_json(running.port, "/api/projects/divider")
    assert set(project) >= {"info", "report", "questions", "llm", "browser_found", "previews", "simulation", "run_options", "active_run", "runs"}
    # the report is build_report_data of the same files, verbatim: nothing is computed on the way
    loaded, sha = load_ir_file(ir_path)
    assert project["report"] == build_report_data(loaded, ir_path, ir_path.parent, ir_sha=sha).model_dump(mode="json")
    assert [row["stage"] for row in project["report"]["stages"]["rows"]] == [str(s) for s in STAGE_ORDER]
    assert project["info"]["design_hash"] == ir.content_hash() and project["info"]["workdir_mismatch"] is False
    assert project["questions"]["required"] == [] and all(q["required"] is False for q in project["questions"]["optional"])
    assert project["llm"]["pinned_model"] is None and [p["name"] for p in project["llm"]["providers"]] == ["openrouter", "claude"]
    previews = project["previews"]
    assert previews["schematic"] and previews["schematic_file"] == "divider.kicad_sch"
    assert previews["board"] and previews["board_file"] == "divider.kicad_pcb"
    assert previews["waveforms"] == FIXTURE_WAVEFORMS and previews["report_html"] is True and previews["report_html_file"] == "report.html"
    assert previews["bom"] and previews["cpl"]
    assert [(r["stage"], r["md"], r["html"], r["pdf"]) for r in previews["reports"]] == [
        (str(stage), name, name.removesuffix(".md") + ".html", THEORY_PDF if stage is Stage.ARCHITECTURE else None) for stage, name in STAGE_REPORTS.items()
    ]
    sim = project["simulation"]
    assert sim["engine_version"] == "ngspice-test" and sim["current"] is False and sim["freshness"].startswith(RESULTS_NOT_CURRENT)
    assert [a["id"] for a in sim["analyses"]] == ["op", "tran", "dc", "ac", "tran_failed", "tran_env", "bad id!"]
    assert sim["analyses"][0]["op"] == {"vin": 12.0, "vout": 6.0, "vvin#branch": -6e-4}
    assert sim["analyses"][5]["unverifiable"] == "the engine was not available"
    assert project["run_options"]["review"] == [] and project["run_options"]["stage-reports"] == ["no_pdf", "browser"]
    assert "answer" not in project["run_options"]["run"] and "llm_budget_usd" in project["run_options"]["run"]
    assert project["active_run"] is None and [r["id"] for r in project["runs"]] == ["001-run"]
    assert ir_path.read_bytes() == before and (ir_path.parent / "pipeline.json").read_bytes() == pipeline_before, "the GUI never writes the IR or the run log"


def test_create_project_through_the_api(tmp_path: Path):
    running = Running(tmp_path / "root")
    try:
        status, _h, body = post(running.port, "/api/projects", {"name": "osc-1", "request": "5 V 입력, 1 kHz 구형파 발진기"})
        assert status == 201
        made = json.loads(body)
        ir_path = tmp_path / "root" / "osc-1" / "ir.json"
        assert made["name"] == "osc-1" and made["ir_path"] == str(ir_path.resolve())
        ir = CircuitIR.load(ir_path)
        assert ir.requirements.raw_input == "5 V 입력, 1 kHz 구형파 발진기" and ir.project.workdir == str(ir_path.parent.resolve())
        project = get_json(running.port, "/api/projects/osc-1")
        assert project["info"]["last_run"] is None and project["report"]["stages"] is None and project["previews"]["waveforms"] == []
        assert project["previews"]["schematic"] is False and project["previews"]["board"] is False and project["previews"]["reports"] == []
        # refused: a bad name, an existing folder, a field that is not a project field, a body that is not an object
        for bad in ({"name": "../x"}, {"name": "한글"}, {"name": "osc-1"}, {"name": "ok", "extra": 1}, {"name": 5}):
            status, _h, body = post(running.port, "/api/projects", bad)
            assert status == 400 and one_line(body), bad
        status, _h, body = request(running.port, "POST", "/api/projects", raw=b"[1]", headers={"Content-Type": "application/json"})
        assert status == 400 and "JSON object" in one_line(body)
        assert sorted(p.name for p in (tmp_path / "root").iterdir()) == ["osc-1"]
    finally:
        running.stop()


def test_post_needs_the_pages_own_origin_and_a_json_body(tmp_path: Path):
    running = Running(tmp_path / "root")
    port = running.port
    try:
        for origin in ("http://evil.example", "null", f"http://127.0.0.1:{port + 1}", f"https://127.0.0.1:{port}", f"http://evil.example:{port}"):
            status, _h, body = post(port, "/api/projects", {"name": "x1"}, headers={"Origin": origin})
            assert status == 403 and "another origin" in one_line(body), origin
        assert not (tmp_path / "root").exists(), "a refused request writes nothing"
        for i, origin in enumerate((None, f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}")):
            headers = {"Origin": origin} if origin else {}
            status, _h, body = post(port, "/api/projects", {"name": f"ok{i}"}, headers=headers)
            assert status == 201, (origin, body)
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", ""):
            status, _h, body = request(port, "POST", "/api/projects", raw=b'{"name": "x2"}', headers={"Content-Type": ctype} if ctype else {})
            assert status == 415 and "application/json" in one_line(body), ctype
        status, _h, _b = request(port, "POST", "/api/projects", raw=b'{"name": "x3"}', headers={"Content-Type": "application/json; charset=utf-8"})
        assert status == 201
        for raw in (b"{", b'{"name": NaN}', b"\xff\xfe"):
            status, _h, body = request(port, "POST", "/api/projects", raw=raw, headers={"Content-Type": "application/json"})
            assert status == 400 and one_line(body).startswith("the body is not JSON"), raw
        # a body over the limit, or one without Content-Length, is refused before it is read
        for length_header in (f"Content-Length: {server_module.MAX_BODY_BYTES + 1}\r\n", "Transfer-Encoding: chunked\r\n", ""):
            status_line, body = raw_post(port, "/api/projects", length_header)
            code = 413 if "Content-Length" in length_header else 411
            assert status_line.startswith(f"HTTP/1.1 {code} ") and one_line(body), length_header
        assert sorted(p.name for p in (tmp_path / "root").iterdir()) == ["ok0", "ok1", "ok2", "ok3", "x3"]
    finally:
        running.stop()


def test_the_run_endpoints_need_the_pages_own_origin_and_a_json_body(tmp_path: Path):
    """A page on another origin can start no run, review or report re-write - not even an ``--online`` one; a CORS simple request
    (text/plain, no Origin) cannot either. The page's own origin starts one (the check is not vacuous)."""
    running = Running(tmp_path / "root", runs=RunManager(python_args=[sys.executable, "-c", "print('ran')"]))
    port = running.port
    try:
        assert post(port, "/api/projects", {"name": "p"})[0] == 201
        body = json.dumps({"options": {"online": True}}).encode("utf-8")
        for kind in ("run", "review", "stage-reports"):
            for origin in ("http://evil.example", "null", f"http://evil.example:{port}"):
                status, _h, data = request(port, "POST", f"/api/projects/p/{kind}", raw=body, headers={"Content-Type": "application/json", "Origin": origin})
                assert status == 403 and "another origin" in one_line(data), (kind, origin)
            for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
                status, _h, data = request(port, "POST", f"/api/projects/p/{kind}", raw=body, headers={"Content-Type": ctype})
                assert status == 415 and "application/json" in one_line(data), (kind, ctype)
        assert not (tmp_path / "root" / "p" / "gui").exists(), "no log written, no process started"
        run_id = start_run(port, "p", headers={"Origin": f"http://127.0.0.1:{port}"})
        assert run_id == "001-run" and wait_run(port, "p", run_id)["exit_code"] == 0
    finally:
        running.stop()


# --------------------------------------------------------------------------- previews


def test_schematic_preview_draws_every_component_stub_and_label(shared):
    running, ir, *_ = shared
    status, headers, body = get(running.port, "/preview/divider/schematic.svg")
    assert status == 200 and headers["content-type"] == "image/svg+xml; charset=utf-8" and headers["content-security-policy"] == DATA_CSP
    root = parse_svg(body)
    symbols = [g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "symbol"]
    assert sorted(g.get("data-ref") for g in symbols) == sorted(c.ref for c in ir.components), "one symbol group per component"
    net_pins = sum(len(n.pins) for n in ir.nets)
    assert len(classed(root, "line", "wire")) == net_pins, "one wire stub per net pin"
    labels = [g for g in root.iter(f"{SVG_NS}g") if g.get("class") == "global-label"]
    assert len(labels) == net_pins and sorted({g.get("data-net") for g in labels}) == sorted(n.name for n in ir.nets)
    assert get(running.port, "/preview/divider/schematic.svg")[2] == body, "deterministic"


def test_board_preview_carries_the_layer_classes(shared):
    running, ir, *_ = shared
    status, headers, body = get(running.port, "/preview/divider/board.svg?layers=F.Cu,B.Cu")
    assert status == 200 and headers["content-type"] == "image/svg+xml; charset=utf-8"
    root = parse_svg(body)
    groups = {cls for g in root.iter(f"{SVG_NS}g") for cls in (g.get("class") or "").split()}
    assert {"layer-F_Cu", "layer-B_Cu", "pads", "vias", "labels"} <= groups
    assert len(classed(root, "rect", "outline")) == 1
    for layer in ("F.Cu", "B.Cu"):
        group = next(g for g in root.iter(f"{SVG_NS}g") if g.get("class") == f"layer-{layer.replace('.', '_')}")
        assert group.get("data-layer") == layer
        assert [t.get("data-layer") for t in classed(group, "line", "track")] == [layer] * sum(1 for t in ir.pcb.tracks if t.layer == layer)
    assert len(classed(root, "line", "track")) == len(ir.pcb.tracks) and len(classed(root, "g", "via")) == len(ir.pcb.vias)


def test_waveform_previews_draw_every_recorded_analysis(shared):
    running, *_ = shared
    drawn: dict[str, list[str]] = {}
    for fig_id in FIXTURE_WAVEFORMS:
        status, headers, body = get(running.port, f"/preview/divider/waveform/{fig_id}.svg")
        assert status == 200 and headers["content-type"] == "image/svg+xml; charset=utf-8", fig_id
        drawn[fig_id] = [p.get("data-name") for p in classed(parse_svg(body), "polyline", "series")]
    # every vector as recorded, voltages and currents apart, four per chart; an AC analysis by magnitude only
    assert drawn == {"tran": ["vin", "vout", "n1", "n2"], "tran_2": ["n3"], "tran_i": ["vvin#branch"], "dc": ["vout"], "ac": ["vout"]}
    for fig_id in ("op", "tran_failed", "tran_env", "bad id!", "nope"):
        status, _h, body = get(running.port, f"/preview/divider/waveform/{fig_id.replace(' ', '%20')}.svg")
        assert status == 404 and one_line(body) == NO_WAVEFORM, fig_id


def test_waveform_plan_ids_follow_waveform_figures():
    plans = waveform_plan(fixture_results())
    assert [(p.id, p.analysis_id, p.vectors) for p in plans] == [
        ("tran", "tran", ("vin", "vout", "n1", "n2")), ("tran_2", "tran", ("n3",)), ("tran_i", "tran", ("vvin#branch",)),
        ("dc", "dc", ("vout",)), ("ac", "ac", ("vout",)),
    ]
    assert plans[0].title.endswith("(1/2)") and plans[2].title.endswith("(전류)")
    # an analysis id that collides with another chart's id is skipped, never drawn over it
    data = fixture_results()
    data["analyses"]["tran_i"] = data["analyses"]["dc"]
    assert [p.id for p in waveform_plan(data)].count("tran_i") == 1
    assert waveform_plan({}) == [] and waveform_plan({"analyses": []}) == []


def test_tables_artifacts_and_the_ir_file(shared):
    running, ir, _lib, ir_path = shared
    bom = get_json(running.port, "/preview/divider/bom.json")
    assert bom["file"] == "bom.csv" and bom["columns"][:3] == ["Reference", "Value", "Description"]
    assert [row["Reference"] for row in bom["rows"]] == sorted(c.ref for c in ir.components)
    assert {row["Reference"]: row["Value"] for row in bom["rows"]} == {c.ref: c.value for c in ir.components}
    assert len(bom["not_verified"]) == len(bom["rows"]) and all("MPN" in cols or row["MPN"] != "NOT_VERIFIED" for cols, row in zip(bom["not_verified"], bom["rows"]))
    assert bom["notes"] == list(ir.artifacts["bom"].notes)
    cpl = get_json(running.port, "/preview/divider/cpl.json")
    assert cpl["columns"] == ["Designator", "Mid X", "Mid Y", "Rotation", "Layer"] and len(cpl["rows"]) == len(ir.pcb.placements)
    artifacts = get_json(running.port, "/preview/divider/artifacts.json")
    assert artifacts["design_hash"] == ir.content_hash()
    rows = {row["kind"]: row for row in artifacts["artifacts"]}
    assert sorted(rows) == sorted(str(k) for k in ir.artifacts)
    for kind, art in ir.artifacts.items():
        row = rows[str(kind)]
        assert row["path"] == Path(art.path).name == row["download"] and row["inside_workdir"] is True
        assert row["content_hash"] == art.content_hash and row["disk"] == "on disk" and row["matches_design_hash"] is True and row["freshness"] == "fresh"
    status, headers, body = get(running.port, "/preview/divider/ir.json")
    assert status == 200 and headers["content-type"] == "application/json; charset=utf-8" and body == ir_path.read_bytes()


def test_bom_free_text_cells_are_decoded_and_every_other_cell_shown_as_written(tmp_path: Path):
    # free_text_cell wrote '-12V for -12V and ''quoted for 'quoted; an identity cell is never neutralised, so it is shown as written
    (tmp_path / "bom.csv").write_text(
        "Reference,Value,Description,MPN\nR1,'-12V,''quoted,NOT_VERIFIED\nR2,10k,plain,'kept\n", encoding="utf-8"
    )
    table = bom_table(tmp_path)
    assert table["columns"] == ["Reference", "Value", "Description", "MPN"] and table["file"] == "bom.csv"
    assert table["rows"] == [
        {"Reference": "R1", "Value": "-12V", "Description": "'quoted", "MPN": "NOT_VERIFIED"},
        {"Reference": "R2", "Value": "10k", "Description": "plain", "MPN": "'kept"},
    ]
    assert table["not_verified"] == [["MPN"], []]
    (tmp_path / "bad.csv").write_text("A,B\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 has 3 cells"):
        read_csv_table(tmp_path / "bad.csv")
    assert bom_table(tmp_path / "nowhere") is None


def test_report_routes_serve_the_documents_with_their_own_csp(shared):
    running, *_ = shared
    port = running.port
    status, headers, body = get(port, "/preview/divider/report.html")
    assert status == 200 and headers["content-type"] == "text/html; charset=utf-8" and headers["content-security-policy"] == DOCUMENT_CSP
    assert b"<script" not in body and b"divider - design report" in body
    quoted = "/preview/divider/reports/" + "%s"
    from urllib.parse import quote

    status, headers, body = get(port, quoted % quote(THEORY_HTML))
    assert status == 200 and headers["content-type"] == "text/html; charset=utf-8" and headers["content-security-policy"] == DOCUMENT_CSP
    assert "이론 보고서" in body.decode("utf-8")
    status, headers, body = get(port, quoted % quote(THEORY_MD))
    assert status == 200 and headers["content-type"] == "text/plain; charset=utf-8" and body.decode("utf-8").startswith("# 이론 보고서")
    status, headers, body = get(port, quoted % quote(THEORY_PDF))
    assert status == 200 and headers["content-type"] == "application/pdf" and body.startswith(b"%PDF")
    assert headers["content-disposition"].startswith("inline; ") and "filename*=UTF-8''" in headers["content-disposition"]
    # the one response without a CSP (the module docstring's exception): the browser's own PDF viewer renders it inline
    assert "content-security-policy" not in headers
    assert headers["x-content-type-options"] == "nosniff" and headers["cache-control"] == "no-store" and headers["x-frame-options"] == "SAMEORIGIN"
    status, headers, _b = get(port, "/files/divider/reports/" + quote(THEORY_PDF))
    assert status == 200 and headers["content-security-policy"] == DATA_CSP, "the same PDF as a download keeps DATA_CSP"
    for name in ("ir.json", "..%2Fir.json", "report.html", quote(STAGE_REPORTS[Stage.PCB].removesuffix(".md") + ".pdf"), "notes.txt"):
        status, _h, body = get(port, quoted % name)
        assert status == 404, name
        one_line(body)


def test_files_are_confined_to_the_workdir(fresh):
    from urllib.parse import quote

    running, _ir, _lib, ir_path = fresh
    port = running.port
    workdir = ir_path.parent
    status, headers, body = get(port, "/files/divider/bom.csv")
    assert status == 200 and headers["content-type"] == "application/octet-stream" and body == (workdir / "bom.csv").read_bytes()
    assert headers["content-disposition"] == "attachment; filename=\"bom.csv\"; filename*=UTF-8''bom.csv"
    status, _h, body = get(port, "/files/divider/spice/tran/divider.tran.raw")
    assert status == 200 and body.startswith(b"Title:")
    outside = workdir.parent / "outside.txt"
    outside.write_text("secret outside the workdir", encoding="utf-8")
    links_made = True
    try:
        (workdir / "out_link.txt").symlink_to(outside)
        (workdir / "in_link.csv").symlink_to(workdir / "bom.csv")
        (workdir / "spice_link").symlink_to(workdir / "spice", target_is_directory=True)
    except (OSError, NotImplementedError):
        links_made = False
    paths = [
        "/files/divider/..", "/files/divider/../outside.txt", "/files/divider/%2e%2e/outside.txt", "/files/divider/%2Fetc%2Fpasswd",
        "/files/divider//etc/passwd", "/files/divider/C:%5Cwindows", "/files/divider/reports", "/files/divider/spice",
        "/files/divider/CON", "/files/divider/nul.txt", "/files/divider/missing.csv", "/files/nope/bom.csv",
        # an absolute path as one segment (POSIX /tmp/..., or C:\\Users\\...), percent-encoded: never a raw space on the request line
        "/files/divider/" + quote(str(outside), safe=""), "/files/divider/" + quote("C:\\Users\\John Doe\\outside.txt", safe=""),
    ]
    if links_made:
        paths += ["/files/divider/out_link.txt", "/files/divider/in_link.csv", "/files/divider/spice_link/results.json"]
    for path in paths:
        status, _h, body = get(port, path)
        assert status == 404 and b"secret" not in body, path
        one_line(body)
    assert confined_file(workdir, ("bom.csv",)) == (workdir / "bom.csv").resolve()
    assert confined_file(workdir, ()) is None and confined_file(workdir, ("..", workdir.name, "bom.csv")) is None
    assert workdir_parts(workdir / "a" / "b.txt", workdir) == ("a", "b.txt") and workdir_parts(outside, workdir) is None
    assert workdir_parts("rel/x.csv", workdir) == ("rel", "x.csv") and workdir_parts(workdir, workdir) is None
    if links_made:  # a link is never packed either
        names = zipfile.ZipFile(io.BytesIO(get(port, "/files/divider.zip")[2])).namelist()
        assert not any("link" in n for n in names)


def test_the_zip_holds_the_project_and_nothing_else_and_is_deterministic(shared):
    running, ir, _lib, ir_path = shared
    workdir = ir_path.parent
    status, headers, body = get(running.port, "/files/divider.zip")
    assert status == 200 and headers["content-type"] == "application/zip"
    assert headers["content-disposition"].startswith('attachment; filename="divider.zip"')
    zf = zipfile.ZipFile(io.BytesIO(body))
    expected = sorted(
        {Path(a.path).name for a in ir.artifacts.values()}
        | {"ir.json", "pipeline.json", "report.html", "spice/results.json", "spice/tran/divider.tran.raw"}
        | {f"reports/{p.name}" for p in (workdir / "reports").iterdir()}
    )
    assert zf.namelist() == [f"divider/{n}" for n in expected]
    assert not any("/sources/" in n or "/gui/" in n for n in zf.namelist())
    assert all(info.date_time == ZIP_DATE_TIME for info in zf.infolist())
    assert zf.read("divider/ir.json") == ir_path.read_bytes() and zf.read("divider/bom.csv") == (workdir / "bom.csv").read_bytes()
    os.utime(workdir / "bom.csv", (1_000_000_000, 1_000_000_000))  # a new mtime changes nothing
    assert get(running.port, "/files/divider.zip")[2] == body


def test_a_broken_ir_json_or_a_failed_render_is_a_one_line_503(fresh, tmp_path: Path):
    running, ir, _lib, ir_path = fresh
    port = running.port
    # a schematic the renderer refuses, and a board whose footprints are not in the server's library
    sch = ir_path.parent / "divider.kicad_sch"
    sch.write_text("(not_a_schematic)", encoding="utf-8")
    status, _h, body = get(port, "/preview/divider/schematic.svg")
    assert status == 503 and one_line(body).startswith("schematic unavailable: ")
    no_lib = Running(ir_path.parent.parent, library=KicadLibrary(roots=[tmp_path / "empty"]))
    try:
        status, _h, body = get(no_lib.port, "/preview/divider/board.svg")
        assert status == 503 and "was not found in a KiCad library" in one_line(body)
    finally:
        no_lib.stop()
    (ir_path.parent / "bom.csv").write_text("Reference,Value\nR1,1k,extra\n", encoding="utf-8")
    status, _h, body = get(port, "/preview/divider/bom.json")
    assert status == 503 and "line 2 has 3 cells" in one_line(body)
    ir_path.write_text("{", encoding="utf-8")
    for path in ("/api/projects/divider", "/preview/divider/schematic.svg", "/preview/divider/report.html", "/files/divider/cpl.csv",
                 "/files/divider.zip", "/api/projects/divider/runs"):
        status, headers, body = get(port, path)
        assert status == 503 and headers["content-type"] == "text/plain; charset=utf-8", path
        assert one_line(body).startswith("project unavailable: "), path
    listing = get_json(port, "/api/projects")
    assert listing["projects"][0]["error"].startswith("ir.json을 읽을 수 없습니다")
    status, _h, _b = post(port, "/api/projects/divider/run", {})
    assert status == 400, "a project that cannot be read cannot run"


def test_each_file_preview_carries_its_artifact_facts(shared):
    """The schematic / board / BOM / CPL previews carry their artifact row's facts (copied, never recomputed)."""
    running, ir, _lib, ir_path = shared
    project = get_json(running.port, "/api/projects/divider")
    rows = {r["kind"]: r for r in get_json(running.port, "/preview/divider/artifacts.json")["artifacts"]}
    previews = project["previews"]
    for key, kind in (("schematic_state", "kicad_sch"), ("board_state", "kicad_pcb"), ("bom_state", "bom"), ("cpl_state", "cpl")):
        state = previews[key]
        assert state == {k: rows[kind][k] for k in state} and set(state) == {"kind", "path", "download", "freshness", "matches_design_hash", "disk", "disk_matches"}
        assert state["matches_design_hash"] is True and state["freshness"] == "fresh" and state["disk"] == "on disk" and state["disk_matches"] is True
    assert get_json(running.port, "/preview/divider/bom.json")["state"] == previews["bom_state"]
    assert get_json(running.port, "/preview/divider/cpl.json")["state"] == previews["cpl_state"]


def test_a_stale_or_missing_registered_file_is_labelled_not_passed_off_as_current(fresh):
    """A schematic / BOM compiled from an earlier design is drawn with its facts (not the current design hash); a registered file
    that is gone says so - the stage ran and registered it - instead of saying the stage has not run."""
    running, ir, _lib, ir_path = fresh
    port = running.port
    data = json.loads(ir_path.read_text(encoding="utf-8"))
    r1 = next(c for c in data["components"] if c["ref"] == "R1")
    r1["value"] = r1["value"] + "X"  # a hand edit: the compiled files now describe the previous design
    ir_path.write_text(json.dumps(data), encoding="utf-8")
    project = get_json(port, "/api/projects/divider")
    for key in ("schematic_state", "board_state", "bom_state", "cpl_state"):
        state = project["previews"][key]
        assert state["matches_design_hash"] is False and state["freshness"] == "stale" and state["disk_matches"] is True, key
    assert project["labels"]["stale"] == "낡음 (이전 설계 해시)"
    assert get(port, "/preview/divider/schematic.svg")[0] == 200, "still drawn: the file as it is, labelled by its facts"
    bom = get_json(port, "/preview/divider/bom.json")
    assert bom["state"]["matches_design_hash"] is False and bom["labels"]["stale"] == "낡음 (이전 설계 해시)"
    # the registered files are deleted by hand: the sentence names the registered file, never "the stage has not run"
    workdir = ir_path.parent
    for path, route, word in (("divider.kicad_sch", "schematic.svg", "회로도"), ("bom.csv", "bom.json", "BOM"), ("cpl.csv", "cpl.json", "CPL")):
        (workdir / path).unlink()
        status, _h, body = get(port, f"/preview/divider/{route}")
        line = one_line(body)
        assert status == 404 and line.startswith(f"{word} 파일 없음: IR에 등록된 파일({path})") and "ir.artifacts에 등록" in line, line
        assert "아직" not in line
    project = get_json(port, "/api/projects/divider")
    assert project["previews"]["schematic"] is False and project["previews"]["schematic_state"]["disk"] == "missing on disk"
    assert project["previews"]["schematic_state"]["disk_matches"] is False and project["labels"]["missing on disk"] == "디스크에 없음"


def test_empty_state_sentences_name_the_missing_file_and_real_stages():
    """No empty state claims a stage "has not run" (the copied stage table says whether it ran), and every stage it names exists."""
    from ai_eda.gui.page import APP_HTML, APP_JS

    names = {s.name for s in Stage}
    for sentence in (NO_SCHEMATIC, NO_BOARD, NO_RESULTS, NO_BOM, NO_CPL):
        assert "아직 실행되지 않았습니다" not in sentence, sentence
        stages = re.findall(r"\b([A-Z][A-Z_]+) 단계", sentence)
        assert stages and set(stages) <= names, sentence
        overview_rows = re.findall(r"개요의 ([a-z_]+) 줄", sentence)
        assert overview_rows and set(overview_rows) <= {str(s) for s in Stage}, sentence
    named = set(re.findall(r"\b([A-Z][A-Z_]{2,}) 단계", APP_HTML + APP_JS))
    assert named and named <= names, named - names
    assert "SIMULATION 단계" not in APP_JS + APP_HTML and "BOM 단계" not in APP_JS + APP_HTML
    # a boolean the page turns into words is Korean on both branches
    assert "'성공 (기록)'" in APP_JS and "succeeded (기록)" not in APP_JS


def test_the_page_shows_the_report_labels_in_korean(shared):
    """Every hash / file / run label of the report (the closed set of ai_eda.report.data) has a Korean wording, the composed
    ``(IR <hash>)`` forms included, and the project JSON carries the ones it holds; other text is never translated."""
    import ai_eda.report.data as data_module
    from ai_eda.gui.labels import KOREAN_LABELS, korean_label, label_map
    from ai_eda.ir.validation import ValidationResult, ValidationStatus
    from ai_eda.report.data import _Freshness, run_hash_label
    from ai_eda.report.pipeline_log import PipelineRecord

    constants = ["FRESH", "STALE", "UNSTAMPED_PRODUCED", "UNSTAMPED_CARRIED", "UNSTAMPED_UNKNOWN", "ON_DISK", "CHANGED_ON_DISK", "MISSING_ON_DISK",
                 "EVIDENCE_OK", "EVIDENCE_NO_HASH", "EVIDENCE_NO_PATH", "EVIDENCE_MISSING", "PIPELINE_DESCRIBES_IR", "PIPELINE_STALE_IR",
                 "RUN_CURRENT_IR", "RUN_EARLIER_IR", "NO_RECORDED_RUN", "OPINION", "MODEL_OUTPUT", "AGGREGATE_NOTE"]
    assert sorted(KOREAN_LABELS) == sorted(getattr(data_module, name) for name in constants)
    hangul = re.compile(r"[\uac00-\ud7a3]")
    for name in constants:
        assert hangul.search(korean_label(getattr(data_module, name)) or ""), name
    stale = _Freshness("sha256:" + "a" * 64, None).label(
        ValidationResult(check_id="x", status=ValidationStatus.PASS, message="m", ir_hash="sha256:" + "b" * 64), 0)
    assert stale.startswith("stale (IR ") and korean_label(stale) == "낡음: 이전 설계 해시 (IR sha256:bbbbbbbbb)"
    running, ir, _lib, ir_path = shared
    record = PipelineRecord.model_validate({**json.loads((ir_path.parent / "pipeline.json").read_text(encoding="utf-8")), "ir_hash": "sha256:" + "c" * 64})
    earlier = run_hash_label(record, ir.content_hash())
    assert korean_label(earlier) == "이전 IR 버전에 대한 실행 기록 (IR sha256:ccccccccc)"
    for text in ("fresh!", "no pipeline.json in /x: run `ai-eda run` to record stage outcomes", "stale (IR x", 5, None):
        assert korean_label(text) is None, text
    # the project JSON carries a Korean wording for every label the page shows, and nothing else
    project = get_json(running.port, "/api/projects/divider")
    report = project["report"]
    shown = {report["meta"]["pipeline_note"], report["stages"]["run_hash_label"], report["release"]["freshness"], report["validation"]["aggregate_note"]}
    for row in report["validation"]["latest"]:
        shown |= {row["freshness"], *(e["state"] for e in row["evidence"])} | ({row["note"]} if row["note"] else set())
    shown |= {r["disk"] for r in report["artifacts"]} | {r["freshness"] for r in report["artifacts"]}
    assert shown <= set(project["labels"]), shown - set(project["labels"])
    assert project["labels"] == label_map(report, get_json(running.port, "/preview/divider/artifacts.json")["artifacts"])
    assert all(hangul.search(v) for v in project["labels"].values())


def _contrast(fg: str, bg: str) -> float:
    def lum(hex_colour: str) -> float:
        rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_warning_text_meets_wcag_aa_on_every_list_background():
    """The project list's warning lines (12 px text) are at least 4.5:1 on the list item, the selected item and white."""
    from ai_eda.gui.page import APP_CSS

    tokens = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-f]{6})", APP_CSS))
    rule = re.search(r"\.pwarn \{([^}]*)\}", APP_CSS).group(1)
    colour = tokens[re.search(r"color: var\(--([a-z-]+)\)", rule).group(1)]
    selected = re.search(r'\.project-list a\[aria-current="page"\] \{[^}]*background: (#[0-9a-f]{6})', APP_CSS).group(1)
    for background in (tokens["surface"], tokens["panel"], selected, "#ffffff"):
        assert _contrast(colour, background) >= 4.5, (colour, background, _contrast(colour, background))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_the_confirm_design_guard_is_the_circuit_agents_rule(shared, tmp_path: Path):
    """The page refuses to send confirm_design=yes with an answer that makes the CLI ignore it - exactly those (the rule the server
    copies from the circuit agent's key sets): a control answer such as pcb.routing=skip still goes with it."""
    from ai_eda.agents.keys import CONTROL_KEYS, REQUIREMENT_DECISION_KEYS
    from ai_eda.gui.page import APP_JS

    running, *_ = shared
    rule = get_json(running.port, "/api/projects/divider")["answers_rule"]
    assert rule == {"confirm_key": CONFIRM_DESIGN_KEY, "control_keys": sorted(CONTROL_KEYS), "decision_keys": sorted(REQUIREMENT_DECISION_KEYS)}
    function = re.search(r"function confirmVoiders\(answers\) \{.*?\n\}", APP_JS, re.S).group(0)
    cases = [
        {"confirm_design": "yes", "operating_temperature": "0 to 40 C"},
        {"confirm_design": "yes", "mains_powered": "no", "pcb.routing": "skip"},
        {"confirm_design": "yes", "pcb.routing": "skip", "pcb.placement": "skip"},
        {"confirm_design": "yes", "confirm_requirements": "yes"},
        {"confirm_design": "yes"},
        {"operating_temperature": "0 to 40 C", "application": "bench"},
    ]
    script = tmp_path / "guard.js"
    script.write_text(f"const state = {{data: {{answers_rule: {json.dumps(rule)}}}}};\n{function}\n"
                      f"console.log(JSON.stringify({json.dumps(cases)}.map(confirmVoiders)));\n", encoding="utf-8")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.returncode == 0, result.stderr
    # the circuit agent's own predicate (ai_eda/agents/circuit.py): what it lists as "given in this run" beside the confirmation
    expected = [sorted(k for k in c if k != CONFIRM_DESIGN_KEY and (k not in CONTROL_KEYS or k in REQUIREMENT_DECISION_KEYS)) if c.get(CONFIRM_DESIGN_KEY) else [] for c in cases]
    assert [sorted(v) for v in json.loads(result.stdout)] == expected == [["operating_temperature"], ["mains_powered"], [], ["confirm_requirements"], [], []]


def test_the_default_model_of_each_provider_is_the_clis_resolution(shared):
    """An empty model field means the first provider's default model: the project JSON names it (for the page's pin hint) exactly as
    the CLI resolves it, so an empty field under a pin to another model is warned about before the CLI refuses it."""
    from ai_eda.cli import parse_llm_options  # noqa: F401 - the resolution below is the one parse_llm_options uses
    from ai_eda.llm.router import default_model, parse_model_spec

    running, *_ = shared
    llm = get_json(running.port, "/api/projects/divider")["llm"]
    assert set(llm["default_specs"]) == {r["name"] for r in llm["providers"]}
    for name, spec in llm["default_specs"].items():
        assert spec == parse_model_spec(default_model(name), name).spec and spec.startswith(f"{name}:")
    from ai_eda.gui.page import APP_JS

    assert "default_specs" in APP_JS and "모델 칸을 비워 두면 이번 실행은 기본 모델" in APP_JS


def test_a_project_ai_eda_new_named_with_a_dot_or_hangul_is_served(tmp_path: Path):
    running = Running(tmp_path / "root")
    try:
        from urllib.parse import quote

        for name in ("osc.v2", "발진기"):
            with redirect_stdout(io.StringIO()):
                assert cli_main(["new", name, "--dir", str(tmp_path / "root" / name)]) == 0
            q = quote(name)
            project = get_json(running.port, f"/api/projects/{q}")
            assert project["info"]["name"] == name and project["info"]["error"] is None
            assert get(running.port, f"/files/{q}.zip")[0] == 200 and get(running.port, f"/files/{q}/ir.json")[0] == 200
        assert sorted(p["name"] for p in get_json(running.port, "/api/projects")["projects"]) == sorted(["osc.v2", "발진기"])
        for bad in ("..", ".", "%2E%2E"):
            assert get(running.port, f"/api/projects/{bad}")[0] == 404, bad
    finally:
        running.stop()


def test_a_project_without_results_has_no_waveforms(tmp_path: Path):
    running = Running(tmp_path / "root")
    try:
        assert post(running.port, "/api/projects", {"name": "empty"})[0] == 201
        project = get_json(running.port, "/api/projects/empty")
        assert project["simulation"]["file"] is None and project["simulation"]["waveforms"] == []
        status, _h, body = get(running.port, "/preview/empty/waveform/tran.svg")
        assert status == 404 and one_line(body) == NO_RESULTS
        for path in ("/preview/empty/schematic.svg", "/preview/empty/board.svg", "/preview/empty/bom.json", "/preview/empty/cpl.json"):
            status, _h, body = get(running.port, path)
            assert status == 404 and "없음" in one_line(body), path
        assert get(running.port, "/preview/empty/report.html")[0] == 200, "the report renders for any readable IR"
        names = zipfile.ZipFile(io.BytesIO(get(running.port, "/files/empty.zip")[2])).namelist()
        assert names == ["empty/ir.json"]
    finally:
        running.stop()


@needs_dll
def test_a_fresh_spice_run_is_labelled_current(tmp_path: Path):
    ir, lib, _ir_path = build_project(tmp_path / "root", "osc", ASTABLE, spice=True)
    running = Running(tmp_path / "root", library=lib)
    try:
        sim = get_json(running.port, "/api/projects/osc")["simulation"]
        assert sim["current"] is True and sim["freshness"] == RESULTS_CURRENT
        ids = [w["id"] for w in sim["waveforms"]]
        assert ids[0] == "tran" and "tran_i" in ids
        status, _h, body = get(running.port, "/preview/osc/waveform/tran.svg")
        assert status == 200 and parse_svg(body) is not None
    finally:
        running.stop()


# --------------------------------------------------------------------------- runs


def test_a_fresh_project_run_blocks_on_its_first_question_then_answers_go_through(tmp_path: Path):
    running = Running(tmp_path / "root")
    port = running.port
    try:
        assert post(port, "/api/projects", {"name": "osc", "request": "5 V 입력, 1 kHz 구형파 발진기"})[0] == 201
        run_id = start_run(port, "osc", options={"no_pdf": True})
        assert run_id == "001-run"
        status = wait_run(port, "osc", run_id)
        assert status["exit_code"] == 1 and status["kind"] == "run" and status["log_tail"].splitlines()[-1] == f"{EXIT_PREFIX}1"
        assert " -m ai_eda.cli run " in status["log_tail"].splitlines()[0] and "--no-pdf" in status["log_tail"].splitlines()[0]
        project = get_json(port, "/api/projects/osc")
        assert [q["key"] for q in project["questions"]["required"]] == ["application", "jurisdiction"]
        assert project["info"]["last_run"]["blocked"] is True and project["active_run"] is None
        assert get_json(port, "/api/projects/osc/runs") == {"runs": [{"id": "001-run", "kind": "run", "size": status["log_bytes"], "running": False}], "active": None}
        # answers with spaces and '=' reach the IR as one argv item each
        run_id = start_run(port, "osc", answers={"application": "교육용 발진기 = 데모", "jurisdiction": "KR"}, options={"no_pdf": True})
        status = wait_run(port, "osc", run_id)
        assert status["exit_code"] == 0, status["log_tail"][-2000:]
        project = get_json(port, "/api/projects/osc")
        assert project["questions"]["required"] == [] and project["info"]["last_run"]["blocked"] is False
        # review and stage-reports are runs of their own kind
        for kind in ("review", "stage-reports"):
            run_id = start_run(port, "osc", kind, options={"no_pdf": True} if kind == "stage-reports" else {})
            status = wait_run(port, "osc", run_id)
            assert run_id.endswith(f"-{kind}") and status["exit_code"] is not None and f"-m ai_eda.cli {kind} " in status["log_tail"].splitlines()[0]
        status_code, _h, body = get(port, "/api/projects/osc/runs/999-run")
        assert status_code == 404 and one_line(body)
        status_code, _h, _b = get(port, "/api/projects/osc/runs/nope")
        assert status_code == 404
    finally:
        running.stop()


def test_a_refused_form_starts_nothing(tmp_path: Path):
    running = Running(tmp_path / "root")
    port = running.port
    try:
        assert post(port, "/api/projects", {"name": "p"})[0] == 201
        for body in ({"options": {"bogus": 1}}, {"options": {"answer": ["a=b"]}}, {"answers": {"Bad Key": "x"}}, {"answers": {"a": "x\ny"}},
                     {"options": {"llm_budget_usd": "lots"}}, {"options": {"trust_host": "-x"}}, {"extra": 1}, {"options": [1]}):
            status, _h, data = post(port, "/api/projects/p/run", body)
            assert status == 400 and one_line(data), body
        status, _h, data = post(port, "/api/projects/p/review", {"answers": {"a": "b"}})
        assert status == 400 and one_line(data)
        status, _h, data = post(port, "/api/projects/p/stage-reports", {"options": {"online": True}})
        assert status == 400 and "online" in one_line(data)
        status, _h, _d = post(port, "/api/projects/nope/run", {})
        assert status == 404
        assert not (tmp_path / "root" / "p" / "gui").exists(), "nothing started, no log written"
    finally:
        running.stop()


def test_one_active_run_per_project_is_a_409(tmp_path: Path):
    running = Running(tmp_path / "root", runs=RunManager(python_args=SLEEPER))
    port = running.port
    try:
        for name in ("a", "b"):
            assert post(port, "/api/projects", {"name": name})[0] == 201
        first = start_run(port, "a")
        for kind in ("run", "review", "stage-reports"):
            status, _h, body = post(port, f"/api/projects/a/{kind}", {})
            assert status == 409 and first in one_line(body), kind
        assert get_json(port, "/api/projects/a")["active_run"] == first
        assert get_json(port, "/api/projects/a/runs")["active"] == first
        other = start_run(port, "b")  # another project runs at the same time
        assert wait_run(port, "a", first)["exit_code"] == 0 and wait_run(port, "b", other)["exit_code"] == 0
        assert start_run(port, "a", "review") == "002-review", "the slot is free again"
        wait_run(port, "a", "002-review")
    finally:
        running.stop()


# --------------------------------------------------------------------------- the LLM panel


def test_llm_providers_come_from_describe_providers_and_never_carry_a_key(shared, fake: FakeClaudeCli, monkeypatch: pytest.MonkeyPatch):
    running, *_ = shared
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    status, _h, body = get(running.port, "/api/projects/divider")
    assert status == 200 and SECRET.encode() not in body and b"GUI-TEST-SECRET" not in body
    rows = {r["name"]: r for r in json.loads(body)["llm"]["providers"]}
    assert rows["openrouter"] == {"name": "openrouter", "available": True, "reason": "OPENROUTER_API_KEY set", "billing": "per_call", "logged_in": None}
    assert rows["claude"]["available"] is True and rows["claude"]["billing"] == "subscription" and rows["claude"]["logged_in"] is True
    assert str(fake.exe) in rows["claude"]["reason"] and fake.prompt_calls() == [], "describing a provider never calls the model"
    for path in ("/api/projects", "/api/projects/divider/runs", "/preview/divider/report.html", "/preview/divider/ir.json"):
        assert SECRET.encode() not in get(running.port, path)[2], path
    monkeypatch.delenv("OPENROUTER_API_KEY")
    fake.set_auth({"loggedIn": False})
    rows = {r["name"]: r for r in get_json(running.port, "/api/projects/divider")["llm"]["providers"]}
    assert rows["openrouter"]["available"] is False and rows["openrouter"]["reason"] == "OPENROUTER_API_KEY not set"
    assert rows["claude"]["available"] is False and rows["claude"]["logged_in"] is False


def test_provider_rows_are_reused_for_their_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ai_eda.llm.providers as providers

    calls: list[int] = []
    real = providers.describe_providers

    def counted(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(providers, "describe_providers", counted)
    running = Running(tmp_path / "root", providers_ttl=60)
    try:
        assert post(running.port, "/api/projects", {"name": "p"})[0] == 201
        for _ in range(3):
            get_json(running.port, "/api/projects/p")
        assert len(calls) == 1
    finally:
        running.stop()


def test_llm_runs_through_the_api_log_usage_and_respect_the_pin(tmp_path: Path, fake: FakeClaudeCli):
    running = Running(tmp_path / "root")
    port = running.port
    try:
        assert post(port, "/api/projects", {"name": "llm", "request": LLM_REQUEST})[0] == 201
        fake.queue(success_structured(EXTRACTION))
        run_id = start_run(port, "llm", options={"llm": "claude", "llm_claude_cli": str(fake.exe), "no_pdf": True})
        status = wait_run(port, "llm", run_id)
        usage = next(line for line in status["log_tail"].splitlines() if line.startswith("LLM usage:"))
        assert status["exit_code"] == 1 and usage.startswith("LLM usage: 1 served call(s)") and "(not charged)" in usage
        project = get_json(port, "/api/projects/llm")
        assert project["llm"]["pinned_model"] == CLAUDE_SPEC
        assert any(q["key"] == "confirm_requirements" for q in project["questions"]["required"])
        # another model without the checkbox: the CLI refuses with its Korean sentence, exit 2, before any call
        ir_path = tmp_path / "root" / "llm" / "ir.json"
        before = ir_path.read_bytes()
        run_id = start_run(port, "llm", options={"llm": "claude", "llm_model": "claude:model-e", "llm_claude_cli": str(fake.exe)})
        status = wait_run(port, "llm", run_id)
        assert status["exit_code"] == 2 and model_pin_refusal(CLAUDE_SPEC, "claude:model-e") in status["log_tail"].splitlines()
        assert len(fake.prompt_calls()) == 1 and ir_path.read_bytes() == before
    finally:
        running.stop()


# --------------------------------------------------------------------------- no outbound connection


def test_the_gui_process_opens_no_outbound_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every socket connect of this process to a non-loopback address raises; the whole API is driven, including a real run (its own process)."""
    ir, lib, _ir_path = build_project(tmp_path / "root")
    attempts: list[object] = []
    loopback = {"127.0.0.1", "localhost", "::1"}
    real_create, real_connect, real_connect_ex = socket.create_connection, socket.socket.connect, socket.socket.connect_ex

    def refuse(address: object) -> None:
        attempts.append(address)
        raise OSError(f"outbound connection to {address!r} refused by the test")

    def create_connection(address, *a, **kw):
        if address[0] not in loopback:
            refuse(address)
        return real_create(address, *a, **kw)

    def connect(self, address):
        if isinstance(address, tuple) and address[0] not in loopback:
            refuse(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and address[0] not in loopback:
            refuse(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    running = Running(tmp_path / "root", library=lib)
    port = running.port
    try:
        from urllib.parse import quote

        assert get(port, "/")[0] == 200 and get(port, "/static/app.js")[0] == 200
        assert post(port, "/api/projects", {"name": "second", "request": "12 V 입력, 5 V 출력 분압기"})[0] == 201
        assert [p["name"] for p in get_json(port, "/api/projects")["projects"]] == ["divider", "second"]
        project = get_json(port, "/api/projects/divider")
        get_json(port, "/api/projects/second")
        paths = ["/preview/divider/schematic.svg", "/preview/divider/board.svg", "/preview/divider/report.html", "/preview/divider/bom.json",
                 "/preview/divider/cpl.json", "/preview/divider/artifacts.json", "/preview/divider/ir.json", "/files/divider.zip", "/files/divider/bom.csv"]
        paths += [f"/preview/divider/waveform/{w}.svg" for w in project["previews"]["waveforms"]]
        paths += [f"/preview/divider/reports/{quote(f)}" for r in project["previews"]["reports"] for f in (r["md"], r["html"], r["pdf"]) if f]
        for path in paths:
            assert get(port, path)[0] == 200, path
        run_id = start_run(port, "second", options={"no_pdf": True})
        status = wait_run(port, "second", run_id)
        # the child reached the pipeline (not an interpreter that could not import ai_eda, which also exits 1)
        assert status["exit_code"] == 1 and "BLOCKED - answer these to continue" in status["log_tail"], status["log_tail"][-2000:]
        assert [q["key"] for q in get_json(port, "/api/projects/second")["questions"]["required"]] == ["application", "jurisdiction"]
        assert get_json(port, "/api/projects/second/runs")["runs"][0]["id"] == run_id
    finally:
        running.stop()
    assert attempts == []


# --------------------------------------------------------------------------- the CLI


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_ai_eda_gui_starts_prints_the_url_and_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[int, bytes]] = []
    opened: list[str] = []

    def serve_once(self: GuiServer) -> None:
        """Serve for real in a thread, answer one request, then stop the way Ctrl-C does."""
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        try:
            status, _h, body = get(self.port, "/api/projects")
            seen.append((status, body))
        finally:
            self._server.shutdown()
            thread.join(timeout=10)
        raise KeyboardInterrupt

    monkeypatch.setattr(GuiServer, "serve_forever", serve_once)
    monkeypatch.setattr(server_module.webbrowser, "open", lambda url: opened.append(url) or True)
    root = tmp_path / "projects"
    code, out, err = _cli("gui", "--root", str(root), "--port", "0", "--open")
    assert code == 0 and err == ""
    url = re.search(r"http://127\.0\.0\.1:(\d+)/", out)
    assert url is not None and int(url.group(1)) > 0 and "127.0.0.1 only" in out and str(root.resolve()) in out
    assert opened == [url.group(0)]
    assert seen and seen[0][0] == 200 and json.loads(seen[0][1]) == {"root": str(root.resolve()), "projects": []}
    # without --open nothing is opened; the default root is projects under the cwd
    monkeypatch.chdir(tmp_path)
    code, out, _err = _cli("gui", "--port", "0")
    assert code == 0 and len(opened) == 1 and str((tmp_path / "projects").resolve()) in out


@pytest.mark.parametrize("port", ["70000", "-1", "65536", "http"])
def test_a_port_outside_0_65535_is_a_usage_error_not_a_traceback(tmp_path: Path, port: str):
    for command in (("gui", "--root", str(tmp_path)), ("serve", str(tmp_path / "ir.json"))):
        err = io.StringIO()
        with redirect_stderr(err), pytest.raises(SystemExit) as exit_info:
            cli_main([*command, "--port", port])
        assert exit_info.value.code == 2 and "Traceback" not in err.getvalue(), command
        assert "--port" in err.getvalue() and ("0-65535" in err.getvalue() or "not a port number" in err.getvalue())
    assert not any(tmp_path.iterdir()), "nothing was served or written"
    if port.lstrip("-").isdigit():  # the function itself (no parser in front) also answers 2 with one line
        with redirect_stderr(io.StringIO()) as err:
            assert serve_gui(tmp_path, int(port)) == 2
        assert "could not listen on 127.0.0.1" in err.getvalue() and "Traceback" not in err.getvalue()


def test_serve_gui_reports_a_taken_port(tmp_path: Path):
    running = Running(tmp_path)
    try:
        err = io.StringIO()
        with redirect_stderr(err):
            assert serve_gui(tmp_path, running.port) == 2
        assert "could not listen on 127.0.0.1" in err.getvalue()
    finally:
        running.stop()


def test_request_path_and_header_helpers():
    assert split_path("/") == [] and split_path("/api/projects?x=1") == ["api", "projects"]
    assert split_path("/preview/p/reports/%EC%9D%B4%EB%A1%A0.md") == ["preview", "p", "reports", "이론.md"]
    for bad in ("/a//b", "/a/", "/%2F", "/a%00b", "/%ff", "relative"):
        assert split_path(bad) is None, bad
    assert content_disposition("attachment", "01_이론_보고서.pdf") == (
        "attachment; filename=\"01_______.pdf\"; filename*=UTF-8''01_%EC%9D%B4%EB%A1%A0_%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf"
    )
    assert content_disposition("inline", 'a"b.txt') == "inline; filename=\"a_b.txt\"; filename*=UTF-8''a%22b.txt"
