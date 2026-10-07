"""``markdown_to_html``, browser discovery and headless printing (``ai_eda/report/pdf.py``).

The converter tests are pure and pin the security property of the page:
every text run of the Markdown (which comes from IR strings) is escaped,
only a figure's own SVG is inserted raw, and the page names no external
resource. The discovery tests isolate :func:`find_browser` from the real
machine (every environment variable and PATH it reads points into
``tmp_path`` and the fixed roots are emptied), except one gated test that
proves the Playwright default root finds the real Chromium without any
environment variable where that root exists. :func:`html_to_pdf` is
exercised with fake browsers written as scripts (a ``.cmd`` wrapper makes
them executable on Windows) and once against a real Chromium / Chrome /
Edge when one is found (skipped otherwise).
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import ai_eda.report.pdf as pdf
from ai_eda.report import theory_report
from ai_eda.report.pdf import (
    MISSING_FIGURE,
    NO_BROWSER_REASON,
    PRINT_FLAGS,
    PdfResult,
    browser_version,
    find_browser,
    html_to_pdf,
    markdown_to_html,
)

EXTERNAL_RE = re.compile(r"(?:src|href)\s*=\s*['\"]?\s*https?://", re.IGNORECASE)


@dataclass(frozen=True)
class Fig:
    """Shaped like ``ai_eda.report.figures.Figure`` without importing it (the modules stay decoupled)."""

    id: str
    title: str
    caption: str
    svg: str


WAVE = Fig("wave", "출력 파형", "v(OUT), 10 ms 구간", '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text>x</text></svg>')


def _body(html: str) -> str:
    return html.split("<main>", 1)[1].split("</main>", 1)[0].strip()


def _md(md: str, **figs: Fig) -> str:
    return _body(markdown_to_html(md, title="t", figures=figs))


# --------------------------------------------------------------------------- markdown_to_html


def test_document_skeleton_and_print_css():
    html = markdown_to_html("# 제목\n\n본문.\n", title="이론 보고서: <p>&")
    assert html.startswith("<!doctype html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n")
    assert "<title>이론 보고서: &lt;p&gt;&amp;</title>" in html
    assert html.count("<style>") == 1 and "@page { size: A4; margin: 16mm; }" in html
    assert "page-break-after: avoid" in html and "page-break-inside: avoid" in html
    assert '"Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "WenQuanYi Zen Hei", sans-serif' in html
    assert "figure svg { max-width: 100%;" in html and "1px solid #d8d6cf" in html and "font-size: 10.5pt" in html
    assert html.endswith("</main>\n</body>\n</html>\n")
    assert "\r" not in html
    assert _body(html) == "<h1>제목</h1>\n<p>본문.</p>"


def test_deterministic_and_line_ending_neutral():
    a = markdown_to_html("# a\n\n- x\n- y\n", title="t", figures={"wave": WAVE})
    assert a == markdown_to_html("# a\n\n- x\n- y\n", title="t", figures={"wave": WAVE})
    assert a == markdown_to_html("# a\r\n\r\n- x\r\n- y\r\n", title="t", figures={"wave": WAVE})


def test_headings_paragraphs_and_rules():
    assert _md("# 하나\n## 둘\n### 셋\n#### 넷\n") == "<h1>하나</h1>\n<h2>둘</h2>\n<h3>셋</h3>\n<h4>넷</h4>"
    assert _md("첫 줄\n둘째 줄\n\n다음 문단\n") == "<p>첫 줄\n둘째 줄</p>\n<p>다음 문단</p>"
    assert _md("위\n\n---\n\n아래\n") == "<p>위</p>\n<hr>\n<p>아래</p>"
    assert _md("a --- b\n") == "<p>a --- b</p>"  # a rule is a line of its own
    assert _md("#없음\n") == "<p>#없음</p>"  # a heading needs the space


def test_lists_flat_nested_ordered_and_loose():
    assert _md("- a\n- b\n") == "<ul><li>a</li><li>b</li></ul>"
    assert _md("- a\n  - b\n  - c\n- d\n") == "<ul><li>a<ul><li>b</li><li>c</li></ul></li><li>d</li></ul>"
    assert _md("1. x\n2. y\n") == "<ol><li>x</li><li>y</li></ol>"
    assert _md("3. x\n4. y\n") == '<ol start="3"><li>x</li><li>y</li></ol>'
    assert _md("- 세부 확인:\n  - symbol: PASS\n  - footprint: PASS\n") == "<ul><li>세부 확인:<ul><li>symbol: PASS</li><li>footprint: PASS</li></ul></li></ul>"
    assert _md("- a\n  continued\n- b\n") == "<ul><li>a\ncontinued</li><li>b</li></ul>"
    assert _md("- a\nlazy\n- b\n") == "<ul><li>a\nlazy</li><li>b</li></ul>"
    assert _md("- a\n\n- b\n") == "<ul><li><p>a</p></li><li><p>b</p></li></ul>"
    assert _md("-1 V 로 둔다\n") == "<p>-1 V 로 둔다</p>"  # not a bullet


def test_list_item_with_code_block_and_continuation_paragraph():
    md = (
        "- IPC-2221 (폭 0.4 mm):\n\n      I = k · ΔT^0.44 · A^0.725\n      I = 1.231 A\n\n  여유는 246 배입니다.\n\n"
        "- 간격 0.1 mm:\n  두 번째 줄.\n"
    )
    assert _md(md) == (
        "<ul><li><p>IPC-2221 (폭 0.4 mm):</p><pre><code>I = k · ΔT^0.44 · A^0.725\nI = 1.231 A</code></pre><p>여유는 246 배입니다.</p></li>"
        "<li><p>간격 0.1 mm:\n두 번째 줄.</p></li></ul>"
    )


def test_table_with_alignment_escaped_pipes_and_short_rows():
    md = "| 키 | 값 | 비고 |\n|:---|:---:|---:|\n| `a` | 1 | x \\| y |\n| b | 2 |\n"
    assert _md(md) == (
        "<table><thead><tr><th class=\"align-left\">키</th><th class=\"align-center\">값</th><th class=\"align-right\">비고</th></tr></thead>"
        "<tbody><tr><td class=\"align-left\"><code>a</code></td><td class=\"align-center\">1</td><td class=\"align-right\">x | y</td></tr>"
        "<tr><td class=\"align-left\">b</td><td class=\"align-center\">2</td><td class=\"align-right\"></td></tr></tbody></table>"
    )
    plain = _md("| a | b |\n|---|---|\n| **합계** | 64 |\n\n다음.\n")
    assert plain == "<table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td><strong>합계</strong></td><td>64</td></tr></tbody></table>\n<p>다음.</p>"
    assert _md("| a | b |\n| 1 | 2 |\n") == "<p>| a | b |\n| 1 | 2 |</p>"  # no separator row: text


def test_code_blocks_keep_angle_brackets_literally():
    fenced = _md("```\nif a < b and c > d: x = &y\n  <script>\n```\n\n뒤.\n")
    assert fenced == "<pre><code>if a &lt; b and c &gt; d: x = &amp;y\n  &lt;script&gt;</code></pre>\n<p>뒤.</p>"
    indented = _md("판정식:\n\n    |측정값 − 공칭값| ≤ max(a, b)\n\n    둘째 블록 <x>\n\n끝.\n")
    assert indented == "<p>판정식:</p>\n<pre><code>|측정값 − 공칭값| ≤ max(a, b)\n\n둘째 블록 &lt;x&gt;</code></pre>\n<p>끝.</p>"
    assert _md("```\nnever closed <\n") == "<pre><code>never closed &lt;</code></pre>"


def test_inline_bold_italic_and_code():
    assert _md("**굵게** 와 *기울임* 그리고 `code`\n") == "<p><strong>굵게</strong> 와 <em>기울임</em> 그리고 <code>code</code></p>"
    assert _md("**트랜지스터 모델** `.model Q NPN (TR=200n)` — 값\n") == "<p><strong>트랜지스터 모델</strong> <code>.model Q NPN (TR=200n)</code> — 값</p>"
    assert _md("A* 탐색으로 잇습니다 (`calc.tran_*`)\n") == "<p>A* 탐색으로 잇습니다 (<code>calc.tran_*</code>)</p>"  # a lone star stays
    assert _md("`**not bold**` and `*not italic*`\n") == "<p><code>**not bold**</code> and <code>*not italic*</code></p>"
    assert _md("V_cc − V_BE 와 R_b\n") == "<p>V_cc − V_BE 와 R_b</p>"  # underscores are never emphasis
    assert _md("## 제목 **강조** `코드`\n") == "<h2>제목 <strong>강조</strong> <code>코드</code></h2>"


def test_markdown_text_is_escaped_everywhere():
    hostile = "<script>alert(1)</script> & \"q\" 'q'"
    html = markdown_to_html(
        f"# {hostile}\n\n{hostile}\n\n- {hostile}\n\n| {hostile} |\n|---|\n| {hostile} |\n\n`{hostile}`\n\n**{hostile}**\n\n```\n{hostile}\n```\n",
        title=hostile,
    )
    assert "<script" not in html and "alert(1)" in html
    assert html.count("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;q&quot; &#x27;q&#x27;") == 9
    assert "<h1>&lt;script&gt;" in html and "<code>&lt;script&gt;" in html and "<strong>&lt;script&gt;" in html
    assert "<td>&lt;script&gt;" in html and "<th>&lt;script&gt;" in html and "<li>&lt;script&gt;" in html
    assert "<title>&lt;script&gt;" in html


def test_links_images_and_html_are_text_not_markup():
    html = markdown_to_html("[x](https://e.invalid/p) ![y](https://e.invalid/a.png) <a href=\"https://e.invalid\">z</a> <img src=x>\n", title="t")
    assert "<a " not in html and "<img" not in html
    assert "[x](https://e.invalid/p)" in html and "&lt;a href=&quot;https://e.invalid&quot;&gt;z&lt;/a&gt;" in html
    assert not EXTERNAL_RE.search(html) and "<link" not in html and "<script" not in html and "@import" not in html


def test_figure_placeholder_inserts_the_svg_raw_and_drops_the_caption_line():
    md = "## 파형\n\n![fig](fig:wave)\n*v(OUT), 10 ms 구간*\n\n다음 문단.\n"
    body = _md(md, wave=WAVE)
    assert body == (
        "<h2>파형</h2>\n<figure>" + WAVE.svg + "<figcaption>출력 파형 — v(OUT), 10 ms 구간</figcaption></figure>\n<p>다음 문단.</p>"
    )
    # a blank line between placeholder and caption, and a caption-less figure
    assert _md("![fig](fig:wave)\n\n*캡션*\n\n뒤.\n", wave=WAVE).endswith("</figcaption></figure>\n<p>뒤.</p>")
    bare = Fig("b", "제목 <b>", "", "<svg/>")
    assert _md("![fig](fig:b)\n\n뒤.\n", b=bare) == "<figure><svg/><figcaption>제목 &lt;b&gt;</figcaption></figure>\n<p>뒤.</p>"
    # the figure carries the caption; a following non-italic paragraph is kept
    assert _md("![fig](fig:wave)\n본문.\n", wave=WAVE).endswith("</figure>\n<p>본문.</p>")


def test_figcaption_renders_code_spans_and_escapes_like_any_text_run():
    """A caption is Markdown like the italic line it replaces: `code` becomes <code>, the text is escaped once, and a stray * never opens an italic run."""
    fig = Fig("t", "제목 `x` <b>", "해석 `tran`: `tran 5.00u 20m 10m uic` (`calc.astable.f`) & *별*", "<svg/>")
    body = _md("![fig](fig:t)\n", t=fig)
    assert body == (
        "<figure><svg/><figcaption>제목 <code>x</code> &lt;b&gt; — 해석 <code>tran</code>: <code>tran 5.00u 20m 10m uic</code> "
        "(<code>calc.astable.f</code>) &amp; ∗별∗</figcaption></figure>"
    )
    assert "`" not in body and "<em>" not in body


def test_unknown_figure_id_renders_a_visible_note_and_keeps_the_caption():
    body = _md("![fig](fig:nope<x>)\n*캡션*\n", wave=WAVE)
    assert body == f'<p class="missing-figure">{MISSING_FIGURE}: nope&lt;x&gt;</p>\n<p><em>캡션</em></p>'
    assert "<svg" not in body
    # a placeholder inside a paragraph is text
    assert _md("앞 ![fig](fig:wave) 뒤\n", wave=WAVE) == "<p>앞 ![fig](fig:wave) 뒤</p>"


def test_figures_mapping_is_optional():
    assert _md("![fig](fig:x)\n") == f'<p class="missing-figure">{MISSING_FIGURE}: x</p>'
    assert _body(markdown_to_html("본문\n", title="t")) == "<p>본문</p>"


def test_no_external_resource_even_with_figures():
    html = markdown_to_html("# a\n\n![fig](fig:wave)\n\n| a |\n|---|\n| https://e.invalid |\n", title="t", figures={"wave": WAVE})
    assert not EXTERNAL_RE.search(html)
    assert "<link" not in html and "<script src" not in html and "<script" not in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html


def test_real_theory_report_converts(tmp_path: Path):
    """The astable template's theory report (fence, ordered list, tables, bold, code) renders without a stray marker."""
    from tests.test_circuit_templates import ASTABLE, BASE
    from tests.test_stage_reports import _build

    ir, _lib = _build(tmp_path, "osc", {**BASE, **ASTABLE})
    md = theory_report(ir)
    html = markdown_to_html(md, title="이론 보고서", figures={})
    body = _body(html)
    assert body.startswith("<h1>이론 보고서: ")
    assert "<pre><code>VCC ---+" in body and "<ol><li>" in body and body.count("<table>") >= 3
    assert "<strong>" in body and "<code>calc.astable.c_for_frequency</code>" in body
    assert "|---|" not in body and "**" not in body and "```" not in body and "<script" not in body
    assert not EXTERNAL_RE.search(html)
    assert html == markdown_to_html(md, title="이론 보고서", figures={})


# --------------------------------------------------------------------------- find_browser


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Nothing of the real machine reaches find_browser: every env var and PATH it reads points into tmp_path, the fixed roots are empty."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv(pdf.BROWSER_ENV, raising=False)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv(pdf.PLAYWRIGHT_ENV, str(empty))
    for name in pdf.WINDOWS_ROOT_ENVS:
        monkeypatch.setenv(name, str(empty))
    monkeypatch.setattr(pdf, "PLAYWRIGHT_DEFAULT_ROOTS", ())
    monkeypatch.setattr(pdf, "MACOS_CANDIDATES", ())
    return tmp_path


def _executable(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _exe_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def test_find_browser_nothing_found(isolated: Path):
    assert find_browser() is None


def test_find_browser_env_var_existing_or_missing(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    exe = _executable(isolated / "custom" / "my-chrome")
    monkeypatch.setenv(pdf.BROWSER_ENV, str(exe))
    assert find_browser() == exe
    monkeypatch.setenv(pdf.BROWSER_ENV, str(isolated / "custom" / "missing"))
    assert find_browser() is None  # a missing explicit path is not a browser; nothing else is on this machine
    monkeypatch.setenv(pdf.BROWSER_ENV, str(isolated / "custom"))
    assert find_browser() is None  # a directory is not one either


def test_find_browser_on_path_in_name_order(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    bindir = isolated / "bin"
    chrome = _executable(bindir / _exe_name("google-chrome"))
    monkeypatch.setenv("PATH", str(bindir))
    assert find_browser() == chrome
    chromium = _executable(bindir / _exe_name("chromium"))
    assert find_browser() == chromium  # PATH_NAMES order, not directory order
    explicit = _executable(isolated / "x" / "edge")
    monkeypatch.setenv(pdf.BROWSER_ENV, str(explicit))
    assert find_browser() == explicit  # the explicit variable wins


def test_find_browser_playwright_tree_highest_build(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    root = isolated / "pw"
    old = _executable(root / "chromium-1100" / "chrome-linux" / "chrome")
    new = _executable(root / "chromium-1194" / "chrome-linux" / "chrome")
    _executable(root / "chromium_headless_shell-1194" / "chrome-linux" / "headless_shell")
    (root / "chromium-1300").mkdir()  # a build directory without the binary
    monkeypatch.setenv(pdf.PLAYWRIGHT_ENV, str(root))
    assert find_browser() == new and old != new
    monkeypatch.setenv(pdf.PLAYWRIGHT_ENV, str(isolated / "empty"))
    assert find_browser() is None
    monkeypatch.setattr(pdf, "PLAYWRIGHT_DEFAULT_ROOTS", (isolated / "nowhere", root))
    assert find_browser() == new  # the default root, no env var
    win_root = isolated / "pw-win"
    win = _executable(win_root / "chromium-1194" / "chrome-win" / "chrome.exe")
    monkeypatch.setenv(pdf.PLAYWRIGHT_ENV, str(win_root))
    assert find_browser() == win


def test_find_browser_windows_and_macos_paths(isolated: Path, monkeypatch: pytest.MonkeyPatch):
    pf = isolated / "pf"
    chrome = _executable(pf / "Google" / "Chrome" / "Application" / "chrome.exe")
    monkeypatch.setenv("ProgramFiles", str(pf))
    assert find_browser() == chrome
    monkeypatch.setenv("ProgramFiles", str(isolated / "empty"))
    edge = _executable(isolated / "lad" / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    monkeypatch.setenv("LOCALAPPDATA", str(isolated / "lad"))
    assert find_browser() == edge
    monkeypatch.setenv("LOCALAPPDATA", str(isolated / "empty"))
    mac = _executable(isolated / "Applications" / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome")
    monkeypatch.setattr(pdf, "MACOS_CANDIDATES", (mac,))
    assert find_browser() == mac


_PW_REAL = [p for root in pdf.PLAYWRIGHT_DEFAULT_ROOTS for pattern in pdf.PLAYWRIGHT_GLOBS for p in root.glob(pattern) if p.is_file()]


@pytest.mark.skipif(not _PW_REAL, reason="no Chromium under the Playwright default root on this machine")
def test_find_browser_playwright_default_root_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The real Playwright Chromium is found by the default-root rule alone: no AI_EDA_BROWSER, no PLAYWRIGHT_BROWSERS_PATH, nothing on PATH."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv(pdf.BROWSER_ENV, raising=False)
    monkeypatch.delenv(pdf.PLAYWRIGHT_ENV, raising=False)
    monkeypatch.setenv("PATH", str(empty))
    for name in pdf.WINDOWS_ROOT_ENVS:
        monkeypatch.setenv(name, str(empty))
    found = find_browser()
    assert found is not None and found in _PW_REAL and found.parent.parent.parent in pdf.PLAYWRIGHT_DEFAULT_ROOTS


# --------------------------------------------------------------------------- html_to_pdf with fake browsers

MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n110\n%%EOF\n"
)
NOISE = "[1:1:0000/000000.000000:ERROR:dbus/bus.cc:408] Failed to connect to the bus: no such file\n"

FAKE_WRITES = f"""import json, sys
if "--version" in sys.argv:
    print("FakeBrowser 1.0"); sys.exit(0)
target = [a for a in sys.argv[1:] if a.startswith("--print-to-pdf=")][0][len("--print-to-pdf="):]
open(target, "wb").write({MINIMAL_PDF!r})
open(target + ".argv.json", "w", encoding="utf-8").write(json.dumps(sys.argv[1:]))
sys.stderr.write({NOISE!r} + "%d bytes written to file %s\\n" % ({len(MINIMAL_PDF)}, target))
"""
FAKE_FAILS = f"""import sys
sys.stderr.write({NOISE!r} + "Failed to open file: boom\\nsecond line\\nthird line\\nfourth line\\n")
sys.exit(1)
"""
FAKE_WRITES_NOTHING = "import sys\nsys.exit(0)\n"
FAKE_WRITES_EMPTY = """import sys
target = [a for a in sys.argv[1:] if a.startswith("--print-to-pdf=")][0][len("--print-to-pdf="):]
open(target, "wb").close()
"""
FAKE_SLEEPS = "import time\ntime.sleep(5)\n"


def _fake_browser(folder: Path, name: str, body: str) -> Path:
    """A fake browser: a Python script run through the current interpreter (a .cmd wrapper on Windows, a shebang elsewhere)."""
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    if sys.platform == "win32":
        wrapper = folder / f"{name}.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        return wrapper
    exe = folder / name
    return _executable(exe, f"#!{sys.executable}\n{body}")


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "01_이론_보고서.html"
    path.parent.mkdir()
    path.write_text(markdown_to_html("# 한글 보고서\n\n| 키 | 값 |\n|---|---|\n| f | 1 kHz |\n", title="한글"), encoding="utf-8", newline="\n")
    return path


def test_html_to_pdf_ok_with_a_browser_that_writes(tmp_path: Path, html_file: Path):
    fake = _fake_browser(tmp_path / "fake", "writes", FAKE_WRITES)
    out = tmp_path / "out" / "sub" / "01_이론_보고서.pdf"  # the folder does not exist yet
    res = html_to_pdf(html_file, out, browser=fake)
    assert res == PdfResult(True, out, None, fake)
    assert out.read_bytes() == MINIMAL_PDF
    argv = json.loads(Path(str(out) + ".argv.json").read_text(encoding="utf-8"))
    for flag in PRINT_FLAGS:
        assert flag in argv
    # the browser runs with name resolution and the proxy off: a file: URL needs neither, and without them a headless
    # Chromium's own services opened outbound sockets on every print (measured) in spite of --disable-background-networking
    assert "--no-proxy-server" in argv and "--host-resolver-rules=MAP * ~NOTFOUND" in argv and "--disable-background-networking" in argv
    assert argv[-1] == html_file.resolve().as_uri() and argv[-1].startswith("file:///")
    assert argv[-2] == f"--print-to-pdf={out.resolve()}"
    assert sum(a.startswith("--user-data-dir=") for a in argv) == 1
    assert browser_version(fake) == "FakeBrowser 1.0"


def test_html_to_pdf_failure_reasons(tmp_path: Path, html_file: Path):
    out = tmp_path / "o.pdf"
    fails = _fake_browser(tmp_path / "fake", "fails", FAKE_FAILS)
    res = html_to_pdf(html_file, out, browser=fails)
    assert res.ok is False and res.path is None and res.browser == fails
    assert res.reason == "exit code 1: Failed to open file: boom | second line | third line"  # dbus noise dropped, three lines kept
    assert not out.exists()
    nothing = _fake_browser(tmp_path / "fake", "nothing", FAKE_WRITES_NOTHING)
    res = html_to_pdf(html_file, out, browser=nothing)
    assert res == PdfResult(False, None, "no PDF written (no stderr)", nothing)
    empty = _fake_browser(tmp_path / "fake", "empty", FAKE_WRITES_EMPTY)
    res = html_to_pdf(html_file, out, browser=empty)
    assert res.ok is False and res.reason == "no PDF written (no stderr)"
    assert browser_version(nothing) is None  # prints nothing
    assert browser_version(tmp_path / "missing") is None


def test_html_to_pdf_removes_a_stale_pdf_first(tmp_path: Path, html_file: Path):
    out = tmp_path / "o.pdf"
    out.write_bytes(b"%PDF-1.4 stale")
    nothing = _fake_browser(tmp_path / "fake", "nothing", FAKE_WRITES_NOTHING)
    res = html_to_pdf(html_file, out, browser=nothing)
    assert res.ok is False and res.reason.startswith("no PDF written") and not out.exists()


def test_html_to_pdf_timeout(tmp_path: Path, html_file: Path):
    sleeps = _fake_browser(tmp_path / "fake", "sleeps", FAKE_SLEEPS)
    res = html_to_pdf(html_file, tmp_path / "o.pdf", browser=sleeps, timeout_s=1)
    assert res == PdfResult(False, None, "timed out after 1 s", sleeps)


def test_html_to_pdf_without_browser_or_html(tmp_path: Path, html_file: Path, isolated: Path):
    res = html_to_pdf(html_file, tmp_path / "o.pdf")
    assert res == PdfResult(False, None, NO_BROWSER_REASON, None)
    fake = _fake_browser(tmp_path / "fake", "writes", FAKE_WRITES)
    res = html_to_pdf(tmp_path / "in" / "missing.html", tmp_path / "o.pdf", browser=fake)
    assert res == PdfResult(False, None, "HTML file not found: missing.html", fake)
    res = html_to_pdf(html_file, tmp_path / "o.pdf", browser=tmp_path / "not-a-browser")
    assert res.ok is False and res.reason.startswith("cannot run not-a-browser") and res.browser == tmp_path / "not-a-browser"


# --------------------------------------------------------------------------- the real browser


@pytest.mark.skipif(find_browser() is None, reason="no Chromium / Chrome / Edge found")
def test_real_browser_prints_korean_html(tmp_path: Path, html_file: Path):
    browser = find_browser()
    assert browser is not None
    out = tmp_path / "out" / "한글.pdf"
    res = html_to_pdf(html_file, out, browser=browser)
    assert res.ok, res.reason
    assert res.path == out and res.browser == browser and res.reason is None
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1024
    version = browser_version(browser)
    assert version is None or version.strip()  # best effort; Chrome on Windows prints nothing
