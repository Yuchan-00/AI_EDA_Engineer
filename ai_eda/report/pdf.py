"""Self-contained HTML and PDF renderings of the Korean stage reports.

Invariant: the HTML is a *view* like the Markdown it is rendered from
(:mod:`ai_eda.report.stages`): :func:`markdown_to_html` computes no status,
registers no artifact, saves nothing and adds no wall-clock and no absolute
path of its own, so the same Markdown, title and figures give a
byte-identical page. The page loads no external resource of any kind - no
``<link>``, no ``<script>``, no ``@import``, no ``src=`` / ``href=`` at all:
fonts come from a local font stack and figures are inline ``<svg>``. Every
text node and attribute value goes through exactly one escape function,
:func:`ai_eda.report.html.esc`; the only markup inserted raw is the ``svg``
of a figure, which our own code generates (:mod:`ai_eda.report.figures`),
never the Markdown text, which comes from IR strings and may contain
``<script>``, ``&``, quotes or anything else.

The converter understands the Markdown subset the report builders emit and
nothing more: ``#``..``######`` headings, paragraphs, ``-`` / ``1.`` lists
(nested by indentation; an item may hold continuation paragraphs and a code
block), pipe tables whose second row is the ``---`` separator (``:---:``
alignment, ``\\|`` inside a cell), fenced and four-space-indented code
blocks, inline ``code``, ``**bold**``, ``*italic*``, ``---`` rules and the
figure placeholder line ``![fig](fig:<id>)`` (an unknown id renders a
visible :data:`MISSING_FIGURE` note). Everything else - HTML tags, links,
images, underscores - is text and is escaped.

The PDF is the HTML printed by a locally installed Chromium / Chrome / Edge
in headless mode, discovered at runtime like ``kicad-cli``
(:func:`find_browser`) and never bundled. :func:`html_to_pdf` launches that
local program on a local file with name resolution and the proxy switched
off (:data:`PRINT_FLAGS`), so neither our code nor the browser's own
services can reach any host (measured: zero ``connect()`` calls); it is not
an :class:`~ai_eda.security.ExternalAction`, like ``serve``. The PDF bytes
carry the browser's own creation date (``/CreationDate``) and producer, so
they are a derived document like a SPICE rawfile, not deterministic bytes,
and are never hashed into the IR; the HTML beside them is the deterministic
one. No browser found means no PDF and a reason, never an exception.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from ai_eda.report.html import esc

#: what the page prints in place of a placeholder whose figure id is unknown (followed by ``: <id>``)
MISSING_FIGURE = "그림 없음"
#: the environment variable naming the browser binary explicitly (it must exist)
BROWSER_ENV = "AI_EDA_BROWSER"
#: Playwright's browser root override (its own convention)
PLAYWRIGHT_ENV = "PLAYWRIGHT_BROWSERS_PATH"
#: binary names looked up on PATH, in order
PATH_NAMES: tuple[str, ...] = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "msedge")
#: Playwright roots searched after ``$PLAYWRIGHT_BROWSERS_PATH``
PLAYWRIGHT_DEFAULT_ROOTS: tuple[Path, ...] = (Path("/opt/pw-browsers"),)
#: where a Playwright root keeps its Chromium binary (highest build number wins)
PLAYWRIGHT_GLOBS: tuple[str, ...] = ("chromium-*/chrome-linux/chrome", "chromium-*/chrome-win/chrome.exe")
#: Windows install roots (environment variables) and the Chrome / Edge paths under them
WINDOWS_ROOT_ENVS: tuple[str, ...] = ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
WINDOWS_SUFFIXES: tuple[str, ...] = ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe")
#: macOS install path
MACOS_CANDIDATES: tuple[Path, ...] = (Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),)
#: the headless print command (the profile directory, the output file and the file URL are appended per run).
#: The last two switch name resolution and the proxy off: a ``file:`` URL needs neither, and without them a
#: headless Chromium 141 still contacted its own service hosts on every print in spite of the ``--disable-*`` flags
#: (measured with strace: 9 non-loopback ``connect()`` calls per print; with them 0, same PDF bytes).
PRINT_FLAGS: tuple[str, ...] = (
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--no-first-run",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--no-pdf-header-footer",
    "--no-proxy-server",
    "--host-resolver-rules=MAP * ~NOTFOUND",
)
DEFAULT_TIMEOUT_S = 120.0
NO_BROWSER_REASON = f"no browser found (chromium / chrome / edge on PATH, a Playwright install, or {BROWSER_ENV})"
#: browser stderr that says nothing about the print job (a headless Chromium logs these on every Linux run without a session bus)
_NOISE_RE = re.compile(r"dbus/|Fontconfig")


class FigureLike(Protocol):
    """What a figure must offer: the ``Figure`` of :mod:`ai_eda.report.figures` or any object shaped like it."""

    id: str
    title: str
    caption: str
    svg: str


# --- Markdown → HTML -------------------------------------------------------------------

_CSS = """
@page { size: A4; margin: 16mm; }
html { background: #fff; }
body { margin: 0 auto; max-width: 178mm; padding: 10mm 16px; background: #fff; color: #0b0b0b;
  font-family: "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "WenQuanYi Zen Hei", sans-serif;
  font-size: 10.5pt; line-height: 1.55; word-break: keep-all; overflow-wrap: anywhere; }
h1 { font-size: 18pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 1px solid #d8d6cf; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h4, h5, h6 { font-size: 10.5pt; margin: 10pt 0 3pt; }
h1, h2, h3, h4, h5, h6 { page-break-after: avoid; break-after: avoid; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin: 0 0 2pt; }
li > p { margin: 0 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 9pt; page-break-inside: avoid; break-inside: avoid; }
th, td { border: 1px solid #d8d6cf; padding: 2pt 5pt; text-align: left; vertical-align: top; min-width: 5em; }
/* min-width keeps a short column (a key, a value) from being crushed to one character per line by a long-text column;
   overflow-wrap: anywhere (inherited) keeps a 64-hex hash or a long identifier from pushing the table past the page edge */
th { background: #f3f2ee; font-weight: 600; }
thead { display: table-header-group; }
.align-left { text-align: left; }
.align-center { text-align: center; }
.align-right { text-align: right; }
code, pre { font-family: "Cascadia Mono", Consolas, "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono CJK KR", monospace; font-size: 9pt; }
code { background: #f3f2ee; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f3f2ee; border: 1px solid #e6e4dc; padding: 5pt 7pt; margin: 4pt 0 8pt; white-space: pre-wrap; overflow-wrap: anywhere; }
pre code { background: none; padding: 0; }
hr { border: 0; border-top: 1px solid #d8d6cf; margin: 10pt 0; }
figure { margin: 8pt 0 10pt; max-width: 100%; page-break-inside: avoid; break-inside: avoid; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption { font-size: 9pt; color: #52514e; margin-top: 3pt; }
.missing-figure { color: #8a8983; font-style: italic; border: 1px dashed #d8d6cf; padding: 4pt 6pt; }
@media print { body { max-width: none; padding: 0; } }
"""

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_HR_RE = re.compile(r"^ {0,3}-{3,}[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*[^`]*$")
_INDENT_CODE_RE = re.compile(r"^ {4}")
_LIST_RE = re.compile(r"^( {0,3})([-*+]|\d{1,9}[.)])(?:( +)(.*))?$")
_PLACEHOLDER_RE = re.compile(r"^!\[fig\]\(fig:([^\s()]+)\)$")
#: a whole line that is one italic run: the caption the Markdown reader sees under a placeholder
_ITALIC_LINE_RE = re.compile(r"^\*[^*\s](?:[^*]*[^*\s])?\*$")
_SEP_CELL_RE = re.compile(r"^(:?)-+(:?)$")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_CODE_SPAN_RE = re.compile(r"(`[^`\n]+`)")
_STRONG_RE = re.compile(r"\*\*(?=[^\s*])(.+?)(?<=[^\s*])\*\*")
_EM_RE = re.compile(r"(?<!\*)\*(?=[^\s*])([^*]+?)(?<=[^\s*])\*(?!\*)")
_ALIGNMENTS = {("", ""): None, (":", ""): "left", ("", ":"): "right", (":", ":"): "center"}


def _inline(text: str) -> str:
    """Inline Markdown of one text run: code spans, then ``**bold**`` and ``*italic*`` on the escaped rest."""
    out: list[str] = []
    for k, part in enumerate(_CODE_SPAN_RE.split(text)):
        if k % 2:
            out.append(f"<code>{esc(part[1:-1])}</code>")
        else:
            s = _STRONG_RE.sub(r"<strong>\1</strong>", esc(part))
            out.append(_EM_RE.sub(r"<em>\1</em>", s))
    return "".join(out)


def _pre(lines: list[str]) -> str:
    return f"<pre><code>{esc(chr(10).join(lines))}</code></pre>"


def _figure(fig: FigureLike) -> str:
    """The figure's own ``svg`` raw (our code generated it) under its caption rendered like any other text run (code spans become ``<code>``; ``*`` is neutralised as in the Markdown caption line, so no italic run is ever opened)."""
    title = _inline(fig.title.replace("*", "∗"))
    caption = title + (f" — {_inline(fig.caption.replace('*', '∗'))}" if fig.caption else "")
    return f"<figure>{fig.svg}<figcaption>{caption}</figcaption></figure>"


def _missing_figure(figure_id: str) -> str:
    return f'<p class="missing-figure">{esc(f"{MISSING_FIGURE}: {figure_id}")}</p>'


def _split_row(line: str) -> list[str]:
    """The cells of a pipe-table row: split on unescaped ``|``, outer pipes dropped, ``\\|`` restored."""
    s = line.strip()
    cells = _UNESCAPED_PIPE_RE.split(s)
    if s.startswith("|"):
        cells = cells[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        cells = cells[:-1]
    return [c.replace("\\|", "|").strip() for c in cells]


def _separator(line: str) -> list[str | None] | None:
    """The column alignments of a ``|---|:---:|`` row, ``None`` when the line is not one."""
    if not line.lstrip().startswith("|"):
        return None
    cells = _split_row(line)
    if not cells:
        return None
    aligns: list[str | None] = []
    for c in cells:
        m = _SEP_CELL_RE.match(c)
        if not m:
            return None
        aligns.append(_ALIGNMENTS[(m.group(1), m.group(2))])
    return aligns


def _table_start(lines: list[str], i: int) -> list[str | None] | None:
    if not lines[i].lstrip().startswith("|") or i + 1 >= len(lines):
        return None
    return _separator(lines[i + 1])


def _table(header_line: str, aligns: list[str | None], row_lines: list[str]) -> str:
    def cell(tag: str, text: str, k: int) -> str:
        align = aligns[k] if k < len(aligns) else None
        cls = f' class="align-{esc(align)}"' if align else ""
        return f"<{tag}{cls}>{_inline(text)}</{tag}>"

    header = _split_row(header_line)
    head = "<tr>" + "".join(cell("th", h, k) for k, h in enumerate(header)) + "</tr>"
    body: list[str] = []
    for rl in row_lines:
        cells = _split_row(rl)
        cells += [""] * (len(header) - len(cells))  # a short row is padded, a long one keeps every cell
        body.append("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(cells)) + "</tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _starts_block(lines: list[str], j: int) -> bool:
    """Whether line ``j`` interrupts a paragraph (heading, rule, fence, list item, placeholder or table)."""
    ln = lines[j]
    return bool(
        _HEADING_RE.match(ln) or _HR_RE.match(ln) or _FENCE_RE.match(ln) or _LIST_RE.match(ln)
        or _PLACEHOLDER_RE.match(ln.strip()) or _table_start(lines, j) is not None
    )


def _same_list(m: re.Match[str] | None, indent: int, ordered: bool) -> bool:
    return m is not None and len(m.group(1)) == indent and m.group(2)[0].isdigit() == ordered


def _collect_list(lines: list[str], i: int) -> tuple[list[list[str]], bool, int]:
    """The items (as dedented line lists) of the list starting at ``i``, whether it is loose, and the index after it."""
    first = _LIST_RE.match(lines[i])
    assert first is not None
    indent, ordered = len(first.group(1)), first.group(2)[0].isdigit()
    items: list[list[str]] = []
    loose = False
    j, n = i, len(lines)
    while j < n:
        m = _LIST_RE.match(lines[j])
        if not _same_list(m, indent, ordered):
            break
        assert m is not None
        spaces = m.group(3) or " "
        content_indent = indent + len(m.group(2)) + (len(spaces) if len(spaces) <= 4 else 1)
        item = [m.group(4) or ""]
        j += 1
        blanks = 0
        while j < n:
            ln = lines[j]
            if not ln.strip():
                blanks += 1
                j += 1
                continue
            if len(ln) - len(ln.lstrip(" ")) >= content_indent:
                if blanks:
                    item.extend([""] * blanks)
                    loose = True
                    blanks = 0
                item.append(ln[content_indent:])
                j += 1
                continue
            if not blanks and item[-1].strip() and not _starts_block(lines, j):
                item.append(ln.strip())  # lazy continuation of the item's paragraph
                j += 1
                continue
            break
        items.append(item)
        if blanks and j < n and _same_list(_LIST_RE.match(lines[j]), indent, ordered):
            loose = True
    return items, loose, j


def _render_blocks(lines: list[str], figures: Mapping[str, FigureLike], *, tight: bool = False) -> list[str]:
    """The block-level HTML fragments of ``lines``; ``tight`` renders paragraphs bare (inside a tight list item)."""
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _INDENT_CODE_RE.match(line):
            body: list[str] = []
            j = i
            while j < n:
                if _INDENT_CODE_RE.match(lines[j]):
                    body.append(lines[j][4:])
                    j += 1
                    continue
                if lines[j].strip():
                    break
                k = j
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and _INDENT_CODE_RE.match(lines[k]):
                    body.extend([""] * (k - j))
                    j = k
                    continue
                break
            out.append(_pre(body))
            i = j
            continue
        m = _FENCE_RE.match(line)
        if m:
            fence = m.group(1)
            close = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
            j = i + 1
            body = []
            while j < n and not close.match(lines[j]):
                body.append(lines[j])
                j += 1
            if j >= n:  # never closed: runs to the end, without the trailing blank lines
                while body and not body[-1].strip():
                    body.pop()
            out.append(_pre(body))
            i = j + 1
            continue
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if _HR_RE.match(line):
            out.append("<hr>")
            i += 1
            continue
        m = _PLACEHOLDER_RE.match(line.strip())
        if m:
            fig = figures.get(m.group(1))
            i += 1
            if fig is None:
                out.append(_missing_figure(m.group(1)))
                continue
            out.append(_figure(fig))
            k = i
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _ITALIC_LINE_RE.match(lines[k].strip()):
                i = k + 1  # the reader's caption line: the figure carries it as <figcaption>
            continue
        aligns = _table_start(lines, i)
        if aligns is not None:
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(_table(line, aligns, lines[i + 2:j]))
            i = j
            continue
        m = _LIST_RE.match(line)
        if m:
            items, loose, j = _collect_list(lines, i)
            ordered = m.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            start = f' start="{esc(int(m.group(2)[:-1]))}"' if ordered and int(m.group(2)[:-1]) != 1 else ""
            body_html = "".join(f"<li>{''.join(_render_blocks(it, figures, tight=not loose))}</li>" for it in items)
            out.append(f"<{tag}{start}>{body_html}</{tag}>")
            i = j
            continue
        buf = [line.strip()]
        j = i + 1
        while j < n and lines[j].strip() and not _starts_block(lines, j):
            buf.append(lines[j].strip())
            j += 1
        text = _inline("\n".join(buf))
        out.append(text if tight else f"<p>{text}</p>")
        i = j
    return out


def markdown_to_html(md: str, *, title: str, figures: Mapping[str, FigureLike] | None = None) -> str:
    """One self-contained page of the Markdown ``md`` with the figures whose ids its placeholders name.

    Deterministic for equal inputs; ``title`` and every text run are
    escaped; no ``src=``, ``href=``, ``<link>`` or ``<script>`` is ever
    emitted, and the page's Content-Security-Policy forbids any resource
    load even if a figure's SVG asked for one.
    """
    lines = [ln.expandtabs(4) for ln in md.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    body = "\n".join(_render_blocks(lines, figures or {}))
    return (
        '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )


# --- browser discovery -------------------------------------------------------------------


def _playwright_key(path: Path) -> tuple[int, str]:
    """Sort key for ``<root>/chromium-<build>/...``: the numeric build, then the path (``chromium-1194`` > ``chromium-999``)."""
    build = path.parent.parent.name.rsplit("-", 1)[-1]
    return (int(build) if build.isdigit() else -1, str(path))


def find_browser() -> Path | None:
    """The Chromium / Chrome / Edge binary to print with, or ``None``.

    Order: ``$AI_EDA_BROWSER`` (when it names an existing file), the
    :data:`PATH_NAMES` on PATH, the Playwright roots
    (``$PLAYWRIGHT_BROWSERS_PATH``, then :data:`PLAYWRIGHT_DEFAULT_ROOTS`;
    highest build first), the Windows install paths under
    :data:`WINDOWS_ROOT_ENVS`, then :data:`MACOS_CANDIDATES`.
    """
    explicit = os.environ.get(BROWSER_ENV)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for name in PATH_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    roots: list[Path] = []
    playwright = os.environ.get(PLAYWRIGHT_ENV)
    if playwright:
        roots.append(Path(playwright))
    roots.extend(PLAYWRIGHT_DEFAULT_ROOTS)
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in PLAYWRIGHT_GLOBS:
            matches = sorted((p for p in root.glob(pattern) if p.is_file()), key=_playwright_key, reverse=True)
            if matches:
                return matches[0]
    for env_name in WINDOWS_ROOT_ENVS:
        base = os.environ.get(env_name)
        if not base:
            continue
        for suffix in WINDOWS_SUFFIXES:
            candidate = Path(base) / suffix
            if candidate.is_file():
                return candidate
    for candidate in MACOS_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def browser_version(path: Path | str, timeout_s: float = 30.0) -> str | None:
    """``<name> <version>`` as ``--version`` prints it, or ``None`` (best effort: Chrome on Windows prints nothing)."""
    try:
        proc = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            return line.strip()
    return None


# --- printing ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfResult:
    """What :func:`html_to_pdf` did: ``ok`` only when ``path`` exists and is non-empty; otherwise ``reason``."""

    ok: bool
    path: Path | None
    reason: str | None
    browser: Path | None


def _reason(prefix: str, stderr: str | None, limit: int = 3) -> str:
    """``prefix: <first stderr lines>`` with the browser's session-bus noise left out (kept when nothing else is there)."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    kept = [ln for ln in lines if not _NOISE_RE.search(ln)] or lines
    if not kept:
        return f"{prefix} (no stderr)"
    return prefix + ": " + " | ".join(ln[:240] for ln in kept[:limit])


def html_to_pdf(html_path: Path | str, pdf_path: Path | str, browser: Path | str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> PdfResult:
    """Print ``html_path`` to ``pdf_path`` with a headless browser; never raises into the pipeline.

    ``browser`` defaults to :func:`find_browser`; none found, a missing HTML
    file, a browser that cannot start, a timeout, a non-zero exit or a run
    that leaves no (or an empty) PDF each give ``ok=False`` with a reason made
    of the first stderr lines. A stale file at ``pdf_path`` is removed first,
    so ``ok`` always means this run wrote the PDF. The browser gets a fresh
    temporary profile (``--user-data-dir``) so the print job never joins a
    desktop Chrome / Edge that is already running (which would swallow the
    command and write nothing) and never reads the user's profile.

    This launches a local program on a local file and opens no socket by
    our doing; the browser runs with name resolution disabled and no proxy
    (:data:`PRINT_FLAGS`: ``--host-resolver-rules=MAP * ~NOTFOUND``,
    ``--no-proxy-server``), so its own services cannot reach any host
    either - measured on a headless Chromium 141, which without those two
    flags opened outbound sockets to its service hosts on every print
    despite ``--disable-background-networking``; with them strace shows no
    ``connect()`` at all and the same PDF. It is not an
    :class:`~ai_eda.security.ExternalAction`. Nothing of the report is sent
    anywhere in either case. The bytes carry the browser's creation date
    and producer: a derived document like a rawfile, not deterministic
    output, never an artifact.
    """
    html = Path(html_path)
    pdf = Path(pdf_path)
    exe = Path(browser) if browser is not None else find_browser()
    if exe is None:
        return PdfResult(False, None, NO_BROWSER_REASON, None)
    if not html.is_file():
        return PdfResult(False, None, f"HTML file not found: {html.name}", exe)
    try:
        html_abs = html.resolve()
        pdf_abs = pdf.resolve()
        pdf_abs.parent.mkdir(parents=True, exist_ok=True)
        if pdf_abs.exists():
            pdf_abs.unlink()
    except OSError as e:
        return PdfResult(False, None, f"cannot prepare {pdf.name}: {e.strerror or e}", exe)
    with tempfile.TemporaryDirectory(prefix="ai-eda-pdf-", ignore_cleanup_errors=True) as profile:
        args = [str(exe), *PRINT_FLAGS, f"--user-data-dir={profile}", f"--print-to-pdf={pdf_abs}", html_abs.as_uri()]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return PdfResult(False, None, f"timed out after {timeout_s:g} s", exe)
        except OSError as e:
            return PdfResult(False, None, f"cannot run {exe.name}: {e.strerror or e}", exe)
    if proc.returncode != 0:
        return PdfResult(False, None, _reason(f"exit code {proc.returncode}", proc.stderr), exe)
    if not pdf_abs.is_file() or pdf_abs.stat().st_size == 0:
        return PdfResult(False, None, _reason("no PDF written", proc.stderr), exe)
    return PdfResult(True, pdf, None, exe)


__all__ = [
    "BROWSER_ENV",
    "DEFAULT_TIMEOUT_S",
    "MACOS_CANDIDATES",
    "MISSING_FIGURE",
    "NO_BROWSER_REASON",
    "PATH_NAMES",
    "PLAYWRIGHT_DEFAULT_ROOTS",
    "PLAYWRIGHT_ENV",
    "PLAYWRIGHT_GLOBS",
    "PRINT_FLAGS",
    "WINDOWS_ROOT_ENVS",
    "WINDOWS_SUFFIXES",
    "FigureLike",
    "PdfResult",
    "browser_version",
    "find_browser",
    "html_to_pdf",
    "markdown_to_html",
]
