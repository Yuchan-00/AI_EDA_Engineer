"""A loopback-only HTTP server that re-renders the report on every request.

Invariant: the listener binds :data:`LOOPBACK` (127.0.0.1) and nothing
else - there is no host option, by design. It serves ``GET /`` only (every
other path is 404), answers only requests whose ``Host`` header names this
machine (127.0.0.1, localhost or [::1], optional port; anything else gets
421 with an empty body, so a page on a DNS-rebound name cannot read the
report from a browser tab), never sends a file, a directory listing or a
traceback (a render failure is a 503 with a one-line escaped message), and
opens no outbound connection: no :class:`~ai_eda.security.ExternalAction`
is involved because nothing leaves the machine. ``Cache-Control: no-store``
and a fresh render per request make a later ``ai-eda run`` visible on
refresh. The IR is never saved.
"""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ai_eda.report.html import esc

LOOPBACK = "127.0.0.1"
#: Host header values (before an optional ``:port``) that name this machine
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def host_allowed(host_header: str | None) -> bool:
    """Whether a ``Host`` header names the loopback listener (``[::1]:8765`` keeps its brackets)."""
    if not host_header:
        return False
    value = host_header.strip()
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return False
        name, rest = value[: end + 1], value[end + 1:]
    else:
        name, _sep, rest = value.partition(":")
        rest = f":{rest}" if _sep else ""
    if rest and not (rest.startswith(":") and rest[1:].isdigit()):
        return False
    return name.lower() in ALLOWED_HOSTS


def render_page(ir_path: Path) -> str:
    """The report of ``ir_path`` right now (reads ir.json and pipeline.json afresh, each exactly once; saves nothing)."""
    from ai_eda.cli import project_workdir
    from ai_eda.report.data import build_report_data, load_ir_file
    from ai_eda.report.html import render_html

    ir, ir_sha = load_ir_file(ir_path)
    return render_html(build_report_data(ir, ir_path, project_workdir(ir, ir_path), ir_sha=ir_sha))


class _Handler(BaseHTTPRequestHandler):
    server_version = "ai-eda-report"
    sys_version = ""  # no Python version in the Server header
    protocol_version = "HTTP/1.1"

    def version_string(self) -> str:
        return self.server_version  # the Server header: no Python version, no trailing space

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - BaseHTTPRequestHandler's signature
        return  # quiet: the page is the output, not a request log

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if not host_allowed(self.headers.get("Host")):
            self._send(421, b"", "text/plain; charset=utf-8")
            return
        if urlsplit(self.path).path != "/":
            self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return
        try:
            page = render_page(self.server.ir_path)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - one escaped line, never a traceback
            line = " ".join(str(e).split())
            self._send(503, f"report unavailable: {esc(line)}\n".encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")


class ReportServer:
    """``ThreadingHTTPServer`` on 127.0.0.1 serving :func:`render_page` of one ir.json (``port`` 0 = ephemeral)."""

    def __init__(self, ir_path: Path, port: int = 0) -> None:
        self.ir_path = Path(ir_path)
        self._server = ThreadingHTTPServer((LOOPBACK, port), _Handler)
        self._server.daemon_threads = True
        self._server.ir_path = self.ir_path  # type: ignore[attr-defined]

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


def serve(ir_path: Path, port: int) -> int:
    """Run the server until Ctrl-C (exit 0); a bind failure prints the OSError and returns 2."""
    try:
        server = ReportServer(ir_path, port)
    except OSError as e:
        print(f"could not listen on {LOOPBACK}:{port}: {e}", file=sys.stderr)
        return 2
    print(f"serving the report of {ir_path} at {server.url} (127.0.0.1 only, GET / only, read-only; Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()  # not shutdown(): that waits for a serve_forever loop in another thread
    return 0
