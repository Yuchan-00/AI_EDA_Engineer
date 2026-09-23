"""A local fake of the web the document archive fetches from, for tests without the network.

:class:`FakeSources` is a threaded HTTP server on ``127.0.0.1`` that serves
scripted documents keyed by the *https* URL the archive asks for: PDFs (from
:mod:`tests.pdf_fixture`), HTML, plain text, XML, redirects (same host, cross
host, relative ``Location``), 404 pages with big bodies, interstitial /
bot-protection pages of the kinds met on real manufacturer and regulator
sites (Cloudflare "Just a moment", Akamai "Access Denied", a JavaScript
viewer shell, an iframe shell, an "unblock" page), and slow responses.

The archive only ever speaks https, so the fake does not weaken that: its
:meth:`FakeSources.transport` is an httpx transport that reroutes every
request to the loopback server while leaving the request's ``Host`` header
(``www.ti.com``, ``eur-lex.europa.eu`` …) untouched; the server routes by
that header and records every request (:attr:`FakeSources.requests`) so
tests can assert that a refused fetch opened no connection at all.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx

from tests.pdf_fixture import build_pdf

#: the interstitial pages the archive must classify as ``blocked``
INTERSTITIALS: dict[str, tuple[int, str, str]] = {
    # kind: (status, content type, body)
    "cloudflare": (403, "text/html; charset=utf-8",
                   "<!DOCTYPE html><html><head><title>Just a moment...</title><script src='/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1'></script></head>"
                   "<body><div class='main-wrapper'><h1>www.jst.com</h1><p>Verifying you are human. This may take a few seconds.</p>"
                   "<p>Enable JavaScript and cookies to continue</p><noscript>cf-chl</noscript></div></body></html>"),
    "access_denied": (403, "text/html",
                      "<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY><H1>Access Denied</H1>You don't have permission to access this resource."
                      "<P>Reference #18.1234abcd</P></BODY></HTML>"),
    "unblock": (200, "text/html; charset=utf-8",
                "<!DOCTYPE html><html><head><title>Unblock request</title></head><body><h1>Please complete the CAPTCHA to unblock your request</h1>"
                "<p>Automated requests are being rate limited. Complete the challenge below to continue to the document you requested. "
                "This page is shown instead of the document when automated traffic is detected.</p><form><input name='captcha'></form></body></html>"),
    "js_shell": (200, "text/html; charset=utf-8",
                 "<!DOCTYPE html><html><head><title>Datasheet viewer</title><script src='/static/viewer.js'></script><style>body{margin:0}</style></head>"
                 "<body><noscript>please enable javascript</noscript><div id='app'></div><script>window.__DS__={id:'C25792'};</script></body></html>"),
    "iframe_shell": (200, "text/html; charset=utf-8",
                     "<html><head><title>법령</title></head><body><iframe id='lawService' src='/LSW//lsInfoP.do?lsiSeq=276245'></iframe></body></html>"),
}

#: a 404 whose body is a full home page (the yageo.com case: the status, not the size, decides)
NOT_FOUND_HTML = (
    "<!DOCTYPE html><html><head><title>Page not found</title></head><body><nav>Products Support About Contact Careers News</nav>"
    "<main><h1>Sorry, we could not find that page</h1>" + "<p>Browse our product families: resistors, capacitors, inductors, sensors and more. "
    "Our catalogue covers thousands of part numbers across all packages.</p>" * 40 + "</main><footer>Copyright</footer></body></html>"
)


@dataclass
class RecordedRequest:
    method: str
    host: str
    path: str
    query: str
    headers: dict[str, str]  # lower-cased names

    @property
    def url(self) -> str:
        return f"https://{self.host}{self.path}" + (f"?{self.query}" if self.query else "")

    @property
    def user_agent(self) -> str | None:
        return self.headers.get("user-agent")


@dataclass
class ScriptedResponse:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/octet-stream"
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0  # seconds before the status line
    hits: int = 0


def _key(url: str) -> str:
    """Routing key: host without ``www.``, path and query percent-decoded (a Korean path arrives percent-encoded)."""
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{unquote(p.path) or '/'}" + (f"?{unquote(p.query)}" if p.query else "")


class _LoopbackTransport(httpx.BaseTransport):
    """Reroutes every request to the fake server; the ``Host`` header (and so the https origin) is kept.

    The URL the client *asked for* (scheme included) is recorded in
    ``fake.client_urls`` before rewriting, so a test can prove the archive
    never requested plain ``http``.
    """

    def __init__(self, fake: FakeSources) -> None:
        self._fake = fake
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._fake.client_urls.append(str(request.url))
        request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=self._fake.port)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class FakeSources:
    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.requests: list[RecordedRequest] = []
        #: every URL an httpx client built from :meth:`client` asked for, as the client saw it (scheme included)
        self.client_urls: list[str] = []
        self._docs: dict[str, ScriptedResponse] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> FakeSources:
        if self._server is not None:
            return self
        fake = self

        class Handler(_Handler):
            pass

        Handler.fake = fake  # type: ignore[attr-defined]
        self._server = ThreadingHTTPServer((self.host, 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> FakeSources:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    def transport(self) -> httpx.BaseTransport:
        return _LoopbackTransport(self)

    def client(self, **kw: Any) -> httpx.Client:
        """An httpx client whose every request lands on this fake (redirects are never followed by the client itself)."""
        kw.setdefault("follow_redirects", False)
        return httpx.Client(transport=self.transport(), **kw)

    # ------------------------------------------------------------ scripting

    def serve(self, url: str, body: bytes | str, content_type: str = "application/octet-stream", status: int = 200,
              headers: dict[str, str] | None = None, delay: float = 0.0) -> ScriptedResponse:
        data = body.encode("utf-8") if isinstance(body, str) else body
        s = ScriptedResponse(status=status, body=data, content_type=content_type, headers=dict(headers or {}), delay=delay)
        with self._lock:
            self._docs[_key(url)] = s
        return s

    def add_pdf(self, url: str, pages: list[list[str]], content_type: str = "application/pdf", **kw: Any) -> bytes:
        data = build_pdf(pages)
        self.serve(url, data, content_type, **kw)
        return data

    def add_html(self, url: str, html: str, charset: str = "utf-8", **kw: Any) -> bytes:
        data = html.encode(charset)
        self.serve(url, data, f"text/html; charset={charset}", **kw)
        return data

    def add_text(self, url: str, text: str, charset: str = "utf-8", content_type: str = "text/plain", **kw: Any) -> bytes:
        data = text.encode(charset)
        self.serve(url, data, f"{content_type}; charset={charset}", **kw)
        return data

    def add_xml(self, url: str, xml: str, charset: str = "utf-8", **kw: Any) -> bytes:
        data = xml.encode(charset)
        self.serve(url, data, "application/xml", **kw)
        return data

    def add_redirect(self, url: str, location: str, status: int = 302) -> ScriptedResponse:
        return self.serve(url, b"", "text/html", status=status, headers={"Location": location})

    def add_missing(self, url: str, body: str = NOT_FOUND_HTML, status: int = 404) -> ScriptedResponse:
        return self.serve(url, body, "text/html; charset=utf-8", status=status)

    def add_interstitial(self, url: str, kind: str = "cloudflare") -> ScriptedResponse:
        status, ct, body = INTERSTITIALS[kind]
        return self.serve(url, body, ct, status=status)

    def add_slow(self, url: str, delay: float, body: bytes | str = b"slow", content_type: str = "text/plain") -> ScriptedResponse:
        return self.serve(url, body, content_type, delay=delay)

    # ------------------------------------------------------------ recording

    def requests_for(self, host: str) -> list[RecordedRequest]:
        h = host.lower()
        return [r for r in self.requests if r.host == h or r.host == "www." + h or "www." + r.host == h]

    def hits(self, url: str) -> int:
        with self._lock:
            s = self._docs.get(_key(url))
        return s.hits if s is not None else 0

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.client_urls.clear()
            self._docs.clear()

    # ------------------------------------------------------------ internals

    def _lookup(self, host: str, path: str, query: str) -> ScriptedResponse | None:
        with self._lock:
            s = self._docs.get(_key(f"https://{host}{path}" + (f"?{query}" if query else "")))
            if s is not None:
                s.hits += 1
            return s

    def _record(self, req: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(req)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fake: FakeSources

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature fixed by the base class
        pass

    def _serve(self) -> None:
        parts = urlsplit(self.path)
        host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
        req = RecordedRequest(method=self.command, host=host, path=parts.path, query=parts.query,
                              headers={k.lower(): v for k, v in self.headers.items()})
        self.fake._record(req)
        s = self.fake._lookup(host, parts.path, parts.query)
        if s is None:
            payload = f"<html><head><title>Not scripted</title></head><body><h1>404</h1><p>{req.url} is not scripted in this fake</p></body></html>".encode()
            self._send(404, payload, "text/html; charset=utf-8", {})
            return
        if s.delay:
            time.sleep(s.delay)
        self._send(s.status, s.body if self.command != "HEAD" else b"", s.content_type, s.headers, length=len(s.body))

    def _send(self, status: int, payload: bytes, content_type: str, headers: dict[str, str], length: int | None = None) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload) if length is None else length))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if payload:
                self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
        self._serve()

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802 - the archive never POSTs; a POST is recorded and refused
        parts = urlsplit(self.path)
        host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
        self.fake._record(RecordedRequest(method="POST", host=host, path=parts.path, query=parts.query,
                                          headers={k.lower(): v for k, v in self.headers.items()}))
        self._send(405, b"method not allowed", "text/plain", {})


__all__ = ["INTERSTITIALS", "NOT_FOUND_HTML", "FakeSources", "RecordedRequest", "ScriptedResponse"]
