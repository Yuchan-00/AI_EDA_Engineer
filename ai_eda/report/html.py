"""Render :class:`~ai_eda.report.data.ReportData` as one self-contained HTML page.

Invariant: there is exactly one path from data to markup, :func:`esc`
(``html.escape(str(value), quote=True)``), and every text and attribute
value goes through it inside :func:`tag`, :func:`rows` and :func:`kv`.
:func:`block` only nests markup those builders returned; no f-string in
this module interpolates a data value into markup directly. The page loads
no external resource (no ``<link>``, no ``<script>``, no ``@import``, no
image) and carries no render time: the header names the sha256 of the
ir.json and pipeline.json it was built from, so unchanged inputs give a
byte-identical page.
"""

from __future__ import annotations

import html as _html

from ai_eda.report.data import DomainSection, ReportData, ValidationRow

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1200px; padding: 16px; }
h1 { font-size: 1.6em; margin: 0 0 4px; }
h2 { font-size: 1.2em; margin: 28px 0 8px; border-bottom: 1px solid #8884; padding-bottom: 4px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; }
th, td { border: 1px solid #8884; padding: 3px 6px; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #8882; }
td.num { text-align: right; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
pre { white-space: pre-wrap; word-break: break-all; background: #8881; padding: 6px; margin: 4px 0; }
.st { font-weight: 600; white-space: nowrap; }
.st-PASS { color: #1a7f37; }
.st-FAIL { color: #cf222e; }
.st-NOT_VERIFIED, .st-UNRESOLVED { color: #9a6700; }
.st-USER_INPUT_REQUIRED { color: #0969da; }
.st-NOT_APPLICABLE { color: #6e7781; }
.note { color: #6e7781; font-style: italic; }
.warn { color: #9a6700; }
.hash { font-family: ui-monospace, monospace; font-size: 11px; word-break: break-all; }
details { margin: 4px 0; }
summary { cursor: pointer; }
ul.plain { margin: 4px 0; padding-left: 20px; }
"""

_DOCTYPE = "<!DOCTYPE html>\n"
_BR = "<br>"
_META = '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'


def esc(value: object) -> str:
    """The one escaping function: ``None`` renders as nothing, everything else as escaped text (quotes included)."""
    if value is None:
        return ""
    return _html.escape(str(value), quote=True)


def _attrs(attrs: dict[str, object]) -> str:
    out = []
    for k, v in attrs.items():
        if v is None or v is False:
            continue
        name = k.rstrip("_").replace("_", "-")
        out.append(f' {esc(name)}="{esc(v)}"')
    return "".join(out)


def tag(name: str, text: object = "", **attrs: object) -> str:
    """``<name attrs>escaped text</name>``: ``text`` is data and is always escaped."""
    return f"<{name}{_attrs(attrs)}>{esc(text)}</{name}>"


def block(name: str, *children: str, **attrs: object) -> str:
    """``<name attrs>children</name>`` where every child is markup returned by :func:`tag` / :func:`block` / :func:`rows` / :func:`kv`."""
    return f"<{name}{_attrs(attrs)}>{''.join(children)}</{name}>"


def status_cell(status: object) -> str:
    """A status rendered verbatim (escaped) with a class derived from the same escaped value."""
    s = "" if status is None else str(status)
    return tag("td", s, class_=f"st st-{s}" if s else None)


def rows(header: list[str], body: list[list[object]]) -> str:
    """A table whose every cell is escaped data, except a cell given as :class:`Markup` (already built by these builders)."""
    head = block("tr", *(tag("th", h) for h in header))
    lines = []
    for r in body:
        lines.append(block("tr", *(c.markup if isinstance(c, Markup) else tag("td", c) for c in r)))
    return block("table", block("thead", head), block("tbody", *lines))


def kv(pairs: list[tuple[str, object]]) -> str:
    """A two-column table of escaped key / value pairs (a :class:`Markup` value is nested as built)."""
    return block("table", block("tbody", *(block("tr", tag("th", k), v.markup if isinstance(v, Markup) else tag("td", v)) for k, v in pairs)))


class Markup:
    """A cell that is already markup from the builders above (never raw data)."""

    __slots__ = ("markup",)

    def __init__(self, markup: str) -> None:
        self.markup = markup


def _pre(text: object) -> str:
    return tag("pre", text)


def _details(summary: object, *children: str) -> str:
    return block("details", tag("summary", summary), *children)


def _list(items: list[object]) -> str:
    return block("ul", *(tag("li", i) for i in items), class_="plain")


# --- sections -----------------------------------------------------------------


def _header(data: ReportData) -> str:
    m = data.meta
    pairs: list[tuple[str, object]] = [
        ("project", f"{m.project_id} - {m.project_name}"),
        ("description", m.description),
        ("design hash", Markup(tag("td", m.design_hash, class_="hash"))),
        ("ir.json", Markup(block("td", tag("span", m.ir_file_path), _BR, tag("span", m.ir_file_sha256, class_="hash")))),
        (
            "pipeline.json",
            Markup(
                block(
                    "td",
                    tag("span", m.pipeline_file_path or "(none)"),
                    _BR,
                    tag("span", m.pipeline_file_sha256 or "", class_="hash"),
                    tag("div", m.pipeline_note, class_="note"),
                )
            ),
        ),
        ("IR schema", m.schema_version),
        ("ai-eda", m.ai_eda_version),
        ("created", m.created_at),
        ("workdir", m.workdir),
    ]
    return block(
        "header",
        tag("h1", f"{m.project_name} - design report"),
        tag("p", "Read-only view of what ir.json and pipeline.json record. Every status below is copied from a stored result; "
                 "the page computes none and is not a verdict.", class_="note"),
        kv(pairs),
    )


def _release(data: ReportData) -> str:
    r = data.release
    if r.status is None:
        return block("section", tag("h2", "RELEASE"), tag("p", r.note, class_="warn"))
    return block(
        "section",
        tag("h2", "RELEASE"),
        block("p", tag("span", "recorded status: "), tag("span", r.status, class_=f"st st-{r.status}"), tag("span", f" at {r.at}")),
        tag("p", r.note, class_="note"),
        _list(list(r.reasons)),
    )


def _aggregate(data: ReportData) -> str:
    v = data.validation
    return block("p", tag("span", "aggregate: "), tag("span", v.aggregate, class_=f"st st-{v.aggregate}"), tag("span", f" - {v.aggregate_note}", class_="note"))


def _stages(data: ReportData) -> str:
    s = data.stages
    if s is None:
        return block("section", tag("h2", "Pipeline stages"), tag("p", data.stages_reason or "", class_="warn"))
    facts: list[tuple[str, object]] = [
        ("run IR hash", Markup(block("td", tag("span", s.run_ir_hash, class_="hash"), tag("span", f" - {s.run_hash_label}", class_="note")))),
        ("recorded by", f"ai-eda {s.ai_eda_version} on {s.ir_path}"),
        ("results before the run", s.results_before),
        ("blocked", "yes" if s.blocked else "no"),
        ("last stage started", s.current or ""),
    ]
    if s.aborted:
        facts.append(("aborted", f"last run aborted: {s.aborted} in stage {s.aborted_stage or '?'}"))
    body = [[r.stage, Markup(status_cell(r.status)), r.message, r.questions, r.at] for r in s.rows]
    return block("section", tag("h2", "Pipeline stages"), kv(facts), rows(["stage", "status", "message", "questions", "at"], body))


def _questions(data: ReportData) -> str:
    if not data.questions:
        return block("section", tag("h2", "Open questions"), tag("p", "none recorded", class_="note"))
    body = [
        [
            q.key,
            "required" if q.required else "optional",
            q.source_label,
            Markup(block("td", tag("span", q.question), tag("span", " (model output)" if q.source == "llm" else "", class_="warn"))),
            ", ".join(q.options),
            q.rationale,
            q.origin,
            Markup(block("td", tag("code", q.command))),
        ]
        for q in data.questions
    ]
    return block(
        "section",
        tag("h2", "Open questions"),
        rows(["key", "required", "source", "question", "options", "rationale", "origin", "answer with"], body),
    )


def _requirements(data: ReportData) -> str:
    r = data.requirements
    parts = [tag("h2", "Requirements"), tag("h3", "request"), _pre(r.raw_input or "(empty)")]
    if r.corrections:
        parts += [tag("h3", "corrections (user)"), _list(list(r.corrections))]
    parts.append(
        rows(
            ["id", "text", "kind", "status", "category", "value", "unit", "provenance", "needs verification"],
            [[x.id, x.text, x.kind, x.status, x.category, x.value, x.unit, x.provenance_kind, "yes" if x.needs_verification else "no"] for x in r.rows],
        )
    )
    if r.parameters:
        parts += [tag("h3", "parameters"), rows(["key", "value", "unit", "provenance", "tool"], [[p.key, p.value, p.unit, p.provenance_kind, p.tool] for p in r.parameters])]
    if r.conflicts:
        parts += [tag("h3", "conflicts"), _list(list(r.conflicts))]
    if r.extraction:
        parts += [
            tag("h3", "extraction cache (model output, not design data)"),
            rows(["request hash", "confirmed", "presented"], [[e.request_hash, "yes" if e.confirmed else "no", "yes" if e.presented else "no"] for e in r.extraction]),
        ]
    return block("section", *parts)


def _evidence_markup(row: ValidationRow) -> str:
    if not row.evidence:
        return ""
    return rows(["evidence", "path", "url", "hash", "state"], [[e.description, e.path, e.url, e.content_hash, e.state] for e in row.evidence])


def _result_rows(latest: list[ValidationRow]) -> str:
    body = []
    for r in latest:
        tool = f"{r.tool} {r.tool_version}".strip() if r.tool else ""
        status = block("td", tag("span", r.status, class_=f"st st-{r.status}"), tag("div", "opinion (no tool)" if r.opinion else "", class_="warn"),
                       tag("div", r.note, class_="warn"))
        artifact = f"{r.artifact_kind} {r.artifact_hash}".strip()
        extra = [tag("span", r.message)]
        ev = _evidence_markup(r)
        if ev:
            extra.append(ev)
        if r.details_json:
            extra.append(_details("details", _pre(r.details_json)))
        body.append([r.check_id, Markup(status), tool, r.freshness, Markup(tag("td", artifact, class_="hash")), Markup(block("td", *extra)), r.timestamp])
    return rows(["check", "status", "tool", "ir_hash", "artifact", "message / evidence / details", "at"], body)


def _validation(data: ReportData) -> str:
    v = data.validation
    history = rows(["check", "#", "status", "at", "message"], [[h.check_id, h.index, h.status, h.timestamp, h.message] for h in v.history])
    return block(
        "section",
        tag("h2", "Validation log (latest per check)"),
        _aggregate(data),
        _result_rows(v.latest),
        _details(f"full history ({len(v.history)} results, grouped by check)", history),
    )


def _artifacts(data: ReportData) -> str:
    if not data.artifacts:
        return block("section", tag("h2", "Artifacts"), tag("p", "none registered", class_="note"))
    body = [
        [
            a.kind, a.path, a.files, f"{a.generator} {a.generator_version}".strip(), Markup(tag("td", a.content_hash, class_="hash")),
            Markup(tag("td", a.generated_from_ir_hash, class_="hash")), a.freshness, a.disk, "; ".join(a.notes), a.created_at,
        ]
        for a in data.artifacts
    ]
    return block("section", tag("h2", "Artifacts"), rows(["kind", "path", "files", "generator", "content hash", "from IR", "IR", "disk", "notes", "created"], body))


def _review(data: ReportData) -> str:
    r = data.review
    body = [[x.area, Markup(status_cell(x.status)), x.message, x.freshness, x.evidence, x.timestamp] for x in r.rows]
    counts = ", ".join(f"{k}={v}" for k, v in r.counts.items())
    return block(
        "section",
        tag("h2", "Independent review (14 areas)"),
        tag("p", f"counts: {counts or 'none'}"),
        tag("p", r.note, class_="note"),
        rows(["area", "status", "message", "ir_hash", "evidence", "at"], body),
    )


def _repair(data: ReportData) -> str:
    r = data.repair
    if r is None:
        return block("section", tag("h2", "Repair loop"), tag("p", "no repair.loop result recorded", class_="note"))
    return block(
        "section",
        tag("h2", "Repair loop"),
        kv([("status", Markup(status_cell(r.status))), ("message", r.message), ("ir_hash", r.freshness), ("iterations", r.iterations),
            ("stopped", r.stopped_reason), ("at", r.timestamp)]),
        _details("actions", _pre(r.actions_json)),
        _details("unresolved", _pre(r.unresolved_json)),
        _details("final review", _pre(r.final_review_json)),
    )


def _domain(d: DomainSection) -> str:
    parts = [tag("h2", d.title), tag("p", d.note, class_="note")]
    if d.results:
        parts.append(_result_rows(d.results))
    else:
        parts.append(tag("p", f"no result with a check id starting with {', '.join(d.prefixes)}", class_="note"))
    parts.append(_details(f"{d.title.lower()} IR section (JSON, as stored)", _pre(d.ir_json)))
    return block("section", *parts)


def render_html(data: ReportData) -> str:
    """The whole page; deterministic for equal ``data``."""
    head = block("head", *(_META, tag("title", f"{data.meta.project_name} - design report"), block("style", _CSS)))
    body = block(
        "body",
        _header(data),
        _release(data),
        _stages(data),
        _questions(data),
        _requirements(data),
        _validation(data),
        _artifacts(data),
        _review(data),
        _repair(data),
        _domain(data.regulatory),
        _domain(data.components),
        _domain(data.simulation),
        tag("footer", "ai-eda report: a view of the files named in the header; it registers no artifact and decides nothing.", class_="note"),
    )
    return _DOCTYPE + block("html", head, body, lang="en") + "\n"
