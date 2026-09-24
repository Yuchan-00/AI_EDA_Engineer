"""``ai-eda report`` / ``ai-eda serve`` and the ``pipeline.json`` run log.

The report is a view: every status it shows is copied from ``ir.validation``
or from the recorded ``pipeline.json``, it computes none, and rendering or
serving it never saves the IR (its bytes are compared before and after).
Every string from the files is escaped through one function, the page loads
nothing external, and unchanged inputs render byte-identically.
"""

from __future__ import annotations

import http.client
import io
import json
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.cli import main as cli_main
from ai_eda.ir.regulatory import GroundedQuote
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitIR,
    Evidence,
    MissingInformation,
    ProjectMeta,
    RegulatoryProvenance,
    RegulatoryRequirement,
    Requirement,
    RequirementKind,
    ValidationResult,
    ValidationStatus,
    user_requirement,
)
from ai_eda.regulatory.research import OFFLINE_REASON
from ai_eda.report import (
    PIPELINE_FILE,
    PipelineRecord,
    PipelineRecordError,
    ReportServer,
    build_report_data,
    esc,
    load_pipeline_record,
    render_html,
    render_report_file,
    save_pipeline_record,
)
from ai_eda.report.data import (
    AGGREGATE_NOTE,
    CHANGED_ON_DISK,
    EVIDENCE_MISSING,
    EVIDENCE_OK,
    FRESH,
    MISSING_ON_DISK,
    NO_RECORDED_RUN,
    ON_DISK,
    OPINION,
    PIPELINE_DESCRIBES_IR,
    PIPELINE_STALE_IR,
    STALE,
    UNSTAMPED_CARRIED,
    UNSTAMPED_PRODUCED,
    UNSTAMPED_UNKNOWN,
)
from ai_eda.report.pipeline_log import sha256_of_file
from ai_eda.report.server import host_allowed, serve
from ai_eda.review.areas import ReviewArea
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.workflow import STAGE_ORDER, Orchestrator, PipelineState, Stage, StageOutcome

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[0-9.]*(?:Z|[+-]\d{2}:\d{2})?")


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _offline_ctx(workdir: Path, **answers: str) -> AgentContext:
    # an empty library root: the same offline run on every machine (test_orchestrator.py's pattern)
    return AgentContext(workdir=workdir, answers={"application": "test", "jurisdiction": "EU", **answers},
                        tools={"kicad_library": KicadLibrary(roots=[workdir / "nolib"])})


def _run_and_record(ir: CircuitIR, workdir: Path, *, stop_after: Stage | None = None) -> tuple[Path, PipelineState]:
    """One offline pipeline run, then ir.json + pipeline.json written the way ``cmd_run`` writes them."""
    results_before = len(ir.validation.results)
    state = PipelineState()
    Orchestrator(_offline_ctx(workdir)).run(ir, stop_after=stop_after, state=state)
    ir_path = ir.save(workdir / "ir.json")
    save_pipeline_record(state, ir, ir_path, workdir, results_before=results_before, ir_file_sha256=sha256_of_file(ir_path))
    return ir_path, state


@pytest.fixture
def project(divider_ir: CircuitIR, tmp_path: Path) -> tuple[Path, PipelineState]:
    return _run_and_record(divider_ir, tmp_path)


# --------------------------------------------------------------------------- pipeline.json


def test_cmd_run_records_pipeline_json(tmp_path: Path):
    code, out, _ = _cli("new", "demo", "--dir", str(tmp_path / "demo"), "--request", "a 5 V regulator")
    assert code == 0
    ir_path = tmp_path / "demo" / "ir.json"
    code, out, err = _cli("run", str(ir_path), "--answer", "application=bench supply", "--answer", "jurisdiction=EU")
    record_path = tmp_path / "demo" / PIPELINE_FILE
    assert record_path.is_file(), err
    assert f"stage outcomes recorded in {record_path}" in out
    # the line about it starts with a space: test_cli_llm's stdout parser (first word = stage) never sees it
    line = next(line for line in out.splitlines() if PIPELINE_FILE in line)
    assert line.startswith(" ")
    record = load_pipeline_record(tmp_path / "demo")
    assert isinstance(record, PipelineRecord)
    ir = CircuitIR.load(ir_path)
    assert record.ir_hash == ir.content_hash()
    assert record.ir_file_sha256 == sha256_of_file(ir_path)
    assert record.results_before == 0 and record.aborted is None and record.aborted_stage is None
    assert record.ir_path == str(ir_path)
    assert not record.state.blocked
    assert record.state.outcomes[-1].stage == Stage.RELEASE
    assert record.state.outcomes[-1].status is not ValidationStatus.PASS
    assert record.state.outcomes[-1].message.startswith("not releasable: ")
    # a second run starts where the first left the log
    code, out, _ = _cli("run", str(ir_path), "--answer", "application=bench supply", "--answer", "jurisdiction=EU")
    assert load_pipeline_record(tmp_path / "demo").results_before == len(ir.validation.results)


def test_pipeline_record_written_when_the_run_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _cli("new", "demo", "--dir", str(tmp_path / "demo"))
    ir_path = tmp_path / "demo" / "ir.json"

    def boom(self, ir, ctx):
        raise RuntimeError("secret-looking text https://example.invalid/?token=abc")

    monkeypatch.setattr(Orchestrator, "_ir_validate", boom)
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(RuntimeError, match="secret-looking"), redirect_stdout(out), redirect_stderr(err):
        cli_main(["run", str(ir_path), "--answer", "application=x", "--answer", "jurisdiction=EU"])
    assert "pipeline aborted" in err.getvalue()
    record_path = tmp_path / "demo" / PIPELINE_FILE
    assert record_path.is_file()
    text = record_path.read_text(encoding="utf-8")
    assert "token=abc" not in text and "secret-looking" not in text  # only the type name is stored
    record = load_pipeline_record(tmp_path / "demo")
    assert record.aborted == "RuntimeError" and record.aborted_stage == Stage.IR_BUILD
    assert record.state.current == Stage.IR_BUILD
    assert [o.stage for o in record.state.outcomes] == STAGE_ORDER[: STAGE_ORDER.index(Stage.IR_BUILD)]
    assert record.ir_file_sha256 == sha256_of_file(ir_path)
    # the partial outcomes were printed like a finished run's
    assert "requirement_analysis" in out.getvalue()
    html = render_report_file(ir_path).read_text(encoding="utf-8")
    assert "last run aborted: RuntimeError" in html and "in stage ir_build" in html
    assert "not reached" in html and "no recorded run" in html and "evidence-backed" not in html


def test_load_pipeline_record_rejects_unknown_keys_and_other_versions(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    assert load_pipeline_record(tmp_path / "empty-dir") is None
    record_path = tmp_path / PIPELINE_FILE
    good = json.loads(record_path.read_text(encoding="utf-8"))
    assert load_pipeline_record(tmp_path) is not None

    bad = json.loads(json.dumps(good))
    bad["state"]["outcomes"][0]["bogus"] = 1
    record_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(PipelineRecordError, match=r"unknown key.*state\.outcomes\[0\]\.bogus"):
        load_pipeline_record(tmp_path)
    # the report shows the message instead of guessing: no stage table, no RELEASE outcome
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    assert data.stages is None and "bogus" in (data.stages_reason or "") and "bogus" in data.meta.pipeline_note
    assert data.release.status is None and data.release.note == NO_RECORDED_RUN
    html = render_html(data)
    assert "bogus" in html and "evidence-backed" not in html

    other = json.loads(json.dumps(good))
    other["schema_version"] = "2"
    record_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(PipelineRecordError, match="schema_version '2'"):
        load_pipeline_record(tmp_path)
    record_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PipelineRecordError, match="not readable as JSON"):
        load_pipeline_record(tmp_path)
    record_path.write_text("[]", encoding="utf-8")
    with pytest.raises(PipelineRecordError, match="not a pipeline record object"):
        load_pipeline_record(tmp_path)
    record_path.write_text(json.dumps({"schema_version": "1", "ir_hash": 5}), encoding="utf-8")
    with pytest.raises(PipelineRecordError, match="not a valid pipeline record"):
        load_pipeline_record(tmp_path)


# --------------------------------------------------------------------------- the report of a real offline run


def test_report_of_offline_divider_run(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, state = project
    before = ir_path.read_bytes()
    out = render_report_file(ir_path)
    assert out == tmp_path / "report.html"
    html = out.read_text(encoding="utf-8")
    assert ir_path.read_bytes() == before  # rendering never saves the IR
    ir = CircuitIR.load(ir_path)
    design_hash = ir.content_hash()
    assert design_hash in html and sha256_of_file(ir_path) in html and sha256_of_file(tmp_path / PIPELINE_FILE) in html
    assert PIPELINE_DESCRIBES_IR in html
    for stage in Stage:
        assert str(stage) in html
    for o in state.outcomes:
        assert esc(o.message) in html
    # nothing external, no script, ever
    assert "<script" not in html and "<link" not in html and "@import" not in html and "http://" not in html.replace("https://", "")
    data = build_report_data(ir, ir_path, tmp_path)
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["compile.bom"].status == "PASS" and latest["compile.bom"].freshness == FRESH
    assert latest["compile.bom"].artifact_kind == "bom" and latest["compile.bom"].artifact_hash == ir.artifacts[ArtifactKind.BOM].content_hash
    arts = {a.kind: a for a in data.artifacts}
    assert arts["bom"].path.endswith("bom.csv") and arts["bom"].disk == ON_DISK and arts["bom"].freshness == FRESH
    assert data.repair is not None and data.repair.iterations == "1" and "no repairable failures remain" in data.repair.stopped_reason
    assert "no repairable failures remain" in html
    assert [r.area for r in data.review.rows] == [str(a) for a in ReviewArea]
    pcb_vs_bom = latest["review.pcb_vs_bom"]
    assert pcb_vs_bom.evidence and pcb_vs_bom.evidence[0].path == str(tmp_path / "bom.csv") and pcb_vs_bom.evidence[0].state == EVIDENCE_OK
    assert EVIDENCE_OK in html
    # regulatory: the stored code is 'offline', the reason string is OFFLINE_REASON (in the result message)
    assert [r.source_status for r in ir.regulatory.requirements] == ["offline"] * 4
    assert '"source_status": "offline"' in data.regulatory.ir_json and OFFLINE_REASON in latest["regulatory.sources"].message
    assert esc(OFFLINE_REASON) in html and "reg.EU.LVD.2014-35-EU" in html
    assert {r.check_id for r in data.regulatory.results} >= {"regulatory.sources", "regulatory.applicability", "regulatory.compliance"}
    # components: the existence rows with their sub-checks (in details), the tags as stored, labelled as claims
    assert {"component.existence.R1", "component.existence.R2", "component.fit"} <= {r.check_id for r in data.components.results}
    assert '"name": "symbol"' in latest["component.existence.R1"].details_json and "datasheet_pointer" in html
    assert '"ref": "R1"' in data.components.ir_json and "is a claim" in html
    # RELEASE: the recorded outcome, reasons split from the message; the aggregate line is not a verdict
    release = state.outcome(Stage.RELEASE)
    assert data.release.status == "FAIL" == str(release.status)
    assert data.release.reasons == release.message.removeprefix("not releasable: ").split("; ") and len(data.release.reasons) >= 1
    assert data.release.reasons[0].startswith("overall validation is FAIL (") and "not releasable" not in data.release.reasons[0]
    assert all(esc(r) in html for r in data.release.reasons)
    assert AGGREGATE_NOTE in html and data.validation.aggregate == str(ir.validation.overall())
    # every unstamped agent row was produced by the recorded run
    assert latest["component.existence.R1"].freshness == UNSTAMPED_PRODUCED
    # the optional questions from the stage outcomes carry ready commands
    cmds = {q.key: q.command for q in data.questions}
    assert cmds["mains_powered"] == f"ai-eda run {ir_path} --answer mains_powered=<yes|no>"
    assert esc(cmds["mains_powered"]) in html
    assert all(q.origin.startswith("stage ") for q in data.questions)
    # a PASS without a tool is shown as an opinion (domain.analog.bias here is NOT_VERIFIED without a tool: not an opinion)
    assert not latest["domain.analog.bias"].opinion and OPINION not in html


def test_report_is_byte_identical_for_unchanged_inputs(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    first = render_report_file(ir_path, tmp_path / "a.html").read_text(encoding="utf-8")
    second = render_report_file(ir_path, tmp_path / "b.html").read_text(encoding="utf-8")
    assert first == second
    stored = ir_path.read_text(encoding="utf-8") + (tmp_path / PIPELINE_FILE).read_text(encoding="utf-8")
    stamps = ISO_RE.findall(first)
    assert len(stamps) > 40
    assert all(s in stored for s in stamps), [s for s in stamps if s not in stored][:3]  # no render-time clock anywhere


def test_stale_missing_unstamped_and_carried_over_labels(divider_ir: CircuitIR, tmp_path: Path):
    ir_path, _state = _run_and_record(divider_ir, tmp_path)
    n_first = len(divider_ir.validation.results)
    # run 2 stops after REGULATORY_RESEARCH: it re-produces the regulatory rows, not the component ones
    _run_and_record(divider_ir, tmp_path, stop_after=Stage.REGULATORY_RESEARCH)
    record = load_pipeline_record(tmp_path)
    assert record.results_before == n_first and record.state.outcomes[-1].stage == Stage.REGULATORY_RESEARCH
    ir = CircuitIR.load(ir_path)
    data = build_report_data(ir, ir_path, tmp_path)
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["regulatory.sources"].freshness == UNSTAMPED_PRODUCED
    assert latest["component.existence.R1"].freshness == UNSTAMPED_CARRIED
    assert latest["compile.bom"].freshness == FRESH
    html = render_html(data)
    assert UNSTAMPED_CARRIED in html and UNSTAMPED_PRODUCED in html and UNSTAMPED_UNKNOWN not in html
    assert data.release.status is None and "stopped before RELEASE" in data.release.note

    # the design changes and ir.json is saved without a run: stamped rows and artifacts are stale, the run log describes an older file
    ir.parameters["v_in"] = user_requirement(24.0, "V")
    ir.save(ir_path)
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["compile.bom"].freshness.startswith(f"{STALE} (IR ")
    assert latest["component.existence.R1"].freshness == UNSTAMPED_UNKNOWN
    assert data.meta.pipeline_note == PIPELINE_STALE_IR and data.stages is not None
    assert data.stages.run_hash_label.startswith("recorded for an earlier IR version")
    arts = {a.kind: a for a in data.artifacts}
    assert arts["bom"].freshness == STALE and arts["bom"].disk == ON_DISK

    # files move under the log: the report reports the disk, it does not repair it
    (tmp_path / "bom.csv").unlink()
    with (tmp_path / "cpl.csv").open("a", encoding="utf-8") as f:
        f.write("\n")
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    arts = {a.kind: a for a in data.artifacts}
    assert arts["bom"].disk == MISSING_ON_DISK and arts["cpl"].disk == CHANGED_ON_DISK
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["review.pcb_vs_bom"].evidence[0].state == EVIDENCE_MISSING
    html = render_html(data)
    assert MISSING_ON_DISK in html and CHANGED_ON_DISK in html

    # no pipeline.json at all: no stage table, no RELEASE, unstamped rows are of an unknown run
    (tmp_path / PIPELINE_FILE).unlink()
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    assert data.stages is None and PIPELINE_FILE in (data.stages_reason or "")
    assert data.release.status is None and data.release.note == NO_RECORDED_RUN
    assert {r.freshness for r in data.validation.latest if r.check_id.startswith("component.existence")} == {UNSTAMPED_UNKNOWN}
    html = render_html(data)
    assert NO_RECORDED_RUN in html and "evidence-backed" not in html


def test_evidence_changed_on_disk_is_reported(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    with (tmp_path / "bom.csv").open("a", encoding="utf-8") as f:
        f.write("# edited by hand\n")
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["review.pcb_vs_bom"].evidence[0].state == CHANGED_ON_DISK
    assert {a.kind: a.disk for a in data.artifacts}["bom"] == CHANGED_ON_DISK


# --------------------------------------------------------------------------- hand-built inputs


def _plain_ir(tmp_path: Path, name: str = "p") -> CircuitIR:
    return CircuitIR(project=ProjectMeta(id="p", name=name, workdir=str(tmp_path)))


def _record(tmp_path: Path, ir: CircuitIR, outcomes: list[StageOutcome], **kw) -> Path:
    ir_path = ir.save(tmp_path / "ir.json")
    state = PipelineState(outcomes=outcomes, current=outcomes[-1].stage if outcomes else None)
    save_pipeline_record(state, ir, ir_path, tmp_path, results_before=kw.pop("results_before", 0),
                         ir_file_sha256=sha256_of_file(ir_path), **kw)
    return ir_path


def test_release_section_renders_pass_verbatim(tmp_path: Path):
    ir = _plain_ir(tmp_path)
    ir.validation.add(ValidationResult(check_id="x.tool", status=ValidationStatus.PASS, tool="x", ir_hash=ir.content_hash()))
    ir_path = _record(tmp_path, ir, [StageOutcome(stage=s, status=ValidationStatus.PASS) for s in STAGE_ORDER[:-1]]
                      + [StageOutcome(stage=Stage.RELEASE, status=ValidationStatus.PASS, message="evidence-backed release")])
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    assert data.release.status == "PASS" and data.release.reasons == ["evidence-backed release"]
    html = render_html(data)
    assert "evidence-backed release" in html and "not releasable" not in html
    assert '<span class="st st-PASS">PASS</span>' in html


def test_pass_without_tool_is_labelled_opinion(tmp_path: Path):
    ir = _plain_ir(tmp_path)
    ir.validation.add(ValidationResult(check_id="someone.says", status=ValidationStatus.PASS, message="looks fine", ir_hash=ir.content_hash()))
    ir.validation.add(ValidationResult(check_id="tool.says", status=ValidationStatus.PASS, message="checked", tool="t", ir_hash=ir.content_hash()))
    ir_path = ir.save(tmp_path / "ir.json")
    data = build_report_data(ir, ir_path, tmp_path)
    latest = {r.check_id: r for r in data.validation.latest}
    assert latest["someone.says"].opinion and not latest["tool.says"].opinion
    assert render_html(data).count(OPINION) == 1
    # NOT_VERIFIED stays NOT_VERIFIED and an empty log aggregates to NOT_VERIFIED, never PASS
    empty = _plain_ir(tmp_path)
    p2 = empty.save(tmp_path / "empty" / "ir.json")
    d2 = build_report_data(empty, p2, tmp_path / "empty")
    assert d2.validation.aggregate == "NOT_VERIFIED" and d2.release.status is None


def test_questions_have_answer_commands_and_model_labels(tmp_path: Path):
    ir = _plain_ir(tmp_path)
    ir.requirements.missing = [
        MissingInformation(key="application", question="What is it for?", required=True),
        MissingInformation(key="mains_powered", question="Mains?", required=False, options=["yes", "no"]),
        MissingInformation(key="k with space", question="Model asks?", required=False, options=["a", "b"], source="llm"),
    ]
    ir_path = _record(tmp_path, ir, [StageOutcome(stage=Stage.REQUIREMENT_ANALYSIS, status=ValidationStatus.USER_INPUT_REQUIRED,
                                                   questions=[MissingInformation(key="radio", question="Radio?", options=["yes", "no"], required=False)])])
    data = build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path)
    rows = {q.key: q for q in data.questions}
    assert rows["application"].command == f"ai-eda run {ir_path} --answer application=<value>" and rows["application"].required
    assert rows["mains_powered"].command.endswith("--answer mains_powered=<yes|no>")
    # a model question's options are model output too: the placeholder, never a pre-filled option
    assert rows["k with space"].command.endswith("--answer k with space=<value>") and rows["k with space"].source_label == "model question"
    assert rows["radio"].origin == "stage requirement_analysis" and rows["application"].origin == "ir.requirements.missing"
    html = render_html(data)
    assert "model question" in html and "(model output)" in html and "&lt;yes|no&gt;" in html


def test_every_untrusted_string_is_escaped(tmp_path: Path):
    hostile = "<script>alert(1)</script>"
    img = '"><img src=x onerror=alert(1)>'
    td = "</td><script>x</script>"
    ir = CircuitIR(project=ProjectMeta(id="p", name=hostile, description=img, workdir=str(tmp_path)))
    ir.requirements.raw_input = f"request {hostile}"
    ir.requirements.corrections = [img]
    ir.requirements.requirements = [Requirement(id="req.x", key="x", text=hostile, kind=RequirementKind.EXPLICIT, category=img)]
    ir.requirements.missing = [MissingInformation(key=f"k{img}", question=img, source="llm", options=[td], rationale=hostile)]
    ir.parameters[f"p{hostile}"] = user_requirement(1.0, f"V{img}")
    from tests.conftest import make_component

    c = make_component("R1", "<b>x</b>")
    c.description = td
    ir.components = [c]
    ir.regulatory.requirements = [
        RegulatoryRequirement(id="reg.x", jurisdiction="EU", title=hostile, summary=img,
                              provenance=RegulatoryProvenance(jurisdiction="EU", authority=td, source_title=hostile),
                              grounded_quotes=[GroundedQuote(section=img, quote=hostile, reason=td)])
    ]
    ir.artifacts[ArtifactKind.BOM] = ArtifactRef(kind=ArtifactKind.BOM, path=f"{tmp_path}/{td}.csv", content_hash="sha256:0", generated_from_ir_hash="x",
                                                generator=hostile, notes=[img])
    ir.validation.add(
        ValidationResult(check_id="check.x", status=ValidationStatus.FAIL, message=hostile, tool=img, tool_version=td,
                         details={"k": "<svg onload=alert(1)>", hostile: [img]}, evidence=[Evidence(description=td, path=f"{tmp_path}/{td}", url=img)])
    )
    ir.validation.add(ValidationResult(check_id="repair.loop", status=ValidationStatus.FAIL, message=img, tool="repair.loop",
                                       details={"iterations": 1, "stopped_reason": hostile, "actions": [td], "unresolved": [], "final_review": {}}))
    ir.validation.add(ValidationResult(check_id=str(ReviewArea.ERC), status=ValidationStatus.FAIL, message=td, tool="independent_reviewer"))
    outcomes = [StageOutcome(stage=Stage.REQUIREMENT_ANALYSIS, status=ValidationStatus.FAIL, message=hostile,
                             questions=[MissingInformation(key="q", question=img, options=[hostile])]),
                StageOutcome(stage=Stage.RELEASE, status=ValidationStatus.FAIL, message=f"not releasable: {td}; {img}")]
    ir_path = _record(tmp_path, ir, outcomes)
    html = render_html(build_report_data(CircuitIR.load(ir_path), ir_path, tmp_path))
    assert "<script" not in html and "<svg" not in html and "<img" not in html and "<b>x</b>" not in html
    assert html.count("onerror=") == html.count("&quot;&gt;&lt;img src=x onerror=") > 10
    assert html.count("&lt;script&gt;") >= 3 and "&lt;/td&gt;" in html
    assert esc('<a href="x">') == "&lt;a href=&quot;x&quot;&gt;" and esc(None) == "" and esc(5) == "5"
    # attributes are escaped too: the title and the status classes are built from data
    assert "<title>&lt;script&gt;alert(1)&lt;/script&gt; - design report</title>" in html
    # the hostile strings appear (escaped): nothing was silently dropped
    assert html.count(esc(hostile)) >= 12 and html.count(esc(td)) >= 8


# --------------------------------------------------------------------------- CLI


def test_cli_report_paths_and_exit_codes(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    before = ir_path.read_bytes()
    record_before = (tmp_path / PIPELINE_FILE).read_bytes()
    code, out, err = _cli("report", str(tmp_path / "missing.json"))
    assert code == 2 and "missing.json" in err and out == ""
    for target in (ir_path, tmp_path / PIPELINE_FILE, tmp_path / "sub" / ".." / "ir.json"):
        code, out, err = _cli("report", str(ir_path), "-o", str(target))
        assert code == 2 and "refusing" in err and out == "", target
    assert ir_path.read_bytes() == before and (tmp_path / PIPELINE_FILE).read_bytes() == record_before
    code, out, err = _cli("report", str(ir_path), "-o", str(tmp_path))  # a directory: unwritable
    assert code == 2 and "could not write" in err
    code, out, err = _cli("report", str(ir_path))
    assert code == 0 and out.strip() == f"wrote {tmp_path / 'report.html'}" and err == ""
    assert (tmp_path / "report.html").read_bytes() == render_report_file(ir_path, tmp_path / "again.html").read_bytes()
    assert ir_path.read_bytes() == before
    code, out, _ = _cli("report", str(ir_path), "-o", str(tmp_path / "custom" / "r.html"))
    assert code == 2  # the parent directory does not exist: OSError, reported, exit 2 - never 1
    (tmp_path / "custom").mkdir()
    code, out, _ = _cli("report", str(ir_path), "-o", str(tmp_path / "custom" / "r.html"))
    assert code == 0 and (tmp_path / "custom" / "r.html").is_file()
    # a design whose RELEASE is FAIL still exits 0: the report is not a verdict
    assert _state.outcome(Stage.RELEASE).status is ValidationStatus.FAIL
    # IR errors are exit 2: an unknown key, a relative workdir that names another directory
    broken = tmp_path / "broken" / "ir.json"
    broken.parent.mkdir()
    raw = json.loads(before)
    raw["bogus"] = 1
    broken.write_text(json.dumps(raw), encoding="utf-8")
    code, _, err = _cli("report", str(broken))
    assert code == 2 and "unknown key" in err
    raw = json.loads(before)
    raw["project"]["workdir"] = "elsewhere/other"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    code, _, err = _cli("report", str(broken))
    assert code == 2 and "is relative" in err
    code, _, err = _cli("serve", str(broken))
    assert code == 2 and "is relative" in err
    code, _, err = _cli("serve", str(tmp_path / "missing.json"))
    assert code == 2 and "missing.json" in err
    broken.write_text("{", encoding="utf-8")  # not JSON at all: an IR error, exit 2, no traceback
    for cmd in ("report", "serve"):
        code, out, err = _cli(cmd, str(broken))
        assert code == 2 and str(broken) in err and out == "", cmd


# --------------------------------------------------------------------------- server


def test_host_header_rule():
    for ok in ("127.0.0.1", "127.0.0.1:8765", "localhost", "LOCALHOST:80", "[::1]", "[::1]:1"):
        assert host_allowed(ok), ok
    for bad in (None, "", "evil.example", "127.0.0.1.evil", "127.0.0.1:abc", "[::1", "localhost:80:1", "127.0.0.1 evil", "::1"):
        assert not host_allowed(bad), bad


def _get(port: int, path: str = "/", headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def test_serve_loopback_read_only_and_live(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    before = ir_path.read_bytes()
    server = ReportServer(ir_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.port
        assert port > 0 and server.url == f"http://127.0.0.1:{port}/"
        status, headers, body = _get(port)  # http.client sends Host: 127.0.0.1:<port>
        assert status == 200 and headers["content-type"] == "text/html; charset=utf-8"
        assert headers["cache-control"] == "no-store" and headers["server"] == "ai-eda-report" and "Python" not in headers["server"]
        page = body.decode("utf-8")
        assert page == render_report_file(ir_path, tmp_path / "cmp.html").read_text(encoding="utf-8")
        assert "<script" not in page
        for path in ("/ir.json", "/pipeline.json", "/../ir.json", "/bom.csv", "/index.html"):
            status, _h, body = _get(port, path)
            assert status == 404 and b"<" not in body, path
        status, _h, body = _get(port, "/?x=1")
        assert status == 200
        # DNS-rebinding guard: a Host that is not this machine gets 421 and nothing else
        for host in ("evil.example", "evil.example:%d" % port, "127.0.0.1.evil.example", ""):
            status, headers, body = _get(port, headers={"Host": host})
            assert status == 421 and body == b"" and headers["content-length"] == "0", host
        for host in (f"localhost:{port}", "127.0.0.1", f"[::1]:{port}"):
            assert _get(port, headers={"Host": host})[0] == 200, host
        # live: a change to ir.json is visible on the next request, without restarting
        ir = CircuitIR.load(ir_path)
        ir.project.name = "renamed-live"
        ir.save(ir_path)
        status, _h, body = _get(port)
        assert status == 200 and b"renamed-live - design report" in body and PIPELINE_STALE_IR.encode() in body
        # a half-written ir.json is a 503 with one line, never a traceback; the server keeps serving
        ir_path.write_text("{", encoding="utf-8")
        status, headers, body = _get(port)
        assert status == 503 and body.startswith(b"report unavailable: ") and b"Traceback" not in body and body.count(b"\n") == 1
        assert headers["content-type"].startswith("text/plain")
        ir_path.write_bytes(before)
        assert _get(port)[0] == 200
        assert ir_path.read_bytes() == before  # serving never saves the IR
        # the port is taken: serve() reports the OSError and returns 2 instead of raising
        err = io.StringIO()
        with redirect_stderr(err):
            assert serve(ir_path, port) == 2
        assert "could not listen on 127.0.0.1" in err.getvalue()
    finally:
        server.shutdown()
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_serve_returns_zero_on_keyboard_interrupt(project: tuple[Path, PipelineState], monkeypatch: pytest.MonkeyPatch):
    ir_path, _state = project

    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(ReportServer, "serve_forever", interrupted)
    code, out, err = _cli("serve", str(ir_path), "--port", "0")
    assert code == 0 and "http://127.0.0.1:" in out and "127.0.0.1 only" in out and err == ""


def test_pipeline_json_is_not_an_artifact_and_not_hashed(project: tuple[Path, PipelineState], tmp_path: Path):
    ir_path, _state = project
    ir = CircuitIR.load(ir_path)
    assert all(not Path(a.path).name == PIPELINE_FILE for a in ir.artifacts.values())
    assert PIPELINE_FILE not in json.dumps(ir.design_dict())
    h = ir.content_hash()
    (tmp_path / PIPELINE_FILE).unlink()
    assert CircuitIR.load(ir_path).content_hash() == h
