"""The GUI's projects root and run launcher (``ai_eda/gui/projects.py``, ``ai_eda/gui/runs.py``).

Projects: a folder under the root is a project when it holds ir.json; ``create``
goes through :func:`ai_eda.cli.new_project`, the code path of ``ai-eda new``;
a copied project whose recorded workdir is elsewhere is listed with a warning,
an unreadable one with its error only. Runs: the form is rendered only by the
CLI's own :func:`~ai_eda.cli.run_option_flags` and refused before any process
starts; real subprocesses of the CLI (a fresh project blocks on its first
question within a second or two) write ``<workdir>/gui/runs/<NNN>-<kind>.log``
and the questions appear in the project afterwards; a sleeping fake process
(``python_args``) holds the one-run-per-project slot for the 409. The ``claude``
provider is the fake of ``tests/fake_claude_cli.py``, named explicitly with
``llm_claude_cli`` because the child process does not see this suite's
discovery monkeypatch.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import ai_eda.cli as cli
from ai_eda.cli import RUN_OPTIONS, build_parser, model_pin_refusal, parse_answers, run_option_flags
from ai_eda.gui.projects import (
    PROJECT_NAME_PATTERN,
    ProjectError,
    ProjectNotFoundError,
    ProjectsRoot,
    valid_project_name,
)
from ai_eda.gui.runs import (
    ANSWER_KEY_PATTERN,
    CHILD_ENV_ADDITIONS,
    CLI_PYTHON_ARGS,
    COMMAND_PREFIX,
    EXIT_PREFIX,
    LOG_DIR,
    RUN_KINDS,
    RunConflictError,
    RunManager,
    RunNotFoundError,
    RunRequestError,
    RunStartError,
    allowed_options,
    child_env,
    command_line,
    form_args,
)
from ai_eda.ir import CircuitIR
from ai_eda.report.data import RUN_CURRENT_IR
from tests.fake_claude_cli import DEFAULT_MODEL, MARKER_ENV, FakeClaudeCli, success_structured

DATA = Path(__file__).parent / "data" / "fake_llm_requirements.json"
LLM_REQUEST = json.loads(DATA.read_text(encoding="utf-8"))["request"]
EXTRACTION = json.loads(DATA.read_text(encoding="utf-8"))["responses"][0]["structured"]
CLAUDE_SPEC = f"claude:{DEFAULT_MODEL}"
REQUEST = "5 V 입력, 1 kHz 구형파 발진기"
#: what every GUI run starts with (RunManager's default python_args): -P keeps the workdir off the child's sys.path
CLI_PREFIX = [sys.executable, "-P", "-m", "ai_eda.cli"]
#: a real CLI run of a small project finishes in about a second; generous for a slow CI machine
RUN_TIMEOUT = 120.0
#: a fake run that holds its project's slot for two seconds, printing on both streams (Korean included)
SLEEPER = [
    sys.executable, "-c",
    "import sys, time; print('out-1 한글'); print('err-1', file=sys.stderr); time.sleep(2); print('out-2')",
]


@pytest.fixture
def root(tmp_path: Path) -> ProjectsRoot:
    return ProjectsRoot(tmp_path / "projects")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClaudeCli:
    """The fake ``claude`` first on PATH (the child inherits PATH), no OpenRouter key."""
    f = FakeClaudeCli(tmp_path / "bin")
    f.install(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return f


def _log_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _assert_command_line(lines: list[str], argv: list[str]) -> None:
    """The log's first line is exactly ``argv`` (and, on POSIX, splits back into it: every value one argv item)."""
    assert lines[0] == COMMAND_PREFIX + command_line(argv)
    if os.name != "nt":
        assert shlex.split(lines[0][len(COMMAND_PREFIX):]) == argv


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as e:  # Windows without the symlink privilege
        pytest.skip(f"cannot create a symbolic link here: {e}")


# --------------------------------------------------------------------------- projects


def test_an_empty_or_missing_root_lists_nothing(tmp_path: Path):
    assert ProjectsRoot(tmp_path / "missing").list() == []
    (tmp_path / "empty").mkdir()
    assert ProjectsRoot(tmp_path / "empty").list() == []
    assert ProjectsRoot(tmp_path / "empty").root == (tmp_path / "empty").resolve()


def test_create_writes_what_ai_eda_new_writes_through_the_same_function(tmp_path: Path, root: ProjectsRoot, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []
    real = cli.new_project

    def spy(name, request=None, workdir=None):
        calls.append((name, request, Path(workdir) if workdir is not None else None))
        return real(name, request, workdir)

    monkeypatch.setattr(cli, "new_project", spy)
    info = root.create("demo", REQUEST)
    folder = root.root / "demo"
    assert calls == [("demo", REQUEST, folder)]
    assert info.name == "demo" and info.folder == folder and info.ir_path == folder / "ir.json" and info.error is None
    assert info.workdir == folder and info.recorded_workdir == str(folder) and not info.workdir_mismatch and info.workdir_exists
    assert info.warning is None and info.last_run is None
    ir = CircuitIR.load(info.ir_path)
    assert info.design_hash == ir.content_hash()
    assert (ir.project.id, ir.project.name, ir.project.workdir, ir.requirements.raw_input) == ("demo", "demo", str(folder), REQUEST)
    assert sorted(p.name for p in folder.iterdir()) == ["ir.json"]

    # `ai-eda new` calls the same function; its output is unchanged, and its default folder is projects/<name> under the cwd
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli.main(["new", "other", "--request", REQUEST]) == 0
    assert out.getvalue() == f"created {folder.parent / 'other' / 'ir.json'}\n"
    assert calls[-1] == ("other", REQUEST, None)
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli.main(["new", "third", "--dir", str(tmp_path / "elsewhere")]) == 0
    assert out.getvalue() == f"created {tmp_path.resolve() / 'elsewhere' / 'ir.json'}\n"
    assert calls[-1] == ("third", None, tmp_path / "elsewhere")
    cli_ir = CircuitIR.load(folder.parent / "other" / "ir.json")
    gui_ir = CircuitIR.load(info.ir_path)
    skip = {"project": {"id", "name", "workdir", "created_at"}}
    assert cli_ir.model_dump(exclude=skip) == gui_ir.model_dump(exclude=skip)
    # the root lists both, by folder name; the CLI-made one is a project like any other
    assert [p.name for p in root.list()] == ["demo", "other"]


@pytest.mark.parametrize(
    "name",
    ["", "-demo", "_demo", ".demo", "a b", "../x", "x/y", "x\\y", "a" * 65, "데모", "demo\n", "demo.1", "CON", "nul", "Com1", "lpt9"],
)
def test_create_refuses_a_bad_name_and_writes_nothing(root: ProjectsRoot, name: str):
    assert not valid_project_name(name)
    with pytest.raises(ProjectError, match="프로젝트 이름은"):
        root.create(name, REQUEST)
    assert not root.root.exists() or list(root.root.iterdir()) == []


def test_name_rule_edges_and_non_text_input(root: ProjectsRoot):
    assert PROJECT_NAME_PATTERN == r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
    for good in ("a", "0", "A" * 64, "osc_demo", "astable-1khz", "con1", "console"):
        assert valid_project_name(good), good
    assert not valid_project_name(None) and not valid_project_name(7)
    with pytest.raises(ProjectError):
        root.create(123, REQUEST)  # type: ignore[arg-type]
    with pytest.raises(ProjectError, match="요청문"):
        root.create("demo", ["not", "text"])  # type: ignore[arg-type]
    assert not (root.root / "demo").exists()
    assert CircuitIR.load(root.create("blank", None).ir_path).requirements.raw_input == ""


def test_create_refuses_an_existing_folder_file_or_link_and_leaves_it_alone(root: ProjectsRoot, tmp_path: Path):
    root.root.mkdir(parents=True)
    taken = root.root / "taken"
    taken.mkdir()
    (taken / "notes.txt").write_text("mine", encoding="utf-8")
    (root.root / "afile").write_text("x", encoding="utf-8")
    existing = root.create("demo", REQUEST)
    before = existing.ir_path.read_bytes()
    for name in ("taken", "afile", "demo"):
        with pytest.raises(ProjectError, match="이미 있는 폴더"):
            root.create(name, "another request")
    assert sorted(p.name for p in taken.iterdir()) == ["notes.txt"] and (taken / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert existing.ir_path.read_bytes() == before
    _symlink_or_skip(root.root / "linked", tmp_path / "nowhere", directory=True)
    with pytest.raises(ProjectError, match="이미 있는 폴더"):
        root.create("linked", REQUEST)
    assert not (tmp_path / "nowhere").exists()


def test_a_folder_without_ir_json_is_not_a_project(root: ProjectsRoot):
    root.create("real", REQUEST)
    (root.root / "empty").mkdir()
    (root.root / "docs").mkdir()
    (root.root / "docs" / "ir.json.bak").write_text("{}", encoding="utf-8")
    (root.root / "ir.json").write_text("{}", encoding="utf-8")  # a file at the root is not a project folder
    assert [p.name for p in root.list()] == ["real"]
    for name in ("empty", "docs", "nope", "../projects", ""):
        with pytest.raises(ProjectNotFoundError):
            root.get(name)
    assert root.get("real").error is None


def test_a_mismatched_recorded_workdir_is_listed_with_the_warning(root: ProjectsRoot, tmp_path: Path):
    alpha = root.create("alpha", REQUEST)
    shutil.copytree(alpha.folder, root.root / "beta")
    listed = {p.name: p for p in root.list()}
    beta = listed["beta"]
    # listed by its own folder and confined to it: the recorded (absolute) workdir is alpha's and is never followed
    assert beta.error is None and beta.folder == root.root / "beta" and beta.ir_path == root.root / "beta" / "ir.json"
    assert beta.workdir == root.root / "beta" and beta.workdir_mismatch and beta.workdir_exists
    assert beta.recorded_workdir == str(alpha.folder)
    assert beta.warning is not None and "기록된 작업 폴더" in beta.warning and str(alpha.folder) in beta.warning
    assert "ai-eda relocate" in beta.warning and str(beta.ir_path) in beta.warning
    assert not listed["alpha"].workdir_mismatch and listed["alpha"].warning is None
    # a run of it is refused before anything starts, as the CLI refuses it: nothing is written in either folder
    with pytest.raises(RunRequestError, match="relocate"):
        RunManager(python_args=SLEEPER).start_project("run", beta)
    with pytest.raises(RunRequestError, match="ai-eda relocate"):  # the manager's own placement: the CLI's message
        RunManager(python_args=SLEEPER).start("run", beta.ir_path, [])
    assert not (root.root / "beta" / "gui").exists() and not (alpha.folder / "gui").exists()
    # the recorded workdir no longer exists: the same mismatch - still listed (not hidden), confined, refused
    ir = CircuitIR.load(beta.ir_path)
    ir.project.workdir = str(tmp_path / "gone")
    ir.save(beta.ir_path)
    beta = root.get("beta")
    assert beta.workdir_mismatch and beta.workdir == root.root / "beta" and beta.workdir_exists and beta.last_run is None
    assert "ai-eda relocate" in (beta.warning or "")
    with pytest.raises(RunRequestError, match="relocate"):
        RunManager(python_args=SLEEPER).start_project("run", beta)
    assert not (tmp_path / "gone").exists()
    # a relative workdir naming the folder is the folder; one that does not is refused by project_workdir: error only
    ir.project.workdir = "beta"
    ir.save(beta.ir_path)
    beta = root.get("beta")
    assert beta.error is None and beta.workdir == root.root / "beta" and not beta.workdir_mismatch
    ir.project.workdir = "elsewhere/beta"
    ir.save(beta.ir_path)
    beta = root.get("beta")
    assert beta.error is not None and "relative" in beta.error and beta.workdir is None and beta.design_hash is None
    with pytest.raises(RunRequestError, match="실행할 수 없는 프로젝트"):
        RunManager(python_args=SLEEPER).start_project("run", beta)


def test_an_unreadable_project_is_listed_with_its_error_and_nothing_else(root: ProjectsRoot):
    root.create("good", REQUEST)
    for name, text in (("garbage", "{not json"), ("oldschema", json.dumps({"schema_version": "0", "project": {}})), ("typo", None)):
        folder = root.root / name
        folder.mkdir()
        if text is None:  # a key the models would drop: CircuitIR.load refuses it rather than truncating the design
            data = json.loads((root.root / "good" / "ir.json").read_text(encoding="utf-8"))
            data["project"]["workdir"] = str(folder)
            data["serves_requirement"] = []
            text = json.dumps(data)
        (folder / "ir.json").write_text(text, encoding="utf-8")
    unsafe = [] if os.name == "nt" else ["a\\b", "nul"]  # names Windows cannot even create: not one safe path component
    for name in unsafe:
        (root.root / name).mkdir()
        shutil.copy(root.root / "good" / "ir.json", root.root / name / "ir.json")
    listed = {p.name: p for p in root.list()}
    assert sorted(listed) == sorted(["garbage", "good", "oldschema", "typo", *unsafe])
    assert listed["good"].error is None
    for name in ("garbage", "oldschema", "typo", *unsafe):
        p = listed[name]
        assert p.error, name
        assert (p.workdir, p.design_hash, p.last_run, p.warning, p.workdir_mismatch) == (None, None, None, None, False), name
        assert "\n" not in p.error
    assert listed["garbage"].error.startswith("ir.json을 읽을 수 없습니다")
    assert "schema_version" in listed["oldschema"].error and "serves_requirement" in listed["typo"].error
    for name in unsafe:
        assert "경로의 한 칸" in listed[name].error
        with pytest.raises(ProjectNotFoundError):
            root.get(name)
    assert root.get("garbage").error == listed["garbage"].error


@pytest.mark.parametrize("name", ["osc.v2", "발진기", "my project", "-lead"])
def test_a_project_ai_eda_new_made_under_any_name_opens_and_runs(root: ProjectsRoot, name: str):
    """``ai-eda new`` accepts any name: the GUI lists, opens and runs such a folder like its own (the strict rule is for create only)."""
    root.root.mkdir(parents=True)
    with redirect_stdout(io.StringIO()):
        assert cli.main(["new", "--dir", str(root.root / name), "--request", REQUEST, "--", name]) == 0
    assert not valid_project_name(name)
    (listed,) = root.list()
    assert listed.name == name and listed.error is None and listed.workdir == root.root / name and not listed.workdir_mismatch
    info = root.get(name)
    assert info == listed
    manager = RunManager(python_args=[sys.executable, "-c", "print('ran')"])
    handle = manager.start_project("run", info)
    assert manager.wait(handle, 30) == 0 and _log_lines(handle.log_path)[1:] == ["ran", f"{EXIT_PREFIX}0"]
    _assert_command_line(_log_lines(handle.log_path), [sys.executable, "-c", "print('ran')", "run", str(info.ir_path)])
    with pytest.raises(ProjectError, match="프로젝트 이름은"):
        root.create(f"{name}-2", REQUEST)  # the GUI itself still creates only rule-abiding names


def test_a_symbolic_link_is_listed_with_an_error_and_never_read(root: ProjectsRoot, tmp_path: Path):
    outside = ProjectsRoot(tmp_path / "outside").create("secret", REQUEST)
    root.root.mkdir(parents=True)
    _symlink_or_skip(root.root / "linked", outside.folder, directory=True)
    (root.root / "filelink").mkdir()
    _symlink_or_skip(root.root / "filelink" / "ir.json", outside.ir_path, directory=False)
    listed = {p.name: p for p in root.list()}
    assert "심볼릭 링크" in (listed["linked"].error or "") and listed["linked"].workdir is None
    assert "심볼릭 링크" in (listed["filelink"].error or "") and listed["filelink"].design_hash is None


def test_an_unreadable_pipeline_json_is_the_last_run_error(root: ProjectsRoot):
    info = root.create("demo", REQUEST)
    (info.workdir / "pipeline.json").write_text("[1, 2]", encoding="utf-8")
    last = root.get("demo").last_run
    assert last is not None and last.error and "not a pipeline record" in last.error
    assert (last.blocked, last.last_stage, last.open_questions) == (False, None, [])


# --------------------------------------------------------------------------- the form


def test_every_form_field_is_rendered_by_run_option_flags():
    options = {
        "llm": "claude,openrouter", "llm_model": "claude:model-a",
        "llm_fallback": ["same", "", "openrouter:vendor/model-b"], "llm_task_model": {"review": "claude:model-c"},
        "llm_allow_model_change": True, "llm_claude_cli": "/opt/my tools/claude", "llm_claude_fallback_model": "model-d",
        "llm_budget_usd": "0.05", "llm_budget_tokens": 5000.0, "online": True, "trust_host": ["www.ti.com", "example.org"],
        "datasheet_url": {"R1": "https://example.org/r.pdf?a=b&c=d"}, "source_url": ["eu.lvd=https://eur-lex.europa.eu/x"],
        "sources_dir": "/tmp/src dir", "catalog": "my catalog.csv", "catalog_date": "2026-09-01",
        "catalog_authority": "JLCPCB export", "catalog_supplier": "JLCPCB", "regulatory_candidates": "cands.json",
        "fab_capability": "fab cap.json", "no_pdf": True, "browser": "/opt/chrome",
    }
    # the form covers every run option but answer (a new RUN_OPTIONS entry makes this test name it)
    assert set(options) == set(allowed_options("run")) == {o.dest for o in RUN_OPTIONS} - {"answer"}
    answers = {"application": "hobby board = demo", "highest_rated_voltage": "12 V DC", "pcb.placement": "skip", "protection": ""}
    args = form_args("run", answers, options)
    expected_values = {
        **options,
        "answer": {"application": "hobby board = demo", "highest_rated_voltage": "12 V DC", "pcb.placement": "skip"},
        "llm_fallback": ["same", "openrouter:vendor/model-b"], "llm_budget_usd": 0.05, "llm_budget_tokens": 5000,
    }
    assert args == run_option_flags(**expected_values)
    assert args[:2] == ["--answer", "application=hobby board = demo"]  # one argv item: spaces and '=' kept verbatim
    ns = build_parser().parse_args(["run", "ir.json", *args])
    assert parse_answers(ns.answer) == expected_values["answer"]
    assert ns.llm == "claude,openrouter" and ns.llm_model == "claude:model-a" and ns.llm_allow_model_change is True
    assert ns.llm_fallback == ["same", "openrouter:vendor/model-b"] and ns.llm_task_model == ["review=claude:model-c"]
    assert ns.llm_claude_cli == "/opt/my tools/claude" and ns.llm_claude_fallback_model == "model-d"
    assert ns.llm_budget_usd == 0.05 and ns.llm_budget_tokens == 5000 and ns.online is True and ns.no_pdf is True
    assert ns.trust_host == ["www.ti.com", "example.org"] and ns.datasheet_url == ["R1=https://example.org/r.pdf?a=b&c=d"]
    assert ns.source_url == ["eu.lvd=https://eur-lex.europa.eu/x"] and ns.sources_dir == "/tmp/src dir"
    assert (ns.catalog, ns.catalog_date, ns.catalog_authority, ns.catalog_supplier) == ("my catalog.csv", "2026-09-01", "JLCPCB export", "JLCPCB")
    assert ns.regulatory_candidates == "cands.json" and ns.fab_capability == "fab cap.json" and ns.browser == "/opt/chrome"
    # an empty field adds nothing: the GUI never adds a flag the user did not set
    empty = {"llm": "", "llm_model": None, "online": False, "trust_host": [], "llm_budget_usd": "", "llm_budget_tokens": None, "llm_task_model": {}}
    assert form_args("run", {"application": ""}, empty) == [] and form_args("run") == []
    assert form_args("run", None, {"llm_budget_usd": 1, "llm_budget_tokens": "250"}) == ["--llm-budget-usd", "1.0", "--llm-budget-tokens", "250"]
    # the other kinds take what their own subcommand parser accepts
    assert allowed_options("stage-reports") == ("no_pdf", "browser") and allowed_options("review") == ()
    sr = form_args("stage-reports", None, {"no_pdf": True, "browser": "/opt/chrome"})
    assert sr == ["--no-pdf", "--browser", "/opt/chrome"]
    ns = build_parser().parse_args(["stage-reports", "ir.json", *sr])
    assert ns.no_pdf is True and ns.browser == "/opt/chrome"
    assert form_args("review") == [] and form_args("review", {}, {}) == []
    assert RUN_KINDS == ("run", "review", "stage-reports") and ANSWER_KEY_PATTERN == r"^[a-z0-9_.]+$"


@pytest.mark.parametrize(
    ("kind", "answers", "options", "message"),
    [
        ("delete", None, None, "알 수 없는 실행 종류"),
        ("run", None, {"nope": 1}, "run에 없는 실행 옵션: nope"),
        ("run", None, {"answer": ["application=x"]}, "답변은 answers로"),
        ("run", None, ["llm"], "options:"),
        ("run", ["application=x"], None, "answers:"),
        ("run", {"Application": "x"}, None, "답변 키 'Application'"),
        ("run", {"a b": "x"}, None, "답변 키 'a b'"),
        ("run", {"a=b": "x"}, None, "답변 키 'a=b'"),
        ("run", {"적용": "x"}, None, "답변 키"),
        ("run", {"": "x"}, None, "답변 키"),
        ("run", {"application\n": "x"}, None, "답변 키"),
        ("run", {"application": 5}, None, "문자열이어야"),
        ("run", {"application": "two\nlines"}, None, "제어 문자"),
        ("run", {"application": "nul\x00byte"}, None, "제어 문자"),
        ("run", None, {"online": "yes"}, "참/거짓"),
        ("run", None, {"llm_budget_usd": "abc"}, "숫자가 아닙니다"),
        ("run", None, {"llm_budget_usd": "nan"}, "유한한 숫자"),
        ("run", None, {"llm_budget_usd": float("inf")}, "유한한 숫자"),
        ("run", None, {"llm_budget_usd": True}, "숫자여야"),
        ("run", None, {"llm_budget_tokens": 1.5}, "정수여야"),
        ("run", None, {"llm_model": "--online"}, "'-'로 시작하는 값"),
        ("run", None, {"llm_fallback": ["same", "-x"]}, "'-'로 시작하는 값"),
        ("run", None, {"llm_task_model": {"-review": "claude:model-c"}}, "'-'로 시작하는 값"),
        ("run", None, {"llm_task_model": {"a=b": "claude:model-c"}}, "'='"),
        ("run", None, {"trust_host": {"a": "b"}}, "목록이어야"),
        ("run", None, {"trust_host": "a\tb"}, "제어 문자"),
        ("run", None, {"catalog": 5}, "문자열이어야"),
        ("run", None, {"llm_fallback": 5}, "목록이어야"),
        ("review", {"application": "x"}, None, "review는 답변을 받지 않습니다"),
        ("review", None, {"no_pdf": True}, "review에 없는 실행 옵션: no_pdf"),
        ("stage-reports", None, {"online": True}, "stage-reports에 없는 실행 옵션: online"),
        ("stage-reports", {"confirm_design": "yes"}, None, "답변을 받지 않습니다"),
    ],
)
def test_a_refused_form_starts_nothing_and_writes_nothing(root: ProjectsRoot, kind: str, answers, options, message: str):
    info = root.create("demo", REQUEST)
    marker = info.workdir / "STARTED"
    manager = RunManager(python_args=[sys.executable, "-c", "open('STARTED', 'w').close()"])
    with pytest.raises(RunRequestError, match=message.replace("(", r"\(")):
        manager.start_project(kind, info, answers, options)
    assert manager.active(info.workdir) is None and manager.list_runs(info.workdir) == []
    assert not marker.exists() and not (info.workdir / "gui").exists()


# --------------------------------------------------------------------------- real CLI subprocesses


def test_a_fresh_project_run_blocks_on_its_first_question(root: ProjectsRoot, monkeypatch: pytest.MonkeyPatch):
    secret = "sk-or-v1-GUI-TEST-SECRET-VALUE"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)  # inherited by the child; never on the command line or in the log
    info = root.create("demo", REQUEST)
    manager = RunManager()
    t0 = time.monotonic()
    handle = manager.start_project("run", info)
    assert manager.wait(handle, RUN_TIMEOUT) == 1
    elapsed = time.monotonic() - t0
    assert elapsed < 60, elapsed  # measured ~0.5 s: it stops at the first stage
    assert handle.id == "001-run" and handle.kind == "run" and handle.workdir == info.workdir
    assert handle.log_path == info.workdir / LOG_DIR / "001-run.log" == info.workdir / "gui" / "runs" / "001-run.log"
    lines = _log_lines(handle.log_path)
    assert lines[0].startswith(COMMAND_PREFIX)
    _assert_command_line(lines, [*CLI_PREFIX, "run", str(info.ir_path)])
    text = "\n".join(lines)
    assert "requirement_analysis     USER_INPUT_REQUIRED" in text and "BLOCKED - answer these to continue" in text
    assert "  [application] What is the intended application" in text and "  [jurisdiction] Which markets" in text
    assert lines[-1] == f"{EXIT_PREFIX}1" and secret not in text
    status = manager.status(info.workdir, "001-run")
    assert (status.id, status.kind, status.running, status.exit_code) == ("001-run", "run", False, 1)
    assert status.log_bytes == handle.log_path.stat().st_size and status.log_tail.splitlines() == lines and status.started == handle.started
    assert manager.active(info.workdir) is None
    # the questions are what the project shows afterwards, copied from pipeline.json
    last = root.get("demo").last_run
    assert last is not None and last.error is None and last.blocked and last.aborted is None
    assert (last.last_stage, last.last_status, last.stages_recorded) == ("requirement_analysis", "USER_INPUT_REQUIRED", 1)
    assert [q.key for q in last.open_questions] == ["application", "jurisdiction"] and all(q.required for q in last.open_questions)
    assert [q.key for q in last.optional_questions] == ["operating_temperature", "protection"]
    assert last.open_questions[0].rationale and last.open_questions[0].source == "system" and last.run_hash_label == RUN_CURRENT_IR

    # answered from the form: spaces and '=' in a value reach the IR verbatim, as one argv item
    handle = manager.start_project("run", info, {"application": "hobby board = demo", "jurisdiction": "EU"}, {"no_pdf": True})
    assert handle.id == "002-run" and manager.wait(handle, RUN_TIMEOUT) == 0
    lines = _log_lines(handle.log_path)
    _assert_command_line(lines, [
        *CLI_PREFIX, "run", str(info.ir_path), "--answer", "application=hobby board = demo", "--answer", "jurisdiction=EU", "--no-pdf",
    ])
    assert lines[-1] == f"{EXIT_PREFIX}0" and any(line.startswith("release ") for line in lines)
    assert CircuitIR.load(info.ir_path).requirements.get("application").text == "application: hobby board = demo"
    last = root.get("demo").last_run
    assert last is not None and not last.blocked and last.open_questions == [] and last.last_stage == "release"
    assert [(r.id, r.kind, r.running) for r in manager.list_runs(info.workdir)] == [("001-run", "run", False), ("002-run", "run", False)]
    # after a GUI restart the exit code is read back from the log's closing line
    again = RunManager().status(info.workdir, "001-run")
    assert (again.running, again.exit_code, again.started) == (False, 1, None) and again.log_tail.splitlines() == _log_lines(info.workdir / LOG_DIR / "001-run.log")


def test_review_and_stage_reports_kinds(root: ProjectsRoot):
    info = root.create("demo", REQUEST)
    manager = RunManager()
    handle = manager.start_project("review", info)
    assert handle.id == "001-review" and manager.wait(handle, RUN_TIMEOUT) == 0
    lines = _log_lines(handle.log_path)
    _assert_command_line(lines, [*CLI_PREFIX, "review", str(info.ir_path)])
    assert any(line.startswith("review.erc ") for line in lines) and any(line.startswith("review USER_INPUT_REQUIRED:") for line in lines)
    assert lines[-1] == f"{EXIT_PREFIX}0"
    handle = manager.start_project("stage-reports", info, None, {"no_pdf": True})
    assert handle.id == "002-stage-reports" and manager.wait(handle, RUN_TIMEOUT) == 0
    lines = _log_lines(handle.log_path)
    _assert_command_line(lines, [*CLI_PREFIX, "stage-reports", str(info.ir_path), "--no-pdf"])
    text = "\n".join(lines)
    # stderr is in the same log, and the Korean file names read back as UTF-8
    assert "no run record" in text and "01_이론_보고서.md" in text and "not requested (--no-pdf)" in text
    assert sorted(p.name for p in (info.workdir / "reports").glob("*.md")) == ["01_이론_보고서.md", "02_부품선정_보고서.md", "03_회로_보고서.md", "04_최종_보고서.md"]
    assert [r.id for r in manager.list_runs(info.workdir)] == ["001-review", "002-stage-reports"]
    assert manager.status(info.workdir, "002-stage-reports").kind == "stage-reports"


def test_one_active_run_per_project_is_a_conflict(root: ProjectsRoot):
    alpha = root.create("alpha", REQUEST)
    beta = root.create("beta", REQUEST)
    manager = RunManager(python_args=SLEEPER)
    first = manager.start_project("run", alpha)
    assert manager.active(alpha.workdir) == first
    with pytest.raises(RunConflictError, match="이미 실행 중"):
        manager.start_project("run", alpha)
    with pytest.raises(RunConflictError):
        manager.start_project("review", alpha)
    other = manager.start_project("run", beta)  # another project runs at the same time
    assert [r.id for r in manager.list_runs(alpha.workdir)] == ["001-run"] and manager.list_runs(alpha.workdir)[0].running
    # live: the log shows the child's first lines while it runs, stdout and stderr in the order written
    deadline = time.monotonic() + 30
    status = manager.status(alpha.workdir, first.id)
    while "err-1" not in status.log_tail.splitlines() and time.monotonic() < deadline:
        time.sleep(0.05)
        status = manager.status(alpha.workdir, first.id)
    assert status.running and status.exit_code is None and status.log_tail.splitlines()[1:] == ["out-1 한글", "err-1"]
    assert manager.wait(first, 30) == 0 and manager.wait(other, 30) == 0
    lines = _log_lines(first.log_path)
    _assert_command_line(lines, [*SLEEPER, "run", str(alpha.ir_path)])
    assert lines[1:] == ["out-1 한글", "err-1", "out-2", f"{EXIT_PREFIX}0"]
    assert manager.active(alpha.workdir) is None and not manager.status(alpha.workdir, first.id).running
    again = manager.start_project("run", alpha)  # the slot is free again
    assert again.id == "002-run"
    with pytest.raises(TimeoutError):
        manager.wait(again, 0.01)
    assert manager.wait(again, 30) == 0


@pytest.mark.skipif(os.name == "nt", reason="the cross-process slot is a POSIX flock; on Windows the slot is the GUI process's own")
def test_a_run_whose_gui_died_still_holds_its_project(root: ProjectsRoot):
    """A GUI killed mid-run leaves its child running: a new GUI process sees that run as running and refuses a second one until it ends."""
    info = root.create("demo", REQUEST)
    gui_a = (
        "import os, signal, sys\n"
        "from ai_eda.gui.projects import ProjectsRoot\n"
        "from ai_eda.gui.runs import RunManager\n"
        f"info = ProjectsRoot({str(root.root)!r}).get('demo')\n"
        "handle = RunManager([sys.executable, '-c', 'import time; print(\"orphan started\", flush=True); time.sleep(5)']).start_project('run', info)\n"
        "print(handle.id, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"  # the GUI dies without a word; its child is not signalled
    )
    result = subprocess.run([sys.executable, "-c", gui_a], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.stdout.strip() == "001-run", result.stderr[-2000:]
    manager = RunManager(python_args=SLEEPER)  # the next GUI process: it knows nothing of 001-run but its log
    status = manager.status(info.workdir, "001-run")
    assert status.running and status.exit_code is None, status
    assert manager.active_id(info.workdir) == "001-run" and manager.active(info.workdir) is None
    assert [(r.id, r.running) for r in manager.list_runs(info.workdir)] == [("001-run", True)]
    with pytest.raises(RunConflictError, match="001-run"):
        manager.start_project("run", info)
    assert [r.id for r in manager.list_runs(info.workdir)] == ["001-run"], "a refused start writes no log"
    # when the orphan ends, its log has no closing line and no holder: not running, no code, and the slot is free again
    deadline = time.monotonic() + 60
    while manager.status(info.workdir, "001-run").running:
        assert time.monotonic() < deadline, "the orphaned run never ended"
        time.sleep(0.1)
    status = manager.status(info.workdir, "001-run")
    assert (status.running, status.exit_code) == (False, None) and "orphan started" in status.log_tail
    assert manager.active_id(info.workdir) is None
    handle = manager.start_project("run", info)
    assert handle.id == "002-run" and manager.active_id(info.workdir) == "002-run" and manager.wait(handle, 30) == 0
    assert manager.active_id(info.workdir) is None


def test_start_places_a_run_from_ir_json_and_refuses_what_it_cannot_run(root: ProjectsRoot, tmp_path: Path):
    info = root.create("demo", REQUEST)
    manager = RunManager(python_args=[sys.executable, "-c", "pass"])
    for kind, args in (("delete", []), ("run", ["--no-pdf", "nul\x00"]), ("run", [5])):
        with pytest.raises(RunRequestError):
            manager.start(kind, info.ir_path, args)  # type: ignore[list-item]
    (root.root / "broken").mkdir()
    (root.root / "broken" / "ir.json").write_text("{", encoding="utf-8")
    with pytest.raises(RunRequestError, match="작업 폴더를 정할 수 없습니다"):
        manager.start("run", root.root / "broken" / "ir.json", [])
    assert not (info.workdir / "gui").exists() and not (root.root / "broken" / "gui").exists()
    handle = manager.start("run", info.ir_path, ["--no-pdf"])  # the workdir is read from ir.json, as `ai-eda run` reads it
    assert handle.workdir == info.workdir and manager.wait(handle, 30) == 0
    # a process that cannot be started: the reason closes its log and the slot stays free
    broken = RunManager(python_args=[str(tmp_path / "no-such-python")])
    with pytest.raises(RunStartError, match="실행을 시작할 수 없습니다"):
        broken.start_project("run", info)
    assert broken.active(info.workdir) is None
    assert _log_lines(info.workdir / LOG_DIR / "002-run.log")[-1].startswith("# could not start: FileNotFoundError")
    # a log folder that is a link is refused before anything is written
    other = root.create("other", REQUEST)
    _symlink_or_skip(other.workdir / "gui", tmp_path, directory=True)
    with pytest.raises(RunRequestError, match="심볼릭 링크"):
        manager.start_project("run", other)
    assert not (tmp_path / "runs").exists()


def test_status_knows_only_run_log_names(root: ProjectsRoot):
    info = root.create("demo", REQUEST)
    manager = RunManager()
    (info.workdir / "secret.log").write_text("no", encoding="utf-8")
    for run_id in ("001-run", "../secret", "../../ir", "001-delete", "abc", "01-run", "001-run.log", ""):
        with pytest.raises(RunNotFoundError):
            manager.status(info.workdir, run_id)
    assert manager.list_runs(info.workdir) == []


def test_the_child_environment_is_the_users_own_plus_utf8_and_unbuffered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_EDA_GUI_TEST_MARKER", "present")
    env = child_env()
    assert env["AI_EDA_GUI_TEST_MARKER"] == "present"
    assert {k: v for k, v in env.items() if k not in CHILD_ENV_ADDITIONS} == {k: v for k, v in os.environ.items() if k not in CHILD_ENV_ADDITIONS}
    windows = {"NoDefaultCurrentDirectoryInExePath": "1"} if os.name == "nt" else {}
    assert CHILD_ENV_ADDITIONS == {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **windows}
    assert all(env[k] == v for k, v in CHILD_ENV_ADDITIONS.items())


def test_a_run_never_imports_code_from_the_project_folder(root: ProjectsRoot):
    """The child runs with cwd = the workdir, but ``-P`` keeps it off sys.path: a planted ``ai_eda`` package or a
    ``csv.py`` shadowing the stdlib in the project folder never runs; the installed CLI does."""
    assert list(CLI_PYTHON_ARGS) == CLI_PREFIX and RunManager().python_args == CLI_PREFIX
    info = root.create("victim", REQUEST)
    marker = info.workdir / "PLANTED_CODE_RAN.txt"
    payload = f"import sys; open({str(marker)!r}, 'a', encoding='utf-8').write(repr(sys.argv) + '\\n'); print('PLANTED CODE RAN')\n"
    (info.workdir / "ai_eda").mkdir()
    (info.workdir / "ai_eda" / "__init__.py").write_text("", encoding="utf-8")
    (info.workdir / "ai_eda" / "cli.py").write_text(payload, encoding="utf-8")
    for shadow in ("csv.py", "argparse.py", "json.py", "pydantic.py"):
        (info.workdir / shadow).write_text(payload, encoding="utf-8")
    manager = RunManager()
    handle = manager.start_project("run", info, None, {"no_pdf": True})
    assert manager.wait(handle, RUN_TIMEOUT) == 1
    lines = _log_lines(handle.log_path)
    _assert_command_line(lines, [*CLI_PREFIX, "run", str(info.ir_path), "--no-pdf"])
    text = "\n".join(lines)
    assert not marker.exists() and "PLANTED CODE RAN" not in text, text[-2000:]
    # the real CLI ran: it stopped at the first questions, as for any fresh project
    assert "BLOCKED - answer these to continue" in text and "  [application] What is the intended application" in text
    assert lines[-1] == f"{EXIT_PREFIX}1"


# --------------------------------------------------------------------------- the LLM fields, served by the fake Claude Code CLI


def test_a_run_on_the_fake_claude_cli_logs_its_llm_usage(root: ProjectsRoot, fake: FakeClaudeCli, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(MARKER_ENV, "1")  # proves the child got the user's environment (the fake records presence only)
    info = root.create("llm", LLM_REQUEST)
    fake.queue(success_structured(EXTRACTION))
    manager = RunManager()
    handle = manager.start_project("run", info, None, {"llm": "claude", "llm_claude_cli": str(fake.exe), "no_pdf": True})
    assert manager.wait(handle, RUN_TIMEOUT) == 1
    lines = _log_lines(handle.log_path)
    _assert_command_line(lines, [*CLI_PREFIX, "run", str(info.ir_path), "--llm", "claude", "--llm-claude-cli", str(fake.exe), "--no-pdf"])
    usage = next(line for line in lines if line.startswith("LLM usage:"))
    assert usage.startswith("LLM usage: 1 served call(s)") and "budget none (not required: subscription only)" in usage and "(not charged)" in usage
    assert any(line.startswith(f"  LLM model pinned to this project: {CLAUDE_SPEC} ") for line in lines) and lines[-1] == f"{EXIT_PREFIX}1"
    (call,) = fake.prompt_calls()
    assert call["env"][MARKER_ENV] is True
    assert CircuitIR.load(info.ir_path).requirements.llm_model_spec == CLAUDE_SPEC
    last = root.get("llm").last_run
    assert last is not None and last.blocked and any(q.key == "confirm_requirements" for q in last.open_questions)


def test_a_pinned_project_run_with_another_model_exits_2_with_the_korean_sentence(root: ProjectsRoot, fake: FakeClaudeCli):
    info = root.create("pinned", LLM_REQUEST)
    ir = CircuitIR.load(info.ir_path)
    ir.requirements.llm_model_spec = CLAUDE_SPEC
    ir.save(info.ir_path)
    before = info.ir_path.read_bytes()
    manager = RunManager()
    options = {"llm": "claude", "llm_model": "claude:model-e", "llm_claude_cli": str(fake.exe)}
    handle = manager.start_project("run", info, None, options)
    assert manager.wait(handle, RUN_TIMEOUT) == 2
    lines = _log_lines(handle.log_path)
    assert model_pin_refusal(CLAUDE_SPEC, "claude:model-e") in lines and lines[-1] == f"{EXIT_PREFIX}2"
    assert manager.status(info.workdir, handle.id).exit_code == 2
    assert fake.prompt_calls() == [] and info.ir_path.read_bytes() == before
    # with the checkbox the same form runs (and would re-pin after a served call)
    fake.queue(success_structured(EXTRACTION))
    handle = manager.start_project("run", info, None, {**options, "llm_allow_model_change": True, "no_pdf": True})
    assert manager.wait(handle, RUN_TIMEOUT) == 1
    assert "--llm-allow-model-change" in _log_lines(handle.log_path)[0]
