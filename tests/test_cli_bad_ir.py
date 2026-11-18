"""``ai-eda run`` / ``review`` on an ir.json that cannot be loaded: one stderr line and exit 2, never a traceback.

A usage error (missing file, not JSON, an unknown key the IR models would drop, another schema version) is not
a design verdict, so it must not look like one: exit 1 means "blocked on a question or the last stage FAILed".
``report`` already behaved this way; this pins ``run`` and ``review`` to it.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ai_eda.cli import main as cli_main


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    code, out, _ = _cli("new", "demo", "--dir", str(tmp_path / "demo"), "--request", "a 12 V to 5 V divider for a bench")
    assert code == 0 and "created" in out
    return tmp_path / "demo" / "ir.json"


@pytest.mark.parametrize("command", ["run", "review"])
def test_an_unknown_key_is_refused_with_one_line_and_exit_2(project: Path, command: str):
    data = json.loads(project.read_text(encoding="utf-8"))
    data["bogus_key"] = 1
    project.write_text(json.dumps(data), encoding="utf-8")
    before = project.read_bytes()
    code, out, err = _cli(command, str(project))
    assert code == 2 and out == ""
    assert "unknown key(s) the IR models would drop: bogus_key" in err and "Traceback" not in err and err.count("\n") == 1
    assert project.read_bytes() == before  # refused before anything is run, saved or locked
    assert not (project.parent / ".ai-eda.lock").exists() and not (project.parent / "pipeline.json").exists()


@pytest.mark.parametrize("command", ["run", "review"])
def test_a_missing_or_non_json_file_is_exit_2(tmp_path: Path, command: str):
    code, out, err = _cli(command, str(tmp_path / "missing.json"))
    assert code == 2 and out == "" and "missing.json" in err and "Traceback" not in err
    broken = tmp_path / "ir.json"
    broken.write_text("{", encoding="utf-8")
    code, out, err = _cli(command, str(broken))
    assert code == 2 and out == "" and "ir.json" in err and "Traceback" not in err


@pytest.mark.parametrize("command", ["run", "review"])
def test_a_file_that_breaks_between_the_first_load_and_the_locked_reload_is_exit_2(project: Path, command: str, monkeypatch):
    # run / review load ir.json once to find the workdir, take the lock, then load again: a file that
    # became unreadable in between is refused the same way
    import ai_eda.cli as cli

    real_load, calls = cli._load, []

    def flaky(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("vanished between load and lock")
        return real_load(path)

    monkeypatch.setattr(cli, "_load", flaky)
    code, out, err = _cli(command, str(project))
    assert code == 2 and out == "" and "vanished between load and lock" in err and "Traceback" not in err
    assert len(calls) == 2  # refused at the locked reload, nothing ran after it
