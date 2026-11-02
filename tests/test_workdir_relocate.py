"""A copied or moved project, ``ai-eda relocate``, the per-project lock and ``pipeline.json``'s ``ir_file`` (``ai_eda/workdir.py``).

A project writes only into the folder that holds its ir.json: a recorded
absolute workdir naming another folder is refused by ``run`` / ``review`` /
``report`` / ``stage-reports`` (both paths, "Nothing was run", the
``ai-eda relocate`` remedy) and neither folder changes by a byte;
``relocate`` re-records the folder and rebases only the locator paths under
the old one, drops the SPICE results reference and keeps the design hash;
``run`` / ``review`` / ``stage-reports`` / ``relocate`` hold a non-blocking
lock and read ir.json again once they hold it; the GUI refuses a run while a
command-line one holds the lock and its own child is never blocked;
``pipeline.json`` names the ir.json it describes and a report says when that
is another one; ``ai-eda new`` never replaces an ir.json. No KiCad, no
ngspice, no network, no model.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

import ai_eda.workdir as workdir_module
from ai_eda.cli import main as cli_main, new_project, project_workdir
from ai_eda.errors import IRSchemaError
from ai_eda.gui.preview import project_zip, zip_members
from ai_eda.gui.projects import ProjectsRoot
from ai_eda.gui.runs import LOG_DIR, RunConflictError, RunManager
from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitIR,
    Evidence,
    ProjectMeta,
    RegulatoryProvenance,
    RegulatoryRequirement,
    ValidationResult,
    ValidationStatus,
)
from ai_eda.ir.components import LibraryRef
from ai_eda.ir.provenance import SourceRef
from ai_eda.report import build_report_data, load_ir_file, load_pipeline_record
from ai_eda.report.data import (
    PIPELINE_DESCRIBES_IR,
    PIPELINE_OTHER_IR,
    PIPELINE_STALE_IR,
    RUN_CURRENT_IR,
)
from ai_eda.report.pipeline_log import PIPELINE_FILE, save_pipeline_record, sha256_of_file
from ai_eda.report.server import render_page
from ai_eda.workdir import (
    BUSY_PHRASE,
    LOCK_FILE,
    ProjectLock,
    WorkdirMismatchError,
    foreign_absolute,
    is_locked,
    relocate_project,
    workdir_mismatch,
)
from ai_eda.workflow import Orchestrator, PipelineState

REQUEST = "5 V 입력, 1 kHz 구형파 발진기"
MISMATCH_PHRASES = ("is not this ir.json's directory", "Nothing was run", "ai-eda relocate")
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="flock is POSIX; Windows uses msvcrt.locking (not measured) and is_locked answers False there")
RUN_TIMEOUT = 120.0


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _tree(folder: Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` of every file below ``folder`` (links by their target text)."""
    out: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in filenames:
            p = Path(dirpath) / name
            data = os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
            out[p.relative_to(folder).as_posix()] = hashlib.sha256(data).hexdigest()
    return out


def _new(folder: Path, name: str = "demo") -> Path:
    code, out, err = _cli("new", name, "--dir", str(folder), "--request", REQUEST)
    assert code == 0, err
    return folder.resolve() / "ir.json"


def _ran(folder: Path, name: str = "demo") -> Path:
    """A new project run once (it blocks on its first questions): pipeline.json, the lock file and the run's outputs exist."""
    ir_path = _new(folder, name)
    code, _out, err = _cli("run", str(ir_path))
    assert code in (0, 1), err
    assert (folder / PIPELINE_FILE).is_file() and (folder / LOCK_FILE).is_file()
    return ir_path


def _copy(src: Path, dst: Path) -> Path:
    """``src`` copied to ``dst`` without its lock file (so a refused copy that grows one would show)."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(LOCK_FILE))
    return dst.resolve() / "ir.json"


def _assert_mismatch(err: str, recorded: Path | str, here: Path) -> None:
    for phrase in (*MISMATCH_PHRASES, str(recorded), str(here)):
        assert phrase in err, (phrase, err)


# --------------------------------------------------------------------------- 1 / 2: copied and moved projects are refused


def test_a_copied_project_is_refused_by_run_review_report_and_stage_reports(tmp_path: Path):
    original = tmp_path / "orig"
    _ran(original)
    copy_ir = _copy(original, tmp_path / "copy")
    before = (_tree(original), _tree(tmp_path / "copy"))
    for argv in (("run", str(copy_ir)), ("review", str(copy_ir)), ("report", str(copy_ir)), ("stage-reports", str(copy_ir), "--no-pdf")):
        code, out, err = _cli(*argv)
        assert code == 2, (argv, out, err)
        _assert_mismatch(err, original.resolve(), copy_ir.parent)
    # nothing ran, nothing was read from or written into either folder - no lock file in the copy either
    assert (_tree(original), _tree(tmp_path / "copy")) == before
    assert not (tmp_path / "copy" / LOCK_FILE).exists()
    # serve's page is refused the same way (the handler turns it into a 503 line)
    with pytest.raises(WorkdirMismatchError, match="ai-eda relocate"):
        render_page(copy_ir)


def test_a_moved_project_is_refused_and_the_old_path_is_not_recreated(tmp_path: Path):
    old = tmp_path / "a" / "proj"
    _ran(old)
    (tmp_path / "b").mkdir()
    moved = tmp_path / "b" / "proj"
    os.rename(old, moved)
    before = _tree(moved)
    for argv in (("run", str(moved / "ir.json")), ("review", str(moved / "ir.json")), ("report", str(moved / "ir.json")),
                 ("stage-reports", str(moved / "ir.json"), "--no-pdf")):
        code, _out, err = _cli(*argv)
        assert code == 2, (argv, err)
        _assert_mismatch(err, old.resolve(), moved.resolve())
    assert not old.exists() and sorted(p.name for p in (tmp_path / "a").iterdir()) == []
    assert _tree(moved) == before


# --------------------------------------------------------------------------- 3 / 4: relocate


def _populated(old: Path) -> tuple[CircuitIR, dict[str, Path]]:
    """An IR recorded in ``old`` with single + multi-file artifacts, SPICE results with evidence, archived documents, and paths outside."""
    files = {
        "sch": old / "demo.kicad_sch", "job": old / "gerbers" / "demo.gbrjob", "gtl": old / "gerbers" / "demo-F_Cu.gtl",
        "gbl": old / "gerbers" / "demo-B_Cu.gbl", "netlist": old / "spice" / "demo.cir", "results": old / "spice" / "results.json",
        "raw": old / "spice" / "op.raw", "datasheet": old / "sources" / "ds.pdf", "lvd": old / "sources" / "lvd.html",
    }
    for key, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key} bytes\n".encode())
    outside = old.parent / "elsewhere" / "log.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("outside\n", encoding="utf-8")
    ir = CircuitIR(project=ProjectMeta(id="demo", name="demo", workdir=str(old)))
    ir.requirements.raw_input = REQUEST
    h = sha256_of_file
    ir.artifacts[ArtifactKind.SCHEMATIC] = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path=str(files["sch"]), content_hash=h(files["sch"]))
    ir.artifacts[ArtifactKind.GERBER] = ArtifactRef(kind=ArtifactKind.GERBER, path=str(files["job"]), files=[str(files["gtl"]), str(files["gbl"])])
    ir.artifacts[ArtifactKind.SPICE_NETLIST] = ArtifactRef(kind=ArtifactKind.SPICE_NETLIST, path=str(files["netlist"]), content_hash=h(files["netlist"]))
    ir.artifacts[ArtifactKind.SPICE_RESULT] = ArtifactRef(kind=ArtifactKind.SPICE_RESULT, path=str(files["results"]), content_hash=h(files["results"]))
    ir.validation.add(ValidationResult(
        check_id="spice", status=ValidationStatus.PASS, tool="ngspice", details={"raw_output_path": str(files["raw"])},
        evidence=[Evidence(description="results", path=str(files["results"]), content_hash=h(files["results"])),
                  Evidence(description="a log outside the project", path=str(outside))],
    ))
    ir.validation.add(ValidationResult(
        check_id="spice.v_out", status=ValidationStatus.PASS, tool="ngspice", message=f"read from {files['raw']}",
        evidence=[Evidence(description="rawfile", path=str(files["raw"]), content_hash=h(files["raw"])), Evidence(description="url only", url="https://example.org/x")],
    ))
    from tests.conftest import make_component

    part = make_component("R1", "10k")
    part.datasheet = SourceRef(title="R datasheet", document_path=str(files["datasheet"]), content_hash=h(files["datasheet"]))
    part.symbol = LibraryRef(library="Device", name="R", verified=True, library_path="/usr/share/kicad/symbols/Device.kicad_sym")
    ir.components.append(part)
    ir.regulatory.requirements.append(RegulatoryRequirement(
        id="reg.EU.LVD", jurisdiction="EU", title="LVD",
        provenance=RegulatoryProvenance(jurisdiction="EU", authority="EC", source_title="LVD", source_document=str(files["lvd"]), content_hash=h(files["lvd"])),
    ))
    return ir, files


def test_relocate_rebases_only_paths_under_the_old_workdir_and_keeps_the_design_hash(tmp_path: Path):
    old = (tmp_path / "old").resolve()
    ir, files = _populated(old)
    ir.save(old / "ir.json")
    design_hash = ir.content_hash()
    shutil.copytree(old, tmp_path / "new")
    new = (tmp_path / "new").resolve()
    (new / "gerbers" / "demo-B_Cu.gbl").unlink()  # a member missing at the new place: listed, not an error
    results_bytes = (new / "spice" / "results.json").read_bytes()
    old_tree = _tree(old)

    code, out, err = _cli("relocate", str(new / "ir.json"))
    assert code == 0, err
    assert f"project.workdir {old} -> {new}" in out and "rebased 9 path(s)" in out
    assert "artifact_paths 3, artifact_files 2, evidence 2, source_documents 1, regulatory_documents 1" in out
    assert "2 path(s) outside the old folder left as they are" in out
    assert "SPICE results reference dropped" in out and f"design hash unchanged ({design_hash[:23]})" in out
    assert "  not on disk at the new place: gerbers/demo-B_Cu.gbl" in out

    moved = CircuitIR.load(new / "ir.json")
    assert moved.content_hash() == design_hash and moved.project.workdir == str(new)
    assert moved.artifacts[ArtifactKind.SCHEMATIC].path == str(new / "demo.kicad_sch")
    assert moved.artifacts[ArtifactKind.GERBER].path == str(new / "gerbers" / "demo.gbrjob")
    assert moved.artifacts[ArtifactKind.GERBER].files == [str(new / "gerbers" / "demo-F_Cu.gtl"), str(new / "gerbers" / "demo-B_Cu.gbl")]
    assert moved.artifacts[ArtifactKind.SPICE_NETLIST].path == str(new / "spice" / "demo.cir")
    # the SPICE results reference is gone; the file itself is untouched (tool output is never rewritten)
    assert ArtifactKind.SPICE_RESULT not in moved.artifacts and (new / "spice" / "results.json").read_bytes() == results_bytes
    spice, v_out = moved.validation.latest("spice"), moved.validation.latest("spice.v_out")
    assert [e.path for e in spice.evidence] == [str(new / "spice" / "results.json"), str(old.parent / "elsewhere" / "log.txt")]
    assert [e.path for e in v_out.evidence] == [str(new / "spice" / "op.raw"), None]
    # a run log is not rewritten: details and messages keep the old paths
    assert spice.details == {"raw_output_path": str(files["raw"])} and str(files["raw"]) in v_out.message
    part = moved.component("R1")
    assert part.datasheet.document_path == str(new / "sources" / "ds.pdf")
    assert part.symbol.library_path == "/usr/share/kicad/symbols/Device.kicad_sym"  # outside the old folder: as written
    assert moved.regulatory.requirements[0].provenance.source_document == str(new / "sources" / "lvd.html")
    # the old folder is not touched
    assert _tree(old) == old_tree


def test_relocate_twice_is_a_no_op_and_dry_run_writes_nothing(tmp_path: Path):
    old = (tmp_path / "old").resolve()
    ir, _files = _populated(old)
    ir.save(old / "ir.json")
    shutil.copytree(old, tmp_path / "new")
    new_ir = (tmp_path / "new").resolve() / "ir.json"
    before = _tree(new_ir.parent)

    code, out, err = _cli("relocate", str(new_ir), "--dry-run")
    assert code == 0, err
    assert out.startswith(f"would relocate {new_ir}: project.workdir {old} -> {new_ir.parent}") and "would rebase 9 path(s)" in out
    assert "SPICE results reference would be dropped" in out and "nothing was written (--dry-run)" in out
    assert _tree(new_ir.parent) == before and not (new_ir.parent / LOCK_FILE).exists()  # not even a lock file
    assert relocate_project(new_ir, dry_run=True).changed and _tree(new_ir.parent) == before

    assert _cli("relocate", str(new_ir))[0] == 0
    once = new_ir.read_bytes()
    code, out, err = _cli("relocate", str(new_ir))
    assert code == 0 and out == f"nothing to relocate: project.workdir already is {new_ir.parent}\n", (out, err)
    assert new_ir.read_bytes() == once
    report = relocate_project(new_ir)
    assert not report.changed and report.rebased_total == 0 and new_ir.read_bytes() == once


def test_relocate_refuses_an_unreadable_ir_and_changes_nothing(tmp_path: Path):
    (tmp_path / "p").mkdir()
    bad = tmp_path / "p" / "ir.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _out, err = _cli("relocate", str(bad))
    assert code == 2 and str(bad) in err
    assert sorted(p.name for p in bad.parent.iterdir()) == ["ir.json"]  # no lock file for a refused IR
    code, _out, err = _cli("relocate", str(tmp_path / "missing" / "ir.json"))
    assert code == 2 and not (tmp_path / "missing").exists()


def test_relocate_would_never_change_the_design_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The guard: were a rebased field ever part of the design view, relocate refuses and writes nothing."""
    old = (tmp_path / "old").resolve()
    ir, _files = _populated(old)
    ir.save(old / "ir.json")
    shutil.copytree(old, tmp_path / "new")
    new_ir = (tmp_path / "new").resolve() / "ir.json"
    before = new_ir.read_bytes()
    real = CircuitIR.content_hash
    calls = {"n": 0}

    def drifting(self):
        calls["n"] += 1
        return real(self) + ("x" if calls["n"] > 1 else "")

    monkeypatch.setattr(CircuitIR, "content_hash", drifting)
    with pytest.raises(RuntimeError, match="relocate would change the design hash"):
        relocate_project(new_ir)
    assert new_ir.read_bytes() == before


# --------------------------------------------------------------------------- 5: a relocated copy runs into its own folder only


def test_a_relocated_copy_runs_into_its_own_folder_only(tmp_path: Path):
    original = tmp_path / "orig"
    orig_ir = _ran(original)
    copy_ir = _copy(original, tmp_path / "copy")
    orig_tree = _tree(original)
    assert _cli("relocate", str(copy_ir))[0] == 0
    code, _out, err = _cli("run", str(copy_ir))
    assert code in (0, 1), err
    assert _tree(original) == orig_tree
    assert json.loads((copy_ir.parent / PIPELINE_FILE).read_text(encoding="utf-8"))["ir_file"] == str(copy_ir)
    assert json.loads((original / PIPELINE_FILE).read_text(encoding="utf-8"))["ir_file"] == str(orig_ir)
    assert CircuitIR.load(copy_ir).project.workdir == str(copy_ir.parent)


# --------------------------------------------------------------------------- 6: the legacy rules


def test_legacy_relative_and_missing_workdir_rules_are_unchanged(tmp_path: Path):
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    ir = CircuitIR(project=ProjectMeta(id="demo", name="demo", workdir="projects/demo"))
    path = ir.save(project / "ir.json")
    here = project.resolve()
    for recorded in ("projects/demo", "demo", ".", None, str(project), str(here)):
        ir.project.workdir = recorded
        assert project_workdir(ir, path) == here and workdir_mismatch(ir, path) is None, recorded
    ir.project.workdir = "other/dir"
    assert workdir_mismatch(ir, path) is None  # a relative workdir is judged by project_workdir, not as a mismatch
    with pytest.raises(IRSchemaError, match=r"is relative and does not name this ir.json's directory.*ai-eda relocate"):
        project_workdir(ir, path)
    # relocate: nothing to do for None / a relative workdir naming the folder; a refused relative one is re-recorded, nothing rebased
    for recorded in (None, "projects/demo"):
        ir.project.workdir = recorded
        ir.save(path)
        code, out, _err = _cli("relocate", str(path))
        assert code == 0 and out.startswith("nothing to relocate"), out
        assert CircuitIR.load(path).project.workdir == recorded
    ir.project.workdir = "other/dir"
    ir.save(path)
    code, out, _err = _cli("relocate", str(path))
    assert code == 0 and "is relative and no artifact lies outside this folder; nothing was rebased" in out and "rebased 0 path(s) (none)" in out
    assert CircuitIR.load(path).project.workdir == str(here)
    assert project_workdir(CircuitIR.load(path), path) == here


def _with_artifacts(folder: Path, workdir: str | None, *, gerbers_only: bool = False) -> Path:
    """An ir.json in ``folder`` whose schematic / BOM (or only a gerber set) are registered by absolute path in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    ir = CircuitIR(project=ProjectMeta(id="demo", name="demo", workdir=workdir))
    if gerbers_only:
        (folder / "gerbers").mkdir()
        (folder / "gerbers" / "demo-F_Cu.gbr").write_text("G04*\n", encoding="utf-8")
        ir.artifacts[ArtifactKind.GERBER] = ArtifactRef(kind=ArtifactKind.GERBER, path=str(folder / "gerbers"), files=[str(folder / "gerbers" / "demo-F_Cu.gbr")],
                                                        content_hash="sha256:" + "0" * 64, generated_from_ir_hash="sha256:" + "1" * 64)
    else:
        for kind, name in ((ArtifactKind.SCHEMATIC, "demo.kicad_sch"), (ArtifactKind.BOM, "bom.csv")):
            (folder / name).write_text(name, encoding="utf-8")
            ir.artifacts[kind] = ArtifactRef(kind=kind, path=str(folder / name), content_hash="sha256:" + "0" * 64, generated_from_ir_hash="sha256:" + "1" * 64)
    return ir.save(folder / "ir.json")


def _artifact_paths(ir_path: Path) -> list[str]:
    ir = CircuitIR.load(ir_path)
    return [p for a in ir.artifacts.values() for p in (a.path, *a.files)]


def test_a_copy_with_an_unrecorded_or_relative_workdir_is_refused_and_relocated_from_its_artifacts(tmp_path: Path):
    """Regression: a copy whose workdir was None, or relative, kept its artifact paths in the original folder; relocate exited 0
    ('nothing to relocate' / 'rebased 0 path(s)') and the next review read the original's files."""
    # workdir None: refused (naming the folder the artifacts are in), relocated from the root-level artifacts' folder
    orig = tmp_path / "s5" / "orig"
    _with_artifacts(orig, None)
    copy = _copy(orig, tmp_path / "s5" / "copy")
    with pytest.raises(WorkdirMismatchError, match=r"its registered artifacts lie in .*orig, not in this ir.json's directory") as e:
        project_workdir(CircuitIR.load(copy), copy)
    assert e.value.stray and all(str(orig) in p for p in e.value.stray) and workdir_mismatch(CircuitIR.load(copy), copy) == (str(orig), copy.parent)
    code, _out, err = _cli("review", str(copy))
    assert code == 2 and "ai-eda relocate" in err and "Nothing was run" in err
    code, out, err = _cli("relocate", str(copy))
    assert code == 0 and f"the artifacts were compiled into {orig}" in out and "rebased 2 path(s) (artifact_paths 2)" in out, (out, err)
    assert all(Path(p).parent == copy.parent for p in _artifact_paths(copy)) and CircuitIR.load(copy).project.workdir == str(copy.parent)
    assert project_workdir(CircuitIR.load(copy), copy) == copy.parent
    # a relative workdir the copy does not end with: refused as relative; relocate finds the ancestor that ends with it
    orig = tmp_path / "s8" / "projects" / "demo"
    _with_artifacts(orig, "projects/demo")
    copy = _copy(orig, tmp_path / "s8" / "projects" / "demo2")
    with pytest.raises(IRSchemaError, match="is relative"):
        project_workdir(CircuitIR.load(copy), copy)
    code, out, _err = _cli("relocate", str(copy))
    assert code == 0 and "rebased 2 path(s)" in out and f"the artifacts were compiled into {orig}" in out, out  # formerly 'rebased 0 path(s)'
    assert all(Path(p).parent == copy.parent for p in _artifact_paths(copy))
    # the whole tree copied: the relative workdir still matches the copy's tail, but the artifacts are the original's
    r1 = tmp_path / "r1" / "projects" / "demo"
    _with_artifacts(r1, "projects/demo")
    shutil.copytree(tmp_path / "r1", tmp_path / "r2")
    copy = (tmp_path / "r2" / "projects" / "demo" / "ir.json").resolve()
    with pytest.raises(WorkdirMismatchError, match="its registered artifacts lie in"):
        project_workdir(CircuitIR.load(copy), copy)  # formerly accepted: its review read r1's files
    code, out, _err = _cli("relocate", str(copy))
    assert code == 0 and "rebased 2 path(s)" in out and CircuitIR.load(copy).project.workdir == str(copy.parent), out
    assert all(Path(p).parent == copy.parent for p in _artifact_paths(copy))


def test_relocate_refuses_what_it_cannot_rebase_and_writes_nothing(tmp_path: Path):
    # only a gerber set (in a subfolder): the old folder cannot be told from the IR
    orig = tmp_path / "orig"
    _with_artifacts(orig, None, gerbers_only=True)
    copy = _copy(orig, tmp_path / "copy")
    before = copy.read_bytes()
    code, _out, err = _cli("relocate", str(copy))
    assert code == 2 and "--from <old folder>" in err and "Nothing was written" in err and copy.read_bytes() == before
    code, out, err = _cli("relocate", str(copy), "--from", "relative/dir")
    assert code == 2 and "is not an absolute path" in err and copy.read_bytes() == before
    code, out, err = _cli("relocate", str(copy), "--from", str(orig))
    assert code == 0, err
    assert _artifact_paths(copy) == [str(copy.parent / "gerbers"), str(copy.parent / "gerbers" / "demo-F_Cu.gbr")]
    # artifacts from two different folders: rebasing from one leaves the other outside - refused, nothing written
    a, b = tmp_path / "a", tmp_path / "b"
    ir_path = _with_artifacts(a, str(a))
    ir = CircuitIR.load(ir_path)
    (b).mkdir()
    (b / "cpl.csv").write_text("cpl", encoding="utf-8")
    ir.artifacts[ArtifactKind.CPL] = ArtifactRef(kind=ArtifactKind.CPL, path=str(b / "cpl.csv"), content_hash="sha256:" + "0" * 64, generated_from_ir_hash="sha256:" + "1" * 64)
    ir.save(ir_path)
    copy = _copy(a, tmp_path / "c")
    before = copy.read_bytes()
    code, _out, err = _cli("relocate", str(copy))
    assert code == 2 and "would still lie outside this folder" in err and str(b / "cpl.csv") in err and copy.read_bytes() == before


# --------------------------------------------------------------------------- 7 / 8: the lock


@POSIX_ONLY
def test_a_second_command_is_refused_while_the_project_lock_is_held(tmp_path: Path):
    ir_path = _ran(tmp_path / "p")
    folder = ir_path.parent
    before = ir_path.read_bytes()
    with ProjectLock(folder) as held:
        assert held.held and is_locked(folder)
        for argv in (("run", str(ir_path)), ("review", str(ir_path)), ("stage-reports", str(ir_path), "--no-pdf"), ("relocate", str(ir_path))):
            code, _out, err = _cli(*argv)
            assert code == 2 and BUSY_PHRASE in err and LOCK_FILE in err, (argv, err)
        code, _out, err = _cli("report", str(ir_path))  # a read-only view takes no lock
        assert code == 0, err
        assert _cli("relocate", str(ir_path), "--dry-run")[0] == 0  # nor does a dry run
    assert ir_path.read_bytes() == before and not is_locked(folder)
    assert _cli("review", str(ir_path))[0] != 2


@POSIX_ONLY
def test_the_lock_is_released_after_a_run_and_after_an_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir_path = _ran(tmp_path / "p")
    folder = ir_path.parent
    assert not is_locked(folder)
    with ProjectLock(folder):
        pass

    def boom(self, ir, **kwargs):
        raise RuntimeError("a defect in a stage")

    monkeypatch.setattr(Orchestrator, "run", boom)
    with pytest.raises(RuntimeError, match="a defect in a stage"):
        cli_main(["run", str(ir_path)])
    assert not is_locked(folder) and (folder / LOCK_FILE).is_file()  # released, never deleted
    with ProjectLock(folder) as again:
        assert again.held


@POSIX_ONLY
def test_a_linked_lock_file_is_refused_and_an_unlockable_folder_proceeds_with_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir_path = _new(tmp_path / "p")
    folder = ir_path.parent
    target = tmp_path / "elsewhere.lock"
    (folder / LOCK_FILE).symlink_to(target)
    code, _out, err = _cli("review", str(ir_path))
    assert code == 2 and "is a symbolic link" in err and not target.exists()
    (folder / LOCK_FILE).unlink()
    assert not is_locked(folder)

    def no_flock(fd, op):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(workdir_module.fcntl, "flock", no_flock)
    code, _out, err = _cli("review", str(ir_path))
    assert code != 2 and "could not lock" in err and "concurrent ai-eda runs of this project are not prevented" in err


@POSIX_ONLY
def test_a_lock_file_this_user_cannot_write_is_still_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A read-only folder or another user's lock file: the lock is taken through a read-only descriptor (a lock needs no write)."""
    ir_path = _new(tmp_path / "p")
    folder = ir_path.parent
    (folder / LOCK_FILE).touch()
    real_open = os.open

    def no_write(path, flags, *args, **kwargs):
        if str(path).endswith(LOCK_FILE) and flags & os.O_RDWR:
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(workdir_module.os, "open", no_write)
    with ProjectLock(folder) as lock:
        assert lock.held and lock.note is None and is_locked(folder)
        code, _out, err = _cli("review", str(ir_path))
        assert code == 2 and BUSY_PHRASE in err
    assert not is_locked(folder)


# --------------------------------------------------------------------------- 9: the GUI and the CLI lock


@POSIX_ONLY
def test_the_gui_refuses_while_the_cli_holds_the_lock_and_its_own_child_is_not_blocked(tmp_path: Path):
    root = ProjectsRoot(tmp_path / "projects")
    info = root.create("demo", REQUEST)
    manager = RunManager()  # the real CLI as the child: python -P -m ai_eda.cli
    with ProjectLock(info.workdir):
        with pytest.raises(RunConflictError, match=r"\.ai-eda\.lock"):
            manager.start_project("review", info)
        assert not (info.workdir / LOG_DIR).exists()  # refused before a log was made
    handle = manager.start_project("review", info)
    code = manager.wait(handle, timeout=RUN_TIMEOUT)
    log = handle.log_path.read_text(encoding="utf-8")
    assert code != 2 and BUSY_PHRASE not in log, log
    assert not is_locked(info.workdir)


# --------------------------------------------------------------------------- 10 / 14: pipeline.json's ir_file


def test_pipeline_record_names_its_ir_file_and_the_report_says_when_it_describes_another(tmp_path: Path):
    root = tmp_path / "projects"
    orig_ir = _ran(root / "alpha", "alpha")
    record = load_pipeline_record(orig_ir.parent).record
    assert record.ir_file == str(orig_ir) and record.ir_path == str(orig_ir)
    ir, sha = load_ir_file(orig_ir)
    own = build_report_data(ir, orig_ir, orig_ir.parent, ir_sha=sha)
    assert own.meta.pipeline_note == PIPELINE_DESCRIBES_IR and own.stages.run_hash_label == RUN_CURRENT_IR

    copy_ir = _copy(root / "alpha", root / "beta")
    assert _cli("relocate", str(copy_ir))[0] == 0
    ir, sha = load_ir_file(copy_ir)
    assert ir.content_hash() == record.ir_hash  # a copy has the same design hash: the hash label alone would say "current"
    data = build_report_data(ir, copy_ir, copy_ir.parent, ir_sha=sha)
    assert data.meta.pipeline_note == f"{PIPELINE_OTHER_IR}: {orig_ir}"
    assert data.stages.run_hash_label == PIPELINE_OTHER_IR
    assert data.release.freshness == PIPELINE_OTHER_IR and not data.release.current
    code, _out, err = _cli("report", str(copy_ir))
    assert code == 0 and PIPELINE_OTHER_IR in (copy_ir.parent / "report.html").read_text(encoding="utf-8")
    code, _out, err = _cli("stage-reports", str(copy_ir), "--no-pdf")
    assert code == 0 and f"{PIPELINE_FILE} describes another ir.json ({orig_ir})" in err
    for md in (copy_ir.parent / "reports").glob("*.md"):
        assert str(orig_ir.parent) not in md.read_text(encoding="utf-8"), md  # the stage reports carry no path
    listed = {p.name: p for p in ProjectsRoot(root).list()}
    beta, alpha = listed["beta"].last_run, listed["alpha"].last_run
    assert beta.other_ir == str(orig_ir) and beta.run_hash_label == PIPELINE_OTHER_IR
    assert alpha.other_ir is None and alpha.run_hash_label == RUN_CURRENT_IR


def test_an_old_pipeline_record_without_ir_file_still_loads(tmp_path: Path):
    ir = CircuitIR(project=ProjectMeta(id="demo", name="demo", workdir=str(tmp_path)))
    ir_path = ir.save(tmp_path / "ir.json")
    save_pipeline_record(PipelineState(), ir, ir_path, tmp_path, results_before=0, ir_file_sha256=sha256_of_file(ir_path))
    raw = json.loads((tmp_path / PIPELINE_FILE).read_text(encoding="utf-8"))
    del raw["ir_file"]  # the record as the code before ir_file wrote it
    (tmp_path / PIPELINE_FILE).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    record = load_pipeline_record(tmp_path).record
    assert record.ir_file is None
    loaded, sha = load_ir_file(ir_path)
    data = build_report_data(loaded, ir_path, tmp_path, ir_sha=sha)
    assert data.meta.pipeline_note == PIPELINE_DESCRIBES_IR and data.stages.run_hash_label == RUN_CURRENT_IR
    ir.requirements.raw_input = "changed"  # the sha rules still work without ir_file
    ir.save(ir_path)
    loaded, sha = load_ir_file(ir_path)
    assert build_report_data(loaded, ir_path, tmp_path, ir_sha=sha).meta.pipeline_note == PIPELINE_STALE_IR
    summary = ProjectsRoot(tmp_path.parent).info(tmp_path).last_run
    assert summary.other_ir is None and summary.run_hash_label.startswith("recorded for an earlier IR version")


# --------------------------------------------------------------------------- 11 / 12: new, and the lock file is no artifact


def test_new_refuses_an_existing_ir_json(tmp_path: Path):
    ir_path = _new(tmp_path / "p")
    ir = CircuitIR.load(ir_path)
    ir.requirements.raw_input = "a design worth keeping"
    ir.save(ir_path)
    before = ir_path.read_bytes()
    code, out, err = _cli("new", "other", "--dir", str(tmp_path / "p"), "--request", "x")
    assert code == 2 and out == "" and f"refusing to create {ir_path}: an ir.json already exists there" in err
    assert ir_path.read_bytes() == before
    with pytest.raises(FileExistsError):
        new_project("other", "x", tmp_path / "p")
    # an existing folder without ir.json is fine
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("mine", encoding="utf-8")
    assert _cli("new", "e", "--dir", str(tmp_path / "empty"))[0] == 0
    assert (tmp_path / "empty" / "notes.txt").read_text(encoding="utf-8") == "mine"
    # a dangling link named ir.json is an existing entry too
    (tmp_path / "linked").mkdir()
    try:
        (tmp_path / "linked" / "ir.json").symlink_to(tmp_path / "nowhere.json")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"cannot create a symbolic link here: {e}")
    code, _out, err = _cli("new", "l", "--dir", str(tmp_path / "linked"))
    assert code == 2 and "an ir.json already exists" in err and not (tmp_path / "nowhere.json").exists()


def test_new_with_a_dir_that_is_a_file_names_the_file_not_an_ir_json(tmp_path: Path):
    """Regression: `new --dir <a regular file>` said 'an ir.json already exists there'; `--dir <file>/sub` ended in a traceback."""
    afile = tmp_path / "afile"
    afile.write_text("not a folder", encoding="utf-8")
    for target in (afile, afile / "sub"):
        code, out, err = _cli("new", "x", "--dir", str(target))
        assert code == 2 and out == "" and "is not a folder" in err and "an ir.json already exists" not in err, (target, err)
        assert "Traceback" not in err
    assert afile.read_text(encoding="utf-8") == "not a folder" and sorted(p.name for p in tmp_path.iterdir()) == ["afile"]
    # the real refusal keeps its reason and its type (the GUI's `except FileExistsError` still catches it)
    from ai_eda.cli import ProjectExistsError

    ir_path = _new(tmp_path / "p")
    with pytest.raises(ProjectExistsError) as e:
        new_project("again", "x", ir_path.parent)
    assert isinstance(e.value, FileExistsError) and e.value.filename == str(ir_path)


def test_the_lock_file_is_neither_an_artifact_nor_in_the_project_zip(tmp_path: Path):
    ir_path = _ran(tmp_path / "p")
    folder = ir_path.parent
    ir = CircuitIR.load(ir_path)
    paths = [p for art in ir.artifacts.values() for p in (art.path, *art.files)]
    assert not any(Path(p).name == LOCK_FILE for p in paths)
    assert all(parts[-1] != LOCK_FILE for parts in zip_members(ir, folder))
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(project_zip(ir, ir_path.read_bytes(), folder, "p"))).namelist()
    assert names and not any(n.endswith(LOCK_FILE) for n in names)
    # nor in the design hash: the same IR hashes the same with or without the file
    h = ir.content_hash()
    (folder / LOCK_FILE).unlink()
    assert CircuitIR.load(ir_path).content_hash() == h


# --------------------------------------------------------------------------- 13: the IR is read again under the lock


def test_run_reads_the_ir_again_once_the_lock_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ir_path = _new(tmp_path / "p")
    real_acquire = ProjectLock.acquire
    rewritten = {"done": False}

    def acquire_then_rewrite(self):
        real_acquire(self)
        if not rewritten["done"]:  # another command saved ir.json while this one waited for the lock
            rewritten["done"] = True
            ir = CircuitIR.load(ir_path)
            ir.requirements.raw_input = "rewritten while the lock was taken"
            ir.save(ir_path)

    monkeypatch.setattr(ProjectLock, "acquire", acquire_then_rewrite)
    code, _out, err = _cli("run", str(ir_path))
    assert code in (0, 1), err
    assert rewritten["done"] and CircuitIR.load(ir_path).requirements.raw_input == "rewritten while the lock was taken"

    # relocate the same way: its saved IR keeps the field rewritten under the lock
    copy_ir = _copy(ir_path.parent, tmp_path / "copy")
    ir_path = copy_ir
    rewritten["done"] = False
    code, _out, err = _cli("relocate", str(copy_ir))
    assert code == 0, err
    saved = CircuitIR.load(copy_ir)
    assert rewritten["done"] and saved.requirements.raw_input == "rewritten while the lock was taken"
    assert saved.project.workdir == str(copy_ir.parent)


# --------------------------------------------------------------------------- 15: a workdir recorded on the other operating system


def test_foreign_absolute_paths_are_recognised():
    if os.name == "nt":
        assert foreign_absolute("/proj") and foreign_absolute("\\proj") and not foreign_absolute("C:\\proj")
    else:
        assert foreign_absolute("C:\\proj") and foreign_absolute("c:/proj/x") and foreign_absolute("\\\\server\\share\\x")
        assert not foreign_absolute("/proj") and not foreign_absolute("proj\\x") and not foreign_absolute("C:proj")


@pytest.mark.skipif(os.name == "nt", reason="the Windows-path case is the Linux side; the mirrored /proj case runs on Windows")
def test_a_workdir_recorded_on_another_os_is_refused_and_relocated(tmp_path: Path):
    folder = (tmp_path / "proj").resolve()
    folder.mkdir()
    ir = CircuitIR(project=ProjectMeta(id="w", name="w", workdir="C:\\Users\\me\\projects\\w"))
    ir.artifacts[ArtifactKind.SCHEMATIC] = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path="C:\\Users\\me\\projects\\w\\w.kicad_sch")
    ir.artifacts[ArtifactKind.SPICE_RESULT] = ArtifactRef(kind=ArtifactKind.SPICE_RESULT, path="C:\\Users\\me\\projects\\w\\spice\\results.json")
    ir.validation.add(ValidationResult(
        check_id="spice.v", status=ValidationStatus.PASS, tool="ngspice",
        evidence=[Evidence(description="raw", path="c:\\users\\ME\\Projects\\W\\spice\\op.raw"),  # Windows paths compare case-insensitively
                  Evidence(description="elsewhere", path="D:\\other\\x.pdf")],
    ))
    ir_path = ir.save(folder / "ir.json")
    design_hash = ir.content_hash()
    code, _out, err = _cli("run", str(ir_path))
    assert code == 2 and "ai-eda relocate" in err and "C:\\Users\\me\\projects\\w" in err and "Nothing was run" in err
    assert sorted(p.name for p in folder.iterdir()) == ["ir.json"]
    code, out, err = _cli("relocate", str(ir_path))
    assert code == 0 and "rebased 2 path(s) (artifact_paths 1, evidence 1)" in out and "1 path(s) outside the old folder" in out, (out, err)
    moved = CircuitIR.load(ir_path)
    assert moved.project.workdir == str(folder) and moved.content_hash() == design_hash
    assert moved.artifacts[ArtifactKind.SCHEMATIC].path == str(folder / "w.kicad_sch")
    assert ArtifactKind.SPICE_RESULT not in moved.artifacts
    assert [e.path for e in moved.validation.latest("spice.v").evidence] == [str(folder / "spice" / "op.raw"), "D:\\other\\x.pdf"]
    assert project_workdir(moved, ir_path) == folder


@pytest.mark.skipif(os.name != "nt", reason="the mirrored case: a POSIX workdir read on Windows (not run on Linux)")
def test_a_posix_workdir_read_on_windows_is_refused_and_relocated(tmp_path: Path):  # pragma: no cover - Windows only
    folder = (tmp_path / "proj").resolve()
    folder.mkdir()
    ir = CircuitIR(project=ProjectMeta(id="w", name="w", workdir="/home/me/projects/w"))
    ir.artifacts[ArtifactKind.SCHEMATIC] = ArtifactRef(kind=ArtifactKind.SCHEMATIC, path="/home/me/projects/w/w.kicad_sch")
    ir_path = ir.save(folder / "ir.json")
    code, _out, err = _cli("run", str(ir_path))
    assert code == 2 and "ai-eda relocate" in err
    assert _cli("relocate", str(ir_path))[0] == 0
    assert CircuitIR.load(ir_path).artifacts[ArtifactKind.SCHEMATIC].path == str(folder / "w.kicad_sch")
