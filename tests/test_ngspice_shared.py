"""NgspiceShared against the real ngspice.dll bundled with KiCad (ngspice-46 on this machine).

The DLL-backed tests skip only when no ngspice.dll can be found; on the
development machine (KiCad 10.0.6 installed) none of them skips. The pure
tests (command normalisation, deck validation, result helpers, DLL discovery)
always run.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

from ai_eda.errors import ToolExecutionError, ToolUnavailableError
from ai_eda.tools.spice import (
    NgspiceRunner,
    NgspiceShared,
    SpiceAnalysis,
    SpiceResult,
    normalise_command,
    rawfile,
    validate_deck,
)
from ai_eda.tools.spice.ngspice_shared import INFORMATIONAL_STDERR_RE, EngineDead, _Capture, _Engine, bind_reset, find_ngspice_dll
from tests.conftest import rawfile_command_ok

DLL_PRESENT = NgspiceShared().available()
needs_dll = pytest.mark.skipif(not DLL_PRESENT, reason="ngspice.dll (KiCad's bundled ngspice shared library) not found")

DIVIDER = ["divider", "V1 VIN 0 DC 12", "R1 VIN VOUT 10k", "R2 VOUT 0 10k", ".end"]
#: 1 ns edges (not 0) so ngspice sees a true step at t=0; period 2 s >> 5 ms simulated, so v(in) stays at 5 V
RC_TRAN = ["rc lowpass", "V1 IN 0 PULSE(0 5 0 1n 1n 1 2)", "R1 IN OUT 1k", "C1 OUT 0 1u", ".end"]
RC_AC = ["rc ac", "V1 IN 0 DC 0 AC 1", "R1 IN OUT 1k", "C1 OUT 0 1u", ".end"]
GAIN = ["xspice gain", "V1 IN 0 DC 1.5", "A1 IN OUT gainblk", ".model gainblk gain(in_offset=0 gain=2.0 out_offset=0)", "R1 OUT 0 1k", ".end"]
R_OHM, C_FARAD = 1e3, 1e-6


@pytest.fixture
def work(tmp_path: Path) -> Path:
    d = tmp_path / "경로 test"  # a space and Korean characters on purpose: paths go through `source '<path>'`
    d.mkdir()
    return d


@pytest.fixture
def runner() -> NgspiceShared:
    return NgspiceShared()


def write_deck(work: Path, name: str, lines: list[str]) -> Path:
    p = work / f"{name}.cir"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def sha256_of(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------------------- pure (no DLL)


def test_normalise_command_lowercases_and_validates():
    assert normalise_command(None, SpiceAnalysis.OP) == "op"
    assert normalise_command("  OP ", "op") == "op"
    assert normalise_command("dc VVIN 0 12 1", SpiceAnalysis.DC) == "dc vvin 0 12 1"  # ngspice matches device names case-sensitively
    assert normalise_command("TRAN 10U 5M", SpiceAnalysis.TRAN) == "tran 10u 5m"
    assert normalise_command("ac dec 10 1 1MEG", SpiceAnalysis.AC) == "ac dec 10 1 1meg"
    with pytest.raises(ValueError, match="needs an explicit command"):
        normalise_command(None, SpiceAnalysis.TRAN)
    with pytest.raises(ValueError, match="not allowed"):
        normalise_command("op; shell dir", SpiceAnalysis.OP)  # ';' would chain a second ngspice command
    with pytest.raises(ValueError, match="not allowed"):
        normalise_command("tran 10u 5m $foo", SpiceAnalysis.TRAN)
    with pytest.raises(ValueError, match="does not start with"):
        normalise_command("tran 10u 5m", SpiceAnalysis.OP)
    with pytest.raises(ValueError, match="does not start with"):
        normalise_command("run", SpiceAnalysis.OP)
    with pytest.raises(ValueError, match="takes no arguments"):
        normalise_command("op all", SpiceAnalysis.OP)
    with pytest.raises(ValueError, match="needs arguments"):
        normalise_command("dc", SpiceAnalysis.DC)


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["quote", "V1 n'1 0 DC 1", "R1 n'1 0 1k", ".end"], "quote/backtick"),
        (["no nodes", "R1 0 0 1k", ".end"], "no non-ground node"),
        (["control", "V1 A 0 DC 1", "R1 A 0 1k", ".control", "echo hi", ".endc", ".end"], ".control"),
        (["paren", "V1 Net-(R1-Pad2) 0 DC 1", "R1 Net-(R1-Pad2) 0 1k", ".end"], "characters ngspice mangles"),
        (["card", "V1 A 0 DC 1", "R1 A 0 1k", ".op", ".end"], "analysis card"),
        (["", "V1 A 0 DC 1", "R1 A 0 1k", ".end"], "title"),
        (["no end", "V1 A 0 DC 1", "R1 A 0 1k"], ".end must be the last"),
        (["missing value", "V1 IN 0 DC 1", "R1 IN OUT", "R2 OUT 0 1k", ".end"], "needs 2 nodes and a value"),
        (["korean", "V1 노드 0 DC 1", "R1 노드 0 1k", ".end"], "non-ASCII"),
        (["dup", "V1 A 0 DC 1", "R1 A 0 1k", "R1 A 0 2k", ".end"], "already used"),
        (["include", "V1 A 0 DC 1", "R1 A 0 1k", ".include models.lib", ".end"], ".include"),
        (["case", "V1 VOUT 0 DC 1", "R1 vout 0 1k", ".end"], "collide after ngspice lowercases"),
    ],
)
def test_validate_deck_rejects_hazardous_decks(lines: list[str], expected: str):
    problems, info = validate_deck("\n".join(lines) + "\n")
    assert problems, lines
    assert any(expected in p for p in problems), problems
    assert info["title"] == lines[0].strip()


def test_validate_deck_accepts_plain_decks():
    for lines in (DIVIDER, RC_TRAN, RC_AC, GAIN):
        problems, info = validate_deck("\n".join(lines) + "\n")
        assert problems == [], (lines, problems)
        assert info["title"] == lines[0]
    subckt = ["sub", ".subckt div a b", "R1 a b 1k", "R2 b 0 1k", ".ends", "V1 IN 0 DC 1", "X1 IN OUT div", "R1 OUT 0 1k", ".end"]
    assert validate_deck("\n".join(subckt) + "\n")[0] == []  # R1 inside the subckt does not collide with the top-level R1


def test_result_helpers_interpolate_on_the_scale():
    res = SpiceResult(
        engine="x", engine_version="x", netlist_path="n", netlist_hash="sha256:0", analysis=SpiceAnalysis.TRAN, command="tran 1 3",
        vectors={"time": [0.0, 1.0, 2.0, 3.0], "out": [0.0, 10.0, 20.0, 10.0]}, scale="time", n_points=4, succeeded=True,
    )
    assert res.value_at("out", 1.0) == 10.0  # exact hit
    assert res.value_at("out", 0.5) == 5.0
    assert res.value_at("out", 2.5) == 15.0
    assert res.value_at("v(OUT)", 2.5) == 15.0  # rawfile-style, case-insensitive lookup
    assert res.final("out") == 10.0 and res.max("out") == 20.0 and res.min("out") == 0.0
    with pytest.raises(ValueError, match="outside"):
        res.value_at("out", 3.5)
    with pytest.raises(KeyError):
        res.vector("nosuch")
    op = res.model_copy(update={"analysis": SpiceAnalysis.OP, "scale": None})
    with pytest.raises(ValueError, match="no sweep scale"):
        op.value_at("out", 1.0)
    # non-monotonic scale (nested dc sweeps are flattened): the first bracketing segment is used
    nested = res.model_copy(update={"scale": "v-sweep", "vectors": {"v-sweep": [0.0, 6.0, 12.0, 0.0, 6.0, 12.0], "vout": [0.0, 4.0, 8.0, 0.0, 3.0, 6.0]}})
    assert nested.value_at("vout", 3.0) == 2.0


def test_dll_discovery_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NGSPICE_DLL", str(tmp_path / "missing" / "ngspice.dll"))
    assert find_ngspice_dll() is None  # an explicit path that does not exist is not silently replaced
    absent = NgspiceShared(dll=tmp_path / "missing" / "ngspice.dll")
    assert absent is NgspiceShared(dll=tmp_path / "missing" / "ngspice.dll")
    assert not absent.available()
    with pytest.raises(ToolUnavailableError):
        absent.version()
    deck = tmp_path / "x.cir"
    deck.write_text("t\nR1 a 0 1k\n.end\n", encoding="utf-8")
    with pytest.raises(ToolUnavailableError):
        absent.run(deck, SpiceAnalysis.OP, tmp_path)


@pytest.mark.skipif(shutil.which("ngspice") is not None, reason="an ngspice binary is on PATH")
def test_binary_runner_is_optional_and_unavailable_here(tmp_path: Path):
    ng = NgspiceRunner()
    assert ng.engine == "ngspice"
    assert not ng.available()
    with pytest.raises(ToolUnavailableError):
        ng.version()
    with pytest.raises(ToolUnavailableError):
        ng.run(tmp_path / "x.cir", SpiceAnalysis.OP, tmp_path)
    assert NgspiceRunner.batch_deck("t\nR1 a 0 1k\n.end\n", "tran 10u 5m") == "t\nR1 a 0 1k\n.tran 10u 5m\n.end\n"
    assert NgspiceRunner.batch_deck("t\r\nR1 a 0 1k\r\n", "op") == "t\nR1 a 0 1k\n.op\n.end\n"


@pytest.mark.skipif(os.name == "nt", reason="a #! script stands in for the binary")
def test_the_batch_binary_reads_no_spiceinit(tmp_path: Path):
    """ngspice 45 and later read ``.spiceinit`` from the deck's folder (the output folder) before the working directory;
    ``-n`` (``--no-spiceinit``) keeps every one out, and the binary still runs in an empty folder."""
    log = tmp_path / "argv.log"
    fake = tmp_path / "bin" / "ngspice"
    fake.parent.mkdir()
    fake.write_text(f"#!/bin/sh\necho \"$PWD|$*\" >> '{log}'\necho 'ngspice-99 : fake'\n", encoding="utf-8")
    fake.chmod(0o755)
    out = tmp_path / "out"
    deck = write_deck(tmp_path, "divider", DIVIDER)
    res = NgspiceRunner(binary=str(fake)).run(deck, SpiceAnalysis.OP, out)
    assert not res.succeeded  # the fake writes no rawfile
    run_line = next(ln for ln in log.read_text(encoding="utf-8").splitlines() if "--version" not in ln)
    cwd, argv = run_line.split("|", 1)
    assert argv.split()[:3] == ["-n", "-b", "-r"] and argv.endswith(str(out.resolve() / "divider.op.cir"))
    assert Path(cwd).name.startswith("ai-eda-ngspice-") and not Path(cwd).exists()


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="no ngspice binary on PATH")
def test_the_real_batch_binary_skips_the_users_and_the_output_folders_spiceinit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without ``-n`` ngspice-42 sources ``~/.spiceinit`` (measured 2026-10-14: its ``echo`` line is in the log); 45 and
    later also the deck folder's. With ``-n`` neither is read and ``spinit`` still loads the code models."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".spiceinit").write_text("echo ai_eda_home_spiceinit_probe\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    out = tmp_path / "out"
    out.mkdir()
    (out / ".spiceinit").write_text("echo ai_eda_deckdir_spiceinit_probe\n", encoding="utf-8")
    res = NgspiceRunner().run(write_deck(tmp_path, "gain", GAIN), SpiceAnalysis.OP, out)
    assert res.succeeded, res.errors
    assert "ai_eda_home_spiceinit_probe" not in res.log and "ai_eda_deckdir_spiceinit_probe" not in res.log
    assert abs(res.vectors["out"][0] - 3.0) < 1e-9  # the XSPICE gain block: spinit's code models are loaded


def test_bind_reset_is_optional_for_builds_without_ngspice_reset():
    """KiCad's ngspice-46 exports ``ngSpice_Reset``; Debian/Ubuntu ``libngspice0`` (ngspice-42) does not - the runner still loads it."""
    from ctypes import c_int

    class Modern:
        class ngSpice_Reset:  # a ctypes function pointer stand-in: argtypes / restype are assigned
            pass

    class Old:
        def __getattr__(self, name: str):
            raise AttributeError(f"undefined symbol: {name}")

    assert bind_reset(Modern()) is True
    assert Modern.ngSpice_Reset.argtypes == [] and Modern.ngSpice_Reset.restype is c_int
    assert bind_reset(Old()) is False


def test_recover_without_ngspice_reset_marks_the_engine_dead_with_the_reason():
    eng = _Engine.__new__(_Engine)
    eng.dead, eng.dead_reason, eng.reset_supported, eng.dll_path = False, "", False, Path("/usr/lib/x86_64-linux-gnu/libngspice.so.0")
    with pytest.raises(EngineDead, match="does not export ngSpice_Reset"):
        eng.recover()
    assert eng.dead and "restart the process" in eng.dead_reason


POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="Windows does not delete a process's working directory")


def test_the_engine_starts_in_an_empty_folder_and_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ai_eda.tools.spice import ngspice_shared

    home = tmp_path / "project"
    home.mkdir()
    (home / ".spiceinit").write_text("shell touch planted\n", encoding="utf-8")
    monkeypatch.chdir(home)
    with ngspice_shared._empty_working_directory():
        inside = Path(os.getcwd())
        assert inside != home and inside.name.startswith("ai-eda-ngspice-init-") and os.listdir(inside) == []
    assert Path(os.getcwd()) == home and not inside.exists()
    with pytest.raises(RuntimeError), ngspice_shared._empty_working_directory():
        raise RuntimeError("back home on the way out too")
    assert Path(os.getcwd()) == home


@POSIX_ONLY
def test_a_deleted_working_directory_is_no_reason_to_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``os.getcwd()`` raises once the working directory is removed (a shell left in a deleted folder): that folder holds no
    ``.spiceinit``, so the block runs in the empty folder all the same and the process stays there - nowhere to return to."""
    from ai_eda.tools.spice import ngspice_shared

    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    with pytest.raises(OSError):
        os.getcwd()
    with ngspice_shared._empty_working_directory():
        inside = Path(os.getcwd())
        assert inside.name.startswith("ai-eda-ngspice-init-") and os.listdir(inside) == []
    assert not inside.exists()  # removed after the block, with the process still in it


def test_an_empty_folder_that_cannot_be_made_or_entered_is_tool_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import contextlib

    from ai_eda.tools.spice import ngspice_shared

    def no_folder(**kwargs):
        raise PermissionError(13, "Permission denied", str(tmp_path))

    monkeypatch.setattr(ngspice_shared.tempfile, "TemporaryDirectory", no_folder)
    with pytest.raises(ToolUnavailableError, match="cannot create an empty folder"):
        with ngspice_shared._empty_working_directory():
            pass
    not_a_folder = tmp_path / "file"
    not_a_folder.write_text("x", encoding="utf-8")
    monkeypatch.setattr(ngspice_shared.tempfile, "TemporaryDirectory", lambda **kwargs: contextlib.nullcontext(str(not_a_folder)))
    with pytest.raises(ToolUnavailableError, match="cannot enter the empty folder"):
        with ngspice_shared._empty_working_directory():
            pass
    with pytest.raises(ToolUnavailableError, match="cannot enter the empty folder"):  # at engine start: not a raw OSError
        _Engine(tmp_path / "libngspice.so.0", None)


def test_the_library_is_loaded_and_the_engine_started_in_the_empty_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The dynamic loader resolves an empty or relative ``LD_LIBRARY_PATH`` entry against the working directory when the
    library's dependencies load, and ``ngSpice_Init`` / the code models read ``./.spiceinit``: both run in the empty folder."""
    from unittest import mock

    from ai_eda.tools.spice import ngspice_shared

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    seen: dict[str, Path] = {}

    def fake_cdll(path: str):
        seen["load"] = Path(os.getcwd())
        assert os.listdir(".") == []
        return mock.MagicMock()

    def fake_init(self) -> None:
        seen["init"] = Path(os.getcwd())

    monkeypatch.setattr(ngspice_shared.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(_Engine, "_init_engine", fake_init)
    eng = _Engine(tmp_path / "libngspice.so.0", None)
    assert seen["load"] == seen["init"] != project and seen["load"].name.startswith("ai-eda-ngspice-init-")
    assert Path(os.getcwd()) == project
    eng.lib.ngSpice_Reset.return_value = 0
    eng.recover()  # ngSpice_Init again: in an empty folder again
    assert seen["init"] != project and seen["init"].name.startswith("ai-eda-ngspice-init-") and Path(os.getcwd()) == project

    def no_folder(**kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ngspice_shared.tempfile, "TemporaryDirectory", no_folder)
    with pytest.raises(EngineDead, match="cannot restart ngspice"):
        eng.recover()
    assert eng.dead


def test_informational_stderr_is_kept_in_the_transcript_but_is_not_an_error():
    cap = _Capture()
    cap.chars += ["stderr Using SPARSE 1.3 as Direct Linear Solver", "stderr Using KLU as Direct Linear Solver", "stderr Error: unknown subckt: x1", "stdout Circuit: t"]
    assert cap.stderr() == ["Error: unknown subckt: x1"]
    assert cap.stdout() == ["Circuit: t"]
    assert len(cap.chars) == 4  # the transcript (SpiceResult.log) still shows the banner
    assert not INFORMATIONAL_STDERR_RE.match("Warning: singular matrix")


# --------------------------------------------------------------------------------------- real DLL


@needs_dll
def test_singleton_and_engine_setup(runner: NgspiceShared):
    assert runner is NgspiceShared()
    assert runner.engine == "ngspice-shared"
    assert runner.version().startswith("ngspice-")
    assert runner.build()
    assert runner.codemodels_loaded, runner.engine_log()
    assert runner.self_test()["ok"]
    assert "sharedmode" in runner.engine_settings()
    log = runner.engine_log()
    assert any(runner.version() in line for line in log)
    assert any(line.startswith("self-test:") for line in log)


@needs_dll
def test_divider_op(runner: NgspiceShared, work: Path):
    deck = write_deck(work, "divider", DIVIDER)
    res = runner.run(deck, SpiceAnalysis.OP, work)
    assert res.succeeded, res.errors
    assert res.errors == []
    assert abs(res.vectors["vout"][0] - 6.0) < 1e-9
    assert res.vectors["vin"] == [12.0]
    assert abs(res.vectors["v1#branch"][0] + 0.0006) < 1e-12
    assert res.vector_types == {"vout": "voltage", "vin": "voltage", "v1#branch": "current"}
    assert res.engine == "ngspice-shared" and res.engine_version == runner.version()
    assert res.command == "op" and res.analysis == SpiceAnalysis.OP
    assert res.plot_name == "op1" and res.plot_type == "Operating Point"
    assert res.scale is None and res.n_points == 1
    assert res.netlist_path == str(deck.resolve()) and res.netlist_hash == sha256_of(deck)
    assert res.raw_output_path and Path(res.raw_output_path).parent == work.resolve()
    assert res.raw_output_hash == sha256_of(res.raw_output_path)
    assert "stdout Circuit: divider" in res.log.splitlines()
    # KiCad's build prints nothing on stderr for a clean run; Debian/Ubuntu libngspice0 (42) announces its linear solver there
    stderr_lines = [ln for ln in res.log.splitlines() if ln.startswith("stderr ")]
    assert all(INFORMATIONAL_STDERR_RE.match(ln[7:].strip()) for ln in stderr_lines), stderr_lines
    assert "3 : v1 vin 0 dc 12" in res.listing  # the deck as ngspice parsed it
    assert res.final("vout") == res.vectors["vout"][0] == res.final("v(VOUT)")
    assert res.final("i(v1)") == res.vectors["v1#branch"][0]
    with pytest.raises(ValueError):
        res.value_at("vout", 0.0)


@needs_dll
def test_dc_sweep_shape(runner: NgspiceShared, work: Path):
    deck = write_deck(work, "divider", DIVIDER)
    res = runner.run(deck, SpiceAnalysis.DC, work, "dc V1 0 12 1")  # uppercase device name: must be lowercased to run
    assert res.succeeded, res.errors
    assert res.command == "dc v1 0 12 1"
    assert res.plot_name == "dc1" and res.plot_type == "DC transfer characteristic"
    assert res.scale == "v-sweep" and res.n_points == 13
    assert res.vectors["v-sweep"] == [float(v) for v in range(13)]
    assert all(abs(v - s / 2) < 1e-9 for v, s in zip(res.vectors["vout"], res.vectors["v-sweep"]))
    assert abs(res.value_at("vout", 6.0) - 3.0) < 1e-9
    assert abs(res.value_at("vout", 5.5) - 2.75) < 1e-9
    assert abs(res.final("vout") - 6.0) < 1e-9 and abs(res.max("vout") - 6.0) < 1e-9 and res.min("vout") == 0.0
    with pytest.raises(ValueError, match="outside"):
        res.value_at("vout", 13.0)


@needs_dll
def test_rc_tran_value_at_one_ms(runner: NgspiceShared, work: Path):
    """v(out) of a 1k/1u RC step response at t = 1 ms = 5 V (1 - e^-1) = 3.1606 V.

    The step is PULSE(0 5 0 1n 1n 1 2): rise/fall 1 ns (ngspice needs a
    non-zero edge to see a step at t = 0; the 1 ns edge is negligible against
    the 1 ms time constant), pulse width 1 s and period 2 s so the input stays
    at 5 V for the whole 5 ms. The original "tran anomaly" (0.0014 V at 1 ms)
    was a Python-side aliasing bug: ngGet_Vec_Info returns one static struct,
    so the runner copies each vector before the next call.
    """
    deck = write_deck(work, "rc", RC_TRAN)
    res = runner.run(deck, SpiceAnalysis.TRAN, work, "tran 10u 5m")
    assert res.succeeded, res.errors
    assert res.plot_name == "tran1" and res.plot_type == "Transient Analysis" and res.scale == "time"
    time = res.vectors["time"]
    assert res.n_points == len(time) > 100
    assert time[0] == 0.0 and abs(time[-1] - 5e-3) < 1e-12
    assert all(b > a for a, b in zip(time, time[1:]))
    tau = R_OHM * C_FARAD
    expect_1ms = 5.0 * (1 - math.exp(-1e-3 / tau))
    assert abs(res.value_at("out", 1e-3) - expect_1ms) <= 0.02 * expect_1ms
    expect_end = 5.0 * (1 - math.exp(-5e-3 / tau))
    assert abs(res.final("out") - expect_end) <= 0.02 * expect_end
    assert res.max("out") <= 5.0 + 1e-9 and res.min("out") >= -1e-9
    assert res.vectors["in"][-1] == 5.0


@needs_dll
def test_rc_ac_lowpass_corner(runner: NgspiceShared, work: Path):
    deck = write_deck(work, "rc_ac", RC_AC)
    res = runner.run(deck, SpiceAnalysis.AC, work, "ac dec 10 1 1meg")
    assert res.succeeded, res.errors
    assert res.plot_name == "ac1" and res.plot_type == "AC Analysis" and res.scale == "frequency"
    assert res.n_points == 61
    for base in ("frequency", "in", "out", "v1#branch"):
        assert {base, f"{base}.phase_deg", f"{base}.real", f"{base}.imag"} <= set(res.vectors)
    # the scale's magnitude is the frequency (its imaginary part is 0)
    assert all(close(m, re) for m, re in zip(res.vectors["frequency"], res.vectors["frequency.real"]))
    assert res.vectors["frequency.imag"] == [0.0] * 61
    assert close(res.vectors["frequency"][0], 1.0) and abs(res.vectors["frequency"][-1] - 1e6) < 1e-6
    fc = 1 / (2 * math.pi * R_OHM * C_FARAD)
    assert abs(res.value_at("out", fc) - 1 / math.sqrt(2)) <= 0.01 / math.sqrt(2)
    assert abs(res.value_at("out.phase_deg", fc) + 45.0) <= 1.0
    assert abs(res.vectors["out"][0] - 1.0) < 1e-3  # 1 Hz: passband
    assert all(close(v, 1.0) for v in res.vectors["in"])


@needs_dll
def test_xspice_gain_block_proves_code_models(runner: NgspiceShared, work: Path):
    res = runner.run(write_deck(work, "gain", GAIN), SpiceAnalysis.OP, work)
    assert res.succeeded, res.errors
    assert res.vectors["out"] == [3.0]


@needs_dll
def test_failing_netlists_report_errors_and_the_engine_survives(runner: NgspiceShared, work: Path):
    singular = write_deck(work, "singular", ["singular", "V1 A 0 DC 1", "V2 A 0 DC 2", "R1 A 0 1k", ".end"])
    res = runner.run(singular, SpiceAnalysis.OP, work)
    assert not res.succeeded and res.vectors == {}
    assert any("singular matrix" in e for e in res.errors), res.errors
    assert "stderr Warning: singular matrix" in res.log

    bad_model = write_deck(work, "bad_model", ["bad model", "V1 IN 0 DC 1", "Q1 OUT IN 0 nosuchmodel", "R1 OUT 0 1k", ".end"])
    res = runner.run(bad_model, SpiceAnalysis.OP, work)
    assert not res.succeeded
    assert any("could not find a valid modelname" in e for e in res.errors), res.errors

    # singular matrix that ngspice "recovers" from with a bogus operating point: only stderr reveals it
    cap_node = write_deck(work, "cap_node", ["cap node", "V1 IN 0 DC 1", "C1 IN OUT 1u", "C2 OUT 0 1u", ".end"])
    res = runner.run(cap_node, SpiceAnalysis.OP, work)
    assert not res.succeeded and any("singular matrix" in e for e in res.errors)

    unknown_source = runner.run(write_deck(work, "divider", DIVIDER), SpiceAnalysis.DC, work, "dc v9 0 1 1")
    assert not unknown_source.succeeded and any("not in the circuit" in e for e in unknown_source.errors)

    # hazardous decks never reach the DLL (a zero-node deck would crash this very process)
    for name, lines in (
        ("nonodes", ["no nodes", "R1 0 0 1k", ".end"]),
        ("quote", ["quote", "V1 n'1 0 DC 1", "R1 n'1 0 1k", ".end"]),
        ("control", ["control", "V1 A 0 DC 1", "R1 A 0 1k", ".control", "shell echo hi", ".endc", ".end"]),
        ("card", ["card", *DIVIDER[1:-1], ".op", ".end"]),
    ):
        res = runner.run(write_deck(work, name, lines), SpiceAnalysis.OP, work)
        assert not res.succeeded and res.errors and res.errors[0].startswith("netlist rejected before reaching ngspice"), (name, res.errors)
        assert res.log == ""

    ok = runner.run(write_deck(work, "divider", DIVIDER), SpiceAnalysis.OP, work)
    assert ok.succeeded and abs(ok.vectors["vout"][0] - 6.0) < 1e-9
    assert runner is NgspiceShared()


@needs_dll
def test_ten_sequential_runs_give_identical_vectors(runner: NgspiceShared, work: Path):
    deck = write_deck(work, "rc", RC_TRAN)
    first = runner.run(deck, SpiceAnalysis.TRAN, work, "tran 10u 5m")
    assert first.succeeded, first.errors
    for _ in range(9):
        again = runner.run(deck, SpiceAnalysis.TRAN, work, "tran 10u 5m")
        assert again.succeeded, again.errors
        assert again.vectors == first.vectors
        assert again.n_points == first.n_points and again.plot_name == first.plot_name
        assert again.netlist_hash == first.netlist_hash


@needs_dll
def test_netlist_path_with_space_and_korean(runner: NgspiceShared, work: Path):
    assert " " in work.name and any(ord(ch) > 127 for ch in work.name)
    deck = write_deck(work, "분압 divider", DIVIDER)
    res = runner.run(deck, SpiceAnalysis.OP, work / "출력 raw")
    assert res.succeeded, res.errors
    assert "경로 test" in res.netlist_path and res.netlist_path.endswith("분압 divider.cir")
    assert Path(res.raw_output_path).parent.name == "출력 raw" and Path(res.raw_output_path).is_file()
    assert abs(res.vectors["vout"][0] - 6.0) < 1e-9


@needs_dll
def test_rawfile_round_trip_matches_the_in_memory_vectors(runner: NgspiceShared, work: Path):
    deck = write_deck(work, "rc", RC_TRAN)
    ascii_res = runner.run(deck, SpiceAnalysis.TRAN, work, "tran 10u 5m")
    assert ascii_res.succeeded, ascii_res.errors
    plot = rawfile.parse(ascii_res.raw_output_path)
    assert not plot.binary and plot.title == "rc lowpass" and plot.plotname == "Transient Analysis"
    assert rawfile_command_ok(plot.command, ascii_res.engine_version), plot.command
    assert plot.n_points == ascii_res.n_points and plot.scale == "time"
    back = plot.as_plot_vectors()
    assert set(back) == set(ascii_res.vectors)
    for name, values in back.items():  # ASCII rawfiles carry 15 significant digits
        assert all(close(a, b) for a, b in zip(values, ascii_res.vectors[name])), name
    assert sha256_of(ascii_res.raw_output_path) == ascii_res.raw_output_hash

    binary_res = runner.run(deck, SpiceAnalysis.TRAN, work, "tran 10u 5m", raw_format="binary")
    assert binary_res.succeeded, binary_res.errors
    bplot = rawfile.parse(binary_res.raw_output_path)
    assert bplot.binary and bplot.as_plot_vectors() == binary_res.vectors  # bit-exact
    assert binary_res.vectors == ascii_res.vectors

    ac_res = runner.run(write_deck(work, "rc_ac", RC_AC), SpiceAnalysis.AC, work, "ac dec 10 1 1meg")
    assert ac_res.succeeded, ac_res.errors
    aplot = rawfile.parse(ac_res.raw_output_path)
    assert aplot.is_complex and aplot.scale == "frequency"
    aback = aplot.as_plot_vectors()
    assert set(aback) == set(ac_res.vectors)
    for name, values in aback.items():
        assert all(close(a, b) for a, b in zip(values, ac_res.vectors[name])), name


@needs_dll
def test_timeout_halts_a_runaway_simulation(runner: NgspiceShared, work: Path):
    runaway = write_deck(work, "runaway", ["runaway", "V1 IN 0 PULSE(0 5 0 1p 1p 1n 2n)", "R1 IN OUT 1k", "C1 OUT 0 1u", ".end"])
    res = runner.run(runaway, SpiceAnalysis.TRAN, work, "tran 1 100", timeout_s=1.0)
    assert not res.succeeded and res.timed_out
    assert any(e.startswith("timeout:") for e in res.errors), res.errors
    assert res.vectors == {} and res.raw_output_path is None
    assert res.elapsed_s < 15.0
    ok = runner.run(write_deck(work, "divider", DIVIDER), SpiceAnalysis.OP, work)
    assert ok.succeeded and abs(ok.vectors["vout"][0] - 6.0) < 1e-9


@needs_dll
def test_caller_errors_raise_instead_of_failing_silently(runner: NgspiceShared, work: Path):
    with pytest.raises(ToolExecutionError):
        runner.run(work / "does_not_exist.cir", SpiceAnalysis.OP, work)
    deck = write_deck(work, "divider", DIVIDER)
    with pytest.raises(ValueError):
        runner.run(deck, SpiceAnalysis.OP, work, "op; shell dir")
    with pytest.raises(ValueError):
        runner.run(deck, SpiceAnalysis.TRAN, work)  # tran has no default command
    with pytest.raises(ValueError):
        runner.run(deck, SpiceAnalysis.OP, work, raw_format="csv")  # type: ignore[arg-type]


_PROBE = """
import contextlib, json, sys
from ai_eda.tools.spice import ngspice_shared
if sys.argv[1] == "unguarded":  # the control: what ngspice itself does with the working directory's .spiceinit
    ngspice_shared._empty_working_directory = contextlib.nullcontext
print(json.dumps(ngspice_shared.NgspiceShared().engine_settings()))
"""


_VERSION_PROBE = """
from ai_eda.tools.spice.ngspice_shared import NgspiceShared
print(NgspiceShared().version())
"""


def _needs_libgomp() -> bool:
    import subprocess

    dll = NgspiceShared().dll_path
    ldd = shutil.which("ldd")
    if dll is None or ldd is None or not sys.platform.startswith("linux"):
        return False
    out = subprocess.run([ldd, str(dll)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return "libgomp.so.1" in out.stdout


@needs_dll
@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("gcc") is None, reason="builds a shared object with gcc (Linux)")
def test_a_relative_ld_library_path_entry_cannot_lend_ngspice_a_dependency(tmp_path: Path):
    """Reported by the installer review: with ``LD_LIBRARY_PATH=:/opt/x/lib`` (an empty entry: the working directory) the
    loader took ``libgomp.so.1`` from the project folder while loading ngspice. The library is loaded in the empty folder.
    Measured 2026-10-14 (ngspice-42): before the fix the planted file was taken - "undefined symbol: GOMP_single_start" -
    and the engine was unusable; now the system's libgomp is."""
    import subprocess

    if not _needs_libgomp():
        pytest.skip("this ngspice library does not need libgomp.so.1")
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "planted-constructor-ran"
    source = tmp_path / "planted.c"
    source.write_text(
        '#include <stdio.h>\n__attribute__((constructor)) static void planted(void) { FILE *f = fopen("%s", "a"); if (f) { fputs("ran", f); fclose(f); } }\n' % marker,
        encoding="utf-8",
    )
    built = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-Wl,-soname,libgomp.so.1", "-o", str(project / "libgomp.so.1"), str(source)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if built.returncode != 0:
        pytest.skip(f"gcc could not build the planted library: {built.stderr.strip()[:200]}")
    repo = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")]),
        "LD_LIBRARY_PATH": os.pathsep + str(tmp_path / "opt" / "lib"),
    }
    out = subprocess.run(
        [sys.executable, "-c", _VERSION_PROBE], cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1].startswith("ngspice-") and not marker.exists()


_DELETED_CWD_PROBE = """
import os, sys
os.chdir(sys.argv[1])
os.rmdir(sys.argv[1])
from ai_eda.tools.spice.ngspice_shared import NgspiceShared
print(NgspiceShared().version())
"""


@needs_dll
@POSIX_ONLY
def test_the_engine_starts_from_a_deleted_working_directory(tmp_path: Path):
    """Reported by the installer review: ``ai-eda run`` started from a deleted folder died with a raw FileNotFoundError at
    the SPICE stage (``os.getcwd()``); the engine starts (its own process: the engine is a process-wide singleton)."""
    import subprocess
    import sys

    gone = tmp_path / "gone"
    gone.mkdir()
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run(
        [sys.executable, "-c", _DELETED_CWD_PROBE, str(gone)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1].startswith("ngspice-")


@needs_dll
def test_a_spiceinit_in_the_working_directory_is_never_read(tmp_path: Path):
    """``ngSpice_Init`` sources ``./.spiceinit`` (its ``shell`` lines run commands): the engine starts in an empty folder.

    Measured 2026-10-14 with ngspice-42: a ``.spiceinit`` holding ``shell touch <file>`` created the file. The probe
    here only ``set``s a variable, so the control (the unguarded init) runs nothing; each init is its own process
    because the engine is a process-wide singleton.
    """
    import json
    import subprocess
    import sys

    project = tmp_path / "project"
    project.mkdir()
    (project / ".spiceinit").write_text("set ai_eda_cwd_probe=planted\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")])}

    def settings(mode: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE, mode], cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
        )
        assert out.returncode == 0, out.stderr
        return "\n".join(json.loads(out.stdout.strip().splitlines()[-1]))

    assert "ai_eda_cwd_probe" in settings("unguarded")  # the file is read when nothing prevents it
    assert "ai_eda_cwd_probe" not in settings("guarded")
    assert Path.cwd() != project
