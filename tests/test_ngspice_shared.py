"""NgspiceShared against the real ngspice.dll bundled with KiCad (ngspice-46 on this machine).

The DLL-backed tests skip only when no ngspice.dll can be found; on the
development machine (KiCad 10.0.6 installed) none of them skips. The pure
tests (command normalisation, deck validation, result helpers, DLL discovery)
always run.
"""

from __future__ import annotations

import hashlib
import math
import shutil
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
from ai_eda.tools.spice.ngspice_shared import find_ngspice_dll

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
    assert "stderr" not in res.log
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
    assert plot.command.startswith(ascii_res.engine_version + ",")
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
