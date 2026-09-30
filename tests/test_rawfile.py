"""ngspice rawfile parser: hand-built ASCII/binary samples (always) and files the real ngspice.dll writes
for op / dc / tran / ac in both formats (skipped only when the DLL is absent)."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis, rawfile, result_from_rawfile
from ai_eda.tools.spice.rawfile import PLOTNAMES, RawPlot, canonical_name, complex_convention, parse, parse_all, raw_variable_name
from tests.conftest import rawfile_command_ok

DLL_PRESENT = NgspiceShared().available()
needs_dll = pytest.mark.skipif(not DLL_PRESENT, reason="ngspice.dll (KiCad's bundled ngspice shared library) not found")

ASCII_TRAN = (
    "Title: rc probe\r\n"
    "Date: Mon Sep 21 16:05:41  2026\r\n"
    "Command: ngspice-46, Build Apr 14 2026   05:15:29\r\n"
    "Plotname: Transient Analysis\r\n"
    "Flags: real\r\n"
    "No. Variables: 3\r\n"
    "No. Points: 2\r\n"
    "Variables:\r\n"
    "\t0\ttime\ttime\r\n"
    "\t1\tv(out)\tvoltage\r\n"
    "\t2\ti(v1)\tcurrent\r\n"
    "Values:\r\n"
    " 0\t0.000000000000000e+00\r\n"
    "\t1.500000000000000e+00\r\n"
    "\t-2.500000000000000e-03\r\n"
    "\r\n"
    " 1\t1.000000000000000e-05\r\n"
    "\t2.000000000000000e+00\r\n"
    "\t-3.000000000000000e-03\r\n"
    "\r\n"
    "\r\n"
)

ASCII_AC = (
    "Title: rc ac\n"
    "Date: Mon Sep 21 16:05:41  2026\n"
    "Command: ngspice-46, Build Apr 14 2026   05:15:29\n"
    "Plotname: AC Analysis\n"
    "Flags: complex\n"
    "No. Variables: 2\n"
    "No. Points: 2\n"
    "Variables:\n"
    "\t0\tfrequency\tfrequency grid=3\n"
    "\t1\tv(out)\tvoltage\n"
    "Values:\n"
    " 0\t1.000000000000000e+00,0.000000000000000e+00\n"
    "\t3.000000000000000e+00,4.000000000000000e+00\n"
    "\n"
    " 1\t1.000000000000000e+01,0.000000000000000e+00\n"
    "\t0.000000000000000e+00,-1.000000000000000e+00\n"
    "\n"
)


def binary_rawfile(plotname: str, flags: str, variables: list[tuple[str, str]], rows: list[list[float]]) -> bytes:
    header = (
        f"Title: made up\nDate: today\nCommand: ngspice-46, Build x\nPlotname: {plotname}\nFlags: {flags}\n"
        f"No. Variables: {len(variables)}\nNo. Points: {len(rows)}\nVariables:\n"
        + "".join(f"\t{i}\t{n}\t{t}\n" for i, (n, t) in enumerate(variables))
        + "Binary:\n"
    ).encode()
    body = b"".join(struct.pack("<d", v) for row in rows for v in row)
    return header + body


def test_parse_ascii_real_plot_with_crlf(tmp_path: Path):
    p = tmp_path / "tran.raw"
    p.write_bytes(ASCII_TRAN.encode())
    plot = parse(p)
    assert isinstance(plot, RawPlot)
    assert plot.title == "rc probe" and plot.date.startswith("Mon Sep 21") and plot.command.startswith("ngspice-46,")
    assert plot.plotname == "Transient Analysis" and plot.flags == "real" and not plot.is_complex
    assert plot.n_points == 2 and plot.n_variables == 3 and not plot.binary
    assert plot.variables == [("time", "time"), ("v(out)", "voltage"), ("i(v1)", "current")]
    assert plot.scale == "time"
    assert plot.vectors == {"time": [0.0, 1e-5], "v(out)": [1.5, 2.0], "i(v1)": [-2.5e-3, -3e-3]}
    assert plot.as_plot_vectors() == {"time": [0.0, 1e-5], "out": [1.5, 2.0], "v1#branch": [-2.5e-3, -3e-3]}
    assert plot.header["No. Points"] == "2"


def test_parse_ascii_complex_plot_uses_the_complex_convention(tmp_path: Path):
    p = tmp_path / "ac.raw"
    p.write_text(ASCII_AC, encoding="utf-8", newline="\n")
    plot = parse(p)
    assert plot.is_complex and plot.scale == "frequency"
    assert plot.variables == [("frequency", "frequency"), ("v(out)", "voltage")]  # 'grid=3' extra dropped from the type
    assert plot.vectors["frequency"] == [1.0, 10.0]
    assert plot.vectors["frequency.imag"] == [0.0, 0.0]
    assert plot.vectors["v(out)"] == [5.0, 1.0]
    assert plot.vectors["v(out).real"] == [3.0, 0.0] and plot.vectors["v(out).imag"] == [4.0, -1.0]
    assert abs(plot.vectors["v(out).phase_deg"][0] - math.degrees(math.atan2(4.0, 3.0))) < 1e-12
    assert abs(plot.vectors["v(out).phase_deg"][1] + 90.0) < 1e-12
    back = plot.as_plot_vectors()
    assert set(back) == {"frequency", "frequency.phase_deg", "frequency.real", "frequency.imag", "out", "out.phase_deg", "out.real", "out.imag"}


def test_parse_binary_plot_is_exact(tmp_path: Path):
    rows = [[0.0, 12.0, 6.000000000000001], [1e-5, 12.0, 6.000000000000001]]
    p = tmp_path / "dc.raw"
    p.write_bytes(binary_rawfile("DC transfer characteristic", "real", [("v(v-sweep)", "voltage"), ("v(vin)", "voltage"), ("v(vout)", "voltage")], rows))
    plot = parse(p)
    assert plot.binary and plot.scale == "v(v-sweep)"
    assert plot.vectors["v(vout)"] == [6.000000000000001, 6.000000000000001]
    assert plot.as_plot_vectors() == {"v-sweep": [0.0, 1e-5], "vin": [12.0, 12.0], "vout": [6.000000000000001, 6.000000000000001]}


def test_parse_binary_complex_plot(tmp_path: Path):
    rows = [[1.0, 0.0, 3.0, 4.0], [10.0, 0.0, 0.0, -1.0]]  # (re, im) pairs per variable
    p = tmp_path / "ac.raw"
    p.write_bytes(binary_rawfile("AC Analysis", "complex", [("frequency", "frequency"), ("v(out)", "voltage")], rows))
    plot = parse(p)
    assert plot.is_complex and plot.vectors["v(out)"] == [5.0, 1.0] and plot.vectors["frequency"] == [1.0, 10.0]


def test_operating_point_has_no_scale(tmp_path: Path):
    p = tmp_path / "op.raw"
    p.write_bytes(binary_rawfile("Operating Point", "real", [("v(vin)", "voltage"), ("i(v1)", "current"), ("v(vout)", "voltage")], [[12.0, -0.0006, 6.0]]))
    plot = parse(p)
    assert plot.scale is None and plot.n_points == 1
    assert plot.as_plot_vectors() == {"vin": [12.0], "v1#branch": [-0.0006], "vout": [6.0]}


def test_several_plots_in_one_file(tmp_path: Path):
    p = tmp_path / "two.raw"
    p.write_bytes(ASCII_TRAN.encode() + ASCII_AC.encode())
    plots = parse_all(p)
    assert [pl.plotname for pl in plots] == ["Transient Analysis", "AC Analysis"]
    assert parse(p).plotname == "Transient Analysis"
    mixed = tmp_path / "mixed.raw"
    mixed.write_bytes(binary_rawfile("Operating Point", "real", [("v(a)", "voltage")], [[1.0]]) + ASCII_TRAN.encode())
    assert [pl.binary for pl in parse_all(mixed)] == [True, False]


@pytest.mark.parametrize(
    "content, match",
    [
        (b"", "empty file"),
        (b"Title: x\nPlotname: y\n", "without a 'Binary:' or 'Values:'"),
        (ASCII_TRAN.replace("No. Variables: 3", "No. Variables: 2").encode(), "variable lines"),
        (ASCII_TRAN.replace(" 1\t1.000000000000000e-05", " 7\t1.000000000000000e-05").encode(), "expected point index 1"),
        (ASCII_TRAN.replace("No. Points: 2", "No. Points: 3").encode(), "ends after 2 of 3 points"),
        (ASCII_TRAN.replace("2.000000000000000e+00", "two").encode(), "bad value"),
        (b"Title: x\nNo. Variables: x\nNo. Points: 1\nVariables:\n\t0\ta\tvoltage\nValues:\n 0\t1\n", "missing/invalid"),
        (binary_rawfile("Operating Point", "real", [("v(a)", "voltage")], [[1.0]])[:-3], "binary body has 5 bytes, expected 8"),
        (b"Title: x\nNo. Variables: 1\nNo. Points: 1\nVariables:\n\t0\ta\nValues:\n 0\t1\n", "malformed variable line"),
        (b"Title: x\nNo. Variables: 1\nNo. Points: 1\nVariables:\n\t0\ta\tvoltage\ngarbage without colon\nValues:\n 0\t1\n", "unexpected header line"),
    ],
)
def test_structural_problems_are_errors_not_guesses(tmp_path: Path, content: bytes, match: str):
    p = tmp_path / "bad.raw"
    p.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        parse(p)


def test_name_mapping_between_plot_vectors_and_rawfile_variables():
    assert canonical_name("v(vout)") == "vout"
    assert canonical_name("v(v-sweep)") == "v-sweep"
    assert canonical_name("i(i-sweep)") == "i-sweep"
    assert canonical_name("i(v1)") == "v1#branch"
    assert canonical_name("i(a1)") == "a1#branch"
    for unchanged in ("time", "frequency", "res-sweep", "temp-sweep"):
        assert canonical_name(unchanged) == unchanged
    assert raw_variable_name("vout", 3) == "v(vout)"
    assert raw_variable_name("v-sweep", 3) == "v(v-sweep)"
    assert raw_variable_name("i-sweep", 4) == "i(i-sweep)"
    assert raw_variable_name("v1#branch", 4) == "i(v1)"
    assert raw_variable_name("a1#branch_1_0", 4) == "i(a1)"
    assert raw_variable_name("time", 1) == "time" and raw_variable_name("frequency", 2) == "frequency"
    assert raw_variable_name("res-sweep", 15) == "res-sweep" and raw_variable_name("temp-sweep", 14) == "temp-sweep"
    expanded = complex_convention("x", [(3.0, 4.0)])
    assert expanded["x"] == [5.0] and expanded["x.real"] == [3.0] and expanded["x.imag"] == [4.0]
    assert abs(expanded["x.phase_deg"][0] - math.degrees(math.atan2(4.0, 3.0))) < 1e-12


# --------------------------------------------------------------------------------------- files the real DLL writes

DECKS = {
    SpiceAnalysis.OP: (["divider", "V1 VIN 0 DC 12", "R1 VIN VOUT 10k", "R2 VOUT 0 10k", ".end"], None),
    SpiceAnalysis.DC: (["divider dc", "V1 VIN 0 DC 12", "R1 VIN VOUT 10k", "R2 VOUT 0 10k", ".end"], "dc v1 0 12 1"),
    SpiceAnalysis.TRAN: (["rc tran", "V1 IN 0 PULSE(0 5 0 1n 1n 1 2)", "R1 IN OUT 1k", "C1 OUT 0 1u", ".end"], "tran 10u 5m"),
    SpiceAnalysis.AC: (["rc ac", "V1 IN 0 DC 0 AC 1", "R1 IN OUT 1k", "C1 OUT 0 1u", ".end"], "ac dec 10 1 1meg"),
}
EXPECTED_SCALE = {SpiceAnalysis.OP: None, SpiceAnalysis.DC: "v(v-sweep)", SpiceAnalysis.TRAN: "time", SpiceAnalysis.AC: "frequency"}


def close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


@needs_dll
@pytest.mark.parametrize("raw_format", ["ascii", "binary"])
@pytest.mark.parametrize("analysis", list(DECKS))
def test_dll_written_rawfiles_round_trip(tmp_path: Path, analysis: SpiceAnalysis, raw_format: str):
    lines, command = DECKS[analysis]
    deck = tmp_path / f"{analysis.value}.cir"
    deck.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    runner = NgspiceShared()
    res = runner.run(deck, analysis, tmp_path, command, raw_format=raw_format)  # type: ignore[arg-type]
    assert res.succeeded, res.errors
    plot = parse(res.raw_output_path)
    assert plot.binary == (raw_format == "binary")
    assert plot.plotname == PLOTNAMES[analysis.value] == res.plot_type
    assert plot.title == lines[0]
    assert rawfile_command_ok(plot.command, runner.version()), plot.command
    assert plot.date
    assert plot.n_points == res.n_points and plot.n_variables == len(res.vector_types)
    assert plot.scale == EXPECTED_SCALE[analysis]
    assert plot.is_complex == (analysis == SpiceAnalysis.AC)
    assert plot.variables[0][0] == (plot.scale or plot.variables[0][0])
    for name, typ in plot.variables:
        assert res.vector_types[canonical_name(name)] == typ.split()[0]
    back = plot.as_plot_vectors()
    assert set(back) == set(res.vectors)
    if plot.binary:
        assert back == res.vectors
    else:
        for name, values in back.items():
            assert all(close(a, b) for a, b in zip(values, res.vectors[name])), name


@needs_dll
def test_result_from_rawfile_builds_the_batch_runner_result(tmp_path: Path):
    """The batch ``NgspiceRunner`` reads its result from a rawfile; prove that path on a file the DLL wrote."""
    lines, command = DECKS[SpiceAnalysis.TRAN]
    deck = tmp_path / "rc.cir"
    deck.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    shared = NgspiceShared().run(deck, SpiceAnalysis.TRAN, tmp_path, command, raw_format="binary")
    assert shared.succeeded, shared.errors
    res = result_from_rawfile(
        Path(shared.raw_output_path), engine="ngspice", engine_version="ngspice-46", netlist_path=deck,
        netlist_hash=shared.netlist_hash, analysis=SpiceAnalysis.TRAN, command=command,
    )
    assert res.succeeded and res.errors == []
    assert res.engine == "ngspice" and res.plot_type == "Transient Analysis" and res.plot_name is None
    assert res.scale == "time" and res.n_points == shared.n_points
    assert res.vectors == shared.vectors and res.vector_types == shared.vector_types
    assert res.raw_output_path == shared.raw_output_path and res.raw_output_hash == shared.raw_output_hash
    assert abs(res.value_at("out", 1e-3) - shared.value_at("out", 1e-3)) < 1e-12
    wrong = result_from_rawfile(
        Path(shared.raw_output_path), engine="ngspice", engine_version="ngspice-46", netlist_path=deck,
        netlist_hash=shared.netlist_hash, analysis=SpiceAnalysis.OP, command="op",
    )
    assert not wrong.succeeded and any("expected 'Operating Point'" in e for e in wrong.errors)
