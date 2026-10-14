"""The RF calculators' matching networks and low-pass prototypes, checked by ngspice (skipped without it).

Each deck is built from the calculator outputs (spelled by
``format_spice_number``) and run through :class:`~ai_eda.tools.spice.NgspiceShared`
with an ``ac lin`` sweep around the frequency of interest, a 1 V AC source
behind a 50 ohm source resistance:

* a match is right when the network presents 50 ohm at f: ``v(in)`` = 0.5 at
  0 degrees (the matched divider) and, the network being lossless, the load
  voltage is ``0.5 sqrt(R_load / 50)`` (0.25 V into 12.5 ohm - the value the
  RF foundation's ``rf_gap/lmatch.cir`` measured on ngspice-42);
* a doubly terminated lossless prototype's insertion loss
  ``-20 log10(|v(out)| / 0.5)`` equals the calculator's analytic attenuation
  (Butterworth at 1.8 GHz for f_c 1.05 GHz: 23.43 dB; Chebyshev 0.5 dB, n = 5,
  at 2 f_c: 42.04 dB) within 0.05 dB.

Measured on ngspice-42 (Ubuntu libngspice0, 2026-09-28) when this test was
written: every match within 1e-15 V of 0.5 V and 1e-13 degrees of 0, both
insertion losses within 1e-13 dB of the analytic value (the elements are exact,
the network lossless); the tolerances above are far looser on purpose.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai_eda.tools.calc import format_spice_number
from ai_eda.tools.calc import rf
from ai_eda.tools.spice import NgspiceShared, SpiceAnalysis

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")

F0 = 900e6


def _n(x: float) -> str:
    return format_spice_number(x)


def _run(tmp_path: Path, name: str, lines: list[str], f_lo: float, f_hi: float):
    deck = tmp_path / f"{name}.cir"
    deck.write_text("\n".join([name, "V1 src 0 DC 0 AC 1", "RS src in 50", *lines, ".end"]) + "\n", encoding="utf-8", newline="\n")
    res = runner.run(deck, SpiceAnalysis.AC, tmp_path / name, f"ac lin 3 {_n(f_lo)} {_n(f_hi)}")
    assert res.succeeded, res.errors
    return res


def _matched(res, load_ohm: float, node: str) -> None:
    assert res.value_at("in", F0) == pytest.approx(0.5, abs=1e-6)
    assert res.value_at("in.phase_deg", F0) == pytest.approx(0.0, abs=1e-4)
    assert res.value_at(node, F0) == pytest.approx(0.5 * math.sqrt(load_ohm / 50.0), abs=1e-6)


@needs_ngspice
def test_both_l_matches_present_50_ohm_at_their_frequency(tmp_path: Path) -> None:
    lp = rf.lmatch(F0, 50.0, 12.5, "lowpass")
    res = _run(tmp_path, "lmatch_lp", [f"L1 in mid {_n(lp.series)}", f"C1 in 0 {_n(lp.shunt)}", "RL mid 0 12.5"], F0 - 1e6, F0 + 1e6)
    _matched(res, 12.5, "mid")
    assert res.value_at("mid", F0) == pytest.approx(0.25, abs=1e-6)  # rf_gap/lmatch.cir
    hp = rf.lmatch(F0, 50.0, 12.5, "highpass")
    res = _run(tmp_path, "lmatch_hp", [f"C1 in mid {_n(hp.series)}", f"L1 in 0 {_n(hp.shunt)}", "RL mid 0 12.5"], F0 - 1e6, F0 + 1e6)
    _matched(res, 12.5, "mid")
    # off frequency the match degrades: not a flat 50 ohm
    off = _run(tmp_path, "lmatch_off", [f"L1 in mid {_n(lp.series)}", f"C1 in 0 {_n(lp.shunt)}", "RL mid 0 12.5"], 1.5 * F0, 1.5 * F0 + 2e6)
    assert abs(off.vectors["in.phase_deg"][0]) > 10.0  # measured -39.7 degrees at 1.5 f0 (|v(in)| 0.508 V)


@needs_ngspice
def test_the_pi_match_presents_50_ohm_at_its_frequency(tmp_path: Path) -> None:
    pm = rf.pi_match(F0, 50.0, 12.5, 5.0)
    lines = [f"C1 in 0 {_n(pm.c_source)}", f"L1 in out {_n(pm.l_series)}", f"C2 out 0 {_n(pm.c_load)}", "RL out 0 12.5"]
    _matched(_run(tmp_path, "pimatch", lines, F0 - 1e6, F0 + 1e6), 12.5, "out")


def _ladder(g: list[float], f_c: float) -> list[str]:
    """A shunt-C-first ladder of the prototype g_1..g_n between the 50 ohm source (node in) and a 50 ohm load (node out)."""
    n = len(g) - 1
    nodes = ["in", *[f"n{i}" for i in range(1, n // 2)], "out"]  # one node after each series inductor
    lines, node = [], 0
    for k in range(1, n + 1):
        if k % 2 == 1:  # shunt capacitor at the current node
            lines.append(f"C{k} {nodes[node]} 0 {_n(rf.lpf_shunt_c_farads(g[k - 1], f_c, 50.0))}")
        else:  # series inductor to the next node
            lines.append(f"L{k} {nodes[node]} {nodes[node + 1]} {_n(rf.lpf_series_l_henries(g[k - 1], f_c, 50.0))}")
            node += 1
    return [*lines, "RL out 0 50"]


@needs_ngspice
def test_the_butterworth_and_chebyshev_prototypes_attenuate_as_calculated(tmp_path: Path) -> None:
    f_c = 1.05e9
    res = _run(tmp_path, "bw5", _ladder(rf.butterworth_g_values(5), f_c), 1.8e9, 1.8e9 + 2e6)
    measured = -20 * math.log10(res.vectors["out"][0] / 0.5)
    assert measured == pytest.approx(rf.butterworth_attenuation_db(5, 1.8e9, f_c), abs=0.05) and round(measured, 2) == 23.43
    cheb = rf.chebyshev_g_values(5, 0.5)
    res = _run(tmp_path, "ch5", _ladder(cheb, 1e9), 2e9, 2e9 + 2e6)
    measured = -20 * math.log10(res.vectors["out"][0] / 0.5)
    assert measured == pytest.approx(rf.chebyshev_attenuation_db(5, 0.5, 2e9, 1e9), abs=0.05) and round(measured, 2) == 42.04
    # at the ripple-band edge the Chebyshev loss is the ripple
    res = _run(tmp_path, "ch5_edge", _ladder(cheb, 1e9), 1e9, 1e9 + 2e3)
    assert -20 * math.log10(res.vectors["out"][0] / 0.5) == pytest.approx(0.5, abs=0.05)
