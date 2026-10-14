"""``ai-eda selftest`` (``ai_eda/selftest.py``): what counts as "ran", and the KiCad configuration line.

The run itself needs KiCad and ngspice and is exercised by the install CI
(``.github/workflows/install.yml``); here the run record (``ir.json`` /
``pipeline.json``) is written by hand, so every rule is pinned on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_eda.selftest import CONFIGURATION_VIOLATIONS, Check, _configuration_check, _violation_lines


def _ir(workdir: Path, *results: dict) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "ir.json").write_text(json.dumps({"validation": {"results": list(results)}}), encoding="utf-8")


def _result(check_id: str, status: str, warnings: list[str], **extra: list) -> dict:
    return {"check_id": check_id, "status": status, "message": f"0 error(s), {len(warnings)} warning(s)", "details": {"warnings": [{"type": t, "severity": "warning"} for t in warnings], **extra}}


def test_a_check_ran_only_on_a_tool_verdict():
    assert Check("x", "erc", True, status="FAIL", message="0 error(s), 6 warning(s)").ran  # a FAIL is still a verdict
    assert Check("x", "spice", True, status="PASS", message="1 analysis(es) run").ran
    assert not Check("x", "spice", True, status="NOT_VERIFIED", message="ngspice not available").ran
    assert not Check("x", "erc", True, status="FAIL", message="kicad-cli not found").ran
    assert not Check("x", "erc", True).ran  # NOT RUN
    assert Check("x", "architecture", True, status="NOT_VERIFIED", message="17 proposal(s) applied", proposes=True).ran
    assert not Check("x", "architecture", True, status="NOT_VERIFIED", message="no template", proposes=True).ran


def test_a_drc_that_ran_counts_whatever_kicad_cli_wrote_on_stderr():
    """The DRC stage appends ``; schematic parity not evaluated (<reason>)`` and the reason quotes kicad-cli's stderr lines:
    a font warning saying "not found" there is not a missing tool. The markers are read in the first clause only."""
    parity = (
        "schematic parity not evaluated (kicad-cli wrote on stderr, so parity may not have run: "
        "12:00:01: Warning: Font 'Noto Sans CJK KR' not found; substituting 'KiCad Font'.)"
    )
    assert Check("x", "drc", True, status="PASS", message=f"0 error(s), 0 warning(s); {parity}").ran
    assert Check("x", "drc", True, status="FAIL", message=f"2 error(s), 0 warning(s); {parity}").ran
    assert Check("x", "drc", True, status="PASS", message="0 error(s), 0 warning(s); schematic parity: 0 issue(s)").ran
    assert not Check("x", "drc", True, status="FAIL", message=f"kicad-cli not found; {parity}").ran  # the first clause still counts
    assert not Check("x", "drc", True, status="NOT_VERIFIED", message="kicad-cli not available").ran


def test_library_table_violations_fail_the_configuration_line(tmp_path: Path):
    """The fresh-install picture of the CI (2026-10-14): 3 parts, each missing from the global library tables."""
    _ir(
        tmp_path,
        _result("kicad.erc", "FAIL", ["lib_symbol_issues"] * 3 + ["footprint_link_issues"] * 3),
        _result("kicad.drc", "FAIL", ["lib_footprint_issues"] * 3, unconnected_items=[], schematic_parity=[]),
    )
    check = _configuration_check(tmp_path, required=True)
    assert check.status == "FAIL" and not check.ran
    assert check.message == (
        "KiCad did not resolve its libraries (its global library tables, or the libraries they name: `ai-eda doctor` shows the tables, "
        "`ai-eda setup-kicad` writes missing ones): footprint_link_issues x3, lib_footprint_issues x3, lib_symbol_issues x3"
    )
    assert _violation_lines(tmp_path) == ["DRC FAIL: lib_footprint_issues x3", "ERC FAIL: footprint_link_issues x3, lib_symbol_issues x3"]
    assert {"lib_symbol_issues", "footprint_link_issues", "lib_footprint_issues"} == CONFIGURATION_VIOLATIONS


def test_design_findings_do_not_fail_the_configuration_line(tmp_path: Path):
    """A design's own warnings are the design's verdict, not an installation problem; the latest result counts."""
    _ir(
        tmp_path,
        _result("kicad.erc", "FAIL", ["lib_symbol_issues"]),  # an earlier run, before setup-kicad
        _result("kicad.erc", "FAIL", ["power_pin_not_driven"]),
        _result("kicad.drc", "PASS", [], unconnected_items=[], schematic_parity=[]),
    )
    check = _configuration_check(tmp_path, required=True)
    assert check.status == "PASS" and check.ran and check.message == "no library-table violation in DRC / ERC"
    assert _violation_lines(tmp_path) == ["DRC PASS: 0 error(s), 0 warning(s)", "ERC FAIL: power_pin_not_driven x1"]


def test_no_kicad_result_is_not_run(tmp_path: Path):
    _ir(tmp_path, {"check_id": "calc.recompute", "status": "PASS", "message": "", "details": {}})
    check = _configuration_check(tmp_path, required=False)
    assert check.status == "NOT RUN" and not check.ran and check.message == "no kicad-cli ERC / DRC result"
    assert _configuration_check(tmp_path / "missing", required=True).status == "NOT RUN"
