"""Runs real kicad-cli when installed; skipped otherwise."""

from pathlib import Path

import pytest

from ai_eda.ir import LibraryRef, ValidationStatus
from ai_eda.tools.kicad import KicadCli, KicadLibrary

kicad = KicadCli()
pytestmark = pytest.mark.skipif(not kicad.available(), reason="kicad-cli not installed")


def _demo_schematic() -> Path | None:
    for root in KicadLibrary().roots:
        demos = root / "demos"
        if demos.exists():
            for p in demos.rglob("*.kicad_sch"):
                return p
    return None


def test_version():
    assert kicad.version()[0].isdigit()


def test_library_resolves_real_symbol_and_footprint():
    lib = KicadLibrary()
    assert lib.resolve_symbol(LibraryRef(library="Device", name="R")).verified
    assert not lib.resolve_symbol(LibraryRef(library="Device", name="DefinitelyNotASymbol")).verified
    assert lib.resolve_footprint(LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric")).verified
    assert not lib.resolve_footprint(LibraryRef(library="Nope", name="X")).verified


def test_erc_on_demo_schematic(tmp_path: Path):
    sch = _demo_schematic()
    if sch is None:
        pytest.skip("no KiCad demo schematic found")
    res = kicad.run_erc(sch, tmp_path / "erc.json")
    assert res.tool == "kicad-cli"
    assert res.status in (ValidationStatus.PASS, ValidationStatus.FAIL)
    assert res.artifact_hash.startswith("sha256:")
    assert (tmp_path / "erc.json").exists()
    assert res.evidence[0].path.endswith("erc.json")
