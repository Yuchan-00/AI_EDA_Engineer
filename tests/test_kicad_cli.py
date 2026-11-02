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


# --------------------------------------------------------------------------- project file rules (the measurement behind PROJECT_RULES_MEASURED_VERSIONS)


def test_sibling_project_rules_are_applied_by_drc_canary(tmp_path: Path):
    """Does *this* kicad-cli read ``<stem>.kicad_pro`` beside the board and apply ``board.design_settings.rules``?

    Written on a machine without KiCad, so it asserts the expected outcome and
    fails honestly if kicad-cli behaves otherwise: a ``min_track_width`` of
    5 mm must produce ``track_width`` violations on the vertical-slice board
    (its naive tracks are 0.25 mm), 0.127 mm none; ERC must report the same
    violations with and without the project file; kicad-cli must not rewrite
    the project file. Once this passes on a kicad-cli version, that version
    is added to :data:`~ai_eda.tools.kicad.cli.PROJECT_RULES_MEASURED_VERSIONS`
    (with the observed facts in the module docstring) and the capability
    check may accept DRC as proof of the fab minimums on it.
    """
    from ai_eda.compilers import CompileContext, PCBCompiler, ProjectFileCompiler, SchematicCompiler
    from ai_eda.ir import ArtifactKind, authoritative
    from ai_eda.tools.kicad.cli import project_rules, run_drc_for, run_erc_for, sibling_project
    from ai_eda.tools.routing import route_naive
    from tests.conftest import DS
    from tests.fixtures_kicad import divider_with_connector_ir

    lib = KicadLibrary()
    if lib.footprint_file("Resistor_SMD", "R_0603_1608Metric") is None or lib.symbol_file("Device") is None:
        pytest.skip("KiCad libraries not installed")
    ir = divider_with_connector_ir(tmp_path, lib)
    ir.pcb.tracks = route_naive(ir, lib)
    assert all(t.width_mm < 5.0 for t in ir.pcb.tracks) and all(t.width_mm >= 0.127 for t in ir.pcb.tracks)
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})

    def compile_all() -> None:
        ir.artifacts[ArtifactKind.SCHEMATIC] = SchematicCompiler().compile(ir, ctx)
        ir.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(ir, ctx)
        ir.artifacts[ArtifactKind.KICAD_PROJECT] = ProjectFileCompiler().compile(ir, ctx)
        assert Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path) == sibling_project(Path(ir.artifacts[ArtifactKind.PCB].path))

    # an impossible rule: every track must violate it
    ir.pcb.manufacturing.min_track_width_mm = authoritative(5.0, DS, "mm")
    ir.pcb.manufacturing.min_clearance_mm = authoritative(0.127, DS, "mm")
    compile_all()
    assert project_rules(Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path)) == {"min_track_width": 5.0, "min_clearance": 0.127}
    before = Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path).read_bytes()
    strict = run_drc_for(ir, kicad, tmp_path)
    beside = sorted(p.name for p in tmp_path.iterdir())
    print("kicad-cli", kicad.version(), "files beside the board after DRC:", beside)
    assert strict.details["project_present"] is True and strict.details["project_rewritten"] is False
    assert Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path).read_bytes() == before and ir.artifacts[ArtifactKind.KICAD_PROJECT].matches_disk()
    assert strict.details["project_hash"] == ir.artifacts[ArtifactKind.KICAD_PROJECT].content_hash and strict.details["design_rules"] == {"min_track_width": 5.0, "min_clearance": 0.127}
    types = sorted({str(v.get("type")) for v in strict.details["errors"] + strict.details["warnings"]})
    assert strict.status is ValidationStatus.FAIL and "track_width" in types, f"kicad-cli {kicad.version()} did not apply the sibling project's min_track_width (violation types: {types})"
    assert sum(1 for v in strict.details["errors"] + strict.details["warnings"] if v.get("type") == "track_width") == len(ir.pcb.tracks)

    # the real fab minimum: the board passes, with the rules recorded on the result
    ir.pcb.manufacturing.min_track_width_mm = authoritative(0.127, DS, "mm")
    compile_all()
    clean = run_drc_for(ir, kicad, tmp_path)
    assert clean.status is ValidationStatus.PASS and clean.details["errors"] == [] and clean.details["warnings"] == []
    assert clean.details["design_rules"] == {"min_track_width": 0.127, "min_clearance": 0.127} and clean.details["project_hash"] == ir.artifacts[ArtifactKind.KICAD_PROJECT].content_hash
    assert clean.details["project_rewritten"] is False and ir.artifacts[ArtifactKind.KICAD_PROJECT].matches_disk()

    # ERC: the same report with and without the project file beside the schematic
    with_project = run_erc_for(ir, kicad, tmp_path)
    assert with_project.details["project_present"] is True and with_project.details["project_hash"] == ir.artifacts[ArtifactKind.KICAD_PROJECT].content_hash
    Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path).unlink()
    without_project = run_erc_for(ir, kicad, tmp_path)
    assert without_project.details["project_present"] is False and "project_hash" not in without_project.details and "beside" in without_project.details["project_reason"]
    for key in ("errors", "warnings", "other_severity", "excluded"):
        assert with_project.details[key] == without_project.details[key], key
    assert with_project.status is without_project.status is ValidationStatus.PASS
