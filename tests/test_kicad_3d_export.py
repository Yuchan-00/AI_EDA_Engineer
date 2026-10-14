"""Canaries for kicad-cli's own 3D output (``pcb export step`` / ``pcb export glb`` / ``pcb render``); skipped without kicad-cli.

The flags :mod:`ai_eda.tools.kicad.cli` passes come from the KiCad
documentation and are NOT measured on kicad-cli 10.0.6. Run these on a
machine with kicad-cli: the first records which of the flags the binary's
``--help`` lists, the second exports the vertical-slice board and checks
that a STEP, a glTF binary and a PNG come back. When both pass on a version,
add it to :data:`~ai_eda.tools.kicad.cli.EXPORT_3D_MEASURED_VERSIONS` with
what was observed in that module's docstring. A failure here is the finding
(a flag the binary does not know), not something to work around.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.kicad.cli import GLB_EXPORT_FLAGS, STEP_EXPORT_FLAGS

kicad = KicadCli()
pytestmark = pytest.mark.skipif(not kicad.available(), reason="kicad-cli not installed")

RENDER_FLAGS = ("--side", "--width", "--height", "--quality", "--background", "--zoom", "--perspective", "--output")


@pytest.mark.parametrize(
    ("subcommand", "flags"),
    [
        (["pcb", "export", "step"], (*STEP_EXPORT_FLAGS, "--output")),
        (["pcb", "export", "glb"], (*GLB_EXPORT_FLAGS, "--output")),
        (["pcb", "render"], RENDER_FLAGS),
    ],
)
def test_3d_export_flags_are_listed_by_help_canary(subcommand: list[str], flags: tuple[str, ...]):
    accepted = kicad.accepted_flags(subcommand)
    missing = sorted(set(flags) - accepted)
    print("kicad-cli", kicad.version(), " ".join(subcommand), "missing flags:", missing, "listed:", sorted(accepted))
    assert not missing, f"kicad-cli {kicad.version()} {' '.join(subcommand)} does not list {missing}"


def test_3d_exports_write_step_glb_and_png_canary(tmp_path: Path):
    from ai_eda.compilers import CompileContext, PCBCompiler
    from ai_eda.tools.routing import route_naive
    from tests.fixtures_kicad import divider_with_connector_ir

    lib = KicadLibrary()
    if lib.footprint_file("Resistor_SMD", "R_0603_1608Metric") is None or lib.symbol_file("Device") is None:
        pytest.skip("KiCad libraries not installed")
    ir = divider_with_connector_ir(tmp_path, lib)
    ir.pcb.tracks = route_naive(ir, lib)
    ctx = CompileContext(workdir=tmp_path, tools={"kicad_library": lib})
    pcb = Path(PCBCompiler().compile(ir, ctx).path)
    step = kicad.export_step(pcb, tmp_path / "3d" / "board.step")
    glb = kicad.export_glb(pcb, tmp_path / "3d" / "board.glb")
    renders = [kicad.render(pcb, tmp_path / "3d" / f"{side}.png", side, width=800, height=450) for side in ("top", "bottom")]
    sizes = {p.name: p.stat().st_size for p in (step, glb, *renders)}
    print("kicad-cli", kicad.version(), "3D outputs:", sizes)
    assert all(size > 0 for size in sizes.values())
