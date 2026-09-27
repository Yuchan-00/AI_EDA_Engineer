"""3D outputs of the board: the built-in preview GLB (compiled from the IR) and KiCad's own STEP / GLB / render (kicad-cli exports).

Two routes, two kinds of artifact, and no status claimed by either (a
picture of the board is not evidence about it):

* :class:`Preview3DCompiler` (``ArtifactKind.MODEL_3D``,
  ``<workdir>/<project>.preview.glb``) - a compiler like the PCB compiler: a
  pure function of the IR and the libraries on disk (the footprints from the
  KiCad library, body heights from the STEP files the footprints' ``(model
  ...)`` name, :mod:`ai_eda.tools.model3d`), byte-deterministic (same IR +
  same libraries -> byte-identical GLB, no timestamp, no path), registered
  with ``generated_from_ir_hash``. It refuses what the PCB compiler refuses
  (no ``ir.pcb``, no components, a project id that is no file stem) and what
  the scene refuses (:class:`~ai_eda.tools.model3d.scene.SceneError`, a
  :class:`~ai_eda.errors.CompileError`: no outline, a component without a
  placement or a footprint on disk). The 3D library is
  ``ctx.tools["model3d_dir"]`` when the key is given (a directory or
  ``None`` = none: every body a flat outline), else
  :func:`~ai_eda.tools.model3d.models.find_3dmodel_dir` of the KiCad library.
  The orchestrator compiles it in the PCB stage right after the board and
  records no ``compile.model_3d`` result: the stage message says what was
  written (bodies with / without a STEP height) or why not, and a refusal
  never changes the stage's status.
* :class:`StepExporter` / :class:`GlbExporter` / :class:`RenderExporter`
  (``ArtifactKind.KICAD_STEP`` / ``KICAD_GLB`` / ``KICAD_RENDER``, under
  ``<workdir>/3d/``) - exporters like the gerber set: they run
  ``kicad-cli pcb export step`` / ``pcb export glb`` / ``pcb render`` (top
  and bottom) on the fresh compiled ``.kicad_pcb`` only
  (:func:`~ai_eda.compilers.gerber._fresh_pcb`), and are registered with the
  hash of the files written. The flags are the KiCad documentation's, NOT
  measured on 10.0.6 (:data:`~ai_eda.tools.kicad.cli.EXPORT_3D_MEASURED_VERSIONS`
  is empty; the canary ``tests/test_kicad_3d_export.py`` skips without
  kicad-cli). KiCad writes its own timestamp / generator data into these
  files, so the hash says *which* file, not reproducibility. The
  MANUFACTURING_OUTPUTS stage runs them after the gerbers when kicad-cli is
  available and says ``3D export skipped: kicad-cli not found`` otherwise;
  a failed export is a note, never a FAIL.
"""

from __future__ import annotations

from pathlib import Path
from types import EllipsisType

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.compilers.gerber import _Exporter
from ai_eda.compilers.pcb import _BAD_STEM_RE, PCBCompiler
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.library import LibraryLookupError
from ai_eda.tools.model3d import Scene, build_scene, write_glb

__all__ = [
    "PREVIEW_SUFFIX",
    "EXPORT_3D_DIR",
    "RENDER_VIEWS",
    "MODEL_DIR_TOOL",
    "Preview3DCompiler",
    "StepExporter",
    "GlbExporter",
    "RenderExporter",
    "preview_path",
]

#: the preview GLB's file name after the project id (``<project>.preview.glb`` in the workdir)
PREVIEW_SUFFIX = ".preview.glb"
#: the workdir subdirectory KiCad's 3D exports are written into
EXPORT_3D_DIR = "3d"
#: the sides ``kicad-cli pcb render`` is run for (one PNG each)
RENDER_VIEWS: tuple[str, ...] = ("top", "bottom")
#: the ``ctx.tools`` key that overrides the 3D model library directory (a Path, or None for none)
MODEL_DIR_TOOL = "model3d_dir"


def preview_path(ir: CircuitIR, workdir: Path | str) -> Path:
    """Where :class:`Preview3DCompiler` writes the preview of ``ir``."""
    return Path(workdir) / f"{ir.project.id}{PREVIEW_SUFFIX}"


class Preview3DCompiler(Compiler):
    """The built-in 3D preview of the placed board as a glTF 2.0 binary (module docstring)."""

    id = "compiler.preview_glb"
    version = "0.1"
    kind = ArtifactKind.MODEL_3D

    @staticmethod
    def _model_dir(ctx: CompileContext) -> Path | None | EllipsisType:
        if MODEL_DIR_TOOL not in ctx.tools:
            return ...
        value = ctx.tools[MODEL_DIR_TOOL]
        if value is not None and not isinstance(value, (str, Path)):
            raise CompileError(f"ctx.tools[{MODEL_DIR_TOOL!r}] is not a directory path: {type(value).__name__}")
        return Path(value) if value is not None else None

    def scene(self, ir: CircuitIR, ctx: CompileContext) -> Scene:
        """The scene the GLB is written from; refuses (module docstring) instead of guessing."""
        if ir.pcb is None:
            raise NothingToCompileError("ir.pcb is None: no board to picture in 3D")
        if not ir.components:
            raise NothingToCompileError("IR has no components: no board to picture in 3D")
        if not ir.project.id or _BAD_STEM_RE.search(ir.project.id):
            raise CompileError(f"project id {ir.project.id!r} is not usable as a file stem (it names the preview file)")
        library = PCBCompiler._library(ctx)
        try:
            return build_scene(ir, library, model_dir=self._model_dir(ctx))
        except LibraryLookupError as e:
            raise CompileError(str(e)) from e

    def compile_scene(self, ir: CircuitIR, ctx: CompileContext) -> tuple[ArtifactRef, Scene]:
        """The registered preview and the scene it was written from (the orchestrator's stage message counts its bodies)."""
        scene = self.scene(ir, ctx)
        return self._write(ir, preview_path(ir, ctx.workdir), write_glb(scene)), scene

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        return self.compile_scene(ir, ctx)[0]


class _Kicad3DExporter(_Exporter):
    """KiCad's own 3D output of the compiled board (flags documented, NOT measured on 10.0.6)."""

    version = "0.1"
    export_dir = EXPORT_3D_DIR
    what = "3D export"


class StepExporter(_Kicad3DExporter):
    """``kicad-cli pcb export step``: ``3d/<stem>.step`` with KiCad's own part models."""

    id = "export.kicad_step"
    kind = ArtifactKind.KICAD_STEP

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        return [kicad.export_step(pcb, out_dir / f"{pcb.stem}.step")]


class GlbExporter(_Kicad3DExporter):
    """``kicad-cli pcb export glb``: ``3d/<stem>.glb``."""

    id = "export.kicad_glb"
    kind = ArtifactKind.KICAD_GLB

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        return [kicad.export_glb(pcb, out_dir / f"{pcb.stem}.glb")]


class RenderExporter(_Kicad3DExporter):
    """``kicad-cli pcb render`` from :data:`RENDER_VIEWS`: ``3d/<stem>-top.png``, ``3d/<stem>-bottom.png``; ``path`` is the top view."""

    id = "export.kicad_render"
    kind = ArtifactKind.KICAD_RENDER

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        return [kicad.render(pcb, out_dir / f"{pcb.stem}-{side}.png", side) for side in RENDER_VIEWS]
