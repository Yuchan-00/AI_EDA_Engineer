"""Gerber and drill exporters: PCB artifact -> fab files via ``kicad-cli``.

These are registered like compilers (``ArtifactKind.GERBER`` /
``ArtifactKind.DRILL``) so "regenerate from IR" works for them too, but they
do not read the IR directly: they run ``kicad-cli pcb export`` on the
compiled ``.kicad_pcb``. They therefore refuse when that board artifact is
missing, stale with respect to the IR, or differs from what is on disk - an
export of the wrong board would otherwise be stamped with the current IR
hash and look fresh.

kicad-cli embeds a timestamp in every export, so two exports of the same
board differ byte-for-byte; the artifact hash is evidence of *which* files
were checked, not a reproducibility claim.

The gerber layer list follows ``ir.pcb.layers`` (``F.Cu``, every
``In<n>.Cu``, ``B.Cu`` plus the fab set), and zones are refilled at plot
time (``--check-zones``), so the plots contain the pour the IR specifies.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, hash_file_set
from ai_eda.tools.kicad.cli import KicadCli, gerber_layers

#: subdirectory of the workdir that receives every manufacturing export
EXPORT_DIR = "gerber"


def _kicad(ctx: CompileContext) -> KicadCli:
    kicad = ctx.tools.get("kicad_cli")
    if not isinstance(kicad, KicadCli):
        raise ToolUnavailableError("gerber/drill export needs ctx.tools['kicad_cli'] (a KicadCli)")
    if not kicad.available():
        raise ToolUnavailableError("kicad-cli not found")
    return kicad


def _fresh_pcb(ir: CircuitIR) -> Path:
    art = ir.artifacts.get(ArtifactKind.PCB)
    if art is None:
        raise NothingToCompileError("no PCB artifact: nothing to export (compile the board first)")
    if art.is_stale(ir.content_hash()):
        raise CompileError(f"PCB artifact was generated from IR {art.generated_from_ir_hash} but the IR is now {ir.content_hash()}; regenerate the board before exporting")
    if not art.matches_disk():
        raise CompileError(f"PCB artifact {art.path} on disk does not match its recorded hash; regenerate the board before exporting")
    return Path(art.path)


class _Exporter(Compiler):
    version = "0.2"

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        raise NotImplementedError

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        kicad = _kicad(ctx)
        pcb = _fresh_pcb(ir)
        files = self._export(kicad, pcb, Path(ctx.workdir) / EXPORT_DIR, ir)
        return ArtifactRef(
            kind=self.kind,
            path=str(self._primary(files)),
            content_hash=hash_file_set(files),
            generated_from_ir_hash=ir.content_hash(),
            generator=self.id,
            generator_version=f"{self.version}/kicad-cli {kicad.version()}",
            files=[str(f) for f in files],
        )

    @staticmethod
    def _primary(files: list[Path]) -> Path:
        return files[0]


class GerberExporter(_Exporter):
    """Fab gerber set for every copper layer of ``ir.pcb`` (``*.gbr`` + ``<stem>-job.gbrjob``); ``path`` is the manifest."""

    id = "export.gerber"
    kind = ArtifactKind.GERBER

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        if ir.pcb is None:
            raise CompileError("ir.pcb is None: the gerber layer list comes from ir.pcb.layers")
        return kicad.export_gerbers(pcb, out_dir, gerber_layers(layer.name for layer in ir.pcb.layers), check_zones=True)

    @staticmethod
    def _primary(files: list[Path]) -> Path:
        return next(f for f in files if f.suffix.lower() == ".gbrjob")


class DrillExporter(_Exporter):
    """Merged Excellon drill file ``<stem>.drl``."""

    id = "export.drill"
    kind = ArtifactKind.DRILL

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        return kicad.export_drill(pcb, out_dir)
