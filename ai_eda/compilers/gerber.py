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
The drill set is ``<stem>.drl`` plus one file per blind / buried / micro via
span of the IR (:func:`drill_pairs`, the names kicad-cli 10.0.6 writes).
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, hash_file_set
from ai_eda.ir.pcb import copper_position
from ai_eda.tools.kicad.cli import KicadCli, gerber_layers

#: subdirectory of the workdir that receives every manufacturing export
EXPORT_DIR = "gerber"
#: the drill export's version on a board with blind / buried / micro vias (their span files); every other board stays ``_Exporter.version``
TYPED_VIA_VERSION = "0.3"


def _kicad(ctx: CompileContext, what: str = "gerber/drill export") -> KicadCli:
    kicad = ctx.tools.get("kicad_cli")
    if not isinstance(kicad, KicadCli):
        raise ToolUnavailableError(f"{what} needs ctx.tools['kicad_cli'] (a KicadCli)")
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
    #: the workdir subdirectory the export writes into
    export_dir: str = EXPORT_DIR
    #: what the refusal without a KicadCli calls this export
    what: str = "gerber/drill export"

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        raise NotImplementedError

    def _version(self, ir: CircuitIR) -> str:
        """The exporter version this export is stamped with (a mechanism that acts only on some boards stamps its own)."""
        return self.version

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        kicad = _kicad(ctx, self.what)
        pcb = _fresh_pcb(ir)
        files = self._export(kicad, pcb, Path(ctx.workdir) / self.export_dir, ir)
        return ArtifactRef(
            kind=self.kind,
            path=str(self._primary(files)),
            content_hash=hash_file_set(files),
            generated_from_ir_hash=ir.content_hash(),
            generator=self.id,
            generator_version=f"{self._version(ir)}/kicad-cli {kicad.version()}",
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


def _drill_layer_name(layer: str) -> str:
    """kicad-cli 10.0.6's name of a copper layer in a drill pair file name: ``F.Cu`` front, ``In<k>.Cu`` in<k>, ``B.Cu`` back (measured)."""
    if layer == "F.Cu":
        return "front"
    if layer == "B.Cu":
        return "back"
    return "in" + str(copper_position(layer))


def drill_pairs(ir: CircuitIR) -> list[str]:
    """The drill pair names of ``ir``'s blind / buried / micro vias, sorted and unique: ``<upper>-<lower>`` as kicad-cli 10.0.6 names the files.

    One name per span whatever the via kinds on it (a micro and a blind via
    ``F.Cu``-``In2.Cu`` share ``<stem>-front-in2.drl``, measured); a through
    via has no pair file, and neither has a typed via whose span is the
    outer pair ``F.Cu``-``B.Cu`` (a micro via on a 2-layer board): kicad-cli
    10.0.6 writes it into ``<stem>.drl`` (measured: ``MixedPlating,1,2``, no
    ``<stem>-front-back.drl``). An IR without such a via gives ``[]``.
    """
    if ir.pcb is None:
        return []
    return sorted({f"{_drill_layer_name(v.layers[0])}-{_drill_layer_name(v.layers[1])}" for v in ir.pcb.vias
                   if getattr(v, "kind", None) is not None and set(v.layers) != {"F.Cu", "B.Cu"}})


class DrillExporter(_Exporter):
    """Merged Excellon drill file ``<stem>.drl``, plus one ``<stem>-<pair>.drl`` per blind / buried / micro via span (:func:`drill_pairs`).

    An export with pair files is stamped ``0.3`` (:data:`TYPED_VIA_VERSION`);
    a board without typed vias exports exactly as before (``0.2``).
    """

    id = "export.drill"
    kind = ArtifactKind.DRILL

    def _export(self, kicad: KicadCli, pcb: Path, out_dir: Path, ir: CircuitIR) -> list[Path]:
        return kicad.export_drill(pcb, out_dir, drill_pairs(ir))

    def _version(self, ir: CircuitIR) -> str:
        return TYPED_VIA_VERSION if drill_pairs(ir) else self.version
