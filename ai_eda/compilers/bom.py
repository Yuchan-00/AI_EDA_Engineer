"""BOM and CPL generation directly from the IR.

The BOM refuses to print an MPN / manufacturer / package / supplier part
number that is not authoritative, and it does not print a blank for one that
is *absent* either: both are emitted as ``NOT_VERIFIED`` so the gap is
visible in the file itself rather than papered over (a blank cell reads as
"nothing to say", an unknown identity is a finding). ``DatasheetHash`` is
the sha256 of the archived datasheet the MPN's own SourceRef names (the copy
the MPN was found in, ``document_path`` + ``content_hash``) - the evidence
pointer travels with the BOM; an MPN whose datasheet grounding is absent has
``NOT_VERIFIED`` there even when its value prints.

The BOM is opened in spreadsheet software and uploaded to assembly houses,
so an identity / sourcing cell that such software would execute (a value
starting with ``=``, ``+``, ``-``, ``@``, or carrying a control character -
CSV formula injection from a community catalog dump or a hand-edited IR)
is refused with :class:`~ai_eda.errors.CompileError` rather than written.
The free-text ``Value`` / ``Description`` columns are the design's own
words and are written as they stand (known gap: they are not neutralised,
because the reviewer compares ``Value`` with the board).

The CPL writes the IR placement as-is: ``Mid X`` / ``Mid Y`` are the
footprint anchor in board coordinates (mm, Y down), ``Rotation`` the IR
rotation, ``Layer`` ``Top`` / ``Bottom``. The reviewer compares these rows
with the footprints of the compiled board.
"""

from __future__ import annotations

import csv
import io

from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, ProvenanceKind, Traced
from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.parts.catalog import unsafe_cell

NOT_VERIFIED = "NOT_VERIFIED"


def _fact(t: Traced | None, where: str = "") -> str:
    """Cell text for a traced identity value: the value if authoritative, else ``NOT_VERIFIED`` (also when absent); never a cell a spreadsheet would execute."""
    if t is None:
        return NOT_VERIFIED
    if t.provenance.kind in (ProvenanceKind.AUTHORITATIVE, ProvenanceKind.USER_REQUIREMENT):
        text = str(t.value)
        why = unsafe_cell(text)
        if why is not None:
            raise CompileError(f"BOM cell {where or 'identity'} = {text!r} {why}; refusing to write a cell spreadsheet software would execute")
        return text
    return NOT_VERIFIED


def _datasheet_hash(mpn: Traced | None) -> str:
    """The sha256 of the archived datasheet an authoritative MPN was grounded in, else ``NOT_VERIFIED``."""
    if mpn is None or mpn.provenance.kind is not ProvenanceKind.AUTHORITATIVE:
        return NOT_VERIFIED
    src = mpn.provenance.source
    if src is None or not src.content_hash or not src.document_path:
        return NOT_VERIFIED
    return src.content_hash


class BOMCompiler(Compiler):
    id = "compiler.bom"
    kind = ArtifactKind.BOM

    COLUMNS = ["Reference", "Value", "Description", "Manufacturer", "MPN", "Package", "Footprint", "Supplier", "SupplierPN", "DatasheetHash", "Qty"]

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        for c in sorted(ir.components, key=lambda c: c.ref):
            supplier = c.sourcing[0] if c.sourcing else None
            w.writerow(
                [
                    c.ref,
                    c.value,
                    c.description,
                    _fact(c.manufacturer, f"{c.ref}.Manufacturer"),
                    _fact(c.mpn, f"{c.ref}.MPN"),
                    _fact(c.package, f"{c.ref}.Package"),
                    f"{c.footprint.library}:{c.footprint.name}" if c.footprint and c.footprint.verified else NOT_VERIFIED,
                    supplier.supplier if supplier else NOT_VERIFIED,
                    _fact(supplier.supplier_part_number, f"{c.ref}.SupplierPN") if supplier else NOT_VERIFIED,
                    _datasheet_hash(c.mpn),
                    1,
                ]
            )
        return self._write(ir, ctx.workdir / "bom.csv", buf.getvalue())


class CPLCompiler(Compiler):
    id = "compiler.cpl"
    kind = ArtifactKind.CPL

    COLUMNS = ["Designator", "Mid X", "Mid Y", "Rotation", "Layer"]

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        if ir.pcb is not None:
            for p in sorted(ir.pcb.placements, key=lambda p: p.component_ref):
                w.writerow([p.component_ref, f"{p.x_mm:.4f}mm", f"{p.y_mm:.4f}mm", f"{p.rotation_deg:g}", p.side.value.capitalize()])
        return self._write(ir, ctx.workdir / "cpl.csv", buf.getvalue())
