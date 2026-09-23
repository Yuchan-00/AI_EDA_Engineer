"""BOM and CPL generation directly from the IR.

The BOM prints an MPN / manufacturer / package / supplier part number only
when its provenance is ``authoritative`` (found in an archived datasheet)
or ``user_requirement`` (typed by the user); a model proposal, an
assumption or a derived value prints as ``NOT_VERIFIED``, and so does an
*absent* identity - never a blank, so the gap is visible in the file
itself rather than papered over (a blank cell reads as "nothing to say",
an unknown identity is a finding). ``DatasheetHash`` is the sha256 of the
archived datasheet the MPN's own SourceRef names (the copy the MPN was
found in, ``document_path`` + ``content_hash``) - the evidence pointer
travels with the BOM; an MPN whose datasheet grounding is absent (a
user-typed MPN included) has ``NOT_VERIFIED`` there even when its value
prints, and ``ir.component_provenance`` / the reviewer say the same.

The BOM is opened in spreadsheet software and uploaded to assembly houses,
so a reference, identity, footprint or sourcing cell that such software
would execute (a value starting with ``=``, ``+``, ``-``, ``@``, or
carrying a control character - CSV formula injection from a community
catalog dump, a ``--catalog-supplier`` label or a hand-edited IR) is
refused with :class:`~ai_eda.errors.CompileError` rather than written; the
check runs on the stripped text, as the catalog reader's does. The
free-text ``Value`` / ``Description`` columns are the design's own words
and are written as they stand (known gap: they are not neutralised,
because the reviewer compares ``Value`` with the board and rail names such
as ``-12V`` are legitimate values).

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
from ai_eda.compilers.base import CompileContext, Compiler, check_finite
from ai_eda.parts.catalog import unsafe_cell
from ai_eda.tools.kicad import sexpr

NOT_VERIFIED = "NOT_VERIFIED"


def _plain(text: str, where: str) -> str:
    """``text`` as a cell, or :class:`CompileError` when spreadsheet software would execute it (checked stripped, like the catalog does)."""
    why = unsafe_cell(text.strip())
    if why is not None:
        raise CompileError(f"BOM cell {where} = {text!r} {why}; refusing to write a cell spreadsheet software would execute")
    return text


def _fact(t: Traced | None, where: str = "") -> str:
    """Cell text for a traced identity value: the value if authoritative or typed by the user, else ``NOT_VERIFIED`` (also when absent); never a cell a spreadsheet would execute."""
    if t is None:
        return NOT_VERIFIED
    if t.provenance.kind in (ProvenanceKind.AUTHORITATIVE, ProvenanceKind.USER_REQUIREMENT):
        return _plain(str(t.value), where or "identity")
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
                    _plain(c.ref, "Reference"),
                    c.value,
                    c.description,
                    _fact(c.manufacturer, f"{c.ref}.Manufacturer"),
                    _fact(c.mpn, f"{c.ref}.MPN"),
                    _fact(c.package, f"{c.ref}.Package"),
                    _plain(f"{c.footprint.library}:{c.footprint.name}", f"{c.ref}.Footprint") if c.footprint and c.footprint.verified else NOT_VERIFIED,
                    _plain(supplier.supplier, f"{c.ref}.Supplier") if supplier else NOT_VERIFIED,
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
                check_finite(p.model_dump(mode="json"), f"placement {p.component_ref}")
                # the rotation is written exactly as the board writer writes it (fixed decimals, no exponent, never -0),
                # so the reviewer's CPL-vs-board comparison never sees a formatting difference
                w.writerow([_plain(p.component_ref, "Designator"), f"{p.x_mm:.4f}mm", f"{p.y_mm:.4f}mm", sexpr.fmt_num(float(p.rotation_deg)), p.side.value.capitalize()])
        return self._write(ir, ctx.workdir / "cpl.csv", buf.getvalue())
