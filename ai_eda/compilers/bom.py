"""BOM and CPL generation directly from the IR.

The BOM refuses to print an MPN / manufacturer / package / supplier part
number that is not authoritative, and it does not print a blank for one that
is *absent* either: both are emitted as ``NOT_VERIFIED`` so the gap is
visible in the file itself rather than papered over (a blank cell reads as
"nothing to say", an unknown identity is a finding).

The CPL writes the IR placement as-is: ``Mid X`` / ``Mid Y`` are the
footprint anchor in board coordinates (mm, Y down), ``Rotation`` the IR
rotation, ``Layer`` ``Top`` / ``Bottom``. The reviewer compares these rows
with the footprints of the compiled board.
"""

from __future__ import annotations

import csv
import io

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, ProvenanceKind, Traced
from ai_eda.compilers.base import CompileContext, Compiler

NOT_VERIFIED = "NOT_VERIFIED"


def _fact(t: Traced | None) -> str:
    """Cell text for a traced identity value: the value if authoritative, else ``NOT_VERIFIED`` (also when absent)."""
    if t is None:
        return NOT_VERIFIED
    if t.provenance.kind in (ProvenanceKind.AUTHORITATIVE, ProvenanceKind.USER_REQUIREMENT):
        return str(t.value)
    return NOT_VERIFIED


class BOMCompiler(Compiler):
    id = "compiler.bom"
    kind = ArtifactKind.BOM

    COLUMNS = ["Reference", "Value", "Description", "Manufacturer", "MPN", "Package", "Footprint", "Supplier", "SupplierPN", "Qty"]

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
                    _fact(c.manufacturer),
                    _fact(c.mpn),
                    _fact(c.package),
                    f"{c.footprint.library}:{c.footprint.name}" if c.footprint and c.footprint.verified else NOT_VERIFIED,
                    supplier.supplier if supplier else NOT_VERIFIED,
                    _fact(supplier.supplier_part_number) if supplier else NOT_VERIFIED,
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
