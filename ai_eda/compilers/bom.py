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
so no cell is written that such software would execute (CSV formula
injection from a community catalog dump, a ``--catalog-supplier`` label or
a hand-edited IR). Two rules, both from
:mod:`ai_eda.tools.manufacturing.csv_cells`:

* ``Reference``, the identity cells (``Manufacturer``, ``MPN``,
  ``Package``), ``Footprint``, the sourcing cells (``Supplier``,
  ``SupplierPN``), ``DatasheetHash`` and the CPL ``Designator`` are
  *refused* with :class:`~ai_eda.errors.CompileError` when they start with
  ``=``, ``+``, ``-``, ``@`` or carry a control character (checked on the
  stripped text, as the catalog reader's check is): such a cell is a
  verdict on the IR, and nothing is written.
* ``Value`` and ``Description`` are the design's own words (``-12V`` is a
  legitimate rail name), so they are *neutralised* instead: a cell that
  would execute is written as the original text with a leading apostrophe
  (:data:`TEXT_PREFIX`), the alteration is listed in ``ArtifactRef.notes``
  and reaches ``compile.bom`` (``details['neutralised']`` + message) and
  the stage message, so the change is reported rather than silent. The
  encoding is injective and :func:`bom_cell_text` is its single decoder;
  the reviewer compares the decoded cell with the board's value. A control
  character anywhere in a free-text cell (a newline included: a line-based
  reader would see the next line as a new row that may itself start with
  ``=``, and no prefix protects that) is refused, not neutralised.

That the apostrophe stays a visible literal character when Excel /
LibreOffice / a fab importer read the CSV is their documented behaviour,
not measured by this project; the code asserts only that the written cell
no longer starts with a formula character.

The CPL writes the IR placement as-is: ``Mid X`` / ``Mid Y`` / ``Rotation``
are numbers the compiler formats (a leading ``-`` is a sign, not a formula)
and the reviewer parses them as floats: they are neither refused nor
neutralised; ``Designator`` is refused like ``Reference``. ``Mid X`` /
``Mid Y`` are the footprint anchor in board coordinates (mm, Y down),
``Rotation`` the IR rotation, ``Layer`` ``Top`` / ``Bottom``. The reviewer
compares these rows with the footprints of the compiled board.
"""

from __future__ import annotations

import csv
import io

from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, ProvenanceKind, Traced
from ai_eda.compilers.base import CompileContext, Compiler, check_finite
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.manufacturing.csv_cells import TEXT_PREFIX, bom_cell_text, free_text_cell, unsafe_cell

NOT_VERIFIED = "NOT_VERIFIED"

__all__ = ["BOMCompiler", "CPLCompiler", "NOT_VERIFIED", "TEXT_PREFIX", "bom_cell_text", "free_text_cell"]


def _plain(text: str, where: str) -> str:
    """``text`` as a cell, or :class:`CompileError` when spreadsheet software would execute it (checked stripped, like the catalog does)."""
    why = unsafe_cell(text.strip())
    if why is not None:
        raise CompileError(f"BOM cell {where} = {text!r} {why}; refusing to write a cell spreadsheet software would execute")
    return text


def _free_text(text: str, where: str) -> tuple[str, str | None]:
    """A ``Value`` / ``Description`` cell through :func:`free_text_cell`; a control character is a :class:`CompileError`."""
    try:
        return free_text_cell(text, where)
    except ValueError as e:
        raise CompileError(str(e)) from e


def _fact(t: Traced | None, where: str = "") -> str:
    """Cell text for a traced identity value: the value if authoritative or typed by the user, else ``NOT_VERIFIED`` (also when absent); never a cell a spreadsheet would execute."""
    if t is None:
        return NOT_VERIFIED
    if t.provenance.kind in (ProvenanceKind.AUTHORITATIVE, ProvenanceKind.USER_REQUIREMENT):
        return _plain(str(t.value), where or "identity")
    return NOT_VERIFIED


def _datasheet_hash(mpn: Traced | None, where: str) -> str:
    """The sha256 of the archived datasheet an authoritative MPN was grounded in, else ``NOT_VERIFIED``; an evidence cell, gated like an identity."""
    if mpn is None or mpn.provenance.kind is not ProvenanceKind.AUTHORITATIVE:
        return NOT_VERIFIED
    src = mpn.provenance.source
    if src is None or not src.content_hash or not src.document_path:
        return NOT_VERIFIED
    return _plain(src.content_hash, where)


class BOMCompiler(Compiler):
    id = "compiler.bom"
    #: 0.2: free-text cells are neutralised (leading apostrophe) - a 0.1 file for the same IR may differ in those bytes
    version = "0.2"
    kind = ArtifactKind.BOM

    COLUMNS = ["Reference", "Value", "Description", "Manufacturer", "MPN", "Package", "Footprint", "Supplier", "SupplierPN", "DatasheetHash", "Qty"]

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        notes: list[str] = []
        for c in sorted(ir.components, key=lambda c: c.ref):
            supplier = c.sourcing[0] if c.sourcing else None
            value, value_note = _free_text(c.value, f"{c.ref}.Value")
            description, description_note = _free_text(c.description, f"{c.ref}.Description")
            w.writerow(
                [
                    _plain(c.ref, "Reference"),
                    value,
                    description,
                    _fact(c.manufacturer, f"{c.ref}.Manufacturer"),
                    _fact(c.mpn, f"{c.ref}.MPN"),
                    _fact(c.package, f"{c.ref}.Package"),
                    _plain(f"{c.footprint.library}:{c.footprint.name}", f"{c.ref}.Footprint") if c.footprint and c.footprint.verified else NOT_VERIFIED,
                    _plain(supplier.supplier, f"{c.ref}.Supplier") if supplier else NOT_VERIFIED,
                    _fact(supplier.supplier_part_number, f"{c.ref}.SupplierPN") if supplier else NOT_VERIFIED,
                    _datasheet_hash(c.mpn, f"{c.ref}.DatasheetHash"),
                    1,
                ]
            )
            # row order, Value before Description: the notes are read next to the file
            notes.extend(n for n in (value_note, description_note) if n)
        return self._write(ir, ctx.workdir / "bom.csv", buf.getvalue(), notes=notes)


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
