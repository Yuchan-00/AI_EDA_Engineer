"""BOM and CPL generation directly from the IR.

The BOM prints an MPN / manufacturer / package / supplier part number only
when its provenance is ``authoritative`` (found in an archived datasheet)
or ``user_requirement`` (typed by the user), and a footprint only when the
KiCad library holds it. Any other such cell - a model proposal, an
assumption, a derived value, an *absent* value - is left **empty**, and the
row's verification status says so in a column of its own,
``Unverified`` (:data:`UNVERIFIED_COLUMN`): each such cell by column name
with why (``MPN (not stated)``, ``Package (assumption)``,
``DatasheetHash (MPN not found in an archived datasheet)``), joined by
``; ``, or ``none`` (:data:`NOTHING_UNVERIFIED`) when the row has none - so
the gap stays a finding in the file without sitting in a part-number cell.
``compiler.bom`` 0.2 wrote the literal ``NOT_VERIFIED`` into the identity
and sourcing cells instead, which a distributor's or an assembler's BOM
upload reads as a part number; :func:`unverified_columns` reads the status
of either version (the GUI's one reader). ``DatasheetHash`` is the sha256 of the
archived datasheet the MPN's own SourceRef names (the copy the MPN was
found in, ``document_path`` + ``content_hash``) - the evidence pointer
travels with the BOM; an MPN whose datasheet grounding is absent (a
user-typed MPN included) leaves it empty and listed in ``Unverified`` even
when its value prints, and ``ir.component_provenance`` / the reviewer say
the same.

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

The CPL (``compiler.cpl`` 0.3): ``Mid X`` / ``Mid Y`` / ``Rotation`` are
numbers the compiler formats (a leading ``-`` is a sign, not a formula) and
the reviewer parses them as floats: they are neither refused nor
neutralised; ``Designator`` is refused like ``Reference``. ``Mid X`` /
``Mid Y`` are in the frame of the board's Gerbers (:data:`CPL_FRAME`): mm
from KiCad's page origin (0, 0) - the corner of the outline at
``ir.pcb.outline``'s origin, (0, 0) on every board the pipeline lays out -,
X to the right and **Y up**, so a part on the board has a negative
``Mid Y``. That is where ``kicad-cli pcb export gerbers`` puts the copper
(absolute coordinates, ``useauxorigin no``, Gerber's Y axis up: the 9 V
piano board's Edge_Cuts run from Y 0 to Y -141.5) and how KiCad's own
position file writes ``PosY``; ``compiler.cpl`` 0.2 wrote KiCad's page
frame (Y down), the opposite sign of every Gerber - an assembler that
lines the CPL up with the copper mirrors such a file. Neither JLCPCB's
CPL page ("Mid X/Mid Y - The X/Y coordinate of the component centroid.
Required units: Metric(mm)", jlcpcb.com/help/article/pick-place-file-for-pcb-assembly,
read 2026-11-19) nor its KiCad guide states an origin or an axis
direction, so the file's frame is the one the Gerbers in the same order
have. JLCPCB's column names are kept exactly, and a CSV has no comment
line an importer is known to skip, so the frame is stated here, in the
README and in the reviewer's message, not in the file. The centre is the
part's centre, as an
assembler's centroid file means it (JLCPCB's ``Mid X`` / ``Mid Y``: the
centre of the part, not its footprint origin): the centre of the box around
the footprint's pad positions (every pad's ``(at)``, where its hole or its
land is, placed as the board places it). That is the footprint origin for a
chip whose origin is its middle, and the middle of the pin field for a
through-hole part whose origin is pin 1 (KiCad's convention: 3.8 to 7.6 mm
apart on a DIP-14, a TO-220, a radial capacitor). A part whose footprint
cannot be read here, or has no pad, is written at its anchor and named in
the artifact's notes. ``Rotation`` is the IR rotation, ``Layer`` ``Top`` /
``Bottom``. The reviewer compares these rows with the footprints of the
compiled board, measuring the same centre from the pads the board file
holds, in the frame the CPL's version wrote: a CPL written by
``compiler.cpl`` 0.1 (``Mid X`` / ``Mid Y`` the footprint anchor, page frame)
is compared with the anchors and one written by 0.2 (the centre, page frame,
Y down) with the centres in that frame, and both are failed to be
recompiled; the message names the frame it read.

A part whose library footprint keeps itself out of the fabrication outputs
- KiCad's ``(attr exclude_from_bom)`` / ``(attr exclude_from_pos_files)``,
which KiCad's own libraries set on test point pads, mounting holes and
fiducials (:func:`~ai_eda.tools.kicad.library.fabrication_exclusions`) - has
no BOM row (``compiler.bom`` 0.4) and no CPL row (``compiler.cpl`` 0.4): it
is copper or a hole, nothing an assembler buys or places. The artifact's
notes name each part left out and why; :func:`bom_refs` / :func:`cpl_refs`
are the rows the reviewer expects. Earlier versions wrote every component
and every placement. A footprint that cannot be read is never assumed to be
excluded.
"""

from __future__ import annotations

import csv
import io

from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Placement, ProvenanceKind, Traced
from ai_eda.compilers.base import CompileContext, Compiler, check_finite
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import pad_center
from ai_eda.tools.kicad.library import EXCLUDE_FROM_BOM, EXCLUDE_FROM_POS_FILES, KicadLibrary, LibraryLookupError, fabrication_exclusions
from ai_eda.tools.manufacturing.csv_cells import TEXT_PREFIX, bom_cell_text, free_text_cell, unsafe_cell

#: the cell text ``compiler.bom`` 0.2 wrote for an identity / sourcing / evidence value nothing verified (read by :func:`unverified_columns`)
NOT_VERIFIED = "NOT_VERIFIED"
#: the BOM's verification-status column (module docstring): the cells of the row left empty because nothing verified them, with why
UNVERIFIED_COLUMN = "Unverified"
#: the status cell of a row whose every identity, sourcing and evidence cell is verified
NOTHING_UNVERIFIED = "none"
#: why a cell is empty, by case (the text after the column name in ``Unverified``)
NOT_STATED = "not stated"
NOT_IN_LIBRARY = "footprint not found in the KiCad library"
NO_MPN = "no MPN"
MPN_NOT_GROUNDED = "MPN not found in an archived datasheet"
NO_ARCHIVED_COPY = "the MPN's source names no archived copy"
#: the frame ``Mid X`` / ``Mid Y`` are written in (``compiler.cpl`` 0.3, module docstring): the Gerbers' frame
CPL_FRAME = "the Gerbers' frame: mm from KiCad's page origin (0, 0), X right, Y up (a part on the board has a negative Mid Y)"

__all__ = [
    "BOMCompiler", "CPLCompiler", "CPL_FRAME", "NOTHING_UNVERIFIED", "NOT_VERIFIED", "TEXT_PREFIX", "UNVERIFIED_COLUMN",
    "bom_cell_text", "bom_refs", "cpl_refs", "free_text_cell", "gerber_frame", "is_exclusion_note", "unverified_columns",
]


def _library(tools: dict) -> KicadLibrary:
    library = tools.get("kicad_library")
    return library if isinstance(library, KicadLibrary) else KicadLibrary()


#: what every note of a part left out of the BOM / CPL says after its reference (:func:`is_exclusion_note`)
EXCLUDED_NOTE = "not in the "


def _excluded_note(ref: str, output: str, token: str, footprint) -> str:
    return (f"{ref}: {EXCLUDED_NOTE}{output}: its footprint {footprint.library}:{footprint.name} carries ({token}) - copper or a hole, "
            "nothing to buy or place")


def is_exclusion_note(note: str) -> bool:
    """Whether a BOM / CPL artifact note names a part left out (module docstring), not a neutralised cell."""
    return note.split(": ", 1)[-1].startswith(EXCLUDED_NOTE + "BOM:") or note.split(": ", 1)[-1].startswith(EXCLUDED_NOTE + "CPL:")


def bom_refs(ir: CircuitIR, library: KicadLibrary) -> set[str]:
    """The references the BOM lists: every component but those whose footprint is excluded from the BOM (module docstring)."""
    return {c.ref for c in ir.components if EXCLUDE_FROM_BOM not in fabrication_exclusions(library, c.footprint)}


def cpl_refs(ir: CircuitIR, library: KicadLibrary) -> set[str]:
    """The designators the CPL lists: every placement but those whose footprint is excluded from position files (module docstring)."""
    if ir.pcb is None:
        return set()
    comps = {c.ref: c for c in ir.components}
    return {p.component_ref for p in ir.pcb.placements
            if EXCLUDE_FROM_POS_FILES not in fabrication_exclusions(library, comps[p.component_ref].footprint if p.component_ref in comps else None)}


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


#: a cell and why it is empty (``None``: the cell holds a verified value)
Cell = tuple[str, "str | None"]


def _fact(t: Traced | None, where: str = "") -> Cell:
    """A traced identity value's cell: the value if authoritative or typed by the user, else empty with why (absent, or its provenance kind);
    never a cell a spreadsheet would execute."""
    if t is None or t.value in (None, ""):
        return "", NOT_STATED
    if t.provenance.kind in (ProvenanceKind.AUTHORITATIVE, ProvenanceKind.USER_REQUIREMENT):
        return _plain(str(t.value), where or "identity"), None
    return "", str(t.provenance.kind)


def _datasheet_hash(mpn: Traced | None, where: str) -> Cell:
    """The sha256 of the archived datasheet an authoritative MPN was grounded in, else empty with why; an evidence cell, gated like an identity."""
    if mpn is None or mpn.value in (None, ""):
        return "", NO_MPN
    if mpn.provenance.kind is not ProvenanceKind.AUTHORITATIVE:
        return "", MPN_NOT_GROUNDED
    src = mpn.provenance.source
    if src is None or not src.content_hash or not src.document_path:
        return "", NO_ARCHIVED_COPY
    return _plain(src.content_hash, where), None


def unverified_columns(row: dict[str, str]) -> list[str]:
    """The columns of a BOM row whose value nothing verified: those its ``Unverified`` cell names (``compiler.bom`` 0.3), and any cell
    reading ``NOT_VERIFIED`` (a 0.2 file, which has no status column), in the row's column order."""
    named: set[str] = set()
    status = row.get(UNVERIFIED_COLUMN, "")
    if status and status != NOTHING_UNVERIFIED:
        named = {item.split(" (", 1)[0].strip() for item in status.split(";")}
    return [c for c, v in row.items() if c != UNVERIFIED_COLUMN and (c in named or v == NOT_VERIFIED)]


class BOMCompiler(Compiler):
    id = "compiler.bom"
    #: 0.2: free-text cells are neutralised (leading apostrophe) - a 0.1 file for the same IR may differ in those bytes;
    #: 0.3: an unverified identity / sourcing / evidence cell is empty, its status in the ``Unverified`` column (0.2: ``NOT_VERIFIED`` in the cell);
    #: 0.4: a part whose footprint is excluded from the BOM has no row (module docstring)
    version = "0.4"
    kind = ArtifactKind.BOM

    COLUMNS = ["Reference", "Value", "Description", "Manufacturer", "MPN", "Package", "Footprint", "Supplier", "SupplierPN", "DatasheetHash", "Qty",
               UNVERIFIED_COLUMN]

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        notes: list[str] = []
        library = _library(ctx.tools)
        for c in sorted(ir.components, key=lambda c: c.ref):
            if EXCLUDE_FROM_BOM in fabrication_exclusions(library, c.footprint):
                notes.append(_excluded_note(c.ref, "BOM", EXCLUDE_FROM_BOM, c.footprint))
                continue
            supplier = c.sourcing[0] if c.sourcing else None
            value, value_note = _free_text(c.value, f"{c.ref}.Value")
            description, description_note = _free_text(c.description, f"{c.ref}.Description")
            if c.footprint is None:
                footprint: Cell = ("", NOT_STATED)
            elif not c.footprint.verified:
                footprint = ("", NOT_IN_LIBRARY)
            else:
                footprint = (_plain(f"{c.footprint.library}:{c.footprint.name}", f"{c.ref}.Footprint"), None)
            checked: dict[str, Cell] = {
                "Manufacturer": _fact(c.manufacturer, f"{c.ref}.Manufacturer"),
                "MPN": _fact(c.mpn, f"{c.ref}.MPN"),
                "Package": _fact(c.package, f"{c.ref}.Package"),
                "Footprint": footprint,
                "Supplier": (_plain(supplier.supplier, f"{c.ref}.Supplier"), None) if supplier else ("", NOT_STATED),
                "SupplierPN": _fact(supplier.supplier_part_number, f"{c.ref}.SupplierPN") if supplier else ("", NOT_STATED),
                "DatasheetHash": _datasheet_hash(c.mpn, f"{c.ref}.DatasheetHash"),
            }
            status = "; ".join(f"{column} ({why})" for column, (_cell, why) in checked.items() if why is not None) or NOTHING_UNVERIFIED
            w.writerow([_plain(c.ref, "Reference"), value, description, *(cell for cell, _why in checked.values()), 1, status])
            # row order, Value before Description: the notes are read next to the file
            notes.extend(n for n in (value_note, description_note) if n)
        return self._write(ir, ctx.workdir / "bom.csv", buf.getvalue(), notes=notes)


def gerber_frame(x: float, y: float) -> tuple[float, float]:
    """A board point (KiCad's page frame, mm, Y down) in the Gerbers' frame (:data:`CPL_FRAME`: Y up)."""
    return x, -y


def _mm(v: float) -> str:
    """``v`` as a CPL cell: four decimals and ``mm``, never ``-0.0000``."""
    return f"{round(v, 4) + 0.0:.4f}mm"


def part_centre(ir: CircuitIR, placement: Placement, library: KicadLibrary) -> tuple[float, float, str | None]:
    """``(x, y, None)``: the centre of the box around the pad positions of ``placement``'s footprint, placed as the board places it
    (module docstring, the CPL); ``(anchor x, anchor y, why)`` where the footprint cannot be read or has no pad."""
    comp = next((c for c in ir.components if c.ref == placement.component_ref), None)
    if comp is None or comp.footprint is None:
        return placement.x_mm, placement.y_mm, "no footprint in the IR"
    try:
        fp = library.load_footprint(comp.footprint)
    except (LibraryLookupError, CompileError) as e:
        return placement.x_mm, placement.y_mm, f"footprint {comp.footprint.library}:{comp.footprint.name} not read ({e})"
    points = [pad_center(placement, pad) for pad in fp.pads]
    if not points:
        return placement.x_mm, placement.y_mm, f"footprint {fp.lib_id} has no pad"
    xs, ys = [x for x, _y in points], [y for _x, y in points]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, None


class CPLCompiler(Compiler):
    id = "compiler.cpl"
    #: 0.2: ``Mid X`` / ``Mid Y`` are the part's centre (the pads' box), no longer the footprint anchor; 0.3: in the Gerbers' frame
    #: (Y up, :data:`CPL_FRAME`), no longer KiCad's page frame (Y down) - module docstring; 0.4: a part whose footprint is excluded from
    #: position files has no row
    version = "0.4"
    kind = ArtifactKind.CPL

    COLUMNS = ["Designator", "Mid X", "Mid Y", "Rotation", "Layer"]

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        notes: list[str] = []
        if ir.pcb is not None:
            library = _library(ctx.tools)
            comps = {c.ref: c for c in ir.components}
            for p in sorted(ir.pcb.placements, key=lambda p: p.component_ref):
                check_finite(p.model_dump(mode="json"), f"placement {p.component_ref}")
                comp = comps.get(p.component_ref)
                if comp is not None and EXCLUDE_FROM_POS_FILES in fabrication_exclusions(library, comp.footprint):
                    notes.append(_excluded_note(p.component_ref, "CPL", EXCLUDE_FROM_POS_FILES, comp.footprint))
                    continue
                x, y, why = part_centre(ir, p, library)
                if why is not None:
                    notes.append(f"{p.component_ref}: Mid X / Mid Y are the footprint anchor, not the part's centre: {why}")
                # the rotation is written exactly as the board writer writes it (fixed decimals, no exponent, never -0),
                # so the reviewer's CPL-vs-board comparison never sees a formatting difference
                gx, gy = gerber_frame(x, y)
                w.writerow([_plain(p.component_ref, "Designator"), _mm(gx), _mm(gy), sexpr.fmt_num(float(p.rotation_deg)), p.side.value.capitalize()])
        return self._write(ir, ctx.workdir / "cpl.csv", buf.getvalue(), notes=notes)
