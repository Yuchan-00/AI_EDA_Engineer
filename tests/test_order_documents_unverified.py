"""Order documents state an unverified part identity without putting it where a part number goes.

* The BOM leaves an identity / sourcing / evidence cell that nothing verified empty and keeps its status in a column of its
  own (``Unverified``: column and why): the 9 V piano's rev 3 BOM wrote the literal ``NOT_VERIFIED`` into Manufacturer and
  MPN for its 48 drawer resistors and the speaker terminal, and into Supplier / SupplierPN / DatasheetHash for every row - a
  distributor's or an assembler's BOM upload reads that text as a part number.
* A datasheet link the IR keeps from the library symbol for a part whose stated MPN it is not shown to belong to (LM324N
  with the library's lm2902-n.pdf, an onsemi BC33725BU with Diotec's bc337.pdf) is left out of the schematic and the board
  by :func:`ai_eda.compilers.schematic.datasheet_field`; the parts report printed it as the part's datasheet ("IR 기재")
  and the component stage's note named it without saying whose document it is.

Runs without KiCad.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace

from ai_eda.agents.component import ComponentAgent
from ai_eda.compilers import BOMCompiler, CompileContext
from ai_eda.compilers.bom import NOT_VERIFIED, NOTHING_UNVERIFIED, UNVERIFIED_COLUMN, unverified_columns
from ai_eda.compilers.schematic import LIBRARY_AUTHORITY, datasheet_field
from ai_eda.gui.preview import bom_table
from ai_eda.ir import CircuitIR, Provenance, ProvenanceKind, ProjectMeta, SourceRef, SourcingInfo, Traced, authoritative
from ai_eda.parts.pointers import DatasheetPointer
from ai_eda.report.stages import parts_report
from tests.conftest import make_component

LIB_URL = "http://www.ti.com/lit/ds/symlink/lm2902-n.pdf"
#: what the parts pointer records for a symbol's own Datasheet property (ai_eda.parts.pointers)
LIB_DS = SourceRef(title="Datasheet of Amplifier_Operational:LM324 (KiCad Datasheet property)", url=LIB_URL,
                   authority=f"{LIBRARY_AUTHORITY} Amplifier_Operational")
ARCHIVED = SourceRef(title="LM324 datasheet", authority="Texas Instruments", url="https://www.ti.com/lit/ds/symlink/lm324.pdf",
                     content_hash="sha256:" + "ab" * 32, document_path="sources/lm324.pdf", section="page 35")
IDENTITY_AND_SOURCING = ("Manufacturer", "MPN", "Package", "Footprint", "Supplier", "SupplierPN", "DatasheetHash")


def _user(value: str) -> Traced:
    return Traced(value=value, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="design file sha256:00"))


def _ir(tmp_path: Path, *components) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="t", name="t", workdir=str(tmp_path)))
    ir.components = list(components)
    return ir


def _bom(tmp_path: Path, *components) -> dict[str, dict[str, str]]:
    art = BOMCompiler().compile(_ir(tmp_path, *components), CompileContext(workdir=tmp_path, tools={}))
    return {row["Reference"]: row for row in csv.DictReader(io.StringIO(Path(art.path).read_text(encoding="utf-8")))}


def _drawer_resistor():
    """A drawer resistor: no maker, no MPN, no package, no supplier (rev 3's R1 .. R48)."""
    r = make_component("R1", "56R")
    r.mpn = r.package = r.manufacturer = None
    return r


def _bought_opamp():
    """A bought part typed in the design file: maker and MPN are the user's word, nothing grounds them (rev 3's U1)."""
    u = make_component("U1", "LM324N")
    u.manufacturer, u.mpn, u.package = _user("Texas Instruments"), _user("LM324N"), None
    return u


def _grounded():
    """Every identity, sourcing and evidence cell verified."""
    c = make_component("C1", "100n")
    c.manufacturer = authoritative("WIMA", ARCHIVED)
    c.mpn = authoritative("MKS2C031001A00JSSD", ARCHIVED)
    c.package = authoritative("THT", ARCHIVED)
    c.sourcing = [SourcingInfo(supplier="Catalog", supplier_part_number=authoritative("C1", ARCHIVED))]
    return c


def test_bom_leaves_an_unverified_identity_empty_and_names_it_in_its_own_column(tmp_path: Path):
    rows = _bom(tmp_path, _drawer_resistor(), _bought_opamp(), _grounded())
    for row in rows.values():
        # no cell an upload reads as a part number carries the status text
        assert all(row[c] != NOT_VERIFIED for c in IDENTITY_AND_SOURCING), row
    r1 = rows["R1"]
    assert (r1["Manufacturer"], r1["MPN"], r1["Package"], r1["Supplier"], r1["SupplierPN"], r1["DatasheetHash"]) == ("",) * 6
    assert r1["Footprint"] == "Resistor_SMD:R_0603_1608Metric"
    assert r1[UNVERIFIED_COLUMN] == ("Manufacturer (not stated); MPN (not stated); Package (not stated); Supplier (not stated); "
                                     "SupplierPN (not stated); DatasheetHash (no MPN)")
    u1 = rows["U1"]
    # the user's word prints, and that no datasheet grounds the MPN stays said - in the status column, not in the hash cell
    assert (u1["Manufacturer"], u1["MPN"], u1["DatasheetHash"]) == ("Texas Instruments", "LM324N", "")
    assert u1[UNVERIFIED_COLUMN] == ("Package (not stated); Supplier (not stated); SupplierPN (not stated); "
                                     "DatasheetHash (MPN not found in an archived datasheet)")
    c1 = rows["C1"]
    assert c1["DatasheetHash"] == ARCHIVED.content_hash and c1["SupplierPN"] == "C1" and c1[UNVERIFIED_COLUMN] == NOTHING_UNVERIFIED
    assert list(rows["R1"])[-2:] == ["Qty", UNVERIFIED_COLUMN]  # the assembler's columns keep their place; the status is appended


def test_a_proposal_or_an_unlisted_footprint_is_empty_with_its_provenance_named(tmp_path: Path):
    r = make_component("R2", "10k", verified_lib=False)
    r.manufacturer = Traced(value="Yageo", provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED, note="model"))
    row = _bom(tmp_path, r)["R2"]
    assert (row["Manufacturer"], row["Footprint"]) == ("", "")
    assert "Manufacturer (llm_generated)" in row[UNVERIFIED_COLUMN]
    assert "Footprint (footprint not found in the KiCad library)" in row[UNVERIFIED_COLUMN]


def test_the_gui_reads_the_status_column_and_a_02_file_alike(tmp_path: Path):
    _bom(tmp_path, _drawer_resistor(), _grounded())
    table = bom_table(tmp_path)
    by_ref = {row["Reference"]: cols for row, cols in zip(table["rows"], table["not_verified"])}
    assert by_ref == {"C1": [], "R1": ["Manufacturer", "MPN", "Package", "Supplier", "SupplierPN", "DatasheetHash"]}
    # the empty cells a 0.3 row names are marked (the page draws them as NOT_VERIFIED), whatever else the row holds
    (tmp_path / "bom.csv").write_text("Reference,MPN,Supplier,Qty,Unverified\nJ2,,,1,MPN (not stated); Supplier (not stated)\n", encoding="utf-8")
    assert bom_table(tmp_path)["not_verified"] == [["MPN", "Supplier"]]
    # a compiler.bom 0.2 file: no status column, the status in the cell
    old = {"Reference": "R1", "MPN": NOT_VERIFIED, "Footprint": "Device:R", "DatasheetHash": NOT_VERIFIED}
    assert unverified_columns(old) == ["MPN", "DatasheetHash"]


def test_parts_report_does_not_print_the_library_link_as_the_parts_datasheet():
    u1 = _bought_opamp()
    u1.datasheet = LIB_DS
    assert datasheet_field(u1) == ""  # the schematic's rule, unchanged
    text = parts_report(_ir(Path("."), u1))
    assert f"- 데이터시트: {LIB_DS.title}, {LIB_URL}" not in text
    line = next(x for x in text.splitlines() if x.startswith("- 데이터시트:"))
    assert line.startswith("- 데이터시트: 없음 - ") and "KiCad 라이브러리 심볼에 적힌 링크이며 MPN `LM324N` 의 문서인지 확인되지 않았습니다" in line
    assert "회로도·기판의 Datasheet 칸은 비어 있습니다" in line
    # nothing checked what the linked document is: the report states only that it is the symbol's link, never a classification
    # (an Infineon irf4905.pdf a symbol links may well be the stated IRF4905PbF's own document)
    assert "계열" not in text


def test_library_facts_datasheet_line_does_not_classify_the_symbols_link():
    from ai_eda.report.stages import _library_facts

    sym = SimpleNamespace(properties={"Description": "Quad op-amp", "Datasheet": LIB_URL}, extends_from=None, pins=[])
    library = SimpleNamespace(load_symbol=lambda ref: sym)
    u1 = _bought_opamp()
    line = next(x for x in _library_facts(u1, library) if x.startswith("- Datasheet (KiCad 라이브러리 기재):"))
    assert line == f"- Datasheet (KiCad 라이브러리 기재): {LIB_URL} - 심볼에 적힌 링크이며 MPN `LM324N` 의 문서인지 확인되지 않았습니다"
    # no stated MPN: the symbol's link is printed as the library states it
    r = _drawer_resistor()
    line = next(x for x in _library_facts(r, library) if x.startswith("- Datasheet (KiCad 라이브러리 기재):"))
    assert line == f"- Datasheet (KiCad 라이브러리 기재): {LIB_URL}"


def test_parts_report_keeps_a_link_that_belongs_and_names_the_mpns_own_document():
    # no stated MPN: the IR's link is the part's link as before (datasheet_field keeps it)
    r = _drawer_resistor()
    r.datasheet = LIB_DS
    assert f"- 데이터시트: {LIB_DS.title}, {LIB_URL}, 출처 {LIB_DS.authority}" in parts_report(_ir(Path("."), r))
    # an MPN found in an archived document with a URL: that document is the part's datasheet, the library link is named as such
    u = make_component("U1", "LM324N")
    u.mpn, u.datasheet = authoritative("LM324N", ARCHIVED), LIB_DS
    assert datasheet_field(u) == ARCHIVED.url
    lines = [x for x in parts_report(_ir(Path("."), u)).splitlines() if x.startswith(("- 데이터시트:", "- IR에 남은"))]
    assert lines[0] == f"- 데이터시트: {ARCHIVED.url} (MPN `LM324N` 을 찾은 문서; 회로도·기판의 Datasheet 칸)"
    assert "KiCad 라이브러리 심볼에 적힌 링크이며 MPN `LM324N` 의 문서인지 확인되지 않았습니다" in lines[1]
    assert "계열" not in lines[1]


def test_component_stage_note_says_a_library_link_is_not_the_stated_parts_document():
    pointer = DatasheetPointer(ref=LIB_DS, origin="kicad_symbol", note="Datasheet property of KiCad symbol Amplifier_Operational:LM324", url=LIB_URL)
    report = SimpleNamespace(document=None, pointer=pointer, mpn_hit=None, sourcing=None)
    notes: list[str] = []
    c = _bought_opamp()
    c.datasheet = None
    ComponentAgent._apply_report(c, report, notes)
    assert notes == [f"U1: datasheet pointer recorded from kicad_symbol ({LIB_URL}); not archived; the library symbol's own link, "
                     "not checked for MPN 'LM324N' (left out of the schematic and the board)"]
    # a part without a stated MPN: the link is its link, the note as before
    notes = []
    r = _drawer_resistor()
    ComponentAgent._apply_report(r, report, notes)
    assert notes == [f"R1: datasheet pointer recorded from kicad_symbol ({LIB_URL}); not archived"]
