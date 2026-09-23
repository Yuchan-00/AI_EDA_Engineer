"""CatalogSource: alias-driven CSV import, hashed file as the source, MPN lookup, honest parsing of odd cells."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_eda.ir import ProvenanceKind
from ai_eda.parts import CatalogError, CatalogSource, mpn_key
from ai_eda.parts.catalog import HEADER_ALIASES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS, map_headers, parse_price, parse_stock

SAMPLE = Path(__file__).parent / "data" / "catalog_sample.csv"


def test_sample_export_loads_with_community_headers():
    cat = CatalogSource.load(SAMPLE, "2026-09-23", "JLCPCB export", supplier="JLCPCB")
    assert cat.sha256 == "sha256:" + hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert cat.columns == {"mpn": 1, "manufacturer": 2, "package": 3, "supplier_part_number": 0, "stock": 4, "unit_price": 6, "assembly_class": 5, "description": 8, "datasheet_url": 7}
    assert [r.mpn for r in cat.rows] == ["0402WGF4702TCE", "LM2596SX-5.0/NOPB", "RC0603FR-0710KL", "VR1-0603-200V-A", "NOSTOCK-1"]
    assert cat.notes == ["line 7: empty mpn; skipped"]
    first = cat.rows[0]
    assert first.line == 2 and first.package == "0402" and first.supplier_part_number == "C25792" and first.stock == 1000000 and first.unit_price == 0.0011 and first.currency == "USD"
    assert first.assembly_class == "Basic" and first.datasheet_url.startswith("https://datasheet.lcsc.com/") and first.raw["package"] == "0402\t"
    ti = cat.lookup("lm2596sx-5.0/nopb")
    assert ti is not None and ti.unit_price == 1.4321 and ti.notes == ["price taken from the first quantity break 1-499: 1.4321"]
    assert ti.description == '5V 3A step-down regulator, "SIMPLE SWITCHER"'
    yageo = cat.lookup(" RC0603FR-07 10KL ")
    assert yageo is not None and yageo.supplier_part_number == "C25804" and "normalised to C25804" in yageo.notes[0]
    nostock = cat.lookup("NOSTOCK-1")
    assert nostock.stock is None and nostock.unit_price is None and nostock.currency is None
    assert nostock.notes == ["stock 'n/a' is not an integer", "price 'call' not parseable"]
    assert cat.lookup("NOPE") is None and cat.lookup(None) is None and cat.lookup("  ") is None
    assert cat.duplicates == {}
    d = cat.describe()
    assert d["rows"] == 5 and d["sha256"] == cat.sha256 and d["supplier"] == "JLCPCB" and d["retrieved_at"] == "2026-09-23"


def test_sourcing_info_is_authoritative_to_the_file_row():
    cat = CatalogSource.load(SAMPLE, datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc), "LCSC export")
    assert cat.supplier == "LCSC export"  # supplier defaults to the authority
    row = cat.lookup("VR1-0603-200V-A")
    info = cat.sourcing_info(row)
    assert info.supplier == "LCSC export" and info.currency == "USD"
    assert info.supplier_part_number.value == "C7171" and info.stock.value == 120 and info.unit_price.value == 0.045 and info.unit_price.unit == "USD" and info.assembly_class.value == "Extended"
    for t in (info.supplier_part_number, info.stock, info.unit_price, info.assembly_class):
        assert t.provenance.kind is ProvenanceKind.AUTHORITATIVE
        src = t.provenance.source
        assert src.content_hash == cat.sha256 and src.document_path == str(SAMPLE) and src.section == "line 5" and src.authority == "LCSC export"
        assert src.retrieved_at == datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc) and src.title == "LCSC export catalog export catalog_sample.csv"
        assert t.provenance.note.startswith("catalog row 5 for MPN 'VR1-0603-200V-A'")
    nostock = cat.sourcing_info(cat.lookup("NOSTOCK-1"))
    assert nostock.stock is None and nostock.unit_price is None and nostock.supplier_part_number.value == "C99"
    assert "not an integer" in nostock.supplier_part_number.provenance.note


def test_header_aliases_of_the_documented_community_dumps():
    lrks = "code,url,file,library,deleted,lastSeen,brand,model,package,type,describe,erpComponentName,price,stock,MOQ,category,firstSeen,pcbaMinQty,pcbaMinPrice".split(",")
    m, notes = map_headers(lrks)
    assert m["mpn"] == lrks.index("model") and m["manufacturer"] == lrks.index("brand") and m["supplier_part_number"] == lrks.index("code")
    assert m["assembly_class"] == lrks.index("library") and m["datasheet_url"] == lrks.index("file") and m["unit_price"] == lrks.index("price")
    assert any("'url' also means datasheet_url" in n for n in notes) and any("'type' also means assembly_class" in n for n in notes)
    jose = "Manufacturer,LCSC Part #,Description,Package,Stock,Type,MFR.Part #,Datasheet,Price (USD)".split(",")
    mj, _ = map_headers(jose)
    assert mj["assembly_class"] == jose.index("Type") and mj["mpn"] == jose.index("MFR.Part #") and mj["supplier_part_number"] == jose.index("LCSC Part #")
    cdfer = "lcsc,fetched_at,present,sync_seen,category,subcategory,mfr,package,joints,manufacturer,library_type,preferred,last_on_stock,description,datasheet,stock,price,attributes".split(",")
    m2, _ = map_headers(cdfer)
    assert m2["mpn"] == cdfer.index("mfr") and m2["manufacturer"] == cdfer.index("manufacturer") and m2["supplier_part_number"] == cdfer.index("lcsc")
    assert m2["assembly_class"] == cdfer.index("library_type") and m2["stock"] == cdfer.index("stock")
    canonical = ["mpn", "manufacturer", "package", *OPTIONAL_COLUMNS]
    m3, notes3 = map_headers(canonical)
    assert m3 == {c: i for i, c in enumerate(canonical)} and notes3 == []
    assert set(HEADER_ALIASES) == set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
    for canonical_name, aliases in HEADER_ALIASES.items():
        assert canonical_name in aliases


def test_refusals(tmp_path: Path):
    with pytest.raises(CatalogError, match="lacks the required column"):
        (tmp_path / "nomfr.csv").write_text("MPN,Package\nX,0603\n", encoding="utf-8")
        CatalogSource.load(tmp_path / "nomfr.csv", "2026-09-23", "test")
    with pytest.raises(CatalogError, match="ISO 8601"):
        CatalogSource.load(SAMPLE, "yesterday", "test")
    with pytest.raises(CatalogError, match="cannot read"):
        CatalogSource.load(tmp_path / "missing.csv", "2026-09-23", "test")
    (tmp_path / "empty.csv").write_bytes(b"")
    with pytest.raises(CatalogError, match="empty"):
        CatalogSource.load(tmp_path / "empty.csv", "2026-09-23", "test")


def test_cell_parsers_and_keys():
    assert parse_stock("1,200") == (1200, None) and parse_stock("") == (None, None) and parse_stock("many")[0] is None
    assert parse_price("0.5") == (0.5, None) and parse_price("1-499:0.0099,500-1499:0.008")[0] == 0.0099 and parse_price("$1")[0] is None
    assert parse_price("10+:2.5") == (2.5, "price taken from the first quantity break 10-+: 2.5")
    assert mpn_key(" LM2931AZ-5.0 / NOPB ") == "lm2931az-5.0/nopb" and mpn_key("abc") == mpn_key("ABC") != mpn_key("ab-c")


def test_encoding_fallback_and_duplicates(tmp_path: Path):
    p = tmp_path / "cp949.csv"
    p.write_bytes("mpn,manufacturer,package,stock\nX1,삼성전기,0603,5\nx1,Samsung,0603,6\n".encode("cp949"))
    cat = CatalogSource.load(p, "2026-09-23", "test")
    assert cat.encoding == "cp949" and cat.rows[0].manufacturer == "삼성전기"
    assert cat.lookup("X1").line == 2 and cat.duplicates == {"x1": 2}  # the first row wins, the duplicate is counted
    assert cat.sourcing_info(cat.lookup("X1")).currency is None and cat.sourcing_info(cat.lookup("X1")).unit_price is None
