"""Component existence check against a synthetic KiCad library and a local fake of the web (no network, no real libraries needed).

What is proven: symbol / footprint are resolved from library files on disk
(FAIL when absent); the datasheet pointer comes from the user, the IR or the
KiCad ``Datasheet`` property in that order and never from a model; the
datasheet is fetched only in an online session, only from a trusted origin,
once per run, and archived by hash; the MPN must be found verbatim in the
archived text (NOT_VERIFIED otherwise, never FAIL); blocked / missing /
refused / offline / tampered / unextractable outcomes are honest
NOT_VERIFIED results with the reason; a catalog row backs SourcingInfo with
the file's hash as its source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.ir import Component, LibraryRef, Pin, PinElectricalType, Provenance, ProvenanceKind, SourceRef, ValidationStatus as S, assumption, llm_generated
from ai_eda.parts import CatalogSource, check_component_existence, datasheet_pointer, examine_component, locate_datasheet
from ai_eda.parts.existence import CHECK_PREFIX, TOOL, TOOL_VERSION
from ai_eda.security import ApprovalGate
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S as SX
from ai_eda.tools.sources import DocumentArchive, NetworkPolicy, sha256_of
from tests.fake_sources import FakeSources
from tests.pdf_fixture import build_pdf

VENDOR_HOST = "www.example-vendor.com"
VR1_URL = f"https://{VENDOR_HOST}/ds/vr1.pdf"
VR1_MPN = "VR1-0603-200V-A"
#: a datasheet-like fixture: family name on page 1, orderable code and characteristics on page 2
VR1_PAGES: list[list[str]] = [
    ["VR1 Series", "200 V thin film chip resistors", "Example Vendor Corp"],
    ["Ordering information", f"Part number: {VR1_MPN}", "Package 0603", "Maximum operating voltage 200 V", "Power rating 0.1 W", "Tolerance +/- 1 %"],
    ["Revision history", "Rev. 3, 2026-01"],
]
CATALOG_CSV = Path(__file__).parent / "data" / "catalog_sample.csv"

_probe = KicadLibrary()
HAS_LIBS = _probe.symbol_file("Regulator_Linear") is not None and _probe.symbol_file("Device") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad libraries not installed")


# --------------------------------------------------------------------------- fixtures shared with the agent tests


def _effects() -> list:
    return SX("effects", SX("font", SX("size", 1.27, 1.27)))


def _prop(key: str, value: str, hide: bool = False) -> list:
    return SX("property", Q(key), Q(value), SX("at", 0, 0, 0), SX("hide", True) if hide else None, _effects())


def _pin(number: str, x: float, angle: int) -> list:
    return SX("pin", "passive", "line", SX("at", x, 0, angle), SX("length", 2.54), SX("name", Q("~"), _effects()), SX("number", Q(number), _effects()))


def _symbol(name: str, datasheet: str, description: str = "test part") -> list:
    return SX(
        "symbol", Q(name), SX("pin_names", SX("offset", 1.016)), SX("exclude_from_sim", False), SX("in_bom", True), SX("on_board", True),
        _prop("Reference", "R"), _prop("Value", name), _prop("Footprint", "Test:FP", True), _prop("Datasheet", datasheet, True), _prop("Description", description, True),
        SX("symbol", Q(f"{name}_0_1"), SX("rectangle", SX("start", -2.54, 1.016), SX("end", 2.54, -1.016), SX("stroke", SX("width", 0.254), SX("type", "default")), SX("fill", SX("type", "none")))),
        SX("symbol", Q(f"{name}_1_1"), _pin("1", -5.08, 0), _pin("2", 5.08, 180)),
        SX("embedded_fonts", False),
    )


def _footprint(name: str) -> list:
    return SX(
        "footprint", Q(name), SX("version", 20260206), SX("generator", Q("pcbnew")), SX("layer", Q("F.Cu")), SX("descr", Q("test fp")), SX("attr", "smd"),
        SX("fp_rect", SX("start", -1, -0.6), SX("end", 1, 0.6), SX("stroke", SX("width", 0.05), SX("type", "solid")), SX("fill", "no"), SX("layer", Q("F.CrtYd"))),
        SX("pad", Q("1"), "smd", "roundrect", SX("at", -0.8, 0), SX("size", 0.8, 0.9), SX("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste")), SX("roundrect_rratio", 0.25)),
        SX("pad", Q("2"), "smd", "roundrect", SX("at", 0.8, 0), SX("size", 0.8, 0.9), SX("layers", Q("F.Cu"), Q("F.Mask"), Q("F.Paste")), SX("roundrect_rratio", 0.25)),
        SX("embedded_fonts", False),
    )


def synthetic_library(root: Path, datasheet_url: str = VR1_URL) -> KicadLibrary:
    """A KiCad library root with symbols ``Test:VR1`` (Datasheet = ``datasheet_url``), ``Test:Derived`` (extends VR1), ``Test:NoDs`` (""),
    ``Test:Tilde`` ("~"), ``Test:Typo`` ("hhttps://..."), ``Test:Quoted`` (leading ``"``) and the footprint ``Test:FP``."""
    lib = SX(
        "kicad_symbol_lib", SX("version", 20251024), SX("generator", Q("kicad_symbol_editor")), SX("generator_version", Q("10.0")),
        _symbol("VR1", datasheet_url, "VR1 series 200 V thin film resistor"),
        SX("symbol", Q("Derived"), SX("extends", Q("VR1")), _prop("Value", "Derived"), SX("embedded_fonts", False)),
        _symbol("NoDs", ""),
        _symbol("Tilde", "~"),
        _symbol("Typo", "hhttps://www.onsemi.com/pub/Collateral/MC7900-D.PDF"),
        _symbol("Quoted", '"https://assets.nexperia.com/documents/data-sheet/PESD5V0L1ULD.pdf'),
    )
    (root / "symbols").mkdir(parents=True, exist_ok=True)
    (root / "symbols" / "Test.kicad_sym").write_text(sexpr.dumps(lib), encoding="utf-8")
    pretty = root / "footprints" / "Test.pretty"
    pretty.mkdir(parents=True, exist_ok=True)
    sexpr.dump_file(_footprint("FP"), pretty / "FP.kicad_mod")
    return KicadLibrary(roots=[root])


def make_part(ref: str = "R1", mpn: str | None = VR1_MPN, symbol: str | None = "VR1", footprint: str | None = "FP",
              datasheet: SourceRef | None = None, *, value: str = "200V") -> Component:
    """An unverified part: assumption-provenance MPN, library references not yet resolved."""
    prov = Provenance(kind=ProvenanceKind.ASSUMPTION, note="fixture pin")
    return Component(
        ref=ref, value=value, description="VR1 series 200 V thin film resistor",
        mpn=assumption(mpn, note="fixture MPN, unverified") if mpn is not None else None,
        datasheet=datasheet,
        pins=[Pin(number="1", name="~", electrical_type=PinElectricalType.PASSIVE, provenance=prov), Pin(number="2", name="~", electrical_type=PinElectricalType.PASSIVE, provenance=prov)],
        symbol=LibraryRef(library="Test", name=symbol) if symbol else None,
        footprint=LibraryRef(library="Test", name=footprint) if footprint else None,
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="fixture"),
    )


def online_policy(**kw) -> NetworkPolicy:
    kw.setdefault("gate", ApprovalGate())
    return NetworkPolicy(approved=True, **kw)


def offline_policy(**kw) -> NetworkPolicy:
    kw.setdefault("gate", ApprovalGate())
    return NetworkPolicy(approved=False, **kw)


def make_archive(root: Path, fake: FakeSources, policy: NetworkPolicy) -> DocumentArchive:
    return DocumentArchive(root / "sources", policy, client=fake.client())


@pytest.fixture
def fake():
    with FakeSources() as f:
        yield f


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return synthetic_library(tmp_path / "kicad")


def checks(result) -> dict[str, dict]:
    return {c["name"]: c for c in result.details["checks"]}


# --------------------------------------------------------------------------- the happy path


def test_existing_part_passes_with_evidence(fake, tmp_path, lib):
    data = fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    res = check_component_existence(make_part(), lib, archive)
    assert res.status is S.PASS, res.message
    assert res.check_id == f"{CHECK_PREFIX}R1" and res.tool == TOOL and res.tool_version == TOOL_VERSION
    assert res.artifact_hash == sha256_of(data)
    c = checks(res)
    assert {n: c[n]["status"] for n in c} == {"symbol": "PASS", "footprint": "PASS", "datasheet_pointer": "PASS", "datasheet_archived": "PASS", "mpn_in_datasheet": "PASS", "catalog": "NOT_APPLICABLE"}
    assert c["symbol"]["details"]["library_path"].endswith("Test.kicad_sym") and c["symbol"]["details"]["ir_verified_flag"] is False  # reported, not repaired
    assert c["datasheet_pointer"]["details"]["origin"] == "kicad_symbol" and c["datasheet_pointer"]["details"]["url"] == VR1_URL
    assert c["datasheet_pointer"]["details"]["trust_reason"].startswith("KiCad Datasheet field of Test:VR1")
    assert c["datasheet_archived"]["details"]["source"] == "fetch" and c["datasheet_archived"]["details"]["trusted_by"].startswith("KiCad Datasheet field")
    assert c["mpn_in_datasheet"]["details"]["page"] == 2 and c["mpn_in_datasheet"]["details"]["matched"] == VR1_MPN and "[VR1-0603-200V-A]" in c["mpn_in_datasheet"]["details"]["context"]
    [ev] = res.evidence
    assert ev.content_hash == sha256_of(data) and Path(ev.path).read_bytes() == data and ev.url == VR1_URL
    assert res.details["datasheet"]["archived"]["sha256"] == sha256_of(data) and res.details["datasheet"]["archived"]["pages"] == 3
    assert [r.host for r in fake.requests] == [VENDOR_HOST] and fake.client_urls == [VR1_URL]
    assert archive.policy.trust_reasons["example-vendor.com"].startswith("KiCad Datasheet field of Test:VR1")
    assert "symbol PASS, footprint PASS, datasheet_pointer PASS, datasheet_archived PASS, mpn_in_datasheet PASS, catalog NOT_APPLICABLE" == res.message


def test_report_carries_the_document_and_the_hit(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    report = examine_component(make_part(), lib, make_archive(tmp_path, fake, online_policy()))
    assert report.status is S.PASS and report.document is not None and report.document.page_count == 3
    assert report.mpn_hit.page == 2 and report.mpn_hit.section == "page 2" and report.fetch.ok
    assert report.pointer.origin == "kicad_symbol" and report.catalog_row is None and report.sourcing is None
    assert report.check("mpn_in_datasheet").status is S.PASS and report.check("nope") is None


def test_mpn_case_and_whitespace_insensitive_but_token_bounded(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    lower = check_component_existence(make_part(mpn="vr1-0603-200v-a"), lib, archive)
    assert lower.status is S.PASS and checks(lower)["mpn_in_datasheet"]["details"]["matched"] == VR1_MPN
    spaced = check_component_existence(make_part(ref="R2", mpn="VR1-0603-200V- A"), lib, archive)
    assert spaced.status is S.PASS
    family = check_component_existence(make_part(ref="R3", mpn="VR1-0603-200"), lib, archive)  # continued by a letter in the text: not a hit
    assert family.status is S.NOT_VERIFIED and "MPN 'VR1-0603-200' not found in datasheet" in family.message
    # a part number is a whole identifier: a hyphen-delimited prefix of the orderable code is NOT a hit (it names a
    # family or another code, not this part; 2026-09-23 review finding)
    prefix = check_component_existence(make_part(ref="R4", mpn="VR1-0603"), lib, archive)
    assert prefix.status is S.NOT_VERIFIED and "MPN 'VR1-0603' not found in datasheet" in prefix.message
    assert fake.hits(VR1_URL) == 1  # one attempt per URL per run, however many parts share the datasheet


# --------------------------------------------------------------------------- honest NOT_VERIFIED outcomes


def test_mpn_absent_is_not_verified_never_fail(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    res = check_component_existence(make_part(mpn="VR1-0603-999V-Z"), lib, make_archive(tmp_path, fake, online_policy()))
    assert res.status is S.NOT_VERIFIED
    c = checks(res)
    assert c["datasheet_archived"]["status"] == "PASS" and c["mpn_in_datasheet"]["status"] == "NOT_VERIFIED"
    assert "mpn_in_datasheet: MPN 'VR1-0603-999V-Z' not found in datasheet (3 page(s) searched" in res.message
    assert res.artifact_hash is not None and res.evidence  # the datasheet that was searched is still the evidence


def test_no_mpn_in_the_ir(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    res = check_component_existence(make_part(mpn=None), lib, make_archive(tmp_path, fake, online_policy()))
    assert res.status is S.NOT_VERIFIED and "no MPN in the IR" in res.message and checks(res)["datasheet_archived"]["status"] == "PASS"


@pytest.mark.parametrize("kind,word", [("cloudflare", "blocked"), ("access_denied", "blocked"), ("js_shell", "blocked"), ("missing", "missing")])
def test_blocked_and_missing_pages_are_not_verified_with_the_reason(fake, tmp_path, lib, kind, word):
    if kind == "missing":
        fake.add_missing(VR1_URL)
    else:
        fake.add_interstitial(VR1_URL, kind)
    res = check_component_existence(make_part(), lib, make_archive(tmp_path, fake, online_policy()))
    assert res.status is S.NOT_VERIFIED
    c = checks(res)
    assert c["datasheet_archived"]["status"] == "NOT_VERIFIED" and c["datasheet_archived"]["details"]["fetch_status"] == word
    assert f"not archived ({word})" in c["datasheet_archived"]["message"]
    assert c["mpn_in_datasheet"]["message"] == "datasheet not archived: MPN not checked"
    assert res.artifact_hash is None and res.evidence == []
    assert res.details["datasheet"]["fetch"]["status"] == word and res.details["datasheet"]["fetch"]["body_sha256"]


def test_offline_session_fetches_nothing(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, offline_policy())
    res = check_component_existence(make_part(), lib, archive)
    assert res.status is S.NOT_VERIFIED
    assert "not fetched (offline" in checks(res)["datasheet_archived"]["message"]
    assert checks(res)["datasheet_pointer"]["status"] == "PASS"  # the pointer is known, the document is not
    assert fake.requests == [] and archive.attempts == {}
    assert check_component_existence(make_part(), lib, archive, fetch=False).status is S.NOT_VERIFIED


def test_without_an_archive_nothing_is_verified(fake, tmp_path, lib):
    res = check_component_existence(make_part(), lib)
    assert res.status is S.NOT_VERIFIED and "no document archive" in checks(res)["datasheet_archived"]["message"]
    assert checks(res)["symbol"]["status"] == "PASS"


def test_unresolved_footprint_or_symbol_fails(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    res = check_component_existence(make_part(footprint="Nope"), lib, archive)
    assert res.status is S.FAIL
    assert checks(res)["footprint"]["status"] == "FAIL" and "footprint Test:Nope does not exist in the KiCad libraries" in res.message
    assert checks(res)["mpn_in_datasheet"]["status"] == "PASS"  # the other checks still ran and are reported
    sym = check_component_existence(make_part(ref="R2", symbol="Ghost"), lib, archive)
    assert sym.status is S.FAIL and checks(sym)["symbol"]["status"] == "FAIL"
    assert checks(sym)["datasheet_pointer"]["status"] == "NOT_VERIFIED" and "symbol Test:Ghost not found" in checks(sym)["datasheet_pointer"]["message"]
    none = check_component_existence(make_part(ref="R3", symbol=None, footprint=None), lib, archive)
    assert none.status is S.NOT_VERIFIED and checks(none)["symbol"]["message"] == "no symbol reference in the IR"
    assert check_component_existence(make_part(ref="R4"), None, archive).status is S.NOT_VERIFIED  # no library: nothing resolved, nothing failed


def test_no_datasheet_pointer_and_unusable_pointers(fake, tmp_path, lib):
    archive = make_archive(tmp_path, fake, online_policy())
    for name in ("NoDs", "Tilde"):
        res = check_component_existence(make_part(symbol=name), lib, archive)
        assert res.status is S.NOT_VERIFIED, name
        assert "no datasheet pointer" in checks(res)["datasheet_pointer"]["message"] and "empty Datasheet property" in checks(res)["datasheet_pointer"]["message"]
        assert res.details["datasheet"]["pointer"] is None
    typo = check_component_existence(make_part(symbol="Typo"), lib, archive)
    assert typo.status is S.NOT_VERIFIED
    assert checks(typo)["datasheet_pointer"]["details"]["url_error"].startswith("unsupported scheme 'hhttps'")
    assert "URL unusable" in checks(typo)["datasheet_archived"]["message"]
    assert fake.requests == []
    quoted = locate_datasheet(make_part(symbol="Quoted"), lib).pointer
    assert quoted.url == "https://assets.nexperia.com/documents/data-sheet/PESD5V0L1ULD.pdf" and quoted.host == "assets.nexperia.com"


def test_pointer_priority_user_then_ir_then_kicad(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    user_url = "https://mirror.example.org/vr1-copy.pdf"
    fake.add_pdf(user_url, VR1_PAGES)
    ir_url = "https://cdn.other-host.example/ds/vr1.pdf"
    fake.add_pdf(ir_url, VR1_PAGES)
    # IR reference on a host nobody trusts: refused before any socket use
    part = make_part(datasheet=SourceRef(title="vendor PDF", url=ir_url))
    assert locate_datasheet(part, lib).pointer.origin == "ir"
    archive = make_archive(tmp_path, fake, online_policy())
    res = check_component_existence(part, lib, archive)
    assert res.status is S.NOT_VERIFIED and "not archived (refused)" in checks(res)["datasheet_archived"]["message"]
    assert fake.requests == []
    # IR reference on the KiCad datasheet's own host: trusted by rule (a)
    same_host = make_part(ref="R2", datasheet=SourceRef(title="vendor PDF", url=f"https://{VENDOR_HOST}/ds/vr1-rev3.pdf"))
    fake.add_pdf(f"https://{VENDOR_HOST}/ds/vr1-rev3.pdf", VR1_PAGES)
    assert locate_datasheet(same_host, lib).pointer.trust_reason.startswith("KiCad Datasheet field")
    assert check_component_existence(same_host, lib, archive).status is S.PASS
    # the user's URL wins over both and is trusted exactly (rule (c))
    policy = online_policy(user_urls={"R3": user_url}, gate=ApprovalGate())
    archive3 = make_archive(tmp_path / "u", fake, policy)
    res3 = check_component_existence(make_part(ref="R3", datasheet=SourceRef(title="x", url=ir_url)), lib, archive3, user_urls=policy.user_urls)
    assert res3.status is S.PASS and checks(res3)["datasheet_pointer"]["details"]["origin"] == "user"
    assert res3.details["datasheet"]["archived"]["url"] == user_url
    # an IR reference that is only a title is not a pointer: the KiCad property is used
    titled = make_part(ref="R4", datasheet=SourceRef(title="just a title"))
    search = locate_datasheet(titled, lib)
    assert search.pointer.origin == "kicad_symbol" and any("not a pointer" in n for n in search.notes)


def test_tampered_archived_copy_is_not_verified(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    archive = make_archive(tmp_path, fake, online_policy())
    report = examine_component(make_part(), lib, archive)
    doc = report.document
    ref = doc.source_ref(title="VR1 datasheet")
    part = make_part(datasheet=ref)
    again = make_archive(tmp_path, fake, online_policy(gate=ApprovalGate()))  # a new run: the IR reference names the archived copy
    ok = check_component_existence(part, lib, again)
    assert ok.status is S.PASS and checks(ok)["datasheet_archived"]["details"]["source"] == "ir_reference" and fake.hits(VR1_URL) == 1
    doc.path.write_bytes(build_pdf([["VR1 Series", f"Part number: {VR1_MPN}", "Maximum operating voltage 250 V"]]))  # an edited datasheet
    res = check_component_existence(part, lib, again)
    assert res.status is S.NOT_VERIFIED and "tampered" in checks(res)["datasheet_archived"]["message"]
    assert checks(res)["mpn_in_datasheet"]["status"] == "NOT_VERIFIED" and res.artifact_hash is None
    assert fake.hits(VR1_URL) == 1  # a tampered copy is never silently replaced by a re-fetch


def test_unextractable_pdf_is_archived_but_the_mpn_is_not_checked(fake, tmp_path, lib):
    fake.serve(VR1_URL, b"%PDF-1.7\n" + bytes(range(256)) * 8 + b"\n%%EOF\n", "application/pdf")
    res = check_component_existence(make_part(), lib, make_archive(tmp_path, fake, online_policy()))
    assert res.status is S.NOT_VERIFIED
    assert checks(res)["datasheet_archived"]["status"] == "PASS"
    assert "datasheet text not extractable" in checks(res)["mpn_in_datasheet"]["message"]
    assert res.details["datasheet"]["archived"]["text_available"] is False


# --------------------------------------------------------------------------- catalog


def test_catalog_hit_backs_sourcing_and_miss_is_not_verified(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    catalog = CatalogSource.load(CATALOG_CSV, "2026-09-23", "JLCPCB export", supplier="JLCPCB")
    archive = make_archive(tmp_path, fake, online_policy())
    report = examine_component(make_part(), lib, archive, catalog)
    assert report.status is S.PASS and report.catalog_row.line == 5 and report.catalog_row.supplier_part_number == "C7171"
    s = report.sourcing
    assert s.supplier == "JLCPCB" and s.supplier_part_number.value == "C7171" and s.stock.value == 120 and s.unit_price.value == 0.045 and s.unit_price.unit == "USD"
    assert s.assembly_class.value == "Extended"
    for t in (s.supplier_part_number, s.stock, s.unit_price, s.assembly_class):
        assert t.provenance.kind is ProvenanceKind.AUTHORITATIVE
        assert t.provenance.source.content_hash == catalog.sha256 == sha256_of(CATALOG_CSV.read_bytes())
        assert t.provenance.source.document_path == str(CATALOG_CSV) and t.provenance.source.section == "line 5"
    res = report.result(catalog)
    assert res.status is S.PASS and checks(res)["catalog"]["status"] == "PASS"
    assert [e.content_hash for e in res.evidence] == [report.document.sha256, catalog.sha256]
    assert res.details["catalog"]["row"]["mpn"] == VR1_MPN and res.details["catalog"]["sha256"] == catalog.sha256
    miss = check_component_existence(make_part(ref="R2", mpn="VR1-0603-999V-Z"), lib, archive, catalog)
    assert miss.status is S.NOT_VERIFIED and "not in catalog" in checks(miss)["catalog"]["message"]
    assert [e.description for e in miss.evidence] == ["archived datasheet of R2"]


# --------------------------------------------------------------------------- pointers


def test_datasheet_pointer_api_and_derived_symbols(lib):
    ref = datasheet_pointer(make_part(), lib)
    assert ref is not None and ref.url == VR1_URL and ref.authority == "KiCad symbol library Test" and ref.content_hash is None
    assert datasheet_pointer(make_part(symbol="Derived"), lib).url == VR1_URL  # extends: inherits the parent's property
    assert datasheet_pointer(make_part(symbol="NoDs"), lib) is None and datasheet_pointer(make_part(symbol="Tilde"), lib) is None
    assert datasheet_pointer(make_part(symbol="Ghost"), lib) is None
    assert datasheet_pointer(make_part(), None) is None
    own = SourceRef(title="mine", content_hash="sha256:" + "0" * 64)
    assert datasheet_pointer(make_part(datasheet=own), lib) is own
    p = locate_datasheet(make_part(symbol="Typo"), lib).pointer
    assert p is not None and not p.fetchable and p.url_error and p.ref.url.startswith("hhttps://")
    http = synthetic_library(Path(lib.roots[0]) / "http", "http://www.example-vendor.com/ds/vr1.pdf")
    q = locate_datasheet(make_part(), http).pointer
    assert q.url == VR1_URL and q.url_note == "http upgraded to https (plain http is never used)"


@needs_libs
def test_real_kicad_symbols_carry_datasheet_urls():
    real = KicadLibrary()
    lm = make_part(symbol="LM7805_TO220")
    lm.symbol = LibraryRef(library="Regulator_Linear", name="LM7805_TO220")
    p = locate_datasheet(lm, real).pointer
    assert p is not None and p.origin == "kicad_symbol" and p.url == "https://www.onsemi.cn/PowerSolutions/document/MC7800-D.PDF" and p.host == "onsemi.cn"
    ap = make_part(symbol="AP2127K-3.3")
    ap.symbol = LibraryRef(library="Regulator_Linear", name="AP2127K-3.3")
    assert locate_datasheet(ap, real).pointer.url == "https://www.diodes.com/assets/Datasheets/AP2127.pdf"  # derived symbol, parent's URL
    r = make_part(symbol="R")
    r.symbol = LibraryRef(library="Device", name="R")
    assert datasheet_pointer(r, real) is None  # generic passives carry no datasheet


def test_llm_generated_mpn_is_checked_like_any_other(fake, tmp_path, lib):
    fake.add_pdf(VR1_URL, VR1_PAGES)
    part = make_part()
    part.mpn = llm_generated(VR1_MPN, "some/model", note="proposed")
    res = check_component_existence(part, lib, make_archive(tmp_path, fake, online_policy()))
    assert res.status is S.PASS and part.mpn.provenance.kind is ProvenanceKind.LLM_GENERATED  # the check reports; it never mutates the IR
