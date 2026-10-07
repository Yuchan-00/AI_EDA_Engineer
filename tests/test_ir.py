from ai_eda.ir import (
    CircuitIR,
    ProjectMeta,
    ProvenanceKind,
    SourceRef,
    ValidationResult,
    ValidationStatus,
    assumption,
    authoritative,
    llm_generated,
    user_requirement,
)


def test_provenance_helpers():
    assert user_requirement(5).provenance.is_authoritative
    assert authoritative(5, SourceRef(title="ds")).provenance.is_authoritative
    assert not llm_generated("LM7805", model="x").provenance.is_authoritative
    assert llm_generated("LM7805", model="x").provenance.needs_verification
    assert assumption(25, note="room temp").provenance.needs_verification


def test_hash_ignores_validation_and_artifacts(divider_ir: CircuitIR):
    h = divider_ir.content_hash()
    divider_ir.validation.add(ValidationResult(check_id="x", status=ValidationStatus.FAIL))
    assert divider_ir.content_hash() == h


def test_hash_changes_with_design(divider_ir: CircuitIR):
    h = divider_ir.content_hash()
    divider_ir.parameters["v_in"] = user_requirement(9.0, "V")
    assert divider_ir.content_hash() != h


def test_json_roundtrip(divider_ir: CircuitIR, tmp_path):
    p = divider_ir.save(tmp_path / "ir.json")
    loaded = CircuitIR.load(p)
    assert loaded.content_hash() == divider_ir.content_hash()
    assert loaded.component("R1").mpn.provenance.kind == ProvenanceKind.AUTHORITATIVE


def test_net_lookup(divider_ir: CircuitIR):
    from ai_eda.ir import PinRef

    assert divider_ir.net_of(PinRef(component_ref="R2", pin_number="1")).name == "VOUT"
    assert divider_ir.net_of(PinRef(component_ref="R9", pin_number="1")) is None


def test_empty_project():
    ir = CircuitIR(project=ProjectMeta(id="e", name="empty"))
    assert ir.validation.overall() == ValidationStatus.NOT_VERIFIED


def test_loads_checks_like_load_and_load_reads_the_file_once(divider_ir: CircuitIR, tmp_path, monkeypatch):
    """``CircuitIR.loads`` is the one checker (schema version, unknown keys, Infinity); ``load`` reads the bytes exactly once."""
    import json
    from pathlib import Path

    import pytest

    from ai_eda.errors import IRSchemaError

    divider_ir.parameters["v_in"] = user_requirement(12.0, "V")
    p = divider_ir.save(tmp_path / "ir.json")
    raw = p.read_bytes()
    assert b'"value": 12.0' in raw
    assert CircuitIR.loads(raw, source="x").content_hash() == divider_ir.content_hash()
    assert CircuitIR.loads(raw.decode("utf-8")).content_hash() == divider_ir.content_hash()
    with pytest.raises(IRSchemaError, match="here.json: schema_version 'banana'"):
        CircuitIR.loads(raw.replace(b'"schema_version": "0.1"', b'"schema_version": "banana"'), source="here.json")
    with pytest.raises(IRSchemaError, match=r"unknown key.*components\[0\]\.bogus"):
        d = json.loads(raw)
        d["components"][0]["bogus"] = 1
        CircuitIR.loads(json.dumps(d))
    with pytest.raises(IRSchemaError, match="Infinity"):
        CircuitIR.loads(raw.replace(b'"value": 12.0', b'"value": Infinity', 1))
    with pytest.raises(IRSchemaError, match="not an IR object"):
        CircuitIR.loads(b"[]")

    reads: list[str] = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(str(self)), real(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: pytest.fail("load must not read the file as text (a second read)"))
    assert CircuitIR.load(p).content_hash() == divider_ir.content_hash()
    assert reads == [str(p)]


def test_save_is_atomic_and_byte_identical(divider_ir: CircuitIR, tmp_path, monkeypatch):
    """``save`` writes a temporary file beside the target and moves it over: a reader sees the old file or the new one, never a torn one."""
    import os
    import stat
    from pathlib import Path

    import pytest

    p = tmp_path / "ir.json"
    assert divider_ir.save(p) == p
    expected = divider_ir.model_dump_json(indent=2)
    assert p.read_text(encoding="utf-8") == expected  # the same text a plain write gives
    assert [q.name for q in tmp_path.iterdir()] == ["ir.json"]  # no temporary file left behind
    old_bytes = p.read_bytes()
    os.chmod(p, 0o600)

    seen: list[bytes] = []
    real_replace = os.replace

    def replace(src, dst):
        seen.append(Path(dst).read_bytes())  # what a reader gets right before the move: the complete old file
        assert Path(src).read_text(encoding="utf-8") != expected  # the new text really is in the temporary file
        return real_replace(src, dst)

    divider_ir.parameters["v_in"] = user_requirement(9.0, "V")
    monkeypatch.setattr(os, "replace", replace)
    divider_ir.save(p)
    assert seen == [old_bytes]
    assert p.read_text(encoding="utf-8") == divider_ir.model_dump_json(indent=2)
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o600  # an existing file keeps its permission bits
    assert [q.name for q in tmp_path.iterdir()] == ["ir.json"]

    # the move fails: the target is untouched and the temporary file is removed
    after = p.read_bytes()
    divider_ir.parameters["v_in"] = user_requirement(3.0, "V")
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        divider_ir.save(p)
    assert p.read_bytes() == after and [q.name for q in tmp_path.iterdir()] == ["ir.json"]
