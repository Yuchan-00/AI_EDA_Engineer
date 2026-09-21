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
