from pathlib import Path

from ai_eda.ir import (
    CircuitIR,
    Net,
    PinRef,
    Provenance,
    ProvenanceKind,
    ValidationResult,
    ValidationState,
    ValidationStatus,
    llm_generated,
    worst_status,
)
from ai_eda.validation import ValidationContext, default_registry
import ai_eda.validation.structural  # noqa: F401  (registers validators)
import ai_eda.validation.domain  # noqa: F401

S = ValidationStatus


def test_worst_status_ordering():
    assert worst_status([]) == S.NOT_VERIFIED
    assert worst_status([S.PASS, S.PASS]) == S.PASS
    assert worst_status([S.PASS, S.NOT_VERIFIED]) == S.NOT_VERIFIED
    assert worst_status([S.PASS, S.USER_INPUT_REQUIRED, S.NOT_VERIFIED]) == S.USER_INPUT_REQUIRED
    assert worst_status([S.FAIL, S.PASS]) == S.FAIL
    assert worst_status([S.NOT_APPLICABLE]) == S.NOT_APPLICABLE


def test_state_uses_latest_per_check():
    st = ValidationState()
    st.add(ValidationResult(check_id="a", status=S.FAIL))
    st.add(ValidationResult(check_id="a", status=S.PASS))
    assert st.overall() == S.PASS
    assert st.failing() == []


def test_registry_selects_by_domain(divider_ir: CircuitIR):
    ids = {v.id for v in default_registry.select(divider_ir)}
    assert "ir.connectivity" in ids  # domain-independent
    assert "domain.analog.bias" in ids  # ANALOG
    assert "domain.rf.impedance" not in ids


def test_connectivity_validator_detects_bad_pin(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.nets.append(Net(name="BAD", pins=[PinRef(component_ref="R1", pin_number="9")], provenance=Provenance(kind=ProvenanceKind.DERIVED)))
    results = default_registry.get("ir.connectivity").validate(divider_ir, ValidationContext(workdir=tmp_path))
    assert results[0].status == S.FAIL
    assert "no pin 9" in results[0].message


def test_component_provenance_flags_llm_mpn(divider_ir: CircuitIR, tmp_path: Path):
    divider_ir.components[0].mpn = llm_generated("GUESSED-123", model="m")
    results = default_registry.get("ir.component_provenance").validate(divider_ir, ValidationContext(workdir=tmp_path))
    assert results[0].status == S.NOT_VERIFIED
    assert "R1.mpn" in results[0].details["unverified"]


def test_domain_validators_never_pass_without_backend(divider_ir: CircuitIR, tmp_path: Path):
    """No SPICE run attached: every op-reading validator is NOT_VERIFIED and says what it needs (never PASS, never an exception)."""
    for check in ("domain.analog.bias", "domain.power.thermal", "component.fit"):
        for r in default_registry.get(check).validate(divider_ir, ValidationContext(workdir=tmp_path)):
            assert r.status == S.NOT_VERIFIED and r.check_id == check, (check, r.message)
