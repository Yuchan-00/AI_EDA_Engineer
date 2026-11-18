"""Action grouping copies stored results, never manufactures verdicts.

The portable fixture mirrors the saved STM32 run's 54 NOT_VERIFIED causes;
it needs neither that local project nor any EDA tool installed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ai_eda.ir import ValidationResult, ValidationStatus as S
from ai_eda.ir.validation import ValidationState
from ai_eda.report.actions import group_actions


def _result(check: str, message: str = "", *, status: S = S.NOT_VERIFIED, details: dict | None = None) -> ValidationResult:
    return ValidationResult(check_id=check, status=status, message=message, details=details or {}, tool="fixture",
                            timestamp="2026-01-01T00:00:00Z")


def saved_run_results() -> list[ValidationResult]:
    """37 parts plus the saved run's 17 other NOT_VERIFIED results (emitted reason shapes)."""
    refs = [f"C{i}" for i in range(1, 16)] + ["D1"] + [f"J{i}" for i in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11)]
    refs += ["L1"] + [f"R{i}" for i in range(1, 7)] + ["SW1", "U1", "U2", "Y1"]
    results = []
    for ref in refs:
        name, reason = ("datasheet_archived", "not fetched (offline: no online session was opened)") if ref in ("U1", "U2") else (
            "datasheet_pointer", "no datasheet pointer: IR has no datasheet pointer and the KiCad symbol has an empty Datasheet property")
        checks = [{"name": name, "status": "NOT_VERIFIED", "message": reason},
                  {"name": "mpn_in_datasheet", "status": "NOT_VERIFIED", "message": "no MPN in the IR: nothing to look for"}]
        results.append(_result(f"component.existence.{ref}", f"{name} NOT_VERIFIED, mpn_in_datasheet NOT_VERIFIED; {name}: {reason}; mpn_in_datasheet: no MPN in the IR: nothing to look for",
                               details={"ref": ref, "mpn": None, "checks": checks}))
    compared = [{"limit": "min_track_width_mm", "status": "NOT_VERIFIED",
                 "message": "no min_track_width_mm in ir.pcb.manufacturing: track widths not judged", "limit_value": None, "grounded": False}]
    results += [
        _result("mfg.capability", "0 limit(s) met in the IR, 7 not proven", details={"compared": compared}),
        _result("review.manufacturing_capabilities", "0 limit(s) met in the IR, 7 not proven", details={"live": {"compared": compared}}),
        _result("pcb.routing.clearance", "no clearance limit in ir.pcb.manufacturing; the router's own clearance is a parameter, not a rule"),
        _result("pcb.silk.size", "no min_silk_text_height_mm in ir.pcb.manufacturing (ground it with --fab-capability): not judged"),
        _result("domain.power.pi", "power integrity backend not implemented"),
        _result("domain.power.thermal", "no component carries theta_ja / t_j_max (datasheet facts: theta_ja in K/W, t_j_max in degC)"),
        _result("ir.component_provenance", "component identity must be grounded: J1: no MPN: nobody can source this part"),
        _result("review.component_provenance", "component identity not fully verified: J1: no MPN: nobody can source this part"),
        _result("regulatory.applicability", "9 of 10 candidate(s) undecided - answer mains_powered, radio", details={"missing_keys": ["mains_powered", "radio"]}),
        _result("calc.recompute", "no derived parameters to recompute"),
        _result("review.calculations_vs_design", "recompute: no derived parameters to recompute"),
        _result("component.fit", "37 component(s) at the operating point (op): J1 NOT_VERIFIED (excluded from SPICE)"),
        _result("pcb.routing.connectivity", "2 net(s) joined only by a copper pour, which this check cannot judge"),
        _result("regulatory.compliance", "compliance is not assessed by this system"),
        _result("regulatory.research", "sources NOT_VERIFIED; applicability NOT_VERIFIED; compliance NOT_VERIFIED (never assessed here)"),
        _result("regulatory.sources", "10 candidate(s) for KR: 0 archived and grounded; not fetched (offline)"),
        _result("review.regulatory_provenance", "10 requirement(s) lack provenance fields; applicability undecided"),
    ]
    return results


def test_many_results_are_lossless_and_grouped_by_input():
    results = saved_run_results()
    assert len(results) == 54
    before = [r.model_dump_json() for r in results]
    groups = group_actions(results)
    checks = [c for g in groups for c in g.checks]
    assert Counter(c.check_id for c in checks) == Counter(r.check_id for r in results)
    assert {g.id: len(g.checks) for g in groups} == {
        "missing_part_identity": 2, "missing_part_documents": 36, "offline_part_documents": 2,
        "missing_fab_inputs": 4, "unsupported_backends": 1, "scope_questions": 1, "unmapped": 8,
    }
    stored = {r.check_id: r for r in results}
    for group in groups:
        assert group.id and group.title and group.next_action
        assert group.role in ("user", "software", "undetermined")
        assert group.check_ids == sorted(c.check_id for c in group.checks)
        for check in group.checks:
            assert (check.status, check.message) == (str(stored[check.check_id].status), stored[check.check_id].message)
    docs = next(g for g in groups if g.id == "missing_part_documents")
    assert {f"component.existence.C{i}" for i in range(1, 16)} <= set(docs.check_ids)
    assert before == [r.model_dump_json() for r in results]
    assert groups == group_actions(iter(results)) == group_actions(reversed(results))


def test_last_occurrence_wins_even_when_timestamps_disagree():
    results = [_result("fixed", "old failure", status=S.FAIL), _result("fixed", "now checked", status=S.PASS),
               _result("changed", "no MPN in the IR: old cause"), _result("changed", "new unknown cause"),
               _result("not_applicable", "retired check", status=S.NOT_APPLICABLE)]
    results[0].timestamp = results[0].timestamp.replace(year=2030)
    groups = group_actions(results)
    checks = [c for g in groups for c in g.checks]
    assert [(c.check_id, c.status, c.message) for c in checks] == [
        ("changed", "NOT_VERIFIED", "new unknown cause")]  # a NOT_APPLICABLE check asks for nothing
    assert groups == group_actions(ValidationState(results=results).latest_by_check().values())


@pytest.mark.parametrize("status", list(S))
def test_every_non_pass_status_is_copied_without_reclassification(status):
    groups = group_actions([_result("unknown", "original reason", status=status)])
    if status in (S.PASS, S.NOT_APPLICABLE):  # nothing is asked of anyone
        assert groups == []
    else:
        assert groups[0].id == "unmapped"
        assert groups[0].checks[0].model_dump() == {"check_id": "unknown", "status": str(status), "message": "original reason"}


@pytest.mark.parametrize("details", [{}, {"checks": None}, {"checks": [None, {}, {"message": None}]}])
def test_different_causes_in_one_family_and_missing_metadata(details):
    results = [_result("component.existence.A", "no MPN in the IR: nothing to look for", details=details),
               _result("component.existence.B", "no datasheet pointer: nothing to archive"),
               _result("component.existence.C", "not fetched (offline: no online session was opened)"),
               _result("component.existence.D", "footprint is unreadable", status=S.FAIL),
               _result("unknown", "power integrity backend not implemented"), _result("empty")]
    groups = {g.id: g.check_ids for g in group_actions(results)}
    assert groups == {"missing_part_identity": ["component.existence.A"], "missing_part_documents": ["component.existence.B"],
                      "offline_part_documents": ["component.existence.C"], "unmapped": ["component.existence.D", "empty", "unknown"]}


def test_summary_message_fallback_and_known_structured_questions():
    rows = [_result("component.existence.U1", "symbol PASS; datasheet_archived: not fetched (offline: no online session was opened)"),
            _result("requirements.crosscheck", "withheld until you answer: optional_key", status=S.USER_INPUT_REQUIRED, details={"open": ["optional_key"]}),
            _result("mfg.capability_source", "no fab capability file (pass --fab-capability FILE to ground the fab limits on the vendor page)"),
            _result("mfg.capability", "backend evidence missing", details={"compared": None}),
            _result("review.manufacturing_capabilities", "unknown", details={"live": None})]
    assert {g.id: g.check_ids for g in group_actions(rows)} == {
        "offline_part_documents": ["component.existence.U1"], "missing_fab_inputs": ["mfg.capability_source"],
        "scope_questions": ["requirements.crosscheck"], "unmapped": ["mfg.capability", "review.manufacturing_capabilities"]}
    assert group_actions([]) == group_actions([_result("pass", status=S.PASS)]) == []
