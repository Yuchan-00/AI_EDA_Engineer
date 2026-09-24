"""Fab capability: the limits in the IR and the deterministic comparison of the board against them.

Invariants (``mfg.capability`` is produced only by :func:`check_capability`,
tool ``mfg.capability_check``; it reads the IR, the artifact bookkeeping and
the latest ``kicad.drc`` result, never a session tool):

* The limits are design content: ``ir.pcb.manufacturing`` (recorded there by
  the FAB_CAPABILITY stage from the grounded vendor page, or by the user).
  A limit whose provenance is not ``authoritative`` - an assumption, a model
  value, a derived number, and also a ``user_requirement`` (the user asserted
  it; the vendor page did not) - is compared for a FAIL but its row can only
  be NOT_VERIFIED ("not grounded on the vendor page"), never PASS. The
  provenance is read as the IR states it: this check does not open the
  archive, so a limit whose *value* was edited after grounding is caught by
  the FAB_CAPABILITY agent's re-verification and the reviewer
  (:func:`~ai_eda.tools.manufacturing.capability_file.relocate_limits`,
  ``value_mismatch``), whose verdicts gate the stored ``mfg.capability``.
* IR comparisons are exact (mm, no tolerance) and need no KiCad: a track
  narrower than ``min_track_width_mm``, a via drill / diameter below the
  minimum, a copper layer count outside ``layer_count_options``, a zone
  clearance below ``min_clearance_mm``, an IR via hole closer to the outline
  than ``min_hole_to_edge_mm`` - each a FAIL row naming the items, and the
  result carries ``details["repair"] = "fab_capability_shortfall"`` (a human
  decides; the repair loop never touches copper). A row PASSes only when
  something was compared: with no tracks, no zones carrying a clearance or
  no vias in the IR the row is NOT_APPLICABLE ("no tracks in the IR", ...)
  and is not counted among the "limit(s) met"; the copper layer list is
  always compared.
* Geometry the IR does not state - the clearance *between* items and the
  copper of library footprints - is provable only by DRC run with the fab
  minimums as rules. Those rows PASS only when the latest ``kicad.drc`` is
  tool-backed and PASS, ran on the fresh board artifact (hash, not stale,
  on disk), read the fresh ``.kicad_pro`` artifact (``details["project_hash"]``
  equals the artifact's hash, which the deterministic project compiler wrote:
  no rule severities in it), ignored none of the relevant checks, wrote
  every mapped rule at or above the limit, **and** the running kicad-cli is in
  :data:`~ai_eda.tools.kicad.cli.PROJECT_RULES_MEASURED_VERSIONS` - the
  measured fact that this kicad-cli applies a sibling project's rules. Until
  that measurement is recorded the rows read NOT_VERIFIED and say so; a DRC
  FAIL makes them NOT_VERIFIED too (the violation is already a FAIL under
  ``kicad.drc``; not a second one).
* ``board_thickness_mm`` and ``copper_weight_oz`` are chosen values, not
  minimums; footprint hole-to-edge and annular ring have no DRC rule this
  system writes. They are listed under ``details["not_compared"]`` and named
  in the message, never counted as compared.
* The overall status is the worst row (NOT_APPLICABLE rows do not count);
  without ``ir.pcb`` the result is NOT_VERIFIED. There is no path to PASS
  without DRC evidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ai_eda.ir import ArtifactKind, CircuitIR, ManufacturingConstraints, ProvenanceKind, SourceRef, Traced, ValidationResult, ValidationStatus, worst_status

TOOL_ID = "mfg.capability_check"
TOOL_VERSION = "0.1"
CHECK_ID = "mfg.capability"
REPAIR = "fab_capability_shortfall"

#: DRC check types that prove a fab minimum; a report that ignored one of them proves nothing about it
RELEVANT_DRC_TYPES: frozenset[str] = frozenset({"track_width", "clearance", "via_diameter", "copper_edge_clearance", "hole_size", "drill_out_of_range",
                                                 "hole_clearance", "hole_near_hole", "via_hole_larger_than_pad", "microvia_drill_out_of_range"})
#: what is never compared, and why
NOT_COMPARED: dict[str, str] = {
    "board_thickness_mm": "a chosen value written to the board file, not a minimum to compare",
    "copper_weight_oz": "a chosen value; no DRC rule represents it",
    "footprint_hole_to_edge": "library footprint holes are not in the IR and no DRC rule this system writes proves hole-to-edge (KiCad has copper-to-edge only)",
    "annular_ring": "no DRC rule this system writes constrains the annular width",
}
#: geometry only DRC can prove: row name -> the mapped limits it needs
GEOMETRY_ROWS: dict[str, tuple[str, ...]] = {
    "clearance_between_items": ("min_clearance_mm",),
    "footprint_copper": ("min_track_width_mm", "min_via_diameter_mm", "min_via_drill_mm"),
}


class FabCapability(BaseModel):
    fab: str
    constraints: ManufacturingConstraints
    source: SourceRef | None = None

    @classmethod
    def from_ir(cls, ir: CircuitIR) -> FabCapability:
        """The limits the IR holds (empty constraints when there is no board)."""
        if ir.pcb is None:
            return cls(fab="", constraints=ManufacturingConstraints())
        m = ir.pcb.manufacturing
        source = None
        for key in sorted(ManufacturingConstraints.model_fields):
            traced = getattr(m, key) if key != "fab" else None
            if isinstance(traced, Traced) and traced.provenance.source is not None:
                source = traced.provenance.source
                break
        return cls(fab=m.fab or "", constraints=m, source=source)

    def verification_status(self) -> ValidationStatus:
        traced = [
            v for v in self.constraints.model_dump(exclude={"fab"}).values() if v is not None
        ]
        if not traced:
            return ValidationStatus.NOT_VERIFIED
        kinds = {
            getattr(getattr(self.constraints, k), "provenance").kind
            for k, v in self.constraints.model_dump(exclude={"fab"}).items()
            if v is not None
        }
        return ValidationStatus.PASS if kinds == {ProvenanceKind.AUTHORITATIVE} else ValidationStatus.NOT_VERIFIED


def jlcpcb_capability_unverified() -> FabCapability:
    """Placeholder: no limits filled in. They must be grounded on JLCPCB's official capability page (``--fab-capability``)."""
    return FabCapability(fab="JLCPCB", constraints=ManufacturingConstraints(fab="JLCPCB"))


# --------------------------------------------------------------------------- the check


def _row(limit: str, status: ValidationStatus, message: str, **extra: Any) -> dict[str, Any]:
    return {"limit": limit, "status": str(status), "message": message, **extra}


def _grounded(traced: Traced | None) -> bool:
    return traced is not None and traced.provenance.kind is ProvenanceKind.AUTHORITATIVE


def _compared_row(limit: str, traced: Traced, offenders: list[str], what: str, *, compared: int, none: str, met: str = "at or above", unmet: str = "below") -> dict[str, Any]:
    """FAIL when ``offenders`` is non-empty; NOT_APPLICABLE (``none``) when nothing was compared; else PASS for an authoritative limit and NOT_VERIFIED for one nobody grounded.

    ``met`` / ``unmet`` are the words for the comparison (``at or above`` /
    ``below`` for a minimum; ``in the options`` / ``not in`` for the layer list).
    """
    value = traced.value
    if offenders:
        return _row(limit, ValidationStatus.FAIL, f"{len(offenders)} {what} {unmet} the {limit} limit {value}: {', '.join(offenders[:8])}" + (" ..." if len(offenders) > 8 else ""),
                    limit_value=value, offenders=offenders, grounded=_grounded(traced))
    if compared == 0:
        return _row(limit, ValidationStatus.NOT_APPLICABLE, none, limit_value=value, offenders=[], grounded=_grounded(traced))
    if not _grounded(traced):
        return _row(limit, ValidationStatus.NOT_VERIFIED, f"{what} {met} {value} in the IR, but the limit is not grounded on the vendor page ({traced.provenance.kind})",
                    limit_value=value, offenders=[], grounded=False)
    return _row(limit, ValidationStatus.PASS, f"every {what} {met} {value} in the IR", limit_value=value, offenders=[], grounded=True)


def _missing_row(limit: str, why: str) -> dict[str, Any]:
    return _row(limit, ValidationStatus.NOT_VERIFIED, f"no {limit} in ir.pcb.manufacturing: {why}", limit_value=None, grounded=False)


def _drc_proof(ir: CircuitIR) -> tuple[dict[str, float] | None, str | None, ValidationResult | None]:
    """``(design rules DRC ran with, None, result)`` when the latest kicad.drc proves geometry against them, else ``(None, reason, result)``."""
    from ai_eda.tools.kicad.cli import PROJECT_RULES_MEASURED_VERSIONS

    res = ir.validation.latest("kicad.drc")
    if res is None:
        return None, "kicad.drc has not been run", None
    if not res.is_tool_backed:
        return None, "kicad.drc result is not tool-backed (no tool): an opinion", res
    pcb = ir.artifacts.get(ArtifactKind.PCB)
    if pcb is None:
        return None, "no kicad_pcb artifact", res
    if pcb.is_stale(ir.content_hash()):
        return None, "kicad_pcb artifact was generated from another IR version", res
    if not pcb.matches_disk():
        return None, "kicad_pcb on disk does not match its recorded hash", res
    if not res.artifact_hash or res.artifact_hash != pcb.content_hash:
        return None, "kicad.drc ran on a different board", res
    if res.status is not ValidationStatus.PASS:
        types = sorted({str(v.get("type")) for k in ("errors", "warnings", "unconnected_items", "schematic_parity") for v in res.details.get(k, []) if isinstance(v, dict)})
        return None, f"board has DRC violations (types {types}): geometry not proven against the fab rules", res
    pro = ir.artifacts.get(ArtifactKind.KICAD_PROJECT)
    if pro is None:
        return None, "DRC ran without a .kicad_pro: KiCad's built-in rules prove nothing about the fab minimums", res
    if pro.is_stale(ir.content_hash()):
        return None, "kicad_pro artifact was generated from another IR version; regenerate it and re-run DRC", res
    if not pro.matches_disk():
        return None, "kicad_pro on disk does not match its recorded hash (edited after it was compiled); regenerate it and re-run DRC", res
    recorded = res.details.get("project_hash")
    if recorded is None:
        return None, "DRC ran without the fresh .kicad_pro beside the board" + (f" ({res.details['project_reason']})" if res.details.get("project_reason") else "") + ": KiCad's built-in rules prove nothing about the fab minimums", res
    if recorded != pro.content_hash:
        return None, f"kicad.drc read a different .kicad_pro ({str(recorded)[:19]}…) than the fresh artifact ({pro.content_hash[:19]}…); re-run DRC", res
    ignored = [str(t) for t in res.details.get("ignored_checks", []) or []]
    relevant = sorted(t for t in ignored if t in RELEVANT_DRC_TYPES or "hole" in t or "drill" in t)
    if relevant:
        return None, f"DRC ignored check(s) {relevant}: the fab minimums were not checked", res
    rules = res.details.get("design_rules")
    if not isinstance(rules, dict):
        return None, "DRC ran without a .kicad_pro: KiCad's built-in rules prove nothing about the fab minimums", res
    version = str(res.tool_version or "")
    if version not in PROJECT_RULES_MEASURED_VERSIONS:
        return None, f"project-rule application by kicad-cli {version or '?'} not measured (PROJECT_RULES_MEASURED_VERSIONS)", res
    return {str(k): float(v) for k, v in rules.items()}, None, res


def _geometry_row(name: str, keys: tuple[str, ...], m: ManufacturingConstraints, rules: dict[str, float] | None, reason: str | None) -> dict[str, Any]:
    from ai_eda.compilers.pcb import MANUFACTURING_RULE_KEYS

    present = [(k, getattr(m, k)) for k in keys if getattr(m, k) is not None]
    if not present:
        return _row(name, ValidationStatus.NOT_VERIFIED, f"no limit in ir.pcb.manufacturing for {name} ({', '.join(keys)})", limits=list(keys), grounded=False)
    ungrounded = [k for k, t in present if not _grounded(t)]
    if ungrounded:
        return _row(name, ValidationStatus.NOT_VERIFIED, f"limit(s) {ungrounded} not grounded on the vendor page; DRC evidence can not prove a limit nobody grounded", limits=[k for k, _ in present], grounded=False)
    if rules is None:
        return _row(name, ValidationStatus.NOT_VERIFIED, str(reason), limits=[k for k, _ in present], grounded=True)
    for k, t in present:
        rule_key = MANUFACTURING_RULE_KEYS[k]
        if rule_key not in rules:
            return _row(name, ValidationStatus.NOT_VERIFIED, f"DRC rule {rule_key} not written for {k}", limits=[k for k, _ in present], grounded=True)
        if rules[rule_key] < float(t.value):
            return _row(name, ValidationStatus.NOT_VERIFIED, f"DRC rule {rule_key} {rules[rule_key]:g} < limit {float(t.value):g}", limits=[k for k, _ in present], grounded=True)
    return _row(name, ValidationStatus.PASS, "kicad.drc PASS on the fresh board with the fab minimums as project rules (" + ", ".join(f"{MANUFACTURING_RULE_KEYS[k]}={rules[MANUFACTURING_RULE_KEYS[k]]:g}" for k, _ in present) + ")",
                limits=[k for k, _ in present], grounded=True)


def _hole_to_edge_offenders(ir: CircuitIR, limit: float) -> list[str] | None:
    """IR vias whose hole edge lies closer to the outline than ``limit``; ``None`` when there is no outline to measure from."""
    assert ir.pcb is not None
    outline = ir.pcb.outline
    if outline is None:
        return None
    x0, y0 = outline.origin_x_mm, outline.origin_y_mm
    x1, y1 = x0 + outline.width_mm, y0 + outline.height_mm
    out: list[str] = []
    for i, via in enumerate(ir.pcb.vias):
        r = via.drill_mm / 2.0
        distance = min(via.x_mm - r - x0, x1 - (via.x_mm + r), via.y_mm - r - y0, y1 - (via.y_mm + r))
        if distance < limit:
            out.append(f"via[{i}:{via.net}] hole edge {distance:g} mm from the outline")
    return out


def check_capability(ir: CircuitIR) -> ValidationResult:
    """``mfg.capability`` for ``ir`` (module docstring): deterministic, reads only the IR, its artifacts and the latest ``kicad.drc``."""
    stamp = dict(check_id=CHECK_ID, tool=TOOL_ID, tool_version=TOOL_VERSION)
    if ir.pcb is None:
        return ValidationResult(status=ValidationStatus.NOT_VERIFIED, message="ir.pcb is None: no board to compare against the fab limits",
                                details={"fab": None, "compared": [], "not_compared": dict(NOT_COMPARED), "limits": str(ValidationStatus.NOT_VERIFIED)}, **stamp)
    m = ir.pcb.manufacturing
    rows: list[dict[str, Any]] = []
    # --- IR comparisons -----------------------------------------------------------
    t = m.min_track_width_mm
    if t is None:
        rows.append(_missing_row("min_track_width_mm", "track widths not judged"))
    else:
        rows.append(_compared_row("min_track_width_mm", t, [f"track[{i}:{tr.net}] width {tr.width_mm:g}" for i, tr in enumerate(ir.pcb.tracks) if tr.width_mm < float(t.value)], "track width(s)",
                                  compared=len(ir.pcb.tracks), none="no tracks in the IR"))
    if not ir.pcb.vias:
        rows.append(_row("min_via_drill_mm", ValidationStatus.NOT_APPLICABLE, "no vias in the IR", limit_value=None if m.min_via_drill_mm is None else m.min_via_drill_mm.value))
        rows.append(_row("min_via_diameter_mm", ValidationStatus.NOT_APPLICABLE, "no vias in the IR", limit_value=None if m.min_via_diameter_mm is None else m.min_via_diameter_mm.value))
    else:
        t = m.min_via_drill_mm
        if t is None:
            rows.append(_missing_row("min_via_drill_mm", "via drills not judged"))
        else:
            rows.append(_compared_row("min_via_drill_mm", t, [f"via[{i}:{v.net}] drill {v.drill_mm:g}" for i, v in enumerate(ir.pcb.vias) if v.drill_mm < float(t.value)], "via drill(s)",
                                      compared=len(ir.pcb.vias), none="no vias in the IR"))
        t = m.min_via_diameter_mm
        if t is None:
            rows.append(_missing_row("min_via_diameter_mm", "via diameters not judged"))
        else:
            rows.append(_compared_row("min_via_diameter_mm", t, [f"via[{i}:{v.net}] diameter {v.diameter_mm:g}" for i, v in enumerate(ir.pcb.vias) if v.diameter_mm < float(t.value)], "via diameter(s)",
                                      compared=len(ir.pcb.vias), none="no vias in the IR"))
    t = m.layer_count_options
    if t is not None:
        count = len(ir.pcb.layers)
        rows.append(_compared_row("layer_count_options", t, [] if count in list(t.value) else [f"{count} copper layer(s) not in {list(t.value)}"], "copper layer count",
                                  compared=1, none="", met="in the options", unmet="not in"))
    t = m.min_clearance_mm
    if t is None:
        rows.append(_missing_row("min_clearance_mm", "zone clearances not judged"))
    else:
        with_clearance = [(i, z) for i, z in enumerate(ir.pcb.zones) if z.clearance_mm is not None]
        rows.append(_compared_row("min_clearance_mm", t, [f"zone[{i}:{z.net}] clearance {z.clearance_mm:g}" for i, z in with_clearance if z.clearance_mm < float(t.value)], "zone clearance(s)",
                                  compared=len(with_clearance), none="no zones with a clearance in the IR"))
    t = m.min_hole_to_edge_mm
    if t is not None:
        offenders = _hole_to_edge_offenders(ir, float(t.value))
        if not ir.pcb.vias:
            rows.append(_row("min_hole_to_edge_mm", ValidationStatus.NOT_APPLICABLE, "no vias in the IR", limit_value=t.value, offenders=[], grounded=_grounded(t)))
        elif offenders is None:
            rows.append(_row("min_hole_to_edge_mm", ValidationStatus.NOT_VERIFIED, "ir.pcb.outline is None: hole-to-edge distances can not be measured", limit_value=t.value, grounded=_grounded(t)))
        else:
            rows.append(_compared_row("min_hole_to_edge_mm", t, offenders, "IR via hole-to-edge distance(s)", compared=len(ir.pcb.vias), none="no vias in the IR"))
    # --- geometry only DRC proves ---------------------------------------------------
    rules, reason, drc = _drc_proof(ir)
    for name, keys in GEOMETRY_ROWS.items():
        rows.append(_geometry_row(name, keys, m, rules, reason))
    # --- verdict ----------------------------------------------------------------------
    statuses = [ValidationStatus(r["status"]) for r in rows]
    status = worst_status(statuses)
    failed = [r["limit"] for r in rows if r["status"] == str(ValidationStatus.FAIL)]
    unverified = [f"{r['limit']}: {r['message']}" for r in rows if r["status"] == str(ValidationStatus.NOT_VERIFIED)]
    passed = [r["limit"] for r in rows if r["status"] == str(ValidationStatus.PASS)]
    details: dict[str, Any] = {
        "fab": m.fab, "compared": rows, "not_compared": dict(NOT_COMPARED), "limits": str(FabCapability.from_ir(ir).verification_status()),
        "drc_proof": reason or "kicad.drc PASS with the fab minimums as project rules", "design_rules_drc_ran_with": rules,
    }
    if drc is not None and drc.artifact_hash:
        details["drc_artifact_hash"] = drc.artifact_hash
    if status is ValidationStatus.FAIL:
        details["repair"] = REPAIR
        message = f"board below the fab minimums ({', '.join(failed)}): " + "; ".join(r["message"] for r in rows if r["status"] == str(ValidationStatus.FAIL))
    elif status is ValidationStatus.PASS:
        message = f"board compared against the fab minimums of {m.fab or 'the fab'} ({', '.join(passed)}); not compared: {', '.join(NOT_COMPARED)}"
    else:
        message = f"{len(passed)} limit(s) met in the IR (NOT_APPLICABLE rows not counted), {len(unverified)} not proven: " + "; ".join(unverified[:4]) + (" ..." if len(unverified) > 4 else "") + f"; not compared: {', '.join(NOT_COMPARED)}"
    return ValidationResult(status=status, message=message, details=details, **stamp)


__all__ = [
    "CHECK_ID",
    "GEOMETRY_ROWS",
    "NOT_COMPARED",
    "RELEVANT_DRC_TYPES",
    "REPAIR",
    "TOOL_ID",
    "TOOL_VERSION",
    "FabCapability",
    "check_capability",
    "jlcpcb_capability_unverified",
]
